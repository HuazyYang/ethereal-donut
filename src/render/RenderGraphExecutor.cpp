/*
* Copyright (c) 2026, ethereal contributors. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// The executor (design 9.1) and the registry's runtime half (4): physical resources and their states, the pools,
// fences, command lists and the lowering of a plan.
//
// - A plan is lowered once, when it first executes: its PlanBarriers become nvrhi::ResourceBarrier arrays. Each frame
//   patches only the slots of resources that change between frames: imports, persistent resources, history layers
//   (which rotate, 5.2) and extracted textures.
// - Physical resources belong to a structure instance (all masks of one structural hash) or to a persistent record.
//   The registry tracks their states (PhysicalResource::states). While one plan runs frame after frame its cyclic
//   model holds and nothing is checked; on a plan switch the old plan's states become exact, and the new plan's
//   frame-start states are reached with patch barriers in a prologue, after a wait for the previous frame (6.12).
// - Steady state: no allocation and no create* call. Lists, contexts, fences and resources are made once.

#include "RenderGraphCompile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>

namespace donut::render::rdg
{
    using nvrhi::CommandQueue;
    using nvrhi::ResourceStates;

    // ------------------------------------------------------------------------------------------------ plan objects

    static std::atomic<uint64_t> s_PlanSerial{ 0 };

    struct TrackerOp
    {
        uint32_t resource;
        nvrhi::SubresourceRegion region;
        ResourceStates before;
        ResourceStates after;
    };

    struct LoweredBatch
    {
        uint32_t explicitBegin;
        uint32_t explicitCount;
        uint32_t trackerBegin;
        uint32_t trackerCount;
    };

    // A pointer of a lowered barrier that names a graph resource: field 0 is the transition's or UAV barrier's resource
    // or the aliasing barrier's `after`, field 1 the aliasing barrier's `before`.
    struct PatchSlot
    {
        uint32_t barrier;
        uint32_t resource;
        uint8_t field;
    };

    // A raster pass's framebuffer: colour slots 0..7, the depth target (c_DepthAttachment, 8), and the shading-rate
    // surface its ShadingRateRead names (9).
    constexpr uint32_t c_FramebufferSlots = 10;
    constexpr uint32_t c_ShadingRateSlot = 9;

    struct FramebufferEntry
    {
        nvrhi::ITexture* attachments[c_FramebufferSlots] = {};
        nvrhi::AutoPtr<nvrhi::IFramebuffer> framebuffer;
        uint64_t lastUsed = 0;
    };

    class PassContext;

    struct PlanInstance
    {
        StructureInstance* structure = nullptr;
        std::vector<nvrhi::ResourceBarrier> lowered;
        std::vector<TrackerOp> trackerOps;
        std::vector<LoweredBatch> batches;           // [schedule index * 3 + slot]
        std::vector<PatchSlot> fixedSlots;           // transients: resolved once
        std::vector<PatchSlot> frameSlots;           // patched every frame
        bool fixedResolved = false;
        std::vector<std::array<FramebufferEntry, 4>> framebuffers;
        std::vector<nvrhi::FramebufferInfoEx> framebufferInfos;
        std::vector<uint64_t> segmentValue;          // this frame's fence value of each segment
        std::vector<uint64_t> previousSegmentValue;
        std::vector<uint32_t> segmentOrdinal;        // per segment: its index among its stream's segments

        // Recording (10.1), sized once per plan instance: each segment's list, context and result, each pass's timer
        // slot, and the tasks of a parallel recording (their segments, in plan order).
        std::vector<nvrhi::ICommandList*> segmentLists;
        std::vector<PassContext*> segmentContexts;
        std::vector<nvrhi::FRESULT> segmentResults;
        std::vector<uint32_t> timerSlots;
        std::vector<uint64_t> segmentCost;           // each segment's last recording, in ns: the next frame's weights
        std::vector<uint32_t> segmentGroup;
        std::vector<uint64_t> groupWeight;
        std::vector<uint32_t> groupOrder;
        std::vector<uint32_t> groupTask;
        std::vector<uint64_t> taskLoad;
        std::vector<uint32_t> taskBegin;
        std::vector<uint32_t> taskSegments;
        std::vector<std::pair<const void*, uint32_t>> passOwners;
    };

    CompiledRenderGraph::CompiledRenderGraph()
        : m_Serial(++s_PlanSerial)
    {
    }

    CompiledRenderGraph::~CompiledRenderGraph() = default;

    // ------------------------------------------------------------------------------------------------ descriptions

    static std::unique_ptr<PhysicalResource> CreatePhysical(nvrhi::IDevice* device, const ResourceRecord& record,
                                                           const BackendProfile& profile, RegistryRuntime& runtime)
    {
        auto physical = std::make_unique<PhysicalResource>();
        physical->isBuffer = record.isBuffer;
        physical->descHash = record.descHash;
        if (record.isBuffer)
        {
            if (device->createBuffer(MakeBufferDesc(record), physical->buffer.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                return nullptr;
            physical->resource = physical->buffer.Get();
        }
        else
        {
            if (device->createTexture(MakeTextureDesc(record), physical->texture.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                return nullptr;
            physical->resource = physical->texture.Get();
        }
        ++runtime.objectsCreated;
        physical->planes = UnitPlaneCount(record, profile);
        physical->mips = MipCount(record);
        physical->slices = SliceCount(record);
        physical->states.assign(size_t(physical->planes) * physical->mips * physical->slices, ResourceStates::Unknown);
        physical->fresh = true;
        return physical;
    }

    // A physical resource for `record`: a free one of the same description, or a new one.
    static std::unique_ptr<PhysicalResource> AcquirePhysical(std::vector<std::unique_ptr<PhysicalResource>>& pool,
                                                            nvrhi::IDevice* device, const ResourceRecord& record,
                                                            const BackendProfile& profile, RegistryRuntime& runtime)
    {
        const uint32_t planes = UnitPlaneCount(record, profile);
        for (size_t i = 0; i < pool.size(); ++i)
        {
            PhysicalResource& candidate = *pool[i];
            if (candidate.isBuffer == record.isBuffer && candidate.descHash == record.descHash && candidate.planes == planes)
            {
                std::unique_ptr<PhysicalResource> taken = std::move(pool[i]);
                pool.erase(pool.begin() + ptrdiff_t(i));
                taken->stateOwner = 0;      // its states are exact
                return taken;
            }
        }
        return CreatePhysical(device, record, profile, runtime);
    }

    // ------------------------------------------------------------------------------------------------ contexts

    struct ExecutorFrame;

    class PrepareContext final : public nvrhi::ObjectImpl<IPrepareContext>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(PrepareContext)
        NVRHI_IMPLEMENTS_INTERFACE(IPrepareContext)
        NVRHI_END_INTERFACE_TABLE()

        nvrhi::IDevice* GetDevice() noexcept override;
        nvrhi::ICommandList* GetUploadCommandList() noexcept override;
        const nvrhi::FramebufferInfoEx& GetFramebufferInfo() noexcept override;
        const nvrhi::TextureDesc* GetTextureDesc(ResourceId id) noexcept override;
        const nvrhi::BufferDesc* GetBufferDesc(ResourceId id) noexcept override;

        ExecutorFrame* frame = nullptr;
        uint32_t sched = 0;
    };

    class PassContext final : public nvrhi::ObjectImpl<IPassContext>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(PassContext)
        NVRHI_IMPLEMENTS_INTERFACE(IPassContext)
        NVRHI_END_INTERFACE_TABLE()

        nvrhi::ICommandList* GetCommandList() noexcept override { return commandList; }
        nvrhi::IResource* GetResource(ResourceId id) noexcept override;
        nvrhi::ITexture* GetTexture(ResourceId id) noexcept override;
        nvrhi::IBuffer* GetBuffer(ResourceId id) noexcept override;
        nvrhi::IFramebuffer* GetFramebuffer() noexcept override;
        bool IsNewAllocation(ResourceId id) noexcept override;
        uint64_t GetFrameIndex() noexcept override;

        ExecutorFrame* frame = nullptr;
        uint32_t sched = 0;
        nvrhi::ICommandList* commandList = nullptr;
        std::vector<nvrhi::IResource*> trackedInList;   // the tracker fallback (9.1): resources this list tracks
    };

    // The state of one Execute call.
    struct ExecutorFrame
    {
        RenderGraph& graph;
        CompiledRenderGraph& plan;
        PlanInstance& instance;
        ResourceRegistry& registry;
        RegistryRuntime& runtime;
        nvrhi::IDevice* device;
        uint64_t frameIndex;
        nvrhi::ICommandList* prologue = nullptr;
        bool prologueOpen = false;
        RegistryRuntime::TimerSet* timers = nullptr;   // ExecuteDesc::timers (11.3)
        // Updated by every recording thread (10.1).
        std::atomic<uint32_t> barriers{ 0 };
        std::atomic<uint32_t> stateCalls{ 0 };
        std::atomic<uint32_t> passes{ 0 };
        std::atomic<uint32_t> framebuffersCreated{ 0 };
        uint32_t recordTasks = 1;

        // The prologue opens on demand, so that a frame without patches or uploads neither records nor submits it:
        // the patch barriers first, then Prepare's uploads with automatic barriers, as for any upload outside a graph.
        nvrhi::ICommandList* OpenPrologue()
        {
            if (!prologueOpen)
            {
                prologueOpen = true;
                prologue->open();
                if (!runtime.patches.empty())
                    RecordPatches();
            }
            return prologue;
        }

        void RecordPatches()
        {
            if (plan.m_Profile.explicitBarriers)
            {
                prologue->setEnableAutomaticBarriers(false);
                prologue->barriers(runtime.patches.data(), uint32_t(runtime.patches.size()));
                prologue->setEnableAutomaticBarriers(true);
                return;
            }
            // A device without ICommandList::barriers: the same transitions through the tracker (9.1). The tracker
            // cannot start from Unknown; a fresh resource is in its creation state, Common on every backend.
            for (size_t i = 0; i < runtime.patches.size(); ++i)
            {
                nvrhi::TransitionBarrier patch = runtime.patches[i].transition;
                if (patch.before == ResourceStates::Unknown)
                    patch.before = ResourceStates::Common;
                const PhysicalResource* owner = runtime.patchOwners[i];
                if (owner->texture)
                {
                    const nvrhi::TextureSubresourceSet set(patch.region.baseMip, patch.region.mipCount, patch.region.baseSlice,
                                                           patch.region.sliceCount);
                    prologue->beginTrackingTextureState(owner->texture, set, patch.before);
                    prologue->setTextureState(owner->texture, set, patch.after);
                }
                else if (owner->buffer)
                {
                    prologue->beginTrackingBufferState(owner->buffer, patch.before);
                    prologue->setBufferState(owner->buffer, patch.after);
                }
            }
            prologue->commitBarriers();
        }

        void ReportError(const std::string& text) const
        {
            if (nvrhi::IMessageCallback* callback = device->getMessageCallback())
                callback->message(nvrhi::MessageSeverity::Error, ("render graph: " + text).c_str());
        }

        const FrameResource* Resolve(ResourceId id) const
        {
            const uint32_t index = GetResourceIndex(id);
            if (id == InvalidResourceId || index >= runtime.frameResources.size())
                return nullptr;
            return &runtime.frameResources[index];
        }

        bool BuildFramebufferDesc(uint32_t sched, nvrhi::FramebufferDesc& desc, nvrhi::ITexture* (&attachments)[c_FramebufferSlots]) const;
    };

    static nvrhi::TextureSubresourceSet ToSubresourceSet(const nvrhi::SubresourceRegion& region)
    {
        return nvrhi::TextureSubresourceSet(region.baseMip, region.mipCount, region.baseSlice, region.sliceCount);
    }

    bool ExecutorFrame::BuildFramebufferDesc(uint32_t sched, nvrhi::FramebufferDesc& desc, nvrhi::ITexture* (&attachments)[c_FramebufferSlots]) const
    {
        const PlanPass& pass = plan.m_Passes[sched];
        nvrhi::FramebufferAttachment slots[9];
        for (nvrhi::ITexture*& attachment : attachments)
            attachment = nullptr;
        for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount; ++u)
        {
            const PlanUse& use = plan.m_Uses[u];
            if ((use.access & Access::ShadingRateRead) != Access::None && runtime.frameResources[use.resource].texture)
            {
                attachments[c_ShadingRateSlot] = runtime.frameResources[use.resource].texture;
                continue;
            }
            if (use.attachment == c_NoAttachment)
                continue;
            nvrhi::ITexture* texture = runtime.frameResources[use.resource].texture;
            attachments[use.attachment] = texture;
            slots[use.attachment] = nvrhi::FramebufferAttachment()
                .setTexture(texture)
                .setSubresources(nvrhi::TextureSubresourceSet(use.region.baseMip, 1, use.region.baseSlice, use.region.sliceCount))
                .setReadOnly(use.attachment == c_DepthAttachment && !use.isWrite);
        }
        for (uint32_t slot = 0; slot < 8; ++slot)
        {
            if (!attachments[slot])
            {
                for (uint32_t later = slot + 1; later < 8; ++later)
                {
                    if (attachments[later])
                        return false;   // colour targets must be contiguous from slot 0
                }
                break;
            }
            desc.addColorAttachment(slots[slot]);
        }
        if (attachments[c_DepthAttachment])
            desc.setDepthAttachment(slots[c_DepthAttachment]);
        if (attachments[c_ShadingRateSlot])
            desc.setShadingRateAttachment(nvrhi::FramebufferAttachment().setTexture(attachments[c_ShadingRateSlot]));
        return true;
    }

    nvrhi::IDevice* PrepareContext::GetDevice() noexcept { return frame->device; }

    nvrhi::ICommandList* PrepareContext::GetUploadCommandList() noexcept
    {
        return frame->OpenPrologue();
    }

    const nvrhi::FramebufferInfoEx& PrepareContext::GetFramebufferInfo() noexcept
    {
        nvrhi::FramebufferInfoEx& info = frame->instance.framebufferInfos[sched];
        nvrhi::FramebufferDesc desc;
        nvrhi::ITexture* attachments[c_FramebufferSlots];
        info = frame->BuildFramebufferDesc(sched, desc, attachments) ? nvrhi::FramebufferInfoEx(desc) : nvrhi::FramebufferInfoEx();
        return info;
    }

    const nvrhi::TextureDesc* PrepareContext::GetTextureDesc(ResourceId id) noexcept
    {
        const FrameResource* resource = frame->Resolve(id);
        return resource && resource->texture ? &resource->texture->getDesc() : nullptr;
    }

    const nvrhi::BufferDesc* PrepareContext::GetBufferDesc(ResourceId id) noexcept
    {
        const FrameResource* resource = frame->Resolve(id);
        return resource && resource->buffer ? &resource->buffer->getDesc() : nullptr;
    }

    nvrhi::IResource* PassContext::GetResource(ResourceId id) noexcept
    {
        const FrameResource* resource = frame->Resolve(id);
        return resource ? resource->resource : nullptr;
    }

    nvrhi::ITexture* PassContext::GetTexture(ResourceId id) noexcept
    {
        const FrameResource* resource = frame->Resolve(id);
        return resource ? resource->texture : nullptr;
    }

    nvrhi::IBuffer* PassContext::GetBuffer(ResourceId id) noexcept
    {
        const FrameResource* resource = frame->Resolve(id);
        return resource ? resource->buffer : nullptr;
    }

    nvrhi::IFramebuffer* PassContext::GetFramebuffer() noexcept
    {
        nvrhi::FramebufferDesc desc;
        nvrhi::ITexture* attachments[c_FramebufferSlots];
        if (!frame->BuildFramebufferDesc(sched, desc, attachments))
        {
            frame->ReportError("the colour targets of a pass are not contiguous from slot 0");
            return nullptr;
        }
        // Cached per attachment set, so that a back buffer rotating through its images creates no framebuffer in
        // steady state.
        std::array<FramebufferEntry, 4>& cache = frame->instance.framebuffers[sched];
        FramebufferEntry* oldest = &cache[0];
        for (FramebufferEntry& entry : cache)
        {
            if (entry.framebuffer && std::equal(std::begin(entry.attachments), std::end(entry.attachments), std::begin(attachments)))
            {
                entry.lastUsed = frame->frameIndex;
                return entry.framebuffer;
            }
            if (!entry.framebuffer || entry.lastUsed < oldest->lastUsed)
                oldest = &entry;
        }
        oldest->framebuffer = nullptr;
        if (frame->device->createFramebuffer(desc, oldest->framebuffer.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
            return nullptr;
        frame->framebuffersCreated.fetch_add(1, std::memory_order_relaxed);
        std::copy(std::begin(attachments), std::end(attachments), std::begin(oldest->attachments));
        oldest->lastUsed = frame->frameIndex;
        return oldest->framebuffer;
    }

    bool PassContext::IsNewAllocation(ResourceId id) noexcept
    {
        const uint32_t index = GetResourceIndex(id);
        if (id == InvalidResourceId || index >= frame->graph.ResourceCount())
            return false;
        const ResourceRecord& record = frame->graph.GetResource(index);
        switch (record.kind)
        {
        case ResourceKind::Transient:
        case ResourceKind::PassLocal:
            return true;
        case ResourceKind::Persistent:
        case ResourceKind::History:
        {
            // 5.2: layer k is new while it has not been written since the history was (re)allocated, that is, for
            // the first k frames; a persistent resource on its first frame.
            const PersistentEntry& entry = frame->registry.GetPersistent(record.registryEntry);
            const uint64_t framesSince = frame->frameIndex - entry.allocatedFrame;
            if (record.kind == ResourceKind::Persistent)
                return framesSince == 0;
            return framesSince < record.historyLayer;
        }
        default:
            return false;
        }
    }

    uint64_t PassContext::GetFrameIndex() noexcept { return frame->frameIndex; }

    // ------------------------------------------------------------------------------------------------ instances

    // 7.5, the Placed tier. The compiler placed estimated sizes; here the virtual resources report their actual memory
    // requirements (the probe), and a first-fit placement per planned heap uses those. Two objects share bytes only
    // where the plan placed them sharing bytes, so that its aliasing barriers cover every actual overlap. Then the
    // heaps are created and the resources bound. On D3D12 resource heap tier 1 only render-target and depth-stencil
    // textures are placed; the other classes stay pooled. Returns false when a heap or a binding fails.
    static bool PlaceStructure(StructureInstance& instance, const CompiledRenderGraph& plan, const RenderGraph& graph,
                               RegistryRuntime& runtime, nvrhi::IDevice* device)
    {
        const StructurePlan& structure = *plan.m_Structure;
        const size_t count = structure.physicals.size();

        std::vector<uint32_t> first(count, ~0u);
        std::vector<uint32_t> last(count, 0);
        for (size_t r = 0; r < structure.resourcePhysical.size(); ++r)
        {
            const uint32_t p = structure.resourcePhysical[r];
            if (p == c_None || p >= count || structure.resourceFirst[r] == c_None)
                continue;
            first[p] = std::min(first[p], structure.resourceFirst[r]);
            last[p] = std::max(last[p], structure.resourceLast[r]);
        }

        struct Candidate
        {
            uint32_t physical;
            uint64_t size;
            uint64_t alignment;
            uint64_t offset;
        };
        std::vector<Candidate> candidates;
        for (uint32_t p = 0; p < count; ++p)
        {
            const PhysicalObject& object = structure.physicals[p];
            if (object.kind != PhysicalKind::Placed || object.heap >= structure.heaps.size() || first[p] == ~0u)
                continue;
            const ResourceRecord& record = graph.GetResource(object.firstResource);
            if (plan.m_Profile.resourceHeapTier < 2 && (record.isBuffer || !(record.texture.isRenderTarget || record.hasDepth)))
                continue;

            auto physical = std::make_unique<PhysicalResource>();
            physical->isBuffer = record.isBuffer;
            physical->descHash = record.descHash;
            nvrhi::MemoryRequirements requirements;
            if (record.isBuffer)
            {
                nvrhi::BufferDesc desc = MakeBufferDesc(record);
                desc.isVirtual = true;
                if (device->createBuffer(desc, physical->buffer.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                    return false;
                physical->resource = physical->buffer.Get();
                device->getBufferMemoryRequirements(requirements, physical->buffer);
            }
            else
            {
                nvrhi::TextureDesc desc = MakeTextureDesc(record);
                desc.isVirtual = true;
                if (device->createTexture(desc, physical->texture.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                    return false;
                physical->resource = physical->texture.Get();
                device->getTextureMemoryRequirements(requirements, physical->texture);
            }
            ++runtime.objectsCreated;
            if (requirements.size == 0)
                return false;
            physical->planes = UnitPlaneCount(record, plan.m_Profile);
            physical->mips = MipCount(record);
            physical->slices = SliceCount(record);
            physical->states.assign(size_t(physical->planes) * physical->mips * physical->slices, ResourceStates::Unknown);
            physical->fresh = true;
            physical->placed = true;
            physical->heap = object.heap;
            physical->heapSize = requirements.size;
            instance.placedObjectBytes += requirements.size;
            instance.physicals[p] = std::move(physical);
            candidates.push_back(Candidate{ p, requirements.size, std::max<uint64_t>(requirements.alignment, 1), 0 });
        }

        // The plan's byte sharing: the only overlaps the placement may produce.
        auto plannedOverlap = [&](uint32_t a, uint32_t b) {
            const PhysicalObject& x = structure.physicals[a];
            const PhysicalObject& y = structure.physicals[b];
            return x.heap == y.heap && x.offset < y.offset + y.size && y.offset < x.offset + x.size;
        };
        std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.size > b.size;
        });
        std::vector<uint64_t> heapEnd(structure.heaps.size(), 0);
        std::vector<uint64_t> offsets;
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            Candidate& c = candidates[i];
            const uint32_t heap = structure.physicals[c.physical].heap;
            offsets.assign(1, 0);
            for (size_t j = 0; j < i; ++j)
            {
                if (structure.physicals[candidates[j].physical].heap == heap)
                    offsets.push_back(candidates[j].offset + candidates[j].size);
            }
            std::sort(offsets.begin(), offsets.end());
            for (uint64_t start : offsets)
            {
                const uint64_t offset = (start + c.alignment - 1) / c.alignment * c.alignment;
                bool fits = true;
                for (size_t j = 0; j < i && fits; ++j)
                {
                    const Candidate& q = candidates[j];
                    if (structure.physicals[q.physical].heap != heap || offset >= q.offset + q.size || q.offset >= offset + c.size)
                        continue;
                    const bool disjoint = last[c.physical] < first[q.physical] || last[q.physical] < first[c.physical];
                    fits = disjoint && plannedOverlap(c.physical, q.physical);
                }
                if (fits)
                {
                    c.offset = offset;
                    break;
                }
            }
            heapEnd[heap] = std::max(heapEnd[heap], c.offset + c.size);
        }

        instance.heaps.resize(structure.heaps.size());
        for (size_t h = 0; h < structure.heaps.size(); ++h)
        {
            if (heapEnd[h] == 0)
                continue;
            if (device->createHeap(nvrhi::HeapDesc().setCapacity(heapEnd[h]).setType(nvrhi::HeapType::DeviceLocal)
                    .setDebugName("render graph heap"), instance.heaps[h].ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                return false;
            ++runtime.objectsCreated;
            instance.heapBytes += heapEnd[h];
        }
        for (const Candidate& c : candidates)
        {
            PhysicalResource& physical = *instance.physicals[c.physical];
            nvrhi::IHeap* heap = instance.heaps[physical.heap];
            physical.heapOffset = c.offset;
            const bool bound = physical.isBuffer ? device->bindBufferMemory(physical.buffer, heap, c.offset)
                                                 : device->bindTextureMemory(physical.texture, heap, c.offset);
            if (!bound)
                return false;
        }
        return true;
    }

    static bool ActuallyOverlap(const PhysicalResource& a, const PhysicalResource& b)
    {
        return a.placed && b.placed && &a != &b && a.heap == b.heap &&
            a.heapOffset < b.heapOffset + b.heapSize && b.heapOffset < a.heapOffset + a.heapSize;
    }

    // The plan's aliasing barriers name the outgoing resource the plan overlapped with the incoming one. After the
    // actual placement the incoming one may be unplaced: the barrier has no `after` then, and RecordBatch skips it.
    // Or the two may not overlap: the barrier names no `before`, which orders any aliasing in its heap (nvrhi rejects
    // a named pair that does not overlap).
    //
    // Then the check of the actual placement (6.7): every resource of this plan whose bytes another placed resource of
    // the instance shares starts with an aliasing barrier, across masks and the frame boundary alike. D3D12's debug
    // layer and GBV do not track aliasing, so this is what detects a missing barrier on the GPU (11.4,
    // SkipAliasingBarriers). Returns the number of uncovered resources, each reported as an error.
    static uint32_t MatchAliasingBarriers(PlanInstance& instance, const StructureInstance& structure,
                                          const CompiledRenderGraph& plan, nvrhi::IDevice* device)
    {
        auto find = [&](nvrhi::IResource* resource) -> const PhysicalResource* {
            for (const std::unique_ptr<PhysicalResource>& physical : structure.physicals)
            {
                if (physical && physical->resource == resource)
                    return physical.get();
            }
            return nullptr;
        };
        for (nvrhi::ResourceBarrier& barrier : instance.lowered)
        {
            if (barrier.type != nvrhi::BarrierType::Aliasing)
                continue;
            const PhysicalResource* after = barrier.aliasing.after ? find(barrier.aliasing.after) : nullptr;
            const PhysicalResource* before = barrier.aliasing.before ? find(barrier.aliasing.before) : nullptr;
            if (!after || !after->placed)
            {
                barrier.aliasing.before = nullptr;
                barrier.aliasing.after = nullptr;
                continue;
            }
            if (before && !ActuallyOverlap(*before, *after))
                barrier.aliasing.before = nullptr;
        }

        uint32_t uncovered = 0;
        for (uint32_t r = 0; r < uint32_t(plan.m_Resources.size()); ++r)
        {
            const PlanResource& resource = plan.m_Resources[r];
            if (resource.firstPass == c_None || resource.physical == c_None || resource.physical >= structure.physicals.size())
                continue;
            const PhysicalResource* physical = structure.physicals[resource.physical].get();
            if (!physical || !physical->placed)
                continue;
            bool shared = false;
            for (const std::unique_ptr<PhysicalResource>& other : structure.physicals)
                shared = shared || (other && ActuallyOverlap(*physical, *other));
            if (!shared)
                continue;
            const LoweredBatch& batch = instance.batches[size_t(resource.firstPass) * c_SlotCount + uint32_t(BatchSlot::Prologue)];
            bool covered = false;
            for (uint32_t b = batch.explicitBegin; b < batch.explicitBegin + batch.explicitCount && !covered; ++b)
            {
                const nvrhi::ResourceBarrier& barrier = instance.lowered[b];
                covered = barrier.type == nvrhi::BarrierType::Aliasing && barrier.aliasing.after == physical->resource;
            }
            if (covered)
                continue;
            ++uncovered;
            if (nvrhi::IMessageCallback* callback = device->getMessageCallback())
            {
                const std::string text = std::string("render graph: placed resource '") + plan.GetName(resource.nameOffset) +
                    "' shares heap memory with another resource but has no aliasing barrier at its first use";
                callback->message(nvrhi::MessageSeverity::Error, text.c_str());
            }
        }
        return uncovered;
    }

    static StructureInstance* AcquireStructureInstance(const CompiledRenderGraph& plan, const RenderGraph& graph,
                                                       RegistryRuntime& runtime, nvrhi::IDevice* device)
    {
        for (const std::unique_ptr<StructureInstance>& instance : runtime.instances)
        {
            if (instance->structure.get() == plan.m_Structure.get())
                return instance.get();
        }

        auto instance = std::make_unique<StructureInstance>();
        instance->structure = plan.m_Structure;
        const StructurePlan& structure = *plan.m_Structure;
        instance->physicals.resize(structure.physicals.size());
        if (structure.tier == MemoryTier::Placed && !PlaceStructure(*instance, plan, graph, runtime, device))
        {
            // 7.5: no heaps after all; every object is dedicated, and the aliasing barriers lose their resources.
            for (std::unique_ptr<PhysicalResource>& physical : instance->physicals)
                physical.reset();
            instance->heaps.clear();
            instance->heapBytes = 0;
            instance->placedObjectBytes = 0;
            if (nvrhi::IMessageCallback* callback = device->getMessageCallback())
                callback->message(nvrhi::MessageSeverity::Warning,
                    "render graph: the Placed tier failed to place a plan's resources; they are dedicated instead");
        }
        for (size_t p = 0; p < structure.physicals.size(); ++p)
        {
            if (instance->physicals[p])
                continue;   // placed
            const ResourceRecord& record = graph.GetResource(structure.physicals[p].firstResource);
            // Persistent resources and history layers live in the registry's records; extracted textures in a pool.
            if (record.kind == ResourceKind::Persistent || record.kind == ResourceKind::History || record.extracted)
                continue;
            instance->physicals[p] = AcquirePhysical(runtime.freePool, device, record, plan.m_Profile, runtime);
            if (!instance->physicals[p])
                return nullptr;
        }
        runtime.instances.push_back(std::move(instance));
        return runtime.instances.back().get();
    }

    // Structure instances that no plan uses any more give their memory to the pool; long-unused pool entries go.
    static void CollectInstances(RegistryRuntime& runtime)
    {
        // 7.5: retired heaps and placed resources go when the GPU has passed the frame they were retired after.
        for (size_t i = 0; i < runtime.retired.size();)
        {
            bool done = true;
            for (uint32_t q = 0; q < c_MaxStreams && done; ++q)
            {
                const uint64_t value = runtime.retired[i].values[q];
                done = !value || !runtime.fences[q] || runtime.fences[q]->getCompletedValue() >= value;
            }
            if (done)
                runtime.retired.erase(runtime.retired.begin() + ptrdiff_t(i));
            else
                ++i;
        }

        for (size_t i = 0; i < runtime.instances.size();)
        {
            StructureInstance& instance = *runtime.instances[i];
            if (instance.structure.use_count() > 1)
            {
                ++i;
                continue;
            }
            bool placed = !instance.heaps.empty();
            for (std::unique_ptr<PhysicalResource>& physical : instance.physicals)
            {
                if (physical && physical->placed)
                    placed = true;
                else if (physical)
                    runtime.freePool.push_back(std::move(physical));
            }
            if (placed)
                runtime.retired.push_back(RegistryRuntime::RetiredInstance{ std::move(runtime.instances[i]), runtime.lastFrameValues });
            runtime.instances.erase(runtime.instances.begin() + ptrdiff_t(i));
        }
        constexpr uint64_t c_PoolAge = 120;
        for (size_t i = 0; i < runtime.freePool.size();)
        {
            if (runtime.freePool[i]->lastUsedFrame + c_PoolAge < runtime.framesExecuted)
                runtime.freePool.erase(runtime.freePool.begin() + ptrdiff_t(i));
            else
                ++i;
        }
    }

    static PhysicalResource* PersistentPhysical(ResourceRegistry& registry, const ResourceRecord& record, uint32_t index,
                                                nvrhi::IDevice* device, const BackendProfile& profile)
    {
        PersistentEntry& entry = registry.GetPersistentMutable(record.registryEntry);
        const uint32_t count = record.kind == ResourceKind::History ? std::max<uint32_t>(record.historyLayerCount, 1) : 1;
        if (entry.physicals.size() != count || entry.physicalGeneration != entry.generation)
        {
            // A new generation (a new description or layer count, 5.2): new textures. In-flight frames keep the old
            // ones alive through their command lists.
            entry.physicals.clear();
            for (uint32_t i = 0; i < count; ++i)
            {
                entry.physicals.push_back(CreatePhysical(device, record, profile, registry.m_Runtime));
                if (!entry.physicals.back())
                    return nullptr;
            }
            entry.physicalGeneration = entry.generation;
        }
        return entry.physicals[index].get();
    }

    static uint32_t HistoryPhysicalIndex(uint64_t frameIndex, uint32_t layer, uint32_t layerCount)
    {
        // 5.2: layer k of frame f is physical texture (f - k) mod L.
        return uint32_t((frameIndex + uint64_t(layerCount) * 64 - layer) % layerCount);
    }

    static PhysicalResource* AcquireExtracted(RegistryRuntime& runtime, const ResourceRecord& record, nvrhi::IDevice* device,
                                              const BackendProfile& profile)
    {
        for (const std::unique_ptr<PhysicalResource>& candidate : runtime.extractPool)
        {
            if (candidate->isBuffer || candidate->descHash != record.descHash || !candidate->texture)
                continue;
            // Free when the pool holds the only reference: the caller released it and no command list holds it.
            candidate->texture->AddRef();
            if (candidate->texture->Release() == 1)
                return candidate.get();
        }
        std::unique_ptr<PhysicalResource> created = CreatePhysical(device, record, profile, runtime);
        if (!created)
            return nullptr;
        runtime.extractPool.push_back(std::move(created));
        return runtime.extractPool.back().get();
    }

    // Lowers the plan's barriers once (10.1): explicit barriers, or tracker operations for host-kept resources and for
    // a device without explicit barriers (9.1, item 6).
    static void LowerPlan(CompiledRenderGraph& plan, PlanInstance& instance)
    {
        const BackendProfile& profile = plan.m_Profile;
        const uint32_t schedCount = uint32_t(plan.m_Passes.size());
        instance.batches.assign(size_t(schedCount) * c_SlotCount, LoweredBatch{});
        for (uint32_t s = 0; s < schedCount; ++s)
        {
            const PlanPass& pass = plan.m_Passes[s];
            for (uint32_t slot = 0; slot < c_SlotCount; ++slot)
            {
                LoweredBatch& batch = instance.batches[size_t(s) * c_SlotCount + slot];
                batch.explicitBegin = uint32_t(instance.lowered.size());
                batch.trackerBegin = uint32_t(instance.trackerOps.size());
                for (uint32_t b = pass.batchBegin[slot]; b < pass.batchBegin[slot] + pass.batchCount[slot]; ++b)
                {
                    const PlanBarrier& barrier = plan.m_Barriers[b];
                    const bool hostKept = barrier.resource != c_None && plan.m_Resources[barrier.resource].kind == ResourceKind::HostKept;
                    if (barrier.type == nvrhi::BarrierType::Transition && (hostKept || !profile.explicitBarriers))
                    {
                        // Host-kept resources go through the tracker, so that nvrhi's restore at close() sees every
                        // change (5.2). Splits and ownership halves have no tracker form: the end half or the acquire
                        // half carries the change.
                        if ((barrier.flags & (nvrhi::BarrierFlags::SplitBegin | nvrhi::BarrierFlags::QueueRelease)) != 0)
                            continue;
                        instance.trackerOps.push_back(TrackerOp{ barrier.resource, barrier.region, barrier.before, barrier.after });
                        continue;
                    }
                    if (!profile.explicitBarriers)
                    {
                        if (barrier.type == nvrhi::BarrierType::Uav && barrier.resource != c_None)
                            instance.trackerOps.push_back(TrackerOp{ barrier.resource, nvrhi::SubresourceRegion(),
                                ResourceStates::UnorderedAccess, ResourceStates::UnorderedAccess });
                        continue;   // aliasing needs placed memory, which this path does not have
                    }

                    nvrhi::ResourceBarrier lowered;
                    const uint32_t index = uint32_t(instance.lowered.size());
                    switch (barrier.type)
                    {
                    case nvrhi::BarrierType::Transition:
                        lowered = nvrhi::ResourceBarrier::Transition(nullptr, barrier.before, barrier.after, barrier.region, barrier.flags);
                        lowered.transition.stagesBefore = barrier.stagesBefore;
                        lowered.transition.stagesAfter = barrier.stagesAfter;
                        lowered.transition.otherQueue = barrier.otherQueue;
                        break;
                    case nvrhi::BarrierType::Uav:
                        lowered = nvrhi::ResourceBarrier::Uav(nullptr);
                        break;
                    default:
                        lowered = nvrhi::ResourceBarrier::Aliasing(nullptr, nullptr);
                        if (barrier.aliasBefore != c_None)
                            instance.fixedSlots.push_back(PatchSlot{ index, barrier.aliasBefore, 1 });
                        break;
                    }
                    if (barrier.resource != c_None)
                    {
                        const PlanResource& resource = plan.m_Resources[barrier.resource];
                        const bool fixed = (resource.kind == ResourceKind::Transient || resource.kind == ResourceKind::PassLocal) &&
                            !resource.extracted;
                        (fixed ? instance.fixedSlots : instance.frameSlots).push_back(PatchSlot{ index, barrier.resource, 0 });
                    }
                    instance.lowered.push_back(lowered);
                }
                batch.explicitCount = uint32_t(instance.lowered.size()) - batch.explicitBegin;
                batch.trackerCount = uint32_t(instance.trackerOps.size()) - batch.trackerBegin;
            }
        }

        instance.framebuffers.resize(schedCount);
        instance.framebufferInfos.resize(schedCount);
        instance.segmentValue.assign(plan.m_Segments.size(), 0);
        instance.previousSegmentValue.assign(plan.m_Segments.size(), 0);
        instance.segmentOrdinal.resize(plan.m_Segments.size());
        uint32_t perStream[c_MaxStreams] = {};
        for (size_t i = 0; i < plan.m_Segments.size(); ++i)
            instance.segmentOrdinal[i] = perStream[StreamOf(plan.m_Segments[i].queue, plan.m_Segments[i].lane)]++;
    }

    static void PatchSlots(const std::vector<PatchSlot>& slots, PlanInstance& instance, const RegistryRuntime& runtime)
    {
        for (const PatchSlot& slot : slots)
        {
            nvrhi::ResourceBarrier& barrier = instance.lowered[slot.barrier];
            nvrhi::IResource* resource = runtime.frameResources[slot.resource].resource;
            switch (barrier.type)
            {
            case nvrhi::BarrierType::Transition: barrier.transition.resource = resource; break;
            case nvrhi::BarrierType::Uav:        barrier.uav.resource = resource; break;
            default:
                if (slot.field == 1)
                    barrier.aliasing.before = resource;
                else
                    barrier.aliasing.after = resource;
                break;
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ states

    static void ApplyRegion(PhysicalResource& physical, const nvrhi::SubresourceRegion& region, ResourceStates state,
                            std::vector<ResourceStates>& states)
    {
        uint32_t planeBits = 0;
        if (physical.planes == 1)
            planeBits = region.planes != nvrhi::PlaneMask::None ? 1u : 0u;
        else
            planeBits = ((region.planes & nvrhi::PlaneMask::Depth) != 0 ? 1u : 0u) | ((region.planes & nvrhi::PlaneMask::Stencil) != 0 ? 2u : 0u);
        const uint32_t mip0 = std::min(region.baseMip, physical.mips);
        const uint32_t mip1 = mip0 + std::min(region.mipCount, physical.mips - mip0);
        const uint32_t slice0 = std::min(region.baseSlice, physical.slices);
        const uint32_t slice1 = slice0 + std::min(region.sliceCount, physical.slices - slice0);
        for (uint32_t plane = 0; plane < physical.planes; ++plane)
        {
            if ((planeBits & (1u << plane)) == 0)
                continue;
            for (uint32_t mip = mip0; mip < mip1; ++mip)
                for (uint32_t slice = slice0; slice < slice1; ++slice)
                    states[(size_t(plane) * physical.mips + mip) * physical.slices + slice] = state;
        }
    }

    // The plan's frame-start states of the resource that `first` begins, applied over `states`.
    static void ApplyEntries(const CompiledRenderGraph& plan, size_t first, PhysicalResource& physical, std::vector<ResourceStates>& states)
    {
        const uint32_t resource = plan.m_EntryStates[first].resource;
        for (size_t e = first; e < plan.m_EntryStates.size() && plan.m_EntryStates[e].resource == resource; ++e)
            ApplyRegion(physical, plan.m_EntryStates[e].region, plan.m_EntryStates[e].state, states);
    }

    static size_t FindEntries(const CompiledRenderGraph& plan, uint32_t resource)
    {
        auto it = std::lower_bound(plan.m_EntryStates.begin(), plan.m_EntryStates.end(), resource,
            [](const PlanEntryState& entry, uint32_t value) { return entry.resource < value; });
        if (it == plan.m_EntryStates.end() || it->resource != resource)
            return c_None;
        return size_t(it - plan.m_EntryStates.begin());
    }

    // On a switch away from `previous`: its cyclic model becomes exact states (6.12). Non-history resources are already
    // in their frame-start states; a history texture is in the frame-start state of the layer it is at frame `frame`,
    // or of the next layer that `previous` uses (the layers it skips leave the texture alone).
    static void FinalizeStates(CompiledRenderGraph& previous, ResourceRegistry& registry, uint64_t frame)
    {
        const uint64_t serial = previous.m_Serial;
        StructureInstance* structure = previous.m_Instance ? previous.m_Instance->structure : nullptr;
        for (size_t e = 0; e < previous.m_EntryStates.size();)
        {
            const uint32_t resource = previous.m_EntryStates[e].resource;
            size_t next = e;
            while (next < previous.m_EntryStates.size() && previous.m_EntryStates[next].resource == resource)
                ++next;
            const PlanResource& record = previous.m_Resources[resource];
            if (record.kind == ResourceKind::History)
            {
                PersistentEntry& entry = registry.GetPersistentMutable(record.registryEntry);
                const uint32_t layers = std::max<uint32_t>(record.historyLayerCount, 1);
                for (uint32_t x = 0; x < entry.physicals.size(); ++x)
                {
                    PhysicalResource* physical = entry.physicals[x].get();
                    if (!physical || physical->stateOwner != serial)
                        continue;
                    const uint32_t layer = uint32_t((frame + uint64_t(layers) * 64 - x) % layers);
                    for (uint32_t j = 0; j < layers; ++j)
                    {
                        const uint32_t candidate = (layer + j) % layers;
                        // The plan's resource of that layer: the same history, layer `candidate`.
                        for (uint32_t r = 0; r < previous.m_Resources.size(); ++r)
                        {
                            const PlanResource& other = previous.m_Resources[r];
                            if (other.kind != ResourceKind::History || other.registryEntry != record.registryEntry ||
                                other.historyLayer != candidate)
                                continue;
                            const size_t first = FindEntries(previous, r);
                            if (first != c_None)
                            {
                                ApplyEntries(previous, first, *physical, physical->states);
                                j = layers;
                            }
                            break;
                        }
                    }
                    physical->stateOwner = 0;
                }
            }
            else if (record.kind == ResourceKind::Persistent)
            {
                PersistentEntry& entry = registry.GetPersistentMutable(record.registryEntry);
                if (!entry.physicals.empty() && entry.physicals[0]->stateOwner == serial)
                    entry.physicals[0]->stateOwner = 0;
            }
            else if (structure && record.physical != c_None && record.physical < structure->physicals.size())
            {
                PhysicalResource* physical = structure->physicals[record.physical].get();
                if (physical && physical->stateOwner == serial)
                    physical->stateOwner = 0;
            }
            e = next;
        }
    }

    // The barriers that bring every physical resource the plan touches into the plan's frame-start states: after
    // creation, and after a plan switch.
    static void BuildPatches(ExecutorFrame& frame, bool switched)
    {
        CompiledRenderGraph& plan = frame.plan;
        RegistryRuntime& runtime = frame.runtime;
        const BackendProfile& profile = plan.m_Profile;
        runtime.patches.clear();
        runtime.patchOwners.clear();
        for (size_t e = 0; e < plan.m_EntryStates.size();)
        {
            const uint32_t resource = plan.m_EntryStates[e].resource;
            size_t next = e;
            while (next < plan.m_EntryStates.size() && plan.m_EntryStates[next].resource == resource)
                ++next;
            PhysicalResource* physical = runtime.frameResources[resource].physical;
            const PlanResource& record = plan.m_Resources[resource];
            if (!physical || record.kind == ResourceKind::HostKept || record.kind == ResourceKind::Imported ||
                (!switched && !physical->fresh && physical->stateOwner == plan.m_Serial))
            {
                e = next;
                continue;
            }

            runtime.scratchStates = physical->states;
            ApplyEntries(plan, e, *physical, runtime.scratchStates);
            const bool transient = record.kind == ResourceKind::Transient || record.kind == ResourceKind::PassLocal;
            const bool contentIrrelevant = physical->fresh || transient;

            // One barrier when the whole resource moves together, else one per subresource.
            bool uniform = true;
            for (size_t i = 1; i < runtime.scratchStates.size() && uniform; ++i)
                uniform = runtime.scratchStates[i] == runtime.scratchStates[0] && physical->states[i] == physical->states[0];
            const uint32_t count = uniform ? 1u : uint32_t(runtime.scratchStates.size());
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceStates before = physical->fresh ? ResourceStates::Unknown : physical->states[i];
                const ResourceStates after = runtime.scratchStates[i];
                if (after == ResourceStates::Unknown || (before == after && !physical->fresh))
                    continue;
                nvrhi::SubresourceRegion region;
                if (!uniform)
                {
                    const uint32_t slice = i % physical->slices;
                    const uint32_t mip = (i / physical->slices) % physical->mips;
                    const uint32_t plane = i / (physical->slices * physical->mips);
                    region.setMips(mip, 1).setSlices(slice, 1);
                    if (physical->planes == 2)
                        region.setPlanes(plane == 0 ? nvrhi::PlaneMask::Depth : nvrhi::PlaneMask::Stencil);
                }
                const bool discard = contentIrrelevant &&
                    (profile.barrierModel != BarrierModel::D3D12Legacy ||
                     LegacyDiscardable(after, record.isBuffer, record.hasDepth, record.isRenderTarget));
                runtime.patches.push_back(nvrhi::ResourceBarrier::Transition(physical->resource, before, after, region,
                    discard ? nvrhi::BarrierFlags::Discard : nvrhi::BarrierFlags::None));
                runtime.patchOwners.push_back(physical);
            }
            physical->states.swap(runtime.scratchStates);
            physical->fresh = false;
            physical->stateOwner = plan.m_Serial;
            e = next;
        }
    }

    // ------------------------------------------------------------------------------------------------ recording

    static void ApplyBatch(ExecutorFrame& frame, nvrhi::ICommandList* list, std::vector<nvrhi::IResource*>& tracked,
                           uint32_t sched, uint32_t slot)
    {
        const PlanInstance& instance = frame.instance;
        const LoweredBatch& batch = instance.batches[size_t(sched) * c_SlotCount + slot];
        if (batch.explicitCount + batch.trackerCount)
            frame.barriers.fetch_add(batch.explicitCount + batch.trackerCount, std::memory_order_relaxed);
        if (batch.trackerCount)
            frame.stateCalls.fetch_add(batch.trackerCount, std::memory_order_relaxed);
        if (batch.trackerCount)
        {
            for (uint32_t t = batch.trackerBegin; t < batch.trackerBegin + batch.trackerCount; ++t)
            {
                const TrackerOp& op = instance.trackerOps[t];
                const FrameResource& resource = frame.runtime.frameResources[op.resource];
                const PlanResource& record = frame.plan.m_Resources[op.resource];
                // The fallback starts tracking a resource in each list at the state the plan says it holds.
                if (record.kind != ResourceKind::HostKept)
                {
                    if (std::find(tracked.begin(), tracked.end(), resource.resource) == tracked.end())
                    {
                        tracked.push_back(resource.resource);
                        if (resource.texture)
                            list->beginTrackingTextureState(resource.texture, nvrhi::AllSubresources, op.before);
                        else if (resource.buffer)
                            list->beginTrackingBufferState(resource.buffer, op.before);
                    }
                }
                if (resource.texture)
                    list->setTextureState(resource.texture, ToSubresourceSet(op.region), op.after);
                else if (resource.buffer)
                    list->setBufferState(resource.buffer, op.after);
            }
            list->commitBarriers();
        }
        if (batch.explicitCount)
        {
            // An aliasing barrier without `after` names a resource the Placed tier left dedicated (7.5): skipped.
            const nvrhi::ResourceBarrier* barriers = &instance.lowered[batch.explicitBegin];
            uint32_t run = 0;
            for (uint32_t i = 0; i < batch.explicitCount; ++i)
            {
                if (barriers[i].type != nvrhi::BarrierType::Aliasing || barriers[i].aliasing.after)
                    continue;
                if (i > run)
                    list->barriers(barriers + run, i - run);
                run = i + 1;
            }
            if (batch.explicitCount > run)
                list->barriers(barriers + run, batch.explicitCount - run);
        }
    }

    static void RecordClears(ExecutorFrame& frame, nvrhi::ICommandList* list, uint32_t sched)
    {
        const PlanPass& pass = frame.plan.m_Passes[sched];
        if (!pass.clearMask)
        {
            bool any = false;
            for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount && !any; ++u)
                any = frame.plan.m_Uses[u].hasClear;
            if (!any)
                return;
        }
        for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount; ++u)
        {
            const PlanUse& use = frame.plan.m_Uses[u];
            if (!use.hasClear)
                continue;
            const ClearValue& clear = frame.graph.GetAccess(use.accessIndex).clear;
            const FrameResource& resource = frame.runtime.frameResources[use.resource];
            if (resource.buffer)
            {
                list->clearBufferUInt(resource.buffer, uint32_t(clear.color[0]));
                continue;
            }
            if (!resource.texture)
                continue;
            const nvrhi::TextureSubresourceSet subresources = ToSubresourceSet(use.region);
            const nvrhi::FormatInfo& format = nvrhi::getFormatInfo(resource.texture->getDesc().format);
            if (format.hasDepth || format.hasStencil)
                list->clearDepthStencilTexture(resource.texture, subresources, format.hasDepth, clear.depth,
                                               format.hasStencil, clear.stencil);
            else if (format.kind == nvrhi::FormatKind::Integer)
                list->clearTextureUInt(resource.texture, subresources, uint32_t(clear.color[0]));
            else
                list->clearTextureFloat(resource.texture, subresources,
                                        nvrhi::Color(clear.color[0], clear.color[1], clear.color[2], clear.color[3]));
        }
    }

    // 11.4 FillTransients: the first write of a transient also fills it, where nvrhi can clear in the state the pass
    // left it in (D3D12 and D3D11; Vulkan clears need the transfer layout).
    static void FillTransients(ExecutorFrame& frame, nvrhi::ICommandList* list, uint32_t sched)
    {
        const BackendProfile& profile = frame.plan.m_Profile;
        if ((frame.plan.m_Options.debugFlags & DebugFlags::FillTransients) == DebugFlags::None ||
            profile.barrierModel == BarrierModel::Vulkan)
            return;
        const PlanPass& pass = frame.plan.m_Passes[sched];
        for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount; ++u)
        {
            const PlanUse& use = frame.plan.m_Uses[u];
            const PlanResource& record = frame.plan.m_Resources[use.resource];
            if (!use.isWrite || use.hasClear || record.firstPass != sched ||
                (record.kind != ResourceKind::Transient && record.kind != ResourceKind::PassLocal))
                continue;
            const nvrhi::ShaderType stages = use.stages == nvrhi::ShaderType::None ? DefaultStages(pass.type) : use.stages;
            const PlanResource& lowered = frame.plan.m_Resources[use.resource];
            const ResourceStates state = LowerAccess(use.access, stages, !lowered.isBuffer && (lowered.hasDepth || lowered.hasStencil),
                                                     frame.plan.m_Profile);
            if (profile.barrierModel != BarrierModel::Implicit &&
                state != ClearState(record.isBuffer, record.hasDepth, record.isRenderTarget, profile))
                continue;
            const FrameResource& resource = frame.runtime.frameResources[use.resource];
            if (resource.buffer && resource.buffer->getDesc().canHaveUAVs)
                list->clearBufferUInt(resource.buffer, 0xbaadf00du);
            else if (resource.texture && (record.isRenderTarget || resource.texture->getDesc().isUAV))
            {
                const nvrhi::FormatInfo& format = nvrhi::getFormatInfo(resource.texture->getDesc().format);
                if (format.hasDepth || format.hasStencil)
                    list->clearDepthStencilTexture(resource.texture, ToSubresourceSet(use.region), format.hasDepth, 0.5f, format.hasStencil, 0x5a);
                else if (format.kind == nvrhi::FormatKind::Integer)
                    list->clearTextureUInt(resource.texture, ToSubresourceSet(use.region), 0xbaadf00du);
                else
                    list->clearTextureFloat(resource.texture, ToSubresourceSet(use.region), nvrhi::Color(1.0f, 0.0f, 1.0f, 1.0f));
            }
        }
    }

    // Tracked and Restore passes (9.1): the tracker starts from the states the plan put the ranges in, which a merged
    // read group can make wider than the pass's own access; Restore returns them there.
    static void BeginTracking(ExecutorFrame& frame, nvrhi::ICommandList* list, uint32_t sched, bool restore)
    {
        const PlanPass& pass = frame.plan.m_Passes[sched];
        if (pass.trackedCount)
            frame.stateCalls.fetch_add(pass.trackedCount, std::memory_order_relaxed);
        for (uint32_t t = pass.trackedBegin; t < pass.trackedBegin + pass.trackedCount; ++t)
        {
            const PlanTrackedState& tracked = frame.plan.m_TrackedStates[t];
            if (frame.plan.m_Resources[tracked.resource].kind == ResourceKind::HostKept)
                continue;
            const FrameResource& resource = frame.runtime.frameResources[tracked.resource];
            if (!restore)
            {
                if (resource.texture)
                    list->beginTrackingTextureState(resource.texture, ToSubresourceSet(tracked.region), tracked.state);
                else if (resource.buffer)
                    list->beginTrackingBufferState(resource.buffer, tracked.state);
            }
            else
            {
                if (resource.texture)
                    list->setTextureState(resource.texture, ToSubresourceSet(tracked.region), tracked.state);
                else if (resource.buffer)
                    list->setBufferState(resource.buffer, tracked.state);
            }
        }
        if (restore)
            list->commitBarriers();
    }

    // Tracked mode, validation builds: any state change by the body is an undeclared access (9.1).
    static void CheckTracked(ExecutorFrame& frame, nvrhi::ICommandList* list, uint32_t sched)
    {
#ifndef NDEBUG
        const PlanPass& pass = frame.plan.m_Passes[sched];
        for (uint32_t t = pass.trackedBegin; t < pass.trackedBegin + pass.trackedCount; ++t)
        {
            const PlanTrackedState& tracked = frame.plan.m_TrackedStates[t];
            if (frame.plan.m_Resources[tracked.resource].kind == ResourceKind::HostKept)
                continue;
            const FrameResource& resource = frame.runtime.frameResources[tracked.resource];
            ResourceStates actual = tracked.state;
            if (resource.texture)
                actual = list->getTextureSubresourceState(resource.texture, tracked.region.baseSlice, tracked.region.baseMip);
            else if (resource.buffer)
                actual = list->getBufferState(resource.buffer);
            if (actual != tracked.state)
                frame.ReportError("pass '" + std::string(frame.plan.GetName(pass.nameOffset)) + "' left resource #" +
                    std::to_string(tracked.resource) + " in " + StatesToString(actual) + " instead of " +
                    StatesToString(tracked.state) + ": an undeclared access (Tracked mode)");
        }
#else
        (void)frame;
        (void)list;
        (void)sched;
#endif
    }

    static nvrhi::FRESULT RecordSegment(ExecutorFrame& frame, uint32_t segmentIndex, nvrhi::ICommandList* list, PassContext* context)
    {
        const CompiledRenderGraph& plan = frame.plan;
        const PlanSegment& segment = plan.m_Segments[segmentIndex];
        // The plan is the barrier source wherever there are barriers: through ICommandList::barriers, or through the
        // tracker's setState and commitBarriers in the fallback (9.1). Either way the automatic barriers stay off,
        // except around Tracked bodies.
        const bool planBarriers = plan.m_Profile.barrierModel != BarrierModel::Implicit;
        nvrhi::FRESULT result = nvrhi::FS_OK;
        context->trackedInList.clear();

        list->open();
        if (planBarriers)
            list->setEnableAutomaticBarriers(false);

        // Graph markers (PushMarker, PopMarker) that are open at the segment's first pass, then those between passes.
        uint32_t openMarkers = 0;
        uint32_t markerCursor = 0;
        const uint32_t firstProgram = plan.m_Passes[segment.firstPass].program;
        for (; markerCursor < frame.graph.MarkerCount(); ++markerCursor)
        {
            const MarkerRecord& marker = frame.graph.GetMarker(markerCursor);
            if (marker.beforePass > firstProgram)
                break;
            if (marker.name)
            {
                list->beginMarker(marker.name);
                ++openMarkers;
            }
            else if (openMarkers)
            {
                list->endMarker();
                --openMarkers;
            }
        }

        uint32_t recorded = 0;
        for (uint32_t s = segment.firstPass; s < plan.m_Passes.size() && recorded < segment.passCount; ++s)
        {
            const PlanPass& pass = plan.m_Passes[s];
            if (!pass.enabled || pass.segment != segmentIndex)
                continue;
            ++recorded;
            for (; markerCursor < frame.graph.MarkerCount(); ++markerCursor)
            {
                const MarkerRecord& marker = frame.graph.GetMarker(markerCursor);
                if (marker.beforePass > pass.program)
                    break;
                if (marker.name)
                {
                    list->beginMarker(marker.name);
                    ++openMarkers;
                }
                else if (openMarkers)
                {
                    list->endMarker();
                    --openMarkers;
                }
            }

            const PassRecord& record = frame.graph.GetPass(pass.program);
            const char* name = plan.GetName(pass.nameOffset);
            list->beginMarker(name);
            ApplyBatch(frame, list, context->trackedInList, s, uint32_t(BatchSlot::Prologue));
            RecordClears(frame, list, s);
            ApplyBatch(frame, list, context->trackedInList, s, uint32_t(BatchSlot::PostClear));
            FillTransients(frame, list, s);

            // D3D11 and CUDA have no barriers to track: Tracked and Restore change nothing there.
            const bool restore = planBarriers && (record.flags & PassFlags::Restore) != PassFlags::None;
            const bool tracked = planBarriers && (restore || (record.flags & PassFlags::Tracked) != PassFlags::None);
            if (tracked)
            {
                BeginTracking(frame, list, s, false);
                if (planBarriers)
                    list->setEnableAutomaticBarriers(true);
            }
            else if (planBarriers)
            {
                // Native: the plan is the only barrier source; the validation layer reports any other (8.2).
                list->setExplicitBarrierScope(name);
            }

            context->sched = s;
            context->commandList = list;
            nvrhi::ITimerQuery* timer = nullptr;
            if (frame.timers && frame.instance.timerSlots[s] != ~0u)
            {
                timer = frame.timers->queries[frame.instance.timerSlots[s]];
                list->beginTimerQuery(timer);
            }
            frame.passes.fetch_add(1, std::memory_order_relaxed);
            const nvrhi::FRESULT passResult = record.pass->Execute(context, record.data);
            if (timer)
                list->endTimerQuery(timer);
            if (passResult != nvrhi::FS_OK && result == nvrhi::FS_OK)
            {
                result = passResult;
                frame.ReportError(std::string("pass '") + name + "' failed in Execute");
            }

            if (tracked)
            {
                if (restore)
                    BeginTracking(frame, list, s, true);
                else
                    CheckTracked(frame, list, s);
                if (planBarriers)
                    list->setEnableAutomaticBarriers(false);
            }
            else if (planBarriers)
            {
                list->setExplicitBarrierScope(nullptr);
            }
            ApplyBatch(frame, list, context->trackedInList, s, uint32_t(BatchSlot::Epilogue));
            list->endMarker();
        }

        while (openMarkers--)
            list->endMarker();
        list->close();
        return result;
    }

    // A list of a stream. On a device with a queue pool (nvrhi ADR 0016) it is pinned to its lane's queue, so that
    // open() does not pick another queue of the type; elsewhere the type's one queue takes it.
    static nvrhi::FRESULT CreateStreamList(nvrhi::IDevice* device, CommandQueue queue, uint32_t lane, nvrhi::ICommandList** out)
    {
        const bool immediate = device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D11;
        nvrhi::CommandListParameters params = nvrhi::CommandListParameters().setQueueType(queue).setEnableImmediateExecution(immediate);
        nvrhi::AutoPtr<nvrhi::ICommandQueue> pinned;
        if (device->getCommandQueueCount(queue) > 1)
        {
            const nvrhi::FRESULT result = device->getCommandQueue(queue, lane, pinned.ReleaseAndGetAddressOf());
            if (result != nvrhi::FS_OK)
                return result;
            params.setCommandQueue(pinned.Get());
        }
        return device->createCommandList(params, out);
    }

    static nvrhi::ICommandList* SegmentList(ExecutorFrame& frame, uint32_t segmentIndex)
    {
        RegistryRuntime& runtime = frame.runtime;
        const PlanSegment& segment = frame.plan.m_Segments[segmentIndex];
        const CommandQueue queue = segment.queue;
        std::vector<nvrhi::AutoPtr<nvrhi::ICommandList>>& lists = runtime.commandLists[StreamOf(queue, segment.lane)];
        const uint32_t ordinal = frame.instance.segmentOrdinal[segmentIndex];
        while (lists.size() <= ordinal)
        {
            nvrhi::AutoPtr<nvrhi::ICommandList> list;
            if (CreateStreamList(frame.device, queue, segment.lane, list.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                return nullptr;
            ++runtime.objectsCreated;
            lists.push_back(list);
        }
        return lists[ordinal];
    }

    static PassContext* SegmentContext(ExecutorFrame& frame, uint32_t segmentIndex)
    {
        RegistryRuntime& runtime = frame.runtime;
        while (runtime.passContexts.size() <= segmentIndex)
        {
            runtime.passContexts.push_back(nvrhi::TakeOver(static_cast<IPassContext*>(MAKE_RC_OBJ(PassContext))));
            ++runtime.objectsCreated;
        }
        PassContext* context = static_cast<PassContext*>(runtime.passContexts[segmentIndex].Get());
        context->frame = &frame;
        return context;
    }

    // ------------------------------------------------------------------------------------------------ parallel recording

    // 11.3: each recorded pass's timer, in schedule order, on the calling thread before any recording.
    static void AssignTimerSlots(ExecutorFrame& frame)
    {
        const CompiledRenderGraph& plan = frame.plan;
        PlanInstance& instance = frame.instance;
        instance.timerSlots.assign(plan.m_Passes.size(), ~0u);
        if (!frame.timers)
            return;
        RegistryRuntime::TimerSet& set = *frame.timers;
        for (uint32_t s = 0; s < uint32_t(plan.m_Passes.size()); ++s)
        {
            const PlanPass& pass = plan.m_Passes[s];
            if (!pass.enabled || plan.m_Segments[pass.segment].queue == CommandQueue::Copy || set.used >= set.queries.size())
                continue;
            instance.timerSlots[s] = set.used;
            set.names[set.used].assign(plan.GetName(pass.nameOffset));
            ++set.used;
        }
    }

    // 10.1: the tasks of a parallel recording. A pass object records on one thread, so the segments that share one
    // (a pass's per-view uses) form a group. The groups go to at most `maxTasks` tasks, the largest group first to the
    // least loaded task; each task records its segments in plan order. Returns the number of tasks; below 2 the
    // segments are recorded on the calling thread.
    static uint32_t PlanRecordingTasks(ExecutorFrame& frame, uint32_t maxTasks)
    {
        const CompiledRenderGraph& plan = frame.plan;
        PlanInstance& instance = frame.instance;
        const uint32_t segmentCount = uint32_t(plan.m_Segments.size());

        std::vector<uint32_t>& group = instance.segmentGroup;
        group.resize(segmentCount);
        for (uint32_t i = 0; i < segmentCount; ++i)
            group[i] = i;
        auto find = [&group](uint32_t x) {
            while (group[x] != x)
            {
                group[x] = group[group[x]];
                x = group[x];
            }
            return x;
        };

        instance.passOwners.clear();
        for (const PlanPass& pass : plan.m_Passes)
        {
            if (pass.enabled)
                instance.passOwners.emplace_back(frame.graph.GetPass(pass.program).pass, pass.segment);
        }
        std::sort(instance.passOwners.begin(), instance.passOwners.end());
        for (size_t i = 1; i < instance.passOwners.size(); ++i)
        {
            if (instance.passOwners[i].first != instance.passOwners[i - 1].first)
                continue;
            const uint32_t a = find(instance.passOwners[i].second);
            const uint32_t b = find(instance.passOwners[i - 1].second);
            if (a != b)
                group[std::max(a, b)] = std::min(a, b);
        }

        // A group weighs what its segments took to record last frame; until each has been recorded once, its passes.
        bool measured = instance.segmentCost.size() == segmentCount;
        for (uint32_t i = 0; i < segmentCount && measured; ++i)
            measured = instance.segmentCost[i] != 0;
        instance.groupWeight.assign(segmentCount, 0);
        instance.groupOrder.clear();
        for (uint32_t i = 0; i < segmentCount; ++i)
        {
            const uint32_t root = find(i);
            instance.groupWeight[root] += measured ? instance.segmentCost[i] : plan.m_Segments[i].passCount;
            if (root == i)
                instance.groupOrder.push_back(i);
        }
        const uint32_t tasks = std::min(maxTasks, uint32_t(instance.groupOrder.size()));
        if (tasks < 2)
            return tasks;

        std::stable_sort(instance.groupOrder.begin(), instance.groupOrder.end(), [&instance](uint32_t a, uint32_t b) {
            return instance.groupWeight[a] > instance.groupWeight[b];
        });
        instance.taskLoad.assign(tasks, 0);
        instance.groupTask.assign(segmentCount, 0);
        for (uint32_t root : instance.groupOrder)
        {
            const uint32_t task = uint32_t(std::min_element(instance.taskLoad.begin(), instance.taskLoad.end()) - instance.taskLoad.begin());
            instance.groupTask[root] = task;
            instance.taskLoad[task] += instance.groupWeight[root];
        }

        instance.taskBegin.assign(tasks + 1, 0);
        for (uint32_t i = 0; i < segmentCount; ++i)
            ++instance.taskBegin[instance.groupTask[find(i)] + 1];
        for (uint32_t t = 0; t < tasks; ++t)
            instance.taskBegin[t + 1] += instance.taskBegin[t];
        instance.taskSegments.resize(segmentCount);
        instance.taskLoad.assign(instance.taskBegin.begin(), instance.taskBegin.end() - 1);   // reused as cursors
        for (uint32_t i = 0; i < segmentCount; ++i)
            instance.taskSegments[instance.taskLoad[instance.groupTask[find(i)]]++] = i;
        return tasks;
    }

    // A ParallelForFunction task: one task's segments. Nothing escapes to the host's scheduler.
    // Records one segment and keeps what it took, for the next frame's task weights.
    static void RecordTimedSegment(ExecutorFrame& frame, uint32_t segment)
    {
        PlanInstance& instance = frame.instance;
        const auto start = std::chrono::steady_clock::now();
        try
        {
            instance.segmentResults[segment] = RecordSegment(frame, segment, instance.segmentLists[segment],
                                                             instance.segmentContexts[segment]);
        }
        catch (...)
        {
            instance.segmentResults[segment] = nvrhi::FE_OUT_OF_MEMORY;
        }
        const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
        instance.segmentCost[segment] = std::max<uint64_t>(uint64_t(nanoseconds.count()), 1);
    }

    static void RecordTask(void* context, uint32_t task)
    {
        ExecutorFrame& frame = *static_cast<ExecutorFrame*>(context);
        PlanInstance& instance = frame.instance;
        for (uint32_t i = instance.taskBegin[task]; i < instance.taskBegin[task + 1]; ++i)
            RecordTimedSegment(frame, instance.taskSegments[i]);
    }

    // ------------------------------------------------------------------------------------------------ execute
    // 11.3: publishes the newest timer set whose queries have completed and frees every completed set. Each set is
    // polled once: a backend may report a query it has read as not started (Vulkan's getTimerQueryTime), so a set is
    // freed on that poll, not on a second one.
    static void CollectTimers(RegistryRuntime& runtime, nvrhi::IDevice* device)
    {
        bool ready[std::tuple_size<decltype(runtime.timerSets)>::value] = {};
        RegistryRuntime::TimerSet* newest = nullptr;
        for (size_t s = 0; s < runtime.timerSets.size(); ++s)
        {
            RegistryRuntime::TimerSet& set = runtime.timerSets[s];
            if (!set.pending)
                continue;
            ready[s] = true;
            for (uint32_t q = 0; q < set.used && ready[s]; ++q)
                ready[s] = device->pollTimerQuery(set.queries[q]);
            if (ready[s] && (!newest || set.frameIndex > newest->frameIndex))
                newest = &set;
        }
        if (newest)
        {
            runtime.timingNames.resize(newest->used);
            runtime.timings.resize(newest->used);
            for (uint32_t q = 0; q < newest->used; ++q)
            {
                runtime.timingNames[q] = newest->names[q];
                runtime.timings[q] = PassTiming{ runtime.timingNames[q].c_str(), newest->frameIndex,
                                                 device->getTimerQueryTime(newest->queries[q]) * 1000.f };
            }
        }
        // Every completed set is at most as new as the published one.
        for (size_t s = 0; s < runtime.timerSets.size(); ++s)
        {
            RegistryRuntime::TimerSet& set = runtime.timerSets[s];
            if (!ready[s])
                continue;
            for (uint32_t q = 0; q < set.used; ++q)
                device->resetTimerQuery(set.queries[q]);
            set.pending = false;
            set.used = 0;
        }
    }
    // A free set for this frame, with a query per enabled pass; nullptr when every set still waits for the GPU.
    static RegistryRuntime::TimerSet* AcquireTimers(RegistryRuntime& runtime, nvrhi::IDevice* device, const CompiledRenderGraph& plan,
                                                    uint64_t frameIndex)
    {
        CollectTimers(runtime, device);
        for (RegistryRuntime::TimerSet& set : runtime.timerSets)
        {
            if (set.pending)
                continue;
            const size_t needed = plan.m_Passes.size();
            while (set.queries.size() < needed)
            {
                nvrhi::AutoPtr<nvrhi::ITimerQuery> query;
                if (device->createTimerQuery(query.ReleaseAndGetAddressOf()) != nvrhi::FS_OK || !query)
                {
                    runtime.timersSupported = false;
                    return nullptr;
                }
                ++runtime.objectsCreated;
                set.queries.push_back(query);
            }
            if (set.names.size() < needed)
                set.names.resize(needed);
            set.used = 0;
            set.frameIndex = frameIndex;
            return &set;
        }
        return nullptr;
    }

    void FinalizeLastPlan(ResourceRegistry& registry) noexcept
    {
        RegistryRuntime& runtime = registry.m_Runtime;
        if (runtime.lastPlan)
        {
            // A history physical is in the frame-start state of the layer it serves in the frame after the last
            // executed one: the cyclic wrap at the end of that frame put it there.
            FinalizeStates(*runtime.lastPlan, registry, runtime.lastFrameIndex + 1);
            runtime.lastPlan = nullptr;
        }
    }

    nvrhi::FRESULT ExecutePlan(RenderGraph& graph, CompiledRenderGraph& plan, const ExecuteDesc& desc) noexcept
    {
        ResourceRegistry& registry = *graph.GetRegistry();
        RegistryRuntime& runtime = registry.m_Runtime;
        nvrhi::IDevice* device = registry.GetDevice();
        const BackendProfile& profile = plan.m_Profile;
        if (!device)
            return nvrhi::FE_UNSUPPORTED;
        if (device->getGraphicsAPI() != profile.api)
            return nvrhi::FE_INVALID_ARGS;

        try
        {
            // The host retires command lists with runGarbageCollection once per frame (10.1), as in any nvrhi frame;
            // DeviceManager does. The executor does not, so that a frame collects once.
            if (!runtime.fencesChecked)
            {
                runtime.fencesChecked = true;
                // D3D11 has one in-order queue: a fence would only signal and wait in the immediate context on every
                // submission, which costs the frame (P2.1 measured 10 %). One event query per frame tracks completion.
                runtime.fencesSupported = device->getGraphicsAPI() != nvrhi::GraphicsAPI::D3D11;
                static const char* const c_Names[3] = { "render graph: graphics", "render graph: compute", "render graph: copy" };
                for (uint32_t q = 0; q < 3 && runtime.fencesSupported; ++q)
                {
                    nvrhi::AutoPtr<nvrhi::IFence>& fence = runtime.fences[StreamOf(CommandQueue(q), 0)];
                    if (device->createFence(nvrhi::FenceDesc().setDebugName(c_Names[q]), fence.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                        runtime.fencesSupported = false;
                    else
                        ++runtime.objectsCreated;
                }
                if (!runtime.fencesSupported)
                {
                    // D3D11 (above), or a device without fences (8.3): one event query per frame instead.
                    for (auto& fence : runtime.fences)
                        fence = nullptr;
                    for (nvrhi::AutoPtr<nvrhi::IEventQuery>& query : runtime.frameQueries)
                    {
                        if (device->createEventQuery(query.ReleaseAndGetAddressOf()) == nvrhi::FS_OK)
                            ++runtime.objectsCreated;
                    }
                }
            }
            if (!runtime.prepareContext)
            {
                runtime.prepareContext = nvrhi::TakeOver(static_cast<IPrepareContext*>(MAKE_RC_OBJ(PrepareContext)));
                ++runtime.objectsCreated;
            }
            if (!runtime.prologueList)
            {
                // Graphics lane 0, the stream whose fence it signals.
                if (CreateStreamList(device, CommandQueue::Graphics, 0, runtime.prologueList.ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                    return nvrhi::FE_OUT_OF_MEMORY;
                ++runtime.objectsCreated;
            }

            CollectInstances(runtime);
            StructureInstance* structure = AcquireStructureInstance(plan, graph, runtime, device);
            if (!structure)
                return nvrhi::FE_OUT_OF_MEMORY;
            if (!plan.m_Instance)
            {
                // Queue lanes (nvrhi ADR 0016): every lane is a queue of the device's pool, with a fence of its own.
                for (const PlanSegment& segment : plan.m_Segments)
                {
                    if (segment.lane == 0)
                        continue;
                    if (segment.lane >= device->getCommandQueueCount(segment.queue))
                        return nvrhi::FE_INVALID_ARGS;   // a profile with more lanes than the device has queues
                    const uint32_t stream = StreamOf(segment.queue, segment.lane);
                    if (runtime.fencesSupported && !runtime.fences[stream])
                    {
                        const std::string name = std::string("render graph: ") + QueueName(segment.queue) + " lane " +
                            std::to_string(segment.lane);
                        if (device->createFence(nvrhi::FenceDesc().setDebugName(name.c_str()),
                                                runtime.fences[stream].ReleaseAndGetAddressOf()) != nvrhi::FS_OK)
                            return nvrhi::FE_OUT_OF_MEMORY;
                        ++runtime.objectsCreated;
                    }
                }
                plan.m_Instance = std::make_unique<PlanInstance>();
                plan.m_Instance->structure = structure;
                LowerPlan(plan, *plan.m_Instance);
            }
            PlanInstance& instance = *plan.m_Instance;
            const uint64_t frameIndex = graph.FrameIndexValue();
            ExecutorFrame frame{ graph, plan, instance, registry, runtime, device, frameIndex };
            const auto recordStart = std::chrono::steady_clock::now();
            if (desc.timers && runtime.timersSupported)
                frame.timers = AcquireTimers(runtime, device, plan, frameIndex);

            // This frame's resources.
            const uint32_t resourceCount = graph.ResourceCount();
            runtime.frameResources.resize(resourceCount);
            for (uint32_t r = 0; r < resourceCount; ++r)
            {
                const ResourceRecord& record = graph.GetResource(r);
                FrameResource& resource = runtime.frameResources[r];
                resource = FrameResource{ nullptr, nullptr, nullptr, nullptr };
                if (plan.m_Resources[r].firstPass == c_None && !record.extracted)
                    continue;   // unused this frame
                switch (record.kind)
                {
                case ResourceKind::Imported:
                case ResourceKind::HostKept:
                    resource.resource = record.imported;
                    break;
                case ResourceKind::Persistent:
                    resource.physical = PersistentPhysical(registry, record, 0, device, profile);
                    break;
                case ResourceKind::History:
                    resource.physical = PersistentPhysical(registry, record,
                        HistoryPhysicalIndex(frameIndex, record.historyLayer, std::max<uint32_t>(record.historyLayerCount, 1)), device, profile);
                    break;
                default:
                    if (record.extracted)
                        resource.physical = AcquireExtracted(runtime, record, device, profile);
                    else if (plan.m_Resources[r].physical != c_None)
                        resource.physical = structure->physicals[plan.m_Resources[r].physical].get();
                    break;
                }
                if (resource.physical)
                {
                    resource.resource = resource.physical->resource;
                    resource.physical->lastUsedFrame = runtime.framesExecuted;
                }
                else if (record.kind != ResourceKind::Imported && record.kind != ResourceKind::HostKept)
                {
                    return nvrhi::FE_OUT_OF_MEMORY;
                }
                if (resource.resource)
                {
                    if (record.isBuffer)
                        resource.buffer = static_cast<nvrhi::IBuffer*>(resource.resource);
                    else
                        resource.texture = static_cast<nvrhi::ITexture*>(resource.resource);
                }
            }

            if (!instance.fixedResolved)
            {
                PatchSlots(instance.fixedSlots, instance, runtime);
                MatchAliasingBarriers(instance, *structure, plan, device);
                instance.fixedResolved = true;
            }
            PatchSlots(instance.frameSlots, instance, runtime);

            // A plan switch (6.12): the old plan's states become exact, then the new plan's frame-start states.
            const bool switched = runtime.lastPlan.Get() != &plan;
            if (switched && runtime.lastPlan)
                FinalizeStates(*runtime.lastPlan, registry, runtime.lastFrameIndex + 1);
            BuildPatches(frame, switched);

            // The prologue: patches, then Prepare (13.1), which records static uploads into the same list. It is the
            // frame's first graphics submission, so it carries the queued waits (6.8).
            nvrhi::ICommandList* prologue = runtime.prologueList;
            frame.prologue = prologue;
            if (!runtime.patches.empty())
                frame.OpenPrologue();
            PrepareContext* prepare = static_cast<PrepareContext*>(runtime.prepareContext.Get());
            prepare->frame = &frame;
            nvrhi::FRESULT result = nvrhi::FS_OK;
            for (uint32_t s = 0; s < plan.m_Passes.size(); ++s)
            {
                const PlanPass& pass = plan.m_Passes[s];
                if (!pass.enabled)
                    continue;
                const PassRecord& record = graph.GetPass(pass.program);
                prepare->sched = s;
                const nvrhi::FRESULT prepared = record.pass->Prepare(prepare, record.data);
                if (prepared != nvrhi::FS_OK && result == nvrhi::FS_OK)
                {
                    result = prepared;
                    frame.ReportError(std::string("pass '") + plan.GetName(pass.nameOffset) + "' failed in Prepare");
                }
            }
            if (frame.prologueOpen)
                prologue->close();

            // Record every segment, on worker threads where the host gives a scheduler (10.1), then submit in plan
            // order. The lists, contexts and timer slots are taken first, on this thread.
            const uint32_t segmentCount = uint32_t(plan.m_Segments.size());
            instance.segmentLists.resize(segmentCount);
            instance.segmentContexts.resize(segmentCount);
            instance.segmentResults.assign(segmentCount, nvrhi::FS_OK);
            for (uint32_t i = 0; i < segmentCount; ++i)
            {
                instance.segmentLists[i] = SegmentList(frame, i);
                if (!instance.segmentLists[i])
                    return nvrhi::FE_OUT_OF_MEMORY;
                instance.segmentContexts[i] = SegmentContext(frame, i);
            }
            AssignTimerSlots(frame);
            instance.segmentCost.resize(segmentCount, 0);
            // The recording tasks (from the instance and the previous frame's segment costs), planned before the
            // prologue is submitted: their planning can fail (out of memory), and nothing that can fail comes between
            // that submission and the recording (F9-R1).
            const bool parallel =
                desc.parallelFor && profile.recordingMode != RecordingMode::ImmediateSerial && segmentCount > 1;
            const uint32_t tasks =
                parallel ? PlanRecordingTasks(frame, desc.workerCount ? desc.workerCount : segmentCount) : 0;

            // The prologue (the patches, then Prepare's uploads) is submitted before any segment records (9.1): nvrhi
            // applies a permanent state that the upload list sets (Scene::WriteMeshBufferData: the mesh buffers made
            // at extraction) only when that list is submitted, and the segments' lists must find it while they record
            // (backlog S-1). It is still the frame's first graphics submission, and every stream's first segment waits
            // for it, so the GPU sees the same order. Past the frame's last return (and throw) before recording: a frame
            // that leaves earlier loses its Prepare, as before.
            const auto prologueStart = std::chrono::steady_clock::now();
            const bool fences = runtime.fencesSupported;
            const bool prologueNeeded = frame.prologueOpen;
            uint64_t prologueValue = 0;
            constexpr uint32_t c_PrologueStream = 0;   // StreamOf(Graphics, 0)
            if (prologueNeeded)
            {
                nvrhi::FenceValue waits[c_MaxStreams];
                uint32_t waitCount = 0;
                if (fences && switched && !runtime.patches.empty())
                {
                    // The patch follows the previous frame on every stream.
                    for (uint32_t q = 0; q < c_MaxStreams; ++q)
                    {
                        if (q != c_PrologueStream && runtime.lastFrameValues[q])
                            waits[waitCount++] = nvrhi::FenceValue{ runtime.fences[q], runtime.lastFrameValues[q] };
                    }
                }
                nvrhi::ICommandList* const lists[] = { prologue };
                if (fences)
                {
                    prologueValue = ++runtime.fenceValues[c_PrologueStream];
                    const nvrhi::FenceValue signal{ runtime.fences[c_PrologueStream], prologueValue };
                    device->submit(nvrhi::SubmitDesc().setCommandLists(lists, 1).setQueue(CommandQueue::Graphics)
                        .setWaits(waits, waitCount).setSignals(&signal, 1));
                }
                else
                {
                    device->executeCommandList(prologue, CommandQueue::Graphics);
                }
            }
            const auto prologueEnd = std::chrono::steady_clock::now();
            if (tasks > 1)
            {
                desc.parallelFor(desc.schedulerContext, tasks, RecordTask, &frame);
            }
            else
            {
                for (uint32_t i = 0; i < segmentCount; ++i)
                    RecordTimedSegment(frame, i);
            }
            for (uint32_t i = 0; i < segmentCount; ++i)
            {
                if (instance.segmentResults[i] != nvrhi::FS_OK && result == nvrhi::FS_OK)
                    result = instance.segmentResults[i];
            }
            runtime.objectsCreated += frame.framebuffersCreated.load(std::memory_order_relaxed);
            frame.recordTasks = tasks > 1 ? tasks : 1;

            const auto submitStart = std::chrono::steady_clock::now();

            // Per stream: a (queue, lane) pair. Segments of one stream run in order; streams order through fences.
            std::array<bool, c_MaxStreams> firstOfQueue;
            firstOfQueue.fill(true);
            uint32_t submissions = 0;
            for (uint32_t i = 0; i < segmentCount; )
            {
                const PlanSegment& segment = plan.m_Segments[i];
                const uint32_t q = StreamOf(segment.queue, segment.lane);
                if (!fences)
                {
                    device->executeCommandList(runtime.commandLists[q][instance.segmentOrdinal[i]], segment.queue);
                    firstOfQueue[q] = false;
                    ++submissions;
                    ++i;
                    continue;
                }
                // One submission for a run of segments, in plan order, on one stream: every segment after the first
                // waits for nothing, and nothing waits for one before the last (PlanSegment::signals). Every wait and
                // every signal that some segment waits for stays where separate submissions put it; the run's
                // segments share its one fence value.
                constexpr uint32_t c_MaxBatch = 16;
                nvrhi::ICommandList* lists[c_MaxBatch];
                uint32_t end = i;
                do
                {
                    lists[end - i] = runtime.commandLists[q][instance.segmentOrdinal[end]];
                    ++end;
                } while (end < segmentCount && end - i < c_MaxBatch && plan.m_Segments[end].queue == segment.queue &&
                         plan.m_Segments[end].lane == segment.lane && plan.m_Segments[end].waitCount == 0 &&
                         !plan.m_Segments[end - 1].signals);
                nvrhi::FenceValue waits[c_MaxStreams];
                uint32_t waitCount = 0;
                auto addWait = [&](uint32_t queue, uint64_t value) {
                    if (!value || queue == q)
                        return;
                    for (uint32_t w = 0; w < waitCount; ++w)
                    {
                        if (waits[w].fence == runtime.fences[queue].Get())
                        {
                            waits[w].value = std::max(waits[w].value, value);
                            return;
                        }
                    }
                    if (waitCount < c_MaxStreams)
                        waits[waitCount++] = nvrhi::FenceValue{ runtime.fences[queue], value };
                };
                for (uint32_t w = 0; w < segment.waitCount; ++w)
                {
                    const SegmentWait& wait = plan.m_SegmentWaits[segment.waitBegin + w];
                    const uint32_t producer = StreamOf(wait.queue, wait.lane);
                    if (wait.frameDelta == 0)
                        addWait(producer, instance.segmentValue[wait.segment]);
                    else if (!switched)
                        addWait(producer, instance.previousSegmentValue[wait.segment]);
                }
                if (firstOfQueue[q])
                {
                    // The prologue's patches and uploads come before every stream's work.
                    if (prologueNeeded)
                        addWait(c_PrologueStream, prologueValue);
                    if (switched)
                    {
                        // 6.12: after a plan switch the first segment of each stream follows the whole previous frame.
                        for (uint32_t other = 0; other < c_MaxStreams; ++other)
                            addWait(other, runtime.lastFrameValues[other]);
                    }
                }
                firstOfQueue[q] = false;
                const uint64_t value = ++runtime.fenceValues[q];
                for (uint32_t s = i; s < end; ++s)
                    instance.segmentValue[s] = value;
                const nvrhi::FenceValue signal{ runtime.fences[q], value };
                device->submit(nvrhi::SubmitDesc().setCommandLists(lists, end - i).setQueue(segment.queue)
                    .setWaits(waits, waitCount).setSignals(&signal, 1));
                ++submissions;
                i = end;
            }

            if (!fences && runtime.frameQueries[runtime.frameQueryNext])
            {
                const uint32_t q = runtime.frameQueryNext;
                nvrhi::IEventQuery* query = runtime.frameQueries[q];
                if (runtime.frameQueryPending[q])
                    device->waitEventQuery(query);
                device->resetEventQuery(query);
                device->setEventQuery(query, CommandQueue::Graphics);
                runtime.frameQueryPending[q] = true;
                runtime.frameQueryNext = (q + 1) % uint32_t(runtime.frameQueries.size());
            }

            // Extracted textures go to their callers (5.2).
            for (uint32_t r = 0; r < resourceCount; ++r)
            {
                const ResourceRecord& record = graph.GetResource(r);
                if (record.extracted && record.extractOut && runtime.frameResources[r].texture)
                {
                    *record.extractOut = runtime.frameResources[r].texture;
                    (*record.extractOut)->AddRef();
                }
            }

            runtime.lastFrameValues = runtime.fenceValues;
            instance.previousSegmentValue.swap(instance.segmentValue);
            // 11.3: the run statistics, and the timer set waits for the GPU.
            const auto submitEnd = std::chrono::steady_clock::now();
            RunStatistics& run = runtime.run;
            run.frameIndex = frameIndex;
            // the prologue's submission, made before the recording, counts as submission time
            using Milliseconds = std::chrono::duration<float, std::milli>;
            const auto prologueTime = prologueEnd - prologueStart;
            run.recordMilliseconds = Milliseconds(submitStart - recordStart - prologueTime).count();
            run.submitMilliseconds = Milliseconds(submitEnd - submitStart + prologueTime).count();
            run.passes = frame.passes.load(std::memory_order_relaxed);
            run.recordTasks = frame.recordTasks;
            run.segments = segmentCount;
            run.submissions = submissions;
            run.barriers = frame.barriers.load(std::memory_order_relaxed) + uint32_t(runtime.patches.size());
            run.stateCalls = frame.stateCalls.load(std::memory_order_relaxed) +
                (plan.m_Profile.explicitBarriers ? 0u : 2u * uint32_t(runtime.patches.size()));
            run.planSwitched = switched;
            if (frame.timers && frame.timers->used)
                frame.timers->pending = true;

            runtime.lastPlan = &plan;
            runtime.lastFrameIndex = frameIndex;
            ++runtime.framesExecuted;
            return result;
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
    }
}
