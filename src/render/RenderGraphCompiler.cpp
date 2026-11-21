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

// The compiler (design 7): phases 1-5 and 7 over one shared compile state, and the plan they produce. Phase 6 is in
// RenderGraphTransitions.cpp.
//
// Two levels (7.7, 13.5):
// - The structure, per structural hash: validation, queue assignment (async placement, 6.8), the superset's
//   lifetimes (every pass enabled) and the memory plan (7.5). Every pass-enable mask shares it.
// - The plan, per mask: the Cull policies and culling, the schedule, the transitions, the segments.

#include "RenderGraphCompile.h"

#include <algorithm>
#include <unordered_map>

namespace donut::render::rdg
{
    using nvrhi::CommandQueue;
    using nvrhi::ResourceStates;

    uint64_t HashOptions(const CompileOptions& options) noexcept
    {
        // StrictValidation changes what Compile checks, not the plan; maxCachedPlans changes the cache.
        const DebugFlags planFlags = options.debugFlags & ~DebugFlags::StrictValidation;
        uint64_t hash = HashCombine(c_HashSeed, uint64_t(options.placedResources) | (uint64_t(planFlags) << 8) |
            (uint64_t(options.scheduleFlags) << 40));
        return HashCombine(hash, options.recordGranularity);
    }

    namespace
    {
        bool IsTransientKind(const ResourceRecord& record) noexcept
        {
            return record.kind == ResourceKind::Transient || record.kind == ResourceKind::PassLocal;
        }

        nvrhi::SubresourceRegion ResolveRegion(const ResourceRecord& record, const nvrhi::SubresourceRegion& region,
                                               const BackendProfile& profile) noexcept
        {
            nvrhi::SubresourceRegion result;
            if (record.isBuffer)
            {
                result.baseMip = 0;
                result.mipCount = 1;
                result.baseSlice = 0;
                result.sliceCount = 1;
                result.planes = nvrhi::PlaneMask::All;
                return result;
            }
            const uint32_t mips = MipCount(record);
            const uint32_t slices = SliceCount(record);
            result.baseMip = std::min(region.baseMip, mips);
            result.mipCount = std::min(region.mipCount, mips - result.baseMip);
            result.baseSlice = std::min(region.baseSlice, slices);
            result.sliceCount = std::min(region.sliceCount, slices - result.baseSlice);
            if (UnitPlaneCount(record, profile) == 1)
            {
                // One tracked plane: any named plane of the format selects it.
                nvrhi::PlaneMask own = record.hasDepth ? (nvrhi::PlaneMask::Depth | nvrhi::PlaneMask::Stencil) : nvrhi::PlaneMask::Color;
                if (profile.planeModel == PlaneModel::Ignored || profile.barrierModel == BarrierModel::Implicit)
                    own = nvrhi::PlaneMask::All;
                result.planes = (region.planes & own) != 0 ? nvrhi::PlaneMask::All : nvrhi::PlaneMask::None;
            }
            else
            {
                result.planes = region.planes & (nvrhi::PlaneMask::Depth | nvrhi::PlaneMask::Stencil);
            }
            return result;
        }

        // The planes a declared region names, in terms of the format: Depth and Stencil bits, or Color.
        uint32_t DeclaredPlanes(const ResourceRecord& record, nvrhi::PlaneMask planes) noexcept
        {
            if (record.isBuffer)
                return 1;
            if (!record.hasDepth)
                return (planes & nvrhi::PlaneMask::Color) != 0 ? 1u : 0u;
            uint32_t bits = 0;
            if ((planes & nvrhi::PlaneMask::Depth) != 0)
                bits |= 2;
            if (record.hasStencil && (planes & nvrhi::PlaneMask::Stencil) != 0)
                bits |= 4;
            return bits;
        }

        // Whether a declared region covers every subresource and plane of the resource.
        bool CoversWholeResource(const ResourceRecord& record, const nvrhi::SubresourceRegion& region) noexcept
        {
            if (record.isBuffer)
                return true;
            const uint32_t allPlanes = record.hasDepth ? (record.hasStencil ? 6u : 2u) : 1u;
            return region.baseMip == 0 && region.mipCount >= MipCount(record) && region.baseSlice == 0 &&
                region.sliceCount >= SliceCount(record) && DeclaredPlanes(record, region.planes) == allPlanes;
        }

        // A write that leaves nothing of the previous contents: a FullWrite or Clear of the whole resource.
        bool OverwritesAll(const ResourceRecord& record, const AccessRecord& access) noexcept
        {
            return access.isWrite && (access.flags & (AccessFlags::FullWrite | AccessFlags::Clear)) != AccessFlags::None &&
                CoversWholeResource(record, access.region);
        }

        bool DeclaredRegionsOverlap(const ResourceRecord& record, const nvrhi::SubresourceRegion& a,
                                    const nvrhi::SubresourceRegion& b) noexcept
        {
            if (record.isBuffer)
                return true;
            const uint64_t a0 = a.baseMip, a1 = uint64_t(a.baseMip) + a.mipCount;
            const uint64_t b0 = b.baseMip, b1 = uint64_t(b.baseMip) + b.mipCount;
            const uint64_t s0 = a.baseSlice, s1 = uint64_t(a.baseSlice) + a.sliceCount;
            const uint64_t t0 = b.baseSlice, t1 = uint64_t(b.baseSlice) + b.sliceCount;
            return a0 < b1 && b0 < a1 && s0 < t1 && t0 < s1 &&
                (DeclaredPlanes(record, a.planes) & DeclaredPlanes(record, b.planes)) != 0;
        }

        bool AllowedOverlap(Access a, Access b) noexcept
        {
            if (a == b)
                return true;
            const Access combined = a | b;
            const Access depthAndSampling = Access::DepthRead | Access::StencilRead | Access::ShaderRead;
            const Access uav = Access::UAVRead | Access::UAVWrite;
            const Access blend = Access::ColorTarget | Access::ColorTargetRead;
            return (combined & ~depthAndSampling) == Access::None || (combined & ~uav) == Access::None ||
                (combined & ~blend) == Access::None;
        }

        // ------------------------------------------------------------------------------------ phase 1: validate

        void ValidateStructure(CompileState& state)
        {
            const RenderGraph& graph = state.graph;
            const BackendProfile& profile = state.profile;

            for (uint32_t p = 0; p < graph.PassCount(); ++p)
            {
                const PassRecord& pass = graph.GetPass(p);

                // 6.2: the declared ranges of one resource in one pass may overlap only in three combinations.
                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    const AccessRecord& first = graph.GetAccess(a);
                    for (uint32_t b = a + 1; b < pass.firstAccess + pass.accessCount; ++b)
                    {
                        const AccessRecord& second = graph.GetAccess(b);
                        if (first.resource != second.resource)
                            continue;
                        const ResourceRecord& record = graph.GetResource(first.resource);
                        if (DeclaredRegionsOverlap(record, first.region, second.region) && !AllowedOverlap(first.access, second.access))
                            state.Error(state.PassName(p) + ": the accesses " + AccessToString(first.access) + " and " +
                                AccessToString(second.access) + " of " + state.ResourceName(first.resource) +
                                " overlap; only depth reads with shader reads, UAV reads with UAV writes and colour "
                                "targets with blending may overlap within a pass (6.2)");
                    }
                }
                const uint32_t typeBit = 1u << uint32_t(pass.type);
                if ((uint32_t(profile.passTypes) & typeBit) == 0)
                {
                    state.Error(state.PassName(p) + ": the profile has no " +
                        (pass.type == PassType::Raster ? std::string("raster") : pass.type == PassType::Compute ? std::string("compute") :
                         pass.type == PassType::Copy ? std::string("copy") : std::string("ray-tracing")) + " passes (9.2)");
                }
                const bool async = (pass.flags & PassFlags::AsyncCompute) != PassFlags::None;
                if (async && pass.type == PassType::Raster)
                    state.Error(state.PassName(p) + ": a raster pass cannot run on the compute queue (AsyncCompute)");

                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    const AccessRecord& access = graph.GetAccess(a);
                    const ResourceRecord& record = graph.GetResource(access.resource);
                    const std::string where = state.PassName(p) + ": " + state.ResourceName(access.resource);
                    const Access bits = access.access;
                    auto has = [bits](Access mask) { return (bits & mask) != Access::None; };

                    if (async && record.kind == ResourceKind::HostKept)
                        state.Error(where + " is a back buffer (host-kept); an AsyncCompute pass cannot touch it (6.8)");
                    if (access.attachment != c_NoAttachment && pass.type != PassType::Raster)
                        state.Error(where + " is bound as a render target in a pass that is not a raster pass");

                    if (record.isBuffer)
                    {
                        const BufferInfo& info = record.buffer;
                        if (has(Access::UAVRead | Access::UAVWrite) && !info.canHaveUAVs)
                            state.Error(where + ": a UAV access on a buffer without canHaveUAVs");
                        if (has(Access::VertexBuffer) && !info.isVertexBuffer)
                            state.Error(where + ": a vertex-buffer access on a buffer without isVertexBuffer");
                        if (has(Access::IndexBuffer) && !info.isIndexBuffer)
                            state.Error(where + ": an index-buffer access on a buffer without isIndexBuffer");
                        if (has(Access::ConstantBuffer) && !info.isConstantBuffer)
                            state.Error(where + ": a constant-buffer access on a buffer without isConstantBuffer");
                        if (has(Access::IndirectArgs) && !info.isDrawIndirectArgs)
                            state.Error(where + ": an indirect-arguments access on a buffer without isDrawIndirectArgs");
                    }
                    else
                    {
                        const TextureInfo& info = record.texture;
                        if (has(Access::UAVRead | Access::UAVWrite) && !info.isUAV)
                            state.Error(where + ": a UAV access on a texture without isUAV");
                        if (has(Access::ColorTarget | Access::ColorTargetRead) && !info.isRenderTarget)
                            state.Error(where + ": a colour-target access on a texture without isRenderTarget");
                        if (has(Access::DepthRead | Access::DepthWrite | Access::StencilRead | Access::StencilWrite) && !record.hasDepth)
                            state.Error(where + ": a depth or stencil access on a format without depth");
                        if (has(Access::DepthWrite | Access::StencilWrite) && !info.isRenderTarget)
                            state.Error(where + ": a depth write on a texture without isRenderTarget");
                        if (has(Access::ShaderRead) && !info.isShaderResource)
                            state.Error(where + ": a shader read on a texture without isShaderResource");
                        if (profile.api == nvrhi::GraphicsAPI::CUDA && has(Access::UAVWrite))
                        {
                            // 2.1: CUDA surfaces are uncompressed, not 3-component and not depth-stencil.
                            const nvrhi::FormatInfo& format = nvrhi::getFormatInfo(info.format);
                            const bool threeComponent = format.hasRed && format.hasGreen && format.hasBlue && !format.hasAlpha;
                            if (format.blockSize > 1 || threeComponent || format.hasDepth || format.hasStencil)
                                state.Error(where + ": CUDA cannot write this format through a surface (compressed, "
                                    "3-component or depth-stencil)");
                        }
                    }
                    if (access.hasClear && has(Access::DepthWrite | Access::StencilWrite) && !profile.depthClears)
                        state.Error(where + ": the profile has no depth clears (CUDA, 6.9)");
                    if (access.hasClear)
                    {
                        // nvrhi clears a texture through an RTV, a DSV or a UAV, and a buffer through a UAV.
                        const bool clearable = record.isBuffer ? record.buffer.canHaveUAVs
                                                               : (record.texture.isRenderTarget || record.texture.isUAV);
                        if (!clearable)
                            state.Error(where + ": a Clear needs a render target or UAV texture, or a buffer with canHaveUAVs");
                    }
                }
            }
        }

        // ------------------------------------------------------------------------------------ phase 2: cull

        void ApplyMask(CompileState& state)
        {
            const RenderGraph& graph = state.graph;
            const uint32_t passCount = graph.PassCount();
            const uint32_t accessCount = graph.AccessCount();

            state.enabled.assign(passCount, 1);
            state.live.assign(passCount, 0);
            state.accessResource.resize(accessCount);
            for (uint32_t a = 0; a < accessCount; ++a)
                state.accessResource[a] = graph.GetAccess(a).resource;

            if (!state.superset)
            {
                for (uint32_t p = 0; p < passCount; ++p)
                    state.enabled[p] = graph.GetPass(p).enabled ? 1 : 0;

                // Exclusion groups (13.5): never two enabled members.
                std::unordered_map<uint32_t, uint32_t> groupOwner;
                for (uint32_t p = 0; p < passCount; ++p)
                {
                    const uint32_t group = graph.GetPass(p).exclusionGroup;
                    if (!group || !state.enabled[p])
                        continue;
                    auto [it, inserted] = groupOwner.emplace(group, p);
                    if (!inserted)
                        state.Error(state.PassName(p) + " and " + state.PassName(it->second) + " are both enabled, but "
                            "they share exclusion group " + std::to_string(group) + " (13.5)");
                }

                // Disabled outputs (13.5), in program order, so that Cull cascades.
                for (uint32_t p = 0; p < passCount; ++p)
                {
                    if (state.enabled[p])
                        continue;
                    const PassRecord& pass = graph.GetPass(p);
                    for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                    {
                        const AccessRecord& written = graph.GetAccess(a);
                        if (!written.isWrite)
                            continue;
                        DisabledOutput policy = DisabledOutput::Cull;
                        uint32_t substitute = c_None;
                        for (uint32_t i = 0; i < graph.PolicyCount(); ++i)
                        {
                            const PolicyRecord& record = graph.GetPolicy(i);
                            if (record.pass == p && record.resource == written.resource && record.version == written.version)
                            {
                                policy = record.policy;
                                substitute = record.substitute;
                                break;
                            }
                        }
                        if (policy != DisabledOutput::Cull)
                        {
                            const ResourceRecord& output = graph.GetResource(written.resource);
                            const ResourceRecord& input = graph.GetResource(substitute);
                            if (output.isBuffer != input.isBuffer || output.descHash != input.descHash)
                                state.Error(state.PassName(p) + ": the " + (policy == DisabledOutput::Substitute ?
                                    std::string("substitute ") : std::string("passthrough input ")) + state.ResourceName(substitute) +
                                    " does not match the description of the output " + state.ResourceName(written.resource) + " (13.5)");
                        }

                        for (uint32_t q = p + 1; q < passCount; ++q)
                        {
                            const PassRecord& consumer = graph.GetPass(q);
                            for (uint32_t c = consumer.firstAccess; c < consumer.firstAccess + consumer.accessCount; ++c)
                            {
                                const AccessRecord& access = graph.GetAccess(c);
                                if (access.resource != written.resource)
                                    continue;
                                const bool fullWrite = OverwritesAll(graph.GetResource(access.resource), access);
                                const bool consumes = access.isWrite ? (!fullWrite && access.prevVersion == written.version)
                                                                     : access.version == written.version;
                                if (!consumes)
                                    continue;
                                if (policy == DisabledOutput::Cull || (access.isWrite && substitute != written.resource))
                                    state.enabled[q] = 0;
                                else if (!access.isWrite)
                                    state.accessResource[c] = substitute;
                            }
                        }
                    }
                }
            }

            // Reference counts (7, phase 2): a pass lives if it has a side effect, writes something that outlives
            // the graph, presents, or produces what a live pass needs.
            std::vector<uint32_t> lastWriter(graph.ResourceCount(), c_None);
            std::vector<uint32_t> depBegin(passCount + 1, 0);
            std::vector<uint32_t> deps;
            std::vector<uint8_t> root(passCount, 0);
            for (uint32_t p = 0; p < passCount; ++p)
            {
                depBegin[p] = uint32_t(deps.size());
                if (!state.enabled[p])
                    continue;
                const PassRecord& pass = graph.GetPass(p);
                if ((pass.flags & (PassFlags::NeverCull | PassFlags::SideEffect)) != PassFlags::None)
                    root[p] = 1;
                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    const AccessRecord& access = graph.GetAccess(a);
                    const uint32_t resource = state.accessResource[a];
                    const ResourceRecord& record = graph.GetResource(resource);
                    // A write that overwrites only part of the resource keeps the earlier writers alive (the other
                    // mips, slices or planes may still be read).
                    if (!OverwritesAll(record, access) && lastWriter[resource] != c_None)
                        deps.push_back(lastWriter[resource]);
                    if (access.isWrite && (record.kind == ResourceKind::Imported || record.kind == ResourceKind::HostKept ||
                        record.kind == ResourceKind::Persistent || record.kind == ResourceKind::History || record.extracted))
                        root[p] = 1;
                    if ((access.access & (Access::Present | Access::HostRead)) != Access::None)
                        root[p] = 1;
                }
                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    if (graph.GetAccess(a).isWrite)
                        lastWriter[state.accessResource[a]] = p;
                }
            }
            depBegin[passCount] = uint32_t(deps.size());
            for (uint32_t p = passCount; p-- > 0;)
            {
                if (!state.enabled[p])
                    continue;
                if (root[p])
                    state.live[p] = 1;
                if (!state.live[p])
                    continue;
                for (uint32_t d = depBegin[p]; d < depBegin[p + 1]; ++d)
                    state.live[deps[d]] = 1;
            }
        }

        // ------------------------------------------------------------------------------------ phase 3: schedule

        void BuildSchedule(CompileState& state, const std::vector<CommandQueue>& passQueues)
        {
            const RenderGraph& graph = state.graph;
            const uint32_t passCount = graph.PassCount();
            state.schedule.clear();
            state.scheduledLive.clear();
            state.schedQueue.clear();
            state.schedLane.clear();
            state.scheduleOf.assign(passCount, c_None);
            const std::vector<uint8_t>* passLanes = state.structure && !state.structure->passLanes.empty()
                ? &state.structure->passLanes : nullptr;
            for (uint32_t p = 0; p < passCount; ++p)
            {
                // Program order. Disabled passes stay in the schedule, without work, so that the plan can show them.
                if (!state.live[p] && state.enabled[p])
                    continue;
                state.scheduleOf[p] = uint32_t(state.schedule.size());
                state.schedule.push_back(p);
                state.scheduledLive.push_back(state.live[p]);
                state.schedQueue.push_back(passQueues[p]);
                state.schedLane.push_back(passLanes ? (*passLanes)[p] : uint8_t(0));
            }
        }

        void BuildUses(CompileState& state)
        {
            const RenderGraph& graph = state.graph;
            const uint32_t count = uint32_t(state.schedule.size());
            state.uses.clear();
            state.useState.clear();
            state.useStages.clear();
            state.useBegin.assign(count, 0);
            state.useCount.assign(count, 0);
            const Access shaderAccess = Access::ShaderRead | Access::UAVRead | Access::UAVWrite | Access::ConstantBuffer |
                Access::AccelStructRead;

            for (uint32_t s = 0; s < count; ++s)
            {
                state.useBegin[s] = uint32_t(state.uses.size());
                if (!state.scheduledLive[s])
                    continue;
                const PassRecord& pass = graph.GetPass(state.schedule[s]);
                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    const AccessRecord& access = graph.GetAccess(a);
                    const uint32_t resource = state.accessResource[a];
                    const ResourceRecord& record = graph.GetResource(resource);
                    PlanUse use{};
                    use.accessIndex = a;
                    use.resource = resource;
                    use.access = access.access;
                    use.stages = access.stages;
                    use.flags = access.flags;
                    use.region = ResolveRegion(record, access.region, state.profile);
                    use.attachment = access.attachment;
                    use.isWrite = access.isWrite;
                    use.hasClear = access.hasClear;
                    if (!record.isBuffer && (use.region.planes == nvrhi::PlaneMask::None || use.region.mipCount == 0 ||
                        use.region.sliceCount == 0))
                    {
                        state.Error(state.PassName(state.schedule[s]) + ": the region of its access to " +
                            state.ResourceName(resource) + " names no subresource");
                        continue;
                    }
                    const nvrhi::ShaderType stages = access.stages == nvrhi::ShaderType::None ? DefaultStages(pass.type) : access.stages;
                    state.uses.push_back(use);
                    state.useState.push_back(LowerAccess(access.access, stages, !record.isBuffer && (record.hasDepth || record.hasStencil),
                                                         state.profile));
                    state.useStages.push_back((access.access & shaderAccess) != Access::None ? stages : nvrhi::ShaderType::None);
                }
                state.useCount[s] = uint32_t(state.uses.size()) - state.useBegin[s];
            }
        }

        // ------------------------------------------------------------------------------------ phase 4: lifetimes

        void ComputeLifetimes(CompileState& state)
        {
            const uint32_t resourceCount = state.graph.ResourceCount();
            state.first.assign(resourceCount, c_None);
            state.last.assign(resourceCount, c_None);
            state.queueMask.assign(resourceCount, 0);
            state.streamMask.assign(resourceCount, 0);
            for (uint32_t s = 0; s < uint32_t(state.schedule.size()); ++s)
            {
                for (uint32_t u = state.useBegin[s]; u < state.useBegin[s] + state.useCount[s]; ++u)
                {
                    const uint32_t r = state.uses[u].resource;
                    if (state.first[r] == c_None)
                        state.first[r] = s;
                    state.last[r] = s;
                    state.queueMask[r] |= uint8_t(1u << uint32_t(state.schedQueue[s]));
                    state.streamMask[r] |= 1u << state.StreamOfSched(s);
                }
            }
        }

        // ------------------------------------------------------------------------------------ the structure

        // Every non-import resource is its own physical resource: the provisional memory plan of cost estimates.
        void MakeDedicatedPlan(const RenderGraph& graph, StructurePlan& structure)
        {
            const uint32_t resourceCount = graph.ResourceCount();
            structure.resourcePhysical.assign(resourceCount, c_None);
            structure.resourceAliased.assign(resourceCount, 0);
            structure.overlaps.assign(resourceCount, {});
            structure.physicals.clear();
            for (uint32_t r = 0; r < resourceCount; ++r)
            {
                const ResourceRecord& record = graph.GetResource(r);
                if (record.kind == ResourceKind::Imported || record.kind == ResourceKind::HostKept)
                    continue;
                PhysicalObject physical{};
                physical.kind = PhysicalKind::Dedicated;
                physical.firstResource = r;
                physical.heap = c_None;
                uint64_t alignment = 0;
                EstimateResourceSize(record, physical.size, alignment);
                structure.resourcePhysical[r] = uint32_t(structure.physicals.size());
                structure.physicals.push_back(physical);
            }
        }

        // The barrier cost of the superset under a queue assignment: transitions and ownership transfers (6.8).
        uint32_t EstimateBarrierCost(const RenderGraph& graph, const BackendProfile& profile, const CompileOptions& options,
                                     const std::vector<CommandQueue>& passQueues)
        {
            StructurePlan provisional;
            provisional.passQueues = passQueues;
            MakeDedicatedPlan(graph, provisional);
            CompileState state(graph, profile, options);
            state.structure = &provisional;
            state.superset = true;
            state.quiet = true;
            ApplyMask(state);
            BuildSchedule(state, passQueues);
            BuildUses(state);
            ComputeLifetimes(state);
            RunTransitions(state);
            return state.transitionCount * profile.lattice.transition + state.transferCount * profile.lattice.queueTransfer;
        }

        struct PlacementItem
        {
            uint32_t resource;
            uint64_t size;
            uint64_t alignment;
            uint32_t heap;
            uint64_t offset;
        };

        // First-fit placement in a (bytes x lifetime) plane, largest first, ties by declaration index (7.5).
        void PlaceResources(const RenderGraph& graph, const BackendProfile& profile, const StructurePlan& structure,
                            const std::vector<uint32_t>& candidates, const std::vector<uint64_t>& sizes,
                            const std::vector<uint64_t>& alignments, const std::vector<uint32_t>& exGroup,
                            const std::vector<uint32_t>& exPass, bool noAliasing, std::vector<PlacementItem>& placed,
                            std::vector<HeapInfo>& heaps)
        {
            placed.clear();
            heaps.clear();
            std::vector<uint32_t> order = candidates;
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                if (sizes[a] != sizes[b])
                    return sizes[a] > sizes[b];
                return a < b;
            });

            for (uint32_t r : order)
            {
                const ResourceRecord& record = graph.GetResource(r);
                // The queue of its passes: placed resources never cross queues.
                const CommandQueue resourceQueue = structure.resourceFirst[r] == c_None ? CommandQueue::Graphics :
                    structure.passQueues[structure.resourceFirst[r]];

                uint8_t heapClass = 0;
                if (profile.resourceHeapTier < 2)
                    heapClass = record.isBuffer ? 3 : ((record.texture.isRenderTarget || record.hasDepth) ? 1 : 2);

                uint32_t heap = c_None;
                for (uint32_t h = 0; h < uint32_t(heaps.size()); ++h)
                {
                    if (heaps[h].queue == resourceQueue && heaps[h].heapClass == heapClass)
                        heap = h;
                }
                if (heap == c_None)
                {
                    heap = uint32_t(heaps.size());
                    heaps.push_back(HeapInfo{ resourceQueue, heapClass, 0 });
                }

                // The byte ranges of resources in this heap that are live at the same time.
                std::vector<std::pair<uint64_t, uint64_t>> busy;
                for (const PlacementItem& other : placed)
                {
                    if (other.heap != heap)
                        continue;
                    const uint32_t o = other.resource;
                    bool conflict = noAliasing ||
                        !(structure.resourceLast[o] < structure.resourceFirst[r] || structure.resourceLast[r] < structure.resourceFirst[o]);
                    if (conflict && exGroup[r] && exGroup[r] == exGroup[o] && exPass[r] != exPass[o])
                        conflict = false;   // 7.4: members of one exclusion group never run together
                    if (conflict)
                        busy.emplace_back(other.offset, other.offset + other.size);
                }
                std::sort(busy.begin(), busy.end());
                uint64_t offset = 0;
                for (const auto& range : busy)
                {
                    if (offset + sizes[r] <= range.first)
                        break;
                    offset = std::max(offset, (range.second + alignments[r] - 1) & ~(alignments[r] - 1));
                }
                placed.push_back(PlacementItem{ r, sizes[r], alignments[r], heap, offset });
                heaps[heap].size = std::max(heaps[heap].size, offset + sizes[r]);
            }
        }

        // Queue lanes (BackendProfile::queueLanes, nvrhi ADR 0016). A pass of a type with more than one lane takes the
        // lane of its latest producer of that type, so that a chain stays on one stream; a pass that depends on nothing
        // of its type starts on the lane with the fewest passes so far. The dependencies are the superset's, by
        // resource: the last writer, and for a write the readers since it. Empty when every type has one lane.
        std::vector<uint8_t> AssignLanes(const RenderGraph& graph, const BackendProfile& profile,
                                         const std::vector<CommandQueue>& passQueues)
        {
            std::vector<uint8_t> lanes;
            std::array<uint32_t, 3> laneCount;
            bool any = false;
            for (uint32_t q = 0; q < 3; ++q)
            {
                laneCount[q] = std::min<uint32_t>(std::max<uint32_t>(profile.queueLanes[q], 1u), c_MaxQueueLanes);
                any = any || laneCount[q] > 1;
            }
            if (!any)
                return lanes;

            const uint32_t passCount = graph.PassCount();
            lanes.assign(passCount, 0);
            std::array<std::array<uint32_t, c_MaxQueueLanes>, 3> load = {};
            std::vector<uint32_t> lastWriter(graph.ResourceCount(), c_None);
            std::vector<std::vector<uint32_t>> readers(graph.ResourceCount());
            for (uint32_t p = 0; p < passCount; ++p)
            {
                const PassRecord& pass = graph.GetPass(p);
                const uint32_t q = uint32_t(passQueues[p]);
                uint32_t latest = c_None;
                auto consider = [&](uint32_t producer) {
                    if (producer != c_None && passQueues[producer] == passQueues[p] && (latest == c_None || producer > latest))
                        latest = producer;
                };
                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    const AccessRecord& access = graph.GetAccess(a);
                    consider(lastWriter[access.resource]);
                    if (access.isWrite)
                    {
                        for (uint32_t reader : readers[access.resource])
                            consider(reader);
                    }
                }
                uint32_t lane = 0;
                if (latest != c_None)
                    lane = lanes[latest];
                else
                {
                    for (uint32_t l = 1; l < laneCount[q]; ++l)
                    {
                        if (load[q][l] < load[q][lane])
                            lane = l;
                    }
                }
                lanes[p] = uint8_t(lane);
                ++load[q][lane];
                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    const AccessRecord& access = graph.GetAccess(a);
                    if (access.isWrite)
                    {
                        lastWriter[access.resource] = p;
                        readers[access.resource].clear();
                    }
                    else if (std::find(readers[access.resource].begin(), readers[access.resource].end(), p) == readers[access.resource].end())
                        readers[access.resource].push_back(p);
                }
            }
            return lanes;
        }

        std::shared_ptr<StructurePlan> BuildStructure(const RenderGraph& graph, const BackendProfile& profile,
                                                      const CompileOptions& options, uint64_t key, std::string& diagnostics,
                                                      bool& failed)
        {
            auto structure = std::make_shared<StructurePlan>();
            structure->key = key;
            const uint32_t passCount = graph.PassCount();
            const uint32_t resourceCount = graph.ResourceCount();
            const bool serialize = (options.debugFlags & DebugFlags::SerializeQueues) != DebugFlags::None;

            // Queue assignment: async passes are declared, never inferred (6.8).
            structure->passQueues.assign(passCount, CommandQueue::Graphics);
            structure->demoted.assign(passCount, 0);
            for (uint32_t p = 0; p < passCount; ++p)
            {
                const PassRecord& pass = graph.GetPass(p);
                if ((pass.flags & PassFlags::AsyncCompute) == PassFlags::None || pass.type == PassType::Raster)
                    continue;
                if (serialize)
                    continue;
                if (!profile.computeQueue)
                {
                    structure->demoted[p] = 1;
                    structure->warnings.push_back("pass '" + std::string(pass.name ? pass.name : "") + "' (" +
                        std::to_string(p) + ") runs on the graphics queue: the profile has no compute queue (6.8)");
                    continue;
                }
                structure->passQueues[p] = CommandQueue::Compute;
            }

            // States and stages that are illegal on a pass's queue (6.8).
            CompileState check(graph, profile, options);
            check.diagnostics = &diagnostics;
            for (uint32_t p = 0; p < passCount; ++p)
            {
                if (structure->passQueues[p] == CommandQueue::Graphics)
                    continue;
                const PassRecord& pass = graph.GetPass(p);
                for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                {
                    const AccessRecord& access = graph.GetAccess(a);
                    const nvrhi::ShaderType stages = access.stages == nvrhi::ShaderType::None ? DefaultStages(pass.type) : access.stages;
                    const ResourceRecord& accessed = graph.GetResource(access.resource);
                    const ResourceStates state = LowerAccess(access.access, stages,
                        !accessed.isBuffer && (accessed.hasDepth || accessed.hasStencil), profile);
                    const bool graphicsStages = (stages & (nvrhi::ShaderType::AllGraphics)) != 0;
                    if (!IsStateLegalOnQueue(state, structure->passQueues[p], profile) || graphicsStages)
                        check.Error(check.PassName(p) + ": " + AccessToString(access.access) + " on " +
                            check.ResourceName(access.resource) + " is illegal on the " +
                            QueueName(structure->passQueues[p]) + " queue (6.8)");
                }
            }
            if (check.failed)
            {
                failed = true;
                return nullptr;
            }

            // LatticePolicy (6.8): an async pass whose extra barriers cost more than its overlap gains is demoted.
            if (profile.barrierModel != BarrierModel::Implicit)
            {
                for (uint32_t p = 0; p < passCount; ++p)
                {
                    if (structure->passQueues[p] != CommandQueue::Compute)
                        continue;
                    const uint32_t withAsync = EstimateBarrierCost(graph, profile, options, structure->passQueues);
                    std::vector<CommandQueue> demoted = structure->passQueues;
                    demoted[p] = CommandQueue::Graphics;
                    const uint32_t without = EstimateBarrierCost(graph, profile, options, demoted);
                    if (withAsync > without + profile.lattice.asyncOverlapGain)
                    {
                        structure->passQueues = demoted;
                        structure->demoted[p] = 1;
                        const PassRecord& pass = graph.GetPass(p);
                        structure->warnings.push_back("pass '" + std::string(pass.name ? pass.name : "") + "' (" +
                            std::to_string(p) + ") runs on the graphics queue: its barriers cost " +
                            std::to_string(withAsync - without) + ", more than the overlap it gains (" +
                            std::to_string(profile.lattice.asyncOverlapGain) + ", LatticePolicy, 6.8)");
                    }
                }
            }

            // Queue lanes (nvrhi ADR 0016), on the final queues. SerializeQueues keeps every pass on lane 0.
            if (!serialize)
                structure->passLanes = AssignLanes(graph, profile, structure->passQueues);

            // Superset lifetimes: every pass enabled. The schedule is program order, so schedule indices of the
            // superset are program indices of its live passes.
            {
                StructurePlan provisional;
                provisional.passQueues = structure->passQueues;
                provisional.passLanes = structure->passLanes;
                MakeDedicatedPlan(graph, provisional);
                CompileState superset(graph, profile, options);
                superset.structure = &provisional;
                superset.superset = true;
                superset.quiet = true;
                ApplyMask(superset);
                BuildSchedule(superset, structure->passQueues);
                BuildUses(superset);
                ComputeLifetimes(superset);

                structure->resourceFirst.assign(resourceCount, c_None);
                structure->resourceLast.assign(resourceCount, c_None);
                // The streams of each resource: a resource used on more than one (queue, lane) crosses queues (7.4).
                std::vector<uint32_t> streamMask(resourceCount, 0);
                for (uint32_t r = 0; r < resourceCount; ++r)
                {
                    if (superset.first[r] != c_None)
                    {
                        structure->resourceFirst[r] = superset.schedule[superset.first[r]];
                        structure->resourceLast[r] = superset.schedule[superset.last[r]];
                    }
                    streamMask[r] = superset.streamMask[r];
                }

                // A substitute or passthrough input takes over the output's readers when the pass is off (13.5).
                for (uint32_t i = 0; i < graph.PolicyCount(); ++i)
                {
                    const PolicyRecord& policy = graph.GetPolicy(i);
                    if (policy.policy == DisabledOutput::Cull || policy.substitute == c_None)
                        continue;
                    for (uint32_t a = 0; a < graph.AccessCount(); ++a)
                    {
                        const AccessRecord& access = graph.GetAccess(a);
                        if (access.isWrite || access.resource != policy.resource || access.version != policy.version)
                            continue;
                        uint32_t pass = 0;
                        while (pass + 1 < passCount && graph.GetPass(pass + 1).firstAccess <= a)
                            ++pass;
                        const uint32_t s = policy.substitute;
                        structure->resourceFirst[s] = std::min(structure->resourceFirst[s], pass);
                        structure->resourceLast[s] = structure->resourceLast[s] == c_None ? pass : std::max(structure->resourceLast[s], pass);
                        streamMask[s] |= 1u << StreamOf(structure->passQueues[pass],
                            structure->passLanes.empty() ? 0u : structure->passLanes[pass]);
                    }
                }

                structure->resourceCrossQueue.assign(resourceCount, 0);
                for (uint32_t r = 0; r < resourceCount; ++r)
                {
                    const uint32_t mask = streamMask[r];
                    structure->resourceCrossQueue[r] = (mask & (mask - 1)) != 0 ? 1 : 0;
                }
            }

            // Exclusion-group membership of each resource's users (7.4).
            std::vector<uint32_t> exGroup(resourceCount, 0);
            std::vector<uint32_t> exPass(resourceCount, c_None);
            {
                std::vector<uint8_t> mixed(resourceCount, 0);
                for (uint32_t p = 0; p < passCount; ++p)
                {
                    const PassRecord& pass = graph.GetPass(p);
                    for (uint32_t a = pass.firstAccess; a < pass.firstAccess + pass.accessCount; ++a)
                    {
                        const uint32_t r = graph.GetAccess(a).resource;
                        if (mixed[r])
                            continue;
                        if (exPass[r] == c_None)
                        {
                            exPass[r] = p;
                            exGroup[r] = pass.exclusionGroup;
                        }
                        else if (exPass[r] != p)
                        {
                            mixed[r] = 1;
                            exGroup[r] = 0;
                        }
                    }
                }
            }

            // The memory plan (7.5).
            const bool forceDedicated = (options.debugFlags & DebugFlags::ForceDedicated) != DebugFlags::None;
            const bool noAliasing = (options.debugFlags & DebugFlags::NoAliasing) != DebugFlags::None;
            structure->tier = (options.placedResources && profile.maxMemoryTier == MemoryTier::Placed && !forceDedicated)
                ? MemoryTier::Placed : MemoryTier::Pooled;

            std::vector<uint64_t> sizes(resourceCount, 0);
            std::vector<uint64_t> alignments(resourceCount, 1);
            std::vector<uint32_t> candidates;
            for (uint32_t r = 0; r < resourceCount; ++r)
            {
                const ResourceRecord& record = graph.GetResource(r);
                EstimateResourceSize(record, sizes[r], alignments[r]);
                if (!IsTransientKind(record) || record.extracted)
                    continue;
                if (structure->resourceFirst[r] == c_None)
                    continue;
                structure->poolableTransientBytes += sizes[r];
                if ((record.flags & (ResourceFlags::NoAlias | ResourceFlags::DebugCapture)) != ResourceFlags::None ||
                    structure->resourceCrossQueue[r])
                {
                    structure->unplacedTransientBytes += sizes[r];
                    continue;
                }
                candidates.push_back(r);
            }

            // The placement estimate, on every backend.
            std::vector<PlacementItem> placed;
            std::vector<HeapInfo> heaps;
            PlaceResources(graph, profile, *structure, candidates, sizes, alignments, exGroup, exPass, false, placed, heaps);
            for (const HeapInfo& heap : heaps)
                structure->placedPeakBytes += heap.size;

            MakeDedicatedPlan(graph, *structure);
            if (forceDedicated)
                return structure;

            if (structure->tier == MemoryTier::Placed)
            {
                if (noAliasing)
                    PlaceResources(graph, profile, *structure, candidates, sizes, alignments, exGroup, exPass, true, placed, heaps);
                structure->heaps = heaps;
                for (const PlacementItem& item : placed)
                {
                    PhysicalObject& physical = structure->physicals[structure->resourcePhysical[item.resource]];
                    physical.kind = PhysicalKind::Placed;
                    physical.heap = item.heap;
                    physical.offset = item.offset;
                    physical.size = item.size;
                }
                for (size_t i = 0; i < placed.size(); ++i)
                {
                    for (size_t j = i + 1; j < placed.size(); ++j)
                    {
                        const PlacementItem& a = placed[i];
                        const PlacementItem& b = placed[j];
                        if (a.heap != b.heap || a.offset + a.size <= b.offset || b.offset + b.size <= a.offset)
                            continue;
                        structure->overlaps[a.resource].push_back(b.resource);
                        structure->overlaps[b.resource].push_back(a.resource);
                    }
                }
                for (uint32_t r = 0; r < resourceCount; ++r)
                {
                    std::sort(structure->overlaps[r].begin(), structure->overlaps[r].end());
                    structure->resourceAliased[r] = structure->overlaps[r].empty() ? 0 : 1;
                }
                return structure;
            }

            // Pooled: whole-resource reuse of equal descriptions with disjoint lifetimes, on one queue.
            struct Slot
            {
                bool isBuffer;
                uint64_t descHash;
                uint32_t stream;
                uint32_t last;
                uint32_t physical;
            };
            std::vector<Slot> slots;
            std::vector<uint32_t> order = candidates;
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                return structure->resourceFirst[a] < structure->resourceFirst[b];
            });
            for (uint32_t r : order)
            {
                const ResourceRecord& record = graph.GetResource(r);
                const uint32_t first = structure->resourceFirst[r];
                const uint32_t stream = StreamOf(structure->passQueues[first], structure->passLanes.empty() ? 0u : structure->passLanes[first]);
                Slot* chosen = nullptr;
                for (Slot& slot : slots)
                {
                    if (slot.isBuffer == record.isBuffer && slot.descHash == record.descHash && slot.stream == stream &&
                        slot.last < structure->resourceFirst[r])
                    {
                        chosen = &slot;
                        break;
                    }
                }
                if (chosen)
                {
                    structure->resourcePhysical[r] = chosen->physical;
                    structure->physicals[chosen->physical].kind = PhysicalKind::PooledSlot;
                    chosen->last = structure->resourceLast[r];
                }
                else
                {
                    structure->physicals[structure->resourcePhysical[r]].kind = PhysicalKind::PooledSlot;
                    slots.push_back(Slot{ record.isBuffer, record.descHash, stream, structure->resourceLast[r],
                        structure->resourcePhysical[r] });
                }
            }
            return structure;
        }

        // ------------------------------------------------------------------------------------ phase 7: segments

        void BuildSegments(const CompileState& state, CompiledRenderGraph& plan, std::vector<uint32_t>& segOf,
                           std::vector<uint8_t>& splitJoined)
        {
            const uint32_t count = uint32_t(state.schedule.size());
            segOf.assign(count, c_None);

            // Segments, waits and what each queue has observed are per stream: a (queue, lane) pair.
            std::vector<uint8_t> isProducer(count, 0);
            std::vector<std::vector<const WaitEdge*>> waitsOf(count);
            for (const WaitEdge& wait : state.waits)
            {
                waitsOf[wait.waiter].push_back(&wait);
                if (wait.frameDelta == 0 && state.StreamOfSched(wait.producer) != state.StreamOfSched(wait.waiter))
                    isProducer[wait.producer] = 1;
            }

            // known[q][q2]: the latest segment of stream q2 whose completion stream q has observed this frame.
            std::array<std::array<int32_t, c_MaxStreams>, c_MaxStreams> known;
            for (auto& row : known)
                row.fill(-1);
            std::vector<std::array<int32_t, c_MaxStreams>> segKnown;
            struct PendingWait { uint32_t segment; uint32_t stream; uint32_t producer; };
            std::vector<PendingWait> pending;
            std::vector<std::vector<SegmentWait>> segmentWaits;

            int32_t current = -1;
            uint32_t previousLive = c_None;
            const uint32_t granularity = state.options.recordGranularity;

            for (uint32_t s = 0; s < count; ++s)
            {
                if (!state.scheduledLive[s])
                    continue;
                const CommandQueue queue = state.schedQueue[s];
                const uint32_t q = state.StreamOfSched(s);

                // This frame's waits that the stream has not observed yet.
                std::array<int32_t, c_MaxStreams> needed;
                needed.fill(-1);
                bool need = false;
                for (const WaitEdge* wait : waitsOf[s])
                {
                    if (wait->frameDelta != 0)
                        continue;
                    const uint32_t pq = state.StreamOfSched(wait->producer);
                    if (pq == q)
                        continue;
                    const int32_t producerSegment = int32_t(segOf[wait->producer]);
                    if (producerSegment > known[q][pq])
                    {
                        needed[pq] = std::max(needed[pq], producerSegment);
                        need = true;
                    }
                }

                const bool newSegment = current < 0 ||
                    StreamOf(plan.m_Segments[current].queue, plan.m_Segments[current].lane) != q ||
                    (granularity && plan.m_Segments[current].passCount >= granularity) ||
                    (previousLive != c_None && isProducer[previousLive]) || need;
                if (newSegment)
                {
                    current = int32_t(plan.m_Segments.size());
                    PlanSegment segment{};
                    segment.queue = queue;
                    segment.lane = state.schedLane[s];
                    segment.firstPass = s;
                    plan.m_Segments.push_back(segment);
                    segmentWaits.emplace_back();
                    segKnown.push_back(known[q]);
                }
                for (uint32_t pq = 0; pq < c_MaxStreams; ++pq)
                {
                    if (needed[pq] < 0)
                        continue;
                    segmentWaits[current].push_back(SegmentWait{ QueueOfStream(pq), uint32_t(needed[pq]), 0, uint8_t(LaneOfStream(pq)) });
                    // What the producer's stream had observed comes with it.
                    const auto& theirs = segKnown[needed[pq]];
                    for (uint32_t k = 0; k < c_MaxStreams; ++k)
                        known[q][k] = std::max(known[q][k], theirs[k]);
                    known[q][pq] = std::max(known[q][pq], needed[pq]);
                }
                known[q][q] = current;
                segKnown[current] = known[q];

                for (const WaitEdge* wait : waitsOf[s])
                {
                    if (wait->frameDelta == 0 || state.StreamOfSched(wait->producer) == q)
                        continue;
                    pending.push_back(PendingWait{ uint32_t(current), state.StreamOfSched(wait->producer), wait->producer });
                }

                ++plan.m_Segments[current].passCount;
                segOf[s] = uint32_t(current);
                previousLive = s;
            }

            // Previous-frame waits (6.12): the producer's segment of the previous frame; one per stream, the latest.
            for (const PendingWait& wait : pending)
            {
                const uint32_t producerSegment = segOf[wait.producer];
                if (producerSegment == c_None)
                    continue;
                bool merged = false;
                for (SegmentWait& existing : segmentWaits[wait.segment])
                {
                    if (StreamOf(existing.queue, existing.lane) == wait.stream && existing.frameDelta == -1)
                    {
                        existing.segment = std::max(existing.segment, producerSegment);
                        merged = true;
                    }
                }
                if (!merged)
                    segmentWaits[wait.segment].push_back(SegmentWait{ QueueOfStream(wait.stream), producerSegment, -1,
                        uint8_t(LaneOfStream(wait.stream)) });
            }

            for (uint32_t i = 0; i < uint32_t(plan.m_Segments.size()); ++i)
            {
                PlanSegment& segment = plan.m_Segments[i];
                segment.waitBegin = uint32_t(plan.m_SegmentWaits.size());
                segment.waitCount = uint32_t(segmentWaits[i].size());
                for (const SegmentWait& wait : segmentWaits[i])
                {
                    plan.m_SegmentWaits.push_back(wait);
                    plan.m_Segments[wait.segment].signals = true;
                }
            }

            // 6.4: a split never crosses a command-list boundary.
            splitJoined.assign(state.splits.size(), 0);
            for (uint32_t i = 0; i < uint32_t(state.splits.size()); ++i)
            {
                const SplitPair& pair = state.splits[i];
                if (segOf[pair.begin] != segOf[pair.end])
                    splitJoined[i] = 1;
            }
        }

        // ------------------------------------------------------------------------------------ the plan

        struct UnitRow
        {
            uint8_t plane;
            uint32_t mip0;
            uint32_t mip1;
            uint32_t slice0;
            uint32_t slice1;
        };

        // 6.10: regions merge only when equal or adjacent with equal planes, never into a bounding box; a region
        // that covers the whole resource becomes the default region.
        void MergeUnits(const ResourceRecord& record, const BackendProfile& profile, std::vector<UnitKey>& units,
                        std::vector<nvrhi::SubresourceRegion>& regions)
        {
            regions.clear();
            for (const UnitKey& unit : units)
            {
                if (unit.plane == c_WholeUnit)
                {
                    regions.push_back(nvrhi::SubresourceRegion());
                    return;
                }
            }
            const uint32_t planeCount = UnitPlaneCount(record, profile);
            const uint32_t mips = MipCount(record);
            const uint32_t slices = SliceCount(record);

            std::sort(units.begin(), units.end(), [](const UnitKey& a, const UnitKey& b) {
                if (a.plane != b.plane) return a.plane < b.plane;
                if (a.mip != b.mip) return a.mip < b.mip;
                return a.slice0 < b.slice0;
            });

            // Slices within one plane and mip.
            std::vector<UnitRow> rows;
            for (const UnitKey& unit : units)
            {
                if (!rows.empty() && rows.back().plane == unit.plane && rows.back().mip0 == unit.mip &&
                    rows.back().slice1 == unit.slice0)
                    rows.back().slice1 = unit.slice1;
                else
                    rows.push_back(UnitRow{ unit.plane, unit.mip, unit.mip + 1, unit.slice0, unit.slice1 });
            }
            // Mips with equal slice ranges.
            std::sort(rows.begin(), rows.end(), [](const UnitRow& a, const UnitRow& b) {
                if (a.plane != b.plane) return a.plane < b.plane;
                if (a.slice0 != b.slice0) return a.slice0 < b.slice0;
                if (a.slice1 != b.slice1) return a.slice1 < b.slice1;
                return a.mip0 < b.mip0;
            });
            std::vector<UnitRow> blocks;
            for (const UnitRow& row : rows)
            {
                if (!blocks.empty() && blocks.back().plane == row.plane && blocks.back().slice0 == row.slice0 &&
                    blocks.back().slice1 == row.slice1 && blocks.back().mip1 == row.mip0)
                    blocks.back().mip1 = row.mip1;
                else
                    blocks.push_back(row);
            }
            // Planes with equal ranges.
            std::vector<std::pair<UnitRow, uint32_t>> merged;   // block, plane bits
            for (const UnitRow& block : blocks)
            {
                bool found = false;
                for (auto& entry : merged)
                {
                    if (entry.first.mip0 == block.mip0 && entry.first.mip1 == block.mip1 &&
                        entry.first.slice0 == block.slice0 && entry.first.slice1 == block.slice1)
                    {
                        entry.second |= 1u << block.plane;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    merged.emplace_back(block, 1u << block.plane);
            }
            const uint32_t allPlanes = planeCount == 2 ? 3u : 1u;
            for (const auto& [block, planeBits] : merged)
            {
                nvrhi::SubresourceRegion region;
                const bool wholeRange = block.mip0 == 0 && block.mip1 == mips && block.slice0 == 0 && block.slice1 == slices;
                if (wholeRange && planeBits == allPlanes)
                {
                    regions.push_back(region);
                    continue;
                }
                region.baseMip = block.mip0;
                region.mipCount = block.mip1 - block.mip0;
                region.baseSlice = block.slice0;
                region.sliceCount = block.slice1 - block.slice0;
                if (planeCount == 1 || planeBits == allPlanes)
                    region.planes = nvrhi::PlaneMask::All;
                else
                    region.planes = (planeBits & 1) ? nvrhi::PlaneMask::Depth : nvrhi::PlaneMask::Stencil;
                regions.push_back(region);
            }
        }

        uint32_t TypeOrder(nvrhi::BarrierType type) noexcept
        {
            // 6.10: aliasing barriers first, then transitions, then UAV barriers.
            switch (type)
            {
            case nvrhi::BarrierType::Aliasing:   return 0;
            case nvrhi::BarrierType::Transition: return 1;
            default:                             return 2;
            }
        }

        bool SameBarrierKey(const EmittedBarrier& a, const EmittedBarrier& b) noexcept
        {
            return a.sched == b.sched && a.slot == b.slot && a.queue == b.queue && a.barrier.type == b.barrier.type &&
                a.barrier.flags == b.barrier.flags && a.barrier.reason == b.barrier.reason &&
                a.barrier.otherQueue == b.barrier.otherQueue && a.barrier.resource == b.barrier.resource &&
                a.barrier.aliasBefore == b.barrier.aliasBefore && a.barrier.before == b.barrier.before &&
                a.barrier.after == b.barrier.after && a.barrier.stagesBefore == b.barrier.stagesBefore &&
                a.barrier.stagesAfter == b.barrier.stagesAfter && a.pairId == b.pairId;
        }

        CompiledRenderGraph* BuildPlan(CompileState& state, std::shared_ptr<const StructurePlan> structure, uint64_t key,
                                       uint64_t maskHash)
        {
            const RenderGraph& graph = state.graph;
            nvrhi::AutoPtr<CompiledRenderGraph> plan = nvrhi::TakeOver(MAKE_RC_OBJ(CompiledRenderGraph));
            plan->m_Key = key;
            plan->m_StructuralHash = graph.StructuralHashValue();
            plan->m_MaskHash = maskHash;
            plan->m_ProfileHash = HashProfile(state.profile);
            plan->m_Profile = state.profile;
            plan->m_Options = state.options;
            plan->m_Structure = structure;
            plan->m_ProgramPassCount = graph.PassCount();
            plan->m_AccessCount = graph.AccessCount();
            plan->m_Names.push_back('\0');

            // Segments first: they decide which splits stay split.
            std::vector<uint32_t> segOf;
            std::vector<uint8_t> splitJoined;
            BuildSegments(state, *plan, segOf, splitJoined);

            // Resources.
            const uint32_t resourceCount = graph.ResourceCount();
            plan->m_Resources.resize(resourceCount);
            for (uint32_t r = 0; r < resourceCount; ++r)
            {
                const ResourceRecord& record = graph.GetResource(r);
                PlanResource& resource = plan->m_Resources[r];
                resource.nameOffset = record.name ? plan->AddName(record.name) : 0;
                resource.kind = record.kind;
                resource.isBuffer = record.isBuffer;
                resource.hasDepth = record.hasDepth;
                resource.hasStencil = record.hasStencil;
                resource.isRenderTarget = !record.isBuffer && record.texture.isRenderTarget;
                resource.extracted = record.extracted;
                resource.standIn = record.standIn;
                resource.crossQueue = structure->resourceCrossQueue[r] != 0;
                resource.aliased = structure->resourceAliased[r] != 0;
                resource.physical = structure->resourcePhysical[r];
                resource.firstPass = state.first[r];
                resource.lastPass = state.last[r];
                resource.heap = c_None;
                if (resource.physical != c_None)
                {
                    const PhysicalObject& physical = structure->physicals[resource.physical];
                    resource.size = physical.size;
                    if (physical.kind == PhysicalKind::Placed)
                    {
                        resource.heap = physical.heap;
                        resource.heapOffset = physical.offset;
                    }
                }
                else
                {
                    uint64_t alignment = 0;
                    EstimateResourceSize(record, resource.size, alignment);
                }
                resource.registryEntry = record.registryEntry;
                resource.historyLayer = record.historyLayer;
                resource.historyLayerCount = record.historyLayerCount;
                resource.mipLevels = MipCount(record);
                resource.arraySlices = SliceCount(record);
                resource.entry = state.entryState.empty() ? record.entry : state.entryState[r];
                resource.exit = record.exit;
                resource.creation = resource.entry;
                resource.descHash = record.descHash;
            }

            // Frame-start states, one region per (resource, state).
            {
                std::vector<CompileState::EntryUnit>& entries = state.entryUnits;
                std::stable_sort(entries.begin(), entries.end(), [](const CompileState::EntryUnit& a, const CompileState::EntryUnit& b) {
                    if (a.resource != b.resource) return a.resource < b.resource;
                    return uint32_t(a.state) < uint32_t(b.state);
                });
                std::vector<UnitKey> groupUnits;
                std::vector<nvrhi::SubresourceRegion> groupRegions;
                size_t first = 0;
                while (first < entries.size())
                {
                    size_t end = first;
                    groupUnits.clear();
                    while (end < entries.size() && entries[end].resource == entries[first].resource &&
                        entries[end].state == entries[first].state)
                        groupUnits.push_back(entries[end++].unit);
                    MergeUnits(graph.GetResource(entries[first].resource), state.profile, groupUnits, groupRegions);
                    for (const nvrhi::SubresourceRegion& region : groupRegions)
                        plan->m_EntryStates.push_back(PlanEntryState{ entries[first].resource, region, entries[first].state });
                    first = end;
                }
            }

            // Barriers: drop the begin halves of joined splits, then sort and merge (6.10).
            for (EmittedBarrier& emitted : state.emitted)
            {
                if (emitted.pairId == c_None || !splitJoined[emitted.pairId])
                    continue;
                if ((emitted.barrier.flags & nvrhi::BarrierFlags::SplitBegin) != 0)
                    emitted.sched = c_None;   // dropped
                else
                    emitted.barrier.flags = emitted.barrier.flags & ~nvrhi::BarrierFlags::SplitEnd;
                emitted.pairId = c_None;
            }
            std::vector<uint32_t> order;
            order.reserve(state.emitted.size());
            for (uint32_t i = 0; i < uint32_t(state.emitted.size()); ++i)
            {
                if (state.emitted[i].sched != c_None)
                    order.push_back(i);
            }
            std::stable_sort(order.begin(), order.end(), [&](uint32_t ia, uint32_t ib) {
                const EmittedBarrier& a = state.emitted[ia];
                const EmittedBarrier& b = state.emitted[ib];
                if (a.sched != b.sched) return a.sched < b.sched;
                if (a.slot != b.slot) return a.slot < b.slot;
                if (TypeOrder(a.barrier.type) != TypeOrder(b.barrier.type)) return TypeOrder(a.barrier.type) < TypeOrder(b.barrier.type);
                if (a.barrier.resource != b.barrier.resource) return a.barrier.resource < b.barrier.resource;
                if (a.barrier.flags != b.barrier.flags) return uint32_t(a.barrier.flags) < uint32_t(b.barrier.flags);
                if (a.barrier.before != b.barrier.before) return uint32_t(a.barrier.before) < uint32_t(b.barrier.before);
                if (a.barrier.after != b.barrier.after) return uint32_t(a.barrier.after) < uint32_t(b.barrier.after);
                if (a.barrier.reason != b.barrier.reason) return a.barrier.reason < b.barrier.reason;
                if (a.barrier.otherQueue != b.barrier.otherQueue) return a.barrier.otherQueue < b.barrier.otherQueue;
                if (a.barrier.stagesBefore != b.barrier.stagesBefore) return uint32_t(a.barrier.stagesBefore) < uint32_t(b.barrier.stagesBefore);
                if (a.barrier.stagesAfter != b.barrier.stagesAfter) return uint32_t(a.barrier.stagesAfter) < uint32_t(b.barrier.stagesAfter);
                if (a.barrier.aliasBefore != b.barrier.aliasBefore) return a.barrier.aliasBefore < b.barrier.aliasBefore;
                return a.pairId < b.pairId;
            });

            const uint32_t schedCount = uint32_t(state.schedule.size());
            std::vector<std::array<uint32_t, c_SlotCount>> batchBegin(schedCount);
            std::vector<std::array<uint32_t, c_SlotCount>> batchCount(schedCount);
            for (uint32_t s = 0; s < schedCount; ++s)
            {
                batchBegin[s].fill(0);
                batchCount[s].fill(0);
            }

            std::vector<UnitKey> units;
            std::vector<nvrhi::SubresourceRegion> regions;
            size_t i = 0;
            while (i < order.size())
            {
                size_t j = i + 1;
                while (j < order.size() && SameBarrierKey(state.emitted[order[i]], state.emitted[order[j]]))
                    ++j;
                const EmittedBarrier& head = state.emitted[order[i]];
                std::array<uint32_t, c_SlotCount>& begins = batchBegin[head.sched];
                std::array<uint32_t, c_SlotCount>& counts = batchCount[head.sched];
                if (counts[head.slot] == 0)
                    begins[head.slot] = uint32_t(plan->m_Barriers.size());

                if (head.barrier.type == nvrhi::BarrierType::Transition)
                {
                    units.clear();
                    for (size_t k = i; k < j; ++k)
                        units.push_back(state.emitted[order[k]].unit);
                    MergeUnits(graph.GetResource(head.barrier.resource), state.profile, units, regions);
                    for (const nvrhi::SubresourceRegion& region : regions)
                    {
                        PlanBarrier barrier = head.barrier;
                        barrier.region = region;
                        plan->m_Barriers.push_back(barrier);
                        ++counts[head.slot];
                    }
                }
                else
                {
                    plan->m_Barriers.push_back(head.barrier);
                    ++counts[head.slot];
                }
                i = j;
            }

            // Passes.
            std::vector<std::vector<PassWait>> passWaits(schedCount);
            for (const WaitEdge& wait : state.waits)
            {
                auto& list = passWaits[wait.waiter];
                bool duplicate = false;
                for (const PassWait& existing : list)
                    duplicate = duplicate || (existing.producer == wait.producer && existing.frameDelta == wait.frameDelta);
                if (!duplicate)
                    list.push_back(PassWait{ wait.producer, wait.frameDelta });
            }

            plan->m_ScheduleOfProgram.assign(graph.PassCount(), c_None);
            plan->m_Passes.resize(schedCount);
            for (uint32_t s = 0; s < schedCount; ++s)
            {
                const uint32_t program = state.schedule[s];
                const PassRecord& record = graph.GetPass(program);
                PlanPass& pass = plan->m_Passes[s];
                pass.program = program;
                pass.nameOffset = record.name ? plan->AddName(record.name) : 0;
                pass.queue = state.schedQueue[s];
                pass.lane = state.schedLane[s];
                pass.type = record.type;
                pass.flags = record.flags;
                pass.enabled = state.scheduledLive[s] != 0;
                pass.segment = segOf[s];
                for (uint32_t slot = 0; slot < c_SlotCount; ++slot)
                {
                    pass.batchBegin[slot] = batchBegin[s][slot];
                    pass.batchCount[slot] = batchCount[s][slot];
                }
                pass.waitBegin = uint32_t(plan->m_PassWaits.size());
                pass.waitCount = uint32_t(passWaits[s].size());
                plan->m_PassWaits.insert(plan->m_PassWaits.end(), passWaits[s].begin(), passWaits[s].end());
                pass.useBegin = uint32_t(plan->m_Uses.size());
                pass.useCount = state.useCount[s];
                pass.trackedBegin = 0;
                pass.trackedCount = 0;
                for (uint32_t u = state.useBegin[s]; u < state.useBegin[s] + state.useCount[s]; ++u)
                    plan->m_Uses.push_back(state.uses[u]);
                pass.clearMask = state.clearMask.empty() ? 0 : state.clearMask[s];
                for (uint32_t a = 0; a < 9; ++a)
                {
                    pass.load[a] = state.load.empty() ? LoadOp::Load : state.load[s][a];
                    pass.store[a] = state.store.empty() ? StoreOp::Store : state.store[s][a];
                }
                plan->m_ScheduleOfProgram[program] = s;
            }

            // The states Tracked and Restore bodies find, per pass, one region per (resource, state).
            {
                std::vector<CompileState::TrackedUnit>& tracked = state.trackedUnits;
                std::stable_sort(tracked.begin(), tracked.end(), [](const CompileState::TrackedUnit& a, const CompileState::TrackedUnit& b) {
                    if (a.sched != b.sched) return a.sched < b.sched;
                    if (a.resource != b.resource) return a.resource < b.resource;
                    return uint32_t(a.state) < uint32_t(b.state);
                });
                std::vector<UnitKey> groupUnits;
                std::vector<nvrhi::SubresourceRegion> groupRegions;
                size_t first = 0;
                while (first < tracked.size())
                {
                    size_t end = first;
                    groupUnits.clear();
                    while (end < tracked.size() && tracked[end].sched == tracked[first].sched &&
                        tracked[end].resource == tracked[first].resource && tracked[end].state == tracked[first].state)
                        groupUnits.push_back(tracked[end++].unit);
                    PlanPass& pass = plan->m_Passes[tracked[first].sched];
                    if (pass.trackedCount == 0)
                        pass.trackedBegin = uint32_t(plan->m_TrackedStates.size());
                    MergeUnits(graph.GetResource(tracked[first].resource), state.profile, groupUnits, groupRegions);
                    for (const nvrhi::SubresourceRegion& region : groupRegions)
                    {
                        plan->m_TrackedStates.push_back(PlanTrackedState{ tracked[first].resource, region, tracked[first].state });
                        ++pass.trackedCount;
                    }
                    first = end;
                }
            }

            // Statistics (10.3, 11.3).
            PlanStatistics& stats = plan->m_Statistics;
            stats = PlanStatistics();
            for (uint32_t s = 0; s < schedCount; ++s)
                stats.scheduledPasses += state.scheduledLive[s] ? 1 : 0;
            for (uint32_t p = 0; p < graph.PassCount(); ++p)
                stats.culledPasses += (state.enabled[p] && !state.live[p]) ? 1 : 0;
            stats.segments = uint32_t(plan->m_Segments.size());
            stats.barriers = uint32_t(plan->m_Barriers.size());
            for (const PlanBarrier& barrier : plan->m_Barriers)
            {
                switch (barrier.type)
                {
                case nvrhi::BarrierType::Transition:
                    if ((barrier.flags & nvrhi::BarrierFlags::SplitBegin) != 0)
                        ++stats.splitBarriers;
                    else
                        ++stats.transitions;
                    if ((barrier.flags & nvrhi::BarrierFlags::QueueRelease) != 0)
                        ++stats.queueTransfers;
                    break;
                case nvrhi::BarrierType::Uav:
                    ++stats.uavBarriers;
                    break;
                case nvrhi::BarrierType::Aliasing:
                    ++stats.aliasingBarriers;
                    break;
                }
            }
            stats.fenceWaits = uint32_t(plan->m_SegmentWaits.size());
            uint32_t streams = 0;
            for (const PlanSegment& segment : plan->m_Segments)
            {
                stats.fenceSignals += segment.signals ? 1 : 0;
                streams |= 1u << StreamOf(segment.queue, segment.lane);
            }
            for (; streams; streams &= streams - 1)
                ++stats.streams;
            stats.heaps = uint32_t(structure->heaps.size());
            stats.poolableTransientBytes = structure->poolableTransientBytes;
            stats.placedPeakBytes = structure->placedPeakBytes;
            stats.unplacedTransientBytes = structure->unplacedTransientBytes;

            return plan.Detach();
        }
    }

    nvrhi::FRESULT CompileGraph(const RenderGraph& graph, const BackendProfile& profile, const CompileOptions& options,
                                uint64_t key, uint64_t maskHash, std::shared_ptr<const StructurePlan> structure,
                                CompiledRenderGraph** outPlan, std::string& diagnostics)
    {
        *outPlan = nullptr;

        CompileState validation(graph, profile, options);
        validation.diagnostics = &diagnostics;
        ValidateStructure(validation);
        if (validation.failed)
            return nvrhi::FE_INVALID_ARGS;

        if (!structure)
        {
            bool failed = false;
            std::shared_ptr<StructurePlan> built = BuildStructure(graph, profile, options, key, diagnostics, failed);
            if (failed || !built)
                return nvrhi::FE_INVALID_ARGS;
            structure = built;
        }
        for (const std::string& warning : structure->warnings)
            diagnostics += "warning: " + warning + "\n";

        CompileState state(graph, profile, options);
        state.diagnostics = &diagnostics;
        state.structure = structure.get();
        ApplyMask(state);
        if (state.failed)
            return nvrhi::FE_INVALID_ARGS;
        BuildSchedule(state, structure->passQueues);
        BuildUses(state);
        ComputeLifetimes(state);
        if (state.failed)
            return nvrhi::FE_INVALID_ARGS;
        RunTransitions(state);
        if (state.failed)
            return nvrhi::FE_INVALID_ARGS;

        *outPlan = BuildPlan(state, structure, key, maskHash);
        return nvrhi::FS_OK;
    }

    // ------------------------------------------------------------------------------------------------ plan queries

    uint32_t CompiledRenderGraph::AddName(const char* name)
    {
        const uint32_t offset = uint32_t(m_Names.size());
        m_Names.insert(m_Names.end(), name, name + std::strlen(name) + 1);
        return offset;
    }

    nvrhi::FRESULT CompiledRenderGraph::GetScheduledPass(uint32_t index, ScheduledPass& out) noexcept
    {
        if (index >= m_Passes.size())
            return nvrhi::FE_INVALID_ARGS;
        const PlanPass& pass = m_Passes[index];
        out.pass = PassId(pass.program);
        out.name = GetName(pass.nameOffset);
        out.queue = pass.queue;
        out.lane = pass.lane;
        out.segment = pass.segment;
        out.enabled = pass.enabled;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT CompiledRenderGraph::GetBatch(uint32_t index, BatchSlot slot, BarrierBatch& out) noexcept
    {
        out = BarrierBatch();
        if (index >= m_Passes.size() || uint32_t(slot) >= c_SlotCount)
            return nvrhi::FE_INVALID_ARGS;
        const PlanPass& pass = m_Passes[index];
        out.count = pass.batchCount[uint32_t(slot)];
        out.barriers = out.count ? &m_Barriers[pass.batchBegin[uint32_t(slot)]] : nullptr;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT CompiledRenderGraph::GetAttachmentOps(uint32_t index, uint32_t attachment, LoadOp& load, StoreOp& store) noexcept
    {
        if (index >= m_Passes.size() || attachment > c_DepthAttachment)
            return nvrhi::FE_INVALID_ARGS;
        load = m_Passes[index].load[attachment];
        store = m_Passes[index].store[attachment];
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT CompiledRenderGraph::GetResourceLifetime(uint32_t resource, ResourceLifetime& out) noexcept
    {
        if (resource >= m_Resources.size())
            return nvrhi::FE_INVALID_ARGS;
        const PlanResource& record = m_Resources[resource];
        out.firstPass = record.firstPass;
        out.lastPass = record.lastPass;
        out.heap = record.heap;
        out.heapOffset = record.heapOffset;
        out.size = record.size;
        out.crossQueue = record.crossQueue;
        out.aliased = record.aliased;
        return nvrhi::FS_OK;
    }
}
