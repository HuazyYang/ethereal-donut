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

// The render graph's private concrete classes (design 3.1). Graph code includes only nvrhi, donut/core, other
// RenderGraph* headers and the standard library (3.2).
//
// Layers:
// - Recording (RenderGraph.cpp): the frame's resources, passes and accesses in a frame arena; no heap allocation in
//   steady state.
// - Compiling (RenderGraphCompiler.cpp, RenderGraphTransitions.cpp): plain C++ over dense arrays. Two levels are
//   cached by the registry: the structure (queue assignment and memory plan, per structural hash) and the plan
//   (schedule, barriers and segments, per pass-enable mask).
// - The plan (CompiledRenderGraph): device-free data. Barriers name graph resource indices.

#pragma once

#include <donut/render/RenderGraph.h>
#include <nvrhi/core/foundation.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace donut::render::rdg
{
    class RenderGraph;
    class ResourceRegistry;
    class CompiledRenderGraph;

    // ------------------------------------------------------------------------------------------------ hashing

    constexpr uint64_t c_HashSeed = 0xcbf29ce484222325ull;

    inline uint64_t HashCombine(uint64_t hash, uint64_t value) noexcept
    {
        hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
        return hash * 0x100000001b3ull;
    }

    inline uint64_t HashString(const char* text) noexcept
    {
        uint64_t hash = c_HashSeed;
        if (text)
        {
            for (const char* c = text; *c; ++c)
                hash = (hash ^ uint8_t(*c)) * 0x100000001b3ull;
        }
        return hash;
    }

    // ------------------------------------------------------------------------------------------------ arena

    // Blocks are kept across frames; Reset rewinds them. A frame that needs more than the blocks hold adds a block,
    // so the steady state allocates nothing.
    class FrameArena
    {
    public:
        FrameArena() = default;
        FrameArena(const FrameArena&) = delete;
        FrameArena& operator=(const FrameArena&) = delete;
        ~FrameArena();

        void* Allocate(size_t size, size_t alignment) noexcept;
        void Reset() noexcept;
        size_t GetBlockCount() const noexcept { return m_Blocks.size(); }

    private:
        struct Block
        {
            char* data;
            size_t size;
        };

        static constexpr size_t c_BlockSize = 256 * 1024;

        std::vector<Block> m_Blocks;
        size_t m_Current = 0;
        size_t m_Offset = 0;
    };

    // A growable array of trivially copyable records in the arena. Growing copies into a larger block of the arena;
    // the old block is reclaimed at Reset.
    template <typename T>
    struct ArenaArray
    {
        static_assert(std::is_trivially_copyable_v<T>, "arena arrays hold plain records");

        T* data = nullptr;
        uint32_t size = 0;
        uint32_t capacity = 0;

        T* Push(FrameArena& arena) noexcept
        {
            if (size == capacity)
            {
                const uint32_t newCapacity = capacity ? capacity * 2 : 16;
                T* newData = static_cast<T*>(arena.Allocate(sizeof(T) * newCapacity, alignof(T)));
                if (!newData)
                    return nullptr;
                if (size)
                    std::memcpy(static_cast<void*>(newData), data, sizeof(T) * size);
                data = newData;
                capacity = newCapacity;
            }
            T* result = &data[size++];
            std::memset(static_cast<void*>(result), 0, sizeof(T));
            return result;
        }

        void Clear() noexcept
        {
            data = nullptr;
            size = 0;
            capacity = 0;
        }

        T& operator[](uint32_t index) noexcept { return data[index]; }
        const T& operator[](uint32_t index) const noexcept { return data[index]; }
    };

    // ------------------------------------------------------------------------------------------------ records

    enum class ResourceKind : uint8_t
    {
        Transient,     // IRenderGraph::CreateTexture/Buffer
        PassLocal,     // ISetupContext::CreateTransientTexture/Buffer
        Imported,      // ImportTexture/Buffer, or a device-free stand-in
        HostKept,      // ImportHostKeptTexture: a keepInitialState texture such as the back buffer
        Persistent,    // GetPersistentTexture/Buffer
        History,       // one layer of GetHistory
    };

    // The fields of nvrhi::TextureDesc that a plan needs, without the name string.
    struct TextureInfo
    {
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        uint32_t arraySize;
        uint32_t mipLevels;
        uint32_t sampleCount;
        uint32_t sampleQuality;
        nvrhi::Format format;
        nvrhi::TextureDimension dimension;
        bool isShaderResource;
        bool isRenderTarget;
        bool isUAV;
        bool isTypeless;
        bool isShadingRateSurface;
        bool useClearValue;
        bool keepInitialState;
        nvrhi::Color clearValue;
        nvrhi::ResourceStates initialState;
        nvrhi::SharedResourceFlags sharedResourceFlags;
    };

    struct BufferInfo
    {
        uint64_t byteSize;
        uint32_t structStride;
        nvrhi::Format format;
        bool canHaveUAVs;
        bool canHaveTypedViews;
        bool canHaveRawViews;
        bool isVertexBuffer;
        bool isIndexBuffer;
        bool isConstantBuffer;
        bool isDrawIndirectArgs;
        bool isAccelStructBuildInput;
        bool isAccelStructStorage;
        bool isShaderBindingTable;
        bool isVolatile;
        bool keepInitialState;
        nvrhi::ResourceStates initialState;
        nvrhi::CpuAccessMode cpuAccess;
    };

    TextureInfo MakeTextureInfo(const nvrhi::TextureDesc& desc) noexcept;
    BufferInfo MakeBufferInfo(const nvrhi::BufferDesc& desc) noexcept;
    uint64_t HashTextureInfo(const TextureInfo& info) noexcept;
    uint64_t HashBufferInfo(const BufferInfo& info) noexcept;

    struct ResourceRecord
    {
        ResourceKind kind;
        bool isBuffer;
        bool standIn;             // a device-free import (ImportTextureDesc / ImportBufferDesc)
        bool extracted;
        bool hasDepth;
        bool hasStencil;
        ResourceFlags flags;
        uint32_t currentVersion;
        uint32_t currentWriter;   // program index of the pass that wrote the current version; ~0u: none
        uint32_t ownerPass;       // PassLocal: the pass that created it
        uint32_t registryEntry;   // Persistent, History: the registry's entry
        uint32_t generation;      // Persistent, History: the entry's allocation generation
        uint8_t historyLayer;
        uint8_t historyLayerCount;
        nvrhi::ResourceStates entry;
        nvrhi::ResourceStates exit;
        nvrhi::IResource* imported;        // not AddRef'd: the caller keeps it alive for the frame
        nvrhi::ITexture** extractOut;
        const char* name;                  // in the arena
        uint64_t descHash;
        TextureInfo texture;
        BufferInfo buffer;
    };

    constexpr uint8_t c_NoAttachment = 0xff;
    constexpr uint8_t c_DepthAttachment = 8;

    struct AccessRecord
    {
        uint32_t resource;
        uint32_t version;       // read: the version read; write: the version produced
        uint32_t prevVersion;   // write: the version written over; read: equal to version
        Access access;
        nvrhi::ShaderType stages;
        AccessFlags flags;
        nvrhi::SubresourceRegion region;
        uint8_t attachment;
        bool isWrite;
        bool hasClear;
        ClearValue clear;
    };

    struct PolicyRecord
    {
        uint32_t pass;
        uint32_t resource;
        uint32_t version;        // the version the disabled pass writes
        uint32_t substitute;     // resource index; ~0u for Cull
        DisabledOutput policy;
    };

    struct PassRecord
    {
        const char* name;
        PassType type;
        PassFlags flags;
        uint32_t exclusionGroup;
        uint64_t nameHash;
        IGpuPass* pass;          // AddRef'd until Reset
        void* data;
        uint32_t firstAccess;
        uint32_t accessCount;
        bool enabled;
    };

    struct MarkerRecord
    {
        uint32_t beforePass;     // program index of the next pass
        const char* name;        // nullptr: pop
    };

    enum class RecordError : uint8_t
    {
        InvalidId,             // resource: the bad index, value: the low word of the id
        StaleVersion,          // resource, value: the version used, other: the pass that wrote the current one
        SetupContextClosed,    // a setup-context call outside Setup
        NestedAddPass,
        Frozen,                // a declaration after Compile
        HistoryWrite,          // resource: a layer other than ids[0]
        PassLocalForeign,      // resource, other: the owner pass
        TextureAccessOnBuffer, // resource
        SetupFailed,           // value: the FRESULT
        BadLayerCount,         // value: the layer count
        ExtractNotTransient,   // resource
        DisabledOutputNotWritten, // resource
        NullPass,
        BadAttachmentSlot,     // value: the slot
        ArenaExhausted,
        PersistentKindMismatch, // a name used for a texture, a buffer and a history at once
        KeepInitialStateTransient, // resource
        TrackedImportOfHostKept,   // resource
        NullImport,
        NullName,
        WriteBitsInRead,       // resource
        ClearOnReadOnlyDepth,  // resource
    };

    struct ErrorRecord
    {
        RecordError code;
        uint32_t pass;           // program index; ~0u outside a pass
        uint32_t resource;
        uint32_t other;
        uint64_t value;
    };

    // ------------------------------------------------------------------------------------------------ profiles

    // The lowering of accesses to nvrhi states (6.1) and the lattice of 6.3. RenderGraphProfile.cpp.
    nvrhi::ShaderType DefaultStages(PassType type) noexcept;
    nvrhi::ResourceStates LowerAccess(Access access, nvrhi::ShaderType stages) noexcept;
    // The state of one access of one resource (6.1). On Vulkan, nvrhi samples a depth or stencil texture in
    // eDepthStencilReadOnlyOptimal, ShaderResource | DepthRead, and its descriptors declare that layout: a shader read
    // of such a texture lowers to that pair.
    nvrhi::ResourceStates LowerAccess(Access access, nvrhi::ShaderType stages, bool depthStencilTexture,
                                      const BackendProfile& profile) noexcept;
    bool IsStateLegalOnQueue(nvrhi::ResourceStates state, nvrhi::CommandQueue queue, const BackendProfile& profile) noexcept;

    struct CombineResult
    {
        bool compatible;
        nvrhi::ResourceStates state;
        uint32_t extraCost;
    };

    // Two read states of one epoch: a single state that serves both, or incompatible.
    CombineResult CombineReadStates(nvrhi::ResourceStates a, nvrhi::ResourceStates b, bool isTexture,
                                    const BackendProfile& profile) noexcept;
    // Vulkan: whether a transition between the two states changes the image layout.
    bool ChangesVulkanLayout(nvrhi::ResourceStates before, nvrhi::ResourceStates after) noexcept;
    // The state a Clear access is recorded in (6.9), as nvrhi's clear functions need it; Unknown when the profile
    // records no barriers.
    nvrhi::ResourceStates ClearState(bool isBuffer, bool hasDepth, bool isRenderTarget, const BackendProfile& profile) noexcept;
    bool QueuesShareFamily(nvrhi::CommandQueue a, nvrhi::CommandQueue b, const BackendProfile& profile) noexcept;
    uint64_t HashProfile(const BackendProfile& profile) noexcept;

    // Estimated placement size and alignment of a resource (7.5), the same on every backend.
    void EstimateResourceSize(const ResourceRecord& record, uint64_t& size, uint64_t& alignment) noexcept;

    // ------------------------------------------------------------------------------------------------ plan data

    constexpr uint32_t c_None = ~0u;

    // Where the barriers of a scheduled pass sit: BatchSlot order.
    constexpr uint32_t c_SlotCount = 3;

    // A cross-queue dependency of a scheduled pass: it waits for `producer` (a schedule index) of this frame or of
    // the previous one.
    struct PassWait
    {
        uint32_t producer;
        int8_t frameDelta;    // 0 or -1
    };

    struct PlanPass
    {
        uint32_t program;                 // PassId
        uint32_t nameOffset;              // into CompiledRenderGraph::m_Names
        nvrhi::CommandQueue queue;
        PassType type;
        PassFlags flags;                  // as declared: Tracked, Restore, AsyncCompute (11.3 dumps)
        bool enabled;
        uint32_t segment;
        uint32_t batchBegin[c_SlotCount];
        uint32_t batchCount[c_SlotCount];
        uint32_t waitBegin;
        uint32_t waitCount;
        uint32_t useBegin;
        uint32_t useCount;
        uint32_t trackedBegin;            // into CompiledRenderGraph::m_TrackedStates: Tracked and Restore passes
        uint32_t trackedCount;
        uint32_t clearMask;               // attachments recorded with a Clear access
        LoadOp load[9];
        StoreOp store[9];
    };

    // One access of one scheduled pass, as compiled (after Substitute and Passthrough redirection). The verifier
    // replays these.
    struct PlanUse
    {
        uint32_t accessIndex;             // the graph's access record (stable for one structure)
        uint32_t resource;
        Access access;
        nvrhi::ShaderType stages;
        AccessFlags flags;
        nvrhi::SubresourceRegion region;  // resolved against the resource
        uint8_t attachment;
        bool isWrite;
        bool hasClear;
    };

    struct SegmentWait
    {
        nvrhi::CommandQueue queue;
        uint32_t segment;
        int8_t frameDelta;
    };

    struct PlanSegment
    {
        nvrhi::CommandQueue queue;
        uint32_t firstPass;               // schedule index
        uint32_t passCount;
        uint32_t waitBegin;
        uint32_t waitCount;
        bool signals;                     // another segment waits for it, in this frame or the next
    };

    // One graph resource of the compiled frame.
    struct PlanResource
    {
        uint32_t nameOffset;
        ResourceKind kind;
        bool isBuffer;
        bool hasDepth;
        bool hasStencil;
        bool isRenderTarget;
        bool extracted;
        bool crossQueue;
        bool aliased;
        bool standIn;
        uint32_t physical;                // index into the memory plan's physical objects; c_None for imports
        uint32_t firstPass;               // schedule indices; c_None when unused
        uint32_t lastPass;
        uint32_t heap;
        uint64_t heapOffset;
        uint64_t size;
        uint32_t registryEntry;
        uint8_t historyLayer;
        uint8_t historyLayerCount;
        uint32_t mipLevels;
        uint32_t arraySlices;
        nvrhi::ResourceStates entry;      // declared entry (imports) or the state the plan assumes at frame start
        nvrhi::ResourceStates exit;       // declared exit (imports, extracted)
        nvrhi::ResourceStates creation;   // the state a new physical resource is created in: its frame-start state
        uint64_t descHash;
    };

    // The state a region of a physical resource is in at frame start, by the plan's cyclic model (6.12): the state its
    // previous frame left it in. `resource` names the physical resource through its first occupant (or the history
    // layer). The registry compares these with the states it tracks and patches the difference on a plan switch.
    // The state a Tracked or Restore pass's body finds a range in (9.1): the state of the read group or write the
    // pass takes part in, which a merged read group widens beyond the pass's own access.
    struct PlanTrackedState
    {
        uint32_t resource;
        nvrhi::SubresourceRegion region;
        nvrhi::ResourceStates state;
    };

    struct PlanEntryState
    {
        uint32_t resource;
        nvrhi::SubresourceRegion region;
        nvrhi::ResourceStates state;
    };

    // ------------------------------------------------------------------------------------------------ memory plan

    enum class PhysicalKind : uint8_t { Dedicated, Placed, PooledSlot };

    struct PhysicalObject
    {
        PhysicalKind kind;
        uint32_t firstResource;           // the occupant that describes it
        uint32_t heap;                    // Placed
        uint64_t offset;
        uint64_t size;
    };

    struct HeapInfo
    {
        nvrhi::CommandQueue queue;
        uint8_t heapClass;                // 0: any; tier 1: 1 RT/DS textures, 2 other textures, 3 buffers
        uint64_t size;
    };

    // Per structural hash (7.7): what every pass-enable mask of one graph shares.
    struct StructurePlan
    {
        uint64_t key = 0;
        MemoryTier tier = MemoryTier::Pooled;
        std::vector<nvrhi::CommandQueue> passQueues;     // per program pass, after async placement
        std::vector<uint8_t> demoted;                    // per program pass: an async pass placed on graphics
        std::vector<uint32_t> resourcePhysical;          // per resource; c_None for imports
        std::vector<uint8_t> resourceAliased;
        std::vector<uint8_t> resourceCrossQueue;
        std::vector<uint32_t> resourceFirst;             // superset lifetimes (schedule = program order)
        std::vector<uint32_t> resourceLast;
        std::vector<PhysicalObject> physicals;
        std::vector<HeapInfo> heaps;
        std::vector<std::vector<uint32_t>> overlaps;     // per resource: placed resources sharing bytes with it
        uint64_t poolableTransientBytes = 0;
        uint64_t placedPeakBytes = 0;
        uint64_t unplacedTransientBytes = 0;
        std::vector<std::string> warnings;
    };

    // ------------------------------------------------------------------------------------------------ the plan

    struct PlanInstance;   // the executor's lowered form of a plan (RenderGraphExecutor.cpp)

    class CompiledRenderGraph final : public nvrhi::ObjectImpl<ICompiledRenderGraph>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(CompiledRenderGraph)
        NVRHI_IMPLEMENTS_INTERFACE(ICompiledRenderGraph)
        NVRHI_END_INTERFACE_TABLE()

        CompiledRenderGraph();
        ~CompiledRenderGraph();

        // ICompiledRenderGraph
        uint64_t GetPlanHash() noexcept override { return m_Key; }
        uint64_t GetPlanEnableMaskHash() noexcept override { return m_MaskHash; }
        uint32_t GetScheduledPassCount() noexcept override { return uint32_t(m_Passes.size()); }
        nvrhi::FRESULT GetScheduledPass(uint32_t index, ScheduledPass& out) noexcept override;
        nvrhi::FRESULT GetBatch(uint32_t index, BatchSlot slot, BarrierBatch& out) noexcept override;
        nvrhi::FRESULT GetAttachmentOps(uint32_t index, uint32_t attachment, LoadOp& load, StoreOp& store) noexcept override;
        nvrhi::FRESULT GetResourceLifetime(uint32_t resource, ResourceLifetime& out) noexcept override;
        void GetStatistics(PlanStatistics& out) noexcept override { out = m_Statistics; }
        nvrhi::FRESULT WriteDump(DumpFormat format, nvrhi::string& out) noexcept override;

        const char* GetName(uint32_t offset) const noexcept { return m_Names.data() + offset; }
        uint32_t AddName(const char* name);

        uint64_t m_Key = 0;                 // structural hash, profile and options
        uint64_t m_StructuralHash = 0;
        uint64_t m_MaskHash = 0;
        uint64_t m_ProfileHash = 0;
        BackendProfile m_Profile;
        CompileOptions m_Options;
        std::shared_ptr<const StructurePlan> m_Structure;
        uint32_t m_ProgramPassCount = 0;
        uint32_t m_AccessCount = 0;

        std::vector<PlanPass> m_Passes;
        std::vector<PlanBarrier> m_Barriers;
        std::vector<PassWait> m_PassWaits;
        std::vector<PlanUse> m_Uses;
        std::vector<PlanSegment> m_Segments;
        std::vector<SegmentWait> m_SegmentWaits;
        std::vector<PlanResource> m_Resources;
        std::vector<uint32_t> m_ScheduleOfProgram;   // per program pass: schedule index or c_None
        std::vector<PlanEntryState> m_EntryStates;
        std::vector<PlanTrackedState> m_TrackedStates;   // per scheduled pass, PlanPass::trackedBegin
        std::vector<char> m_Names;
        PlanStatistics m_Statistics;

        uint64_t m_Serial = 0;                       // unique per plan object, never reused
        std::unique_ptr<PlanInstance> m_Instance;    // created by the first Execute
    };

    // ------------------------------------------------------------------------------------------------ runtime (9.1)

    // A GPU resource the registry owns: the memory of transients, a persistent resource, a history layer, or an
    // extracted texture. `states` (planes x mips x slices) is exact when `stateOwner` is 0. Otherwise it is the
    // frame-start state of plan `stateOwner`'s cyclic model (6.12), which holds as long as that plan runs every frame.
    struct PhysicalResource
    {
        nvrhi::AutoPtr<nvrhi::ITexture> texture;
        nvrhi::AutoPtr<nvrhi::IBuffer> buffer;
        nvrhi::IResource* resource = nullptr;
        bool isBuffer = false;
        uint64_t descHash = 0;
        uint32_t planes = 1;
        uint32_t mips = 1;
        uint32_t slices = 1;
        std::vector<nvrhi::ResourceStates> states;
        bool fresh = true;                // never used: in the backend's creation state
        uint64_t stateOwner = 0;
        uint64_t lastUsedFrame = 0;
        // The Placed tier (7.5): bound to a heap of its structure instance at an offset; never pooled.
        bool placed = false;
        uint32_t heap = ~0u;
        uint64_t heapOffset = 0;
        uint64_t heapSize = 0;
    };

    // The physical resources of one structure (one memory plan), shared by the plans of all its masks.
    struct StructureInstance
    {
        std::shared_ptr<const StructurePlan> structure;
        std::vector<std::unique_ptr<PhysicalResource>> physicals;   // per PhysicalObject
        std::vector<nvrhi::AutoPtr<nvrhi::IHeap>> heaps;            // the Placed tier: per planned heap
        uint64_t heapBytes = 0;
        uint64_t placedObjectBytes = 0;                              // the placed objects' own sizes, summed
    };

    // A resource of the frame being executed.
    struct FrameResource
    {
        nvrhi::IResource* resource;
        nvrhi::ITexture* texture;
        nvrhi::IBuffer* buffer;
        PhysicalResource* physical;       // nullptr for imports
    };

    struct RegistryRuntime
    {
        std::vector<std::unique_ptr<StructureInstance>> instances;
        std::vector<std::unique_ptr<PhysicalResource>> freePool;     // transient memory of retired structures
        // 7.5 heap retirement: collected instances with heaps or placed resources, kept until every queue's fence
        // passes the values of the last frame before their collection. Barrier arrays hold raw pointers, and nvrhi's
        // explicit barriers keep no references, so the registry holds them until then.
        struct RetiredInstance
        {
            std::unique_ptr<StructureInstance> instance;
            std::array<uint64_t, 3> values;
        };
        std::vector<RetiredInstance> retired;
        std::vector<std::unique_ptr<PhysicalResource>> extractPool;
        std::array<nvrhi::AutoPtr<nvrhi::IFence>, 3> fences;
        std::array<uint64_t, 3> fenceValues = {};
        std::array<uint64_t, 3> lastFrameValues = {};                // each queue's last signal of the last frame
        bool fencesChecked = false;
        bool fencesSupported = false;
        std::array<std::vector<nvrhi::AutoPtr<nvrhi::ICommandList>>, 3> commandLists;
        nvrhi::AutoPtr<nvrhi::ICommandList> prologueList;
        // Without fences (D3D11, 8.3): one event query per frame, in a ring. A query's results are read before it is
        // ended again, as D3D11 expects; a query three frames old has normally completed.
        std::array<nvrhi::AutoPtr<nvrhi::IEventQuery>, 3> frameQueries;
        std::array<bool, 3> frameQueryPending = {};
        uint32_t frameQueryNext = 0;
        nvrhi::AutoPtr<CompiledRenderGraph> lastPlan;
        std::vector<FrameResource> frameResources;
        std::vector<nvrhi::ResourceBarrier> patches;
        std::vector<PhysicalResource*> patchOwners;                  // per patch: the tracker fallback's resource
        std::vector<nvrhi::ResourceStates> scratchStates;
        nvrhi::AutoPtr<IPrepareContext> prepareContext;
        std::vector<nvrhi::AutoPtr<IPassContext>> passContexts;      // one per segment
        uint64_t objectsCreated = 0;
        uint64_t framesExecuted = 0;

        // 11.3: the last Execute's statistics, and per-pass timers in a ring of frames, read back when they complete.
        RunStatistics run;
        struct TimerSet
        {
            std::vector<nvrhi::AutoPtr<nvrhi::ITimerQuery>> queries;
            std::vector<std::string> names;
            uint32_t used = 0;
            uint64_t frameIndex = 0;
            bool pending = false;
        };
        std::array<TimerSet, 4> timerSets;
        bool timersSupported = true;
        std::vector<std::string> timingNames;
        std::vector<PassTiming> timings;
        uint64_t lastFrameIndex = 0;                                 // the graph frame index of the last Execute
    };

    // Runs a plan that Compile returned for the graph's current frame (9.1). RenderGraphExecutor.cpp.
    // The descriptions a record holds, for the executor's create calls and IRenderGraph::GetTextureDesc. The
    // initial state is the backend's (D3D12 COMMON, Vulkan UNDEFINED); the first frame's patch moves it.
    inline nvrhi::TextureDesc MakeTextureDesc(const ResourceRecord& record, bool withName = true)
    {
        const TextureInfo& info = record.texture;
        nvrhi::TextureDesc desc;
        desc.width = info.width;
        desc.height = info.height;
        desc.depth = info.depth;
        desc.arraySize = info.arraySize;
        desc.mipLevels = info.mipLevels;
        desc.sampleCount = info.sampleCount;
        desc.sampleQuality = info.sampleQuality;
        desc.format = info.format;
        desc.dimension = info.dimension;
        desc.isShaderResource = info.isShaderResource;
        desc.isRenderTarget = info.isRenderTarget;
        desc.isUAV = info.isUAV;
        desc.isTypeless = info.isTypeless;
        desc.isShadingRateSurface = info.isShadingRateSurface;
        desc.useClearValue = info.useClearValue;
        desc.clearValue = info.clearValue;
        desc.sharedResourceFlags = info.sharedResourceFlags;
        // Created in the backend's initial state (D3D12 COMMON, Vulkan UNDEFINED); the first frame's patch moves it.
        desc.initialState = nvrhi::ResourceStates::Unknown;
        desc.keepInitialState = false;
        if (withName && record.name)
            desc.debugName = record.name;
        return desc;
    }

    inline nvrhi::BufferDesc MakeBufferDesc(const ResourceRecord& record, bool withName = true)
    {
        const BufferInfo& info = record.buffer;
        nvrhi::BufferDesc desc;
        desc.byteSize = info.byteSize;
        desc.structStride = info.structStride;
        desc.format = info.format;
        desc.canHaveUAVs = info.canHaveUAVs;
        desc.canHaveTypedViews = info.canHaveTypedViews;
        desc.canHaveRawViews = info.canHaveRawViews;
        desc.isVertexBuffer = info.isVertexBuffer;
        desc.isIndexBuffer = info.isIndexBuffer;
        desc.isConstantBuffer = info.isConstantBuffer;
        desc.isDrawIndirectArgs = info.isDrawIndirectArgs;
        desc.isAccelStructBuildInput = info.isAccelStructBuildInput;
        desc.isAccelStructStorage = info.isAccelStructStorage;
        desc.isShaderBindingTable = info.isShaderBindingTable;
        desc.cpuAccess = info.cpuAccess;
        desc.initialState = nvrhi::ResourceStates::Unknown;
        desc.keepInitialState = false;
        if (withName && record.name)
            desc.debugName = record.name;
        return desc;
    }

    nvrhi::FRESULT ExecutePlan(RenderGraph& graph, CompiledRenderGraph& plan, const ExecuteDesc& desc) noexcept;

    // D3D12 legacy DiscardResource (6.6): a depth texture is discarded in DEPTH_WRITE, another render-target texture
    // in RENDER_TARGET, any other texture or a buffer in UNORDERED_ACCESS. The debug layer rejects the other states.
    inline bool LegacyDiscardable(nvrhi::ResourceStates state, bool isBuffer, bool hasDepth, bool isRenderTarget) noexcept
    {
        if (!isBuffer && hasDepth)
            return state == nvrhi::ResourceStates::DepthWrite;
        if (!isBuffer && isRenderTarget)
            return state == nvrhi::ResourceStates::RenderTarget;
        return state == nvrhi::ResourceStates::UnorderedAccess;
    }
    // Makes the last executed plan's cyclic states exact (6.12) and forgets the plan; before ReleaseResources drops it.
    void FinalizeLastPlan(ResourceRegistry& registry) noexcept;

    // ------------------------------------------------------------------------------------------------ registry

    struct PersistentEntry
    {
        std::string name;
        uint64_t nameHash = 0;
        uint8_t kind = 0;                   // 0 texture, 1 buffer, 2 history
        uint8_t layerCount = 0;
        uint64_t descHash = 0;
        uint32_t generation = 0;
        uint64_t allocatedFrame = 0;        // the frame of the current generation's first use
        uint64_t lastFrame = 0;
        bool live = true;
        std::vector<std::unique_ptr<PhysicalResource>> physicals;   // 1, or the history's layer count
        uint32_t physicalGeneration = 0;
    };

    struct PlanCacheEntry
    {
        uint64_t key = 0;
        uint64_t maskHash = 0;
        uint32_t passCount = 0;
        uint32_t resourceCount = 0;
        uint32_t accessCount = 0;
        uint64_t lastUsed = 0;
        nvrhi::AutoPtr<CompiledRenderGraph> plan;
    };

    class ResourceRegistry final : public nvrhi::ObjectImpl<IResourceRegistry>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(ResourceRegistry)
        NVRHI_IMPLEMENTS_INTERFACE(IResourceRegistry)
        NVRHI_END_INTERFACE_TABLE()

        explicit ResourceRegistry(nvrhi::IDevice* device);
        ~ResourceRegistry();

        // IResourceRegistry
        nvrhi::IDevice* GetDevice() noexcept override { return m_Device; }
        void ReleaseResources() noexcept override;
        nvrhi::FRESULT ReleasePersistent(const char* name) noexcept override;
        void ClearPlanCache() noexcept override;
        void GetRegistryStatistics(RegistryStatistics& out) noexcept override;
        void GetRunStatistics(RunStatistics& out) noexcept override;
        uint32_t GetPassTimingCount() noexcept override;
        nvrhi::FRESULT GetPassTiming(uint32_t index, PassTiming& out) noexcept override;

        // Persistent records: the entry for `name`, created or reallocated when the description changes.
        uint32_t AcquirePersistent(const char* name, uint8_t kind, uint64_t descHash, uint8_t layerCount,
                                   uint64_t frameIndex, uint32_t& generation, bool& kindMismatch);
        const PersistentEntry& GetPersistent(uint32_t index) const { return m_Persistent[index]; }
        PersistentEntry& GetPersistentMutable(uint32_t index) { return m_Persistent[index]; }

        // The executor's state (RenderGraphExecutor.cpp).
        RegistryRuntime m_Runtime;
        // Waits until the GPU has finished every frame this registry submitted.
        void WaitForFrames() noexcept;

        // Plan cache (7.7).
        CompiledRenderGraph* FindPlan(uint64_t key, uint64_t maskHash, uint32_t passCount, uint32_t resourceCount,
                                      uint32_t accessCount, uint64_t frameIndex) noexcept;
        void InsertPlan(CompiledRenderGraph* plan, uint32_t passCount, uint32_t resourceCount, uint32_t accessCount,
                        uint32_t maxCachedPlans, uint64_t frameIndex);
        std::shared_ptr<const StructurePlan> FindStructure(uint64_t key) const noexcept;
        void CountMiss() noexcept { ++m_PlanMisses; }

        RenderGraph* m_Graph = nullptr;   // the one graph (12): not AddRef'd, it holds this registry

    private:
        nvrhi::IDevice* m_Device;
        nvrhi::AutoPtr<nvrhi::IDevice> m_DeviceReference;
        std::vector<PersistentEntry> m_Persistent;
        std::vector<PlanCacheEntry> m_Plans;
        uint64_t m_PlanHits = 0;
        uint64_t m_PlanMisses = 0;
    };

    // ------------------------------------------------------------------------------------------------ the graph

    class RenderGraph final : public nvrhi::ObjectImpl<IRenderGraph, ISetupContext>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(RenderGraph)
        NVRHI_IMPLEMENTS_INTERFACE(IRenderGraph)
        NVRHI_IMPLEMENTS_INTERFACE(ISetupContext)
        NVRHI_END_INTERFACE_TABLE()

        explicit RenderGraph(ResourceRegistry* registry);
        ~RenderGraph();

        // IRenderGraph
        IResourceRegistry* GetResourceRegistry() noexcept override { return m_Registry; }
        void Reset() noexcept override;
        ResourceId CreateTexture(const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept override;
        ResourceId CreateBuffer(const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept override;
        ResourceId ImportTexture(nvrhi::ITexture* texture, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept override;
        ResourceId ImportHostKeptTexture(nvrhi::ITexture* texture) noexcept override;
        ResourceId ImportBuffer(nvrhi::IBuffer* buffer, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept override;
        ResourceId ImportTextureDesc(const nvrhi::TextureDesc& desc, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept override;
        ResourceId ImportBufferDesc(const nvrhi::BufferDesc& desc, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept override;
        ResourceId GetPersistentTexture(const char* name, const nvrhi::TextureDesc& desc) noexcept override;
        ResourceId GetPersistentBuffer(const char* name, const nvrhi::BufferDesc& desc) noexcept override;
        nvrhi::FRESULT GetHistory(const char* name, const nvrhi::TextureDesc& desc, uint32_t layerCount, ResourceId* ids) noexcept override;
        void Extract(ResourceId texture, nvrhi::ResourceStates exit, nvrhi::ITexture** outTexture) noexcept override;
        nvrhi::FRESULT AddPass(const PassDesc& desc, IGpuPass* pass, void* data, PassId* outPass) noexcept override;
        void SetPassEnabled(PassId pass, bool enabled) noexcept override;
        void PushMarker(const char* name) noexcept override;
        void PopMarker() noexcept override;
        void* AllocateFrameData(size_t size, size_t alignment) noexcept override;
        nvrhi::FRESULT Compile(const BackendProfile& profile, const CompileOptions& options, ICompiledRenderGraph** outPlan) noexcept override;
        nvrhi::FRESULT Execute(ICompiledRenderGraph* plan, const ExecuteDesc& desc) noexcept override;
        uint64_t GetStructuralHash() noexcept override { return m_StructuralHash; }
        uint64_t GetEnableMaskHash() noexcept override;
        uint32_t GetPassCount() noexcept override { return m_Passes.size; }
        uint32_t GetResourceCount() noexcept override { return m_Resources.size; }
        uint64_t GetFrameIndex() noexcept override { return m_FrameIndex; }
        nvrhi::FRESULT GetTextureDesc(ResourceId texture, nvrhi::TextureDesc& out) noexcept override;
        nvrhi::FRESULT GetBufferDesc(ResourceId buffer, nvrhi::BufferDesc& out) noexcept override;
        void GetDiagnostics(nvrhi::string& out) noexcept override;

        // ISetupContext
        ResourceId CreateTransientTexture(const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept override;
        ResourceId CreateTransientBuffer(const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept override;
        void Read(ResourceId resource, Access access, nvrhi::ShaderType stages, const nvrhi::SubresourceRegion& region) noexcept override;
        ResourceId Write(ResourceId resource, Access access, nvrhi::ShaderType stages, const nvrhi::SubresourceRegion& region, AccessFlags flags) noexcept override;
        ResourceId SetColorTarget(uint32_t slot, ResourceId texture, const nvrhi::SubresourceRegion& region, AccessFlags flags, const ClearValue* clear) noexcept override;
        ResourceId SetDepthTarget(ResourceId texture, const nvrhi::SubresourceRegion& region, bool readOnly, AccessFlags flags,
                                  const ClearValue* clear) noexcept override;
        void SetDisabledOutput(ResourceId written, DisabledOutput policy, ResourceId substituteOrInput) noexcept override;

        // For the compiler and the verifier.
        uint32_t ResourceCount() const noexcept { return m_Resources.size; }
        uint32_t PassCount() const noexcept { return m_Passes.size; }
        uint32_t AccessCount() const noexcept { return m_Accesses.size; }
        uint32_t PolicyCount() const noexcept { return m_Policies.size; }
        const ResourceRecord& GetResource(uint32_t index) const noexcept { return m_Resources[index]; }
        const PassRecord& GetPass(uint32_t index) const noexcept { return m_Passes[index]; }
        const AccessRecord& GetAccess(uint32_t index) const noexcept { return m_Accesses[index]; }
        const PolicyRecord& GetPolicy(uint32_t index) const noexcept { return m_Policies[index]; }
        ResourceRegistry* GetRegistry() const noexcept { return m_Registry; }
        uint64_t StructuralHashValue() const noexcept { return m_StructuralHash; }
        uint64_t FrameIndexValue() const noexcept { return m_FrameIndex; }
        uint32_t MarkerCount() const noexcept { return m_Markers.size; }
        const MarkerRecord& GetMarker(uint32_t index) const noexcept { return m_Markers[index]; }
        CompiledRenderGraph* GetCompiledPlan() const noexcept { return m_Plan; }

    private:
        ResourceRecord* NewResource(ResourceKind kind, bool isBuffer) noexcept;
        ResourceId AddTextureResource(ResourceKind kind, const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept;
        ResourceId AddBufferResource(ResourceKind kind, const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept;
        // The record of `id` if it names the current version of a resource; otherwise records the error.
        ResourceRecord* CheckId(ResourceId id, bool forWrite) noexcept;
        bool CheckOpen() noexcept;
        bool CheckNotFrozen() noexcept;
        AccessRecord* AddAccess(ResourceRecord* record, uint32_t index, Access access, nvrhi::ShaderType stages,
                                const nvrhi::SubresourceRegion& region, AccessFlags flags, bool isWrite) noexcept;
        void RecordErrorAt(RecordError code, uint32_t resource = c_None, uint32_t other = c_None, uint64_t value = 0) noexcept;
        const char* CopyString(const char* text) noexcept;
        void Hash(uint64_t value) noexcept { m_StructuralHash = HashCombine(m_StructuralHash, value); }
        void FormatErrors(std::string& out) const;
        std::string DescribeResource(uint32_t index) const;
        std::string DescribePass(uint32_t index) const;

        nvrhi::AutoPtr<ResourceRegistry> m_Registry;
        FrameArena m_Arena;
        ArenaArray<ResourceRecord> m_Resources;
        ArenaArray<PassRecord> m_Passes;
        ArenaArray<AccessRecord> m_Accesses;
        ArenaArray<PolicyRecord> m_Policies;
        ArenaArray<MarkerRecord> m_Markers;
        ArenaArray<ErrorRecord> m_Errors;
        uint64_t m_StructuralHash = c_HashSeed;
        uint64_t m_FrameIndex = 0;
        uint32_t m_NextVersion = 1;
        uint32_t m_OpenPass = c_None;
        bool m_InSetup = false;
        bool m_Frozen = false;
        nvrhi::AutoPtr<CompiledRenderGraph> m_Plan;
        uint64_t m_PlanProfileHash = 0;
        uint64_t m_PlanOptionsHash = 0;
        std::string m_Diagnostics;
    };

    // ------------------------------------------------------------------------------------------------ compiler

    uint64_t HashOptions(const CompileOptions& options) noexcept;

    // Compiles the graph's recorded frame. `structure` is the cached structure for the key, or nullptr; a new one is
    // returned through `outStructure` when it was compiled. Errors and warnings go to `diagnostics`, one per line.
    nvrhi::FRESULT CompileGraph(const RenderGraph& graph, const BackendProfile& profile, const CompileOptions& options,
                                uint64_t key, uint64_t maskHash, std::shared_ptr<const StructurePlan> structure,
                                CompiledRenderGraph** outPlan, std::string& diagnostics);

    // The verifier (11.1). RenderGraphVerifier.cpp. With `previous`, it replays a plan switch (6.12): `previous`'s
    // frame, then `plan`'s, with the registry's switch patch between them (the shared physical resources are moved
    // to the states `plan` expects, and the new frame waits for every queue of the old one).
    nvrhi::FRESULT VerifyCompiledPlan(const CompiledRenderGraph& plan, const CompiledRenderGraph* previous,
                                      std::string& report);

    // Text forms for diagnostics and dumps. RenderGraphDump.cpp.
    std::string StatesToString(nvrhi::ResourceStates states);
    std::string AccessToString(Access access);
    const char* QueueName(nvrhi::CommandQueue queue) noexcept;
    const char* ReasonName(BarrierReason reason) noexcept;
    const char* KindName(ResourceKind kind) noexcept;
    std::string DescribeBarrier(const CompiledRenderGraph& plan, const PlanBarrier& barrier);
}
