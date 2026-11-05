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

// The render graph (aggregate docs/rdg-design.md, section 12). The interfaces follow nvrhi's ABI rules
// (com-interface-abi.md): COM interfaces with IIDs, noexcept virtuals returning FRESULT where they can fail,
// raw pointers in parameters, no STL, no struct by value through a virtual (R1), no overloaded virtuals (R3),
// no default arguments on virtuals. Ids are enums so that they may cross virtual calls by value.

#pragma once

#include <nvrhi/nvrhi.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

// Bit operators for the flag enums below. nvrhi undefines its own macro at the end of nvrhi.h.
#define DONUT_RENDER_GRAPH_FLAG_OPERATORS(T) \
    constexpr T operator|(T a, T b) noexcept { return T(std::underlying_type_t<T>(a) | std::underlying_type_t<T>(b)); } \
    constexpr T operator&(T a, T b) noexcept { return T(std::underlying_type_t<T>(a) & std::underlying_type_t<T>(b)); } \
    constexpr T operator~(T a) noexcept { return T(~std::underlying_type_t<T>(a)); } \
    constexpr bool operator!(T a) noexcept { return std::underlying_type_t<T>(a) == 0; } \
    constexpr T& operator|=(T& a, T b) noexcept { return a = a | b; } \
    constexpr T& operator&=(T& a, T b) noexcept { return a = a & b; }

namespace donut::render
{
    // ------------------------------------------------------------------------------------------------ ids (5.1)

    // A texture or buffer in one graph: index in the low 32 bits, version in the high 32. A write returns a new
    // version, and any other version than the current one is rejected by Compile, which also rejects an id kept
    // from an earlier frame: versions come from a counter that Reset never clears.
    enum class ResourceId : uint64_t {};
    constexpr ResourceId InvalidResourceId = ResourceId(~0ull);
    constexpr uint32_t c_MaxHistoryLayers = 8;

    constexpr ResourceId MakeResourceId(uint32_t index, uint32_t version) noexcept
    {
        return ResourceId(uint64_t(index) | (uint64_t(version) << 32));
    }
    constexpr uint32_t GetResourceIndex(ResourceId id) noexcept { return uint32_t(uint64_t(id)); }
    constexpr uint32_t GetResourceVersion(ResourceId id) noexcept { return uint32_t(uint64_t(id) >> 32); }

    // A pass added in the current frame, in AddPass order.
    enum class PassId : uint32_t {};
    constexpr PassId InvalidPassId = PassId(~0u);

    // ------------------------------------------------------------------------------------------ accesses (6.1)

    enum class Access : uint32_t
    {
        None                  = 0,

        // Reads
        VertexBuffer          = 1u << 0,
        IndexBuffer           = 1u << 1,
        IndirectArgs          = 1u << 2,
        ConstantBuffer        = 1u << 3,
        ShaderRead            = 1u << 4,    // SRV
        UAVRead               = 1u << 5,    // a UAV that the pass only reads; cannot share a D3D12 state with SRVs
        ColorTargetRead       = 1u << 6,    // blending: always declared with ColorTarget
        DepthRead             = 1u << 7,    // depth test without writes
        StencilRead           = 1u << 8,
        CopySource            = 1u << 9,
        ResolveSource         = 1u << 10,
        ShadingRateRead       = 1u << 11,
        AccelStructRead       = 1u << 12,
        AccelStructBuildInput = 1u << 13,
        Present               = 1u << 14,
        HostRead              = 1u << 15,

        // Writes
        UAVWrite              = 1u << 16,
        ColorTarget           = 1u << 17,
        DepthWrite            = 1u << 18,
        StencilWrite          = 1u << 19,
        CopyDest              = 1u << 20,
        ResolveDest           = 1u << 21,
        AccelStructWrite      = 1u << 22,

        AllReads              = 0x0000ffffu,
        AllWrites             = 0x007f0000u,
    };
    DONUT_RENDER_GRAPH_FLAG_OPERATORS(Access)

    enum class AccessFlags : uint32_t
    {
        None         = 0,
        Clear        = 1,   // the pass initialises the range with a clear as its first operation (6.9)
        FullWrite    = 2,   // the pass writes every element of the range: makes a discard-before legal (6.6)
        RelaxedOrder = 4,   // consecutive relaxed UAVWrite accesses need no UAV barrier between them (6.5)
        Preserve     = 8,   // never infer a discard for this access
    };
    DONUT_RENDER_GRAPH_FLAG_OPERATORS(AccessFlags)

    // ------------------------------------------------------------------------------------- passes and resources

    enum class PassType : uint8_t { Raster, Compute, Copy, RayTracing };

    enum class PassFlags : uint32_t
    {
        None         = 0,
        AsyncCompute = 1,    // runs on the compute queue where the profile has one (6.8)
        NeverCull    = 2,
        SideEffect   = 4,    // writes something the graph does not see (a readback, a query): never culled
        Tracked      = 8,    // the body relies on automatic barriers (9.1)
        Restore      = 16,   // Tracked, and the declared ranges are returned to their declared states after the body
    };
    DONUT_RENDER_GRAPH_FLAG_OPERATORS(PassFlags)

    enum class ResourceFlags : uint32_t
    {
        None         = 0,
        NoAlias      = 1,   // kept out of placement (7.5)
        DebugCapture = 2,   // dedicated memory, so that a capture tool sees one stable object
    };
    DONUT_RENDER_GRAPH_FLAG_OPERATORS(ResourceFlags)

    // What consumers of a disabled pass's output see (13.5).
    enum class DisabledOutput : uint8_t { Cull, Substitute, Passthrough };

    // Colour clears use `color`; integer formats take color[0] as the value. Depth-stencil clears use the rest.
    struct ClearValue
    {
        float color[4];
        float depth;
        uint8_t stencil;
    };

    struct PassDesc
    {
        const char* name;
        PassType type;
        PassFlags flags;
        uint32_t exclusionGroup = 0;   // 0: none; passes of one group are never enabled together (13.5)
    };

    // ------------------------------------------------------------------------------- backend profiles (9.2)

    enum class BarrierModel : uint8_t
    {
        Implicit,        // D3D11, CUDA: no barriers are recorded
        D3D12Legacy,     // resource states
        D3D12Enhanced,   // layout, access and sync
        Vulkan,          // sync2: layout, access and stage
    };

    enum class PlaneModel : uint8_t { Ignored, PlaneSlices, AspectMasks };

    enum class MemoryTier : uint8_t { Pooled, Placed };

    enum class RecordingMode : uint8_t
    {
        Immediate,        // D3D12, Vulkan: command lists record in parallel
        ImmediateSerial,  // D3D11: one recording thread
        Deferred,         // CUDA: recorded lists are replayed at submit
    };

    enum class PassTypeMask : uint8_t
    {
        None       = 0,
        Raster     = 1 << uint8_t(PassType::Raster),
        Compute    = 1 << uint8_t(PassType::Compute),
        Copy       = 1 << uint8_t(PassType::Copy),
        RayTracing = 1 << uint8_t(PassType::RayTracing),
        All        = 0x0f,
    };
    DONUT_RENDER_GRAPH_FLAG_OPERATORS(PassTypeMask)

    // The costs that decide between one merged state and split read groups (6.3), and when async compute pays
    // for its extra barriers (6.8). Units are abstract; one transition costs `transition`.
    struct LatticePolicy
    {
        uint16_t transition = 4;
        uint16_t generalLayout = 6;          // Vulkan GENERAL instead of a read-only layout, per epoch
        uint16_t queueTransfer = 6;          // one release/acquire pair
        uint16_t asyncOverlapGain = 16;      // what one async pass gains by overlapping graphics work
        uint8_t splitMinGap = 1;             // passes between the begin and end points before a barrier is split
        bool splitLayoutTransitionsOnly = false;   // Vulkan: events are spent on layout transitions only
        uint64_t splitMinBytes = 0;          // Vulkan: and only on resources at least this large
    };

    // Computed once per device; the compiler sees only this (9.2).
    struct BackendProfile
    {
        nvrhi::GraphicsAPI api = nvrhi::GraphicsAPI::D3D12;
        BarrierModel barrierModel = BarrierModel::D3D12Legacy;
        PlaneModel planeModel = PlaneModel::PlaneSlices;
        MemoryTier maxMemoryTier = MemoryTier::Pooled;      // Placed: heaps that the memory plan may use
        RecordingMode recordingMode = RecordingMode::Immediate;
        PassTypeMask passTypes = PassTypeMask::All;
        bool explicitBarriers = true;           // ICommandList::barriers; false lowers through the tracker (9.1)
        bool splitBarriers = true;
        bool aliasingBarriers = true;
        bool computeQueue = true;
        bool copyQueue = true;
        bool queueOwnershipTransfer = false;    // Vulkan with a compute or copy family other than graphics
        bool depthClears = true;                // false on CUDA (6.9)
        uint8_t resourceHeapTier = 2;           // D3D12 tier 1 keeps render-target/depth textures, other textures
                                                // and buffers in separate heaps (7.5)
        LatticePolicy lattice;
    };

    // The device-free profiles that tests and rdg_bench compile against.
    enum class ReferenceProfile : uint8_t
    {
        D3D12Legacy,
        D3D12Enhanced,
        Vulkan,                        // one family for graphics and compute
        VulkanSeparateComputeFamily,   // ownership transfers between graphics and compute
        D3D11,
        CUDA,
    };

    // ------------------------------------------------------------------------------- compile and execute

    enum class DebugFlags : uint32_t
    {
        None                  = 0,
        ForceDedicated        = 1 << 0,   // every transient gets its own memory
        NoAliasing            = 1 << 1,   // Placed tier without overlap
        BarrierAfterEveryPass = 1 << 2,   // a full barrier after every pass
        FillTransients        = 1 << 3,   // a discard-before also fills the range with a pattern
        SerializeQueues       = 1 << 4,   // async passes run on the graphics queue
        StrictValidation      = 1 << 5,   // the verifier runs on every Compile, not only on a miss
        SkipAliasingBarriers  = 1 << 6,   // tests only: drops aliasing barriers, so that the aliasing test fails
    };
    DONUT_RENDER_GRAPH_FLAG_OPERATORS(DebugFlags)

    enum class ScheduleFlags : uint32_t
    {
        None       = 0,
        GroupReads = 1,   // readers inside a read epoch may be reordered to bring compatible ones together (6.3)
    };
    DONUT_RENDER_GRAPH_FLAG_OPERATORS(ScheduleFlags)

    struct CompileOptions
    {
        bool placedResources = false;     // opt in to the Placed tier where the profile has it (7.5)
        uint32_t maxCachedPlans = 8;      // least recently used; 0 compiles every frame (7.7)
        DebugFlags debugFlags = DebugFlags::None;
        ScheduleFlags scheduleFlags = ScheduleFlags::None;
        uint32_t recordGranularity = 0;   // passes per segment when recording in parallel; 0: no parallel split
    };

    // Runs task(taskContext, i) for every i in [0, count) and returns when all have finished.
    typedef void (*TaskFunction)(void* taskContext, uint32_t index);
    typedef void (*ParallelForFunction)(void* schedulerContext, uint32_t count, TaskFunction task, void* taskContext);

    struct ExecuteDesc
    {
        ParallelForFunction parallelFor = nullptr;   // nullptr: segments are recorded on the calling thread
        void* schedulerContext = nullptr;
        uint32_t workerCount = 0;
        bool timers = false;                         // per-pass GPU timers (11.3)
    };

    // --------------------------------------------------------------------------------- compiled plan (11.3)

    enum class BarrierReason : uint8_t
    {
        RAW,
        WAR,
        WAW,
        UavOrder,
        ReadGroupChange,
        Alias,
        QueueTransfer,
        Discard,
        ExitState,
        Persistent,
        Debug,            // BarrierAfterEveryPass
    };

    // Where a batch sits around a pass: before the body, after the clears of a Clear access (6.9), after the body.
    enum class BatchSlot : uint8_t { Prologue, PostClear, Epilogue };

    enum class LoadOp : uint8_t { Load, Clear, DontCare };
    enum class StoreOp : uint8_t { Store, DontCare };

    // A barrier of the plan, device-free: resources are graph indices (GetResourceIndex) of the compiled frame. The
    // executor lowers the plan's barriers to nvrhi::ResourceBarrier arrays once, when it instantiates the plan, and
    // per frame patches only the slots of imports and history layers (10.1). The RPS counterpart is the transition
    // command of its compiled command stream (rpsRenderGraphGetDiagnosticInfo, RpsCmdDiagnosticInfo::transition).
    struct PlanBarrier
    {
        nvrhi::BarrierType type = nvrhi::BarrierType::Transition;
        nvrhi::BarrierFlags flags = nvrhi::BarrierFlags::None;
        BarrierReason reason = BarrierReason::RAW;
        nvrhi::CommandQueue otherQueue = nvrhi::CommandQueue::Graphics;   // QueueRelease / QueueAcquire
        uint32_t resource = ~0u;          // Transition, Uav (~0u: every UAV access); Aliasing: the incoming resource
        uint32_t aliasBefore = ~0u;       // Aliasing: the outgoing resource, ~0u when several are overlapped
        nvrhi::SubresourceRegion region;
        nvrhi::ResourceStates before = nvrhi::ResourceStates::Unknown;
        nvrhi::ResourceStates after = nvrhi::ResourceStates::Unknown;
        nvrhi::ShaderType stagesBefore = nvrhi::ShaderType::None;
        nvrhi::ShaderType stagesAfter = nvrhi::ShaderType::None;
    };

    struct BarrierBatch
    {
        const PlanBarrier* barriers = nullptr;
        uint32_t count = 0;
    };

    struct ScheduledPass
    {
        PassId pass = InvalidPassId;
        const char* name = nullptr;
        nvrhi::CommandQueue queue = nvrhi::CommandQueue::Graphics;
        uint32_t segment = 0;
        bool enabled = true;          // false: disabled by SetPassEnabled, or culled through a Cull policy
    };

    struct ResourceLifetime
    {
        uint32_t firstPass = ~0u;     // schedule indices; ~0u: unused (culled)
        uint32_t lastPass = ~0u;
        uint32_t heap = ~0u;          // Placed tier; ~0u otherwise
        uint64_t heapOffset = 0;
        uint64_t size = 0;
        bool crossQueue = false;
        bool aliased = false;
    };

    struct PlanStatistics
    {
        uint32_t scheduledPasses = 0;
        uint32_t culledPasses = 0;
        uint32_t segments = 0;
        uint32_t barriers = 0;            // every PlanBarrier in every batch
        uint32_t transitions = 0;
        uint32_t splitBarriers = 0;       // SplitBegin halves
        uint32_t uavBarriers = 0;
        uint32_t aliasingBarriers = 0;
        uint32_t queueTransfers = 0;      // release/acquire pairs
        uint32_t fenceWaits = 0;
        uint32_t fenceSignals = 0;
        uint32_t heaps = 0;
        uint64_t poolableTransientBytes = 0;   // the sum over transients, excluding imports and histories
        uint64_t placedPeakBytes = 0;          // the placement estimate, on every backend (7.5)
        uint64_t compileMicroseconds = 0;
        bool cacheHit = false;
    };

    enum class DumpFormat : uint8_t { Json, Graphviz };

    NVRHI_IID(ICompiledRenderGraph, "849c54d3-e27b-4369-9521-c20bc3639957")
    struct ICompiledRenderGraph : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ICompiledRenderGraph)
        virtual uint64_t GetPlanHash() noexcept = 0;            // structural hash of the graph it was compiled from
        virtual uint64_t GetPlanEnableMaskHash() noexcept = 0;
        virtual uint32_t GetScheduledPassCount() noexcept = 0;
        // FE_INVALID_ARGS for an index at or above GetScheduledPassCount.
        virtual nvrhi::FRESULT GetScheduledPass(uint32_t index, ScheduledPass& out) noexcept = 0;
        virtual nvrhi::FRESULT GetBatch(uint32_t index, BatchSlot slot, BarrierBatch& out) noexcept = 0;
        // Per attachment of a raster pass: colour slots 0..7, then the depth target as slot 8.
        virtual nvrhi::FRESULT GetAttachmentOps(uint32_t index, uint32_t attachment, LoadOp& load, StoreOp& store) noexcept = 0;
        // By graph resource index (GetResourceIndex) of the compiled frame.
        virtual nvrhi::FRESULT GetResourceLifetime(uint32_t resource, ResourceLifetime& out) noexcept = 0;
        virtual void GetStatistics(PlanStatistics& out) noexcept = 0;
        virtual nvrhi::FRESULT WriteDump(DumpFormat format, nvrhi::string& out) noexcept = 0;
    };

    // --------------------------------------------------------------------------------- registry (4)

    struct RegistryStatistics
    {
        uint32_t cachedPlans = 0;
        uint64_t planHits = 0;
        uint64_t planMisses = 0;
        uint32_t persistentResources = 0;
        uint32_t pooledResources = 0;
        uint32_t heaps = 0;
        uint64_t heapBytes = 0;
        uint64_t pooledBytes = 0;
        uint64_t framesExecuted = 0;
    };

    // What survives between frames: persistent resources and their cross-frame states, the plan cache, transient
    // heaps and pools, framebuffer and binding caches, per-queue fences and retirement.
    NVRHI_IID(IResourceRegistry, "639705d2-183f-43ee-8c37-1b10c11af515")
    struct IResourceRegistry : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IResourceRegistry)
        virtual nvrhi::IDevice* GetDevice() noexcept = 0;   // nullptr: record and compile only. Not AddRef'd
        // Drops pooled and placed memory and the cached plans, after the GPU has finished with them; the passes
        // of the cached plans get OnResourcesReleased. Persistent resources stay.
        virtual void ReleaseResources() noexcept = 0;
        // Drops one persistent resource or history. FE_NOT_FOUND when there is none of that name.
        virtual nvrhi::FRESULT ReleasePersistent(const char* name) noexcept = 0;
        virtual void ClearPlanCache() noexcept = 0;
        virtual void GetRegistryStatistics(RegistryStatistics& out) noexcept = 0;
    };

    // --------------------------------------------------------------------------------- pass interfaces

    // The accesses of one pass use, declared in IGpuPass::Setup (13.2). Valid only until that Setup returns; a
    // call after that is a validation error reported by Compile, and Write returns InvalidResourceId.
    NVRHI_IID(ISetupContext, "a1957fde-d0da-4eb9-9d03-29a0abe6fe03")
    struct ISetupContext : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ISetupContext)
        // Scratch that lives only inside this pass use (5.2): no other pass may name it. The pass still declares
        // its accesses below; the first one must be a FullWrite or a Clear.
        virtual ResourceId CreateTransientTexture(const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept = 0;
        virtual ResourceId CreateTransientBuffer(const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept = 0;
        // Reads do not create versions. The region is ignored for buffers; {} is the whole resource. Stages
        // None: derived from the pass type.
        virtual void Read(ResourceId resource, Access access, nvrhi::ShaderType stages,
                          const nvrhi::SubresourceRegion& region) noexcept = 0;
        // Returns the new version.
        virtual ResourceId Write(ResourceId resource, Access access, nvrhi::ShaderType stages,
                                 const nvrhi::SubresourceRegion& region, AccessFlags flags) noexcept = 0;
        virtual ResourceId SetColorTarget(uint32_t slot, ResourceId texture, const nvrhi::SubresourceRegion& region,
                                          AccessFlags flags, _In_opt_ const ClearValue* clear) noexcept = 0;
        virtual ResourceId SetDepthTarget(ResourceId texture, bool readOnly, AccessFlags flags,
                                          _In_opt_ const ClearValue* clear) noexcept = 0;
        // What consumers of `written` see when this pass is disabled (13.5).
        virtual void SetDisabledOutput(ResourceId written, DisabledOutput policy, ResourceId substituteOrInput) noexcept = 0;
    };

    // Serial, before recording: pipelines and static uploads (13.1).
    NVRHI_IID(IPrepareContext, "b6d55cce-3e33-4bcc-b684-b2606028f9b9")
    struct IPrepareContext : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IPrepareContext)
        virtual nvrhi::IDevice* GetDevice() noexcept = 0;
        virtual nvrhi::ICommandList* GetUploadCommandList() noexcept = 0;   // first list of the first graphics submit
        virtual const nvrhi::FramebufferInfoEx& GetFramebufferInfo() noexcept = 0;   // raster passes
        virtual const nvrhi::TextureDesc* GetTextureDesc(ResourceId id) noexcept = 0;   // nullptr for a buffer
        virtual const nvrhi::BufferDesc* GetBufferDesc(ResourceId id) noexcept = 0;     // nullptr for a texture
    };

    NVRHI_IID(IPassContext, "38dba484-cfea-4c57-b1a8-f551f16b3cef")
    struct IPassContext : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IPassContext)
        virtual nvrhi::ICommandList* GetCommandList() noexcept = 0;
        virtual nvrhi::IResource* GetResource(ResourceId id) noexcept = 0;
        virtual nvrhi::ITexture* GetTexture(ResourceId id) noexcept = 0;    // nullptr for a buffer
        virtual nvrhi::IBuffer* GetBuffer(ResourceId id) noexcept = 0;      // nullptr for a texture
        virtual nvrhi::IFramebuffer* GetFramebuffer() noexcept = 0;         // cached per allocation generation
        virtual bool IsNewAllocation(ResourceId id) noexcept = 0;
        virtual uint64_t GetFrameIndex() noexcept = 0;
    };

    // The base of every donut pass (13.1).
    NVRHI_IID(IGpuPass, "93b8a782-f116-45b2-b56a-d779d00dfe36")
    struct IGpuPass : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IGpuPass)
        // Declares the accesses of one use through `context`; `data` is that use's frame-arena record. Called
        // only by IRenderGraph::AddPass. FE_UNSUPPORTED: this pass cannot run in a graph yet.
        virtual nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept = 0;
        // Serial, before any recording. Creates pipelines and records static uploads. Must not write volatile
        // constant buffers.
        virtual nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept = 0;
        // Records that use. May run on a worker thread.
        virtual nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept = 0;
        // The registry released the resources this pass's caches may reference (resize, plan eviction). Drop
        // framebuffers and binding sets that point at them.
        virtual void OnResourcesReleased() noexcept = 0;
    };

    // --------------------------------------------------------------------------------- the graph

    // The frame's resources, passes and accesses. Bound to one registry and reused across frames. Declaring a
    // resource records a description; no GPU object is created until a plan is instantiated.
    NVRHI_IID(IRenderGraph, "dd5ebc18-e2b7-4efa-a9ac-72d18173b0f0")
    struct IRenderGraph : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IRenderGraph)
        virtual IResourceRegistry* GetResourceRegistry() noexcept = 0;     // not AddRef'd
        virtual void Reset() noexcept = 0;                                 // starts a frame; resets the arena

        // Resources (5.2). The ids are valid until the next Reset.
        virtual ResourceId CreateTexture(const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept = 0;
        virtual ResourceId CreateBuffer(const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept = 0;
        virtual ResourceId ImportTexture(nvrhi::ITexture* texture, nvrhi::ResourceStates entry,
                                         nvrhi::ResourceStates exit) noexcept = 0;
        virtual ResourceId ImportHostKeptTexture(nvrhi::ITexture* texture) noexcept = 0;
        virtual ResourceId ImportBuffer(nvrhi::IBuffer* buffer, nvrhi::ResourceStates entry,
                                        nvrhi::ResourceStates exit) noexcept = 0;
        // Device-free imports, for tests and rdg_bench: a stand-in described by `desc`. Execute rejects a plan
        // that holds one (FE_INVALID_ARGS).
        virtual ResourceId ImportTextureDesc(const nvrhi::TextureDesc& desc, nvrhi::ResourceStates entry,
                                             nvrhi::ResourceStates exit) noexcept = 0;
        virtual ResourceId ImportBufferDesc(const nvrhi::BufferDesc& desc, nvrhi::ResourceStates entry,
                                            nvrhi::ResourceStates exit) noexcept = 0;
        // Kept by the registry under `name`; a new id for the same resource each frame.
        virtual ResourceId GetPersistentTexture(const char* name, const nvrhi::TextureDesc& desc) noexcept = 0;
        virtual ResourceId GetPersistentBuffer(const char* name, const nvrhi::BufferDesc& desc) noexcept = 0;
        // `layerCount` ids, 2..c_MaxHistoryLayers: ids[0] is this frame's layer (writable), ids[k] was written k
        // frames ago (read-only) (5.2). FE_INVALID_ARGS for another layer count.
        virtual nvrhi::FRESULT GetHistory(const char* name, const nvrhi::TextureDesc& desc, uint32_t layerCount,
                                          ResourceId* ids) noexcept = 0;
        // Keeps a transient past the end of the graph; `outTexture` is set when the graph executes.
        virtual void Extract(ResourceId texture, nvrhi::ResourceStates exit, nvrhi::ITexture** outTexture) noexcept = 0;

        // Passes. AddPass calls pass->Setup(context, data) with a setup context for the new pass, and holds a
        // reference to `pass` until Reset; `data` must come from AllocateFrameData (or be nullptr).
        virtual nvrhi::FRESULT AddPass(const PassDesc& desc, IGpuPass* pass, void* data,
                                       _Out_opt_ PassId* outPass) noexcept = 0;
        // Not part of the structural hash (7.7, 13.5).
        virtual void SetPassEnabled(PassId pass, bool enabled) noexcept = 0;
        virtual void PushMarker(const char* name) noexcept = 0;
        virtual void PopMarker() noexcept = 0;
        // Frame arena. Released at Reset without running destructors: use NewFrameData (below).
        virtual void* AllocateFrameData(size_t size, size_t alignment) noexcept = 0;

        // Compiles the recorded frame, or returns the cached plan for its structural hash (7.7), and freezes the
        // graph until the next Reset. No device calls.
        virtual nvrhi::FRESULT Compile(const BackendProfile& profile, const CompileOptions& options,
                                       ICompiledRenderGraph** outPlan) noexcept = 0;
        // Runs a plan that Compile returned for this frame (9.1). FE_UNSUPPORTED when the registry has no device;
        // FE_INVALID_ARGS for a plan from another graph or frame.
        virtual nvrhi::FRESULT Execute(ICompiledRenderGraph* plan, const ExecuteDesc& desc) noexcept = 0;

        virtual uint64_t GetStructuralHash() noexcept = 0;
        virtual uint64_t GetEnableMaskHash() noexcept = 0;
        virtual uint32_t GetPassCount() noexcept = 0;
        virtual uint32_t GetResourceCount() noexcept = 0;
        virtual uint64_t GetFrameIndex() noexcept = 0;     // frames started with Reset since the graph was created
        // The validation errors and warnings of the last Compile, one per line (empty when there were none).
        virtual void GetDiagnostics(nvrhi::string& out) noexcept = 0;
    };

    // The two-layer history that TAA uses. Inline, not virtual.
    inline nvrhi::FRESULT GetHistoryPair(IRenderGraph* graph, const char* name, const nvrhi::TextureDesc& desc,
                                         ResourceId* current, ResourceId* previous) noexcept
    {
        ResourceId ids[2] = { InvalidResourceId, InvalidResourceId };
        const nvrhi::FRESULT result = graph->GetHistory(name, desc, 2, ids);
        *current = ids[0];
        *previous = ids[1];
        return result;
    }

    // Places one pass use's data in the frame arena. Inline, not virtual: it is outside the ABI.
    template <typename T, typename... Args>
    T* NewFrameData(IRenderGraph* graph, Args&&... args) noexcept
    {
        static_assert(std::is_trivially_destructible_v<T>,
                      "frame data is released at Reset without running destructors");
        void* memory = graph->AllocateFrameData(sizeof(T), alignof(T));
        return memory ? new (memory) T{ std::forward<Args>(args)... } : nullptr;
    }

    // `device` may be nullptr: such a registry supports recording and compiling only (tests, rdg_bench).
    nvrhi::FRESULT CreateResourceRegistry(nvrhi::IDevice* device, IResourceRegistry** out) noexcept;
    // One graph per registry: FE_INVALID_ARGS if `registry` already has one.
    nvrhi::FRESULT CreateRenderGraph(IResourceRegistry* registry, IRenderGraph** out) noexcept;
    nvrhi::FRESULT ComputeBackendProfile(nvrhi::IDevice* device, BackendProfile* out) noexcept;
    nvrhi::FRESULT GetReferenceProfile(ReferenceProfile kind, BackendProfile* out) noexcept;

    // The device-free verifier (11.1): replays `plan` against a simulated per-subresource state machine for two
    // consecutive frames, or for `previous` followed by `plan` (a plan switch, 6.12) when `previous` is set.
    // `graph` must still hold the frame `plan` was compiled from. FS_OK when every rule holds; otherwise
    // FE_GENERIC_ERROR, with one line per violation in `report`.
    nvrhi::FRESULT VerifyPlan(ICompiledRenderGraph* plan, IRenderGraph* graph, _In_opt_ ICompiledRenderGraph* previous,
                              _In_opt_ nvrhi::string* report) noexcept;
}

#undef DONUT_RENDER_GRAPH_FLAG_OPERATORS
