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

// The world: entities, archetypes, chunks, columns, sparse sets, resources (aggregate docs/ecs-design.md, sections
// 4 and 9.1). IWorld is data with primitive accessors (rule 3.3.7); systems, observers and services operate on it.
// Interfaces follow nvrhi's ABI rules (com-interface-abi.md).

#pragma once

#include <donut/engine/EcsCommandBuffer.h>
#include <donut/engine/EcsTypes.h>

namespace donut::engine
{
    struct IQuery;   // EcsQuery.h

    // ------------------------------------------------------------------------------------------- observers (6.6)

    enum class ObserverId : uint32_t { Invalid = 0xffffffffu };

    enum class ObserverEvent : uint8_t
    {
        OnAdd,       // after the component was added and constructed or set
        OnRemove,    // before the component's value is destroyed: the batch still reads it
        OnSet,       // after ICommandBuffer::SetComponents wrote it
        OnMatch,     // the entity started to match the filter terms (no observed component)
        OnUnmatch,   // the entity stops matching them; delivered before the change, so the batch still reads it
    };

    // A run of consecutive rows of one chunk, delivered to an observer or a simulation domain (8.1).
    struct EntityBatch
    {
        const Entity* entities;
        void* const* columns;      // the observed component first (none for OnMatch/OnUnmatch), then one per data
                                   // term of the filter, in term order; null for a tag
        uint32_t count;
        uint32_t archetypeIndex;   // for diagnostics (IWorld::GetArchetypeInfo)
    };
    static_assert(sizeof(void*) != 8 || sizeof(EntityBatch) == 24, "EntityBatch layout");

    struct ObserverDesc
    {
        const char* debugName;
        const QueryTerm* filterTerms;   // terms the entity must match; their data terms add batch columns. For
                                        // OnMatch/OnUnmatch they are the query whose membership is observed
        uint32_t filterTermCount;
        ComponentId component;          // the observed component; Invalid for OnMatch/OnUnmatch
        ObserverEvent event;
        Access access;                  // to the observed component's column: Read, or Write for observers that fill
                                        // in derived values (a GPU slot, an engine index); Filter (no column) only
                                        // for a tag. Ignored for OnMatch/OnUnmatch
        uint8_t reserved[6];
    };
    static_assert(sizeof(void*) != 8 || sizeof(ObserverDesc) == 32, "ObserverDesc layout");

    // Delivery: single-threaded, in playback order (or at once for an immediate structural change), one batch per run of
    // rows in a chunk. Removals (OnRemove, OnUnmatch) are delivered before each row changes, so a destruction or a move
    // gives one batch per entity; a LoadSnapshot removes whole chunks.
    //
    // AddObserver fails with FE_INVALID_ARGS (and a validation report) for: filter terms on Sparse components, Changed,
    // Added, Enabled and Disabled terms (filters match per archetype), a filter that excludes the observed component,
    // OnSet on Buffer and Chunk components, Write on a Shared component, Filter access on a component with data,
    // OnMatch/OnUnmatch with a component or without terms. AddObserver and RemoveObserver fail with FE_INVALID_ARGS
    // inside a phase (unless an exclusive system runs) and while an observer runs.
    //
    // Re-entrancy: inside OnEntities an observer may read the world (IsAlive, HasComponent, GetComponent, GetResource,
    // queries), write the batch's Write columns and, with GetComponentForWrite, values of other entities, and record
    // into `commands`. Immediate structural changes, snapshots, and adding or removing observers fail with
    // FE_INVALID_ARGS while an observer runs; Playback and registering a new component type fail with FE_OUT_OF_DATE.
    //
    // `commands` is valid only during the callback. It is played back after every observer of the triggering change has
    // run: before an immediate structural call returns, or before the sync point ends. Its changes may trigger further
    // observers; the cascade repeats in order until no commands remain, and fails with FE_GENERIC_ERROR after 64
    // rounds.
    // After a LoadSnapshot, observer and cascade failures are reported through validation, not returned: the load has
    // already replaced the content. Commands that OnRemove observers record while a load clears the world are played
    // back after the load, against the loaded entities.
    NVRHI_IID(IObserver, "6603d0ec-8699-4cb9-a992-1e64923ae188")
    struct IObserver : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IObserver)
        virtual nvrhi::FRESULT OnEntities(ObserverEvent event, const EntityBatch& batch,
                                          ICommandBuffer* commands) noexcept = 0;
    };

    // ------------------------------------------------------------------------------------------ snapshots (11.4)

    NVRHI_IID(IStateWriter, "73779b0d-f63e-4ef8-812a-9bcdf2f97190")
    struct IStateWriter : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IStateWriter)
        virtual nvrhi::FRESULT Write(_In_reads_bytes_(size) const void* data, uint64_t size) noexcept = 0;
    };

    NVRHI_IID(IStateReader, "3e4c2814-ad3d-4a85-ae14-f4309c314a63")
    struct IStateReader : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IStateReader)
        // FE_INVALID_ARGS when fewer than `size` bytes remain.
        virtual nvrhi::FRESULT Read(uint64_t size, _Out_writes_bytes_(size) void* data) noexcept = 0;
    };

    // ------------------------------------------------------------------------------------------------- world

    struct WorldDesc
    {
        const char* debugName = nullptr;
        uint32_t chunkSize = 16 * 1024;          // D6
        uint32_t largeChunkSize = 64 * 1024;     // for archetypes whose rows exceed largeRowThreshold bytes
        uint32_t largeRowThreshold = 128;
        uint32_t initialEntityCapacity = 1024;
        bool validation = false;                 // the checks of 11.3, equal sort keys across buffers (6.5)
        uint8_t reserved[7] = {};
    };
    static_assert(sizeof(void*) != 8 || sizeof(WorldDesc) == 32, "WorldDesc layout");

    struct WorldStats
    {
        uint32_t entityCount;
        uint32_t entityCapacity;
        uint32_t componentTypeCount;
        uint32_t archetypeCount;
        uint32_t chunkCount;
        uint32_t sparseSetCount;
        uint64_t chunkBytes;            // reserved by chunks
        uint64_t rowBytes;              // occupied by rows
        uint64_t bufferHeapBytes;       // dynamic buffers that spilled to the heap
        uint64_t commandsPlayedBack;    // since creation
        uint64_t observerBatches;       // since creation
        uint64_t version;
        uint64_t structuralVersion;
    };
    static_assert(sizeof(WorldStats) == 80, "WorldStats layout");

    struct ArchetypeInfo
    {
        const ComponentId* components;  // sorted; valid until the next structural change
        uint32_t componentCount;
        uint32_t entityCount;
        uint32_t chunkCount;
        uint32_t chunkSize;
        uint32_t rowSize;
        uint32_t rowsPerChunk;
    };
    static_assert(sizeof(void*) != 8 || sizeof(ArchetypeInfo) == 32, "ArchetypeInfo layout");

    NVRHI_IID(IWorld, "c2bb40bd-b7a3-44b9-bd7e-51467522f190")
    struct IWorld : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IWorld)

        // Component types (4.2). Registering a GUID again returns its id when the layout matches, FE_INVALID_ARGS when
        // it does not. A new type (here and in RegisterResource) is refused with FE_OUT_OF_DATE inside a phase, unless
        // an exclusive system runs, and while an observer runs: systems may be reading the registry. See
        // ComponentTypeDesc for what must outlive the world.
        virtual nvrhi::FRESULT RegisterComponent(const ComponentTypeDesc& desc, ComponentId& outId) noexcept = 0;
        virtual ComponentId FindComponent(const nvrhi::GUID& uuid) noexcept = 0;
        virtual const ComponentTypeDesc* GetComponentTypeDesc(ComponentId component) noexcept = 0;   // not owned
        virtual uint32_t GetComponentTypeCount() noexcept = 0;

        // Resources (4.9): a descriptor with ComponentFlags::Resource; the value is constructed, or copied from
        // `initialValue`. Access is declared like component access.
        virtual nvrhi::FRESULT RegisterResource(const ComponentTypeDesc& desc, _In_opt_ const void* initialValue,
                                                ComponentId& outId) noexcept = 0;
        virtual void* GetResource(ComponentId resource) noexcept = 0;

        // Immediate structural changes: outside phases, or from an exclusive system (rule 3.3.3). Rows are complete
        // (values in place) before observers run, and observers run before the call returns.
        //
        // ReserveEntities hands out ids that are not alive yet, so that the rows of one CreateEntities call can name
        // each other (Parent, SceneLinks of an attached subgraph, design 5.3). A reserved id stays reserved until
        // CreateEntities uses it or DestroyEntities releases it.
        virtual nvrhi::FRESULT ReserveEntities(uint32_t count, _Out_writes_(count) Entity* outEntities) noexcept = 0;
        // outEntities may be null when desc.reservedEntities names the ids.
        virtual nvrhi::FRESULT CreateEntities(const EntityCreateDesc& desc,
                                              _Out_writes_opt_(desc.entityCount) Entity* outEntities) noexcept = 0;
        virtual nvrhi::FRESULT DestroyEntities(_In_reads_(count) const Entity* entities, uint32_t count) noexcept = 0;
        virtual nvrhi::FRESULT AddComponent(Entity entity, ComponentId component,
                                            _In_opt_ const void* value) noexcept = 0;
        virtual nvrhi::FRESULT RemoveComponent(Entity entity, ComponentId component) noexcept = 0;
        virtual nvrhi::FRESULT Instantiate(Entity prefab, uint32_t count,
                                           _Out_writes_(count) Entity* outEntities) noexcept = 0;   // 4.10
        // Copies rows from another world, matching component types by GUID (design 5.3). Entity fields that name an
        // entity of the copied set are remapped to its copy; others are copied as they are. FE_NOT_FOUND when a
        // component type is not registered here.
        virtual nvrhi::FRESULT CopyEntitiesFrom(IWorld* source, _In_reads_(count) const Entity* entities,
                                                uint32_t count, _Out_writes_(count) Entity* outEntities) noexcept = 0;
        // The entity a command-buffer placeholder became; Null if it was destroyed or never played back. Valid from the
        // end of the Playback that created it until the next Playback starts.
        virtual Entity ResolvePlaceholder(Entity placeholder) noexcept = 0;

        // Per-entity access: a lookup in the entity table, for code outside the hot path.
        virtual bool IsAlive(Entity entity) noexcept = 0;
        virtual bool HasComponent(Entity entity, ComponentId component) noexcept = 0;
        virtual const void* GetComponent(Entity entity, ComponentId component) noexcept = 0;   // null if absent
        // Stamps the entity's chunk column with `writeVersion`: the running system's version (ISystemContext), or
        // c_WorldVersion outside phases.
        virtual void* GetComponentForWrite(Entity entity, ComponentId component, uint64_t writeVersion) noexcept = 0;
        virtual nvrhi::FRESULT SetEnabled(Entity entity, ComponentId component, bool enabled) noexcept = 0;
        virtual bool IsEnabled(Entity entity, ComponentId component) noexcept = 0;
        // Not structural; safe from parallel systems for distinct entities. Stamps like GetComponentForWrite.
        virtual nvrhi::FRESULT ResizeBuffer(Entity entity, ComponentId component, uint32_t count,
                                            uint64_t writeVersion) noexcept = 0;

        // Queries (6.1), command buffers (6.5), observers (6.6).
        virtual nvrhi::FRESULT CreateQuery(const QueryDesc& desc, IQuery** ppQuery) noexcept = 0;
        virtual nvrhi::FRESULT CreateCommandBuffer(ICommandBuffer** ppBuffer) noexcept = 0;
        // Plays back the buffers of one system: merged by sort key, stable within a buffer (EcsCommandBuffer.h), then
        // the observers' commands. Every buffer played back is reset. Returns the failure of the earliest failing
        // command in merged order, as if the commands were applied one by one; the other commands still apply.
        // - An entry that cannot be played back (null, a buffer of another world, listed again, or being recorded by
        //   another thread) is skipped and left as it is, and the call returns FE_INVALID_ARGS; the other buffers still
        //   apply and are reset. A buffer listed twice is played back once.
        // - Inside a phase, unless an exclusive system runs, and while an observer runs, the call returns
        //   FE_OUT_OF_DATE and does nothing: no command applies, no buffer is reset.
        virtual nvrhi::FRESULT Playback(_In_reads_(count) ICommandBuffer* const* buffers, uint32_t count) noexcept = 0;
        virtual nvrhi::FRESULT AddObserver(const ObserverDesc& desc, IObserver* observer,
                                           ObserverId& outId) noexcept = 0;
        virtual nvrhi::FRESULT RemoveObserver(ObserverId observer) noexcept = 0;

        // Versions (4.5). GetVersion is the version that writes outside phases stamp. AdvanceVersion hands out the next
        // one; the scheduler calls it once per system run, in schedule order, and once after each phase.
        virtual uint64_t GetVersion() noexcept = 0;
        virtual uint64_t AdvanceVersion() noexcept = 0;
        virtual uint64_t GetStructuralVersion() noexcept = 0;
        // True while a scheduler phase runs (the scene-graph API rule, design 5.6).
        virtual bool IsInPhase() noexcept = 0;

        // Snapshots and diagnostics (11.1, 11.4). A snapshot holds the serialisable components (Relocatable, not
        // NoSerialize); loading replaces the world's content and keeps entity ids, so references stay valid. Component
        // ids are matched by GUID. ComputeHash covers entities, archetypes and the field bytes of serialisable columns.
        virtual nvrhi::FRESULT SaveSnapshot(IStateWriter* writer) noexcept = 0;
        // Observers see a load as the removal of every entity, then the addition of the snapshot's. Once the content is
        // replaced the result is FS_OK: failures of observers and of their cascade are reported through validation.
        virtual nvrhi::FRESULT LoadSnapshot(IStateReader* reader) noexcept = 0;
        virtual uint64_t ComputeHash() noexcept = 0;
        virtual WorldStats& GetStats(WorldStats& retVal) noexcept = 0;
        virtual uint32_t GetArchetypeCount() noexcept = 0;
        virtual nvrhi::FRESULT GetArchetypeInfo(uint32_t archetype, ArchetypeInfo& out) noexcept = 0;
    };

    nvrhi::FRESULT CreateWorld(const WorldDesc& desc, IWorld** ppWorld) noexcept;

    // ------------------------------------------------- typed helpers (header-only, outside the ABI surface)

    // Registers T if needed; Invalid on a layout clash, or for a new type where IWorld::RegisterComponent refuses one.
    template <typename T>
    ComponentId RegisterComponent(IWorld* world) noexcept
    {
        ComponentId id = ComponentId::Invalid;
        world->RegisterComponent(ComponentTraits<T>::Desc(), id);
        return id;
    }

    template <typename T>
    ComponentId FindComponent(IWorld* world) noexcept
    {
        return world->FindComponent(ComponentTraits<T>::Desc().uuid);
    }

    // A GUID lookup per call: cache the id in hot code, or use a query.
    template <typename T>
    const T* GetComponent(IWorld* world, Entity entity) noexcept
    {
        return static_cast<const T*>(world->GetComponent(entity, FindComponent<T>(world)));
    }

    template <typename T>
    T* GetComponentForWrite(IWorld* world, Entity entity, uint64_t writeVersion = c_WorldVersion) noexcept
    {
        return static_cast<T*>(world->GetComponentForWrite(entity, FindComponent<T>(world), writeVersion));
    }

    template <typename T>
    T* GetResource(IWorld* world) noexcept
    {
        return static_cast<T*>(world->GetResource(FindComponent<T>(world)));
    }
}
