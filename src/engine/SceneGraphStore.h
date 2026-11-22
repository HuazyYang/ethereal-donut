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

// The scene graph over the world (aggregate docs/ecs-design.md section 5, 4.6, 4.7; plan step E3), private to
// donut_engine: orphan records, the graph's storage (world, scheduler, systems, the leaf observer), and the structural
// operations that SceneGraph and SceneGraphNode forward to it.
//
// Rules:
// - One world per graph (D18). The graph's API runs on one thread at a time, outside phases (5.6); concurrent readers
//   (views walking the graph in parallel) are allowed between structural changes. Readers never write the store except
//   the on-demand caches, which are filled under m_CacheMutex and published per node with release order.
// - Structural operations create and destroy entities in pre-order, in batches of consecutive nodes with the same
//   components, so that entity ids, row order and the leaf observer's order are deterministic (5.3, 5.5).
// - Propagation and Bounds read the archetype storage directly (EcsWorldImpl.h): they test chunk versions per column
//   (D9) and the parent's chunk version per row, which no public query exposes, and they stamp only the chunks they
//   write. They run as systems of the graph's scheduler, in the TransformPropagation and Bounds phases. GpuScene reads
//   the mesh-instance chunks the same way (plan step E4).
// - Asset tables and slots (step E4): the leaf observer fills a mesh instance's MeshRef, RenderFlags and
//   GpuInstanceSlot columns (table references, stable slots) before it calls RegisterLeaf, and releases them before it
//   calls UnregisterLeaf. Every structural operation and InvalidateContent first advances the world's version, so that
//   what they stamp is newer than any version a reader (GpuScene) took before them.

#pragma once

#include <donut/engine/SceneGraph.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>

#include "EcsWorldImpl.h"

#include <array>
#include <atomic>
#include <mutex>
#include <vector>

namespace donut::engine
{
    // The state of a node that is in no graph (5.3). The node part is typed (it is the same for every node, and attach
    // moves it with plain copies); leaf columns, keyed by GUID, join it with the first columnar leaf (step E4).
    class OrphanRecord
    {
    public:
        // Links, as the old node kept them: a node owns its first child and every child owns its next sibling; the
        // parent, last-child and previous-sibling links do not own. They and the leaf come first: a walk over a
        // subgraph reads only them.
        SceneGraphNode* parent = nullptr;
        nvrhi::AutoPtr<SceneGraphNode> firstChild;
        nvrhi::AutoPtr<SceneGraphNode> nextSibling;
        SceneGraphNode* lastChild = nullptr;
        SceneGraphNode* prevSibling = nullptr;
        nvrhi::AutoPtr<SceneGraphLeaf> leaf;
        bool hasLocal = false;

        // Component values in their world-independent form.
        LocalTransform local = { dm::double3(0.0), dm::dquat::identity(), dm::double3(1.0) };
        WorldTransform world = { dm::daffine3::identity() };

        // Records are recycled through a process-wide pool (bounded): attach frees one per node, detach and the
        // construction of orphaned subgraphs (on any thread) make them.
        static OrphanRecord* Create();
        static void Destroy(OrphanRecord* record) noexcept;
        static void Destroy(OrphanRecord* const* records, size_t count) noexcept;
    };

    class SceneGraphStore
    {
    public:
        // The columns that node accessors read; a per-archetype table maps them to column indices.
        enum Slot : uint32_t
        {
            SlotNode, SlotLinks, SlotParent, SlotLocal, SlotLocalMatrix, SlotWorld, SlotLeaf, SlotLocalBounds,
            SlotWorldBounds, SlotChunkBounds, SlotMeshRef, SlotRenderFlags, SlotInstance, SlotCount
        };
        static constexpr uint8_t c_Absent = 0xff;

        struct RowRef
        {
            const ecs::Archetype* archetype = nullptr;
            uint32_t archetypeIndex = 0;
            uint32_t chunk = 0;
            uint32_t row = 0;
        };

        explicit SceneGraphStore(SceneGraph* graph) noexcept;
        // Detaches the nodes that are referenced elsewhere into records (5.3); then the world goes.
        ~SceneGraphStore();

        SceneGraphStore(const SceneGraphStore&) = delete;
        SceneGraphStore& operator=(const SceneGraphStore&) = delete;

        SceneGraph* GetGraph() const noexcept { return m_Graph; }
        IWorld* GetWorld() const noexcept { return m_World; }
        IScheduler* GetScheduler() const noexcept { return m_Scheduler; }
        void SetJobSystem(IJobSystem* jobSystem) noexcept;

        // ----------------------------------------------------------------------------------- rows (5.2)
        // The accessors of every node run these: inline.
        uint8_t ColumnIndex(const ecs::Archetype& archetype, uint32_t archetypeIndex, Slot slot) const noexcept
        {
            if (archetypeIndex < m_ArchetypeColumns.size())
                return m_ArchetypeColumns[archetypeIndex][slot];
            // an archetype made since the last synchronisation (by code outside the graph): look it up, write nothing
            const uint32_t column = archetype.FindColumn(m_Ids[slot]);
            return column == ecs::c_NoColumn ? c_Absent : uint8_t(column);
        }
        bool Locate(Entity entity, RowRef& out) const noexcept
        {
            // WorldImpl::FindAlive, inline
            const uint32_t index = GetEntityIndex(entity);
            if (!m_Impl || index == 0 || index >= m_Impl->GetEntityTableSize())
                return false;
            const ecs::EntityRecord& record = m_Impl->GetEntityRecord(index);
            if (record.archetype >= ecs::c_ClaimedSlot || record.generation != GetEntityGeneration(entity))
                return false;
            out.archetypeIndex = record.archetype;
            out.archetype = &m_Impl->GetArchetypeStorage(record.archetype);
            out.chunk = record.chunk;
            out.row = record.row;
            return true;
        }
        // The row's value in a column; null where the archetype lacks it.
        void* Column(const RowRef& row, Slot slot) const noexcept
        {
            const uint8_t column = ColumnIndex(*row.archetype, row.archetypeIndex, slot);
            if (column == c_Absent)
                return nullptr;
            const ecs::ArchetypeColumn& c = row.archetype->columns[column];
            uint8_t* memory = row.archetype->chunks[row.chunk].memory;
            if (c.kind == ecs::StorageKind::Chunk)
                return memory + c.offset;
            return memory + c.offset + size_t(row.row) * c.stride;
        }
        template <typename T>
        T* Get(Entity entity, Slot slot) const noexcept
        {
            RowRef row;
            return Locate(entity, row) ? static_cast<T*>(Column(row, slot)) : nullptr;
        }
        // A node's row through its row hint when the hint is still valid (no entity-table lookup), else Locate.
        bool LocateNode(const SceneGraphNode* node, RowRef& out) const noexcept
        {
            if (node->m_HintMoves != ~uint64_t(0))
            {
                const ecs::Archetype& archetype = m_Impl->GetArchetypeStorage(node->m_HintArchetype);
                if (archetype.rowMoves == node->m_HintMoves)
                {
                    out.archetype = &archetype;
                    out.archetypeIndex = node->m_HintArchetype;
                    out.chunk = node->m_HintChunk;
                    out.row = node->m_HintRow;
                    return true;
                }
            }
            return Locate(node->m_Entity, out);
        }
        template <typename T>
        T* GetOf(const SceneGraphNode* node, Slot slot) const noexcept
        {
            RowRef row;
            return LocateNode(node, row) ? static_cast<T*>(Column(row, slot)) : nullptr;
        }
        static void RememberRow(SceneGraphNode* node, const RowRef& row) noexcept
        {
            node->m_HintArchetype = row.archetypeIndex;
            node->m_HintChunk = row.chunk;
            node->m_HintRow = row.row;
            node->m_HintMoves = row.archetype->rowMoves;
        }
        static void ForgetRow(SceneGraphNode* node) noexcept { node->m_HintMoves = ~uint64_t(0); }
        // Stamps the row's chunk column with the world's version, as GetComponentForWrite outside phases does.
        void Stamp(const RowRef& row, Slot slot) const noexcept
        {
            const uint8_t column = ColumnIndex(*row.archetype, row.archetypeIndex, slot);
            if (column != c_Absent)
                ecs::StampVersion(row.archetype->Versions(row.chunk) + column, m_Impl->GetStampVersion());
        }
        SceneGraphNode* NodeOf(Entity entity) const noexcept;
        // The GpuInstanceSlot of an attached mesh instance; null while it is in no graph or holds no slot (before the
        // leaf observer reached it, after it released it): MeshInstance::GetInstanceIndex.
        static const GpuInstanceSlot* InstanceSlotOf(const SceneGraphLeaf* leaf) noexcept;

        // ------------------------------------------------------------------------------- structure (5.3)
        // `child` is in no graph; `parent` is attached here, or null to make `child` the graph's root.
        nvrhi::AutoPtr<SceneGraphNode> AttachOrphan(SceneGraphNode* parent, SceneGraphNode* child);
        // `source` is attached to `sourceStore` (this store or another): a deep copy goes under `parent`.
        nvrhi::AutoPtr<SceneGraphNode> AttachCopy(SceneGraphNode* parent, SceneGraphNode* source,
                                                  SceneGraphStore* sourceStore);
        // `node` is attached here: its subgraph moves into records, its entities are destroyed.
        nvrhi::AutoPtr<SceneGraphNode> Detach(SceneGraphNode* node);
        void SetLeaf(SceneGraphNode* node, SceneGraphLeaf* leaf);
        void SetTransform(SceneGraphNode* node, const dm::double3* translation, const dm::dquat* rotation,
                          const dm::double3* scaling);
        void InvalidateContent(SceneGraphNode* node);

        // ------------------------------------------------------------------------------ refresh (5.4, 5.7)
        // Runs the phases when anything changed; returns whether structure changed since the last refresh.
        bool Refresh(uint32_t frameIndex);
        // After a Refresh that saw structural changes: the mesh table follows meshes edited in place (new geometry
        // counts get new geometry slots, and so do the geometry-instance slots of their instances).
        void ReconcileMeshes();
        bool HasPendingStructureChanges() const noexcept { return m_PendingStructure; }
        bool HasPendingTransformChanges() const noexcept { return m_PendingTransform || m_LastRefreshMoved; }

        // The on-demand caches of SceneGraphNode (5.7).
        struct CacheFrame
        {
            const SceneGraphNode* node;
            SceneGraphNode* nextChild;
            dm::box3 bounds;
            SceneContentFlags leafContent;
            SceneContentFlags content;
        };
        uint64_t GetEpoch() const noexcept { return m_Epoch.load(std::memory_order_relaxed); }
        void ComputeSubgraphCache(const SceneGraphNode* node);
        // The leaf's box and content flags, then each child's in sibling order, as the old Refresh unioned them. With
        // `publish`, every node computed gets its cache stamped with `epoch` (under m_CacheMutex).
        static void ComputeSubgraph(const SceneGraphNode* node, std::vector<CacheFrame>& stack, uint64_t epoch,
                                    bool publish, dm::box3& outBounds, SceneContentFlags& outLeafContent,
                                    SceneContentFlags& outContent);

        // Links of nodes in no graph (the old AppendChild and RemoveChild, on records).
        // Takes `child` out of its parent's list and hands over the reference the list held (null without a parent).
        static nvrhi::AutoPtr<SceneGraphNode> UnlinkOrphan(SceneGraphNode* child);
        // Links `child` as the last child of `parent`, taking it out of its current parent's list first.
        static void AppendOrphan(SceneGraphNode* parent, SceneGraphNode* child);

        // The systems' bodies (SceneGraphStore.cpp).
        nvrhi::FRESULT RunPropagation(ISystemContext* context) noexcept;
        nvrhi::FRESULT RunBounds(ISystemContext* context) noexcept;

        // ------------------------------------------------------------------------- GPU scene (step E4)
        // Read by GpuScene (a friend), which walks the mesh-instance chunks through the archetype storage.
        const SlotAllocator& GetInstanceSlots() const noexcept { return m_InstanceSlots; }
        const SlotAllocator& GetGeometryInstanceSlots() const noexcept { return m_GeometryInstanceSlots; }
        // Counts every Refresh call, also the ones with nothing to do: a GpuScene update after a new one is a new frame
        // for the previous transforms.
        uint64_t GetRefreshCalls() const noexcept { return m_RefreshCalls; }
        // Counts the ReconcileMeshes calls that gave a mesh new geometry slots.
        uint64_t GetLayoutSerial() const noexcept { return m_LayoutSerial; }
        // Unique per store in the process (an address can be reused).
        uint64_t GetSerial() const noexcept { return m_Serial; }

        // ------------------------------------------------------------------ render read (plan step E5.1)
        // SceneGraph::BeginRenderRead and EndRenderRead: the scopes the render service keeps open while a frame's
        // snapshot points at this graph's leaves and assets (open point 12). The count is kept in every build; only
        // Debug builds check it, in the API's write entry points (SceneGraph.cpp), so Release writes cost nothing.
        void BeginRenderRead() noexcept { ++m_RenderReads; }
        void EndRenderRead() noexcept
        {
            if (m_RenderReads != 0)
                --m_RenderReads;
        }
        bool IsInRenderRead() const noexcept { return m_RenderReads != 0; }
        // Logs one warning for a write through the API while a scope is open (and the write is not part of another
        // one, RenderReadQuiet). The write still applies.
        void WarnRenderRead(const char* operation) const noexcept;
        // SceneGraph::RunReadOnlyPhase: runs the phase; when it wrote nothing and nothing else ran on the world since the
        // last Refresh, moves m_RefreshedVersion past the phase's versions, so that the next Refresh may skip.
        nvrhi::FRESULT RunReadOnlyPhase(Phase phase) noexcept;
        // SceneGraph::ClaimRenderService and GetSchedulerGeneration.
        bool ClaimRenderService() noexcept { return !m_RenderServiceClaimed.exchange(true); }
        void ReleaseRenderService() noexcept { m_RenderServiceClaimed.store(false); }
        uint64_t GetSchedulerGeneration() const noexcept { return m_SchedulerGeneration; }
        // Inside one API call that makes others (SetRootNode: Detach, then Attach), so that it warns once.
        struct RenderReadQuiet
        {
            SceneGraphStore& store;
            explicit RenderReadQuiet(SceneGraphStore& s) noexcept : store(s) { ++store.m_RenderReadQuiet; }
            ~RenderReadQuiet() { --store.m_RenderReadQuiet; }
            RenderReadQuiet(const RenderReadQuiet&) = delete;
            RenderReadQuiet& operator=(const RenderReadQuiet&) = delete;
        };

        // Test seams: entity ids of the graph's components, and counters.
        ComponentId GetComponentId(Slot slot) const noexcept { return m_Ids[slot]; }
        ComponentId GetDepthId() const noexcept { return m_DepthId; }
        struct Counters
        {
            uint64_t refreshes = 0;            // Refresh calls that ran the phases
            uint64_t skippedRefreshes = 0;     // Refresh calls with nothing to do
            uint64_t propagatedChunks = 0;     // chunks propagation wrote
            uint64_t propagatedRows = 0;
            uint64_t boundsChunks = 0;         // chunks the Bounds system wrote
            uint64_t deepRows = 0;             // rows of the last depth bucket visited (c_MaxHierarchyDepth)
            uint64_t createCalls = 0;          // CreateEntities calls of attach and copy (one per batch)
        };
        const Counters& GetCounters() const noexcept { return m_Counters; }
        // Test seam: the CreateEntities call of attach number `calls` from now (0: the next) is refused by the world
        // as an out-of-memory failure would be (it consumes the values); negative disarms it.
        static void FailCreateAfter(int32_t calls) noexcept { s_FailCreateCountdown.store(calls); }
        // Test seam: the mesh-instance acquisition number `calls` from now throws std::bad_alloc before it allocates
        // anything (as the tables would out of memory); the leaf observer catches it. Negative disarms it.
        static void FailAcquireAfter(int32_t calls) noexcept { s_FailAcquireCountdown.store(calls); }

    private:
        // A node of a subgraph that is being attached, in pre-order. Its values are read from its record.
        struct NewNode
        {
            SceneGraphNode* node;      // one reference, handed to the world
            OrphanRecord* record;      // null for a node without state
            SceneGraphLeaf* leaf;      // one reference, handed to the world, or null
            uint32_t parent;           // index in the list; ~0u for the first node
            uint32_t leafType;         // index in m_LeafTypes, ~0u without a leaf
            dm::box3 localBounds;
            bool hasLocal;
        };

        struct LeafType
        {
            const LeafTypeDesc* desc;
            uint32_t firstTag;         // in m_LeafTypeTags: the tags, then the columns
            uint32_t tagCount;
            uint32_t columnCount;      // the class's columns (LeafTypeDesc::columns), after its tags
        };

        struct LevelArchetype
        {
            uint32_t archetype;
            uint8_t local;
            uint8_t matrix;
            uint8_t parent;
            uint8_t world;
        };

        struct BoundsArchetype
        {
            uint32_t archetype;
            uint8_t world;
            uint8_t localBounds;
            uint8_t worldBounds;
            uint8_t chunkBounds;
        };

        struct JointArchetype
        {
            uint32_t archetype;
            uint8_t local;
            uint8_t leaf;
        };

        struct Task
        {
            uint32_t archetype;        // index in the level or bounds list
            uint32_t chunk;
            bool whole;                // the chunk's own columns changed: every row
        };

        struct PropagationContext;
        struct BoundsContext;
        struct DeepContext;

        friend class GpuScene;

        void CreateSystems() noexcept;
        // The leaf observer's work per leaf (OnAdd, OnRemove), and the mesh-instance columns it fills and releases.
        void OnLeaf(Entity entity, SceneGraphLeaf* leaf, bool added);
        void AcquireInstance(const RowRef& row, MeshInstance* instance, MeshTable::Changes& changes);
        void ReleaseInstance(const RowRef& row, MeshTable::Changes& changes);
        void RaiseAssetCallbacks(const MeshTable::Changes& changes);
        void SyncArchetypes() noexcept;
        uint32_t ResolveLeafType(SceneGraphLeaf* leaf);
        uint32_t DepthOf(Entity entity) const noexcept;
        void ResolveLeaves();
        bool CreateNodes(Entity parentEntity, uint32_t parentDepth);   // false when the world refused a batch
        void AbandonNodes(uint32_t first);
        void LinkUnder(Entity parentEntity, Entity child);
        void GatherSubgraph(Entity root);
        void MoveToRecords(bool unlinkRoot);
        void BumpEpoch() noexcept { m_Epoch.fetch_add(1, std::memory_order_relaxed); }
        void UpdateSystemLists() noexcept;
        void PropagateRow(const LevelArchetype& level, const ecs::Archetype& archetype, uint32_t chunk, uint32_t row,
                          const dm::daffine3* parentWorld, bool compose) const noexcept;
        bool ParentWritten(Entity parent, uint64_t version, const dm::daffine3** outWorld,
                           bool readVersion) const noexcept;
        void PropagateDeepBucket(IJobSystem* jobs, uint64_t since, uint64_t version, bool parentLevelWrote);
        const LevelArchetype* DeepLevel(uint32_t archetypeIndex) const noexcept;
        static bool DeepChunkDirty(const LevelArchetype& level, const ecs::Archetype& archetype, uint32_t chunk,
                                   uint64_t since) noexcept;
        bool PropagateDeepRow(Entity entity, bool parentInBucket, uint64_t since, uint64_t version) noexcept;
        static void RunDeepTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        void MarkSkinnedJoints(uint64_t since);
        static void RunPropagationTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        static void RunBoundsTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;

        static std::atomic<int32_t> s_FailCreateCountdown;
        static std::atomic<int32_t> s_FailAcquireCountdown;

        SceneGraph* m_Graph;
        nvrhi::AutoPtr<IWorld> m_World;
        ecs::WorldImpl* m_Impl = nullptr;
        nvrhi::AutoPtr<IJobSystem> m_Jobs;
        nvrhi::AutoPtr<IScheduler> m_Scheduler;
        nvrhi::AutoPtr<IObserver> m_LeafObserver;
        ComponentId m_Ids[SlotCount] = {};
        ComponentId m_DepthId = ComponentId::Invalid;
        ComponentId m_JointTag = ComponentId::Invalid;   // SkinnedMeshReferenceLeaf, once a joint leaf was attached

        // Column indices per archetype, for the accessors; synchronised after structural changes.
        std::vector<std::array<uint8_t, SlotCount>> m_ArchetypeColumns;

        // Leaf classes seen so far and their tag and column ids.
        std::vector<LeafType> m_LeafTypes;
        std::vector<ComponentId> m_LeafTypeTags;

        // Stable slots of the mesh instances (step E4): instance-buffer rows and geometry-instance ranges.
        SlotAllocator m_InstanceSlots;
        SlotAllocator m_GeometryInstanceSlots;
        MeshTable::Changes m_Changes;      // the leaf observer's, kept for its capacity
        uint64_t m_RefreshCalls = 0;
        uint64_t m_LayoutSerial = 0;
        uint64_t m_Serial = 0;

        // Structural scratch, kept for its capacity by the thread (SceneGraphStore.cpp), so that graphs that are made
        // and dropped (an importer's, a benchmark's) do not grow it again in fresh memory.
        struct Scratch
        {
            std::vector<NewNode> newNodes;
            std::vector<Entity> entities;
            std::vector<uint32_t> parents;
            std::vector<SceneGraphNode*> nodes;
            std::vector<OrphanRecord*> records;
            std::vector<uint32_t> firstChild, lastChild, prevSibling, nextSibling, depth;
            std::vector<ComponentId> runComponents;
            std::vector<const void*> runValues;
            std::vector<uint8_t> valueBytes;
        };
        // The calling thread's scratch for one structural operation. Nested operations (a leaf callback of one graph
        // that changes another graph) take the next one. During thread exit a scope makes scratch of its own.
        Scratch* m_S = nullptr;
        struct ScratchScope
        {
            SceneGraphStore& store;
            Scratch* previous;
            Scratch* owned = nullptr;   // when the thread's scratch is gone or nested too deeply
            explicit ScratchScope(SceneGraphStore& s) noexcept;
            ~ScratchScope();
        };
        // Whether the world takes structural changes now (outside phases and observers); reports when it does not.
        bool CanChangeStructure(const char* operation) const noexcept;

        // Refresh state.
        bool m_PendingStructure = false;
        bool m_PendingTransform = false;
        bool m_PendingContent = false;   // InvalidateContent: LocalBounds changed
        bool m_LastRefreshMoved = false;
        uint64_t m_RefreshedVersion = 0;
        uint64_t m_RefreshedStructuralVersion = 0;
        uint32_t m_FrameIndex = 0;
        std::atomic<uint64_t> m_Epoch{ 1 };

        // Render-read scopes (BeginRenderRead) and the nesting of API calls that warn once (RenderReadQuiet).
        uint32_t m_RenderReads = 0;
        uint32_t m_RenderReadQuiet = 0;
        std::atomic<bool> m_RenderServiceClaimed{ false };
        uint64_t m_SchedulerGeneration = 0;   // CreateSystems calls

        // System state: the archetypes per depth bucket, the leaf archetypes for bounds and joints, the tasks.
        std::vector<LevelArchetype> m_Levels[c_MaxHierarchyDepth + 1];
        std::vector<uint8_t> m_ArchetypeLevel;   // per archetype: its bucket, or c_Absent
        std::vector<uint32_t> m_ArchetypeLevelSlot;   // per archetype: its index in its bucket's list
        std::vector<BoundsArchetype> m_BoundsArchetypes;
        std::vector<JointArchetype> m_JointArchetypes;
        uint32_t m_SystemArchetypesSeen = 0;
        std::vector<Task> m_Tasks;
        std::vector<uint8_t> m_DeepMarks;        // per entity index, the rows of the last bucket visited and written
        std::vector<Entity> m_DeepEntries;
        std::vector<Entity> m_DeepStack;
        Counters m_Counters;

        // The on-demand caches.
        std::mutex m_CacheMutex;
        std::vector<CacheFrame> m_CacheStack;
    };
}
