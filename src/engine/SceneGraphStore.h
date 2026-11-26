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
//   the on-demand caches, which are filled under m_CacheMutex and published per node with release order. No reader
//   may run during Refresh (it writes the columns, the caches and the row hints); readers after Refresh returned see
//   its writes through the release of m_SubgraphEpoch and the acquire of IsSubgraphCurrent.
// - Subgraph bounds and content flags (5.7) are computed by Refresh for the nodes with children along the changed
//   paths, in parallel per depth bucket (PrepareSubgraphs, the Bounds system, UpdateSubgraphs), into the nodes' caches.
//   A node's own values stay in its row only (design R3: one home per item of state). Until a structural change,
//   SetLeaf or InvalidateContent advances the epoch again, the getters read the row (a node without children) or the
//   cache (a node with children), without a lock. In between (a change not yet refreshed) the getters fall back to
//   the on-demand caches above, as before.
// - Structural operations create and destroy entities in pre-order, in batches of consecutive nodes with the same
//   components, so that entity ids, row order and the leaf observer's order are deterministic (5.3, 5.5).
// - Propagation and Bounds read the archetype storage directly (EcsWorldImpl.h): they test chunk versions per column
//   (D9) and the parent's chunk version per row, which no public query exposes, and they stamp only the chunks they
//   write. They run as systems of the graph's scheduler, in the TransformPropagation and Bounds phases. GpuScene reads
//   the mesh-instance chunks the same way (plan step E4).
// - Leaf columns (E6.3): the values of a leaf class with LeafTypeDesc::leafRecord have one home, the row while the leaf
//   is attached and the leaf's LeafRecord otherwise. They move in when the rows exist and before any observer runs, and
//   out after the last observer of the row's removal and before the row goes (WorldImpl::RowHooks). While a removal
//   runs, the leaf's record names the row (LeafRecord::IsInRow), so that the observers' Get and Edit still find it.
//   Graphs that never saw such a class (m_AnyLeafRecord) do none of this.
// - Simulation-driven nodes (E6.2a): the rows with SimulationDriven have archetypes of their own. Their pose has one
//   home, the SimPose columns; propagation does not write their WorldTransform, which the interpolation system writes
//   (a phase of its own after PostSimulation), and their children follow them. A LocalTransform that such a row has is
//   neither read nor written: the column table hides it (SlotLocal is absent there), so that the paths of authored nodes
//   stay as they were; the few that find none ask whether the row is driven.
// - Teleports (E6.2b): a setter on a driven node fills the row's SimTeleport, its request's only home, in the parent's
//   space. A system of a phase of its own after BeginFrame resolves the enabled ones to the world, over the simulated
//   poses of driven ancestors, as binding, unbinding and the local getters are: what a getter returned, set again, moves
//   nothing, whatever the interpolation shows. The domain consumes them.
// - Asset tables and slots (step E4): the leaf observer fills a mesh instance's MeshRef, RenderFlags and
//   GpuInstanceSlot columns (table references, stable slots) before it calls RegisterLeaf, and releases them before it
//   calls UnregisterLeaf. Every structural operation and InvalidateContent first advances the world's version, so that
//   what they stamp is newer than any version a reader (GpuScene) took before them.
// - Runtime-only mesh instances (design 5.8; plan step E7, item E7.1): rows with RuntimeMeshInstance and no node or
//   leaf. Observers registered after the leaf observer do per batch what AcquireInstance and ReleaseInstance do per
//   leaf, from the row's MeshRef, on the membership {RuntimeMeshInstance, GpuInstanceSlot, MeshRef, no SceneNode}
//   (OnMatch, OnUnmatch): one bulk table reference per run of rows with one mesh, the slots, RenderFlags and
//   LocalBounds from the mesh. A row holds a reference exactly while it holds an instance slot
//   (GpuInstanceSlot::instance != c_NoInstanceSlot), which it can only while it is a member. A row that arrives with
//   the tag and without
//   GpuInstanceSlot (a snapshot's: the column is NoSerialize) gets the column through the observer's commands, and its
//   slots when that makes it a member. Propagation takes member archetypes (the tag, no SceneNode) as depth-0 roots
//   whatever Parent or HierarchyDepth they carry (LevelArchetype::runtime), and the subgraph passes skip them: they
//   belong to no node's subgraph. The passes that read SceneNode (row hints, teardown) skip archetypes without it.
//   MeshRef is read when a row joins and when it leaves: it must not change in between. RenderFlags and LocalBounds
//   are taken when it joins: nothing refreshes them after the mesh is edited in place (there is no InvalidateContent
//   for runtime rows), and a mesh with a skin prototype is taken as it is (runtime skinned rows are open point 11).
// - Asset callbacks may call back into the graph (AcquireRuntimeMesh, destroying runtime rows) while the changes they
//   report are being raised: each nesting level collects its changes in its own MeshTable::Changes (ChangesScope).

#pragma once

#include <donut/engine/SceneGraph.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>

#include "EcsWorldImpl.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace donut::engine
{
    // The state of a node that is in no graph (5.3). The node part is typed (it is the same for every node, and attach
    // moves it with plain copies); a leaf's columns have a record of their own, which the leaf owns (LeafRecord).
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

    // Whether a leaf column is one value per row, which a record can hold and Get and Edit hand out: no tag, buffer,
    // Shared, Chunk or Sparse component.
    bool IsPlainColumn(const ComponentTypeDesc& desc) noexcept;

    // The material domains of a mesh's geometries as content flags: MeshInstance::GetContentFlags, and the content of a
    // runtime-only mesh instance (RenderFlags::contentFlags). None for a null mesh.
    SceneContentFlags GetMeshContentFlags(const MeshInfo* mesh) noexcept;

    // The column values of a leaf that is in no graph (LeafTypeDesc::leafRecord, E6.3): one block, this header and then
    // the values of the class's plain columns in the order of LeafTypeDesc::columns, found by GUID. All of them are
    // constructed, except while a removal runs (IsInRow): then the row still holds them.
    class LeafRecord
    {
    public:
        // Every value in its default state (the construct hook, zeros without one). Null when out of memory.
        static LeafRecord* Create(const LeafTypeDesc& type) noexcept;
        // A record whose values are still in the row of `entity`; TakeFromRow fetches them.
        static LeafRecord* CreateInRow(const LeafTypeDesc& type, SceneGraphStore* store, Entity entity) noexcept;
        // Destroys the values that it holds, and gives the block back: blocks are recycled through a process-wide pool
        // (bounded), on any thread.
        static void Destroy(LeafRecord* record) noexcept;

        bool IsInRow() const noexcept { return m_Store != nullptr; }
        SceneGraphStore* GetStore() const noexcept { return m_Store; }
        Entity GetEntity() const noexcept { return m_Entity; }
        // The value of a column; null when the record has none. Not while IsInRow.
        void* Find(const nvrhi::GUID& uuid) noexcept;
        // Whether the record's columns are the plain columns of `type`, in any order.
        bool HasColumnsOf(const LeafTypeDesc& type) const noexcept;

        // Moves every value out, destroying what the row lacks a column for, and gives the block back.
        static void MoveIntoRow(LeafRecord* record, ecs::WorldImpl& world, const ecs::Archetype& archetype,
                                uint32_t chunk, uint32_t row) noexcept;
        // Moves the values in from the row, which keeps default ones; what the row lacks starts in its default state.
        void TakeFromRow(ecs::WorldImpl& world, const ecs::Archetype& archetype, uint32_t chunk, uint32_t row) noexcept;

        // Test seam: the records alive in the process.
        static int64_t GetLiveCount() noexcept { return s_LiveCount.load(); }

    private:
        static LeafRecord* Allocate(const LeafTypeDesc& type) noexcept;
        static void Free(LeafRecord* record) noexcept;
        template <typename Visit> void ForEachValue(Visit&& visit) noexcept;

        static std::atomic<int64_t> s_LiveCount;
        const ComponentTypeDesc* const* m_Columns = nullptr;
        uint32_t m_ColumnCount = 0;
        uint32_t m_SizeClass = 0;
        SceneGraphStore* m_Store = nullptr;   // while IsInRow
        Entity m_Entity = Entity::Null;
        LeafRecord* m_NextFree = nullptr;     // in the pool
    };

    class SceneGraphStore
    {
    public:
        // The columns that node accessors read; a per-archetype table maps them to column indices.
        enum Slot : uint32_t
        {
            SlotNode, SlotLinks, SlotParent, SlotLocal, SlotLocalMatrix, SlotWorld, SlotLeaf, SlotLocalBounds,
            SlotWorldBounds, SlotChunkBounds, SlotMeshRef, SlotRenderFlags, SlotInstance, SlotDriven, SlotSimPose,
            SlotSimScale, SlotCount
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
            return LookUpColumn(archetype, slot);
        }
        // The column of a slot by the component id; the local transform of a driven archetype is hidden (c_Absent).
        uint8_t LookUpColumn(const ecs::Archetype& archetype, Slot slot) const noexcept;
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
            return LocateNodeSlow(node, out);
        }
        // The entity-table lookup of LocateNode, out of line: the getters inline LocateNode, and the cold lookup inlined
        // with it made their code larger and the legacy walk about 3 ms slower per view (bench_render_extract).
        bool LocateNodeSlow(const SceneGraphNode* node, RowRef& out) const noexcept;
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
        // Stamps the row's chunk column (and its row version, E7.5) with the world's version, as GetComponentForWrite
        // outside phases does.
        void Stamp(const RowRef& row, Slot slot) const noexcept
        {
            const uint8_t column = ColumnIndex(*row.archetype, row.archetypeIndex, slot);
            if (column != c_Absent)
                ecs::StampRowWrite(*row.archetype, row.chunk, column, row.row, m_Impl->GetStampVersion());
        }
        // The node of an attached entity: through the store's table by entity index (no entity-table lookup), else the
        // SceneNode column.
        SceneGraphNode* NodeOf(Entity entity) const noexcept
        {
            const uint32_t index = GetEntityIndex(entity);
            if (index < m_NodeOfIndex.size() && m_NodeOfIndex[index].entity == entity)
                return m_NodeOfIndex[index].node;
            return NodeOfSlow(entity);
        }
        SceneGraphNode* NodeOfSlow(Entity entity) const noexcept;
        // The GpuInstanceSlot of an attached mesh instance; null while it is in no graph or holds no slot (before the
        // leaf observer reached it, after it released it): MeshInstance::GetInstanceIndex.
        static const GpuInstanceSlot* InstanceSlotOf(const SceneGraphLeaf* leaf) noexcept;

        // The store of an attached leaf, whose node's row exists; null for another.
        static SceneGraphStore* StoreOf(const SceneGraphLeaf* leaf) noexcept
        {
            const SceneGraphNode* node = leaf->GetNode();
            return node && !node->m_Record ? node->m_Store : nullptr;
        }
        // SceneGraphLeaf::Get and Edit of a leaf that has a record or is attached: the value of a column that its class
        // declares, from its record or its row; `write` stamps the row's column. Null when there is none.
        static void* LeafColumn(const SceneGraphLeaf* leaf, const ComponentTypeDesc& type, bool write) noexcept;
        static bool DeclaresColumn(const LeafTypeDesc& type, const nvrhi::GUID& uuid) noexcept;
        // Whether a leaf with a record (a class with leafRecord) is on another node than `node`: a second one would give
        // its values two homes.
        static bool IsOnAnotherNode(const SceneGraphLeaf* leaf, const SceneGraphNode* node) noexcept
        {
            if (!leaf->GetTypeDesc().leafRecord)
                return false;
            const SceneGraphNode* holder = leaf->GetNode();
            return holder && holder != node;
        }

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

        // ------------------------------------------------------------ simulation-driven nodes (4.7, 8.4; E6.2a)
        bool IsDriven(const RowRef& row) const noexcept
        {
            return ColumnIndex(*row.archetype, row.archetypeIndex, SlotDriven) != c_Absent;
        }
        // SceneGraph::SetSimulationDriven of a node that is attached here.
        nvrhi::FRESULT SetSimulationDriven(SceneGraphNode* node, bool driven);
        // The local getters of a driven node, from its simulated pose and the simulated world of its parent (nothing is
        // stored), not from what is shown: the matrix, and the parts that a teleport request (SetTransform) and unbinding
        // take. Below a parent that shears, no parts can express the matrix; they are the ones that give the pose again
        // when they are requested. O(depth). False for another node.
        bool GetDrivenLocal(const SceneGraphNode* node, dm::daffine3& outMatrix) const noexcept;
        bool GetDrivenLocal(const SceneGraphNode* node, LocalTransform& out) const noexcept;
        // For Light: what a driven node's setters take so that its simulated world becomes `world` (its parts). False for
        // another node, and for a matrix without parts.
        static bool ToDrivenLocal(const SceneGraphNode* node, const dm::daffine3& world, LocalTransform& out) noexcept;
        // The phase of the interpolation system, right after PostSimulation; c_NoPhase when there is none.
        static constexpr Phase c_NoPhase = Phase(~0u);   // Phase::Count is the first named phase
        Phase GetInterpolationPhase() const noexcept { return m_InterpolationPhase; }
        // The phase of the system that resolves teleport requests, right after BeginFrame; likewise.
        Phase GetTeleportPhase() const noexcept { return m_TeleportPhase; }
        // SceneGraph::Simulate.
        nvrhi::FRESULT Simulate(const FrameTime& time) noexcept;

        // ------------------------------------------------------ runtime-only mesh instances (5.8; E7.1)
        // SceneGraph's calls of the same names (SceneGraph.h).
        MeshId AcquireRuntimeMesh(MeshInfo* mesh) noexcept;
        void ReleaseRuntimeMesh(MeshId mesh) noexcept;
        uint32_t GetRuntimeMeshInstanceComponents(ComponentId* out, uint32_t capacity,
                                                  bool simulationDriven) const noexcept;
        uint32_t GetRuntimeMeshInstanceCount() const noexcept { return m_RuntimeRows; }
        // The references the application holds through AcquireRuntimeMesh (test seam).
        uint32_t GetRuntimeMeshHolds(MeshId mesh) const noexcept;
        // The observer's work for a batch of rows that became members (`added`, OnMatch) or stop being ones (OnUnmatch).
        void OnRuntimeRows(const EntityBatch& batch, bool added) noexcept;
        // The observer's work for rows that got the tag without the slot column: records its addition in `commands`.
        void AddRuntimeSlotColumn(const EntityBatch& batch, ICommandBuffer* commands) noexcept;

        // ------------------------------------------------------------------------------ refresh (5.4, 5.7)
        // Runs the phases when anything changed; returns whether structure changed since the last refresh.
        bool Refresh(uint32_t frameIndex);
        // After a Refresh that saw structural changes: the mesh table follows meshes edited in place (new geometry
        // counts get new geometry slots, and so do the geometry-instance slots of their instances).
        void ReconcileMeshes();
        bool HasPendingStructureChanges() const noexcept { return m_PendingStructure; }
        bool HasPendingTransformChanges() const noexcept { return m_PendingTransform || m_LastRefreshMoved; }
        // Set with every edit that sets m_PendingStructure or m_PendingTransform, cleared by the propagation system, not
        // by Refresh: the pending flags outlive a propagation that another caller ran.
        bool HasStaleWorldTransforms() const noexcept { return m_StaleWorldTransforms; }

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
        // Whether the subgraph values of the last Refresh are current: no structural change, SetLeaf or
        // InvalidateContent since (UpdateSubgraphs).
        bool IsSubgraphCurrent() const noexcept
        {
            return m_SubgraphEpoch.load(std::memory_order_acquire) == m_Epoch.load(std::memory_order_relaxed);
        }
        // A row's own part of its subgraph: its leaf's box in world space (WorldBounds; empty without a leaf) and its
        // leaf's content flags (RenderFlags for a mesh instance, else the leaf's GetContentFlags). Reports whether the
        // row has children.
        bool OwnPart(const RowRef& row, dm::box3& outBounds, SceneContentFlags& outContent) const noexcept;
        // The lazy path's own box of an attached row: LocalBounds times the float WorldTransform, what the Bounds system
        // writes into WorldBounds at the next Refresh (WorldBounds of a row attached since is not written yet).
        dm::box3 OwnBoundsFromLocal(const RowRef& row) const noexcept
        {
            const LeafObject* leaf = static_cast<const LeafObject*>(Column(row, SlotLeaf));
            const LocalBounds* local = static_cast<const LocalBounds*>(Column(row, SlotLocalBounds));
            const WorldTransform* world = static_cast<const WorldTransform*>(Column(row, SlotWorld));
            if (!leaf || !leaf->object || !local || local->bounds.isempty())
                return dm::box3::empty();
            return local->bounds * dm::affine3(world ? world->transform : dm::daffine3::identity());
        }
        // A leaf's content flags and box through its virtual calls, from code that must not throw (job tasks, noexcept
        // getters): an exception from an override logs a warning and gives None, or `fallback`.
        static SceneContentFlags SafeContentFlags(const SceneGraphLeaf* leaf) noexcept;
        static dm::box3 SafeLocalBoundingBox(SceneGraphLeaf* leaf, const dm::box3& fallback) noexcept;
        bool HasChildren(const RowRef& row) const noexcept
        {
            const SceneLinks* links = static_cast<const SceneLinks*>(Column(row, SlotLinks));
            return links && links->firstChild != Entity::Null;
        }
        dm::box3 OwnBounds(const RowRef& row) const noexcept
        {
            const LeafObject* leaf = static_cast<const LeafObject*>(Column(row, SlotLeaf));
            const WorldBounds* world = static_cast<const WorldBounds*>(Column(row, SlotWorldBounds));
            return leaf && leaf->object && world ? world->bounds : dm::box3::empty();
        }
        SceneContentFlags OwnContent(const RowRef& row) const noexcept
        {
            if (const RenderFlags* flags = static_cast<const RenderFlags*>(Column(row, SlotRenderFlags)))
                return SceneContentFlags(flags->contentFlags);
            const LeafObject* leaf = static_cast<const LeafObject*>(Column(row, SlotLeaf));
            return leaf && leaf->object ? SafeContentFlags(leaf->object) : SceneContentFlags::None;
        }
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
        nvrhi::FRESULT RunTeleports(ISystemContext* context) noexcept;
        nvrhi::FRESULT RunInterpolation(ISystemContext* context) noexcept;
        // Before the phases of a Refresh: the pass's marks and summaries ready, so that the Bounds system marks the paths
        // of the rows whose boxes changed (incremental mode). After them: the subgraph values of the nodes with children
        // along the marked paths (all of them after a structural change), and the row hints of the archetypes whose rows
        // moved.
        void PrepareSubgraphs() noexcept;
        void UpdateSubgraphs() noexcept;
        void RefreshRowHints() noexcept;

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
            uint64_t interpolatedChunks = 0;   // chunks of driven rows the interpolation wrote
            uint64_t resolvedTeleports = 0;    // requests resolved to the world
            uint64_t deepRows = 0;             // rows of the last depth bucket visited (c_MaxHierarchyDepth)
            uint64_t createCalls = 0;          // CreateEntities calls of attach and copy (one per batch)
            uint64_t subgraphPasses = 0;       // UpdateSubgraphs runs (full ones included)
            uint64_t fullSubgraphPasses = 0;   // after a structural change, SetLeaf or InvalidateContent
            uint64_t subgraphNodes = 0;        // nodes with children whose values a pass computed
            uint64_t summarisedChunks = 0;     // chunks a pass took whole from their summary
            uint64_t staticSkips = 0;          // E7.4 remedy 3: Static archetypes a propagation or bounds run left out
            uint64_t childlessChunks = 0;      // E7.4 remedy 4: chunks whose subgraph step the Bounds system took from
                                               // their summary
        };
        const Counters& GetCounters() const noexcept { return m_Counters; }
        // Test seam: the CreateEntities call of attach number `calls` from now (0: the next) is refused by the world
        // as an out-of-memory failure would be (it consumes the values); negative disarms it.
        static void FailCreateAfter(int32_t calls) noexcept { s_FailCreateCountdown.store(calls); }
        // Test seam: the mesh-instance acquisition number `calls` from now throws std::bad_alloc before it allocates
        // anything (as the tables would out of memory); the leaf observer catches it. Negative disarms it.
        static void FailAcquireAfter(int32_t calls) noexcept { s_FailAcquireCountdown.store(calls); }
        // Test seam: the subgraph pass's storage fails to allocate `calls` passes from now (0: the next); negative
        // disarms it. The getters then fall back to the on-demand caches until a pass succeeds.
        static void FailSubgraphAfter(int32_t calls) noexcept { s_FailSubgraphCountdown.store(calls); }
        // Test seam: the number of rows from which the subgraph pass runs a bucket on the job system (default 4096).
        static void SetSerialSubgraphRows(uint32_t rows) noexcept { s_SerialSubgraphRows.store(rows); }
        // Test seam: the subgraph passes (all stores) redone as full passes because a phase of their Refresh changed the
        // world's structure (commands played back, an exclusive system).
        static uint64_t GetSubgraphRedoCount() noexcept { return s_SubgraphRedos.load(); }

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
            uint32_t firstTag;         // in m_LeafTypeTags: the tags, then the columns; LeafObject is just before them,
                                       // and the bounds' components just after, so that SetLeaf has its lists there
            uint32_t tagCount;
            uint32_t columnCount;      // the class's columns (LeafTypeDesc::columns), after its tags
            bool separate;             // some of them are Sparse or Shared: not part of the row's move in SetLeaf
        };

        // The nodes of a structural change whose leaves' values move into the rows (the RowHooks of Attach and SetLeaf).
        struct LeafMove
        {
            SceneGraphStore* store;
            const NewNode* nodes;
            const Entity* entities;
        };

        struct LevelArchetype
        {
            uint32_t archetype;
            uint8_t local;
            uint8_t matrix;
            uint8_t parent;
            uint8_t world;
            bool driven;               // the interpolation writes its rows' worlds, not propagation
            bool runtime;              // runtime-only mesh instances: roots (parent c_Absent), in no node's subgraph
            // E7.4 remedy 3: carries Static in a graph that honours it (SceneGraph::SetStaticTag), not driven, not in
            // the deep bucket; and its entityCount and rowMoves at the run that last took it (RunPropagation)
            bool staticRows = false;
            uint32_t settledCount = ~0u;
            uint64_t settledMoves = ~0ull;
        };

        // Rows that the interpolation writes: driven, with the columns that SetSimulationDriven adds.
        struct DrivenArchetype
        {
            uint32_t archetype;
            uint8_t pose;
            uint8_t previous;
            uint8_t scale;
            uint8_t body;
            uint8_t world;
            uint8_t teleport;          // c_Absent: no requests
            uint8_t parent;            // c_Absent: no parent
        };

        // The simulated world of a row's parent, kept for the next row with the same parent.
        struct ParentWorld
        {
            Entity parent = Entity::Null;
            dm::daffine3 world = dm::daffine3::identity();
        };

        struct BoundsArchetype
        {
            uint32_t archetype;
            uint8_t world;
            uint8_t localBounds;
            uint8_t worldBounds;
            uint8_t chunkBounds;
            uint8_t leaf;              // LeafObject, for leaves with dynamic bounds (c_Absent otherwise)
            // E7.4 remedy 3: carries Static in a graph that honours it, not driven, no dynamic bounds; and its
            // entityCount, rowMoves and m_ContentChanges at the run that last took it (RunBounds)
            bool staticRows = false;
            uint32_t settledCount = ~0u;
            uint64_t settledMoves = ~0ull;
            uint64_t settledContent = ~0ull;
        };

        struct JointArchetype
        {
            uint32_t archetype;
            uint8_t moved;             // what changes when a joint moves: LocalTransform, WorldTransform of a driven one
            uint8_t leaf;
        };

        struct Task
        {
            uint32_t archetype;        // index in the level or bounds list
            uint32_t chunk;
            bool whole;                // the chunk's own columns changed: every row
            const uint64_t* localRows; // propagation, E7.5: a whole chunk that changed only in LocalTransform, whose
                                       // row versions tell the rows to compose (LocalRowsChanged); null: every row
        };

        struct PropagationContext;
        struct BoundsContext;
        struct InterpolationContext;
        struct TeleportContext;
        struct DeepContext;
        struct SubgraphContext;

        // A box and content flags as order-independent keys: each float maps to an unsigned key that orders as the
        // float does, -0 below +0 (UpdateSubgraphs: unions by min and max of keys give the same bits in any order).
        struct SubgraphKeys
        {
            uint32_t keys[6];          // mins x y z, maxs x y z
            uint32_t content;
        };
        // One accumulator per entity index (a node with children marked in this pass), merged into by CAS.
        struct SubgraphAccumulator
        {
            std::atomic<uint32_t> keys[6];
            std::atomic<uint32_t> content;
        };
        // What a chunk gives its parent when every row has the same parent and no row has children: the union of the
        // rows' own parts, valid while none of the chunk's relevant versions is newer than `version`.
        struct ChunkSummary
        {
            uint64_t version = 0;      // 0: not computed
            Entity parent = Entity::Null;
            bool single = false;       // one parent (or none: roots) and no row with children
            SubgraphKeys keys{};
        };
        struct SubgraphFrame
        {
            Entity entity;
            Entity nextChild;
            SubgraphKeys keys;
            SceneContentFlags own;
        };

        void MarkPath(Entity entity, uint32_t pass) noexcept;
        // The Bounds system's part of an incremental pass, for one chunk it wrote: its summary and the marks. With
        // `childless` (SceneGraph::SetChildlessBoundsPath), a chunk whose summary has one parent and no children takes
        // them from the summary instead of its rows, and returns true.
        bool SubgraphFromBounds(const ecs::Archetype& archetype, uint32_t archetypeIndex, uint32_t chunk,
                                const dm::box3& chunkBounds, bool childless) noexcept;
        struct SubgraphRun;
        void SubgraphChunk(SubgraphContext& ctx, uint32_t archetypeIndex, uint32_t chunk, SubgraphRun& run) noexcept;
        void FinalizeSubgraph(SceneGraphNode* node, const SubgraphKeys& keys, uint64_t epoch) const noexcept;
        static void RunSubgraphMarks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        static void RunSubgraphChunks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        static void RunSubgraphDeep(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        static void RunRowHints(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;

        friend class GpuScene;

        // A scheduler with the graph's phases and systems.
        struct Systems
        {
            nvrhi::AutoPtr<IScheduler> scheduler;
            Phase interpolation = c_NoPhase;
            Phase teleport = c_NoPhase;
        };
        // Makes them on `jobs`, and nothing else: a warning for each one that cannot be made, and false then.
        bool CreateSystems(IJobSystem* jobs, Systems& out) noexcept;
        // Makes them the graph's, in place of the ones it has.
        void TakeSystems(const Systems& systems) noexcept;
        // The changes of the asset tables that one call collects and raises (RaiseAssetCallbacks): a level of their own
        // per nesting, since a callback may call back into the graph while its level is being raised. The levels are
        // kept for their capacity; Get is null when a new level cannot be made (out of memory).
        class ChangesScope
        {
        public:
            explicit ChangesScope(SceneGraphStore& store) noexcept;
            ~ChangesScope();
            ChangesScope(const ChangesScope&) = delete;
            ChangesScope& operator=(const ChangesScope&) = delete;
            MeshTable::Changes* Get() const noexcept { return m_Changes; }
        private:
            SceneGraphStore& m_Store;
            MeshTable::Changes* m_Changes = nullptr;
        };
        // The leaf observer's work per leaf (OnAdd, OnRemove), and the mesh-instance columns it fills and releases.
        void OnLeaf(Entity entity, SceneGraphLeaf* leaf, bool added);
        void AcquireInstance(const RowRef& row, MeshInstance* instance, MeshTable::Changes& changes);
        void ReleaseInstance(const RowRef& row, MeshTable::Changes& changes);
        void RaiseAssetCallbacks(const MeshTable::Changes& changes);
        void SyncArchetypes() noexcept;
        uint32_t ResolveLeafType(SceneGraphLeaf* leaf);
        bool IsSeparate(ComponentId id) const noexcept;   // Sparse or Shared (LeafType::separate)
        uint32_t DepthOf(Entity entity) const noexcept;
        void ResolveLeaves();
        bool CreateNodes(Entity parentEntity, uint32_t parentDepth);   // false when the world refused a batch
        void AbandonNodes(uint32_t first);
        void LinkUnder(Entity parentEntity, Entity child);
        // Leaf records (E6.3). BeginLeavingRow: `leaf` is about to leave its row; its record names the row from now on.
        // OnRowRemoving: the row's values move into that record. FinishLeavingRows: the same for entities that a failed
        // DestroyEntities kept.
        void BeginLeavingRow(SceneGraphLeaf* leaf, const SceneGraphNode* node, Entity entity) noexcept;
        static void OnRowsAdded(void* context, uint32_t count) noexcept;
        static void OnRowRemoving(void* context, Entity entity, const ecs::Archetype& archetype, uint32_t chunk,
                                  uint32_t row) noexcept;
        void FinishLeavingRows(const Entity* entities, size_t count) noexcept;
        void TakeLeafValuesAtTeardown(bool worldIsHeld) noexcept;
        ecs::WorldImpl::RowHooks LeavingHooks() noexcept
        {
            return ecs::WorldImpl::RowHooks{ this, nullptr, &SceneGraphStore::OnRowRemoving };
        }
        // Delivers the held batch and the observers' commands; the first failure joins `result` (WarnObservers).
        void FlushObservers(nvrhi::FRESULT& result) noexcept;
        // One warning, in every build, for the observers' failures of one call: they undo nothing.
        static void WarnObservers(const char* operation, nvrhi::FRESULT result) noexcept;
        // Destroys entities, moving their leaves' values out (BeginLeavingRow came before).
        nvrhi::FRESULT DestroyRows(const Entity* entities, size_t count, const char* operation) noexcept;
        void GatherSubgraph(Entity root);
        void MoveToRecords(bool unlinkRoot);
        void BumpEpoch() noexcept { m_Epoch.fetch_add(1, std::memory_order_relaxed); }
        void UpdateSystemLists() noexcept;
        void PropagateRow(const LevelArchetype& level, const ecs::Archetype& archetype, uint32_t chunk, uint32_t row,
                          const dm::daffine3* parentWorld, bool compose) const noexcept;
        bool ParentWritten(Entity parent, uint64_t since, const dm::daffine3** outWorld,
                           bool readVersion) const noexcept;
        void PropagateDeepBucket(IJobSystem* jobs, uint64_t since, uint64_t version, bool parentLevelWrote);
        const LevelArchetype* DeepLevel(uint32_t archetypeIndex) const noexcept;
        static bool DrivenWritten(const LevelArchetype& level, const ecs::Archetype& archetype, uint64_t since) noexcept;
        static bool DeepChunkDirty(const LevelArchetype& level, const ecs::Archetype& archetype, uint32_t chunk,
                                   uint64_t since) noexcept;
        // E7.5: the row versions of LocalTransform when a chunk changed since `since` only there and they tell the
        // changed rows apart (RowTracked, no chunk-only write); null otherwise (every row is composed).
        static const uint64_t* LocalRowsChanged(const LevelArchetype& level, const ecs::Archetype& archetype,
                                                uint32_t chunk, uint64_t since) noexcept;
        bool PropagateDeepRow(Entity entity, bool parentInBucket, uint64_t since, uint64_t version) noexcept;
        static void RunDeepTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        void MarkSkinnedJoints(uint64_t since);
        static void RunPropagationTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        static void RunBoundsTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        // E7.4 remedy 3 (SceneGraph::SetStaticTag): the Static tag's id once registered (looked up, never registered by
        // the graph), whether a row carries it, the propagation tasks of a Static archetype, and in Debug builds the
        // warning about writes of its LocalTransform.
        ComponentId ResolveStaticTag() noexcept;
        bool IsStaticRow(const RowRef& row) noexcept;
        void QueueStaticLevel(LevelArchetype& level, uint32_t index, const ecs::Archetype& archetype,
                              uint64_t since);
        void WarnStaticWrites(const LevelArchetype& level, const ecs::Archetype& archetype, uint64_t since) const;
        static void RunTeleportTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;
        static void RunInterpolationTasks(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept;

        nvrhi::FRESULT BindToSimulation(SceneGraphNode* node);
        nvrhi::FRESULT UnbindFromSimulation(SceneGraphNode* node, const RowRef& boundRow);
        // The move of the row of `node`; what both share around it.
        nvrhi::FRESULT ChangeDriven(SceneGraphNode* node, ecs::WorldImpl::ComponentChange& change,
                                    const ecs::WorldImpl::RowHooks* hooks);
        // SetTransform of a driven node: fills and enables its teleport request.
        void RequestTeleport(const SceneGraphNode* node, const dm::double3* translation, const dm::dquat* rotation,
                             const dm::double3* scaling) noexcept;
        // A driven row's simulated world: its SimPose and SimScale (its WorldTransform when it lacks them); its parts.
        dm::daffine3 SimulatedPose(const RowRef& row) const noexcept;
        LocalTransform SimulatedParts(const RowRef& row) const noexcept;
        // Parts in the world made relative to the simulated world of the row's parent (PartsInParent).
        LocalTransform InParentOf(const RowRef& row, const LocalTransform& world) const noexcept;
        // The simulated world transform of an entity: composed from the local transforms as they are now, propagated or
        // not, from the simulated pose of the nearest driven ancestor (or of the entity itself) down, else from the root.
        // Where nothing is driven it is what propagation gives, bit for bit. It only reads, allocates nothing and does
        // not recurse: getters on several threads and the jobs of the teleport system call it. O(depth): the rows are
        // read once, or (more than 64 rows with a local transform) once per cut of the chain into 64 pieces and twice
        // more: 200,000 rows four times.
        dm::daffine3 ComposeWorld(Entity entity) const noexcept;
        dm::daffine3 ComposeLongChain(Entity entity) const noexcept;
        // The local transform that a driven row gets when it is not driven any more: what the local getters give (the
        // parts of its simulated pose in the space of its parent, at its simulated world, so that both are of the same
        // time). Warns when its matrix does not give the pose (a parent whose scale shears it or has no inverse);
        // `operation` names the call.
        LocalTransform SimulatedLocal(const RowRef& row, ParentWorld& cache, const char* operation) const;

        static std::atomic<int32_t> s_FailCreateCountdown;
        static std::atomic<int32_t> s_FailAcquireCountdown;
        static std::atomic<int32_t> s_FailSubgraphCountdown;
        static std::atomic<uint32_t> s_SerialSubgraphRows;
        static std::atomic<uint64_t> s_SubgraphRedos;
        // The pass's storage (marks, accumulators, summaries, stacks) for the world as it is now; false when it cannot
        // allocate.
        bool EnsureSubgraphStorage() noexcept;
        // Whether a bounds archetype of leaves with dynamic bounds has rows (Refresh does not skip then).
        bool HasDynamicBoundsRows() const noexcept;

        SceneGraph* m_Graph;
        nvrhi::AutoPtr<IWorld> m_World;
        ecs::WorldImpl* m_Impl = nullptr;
        nvrhi::AutoPtr<IJobSystem> m_Jobs;
        nvrhi::AutoPtr<IScheduler> m_Scheduler;
        nvrhi::AutoPtr<IObserver> m_LeafObserver;
        nvrhi::AutoPtr<IObserver> m_RuntimeObserver;   // on RuntimeMeshInstance, after the leaf observer
        ComponentId m_RuntimeTag = ComponentId::Invalid;
        uint32_t m_RuntimeRows = 0;                     // rows that hold slots (GetRuntimeMeshInstanceCount)
        std::vector<uint32_t> m_RuntimeMeshHolds;       // per mesh slot: the application's AcquireRuntimeMesh references
        ComponentId m_Ids[SlotCount] = {};
        ComponentId m_SimPreviousId = ComponentId::Invalid;
        ComponentId m_SimBodyId = ComponentId::Invalid;
        ComponentId m_SimTeleportId = ComponentId::Invalid;
        ComponentId m_FixedStepId = ComponentId::Invalid;   // the resource; the scheduler registers it
        ComponentId m_DepthId = ComponentId::Invalid;
        ComponentId m_JointTag = ComponentId::Invalid;   // SkinnedMeshReferenceLeaf, once a joint leaf was attached
        ComponentId m_DynamicBoundsTag = ComponentId::Invalid;   // DynamicBoundsLeaf, once such a leaf was attached
        // E7.4 remedy 3: SceneGraph::GetStaticTag when the graph was made; the tag's id once something registered it
        bool m_StaticTagHonoured = false;
        ComponentId m_StaticTag = ComponentId::Invalid;
        uint64_t m_ContentChanges = 0;   // InvalidateContent calls that changed a row: Static bounds read them again

        // Column indices per archetype, for the accessors; synchronised after structural changes.
        std::vector<std::array<uint8_t, SlotCount>> m_ArchetypeColumns;

        // Leaf classes seen so far and their tag and column ids.
        std::vector<LeafType> m_LeafTypes;
        std::vector<ComponentId> m_LeafTypeTags;
        bool m_AnyLeafRecord = false;      // one of them keeps its columns in records (LeafTypeDesc::leafRecord)

        // Stable slots of the mesh instances (step E4): instance-buffer rows and geometry-instance ranges.
        SlotAllocator m_InstanceSlots;
        SlotAllocator m_GeometryInstanceSlots;
        std::vector<std::unique_ptr<MeshTable::Changes>> m_ChangesLevels;   // ChangesScope's, by nesting depth
        uint32_t m_ChangesDepth = 0;
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
            std::vector<ecs::WorldImpl::NewValue> newValues;
            std::vector<ComponentId> separateIds;
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
        bool m_StaleWorldTransforms = false;   // HasStaleWorldTransforms
        uint64_t m_RefreshedVersion = 0;
        uint64_t m_RefreshedStructuralVersion = 0;
        uint32_t m_FrameIndex = 0;
        // Skinned instances are marked with the frame of a Refresh, so only the propagation of a Refresh marks, against
        // the version of the last one that did: what a propagation in between (Simulate's, the phase run by hand)
        // propagated is still found.
        bool m_MarkJoints = false;             // while Refresh runs its phases
        uint64_t m_JointsMarkedVersion = 0;    // of this scheduler's propagation; 0: every joint counts as moved
        std::atomic<uint64_t> m_Epoch{ 1 };

        // Render-read scopes (BeginRenderRead) and the nesting of API calls that warn once (RenderReadQuiet).
        uint32_t m_RenderReads = 0;
        uint32_t m_RenderReadQuiet = 0;
        std::atomic<bool> m_RenderServiceClaimed{ false };
        uint64_t m_SchedulerGeneration = 0;   // CreateSystems calls
        Phase m_InterpolationPhase = c_NoPhase;
        Phase m_TeleportPhase = c_NoPhase;
        // The interpolation's alpha per domain at its last run: a chunk is computed again when its poses changed or the
        // alpha of a domain of its rows did.
        double m_LastAlphas[c_MaxSimulationDomains] = {};

        // System state: the archetypes per depth bucket, the leaf archetypes for bounds and joints, the tasks.
        std::vector<LevelArchetype> m_Levels[c_MaxHierarchyDepth + 1];
        std::vector<uint8_t> m_ArchetypeLevel;   // per archetype: its bucket, or c_Absent
        std::vector<uint32_t> m_ArchetypeLevelSlot;   // per archetype: its index in its bucket's list
        std::vector<BoundsArchetype> m_BoundsArchetypes;
        std::vector<JointArchetype> m_JointArchetypes;
        std::vector<DrivenArchetype> m_DrivenArchetypes;
        uint32_t m_SystemArchetypesSeen = 0;
        std::vector<Task> m_Tasks;
        std::vector<uint8_t> m_DeepMarks;        // per entity index, the rows of the last bucket visited and written
        std::vector<Entity> m_DeepEntries;
        std::vector<Entity> m_DeepStack;
        Counters m_Counters;

        // The on-demand caches.
        std::mutex m_CacheMutex;
        std::vector<CacheFrame> m_CacheStack;

        // UpdateSubgraphs: the epoch its values belong to (IsSubgraphCurrent), the pass
        // number marks are compared with, marks and accumulators per entity index, summaries per archetype and chunk,
        // the chunk tasks, the deep bucket's entries and per-thread stacks. Capacity kept.
        std::atomic<uint64_t> m_SubgraphEpoch{ 0 };
        uint32_t m_SubgraphPass = 0;
        bool m_SubgraphPrepared = false;     // between PrepareSubgraphs and UpdateSubgraphs
        bool m_SubgraphFull = false;         // this pass computes every node with children
        bool m_SubgraphForceFull = false;    // the Bounds system ran outside a Refresh: the next pass is a full one
        uint64_t m_SubgraphStructuralVersion = 0;   // the world's structural version at the last pass
        uint64_t m_PreparedStructuralVersion = 0;   // ... when PrepareSubgraphs ran (a phase may change it)
        std::atomic<bool> m_SubgraphNeedsFull{ false };   // an index the pass's storage did not cover: redo as a full one
        std::atomic<uint64_t> m_SubgraphMarked{ 0 };   // marks made in this pass
        size_t m_SubgraphCapacity = 0;
        std::unique_ptr<std::atomic<uint32_t>[]> m_SubgraphMarks;
        std::unique_ptr<SubgraphAccumulator[]> m_SubgraphAccumulators;
        std::vector<std::vector<ChunkSummary>> m_ChunkSummaries;
        std::vector<Task> m_SubgraphTasks;
        std::vector<Entity> m_SubgraphEntries;
        std::vector<std::vector<SubgraphFrame>> m_SubgraphStacks;

        // NodeOf's table by entity index, written by attach and cleared by detach; the row-move counts the hints of each
        // archetype were last refreshed at (RefreshRowHints).
        struct NodeOfEntry
        {
            Entity entity = Entity::Null;
            SceneGraphNode* node = nullptr;
        };
        std::vector<NodeOfEntry> m_NodeOfIndex;
        std::vector<uint64_t> m_HintedMoves;
    };
}
