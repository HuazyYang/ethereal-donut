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

// The scene graph over the world (plan step E3): storage, structural operations, the TransformPropagation and Bounds
// systems and the leaf observer. See SceneGraphStore.h for the rules.

#include "SceneGraphStore.h"

#include <donut/core/log.h>
#include <donut/core/query_cast.h>

#include <algorithm>
#include <cassert>
#include <exception>
#include <cstring>
#include <new>
#include <unordered_map>
#include <unordered_set>
#if defined(_M_X64) || defined(__x86_64__)
#include <xmmintrin.h>
#endif
#include <cstdlib>
#include <thread>

namespace donut::engine
{
    // ------------------------------------------------------------------------------ reference hooks (D14)

    SceneNode::SceneNode(const SceneNode& other) noexcept
        : object(other.object)
    {
        if (object)
            object->AddRef();
    }

    SceneNode& SceneNode::operator=(const SceneNode& other) noexcept
    {
        if (other.object)
            other.object->AddRef();
        if (object)
            object->Release();
        object = other.object;
        return *this;
    }

    SceneNode& SceneNode::operator=(SceneNode&& other) noexcept
    {
        if (this != &other)
        {
            if (object)
                object->Release();
            object = other.object;
            other.object = nullptr;
        }
        return *this;
    }

    SceneNode::~SceneNode()
    {
        if (object)
            object->Release();
    }

    LeafObject::LeafObject(const LeafObject& other) noexcept
        : object(other.object)
    {
        if (object)
            object->AddRef();
    }

    LeafObject& LeafObject::operator=(const LeafObject& other) noexcept
    {
        if (other.object)
            other.object->AddRef();
        if (object)
            object->Release();
        object = other.object;
        return *this;
    }

    LeafObject& LeafObject::operator=(LeafObject&& other) noexcept
    {
        if (this != &other)
        {
            if (object)
                object->Release();
            object = other.object;
            other.object = nullptr;
        }
        return *this;
    }

    LeafObject::~LeafObject()
    {
        if (object)
            object->Release();
    }

    namespace
    {
        using namespace ecs;

        constexpr uint32_t c_None = ~0u;

        // The job system that graphs share unless SetJobSystem gives them another. It is never released: its workers
        // would have to be joined from a static destructor, after the process may have ended them. So a module that
        // links donut_engine statically and is unloaded (a DLL) leaves its threads running code that is gone: such a
        // module gives its graphs a job system it owns (SetJobSystem) and releases it before it unloads.
        IJobSystem* GetSharedJobSystem() noexcept
        {
            static IJobSystem* const s_Jobs = []() noexcept
            {
                JobSystemDesc desc;
                desc.debugName = "SceneGraph";
                desc.maxExternalThreads = 64;   // every thread that may refresh a graph at the same time
                IJobSystem* jobs = nullptr;
                if (NVRHI_FAILED(CreateJobSystem(desc, &jobs)))
                    log::warning("SceneGraph: cannot create the shared job system");
                return jobs;
            }();
            return s_Jobs;
        }

        // The local transform as the old SceneGraphNode::UpdateLocalTransform composed it; the trace goldens pin the
        // order of the operations bit for bit.
        dm::daffine3 ComposeLocal(const LocalTransform& local) noexcept
        {
            dm::daffine3 transform = dm::scaling(local.scaling);
            transform *= local.rotation.toAffine();
            transform *= dm::translation(local.translation);
            return transform;
        }

        // The threads that can take ranges: IJobSystem::GetThreadCount also counts the external slots (64 in the shared
        // system), which are mostly idle; the hardware bounds the workers that can run at once.
        uint32_t WorkingThreads(IJobSystem* jobs) noexcept
        {
            static const uint32_t s_Hardware = std::max(1u, std::thread::hardware_concurrency());
            return jobs ? std::max(1u, std::min(jobs->GetThreadCount(), s_Hardware)) : 1u;
        }

        nvrhi::FLONG ReferenceCount(nvrhi::IObject* object) noexcept
        {
            object->AddRef();
            return object->Release();
        }

        uint8_t* ColumnBase(const Archetype& archetype, uint32_t chunk, uint8_t column) noexcept
        {
            return archetype.chunks[chunk].memory + archetype.columns[column].offset;
        }

        // Walks over subgraphs are chains of cache misses; the passes after them fetch a few nodes ahead.
        constexpr size_t c_PrefetchDistance = 8;
        inline void Prefetch(const void* address) noexcept
        {
#if defined(_M_X64) || defined(__x86_64__)
            _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(address);
#else
            (void)address;
#endif
        }

        // Records are recycled (OrphanRecord::Create): an attach gives one back per node, and the next detach or
        // import takes them, without a heap call each. The pool keeps at most c_MaxPooledRecords; it is never
        // destroyed, so that nodes destroyed during static destruction can still give theirs back.
        constexpr size_t c_MaxPooledRecords = size_t(1) << 16;
        struct RecordPool
        {
            std::mutex mutex;
            std::vector<OrphanRecord*> records;
        };
        RecordPool& GetRecordPool() noexcept
        {
            static RecordPool* const s_Pool = new RecordPool();
            return *s_Pool;
        }

        // The graph's observer on LeafObject (design 5.5): it maintains the leaf vectors through the virtual
        // RegisterLeaf and UnregisterLeaf, in creation and destruction order.
        class LeafObserver final : public nvrhi::ObjectImpl<IObserver>
        {
        public:
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(LeafObserver)
            NVRHI_IMPLEMENTS_INTERFACE(IObserver)
            NVRHI_END_INTERFACE_TABLE()

            using Register = void (*)(SceneGraph* graph, SceneGraphLeaf* leaf, bool added);

            LeafObserver(SceneGraph* graph, Register registerLeaf) noexcept
                : m_Graph(graph), m_Register(registerLeaf) { }

            nvrhi::FRESULT OnEntities(ObserverEvent event, const EntityBatch& batch, ICommandBuffer*) noexcept override
            {
                const LeafObject* leaves = static_cast<const LeafObject*>(batch.columns[0]);
                const bool added = event == ObserverEvent::OnAdd;
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    if (!leaves[i].object)
                        continue;
                    // RegisterLeaf overrides and the OnMeshAdded callbacks may throw; this boundary is noexcept
                    try
                    {
                        m_Register(m_Graph, leaves[i].object, added);
                    }
                    catch (const std::exception& error)
                    {
                        log::warning("SceneGraph: %s threw '%s'; the leaf is skipped",
                                     added ? "RegisterLeaf" : "UnregisterLeaf", error.what());
                    }
                    catch (...)
                    {
                        log::warning("SceneGraph: %s threw; the leaf is skipped",
                                     added ? "RegisterLeaf" : "UnregisterLeaf");
                    }
                }
                return nvrhi::FS_OK;
            }

        private:
            SceneGraph* m_Graph;   // not a reference: the graph owns the world, which owns this observer
            Register m_Register;
        };
    }

    OrphanRecord* OrphanRecord::Create()
    {
        void* memory = nullptr;
        {
            RecordPool& pool = GetRecordPool();
            std::lock_guard<std::mutex> lock(pool.mutex);
            if (!pool.records.empty())
            {
                memory = pool.records.back();
                pool.records.pop_back();
            }
        }
        if (!memory)
            memory = ::operator new(sizeof(OrphanRecord));
        return new (memory) OrphanRecord();
    }

    void OrphanRecord::Destroy(OrphanRecord* record) noexcept
    {
        if (record)
            Destroy(&record, 1);
    }

    void OrphanRecord::Destroy(OrphanRecord* const* records, size_t count) noexcept
    {
        // destroyed first: releasing children may give their records back too
        for (size_t i = 0; i < count; ++i)
            records[i]->~OrphanRecord();
        RecordPool& pool = GetRecordPool();
        size_t kept = 0;
        {
            std::lock_guard<std::mutex> lock(pool.mutex);
            kept = std::min(count, c_MaxPooledRecords - std::min(c_MaxPooledRecords, pool.records.size()));
            pool.records.insert(pool.records.end(), records, records + kept);
        }
        for (size_t i = kept; i < count; ++i)
            ::operator delete(records[i]);
    }

    struct SceneGraphStore::PropagationContext
    {
        SceneGraphStore* store;
        uint32_t level;
        uint64_t since;
        uint64_t version;
        std::atomic<bool> wrote{ false };
        std::atomic<uint64_t> chunks{ 0 };
        std::atomic<uint64_t> rows{ 0 };
    };

    struct SceneGraphStore::BoundsContext
    {
        SceneGraphStore* store;
        uint64_t version;
        std::atomic<uint64_t> chunks{ 0 };
    };

    // ---------------------------------------------------------------------------------------- the store

    SceneGraphStore::SceneGraphStore(SceneGraph* graph) noexcept
        : m_Graph(graph)
    {
        WorldDesc desc;
        desc.debugName = "SceneGraph";
#ifndef NDEBUG
        desc.validation = true;
#endif
        if (NVRHI_FAILED(CreateWorld(desc, &m_World)))
        {
            log::warning("SceneGraph: cannot create the world");
            std::abort();
        }
        m_Impl = GetWorldImpl(m_World);
        // chunk pages come from the process-wide slab cache: a graph made after another was dropped reuses its memory
        m_Impl->ShareChunkPages();

        m_Ids[SlotNode] = RegisterComponent<SceneNode>(m_World);
        m_Ids[SlotLinks] = RegisterComponent<SceneLinks>(m_World);
        m_Ids[SlotParent] = RegisterComponent<Parent>(m_World);
        m_Ids[SlotLocal] = RegisterComponent<LocalTransform>(m_World);
        m_Ids[SlotLocalMatrix] = RegisterComponent<LocalMatrix>(m_World);
        m_Ids[SlotWorld] = RegisterComponent<WorldTransform>(m_World);
        m_Ids[SlotLeaf] = RegisterComponent<LeafObject>(m_World);
        m_Ids[SlotLocalBounds] = RegisterComponent<LocalBounds>(m_World);
        m_Ids[SlotWorldBounds] = RegisterComponent<WorldBounds>(m_World);
        m_Ids[SlotChunkBounds] = RegisterComponent<ChunkWorldBounds>(m_World);
        m_DepthId = RegisterComponent<HierarchyDepth>(m_World);

        m_LeafObserver = MAKE_RC_OBJ_PTR(LeafObserver, graph, [](SceneGraph* g, SceneGraphLeaf* leaf, bool added)
        {
            if (added)
                g->RegisterLeaf(leaf);
            else
                g->UnregisterLeaf(leaf);
        });
        ObserverDesc observer{};
        observer.debugName = "SceneGraph leaves";
        observer.component = m_Ids[SlotLeaf];
        observer.access = Access::Read;
        ObserverId id = ObserverId::Invalid;
        observer.event = ObserverEvent::OnAdd;
        m_World->AddObserver(observer, m_LeafObserver, id);
        observer.event = ObserverEvent::OnRemove;
        m_World->AddObserver(observer, m_LeafObserver, id);

        m_Jobs = GetSharedJobSystem();
        CreateSystems();
    }

    void SceneGraphStore::CreateSystems() noexcept
    {
        m_Scheduler.Reset();
        if (!m_Jobs || NVRHI_FAILED(CreateScheduler(m_World, m_Jobs, &m_Scheduler)))
        {
            log::warning("SceneGraph: cannot create the scheduler");
            return;
        }

        const ComponentAccess propagationAccess[] = {
            { m_Ids[SlotLocal], Access::Read, {} },
            { m_Ids[SlotLocalMatrix], Access::Write, {} },
            { m_Ids[SlotParent], Access::Read, {} },
            { m_DepthId, Access::Read, {} },
            { m_Ids[SlotWorld], Access::Write, {} },
        };
        SystemDesc propagation{};
        propagation.name = "SceneGraph.TransformPropagation";
        propagation.accesses = propagationAccess;
        propagation.accessCount = uint32_t(std::size(propagationAccess));
        propagation.phase = Phase::TransformPropagation;
        propagation.flags = SystemFlags::MainThread;   // it splits its levels with ParallelFor itself

        const ComponentAccess boundsAccess[] = {
            { m_Ids[SlotWorld], Access::Read, {} },
            { m_Ids[SlotLocalBounds], Access::Read, {} },
            { m_Ids[SlotWorldBounds], Access::Write, {} },
            { m_Ids[SlotChunkBounds], Access::Write, {} },
        };
        SystemDesc bounds{};
        bounds.name = "SceneGraph.Bounds";
        bounds.accesses = boundsAccess;
        bounds.accessCount = uint32_t(std::size(boundsAccess));
        bounds.phase = Phase::Bounds;
        bounds.flags = SystemFlags::MainThread;

        SceneGraphStore* self = this;
        nvrhi::AutoPtr<ISystem> system;
        SystemId id = SystemId::Invalid;
        if (NVRHI_SUCCEEDED(CreateLambdaSystem(propagation,
                [self](ISystemContext* context) noexcept { return self->RunPropagation(context); }, &system)))
            m_Scheduler->AddSystem(system, id);
        system.Reset();
        if (NVRHI_SUCCEEDED(CreateLambdaSystem(bounds,
                [self](ISystemContext* context) noexcept { return self->RunBounds(context); }, &system)))
            m_Scheduler->AddSystem(system, id);
        if (NVRHI_FAILED(m_Scheduler->Build()))
            log::warning("SceneGraph: the scheduler's phases do not build");
    }

    void SceneGraphStore::SetJobSystem(IJobSystem* jobSystem) noexcept
    {
        nvrhi::AutoPtr<IJobSystem> jobs = jobSystem ? jobSystem : GetSharedJobSystem();
        if (jobs == m_Jobs)
            return;
        m_Jobs = jobs;
        CreateSystems();
        // the new systems have not run: their first run treats every chunk as changed
    }

    SceneGraphStore::~SceneGraphStore()
    {
        // Lifetime (5.3): every node whose object is referenced elsewhere is detached into records first, so that it
        // outlives the graph as the root of its own subgraph, as a referenced child outlived its parent before. The
        // world holds one reference to each node; the graph has released its root already. One pre-order pass from
        // each root (a node without a Parent): a referenced node takes its subgraph along, so the walk skips it.
        SyncArchetypes();
        {
            ScratchScope scope(*this);
            std::vector<Entity> roots;
            const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
                if (ColumnIndex(archetype, a, SlotNode) == c_Absent ||
                    ColumnIndex(archetype, a, SlotParent) != c_Absent)
                    continue;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                    roots.insert(roots.end(), archetype.Entities(c), archetype.Entities(c) + archetype.chunks[c].count);
            }
            std::sort(roots.begin(), roots.end());
            std::vector<Entity> tops;
            for (Entity root : roots)
            {
                Entity entity = root;
                while (entity != Entity::Null)
                {
                    SceneGraphNode* node = NodeOf(entity);
                    const SceneLinks* links = Get<SceneLinks>(entity, SlotLinks);
                    const bool referenced = node && ReferenceCount(node) > 1;
                    if (referenced)
                        tops.push_back(entity);
                    if (!referenced && links && links->firstChild != Entity::Null)
                    {
                        entity = links->firstChild;
                        continue;
                    }
                    // the next sibling, or that of the nearest ancestor that has one, inside the root's subgraph
                    for (;;)
                    {
                        if (entity == root)
                        {
                            entity = Entity::Null;
                            break;
                        }
                        const SceneLinks* current = Get<SceneLinks>(entity, SlotLinks);
                        if (current && current->nextSibling != Entity::Null)
                        {
                            entity = current->nextSibling;
                            break;
                        }
                        const Parent* parent = Get<Parent>(entity, SlotParent);
                        entity = parent ? parent->entity : root;
                    }
                }
            }
            // after the walk: moving a subgraph into records keeps the rows, but the nodes no longer name them
            for (Entity top : tops)
            {
                GatherSubgraph(top);
                MoveToRecords(false);
            }
        }

        // Nothing is delivered while the world is torn down: leaves are not unregistered, as before.
        m_Scheduler.Reset();
        m_LeafObserver.Reset();
        m_Impl = nullptr;
        m_World.Reset();
        m_Jobs.Reset();
    }

    // ------------------------------------------------------------------------------------------ rows

    void SceneGraphStore::SyncArchetypes() noexcept
    {
        const uint32_t count = m_Impl->GetArchetypeStorageCount();
        for (uint32_t a = uint32_t(m_ArchetypeColumns.size()); a < count; ++a)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
            std::array<uint8_t, SlotCount> columns;
            for (uint32_t s = 0; s < SlotCount; ++s)
            {
                const uint32_t column = archetype.FindColumn(m_Ids[s]);
                assert(column == c_NoColumn || column < c_Absent);
                columns[s] = column == c_NoColumn ? c_Absent : uint8_t(column);
            }
            m_ArchetypeColumns.push_back(columns);
        }
    }

    SceneGraphNode* SceneGraphStore::NodeOf(Entity entity) const noexcept
    {
        const SceneNode* node = Get<SceneNode>(entity, SlotNode);
        return node ? node->object : nullptr;
    }

    uint32_t SceneGraphStore::DepthOf(Entity entity) const noexcept
    {
        RowRef row;
        if (!Locate(entity, row))
            return 0;
        const uint32_t column = row.archetype->FindColumn(m_DepthId);
        if (column == c_NoColumn)
            return 0;
        uint32_t depth = 0;
        std::memcpy(&depth, row.archetype->sharedValues + row.archetype->columns[column].offset, sizeof(depth));
        return depth;
    }

    uint32_t SceneGraphStore::ResolveLeafType(SceneGraphLeaf* leaf)
    {
        const LeafTypeDesc& desc = leaf->GetTypeDesc();
        for (uint32_t i = 0; i < uint32_t(m_LeafTypes.size()); ++i)
        {
            if (m_LeafTypes[i].desc == &desc)
                return i;
        }
        LeafType type{ &desc, uint32_t(m_LeafTypeTags.size()), 0 };
        for (uint32_t t = 0; t < desc.tagCount; ++t)
        {
            ComponentId id = ComponentId::Invalid;
            if (NVRHI_FAILED(m_World->RegisterComponent(*desc.tags[t], id)))
            {
                log::warning("SceneGraph: cannot register the leaf tag %s", desc.tags[t]->name);
                continue;
            }
            if (desc.tags[t]->uuid == ComponentTraits<SkinnedMeshReferenceLeaf>::Desc().uuid)
                m_JointTag = id;
            m_LeafTypeTags.push_back(id);
            ++type.tagCount;
        }
        m_LeafTypes.push_back(type);
        return uint32_t(m_LeafTypes.size() - 1);
    }

    // ------------------------------------------------------------------------------------- orphan links

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::UnlinkOrphan(SceneGraphNode* child)
    {
        OrphanRecord* record = child ? child->m_Record : nullptr;
        if (!record || !record->parent)
            return nullptr;
        OrphanRecord& parent = *record->parent->m_Record;

        SceneGraphNode* prev = record->prevSibling;
        nvrhi::AutoPtr<SceneGraphNode>& link = prev ? prev->m_Record->nextSibling : parent.firstChild;
        assert(link.Get() == child);

        // take the list's reference to the child, and close the gap with the child's next sibling
        nvrhi::AutoPtr<SceneGraphNode> removed = std::move(link);
        link = std::move(record->nextSibling);
        if (link)
            link->m_Record->prevSibling = prev;
        else
            parent.lastChild = prev;

        record->parent = nullptr;
        record->prevSibling = nullptr;
        return removed;
    }

    void SceneGraphStore::AppendOrphan(SceneGraphNode* parent, SceneGraphNode* child)
    {
        assert(parent && child && !parent->m_Store && !child->m_Store);
#ifndef NDEBUG
        // linking a node under itself or under one of its descendants would make a cycle
        for (SceneGraphNode* ancestor = parent; ancestor;
             ancestor = ancestor->m_Record ? ancestor->m_Record->parent : nullptr)
            assert(ancestor != child);
#endif
        // A node is linked into at most one child list: a node that moves brings the reference its old list held,
        // any other node gets a new one.
        nvrhi::AutoPtr<SceneGraphNode> link = UnlinkOrphan(child);
        if (!link)
            link = child;

        OrphanRecord& record = child->EnsureRecord();
        OrphanRecord& parentRecord = parent->EnsureRecord();
        assert(!record.nextSibling && !record.prevSibling);
        record.parent = parent;
        record.prevSibling = parentRecord.lastChild;
        (parentRecord.lastChild ? parentRecord.lastChild->m_Record->nextSibling : parentRecord.firstChild) =
            std::move(link);
        parentRecord.lastChild = child;
    }

    // ------------------------------------------------------------------------------- attach (5.3)

    namespace
    {
        // Scratch per thread and nesting depth, kept for its capacity.
        constexpr uint32_t c_ScratchDepth = 4;
        thread_local uint32_t t_ScratchDepth = 0;
    }

    namespace
    {
        // Set when the thread's scratch is destroyed (thread exit, or the main thread's before static destruction): a
        // graph torn down after that gets scratch of its own. Trivially destructible, so it stays readable then.
        thread_local bool t_ScratchGone = false;
    }

    SceneGraphStore::ScratchScope::ScratchScope(SceneGraphStore& s) noexcept
        : store(s), previous(s.m_S)
    {
        if (t_ScratchGone || t_ScratchDepth >= c_ScratchDepth)
        {
            owned = new Scratch();
            store.m_S = owned;
            return;
        }
        struct Holder
        {
            Scratch scratch[c_ScratchDepth];
            ~Holder() { t_ScratchGone = true; }
        };
        thread_local Holder holder;
        store.m_S = &holder.scratch[t_ScratchDepth];
        ++t_ScratchDepth;
    }

    SceneGraphStore::ScratchScope::~ScratchScope()
    {
        if (owned)
            delete owned;
        else
            --t_ScratchDepth;
        store.m_S = previous;
    }

    bool SceneGraphStore::CanChangeStructure(const char* operation) const noexcept
    {
        if (m_Impl->CanChangeStructure())
            return true;
        // design 5.6: the scene-graph API changes structure outside phases; observers cannot change it either
        log::warning("SceneGraph::%s: the graph's world takes no structural change now (inside a phase or a leaf "
                     "callback); nothing changed", operation);
        return false;
    }

    // The leaf classes and local boxes of the gathered nodes. A separate pass, so that the leaves, which the walk does
    // not touch, can be fetched ahead.
    void SceneGraphStore::ResolveLeaves()
    {
        std::vector<NewNode>& nodes = m_S->newNodes;
        const size_t n = nodes.size();
        for (size_t i = 0; i < n; ++i)
        {
            if (i + c_PrefetchDistance < n)
                Prefetch(nodes[i + c_PrefetchDistance].leaf);
            NewNode& entry = nodes[i];
            if (entry.leaf)
            {
                entry.leafType = ResolveLeafType(entry.leaf);
                entry.localBounds = entry.leaf->GetLocalBoundingBox();
            }
        }
    }

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::AttachOrphan(SceneGraphNode* parent, SceneGraphNode* child)
    {
        assert(!child->m_Store);
        if (!CanChangeStructure("Attach"))
            return nullptr;
        ScratchScope scope(*this);

        // The reference that the subgraph's root brings: its old orphan parent's link, or a new one.
        nvrhi::AutoPtr<SceneGraphNode> incoming = UnlinkOrphan(child);
        if (!incoming)
            incoming = child;

        // The subgraph in pre-order, from the records' links. The references move into the world: each link's to the
        // node it names, the record's to its leaf.
        std::vector<NewNode>& nodes = m_S->newNodes;
        nodes.clear();
        bool anyLocal = false;
        uint32_t parentIndex = c_None;
        SceneGraphNode* node = child;
        while (node)
        {
            OrphanRecord* record = node->m_Record;
            const uint32_t index = uint32_t(nodes.size());
            NewNode entry{ node, record, nullptr, parentIndex, c_None, dm::box3::empty(), false };
            SceneGraphNode* firstChild = nullptr;
            if (record)
            {
                entry.leaf = record->leaf.Detach();
                entry.hasLocal = record->hasLocal;
                anyLocal |= record->hasLocal;
                firstChild = record->firstChild.Get();
            }
            nodes.push_back(entry);
            if (firstChild)
            {
                parentIndex = index;
                node = firstChild;
                continue;
            }
            for (;;)
            {
                if (node == child)
                {
                    node = nullptr;
                    break;
                }
                OrphanRecord& current = *node->m_Record;
                if (current.nextSibling)
                {
                    node = current.nextSibling.Get();
                    break;
                }
                node = current.parent;
                parentIndex = nodes[parentIndex].parent;
            }
        }
        for (const NewNode& entry : nodes)
        {
            if (entry.record)
            {
                (void)entry.record->firstChild.Detach();
                (void)entry.record->nextSibling.Detach();
            }
        }
        (void)incoming.Detach();
        ResolveLeaves();

        nvrhi::AutoPtr<SceneGraphNode> result(child);
        const Entity parentEntity = parent ? parent->m_Entity : Entity::Null;
        const bool created = CreateNodes(parentEntity, parent ? DepthOf(parentEntity) : 0);

        m_PendingStructure = true;
        m_PendingTransform |= anyLocal;
        BumpEpoch();
        return created ? result : nullptr;
    }

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::AttachCopy(SceneGraphNode* parent, SceneGraphNode* source,
                                                               SceneGraphStore* sourceStore)
    {
        if (!CanChangeStructure("Attach"))
            return nullptr;
        ScratchScope scope(*this);

        // The source subgraph as it is before the call (the parent may lie inside it), copied into new nodes in no
        // graph, with records that hold the local transforms and cloned leaves; then attached like any orphans.
        std::vector<NewNode>& nodes = m_S->newNodes;
        nodes.clear();
        std::unordered_map<SceneGraphNode*, uint32_t> nodeMap;
        bool anyLocal = false;
        uint32_t parentIndex = c_None;
        SceneGraphWalker walker(source);
        while (walker)
        {
            SceneGraphNode* original = walker.Get();
            const uint32_t index = uint32_t(nodes.size());
            SceneGraphNode* copy = MAKE_RC_OBJ_PTR(SceneGraphNode).Detach();   // its reference goes to the world
            copy->m_Name = original->m_Name;
            OrphanRecord& record = copy->EnsureRecord();

            RowRef row;
            if (sourceStore->Locate(original->m_Entity, row))
            {
                if (const auto* local = static_cast<const LocalTransform*>(sourceStore->Column(row, SlotLocal)))
                {
                    record.local = *local;
                    record.hasLocal = true;
                    anyLocal = true;
                }
            }
            SceneGraphLeaf* clone = nullptr;
            if (SceneGraphLeaf* leaf = original->GetLeaf())
            {
                clone = leaf->Clone().Detach();
                clone->m_Node = nvrhi::WeakPtr<SceneGraphNode>(copy);
            }
            nodes.push_back(NewNode{ copy, &record, clone, parentIndex, c_None, dm::box3::empty(), record.hasLocal });
            nodeMap[original] = index;

            int deltaDepth = walker.Next(true);
            if (deltaDepth > 0)
                parentIndex = index;
            else
            {
                while (deltaDepth++ < 0)
                    parentIndex = nodes[parentIndex].parent;
            }
        }

        // Point the cloned animations, skins and joint references at the new nodes.
        auto mapped = [&](SceneGraphNode* original) -> const NewNode*
        {
            auto it = nodeMap.find(original);
            return it != nodeMap.end() ? &nodes[it->second] : nullptr;
        };
        for (const NewNode& entry : nodes)
        {
            if (auto animation = query_cast<SceneGraphAnimation>(entry.leaf))
            {
                for (const auto& channel : animation->GetChannels())
                {
                    if (const NewNode* target = mapped(channel->GetTargetNode()))
                        channel->SetTargetNode(target->node);
                }
            }
            else if (auto skinnedInstance = query_cast<SkinnedMeshInstance>(entry.leaf))
            {
                for (auto& joint : skinnedInstance->joints)
                {
                    auto jointNode = joint.node.Lock();
                    if (const NewNode* target = mapped(jointNode.Get()))
                        joint.node = target->node;
                }
            }
            else if (auto meshReference = query_cast<SkinnedMeshReference>(entry.leaf))
            {
                auto instance = meshReference->m_Instance.Lock();
                if (instance)
                {
                    if (const NewNode* target = mapped(instance->GetNode()))
                        meshReference->m_Instance = query_cast<SkinnedMeshInstance>(target->leaf);
                    else
                        meshReference->m_Instance.Reset();
                }
            }
        }
        ResolveLeaves();

        nvrhi::AutoPtr<SceneGraphNode> result(nodes.front().node);
        const Entity parentEntity = parent ? parent->m_Entity : Entity::Null;
        const bool created = CreateNodes(parentEntity, parent ? DepthOf(parentEntity) : 0);

        m_PendingStructure = true;
        m_PendingTransform |= anyLocal;
        BumpEpoch();
        return created ? result : nullptr;
    }

    // Creates the entities of the gathered nodes in pre-order, one CreateEntities call per run of nodes with the same
    // components, and links the first under `parentEntity`. Every node hands its reference and its leaf's to the world;
    // its record is read, then given back to the pool.
    std::atomic<int32_t> SceneGraphStore::s_FailCreateCountdown{ -1 };

    bool SceneGraphStore::CreateNodes(Entity parentEntity, uint32_t parentDepth)
    {
        Scratch& S = *m_S;
        const std::vector<NewNode>& nodes = S.newNodes;
        const uint32_t n = uint32_t(nodes.size());
        S.entities.resize(n);
        if (NVRHI_FAILED(m_World->ReserveEntities(n, S.entities.data())))
        {
            log::warning("SceneGraph: cannot reserve %u entities", n);
            std::abort();
        }

        // The links between the new nodes, by index, and their depth buckets.
        S.firstChild.assign(n, c_None);
        S.lastChild.assign(n, c_None);
        S.prevSibling.assign(n, c_None);
        S.nextSibling.assign(n, c_None);
        S.depth.resize(n);
        S.depth[0] = parentEntity != Entity::Null ? std::min(parentDepth + 1, c_MaxHierarchyDepth) : 0;
        for (uint32_t i = 1; i < n; ++i)
        {
            const uint32_t p = nodes[i].parent;
            if (S.lastChild[p] == c_None)
                S.firstChild[p] = i;
            else
            {
                S.nextSibling[S.lastChild[p]] = i;
                S.prevSibling[i] = S.lastChild[p];
            }
            S.lastChild[p] = i;
            S.depth[i] = std::min(S.depth[p] + 1, c_MaxHierarchyDepth);
        }
        Entity rootPrev = Entity::Null;
        if (parentEntity != Entity::Null)
        {
            if (const SceneLinks* links = Get<SceneLinks>(parentEntity, SlotLinks))
                rootPrev = links->lastChild;
        }
        auto entityAt = [&](uint32_t index) noexcept { return index == c_None ? Entity::Null : S.entities[index]; };

        // The handles name their entities before the rows exist, so that the leaf observer finds them; a node reads
        // its record until its batch is created. A cache filled in another graph is not this graph's.
        for (uint32_t i = 0; i < n; ++i)
        {
            nodes[i].node->m_Store = this;
            nodes[i].node->m_Entity = S.entities[i];
            nodes[i].node->m_CacheEpoch.store(0, std::memory_order_relaxed);
        }

        static const OrphanRecord s_Defaults;
        uint32_t first = 0;
        while (first < n)
        {
            const bool hasParent = first > 0 || parentEntity != Entity::Null;
            const NewNode& head = nodes[first];
            uint32_t end = first + 1;
            while (hasParent && end < n && nodes[end].hasLocal == head.hasLocal && nodes[end].leafType == head.leafType)
                ++end;
            const uint32_t count = end - first;
            const bool hasLeaf = head.leafType != c_None;

            // Value arrays, 16-byte aligned, in one block.
            enum { ValueNode, ValueLinks, ValueDepth, ValueWorld, ValueParent, ValueLocal, ValueLeaf, ValueLocalBounds,
                   ValueWorldBounds, ValueCount };
            size_t offsets[ValueCount] = {};
            size_t total = 0;
            auto reserve = [&](uint32_t slot, size_t elementSize)
            {
                offsets[slot] = total;
                total += (elementSize * count + 15) & ~size_t(15);
            };
            reserve(ValueNode, sizeof(SceneNode));
            reserve(ValueLinks, sizeof(SceneLinks));
            reserve(ValueDepth, sizeof(HierarchyDepth));
            reserve(ValueWorld, sizeof(WorldTransform));
            if (hasParent)
                reserve(ValueParent, sizeof(Parent));
            if (head.hasLocal)
                reserve(ValueLocal, sizeof(LocalTransform));
            if (hasLeaf)
            {
                reserve(ValueLeaf, sizeof(LeafObject));
                reserve(ValueLocalBounds, sizeof(LocalBounds));
            }
            if (S.valueBytes.size() < total + 16)
                S.valueBytes.resize(total + 16);
            uint8_t* base = reinterpret_cast<uint8_t*>(
                (reinterpret_cast<uintptr_t>(S.valueBytes.data()) + 15) & ~uintptr_t(15));

            SceneNode* sceneNodes = reinterpret_cast<SceneNode*>(base + offsets[ValueNode]);
            SceneLinks* links = reinterpret_cast<SceneLinks*>(base + offsets[ValueLinks]);
            HierarchyDepth* depths = reinterpret_cast<HierarchyDepth*>(base + offsets[ValueDepth]);
            WorldTransform* worlds = reinterpret_cast<WorldTransform*>(base + offsets[ValueWorld]);
            Parent* parents = reinterpret_cast<Parent*>(base + offsets[ValueParent]);
            LocalTransform* locals = reinterpret_cast<LocalTransform*>(base + offsets[ValueLocal]);
            LeafObject* leaves = reinterpret_cast<LeafObject*>(base + offsets[ValueLeaf]);
            LocalBounds* localBounds = reinterpret_cast<LocalBounds*>(base + offsets[ValueLocalBounds]);
            for (uint32_t k = 0; k < count; ++k)
            {
                const uint32_t i = first + k;
                if (i + c_PrefetchDistance < n && nodes[i + c_PrefetchDistance].record)
                {
                    const char* ahead = reinterpret_cast<const char*>(nodes[i + c_PrefetchDistance].record);
                    for (size_t line = 0; line < sizeof(OrphanRecord); line += 64)
                        Prefetch(ahead + line);
                }
                const NewNode& entry = nodes[i];
                const OrphanRecord& record = entry.record ? *entry.record : s_Defaults;
                new (sceneNodes + k) SceneNode(entry.node);
                links[k] = SceneLinks{ entityAt(S.firstChild[i]), entityAt(S.lastChild[i]),
                                       i == 0 ? rootPrev : entityAt(S.prevSibling[i]),
                                       i == 0 ? Entity::Null : entityAt(S.nextSibling[i]) };
                depths[k] = HierarchyDepth{ S.depth[i] };
                new (worlds + k) WorldTransform(record.world);
                if (hasParent)
                    parents[k] = Parent{ i == 0 ? parentEntity : S.entities[entry.parent] };
                if (head.hasLocal)
                    new (locals + k) LocalTransform(record.local);
                if (hasLeaf)
                {
                    new (leaves + k) LeafObject(entry.leaf);
                    new (localBounds + k) LocalBounds{ entry.localBounds };
                }
            }

            S.runComponents.clear();
            S.runValues.clear();
            auto add = [&](ComponentId id, const void* values)
            {
                S.runComponents.push_back(id);
                S.runValues.push_back(values);
            };
            add(m_Ids[SlotNode], sceneNodes);
            add(m_Ids[SlotLinks], links);
            add(m_DepthId, depths);
            add(m_Ids[SlotWorld], worlds);
            if (hasParent)
                add(m_Ids[SlotParent], parents);
            if (head.hasLocal)
            {
                add(m_Ids[SlotLocal], locals);
                add(m_Ids[SlotLocalMatrix], nullptr);   // composed by the next propagation (the rows are new)
            }
            if (hasLeaf)
            {
                add(m_Ids[SlotLeaf], leaves);
                add(m_Ids[SlotLocalBounds], localBounds);
                add(m_Ids[SlotWorldBounds], nullptr);   // written by the Bounds system before anything reads it
                add(m_Ids[SlotChunkBounds], nullptr);
                const LeafType& type = m_LeafTypes[head.leafType];
                for (uint32_t t = 0; t < type.tagCount; ++t)
                    add(m_LeafTypeTags[type.firstTag + t], nullptr);
            }

            if (s_FailCreateCountdown.load(std::memory_order_relaxed) >= 0 &&
                s_FailCreateCountdown.fetch_sub(1, std::memory_order_relaxed) == 0)
                add(ComponentId(0x7ffffff0u), nullptr);   // test seam: the world refuses the call and drops the values

            // The batch's nodes read their rows from now on (the leaf observer of the batch reads them): their records
            // leave them now and go back to the pool after the call. Their references and their leaves' stay held
            // until the rows hold theirs, so that a refused call cannot release the last one.
            for (uint32_t i = first; i < end; ++i)
            {
                nodes[i].node->m_Record = nullptr;
                nodes[i].node->AddRef();
                if (nodes[i].leaf)
                    nodes[i].leaf->AddRef();
            }

            EntityCreateDesc desc{};
            desc.components = S.runComponents.data();
            desc.values = S.runValues.data();
            desc.reservedEntities = S.entities.data() + first;
            desc.componentCount = uint32_t(S.runComponents.size());
            desc.entityCount = count;
            desc.transfer = ValueTransfer::Relocate;   // the world takes the references and the values
            const nvrhi::FRESULT result = m_World->CreateEntities(desc, nullptr);
            ++m_Counters.createCalls;
            if (NVRHI_FAILED(result))
            {
                log::warning("SceneGraph: cannot create %u entities (%d); the nodes from there on stay in no graph",
                             count, int(result));
                AbandonNodes(first);
                if (first > 0 && parentEntity != Entity::Null)
                    LinkUnder(parentEntity, S.entities[0]);
                SyncArchetypes();
                return false;
            }

            // the rows hold the references now; the nodes remember where their rows are
            S.records.clear();
            for (uint32_t i = first; i < end; ++i)
            {
                if (nodes[i].record)
                    S.records.push_back(nodes[i].record);
                if (nodes[i].leaf)
                    nodes[i].leaf->Release();
                RowRef row;
                if (Locate(S.entities[i], row))
                    RememberRow(nodes[i].node, row);
                nodes[i].node->Release();
            }
            OrphanRecord::Destroy(S.records.data(), S.records.size());
            first = end;
        }

        if (parentEntity != Entity::Null)
            LinkUnder(parentEntity, S.entities[0]);
        SyncArchetypes();
        return true;
    }

    // A refused CreateEntities call (out of memory): the batch from `first` lost the references it gave the world,
    // and holds the ones taken before the call; the nodes after it never gave theirs. Every node from `first` on goes
    // back to its record (links to the others are lost) with its leaf, its reserved id is released, and the reference
    // that would have gone to the world is released: a node nothing else holds goes, as with a Detach nobody keeps.
    void SceneGraphStore::AbandonNodes(uint32_t first)
    {
        Scratch& S = *m_S;
        std::vector<NewNode>& nodes = S.newNodes;
        const uint32_t n = uint32_t(nodes.size());
        if (NVRHI_FAILED(m_World->DestroyEntities(S.entities.data() + first, n - first)))
            log::warning("SceneGraph: cannot release the reserved ids of a failed attach");

        // The created nodes' links to abandoned ones go: in pre-order, a node's created children are a prefix of its
        // children, so its last child becomes the last created one.
        for (uint32_t i = 0; i < first; ++i)
        {
            SceneLinks* links = Get<SceneLinks>(S.entities[i], SlotLinks);
            if (!links)
                continue;
            uint32_t lastCreated = c_None;
            for (uint32_t child = S.firstChild[i]; child != c_None && child < first; child = S.nextSibling[child])
                lastCreated = child;
            if (lastCreated == c_None)
            {
                links->firstChild = Entity::Null;
                links->lastChild = Entity::Null;
            }
            else
                links->lastChild = S.entities[lastCreated];
            if (i != 0 && S.nextSibling[i] != c_None && S.nextSibling[i] >= first)
                links->nextSibling = Entity::Null;
        }
        for (uint32_t i = first; i < n; ++i)
        {
            NewNode& entry = nodes[i];
            SceneGraphNode* node = entry.node;
            node->m_Store = nullptr;
            node->m_Entity = Entity::Null;
            ForgetRow(node);
            OrphanRecord* record = entry.record ? entry.record : OrphanRecord::Create();
            record->parent = nullptr;
            record->prevSibling = nullptr;
            record->lastChild = nullptr;
            record->leaf.Attach(entry.leaf);   // the batch's held reference, or the one that never left
            node->m_Record = record;
            node->Release();
        }
    }

    // Appends `child`, whose SceneLinks already name the parent's old last child as its previous sibling.
    void SceneGraphStore::LinkUnder(Entity parentEntity, Entity child)
    {
        RowRef parentRow;
        if (!Locate(parentEntity, parentRow))
            return;
        SceneLinks* parentLinks = static_cast<SceneLinks*>(Column(parentRow, SlotLinks));
        if (parentLinks->lastChild != Entity::Null)
        {
            RowRef lastRow;
            if (Locate(parentLinks->lastChild, lastRow))
            {
                static_cast<SceneLinks*>(Column(lastRow, SlotLinks))->nextSibling = child;
                Stamp(lastRow, SlotLinks);
            }
        }
        else
            parentLinks->firstChild = child;
        parentLinks->lastChild = child;
        Stamp(parentRow, SlotLinks);
    }

    // ------------------------------------------------------------------------------- detach (5.3)

    // The subgraph of `root` in pre-order: m_S->entities, m_S->parents (indices) and m_S->nodes.
    void SceneGraphStore::GatherSubgraph(Entity root)
    {
        m_S->entities.clear();
        m_S->parents.clear();
        m_S->nodes.clear();
        uint32_t parentIndex = c_None;
        Entity entity = root;
        while (entity != Entity::Null)
        {
            const uint32_t index = uint32_t(m_S->entities.size());
            m_S->entities.push_back(entity);
            m_S->parents.push_back(parentIndex);
            m_S->nodes.push_back(NodeOf(entity));
            const SceneLinks* links = Get<SceneLinks>(entity, SlotLinks);
            if (links && links->firstChild != Entity::Null)
            {
                parentIndex = index;
                entity = links->firstChild;
                continue;
            }
            for (;;)
            {
                if (entity == root)
                {
                    entity = Entity::Null;
                    break;
                }
                const SceneLinks* current = Get<SceneLinks>(entity, SlotLinks);
                if (current && current->nextSibling != Entity::Null)
                {
                    entity = current->nextSibling;
                    break;
                }
                entity = m_S->entities[parentIndex];
                parentIndex = m_S->parents[parentIndex];
            }
        }
    }

    // Moves the gathered subgraph's rows into records, in pre-order: the children links of the records take references,
    // the root's caller keeps its own. With `unlinkRoot`, the root also leaves its parent's child list.
    void SceneGraphStore::MoveToRecords(bool unlinkRoot)
    {
        const uint32_t n = uint32_t(m_S->entities.size());
        if (unlinkRoot)
        {
            RowRef rootRow;
            if (Locate(m_S->entities[0], rootRow))
            {
                const SceneLinks rootLinks = *static_cast<const SceneLinks*>(Column(rootRow, SlotLinks));
                const Parent* parent = static_cast<const Parent*>(Column(rootRow, SlotParent));
                RowRef parentRow;
                if (parent && Locate(parent->entity, parentRow))
                {
                    SceneLinks* parentLinks = static_cast<SceneLinks*>(Column(parentRow, SlotLinks));
                    RowRef siblingRow;
                    if (rootLinks.prevSibling != Entity::Null && Locate(rootLinks.prevSibling, siblingRow))
                    {
                        static_cast<SceneLinks*>(Column(siblingRow, SlotLinks))->nextSibling = rootLinks.nextSibling;
                        Stamp(siblingRow, SlotLinks);
                    }
                    else
                        parentLinks->firstChild = rootLinks.nextSibling;
                    if (rootLinks.nextSibling != Entity::Null && Locate(rootLinks.nextSibling, siblingRow))
                    {
                        static_cast<SceneLinks*>(Column(siblingRow, SlotLinks))->prevSibling = rootLinks.prevSibling;
                        Stamp(siblingRow, SlotLinks);
                    }
                    else
                        parentLinks->lastChild = rootLinks.prevSibling;
                    Stamp(parentRow, SlotLinks);
                }
            }
        }

        for (uint32_t i = 0; i < n; ++i)
        {
            SceneGraphNode* node = m_S->nodes[i];
            OrphanRecord* record = OrphanRecord::Create();
            RowRef row;
            if (Locate(m_S->entities[i], row))
            {
                if (const LocalTransform* local = static_cast<const LocalTransform*>(Column(row, SlotLocal)))
                {
                    record->local = *local;
                    record->hasLocal = true;
                }
                if (const WorldTransform* world = static_cast<const WorldTransform*>(Column(row, SlotWorld)))
                    record->world = *world;
                if (const LeafObject* leaf = static_cast<const LeafObject*>(Column(row, SlotLeaf)))
                    record->leaf = leaf->object;
            }
            if (i > 0)
            {
                SceneGraphNode* parent = m_S->nodes[m_S->parents[i]];
                OrphanRecord& parentRecord = *parent->m_Record;
                record->parent = parent;
                record->prevSibling = parentRecord.lastChild;
                (parentRecord.lastChild ? parentRecord.lastChild->m_Record->nextSibling : parentRecord.firstChild) =
                    node;
                parentRecord.lastChild = node;
            }
            node->m_Record = record;
            node->m_Store = nullptr;
            node->m_Entity = Entity::Null;
            node->m_CacheEpoch.store(0, std::memory_order_relaxed);
            ForgetRow(node);
        }
    }

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::Detach(SceneGraphNode* node)
    {
        assert(node->m_Store == this);
        nvrhi::AutoPtr<SceneGraphNode> result(node);
        if (!CanChangeStructure("Detach"))
            return result;
        ScratchScope scope(*this);

        GatherSubgraph(node->m_Entity);
        MoveToRecords(true);

        // The leaf observer unregisters the leaves in pre-order; the graph removes them from its vectors in one pass.
        m_Graph->m_DeferLeafRemoval = true;
        if (NVRHI_FAILED(m_World->DestroyEntities(m_S->entities.data(), uint32_t(m_S->entities.size()))))
            log::warning("SceneGraph: cannot destroy the entities of a detached subgraph");
        m_Graph->m_DeferLeafRemoval = false;
        m_Graph->CompactLeafVectors();

        SyncArchetypes();
        m_PendingStructure = true;
        BumpEpoch();
        return result;
    }

    // ------------------------------------------------------------------------------- node writes (5.2)

    void SceneGraphStore::SetLeaf(SceneGraphNode* node, SceneGraphLeaf* leaf)
    {
        if (!CanChangeStructure("SetLeaf"))
            return;
        const Entity entity = node->m_Entity;
        RowRef row;
        if (!Locate(entity, row))
            return;
        const LeafObject* current = static_cast<const LeafObject*>(Column(row, SlotLeaf));
        nvrhi::AutoPtr<SceneGraphLeaf> old = current ? current->object : nullptr;
        const uint32_t oldType = old ? ResolveLeafType(old) : c_None;
        const uint32_t newType = leaf ? ResolveLeafType(leaf) : c_None;

        // The old leaf leaves first (its OnRemove unregisters it), then the new one comes (OnAdd registers it), as
        // the old SetLeaf unregistered and registered.
        if (old)
        {
            old->m_Node.Reset();
            m_World->RemoveComponent(entity, m_Ids[SlotLeaf]);
        }
        if (oldType != newType)
        {
            if (old)
            {
                const LeafType& type = m_LeafTypes[oldType];
                for (uint32_t t = 0; t < type.tagCount; ++t)
                    m_World->RemoveComponent(entity, m_LeafTypeTags[type.firstTag + t]);
                if (!leaf)
                {
                    m_World->RemoveComponent(entity, m_Ids[SlotLocalBounds]);
                    m_World->RemoveComponent(entity, m_Ids[SlotWorldBounds]);
                    m_World->RemoveComponent(entity, m_Ids[SlotChunkBounds]);
                }
            }
            if (leaf)
            {
                if (!old)
                {
                    const WorldBounds empty{ dm::box3::empty() };
                    m_World->AddComponent(entity, m_Ids[SlotWorldBounds], &empty);
                    m_World->AddComponent(entity, m_Ids[SlotChunkBounds], nullptr);
                }
                const LeafType& type = m_LeafTypes[newType];
                for (uint32_t t = 0; t < type.tagCount; ++t)
                    m_World->AddComponent(entity, m_LeafTypeTags[type.firstTag + t], nullptr);
            }
        }
        if (leaf)
        {
            leaf->m_Node = nvrhi::WeakPtr<SceneGraphNode>(node);
            const LocalBounds bounds{ leaf->GetLocalBoundingBox() };
            m_World->AddComponent(entity, m_Ids[SlotLocalBounds], &bounds);   // writes it when it is there
            LeafObject value(leaf);
            leaf->AddRef();   // `value` holds one reference; AddComponent copies it into the row
            m_World->AddComponent(entity, m_Ids[SlotLeaf], &value);
        }

        SyncArchetypes();
        m_PendingStructure = true;
        BumpEpoch();
    }

    void SceneGraphStore::SetTransform(SceneGraphNode* node, const dm::double3* translation, const dm::dquat* rotation,
                                       const dm::double3* scaling)
    {
        assert(!m_World->IsInPhase());   // design 5.6: the API writes outside phases
        RowRef row;
        if (!LocateNode(node, row))
            return;
        RememberRow(node, row);
        // the hot path of the API (design 10.1): one column lookup, the write, the stamp
        const uint8_t column = ColumnIndex(*row.archetype, row.archetypeIndex, SlotLocal);
        if (column != c_Absent)
        {
            const ArchetypeColumn& c = row.archetype->columns[column];
            uint8_t* memory = row.archetype->chunks[row.chunk].memory;
            LocalTransform* local = reinterpret_cast<LocalTransform*>(memory + c.offset + size_t(row.row) * c.stride);
            if (scaling) local->scaling = *scaling;
            if (rotation) local->rotation = *rotation;
            if (translation) local->translation = *translation;
            StampVersion(row.archetype->Versions(row.chunk) + column, m_Impl->GetStampVersion());
        }
        else
        {
            // the first transform of a node adds the component (a structural change, outside phases)
            if (!CanChangeStructure("SetTransform"))
                return;
            static const OrphanRecord s_Defaults;
            LocalTransform value = s_Defaults.local;
            if (scaling) value.scaling = *scaling;
            if (rotation) value.rotation = *rotation;
            if (translation) value.translation = *translation;
            m_World->AddComponent(node->m_Entity, m_Ids[SlotLocal], &value);
            m_World->AddComponent(node->m_Entity, m_Ids[SlotLocalMatrix], nullptr);
            SyncArchetypes();
        }
        m_PendingTransform = true;
    }

    void SceneGraphStore::InvalidateContent(SceneGraphNode* node)
    {
        RowRef row;
        if (Locate(node->m_Entity, row))
        {
            const LeafObject* leaf = static_cast<const LeafObject*>(Column(row, SlotLeaf));
            LocalBounds* bounds = static_cast<LocalBounds*>(Column(row, SlotLocalBounds));
            if (leaf && leaf->object && bounds)
            {
                bounds->bounds = leaf->object->GetLocalBoundingBox();
                Stamp(row, SlotLocalBounds);
                Stamp(row, SlotLeaf);   // consumers that want to know see the leaf column changed (5.4)
                m_PendingContent = true;   // the next Refresh runs Bounds for it
            }
        }
        BumpEpoch();
    }

    // ------------------------------------------------------------------------------------- refresh

    bool SceneGraphStore::Refresh(uint32_t frameIndex)
    {
        SyncArchetypes();
        const bool structure = m_PendingStructure;
        const bool moved = m_PendingTransform;
        // Nothing changed through the API (structure, transforms, content), no structural change and no phase ran on
        // the world since: nothing to do. Writes into the world outside phases are not seen here (SceneGraph.h).
        if (!structure && !moved && !m_PendingContent &&
            m_World->GetStructuralVersion() == m_RefreshedStructuralVersion &&
            m_World->GetVersion() == m_RefreshedVersion)
        {
            m_LastRefreshMoved = false;
            ++m_Counters.skippedRefreshes;
            return false;
        }

        m_FrameIndex = frameIndex;
        const nvrhi::FRESULT result = m_Scheduler ? m_Scheduler->RunPhases(Phase::TransformPropagation, Phase::Bounds)
                                                  : nvrhi::FE_INVALID_ARGS;
        if (NVRHI_FAILED(result))
        {
            // the changes stay pending for the next Refresh
            log::warning("SceneGraph::Refresh: the phases did not run (no scheduler, or inside a phase); the changes "
                         "stay pending");
            return false;
        }
        ++m_Counters.refreshes;

        m_PendingStructure = false;
        m_PendingTransform = false;
        m_PendingContent = false;
        m_LastRefreshMoved = moved;
        m_RefreshedStructuralVersion = m_World->GetStructuralVersion();
        m_RefreshedVersion = m_World->GetVersion();
        BumpEpoch();
        return structure;
    }

    void SceneGraphStore::UpdateSystemLists() noexcept
    {
        const uint32_t count = m_Impl->GetArchetypeStorageCount();
        for (uint32_t a = m_SystemArchetypesSeen; a < count; ++a)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
            auto column = [&](ComponentId id) -> uint8_t
            {
                const uint32_t c = id == ComponentId::Invalid ? c_NoColumn : archetype.FindColumn(id);
                return c == c_NoColumn ? c_Absent : uint8_t(c);
            };
            const uint8_t world = column(m_Ids[SlotWorld]);
            const uint8_t local = column(m_Ids[SlotLocal]);
            uint8_t level = c_Absent;
            if (world != c_Absent && archetype.components.size() > 0)
            {
                uint32_t depth = 0;
                const uint8_t depthColumn = column(m_DepthId);
                if (depthColumn != c_Absent)
                    std::memcpy(&depth, archetype.sharedValues + archetype.columns[depthColumn].offset, sizeof(depth));
                level = uint8_t(std::min(depth, c_MaxHierarchyDepth));
                m_ArchetypeLevelSlot.resize(a + 1, c_None);
                m_ArchetypeLevelSlot[a] = uint32_t(m_Levels[level].size());
                m_Levels[level].push_back(
                    LevelArchetype{ a, local, column(m_Ids[SlotLocalMatrix]), column(m_Ids[SlotParent]), world });

                const uint8_t localBounds = column(m_Ids[SlotLocalBounds]);
                const uint8_t worldBounds = column(m_Ids[SlotWorldBounds]);
                const uint8_t chunkBounds = column(m_Ids[SlotChunkBounds]);
                if (localBounds != c_Absent && worldBounds != c_Absent && chunkBounds != c_Absent)
                    m_BoundsArchetypes.push_back(BoundsArchetype{ a, world, localBounds, worldBounds, chunkBounds });
            }
            m_ArchetypeLevel.push_back(level);

            const uint8_t leaf = column(m_Ids[SlotLeaf]);
            if (column(m_JointTag) != c_Absent && local != c_Absent && leaf != c_Absent)
                m_JointArchetypes.push_back(JointArchetype{ a, local, leaf });
        }
        m_SystemArchetypesSeen = count;
    }

    // ------------------------------------------------------------------- TransformPropagation (4.6, D5)

    // Whether the parent's chunk was written in this run (its WorldTransform version is this run's), and its world.
    bool SceneGraphStore::ParentWritten(Entity parent, uint64_t version, const dm::daffine3** outWorld,
                                        bool readVersion) const noexcept
    {
        const EntityRecord* record = m_Impl->FindAlive(parent);
        if (!record)
        {
            *outWorld = nullptr;
            return false;
        }
        const Archetype& archetype = m_Impl->GetArchetypeStorage(record->archetype);
        const uint8_t column = ColumnIndex(archetype, record->archetype, SlotWorld);
        if (column == c_Absent)
        {
            *outWorld = nullptr;
            return false;
        }
        *outWorld = &reinterpret_cast<const WorldTransform*>(ColumnBase(archetype, record->chunk, column))[record->row]
            .transform;
        // a shallower level's chunk: finished before this level started, so a plain read
        return readVersion && archetype.Versions(record->chunk)[column] == version;
    }

    // world = local * parent world for a row with a local transform, a bit copy of the parent's world without one
    // (no multiplication: no rounding, signed zeros kept); local, or the identity, for a row without a parent.
    // With `compose`, the local transform is composed again (its chunk's LocalTransform changed) and cached in
    // LocalMatrix; otherwise the cache is current.
    void SceneGraphStore::PropagateRow(const LevelArchetype& level, const Archetype& archetype, uint32_t chunk,
                                       uint32_t row, const dm::daffine3* parentWorld, bool compose) const noexcept
    {
        WorldTransform* worlds = reinterpret_cast<WorldTransform*>(ColumnBase(archetype, chunk, level.world));
        if (level.local == c_Absent)
        {
            worlds[row].transform = parentWorld ? *parentWorld : dm::daffine3::identity();
            return;
        }
        LocalMatrix* matrices = level.matrix != c_Absent
            ? reinterpret_cast<LocalMatrix*>(ColumnBase(archetype, chunk, level.matrix)) : nullptr;
        dm::daffine3 localTransform;
        if (compose || !matrices)
        {
            const LocalTransform* locals =
                reinterpret_cast<const LocalTransform*>(ColumnBase(archetype, chunk, level.local));
            localTransform = ComposeLocal(locals[row]);
            if (matrices)
                matrices[row].transform = localTransform;
        }
        else
            localTransform = matrices[row].transform;
        worlds[row].transform = parentWorld ? localTransform * *parentWorld : localTransform;
    }

    void SceneGraphStore::RunPropagationTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        PropagationContext& ctx = *static_cast<PropagationContext*>(context);
        const SceneGraphStore& store = *ctx.store;
        const std::vector<LevelArchetype>& levels = store.m_Levels[ctx.level];
        uint64_t chunks = 0;
        uint64_t rows = 0;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_Tasks[t];
            const LevelArchetype& level = levels[task.archetype];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(level.archetype);
            const uint32_t count = archetype.chunks[task.chunk].count;
            const Parent* parents = level.parent != c_Absent
                ? reinterpret_cast<const Parent*>(ColumnBase(archetype, task.chunk, level.parent)) : nullptr;

            uint32_t written = 0;
            Entity cachedParent = Entity::Null;
            const dm::daffine3* parentWorld = nullptr;
            bool parentWritten = false;
            for (uint32_t r = 0; r < count; ++r)
            {
                if (parents)
                {
                    const Entity parent = parents[r].entity;
                    if (parent != cachedParent)
                    {
                        cachedParent = parent;
                        parentWritten = store.ParentWritten(parent, ctx.version, &parentWorld, true);
                    }
                    if (!parentWorld || (!task.whole && !parentWritten))
                        continue;
                }
                else if (!task.whole)
                    continue;
                store.PropagateRow(level, archetype, task.chunk, r, parents ? parentWorld : nullptr, task.whole);
                ++written;
            }
            if (written)
            {
                if (task.whole && level.matrix != c_Absent)
                    StampVersion(archetype.Versions(task.chunk) + level.matrix, ctx.version);
                StampVersion(archetype.Versions(task.chunk) + level.world, ctx.version);
                ++chunks;
                rows += written;
            }
        }
        if (chunks)
        {
            ctx.wrote.store(true, std::memory_order_relaxed);
            ctx.chunks.fetch_add(chunks, std::memory_order_relaxed);
            ctx.rows.fetch_add(rows, std::memory_order_relaxed);
        }
    }

    nvrhi::FRESULT SceneGraphStore::RunPropagation(ISystemContext* context) noexcept
    {
        const uint64_t since = context->GetLastRunVersion();
        const uint64_t version = context->GetVersion();
        IJobSystem* jobs = context->GetJobSystem();
        UpdateSystemLists();

        // Level by level: a level reads the worlds that the previous one finished. A chunk is computed whole when its
        // own local transforms, parents or rows changed since the last run; otherwise only its rows whose parent's
        // chunk this run wrote, and only when the level above wrote anything.
        bool levelAboveWrote = false;
        for (uint32_t d = 0; d < c_MaxHierarchyDepth; ++d)
        {
            const std::vector<LevelArchetype>& levels = m_Levels[d];
            m_Tasks.clear();
            for (uint32_t i = 0; i < uint32_t(levels.size()); ++i)
            {
                const LevelArchetype& level = levels[i];
                const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    if (archetype.chunks[c].count == 0)
                        continue;
                    const uint64_t* versions = archetype.Versions(c);
                    const bool whole = (level.local != c_Absent && versions[level.local] > since) ||
                                       (level.parent != c_Absent && versions[level.parent] > since) ||
                                       versions[archetype.AddedVersionSlot()] > since;
                    if (whole || levelAboveWrote)
                        m_Tasks.push_back(Task{ i, c, whole });
                }
            }
            if (m_Tasks.empty())
            {
                levelAboveWrote = false;
                continue;
            }

            PropagationContext ctx;
            ctx.store = this;
            ctx.level = d;
            ctx.since = since;
            ctx.version = version;
            const uint32_t taskCount = uint32_t(m_Tasks.size());
            const uint32_t grain = std::max(1u, taskCount / (WorkingThreads(jobs) * 4));
            if (!jobs || grain >= taskCount)
                RunPropagationTasks(&ctx, 0, taskCount, 0);
            else
                jobs->ParallelFor(taskCount, grain, &RunPropagationTasks, &ctx);
            levelAboveWrote = ctx.wrote.load(std::memory_order_relaxed);
            m_Counters.propagatedChunks += ctx.chunks.load(std::memory_order_relaxed);
            m_Counters.propagatedRows += ctx.rows.load(std::memory_order_relaxed);
        }
        PropagateDeepBucket(jobs, since, version, levelAboveWrote);
        MarkSkinnedJoints(since);
        return nvrhi::FS_OK;
    }

    // The last bucket holds every entity at depth c_MaxHierarchyDepth or deeper, so a row's parent may be in it too.
    // Its entries (rows whose parent is in a shallower bucket) root independent subtrees, which run in parallel, each
    // depth first through SceneLinks, so that a parent is always computed before its children. Rows that no entry
    // reaches (entities with a Parent but no SceneLinks) take a serial pass that resolves each row's parent first.
    // Marks per entity index record which rows were visited and written; each is written by the one thread that owns
    // its row.
    struct SceneGraphStore::DeepContext
    {
        SceneGraphStore* store;
        uint64_t since;
        uint64_t version;
        std::atomic<uint64_t> rows{ 0 };
        std::atomic<uint64_t> written{ 0 };
    };

    namespace
    {
        constexpr uint8_t c_DeepVisited = 1;
        constexpr uint8_t c_DeepWritten = 2;
    }

    const SceneGraphStore::LevelArchetype* SceneGraphStore::DeepLevel(uint32_t archetypeIndex) const noexcept
    {
        if (archetypeIndex >= m_ArchetypeLevel.size() || m_ArchetypeLevel[archetypeIndex] != c_MaxHierarchyDepth)
            return nullptr;
        return &m_Levels[c_MaxHierarchyDepth][m_ArchetypeLevelSlot[archetypeIndex]];
    }

    bool SceneGraphStore::DeepChunkDirty(const LevelArchetype& level, const Archetype& archetype, uint32_t chunk,
                                         uint64_t since) noexcept
    {
        const uint64_t* versions = archetype.Versions(chunk);
        return (level.local != c_Absent && versions[level.local] > since) ||
               (level.parent != c_Absent && versions[level.parent] > since) ||
               versions[archetype.AddedVersionSlot()] > since;
    }

    // One row of the last bucket whose parent is final: computed when its chunk changed or its parent was written.
    bool SceneGraphStore::PropagateDeepRow(Entity entity, bool parentInBucket, uint64_t since,
                                           uint64_t version) noexcept
    {
        const uint32_t index = GetEntityIndex(entity);
        const EntityRecord& record = m_Impl->GetEntityRecord(index);
        const LevelArchetype* level = DeepLevel(record.archetype);
        m_DeepMarks[index] |= c_DeepVisited;
        if (!level || level->parent == c_Absent)
            return false;
        const Archetype& archetype = m_Impl->GetArchetypeStorage(record.archetype);
        const Entity parent =
            reinterpret_cast<const Parent*>(ColumnBase(archetype, record.chunk, level->parent))[record.row].entity;
        // A parent in the bucket is in this subtree, computed on this thread before its children: its mark says whether
        // it was written; its chunk's version may be stamped by another subtree at this moment, so it is not read.
        const dm::daffine3* parentWorld = nullptr;
        bool parentWritten = ParentWritten(parent, version, &parentWorld, !parentInBucket);
        if (parentInBucket)
            parentWritten = (m_DeepMarks[GetEntityIndex(parent)] & c_DeepWritten) != 0;
        const bool dirty = DeepChunkDirty(*level, archetype, record.chunk, since);
        if (!parentWorld || (!parentWritten && !dirty))
            return false;
        PropagateRow(*level, archetype, record.chunk, record.row, parentWorld, dirty);
        uint64_t* versions = archetype.Versions(record.chunk);
        if (dirty && level->matrix != c_Absent)
            StampVersion(versions + level->matrix, version);
        StampVersion(versions + level->world, version);
        m_DeepMarks[index] |= c_DeepWritten;
        return true;
    }

    void SceneGraphStore::RunDeepTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        DeepContext& ctx = *static_cast<DeepContext*>(context);
        SceneGraphStore& store = *ctx.store;
        thread_local std::vector<Entity> stack;   // per thread, kept for its capacity
        uint64_t rows = 0;
        uint64_t written = 0;
        for (uint32_t e = begin; e < end; ++e)
        {
            const Entity entry = store.m_DeepEntries[e];
            stack.clear();
            stack.push_back(entry);
            while (!stack.empty())
            {
                const Entity entity = stack.back();
                stack.pop_back();
                written += store.PropagateDeepRow(entity, entity != entry, ctx.since, ctx.version) ? 1 : 0;
                ++rows;
                const SceneLinks* links = store.Get<SceneLinks>(entity, SlotLinks);
                for (Entity child = links ? links->firstChild : Entity::Null; child != Entity::Null;)
                {
                    stack.push_back(child);
                    const SceneLinks* childLinks = store.Get<SceneLinks>(child, SlotLinks);
                    child = childLinks ? childLinks->nextSibling : Entity::Null;
                }
            }
        }
        ctx.rows.fetch_add(rows, std::memory_order_relaxed);
        ctx.written.fetch_add(written, std::memory_order_relaxed);
    }

    void SceneGraphStore::PropagateDeepBucket(IJobSystem* jobs, uint64_t since, uint64_t version, bool parentLevelWrote)
    {
        const std::vector<LevelArchetype>& levels = m_Levels[c_MaxHierarchyDepth];
        if (levels.empty())
            return;
        bool any = parentLevelWrote;
        for (const LevelArchetype& level : levels)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount() && !any; ++c)
                any = archetype.chunks[c].count != 0 && DeepChunkDirty(level, archetype, c, since);
        }
        if (!any)
            return;

        if (m_DeepMarks.size() < m_Impl->GetEntityTableSize())
            m_DeepMarks.resize(m_Impl->GetEntityTableSize(), 0);

        // The entries, in storage order.
        m_DeepEntries.clear();
        for (const LevelArchetype& level : levels)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
            if (level.parent == c_Absent)
                continue;
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const Entity* entities = archetype.Entities(c);
                const Parent* parents = reinterpret_cast<const Parent*>(ColumnBase(archetype, c, level.parent));
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                {
                    const EntityRecord* parentRecord = m_Impl->FindAlive(parents[r].entity);
                    if (parentRecord && !DeepLevel(parentRecord->archetype))
                        m_DeepEntries.push_back(entities[r]);
                }
            }
        }

        DeepContext ctx;
        ctx.store = this;
        ctx.since = since;
        ctx.version = version;
        const uint32_t entryCount = uint32_t(m_DeepEntries.size());
        const uint32_t grain = std::max(1u, entryCount / (WorkingThreads(jobs) * 4));
        if (entryCount != 0)
        {
            if (!jobs || grain >= entryCount)
                RunDeepTasks(&ctx, 0, entryCount, 0);
            else
                jobs->ParallelFor(entryCount, grain, &RunDeepTasks, &ctx);
        }
        m_Counters.deepRows += ctx.rows.load(std::memory_order_relaxed);
        m_Counters.propagatedRows += ctx.written.load(std::memory_order_relaxed);

        // The rows that no entry reached, serially: each resolves its parent first (an explicit stack).
        for (const LevelArchetype& start : levels)
        {
            const Archetype& startArchetype = m_Impl->GetArchetypeStorage(start.archetype);
            for (uint32_t c = 0; c < startArchetype.GetChunkCount(); ++c)
            {
                const Entity* entities = startArchetype.Entities(c);
                for (uint32_t r = 0; r < startArchetype.chunks[c].count; ++r)
                {
                    if (m_DeepMarks[GetEntityIndex(entities[r])] & c_DeepVisited)
                        continue;
                    m_DeepStack.clear();
                    m_DeepStack.push_back(entities[r]);
                    while (!m_DeepStack.empty())
                    {
                        const Entity entity = m_DeepStack.back();
                        const uint32_t index = GetEntityIndex(entity);
                        if (m_DeepMarks[index] & c_DeepVisited)
                        {
                            m_DeepStack.pop_back();
                            continue;
                        }
                        const EntityRecord& record = m_Impl->GetEntityRecord(index);
                        const LevelArchetype* level = DeepLevel(record.archetype);
                        bool parentInBucket = false;
                        if (level && level->parent != c_Absent)
                        {
                            const Archetype& archetype = m_Impl->GetArchetypeStorage(record.archetype);
                            const Entity parent = reinterpret_cast<const Parent*>(
                                ColumnBase(archetype, record.chunk, level->parent))[record.row].entity;
                            const EntityRecord* parentRecord = m_Impl->FindAlive(parent);
                            parentInBucket = parentRecord && DeepLevel(parentRecord->archetype);
                            if (parentInBucket && !(m_DeepMarks[GetEntityIndex(parent)] & c_DeepVisited))
                            {
                                m_DeepStack.push_back(parent);
                                continue;
                            }
                        }
                        m_Counters.propagatedRows += PropagateDeepRow(entity, parentInBucket, since, version) ? 1 : 0;
                        ++m_Counters.deepRows;
                        m_DeepStack.pop_back();
                    }
                }
            }
        }

        for (const LevelArchetype& level : levels)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const Entity* entities = archetype.Entities(c);
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                    m_DeepMarks[GetEntityIndex(entities[r])] = 0;
            }
        }
    }

    // SkinnedMeshInstance::GetLastUpdateFrameIndex: the frame at which a joint node's own local transform changed. Per
    // chunk (D9), so a joint that shares a chunk with one that moved counts as moved; that only adds a skinning pass.
    void SceneGraphStore::MarkSkinnedJoints(uint64_t since)
    {
        for (const JointArchetype& joints : m_JointArchetypes)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(joints.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const uint64_t* versions = archetype.Versions(c);
                if (versions[joints.local] <= since && versions[archetype.AddedVersionSlot()] <= since)
                    continue;
                const LeafObject* leaves = reinterpret_cast<const LeafObject*>(ColumnBase(archetype, c, joints.leaf));
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                {
                    if (auto reference = query_cast<SkinnedMeshReference>(leaves[r].object))
                    {
                        if (auto instance = reference->m_Instance.Lock())
                            instance->m_LastUpdateFrameIndex = m_FrameIndex;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------ Bounds (4.7)

    void SceneGraphStore::RunBoundsTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        BoundsContext& ctx = *static_cast<BoundsContext*>(context);
        const SceneGraphStore& store = *ctx.store;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_Tasks[t];
            const BoundsArchetype& bounds = store.m_BoundsArchetypes[task.archetype];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(bounds.archetype);
            const uint32_t count = archetype.chunks[task.chunk].count;
            const WorldTransform* worlds =
                reinterpret_cast<const WorldTransform*>(ColumnBase(archetype, task.chunk, bounds.world));
            const LocalBounds* locals =
                reinterpret_cast<const LocalBounds*>(ColumnBase(archetype, task.chunk, bounds.localBounds));
            WorldBounds* results =
                reinterpret_cast<WorldBounds*>(ColumnBase(archetype, task.chunk, bounds.worldBounds));
            dm::box3 chunkBounds = dm::box3::empty();
            for (uint32_t r = 0; r < count; ++r)
            {
                const dm::box3& local = locals[r].bounds;
                results[r].bounds = local.isempty() ? dm::box3::empty() : local * dm::affine3(worlds[r].transform);
                chunkBounds |= results[r].bounds;
            }
            reinterpret_cast<ChunkWorldBounds*>(ColumnBase(archetype, task.chunk, bounds.chunkBounds))->bounds =
                chunkBounds;
            uint64_t* versions = archetype.Versions(task.chunk);
            StampVersion(versions + bounds.worldBounds, ctx.version);
            StampVersion(versions + bounds.chunkBounds, ctx.version);
        }
        ctx.chunks.fetch_add(end - begin, std::memory_order_relaxed);
    }

    nvrhi::FRESULT SceneGraphStore::RunBounds(ISystemContext* context) noexcept
    {
        const uint64_t since = context->GetLastRunVersion();
        IJobSystem* jobs = context->GetJobSystem();
        UpdateSystemLists();
        m_Tasks.clear();
        for (uint32_t i = 0; i < uint32_t(m_BoundsArchetypes.size()); ++i)
        {
            const BoundsArchetype& bounds = m_BoundsArchetypes[i];
            const Archetype& archetype = m_Impl->GetArchetypeStorage(bounds.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                if (archetype.chunks[c].count == 0)
                    continue;
                const uint64_t* versions = archetype.Versions(c);
                if (versions[bounds.world] > since || versions[bounds.localBounds] > since ||
                    versions[archetype.AddedVersionSlot()] > since)
                    m_Tasks.push_back(Task{ i, c, true });
            }
        }
        if (m_Tasks.empty())
            return nvrhi::FS_OK;

        BoundsContext ctx;
        ctx.store = this;
        ctx.version = context->GetVersion();
        const uint32_t taskCount = uint32_t(m_Tasks.size());
        const uint32_t grain = std::max(1u, taskCount / (WorkingThreads(jobs) * 4));
        if (!jobs || grain >= taskCount)
            RunBoundsTasks(&ctx, 0, taskCount, 0);
        else
            jobs->ParallelFor(taskCount, grain, &RunBoundsTasks, &ctx);
        m_Counters.boundsChunks += ctx.chunks.load(std::memory_order_relaxed);
        return nvrhi::FS_OK;
    }

    // ---------------------------------------------------------------------- subgraph caches (5.7)

    void SceneGraphStore::ComputeSubgraph(const SceneGraphNode* node, std::vector<CacheFrame>& stack, uint64_t epoch,
                                          bool publish, dm::box3& outBounds, SceneContentFlags& outLeafContent,
                                          SceneContentFlags& outContent)
    {
        // The node's own part: its leaf's box in world space (an empty box adds nothing) and its content flags.
        auto start = [](const SceneGraphNode* current)
        {
            CacheFrame frame{ current, current->GetFirstChild(), dm::box3::empty(), SceneContentFlags::None,
                              SceneContentFlags::None };
            if (SceneGraphLeaf* leaf = current->GetLeaf())
            {
                const dm::box3 localBoundingBox = leaf->GetLocalBoundingBox();
                if (!localBoundingBox.isempty())
                    frame.bounds = localBoundingBox * current->GetLocalToWorldTransformFloat();
                frame.leafContent = leaf->GetContentFlags();
            }
            frame.content = frame.leafContent;
            return frame;
        };
        auto valid = [&](const SceneGraphNode* current)
        {
            return publish && current->m_CacheEpoch.load(std::memory_order_acquire) == epoch;
        };

        stack.clear();
        stack.push_back(start(node));
        for (;;)
        {
            CacheFrame& top = stack.back();
            if (SceneGraphNode* child = top.nextChild)
            {
                top.nextChild = child->GetNextSibling();
                if (valid(child))
                {
                    top.bounds |= child->m_CachedBounds;
                    top.content |= child->m_CachedSubgraphContent;
                }
                else
                    stack.push_back(start(child));
                continue;
            }

            const CacheFrame done = top;
            stack.pop_back();
            if (publish)
            {
                done.node->m_CachedBounds = done.bounds;
                done.node->m_CachedLeafContent = done.leafContent;
                done.node->m_CachedSubgraphContent = done.content;
                done.node->m_CacheEpoch.store(epoch, std::memory_order_release);
            }
            if (stack.empty())
            {
                outBounds = done.bounds;
                outLeafContent = done.leafContent;
                outContent = done.content;
                return;
            }
            stack.back().bounds |= done.bounds;
            stack.back().content |= done.content;
        }
    }

    void SceneGraphStore::ComputeSubgraphCache(const SceneGraphNode* node)
    {
        std::lock_guard<std::mutex> lock(m_CacheMutex);
        const uint64_t epoch = GetEpoch();
        if (node->m_CacheEpoch.load(std::memory_order_acquire) == epoch)
            return;
        dm::box3 bounds;
        SceneContentFlags leafContent;
        SceneContentFlags content;
        ComputeSubgraph(node, m_CacheStack, epoch, true, bounds, leafContent, content);
    }
}
