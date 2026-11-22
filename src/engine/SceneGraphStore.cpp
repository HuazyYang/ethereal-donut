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

// The scene graph over the world (plan steps E3, E4): storage, structural operations, the TransformPropagation and
// Bounds systems, the leaf observer, and the mesh-instance columns and slots. See SceneGraphStore.h for the rules.

#include "SceneGraphStore.h"
#include "EcsSchedulerImpl.h"

#include <donut/core/log.h>
#include <donut/core/query_cast.h>

#include <algorithm>
#include <cassert>
#include <exception>
#include <cstring>
#include <limits>
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

        // The graph's observer on LeafObject (design 5.5): it fills a mesh instance's columns (asset tables, slots)
        // and maintains the leaf vectors through the virtual RegisterLeaf and UnregisterLeaf, in creation and
        // destruction order.
        class LeafObserver final : public nvrhi::ObjectImpl<IObserver>
        {
        public:
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(LeafObserver)
            NVRHI_IMPLEMENTS_INTERFACE(IObserver)
            NVRHI_END_INTERFACE_TABLE()

            using Register = void (*)(SceneGraphStore* store, Entity entity, SceneGraphLeaf* leaf, bool added);

            LeafObserver(SceneGraphStore* store, Register registerLeaf) noexcept
                : m_Store(store), m_Register(registerLeaf) { }

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
                        m_Register(m_Store, batch.entities[i], leaves[i].object, added);
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
            SceneGraphStore* m_Store;   // not a reference: the store owns the world, which owns this observer
            Register m_Register;
        };

        // A process-wide serial per store (SceneGraphStore::GetSerial).
        std::atomic<uint64_t> g_StoreSerial{ 0 };

        RenderFlags MakeRenderFlags(MeshInstance* instance) noexcept
        {
            RenderFlags flags{ 0, uint32_t(instance->GetContentFlags()) };
            if (const MeshInfo* mesh = instance->GetMesh())
            {
                // as Scene::UpdateInstance set InstanceData::flags
                if (mesh->type == MeshType::CurveDisjointOrthogonalTriangleStrips)
                    flags.instanceFlags |= c_InstanceFlagCurveDisjointOrthogonalTriangleStrips;
                else if (mesh->type == MeshType::CurveLinearSweptSpheres)
                    flags.instanceFlags |= c_InstanceFlagCurveLinearSweptSpheres;
            }
            return flags;
        }
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

    namespace
    {
        // The subgraph rule for NaN: a box with a NaN anywhere counts as the infinite box (it is never culled, and a
        // union with it is the same in any order), in the pass, its summaries, the Bounds system's chunk union
        // (ChunkWorldBounds) and the lazy path.
        inline bool HasNan(const dm::box3& box) noexcept
        {
            return box.m_mins.x != box.m_mins.x || box.m_mins.y != box.m_mins.y || box.m_mins.z != box.m_mins.z ||
                   box.m_maxs.x != box.m_maxs.x || box.m_maxs.y != box.m_maxs.y || box.m_maxs.z != box.m_maxs.z;
        }

        inline dm::box3 InfiniteBox() noexcept
        {
            const float infinity = std::numeric_limits<float>::infinity();
            return dm::box3(dm::float3(-infinity), dm::float3(infinity));
        }

        inline dm::box3 NanToInfinite(const dm::box3& box) noexcept
        {
            return HasNan(box) ? InfiniteBox() : box;
        }
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
        bool subgraph;   // an incremental subgraph pass is prepared: summaries and marks of the chunks written
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
        m_Ids[SlotMeshRef] = RegisterComponent<MeshRef>(m_World);
        m_Ids[SlotRenderFlags] = RegisterComponent<RenderFlags>(m_World);
        m_Ids[SlotInstance] = RegisterComponent<GpuInstanceSlot>(m_World);
        m_DepthId = RegisterComponent<HierarchyDepth>(m_World);
        m_Serial = ++g_StoreSerial;

        m_LeafObserver = MAKE_RC_OBJ_PTR(LeafObserver, this,
            [](SceneGraphStore* store, Entity entity, SceneGraphLeaf* leaf, bool added)
            {
                store->OnLeaf(entity, leaf, added);
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
        ++m_SchedulerGeneration;
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
            { m_Ids[SlotLocalBounds], Access::Write, {} },   // read again from leaves with dynamic bounds
            { m_Ids[SlotLeaf], Access::Read, {} },
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

#if defined(_MSC_VER)
    __declspec(noinline)
#endif
    bool SceneGraphStore::LocateNodeSlow(const SceneGraphNode* node, RowRef& out) const noexcept
    {
        return Locate(node->m_Entity, out);
    }

    SceneGraphNode* SceneGraphStore::NodeOfSlow(Entity entity) const noexcept
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
        LeafType type{ &desc, uint32_t(m_LeafTypeTags.size()), 0, 0 };
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
        if (desc.dynamicBounds)
        {
            const ComponentId id = RegisterComponent<DynamicBoundsLeaf>(m_World);
            if (id != ComponentId::Invalid)
            {
                m_DynamicBoundsTag = id;
                m_LeafTypeTags.push_back(id);
                ++type.tagCount;
            }
            else
                log::warning("SceneGraph: cannot register the dynamic-bounds tag");
        }
        // The class's columns (D19), zero until something fills them (the mesh-instance columns: OnLeaf).
        for (uint32_t c = 0; c < desc.columnCount; ++c)
        {
            ComponentId id = ComponentId::Invalid;
            if (NVRHI_FAILED(m_World->RegisterComponent(*desc.columns[c], id)))
            {
                log::warning("SceneGraph: cannot register the leaf column %s", desc.columns[c]->name);
                continue;
            }
            m_LeafTypeTags.push_back(id);
            ++type.columnCount;
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

    nvrhi::FRESULT SceneGraphStore::RunReadOnlyPhase(Phase phase) noexcept
    {
        if (!m_Scheduler)
            return nvrhi::FE_INVALID_ARGS;
        // A system that declares a write (or runs alone) may change anything: the phase is then an ordinary one.
        const bool writers = ecs::PhaseDeclaresWrites(m_Scheduler, phase);
#ifndef NDEBUG
        if (writers)
            log::warning("SceneGraph::RunReadOnlyPhase: a system of phase %u declares a write or is exclusive; the next "
                         "Refresh runs its phases", uint32_t(phase));
#endif
        const uint64_t version = m_World->GetVersion();
        const uint64_t structuralVersion = m_World->GetStructuralVersion();
        ecs::SchedulerCounters before{};
        const bool counted = ecs::GetSchedulerCounters(m_Scheduler, before);
        const nvrhi::FRESULT result = m_Scheduler->RunPhase(phase);
        if (NVRHI_FAILED(result) || writers || !counted)
            return result;
        ecs::SchedulerCounters after{};
        ecs::GetSchedulerCounters(m_Scheduler, after);
        // Current before the phase (nothing ran on the world since the last Refresh), and the phase changed nothing:
        // no command played back, no structural change. Its versions are then not a change.
        if (m_RefreshedVersion == version && m_RefreshedStructuralVersion == structuralVersion &&
            m_World->GetStructuralVersion() == structuralVersion && after.playbacks == before.playbacks)
            m_RefreshedVersion = m_World->GetVersion();
        return result;
    }

    void SceneGraphStore::WarnRenderRead(const char* operation) const noexcept
    {
        // Open point 12 of the E5 sub-plan, beside the phase check of design 5.6: the render service's snapshot points
        // at this graph's leaves and assets until its EndFrame. A warning, not an error (log::error breaks into the
        // debugger); the write applies, so Debug and Release behave alike.
        if (m_RenderReads != 0 && m_RenderReadQuiet == 0)
            log::warning("SceneGraph::%s while a frame records (between RenderService::Extract and EndFrame): the "
                         "frame's snapshot may point at what this changes; the write applies", operation);
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
        m_World->AdvanceVersion();   // what this stamps is newer than any version read before it (GpuScene)
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
        m_World->AdvanceVersion();
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
    std::atomic<int32_t> SceneGraphStore::s_FailSubgraphCountdown{ -1 };
    std::atomic<uint64_t> SceneGraphStore::s_SubgraphRedos{ 0 };
    std::atomic<uint32_t> SceneGraphStore::s_SerialSubgraphRows{ 4096 };

    SceneContentFlags SceneGraphStore::SafeContentFlags(const SceneGraphLeaf* leaf) noexcept
    {
        try
        {
            return leaf->GetContentFlags();
        }
        catch (...)
        {
            log::warning("SceneGraph: a leaf's GetContentFlags threw; taken as None");
            return SceneContentFlags::None;
        }
    }

    dm::box3 SceneGraphStore::SafeLocalBoundingBox(SceneGraphLeaf* leaf, const dm::box3& fallback) noexcept
    {
        try
        {
            return leaf->GetLocalBoundingBox();
        }
        catch (...)
        {
            log::warning("SceneGraph: a leaf's GetLocalBoundingBox threw; its box stays as it was");
            return fallback;
        }
    }

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
        // NodeOf's table covers the new ids before any row exists (a growth that throws leaves nothing half made)
        uint32_t highest = 0;
        for (Entity entity : S.entities)
            highest = std::max(highest, GetEntityIndex(entity));
        if (highest >= m_NodeOfIndex.size())
            m_NodeOfIndex.resize(std::max<size_t>(size_t(highest) + 1, m_NodeOfIndex.size() * 2));

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
                for (uint32_t t = 0; t < type.tagCount + type.columnCount; ++t)
                    add(m_LeafTypeTags[type.firstTag + t], nullptr);   // columns: filled by the leaf observer
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
                m_NodeOfIndex[GetEntityIndex(S.entities[i])] = NodeOfEntry{ S.entities[i], nodes[i].node };
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
            const uint32_t index = GetEntityIndex(m_S->entities[i]);
            if (index < m_NodeOfIndex.size())
                m_NodeOfIndex[index] = NodeOfEntry{};
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
        m_World->AdvanceVersion();
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
        m_World->AdvanceVersion();
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
                for (uint32_t t = 0; t < type.tagCount + type.columnCount; ++t)
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
                for (uint32_t t = 0; t < type.tagCount + type.columnCount; ++t)
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
        if (Locate(entity, row))
            RememberRow(node, row);   // its row moved to another archetype
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
            m_World->AdvanceVersion();
            static const OrphanRecord s_Defaults;
            LocalTransform value = s_Defaults.local;
            if (scaling) value.scaling = *scaling;
            if (rotation) value.rotation = *rotation;
            if (translation) value.translation = *translation;
            m_World->AddComponent(node->m_Entity, m_Ids[SlotLocal], &value);
            m_World->AddComponent(node->m_Entity, m_Ids[SlotLocalMatrix], nullptr);
            SyncArchetypes();
            if (Locate(node->m_Entity, row))
                RememberRow(node, row);   // its row moved to another archetype
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
                m_World->AdvanceVersion();   // the stamps below are newer than what GpuScene read
                bounds->bounds = leaf->object->GetLocalBoundingBox();
                Stamp(row, SlotLocalBounds);
                Stamp(row, SlotLeaf);   // consumers that want to know see the leaf column changed (5.4)
                // a mesh instance's material domains may have changed with its content
                if (RenderFlags* flags = static_cast<RenderFlags*>(Column(row, SlotRenderFlags)))
                {
                    *flags = MakeRenderFlags(static_cast<MeshInstance*>(leaf->object));
                    Stamp(row, SlotRenderFlags);
                }
                m_PendingContent = true;   // the next Refresh runs Bounds for it
                BumpEpoch();
            }
        }
        // a node without a leaf has no content: nothing changed
    }

    // ------------------------------------------------------------------------- mesh instances (step E4)

    const GpuInstanceSlot* SceneGraphStore::InstanceSlotOf(const SceneGraphLeaf* leaf) noexcept
    {
        const SceneGraphNode* node = leaf->GetNode();
        if (!node || !node->m_Store)
            return nullptr;
        const GpuInstanceSlot* slot = node->m_Store->GetOf<GpuInstanceSlot>(node, SlotInstance);
        return slot && slot->instance != c_NoInstanceSlot ? slot : nullptr;
    }

    void SceneGraphStore::OnLeaf(Entity entity, SceneGraphLeaf* leaf, bool added)
    {
        // The asset tables and slots change first, and SceneGraph's callbacks run after them (they may throw), then
        // the virtual RegisterLeaf or UnregisterLeaf: as RegisterLeaf raised OnMeshAdded before it added the leaf to
        // its vectors.
        m_Changes.Clear();
        RowRef row;
        if (Locate(entity, row) && ColumnIndex(*row.archetype, row.archetypeIndex, SlotMeshRef) != c_Absent)
        {
            // Only MeshInstance classes declare these columns (MeshInstance::GetTypeDesc, inherited by subclasses).
            assert(query_cast<MeshInstance>(leaf));
            if (added)
                AcquireInstance(row, static_cast<MeshInstance*>(leaf), m_Changes);
            else
                ReleaseInstance(row, m_Changes);
        }
        RaiseAssetCallbacks(m_Changes);
        m_Changes.Clear();
        if (added)
            m_Graph->RegisterLeaf(leaf);
        else
            m_Graph->UnregisterLeaf(leaf);
    }

    std::atomic<int32_t> SceneGraphStore::s_FailAcquireCountdown{ -1 };

    void SceneGraphStore::AcquireInstance(const RowRef& row, MeshInstance* instance, MeshTable::Changes& changes)
    {
        // Until every slot is allocated, the row holds no slot (c_NoInstanceSlot): an exception from the table (out of
        // memory) leaves nothing for ReleaseInstance to free.
        // The same for the mesh reference: a row copied through the world (Instantiate) arrives with its source's
        // MeshId, a reference it never took.
        GpuInstanceSlot* slot = static_cast<GpuInstanceSlot*>(Column(row, SlotInstance));
        if (slot)
            *slot = GpuInstanceSlot{};
        MeshRef* ref = static_cast<MeshRef*>(Column(row, SlotMeshRef));
        ref->mesh = MeshId::Invalid;
        if (s_FailAcquireCountdown.load(std::memory_order_relaxed) >= 0 &&
            s_FailAcquireCountdown.fetch_sub(1, std::memory_order_relaxed) == 0)
            throw std::bad_alloc();   // test seam (FailAcquireAfter)

        SceneGraph& graph = *m_Graph;
        const MeshId mesh = graph.m_Meshes.Acquire(instance->GetMesh(), graph.m_Materials, changes);
        graph.m_MaxGeometryCountPerMesh = std::max(graph.m_MaxGeometryCountPerMesh, changes.geometriesAdded);
        const uint32_t geometryCount =
            mesh != MeshId::Invalid ? graph.m_Meshes.GetMeshRow(MeshTable::SlotOf(mesh)).geometryCount : 0;

        // Written without a stamp: the row was just created or moved here, so its chunk's added version is newer than
        // anything a reader saw.
        ref->mesh = mesh;
        if (RenderFlags* flags = static_cast<RenderFlags*>(Column(row, SlotRenderFlags)))
            *flags = MakeRenderFlags(instance);
        if (slot)
        {
            const uint32_t first = m_GeometryInstanceSlots.Allocate(geometryCount);
            slot->instance = m_InstanceSlots.Allocate(1);
            slot->firstGeometryInstance = first;
            slot->geometryCount = geometryCount;
        }
    }

    void SceneGraphStore::ReleaseInstance(const RowRef& row, MeshTable::Changes& changes)
    {
        GpuInstanceSlot* slot = static_cast<GpuInstanceSlot*>(Column(row, SlotInstance));
        if (slot && slot->instance != c_NoInstanceSlot)
        {
            m_InstanceSlots.Free(slot->instance, 1);
            m_GeometryInstanceSlots.Free(slot->firstGeometryInstance, slot->geometryCount);
        }
        if (slot)
            *slot = GpuInstanceSlot{};   // the getters report -1 from here on (UnregisterLeaf overrides see that)
        MeshRef* ref = static_cast<MeshRef*>(Column(row, SlotMeshRef));
        const MeshId mesh = ref->mesh;
        ref->mesh = MeshId::Invalid;
        m_Graph->m_Meshes.Release(mesh, m_Graph->m_Materials, changes);
    }

    void SceneGraphStore::RaiseAssetCallbacks(const MeshTable::Changes& changes)
    {
        const SceneGraph& graph = *m_Graph;
        for (const MeshTable::Change& change : changes.events)
        {
            if (change.mesh)
            {
                const auto& callback = change.added ? graph.OnMeshAdded : graph.OnMeshRemoved;
                if (callback)
                    callback(change.mesh);
            }
            else if (change.material)
            {
                const auto& callback = change.added ? graph.OnMaterialAdded : graph.OnMaterialRemoved;
                if (callback)
                    callback(change.material);
            }
        }
    }

    void SceneGraphStore::ReconcileMeshes()
    {
        SceneGraph& graph = *m_Graph;
        MeshTable& meshes = graph.m_Meshes;
        m_Changes.Clear();
        std::vector<uint8_t> reranged;   // per mesh slot; empty while none was
        const uint32_t slotEnd = meshes.GetSlotEnd();
        for (uint32_t slot = 0; slot < slotEnd; ++slot)
        {
            if (meshes.GetRow(slot).object && meshes.Reconcile(slot, graph.m_Materials, m_Changes))
            {
                reranged.resize(slotEnd, 0);
                reranged[slot] = 1;
            }
        }
        // A material shared with another graph gets this graph's slot written back (Reconcile does it for meshes and
        // geometries): the last graph refreshed wins, as with the old renumbering. GpuScene does not read the objects'
        // slots; other consumers (MaterialConstants::materialID of shaders without bindless) do.
        graph.m_Materials.WriteSlots();
        if (!reranged.empty())
        {
            // The instances of a mesh with a new geometry count get geometry-instance ranges of that count; their
            // chunks are stamped, so that GpuScene writes them again (as it writes every row after a layout change).
            m_World->AdvanceVersion();
            ++m_LayoutSerial;
            SyncArchetypes();
            const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
                const uint8_t refColumn = ColumnIndex(archetype, a, SlotMeshRef);
                const uint8_t slotColumn = ColumnIndex(archetype, a, SlotInstance);
                if (refColumn == c_Absent || slotColumn == c_Absent)
                    continue;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    const MeshRef* refs = reinterpret_cast<const MeshRef*>(ColumnBase(archetype, c, refColumn));
                    GpuInstanceSlot* slots = reinterpret_cast<GpuInstanceSlot*>(ColumnBase(archetype, c, slotColumn));
                    bool wrote = false;
                    for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                    {
                        const uint32_t mesh = MeshTable::SlotOf(refs[r].mesh);
                        if (refs[r].mesh == MeshId::Invalid || mesh >= reranged.size() || !reranged[mesh] ||
                            slots[r].instance == c_NoInstanceSlot)
                            continue;
                        const uint32_t count = meshes.GetMeshRow(mesh).geometryCount;
                        m_GeometryInstanceSlots.Free(slots[r].firstGeometryInstance, slots[r].geometryCount);
                        slots[r].firstGeometryInstance = m_GeometryInstanceSlots.Allocate(count);
                        slots[r].geometryCount = count;
                        wrote = true;
                    }
                    if (wrote)
                        StampVersion(archetype.Versions(c) + slotColumn, m_Impl->GetStampVersion());
                }
            }
        }
        // Callbacks after the tables are consistent; an exception from one is reported, as the leaf observer does.
        try
        {
            RaiseAssetCallbacks(m_Changes);
        }
        catch (const std::exception& error)
        {
            log::warning("SceneGraph::Refresh: an asset callback threw '%s'", error.what());
        }
        catch (...)
        {
            log::warning("SceneGraph::Refresh: an asset callback threw");
        }
        m_Changes.Clear();
    }

    // ------------------------------------------------------------------------------------- refresh

    bool SceneGraphStore::Refresh(uint32_t frameIndex)
    {
        ++m_RefreshCalls;
        SyncArchetypes();
        const bool structure = m_PendingStructure;
        const bool moved = m_PendingTransform;
        // Nothing changed through the API (structure, transforms, content), no structural change and no phase ran on
        // the world since: nothing to do. Writes into the world outside phases are not seen here (SceneGraph.h).
        const bool unchanged = !structure && !moved && !m_PendingContent &&
                               m_World->GetStructuralVersion() == m_RefreshedStructuralVersion &&
                               m_World->GetVersion() == m_RefreshedVersion;
        // Leaves with dynamic bounds change their boxes by themselves: then only the Bounds phase runs (and the
        // incremental subgraph pass for what it changed).
        const bool dynamicOnly = unchanged && HasDynamicBoundsRows();
        if (unchanged && !dynamicOnly)
        {
            m_LastRefreshMoved = false;
            ++m_Counters.skippedRefreshes;
            return false;
        }

        m_FrameIndex = frameIndex;
        PrepareSubgraphs();
        const nvrhi::FRESULT result = !m_Scheduler ? nvrhi::FE_INVALID_ARGS
            : dynamicOnly ? m_Scheduler->RunPhases(Phase::Bounds, Phase::Bounds)
                          : m_Scheduler->RunPhases(Phase::TransformPropagation, Phase::Bounds);
        if (NVRHI_FAILED(result))
        {
            m_SubgraphPrepared = false;
            m_SubgraphForceFull = true;
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
        // The subgraph values of the nodes with children, computed now, in parallel (5.7): the getters read them
        // without a lock. The epoch only moves with structural changes, SetLeaf and InvalidateContent; when the pass
        // cannot run (out of memory), it moves here, and the getters fall back to the on-demand caches.
        RefreshRowHints();
        const uint64_t passes = m_Counters.subgraphPasses;
        UpdateSubgraphs();
        if (m_Counters.subgraphPasses == passes)
        {
            // the pass did not complete: the values it had are stale now (the rows moved), and what it wrote this time
            // is stamped with the epoch that moves below; the next pass is a full one
            m_SubgraphForceFull = true;
            m_SubgraphEpoch.store(~0ull, std::memory_order_release);
        }
        if (!IsSubgraphCurrent())
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
                const bool dynamicBounds =
                    m_DynamicBoundsTag != ComponentId::Invalid && column(m_DynamicBoundsTag) != c_Absent;
                if (localBounds != c_Absent && worldBounds != c_Absent && chunkBounds != c_Absent)
                    m_BoundsArchetypes.push_back(BoundsArchetype{ a, world, localBounds, worldBounds, chunkBounds,
                                                                  dynamicBounds ? column(m_Ids[SlotLeaf]) : c_Absent });
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
            LocalBounds* locals =
                reinterpret_cast<LocalBounds*>(ColumnBase(archetype, task.chunk, bounds.localBounds));
            uint64_t* versions = archetype.Versions(task.chunk);

            // Leaves with dynamic bounds: their boxes again. A chunk queued only for them (task.whole false) is written
            // only when one of its boxes changed.
            if (bounds.leaf != c_Absent)
            {
                const LeafObject* leaves =
                    reinterpret_cast<const LeafObject*>(ColumnBase(archetype, task.chunk, bounds.leaf));
                bool changed = false;
                for (uint32_t r = 0; r < count; ++r)
                {
                    if (!leaves[r].object)
                        continue;
                    const dm::box3 box = SafeLocalBoundingBox(leaves[r].object, locals[r].bounds);
                    if (std::memcmp(&box, &locals[r].bounds, sizeof(box)) != 0)
                    {
                        locals[r].bounds = box;
                        changed = true;
                    }
                }
                if (changed)
                    StampVersion(versions + bounds.localBounds, ctx.version);
                else if (!task.whole)
                    continue;
            }

            WorldBounds* results =
                reinterpret_cast<WorldBounds*>(ColumnBase(archetype, task.chunk, bounds.worldBounds));
            dm::box3 chunkBounds = dm::box3::empty();
            bool nan = false;
            for (uint32_t r = 0; r < count; ++r)
            {
                const dm::box3& local = locals[r].bounds;
                const dm::box3 world = local.isempty() ? dm::box3::empty() : local * dm::affine3(worlds[r].transform);
                results[r].bounds = world;
                chunkBounds |= world;
                // a NaN would fall out of the union (dm::min keeps the other operand): such a chunk is never culled
                nan = nan || HasNan(world);
            }
            if (nan)
                chunkBounds = InfiniteBox();
            reinterpret_cast<ChunkWorldBounds*>(ColumnBase(archetype, task.chunk, bounds.chunkBounds))->bounds =
                chunkBounds;
            StampVersion(versions + bounds.worldBounds, ctx.version);
            StampVersion(versions + bounds.chunkBounds, ctx.version);
            if (ctx.subgraph)
                ctx.store->SubgraphFromBounds(archetype, bounds.archetype, task.chunk, chunkBounds);
        }
        ctx.chunks.fetch_add(end - begin, std::memory_order_relaxed);
    }

    nvrhi::FRESULT SceneGraphStore::RunBounds(ISystemContext* context) noexcept
    {
        const uint64_t since = context->GetLastRunVersion();
        IJobSystem* jobs = context->GetJobSystem();
        UpdateSystemLists();
        if (!m_SubgraphPrepared)
        {
            // run by someone else than Refresh: no marks for what it changes, so the pass's values are stale until a
            // full pass, and the getters fall back to the on-demand caches (a new epoch) until then
            m_SubgraphForceFull = true;
            BumpEpoch();
        }
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
                const bool changed = versions[bounds.world] > since || versions[bounds.localBounds] > since ||
                                     versions[archetype.AddedVersionSlot()] > since;
                // a chunk of leaves with dynamic bounds is read at every run
                if (changed || bounds.leaf != c_Absent)
                    m_Tasks.push_back(Task{ i, c, changed });
            }
        }
        if (m_Tasks.empty())
            return nvrhi::FS_OK;

        BoundsContext ctx;
        ctx.store = this;
        ctx.version = context->GetVersion();
        ctx.subgraph = m_SubgraphPrepared && !m_SubgraphFull;
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
            // An attached node: from its row, as Refresh's pass reads it (LocalBounds, not the leaf's box now), so that
            // the two paths give the same values.
            RowRef row;
            if (current->m_Store && !current->m_Record && current->m_Store->LocateNode(current, row))
            {
                frame.bounds = NanToInfinite(current->m_Store->OwnBoundsFromLocal(row));
                frame.leafContent = current->m_Store->OwnContent(row);
            }
            else if (SceneGraphLeaf* leaf = current->GetLeaf())
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

    // ------------------------------------------------------------------- subgraph values in Refresh (5.7)
    //
    // Refresh computes the subgraph bounds and content flags of the nodes with children whose subtree changed, into the
    // nodes' caches (derived values, design 4.6), so that the getters (and so the legacy strategies' walks) read them
    // without a lock. A node's own values (its box, content flags, links, leaf) stay in its row only (design R3).
    //   Full mode (the epoch moved since the last pass: a structural change, SetLeaf or InvalidateContent; the world's
    //   structural version moved; or the Bounds system ran outside a Refresh): step 1 runs over every chunk: every
    //   chunk gets its summary and every node with children a mark.
    //   Incremental mode: the Bounds system does step 1 for the chunks it wrote (their boxes changed): the chunks'
    //   summaries, and marks on the paths from the rows with children, and from the other rows' parents, up to the
    //   roots. Nothing marked: nothing more to do.
    //   Step 2, bottom-up: the deep bucket's subtrees (a depth-first walk from each entry, whose parent is shallower),
    //   then the depth buckets from the deepest: each row whose parent is marked adds its subgraph value (its own part,
    //   plus its accumulator when it is marked, or its node's values when it has children and is not marked) to its
    //   parent's accumulator; a marked row stores its value in its node. A chunk whose rows share one parent and have
    //   no children adds its summary (the row-order union of its rows' own parts) instead of its rows.
    // Marks hold the pass's number per entity index; accumulators are merged into by CAS, after a task merged a run of
    // one parent locally. Unions across rows and chunks are taken on keys that order as the floats do (-0 below +0),
    // by min and max, so the result does not depend on the order of the rows or of the threads; it equals the old
    // sequential union except for the sign of a zero that ties with another zero, and NaN (which the old union kept
    // or dropped by position; a chunk with a NaN box has an infinite summary).

    namespace
    {
        inline uint32_t KeyOfFloat(float value) noexcept
        {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        }

        inline float FloatOfKey(uint32_t key) noexcept
        {
            const uint32_t bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

    }

    struct SceneGraphStore::SubgraphContext
    {
        SceneGraphStore* store;
        uint32_t pass;
        uint64_t epoch;
        uint64_t version;   // the world's version now: summaries computed in this pass
        bool full;
        const Task* tasks;
        std::atomic<bool> failed{ false };
        std::atomic<uint64_t> nodes{ 0 };
        std::atomic<uint64_t> summarised{ 0 };
    };

    bool SceneGraphStore::OwnPart(const RowRef& row, dm::box3& outBounds, SceneContentFlags& outContent) const noexcept
    {
        outBounds = OwnBounds(row);
        outContent = OwnContent(row);
        return HasChildren(row);
    }

    namespace
    {
        void KeysOf(const dm::box3& nanBox, SceneContentFlags content, uint32_t* keys, uint32_t& outContent) noexcept
        {
            const dm::box3 box = NanToInfinite(nanBox);
            keys[0] = KeyOfFloat(box.m_mins.x);
            keys[1] = KeyOfFloat(box.m_mins.y);
            keys[2] = KeyOfFloat(box.m_mins.z);
            keys[3] = KeyOfFloat(box.m_maxs.x);
            keys[4] = KeyOfFloat(box.m_maxs.y);
            keys[5] = KeyOfFloat(box.m_maxs.z);
            outContent = uint32_t(content);
        }

        void EmptyKeys(uint32_t* keys, uint32_t& content) noexcept
        {
            KeysOf(dm::box3::empty(), SceneContentFlags::None, keys, content);
        }

        void UniteKeys(uint32_t* keys, uint32_t& content, const uint32_t* other, uint32_t otherContent) noexcept
        {
            for (int i = 0; i < 3; ++i)
                keys[i] = std::min(keys[i], other[i]);
            for (int i = 3; i < 6; ++i)
                keys[i] = std::max(keys[i], other[i]);
            content |= otherContent;
        }

        dm::box3 BoxOfKeys(const uint32_t* keys) noexcept
        {
            return dm::box3(dm::float3(FloatOfKey(keys[0]), FloatOfKey(keys[1]), FloatOfKey(keys[2])),
                            dm::float3(FloatOfKey(keys[3]), FloatOfKey(keys[4]), FloatOfKey(keys[5])));
        }
    }

    // Marks `entity` and its ancestors up to the first one this pass marked already; each new mark gets an empty
    // accumulator. Threads may mark at once: the first CAS on a mark wins it.
    void SceneGraphStore::MarkPath(Entity entity, uint32_t pass) noexcept
    {
        while (entity != Entity::Null)
        {
            const uint32_t index = GetEntityIndex(entity);
            if (index >= m_SubgraphCapacity)
            {
                m_SubgraphNeedsFull.store(true, std::memory_order_relaxed);   // made during the phases
                return;
            }
            std::atomic<uint32_t>& mark = m_SubgraphMarks[index];
            uint32_t old = mark.load(std::memory_order_relaxed);
            for (;;)
            {
                if (old == pass)
                    return;   // marked already, and so are its ancestors (or will be by the thread that marked it)
                if (mark.compare_exchange_weak(old, pass, std::memory_order_relaxed))
                    break;
            }
            if (!m_SubgraphMarked.load(std::memory_order_relaxed))
                m_SubgraphMarked.store(1, std::memory_order_relaxed);
            SubgraphAccumulator& accumulator = m_SubgraphAccumulators[index];
            uint32_t keys[6];
            uint32_t content;
            EmptyKeys(keys, content);
            for (int i = 0; i < 6; ++i)
                accumulator.keys[i].store(keys[i], std::memory_order_relaxed);
            accumulator.content.store(content, std::memory_order_relaxed);

            RowRef row;
            if (!Locate(entity, row))
                return;
            const Parent* parent = static_cast<const Parent*>(Column(row, SlotParent));
            entity = parent ? parent->entity : Entity::Null;
        }
    }

    // The subgraph values of a node with children into its cache (its own values stay in its row).
    void SceneGraphStore::FinalizeSubgraph(SceneGraphNode* node, const SubgraphKeys& keys, uint64_t epoch) const noexcept
    {
        if (!node)
            return;
        node->m_CachedBounds = BoxOfKeys(keys.keys);
        node->m_CachedSubgraphContent = SceneContentFlags(keys.content);
        node->m_CacheEpoch.store(epoch, std::memory_order_release);
    }

    namespace
    {
        // Merges keys into an accumulator by CAS.
        template <typename Accumulator>
        void MergeInto(Accumulator& accumulator, const uint32_t* keys, uint32_t content) noexcept
        {
            for (int i = 0; i < 6; ++i)
            {
                std::atomic<uint32_t>& target = accumulator.keys[i];
                uint32_t current = target.load(std::memory_order_relaxed);
                if (i < 3)
                {
                    while (keys[i] < current &&
                           !target.compare_exchange_weak(current, keys[i], std::memory_order_relaxed)) { }
                }
                else
                {
                    while (keys[i] > current &&
                           !target.compare_exchange_weak(current, keys[i], std::memory_order_relaxed)) { }
                }
            }
            if (content)
                accumulator.content.fetch_or(content, std::memory_order_relaxed);
        }
    }

    // A run of merges into one parent's accumulator, kept by a task and flushed when the parent changes and at its end,
    // so that the chunks and rows of one parent (all of a flat scene's) cost one CAS merge per task.
    struct SceneGraphStore::SubgraphRun
    {
        Entity parent = Entity::Null;
        uint32_t keys[6];
        uint32_t content = 0;
        uint64_t summarised = 0;   // chunks taken whole, counted per task

        SubgraphRun() noexcept { EmptyKeys(keys, content); }

        void Add(SceneGraphStore& store, Entity target, const uint32_t* otherKeys, uint32_t otherContent) noexcept
        {
            if (target != parent)
            {
                Flush(store);
                parent = target;
            }
            UniteKeys(keys, content, otherKeys, otherContent);
        }

        void Flush(SceneGraphStore& store) noexcept
        {
            if (parent != Entity::Null)
                MergeInto(store.m_SubgraphAccumulators[GetEntityIndex(parent)], keys, content);
            parent = Entity::Null;
            EmptyKeys(keys, content);
        }
    };

    // Step 1 of a full pass, over every chunk: the chunk's summary and the marks (every node with children). The node
    // objects get nothing here: their own values stay in their rows (design R3).
    void SceneGraphStore::RunSubgraphMarks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        SubgraphContext& ctx = *static_cast<SubgraphContext*>(context);
        SceneGraphStore& store = *ctx.store;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = ctx.tasks[t];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(task.archetype);
            const uint32_t count = archetype.chunks[task.chunk].count;
            const uint8_t linksColumn = store.ColumnIndex(archetype, task.archetype, SlotLinks);
            if (linksColumn == c_Absent || count == 0)
                continue;
            auto columnOf = [&](Slot slot) noexcept -> const void*
            {
                const uint8_t column = store.ColumnIndex(archetype, task.archetype, slot);
                return column != c_Absent ? ColumnBase(archetype, task.chunk, column) : nullptr;
            };
            const SceneLinks* links = static_cast<const SceneLinks*>(columnOf(SlotLinks));
            const Parent* parents = static_cast<const Parent*>(columnOf(SlotParent));
            const LeafObject* leaves = static_cast<const LeafObject*>(columnOf(SlotLeaf));
            const WorldBounds* worldBounds = static_cast<const WorldBounds*>(columnOf(SlotWorldBounds));
            const RenderFlags* renderFlags = static_cast<const RenderFlags*>(columnOf(SlotRenderFlags));
            const Entity* entities = archetype.Entities(task.chunk);

            if (task.archetype >= store.m_ChunkSummaries.size() ||
                task.chunk >= store.m_ChunkSummaries[task.archetype].size())
            {
                ctx.failed.store(true, std::memory_order_relaxed);   // EnsureSubgraphStorage covers every chunk
                continue;
            }
            ChunkSummary& summary = store.m_ChunkSummaries[task.archetype][task.chunk];
            summary.single = true;
            summary.parent = parents ? parents[0].entity : Entity::Null;
            dm::box3 summaryBounds = dm::box3::empty();
            uint32_t summaryContent = 0;

            Entity markedParent = Entity::Null;
            for (uint32_t r = 0; r < count; ++r)
            {
                const bool children = links[r].firstChild != Entity::Null;
                const Entity parent = parents ? parents[r].entity : Entity::Null;
                SceneGraphLeaf* leaf = leaves ? leaves[r].object : nullptr;
                const dm::box3 bounds = leaf && worldBounds ? worldBounds[r].bounds : dm::box3::empty();
                const SceneContentFlags own = renderFlags ? SceneContentFlags(renderFlags[r].contentFlags)
                                              : leaf ? SafeContentFlags(leaf) : SceneContentFlags::None;
                if (children || parent != summary.parent)
                    summary.single = false;
                summaryBounds |= NanToInfinite(bounds);
                summaryContent |= uint32_t(own);

                // the paths to mark: a row with children from itself, else from its parent (once per run of rows)
                if (children)
                    store.MarkPath(entities[r], ctx.pass);
                else if (!ctx.full && parent != markedParent)
                {
                    store.MarkPath(parent, ctx.pass);
                    markedParent = parent;
                }
            }
            uint32_t ignored;
            KeysOf(summaryBounds, SceneContentFlags::None, summary.keys.keys, ignored);
            summary.keys.content = summaryContent;
            summary.version = ctx.version;
        }
    }

    // Step 2, one chunk of a depth bucket (not the deep one): its rows' values into their parents' accumulators, its
    // marked rows' values into their nodes.
    void SceneGraphStore::SubgraphChunk(SubgraphContext& ctx, uint32_t archetypeIndex, uint32_t chunk,
                                        SubgraphRun& run) noexcept
    {
        const uint32_t pass = ctx.pass;
        auto marked = [&](Entity entity) noexcept
        {
            const uint32_t index = GetEntityIndex(entity);
            return entity != Entity::Null && index < m_SubgraphCapacity &&
                   m_SubgraphMarks[index].load(std::memory_order_relaxed) == pass;
        };
        // A chunk taken whole from its summary touches nothing but the summary.
        if (archetypeIndex >= m_ChunkSummaries.size() || chunk >= m_ChunkSummaries[archetypeIndex].size())
        {
            ctx.failed.store(true, std::memory_order_relaxed);   // EnsureSubgraphStorage covers every chunk
            return;
        }
        ChunkSummary& summary = m_ChunkSummaries[archetypeIndex][chunk];
        if (summary.version != 0 && summary.single)
        {
            if (marked(summary.parent))
            {
                run.Add(*this, summary.parent, summary.keys.keys, summary.keys.content);
                ++run.summarised;
            }
            return;
        }

        const Archetype& archetype = m_Impl->GetArchetypeStorage(archetypeIndex);
        const uint32_t count = archetype.chunks[chunk].count;
        if (count == 0)
            return;
        const uint8_t parentColumn = ColumnIndex(archetype, archetypeIndex, SlotParent);
        const uint8_t nodeColumn = ColumnIndex(archetype, archetypeIndex, SlotNode);
        const Parent* parents = parentColumn != c_Absent
            ? reinterpret_cast<const Parent*>(ColumnBase(archetype, chunk, parentColumn)) : nullptr;
        const SceneNode* nodes = nodeColumn != c_Absent
            ? reinterpret_cast<const SceneNode*>(ColumnBase(archetype, chunk, nodeColumn)) : nullptr;
        const Entity* entities = archetype.Entities(chunk);

        // The summary: one parent, no row with children. Step 1 of a full pass computes every chunk's; in an incremental
        // pass the Bounds system computed the summaries of the chunks whose boxes changed, and nothing else changes one
        // without a full pass (links, parents, leaves and content flags change with the epoch).
        if (summary.version == 0)
        {
            summary.single = true;
            summary.parent = parents ? parents[0].entity : Entity::Null;
            dm::box3 summaryBounds = dm::box3::empty();
            uint32_t summaryContent = 0;
            for (uint32_t r = 0; r < count; ++r)
            {
                const RowRef row{ &archetype, archetypeIndex, chunk, r };
                dm::box3 bounds;
                SceneContentFlags own;
                const bool children = OwnPart(row, bounds, own);
                if (children || (parents && parents[r].entity != summary.parent))
                    summary.single = false;
                summaryBounds |= NanToInfinite(bounds);
                summaryContent |= uint32_t(own);
            }
            uint32_t ignored;
            KeysOf(summaryBounds, SceneContentFlags::None, summary.keys.keys, ignored);
            summary.keys.content = summaryContent;
            summary.version = ctx.version;
        }
        if (summary.single)
        {
            if (marked(summary.parent))
            {
                run.Add(*this, summary.parent, summary.keys.keys, summary.keys.content);
                ++run.summarised;
            }
            return;
        }

        auto columnOf = [&](Slot slot) noexcept -> const void*
        {
            const uint8_t column = ColumnIndex(archetype, archetypeIndex, slot);
            return column != c_Absent ? ColumnBase(archetype, chunk, column) : nullptr;
        };
        const SceneLinks* links = static_cast<const SceneLinks*>(columnOf(SlotLinks));
        const LeafObject* leaves = static_cast<const LeafObject*>(columnOf(SlotLeaf));
        const WorldBounds* worldBounds = static_cast<const WorldBounds*>(columnOf(SlotWorldBounds));
        const RenderFlags* renderFlags = static_cast<const RenderFlags*>(columnOf(SlotRenderFlags));
        uint64_t computed = 0;
        for (uint32_t r = 0; r < count; ++r)
        {
            const Entity parent = parents ? parents[r].entity : Entity::Null;
            const bool parentMarked = marked(parent);
            const bool rowMarked = marked(entities[r]);
            if (!parentMarked && !rowMarked)
                continue;
            const SceneGraphLeaf* leaf = leaves ? leaves[r].object : nullptr;
            const dm::box3 bounds = leaf && worldBounds ? worldBounds[r].bounds : dm::box3::empty();
            const SceneContentFlags own = renderFlags ? SceneContentFlags(renderFlags[r].contentFlags)
                                          : leaf ? SafeContentFlags(leaf) : SceneContentFlags::None;
            const bool children = links && links[r].firstChild != Entity::Null;
            SubgraphKeys value;
            KeysOf(bounds, own, value.keys, value.content);
            SceneGraphNode* node = nodes ? nodes[r].object : nullptr;
            if (children)
            {
                if (rowMarked)
                {
                    const SubgraphAccumulator& accumulator = m_SubgraphAccumulators[GetEntityIndex(entities[r])];
                    uint32_t keys[6];
                    for (int i = 0; i < 6; ++i)
                        keys[i] = accumulator.keys[i].load(std::memory_order_relaxed);
                    UniteKeys(value.keys, value.content, keys, accumulator.content.load(std::memory_order_relaxed));
                    FinalizeSubgraph(node, value, ctx.epoch);
                    ++computed;
                }
                else if (node)
                {
                    // a subtree that did not change: the values the node keeps from an earlier pass
                    KeysOf(node->m_CachedBounds, node->m_CachedSubgraphContent, value.keys, value.content);
                }
            }
            if (parentMarked)
                run.Add(*this, parent, value.keys, value.content);
        }
        if (computed)
            ctx.nodes.fetch_add(computed, std::memory_order_relaxed);
    }

    void SceneGraphStore::RunSubgraphChunks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        SubgraphContext& ctx = *static_cast<SubgraphContext*>(context);
        SubgraphRun run;
        for (uint32_t t = begin; t < end; ++t)
            ctx.store->SubgraphChunk(ctx, ctx.tasks[t].archetype, ctx.tasks[t].chunk, run);
        run.Flush(*ctx.store);
        if (run.summarised)
            ctx.summarised.fetch_add(run.summarised, std::memory_order_relaxed);
    }

    // The deep bucket: the subtree of each entry (a row whose parent is shallower, or a root) depth first, children in
    // sibling order; every marked node with children in it gets its values, and the entry's value goes to its parent.
    void SceneGraphStore::RunSubgraphDeep(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept
    {
        SubgraphContext& ctx = *static_cast<SubgraphContext*>(context);
        SceneGraphStore& store = *ctx.store;
        std::vector<SubgraphFrame>& stack = store.m_SubgraphStacks[worker];
        const uint32_t pass = ctx.pass;
        auto marked = [&](Entity entity) noexcept
        {
            const uint32_t index = GetEntityIndex(entity);
            return entity != Entity::Null && index < store.m_SubgraphCapacity &&
                   store.m_SubgraphMarks[index].load(std::memory_order_relaxed) == pass;
        };
        // The value of `entity` when it needs no walk (no children, or not marked); false when it must be walked.
        auto leafValue = [&](Entity entity, SubgraphKeys& value, SceneContentFlags& own, Entity& firstChild,
                             Entity& nextSibling) noexcept
        {
            RowRef row;
            if (!store.Locate(entity, row))
            {
                EmptyKeys(value.keys, value.content);
                own = SceneContentFlags::None;
                firstChild = nextSibling = Entity::Null;
                return true;
            }
            dm::box3 bounds;
            const bool children = store.OwnPart(row, bounds, own);
            KeysOf(bounds, own, value.keys, value.content);
            const SceneLinks* links = static_cast<const SceneLinks*>(store.Column(row, SlotLinks));
            firstChild = links ? links->firstChild : Entity::Null;
            nextSibling = links ? links->nextSibling : Entity::Null;
            if (!children)
                return true;
            if (!marked(entity))
            {
                const SceneNode* node = static_cast<const SceneNode*>(store.Column(row, SlotNode));
                if (node && node->object)
                    KeysOf(node->object->m_CachedBounds, node->object->m_CachedSubgraphContent, value.keys,
                           value.content);
                return true;
            }
            return false;
        };

        uint64_t computed = 0;
        for (uint32_t e = begin; e < end; ++e)
        {
            const Entity entry = store.m_SubgraphEntries[e];
            SubgraphKeys value;
            SceneContentFlags own;
            Entity firstChild, nextSibling;
            if (!leafValue(entry, value, own, firstChild, nextSibling))
            {
                stack.clear();
                try
                {
                    stack.push_back(SubgraphFrame{ entry, firstChild, value, own });
                    while (!stack.empty())
                    {
                        SubgraphFrame& top = stack.back();
                        if (top.nextChild != Entity::Null)
                        {
                            const Entity child = top.nextChild;
                            SubgraphKeys childValue;
                            SceneContentFlags childOwn;
                            Entity childFirst, childNext;
                            const bool done = leafValue(child, childValue, childOwn, childFirst, childNext);
                            top.nextChild = childNext;
                            if (done)
                                UniteKeys(top.keys.keys, top.keys.content, childValue.keys, childValue.content);
                            else
                                stack.push_back(SubgraphFrame{ child, childFirst, childValue, childOwn });
                            continue;
                        }
                        const SubgraphFrame finished = top;
                        stack.pop_back();
                        store.FinalizeSubgraph(store.NodeOf(finished.entity), finished.keys, ctx.epoch);
                        ++computed;
                        if (stack.empty())
                            value = finished.keys;
                        else
                            UniteKeys(stack.back().keys.keys, stack.back().keys.content, finished.keys.keys,
                                      finished.keys.content);
                    }
                }
                catch (...)
                {
                    ctx.failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            // the entry's parent is shallower: its accumulator
            RowRef row;
            if (store.Locate(entry, row))
            {
                const Parent* parent = static_cast<const Parent*>(store.Column(row, SlotParent));
                if (parent && marked(parent->entity))
                    MergeInto(store.m_SubgraphAccumulators[GetEntityIndex(parent->entity)], value.keys, value.content);
            }
        }
        if (computed)
            ctx.nodes.fetch_add(computed, std::memory_order_relaxed);
    }

    bool SceneGraphStore::EnsureSubgraphStorage() noexcept
    {
        if (s_FailSubgraphCountdown.load(std::memory_order_relaxed) >= 0 &&
            s_FailSubgraphCountdown.fetch_sub(1, std::memory_order_relaxed) == 0)
            return false;   // test seam: as out of memory

        // Marks and accumulators by entity index.
        const size_t entities = m_Impl->GetEntityTableSize();
        if (entities > m_SubgraphCapacity)
        {
            const size_t capacity = std::max(entities + entities / 4 + 1024, m_SubgraphCapacity * 2);
            std::unique_ptr<std::atomic<uint32_t>[]> marks(new (std::nothrow) std::atomic<uint32_t>[capacity]);
            std::unique_ptr<SubgraphAccumulator[]> accumulators(new (std::nothrow) SubgraphAccumulator[capacity]);
            if (!marks || !accumulators)
                return false;
            for (size_t i = 0; i < capacity; ++i)
                marks[i].store(0, std::memory_order_relaxed);
            m_SubgraphMarks = std::move(marks);
            m_SubgraphAccumulators = std::move(accumulators);
            m_SubgraphCapacity = capacity;
            m_SubgraphPass = 0;
        }
        // A summary per chunk of every archetype; a stack per thread.
        try
        {
            const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
            if (m_ChunkSummaries.size() < archetypeCount)
                m_ChunkSummaries.resize(archetypeCount);
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const uint32_t chunks = m_Impl->GetArchetypeStorage(a).GetChunkCount();
                if (m_ChunkSummaries[a].size() < chunks)
                    m_ChunkSummaries[a].resize(chunks);
            }
            const size_t threads = m_Jobs ? m_Jobs->GetThreadCount() : 1;
            if (m_SubgraphStacks.size() < threads)
                m_SubgraphStacks.resize(threads);
        }
        catch (...)
        {
            return false;
        }
        return true;
    }

    // A new pass number (marks need no clearing), the mode, and the storage for the world before the phases. A phase
    // that changes the structure (commands played back before it, an exclusive system) is caught by UpdateSubgraphs.
    void SceneGraphStore::PrepareSubgraphs() noexcept
    {
        m_SubgraphPrepared = false;
        // Full when the epoch moved (a structural change through the API, SetLeaf, InvalidateContent) or the world's
        // structure did otherwise (the first transform of a node moves its row to another archetype): rows that joined
        // a chunk make its summary stale.
        m_PreparedStructuralVersion = m_World->GetStructuralVersion();
        m_SubgraphFull = m_SubgraphForceFull || m_SubgraphEpoch.load(std::memory_order_relaxed) != GetEpoch() ||
                         m_PreparedStructuralVersion != m_SubgraphStructuralVersion;
        m_SubgraphMarked.store(0, std::memory_order_relaxed);
        m_SubgraphNeedsFull.store(false, std::memory_order_relaxed);
        if (!EnsureSubgraphStorage())
            return;   // the getters fall back to the on-demand caches
        if (++m_SubgraphPass == 0)
        {
            for (size_t i = 0; i < m_SubgraphCapacity; ++i)
                m_SubgraphMarks[i].store(0, std::memory_order_relaxed);
            m_SubgraphPass = 1;
        }
        m_SubgraphPrepared = true;
    }

    bool SceneGraphStore::HasDynamicBoundsRows() const noexcept
    {
        for (const BoundsArchetype& bounds : m_BoundsArchetypes)
        {
            if (bounds.leaf == c_Absent)
                continue;
            const Archetype& archetype = m_Impl->GetArchetypeStorage(bounds.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                if (archetype.chunks[c].count != 0)
                    return true;
            }
        }
        return false;
    }

    // An incremental pass's step 1 for a chunk the Bounds system wrote (its boxes changed): the summary (its bounds the
    // chunk's union, ChunkWorldBounds, which the Bounds system just made in row order), and the paths to mark, from a
    // row with children itself, else from its parent (once per run of rows of one parent).
    void SceneGraphStore::SubgraphFromBounds(const Archetype& archetype, uint32_t archetypeIndex, uint32_t chunk,
                                             const dm::box3& chunkBounds) noexcept
    {
        const uint32_t count = archetype.chunks[chunk].count;
        const uint8_t linksColumn = ColumnIndex(archetype, archetypeIndex, SlotLinks);
        if (linksColumn == c_Absent || count == 0)
            return;
        const SceneLinks* links = reinterpret_cast<const SceneLinks*>(ColumnBase(archetype, chunk, linksColumn));
        const uint8_t parentColumn = ColumnIndex(archetype, archetypeIndex, SlotParent);
        const Parent* parents = parentColumn != c_Absent
            ? reinterpret_cast<const Parent*>(ColumnBase(archetype, chunk, parentColumn)) : nullptr;
        const Entity* entities = archetype.Entities(chunk);
        auto columnOf = [&](Slot slot) noexcept -> const void*
        {
            const uint8_t column = ColumnIndex(archetype, archetypeIndex, slot);
            return column != c_Absent ? ColumnBase(archetype, chunk, column) : nullptr;
        };
        const LeafObject* leaves = static_cast<const LeafObject*>(columnOf(SlotLeaf));
        const RenderFlags* renderFlags = static_cast<const RenderFlags*>(columnOf(SlotRenderFlags));
        if (archetypeIndex >= m_ChunkSummaries.size() || chunk >= m_ChunkSummaries[archetypeIndex].size())
        {
            m_SubgraphNeedsFull.store(true, std::memory_order_relaxed);   // an archetype or chunk made during the phases
            return;
        }
        ChunkSummary& summary = m_ChunkSummaries[archetypeIndex][chunk];
        summary.single = true;
        summary.parent = parents ? parents[0].entity : Entity::Null;
        uint32_t summaryContent = 0;
        Entity markedParent = Entity::Null;
        for (uint32_t r = 0; r < count; ++r)
        {
            const bool children = links[r].firstChild != Entity::Null;
            const Entity parent = parents ? parents[r].entity : Entity::Null;
            if (children || parent != summary.parent)
                summary.single = false;
            const SceneGraphLeaf* leaf = leaves ? leaves[r].object : nullptr;
            summaryContent |= renderFlags ? renderFlags[r].contentFlags
                              : leaf ? uint32_t(SafeContentFlags(leaf)) : 0u;
            if (children)
                MarkPath(entities[r], m_SubgraphPass);
            else if (parent != markedParent)
            {
                MarkPath(parent, m_SubgraphPass);
                markedParent = parent;
            }
        }
        uint32_t ignored;
        KeysOf(chunkBounds, SceneContentFlags::None, summary.keys.keys, ignored);
        summary.keys.content = summaryContent;
        summary.version = m_World->GetVersion();
    }

    void SceneGraphStore::UpdateSubgraphs() noexcept
    {
        IJobSystem* jobs = m_Jobs;
        const uint32_t threads = WorkingThreads(jobs);
        if (!m_SubgraphPrepared)
            return;   // out of memory before the phases: the getters fall back to the on-demand caches
        m_SubgraphPrepared = false;
        bool full = m_SubgraphFull;
        if (m_World->GetStructuralVersion() != m_PreparedStructuralVersion ||
            m_SubgraphNeedsFull.load(std::memory_order_relaxed))
        {
            // A phase changed the structure (commands played back before it, an exclusive system) or made rows the
            // storage did not cover: the Bounds system's marks and summaries may be incomplete. Again, in full, over
            // the world as it is now.
            SyncArchetypes();
            UpdateSystemLists();
            if (!EnsureSubgraphStorage())
                return;
            if (++m_SubgraphPass == 0)
            {
                for (size_t i = 0; i < m_SubgraphCapacity; ++i)
                    m_SubgraphMarks[i].store(0, std::memory_order_relaxed);
                m_SubgraphPass = 1;
            }
            full = true;
            s_SubgraphRedos.fetch_add(1);
        }
        // 5: a full pass moves the epoch, so that what the lazy path stamped before (on nodes without children too)
        // is not taken for current
        if (full)
            BumpEpoch();
        const uint64_t epoch = GetEpoch();

        SubgraphContext ctx;
        ctx.store = this;
        ctx.pass = m_SubgraphPass;
        ctx.epoch = epoch;
        ctx.version = m_World->GetVersion();
        ctx.full = full;

        try
        {
            // 1. Marks and summaries, in full mode, over every chunk (in incremental mode the Bounds system
            // did it for the chunks it wrote).
            if (full)
            {
                m_SubgraphTasks.clear();
                uint32_t rows = 0;
                for (uint32_t d = 0; d <= c_MaxHierarchyDepth; ++d)
                {
                    for (const LevelArchetype& level : m_Levels[d])
                    {
                        const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                        for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                        {
                            if (archetype.chunks[c].count == 0)
                                continue;
                            m_SubgraphTasks.push_back(Task{ level.archetype, c, true });
                            rows += archetype.chunks[c].count;
                        }
                    }
                }
                ctx.tasks = m_SubgraphTasks.data();
                const uint32_t markTasks = uint32_t(m_SubgraphTasks.size());
                if (jobs && rows >= s_SerialSubgraphRows.load(std::memory_order_relaxed) && markTasks > 1)
                    jobs->ParallelFor(markTasks, std::max(1u, markTasks / (threads * 4)), &RunSubgraphMarks, &ctx);
                else if (markTasks)
                    RunSubgraphMarks(&ctx, 0, markTasks, 0);
            }
            if (m_SubgraphMarked.load(std::memory_order_relaxed) == 0)
            {
                // nothing marked: every node keeps its values
                m_SubgraphForceFull = false;
                m_SubgraphStructuralVersion = m_World->GetStructuralVersion();   // the structure this pass saw
                m_SubgraphEpoch.store(epoch, std::memory_order_release);
                ++m_Counters.subgraphPasses;
                m_Counters.fullSubgraphPasses += full ? 1 : 0;
                return;
            }

            // 2a. The deep bucket's entries: rows whose parent is in a shallower bucket (or roots), when they or their
            // parents are marked.
            m_SubgraphEntries.clear();
            for (const LevelArchetype& level : m_Levels[c_MaxHierarchyDepth])
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    const uint32_t count = archetype.chunks[c].count;
                    const Entity* rowEntities = archetype.Entities(c);
                    const Parent* parents = level.parent != c_Absent
                        ? reinterpret_cast<const Parent*>(ColumnBase(archetype, c, level.parent)) : nullptr;
                    for (uint32_t r = 0; r < count; ++r)
                    {
                        const Entity parent = parents ? parents[r].entity : Entity::Null;
                        auto isMarked = [&](Entity entity)
                        {
                            const uint32_t index = GetEntityIndex(entity);
                            return entity != Entity::Null && index < m_SubgraphCapacity &&
                                   m_SubgraphMarks[index].load(std::memory_order_relaxed) == ctx.pass;
                        };
                        if (!isMarked(parent) && !isMarked(rowEntities[r]))
                            continue;
                        bool entry = parent == Entity::Null;
                        if (!entry)
                        {
                            const EntityRecord* record = m_Impl->FindAlive(parent);
                            entry = record && record->archetype < m_ArchetypeLevel.size() &&
                                    m_ArchetypeLevel[record->archetype] != c_MaxHierarchyDepth;
                        }
                        if (entry)
                            m_SubgraphEntries.push_back(rowEntities[r]);
                    }
                }
            }
        }
        catch (...)
        {
            return;   // out of memory: the getters fall back to the on-demand caches
        }
        const uint32_t entryCount = uint32_t(m_SubgraphEntries.size());
        if (jobs && entryCount > 1)
            jobs->ParallelFor(entryCount, 1, &RunSubgraphDeep, &ctx);
        else if (entryCount)
            RunSubgraphDeep(&ctx, 0, entryCount, 0);

        // 2b. The depth buckets from the deepest shallow one up to the roots.
        for (int32_t d = int32_t(c_MaxHierarchyDepth) - 1; d >= 0 && !ctx.failed.load(); --d)
        {
            m_SubgraphTasks.clear();
            uint32_t rows = 0;
            try
            {
                for (const LevelArchetype& level : m_Levels[d])
                {
                    const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                    for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                    {
                        if (archetype.chunks[c].count == 0)
                            continue;
                        m_SubgraphTasks.push_back(Task{ level.archetype, c, true });
                        rows += archetype.chunks[c].count;
                    }
                }
            }
            catch (...)
            {
                return;
            }
            const uint32_t taskCount = uint32_t(m_SubgraphTasks.size());
            ctx.tasks = m_SubgraphTasks.data();
            if (jobs && rows >= s_SerialSubgraphRows.load(std::memory_order_relaxed) && taskCount > 1)
                jobs->ParallelFor(taskCount, std::max(1u, taskCount / (threads * 4)), &RunSubgraphChunks, &ctx);
            else if (taskCount)
                RunSubgraphChunks(&ctx, 0, taskCount, 0);
        }
        if (ctx.failed.load())
            return;

        m_SubgraphForceFull = false;
        m_SubgraphStructuralVersion = m_World->GetStructuralVersion();   // the structure this pass saw
        m_SubgraphEpoch.store(epoch, std::memory_order_release);
        ++m_Counters.subgraphPasses;
        m_Counters.fullSubgraphPasses += full ? 1 : 0;
        m_Counters.subgraphNodes += ctx.nodes.load(std::memory_order_relaxed);
        m_Counters.summarisedChunks += ctx.summarised.load(std::memory_order_relaxed);
    }

    // ---------------------------------------------------------------------------- row hints (5.2)

    void SceneGraphStore::RunRowHints(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        SceneGraphStore& store = *static_cast<SceneGraphStore*>(context);
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_SubgraphTasks[t];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(task.archetype);
            const uint8_t nodeColumn = store.ColumnIndex(archetype, task.archetype, SlotNode);
            const SceneNode* nodes = reinterpret_cast<const SceneNode*>(ColumnBase(archetype, task.chunk, nodeColumn));
            const uint32_t count = archetype.chunks[task.chunk].count;
            for (uint32_t r = 0; r < count; ++r)
            {
                if (nodes[r].object)
                    RememberRow(nodes[r].object, RowRef{ &archetype, task.archetype, task.chunk, r });
            }
        }
    }

    // The hints of an archetype go stale for all its rows when one row moves (Archetype::rowMoves): after structural
    // changes, Refresh writes them again for the archetypes whose rows moved, so that the getters keep their fast path.
    void SceneGraphStore::RefreshRowHints() noexcept
    {
        const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
        uint32_t rows = 0;
        try
        {
            if (m_HintedMoves.size() < archetypeCount)
                m_HintedMoves.resize(archetypeCount, ~uint64_t(0));
            m_SubgraphTasks.clear();
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
                if (m_HintedMoves[a] == archetype.rowMoves || ColumnIndex(archetype, a, SlotNode) == c_Absent)
                    continue;
                m_HintedMoves[a] = archetype.rowMoves;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    if (archetype.chunks[c].count == 0)
                        continue;
                    m_SubgraphTasks.push_back(Task{ a, c, true });
                    rows += archetype.chunks[c].count;
                }
            }
        }
        catch (...)
        {
            return;   // the hints stay stale: the getters take the entity table
        }
        const uint32_t taskCount = uint32_t(m_SubgraphTasks.size());
        if (m_Jobs && rows >= s_SerialSubgraphRows.load(std::memory_order_relaxed) && taskCount > 1)
            m_Jobs->ParallelFor(taskCount, std::max(1u, taskCount / (WorkingThreads(m_Jobs) * 4)), &RunRowHints, this);
        else if (taskCount)
            RunRowHints(this, 0, taskCount, 0);
    }
}
