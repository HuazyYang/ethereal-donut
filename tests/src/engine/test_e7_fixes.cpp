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

// Regression tests of the E7 fix track (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md, stage F), engine side:
// one case per backlog item, named after it. Each fails without its fix (the two i5_r2 cases in Debug builds only:
// Release builds never refused).
//   i5_r1_deep_static_bounds        a Static row in the deep bucket gets its boxes with its moves
//   i5_r2_deep_static_set_transform SetTransform refuses (Debug) only rows that propagation leaves out
//   i5_r2_static_first_transform    the first transform of a Static node without one is written in Debug too
//   i5_r3_static_rows_join_late     rows joining a Static archetype after propagation get boxes of their worlds
//   i5_r4_compose_through_static    a pose composed below a Static row starts from its world (binding)
//   i3_r1_r2_read_only_list         SimTeleport and SimPosePrevious are on donut's read-only list
//   e72_r4_mesh_columns_of_other_classes  no mesh-instance columns for a leaf class that is no MeshInstance
//   e72_r4_mesh_tags_of_other_classes     no mesh-instance tags for a leaf class that is not of their class
//   i3_n4_missing_name_warned_once  an ordering against a missing name warns once while it stays missing
//   i4_r2_shared_null_views         views without masks or row versions share one process-wide null array
//   i4_r4_row_tracked_reregistered  RowTracked asked of a type registered without it: a warning, flags kept; refused
//                                   for a kind without a value per row (F3-N1)
//   i6_r5_gpu_scene_stats_checked   GpuScene::GetStats is an owner-checked call while a pipelined frame records
//   i3_r5_editor_precision          the inspector's number editors round-trip every digit (typed through the seam)
//   t8_1_deferred_reconcile_retried a deferred mesh reconciliation that fails at EndRenderRead runs at the next Refresh
//   t7_2_shared_job_workers         DONUT_JOB_WORKERS sets the shared job system's workers (ctest runs it at 2 and
//                                   passes the 2 it expects after the case's name)
//   t7_2_invalid_workers            a DONUT_JOB_WORKERS that is no count from 1 to 256 gives the default and a warning
//   i4_r5_row_version_bytes         WorldStats::rowVersionBytes counts the row-version blocks

#include <donut/app/WorldInspector.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include "WorldInspectorDetail.h"   // the inspector's test seam (src/app)
#include "SceneGraphStore.h"        // its test seams (src/engine)
#include "SharedJobSystem.h"        // GetSharedJobSystem, GetSharedJobWorkersOverride (src/engine)

#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
#ifndef NDEBUG
    constexpr uint32_t c_ValidationWarnings = 1;   // a scene graph's world validates in Debug builds
#else
    constexpr uint32_t c_ValidationWarnings = 0;
#endif
    std::atomic<uint32_t> g_Warnings{ 0 };   // log::warning calls; a case that provokes some takes them back
    uint32_t TakeWarnings() { return g_Warnings.exchange(0); }
    uint32_t g_ExpectedWorkers = 0;   // t7_2_shared_job_workers: the count given after the case's name, 0 if none

    // DONUT_JOB_WORKERS: read into value (false when unset), and set (nullptr unsets).
    bool GetWorkersVariable(std::string& value)
    {
#if defined(_MSC_VER)
        char* found = nullptr;
        size_t length = 0;
        if (_dupenv_s(&found, &length, "DONUT_JOB_WORKERS") != 0 || !found)
            return false;
        value = found;
        std::free(found);
#else
        const char* found = std::getenv("DONUT_JOB_WORKERS");
        if (!found)
            return false;
        value = found;
#endif
        return true;
    }
    void SetWorkersVariable(const char* value)
    {
#if defined(_MSC_VER)
        _putenv_s("DONUT_JOB_WORKERS", value ? value : "");
#else
        if (value)
            setenv("DONUT_JOB_WORKERS", value, 1);
        else
            unsetenv("DONUT_JOB_WORKERS");
#endif
    }

    bool SameBits(const void* a, const void* b, size_t size) { return std::memcmp(a, b, size) == 0; }
    template <typename T> bool SameValue(const T& a, const T& b) { return SameBits(&a, &b, sizeof(T)); }
    bool Near(const dm::daffine3& a, const dm::daffine3& b, double tolerance)
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                if (std::abs(a.m_linear[r][c] - b.m_linear[r][c]) > tolerance)
                    return false;
        for (int c = 0; c < 3; ++c)
            if (std::abs(a.m_translation[c] - b.m_translation[c]) > tolerance)
                return false;
        return true;
    }

    nvrhi::AutoPtr<IJobSystem> MakeJobs(uint32_t workers)
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        JobSystemDesc desc;
        desc.debugName = "test_e7_fixes";
        desc.workerThreadCount = workers;
        CHECK(NVRHI_SUCCEEDED(CreateJobSystem(desc, &jobs)));
        return jobs;
    }

    nvrhi::AutoPtr<MeshInfo> MakeMesh()
    {
        auto material = MAKE_RC_OBJ_PTR(Material);
        material->domain = MaterialDomain::Opaque;
        auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
        geometry->material = material;
        geometry->numIndices = 3;
        geometry->numVertices = 3;
        geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f, 2.f, 1.f));
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->type = MeshType::Triangles;
        mesh->geometries.push_back(geometry);
        mesh->objectSpaceBounds = geometry->objectSpaceBounds;
        return mesh;
    }

    // A graph with its own job system, made with SceneGraph::SetStaticTag as given (the switch is restored).
    struct Graph
    {
        nvrhi::AutoPtr<IJobSystem> jobs = MakeJobs(4);
        nvrhi::AutoPtr<SceneGraph> graph;
        IWorld* world = nullptr;
        ComponentId staticId = ComponentId::Invalid;
        uint64_t frame = 0;
        double time = 0.0;

        explicit Graph(bool staticTag)
        {
            const bool was = SceneGraph::GetStaticTag();
            SceneGraph::SetStaticTag(staticTag);
            graph = MAKE_RC_OBJ_PTR(SceneGraph);
            SceneGraph::SetStaticTag(was);
            graph->SetJobSystem(jobs);
            world = graph->GetWorld();
            staticId = RegisterComponent<Static>(world);
            CHECK(staticId != ComponentId::Invalid);
            graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        }

        SceneGraphNode* Add(SceneGraphNode* parent, const dm::double3& translation, MeshInfo* mesh, bool isStatic)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(translation);
            if (mesh)
                node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
            SceneGraphNode* attached = graph->Attach(parent, node);
            CHECK(attached);
            if (isStatic)
                CHECK(world->AddComponent(graph->GetEntity(attached), staticId, nullptr) == nvrhi::FS_OK);
            return attached;
        }

        // A frame as a sample runs it without a render service: Simulate, Refresh, EndFrame.
        void Frame()
        {
            const double delta = 1.0 / 60.0;
            time += delta;
            CHECK(graph->Simulate(FrameTime{ time, delta, frame++ }) == nvrhi::FS_OK);
            graph->Refresh(uint32_t(frame));
            CHECK(graph->GetScheduler()->RunPhase(Phase::EndFrame) == nvrhi::FS_OK);
        }

        template <typename T> const T* Get(Entity entity) { return GetComponent<T>(world, entity); }
        template <typename T> const T* Get(SceneGraphNode* node) { return Get<T>(graph->GetEntity(node)); }
    };

    // Two graphs built and run alike, the first made with the Static switch off, the second with it on.
    struct Pair
    {
        std::unique_ptr<Graph> g[2];
        Pair()
        {
            g[0] = std::make_unique<Graph>(false);
            g[1] = std::make_unique<Graph>(true);
        }
        void Frame()
        {
            g[0]->Frame();
            g[1]->Frame();
        }
    };

    // The results of a node in the two graphs: its world, its row's world box and its subgraph box.
    bool SameNode(Pair& pair, SceneGraphNode* a, SceneGraphNode* b)
    {
        return SameValue(a->GetLocalToWorldTransform(), b->GetLocalToWorldTransform()) &&
               SameValue(*pair.g[0]->Get<WorldBounds>(a), *pair.g[1]->Get<WorldBounds>(b)) &&
               SameValue(a->GetGlobalBoundingBox(), b->GetGlobalBoundingBox());
    }
}

// ------------------------------------------------------------------------------------------------- I5-R1

// A chain of 80 nodes under a moving node; the nodes in the deep bucket (depth c_MaxHierarchyDepth and below) carry
// Static, which the deep bucket ignores: their worlds follow the moves there, and so must their boxes (the Bounds entry
// of their archetype is not Static either). Against a graph with the switch off, over moves after the rows settled.
void test_i5_r1_deep_static_bounds()
{
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh();
    Pair pair;
    std::vector<SceneGraphNode*> chains[2];
    SceneGraphNode* movers[2] = {};
    for (int i = 0; i < 2; ++i)
    {
        Graph& g = *pair.g[i];
        movers[i] = g.Add(g.graph->GetRootNode(), dm::double3(0.0), nullptr, false);   // depth 1
        SceneGraphNode* parent = movers[i];
        for (uint32_t n = 0; n < 80; ++n)
        {
            const uint32_t depth = n + 2;
            parent = g.Add(parent, dm::double3(0.5, 0.0, 0.0), mesh, depth > c_MaxHierarchyDepth + 2);
            chains[i].push_back(parent);
        }
    }
    for (int f = 0; f < 3; ++f)
        pair.Frame();   // the deep Static archetype settles
    for (int f = 1; f <= 4; ++f)
    {
        for (int i = 0; i < 2; ++i)
            movers[i]->SetTranslation(dm::double3(10.0 * f, 2.0 * f, 0.0));
        pair.Frame();
        for (size_t n = 0; n < chains[0].size(); ++n)
            CHECK(SameNode(pair, chains[0][n], chains[1][n]));
        // the deepest row is where its parent's moves put it, and its box with it
        const WorldBounds* box = pair.g[1]->Get<WorldBounds>(chains[1].back());
        CHECK(box->bounds.m_mins.x > float(10.0 * f) + 30.f);
    }
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------- I5-R2

// SetTransform of a Static row is refused in Debug builds only where propagation leaves the row out: a settled row of
// a Static archetype above the deep bucket. A Static row in the deep bucket, and one whose row joined its archetype
// since the last propagation (its write is its initial value), are written and propagated in both builds, as in a
// graph with the switch off.
void test_i5_r2_deep_static_set_transform()
{
#ifndef NDEBUG
    const bool debug = true;
#else
    const bool debug = false;
#endif
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh();
    Pair pair;
    SceneGraphNode* deep[2] = {};
    SceneGraphNode* shallow[2] = {};
    for (int i = 0; i < 2; ++i)
    {
        Graph& g = *pair.g[i];
        SceneGraphNode* parent = g.graph->GetRootNode();
        for (uint32_t depth = 1; depth <= c_MaxHierarchyDepth + 3; ++depth)
            parent = g.Add(parent, dm::double3(0.0, 0.25, 0.0), mesh, depth == c_MaxHierarchyDepth + 3);
        deep[i] = parent;
        shallow[i] = g.Add(g.graph->GetRootNode(), dm::double3(3.0, 0.0, 0.0), mesh, true);
    }
    for (int f = 0; f < 3; ++f)
        pair.Frame();
    CHECK(TakeWarnings() == 0);

    // the deep Static row: written, propagated, no warning
    for (int i = 0; i < 2; ++i)
        deep[i]->SetTranslation(dm::double3(7.0, 0.25, -1.0));
    CHECK(TakeWarnings() == 0);
    CHECK(deep[1]->GetTranslation().x == 7.0);
    pair.Frame();
    CHECK(SameNode(pair, deep[0], deep[1]));
    CHECK(deep[1]->GetLocalToWorldTransform().m_translation.x == 7.0);

    // a Static node attached now and placed before its first propagation: written, propagated, no warning
    SceneGraphNode* late[2] = {};
    for (int i = 0; i < 2; ++i)
    {
        late[i] = pair.g[i]->Add(pair.g[i]->graph->GetRootNode(), dm::double3(1.0, 1.0, 1.0), mesh, true);
        late[i]->SetTranslation(dm::double3(-4.0, 1.0, 1.0));
    }
    CHECK(TakeWarnings() == 0);
    CHECK(late[1]->GetTranslation().x == -4.0);
    pair.Frame();
    CHECK(SameNode(pair, late[0], late[1]));
    CHECK(late[1]->GetLocalToWorldTransform().m_translation.x == -4.0);

    // a settled Static row above the deep bucket: refused in Debug (one warning), written in Release; not moved
    const dm::daffine3 before = shallow[1]->GetLocalToWorldTransform();
    shallow[1]->SetTranslation(dm::double3(50.0, 0.0, 0.0));
    CHECK(TakeWarnings() == (debug ? 1u : 0u));
    CHECK((shallow[1]->GetTranslation().x == 50.0) == !debug);
    pair.Frame();
    CHECK(SameValue(shallow[1]->GetLocalToWorldTransform(), before));
    CHECK(TakeWarnings() == 0);
}

// A settled Static node that never had a transform (no LocalTransform column): its first SetTransform adds the
// column, a structural change that moves its row into another Static archetype, which the next propagation takes as
// a row that joined. Release builds always did so; Debug builds refused it, and now write it too (review of F1, the
// I5-R2 class). The same in both builds, and as in a graph with the switch off.
void test_i5_r2_static_first_transform()
{
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh();
    Pair pair;
    SceneGraphNode* bare[2] = {};
    for (int i = 0; i < 2; ++i)
    {
        Graph& g = *pair.g[i];
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);   // no transform: no LocalTransform column
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
        bare[i] = g.graph->Attach(g.graph->GetRootNode(), node);
        CHECK(bare[i]);
        CHECK(!g.world->HasComponent(g.graph->GetEntity(bare[i]), FindComponent<LocalTransform>(g.world)));
        CHECK(g.world->AddComponent(g.graph->GetEntity(bare[i]), g.staticId, nullptr) == nvrhi::FS_OK);
    }
    for (int f = 0; f < 3; ++f)
        pair.Frame();   // settled
    for (int i = 0; i < 2; ++i)
        bare[i]->SetTranslation(dm::double3(3.0, -2.0, 1.0));
    CHECK(TakeWarnings() == 0);
    CHECK(bare[1]->GetTranslation().x == 3.0);
    pair.Frame();
    CHECK(SameNode(pair, bare[0], bare[1]));
    CHECK(bare[1]->GetLocalToWorldTransform().m_translation.x == 3.0);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------- I5-R3

// Runtime-only rows that join a settled Static archetype between the TransformPropagation and Bounds systems (an
// exclusive system of the Bounds phase, before the graph's): Bounds takes them in that frame from worlds that are not
// propagated yet, propagation takes them in the next, and Bounds must take them again then. Against a graph with the
// switch off.
void test_i5_r3_static_rows_join_late()
{
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh();
    Pair pair;
    struct Content
    {
        MeshId mesh = MeshId::Invalid;
        std::vector<ComponentId> ids;          // GetRuntimeMeshInstanceComponents, then Static
        std::vector<Entity> rows;
        std::vector<Entity> late;
        bool armed = false;
    };
    Content content[2];
    constexpr uint32_t c_Late = 40;
    auto spawn = [](IWorld* world, Content& c, uint32_t count, double x0, std::vector<Entity>& out)
    {
        std::vector<LocalTransform> locals(count);
        std::vector<MeshRef> refs(count, MeshRef{ c.mesh });
        for (uint32_t r = 0; r < count; ++r)
            locals[r] = LocalTransform{ dm::double3(x0 + 2.0 * r, 3.0, -1.0), dm::dquat::identity(), dm::double3(0.5) };
        std::vector<const void*> values(c.ids.size(), nullptr);
        for (size_t i = 0; i < c.ids.size(); ++i)
        {
            if (c.ids[i] == FindComponent<LocalTransform>(world))
                values[i] = locals.data();
            else if (c.ids[i] == FindComponent<MeshRef>(world))
                values[i] = refs.data();
        }
        EntityCreateDesc desc{};
        desc.components = c.ids.data();
        desc.values = values.data();
        desc.componentCount = uint32_t(c.ids.size());
        desc.entityCount = count;
        const size_t first = out.size();
        out.resize(first + count, Entity::Null);
        return world->CreateEntities(desc, out.data() + first);
    };
    for (int i = 0; i < 2; ++i)
    {
        Graph& g = *pair.g[i];
        Content& c = content[i];
        c.mesh = g.graph->AcquireRuntimeMesh(mesh);
        c.ids.resize(g.graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
        g.graph->GetRuntimeMeshInstanceComponents(c.ids.data(), uint32_t(c.ids.size()));
        c.ids.push_back(g.staticId);
        CHECK(spawn(g.world, c, 100, 0.0, c.rows) == nvrhi::FS_OK);

        static const char* const s_Before[] = { "SceneGraph.Bounds" };
        SystemDesc desc{};
        desc.name = "join late";
        desc.phase = Phase::Bounds;
        desc.runBefore = s_Before;
        desc.runBeforeCount = 1;
        desc.flags = SystemFlags::Exclusive;
        Content* self = &c;
        nvrhi::AutoPtr<ISystem> system;
        CHECK(CreateLambdaSystem(desc, [self, spawn, c_Late](ISystemContext* context) noexcept
        {
            if (!self->armed)
                return nvrhi::FS_OK;
            self->armed = false;
            return spawn(context->GetWorld(), *self, c_Late, 1000.0, self->late);
        }, &system) == nvrhi::FS_OK);
        SystemId id = SystemId::Invalid;
        CHECK(g.graph->GetScheduler()->AddSystem(system, id) == nvrhi::FS_OK);
    }
    for (int f = 0; f < 3; ++f)
        pair.Frame();   // the Static archetype settles

    content[0].armed = content[1].armed = true;
    pair.Frame();   // the rows join after propagation
    CHECK(content[0].late.size() == c_Late && content[1].late.size() == c_Late);
    for (int f = 0; f < 3; ++f)
    {
        pair.Frame();
        const ComponentId chunkBounds[2] = { FindComponent<ChunkWorldBounds>(pair.g[0]->world),
                                             FindComponent<ChunkWorldBounds>(pair.g[1]->world) };
        for (uint32_t r = 0; r < c_Late; ++r)
        {
            const Entity a = content[0].late[r];
            const Entity b = content[1].late[r];
            CHECK(SameValue(*pair.g[0]->Get<WorldTransform>(a), *pair.g[1]->Get<WorldTransform>(b)));
            CHECK(SameValue(*pair.g[0]->Get<WorldBounds>(a), *pair.g[1]->Get<WorldBounds>(b)));
            CHECK(SameBits(pair.g[0]->world->GetComponent(a, chunkBounds[0]),
                           pair.g[1]->world->GetComponent(b, chunkBounds[1]), sizeof(ChunkWorldBounds)));
            // at its place: x from 1000 on
            CHECK(pair.g[1]->Get<WorldBounds>(b)->bounds.m_mins.x > 990.f);
        }
    }
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------- I5-R4

// A node below a Static node whose parent moved after it settled: the Static node stays (it does not follow its
// parent), and so does its child, whose world propagation composes from the Static node's world. Binding the child to
// the simulation takes its pose from that world, not from the Static node's local transform over its parent's new
// world: the child does not jump.
void test_i5_r4_compose_through_static()
{
    Graph g(true);
    SceneGraphNode* mover = g.Add(g.graph->GetRootNode(), dm::double3(1.0, 0.0, 0.0), nullptr, false);
    SceneGraphNode* still = g.Add(mover, dm::double3(5.0, 0.0, 0.0), nullptr, true);
    SceneGraphNode* child = g.Add(still, dm::double3(0.0, 1.0, 0.0), nullptr, false);
    for (int f = 0; f < 3; ++f)
        g.Frame();
    mover->SetTranslation(dm::double3(10.0, 0.0, 0.0));
    g.Frame();
    CHECK(mover->GetLocalToWorldTransform().m_translation.x == 10.0);
    CHECK(still->GetLocalToWorldTransform().m_translation.x == 6.0);   // Static: where it was
    const dm::daffine3 shown = child->GetLocalToWorldTransform();
    CHECK(shown.m_translation.x == 6.0 && shown.m_translation.y == 1.0);

    CHECK(g.graph->SetSimulationDriven(child, true) == nvrhi::FS_OK);
    CHECK(Near(child->GetLocalToWorldTransform(), shown, 1e-12));
    const SimPose* pose = g.Get<SimPose>(child);
    CHECK(pose && std::abs(pose->translation.x - 6.0) < 1e-12);
    g.Frame();
    CHECK(Near(child->GetLocalToWorldTransform(), shown, 1e-12));
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------- E72-R4

namespace e7_fixes
{
    // A leaf class that is no MeshInstance but declares the mesh-instance columns MeshRef and GpuInstanceSlot (and
    // RenderFlags), which would put its rows in the render service's instance query, whose rows' leaves are cast to
    // MeshInstance.
    class MeshColumnsLeaf;
    NVRHI_CLASS_CLSID(MeshColumnsLeaf, "1f5a9c3e-7b24-4d86-9e10-3c2b8a4f6d71")
    class MeshColumnsLeaf final : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(MeshColumnsLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MeshColumnsLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(MeshColumnsLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        const LeafTypeDesc& GetTypeDesc() const override
        {
            static const ComponentTypeDesc* const s_Columns[] = { &ComponentTraits<MeshRef>::Desc(),
                                                                  &ComponentTraits<RenderFlags>::Desc(),
                                                                  &ComponentTraits<GpuInstanceSlot>::Desc() };
            static const LeafTypeDesc s_Desc{ nvrhi::uuid_of<MeshColumnsLeaf>(), nullptr, 0, s_Columns,
                                              uint32_t(std::size(s_Columns)) };
            return s_Desc;
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return MAKE_RC_OBJ_PTR(MeshColumnsLeaf); }
    };

    // A leaf class that is no (skinned) mesh instance but carries the tags MeshInstanceLeaf and
    // SkinnedMeshInstanceLeaf, whose rows the render service's skinned query casts to SkinnedMeshInstance.
    class MeshTagsLeaf;
    NVRHI_CLASS_CLSID(MeshTagsLeaf, "8c2e4a61-0d3b-4f97-b5a8-6e1f2c9d7b30")
    class MeshTagsLeaf final : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(MeshTagsLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MeshTagsLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(MeshTagsLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        const LeafTypeDesc& GetTypeDesc() const override
        {
            static const ComponentTypeDesc* const s_Tags[] = { &ComponentTraits<MeshInstanceLeaf>::Desc(),
                                                               &ComponentTraits<SkinnedMeshInstanceLeaf>::Desc() };
            static const LeafTypeDesc s_Desc{ nvrhi::uuid_of<MeshTagsLeaf>(), s_Tags, uint32_t(std::size(s_Tags)),
                                              nullptr, 0 };
            return s_Desc;
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return MAKE_RC_OBJ_PTR(MeshTagsLeaf); }
    };
}

// The graph refuses the mesh-instance columns (MeshRef, RenderFlags, GpuInstanceSlot) of a leaf class that is no
// MeshInstance (one warning per class and graph, when the class is first attached): its rows get none, so no instance
// query takes them and nothing casts their leaf to MeshInstance (InvalidateContent included). A MeshInstance's rows
// keep all three.
void test_e72_r4_mesh_columns_of_other_classes()
{
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh();
    Graph g(false);
    SceneGraphNode* root = g.graph->GetRootNode();
    auto first = MAKE_RC_OBJ_PTR(SceneGraphNode);
    first->SetLeaf(MAKE_RC_OBJ_PTR(e7_fixes::MeshColumnsLeaf));
    SceneGraphNode* a = g.graph->Attach(root, first);
    CHECK(TakeWarnings() == 1);
    auto second = MAKE_RC_OBJ_PTR(SceneGraphNode);
    second->SetLeaf(MAKE_RC_OBJ_PTR(e7_fixes::MeshColumnsLeaf));
    SceneGraphNode* b = g.graph->Attach(root, second);
    CHECK(TakeWarnings() == 0);   // the class is known: no second warning
    SceneGraphNode* m = g.Add(root, dm::double3(0.0), mesh, false);
    g.Frame();
    const ComponentId meshRef = FindComponent<MeshRef>(g.world);
    const ComponentId slot = FindComponent<GpuInstanceSlot>(g.world);
    const ComponentId flags = FindComponent<RenderFlags>(g.world);
    for (SceneGraphNode* node : { a, b })
    {
        const Entity entity = g.graph->GetEntity(node);
        CHECK(!g.world->HasComponent(entity, meshRef) && !g.world->HasComponent(entity, slot));
        CHECK(!g.world->HasComponent(entity, flags));
    }
    a->InvalidateContent();   // reads no mesh-instance column of the row
    g.Frame();
    const Entity instance = g.graph->GetEntity(m);
    CHECK(g.world->HasComponent(instance, meshRef) && g.world->HasComponent(instance, slot) &&
          g.world->HasComponent(instance, flags));
    CHECK(TakeWarnings() == 0);
}

// The tags MeshInstanceLeaf and SkinnedMeshInstanceLeaf are refused to a class that is not of their class (one
// warning per class and graph): the skinned query of the render service never takes such a row.
void test_e72_r4_mesh_tags_of_other_classes()
{
    Graph g(false);
    SceneGraphNode* root = g.graph->GetRootNode();
    SceneGraphNode* nodes[2] = {};
    for (SceneGraphNode*& node : nodes)
    {
        auto made = MAKE_RC_OBJ_PTR(SceneGraphNode);
        made->SetLeaf(MAKE_RC_OBJ_PTR(e7_fixes::MeshTagsLeaf));
        node = g.graph->Attach(root, made);
    }
    CHECK(TakeWarnings() == 1);
    g.Frame();
    const ComponentId meshTag = FindComponent<MeshInstanceLeaf>(g.world);
    const ComponentId skinnedTag = FindComponent<SkinnedMeshInstanceLeaf>(g.world);
    for (SceneGraphNode* node : nodes)
    {
        const Entity entity = g.graph->GetEntity(node);
        CHECK(meshTag == ComponentId::Invalid || !g.world->HasComponent(entity, meshTag));
        CHECK(skinnedTag == ComponentId::Invalid || !g.world->HasComponent(entity, skinnedTag));
    }
    CHECK(TakeWarnings() == 0);
}

// -------------------------------------------------------------------------------------------------- I3-N4

// A system ordered against a name that matches no system: the scheduler warns once while the name stays missing,
// not at every rebuild (another system added, a listing for an inspector that rebuilds the order); the diagnostics
// report it at every build. When the name comes to name a system of another phase, that is a new mismatch: one
// warning. Once the ordering is gone (its system removed) and comes back, it warns again.
void test_i3_n4_missing_name_warned_once()
{
    Graph g(false);
    IScheduler* scheduler = g.graph->GetScheduler();
    auto add = [&](const char* name, const char* before, Phase phase = Phase::EndFrame)
    {
        const char* const list[1] = { before };
        SystemDesc desc{};
        desc.name = name;
        desc.phase = phase;
        desc.runBefore = before ? list : nullptr;
        desc.runBeforeCount = before ? 1 : 0;
        nvrhi::AutoPtr<ISystem> system;
        CHECK(CreateLambdaSystem(desc, [](ISystemContext*) noexcept { return nvrhi::FS_OK; }, &system) ==
              nvrhi::FS_OK);
        SystemId id = SystemId::Invalid;
        CHECK(scheduler->AddSystem(system, id) == nvrhi::FS_OK);
        return id;
    };
    const SystemId lost = add("lost", "nobody");
    CHECK(scheduler->Build() == nvrhi::FS_OK);
    CHECK(TakeWarnings() == 1);
    add("other", nullptr);
    CHECK(scheduler->Build() == nvrhi::FS_OK);
    SystemInfo info{};
    CHECK(scheduler->GetSystemInfo(0, info) == nvrhi::FS_OK);
    g.Frame();
    CHECK(TakeWarnings() == 0);
    nvrhi::string text;
    scheduler->GetDiagnostics(text);
    CHECK(std::string(text.c_str()).find("'lost' runs before 'nobody', which matches no system") != std::string::npos);
    add("listed", nullptr);
    CHECK(scheduler->GetSystemInfo(0, info) == nvrhi::FS_OK);   // rebuilds the order, without Build
    CHECK(TakeWarnings() == 0);

    // 'nobody' is a system now, of another phase: another mismatch, warned once
    const SystemId nobody = add("nobody", nullptr, Phase::BeginFrame);
    CHECK(scheduler->Build() == nvrhi::FS_OK);
    CHECK(TakeWarnings() == 1);
    add("more", nullptr);
    CHECK(scheduler->Build() == nvrhi::FS_OK);
    CHECK(TakeWarnings() == 0);
    CHECK(scheduler->RemoveSystem(nobody) == nvrhi::FS_OK);
    CHECK(scheduler->Build() == nvrhi::FS_OK);
    CHECK(TakeWarnings() == 1);   // matches no system again

    CHECK(scheduler->RemoveSystem(lost) == nvrhi::FS_OK);
    CHECK(scheduler->Build() == nvrhi::FS_OK);
    add("lost", "nobody");
    CHECK(scheduler->Build() == nvrhi::FS_OK);
    CHECK(TakeWarnings() == 1);
}

// ------------------------------------------------------------------------------------------- I4 (RowTracked)

namespace e7_fixes
{
    struct Plain
    {
        float value;
    };
    DONUT_ECS_COMPONENT(Plain, "5e0c7a2b-91d4-4c3f-8b6e-2a7d1f9c4e83", DONUT_ECS_FIELDS(value))
    struct Marker { };
    DONUT_ECS_TAG(Marker, "b3a91f7e-6d28-4e05-9c4a-1f8e2d7b6c59")
    struct Tracked
    {
        float value;
    };
    DONUT_ECS_COMPONENT_EX(Tracked, "0d6f3b8a-2c41-47e9-a5d2-9e1b7c3f8a64",
                           ::donut::engine::ComponentFlags::RowTracked, 0, DONUT_ECS_FIELDS(value))
    struct Setting
    {
        float value;
    };
    DONUT_ECS_COMPONENT_EX(Setting, "7a3e9c15-4b6d-4f28-8e01-c5d2b9a7f364",
                           ::donut::engine::ComponentFlags::Resource, 0, DONUT_ECS_FIELDS(value))
}

// A query whose chunks have no Enableable, Sparse or RowTracked term hands out views whose masks and row versions are
// one null array, the same for every query (backlog I4-R2: no block per query for it).
void test_i4_r2_shared_null_views()
{
    Graph g(false);
    const ComponentId plain = RegisterComponent<e7_fixes::Plain>(g.world);
    const ComponentId ids[1] = { plain };
    EntityCreateDesc desc{};
    desc.components = ids;
    desc.componentCount = 1;
    desc.entityCount = 3;
    Entity made[3];
    CHECK(g.world->CreateEntities(desc, made) == nvrhi::FS_OK);
    Query<Read<e7_fixes::Plain>> first(g.world, "first");
    Query<Write<e7_fixes::Plain>> second(g.world, "second");
    const void* arrays[2][3] = {};
    uint32_t chunks[2] = {};
    first.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, const e7_fixes::Plain*)
    {
        arrays[0][0] = view.enabledMasks;
        arrays[0][1] = view.presenceMasks;
        arrays[0][2] = view.rowVersions;
        CHECK(!view.enabledMasks[0] && !view.presenceMasks[0] && !view.rowVersions[0]);
        ++chunks[0];
    });
    second.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, e7_fixes::Plain*)
    {
        arrays[1][0] = view.enabledMasks;
        arrays[1][1] = view.presenceMasks;
        arrays[1][2] = view.rowVersions;
        ++chunks[1];
    });
    CHECK(chunks[0] == 1 && chunks[1] == 1);
    for (int q = 0; q < 2; ++q)
        for (int a = 0; a < 3; ++a)
            CHECK(arrays[q][a] == arrays[0][0]);
}

// Registering a GUID again with RowTracked when it was registered without it: FS_OK and the existing id, but the rows
// get no row versions, so a warning says so, once per type; the type keeps its flags. A kind without a value per row
// (a tag here) is refused as a fresh registration of it is (FE_INVALID_ARGS), with one warning per type (F3-N1). The
// same flags again: no warning.
void test_i4_r4_row_tracked_reregistered()
{
    Graph g(false);
    const ComponentId plain = RegisterComponent<e7_fixes::Plain>(g.world);
    const ComponentId marker = RegisterComponent<e7_fixes::Marker>(g.world);
    CHECK(plain != ComponentId::Invalid && marker != ComponentId::Invalid);
    CHECK(TakeWarnings() == 0);
    ComponentTypeDesc tracked = ComponentTraits<e7_fixes::Plain>::Desc();
    tracked.flags = tracked.flags | ComponentFlags::RowTracked;
    ComponentId id = ComponentId::Invalid;
    CHECK(g.world->RegisterComponent(tracked, id) == nvrhi::FS_OK && id == plain);
    CHECK(TakeWarnings() == 1);
    CHECK(g.world->RegisterComponent(tracked, id) == nvrhi::FS_OK && id == plain);
    CHECK(TakeWarnings() == 0);   // once per type
    CHECK(!(g.world->GetComponentTypeDesc(plain)->flags & ComponentFlags::RowTracked));   // kept
    ComponentTypeDesc trackedTag = ComponentTraits<e7_fixes::Marker>::Desc();
    trackedTag.flags = trackedTag.flags | ComponentFlags::RowTracked;
    // the refusal is reported as a fresh registration's is: through the world's validation (Debug for a graph's
    // world), at every call (F4-N3)
    CHECK(g.world->RegisterComponent(trackedTag, id) == nvrhi::FE_INVALID_ARGS && id == ComponentId::Invalid);
    CHECK(TakeWarnings() == c_ValidationWarnings);
    CHECK(g.world->RegisterComponent(trackedTag, id) == nvrhi::FE_INVALID_ARGS);
    CHECK(TakeWarnings() == c_ValidationWarnings);
    CHECK(!(g.world->GetComponentTypeDesc(marker)->flags & ComponentFlags::RowTracked));
    CHECK(g.world->RegisterComponent(ComponentTraits<e7_fixes::Marker>::Desc(), id) == nvrhi::FS_OK && id == marker);
    // a Resource, through RegisterResource (F4 review N4): refused as a fresh one is
    const e7_fixes::Setting setting{ 2.f };
    ComponentId resource = ComponentId::Invalid;
    CHECK(g.world->RegisterResource(ComponentTraits<e7_fixes::Setting>::Desc(), &setting, resource) == nvrhi::FS_OK);
    ComponentTypeDesc trackedResource = ComponentTraits<e7_fixes::Setting>::Desc();
    trackedResource.flags = trackedResource.flags | ComponentFlags::RowTracked;
    CHECK(g.world->RegisterResource(trackedResource, &setting, id) == nvrhi::FE_INVALID_ARGS);
    CHECK(TakeWarnings() == c_ValidationWarnings);
    CHECK(g.world->RegisterResource(ComponentTraits<e7_fixes::Setting>::Desc(), &setting, id) == nvrhi::FS_OK &&
          id == resource);
    CHECK(g.world->RegisterComponent(ComponentTraits<e7_fixes::Plain>::Desc(), id) == nvrhi::FS_OK && id == plain);
    // registered with it, asked without it: the rows keep their row versions; no warning
    const ComponentId rowTracked = RegisterComponent<e7_fixes::Tracked>(g.world);
    ComponentTypeDesc untracked = ComponentTraits<e7_fixes::Tracked>::Desc();
    untracked.flags = ComponentFlags(uint32_t(untracked.flags) & ~uint32_t(ComponentFlags::RowTracked));
    CHECK(g.world->RegisterComponent(untracked, id) == nvrhi::FS_OK && id == rowTracked);
    CHECK(TakeWarnings() == 0);
}

// WorldStats::rowVersionBytes: the row-version blocks of the RowTracked columns, one per chunk index an archetype
// made, (rowsPerChunk + 1) versions per RowTracked column; zero without RowTracked components. chunkBytes stays the
// chunks' pages alone.
void test_i4_r5_row_version_bytes()
{
    Graph g(false);
    WorldStats before{};
    g.world->GetStats(before);
    const ComponentId tracked = RegisterComponent<e7_fixes::Tracked>(g.world);
    const ComponentId ids[1] = { tracked };
    EntityCreateDesc desc{};
    desc.components = ids;
    desc.componentCount = 1;
    desc.entityCount = 5000;
    std::vector<Entity> made(5000);
    CHECK(g.world->CreateEntities(desc, made.data()) == nvrhi::FS_OK);
    WorldStats after{};
    g.world->GetStats(after);
    uint64_t expected = 0;
    uint64_t chunkBytes = 0;
    for (uint32_t a = 0; a < after.archetypeCount; ++a)
    {
        ArchetypeInfo info{};
        CHECK(g.world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
        chunkBytes += uint64_t(info.chunkCount) * info.chunkSize;
        if (std::find(info.components, info.components + info.componentCount, tracked) !=
            info.components + info.componentCount)
            expected += uint64_t(info.chunkCount) * (uint64_t(info.rowsPerChunk) + 1) * sizeof(uint64_t);
    }
    CHECK(expected > 0 && after.rowVersionBytes - before.rowVersionBytes == expected);
    CHECK(after.chunkBytes == chunkBytes);
}

// ------------------------------------------------------------------------------------------------- I3-R5

namespace e7_fixes
{
    struct Precise
    {
        double wide;
        float narrow;
        uint32_t pad;
    };
    DONUT_ECS_COMPONENT(Precise, "3c8e1f74-9a2b-4d65-b0e7-5f1a2c9d8e46", DONUT_ECS_FIELDS(wide, narrow))

    // Types the editor's text back unchanged (the field shown, then committed as typed), as a user who edits a digit
    // of a value commits the digits shown.
    bool TypeAsShown(void* context, engine::ComponentId, uint32_t, uint32_t, char* text, uint32_t)
    {
        ++*static_cast<uint32_t*>(context);
        return text[0] != 0;
    }
}

// The number editors show every digit their type holds (%.17g for Double, %.9g for Float; backlog I3-R5): a value
// typed back as shown is the same value, bit for bit. With %.6g, 123456789.123456789 came back as 123457000.
void test_i3_r5_editor_precision()
{
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(1280.f, 1024.f);
    io.DeltaTime = 1.f / 60.f;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    {
        Graph g(false);
        const ComponentId precise = RegisterComponent<e7_fixes::Precise>(g.world);
        const e7_fixes::Precise value{ 123456789.123456789, 0.1f + 1e-7f, 0 };
        const ComponentId ids[1] = { precise };
        const void* values[1] = { &value };
        EntityCreateDesc desc{};
        desc.components = ids;
        desc.values = values;
        desc.componentCount = 1;
        desc.entityCount = 1;
        Entity entity = Entity::Null;
        CHECK(g.world->CreateEntities(desc, &entity) == nvrhi::FS_OK);

        uint32_t typed = 0;
        app::detail::WorldInspectorTestHook hook;
        hook.select = entity;
        hook.openAll = true;
        hook.type = &e7_fixes::TypeAsShown;
        hook.context = &typed;
        app::detail::SetWorldInspectorTestHook(&hook);
        app::WorldInspector inspector;
        for (int f = 0; f < 2; ++f)
        {
            ImGui::NewFrame();
            inspector.Draw(g.world, nullptr);
            ImGui::Render();
        }
        app::detail::SetWorldInspectorTestHook(nullptr);
        CHECK(typed >= 2);   // both fields were typed
        CHECK(inspector.Apply(g.world) == nvrhi::FS_OK);
        const e7_fixes::Precise* after = GetComponent<e7_fixes::Precise>(g.world, entity);
        CHECK(after && SameValue(after->wide, value.wide) && SameValue(after->narrow, value.narrow));
    }
    ImGui::DestroyContext();
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------- T8-1

// A mesh edited in place inside a Defer scope: its reconciliation waits for the last EndRenderRead, and when it fails
// there (out of memory: the store's test seam), one warning, the mesh table keeps its old ranges, and the next Refresh
// runs it, also with nothing else changed (SceneGraph::Refresh's retry).
void test_t8_1_deferred_reconcile_retried()
{
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh();
    Graph g(false);
    g.Add(g.graph->GetRootNode(), dm::double3(0.0), mesh, false);
    g.graph->Refresh(0);
    const uint32_t slot = MeshTable::SlotOf(g.graph->GetMeshes().Find(mesh));
    CHECK(g.graph->GetMeshes().GetMeshRow(slot).geometryCount == 1);

    const uint32_t scope = g.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
    auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);   // an in-place edit, then a structural change
    geometry->material = mesh->geometries[0]->material;
    geometry->numIndices = 3;
    geometry->numVertices = 3;
    mesh->geometries.push_back(geometry);
    g.Add(g.graph->GetRootNode(), dm::double3(1.0), nullptr, false);
    g.graph->Refresh(1);
    CHECK(g.graph->GetMeshes().GetMeshRow(slot).geometryCount == 1);   // deferred
    SceneGraphStore::FailReconcileAfter(0);
    g.graph->EndRenderRead(scope);
    SceneGraphStore::FailReconcileAfter(-1);
    CHECK(TakeWarnings() == 1);   // "the deferred mesh reconciliation failed"
    CHECK(g.graph->GetMeshes().GetMeshRow(slot).geometryCount == 1);

    g.graph->Refresh(2);   // nothing else changed: the retry
    CHECK(g.graph->GetMeshes().GetMeshRow(slot).geometryCount == 2);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------- T7-2

// The shared job system takes its worker count from DONUT_JOB_WORKERS when it is set (ctest test_e7_fixes_job_workers
// runs this case with 2): GetThreadCount is its 64 external slots and those workers. Unset: the default.
// The expected count comes from the command line (ctest: the count it sets DONUT_JOB_WORKERS to), not from the
// function under test. Without it (a run of every case) the case only checks that the shared system is made.
void test_t7_2_shared_job_workers()
{
    const uint32_t workers = GetSharedJobWorkersOverride();
    IJobSystem* shared = GetSharedJobSystem();
    CHECK(shared);
    if (g_ExpectedWorkers != 0)
    {
        CHECK(workers == g_ExpectedWorkers);
        CHECK(shared->GetThreadCount() == 64 + g_ExpectedWorkers);   // 64 external slots
    }
    else
    {
        CHECK(shared->GetThreadCount() > 64);
    }
    std::printf("  shared job system: %u threads (DONUT_JOB_WORKERS %u)\n", shared->GetThreadCount(), workers);
    CHECK(TakeWarnings() == 0);
}

// Each value that is no count from 1 to 256 gives the default (0) and one warning; the bounds are accepted. The case
// sets the variable and restores it after, and never makes the shared system (it reads the variable once, when made).
void test_t7_2_invalid_workers()
{
    std::string previous;
    const bool wasSet = GetWorkersVariable(previous);
    struct Restore
    {
        const std::string& value;
        bool set;
        ~Restore() { SetWorkersVariable(set ? value.c_str() : nullptr); }
    } restore{ previous, wasSet };

    struct Value { const char* text; uint32_t workers; };
    const Value values[] = { { "1", 1 }, { "8", 8 }, { "256", 256 } };
    for (const Value& v : values)
    {
        SetWorkersVariable(v.text);
        CHECK(GetSharedJobWorkersOverride() == v.workers);
        CHECK(TakeWarnings() == 0);
    }
    const char* const invalid[] = { "0", "257", "abc", "2x", "-1", "99999999999999999999" };
    for (const char* text : invalid)
    {
        SetWorkersVariable(text);
        CHECK(GetSharedJobWorkersOverride() == 0);
        CHECK(TakeWarnings() == 1);
    }
    // An empty value: Windows removes the variable (unset: no warning); elsewhere it is set and empty (a warning).
    SetWorkersVariable("");
    CHECK(GetSharedJobWorkersOverride() == 0);
#if defined(_WIN32)
    CHECK(TakeWarnings() == 0);
#else
    CHECK(TakeWarnings() == 1);
#endif
    SetWorkersVariable(nullptr);
    CHECK(GetSharedJobWorkersOverride() == 0);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------- I6-R5

// GpuScene::GetStats reads the statistics the recording's upload writes: while a pipelined frame records, a call on
// the main thread outside recording work warns once in Debug builds, like the other owner-checked calls; inside a
// PipelinedRecordingWork scope it does not.
void test_i6_r5_gpu_scene_stats_checked()
{
#ifndef NDEBUG
    const uint32_t expected = 1;
#else
    const uint32_t expected = 0;
#endif
    GpuSceneDesc desc;
    desc.geometryAndMaterialBuffers = true;
    GpuScene gpu(nullptr, desc);
    (void)gpu.GetStats();
    CHECK(TakeWarnings() == 0);   // serial: nothing checked
    gpu.BeginPipelinedRecording();
    (void)gpu.GetStats();
    (void)gpu.HasPendingAssets();
    (void)gpu.IsScatterPending();
    CHECK(TakeWarnings() == 0);   // reads before the hand-off: no recording runs (the I7-R4 split)
    gpu.HandOverPipelinedRecording();
    {
        PipelinedRecordingWork work;
        (void)gpu.GetStats();
    }
    CHECK(TakeWarnings() == 0);
    (void)gpu.GetStats();
    (void)gpu.GetStats();
    CHECK(TakeWarnings() == expected);   // once per recording
    gpu.EndPipelinedRecording();
    (void)gpu.GetStats();
    CHECK(TakeWarnings() == 0);

    // a mutating call is checked from the begin (the end of the frame's Build), before the hand-off
    gpu.BeginPipelinedRecording();
    gpu.InvalidateAssets();
    CHECK(TakeWarnings() == expected);
    gpu.EndPipelinedRecording();
    CHECK(TakeWarnings() == 0);
}

// ---------------------------------------------------------------------------------------------- I3-R1, I3-R2

// The domain's columns that an edit through the world would bypass: SimTeleport (RequestTeleport's normalising and
// space) and SimPosePrevious (the domain derives it) are on donut's read-only list; SimPose and SimScale are not
// (open point 6).
void test_i3_r1_r2_read_only_list()
{
    Graph g(false);
    ComponentId ids[c_MaxInspectorReadOnlyComponents] = {};
    const uint32_t count = app::GetDonutReadOnlyComponents(g.world, ids, c_MaxInspectorReadOnlyComponents);
    CHECK(count == 16);
    auto listed = [&](ComponentId id)
    {
        return id != ComponentId::Invalid && std::find(ids, ids + count, id) != ids + count;
    };
    CHECK(listed(FindComponent<SimTeleport>(g.world)));
    CHECK(listed(FindComponent<SimPosePrevious>(g.world)));
    CHECK(listed(FindComponent<SimBodyRef>(g.world)));
    CHECK(!listed(FindComponent<SimPose>(g.world)) && !listed(FindComponent<SimScale>(g.world)));
    CHECK(!listed(FindComponent<LocalTransform>(g.world)));
}

int main(int argc, char** argv)
{
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        { "i5_r1_deep_static_bounds", test_i5_r1_deep_static_bounds },
        { "i5_r2_deep_static_set_transform", test_i5_r2_deep_static_set_transform },
        { "i5_r2_static_first_transform", test_i5_r2_static_first_transform },
        { "i5_r3_static_rows_join_late", test_i5_r3_static_rows_join_late },
        { "i5_r4_compose_through_static", test_i5_r4_compose_through_static },
        { "i3_r1_r2_read_only_list", test_i3_r1_r2_read_only_list },
        { "e72_r4_mesh_columns_of_other_classes", test_e72_r4_mesh_columns_of_other_classes },
        { "e72_r4_mesh_tags_of_other_classes", test_e72_r4_mesh_tags_of_other_classes },
        { "i3_n4_missing_name_warned_once", test_i3_n4_missing_name_warned_once },
        { "i4_r2_shared_null_views", test_i4_r2_shared_null_views },
        { "i4_r4_row_tracked_reregistered", test_i4_r4_row_tracked_reregistered },
        { "i4_r5_row_version_bytes", test_i4_r5_row_version_bytes },
        { "i6_r5_gpu_scene_stats_checked", test_i6_r5_gpu_scene_stats_checked },
        { "i3_r5_editor_precision", test_i3_r5_editor_precision },
        { "t8_1_deferred_reconcile_retried", test_t8_1_deferred_reconcile_retried },
        { "t7_2_shared_job_workers", test_t7_2_shared_job_workers },
        { "t7_2_invalid_workers", test_t7_2_invalid_workers },
    };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2)
        g_ExpectedWorkers = uint32_t(std::strtoul(argv[2], nullptr, 10));
    int failed = 0;
    for (const Case& c : cases)
    {
        if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
            continue;
        try
        {
            c.run();
            CHECK(TakeWarnings() == 0);
            std::printf("%s: passed\n", c.name);
        }
        catch (const std::runtime_error& err)
        {
            std::fprintf(stderr, "%s: FAILED at %s\n", c.name, err.what());
            TakeWarnings();
            ++failed;
        }
    }
    return failed ? 1 : 0;
}
