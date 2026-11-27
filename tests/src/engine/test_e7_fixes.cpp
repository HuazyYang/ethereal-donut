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

#include <donut/app/WorldInspector.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    std::atomic<uint32_t> g_Warnings{ 0 };   // log::warning calls; a case that provokes some takes them back
    uint32_t TakeWarnings() { return g_Warnings.exchange(0); }

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
