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

// The scene graph over the world (aggregate docs/plans/2026-11-21-donut-ecs.md, step E3; docs/ecs-design.md 4.6, 4.7,
// 4.11, section 5): the columns of attached nodes and their links, handles and orphan records through attach and
// detach, the lifetime rules of 5.3 including graph teardown, leaf registration through the LeafObject observer, depth
// buckets and the last bucket's path against a reference composition, bounds and chunk bounds, the on-demand caches
// (also read by threads at once), pending-change flags, the joint frame index, determinism at 1, 2 and 8 workers, and
// zero heap allocations in steady state. The ECS heap counter comes from the private header EcsWorldImpl.h.

#include <donut/engine/SceneGraph.h>
#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/JobSystem.h>
#include <donut/core/query_cast.h>
#include <donut/tests/e7_switches.h>
#include <donut/tests/utils.h>
#include <donut/core/log.h>

#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>

#include "EcsWorldImpl.h"
#include "SceneGraphStore.h"

// InstanceData, for the GpuScene rows of the Static switch's cases (stage T5); the shader header needs the math names.
using namespace donut::math;
#include <donut/shaders/bindless.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test and donut_engine, which links statically). The world's own blocks
// come from nvrhi_core's allocator, which ecs::GetHeapAllocationCount counts.
namespace
{
    std::atomic<uint64_t> g_NewCount{ 0 };

    void* CountedAllocate(size_t size) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
        return std::malloc(size ? size : 1);
    }

    void* CountedAllocateAligned(size_t size, size_t alignment) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
        return _aligned_malloc(size ? size : 1, alignment);
#else
        return std::aligned_alloc(alignment, ((size ? size : 1) + alignment - 1) & ~(alignment - 1));
#endif
    }

    void CountedFreeAligned(void* memory) noexcept
    {
#if defined(_MSC_VER)
        _aligned_free(memory);
#else
        std::free(memory);
#endif
    }
}

void* operator new(size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new[](size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new(size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }
void operator delete(void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete[](void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete(void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }

using namespace donut;
using namespace donut::engine;

// log::warning calls; main fails a case that leaves any (Debug worlds report validation errors through it). A case that
// provokes warnings on purpose checks them and takes them back.
std::atomic<uint32_t> g_Warnings{ 0 };

namespace
{
    class Rng
    {
    public:
        explicit Rng(uint64_t seed) : m_State(seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull) { }
        uint64_t Next()
        {
            m_State ^= m_State >> 12;
            m_State ^= m_State << 25;
            m_State ^= m_State >> 27;
            return m_State * 0x2545F4914F6CDD1Dull;
        }
        uint32_t Below(uint32_t bound) { return bound ? uint32_t(Next() % bound) : 0; }
        double Range(double lo, double hi)
        {
            return lo + (hi - lo) * (double(Next() >> 11) * (1.0 / 9007199254740992.0));
        }

    private:
        uint64_t m_State;
    };

    nvrhi::AutoPtr<SceneGraphNode> MakeNode(const char* name)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName(name);
        return node;
    }

    nvrhi::AutoPtr<MeshInfo> MakeMesh(SceneTypeFactory* factory, float size)
    {
        auto material = factory->CreateMaterial();
        auto geometry = factory->CreateMeshGeometry();
        geometry->material = material;
        geometry->objectSpaceBounds = dm::box3(dm::float3(-size), dm::float3(size));
        auto mesh = factory->CreateMesh();
        mesh->geometries.push_back(geometry);
        mesh->objectSpaceBounds = geometry->objectSpaceBounds;
        return mesh;
    }

    void RandomMove(Rng& rng, SceneGraphNode* node)
    {
        const double tx = rng.Range(-5.0, 5.0);
        const double ty = rng.Range(-5.0, 5.0);
        const double tz = rng.Range(-5.0, 5.0);
        const double qx = rng.Range(-1.0, 1.0);
        const double qy = rng.Range(-1.0, 1.0);
        const double qz = rng.Range(-1.0, 1.0);
        const double qw = rng.Range(0.2, 1.0);
        const double s = rng.Range(0.9, 1.1);
        const double length = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        const dm::double3 translation(tx, ty, tz);
        const dm::dquat rotation(qw / length, qx / length, qy / length, qz / length);
        const dm::double3 scaling(s);
        node->SetTransform(&translation, &rotation, &scaling);
    }

    bool SameBits(const void* a, const void* b, size_t size) { return std::memcmp(a, b, size) == 0; }

    // The world transform as the old SceneGraph::Refresh composed it: local * parent world for a node with a transform,
    // the parent's world without one, local (or the identity) for the root.
    void ReferenceWorlds(SceneGraphNode* root, std::unordered_map<SceneGraphNode*, dm::daffine3>& out, IWorld* world,
                         SceneGraph* graph)
    {
        const ComponentId local = FindComponent<LocalTransform>(world);
        for (SceneGraphWalker walker(root); walker; walker.Next(true))
        {
            SceneGraphNode* node = walker.Get();
            const bool hasLocal = world->HasComponent(graph->GetEntity(node), local);
            dm::daffine3 localTransform = dm::scaling(node->GetScaling());
            localTransform *= node->GetRotation().toAffine();
            localTransform *= dm::translation(node->GetTranslation());
            if (!hasLocal)
                localTransform = dm::daffine3::identity();
            SceneGraphNode* parent = node == root ? nullptr : node->GetParent();
            if (!parent)
                out[node] = localTransform;
            else
                out[node] = hasLocal ? localTransform * out.at(parent) : out.at(parent);
        }
    }

    void CheckWorlds(SceneGraph* graph)
    {
        std::unordered_map<SceneGraphNode*, dm::daffine3> reference;
        ReferenceWorlds(graph->GetRootNode(), reference, graph->GetWorld(), graph);
        for (const auto& [node, expected] : reference)
        {
            const dm::daffine3 actual = node->GetLocalToWorldTransform();
            CHECK(SameBits(&actual, &expected, sizeof(actual)));
        }
    }

    nvrhi::AutoPtr<IJobSystem> MakeJobs(uint32_t workers)
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        JobSystemDesc desc;
        desc.debugName = "test_scene_graph_store";
        desc.workerThreadCount = workers;
        CHECK(NVRHI_SUCCEEDED(CreateJobSystem(desc, &jobs)));
        return jobs;
    }
}

// Every attached node is an entity with the columns of design 5.1; handles map both ways.
void test_columns_of_attached_nodes()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    IWorld* world = graph->GetWorld();
    graph->SetRootNode(MakeNode("root"));
    auto car = graph->Attach(graph->GetRootNode(), MakeNode("car"));
    auto body = MakeNode("body");
    body->SetTranslation(dm::double3(1.0, 2.0, 3.0));
    body->SetLeaf(factory->CreateMeshInstance(MakeMesh(factory, 1.f)));
    graph->Attach(car, body);
    auto spot = graph->AttachLeafNode(car, MAKE_RC_OBJ_PTR(SpotLight));
    graph->Refresh(0);

    const Entity rootEntity = graph->GetEntity(graph->GetRootNode());
    const Entity carEntity = graph->GetEntity(car);
    const Entity bodyEntity = graph->GetEntity(body);
    const Entity spotEntity = graph->GetEntity(spot);
    CHECK(rootEntity != Entity::Null && carEntity != Entity::Null && bodyEntity != Entity::Null);
    CHECK(graph->GetNode(bodyEntity) == body.Get());
    CHECK(graph->GetNode(Entity::Null) == nullptr);
    CHECK(graph->GetEntity(MakeNode("orphan")) == Entity::Null);

    // car: no leaf, no transform
    CHECK(world->HasComponent(carEntity, FindComponent<SceneNode>(world)));
    CHECK(!world->HasComponent(carEntity, FindComponent<LocalTransform>(world)));
    CHECK(!world->HasComponent(carEntity, FindComponent<LeafObject>(world)));
    CHECK(GetComponent<Parent>(world, carEntity)->entity == rootEntity);
    CHECK(GetComponent<HierarchyDepth>(world, carEntity)->depth == 1);
    CHECK(GetComponent<HierarchyDepth>(world, rootEntity)->depth == 0);
    CHECK(!world->HasComponent(rootEntity, FindComponent<Parent>(world)));

    // body: transform and mesh leaf
    CHECK(world->HasComponent(bodyEntity, FindComponent<LocalTransform>(world)));
    CHECK(world->HasComponent(bodyEntity, FindComponent<MeshInstanceLeaf>(world)));
    CHECK(GetComponent<LeafObject>(world, bodyEntity)->object == body->GetLeaf());
    CHECK(GetComponent<HierarchyDepth>(world, bodyEntity)->depth == 2);
    CHECK(all(GetComponent<LocalTransform>(world, bodyEntity)->translation == dm::double3(1.0, 2.0, 3.0)));
    const dm::box3 bodyBounds = GetComponent<WorldBounds>(world, bodyEntity)->bounds;
    CHECK(all(bodyBounds.m_mins == dm::float3(0.f, 1.f, 2.f)) && all(bodyBounds.m_maxs == dm::float3(2.f, 3.f, 4.f)));

    // spot: one tag per class in its chain
    CHECK(world->HasComponent(spotEntity, FindComponent<LightLeaf>(world)));
    CHECK(world->HasComponent(spotEntity, FindComponent<SpotLightLeaf>(world)));
    CHECK(FindComponent<PointLightLeaf>(world) == ComponentId::Invalid ||
          !world->HasComponent(spotEntity, FindComponent<PointLightLeaf>(world)));
    CHECK(GetComponent<WorldBounds>(world, spotEntity)->bounds.isempty());

    // SceneLinks agree with the node links
    const SceneLinks* carLinks = GetComponent<SceneLinks>(world, carEntity);
    CHECK(carLinks->firstChild == bodyEntity && carLinks->lastChild == spotEntity);
    CHECK(GetComponent<SceneLinks>(world, bodyEntity)->nextSibling == spotEntity);
    CHECK(GetComponent<SceneLinks>(world, spotEntity)->prevSibling == bodyEntity);
    CHECK(car->GetFirstChild() == body.Get() && car->GetLastChild() == spot.Get());

    // the leaf type descriptors (D19)
    CHECK(spot->GetLeaf()->GetTypeDesc().tagCount == 2);
    CHECK(body->GetLeaf()->GetTypeDesc().classId == IID_MeshInstance);
    auto reference = MAKE_RC_OBJ_PTR(SkinnedMeshReference, static_cast<SkinnedMeshInstance*>(nullptr));
    CHECK(reference->GetTypeDesc().tagCount == 1);

    // SetLeaf to another kind changes the tags; the leaf vectors follow
    spot->SetLeaf(MAKE_RC_OBJ_PTR(PerspectiveCamera));
    CHECK(!world->HasComponent(spotEntity, FindComponent<LightLeaf>(world)));
    CHECK(world->HasComponent(spotEntity, FindComponent<CameraLeaf>(world)));
    CHECK(world->HasComponent(spotEntity, FindComponent<PerspectiveCameraLeaf>(world)));
    CHECK(graph->GetLights().empty() && graph->GetCameras().size() == 1);
    // the first transform of a node adds LocalTransform (and its cache)
    car->SetScaling(dm::double3(2.0));
    CHECK(world->HasComponent(carEntity, FindComponent<LocalTransform>(world)));
    CHECK(world->HasComponent(carEntity, FindComponent<LocalMatrix>(world)));
    graph->Refresh(1);
    CHECK(all(body->GetLocalToWorldTransform().m_translation == dm::double3(2.0, 4.0, 6.0)));
    CheckWorlds(graph);
}

// Orphan records: state set before attach moves into rows; detach moves it back; links and leaves survive both ways.
void test_orphan_records_round_trip()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    graph->GetRootNode()->SetTranslation(dm::double3(10.0, 0.0, 0.0));
    auto other = graph->Attach(graph->GetRootNode(), MakeNode("other"));
    other->SetTranslation(dm::double3(0.0, 5.0, 0.0));

    // an orphaned subgraph: p(T) -> q -> r(T, mesh), p -> s(light)
    auto p = MakeNode("p");
    p->SetTranslation(dm::double3(1.0, 0.0, 0.0));
    auto q = MakeNode("q");
    auto r = MakeNode("r");
    r->SetTranslation(dm::double3(0.0, 0.0, 2.0));
    auto leaf = factory->CreateMeshInstance(mesh);
    r->SetLeaf(leaf);
    auto s = MakeNode("s");
    s->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
    graph->Attach(q, r);
    graph->Attach(p, q);
    graph->Attach(p, s);
    CHECK(!p->GetGraph() && r->GetParent() == q.Get() && p->GetLastChild() == s.Get());
    CHECK(all(r->GetTranslation() == dm::double3(0.0, 0.0, 2.0)));
    CHECK(graph->GetMeshInstances().empty());

    graph->Attach(graph->GetRootNode(), p);
    CHECK(p->GetGraph() == graph.Get() && r->GetGraph() == graph.Get());
    CHECK(graph->GetMeshInstances().size() == 1 && graph->GetLights().size() == 1);
    CHECK(leaf->GetNode() == r.Get());
    graph->Refresh(0);
    CHECK(all(r->GetLocalToWorldTransform().m_translation == dm::double3(11.0, 0.0, 2.0)));
    CHECK(all(q->GetLocalToWorldTransform().m_translation == dm::double3(11.0, 0.0, 0.0)));
    CheckWorlds(graph);

    WorldStats stats;
    graph->GetWorld()->GetStats(stats);
    CHECK(stats.entityCount == 6);

    // detach: the rows move back into records; the entities go
    auto detached = graph->Detach(p);
    CHECK(detached.Get() == p.Get() && !p->GetGraph() && !p->GetParent());
    graph->GetWorld()->GetStats(stats);
    CHECK(stats.entityCount == 2);
    CHECK(graph->GetMeshInstances().empty() && graph->GetLights().empty());
    CHECK(r->GetParent() == q.Get() && q->GetParent() == p.Get() && p->GetFirstChild() == q.Get());
    CHECK(r->GetLeaf() == leaf.Get() && leaf->GetNode() == r.Get());
    CHECK(all(r->GetTranslation() == dm::double3(0.0, 0.0, 2.0)));
    // the world transform of its last graph is kept while it is in none
    CHECK(all(r->GetLocalToWorldTransform().m_translation == dm::double3(11.0, 0.0, 2.0)));

    // edits while detached, then attached elsewhere
    q->SetRotation(dm::dquat::identity());
    graph->Attach(other, p);
    graph->Refresh(1);
    CHECK(all(r->GetLocalToWorldTransform().m_translation == dm::double3(11.0, 5.0, 2.0)));
    CHECK(GetComponent<HierarchyDepth>(graph->GetWorld(), graph->GetEntity(r))->depth == 4);
    CheckWorlds(graph);
}

// Detach keeps the order of the remaining leaves in every vector, and batch removal leaves no marks behind.
void test_detach_keeps_leaf_order()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> groups;
    std::vector<MeshInstance*> expected;
    for (int g = 0; g < 4; g++)
    {
        auto group = graph->Attach(graph->GetRootNode(), MakeNode("group"));
        groups.push_back(group);
        for (int i = 0; i < 50; i++)
        {
            auto instance = factory->CreateMeshInstance(mesh);
            graph->AttachLeafNode(group, instance);
            if (g != 1)
                expected.push_back(instance);
        }
    }
    auto detached = graph->Detach(groups[1]);
    CHECK(graph->GetMeshInstances().size() == expected.size());
    for (size_t i = 0; i < expected.size(); i++)
        CHECK(graph->GetMeshInstances()[i].Get() == expected[i]);

    // attached again: appended in pre-order, and a second detach removes exactly them again
    graph->Attach(graph->GetRootNode(), detached);
    CHECK(graph->GetMeshInstances().size() == 200);
    graph->Detach(groups[2]);
    CHECK(graph->GetMeshInstances().size() == 150);
    graph->Detach(groups[1]);
    CHECK(graph->GetMeshInstances().size() == 100);
}

// Teardown (5.3): nodes referenced elsewhere outlive the graph as roots of intact subgraphs; the rest goes.
void test_teardown_keeps_referenced_nodes()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);
    nvrhi::AutoPtr<SceneGraphNode> held;
    nvrhi::AutoPtr<SceneGraphLeaf> heldLeaf;
    nvrhi::WeakPtr<SceneGraphNode> weakSibling;
    nvrhi::WeakPtr<SceneGraphNode> weakChild;
    nvrhi::WeakPtr<SceneGraphNode> weakLeafNode;
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MakeNode("root"));
        auto a = graph->Attach(graph->GetRootNode(), MakeNode("a"));
        a->SetTranslation(dm::double3(3.0, 0.0, 0.0));
        auto a1 = graph->Attach(a, MakeNode("a1"));
        auto a2 = graph->Attach(a, MakeNode("a2"));
        auto b = graph->Attach(graph->GetRootNode(), MakeNode("b"));
        auto leaf = factory->CreateMeshInstance(mesh);
        auto c = graph->AttachLeafNode(graph->GetRootNode(), leaf);
        graph->Refresh(0);
        held = a;
        heldLeaf = leaf;
        weakSibling = nvrhi::WeakPtr<SceneGraphNode>(b.Get());
        weakChild = nvrhi::WeakPtr<SceneGraphNode>(a1.Get());
        weakLeafNode = nvrhi::WeakPtr<SceneGraphNode>(c.Get());
    }
    CHECK(held && !held->GetGraph() && !held->GetParent() && !held->GetNextSibling() && !held->GetPrevSibling());
    CHECK(held->GetFirstChild() && held->GetFirstChild()->GetName() == "a1");
    CHECK(held->GetLastChild()->GetName() == "a2");
    CHECK(weakChild.Lock());
    CHECK(!weakSibling.Lock());
    CHECK(!weakLeafNode.Lock());
    CHECK(heldLeaf->GetNode() == nullptr);
    CHECK(all(held->GetTranslation() == dm::double3(3.0, 0.0, 0.0)));
    CHECK(all(held->GetLocalToWorldTransform().m_translation == dm::double3(3.0, 0.0, 0.0)));

    // the survivor attaches to another graph as any orphan does
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    graph->Attach(graph->GetRootNode(), held);
    graph->Refresh(0);
    CHECK(held->GetGraph() == graph.Get() && held->GetFirstChild()->GetGraph() == graph.Get());
    held.Reset();
    graph.Reset();
    CHECK(!weakChild.Lock());
}

// Depth buckets past c_MaxHierarchyDepth: chains and a branching tree, moved at the top, in the middle and at the
// bottom, and parts detached, match the reference composition bit for bit at 1, 2 and 8 workers with equal hashes.
void test_deep_hierarchy_and_determinism()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 0.5f);
    uint64_t hashes[3] = {};
    const uint32_t workers[3] = { 1, 2, 8 };
    for (int w = 0; w < 3; w++)
    {
        auto jobs = MakeJobs(workers[w]);
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetJobSystem(jobs);
        Rng rng(7);
        graph->SetRootNode(MakeNode("root"));
        std::vector<SceneGraphNode*> nodes;
        for (int chain = 0; chain < 6; chain++)
        {
            SceneGraphNode* parent = graph->GetRootNode();
            for (int d = 0; d < 150; d++)
            {
                auto node = MakeNode("c");
                if (rng.Below(3) != 0)
                    RandomMove(rng, node);
                if (rng.Below(4) == 0)
                    node->SetLeaf(factory->CreateMeshInstance(mesh));
                parent = graph->Attach(parent, node);
                nodes.push_back(parent);
                // branches below the last bucket's depth
                if (d > 70 && rng.Below(10) == 0)
                {
                    auto branch = MakeNode("b");
                    RandomMove(rng, branch);
                    graph->Attach(parent, branch);
                }
            }
        }
        graph->Refresh(0);
        CheckWorlds(graph);
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> detached;   // keeps the raw pointers in `nodes` valid
        for (int frame = 1; frame < 8; frame++)
        {
            for (int i = 0; i < 12; i++)
                RandomMove(rng, nodes[rng.Below(uint32_t(nodes.size()))]);
            if (frame == 4)
                detached.push_back(graph->Detach(nodes[100]));   // part of the first chain, depth 101
            if (frame == 5)
                detached.push_back(graph->Detach(nodes[3 * 150 + 80]));   // inside the last bucket of the fourth chain
            if (frame == 6)
                graph->Attach(nodes[5 * 150 + 120], detached[0]);   // under the sixth chain's deep part
            graph->Refresh(uint32_t(frame));
            CheckWorlds(graph);
        }
        hashes[w] = graph->GetWorld()->ComputeHash();
    }
    CHECK(hashes[0] == hashes[1] && hashes[0] == hashes[2]);
}

// WorldBounds is the leaf's local box times the float world transform; ChunkWorldBounds is their union per chunk.
void test_bounds_and_chunk_bounds()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    IWorld* world = graph->GetWorld();
    Rng rng(11);
    graph->SetRootNode(MakeNode("root"));
    std::vector<SceneGraphNode*> nodes;
    for (int i = 0; i < 600; i++)
    {
        auto node = MakeNode("i");
        RandomMove(rng, node);
        node->SetLeaf(factory->CreateMeshInstance(MakeMesh(factory, 0.25f + float(i % 4))));
        nodes.push_back(graph->Attach(graph->GetRootNode(), node));
    }
    graph->Refresh(0);
    for (int i = 0; i < 50; i++)
        RandomMove(rng, nodes[rng.Below(uint32_t(nodes.size()))]);
    graph->Refresh(1);

    for (SceneGraphNode* node : nodes)
    {
        const dm::box3 expected = node->GetLeaf()->GetLocalBoundingBox() * node->GetLocalToWorldTransformFloat();
        const dm::box3 actual = GetComponent<WorldBounds>(world, graph->GetEntity(node))->bounds;
        CHECK(SameBits(&expected, &actual, sizeof(actual)));
        const dm::box3 global = node->GetGlobalBoundingBox();
        CHECK(SameBits(&expected, &global, sizeof(global)));
    }

    Query<Read<WorldBounds>, ReadChunk<ChunkWorldBounds>> query(world);
    uint32_t chunks = 0;
    query.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, const WorldBounds* bounds,
                                          const ChunkWorldBounds* chunk)
    {
        dm::box3 expected = dm::box3::empty();
        for (uint32_t r = 0; r < view.count; r++)
            expected |= bounds[r].bounds;
        CHECK(SameBits(&expected, &chunk->bounds, sizeof(expected)));
        ++chunks;
    });
    CHECK(chunks >= 2);

    // the root's subgraph box is the union of all of them
    dm::box3 all = dm::box3::empty();
    for (SceneGraphNode* node : nodes)
        all |= node->GetGlobalBoundingBox();
    const dm::box3 root = graph->GetRootNode()->GetGlobalBoundingBox();
    CHECK(SameBits(&all, &root, sizeof(root)));
}

// The on-demand caches change with the graph, and threads that read them at once agree.
void test_subgraph_caches()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    auto group = graph->Attach(graph->GetRootNode(), MakeNode("group"));
    Rng rng(3);
    for (int i = 0; i < 2000; i++)
    {
        auto node = MakeNode("i");
        RandomMove(rng, node);
        if (i % 3 == 0)
            node->SetLeaf(factory->CreateMeshInstance(MakeMesh(factory, 1.f)));
        else if (i % 3 == 1)
            node->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
        graph->Attach(i % 2 ? group.Get() : graph->GetRootNode(), node);
    }
    graph->Refresh(0);
    SceneGraphNode* root = graph->GetRootNode();
    CHECK(group->GetSubgraphContentFlags() == (SceneContentFlags::OpaqueMeshes | SceneContentFlags::Lights));
    CHECK(group->GetLeafContentFlags() == SceneContentFlags::None);
    const dm::box3 before = group->GetGlobalBoundingBox();

    // eight threads read the caches at once, after a change
    group->SetTranslation(dm::double3(100.0, 0.0, 0.0));
    graph->Refresh(1);
    std::vector<dm::box3> results(8);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; t++)
    {
        threads.emplace_back([&, t]()
        {
            dm::box3 box = dm::box3::empty();
            for (SceneGraphWalker walker(root); walker; walker.Next(true))
                box |= walker->GetGlobalBoundingBox();
            results[size_t(t)] = box;
        });
    }
    for (auto& thread : threads)
        thread.join();
    for (int t = 1; t < 8; t++)
        CHECK(SameBits(&results[0], &results[size_t(t)], sizeof(dm::box3)));
    const dm::box3 after = group->GetGlobalBoundingBox();
    CHECK(after.m_mins.x > before.m_mins.x + 99.f && after.m_mins.x < before.m_mins.x + 101.f);

    // content flags follow SetLeaf and Detach
    auto blended = MAKE_RC_OBJ_PTR(SceneGraphAnimation);
    group->SetLeaf(blended);
    CHECK((root->GetSubgraphContentFlags() & SceneContentFlags::Animations) != 0);
    graph->Detach(group);
    CHECK((root->GetSubgraphContentFlags() & SceneContentFlags::Animations) == 0);
}

// Pending changes, skipped refreshes, the deprecated dirty-flag shim, and the joint frame index of skinned meshes.
void test_pending_changes_and_joints()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    CHECK(graph->HasPendingStructureChanges());
    graph->Refresh(0);
    CHECK(!graph->HasPendingStructureChanges() && !graph->HasPendingTransformChanges());

    auto node = graph->Attach(graph->GetRootNode(), MakeNode("x"));
    CHECK(graph->HasPendingStructureChanges() && !graph->HasPendingTransformChanges());
    graph->Refresh(1);
    node->SetTranslation(dm::double3(1.0));
    CHECK(graph->HasPendingTransformChanges() && !graph->HasPendingStructureChanges());
    CHECK((node->GetDirtyFlags() & SceneGraphNode::DirtyFlags::SubgraphTransforms) != 0);
    graph->Refresh(2);
    // the previous-frame transforms are still pending for one more refresh
    CHECK(graph->HasPendingTransformChanges());
    graph->Refresh(3);
    CHECK(!graph->HasPendingTransformChanges());
    const uint64_t version = graph->GetWorld()->GetVersion();
    graph->Refresh(4);   // nothing changed: the phases do not run
    CHECK(graph->GetWorld()->GetVersion() == version);

    // a joint node that moves marks its skinned instance with the frame index
    auto skinned = factory->CreateSkinnedMeshInstance(factory, MakeMesh(factory, 1.f));
    auto skin = graph->AttachLeafNode(graph->GetRootNode(), skinned);
    auto joint = MakeNode("joint");
    joint->SetTranslation(dm::double3(0.0));
    joint->SetLeaf(MAKE_RC_OBJ_PTR(SkinnedMeshReference, skinned.Get()));
    graph->Attach(graph->GetRootNode(), joint);
    graph->Refresh(10);
    CHECK(skinned->GetLastUpdateFrameIndex() == 10);
    graph->Refresh(11);
    CHECK(skinned->GetLastUpdateFrameIndex() == 10);
    joint->SetTranslation(dm::double3(1.0, 0.0, 0.0));
    graph->Refresh(12);
    CHECK(skinned->GetLastUpdateFrameIndex() == 12);
    node->SetTranslation(dm::double3(2.0));   // another node, in another chunk
    graph->Refresh(13);
    CHECK(skinned->GetLastUpdateFrameIndex() == 12);
}

// Row hints: setters and getters find rows through the node's remembered place while its archetype moved no row;
// after swap-removes, archetype moves and re-attaches they fall back to the entity table and stay correct.
void test_row_hints()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    std::vector<dm::double3> expected;
    for (int i = 0; i < 700; i++)
    {
        auto node = MakeNode("n");
        node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
        if (i % 2)
            node->SetLeaf(factory->CreateMeshInstance(mesh));
        nodes.push_back(graph->Attach(graph->GetRootNode(), node));
        expected.push_back(dm::double3(double(i), 0.0, 0.0));
    }
    auto check = [&]()
    {
        for (size_t i = 0; i < nodes.size(); i++)
            CHECK(all(nodes[i]->GetTranslation() == expected[i]));
    };
    check();
    for (int round = 0; round < 4; round++)
    {
        // swap-removes in both archetypes: the remembered rows of the others are stale now
        for (size_t i = size_t(round); i < nodes.size(); i += 9)
        {
            graph->Detach(nodes[i]);
            if (round % 2)
                graph->Attach(graph->GetRootNode(), nodes[i]);   // back at another row
        }
        check();
        // archetype moves: a leaf added or removed
        for (size_t i = size_t(round); i < nodes.size(); i += 13)
            if (nodes[i]->GetGraph())
                nodes[i]->SetLeaf(i % 2 ? static_cast<SceneGraphLeaf*>(MAKE_RC_OBJ_PTR(PointLight))
                                        : factory->CreateMeshInstance(mesh).Get());
        for (size_t i = 0; i < nodes.size(); i++)
        {
            expected[i] = dm::double3(double(i), double(round + 1), 0.0);
            nodes[i]->SetTranslation(expected[i]);   // attached or not
        }
        check();
        graph->Refresh(uint32_t(round));
        CheckWorlds(graph);
    }
}

namespace
{
    // A leaf whose box changes by itself, as SPH particles' and GVDB volumes' do.
    NVRHI_CLASS_CLSID(GrowingLeaf, "5e2f2ae9-5107-49c3-b5c5-796cb61df8ff")
    class GrowingLeaf final : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(GrowingLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(GrowingLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(GrowingLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        float size = 1.f;
        dm::box3 GetLocalBoundingBox() override { return dm::box3(dm::float3(-size), dm::float3(size)); }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return MAKE_RC_OBJ_PTR(GrowingLeaf); }
    };

    // A graph whose RegisterLeaf reads the node and writes its transform, and throws for lights.
    class ProbingGraph final : public SceneGraph
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        uint32_t registered = 0;
        uint32_t nodeChecks = 0;

    protected:
        void RegisterLeaf(SceneGraphLeaf* leaf) override
        {
            if (query_cast<PointLight>(leaf))
                throw std::runtime_error("probe");
            SceneGraphNode* node = leaf->GetNode();
            CHECK(node && node->GetLeaf() == leaf);
            CHECK(GetEntity(node) != Entity::Null && GetNode(GetEntity(node)) == node);
            CHECK(node->GetGraph() == this);
            // a node with a transform takes a write; it must reach the row
            if (all(node->GetScaling() == dm::double3(2.0)))
                node->SetTranslation(dm::double3(7.0, 0.0, 0.0));
            ++nodeChecks;
            ++registered;
            SceneGraph::RegisterLeaf(leaf);
        }
    };
}

// The review fixes of step E3, one block each.
void test_review_fixes()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);

    // 1. A node moved to another graph does not serve the old graph's cached bounds, even at an equal epoch.
    {
        auto a = MAKE_RC_OBJ_PTR(SceneGraph);
        a->SetRootNode(MakeNode("root"));
        auto node = MakeNode("n");
        node->SetLeaf(factory->CreateMeshInstance(mesh));
        a->Attach(a->GetRootNode(), node);
        a->Refresh(0);
        const dm::box3 inA = node->GetGlobalBoundingBox();
        CHECK(inA.m_mins.x == -1.f);
        a->Detach(node);
        auto b = MAKE_RC_OBJ_PTR(SceneGraph);
        auto root = MakeNode("root");
        root->SetTranslation(dm::double3(10.0, 0.0, 0.0));
        b->SetRootNode(root);
        b->Attach(b->GetRootNode(), node);
        b->Refresh(0);
        CHECK(node->GetGlobalBoundingBox().m_mins.x == 9.f);
    }

    // 2. InvalidateContent makes the next Refresh run Bounds; writes through the world outside phases are not seen
    //    until something else makes the phases run (SceneGraph.h documents it).
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        IWorld* world = graph->GetWorld();
        graph->SetRootNode(MakeNode("root"));
        auto leaf = MAKE_RC_OBJ_PTR(GrowingLeaf);
        auto node = graph->AttachLeafNode(graph->GetRootNode(), leaf);
        auto other = graph->Attach(graph->GetRootNode(), MakeNode("other"));
        other->SetTranslation(dm::double3(0.0));
        graph->Refresh(0);
        CHECK(GetComponent<WorldBounds>(world, graph->GetEntity(node))->bounds.m_maxs.x == 1.f);
        leaf->size = 3.f;
        node->InvalidateContent();
        graph->Refresh(1);
        CHECK(GetComponent<WorldBounds>(world, graph->GetEntity(node))->bounds.m_maxs.x == 3.f);
        CHECK(GetComponent<ChunkWorldBounds>(world, graph->GetEntity(node))->bounds.m_maxs.x == 3.f);

        LocalTransform* local = GetComponentForWrite<LocalTransform>(world, graph->GetEntity(other));
        local->translation = dm::double3(5.0, 0.0, 0.0);
        graph->Refresh(2);   // skipped: nothing went through the API
        CHECK(other->GetLocalToWorldTransform().m_translation.x == 0.0);
        node->InvalidateContent();
        graph->Refresh(3);   // runs: the earlier write's chunk version is newer than propagation's last run
        CHECK(other->GetLocalToWorldTransform().m_translation.x == 5.0);
    }

    // 3 and 9. RegisterLeaf reads the node and writes its transform during the attach; an exception is caught at the
    //    observer, logged, and the rest of the batch still registers.
    {
        auto graph = MAKE_RC_OBJ_PTR(ProbingGraph);
        graph->SetRootNode(MakeNode("root"));
        auto top = MakeNode("top");
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        for (int i = 0; i < 20; i++)
        {
            auto node = MakeNode("n");
            node->SetScaling(dm::double3(i % 2 ? 2.0 : 1.0));
            node->SetLeaf(i == 5 ? static_cast<SceneGraphLeaf*>(MAKE_RC_OBJ_PTR(PointLight))
                                 : factory->CreateMeshInstance(mesh).Get());
            graph->Attach(top, node);
            nodes.push_back(node);
        }
        const uint32_t warnings = g_Warnings.load();
        graph->Attach(graph->GetRootNode(), top);
        CHECK(g_Warnings.load() == warnings + 1);   // the caught exception
        g_Warnings = warnings;
        CHECK(graph->registered == 19 && graph->GetMeshInstances().size() == 19 && graph->GetLights().empty());
        for (int i = 0; i < 20; i++)
        {
            if (i % 2 && i != 5)
                CHECK(nodes[size_t(i)]->GetTranslation().x == 7.0);
        }
    }

    // 4. A Refresh that cannot run its phases (from inside a phase) keeps the changes pending.
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MakeNode("root"));
        auto node = graph->Attach(graph->GetRootNode(), MakeNode("n"));
        graph->Refresh(0);
        node->SetTranslation(dm::double3(4.0, 0.0, 0.0));
        SceneGraph* raw = graph;
        SystemDesc desc{};
        desc.name = "reentrant";
        desc.phase = Phase::BeginFrame;
        desc.flags = SystemFlags::MainThread;
        nvrhi::AutoPtr<ISystem> system;
        CHECK(NVRHI_SUCCEEDED(CreateLambdaSystem(desc, [raw](ISystemContext*) { raw->Refresh(1); }, &system)));
        SystemId id = SystemId::Invalid;
        CHECK(NVRHI_SUCCEEDED(graph->GetScheduler()->AddSystem(system, id)));
        const uint32_t warnings = g_Warnings.load();
        CHECK(NVRHI_SUCCEEDED(graph->GetScheduler()->RunPhase(Phase::BeginFrame)));
        CHECK(g_Warnings.load() > warnings);
        g_Warnings = warnings;
        CHECK(graph->HasPendingTransformChanges());
        CHECK(node->GetLocalToWorldTransform().m_translation.x == 0.0);
        CHECK(NVRHI_SUCCEEDED(graph->GetScheduler()->RemoveSystem(id)));
        graph->Refresh(2);
        CHECK(node->GetLocalToWorldTransform().m_translation.x == 4.0);
    }

    // 8. One leaf on two nodes: a Detach of one removes one entry, as the old erase of the first entry did.
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MakeNode("root"));
        auto leaf = factory->CreateMeshInstance(mesh);
        auto a = graph->Attach(graph->GetRootNode(), MakeNode("a"));
        auto b = graph->Attach(graph->GetRootNode(), MakeNode("b"));
        a->SetLeaf(leaf);
        b->SetLeaf(leaf);
        CHECK(graph->GetMeshInstances().size() == 2);
        graph->Detach(a);
        CHECK(graph->GetMeshInstances().size() == 1 && graph->GetMeshInstances()[0].Get() == leaf.Get());
        graph->Attach(graph->GetRootNode(), a);
        CHECK(graph->GetMeshInstances().size() == 2);
        graph->Detach(a);
        graph->Detach(b);
        CHECK(graph->GetMeshInstances().empty());
    }

    // 10. A refused CreateEntities (the test seam: as out of memory) leaves the batches created before it attached and
    //     the nodes from it on in no graph, alive while referenced, with their leaves; the graph stays consistent.
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MakeNode("root"));
        graph->Refresh(0);
        auto top = MakeNode("top");                     // batch 0: no transform, no leaf
        auto group = MakeNode("group");                 // batch 1: a transform
        group->SetTranslation(dm::double3(1.0, 0.0, 0.0));
        graph->Attach(top, group);
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> leaves;
        for (int i = 0; i < 10; i++)                    // batch 2: transforms and leaves
        {
            auto node = MakeNode("leaf");
            node->SetTranslation(dm::double3(0.0, double(i), 0.0));
            node->SetLeaf(factory->CreateMeshInstance(mesh));
            graph->Attach(group, node);
            leaves.push_back(node);
        }
        WorldStats stats;
        graph->GetWorld()->GetStats(stats);
        const uint32_t before = stats.entityCount;
        SceneGraphStore::FailCreateAfter(1);
        const uint32_t warnings = g_Warnings.load();
        CHECK(!graph->Attach(graph->GetRootNode(), top));
        SceneGraphStore::FailCreateAfter(-1);
        CHECK(g_Warnings.load() > warnings);
        g_Warnings = warnings;
        graph->GetWorld()->GetStats(stats);
        CHECK(stats.entityCount == before + 1);         // only `top`
        CHECK(top->GetGraph() == graph.Get() && top->GetParent() == graph->GetRootNode());
        CHECK(!group->GetGraph() && all(group->GetTranslation() == dm::double3(1.0, 0.0, 0.0)));
        for (const auto& node : leaves)
            CHECK(!node->GetGraph() && node->GetLeaf() && node->GetLeaf()->GetNode() == node.Get());
        CHECK(graph->GetMeshInstances().empty());
        graph->Refresh(1);
        CheckWorlds(graph);
        // the abandoned nodes attach again as orphans
        graph->Attach(top, group);
        graph->Attach(group, leaves[0]);
        graph->Refresh(2);
        CHECK(graph->GetMeshInstances().size() == 1);
        CHECK(leaves[0]->GetLocalToWorldTransform().m_translation.x == 1.0);
        CheckWorlds(graph);
    }
}

// HasStaleWorldTransforms: set by every structural and transform call of the API (the list is SceneGraph.h's), cleared
// by the run of propagation that brings the world transforms up to date, and not set by anything else. Unlike the
// pending flags it does not stay for the frame after a move, and a propagation run by hand clears it.
void test_stale_world_transforms()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    auto source = MAKE_RC_OBJ_PTR(SceneGraph);   // another graph, to copy from
    CHECK(!graph->HasStaleWorldTransforms());    // no edit yet
    uint32_t frame = 0;

    auto expect = [](const char* what, bool condition, const char* problem)
    {
        if (!condition)
            throw std::runtime_error(std::string("stale_world_transforms: ") + what + ": " + problem + "\n");
    };
    auto settle = [&](const char* what)
    {
        graph->Refresh(frame++);
        graph->Refresh(frame++);   // the previous-frame transforms of a move in the frame before
        expect(what, !graph->HasPendingStructureChanges() && !graph->HasPendingTransformChanges(), "not settled");
        expect(what, !graph->HasStaleWorldTransforms(), "set before the call");
    };
    // `moves`: the call sets transforms, so the pending transform flag stays for the frame after the Refresh.
    auto edit = [&](const char* what, bool moves, auto&& call)
    {
        settle(what);
        call();
        expect(what, graph->HasStaleWorldTransforms(), "not set by the call");
        expect(what, graph->HasPendingTransformChanges() == moves, "the pending transform flag");
        graph->Refresh(frame++);
        expect(what, !graph->HasStaleWorldTransforms(), "not cleared by Refresh");
        expect(what, graph->HasPendingTransformChanges() == moves, "the pending transform flag of the frame after");
        if (graph->GetEntity(graph->GetRootNode()) != Entity::Null)   // the root that Detach leaves is in no graph
            CheckWorlds(graph);
        graph->Refresh(frame++);
        expect(what, !graph->HasStaleWorldTransforms() && !graph->HasPendingTransformChanges(), "set in a still frame");
    };
    auto noEdit = [&](const char* what, auto&& call)
    {
        settle(what);
        call();
        expect(what, !graph->HasStaleWorldTransforms(), "set by a call that is no edit");
    };

    const dm::double3 t(1.0, 2.0, 3.0);
    const dm::dquat r = dm::rotationQuat(dm::double3(0.1, 0.2, 0.3));
    const dm::double3 scale(1.5);

    // ---- structure: SetRootNode, Attach, AttachLeafNode, Detach
    edit("SetRootNode", false, [&]() { graph->SetRootNode(MakeNode("root")); });
    nvrhi::AutoPtr<SceneGraphNode> bare, moved, parent;
    edit("Attach", false, [&]() { bare = graph->Attach(graph->GetRootNode(), MakeNode("bare")); });
    auto withTransform = MakeNode("moved");
    withTransform->SetTranslation(t);
    edit("Attach with a transform", true, [&]() { moved = graph->Attach(graph->GetRootNode(), withTransform); });
    auto subtree = MakeNode("parent");
    auto subtreeChild = MakeNode("child");
    subtreeChild->SetRotation(r);
    graph->Attach(subtree, subtreeChild);
    edit("Attach of a subtree", true, [&]() { parent = graph->Attach(graph->GetRootNode(), subtree); });
    edit("Attach of a copy", true, [&]() { graph->Attach(bare, parent); });
    source->SetRootNode(MakeNode("source"));
    source->Refresh(0);
    edit("Attach of another graph's node", false, [&]() { graph->Attach(bare, source->GetRootNode()); });
    CHECK(!source->HasStaleWorldTransforms());
    nvrhi::AutoPtr<SceneGraphNode> lightNode;
    edit("AttachLeafNode", false, [&]()
    {
        lightNode = graph->AttachLeafNode(parent, MAKE_RC_OBJ_PTR(SpotLight));
    });
    edit("Detach", false, [&]() { graph->Detach(bare->GetFirstChild()); });
    edit("Detach of a leaf node", false, [&]() { graph->Detach(lightNode); });
    edit("Attach again", false, [&]() { graph->Attach(parent, lightNode); });

    // ---- structure: SetLeaf
    auto leafNode = graph->Attach(graph->GetRootNode(), MakeNode("leaf"));
    edit("SetLeaf, the first", false, [&]() { leafNode->SetLeaf(MAKE_RC_OBJ_PTR(GrowingLeaf)); });
    edit("SetLeaf, the same class", false, [&]() { leafNode->SetLeaf(MAKE_RC_OBJ_PTR(GrowingLeaf)); });
    edit("SetLeaf, another class", false, [&]()
    {
        leafNode->SetLeaf(factory->CreateMeshInstance(MakeMesh(factory, 1.f)));
    });
    edit("SetLeaf, none", false, [&]() { leafNode->SetLeaf(nullptr); });

    // ---- transforms: the four setters, each as the node's first and as a later one
    auto setters = [&](const char* name, auto&& set)
    {
        auto node = graph->Attach(parent, MakeNode("set"));
        edit((std::string(name) + ", the first").c_str(), true, [&]() { set(node.Get()); });
        edit((std::string(name) + ", again").c_str(), true, [&]() { set(node.Get()); });
    };
    setters("SetTransform", [&](SceneGraphNode* node) { node->SetTransform(&t, &r, &scale); });
    setters("SetTransform, no part", [&](SceneGraphNode* node) { node->SetTransform(nullptr, nullptr, nullptr); });
    setters("SetTranslation", [&](SceneGraphNode* node) { node->SetTranslation(t); });
    setters("SetRotation", [&](SceneGraphNode* node) { node->SetRotation(r); });
    setters("SetScaling", [&](SceneGraphNode* node) { node->SetScaling(scale); });

    // ---- transforms: the calls that go through the setters
    const Light* light = query_cast<Light>(lightNode->GetLeaf());
    CHECK(light);
    edit("Light::SetPosition", true, [&]() { light->SetPosition(t); });
    edit("Light::SetDirection", true, [&]() { light->SetDirection(dm::double3(0.0, -1.0, 0.5)); });
    edit("SceneGraphAnimationChannel::Apply", true, [&]()
    {
        auto sampler = MAKE_RC_OBJ_PTR(animation::Sampler);
        sampler->AddKeyframe({ 0.f, dm::float4(2.f, 0.f, 0.f, 0.f) });
        auto channel = MAKE_RC_OBJ_PTR(SceneGraphAnimationChannel, sampler.Get(), moved.Get(),
                                       AnimationAttribute::Translation);
        CHECK(channel->Apply(0.f));
    });
    edit("SceneGraphAnimation::Apply", true, [&]()
    {
        auto sampler = MAKE_RC_OBJ_PTR(animation::Sampler);
        sampler->AddKeyframe({ 0.f, dm::float4(0.f, 0.f, 0.f, 1.f) });
        auto animation = MAKE_RC_OBJ_PTR(SceneGraphAnimation);
        animation->AddChannel(MAKE_RC_OBJ_PTR(SceneGraphAnimationChannel, sampler.Get(), moved.Get(),
                                              AnimationAttribute::Rotation));
        (void)animation->Apply(0.f);   // its result does not tell whether the channels applied
    });
    edit("several edits before one Refresh", true, [&]()
    {
        moved->SetTranslation(dm::double3(4.0));
        graph->Attach(moved, MakeNode("below"));
        moved->SetTranslation(dm::double3(5.0));
    });

    // ---- what is no edit of this graph
    auto orphan = MakeNode("orphan");
    noEdit("getters", [&]()
    {
        (void)moved->GetLocalToWorldTransform();
        (void)graph->GetRootNode()->GetGlobalBoundingBox();
        (void)graph->GetRootNode()->GetSubgraphContentFlags();
        (void)moved->GetDirtyFlags();
        CHECK(graph->FindNode("/moved") == moved.Get());
    });
    noEdit("names, content and leaf properties", [&]()
    {
        moved->SetName("renamed");
        light->SetName("renamed light");
        lightNode->InvalidateContent();
        CHECK(lightNode->GetLeaf()->SetProperty("color", dm::float4(1.f)));
    });
    noEdit("nodes in no graph", [&]()
    {
        auto child = MakeNode("child");
        graph->Attach(orphan, child);
        orphan->SetTransform(&t, &r, &scale);
        child->SetTranslation(t);
        child->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
        graph->Detach(child);
    });
    noEdit("another graph", [&]()
    {
        source->GetRootNode()->SetTranslation(t);
        CHECK(source->HasStaleWorldTransforms());
    });
    noEdit("a write through the world", [&]()
    {
        IWorld* world = graph->GetWorld();
        LocalTransform* local = GetComponentForWrite<LocalTransform>(world, graph->GetEntity(moved));
        local->scaling = local->scaling;
    });
    noEdit("RunReadOnlyPhase", [&]() { CHECK(graph->RunReadOnlyPhase(Phase::RenderExtract) == nvrhi::FS_OK); });
    noEdit("SetJobSystem", [&]() { graph->SetJobSystem(MakeJobs(2)); });
    CheckWorlds(graph);

    // ---- calls that are refused: a structural one inside a phase sets nothing, a Refresh inside a phase clears nothing
    IScheduler* scheduler = graph->GetScheduler();
    uint32_t runs = 0;
    SystemDesc desc{};
    desc.name = "test.Refused";
    desc.phase = Phase::PreSimulation;
    desc.flags = SystemFlags::MainThread;
    nvrhi::AutoPtr<ISystem> refused;
    CHECK(CreateLambdaSystem(desc, [&](ISystemContext*) noexcept
    {
        ++runs;
        graph->Attach(graph->GetRootNode(), orphan);
        graph->Refresh(frame);
    }, &refused) == nvrhi::FS_OK);
    SystemId refusedId = SystemId::Invalid;
    CHECK(scheduler->AddSystem(refused, refusedId) == nvrhi::FS_OK);
    const uint32_t warnings = g_Warnings.load();
    noEdit("Attach inside a phase", [&]() { CHECK(scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK); });
    CHECK(runs == 1 && !orphan->GetGraph());
    edit("Refresh inside a phase", true, [&]()
    {
        moved->SetTranslation(dm::double3(6.0));
        CHECK(scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
    });
    CHECK(runs == 2 && g_Warnings.load() > warnings);
    g_Warnings = warnings;
    CHECK(scheduler->RemoveSystem(refusedId) == nvrhi::FS_OK);

    // ---- propagation run by hand clears it, and leaves the pending flags to the next Refresh, which still runs
    settle("by hand");
    moved->SetTranslation(dm::double3(7.0));
    graph->Attach(moved, MakeNode("below"));
    CHECK(scheduler->RunPhase(Phase::Bounds) == nvrhi::FS_OK);
    CHECK(graph->HasStaleWorldTransforms());
    CHECK(scheduler->RunPhase(Phase::TransformPropagation) == nvrhi::FS_OK);
    CHECK(!graph->HasStaleWorldTransforms());
    CHECK(graph->HasPendingStructureChanges() && graph->HasPendingTransformChanges());
    CheckWorlds(graph);
    const uint64_t version = graph->GetWorld()->GetVersion();
    graph->Refresh(frame++);
    CHECK(graph->GetWorld()->GetVersion() != version);
    CHECK(!graph->HasPendingStructureChanges() && graph->HasPendingTransformChanges());
    CHECK(!graph->HasStaleWorldTransforms());
    moved->SetTranslation(dm::double3(8.0));   // an edit after it is stale again
    CHECK(graph->HasStaleWorldTransforms());
    graph->Refresh(frame++);
    CheckWorlds(graph);

    // ---- the root: Detach leaves a root in no graph, SetRootNode replaces one
    edit("SetRootNode, another", false, [&]() { graph->SetRootNode(MakeNode("other root")); });
    edit("Detach of the root", false, [&]() { graph->Detach(graph->GetRootNode()); });
    nvrhi::AutoPtr<SceneGraphNode> newRoot = MakeNode("new root");
    newRoot->SetTranslation(t);
    edit("Attach as the root", true, [&]() { graph->Attach(nullptr, newRoot); });
}

// The simulation components and the fixed-step resource register as what domains need them to be (Relocatable), and
// their GUIDs are no other type's of a graph's world, with every leaf class of SceneGraph.h attached.
void test_simulation_components()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    auto skinned = factory->CreateSkinnedMeshInstance(factory, MakeMesh(factory, 1.f));
    SceneGraphNode* root = graph->GetRootNode();
    graph->AttachLeafNode(root, factory->CreateMeshInstance(MakeMesh(factory, 1.f)));
    graph->AttachLeafNode(root, skinned);
    graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(SkinnedMeshReference, skinned.Get()));
    graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(PerspectiveCamera));
    graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(OrthographicCamera));
    graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(DirectionalLight));
    graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(SpotLight));
    graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(PointLight));
    graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(SceneGraphAnimation));
    RegisterComponent<DynamicBoundsLeaf>(graph->GetWorld());   // the tag of leaves with dynamic bounds
    nvrhi::AutoPtr<IWorld> world;
    WorldDesc worldDesc;
    worldDesc.debugName = "test_scene_graph_store";
    worldDesc.validation = true;
    CHECK(CreateWorld(worldDesc, &world) == nvrhi::FS_OK);
    const uint32_t typeCount = graph->GetWorld()->GetComponentTypeCount();
    CHECK(typeCount >= 27);   // 14 of the store, FrameTime, 11 leaf-class tags, DynamicBoundsLeaf
    auto ownGuid = [&](const ComponentTypeDesc& desc)
    {
        for (uint32_t i = 0; i < typeCount; ++i)
        {
            const ComponentTypeDesc* other = graph->GetWorld()->GetComponentTypeDesc(ComponentId(i));
            CHECK(other);
            CHECK(other->uuid != desc.uuid || std::strcmp(other->name, desc.name) == 0);
        }
    };
    std::vector<ComponentId> ids;
    auto component = [&](const ComponentTypeDesc& desc, uint32_t size, ComponentFlags flags)
    {
        ownGuid(desc);
        ComponentId id = ComponentId::Invalid;
        CHECK(world->FindComponent(desc.uuid) == ComponentId::Invalid);
        CHECK(world->RegisterComponent(desc, id) == nvrhi::FS_OK);
        CHECK(desc.size == size && desc.flags == flags);
        ids.push_back(id);
    };
    component(ComponentTraits<SimulationDriven>::Desc(), 0, ComponentFlags::Tag | ComponentFlags::Relocatable);
    component(ComponentTraits<SimPose>::Desc(), 56, ComponentFlags::Relocatable);
    component(ComponentTraits<SimPosePrevious>::Desc(), 56, ComponentFlags::Relocatable);
    component(ComponentTraits<SimScale>::Desc(), 24, ComponentFlags::Relocatable);
    component(ComponentTraits<SimBodyRef>::Desc(), 8, ComponentFlags::Relocatable);
    component(ComponentTraits<SimTeleport>::Desc(), 88, ComponentFlags::Enableable | ComponentFlags::Relocatable);

    // a row bound to no domain yet
    EntityCreateDesc create{};
    create.components = ids.data();
    create.componentCount = uint32_t(ids.size());
    create.entityCount = 1;
    Entity entity = Entity::Null;
    CHECK(world->CreateEntities(create, &entity) == nvrhi::FS_OK);
    CHECK(GetComponent<SimBodyRef>(world, entity)->domain == uint32_t(SimulationDomainId::Invalid));
    CHECK(world->SetEnabled(entity, ids.back(), false) == nvrhi::FS_OK && !world->IsEnabled(entity, ids.back()));

    // NoSerialize: what the scheduler publishes is not part of the world's hash
    ComponentId id = ComponentId::Invalid;
    ownGuid(ComponentTraits<FixedStepState>::Desc());
    CHECK(world->FindComponent(ComponentTraits<FixedStepState>::Desc().uuid) == ComponentId::Invalid);
    CHECK(world->RegisterResource(ComponentTraits<FixedStepState>::Desc(), nullptr, id) == nvrhi::FS_OK);
    const uint64_t hash = world->ComputeHash();
    FixedStepState* state = GetResource<FixedStepState>(world);
    CHECK(state);
    for (const FixedStepEntry& entry : state->domains)
        CHECK(entry.accumulator == 0.0 && entry.fixedTimeStep == 0.0 && entry.stepIndex == 0 &&
              entry.stepsPlanned == 0 && entry.stepsTaken == 0 && entry.stepsDropped == 0);
    state->domains[c_MaxSimulationDomains - 1].stepIndex = 7;
    CHECK(world->ComputeHash() == hash);
}

// Steady state: moves, refreshes and reads allocate nothing.
void test_steady_state_allocations()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);
    auto jobs = MakeJobs(4);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetJobSystem(jobs);
    Rng rng(5);
    graph->SetRootNode(MakeNode("root"));
    std::vector<SceneGraphNode*> nodes;
    SceneGraphNode* parent = graph->GetRootNode();
    for (int i = 0; i < 3000; i++)
    {
        auto node = MakeNode("n");
        RandomMove(rng, node);
        if (i % 2)
            node->SetLeaf(factory->CreateMeshInstance(mesh));
        // a flat part, and a chain past the last depth bucket
        SceneGraphNode* attached = graph->Attach(i < 2900 ? graph->GetRootNode() : parent, node);
        if (i >= 2900)
            parent = attached;
        nodes.push_back(attached);
    }
    auto frame = [&](uint32_t index)
    {
        for (int i = 0; i < 300; i++)
            RandomMove(rng, nodes[rng.Below(uint32_t(nodes.size()))]);
        RandomMove(rng, nodes[2900]);
        graph->Refresh(index);
        float sum = 0.f;
        for (int i = 0; i < 100; i++)
            sum += nodes[rng.Below(uint32_t(nodes.size()))]->GetLocalToWorldTransformFloat().m_translation.x;
        sum += graph->GetRootNode()->GetGlobalBoundingBox().m_mins.x;
        return sum;
    };
    for (uint32_t i = 0; i < 4; i++)
        (void)frame(i);
    const uint64_t newBefore = g_NewCount.load();
    const uint64_t heapBefore = ecs::GetHeapAllocationCount();
    float sum = 0.f;
    for (uint32_t i = 4; i < 12; i++)
        sum += frame(i);
    const uint64_t newCalls = g_NewCount.load() - newBefore;
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
    std::printf("steady state: %llu operator new, %llu storage blocks (checksum %g)\n", (unsigned long long)newCalls,
                (unsigned long long)heapBlocks, double(sum));
    CHECK(newCalls == 0);
    CHECK(heapBlocks == 0);

    // the counter sees an allocation
    std::vector<int> probe(1000);
    CHECK(g_NewCount.load() > newBefore && probe.size() == 1000);
}

// The subgraph values Refresh computes (incremental and full passes, the deep bucket, rows that change archetype on
// their first transform, InvalidateContent, attach and detach) against a brute-force walk: each node's leaf box in
// world space and content flags, united over its subtree; and the links the node objects keep. Equal as floats (the
// sign of a zero that ties may differ: the union is order-independent).
namespace
{
    struct Expected
    {
        dm::box3 bounds;
        SceneContentFlags leaf;
        SceneContentFlags subgraph;
    };

    Expected BruteForce(SceneGraphNode* node, std::unordered_map<const SceneGraphNode*, Expected>& out)
    {
        Expected e{ dm::box3::empty(), SceneContentFlags::None, SceneContentFlags::None };
        if (SceneGraphLeaf* leaf = node->GetLeaf())
        {
            const dm::box3 local = leaf->GetLocalBoundingBox();
            if (!local.isempty())
                e.bounds = local * node->GetLocalToWorldTransformFloat();
            e.leaf = leaf->GetContentFlags();
        }
        e.subgraph = e.leaf;
        for (SceneGraphNode* child = node->GetFirstChild(); child; child = child->GetNextSibling())
        {
            const Expected c = BruteForce(child, out);
            e.bounds |= c.bounds;
            e.subgraph |= c.subgraph;
        }
        out[node] = e;
        return e;
    }

    bool SameBox(const dm::box3& a, const dm::box3& b)
    {
        return all(a.m_mins == b.m_mins) && all(a.m_maxs == b.m_maxs);   // as floats: -0 equals +0
    }

    // Every attached node's getters against the brute force walk; returns the expected values.
    std::unordered_map<const SceneGraphNode*, Expected> CheckAgainstBruteForce(SceneGraph* graph)
    {
        std::unordered_map<const SceneGraphNode*, Expected> expected;
        BruteForce(graph->GetRootNode(), expected);
        size_t visited = 0;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
        {
            SceneGraphNode* node = walker.Get();
            const Expected& e = expected[node];
            CHECK(SameBox(node->GetGlobalBoundingBox(), e.bounds));
            CHECK(node->GetLeafContentFlags() == e.leaf);
            CHECK(node->GetSubgraphContentFlags() == e.subgraph);
            ++visited;
        }
        CHECK(visited == expected.size());
        return expected;
    }
}

void test_subgraph_refresh()
{
    // every bucket and the marks on the job system (the serial path below 4096 rows is what the other cases run)
    SceneGraphStore::SetSerialSubgraphRows(1);
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    {
        // A group without a transform or leaf under the root, and the root's other children, transformed, without leaves
        // or children (one chunk, taken whole from its summary). The group's first transform moves its row into that
        // chunk: the world's structure changed without the epoch, and the next pass must not take the chunk whole.
        auto small = MAKE_RC_OBJ_PTR(SceneGraph);
        small->SetRootNode(MakeNode("root"));
        auto plain = MakeNode("plain");
        small->Attach(small->GetRootNode(), plain);
        auto mesh = MakeMesh(factory, 1.f);
        for (int i = 0; i < 20; ++i)
        {
            auto child = MakeNode("c");
            child->SetTranslation(dm::double3(double(i), 0.0, 0.0));
            child->SetLeaf(factory->CreateMeshInstance(mesh));
            small->Attach(plain, child);
            auto other = MakeNode("o");
            other->SetTranslation(dm::double3(0.0, double(i), 0.0));
            small->Attach(small->GetRootNode(), other);
        }
        small->Refresh(1);
        plain->GetFirstChild()->SetTranslation(dm::double3(-1.0, 0.0, 0.0));   // an incremental pass
        small->Refresh(2);
        plain->SetTranslation(dm::double3(100.0, 0.0, 0.0));
        small->Refresh(3);
        std::unordered_map<const SceneGraphNode*, Expected> expected;
        BruteForce(small->GetRootNode(), expected);
        CHECK(SameBox(plain->GetGlobalBoundingBox(), expected[plain.Get()].bounds));
        CHECK(SameBox(small->GetRootNode()->GetGlobalBoundingBox(), expected[small->GetRootNode()].bounds));
        CHECK(plain->GetGlobalBoundingBox().m_mins.x > 97.f);   // the first child at x = -1, box from -1 to +1
    }

    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    Rng rng(77);
    std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
    for (int i = 0; i < 4; ++i)
        meshes.push_back(MakeMesh(factory, 0.25f + 0.5f * float(i)));
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> all;
    auto newNode = [&](bool transform)
    {
        auto node = MakeNode("n");
        if (transform)
            RandomMove(rng, node);
        const uint32_t pick = rng.Below(5);
        if (pick < 3)
            node->SetLeaf(factory->CreateMeshInstance(meshes[rng.Below(4)]));
        else if (pick == 3)
            node->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
        all.push_back(node);
        return node;
    };
    // a wide part under the root, nested groups, and a chain deeper than the depth buckets (c_MaxHierarchyDepth)
    for (int i = 0; i < 600; ++i)
        graph->Attach(graph->GetRootNode(), newNode(true));
    for (int g = 0; g < 20; ++g)
    {
        auto group = newNode(g % 2 == 0);
        graph->Attach(all[rng.Below(uint32_t(all.size()))], group);
        for (int i = 0; i < 30; ++i)
            graph->Attach(i % 3 ? group.Get() : all[rng.Below(uint32_t(all.size()))].Get(), newNode(i % 4 != 0));
    }
    SceneGraphNode* tip = graph->GetRootNode();
    for (int d = 0; d < 90; ++d)
    {
        auto node = newNode(d % 5 != 0);
        graph->Attach(tip, node);
        if (d % 7 == 0)
            graph->Attach(tip, newNode(true));   // a sibling in the chain
        tip = node;
    }
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> detached;

    auto attachedNodes = [&]()
    {
        std::vector<SceneGraphNode*> nodes;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
            nodes.push_back(walker.Get());
        return nodes;
    };

    std::unordered_map<const SceneGraphNode*, Expected> previous = CheckAgainstBruteForce(graph);
    for (uint32_t frame = 1; frame <= 48; ++frame)
    {
        auto nodes = attachedNodes();
        // moves every frame (incremental passes); a node's first transform moves its row to another archetype
        for (uint32_t k = 0; k < uint32_t(nodes.size()) / 10; ++k)
            RandomMove(rng, nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]);
        bool structural = false;
        if (frame % 5 == 1)
        {
            // SetLeaf: a mesh, a light, or none
            SceneGraphNode* node = nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)];
            const uint32_t pick = rng.Below(3);
            node->SetLeaf(pick == 0 ? nvrhi::AutoPtr<SceneGraphLeaf>(factory->CreateMeshInstance(meshes[rng.Below(4)]))
                          : pick == 1 ? nvrhi::AutoPtr<SceneGraphLeaf>(MAKE_RC_OBJ_PTR(PointLight)) : nullptr);
            structural = true;
        }
        if (frame % 3 == 0)
        {
            structural = true;
            // edits that make the next pass a full one
            switch (frame % 4)
            {
            case 0:
                graph->Attach(nodes[rng.Below(uint32_t(nodes.size()))], newNode(true));
                break;
            case 1:
                if (!detached.empty())
                {
                    graph->Attach(nodes[rng.Below(uint32_t(nodes.size()))], detached.back());
                    detached.pop_back();
                }
                else
                    detached.push_back(graph->Detach(nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]));
                break;
            case 2:
            {
                // a mesh's box edited in place, then InvalidateContent on one instance of it (the others keep theirs)
                meshes[0]->objectSpaceBounds = dm::box3(dm::float3(-3.f), dm::float3(1.f + float(frame)));
                for (SceneGraphNode* node : nodes)
                {
                    if (auto instance = query_cast<MeshInstance>(node->GetLeaf()); instance &&
                        instance->GetMesh() == meshes[0].Get())
                        node->InvalidateContent();
                }
                break;
            }
            default:
                detached.push_back(graph->Detach(nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]));
                break;
            }
        }

        // Between the changes and Refresh: after a structural change the getters compute on demand from the rows
        // (equal to the brute force now: the new rows' transforms are their records'); with moves alone they still
        // give the last Refresh's values.
        if (structural)
            CheckAgainstBruteForce(graph);
        else
        {
            for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
            {
                const Expected& e = previous[walker.Get()];
                CHECK(SameBox(walker->GetGlobalBoundingBox(), e.bounds));
                CHECK(walker->GetSubgraphContentFlags() == e.subgraph);
            }
        }
        // the Bounds phase run outside Refresh (the next pass must be a full one), a read-only phase, and a pass whose
        // storage cannot be allocated (the getters fall back to the on-demand caches)
        if (frame % 7 == 3)
            CHECK(graph->GetScheduler()->RunPhase(Phase::Bounds) == nvrhi::FS_OK);
        if (frame % 7 == 5)
            CHECK(graph->RunReadOnlyPhase(Phase::RenderExtract) == nvrhi::FS_OK);
        if (frame % 11 == 6)
            SceneGraphStore::FailSubgraphAfter(0);
        graph->Refresh(frame);
        SceneGraphStore::FailSubgraphAfter(-1);

        // eight readers at once, then the brute force
        nodes = attachedNodes();
        std::vector<std::vector<dm::box3>> results(8);
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t)
        {
            threads.emplace_back([&, t]()
            {
                for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
                    results[size_t(t)].push_back(walker->GetGlobalBoundingBox());
            });
        }
        for (auto& thread : threads)
            thread.join();
        std::unordered_map<const SceneGraphNode*, Expected> expected;
        BruteForce(graph->GetRootNode(), expected);
        CHECK(expected.size() == nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            SceneGraphNode* node = nodes[i];
            const Expected& e = expected[node];
            CHECK(SameBox(node->GetGlobalBoundingBox(), e.bounds));
            CHECK(node->GetLeafContentFlags() == e.leaf);
            CHECK(node->GetSubgraphContentFlags() == e.subgraph);
            for (int t = 0; t < 8; ++t)
                CHECK(SameBits(&results[size_t(t)][i], &results[0][i], sizeof(dm::box3)));
            for (SceneGraphNode* child = node->GetFirstChild(); child; child = child->GetNextSibling())
                CHECK(child->GetParent() == node);
        }
        previous = std::move(expected);
    }
    SceneGraphStore::SetSerialSubgraphRows(4096);
    CHECK(g_Warnings.load() == 0);
}

// Structural changes inside Refresh's phases: commands that a user system in TransformPropagation records (played back
// before the Bounds phase: rows move to new archetypes and chunks), and an exclusive system in the Bounds phase that
// moves rows immediately. The pass sees the structure change and redoes itself as a full pass; every node's values
// equal the brute force.
namespace
{
    struct Tagged { };
    DONUT_ECS_TAG(Tagged, "5a8e2f41-7c3b-4e5d-9a1f-0b6c2d8e4f73");
}

void test_phase_structure()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    Rng rng(91);
    auto mesh = MakeMesh(factory, 0.5f);
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 6000; ++i)
    {
        auto node = MakeNode("n");
        RandomMove(rng, node);
        node->SetLeaf(factory->CreateMeshInstance(mesh));
        graph->Attach(i % 4 || nodes.empty() ? graph->GetRootNode() : nodes[rng.Below(uint32_t(nodes.size()))].Get(), node);
        nodes.push_back(node);
    }
    graph->Refresh(1);
    CheckAgainstBruteForce(graph);

    IWorld* world = graph->GetWorld();
    const ComponentId tag = RegisterComponent<Tagged>(world);
    std::vector<Entity> entities;   // the rows to toggle this frame
    std::vector<Entity> adds, removes;
    std::unordered_map<uint64_t, bool> tagged;
    bool commands = false;
    bool exclusive = false;
    uint32_t failures = 0;
    SystemDesc recorderDesc{};
    recorderDesc.name = "test.StructureRecorder";
    recorderDesc.phase = Phase::TransformPropagation;
    recorderDesc.flags = SystemFlags::MainThread;
    nvrhi::AutoPtr<ISystem> recorder;
    CHECK(CreateLambdaSystem(recorderDesc, [&](ISystemContext* context) noexcept
    {
        if (commands)
        {
            adds.clear();
            removes.clear();
            for (Entity entity : entities)
            {
                bool& on = tagged[uint64_t(entity)];
                (on ? removes : adds).push_back(entity);
                on = !on;
            }
            if (!adds.empty())
                context->GetCommandBuffer()->AddComponents(adds.data(), uint32_t(adds.size()), tag, nullptr, 0);
            if (!removes.empty())
                context->GetCommandBuffer()->RemoveComponents(removes.data(), uint32_t(removes.size()), tag, 0);
        }
        return nvrhi::FS_OK;
    }, &recorder) == nvrhi::FS_OK);
    SystemDesc exclusiveDesc{};
    exclusiveDesc.name = "test.StructureExclusive";
    exclusiveDesc.phase = Phase::Bounds;
    exclusiveDesc.flags = SystemFlags::Exclusive;
    const char* const afterBounds[] = { "SceneGraph.Bounds" };   // after the marks and summaries of the pass
    exclusiveDesc.runAfter = afterBounds;
    exclusiveDesc.runAfterCount = 1;
    nvrhi::AutoPtr<ISystem> mover;
    CHECK(CreateLambdaSystem(exclusiveDesc, [&](ISystemContext*) noexcept
    {
        if (exclusive)
        {
            for (Entity entity : entities)
            {
                bool& on = tagged[uint64_t(entity)];
                failures += (on ? world->RemoveComponent(entity, tag) : world->AddComponent(entity, tag, nullptr)) != nvrhi::FS_OK;
                on = !on;
            }
        }
        return nvrhi::FS_OK;
    }, &mover) == nvrhi::FS_OK);
    SystemId recorderId = SystemId::Invalid, moverId = SystemId::Invalid;
    CHECK(graph->GetScheduler()->AddSystem(recorder, recorderId) == nvrhi::FS_OK);
    CHECK(graph->GetScheduler()->AddSystem(mover, moverId) == nvrhi::FS_OK);

    const uint64_t redosBefore = SceneGraphStore::GetSubgraphRedoCount();
    for (uint32_t frame = 2; frame <= 9; ++frame)
    {
        // a third of the nodes move; the systems move a quarter of the rows to the tagged archetypes and back
        for (size_t i = frame % 3; i < nodes.size(); i += 3)
            RandomMove(rng, nodes[i]);
        entities.clear();
        for (size_t i = frame % 4; i < nodes.size(); i += 4)
            entities.push_back(graph->GetEntity(nodes[i]));
        commands = frame % 2 == 0;
        exclusive = frame % 2 == 1;
        graph->Refresh(frame);
        CheckAgainstBruteForce(graph);
    }
    commands = exclusive = false;
    CHECK(failures == 0);
    CHECK(SceneGraphStore::GetSubgraphRedoCount() > redosBefore);
    CHECK(graph->GetScheduler()->RemoveSystem(recorderId) == nvrhi::FS_OK);
    CHECK(graph->GetScheduler()->RemoveSystem(moverId) == nvrhi::FS_OK);
    CHECK(g_Warnings.load() == 0);
}

// NaN boxes: the same infinite rule in a full and an incremental pass, in a summary chunk and row by row.
void test_subgraph_nan()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    auto mesh = MakeMesh(factory, 0.5f);
    auto nanMesh = MakeMesh(factory, 0.5f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    nanMesh->objectSpaceBounds = dm::box3(dm::float3(nan), dm::float3(nan));
    auto group = MakeNode("group");
    graph->Attach(graph->GetRootNode(), group);
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 40; ++i)
    {
        auto node = MakeNode("n");
        node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
        node->SetLeaf(factory->CreateMeshInstance(i == 13 ? nanMesh : mesh));
        graph->Attach(i < 20 ? group.Get() : graph->GetRootNode(), node);   // the NaN under the group
        nodes.push_back(node);
    }
    graph->Refresh(1);   // full
    const dm::box3 full = graph->GetRootNode()->GetGlobalBoundingBox();
    const dm::box3 fullGroup = group->GetGlobalBoundingBox();
    nodes[3]->SetTranslation(dm::double3(3.0, 1.0, 0.0));
    nodes[30]->SetTranslation(dm::double3(30.0, 1.0, 0.0));
    graph->Refresh(2);   // incremental
    const dm::box3 incremental = graph->GetRootNode()->GetGlobalBoundingBox();
    const float infinity = std::numeric_limits<float>::infinity();
    for (const dm::box3* box : { &full, &fullGroup, &incremental })
    {
        CHECK(box->m_mins.x == -infinity && box->m_maxs.x == infinity);
        CHECK(box->m_mins.y == -infinity && box->m_maxs.z == infinity);
    }
    CHECK(SameBits(&full, &incremental, sizeof(full)));
    CHECK(SameBits(&fullGroup, &group->GetGlobalBoundingBox(), sizeof(fullGroup)));
}

// The walker against an independent pre-order (recursion over the node links), whole and with subtrees skipped, and
// in lockstep with a walk over the node getters while the graph changes the way the old graph allowed: first
// transforms (rows move to other archetypes), attaches under the current node, detaches of nodes the walk has left.
namespace
{
    void PreOrder(SceneGraphNode* node, std::vector<SceneGraphNode*>& out)
    {
        out.push_back(node);
        for (SceneGraphNode* child = node->GetFirstChild(); child; child = child->GetNextSibling())
            PreOrder(child, out);
    }

    // The walker's algorithm over the node getters (GetFirstChild, GetNextSibling, GetParent), as the old graph ran it.
    SceneGraphNode* ReferenceNext(SceneGraphNode* current, SceneGraphNode* scope, bool allowChildren)
    {
        if (allowChildren)
        {
            if (SceneGraphNode* child = current->GetFirstChild())
                return child;
        }
        while (current)
        {
            if (current == scope)
                return nullptr;
            if (SceneGraphNode* sibling = current->GetNextSibling())
                return sibling;
            current = current->GetParent();
        }
        return nullptr;
    }
}

void test_walker_order()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(MakeNode("root"));
    Rng rng(123);
    auto mesh = MakeMesh(factory, 0.5f);
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> all;
    for (int i = 0; i < 3000; ++i)
    {
        auto node = MakeNode("n");
        if (rng.Below(2))
            RandomMove(rng, node);
        if (rng.Below(3))
            node->SetLeaf(factory->CreateMeshInstance(mesh));
        SceneGraphNode* parent = all.empty() || rng.Below(5) == 0 ? graph->GetRootNode()
                                 : all[rng.Below(uint32_t(all.size()))].Get();
        graph->Attach(parent, node);
        all.push_back(node);
    }
    graph->Refresh(1);

    // the whole walk and walks of subtrees against the recursion
    for (int k = 0; k < 20; ++k)
    {
        SceneGraphNode* scope = k == 0 ? graph->GetRootNode() : all[rng.Below(uint32_t(all.size()))].Get();
        std::vector<SceneGraphNode*> expected, actual;
        PreOrder(scope, expected);
        for (SceneGraphWalker walker(scope); walker; walker.Next(true))
            actual.push_back(walker.Get());
        CHECK(expected == actual);
    }
    // subtrees skipped at random, in lockstep with the reference
    {
        SceneGraphWalker walker(graph->GetRootNode());
        SceneGraphNode* reference = graph->GetRootNode();
        while (walker)
        {
            CHECK(walker.Get() == reference);
            const bool allow = rng.Below(4) != 0;
            reference = ReferenceNext(reference, graph->GetRootNode(), allow);
            walker.Next(allow);
        }
        CHECK(reference == nullptr);
    }
    // the graph changes during the walk; both walks see the same structure at each step
    {
        SceneGraphWalker walker(graph->GetRootNode());
        SceneGraphNode* reference = graph->GetRootNode();
        std::vector<SceneGraphNode*> left;   // visited nodes that are not ancestors of the current one
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> kept;   // the detached subtrees
        uint32_t steps = 0, attached = 0, detachedCount = 0;
        while (walker)
        {
            CHECK(walker.Get() == reference);
            SceneGraphNode* current = walker.Get();
            const uint32_t pick = rng.Below(10);
            if (pick == 0)
                current->SetTranslation(dm::double3(rng.Range(-1, 1), 0.0, 0.0));   // maybe a first transform
            else if (pick == 1)
            {
                graph->Attach(current, MakeNode("new"));   // seen at the next step, as before
                ++attached;
            }
            else if (pick == 2 && !left.empty())
            {
                // a visited leaf node (childless when visited): still attached unless a detached subtree took it
                const size_t index = rng.Below(uint32_t(left.size()));
                SceneGraphNode* victim = left[index];
                bool ancestor = false;
                for (SceneGraphNode* up = current; up; up = up->GetParent())
                    ancestor = ancestor || up == victim;
                if (!ancestor && victim->GetGraph() == graph.Get() && victim != graph->GetRootNode())
                {
                    kept.push_back(graph->Detach(victim));
                    ++detachedCount;
                }
            }
            if (current->GetFirstChild() == nullptr)
                left.push_back(current);
            reference = ReferenceNext(reference, graph->GetRootNode(), true);
            walker.Next(true);
            ++steps;
        }
        CHECK(reference == nullptr);
        CHECK(attached > 0 && detachedCount > 0);
    }
    graph->Refresh(2);
}

// ------------------------------------------------------- the Bounds path for chunks without children (E7.4 remedy 4)

namespace
{
    // SceneGraph::m_Store (an explicit instantiation may name a private member), for the store's counters.
    template <typename Tag, typename Tag::Type Member>
    struct PrivateMember
    {
        friend typename Tag::Type GetMember(Tag) { return Member; }
    };
    struct StoreTag
    {
        using Type = SceneGraphStore* SceneGraph::*;
        friend Type GetMember(StoreTag);
    };
    template struct PrivateMember<StoreTag, &SceneGraph::m_Store>;
    const SceneGraphStore::Counters& CountersOf(SceneGraph* graph)
    {
        return (graph->*GetMember(StoreTag()))->GetCounters();
    }
}

// SceneGraph::SetChildlessBoundsPath is off by default and read at each Bounds run. Two graphs built and moved alike
// (leaves under the root, groups of leaves, leaves with children), the second refreshed with it on: every node's
// subgraph box and content equal, and equal to the brute force, over incremental passes and the full passes of an
// attach and of InvalidateContent; the path is taken only by the second graph.
void test_childless_bounds_path()
{
    // Stage T1: on in the switches-on runs (tests/src/e7_switches.cpp); the first graph is refreshed with it off
    // explicitly, and the value found is put back at the end.
    const bool childlessFound = SceneGraph::GetChildlessBoundsPath();
    CHECK(childlessFound == donut::tests::E7SwitchRequested("childless"));
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
    for (int i = 0; i < 3; ++i)
        meshes.push_back(MakeMesh(factory, 0.5f + float(i)));
    struct Built
    {
        nvrhi::AutoPtr<SceneGraph> graph;
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    };
    auto build = [&]()
    {
        Built b;
        b.graph = MAKE_RC_OBJ_PTR(SceneGraph);
        b.graph->SetRootNode(MakeNode("root"));
        Rng rng(91);
        auto add = [&](SceneGraphNode* parent, uint32_t i)
        {
            auto node = MakeNode("n");
            RandomMove(rng, node);
            node->SetLeaf(factory->CreateMeshInstance(meshes[i % 3]));
            b.graph->Attach(parent, node);
            b.nodes.push_back(node);
            return node.Get();
        };
        for (uint32_t i = 0; i < 400; ++i)
            add(b.graph->GetRootNode(), i);
        for (uint32_t g = 0; g < 8; ++g)
        {
            auto group = MakeNode("group");
            RandomMove(rng, group);
            b.graph->Attach(b.graph->GetRootNode(), group);
            b.nodes.push_back(group);
            for (uint32_t i = 0; i < 40; ++i)
                add(group, i);
        }
        for (uint32_t i = 0; i < 10; ++i)
        {
            SceneGraphNode* parent = add(b.graph->GetRootNode(), i);
            for (uint32_t k = 0; k < 3; ++k)
                add(parent, k);
        }
        return b;
    };
    Built built[2] = { build(), build() };
    auto refresh = [&](uint32_t frame)
    {
        SceneGraph::SetChildlessBoundsPath(false);
        built[0].graph->Refresh(frame);
        SceneGraph::SetChildlessBoundsPath(true);
        built[1].graph->Refresh(frame);
        SceneGraph::SetChildlessBoundsPath(false);
    };
    auto compare = [&]()
    {
        CheckAgainstBruteForce(built[1].graph);
        SceneGraphWalker a(built[0].graph->GetRootNode());
        SceneGraphWalker b(built[1].graph->GetRootNode());
        uint32_t nodes = 0;
        for (; a && b; a.Next(true), b.Next(true), ++nodes)
        {
            const dm::box3 boxA = a->GetGlobalBoundingBox();
            const dm::box3 boxB = b->GetGlobalBoundingBox();
            CHECK(SameBits(&boxA, &boxB, sizeof(dm::box3)));
            CHECK(a->GetSubgraphContentFlags() == b->GetSubgraphContentFlags());
        }
        CHECK(!a && !b && nodes > 700);
    };
    refresh(1);
    compare();
    for (uint32_t frame = 2; frame <= 12; ++frame)
    {
        for (uint32_t k = 0; k < 60; ++k)
        {
            for (Built& b : built)
            {
                Rng rng(frame * 1000 + k);
                RandomMove(rng, b.nodes[rng.Below(uint32_t(b.nodes.size()))]);
            }
        }
        if (frame == 5)
            for (Built& b : built)
                b.graph->Attach(b.nodes[3], MakeNode("late"));   // a leaf gets a child: a full pass
        if (frame == 8)
            for (Built& b : built)
                b.nodes[17]->InvalidateContent();
        refresh(frame);
        compare();
    }
    const SceneGraphStore::Counters& off = CountersOf(built[0].graph);
    const SceneGraphStore::Counters& on = CountersOf(built[1].graph);
    CHECK(off.childlessChunks == 0 && on.childlessChunks > 0);
    CHECK(off.subgraphPasses == on.subgraphPasses && off.fullSubgraphPasses == on.fullSubgraphPasses);
    CHECK(on.subgraphPasses > on.fullSubgraphPasses);
    CHECK(!SceneGraph::GetChildlessBoundsPath());
    SceneGraph::SetChildlessBoundsPath(childlessFound);
    std::printf("  equal over 12 frames (%llu passes, %llu full); %llu chunks took the path, %llu bounds chunks\n",
                (unsigned long long)on.subgraphPasses, (unsigned long long)on.fullSubgraphPasses,
                (unsigned long long)on.childlessChunks, (unsigned long long)on.boundsChunks);
}

// The childless Bounds path under churn (stage T5; E7.4 remedy 4): two graphs built and edited alike, the second
// refreshed with SceneGraph::SetChildlessBoundsPath on, for 60 frames of moves of leaves and groups (incremental
// passes) and, every fourth frame, one structural or content edit (a full pass next): a leaf given a child (its chunk's
// summary is no longer childless) and losing it again, a leaf leaving (a hole) while another gets a child for good, a
// re-attach under a group, SetLeaf to another mesh and to none, InvalidateContent, and the box of a shared mesh edited
// (InvalidateContent on its users). After every Refresh both graphs equal the brute force and each other (every node's
// subgraph box and content flags); the second took the path, the first never did.
void test_childless_bounds_churn()
{
    const bool childlessFound = SceneGraph::GetChildlessBoundsPath();
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
    for (int i = 0; i < 4; ++i)
        meshes.push_back(MakeMesh(factory, 0.5f + float(i)));
    struct Built
    {
        nvrhi::AutoPtr<SceneGraph> graph;
        std::vector<SceneGraphNode*> leaves;    // attached leaves (no children of their own when made)
        std::vector<SceneGraphNode*> groups;    // nodes without a leaf, with leaf children
        std::vector<SceneGraphNode*> users;     // nodes of the mesh whose box is edited
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> pool;
        SceneGraphNode* pendingChild = nullptr;   // a child given to a leaf, taken away the next frame
        Rng rng{ 0x5eed };
        SceneGraphNode* AddLeaf(SceneGraphNode* parent, uint32_t mesh)
        {
            auto node = MakeNode("leaf");
            RandomMove(rng, node);
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, (*meshList)[mesh].Get()));
            SceneGraphNode* attached = graph->Attach(parent, node);
            leaves.push_back(attached);
            if (mesh == 3)
                users.push_back(attached);
            return attached;
        }
        const std::vector<nvrhi::AutoPtr<MeshInfo>>* meshList = nullptr;
    };
    Built built[2];
    for (Built& b : built)
    {
        b.meshList = &meshes;
        b.graph = MAKE_RC_OBJ_PTR(SceneGraph);
        b.graph->SetRootNode(MakeNode("root"));
        for (uint32_t i = 0; i < 500; ++i)
            b.AddLeaf(b.graph->GetRootNode(), i % 4);
        for (uint32_t g = 0; g < 10; ++g)
        {
            auto group = MakeNode("group");
            RandomMove(b.rng, group);
            SceneGraphNode* attached = b.graph->Attach(b.graph->GetRootNode(), group);
            b.groups.push_back(attached);
            for (uint32_t i = 0; i < 30; ++i)
                b.AddLeaf(attached, i % 4);
        }
    }
    auto refresh = [&](uint32_t frame)
    {
        SceneGraph::SetChildlessBoundsPath(false);
        built[0].graph->Refresh(frame);
        SceneGraph::SetChildlessBoundsPath(true);
        built[1].graph->Refresh(frame);
        SceneGraph::SetChildlessBoundsPath(childlessFound);
    };
    auto compare = [&]()
    {
        CheckAgainstBruteForce(built[0].graph);
        CheckAgainstBruteForce(built[1].graph);
        SceneGraphWalker a(built[0].graph->GetRootNode());
        SceneGraphWalker b(built[1].graph->GetRootNode());
        for (; a && b; a.Next(true), b.Next(true))
        {
            const dm::box3 boxA = a->GetGlobalBoundingBox();
            const dm::box3 boxB = b->GetGlobalBoundingBox();
            CHECK(SameBits(&boxA, &boxB, sizeof(dm::box3)));
            CHECK(a->GetSubgraphContentFlags() == b->GetSubgraphContentFlags());
        }
        CHECK(!a && !b);
    };
    refresh(1);
    compare();
    for (uint32_t frame = 2; frame <= 60; ++frame)
    {
        // incremental passes in most frames (moves only); every fourth frame one structural edit or content change,
        // which makes the next pass a full one
        const bool structural = frame % 4 == 0;
        const uint32_t kind = (frame / 4) % 6;
        const bool meshEdited = structural && kind == 5;
        if (meshEdited)
        {
            const float s = 1.f + 0.05f * float(frame);
            meshes[3]->objectSpaceBounds = dm::box3(dm::float3(-s), dm::float3(s));
            meshes[3]->geometries[0]->objectSpaceBounds = meshes[3]->objectSpaceBounds;
        }
        for (Built& b : built)
        {
            Rng& rng = b.rng;
            for (uint32_t k = 0; k < 40; ++k)
                RandomMove(rng, b.leaves[rng.Below(uint32_t(b.leaves.size()))]);
            if (frame % 3 == 0)
                RandomMove(rng, b.groups[rng.Below(uint32_t(b.groups.size()))]);
            if (!structural)
                continue;
            if (kind == 0)   // a leaf gets a child (its chunk's summary is no longer childless) ...
                b.pendingChild = b.AddLeaf(b.leaves[rng.Below(uint32_t(b.leaves.size()))], 1);
            if (kind == 1 && b.pendingChild)
            {
                // ... and loses it again
                b.leaves.erase(std::find(b.leaves.begin(), b.leaves.end(), b.pendingChild));
                b.pool.push_back(b.graph->Detach(b.pendingChild));
                b.pendingChild = nullptr;
            }
            if (kind == 2)
            {
                // a leaf leaves the graph (a hole in its chunk), another gets a child for good
                const uint32_t i = rng.Below(uint32_t(b.leaves.size()));
                SceneGraphNode* node = b.leaves[i];
                if (node->GetFirstChild() == nullptr && node != b.pendingChild &&
                    std::find(b.users.begin(), b.users.end(), node) == b.users.end())
                {
                    b.leaves.erase(b.leaves.begin() + i);
                    b.pool.push_back(b.graph->Detach(node));
                }
                b.AddLeaf(b.leaves[rng.Below(uint32_t(b.leaves.size()))], 2);
            }
            if (kind == 3 && !b.pool.empty())
            {
                SceneGraphNode* group = b.groups[rng.Below(uint32_t(b.groups.size()))];
                SceneGraphNode* attached = b.graph->Attach(group, b.pool.back());
                b.pool.pop_back();
                b.leaves.push_back(attached);
            }
            if (kind == 4)
            {
                b.leaves[rng.Below(uint32_t(b.leaves.size()))]->SetLeaf(
                    MAKE_RC_OBJ_PTR(MeshInstance, meshes[rng.Below(3)].Get()));
                b.leaves[rng.Below(uint32_t(b.leaves.size()))]->SetLeaf(nullptr);
            }
            if (kind == 5)
            {
                b.leaves[rng.Below(uint32_t(b.leaves.size()))]->InvalidateContent();
                for (SceneGraphNode* node : b.users)
                    node->InvalidateContent();
            }
        }
        refresh(frame);
        compare();
    }
    const SceneGraphStore::Counters& off = CountersOf(built[0].graph);
    const SceneGraphStore::Counters& on = CountersOf(built[1].graph);
    CHECK(off.childlessChunks == 0 && on.childlessChunks > 50);
    CHECK(off.subgraphPasses == on.subgraphPasses && off.fullSubgraphPasses == on.fullSubgraphPasses);
    CHECK(on.subgraphPasses > 2 * on.fullSubgraphPasses);
    std::printf("  60 frames of churn equal to the brute force with and without the path (%llu passes, %llu full); "
                "%llu chunks took it\n", (unsigned long long)on.subgraphPasses, (unsigned long long)on.fullSubgraphPasses,
                (unsigned long long)on.childlessChunks);
}

// ------------------------------------------------------------ the Static tag under churn (stage T5; E7.4 remedy 3)

namespace
{
    // A leaf class with LeafTypeDesc::dynamicBounds: its box follows `box`, without InvalidateContent.
    class DynamicBoundsMesh final : public MeshInstance
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        explicit DynamicBoundsMesh(MeshInfo* mesh) : MeshInstance(mesh) { }
        dm::box3 box = dm::box3(dm::float3(-1.f), dm::float3(1.f));
        dm::box3 GetLocalBoundingBox() override { return box; }
        const LeafTypeDesc& GetTypeDesc() const override
        {
            static const LeafTypeDesc s_Desc = [this]()
            {
                LeafTypeDesc desc = MeshInstance::GetTypeDesc();
                desc.dynamicBounds = true;
                return desc;
            }();
            return s_Desc;
        }
    };

    // The local transform as propagation composes it (SceneGraphStore.cpp, ComposeLocal).
    dm::daffine3 ComposeLocalTransform(const LocalTransform& local)
    {
        dm::daffine3 transform = dm::scaling(local.scaling);
        transform *= local.rotation.toAffine();
        transform *= dm::translation(local.translation);
        return transform;
    }

    // A scene for the Static switch (stage T5), built and edited alike in any graph from one seed: Static nodes with
    // leaves under the root (some with Static children, some with moving children of their own), moving nodes, Static
    // children of moving parents, a chain of 70 nodes whose nodes from depth 59 on are Static (59-62 honoured, 63 and
    // deeper in the deep bucket, which the tag does not take out), Static nodes without a transform of their own,
    // Static nodes with dynamic-bounds leaves, and runtime-only rows with and without the tag. `Churn` edits it the
    // same way in every graph: moves, Static subtrees attached (rows join Static chunks) and detached (they leave,
    // swap-remove holes) and attached again, first transforms of Static nodes, SetLeaf, reparenting under Static
    // nodes, a mesh box edited with InvalidateContent, dynamic boxes, runtime rows destroyed and made; `violate` adds
    // what the documented contract leaves frozen: moves of Static rows' parents and of the chain's top, and writes of
    // Static rows' LocalTransform (through the world, and through SetTranslation).
    struct StaticScene
    {
        nvrhi::AutoPtr<SceneGraph> graph;
        IWorld* world = nullptr;
        ComponentId staticId = ComponentId::Invalid;
        std::vector<MeshInfo*> meshes;   // [0..2] shared; [3] the one whose box the churn edits
        std::vector<SceneGraphNode*> staticTops, movers, movingParents, deep, untransformed, editedUsers;
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> pool;   // detached Static subtrees
        std::vector<DynamicBoundsMesh*> dynamics;
        std::vector<ComponentId> runtimeComponents;
        MeshId runtimeMesh = MeshId::Invalid;
        std::vector<Entity> runtimeStatic, runtimePlain;
        Rng rng;
        uint32_t made = 0;

        StaticScene(bool honoured, uint64_t seed, const std::vector<MeshInfo*>& meshList) : meshes(meshList), rng(seed)
        {
            const bool found = SceneGraph::GetStaticTag();
            SceneGraph::SetStaticTag(honoured);
            graph = MAKE_RC_OBJ_PTR(SceneGraph);
            SceneGraph::SetStaticTag(found);
            world = graph->GetWorld();
            staticId = RegisterComponent<Static>(world);
            graph->SetRootNode(MakeNode("root"));
            SceneGraphNode* root = graph->GetRootNode();
            for (uint32_t i = 0; i < 240; ++i)
                staticTops.push_back(AddStaticSubtree(root));
            for (uint32_t i = 0; i < 120; ++i)
                movers.push_back(AddNode(root, true, false, meshes[i % 3]));
            for (uint32_t i = 0; i < 30; ++i)
            {
                SceneGraphNode* parent = AddNode(root, true, false, meshes[i % 3]);
                movingParents.push_back(parent);
                for (uint32_t k = 0; k < 2; ++k)
                    AddNode(parent, true, true, meshes[(i + k) % 3]);
            }
            SceneGraphNode* parent = root;
            for (uint32_t d = 1; d <= 70; ++d)
            {
                parent = AddNode(parent, true, d >= 59, d % 3 ? meshes[d % 3] : nullptr);
                deep.push_back(parent);
            }
            for (uint32_t i = 0; i < 24; ++i)
                untransformed.push_back(AddNode(root, false, true, meshes[i % 3]));
            for (uint32_t i = 0; i < 16; ++i)
            {
                auto leaf = MAKE_RC_OBJ_PTR(DynamicBoundsMesh, meshes[0]);
                dynamics.push_back(leaf.Get());
                SceneGraphNode* node = AddNode(root, true, true, nullptr);
                node->SetLeaf(leaf);
            }
            for (uint32_t i = 0; i < 20; ++i)
                editedUsers.push_back(AddNode(root, true, true, meshes[3]));

            runtimeComponents.resize(graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
            graph->GetRuntimeMeshInstanceComponents(runtimeComponents.data(), uint32_t(runtimeComponents.size()));
            runtimeMesh = graph->AcquireRuntimeMesh(meshes[1]);
            CHECK(runtimeMesh != MeshId::Invalid);
            SpawnRuntime(160, true);
            SpawnRuntime(80, false);
        }
        ~StaticScene()
        {
            if (graph && runtimeMesh != MeshId::Invalid)
                graph->ReleaseRuntimeMesh(runtimeMesh);
        }

        void Tag(SceneGraphNode* node)
        {
            const Entity entity = graph->GetEntity(node);
            if (!world->HasComponent(entity, staticId))
                CHECK(world->AddComponent(entity, staticId, nullptr) == nvrhi::FS_OK);
        }
        SceneGraphNode* AddNode(SceneGraphNode* parent, bool transform, bool isStatic, MeshInfo* mesh)
        {
            auto node = MakeNode("n");
            if (transform)
                RandomMove(rng, node);
            if (mesh)
                node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
            SceneGraphNode* attached = graph->Attach(parent, node);
            if (isStatic)
                Tag(attached);
            ++made;
            return attached;
        }
        // A Static node with a leaf, with 0-2 Static children and, one in three, a moving child.
        SceneGraphNode* AddStaticSubtree(SceneGraphNode* parent)
        {
            SceneGraphNode* top = AddNode(parent, true, true, meshes[rng.Below(3)]);
            for (uint32_t k = rng.Below(3); k > 0; --k)
                AddNode(top, true, true, meshes[rng.Below(3)]);
            if (rng.Below(3) == 0)
                movers.push_back(AddNode(top, true, false, meshes[rng.Below(3)]));
            return top;
        }
        void TagSubtree(SceneGraphNode* top)
        {
            // re-attached Static subtrees: their nodes carry the tag again (moving children are not Static)
            for (SceneGraphWalker walker(top); walker; walker.Next(true))
                if (std::find(movers.begin(), movers.end(), walker.Get()) == movers.end())
                    Tag(walker.Get());
        }
        void SpawnRuntime(uint32_t count, bool isStatic)
        {
            std::vector<LocalTransform> locals(count);
            std::vector<MeshRef> refs(count, MeshRef{ runtimeMesh });
            for (LocalTransform& local : locals)
                local = LocalTransform{ dm::double3(rng.Range(-30, 30), rng.Range(-30, 30), rng.Range(-30, 30)),
                                        dm::dquat::identity(), dm::double3(rng.Range(0.5, 2.0)) };
            std::vector<ComponentId> ids = runtimeComponents;
            std::vector<const void*> values(ids.size(), nullptr);
            for (size_t c = 0; c < ids.size(); ++c)
            {
                if (ids[c] == FindComponent<LocalTransform>(world))
                    values[c] = locals.data();
                else if (ids[c] == FindComponent<MeshRef>(world))
                    values[c] = refs.data();
            }
            if (isStatic)
            {
                ids.push_back(staticId);
                values.push_back(nullptr);
            }
            EntityCreateDesc desc{};
            desc.components = ids.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(ids.size());
            desc.entityCount = count;
            std::vector<Entity> rows(count);
            CHECK(world->CreateEntities(desc, rows.data()) == nvrhi::FS_OK);
            std::vector<Entity>& list = isStatic ? runtimeStatic : runtimePlain;
            list.insert(list.end(), rows.begin(), rows.end());
        }
        void DestroyRuntime(uint32_t count)
        {
            std::vector<Entity> victims;
            for (uint32_t k = 0; k < count && !runtimeStatic.empty(); ++k)
            {
                const uint32_t i = rng.Below(uint32_t(runtimeStatic.size()));
                victims.push_back(runtimeStatic[i]);
                runtimeStatic.erase(runtimeStatic.begin() + i);
            }
            CHECK(world->DestroyEntities(victims.data(), uint32_t(victims.size())) == nvrhi::FS_OK);
        }

        // One frame's edits (before Refresh); the mesh box edit is the caller's (the mesh is shared).
        void Churn(uint32_t frame, bool violate, bool meshEdited)
        {
            for (uint32_t k = 0; k < 12; ++k)
                RandomMove(rng, movers[rng.Below(uint32_t(movers.size()))]);
            if (frame % 3 == 0)
                staticTops.push_back(AddStaticSubtree(graph->GetRootNode()));   // rows join Static chunks
            if (frame % 4 == 1 && staticTops.size() > 1)
            {
                // a Static subtree leaves: holes in its chunks, the last rows move in (its moving children leave too)
                const uint32_t i = rng.Below(uint32_t(staticTops.size()));
                SceneGraphNode* top = staticTops[i];
                staticTops.erase(staticTops.begin() + i);
                for (SceneGraphWalker walker(top); walker; walker.Next(true))
                {
                    const auto it = std::find(movers.begin(), movers.end(), walker.Get());
                    if (it != movers.end())
                        movers.erase(it);
                }
                pool.push_back(graph->Detach(top));
            }
            if (frame % 5 == 2 && !pool.empty())
            {
                SceneGraphNode* top = graph->Attach(graph->GetRootNode(), pool.back());
                pool.pop_back();
                TagSubtree(top);
                staticTops.push_back(top);
            }
            if (frame % 6 == 3 && !untransformed.empty())
            {
                untransformed.back()->SetTranslation(dm::double3(rng.Range(-9, 9), 1.0, 2.0));   // its first transform
                untransformed.pop_back();
            }
            if (frame % 7 == 4)
                staticTops[rng.Below(uint32_t(staticTops.size()))]->SetLeaf(
                    MAKE_RC_OBJ_PTR(MeshInstance, meshes[rng.Below(3)]));
            if (frame % 8 == 5 && movers.size() > 10)
            {
                // a moving node goes under a Static node
                SceneGraphNode* mover = movers[rng.Below(uint32_t(movers.size()))];
                SceneGraphNode* top = staticTops[rng.Below(uint32_t(staticTops.size()))];
                auto detached = graph->Detach(mover);
                const auto it = std::find(movers.begin(), movers.end(), mover);
                movers.erase(it);
                movers.push_back(graph->Attach(top, detached));
            }
            if (meshEdited)
                for (SceneGraphNode* node : editedUsers)
                    node->InvalidateContent();
            dynamics[rng.Below(uint32_t(dynamics.size()))]->box =
                dm::box3(dm::float3(-float(rng.Range(0.5, 2.0))), dm::float3(float(rng.Range(0.5, 2.0))));
            if (frame % 3 == 1)
            {
                DestroyRuntime(12);
                SpawnRuntime(12, true);
            }
            if (!violate)
                return;
            for (uint32_t k = 0; k < 4; ++k)
                RandomMove(rng, movingParents[rng.Below(uint32_t(movingParents.size()))]);
            if (frame % 2 == 0)
                RandomMove(rng, deep[0]);
            // writes of Static rows' local transforms: through the world, and through the setter
            world->AdvanceVersion();
            const Entity written = graph->GetEntity(staticTops[rng.Below(uint32_t(staticTops.size()))]);
            if (auto* local = static_cast<LocalTransform*>(
                    world->GetComponentForWrite(written, FindComponent<LocalTransform>(world), c_WorldVersion)))
                local->translation.y += 0.5;
            staticTops[rng.Below(uint32_t(staticTops.size()))]->SetTranslation(dm::double3(rng.Range(-9, 9), 3.0, 0.0));
        }
    };

    struct RowPlace
    {
        uint32_t archetype, chunk, row;
    };

    // Whether a row is one the tag takes out (SceneGraph.h, SetStaticTag): Static, below the deep bucket, not driven.
    bool HonouredStatic(IWorld* world, Entity entity, ComponentId staticId)
    {
        if (!world->HasComponent(entity, staticId))
            return false;
        const ComponentId driven = FindComponent<SimulationDriven>(world);
        if (driven != ComponentId::Invalid && world->HasComponent(entity, driven))
            return false;
        const auto* depth = static_cast<const HierarchyDepth*>(world->GetComponent(entity,
                                                                                  FindComponent<HierarchyDepth>(world)));
        return !depth || depth->depth < c_MaxHierarchyDepth;
    }
}

// The Static switch where its documentation promises the switch-off results (stage T5): a scene whose Static rows and
// their ancestors never move and whose Static rows' local transforms are never written after they settle, edited for
// 36 frames (StaticScene::Churn without `violate`): rows join and leave Static chunks, Static nodes get their first
// transform and a new leaf, moving nodes go under Static nodes, a mesh's box is edited (InvalidateContent), dynamic
// boxes change, runtime-only rows with the tag are destroyed and made. Two graphs, the tag honoured in the second; after
// every Refresh: every node's world transform (bits), box and content flags, every runtime row's world transform and
// box, and GpuScene's instance rows equal between them, both against the brute force; the second graph left Static
// archetypes out. Then 10 frames of moves only: the second graph allocates nothing.
void test_static_tag_promised()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    std::vector<nvrhi::AutoPtr<MeshInfo>> owned;
    std::vector<MeshInfo*> meshes;
    for (int i = 0; i < 4; ++i)
    {
        owned.push_back(MakeMesh(factory, 0.5f + 0.5f * float(i)));
        meshes.push_back(owned.back());
    }
    StaticScene scenes[2] = { StaticScene(false, 0x57a7, meshes), StaticScene(true, 0x57a7, meshes) };
    GpuScene gpu[2] = { GpuScene(nullptr, GpuSceneDesc{}), GpuScene(nullptr, GpuSceneDesc{}) };
    auto compare = [&](uint32_t frame)
    {
        SceneGraphWalker a(scenes[0].graph->GetRootNode());
        SceneGraphWalker b(scenes[1].graph->GetRootNode());
        uint32_t nodes = 0;
        for (; a && b; a.Next(true), b.Next(true), ++nodes)
        {
            const dm::daffine3 wa = a->GetLocalToWorldTransform(), wb = b->GetLocalToWorldTransform();
            const dm::box3 ba = a->GetGlobalBoundingBox(), bb = b->GetGlobalBoundingBox();
            if (!SameBits(&wa, &wb, sizeof(wa)) || !SameBox(ba, bb) ||
                a->GetSubgraphContentFlags() != b->GetSubgraphContentFlags())
            {
                std::fprintf(stderr, "frame %u: node %u differs (static %d)\n", frame, nodes,
                             int(scenes[1].world->HasComponent(scenes[1].graph->GetEntity(b.Get()), scenes[1].staticId)));
                CHECK(false);
            }
        }
        CHECK(!a && !b && nodes > 600);
        for (int list = 0; list < 2; ++list)
        {
            const std::vector<Entity>& x = list ? scenes[0].runtimeStatic : scenes[0].runtimePlain;
            const std::vector<Entity>& y = list ? scenes[1].runtimeStatic : scenes[1].runtimePlain;
            CHECK(x.size() == y.size());
            for (size_t i = 0; i < x.size(); ++i)
            {
                CHECK(SameBits(GetComponent<WorldTransform>(scenes[0].world, x[i]),
                               GetComponent<WorldTransform>(scenes[1].world, y[i]), sizeof(WorldTransform)));
                CHECK(SameBits(GetComponent<WorldBounds>(scenes[0].world, x[i]),
                               GetComponent<WorldBounds>(scenes[1].world, y[i]), sizeof(WorldBounds)));
            }
        }
        for (uint32_t g = 0; g < 2; ++g)
            CHECK(NVRHI_SUCCEEDED(gpu[g].Update(scenes[g].graph)));
        CHECK(gpu[0].GetInstanceRowCount() == gpu[1].GetInstanceRowCount());
        CHECK(SameBits(gpu[0].GetInstanceRows(), gpu[1].GetInstanceRows(),
                       sizeof(InstanceData) * gpu[0].GetInstanceRowCount()));
        CheckAgainstBruteForce(scenes[0].graph);
        CheckAgainstBruteForce(scenes[1].graph);
    };
    for (uint32_t frame = 1; frame <= 36; ++frame)
    {
        const bool meshEdited = frame % 5 == 4;
        if (meshEdited)
        {
            const float s = 0.5f + 0.1f * float(frame);
            meshes[3]->objectSpaceBounds = dm::box3(dm::float3(-s), dm::float3(s));
            meshes[3]->geometries[0]->objectSpaceBounds = meshes[3]->objectSpaceBounds;
        }
        for (StaticScene& scene : scenes)
        {
            if (frame > 1)
                scene.Churn(frame, false, meshEdited);
            scene.graph->Refresh(frame);
        }
        compare(frame);
    }
    const SceneGraphStore::Counters off = CountersOf(scenes[0].graph);   // copies: the steady frames go on below
    const SceneGraphStore::Counters on = CountersOf(scenes[1].graph);
    CHECK(off.staticSkips == 0 && on.staticSkips > 36);
    CHECK(on.propagatedRows < off.propagatedRows);

    // steady state: moves of moving nodes and dynamic boxes only
    uint32_t frame = 36;
    auto steady = [&]()
    {
        ++frame;
        StaticScene& scene = scenes[1];
        for (uint32_t k = 0; k < 12; ++k)
            RandomMove(scene.rng, scene.movers[scene.rng.Below(uint32_t(scene.movers.size()))]);
        scene.dynamics[frame % scene.dynamics.size()]->box = dm::box3(dm::float3(-float(frame % 3 + 1)),
                                                                       dm::float3(1.f));
        scene.graph->Refresh(frame);
        CHECK(NVRHI_SUCCEEDED(gpu[1].Update(scene.graph)));
    };
    for (int i = 0; i < 5; ++i)
        steady();
    const uint64_t newBefore = g_NewCount.load();
    const uint64_t heapBefore = ecs::GetHeapAllocationCount();
    for (int i = 0; i < 10; ++i)
        steady();
    const uint64_t newCalls = g_NewCount.load() - newBefore;
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
    std::printf("  36 frames equal with the tag honoured (%llu and %llu rows propagated, Static archetypes left out "
                "%llu times); steady state with it: %llu operator new, %llu storage blocks\n",
                (unsigned long long)off.propagatedRows, (unsigned long long)on.propagatedRows,
                (unsigned long long)on.staticSkips, (unsigned long long)newCalls, (unsigned long long)heapBlocks);
    CHECK(newCalls == 0 && heapBlocks == 0);
}

// The Static switch's documented contract where it differs from the switch off (stage T5; SceneGraph.h, SetStaticTag,
// as fixed in F1): the scene of test_static_tag_promised with the tag honoured, edited for 40 frames with `violate` too
// (Static rows' parents and the deep chain's top move; Static rows' local transforms are written through the world and
// through SetTranslation). After every Refresh, from the rows' places (archetype, chunk, row) before and after: a
// Static row the tag takes out (below the deep bucket, not driven) whose chunk got rows (a join, a row moved into a
// hole: a place with another row, or the chunk's Added version stamped since the last Refresh) or that is new is
// composed from its current local transform and its parent's current world (so are the rows sharing that chunk:
// documented at F1, backlog F1-N1); any other such row keeps the world it had (also where the switch off would move
// it: counted); every other row (moving, deep bucket, runtime) is composed from its local
// and its parent's world as stored; rows' boxes follow the stored worlds (the brute force). Debug builds refuse the
// setter's writes of settled Static rows and warn about the world's (counted, then taken back); Release builds warn
// about neither.
void test_static_tag_contract()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    std::vector<nvrhi::AutoPtr<MeshInfo>> owned;
    std::vector<MeshInfo*> meshes;
    for (int i = 0; i < 4; ++i)
    {
        owned.push_back(MakeMesh(factory, 0.5f + 0.5f * float(i)));
        meshes.push_back(owned.back());
    }
    StaticScene scene(true, 0xc0de, meshes);
    IWorld* world = scene.world;
    ecs::WorldImpl* impl = ecs::GetWorldImpl(world);
    std::map<Entity, dm::daffine3> previousWorlds;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Entity> previousPlaces;
    const uint32_t warnings = g_Warnings.load();
    uint64_t frozen = 0, recomposed = 0, followed = 0, diverged = 0;
    uint64_t lastRefreshVersion = 0;   // the world's version after the last Refresh: later stamps are this frame's

    // Every row of the graph's world with a WorldTransform: nodes and runtime rows.
    auto rows = [&]()
    {
        std::vector<Entity> out;
        for (SceneGraphWalker walker(scene.graph->GetRootNode()); walker; walker.Next(true))
            out.push_back(scene.graph->GetEntity(walker.Get()));
        out.insert(out.end(), scene.runtimeStatic.begin(), scene.runtimeStatic.end());
        out.insert(out.end(), scene.runtimePlain.begin(), scene.runtimePlain.end());
        return out;
    };
    auto fresh = [&](Entity entity)
    {
        const LocalTransform* local = GetComponent<LocalTransform>(world, entity);
        const Parent* parent = GetComponent<Parent>(world, entity);
        const bool hasParent = parent && parent->entity != Entity::Null;
        const dm::daffine3 parentWorld = hasParent
            ? GetComponent<WorldTransform>(world, parent->entity)->transform : dm::daffine3::identity();
        if (!local)
            return parentWorld;
        return hasParent ? ComposeLocalTransform(*local) * parentWorld : ComposeLocalTransform(*local);
    };
    for (uint32_t frame = 1; frame <= 40; ++frame)
    {
        const bool meshEdited = frame % 5 == 4;
        if (meshEdited)
        {
            const float s = 0.5f + 0.1f * float(frame);
            meshes[3]->objectSpaceBounds = dm::box3(dm::float3(-s), dm::float3(s));
            meshes[3]->geometries[0]->objectSpaceBounds = meshes[3]->objectSpaceBounds;
        }
        if (frame > 1)
            scene.Churn(frame, true, meshEdited);
        scene.graph->Refresh(frame);
        const uint64_t versionAfterLastRefresh = lastRefreshVersion;
        lastRefreshVersion = world->GetVersion();

        const std::vector<Entity> all = rows();
        std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Entity> places;
        std::map<std::pair<uint32_t, uint32_t>, bool> chunkGotRow;   // a place of the chunk has another row than before
        for (Entity entity : all)
        {
            const ecs::EntityRecord* record = impl->FindAlive(entity);
            CHECK(record);
            const auto place = std::make_tuple(record->archetype, record->chunk, record->row);
            places[place] = entity;
            const auto it = previousPlaces.find(place);
            // a place with another row, or rows added since the last Refresh (also a row that left the chunk's end and
            // came back to the same place: SetLeaf, a re-attach)
            const ecs::Archetype& archetype = impl->GetArchetypeStorage(record->archetype);
            const uint64_t added = archetype.Versions(record->chunk)[archetype.AddedVersionSlot()];
            if (it == previousPlaces.end() || it->second != entity || added >= versionAfterLastRefresh)
                chunkGotRow[{ record->archetype, record->chunk }] = true;
        }
        for (Entity entity : all)
        {
            const dm::daffine3 stored = GetComponent<WorldTransform>(world, entity)->transform;
            const ecs::EntityRecord* record = impl->FindAlive(entity);
            const auto prev = previousWorlds.find(entity);
            if (frame > 1 && HonouredStatic(world, entity, scene.staticId))
            {
                const bool taken = prev == previousWorlds.end() ||
                                   chunkGotRow.count({ record->archetype, record->chunk }) != 0;
                const dm::daffine3 expected = taken ? fresh(entity) : prev->second;
                if (!SameBits(&stored, &expected, sizeof(stored)))
                {
                    const dm::daffine3 composed = fresh(entity);
                    const auto* depth = GetComponent<HierarchyDepth>(world, entity);
                    std::fprintf(stderr, "frame %u: Static row %llu (%s; archetype %u chunk %u row %u, depth %d, local "
                                 "%d, runtime %d): not the world the contract gives; it is %s\n", frame,
                                 (unsigned long long)entity, taken ? "taken" : "settled", record->archetype,
                                 record->chunk, record->row, depth ? int(depth->depth) : -1,
                                 int(GetComponent<LocalTransform>(world, entity) != nullptr),
                                 int(std::find(scene.runtimeStatic.begin(), scene.runtimeStatic.end(), entity) !=
                                     scene.runtimeStatic.end()),
                                 SameBits(&stored, &composed, sizeof(stored)) ? "composed again" :
                                 prev != previousWorlds.end() && SameBits(&stored, &prev->second, sizeof(stored))
                                     ? "the world it had" : "neither");
                    CHECK(false);
                }
                (taken ? recomposed : frozen) += 1;
                if (!taken)
                {
                    const dm::daffine3 composed = fresh(entity);
                    diverged += SameBits(&stored, &composed, sizeof(stored)) ? 0 : 1;   // the switch off would move it
                }
            }
            else
            {
                const dm::daffine3 expected = fresh(entity);
                if (!SameBits(&stored, &expected, sizeof(stored)))
                {
                    std::fprintf(stderr, "frame %u: row %llu (static %d): not composed from its parent's world\n",
                                 frame, (unsigned long long)entity, int(world->HasComponent(entity, scene.staticId)));
                    CHECK(false);
                }
                ++followed;
            }
            // the rows' boxes follow their stored worlds (every row's WorldBounds here; the subgraph boxes in the brute
            // force below)
            const LocalBounds* local = GetComponent<LocalBounds>(world, entity);
            const WorldBounds* box = GetComponent<WorldBounds>(world, entity);
            if (local && box && !local->bounds.isempty())
            {
                const dm::box3 expectedBox = local->bounds * dm::affine3(stored);
                CHECK(SameBox(box->bounds, expectedBox));
            }
        }
        CheckAgainstBruteForce(scene.graph);
        previousPlaces.swap(places);
        previousWorlds.clear();
        for (Entity entity : all)
            previousWorlds[entity] = GetComponent<WorldTransform>(world, entity)->transform;
    }
#ifndef NDEBUG
    CHECK(g_Warnings.load() > warnings);   // the refused setter calls and the world's writes
#else
    CHECK(g_Warnings.load() == warnings);
#endif
    g_Warnings = warnings;
    std::printf("  40 frames of the contract: %llu settled Static rows kept their worlds (%llu where the switch off "
                "would have moved them), %llu were composed again (their chunks got rows), %llu other rows followed "
                "their parents\n", (unsigned long long)frozen, (unsigned long long)diverged,
                (unsigned long long)recomposed, (unsigned long long)followed);
    CHECK(frozen > 1000 && diverged > 100 && recomposed > 100 && followed > 1000);
}

// Propagation by row with RowTracked transforms (stage T8, the killer of the work-only mutants of I4): 64 parents
// under the root in one chunk, each with a child (another chunk), a graph made with SceneGraph::SetRowTrackedTransforms
// on and one with it off. One parent moves: by row, propagation composes that parent and its child (2 rows); by chunk,
// the parents' chunk whole and every child (128 rows). The worlds are equal.
void test_row_tracked_by_row()
{
    const bool found = SceneGraph::GetRowTrackedTransforms();
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);
    struct Built
    {
        nvrhi::AutoPtr<SceneGraph> graph;
        std::vector<SceneGraphNode*> parents, children;
    };
    Built built[2];
    for (uint32_t tracked = 0; tracked < 2; ++tracked)
    {
        Built& b = built[tracked];
        SceneGraph::SetRowTrackedTransforms(tracked != 0);
        b.graph = MAKE_RC_OBJ_PTR(SceneGraph);
        SceneGraph::SetRowTrackedTransforms(found);
        b.graph->SetRootNode(MakeNode("root"));
        for (uint32_t i = 0; i < 64; ++i)
        {
            auto parent = MakeNode("parent");
            parent->SetTranslation(dm::double3(double(i), 0.0, 0.0));
            parent->SetLeaf(factory->CreateMeshInstance(mesh));
            b.parents.push_back(b.graph->Attach(b.graph->GetRootNode(), parent));
        }
        for (SceneGraphNode* parent : b.parents)
        {
            auto child = MakeNode("child");
            child->SetTranslation(dm::double3(0.0, 1.0, 0.0));
            child->SetLeaf(factory->CreateMeshInstance(mesh));
            b.children.push_back(b.graph->Attach(parent, child));
        }
        b.graph->Refresh(1);
        b.graph->Refresh(2);
    }
    uint64_t moved[2] = {};
    for (uint32_t frame = 3; frame <= 6; ++frame)
    {
        for (uint32_t tracked = 0; tracked < 2; ++tracked)
        {
            Built& b = built[tracked];
            const uint64_t rows = CountersOf(b.graph).propagatedRows;
            b.parents[frame * 7 % 64]->SetTranslation(dm::double3(double(frame), 5.0, 0.0));
            b.graph->Refresh(frame);
            moved[tracked] = CountersOf(b.graph).propagatedRows - rows;
            CHECK(moved[tracked] == (tracked ? 2u : 128u));
        }
        for (size_t i = 0; i < 64; ++i)
        {
            const dm::daffine3 a = built[0].children[i]->GetLocalToWorldTransform();
            const dm::daffine3 b = built[1].children[i]->GetLocalToWorldTransform();
            CHECK(SameBits(&a, &b, sizeof(a)));
        }
    }
    std::printf("  one parent of 64 moved: %llu rows composed by row, %llu by chunk\n", (unsigned long long)moved[1],
                (unsigned long long)moved[0]);
}

// The childless Bounds path where it alone marks the parent (stage T8, the killer of mutants of the path's mark and
// keys): two graphs of 600 leaves directly under the root (every chunk childless, one parent) and nothing else, the
// second refreshed with the path on; 20 frames of moves of a few leaves each (incremental passes only): the root's box
// and every leaf's equal the brute force and the other graph's, and the path was taken by every chunk the Bounds
// system wrote.
void test_childless_root_only()
{
    const bool found = SceneGraph::GetChildlessBoundsPath();
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 0.5f);
    nvrhi::AutoPtr<SceneGraph> graphs[2];
    std::vector<SceneGraphNode*> leaves[2];
    for (uint32_t g = 0; g < 2; ++g)
    {
        graphs[g] = MAKE_RC_OBJ_PTR(SceneGraph);
        graphs[g]->SetRootNode(MakeNode("root"));
        Rng rng(0x600);
        for (uint32_t i = 0; i < 600; ++i)
        {
            auto node = MakeNode("leaf");
            RandomMove(rng, node);
            node->SetLeaf(factory->CreateMeshInstance(mesh));
            leaves[g].push_back(graphs[g]->Attach(graphs[g]->GetRootNode(), node));
        }
    }
    auto refresh = [&](uint32_t frame)
    {
        SceneGraph::SetChildlessBoundsPath(false);
        graphs[0]->Refresh(frame);
        SceneGraph::SetChildlessBoundsPath(true);
        graphs[1]->Refresh(frame);
        SceneGraph::SetChildlessBoundsPath(found);
    };
    refresh(1);
    for (uint32_t frame = 2; frame <= 21; ++frame)
    {
        for (uint32_t g = 0; g < 2; ++g)
        {
            Rng rng(frame);
            for (uint32_t k = 0; k < 3; ++k)
            {
                // far out, so that the root's box changes with them
                SceneGraphNode* node = leaves[g][rng.Below(600)];
                node->SetTranslation(dm::double3(rng.Range(-50.0, 50.0) * double(frame), 0.0, 0.0));
            }
        }
        refresh(frame);
        CheckAgainstBruteForce(graphs[0]);
        CheckAgainstBruteForce(graphs[1]);
        const dm::box3 a = graphs[0]->GetRootNode()->GetGlobalBoundingBox();
        const dm::box3 b = graphs[1]->GetRootNode()->GetGlobalBoundingBox();
        CHECK(SameBits(&a, &b, sizeof(a)));
    }
    const SceneGraphStore::Counters& on = CountersOf(graphs[1]);
    CHECK(on.childlessChunks > 0 && on.subgraphPasses > on.fullSubgraphPasses);
    std::printf("  600 leaves under the root, 20 frames: the root's box equal to the brute force (%llu chunks took the "
                "path)\n", (unsigned long long)on.childlessChunks);
}

// A Static row's writes before and after the run that takes it (stage T8; SceneGraph.h, SetStaticTag): in a graph made
// with the switch on, a node attached and tagged Static, then moved before any Refresh, takes that move (its initial
// value, in both builds, no warning); once a propagation run took it, SetTranslation is refused with one warning in
// Debug builds and written but not propagated in Release builds; in both its world stays.
void test_static_initial_writes()
{
    const bool found = SceneGraph::GetStaticTag();
    SceneGraph::SetStaticTag(true);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    SceneGraph::SetStaticTag(found);
    IWorld* world = graph->GetWorld();
    const ComponentId staticId = RegisterComponent<Static>(world);
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto mesh = MakeMesh(factory, 1.f);
    graph->SetRootNode(MakeNode("root"));
    graph->Refresh(1);
    auto made = MakeNode("static");
    made->SetTranslation(dm::double3(1.0, 0.0, 0.0));
    made->SetLeaf(factory->CreateMeshInstance(mesh));
    SceneGraphNode* node = graph->Attach(graph->GetRootNode(), made);
    CHECK(world->AddComponent(graph->GetEntity(node), staticId, nullptr) == nvrhi::FS_OK);
    const uint32_t warnings = g_Warnings.load();
    node->SetTranslation(dm::double3(2.0, 0.0, 0.0));   // before the run that takes the row: its initial value
    CHECK(g_Warnings.load() == warnings && node->GetTranslation().x == 2.0);
    graph->Refresh(2);
    CHECK(node->GetLocalToWorldTransform().m_translation.x == 2.0);
    node->SetTranslation(dm::double3(3.0, 0.0, 0.0));   // settled: refused in Debug, not propagated in Release
#ifndef NDEBUG
    CHECK(g_Warnings.load() == warnings + 1 && node->GetTranslation().x == 2.0);
#else
    CHECK(g_Warnings.load() == warnings && node->GetTranslation().x == 3.0);
#endif
    graph->Refresh(3);
    CHECK(node->GetLocalToWorldTransform().m_translation.x == 2.0);
    g_Warnings = warnings;
    std::printf("  a Static row's initial value taken; a later write %s\n",
#ifndef NDEBUG
                "refused with one warning"
#else
                "written, not propagated"
#endif
    );
}

int main(int argc, char** argv)
{
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        { "columns", test_columns_of_attached_nodes },
        { "records", test_orphan_records_round_trip },
        { "leaf_order", test_detach_keeps_leaf_order },
        { "teardown", test_teardown_keeps_referenced_nodes },
        { "deep", test_deep_hierarchy_and_determinism },
        { "bounds", test_bounds_and_chunk_bounds },
        { "caches", test_subgraph_caches },
        { "subgraph_refresh", test_subgraph_refresh },
        { "phase_structure", test_phase_structure },
        { "subgraph_nan", test_subgraph_nan },
        { "walker_order", test_walker_order },
        { "pending", test_pending_changes_and_joints },
        { "stale_world_transforms", test_stale_world_transforms },
        { "simulation_components", test_simulation_components },
        { "row_hints", test_row_hints },
        { "review_fixes", test_review_fixes },
        { "childless_bounds_path", test_childless_bounds_path },
        { "childless_bounds_churn", test_childless_bounds_churn },
        { "static_tag_promised", test_static_tag_promised },
        { "static_tag_contract", test_static_tag_contract },
        { "row_tracked_by_row", test_row_tracked_by_row },
        { "childless_root_only", test_childless_root_only },
        { "static_initial_writes", test_static_initial_writes },
        { "steady_state", test_steady_state_allocations },
    };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
                continue;
            c.run();
            CHECK(g_Warnings.load() == 0);
            std::printf("%s: passed\n", c.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    return 0;
}
