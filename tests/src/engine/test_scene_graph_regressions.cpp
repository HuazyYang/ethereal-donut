// Regression tests for scene-graph defects fixed before the ECS work (aggregate docs/plans/2026-11-21-donut-ecs.md,
// step E0):
// - SceneGraph::UnregisterLeaf returned from the MeshInstance branch before it reached the SkinnedMeshInstance
//   branch, so a detached or replaced skinned instance stayed in GetSkinnedMeshInstances().
// - Attaching a subgraph that had been refreshed in a graph before (detached and attached again, or the old root
//   attached under a new one) marked only its root as new structure. Refresh then stopped one level below that
//   root, so deeper nodes kept the world transforms and bounds of their old place.

#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include <cstdio>

using namespace donut;
using namespace donut::engine;

namespace
{
    nvrhi::AutoPtr<SceneGraphNode> make_node(const char* name)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName(name);
        return node;
    }

    // A mesh with one opaque geometry and bounds [-1, 1]^3; no device objects.
    nvrhi::AutoPtr<MeshInfo> make_mesh(SceneTypeFactory* factory, const char* name)
    {
        auto material = factory->CreateMaterial();
        material->name = name;
        auto geometry = factory->CreateMeshGeometry();
        geometry->material = material;
        geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
        auto mesh = factory->CreateMesh();
        mesh->name = name;
        mesh->geometries.push_back(geometry);
        mesh->objectSpaceBounds = geometry->objectSpaceBounds;
        return mesh;
    }

    bool same_box(const dm::box3& a, const dm::box3& b)
    {
        return all(a.m_mins == b.m_mins) && all(a.m_maxs == b.m_maxs);
    }
}

// Detaching a skinned instance takes it out of both leaf vectors; attaching it again puts it back once.
void test_detach_unregisters_skinned_instance()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));

    auto skinned = factory->CreateSkinnedMeshInstance(factory, make_mesh(factory, "prototype"));
    auto node = graph->AttachLeafNode(graph->GetRootNode(), skinned);
    CHECK(graph->GetMeshInstances().size() == 1);
    CHECK(graph->GetSkinnedMeshInstances().size() == 1);
    CHECK(graph->GetMeshes().size() == 2);   // the skinned mesh and its prototype

    graph->Detach(node);
    CHECK(graph->GetMeshInstances().empty());
    CHECK(graph->GetSkinnedMeshInstances().empty());
    CHECK(graph->GetMeshes().empty());

    graph->Attach(graph->GetRootNode(), node);
    CHECK(graph->GetMeshInstances().size() == 1);
    CHECK(graph->GetSkinnedMeshInstances().size() == 1);
    CHECK(graph->GetSkinnedMeshInstances()[0].Get() == skinned.Get());
}

// Replacing a skinned instance with SetLeaf unregisters it from both leaf vectors.
void test_set_leaf_unregisters_replaced_skinned_instance()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));

    auto node = graph->AttachLeafNode(graph->GetRootNode(),
        factory->CreateSkinnedMeshInstance(factory, make_mesh(factory, "prototype")));
    CHECK(graph->GetSkinnedMeshInstances().size() == 1);

    auto plain = factory->CreateMeshInstance(make_mesh(factory, "plain"));
    node->SetLeaf(plain);
    CHECK(graph->GetMeshInstances().size() == 1);
    CHECK(graph->GetMeshInstances()[0].Get() == plain.Get());
    CHECK(graph->GetSkinnedMeshInstances().empty());
}

// A subgraph three levels deep is refreshed in place, detached, and attached under a parent that sits elsewhere.
// Every node of it, down to the deepest one, takes the world transform and bounds of its new place.
void test_reattached_subgraph_refreshes_completely()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    auto left = graph->Attach(graph->GetRootNode(), make_node("left"));
    auto right = graph->Attach(graph->GetRootNode(), make_node("right"));
    right->SetTranslation(dm::double3(10.0, 0.0, 0.0));

    auto p = graph->Attach(left, make_node("p"));
    auto q = graph->Attach(p, make_node("q"));
    auto r = graph->Attach(q, make_node("r"));
    r->SetLeaf(factory->CreateMeshInstance(make_mesh(factory, "mesh")));
    graph->Refresh(0);
    CHECK(all(r->GetLocalToWorldTransform().m_translation == dm::double3(0.0)));

    graph->Detach(p);
    graph->Refresh(1);
    graph->Attach(right, p);
    graph->Refresh(2);

    const dm::box3 moved(dm::float3(9.f, -1.f, -1.f), dm::float3(11.f, 1.f, 1.f));
    CHECK(all(q->GetLocalToWorldTransform().m_translation == dm::double3(10.0, 0.0, 0.0)));
    CHECK(all(r->GetLocalToWorldTransform().m_translation == dm::double3(10.0, 0.0, 0.0)));
    CHECK(all(r->GetLocalToWorldTransformFloat().m_translation == dm::float3(10.f, 0.f, 0.f)));
    CHECK(same_box(r->GetGlobalBoundingBox(), moved));
    CHECK(same_box(q->GetGlobalBoundingBox(), moved));
    CHECK(same_box(p->GetGlobalBoundingBox(), moved));
    CHECK(same_box(graph->GetRootNode()->GetGlobalBoundingBox(), moved));
}

// The old root of a graph attached under a new, translated root: the same refresh, through SetRootNode.
void test_old_root_under_new_root_refreshes_completely()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    auto a = graph->Attach(graph->GetRootNode(), make_node("a"));
    auto b = graph->Attach(a, make_node("b"));
    b->SetLeaf(factory->CreateMeshInstance(make_mesh(factory, "mesh")));
    graph->Refresh(0);

    auto newRoot = make_node("newRoot");
    newRoot->SetTranslation(dm::double3(0.0, 5.0, 0.0));
    auto oldRoot = graph->SetRootNode(newRoot);
    graph->Attach(newRoot, oldRoot);
    graph->Refresh(1);

    CHECK(graph->GetRootNode() == newRoot.Get());
    CHECK(oldRoot->GetParent() == newRoot.Get());
    CHECK(all(b->GetLocalToWorldTransform().m_translation == dm::double3(0.0, 5.0, 0.0)));
    CHECK(same_box(b->GetGlobalBoundingBox(), dm::box3(dm::float3(-1.f, 4.f, -1.f), dm::float3(1.f, 6.f, 1.f))));
    CHECK(graph->GetMeshInstances().size() == 1);
}

int main(int, char**)
{
    try
    {
        test_detach_unregisters_skinned_instance();
        test_set_leaf_unregisters_replaced_skinned_instance();
        test_reattached_subgraph_refreshes_completely();
        test_old_root_under_new_root_refreshes_completely();
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    return 0;
}
