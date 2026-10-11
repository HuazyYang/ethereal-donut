// Scene graph node hierarchy: children form a first-child / next-sibling list with tail append.
// Destroying a node with a very large number of children must not overflow the stack
// (NVIDIAGameWorks/donut#18), children keep their insertion order, and a node that is still referenced
// from outside survives the destruction of its parent with its own subgraph intact.

#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    // Larger than any child count that the recursive sibling-list destructor survived on a 1 MB stack.
    constexpr int c_WideChildCount = 1000000;

    nvrhi::AutoPtr<SceneGraphNode> make_node(const char* name)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName(name);
        return node;
    }

    // Child names in list order, after checking that every forward, backward, tail and parent link agrees.
    std::vector<std::string> child_names(SceneGraphNode* parent)
    {
        std::vector<std::string> names;
        SceneGraphNode* prev = nullptr;
        for (SceneGraphNode* child = parent->GetFirstChild(); child; child = child->GetNextSibling())
        {
            CHECK(child->GetParent() == parent);
            CHECK(child->GetPrevSibling() == prev);
            names.push_back(child->GetName());
            prev = child;
        }
        CHECK(parent->GetLastChild() == prev);
        return names;
    }

    bool is_unlinked(SceneGraphNode* node)
    {
        return !node->GetParent() && !node->GetPrevSibling() && !node->GetNextSibling();
    }

    // Attaches `count` children to `parent` and returns the middle one, which only the parent keeps alive.
    SceneGraphNode* build_wide(SceneGraph* graph, SceneGraphNode* parent, int count)
    {
        SceneGraphNode* middle = nullptr;
        for (int i = 0; i < count; i++)
        {
            auto child = graph->Attach(parent, MAKE_RC_OBJ_PTR(SceneGraphNode));
            if (i == count / 2)
                middle = child.Get();
        }
        return middle;
    }

    //   root
    //   +- a
    //   |  +- a1
    //   |  +- a2
    //   +- b
    //   +- c
    //      +- c1
    struct SmallTree
    {
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        nvrhi::AutoPtr<SceneGraphNode> root, a, a1, a2, b, c, c1;

        SmallTree()
        {
            graph->SetRootNode(make_node("root"));
            root = graph->GetRootNode();
            a = graph->Attach(root, make_node("a"));
            a1 = graph->Attach(a, make_node("a1"));
            a2 = graph->Attach(a, make_node("a2"));
            b = graph->Attach(root, make_node("b"));
            c = graph->Attach(root, make_node("c"));
            c1 = graph->Attach(c, make_node("c1"));
        }
    };
}

// The issue #18 repro: a parent with a million children is destroyed without a stack overflow,
// both as an orphaned subgraph and as the root of a graph.
void test_wide_orphan_subgraph_destruction()
{
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    auto parent = make_node("parent");

    nvrhi::WeakPtr<SceneGraphNode> probe(build_wide(graph, parent, c_WideChildCount));
    CHECK(probe.Lock());

    parent.Reset();
    CHECK(!probe.Lock());
}

void test_wide_graph_destruction()
{
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));

    nvrhi::WeakPtr<SceneGraphNode> probe(build_wide(graph, graph->GetRootNode(), c_WideChildCount));
    CHECK(probe.Lock());

    // Walking the wide level is iterative as well
    int visited = 0;
    for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
        ++visited;
    CHECK(visited == c_WideChildCount + 1);

    graph.Reset();
    CHECK(!probe.Lock());
}

void test_append_preserves_insertion_order()
{
    SmallTree tree;

    CHECK((child_names(tree.root) == std::vector<std::string>{ "a", "b", "c" }));
    CHECK((child_names(tree.a) == std::vector<std::string>{ "a1", "a2" }));
    CHECK((child_names(tree.c) == std::vector<std::string>{ "c1" }));
    CHECK(child_names(tree.b).empty());
    CHECK(!tree.b->GetFirstChild() && !tree.b->GetLastChild());
}

void test_walker_visits_in_preorder()
{
    SmallTree tree;

    std::vector<std::string> names;
    std::vector<int> depths;
    SceneGraphWalker walker(tree.root);
    while (walker)
    {
        names.push_back(walker->GetName());
        depths.push_back(walker.Next(true));
    }
    CHECK((names == std::vector<std::string>{ "root", "a", "a1", "a2", "b", "c", "c1" }));
    CHECK((depths == std::vector<int>{ 1, 1, 0, -1, 0, 1, -2 }));

    // A walker scoped to a subgraph stays inside it
    names.clear();
    for (SceneGraphWalker scoped(tree.a); scoped; scoped.Next(true))
        names.push_back(scoped->GetName());
    CHECK((names == std::vector<std::string>{ "a", "a1", "a2" }));

    // Skipping children moves to the next sibling
    SceneGraphWalker skipping(tree.a, tree.root);
    CHECK(skipping.Next(false) == 0);
    CHECK(skipping.Get() == tree.b.Get());
}

void test_detach_keeps_sibling_order()
{
    SmallTree tree;
    auto d = tree.graph->Attach(tree.root, make_node("d"));

    // middle
    auto detached = tree.graph->Detach(tree.b);
    CHECK(detached.Get() == tree.b.Get());
    CHECK(is_unlinked(tree.b));
    CHECK(!tree.b->GetGraph());
    CHECK((child_names(tree.root) == std::vector<std::string>{ "a", "c", "d" }));

    // first
    tree.graph->Detach(tree.a);
    CHECK(is_unlinked(tree.a));
    CHECK((child_names(tree.a) == std::vector<std::string>{ "a1", "a2" }));
    CHECK((child_names(tree.root) == std::vector<std::string>{ "c", "d" }));

    // last
    tree.graph->Detach(d);
    CHECK(is_unlinked(d));
    CHECK((child_names(tree.root) == std::vector<std::string>{ "c" }));

    // only
    tree.graph->Detach(tree.c);
    CHECK(child_names(tree.root).empty());
    CHECK(!tree.root->GetFirstChild() && !tree.root->GetLastChild());

    // A detached subgraph appends at the end when it is attached again
    tree.graph->Attach(tree.root, tree.c);
    tree.graph->Attach(tree.root, tree.a);
    CHECK((child_names(tree.root) == std::vector<std::string>{ "c", "a" }));
    CHECK((child_names(tree.a) == std::vector<std::string>{ "a1", "a2" }));
    CHECK(tree.a->GetGraph() == tree.graph.Get());
}

void test_detach_returns_the_only_reference()
{
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));

    // The child list holds the only reference to the node
    SceneGraphNode* raw = graph->Attach(graph->GetRootNode(), make_node("x")).Get();
    nvrhi::WeakPtr<SceneGraphNode> weak(raw);

    auto detached = graph->Detach(raw);
    CHECK(detached.Get() == raw);
    CHECK(weak.Lock());
    CHECK(is_unlinked(detached));

    detached.Reset();
    CHECK(!weak.Lock());
}

void test_external_reference_outlives_parent()
{
    SmallTree tree;
    auto heldA = tree.a;
    nvrhi::WeakPtr<SceneGraphNode> weakB(tree.b.Get());

    tree = SmallTree();   // drops the old graph and every reference except heldA
    CHECK(!weakB.Lock());

    // The held node becomes the root of its own subgraph, which is intact
    CHECK(is_unlinked(heldA));
    CHECK((child_names(heldA) == std::vector<std::string>{ "a1", "a2" }));
}

void test_attach_reparents_orphan()
{
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    auto p1 = make_node("p1");
    auto p2 = make_node("p2");
    graph->Attach(p1, make_node("x"));
    auto y = graph->Attach(p1, make_node("y"));
    graph->Attach(p1, make_node("z"));
    graph->Attach(p2, make_node("w"));

    // y is only referenced by p1's list and by `y`; moving it must keep it alive and unlink it from p1
    SceneGraphNode* raw = y.Get();
    y.Reset();
    graph->Attach(p2, raw);
    CHECK((child_names(p1) == std::vector<std::string>{ "x", "z" }));
    CHECK((child_names(p2) == std::vector<std::string>{ "w", "y" }));
}

void test_copy_attach_preserves_order()
{
    SmallTree source;
    auto target = MAKE_RC_OBJ_PTR(SceneGraph);
    target->SetRootNode(make_node("target"));

    // `a` belongs to another graph, so Attach makes a deep copy
    auto copy = target->Attach(target->GetRootNode(), source.a);
    CHECK(copy.Get() != source.a.Get());
    CHECK(copy->GetGraph() == target.Get());
    CHECK((child_names(copy) == std::vector<std::string>{ "a1", "a2" }));
    CHECK((child_names(source.a) == std::vector<std::string>{ "a1", "a2" }));

    // Copying the root of a graph keeps the order on every level
    auto rootCopy = target->Attach(target->GetRootNode(), source.root);
    CHECK((child_names(target->GetRootNode()) == std::vector<std::string>{ "a", "root" }));
    CHECK((child_names(rootCopy) == std::vector<std::string>{ "a", "b", "c" }));
    CHECK((child_names(rootCopy->GetFirstChild()) == std::vector<std::string>{ "a1", "a2" }));
}

// A copied node is linked under its copied parent before its transform is set, so the change propagates to
// the copied ancestors and Refresh does not skip the copied subgraph.
void test_copy_attach_refreshes_transforms()
{
    SmallTree source;
    source.a1->SetTranslation(dm::double3(1.0, 2.0, 3.0));
    source.graph->Refresh(0);   // clears the dirty flags of the source graph
    CHECK(all(source.a1->GetLocalToWorldTransform().m_translation == dm::double3(1.0, 2.0, 3.0)));

    auto target = MAKE_RC_OBJ_PTR(SceneGraph);
    target->SetRootNode(make_node("target"));
    target->Refresh(0);

    auto copy = target->Attach(target->GetRootNode(), source.a);
    SceneGraphNode* a1Copy = copy->GetFirstChild();
    CHECK(a1Copy && a1Copy->GetName() == "a1");
    CHECK((copy->GetDirtyFlags() & SceneGraphNode::DirtyFlags::SubgraphTransforms) != 0);

    target->Refresh(1);
    CHECK(all(a1Copy->GetLocalToWorldTransform().m_translation == dm::double3(1.0, 2.0, 3.0)));
}

// Every node of a copied subgraph is new, so Refresh must reach all of them, also the ones without a
// transform or a leaf of their own: they inherit the global transform of the target parent.
void test_copy_attach_refreshes_whole_copy()
{
    auto source = MAKE_RC_OBJ_PTR(SceneGraph);
    source->SetRootNode(make_node("root"));
    auto x = source->Attach(source->GetRootNode(), make_node("x"));
    auto y = source->Attach(x, make_node("y"));
    auto z = source->Attach(y, make_node("z"));
    source->Refresh(0);

    auto target = MAKE_RC_OBJ_PTR(SceneGraph);
    target->SetRootNode(make_node("target"));
    target->GetRootNode()->SetTranslation(dm::double3(5.0, 0.0, 0.0));
    target->Refresh(0);

    auto copy = target->Attach(target->GetRootNode(), x);
    SceneGraphNode* zCopy = copy->GetFirstChild()->GetFirstChild();
    CHECK(zCopy && zCopy->GetName() == "z");

    target->Refresh(1);
    CHECK(all(zCopy->GetLocalToWorldTransform().m_translation == dm::double3(5.0, 0.0, 0.0)));
}

// A subgraph assembled outside of any graph is refreshed completely once it is attached to one.
void test_orphan_subgraph_refreshes_after_attach()
{
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    graph->GetRootNode()->SetTranslation(dm::double3(5.0, 0.0, 0.0));
    graph->Refresh(0);

    auto p = make_node("p");
    auto q = make_node("q");
    auto r = make_node("r");
    r->SetTranslation(dm::double3(1.0, 0.0, 0.0));   // set while r has no parent yet
    graph->Attach(q, r);
    graph->Attach(p, q);
    graph->Attach(graph->GetRootNode(), p);

    graph->Refresh(1);
    CHECK(all(r->GetLocalToWorldTransform().m_translation == dm::double3(6.0, 0.0, 0.0)));
}

// Copying a subgraph under a node inside that same subgraph copies the subgraph as it was before the call.
void test_copy_attach_into_own_subgraph()
{
    SmallTree tree;

    auto copy = tree.graph->Attach(tree.a1, tree.a);
    CHECK(copy.Get() != tree.a.Get());
    CHECK((child_names(tree.a1) == std::vector<std::string>{ "a" }));
    CHECK((child_names(copy) == std::vector<std::string>{ "a1", "a2" }));
    CHECK(child_names(copy->GetFirstChild()).empty());
}

// Making a node that sits in an orphaned subgraph the root of a graph takes it out of its old parent.
void test_set_root_unlinks_from_orphan_parent()
{
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    auto p = make_node("p");
    auto x = graph->Attach(p, make_node("x"));
    graph->Attach(p, make_node("y"));

    graph->SetRootNode(x);
    CHECK(graph->GetRootNode() == x.Get());
    CHECK(is_unlinked(x));
    CHECK((child_names(p) == std::vector<std::string>{ "y" }));
}

void test_children_range()
{
    SmallTree tree;

    std::vector<std::string> names;
    for (SceneGraphNode* child : tree.root->GetChildren())
        names.push_back(child->GetName());
    CHECK((names == std::vector<std::string>{ "a", "b", "c" }));

    auto children = tree.root->GetChildren();
    CHECK(std::distance(children.begin(), children.end()) == 3);
    CHECK(tree.b->GetChildren().begin() == tree.b->GetChildren().end());
}

void test_find_node()
{
    SmallTree tree;

    CHECK(tree.graph->FindNode("/a/a2") == tree.a2.Get());
    CHECK(tree.graph->FindNode("/c/c1") == tree.c1.Get());
    CHECK(tree.graph->FindNode("/c/missing") == nullptr);
    CHECK(tree.graph->FindNode("../b", tree.a) == tree.b.Get());
}

int main(int, char**)
{
    try
    {
        test_wide_orphan_subgraph_destruction();
        test_wide_graph_destruction();
        test_append_preserves_insertion_order();
        test_walker_visits_in_preorder();
        test_detach_keeps_sibling_order();
        test_detach_returns_the_only_reference();
        test_external_reference_outlives_parent();
        test_attach_reparents_orphan();
        test_copy_attach_preserves_order();
        test_copy_attach_refreshes_transforms();
        test_copy_attach_refreshes_whole_copy();
        test_orphan_subgraph_refreshes_after_attach();
        test_copy_attach_into_own_subgraph();
        test_set_root_unlinks_from_orphan_parent();
        test_children_range();
        test_find_node();
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    return 0;
}
