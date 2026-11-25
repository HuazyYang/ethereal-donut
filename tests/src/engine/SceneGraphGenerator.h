// The scene generator and the per-frame edits of the scene-graph trace (test_scene_graph_trace.cpp describes them; plan
// step E0.3), shared with test_scene_graph_leaf_columns (E6.3). Changing what it draws or does changes the goldens.
#pragma once

#include <donut/engine/SceneGraph.h>
#include <donut/engine/KeyframeAnimation.h>
#include <donut/tests/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    // ------------------------------------------------------------------------------------------ generator

    // xorshift64*: the same sequence with every compiler and standard library.
    class Rng
    {
    public:
        explicit Rng(uint64_t seed)
            : m_State(seed ^ 0x9E3779B97F4A7C15ull)
        {
            if (m_State == 0)
                m_State = 1;
        }

        uint64_t Next()
        {
            m_State ^= m_State >> 12;
            m_State ^= m_State << 25;
            m_State ^= m_State >> 27;
            return m_State * 0x2545F4914F6CDD1Dull;
        }

        // [0, n) for n > 0
        uint32_t Below(size_t n) { return uint32_t((Next() >> 32) % uint64_t(n)); }
        bool Percent(uint32_t percent) { return Below(100) < percent; }
        // [0, 1), 53 random bits
        double Unit() { return double(Next() >> 11) * (1.0 / 9007199254740992.0); }
        double Range(double lo, double hi) { return lo + (hi - lo) * Unit(); }

    private:
        uint64_t m_State;
    };

    // ------------------------------------------------------------------------------------------ scenes

    struct Profile
    {
        uint32_t movingPercent;     // share of the built nodes that get a new transform on every frame
        uint32_t detach;            // per-frame chances of each edit, in percent
        uint32_t reattach;
        uint32_t setLeaf;
        uint32_t copy;
        uint32_t orphan;
        uint32_t leafNode;
        uint32_t setRoot;
        size_t copyLimit;           // the largest subgraph that is deep-copied
    };

    constexpr Profile c_SmallProfile = { 30, 50, 60, 50, 40, 40, 40, 15, 8 };
    constexpr Profile c_LargeProfile = { 10, 20, 30, 30, 15, 20, 20, 5, 64 };

    struct Scene
    {
        Scene(uint64_t seed, const Profile& sceneProfile)
            : rng(seed)
            , profile(sceneProfile)
        { }

        Rng rng;
        Profile profile;
        double scaleSpread = 0.5;   // scalings are drawn from [1 - spread, 1 + spread]
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
        std::vector<nvrhi::AutoPtr<MeshInfo>> skinPrototypes;
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> movers;
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> detached;   // detached subgraphs, attached again later
        size_t nodeLimit = 0;       // copies do not grow the graph beyond this
        uint32_t nextName = 0;
    };

    nvrhi::AutoPtr<SceneGraphNode> NewNode(Scene& s)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName("n" + std::to_string(s.nextName++));
        return node;
    }

    dm::double3 RandomVector(Rng& rng, double lo, double hi)
    {
        const double x = rng.Range(lo, hi);
        const double y = rng.Range(lo, hi);
        const double z = rng.Range(lo, hi);
        return dm::double3(x, y, z);
    }

    dm::dquat RandomRotation(Rng& rng)
    {
        const double x = rng.Range(-1.0, 1.0);
        const double y = rng.Range(-1.0, 1.0);
        const double z = rng.Range(-1.0, 1.0);
        const double w = rng.Range(-1.0, 1.0);
        const double length = std::sqrt(x * x + y * y + z * z + w * w);
        if (length < 1e-3)
            return dm::dquat::identity();
        return dm::dquat(w / length, x / length, y / length, z / length);
    }

    // New values through one of the setter forms; every form draws the same random numbers.
    void Move(Scene& s, SceneGraphNode* node)
    {
        const uint32_t form = s.rng.Below(5);
        const dm::double3 translation = RandomVector(s.rng, -8.0, 8.0);
        const dm::dquat rotation = RandomRotation(s.rng);
        const dm::double3 scaling = RandomVector(s.rng, 1.0 - s.scaleSpread, 1.0 + s.scaleSpread);
        switch (form)
        {
        case 0: node->SetTransform(&translation, &rotation, &scaling); break;
        case 1: node->SetTranslation(translation); break;
        case 2: node->SetRotation(rotation); break;
        case 3: node->SetScaling(scaling); break;
        default: node->SetTransform(&translation, nullptr, &scaling); break;
        }
    }

    dm::box3 RandomBounds(Rng& rng)
    {
        const float x0 = float(rng.Range(-2.0, -0.25));
        const float y0 = float(rng.Range(-2.0, -0.25));
        const float z0 = float(rng.Range(-2.0, -0.25));
        const float x1 = float(rng.Range(0.25, 2.0));
        const float y1 = float(rng.Range(0.25, 2.0));
        const float z1 = float(rng.Range(0.25, 2.0));
        return dm::box3(dm::float3(x0, y0, z0), dm::float3(x1, y1, z1));
    }

    nvrhi::AutoPtr<Material> NewMaterial(Scene& s, const char* name, MaterialDomain domain)
    {
        auto material = s.factory->CreateMaterial();
        material->name = name;
        material->domain = domain;
        return material;
    }

    // A mesh without device objects: geometries with materials (or none) and bounds.
    nvrhi::AutoPtr<MeshInfo> NewMesh(Scene& s, const char* name, std::initializer_list<Material*> materials)
    {
        auto mesh = s.factory->CreateMesh();
        mesh->name = name;
        mesh->objectSpaceBounds = dm::box3::empty();
        for (Material* material : materials)
        {
            auto geometry = s.factory->CreateMeshGeometry();
            geometry->material = material;
            geometry->objectSpaceBounds = RandomBounds(s.rng);
            mesh->objectSpaceBounds |= geometry->objectSpaceBounds;
            mesh->geometries.push_back(geometry);
        }
        return mesh;
    }

    void CreateAssets(Scene& s)
    {
        auto opaque = NewMaterial(s, "opaque", MaterialDomain::Opaque);
        auto alphaTested = NewMaterial(s, "alpha_tested", MaterialDomain::AlphaTested);
        auto blended = NewMaterial(s, "blended", MaterialDomain::AlphaBlended);
        auto transmissive = NewMaterial(s, "transmissive", MaterialDomain::Transmissive);

        s.meshes.push_back(NewMesh(s, "box", { opaque }));
        s.meshes.push_back(NewMesh(s, "two_part", { opaque, alphaTested }));
        s.meshes.push_back(NewMesh(s, "glass", { blended }));
        s.meshes.push_back(NewMesh(s, "window", { transmissive }));
        s.meshes.push_back(NewMesh(s, "empty", { }));           // no geometry: empty bounds, no content flags
        s.meshes.push_back(NewMesh(s, "unlit", { nullptr }));   // a geometry without a material
        s.skinPrototypes.push_back(NewMesh(s, "skin_a", { opaque }));
        s.skinPrototypes.push_back(NewMesh(s, "skin_b", { opaque, blended }));
    }

    enum class LeafKind
    {
        Mesh,
        Skinned,
        DirectionalLight,
        PointLight,
        SpotLight,
        PerspectiveCamera,
        OrthographicCamera
    };

    LeafKind RandomLeafKind(Rng& rng)
    {
        const uint32_t r = rng.Below(16);
        if (r < 8) return LeafKind::Mesh;
        if (r == 8) return LeafKind::Skinned;
        if (r == 9) return LeafKind::DirectionalLight;
        if (r == 10) return LeafKind::PointLight;
        if (r == 11) return LeafKind::SpotLight;
        if (r < 14) return LeafKind::PerspectiveCamera;
        return LeafKind::OrthographicCamera;
    }

    nvrhi::AutoPtr<SceneGraphLeaf> NewLeaf(Scene& s, LeafKind kind)
    {
        switch (kind)
        {
        case LeafKind::Mesh:
            return s.factory->CreateMeshInstance(s.meshes[s.rng.Below(s.meshes.size())]);
        case LeafKind::Skinned:
            // no joints: only the characters' instances have joints
            return s.factory->CreateSkinnedMeshInstance(s.factory,
                                                        s.skinPrototypes[s.rng.Below(s.skinPrototypes.size())]);
        case LeafKind::DirectionalLight:
            return MAKE_RC_OBJ_PTR(DirectionalLight);
        case LeafKind::PointLight:
            return MAKE_RC_OBJ_PTR(PointLight);
        case LeafKind::SpotLight:
            return MAKE_RC_OBJ_PTR(SpotLight);
        case LeafKind::PerspectiveCamera:
            return MAKE_RC_OBJ_PTR(PerspectiveCamera);
        case LeafKind::OrthographicCamera:
        default:
            return MAKE_RC_OBJ_PTR(OrthographicCamera);
        }
    }

    // Linear keyframes at 0, 1, 2 and 3 seconds; rotations stay in one hemisphere, so they never interpolate to zero.
    nvrhi::AutoPtr<animation::Sampler> NewSampler(Scene& s, AnimationAttribute attribute)
    {
        auto sampler = MAKE_RC_OBJ_PTR(animation::Sampler);
        sampler->SetInterpolationMode(animation::InterpolationMode::Linear);
        for (int k = 0; k < 4; k++)
        {
            animation::Keyframe key;
            key.time = float(k);
            if (attribute == AnimationAttribute::Rotation)
            {
                const float x = float(s.rng.Range(-0.5, 0.5));
                const float y = float(s.rng.Range(-0.5, 0.5));
                const float z = float(s.rng.Range(-0.5, 0.5));
                const float w = float(s.rng.Range(0.5, 1.0));
                key.value = dm::float4(x, y, z, w);
            }
            else
            {
                const bool scaling = attribute == AnimationAttribute::Scaling;
                const dm::double3 v = scaling ? RandomVector(s.rng, 0.8, 1.2) : RandomVector(s.rng, -1.0, 1.0);
                key.value = dm::float4(float(v.x), float(v.y), float(v.z), 0.f);
            }
            sampler->AddKeyframe(key);
        }
        return sampler;
    }

    void Track(Scene& s, SceneGraphNode* node)
    {
        if (s.rng.Percent(s.profile.movingPercent))
            s.movers.push_back(node);
    }

    // Children with transforms under `parent`, which may be in the graph or in an orphaned subgraph. Half of the leaves
    // are set before the node is attached and half after, so both registration paths run.
    void AddFlat(Scene& s, SceneGraphNode* parent, uint32_t count)
    {
        for (uint32_t i = 0; i < count; i++)
        {
            auto node = NewNode(s);
            Move(s, node);
            const LeafKind kind = (i % 10 == 9) ? RandomLeafKind(s.rng) : LeafKind::Mesh;
            auto leaf = NewLeaf(s, kind);
            if (i % 2 == 0)
                node->SetLeaf(leaf);
            s.graph->Attach(parent, node);
            if (i % 2 != 0)
                node->SetLeaf(leaf);
            Track(s, node);
        }
    }

    // A chain under `parent`; every third node has no transform of its own and takes its parent's world transform.
    void AddChain(Scene& s, SceneGraphNode* parent, uint32_t depth)
    {
        for (uint32_t i = 1; i <= depth; i++)
        {
            auto node = NewNode(s);
            if (i % 3 != 2)
                Move(s, node);
            if (i % 4 == 0)
                node->SetLeaf(NewLeaf(s, LeafKind::Mesh));
            else if (i % 25 == 0)
                node->SetLeaf(NewLeaf(s, LeafKind::PointLight));
            s.graph->Attach(parent, node);
            Track(s, node);
            parent = node;
        }
    }

    // A random tree under `top`: each node goes under one of the last few nodes (deep) or under any node (wide);
    // now and then an orphaned cluster is assembled off the graph and attached in one piece.
    void AddTree(Scene& s, SceneGraphNode* top, uint32_t count)
    {
        std::vector<SceneGraphNode*> nodes = { top };   // owned by the tree; nothing is detached while it is built
        while (nodes.size() <= count)
        {
            SceneGraphNode* parent = s.rng.Percent(50)
                ? nodes[nodes.size() - 1 - s.rng.Below(std::min<size_t>(8, nodes.size()))]
                : nodes[s.rng.Below(nodes.size())];

            auto node = NewNode(s);
            if (s.rng.Percent(75))
                Move(s, node);
            if (s.rng.Percent(60))
                node->SetLeaf(NewLeaf(s, RandomLeafKind(s.rng)));

            if (s.rng.Percent(5))
            {
                // an orphaned cluster under the new node
                const uint32_t clusterSize = 2 + s.rng.Below(8);
                std::vector<SceneGraphNode*> cluster = { node };
                for (uint32_t i = 0; i < clusterSize; i++)
                {
                    auto member = NewNode(s);
                    if (s.rng.Percent(75))
                        Move(s, member);
                    if (s.rng.Percent(60))
                        member->SetLeaf(NewLeaf(s, RandomLeafKind(s.rng)));
                    s.graph->Attach(cluster[s.rng.Below(cluster.size())], member);
                    cluster.push_back(member);
                }
                s.graph->Attach(parent, node);
                for (SceneGraphNode* member : cluster)
                {
                    nodes.push_back(member);
                    Track(s, member);
                }
                continue;
            }

            s.graph->Attach(parent, node);
            nodes.push_back(node);
            Track(s, node);
        }
    }

    // A skinned character as an orphaned subgraph, the way an importer builds one:
    //   character (T)
    //   +- joint 0 (T, SkinnedMeshReference) +- joint 1 ... +- joint n-1 (a chain)
    //   +- skin (SkinnedMeshInstance with the joints)
    //   +- animation (a translation, rotation or scaling channel for every joint after the first)
    nvrhi::AutoPtr<SceneGraphNode> NewCharacter(Scene& s, uint32_t jointCount)
    {
        auto character = NewNode(s);
        Move(s, character);
        auto skinned = s.factory->CreateSkinnedMeshInstance(s.factory,
                                                            s.skinPrototypes[s.rng.Below(s.skinPrototypes.size())]);

        std::vector<SceneGraphNode*> joints;
        SceneGraphNode* parent = character;
        for (uint32_t i = 0; i < jointCount; i++)
        {
            auto joint = NewNode(s);
            Move(s, joint);
            joint->SetLeaf(MAKE_RC_OBJ_PTR(SkinnedMeshReference, skinned.Get()));
            s.graph->Attach(parent, joint);
            skinned->joints.push_back(
                SkinnedMeshJoint{ nvrhi::WeakPtr<SceneGraphNode>(joint.Get()), dm::float4x4::identity() });
            joints.push_back(joint);
            parent = joint;
        }

        auto skin = NewNode(s);
        skin->SetLeaf(skinned);
        s.graph->Attach(character, skin);

        auto animation = MAKE_RC_OBJ_PTR(SceneGraphAnimation);
        for (uint32_t i = 1; i < jointCount; i++)
        {
            const AnimationAttribute attribute = (i % 3 == 1) ? AnimationAttribute::Translation
                : (i % 3 == 2) ? AnimationAttribute::Rotation
                : AnimationAttribute::Scaling;
            animation->AddChannel(
                MAKE_RC_OBJ_PTR(SceneGraphAnimationChannel, NewSampler(s, attribute), joints[i], attribute));
        }
        auto animationNode = NewNode(s);
        animationNode->SetLeaf(animation);
        s.graph->Attach(character, animationNode);

        return character;
    }

    void AddLightsAndCameras(Scene& s, SceneGraphNode* parent)
    {
        for (LeafKind kind : { LeafKind::DirectionalLight, LeafKind::PointLight, LeafKind::PerspectiveCamera })
        {
            auto node = s.graph->AttachLeafNode(parent, NewLeaf(s, kind));
            node->SetName("n" + std::to_string(s.nextName++));
            Move(s, node);
        }
    }

    // Flat instances attached one by one under a root without a transform.
    void BuildFlat(Scene& s, uint32_t size)
    {
        CreateAssets(s);
        s.graph->SetRootNode(NewNode(s));
        AddFlat(s, s.graph->GetRootNode(), size);
    }

    // A deep chain assembled as an orphaned subgraph and handed over with SetRootNode.
    void BuildChain(Scene& s, uint32_t size)
    {
        CreateAssets(s);
        s.scaleSpread = 0.1;
        auto root = NewNode(s);
        AddChain(s, root, size);
        s.graph->SetRootNode(root);
    }

    // A wide and deep tree built in the graph under a root with a transform.
    void BuildTree(Scene& s, uint32_t size)
    {
        CreateAssets(s);
        s.scaleSpread = 0.25;
        auto root = NewNode(s);
        Move(s, root);
        s.graph->SetRootNode(root);
        AddTree(s, root, size);
    }

    // Skinned characters attached to the graph one by one, with lights and a camera.
    void BuildCrowd(Scene& s, uint32_t size)
    {
        CreateAssets(s);
        s.graph->SetRootNode(NewNode(s));
        for (uint32_t i = 0; i < size; i++)
        {
            auto character = NewCharacter(s, 5);
            s.graph->Attach(s.graph->GetRootNode(), character);
            Track(s, character);
        }
        AddLightsAndCameras(s, s.graph->GetRootNode());
    }

    // All of the above: a flat group, a chain and a tree assembled off the graph under a transformed root that is
    // handed over with SetRootNode, then characters, lights and cameras attached in the graph.
    void BuildMixed(Scene& s, uint32_t size)
    {
        CreateAssets(s);
        s.scaleSpread = 0.25;
        auto root = NewNode(s);
        Move(s, root);

        auto flat = NewNode(s);   // a group node without a transform
        AddFlat(s, flat, size / 4);
        s.graph->Attach(root, flat);

        auto chain = NewNode(s);
        Move(s, chain);
        AddChain(s, chain, std::max(4u, size / 16));
        s.graph->Attach(root, chain);

        auto tree = NewNode(s);
        AddTree(s, tree, size / 4);
        s.graph->Attach(root, tree);

        s.graph->SetRootNode(root);

        for (uint32_t i = 0; i < std::max(1u, size / 100); i++)
        {
            auto character = NewCharacter(s, 4);
            s.graph->Attach(root, character);
            Track(s, character);
        }
        AddLightsAndCameras(s, root);
    }

    // ------------------------------------------------------------------------------------------ edits

    bool g_Verbose = false;   // ECS_TRACE_VERBOSE=1 prints the edits; they are not part of the records

    const char* Name(const SceneGraphNode* node)
    {
        return node->GetName().c_str();
    }

    std::vector<SceneGraphNode*> CollectNodes(SceneGraph* graph)
    {
        std::vector<SceneGraphNode*> nodes;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
            nodes.push_back(walker.Get());
        return nodes;
    }

    size_t SubgraphSize(SceneGraphNode* node)
    {
        size_t size = 0;
        for (SceneGraphWalker walker(node); walker; walker.Next(true))
            ++size;
        return size;
    }

    void EditDetach(Scene& s)
    {
        const auto nodes = CollectNodes(s.graph);
        if (nodes.size() < 2)
            return;
        SceneGraphNode* node = nodes[1 + s.rng.Below(nodes.size() - 1)];
        if (g_Verbose)
            std::printf("    detach %s\n", Name(node));
        s.detached.push_back(s.graph->Detach(node));
    }

    // A detached subgraph moves back in (it is orphaned, so it is not copied).
    void EditReattach(Scene& s)
    {
        if (s.detached.empty())
            return;
        const size_t which = s.rng.Below(s.detached.size());
        const auto nodes = CollectNodes(s.graph);
        SceneGraphNode* parent = nodes[s.rng.Below(nodes.size())];
        if (g_Verbose)
            std::printf("    attach %s under %s\n", Name(s.detached[which]), Name(parent));
        s.graph->Attach(parent, s.detached[which]);
        s.detached.erase(s.detached.begin() + ptrdiff_t(which));
    }

    void EditSetLeaf(Scene& s)
    {
        const auto nodes = CollectNodes(s.graph);
        SceneGraphNode* node = nodes[s.rng.Below(nodes.size())];
        const LeafKind kind = RandomLeafKind(s.rng);
        if (g_Verbose)
            std::printf("    set leaf %d on %s\n", int(kind), Name(node));
        node->SetLeaf(NewLeaf(s, kind));
    }

    // Attaching a node that is in the graph copies its subgraph; the parent may lie inside that subgraph.
    void EditCopy(Scene& s)
    {
        const auto nodes = CollectNodes(s.graph);
        SceneGraphNode* source = nodes[s.rng.Below(nodes.size())];
        SceneGraphNode* parent = nodes[s.rng.Below(nodes.size())];
        const size_t size = SubgraphSize(source);
        if (size > s.profile.copyLimit || nodes.size() + size > s.nodeLimit)
            return;
        if (g_Verbose)
            std::printf("    copy %s (%zu nodes) under %s\n", Name(source), size, Name(parent));
        s.graph->Attach(parent, source);
    }

    // A small subgraph assembled off the graph, with transforms set before and after linking, then attached.
    void EditOrphan(Scene& s)
    {
        auto top = NewNode(s);
        if (s.rng.Percent(50))
            Move(s, top);
        std::vector<SceneGraphNode*> built = { top };
        const uint32_t count = 1 + s.rng.Below(5);
        for (uint32_t i = 0; i < count; i++)
        {
            auto node = NewNode(s);
            const bool before = s.rng.Percent(50);
            if (before)
                Move(s, node);
            if (s.rng.Percent(50))
                node->SetLeaf(NewLeaf(s, RandomLeafKind(s.rng)));
            s.graph->Attach(built[s.rng.Below(built.size())], node);
            if (!before && s.rng.Percent(50))
                Move(s, node);
            Track(s, node);
            built.push_back(node);
        }
        const auto nodes = CollectNodes(s.graph);
        SceneGraphNode* parent = nodes[s.rng.Below(nodes.size())];
        if (g_Verbose)
            std::printf("    orphan %s (%zu nodes) under %s\n", Name(top), built.size(), Name(parent));
        s.graph->Attach(parent, top);
    }

    // AttachLeafNode with a new leaf, or with a leaf that is attached already (which is cloned).
    void EditLeafNode(Scene& s)
    {
        const auto nodes = CollectNodes(s.graph);
        SceneGraphNode* parent = nodes[s.rng.Below(nodes.size())];
        nvrhi::AutoPtr<SceneGraphNode> node;
        if (s.rng.Percent(50))
        {
            SceneGraphNode* holder = nodes[s.rng.Below(nodes.size())];
            if (SceneGraphLeaf* leaf = holder->GetLeaf())
                node = s.graph->AttachLeafNode(parent, leaf);
        }
        if (!node)
        {
            const LeafKind kind = RandomLeafKind(s.rng);
            node = s.graph->AttachLeafNode(parent, NewLeaf(s, kind));
        }
        node->SetName("n" + std::to_string(s.nextName++));
        if (g_Verbose)
            std::printf("    leaf node %s under %s\n", Name(node), Name(parent));
    }

    void EditSetRoot(Scene& s)
    {
        const auto nodes = CollectNodes(s.graph);
        if (nodes.size() >= 2 && s.rng.Percent(50))
        {
            // a node inside the graph becomes the root, and the rest of the old graph goes under it
            SceneGraphNode* node = nodes[1 + s.rng.Below(nodes.size() - 1)];
            if (g_Verbose)
                std::printf("    root %s, the old root %s under it\n", Name(node), Name(nodes[0]));
            auto oldRoot = s.graph->SetRootNode(node);
            s.graph->Attach(node, oldRoot);
        }
        else
        {
            // a new root, with the old graph under it
            auto root = NewNode(s);
            if (s.rng.Percent(50))
                Move(s, root);
            if (g_Verbose)
                std::printf("    new root %s, the old root %s under it\n", Name(root), Name(nodes[0]));
            auto oldRoot = s.graph->SetRootNode(root);
            s.graph->Attach(root, oldRoot);
        }
    }

    // Frames 1 and 2 run every edit, so that the full records of the small scenes cover all of them.
    void Step(Scene& s, uint32_t frame)
    {
        if (g_Verbose)
            std::printf("  frame %u\n", frame);
        if (frame % 8 == 7)
            return;   // nothing changes

        for (const auto& node : s.movers)
            Move(s, node);

        const auto animations = s.graph->GetAnimations();
        for (const auto& animation : animations)
            animation->Apply(float(frame) * 0.25f);

        const Profile& p = s.profile;
        const bool first = frame == 1;
        const bool second = frame == 2;
        if (first || s.rng.Percent(p.detach))
            EditDetach(s);
        if (first || s.rng.Percent(p.setLeaf))
            EditSetLeaf(s);
        if (first || s.rng.Percent(p.orphan))
            EditOrphan(s);
        if (first || s.rng.Percent(p.leafNode))
            EditLeafNode(s);
        if (second || s.rng.Percent(p.reattach))
            EditReattach(s);
        if (second || s.rng.Percent(p.copy))
            EditCopy(s);
        if ((second && p.setRoot != 0) || s.rng.Percent(p.setRoot))
            EditSetRoot(s);
    }
}
