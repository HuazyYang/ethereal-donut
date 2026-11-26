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

// The render service's extraction without a device (aggregate docs/plans/2026-11-21-donut-ecs-e5.md, item E5.1;
// docs/ecs-design.md 7.1, 7.4, 7.5, D8): the frame snapshot of the E0 trace scenes (rebuilt through the public API with
// the trace test's generator and edit script) and of a 4,096-instance scene against a brute-force walk of the graph
// (views, lights in GetLights() order with LightConstants byte-equal to FillLightConstants, cameras, every mesh
// instance with its slots, mesh and float world bounds, skinned instances with the legacy selection and joint
// matrices); isolation (writes between Extract and EndFrame leave the snapshot's bytes unchanged and warn once each in
// Debug); equal snapshots at 1, 2 and 8 workers and no scheduler ambiguity; the frame arena reset; no heap allocation
// in steady state; one service per graph; EndFrame idempotent, with first-frame skinning flags set only for executed
// frames; an extraction does not make the next Refresh run (SceneGraph::RunReadOnlyPhase); the service's
// shadow table; the review fixes (exclusive and command-recording systems in the read-only phase, arena failures,
// destroyed joints, a shadow map replaced mid-frame, the old scheduler released, repeated calls in one frame); plan step
// E7, item E7.2: runtime-only mesh instances (SceneComponents.h, RuntimeMeshInstance) beside nodes, taken by the
// brute force from the world (no leaf; the mesh's box under the row's world transform), with the Debug leaf check
// passing over their chunks.
//
//     test_render_extract [case]      runs every case, or the one named; `measure` (by name only) prints timings

#include <donut/render/RenderExtract.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/LightShadows.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/KeyframeAnimation.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include <donut/core/query_cast.h>
#include <donut/tests/utils.h>

#include "CountingJobSystem.h"
#include "EcsWorldImpl.h"       // ecs::GetHeapAllocationCount
#include "RenderVisibility.h"   // the limits of the serial extraction

using namespace donut::math;
#include <donut/shaders/light_cb.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <malloc.h>
#include <stdlib.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

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

// Every operator new is counted, the frame arena's blocks included (FrameArena allocates them with operator new).
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
using namespace donut::render;
using donut::tests::CountingJobSystem;
using donut::tests::ExtractionMode;

namespace
{
    // Warnings logged (log::warning and above); each case says how many it expects.
    std::atomic<uint32_t> g_Warnings{ 0 };

    // The trace test's generator and edit script (tests/src/engine/test_scene_graph_trace.cpp), copied: the same seeds
    // build the same scenes through the public API.
    bool g_Verbose = false;

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

    struct TraceScene
    {
        TraceScene(uint64_t seed, const Profile& sceneProfile)
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

    nvrhi::AutoPtr<SceneGraphNode> NewNode(TraceScene& s)
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
    void Move(TraceScene& s, SceneGraphNode* node)
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

    nvrhi::AutoPtr<Material> NewMaterial(TraceScene& s, const char* name, MaterialDomain domain)
    {
        auto material = s.factory->CreateMaterial();
        material->name = name;
        material->domain = domain;
        return material;
    }

    // A mesh without device objects: geometries with materials (or none) and bounds.
    nvrhi::AutoPtr<MeshInfo> NewMesh(TraceScene& s, const char* name, std::initializer_list<Material*> materials)
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

    void CreateAssets(TraceScene& s)
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

    nvrhi::AutoPtr<SceneGraphLeaf> NewLeaf(TraceScene& s, LeafKind kind)
    {
        switch (kind)
        {
        case LeafKind::Mesh:
            return s.factory->CreateMeshInstance(s.meshes[s.rng.Below(s.meshes.size())]);
        case LeafKind::Skinned:
            // no joints: only the characters' instances have joints
            return s.factory->CreateSkinnedMeshInstance(s.factory, s.skinPrototypes[s.rng.Below(s.skinPrototypes.size())]);
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
    nvrhi::AutoPtr<animation::Sampler> NewSampler(TraceScene& s, AnimationAttribute attribute)
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

    void Track(TraceScene& s, SceneGraphNode* node)
    {
        if (s.rng.Percent(s.profile.movingPercent))
            s.movers.push_back(node);
    }

    // Children with transforms under `parent`, which may be in the graph or in an orphaned subgraph. Half of the leaves
    // are set before the node is attached and half after, so both registration paths run.
    void AddFlat(TraceScene& s, SceneGraphNode* parent, uint32_t count)
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
    void AddChain(TraceScene& s, SceneGraphNode* parent, uint32_t depth)
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
    void AddTree(TraceScene& s, SceneGraphNode* top, uint32_t count)
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
    nvrhi::AutoPtr<SceneGraphNode> NewCharacter(TraceScene& s, uint32_t jointCount)
    {
        auto character = NewNode(s);
        Move(s, character);
        auto skinned = s.factory->CreateSkinnedMeshInstance(s.factory, s.skinPrototypes[s.rng.Below(s.skinPrototypes.size())]);

        std::vector<SceneGraphNode*> joints;
        SceneGraphNode* parent = character;
        for (uint32_t i = 0; i < jointCount; i++)
        {
            auto joint = NewNode(s);
            Move(s, joint);
            joint->SetLeaf(MAKE_RC_OBJ_PTR(SkinnedMeshReference, skinned.Get()));
            s.graph->Attach(parent, joint);
            skinned->joints.push_back(SkinnedMeshJoint{ nvrhi::WeakPtr<SceneGraphNode>(joint.Get()), dm::float4x4::identity() });
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
            animation->AddChannel(MAKE_RC_OBJ_PTR(SceneGraphAnimationChannel, NewSampler(s, attribute), joints[i], attribute));
        }
        auto animationNode = NewNode(s);
        animationNode->SetLeaf(animation);
        s.graph->Attach(character, animationNode);

        return character;
    }

    void AddLightsAndCameras(TraceScene& s, SceneGraphNode* parent)
    {
        for (LeafKind kind : { LeafKind::DirectionalLight, LeafKind::PointLight, LeafKind::PerspectiveCamera })
        {
            auto node = s.graph->AttachLeafNode(parent, NewLeaf(s, kind));
            node->SetName("n" + std::to_string(s.nextName++));
            Move(s, node);
        }
    }

    // Flat instances attached one by one under a root without a transform.
    void BuildFlat(TraceScene& s, uint32_t size)
    {
        CreateAssets(s);
        s.graph->SetRootNode(NewNode(s));
        AddFlat(s, s.graph->GetRootNode(), size);
    }

    // A deep chain assembled as an orphaned subgraph and handed over with SetRootNode.
    void BuildChain(TraceScene& s, uint32_t size)
    {
        CreateAssets(s);
        s.scaleSpread = 0.1;
        auto root = NewNode(s);
        AddChain(s, root, size);
        s.graph->SetRootNode(root);
    }

    // A wide and deep tree built in the graph under a root with a transform.
    void BuildTree(TraceScene& s, uint32_t size)
    {
        CreateAssets(s);
        s.scaleSpread = 0.25;
        auto root = NewNode(s);
        Move(s, root);
        s.graph->SetRootNode(root);
        AddTree(s, root, size);
    }

    // Skinned characters attached to the graph one by one, with lights and a camera.
    void BuildCrowd(TraceScene& s, uint32_t size)
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
    void BuildMixed(TraceScene& s, uint32_t size)
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

    // Mesh instances under a root without a transform, then two skinned characters (4,096 instances for size 4,094),
    // lights and a camera.
    void BuildInstances(TraceScene& s, uint32_t size)
    {
        CreateAssets(s);
        s.graph->SetRootNode(NewNode(s));
        SceneGraphNode* root = s.graph->GetRootNode();
        for (uint32_t i = 0; i < size; i++)
        {
            auto node = NewNode(s);
            Move(s, node);
            node->SetLeaf(NewLeaf(s, LeafKind::Mesh));
            s.graph->Attach(root, node);
            Track(s, node);
        }
        for (uint32_t i = 0; i < 2; i++)
        {
            auto character = NewCharacter(s, 5);
            s.graph->Attach(root, character);
            Track(s, character);
        }
        AddLightsAndCameras(s, root);
    }

    // ------------------------------------------------------------------------------------------ edits


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

    void EditDetach(TraceScene& s)
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
    void EditReattach(TraceScene& s)
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

    void EditSetLeaf(TraceScene& s)
    {
        const auto nodes = CollectNodes(s.graph);
        SceneGraphNode* node = nodes[s.rng.Below(nodes.size())];
        const LeafKind kind = RandomLeafKind(s.rng);
        if (g_Verbose)
            std::printf("    set leaf %d on %s\n", int(kind), Name(node));
        node->SetLeaf(NewLeaf(s, kind));
    }

    // Attaching a node that is in the graph copies its subgraph; the parent may lie inside that subgraph.
    void EditCopy(TraceScene& s)
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
    void EditOrphan(TraceScene& s)
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
    void EditLeafNode(TraceScene& s)
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

    void EditSetRoot(TraceScene& s)
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
    void Step(TraceScene& s, uint32_t frame)
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

    // ------------------------------------------------------------------------------------------ fixtures

    // A device-free render graph and its blackboard, reset per frame as RenderGraphLayer does.
    struct RenderFrame
    {
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;
        FrameData data;

        RenderFrame()
        {
            CHECK(CreateResourceRegistry(nullptr, &registry) == nvrhi::FS_OK);
            CHECK(CreateRenderGraph(registry, &graph) == nvrhi::FS_OK);
        }

        void Begin()
        {
            graph->Reset();
            data.Clear();
        }
    };

    // A perspective view and an orthographic one (a cascade-like view), planar.
    struct TestViews
    {
        nvrhi::AutoPtr<PlanarView> perspective = MAKE_RC_OBJ_PTR(PlanarView);
        nvrhi::AutoPtr<PlanarView> orthographic = MAKE_RC_OBJ_PTR(PlanarView);
        const IView* list[2] = {};

        TestViews()
        {
            perspective->SetViewport(nvrhi::Viewport(1280.f, 720.f));
            const dm::affine3 eye = dm::translation(dm::float3(0.f, 2.f, -20.f));
            const dm::float4x4 projection = dm::perspProjD3DStyleReverse(dm::radians(60.f), 16.f / 9.f, 0.1f);
            perspective->SetMatrices(dm::inverse(eye), projection);
            perspective->UpdateCache();
            orthographic->SetViewport(nvrhi::Viewport(1024.f, 1024.f));
            const dm::affine3 sun = dm::lookatZ(dm::normalize(dm::float3(0.3f, -1.f, 0.2f))) *
                                    dm::translation(dm::float3(0.f, 50.f, 0.f));
            orthographic->SetMatrices(dm::inverse(sun), dm::orthoProjD3DStyle(-40.f, 40.f, -40.f, 40.f, 0.f, 100.f));
            orthographic->UpdateCache();
            list[0] = perspective.Get();
            list[1] = orthographic.Get();
        }

        ExtractDesc Desc(uint32_t frameIndex) const
        {
            ExtractDesc desc;
            desc.views = list;
            desc.viewCount = 2;
            desc.frameIndex = frameIndex;
            return desc;
        }
    };

    nvrhi::AutoPtr<RenderService> MakeService(SceneGraph* graph)
    {
        RenderServiceDesc desc;
        desc.sceneGraph = graph;
        return MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
    }

    // ------------------------------------------------------------------------------------------ snapshot hash

    struct Hasher
    {
        uint64_t value = 0xcbf29ce484222325ull;

        void Bytes(const void* data, size_t size)
        {
            const auto* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; ++i)
            {
                value ^= bytes[i];
                value *= 0x100000001b3ull;
            }
        }
        template <typename T>
        void Value(const T& v) { Bytes(&v, sizeof(v)); }
        template <typename T>
        void Array(const T* data, uint32_t count)
        {
            Value(count);
            if (count)
                Bytes(data, sizeof(T) * count);
        }
    };

    // Every byte the snapshot holds, field by field (the structs' padding is not initialised). `header`: the graph
    // frame index and the world version too, which move with every extraction.
    uint64_t HashSnapshot(const RenderSnapshot& s, bool header)
    {
        Hasher h;
        if (header)
        {
            h.Value(s.graphFrameIndex);
            h.Value(s.worldVersion);
        }
        h.Value(s.frameIndex);
        h.Value(s.renderOrigin);
        h.Value(s.viewCount);
        h.Value(s.descViewCount);
        for (uint32_t v = 0; v < s.viewCount; ++v)
        {
            const ExtractedView& view = s.views[v];
            h.Value(view.view);
            h.Value(view.frustum);
            h.Value(view.viewOrigin);
            h.Value(view.instances);
            for (const SnapshotRange& range : view.packets)
                h.Value(range);
            h.Value(view.light);
            h.Value(view.cascade);
        }
        const VisibleInstances& in = s.instances;
        h.Array(in.entities, in.count);
        h.Array(in.instanceSlots, in.count);
        h.Array(in.meshes, in.count);
        h.Array(in.firstGeometryInstanceSlots, in.count);
        h.Array(in.geometryCounts, in.count);
        h.Array(in.leaves, in.count);
        h.Array(in.meshInfos, in.count);
        h.Array(in.worldBounds, in.count);
        const DrawPackets& packets = s.packets;
        h.Array(packets.keys, packets.count);
        h.Array(packets.visibleInstances, packets.count);
        h.Array(packets.geometryIndices, packets.count);
        h.Array(packets.geometrySlots, packets.count);
        h.Array(packets.materialSlots, packets.count);
        h.Array(packets.cullModes, packets.count);
        h.Array(packets.flags, packets.count);
        h.Value(s.lightCount);
        for (uint32_t i = 0; i < s.lightCount; ++i)
        {
            const ExtractedLight& light = s.lights[i];
            h.Bytes(light.constants, sizeof(LightConstants));
            h.Value(light.entity);
            h.Value(light.leaf);
            h.Value(light.shadowMap);
            h.Value(light.shadowTexture);
            h.Value(light.serviceShadow);
        }
        h.Value(s.cameraCount);
        for (uint32_t i = 0; i < s.cameraCount; ++i)
        {
            h.Value(s.cameras[i].entity);
            h.Value(s.cameras[i].leaf);
            h.Value(s.cameras[i].viewToWorld);
        }
        h.Value(s.skinnedInstanceCount);
        for (uint32_t i = 0; i < s.skinnedInstanceCount; ++i)
        {
            const ExtractedSkinnedInstance& skinned = s.skinnedInstances[i];
            h.Value(skinned.entity);
            h.Value(skinned.leaf);
            h.Value(skinned.jointMatrices);
            h.Value(skinned.firstFrame);
        }
        h.Array(s.jointMatrices, s.jointMatrixCount);
        return h.value;
    }

    // ------------------------------------------------------------------------------------------ brute force

    struct InstanceRow
    {
        Entity entity;
        uint32_t slot;
        MeshId mesh;
        uint32_t firstGeometryInstance;
        uint32_t geometryCount;
        const MeshInstance* leaf;
        const MeshInfo* meshInfo;
        dm::box3 bounds;
    };

    bool SameBits(const void* a, const void* b, size_t size)
    {
        return std::memcmp(a, b, size) == 0;
    }

    // The leaves a test expects pending from a frame that did not execute (E5.5 review fix 1); empty elsewhere.
    std::vector<const SkinnedMeshInstance*> g_ExpectedPendingSkins;

    // The legacy skinning rule and formula (Scene::UpdateSkinnedMeshes), through the node getters, with the service's
    // additions (E5.5 review fix 1): an instance never skinned, and one pending from a frame that did not execute.
    bool LegacySelected(const SkinnedMeshInstance* instance, uint32_t frameIndex)
    {
        return !(instance->GetLastUpdateFrameIndex() + 1 < frameIndex) || !instance->skinningInitialized ||
               std::find(g_ExpectedPendingSkins.begin(), g_ExpectedPendingSkins.end(), instance) !=
                   g_ExpectedPendingSkins.end();
    }

    dm::float4x4 LegacyJointMatrix(const SkinnedMeshInstance* instance, size_t joint)
    {
        const dm::daffine3 worldToRoot = dm::inverse(instance->GetNode()->GetLocalToWorldTransform());
        auto jointNode = instance->joints[joint].node.Lock();
        if (!jointNode)
            return dm::float4x4::identity();   // the service's choice where the legacy path dereferenced null
        const dm::daffine3 jointToRoot = jointNode->GetLocalToWorldTransform() * worldToRoot;
        dm::float4x4 jointMatrix = dm::affineToHomogeneous(dm::affine3(jointToRoot));
        jointMatrix = instance->joints[joint].inverseBindMatrix * jointMatrix;
        return jointMatrix;
    }

    // The snapshot against the graph walked directly. Call it before EndFrame (the first-frame flags).
    void CheckSnapshot(SceneGraph* graph, const RenderSnapshot& s, const ExtractDesc& desc)
    {
        CHECK(s.frameIndex == desc.frameIndex);
        CHECK(s.renderOrigin.x == 0.0 && s.renderOrigin.y == 0.0 && s.renderOrigin.z == 0.0);

        // views, in the desc's order
        CHECK(s.viewCount == desc.viewCount && s.descViewCount == desc.viewCount);
        for (uint32_t v = 0; v < s.viewCount; ++v)
        {
            const ExtractedView& view = s.views[v];
            const dm::frustum frustum = desc.views[v]->GetViewFrustum();
            const dm::float3 origin = desc.views[v]->GetViewOrigin();
            CHECK(view.view == desc.views[v]);
            CHECK(SameBits(&view.frustum, &frustum, sizeof(frustum)));
            CHECK(SameBits(&view.viewOrigin, &origin, sizeof(origin)));
            // E5.2: each view its own range of visible instances, then per pass its range of packets, in order
            const SnapshotRange previousInstances = v != 0 ? s.views[v - 1].instances : SnapshotRange{};
            CHECK(view.instances.first == previousInstances.first + previousInstances.count);
            for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
            {
                const SnapshotRange& previous = p != 0 ? view.packets[p - 1] : v != 0
                    ? s.views[v - 1].packets[c_DrawPacketPassCount - 1] : SnapshotRange{};
                CHECK(view.packets[p].first == previous.first + previous.count);
            }
            CHECK(view.light == c_NoExtractedLight);
        }
        if (s.viewCount != 0)
        {
            const ExtractedView& last = s.views[s.viewCount - 1];
            CHECK(last.instances.first + last.instances.count == s.instances.count);
            const SnapshotRange& lastPackets = last.packets[c_DrawPacketPassCount - 1];
            CHECK(lastPackets.first + lastPackets.count == s.packets.count);
        }

        // lights in GetLights() order, their constants byte-equal to FillLightConstants called directly
        const auto& lights = graph->GetLights();
        CHECK(s.lightCount == lights.size());
        for (uint32_t i = 0; i < s.lightCount; ++i)
        {
            const ExtractedLight& extracted = s.lights[i];
            const Light* light = lights[i].Get();
            LightConstants expected;
            std::memset(&expected, 0, sizeof(expected));
            light->FillLightConstants(expected);
            CHECK(extracted.constants == s.lightConstants + i);
            CHECK(SameBits(extracted.constants, &expected, sizeof(expected)));
            CHECK(extracted.leaf == light);
            CHECK(extracted.entity != Entity::Null && extracted.entity == graph->GetEntity(light->GetNode()));
            CHECK(extracted.shadowMap == light->shadowMap.Get());
            CHECK(!extracted.serviceShadow && extracted.shadowTexture == InvalidResourceId);
        }

        // cameras in GetCameras() order
        const auto& cameras = graph->GetCameras();
        CHECK(s.cameraCount == cameras.size());
        for (uint32_t i = 0; i < s.cameraCount; ++i)
        {
            const dm::affine3 expected = cameras[i]->GetViewToWorldMatrix();
            CHECK(s.cameras[i].leaf == cameras[i].Get());
            CHECK(s.cameras[i].entity == graph->GetEntity(cameras[i]->GetNode()));
            CHECK(SameBits(&s.cameras[i].viewToWorld, &expected, sizeof(expected)));
        }

        // per view, every mesh instance with a slot whose own box passes the view's frustum test (the legacy test): a
        // pre-order walk against the view's range, as sets keyed by entity (test_draw_packets checks the packets)
        std::vector<InstanceRow> all;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
        {
            SceneGraphNode* node = walker.Get();
            const MeshInstance* instance = query_cast<MeshInstance>(node->GetLeaf());
            if (!instance || instance->GetInstanceIndex() < 0)
                continue;
            const MeshInfo* mesh = instance->GetMesh();
            const dm::box3 local = mesh->objectSpaceBounds;
            InstanceRow row;
            row.entity = graph->GetEntity(node);
            row.slot = uint32_t(instance->GetInstanceIndex());
            row.mesh = graph->GetMeshes().Find(mesh);
            row.firstGeometryInstance = uint32_t(instance->GetGeometryInstanceIndex());
            row.geometryCount = uint32_t(mesh->geometries.size());
            row.leaf = instance;
            row.meshInfo = mesh;
            row.bounds = local.isempty() ? dm::box3::empty() : local * node->GetLocalToWorldTransformFloat();
            all.push_back(row);
        }
        // and every runtime-only row of the graph's world with a slot (plan step E7, item E7.2): no leaf, the mesh's
        // box under the row's world transform in float, as the node's
        {
            Query<Read<GpuInstanceSlot>, Read<MeshRef>, Read<WorldTransform>, With<RuntimeMeshInstance>,
                  Without<SceneNode>> rows(graph->GetWorld(), "CheckSnapshot.RuntimeRows");
            rows.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, const GpuInstanceSlot* slots,
                                                  const MeshRef* refs, const WorldTransform* worlds)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                {
                    if (slots[r].instance == c_NoInstanceSlot)
                        continue;
                    const MeshInfo* mesh = graph->GetMeshes().Get(refs[r].mesh);
                    CHECK(mesh != nullptr);
                    const dm::box3 local = mesh->objectSpaceBounds;
                    InstanceRow row;
                    row.entity = view.entities[r];
                    row.slot = slots[r].instance;
                    row.mesh = refs[r].mesh;
                    row.firstGeometryInstance = slots[r].firstGeometryInstance;
                    row.geometryCount = uint32_t(mesh->geometries.size());
                    CHECK(slots[r].geometryCount == row.geometryCount);
                    row.leaf = nullptr;
                    row.meshInfo = mesh;
                    row.bounds = local.isempty() ? dm::box3::empty() : local * dm::affine3(worlds[r].transform);
                    all.push_back(row);
                }
            });
        }
        auto byEntity = [](const InstanceRow& a, const InstanceRow& b) { return a.entity < b.entity; };
        const VisibleInstances& in = s.instances;
        for (uint32_t v = 0; v < s.viewCount; ++v)
        {
            const dm::frustum frustum = desc.views[v]->GetViewFrustum();
            std::vector<InstanceRow> expected;
            for (const InstanceRow& row : all)
            {
                if (frustum.intersectsWith(row.bounds))
                    expected.push_back(row);
            }
            std::vector<InstanceRow> actual;
            const SnapshotRange range = s.views[v].instances;
            for (uint32_t i = range.first; i < range.first + range.count; ++i)
            {
                actual.push_back(InstanceRow{ in.entities[i], in.instanceSlots[i], in.meshes[i],
                                              in.firstGeometryInstanceSlots[i], in.geometryCounts[i], in.leaves[i],
                                              in.meshInfos[i], in.worldBounds[i] });
            }
            std::sort(expected.begin(), expected.end(), byEntity);
            std::sort(actual.begin(), actual.end(), byEntity);
            CHECK(expected.size() == actual.size());
            for (size_t i = 0; i < expected.size() && i < actual.size(); ++i)
            {
                const InstanceRow& e = expected[i];
                const InstanceRow& a = actual[i];
                CHECK(e.entity == a.entity && e.entity != Entity::Null);
                CHECK(e.slot == a.slot && e.mesh == a.mesh && e.mesh != MeshId::Invalid);
                CHECK(e.firstGeometryInstance == a.firstGeometryInstance && e.geometryCount == a.geometryCount);
                CHECK(e.leaf == a.leaf && e.meshInfo == a.meshInfo);
                CHECK(SameBits(&e.bounds, &a.bounds, sizeof(e.bounds)));
            }
        }

        // skinned instances: the legacy selection over the graph's vector, the legacy joint matrices bit for bit
        uint32_t selected = 0;
        for (const auto& instance : graph->GetSkinnedMeshInstances())
        {
            if (LegacySelected(instance, desc.frameIndex))
                ++selected;
        }
        CHECK(s.skinnedInstanceCount == selected);
        uint32_t joints = 0;
        for (uint32_t i = 0; i < s.skinnedInstanceCount; ++i)
        {
            const ExtractedSkinnedInstance& extracted = s.skinnedInstances[i];
            const SkinnedMeshInstance* instance = extracted.leaf;
            const auto& skinnedLeaves = graph->GetSkinnedMeshInstances();
            CHECK(std::find(skinnedLeaves.begin(), skinnedLeaves.end(), instance) != skinnedLeaves.end());
            CHECK(LegacySelected(instance, desc.frameIndex));
            CHECK(extracted.entity == graph->GetEntity(instance->GetNode()));
            CHECK(extracted.firstFrame == !instance->skinningInitialized);
            CHECK(extracted.jointMatrices.first == joints && extracted.jointMatrices.count == instance->joints.size());
            for (uint32_t j = 0; j < extracted.jointMatrices.count; ++j)
            {
                const dm::float4x4 legacy = LegacyJointMatrix(instance, j);
                CHECK(SameBits(&s.jointMatrices[joints + j], &legacy, sizeof(legacy)));
            }
            joints += extracted.jointMatrices.count;
        }
        CHECK(s.jointMatrixCount == joints);
    }

    const RenderSnapshot& ExtractFrame(RenderService* service, RenderFrame& frame, const ExtractDesc& desc)
    {
        frame.Begin();
        SceneFrame* sceneFrame = nullptr;
        CHECK(service->Extract(frame.graph, &frame.data, desc, &sceneFrame) == nvrhi::FS_OK);
        CHECK(sceneFrame && sceneFrame->snapshot);
        CHECK(frame.data.Find<SceneFrame>() == sceneFrame);
        CHECK(sceneFrame->instanceBuffer == InvalidResourceId && sceneFrame->bufferReadCount == 0);
        CHECK(!sceneFrame->sceneBuffersReplaced);
        return *sceneFrame->snapshot;
    }

    // ------------------------------------------------------------------------------------------ cases

    struct SceneSpec
    {
        const char* name;
        uint64_t seed;
        const Profile* profile;
        void (*build)(TraceScene&, uint32_t);
        uint32_t size;
        uint32_t frames;
    };

    // The trace test's small scenes, 4,096 flat nodes with every tenth leaf of another kind, and 4,096 instances.
    const SceneSpec c_Scenes[] = {
        { "flat_small",     0x1001, &c_SmallProfile, BuildFlat,        12, 40 },
        { "chain_small",    0x1002, &c_SmallProfile, BuildChain,       10, 40 },
        { "tree_small",     0x1003, &c_SmallProfile, BuildTree,        20, 40 },
        { "skinned_small",  0x1004, &c_SmallProfile, BuildCrowd,        1, 40 },
        { "mixed_small",    0x1005, &c_SmallProfile, BuildMixed,       24, 40 },
        { "flat_4096",      0x3001, &c_LargeProfile, BuildFlat,      4096, 12 },
        { "instances_4096", 0x3002, &c_LargeProfile, BuildInstances, 4094, 12 },
    };

    void test_trace_scenes()
    {
        TestViews views;
        for (const SceneSpec& spec : c_Scenes)
        {
            TraceScene s(spec.seed, *spec.profile);
            spec.build(s, spec.size);
            s.nodeLimit = CollectNodes(s.graph).size() * 3 / 2 + 64;
            RenderFrame frame;
            auto service = MakeService(s.graph);
            uint32_t instances = 0;
            for (uint32_t f = 0; f < spec.frames; ++f)
            {
                if (f > 0)
                    Step(s, f);
                s.graph->Refresh(f);
                const ExtractDesc desc = views.Desc(f);
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
                CHECK(s.graph->IsInRenderRead());
                CheckSnapshot(s.graph, snapshot, desc);
                const RenderServiceStats& stats = service->GetStats();
                CHECK(stats.views == 2 && stats.visibleInstances == snapshot.instances.count);
                CHECK(stats.lights == snapshot.lightCount && stats.cameras == snapshot.cameraCount);
                CHECK(stats.skinnedInstances == snapshot.skinnedInstanceCount);
                CHECK(stats.snapshotBytes >= sizeof(RenderSnapshot));
                instances = std::max(instances, snapshot.instances.count);
                service->EndFrame(true);
                CHECK(!s.graph->IsInRenderRead());
            }
            std::printf("  %-15s %u frames, up to %u instances\n", spec.name, spec.frames, instances);
            service = nullptr;
        }
        CHECK(g_Warnings.load() == 0);
    }

    // SceneGraph::RunReadOnlyPhase: an extraction is not a change. After it, a Refresh with nothing changed still
    // returns at once (the world's version does not move) and the on-demand subgraph caches stay (a mesh box edited
    // without InvalidateContent is not seen); a real change still propagates; a phase that ran elsewhere, or a
    // RenderExtract system that declares a write, makes the next Refresh run.
    void test_refresh_skip()
    {
        TestViews views;
        TraceScene s(0x7001, c_SmallProfile);
        CreateAssets(s);
        s.graph->SetRootNode(NewNode(s));
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        for (uint32_t i = 0; i < 64; ++i)
        {
            auto node = NewNode(s);
            node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
            node->SetLeaf(s.factory->CreateMeshInstance(s.meshes[0]));
            nodes.push_back(s.graph->Attach(s.graph->GetRootNode(), node));
        }
        IWorld* world = s.graph->GetWorld();
        RenderFrame frame;
        auto service = MakeService(s.graph);
        uint32_t f = 0;
        // One frame: Refresh, then Extract and EndFrame. Returns whether the Refresh ran its phases.
        auto runFrame = [&](bool check = true) -> bool
        {
            const uint64_t before = world->GetVersion();
            s.graph->Refresh(f);
            const bool refreshed = world->GetVersion() != before;
            const ExtractDesc desc = views.Desc(f);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
            if (check)
                CheckSnapshot(s.graph, snapshot, desc);
            service->EndFrame(true);
            ++f;
            return refreshed;
        };
        CHECK(runFrame());    // the first Refresh runs
        CHECK(!runFrame());   // nothing changed since: skipped after the extraction
        CHECK(!runFrame());

        // the subgraph cache stays: a box edited behind the graph's back is not seen while nothing changes
        SceneGraphNode* root = s.graph->GetRootNode();
        const dm::box3 cached = root->GetGlobalBoundingBox();
        MeshInfo* mesh = s.meshes[0];
        const dm::box3 original = mesh->objectSpaceBounds;
        mesh->objectSpaceBounds = dm::box3(dm::float3(-100.f), dm::float3(100.f));
        CHECK(!runFrame(false));   // the brute force would read the edited box; the columns keep the attached one
        const dm::box3 again = root->GetGlobalBoundingBox();
        CHECK(SameBits(&again, &cached, sizeof(cached)));
        mesh->objectSpaceBounds = original;

        // a real change propagates: the moved instance's world bounds are in the next snapshot
        nodes[5]->SetTranslation(dm::double3(5.0, 7.0, 0.0));
        CHECK(runFrame());
        CHECK(!runFrame());

        // another phase on the world between frames: the next Refresh runs
        const uint64_t beforePhase = world->GetVersion();
        CHECK(s.graph->GetScheduler()->RunPhase(Phase::BeginFrame) == nvrhi::FS_OK);
        if (world->GetVersion() != beforePhase)
            CHECK(runFrame());
        CHECK(!runFrame());

        // a RenderExtract system that declares a write: Debug warns at each extraction, and the next Refresh runs
        // (LocalTransform: no RenderService system reads it, so the writer adds no ambiguity of its own)
        const ComponentId localTransform = RegisterComponent<LocalTransform>(world);
        const ComponentAccess write[] = { { localTransform, engine::Access::Write, {} } };
        SystemDesc systemDesc{};
        systemDesc.name = "test.RenderExtractWriter";
        systemDesc.accesses = write;
        systemDesc.accessCount = 1;
        systemDesc.phase = Phase::RenderExtract;
        systemDesc.flags = SystemFlags::MainThread;
        nvrhi::AutoPtr<ISystem> writer;
        CHECK(CreateLambdaSystem(systemDesc, [](ISystemContext*) noexcept {}, &writer) == nvrhi::FS_OK);
        SystemId writerId = SystemId::Invalid;
        CHECK(s.graph->GetScheduler()->AddSystem(writer, writerId) == nvrhi::FS_OK);
        const uint32_t warningsBefore = g_Warnings.load();
        CHECK(!runFrame());   // this frame's Refresh came before the extraction with the writer
        CHECK(runFrame());
#ifdef NDEBUG
        CHECK(g_Warnings.load() == warningsBefore);
#else
        CHECK(g_Warnings.load() == warningsBefore + 2);
#endif
        g_Warnings.store(warningsBefore);
        CHECK(s.graph->GetScheduler()->RemoveSystem(writerId) == nvrhi::FS_OK);
        CHECK(runFrame());    // the last extraction had the writer
        CHECK(!runFrame());
        service = nullptr;
    }

    // Writes between Extract and EndFrame: the snapshot keeps its bytes; Debug builds warn once per write through the
    // API, Release builds not at all; nothing warns after EndFrame.
    void test_isolation()
    {
        TestViews views;
        TraceScene s(0x4001, c_SmallProfile);
        BuildMixed(s, 200);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        s.graph->Refresh(0);
        service->EndFrame(true);   // before any Extract: nothing to do
        ExtractFrame(service, frame, views.Desc(0));
        service->EndFrame(true);
        for (const auto& node : s.movers)
            Move(s, node);
        s.graph->Refresh(1);

        const ExtractDesc desc = views.Desc(1);
        const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
        CheckSnapshot(s.graph, snapshot, desc);
        const uint64_t before = HashSnapshot(snapshot, true);
        const uint32_t warningsBefore = g_Warnings.load();

        // The script: every kind of write through the API, then direct leaf field edits (which the API cannot see).
        uint32_t apiWrites = 0;
        std::vector<SceneGraphNode*> nodes = CollectNodes(s.graph);
        for (uint32_t i = 0; i < 5; ++i)
        {
            Move(s, s.movers[i]);                                         // SetTransform and its forms
            ++apiWrites;
        }
        nvrhi::AutoPtr<SceneGraphNode> detached = s.graph->Detach(nodes[nodes.size() / 2]);   // Detach
        ++apiWrites;
        s.graph->Attach(s.graph->GetRootNode(), detached);              // re-Attach
        ++apiWrites;
        nodes = CollectNodes(s.graph);
        nodes[3]->SetLeaf(s.factory->CreateMeshInstance(s.meshes[1]));   // SetLeaf
        ++apiWrites;
        nodes[4]->InvalidateContent();                                   // InvalidateContent
        ++apiWrites;
        s.graph->AttachLeafNode(s.graph->GetRootNode(), MAKE_RC_OBJ_PTR(PointLight));   // AttachLeafNode (one Attach)
        ++apiWrites;
        auto newRoot = NewNode(s);
        auto oldRoot = s.graph->SetRootNode(newRoot);                    // SetRootNode: one warning for its two parts
        ++apiWrites;
        s.graph->Attach(newRoot, oldRoot);
        ++apiWrites;
        auto orphan = NewNode(s);                                        // a node in no graph: not this graph's write
        orphan->SetTranslation(dm::double3(1.0, 2.0, 3.0));
        for (const auto& light : s.graph->GetLights())
        {
            light->color = dm::float3(0.25f, 0.5f, 0.75f);
            if (auto* directional = query_cast<DirectionalLight>(light.Get()))
                directional->irradiance = 7.f;
        }

        CHECK(HashSnapshot(snapshot, true) == before);
        const uint32_t warnings = g_Warnings.load() - warningsBefore;
#ifdef NDEBUG
        CHECK(warnings == 0);
#else
        CHECK(warnings == apiWrites);
#endif
        std::printf("  %u writes during the frame, %u warnings, snapshot unchanged\n", apiWrites, warnings);

        service->EndFrame(true);
        CHECK(!s.graph->IsInRenderRead());
        Move(s, s.movers[0]);
        nvrhi::AutoPtr<SceneGraphNode> kept = s.graph->Detach(CollectNodes(s.graph)[5]);   // joints may be in it
        CHECK(g_Warnings.load() == warningsBefore + warnings);

        // the next frame sees the writes
        s.graph->Refresh(2);
        const ExtractDesc next = views.Desc(2);
        CheckSnapshot(s.graph, ExtractFrame(service, frame, next), next);
        service->EndFrame(true);
        g_Warnings.store(warningsBefore);
    }

    // The same frames extracted on the calling thread and at 1, 2 and 8 workers, with the ranges, the copy and sort, or
    // both in parallel whatever their size, give the same snapshot bytes (the calls into the job system tell that each
    // combination ran); the graph scheduler has no ambiguity with the service's systems in it.
    void test_workers()
    {
        constexpr uint32_t c_Runs = 6;
        const uint32_t workerCounts[3] = { 1, 2, 8 };
        nvrhi::AutoPtr<CountingJobSystem> jobs[3];
        for (uint32_t k = 0; k < 3; ++k)
        {
            JobSystemDesc desc;
            desc.debugName = "test_render_extract";
            desc.workerThreadCount = workerCounts[k];
            nvrhi::AutoPtr<IJobSystem> inner;
            CHECK(CreateJobSystem(desc, &inner) == nvrhi::FS_OK);
            jobs[k] = MAKE_RC_OBJ_PTR(CountingJobSystem, inner.Get());
        }
        const struct
        {
            uint32_t jobs;
            ExtractionMode mode;
        } runs[c_Runs] = { { 0, { false, false } }, { 0, { true, true } }, { 1, { true, true } }, { 2, { true, true } },
                           { 2, { false, true } }, { 2, { true, false } } };

        TestViews views;
        TraceScene s(0x2005, c_LargeProfile);
        BuildMixed(s, 1200);
        s.nodeLimit = CollectNodes(s.graph).size() * 3 / 2 + 64;
        RenderFrame frame;
        auto service = MakeService(s.graph);
        for (uint32_t f = 0; f < 10; ++f)
        {
            if (f > 0)
                Step(s, f);
            s.graph->Refresh(f);
            uint64_t hashes[c_Runs] = {};
            for (uint32_t k = 0; k < c_Runs; ++k)
            {
                CountingJobSystem* runJobs = jobs[runs[k].jobs];
                s.graph->SetJobSystem(runJobs);
                runs[k].mode.Set();
                const ExtractDesc desc = views.Desc(f);
                const uint32_t calls = runJobs->calls.load();
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
                CHECK(runJobs->calls.load() - calls == runs[k].mode.Calls(snapshot.packets.count));
                hashes[k] = HashSnapshot(snapshot, false);
                if (k == 0)
                    CheckSnapshot(s.graph, snapshot, desc);
                IScheduler* scheduler = s.graph->GetScheduler();
                CHECK(scheduler->Build() == nvrhi::FS_OK);
                CHECK(scheduler->GetAmbiguityCount() == 0);
                service->EndFrame(k == c_Runs - 1);   // the first-frame flags are set once, after the last extraction
            }
            for (uint32_t k = 1; k < c_Runs; ++k)
                CHECK(hashes[k] == hashes[0]);
        }
        ExtractionMode::Reset();
        s.graph->SetJobSystem(nullptr);
        service = nullptr;
        CHECK(g_Warnings.load() == 0);
    }

    // Runtime-only rows beside nodes (plan step E7, item E7.2): rows of every mesh (the one without geometry and the one
    // with a geometry without a material included), made before the nodes so that their chunks come first in the query,
    // and a row whose mesh id names no mesh of the graph (no slot: never visible; one warning when it is made); then
    // the instances, characters, lights and cameras of BuildInstances. 70 frames of moves of nodes and rows; every
    // frame CheckSnapshot (it takes the rows from the world) and equal snapshots serially and in parallel at 1 and 8
    // workers. In Debug the sampled leaf check (every 64th extraction, the query's first chunks first) passes over the
    // rows' chunks without a warning.
    void test_runtime_rows()
    {
        constexpr uint32_t c_Runs = 3;
        const uint32_t workerCounts[2] = { 1, 8 };
        nvrhi::AutoPtr<CountingJobSystem> jobs[2];
        for (uint32_t k = 0; k < 2; ++k)
        {
            JobSystemDesc desc;
            desc.debugName = "test_render_extract.runtime";
            desc.workerThreadCount = workerCounts[k];
            nvrhi::AutoPtr<IJobSystem> inner;
            CHECK(CreateJobSystem(desc, &inner) == nvrhi::FS_OK);
            jobs[k] = MAKE_RC_OBJ_PTR(CountingJobSystem, inner.Get());
        }
        const struct
        {
            uint32_t jobs;
            ExtractionMode mode;
        } runs[c_Runs] = { { 0, { false, false } }, { 0, { true, true } }, { 1, { true, true } } };

        TestViews views;
        TraceScene s(0x7002, c_LargeProfile);
        CreateAssets(s);
        s.graph->SetRootNode(NewNode(s));
        SceneGraphNode* root = s.graph->GetRootNode();
        IWorld* world = s.graph->GetWorld();
        std::vector<MeshId> meshIds;
        for (const auto& mesh : s.meshes)
            meshIds.push_back(s.graph->AcquireRuntimeMesh(mesh));
        std::vector<ComponentId> components(s.graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
        s.graph->GetRuntimeMeshInstanceComponents(components.data(), uint32_t(components.size()));
        auto spawn = [&](const std::vector<MeshRef>& refs, std::vector<LocalTransform>& locals)
        {
            std::vector<const void*> values(components.size(), nullptr);
            for (size_t c = 0; c < components.size(); ++c)
            {
                if (components[c] == FindComponent<LocalTransform>(world))
                    values[c] = locals.data();
                else if (components[c] == FindComponent<MeshRef>(world))
                    values[c] = refs.data();
            }
            EntityCreateDesc desc{};
            desc.components = components.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(components.size());
            desc.entityCount = uint32_t(refs.size());
            std::vector<Entity> rows(refs.size(), Entity::Null);
            CHECK(world->CreateEntities(desc, rows.data()) == nvrhi::FS_OK);
            return rows;
        };
        std::vector<MeshRef> refs;
        std::vector<LocalTransform> locals;
        for (uint32_t i = 0; i < 400; ++i)
        {
            refs.push_back(MeshRef{ meshIds[i % meshIds.size()] });
            locals.push_back(LocalTransform{ RandomVector(s.rng, -30.0, 30.0), RandomRotation(s.rng),
                                             RandomVector(s.rng, 0.5, 1.5) });
        }
        std::vector<Entity> rows = spawn(refs, locals);
        std::vector<MeshRef> unknown = { MeshRef{ MeshId((9u << 24) | 5u) } };
        std::vector<LocalTransform> unknownLocal = { locals[0] };
        const Entity stray = spawn(unknown, unknownLocal)[0];
        CHECK(g_Warnings.load() == 1);   // the row whose id names no mesh: no slot
        g_Warnings.fetch_sub(1);
        CHECK(GetComponent<GpuInstanceSlot>(world, stray)->instance == c_NoInstanceSlot);
        for (uint32_t i = 0; i < 200; i++)
        {
            auto node = NewNode(s);
            Move(s, node);
            node->SetLeaf(NewLeaf(s, LeafKind::Mesh));
            s.graph->Attach(root, node);
            Track(s, node);
        }
        auto character = NewCharacter(s, 5);
        s.graph->Attach(root, character);
        Track(s, character);
        AddLightsAndCameras(s, root);

        RenderFrame frame;
        auto service = MakeService(s.graph);
        uint64_t runtimeVisible = 0, nodeVisible = 0;
        for (uint32_t f = 0; f < 70; ++f)
        {
            if (f > 0)
            {
                for (const auto& node : s.movers)
                    Move(s, node);
                world->AdvanceVersion();
                for (size_t i = f % 10; i < rows.size(); i += 10)
                    GetComponentForWrite<LocalTransform>(world, rows[i])->translation = RandomVector(s.rng, -30.0, 30.0);
            }
            s.graph->Refresh(f);
            uint64_t hashes[c_Runs] = {};
            for (uint32_t k = 0; k < c_Runs; ++k)
            {
                CountingJobSystem* runJobs = jobs[runs[k].jobs];
                s.graph->SetJobSystem(runJobs);
                runs[k].mode.Set();
                const ExtractDesc desc = views.Desc(f);
                const uint32_t calls = runJobs->calls.load();
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
                const uint32_t made = runJobs->calls.load() - calls;
                CHECK(runs[k].mode.parallelRanges ? made > 0 : made == runs[k].mode.Calls(snapshot.packets.count));
                hashes[k] = HashSnapshot(snapshot, false);
                if (k == 0)
                {
                    CheckSnapshot(s.graph, snapshot, desc);
                    for (uint32_t i = 0; i < snapshot.instances.count; ++i)
                    {
                        CHECK(snapshot.instances.entities[i] != stray);
                        runtimeVisible += snapshot.instances.leaves[i] ? 0 : 1;
                        nodeVisible += snapshot.instances.leaves[i] ? 1 : 0;
                    }
                }
                service->EndFrame(k == c_Runs - 1);
            }
            for (uint32_t k = 1; k < c_Runs; ++k)
                CHECK(hashes[k] == hashes[0]);
        }
        ExtractionMode::Reset();
        s.graph->SetJobSystem(nullptr);
        std::printf("  %zu runtime rows and %zu nodes, 70 frames: %llu visible runtime rows, %llu visible node "
                    "instances\n", rows.size(), CollectNodes(s.graph).size(), (unsigned long long)runtimeVisible,
                    (unsigned long long)nodeVisible);
        CHECK(runtimeVisible > 1000 && nodeVisible > 1000);
        service = nullptr;
        for (MeshId id : meshIds)
            s.graph->ReleaseRuntimeMesh(id);
        CHECK(g_Warnings.load() == 0);
    }

    // The service registers the leaf tags its queries named before E7.2, in that order, although its instance query no
    // longer names MeshInstanceLeaf (plan step E7, decision 8): at its first extraction on a world without mesh leaves
    // yet, MeshInstanceLeaf then SkinnedMeshInstanceLeaf, consecutive, so that the world's component ids are those it
    // had.
    void test_component_ids()
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        IWorld* world = graph->GetWorld();
        CHECK(FindComponent<MeshInstanceLeaf>(world) == ComponentId::Invalid);
        CHECK(FindComponent<SkinnedMeshInstanceLeaf>(world) == ComponentId::Invalid);
        auto service = MakeService(graph);
        TestViews views;
        RenderFrame frame;
        graph->Refresh(1);
        (void)ExtractFrame(service, frame, views.Desc(1));   // the service claims the graph at its first Extract
        service->EndFrame(true);
        const ComponentId mesh = FindComponent<MeshInstanceLeaf>(world);
        const ComponentId skinned = FindComponent<SkinnedMeshInstanceLeaf>(world);
        CHECK(mesh != ComponentId::Invalid && skinned != ComponentId::Invalid);
        CHECK(uint32_t(mesh) + 1 == uint32_t(skinned));
        std::printf("  MeshInstanceLeaf %u, SkinnedMeshInstanceLeaf %u\n", uint32_t(mesh), uint32_t(skinned));
    }

    // The snapshot lives in the graph's frame arena: Reset rewinds it, so the next frame's snapshot takes the same
    // memory, and the blackboard's entry goes with FrameData::Clear.
    void test_arena_reset()
    {
        TestViews views;
        TraceScene s(0x3001, c_LargeProfile);
        BuildFlat(s, 4096);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        s.graph->Refresh(0);
        const RenderSnapshot* first = &ExtractFrame(service, frame, views.Desc(0));
        const uint64_t firstHash = HashSnapshot(*first, false);
        const uint64_t firstFrameIndex = first->graphFrameIndex;
        const void* firstLights = first->lights;
        service->EndFrame(false);   // the first-frame skinning flags stay, so the two snapshots are equal

        frame.Begin();
        CHECK(frame.data.Find<SceneFrame>() == nullptr);
        s.graph->Refresh(0);
        SceneFrame* sceneFrame = nullptr;
        CHECK(service->Extract(frame.graph, &frame.data, views.Desc(0), &sceneFrame) == nvrhi::FS_OK);
        const RenderSnapshot* second = sceneFrame->snapshot;
        CHECK(second == first && second->lights == firstLights);
        CHECK(second->graphFrameIndex == firstFrameIndex + 1);
        CHECK(HashSnapshot(*second, false) == firstHash);
        service->EndFrame(true);

        // Extract without a blackboard: the SceneFrame goes into the arena only
        frame.Begin();
        CHECK(service->Extract(frame.graph, nullptr, views.Desc(0), &sceneFrame) == nvrhi::FS_OK);
        CHECK(sceneFrame && sceneFrame->snapshot && frame.data.Find<SceneFrame>() == nullptr);
        service->EndFrame(true);
        CHECK(g_Warnings.load() == 0);
    }

    // Ten frames of moves and animations after warm-up: no operator new (the arena's blocks included), no ECS heap
    // block.
    void test_steady_state()
    {
        TestViews views;
        TraceScene s(0x5001, c_LargeProfile);
        BuildInstances(s, 4094);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        uint32_t f = 0;
        auto run = [&]()
        {
            for (const auto& node : s.movers)
                Move(s, node);
            for (const auto& animation : s.graph->GetAnimations())
                animation->Apply(float(f) * 0.25f);
            s.graph->Refresh(f);
            frame.Begin();
            SceneFrame* sceneFrame = nullptr;
            CHECK(service->Extract(frame.graph, &frame.data, views.Desc(f), &sceneFrame) == nvrhi::FS_OK);
            service->EndFrame(true);
            ++f;
        };
        for (int i = 0; i < 6; ++i)
            run();
        const RenderServiceStats warm = service->GetStats();
        const uint64_t newBefore = g_NewCount.load();
        const uint64_t heapBefore = ecs::GetHeapAllocationCount();
        for (int i = 0; i < 10; ++i)
            run();
        const uint64_t newCalls = g_NewCount.load() - newBefore;
        const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
        const RenderServiceStats& stats = service->GetStats();
        std::printf("  %u instances, %u lights, %u cameras, %u skinned instances, %u joints, %llu snapshot bytes: "
                    "%llu operator new, %llu storage blocks over 10 frames\n", stats.visibleInstances, stats.lights,
                    stats.cameras, stats.skinnedInstances, stats.jointMatrices,
                    (unsigned long long)stats.snapshotBytes, (unsigned long long)newCalls,
                    (unsigned long long)heapBlocks);
        CHECK(warm.visibleInstances > 0 && stats.skinnedInstances > 0 && stats.lights > 0);
        CHECK(warm.packets[uint32_t(DrawPacketPass::Opaque)] > 0);
        CHECK(newCalls == 0 && heapBlocks == 0);
        CHECK(g_Warnings.load() == 0);
    }

    // One service per graph: a second one is refused (FE_INVALID_ARGS) until the first is gone.
    void test_second_service()
    {
        TestViews views;
        TraceScene s(0x1005, c_SmallProfile);
        BuildMixed(s, 24);
        s.graph->Refresh(0);
        RenderFrame frame;
        const ExtractDesc desc = views.Desc(0);

        auto first = MakeService(s.graph);
        auto second = MakeService(s.graph);
        CHECK(first->GetLightShadows() == nullptr);   // nothing made before the first Extract
        frame.Begin();
        CHECK(first->Extract(frame.graph, &frame.data, desc, nullptr) == nvrhi::FS_OK);
        CHECK(first->GetLightShadows() != nullptr && first->GetLightShadows()->GetCount() == 0);
        first->EndFrame(true);

        const uint32_t warningsBefore = g_Warnings.load();
        frame.Begin();
        SceneFrame* sceneFrame = reinterpret_cast<SceneFrame*>(uintptr_t(1));
        CHECK(second->Extract(frame.graph, &frame.data, desc, &sceneFrame) == nvrhi::FE_INVALID_ARGS);
        CHECK(sceneFrame == nullptr && !s.graph->IsInRenderRead());
        CHECK(second->Extract(frame.graph, &frame.data, desc, nullptr) == nvrhi::FE_INVALID_ARGS);
        second->EndFrame(true);   // nothing open

        // no graph at all
        auto none = MAKE_RC_OBJ_PTR(RenderService, nullptr, RenderServiceDesc{});
        CHECK(none->Extract(frame.graph, &frame.data, desc, nullptr) == nvrhi::FE_INVALID_ARGS);
        // bad arguments
        CHECK(first->Extract(nullptr, &frame.data, desc, nullptr) == nvrhi::FE_INVALID_ARGS);
        ExtractDesc noViews = desc;
        noViews.views = nullptr;
        CHECK(first->Extract(frame.graph, &frame.data, noViews, nullptr) == nvrhi::FE_INVALID_ARGS);
        CHECK(!s.graph->IsInRenderRead());
        CHECK(g_Warnings.load() == warningsBefore + 3);   // two refusals, one missing graph

        // the first one gone, the second one takes the graph
        first = nullptr;
        frame.Begin();
        CHECK(second->Extract(frame.graph, &frame.data, desc, nullptr) == nvrhi::FS_OK);
        CheckSnapshot(s.graph, *frame.data.Get<SceneFrame>().snapshot, desc);
        // the add passes of later items accept the open frame and add nothing yet (no scene, no device)
        CHECK(second->AddUploadPasses(frame.graph, &frame.data.Get<SceneFrame>()) == nvrhi::FS_OK);
        CHECK(second->AddSkinningPasses(frame.graph, &frame.data.Get<SceneFrame>()) == nvrhi::FS_OK);
        CHECK(second->AddShadowPasses(frame.graph, &frame.data.Get<SceneFrame>()) == nvrhi::FS_OK);
        CHECK(frame.graph->GetPassCount() == 0);
        second->EndFrame(true);
        g_Warnings.store(warningsBefore);
    }

    // EndFrame: idempotent; a frame without EndFrame is closed by the next Extract; the graph counts scopes, the
    // service keeps one flag. The first-frame skinning flag: EndFrame(true) sets skinningInitialized only for instances
    // a skinning dispatch consumed (AddSkinningPasses, E5.5); a service without a device adds no skinning pass, so
    // every selected instance stays a first frame through executed frames (the dispatching side is checked on the
    // device by test_render_extract_gpu).
    void test_end_frame()
    {
        TestViews views;
        TraceScene s(0x1004, c_SmallProfile);
        BuildCrowd(s, 3);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        uint32_t f = 1;
        auto next = [&]() -> const RenderSnapshot&
        {
            for (const auto& animation : s.graph->GetAnimations())
                animation->Apply(float(f) * 0.25f);
            s.graph->Refresh(f);
            const ExtractDesc desc = views.Desc(f);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
            CheckSnapshot(s.graph, snapshot, desc);
            ++f;
            return snapshot;
        };
        auto allFirst = [](const RenderSnapshot& snapshot)
        {
            CHECK(snapshot.skinnedInstanceCount == 3);
            for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
                CHECK(snapshot.skinnedInstances[i].firstFrame);
        };
        auto noneInitialized = [&]()
        {
            for (const auto& instance : s.graph->GetSkinnedMeshInstances())
                CHECK(!instance->skinningInitialized);
        };

        allFirst(next());
        service->EndFrame(false);   // not executed
        service->EndFrame(true);    // the frame is over: nothing more
        noneInitialized();
        CHECK(!s.graph->IsInRenderRead());

        allFirst(next());           // EndFrame never comes for this one
        CHECK(s.graph->IsInRenderRead());
        // The next frame's animation writes while that scope is still open: Debug warns about each (12 joints).
        const uint32_t warningsBefore = g_Warnings.load();
        allFirst(next());           // closes the open scope first: still one scope
#ifdef NDEBUG
        CHECK(g_Warnings.load() == warningsBefore);
#else
        CHECK(g_Warnings.load() == warningsBefore + 12);
#endif
        g_Warnings.store(warningsBefore);
        // no device: no skinning pass
        CHECK(service->AddSkinningPasses(frame.graph, frame.data.Find<SceneFrame>()) == nvrhi::FS_OK);
        CHECK(frame.graph->GetPassCount() == 0);
        service->EndFrame(true);    // executed, but no dispatch consumed a first frame
        noneInitialized();
        CHECK(!s.graph->IsInRenderRead());
        service->EndFrame(true);
        service->EndFrame(false);
        noneInitialized();

        allFirst(next());
        // the application's own scope nests with the service's
        s.graph->BeginRenderRead();
        service->EndFrame(true);
        CHECK(s.graph->IsInRenderRead());
        s.graph->EndRenderRead();
        CHECK(!s.graph->IsInRenderRead());
        s.graph->EndRenderRead();   // unbalanced: ignored
        CHECK(!s.graph->IsInRenderRead());

        // a service released with its frame open closes its scope
        next();
        CHECK(s.graph->IsInRenderRead());
        service = nullptr;
        CHECK(!s.graph->IsInRenderRead());
        CHECK(g_Warnings.load() == 0);
    }

    // The shadow table's primitive operations (LightShadows.h); the service requests nothing before step E5.6.
    class FakeShadowMap : public IShadowMap
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        explicit FakeShadowMap(std::atomic<int>* live) : m_Live(live) { ++*m_Live; }
        ~FakeShadowMap() override { --*m_Live; }
        dm::float4x4 GetWorldToUvzwMatrix() const override { return dm::float4x4::identity(); }
        const ICompositeView& GetView() const override { return *m_View; }
        nvrhi::ITexture* GetTexture() const override { return nullptr; }
        uint32_t GetNumberOfCascades() const override { return 1; }
        const IShadowMap* GetCascade(uint32_t) const override { return this; }
        uint32_t GetNumberOfPerObjectShadows() const override { return 0; }
        const IShadowMap* GetPerObjectShadow(uint32_t) const override { return nullptr; }
        dm::int2 GetTextureSize() const override { return dm::int2(1024); }
        dm::box2 GetUVRange() const override { return dm::box2(dm::float2(0.f), dm::float2(1.f)); }
        dm::float2 GetFadeRangeInTexels() const override { return dm::float2(0.f); }
        bool IsLitOutOfBounds() const override { return false; }
        void FillShadowConstants(ShadowConstants&) const override { }
    private:
        std::atomic<int>* m_Live;
        nvrhi::AutoPtr<PlanarView> m_View = MAKE_RC_OBJ_PTR(PlanarView);
    };

    Entity MakeEntity(uint32_t index, uint32_t generation)
    {
        return Entity((uint64_t(generation) << 32) | index);
    }

    void test_light_shadow_table()
    {
        std::atomic<int> live{ 0 };
        auto give = [&](LightShadowEntry* entry)
        {
            auto map = MAKE_RC_OBJ_PTR(FakeShadowMap, &live);
            entry->map = map.Get();
            entry->map->AddRef();   // the table's reference
        };
        const LightShadow cascades{ 0, 4, 2048, nvrhi::Format::D32 };
        const LightShadow spot{ 1, 1, 1024, nvrhi::Format::D32 };
        {
            LightShadowTable table;
            const Entity a = MakeEntity(7, 1), b = MakeEntity(3, 2), c = MakeEntity(7, 2);
            give(table.Request(a, cascades, LightShadowKind::Cascaded));
            give(table.Request(b, spot, LightShadowKind::Planar));
            CHECK(table.Request(Entity::Null, spot, LightShadowKind::Planar) == nullptr);
            CHECK(table.GetCount() == 2 && live.load() == 2);
            // ordered by index, then generation
            CHECK(table.GetEntries()[0].light == b && table.GetEntries()[1].light == a);
            CHECK(table.Find(a) && table.Find(a)->map && table.Find(c) == nullptr);
            CHECK(table.Sweep() == 0);

            // the next extraction: `a` changes its configuration (a new map), `b` is not requested (it leaves),
            // `c` (a recycled index) is a light of its own
            const LightShadow twoCascades{ 0, 2, 2048, nvrhi::Format::D32 };
            LightShadowEntry* entry = table.Request(a, twoCascades, LightShadowKind::Cascaded);
            CHECK(entry && entry->map == nullptr && table.GetRetiredCount() == 1);
            give(entry);
            CHECK(table.Request(c, cascades, LightShadowKind::Cascaded)->map == nullptr);
            CHECK(table.GetEntries()[1].light == a && table.GetEntries()[2].light == c);
            CHECK(table.Sweep() == 1);
            CHECK(table.GetCount() == 2 && table.Find(b) == nullptr && table.GetRetiredCount() == 2);
            CHECK(live.load() == 3);
            table.ReleaseRetired();
            CHECK(table.GetRetiredCount() == 0 && live.load() == 1);

            // nothing requested: everything leaves
            CHECK(table.Sweep() == 2 && table.GetCount() == 0 && table.GetRetiredCount() == 1);
            give(table.Request(a, cascades, LightShadowKind::Cascaded));
            CHECK(live.load() == 2);
        }
        CHECK(live.load() == 0);   // the destructor released the entries' and the retired maps
    }

    // RunReadOnlyPhase with a RenderExtract system that is exclusive, and with one that declares reads only but records
    // a command (played back at the phase's sync point): either makes the next Refresh run.
    void test_read_only_phase()
    {
        TestViews views;
        TraceScene s(0x7002, c_SmallProfile);
        BuildFlat(s, 32);
        IWorld* world = s.graph->GetWorld();
        RenderFrame frame;
        auto service = MakeService(s.graph);
        uint32_t f = 0;
        auto runFrame = [&]() -> bool
        {
            const uint64_t before = world->GetVersion();
            s.graph->Refresh(f);
            const bool refreshed = world->GetVersion() != before;
            ExtractFrame(service, frame, views.Desc(f++));
            service->EndFrame(true);
            return refreshed;
        };
        CHECK(runFrame());
        CHECK(!runFrame());

        auto addSystem = [&](SystemFlags flags, const ComponentAccess* accesses, uint32_t accessCount, auto&& update)
        {
            SystemDesc systemDesc{};
            systemDesc.name = "test.RenderExtractSystem";
            systemDesc.accesses = accesses;
            systemDesc.accessCount = accessCount;
            systemDesc.phase = Phase::RenderExtract;
            systemDesc.flags = flags;
            // after the service's chain: an exclusive system conflicts with every other one (no ambiguity report)
            static const char* const c_After[] = { "RenderService.Instances" };
            systemDesc.runAfter = c_After;
            systemDesc.runAfterCount = 1;
            nvrhi::AutoPtr<ISystem> system;
            CHECK(CreateLambdaSystem(systemDesc, std::forward<decltype(update)>(update), &system) == nvrhi::FS_OK);
            SystemId id = SystemId::Invalid;
            CHECK(s.graph->GetScheduler()->AddSystem(system, id) == nvrhi::FS_OK);
            return id;
        };

        // exclusive: Debug warns at the extraction; the next Refresh runs
        const uint32_t warningsBefore = g_Warnings.load();
        SystemId id = addSystem(SystemFlags::Exclusive, nullptr, 0, [](ISystemContext*) noexcept {});
        CHECK(!runFrame());
        CHECK(runFrame());
        CHECK(s.graph->GetScheduler()->RemoveSystem(id) == nvrhi::FS_OK);
        CHECK(runFrame());
        CHECK(!runFrame());
#ifdef NDEBUG
        CHECK(g_Warnings.load() == warningsBefore);
#else
        CHECK(g_Warnings.load() == warningsBefore + 2);
#endif
        g_Warnings.store(warningsBefore);

        // reads only, but a command: an entity created at the sync point; the next Refresh runs. Once only.
        const ComponentId leafObject = RegisterComponent<LeafObject>(world);
        const ComponentAccess reads[] = { { leafObject, engine::Access::Read, {} } };
        bool recorded = false;
        id = addSystem(SystemFlags::MainThread, reads, 1, [&recorded](ISystemContext* context) noexcept
        {
            if (recorded)
                return;
            recorded = true;
            EntityCreateDesc create{};
            create.entityCount = 1;
            Entity placeholder = Entity::Null;
            context->GetCommandBuffer()->CreateEntities(create, MakeSortKey(0, 0, 0), &placeholder);
        });
        CHECK(!runFrame());   // the extraction plays the command back
        CHECK(recorded);
        CHECK(runFrame());    // so this Refresh runs
        CHECK(!runFrame());   // the system records nothing more: skipped again
        CHECK(s.graph->GetScheduler()->RemoveSystem(id) == nvrhi::FS_OK);
        runFrame();
        CHECK(!runFrame());
        service = nullptr;
        CHECK(g_Warnings.load() == warningsBefore);
    }

    // An IRenderGraph whose frame arena refuses allocations after `budget` of them; everything else goes to a real one.
    class FailingRenderGraph : public nvrhi::ObjectImpl<IRenderGraph>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(FailingRenderGraph)
        NVRHI_IMPLEMENTS_INTERFACE(IRenderGraph)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        explicit FailingRenderGraph(IRenderGraph* inner) : m_Inner(inner) { }
        int64_t budget = -1;   // allocations that succeed; negative: all
        uint32_t allocations = 0;

        IResourceRegistry* GetResourceRegistry() noexcept override { return m_Inner->GetResourceRegistry(); }
        void Reset() noexcept override { m_Inner->Reset(); }
        ResourceId CreateTexture(const nvrhi::TextureDesc& d, ResourceFlags f) noexcept override
        {
            return m_Inner->CreateTexture(d, f);
        }
        ResourceId CreateBuffer(const nvrhi::BufferDesc& d, ResourceFlags f) noexcept override
        {
            return m_Inner->CreateBuffer(d, f);
        }
        ResourceId ImportTexture(nvrhi::ITexture* t, nvrhi::ResourceStates e, nvrhi::ResourceStates x) noexcept override
        {
            return m_Inner->ImportTexture(t, e, x);
        }
        ResourceId ImportHostKeptTexture(nvrhi::ITexture* t) noexcept override
        {
            return m_Inner->ImportHostKeptTexture(t);
        }
        ResourceId ImportBuffer(nvrhi::IBuffer* b, nvrhi::ResourceStates e, nvrhi::ResourceStates x) noexcept override
        {
            return m_Inner->ImportBuffer(b, e, x);
        }
        ResourceId ImportTextureDesc(const nvrhi::TextureDesc& d, nvrhi::ResourceStates e,
                                     nvrhi::ResourceStates x) noexcept override
        {
            return m_Inner->ImportTextureDesc(d, e, x);
        }
        ResourceId ImportBufferDesc(const nvrhi::BufferDesc& d, nvrhi::ResourceStates e,
                                    nvrhi::ResourceStates x) noexcept override
        {
            return m_Inner->ImportBufferDesc(d, e, x);
        }
        ResourceId GetPersistentTexture(const char* n, const nvrhi::TextureDesc& d) noexcept override
        {
            return m_Inner->GetPersistentTexture(n, d);
        }
        ResourceId GetPersistentBuffer(const char* n, const nvrhi::BufferDesc& d) noexcept override
        {
            return m_Inner->GetPersistentBuffer(n, d);
        }
        nvrhi::FRESULT GetHistory(const char* n, const nvrhi::TextureDesc& d, uint32_t c,
                                  ResourceId* ids) noexcept override
        {
            return m_Inner->GetHistory(n, d, c, ids);
        }
        void Extract(ResourceId t, nvrhi::ResourceStates x, nvrhi::ITexture** out) noexcept override
        {
            m_Inner->Extract(t, x, out);
        }
        nvrhi::FRESULT AddPass(const PassDesc& d, IGpuPass* p, void* data, PassId* out) noexcept override
        {
            return m_Inner->AddPass(d, p, data, out);
        }
        void SetPassEnabled(PassId p, bool e) noexcept override { m_Inner->SetPassEnabled(p, e); }
        void PushMarker(const char* n) noexcept override { m_Inner->PushMarker(n); }
        void PopMarker() noexcept override { m_Inner->PopMarker(); }
        void* AllocateFrameData(size_t size, size_t alignment) noexcept override
        {
            if (budget >= 0 && int64_t(allocations) >= budget)
                return nullptr;
            ++allocations;
            return m_Inner->AllocateFrameData(size, alignment);
        }
        nvrhi::FRESULT Compile(const BackendProfile& p, const CompileOptions& o,
                               ICompiledRenderGraph** out) noexcept override
        {
            return m_Inner->Compile(p, o, out);
        }
        nvrhi::FRESULT Execute(ICompiledRenderGraph* p, const ExecuteDesc& d) noexcept override
        {
            return m_Inner->Execute(p, d);
        }
        uint64_t GetStructuralHash() noexcept override { return m_Inner->GetStructuralHash(); }
        uint64_t GetEnableMaskHash() noexcept override { return m_Inner->GetEnableMaskHash(); }
        uint32_t GetPassCount() noexcept override { return m_Inner->GetPassCount(); }
        uint32_t GetResourceCount() noexcept override { return m_Inner->GetResourceCount(); }
        uint64_t GetFrameIndex() noexcept override { return m_Inner->GetFrameIndex(); }
        nvrhi::FRESULT GetTextureDesc(ResourceId t, nvrhi::TextureDesc& out) noexcept override
        {
            return m_Inner->GetTextureDesc(t, out);
        }
        nvrhi::FRESULT GetBufferDesc(ResourceId b, nvrhi::BufferDesc& out) noexcept override
        {
            return m_Inner->GetBufferDesc(b, out);
        }
        void GetDiagnostics(nvrhi::string& out) noexcept override { m_Inner->GetDiagnostics(out); }

    private:
        IRenderGraph* m_Inner;
    };

    uint32_t ReferenceCount(nvrhi::IObject* object)
    {
        object->AddRef();
        return uint32_t(object->Release());
    }

    // The frame arena exhausted at every allocation Extract makes, in turn (the snapshot, each system's arrays, the
    // blackboard entry): Extract fails with FE_OUT_OF_MEMORY, the scope is closed, the frame's references (the leaves'
    // shadow maps) are released, and the next frame extracts normally.
    void test_failure_injection()
    {
        std::atomic<int> live{ 0 };
        TestViews views;
        TraceScene s(0x1005, c_SmallProfile);
        BuildMixed(s, 24);
        auto map = MAKE_RC_OBJ_PTR(FakeShadowMap, &live);
        CHECK(!s.graph->GetLights().empty());
        s.graph->GetLights()[0]->shadowMap = map.Get();
        const uint32_t mapReferences = ReferenceCount(map);
        s.graph->Refresh(0);
        RenderFrame frame;
        auto failing = MAKE_RC_OBJ_PTR(FailingRenderGraph, frame.graph.Get());
        auto service = MakeService(s.graph);
        const ExtractDesc desc = views.Desc(0);

        uint32_t failures = 0;
        for (int64_t budget = 0; ; ++budget)
        {
            frame.Begin();
            failing->budget = budget;
            failing->allocations = 0;
            SceneFrame* sceneFrame = nullptr;
            const nvrhi::FRESULT result = service->Extract(failing, &frame.data, desc, &sceneFrame);
            if (result == nvrhi::FS_OK)
            {
                CHECK(sceneFrame && s.graph->IsInRenderRead());
                CHECK(ReferenceCount(map) == mapReferences + 1);   // held for the frame
                CheckSnapshot(s.graph, *sceneFrame->snapshot, desc);
                service->EndFrame(true);
                break;
            }
            CHECK(result == nvrhi::FE_OUT_OF_MEMORY && sceneFrame == nullptr);
            CHECK(!s.graph->IsInRenderRead());
            CHECK(ReferenceCount(map) == mapReferences);
            if (SceneFrame* stale = frame.data.Find<SceneFrame>())
                CHECK(service->AddUploadPasses(failing, stale) == nvrhi::FE_INVALID_ARGS);
            ++failures;
            CHECK(budget < 64);
        }
        std::printf("  %u allocation failures, each refused cleanly\n", failures);
        CHECK(failures >= 6);   // the snapshot, views, lights (2), cameras, the blackboard entry at least
        CHECK(ReferenceCount(map) == mapReferences);
        s.graph->GetLights()[0]->shadowMap = nullptr;
        service = nullptr;
    }

    // The E0.2 regression on the graph path (sub-plan item E5.5): a detached skinned instance is never selected, though
    // its joints keep moving; attached again, it is selected again. Every frame also equals the legacy selection and
    // joint matrices (CheckSnapshot).
    void test_skinned_detach()
    {
        TestViews views;
        TraceScene s(0x1006, c_SmallProfile);
        BuildCrowd(s, 3);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        CHECK(s.graph->GetSkinnedMeshInstances().size() == 3);
        for (const auto& instance : s.graph->GetSkinnedMeshInstances())
            instance->skinningInitialized = true;   // skinned at the load: the legacy rule alone selects
        const SkinnedMeshInstance* leaf = s.graph->GetSkinnedMeshInstances()[1];
        nvrhi::AutoPtr<SceneGraphNode> character = leaf->GetNode()->GetParent();
        nvrhi::AutoPtr<SceneGraphNode> detached;
        for (uint32_t f = 1; f <= 8; ++f)
        {
            for (const SkinnedMeshJoint& joint : leaf->joints)   // its joints move in every frame, attached or not
                if (nvrhi::AutoPtr<SceneGraphNode> node = joint.node.Lock())
                    node->SetTranslation(dm::double3(0.0, 0.01 * double(f), 0.0));
            if (f == 3)
                detached = s.graph->Detach(character);
            if (f == 6)
            {
                s.graph->Attach(s.graph->GetRootNode(), detached);
                detached = nullptr;
            }
            s.graph->Refresh(f);
            const ExtractDesc desc = views.Desc(f);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
            CheckSnapshot(s.graph, snapshot, desc);
            bool selected = false;
            for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
                selected = selected || snapshot.skinnedInstances[i].leaf == leaf;
            CHECK(selected == (f < 3 || f >= 6));
            service->EndFrame(true);
        }
        service = nullptr;
    }

    // E5.5 review fix 1: a skinned instance selected by a frame that did not execute is selected again, across
    // frame-index gaps, until two executed frames have selected it (its dispatch, then the settle); two dropped frames
    // in a row keep it pending; an instance never skinned is selected in every frame.
    void test_skinned_pending()
    {
        TestViews views;
        TraceScene s(0x1007, c_SmallProfile);
        BuildCrowd(s, 3);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        const auto& leaves = s.graph->GetSkinnedMeshInstances();
        CHECK(leaves.size() == 3);
        leaves[0]->skinningInitialized = true;
        leaves[1]->skinningInitialized = true;   // leaves[2] is never skinned: selected in every frame
        const SkinnedMeshInstance* moving = leaves[0];
        const SkinnedMeshInstance* never = leaves[2];
        s.graph->Refresh(0);
        struct Step
        {
            uint32_t frame;
            bool move;        // the joints of `moving` move before the refresh
            bool executed;    // EndFrame(executed)
            bool pending;     // `moving` is expected pending at the extraction
            bool selected;    // `moving` is expected in the snapshot
        };
        // The pending leaves as the rule says (joint versions are per chunk, so a character sharing a chunk with
        // `moving` is selected with it and pending too): a dropped frame makes its selected leaves pending for two
        // executed frames.
        std::vector<std::pair<const SkinnedMeshInstance*, uint32_t>> pending;
        const Step steps[] = {
            { 3, true, false, false, true },    // moved, dropped (frame 3: the load's frame 0 no longer selects)
            { 4, false, false, true, true },    // pending (and the legacy rule), dropped again
            { 7, false, true, true, true },     // a frame-index gap: pending only; executed
            { 8, false, true, true, true },     // the settle; executed
            { 9, false, true, false, false },   // nothing left
            { 11, true, true, false, true },    // moved, executed: the legacy rule alone
            { 15, false, true, false, false },  // a gap after executed frames: nothing pending
        };
        for (const Step& step : steps)
        {
            if (step.move)
            {
                for (const SkinnedMeshJoint& joint : moving->joints)
                    if (nvrhi::AutoPtr<SceneGraphNode> node = joint.node.Lock())
                        node->SetTranslation(dm::double3(0.0, 0.01 * double(step.frame), 0.0));
            }
            s.graph->Refresh(step.frame);
            g_ExpectedPendingSkins.clear();
            for (const auto& entry : pending)
                g_ExpectedPendingSkins.push_back(entry.first);
            CHECK((std::find(g_ExpectedPendingSkins.begin(), g_ExpectedPendingSkins.end(), moving) !=
                   g_ExpectedPendingSkins.end()) == step.pending);
            const ExtractDesc desc = views.Desc(step.frame);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
            CheckSnapshot(s.graph, snapshot, desc);
            bool selected = false, neverSelected = false;
            for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
            {
                selected = selected || snapshot.skinnedInstances[i].leaf == moving;
                neverSelected = neverSelected || snapshot.skinnedInstances[i].leaf == never;
            }
            CHECK(selected == step.selected && neverSelected);
            service->EndFrame(step.executed);
            if (step.executed)
            {
                for (auto& entry : pending)
                    --entry.second;
                pending.erase(std::remove_if(pending.begin(), pending.end(),
                                             [](const auto& entry) { return entry.second == 0; }),
                              pending.end());
            }
            else
            {
                for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
                {
                    const SkinnedMeshInstance* leaf = snapshot.skinnedInstances[i].leaf;
                    auto it = std::find_if(pending.begin(), pending.end(),
                                           [leaf](const auto& entry) { return entry.first == leaf; });
                    if (it != pending.end())
                        it->second = 2;
                    else
                        pending.emplace_back(leaf, 2u);
                }
            }
        }
        g_ExpectedPendingSkins.clear();
        service = nullptr;
    }

    // SceneGraph::Simulate (E6 sub-plan, item E6.2b) propagates joints before Refresh does: what is extracted equals that
    // of the same script without it, frame by frame, instances and matrices, with joints moved before Simulate, between
    // it and Refresh, in both places and not at all, also in the frame after a move and in the one after that.
    void test_skinned_simulate()
    {
        struct Extracted
        {
            std::vector<uint32_t> instances;   // by their index in the graph's vector
            std::vector<dm::float4x4> matrices;
        };
        std::vector<Extracted> results[2];
        for (int simulate = 0; simulate < 2; ++simulate)
        {
            TestViews views;
            TraceScene s(0x1008, c_SmallProfile);
            BuildCrowd(s, 3);
            RenderFrame frame;
            auto service = MakeService(s.graph);
            const auto& leaves = s.graph->GetSkinnedMeshInstances();
            CHECK(leaves.size() == 3);
            for (const auto& instance : leaves)
                instance->skinningInitialized = true;   // the rule of the frames alone selects
            s.graph->Refresh(0);
            for (uint32_t f = 2; f <= 60; ++f)
            {
                const SkinnedMeshInstance* leaf = leaves[f / 9 % 3];
                auto move = [&](double offset)
                {
                    for (const SkinnedMeshJoint& joint : leaf->joints)
                        if (nvrhi::AutoPtr<SceneGraphNode> node = joint.node.Lock())
                            node->SetTranslation(dm::double3(offset, 0.01 * double(f), 0.0));
                };
                const uint32_t what = f % 9;   // 1: before, 4: between, 7: both, else none
                if (what == 1 || what == 7)
                    move(1.0);
                if (simulate)
                    CHECK(s.graph->Simulate(FrameTime{ double(f) / 60.0, 1.0 / 60.0, f }) == nvrhi::FS_OK);
                if (what == 4 || what == 7)
                    move(2.0);
                s.graph->Refresh(f);
                const ExtractDesc desc = views.Desc(f);
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
                CheckSnapshot(s.graph, snapshot, desc);
                Extracted extracted;
                bool moved = false;
                for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
                {
                    for (uint32_t k = 0; k < uint32_t(leaves.size()); ++k)
                    {
                        if (leaves[k].Get() == snapshot.skinnedInstances[i].leaf)
                            extracted.instances.push_back(k);
                    }
                    moved = moved || snapshot.skinnedInstances[i].leaf == leaf;
                }
                extracted.matrices.assign(snapshot.jointMatrices, snapshot.jointMatrices + snapshot.jointMatrixCount);
                // moved in this frame or the one before; after that nothing is
                CHECK(what % 3 == 0 || f < 3 || moved);
                CHECK(what % 3 != 0 || f < 3 || extracted.instances.empty());
                results[simulate].push_back(std::move(extracted));
                service->EndFrame(true);
            }
            service = nullptr;
        }
        for (size_t f = 0; f < results[0].size(); ++f)
        {
            CHECK(results[0][f].instances == results[1][f].instances);
            CHECK(results[0][f].matrices.size() == results[1][f].matrices.size());
            CHECK(results[0][f].matrices.empty() ||
                  SameBits(results[0][f].matrices.data(), results[1][f].matrices.data(),
                           results[0][f].matrices.size() * sizeof(dm::float4x4)));
        }
    }

    // A skinned instance whose joint nodes were destroyed: their matrices are the identity.
    void test_destroyed_joint()
    {
        TestViews views;
        TraceScene s(0x1004, c_SmallProfile);
        BuildCrowd(s, 1);
        s.graph->Refresh(0);
        CHECK(s.graph->GetSkinnedMeshInstances().size() == 1);
        const SkinnedMeshInstance* skinned = s.graph->GetSkinnedMeshInstances()[0];
        CHECK(skinned->joints.size() == 5);
        {
            // joints 2-4 are a chain under joint 1: detached and dropped, they are destroyed
            nvrhi::AutoPtr<SceneGraphNode> joint2 = skinned->joints[2].node.Lock();
            nvrhi::AutoPtr<SceneGraphNode> detached = s.graph->Detach(joint2);
        }
        CHECK(!skinned->joints[2].node.Lock() && !skinned->joints[4].node.Lock() && skinned->joints[1].node.Lock());
        s.graph->Refresh(1);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        const ExtractDesc desc = views.Desc(1);
        const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
        CheckSnapshot(s.graph, snapshot, desc);
        CHECK(snapshot.skinnedInstanceCount == 1 && snapshot.skinnedInstances[0].jointMatrices.count == 5);
        const dm::float4x4 identity = dm::float4x4::identity();
        const dm::float4x4* matrices = snapshot.jointMatrices + snapshot.skinnedInstances[0].jointMatrices.first;
        for (uint32_t j = 2; j < 5; ++j)
            CHECK(SameBits(&matrices[j], &identity, sizeof(identity)));
        service->EndFrame(true);
        service = nullptr;
    }

    // A light's shadow map replaced between Extract and EndFrame: the snapshot's map stays alive until EndFrame.
    void test_shadow_map_reassigned()
    {
        std::atomic<int> live{ 0 };
        TestViews views;
        TraceScene s(0x1005, c_SmallProfile);
        BuildMixed(s, 24);
        Light* light = s.graph->GetLights()[0];
        light->shadowMap = MAKE_RC_OBJ_PTR(FakeShadowMap, &live).Get();
        CHECK(live.load() == 1);
        s.graph->Refresh(0);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(0));
        const IShadowMap* extracted = snapshot.lights[0].shadowMap;
        CHECK(extracted == light->shadowMap.Get());
        light->shadowMap = MAKE_RC_OBJ_PTR(FakeShadowMap, &live).Get();   // the application makes a new one mid-frame
        CHECK(live.load() == 2);                                     // the old one is still held by the frame
        CHECK(extracted->GetNumberOfCascades() == 1);
        service->EndFrame(true);
        CHECK(live.load() == 1);
        light->shadowMap = nullptr;
        CHECK(live.load() == 0);
        service = nullptr;
    }

    // After SetJobSystem the service holds neither the old scheduler nor, through it, the old job system.
    void test_scheduler_release()
    {
        TestViews views;
        TraceScene s(0x1001, c_SmallProfile);
        BuildFlat(s, 12);
        nvrhi::AutoPtr<IJobSystem> jobs;
        JobSystemDesc jobsDesc;
        jobsDesc.workerThreadCount = 2;
        CHECK(CreateJobSystem(jobsDesc, &jobs) == nvrhi::FS_OK);
        const uint32_t alone = ReferenceCount(jobs);
        s.graph->SetJobSystem(jobs);
        const uint64_t generation = s.graph->GetSchedulerGeneration();
        RenderFrame frame;
        auto service = MakeService(s.graph);
        s.graph->Refresh(0);
        ExtractFrame(service, frame, views.Desc(0));
        service->EndFrame(true);
        CHECK(ReferenceCount(jobs) > alone);   // the graph's scheduler holds it
        s.graph->SetJobSystem(nullptr);
        CHECK(s.graph->GetSchedulerGeneration() == generation + 1);
        CHECK(ReferenceCount(jobs) == alone);  // the old scheduler went at once, with the service's systems on it
        // the next frame registers on the new scheduler
        s.graph->Refresh(1);
        const ExtractDesc desc = views.Desc(1);
        CheckSnapshot(s.graph, ExtractFrame(service, frame, desc), desc);
        service->EndFrame(true);
        service = nullptr;
    }

    // Extract twice in one frame (the blackboard entry is reused), the add-pass calls twice, and the add-pass calls
    // outside an open frame.
    void test_repeated_calls()
    {
        TestViews views;
        TraceScene s(0x1005, c_SmallProfile);
        BuildMixed(s, 24);
        s.graph->Refresh(0);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        const ExtractDesc desc = views.Desc(0);
        SceneFrame* sceneFrame = nullptr;
        SceneFrame unopened;
        CHECK(service->AddUploadPasses(frame.graph, &unopened) == nvrhi::FE_INVALID_ARGS);   // no Extract yet
        frame.Begin();
        CHECK(service->Extract(frame.graph, &frame.data, desc, &sceneFrame) == nvrhi::FS_OK);
        const RenderSnapshot* first = sceneFrame->snapshot;
        SceneFrame* again = nullptr;
        CHECK(service->Extract(frame.graph, &frame.data, desc, &again) == nvrhi::FS_OK);   // same frame, no Reset
        CHECK(again == sceneFrame && again->snapshot != first);
        CHECK(HashSnapshot(*again->snapshot, false) == HashSnapshot(*first, false));
        CHECK(s.graph->IsInRenderRead());
        for (int i = 0; i < 2; ++i)
        {
            CHECK(service->AddUploadPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            CHECK(service->AddSkinningPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            CHECK(service->AddShadowPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
        }
        CHECK(frame.graph->GetPassCount() == 0 && sceneFrame->bufferReadCount == 0);
        service->EndFrame(true);
        CHECK(!s.graph->IsInRenderRead());
        CHECK(service->AddUploadPasses(frame.graph, sceneFrame) == nvrhi::FE_INVALID_ARGS);
        CHECK(service->AddSkinningPasses(frame.graph, sceneFrame) == nvrhi::FE_INVALID_ARGS);
        CHECK(service->AddShadowPasses(frame.graph, sceneFrame) == nvrhi::FE_INVALID_ARGS);
        service = nullptr;
    }

    // Not a check, a measurement (run by name only, Release): 100k instances, one view. The extraction's cost without
    // culling, and what a Refresh with nothing changed costs after an extraction (SceneGraph::RunReadOnlyPhase: it
    // still returns at once) and without one.
    void measure_extraction()
    {
        TestViews views;
        TraceScene s(0x6001, c_LargeProfile);
        BuildInstances(s, 100000 - 2);
        RenderFrame frame;
        auto service = MakeService(s.graph);
        ExtractDesc desc = views.Desc(0);
        desc.viewCount = 1;
        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::vector<double> skipped, extract, refreshAfter;
        for (uint32_t f = 0; f < 40; ++f)
        {
            Clock::time_point t0 = Clock::now();
            s.graph->Refresh(f);                         // after the frame before: nothing changed
            Clock::time_point t1 = Clock::now();
            frame.Begin();
            CHECK(service->Extract(frame.graph, &frame.data, desc, nullptr) == nvrhi::FS_OK);
            Clock::time_point t2 = Clock::now();
            service->EndFrame(true);
            if (f >= 8)
            {
                refreshAfter.push_back(ms(t1 - t0));
                extract.push_back(ms(t2 - t1));
            }
        }
        for (uint32_t f = 0; f < 32; ++f)
        {
            Clock::time_point t0 = Clock::now();
            s.graph->Refresh(1000 + f);                  // no extraction in between: returns at once
            skipped.push_back(ms(Clock::now() - t0));
        }
        std::printf("  100k instances, one view: Extract %.3f ms; Refresh with nothing changed %.4f ms after an "
                    "extraction, %.5f ms without\n", median(extract), median(refreshAfter), median(skipped));
        service = nullptr;
    }

    // ------------------------------------------------- deferred releases and pipelining checks (E7 rev 2, part I6)

    std::atomic<uint32_t> g_CountedMeshesDestroyed{ 0 };

    // A mesh whose destruction is counted (MeshInfo has no weak references).
    struct CountedMesh : MeshInfo
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
        ~CountedMesh() override { g_CountedMeshesDestroyed.fetch_add(1); }
    };

    nvrhi::AutoPtr<nvrhi::IWeakReference> WeakOf(nvrhi::IWeakReferenceSource* object)
    {
        nvrhi::AutoPtr<nvrhi::IWeakReference> weak;
        object->GetWeakReference(weak.GetAddressOf());
        CHECK(weak);
        return weak;
    }

    bool Expired(nvrhi::IWeakReference* weak)
    {
        return weak->IsExpired() != 0;
    }

    nvrhi::AutoPtr<MeshInfo> NewCountedMesh(SceneTypeFactory* factory, Material* material, uint32_t geometries)
    {
        nvrhi::AutoPtr<MeshInfo> mesh = MAKE_RC_OBJ_PTR(CountedMesh);
        mesh->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
        for (uint32_t i = 0; i < geometries; ++i)
        {
            auto geometry = factory->CreateMeshGeometry();
            geometry->material = material;
            geometry->objectSpaceBounds = mesh->objectSpaceBounds;
            mesh->geometries.push_back(geometry);
        }
        return mesh;
    }

    // The retire list of a Defer scope: what the graph, its world and its tables drop while the scope is open (a
    // detached subgraph nobody keeps, a leaf replaced by SetLeaf, a mesh and its material whose last instance went, a
    // runtime mesh released) lives until the EndRenderRead that closes the last Defer scope, not before (a Warn scope
    // nested in it closes first), and goes then, on that thread. Refresh's mesh reconciliation waits for it too. In a
    // Warn scope the same drop releases at once.
    void test_defer_retire_list()
    {
        const uint32_t warningsBefore = g_Warnings.load();
        const uint32_t destroyedBefore = g_CountedMeshesDestroyed.load();
        auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        auto root = MAKE_RC_OBJ_PTR(SceneGraphNode);
        graph->SetRootNode(root);

        auto material = factory->CreateMaterial();
        material->domain = MaterialDomain::Opaque;
        auto weakMaterial = WeakOf(material);
        nvrhi::AutoPtr<MeshInfo> mesh = NewCountedMesh(factory, material, 2);
        nvrhi::AutoPtr<MeshInfo> runtimeMesh = NewCountedMesh(factory, material, 1);
        nvrhi::AutoPtr<MeshInfo> edited = factory->CreateMesh();   // kept by the test: edited in place below
        edited->geometries.push_back(factory->CreateMeshGeometry());
        edited->geometries[0]->material = material;
        material = nullptr;

        nvrhi::AutoPtr<SceneGraphLeaf> leafA = factory->CreateMeshInstance(mesh);
        nvrhi::AutoPtr<SceneGraphNode> nodeA = graph->AttachLeafNode(root, leafA);
        auto weakNodeA = WeakOf(nodeA);
        auto weakLeafA = WeakOf(leafA);
        nvrhi::AutoPtr<SceneGraphLeaf> light = MAKE_RC_OBJ_PTR(PointLight);
        nvrhi::AutoPtr<SceneGraphNode> nodeB = graph->AttachLeafNode(root, light);
        auto weakLight = WeakOf(light);
        graph->AttachLeafNode(root, factory->CreateMeshInstance(edited));
        const MeshId runtimeId = graph->AcquireRuntimeMesh(runtimeMesh);
        CHECK(runtimeId != MeshId::Invalid);
        const uint32_t editedSlot = MeshTable::SlotOf(graph->GetMeshes().Find(edited));
        CHECK(graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 1);
        graph->Refresh(0);
        mesh = nullptr;
        runtimeMesh = nullptr;
        leafA = nullptr;
        light = nullptr;
        CHECK(!graph->IsInRenderRead() && !graph->IsRenderReadDeferring());

        graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        CHECK(graph->IsInRenderRead() && graph->IsRenderReadDeferring());
        graph->BeginRenderRead();   // a Warn scope inside the Defer one: the graph still defers
        CHECK(graph->IsRenderReadDeferring());
        (void)graph->Detach(nodeA);   // nobody keeps the subgraph
        nodeA = nullptr;
        nodeB->SetLeaf(MAKE_RC_OBJ_PTR(SpotLight));
        graph->ReleaseRuntimeMesh(runtimeId);
        edited->geometries.push_back(factory->CreateMeshGeometry());   // an in-place edit, then a structural change
        edited->geometries[1]->material = edited->geometries[0]->material;
        graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(PointLight));
        graph->Refresh(1);
        CHECK(!Expired(weakNodeA) && !Expired(weakLeafA) && !Expired(weakLight) && !Expired(weakMaterial));
        CHECK(g_CountedMeshesDestroyed.load() == destroyedBefore);
        CHECK(graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 1);   // reconciled after the scope

        graph->EndRenderRead();   // the Warn scope
        CHECK(graph->IsInRenderRead() && graph->IsRenderReadDeferring());
        CHECK(!Expired(weakNodeA) && !Expired(weakLeafA) && !Expired(weakLight) && !Expired(weakMaterial));
        CHECK(g_CountedMeshesDestroyed.load() == destroyedBefore);
        CHECK(graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 1);

        graph->EndRenderRead();   // the last Defer scope: released here, on this thread
        CHECK(!graph->IsInRenderRead() && !graph->IsRenderReadDeferring());
        CHECK(Expired(weakNodeA) && Expired(weakLeafA) && Expired(weakLight));
        CHECK(g_CountedMeshesDestroyed.load() == destroyedBefore + 2);
        CHECK(!Expired(weakMaterial));   // the edited mesh still uses it
        CHECK(graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 2);

        // A Warn scope keeps nothing: a dropped subgraph goes at once (and Debug builds warn about the Detach).
        nvrhi::AutoPtr<SceneGraphNode> nodeC = graph->AttachLeafNode(root, MAKE_RC_OBJ_PTR(PointLight));
        auto weakNodeC = WeakOf(nodeC);
        graph->Refresh(2);
        graph->BeginRenderRead();
        (void)graph->Detach(nodeC);
        nodeC = nullptr;
        CHECK(Expired(weakNodeC));
        graph->EndRenderRead();
        const uint32_t warnings = g_Warnings.load() - warningsBefore;
#ifdef NDEBUG
        CHECK(warnings == 0);
#else
        CHECK(warnings == 1);
#endif
        g_Warnings.store(warningsBefore);
        std::printf("  two meshes, a node, two leaves kept until the last EndRenderRead; reconciliation deferred\n");
    }

    // Writes through the API: the forms the isolation case lists. Returns how many.
    uint32_t WriteThroughApi(TraceScene& s)
    {
        uint32_t writes = 0;
        for (uint32_t i = 0; i < 3; ++i, ++writes)
            Move(s, s.movers[i]);
        std::vector<SceneGraphNode*> nodes = CollectNodes(s.graph);
        nvrhi::AutoPtr<SceneGraphNode> detached = s.graph->Detach(nodes[nodes.size() / 2]);
        ++writes;
        s.graph->Attach(s.graph->GetRootNode(), detached);
        ++writes;
        nodes = CollectNodes(s.graph);
        nodes[3]->SetLeaf(s.factory->CreateMeshInstance(s.meshes[1]));
        ++writes;
        nodes[4]->InvalidateContent();
        ++writes;
        s.graph->AttachLeafNode(s.graph->GetRootNode(), MAKE_RC_OBJ_PTR(PointLight));
        ++writes;
        auto newRoot = NewNode(s);
        auto oldRoot = s.graph->SetRootNode(newRoot);
        ++writes;
        s.graph->Attach(newRoot, oldRoot);
        ++writes;
        return writes;
    }

    // The writes that warn in a Warn scope (Debug builds) warn in no build inside a Defer scope, nor inside a Warn scope
    // nested in one.
    void test_defer_no_warning()
    {
        TraceScene s(0x4006, c_SmallProfile);
        BuildMixed(s, 120);
        s.graph->Refresh(0);
        const uint32_t warningsBefore = g_Warnings.load();

        s.graph->BeginRenderRead();
        const uint32_t serialWrites = WriteThroughApi(s);
        s.graph->EndRenderRead();
        const uint32_t serialWarnings = g_Warnings.load() - warningsBefore;
#ifdef NDEBUG
        CHECK(serialWarnings == 0);
#else
        CHECK(serialWarnings == serialWrites);
#endif
        s.graph->Refresh(1);

        const uint32_t deferBefore = g_Warnings.load();
        s.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        const uint32_t deferWrites = WriteThroughApi(s);
        CHECK(s.graph->Simulate(FrameTime{ 0.0, 1.0 / 60.0, 1 }) == nvrhi::FS_OK);   // warns in a Warn scope
        s.graph->Refresh(2);
        s.graph->BeginRenderRead();
        const uint32_t nestedWrites = WriteThroughApi(s);
        s.graph->EndRenderRead();
        s.graph->EndRenderRead();
        CHECK(g_Warnings.load() == deferBefore);
        CHECK(!s.graph->IsInRenderRead());
        s.graph->Refresh(3);
        g_Warnings.store(warningsBefore);
        std::printf("  Warn scope: %u writes, %u warnings; Defer scope: %u writes (and %u in a nested Warn scope), "
                    "0 warnings\n", serialWrites, serialWarnings, deferWrites, nestedWrites);
    }

    uint32_t DrainItems(IDrawStrategy& strategy, SceneGraphNode* root, const IView& view)
    {
        strategy.PrepareForView(root, view);
        uint32_t count = 0;
        while (strategy.GetNextItem())
            ++count;
        return count;
    }

    // In a pipelined frame (a Defer scope open on the root's graph) a walking strategy refuses: no draw items and one
    // warning per strategy object; a Warn scope does not refuse; a bound strategy (a packet adapter) records.
    void test_walking_refused()
    {
        TestViews views;
        TraceScene s(0x4007, c_SmallProfile);
        BuildMixed(s, 200);
        s.graph->Refresh(0);
        const uint32_t warningsBefore = g_Warnings.load();
        auto opaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        auto transparent = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
        const IView& view = *views.perspective;
        const uint32_t opaqueItems = DrainItems(*opaque, s.graph->GetRootNode(), view);
        const uint32_t transparentItems = DrainItems(*transparent, s.graph->GetRootNode(), view);
        CHECK(opaqueItems > 0);

        s.graph->BeginRenderRead();   // a serial frame's scope: the walk runs
        CHECK(DrainItems(*opaque, s.graph->GetRootNode(), view) == opaqueItems);
        CHECK(DrainItems(*transparent, s.graph->GetRootNode(), view) == transparentItems);
        s.graph->EndRenderRead();
        CHECK(g_Warnings.load() == warningsBefore);

        s.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        for (uint32_t i = 0; i < 2; ++i)
        {
            CHECK(DrainItems(*opaque, s.graph->GetRootNode(), view) == 0);
            CHECK(DrainItems(*transparent, s.graph->GetRootNode(), view) == 0);
        }
        CHECK(g_Warnings.load() == warningsBefore + 2);   // one per strategy object, in every build
        s.graph->EndRenderRead();

        // A packet adapter records in a pipelined frame.
        RenderFrame frame;
        auto service = MakeService(s.graph);
        const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(1));
        s.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        auto packets = MAKE_RC_OBJ_PTR(PacketDrawStrategy);
        packets->SetPackets(&snapshot, DrawPacketPass::Opaque);
        const uint32_t packetItems = DrainItems(*packets, s.graph->GetRootNode(), view);
        CHECK(packetItems == snapshot.views[0].packets[uint32_t(DrawPacketPass::Opaque)].count);
        CHECK(packetItems > 0);
        opaque->SetPackets(&snapshot);   // bound: not a walk, so not refused
        CHECK(DrainItems(*opaque, s.graph->GetRootNode(), view) == packetItems);
        opaque->SetPackets(nullptr);
        s.graph->EndRenderRead();
        service->EndFrame(true);
        CHECK(g_Warnings.load() == warningsBefore + 2);

        // After the scope the walk runs again, without another warning.
        CHECK(DrainItems(*opaque, s.graph->GetRootNode(), view) == opaqueItems);
        CHECK(g_Warnings.load() == warningsBefore + 2);
        g_Warnings.store(warningsBefore);
        std::printf("  walked %u opaque and %u transparent items serially, none in a Defer scope; %u packet items\n",
                    opaqueItems, transparentItems, packetItems);
        service = nullptr;
    }

    // The owner-thread check of a pipelined recording (GpuScene; Scene forwards to it): calls on the thread that began
    // the recording warn once per recording in Debug builds, calls on another thread do not, nothing is checked before
    // Begin or after End.
    void test_pipelined_owner_check()
    {
        GpuSceneDesc desc;
        desc.uploadMode = GpuSceneUploadMode::Scatter;
        auto gpuScene = MAKE_RC_OBJ_PTR(GpuScene, nullptr, desc);
        const uint32_t warningsBefore = g_Warnings.load();
        gpuScene->CommitScatter(0);
        (void)gpuScene->IsScatterPending();
        CHECK(!gpuScene->IsPipelinedRecording());
        CHECK(g_Warnings.load() == warningsBefore);

#ifdef NDEBUG
        constexpr uint32_t c_PerRecording = 0;
#else
        constexpr uint32_t c_PerRecording = 1;   // one warning per recording
#endif
        for (uint32_t round = 0; round < 2; ++round)
        {
            gpuScene->BeginPipelinedRecording();
            CHECK(gpuScene->IsPipelinedRecording());
            std::thread recorder([&gpuScene]()
            {
                GpuSceneScatterRecords records;
                gpuScene->GetScatterRecords(records);
                (void)gpuScene->HasPendingAssets();
                gpuScene->CommitScatter(records.serial);
            });
            recorder.join();
            CHECK(g_Warnings.load() == warningsBefore + round * c_PerRecording);   // the recording thread's calls
            gpuScene->CommitScatter(0);                           // the main thread's, before the join
            (void)gpuScene->IsScatterPending();
            CHECK(gpuScene->Update(nullptr) == nvrhi::FE_INVALID_ARGS);
            gpuScene->EndPipelinedRecording();
            CHECK(!gpuScene->IsPipelinedRecording());
            gpuScene->CommitScatter(0);
            CHECK(g_Warnings.load() == warningsBefore + (round + 1) * c_PerRecording);
        }
        g_Warnings.store(warningsBefore);
    }

    struct Case
    {
        const char* name;
        void (*run)();
    };
}

int main(int argc, char** argv)
{
#if defined(_MSC_VER)
    // A failed assert reports on stderr and ends the run instead of opening a dialog.
    _set_error_mode(_OUT_TO_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });

    const Case cases[] = {
        { "trace_scenes", test_trace_scenes },
        { "isolation", test_isolation },
        { "refresh_skip", test_refresh_skip },
        { "workers", test_workers },
        { "runtime_rows", test_runtime_rows },
        { "component_ids", test_component_ids },
        { "arena_reset", test_arena_reset },
        { "steady_state", test_steady_state },
        { "second_service", test_second_service },
        { "end_frame", test_end_frame },
        { "light_shadow_table", test_light_shadow_table },
        { "read_only_phase", test_read_only_phase },
        { "failure_injection", test_failure_injection },
        { "destroyed_joint", test_destroyed_joint },
        { "skinned_detach", test_skinned_detach },
        { "skinned_pending", test_skinned_pending },
        { "skinned_simulate", test_skinned_simulate },
        { "shadow_map_reassigned", test_shadow_map_reassigned },
        { "scheduler_release", test_scheduler_release },
        { "repeated_calls", test_repeated_calls },
        { "defer_retire_list", test_defer_retire_list },
        { "defer_no_warning", test_defer_no_warning },
        { "walking_refused", test_walking_refused },
        { "pipelined_owner_check", test_pipelined_owner_check },
        { "measure", measure_extraction },   // by name only
    };
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 ? std::strcmp(argv[1], c.name) != 0 : c.run == measure_extraction)
                continue;
            std::printf("%s\n", c.name);
            c.run();
            CHECK(g_Warnings.load() == 0);
            std::printf("%s: passed\n", c.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    std::printf("render extract: ok\n");
    return 0;
}
