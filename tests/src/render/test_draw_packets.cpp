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

// Visibility and draw packets without a device (aggregate docs/plans/2026-11-21-donut-ecs-e5.md, item E5.2;
// docs/ecs-design.md 7.3, D11):
//
//   legacy_equivalence   3 seeds of 30 frames of a seeded edit script (moves, rotations, attaches, detaches,
//                        re-attaches, SetLeaf, InvalidateContent, material and index edits; materials of every domain,
//                        double-sided ones, multi-geometry, skinned and curve meshes, nodes with children), each frame
//                        for one perspective view and four orthographic cascade views:
//                        - each view's visible instances equal a brute-force legacy-style culling (the own box of every
//                          mesh instance with a slot, dm::frustum::intersectsWith);
//                        - each pass's packets, as a multiset of (instance, geometry, material, cull mode, and the
//                          transparent distance), equal the DrawItems of InstancedOpaqueDrawStrategy and
//                          TransparentDrawStrategy walking the same graph, except items the legacy kept only through a
//                          node's subgraph box: each such item's own box fails the frustum test;
//                        - the packets are sorted by key, their keys and fields are those of their items; the
//                          transparent order is the legacy order wherever the legacy distances are distinct (both
//                          non-increasing in distance) where the legacy order is defined (its comparator is no strict
//                          weak ordering in a view with a skinned multi-geometry instance: such views are counted,
//                          multisets only);
//                        - every frame extracted again at 2 and 8 workers gives the same bytes; the per-frame hashes
//                          match the golden file tests/src/render/golden/render_extract_packets.txt
//                          (ECS_UPDATE_GOLDEN=1 rewrites it), which Release and Debug share
//   combined_double_sided ExtractDesc::drawDoubleSidedMaterialsSeparately off: one None packet per double-sided
//                        transparent geometry, as the legacy strategy with DrawDoubleSidedMaterialsSeparately off
//   steady_state         10 frames of moves after warm-up, five views: no operator new, no ECS heap block
//   radix_sort           the sort against std::stable_sort: masks, packed and unpacked keys, serial and parallel
//   stale_leaves         Debug: a mesh box, then a material domain, edited without InvalidateContent, are reported at the
//                        64th extraction for an instance in no view
//   review_cases         a leaf with dynamic bounds; meshes and materials shared with another graph; a domain edited
//                        without InvalidateContent; a transparent instance with a large subtree; NaN boxes; the view
//                        count changing between frames
//   parallel_sort_workers  a segment sorted in parallel inside the segments' parallel loop: equal at 1, 2, 8 workers
//
//     test_draw_packets [case]

#include <donut/render/DrawStrategy.h>
#include <donut/render/GeometryPasses.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include <donut/core/query_cast.h>
#include <donut/tests/utils.h>

#include "EcsWorldImpl.h"       // ecs::GetHeapAllocationCount
#include "RenderVisibility.h"   // the radix sort's own cases

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <malloc.h>
#include <stdlib.h>
#endif

#ifndef DONUT_TEST_GOLDEN_DIR
#error "DONUT_TEST_GOLDEN_DIR is defined for this test in tests/test-render.cmake"
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

// Every operator new is counted, the frame arena's blocks and the visibility scratch included.
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

namespace
{
    std::atomic<uint32_t> g_Warnings{ 0 };

    // ------------------------------------------------------------------------------------------ the scene script

    // xorshift64*, as test_gpu_scene's script.
    struct Rng
    {
        uint64_t state;
        uint64_t Next()
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 0x2545F4914F6CDD1Dull;
        }
        uint32_t Below(uint32_t bound) { return uint32_t(Next() % bound); }
        double Unit() { return double(Next() >> 11) * (1.0 / 9007199254740992.0); }
        double Range(double lo, double hi) { return lo + (hi - lo) * Unit(); }
    };

    nvrhi::AutoPtr<SceneGraphNode> MakeNode(const std::string& name)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName(name);
        return node;
    }

    // The test_gpu_scene edit script (tests/src/engine/test_gpu_scene.cpp, run_edit_script), widened for culling:
    // materials of every domain, some double-sided; geometries with boxes of their own (a mesh's box is their union),
    // so that the per-geometry tests matter; positions spread across the views' edges; rotations and scalings.
    struct PacketScene
    {
        explicit PacketScene(uint64_t seed) : rng{ seed ^ 0x9E3779B97F4A7C15ull } { }

        Rng rng;
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        std::vector<nvrhi::AutoPtr<Material>> materials;
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> pool;   // detached subgraphs, attached again later
        int made = 0;

        void MakeAssets()
        {
            // domain, double-sided
            const std::pair<MaterialDomain, bool> kinds[] = {
                { MaterialDomain::Opaque, false },
                { MaterialDomain::AlphaTested, true },
                { MaterialDomain::AlphaBlended, false },
                { MaterialDomain::Transmissive, true },
                { MaterialDomain::TransmissiveAlphaTested, false },
                { MaterialDomain::TransmissiveAlphaBlended, true },
                { MaterialDomain::Opaque, true },
                { MaterialDomain::AlphaBlended, true },
            };
            for (size_t i = 0; i < std::size(kinds); ++i)
            {
                auto material = MAKE_RC_OBJ_PTR(Material);
                material->name = "material" + std::to_string(i);
                material->domain = kinds[i].first;
                material->doubleSided = kinds[i].second;
                materials.push_back(material);
            }
            for (int i = 0; i < 9; ++i)
            {
                auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
                mesh->name = "mesh" + std::to_string(i);
                mesh->type = i == 5 ? MeshType::CurveLinearSweptSpheres
                                    : i == 6 ? MeshType::CurveDisjointOrthogonalTriangleStrips : MeshType::Triangles;
                const int geometries = 1 + i % 3;
                dm::box3 bounds = dm::box3::empty();
                for (int g = 0; g < geometries; ++g)
                {
                    auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
                    geometry->material = materials[(i * 3 + g) % materials.size()];
                    geometry->numIndices = 3 * (g + 1);
                    geometry->numVertices = 3;
                    // side by side along x, each 1.6 wide
                    const float x = -2.4f + 1.6f * float(g);
                    geometry->objectSpaceBounds = dm::box3(dm::float3(x, -1.f, -1.f), dm::float3(x + 1.6f, 1.f, 1.f));
                    bounds |= geometry->objectSpaceBounds;
                    mesh->geometries.push_back(geometry);
                }
                mesh->objectSpaceBounds = bounds;
                meshes.push_back(mesh);
            }
        }

        nvrhi::AutoPtr<SceneGraphLeaf> NewLeaf()
        {
            const uint32_t pick = rng.Below(20);
            if (pick == 0)
                return MAKE_RC_OBJ_PTR(PointLight);
            if (pick <= 2)   // skinned copies of multi-geometry triangle meshes
                return factory->CreateSkinnedMeshInstance(factory, meshes[rng.Below(2) ? 1 : 2]);
            return MAKE_RC_OBJ_PTR(MeshInstance, meshes[rng.Below(uint32_t(meshes.size()))].Get());
        }

        void Place(SceneGraphNode* node, double spread)
        {
            node->SetTranslation(dm::double3(rng.Range(-spread, spread), rng.Range(-spread * 0.25, spread * 0.25),
                                             rng.Range(-spread * 0.5, spread * 1.5)));
            if (rng.Below(3) == 0)
            {
                const dm::double3 axis =
                    dm::normalize(dm::double3(rng.Range(-1, 1), rng.Range(0.1, 1), rng.Range(-1, 1)));
                node->SetRotation(dm::rotationQuat(axis, rng.Range(0.0, 6.28)));
            }
            if (rng.Below(5) == 0)
                node->SetScaling(dm::double3(rng.Range(0.5, 2.0)));
        }

        // an orphaned subgraph of 1-6 nodes, some with transforms and some without leaves
        nvrhi::AutoPtr<SceneGraphNode> NewSubgraph()
        {
            auto top = MakeNode("s" + std::to_string(made++));
            Place(top, 40.0);
            top->SetLeaf(NewLeaf());
            const uint32_t children = rng.Below(6);
            for (uint32_t c = 0; c < children; ++c)
            {
                auto child = MakeNode("c" + std::to_string(c));
                if (rng.Below(3))
                    child->SetTranslation(dm::double3(rng.Range(-3, 3), rng.Range(-3, 3), rng.Range(-3, 3)));
                if (rng.Below(5))
                    child->SetLeaf(NewLeaf());
                graph->Attach(top, child);
            }
            return top;
        }

        std::vector<SceneGraphNode*> Attached() const
        {
            std::vector<SceneGraphNode*> nodes;
            for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
                nodes.push_back(walker.Get());
            return nodes;
        }

        void Build(uint32_t subgraphs)
        {
            MakeAssets();
            graph->SetRootNode(MakeNode("root"));
            for (uint32_t i = 0; i < subgraphs; ++i)
            {
                const auto nodes = Attached();
                // mostly under the root, so that the boxes spread; some deeper
                SceneGraphNode* parent = rng.Below(4) ? nodes[0] : nodes[rng.Below(uint32_t(nodes.size()))];
                graph->Attach(parent, NewSubgraph());
            }
        }

        // One frame's edits (every 7th frame is idle), as the test_gpu_scene script.
        void Edit(uint32_t frame)
        {
            if (frame % 7 == 0)
                return;
            auto nodes = Attached();
            const uint32_t moves = uint32_t(nodes.size()) / 15 + 1;
            for (uint32_t k = 0; k < moves; ++k)
            {
                SceneGraphNode* node = nodes[rng.Below(uint32_t(nodes.size()))];
                if (node != nodes[0])
                    Place(node, 40.0);
            }
            for (uint32_t k = rng.Below(3); k > 0; --k)
                graph->Attach(nodes[rng.Below(uint32_t(nodes.size()))], NewSubgraph());
            if (!pool.empty() && rng.Below(2))
            {
                graph->Attach(nodes[rng.Below(uint32_t(nodes.size()))], pool.back());
                pool.pop_back();
            }
            nodes = Attached();
            if (nodes.size() > 1 && rng.Below(2))
                pool.push_back(graph->Detach(nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]));
            nodes = Attached();
            if (nodes.size() > 1 && rng.Below(10) < 3)
                nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]->SetLeaf(NewLeaf());
            if (nodes.size() > 1 && rng.Below(10) < 2)
                nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]->InvalidateContent();
            if (rng.Below(10) < 2)
            {
                Material* material = materials[rng.Below(uint32_t(materials.size()))];
                material->roughness = float(rng.Unit());
                material->dirty = true;
            }
            if (rng.Below(10) == 0)
                meshes[rng.Below(uint32_t(meshes.size()))]->geometries[0]->numIndices += 3;
        }
    };

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

    // One perspective view and four orthographic cascade-like views of growing extent.
    struct TestViews
    {
        static constexpr uint32_t c_Count = 5;
        nvrhi::AutoPtr<PlanarView> views[c_Count];
        const IView* list[c_Count] = {};

        TestViews()
        {
            for (uint32_t i = 0; i < c_Count; ++i)
            {
                views[i] = MAKE_RC_OBJ_PTR(PlanarView);
                list[i] = views[i].Get();
            }
            views[0]->SetViewport(nvrhi::Viewport(1280.f, 720.f));
            const dm::affine3 eye = dm::translation(dm::float3(0.f, 2.f, -20.f));
            views[0]->SetMatrices(dm::inverse(eye), dm::perspProjD3DStyleReverse(dm::radians(60.f), 16.f / 9.f, 0.1f));
            views[0]->UpdateCache();
            const dm::float3 sun = dm::normalize(dm::float3(0.3f, -1.f, 0.2f));
            const float extents[4] = { 6.f, 12.f, 24.f, 48.f };
            const dm::float3 centers[4] = { dm::float3(0.f, 0.f, -10.f), dm::float3(4.f, 0.f, 0.f),
                                            dm::float3(-6.f, 0.f, 10.f), dm::float3(0.f, 0.f, 20.f) };
            for (uint32_t c = 0; c < 4; ++c)
            {
                PlanarView* view = views[1 + c];
                view->SetViewport(nvrhi::Viewport(1024.f, 1024.f));
                const dm::affine3 light = dm::lookatZ(sun) * dm::translation(centers[c] - sun * 60.f);
                const float e = extents[c];
                view->SetMatrices(dm::inverse(light), dm::orthoProjD3DStyle(-e, e, -e, e, 0.f, 120.f));
                view->UpdateCache();
            }
        }

        ExtractDesc Desc(uint32_t frameIndex, bool separate = true) const
        {
            ExtractDesc desc;
            desc.views = list;
            desc.viewCount = c_Count;
            desc.frameIndex = frameIndex;
            desc.drawDoubleSidedMaterialsSeparately = separate;
            return desc;
        }
    };

    nvrhi::AutoPtr<RenderService> MakeService(SceneGraph* graph)
    {
        RenderServiceDesc desc;
        desc.sceneGraph = graph;
        return MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
    }

    const RenderSnapshot& ExtractFrame(RenderService* service, RenderFrame& frame, const ExtractDesc& desc)
    {
        frame.Begin();
        SceneFrame* sceneFrame = nullptr;
        CHECK(service->Extract(frame.graph, &frame.data, desc, &sceneFrame) == nvrhi::FS_OK);
        CHECK(sceneFrame && sceneFrame->snapshot);
        return *sceneFrame->snapshot;
    }

    // ------------------------------------------------------------------------------------------ hashes

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

    // The views' ranges, the visible instances and the packets, without pointers (they differ between runs).
    uint64_t HashPackets(const RenderSnapshot& s)
    {
        Hasher h;
        h.Value(s.viewCount);
        for (uint32_t v = 0; v < s.viewCount; ++v)
        {
            h.Value(s.views[v].instances);
            for (const SnapshotRange& range : s.views[v].packets)
                h.Value(range);
        }
        const VisibleInstances& in = s.instances;
        h.Array(in.entities, in.count);
        h.Array(in.instanceSlots, in.count);
        h.Array(in.meshes, in.count);
        h.Array(in.firstGeometryInstanceSlots, in.count);
        h.Array(in.geometryCounts, in.count);
        h.Array(in.worldBounds, in.count);
        const DrawPackets& p = s.packets;
        h.Array(p.keys, p.count);
        h.Array(p.visibleInstances, p.count);
        h.Array(p.geometryIndices, p.count);
        h.Array(p.geometrySlots, p.count);
        h.Array(p.materialSlots, p.count);
        h.Array(p.cullModes, p.count);
        h.Array(p.flags, p.count);
        return h.value;
    }

    // ------------------------------------------------------------------------------------------ the legacy strategies

    bool OpaqueDomain(MaterialDomain domain)
    {
        return domain == MaterialDomain::Opaque || domain == MaterialDomain::AlphaTested;
    }

    // The own world box of an instance, as the Bounds system makes it from the leaf.
    dm::box3 OwnBox(const MeshInstance* instance)
    {
        const dm::box3 local = instance->GetMesh()->objectSpaceBounds;
        return local.isempty() ? dm::box3::empty() : local * instance->GetNode()->GetLocalToWorldTransformFloat();
    }

    uint32_t GeometryIndexOf(const MeshInfo* mesh, const MeshGeometry* geometry)
    {
        for (size_t g = 0; g < mesh->geometries.size(); ++g)
        {
            if (mesh->geometries[g].Get() == geometry)
                return uint32_t(g);
        }
        return ~0u;
    }

    // One draw: (instance, geometry index, material, cull mode, distance bits); the distance is 0 for opaque draws.
    using Draw = std::tuple<const MeshInstance*, uint32_t, const Material*, uint32_t, uint32_t>;

    Draw DrawOf(const DrawItem& item, bool transparent)
    {
        uint32_t distance = 0;
        if (transparent)
            std::memcpy(&distance, &item.distanceToCamera, sizeof(distance));
        return Draw{ item.instance, GeometryIndexOf(item.mesh, item.geometry), item.material, uint32_t(item.cullMode),
                     distance };
    }

    std::vector<DrawItem> Drain(IDrawStrategy* strategy, SceneGraph* graph, const IView& view)
    {
        std::vector<DrawItem> items;
        strategy->PrepareForView(graph->GetRootNode(), view);
        while (const DrawItem* item = strategy->GetNextItem())
            items.push_back(*item);
        return items;
    }

    // What the comparison saw: the known differences, counted.
    struct Tally
    {
        uint64_t views = 0;
        uint64_t instances = 0;
        uint64_t packets[c_DrawPacketPassCount] = {};
        uint64_t subgraphOnly[c_DrawPacketPassCount] = {};   // legacy items kept only through a subgraph box
        uint64_t legacyUndefined = 0;    // views whose legacy transparent order is undefined (comparator, below)
        uint64_t distanceTies = 0;       // transparent neighbours of two instances at one distance (any order)
        uint64_t geometryCulled = 0;
    };

    // One view of one frame: the snapshot against the legacy strategies on the same graph.
    void CompareView(SceneGraph* graph, const RenderSnapshot& s, uint32_t v, bool separate, Tally& tally)
    {
        const ExtractedView& view = s.views[v];
        const IView& iview = *view.view;
        const dm::frustum frustum = iview.GetViewFrustum();
        const VisibleInstances& in = s.instances;
        const DrawPackets& packets = s.packets;
        ++tally.views;

        // visible instances: a brute-force walk, every mesh instance with a slot whose own box passes
        std::vector<const MeshInstance*> expected;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
        {
            const MeshInstance* instance = query_cast<MeshInstance>(walker->GetLeaf());
            if (instance && instance->GetInstanceIndex() >= 0 && frustum.intersectsWith(OwnBox(instance)))
                expected.push_back(instance);
        }
        std::vector<const MeshInstance*> actual(in.leaves + view.instances.first,
                                                in.leaves + view.instances.first + view.instances.count);
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        CHECK(expected == actual);
        tally.instances += actual.size();
        for (uint32_t i = view.instances.first; i < view.instances.first + view.instances.count; ++i)
        {
            const MeshInstance* leaf = in.leaves[i];
            CHECK(in.instanceSlots[i] == uint32_t(leaf->GetInstanceIndex()));
            CHECK(in.meshInfos[i] == leaf->GetMesh());
            CHECK(in.entities[i] == graph->GetEntity(leaf->GetNode()));
        }

        for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
        {
            const bool transparent = p == uint32_t(DrawPacketPass::Transparent);
            const SnapshotRange range = view.packets[p];
            tally.packets[p] += range.count;

            // the packets: sorted by key, their fields those of their geometry, material and instance
            std::vector<Draw> ours;
            for (uint32_t i = range.first; i < range.first + range.count; ++i)
            {
                if (i > range.first)
                    CHECK(packets.keys[i - 1] <= packets.keys[i]);
                const uint32_t vi = packets.visibleInstances[i];
                CHECK(vi >= view.instances.first && vi < view.instances.first + view.instances.count);
                const MeshInfo* mesh = in.meshInfos[vi];
                CHECK(packets.geometryIndices[i] < mesh->geometries.size());
                const MeshGeometry* geometry = mesh->geometries[packets.geometryIndices[i]];
                const Material* material = geometry->material;
                // this graph's slots: the mesh row's geometry range and the material's row
                const MeshTable::MeshRow& meshRow = graph->GetMeshes().GetMeshRow(MeshTable::SlotOf(in.meshes[vi]));
                const uint32_t geometrySlot = meshRow.firstGeometry + packets.geometryIndices[i];
                const uint32_t materialSlot = MaterialTable::SlotOf(graph->GetMaterials().Find(material));
                CHECK(packets.geometrySlots[i] == geometrySlot);
                CHECK(packets.materialSlots[i] == materialSlot);
                CHECK(OpaqueDomain(material->domain) == !transparent);
                const uint8_t flags = packets.flags[i];
                CHECK(((flags & c_DrawPacketFlagOpaquePass) != 0) == !transparent);
                CHECK(((flags & c_DrawPacketFlagTransparentPass) != 0) == transparent);
                CHECK(((flags & c_DrawPacketFlagDoubleSided) != 0) == material->doubleSided);
                const bool multiple = mesh->geometries.size() > 1;
                const bool skinned = mesh->skinPrototype != nullptr;
                const bool geometryTest = transparent ? multiple && skinned : multiple && !skinned;
                CHECK(((flags & c_DrawPacketFlagGeometryCulled) != 0) == geometryTest);
                tally.geometryCulled += geometryTest ? 1 : 0;
                const uint32_t cull = packets.cullModes[i];
                uint32_t distance = 0;
                if (transparent)
                {
                    const float d = GetTransparentDrawKeyDistance(packets.keys[i]);
                    std::memcpy(&distance, &d, sizeof(distance));
                    CHECK(packets.keys[i] == MakeTransparentDrawKey(d, in.instanceSlots[vi], cull));
                }
                else
                {
                    CHECK(packets.keys[i] == MakeOpaqueDrawKey(uint32_t(material->domain), cull, materialSlot,
                                                               geometrySlot, in.instanceSlots[vi]));
                }
                ours.push_back(Draw{ in.leaves[vi], packets.geometryIndices[i], material, cull, distance });
            }

            // the legacy strategy's items on the same graph
            std::vector<DrawItem> items;
            if (transparent)
            {
                auto strategy = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
                strategy->DrawDoubleSidedMaterialsSeparately = separate;
                items = Drain(strategy, graph, iview);
            }
            else
            {
                auto strategy = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
                items = Drain(strategy, graph, iview);
            }
            std::vector<Draw> legacy;
            std::vector<DrawItem> kept;   // the legacy items with a packet, in the legacy order
            std::vector<Draw> sortedOurs = ours;
            std::sort(sortedOurs.begin(), sortedOurs.end());
            for (const DrawItem& item : items)
            {
                const Draw draw = DrawOf(item, transparent);
                if (std::binary_search(sortedOurs.begin(), sortedOurs.end(), draw))
                {
                    legacy.push_back(draw);
                    kept.push_back(item);
                    continue;
                }
                // kept only through the node's subgraph box: its own box fails (and so would its geometry's)
                CHECK(item.instance->GetInstanceIndex() >= 0);
                CHECK(!frustum.intersectsWith(OwnBox(item.instance)));
                ++tally.subgraphOnly[p];
            }
            std::sort(legacy.begin(), legacy.end());
            CHECK(legacy == sortedOurs);   // as multisets: every packet is a legacy item, once per item

            if (!transparent)
                continue;
            // The transparent order. Ours is non-increasing in distance (the key), and so is the legacy one where it
            // is defined (below); with equal multisets the two orders are then equal wherever distances differ. Equal
            // distances: any order (ours: instance slot, then cull order).
            for (size_t i = 1; i < ours.size(); ++i)
            {
                const float a = GetTransparentDrawKeyDistance(packets.keys[range.first + i - 1]);
                const float b = GetTransparentDrawKeyDistance(packets.keys[range.first + i]);
                CHECK(a >= b);
                tally.distanceTies += a == b && std::get<0>(ours[i - 1]) != std::get<0>(ours[i]) ? 1 : 0;
            }
            // The legacy comparator is a strict weak ordering only while each instance's items share one distance. A
            // skinned multi-geometry instance whose geometries lie at different distances breaks it, and std::sort may
            // then misplace any item, not only that instance's: the legacy order is undefined in that view.
            std::map<const MeshInstance*, float> distanceOf;
            bool legacyDefined = true;
            for (const DrawItem& item : kept)
            {
                auto [it, inserted] = distanceOf.emplace(item.instance, item.distanceToCamera);
                if (!inserted && it->second != item.distanceToCamera)
                    legacyDefined = false;
            }
            if (!legacyDefined)
            {
                ++tally.legacyUndefined;
                continue;
            }
            for (size_t i = 1; i < kept.size(); ++i)
                CHECK(kept[i - 1].distanceToCamera >= kept[i].distanceToCamera);
            // within one instance, None and Front before Back, in both orders
            std::map<const MeshInstance*, uint32_t> lastCull;
            for (const Draw& draw : ours)
            {
                auto it = lastCull.find(std::get<0>(draw));
                const uint32_t order = 2u - std::get<3>(draw);
                if (it != lastCull.end())
                {
                    const MeshInfo* mesh = std::get<0>(draw)->GetMesh();
                    if (!(mesh->skinPrototype && mesh->geometries.size() > 1))
                        CHECK(it->second <= order);
                    it->second = order;
                }
                else
                    lastCull[std::get<0>(draw)] = order;
            }
        }
    }

    // ------------------------------------------------------------------------------------------ cases

    constexpr uint64_t c_Seeds[] = { 11ull, 23ull, 0x5bd1e995ull };
    constexpr uint32_t c_Frames = 30;

    std::filesystem::path GoldenPath()
    {
        return std::filesystem::path(DONUT_TEST_GOLDEN_DIR) / "render_extract_packets.txt";
    }

    void CheckGolden(const std::string& actual)
    {
        const std::filesystem::path path = GoldenPath();
        const char* update = std::getenv("ECS_UPDATE_GOLDEN");
        if (update && std::strcmp(update, "1") == 0)
        {
            std::ofstream file(path, std::ios::binary);
            file << actual;
            CHECK(file.good());
            std::printf("  wrote %s\n", path.generic_string().c_str());
            return;
        }
        std::ifstream file(path, std::ios::binary);
        std::stringstream expected;
        expected << file.rdbuf();
        if (expected.str() != actual)
        {
            std::printf("test_draw_packets: the packets differ from %s (ECS_UPDATE_GOLDEN=1 rewrites it)\n"
                        "expected:\n%sactual:\n%s", path.generic_string().c_str(), expected.str().c_str(),
                        actual.c_str());
            CHECK(false);
        }
    }

    void test_legacy_equivalence()
    {
        nvrhi::AutoPtr<IJobSystem> jobs[3];
        const uint32_t workerCounts[3] = { 1, 2, 8 };
        for (uint32_t k = 0; k < 3; ++k)
        {
            JobSystemDesc desc;
            desc.debugName = "test_draw_packets";
            desc.workerThreadCount = workerCounts[k];
            CHECK(CreateJobSystem(desc, &jobs[k]) == nvrhi::FS_OK);
        }

        TestViews views;
        Tally tally;
        std::string golden = "# Draw packets per frame (test_draw_packets.cpp; ECS_UPDATE_GOLDEN=1 rewrites it)\n"
                             "# seed frame visible-instances opaque-packets transparent-packets hash\n";
        for (uint64_t seed : c_Seeds)
        {
            PacketScene scene(seed);
            scene.Build(150);
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            for (uint32_t f = 1; f <= c_Frames; ++f)
            {
                scene.Edit(f);
                scene.graph->Refresh(f);
                const ExtractDesc desc = views.Desc(f);
                uint64_t hashes[3] = {};
                for (uint32_t k = 0; k < 3; ++k)
                {
                    scene.graph->SetJobSystem(jobs[k]);
                    const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
                    hashes[k] = HashPackets(snapshot);
                    if (k == 0)
                    {
                        for (uint32_t v = 0; v < snapshot.viewCount; ++v)
                            CompareView(scene.graph, snapshot, v, true, tally);
                        const RenderServiceStats& stats = service->GetStats();
                        CHECK(stats.visibleInstances == snapshot.instances.count);
                        CHECK(stats.packets[0] + stats.packets[1] == snapshot.packets.count);
                        char line[128];
                        std::snprintf(line, sizeof(line), "%llx %u %u %u %u %016llx\n", (unsigned long long)seed, f,
                                      snapshot.instances.count, stats.packets[0], stats.packets[1],
                                      (unsigned long long)hashes[0]);
                        golden += line;
                    }
                    service->EndFrame(true);
                }
                CHECK(hashes[0] == hashes[1] && hashes[1] == hashes[2]);
            }
            scene.graph->SetJobSystem(nullptr);
        }
        std::printf("  %llu view-frames: %llu visible instances, %llu opaque and %llu transparent packets, %llu "
                    "after a geometry test\n", (unsigned long long)tally.views, (unsigned long long)tally.instances,
                    (unsigned long long)tally.packets[0], (unsigned long long)tally.packets[1],
                    (unsigned long long)tally.geometryCulled);
        std::printf("  known differences: %llu opaque and %llu transparent legacy items kept only through a subgraph "
                    "box; %llu views with a skinned multi-geometry transparent instance (legacy order undefined: "
                    "multisets compared only); %llu transparent neighbours of two instances at one distance "
                    "(legacy order unspecified)\n",
                    (unsigned long long)tally.subgraphOnly[0], (unsigned long long)tally.subgraphOnly[1],
                    (unsigned long long)tally.legacyUndefined, (unsigned long long)tally.distanceTies);
        CHECK(tally.packets[0] > 1000 && tally.packets[1] > 1000 && tally.geometryCulled > 0);
        CheckGolden(golden);
    }

    // ExtractDesc::drawDoubleSidedMaterialsSeparately off, against the legacy strategy with the flag off.
    void test_combined_double_sided()
    {
        TestViews views;
        Tally tally;
        PacketScene scene(0x77);
        scene.Build(120);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        for (uint32_t f = 1; f <= 6; ++f)
        {
            scene.Edit(f);
            scene.graph->Refresh(f);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f, false));
            for (uint32_t v = 0; v < snapshot.viewCount; ++v)
                CompareView(scene.graph, snapshot, v, false, tally);
            uint32_t none = 0;
            for (uint32_t i = 0; i < snapshot.packets.count; ++i)
            {
                if (snapshot.packets.flags[i] & c_DrawPacketFlagTransparentPass)
                {
                    const bool doubleSided = (snapshot.packets.flags[i] & c_DrawPacketFlagDoubleSided) != 0;
                    CHECK(snapshot.packets.cullModes[i] == uint8_t(doubleSided ? nvrhi::RasterCullMode::None
                                                                               : nvrhi::RasterCullMode::Back));
                    none += doubleSided ? 1 : 0;
                }
            }
            CHECK(none > 0);
            service->EndFrame(true);
        }
    }

    // Ten frames of moves after warm-up, five views: no operator new (the arena's blocks and the visibility scratch
    // included), no ECS heap block. A tenth of the nodes swing between two places 15 m apart every frame, across the
    // views' edges, so that every view's visible set and packets change each frame. The scratch and the arena arrays
    // keep high-water marks (with a quarter of headroom): a frame that sees more than any before grows them, which
    // the warm-up's frames have done here.
    void test_steady_state()
    {
        TestViews views;
        PacketScene scene(0x5001);
        scene.Build(1500);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        struct Mover
        {
            SceneGraphNode* node;
            dm::double3 home;
        };
        std::vector<Mover> movers;
        {
            const auto nodes = scene.Attached();
            for (size_t i = 1; i < nodes.size(); i += 10)
                movers.push_back(Mover{ nodes[i], dm::double3(scene.rng.Range(-40, 40), scene.rng.Range(-10, 10),
                                                              scene.rng.Range(-20, 60)) });
        }
        uint32_t f = 0;
        uint32_t visible[2] = {};
        auto run = [&]()
        {
            const dm::double3 swing = (f & 1) ? dm::double3(12.0, 0.0, 9.0) : dm::double3(0.0);
            for (const Mover& mover : movers)
                mover.node->SetTranslation(mover.home + swing);
            scene.graph->Refresh(++f);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f));
            CHECK(snapshot.packets.count > 0);
            visible[f & 1] = snapshot.instances.count;
            service->EndFrame(true);
        };
        for (int i = 0; i < 6; ++i)
            run();
        const uint64_t newBefore = g_NewCount.load();
        const uint64_t heapBefore = ecs::GetHeapAllocationCount();
        for (int i = 0; i < 10; ++i)
            run();
        const uint64_t newCalls = g_NewCount.load() - newBefore;
        const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
        const RenderServiceStats& stats = service->GetStats();
        std::printf("  %u and %u visible instances in 5 views (two poses), %u opaque and %u transparent packets, %llu "
                    "snapshot bytes: %llu operator new, %llu storage blocks over 10 frames\n", visible[0], visible[1],
                    stats.packets[0], stats.packets[1], (unsigned long long)stats.snapshotBytes,
                    (unsigned long long)newCalls, (unsigned long long)heapBlocks);
        CHECK(visible[0] != visible[1]);
        CHECK(newCalls == 0 && heapBlocks == 0);
    }

    // The sort against std::stable_sort of the positions: random keys under masks of every shape, packed (key and
    // position in one word) and unpacked, serial and on 8 workers, with many equal keys.
    void test_radix_sort()
    {
        JobSystemDesc jobDesc;
        jobDesc.debugName = "test_draw_packets.sort";
        jobDesc.workerThreadCount = 8;
        nvrhi::AutoPtr<IJobSystem> jobs;
        CHECK(CreateJobSystem(jobDesc, &jobs) == nvrhi::FS_OK);

        visibility::RadixSorter sorter;
        Rng rng{ 0xabcdef };
        const uint64_t masks[] = {
            0,                                  // all equal
            0xffull,                            // one digit
            0x00ff00ff00ff00ffull,              // runs
            (uint64_t(1) << 62) - 1,            // 62 bits: unpacked from 4 positions on
            0x3ff0000000000000ull | 0xfffffull, // two runs far apart
            ~uint64_t(0),                       // every bit
        };
        const uint32_t counts[] = { 1, 2, 3, 100, 5000, 70000 };
        uint32_t cases = 0;
        for (uint64_t mask : masks)
        {
            for (uint32_t count : counts)
            {
                for (bool parallel : { false, true })
                {
                    std::vector<uint64_t> keys(count);
                    // few distinct values in half of the runs, so that stability matters
                    const bool fewValues = (cases & 1) != 0;
                    for (uint64_t& key : keys)
                    {
                        const uint64_t random = fewValues ? rng.Next() % 7 * 0x0101010101010101ull : rng.Next();
                        key = (random & mask) | (0x1234000000000000ull & ~mask);
                    }
                    uint64_t keyOr = 0, keyAnd = ~uint64_t(0);
                    for (uint64_t key : keys)
                    {
                        keyOr |= key;
                        keyAnd &= key;
                    }
                    std::vector<uint32_t> order(count);
                    CHECK(sorter.Sort(keys.data(), count, keyOr ^ keyAnd, 7, order.data(), parallel ? jobs.Get()
                                                                                                    : nullptr));
                    std::vector<uint32_t> expected(count);
                    std::iota(expected.begin(), expected.end(), 0u);
                    std::stable_sort(expected.begin(), expected.end(),
                                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
                    for (uint32_t i = 0; i < count; ++i)
                        CHECK(order[i] == expected[i] + 7);
                    ++cases;
                }
            }
        }
        std::printf("  %u sorts equal to std::stable_sort\n", cases);
    }

    // Debug builds check every row of four of the query's chunks at every 64th extraction, visible or not: a mesh box
    // edited without InvalidateContent, and a material domain edited without it, are reported once each; Release
    // builds compile the check out.
    void test_stale_leaves()
    {
        TestViews views;
        PacketScene scene(0x88);
        scene.MakeAssets();
        scene.graph->SetRootNode(MakeNode("root"));
        auto node = MakeNode("stale");
        node->SetTranslation(dm::double3(1000.0, 0.0, 0.0));   // in no view
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, scene.meshes[0].Get()));
        scene.graph->Attach(scene.graph->GetRootNode(), node);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        const uint32_t before = g_Warnings.load();
#ifdef NDEBUG
        const uint32_t expected = 0;
#else
        const uint32_t expected = 1;
#endif
        auto run = [&](uint32_t first, uint32_t last)
        {
            for (uint32_t f = first; f <= last; ++f)
            {
                node->SetTranslation(dm::double3(1000.0, double(f & 1), 0.0));   // so that every Refresh runs
                scene.graph->Refresh(f);
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f));
                CHECK(snapshot.instances.count == 0);
                service->EndFrame(true);
            }
        };
        run(1, 63);
        CHECK(g_Warnings.load() == before);
        scene.meshes[0]->objectSpaceBounds = dm::box3(dm::float3(-2.f), dm::float3(2.f));
        run(64, 64);   // the 64th extraction checks the chunk: the box is stale
        CHECK(g_Warnings.load() == before + expected);
        node->InvalidateContent();
        run(65, 128);   // current again: quiet
        CHECK(g_Warnings.load() == before + expected);
        Material* material = scene.meshes[0]->geometries[0]->material;
        const MaterialDomain domain = material->domain;
        material->domain = MaterialDomain::AlphaBlended;   // the content flags go stale
        run(129, 192);
        CHECK(g_Warnings.load() == before + 2 * expected);
        material->domain = domain;
        run(193, 256);
        CHECK(g_Warnings.load() == before + 2 * expected);
    }

    // A leaf class with LeafTypeDesc::dynamicBounds: its box changes by itself.
    class DynamicMesh final : public MeshInstance
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        explicit DynamicMesh(MeshInfo* mesh) : MeshInstance(mesh) { }
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

    // A3, A4, A5 and the edge cases of the review of E5.2: a leaf with dynamic bounds on a moving node is culled with its
    // box of this frame without InvalidateContent; slots come from this graph's tables when the meshes and materials
    // are shared with another graph; a material domain edited without InvalidateContent moves a geometry to the other
    // pass; a transparent instance with a large subtree (its distance from the subgraph box, on 8 workers) against the
    // legacy strategy; NaN boxes (the chunk is never culled, the rows as the legacy test says); the view count changing
    // between frames.
    void test_review_cases()
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        JobSystemDesc jobDesc;
        jobDesc.debugName = "test_draw_packets.review";
        jobDesc.workerThreadCount = 8;
        CHECK(CreateJobSystem(jobDesc, &jobs) == nvrhi::FS_OK);
        TestViews views;
        Tally tally;

        // A3: dynamic bounds
        {
            PacketScene scene(0x91);
            scene.MakeAssets();
            scene.graph->SetRootNode(MakeNode("root"));
            auto node = MakeNode("dynamic");
            node->SetTranslation(dm::double3(0.0, 2.0, 200.0));
            auto leaf = MAKE_RC_OBJ_PTR(DynamicMesh, scene.meshes[0].Get());
            leaf->box = dm::box3(dm::float3(-1.f), dm::float3(1.f));
            node->SetLeaf(leaf);
            scene.graph->Attach(scene.graph->GetRootNode(), node);
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            scene.graph->Refresh(1);
            CHECK(ExtractFrame(service, frame, views.Desc(1)).views[0].instances.count == 1);
            service->EndFrame(true);
            // the box moves out of the view by itself (the leaf's box), the node moves a little: no InvalidateContent
            leaf->box = dm::box3(dm::float3(-1.f, -1.f, -400.f), dm::float3(1.f, 1.f, -398.f));
            node->SetTranslation(dm::double3(0.0, 2.5, 200.0));
            scene.graph->Refresh(2);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(2));
            CHECK(snapshot.views[0].instances.count == 0);   // z from -200 to -198: behind the camera at -20
            const dm::box3 expected = leaf->box * node->GetLocalToWorldTransformFloat();
            const dm::box3 rootBounds = scene.graph->GetRootNode()->GetGlobalBoundingBox();
            CHECK(std::memcmp(&rootBounds, &expected, sizeof(expected)) == 0);
            service->EndFrame(true);

            // the node stands still and nothing else changes: Refresh still runs the bounds (no skip while a leaf has
            // dynamic bounds), and the box growing back into the view is seen; the root's box follows each frame
            for (uint32_t step = 0; step < 3; ++step)
            {
                leaf->box = dm::box3(dm::float3(-1.f, -1.f, -1.f - 2.f * float(step)), dm::float3(1.f, 1.f, 1.f));
                scene.graph->Refresh(3 + step);
                const RenderSnapshot& still = ExtractFrame(service, frame, views.Desc(3 + step));
                CHECK(still.views[0].instances.count == 1);
                const dm::box3 grown = leaf->box * node->GetLocalToWorldTransformFloat();
                const dm::box3 root = scene.graph->GetRootNode()->GetGlobalBoundingBox();
                CHECK(std::memcmp(&root, &grown, sizeof(grown)) == 0);
                service->EndFrame(true);
            }
        }

        // A4: meshes and materials shared with another graph, which wrote its own slots into them last
        {
            PacketScene scene(0x92);
            scene.Build(60);
            auto other = MAKE_RC_OBJ_PTR(SceneGraph);
            other->SetRootNode(MakeNode("other"));
            for (size_t i = scene.meshes.size(); i-- > 0;)   // the other order: other slots
            {
                auto node = MakeNode("o");
                node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, scene.meshes[i].Get()));
                other->Attach(other->GetRootNode(), node);
            }
            other->Refresh(1);
            uint32_t differ = 0;
            for (const auto& material : scene.materials)
            {
                const MaterialId id = scene.graph->GetMaterials().Find(material);
                differ += id != MaterialId::Invalid && MaterialTable::SlotOf(id) != uint32_t(material->materialID);
            }
            CHECK(differ > 0);   // materialID holds the other graph's slots now
            scene.graph->Refresh(1);
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(1));
            for (uint32_t v = 0; v < snapshot.viewCount; ++v)
                CompareView(scene.graph, snapshot, v, true, tally);   // slots against this graph's tables
            service->EndFrame(true);
        }

        // A5: a material domain edited without InvalidateContent moves its geometries to the other pass
        {
            PacketScene scene(0x93);
            scene.MakeAssets();
            scene.graph->SetRootNode(MakeNode("root"));
            auto node = MakeNode("n");
            node->SetTranslation(dm::double3(0.0, 2.0, 5.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, scene.meshes[0].Get()));   // one geometry, opaque material 0
            scene.graph->Attach(scene.graph->GetRootNode(), node);
            scene.graph->Refresh(1);
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            const RenderSnapshot& first = ExtractFrame(service, frame, views.Desc(1));
            CHECK(first.views[0].packets[0].count == 1 && first.views[0].packets[1].count == 0);
            service->EndFrame(true);
            Material* material = scene.meshes[0]->geometries[0]->material;
            CHECK(material->domain == MaterialDomain::Opaque);
            material->domain = MaterialDomain::AlphaBlended;
            node->SetTranslation(dm::double3(0.0, 2.0, 6.0));
            scene.graph->Refresh(2);
            const RenderSnapshot& second = ExtractFrame(service, frame, views.Desc(2));
            CHECK(second.views[0].packets[0].count == 0 && second.views[0].packets[1].count == 1);
            service->EndFrame(true);
            material->domain = MaterialDomain::Opaque;
        }

        // a transparent instance with a large subtree, on 8 workers
        {
            PacketScene scene(0x94);
            scene.MakeAssets();
            scene.graph->SetRootNode(MakeNode("root"));
            scene.graph->SetJobSystem(jobs);
            auto glass = MakeNode("glass");
            glass->SetTranslation(dm::double3(0.0, 0.0, 10.0));
            glass->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, scene.meshes[2].Get()));   // blended geometries among others
            scene.graph->Attach(scene.graph->GetRootNode(), glass);
            for (int i = 0; i < 3000; ++i)
            {
                auto child = MakeNode("c");
                child->SetTranslation(dm::double3(scene.rng.Range(-30, 30), scene.rng.Range(-5, 5),
                                                  scene.rng.Range(-10, 60)));
                child->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, scene.meshes[scene.rng.Below(4)].Get()));
                scene.graph->Attach(i % 10 ? glass.Get() : scene.graph->GetRootNode(), child);
            }
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            for (uint32_t f = 1; f <= 3; ++f)
            {
                glass->SetTranslation(dm::double3(0.0, double(f), 10.0));
                scene.graph->Refresh(f);
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f));
                for (uint32_t v = 0; v < snapshot.viewCount; ++v)
                    CompareView(scene.graph, snapshot, v, true, tally);
                service->EndFrame(true);
            }
            scene.graph->SetJobSystem(nullptr);
        }

        // NaN boxes: a mesh whose box is NaN shares its chunk with ordinary instances
        {
            PacketScene scene(0x95);
            scene.MakeAssets();
            const float nan = std::numeric_limits<float>::quiet_NaN();
            auto nanMesh = MAKE_RC_OBJ_PTR(MeshInfo);
            nanMesh->name = "nan";
            auto nanGeometry = MAKE_RC_OBJ_PTR(MeshGeometry);
            nanGeometry->material = scene.materials[0];
            nanGeometry->numIndices = 3;
            nanGeometry->numVertices = 3;
            nanGeometry->objectSpaceBounds = dm::box3(dm::float3(nan), dm::float3(nan));
            nanMesh->geometries.push_back(nanGeometry);
            nanMesh->objectSpaceBounds = nanGeometry->objectSpaceBounds;
            scene.graph->SetRootNode(MakeNode("root"));
            for (int i = 0; i < 64; ++i)
            {
                auto node = MakeNode("n");
                node->SetTranslation(dm::double3(i % 2 ? 500.0 : 0.0, 2.0, 5.0 + i));   // half of them in no view
                node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, i == 7 ? nanMesh.Get() : scene.meshes[0].Get()));
                scene.graph->Attach(scene.graph->GetRootNode(), node);
            }
            scene.graph->Refresh(1);
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(1));
            for (uint32_t v = 0; v < snapshot.viewCount; ++v)
            {
                // the visible set as the legacy test says (a NaN box passes every plane)
                const dm::frustum frustum = views.list[v]->GetViewFrustum();
                std::vector<const MeshInstance*> expected;
                for (SceneGraphWalker walker(scene.graph->GetRootNode()); walker; walker.Next(true))
                {
                    const MeshInstance* instance = query_cast<MeshInstance>(walker->GetLeaf());
                    if (instance && frustum.intersectsWith(OwnBox(instance)))
                        expected.push_back(instance);
                }
                const ExtractedView& view = snapshot.views[v];
                std::vector<const MeshInstance*> actual(snapshot.instances.leaves + view.instances.first,
                                                        snapshot.instances.leaves + view.instances.first +
                                                            view.instances.count);
                std::sort(expected.begin(), expected.end());
                std::sort(actual.begin(), actual.end());
                CHECK(expected == actual);
                CHECK(!actual.empty());
            }
            service->EndFrame(true);
        }

        // the view count changes between frames: 5, 2, 5, 1, 0, 3
        {
            PacketScene scene(0x96);
            scene.Build(80);
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            const uint32_t counts[] = { 5, 2, 5, 1, 0, 3 };
            uint32_t f = 0;
            for (uint32_t count : counts)
            {
                scene.Edit(++f);
                scene.graph->Refresh(f);
                ExtractDesc desc = views.Desc(f);
                desc.viewCount = count;
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
                CHECK(snapshot.viewCount == count);
                for (uint32_t v = 0; v < snapshot.viewCount; ++v)
                    CompareView(scene.graph, snapshot, v, true, tally);
                service->EndFrame(true);
            }
        }
    }

    // A segment large enough for the parallel sort inside it (from 16,384 packets, nested in the segments' own parallel
    // loop), and the visibility ranges at 1, 2 and 8 workers: equal bytes.
    void test_parallel_sort_workers()
    {
        nvrhi::AutoPtr<IJobSystem> jobs[3];
        const uint32_t workerCounts[3] = { 1, 2, 8 };
        for (uint32_t k = 0; k < 3; ++k)
        {
            JobSystemDesc desc;
            desc.debugName = "test_draw_packets.sort_workers";
            desc.workerThreadCount = workerCounts[k];
            CHECK(CreateJobSystem(desc, &jobs[k]) == nvrhi::FS_OK);
        }
        TestViews views;
        PacketScene scene(0x97);
        scene.MakeAssets();
        scene.graph->SetRootNode(MakeNode("root"));
        Rng rng{ 5 };
        for (int i = 0; i < 24000; ++i)
        {
            // in the perspective view; meshes 0 and 3 have one opaque geometry each (material 0, material 1)
            auto node = MakeNode("n");
            node->SetTranslation(dm::double3(rng.Range(-12, 12), rng.Range(-5, 8), rng.Range(0, 40)));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, scene.meshes[rng.Below(2) ? 0 : 3].Get()));
            scene.graph->Attach(scene.graph->GetRootNode(), node);
        }
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        uint32_t largest = 0;
        for (uint32_t f = 1; f <= 3; ++f)
        {
            scene.graph->Refresh(f);
            uint64_t hashes[3] = {};
            for (uint32_t k = 0; k < 3; ++k)
            {
                scene.graph->SetJobSystem(jobs[k]);
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f));
                hashes[k] = HashPackets(snapshot);
                largest = std::max(largest, snapshot.views[0].packets[0].count);
                for (uint32_t i = 1; i < snapshot.views[0].packets[0].count; ++i)
                    CHECK(snapshot.packets.keys[i - 1] <= snapshot.packets.keys[i]);
                service->EndFrame(true);
            }
            CHECK(hashes[0] == hashes[1] && hashes[1] == hashes[2]);
        }
        scene.graph->SetJobSystem(nullptr);
        std::printf("  the perspective view's opaque segment: %u packets\n", largest);
        CHECK(largest >= visibility::RadixSorter::c_ParallelThreshold);
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
        { "radix_sort", test_radix_sort },
        { "legacy_equivalence", test_legacy_equivalence },
        { "combined_double_sided", test_combined_double_sided },
        { "steady_state", test_steady_state },
        { "stale_leaves", test_stale_leaves },
        { "review_cases", test_review_cases },
        { "parallel_sort_workers", test_parallel_sort_workers },
    };
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
                continue;
            std::printf("%s\n", c.name);
            const uint32_t warnings = g_Warnings.load();
            c.run();
            if (c.run != test_stale_leaves)
                CHECK(g_Warnings.load() == warnings);
            std::printf("%s: passed\n", c.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    std::printf("draw packets: ok\n");
    return 0;
}
