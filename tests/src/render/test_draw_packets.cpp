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
//                        - every frame extracted on the calling thread and again at 1, 2 and 8 workers, with the
//                          ranges, the copy and sort, or both in parallel whatever the frame's size, gives the same
//                          bytes (the calls into the job system tell that each combination ran); the per-frame hashes
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
//   small_extraction     a scene below the limits of the serial extraction hands nothing to the job system (so no worker
//                        wakes) and allocates nothing; the same frames in parallel give the same bytes
//   parallel_sort_workers  a segment sorted in parallel inside the segments' parallel loop: equal at 1, 2, 8 workers
//   runtime_rows         runtime-only mesh instances (plan step E7, item E7.2): two graphs of the same instances
//                        (meshes, transforms, creation order), one of nodes, one where all instances, or about half of
//                        them, are runtime-only rows; 12 frames of moves and respawns, five views, extracted serially
//                        and in parallel at 1 and 8 workers (equal bytes per graph): each view's visible instances and
//                        each pass's packets equal the node graph's (the same slots and first geometry slots, one
//                        instance per slot in each graph; keys, fields, order, mesh, bounds); runtime rows have no leaf
//   runtime_steady_state 10 frames of moving runtime rows beside nodes after warm-up, serially, then on 8 workers with
//                        every extraction step parallel, then on 8 workers with 100 runtime rows destroyed and made
//                        again each frame: no operator new, no ECS heap block in any of the three
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
#include <donut/tests/e7_switches.h>
#include <donut/tests/utils.h>

#include "CountingJobSystem.h"
#include "EcsWorldImpl.h"       // ecs::GetHeapAllocationCount
#include "RenderVisibility.h"   // the radix sort's own cases, the limits of the serial extraction
#include "RuntimeTwins.h"

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
#include <set>
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
using donut::tests::CountingJobSystem;
using donut::tests::ExtractionMode;

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
        // Run 0 is serial, whatever the frame's size; the others combine the steps at 1, 2 and 8 workers.
        constexpr uint32_t c_Runs = 6;
        const uint32_t workerCounts[3] = { 1, 2, 8 };
        nvrhi::AutoPtr<CountingJobSystem> jobs[3];
        for (uint32_t k = 0; k < 3; ++k)
        {
            JobSystemDesc desc;
            desc.debugName = "test_draw_packets";
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
                uint64_t hashes[c_Runs] = {};
                for (uint32_t k = 0; k < c_Runs; ++k)
                {
                    CountingJobSystem* runJobs = jobs[runs[k].jobs];
                    scene.graph->SetJobSystem(runJobs);
                    runs[k].mode.Set();
                    const uint32_t calls = runJobs->calls.load();
                    const RenderSnapshot& snapshot = ExtractFrame(service, frame, desc);
                    CHECK(runJobs->calls.load() - calls == runs[k].mode.Calls(snapshot.packets.count));
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
                for (uint32_t k = 1; k < c_Runs; ++k)
                    CHECK(hashes[k] == hashes[0]);
            }
            ExtractionMode::Reset();
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

    // A frame of a small scene hands nothing to the job system (E5.8: one sort of 352 packets per frame woke ten
    // workers); in parallel the same frame does, and gives the same bytes.
    void test_small_extraction()
    {
        JobSystemDesc jobDesc;
        jobDesc.debugName = "test_draw_packets.small";
        jobDesc.workerThreadCount = 4;
        nvrhi::AutoPtr<IJobSystem> inner;
        CHECK(CreateJobSystem(jobDesc, &inner) == nvrhi::FS_OK);
        auto jobs = MAKE_RC_OBJ_PTR(CountingJobSystem, inner.Get());

        TestViews views;
        PacketScene scene(0x58);
        scene.Build(25);
        scene.graph->SetJobSystem(jobs);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        uint32_t packets = 0, parallelCalls = 0;
        for (uint32_t f = 1; f <= 10; ++f)
        {
            scene.Edit(f);
            scene.graph->Refresh(f);
            uint32_t calls = jobs->calls.load();
            const RenderSnapshot& serial = ExtractFrame(service, frame, views.Desc(f));
            CHECK(jobs->calls.load() == calls);
            CHECK(serial.packets.count > 100);
            packets = std::max(packets, serial.packets.count);
            const uint64_t hash = HashPackets(serial);
            service->EndFrame(false);

            const ExtractionMode mode{ true, true };
            mode.Set();
            calls = jobs->calls.load();
            const RenderSnapshot& parallel = ExtractFrame(service, frame, views.Desc(f));
            parallelCalls += jobs->calls.load() - calls;
            CHECK(jobs->calls.load() - calls == mode.Calls(parallel.packets.count));
            CHECK(HashPackets(parallel) == hash);
            service->EndFrame(true);
            ExtractionMode::Reset();
        }
        // the serial path allocates nothing in steady state either
        uint32_t f = 11;
        auto run = [&]()
        {
            scene.graph->Refresh(f);
            (void)ExtractFrame(service, frame, views.Desc(f));
            service->EndFrame(true);
            ++f;
        };
        for (int i = 0; i < 6; ++i)
            run();
        const uint64_t newBefore = g_NewCount.load();
        const uint64_t heapBefore = ecs::GetHeapAllocationCount();
        for (int i = 0; i < 10; ++i)
            run();
        CHECK(g_NewCount.load() == newBefore && ecs::GetHeapAllocationCount() == heapBefore);
        scene.graph->SetJobSystem(nullptr);
        std::printf("  up to %u packets in 5 views: no job, no allocation; in parallel %u jobs in 10 frames\n", packets,
                    parallelCalls);
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

    // ------------------------------------------------------------------------------------------ runtime-only rows

    bool SameBits(const dm::box3& x, const dm::box3& y)
    {
        return std::memcmp(&x, &y, sizeof(x)) == 0;
    }

    struct TwinTally
    {
        uint64_t instances = 0;
        uint64_t runtimeInstances = 0;
        uint64_t packets[c_DrawPacketPassCount] = {};
        uint64_t runtimePackets[c_DrawPacketPassCount] = {};
        uint64_t subgraphPackets = 0;   // transparent packets of node instances with a child
        uint64_t remapped = 0;          // visible instances whose slot differs between the graphs
    };

    // One packet of a view and pass: its instance by id, its key built with a given instance slot.
    struct TwinPacket
    {
        uint64_t key;
        uint32_t id;
        uint32_t geometryIndex;
        uint32_t geometrySlot;
        uint32_t materialSlot;
        uint8_t cullMode;
        uint8_t flags;

        bool operator==(const TwinPacket& o) const
        {
            return key == o.key && id == o.id && geometryIndex == o.geometryIndex && geometrySlot == o.geometrySlot &&
                   materialSlot == o.materialSlot && cullMode == o.cullMode && flags == o.flags;
        }
    };

    // The key of packet i of `s` with `slot` for its instance's slot.
    uint64_t KeyWithSlot(const RenderSnapshot& s, uint32_t i, uint32_t slot)
    {
        const DrawPackets& p = s.packets;
        const MeshInfo* mesh = s.instances.meshInfos[p.visibleInstances[i]];
        const Material* material = mesh->geometries[p.geometryIndices[i]]->material;
        if (p.flags[i] & c_DrawPacketFlagTransparentPass)
            return MakeTransparentDrawKey(GetTransparentDrawKeyDistance(p.keys[i]), slot, p.cullModes[i]);
        return MakeOpaqueDrawKey(uint32_t(material->domain), p.cullModes[i], p.materialSlots[i], p.geometrySlots[i],
                                 slot);
    }

    // Graph 1's snapshot `b` against graph 0's `a` (RuntimeTwins): each view's visible instances, matched by id, have
    // the same mesh, bounds, geometry count and slots; each pass's packets, in order, equal graph 0's (keys rebuilt with
    // graph 1's slots, sorted again, stably). Graph 1's runtime rows have no leaf. The twins' slot tables are alike by
    // construction, so an instance's slot and first geometry slot are equal in both graphs (backlog E72-R1: the check
    // was against graph 1 itself), and within each graph distinct instances hold distinct slots (a bijection).
    void CompareTwins(const tests::RuntimeTwins& twins, const RenderSnapshot& a, const RenderSnapshot& b,
                      TwinTally& tally)
    {
        CHECK(a.viewCount == b.viewCount);
        const VisibleInstances& ia = a.instances;
        const VisibleInstances& ib = b.instances;
        std::vector<uint32_t> idsA(ia.count), idsB(ib.count);
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> slotsOf[2];   // id -> (instance slot, first geometry slot)
        for (uint32_t g = 0; g < 2; ++g)
        {
            const VisibleInstances& in = g ? ib : ia;
            std::vector<uint32_t>& ids = g ? idsB : idsA;
            for (uint32_t i = 0; i < in.count; ++i)
            {
                ids[i] = twins.Id(g, in, i);
                CHECK(ids[i] != tests::RuntimeTwins::c_NoId);
                const auto slots = std::make_pair(in.instanceSlots[i], in.firstGeometryInstanceSlots[i]);
                CHECK(slotsOf[g].emplace(ids[i], slots).first->second == slots);   // one slot per instance
            }
        }
        std::map<uint32_t, uint32_t> slotMap;   // graph 0's instance slot -> graph 1's (equal: E72-R1)
        for (const auto& [id, slots] : slotsOf[0])
        {
            const auto it = slotsOf[1].find(id);
            CHECK(it != slotsOf[1].end());
            CHECK(it->second == slots);   // the instance slot and the first geometry slot, both graphs alike
            CHECK(slotMap.emplace(slots.first, it->second.first).first->second == it->second.first);
        }
        CHECK(slotsOf[0].size() == slotsOf[1].size());
        for (uint32_t g = 0; g < 2; ++g)
        {
            // one instance per slot, and per first geometry slot: no two visible instances share one
            std::set<uint32_t> instanceSlots, geometrySlots;
            for (const auto& [id, slots] : slotsOf[g])
            {
                CHECK(instanceSlots.insert(slots.first).second);
                CHECK(geometrySlots.insert(slots.second).second);
            }
        }

        for (uint32_t v = 0; v < a.viewCount; ++v)
        {
            // the visible instances, by id
            const SnapshotRange ra = a.views[v].instances;
            const SnapshotRange rb = b.views[v].instances;
            CHECK(ra.count == rb.count);
            std::vector<std::pair<uint32_t, uint32_t>> byIdA, byIdB;
            for (uint32_t i = ra.first; i < ra.first + ra.count; ++i)
                byIdA.emplace_back(idsA[i], i);
            for (uint32_t i = rb.first; i < rb.first + rb.count; ++i)
                byIdB.emplace_back(idsB[i], i);
            std::sort(byIdA.begin(), byIdA.end());
            std::sort(byIdB.begin(), byIdB.end());
            for (size_t k = 0; k < byIdA.size(); ++k)
            {
                const auto [id, i] = byIdA[k];
                const uint32_t j = byIdB[k].second;
                CHECK(byIdB[k].first == id);
                const tests::RuntimeTwins::Instance& instance = twins.instances[id / 2];
                const bool runtime = id % 2 == 0 && instance.runtime;
                CHECK((ib.leaves[j] == nullptr) == runtime && ia.leaves[i] != nullptr);
                CHECK(ib.instanceSlots[j] == ia.instanceSlots[i]);
                CHECK(ib.firstGeometryInstanceSlots[j] == ia.firstGeometryInstanceSlots[i]);
                CHECK(ib.geometryCounts[j] == ia.geometryCounts[i]);
                CHECK(ib.meshes[j] == ia.meshes[i] && ib.meshInfos[j] == ia.meshInfos[i]);
                CHECK(SameBits(ib.worldBounds[j], ia.worldBounds[i]));
                tally.instances += 1;
                tally.runtimeInstances += runtime ? 1 : 0;
                tally.remapped += ib.instanceSlots[j] != ia.instanceSlots[i] ? 1 : 0;
            }

            // the packets: graph 0's rebuilt with graph 1's slots, in key order, against graph 1's
            for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
            {
                const SnapshotRange pa = a.views[v].packets[p];
                const SnapshotRange pb = b.views[v].packets[p];
                std::vector<TwinPacket> expected, actual;
                for (uint32_t i = pa.first; i < pa.first + pa.count; ++i)
                {
                    const uint32_t vi = a.packets.visibleInstances[i];
                    CHECK(KeyWithSlot(a, i, ia.instanceSlots[vi]) == a.packets.keys[i]);
                    expected.push_back(TwinPacket{ KeyWithSlot(a, i, slotMap.at(ia.instanceSlots[vi])), idsA[vi],
                                                   a.packets.geometryIndices[i], a.packets.geometrySlots[i],
                                                   a.packets.materialSlots[i], a.packets.cullModes[i],
                                                   a.packets.flags[i] });
                }
                std::stable_sort(expected.begin(), expected.end(),
                                 [](const TwinPacket& x, const TwinPacket& y) { return x.key < y.key; });
                for (uint32_t i = pb.first; i < pb.first + pb.count; ++i)
                {
                    const uint32_t vi = b.packets.visibleInstances[i];
                    actual.push_back(TwinPacket{ b.packets.keys[i], idsB[vi], b.packets.geometryIndices[i],
                                                 b.packets.geometrySlots[i], b.packets.materialSlots[i],
                                                 b.packets.cullModes[i], b.packets.flags[i] });
                    const bool runtime = idsB[vi] % 2 == 0 && twins.instances[idsB[vi] / 2].runtime;
                    tally.runtimePackets[p] += runtime ? 1 : 0;
                    const bool parent = idsB[vi] % 2 == 0 && !runtime &&
                                        twins.instances[idsB[vi] / 2].childMesh != tests::RuntimeTwins::c_NoChild;
                    tally.subgraphPackets += parent && p == uint32_t(DrawPacketPass::Transparent) ? 1 : 0;
                }
                CHECK(expected == actual);
                tally.packets[p] += actual.size();
            }
        }
    }

    // Runtime-only rows against nodes of the same instances: all of them, then about half beside nodes (some with a
    // child); 12 frames of moves (none in every fifth) and two respawns of 40 instances; every frame extracted
    // serially, in parallel at 1 worker and at 8 workers, with equal bytes per graph.
    void test_runtime_rows()
    {
        nvrhi::AutoPtr<CountingJobSystem> jobs[2];
        const uint32_t workerCounts[2] = { 1, 8 };
        for (uint32_t k = 0; k < 2; ++k)
        {
            JobSystemDesc desc;
            desc.debugName = "test_draw_packets.runtime";
            desc.workerThreadCount = workerCounts[k];
            nvrhi::AutoPtr<IJobSystem> inner;
            CHECK(CreateJobSystem(desc, &inner) == nvrhi::FS_OK);
            jobs[k] = MAKE_RC_OBJ_PTR(CountingJobSystem, inner.Get());
        }
        constexpr uint32_t c_Runs = 3;
        const struct
        {
            uint32_t jobs;
            ExtractionMode mode;
        } runs[c_Runs] = { { 0, { false, false } }, { 0, { true, true } }, { 1, { true, true } } };

        TestViews views;
        PacketScene assets(0x7e2);
        assets.MakeAssets();
        std::vector<MeshInfo*> meshes;
        for (const auto& mesh : assets.meshes)
            meshes.push_back(mesh);
        for (double share : { 1.0, 0.5 })
        {
            tests::RuntimeTwins twins(share == 1.0 ? 31 : 37);
            twins.Build(meshes, 600, share, 40.0, 6);
            RenderFrame frames[2];
            nvrhi::AutoPtr<RenderService> services[2] = { MakeService(twins.graphs[0]), MakeService(twins.graphs[1]) };
            TwinTally tally;
            for (uint32_t f = 1; f <= 12; ++f)
            {
                if (f > 1 && f % 5 != 0)
                    twins.Move();
                if (f == 4 || f == 8)
                    twins.Respawn(40);
                twins.Refresh(f);
                const ExtractDesc desc = views.Desc(f);
                uint64_t hashes[2][c_Runs] = {};
                for (uint32_t k = 0; k < c_Runs; ++k)
                {
                    CountingJobSystem* runJobs = jobs[runs[k].jobs];
                    runs[k].mode.Set();
                    const RenderSnapshot* snapshots[2] = {};
                    for (uint32_t g = 0; g < 2; ++g)
                    {
                        twins.graphs[g]->SetJobSystem(runJobs);
                        const uint32_t calls = runJobs->calls.load();
                        snapshots[g] = &ExtractFrame(services[g], frames[g], desc);
                        const uint32_t made = runJobs->calls.load() - calls;
                        CHECK(runs[k].mode.parallelRanges ? made > 0
                                                          : made == (snapshots[g]->packets.count > 8192 ? 1u : 0u));
                        hashes[g][k] = HashPackets(*snapshots[g]);
                    }
                    if (k == 0)
                        CompareTwins(twins, *snapshots[0], *snapshots[1], tally);
                    services[0]->EndFrame(true);
                    services[1]->EndFrame(true);
                }
                for (uint32_t g = 0; g < 2; ++g)
                    for (uint32_t k = 1; k < c_Runs; ++k)
                        CHECK(hashes[g][k] == hashes[g][0]);
            }
            ExtractionMode::Reset();
            twins.graphs[0]->SetJobSystem(nullptr);
            twins.graphs[1]->SetJobSystem(nullptr);
            std::printf("  %u of 600 runtime rows: %llu visible instances (%llu runtime rows, %llu with another slot), "
                        "%llu opaque and %llu transparent packets (%llu and %llu of runtime rows, %llu transparent of "
                        "nodes with a child)\n", twins.RuntimeCount(), (unsigned long long)tally.instances,
                        (unsigned long long)tally.runtimeInstances, (unsigned long long)tally.remapped,
                        (unsigned long long)tally.packets[0], (unsigned long long)tally.packets[1],
                        (unsigned long long)tally.runtimePackets[0], (unsigned long long)tally.runtimePackets[1],
                        (unsigned long long)tally.subgraphPackets);
            CHECK(tally.runtimeInstances > 0 && tally.runtimePackets[0] > 100 && tally.runtimePackets[1] > 100);
            CHECK(share == 1.0 ? tally.runtimeInstances == tally.instances
                               : tally.runtimeInstances < tally.instances && tally.subgraphPackets > 0);
        }
    }

    // Compact packet records (plan step E7, item E7.4, remedy 2; part I5): RenderService::SetCompactPacketRecords is off
    // by default; with it on, each extraction's snapshot (the views' ranges, the instances, the packets with their keys
    // and order) equals the one with it off, over a scene edited for eight frames, serially and with every step on the
    // job system. The sorter's two-step form gives Sort's order, with the position in the word and in the index arrays.
    void test_compact_packet_records()
    {
        // Stage T1: on in the switches-on runs (tests/src/e7_switches.cpp); the value found is put back at the end.
        const bool compactFound = RenderService::GetCompactPacketRecords();
        CHECK(compactFound == donut::tests::E7SwitchRequested("compact"));
        JobSystemDesc jobDesc;
        jobDesc.debugName = "test_draw_packets.compact";
        jobDesc.workerThreadCount = 8;
        nvrhi::AutoPtr<IJobSystem> inner;
        CHECK(CreateJobSystem(jobDesc, &inner) == nvrhi::FS_OK);
        auto jobs = MAKE_RC_OBJ_PTR(CountingJobSystem, inner.Get());

        // The sorter: Prepare, the words written by the caller, Finish; against Sort.
        visibility::RadixSorter sorter;
        visibility::RadixSorter reference;
        Rng rng{ 0x5eed };
        uint32_t cases = 0;
        uint32_t unpacked = 0;
        for (uint64_t mask : { uint64_t(0), uint64_t(0xff), (uint64_t(1) << 62) - 1, ~uint64_t(0) })
        {
            for (uint32_t count : { 1u, 2u, 100u, 20000u })
            {
                for (bool parallel : { false, true })
                {
                    std::vector<uint64_t> keys(count);
                    uint64_t keyOr = 0, keyAnd = ~uint64_t(0);
                    for (uint64_t& key : keys)
                    {
                        key = (rng.Next() % 13 * 0x0123456789abcdefull & mask) | (0x4000000000000000ull & ~mask);
                        keyOr |= key;
                        keyAnd &= key;
                    }
                    IJobSystem* sortJobs = parallel ? inner.Get() : nullptr;
                    std::vector<uint32_t> expected(count), order(count);
                    CHECK(reference.Sort(keys.data(), count, keyOr ^ keyAnd, 3, expected.data(), sortJobs));
                    CHECK(sorter.Prepare(count, keyOr ^ keyAnd));
                    if (uint64_t* words = sorter.Words())
                    {
                        uint32_t* indices = sorter.Indices();
                        unpacked += indices ? 1 : 0;
                        for (uint32_t i = 0; i < count; ++i)
                        {
                            words[i] = sorter.Word(keys[i], i);
                            if (indices)
                                indices[i] = i;
                        }
                    }
                    CHECK(sorter.Finish(3, order.data(), sortJobs));
                    CHECK(order == expected);
                    ++cases;
                }
            }
        }
        CHECK(unpacked > 0);
        std::printf("  %u two-step sorts equal to Sort (%u unpacked)\n", cases, unpacked);

        // The extraction, off against on, serially and with every step on the job system.
        const ExtractionMode modes[2] = { { false, false }, { true, true } };
        TestViews views;
        PacketScene scene(0xc0ffee);
        scene.Build(150);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        scene.graph->SetJobSystem(jobs.Get());
        uint64_t packets[c_DrawPacketPassCount] = {};
        uint64_t ties = 0;
        for (uint32_t f = 1; f <= 8; ++f)
        {
            scene.Edit(f);
            scene.graph->Refresh(f);
            const ExtractDesc desc = views.Desc(f, f % 2 == 0);
            for (const ExtractionMode& mode : modes)
            {
                mode.Set();
                uint64_t hashes[2] = {};
                for (uint32_t compact = 0; compact < 2; ++compact)
                {
                    RenderService::SetCompactPacketRecords(compact != 0);
                    const RenderSnapshot& s = ExtractFrame(service, frame, desc);
                    Hasher pointers;
                    pointers.Array(s.instances.leaves, s.instances.count);
                    pointers.Array(s.instances.meshInfos, s.instances.count);
                    hashes[compact] = HashPackets(s) ^ pointers.value;
                    if (compact)
                    {
                        for (uint32_t v = 0; v < s.viewCount; ++v)
                        {
                            for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
                            {
                                const SnapshotRange range = s.views[v].packets[p];
                                packets[p] += range.count;
                                // stage T5: equal keys next to each other (a transparent instance's geometries at
                                // one distance), whose order the hash pins: the stable order of the positions
                                for (uint32_t i = range.first + 1; i < range.first + range.count; ++i)
                                    ties += s.packets.keys[i - 1] == s.packets.keys[i] ? 1 : 0;
                            }
                        }
                    }
                    service->EndFrame(true);
                }
                CHECK(hashes[0] == hashes[1]);
            }
        }
        RenderService::SetCompactPacketRecords(false);
        ExtractionMode::Reset();
        scene.graph->SetJobSystem(nullptr);
        std::printf("  16 extractions equal with compact records: %llu opaque and %llu transparent packets, %llu "
                    "equal keys in stable order\n", (unsigned long long)packets[0], (unsigned long long)packets[1],
                    (unsigned long long)ties);
        CHECK(packets[0] > 1000 && packets[1] > 1000 && ties > 0);
        CHECK(!RenderService::GetCompactPacketRecords());
        RenderService::SetCompactPacketRecords(compactFound);
    }

    // Ten frames of runtime rows moving beside nodes after warm-up, five views: no operator new, no ECS heap block. A
    // tenth of the rows swing between two places every frame through the world, so that the views' sets change. Three
    // runs of warm-up and ten frames each (backlog E72-R7): serial; on 8 workers with every extraction step on the job
    // system (Refresh too); on 8 workers with 100 other runtime rows destroyed and made again every frame, so that the
    // observer frees and allocates their slots and references and the rows return to the chunks' holes.
    void test_runtime_steady_state()
    {
        TestViews views;
        PacketScene assets(0x5002);
        assets.MakeAssets();
        std::vector<MeshInfo*> meshes;
        for (const auto& mesh : assets.meshes)
            meshes.push_back(mesh);
        tests::RuntimeTwins twins(41);
        twins.Build(meshes, 1500, 0.8, 40.0, 0);
        SceneGraph* graph = twins.graphs[1];
        IWorld* world = graph->GetWorld();
        RenderFrame frame;
        auto service = MakeService(graph);
        std::vector<std::pair<Entity, dm::double3>> movers;
        for (size_t i = 0; i < twins.instances.size(); i += 10)
            if (twins.instances[i].runtime)
                movers.emplace_back(twins.instances[i].row, twins.instances[i].local.translation);
        // The respawned rows: runtime instances not among the movers, made again with the same mesh at the other of
        // two places each frame (one CreateEntities per mesh, values in arrays made before the measured frames).
        constexpr uint32_t c_Respawned = 100;
        const ComponentId localId = FindComponent<LocalTransform>(world);
        const ComponentId meshRefId = FindComponent<MeshRef>(world);
        std::vector<std::vector<Entity>> respawnedRows(twins.meshes.size());
        std::vector<std::vector<LocalTransform>> respawnedLocals(twins.meshes.size());
        std::vector<std::vector<MeshRef>> respawnedRefs(twins.meshes.size());
        {
            uint32_t picked = 0;
            for (size_t i = 5; i < twins.instances.size() && picked < c_Respawned; i += 10)
            {
                const tests::RuntimeTwins::Instance& instance = twins.instances[i];
                if (!instance.runtime)
                    continue;
                respawnedRows[instance.mesh].push_back(instance.row);
                respawnedLocals[instance.mesh].push_back(instance.local);
                respawnedRefs[instance.mesh].push_back(MeshRef{ twins.meshIds[1][instance.mesh] });
                ++picked;
            }
            CHECK(picked == c_Respawned);
        }
        std::vector<const void*> respawnValues(twins.components.size(), nullptr);
        uint32_t f = 0;
        auto respawn = [&]()
        {
            const double shift = (f & 1) ? 7.0 : -7.0;
            for (size_t m = 0; m < respawnedRows.size(); ++m)
            {
                std::vector<Entity>& rows = respawnedRows[m];
                if (rows.empty())
                    continue;
                CHECK(world->DestroyEntities(rows.data(), uint32_t(rows.size())) == nvrhi::FS_OK);
                for (LocalTransform& local : respawnedLocals[m])
                    local.translation.x += shift;
                for (size_t c = 0; c < twins.components.size(); ++c)
                {
                    respawnValues[c] = nullptr;
                    if (twins.components[c] == localId)
                        respawnValues[c] = respawnedLocals[m].data();
                    else if (twins.components[c] == meshRefId)
                        respawnValues[c] = respawnedRefs[m].data();
                }
                EntityCreateDesc desc{};
                desc.components = twins.components.data();
                desc.values = respawnValues.data();
                desc.componentCount = uint32_t(twins.components.size());
                desc.entityCount = uint32_t(rows.size());
                CHECK(world->CreateEntities(desc, rows.data()) == nvrhi::FS_OK);
            }
        };

        JobSystemDesc jobDesc;
        jobDesc.debugName = "test_draw_packets.runtime_steady_state";
        jobDesc.workerThreadCount = 8;
        nvrhi::AutoPtr<IJobSystem> jobs;
        CHECK(CreateJobSystem(jobDesc, &jobs) == nvrhi::FS_OK);

        uint32_t visible[2] = {};
        bool respawning = false;
        auto run = [&]()
        {
            const dm::double3 swing = (f & 1) ? dm::double3(12.0, 0.0, 9.0) : dm::double3(0.0);
            world->AdvanceVersion();
            for (const auto& [row, home] : movers)
                GetComponentForWrite<LocalTransform>(world, row)->translation = home + swing;
            if (respawning)
                respawn();
            graph->Refresh(++f);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f));
            CHECK(snapshot.packets.count > 0);
            visible[f & 1] = snapshot.instances.count;
            service->EndFrame(true);
        };
        const char* const runNames[3] = { "serial", "8 workers, parallel steps", "8 workers, 100 rows respawned" };
        for (uint32_t r = 0; r < 3; ++r)
        {
            if (r == 1)
            {
                graph->SetJobSystem(jobs);
                ExtractionMode{ true, true }.Set();
            }
            respawning = r == 2;
            for (int i = 0; i < 6; ++i)
                run();
            const uint64_t newBefore = g_NewCount.load();
            const uint64_t heapBefore = ecs::GetHeapAllocationCount();
            for (int i = 0; i < 10; ++i)
                run();
            const uint64_t newCalls = g_NewCount.load() - newBefore;
            const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
            std::printf("  %s: %u runtime rows (%zu moving) and %u nodes: %u and %u visible instances; %llu operator "
                        "new, %llu storage blocks over 10 frames\n", runNames[r], twins.RuntimeCount(), movers.size(),
                        uint32_t(twins.instances.size()) - twins.RuntimeCount(), visible[0], visible[1],
                        (unsigned long long)newCalls, (unsigned long long)heapBlocks);
            CHECK(visible[0] != visible[1]);
            CHECK(newCalls == 0 && heapBlocks == 0);
        }
        ExtractionMode::Reset();
        graph->SetJobSystem(nullptr);
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
        { "small_extraction", test_small_extraction },
        { "parallel_sort_workers", test_parallel_sort_workers },
        { "runtime_rows", test_runtime_rows },
        { "runtime_steady_state", test_runtime_steady_state },
        { "compact_packet_records", test_compact_packet_records },
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
