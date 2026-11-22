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

// The draw-strategy adapters over draw packets without a device (aggregate docs/plans/2026-11-21-donut-ecs-e5.md, item
// E5.3; docs/ecs-design.md 7.3: "strategies become adapters that produce DrawItems from packets"):
//
//   stream_equivalence   the scenes of test_draw_packets (3 seeds of 30 frames of its edit script, one perspective view
//                        and four orthographic cascade views), each view drawn through one bound strategy of each kind
//                        (InstancedOpaqueDrawStrategy and TransparentDrawStrategy after SetPackets, PacketDrawStrategy for
//                        each pass), prepared for the views in turn as IGeometryPass::AddToGraph does:
//                        - each stream is the view's packets in key order, item by item (instance, mesh, geometry,
//                          material, buffers, cull mode, distance);
//                        - as a multiset it equals the stream of the same strategy object returned to the graph walk
//                          (SetPackets(nullptr)), except items the walk kept only through a node's subgraph box (E5.2's
//                          known difference 1), each of which fails the frustum on its own box;
//                        - the transparent order equals the walk's wherever the walk's order is defined (runs of equal
//                          distance compared as multisets);
//                        - a model of RenderView's batching (a draw per change of buffers, material or cull mode;
//                          consecutive instance indices of one geometry merged) gives exactly the draws of the walk's
//                          items put into key order (the same grouping where the order is the same); opaque: at most as
//                          many draws as the walk's stream; transparent: the walk's draws (as a multiset: the walk orders
//                          one instance's geometries arbitrarily) wherever its order is defined and has no tie between
//                          instances (where it is undefined or tied, std::sort may by chance save the walk a draw:
//                          counted);
//   view_lookup          a view the snapshot does not hold: no item, one warning per view and Bind; the child views of a
//                        composite view through one strategy; an unbound PacketDrawStrategy draws nothing; a strategy
//                        bound, unbound and bound to the next frame's snapshot
//   steady_state         10 frames after warm-up, five views, every strategy bound and drained: no operator new
//   stale_snapshot       Debug: a strategy still bound to the snapshot of a frame the graph has reset gives no item and
//                        one warning (the arena is filled with a pattern at Reset)
//   use_check            Debug: a strategy prepared on two threads and drained on the first gives one warning; prepared
//                        and drained on one thread, then on another, none
//
//     test_draw_strategy_adapters [case]

#include <donut/render/DrawStrategy.h>
#include <donut/render/GeometryPasses.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include <donut/core/query_cast.h>
#include <donut/tests/utils.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <string>
#include <tuple>
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

    // test_draw_packets' scene and edit script (tests/src/render/test_draw_packets.cpp, PacketScene), the same random
    // sequence, so that these are E5.2's scenes. Added: the meshes' buffer groups and index and vertex offsets, which
    // RenderView's batching reads (two groups of meshes share a buffer group, as the meshes of one glTF file do; every
    // geometry has its own first index), and which extraction does not read.
    struct PacketScene
    {
        explicit PacketScene(uint64_t seed) : rng{ seed ^ 0x9E3779B97F4A7C15ull } { }

        Rng rng;
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        std::vector<nvrhi::AutoPtr<Material>> materials;
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
        nvrhi::AutoPtr<BufferGroup> bufferGroups[2] = { MAKE_RC_OBJ_PTR(BufferGroup), MAKE_RC_OBJ_PTR(BufferGroup) };
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
                mesh->buffers = bufferGroups[i < 5 ? 0 : 1];
                mesh->indexOffset = 100u * uint32_t(i);
                mesh->vertexOffset = 50u * uint32_t(i);
                const int geometries = 1 + i % 3;
                dm::box3 bounds = dm::box3::empty();
                for (int g = 0; g < geometries; ++g)
                {
                    auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
                    geometry->material = materials[(i * 3 + g) % materials.size()];
                    geometry->numIndices = 3 * (g + 1);
                    geometry->numVertices = 3;
                    geometry->indexOffsetInMesh = 20u * uint32_t(g);
                    geometry->vertexOffsetInMesh = 10u * uint32_t(g);
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

    // One perspective view and four orthographic cascade-like views of growing extent (test_draw_packets' views).
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

        ExtractDesc Desc(uint32_t frameIndex) const
        {
            ExtractDesc desc;
            desc.views = list;
            desc.viewCount = c_Count;
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

    const RenderSnapshot& ExtractFrame(RenderService* service, RenderFrame& frame, const ExtractDesc& desc)
    {
        frame.Begin();
        SceneFrame* sceneFrame = nullptr;
        CHECK(service->Extract(frame.graph, &frame.data, desc, &sceneFrame) == nvrhi::FS_OK);
        CHECK(sceneFrame && sceneFrame->snapshot);
        return *sceneFrame->snapshot;
    }

    // ------------------------------------------------------------------------------------------ streams

    dm::box3 OwnBox(const MeshInstance* instance)
    {
        const dm::box3 local = instance->GetMesh()->objectSpaceBounds;
        return local.isempty() ? dm::box3::empty() : local * instance->GetNode()->GetLocalToWorldTransformFloat();
    }

    uint32_t DistanceBits(float distance)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &distance, sizeof(bits));
        return bits;
    }

    // One item, comparable: (instance, mesh, geometry, material, buffers, cull mode, distance bits).
    using Item = std::tuple<const MeshInstance*, const MeshInfo*, const MeshGeometry*, const Material*,
                            const BufferGroup*, uint32_t, uint32_t>;

    Item ItemOf(const DrawItem& item)
    {
        return Item{ item.instance, item.mesh, item.geometry, item.material, item.buffers, uint32_t(item.cullMode),
                     DistanceBits(item.distanceToCamera) };
    }

    std::vector<DrawItem> Drain(IDrawStrategy* strategy, SceneGraphNode* root, const IView& view)
    {
        std::vector<DrawItem> items;
        strategy->PrepareForView(root, view);
        while (const DrawItem* item = strategy->GetNextItem())
            items.push_back(*item);
        return items;
    }

    // RenderView's batching (src/render/GeometryPasses.cpp), without a device: a draw ends at a change of buffers,
    // material or cull mode; an item whose geometry starts at the current draw's first index and whose instance index
    // follows the draw's last one joins it. Meshes without buffers (skinned copies, device-free) count as their own.
    struct ModelDraw
    {
        const void* buffers;
        const Material* material;
        uint32_t cullMode;
        uint32_t startIndex;
        uint32_t indexCount;
        int startInstance;
        uint32_t instanceCount;

        bool operator==(const ModelDraw& other) const
        {
            return buffers == other.buffers && material == other.material && cullMode == other.cullMode &&
                   startIndex == other.startIndex && indexCount == other.indexCount &&
                   startInstance == other.startInstance && instanceCount == other.instanceCount;
        }
    };

    std::vector<ModelDraw> Sorted(std::vector<ModelDraw> draws)
    {
        auto tie = [](const ModelDraw& d)
        {
            return std::make_tuple(d.buffers, d.material, d.cullMode, d.startIndex, d.indexCount, d.startInstance,
                                   d.instanceCount);
        };
        std::sort(draws.begin(), draws.end(), [&tie](const ModelDraw& a, const ModelDraw& b) { return tie(a) < tie(b); });
        return draws;
    }

    std::vector<ModelDraw> Batch(const std::vector<DrawItem>& items)
    {
        std::vector<ModelDraw> draws;
        const void* lastBuffers = nullptr;
        const Material* lastMaterial = nullptr;
        uint32_t lastCull = uint32_t(nvrhi::RasterCullMode::Back);
        ModelDraw current{};
        auto flush = [&]()
        {
            if (current.instanceCount)
                draws.push_back(current);
            current.instanceCount = 0;
        };
        for (const DrawItem& item : items)
        {
            if (!item.material)
                continue;
            const void* buffers = item.buffers ? static_cast<const void*>(item.buffers) : item.mesh;
            const bool newBuffers = buffers != lastBuffers;
            const bool newMaterial = item.material != lastMaterial || uint32_t(item.cullMode) != lastCull;
            if (newBuffers || newMaterial)
                flush();
            lastBuffers = buffers;
            lastMaterial = item.material;
            lastCull = uint32_t(item.cullMode);
            const uint32_t startIndex = item.mesh->indexOffset + item.geometry->indexOffsetInMesh;
            const int instance = item.instance->GetInstanceIndex();
            if (current.instanceCount > 0 && current.startIndex == startIndex &&
                current.startInstance + int(current.instanceCount) == instance)
            {
                ++current.instanceCount;
                continue;
            }
            flush();
            current = ModelDraw{ buffers, item.material, uint32_t(item.cullMode), startIndex, item.geometry->numIndices,
                                 instance, 1 };
        }
        flush();
        return draws;
    }

    // What the comparison saw.
    struct Tally
    {
        uint64_t views = 0;
        uint64_t items[c_DrawPacketPassCount] = {};
        uint64_t subgraphOnly[c_DrawPacketPassCount] = {};
        uint64_t boundDraws[c_DrawPacketPassCount] = {};
        uint64_t walkDraws[c_DrawPacketPassCount] = {};      // the walk's whole stream
        uint64_t orderUndefined = 0;                          // views whose legacy transparent order is undefined
        uint64_t tiedRuns = 0;                                // transparent runs of one distance over two instances
        uint64_t tieDraws = 0;   // transparent draws the walk saved by a lucky order of tied or undefined items
    };

    // The items of a bound strategy are the view's packets of its pass, in key order, with E5.2's mapping.
    void CheckAgainstPackets(const RenderSnapshot& s, uint32_t v, DrawPacketPass pass, const std::vector<DrawItem>& items)
    {
        const SnapshotRange range = s.views[v].packets[uint32_t(pass)];
        const DrawPackets& packets = s.packets;
        const VisibleInstances& in = s.instances;
        const bool transparent = pass == DrawPacketPass::Transparent;
        CHECK(items.size() == range.count);
        for (uint32_t k = 0; k < range.count; ++k)
        {
            const uint32_t i = range.first + k;
            const uint32_t vi = packets.visibleInstances[i];
            const MeshInfo* mesh = in.meshInfos[vi];
            const DrawItem& item = items[k];
            CHECK(item.instance == in.leaves[vi]);
            CHECK(item.mesh == mesh);
            CHECK(item.geometry == mesh->geometries[packets.geometryIndices[i]].Get());
            CHECK(item.material == item.geometry->material.Get());
            CHECK(item.buffers == mesh->buffers.Get());
            CHECK(uint32_t(item.cullMode) == packets.cullModes[i]);
            CHECK(DistanceBits(item.distanceToCamera) ==
                  (transparent ? DistanceBits(GetTransparentDrawKeyDistance(packets.keys[i])) : DistanceBits(0.f)));
            CHECK(item.userData == nullptr);
        }
    }

    uint32_t GeometryIndexOf(const DrawItem& item)
    {
        uint32_t geometry = 0;
        while (item.mesh->geometries[geometry].Get() != item.geometry)
            ++geometry;
        return geometry;
    }

    // The walk's items put into key order: what the packets would be had extraction kept exactly these items. Equal
    // keys (the geometries of one transparent instance, measured from one box) keep the geometries' order, as
    // extraction makes them in that order and sorts stably.
    std::vector<DrawItem> InKeyOrder(SceneGraph* graph, std::vector<DrawItem> items, bool transparent)
    {
        auto key = [graph, transparent](const DrawItem& item)
        {
            const uint32_t instanceSlot = uint32_t(item.instance->GetInstanceIndex());
            if (transparent)
                return MakeTransparentDrawKey(item.distanceToCamera, instanceSlot, uint32_t(item.cullMode));
            const MeshTable::MeshRow& row =
                graph->GetMeshes().GetMeshRow(MeshTable::SlotOf(graph->GetMeshes().Find(item.mesh)));
            const uint32_t materialSlot = MaterialTable::SlotOf(graph->GetMaterials().Find(item.material));
            return MakeOpaqueDrawKey(uint32_t(item.material->domain), uint32_t(item.cullMode), materialSlot,
                                     row.firstGeometry + GeometryIndexOf(item), instanceSlot);
        };
        std::stable_sort(items.begin(), items.end(), [&key](const DrawItem& a, const DrawItem& b)
        {
            const uint64_t ka = key(a), kb = key(b);
            return ka != kb ? ka < kb : GeometryIndexOf(a) < GeometryIndexOf(b);
        });
        return items;
    }

    // One view of one frame: the bound strategies against the same strategy objects walking the graph.
    struct Strategies
    {
        nvrhi::AutoPtr<InstancedOpaqueDrawStrategy> opaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        nvrhi::AutoPtr<TransparentDrawStrategy> transparent = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
        nvrhi::AutoPtr<PacketDrawStrategy> packets[c_DrawPacketPassCount] = { MAKE_RC_OBJ_PTR(PacketDrawStrategy),
                                                                             MAKE_RC_OBJ_PTR(PacketDrawStrategy) };

        void Bind(const RenderSnapshot* snapshot)
        {
            opaque->SetPackets(snapshot);
            transparent->SetPackets(snapshot);
            packets[0]->SetPackets(snapshot, DrawPacketPass::Opaque);
            packets[1]->SetPackets(snapshot, DrawPacketPass::Transparent);
        }

        IDrawStrategy* Legacy(uint32_t pass)
        {
            return pass == uint32_t(DrawPacketPass::Opaque) ? static_cast<IDrawStrategy*>(opaque.Get())
                                                            : static_cast<IDrawStrategy*>(transparent.Get());
        }
    };

    void CompareViews(SceneGraph* graph, const RenderSnapshot& s, Strategies& strategies, Tally& tally)
    {
        // Bound: one strategy object serves every view in turn (IGeometryPass::AddToGraph hands one to each child view);
        // the root node is ignored.
        std::vector<DrawItem> bound[TestViews::c_Count][c_DrawPacketPassCount];
        strategies.Bind(&s);
        for (uint32_t v = 0; v < s.viewCount; ++v)
        {
            for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
            {
                const DrawPacketPass pass = DrawPacketPass(p);
                bound[v][p] = Drain(strategies.Legacy(p), nullptr, *s.views[v].view);
                CheckAgainstPackets(s, v, pass, bound[v][p]);
                const std::vector<DrawItem> direct = Drain(strategies.packets[p], nullptr, *s.views[v].view);
                CHECK(direct.size() == bound[v][p].size());
                for (size_t i = 0; i < direct.size(); ++i)
                    CHECK(ItemOf(direct[i]) == ItemOf(bound[v][p][i]));
            }
        }

        // Unbound: the same objects walk the graph again.
        strategies.opaque->SetPackets(nullptr);
        strategies.transparent->SetPackets(nullptr);
        for (uint32_t v = 0; v < s.viewCount; ++v)
        {
            const IView& view = *s.views[v].view;
            const dm::frustum frustum = view.GetViewFrustum();
            ++tally.views;
            for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
            {
                const bool transparent = p == uint32_t(DrawPacketPass::Transparent);
                const std::vector<DrawItem>& ours = bound[v][p];
                const std::vector<DrawItem> walk = Drain(strategies.Legacy(p), graph->GetRootNode(), view);
                tally.items[p] += ours.size();

                // as multisets, except the walk's subgraph-only items
                std::vector<Item> sortedOurs;
                for (const DrawItem& item : ours)
                    sortedOurs.push_back(ItemOf(item));
                std::sort(sortedOurs.begin(), sortedOurs.end());
                std::vector<DrawItem> kept;
                std::vector<Item> sortedKept;
                for (const DrawItem& item : walk)
                {
                    if (std::binary_search(sortedOurs.begin(), sortedOurs.end(), ItemOf(item)))
                    {
                        kept.push_back(item);
                        sortedKept.push_back(ItemOf(item));
                        continue;
                    }
                    CHECK(!frustum.intersectsWith(OwnBox(item.instance)));
                    ++tally.subgraphOnly[p];
                }
                std::sort(sortedKept.begin(), sortedKept.end());
                CHECK(sortedKept == sortedOurs);

                // batching: the walk's items in key order batch exactly as ours (the same grouping where the order is
                // the same); opaque: at most the walk's draws
                const std::vector<ModelDraw> ourDraws = Batch(ours);
                const std::vector<ModelDraw> walkDraws = Batch(walk);
                const std::vector<ModelDraw> keptDraws = Batch(kept);
                if (!(ourDraws == Batch(InKeyOrder(graph, kept, transparent))))
                    std::printf("  view %u pass %u: the walk's items in key order batch differently\n", v, p);
                CHECK(ourDraws == Batch(InKeyOrder(graph, kept, transparent)));
                tally.boundDraws[p] += ourDraws.size();
                tally.walkDraws[p] += walkDraws.size();

                if (!transparent)
                {
                    CHECK(ourDraws.size() <= keptDraws.size());
                    continue;
                }
                // The transparent order: the walk's is undefined where its comparator is no strict weak ordering (an
                // instance whose items lie at different distances: skinned multi-geometry meshes); elsewhere, runs of
                // one distance hold the same items in both, in the same run order.
                std::map<const MeshInstance*, uint32_t> distanceOf;
                bool defined = true;
                for (const DrawItem& item : kept)
                {
                    auto [it, inserted] = distanceOf.emplace(item.instance, DistanceBits(item.distanceToCamera));
                    if (!inserted && it->second != DistanceBits(item.distanceToCamera))
                        defined = false;
                }
                if (!defined)
                {
                    ++tally.orderUndefined;
                    tally.tieDraws += ourDraws.size() > keptDraws.size() ? ourDraws.size() - keptDraws.size() : 0;
                    continue;
                }
                CHECK(kept.size() == ours.size());
                bool tied = false;
                size_t i = 0;
                while (i < ours.size())
                {
                    const uint32_t distance = DistanceBits(ours[i].distanceToCamera);
                    size_t j = i;
                    while (j < ours.size() && DistanceBits(ours[j].distanceToCamera) == distance)
                        ++j;
                    std::vector<Item> a, b;
                    bool instances = false;
                    for (size_t k = i; k < j; ++k)
                    {
                        CHECK(DistanceBits(kept[k].distanceToCamera) == distance);
                        a.push_back(ItemOf(ours[k]));
                        b.push_back(ItemOf(kept[k]));
                        instances = instances || ours[k].instance != ours[i].instance;
                    }
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    CHECK(a == b);
                    tally.tiedRuns += instances ? 1 : 0;
                    tied = tied || instances;
                    i = j;
                }
                // Without ties between instances the two orders are the same up to the order of one instance's
                // geometries (the walk's comparator leaves it to std::sort), and so are the draws as a multiset: one
                // instance's geometries never share a draw. A tie leaves the walk's order between instances to
                // std::sort too, which may by chance put two instances' items of one geometry next to each other where
                // the key (instance slot, then cull order) interleaves their double-sided draws.
                if (!tied)
                {
                    CHECK(Sorted(ourDraws) == Sorted(keptDraws));
                }
                else
                    tally.tieDraws += ourDraws.size() > keptDraws.size() ? ourDraws.size() - keptDraws.size() : 0;
            }
        }
    }

    // ------------------------------------------------------------------------------------------ cases

    constexpr uint64_t c_Seeds[] = { 11ull, 23ull, 0x5bd1e995ull };
    constexpr uint32_t c_Frames = 30;

    void test_stream_equivalence()
    {
        TestViews views;
        Tally tally;
        for (uint64_t seed : c_Seeds)
        {
            PacketScene scene(seed);
            scene.Build(150);
            RenderFrame frame;
            auto service = MakeService(scene.graph);
            Strategies strategies;
            for (uint32_t f = 1; f <= c_Frames; ++f)
            {
                scene.Edit(f);
                scene.graph->Refresh(f);
                const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f));
                CompareViews(scene.graph, snapshot, strategies, tally);
                service->EndFrame(true);
            }
        }
        std::printf("  %llu view-frames: %llu opaque and %llu transparent items; draws (RenderView's batching) bound / "
                    "walk: opaque %llu / %llu, transparent %llu / %llu\n", (unsigned long long)tally.views,
                    (unsigned long long)tally.items[0], (unsigned long long)tally.items[1],
                    (unsigned long long)tally.boundDraws[0], (unsigned long long)tally.walkDraws[0],
                    (unsigned long long)tally.boundDraws[1], (unsigned long long)tally.walkDraws[1]);
        std::printf("  known differences: %llu opaque and %llu transparent walk items kept only through a subgraph box; "
                    "%llu views with an undefined legacy transparent order (multisets only); %llu transparent runs of "
                    "one distance over two or more instances (compared as multisets); %llu transparent draws more than "
                    "the walk's where its order is tied or undefined\n",
                    (unsigned long long)tally.subgraphOnly[0], (unsigned long long)tally.subgraphOnly[1],
                    (unsigned long long)tally.orderUndefined, (unsigned long long)tally.tiedRuns,
                    (unsigned long long)tally.tieDraws);
        CHECK(tally.items[0] > 1000 && tally.items[1] > 1000);
        CHECK(tally.boundDraws[0] < tally.walkDraws[0]);   // packets merge geometries the walk's chunks separated
    }

    void test_view_lookup()
    {
        TestViews views;
        PacketScene scene(0x77);
        scene.Build(120);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        scene.Edit(1);
        scene.graph->Refresh(1);
        const RenderSnapshot* snapshot = &ExtractFrame(service, frame, views.Desc(1));

        auto opaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        auto transparent = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
        auto direct = MAKE_RC_OBJ_PTR(PacketDrawStrategy);

        // unbound PacketDrawStrategy: nothing, no warning
        uint32_t warnings = g_Warnings.load();
        CHECK(Drain(direct, scene.graph->GetRootNode(), *views.list[0]).empty());
        CHECK(g_Warnings.load() == warnings);

        // a view the snapshot does not hold: no item, one warning per view and Bind
        auto stranger = MAKE_RC_OBJ_PTR(PlanarView);
        stranger->SetViewport(nvrhi::Viewport(64.f, 64.f));
        stranger->SetMatrices(dm::affine3::identity(), dm::perspProjD3DStyleReverse(dm::radians(60.f), 1.f, 0.1f));
        stranger->UpdateCache();
        opaque->SetPackets(snapshot);
        direct->SetPackets(snapshot, DrawPacketPass::Transparent);
        CHECK(Drain(opaque, scene.graph->GetRootNode(), *stranger).empty());
        CHECK(g_Warnings.load() == warnings + 1);
        CHECK(Drain(opaque, scene.graph->GetRootNode(), *stranger).empty());
        CHECK(g_Warnings.load() == warnings + 1);
        CHECK(Drain(direct, scene.graph->GetRootNode(), *stranger).empty());
        CHECK(g_Warnings.load() == warnings + 2);
        opaque->SetPackets(snapshot);
        CHECK(Drain(opaque, scene.graph->GetRootNode(), *stranger).empty());
        CHECK(g_Warnings.load() == warnings + 3);
        g_Warnings.fetch_sub(3);

        // a composite view's children through one strategy, in turn: each gets its own packets
        auto composite = MAKE_RC_OBJ_PTR(CompositeView);
        for (uint32_t v = 0; v < TestViews::c_Count; ++v)
            composite->AddView(views.views[v]);
        uint32_t total = 0;
        for (uint32_t v = 0; v < composite->GetNumChildViews(ViewType::PLANAR); ++v)
        {
            const IView* child = composite->GetChildView(ViewType::PLANAR, v);
            const std::vector<DrawItem> items = Drain(opaque, nullptr, *child);
            CheckAgainstPackets(*snapshot, v, DrawPacketPass::Opaque, items);
            total += uint32_t(items.size());
        }
        CHECK(total > 0);

        // unbound: the graph walk, item for item as a strategy that was never bound
        opaque->SetPackets(nullptr);
        transparent->SetPackets(snapshot);
        transparent->SetPackets(nullptr);
        auto freshOpaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        auto freshTransparent = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
        for (uint32_t v = 0; v < TestViews::c_Count; ++v)
        {
            const std::vector<DrawItem> a = Drain(opaque, scene.graph->GetRootNode(), *views.list[v]);
            const std::vector<DrawItem> b = Drain(freshOpaque, scene.graph->GetRootNode(), *views.list[v]);
            CHECK(a.size() == b.size());
            for (size_t i = 0; i < a.size(); ++i)
                CHECK(ItemOf(a[i]) == ItemOf(b[i]));
            const std::vector<DrawItem> c = Drain(transparent, scene.graph->GetRootNode(), *views.list[v]);
            const std::vector<DrawItem> d = Drain(freshTransparent, scene.graph->GetRootNode(), *views.list[v]);
            CHECK(c.size() == d.size());
            for (size_t i = 0; i < c.size(); ++i)
                CHECK(ItemOf(c[i]) == ItemOf(d[i]));
        }
        service->EndFrame(true);

        // bound to the next frame's snapshot (the previous one died with the arena's Reset)
        scene.Edit(2);
        scene.graph->Refresh(2);
        snapshot = &ExtractFrame(service, frame, views.Desc(2));
        opaque->SetPackets(snapshot);
        transparent->SetPackets(snapshot);
        for (uint32_t v = 0; v < TestViews::c_Count; ++v)
        {
            CheckAgainstPackets(*snapshot, v, DrawPacketPass::Opaque, Drain(opaque, nullptr, *views.list[v]));
            CheckAgainstPackets(*snapshot, v, DrawPacketPass::Transparent, Drain(transparent, nullptr, *views.list[v]));
        }
        service->EndFrame(true);
        CHECK(g_Warnings.load() == warnings);
    }

    // Ten frames after warm-up: binding and draining every strategy allocates nothing (the items' storage keeps its
    // capacity), with the extraction's own steady state (test_draw_packets) around it.
    void test_steady_state()
    {
        TestViews views;
        PacketScene scene(0x5001);
        scene.Build(1500);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        Strategies strategies;
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
        uint64_t items = 0;
        auto run = [&]()
        {
            const dm::double3 swing = (f & 1) ? dm::double3(12.0, 0.0, 9.0) : dm::double3(0.0);
            for (const Mover& mover : movers)
                mover.node->SetTranslation(mover.home + swing);
            scene.graph->Refresh(++f);
            const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(f));
            strategies.Bind(&snapshot);
            for (uint32_t v = 0; v < snapshot.viewCount; ++v)
            {
                for (IDrawStrategy* strategy : { static_cast<IDrawStrategy*>(strategies.opaque.Get()),
                                                 static_cast<IDrawStrategy*>(strategies.transparent.Get()),
                                                 static_cast<IDrawStrategy*>(strategies.packets[0].Get()),
                                                 static_cast<IDrawStrategy*>(strategies.packets[1].Get()) })
                {
                    strategy->PrepareForView(nullptr, *snapshot.views[v].view);
                    while (strategy->GetNextItem())
                        ++items;
                }
            }
            service->EndFrame(true);
        };
        for (int i = 0; i < 6; ++i)
            run();
        const uint64_t newBefore = g_NewCount.load();
        items = 0;
        for (int i = 0; i < 10; ++i)
            run();
        const uint64_t newCalls = g_NewCount.load() - newBefore;
        std::printf("  %llu items over 10 frames of five views: %llu operator new\n", (unsigned long long)items,
                    (unsigned long long)newCalls);
        CHECK(items > 0);
        CHECK(newCalls == 0);
    }

    // Debug: the arena is filled with 0xDD at IRenderGraph::Reset, so a strategy prepared after the reset without a new
    // SetPackets sees an implausible view count, gives no item and warns once. Release keeps the old bytes (undefined:
    // not tested).
    void test_stale_snapshot()
    {
#ifndef NDEBUG
        TestViews views;
        PacketScene scene(0x88);
        scene.Build(60);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        scene.graph->Refresh(1);
        const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(1));
        auto opaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        opaque->SetPackets(&snapshot);
        CHECK(!Drain(opaque, nullptr, *views.list[0]).empty());
        service->EndFrame(true);
        frame.Begin();   // the next frame: Reset, no Extract yet
        const uint32_t warnings = g_Warnings.load();
        CHECK(Drain(opaque, nullptr, *views.list[0]).empty());
        CHECK(Drain(opaque, nullptr, *views.list[1]).empty());
        CHECK(g_Warnings.load() == warnings + 1);
        g_Warnings.fetch_sub(1);
        std::printf("  a strategy bound to a reset frame's snapshot: no item, one warning\n");
#else
        std::printf("  Debug only\n");
#endif
    }

    // Debug: the in-use check of the strategies (DrawStrategyUseCheck).
    void test_use_check()
    {
#ifndef NDEBUG
        TestViews views;
        PacketScene scene(0x99);
        scene.Build(60);
        RenderFrame frame;
        auto service = MakeService(scene.graph);
        scene.graph->Refresh(1);
        const RenderSnapshot& snapshot = ExtractFrame(service, frame, views.Desc(1));
        const uint32_t warnings = g_Warnings.load();
        for (bool bound : { true, false })
        {
            auto opaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
            opaque->SetPackets(bound ? &snapshot : nullptr);
            SceneGraphNode* root = scene.graph->GetRootNode();
            // one thread, then another, each preparing and draining: fine
            std::thread([&] { Drain(opaque, root, *views.list[0]); }).join();
            Drain(opaque, root, *views.list[0]);
            CHECK(g_Warnings.load() == warnings + (bound ? 0u : 1u));
            // prepared here, prepared on another thread, drained here: one warning, once per object
            opaque->PrepareForView(root, *views.list[0]);
            std::thread([&] { opaque->PrepareForView(root, *views.list[1]); }).join();
            while (opaque->GetNextItem())
            {
            }
            opaque->PrepareForView(root, *views.list[0]);
            std::thread([&] { opaque->PrepareForView(root, *views.list[1]); }).join();
            opaque->GetNextItem();
            CHECK(g_Warnings.load() == warnings + (bound ? 1u : 2u));
        }
        service->EndFrame(true);
        g_Warnings.fetch_sub(2);
        std::printf("  interleaved use on two threads: one warning per strategy object\n");
#else
        std::printf("  Debug only\n");
#endif
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
        { "stream_equivalence", test_stream_equivalence },
        { "view_lookup", test_view_lookup },
        { "steady_state", test_steady_state },
        { "stale_snapshot", test_stale_snapshot },
        { "use_check", test_use_check },
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
            CHECK(g_Warnings.load() == warnings);
            std::printf("%s: passed\n", c.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    std::printf("draw strategy adapters: ok\n");
    return 0;
}
