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

// Stable slots and the GPU scene without a device (aggregate docs/plans/2026-11-21-donut-ecs.md, step E4;
// docs/ecs-design.md 4.8, 7.2): instance and geometry-instance slots that stay through other instances' detach,
// SetLeaf and refreshes; mesh and material numbering by first use, independent of addresses; SceneGraph's asset
// callbacks; GpuScene's change tracking against a CPU copy of the buffers over seeded edit scripts (every live row equal
// to what the graph says, previous transforms as the old PrevTransform flags gave them); uploads proportional to what
// changed; a second update in a frame; meshes edited in place; switching graphs; no heap allocation in steady state;
// the review fixes of the step; out of memory in an update and in a buffer creation, retried. Scatter mode (plan step
// E5, sub-plan item E5.4): a CPU model of the scatter kernel applied to a mirror equals CPU mode's buffers after every
// frame of the scripts, growth and settles included; records, settles, the resend after an unacknowledged frame,
// CommitScatter, the growth bookkeeping, the records at 1, 2 and 8 workers, and no allocation in steady state.

#include <donut/engine/EcsQuery.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>
#include <donut/core/log.h>
#include <donut/tests/e7_switches.h>
#include <donut/tests/utils.h>

#include "EcsWorldImpl.h"
#include "SceneGraphStore.h"

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>
#include <donut/shaders/gpu_scene_scatter_cb.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <string>
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
    // Armed (>= 0): that many more allocations succeed, the next one fails (operator new throws std::bad_alloc), and
    // the countdown ends at -1.
    std::atomic<int64_t> g_FailNewAfter{ -1 };

    bool FailThisAllocation() noexcept
    {
        if (g_FailNewAfter.load(std::memory_order_relaxed) < 0)
            return false;
        return g_FailNewAfter.fetch_sub(1, std::memory_order_relaxed) == 0;
    }

    void* CountedAllocate(size_t size) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
        if (FailThisAllocation())
            return nullptr;
        return std::malloc(size ? size : 1);
    }

    void* CountedAllocateAligned(size_t size, size_t alignment) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
        if (FailThisAllocation())
            return nullptr;
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

namespace
{
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
    };

    nvrhi::AutoPtr<SceneGraphNode> make_node(const std::string& name)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName(name);
        return node;
    }

    nvrhi::AutoPtr<Material> make_material(const std::string& name, MaterialDomain domain)
    {
        auto material = MAKE_RC_OBJ_PTR(Material);
        material->name = name;
        material->domain = domain;
        return material;
    }

    nvrhi::AutoPtr<MeshInfo> make_mesh(const std::string& name, const std::vector<Material*>& materials,
                                       MeshType type = MeshType::Triangles)
    {
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->name = name;
        mesh->type = type;
        uint32_t index = 0;
        for (Material* material : materials)
        {
            auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
            geometry->material = material;
            geometry->numIndices = 3 * (++index);
            geometry->numVertices = 3;
            geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
            mesh->geometries.push_back(geometry);
        }
        mesh->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
        return mesh;
    }

    // A CPU copy of the three buffers, written only through IGpuSceneWriter: what a device would hold.
    struct Mirror final : IGpuSceneWriter
    {
        std::vector<uint8_t> buffers[3];
        uint64_t bytes = 0;
        uint32_t runs = 0;
        uint32_t resizes = 0;
        uint32_t constantWrites = 0;
        bool ok = true;

        static size_t RowSize(GpuSceneBuffer buffer)
        {
            switch (buffer)
            {
            case GpuSceneBuffer::Instances: return sizeof(InstanceData);
            case GpuSceneBuffer::Geometries: return sizeof(GeometryData);
            default: return sizeof(MaterialConstants);
            }
        }
        bool ResizeBuffer(GpuSceneBuffer buffer, uint32_t rows) noexcept override
        {
            if (failResizes > 0)
            {
                --failResizes;   // a creation that fails: the old buffer stays as it was
                return false;
            }
            // a new buffer: its old content is gone (poisoned, so that rows the upload misses show)
            buffers[uint32_t(buffer)].assign(size_t(rows) * RowSize(buffer), 0xcd);
            ++resizes;
            return true;
        }
        uint32_t failResizes = 0;   // the next this many creations fail
        void WriteRows(GpuSceneBuffer buffer, uint32_t firstRow, uint32_t rowCount, const void* data,
                       size_t rowSize) noexcept override
        {
            std::vector<uint8_t>& target = buffers[uint32_t(buffer)];
            if (rowSize != RowSize(buffer) || (size_t(firstRow) + rowCount) * rowSize > target.size())
            {
                ok = false;
                return;
            }
            std::memcpy(target.data() + size_t(firstRow) * rowSize, data, size_t(rowCount) * rowSize);
            bytes += uint64_t(rowCount) * rowSize;
            ++runs;
        }
        void WriteMaterialConstants(Material* material, const MaterialConstants& constants) noexcept override
        {
            ++constantWrites;
            constantsOf[material] = constants.materialID;
        }
        std::map<const Material*, int> constantsOf;   // the materialID each material's constant buffer got last

        void ResetCounters() { bytes = 0; runs = 0; resizes = 0; constantWrites = 0; }
        const InstanceData* Instances() const { return reinterpret_cast<const InstanceData*>(buffers[0].data()); }
        size_t InstanceRows() const { return buffers[0].size() / sizeof(InstanceData); }
        const GeometryData* Geometries() const { return reinterpret_cast<const GeometryData*>(buffers[1].data()); }
        size_t GeometryRows() const { return buffers[1].size() / sizeof(GeometryData); }
        const MaterialConstants* Materials() const { return reinterpret_cast<const MaterialConstants*>(buffers[2].data()); }
        size_t MaterialRows() const { return buffers[2].size() / sizeof(MaterialConstants); }
    };

    GpuSceneDesc test_desc(uint32_t granularity = 64)
    {
        GpuSceneDesc desc;
        desc.geometryAndMaterialBuffers = true;
        desc.rowGranularity = granularity;
        return desc;
    }

    // The float transform of an instance's node, as InstanceData stores it.
    void transform_of(const MeshInstance* instance, float out[12])
    {
        dm::affineToColumnMajor(instance->GetNode()->GetLocalToWorldTransformFloat(), out);
    }

    std::vector<MeshInstance*> instances_in_order(SceneGraph* graph)
    {
        std::vector<MeshInstance*> result;
        for (const auto& instance : graph->GetMeshInstances())
            result.push_back(instance.Get());
        return result;
    }
}

// Instance and geometry-instance slots: assigned at attach in registration order, kept through other instances'
// detach, SetLeaf and refreshes, reused lowest first.
void test_instance_slots_are_stable()
{
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    auto material = make_material("m", MaterialDomain::Opaque);
    auto one = make_mesh("one", { material });
    auto three = make_mesh("three", { material, material, material });

    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 10; ++i)
    {
        auto instance = MAKE_RC_OBJ_PTR(MeshInstance, (i % 2) ? three.Get() : one.Get());
        CHECK(instance->GetInstanceIndex() == -1 && instance->GetGeometryInstanceIndex() == -1);
        nodes.push_back(graph->AttachLeafNode(graph->GetRootNode(), instance));
    }
    auto instanceOf = [](SceneGraphNode* node) { return static_cast<MeshInstance*>(node->GetLeaf()); };
    int geometryInstance = 0;
    for (int i = 0; i < 10; ++i)
    {
        CHECK(instanceOf(nodes[i])->GetInstanceIndex() == i);
        CHECK(instanceOf(nodes[i])->GetGeometryInstanceIndex() == geometryInstance);
        geometryInstance += (i % 2) ? 3 : 1;
    }
    CHECK(graph->GetInstanceSlotCount() == 10 && graph->GetGeometryInstancesCount() == size_t(geometryInstance));
    graph->Refresh(1);
    CHECK(instanceOf(nodes[9])->GetInstanceIndex() == 9);

    // a detach leaves the others where they are; the detached instance has no slot
    auto detached = graph->Detach(nodes[3]);
    MeshInstance* detachedInstance = instanceOf(detached);
    CHECK(detachedInstance->GetInstanceIndex() == -1);
    graph->Refresh(2);
    for (int i = 0; i < 10; ++i)
        if (i != 3)
            CHECK(instanceOf(nodes[i])->GetInstanceIndex() == i);

    // the next instance takes the lowest free slot, and a geometry-instance range that fits
    auto replacement = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, three.Get()));
    CHECK(instanceOf(replacement)->GetInstanceIndex() == 3);
    CHECK(instanceOf(replacement)->GetGeometryInstanceIndex() == 5);   // instance 3's freed range [5, 8), first fit

    // SetLeaf: the old instance's slot goes back, the new one takes the lowest free
    nodes[5]->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, one.Get()));
    CHECK(instanceOf(nodes[5])->GetInstanceIndex() == 5);
    CHECK(graph->GetInstanceSlotCount() == 10);

    // the detached subgraph attached again gets a slot (lowest free: past the end now)
    graph->Attach(graph->GetRootNode(), detached);
    CHECK(detachedInstance->GetInstanceIndex() == 10);
}

// Mesh and material numbering follows first use, not object addresses: two builds of the same scene whose objects are
// allocated in different orders number them alike.
void test_numbering_by_first_use()
{
    auto build = [](bool reverse)
    {
        std::vector<nvrhi::AutoPtr<Material>> materials(6);
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes(8);
        std::vector<std::vector<uint8_t>> padding;   // perturbs the heap between allocations
        for (int k = 0; k < 6; ++k)
        {
            const int i = reverse ? 5 - k : k;
            padding.emplace_back(size_t(64 + 32 * k));
            materials[i] = make_material("t" + std::to_string(i), MaterialDomain(i % 3));
        }
        for (int k = 0; k < 8; ++k)
        {
            const int i = reverse ? 7 - k : k;
            padding.emplace_back(size_t(48 + 16 * k));
            meshes[i] = make_mesh("m" + std::to_string(i), { materials[(i * 5) % 6], materials[(i + 1) % 6] });
        }
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));
        for (int i = 0; i < 20; ++i)
            graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, meshes[(i * 3) % 8].Get()));
        graph->Refresh(0);

        std::string record;
        for (MeshInfo* mesh : graph->GetMeshes())
        {
            record += mesh->name + "@" + std::to_string(mesh->globalMeshIndex) + ":";
            for (const auto& geometry : mesh->geometries)
                record += std::to_string(geometry->globalGeometryIndex) + "/" + geometry->material->name + "@" +
                          std::to_string(geometry->material->materialID) + " ";
        }
        for (Material* material : graph->GetMaterials())
            record += material->name + " ";
        return std::make_pair(record, graph);
    };
    const auto forward = build(false);
    const auto backward = build(true);
    CHECK(forward.first == backward.first);
    // first use: instance i uses mesh (3i) % 8, so the meshes come in the order 0, 3, 6, 1, 4, 7, 2, 5
    std::vector<std::string> names;
    for (MeshInfo* mesh : forward.second->GetMeshes())
        names.push_back(mesh->name);
    CHECK(names.size() == 8 && names[0] == "m0" && names[1] == "m3" && names[2] == "m6" && names[3] == "m1");
    uint32_t slot = 0;
    for (MeshInfo* mesh : forward.second->GetMeshes())
        CHECK(mesh->globalMeshIndex == int(slot++));
    // geometry slots are each mesh's own consecutive range
    for (MeshInfo* mesh : forward.second->GetMeshes())
        CHECK(mesh->geometries[1]->globalGeometryIndex == mesh->geometries[0]->globalGeometryIndex + 1);
    CHECK(forward.second->GetGeometryCount() == 16);
}

namespace
{
    // A graph whose RegisterLeaf records the instance slot it sees.
    class SlotProbingGraph final : public SceneGraph
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        std::vector<int> seen;
    protected:
        void RegisterLeaf(SceneGraphLeaf* leaf) override
        {
            if (auto instance = query_cast<MeshInstance>(leaf))
                seen.push_back(instance->GetInstanceIndex());
            SceneGraph::RegisterLeaf(leaf);
        }
    };
}

// SceneGraph's callbacks come from the tables, in RegisterLeaf's order, and RegisterLeaf overrides see the slots.
void test_asset_callbacks()
{
    auto graph = MAKE_RC_OBJ_PTR(SlotProbingGraph);
    std::string events;
    graph->OnMeshAdded = [&](const MeshInfo* mesh) { events += "+m:" + mesh->name + " "; };
    graph->OnMeshRemoved = [&](const MeshInfo* mesh) { events += "-m:" + mesh->name + " "; };
    graph->OnMaterialAdded = [&](const Material* material) { events += "+t:" + material->name + " "; };
    graph->OnMaterialRemoved = [&](const Material* material) { events += "-t:" + material->name + " "; };
    graph->SetRootNode(make_node("root"));

    auto stone = make_material("stone", MaterialDomain::Opaque);
    auto glass = make_material("glass", MaterialDomain::AlphaBlended);
    auto house = make_mesh("house", { stone, glass });
    auto rock = make_mesh("rock", { stone });
    auto a = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, house.Get()));
    auto b = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, house.Get()));
    auto c = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, rock.Get()));
    CHECK(events == "+m:house +t:stone +t:glass +m:rock ");
    CHECK(graph->seen == std::vector<int>({ 0, 1, 2 }));
    CHECK(graph->GetMaxGeometryCountPerMesh() == 2);

    events.clear();
    graph->Detach(a);
    CHECK(events.empty());
    graph->Detach(b);
    CHECK(events == "-m:house -t:glass ");
    graph->Detach(c);
    CHECK(events == "-m:house -t:glass -m:rock -t:stone ");
    CHECK(graph->GetMeshes().empty() && graph->GetMaterials().empty());
}

namespace
{
    // The expected rows of every live instance against the mirror and the GpuScene's image, and the previous transforms
    // against the model: the transform of the same entity in the same slot at the previous frame, else the current one.
    struct PrevModel
    {
        struct Entry
        {
            uint32_t slot;
            float transform[12];
        };
        std::map<uint64_t, Entry> last;
    };

    void verify_frame(SceneGraph* graph, const GpuScene& gpu, const Mirror& mirror, PrevModel& model, const char* where)
    {
        CHECK(mirror.ok);
        const InstanceData* image = gpu.GetInstanceRows();
        std::map<uint64_t, PrevModel::Entry> next;
        for (MeshInstance* instance : instances_in_order(graph))
        {
            const int slot = instance->GetInstanceIndex();
            CHECK(slot >= 0 && size_t(slot) < mirror.InstanceRows() && uint32_t(slot) < gpu.GetInstanceRowCount());
            const InstanceData& row = mirror.Instances()[slot];
            if (std::memcmp(&row, &image[slot], sizeof(row)) != 0)
            {
                std::fprintf(stderr, "%s: slot %d: the buffer differs from the image\n", where, slot);
                CHECK(false);
            }
            const MeshInfo* mesh = instance->GetMesh();
            const uint32_t geometries = uint32_t(mesh->geometries.size());
            CHECK(row.numGeometries == geometries);
            CHECK(row.firstGeometryIndex == (geometries ? uint32_t(mesh->geometries[0]->globalGeometryIndex) : ~0u));
            CHECK(int(row.firstGeometryInstanceIndex) == instance->GetGeometryInstanceIndex());
            const uint32_t flags = mesh->type == MeshType::CurveLinearSweptSpheres ? InstanceFlags_CurveLinearSweptSpheres
                : mesh->type == MeshType::CurveDisjointOrthogonalTriangleStrips ? InstanceFlags_CurveDisjointOrthogonalTriangleStrips : 0u;
            CHECK(row.flags == flags);

            PrevModel::Entry entry{};
            entry.slot = uint32_t(slot);
            transform_of(instance, entry.transform);
            CHECK(std::memcmp(&row.transform, entry.transform, sizeof(entry.transform)) == 0);
            const uint64_t entity = uint64_t(graph->GetEntity(instance->GetNode()));
            CHECK(gpu.GetInstanceSlotEntity(uint32_t(slot)) == Entity(entity));
            auto it = model.last.find(entity);
            const float* expectedPrev = it != model.last.end() && it->second.slot == uint32_t(slot)
                ? it->second.transform : entry.transform;
            if (std::memcmp(&row.prevTransform, expectedPrev, sizeof(entry.transform)) != 0)
            {
                std::fprintf(stderr, "%s: slot %d: unexpected previous transform\n", where, slot);
                CHECK(false);
            }
            next[entity] = entry;
        }
        model.last.swap(next);

        // geometry and material rows of every mesh and material in the tables
        for (MeshInfo* mesh : graph->GetMeshes())
        {
            for (const auto& geometry : mesh->geometries)
            {
                const uint32_t slot = uint32_t(geometry->globalGeometryIndex);
                CHECK(slot < mirror.GeometryRows());
                const GeometryData& row = mirror.Geometries()[slot];
                CHECK(row.numIndices == geometry->numIndices && row.numVertices == geometry->numVertices);
                CHECK(row.materialIndex == uint32_t(geometry->material->materialID));
                CHECK(row.indexBufferIndex == -1 && row.vertexBufferIndex == -1);
            }
        }
        for (Material* material : graph->GetMaterials())
        {
            const uint32_t slot = uint32_t(material->materialID);
            CHECK(slot < mirror.MaterialRows());
            MaterialConstants expected = MaterialConstants();
            material->FillConstantBuffer(expected);
            if (std::memcmp(&mirror.Materials()[slot], &expected, sizeof(expected)) != 0)
            {
                const MaterialConstants& actual = mirror.Materials()[slot];
                std::fprintf(stderr, "%s: material '%s' (slot %u): the buffer differs (id %d/%d, roughness %g/%g, "
                             "domain %d/%d, image id %d)\n", where, material->name.c_str(), slot, actual.materialID,
                             expected.materialID, actual.roughness, expected.roughness, actual.domain, expected.domain,
                             gpu.GetMaterialRows()[slot].materialID);
                for (size_t w = 0; w < sizeof(expected) / 4; ++w)
                {
                    uint32_t x = 0, y = 0;
                    std::memcpy(&x, reinterpret_cast<const uint8_t*>(&actual) + 4 * w, 4);
                    std::memcpy(&y, reinterpret_cast<const uint8_t*>(&expected) + 4 * w, 4);
                    if (x != y)
                        std::fprintf(stderr, "  word %zu: %08x / %08x\n", w, x, y);
                }
                CHECK(false);
            }
        }
    }
}

namespace
{
    // The seeded edit script (also driven in scatter mode, plan step E5): `frames` frames of moves (local and of
    // parents), attaches, detaches and re-attaches, SetLeaf, InvalidateContent, material edits, index-count edits and
    // idle frames (every 7th), with skinned instances, lights and curve meshes; after each frame's graph->Refresh,
    // onFrame(graph, frame).
    void run_edit_script(uint64_t seed, uint32_t frames, const std::function<void(SceneGraph*, uint32_t)>& onFrame)
    {
        Rng rng{ seed };
        std::vector<nvrhi::AutoPtr<Material>> materials;
        for (int i = 0; i < 5; ++i)
            materials.push_back(make_material("t" + std::to_string(i), MaterialDomain(i % 4)));
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
        for (int i = 0; i < 7; ++i)
        {
            std::vector<Material*> used;
            for (int g = 0; g <= i % 3; ++g)
                used.push_back(materials[(i + g) % 5]);
            meshes.push_back(make_mesh("m" + std::to_string(i), used,
                i == 5 ? MeshType::CurveLinearSweptSpheres : i == 6 ? MeshType::CurveDisjointOrthogonalTriangleStrips
                                                                  : MeshType::Triangles));
        }
        auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));

        auto newLeaf = [&]() -> nvrhi::AutoPtr<SceneGraphLeaf>
        {
            const uint32_t pick = rng.Below(20);
            if (pick == 0)
                return MAKE_RC_OBJ_PTR(PointLight);
            if (pick == 1)
                return factory->CreateSkinnedMeshInstance(factory, meshes[rng.Below(4)]);
            return MAKE_RC_OBJ_PTR(MeshInstance, meshes[rng.Below(uint32_t(meshes.size()))].Get());
        };
        // an orphaned subgraph of 1-6 nodes, some with transforms and some without leaves
        int made = 0;
        auto newSubgraph = [&]()
        {
            auto top = make_node("s" + std::to_string(made++));
            top->SetTranslation(dm::double3(rng.Unit() * 10.0, rng.Unit(), 0.0));
            top->SetLeaf(newLeaf());
            const uint32_t children = rng.Below(6);
            for (uint32_t c = 0; c < children; ++c)
            {
                auto child = make_node("c" + std::to_string(c));
                if (rng.Below(3))
                    child->SetTranslation(dm::double3(0.0, rng.Unit(), rng.Unit()));
                if (rng.Below(5))
                    child->SetLeaf(newLeaf());
                graph->Attach(top, child);
            }
            return top;
        };
        auto attached = [&]()
        {
            std::vector<SceneGraphNode*> nodes;
            for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
                nodes.push_back(walker.Get());
            return nodes;
        };

        for (int i = 0; i < 120; ++i)
        {
            const auto nodes = attached();
            graph->Attach(nodes[rng.Below(uint32_t(nodes.size()))], newSubgraph());
        }

        std::vector<nvrhi::AutoPtr<SceneGraphNode>> pool;
        for (uint32_t frame = 1; frame <= frames; ++frame)
        {
            if (frame % 7 != 0)
            {
                auto nodes = attached();
                const uint32_t moves = uint32_t(nodes.size()) / 15 + 1;
                for (uint32_t k = 0; k < moves; ++k)
                    nodes[rng.Below(uint32_t(nodes.size()))]->SetTranslation(
                        dm::double3(rng.Unit() * 5.0, rng.Unit() * 5.0, rng.Unit()));
                for (uint32_t k = rng.Below(3); k > 0; --k)
                    graph->Attach(nodes[rng.Below(uint32_t(nodes.size()))], newSubgraph());
                if (!pool.empty() && rng.Below(2))
                {
                    graph->Attach(nodes[rng.Below(uint32_t(nodes.size()))], pool.back());
                    pool.pop_back();
                }
                nodes = attached();
                if (nodes.size() > 1 && rng.Below(2))
                    pool.push_back(graph->Detach(nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]));
                nodes = attached();
                if (rng.Below(10) < 3)
                    nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]->SetLeaf(newLeaf());
                if (rng.Below(10) < 2)
                    nodes[1 + rng.Below(uint32_t(nodes.size()) - 1)]->InvalidateContent();
                if (rng.Below(10) < 2)
                {
                    Material* material = materials[rng.Below(5)];
                    material->roughness = float(rng.Unit());
                    material->dirty = true;
                }
                if (rng.Below(10) == 0)
                    meshes[rng.Below(uint32_t(meshes.size()))]->geometries[0]->numIndices += 3;
            }
            graph->Refresh(frame);
            onFrame(graph, frame);
        }
    }
}

// The change tracking over seeded edit scripts: after every frame the buffers equal the graph, and a second update in
// the same frame writes nothing.
void test_gpu_scene_tracks_changes()
{
    for (uint64_t seed : { 11ull, 23ull, 0x5bd1e995ull })
    {
        GpuScene gpu(nullptr, test_desc());
        Mirror mirror;
        PrevModel model;
        run_edit_script(seed, 30, [&](SceneGraph* graph, uint32_t)
        {
            CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)));
            CHECK(NVRHI_SUCCEEDED(gpu.UploadTo(&mirror)));
            // a second update in the same frame (VXGISample refreshes the buffers twice) changes nothing
            mirror.ResetCounters();
            CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)));
            CHECK(gpu.GetStats().rowsChanged == 0 && gpu.GetStats().rowsSettled == 0);
            CHECK(NVRHI_SUCCEEDED(gpu.UploadTo(&mirror)));
            CHECK(mirror.bytes == 0 && mirror.constantWrites == 0);
            verify_frame(graph, gpu, mirror, model, "script");
        });
    }
}

// Uploads follow what changed: nothing for an idle frame, one row for one move, the settling of the previous frame's
// moves once, and runs of neighbouring rows merged.
void test_uploads_follow_changes()
{
    auto material = make_material("m", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { material });
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 4096; ++i)
    {
        auto node = make_node("n");
        node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        nodes.push_back(graph->Attach(graph->GetRootNode(), node));
    }
    for (int i = 0; i < 4096; ++i)
        CHECK(static_cast<MeshInstance*>(nodes[i]->GetLeaf())->GetInstanceIndex() == i);

    GpuScene gpu(nullptr, test_desc(1024));
    Mirror mirror;
    PrevModel model;
    uint32_t frame = 0;
    auto step = [&]()
    {
        graph->Refresh(++frame);
        mirror.ResetCounters();
        CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)));
        CHECK(NVRHI_SUCCEEDED(gpu.UploadTo(&mirror)));
        verify_frame(graph, gpu, mirror, model, "uploads");
        return gpu.GetStats();
    };

    GpuSceneStats stats = step();
    CHECK(mirror.resizes == 3 && stats.rowsBuilt == 4096);
    stats = step();   // idle
    CHECK(mirror.bytes == 0 && stats.chunksChanged == 0 && stats.rowsBuilt == 0);
    const uint32_t chunks = stats.chunksVisited;
    CHECK(chunks > 1);

    nodes[100]->SetTranslation(dm::double3(1.0, 2.0, 3.0));
    stats = step();
    CHECK(stats.chunksChanged == 1 && stats.rowsChanged == 1 && mirror.bytes == sizeof(InstanceData) && mirror.runs == 1);
    std::printf("uploads: %u chunks, %u rows built for one move\n", chunks, stats.rowsBuilt);
    CHECK(stats.rowsBuilt < 4096 / 2);
    stats = step();   // the moved row's previous transform settles
    CHECK(stats.rowsSettled == 1 && stats.rowsChanged == 0 && mirror.bytes == sizeof(InstanceData));
    stats = step();
    CHECK(mirror.bytes == 0);

    // 256 neighbours: one run
    for (int i = 1000; i < 1256; ++i)
        nodes[i]->SetTranslation(dm::double3(double(i), 1.0, 0.0));
    stats = step();
    CHECK(stats.rowsChanged == 256 && mirror.runs == 1 && mirror.bytes == 256 * sizeof(InstanceData));
    CHECK(stats.chunksChanged < chunks);

    // every 16th row: runs stay apart (the gap is wider than mergeGap); the previous frame's 256 rows settle, 16 of
    // which move again (every chunk is rebuilt now, so the settling happens there)
    for (int i = 0; i < 4096; i += 16)
        nodes[i]->SetTranslation(dm::double3(double(i), 2.0, 0.0));
    stats = step();
    CHECK(stats.chunksChanged == chunks && stats.rowsChanged + stats.rowsSettled == 256 + 240);
    CHECK(mirror.bytes >= 496 * sizeof(InstanceData) && mirror.bytes < 1024 * sizeof(InstanceData));
    stats = step();
    CHECK(stats.rowsSettled == 256);
    stats = step();
    CHECK(mirror.bytes == 0);

    // a material edit writes its row and its constant buffer, nothing else
    material->roughness = 0.25f;
    material->dirty = true;
    stats = step();
    CHECK(stats.materialRowsChanged == 1 && mirror.constantWrites == 1 && mirror.bytes == sizeof(MaterialConstants));
    CHECK(!material->dirty);
}

// A mesh edited in place gets new geometry slots at the next structural refresh, and so do its instances'
// geometry-instance ranges; the rows follow.
void test_meshes_edited_in_place()
{
    auto stone = make_material("stone", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { stone });
    auto other = make_mesh("other", { stone });
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    std::vector<MeshInstance*> instances;
    for (int i = 0; i < 6; ++i)
    {
        auto instance = MAKE_RC_OBJ_PTR(MeshInstance, (i % 2) ? other.Get() : mesh.Get());
        graph->AttachLeafNode(graph->GetRootNode(), instance);
        instances.push_back(instance);
    }
    GpuScene gpu(nullptr, test_desc());
    Mirror mirror;
    PrevModel model;
    graph->Refresh(1);
    gpu.Update(graph);
    gpu.UploadTo(&mirror);
    verify_frame(graph, gpu, mirror, model, "edit 1");

    auto glass = make_material("glass", MaterialDomain::AlphaBlended);
    auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
    geometry->material = glass;
    geometry->numIndices = 6;
    mesh->geometries.push_back(geometry);
    graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, other.Get()));   // a structural change
    graph->Refresh(2);
    CHECK(graph->GetMaterials().size() == 2 && glass->materialID == 1);
    CHECK(mesh->geometries[1]->globalGeometryIndex == mesh->geometries[0]->globalGeometryIndex + 1);
    // geometry-instance ranges: two per instance of `mesh`, none overlapping
    std::vector<int> owner(graph->GetGeometryInstancesCount(), -1);
    for (MeshInstance* instance : instances_in_order(graph))
    {
        const int first = instance->GetGeometryInstanceIndex();
        for (size_t g = 0; g < instance->GetMesh()->geometries.size(); ++g)
        {
            CHECK(owner[first + g] == -1);
            owner[first + g] = instance->GetInstanceIndex();
        }
    }
    gpu.Update(graph);
    gpu.UploadTo(&mirror);
    verify_frame(graph, gpu, mirror, model, "edit 2");
    CHECK(mirror.Instances()[instances[0]->GetInstanceIndex()].numGeometries == 2);
}

// A GpuScene given another graph starts over with it.
void test_another_graph()
{
    auto stone = make_material("stone", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { stone, stone });
    auto build = [&](int count, double offset)
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));
        for (int i = 0; i < count; ++i)
        {
            auto node = make_node("n");
            node->SetTranslation(dm::double3(offset + i, 0.0, 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
            graph->Attach(graph->GetRootNode(), node);
        }
        graph->Refresh(1);
        return graph;
    };
    auto a = build(40, 0.0);
    auto b = build(25, 100.0);
    GpuScene gpu(nullptr, test_desc());
    Mirror mirror;
    PrevModel modelA, modelB;
    gpu.Update(a);
    gpu.UploadTo(&mirror);
    verify_frame(a, gpu, mirror, modelA, "graph a");
    mirror.ResetCounters();
    gpu.Update(b);
    gpu.UploadTo(&mirror);
    // every row is written again for the new graph, into the same buffers (binding sets made from them stay valid)
    CHECK(mirror.resizes == 0 && gpu.GetStats().buffersResized == 0);
    CHECK(mirror.bytes == uint64_t(gpu.GetInstanceRowCount()) * sizeof(InstanceData) +
                          uint64_t(gpu.GetGeometryRowCount()) * sizeof(GeometryData) +
                          uint64_t(gpu.GetMaterialRowCount()) * sizeof(MaterialConstants));
    verify_frame(b, gpu, mirror, modelB, "graph b");
}

// Frames of moves allocate nothing once the scratch has grown: no operator new, no ECS heap block.
void test_steady_state_allocations()
{
    auto material = make_material("m", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { material, material });
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    std::vector<SceneGraphNode*> nodes;
    for (int i = 0; i < 2000; ++i)
    {
        auto node = make_node("n");
        node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        nodes.push_back(graph->Attach(graph->GetRootNode(), node));
    }
    GpuScene gpu(nullptr, test_desc());
    Mirror mirror;
    Rng rng{ 99 };
    uint32_t frame = 0;
    auto run = [&](uint32_t moves)
    {
        for (uint32_t k = 0; k < moves; ++k)
            nodes[rng.Below(2000)]->SetTranslation(dm::double3(rng.Unit(), rng.Unit(), rng.Unit()));
        graph->Refresh(++frame);
        gpu.Update(graph);
        gpu.UploadTo(&mirror);
    };
    for (int i = 0; i < 6; ++i)
        run(400);   // grows the runs and settle lists past what 200 moves need
    const uint64_t newBefore = g_NewCount.load();
    const uint64_t heapBefore = ecs::GetHeapAllocationCount();
    for (int i = 0; i < 8; ++i)
        run(200);
    run(0);
    run(0);
    const uint64_t newCalls = g_NewCount.load() - newBefore;
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
    std::printf("gpu scene steady state: %llu operator new, %llu storage blocks\n", (unsigned long long)newCalls,
                (unsigned long long)heapBlocks);
    CHECK(newCalls == 0 && heapBlocks == 0);
}

namespace
{
    // A graph whose RegisterLeaf and UnregisterLeaf overrides record the slots they see: their own leaf's and those of
    // every mesh instance attached under the same parent.
    class SlotWatchingGraph final : public SceneGraph
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        std::vector<int> unregisteredSlots;
        std::vector<int> siblingSlots;   // at each RegisterLeaf: the slots of the parent's other mesh-instance children
    protected:
        void RegisterLeaf(SceneGraphLeaf* leaf) override
        {
            if (query_cast<MeshInstance>(leaf) && leaf->GetNode() && leaf->GetNode()->GetParent())
            {
                for (SceneGraphNode* sibling : leaf->GetNode()->GetParent()->GetChildren())
                {
                    if (sibling == leaf->GetNode())
                        continue;
                    if (auto instance = query_cast<MeshInstance>(sibling->GetLeaf()))
                        siblingSlots.push_back(instance->GetInstanceIndex());
                }
            }
            SceneGraph::RegisterLeaf(leaf);
        }
        void UnregisterLeaf(SceneGraphLeaf* leaf) override
        {
            if (auto instance = query_cast<MeshInstance>(leaf))
                unregisteredSlots.push_back(instance->GetInstanceIndex());
            SceneGraph::UnregisterLeaf(leaf);
        }
    };
}

// The review fixes of step E4, one block each.
void test_review_fixes()
{
    // 1. A mesh attached between the graph's refresh and the update gets rows without buffers; the next frame's
    //    CreateMeshBuffers is followed by InvalidateAssets (Scene::RefreshBuffers), which fills them again.
    {
        auto material = make_material("m", MaterialDomain::Opaque);
        auto mesh = make_mesh("late", { material });
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));
        graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, make_mesh("early", { material }).Get()));
        GpuScene gpu(nullptr, test_desc());
        Mirror mirror;
        graph->Refresh(1);
        gpu.Update(graph);
        gpu.UploadTo(&mirror);

        graph->Refresh(2);
        graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));   // after the refresh
        gpu.Update(graph);
        gpu.UploadTo(&mirror);
        const uint32_t slot = uint32_t(mesh->geometries[0]->globalGeometryIndex);
        CHECK(mirror.Geometries()[slot].positionOffset == ~0u);   // no buffers yet

        // the next frame: the structure was pending, CreateMeshBuffers gives the mesh its buffers
        CHECK(graph->HasPendingStructureChanges());
        graph->Refresh(3);
        auto buffers = MAKE_RC_OBJ_PTR(BufferGroup);
        buffers->getVertexBufferRange(VertexAttribute::Position) = nvrhi::BufferRange(256, 36);
        mesh->buffers = buffers;
        gpu.InvalidateAssets();
        gpu.Update(graph);
        gpu.UploadTo(&mirror);
        CHECK(mirror.Geometries()[slot].positionOffset == 256);
    }

    // 2. A material shared with another graph: this graph's slot in its geometry rows and constant buffer, whatever the
    //    other graph wrote into Material::materialID; a structural refresh writes this graph's slot back.
    {
        auto shared = make_material("shared", MaterialDomain::Opaque);
        auto other = make_material("other", MaterialDomain::Opaque);
        auto a = MAKE_RC_OBJ_PTR(SceneGraph);
        a->SetRootNode(make_node("root"));
        a->AttachLeafNode(a->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, make_mesh("a", { shared }).Get()));
        auto b = MAKE_RC_OBJ_PTR(SceneGraph);
        b->SetRootNode(make_node("root"));
        b->AttachLeafNode(b->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, make_mesh("b", { other, shared }).Get()));
        CHECK(shared->materialID == 1);   // b's slot: b added it last
        GpuScene gpu(nullptr, test_desc());
        Mirror mirror;
        a->Refresh(1);
        gpu.Update(a);
        gpu.UploadTo(&mirror);
        CHECK(mirror.Geometries()[0].materialIndex == 0);
        CHECK(mirror.Materials()[0].materialID == 0);
        CHECK(mirror.constantsOf[shared.Get()] == 0);
        shared->roughness = 0.75f;
        shared->dirty = true;
        a->Refresh(2);
        gpu.Update(a);
        gpu.UploadTo(&mirror);
        CHECK(mirror.Materials()[0].roughness == 0.75f && mirror.constantsOf[shared.Get()] == 0);
        // a structural refresh of a gives the material a's slot again (the last refresh wins, as before)
        a->AttachLeafNode(a->GetRootNode(), MAKE_RC_OBJ_PTR(PointLight));
        a->Refresh(3);
        CHECK(shared->materialID == 0);
    }

    // 3. Rows without slots report -1, and releasing them frees nothing: during a Detach (UnregisterLeaf), before the
    //    leaf observer reached a leaf of the same batch, and after an acquisition that threw.
    {
        auto material = make_material("m", MaterialDomain::Opaque);
        auto mesh = make_mesh("mesh", { material });
        auto graph = MAKE_RC_OBJ_PTR(SlotWatchingGraph);
        graph->SetRootNode(make_node("root"));
        auto first = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        CHECK(static_cast<MeshInstance*>(first->GetLeaf())->GetInstanceIndex() == 0);

        // a batch of three siblings: the first registered sees the other two without slots
        auto group = make_node("group");
        for (int i = 0; i < 3; ++i)
        {
            auto child = make_node("c");
            child->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
            graph->Attach(group, child);
        }
        graph->siblingSlots.clear();
        graph->Attach(graph->GetRootNode(), group);
        CHECK(graph->siblingSlots.size() == 6);
        CHECK(graph->siblingSlots[0] == -1 && graph->siblingSlots[1] == -1);   // the first: neither sibling has one yet
        CHECK(graph->siblingSlots[4] >= 1 && graph->siblingSlots[5] >= 1);    // the last: both have theirs

        graph->Detach(group);
        CHECK(graph->unregisteredSlots == std::vector<int>({ -1, -1, -1 }));   // released before UnregisterLeaf

        // an acquisition that throws (out of memory): the leaf is skipped and holds no slot; detaching it frees none
        size_t warnings = 0;
        log::SetCallback([&](log::Severity, const char*) { ++warnings; });
        SceneGraphStore::FailAcquireAfter(0);
        auto failed = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        SceneGraphStore::FailAcquireAfter(-1);
        log::ResetCallback();
        CHECK(warnings == 1);
        CHECK(static_cast<MeshInstance*>(failed->GetLeaf())->GetInstanceIndex() == -1);
        CHECK(graph->GetMeshInstances().size() == 1);
        graph->Detach(failed);
        auto next = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        CHECK(static_cast<MeshInstance*>(first->GetLeaf())->GetInstanceIndex() == 0);
        CHECK(static_cast<MeshInstance*>(next->GetLeaf())->GetInstanceIndex() == 1);   // slot 0 was never freed
    }

    // 7. A material that left the table and comes back at another slot gets its constant buffer written again with
    //    the new slot, without a dirty flag.
    {
        auto keep = make_material("keep", MaterialDomain::Opaque);
        auto leave = make_material("leave", MaterialDomain::Opaque);
        auto late = make_material("late", MaterialDomain::Opaque);
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));
        graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, make_mesh("k", { keep }).Get()));
        auto leaving = graph->AttachLeafNode(graph->GetRootNode(),
                                             MAKE_RC_OBJ_PTR(MeshInstance, make_mesh("l", { leave }).Get()));
        GpuScene gpu(nullptr, test_desc());
        Mirror mirror;
        graph->Refresh(1);
        gpu.Update(graph);
        gpu.UploadTo(&mirror);
        CHECK(mirror.constantsOf[leave.Get()] == 1 && !leave->dirty);

        graph->Detach(leaving);
        graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, make_mesh("t", { late }).Get()));
        graph->Attach(graph->GetRootNode(), leaving);   // leave comes back after late took slot 1
        graph->Refresh(2);
        CHECK(leave->materialID == 2 && !leave->dirty);
        mirror.ResetCounters();
        gpu.Update(graph);
        gpu.UploadTo(&mirror);
        CHECK(mirror.constantsOf[leave.Get()] == 2 && mirror.constantsOf[late.Get()] == 1);
    }
}

namespace
{
    // The images and buffers of two GpuScenes over one graph agree for every live instance, mesh and material.
    void compare_scenes(SceneGraph* graph, const GpuScene& a, const Mirror& ma, const GpuScene& b, const Mirror& mb)
    {
        CHECK(ma.ok && mb.ok);
        for (MeshInstance* instance : instances_in_order(graph))
        {
            const uint32_t slot = uint32_t(instance->GetInstanceIndex());
            CHECK(std::memcmp(&a.GetInstanceRows()[slot], &b.GetInstanceRows()[slot], sizeof(InstanceData)) == 0);
            CHECK(std::memcmp(&ma.Instances()[slot], &mb.Instances()[slot], sizeof(InstanceData)) == 0);
            CHECK(std::memcmp(&ma.Instances()[slot], &a.GetInstanceRows()[slot], sizeof(InstanceData)) == 0);
        }
        for (MeshInfo* mesh : graph->GetMeshes())
        {
            for (const auto& geometry : mesh->geometries)
            {
                const uint32_t slot = uint32_t(geometry->globalGeometryIndex);
                CHECK(std::memcmp(&ma.Geometries()[slot], &mb.Geometries()[slot], sizeof(GeometryData)) == 0);
            }
        }
        for (Material* material : graph->GetMaterials())
        {
            const uint32_t slot = uint32_t(material->materialID);
            CHECK(std::memcmp(&ma.Materials()[slot], &mb.Materials()[slot], sizeof(MaterialConstants)) == 0);
        }
        if (ma.constantsOf != mb.constantsOf)
        {
            for (const auto& [material, id] : ma.constantsOf)
                std::fprintf(stderr, "a: %s %d\n", material->name.c_str(), id);
            for (const auto& [material, id] : mb.constantsOf)
                std::fprintf(stderr, "b: %s %d\n", material->name.c_str(), id);
            CHECK(false);
        }
    }
}

// Second review pass: out of memory inside GpuScene::Update and in an upload's buffer creation, and a row copied
// through the world whose acquisition throws.
void test_out_of_memory_paths()
{
    // An allocation fails partway through Update (the k-th of the call, k varying by frame; image growth by one row
    // per slot makes every frame allocate); the retry, and an upload whose buffer creation fails once and is retried,
    // leave the images and buffers equal to those of a GpuScene that never failed.
    {
        Rng rng{ 77 };
        auto stone = make_material("stone", MaterialDomain::Opaque);
        auto glass = make_material("glass", MaterialDomain::AlphaBlended);
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes = { make_mesh("a", { stone }), make_mesh("b", { stone, glass }),
                                                         make_mesh("c", { glass, glass, stone }) };
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        auto attach = [&]()
        {
            auto node = make_node("n");
            node->SetTranslation(dm::double3(rng.Unit(), rng.Unit(), rng.Unit()));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, meshes[rng.Below(3)].Get()));
            nodes.push_back(graph->Attach(graph->GetRootNode(), node));
        };
        for (int i = 0; i < 40; ++i)
            attach();

        GpuSceneDesc desc = test_desc(1);   // every new slot grows the images
        GpuScene reference(nullptr, desc);
        GpuScene failing(nullptr, desc);
        Mirror referenceMirror, failingMirror;
        uint32_t updateFailures = 0, uploadFailures = 0;
        for (uint32_t frame = 1; frame <= 30; ++frame)
        {
            for (uint32_t k = 0; k < 6; ++k)
                nodes[rng.Below(uint32_t(nodes.size()))]->SetTranslation(dm::double3(rng.Unit(), 2.0, rng.Unit()));
            for (uint32_t k = rng.Below(4); k > 0; --k)
                attach();
            if (frame % 3 == 0 && nodes.size() > 10)
            {
                const uint32_t index = rng.Below(uint32_t(nodes.size()));
                graph->Detach(nodes[index]);
                nodes.erase(nodes.begin() + index);
            }
            if (frame % 4 == 0)
            {
                glass->roughness = float(rng.Unit());
                glass->dirty = true;
            }
            graph->Refresh(frame);

            // both update before either uploads: an upload clears the materials' dirty flags, which both read
            CHECK(NVRHI_SUCCEEDED(reference.Update(graph)));

            // Not while the first update makes the GpuScene's state: in an MSVC Debug build a container's noexcept
            // default constructor allocates its iterator proxy, and a failure there terminates whatever the caller does.
            const bool armed = frame > 1;
            g_FailNewAfter.store(armed ? int64_t(frame % 5) : -1);
            const nvrhi::FRESULT result = failing.Update(graph);
            const bool failed = g_FailNewAfter.exchange(-1) < 0 && armed;   // the countdown fired
            if (failed)
            {
                CHECK(result == nvrhi::FE_OUT_OF_MEMORY);
                ++updateFailures;
                CHECK(NVRHI_SUCCEEDED(failing.Update(graph)));
            }
            else
                CHECK(NVRHI_SUCCEEDED(result));
            if (frame % 4 == 1)
            {
                failingMirror.failResizes = 1;
                const nvrhi::FRESULT upload = failing.UploadTo(&failingMirror);
                if (failingMirror.failResizes == 0)
                {
                    CHECK(upload == nvrhi::FE_OUT_OF_MEMORY);
                    ++uploadFailures;
                }
                failingMirror.failResizes = 0;
            }
            CHECK(NVRHI_SUCCEEDED(failing.UploadTo(&failingMirror)));
            CHECK(NVRHI_SUCCEEDED(reference.UploadTo(&referenceMirror)));
            compare_scenes(graph, reference, referenceMirror, failing, failingMirror);
        }
        std::printf("out of memory: %u failed updates, %u failed buffer creations, all recovered\n", updateFailures,
                    uploadFailures);
        CHECK(updateFailures >= 10 && uploadFailures >= 3);
    }

    // Without image growth, every allocation of the update in turn: a fresh GpuScene's second update starts with empty
    // run and settle lists, so its allocations interleave with the rows it writes into the image (a row's settle entry,
    // the row, its run). Failing the k-th of them, for every k, also strikes after a row changed in the image but before
    // it was marked; the retry must still upload it.
    {
        auto material = make_material("m", MaterialDomain::Opaque);
        auto mesh = make_mesh("mesh", { material });
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        for (int i = 0; i < 512; ++i)
        {
            auto node = make_node("n");
            node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
            nodes.push_back(graph->Attach(graph->GetRootNode(), node));
        }
        uint32_t frame = 0;
        graph->Refresh(++frame);
        uint32_t failures = 0;
        for (int64_t k = 0; k < 24; ++k)
        {
            GpuScene reference(nullptr, test_desc(1024));
            GpuScene failing(nullptr, test_desc(1024));
            Mirror referenceMirror, failingMirror;
            CHECK(NVRHI_SUCCEEDED(reference.Update(graph)) && NVRHI_SUCCEEDED(failing.Update(graph)));
            CHECK(NVRHI_SUCCEEDED(reference.UploadTo(&referenceMirror)) &&
                  NVRHI_SUCCEEDED(failing.UploadTo(&failingMirror)));
            for (int i = 0; i < 512; i += 4)   // rows apart: one run each
                nodes[i]->SetTranslation(dm::double3(double(i), double(frame), 1.0));
            graph->Refresh(++frame);
            CHECK(NVRHI_SUCCEEDED(reference.Update(graph)));
            g_FailNewAfter.store(k);
            const nvrhi::FRESULT result = failing.Update(graph);
            if (g_FailNewAfter.exchange(-1) < 0)
            {
                CHECK(result == nvrhi::FE_OUT_OF_MEMORY);
                ++failures;
                CHECK(NVRHI_SUCCEEDED(failing.Update(graph)));
            }
            CHECK(NVRHI_SUCCEEDED(failing.UploadTo(&failingMirror)));
            CHECK(NVRHI_SUCCEEDED(reference.UploadTo(&referenceMirror)));
            compare_scenes(graph, reference, referenceMirror, failing, failingMirror);
        }
        CHECK(failures == 24);
    }

    // A row copied through the world (Instantiate) arrives with its source's MeshRef; when its acquisition throws, the
    // copy holds no reference, so destroying it releases none of the source's.
    {
        auto material = make_material("m", MaterialDomain::Opaque);
        auto mesh = make_mesh("mesh", { material });
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(make_node("root"));
        auto node = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        IWorld* world = graph->GetWorld();
        const Entity source = graph->GetEntity(node);
        CHECK(GetComponent<MeshRef>(world, source)->mesh != MeshId::Invalid);

        size_t warnings = 0;
        log::SetCallback([&](log::Severity, const char*) { ++warnings; });
        SceneGraphStore::FailAcquireAfter(0);
        Entity copy = Entity::Null;
        CHECK(world->Instantiate(source, 1, &copy) == nvrhi::FS_OK);
        SceneGraphStore::FailAcquireAfter(-1);
        log::ResetCallback();
        CHECK(warnings == 1);
        CHECK(GetComponent<MeshRef>(world, copy)->mesh == MeshId::Invalid);
        CHECK(GetComponent<GpuInstanceSlot>(world, copy)->instance == c_NoInstanceSlot);

        CHECK(world->DestroyEntities(&copy, 1) == nvrhi::FS_OK);
        CHECK(graph->GetMeshes().size() == 1 && graph->GetMeshes().Find(mesh) != MeshId::Invalid);
        CHECK(graph->GetMaterials().size() == 1);
        CHECK(static_cast<MeshInstance*>(node->GetLeaf())->GetInstanceIndex() == 0);
    }
}

// ------------------------------------------------------------------------------------- scatter mode (plan step E5)

namespace
{
    GpuSceneDesc scatter_desc(uint32_t granularity = 64)
    {
        GpuSceneDesc desc = test_desc(granularity);
        desc.uploadMode = GpuSceneUploadMode::Scatter;
        return desc;
    }

    // The scatter kernel (gpu_scene_scatter_cs.hlsl) on the CPU, over a byte copy of the instance buffer, and the
    // upload pass's growth copy.
    struct ScatterModel
    {
        std::vector<uint8_t> instances;
        bool ok = true;   // every record named a slot inside the buffer, and no slot twice in one update

        uint8_t* Row(uint32_t slot) { return instances.data() + size_t(slot) * sizeof(InstanceData); }
        size_t Rows() const { return instances.size() / sizeof(InstanceData); }

        // A buffer CreateBuffers replaced: the new one is zero (a fresh D3D12 committed buffer; the GPU checks compare
        // only rows written since), and the growth use copies the old rows in when the frame executes.
        void Grow(const GpuSceneBufferChanges& changes, uint32_t rows, bool executed)
        {
            if (!changes.instancesResized)
                return;
            std::vector<uint8_t> next(size_t(rows) * sizeof(InstanceData), 0);
            if (executed && changes.previousInstanceRows)
            {
                const size_t bytes = std::min(instances.size(), size_t(changes.previousInstanceRows) * sizeof(InstanceData));
                std::memcpy(next.data(), instances.data(), bytes);
            }
            instances.swap(next);
        }

        void Apply(const GpuSceneScatterRecords& records)
        {
            const uint32_t transformAt = GPU_SCENE_ROW_TRANSFORM_WORD * 4;
            const uint32_t prevAt = GPU_SCENE_ROW_PREV_TRANSFORM_WORD * 4;
            const uint32_t transformBytes = GPU_SCENE_TRANSFORM_WORDS * 4;
            std::vector<uint8_t> seen(Rows(), 0);
            auto claim = [&](uint32_t slot)
            {
                if (slot >= Rows() || seen[slot])
                {
                    ok = false;
                    return false;
                }
                seen[slot] = 1;
                return true;
            };
            for (uint32_t i = 0; i < records.settleCount; ++i)
            {
                const uint32_t slot = records.settleSlots[i];
                if (claim(slot))
                    std::memcpy(Row(slot) + prevAt, Row(slot) + transformAt, transformBytes);
            }
            for (uint32_t i = 0; i < records.transformCount; ++i)
            {
                const GpuSceneTransformRecord& record = records.transforms[i];
                if (!claim(record.slot))
                    continue;
                std::memcpy(Row(record.slot) + prevAt, Row(record.slot) + transformAt, transformBytes);
                std::memcpy(Row(record.slot) + transformAt, record.transform, transformBytes);
            }
            for (uint32_t i = 0; i < records.rowCount; ++i)
            {
                const GpuSceneRowRecord& record = records.rows[i];
                if (!claim(record.slot))
                    continue;
                uint8_t* row = Row(record.slot);
                if (record.flags & GPU_SCENE_ROW_FLAG_KEEP_PREVIOUS)
                    std::memcpy(row + prevAt, row + transformAt, transformBytes);
                else
                    std::memcpy(row + prevAt, reinterpret_cast<const uint8_t*>(record.row) + prevAt, transformBytes);
                std::memcpy(row, record.row, prevAt);   // the head and the transform
            }
        }
    };

    // A scatter-mode GpuScene driven as the render service and the upload pass drive it, without a device.
    struct ScatterScene
    {
        GpuScene gpu;
        Mirror assets;        // the geometry and material buffers, written by UploadAssetsTo
        ScatterModel model;   // the instance buffer, written by the kernel model

        explicit ScatterScene(const GpuSceneDesc& desc) : gpu(nullptr, desc) { }

        // Update at extraction; CreateBuffers; then, when the frame executes, the growth copy, Prepare's asset upload,
        // the scatter and its acknowledgement.
        void Update(SceneGraph* graph)
        {
            CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)));
            CHECK(NVRHI_SUCCEEDED(gpu.CreateBuffers()));
        }
        void Execute(bool executed = true)
        {
            model.Grow(gpu.GetBufferChanges(), gpu.GetInstanceRowCount(), executed);
            if (!executed)
                return;
            CHECK(NVRHI_SUCCEEDED(gpu.UploadAssetsTo(&assets)));
            GpuSceneScatterRecords records;
            gpu.GetScatterRecords(records);
            model.Apply(records);
            gpu.CommitScatter(records.serial);
            CHECK(!gpu.IsScatterPending());
        }
    };

    // The scatter buffers against CPU mode's: the instance rows (all of them, or the live ones after a dropped frame,
    // with prevTransform = transform in the frame of the resend), the geometry and material buffers, and the material
    // constant buffers.
    enum class RowsCompared { All, Live, LiveResent };

    void compare_with_cpu_mode(SceneGraph* graph, const Mirror& cpu, const ScatterScene& scatter, RowsCompared rows,
                               const char* where)
    {
        CHECK(cpu.ok && scatter.assets.ok && scatter.model.ok);
        CHECK(scatter.model.instances.size() == cpu.buffers[0].size());
        if (rows == RowsCompared::All)
        {
            if (scatter.model.instances != cpu.buffers[0])
            {
                for (size_t slot = 0; slot < cpu.InstanceRows(); ++slot)
                {
                    if (std::memcmp(scatter.model.instances.data() + slot * sizeof(InstanceData),
                                    cpu.buffers[0].data() + slot * sizeof(InstanceData), sizeof(InstanceData)) != 0)
                        std::fprintf(stderr, "%s: instance slot %zu differs\n", where, slot);
                }
                CHECK(false);
            }
        }
        else
        {
            for (MeshInstance* instance : instances_in_order(graph))
            {
                const size_t slot = size_t(instance->GetInstanceIndex());
                InstanceData expected;
                std::memcpy(&expected, cpu.buffers[0].data() + slot * sizeof(InstanceData), sizeof(expected));
                if (rows == RowsCompared::LiveResent)
                    expected.prevTransform = expected.transform;
                if (std::memcmp(scatter.model.instances.data() + slot * sizeof(InstanceData), &expected,
                                sizeof(expected)) != 0)
                {
                    std::fprintf(stderr, "%s: live instance slot %zu differs\n", where, slot);
                    CHECK(false);
                }
            }
        }
        CHECK(scatter.assets.buffers[1] == cpu.buffers[1]);
        CHECK(scatter.assets.buffers[2] == cpu.buffers[2]);
        CHECK(scatter.assets.constantsOf == cpu.constantsOf);
    }
}

// Scatter mode against CPU mode over the seeded edit scripts: after every frame the model of the kernel applied to a
// mirror equals CPU mode's buffers byte for byte, growth across the 64-row granularity and the settle frames included;
// the records name each slot once. Then the same scripts with two frames dropped (not executed, not acknowledged): the
// next update is a resend, after which the live rows equal CPU mode's (prevTransform = transform in the resend's frame).
void test_scatter_matches_cpu_mode()
{
    for (uint64_t seed : { 11ull, 23ull, 0x5bd1e995ull })
    {
        for (bool drops : { false, true })
        {
            GpuScene cpu(nullptr, test_desc());
            Mirror cpuMirror;
            ScatterScene scatter(scatter_desc());
            uint32_t transformRecords = 0, keepRecords = 0, settleRecords = 0, resends = 0, growths = 0;
            RowsCompared rows = RowsCompared::All;
            run_edit_script(seed, 30, [&](SceneGraph* graph, uint32_t frame)
            {
                // both updates before either upload: they share the materials' dirty flags
                CHECK(NVRHI_SUCCEEDED(cpu.Update(graph)));
                scatter.Update(graph);
                CHECK(NVRHI_SUCCEEDED(cpu.UploadTo(&cpuMirror)));
                GpuSceneScatterRecords records;
                scatter.gpu.GetScatterRecords(records);
                CHECK(records.resend == (frame == 1 || (drops && (frame == 11 || frame == 21))));
                resends += records.resend ? 1 : 0;
                growths += scatter.gpu.GetBufferChanges().instancesResized ? 1 : 0;
                transformRecords += records.transformCount;
                settleRecords += records.settleCount;
                for (uint32_t i = 0; i < records.rowCount; ++i)
                    keepRecords += (records.rows[i].flags & GPU_SCENE_ROW_FLAG_KEEP_PREVIOUS) ? 1 : 0;
                const bool dropped = drops && (frame == 10 || frame == 20);
                scatter.Execute(!dropped);
                if (dropped)
                {
                    CHECK(scatter.gpu.IsScatterPending());
                    rows = RowsCompared::Live;
                    return;
                }
                if (records.resend && frame != 1)
                    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::LiveResent, "scatter resend");
                else
                    compare_with_cpu_mode(graph, cpuMirror, scatter, rows, drops ? "scatter, drops" : "scatter");
            });
            std::printf("scatter seed %llx%s: %u transform, %u keep-previous row and %u settle records; %u resends, "
                        "%u growths\n", (unsigned long long)seed, drops ? " (two frames dropped)" : "", transformRecords,
                        keepRecords, settleRecords, resends, growths);
            CHECK(transformRecords > 0 && settleRecords > 0 && growths > 1);
            CHECK(resends == (drops ? 3u : 1u));
        }
    }
}

// The records follow what changed (4,096 instances, granularity 1,024): a resend first, nothing when idle, one
// transform record for one move and its settle record in the next frame, only the moved rows of changed chunks, a
// keep-previous row record for SetLeaf; growth past 4,096 rows with the old rows to copy; CommitScatter, a dropped
// frame and the resend after it; Upload in scatter mode refused; the asset rows through UploadAssetsTo.
void test_scatter_records()
{
    auto material = make_material("m", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { material });
    auto other = make_mesh("other", { material, material });
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    auto attach = [&](int count)
    {
        for (int i = 0; i < count; ++i)
        {
            auto node = make_node("n");
            node->SetTranslation(dm::double3(double(nodes.size()), 0.0, 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
            nodes.push_back(graph->Attach(graph->GetRootNode(), node));
        }
    };
    attach(4096);

    GpuScene cpu(nullptr, test_desc(1024));
    Mirror cpuMirror;
    ScatterScene scatter(scatter_desc(1024));
    uint32_t frame = 0;
    GpuSceneScatterRecords records;
    auto step = [&](bool executed = true)
    {
        graph->Refresh(++frame);
        CHECK(NVRHI_SUCCEEDED(cpu.Update(graph)));
        scatter.Update(graph);
        CHECK(NVRHI_SUCCEEDED(cpu.UploadTo(&cpuMirror)));
        scatter.gpu.GetScatterRecords(records);
        scatter.Execute(executed);
        return scatter.gpu.GetStats();
    };

    GpuSceneStats stats = step();
    CHECK(records.resend && records.rowCount == 4096 && records.transformCount == 0 && records.settleCount == 0);
    CHECK(scatter.gpu.GetBufferChanges().instancesResized && !scatter.gpu.GetBufferChanges().previousInstanceRows);
    CHECK(scatter.gpu.GetBufferChanges().geometriesResized && scatter.gpu.GetBufferChanges().materialsResized);
    CHECK(stats.rowRecords == 4096 && stats.uploadBytes >= 4096 * sizeof(GpuSceneRowRecord));
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "records 1");
    CHECK(scatter.gpu.Upload(nullptr) == nvrhi::FE_INVALID_ARGS);
    CHECK(scatter.gpu.UploadTo(&cpuMirror) == nvrhi::FE_UNSUPPORTED);
    CHECK(cpu.UploadAssetsTo(&cpuMirror) == nvrhi::FE_UNSUPPORTED);

    stats = step();   // idle
    CHECK(!records.resend && records.rowCount + records.transformCount + records.settleCount == 0);
    CHECK(stats.chunksChanged == 0 && !scatter.gpu.GetBufferChanges().instancesResized);
    const uint32_t chunks = stats.chunksVisited;

    nodes[100]->SetTranslation(dm::double3(1.0, 2.0, 3.0));
    stats = step();
    CHECK(stats.chunksChanged == 1 && records.transformCount == 1 && records.rowCount == 0 && records.settleCount == 0);
    CHECK(records.transforms[0].slot == 100 && stats.uploadBytes == sizeof(GpuSceneTransformRecord));
    CHECK(stats.rowsBuilt < 4096 / 2);
    stats = step();   // its previous transform settles
    CHECK(records.settleCount == 1 && records.settleSlots[0] == 100 && records.transformCount == 0);
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "records settle");
    stats = step();
    CHECK(records.settleCount + records.transformCount + records.rowCount == 0);

    // every 16th row: every chunk is a candidate, only the moved rows are records (in slot order: chunk order)
    for (int i = 0; i < 4096; i += 16)
        nodes[i]->SetTranslation(dm::double3(double(i), 2.0, 0.0));
    stats = step();
    CHECK(stats.chunksChanged == chunks && stats.rowsBuilt == 4096 && records.transformCount == 256);
    for (uint32_t i = 0; i < records.transformCount; ++i)
        CHECK(records.transforms[i].slot == i * 16);
    stats = step();
    CHECK(records.settleCount == 256 && records.transformCount == 0);
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "records strided");

    // SetLeaf to another mesh: the same entity keeps its slot, its head changes: a keep-previous row record
    nodes[7]->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, other.Get()));
    nodes[7]->SetTranslation(dm::double3(7.0, 5.0, 0.0));
    stats = step();
    CHECK(records.rowCount == 1 && records.rows[0].slot == 7);
    std::printf("scatter records: SetLeaf gives a row record with flags %u\n", records.rows[0].flags);
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "records SetLeaf");
    step();
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "records SetLeaf settle");

    // growth past 4,096 rows: the old rows are copied, the new instances are row records
    attach(100);
    stats = step();
    const GpuSceneBufferChanges& changes = scatter.gpu.GetBufferChanges();
    CHECK(changes.instancesResized && changes.previousInstanceRows == 4096 && scatter.gpu.GetInstanceRowCount() == 5120);
    CHECK(changes.previousInstanceBuffer == nullptr);   // no device: the bookkeeping only
    CHECK(records.rowCount == 100 && !records.resend && stats.buffersResized == 1);
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "records growth");
    step();
    CHECK(!scatter.gpu.GetBufferChanges().instancesResized);

    // CommitScatter takes only the last serial
    nodes[5]->SetTranslation(dm::double3(5.0, 9.0, 0.0));
    graph->Refresh(++frame);
    CHECK(NVRHI_SUCCEEDED(cpu.Update(graph)));
    scatter.Update(graph);
    CHECK(NVRHI_SUCCEEDED(cpu.UploadTo(&cpuMirror)));
    scatter.gpu.GetScatterRecords(records);
    CHECK(scatter.gpu.IsScatterPending());
    scatter.gpu.CommitScatter(records.serial - 1);
    CHECK(scatter.gpu.IsScatterPending());
    scatter.Execute();
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "records commit");

    // a dropped frame that grows the buffer: the next update resends every live row, without settle records, and needs
    // no growth copy
    attach(1100);
    for (int i = 0; i < 4096; i += 3)
        nodes[i]->SetTranslation(dm::double3(double(i), 3.0, 1.0));
    step(false);
    CHECK(scatter.gpu.GetBufferChanges().instancesResized && scatter.gpu.IsScatterPending());
    for (int i = 0; i < 4096; i += 5)
        nodes[i]->SetTranslation(dm::double3(double(i), 4.0, 1.0));
    stats = step();
    CHECK(records.resend && records.rowCount == uint32_t(nodes.size()) && records.transformCount == 0 &&
          records.settleCount == 0);
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::LiveResent, "records resend");
    attach(2000);   // grows again: the copy is back
    stats = step();
    CHECK(!records.resend && scatter.gpu.GetBufferChanges().previousInstanceRows == 6144);
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::Live, "records after resend");

    // a material edit: its row and constant buffer through UploadAssetsTo
    material->roughness = 0.75f;
    material->dirty = true;
    graph->Refresh(++frame);
    CHECK(NVRHI_SUCCEEDED(cpu.Update(graph)));
    scatter.Update(graph);
    CHECK(scatter.gpu.HasPendingAssets());
    CHECK(NVRHI_SUCCEEDED(cpu.UploadTo(&cpuMirror)));
    scatter.assets.ResetCounters();
    scatter.Execute();
    CHECK(!scatter.gpu.HasPendingAssets() && scatter.assets.constantWrites == 1);
    CHECK(scatter.assets.bytes == sizeof(MaterialConstants));
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::Live, "records material");
}

// The records are in chunk order whatever the worker count: three equal graphs on job systems of 1, 2 and 8 workers
// give byte-equal records every frame, with moves spread over every chunk (the parallel path).
void test_scatter_workers()
{
    const uint32_t workerCounts[] = { 1, 2, 8 };
    auto material = make_material("m", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { material, material });
    struct Run
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<SceneGraph> graph;
        std::vector<SceneGraphNode*> nodes;
        nvrhi::AutoPtr<GpuScene> gpu;
    };
    Run runs[3];
    for (uint32_t r = 0; r < 3; ++r)
    {
        JobSystemDesc desc;
        desc.debugName = "test_scatter_workers";
        desc.workerThreadCount = workerCounts[r];
        CHECK(CreateJobSystem(desc, &runs[r].jobs) == nvrhi::FS_OK);
        runs[r].graph = MAKE_RC_OBJ_PTR(SceneGraph);
        runs[r].graph->SetJobSystem(runs[r].jobs);
        runs[r].graph->SetRootNode(make_node("root"));
        for (int i = 0; i < 20000; ++i)
        {
            auto node = make_node("n");
            node->SetTranslation(dm::double3(double(i % 100), double(i / 100), 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
            runs[r].nodes.push_back(runs[r].graph->Attach(runs[r].graph->GetRootNode(), node));
        }
        runs[r].gpu = MAKE_RC_OBJ_PTR(GpuScene, nullptr, scatter_desc(1024));
    }
    uint64_t recordsSeen = 0;
    for (uint32_t frame = 1; frame <= 8; ++frame)
    {
        GpuSceneScatterRecords records[3];
        for (uint32_t r = 0; r < 3; ++r)
        {
            Rng rng{ 1000 + frame };
            for (int k = 0; k < 2000; ++k)
                runs[r].nodes[rng.Below(20000)]->SetTranslation(dm::double3(rng.Unit(), rng.Unit(), double(frame)));
            runs[r].graph->Refresh(frame);
            CHECK(NVRHI_SUCCEEDED(runs[r].gpu->Update(runs[r].graph)));
            runs[r].gpu->GetScatterRecords(records[r]);
        }
        for (uint32_t r = 1; r < 3; ++r)
        {
            CHECK(records[r].settleCount == records[0].settleCount);
            CHECK(records[r].transformCount == records[0].transformCount);
            CHECK(records[r].rowCount == records[0].rowCount);
            CHECK(std::memcmp(records[r].settleSlots, records[0].settleSlots, records[0].settleCount * sizeof(uint32_t)) == 0);
            CHECK(std::memcmp(records[r].transforms, records[0].transforms,
                              records[0].transformCount * sizeof(GpuSceneTransformRecord)) == 0);
            CHECK(std::memcmp(records[r].rows, records[0].rows, records[0].rowCount * sizeof(GpuSceneRowRecord)) == 0);
        }
        recordsSeen += records[0].settleCount + records[0].transformCount + records[0].rowCount;
        for (uint32_t r = 0; r < 3; ++r)
            runs[r].gpu->CommitScatter(records[r].serial);
    }
    std::printf("scatter workers: %llu records, equal at 1, 2 and 8 workers\n", (unsigned long long)recordsSeen);
    for (Run& run : runs)
    {
        run.gpu = nullptr;
        run.nodes.clear();
        run.graph = nullptr;
    }
}

// Scatter mode allocates nothing per frame once its arrays have grown: Update (the parallel gather included),
// CreateBuffers, the records, CommitScatter and UploadAssetsTo.
void test_scatter_steady_state_allocations()
{
    auto material = make_material("m", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { material, material });
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    std::vector<SceneGraphNode*> nodes;
    for (int i = 0; i < 6000; ++i)
    {
        auto node = make_node("n");
        node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        nodes.push_back(graph->Attach(graph->GetRootNode(), node));
    }
    GpuScene gpu(nullptr, scatter_desc());
    Mirror assets;
    Rng rng{ 7 };
    uint32_t frame = 0;
    uint64_t records = 0;
    auto run = [&](uint32_t moves)
    {
        for (uint32_t k = 0; k < moves; ++k)
            nodes[rng.Below(6000)]->SetTranslation(dm::double3(rng.Unit(), rng.Unit(), rng.Unit()));
        graph->Refresh(++frame);
        gpu.Update(graph);
        gpu.CreateBuffers();
        gpu.UploadAssetsTo(&assets);
        GpuSceneScatterRecords out;
        gpu.GetScatterRecords(out);
        records += out.transformCount + out.settleCount + out.rowCount;
        gpu.CommitScatter(out.serial);
    };
    for (int i = 0; i < 6; ++i)
        run(1500);   // grows the arrays past what 600 moves need
    const uint64_t newBefore = g_NewCount.load();
    const uint64_t heapBefore = ecs::GetHeapAllocationCount();
    const uint64_t recordsBefore = records;
    for (int i = 0; i < 8; ++i)
        run(600);
    run(0);
    run(0);
    const uint64_t newCalls = g_NewCount.load() - newBefore;
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
    std::printf("scatter steady state: %llu operator new, %llu storage blocks, %llu records\n",
                (unsigned long long)newCalls, (unsigned long long)heapBlocks,
                (unsigned long long)(records - recordsBefore));
    CHECK(newCalls == 0 && heapBlocks == 0 && records > recordsBefore);
}

// Review fix (E5.4, finding 1): a frame whose Prepare ran (UploadAssetsTo, which clears the materials' dirty flags and
// pending constants) but that never executed (no CommitScatter; its writes went to a buffer that the GPU never saw):
// the next update is a resend that writes the geometry and material buffers whole and every material's constant buffer,
// so the buffers equal CPU mode's again.
void test_scatter_lost_prepare()
{
    auto stone = make_material("stone", MaterialDomain::Opaque);
    auto glass = make_material("glass", MaterialDomain::AlphaBlended);
    auto mesh = make_mesh("mesh", { stone, glass });
    auto other = make_mesh("other", { glass });
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    graph->SetRootNode(make_node("root"));
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 100; ++i)
    {
        auto node = make_node("n");
        node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        nodes.push_back(graph->Attach(graph->GetRootNode(), node));
    }
    GpuScene cpu(nullptr, test_desc());
    Mirror cpuMirror;
    ScatterScene scatter(scatter_desc());
    uint32_t frame = 0;
    auto update = [&]()
    {
        graph->Refresh(++frame);
        CHECK(NVRHI_SUCCEEDED(cpu.Update(graph)));
        scatter.Update(graph);
        CHECK(NVRHI_SUCCEEDED(cpu.UploadTo(&cpuMirror)));
    };
    update();
    scatter.Execute();
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::All, "lost prepare 1");

    // the lost frame: a material edit, a new mesh and moves; Prepare writes to a buffer the GPU never sees
    stone->roughness = 0.125f;
    stone->dirty = true;
    graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, other.Get()));
    nodes[3]->SetTranslation(dm::double3(3.0, 1.0, 0.0));
    update();
    CHECK(scatter.gpu.HasPendingAssets());
    Mirror lost = scatter.assets;
    CHECK(NVRHI_SUCCEEDED(scatter.gpu.UploadAssetsTo(&lost)));
    CHECK(!stone->dirty && !scatter.gpu.HasPendingAssets() && scatter.gpu.IsScatterPending());
    scatter.model.Grow(scatter.gpu.GetBufferChanges(), scatter.gpu.GetInstanceRowCount(), false);

    // the next frame resends: every material's constant buffer, the asset buffers whole, every live row
    scatter.assets.ResetCounters();
    scatter.assets.constantsOf.clear();
    update();
    GpuSceneScatterRecords records;
    scatter.gpu.GetScatterRecords(records);
    CHECK(records.resend && scatter.gpu.HasPendingAssets());
    scatter.Execute();
    CHECK(scatter.assets.constantWrites == uint32_t(graph->GetMaterials().size()));
    CHECK(scatter.assets.constantsOf.size() == graph->GetMaterials().size());
    CHECK(scatter.assets.constantsOf[stone.Get()] == stone->materialID);
    CHECK(scatter.assets.bytes >= uint64_t(scatter.gpu.GetGeometryRowCount()) * sizeof(GeometryData) +
                                  uint64_t(scatter.gpu.GetMaterialRowCount()) * sizeof(MaterialConstants));
    CHECK(scatter.assets.buffers[1] == cpuMirror.buffers[1] && scatter.assets.buffers[2] == cpuMirror.buffers[2]);
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::LiveResent, "lost prepare resend");
    update();
    scatter.Execute();
    compare_with_cpu_mode(graph, cpuMirror, scatter, RowsCompared::Live, "lost prepare after");
}

// RowTracked scene columns (plan step E7, item E7.5) behind SceneGraph::SetRowTrackedTransforms, off by default: a
// graph made while it is on registers LocalTransform and WorldTransform RowTracked; then a node's move stamps its row,
// propagation writes and stamps only that row's WorldTransform, and GpuScene builds only that row of the chunk.
void test_row_tracked_transforms()
{
    // Stage T1: on in the switches-on runs (tests/src/e7_switches.cpp); the value found is put back at the end.
    const bool rowTrackedFound = SceneGraph::GetRowTrackedTransforms();
    CHECK(rowTrackedFound == donut::tests::E7SwitchRequested("rowtracked"));
    auto material = make_material("m", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { material });
    for (const bool tracked : { false, true })
    {
        SceneGraph::SetRowTrackedTransforms(tracked);
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        SceneGraph::SetRowTrackedTransforms(false);   // the graph keeps what it was made with
        IWorld* world = graph->GetWorld();
        for (const ComponentId component : { FindComponent<LocalTransform>(world), FindComponent<WorldTransform>(world) })
            CHECK(!!(world->GetComponentTypeDesc(component)->flags & ComponentFlags::RowTracked) == tracked);
        CHECK(!(world->GetComponentTypeDesc(FindComponent<LocalMatrix>(world))->flags & ComponentFlags::RowTracked));

        graph->SetRootNode(make_node("root"));
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        for (int i = 0; i < 64; ++i)
        {
            auto node = make_node("n");
            node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
            nodes.push_back(graph->Attach(graph->GetRootNode(), node));
        }
        GpuScene gpu(nullptr, test_desc());
        Mirror mirror;
        PrevModel model;
        uint32_t frame = 0;
        auto step = [&]()
        {
            graph->Refresh(++frame);
            CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)));
            CHECK(NVRHI_SUCCEEDED(gpu.UploadTo(&mirror)));
            verify_frame(graph, gpu, mirror, model, "row tracked");
            return gpu.GetStats();
        };
        step();
        step();
        nodes[10]->SetTranslation(dm::double3(10.0, 1.0, 0.0));
        GpuSceneStats stats = step();
        CHECK(stats.chunksChanged == 1 && stats.rowsChanged == 1);
        CHECK(tracked ? stats.rowsBuilt == 1 : stats.rowsBuilt == 64);   // one chunk of 64 rows
        stats = step();   // its previous transform settles
        CHECK(stats.rowsSettled == 1 && stats.rowsChanged == 0);
    }
    SceneGraph::SetRowTrackedTransforms(rowTrackedFound);
}

namespace
{
    uint64_t fnv_bytes(uint64_t hash, const void* data, size_t size)
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= bytes[i];
            hash *= 0x100000001b3ull;
        }
        return hash;
    }

    // One frame of a run of the edit script, as the A/B of RowTracked compares it.
    struct TrackedFrame
    {
        uint64_t worlds = 0;        // every attached node's world transform and box, walker order, bit for bit
        uint64_t worldHash = 0;     // IWorld::ComputeHash (row versions are not part of it)
        uint64_t buffers[3] = {};   // the CPU mode mirror's instance, geometry and material buffers
        uint32_t rowsSent = 0;      // rowsChanged + rowsSettled (backlog I4-R1: the split may differ)
        uint32_t rowsBuilt = 0;
        uint32_t scatterRecords = 0;
        std::vector<uint64_t> components;   // per component id: its values over every entity, archetype and row order
        std::pair<uint32_t, uint32_t> staleChunks;   // stale_chunk_bounds
        std::string staleReport;

        // Equal outputs; `exact`: ChunkWorldBounds and the world hash too (backlog T4-1: with the switch on, a chunk
        // box left larger than its rows' union by a removal was not recomputed where the untracked build recomposed the
        // chunk by chance, so the box, and the hash and snapshots that hold it, differed; fixed in F8: Bounds takes a
        // chunk that lost rows in both builds).
        bool Equals(const TrackedFrame& o, bool exact, size_t chunkBoundsId) const
        {
            if (worlds != o.worlds || buffers[0] != o.buffers[0] || buffers[1] != o.buffers[1] ||
                buffers[2] != o.buffers[2] || rowsSent != o.rowsSent || components.size() != o.components.size())
                return false;
            for (size_t c = 0; c < components.size(); ++c)
                if (components[c] != o.components[c] && (exact || c != chunkBoundsId))
                    return false;
            return !exact || (worldHash == o.worldHash && staleChunks == o.staleChunks);
        }
    };

    // Per component id: the bytes of every entity's value (GetComponent: a Chunk component's chunk value, a Shared
    // one's value) and enabled bit, entities in archetype, chunk and row order; for telling which component makes
    // ComputeHash differ.
    std::vector<uint64_t> component_hashes(IWorld* world)
    {
        std::vector<uint64_t> hashes(world->GetComponentTypeCount(), 0xcbf29ce484222325ull);
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            const std::vector<ComponentId> components(info.components, info.components + info.componentCount);
            for (uint32_t c = 0; c < info.chunkCount; ++c)
            {
                ChunkInfo chunk{};
                CHECK(world->GetChunkInfo(a, c, chunk) == nvrhi::FS_OK);
                const std::vector<Entity> entities(chunk.entities, chunk.entities + chunk.count);
                for (const Entity entity : entities)
                {
                    for (const ComponentId component : components)
                    {
                        const ComponentTypeDesc* desc = world->GetComponentTypeDesc(component);
                        uint64_t& hash = hashes[uint32_t(component)];
                        hash = fnv_bytes(hash, &entity, sizeof(entity));
                        // the field bytes only: components without fields hold pointers (SceneNode, LeafObject)
                        if (const void* value = desc->size && desc->fieldCount ? world->GetComponent(entity, component)
                                                                               : nullptr)
                            hash = fnv_bytes(hash, value, desc->size);
                        if (!!(desc->flags & ComponentFlags::Enableable))
                        {
                            const bool enabled = world->IsEnabled(entity, component);
                            hash = fnv_bytes(hash, &enabled, sizeof(enabled));
                        }
                    }
                }
            }
        }
        return hashes;
    }

    // The chunks whose ChunkWorldBounds is not the union of their rows' WorldBounds (stale), and of those the ones whose
    // box does not even contain the union (it would cull a visible row).
    std::pair<uint32_t, uint32_t> stale_chunk_bounds(IWorld* world, std::string* report = nullptr)
    {
        const ComponentId chunkBoundsId = FindComponent<ChunkWorldBounds>(world);
        const ComponentId worldBoundsId = FindComponent<WorldBounds>(world);
        uint32_t stale = 0, missing = 0;
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            if (std::find(info.components, info.components + info.componentCount, chunkBoundsId) ==
                info.components + info.componentCount)
                continue;
            for (uint32_t c = 0; c < info.chunkCount; ++c)
            {
                ChunkInfo chunk{};
                CHECK(world->GetChunkInfo(a, c, chunk) == nvrhi::FS_OK);
                if (chunk.count == 0)
                    continue;
                dm::box3 rows = dm::box3::empty();
                for (uint32_t r = 0; r < chunk.count; ++r)
                    rows |= static_cast<const WorldBounds*>(
                        world->GetComponent(chunk.entities[r], worldBoundsId))->bounds;
                const dm::box3 box =
                    static_cast<const ChunkWorldBounds*>(world->GetComponent(chunk.entities[0], chunkBoundsId))->bounds;
                if (std::memcmp(&box, &rows, sizeof(box)) != 0)
                {
                    if (report)
                    {
                        char line[512];
                        std::snprintf(line, sizeof(line), "    archetype %u chunk %u: %u of %u rows; box (%g %g %g)-"
                                      "(%g %g %g), rows (%g %g %g)-(%g %g %g); versions", a, c, chunk.count,
                                      chunk.capacity,
                                      box.m_mins.x, box.m_mins.y, box.m_mins.z, box.m_maxs.x, box.m_maxs.y, box.m_maxs.z,
                                      rows.m_mins.x, rows.m_mins.y, rows.m_mins.z, rows.m_maxs.x, rows.m_maxs.y,
                                      rows.m_maxs.z);
                        *report += line;
                        for (uint32_t k = 0; k < info.componentCount; ++k)
                        {
                            const char* name = world->GetComponentTypeDesc(info.components[k])->name;
                            std::snprintf(line, sizeof(line), " %s=%llu", name ? name : "?",
                                          (unsigned long long)chunk.versions[k]);
                            *report += line;
                        }
                        std::snprintf(line, sizeof(line), " added=%llu world=%llu\n",
                                      (unsigned long long)chunk.addedVersion, (unsigned long long)world->GetVersion());
                        *report += line;
                    }
                    ++stale;
                    missing += !rows.isempty() && !(box.contains(rows.m_mins) && box.contains(rows.m_maxs)) ? 1 : 0;
                }
            }
        }
        return { stale, missing };
    }

    struct ArchetypeLayout
    {
        uint32_t componentCount, entityCount, chunkCount, chunkSize, rowSize, rowsPerChunk;
        uint64_t components;
        bool operator==(const ArchetypeLayout& o) const
        {
            return componentCount == o.componentCount && entityCount == o.entityCount && chunkCount == o.chunkCount &&
                   chunkSize == o.chunkSize && rowSize == o.rowSize && rowsPerChunk == o.rowsPerChunk &&
                   components == o.components;
        }
    };

    std::vector<ArchetypeLayout> layouts_of(IWorld* world)
    {
        std::vector<ArchetypeLayout> result;
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            result.push_back(ArchetypeLayout{ info.componentCount, info.entityCount, info.chunkCount, info.chunkSize,
                                              info.rowSize, info.rowsPerChunk,
                                              fnv_bytes(0xcbf29ce484222325ull, info.components,
                                                        sizeof(ComponentId) * info.componentCount) });
        }
        return result;
    }
}

// RowTracked against the untracked build (stage T4; revision 1, E7.5's tests): the seeded edit script (moves of nodes
// and parents, attaches, detaches, re-attaches, SetLeaf, InvalidateContent, material and mesh edits, skinned
// instances, lights, curves) run twice per seed, the graph made with SceneGraph::SetRowTrackedTransforms off, then on.
// Every frame: every node's world transform and box bit-equal, every component's field bytes over every entity equal
// (but ChunkWorldBounds), GpuScene's CPU-mode buffers (instance, geometry, material) byte-equal, and rowsChanged +
// rowsSettled equal (I4-R1: a row moved in the last frame may settle instead of being rebuilt with the switch on, so
// the split may differ); verify_frame holds in both. In the tracked run a scatter-mode GpuScene beside it equals CPU
// mode every frame (its records are per changed row there). At the end every archetype's layout (components, rows,
// chunks, chunk and row sizes, rows per chunk) is equal. `exact` (backlog T4-1, fixed in F8; the default run):
// ChunkWorldBounds, the chunks whose box is not their rows' union, and IWorld::ComputeHash equal as well.
void test_row_tracked_equals_untracked(bool exact)
{
    const bool rowTrackedFound = SceneGraph::GetRowTrackedTransforms();
    uint64_t builtTotals[2] = {}, sentTotals[2] = {}, frames = 0, staleTotals[2] = {}, missingTotals[2] = {};
    for (uint64_t seed : { 11ull, 23ull, 0x5bd1e995ull })
    {
        std::vector<TrackedFrame> runs[2];
        std::vector<ArchetypeLayout> layouts[2];
        std::vector<std::string> names;   // component names by id (both worlds register the same types)
        for (uint32_t tracked = 0; tracked < 2; ++tracked)
        {
            SceneGraph::SetRowTrackedTransforms(tracked != 0);   // read once, when run_edit_script makes its graph
            GpuScene gpu(nullptr, test_desc());
            Mirror mirror;
            PrevModel model;
            ScatterScene scatter(scatter_desc());
            bool checkedFlags = false;
            run_edit_script(seed, 40, [&](SceneGraph* graph, uint32_t)
            {
                IWorld* world = graph->GetWorld();
                if (!checkedFlags)
                {
                    for (const ComponentId c : { FindComponent<LocalTransform>(world),
                                                 FindComponent<WorldTransform>(world) })
                        CHECK(!!(world->GetComponentTypeDesc(c)->flags & ComponentFlags::RowTracked) == (tracked != 0));
                    checkedFlags = true;
                }
                TrackedFrame frame;
                uint64_t worlds = 0xcbf29ce484222325ull;
                for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
                {
                    const dm::daffine3 transform = walker->GetLocalToWorldTransform();
                    const dm::box3 box = walker->GetGlobalBoundingBox();
                    worlds = fnv_bytes(worlds, &transform, sizeof(transform));
                    worlds = fnv_bytes(worlds, &box, sizeof(box));
                }
                frame.worlds = worlds;
                frame.worldHash = world->ComputeHash();
                frame.components = component_hashes(world);
                frame.staleChunks = stale_chunk_bounds(world, &frame.staleReport);
                for (uint32_t c = uint32_t(names.size()); c < world->GetComponentTypeCount(); ++c)
                {
                    const char* name = world->GetComponentTypeDesc(ComponentId(c))->name;
                    names.push_back(name ? name : "");
                }
                CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)));
                if (tracked)
                    scatter.Update(graph);   // before either upload: they share the materials' dirty flags
                const GpuSceneStats stats = gpu.GetStats();
                frame.rowsSent = stats.rowsChanged + stats.rowsSettled;
                frame.rowsBuilt = stats.rowsBuilt;
                CHECK(NVRHI_SUCCEEDED(gpu.UploadTo(&mirror)));
                for (uint32_t b = 0; b < 3; ++b)
                    frame.buffers[b] = fnv_bytes(0xcbf29ce484222325ull, mirror.buffers[b].data(),
                                                 mirror.buffers[b].size());
                verify_frame(graph, gpu, mirror, model, tracked ? "row tracked" : "untracked");
                if (tracked)
                {
                    GpuSceneScatterRecords records;
                    scatter.gpu.GetScatterRecords(records);
                    frame.scatterRecords = records.rowCount + records.transformCount + records.settleCount;
                    scatter.Execute();
                    compare_with_cpu_mode(graph, mirror, scatter, RowsCompared::All, "row tracked scatter");
                }
                runs[tracked].push_back(frame);
                layouts[tracked] = layouts_of(world);
            });
            CHECK(checkedFlags);
        }
        CHECK(runs[0].size() == runs[1].size() && runs[0].size() == 40);
        const size_t chunkBoundsId = size_t(std::find(names.begin(), names.end(), "ChunkWorldBounds") - names.begin());
        CHECK(chunkBoundsId < names.size());
        for (size_t f = 0; f < runs[0].size(); ++f)
        {
            if (!runs[0][f].Equals(runs[1][f], exact, chunkBoundsId))
            {
                const TrackedFrame& x = runs[0][f];
                const TrackedFrame& y = runs[1][f];
                std::fprintf(stderr, "seed %llx frame %zu: worlds %d, hash %d, buffers %d %d %d, rows sent %u / %u\n",
                             (unsigned long long)seed, f + 1, x.worlds == y.worlds, x.worldHash == y.worldHash,
                             x.buffers[0] == y.buffers[0], x.buffers[1] == y.buffers[1], x.buffers[2] == y.buffers[2],
                             x.rowsSent, y.rowsSent);
                std::fprintf(stderr, "  chunks with stale bounds %u / %u, of them not containing their rows %u / %u\n",
                             x.staleChunks.first, y.staleChunks.first, x.staleChunks.second, y.staleChunks.second);
                std::fprintf(stderr, "  untracked:\n%s  tracked:\n%s", x.staleReport.c_str(), y.staleReport.c_str());
                if (f > 0)
                    std::fprintf(stderr, "  the frame before, tracked:\n%s", runs[1][f - 1].staleReport.c_str());
                for (size_t c = 0; c < std::min(x.components.size(), y.components.size()); ++c)
                    if (x.components[c] != y.components[c])
                        std::fprintf(stderr, "  component %zu (%s) differs\n", c,
                                     c < names.size() ? names[c].c_str() : "?");
                CHECK(false);
            }
            for (uint32_t t = 0; t < 2; ++t)
            {
                builtTotals[t] += runs[t][f].rowsBuilt;
                sentTotals[t] += runs[t][f].rowsSent;
                staleTotals[t] += runs[t][f].staleChunks.first;
                missingTotals[t] += runs[t][f].staleChunks.second;
            }
            ++frames;
        }
        CHECK(layouts[0].size() == layouts[1].size() && layouts[0].size() > 4);
        for (size_t a = 0; a < layouts[0].size(); ++a)
            CHECK(layouts[0][a] == layouts[1][a]);
    }
    SceneGraph::SetRowTrackedTransforms(rowTrackedFound);
    std::printf("row tracked against untracked%s: %llu frames equal (worlds, components, buffers, rows sent %llu); "
                "rows built %llu untracked, %llu tracked\n", exact ? " (exact)" : "", (unsigned long long)frames,
                (unsigned long long)sentTotals[1], (unsigned long long)builtTotals[0],
                (unsigned long long)builtTotals[1]);
    std::printf("  chunk boxes larger than their rows' union, summed over the frames: %llu untracked, %llu tracked; "
                "boxes missing a row: %llu, %llu\n", (unsigned long long)staleTotals[0],
                (unsigned long long)staleTotals[1], (unsigned long long)missingTotals[0],
                (unsigned long long)missingTotals[1]);
    CHECK(builtTotals[1] <= builtTotals[0] && sentTotals[0] == sentTotals[1]);
    CHECK(missingTotals[0] == 0 && missingTotals[1] == 0);   // stale boxes are larger, never too small
}

// RowTracked cases that the T8 mutants of I4 showed missing (stage T8): with the switch on, (1) a chunk-only write of
// WorldTransform (a Write query that stamps no row) followed by one node's move in the same chunk: propagation spreads
// the chunk-only write before it stamps the moved row, so GpuScene still rebuilds every row; (2) a chunk whose head
// columns changed (RenderFlags rewritten by InvalidateContent after a mesh type edit) while its WorldTransform did not:
// GpuScene rebuilds it whole, not by row; (3) the deep bucket by row: one node of a chain moves at depth 66, and
// GpuScene rebuilds the 5 rows written (66..70) of the deep bucket's chunk, not all of them. verify_frame after each.
void test_row_tracked_killers()
{
    const bool found = SceneGraph::GetRowTrackedTransforms();
    SceneGraph::SetRowTrackedTransforms(true);
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    SceneGraph::SetRowTrackedTransforms(found);
    auto material = make_material("m", MaterialDomain::Opaque);
    auto mesh = make_mesh("mesh", { material });
    auto curve = make_mesh("curve", { material });
    graph->SetRootNode(make_node("root"));
    std::vector<SceneGraphNode*> nodes;
    for (int i = 0; i < 64; ++i)
    {
        auto node = make_node("n");
        node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, i == 20 ? curve.Get() : mesh.Get()));
        nodes.push_back(graph->Attach(graph->GetRootNode(), node));
    }
    SceneGraphNode* parent = graph->GetRootNode();
    std::vector<SceneGraphNode*> chain;
    for (int d = 1; d <= 70; ++d)
    {
        auto node = make_node("c");
        node->SetTranslation(dm::double3(0.0, 0.5, 0.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
        parent = graph->Attach(parent, node);
        chain.push_back(parent);
    }
    GpuScene gpu(nullptr, test_desc());
    Mirror mirror;
    PrevModel model;
    uint32_t frame = 0;
    auto step = [&](const char* where)
    {
        graph->Refresh(++frame);
        CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)));
        CHECK(NVRHI_SUCCEEDED(gpu.UploadTo(&mirror)));
        verify_frame(graph, gpu, mirror, model, where);
        return gpu.GetStats();
    };
    step("load");
    step("settle");

    // (1) a chunk-only write of WorldTransform, then one move in the chunk
    IWorld* world = graph->GetWorld();
    world->AdvanceVersion();
    Query<Write<WorldTransform>, With<MeshInstanceLeaf>> worlds(world, "chunk-only world writer");
    uint32_t written = 0;
    worlds.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, WorldTransform* w)
    {
        for (uint32_t r = 0; r < view.count; ++r)
            w[r].transform.m_translation.z += 1.0;
        written += view.count;
    });
    CHECK(written == 64 + 70);
    nodes[10]->SetTranslation(dm::double3(10.0, 2.0, 0.0));
    GpuSceneStats stats = step("chunk-only write");
    CHECK(stats.rowsChanged >= 64);
    step("settle");

    // (2) the head columns: RenderFlags of one row rewritten by InvalidateContent, its world unchanged
    curve->type = MeshType::CurveLinearSweptSpheres;
    nodes[20]->InvalidateContent();
    stats = step("render flags");
    CHECK(stats.rowsChanged >= 1);
    step("settle");

    // (3) the deep bucket by row
    chain[65]->SetTranslation(dm::double3(0.0, 0.5, 1.0));   // depth 66
    stats = step("deep row");
    CHECK(stats.rowsBuilt == 5 && stats.rowsChanged == 5);
    std::printf("row tracked killers: chunk-only world write, head columns, deep rows by row (%u built)\n",
                stats.rowsBuilt);
}

int main()
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
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a run that dies still shows how far it got
    try
    {
        std::printf("%s\n", "test_instance_slots_are_stable");
        test_instance_slots_are_stable();
        std::printf("%s\n", "test_numbering_by_first_use");
        test_numbering_by_first_use();
        std::printf("%s\n", "test_asset_callbacks");
        test_asset_callbacks();
        std::printf("%s\n", "test_gpu_scene_tracks_changes");
        test_gpu_scene_tracks_changes();
        std::printf("%s\n", "test_uploads_follow_changes");
        test_uploads_follow_changes();
        std::printf("%s\n", "test_meshes_edited_in_place");
        test_meshes_edited_in_place();
        std::printf("%s\n", "test_another_graph");
        test_another_graph();
        std::printf("%s\n", "test_steady_state_allocations");
        test_steady_state_allocations();
        std::printf("%s\n", "test_review_fixes");
        test_review_fixes();
        std::printf("%s\n", "test_out_of_memory_paths");
        test_out_of_memory_paths();
        std::printf("%s\n", "test_scatter_matches_cpu_mode");
        test_scatter_matches_cpu_mode();
        std::printf("%s\n", "test_scatter_records");
        test_scatter_records();
        std::printf("%s\n", "test_scatter_workers");
        test_scatter_workers();
        std::printf("%s\n", "test_scatter_steady_state_allocations");
        test_scatter_steady_state_allocations();
        std::printf("%s\n", "test_scatter_lost_prepare");
        test_scatter_lost_prepare();
        std::printf("%s\n", "test_row_tracked_transforms");
        test_row_tracked_transforms();
        std::printf("%s\n", "test_row_tracked_equals_untracked (exact; backlog T4-1, fixed in F8)");
        test_row_tracked_equals_untracked(true);
        std::printf("%s\n", "test_row_tracked_killers");
        test_row_tracked_killers();
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    std::printf("gpu scene: ok\n");
    return 0;
}
