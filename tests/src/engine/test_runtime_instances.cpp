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

// Runtime-only mesh instances in the scene graph's world (aggregate docs/plans/2026-11-26-donut-ecs-e7.md, item E7.1;
// design 5.8): rows with RuntimeMeshInstance and no node or leaf, made with CreateEntities, Instantiate and command
// buffers (PreSimulation and EndFrame), destroyed the same ways. The graph's observer on the tag: slots equal to a fresh
// allocation and reused lowest first, bulk mesh references per run of one mesh, OnMeshAdded and OnMeshRemoved once
// each, RenderFlags and LocalBounds from the mesh (equal to a leaf's of the same mesh), unknown ids, a Parent (a
// root, one warning), rows without MeshRef (one warning) or GpuInstanceSlot (added). Mixed graphs; what the node API
// does not see; Refresh when only runtime rows changed; GpuScene's rows equal to a node-backed build's; driven rows
// under a toy domain; a 200-frame spawn, move and despawn script whose per-frame hashes are the same at 1, 2 and 8
// workers and those of the golden file (tests/src/engine/golden/runtime_instances_frames.txt; ECS_UPDATE_GOLDEN=1
// rewrites it); graph teardown with the world released and kept; allocations in steady state and of a spawn after
// warm-up; a copy from another world (the observer cannot tell it: documented and fixed here as unsupported); a rewind
// with and without the application's mesh reference; the bulk reference calls of MeshTable. From the review of E7.1:
// slots follow membership (rows destroyed before their slot column is added, the column removed and added, an
// application observer's destroys in a load), the tag on node rows is ignored, asset callbacks that call back into the
// graph, and a Parent naming the root.

#include "SceneGraphGenerator.h"

#include <donut/core/log.h>
#include <donut/engine/AssetTables.h>
#include <donut/engine/EcsCommandBuffer.h>
#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include "EcsWorldImpl.h"
#include "SceneGraphStore.h"

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
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

namespace runtime_test
{
    // The frame at which the script destroys a row.
    struct Lifetime
    {
        uint32_t despawnFrame;
    };
    DONUT_ECS_COMPONENT(Lifetime, "0f6c2d2e-7a51-4b8e-9d3c-5e1f8a6b4c27", DONUT_ECS_FIELDS(despawnFrame))

    // A tag that moves rows to another archetype (a structural change that keeps the runtime tag).
    struct Marked { };
    DONUT_ECS_TAG(Marked, "b5d7e3a1-2c4f-4e6a-8b9d-0a1c3e5f7b92")
}

using namespace runtime_test;

// log::warning calls; main fails a case that leaves any (Debug worlds report validation errors through it). A case that
// provokes warnings on purpose checks and takes them back.
std::atomic<uint32_t> g_Warnings{ 0 };

namespace
{
    std::atomic<int> g_Errors{ 0 };   // expectations that failed in noexcept callbacks
    void Expect(bool condition, int line, const char* text) noexcept
    {
        if (condition)
            return;
        g_Errors.fetch_add(1);
        std::fprintf(stderr, "test_runtime_instances.cpp:%d: expected %s\n", line, text);
    }
#define EXPECT(condition) Expect(bool(condition), __LINE__, #condition)

    uint32_t TakeWarnings() { return g_Warnings.exchange(0); }

    // SceneGraph::m_Store (an explicit instantiation may name a private member).
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
    SceneGraphStore& StoreOf(SceneGraph* graph) { return *(graph->*GetMember(StoreTag())); }
    // SceneGraph::m_Meshes, to bring a row's reference count near its limit.
    struct MeshesTag
    {
        using Type = MeshTable SceneGraph::*;
        friend Type GetMember(MeshesTag);
    };
    template struct PrivateMember<MeshesTag, &SceneGraph::m_Meshes>;
    MeshTable& MeshesOf(SceneGraph* graph) { return graph->*GetMember(MeshesTag()); }
    struct MaterialsTag
    {
        using Type = MaterialTable SceneGraph::*;
        friend Type GetMember(MaterialsTag);
    };
    template struct PrivateMember<MaterialsTag, &SceneGraph::m_Materials>;
    MaterialTable& StoreMaterials(SceneGraph* graph) { return graph->*GetMember(MaterialsTag()); }

    std::string g_LastWarning;   // the text of the last warning (main's log callback)

    bool SameBits(const void* a, const void* b, size_t size) { return std::memcmp(a, b, size) == 0; }
    template <typename T> bool SameValue(const T& a, const T& b) { return SameBits(&a, &b, sizeof(T)); }

    dm::daffine3 ComposeTransform(const LocalTransform& local)
    {
        dm::daffine3 transform = dm::scaling(local.scaling);
        transform *= local.rotation.toAffine();
        transform *= dm::translation(local.translation);
        return transform;
    }

    uint64_t ReferenceCountOf(nvrhi::IObject* object)
    {
        object->AddRef();
        return uint64_t(object->Release());
    }

    // FNV-1a, 64 bits.
    struct Hasher
    {
        uint64_t value = 0xcbf29ce484222325ull;
        void Add(const void* data, size_t size)
        {
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; ++i)
            {
                value ^= bytes[i];
                value *= 0x100000001b3ull;
            }
        }
        template <typename T> void Add(const T& item) { Add(&item, sizeof(T)); }
    };

    nvrhi::AutoPtr<IJobSystem> MakeJobs(uint32_t workers)
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        JobSystemDesc desc;
        desc.debugName = "test_runtime_instances";
        desc.workerThreadCount = workers;
        CHECK(NVRHI_SUCCEEDED(CreateJobSystem(desc, &jobs)));
        return jobs;
    }

    nvrhi::AutoPtr<Material> MakeMaterial(MaterialDomain domain)
    {
        auto material = MAKE_RC_OBJ_PTR(Material);
        material->domain = domain;
        return material;
    }

    // A mesh with one geometry per material; its box grows with `size`.
    nvrhi::AutoPtr<MeshInfo> MakeMesh(const std::vector<Material*>& materials, float size = 1.f,
                                      MeshType type = MeshType::Triangles)
    {
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->type = type;
        uint32_t index = 0;
        for (Material* material : materials)
        {
            auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
            geometry->material = material;
            geometry->numIndices = 3 * (++index);
            geometry->numVertices = 3;
            geometry->objectSpaceBounds = dm::box3(dm::float3(-size), dm::float3(size, 2.f * size, size));
            mesh->geometries.push_back(geometry);
        }
        mesh->objectSpaceBounds = dm::box3(dm::float3(-size), dm::float3(size, 2.f * size, size));
        return mesh;
    }

    LocalTransform MakeLocal(const dm::double3& translation, const dm::dquat& rotation = dm::dquat::identity(),
                             double scale = 1.0)
    {
        return LocalTransform{ translation, rotation, dm::double3(scale) };
    }

    LocalTransform RandomLocal(Rng& rng, double extent = 50.0)
    {
        return MakeLocal(RandomVector(rng, -extent, extent), RandomRotation(rng), rng.Range(0.5, 2.0));
    }

    // ---------------------------------------------------------------------------------------------- the rig

    struct Rig
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        IWorld* world = nullptr;
        std::vector<ComponentId> plain;     // GetRuntimeMeshInstanceComponents
        std::vector<ComponentId> driven;    // ... with the simulation-driven ones
        uint32_t meshesAdded = 0;
        uint32_t meshesRemoved = 0;
        uint32_t materialsAdded = 0;
        uint32_t materialsRemoved = 0;
        uint64_t frame = 0;
        double time = 0.0;

        explicit Rig(uint32_t workers = 2)
            : jobs(MakeJobs(workers))
        {
            graph->SetJobSystem(jobs);
            world = graph->GetWorld();
            plain.resize(graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
            graph->GetRuntimeMeshInstanceComponents(plain.data(), uint32_t(plain.size()));
            driven.resize(graph->GetRuntimeMeshInstanceComponents(nullptr, 0, true));
            graph->GetRuntimeMeshInstanceComponents(driven.data(), uint32_t(driven.size()), true);
            graph->OnMeshAdded = [this](const MeshInfo*) { ++meshesAdded; };
            graph->OnMeshRemoved = [this](const MeshInfo*) { ++meshesRemoved; };
            graph->OnMaterialAdded = [this](const Material*) { ++materialsAdded; };
            graph->OnMaterialRemoved = [this](const Material*) { ++materialsRemoved; };
        }
        ~Rig()
        {
            if (graph)
            {
                graph->OnMeshAdded = nullptr;
                graph->OnMeshRemoved = nullptr;
                graph->OnMaterialAdded = nullptr;
                graph->OnMaterialRemoved = nullptr;
            }
        }

        SceneGraphStore& Store() { return StoreOf(graph); }
        uint32_t References(MeshId id) const
        {
            const MeshTable& meshes = graph->GetMeshes();
            return meshes.Get(id) ? meshes.GetRow(MeshTable::SlotOf(id)).references : 0;
        }
        template <typename T> const T* Get(Entity entity) { return GetComponent<T>(world, entity); }
        const GpuInstanceSlot& Slot(Entity entity)
        {
            const GpuInstanceSlot* slot = Get<GpuInstanceSlot>(entity);
            CHECK(slot);
            return *slot;
        }
        bool HoldsSlot(Entity entity) { return Slot(entity).instance != c_NoInstanceSlot; }

        // Rows made at once (one CreateEntities call, one observer batch per chunk), mesh i of `meshes` on row i, with
        // `extra` components (values per row, or null) after the listed ones.
        std::vector<Entity> Spawn(const std::vector<MeshId>& meshes, const std::vector<LocalTransform>& locals,
                                  const std::vector<std::pair<ComponentId, const void*>>& extra = {},
                                  const std::vector<ComponentId>* components = nullptr)
        {
            CHECK(meshes.size() == locals.size() && !meshes.empty());
            const std::vector<ComponentId>& listed = components ? *components : plain;
            std::vector<MeshRef> refs;
            for (MeshId id : meshes)
                refs.push_back(MeshRef{ id });
            std::vector<ComponentId> ids = listed;
            std::vector<const void*> values(ids.size(), nullptr);
            for (size_t i = 0; i < ids.size(); ++i)
            {
                if (ids[i] == FindComponent<LocalTransform>(world))
                    values[i] = locals.data();
                else if (ids[i] == FindComponent<MeshRef>(world))
                    values[i] = refs.data();
            }
            for (const auto& component : extra)
            {
                ids.push_back(component.first);
                values.push_back(component.second);
            }
            EntityCreateDesc desc{};
            desc.components = ids.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(ids.size());
            desc.entityCount = uint32_t(meshes.size());
            std::vector<Entity> entities(meshes.size(), Entity::Null);
            CHECK(world->CreateEntities(desc, entities.data()) == nvrhi::FS_OK);
            return entities;
        }
        std::vector<Entity> Spawn(MeshId mesh, uint32_t count, double x0 = 0.0)
        {
            std::vector<LocalTransform> locals;
            for (uint32_t i = 0; i < count; ++i)
                locals.push_back(MakeLocal(dm::double3(x0 + double(i), 1.0, -2.0)));
            return Spawn(std::vector<MeshId>(count, mesh), locals);
        }
        void Destroy(const std::vector<Entity>& entities)
        {
            CHECK(world->DestroyEntities(entities.data(), uint32_t(entities.size())) == nvrhi::FS_OK);
        }

        void Refresh() { graph->Refresh(uint32_t(frame)); }
        void SimulateGraph(double delta)
        {
            time += delta;
            CHECK(graph->Simulate(FrameTime{ time, delta, frame++ }) == nvrhi::FS_OK);
        }
        void EndFrame() { CHECK(graph->GetScheduler()->RunPhase(Phase::EndFrame) == nvrhi::FS_OK); }
        // A frame as a sample runs it without a render service: Simulate, Refresh, EndFrame.
        void Frame(double delta = 1.0 / 60.0)
        {
            SimulateGraph(delta);
            Refresh();
            EndFrame();
        }
        SystemId AddSystem(const char* name, Phase phase, const std::vector<ComponentAccess>& accesses,
                           const std::function<nvrhi::FRESULT(ISystemContext*)>& update)
        {
            SystemDesc desc{};
            desc.name = name;
            desc.phase = phase;
            desc.flags = SystemFlags::MainThread;
            desc.accesses = accesses.data();
            desc.accessCount = uint32_t(accesses.size());
            nvrhi::AutoPtr<ISystem> system;
            CHECK(CreateLambdaSystem(desc, [update](ISystemContext* context) noexcept { return update(context); },
                                     &system) == nvrhi::FS_OK);
            SystemId id = SystemId::Invalid;
            CHECK(graph->GetScheduler()->AddSystem(system, id) == nvrhi::FS_OK);
            return id;
        }

        // Every runtime row's entity and slot, in the query's chunk order.
        std::vector<std::pair<Entity, GpuInstanceSlot>> RuntimeRows()
        {
            Query<Read<GpuInstanceSlot>, With<RuntimeMeshInstance>> rows(world, "runtime rows");
            std::vector<std::pair<Entity, GpuInstanceSlot>> out;
            rows.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, const GpuInstanceSlot* slots)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                    out.emplace_back(view.entities[r], slots[r]);
            });
            return out;
        }
        // The archetype a live entity is in.
        uint32_t ArchetypeOf(Entity entity)
        {
            const ecs::EntityRecord* record = ecs::GetWorldImpl(world)->FindAlive(entity);
            CHECK(record);
            return record->archetype;
        }
    };

    // Two materials (opaque, blended), a two-geometry mesh and a one-geometry mesh.
    struct Assets
    {
        nvrhi::AutoPtr<Material> opaque = MakeMaterial(MaterialDomain::Opaque);
        nvrhi::AutoPtr<Material> blended = MakeMaterial(MaterialDomain::AlphaBlended);
        nvrhi::AutoPtr<MeshInfo> two = MakeMesh({ opaque, blended }, 1.f);
        nvrhi::AutoPtr<MeshInfo> one = MakeMesh({ opaque }, 3.f);
    };

    class MemoryWriter final : public nvrhi::ObjectImpl<IStateWriter>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MemoryWriter)
        NVRHI_IMPLEMENTS_INTERFACE(IStateWriter)
        NVRHI_END_INTERFACE_TABLE()

        std::vector<uint8_t> bytes;
        nvrhi::FRESULT Write(const void* data, uint64_t size) noexcept override
        {
            const uint8_t* begin = static_cast<const uint8_t*>(data);
            bytes.insert(bytes.end(), begin, begin + size);
            return nvrhi::FS_OK;
        }
    };

    class MemoryReader final : public nvrhi::ObjectImpl<IStateReader>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MemoryReader)
        NVRHI_IMPLEMENTS_INTERFACE(IStateReader)
        NVRHI_END_INTERFACE_TABLE()

        std::vector<uint8_t> bytes;
        size_t at = 0;
        explicit MemoryReader(const std::vector<uint8_t>& content) : bytes(content) { }
        nvrhi::FRESULT Read(uint64_t size, void* data) noexcept override
        {
            if (size > bytes.size() - at)
                return nvrhi::FE_INVALID_ARGS;
            std::memcpy(data, bytes.data() + at, size_t(size));
            at += size_t(size);
            return nvrhi::FS_OK;
        }
    };

    GpuSceneDesc CpuDesc()
    {
        GpuSceneDesc desc;
        desc.geometryAndMaterialBuffers = true;
        desc.rowGranularity = 64;
        return desc;
    }
}

// -------------------------------------------------------------------------------------------------- cases

// The component lists: 9 plain, 14 driven (the plain ones first), the ids of the graph's world, in SceneComponents.h's
// order; null with capacity 0 asks; a short capacity writes what fits.
void test_components()
{
    Rig rig;
    CHECK(rig.plain.size() == 9 && rig.driven.size() == SceneGraph::c_MaxRuntimeMeshInstanceComponents);
    IWorld* w = rig.world;
    const std::vector<ComponentId> expected = {
        FindComponent<RuntimeMeshInstance>(w), FindComponent<LocalTransform>(w), FindComponent<WorldTransform>(w),
        FindComponent<LocalBounds>(w), FindComponent<WorldBounds>(w), FindComponent<ChunkWorldBounds>(w),
        FindComponent<MeshRef>(w), FindComponent<RenderFlags>(w), FindComponent<GpuInstanceSlot>(w),
        FindComponent<SimulationDriven>(w), FindComponent<SimPose>(w), FindComponent<SimPosePrevious>(w),
        FindComponent<SimScale>(w), FindComponent<SimBodyRef>(w),
    };
    for (ComponentId id : expected)
        CHECK(id != ComponentId::Invalid);
    CHECK(std::equal(rig.plain.begin(), rig.plain.end(), expected.begin()));
    CHECK(std::equal(rig.driven.begin(), rig.driven.end(), expected.begin()));
    // no teleport request: such a row has no setter
    CHECK(std::find(rig.driven.begin(), rig.driven.end(), FindComponent<SimTeleport>(w)) == rig.driven.end());
    ComponentId few[3] = { ComponentId::Invalid, ComponentId::Invalid, ComponentId::Invalid };
    CHECK(rig.graph->GetRuntimeMeshInstanceComponents(few, 2) == 9);
    CHECK(few[0] == expected[0] && few[1] == expected[1] && few[2] == ComponentId::Invalid);
    CHECK(rig.graph->GetRuntimeMeshInstanceComponents(nullptr, 5, true) == 14);
}

// Immediate creation and destruction: slots equal to a fresh allocation (the geometry-instance range first, as for a
// leaf), reused lowest first; one reference per row taken in bulk; OnMeshAdded and OnMeshRemoved once each (and the
// materials once each); RenderFlags and LocalBounds from the mesh; the count of rows that hold slots.
void test_create_and_destroy()
{
    Rig rig;
    Assets assets;
    const MeshId two = rig.graph->AcquireRuntimeMesh(assets.two);
    CHECK(two != MeshId::Invalid && rig.meshesAdded == 1 && rig.materialsAdded == 2);
    CHECK(rig.References(two) == 1 && rig.Store().GetRuntimeMeshHolds(two) == 1);
    // a second acquisition: the same id, no event
    CHECK(rig.graph->AcquireRuntimeMesh(assets.two) == two && rig.meshesAdded == 1 && rig.References(two) == 2);
    rig.graph->ReleaseRuntimeMesh(two);
    CHECK(rig.References(two) == 1 && rig.meshesRemoved == 0);

    std::vector<Entity> rows = rig.Spawn(two, 10);
    CHECK(rig.References(two) == 11 && rig.graph->GetRuntimeMeshInstanceCount() == 10);
    for (uint32_t i = 0; i < 10; ++i)
    {
        const GpuInstanceSlot& slot = rig.Slot(rows[i]);
        CHECK(slot.instance == i && slot.firstGeometryInstance == 2 * i && slot.geometryCount == 2);
        const RenderFlags* flags = rig.Get<RenderFlags>(rows[i]);
        CHECK(flags->instanceFlags == 0 &&
              flags->contentFlags == uint32_t(SceneContentFlags::OpaqueMeshes | SceneContentFlags::BlendedMeshes));
        CHECK(SameValue(rig.Get<LocalBounds>(rows[i])->bounds, assets.two->objectSpaceBounds));
    }
    CHECK(rig.graph->GetInstanceSlotCount() == 10 && rig.graph->GetGeometryInstancesCount() == 20);
    CHECK(rig.graph->GetMaxGeometryCountPerMesh() == 2);

    // the application's reference may go once rows hold theirs
    rig.graph->ReleaseRuntimeMesh(two);
    CHECK(rig.References(two) == 10 && rig.meshesRemoved == 0 && rig.Store().GetRuntimeMeshHolds(two) == 0);

    // destroyed rows free their slots; the next rows take the lowest free ones
    rig.Destroy({ rows[3], rows[7] });
    CHECK(rig.References(two) == 8 && rig.graph->GetRuntimeMeshInstanceCount() == 8);
    std::vector<Entity> more = rig.Spawn(two, 3, 100.0);
    CHECK(rig.Slot(more[0]).instance == 3 && rig.Slot(more[0]).firstGeometryInstance == 6);
    CHECK(rig.Slot(more[1]).instance == 7 && rig.Slot(more[1]).firstGeometryInstance == 14);
    CHECK(rig.Slot(more[2]).instance == 10 && rig.Slot(more[2]).firstGeometryInstance == 20);
    CHECK(rig.References(two) == 11 && rig.graph->GetRuntimeMeshInstanceCount() == 11);

    // a batch of two meshes in runs: [one one two one]
    const MeshId one = rig.graph->AcquireRuntimeMesh(assets.one);
    CHECK(rig.meshesAdded == 2 && rig.materialsAdded == 2);   // its material is in the table already
    std::vector<LocalTransform> locals(4, MakeLocal(dm::double3(5.0)));
    std::vector<Entity> runs = rig.Spawn({ one, one, two, one }, locals);
    CHECK(rig.References(one) == 4 && rig.References(two) == 12);
    CHECK(rig.Slot(runs[0]).geometryCount == 1 && rig.Slot(runs[2]).geometryCount == 2);
    CHECK(rig.Get<RenderFlags>(runs[0])->contentFlags == uint32_t(SceneContentFlags::OpaqueMeshes));
    CHECK(SameValue(rig.Get<LocalBounds>(runs[3])->bounds, assets.one->objectSpaceBounds));
    rig.graph->ReleaseRuntimeMesh(one);

    // everything goes: each mesh removed once, then its materials
    std::vector<Entity> all = rows;
    all.erase(std::remove_if(all.begin(), all.end(), [&](Entity e) { return !rig.world->IsAlive(e); }), all.end());
    all.insert(all.end(), more.begin(), more.end());
    all.insert(all.end(), runs.begin(), runs.end());
    rig.Destroy(all);
    CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 0 && rig.meshesRemoved == 2 && rig.materialsRemoved == 2);
    CHECK(rig.graph->GetMeshes().empty() && rig.graph->GetMaterials().empty());
    // the events held the removed assets only until their callbacks ran
    CHECK(ReferenceCountOf(assets.two) == 1 && ReferenceCountOf(assets.one) == 1);
    CHECK(TakeWarnings() == 0);
}

// Instantiate within the graph's world: the copy arrives with its source's MeshRef and slot values and gets slots and a
// reference of its own; a copy through CopyEntitiesFrom of the same world, too.
void test_instantiate()
{
    Rig rig;
    Assets assets;
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.two);
    const Entity prefab = rig.Spawn(mesh, 1, 3.0)[0];
    Entity copies[5];
    CHECK(rig.world->Instantiate(prefab, 5, copies) == nvrhi::FS_OK);
    CHECK(rig.References(mesh) == 1 + 1 + 5 && rig.graph->GetRuntimeMeshInstanceCount() == 6);
    std::set<uint32_t> slots{ rig.Slot(prefab).instance };
    for (Entity copy : copies)
    {
        CHECK(rig.Get<MeshRef>(copy)->mesh == mesh && rig.Slot(copy).geometryCount == 2);
        slots.insert(rig.Slot(copy).instance);
    }
    CHECK(slots.size() == 6 && *slots.rbegin() == 5);
    Entity sameWorld[2];
    const Entity sources[2] = { prefab, copies[2] };
    CHECK(rig.world->CopyEntitiesFrom(rig.world, sources, 2, sameWorld) == nvrhi::FS_OK);
    CHECK(rig.References(mesh) == 9 && rig.Slot(sameWorld[0]).instance == 6 && rig.Slot(sameWorld[1]).instance == 7);
    // the copies' references are their own: the prefab goes, they stay
    rig.Destroy({ prefab });
    rig.graph->ReleaseRuntimeMesh(mesh);
    CHECK(rig.References(mesh) == 7 && rig.meshesRemoved == 0);
    rig.Refresh();
    for (Entity copy : copies)
        CHECK(rig.Get<WorldTransform>(copy)->transform.m_translation.x == 3.0);
    CHECK(TakeWarnings() == 0);
}

// Command buffers: rows spawned and despawned by a PreSimulation system appear and go at the sync point after
// PostSimulation; a despawn recorded in EndFrame is played back at the next frame's sync point (E2.2's known limit 1),
// so the row is still there, with its slot, for one more Refresh and extraction.
void test_command_buffers()
{
    Rig rig;
    Assets assets;
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.one);
    const ComponentId lifetimeId = RegisterComponent<Lifetime>(rig.world);
    uint32_t spawnNow = 0;
    std::vector<Entity> placeholders;
    std::vector<Entity> despawnNow;
    std::vector<Entity> despawnAtEnd;
    rig.AddSystem("spawner", Phase::PreSimulation, {}, [&](ISystemContext* context)
    {
        ICommandBuffer* commands = context->GetCommandBuffer();
        if (spawnNow)
        {
            std::vector<LocalTransform> locals;
            for (uint32_t i = 0; i < spawnNow; ++i)
                locals.push_back(MakeLocal(dm::double3(double(i), 0.0, 0.0)));
            std::vector<MeshRef> refs(spawnNow, MeshRef{ mesh });
            std::vector<Lifetime> lifetimes(spawnNow, Lifetime{ 0 });
            std::vector<ComponentId> ids = rig.plain;
            ids.push_back(lifetimeId);
            std::vector<const void*> values(ids.size(), nullptr);
            values[1] = locals.data();
            values[6] = refs.data();
            values.back() = lifetimes.data();
            EntityCreateDesc desc{};
            desc.components = ids.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(ids.size());
            desc.entityCount = spawnNow;
            placeholders.resize(spawnNow);
            commands->CreateEntities(desc, MakeSortKey(0, 0, 0), placeholders.data());
            spawnNow = 0;
        }
        if (!despawnNow.empty())
        {
            commands->DestroyEntities(despawnNow.data(), uint32_t(despawnNow.size()), MakeSortKey(1, 0, 0));
            despawnNow.clear();
        }
        return nvrhi::FS_OK;
    });
    rig.AddSystem("end of frame", Phase::EndFrame, {}, [&](ISystemContext* context)
    {
        if (!despawnAtEnd.empty())
        {
            context->GetCommandBuffer()->DestroyEntities(despawnAtEnd.data(), uint32_t(despawnAtEnd.size()),
                                                         MakeSortKey(0, 0, 0));
            despawnAtEnd.clear();
        }
        return nvrhi::FS_OK;
    });

    spawnNow = 50;
    rig.SimulateGraph(0.01);
    std::vector<Entity> rows;
    for (Entity placeholder : placeholders)
        rows.push_back(rig.world->ResolvePlaceholder(placeholder));
    CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 50 && rig.References(mesh) == 51);
    for (uint32_t i = 0; i < 50; ++i)
        CHECK(rig.world->IsAlive(rows[i]) && rig.Slot(rows[i]).instance == i);
    rig.Refresh();
    CHECK(rig.Get<WorldTransform>(rows[49])->transform.m_translation.x == 49.0);
    rig.EndFrame();

    // a PreSimulation despawn: gone after the sync point
    despawnNow = { rows[0], rows[1] };
    rig.SimulateGraph(0.01);
    CHECK(!rig.world->IsAlive(rows[0]) && rig.graph->GetRuntimeMeshInstanceCount() == 48);
    rig.Refresh();
    rig.EndFrame();

    // an EndFrame despawn: still there through the next Refresh (one more extraction), gone after the next sync point
    despawnAtEnd = { rows[2] };
    const GpuInstanceSlot held = rig.Slot(rows[2]);
    rig.EndFrame();
    CHECK(rig.world->IsAlive(rows[2]) && SameValue(rig.Slot(rows[2]), held));
    rig.Refresh();
    CHECK(rig.world->IsAlive(rows[2]) && rig.graph->GetRuntimeMeshInstanceCount() == 48);
    rig.SimulateGraph(0.01);
    CHECK(!rig.world->IsAlive(rows[2]) && rig.graph->GetRuntimeMeshInstanceCount() == 47);
    CHECK(rig.References(mesh) == 48);

    // a spawn and a despawn of the same frame: the freed slots are taken lowest first
    despawnNow = { rows[10], rows[20] };
    spawnNow = 3;
    rig.SimulateGraph(0.01);
    std::vector<uint32_t> newSlots;
    for (Entity placeholder : placeholders)
        newSlots.push_back(rig.Slot(rig.world->ResolvePlaceholder(placeholder)).instance);
    // the spawn's sort key is lower: it is played back first and takes the slots free then (0, 1, 2)
    CHECK(newSlots == std::vector<uint32_t>({ 0, 1, 2 }));
    CHECK(!rig.world->IsAlive(rows[10]) && rig.graph->GetRuntimeMeshInstanceCount() == 48);
    CHECK(TakeWarnings() == 0);
}

// A run of rows whose references would overflow their mesh's count: no slots, their LocalBounds as given, a warning
// that says so (not the one of unknown ids); the count unchanged.
void test_reference_overflow()
{
    Rig rig;
    Assets assets;
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.two);
    CHECK(MeshesOf(rig.graph).AddReferences(mesh, ~0u - 3));   // 2^32 - 2 references now
    const LocalBounds given{ dm::box3(dm::float3(-5.f), dm::float3(5.f)) };
    std::vector<LocalBounds> bounds(3, given);
    std::vector<ComponentId> noBounds = rig.plain;
    noBounds.erase(noBounds.begin() + 3);
    const std::vector<LocalTransform> locals(3, MakeLocal(dm::double3(0.0)));
    std::vector<Entity> rows = rig.Spawn(std::vector<MeshId>(3, mesh), locals,
                                         { { FindComponent<LocalBounds>(rig.world), bounds.data() } }, &noBounds);
    CHECK(TakeWarnings() == 1 && g_LastWarning.find("overflow") != std::string::npos);
    CHECK(rig.References(mesh) == ~0u - 2 && rig.graph->GetRuntimeMeshInstanceCount() == 0);
    for (Entity row : rows)
        CHECK(!rig.HoldsSlot(row) && SameValue(*rig.Get<LocalBounds>(row), given));
    rig.Destroy(rows);
    std::vector<Entity> unknown = rig.Spawn(MeshId((5u << 24) | 7u), 1);
    CHECK(TakeWarnings() == 1 && g_LastWarning.find("name no mesh") != std::string::npos);
    rig.Destroy(unknown);
    MeshTable::Changes changes;
    MeshTable& meshes = MeshesOf(rig.graph);
    meshes.ReleaseReferences(mesh, ~0u - 3, StoreMaterials(rig.graph), changes);
    CHECK(rig.References(mesh) == 1 && changes.events.empty());
    rig.graph->ReleaseRuntimeMesh(mesh);
    CHECK(TakeWarnings() == 0);
}

// Ids that name no mesh of the table (Invalid, one released and freed, a made-up one): no slot, one warning per batch,
// the row stays, its MeshRef as given; it releases nothing when it goes. Rows without MeshRef: one warning; rows without
// GpuInstanceSlot get it.
void test_unknown_ids()
{
    Rig rig;
    Assets assets;
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.two);
    const MeshId freed = rig.graph->AcquireRuntimeMesh(assets.one);
    rig.graph->ReleaseRuntimeMesh(freed);
    CHECK(rig.meshesRemoved == 1 && !rig.graph->GetMeshes().Get(freed));
    const MeshId madeUp = MeshId((7u << 24) | 3u);
    std::vector<LocalTransform> locals(6, MakeLocal(dm::double3(1.0)));
    std::vector<Entity> rows = rig.Spawn({ mesh, MeshId::Invalid, freed, mesh, madeUp, mesh }, locals);
    CHECK(TakeWarnings() == 1);   // one batch
    CHECK(rig.References(mesh) == 4 && rig.graph->GetRuntimeMeshInstanceCount() == 3);
    CHECK(rig.HoldsSlot(rows[0]) && !rig.HoldsSlot(rows[1]) && !rig.HoldsSlot(rows[2]) && rig.HoldsSlot(rows[3]) &&
          !rig.HoldsSlot(rows[4]) && rig.HoldsSlot(rows[5]));
    CHECK(rig.Get<MeshRef>(rows[2])->mesh == freed && rig.Slot(rows[0]).instance == 0 &&
          rig.Slot(rows[3]).instance == 1 && rig.Slot(rows[5]).instance == 2);
    // their LocalBounds is empty (no mesh), so they add nothing to their chunk's box
    rig.Refresh();
    CHECK(rig.Get<LocalBounds>(rows[1])->bounds.isempty() && rig.Get<WorldBounds>(rows[1])->bounds.isempty());
    rig.Destroy(rows);
    CHECK(rig.References(mesh) == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 0 && TakeWarnings() == 0);

    // a row without MeshRef: reported, nothing taken, nothing released when it goes
    std::vector<ComponentId> noRef = rig.plain;
    noRef.erase(noRef.begin() + 6);
    std::vector<Entity> incomplete = rig.Spawn({ mesh, mesh }, std::vector<LocalTransform>(2, locals[0]), {}, &noRef);
    CHECK(TakeWarnings() == 1 && rig.References(mesh) == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 0);
    CHECK(!rig.HoldsSlot(incomplete[0]));
    rig.Destroy(incomplete);
    CHECK(rig.References(mesh) == 1 && TakeWarnings() == 0);

    // rows without GpuInstanceSlot (as a snapshot's are): the observer adds it, with their slots, before the call
    // returns; unknown ids among them get the column with no slot
    std::vector<ComponentId> noSlot(rig.plain.begin(), rig.plain.end() - 1);
    const uint32_t before = ecs::GetWorldImpl(rig.world)->GetArchetypeStorageCount();
    std::vector<Entity> repaired =
        rig.Spawn({ mesh, MeshId::Invalid, mesh }, std::vector<LocalTransform>(3, locals[0]), {}, &noSlot);
    CHECK(TakeWarnings() == 1 && rig.References(mesh) == 3 && rig.graph->GetRuntimeMeshInstanceCount() == 2);
    CHECK(rig.Slot(repaired[0]).instance == 0 && !rig.HoldsSlot(repaired[1]) && rig.Slot(repaired[2]).instance == 1);
    CHECK(rig.Slot(repaired[2]).geometryCount == 2 && rig.Slot(repaired[2]).firstGeometryInstance == 2);
    CHECK(rig.ArchetypeOf(repaired[0]) < before);   // the runtime archetype made above, slot column and all
    rig.Destroy(repaired);
    CHECK(rig.References(mesh) == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 0 && TakeWarnings() == 0);
    rig.graph->ReleaseRuntimeMesh(mesh);
    CHECK(rig.graph->GetMeshes().empty());
}

// AcquireRuntimeMesh and ReleaseRuntimeMesh refuse a null mesh, a call inside a phase, and a release that the caller
// holds no reference for (never acquired, released twice, rows' references only), each with a warning.
void test_refusals()
{
    Rig rig;
    Assets assets;
    CHECK(rig.graph->AcquireRuntimeMesh(nullptr) == MeshId::Invalid && TakeWarnings() == 1);
    const MeshId held = rig.graph->AcquireRuntimeMesh(assets.two);
    MeshId inside = MeshId(1);
    rig.AddSystem("acquires", Phase::PreSimulation, {}, [&](ISystemContext*)
    {
        inside = rig.graph->AcquireRuntimeMesh(assets.one);
        rig.graph->ReleaseRuntimeMesh(held);   // held by the caller, but inside a phase
        return nvrhi::FS_OK;
    });
    rig.SimulateGraph(0.01);
    CHECK(inside == MeshId::Invalid && TakeWarnings() == 2 && rig.References(held) == 1);
    CHECK(rig.graph->GetMeshes().size() == 1 && rig.Store().GetRuntimeMeshHolds(held) == 1);
    rig.graph->ReleaseRuntimeMesh(held);
    CHECK(rig.graph->GetMeshes().empty() && rig.meshesRemoved == 1);
    rig.meshesRemoved = 0;

    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.two);
    const std::vector<Entity> rows = rig.Spawn(mesh, 2);
    rig.graph->ReleaseRuntimeMesh(mesh);
    rig.graph->ReleaseRuntimeMesh(mesh);   // the rows' references are not the caller's
    CHECK(TakeWarnings() == 1 && rig.References(mesh) == 2);
    rig.graph->ReleaseRuntimeMesh(MeshId::Invalid);
    CHECK(TakeWarnings() == 1);
    rig.Destroy(rows);
    CHECK(rig.graph->GetMeshes().empty() && rig.meshesRemoved == 1);
    rig.graph->ReleaseRuntimeMesh(mesh);   // freed
    CHECK(TakeWarnings() == 1);
}

// The bulk reference calls of MeshTable: AddReferences refuses an id that is not in the table and an overflow, adding
// nothing; ReleaseReferences clamps a count above the row's references and ignores an id that is not there, each with
// a warning; reaching zero frees the row as Release does.
void test_bulk_references()
{
    Assets assets;
    MeshTable meshes;
    MaterialTable materials;
    MeshTable::Changes changes;
    const MeshId id = meshes.Acquire(assets.two, materials, changes);
    CHECK(changes.events.size() == 3);
    changes.Clear();
    CHECK(meshes.AddReferences(id, 4) && meshes.GetRow(MeshTable::SlotOf(id)).references == 5);
    CHECK(meshes.AddReferences(id, 0) && meshes.GetRow(MeshTable::SlotOf(id)).references == 5);
    CHECK(!meshes.AddReferences(id, ~0u - 4) && meshes.GetRow(MeshTable::SlotOf(id)).references == 5);
    CHECK(meshes.AddReferences(id, ~0u - 5) && meshes.GetRow(MeshTable::SlotOf(id)).references == ~0u);
    meshes.ReleaseReferences(id, ~0u - 5, materials, changes);
    CHECK(!meshes.AddReferences(MeshId::Invalid, 1) && !meshes.AddReferences(MeshId((1u << 24) | 9u), 1));
    meshes.ReleaseReferences(id, 3, materials, changes);
    CHECK(meshes.GetRow(MeshTable::SlotOf(id)).references == 2 && changes.events.empty() && TakeWarnings() == 0);
    meshes.ReleaseReferences(MeshId::Invalid, 1, materials, changes);
    CHECK(TakeWarnings() == 1 && changes.events.empty());
    meshes.ReleaseReferences(MeshId::Invalid, 0, materials, changes);   // nothing to release: no warning either
    CHECK(TakeWarnings() == 0);
    meshes.ReleaseReferences(id, 0, materials, changes);
    CHECK(TakeWarnings() == 0 && meshes.GetRow(MeshTable::SlotOf(id)).references == 2);
    meshes.ReleaseReferences(id, 7, materials, changes);   // clamped to 2: the row goes
    CHECK(TakeWarnings() == 1 && meshes.empty() && materials.empty() && changes.events.size() == 3);
    CHECK(!changes.events[0].added && changes.events[0].mesh.Get() == assets.two.Get());
    meshes.ReleaseReferences(id, 1, materials, changes);   // freed
    CHECK(TakeWarnings() == 1);
}

namespace
{
    // A leaf class with content flags of its own.
    class OverridingInstance final : public MeshInstance
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        explicit OverridingInstance(MeshInfo* mesh) : MeshInstance(mesh) { }
        SceneContentFlags GetContentFlags() const override { return SceneContentFlags::BlendedMeshes; }
    };
}

// RenderFlags and LocalBounds of a runtime row equal a leaf's of the same mesh: mesh types (the instance flags) and
// material domains (the content flags), the box.
void test_render_flags_equal_leaf()
{
    Rig rig;
    rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
    auto opaque = MakeMaterial(MaterialDomain::Opaque);
    auto tested = MakeMaterial(MaterialDomain::AlphaTested);
    auto blended = MakeMaterial(MaterialDomain::AlphaBlended);
    auto transmissive = MakeMaterial(MaterialDomain::Transmissive);
    std::vector<nvrhi::AutoPtr<MeshInfo>> meshes = {
        MakeMesh({ opaque }), MakeMesh({ tested, blended }, 2.f), MakeMesh({ transmissive }, 0.5f),
        MakeMesh({ opaque, tested }, 1.f, MeshType::CurveDisjointOrthogonalTriangleStrips),
        MakeMesh({ blended }, 4.f, MeshType::CurveLinearSweptSpheres), MakeMesh({}, 1.f),
    };
    meshes[0]->geometries.push_back(nullptr);   // a missing geometry adds nothing
    const SceneContentFlags expected[] = {
        SceneContentFlags::OpaqueMeshes, SceneContentFlags::AlphaTestedMeshes | SceneContentFlags::BlendedMeshes,
        SceneContentFlags::BlendedMeshes, SceneContentFlags::OpaqueMeshes | SceneContentFlags::AlphaTestedMeshes,
        SceneContentFlags::BlendedMeshes, SceneContentFlags::None,
    };
    for (size_t m = 0; m < meshes.size(); ++m)
    {
        const auto& mesh = meshes[m];
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        auto instance = MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get());
        node->SetLeaf(instance);
        SceneGraphNode* attached = rig.graph->Attach(rig.graph->GetRootNode(), node);
        const Entity leafRow = rig.graph->GetEntity(attached);
        const MeshId id = rig.graph->AcquireRuntimeMesh(mesh);
        const Entity row = rig.Spawn(id, 1)[0];
        CHECK(SameValue(*rig.Get<RenderFlags>(row), *rig.Get<RenderFlags>(leafRow)));
        CHECK(rig.Get<RenderFlags>(row)->contentFlags == uint32_t(instance->GetContentFlags()));
        CHECK(rig.Get<RenderFlags>(row)->contentFlags == uint32_t(expected[m]));
        CHECK(SameValue(rig.Get<LocalBounds>(row)->bounds, rig.Get<LocalBounds>(leafRow)->bounds));
        rig.graph->ReleaseRuntimeMesh(id);
    }
    CHECK(rig.Get<RenderFlags>(rig.RuntimeRows()[3].first)->instanceFlags ==
          c_InstanceFlagCurveDisjointOrthogonalTriangleStrips);
    CHECK(rig.Get<RenderFlags>(rig.RuntimeRows()[4].first)->instanceFlags == c_InstanceFlagCurveLinearSweptSpheres);
    // the shared rule keeps MeshInstance's answer for an instance without a mesh
    CHECK(MAKE_RC_OBJ_PTR(MeshInstance, nullptr)->GetContentFlags() == SceneContentFlags::None);
    // a leaf class that overrides GetContentFlags: its row takes the override (the leaf's MakeRenderFlags calls the
    // virtual), with the mesh's instance flags
    auto overriding = MAKE_RC_OBJ_PTR(SceneGraphNode);
    overriding->SetLeaf(MAKE_RC_OBJ_PTR(OverridingInstance, meshes[3].Get()));
    const Entity overridingRow = rig.graph->GetEntity(rig.graph->Attach(rig.graph->GetRootNode(), overriding));
    CHECK(rig.Get<RenderFlags>(overridingRow)->contentFlags == uint32_t(SceneContentFlags::BlendedMeshes));
    CHECK(rig.Get<RenderFlags>(overridingRow)->instanceFlags == c_InstanceFlagCurveDisjointOrthogonalTriangleStrips);
    CHECK(TakeWarnings() == 0);
}

// A graph of nodes and runtime rows: one archetype per kind, slots from one allocator; the node API does not see the
// runtime rows (GetMeshInstances, SceneGraphWalker, GetNode, the root's subgraph box and content flags); propagation
// and bounds compute both; moves of runtime rows by a system; row hints and subgraph passes run over the mix.
void test_mixed_graph()
{
    Rig rig(4);
    Assets assets;
    rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
    auto opaqueMesh = MakeMesh({ assets.opaque }, 1.f);
    std::vector<SceneGraphNode*> nodes;
    Rng rng(5);
    for (int i = 0; i < 40; ++i)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetTranslation(RandomVector(rng, -10.0, 10.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, opaqueMesh.Get()));
        nodes.push_back(rig.graph->Attach(i % 3 == 2 ? nodes[rng.Below(nodes.size())] : rig.graph->GetRootNode(),
                                          node));
    }
    // blended runtime rows far away: neither their box nor their content reaches a node's subgraph
    const MeshId blended = rig.graph->AcquireRuntimeMesh(MakeMesh({ assets.blended }, 1.f));
    std::vector<LocalTransform> locals;
    for (int i = 0; i < 300; ++i)
        locals.push_back(MakeLocal(dm::double3(1000.0 + double(i), 0.0, 0.0), RandomRotation(rng)));
    std::vector<Entity> rows = rig.Spawn(std::vector<MeshId>(300, blended), locals);
    // more nodes after them: slots continue from one allocator
    for (int i = 0; i < 5; ++i)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, opaqueMesh.Get()));
        nodes.push_back(rig.graph->Attach(nodes[i], node));
    }
    rig.Refresh();

    CHECK(rig.graph->GetMeshInstances().size() == 45 && rig.graph->GetInstanceSlotCount() == 345);
    std::set<uint32_t> slots;
    for (const auto& instance : rig.graph->GetMeshInstances())
        slots.insert(uint32_t(instance->GetInstanceIndex()));
    for (Entity row : rows)
        slots.insert(rig.Slot(row).instance);
    CHECK(slots.size() == 345 && *slots.rbegin() == 344);
    uint32_t walked = 0;
    for (SceneGraphWalker walker(rig.graph->GetRootNode()); walker; walker.Next(true))
        ++walked;
    CHECK(walked == 46);
    std::set<uint32_t> runtimeArchetypes;
    for (Entity row : rows)
    {
        CHECK(rig.graph->GetNode(row) == nullptr);
        runtimeArchetypes.insert(rig.ArchetypeOf(row));
    }
    CHECK(runtimeArchetypes.size() == 1);
    ArchetypeInfo info{};
    CHECK(rig.world->GetArchetypeInfo(*runtimeArchetypes.begin(), info) == nvrhi::FS_OK && info.componentCount == 9);
    CHECK(rig.ArchetypeOf(rig.graph->GetEntity(nodes[0])) != *runtimeArchetypes.begin());

    const dm::box3 rootBox = rig.graph->GetRootNode()->GetGlobalBoundingBox();
    CHECK(!rootBox.isempty() && rootBox.m_maxs.x < 100.f);
    CHECK((rig.graph->GetRootNode()->GetSubgraphContentFlags() & SceneContentFlags::BlendedMeshes) ==
          SceneContentFlags::None);
    for (size_t i = 0; i < rows.size(); i += 37)
    {
        CHECK(SameValue(rig.Get<WorldTransform>(rows[i])->transform, ComposeTransform(locals[i])));
        const dm::box3 box = rig.Get<LocalBounds>(rows[i])->bounds * dm::affine3(ComposeTransform(locals[i]));
        CHECK(SameValue(rig.Get<WorldBounds>(rows[i])->bounds, box));
    }

    // a system moves every runtime row; nodes move through the API; both propagate
    const ComponentId localId = FindComponent<LocalTransform>(rig.world);
    Query<Write<LocalTransform>, With<RuntimeMeshInstance>> movers(rig.world, "movers");
    rig.AddSystem("move", Phase::PreSimulation, { { localId, Access::Write, {} } }, [&](ISystemContext* context)
    {
        movers.ParallelForChunks(context, 1, [](const ChunkView& view, LocalTransform* local)
        {
            for (uint32_t r = 0; r < view.count; ++r)
                local[r].translation.y += 1.0;
        });
        return nvrhi::FS_OK;
    });
    nodes[3]->SetTranslation(dm::double3(1.0, 2.0, 3.0));
    rig.Frame();
    for (size_t i = 0; i < rows.size(); i += 29)
    {
        LocalTransform moved = locals[i];
        moved.translation.y += 1.0;
        CHECK(SameValue(rig.Get<WorldTransform>(rows[i])->transform, ComposeTransform(moved)));
    }
    CHECK(rig.graph->GetRootNode()->GetGlobalBoundingBox().m_maxs.x < 100.f);
    // a structural move of runtime rows (a tag added) keeps their slots and references
    const GpuInstanceSlot before = rig.Slot(rows[5]);
    CHECK(rig.world->AddComponent(rows[5], RegisterComponent<Marked>(rig.world), nullptr) == nvrhi::FS_OK);
    CHECK(SameValue(rig.Slot(rows[5]), before) && rig.References(blended) == 301);
    // the tag removed while the row stays: its slots and reference go, the row is plain data; added again: new ones
    const ComponentId tag = FindComponent<RuntimeMeshInstance>(rig.world);
    const uint32_t freedSlot = rig.Slot(rows[7]).instance;
    CHECK(rig.world->RemoveComponent(rows[7], tag) == nvrhi::FS_OK);
    CHECK(!rig.HoldsSlot(rows[7]) && rig.References(blended) == 300 && rig.graph->GetRuntimeMeshInstanceCount() == 299);
    CHECK(rig.Get<MeshRef>(rows[7])->mesh == blended);
    CHECK(rig.world->AddComponent(rows[7], tag, nullptr) == nvrhi::FS_OK);
    CHECK(rig.HoldsSlot(rows[7]) && rig.Slot(rows[7]).instance == freedSlot && rig.References(blended) == 301 &&
          rig.graph->GetRuntimeMeshInstanceCount() == 300);
    rig.Frame();
    CHECK(rig.graph->GetRootNode()->GetGlobalBoundingBox().m_maxs.x < 100.f);
    CHECK(TakeWarnings() == 0);
}

// A Parent on a runtime row is not supported: one warning per batch, and the row is a root (its world is its own
// local transform, whatever the parent does), in no node's subgraph. A HierarchyDepth on it is not read either.
void test_parent_is_ignored()
{
    Rig rig;
    Assets assets;
    rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
    auto parentNode = MAKE_RC_OBJ_PTR(SceneGraphNode);
    parentNode->SetTranslation(dm::double3(100.0, 0.0, 0.0));
    parentNode->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, assets.one.Get()));
    SceneGraphNode* parent = rig.graph->Attach(rig.graph->GetRootNode(), parentNode);
    auto childNode = MAKE_RC_OBJ_PTR(SceneGraphNode);
    childNode->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, assets.one.Get()));
    rig.graph->Attach(parent, childNode);
    rig.Refresh();
    const dm::box3 parentBox = parent->GetGlobalBoundingBox();

    const MeshId mesh = rig.graph->AcquireRuntimeMesh(MakeMesh({ assets.blended }, 1.f));
    const Parent parentValue{ rig.graph->GetEntity(parent) };
    std::vector<Parent> parents(4, parentValue);
    const std::vector<HierarchyDepth> depth(4, HierarchyDepth{ 5 });
    std::vector<LocalTransform> locals;
    Rng rng(9);
    for (int i = 0; i < 4; ++i)
        locals.push_back(MakeLocal(dm::double3(5000.0, double(i), 0.0), RandomRotation(rng)));
    std::vector<Entity> rows = rig.Spawn(std::vector<MeshId>(4, mesh), locals,
                                         { { FindComponent<Parent>(rig.world), parents.data() },
                                           { FindComponent<HierarchyDepth>(rig.world), depth.data() } });
    CHECK(TakeWarnings() == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 4);
    // and with the depth of the last bucket, whose serial pass reads Parent row by row
    const std::vector<HierarchyDepth> deepest(4, HierarchyDepth{ c_MaxHierarchyDepth });
    std::vector<Entity> deep = rig.Spawn(std::vector<MeshId>(4, mesh), locals,
                                         { { FindComponent<Parent>(rig.world), parents.data() },
                                           { FindComponent<HierarchyDepth>(rig.world), deepest.data() } });
    CHECK(TakeWarnings() == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 8);
    rows.insert(rows.end(), deep.begin(), deep.end());
    const std::vector<LocalTransform> first4 = locals;
    locals.insert(locals.end(), first4.begin(), first4.end());
    rig.Refresh();
    for (int i = 0; i < 8; ++i)
        CHECK(SameValue(rig.Get<WorldTransform>(rows[i])->transform, ComposeTransform(locals[i])));
    CHECK(SameValue(parent->GetGlobalBoundingBox(), parentBox));
    CHECK((parent->GetSubgraphContentFlags() & SceneContentFlags::BlendedMeshes) == SceneContentFlags::None);
    parent->SetTranslation(dm::double3(-100.0, 0.0, 0.0));
    rig.Refresh();
    for (int i = 0; i < 8; ++i)
        CHECK(SameValue(rig.Get<WorldTransform>(rows[i])->transform, ComposeTransform(locals[i])));
    CHECK(rig.graph->GetRootNode()->GetGlobalBoundingBox().m_maxs.x < 1000.f);
    CHECK(TakeWarnings() == 0);
}

// Refresh runs when only runtime rows changed (a spawn, a system's move, a despawn) and returns at once when nothing
// did.
void test_refresh_skips()
{
    Rig rig;
    Assets assets;
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.one);
    rig.Refresh();
    const SceneGraphStore::Counters& counters = rig.Store().GetCounters();
    uint64_t runs = counters.refreshes, skips = counters.skippedRefreshes;
    rig.Refresh();
    CHECK(counters.refreshes == runs && counters.skippedRefreshes == skips + 1);
    std::vector<Entity> rows = rig.Spawn(mesh, 20);
    rig.Refresh();
    CHECK(counters.refreshes == runs + 1);
    CHECK(rig.Get<WorldTransform>(rows[19])->transform.m_translation.x == 19.0);
    rig.Refresh();
    CHECK(counters.refreshes == runs + 1 && counters.skippedRefreshes == skips + 2);

    bool move = false;
    Query<Write<LocalTransform>, With<RuntimeMeshInstance>> movers(rig.world, "movers");
    rig.AddSystem("move", Phase::PreSimulation, { { FindComponent<LocalTransform>(rig.world), Access::Write, {} } },
                  [&](ISystemContext* context)
    {
        if (move)
            movers.ForEachChunk(context, [](const ChunkView& view, LocalTransform* local)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                    local[r].translation.z = 7.0;
            });
        return nvrhi::FS_OK;
    });
    move = true;
    rig.SimulateGraph(0.01);
    runs = counters.refreshes;
    rig.Refresh();
    CHECK(counters.refreshes == runs + 1 && rig.Get<WorldTransform>(rows[3])->transform.m_translation.z == 7.0);
    rig.Destroy({ rows[0] });
    rig.Refresh();
    CHECK(counters.refreshes == runs + 2);
    skips = counters.skippedRefreshes;
    rig.Refresh();
    CHECK(counters.skippedRefreshes == skips + 1);
    CHECK(TakeWarnings() == 0);
}

// GpuScene's CPU image of runtime rows equals that of a node-backed build of the same transforms and meshes, after
// slot remapping (by creation order), over a move, a despawn and a spawn.
void test_gpu_scene_equals_nodes()
{
    Assets assets;
    auto nodeTwo = MakeMesh({ assets.opaque, assets.blended }, 1.f);
    auto nodeOne = MakeMesh({ assets.opaque }, 3.f);
    auto rowTwo = MakeMesh({ assets.opaque, assets.blended }, 1.f);
    auto rowOne = MakeMesh({ assets.opaque }, 3.f);
    Rig nodesRig;
    Rig rowsRig;
    nodesRig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
    const MeshId two = rowsRig.graph->AcquireRuntimeMesh(rowTwo);
    const MeshId one = rowsRig.graph->AcquireRuntimeMesh(rowOne);

    Rng rng(77);
    std::vector<LocalTransform> locals;
    std::vector<bool> isTwo;
    for (int i = 0; i < 200; ++i)
    {
        locals.push_back(RandomLocal(rng));
        isTwo.push_back(i == 0 || rng.Percent(60));
    }
    std::vector<SceneGraphNode*> nodes;
    std::vector<MeshId> ids;
    for (int i = 0; i < 200; ++i)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetTransform(&locals[i].translation, &locals[i].rotation, &locals[i].scaling);
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, isTwo[i] ? nodeTwo.Get() : nodeOne.Get()));
        nodes.push_back(nodesRig.graph->Attach(nodesRig.graph->GetRootNode(), node));
        ids.push_back(isTwo[i] ? two : one);
    }
    std::vector<Entity> rows = rowsRig.Spawn(ids, locals);

    GpuScene nodeGpu(nullptr, CpuDesc());
    GpuScene rowGpu(nullptr, CpuDesc());
    uint32_t frame = 0;
    auto compare = [&](const std::vector<size_t>& live)
    {
        ++frame;
        nodesRig.graph->Refresh(frame);
        rowsRig.graph->Refresh(frame);
        CHECK(NVRHI_SUCCEEDED(nodeGpu.Update(nodesRig.graph)) && NVRHI_SUCCEEDED(rowGpu.Update(rowsRig.graph)));
        for (size_t i : live)
        {
            const MeshInstance* instance = static_cast<const MeshInstance*>(nodes[i]->GetLeaf());
            const uint32_t nodeSlot = uint32_t(instance->GetInstanceIndex());
            const GpuInstanceSlot rowSlot = rowsRig.Slot(rows[i]);
            CHECK(nodeSlot < nodeGpu.GetInstanceRowCount() && rowSlot.instance < rowGpu.GetInstanceRowCount());
            const InstanceData& a = nodeGpu.GetInstanceRows()[nodeSlot];
            const InstanceData& b = rowGpu.GetInstanceRows()[rowSlot.instance];
            CHECK(a.flags == b.flags && a.numGeometries == b.numGeometries && a.numGeometries == rowSlot.geometryCount);
            // the remapping: each one's geometry-instance range, the mesh's geometries in table order
            CHECK(a.firstGeometryInstanceIndex == uint32_t(instance->GetGeometryInstanceIndex()) &&
                  b.firstGeometryInstanceIndex == rowSlot.firstGeometryInstance);
            CHECK(a.firstGeometryIndex == b.firstGeometryIndex);
            CHECK(SameBits(&a.transform, &b.transform, sizeof(a.transform)));
            CHECK(SameBits(&a.prevTransform, &b.prevTransform, sizeof(a.prevTransform)));
            CHECK(rowGpu.GetInstanceSlotEntity(rowSlot.instance) == rows[i]);
        }
    };
    std::vector<size_t> live;
    for (size_t i = 0; i < 200; ++i)
        live.push_back(i);
    compare(live);
    compare(live);   // prevTransform is the frame before's

    // a move of every fourth: nodes through the setter, rows through the world (a version advance first, as a phase)
    rowsRig.world->AdvanceVersion();
    for (size_t i = 0; i < 200; i += 4)
    {
        locals[i].translation.x += 0.5;
        nodes[i]->SetTranslation(locals[i].translation);
        GetComponentForWrite<LocalTransform>(rowsRig.world, rows[i])->translation = locals[i].translation;
    }
    compare(live);
    compare(live);

    // a despawn of every fifth and a spawn of 5: the freed slots are reused in both builds alike
    std::vector<Entity> gone;
    std::vector<size_t> kept;
    for (size_t i = 0; i < 200; ++i)
    {
        if (i % 5 == 1)
        {
            gone.push_back(rows[i]);
            nodesRig.graph->Detach(nodes[i]);
        }
        else
            kept.push_back(i);
    }
    rowsRig.Destroy(gone);
    std::vector<LocalTransform> newLocals;
    for (int i = 0; i < 5; ++i)
    {
        newLocals.push_back(RandomLocal(rng));
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetTransform(&newLocals[i].translation, &newLocals[i].rotation, &newLocals[i].scaling);
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, nodeTwo.Get()));
        nodes.push_back(nodesRig.graph->Attach(nodesRig.graph->GetRootNode(), node));
        kept.push_back(nodes.size() - 1);
    }
    for (Entity row : rowsRig.Spawn(std::vector<MeshId>(5, two), newLocals))
        rows.push_back(row);
    for (size_t i = 200; i < 205; ++i)
    {
        const MeshInstance* instance = static_cast<const MeshInstance*>(nodes[i]->GetLeaf());
        CHECK(uint32_t(instance->GetInstanceIndex()) == rowsRig.Slot(rows[i]).instance);
    }
    compare(kept);
    compare(kept);
    rowsRig.graph->ReleaseRuntimeMesh(two);
    rowsRig.graph->ReleaseRuntimeMesh(one);
    CHECK(TakeWarnings() == 0);
}

namespace
{
    // What the interpolation must write, spelled out as test_scene_graph_simulation.cpp's DirectWorld (design 8.4).
    dm::daffine3 DirectWorld(const SimPosePrevious& previous, const SimPose& pose, const SimScale& scale, double alpha)
    {
        double t[3], w, x, y, z;
        if (alpha == 0.0 || alpha == 1.0)
        {
            const dm::double3& translation = alpha == 0.0 ? previous.translation : pose.translation;
            const dm::dquat& rotation = alpha == 0.0 ? previous.rotation : pose.rotation;
            for (int i = 0; i < 3; ++i)
                t[i] = translation[i];
            w = rotation.w, x = rotation.x, y = rotation.y, z = rotation.z;
        }
        else
        {
            const double rest = 1.0 - alpha;
            for (int i = 0; i < 3; ++i)
                t[i] = previous.translation[i] * rest + pose.translation[i] * alpha;
            const dm::dquat& a = previous.rotation;
            const dm::dquat& b = pose.rotation;
            const double dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
            const double side = dot < 0.0 ? -alpha : alpha;
            w = a.w * rest + b.w * side;
            x = a.x * rest + b.x * side;
            y = a.y * rest + b.y * side;
            z = a.z * rest + b.z * side;
            const double length = std::sqrt(w * w + x * x + y * y + z * z);
            if (length > 0.0)
                w = w / length, x = x / length, y = y / length, z = z / length;
        }
        const dm::double3& s = scale.scaling;
        return dm::daffine3(
            (1 - 2 * (y * y + z * z)) * s.x, (2 * (x * y + z * w)) * s.x, (2 * (x * z - y * w)) * s.x,
            (2 * (x * y - z * w)) * s.y, (1 - 2 * (x * x + z * z)) * s.y, (2 * (y * z + x * w)) * s.y,
            (2 * (x * z + y * w)) * s.z, (2 * (y * z - x * w)) * s.z, (1 - 2 * (x * x + y * y)) * s.z,
            t[0], t[1], t[2]);
    }

    // A toy engine over dense arrays (as test_scene_graph_simulation.cpp's BallDomain, without teleports): bodies that
    // drift and spin, addressed by the index OnEntitiesAdded stores in SimBodyRef, reset in OnEntitiesRemoved; poses
    // scattered in engine order (the E6.2a contract).
    class DriftDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DriftDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        struct Body
        {
            SimPose pose;
            SimPosePrevious previous;
            dm::double3 velocity;
            Entity entity;
        };
        ComponentId pose, previous, body, driven, tag;
        std::vector<ComponentAccess> inputs, outputs;
        std::vector<QueryTerm> terms;
        std::vector<Body> bodies;
        std::vector<uint32_t> freeIndices;
        uint32_t added = 0, removed = 0, steps = 0;

        explicit DriftDomain(IWorld* world)
        {
            pose = FindComponent<SimPose>(world);
            previous = FindComponent<SimPosePrevious>(world);
            body = FindComponent<SimBodyRef>(world);
            driven = FindComponent<SimulationDriven>(world);
            tag = FindComponent<RuntimeMeshInstance>(world);
            inputs = { { pose, Access::Read, {} }, { body, Access::Read, {} } };
            outputs = { { pose, Access::Write, {} }, { previous, Access::Write, {} }, { body, Access::Read, {} } };
            terms = { { driven, TermOp::All, Access::Filter, {} }, { tag, TermOp::All, Access::Filter, {} },
                      { body, TermOp::All, Access::Write, {} }, { pose, TermOp::All, Access::Read, {} } };
        }
        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override
        {
            retVal = SimulationDomainDesc{};
            retVal.name = "drift";
            retVal.inputs = inputs.data();
            retVal.inputCount = uint32_t(inputs.size());
            retVal.outputs = outputs.data();
            retVal.outputCount = uint32_t(outputs.size());
            retVal.lifetime.terms = terms.data();
            retVal.lifetime.termCount = uint32_t(terms.size());
            retVal.substeps = 1;
            retVal.maxStepsPerFrame = 4;
            retVal.fixedTimeStep = 0.25;
            retVal.flags = SimulationDomainFlags::Deterministic;
            return retVal;
        }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch& batch, ISimulationContext* context) noexcept override
        {
            SimBodyRef* refs = static_cast<SimBodyRef*>(batch.columns[0]);
            const SimPose* poses = static_cast<const SimPose*>(batch.columns[1]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                uint32_t index = uint32_t(bodies.size());
                if (!freeIndices.empty())
                {
                    index = freeIndices.back();
                    freeIndices.pop_back();
                }
                else
                    bodies.push_back(Body{});
                const double seed = double(GetEntityIndex(batch.entities[i]) % 13);
                bodies[index] = Body{ poses[i], SimPosePrevious{ poses[i].translation, poses[i].rotation },
                                      dm::double3(seed - 6.0, 1.0 + 0.5 * seed, 2.0), batch.entities[i] };
                refs[i].domain = uint32_t(context->GetDomainId());
                refs[i].index = index;
                ++added;
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch& batch, ISimulationContext*) noexcept override
        {
            SimBodyRef* refs = static_cast<SimBodyRef*>(batch.columns[0]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                EXPECT(refs[i].index < bodies.size() && bodies[refs[i].index].entity == batch.entities[i]);
                if (refs[i].index < bodies.size())
                {
                    bodies[refs[i].index].entity = Entity::Null;
                    freeIndices.push_back(refs[i].index);
                }
                refs[i] = SimBodyRef{};
                ++removed;
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT Step(double step, ISimulationContext*) noexcept override
        {
            ++steps;
            for (Body& b : bodies)
            {
                if (b.entity == Entity::Null)
                    continue;
                b.previous = SimPosePrevious{ b.pose.translation, b.pose.rotation };
                b.pose.translation += b.velocity * step;
                const dm::dquat& q = b.pose.rotation;
                const double c = std::cos(step), s = std::sin(step);   // a turn about y
                b.pose.rotation = dm::dquat(q.w * c - q.y * s, q.x * c - q.z * s, q.y * c + q.w * s, q.z * c + q.x * s);
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept override
        {
            if (context->GetStepsThisFrame() == 0)
                return nvrhi::FS_OK;
            ColumnView refs[4], poses[4], previousPoses[4];
            const uint32_t count = context->GetColumnCount(body);
            for (uint32_t first = 0; first < count; first += 4)
            {
                const uint32_t n = context->GetColumns(body, first, c_AllVersions, 4, refs);
                EXPECT(context->GetColumns(pose, first, c_AllVersions, 4, poses) == n);
                EXPECT(context->GetColumns(previous, first, c_AllVersions, 4, previousPoses) == n);
                for (uint32_t v = 0; v < n; ++v)
                {
                    for (uint32_t row = 0; row < refs[v].count; ++row)
                    {
                        const Body& b = bodies[static_cast<const SimBodyRef*>(refs[v].base)[row].index];
                        static_cast<SimPose*>(poses[v].base)[row] = b.pose;
                        static_cast<SimPosePrevious*>(previousPoses[v].base)[row] = b.previous;
                    }
                }
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT CaptureState(IStateWriter*) noexcept override { return nvrhi::FE_NOT_IMPLEMENT; }
        nvrhi::FRESULT RestoreState(IStateReader*) noexcept override { return nvrhi::FE_NOT_IMPLEMENT; }
    };
}

// Driven runtime rows (the plain components and SimulationDriven, SimPose, SimPosePrevious, SimScale, SimBodyRef):
// the domain takes them (OnEntitiesAdded writes SimBodyRef), scatters their poses, the interpolation writes their
// worlds (the LocalTransform they carry is not read), Bounds follows; the observer gives them slots like plain rows;
// removal returns the body (OnEntitiesRemoved) and the slots. A spawn writes both poses as the spawn transform, so the
// first frame shows it unsmeared.
void test_driven_rows()
{
    Rig rig;
    Assets assets;
    nvrhi::AutoPtr<DriftDomain> domain = MAKE_RC_OBJ_PTR(DriftDomain, rig.world);
    SimulationDomainId domainId = SimulationDomainId::Invalid;
    CHECK(rig.graph->GetScheduler()->AddSimulationDomain(domain, domainId) == nvrhi::FS_OK);
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.two);

    Rng rng(3);
    const uint32_t count = 120;
    std::vector<LocalTransform> locals;   // ignored: the pose is the row's
    std::vector<SimPose> poses;
    std::vector<SimPosePrevious> previous;
    std::vector<SimScale> scales;
    for (uint32_t i = 0; i < count; ++i)
    {
        locals.push_back(MakeLocal(dm::double3(-999.0)));
        const LocalTransform spawn = RandomLocal(rng);
        poses.push_back(SimPose{ spawn.translation, spawn.rotation });
        previous.push_back(SimPosePrevious{ spawn.translation, spawn.rotation });
        scales.push_back(SimScale{ spawn.scaling });
    }
    std::vector<ComponentId> ids = rig.driven;
    std::vector<MeshRef> refs(count, MeshRef{ mesh });
    std::vector<const void*> values(ids.size(), nullptr);
    values[1] = locals.data();
    values[6] = refs.data();
    values[10] = poses.data();
    values[11] = previous.data();
    values[12] = scales.data();
    EntityCreateDesc desc{};
    desc.components = ids.data();
    desc.values = values.data();
    desc.componentCount = uint32_t(ids.size());
    desc.entityCount = count;
    std::vector<Entity> rows(count);
    CHECK(rig.world->CreateEntities(desc, rows.data()) == nvrhi::FS_OK);
    CHECK(domain->added == count && rig.graph->GetRuntimeMeshInstanceCount() == count);
    for (uint32_t i = 0; i < count; ++i)
        CHECK(rig.Get<SimBodyRef>(rows[i])->domain == uint32_t(domainId) && rig.Slot(rows[i]).instance == i);

    auto check = [&](double alpha)
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!rig.world->IsAlive(rows[i]))
                continue;
            const dm::daffine3 expected =
                DirectWorld(*rig.Get<SimPosePrevious>(rows[i]), *rig.Get<SimPose>(rows[i]), *rig.Get<SimScale>(rows[i]),
                            alpha);
            CHECK(SameValue(rig.Get<WorldTransform>(rows[i])->transform, expected));
            const dm::box3 box = assets.two->objectSpaceBounds * dm::affine3(expected);
            CHECK(SameValue(rig.Get<WorldBounds>(rows[i])->bounds, box));
            CHECK(SameValue(*rig.Get<LocalTransform>(rows[i]), locals[i]));
        }
    };
    auto alpha = [&]()
    {
        const FixedStepEntry& entry = GetResource<FixedStepState>(rig.world)->domains[uint32_t(domainId)];
        return std::min(entry.accumulator / entry.fixedTimeStep, 1.0);
    };
    // the first frame (no step): the spawn transform, unsmeared
    rig.SimulateGraph(0.1);
    rig.Refresh();
    CHECK(domain->steps == 0);
    check(alpha());
    for (uint32_t i = 0; i < count; i += 17)
    {
        const dm::double3 shown = rig.Get<WorldTransform>(rows[i])->transform.m_translation;
        CHECK(std::abs(shown.x - poses[i].translation.x) < 1e-9 && std::abs(shown.z - poses[i].translation.z) < 1e-9);
    }
    for (int frame = 0; frame < 12; ++frame)
    {
        rig.SimulateGraph(0.25 * (0.6 + 0.1 * double(frame % 5)));
        rig.Refresh();
        check(alpha());
        if (frame == 5)
        {
            std::vector<Entity> gone(rows.begin(), rows.begin() + 30);
            rig.Destroy(gone);
            CHECK(domain->removed == 30 && rig.graph->GetRuntimeMeshInstanceCount() == count - 30);
        }
    }
    CHECK(domain->steps > 6 && rig.References(mesh) == 1 + count - 30);
    const uint64_t interpolated = rig.Store().GetCounters().interpolatedChunks;
    CHECK(interpolated > 0);
    rig.graph->ReleaseRuntimeMesh(mesh);
    CHECK(TakeWarnings() == 0 && g_Errors.load() == 0);
}

// Graph teardown with live runtime rows. The world released with the graph: nothing is delivered, the tables go whole,
// the meshes are released. The world kept: the rows live on as plain data with the slots and MeshRef they had; later
// removals and new tagged rows reach no graph.
void test_teardown()
{
    Assets assets;
    const uint64_t opaqueReferences = ReferenceCountOf(assets.opaque);
    for (bool keepWorld : { false, true })
    {
        nvrhi::AutoPtr<IWorld> kept;
        std::vector<Entity> rows;
        std::vector<GpuInstanceSlot> slots;
        MeshId mesh = MeshId::Invalid;
        std::vector<ComponentId> plain;
        {
            Rig rig;
            rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            mesh = rig.graph->AcquireRuntimeMesh(assets.two);
            rows = rig.Spawn(mesh, 70);
            rig.Frame();
            for (Entity row : rows)
                slots.push_back(rig.Slot(row));
            CHECK(ReferenceCountOf(assets.two) > 1);
            plain = rig.plain;
            if (keepWorld)
                kept = rig.world;
            rig.graph->OnMeshRemoved = [](const MeshInfo*) { CHECK(false); };
            rig.graph.Reset();
        }
        CHECK(ReferenceCountOf(assets.two) == 1 && ReferenceCountOf(assets.opaque) == opaqueReferences);
        if (!keepWorld)
            continue;
        for (size_t i = 0; i < rows.size(); ++i)
        {
            CHECK(kept->IsAlive(rows[i]));
            CHECK(SameValue(*GetComponent<GpuInstanceSlot>(kept, rows[i]), slots[i]));
            CHECK(GetComponent<MeshRef>(kept, rows[i])->mesh == mesh);
        }
        CHECK(kept->DestroyEntities(rows.data(), 35) == nvrhi::FS_OK);
        EntityCreateDesc desc{};
        desc.components = plain.data();
        desc.componentCount = uint32_t(plain.size());
        desc.entityCount = 3;
        Entity more[3];
        CHECK(kept->CreateEntities(desc, more) == nvrhi::FS_OK);
        CHECK(GetComponent<GpuInstanceSlot>(kept, more[0])->instance == c_NoInstanceSlot);
        kept.Reset();
    }
    CHECK(TakeWarnings() == 0);
}

// A copy from another world (CopyEntitiesFrom): the observer sees an OnAdd batch like any creation, with the source's
// MeshRef values, and cannot tell that the id is of another graph's table. So such copies are not supported (as
// SceneGraph.h says): an id that this table lacks gets no slot and a warning; one that this table has by chance names
// this table's mesh, silently. Both are fixed here as the documented behaviour.
void test_copy_from_another_world()
{
    Assets assets;
    auto other = MakeMesh({ assets.opaque }, 9.f);
    Rig source;
    const MeshId id = source.graph->AcquireRuntimeMesh(assets.two);
    std::vector<Entity> rows = source.Spawn(id, 4);

    // a target without meshes: no slot, one warning (the copies, made row by row, coalesce into one batch)
    Rig empty;
    std::vector<Entity> copies(4);
    CHECK(empty.world->CopyEntitiesFrom(source.world, rows.data(), 4, copies.data()) == nvrhi::FS_OK);
    CHECK(TakeWarnings() == 1 && empty.graph->GetRuntimeMeshInstanceCount() == 0);   // the copies are one batch
    for (Entity copy : copies)
        CHECK(!empty.HoldsSlot(copy) && empty.Get<MeshRef>(copy)->mesh == id);

    // a target whose table has a mesh under the same id: the copies take it, without a warning
    Rig target;
    const MeshId targetId = target.graph->AcquireRuntimeMesh(other);
    CHECK(targetId == id);
    CHECK(target.world->CopyEntitiesFrom(source.world, rows.data(), 4, copies.data()) == nvrhi::FS_OK);
    CHECK(TakeWarnings() == 0 && target.graph->GetRuntimeMeshInstanceCount() == 4 && target.References(targetId) == 5);
    for (Entity copy : copies)
    {
        CHECK(target.HoldsSlot(copy) && target.Slot(copy).geometryCount == 1);
        CHECK(SameValue(target.Get<LocalBounds>(copy)->bounds, other->objectSpaceBounds));
    }
    CHECK(source.References(id) == 5);
    source.graph->ReleaseRuntimeMesh(id);
    target.graph->ReleaseRuntimeMesh(targetId);
}

// A snapshot of the graph's world loads back into that graph. While the application holds its AcquireRuntimeMesh
// reference, the ids stay valid across the load's removal and re-addition: the rows get slots and references again.
// Without it, the removal frees the mesh (OnMeshRemoved), the re-added rows name a freed id: no slots (and an empty
// LocalBounds), warnings. With it, the world's content (and hash) is the saved one.
void test_snapshot_rewind()
{
    Assets assets;
    for (bool holdReference : { true, false })
    {
        Rig rig;
        const MeshId two = rig.graph->AcquireRuntimeMesh(assets.two);
        const MeshId one = rig.graph->AcquireRuntimeMesh(assets.one);
        Rng rng(31);
        std::vector<LocalTransform> locals;
        std::vector<MeshId> ids;
        for (int i = 0; i < 150; ++i)
        {
            locals.push_back(RandomLocal(rng));
            ids.push_back(i % 3 ? two : one);
        }
        std::vector<Entity> rows = rig.Spawn(ids, locals);
        rig.Frame();
        auto writer = MAKE_RC_OBJ_PTR(MemoryWriter);
        CHECK(rig.world->SaveSnapshot(writer) == nvrhi::FS_OK);
        const uint64_t savedHash = rig.world->ComputeHash();

        // the world moves on: despawns and spawns
        std::vector<Entity> gone(rows.begin() + 20, rows.begin() + 60);
        rig.Destroy(gone);
        rig.Spawn(one, 30, 500.0);
        rig.Frame();
        CHECK(rig.world->ComputeHash() != savedHash);
        const uint32_t removedBefore = rig.meshesRemoved;
        if (!holdReference)
        {
            rig.graph->ReleaseRuntimeMesh(two);
            rig.graph->ReleaseRuntimeMesh(one);
        }

        auto reader = MAKE_RC_OBJ_PTR(MemoryReader, writer->bytes);
        CHECK(rig.world->LoadSnapshot(reader) == nvrhi::FS_OK);
        // The saved content once the moved rows' chunks have their boxes again (the observer gave the rows their
        // GpuInstanceSlot column, which is not saved, so they are in new chunks); without a mesh the rows' derived
        // LocalBounds is empty.
        rig.Refresh();
        CHECK((rig.world->ComputeHash() == savedHash) == holdReference);
        for (Entity row : rows)
            CHECK(rig.world->IsAlive(row));
        if (holdReference)
        {
            CHECK(TakeWarnings() == 0 && rig.meshesRemoved == removedBefore);
            CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 150 && rig.References(two) == 101 &&
                  rig.References(one) == 51);
            std::set<uint32_t> slots;
            for (Entity row : rows)
                slots.insert(rig.Slot(row).instance);
            CHECK(slots.size() == 150 && *slots.rbegin() == 149);
            rig.Frame();
            for (size_t i = 0; i < rows.size(); i += 11)
                CHECK(SameValue(rig.Get<WorldTransform>(rows[i])->transform, ComposeTransform(locals[i])));
            rig.graph->ReleaseRuntimeMesh(two);
            rig.graph->ReleaseRuntimeMesh(one);
        }
        else
        {
            CHECK(TakeWarnings() == 1 && rig.meshesRemoved == removedBefore + 2);   // one chunk, one batch
            CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 0 && rig.graph->GetMeshes().empty());
            for (Entity row : rows)
                CHECK(!rig.HoldsSlot(row) && rig.Get<LocalBounds>(row)->bounds.isempty());
            rig.Frame();
        }
        CHECK(TakeWarnings() == 0);
    }
}

namespace
{
    // An application observer that destroys the rows it sees lose RuntimeMeshInstance (while armed), through its
    // commands: during a LoadSnapshot they play back after the load, against the loaded entities.
    class DestroyingObserver final : public nvrhi::ObjectImpl<IObserver>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DestroyingObserver)
        NVRHI_IMPLEMENTS_INTERFACE(IObserver)
        NVRHI_END_INTERFACE_TABLE()

        bool armed = false;
        nvrhi::FRESULT OnEntities(ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) noexcept override
        {
            if (armed)
                commands->DestroyEntities(batch.entities, batch.count, 0);
            armed = false;   // once: its own destroys raise it again
            return nvrhi::FS_OK;
        }
    };
}

// A failed command of an observer's cascade fails no scheduler run or load; the graph's world validates in Debug
// builds only, with one warning per dead row.
#ifdef NDEBUG
constexpr uint32_t c_WarningsPerDeadRow = 0;
#else
constexpr uint32_t c_WarningsPerDeadRow = 1;
#endif

// Slots and references follow membership ({RuntimeMeshInstance, GpuInstanceSlot, MeshRef, no SceneNode}), not the tag's
// events (the E7.1 review, S1): (a) rows made without GpuInstanceSlot and destroyed by the same playback before the
// column is added hold nothing; (b) the slot column removed from a live row releases its slots and reference, added
// again takes new ones; (c) rows that an application observer destroys during a LoadSnapshot, before the column is
// added, hold nothing; (d) MeshRef removed from a member releases.
void test_membership()
{
    Rig rig;
    Assets assets;
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.two);
    std::vector<ComponentId> noSlot(rig.plain.begin(), rig.plain.end() - 1);

    // (a)
    bool spawn = true;
    rig.AddSystem("spawn and destroy", Phase::PreSimulation, {}, [&](ISystemContext* context)
    {
        if (!spawn)
            return nvrhi::FS_OK;
        spawn = false;
        std::vector<LocalTransform> locals(20, MakeLocal(dm::double3(1.0)));
        std::vector<MeshRef> refs(20, MeshRef{ mesh });
        std::vector<const void*> values(noSlot.size(), nullptr);
        values[1] = locals.data();
        values[6] = refs.data();
        EntityCreateDesc desc{};
        desc.components = noSlot.data();
        desc.values = values.data();
        desc.componentCount = uint32_t(noSlot.size());
        desc.entityCount = 20;
        Entity placeholders[20];
        ICommandBuffer* commands = context->GetCommandBuffer();
        commands->CreateEntities(desc, MakeSortKey(0, 0, 0), placeholders);
        commands->DestroyEntities(placeholders, 20, MakeSortKey(1, 0, 0));
        return nvrhi::FS_OK;
    });
    rig.time += 0.01;
    // the graph's column addition finds the rows dead: the playback reports it (SceneGraph.h)
    const nvrhi::FRESULT simulated = rig.graph->Simulate(FrameTime{ rig.time, 0.01, rig.frame++ });
    const uint32_t simulateWarnings = TakeWarnings();
    std::printf("membership (a): Simulate 0x%08x, %u warnings\n", uint32_t(simulated), simulateWarnings);
    CHECK(simulated == nvrhi::FS_OK && simulateWarnings == 20 * c_WarningsPerDeadRow);
    CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 0 && rig.References(mesh) == 1);
    CHECK(rig.graph->GetInstanceSlotCount() == 0 && rig.graph->GetGeometryInstancesCount() == 0);

    // (b)
    std::vector<Entity> rows = rig.Spawn(mesh, 6);
    const ComponentId slotId = FindComponent<GpuInstanceSlot>(rig.world);
    CHECK(rig.world->RemoveComponent(rows[2], slotId) == nvrhi::FS_OK);
    CHECK(rig.References(mesh) == 6 && rig.graph->GetRuntimeMeshInstanceCount() == 5);
    CHECK(!rig.Get<GpuInstanceSlot>(rows[2]));
    CHECK(rig.world->AddComponent(rows[2], slotId, nullptr) == nvrhi::FS_OK);
    CHECK(rig.References(mesh) == 7 && rig.Slot(rows[2]).instance == 2 && rig.Slot(rows[2]).geometryCount == 2);
    rig.Destroy(rows);
    CHECK(rig.References(mesh) == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 0);

    // (c)
    rows = rig.Spawn(mesh, 30);
    rig.Frame();
    auto writer = MAKE_RC_OBJ_PTR(MemoryWriter);
    CHECK(rig.world->SaveSnapshot(writer) == nvrhi::FS_OK);
    auto destroyer = MAKE_RC_OBJ_PTR(DestroyingObserver);
    ObserverDesc desc{};
    desc.debugName = "destroyer";
    desc.component = FindComponent<RuntimeMeshInstance>(rig.world);
    desc.access = Access::Filter;
    desc.event = ObserverEvent::OnRemove;
    ObserverId id = ObserverId::Invalid;
    CHECK(rig.world->AddObserver(desc, destroyer, id) == nvrhi::FS_OK);
    destroyer->armed = true;
    auto reader = MAKE_RC_OBJ_PTR(MemoryReader, writer->bytes);
    CHECK(rig.world->LoadSnapshot(reader) == nvrhi::FS_OK);
    destroyer->armed = false;
    // the column additions find the rows dead: a warning per row and the load's report of its observers' failure
    const uint32_t loadWarnings = TakeWarnings();
    std::printf("membership (c): %u warnings\n", loadWarnings);
    CHECK(loadWarnings == 31 * c_WarningsPerDeadRow);
    for (Entity row : rows)
        CHECK(!rig.world->IsAlive(row));
    CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 0 && rig.References(mesh) == 1);
    CHECK(rig.world->RemoveObserver(id) == nvrhi::FS_OK);
    rows = rig.Spawn(mesh, 2);
    CHECK(rig.Slot(rows[0]).instance == 0 && rig.Slot(rows[1]).instance == 1);

    // (d) MeshRef removed from a member: it stops being one while the value is readable, and releases; added again
    const ComponentId refId = FindComponent<MeshRef>(rig.world);
    CHECK(rig.world->RemoveComponent(rows[1], refId) == nvrhi::FS_OK);
    CHECK(rig.References(mesh) == 2 && rig.graph->GetRuntimeMeshInstanceCount() == 1 && !rig.HoldsSlot(rows[1]));
    const MeshRef ref{ mesh };
    CHECK(rig.world->AddComponent(rows[1], refId, &ref) == nvrhi::FS_OK);
    CHECK(rig.References(mesh) == 3 && rig.Slot(rows[1]).instance == 1);
    CHECK(rig.world->RemoveComponent(rows[1], refId) == nvrhi::FS_OK);
    rig.Destroy(rows);   // the row without MeshRef holds nothing to release
    CHECK(rig.References(mesh) == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 0);
    rig.graph->ReleaseRuntimeMesh(mesh);
    CHECK(TakeWarnings() == 0);
}

// The tag on a node's row (the E7.1 review, S2) is ignored: the leaf keeps its slots and its one reference, the row
// stays at its depth below its parent and in its parent's subgraph, a node without a mesh gets no slot column; one
// warning per archetype.
void test_tag_on_node_rows()
{
    Rig rig;
    Assets assets;
    rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
    auto parentNode = MAKE_RC_OBJ_PTR(SceneGraphNode);
    parentNode->SetTranslation(dm::double3(10.0, 0.0, 0.0));
    SceneGraphNode* parent = rig.graph->Attach(rig.graph->GetRootNode(), parentNode);
    auto childNode = MAKE_RC_OBJ_PTR(SceneGraphNode);
    childNode->SetTranslation(dm::double3(1.0, 0.0, 0.0));
    childNode->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, assets.two.Get()));
    SceneGraphNode* child = rig.graph->Attach(parent, childNode);
    rig.Refresh();
    const MeshInstance* leaf = static_cast<const MeshInstance*>(child->GetLeaf());
    const int slot = leaf->GetInstanceIndex();
    const MeshId id = rig.graph->GetMeshes().Find(assets.two);
    CHECK(slot == 0 && rig.References(id) == 1 && rig.meshesAdded == 1 && rig.materialsAdded == 2);

    const ComponentId tag = FindComponent<RuntimeMeshInstance>(rig.world);
    CHECK(rig.world->AddComponent(rig.graph->GetEntity(child), tag, nullptr) == nvrhi::FS_OK);
    CHECK(rig.world->AddComponent(rig.graph->GetEntity(parent), tag, nullptr) == nvrhi::FS_OK);
    CHECK(!rig.world->HasComponent(rig.graph->GetEntity(parent), FindComponent<GpuInstanceSlot>(rig.world)));
    CHECK(leaf->GetInstanceIndex() == slot && rig.References(id) == 1 && rig.graph->GetRuntimeMeshInstanceCount() == 0);
    parent->SetTranslation(dm::double3(20.0, 0.0, 0.0));
    rig.Refresh();
    CHECK(TakeWarnings() == 2);   // the two archetypes
    CHECK(child->GetLocalToWorldTransform().m_translation.x == 21.0);
    CHECK(rig.graph->GetRootNode()->GetGlobalBoundingBox().m_maxs.x > 20.f);
    CHECK((rig.graph->GetRootNode()->GetSubgraphContentFlags() & SceneContentFlags::BlendedMeshes) ==
          SceneContentFlags::BlendedMeshes);
    rig.graph->Detach(parent);
    CHECK(rig.graph->GetMeshes().empty() && rig.graph->GetInstanceSlotCount() == 1);
    CHECK(TakeWarnings() == 0);
}

// Asset callbacks that call back into the graph (the E7.1 review, S3): OnMeshAdded acquiring another mesh, OnMeshRemoved
// destroying runtime rows of another mesh (whose last reference goes inside), and a callback of ReconcileMeshes
// acquiring a mesh. Each nesting level collects its own changes: every event comes once, the tables stay consistent.
void test_reentrant_callbacks()
{
    Rig rig;
    Assets assets;
    auto third = MakeMesh({ assets.blended }, 2.f);
    std::vector<const MeshInfo*> added, removed;
    MeshId nested = MeshId::Invalid;
    std::vector<Entity> victims;
    rig.graph->OnMeshAdded = [&](const MeshInfo* mesh)
    {
        added.push_back(mesh);
        if (mesh == assets.two.Get())
            nested = rig.graph->AcquireRuntimeMesh(assets.one);
    };
    rig.graph->OnMeshRemoved = [&](const MeshInfo* mesh)
    {
        removed.push_back(mesh);
        if (mesh == assets.two.Get() && !victims.empty())
        {
            const std::vector<Entity> rows = victims;
            victims.clear();
            rig.Destroy(rows);
        }
    };
    const MeshId two = rig.graph->AcquireRuntimeMesh(assets.two);
    CHECK(two != MeshId::Invalid && nested != MeshId::Invalid && added.size() == 2);
    CHECK(added[0] == assets.two.Get() && added[1] == assets.one.Get());
    CHECK(rig.References(two) == 1 && rig.References(nested) == 1);
    CHECK(rig.materialsAdded == 2);   // the outer events (two's materials) survived the nested call

    victims = rig.Spawn(nested, 5);
    rig.graph->ReleaseRuntimeMesh(nested);   // the rows hold it now
    CHECK(rig.References(nested) == 5);
    rig.graph->ReleaseRuntimeMesh(two);   // its removal destroys the rows, whose removal removes the other mesh
    CHECK(removed.size() == 2 && removed[0] == assets.two.Get() && removed[1] == assets.one.Get());
    CHECK(rig.graph->GetMeshes().empty() && rig.graph->GetMaterials().empty());
    CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 0 && rig.materialsRemoved == 2);

    // ReconcileMeshes: a geometry with a new material edited into an attached mesh; its OnMaterialAdded acquires a mesh
    rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
    auto edited = MakeMesh({ assets.opaque }, 1.f);
    auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
    node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, edited.Get()));
    rig.graph->Attach(rig.graph->GetRootNode(), node);
    rig.Refresh();
    MeshId fromReconcile = MeshId::Invalid;
    rig.graph->OnMaterialAdded = [&](const Material* material)
    {
        if (material == assets.blended.Get())
            fromReconcile = rig.graph->AcquireRuntimeMesh(third);
    };
    auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
    geometry->material = assets.blended;
    edited->geometries.push_back(geometry);
    rig.graph->Attach(rig.graph->GetRootNode(), MAKE_RC_OBJ_PTR(SceneGraphNode));   // a structural change
    rig.Refresh();
    CHECK(fromReconcile != MeshId::Invalid && rig.References(fromReconcile) == 1);
    CHECK(added.back() == third.Get() && rig.graph->GetMeshes().size() == 2);
    rig.graph->ReleaseRuntimeMesh(fromReconcile);
    CHECK(removed.back() == third.Get() && rig.graph->GetMeshes().size() == 1);
    rig.graph->OnMaterialAdded = nullptr;
    CHECK(TakeWarnings() == 0);
}

// Parent naming the root (depth 0; the E7.1 review, N1): the runtime rows' chunks would be merged into the root's
// subgraph in the same depth-0 pass that finalises it. Their archetype is made before the root's, so that in a serial
// pass its chunks come first, and at 8 workers above the serial threshold they run beside the root's: the root's
// content flags never show the runtime rows'.
void test_parent_is_root()
{
    for (uint32_t serialRows : { 1u << 30, 4096u })
    {
        SceneGraphStore::SetSerialSubgraphRows(serialRows);
        Rig rig(8);
        Assets assets;
        const MeshId mesh = rig.graph->AcquireRuntimeMesh(MakeMesh({ assets.blended }, 1.f));
        const Parent none{ Entity::Null };
        const ComponentId parentId = FindComponent<Parent>(rig.world);
        rig.Destroy(rig.Spawn({ mesh }, { MakeLocal(dm::double3(0.0)) }, { { parentId, &none } }));
        CHECK(TakeWarnings() == 1);
        rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        for (int i = 0; i < 3; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, assets.one.Get()));
            rig.graph->Attach(rig.graph->GetRootNode(), node);
        }
        const uint32_t count = 6000;
        std::vector<Parent> parents(count, Parent{ rig.graph->GetEntity(rig.graph->GetRootNode()) });
        std::vector<LocalTransform> locals;
        for (uint32_t i = 0; i < count; ++i)
            locals.push_back(MakeLocal(dm::double3(5000.0 + double(i), 0.0, 0.0)));
        rig.Spawn(std::vector<MeshId>(count, mesh), locals, { { parentId, parents.data() } });
        CHECK(TakeWarnings() >= 1);   // one per chunk's batch
        for (int frame = 0; frame < 4; ++frame)
        {
            rig.Refresh();
            CHECK((rig.graph->GetRootNode()->GetSubgraphContentFlags() & SceneContentFlags::BlendedMeshes) ==
                  SceneContentFlags::None);
            CHECK(rig.graph->GetRootNode()->GetGlobalBoundingBox().m_maxs.x < 100.f);
            rig.graph->Attach(rig.graph->GetRootNode(), MAKE_RC_OBJ_PTR(SceneGraphNode));   // a full pass again
        }
        rig.graph->ReleaseRuntimeMesh(mesh);
    }
    SceneGraphStore::SetSerialSubgraphRows(4096);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------- the golden script

namespace
{
    constexpr uint32_t c_ScriptFrames = 200;

    struct FrameRecord
    {
        uint64_t world;     // IWorld::ComputeHash
        uint64_t slots;     // every runtime row's entity, slot and mesh, in chunk order
        uint64_t gpu;       // GpuScene's instance rows of the runtime rows and the nodes
        uint32_t rows;      // GetRuntimeMeshInstanceCount
        uint32_t references;
    };

    // Each frame: a PreSimulation system despawns rows whose time has come (those with despawnFrame % 4 == 0 are left
    // to the EndFrame system, so they are seen by one more Refresh), spawns 40-56 rows of two meshes in runs, sometimes
    // instantiates a live row, and moves every row in parallel by chunk; four nodes with leaves move through the API.
    // Hashes after the frame's Refresh, GpuScene update and EndFrame.
    std::vector<FrameRecord> RunScript(uint32_t workers)
    {
        Rig rig(workers);
        Assets assets;
        rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        std::vector<SceneGraphNode*> nodes;
        for (int i = 0; i < 4; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, assets.one.Get()));
            nodes.push_back(rig.graph->Attach(rig.graph->GetRootNode(), node));
        }
        const MeshId meshes[2] = { rig.graph->AcquireRuntimeMesh(assets.two),
                                   rig.graph->AcquireRuntimeMesh(assets.one) };
        const ComponentId lifetimeId = RegisterComponent<Lifetime>(rig.world);
        const ComponentId localId = FindComponent<LocalTransform>(rig.world);
        Query<Read<Lifetime>> lifetimes(rig.world, "lifetimes");
        Query<Write<LocalTransform>, With<RuntimeMeshInstance>, Without<SimulationDriven>> movers(rig.world, "movers");
        uint32_t frame = 0;
        Rng rng(1234);

        std::vector<ComponentId> ids = rig.plain;
        ids.push_back(lifetimeId);
        std::vector<LocalTransform> locals;
        std::vector<MeshRef> refs;
        std::vector<Lifetime> lives;
        std::vector<Entity> placeholders;
        rig.AddSystem("script", Phase::PreSimulation,
                      { { localId, Access::Write, {} }, { lifetimeId, Access::Read, {} } },
                      [&](ISystemContext* context)
        {
            ICommandBuffer* commands = context->GetCommandBuffer();
            lifetimes.ForEachChunk(context, [&](const ChunkView& view, const Lifetime* life)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                {
                    if (life[r].despawnFrame <= frame && life[r].despawnFrame % 4 != 0)
                        commands->DestroyEntity(view.entities[r], MakeSortKey(0, view.chunkIndex, r));
                }
            });
            const uint32_t spawn = 40 + rng.Below(17);
            locals.clear();
            refs.clear();
            lives.clear();
            for (uint32_t i = 0; i < spawn; ++i)
            {
                locals.push_back(RandomLocal(rng, 200.0));
                refs.push_back(MeshRef{ meshes[(i / 7) % 2] });
                lives.push_back(Lifetime{ frame + 2 + rng.Below(24) });
            }
            std::vector<const void*> values(ids.size(), nullptr);
            values[1] = locals.data();
            values[6] = refs.data();
            values.back() = lives.data();
            EntityCreateDesc desc{};
            desc.components = ids.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(ids.size());
            desc.entityCount = spawn;
            placeholders.resize(spawn);
            commands->CreateEntities(desc, MakeSortKey(1, 0, 0), placeholders.data());
            if (frame % 7 == 3)
            {
                Entity prefab = Entity::Null;
                lifetimes.ForEachChunk(context, [&](const ChunkView& view, const Lifetime* life)
                {
                    for (uint32_t r = view.count / 2; r < view.count && prefab == Entity::Null; ++r)
                    {
                        if (life[r].despawnFrame > frame + 1)
                            prefab = view.entities[r];
                    }
                });
                Entity copies[3];
                if (prefab != Entity::Null)
                    commands->Instantiate(prefab, 3, MakeSortKey(2, 0, 0), copies);
            }
            const double step = 0.01 * double(frame % 9);
            movers.ParallelForChunks(context, 1, [step](const ChunkView& view, LocalTransform* local)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                {
                    local[r].translation.x += step;
                    local[r].translation.y -= 0.5 * step;
                }
            });
            return nvrhi::FS_OK;
        });
        rig.AddSystem("despawn at end", Phase::EndFrame, { { lifetimeId, Access::Read, {} } },
                      [&](ISystemContext* context)
        {
            ICommandBuffer* commands = context->GetCommandBuffer();
            lifetimes.ForEachChunk(context, [&](const ChunkView& view, const Lifetime* life)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                {
                    if (life[r].despawnFrame <= frame && life[r].despawnFrame % 4 == 0)
                        commands->DestroyEntity(view.entities[r], MakeSortKey(0, view.chunkIndex, r));
                }
            });
            return nvrhi::FS_OK;
        });

        GpuScene gpu(nullptr, CpuDesc());
        std::vector<FrameRecord> records;
        for (frame = 0; frame < c_ScriptFrames; ++frame)
        {
            nodes[frame % 4]->SetTranslation(dm::double3(double(frame), 0.5 * double(frame % 5), 1.0));
            rig.SimulateGraph(1.0 / 60.0);
            rig.Refresh();
            CHECK(NVRHI_SUCCEEDED(gpu.Update(rig.graph)));
            FrameRecord record{};
            Hasher slots, rows;
            Query<Read<GpuInstanceSlot>, Read<MeshRef>, With<RuntimeMeshInstance>> all(rig.world, "all rows");
            all.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, const GpuInstanceSlot* slot, const MeshRef* ref)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                {
                    slots.Add(view.entities[r]);
                    slots.Add(slot[r]);
                    slots.Add(ref[r]);
                    CHECK(slot[r].instance < gpu.GetInstanceRowCount());
                    rows.Add(gpu.GetInstanceRows()[slot[r].instance]);
                }
            });
            for (const auto& instance : rig.graph->GetMeshInstances())
                rows.Add(gpu.GetInstanceRows()[instance->GetInstanceIndex()]);
            rig.EndFrame();
            record.world = rig.world->ComputeHash();
            record.slots = slots.value;
            record.gpu = rows.value;
            record.rows = rig.graph->GetRuntimeMeshInstanceCount();
            record.references = rig.References(meshes[0]) + rig.References(meshes[1]);
            CHECK(record.references == record.rows + 4 + 2);   // the nodes' leaves and the application's
            records.push_back(record);
        }
        rig.graph->ReleaseRuntimeMesh(meshes[0]);
        rig.graph->ReleaseRuntimeMesh(meshes[1]);
        return records;
    }

    void CheckScriptGolden(const std::vector<FrameRecord>& frames)
    {
        std::string actual = "# the runtime-instance script (test_runtime_instances.cpp; ECS_UPDATE_GOLDEN=1 rewrites "
                             "it)\n# frame world slots gpu-rows runtime-rows references\n";
        for (size_t f = 0; f < frames.size(); ++f)
        {
            char line[128];
            std::snprintf(line, sizeof(line), "%u %016llx %016llx %016llx %u %u\n", uint32_t(f),
                          (unsigned long long)frames[f].world, (unsigned long long)frames[f].slots,
                          (unsigned long long)frames[f].gpu, frames[f].rows, frames[f].references);
            actual += line;
        }
        const std::filesystem::path path =
            std::filesystem::path(DONUT_TEST_GOLDEN_DIR) / "runtime_instances_frames.txt";
        const char* update = std::getenv("ECS_UPDATE_GOLDEN");
        if (update && std::strcmp(update, "1") == 0)
        {
            std::ofstream file(path, std::ios::binary);
            file << actual;
            CHECK(file.good());
            std::printf("wrote %s\n", path.generic_string().c_str());
            return;
        }
        std::ifstream file(path, std::ios::binary);
        std::stringstream expected;
        expected << file.rdbuf();
        if (expected.str() != actual)
        {
            std::istringstream a(actual), e(expected.str());
            std::string lineA, lineE;
            while (std::getline(a, lineA))
            {
                if (!std::getline(e, lineE) || lineA != lineE)
                    break;
            }
            std::printf("the frames differ from %s, first:\n  expected %s\n  actual   %s\n",
                        path.generic_string().c_str(), lineE.c_str(), lineA.c_str());
            CHECK(false);
        }
    }
}

// The script's frames at 1, 2 and 8 workers are the same, and those of the golden file (Release and Debug agree).
void test_script_golden()
{
    const std::vector<FrameRecord> first = RunScript(2);
    uint32_t changed = 0;
    for (uint32_t f = 1; f < c_ScriptFrames; ++f)
        changed += first[f].world != first[f - 1].world ? 1 : 0;
    CHECK(changed == c_ScriptFrames - 1 && first.back().rows > 300);
    for (uint32_t workers : { 1u, 8u })
    {
        const std::vector<FrameRecord> other = RunScript(workers);
        CHECK(other.size() == first.size());
        for (size_t f = 0; f < first.size(); ++f)
            CHECK(SameValue(other[f], first[f]));
    }
    CheckScriptGolden(first);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------- the Static tag (E7.4 remedy 3, part I5)

// SceneGraph::SetStaticTag is off by default and read when a graph is made. Two rigs built and run alike, the second
// made with it on: Static runtime rows and Static nodes (one with a moving child) beside moving rows and nodes; rows
// spawned into and destroyed from the Static archetype; a content change. World transforms, world bounds, chunk bounds
// and subgraph boxes are equal with it on and off, and the second graph leaves its Static archetypes out. Then a write
// of a Static row's LocalTransform: SetTransform is refused in Debug with a warning (written in Release); a write through
// the world is not propagated (a warning in Debug); neither moves the row there. The first graph moves both.
void test_static_tag()
{
    CHECK(!SceneGraph::GetStaticTag());
    Assets assets;
    std::unique_ptr<Rig> rigs[2];
    rigs[0] = std::make_unique<Rig>(4);
    SceneGraph::SetStaticTag(true);
    rigs[1] = std::make_unique<Rig>(4);
    SceneGraph::SetStaticTag(false);
    CHECK(!SceneGraph::GetStaticTag());

    struct Content
    {
        ComponentId staticId = ComponentId::Invalid;
        MeshId mesh = MeshId::Invalid;
        std::vector<Entity> staticRows;
        std::vector<Entity> movingRows;
        std::vector<SceneGraphNode*> staticNodes;
        std::vector<SceneGraphNode*> movingNodes;
        Query<Write<LocalTransform>, With<RuntimeMeshInstance>, Without<Static>> movers;
    };
    Content content[2];
    auto spawnStatic = [&](Rig& rig, Content& c, uint32_t count, double x0)
    {
        std::vector<LocalTransform> locals;
        for (uint32_t i = 0; i < count; ++i)
            locals.push_back(MakeLocal(dm::double3(x0 + 2.0 * double(i), 3.0, -1.0), dm::dquat::identity(), 0.5));
        const std::vector<Entity> rows = rig.Spawn(std::vector<MeshId>(count, c.mesh), locals,
                                                   { { c.staticId, nullptr } });
        c.staticRows.insert(c.staticRows.end(), rows.begin(), rows.end());
    };
    for (uint32_t g = 0; g < 2; ++g)
    {
        Rig& rig = *rigs[g];
        Content& c = content[g];
        c.staticId = RegisterComponent<Static>(rig.world);
        c.mesh = rig.graph->AcquireRuntimeMesh(assets.two);
        rig.graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        spawnStatic(rig, c, 300, 0.0);
        std::vector<LocalTransform> locals;
        for (uint32_t i = 0; i < 200; ++i)
            locals.push_back(MakeLocal(dm::double3(double(i), -4.0, 6.0)));
        c.movingRows = rig.Spawn(std::vector<MeshId>(200, c.mesh), locals);
        for (int i = 0; i < 30; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(dm::double3(double(i), 10.0, 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, assets.one.Get()));
            SceneGraphNode* attached = rig.graph->Attach(rig.graph->GetRootNode(), node);
            if (i % 3 != 0)
            {
                c.movingNodes.push_back(attached);
                continue;
            }
            CHECK(rig.world->AddComponent(rig.graph->GetEntity(attached), c.staticId, nullptr) == nvrhi::FS_OK);
            c.staticNodes.push_back(attached);
            if (i == 0)
            {
                auto child = MAKE_RC_OBJ_PTR(SceneGraphNode);
                child->SetTranslation(dm::double3(0.0, 1.0, 0.0));
                child->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, assets.one.Get()));
                c.movingNodes.push_back(rig.graph->Attach(attached, child));
            }
        }
        CHECK(c.movers.Create(rig.world, "static_tag movers") == nvrhi::FS_OK);
        const ComponentId localId = FindComponent<LocalTransform>(rig.world);
        Query<Write<LocalTransform>, With<RuntimeMeshInstance>, Without<Static>>* movers = &c.movers;
        rig.AddSystem("move", Phase::PreSimulation, { { localId, Access::Write, {} } }, [movers](ISystemContext* context)
        {
            movers->ParallelForChunks(context, 1, [](const ChunkView& view, LocalTransform* local)
            {
                for (uint32_t r = 0; r < view.count; ++r)
                    local[r].translation.y += 1.0;
            });
            return nvrhi::FS_OK;
        });
    }

    // Every row's and node's results, the two graphs against each other.
    auto compare = [&]()
    {
        const ComponentId chunkBoundsId = FindComponent<ChunkWorldBounds>(rigs[0]->world);
        for (int list = 0; list < 2; ++list)
        {
            const std::vector<Entity>& a = list ? content[0].movingRows : content[0].staticRows;
            const std::vector<Entity>& b = list ? content[1].movingRows : content[1].staticRows;
            CHECK(a.size() == b.size());
            for (size_t i = 0; i < a.size(); ++i)
            {
                CHECK(a[i] == b[i]);
                CHECK(SameValue(*rigs[0]->Get<WorldTransform>(a[i]), *rigs[1]->Get<WorldTransform>(b[i])));
                CHECK(SameValue(*rigs[0]->Get<WorldBounds>(a[i]), *rigs[1]->Get<WorldBounds>(b[i])));
                CHECK(SameBits(rigs[0]->world->GetComponent(a[i], chunkBoundsId),
                               rigs[1]->world->GetComponent(b[i], chunkBoundsId), sizeof(ChunkWorldBounds)));
            }
        }
        for (int list = 0; list < 2; ++list)
        {
            const std::vector<SceneGraphNode*>& a = list ? content[0].movingNodes : content[0].staticNodes;
            const std::vector<SceneGraphNode*>& b = list ? content[1].movingNodes : content[1].staticNodes;
            for (size_t i = 0; i < a.size(); ++i)
            {
                CHECK(SameValue(a[i]->GetLocalToWorldTransform(), b[i]->GetLocalToWorldTransform()));
                CHECK(SameValue(a[i]->GetGlobalBoundingBox(), b[i]->GetGlobalBoundingBox()));
            }
        }
        CHECK(SameValue(rigs[0]->graph->GetRootNode()->GetGlobalBoundingBox(),
                        rigs[1]->graph->GetRootNode()->GetGlobalBoundingBox()));
    };

    for (uint32_t f = 1; f <= 8; ++f)
    {
        for (uint32_t g = 0; g < 2; ++g)
        {
            Rig& rig = *rigs[g];
            Content& c = content[g];
            c.movingNodes[f % c.movingNodes.size()]->SetTranslation(dm::double3(double(f), 20.0, 0.0));
            if (f == 3)
                spawnStatic(rig, c, 50, 1000.0);   // rows join the Static archetype: computed once
            if (f == 5)
            {
                // rows leave it: the last rows move into the holes
                rig.Destroy(std::vector<Entity>(c.staticRows.begin() + 10, c.staticRows.begin() + 30));
                c.staticRows.erase(c.staticRows.begin() + 10, c.staticRows.begin() + 30);
            }
            if (f == 6)
                c.staticNodes[2]->InvalidateContent();
            rig.Frame();
        }
        compare();
    }
    const SceneGraphStore::Counters& off = rigs[0]->Store().GetCounters();
    const SceneGraphStore::Counters& on = rigs[1]->Store().GetCounters();
    CHECK(off.staticSkips == 0 && on.staticSkips > 8);
    std::printf("  equal over 8 frames; Static archetypes left out %llu times\n", (unsigned long long)on.staticSkips);
    CHECK(TakeWarnings() == 0);

    // Writes of a Static row's LocalTransform.
#ifndef NDEBUG
    const bool debug = true;
#else
    const bool debug = false;
#endif
    for (uint32_t g = 0; g < 2; ++g)
    {
        Rig& rig = *rigs[g];
        Content& c = content[g];
        const bool honoured = g == 1;
        SceneGraphNode* node = c.staticNodes[1];
        const dm::daffine3 nodeWorld = node->GetLocalToWorldTransform();
        node->SetTranslation(dm::double3(500.0, 0.0, 0.0));
        const bool written = node->GetTranslation().x == 500.0;
        CHECK(written == (!honoured || !debug));
        CHECK(TakeWarnings() == (honoured && debug ? 1u : 0u));
        rig.Frame();
        CHECK(SameValue(node->GetLocalToWorldTransform(), nodeWorld) == honoured);

        const Entity row = c.staticRows[7];
        const WorldTransform rowWorld = *rig.Get<WorldTransform>(row);
        rig.world->AdvanceVersion();
        auto* local = static_cast<LocalTransform*>(
            rig.world->GetComponentForWrite(row, FindComponent<LocalTransform>(rig.world), c_WorldVersion));
        CHECK(local);
        local->translation.x += 100.0;
        rig.Frame();
        CHECK(SameValue(*rig.Get<WorldTransform>(row), rowWorld) == honoured);
        CHECK(TakeWarnings() == (honoured && debug ? 1u : 0u));
    }
    CHECK(TakeWarnings() == 0);
}

// ----------------------------------------------------------------------------------------------- allocations

// Steady state: runtime rows moved every frame by a system (parallel by chunk), Simulate, Refresh, GpuScene update,
// EndFrame: no operator new, no storage block.
void test_steady_state_allocations()
{
    Rig rig(4);
    Assets assets;
    const MeshId mesh = rig.graph->AcquireRuntimeMesh(assets.two);
    Rng rng(8);
    std::vector<LocalTransform> locals;
    for (int i = 0; i < 4000; ++i)
        locals.push_back(RandomLocal(rng));
    rig.Spawn(std::vector<MeshId>(4000, mesh), locals);
    Query<Write<LocalTransform>, With<RuntimeMeshInstance>> movers(rig.world, "movers");
    uint32_t frame = 0;
    rig.AddSystem("move", Phase::PreSimulation, { { FindComponent<LocalTransform>(rig.world), Access::Write, {} } },
                  [&](ISystemContext* context)
    {
        const double step = 0.001 * double(frame % 7);
        movers.ParallelForChunks(context, 1, [step](const ChunkView& view, LocalTransform* local)
        {
            for (uint32_t r = 0; r < view.count; ++r)
                local[r].translation.z += step;
        });
        return nvrhi::FS_OK;
    });
    GpuScene gpu(nullptr, CpuDesc());
    auto run = [&]()
    {
        ++frame;
        rig.SimulateGraph(1.0 / 60.0);
        rig.Refresh();
        CHECK(NVRHI_SUCCEEDED(gpu.Update(rig.graph)));
        rig.EndFrame();
    };
    for (int i = 0; i < 8; ++i)
        run();
    const uint64_t newBefore = g_NewCount.load();
    const uint64_t heapBefore = ecs::GetHeapAllocationCount();
    for (int i = 0; i < 10; ++i)
        run();
    const uint64_t newCalls = g_NewCount.load() - newBefore;
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
    std::printf("steady state, 4000 moving runtime rows: %llu operator new, %llu storage blocks, %llu rows changed\n",
                (unsigned long long)newCalls, (unsigned long long)heapBlocks,
                (unsigned long long)gpu.GetStats().rowsChanged);
    CHECK(newCalls == 0 && heapBlocks == 0 && gpu.GetStats().rowsChanged == 4000);
    rig.graph->ReleaseRuntimeMesh(mesh);
}

// A spawn of 1000 rows into existing chunks after warm-up, with the despawn of the previous 1000, through a
// PreSimulation system's command buffer: the observer's references and slots, playback, the phases and GpuScene
// allocate nothing.
void test_spawn_allocations()
{
    Rig rig(4);
    Assets assets;
    const MeshId meshes[2] = { rig.graph->AcquireRuntimeMesh(assets.two), rig.graph->AcquireRuntimeMesh(assets.one) };
    const ComponentId lifetimeId = RegisterComponent<Lifetime>(rig.world);
    rig.Spawn(meshes[1], 3000, -5000.0);   // rows that stay
    Query<Read<Lifetime>> spawned(rig.world, "spawned");
    std::vector<ComponentId> ids = rig.plain;
    ids.push_back(lifetimeId);
    constexpr uint32_t c_Spawn = 1000;
    std::vector<LocalTransform> locals(c_Spawn);
    std::vector<MeshRef> refs(c_Spawn);
    std::vector<Lifetime> lives(c_Spawn, Lifetime{ 0 });
    std::vector<Entity> placeholders(c_Spawn);
    std::vector<const void*> values(ids.size(), nullptr);
    values[1] = locals.data();
    values[6] = refs.data();
    values.back() = lives.data();
    uint32_t frame = 0;
    rig.AddSystem("spawn and despawn", Phase::PreSimulation, { { lifetimeId, Access::Read, {} } },
                  [&](ISystemContext* context)
    {
        ICommandBuffer* commands = context->GetCommandBuffer();
        spawned.ForEachChunk(context, [&](const ChunkView& view, const Lifetime*)
        {
            commands->DestroyEntities(view.entities, view.count, MakeSortKey(0, view.chunkIndex, 0));
        });
        for (uint32_t i = 0; i < c_Spawn; ++i)
        {
            locals[i] = MakeLocal(dm::double3(double(i), double(frame), 0.0));
            refs[i].mesh = meshes[(i / 100 + frame) % 2];
        }
        EntityCreateDesc desc{};
        desc.components = ids.data();
        desc.values = values.data();
        desc.componentCount = uint32_t(ids.size());
        desc.entityCount = c_Spawn;
        commands->CreateEntities(desc, MakeSortKey(1, 0, 0), placeholders.data());
        return nvrhi::FS_OK;
    });
    GpuScene gpu(nullptr, CpuDesc());
    auto run = [&]()
    {
        ++frame;
        rig.SimulateGraph(1.0 / 60.0);
        rig.Refresh();
        CHECK(NVRHI_SUCCEEDED(gpu.Update(rig.graph)));
        rig.EndFrame();
        CHECK(rig.graph->GetRuntimeMeshInstanceCount() == 3000 + c_Spawn);
    };
    for (int i = 0; i < 8; ++i)
        run();
    const uint64_t newBefore = g_NewCount.load();
    const uint64_t heapBefore = ecs::GetHeapAllocationCount();
    for (int i = 0; i < 6; ++i)
        run();
    const uint64_t newCalls = g_NewCount.load() - newBefore;
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
    std::printf("spawn of %u runtime rows per frame after warm-up: %llu operator new, %llu storage blocks\n", c_Spawn,
                (unsigned long long)newCalls, (unsigned long long)heapBlocks);
    CHECK(newCalls == 0 && heapBlocks == 0);
    CHECK(rig.References(meshes[0]) + rig.References(meshes[1]) == 2 + 3000 + c_Spawn);
    CHECK(rig.graph->GetInstanceSlotCount() == 3000 + c_Spawn);   // the despawned rows' slots are reused
    rig.graph->ReleaseRuntimeMesh(meshes[0]);
    rig.graph->ReleaseRuntimeMesh(meshes[1]);
    CHECK(TakeWarnings() == 0);
}

int main(int argc, char** argv)
{
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        { "components", test_components },
        { "create_and_destroy", test_create_and_destroy },
        { "instantiate", test_instantiate },
        { "command_buffers", test_command_buffers },
        { "unknown_ids", test_unknown_ids },
        { "reference_overflow", test_reference_overflow },
        { "refusals", test_refusals },
        { "bulk_references", test_bulk_references },
        { "render_flags_equal_leaf", test_render_flags_equal_leaf },
        { "mixed_graph", test_mixed_graph },
        { "parent_is_ignored", test_parent_is_ignored },
        { "refresh_skips", test_refresh_skips },
        { "gpu_scene_equals_nodes", test_gpu_scene_equals_nodes },
        { "driven_rows", test_driven_rows },
        { "teardown", test_teardown },
        { "copy_from_another_world", test_copy_from_another_world },
        { "snapshot_rewind", test_snapshot_rewind },
        { "membership", test_membership },
        { "tag_on_node_rows", test_tag_on_node_rows },
        { "reentrant_callbacks", test_reentrant_callbacks },
        { "parent_is_root", test_parent_is_root },
        { "script_golden", test_script_golden },
        { "static_tag", test_static_tag },
        { "steady_state_allocations", test_steady_state_allocations },
        { "spawn_allocations", test_spawn_allocations },
    };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            g_LastWarning = message;
            std::fprintf(stderr, "%s\n", message);
        }
    });
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a run that dies still shows how far it got
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
                continue;
            c.run();
            CHECK(g_Warnings.load() == 0);
            CHECK(g_Errors.load() == 0);
            std::printf("%s: passed\n", c.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
