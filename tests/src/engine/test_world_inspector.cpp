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

// The inspector's library (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md, part I3; sub-plan item E7.7), basic
// semantic tests only: IWorld::GetChunkInfo and IScheduler::GetSystemCount, GetSystemInfo and GetPhaseName on a few
// rows and systems, their refusal inside a phase; donut's read-only list of a scene graph's world; one Draw on a
// headless ImGui context (font atlas built, frames driven by hand); one edit of LocalTransform written by Apply; Apply
// with another world refusing. Widgets are stood in for by the module-private test seam (src/app/WorldInspectorDetail.h).
//
// Stage T3 (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md; E7.7's functional list), after the basic cases: the
// formatter over every FieldType and FieldShape (goldens), cut at every capacity; GetChunkInfo and GetSystemInfo
// against the storage and the observed runs; every panel's text, read back through ImGui's log, against the storage;
// which components are offered for editing (donut's read-only list in a graph's world without the application naming
// it, the application's list, the flags) and that the hooked ones never are; edits of node rows and runtime-only rows
// reaching WorldTransform after Refresh (E7.0's inferred version rule), equal to the node API and to a fresh build, the
// same at 1, 2 and 8 workers; pending edits (the value shown, at most 64, re-edits); the refusals (inside a phase, in
// an observer, another or no world, a destroyed entity); the world held only through pending edits; no allocation of
// the inspector's in frames without edits. Cases that fail on an open backlog item are marked with its id and run
// apart (`--backlog`, ctest test_world_inspector_backlog), so that the default run stays the pass rule's.

#include <donut/app/WorldInspector.h>
#include <donut/core/log.h>
#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include "EcsWorldImpl.h"
#include "WorldInspectorDetail.h"

#include <imgui.h>
#include <imgui_internal.h>   // ImGui::FindWindowByName: "nothing drawn"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test, donut_app and donut_engine, which link statically); ImGui allocates
// through its own functions (malloc), which this does not count. The world's blocks come from nvrhi_core's allocator,
// which ecs::GetHeapAllocationCount counts.
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

namespace inspector_test
{
    struct A { float value; };
    DONUT_ECS_COMPONENT(A, "5d1f0f4c-58a4-4a3b-9b8e-31b0d7a5e101", DONUT_ECS_FIELDS(value));

    struct B { uint32_t value; };
    DONUT_ECS_COMPONENT(B, "0c3b7f0e-2b7e-4d55-8f2a-6a1d3e9c4b02", DONUT_ECS_FIELDS(value));

    // Stage T3. Formatter subjects: every FieldType as a scalar, and every FieldShape, through reflection.
    struct Raw { uint8_t bytes[4]; };   // a field of no known type: Opaque, its bytes
    struct Scalars
    {
        bool flag; int8_t i8; uint8_t u8; int16_t i16; uint16_t u16; int32_t i32; uint32_t u32; int64_t i64;
        uint64_t u64; float f; double d; donut::engine::Entity entity; Raw raw;
    };
    DONUT_ECS_COMPONENT(Scalars, "3b0e5a52-9c1d-4f7e-8a26-1d4b7c9e2f01",
                        DONUT_ECS_FIELDS(flag, i8, u8, i16, u16, i32, u32, i64, u64, f, d, entity, raw));

    struct Shapes
    {
        dm::float3 vector; dm::int2 ivector; dm::bool3 bvector; dm::dquat quaternion; dm::float2x2 m2;
        dm::float3x4 m34; dm::float4x4 m4; dm::affine2 a2; dm::daffine3 a3; dm::box3 box; float array[3];
        uint32_t longArray[20]; donut::engine::Entity entities[2]; Raw raws[10];
    };
    DONUT_ECS_COMPONENT(Shapes, "8f6d2c4a-1b3e-4c5d-9e7f-2a4b6c8d0e02",
                        DONUT_ECS_FIELDS(vector, ivector, bvector, quaternion, m2, m34, m4, a2, a3, box, array,
                                         longArray, entities, raws));

    // Editing by flags (WorldInspector.h): only Relocatable values of one row that are serialised and fit 256 B.
    struct Hooked   // not trivially copyable: hooks, not Relocatable (like SceneNode and LeafObject)
    {
        uint32_t value = 0;
        Hooked() noexcept = default;
        Hooked(const Hooked& other) noexcept : value(other.value) { }
        Hooked& operator=(const Hooked& other) noexcept { value = other.value; return *this; }
        ~Hooked() { }
    };
    DONUT_ECS_COMPONENT(Hooked, "c1a7e3f5-2d4b-4e6a-8c0e-3f5a7b9c1d03", DONUT_ECS_FIELDS(value));
    struct TagC { };
    DONUT_ECS_TAG(TagC, "d2b8f4a6-3e5c-4f7b-9d1f-4a6b8c0d2e04");
    struct Element { float value; };
    DONUT_ECS_BUFFER(Element, "e3c9a5b7-4f6d-4a8c-8e2a-5b7c9d1e3f05", 4, DONUT_ECS_FIELDS(value));
    struct SharedC { uint32_t key; };
    DONUT_ECS_COMPONENT_EX(SharedC, "f4dab6c8-5a7e-4b9d-9f3b-6c8d0e2f4a06", donut::engine::ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(key));
    struct ChunkC { float value; };
    DONUT_ECS_COMPONENT_EX(ChunkC, "05ebc7d9-6b8f-4cae-8a4c-7d9e1f3a5b07", donut::engine::ComponentFlags::Chunk, 0,
                           DONUT_ECS_FIELDS(value));
    struct NoSer { float value; };
    DONUT_ECS_COMPONENT_EX(NoSer, "16fcd8ea-7c9a-4dbf-9b5d-8eaf2a4b6c08", donut::engine::ComponentFlags::NoSerialize, 0,
                           DONUT_ECS_FIELDS(value));
    struct Big { float values[65]; };    // 260 B: shown only
    DONUT_ECS_COMPONENT(Big, "270de9fb-8dab-4ec0-8c6e-9fb03b5c7d09", DONUT_ECS_FIELDS(values));
    struct Edge { float values[64]; };   // 256 B: editable
    DONUT_ECS_COMPONENT(Edge, "381efa0c-9ebc-4fd1-9d7f-a0c14c6d8e10", DONUT_ECS_FIELDS(values));
    struct Toggle { float value; };
    DONUT_ECS_COMPONENT_EX(Toggle, "492f0b1d-afcd-4ae2-8e80-b1d25d7e9f11", donut::engine::ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(value));
    struct SparseC { float value; };
    DONUT_ECS_COMPONENT_EX(SparseC, "5a301c2e-b0de-4bf3-9f91-c2e36e8fa012", donut::engine::ComponentFlags::Sparse, 0,
                           DONUT_ECS_FIELDS(value));
    struct Mixed { float f; donut::engine::Entity e; Raw raw; };   // editable, with fields shown only
    DONUT_ECS_COMPONENT(Mixed, "6b412d3f-c1ef-4c04-8aa2-d3f47f9ab113", DONUT_ECS_FIELDS(f, e, raw));
}

using namespace donut;
using namespace donut::engine;
using namespace inspector_test;

namespace
{
    std::atomic<uint32_t> g_Warnings{ 0 };

    // The index of the archetype with exactly these components (sorted ids).
    uint32_t FindArchetype(IWorld* world, std::vector<ComponentId> components)
    {
        std::sort(components.begin(), components.end());
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            if (info.componentCount == components.size() &&
                std::equal(components.begin(), components.end(), info.components))
                return a;
        }
        CHECK(false);
        return ~0u;
    }

    std::vector<Entity> Create(IWorld* world, std::vector<ComponentId> components, uint32_t count)
    {
        EntityCreateDesc desc{};
        desc.components = components.data();
        desc.componentCount = uint32_t(components.size());
        desc.entityCount = count;
        std::vector<Entity> entities(count);
        CHECK(world->CreateEntities(desc, entities.data()) == nvrhi::FS_OK);
        return entities;
    }

    nvrhi::AutoPtr<ISystem> MakeSystem(const char* name, Phase phase, SystemFlags flags,
                                       std::function<void(ISystemContext*)> update,
                                       const char* runBefore = nullptr)
    {
        SystemDesc desc{};
        desc.name = name;
        desc.phase = phase;
        desc.flags = flags;
        desc.runBefore = runBefore ? &runBefore : nullptr;
        desc.runBeforeCount = runBefore ? 1 : 0;
        nvrhi::AutoPtr<ISystem> system;
        CHECK(CreateLambdaSystem(desc, std::move(update), &system) == nvrhi::FS_OK);
        return system;
    }

    struct Env
    {
        nvrhi::AutoPtr<IWorld> world;
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<IScheduler> scheduler;

        Env()
        {
            WorldDesc worldDesc;
            worldDesc.debugName = "test_world_inspector";
            CHECK(CreateWorld(worldDesc, &world) == nvrhi::FS_OK);
            JobSystemDesc jobDesc;
            jobDesc.debugName = "test_world_inspector";
            jobDesc.workerThreadCount = 2;
            CHECK(CreateJobSystem(jobDesc, &jobs) == nvrhi::FS_OK);
            CHECK(CreateScheduler(world.Get(), jobs.Get(), &scheduler) == nvrhi::FS_OK);
        }

        SystemId Add(nvrhi::AutoPtr<ISystem> system)
        {
            SystemId id = SystemId::Invalid;
            CHECK(scheduler->AddSystem(system.Get(), id) == nvrhi::FS_OK);
            return id;
        }
    };

    // A headless ImGui context: the font atlas built by hand, no renderer.
    struct ImGuiContextScope
    {
        ImGuiContextScope()
        {
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.IniFilename = nullptr;
            io.LogFilename = nullptr;
            io.DisplaySize = ImVec2(1280.f, 1024.f);
            io.DeltaTime = 1.f / 60.f;
            unsigned char* pixels = nullptr;
            int width = 0;
            int height = 0;
            io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
            CHECK(pixels && width > 0 && height > 0);
        }
        ~ImGuiContextScope()
        {
            app::detail::SetWorldInspectorTestHook(nullptr);
            ImGui::DestroyContext();
        }
    };

    void DrawFrame(app::WorldInspector& inspector, IWorld* world, IScheduler* scheduler)
    {
        ImGui::NewFrame();
        inspector.Draw(world, scheduler);
        ImGui::Render();
    }

    // A scene graph's world with a root and three children.
    struct Graph
    {
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        std::vector<Entity> nodes;

        Graph()
        {
            graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            for (int i = 0; i < 3; ++i)
            {
                auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                node->SetTranslation(dm::double3(double(i), 0.0, 0.0));
                nvrhi::AutoPtr<SceneGraphNode> attached = graph->Attach(graph->GetRootNode(), node);
                nodes.push_back(graph->GetEntity(attached.Get()));
                CHECK(nodes.back() != Entity::Null);
            }
            graph->Refresh(0);
        }
        IWorld* World() const { return graph->GetWorld(); }
    };

    struct EditRecorder
    {
        ComponentId target = ComponentId::Invalid;
        dm::double3 translation = dm::double3(0.0);
        bool commit = false;
        std::vector<ComponentId> offered;

        static bool Edit(void* context, ComponentId component, void* value, uint32_t size)
        {
            EditRecorder& self = *static_cast<EditRecorder*>(context);
            self.offered.push_back(component);
            if (!self.commit || component != self.target)
                return false;
            CHECK(size == sizeof(LocalTransform));
            static_cast<LocalTransform*>(value)->translation = self.translation;
            self.commit = false;
            return true;
        }
    };

    bool Equal(const dm::double3& a, const dm::double3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
}

// GetChunkInfo on two archetypes of a few rows: ids in row order, the chunk's versions and added version, capacity,
// writes moving one column's version; indices out of range.
void test_chunk_info()
{
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    const ComponentId b = RegisterComponent<B>(world);
    const uint64_t created = world->GetVersion();
    const std::vector<Entity> onlyA = Create(world, { a }, 5);
    const std::vector<Entity> both = Create(world, { a, b }, 3);
    const uint32_t archA = FindArchetype(world, { a });
    const uint32_t archAB = FindArchetype(world, { a, b });

    ArchetypeInfo info{};
    CHECK(world->GetArchetypeInfo(archA, info) == nvrhi::FS_OK && info.chunkCount == 1);
    ChunkInfo chunk{};
    CHECK(world->GetChunkInfo(archA, 0, chunk) == nvrhi::FS_OK);
    CHECK(chunk.count == 5 && chunk.capacity == info.rowsPerChunk && chunk.capacity >= 5);
    CHECK(std::equal(onlyA.begin(), onlyA.end(), chunk.entities));
    CHECK(chunk.versions[0] == created && chunk.addedVersion == created);

    CHECK(world->GetArchetypeInfo(archAB, info) == nvrhi::FS_OK && info.componentCount == 2);
    const uint32_t columnA = info.components[0] == a ? 0 : 1;
    const uint32_t columnB = 1 - columnA;
    const uint64_t written = world->AdvanceVersion();
    CHECK(written > created && world->GetVersion() == written);
    CHECK(GetComponentForWrite<A>(world, both[1]) != nullptr);
    CHECK(world->GetChunkInfo(archAB, 0, chunk) == nvrhi::FS_OK);
    CHECK(chunk.count == 3 && chunk.capacity == info.rowsPerChunk);
    CHECK(std::equal(both.begin(), both.end(), chunk.entities));
    CHECK(chunk.versions[columnA] == written && chunk.versions[columnB] == created && chunk.addedVersion == created);

    // Out of range: FE_INVALID_ARGS, `out` zeroed.
    CHECK(world->GetChunkInfo(archA, 1, chunk) == nvrhi::FE_INVALID_ARGS);
    CHECK(!chunk.entities && !chunk.versions && chunk.count == 0 && chunk.capacity == 0 && chunk.addedVersion == 0);
    CHECK(world->GetChunkInfo(world->GetArchetypeCount(), 0, chunk) == nvrhi::FE_INVALID_ARGS);
    CHECK(!chunk.entities && chunk.count == 0);
}

// GetSystemCount, GetSystemInfo in run order (a named phase included) with flags, enabled bits and run counts, and
// GetPhaseName; an index out of range; removal.
void test_system_info()
{
    Env env;
    IScheduler* scheduler = env.scheduler.Get();
    Phase custom = Phase::Count;
    CHECK(scheduler->AddPhase("custom", Phase::PreSimulation, custom) == nvrhi::FS_OK);
    const SystemId first = env.Add(MakeSystem("first", Phase::PreSimulation, SystemFlags::None, [](ISystemContext*) {}));
    const SystemId second = env.Add(MakeSystem("second", Phase::PreSimulation, SystemFlags::MainThread,
                                               [](ISystemContext*) {}, "first"));
    const SystemId begin = env.Add(MakeSystem("begin", Phase::BeginFrame, SystemFlags::Exclusive,
                                              [](ISystemContext*) {}));
    const SystemId late = env.Add(MakeSystem("late", custom, SystemFlags::StartDisabled, [](ISystemContext*) {}));
    CHECK(scheduler->GetSystemCount() == 4);
    CHECK(scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);

    struct Expected { const char* name; SystemId id; Phase phase; SystemFlags flags; uint32_t enabled; uint64_t runs; };
    const Expected expected[] = {
        { "begin", begin, Phase::BeginFrame, SystemFlags::Exclusive, 1, 1 },
        { "second", second, Phase::PreSimulation, SystemFlags::MainThread, 1, 1 },
        { "first", first, Phase::PreSimulation, SystemFlags::None, 1, 1 },
        { "late", late, custom, SystemFlags::StartDisabled, 0, 0 },
    };
    for (uint32_t i = 0; i < 4; ++i)
    {
        SystemInfo info{};
        CHECK(scheduler->GetSystemInfo(i, info) == nvrhi::FS_OK);
        CHECK(info.name && std::strcmp(info.name, expected[i].name) == 0);
        CHECK(info.id == expected[i].id && info.phase == expected[i].phase && info.flags == expected[i].flags);
        CHECK(info.enabled == expected[i].enabled && info.stats.runCount == expected[i].runs);
        SystemStats stats{};
        CHECK(scheduler->GetSystemStats(info.id, stats) == nvrhi::FS_OK);
        CHECK(stats.runCount == info.stats.runCount && stats.lastMilliseconds == info.stats.lastMilliseconds);
    }
    SystemInfo info{};
    CHECK(scheduler->GetSystemInfo(4, info) == nvrhi::FE_INVALID_ARGS);
    CHECK(!info.name && info.id == SystemId(0) && info.enabled == 0);

    CHECK(std::strcmp(scheduler->GetPhaseName(Phase::BeginFrame), "BeginFrame") == 0);
    CHECK(std::strcmp(scheduler->GetPhaseName(Phase::EndFrame), "EndFrame") == 0);
    CHECK(std::strcmp(scheduler->GetPhaseName(custom), "custom") == 0);
    CHECK(scheduler->GetPhaseName(Phase(uint32_t(custom) + 1)) == nullptr);

    CHECK(scheduler->SetSystemEnabled(late, true) == nvrhi::FS_OK);
    CHECK(scheduler->RemoveSystem(begin) == nvrhi::FS_OK);
    CHECK(scheduler->GetSystemCount() == 3);
    CHECK(scheduler->GetSystemInfo(2, info) == nvrhi::FS_OK && info.id == late && info.enabled == 1);
    CHECK(scheduler->GetSystemInfo(0, info) == nvrhi::FS_OK && info.id == second);
}

// Inside a phase, GetChunkInfo, GetSystemInfo and WorldInspector::Apply refuse with FE_OUT_OF_DATE; GetChunkInfo
// works from an exclusive system.
void test_refusals_inside_phase()
{
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    Create(world, { a }, 2);
    const uint32_t arch = FindArchetype(world, { a });
    app::WorldInspector inspector;
    nvrhi::FRESULT chunkInPhase = nvrhi::FS_OK;
    nvrhi::FRESULT systemInPhase = nvrhi::FS_OK;
    nvrhi::FRESULT applyInPhase = nvrhi::FS_OK;
    nvrhi::FRESULT chunkInExclusive = nvrhi::FE_NOT_IMPLEMENT;
    uint32_t zeroed = 0;
    env.Add(MakeSystem("reader", Phase::PreSimulation, SystemFlags::None, [&](ISystemContext*)
    {
        ChunkInfo chunk{};
        chunk.count = 7;
        chunkInPhase = world->GetChunkInfo(arch, 0, chunk);
        zeroed += chunk.count == 0 && !chunk.entities ? 1 : 0;
        SystemInfo info{};
        info.enabled = 7;
        systemInPhase = env.scheduler->GetSystemInfo(0, info);
        zeroed += info.enabled == 0 && !info.name ? 1 : 0;
        applyInPhase = inspector.Apply(world);
    }));
    env.Add(MakeSystem("exclusive", Phase::PostSimulation, SystemFlags::Exclusive, [&](ISystemContext*)
    {
        ChunkInfo chunk{};
        chunkInExclusive = world->GetChunkInfo(arch, 0, chunk);
        zeroed += chunk.count == 2 ? 1 : 0;
    }));
    const uint32_t warnings = g_Warnings.load();
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
    CHECK(chunkInPhase == nvrhi::FE_OUT_OF_DATE && systemInPhase == nvrhi::FE_OUT_OF_DATE);
    CHECK(applyInPhase == nvrhi::FE_OUT_OF_DATE && chunkInExclusive == nvrhi::FS_OK && zeroed == 3);
    CHECK(g_Warnings.load() == warnings + 1);   // Apply's
    g_Warnings = warnings;
}

// Donut's read-only list in a scene graph's world: the 18 ids, registered after the store's types; none elsewhere.
void test_read_only_list()
{
    Graph g;
    IWorld* world = g.World();
    const ComponentId expected[] = {
        FindComponent<SceneLinks>(world), FindComponent<Parent>(world), FindComponent<HierarchyDepth>(world),
        FindComponent<LocalMatrix>(world), FindComponent<WorldTransform>(world), FindComponent<LocalBounds>(world),
        FindComponent<WorldBounds>(world), FindComponent<ChunkWorldBounds>(world),
        FindComponent<GpuInstanceSlot>(world), FindComponent<MeshRef>(world), FindComponent<RenderFlags>(world),
        FindComponent<SimBodyRef>(world), FindComponent<SceneNode>(world), FindComponent<LeafObject>(world),
        FindComponent<SimPosePrevious>(world), FindComponent<SimTeleport>(world),   // backlog I3-R1, I3-R2
        FindComponent<SimPose>(world), FindComponent<SimScale>(world),              // I3-R2 (F8)
    };
    for (ComponentId id : expected)
        CHECK(id != ComponentId::Invalid);
    ComponentId ids[c_MaxInspectorReadOnlyComponents] = {};
    CHECK(app::GetDonutReadOnlyComponents(world, nullptr, 0) == 18);
    CHECK(app::GetDonutReadOnlyComponents(world, ids, c_MaxInspectorReadOnlyComponents) == 18);
    CHECK(std::equal(std::begin(expected), std::end(expected), ids));
    CHECK(std::find(ids, ids + 18, FindComponent<LocalTransform>(world)) == ids + 18);

    ComponentId two[3] = { ComponentId::Invalid, ComponentId::Invalid, ComponentId::Invalid };
    CHECK(app::GetDonutReadOnlyComponents(world, two, 2) == 18);
    CHECK(two[0] == expected[0] && two[1] == expected[1] && two[2] == ComponentId::Invalid);

    const ComponentId resource = FindComponent<InspectorReadOnlyComponents>(world);
    CHECK(resource != ComponentId::Invalid);
    CHECK(uint32_t(resource) > uint32_t(FindComponent<RuntimeMeshInstance>(world)));
    for (ComponentId id : expected)
        CHECK(uint32_t(resource) > uint32_t(id));
    const ComponentTypeDesc* desc = world->GetComponentTypeDesc(resource);
    CHECK(desc && !!(desc->flags & ComponentFlags::Resource) && !!(desc->flags & ComponentFlags::NoSerialize));

    Env env;
    CHECK(app::GetDonutReadOnlyComponents(env.world.Get(), ids, c_MaxInspectorReadOnlyComponents) == 0);
    CHECK(app::GetDonutReadOnlyComponents(nullptr, ids, c_MaxInspectorReadOnlyComponents) == 0);
}

// One Draw of every panel (headers and nodes opened by the seam) over a scene graph's world and scheduler on a
// headless context: it draws, offers LocalTransform for editing and no column of the read-only list, and warns of
// nothing.
void test_draw_headless()
{
    ImGuiContextScope context;
    Graph g;
    IWorld* world = g.World();
    app::WorldInspector inspector;
    inspector.SetEntityNames([](void*, Entity entity, char* out, uint32_t capacity)
    {
        return std::snprintf(out, capacity, "node %u", GetEntityIndex(entity)) > 0;
    }, nullptr);
    EditRecorder recorder;
    app::detail::WorldInspectorTestHook hook;
    hook.select = g.nodes[1];
    hook.openAll = true;
    hook.edit = &EditRecorder::Edit;
    hook.context = &recorder;
    app::detail::SetWorldInspectorTestHook(&hook);
    const uint32_t warnings = g_Warnings.load();
    DrawFrame(inspector, world, g.graph->GetScheduler());   // a new window is hidden for its first frame
    DrawFrame(inspector, world, g.graph->GetScheduler());
    const ImDrawData* data = ImGui::GetDrawData();
    CHECK(data && data->Valid && data->TotalVtxCount > 0);
    const ComponentId local = FindComponent<LocalTransform>(world);
    CHECK(std::find(recorder.offered.begin(), recorder.offered.end(), local) != recorder.offered.end());
    ComponentId readOnly[c_MaxInspectorReadOnlyComponents] = {};
    const uint32_t count = app::GetDonutReadOnlyComponents(world, readOnly, c_MaxInspectorReadOnlyComponents);
    for (uint32_t i = 0; i < count; ++i)
        CHECK(std::find(recorder.offered.begin(), recorder.offered.end(), readOnly[i]) == recorder.offered.end());
    hook.openAll = false;
    DrawFrame(inspector, world, nullptr);   // no systems panel
    CHECK(g_Warnings.load() == warnings);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);   // nothing was committed
}

// One edit of LocalTransform, committed through Draw, is written by Apply (and the version moves); another world's
// Apply refuses and drops the edits.
void test_edit_and_apply()
{
    ImGuiContextScope context;
    Graph g;
    IWorld* world = g.World();
    const Entity node = g.nodes[2];
    const dm::double3 before = GetComponent<LocalTransform>(world, node)->translation;
    app::WorldInspector inspector;
    EditRecorder recorder;
    recorder.target = FindComponent<LocalTransform>(world);
    recorder.translation = dm::double3(1.0, 2.0, 3.0);
    recorder.commit = true;
    app::detail::WorldInspectorTestHook hook;
    hook.select = node;
    hook.edit = &EditRecorder::Edit;
    hook.context = &recorder;
    app::detail::SetWorldInspectorTestHook(&hook);
    DrawFrame(inspector, world, g.graph->GetScheduler());
    CHECK(!recorder.commit);   // committed
    CHECK(Equal(GetComponent<LocalTransform>(world, node)->translation, before));   // pending until Apply
    const uint64_t version = world->GetVersion();
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(Equal(GetComponent<LocalTransform>(world, node)->translation, recorder.translation));
    CHECK(world->GetVersion() > version);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);   // nothing pending

    // Another world: refused, the edit dropped with a warning.
    recorder.translation = dm::double3(4.0, 5.0, 6.0);
    recorder.commit = true;
    DrawFrame(inspector, world, nullptr);
    CHECK(!recorder.commit);
    Env other;
    const uint32_t warnings = g_Warnings.load();
    CHECK(inspector.Apply(other.world.Get()) == nvrhi::FE_INVALID_ARGS);
    CHECK(g_Warnings.load() == warnings + 1);
    g_Warnings = warnings;
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);   // dropped: nothing pending
    CHECK(Equal(GetComponent<LocalTransform>(world, node)->translation, dm::double3(1.0, 2.0, 3.0)));
}

// ===================================================================================== stage T3: functional tests

namespace
{
    // Takes the warnings a case expects (at least `atLeast`, exactly `exactly` when it is not ~0u) and clears them.
    void TakeWarnings(uint32_t before, uint32_t exactly, uint32_t atLeast = 0)
    {
        const uint32_t seen = g_Warnings.load() - before;
        if (exactly != ~0u)
            CHECK(seen == exactly);
        CHECK(seen >= atLeast);
        g_Warnings = before;
    }

    uint64_t HashBytes(uint64_t hash, const void* data, size_t size)
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
        return hash;
    }

    template <typename T> bool SameBytes(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

    bool Contains(const std::string& text, const std::string& needle)
    {
        if (text.find(needle) != std::string::npos)
            return true;
        std::fprintf(stderr, "test_world_inspector: the panel text lacks \"%s\"\n", needle.c_str());
        return false;
    }

    std::string Format(const FieldDesc& field, const void* value)
    {
        char text[512];
        app::detail::FormatInspectorField(text, uint32_t(sizeof(text)), field, value);
        return text;
    }

    const FieldDesc& FieldOf(const ComponentTypeDesc& desc, const char* name)
    {
        for (uint32_t f = 0; f < desc.fieldCount; ++f)
            if (std::strcmp(desc.fields[f].name, name) == 0)
                return desc.fields[f];
        CHECK(false);
        return desc.fields[0];
    }

    // ImGui's log of one frame, read back through the clipboard hook (LogToClipboard hands LogFinish's text to it).
    std::string g_Clipboard;
    void CaptureClipboard(ImGuiContext*, const char* text) { g_Clipboard = text ? text : ""; }

    // A hook that records each value Draw offers for editing (the component and the copy the widgets edit) and, when
    // armed, commits `commit` for `target`, as a widget would. It never throws (Draw's internals are noexcept).
    struct OfferLog
    {
        std::vector<ComponentId> components;
        std::vector<std::vector<uint8_t>> values;
        ComponentId target = ComponentId::Invalid;
        std::vector<uint8_t> commit;
        bool armed = false;
        bool sizeMismatch = false;

        static bool Edit(void* context, ComponentId component, void* value, uint32_t size)
        {
            OfferLog& self = *static_cast<OfferLog*>(context);
            const uint8_t* bytes = static_cast<const uint8_t*>(value);
            self.components.push_back(component);
            self.values.emplace_back(bytes, bytes + size);
            if (!self.armed || component != self.target)
                return false;
            if (size != self.commit.size())
            {
                self.sizeMismatch = true;
                return false;
            }
            std::memcpy(value, self.commit.data(), size);
            self.armed = false;
            return true;
        }
        bool Offered(ComponentId component) const
        {
            return std::find(components.begin(), components.end(), component) != components.end();
        }
        // The last copy offered of `component`; empty if it was not offered.
        std::vector<uint8_t> Shown(ComponentId component) const
        {
            for (size_t i = components.size(); i-- > 0;)
                if (components[i] == component)
                    return values[i];
            return {};
        }
        std::set<ComponentId> Set() const { return std::set<ComponentId>(components.begin(), components.end()); }
    };

    // One frame with the seam selecting `entity` (every node open with `openAll`) and `offers` recording; with `text`,
    // ImGui's log of the frame.
    void DrawSelecting(app::WorldInspector& inspector, IWorld* world, IScheduler* scheduler, Entity entity,
                       OfferLog& offers, bool openAll = false, std::string* text = nullptr)
    {
        app::detail::WorldInspectorTestHook hook;
        hook.select = entity;
        hook.openAll = openAll;
        hook.edit = &OfferLog::Edit;
        hook.context = &offers;
        app::detail::SetWorldInspectorTestHook(&hook);
        ImGui::GetPlatformIO().Platform_SetClipboardTextFn = &CaptureClipboard;
        g_Clipboard.clear();
        ImGui::NewFrame();
        if (text)
            ImGui::LogToClipboard();
        inspector.Draw(world, scheduler);
        if (text)
        {
            ImGui::LogFinish();
            *text = g_Clipboard;
        }
        ImGui::Render();
        app::detail::SetWorldInspectorTestHook(nullptr);
        CHECK(!offers.sizeMismatch);
    }

    // Commits `value` as the whole value of T of `entity` through one Draw; true when the inspector offered it.
    template <typename T>
    bool Commit(app::WorldInspector& inspector, IWorld* world, Entity entity, const T& value,
                std::string* text = nullptr)
    {
        OfferLog offers;
        offers.target = FindComponent<T>(world);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        offers.commit.assign(bytes, bytes + sizeof(T));
        offers.armed = true;
        DrawSelecting(inspector, world, nullptr, entity, offers, false, text);
        return !offers.armed;
    }

    // The World panel's "pending edits" count, read from the panel's text.
    uint32_t PendingShown(app::WorldInspector& inspector, IWorld* world)
    {
        OfferLog offers;
        std::string text;
        DrawSelecting(inspector, world, nullptr, Entity::Null, offers, false, &text);
        const size_t at = text.find("pending edits ");
        CHECK(at != std::string::npos);
        return uint32_t(std::strtoul(text.c_str() + at + std::strlen("pending edits "), nullptr, 10));
    }

    // Every chunk of every archetype, copied out of GetChunkInfo.
    struct ChunkCopy
    {
        uint32_t archetype;
        uint32_t chunk;
        std::vector<Entity> entities;
        std::vector<uint64_t> versions;
        uint64_t addedVersion;
        uint32_t capacity;
    };
    std::vector<ChunkCopy> AllChunks(IWorld* world)
    {
        std::vector<ChunkCopy> out;
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            for (uint32_t c = 0; c < info.chunkCount; ++c)
            {
                ChunkInfo chunk{};
                CHECK(world->GetChunkInfo(a, c, chunk) == nvrhi::FS_OK);
                ChunkCopy copy{ a, c, std::vector<Entity>(chunk.entities, chunk.entities + chunk.count),
                                std::vector<uint64_t>(chunk.versions, chunk.versions + info.componentCount),
                                chunk.addedVersion, chunk.capacity };
                out.push_back(std::move(copy));
            }
        }
        return out;
    }

    // The chunk view against the storage: every archetype's chunks hold its rows, within capacity; every listed entity
    // is in exactly one chunk, in an archetype of exactly the components it has; the world's totals are the sums.
    void CheckChunksAgainstStorage(IWorld* world, const std::vector<Entity>& alive, const std::vector<Entity>& dead)
    {
        std::map<Entity, uint32_t> seen;   // entity -> archetype
        uint32_t entities = 0;
        uint32_t chunks = 0;
        uint64_t rowBytes = 0;
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            uint32_t rows = 0;
            for (uint32_t c = 0; c < info.chunkCount; ++c)
            {
                ChunkInfo chunk{};
                CHECK(world->GetChunkInfo(a, c, chunk) == nvrhi::FS_OK);
                CHECK(chunk.capacity == info.rowsPerChunk && chunk.count <= chunk.capacity);
                rows += chunk.count;
                for (uint32_t r = 0; r < chunk.count; ++r)
                    CHECK(seen.emplace(chunk.entities[r], a).second);   // in one chunk only
            }
            CHECK(rows == info.entityCount);
            entities += info.entityCount;
            chunks += info.chunkCount;
            rowBytes += uint64_t(info.entityCount) * info.rowSize;
        }
        WorldStats stats{};
        world->GetStats(stats);
        CHECK(stats.entityCount == entities && stats.chunkCount == chunks && stats.rowBytes == rowBytes);
        CHECK(stats.archetypeCount == world->GetArchetypeCount());
        for (Entity entity : alive)
        {
            CHECK(world->IsAlive(entity));
            const auto found = seen.find(entity);
            CHECK(found != seen.end());
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(found->second, info) == nvrhi::FS_OK);
            for (uint32_t t = 0; t < world->GetComponentTypeCount(); ++t)
            {
                const ComponentTypeDesc* desc = world->GetComponentTypeDesc(ComponentId(t));
                if (!desc || !!(desc->flags & (ComponentFlags::Resource | ComponentFlags::Sparse)))
                    continue;
                const bool inArchetype = std::find(info.components, info.components + info.componentCount,
                                                   ComponentId(t)) != info.components + info.componentCount;
                CHECK(world->HasComponent(entity, ComponentId(t)) == inArchetype);
            }
        }
        for (Entity entity : dead)
            CHECK(!world->IsAlive(entity) && seen.find(entity) == seen.end());
    }

    // ------------------------------------------------------------------------------------- scene-graph rig

    nvrhi::AutoPtr<Material> MakeMaterial()
    {
        auto material = MAKE_RC_OBJ_PTR(Material);
        material->domain = MaterialDomain::Opaque;
        return material;
    }

    nvrhi::AutoPtr<MeshInfo> MakeMesh(Material* material)
    {
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->type = MeshType::Triangles;
        auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
        geometry->material = material;
        geometry->numIndices = 3;
        geometry->numVertices = 3;
        geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f, 2.f, 1.f));
        mesh->geometries.push_back(geometry);
        mesh->objectSpaceBounds = geometry->objectSpaceBounds;
        return mesh;
    }

    LocalTransform MakeLocal(const dm::double3& translation, const dm::dquat& rotation = dm::dquat::identity(),
                             const dm::double3& scaling = dm::double3(1.0))
    {
        return LocalTransform{ translation, rotation, scaling };
    }

    // A unit quaternion about (1, 2, 2) / 3.
    dm::dquat Rotation(double angle)
    {
        const double s = std::sin(angle * 0.5);
        return dm::dquat(std::cos(angle * 0.5), s / 3.0, 2.0 * s / 3.0, 2.0 * s / 3.0);
    }

    // A scene graph with nodes in a binary tree under the root (node i under node (i - 1) / 2, node 0 under the root;
    // a mesh leaf on every even one) and runtime-only rows of the same mesh, refreshed once.
    struct SceneRig
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        nvrhi::AutoPtr<Material> material = MakeMaterial();
        nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh(material.Get());
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        std::vector<Entity> nodeRows;
        std::vector<Entity> runtimeRows;
        uint32_t frame = 0;

        SceneRig(uint32_t workers, const std::vector<LocalTransform>& nodeLocals,
                 const std::vector<LocalTransform>& runtimeLocals)
        {
            if (workers)
            {
                JobSystemDesc desc;
                desc.debugName = "test_world_inspector";
                desc.workerThreadCount = workers;
                CHECK(CreateJobSystem(desc, &jobs) == nvrhi::FS_OK);
                graph->SetJobSystem(jobs.Get());
            }
            graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            for (size_t i = 0; i < nodeLocals.size(); ++i)
            {
                auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                node->SetTransform(&nodeLocals[i].translation, &nodeLocals[i].rotation, &nodeLocals[i].scaling);
                if (i % 2 == 0)
                    node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
                SceneGraphNode* parent = i == 0 ? graph->GetRootNode() : nodes[(i - 1) / 2].Get();
                nvrhi::AutoPtr<SceneGraphNode> attached = graph->Attach(parent, node);
                CHECK(attached);
                nodes.push_back(attached);
                nodeRows.push_back(graph->GetEntity(attached.Get()));
                CHECK(nodeRows.back() != Entity::Null);
            }
            if (!runtimeLocals.empty())
                Spawn(runtimeLocals);
            Refresh();
        }

        IWorld* World() const { return graph->GetWorld(); }
        void Refresh() { graph->Refresh(frame); }

        void Spawn(const std::vector<LocalTransform>& locals)
        {
            IWorld* world = World();
            std::vector<ComponentId> ids(graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
            graph->GetRuntimeMeshInstanceComponents(ids.data(), uint32_t(ids.size()));
            const MeshId id = graph->AcquireRuntimeMesh(mesh.Get());
            CHECK(id != MeshId::Invalid);
            const std::vector<MeshRef> refs(locals.size(), MeshRef{ id });
            std::vector<const void*> values(ids.size(), nullptr);
            for (size_t i = 0; i < ids.size(); ++i)
            {
                if (ids[i] == FindComponent<LocalTransform>(world))
                    values[i] = locals.data();
                else if (ids[i] == FindComponent<MeshRef>(world))
                    values[i] = refs.data();
            }
            EntityCreateDesc desc{};
            desc.components = ids.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(ids.size());
            desc.entityCount = uint32_t(locals.size());
            std::vector<Entity> created(locals.size(), Entity::Null);
            CHECK(world->CreateEntities(desc, created.data()) == nvrhi::FS_OK);
            runtimeRows.insert(runtimeRows.end(), created.begin(), created.end());
            graph->ReleaseRuntimeMesh(id);   // the rows hold theirs
        }

        // WorldTransform and WorldBounds (where the row has them) of `rows`, in order.
        uint64_t Hash(const std::vector<Entity>& rows) const
        {
            IWorld* world = World();
            uint64_t hash = 14695981039346656037ull;
            for (Entity row : rows)
            {
                const WorldTransform* transform = GetComponent<WorldTransform>(world, row);
                const WorldBounds* bounds = GetComponent<WorldBounds>(world, row);
                CHECK(transform);
                hash = HashBytes(hash, transform, sizeof(*transform));
                const uint8_t none = 0xee;
                hash = bounds ? HashBytes(hash, bounds, sizeof(*bounds)) : HashBytes(hash, &none, 1);
            }
            return hash;
        }
    };

    // The rows of two rigs have bit-identical WorldTransform and WorldBounds.
    void CheckSameWorld(const SceneRig& a, const std::vector<Entity>& rowsA, const SceneRig& b,
                        const std::vector<Entity>& rowsB)
    {
        CHECK(rowsA.size() == rowsB.size());
        for (size_t i = 0; i < rowsA.size(); ++i)
        {
            const WorldTransform* ta = GetComponent<WorldTransform>(a.World(), rowsA[i]);
            const WorldTransform* tb = GetComponent<WorldTransform>(b.World(), rowsB[i]);
            CHECK(ta && tb && SameBytes(*ta, *tb));
            const WorldBounds* ba = GetComponent<WorldBounds>(a.World(), rowsA[i]);
            const WorldBounds* bb = GetComponent<WorldBounds>(b.World(), rowsB[i]);
            CHECK(!ba == !bb);
            if (ba)
                CHECK(SameBytes(*ba, *bb));
        }
    }
}

// ------------------------------------------------------------------------------------------- the formatter

// Every FieldType as a scalar, through the reflection of a C++ component (goldens).
void test_format_scalars()
{
    const ComponentTypeDesc& desc = ComponentTraits<Scalars>::Desc();
    CHECK(desc.fieldCount == 13);
    const FieldType types[] = { FieldType::Boolean, FieldType::Int8, FieldType::UInt8, FieldType::Int16,
                                FieldType::UInt16, FieldType::Int32, FieldType::UInt32, FieldType::Int64,
                                FieldType::UInt64, FieldType::Float, FieldType::Double, FieldType::Entity,
                                FieldType::Opaque };
    for (uint32_t f = 0; f < 13; ++f)
    {
        CHECK(desc.fields[f].type == types[f] && desc.fields[f].shape == FieldShape::Scalar);
        CHECK(desc.fields[f].scalarCount == (types[f] == FieldType::Opaque ? 4u : 1u));
    }

    Scalars s;
    std::memset(&s, 0, sizeof(s));
    s.flag = true; s.i8 = -128; s.u8 = 200; s.i16 = -300; s.u16 = 60000; s.i32 = -70000; s.u32 = 4000000000u;
    s.i64 = -9000000000ll; s.u64 = ~0ull; s.f = 0.1f; s.d = 3.14159265358979; s.entity = MakeEntity(5, 2);
    s.raw = Raw{ { 0x0a, 0x0b, 0xc0, 0xff } };
    const char* const expected[] = { "true", "-128", "200", "-300", "60000", "-70000", "4000000000", "-9000000000",
                                     "18446744073709551615", "0.1", "3.14159265", "#5:2", "0a0bc0ff" };
    for (uint32_t f = 0; f < 13; ++f)
        CHECK(Format(desc.fields[f], &s) == expected[f]);

    s.flag = false; s.f = 1234567.f; s.d = 1e-10; s.entity = Entity::Null; s.i8 = 127; s.i64 = 0;
    CHECK(Format(FieldOf(desc, "flag"), &s) == "false");
    CHECK(Format(FieldOf(desc, "f"), &s) == "1.23457e+06");
    CHECK(Format(FieldOf(desc, "d"), &s) == "1e-10");
    CHECK(Format(FieldOf(desc, "entity"), &s) == "null");
    CHECK(Format(FieldOf(desc, "i8"), &s) == "127");
    CHECK(Format(FieldOf(desc, "i64"), &s) == "0");
}

// Every FieldShape: through reflection of a C++ component, and hand-made descriptors for the counts that are not a
// square matrix, an affine transform or an even box (goldens).
void test_format_shapes()
{
    const ComponentTypeDesc& desc = ComponentTraits<Shapes>::Desc();
    CHECK(desc.fieldCount == 14);
    struct Expected { const char* name; FieldType type; FieldShape shape; uint32_t count; const char* text; };
    const Expected expected[] = {
        { "vector", FieldType::Float, FieldShape::Vector, 3, "(1, 2.5, -3)" },
        { "ivector", FieldType::Int32, FieldShape::Vector, 2, "(-1, 7)" },
        { "bvector", FieldType::Boolean, FieldShape::Vector, 3, "(true, false, true)" },
        { "quaternion", FieldType::Double, FieldShape::Quaternion, 4, "(1, 0, 0.5, -0.25)" },
        { "m2", FieldType::Float, FieldShape::Matrix, 4, "[1 2; 3 4]" },
        { "m34", FieldType::Float, FieldShape::Matrix, 12, "[0 1 2 3; 4 5 6 7; 8 9 10 11]" },
        { "m4", FieldType::Float, FieldShape::Matrix, 16, "[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]" },
        { "a2", FieldType::Float, FieldShape::Affine, 6, "[1 2; 3 4] t (5, 6)" },
        { "a3", FieldType::Double, FieldShape::Affine, 12, "[1 2 3; 4 5 6; 7 8 9] t (10, 11, 12)" },
        { "box", FieldType::Float, FieldShape::Box, 6, "min (-1, -2, -3) max (1, 2, 3)" },
        { "array", FieldType::Float, FieldShape::Array, 3, "[0.5, 1, 1.5]" },
        { "longArray", FieldType::UInt32, FieldShape::Array, 20,
          "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ... (20)]" },
        { "entities", FieldType::Entity, FieldShape::Array, 2, "[null, #3:1]" },
        { "raws", FieldType::Opaque, FieldShape::Array, 40,
          "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f... (40 bytes)" },
    };

    Shapes s;
    std::memset(static_cast<void*>(&s), 0, sizeof(s));
    s.vector = dm::float3(1.f, 2.5f, -3.f);
    s.ivector = dm::int2(-1, 7);
    s.bvector = dm::bool3(true, false, true);
    s.quaternion = dm::dquat(1.0, 0.0, 0.5, -0.25);
    const float m2[] = { 1, 2, 3, 4 };
    const float m34[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const float m4[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const float a2[] = { 1, 2, 3, 4, 5, 6 };
    const double a3[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const float box[] = { -1, -2, -3, 1, 2, 3 };
    static_assert(sizeof(s.m2) == sizeof(m2) && sizeof(s.m34) == sizeof(m34) && sizeof(s.m4) == sizeof(m4), "layout");
    static_assert(sizeof(s.a2) == sizeof(a2) && sizeof(s.a3) == sizeof(a3) && sizeof(s.box) == sizeof(box), "layout");
    std::memcpy(&s.m2, m2, sizeof(m2));
    std::memcpy(&s.m34, m34, sizeof(m34));
    std::memcpy(&s.m4, m4, sizeof(m4));
    std::memcpy(&s.a2, a2, sizeof(a2));
    std::memcpy(&s.a3, a3, sizeof(a3));
    std::memcpy(&s.box, box, sizeof(box));
    s.array[0] = 0.5f; s.array[1] = 1.f; s.array[2] = 1.5f;
    for (uint32_t i = 0; i < 20; ++i)
        s.longArray[i] = i;
    s.entities[0] = Entity::Null;
    s.entities[1] = MakeEntity(3, 1);
    for (uint32_t i = 0; i < 40; ++i)
        s.raws[i / 4].bytes[i % 4] = uint8_t(i);

    for (uint32_t f = 0; f < 14; ++f)
    {
        const FieldDesc& field = desc.fields[f];
        CHECK(std::strcmp(field.name, expected[f].name) == 0);
        CHECK(field.type == expected[f].type && field.shape == expected[f].shape);
        CHECK(field.scalarCount == expected[f].count);
        CHECK(Format(field, &s) == expected[f].text);
    }

    // Hand-made: a field at an offset; counts that are not square, affine or even.
    const float values[] = { 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const auto make = [](FieldType type, FieldShape shape, uint32_t count, uint32_t offset)
    {
        return FieldDesc{ "hand", offset, count, type, shape, {} };
    };
    CHECK(Format(make(FieldType::Float, FieldShape::Matrix, 6, 4), values) == "[1 2 3 4 5 6]");
    CHECK(Format(make(FieldType::Float, FieldShape::Matrix, 8, 4), values) == "[1 2 3 4; 5 6 7 8]");
    CHECK(Format(make(FieldType::Float, FieldShape::Matrix, 9, 4), values) == "[1 2 3; 4 5 6; 7 8 9]");
    CHECK(Format(make(FieldType::Float, FieldShape::Matrix, 1, 4), values) == "[1]");
    CHECK(Format(make(FieldType::Float, FieldShape::Affine, 7, 4), values) == "[1 2 3 4 5 6 7]");
    CHECK(Format(make(FieldType::Float, FieldShape::Affine, 2, 4), values) == "[1] t (2)");
    CHECK(Format(make(FieldType::Float, FieldShape::Box, 5, 4), values) == "min (1, 2) max (3, 4, 5)");
    CHECK(Format(make(FieldType::Float, FieldShape::Vector, 1, 0), values) == "(9)");
    CHECK(Format(make(FieldType::Float, FieldShape::Quaternion, 4, 8), values) == "(2, 3, 4, 5)");
    CHECK(Format(make(FieldType::Float, FieldShape::Array, 16, 4), values) ==
          "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]");
    const uint8_t bytes[] = { 0xde, 0xad, 0xbe, 0xef, 0x01 };
    CHECK(Format(make(FieldType::Opaque, FieldShape::Scalar, 1, 0), bytes) == "de");
    CHECK(Format(make(FieldType::Opaque, FieldShape::Vector, 5, 0), bytes) == "deadbeef01");
    CHECK(Format(make(FieldType::Opaque, FieldShape::Array, 4, 1), bytes) == "adbeef01");
    const bool flags[] = { true, false };
    CHECK(Format(make(FieldType::Boolean, FieldShape::Array, 2, 0), flags) == "[true, false]");
    const Entity links[] = { MakeEntity(1, 2), Entity::Null };
    CHECK(Format(make(FieldType::Entity, FieldShape::Vector, 2, 0), links) == "(#1:2, null)");
    const int16_t shorts[] = { -1, 2, -3, 4 };
    CHECK(Format(make(FieldType::Int16, FieldShape::Matrix, 4, 0), shorts) == "[-1 2; -3 4]");
    const uint64_t longs[] = { 1, 18446744073709551615ull };
    CHECK(Format(make(FieldType::UInt64, FieldShape::Box, 2, 0), longs) == "min (1) max (18446744073709551615)");
}

// The output is cut at the capacity: the first capacity - 1 characters of the whole text and a terminator, nothing
// written past it; a null buffer or a capacity of 0 writes nothing.
void test_format_capacity()
{
    const ComponentTypeDesc& shapes = ComponentTraits<Shapes>::Desc();
    const ComponentTypeDesc& scalars = ComponentTraits<Scalars>::Desc();
    Shapes s;
    std::memset(static_cast<void*>(&s), 0, sizeof(s));
    for (uint32_t i = 0; i < 20; ++i)
        s.longArray[i] = 1000000u + i;
    for (uint32_t i = 0; i < 40; ++i)
        s.raws[i / 4].bytes[i % 4] = uint8_t(0xf0 + i % 16);
    const double a3[] = { -1.5, 2.25, 3, 4, 5, 6, 7, 8, 9, 1e10, -1e-10, 12 };
    std::memcpy(&s.a3, a3, sizeof(a3));
    Scalars v;
    std::memset(&v, 0, sizeof(v));
    v.u64 = ~0ull;
    v.entity = MakeEntity(123456, 789);

    const std::pair<const FieldDesc*, const void*> fields[] = {
        { &FieldOf(shapes, "a3"), &s }, { &FieldOf(shapes, "longArray"), &s }, { &FieldOf(shapes, "raws"), &s },
        { &FieldOf(shapes, "box"), &s }, { &FieldOf(scalars, "u64"), &v }, { &FieldOf(scalars, "entity"), &v },
    };
    for (const auto& field : fields)
    {
        const std::string whole = Format(*field.first, field.second);
        CHECK(!whole.empty() && whole.size() < 500);
        for (uint32_t capacity = 1; capacity <= uint32_t(whole.size()) + 2; ++capacity)
        {
            std::vector<char> buffer(capacity + 8, char(0x7f));
            app::detail::FormatInspectorField(buffer.data(), capacity, *field.first, field.second);
            const size_t length = std::min<size_t>(whole.size(), capacity - 1);
            CHECK(buffer[length] == 0 && std::memcmp(buffer.data(), whole.data(), length) == 0);
            for (uint32_t i = capacity; i < capacity + 8; ++i)
                CHECK(buffer[i] == char(0x7f));
        }
        char untouched[4] = { 'x', 'y', 'z', 0 };
        app::detail::FormatInspectorField(untouched, 0, *field.first, field.second);
        CHECK(std::strcmp(untouched, "xyz") == 0);
        app::detail::FormatInspectorField(nullptr, 16, *field.first, field.second);
    }
}

// ------------------------------------------------------------------------------- views against the storage

// GetChunkInfo over archetypes of several chunks, after creation, destruction and moves: every row in one chunk of the
// archetype of its components, the counts and totals equal WorldStats; versions: a write moves only its chunk's
// column, and the chunks whose versions pass a Changed or Added filter are exactly those a query visits.
void test_chunk_info_storage()
{
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    const ComponentId b = RegisterComponent<B>(world);
    std::vector<Entity> alive = Create(world, { a }, 3000);
    const std::vector<Entity> both = Create(world, { a, b }, 2000);
    alive.insert(alive.end(), both.begin(), both.end());
    for (size_t i = 0; i < alive.size(); ++i)
        GetComponentForWrite<A>(world, alive[i])->value = float(i);
    CheckChunksAgainstStorage(world, alive, {});
    ArchetypeInfo info{};
    CHECK(world->GetArchetypeInfo(FindArchetype(world, { a }), info) == nvrhi::FS_OK && info.chunkCount >= 2);
    CHECK(world->GetArchetypeInfo(FindArchetype(world, { a, b }), info) == nvrhi::FS_OK && info.chunkCount >= 2);

    // Churn: every 7th row destroyed, every 5th of the rest of {A} moved to {A, B}.
    std::vector<Entity> dead;
    std::vector<Entity> kept;
    for (size_t i = 0; i < alive.size(); ++i)
        (i % 7 == 3 ? dead : kept).push_back(alive[i]);
    CHECK(world->DestroyEntities(dead.data(), uint32_t(dead.size())) == nvrhi::FS_OK);
    for (size_t i = 0; i < 3000; i += 5)
        if (world->IsAlive(alive[i]))
            CHECK(world->AddComponent(alive[i], b, nullptr) == nvrhi::FS_OK);
    CheckChunksAgainstStorage(world, kept, dead);
    for (size_t i = 0; i < alive.size(); ++i)   // the values travelled with their rows
        if (i % 7 != 3)
            CHECK(GetComponent<A>(world, alive[i])->value == float(i));

    // Versions: one write at a new version moves the B column of its chunk only.
    const std::vector<ChunkCopy> before = AllChunks(world);
    const uint64_t written = world->AdvanceVersion();
    const Entity target = both[1001];
    CHECK(world->IsAlive(target));
    CHECK(GetComponentForWrite<B>(world, target, written) != nullptr);
    const std::vector<ChunkCopy> after = AllChunks(world);
    CHECK(before.size() == after.size());
    uint32_t moved = 0;
    for (size_t i = 0; i < after.size(); ++i)
    {
        CHECK(before[i].entities == after[i].entities && before[i].addedVersion == after[i].addedVersion);
        ArchetypeInfo arch{};
        CHECK(world->GetArchetypeInfo(after[i].archetype, arch) == nvrhi::FS_OK);
        const bool holds = std::find(after[i].entities.begin(), after[i].entities.end(), target) !=
                           after[i].entities.end();
        for (uint32_t c = 0; c < arch.componentCount; ++c)
        {
            const bool expectMove = holds && arch.components[c] == b;
            CHECK(after[i].versions[c] == (expectMove ? written : before[i].versions[c]));
            moved += expectMove ? 1 : 0;
        }
    }
    CHECK(moved == 1);

    // Rows added at a later version stamp their chunks' added version.
    const uint64_t addedAt = world->AdvanceVersion();
    const std::vector<Entity> late = Create(world, { a, b }, 40);
    CHECK(world->GetVersion() == addedAt);
    const std::vector<ChunkCopy> chunks = AllChunks(world);
    for (const ChunkCopy& chunk : chunks)
        for (Entity entity : late)
            if (std::find(chunk.entities.begin(), chunk.entities.end(), entity) != chunk.entities.end())
                CHECK(chunk.addedVersion == addedAt);

    // The filters compare what GetChunkInfo shows: rows a Changed or Added query visits are those of the chunks whose
    // shown version passes.
    Query<Changed<B>> changed(world, "changed B");
    Query<Added<A>> added(world, "added A");
    CHECK(changed && added);
    for (const uint64_t since : { uint64_t(0), written - 1, written, addedAt - 1, addedAt })
    {
        uint32_t expectChanged = 0;
        uint32_t expectAdded = 0;
        for (const ChunkCopy& chunk : chunks)
        {
            ArchetypeInfo arch{};
            CHECK(world->GetArchetypeInfo(chunk.archetype, arch) == nvrhi::FS_OK);
            for (uint32_t c = 0; c < arch.componentCount; ++c)
            {
                if (arch.components[c] == b && chunk.versions[c] > since)
                    expectChanged += uint32_t(chunk.entities.size());
                if (arch.components[c] == a && chunk.addedVersion > since)
                    expectAdded += uint32_t(chunk.entities.size());
            }
        }
        ChunkRequest request{};
        request.sinceVersion = since;
        uint32_t visitedChanged = 0;
        uint32_t visitedAdded = 0;
        changed.ForEachChunk(request, [&](const ChunkView& view, const B*) { visitedChanged += view.count; });
        added.ForEachChunk(request, [&](const ChunkView& view, const A*) { visitedAdded += view.count; });
        CHECK(visitedChanged == expectChanged && visitedAdded == expectAdded);
    }
}

namespace
{
    nvrhi::AutoPtr<ISystem> MakeOrderedSystem(const char* name, Phase phase, SystemFlags flags,
                                              std::function<void(ISystemContext*)> update,
                                              const char* runAfter = nullptr)
    {
        SystemDesc desc{};
        desc.name = name;
        desc.phase = phase;
        desc.flags = flags;
        desc.runAfter = runAfter ? &runAfter : nullptr;
        desc.runAfterCount = runAfter ? 1 : 0;
        nvrhi::AutoPtr<ISystem> system;
        CHECK(CreateLambdaSystem(desc, std::move(update), &system) == nvrhi::FS_OK);
        return system;
    }
}

// GetSystemInfo against six frames of runs: the listed order is the order the systems ran in (constraints given against
// registration order, a named phase, enabling and disabling between frames), run counts are the calls observed, and
// the statistics are GetSystemStats'.
void test_system_info_storage()
{
    Env env;
    IScheduler* scheduler = env.scheduler.Get();
    Phase custom = Phase::Count;
    CHECK(scheduler->AddPhase("custom", Phase::PreSimulation, custom) == nvrhi::FS_OK);
    std::mutex mutex;
    std::vector<std::string> ran;
    std::map<std::string, uint64_t> calls;
    const auto record = [&](const char* name)
    {
        return [&, name](ISystemContext*)
        {
            std::lock_guard<std::mutex> lock(mutex);
            ran.push_back(name);
            ++calls[name];
        };
    };
    std::map<std::string, SystemId> ids;
    ids["end"] = env.Add(MakeOrderedSystem("end", Phase::EndFrame, SystemFlags::MainThread | SystemFlags::StartDisabled,
                                           record("end")));
    ids["c"] = env.Add(MakeOrderedSystem("c", Phase::PreSimulation, SystemFlags::MainThread, record("c"), "b"));
    ids["a"] = env.Add(MakeOrderedSystem("a", Phase::PreSimulation, SystemFlags::MainThread, record("a")));
    ids["b"] = env.Add(MakeOrderedSystem("b", Phase::PreSimulation, SystemFlags::MainThread, record("b"), "a"));
    ids["y"] = env.Add(MakeOrderedSystem("y", custom, SystemFlags::MainThread, record("y"), "x"));
    ids["x"] = env.Add(MakeOrderedSystem("x", custom, SystemFlags::Exclusive, record("x")));
    ids["begin"] = env.Add(MakeOrderedSystem("begin", Phase::BeginFrame, SystemFlags::MainThread, record("begin")));
    CHECK(scheduler->GetSystemCount() == 7);

    for (uint32_t frame = 0; frame < 6; ++frame)
    {
        if (frame == 2)
            CHECK(scheduler->SetSystemEnabled(ids["end"], true) == nvrhi::FS_OK);
        if (frame == 4)
            CHECK(scheduler->SetSystemEnabled(ids["a"], false) == nvrhi::FS_OK);
        std::vector<std::string> listed;
        for (uint32_t i = 0; i < scheduler->GetSystemCount(); ++i)
        {
            SystemInfo info{};
            CHECK(scheduler->GetSystemInfo(i, info) == nvrhi::FS_OK);
            CHECK(info.name && ids.at(info.name) == info.id);
            CHECK(info.enabled == ((frame >= 2 || std::strcmp(info.name, "end") != 0) &&
                                   (frame < 4 || std::strcmp(info.name, "a") != 0) ? 1u : 0u));
            if (info.enabled)
                listed.push_back(info.name);
        }
        ran.clear();
        CHECK(scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
        CHECK(ran == listed);
    }
    const std::vector<std::string> order = { "begin", "a", "b", "c", "x", "y", "end" };
    const std::map<std::string, Phase> phaseOf = { { "begin", Phase::BeginFrame }, { "a", Phase::PreSimulation },
        { "b", Phase::PreSimulation }, { "c", Phase::PreSimulation }, { "x", custom }, { "y", custom },
        { "end", Phase::EndFrame } };
    for (uint32_t i = 0; i < scheduler->GetSystemCount(); ++i)
    {
        SystemInfo info{};
        CHECK(scheduler->GetSystemInfo(i, info) == nvrhi::FS_OK);
        CHECK(order[i] == info.name && info.phase == phaseOf.at(info.name));
        CHECK(info.stats.runCount == calls[info.name]);
        SystemStats stats{};
        CHECK(scheduler->GetSystemStats(info.id, stats) == nvrhi::FS_OK);
        CHECK(SameBytes(stats, info.stats));
        CHECK(info.stats.lastMilliseconds >= 0.0 && info.stats.averageMilliseconds >= 0.0);
    }
    CHECK(calls["a"] == 4 && calls["end"] == 4 && calls["begin"] == 6 && calls["x"] == 6);
}

// A phase with an ordering cycle lists its systems in registration order and does not run; the build's warnings are
// logged.
// Stage T8 (mutant I3-40): GetSystemInfo of a scheduler is refused while a phase of another scheduler of the same world
// runs (the world is in a phase; its systems may be writing the statistics of the shared world's resources).
void test_system_info_other_scheduler()
{
    Env env;
    nvrhi::AutoPtr<IScheduler> second;
    CHECK(CreateScheduler(env.world.Get(), env.jobs.Get(), &second) == nvrhi::FS_OK);
    SystemId listed = SystemId::Invalid;
    CHECK(second->AddSystem(MakeSystem("listed", Phase::PreSimulation, SystemFlags::None, [](ISystemContext*) {}),
                            listed) == nvrhi::FS_OK);
    SystemInfo outside{};
    CHECK(second->GetSystemInfo(0, outside) == nvrhi::FS_OK && outside.id == listed);
    nvrhi::FRESULT inPhase = nvrhi::FS_OK;
    uint32_t zeroed = 0;
    env.Add(MakeSystem("asks", Phase::PreSimulation, SystemFlags::MainThread, [&](ISystemContext*)
    {
        SystemInfo info{};
        info.enabled = 7;
        inPhase = second->GetSystemInfo(0, info);
        zeroed = info.enabled == 0 && !info.name ? 1 : 0;
    }));
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
    CHECK(inPhase == nvrhi::FE_OUT_OF_DATE && zeroed == 1);
    CHECK(second->GetSystemInfo(0, outside) == nvrhi::FS_OK);
}

void test_system_info_cycle()
{
    Env env;
    IScheduler* scheduler = env.scheduler.Get();
    uint32_t runs = 0;
    const uint32_t warnings = g_Warnings.load();
    env.Add(MakeOrderedSystem("first", Phase::BeginFrame, SystemFlags::MainThread, [&](ISystemContext*) {}));
    env.Add(MakeOrderedSystem("p", Phase::PreSimulation, SystemFlags::MainThread, [&](ISystemContext*) { ++runs; },
                              "q"));
    env.Add(MakeOrderedSystem("r", Phase::PreSimulation, SystemFlags::MainThread, [&](ISystemContext*) { ++runs; }));
    env.Add(MakeOrderedSystem("q", Phase::PreSimulation, SystemFlags::MainThread, [&](ISystemContext*) { ++runs; },
                              "p"));
    const char* const expected[] = { "first", "p", "r", "q" };
    for (uint32_t i = 0; i < 4; ++i)
    {
        SystemInfo info{};
        CHECK(scheduler->GetSystemInfo(i, info) == nvrhi::FS_OK);
        CHECK(std::strcmp(info.name, expected[i]) == 0);
    }
    (void)scheduler->RunPhase(Phase::PreSimulation);
    CHECK(runs == 0);
    TakeWarnings(warnings, ~0u, 1);
}

// Every panel's text (ImGui's log of one frame, all nodes open) against the storage: the world totals, each
// archetype's header and sizes, each chunk's row count in the version table, the systems by phase, the browser's page,
// and the selected entity's components, each field the formatter's text of the stored value; editable values offered
// as stored, and once edited as pending. Goldens for a node's LocalTransform, WorldTransform and LocalBounds.
void test_panels_show_storage()
{
    ImGuiContextScope context;
    SceneRig rig(2, { MakeLocal(dm::double3(2.0, 0.0, 0.0)), MakeLocal(dm::double3(0.0, 1.0, 0.0)) },
                 { MakeLocal(dm::double3(4.0, 1.0, -2.0)) });
    IWorld* world = rig.World();
    IScheduler* scheduler = rig.graph->GetScheduler();
    CHECK(rig.graph->Simulate(FrameTime{ 1.0 / 60.0, 1.0 / 60.0, 0 }) == nvrhi::FS_OK);   // run counts
    rig.Refresh();
    app::WorldInspector inspector;
    inspector.SetEntityNames([](void*, Entity entity, char* out, uint32_t capacity)
    {
        return std::snprintf(out, capacity, "node %u", GetEntityIndex(entity)) > 0;
    }, nullptr);
    const auto label = [](Entity entity)
    {
        char text[64];
        std::snprintf(text, sizeof(text), "node %u  #%u:%u", GetEntityIndex(entity), GetEntityIndex(entity),
                      GetEntityGeneration(entity));
        return std::string(text);
    };

    for (const Entity selected : { rig.nodeRows[0], rig.nodeRows[1], rig.runtimeRows[0] })
    {
        OfferLog warm;
        DrawSelecting(inspector, world, scheduler, selected, warm, true);   // a new window is hidden for a frame
        OfferLog offers;
        std::string text;
        WorldStats stats{};
        world->GetStats(stats);
        DrawSelecting(inspector, world, scheduler, selected, offers, true, &text);
        char line[512];

        // World.
        std::snprintf(line, sizeof(line), "entities %u (capacity %u), component types %u, sparse sets %u",
                      stats.entityCount, stats.entityCapacity, stats.componentTypeCount, stats.sparseSetCount);
        CHECK(Contains(text, line));
        std::snprintf(line, sizeof(line), "archetypes %u, chunks %u: %.2f MB reserved, %.2f MB in rows (%.1f %%)",
                      stats.archetypeCount, stats.chunkCount, double(stats.chunkBytes) / (1024.0 * 1024.0),
                      double(stats.rowBytes) / (1024.0 * 1024.0),
                      stats.chunkBytes ? 100.0 * double(stats.rowBytes) / double(stats.chunkBytes) : 0.0);
        CHECK(Contains(text, line));
        std::snprintf(line, sizeof(line), "version %llu, structural version %llu; pending edits 0",
                      (unsigned long long)stats.version, (unsigned long long)stats.structuralVersion);
        CHECK(Contains(text, line));

        // Archetypes: the header, the sizes and every chunk's rows.
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            std::string header = "#" + std::to_string(a) + ": " + std::to_string(info.entityCount) + " rows, " +
                                 std::to_string(info.chunkCount) + " chunks:";
            for (uint32_t c = 0; c < info.componentCount; ++c)
                header += std::string(" ") + world->GetComponentTypeDesc(info.components[c])->name;
            CHECK(Contains(text, header));
            std::snprintf(line, sizeof(line), "row %u B, chunk %u B, %u rows per chunk", info.rowSize,
                          info.chunkSize, info.rowsPerChunk);
            CHECK(Contains(text, line));
            for (uint32_t c = 0; c < info.chunkCount; ++c)
            {
                ChunkInfo chunk{};
                CHECK(world->GetChunkInfo(a, c, chunk) == nvrhi::FS_OK);
                std::snprintf(line, sizeof(line), "%u/%u", chunk.count, chunk.capacity);
                CHECK(Contains(text, line));
            }
        }

        // Systems, in run order, under their phases.
        size_t at = 0;
        Phase phase = Phase::Count;
        for (uint32_t i = 0; i < scheduler->GetSystemCount(); ++i)
        {
            SystemInfo info{};
            CHECK(scheduler->GetSystemInfo(i, info) == nvrhi::FS_OK);
            if (i == 0 || info.phase != phase)
            {
                phase = info.phase;
                const std::string name = std::string("phase ") + scheduler->GetPhaseName(phase);
                at = text.find(name, at);
                CHECK(Contains(text, name) && at != std::string::npos);
            }
            at = text.find(std::string("  ") + info.name, at);
            CHECK(Contains(text, std::string("  ") + info.name) && at != std::string::npos);
        }

        // The browser's page: archetype 0, chunk 0.
        ArchetypeInfo first{};
        CHECK(world->GetArchetypeInfo(0, first) == nvrhi::FS_OK);
        if (first.chunkCount == 0)
        {
            CHECK(Contains(text, "no rows"));
        }
        else
        {
            ChunkInfo chunk{};
            CHECK(world->GetChunkInfo(0, 0, chunk) == nvrhi::FS_OK);
            std::snprintf(line, sizeof(line), "chunk 0 of %u: %u of %u rows", first.chunkCount, chunk.count,
                          chunk.capacity);
            CHECK(Contains(text, line));
            for (uint32_t r = 0; r < chunk.count; ++r)
                CHECK(Contains(text, label(chunk.entities[r])));
        }

        // The selected entity: its label, then every component it has, read-only ones field by field.
        CHECK(Contains(text, label(selected)));
        const ComponentId local = FindComponent<LocalTransform>(world);
        CHECK(offers.Set() == std::set<ComponentId>{ local });
        for (uint32_t t = 0; t < world->GetComponentTypeCount(); ++t)
        {
            const ComponentId component = ComponentId(t);
            const ComponentTypeDesc* desc = world->GetComponentTypeDesc(component);
            if (!desc || !!(desc->flags & ComponentFlags::Resource) || !world->HasComponent(selected, component))
                continue;
            const bool editable = component == local;
            CHECK(Contains(text, std::string(desc->name) + (editable ? "" : " (read-only)")));
            if (!!(desc->flags & ComponentFlags::Tag))
                continue;
            const void* value = world->GetComponent(selected, component);
            CHECK(value || !!(desc->flags & ComponentFlags::Chunk));
            if (!value)
                continue;
            if (editable)
            {
                const std::vector<uint8_t> shown = offers.Shown(component);
                CHECK(shown.size() == desc->size && std::memcmp(shown.data(), value, desc->size) == 0);
                continue;
            }
            for (uint32_t f = 0; f < desc->fieldCount; ++f)
            {
                const FieldDesc& field = desc->fields[f];
                if (field.type == FieldType::Entity && field.shape == FieldShape::Scalar)
                {
                    Entity link;
                    std::memcpy(&link, static_cast<const uint8_t*>(value) + field.offset, sizeof(link));
                    CHECK(Contains(text, std::string(field.name) + ":"));
                    CHECK(Contains(text, link == Entity::Null ? std::string("null") : label(link)));
                }
                else
                    CHECK(Contains(text, std::string(field.name) + ": " + Format(field, value)));
            }
            if (desc->fieldCount == 0)
            {
                const FieldDesc bytes{ "bytes", 0, desc->size, FieldType::Opaque, FieldShape::Array, {} };
                CHECK(Contains(text, "bytes: " + Format(bytes, value)));
            }
        }
    }

    // Goldens: node 0 at (2, 0, 0) under the root, its mesh box.
    const Entity node = rig.nodeRows[0];
    const ComponentTypeDesc* worldDesc = world->GetComponentTypeDesc(FindComponent<WorldTransform>(world));
    const ComponentTypeDesc* localDesc = world->GetComponentTypeDesc(FindComponent<LocalTransform>(world));
    const ComponentTypeDesc* boundsDesc = world->GetComponentTypeDesc(FindComponent<LocalBounds>(world));
    CHECK(Format(FieldOf(*worldDesc, "transform"), GetComponent<WorldTransform>(world, node)) ==
          "[1 0 0; 0 1 0; 0 0 1] t (2, 0, 0)");
    CHECK(Format(FieldOf(*localDesc, "translation"), GetComponent<LocalTransform>(world, node)) == "(2, 0, 0)");
    CHECK(Format(FieldOf(*localDesc, "rotation"), GetComponent<LocalTransform>(world, node)) == "(1, 0, 0, 0)");
    CHECK(Format(FieldOf(*localDesc, "scaling"), GetComponent<LocalTransform>(world, node)) == "(1, 1, 1)");
    CHECK(Format(FieldOf(*boundsDesc, "bounds"), GetComponent<LocalBounds>(world, node)) ==
          "min (-1, -1, -1) max (1, 2, 1)");

    // A pending edit: offered as pending (not as stored), counted by the World panel, marked; the store unchanged.
    LocalTransform edited = *GetComponent<LocalTransform>(world, node);
    edited.translation = dm::double3(7.0, 8.0, 9.0);
    CHECK(Commit(inspector, world, node, edited));
    OfferLog offers;
    std::string text;
    DrawSelecting(inspector, world, scheduler, node, offers, true, &text);
    const std::vector<uint8_t> shown = offers.Shown(FindComponent<LocalTransform>(world));
    CHECK(shown.size() == sizeof(edited) && std::memcmp(shown.data(), &edited, sizeof(edited)) == 0);
    CHECK(Contains(text, "pending edits 1") && Contains(text, "edited; Apply writes it"));
    CHECK(GetComponent<LocalTransform>(world, node)->translation.x == 2.0);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
}

// The Systems panel lists only `world`'s systems: Draw with another world's scheduler warns and skips the panel.
// The regression test of backlog I3-R6 (Draw did not check that the scheduler belongs to the world; fixed in F2).
void test_systems_panel_other_world()
{
    ImGuiContextScope context;
    Env mine;
    Env other;
    other.Add(MakeSystem("other_world_system", Phase::PreSimulation, SystemFlags::None, [](ISystemContext*) {}));
    const ComponentId a = RegisterComponent<A>(mine.world.Get());
    Create(mine.world.Get(), { a }, 1);
    app::WorldInspector inspector;
    OfferLog offers;
    const uint32_t before = g_Warnings.load();
    DrawSelecting(inspector, mine.world.Get(), other.scheduler.Get(), Entity::Null, offers, true);   // warns too
    const uint32_t warnings = g_Warnings.load();
    std::string text;
    DrawSelecting(inspector, mine.world.Get(), other.scheduler.Get(), Entity::Null, offers, true, &text);
    const uint32_t seen = g_Warnings.load() - warnings;
    g_Warnings = before;   // the first Draw's warning too
    CHECK(text.find("other_world_system") == std::string::npos);
    CHECK(seen == 1);
}

// --------------------------------------------------------------------------------------- what is editable

// In a graph's world the inspector offers, of donut's components, only the homes of authored state: LocalTransform of
// nodes (plain and with a mesh) and runtime-only rows; nothing of driven nodes (I3-R2, F8: SimPose and SimScale are
// read-only, and LocalTransform is on a driven row). Every component of donut's read-only list (each present on one
// of these rows) and the hooked ones are refused without the application naming them.
void test_offers_scene_rows()
{
    ImGuiContextScope context;
    SceneRig rig(2, { MakeLocal(dm::double3(1.0, 0.0, 0.0)), MakeLocal(dm::double3(0.0, 1.0, 0.0)),
                      MakeLocal(dm::double3(0.0, 0.0, 1.0)) },
                 { MakeLocal(dm::double3(3.0, 0.0, 0.0)) });
    IWorld* world = rig.World();
    CHECK(rig.graph->SetSimulationDriven(rig.nodes[2].Get(), true) == nvrhi::FS_OK);   // a mesh node, driven
    rig.Refresh();
    ComponentId list[c_MaxInspectorReadOnlyComponents] = {};
    const uint32_t listed = app::GetDonutReadOnlyComponents(world, list, c_MaxInspectorReadOnlyComponents);
    CHECK(listed >= 14);
    const ComponentId local = FindComponent<LocalTransform>(world);
    const std::set<ComponentId> authored = { local, FindComponent<SimPose>(world), FindComponent<SimScale>(world),
                                             FindComponent<SimPosePrevious>(world), FindComponent<SimTeleport>(world) };
    app::WorldInspector inspector;
    std::set<ComponentId> present;
    for (const Entity row : { rig.nodeRows[0], rig.nodeRows[1], rig.nodeRows[2], rig.runtimeRows[0] })
    {
        OfferLog offers;
        DrawSelecting(inspector, world, rig.graph->GetScheduler(), row, offers, true);
        for (uint32_t t = 0; t < world->GetComponentTypeCount(); ++t)
            if (world->HasComponent(row, ComponentId(t)))
                present.insert(ComponentId(t));
        for (const ComponentId offered : offers.Set())
        {
            CHECK(authored.count(offered) == 1);
            CHECK(std::find(list, list + listed, offered) == list + listed);
        }
        if (row == rig.nodeRows[2])
        {
            CHECK(offers.Set().empty());
        }
        else
            CHECK(offers.Set() == std::set<ComponentId>{ local });
    }
    for (uint32_t i = 0; i < listed; ++i)
        CHECK(present.count(list[i]) == 1);   // every listed component was on a row and refused there
    CHECK(present.count(FindComponent<SceneNode>(world)) == 1 && present.count(FindComponent<LeafObject>(world)) == 1);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
}

// Donut's list holds the simulation inputs that the domain owns: SimTeleport (an edit skips RequestTeleport and cannot
// enable the request) and SimPosePrevious (derived by the domain); a driven node offers neither.
// The regression test of backlog I3-R1 (SimTeleport was missing from the list) and I3-R2 (SimPosePrevious, decided
// 2026-11-26); written failing in T3, passing with the fixer's batch F1.
void test_offers_simulation_inputs()
{
    ImGuiContextScope context;
    SceneRig rig(2, { MakeLocal(dm::double3(1.0, 0.0, 0.0)) }, {});
    IWorld* world = rig.World();
    CHECK(rig.graph->SetSimulationDriven(rig.nodes[0].Get(), true) == nvrhi::FS_OK);
    ComponentId list[c_MaxInspectorReadOnlyComponents] = {};
    const uint32_t listed = app::GetDonutReadOnlyComponents(world, list, c_MaxInspectorReadOnlyComponents);
    app::WorldInspector inspector;
    OfferLog offers;
    DrawSelecting(inspector, world, nullptr, rig.nodeRows[0], offers, true);
    CHECK(world->HasComponent(rig.nodeRows[0], FindComponent<SimTeleport>(world)));
    CHECK(!offers.Offered(FindComponent<SimTeleport>(world)) && !offers.Offered(FindComponent<SimPosePrevious>(world)));
    CHECK(std::find(list, list + listed, FindComponent<SimTeleport>(world)) != list + listed);
    CHECK(std::find(list, list + listed, FindComponent<SimPosePrevious>(world)) != list + listed);
}

// Without a graph's list (a plain world with the scene components registered), the serialisable Relocatable ones are
// offered: in a graph's world the list alone refuses them. The application's list refuses more and is replaced, cleared
// and refused when null; NoSerialize, Shared, Chunk and hooked scene components are never offered.
void test_offers_without_list()
{
    ImGuiContextScope context;
    Env env;
    IWorld* world = env.world.Get();
    const std::vector<ComponentId> open = {
        RegisterComponent<SceneLinks>(world), RegisterComponent<Parent>(world), RegisterComponent<LocalTransform>(world),
        RegisterComponent<WorldTransform>(world), RegisterComponent<LocalBounds>(world),
        RegisterComponent<WorldBounds>(world), RegisterComponent<MeshRef>(world), RegisterComponent<RenderFlags>(world),
        RegisterComponent<SimBodyRef>(world),
    };
    const std::vector<ComponentId> closed = {
        RegisterComponent<LocalMatrix>(world), RegisterComponent<GpuInstanceSlot>(world),
        RegisterComponent<HierarchyDepth>(world), RegisterComponent<ChunkWorldBounds>(world),
        RegisterComponent<SceneNode>(world), RegisterComponent<LeafObject>(world),
    };
    std::vector<ComponentId> all = open;
    all.insert(all.end(), closed.begin(), closed.end());
    for (ComponentId id : all)
        CHECK(id != ComponentId::Invalid);
    const Entity entity = Create(world, all, 1)[0];
    CHECK(app::GetDonutReadOnlyComponents(world, nullptr, 0) == 0);

    app::WorldInspector inspector;
    const auto offered = [&]()
    {
        OfferLog offers;
        DrawSelecting(inspector, world, nullptr, entity, offers, true);
        return offers.Set();
    };
    CHECK(offered() == std::set<ComponentId>(open.begin(), open.end()));
    const ComponentId application[] = { open[1], open[6] };   // Parent, MeshRef
    inspector.SetReadOnly(application, 2);
    std::set<ComponentId> expected(open.begin(), open.end());
    expected.erase(open[1]);
    expected.erase(open[6]);
    CHECK(offered() == expected);
    inspector.SetReadOnly(application + 1, 1);   // replaces
    expected.insert(open[1]);
    CHECK(offered() == expected);
    inspector.SetReadOnly(nullptr, 0);
    CHECK(offered() == std::set<ComponentId>(open.begin(), open.end()));
    inspector.SetReadOnly(application, 2);
    const uint32_t warnings = g_Warnings.load();
    inspector.SetReadOnly(nullptr, 2);   // no list: a warning, cleared
    TakeWarnings(warnings, 1);
    CHECK(offered() == std::set<ComponentId>(open.begin(), open.end()));
}

// By flags: tags, buffers, Shared and Chunk values, NoSerialize, hooked (not Relocatable) and values above 256 B are
// never offered; 256 B, Enableable (also disabled) and Sparse values are, and their edits are applied. Fields of type
// Entity and Opaque are shown, not edited.
void test_offers_by_flags()
{
    ImGuiContextScope context;
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId tag = RegisterComponent<TagC>(world);
    const ComponentId buffer = RegisterComponent<Element>(world);
    const ComponentId shared = RegisterComponent<SharedC>(world);
    const ComponentId chunk = RegisterComponent<ChunkC>(world);
    const ComponentId noSer = RegisterComponent<NoSer>(world);
    const ComponentId hooked = RegisterComponent<Hooked>(world);
    const ComponentId big = RegisterComponent<Big>(world);
    const ComponentId edge = RegisterComponent<Edge>(world);
    const ComponentId toggle = RegisterComponent<Toggle>(world);
    const ComponentId sparse = RegisterComponent<SparseC>(world);
    const ComponentId mixed = RegisterComponent<Mixed>(world);
    CHECK(!(world->GetComponentTypeDesc(hooked)->flags & ComponentFlags::Relocatable));
    const Entity entity = Create(world, { tag, buffer, shared, chunk, noSer, hooked, big, edge, toggle, mixed }, 1)[0];
    CHECK(world->AddComponent(entity, sparse, nullptr) == nvrhi::FS_OK);
    CHECK(world->SetEnabled(entity, toggle, false) == nvrhi::FS_OK);
    const Entity linked = Create(world, { edge }, 1)[0];
    Mixed m{ 1.5f, linked, Raw{ { 0x0a, 0x0b, 0x0c, 0x0d } } };
    *GetComponentForWrite<Mixed>(world, entity) = m;

    app::WorldInspector inspector;
    OfferLog offers;
    std::string text;
    DrawSelecting(inspector, world, nullptr, entity, offers, true);
    DrawSelecting(inspector, world, nullptr, entity, offers, true, &text);
    CHECK(offers.Set() == (std::set<ComponentId>{ edge, toggle, sparse, mixed }));
    CHECK(Contains(text, "Toggle (disabled)") && Contains(text, "Big (read-only)") && Contains(text, "TagC (read-only)"));
    CHECK(Contains(text, "Hooked (read-only)") && Contains(text, "NoSer (read-only)"));
    CHECK(Contains(text, "raw: 0a0b0c0d") && Contains(text, "e:"));   // Mixed's shown-only fields
    CHECK(Contains(text, "1.5"));                                         // its number field, an input

    Edge e{};
    e.values[63] = 63.f;
    CHECK(Commit(inspector, world, entity, e));
    CHECK(Commit(inspector, world, entity, Toggle{ 4.f }));
    CHECK(Commit(inspector, world, entity, SparseC{ 5.f }));
    Mixed changed = m;
    changed.f = -2.f;
    CHECK(Commit(inspector, world, entity, changed));
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(GetComponent<Edge>(world, entity)->values[63] == 63.f && GetComponent<Toggle>(world, entity)->value == 4.f);
    CHECK(!world->IsEnabled(entity, toggle));   // the value only
    CHECK(GetComponent<SparseC>(world, entity)->value == 5.f);
    CHECK(GetComponent<Mixed>(world, entity)->f == -2.f && GetComponent<Mixed>(world, entity)->e == linked);
}

// ------------------------------------------------------------------------------- edits reaching the world

// An edit of a node's LocalTransform: pending until Apply, written by it (the node API reads it at once), and seen by
// the next Refresh (E7.0's version rule: Apply advances the version) for the node and its children, bit-identical to a
// graph built with that transform and to the node API; a second edit with no phase in between too. A raw write that
// does not move the version is not seen (the control).
void test_edit_node_reaches_world()
{
    ImGuiContextScope context;
    const std::vector<LocalTransform> start = { MakeLocal(dm::double3(1.0, 0.0, 0.0)),
                                                MakeLocal(dm::double3(0.0, 1.0, 0.0)),
                                                MakeLocal(dm::double3(0.0, 0.0, 1.0)) };
    const LocalTransform first = MakeLocal(dm::double3(5.0, -2.0, 3.0), Rotation(0.7), dm::double3(1.0, 2.0, 3.0));
    const LocalTransform second = MakeLocal(dm::double3(-1.0, 0.5, 0.25), Rotation(-1.3), dm::double3(0.5));
    SceneRig rig(2, start, {});
    IWorld* world = rig.World();
    std::vector<LocalTransform> built = start;
    built[0] = first;
    SceneRig twin(2, built, {});

    const WorldTransform stale = *GetComponent<WorldTransform>(world, rig.nodeRows[1]);
    app::WorldInspector inspector;
    CHECK(Commit(inspector, world, rig.nodeRows[0], first));
    CHECK(SameBytes(*GetComponent<LocalTransform>(world, rig.nodeRows[0]), start[0]));   // pending
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(SameBytes(*GetComponent<LocalTransform>(world, rig.nodeRows[0]), first));
    CHECK(SameBytes(rig.nodes[0]->GetTranslation(), first.translation));
    CHECK(SameBytes(rig.nodes[0]->GetRotation(), first.rotation));
    CHECK(SameBytes(*GetComponent<WorldTransform>(world, rig.nodeRows[1]), stale));   // until Refresh
    rig.Refresh();
    CheckSameWorld(rig, rig.nodeRows, twin, twin.nodeRows);
    CHECK(!SameBytes(*GetComponent<WorldTransform>(world, rig.nodeRows[1]), stale));   // the child moved
    for (size_t i = 0; i < rig.nodes.size(); ++i)
    {
        const dm::daffine3 transform = rig.nodes[i]->GetLocalToWorldTransform();
        CHECK(SameBytes(transform, GetComponent<WorldTransform>(world, rig.nodeRows[i])->transform));
    }

    // A second edit, no phase in between; the twin takes it through the node API.
    CHECK(Commit(inspector, world, rig.nodeRows[1], second));
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    rig.Refresh();
    twin.nodes[1]->SetTransform(&second.translation, &second.rotation, &second.scaling);
    twin.Refresh();
    CheckSameWorld(rig, rig.nodeRows, twin, twin.nodeRows);

    // Two edits of one node before one Apply: the last is written.
    CHECK(Commit(inspector, world, rig.nodeRows[2], first));
    CHECK(Commit(inspector, world, rig.nodeRows[2], second));
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    rig.Refresh();
    twin.nodes[2]->SetTransform(&second.translation, &second.rotation, &second.scaling);
    twin.Refresh();
    CheckSameWorld(rig, rig.nodeRows, twin, twin.nodeRows);

    // The control: a raw write outside phases does not move the version (SceneGraph::Refresh's documentation).
    const WorldTransform before = *GetComponent<WorldTransform>(world, rig.nodeRows[2]);
    GetComponentForWrite<LocalTransform>(world, rig.nodeRows[2])->translation = dm::double3(100.0);
    rig.Refresh();
    CHECK(SameBytes(*GetComponent<WorldTransform>(world, rig.nodeRows[2]), before));
}

// An edit of a runtime-only row's LocalTransform reaches its WorldTransform and WorldBounds after Refresh, bit-identical
// to a row made with that transform; the other rows, the slots and the mesh reference are unchanged.
void test_edit_runtime_reaches_world()
{
    ImGuiContextScope context;
    const std::vector<LocalTransform> start = { MakeLocal(dm::double3(0.0, 1.0, -2.0)),
                                                MakeLocal(dm::double3(1.0, 1.0, -2.0)),
                                                MakeLocal(dm::double3(2.0, 1.0, -2.0)) };
    const LocalTransform edited = MakeLocal(dm::double3(-7.0, 3.0, 11.0), Rotation(2.1), dm::double3(2.0, 1.0, 0.5));
    SceneRig rig(2, { MakeLocal(dm::double3(1.0, 0.0, 0.0)) }, start);
    IWorld* world = rig.World();
    std::vector<LocalTransform> built = start;
    built[1] = edited;
    SceneRig twin(2, { MakeLocal(dm::double3(1.0, 0.0, 0.0)) }, built);

    const GpuInstanceSlot slot = *GetComponent<GpuInstanceSlot>(world, rig.runtimeRows[1]);
    const MeshRef mesh = *GetComponent<MeshRef>(world, rig.runtimeRows[1]);
    const WorldTransform stale = *GetComponent<WorldTransform>(world, rig.runtimeRows[1]);
    const uint64_t others = rig.Hash({ rig.runtimeRows[0], rig.runtimeRows[2], rig.nodeRows[0] });
    app::WorldInspector inspector;
    CHECK(Commit(inspector, world, rig.runtimeRows[1], edited));
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(SameBytes(*GetComponent<WorldTransform>(world, rig.runtimeRows[1]), stale));   // until Refresh
    rig.Refresh();
    CheckSameWorld(rig, rig.runtimeRows, twin, twin.runtimeRows);
    CheckSameWorld(rig, rig.nodeRows, twin, twin.nodeRows);
    CHECK(rig.Hash({ rig.runtimeRows[0], rig.runtimeRows[2], rig.nodeRows[0] }) == others);
    CHECK(SameBytes(*GetComponent<GpuInstanceSlot>(world, rig.runtimeRows[1]), slot));
    CHECK(SameBytes(*GetComponent<MeshRef>(world, rig.runtimeRows[1]), mesh));
}

namespace
{
    // The script of edits_deterministic: frame f edits 8 rows, nodes and runtime rows in turn.
    struct ScriptEdit { bool node; uint32_t index; LocalTransform value; };
    std::vector<ScriptEdit> ScriptFrame(uint32_t frame, uint32_t nodes, uint32_t runtime)
    {
        std::vector<ScriptEdit> edits;
        for (uint32_t k = 0; k < 8; ++k)
        {
            const uint32_t j = frame * 5 + k * 3;
            const bool node = k % 2 == 0;
            const LocalTransform value = MakeLocal(dm::double3(double(k) - 3.0, 0.25 * double(frame), -double(k % 3)),
                                                   Rotation(0.1 * double(frame * 7 + k)),
                                                   dm::double3(1.0 + 0.125 * double(k % 4)));
            edits.push_back({ node, node ? j % nodes : j % runtime, value });
        }
        return edits;
    }

    std::vector<LocalTransform> StartLocals(uint32_t count, double y)
    {
        std::vector<LocalTransform> locals;
        for (uint32_t i = 0; i < count; ++i)
            locals.push_back(MakeLocal(dm::double3(double(i), y, 0.5 * double(i % 3))));
        return locals;
    }

    constexpr uint32_t c_ScriptNodes = 31;
    constexpr uint32_t c_ScriptRuntime = 24;
    constexpr uint32_t c_ScriptFrames = 6;

    // Per frame: the nodes' hash and the runtime rows' hash, after the frame's edits (the inspector's, or with
    // `api` the node API for nodes) and Refresh; odd frames run Simulate first.
    std::vector<std::pair<uint64_t, uint64_t>> RunScript(uint32_t workers, bool api)
    {
        SceneRig rig(workers, StartLocals(c_ScriptNodes, 0.0), StartLocals(c_ScriptRuntime, 1.0));
        IWorld* world = rig.World();
        app::WorldInspector inspector;
        std::vector<std::pair<uint64_t, uint64_t>> hashes;
        for (uint32_t frame = 0; frame < c_ScriptFrames; ++frame)
        {
            rig.frame = frame + 1;
            if (frame % 2 == 1)
                CHECK(rig.graph->Simulate(FrameTime{ double(frame) / 60.0, 1.0 / 60.0, frame }) == nvrhi::FS_OK);
            for (const ScriptEdit& edit : ScriptFrame(frame, c_ScriptNodes, c_ScriptRuntime))
            {
                if (api && edit.node)
                    rig.nodes[edit.index]->SetTransform(&edit.value.translation, &edit.value.rotation,
                                                        &edit.value.scaling);
                else
                    CHECK(Commit(inspector, world, edit.node ? rig.nodeRows[edit.index] : rig.runtimeRows[edit.index],
                                 edit.value));
            }
            CHECK(inspector.Apply(world) == nvrhi::FS_OK);
            rig.Refresh();
            hashes.emplace_back(rig.Hash(rig.nodeRows), rig.Hash(rig.runtimeRows));
        }
        return hashes;
    }
}

// A script of inspector edits over nodes and runtime rows, with Simulate in between: the world's transforms and bounds
// per frame are the same at 1, 2 and 8 workers; the nodes' equal the node API's per frame; the last frame equals a
// graph built with the final transforms.
void test_edits_deterministic()
{
    ImGuiContextScope context;
    const auto one = RunScript(1, false);
    CHECK(RunScript(2, false) == one);
    CHECK(RunScript(8, false) == one);
    const auto api = RunScript(2, true);
    for (uint32_t frame = 0; frame < c_ScriptFrames; ++frame)
        CHECK(api[frame].first == one[frame].first && api[frame].second == one[frame].second);

    std::vector<LocalTransform> nodes = StartLocals(c_ScriptNodes, 0.0);
    std::vector<LocalTransform> runtime = StartLocals(c_ScriptRuntime, 1.0);
    for (uint32_t frame = 0; frame < c_ScriptFrames; ++frame)
        for (const ScriptEdit& edit : ScriptFrame(frame, c_ScriptNodes, c_ScriptRuntime))
            (edit.node ? nodes : runtime)[edit.index] = edit.value;
    SceneRig fresh(2, nodes, runtime);
    CHECK(fresh.Hash(fresh.nodeRows) == one.back().first && fresh.Hash(fresh.runtimeRows) == one.back().second);
}

// ------------------------------------------------------------------------------------------- pending edits

// A re-edit starts from the pending value and replaces it; components of one entity are separate entries; at most
// 64 distinct values are pending: the 65th is refused with a warning, a re-edit of a pending one is not; Apply writes
// all of them at once (one version step) and edits are accepted again after it.
void test_pending_edits()
{
    ImGuiContextScope context;
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    const ComponentId b = RegisterComponent<B>(world);
    const std::vector<Entity> rows = Create(world, { a, b }, 70);
    app::WorldInspector inspector;

    CHECK(Commit(inspector, world, rows[0], A{ 1.f }));
    OfferLog offers;
    DrawSelecting(inspector, world, nullptr, rows[0], offers);
    std::vector<uint8_t> shown = offers.Shown(a);
    CHECK(shown.size() == sizeof(A) && reinterpret_cast<const A*>(shown.data())->value == 1.f);   // pending
    CHECK(reinterpret_cast<const B*>(offers.Shown(b).data())->value == 0u);                         // stored
    CHECK(GetComponent<A>(world, rows[0])->value == 0.f);
    CHECK(Commit(inspector, world, rows[0], A{ 2.f }));
    CHECK(PendingShown(inspector, world) == 1);
    CHECK(Commit(inspector, world, rows[0], B{ 7u }));
    CHECK(PendingShown(inspector, world) == 2);
    for (uint32_t i = 1; i < 63; ++i)
        CHECK(Commit(inspector, world, rows[i], A{ float(i) * 10.f }));
    CHECK(PendingShown(inspector, world) == 64);

    const uint32_t warnings = g_Warnings.load();
    CHECK(Commit(inspector, world, rows[63], A{ 630.f }));   // offered and committed by the widget, refused
    TakeWarnings(warnings, 1);
    CHECK(PendingShown(inspector, world) == 64);
    CHECK(Commit(inspector, world, rows[0], A{ 3.f }));       // a pending one: accepted
    TakeWarnings(warnings, 0);
    OfferLog again;
    DrawSelecting(inspector, world, nullptr, rows[63], again);
    CHECK(reinterpret_cast<const A*>(again.Shown(a).data())->value == 0.f);   // the refused one is not pending

    const uint64_t version = world->GetVersion();
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(world->GetVersion() > version);
    CHECK(GetComponent<A>(world, rows[0])->value == 3.f && GetComponent<B>(world, rows[0])->value == 7u);
    for (uint32_t i = 1; i < 63; ++i)
        CHECK(GetComponent<A>(world, rows[i])->value == float(i) * 10.f);
    CHECK(GetComponent<A>(world, rows[63])->value == 0.f);
    CHECK(PendingShown(inspector, world) == 0);
    CHECK(Commit(inspector, world, rows[63], A{ 630.f }));
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(GetComponent<A>(world, rows[63])->value == 630.f);
}

// Re-edits of a pending value record no further command: Apply plays back one command per pending value.
// The regression test of backlog I3-R4 (every commit appended a SetComponent; fixed in F2).
void test_pending_commands_bounded()
{
    ImGuiContextScope context;
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    const std::vector<Entity> rows = Create(world, { a }, 2);
    app::WorldInspector inspector;
    for (uint32_t i = 0; i < 10; ++i)
        CHECK(Commit(inspector, world, rows[0], A{ float(i) }));
    CHECK(Commit(inspector, world, rows[1], A{ 5.f }));
    WorldStats before{};
    world->GetStats(before);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    WorldStats after{};
    world->GetStats(after);
    CHECK(GetComponent<A>(world, rows[0])->value == 9.f && GetComponent<A>(world, rows[1])->value == 5.f);
    CHECK(after.commandsPlayedBack - before.commandsPlayedBack == 2);
}

// The "edited" mark shows only for a value that is pending: not in the frame whose commit was refused (65th value).
// The regression test of backlog I3-R3 (the mark showed when RecordEdit dropped the edit; fixed in F2).
void test_edited_mark_only_when_pending()
{
    ImGuiContextScope context;
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    const std::vector<Entity> rows = Create(world, { a }, 65);
    app::WorldInspector inspector;
    for (uint32_t i = 0; i < 64; ++i)
        CHECK(Commit(inspector, world, rows[i], A{ 1.f }));
    const uint32_t warnings = g_Warnings.load();
    std::string text;
    CHECK(Commit(inspector, world, rows[64], A{ 1.f }, &text));
    TakeWarnings(warnings, 1);
    CHECK(text.find("edited; Apply writes it") == std::string::npos);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
}

// --------------------------------------------------------------------------------------------- refusals

namespace
{
    class FunctionObserver final : public nvrhi::ObjectImpl<IObserver>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(FunctionObserver)
        NVRHI_IMPLEMENTS_INTERFACE(IObserver)
        NVRHI_END_INTERFACE_TABLE()

        explicit FunctionObserver(std::function<void()> function) : m_Function(std::move(function)) { }
        nvrhi::FRESULT OnEntities(ObserverEvent, const EntityBatch&, ICommandBuffer*) noexcept override
        {
            m_Function();
            return nvrhi::FS_OK;
        }

    private:
        std::function<void()> m_Function;
    };
}

// Nothing is drawn or applied inside a phase (one warning each; the edits kept for the next Apply); Apply from an
// observer (Playback refuses) keeps the edits and does not advance the version; Draw of no world warns; Apply of no
// world drops pending edits; Draw of another world drops them before recording its own; an edit of an entity destroyed
// before Apply does not stop the others.
void test_refusals()
{
    ImGuiContextScope context;
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    const ComponentId b = RegisterComponent<B>(world);
    const std::vector<Entity> rows = Create(world, { a }, 3);
    app::WorldInspector inspector;

    // Inside a phase: Draw draws nothing, Apply keeps the edits.
    CHECK(Commit(inspector, world, rows[0], A{ 1.f }));
    nvrhi::FRESULT inPhase = nvrhi::FS_OK;
    const SystemId inspect = env.Add(MakeSystem("inspect", Phase::PreSimulation, SystemFlags::MainThread,
                                                [&](ISystemContext*)
    {
        inspector.Draw(world, env.scheduler.Get());
        inPhase = inspector.Apply(world);
    }));
    {
        uint32_t warnings = g_Warnings.load();
        g_Clipboard.clear();
        ImGui::GetPlatformIO().Platform_SetClipboardTextFn = &CaptureClipboard;
        ImGui::NewFrame();
        ImGui::LogToClipboard();
        CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
        ImGui::LogFinish();
        ImGui::Render();
        TakeWarnings(warnings, 2);   // Draw's and Apply's
        const ImGuiWindow* window = ImGui::FindWindowByName("World inspector");
        CHECK(window && !window->Active);   // drawn by the commit's frame, not by this one
        CHECK(g_Clipboard.find("pending edits") == std::string::npos);
        CHECK(inPhase == nvrhi::FE_OUT_OF_DATE);
    }
    CHECK(env.scheduler->RemoveSystem(inspect) == nvrhi::FS_OK);
    CHECK(GetComponent<A>(world, rows[0])->value == 0.f);
    CHECK(PendingShown(inspector, world) == 1);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(GetComponent<A>(world, rows[0])->value == 1.f);

    // From an observer: Playback refuses; kept, no version step.
    CHECK(Commit(inspector, world, rows[1], A{ 2.f }));
    nvrhi::FRESULT inObserver = nvrhi::FS_OK;
    uint64_t versionStep = ~0ull;
    auto observer = MAKE_RC_OBJ_PTR(FunctionObserver, std::function<void()>([&]()
    {
        const uint64_t version = world->GetVersion();
        inObserver = inspector.Apply(world);
        versionStep = world->GetVersion() - version;
    }));
    ObserverDesc desc{};
    desc.debugName = "applies";
    desc.component = b;
    desc.event = ObserverEvent::OnAdd;
    desc.access = Access::Read;
    ObserverId observerId = ObserverId::Invalid;
    CHECK(world->AddObserver(desc, observer.Get(), observerId) == nvrhi::FS_OK);
    uint32_t warnings = g_Warnings.load();
    CHECK(world->AddComponent(rows[2], b, nullptr) == nvrhi::FS_OK);
    TakeWarnings(warnings, 1);
    CHECK(inObserver == nvrhi::FE_OUT_OF_DATE && versionStep == 0);
    CHECK(world->RemoveObserver(observerId) == nvrhi::FS_OK);
    CHECK(GetComponent<A>(world, rows[1])->value == 0.f && PendingShown(inspector, world) == 1);
    WorldStats stats{};
    const uint64_t played = world->GetStats(stats).commandsPlayedBack;
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    CHECK(GetComponent<A>(world, rows[1])->value == 2.f);
    // one command for the one pending value: the refused playback left none behind (stage T8, mutant I3-06)
    CHECK(world->GetStats(stats).commandsPlayedBack == played + 1);

    // No world.
    warnings = g_Warnings.load();
    inspector.Draw(nullptr, nullptr);
    TakeWarnings(warnings, 1);
    CHECK(inspector.Apply(nullptr) == nvrhi::FS_OK);   // nothing pending
    CHECK(Commit(inspector, world, rows[0], A{ 9.f }));
    CHECK(inspector.Apply(nullptr) == nvrhi::FE_INVALID_ARGS);
    TakeWarnings(warnings, 1);
    CHECK(inspector.Apply(world) == nvrhi::FS_OK && GetComponent<A>(world, rows[0])->value == 1.f);   // dropped

    // Draw of another world drops the edits before it records its own.
    Env other;
    const ComponentId otherA = RegisterComponent<A>(other.world.Get());
    const Entity otherRow = Create(other.world.Get(), { otherA }, 1)[0];
    CHECK(Commit(inspector, world, rows[0], A{ 9.f }));
    CHECK(Commit(inspector, other.world.Get(), otherRow, A{ 8.f }));
    TakeWarnings(warnings, 1);
    CHECK(inspector.Apply(other.world.Get()) == nvrhi::FS_OK);
    CHECK(GetComponent<A>(other.world.Get(), otherRow)->value == 8.f && GetComponent<A>(world, rows[0])->value == 1.f);

    // A destroyed entity: the other edit is written; nothing stays pending.
    CHECK(Commit(inspector, world, rows[0], A{ 11.f }));
    CHECK(Commit(inspector, world, rows[1], A{ 12.f }));
    CHECK(world->DestroyEntities(&rows[1], 1) == nvrhi::FS_OK);
    const nvrhi::FRESULT result = inspector.Apply(world);
    std::printf("test_world_inspector: Apply with a destroyed entity's edit returns 0x%08x\n", unsigned(result));
    CHECK(GetComponent<A>(world, rows[0])->value == 11.f);
    CHECK(PendingShown(inspector, world) == 0 && inspector.Apply(world) == nvrhi::FS_OK);
}

// The inspector holds a world only through pending edits: from the first edit until Apply, until they are dropped
// (Draw of another world), or until the inspector is destroyed.
void test_world_reference()
{
    ImGuiContextScope context;
    Env env;
    IWorld* world = env.world.Get();
    const ComponentId a = RegisterComponent<A>(world);
    const Entity row = Create(world, { a }, 1)[0];
    const auto references = [world]()
    {
        world->AddRef();
        return uint64_t(world->Release());
    };
    const uint64_t base = references();
    auto inspector = std::make_unique<app::WorldInspector>();
    OfferLog offers;
    DrawSelecting(*inspector, world, env.scheduler.Get(), row, offers, true);
    CHECK(references() == base);
    CHECK(Commit(*inspector, world, row, A{ 1.f }));
    CHECK(references() == base + 1);
    CHECK(Commit(*inspector, world, row, A{ 2.f }));
    CHECK(references() == base + 1);
    CHECK(inspector->Apply(world) == nvrhi::FS_OK);
    CHECK(references() == base);
    CHECK(Commit(*inspector, world, row, A{ 3.f }));
    Env other;
    const uint32_t warnings = g_Warnings.load();
    DrawSelecting(*inspector, other.world.Get(), nullptr, Entity::Null, offers);
    TakeWarnings(warnings, 1);
    CHECK(references() == base);
    CHECK(Commit(*inspector, world, row, A{ 4.f }));
    CHECK(references() == base + 1);
    inspector.reset();
    CHECK(references() == base);
    CHECK(GetComponent<A>(world, row)->value == 2.f);
}

// Frames without edits allocate nothing of the inspector's (operator new, the world's blocks): every panel open over a
// graph's world and scheduler with an entity selected, after an Apply too; nor do Apply with nothing pending and a
// collapsed window. ImGui's own allocations are not counted (its allocator is malloc).
void test_allocations()
{
    ImGuiContextScope context;
    SceneRig rig(2, StartLocals(15, 0.0), StartLocals(8, 1.0));
    IWorld* world = rig.World();
    IScheduler* scheduler = rig.graph->GetScheduler();
    CHECK(rig.graph->Simulate(FrameTime{ 1.0 / 60.0, 1.0 / 60.0, 0 }) == nvrhi::FS_OK);
    rig.Refresh();
    app::WorldInspector inspector;
    inspector.SetEntityNames([](void*, Entity entity, char* out, uint32_t capacity)
    {
        return std::snprintf(out, capacity, "row %u", GetEntityIndex(entity)) > 0;
    }, nullptr);
    struct Counter
    {
        uint32_t offers = 0;
        static bool Edit(void* context, ComponentId, void*, uint32_t)
        {
            ++static_cast<Counter*>(context)->offers;
            return false;
        }
    } counter;
    app::detail::WorldInspectorTestHook hook;
    hook.openAll = true;
    hook.edit = &Counter::Edit;
    hook.context = &counter;
    const auto frames = [&](Entity selected, uint32_t count)
    {
        hook.select = selected;
        app::detail::SetWorldInspectorTestHook(&hook);
        for (uint32_t i = 0; i < count; ++i)
            DrawFrame(inspector, world, scheduler);
        app::detail::SetWorldInspectorTestHook(nullptr);
    };
    const auto measure = [&](const char* what, const std::function<void()>& work)
    {
        const uint64_t news = g_NewCount.load();
        const uint64_t blocks = engine::ecs::GetHeapAllocationCount();
        work();
        const uint64_t newsAfter = g_NewCount.load();
        const uint64_t blocksAfter = engine::ecs::GetHeapAllocationCount();
        std::printf("test_world_inspector: %s: %llu operator new, %llu world blocks\n", what,
                    (unsigned long long)(newsAfter - news), (unsigned long long)(blocksAfter - blocks));
        CHECK(newsAfter == news && blocksAfter == blocks);
    };

    frames(rig.nodeRows[0], 3);   // warm-up: windows, the scheduler's listing
    measure("10 frames, every panel open, a node selected", [&]() { frames(rig.nodeRows[0], 10); });
    CHECK(counter.offers > 0);
    measure("10 frames, a runtime row selected", [&]() { frames(rig.runtimeRows[3], 10); });
    measure("Apply with nothing pending", [&]() { CHECK(inspector.Apply(world) == nvrhi::FS_OK); });

    CHECK(Commit(inspector, world, rig.nodeRows[1], MakeLocal(dm::double3(3.0))));   // may allocate (deviation 5)
    CHECK(inspector.Apply(world) == nvrhi::FS_OK);
    rig.Refresh();
    frames(rig.nodeRows[1], 1);
    measure("10 frames after an edit and Apply", [&]() { frames(rig.nodeRows[1], 10); });

    ImGui::SetWindowCollapsed("World inspector", true, ImGuiCond_Always);
    measure("10 frames, the window collapsed", [&]() { frames(rig.nodeRows[1], 10); });
}

int main(int argc, char** argv)
{
    struct Case
    {
        const char* name;
        void (*run)();
        const char* backlog;   // an open backlog item this case fails on until it is fixed; run with --backlog
    };
    const Case cases[] = {
        { "chunk_info", test_chunk_info, nullptr },
        { "system_info", test_system_info, nullptr },
        { "refusals_inside_phase", test_refusals_inside_phase, nullptr },
        { "read_only_list", test_read_only_list, nullptr },
        { "draw_headless", test_draw_headless, nullptr },
        { "edit_and_apply", test_edit_and_apply, nullptr },
        // stage T3
        { "format_scalars", test_format_scalars, nullptr },
        { "format_shapes", test_format_shapes, nullptr },
        { "format_capacity", test_format_capacity, nullptr },
        { "chunk_info_storage", test_chunk_info_storage, nullptr },
        { "system_info_storage", test_system_info_storage, nullptr },
        { "system_info_cycle", test_system_info_cycle, nullptr },
        { "system_info_other_scheduler", test_system_info_other_scheduler, nullptr },
        { "panels_show_storage", test_panels_show_storage, nullptr },
        { "systems_panel_other_world", test_systems_panel_other_world, nullptr },
        { "offers_scene_rows", test_offers_scene_rows, nullptr },
        { "offers_simulation_inputs", test_offers_simulation_inputs, nullptr },
        { "offers_without_list", test_offers_without_list, nullptr },
        { "offers_by_flags", test_offers_by_flags, nullptr },
        { "edit_node_reaches_world", test_edit_node_reaches_world, nullptr },
        { "edit_runtime_reaches_world", test_edit_runtime_reaches_world, nullptr },
        { "edits_deterministic", test_edits_deterministic, nullptr },
        { "pending_edits", test_pending_edits, nullptr },
        { "pending_commands_bounded", test_pending_commands_bounded, nullptr },
        { "edited_mark_only_when_pending", test_edited_mark_only_when_pending, nullptr },
        { "refusals", test_refusals, nullptr },
        { "world_reference", test_world_reference, nullptr },
        { "allocations", test_allocations, nullptr },
    };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);   // a failed ImGui assert ends the run, no dialog
#endif
    // No argument: every case not marked with a backlog item. --backlog: the marked ones. A name: that case.
    const bool backlog = argc > 1 && std::strcmp(argv[1], "--backlog") == 0;
    const char* only = argc > 1 && !backlog ? argv[1] : nullptr;
    uint32_t failed = 0;
    uint32_t ran = 0;
    for (const Case& c : cases)
    {
        if (only ? std::strcmp(only, c.name) != 0 : backlog != (c.backlog != nullptr))
            continue;
        ++ran;
        g_Warnings = 0;
        try
        {
            c.run();
            CHECK(g_Warnings.load() == 0);
            if (c.backlog)
                std::printf("test_world_inspector: %s passed (marked backlog %s: fixed?)\n", c.name, c.backlog);
            else
                std::printf("test_world_inspector: %s passed\n", c.name);
        }
        catch (const std::runtime_error& err)
        {
            ++failed;
            app::detail::SetWorldInspectorTestHook(nullptr);
            std::fprintf(stderr, "%s\n", err.what());
            if (c.backlog)
                std::printf("test_world_inspector: %s FAILED (backlog %s)\n", c.name, c.backlog);
            else
                std::printf("test_world_inspector: %s FAILED\n", c.name);
        }
    }
    if (only && ran == 0)
    {
        std::fprintf(stderr, "test_world_inspector: no case %s\n", only);
        return 1;
    }
    return failed ? 1 : 0;
}
