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
// with another world refusing. The functional list of E7.7 is stage T's (T3). Widgets are stood in for by the
// module-private test seam (src/app/WorldInspectorDetail.h).

#include <donut/app/WorldInspector.h>
#include <donut/core/log.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include "WorldInspectorDetail.h"

#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#include <cstdlib>
#endif

namespace inspector_test
{
    struct A { float value; };
    DONUT_ECS_COMPONENT(A, "5d1f0f4c-58a4-4a3b-9b8e-31b0d7a5e101", DONUT_ECS_FIELDS(value));

    struct B { uint32_t value; };
    DONUT_ECS_COMPONENT(B, "0c3b7f0e-2b7e-4d55-8f2a-6a1d3e9c4b02", DONUT_ECS_FIELDS(value));
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

// Donut's read-only list in a scene graph's world: the 16 ids, registered after the store's types; none elsewhere.
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
    };
    for (ComponentId id : expected)
        CHECK(id != ComponentId::Invalid);
    ComponentId ids[c_MaxInspectorReadOnlyComponents] = {};
    CHECK(app::GetDonutReadOnlyComponents(world, nullptr, 0) == 16);
    CHECK(app::GetDonutReadOnlyComponents(world, ids, c_MaxInspectorReadOnlyComponents) == 16);
    CHECK(std::equal(std::begin(expected), std::end(expected), ids));
    CHECK(std::find(ids, ids + 16, FindComponent<LocalTransform>(world)) == ids + 16);

    ComponentId two[3] = { ComponentId::Invalid, ComponentId::Invalid, ComponentId::Invalid };
    CHECK(app::GetDonutReadOnlyComponents(world, two, 2) == 16);
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

int main(int argc, char** argv)
{
    struct Case
    {
        const char* name;
        void (*run)();
    };
    const Case cases[] = {
        { "chunk_info", test_chunk_info },
        { "system_info", test_system_info },
        { "refusals_inside_phase", test_refusals_inside_phase },
        { "read_only_list", test_read_only_list },
        { "draw_headless", test_draw_headless },
        { "edit_and_apply", test_edit_and_apply },
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
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
                continue;
            c.run();
            CHECK(g_Warnings.load() == 0);
            std::printf("test_world_inspector: %s passed\n", c.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
