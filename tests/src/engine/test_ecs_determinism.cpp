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

// Determinism of scheduled frames (aggregate docs/plans/2026-11-21-donut-ecs.md, step E2 exit; design 3.3.6, 11.1): a
// synthetic set of 21 systems plus two simulation domains' six, in every built-in phase, with reads and writes,
// parallel chunk loops, structural commands recorded on many workers (creation with Shared values, destruction, Sparse
// adds and removes, Shared moves, buffer appends) and carried across phases to the sync points, two exclusive systems
// with immediate structural changes (one in the middle of a phase), a domain whose steps change structure, main-thread
// systems, Changed and Added filters, a Chunk component, resources and fixed stepping. Every system's versions and the
// filter counts are folded into a resource, so the hash covers them. It runs for a number of frames with fixed
// FrameTime values at 1, 2 and 8 workers, three times each, and requires the same IWorld::ComputeHash and the same
// snapshot bytes (which also pin row order, E1.5) after every frame, Added changes seen exactly once, and the first
// run equal to the golden file.

#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsSchedulerImpl.h"
#include "EcsWorldImpl.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace det_test
{
    using donut::engine::ComponentFlags;

    struct Position { double x, y; };
    DONUT_ECS_COMPONENT(Position, "67799537-02a4-4bc0-b4ec-1114087b754a", DONUT_ECS_FIELDS(x, y));

    struct Velocity { double x, y; };
    DONUT_ECS_COMPONENT(Velocity, "e4d0dbcf-a648-4c29-ac39-a42b77c77b37", DONUT_ECS_FIELDS(x, y));

    struct Health { int32_t hp; };
    DONUT_ECS_COMPONENT(Health, "628664b5-b0e2-4f06-bc51-05e137cbc166", DONUT_ECS_FIELDS(hp));

    struct Age { uint32_t frames; };
    DONUT_ECS_COMPONENT(Age, "369d18cb-3c8c-4032-8c2b-a1ea521395fc", DONUT_ECS_FIELDS(frames));

    struct Seed { uint64_t state; };
    DONUT_ECS_COMPONENT(Seed, "69a4acdb-94a2-403d-9025-235a025544c4", DONUT_ECS_FIELDS(state));

    struct WorldPos { double x, y; };
    DONUT_ECS_COMPONENT(WorldPos, "38da1883-cafa-4a14-8be8-9ef07ed5ee2e", DONUT_ECS_FIELDS(x, y));

    struct Team { uint32_t id; };
    DONUT_ECS_COMPONENT_EX(Team, "1bfc6a14-3775-4245-821f-aeb35bdcacb9", ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(id));

    struct Selected { };
    DONUT_ECS_TAG_EX(Selected, "692d4b1c-071c-4c96-b8bd-6e8f6396e609", ComponentFlags::Sparse);

    struct Beacon { };
    DONUT_ECS_TAG(Beacon, "63d039f3-8000-4dfa-844c-94dd8d477df4");

    struct ChunkBox { double minX, minY, maxX, maxY; };
    DONUT_ECS_COMPONENT_EX(ChunkBox, "b6050749-306c-4768-8130-68018339de76", ComponentFlags::Chunk, 0,
                           DONUT_ECS_FIELDS(minX, minY, maxX, maxY));

    struct Hit { uint32_t frame; int32_t hp; };
    DONUT_ECS_BUFFER(Hit, "41568416-662a-43c2-8984-dfaac4ff5a88", 2, DONUT_ECS_FIELDS(frame, hp));

    struct Stats { uint64_t selected; uint64_t changedChunks; double energy; };
    DONUT_ECS_COMPONENT_EX(Stats, "76180429-45a5-47d4-8a15-73bacd0b888a", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(selected, changedChunks, energy));

    struct SimState { uint64_t steps; double simTime; uint64_t gathers; };
    DONUT_ECS_COMPONENT_EX(SimState, "cb38438d-2a06-4f62-8617-54e279536dbb", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(steps, simTime, gathers));

    // Made at once by an exclusive system in the middle of PreSimulation.
    struct Spark { uint32_t frame; };
    DONUT_ECS_COMPONENT(Spark, "4f1f7919-0c2c-434f-8dad-182a4dc30486", DONUT_ECS_FIELDS(frame));

    // Made by the forge domain's steps, played back at the sync point after each step.
    struct Ember { uint64_t step; };
    DONUT_ECS_COMPONENT(Ember, "03229de2-1edc-4d6f-9c56-b2f880a18c47", DONUT_ECS_FIELDS(step));

    // What the systems saw: their versions, last-run versions and Added/Changed counts, folded in after every frame so
    // that the world hash covers them.
    struct Trace { uint64_t mix; uint64_t observations; };
    DONUT_ECS_COMPONENT_EX(Trace, "ebd4d740-f95c-490c-850a-85bfa064bde8", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(mix, observations));
}

using namespace donut::engine;
using namespace det_test;
namespace ecs = donut::engine::ecs;

namespace
{
    constexpr uint32_t c_Frames = 60;
    constexpr uint32_t c_InitialEntities = 3000;
    constexpr uint32_t c_MaxAge = 40;

    uint64_t Xorshift(uint64_t& state)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    uint64_t Fnv1a(const uint8_t* data, size_t size)
    {
        uint64_t hash = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < size; ++i)
            hash = (hash ^ data[i]) * 0x100000001b3ull;
        return hash;
    }

    ComponentAccess R(ComponentId id) { return ComponentAccess{ id, Access::Read, {} }; }
    ComponentAccess W(ComponentId id) { return ComponentAccess{ id, Access::Write, {} }; }
    ComponentAccess Filter(ComponentId id) { return ComponentAccess{ id, Access::Filter, {} }; }

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

    // A CPU domain whose engine state is a step counter and a clock; it scatters them into a resource.
    class ClockDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(ClockDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        SimulationDomainDesc desc{};
        ComponentId simState = ComponentId::Invalid;
        uint64_t steps = 0;
        uint64_t gathers = 0;
        double simTime = 0.0;

        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override { return retVal = desc; }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch&, ISimulationContext*) noexcept override
        {
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch&, ISimulationContext*) noexcept override
        {
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext*) noexcept override
        {
            ++gathers;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT Step(double dt, ISimulationContext*) noexcept override
        {
            ++steps;
            simTime += dt;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept override
        {
            SimState* state = static_cast<SimState*>(context->GetWorld()->GetResource(simState));
            state->steps = steps;
            state->simTime = simTime;
            state->gathers = gathers;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT CaptureState(IStateWriter*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT RestoreState(IStateReader*) noexcept override { return nvrhi::FS_OK; }
    };

    // A CPU domain that changes structure: every third step records an Ember row, played back at the sync point
    // after the step (the step runs exclusive).
    class ForgeDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(ForgeDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        SimulationDomainDesc desc{};
        ComponentId ember = ComponentId::Invalid;
        uint64_t created = 0;

        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override { return retVal = desc; }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch&, ISimulationContext*) noexcept override
        {
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch&, ISimulationContext*) noexcept override
        {
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT Step(double, ISimulationContext* context) noexcept override
        {
            const uint64_t step = context->GetStepIndex();
            if (step % 3 != 0)
                return nvrhi::FS_OK;
            const ComponentId list[1] = { ember };
            const Ember value{ step };
            const void* values[1] = { &value };
            EntityCreateDesc create{};
            create.components = list;
            create.values = values;
            create.componentCount = 1;
            create.entityCount = 1;
            Entity placeholder;
            context->GetCommandBuffer()->CreateEntities(create, step, &placeholder);
            ++created;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT CaptureState(IStateWriter*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT RestoreState(IStateReader*) noexcept override { return nvrhi::FS_OK; }
    };

    struct Observation
    {
        uint64_t version;
        uint64_t lastRun;
    };

    uint64_t Mix(uint64_t hash, uint64_t value)
    {
        hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
        return hash * 0x100000001b3ull;
    }

    struct Result
    {
        std::vector<uint64_t> worldHashes;
        std::vector<uint64_t> snapshotHashes;
        std::vector<uint64_t> changedPerFrame;   // chunks that changed_counter's Changed<Health> passed
        uint32_t entities = 0;
        uint32_t ambiguities = 0;
        uint64_t validationErrors = 0;
        uint64_t playbacks = 0;
        uint64_t jobs = 0;
    };

    // One run of the scenario.
    Result Run(uint32_t workers)
    {
        WorldDesc worldDesc;
        worldDesc.debugName = "determinism";
        worldDesc.validation = true;
        nvrhi::AutoPtr<IWorld> world;
        CHECK(CreateWorld(worldDesc, &world) == nvrhi::FS_OK);
        JobSystemDesc jobDesc;
        jobDesc.debugName = "determinism";
        jobDesc.workerThreadCount = workers;
        nvrhi::AutoPtr<IJobSystem> jobs;
        CHECK(CreateJobSystem(jobDesc, &jobs) == nvrhi::FS_OK);
        nvrhi::AutoPtr<IScheduler> scheduler;
        CHECK(CreateScheduler(world.Get(), jobs.Get(), &scheduler) == nvrhi::FS_OK);
        IWorld* w = world.Get();

        const ComponentId position = RegisterComponent<Position>(w);
        const ComponentId velocity = RegisterComponent<Velocity>(w);
        const ComponentId health = RegisterComponent<Health>(w);
        const ComponentId age = RegisterComponent<Age>(w);
        const ComponentId seed = RegisterComponent<Seed>(w);
        const ComponentId worldPos = RegisterComponent<WorldPos>(w);
        const ComponentId team = RegisterComponent<Team>(w);
        const ComponentId selected = RegisterComponent<Selected>(w);
        const ComponentId beacon = RegisterComponent<Beacon>(w);
        const ComponentId chunkBox = RegisterComponent<ChunkBox>(w);
        const ComponentId hits = RegisterComponent<BufferOf<Hit>>(w);
        ComponentId stats = ComponentId::Invalid;
        ComponentId simState = ComponentId::Invalid;
        CHECK(w->RegisterResource(ComponentTraits<Stats>::Desc(), nullptr, stats) == nvrhi::FS_OK);
        CHECK(w->RegisterResource(ComponentTraits<SimState>::Desc(), nullptr, simState) == nvrhi::FS_OK);
        const ComponentId spark = RegisterComponent<Spark>(w);
        const ComponentId ember = RegisterComponent<Ember>(w);
        ComponentId trace = ComponentId::Invalid;
        CHECK(w->RegisterResource(ComponentTraits<Trace>::Desc(), nullptr, trace) == nvrhi::FS_OK);
        const ComponentId frameTime = FindComponent<FrameTime>(w);
        CHECK(frameTime != ComponentId::Invalid);

        // The archetype of the actors, and the values of a batch of them.
        const ComponentId actor[9] = { position, velocity, health, age, seed, worldPos, team, chunkBox, hits };
        struct Batch
        {
            std::vector<Position> positions;
            std::vector<Velocity> velocities;
            std::vector<Health> healths;
            std::vector<Age> ages;
            std::vector<Seed> seeds;
            std::vector<WorldPos> worldPositions;
            std::vector<Team> teams;
            const void* values[9] = {};

            void Fill(uint64_t& random, uint32_t count, uint32_t teamId)
            {
                positions.resize(count);
                velocities.resize(count);
                healths.resize(count);
                ages.resize(count);
                seeds.resize(count);
                worldPositions.resize(count);
                teams.assign(count, Team{ teamId });
                for (uint32_t i = 0; i < count; ++i)
                {
                    positions[i] = Position{ double(Xorshift(random) % 1000) * 0.125, double(Xorshift(random) % 200) };
                    velocities[i] = Velocity{ double(int64_t(Xorshift(random) % 64) - 32) * 0.5,
                                              double(Xorshift(random) % 32) };
                    healths[i] = Health{ int32_t(20 + Xorshift(random) % 60) };
                    ages[i] = Age{ uint32_t(Xorshift(random) % 30) };
                    seeds[i] = Seed{ Xorshift(random) | 1 };
                    worldPositions[i] = WorldPos{ 0.0, 0.0 };
                }
                const void* list[9] = { positions.data(), velocities.data(), healths.data(), ages.data(), seeds.data(),
                                        worldPositions.data(), teams.data(), nullptr, nullptr };
                std::memcpy(values, list, sizeof(values));
            }
        };
        uint64_t setupRandom = 0x2545f4914f6cdd1dull;
        for (uint32_t t = 0; t < 3; ++t)
        {
            Batch batch;
            batch.Fill(setupRandom, c_InitialEntities / 3, t);
            EntityCreateDesc desc{};
            desc.components = actor;
            desc.values = batch.values;
            desc.componentCount = 9;
            desc.entityCount = c_InitialEntities / 3;
            std::vector<Entity> created(desc.entityCount);
            CHECK(w->CreateEntities(desc, created.data()) == nvrhi::FS_OK);
        }

        // Queries, made before any phase runs.
        Query<Write<Velocity>> qVelocity(w, "velocity");
        Query<Write<Seed>, Write<Velocity>> qJitter(w, "jitter");
        Query<Write<Position>, Read<Velocity>> qIntegrate(w, "integrate");
        Query<Read<Position>, Read<Age>> qDamage(w, "damage");
        Query<Read<Health>, Read<Age>> qHealthAge(w, "health age");
        Query<Read<Age>, Read<Health>, ReadShared<Team>> qRetag(w, "retag");
        Query<Read<Position>, Write<WorldPos>> qWorldPos(w, "world pos");
        Query<Read<WorldPos>, WriteChunk<ChunkBox>> qBounds(w, "bounds");
        Query<Read<WorldPos>, Read<Health>> qExtract(w, "extract");
        Query<Read<Health>, With<Selected>> qSelected(w, "selected");
        Query<Changed<Health>> qChanged(w, "changed health");
        Query<Write<Age>> qAger(w, "ager");
        Query<Read<Position>, With<Beacon>> qBeacons(w, "beacons");
        Query<Added<Spark>> qAddedSpark(w, "added spark");
        Query<Added<Ember>> qAddedEmber(w, "added ember");
        uint32_t frame = 0;

        // Every system notes its version and last-run version; the frame loop folds them into the Trace resource.
        Observation observed[64] = {};
        uint32_t systemCount = 0;
        auto add = [&](const char* name, Phase phase, std::vector<ComponentAccess> accesses,
                       std::vector<const char*> after, SystemFlags flags, auto&& update,
                       std::vector<const char*> before = {})
        {
            CHECK(systemCount < 64);
            const uint32_t slot = systemCount++;
            auto observing = [&observed, slot, update](ISystemContext* context) mutable -> nvrhi::FRESULT
            {
                observed[slot] = Observation{ context->GetVersion(), context->GetLastRunVersion() };
                if constexpr (std::is_same_v<decltype(update(context)), nvrhi::FRESULT>)
                    return update(context);
                else
                {
                    update(context);
                    return nvrhi::FS_OK;
                }
            };
            SystemDesc desc{};
            desc.name = name;
            desc.accesses = accesses.data();
            desc.accessCount = uint32_t(accesses.size());
            desc.phase = phase;
            desc.runAfter = after.data();
            desc.runAfterCount = uint32_t(after.size());
            desc.runBefore = before.data();
            desc.runBeforeCount = uint32_t(before.size());
            desc.flags = flags;
            nvrhi::AutoPtr<ISystem> system;
            CHECK(CreateLambdaSystem(desc, observing, &system) == nvrhi::FS_OK);
            SystemId id;
            CHECK(scheduler->AddSystem(system.Get(), id) == nvrhi::FS_OK);
        };
        const SystemFlags none = SystemFlags::None;

        // BeginFrame: a main-thread spawner records creations with Shared values; they wait for PostSimulation's sync.
        add("spawner", Phase::BeginFrame, { R(frameTime) }, {}, SystemFlags::MainThread, [&](ISystemContext* context)
        {
            uint64_t random = 0x9e3779b97f4a7c15ull ^ (uint64_t(frame) * 0xbf58476d1ce4e5b9ull);
            for (uint32_t t = 0; t < 3; ++t)
            {
                const uint32_t count = 24 + (frame + t) % 7;
                Batch batch;
                batch.Fill(random, count, (frame + t) % 4);
                EntityCreateDesc desc{};
                desc.components = actor;
                desc.values = batch.values;
                desc.componentCount = 9;
                desc.entityCount = count;
                Entity placeholders[32];
                context->GetCommandBuffer()->CreateEntities(desc, t, placeholders);
            }
        });

        // PreSimulation: velocity chain over parallel chunks.
        add("gravity", Phase::PreSimulation, { W(velocity), R(frameTime) }, {}, none, [&](ISystemContext* context)
        {
            const double dt = context->GetFrameTime().deltaTime;
            qVelocity.ParallelForChunks(context, 1, [dt](const ChunkView& view, Velocity* v)
            {
                for (uint32_t row = 0; row < view.count; ++row)
                    v[row].y -= 9.81 * dt;
            });
        });
        // An exclusive system in the middle of the phase makes Spark rows at once on every third frame; a watcher
        // before it sees them a frame later, one after it (not the phase's last) in the same frame, each once.
        uint32_t sparkBefore[c_Frames] = {};
        uint32_t sparkAfter[c_Frames] = {};
        uint32_t emberAfter[c_Frames] = {};
        auto countChunks = [](const auto& query, ISystemContext* context)
        {
            uint32_t chunks = 0;
            query.ForEachChunk(context, [&chunks](const ChunkView&, const auto*) { ++chunks; });
            return chunks;
        };
        add("spark_before", Phase::PreSimulation, { R(spark) }, {}, none,
            [&](ISystemContext* context) { sparkBefore[frame] = countChunks(qAddedSpark, context); });
        add("spark_maker", Phase::PreSimulation, {}, { "gravity", "spark_before", "clock.gather", "forge.gather" },
            SystemFlags::Exclusive, [&](ISystemContext* context)
        {
            if (frame % 3 != 0)
                return nvrhi::FS_OK;
            const ComponentId list[1] = { spark };
            const Spark values[2] = { { frame }, { frame + 1000 } };
            const void* valueList[1] = { values };
            EntityCreateDesc desc{};
            desc.components = list;
            desc.values = valueList;
            desc.componentCount = 1;
            desc.entityCount = 2;
            Entity made[2];
            return context->GetWorld()->CreateEntities(desc, made);
        });
        add("spark_after", Phase::PreSimulation, { R(spark) }, { "spark_maker" }, none,
            [&](ISystemContext* context) { sparkAfter[frame] = countChunks(qAddedSpark, context); }, { "drag" });
        add("drag", Phase::PreSimulation, { W(velocity) }, { "spark_maker" }, none, [&](ISystemContext* context)
        {
            qVelocity.ParallelForEach(context, 2, [](Entity, Velocity& v) { v.x *= 0.99; v.y *= 0.99; });
        });
        add("jitter", Phase::PreSimulation, { W(seed), W(velocity) }, { "drag" }, none, [&](ISystemContext* context)
        {
            qJitter.ParallelForEach(context, 1, [](Entity, Seed& s, Velocity& v)
            {
                if (Xorshift(s.state) % 16 == 0)
                    v.x = -v.x;
            });
        });

        // FixedSimulation: the clock domain's step, a gate that orders it before the forge's exclusive step (domain
        // systems cannot name each other), a watcher of the forge's Ember rows, then a plain system.
        add("forge_gate", Phase::FixedSimulation, {}, { "clock.step" }, none, [](ISystemContext*) { },
            { "forge.step" });
        add("ember_watch", Phase::FixedSimulation, { R(ember) }, { "forge.step" }, none,
            [&](ISystemContext* context) { emberAfter[frame] = countChunks(qAddedEmber, context); });
        add("integrate", Phase::FixedSimulation, { W(position), R(velocity), R(frameTime) }, { "ember_watch" }, none,
            [&](ISystemContext* context)
        {
            const double dt = context->GetFrameTime().deltaTime;
            qIntegrate.ParallelForChunks(context, 1, [dt](const ChunkView& view, Position* p, const Velocity* v)
            {
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    p[row].x += v[row].x * dt;
                    p[row].y += v[row].y * dt;
                }
            });
        });

        // PostSimulation: damage, then commands from parallel chunk loops (destroy, Sparse add/remove, Shared moves),
        // then an exclusive system that changes structure at once.
        // Damage writes Health per entity (GetComponentForWrite with the system's version) in a third of the chunks
        // each frame, so the Changed<Health> filter of changed_counter passes only some chunks.
        add("damage", Phase::PostSimulation, { R(position), R(age), W(health) }, {}, none, [&](ISystemContext* context)
        {
            qDamage.ParallelForChunks(context, 1, [&](const ChunkView& view, const Position* p, const Age* a)
            {
                if ((view.chunkIndex + frame) % 3 != 0)
                    return;
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    if (p[row].y >= 0.0 && (a[row].frames + frame) % 2 != 0)
                        continue;
                    void* value = context->GetWorld()->GetComponentForWrite(view.entities[row], health,
                                                                            context->GetVersion());
                    Health* h = static_cast<Health*>(value);
                    h->hp -= p[row].y < 0.0 ? 9 : 4;
                }
            });
        });
        // Entities older than c_MaxAge are being destroyed by expire (EndFrame): commands leave them alone.
        add("reaper", Phase::PostSimulation, { R(health), R(age) }, { "damage" }, none, [&](ISystemContext* context)
        {
            qHealthAge.ParallelForChunks(context, 1, [&](const ChunkView& view, const Health* h, const Age* a)
            {
                std::vector<Entity> dead;
                for (uint32_t row = 0; row < view.count; ++row)
                    if (h[row].hp <= 0 && a[row].frames <= c_MaxAge)
                        dead.push_back(view.entities[row]);
                if (!dead.empty())
                    context->GetCommandBuffer()->DestroyEntities(dead.data(), uint32_t(dead.size()),
                                                                 MakeSortKey(0, view.chunkIndex, 0));
            });
        });
        add("selector", Phase::PostSimulation, { R(health), R(age) }, { "damage" }, none, [&](ISystemContext* context)
        {
            qHealthAge.ParallelForChunks(context, 1, [&](const ChunkView& view, const Health* h, const Age* a)
            {
                std::vector<Entity> select;
                std::vector<Entity> unselect;
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    if (h[row].hp <= 0 || a[row].frames > c_MaxAge)
                        continue;
                    if (h[row].hp % 4 == 0)
                        select.push_back(view.entities[row]);
                    else if (h[row].hp % 4 == 1)
                        unselect.push_back(view.entities[row]);
                }
                ICommandBuffer* commands = context->GetCommandBuffer();
                if (!select.empty())
                    commands->AddComponents(select.data(), uint32_t(select.size()), selected, nullptr,
                                            MakeSortKey(1, view.chunkIndex, 0));
                if (!unselect.empty())
                    commands->RemoveComponents(unselect.data(), uint32_t(unselect.size()), selected,
                                               MakeSortKey(2, view.chunkIndex, 0));
            });
        });
        add("retagger", Phase::PostSimulation, { R(age), R(health), R(team) }, { "damage" }, none,
            [&](ISystemContext* context)
        {
            qRetag.ParallelForChunks(context, 1,
                                     [&](const ChunkView& view, const Age* a, const Health* h, const Team* t)
            {
                std::vector<Entity> moved;
                for (uint32_t row = 0; row < view.count; ++row)
                    if (a[row].frames % 7 == 3 && h[row].hp > 0 && a[row].frames <= c_MaxAge)
                        moved.push_back(view.entities[row]);
                if (moved.empty())
                    return;
                const std::vector<Team> values(moved.size(), Team{ (t->id + 1) % 4 });
                context->GetCommandBuffer()->AddComponents(moved.data(), uint32_t(moved.size()), team, values.data(),
                                                           MakeSortKey(0, view.chunkIndex, 0));
            });
        });
        add("beacons", Phase::PostSimulation, {},
            { "damage", "reaper", "selector", "retagger", "clock.scatter", "forge.scatter" }, SystemFlags::Exclusive,
            [&](ISystemContext* context)
        {
            IWorld* target = context->GetWorld();
            if (frame % 3 == 0)
            {
                const ComponentId list[2] = { position, beacon };
                const Position values[2] = { { double(frame), 1.0 }, { double(frame), 2.0 } };
                const void* valueList[2] = { values, nullptr };
                EntityCreateDesc desc{};
                desc.components = list;
                desc.values = valueList;
                desc.componentCount = 2;
                desc.entityCount = 2;
                Entity made[2];
                if (target->CreateEntities(desc, made) != nvrhi::FS_OK)
                    return nvrhi::FE_GENERIC_ERROR;
            }
            if (frame % 5 == 0)
            {
                Entity first = Entity::Null;
                qBeacons.ForEach(ChunkRequest{}, [&first](Entity entity, const Position&)
                {
                    if (first == Entity::Null)
                        first = entity;
                });
                if (first != Entity::Null && target->DestroyEntities(&first, 1) != nvrhi::FS_OK)
                    return nvrhi::FE_GENERIC_ERROR;
            }
            return nvrhi::FS_OK;
        });

        // TransformPropagation and Bounds: a column copy, and a Chunk component per chunk.
        add("world_pos", Phase::TransformPropagation, { R(position), W(worldPos) }, {}, none,
            [&](ISystemContext* context)
        {
            qWorldPos.ParallelForChunks(context, 1, [](const ChunkView& view, const Position* p, WorldPos* out)
            {
                for (uint32_t row = 0; row < view.count; ++row)
                    out[row] = WorldPos{ p[row].x, p[row].y };
            });
        });
        add("chunk_bounds", Phase::Bounds, { R(worldPos), W(chunkBox) }, {}, none, [&](ISystemContext* context)
        {
            qBounds.ParallelForChunks(context, 1, [](const ChunkView& view, const WorldPos* p, ChunkBox* box)
            {
                ChunkBox result{ 1e300, 1e300, -1e300, -1e300 };
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    result.minX = p[row].x < result.minX ? p[row].x : result.minX;
                    result.minY = p[row].y < result.minY ? p[row].y : result.minY;
                    result.maxX = p[row].x > result.maxX ? p[row].x : result.maxX;
                    result.maxY = p[row].y > result.maxY ? p[row].y : result.maxY;
                }
                *box = result;
            });
        });

        // RenderExtract: serial reductions on the main thread, a Changed filter after it.
        add("extract", Phase::RenderExtract, { R(worldPos), R(health), Filter(selected), W(stats) }, {},
            SystemFlags::MainThread, [&](ISystemContext* context)
        {
            Stats* s = static_cast<Stats*>(context->GetWorld()->GetResource(stats));
            double energy = 0.0;
            qExtract.ForEach(context, [&energy](Entity, const WorldPos& p, const Health& h)
            {
                energy += double(h.hp) * (p.x + p.y);
            });
            uint64_t count = 0;
            qSelected.ForEach(context, [&count](Entity, const Health&) { ++count; });
            s->energy = energy;
            s->selected = count;
        });
        add("changed_counter", Phase::RenderExtract, { R(health), W(stats) }, { "extract" }, none,
            [&](ISystemContext* context)
        {
            uint64_t count = 0;
            qChanged.ForEachChunk(context, [&count](const ChunkView&, const Health*) { ++count; });
            static_cast<Stats*>(context->GetWorld()->GetResource(stats))->changedChunks += count;
        });

        // EndFrame: age, then expiry and buffer appends, played back at the next frame's PostSimulation sync point.
        add("ager", Phase::EndFrame, { W(age) }, {}, none, [&](ISystemContext* context)
        {
            qAger.ParallelForEach(context, 2, [](Entity, Age& a) { ++a.frames; });
        });
        add("expire", Phase::EndFrame, { R(age), R(health) }, { "ager" }, none, [&](ISystemContext* context)
        {
            qHealthAge.ParallelForChunks(context, 1, [&](const ChunkView& view, const Health* h, const Age* a)
            {
                std::vector<Entity> old;
                ICommandBuffer* commands = context->GetCommandBuffer();
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    if (a[row].frames > c_MaxAge)
                        old.push_back(view.entities[row]);
                    else if (a[row].frames == 20)
                    {
                        const Hit hit{ frame, h[row].hp };
                        commands->AppendToBuffer(view.entities[row], hits, &hit, 1,
                                                 MakeSortKey(1, view.chunkIndex, row));
                    }
                }
                if (!old.empty())
                    commands->DestroyEntities(old.data(), uint32_t(old.size()), MakeSortKey(0, view.chunkIndex, 0));
            });
        });

        // A domain with a fixed step that does not divide the frame time.
        nvrhi::AutoPtr<ClockDomain> clock = MAKE_RC_OBJ_PTR(ClockDomain);
        clock->simState = simState;
        clock->desc.name = "clock";
        clock->desc.fixedTimeStep = 1.0 / 90;
        clock->desc.maxStepsPerFrame = 3;
        clock->desc.substeps = 1;
        clock->desc.flags = SimulationDomainFlags::Deterministic;
        const ComponentAccess outputs[1] = { W(simState) };
        clock->desc.outputs = outputs;
        clock->desc.outputCount = 1;
        SimulationDomainId clockId;
        CHECK(scheduler->AddSimulationDomain(clock.Get(), clockId) == nvrhi::FS_OK);

        // A domain that changes structure: a sync point inside FixedSimulation after each step.
        nvrhi::AutoPtr<ForgeDomain> forge = MAKE_RC_OBJ_PTR(ForgeDomain);
        forge->ember = ember;
        forge->desc.name = "forge";
        forge->desc.fixedTimeStep = 1.0 / 45;
        forge->desc.maxStepsPerFrame = 2;
        forge->desc.substeps = 1;
        forge->desc.flags = SimulationDomainFlags::ChangesStructure | SimulationDomainFlags::Deterministic;
        SimulationDomainId forgeId;
        CHECK(scheduler->AddSimulationDomain(forge.Get(), forgeId) == nvrhi::FS_OK);

        CHECK(scheduler->Build() == nvrhi::FS_OK);
        Result result;
        result.ambiguities = scheduler->GetAmbiguityCount();

        double time = 0.0;
        uint64_t previousChanged = 0;
        for (frame = 0; frame < c_Frames; ++frame)
        {
            // Fixed frame times, with a hitch every 13th frame.
            const double delta = frame % 13 == 12 ? 0.05 : 1.0 / 60;
            time += delta;
            scheduler->SetFrameTime(FrameTime{ time, delta, frame });
            const uint64_t embersBefore = forge->created;
            CHECK(scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);

            // Each Added change is seen exactly once (E2.2 review fix 1): by the watcher after the exclusive system or
            // the step in the same frame, by the watcher before it at its next run.
            CHECK(sparkAfter[frame] == (frame % 3 == 0 ? 1u : 0u));
            CHECK(sparkBefore[frame] == (frame % 3 == 1 ? 1u : 0u));
            CHECK(emberAfter[frame] == (forge->created != embersBefore ? 1u : 0u));

            // What the systems saw enters the world hash through the Trace resource.
            Trace* traced = static_cast<Trace*>(w->GetResource(trace));
            for (uint32_t s = 0; s < systemCount; ++s)
                traced->mix = Mix(Mix(traced->mix, observed[s].version), observed[s].lastRun);
            traced->mix = Mix(Mix(Mix(traced->mix, sparkBefore[frame]), sparkAfter[frame]), emberAfter[frame]);
            const uint64_t changedChunks = GetResource<Stats>(w)->changedChunks;
            traced->mix = Mix(traced->mix, changedChunks);
            ++traced->observations;
            result.changedPerFrame.push_back(changedChunks - previousChanged);
            previousChanged = changedChunks;

            result.worldHashes.push_back(w->ComputeHash());
            nvrhi::AutoPtr<MemoryWriter> writer = MAKE_RC_OBJ_PTR(MemoryWriter);
            CHECK(w->SaveSnapshot(writer.Get()) == nvrhi::FS_OK);
            result.snapshotHashes.push_back(Fnv1a(writer->bytes.data(), writer->bytes.size()));
        }
        WorldStats stats2{};
        w->GetStats(stats2);
        result.entities = stats2.entityCount;
        result.validationErrors = ecs::GetWorldImpl(w)->GetValidationErrorCount();
        ecs::SchedulerCounters counters{};
        CHECK(ecs::GetSchedulerCounters(scheduler.Get(), counters));
        result.playbacks = counters.playbacks;
        result.jobs = counters.jobsScheduled;
        CHECK(clock->steps > c_Frames && GetResource<SimState>(w)->steps == clock->steps);
        nvrhi::string diagnostics;
        scheduler->GetDiagnostics(diagnostics);
        if (std::string(diagnostics.c_str()).find("failed") != std::string::npos)
        {
            fprintf(stderr, "%s", diagnostics.c_str());
            CHECK(false);
        }
        return result;
    }
}

// The per-frame hashes of the first run against tests/src/engine/golden/ecs_scheduler_frames.txt: runs that agree with
// each other can still all be wrong (a Changed filter that passes everything, versions in another order), and the
// Trace resource puts what the systems saw into the hash. ECS_UPDATE_GOLDEN=1 rewrites the file.
static void CheckGolden(const Result& result)
{
    std::string actual = "# ECS scheduler determinism (test_ecs_determinism.cpp; ECS_UPDATE_GOLDEN=1 rewrites it)\n"
                         "# frame world-hash snapshot-fnv1a changed-health-chunks\n";
    for (uint32_t f = 0; f < c_Frames; ++f)
    {
        char line[96];
        std::snprintf(line, sizeof(line), "%u %016llx %016llx %llu\n", f, (unsigned long long)result.worldHashes[f],
                      (unsigned long long)result.snapshotHashes[f], (unsigned long long)result.changedPerFrame[f]);
        actual += line;
    }
    const std::filesystem::path path = std::filesystem::path(DONUT_TEST_GOLDEN_DIR) / "ecs_scheduler_frames.txt";
    const char* update = std::getenv("ECS_UPDATE_GOLDEN");
    if (update && std::strcmp(update, "1") == 0)
    {
        std::ofstream file(path, std::ios::binary);
        file << actual;
        CHECK(file.good());
        printf("test_ecs_determinism: wrote %s\n", path.generic_string().c_str());
        return;
    }
    std::ifstream file(path, std::ios::binary);
    std::stringstream expected;
    expected << file.rdbuf();
    if (expected.str() != actual)
    {
        printf("test_ecs_determinism: the frames differ from %s (ECS_UPDATE_GOLDEN=1 rewrites it)\nexpected:\n%s"
               "actual:\n%s", path.generic_string().c_str(), expected.str().c_str(), actual.c_str());
        CHECK(false);
    }
}

static void test_determinism()
{
    Result reference;
    uint32_t runs = 0;
    for (uint32_t repeat = 0; repeat < 3; ++repeat)
    {
        for (uint32_t workers : { 1u, 2u, 8u })
        {
            const Result result = Run(workers);
            CHECK(result.ambiguities == 0);
            CHECK(result.validationErrors == 0);
            CHECK(result.worldHashes.size() == c_Frames);
            if (runs == 0)
            {
                reference = result;
                // The world changes every frame, and the Changed filter passes a varying number of chunks.
                for (uint32_t f = 1; f < c_Frames; ++f)
                    CHECK(result.worldHashes[f] != result.worldHashes[f - 1]);
                uint64_t fewest = ~0ull;
                uint64_t most = 0;
                for (uint32_t f = 1; f < c_Frames; ++f)
                {
                    fewest = std::min(fewest, result.changedPerFrame[f]);
                    most = std::max(most, result.changedPerFrame[f]);
                }
                CHECK(fewest < most);
                CheckGolden(result);
            }
            for (uint32_t f = 0; f < c_Frames; ++f)
            {
                if (result.worldHashes[f] != reference.worldHashes[f] ||
                    result.snapshotHashes[f] != reference.snapshotHashes[f])
                {
                    fprintf(stderr, "frame %u differs at %u workers (repeat %u): hash %016llx, expected %016llx\n", f,
                            workers, repeat, (unsigned long long)result.worldHashes[f],
                            (unsigned long long)reference.worldHashes[f]);
                    CHECK(false);
                }
            }
            CHECK(result.entities == reference.entities && result.playbacks == reference.playbacks &&
                  result.jobs == reference.jobs);
            printf("test_ecs_determinism: %u workers, repeat %u: %u frames, final world hash %016llx, snapshot "
                   "%016llx, %u entities, %llu playbacks, %llu system jobs\n", workers, repeat, c_Frames,
                   (unsigned long long)result.worldHashes.back(), (unsigned long long)result.snapshotHashes.back(),
                   result.entities, (unsigned long long)result.playbacks, (unsigned long long)result.jobs);
            ++runs;
        }
    }
}

int main(int, char**)
{
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    try
    {
        test_determinism();
        printf("test_ecs_determinism: determinism passed\n");
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
