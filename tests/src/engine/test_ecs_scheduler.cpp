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

// The ECS scheduler (aggregate docs/plans/2026-11-21-donut-ecs.md, step E2.2; design 6.2-6.5, 8.4, 11.3): ordering from
// declared access and runBefore/runAfter against a brute-force model over seeded random system sets at 1, 2 and 8
// workers, cycle and ambiguity reports, exclusive and main-thread systems, per-system versions and Changed/Added
// filters across frames, command-buffer playback order at sync points, fixed stepping of simulation domains, the Debug
// access checks, the fallback when the job table is full, the API's argument checks, and an allocation counter that
// proves a fixed system set allocates nothing per frame. Internals are read through the private headers
// (src/engine/EcsSchedulerImpl.h, EcsWorldImpl.h).

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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test, and donut_engine, which links statically) is counted. The storage's
// own blocks, the command buffers' arenas included, come from nvrhi_core's allocator, which ecs::GetHeapAllocationCount
// counts.
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

// ------------------------------------------------------------------------------------------ test components

namespace sched_test
{
    using donut::engine::ComponentFlags;

    struct A { float value; };
    DONUT_ECS_COMPONENT(A, "2c74d3e3-d2ef-4e75-b316-70de35dfa379", DONUT_ECS_FIELDS(value));

    struct B { float value; };
    DONUT_ECS_COMPONENT(B, "026f8cdc-1c9c-4225-9514-91714676ac47", DONUT_ECS_FIELDS(value));

    struct C { uint32_t value; };
    DONUT_ECS_COMPONENT(C, "71a2ee6b-9bb0-4ac5-b292-4f0df2a897d4", DONUT_ECS_FIELDS(value));

    struct D { uint32_t value; };
    DONUT_ECS_COMPONENT(D, "72ab5085-b60f-4404-a08a-b699228c5492", DONUT_ECS_FIELDS(value));

    struct E { uint32_t value; };
    DONUT_ECS_COMPONENT(E, "eceee7a2-4a83-493f-b843-d9cb78b7119d", DONUT_ECS_FIELDS(value));

    struct F { uint32_t value; };
    DONUT_ECS_COMPONENT(F, "09ad899a-a35b-4287-bb2e-50120a291b0a", DONUT_ECS_FIELDS(value));

    struct Marker { };
    DONUT_ECS_TAG(Marker, "ed0091b9-69e7-471e-bf32-4420460d7a47");

    struct Elem { uint32_t value; };
    DONUT_ECS_BUFFER(Elem, "77515fe8-ba47-4978-98f5-0256c7994f73", 2, DONUT_ECS_FIELDS(value));

    struct Counter { uint64_t value; };
    DONUT_ECS_COMPONENT_EX(Counter, "69b2492e-f7e1-47b6-a552-3cb53a4c339a", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(value));
}

using namespace donut::engine;
using namespace sched_test;
namespace ecs = donut::engine::ecs;

namespace
{
    // --------------------------------------------------------------------------------------------- helpers

    ComponentAccess R(ComponentId id) { return ComponentAccess{ id, Access::Read, {} }; }
    ComponentAccess W(ComponentId id) { return ComponentAccess{ id, Access::Write, {} }; }
    ComponentAccess Filter(ComponentId id) { return ComponentAccess{ id, Access::Filter, {} }; }

    struct Spec
    {
        std::string name;
        Phase phase = Phase::PreSimulation;
        std::vector<ComponentAccess> accesses;
        std::vector<std::string> before;
        std::vector<std::string> after;
        SystemFlags flags = SystemFlags::None;
    };

    template <typename Function>
    SystemId Add(IScheduler* scheduler, const Spec& spec, Function&& update, nvrhi::FRESULT expected = nvrhi::FS_OK)
    {
        std::vector<const char*> before;
        std::vector<const char*> after;
        for (const std::string& name : spec.before)
            before.push_back(name.c_str());
        for (const std::string& name : spec.after)
            after.push_back(name.c_str());
        SystemDesc desc{};
        desc.name = spec.name.c_str();
        desc.accesses = spec.accesses.data();
        desc.accessCount = uint32_t(spec.accesses.size());
        desc.phase = spec.phase;
        desc.runBefore = before.data();
        desc.runAfter = after.data();
        desc.runBeforeCount = uint32_t(before.size());
        desc.runAfterCount = uint32_t(after.size());
        desc.flags = spec.flags;
        nvrhi::AutoPtr<ISystem> system;
        CHECK(CreateLambdaSystem(desc, std::forward<Function>(update), &system) == nvrhi::FS_OK);
        SystemId id = SystemId::Invalid;
        CHECK(scheduler->AddSystem(system.Get(), id) == expected);
        CHECK((id != SystemId::Invalid) == (expected == nvrhi::FS_OK));
        return id;
    }

    struct Env
    {
        nvrhi::AutoPtr<IWorld> world;
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<IScheduler> scheduler;
        ComponentId a, b, c, d, e, f, marker, counter;

        Env(uint32_t workers, bool validation)
        {
            WorldDesc worldDesc;
            worldDesc.debugName = "scheduler test";
            worldDesc.validation = validation;
            CHECK(CreateWorld(worldDesc, &world) == nvrhi::FS_OK);
            JobSystemDesc jobDesc;
            jobDesc.debugName = "scheduler test";
            jobDesc.workerThreadCount = workers;
            CHECK(CreateJobSystem(jobDesc, &jobs) == nvrhi::FS_OK);
            CHECK(CreateScheduler(world.Get(), jobs.Get(), &scheduler) == nvrhi::FS_OK);
            a = RegisterComponent<A>(world.Get());
            b = RegisterComponent<B>(world.Get());
            c = RegisterComponent<C>(world.Get());
            d = RegisterComponent<D>(world.Get());
            e = RegisterComponent<E>(world.Get());
            f = RegisterComponent<F>(world.Get());
            marker = RegisterComponent<Marker>(world.Get());
            CHECK(world->RegisterResource(ComponentTraits<Counter>::Desc(), nullptr, counter) == nvrhi::FS_OK);
        }

        ecs::WorldImpl& Impl() const { return *ecs::GetWorldImpl(world.Get()); }

        ecs::SchedulerCounters Counters() const
        {
            ecs::SchedulerCounters counters{};
            CHECK(ecs::GetSchedulerCounters(scheduler.Get(), counters));
            return counters;
        }

        std::vector<SystemId> Order(Phase phase) const
        {
            std::vector<SystemId> ids(256);
            const uint32_t count = ecs::GetScheduleOrder(scheduler.Get(), phase, ids.data(), uint32_t(ids.size()));
            ids.resize(count);
            return ids;
        }

        // `count` entities with the components, default values.
        std::vector<Entity> Create(std::initializer_list<ComponentId> components, uint32_t count) const
        {
            std::vector<ComponentId> list(components);
            EntityCreateDesc desc{};
            desc.components = list.data();
            desc.componentCount = uint32_t(list.size());
            desc.entityCount = count;
            std::vector<Entity> entities(count);
            CHECK(world->CreateEntities(desc, entities.data()) == nvrhi::FS_OK);
            return entities;
        }
    };

    std::string Diagnostics(IScheduler* scheduler)
    {
        nvrhi::string text;
        scheduler->GetDiagnostics(text);
        return std::string(text.c_str());
    }

    // Start and end ticks of system runs, from one counter: p ran before q when end[p] < start[q].
    struct Trace
    {
        static constexpr uint32_t c_Max = 64;
        std::atomic<uint64_t> tick{ 1 };
        uint64_t start[c_Max] = {};
        uint64_t end[c_Max] = {};
        uint32_t runs[c_Max] = {};
        std::thread::id thread[c_Max];
        std::atomic<uint32_t> running{ 0 };
        std::atomic<uint32_t> maxRunning{ 0 };   // the most systems seen running at once
        std::atomic<uint32_t> errors{ 0 };

        void Reset()
        {
            for (uint32_t i = 0; i < c_Max; ++i)
            {
                start[i] = end[i] = 0;
                runs[i] = 0;
            }
            maxRunning = 0;
        }
        void Begin(uint32_t i)
        {
            const uint32_t now = running.fetch_add(1) + 1;
            uint32_t seen = maxRunning.load();
            while (now > seen && !maxRunning.compare_exchange_weak(seen, now))
            {
            }
            start[i] = tick.fetch_add(1);
            thread[i] = std::this_thread::get_id();
            ++runs[i];
        }
        void End(uint32_t i)
        {
            end[i] = tick.fetch_add(1);
            running.fetch_sub(1);
        }
        bool Before(uint32_t p, uint32_t q) const { return end[p] != 0 && start[q] != 0 && end[p] < start[q]; }
        bool Apart(uint32_t p, uint32_t q) const { return Before(p, q) || Before(q, p); }
    };

    // Work long enough that independent systems overlap on several workers.
    void Spin(uint32_t microseconds)
    {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(microseconds);
        while (std::chrono::steady_clock::now() < until)
        {
        }
    }

    struct Xorshift
    {
        uint64_t state;
        explicit Xorshift(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ull) { }
        uint64_t Next()
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        }
        uint32_t Below(uint32_t bound) { return uint32_t(Next() % bound); }
    };

    const char* const c_PhaseNames[] = { "BeginFrame", "PreSimulation", "FixedSimulation", "PostSimulation",
                                         "TransformPropagation", "Bounds", "RenderExtract", "EndFrame" };
}

// ------------------------------------------------------------------------------------------------ ordering

// A small graph with a known order: explicit edges against registration order, conflicts, a free system.
static void test_ordering()
{
    for (uint32_t workers : { 1u, 2u, 8u })
    {
        Env env(workers, true);
        Trace trace;
        enum { Integrate, Accelerate, RenderRead, Count, Late, N };
        auto traced = [&trace](uint32_t i)
        {
            return [&trace, i](ISystemContext*) { trace.Begin(i); Spin(50); trace.End(i); };
        };
        const SystemId integrate = Add(env.scheduler.Get(),
                                       { "integrate", Phase::PreSimulation, { W(env.a), R(env.b) } },
                                       traced(Integrate));
        const SystemId accelerate = Add(env.scheduler.Get(), { "accelerate", Phase::PreSimulation, { W(env.b) },
                                        { "integrate" } }, traced(Accelerate));
        const SystemId renderRead = Add(env.scheduler.Get(), { "render_read", Phase::PreSimulation, { R(env.a) }, {},
                                        { "integrate" } }, traced(RenderRead));
        const SystemId count = Add(env.scheduler.Get(), { "count", Phase::PreSimulation, { W(env.counter) } },
                                   traced(Count));
        const SystemId late = Add(env.scheduler.Get(), { "late", Phase::PreSimulation, { R(env.b) }, { "accelerate" } },
                                  traced(Late));
        CHECK(env.scheduler->Build() == nvrhi::FS_OK);
        CHECK(env.scheduler->GetAmbiguityCount() == 0);
        // Kahn's order with ties by registration: count (3) and late (4) are ready first.
        const std::vector<SystemId> expected = { count, late, accelerate, integrate, renderRead };
        CHECK(env.Order(Phase::PreSimulation) == expected);

        for (uint32_t run = 0; run < 40; ++run)
        {
            trace.Reset();
            CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
            for (uint32_t i = 0; i < N; ++i)
                CHECK(trace.runs[i] == 1);
            CHECK(trace.Before(Late, Accelerate));
            CHECK(trace.Before(Accelerate, Integrate));
            CHECK(trace.Before(Integrate, RenderRead));
        }
        SystemStats stats{};
        CHECK(env.scheduler->GetSystemStats(count, stats) == nvrhi::FS_OK);
        CHECK(stats.runCount == 40 && stats.lastMilliseconds > 0.0 && stats.averageMilliseconds > 0.0);
        const ecs::SchedulerCounters counters = env.Counters();
        CHECK(counters.systemRuns == 40 * N && counters.jobsScheduled == 40 * N && counters.inlineRuns == 0);
        // The job graph: late -> accelerate -> integrate -> render_read, count alone: 3 edges, and a sync job over the
        // two sinks.
        CHECK(counters.edges == 3 + 2);
        CHECK(counters.syncJobs == 40);
    }
}

// Seeded random system sets checked against a brute-force model: the schedule order (Kahn, ties by registration), the
// ambiguity count, and at run time that every ordered pair ran in order, exclusive systems alone and main-thread ones
// on the calling thread, disabled ones not at all.
static void test_random_graphs()
{
    uint32_t checkedPairs = 0;
    uint32_t ambiguousTotal = 0;
    for (uint32_t seed = 1; seed <= 12; ++seed)
    {
        for (uint32_t workers : { 1u, 2u, 8u })
        {
            Xorshift random(seed * 7919u);
            Env env(workers, false);
            const ComponentId components[] = { env.a, env.b, env.c, env.d, env.e, env.f };
            const uint32_t n = 10 + random.Below(20);
            std::vector<uint32_t> rank(n);   // a hidden topological order that the explicit edges follow
            for (uint32_t i = 0; i < n; ++i)
                rank[i] = i;
            for (uint32_t i = n; i > 1; --i)
                std::swap(rank[i - 1], rank[random.Below(i)]);

            struct Model
            {
                std::vector<ComponentAccess> accesses;
                bool exclusive = false;
                bool mainThread = false;
                bool enabled = true;
            };
            std::vector<Model> model(n);
            std::vector<std::vector<uint8_t>> edge(n, std::vector<uint8_t>(n, 0));   // registration indices
            std::vector<Spec> specs(n);
            for (uint32_t i = 0; i < n; ++i)
            {
                specs[i].name = "s" + std::to_string(i);
                const uint32_t accessCount = random.Below(4);
                for (uint32_t k = 0; k < accessCount; ++k)
                {
                    const uint32_t roll = random.Below(10);
                    const Access access = roll < 4 ? Access::Write : roll < 9 ? Access::Read : Access::Filter;
                    model[i].accesses.push_back(ComponentAccess{ components[random.Below(6)], access, {} });
                }
                model[i].exclusive = random.Below(14) == 0;
                model[i].mainThread = !model[i].exclusive && random.Below(6) == 0;
                model[i].enabled = random.Below(8) != 0;
                specs[i].accesses = model[i].accesses;
                specs[i].flags = (model[i].exclusive ? SystemFlags::Exclusive : SystemFlags::None) |
                                 (model[i].mainThread ? SystemFlags::MainThread : SystemFlags::None) |
                                 (model[i].enabled ? SystemFlags::None : SystemFlags::StartDisabled);
            }
            for (uint32_t i = 0; i < n; ++i)
            {
                for (uint32_t j = 0; j < n; ++j)
                {
                    if (rank[i] >= rank[j] || random.Below(9) != 0)
                        continue;
                    edge[i][j] = 1;   // i before j, said by either side
                    if (random.Below(2))
                        specs[i].before.push_back(specs[j].name);
                    else
                        specs[j].after.push_back(specs[i].name);
                }
            }

            Trace trace;
            const std::thread::id caller = std::this_thread::get_id();
            std::vector<SystemId> ids(n);
            for (uint32_t i = 0; i < n; ++i)
            {
                const bool exclusive = model[i].exclusive;
                ids[i] = Add(env.scheduler.Get(), specs[i], [&trace, i, exclusive](ISystemContext* context)
                {
                    trace.Begin(i);
                    if (exclusive && trace.running.load() != 1)
                        ++trace.errors;
                    if (!context->GetWorld()->IsInPhase())
                        ++trace.errors;
                    Spin(20);
                    trace.End(i);
                });
            }

            // The model's order: Kahn's algorithm with the lowest registration index first.
            std::vector<uint32_t> order;
            std::vector<uint32_t> inDegree(n, 0);
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t j = 0; j < n; ++j)
                    inDegree[j] += edge[i][j];
            std::vector<uint8_t> done(n, 0);
            for (uint32_t step = 0; step < n; ++step)
            {
                uint32_t next = n;
                for (uint32_t i = 0; i < n && next == n; ++i)
                    if (!done[i] && inDegree[i] == 0)
                        next = i;
                CHECK(next != n);
                done[next] = 1;
                order.push_back(next);
                for (uint32_t j = 0; j < n; ++j)
                    inDegree[j] -= edge[next][j];
            }
            std::vector<SystemId> expectedOrder;
            for (uint32_t i : order)
                expectedOrder.push_back(ids[i]);
            CHECK(env.Order(Phase::PreSimulation) == expectedOrder);

            // Explicit reachability, conflicts, ambiguities.
            std::vector<std::vector<uint8_t>> reach = edge;
            for (uint32_t k = 0; k < n; ++k)
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t j = 0; j < n; ++j)
                        if (reach[i][k] && reach[k][j])
                            reach[i][j] = 1;
            auto conflicts = [&](uint32_t i, uint32_t j)
            {
                if (model[i].exclusive || model[j].exclusive)
                    return true;
                for (const ComponentAccess& x : model[i].accesses)
                    for (const ComponentAccess& y : model[j].accesses)
                        if (x.component == y.component && x.access != Access::Filter && y.access != Access::Filter &&
                            (x.access == Access::Write || y.access == Access::Write))
                            return true;
                return false;
            };
            uint32_t ambiguities = 0;
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t j = i + 1; j < n; ++j)
                    if (conflicts(i, j) && !reach[i][j] && !reach[j][i])
                        ++ambiguities;
            CHECK(env.scheduler->GetAmbiguityCount() == ambiguities);
            ambiguousTotal += ambiguities;

            // Everything that must be ordered: explicit paths and conflicts, directed by the schedule order.
            std::vector<uint32_t> position(n);
            for (uint32_t p = 0; p < n; ++p)
                position[order[p]] = p;
            for (uint32_t run = 0; run < 6; ++run)
            {
                trace.Reset();
                CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
                for (uint32_t i = 0; i < n; ++i)
                {
                    CHECK(trace.runs[i] == (model[i].enabled ? 1u : 0u));
                    if (model[i].enabled && (model[i].mainThread || model[i].exclusive))
                        CHECK(trace.thread[i] == caller);
                }
                for (uint32_t i = 0; i < n; ++i)
                {
                    for (uint32_t j = 0; j < n; ++j)
                    {
                        if (i == j || !model[i].enabled || !model[j].enabled || position[i] > position[j])
                            continue;
                        if (reach[i][j] || conflicts(i, j))
                        {
                            CHECK(trace.Before(i, j));
                            ++checkedPairs;
                        }
                    }
                }
            }
            CHECK(trace.errors == 0);
            // Enabling everything changes the job graph, not the order or the reports.
            for (uint32_t i = 0; i < n; ++i)
                CHECK(env.scheduler->SetSystemEnabled(ids[i], true) == nvrhi::FS_OK);
            trace.Reset();
            CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
            for (uint32_t i = 0; i < n; ++i)
                CHECK(trace.runs[i] == 1);
            CHECK(env.Order(Phase::PreSimulation) == expectedOrder);
            CHECK(env.scheduler->GetAmbiguityCount() == ambiguities);
            CHECK(trace.errors == 0);
        }
    }
    printf("test_ecs_scheduler: random graphs: %u ordered pairs checked, %u ambiguities\n", checkedPairs,
           ambiguousTotal);
}

// ---------------------------------------------------------------------------------- ambiguities and cycles

static void test_ambiguities_and_cycles()
{
    Env env(2, true);
    auto nothing = [](ISystemContext*) { };

    // Two writers of A without an order: one ambiguity, run in registration order, never at once.
    Trace trace;
    Add(env.scheduler.Get(), { "w1", Phase::PreSimulation, { W(env.a) } },
        [&trace](ISystemContext*) { trace.Begin(0); Spin(200); trace.End(0); });
    Add(env.scheduler.Get(), { "w2", Phase::PreSimulation, { W(env.a) } },
        [&trace](ISystemContext*) { trace.Begin(1); Spin(200); trace.End(1); });
    CHECK(env.scheduler->Build() == nvrhi::FS_OK);
    CHECK(env.scheduler->GetAmbiguityCount() == 1);
    std::string text = Diagnostics(env.scheduler.Get());
    CHECK(text.find("ambiguity: 'w1' and 'w2' both access 'sched_test::A'") != std::string::npos ||
          text.find("ambiguity: 'w1' and 'w2' both access") != std::string::npos);
    for (uint32_t run = 0; run < 20; ++run)
    {
        trace.Reset();
        CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
        CHECK(trace.Before(0, 1));
    }

    // Readers never conflict, nor do Filter accesses; a reader and a writer do.
    Add(env.scheduler.Get(), { "r1", Phase::PostSimulation, { R(env.b) } }, nothing);
    Add(env.scheduler.Get(), { "r2", Phase::PostSimulation, { R(env.b), Filter(env.c) } }, nothing);
    Add(env.scheduler.Get(), { "f1", Phase::PostSimulation, { W(env.c) } }, nothing);
    CHECK(env.scheduler->GetAmbiguityCount() == 1);
    const SystemId writerB = Add(env.scheduler.Get(), { "wb", Phase::PostSimulation, { W(env.b) } }, nothing);
    CHECK(env.scheduler->GetAmbiguityCount() == 3);   // r1/wb, r2/wb
    CHECK(env.scheduler->RemoveSystem(writerB) == nvrhi::FS_OK);
    CHECK(env.scheduler->GetAmbiguityCount() == 1);

    // A transitive explicit order resolves an ambiguity: x before y before z, x and z both write D.
    Add(env.scheduler.Get(), { "x", Phase::Bounds, { W(env.d) }, { "y" } }, nothing);
    Add(env.scheduler.Get(), { "z", Phase::Bounds, { W(env.d) }, {}, { "y" } }, nothing);
    Add(env.scheduler.Get(), { "y", Phase::Bounds, { R(env.e) } }, nothing);
    CHECK(env.scheduler->GetAmbiguityCount() == 1);
    // An exclusive system conflicts with every other system of its phase.
    const SystemId exclusive = Add(env.scheduler.Get(), { "ex", Phase::Bounds, {}, {}, {}, SystemFlags::Exclusive },
                                   nothing);
    CHECK(env.scheduler->GetAmbiguityCount() == 4);
    CHECK(Diagnostics(env.scheduler.Get()).find("('ex' is exclusive)") != std::string::npos);
    CHECK(env.scheduler->RemoveSystem(exclusive) == nvrhi::FS_OK);
    Add(env.scheduler.Get(), { "ex2", Phase::Bounds, {}, {}, { "z" }, SystemFlags::Exclusive }, nothing);
    CHECK(env.scheduler->GetAmbiguityCount() == 1);   // after z, which is after y, after x

    // Names that match nothing, or a system of another phase: ignored and reported.
    Add(env.scheduler.Get(), { "lost", Phase::EndFrame, { R(env.f) }, { "nobody" }, { "w1" } }, nothing);
    CHECK(env.scheduler->Build() == nvrhi::FS_OK);
    text = Diagnostics(env.scheduler.Get());
    CHECK(text.find("'lost' runs before 'nobody', which matches no system; ignored") != std::string::npos);
    CHECK(text.find("'lost' runs after 'w1', a system of phase 'PreSimulation'; ignored") != std::string::npos);

    // A cycle: Build fails, the phase does not run, the others do; a range with it runs nothing.
    Add(env.scheduler.Get(), { "c1", Phase::RenderExtract, {}, { "c2" } }, nothing);
    const SystemId c2 = Add(env.scheduler.Get(), { "c2", Phase::RenderExtract, {}, { "c3" } }, nothing);
    uint32_t c3Runs = 0;
    Add(env.scheduler.Get(), { "c3", Phase::RenderExtract, {}, { "c1" } }, [&c3Runs](ISystemContext*) { ++c3Runs; });
    CHECK(env.scheduler->Build() == nvrhi::FE_INVALID_ARGS);
    text = Diagnostics(env.scheduler.Get());
    CHECK(text.find("ordering cycle 'c1' -> 'c2' -> 'c3' -> 'c1'") != std::string::npos);
    CHECK(env.scheduler->RunPhase(Phase::RenderExtract) == nvrhi::FE_INVALID_ARGS);
    trace.Reset();
    CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
    CHECK(trace.runs[0] == 1);
    trace.Reset();
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FE_INVALID_ARGS);
    CHECK(trace.runs[0] == 0 && c3Runs == 0);
    CHECK(env.scheduler->RemoveSystem(c2) == nvrhi::FS_OK);
    CHECK(env.scheduler->Build() == nvrhi::FS_OK);
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
    CHECK(trace.runs[0] == 1 && c3Runs == 1);

    // A system that names itself is a cycle of one.
    const SystemId self = Add(env.scheduler.Get(), { "self", Phase::BeginFrame, {}, { "self" } }, nothing);
    CHECK(env.scheduler->Build() == nvrhi::FE_INVALID_ARGS);
    CHECK(Diagnostics(env.scheduler.Get()).find("ordering cycle 'self' -> 'self'") != std::string::npos);
    CHECK(env.scheduler->RemoveSystem(self) == nvrhi::FS_OK);
    CHECK(env.scheduler->Build() == nvrhi::FS_OK);
}

// ------------------------------------------------------------------------------ exclusive and main thread

static void test_exclusive_and_main_thread()
{
    for (uint32_t workers : { 1u, 2u, 8u })
    {
        Env env(workers, true);
        Trace trace;
        const std::thread::id caller = std::this_thread::get_id();
        std::atomic<uint32_t> failures{ 0 };
        std::atomic<bool> independentStarted{ false };
        std::atomic<uint32_t> sawIndependent{ 0 };
        ComponentId pair[2] = { env.a, env.b };

        enum { J1, M, J2, J5, Ex, J3, J4, N };
        Add(env.scheduler.Get(), { "j1", Phase::PreSimulation, { W(env.a) } }, [&](ISystemContext* context)
        {
            trace.Begin(J1);
            // Immediate structural changes are refused inside a phase.
            EntityCreateDesc desc{};
            desc.components = pair;
            desc.componentCount = 2;
            desc.entityCount = 1;
            Entity entity;
            if (context->GetWorld()->CreateEntities(desc, &entity) != nvrhi::FE_INVALID_ARGS ||
                !context->GetWorld()->IsInPhase())
                ++failures;
            Spin(100);
            trace.End(J1);
        });
        Add(env.scheduler.Get(), { "m", Phase::PreSimulation, { R(env.a) }, {}, { "j1" }, SystemFlags::MainThread },
            [&](ISystemContext*)
        {
            trace.Begin(M);
            // j5 does not depend on m: it was scheduled before m runs, so it can start meanwhile.
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!independentStarted.load() && std::chrono::steady_clock::now() < until)
                std::this_thread::yield();
            sawIndependent += independentStarted.load() ? 1 : 0;
            trace.End(M);
        });
        Add(env.scheduler.Get(), { "j2", Phase::PreSimulation, { W(env.a) }, {}, { "m" } },
            [&](ISystemContext*) { trace.Begin(J2); Spin(50); trace.End(J2); });
        Add(env.scheduler.Get(), { "j5", Phase::PreSimulation, { W(env.c) } },
            [&](ISystemContext*) { trace.Begin(J5); independentStarted = true; Spin(50); trace.End(J5); });
        Add(env.scheduler.Get(), { "ex", Phase::PreSimulation, {}, {}, { "j2" }, SystemFlags::Exclusive },
            [&](ISystemContext* context)
        {
            trace.Begin(Ex);
            if (trace.running.load() != 1)
                ++failures;
            // An exclusive system changes structure at once.
            EntityCreateDesc desc{};
            desc.components = pair;
            desc.componentCount = 2;
            desc.entityCount = 3;
            Entity entities[3];
            if (context->GetWorld()->CreateEntities(desc, entities) != nvrhi::FS_OK)
                ++failures;
            trace.End(Ex);
        });
        Add(env.scheduler.Get(), { "j3", Phase::PreSimulation, { W(env.b) } },
            [&](ISystemContext*) { trace.Begin(J3); Spin(50); trace.End(J3); });
        Add(env.scheduler.Get(), { "j4", Phase::PreSimulation, { R(env.b) }, {}, { "j3" } },
            [&](ISystemContext*) { trace.Begin(J4); Spin(50); trace.End(J4); });

        Query<Read<A>> all(env.world.Get());
        for (uint32_t run = 0; run < 10; ++run)
        {
            trace.Reset();
            independentStarted = false;
            CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
            CHECK(!env.world->IsInPhase());
            for (uint32_t i = 0; i < N; ++i)
                CHECK(trace.runs[i] == 1);
            CHECK(trace.thread[M] == caller && trace.thread[Ex] == caller);
            CHECK(trace.Before(J1, M) && trace.Before(M, J2) && trace.Before(J2, Ex) && trace.Before(J3, J4));
            // The exclusive one runs after everything before it in the schedule order, before everything after it.
            for (uint32_t i : { J1, M, J2, J5 })
                CHECK(trace.Before(i, Ex));
            for (uint32_t i : { J3, J4 })
                CHECK(trace.Before(Ex, i));
            CHECK(all.CountEntities() == 3 * (run + 1));
        }
        CHECK(failures == 0);
        CHECK(sawIndependent == 10);
    }
}

// ----------------------------------------------------------------------------------- versions and filters

static void test_versions_and_changed()
{
    Env env(4, true);
    env.Create({ env.a, env.b }, 12000);
    Query<Changed<A>> changedA(env.world.Get(), "changed A");
    Query<Changed<B>> changedB(env.world.Get(), "changed B");
    Query<Write<A>> writeA(env.world.Get(), "write A");
    Query<Write<B>> writeB(env.world.Get(), "write B");
    const uint32_t chunks = writeA.Get()->GetChunkCount();
    CHECK(chunks > 4);

    struct Seen
    {
        uint32_t chunks[64] = {};
        uint64_t version[64] = {};
        uint64_t lastRun[64] = {};
    };
    Seen beforeWriter, afterWriter, self, writer;
    uint32_t frame = 0;
    auto record = [&frame](Seen& seen, ISystemContext* context)
    {
        seen.version[frame] = context->GetVersion();
        seen.lastRun[frame] = context->GetLastRunVersion();
    };
    auto countChunks = [](const Query<Changed<A>>& query, ISystemContext* context)
    {
        uint32_t count = 0;
        query.ForEachChunk(context, [&count](const ChunkView&, const A*) { ++count; });
        return count;
    };
    Add(env.scheduler.Get(), { "before_writer", Phase::PreSimulation, { R(env.a) }, { "writer" } },
        [&](ISystemContext* context)
    {
        record(beforeWriter, context);
        beforeWriter.chunks[frame] = countChunks(changedA, context);
    });
    Add(env.scheduler.Get(), { "writer", Phase::PreSimulation, { W(env.a) } }, [&](ISystemContext* context)
    {
        record(writer, context);
        if (frame % 3 == 0)
            writeA.ParallelForChunks(context, 2, [](const ChunkView& view, A* values)
            {
                for (uint32_t row = 0; row < view.count; ++row)
                    values[row].value += 1.0f;
            });
    });
    Add(env.scheduler.Get(), { "after_writer", Phase::PreSimulation, { R(env.a) }, {}, { "writer" } },
        [&](ISystemContext* context)
    {
        record(afterWriter, context);
        afterWriter.chunks[frame] = countChunks(changedA, context);
    });
    Add(env.scheduler.Get(), { "self", Phase::PreSimulation, { W(env.b) } }, [&](ISystemContext* context)
    {
        record(self, context);
        uint32_t count = 0;
        changedB.ForEachChunk(context, [&count](const ChunkView&, const B*) { ++count; });
        self.chunks[frame] = count;
        writeB.ForEach(context, [](Entity, B& value) { value.value += 1.0f; });
    });
    CHECK(env.scheduler->Build() == nvrhi::FS_OK && env.scheduler->GetAmbiguityCount() == 0);

    const uint64_t errors = env.Impl().GetValidationErrorCount();
    for (frame = 0; frame < 12; ++frame)
    {
        const uint64_t before = env.world->GetVersion();
        CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
        // One version per system, in schedule order, then one after the phase.
        CHECK(beforeWriter.version[frame] == before + 1 && writer.version[frame] == before + 2);
        CHECK(afterWriter.version[frame] == before + 3 && self.version[frame] == before + 4);
        CHECK(env.world->GetVersion() == before + 5);
        if (frame == 0)
        {
            CHECK(beforeWriter.lastRun[0] == c_AllVersions && self.lastRun[0] == c_AllVersions);
        }
        else
        {
            CHECK(beforeWriter.lastRun[frame] == beforeWriter.version[frame - 1] &&
                  self.lastRun[frame] == self.version[frame - 1]);
        }

        // The writer writes every third frame: the system before it sees that a frame later, the one after it at once.
        const bool afterSees = frame % 3 == 0;
        const bool beforeSees = frame == 0 || frame % 3 == 1;
        CHECK(afterWriter.chunks[frame] == (afterSees ? chunks : 0));
        CHECK(beforeWriter.chunks[frame] == (beforeSees ? chunks : 0));
        // A system's own writes are not changes for its next run.
        CHECK(self.chunks[frame] == (frame == 0 ? chunks : 0));
    }
    CHECK(env.Impl().GetValidationErrorCount() == errors);

    // Playback at a sync point stamps the version taken after the phase: newer than every system of the phase, the
    // last one included. Added sees the entities that a command buffer created.
    Env env2(2, true);
    const std::vector<Entity> targets = env2.Create({ env2.a }, 10);
    Query<Changed<A>> changed2(env2.world.Get());
    Query<Added<A>> added2(env2.world.Get());
    uint32_t seenChanged[16] = {};
    uint32_t seenAdded[16] = {};
    uint32_t step = 0;
    Add(env2.scheduler.Get(), { "setter", Phase::PostSimulation }, [&](ISystemContext* context)
    {
        if (step % 2 == 1)
        {
            const A value{ float(step) };
            context->GetCommandBuffer()->SetComponent(targets[3], env2.a, &value, 0);
        }
        if (step == 4)
        {
            const ComponentId list[1] = { env2.a };
            EntityCreateDesc desc{};
            desc.components = list;
            desc.componentCount = 1;
            desc.entityCount = 2;
            Entity placeholders[2];
            context->GetCommandBuffer()->CreateEntities(desc, 1, placeholders);
        }
    });
    Add(env2.scheduler.Get(), { "watcher", Phase::PostSimulation, { R(env2.a) } }, [&](ISystemContext* context)
    {
        changed2.ForEachChunk(context, [&](const ChunkView&, const A*) { ++seenChanged[step]; });
        added2.ForEachChunk(context, [&](const ChunkView&, const A*) { ++seenAdded[step]; });
    });
    for (step = 0; step < 8; ++step)
        CHECK(env2.scheduler->RunPhase(Phase::PostSimulation) == nvrhi::FS_OK);
    // Step 0 sees everything; the sets of steps 1, 3, 5 and the creation of step 4 are seen one step later.
    CHECK(seenChanged[0] == 1 && seenAdded[0] == 1);
    for (step = 1; step < 8; ++step)
    {
        CHECK(seenChanged[step] == (step % 2 == 0 || step == 5 ? 1u : 0u));
        CHECK(seenAdded[step] == (step == 5 ? 1u : 0u));
    }
    CHECK(GetComponent<A>(env2.world.Get(), targets[3])->value == 7.0f);
}

// ---------------------------------------------------------------------------------- command-buffer playback

static void test_playback_order()
{
    Env env(8, true);
    const std::vector<Entity> sources = env.Create({ env.a, env.c }, 5000);
    {
        // Initial values: C holds the row's index in creation order.
        uint32_t index = 0;
        Query<Write<C>> query(env.world.Get());
        query.ForEach(ChunkRequest{}, [&index](Entity, C& value) { value.value = index++; });
    }
    Query<Read<C>> sourcesQuery(env.world.Get(), "sources");
    Query<Read<D>> spawned(env.world.Get(), "spawned");
    const uint32_t sourceChunks = sourcesQuery.Get()->GetChunkCount();
    CHECK(sourceChunks > 4);

    // Registered first, ordered last: its set wins.
    Add(env.scheduler.Get(), { "late_setter", Phase::PostSimulation, {}, {}, { "early_setter" } },
        [&](ISystemContext* context)
    {
        const A value{ 2.0f };
        context->GetCommandBuffer()->SetComponent(sources[0], env.a, &value, 0);
    });
    Add(env.scheduler.Get(), { "early_setter", Phase::PostSimulation }, [&](ISystemContext* context)
    {
        const A value{ 1.0f };
        context->GetCommandBuffer()->SetComponent(sources[0], env.a, &value, 0);
    });
    // Parallel recording: one entity per source row, keyed by (chunk, row): the result does not depend on which
    // worker recorded what.
    Add(env.scheduler.Get(), { "spawner", Phase::PostSimulation, { R(env.c) } }, [&](ISystemContext* context)
    {
        sourcesQuery.ParallelForChunks(context, 1, [&](const ChunkView& view, const C* values)
        {
            ICommandBuffer* commands = context->GetCommandBuffer();
            const ComponentId list[1] = { env.d };
            for (uint32_t row = 0; row < view.count; ++row)
            {
                const D value{ values[row].value };
                const void* valueList[1] = { &value };
                EntityCreateDesc desc{};
                desc.components = list;
                desc.values = valueList;
                desc.componentCount = 1;
                desc.entityCount = 1;
                Entity placeholder;
                commands->CreateEntities(desc, MakeSortKey(0, view.chunkIndex, row), &placeholder);
            }
        });
    });
    // A phase without a sync point: its commands wait for the next one, and play back before later phases' commands.
    Add(env.scheduler.Get(), { "pre_setter", Phase::PreSimulation }, [&](ISystemContext* context)
    {
        const A value{ 5.0f };
        context->GetCommandBuffer()->SetComponent(sources[1], env.a, &value, 0);
        const A other{ 9.0f };
        context->GetCommandBuffer()->SetComponent(sources[2], env.a, &other, 0);
    });
    Add(env.scheduler.Get(), { "post_setter", Phase::PostSimulation }, [&](ISystemContext* context)
    {
        const A value{ 7.0f };
        context->GetCommandBuffer()->SetComponent(sources[1], env.a, &value, 0);
    });
    // A main-thread system records into the buffer of threads without a slot.
    Add(env.scheduler.Get(), { "main_destroyer", Phase::PostSimulation, {}, {}, {}, SystemFlags::MainThread },
        [&](ISystemContext* context)
    {
        if (context->GetWorld()->IsAlive(sources[4999]))
            context->GetCommandBuffer()->DestroyEntity(sources[4999], 0);
    });

    for (uint32_t round = 0; round < 3; ++round)
    {
        // Undo the last round.
        if (round > 0)
        {
            std::vector<Entity> old;
            spawned.ForEach(ChunkRequest{}, [&old](Entity entity, const D&) { old.push_back(entity); });
            CHECK(env.world->DestroyEntities(old.data(), uint32_t(old.size())) == nvrhi::FS_OK);
        }
        CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
        CHECK(GetComponent<A>(env.world.Get(), sources[1])->value == (round == 0 ? 0.0f : 7.0f));
        CHECK(env.scheduler->RunPhase(Phase::PostSimulation) == nvrhi::FS_OK);
        CHECK(GetComponent<A>(env.world.Get(), sources[0])->value == 2.0f);
        CHECK(GetComponent<A>(env.world.Get(), sources[1])->value == 7.0f);
        CHECK(GetComponent<A>(env.world.Get(), sources[2])->value == 9.0f);
        CHECK(!env.world->IsAlive(sources[4999]));
        // The spawned entities are in source order (the destroyed source was alive when recorded in round 0 only).
        std::vector<uint32_t> values;
        spawned.ForEach(ChunkRequest{}, [&values](Entity, const D& value) { values.push_back(value.value); });
        CHECK(values.size() == (round == 0 ? 5000u : 4999u));
        for (uint32_t i = 0; i < values.size(); ++i)
            CHECK(values[i] == i);
    }

    // A phase that runs again before a sync point first plays back its pending commands.
    const ecs::SchedulerCounters before = env.Counters();
    CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
    CHECK(env.Counters().syncPoints == before.syncPoints);
    CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
    CHECK(env.Counters().syncPoints == before.syncPoints + 1);
    CHECK(GetComponent<A>(env.world.Get(), sources[1])->value == 5.0f);
    // RemoveSystem keeps a removed system's pending commands for the next sync point.
    uint32_t preSetter = 0;
    for (SystemId id : env.Order(Phase::PreSimulation))
        preSetter = uint32_t(id);
    A reset{ 0.0f };
    *GetComponentForWrite<A>(env.world.Get(), sources[2]) = reset;
    CHECK(env.scheduler->RemoveSystem(SystemId(preSetter)) == nvrhi::FS_OK);
    CHECK(env.scheduler->RunPhase(Phase::RenderExtract) == nvrhi::FS_OK);
    CHECK(GetComponent<A>(env.world.Get(), sources[2])->value == 9.0f);
    CHECK(env.Impl().GetValidationErrorCount() == 0);
}

// ----------------------------------------------------------------------------------------- fixed stepping

namespace
{
    class TestDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(TestDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        SimulationDomainDesc desc{};
        uint32_t gathers = 0;
        uint32_t scatters = 0;
        uint32_t steps = 0;
        double dts[64] = {};
        uint64_t stepIndices[64] = {};
        uint64_t gatherVersions[64] = {};
        // ChangesStructure: each step creates one Marker entity and checks that the earlier steps' ones exist.
        IQuery* markers = nullptr;
        ComponentId marker = ComponentId::Invalid;
        uint32_t errors = 0;
        Entity lastPlaceholder = Entity::Null;
        std::thread::id stepThread;

        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override { return retVal = desc; }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch&, ISimulationContext*) noexcept override
        {
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch&, ISimulationContext*) noexcept override
        {
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext* context) noexcept override
        {
            gatherVersions[gathers % 64] = context->GetLastGatherVersion();
            ++gathers;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT Step(double dt, ISimulationContext* context) noexcept override
        {
            stepThread = std::this_thread::get_id();
            dts[steps % 64] = dt;
            stepIndices[steps % 64] = context->GetStepIndex();
            if (markers)
            {
                if (markers->CountEntities() != steps)
                    ++errors;
                const ComponentId list[1] = { marker };
                EntityCreateDesc create{};
                create.components = list;
                create.componentCount = 1;
                create.entityCount = 1;
                context->GetCommandBuffer()->CreateEntities(create, 0, &lastPlaceholder);
            }
            ++steps;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept override
        {
            if (context->GetColumnCount(ComponentId(0)) != 0 || context->GetCommandList() != nullptr)
                ++errors;
            ++scatters;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT CaptureState(IStateWriter*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT RestoreState(IStateReader*) noexcept override { return nvrhi::FS_OK; }
    };

    FrameTime MakeFrame(double time, double delta, uint64_t index) { return FrameTime{ time, delta, index }; }
}

static void test_fixed_simulation()
{
    Env env(2, true);
    nvrhi::AutoPtr<TestDomain> domain = MAKE_RC_OBJ_PTR(TestDomain);
    domain->desc.name = "dom";
    domain->desc.fixedTimeStep = 0.25;
    domain->desc.maxStepsPerFrame = 3;
    domain->desc.substeps = 1;
    const ComponentAccess inputs[1] = { R(env.a) };
    const ComponentAccess outputs[1] = { W(env.b) };
    domain->desc.inputs = inputs;
    domain->desc.inputCount = 1;
    domain->desc.outputs = outputs;
    domain->desc.outputCount = 1;
    SimulationDomainId id = SimulationDomainId::Invalid;
    CHECK(env.scheduler->AddSimulationDomain(domain.Get(), id) == nvrhi::FS_OK && id != SimulationDomainId::Invalid);

    // A plain FixedSimulation system runs once per RunPhase; the domain steps as its accumulator says.
    uint32_t plainRuns = 0;
    const FrameTime* seenTime = nullptr;
    Add(env.scheduler.Get(), { "plain_fixed", Phase::FixedSimulation, { R(env.counter) }, {}, { "dom.step" } },
        [&](ISystemContext* context) { ++plainRuns; seenTime = &context->GetFrameTime(); });
    CHECK(env.scheduler->Build() == nvrhi::FS_OK && env.scheduler->GetAmbiguityCount() == 0);
    std::string text = Diagnostics(env.scheduler.Get());
    CHECK(text.find("phase 'PreSimulation': 'dom.gather'") != std::string::npos);
    CHECK(text.find("phase 'FixedSimulation': 'dom.step', 'plain_fixed'") != std::string::npos);
    CHECK(text.find("phase 'PostSimulation': 'dom.scatter'") != std::string::npos);

    // Deltas in binary fractions: 0.375 -> 1 step (0.125 left); 0.125 -> 1 (0); 2.0 -> 8 whole, 3 run, the rest
    // dropped; 0.0625 -> 0; 0.4375 -> 2 (0.5 total); 0.1875 -> 0 (0.1875 left); 0.0625 -> 1.
    const double deltas[] = { 0.375, 0.125, 2.0, 0.0625, 0.4375, 0.1875, 0.0625 };
    const uint32_t expectedSteps[] = { 1, 1, 3, 0, 2, 0, 1 };
    double time = 0.0;
    uint32_t totalSteps = 0;
    for (uint32_t frame = 0; frame < 7; ++frame)
    {
        time += deltas[frame];
        const FrameTime frameTime = MakeFrame(time, deltas[frame], frame);
        env.scheduler->SetFrameTime(frameTime);
        const FrameTime* resource = GetResource<FrameTime>(env.world.Get());
        CHECK(resource && resource->time == time && resource->deltaTime == deltas[frame] &&
              resource->frameIndex == frame);
        CHECK(env.scheduler->RunPhases(Phase::PreSimulation, Phase::PostSimulation) == nvrhi::FS_OK);
        totalSteps += expectedSteps[frame];
        CHECK(domain->steps == totalSteps);
        CHECK(domain->gathers == frame + 1 && domain->scatters == frame + 1 && plainRuns == frame + 1);
        CHECK(seenTime == resource);
    }
    for (uint32_t s = 0; s < totalSteps; ++s)
        CHECK(domain->dts[s] == 0.25 && domain->stepIndices[s] == s);
    CHECK(env.Counters().fixedSteps == totalSteps);
    CHECK(domain->gatherVersions[0] == c_AllVersions && domain->gatherVersions[1] > domain->gatherVersions[0]);
    CHECK(domain->errors == 0);

    // A domain that changes structure: a sync point after every step, its step system exclusive.
    nvrhi::AutoPtr<TestDomain> builder = MAKE_RC_OBJ_PTR(TestDomain);
    builder->desc.name = "builder";
    builder->desc.fixedTimeStep = 0.5;
    builder->desc.maxStepsPerFrame = 4;
    builder->desc.flags = SimulationDomainFlags::ChangesStructure;
    Query<With<Marker>> markers(env.world.Get());
    builder->markers = markers.Get();
    builder->marker = env.marker;
    SimulationDomainId builderId = SimulationDomainId::Invalid;
    CHECK(env.scheduler->AddSimulationDomain(builder.Get(), builderId) == nvrhi::FS_OK);
    // The new step system is exclusive: plain_fixed is ordered after dom.step only.
    CHECK(env.scheduler->GetAmbiguityCount() == 2);
    env.scheduler->SetFrameTime(MakeFrame(time + 1.5, 1.5, 7));
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(builder->steps == 3 && builder->errors == 0 && markers.CountEntities() == 3);
    CHECK(builder->stepThread == std::this_thread::get_id());
    CHECK(env.world->ResolvePlaceholder(builder->lastPlaceholder) != Entity::Null);
    CHECK(env.scheduler->RemoveSimulationDomain(builderId) == nvrhi::FS_OK);
    CHECK(env.scheduler->RemoveSimulationDomain(builderId) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->GetAmbiguityCount() == 0);

    // Arguments.
    nvrhi::AutoPtr<TestDomain> bad = MAKE_RC_OBJ_PTR(TestDomain);
    bad->desc.name = "bad";
    bad->desc.fixedTimeStep = 0.0;
    bad->desc.maxStepsPerFrame = 1;
    SimulationDomainId badId;
    CHECK(env.scheduler->AddSimulationDomain(bad.Get(), badId) == nvrhi::FE_INVALID_ARGS);
    CHECK(badId == SimulationDomainId::Invalid);
    bad->desc.fixedTimeStep = 0.1;
    bad->desc.maxStepsPerFrame = 0;
    CHECK(env.scheduler->AddSimulationDomain(bad.Get(), badId) == nvrhi::FE_INVALID_ARGS);
    bad->desc.maxStepsPerFrame = 1;
    bad->desc.queue = SimulationQueue::Compute;
    CHECK(env.scheduler->AddSimulationDomain(bad.Get(), badId) == nvrhi::FS_OK);
    CHECK(env.scheduler->RemoveSimulationDomain(badId) == nvrhi::FS_OK);
    bad->desc.queue = SimulationQueue::Cpu;
    bad->desc.name = "dom";   // dom.gather is taken
    CHECK(env.scheduler->AddSimulationDomain(bad.Get(), badId) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->AddSimulationDomain(nullptr, badId) == nvrhi::FE_INVALID_ARGS);
    // A domain's systems are removed with the domain only.
    const std::vector<SystemId> fixedOrder = env.Order(Phase::FixedSimulation);
    CHECK(fixedOrder.size() == 2);
    CHECK(env.scheduler->RemoveSystem(fixedOrder[0]) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
    CHECK(env.Order(Phase::FixedSimulation).size() == 1 && env.Order(Phase::PreSimulation).empty());
}

// Structural work inside a phase (an exclusive system's immediate changes, the sync points after the steps of a domain
// that changes structure) stamps the running system's own version: a system after it in the phase that is not the
// phase's last sees each change exactly once, a system before it once at its next run.
static void test_in_phase_stamps()
{
    for (uint32_t workers : { 1u, 4u })
    {
        Env env(workers, true);
        Query<Added<A>> addedA(env.world.Get(), "added A");
        Query<Changed<C>> changedC(env.world.Get(), "changed C");
        Query<Added<Marker>> addedMarker(env.world.Get(), "added marker");
        Query<With<Marker>> markers(env.world.Get(), "markers");
        const Entity holder = env.Create({ env.c }, 1)[0];
        constexpr uint32_t c_Frames = 8;
        uint32_t frame = 0;
        uint32_t before[c_Frames] = {};
        uint32_t after[c_Frames] = {};
        uint32_t afterChanged[c_Frames] = {};
        uint32_t stepWatch[c_Frames] = {};
        auto count = [](const auto& query, ISystemContext* context)
        {
            uint32_t chunks = 0;
            query.ForEachChunk(context, [&chunks](const ChunkView&, const auto*) { ++chunks; });
            return chunks;
        };

        // PreSimulation: before -> ex (exclusive: creates A rows on frames 0 and 3, sets C on frame 5) -> after ->
        // tail, so that `after` is not the phase's last system.
        Add(env.scheduler.Get(), { "before", Phase::PreSimulation, { R(env.a) } },
            [&](ISystemContext* context) { before[frame] = count(addedA, context); });
        Add(env.scheduler.Get(),
            { "ex", Phase::PreSimulation, {}, {}, { "before", "b.gather" }, SystemFlags::Exclusive },
            [&](ISystemContext* context)
        {
            IWorld* world = context->GetWorld();
            if (frame == 0 || frame == 3)
            {
                const ComponentId list[1] = { env.a };
                EntityCreateDesc desc{};
                desc.components = list;
                desc.componentCount = 1;
                desc.entityCount = 2;
                Entity made[2];
                return world->CreateEntities(desc, made);
            }
            if (frame == 5)
                static_cast<C*>(world->GetComponentForWrite(holder, env.c, c_WorldVersion))->value = 5;
            return nvrhi::FS_OK;
        });
        Add(env.scheduler.Get(), { "after", Phase::PreSimulation, { R(env.a), R(env.c) }, {}, { "ex" } },
            [&](ISystemContext* context)
        {
            after[frame] = count(addedA, context);
            afterChanged[frame] = count(changedC, context);
        });
        Add(env.scheduler.Get(), { "tail", Phase::PreSimulation, { R(env.b) }, {}, { "after" } },
            [](ISystemContext*) { });

        // FixedSimulation: the step of a domain that changes structure (a Marker row per step, steps on frames 0 and
        // 3), then a watcher that is not the phase's last.
        nvrhi::AutoPtr<TestDomain> builder = MAKE_RC_OBJ_PTR(TestDomain);
        builder->desc.name = "b";
        builder->desc.fixedTimeStep = 1.0;
        builder->desc.maxStepsPerFrame = 1;
        builder->desc.flags = SimulationDomainFlags::ChangesStructure;
        builder->markers = markers.Get();
        builder->marker = env.marker;
        SimulationDomainId builderId;
        CHECK(env.scheduler->AddSimulationDomain(builder.Get(), builderId) == nvrhi::FS_OK);
        Add(env.scheduler.Get(), { "step_watch", Phase::FixedSimulation, { R(env.marker) }, {}, { "b.step" } },
            [&](ISystemContext* context) { stepWatch[frame] = count(addedMarker, context); });
        Add(env.scheduler.Get(), { "step_tail", Phase::FixedSimulation, { R(env.b) }, {}, { "step_watch" } },
            [](ISystemContext*) { });
        CHECK(env.scheduler->Build() == nvrhi::FS_OK && env.scheduler->GetAmbiguityCount() == 0);

        double time = 0.0;
        for (frame = 0; frame < c_Frames; ++frame)
        {
            const double delta = frame == 0 || frame == 3 ? 1.0 : 0.0;
            time += delta;
            env.scheduler->SetFrameTime(MakeFrame(time, delta, frame));
            CHECK(env.scheduler->RunPhases(Phase::PreSimulation, Phase::FixedSimulation) == nvrhi::FS_OK);
        }
        for (frame = 0; frame < c_Frames; ++frame)
        {
            // `after` sees the rows created on frames 0 and 3 in that frame only; `before` a frame later.
            CHECK(after[frame] == (frame == 0 || frame == 3 ? 1u : 0u));
            CHECK(before[frame] == (frame == 1 || frame == 4 ? 1u : 0u));
            // Frame 0 sees the holder's chunk as new; the exclusive write on frame 5 is seen once.
            CHECK(afterChanged[frame] == (frame == 0 || frame == 5 ? 1u : 0u));
            CHECK(stepWatch[frame] == (frame == 0 || frame == 3 ? 1u : 0u));
        }
        CHECK(builder->steps == 2 && builder->errors == 0 && markers.CountEntities() == 2);
        CHECK(env.Impl().GetValidationErrorCount() == 0);
    }
}

// ------------------------------------------------------------------------------------------- access checks

static void test_access_checks()
{
    Env env(4, true);
    env.Create({ env.a, env.b, env.c }, 12000);
    Query<Read<A>, Write<B>> readAWriteB(env.world.Get(), "readA writeB");
    Query<Read<A>, Read<C>> readAReadC(env.world.Get(), "readA readC");
    Query<Write<A>> writeA(env.world.Get(), "writeA");
    Query<Read<A>, With<C>> readAWithC(env.world.Get(), "readA withC");
    CHECK(readAWriteB.Get()->GetChunkCount() > 8);

    enum Case { Good, ReadUndeclared, WriteDeclaredRead, FilterOnly, Exclusive, NoSystem, Count };
    SystemId ids[Count];
    ids[Good] = Add(env.scheduler.Get(), { "good", Phase::PreSimulation, { R(env.a), W(env.b) } },
                    [&](ISystemContext* context)
    {
        readAWriteB.ParallelForChunks(context, 1, [](const ChunkView&, const A*, B*) { });
    });
    ids[ReadUndeclared] = Add(env.scheduler.Get(), { "read_undeclared", Phase::PreSimulation, { R(env.a) } },
                              [&](ISystemContext* context)
    {
        readAReadC.ParallelForChunks(context, 1, [](const ChunkView&, const A*, const C*) { });
    });
    ids[WriteDeclaredRead] = Add(env.scheduler.Get(), { "write_declared_read", Phase::PreSimulation, { R(env.a) } },
                                 [&](ISystemContext* context)
    {
        writeA.ForEachChunk(context, [](const ChunkView&, A*) { });
    });
    ids[FilterOnly] = Add(env.scheduler.Get(), { "filter", Phase::PreSimulation, { R(env.a), Filter(env.c) } },
                          [&](ISystemContext* context)
    {
        readAWithC.ForEachChunk(context, [](const ChunkView&, const A*) { });
    });
    ids[Exclusive] = Add(env.scheduler.Get(), { "exclusive", Phase::PreSimulation, {}, {}, {}, SystemFlags::Exclusive },
                         [&](ISystemContext* context)
    {
        writeA.ForEachChunk(context, [](const ChunkView&, A*) { });
    });
    ids[NoSystem] = Add(env.scheduler.Get(), { "no_system", Phase::PreSimulation, { W(env.a) } },
                        [&](ISystemContext* context)
    {
        // Without ChunkRequest::system nothing is checked.
        ChunkRequest request{ context->GetLastRunVersion(), context->GetVersion(), nullptr };
        readAReadC.ForEachChunk(request, [](const ChunkView&, const A*, const C*) { });
    });
    const uint64_t expected[Count] = { 0, 1, 1, 0, 0, 0 };
    for (uint32_t which = 0; which < Count; ++which)
    {
        for (uint32_t i = 0; i < Count; ++i)
            CHECK(env.scheduler->SetSystemEnabled(ids[i], i == which) == nvrhi::FS_OK);
        for (uint32_t run = 0; run < 3; ++run)
        {
            const uint64_t before = env.Impl().GetValidationErrorCount();
            CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
            // Once per query and run, however many chunk batches the system asked for.
            CHECK(env.Impl().GetValidationErrorCount() - before == expected[which]);
        }
    }
    for (uint32_t i = 0; i < Count; ++i)
        CHECK(env.scheduler->RemoveSystem(ids[i]) == nvrhi::FS_OK);

    // Writes through GetComponentForWrite and ResizeBuffer: the write version names the system run (one report per
    // call). c_WorldVersion is for writes outside phases. Exclusive systems write anything.
    const ComponentId elems = RegisterComponent<BufferOf<Elem>>(env.world.Get());
    const Entity target = env.Create({ env.a, elems }, 1)[0];
    enum WriteCase { WriteDeclared, WriteUndeclared, WriteReadOnly, WorldVersion, Resize, ResizeDeclared,
                     ExclusiveWrite, WriteCount };
    SystemId writers[WriteCount];
    auto writeTarget = [&env, target](ISystemContext* context)
    {
        if (!context->GetWorld()->GetComponentForWrite(target, env.a, context->GetVersion()))
            return nvrhi::FE_GENERIC_ERROR;
        return nvrhi::FS_OK;
    };
    writers[WriteDeclared] = Add(env.scheduler.Get(), { "w_declared", Phase::Bounds, { W(env.a) } }, writeTarget);
    writers[WriteUndeclared] = Add(env.scheduler.Get(), { "w_undeclared", Phase::Bounds }, writeTarget);
    writers[WriteReadOnly] = Add(env.scheduler.Get(), { "w_read_only", Phase::Bounds, { R(env.a) } }, writeTarget);
    writers[WorldVersion] = Add(env.scheduler.Get(), { "w_world_version", Phase::Bounds, { W(env.a) } },
                                [&env, target](ISystemContext* context)
    {
        context->GetWorld()->GetComponentForWrite(target, env.a, c_WorldVersion);
    });
    auto resize = [&env, target, elems](ISystemContext* context)
    {
        return context->GetWorld()->ResizeBuffer(target, elems, 1, context->GetVersion());
    };
    writers[Resize] = Add(env.scheduler.Get(), { "resize_read_only", Phase::Bounds, { R(elems) } }, resize);
    writers[ResizeDeclared] = Add(env.scheduler.Get(), { "resize_declared", Phase::Bounds, { W(elems) } }, resize);
    writers[ExclusiveWrite] = Add(env.scheduler.Get(), { "w_exclusive", Phase::Bounds, {}, {}, {},
                                  SystemFlags::Exclusive }, writeTarget);
    const uint64_t expectedWrites[WriteCount] = { 0, 1, 1, 1, 1, 0, 0 };
    for (uint32_t which = 0; which < WriteCount; ++which)
    {
        for (uint32_t i = 0; i < WriteCount; ++i)
            CHECK(env.scheduler->SetSystemEnabled(writers[i], i == which) == nvrhi::FS_OK);
        for (uint32_t run = 0; run < 2; ++run)
        {
            const uint64_t before = env.Impl().GetValidationErrorCount();
            CHECK(env.scheduler->RunPhase(Phase::Bounds) == nvrhi::FS_OK);
            CHECK(env.Impl().GetValidationErrorCount() - before == expectedWrites[which]);
        }
    }
    // Outside phases nothing is checked.
    const uint64_t outside = env.Impl().GetValidationErrorCount();
    CHECK(env.world->GetComponentForWrite(target, env.a, c_WorldVersion) != nullptr);
    CHECK(env.world->ResizeBuffer(target, elems, 0, c_WorldVersion) == nvrhi::FS_OK);
    CHECK(env.Impl().GetValidationErrorCount() == outside);
    std::string text = Diagnostics(env.scheduler.Get());
    CHECK(text.find("failed updates") == std::string::npos);

    // Without validation nothing is reported.
    Env quiet(2, false);
    quiet.Create({ quiet.a, quiet.c }, 100);
    Query<Read<A>, Read<C>> quietQuery(quiet.world.Get());
    Add(quiet.scheduler.Get(), { "undeclared", Phase::PreSimulation },
        [&](ISystemContext* context)
    {
        quietQuery.ForEachChunk(context, [](const ChunkView&, const A*, const C*) { });
    });
    CHECK(quiet.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
    CHECK(quiet.Impl().GetValidationErrorCount() == 0);
}

// ------------------------------------------------------------------------------------- job table exhausted

static void test_job_table_full()
{
    for (uint32_t workers : { 1u, 4u })
    {
        Env env(workers, true);
        Trace trace;
        enum { S0, S1, S2, S3, Main, N };
        Add(env.scheduler.Get(), { "s0", Phase::PreSimulation, { W(env.a) } },
            [&](ISystemContext*) { trace.Begin(S0); trace.End(S0); });
        Add(env.scheduler.Get(), { "s1", Phase::PreSimulation, { W(env.a) }, {}, { "s0" } },
            [&](ISystemContext*) { trace.Begin(S1); trace.End(S1); });
        Add(env.scheduler.Get(), { "s2", Phase::PreSimulation, { W(env.b) } },
            [&](ISystemContext*) { trace.Begin(S2); trace.End(S2); });
        Add(env.scheduler.Get(), { "s3", Phase::PreSimulation, { R(env.b) }, {}, { "s2" } },
            [&](ISystemContext*) { trace.Begin(S3); trace.End(S3); });
        Add(env.scheduler.Get(), { "main", Phase::PreSimulation, { R(env.a) }, {}, { "s1" }, SystemFlags::MainThread },
            [&](ISystemContext*) { trace.Begin(Main); trace.End(Main); });

        // A gate job that holds a worker, and jobs behind it until the table is full. The gate has started before the
        // calling thread waits for anything, so that it never picks the gate up itself.
        struct Gate
        {
            std::atomic<bool> started{ false };
            std::atomic<bool> release{ false };
        } gateState;
        std::atomic<bool>& release = gateState.release;
        auto gate = [](void* context, uint32_t, uint32_t, uint32_t)
        {
            Gate& state = *static_cast<Gate*>(context);
            state.started = true;
            while (!state.release.load())
                std::this_thread::yield();
        };
        JobHandle gateJob = JobHandle::Null;
        CHECK(env.jobs->Schedule(JobDesc{ gate, &gateState, 1, 1, "gate" }, nullptr, 0, gateJob) == nvrhi::FS_OK);
        while (!gateState.started.load())
            std::this_thread::yield();
        std::vector<JobHandle> held;
        for (uint32_t attempt = 0; attempt < 3; ++attempt)
        {
            for (;;)
            {
                JobHandle job = JobHandle::Null;
                if (NVRHI_FAILED(env.jobs->Schedule(JobDesc{ nullptr, nullptr, 0, 0, "held" }, &gateJob, 1, job)))
                    break;
                held.push_back(job);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        CHECK(held.size() > 4000);

        trace.Reset();
        const ecs::SchedulerCounters before = env.Counters();
        CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
        const ecs::SchedulerCounters after = env.Counters();
        for (uint32_t i = 0; i < N; ++i)
            CHECK(trace.runs[i] == 1);
        CHECK(trace.Before(S0, S1) && trace.Before(S1, Main) && trace.Before(S2, S3));
        CHECK(after.inlineRuns - before.inlineRuns + after.jobsScheduled - before.jobsScheduled == 4);
        CHECK(after.inlineRuns > before.inlineRuns);

        release = true;
        env.jobs->Wait(gateJob);
        for (JobHandle job : held)
            env.jobs->Wait(job);
        trace.Reset();
        CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
        CHECK(env.Counters().inlineRuns == after.inlineRuns);
        CHECK(trace.Before(S0, S1) && trace.Before(S1, Main) && trace.Before(S2, S3));
    }
}

// --------------------------------------------------------------------------------------------- the API

static void test_api()
{
    Env env(2, false);
    auto nothing = [](ISystemContext*) { };

    // Creation.
    nvrhi::AutoPtr<IScheduler> other;
    CHECK(CreateScheduler(nullptr, env.jobs.Get(), &other) == nvrhi::FE_INVALID_ARGS);
    CHECK(CreateScheduler(env.world.Get(), nullptr, &other) == nvrhi::FE_INVALID_ARGS);
    CHECK(CreateScheduler(env.world.Get(), env.jobs.Get(), nullptr) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->GetWorld() == env.world.Get() && env.scheduler->GetJobSystem() == env.jobs.Get());
    CHECK(GetResource<FrameTime>(env.world.Get()) != nullptr);

    // Named phases: right after `after`, before what ran after it until now; values from Phase::Count up.
    Phase late = Phase::BeginFrame;
    Phase early = Phase::BeginFrame;
    CHECK(env.scheduler->AddPhase("late", Phase::PreSimulation, late) == nvrhi::FS_OK && late == Phase::Count);
    CHECK(env.scheduler->AddPhase("early", Phase::PreSimulation, early) == nvrhi::FS_OK &&
          uint32_t(early) == uint32_t(Phase::Count) + 1);
    Phase bad = Phase::BeginFrame;
    CHECK(env.scheduler->AddPhase("early", Phase::BeginFrame, bad) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->AddPhase("Bounds", Phase::BeginFrame, bad) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->AddPhase(nullptr, Phase::BeginFrame, bad) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->AddPhase("x", Phase(99), bad) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->AddPhase("", Phase::BeginFrame, bad) == nvrhi::FE_INVALID_ARGS);
    std::vector<std::string> ran;
    const Phase phases[] = { Phase::BeginFrame, Phase::PreSimulation, early, late, Phase::FixedSimulation,
                             Phase::PostSimulation, Phase::TransformPropagation, Phase::Bounds, Phase::RenderExtract,
                             Phase::EndFrame };
    const char* const names[] = { "BeginFrame", "PreSimulation", "early", "late", "FixedSimulation", "PostSimulation",
                                  "TransformPropagation", "Bounds", "RenderExtract", "EndFrame" };
    for (uint32_t i = 10; i-- > 0;)
    {
        const std::string name = names[i];
        Add(env.scheduler.Get(), { "in_" + name, phases[i] }, [&ran, name](ISystemContext*) { ran.push_back(name); });
    }
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
    CHECK(ran.size() == 10);
    for (uint32_t i = 0; i < 10; ++i)
        CHECK(ran[i] == names[i]);
    ran.clear();
    CHECK(env.scheduler->RunPhases(early, Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(ran.size() == 3 && ran[0] == "early" && ran[2] == "FixedSimulation");
    CHECK(env.scheduler->RunPhases(Phase::FixedSimulation, early) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->RunPhase(Phase(77)) == nvrhi::FE_INVALID_ARGS);

    // AddSystem arguments.
    Add(env.scheduler.Get(), { "in_EndFrame", Phase::EndFrame }, nothing, nvrhi::FE_INVALID_ARGS);   // name taken
    Add(env.scheduler.Get(), { "", Phase::EndFrame }, nothing, nvrhi::FE_INVALID_ARGS);
    Add(env.scheduler.Get(), { "nowhere", Phase(55) }, nothing, nvrhi::FE_INVALID_ARGS);
    Add(env.scheduler.Get(), { "unregistered", Phase::EndFrame, { R(ComponentId(9999)) } }, nothing,
        nvrhi::FE_INVALID_ARGS);
    Add(env.scheduler.Get(), { "bad_access", Phase::EndFrame, { ComponentAccess{ env.a, Access(7), {} } } }, nothing,
        nvrhi::FE_INVALID_ARGS);
    Add(env.scheduler.Get(), { "bad_flags", Phase::EndFrame, {}, {}, {}, SystemFlags(0x100) }, nothing,
        nvrhi::FE_INVALID_ARGS);
    SystemId none = SystemId(5);
    CHECK(env.scheduler->AddSystem(nullptr, none) == nvrhi::FE_INVALID_ARGS && none == SystemId::Invalid);
    CHECK(env.scheduler->RemoveSystem(SystemId(999)) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->SetSystemEnabled(SystemId::Invalid, true) == nvrhi::FE_INVALID_ARGS);
    SystemStats stats{};
    CHECK(env.scheduler->GetSystemStats(SystemId(999), stats) == nvrhi::FE_INVALID_ARGS);

    // Accesses are merged per component, the strongest kept: Read then Write on A conflicts with a reader of A.
    Add(env.scheduler.Get(), { "merged", Phase::TransformPropagation, { R(env.a), W(env.a), Filter(env.a) } }, nothing);
    Add(env.scheduler.Get(), { "reader", Phase::TransformPropagation, { R(env.a) } }, nothing);
    CHECK(env.scheduler->GetAmbiguityCount() == 1);

    // StartDisabled, SetSystemEnabled; failures are recorded and the phase goes on; ids are reused.
    uint32_t disabledRuns = 0;
    const SystemId disabled = Add(env.scheduler.Get(),
                                  { "disabled", Phase::Bounds, {}, {}, {}, SystemFlags::StartDisabled },
                                  [&](ISystemContext*) { ++disabledRuns; });
    uint32_t failingRuns = 0;
    const SystemId failing = Add(env.scheduler.Get(), { "failing", Phase::Bounds },
                                 [&](ISystemContext*) { ++failingRuns; return nvrhi::FE_GENERIC_ERROR; });
    CHECK(env.scheduler->RunPhase(Phase::Bounds) == nvrhi::FS_OK);
    CHECK(disabledRuns == 0 && failingRuns == 1);
    CHECK(env.scheduler->SetSystemEnabled(disabled, true) == nvrhi::FS_OK);
    CHECK(env.scheduler->RunPhase(Phase::Bounds) == nvrhi::FS_OK);
    CHECK(disabledRuns == 1 && failingRuns == 2);
    CHECK(Diagnostics(env.scheduler.Get()).find("system 'failing': 2 failed updates") != std::string::npos);
    CHECK(env.scheduler->GetSystemStats(failing, stats) == nvrhi::FS_OK && stats.runCount == 2);
    CHECK(env.scheduler->RemoveSystem(failing) == nvrhi::FS_OK);
    CHECK(env.scheduler->GetSystemStats(failing, stats) == nvrhi::FE_INVALID_ARGS);
    const SystemId reused = Add(env.scheduler.Get(), { "failing", Phase::Bounds }, nothing);
    CHECK(reused == failing);

    // Re-entrancy: a system cannot run phases or change the scheduler.
    std::vector<nvrhi::FRESULT> results;
    Add(env.scheduler.Get(), { "reentrant", Phase::Bounds }, [&](ISystemContext*)
    {
        SystemId id;
        Phase phase;
        results.push_back(env.scheduler->RunPhase(Phase::BeginFrame));
        results.push_back(env.scheduler->AddSystem(nullptr, id));
        results.push_back(env.scheduler->SetSystemEnabled(disabled, false));
        results.push_back(env.scheduler->RemoveSystem(disabled));
        results.push_back(env.scheduler->AddPhase("inner", Phase::BeginFrame, phase));
        results.push_back(env.scheduler->Build());
        env.scheduler->SetFrameTime(MakeFrame(100.0, 1.0, 100));
        // Statistics and diagnostics are written by the threads that run systems: read between phases only.
        SystemStats inner{};
        results.push_back(env.scheduler->GetSystemStats(disabled, inner));
        nvrhi::string text("stale");
        env.scheduler->GetDiagnostics(text);
        results.push_back(text.empty() ? nvrhi::FE_INVALID_ARGS : nvrhi::FS_OK);
    });
    CHECK(env.scheduler->RunPhase(Phase::Bounds) == nvrhi::FS_OK);
    CHECK(results.size() == 8);
    for (nvrhi::FRESULT result : results)
        CHECK(result == nvrhi::FE_INVALID_ARGS);
    CHECK(GetResource<FrameTime>(env.world.Get())->frameIndex != 100);
    CHECK(disabledRuns == 2);
    CHECK(env.scheduler->GetSystemStats(disabled, stats) == nvrhi::FS_OK && stats.runCount == 2);
}

// ----------------------------------------------------------------------------------------- steady state

static void test_steady_state_allocations()
{
    Env env(4, true);
    std::vector<Entity> entities = env.Create({ env.a, env.b, env.c }, 3000);
    env.Create({ env.d }, 8);   // the archetype that the spawner fills
    Query<Write<A>, Read<B>> move(env.world.Get());
    Query<Changed<A>> changed(env.world.Get());
    Query<Read<D>> spawned(env.world.Get());
    Query<Read<C>> sources(env.world.Get());
    Query<Read<A>> readA(env.world.Get());

    nvrhi::AutoPtr<TestDomain> domain = MAKE_RC_OBJ_PTR(TestDomain);
    domain->desc.name = "dom";
    domain->desc.fixedTimeStep = 1.0 / 64;
    domain->desc.maxStepsPerFrame = 4;
    SimulationDomainId domainId;
    CHECK(env.scheduler->AddSimulationDomain(domain.Get(), domainId) == nvrhi::FS_OK);

    std::atomic<uint64_t> sink{ 0 };
    Add(env.scheduler.Get(), { "move", Phase::PreSimulation, { W(env.a), R(env.b) } }, [&](ISystemContext* context)
    {
        move.ParallelForChunks(context, 2, [](const ChunkView& view, A* a, const B* b)
        {
            for (uint32_t row = 0; row < view.count; ++row)
                a[row].value += b[row].value + 1.0f;
        });
    });
    Add(env.scheduler.Get(), { "watch", Phase::PreSimulation, { R(env.a) }, {}, { "move" } },
        [&](ISystemContext* context)
    {
        changed.ForEachChunk(context, [&sink](const ChunkView& view, const A*) { sink += view.count; });
    });
    Add(env.scheduler.Get(), { "main", Phase::PreSimulation, { R(env.counter) }, {}, {}, SystemFlags::MainThread },
        [&](ISystemContext* context) { sink += context->GetFrameTime().frameIndex; });
    const SystemId spawn = Add(env.scheduler.Get(), { "spawn", Phase::PostSimulation, { R(env.c) } },
                               [&](ISystemContext* context)
    {
        sources.ParallelForChunks(context, 1, [&](const ChunkView& view, const C*)
        {
            const ComponentId list[1] = { env.d };
            EntityCreateDesc desc{};
            desc.components = list;
            desc.componentCount = 1;
            desc.entityCount = 4;
            Entity placeholders[4];
            context->GetCommandBuffer()->CreateEntities(desc, MakeSortKey(0, view.chunkIndex, 0), placeholders);
        });
    });
    Add(env.scheduler.Get(), { "despawn", Phase::RenderExtract, { R(env.d) } }, [&](ISystemContext* context)
    {
        spawned.ForEachChunk(context, [&](const ChunkView& view, const D*)
        {
            context->GetCommandBuffer()->DestroyEntities(view.entities, view.count, MakeSortKey(0, view.chunkIndex, 0));
        });
    });
    Add(env.scheduler.Get(), { "exclusive", Phase::EndFrame, {}, {}, {}, SystemFlags::Exclusive },
        [&](ISystemContext* context)
    {
        // An immediate structural change in an existing archetype, and back.
        const ComponentId list[1] = { env.d };
        EntityCreateDesc desc{};
        desc.components = list;
        desc.componentCount = 1;
        desc.entityCount = 2;
        Entity made[2];
        if (context->GetWorld()->CreateEntities(desc, made) == nvrhi::FS_OK)
            context->GetWorld()->DestroyEntities(made, 2);
    });
    Add(env.scheduler.Get(), { "bounds", Phase::Bounds, { R(env.a), W(env.e) } }, [&](ISystemContext* context)
    {
        readA.ForEachChunk(context, [&sink](const ChunkView& view, const A*) { sink += view.count; });
    });

    // Which worker records a chunk of `spawn` varies from frame to frame, and a buffer takes its first arena page and
    // command capacity when it first records: warm every buffer of the parallel recorder, as enough frames would.
    // The single-threaded recorders use their home buffer, whichever worker runs them, so warm-up frames suffice.
    ICommandBuffer* buffers[64] = {};
    const uint32_t bufferCount = ecs::GetSystemCommandBuffers(env.scheduler.Get(), spawn, buffers, 64);
    CHECK(bufferCount == env.jobs->GetThreadCount() + 1);
    for (uint32_t i = 0; i < bufferCount; ++i)
    {
        for (uint32_t k = 0; k < 16; ++k)
            buffers[i]->DestroyEntity(Entity::Null, k);
        buffers[i]->Reset();
    }

    double time = 0.0;
    auto frame = [&](uint64_t index)
    {
        time += 1.0 / 60;
        env.scheduler->SetFrameTime(FrameTime{ time, 1.0 / 60, index });
        CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
    };
    for (uint64_t i = 0; i < 3; ++i)
        frame(i);
    const uint64_t news = g_NewCount.load();
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount();
    for (uint64_t i = 3; i < 13; ++i)
        frame(i);
    const uint64_t newsAfter = g_NewCount.load();
    const uint64_t heapBlocksAfter = ecs::GetHeapAllocationCount();
    printf("test_ecs_scheduler: steady state: %llu operator new, %llu storage blocks in 10 frames of 10 systems "
           "(sink %llu)\n", (unsigned long long)(newsAfter - news), (unsigned long long)(heapBlocksAfter - heapBlocks),
           (unsigned long long)(sink.load() & 1));
    CHECK(newsAfter == news);
    CHECK(heapBlocksAfter == heapBlocks);
    CHECK(domain->steps > 0 && env.Impl().GetValidationErrorCount() == 0);
    // The counters see allocations when there are some.
    int* probe = new int[16];
    sink += uintptr_t(probe) & 1;
    delete[] probe;
    CHECK(g_NewCount.load() > newsAfter);
}

int main(int, char**)
{
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "ordering", &test_ordering },
        { "random_graphs", &test_random_graphs },
        { "ambiguities_and_cycles", &test_ambiguities_and_cycles },
        { "exclusive_and_main_thread", &test_exclusive_and_main_thread },
        { "versions_and_changed", &test_versions_and_changed },
        { "playback_order", &test_playback_order },
        { "fixed_simulation", &test_fixed_simulation },
        { "in_phase_stamps", &test_in_phase_stamps },
        { "access_checks", &test_access_checks },
        { "job_table_full", &test_job_table_full },
        { "api", &test_api },
        { "steady_state_allocations", &test_steady_state_allocations },
    };
    (void)c_PhaseNames;
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    try
    {
        for (const Test& test : tests)
        {
            test.run();
            printf("test_ecs_scheduler: %s passed\n", test.name);
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);   // every world, buffer and scheduler freed every block
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
