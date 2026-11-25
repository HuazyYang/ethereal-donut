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

// The job system (aggregate docs/plans/2026-11-21-donut-ecs.md, step E2.1; design 6.4, D10), through its public API
// only: worker indices and external slots, ParallelFor's deterministic split, dependency graphs (chains, fans,
// diamonds) in a valid order with every job and every index run exactly once, nested waits, more external threads than
// slots, a full job table and edge pool, shutdown with pending work, zero allocations in steady state, a seeded stress
// run at 1, 2 and 8 workers with jobs scheduled from inside jobs, a Wait for a job lower on the same stack, a Wait
// blocked on an external slot, overflowing deques, and idle workers that soon sleep, wake again, and look for work again
// before their next sleep. After each test the job table and edge pool must be whole again. A watchdog ends the process
// if a case hangs; the deadlock cases have a 20 s guard of their own.

#include <donut/core/log.h>
#include <donut/engine/JobSystem.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <ctime>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test, and donut_engine, which links statically) is counted. The job
// system allocates nothing through nvrhi's allocator after CreateJobSystem, so this counter sees all of its blocks.
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

// ------------------------------------------------------------------------------------------------- helpers

using donut::engine::CreateJobSystem;
using donut::engine::IJobSystem;
using donut::engine::JobDesc;
using donut::engine::JobHandle;
using donut::engine::JobSystemDesc;

namespace
{
    constexpr uint32_t c_JobCapacity = 4096;     // JobSystem.h, Schedule
    constexpr uint32_t c_EdgeCapacity = 16384;
    constexpr uint32_t c_NoWorker = ~0u;

    // A job function must not throw: checks inside jobs count failures and keep the first line, and the test checks
    // the count on its own thread.
    std::atomic<uint32_t> g_JobFailures{ 0 };
    std::atomic<int> g_FirstJobFailureLine{ 0 };

#define JOB_CHECK(condition)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition) && g_JobFailures.fetch_add(1) == 0)                                                           \
            g_FirstJobFailureLine.store(__LINE__);                                                                     \
    } while (false)

    void CheckJobs(const char* where)
    {
        const uint32_t failures = g_JobFailures.exchange(0);
        if (failures != 0)
            throw std::runtime_error(std::string(where) + ": " + std::to_string(failures) +
                                     " failed checks inside jobs, the first at line " +
                                     std::to_string(g_FirstJobFailureLine.load()));
    }

    // xorshift64* with explicit seeds (the plan's determinism rule: no std:: distributions).
    struct Random
    {
        uint64_t state;
        explicit Random(uint64_t seed) : state(seed * 0x9e3779b97f4a7c15ull + 0x1234567ull) {}
        uint64_t Next()
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 0x2545f4914f6cdd1dull;
        }
        uint32_t Below(uint32_t bound) { return bound ? uint32_t((Next() >> 32) % bound) : 0; }
    };

    uint32_t RangeCount(uint32_t count, uint32_t grainSize)
    {
        const uint32_t grain = grainSize ? grainSize : count;
        return count ? (count - 1) / grain + 1 : 0;
    }

    nvrhi::AutoPtr<IJobSystem> CreateSystem(uint32_t workers, uint32_t externalThreads = 2, uint32_t spinCount = 4096,
                                            uint32_t idleSpinMicroseconds = 30)
    {
        JobSystemDesc desc;
        desc.debugName = "test_job_system";
        desc.workerThreadCount = workers;
        desc.maxExternalThreads = externalThreads;
        desc.spinCount = spinCount;
        desc.idleSpinMicroseconds = idleSpinMicroseconds;
        IJobSystem* system = nullptr;
        CHECK(CreateJobSystem(desc, &system) == nvrhi::FS_OK && system != nullptr);
        return nvrhi::TakeOver(system);
    }

    JobHandle Schedule(IJobSystem* jobs, const JobDesc& desc, const JobHandle* dependencies = nullptr,
                       uint32_t dependencyCount = 0)
    {
        JobHandle handle = JobHandle::Null;
        CHECK(jobs->Schedule(desc, dependencies, dependencyCount, handle) == nvrhi::FS_OK && handle != JobHandle::Null);
        return handle;
    }

    void Spin(uint32_t iterations)
    {
        volatile uint32_t sink = 0;
        for (uint32_t i = 0; i < iterations; ++i)
            sink = sink + i;
    }

    // A job that holds back its dependents until the test opens it.
    struct Gate
    {
        std::atomic<bool> open{ false };

        static void Run(void* context, uint32_t, uint32_t, uint32_t)
        {
            const Gate* gate = static_cast<const Gate*>(context);
            while (!gate->open.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
        JobDesc Desc() { return JobDesc{ &Run, this, 1, 0, "gate" }; }
    };

    struct Counter
    {
        std::atomic<uint64_t> value{ 0 };

        static void Run(void* context, uint32_t begin, uint32_t end, uint32_t)
        {
            static_cast<Counter*>(context)->value.fetch_add(end - begin, std::memory_order_relaxed);
        }
    };

    void AtomicMin(std::atomic<uint64_t>& target, uint64_t value)
    {
        uint64_t current = target.load(std::memory_order_relaxed);
        while (value < current && !target.compare_exchange_weak(current, value, std::memory_order_relaxed))
        {
        }
    }

    void AtomicMax(std::atomic<uint64_t>& target, uint64_t value)
    {
        uint64_t current = target.load(std::memory_order_relaxed);
        while (value > current && !target.compare_exchange_weak(current, value, std::memory_order_relaxed))
        {
        }
    }

    // Runs `body` on a thread of its own; if it has not returned after `seconds`, the test fails (exit code 4) instead
    // of hanging. An exception from `body` is rethrown.
    template <typename F>
    void RunGuarded(const char* what, uint32_t seconds, F&& body)
    {
        std::atomic<bool> done{ false };
        std::exception_ptr failure;
        std::thread thread([&] {
            try
            {
                body();
            }
            catch (...)
            {
                failure = std::current_exception();
            }
            done.store(true);
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (!done.load())
        {
            if (std::chrono::steady_clock::now() > deadline)
            {
                std::fflush(stdout);
                std::fprintf(stderr, "test_job_system: %s did not finish in %u s (a deadlock)\n", what, seconds);
                std::fflush(stderr);
                std::_Exit(4);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        thread.join();
        if (failure)
            std::rethrow_exception(failure);
    }

    // Fills the job table behind a gate with jobs that count into `counter` until Schedule fails, and returns the gate
    // job; `handles` gets the jobs. Every slot must be taken: the gate's and c_JobCapacity - 1 jobs. Right after the
    // last Wait of a test, a few slots can still be held legitimately: Wait returns once a job is marked finished, a
    // moment before the thread that finished it drops the job's own reference, and extra tickets of a job whose ranges
    // were all claimed by others are given back when an idle thread next pops or steals them (microseconds). So a fill
    // that comes up short opens the gate, waits, pauses and tries again; a lost slot fails all 20 attempts.
    JobHandle FillTable(IJobSystem* jobs, Gate& gate, Counter& counter, std::vector<JobHandle>& handles)
    {
        const JobDesc filler{ &Counter::Run, &counter, 1, 0, "fill" };
        for (uint32_t attempt = 0;; ++attempt)
        {
            gate.open.store(false);
            counter.value.store(0);
            handles.clear();
            const JobHandle gateJob = Schedule(jobs, gate.Desc());
            for (;;)
            {
                JobHandle handle = JobHandle(77);
                const nvrhi::FRESULT result = jobs->Schedule(filler, &gateJob, 1, handle);
                if (result != nvrhi::FS_OK)
                {
                    CHECK(result == nvrhi::FE_OUT_OF_MEMORY && handle == JobHandle::Null);
                    break;
                }
                CHECK(handle != JobHandle::Null && handles.size() < c_JobCapacity);
                handles.push_back(handle);
            }
            if (handles.size() + 1 == c_JobCapacity)
                return gateJob;
            gate.open.store(true);
            for (JobHandle handle : handles)
                jobs->Wait(handle);
            jobs->Wait(gateJob);
            if (attempt == 20)   // nothing left pending, so the failure does not hang the system's Release
                throw std::runtime_error("the job table took a gate and " + std::to_string(handles.size()) +
                                         " jobs, not " + std::to_string(c_JobCapacity) + " in all: a slot is lost");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // The whole edge pool is free: one job depends c_EdgeCapacity times on a pending gate. A finished job's edges go
    // back at the end of its completion, a moment after its dependents became runnable, so a failure is retried after
    // a pause; a lost edge fails all 20 attempts.
    void CheckAllEdges(IJobSystem* jobs)
    {
        for (uint32_t attempt = 0;; ++attempt)
        {
            Gate gate;
            const JobHandle gateJob = Schedule(jobs, gate.Desc());
            const std::vector<JobHandle> dependencies(c_EdgeCapacity, gateJob);
            Counter counter;
            JobHandle all = JobHandle::Null;
            const JobDesc desc{ &Counter::Run, &counter, 1, 0, "all edges" };
            const nvrhi::FRESULT result = jobs->Schedule(desc, dependencies.data(), c_EdgeCapacity, all);
            gate.open.store(true);
            jobs->Wait(gateJob);
            if (result == nvrhi::FS_OK)
            {
                jobs->Wait(all);
                CHECK(counter.value.load() == 1);
                return;
            }
            CHECK(result == nvrhi::FE_OUT_OF_MEMORY && all == JobHandle::Null);
            if (attempt == 20)
                throw std::runtime_error("the edge pool is not whole: an edge is lost");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // After a test: the job table and the edge pool are whole again.
    void CheckFullCapacity(IJobSystem* jobs)
    {
        Gate gate;
        Counter counter;
        std::vector<JobHandle> handles;
        const JobHandle gateJob = FillTable(jobs, gate, counter, handles);
        gate.open.store(true);
        for (JobHandle handle : handles)
            jobs->Wait(handle);
        jobs->Wait(gateJob);
        CHECK(counter.value.load() == handles.size());
        CheckAllEdges(jobs);
    }

    // --------------------------------------------------------------------------- dependency graphs with a check

    std::atomic<uint64_t> g_Clock{ 1 };

    struct NodeState
    {
        std::atomic<uint64_t> firstStart{ ~0ull };
        std::atomic<uint64_t> lastEnd{ 0 };
        std::atomic<uint32_t> ranges{ 0 };
        std::atomic<uint64_t> nestedIndices{ 0 };
        std::atomic<uint64_t> childIndices{ 0 };
        std::atomic<uint8_t>* visits = nullptr;   // per index of the node, when the graph counts them
        IJobSystem* jobs = nullptr;
        uint32_t nestedCount = 0;                 // a ParallelFor of this many indices inside each range
        uint32_t work = 0;
        uint32_t childJobs = 0;                   // jobs scheduled and waited for inside each range
        uint32_t childCount = 0;
        uint32_t childGrain = 0;

        static constexpr uint32_t c_MaxChildJobs = 4;

        void Reset()
        {
            firstStart.store(~0ull, std::memory_order_relaxed);
            lastEnd.store(0, std::memory_order_relaxed);
            ranges.store(0, std::memory_order_relaxed);
            nestedIndices.store(0, std::memory_order_relaxed);
            childIndices.store(0, std::memory_order_relaxed);
        }

        static void Nested(void* context, uint32_t begin, uint32_t end, uint32_t)
        {
            static_cast<NodeState*>(context)->nestedIndices.fetch_add(end - begin, std::memory_order_relaxed);
        }

        static void Child(void* context, uint32_t begin, uint32_t end, uint32_t)
        {
            static_cast<NodeState*>(context)->childIndices.fetch_add(end - begin, std::memory_order_relaxed);
        }

        // Jobs scheduled from inside a job (onto this thread's deque), every second one after the one before, then
        // waited for: jobs a job scheduled itself are the ones it may wait for (JobSystem.h).
        void RunChildren()
        {
            JobHandle children[c_MaxChildJobs] = {};
            for (uint32_t k = 0; k < childJobs && k < c_MaxChildJobs; ++k)
            {
                const JobHandle dependency = k % 2 == 1 ? children[k - 1] : JobHandle::Null;
                const JobDesc desc{ &Child, this, childCount, childGrain, "child" };
                JOB_CHECK(jobs->Schedule(desc, &dependency, 1, children[k]) == nvrhi::FS_OK);
            }
            for (uint32_t k = c_MaxChildJobs; k-- > 0;)
                jobs->Wait(children[k]);
        }

        static void Run(void* context, uint32_t begin, uint32_t end, uint32_t workerIndex)
        {
            NodeState* node = static_cast<NodeState*>(context);
            AtomicMin(node->firstStart, g_Clock.fetch_add(1, std::memory_order_relaxed));
            JOB_CHECK(begin < end);
            JOB_CHECK(node->jobs == nullptr || workerIndex < node->jobs->GetThreadCount());
            if (node->visits)
            {
                for (uint32_t i = begin; i < end; ++i)
                    node->visits[i].fetch_add(1, std::memory_order_relaxed);
            }
            Spin(node->work);
            if (node->nestedCount != 0)
                node->jobs->ParallelFor(node->nestedCount, 3, &Nested, node);
            if (node->childJobs != 0)
                node->RunChildren();
            node->ranges.fetch_add(1, std::memory_order_relaxed);
            AtomicMax(node->lastEnd, g_Clock.fetch_add(1, std::memory_order_relaxed));
        }
    };

    struct GraphNode
    {
        uint32_t count = 1;
        uint32_t grainSize = 0;
        uint32_t nestedCount = 0;
        uint32_t work = 0;
        uint32_t childJobs = 0;               // NodeState::RunChildren
        uint32_t childCount = 0;
        uint32_t childGrain = 0;
        std::vector<uint32_t> dependencies;   // earlier nodes
        uint32_t nullDependencies = 0;        // Null handles among the dependencies
        uint32_t staleDependencies = 0;       // handles of finished jobs
    };

    class Graph
    {
    public:
        std::vector<GraphNode> nodes;

        void Prepare(IJobSystem* jobs, bool countVisits)
        {
            m_States.reset(new NodeState[nodes.size()]);
            m_Handles.assign(nodes.size(), JobHandle::Null);
            uint32_t indices = 0;
            for (const GraphNode& node : nodes)
                indices += countVisits ? node.count : 0;
            m_Visits.reset(new std::atomic<uint8_t>[indices ? indices : 1]);
            for (uint32_t i = 0; i < indices; ++i)
                m_Visits[i].store(0, std::memory_order_relaxed);
            m_VisitCount = indices;
            uint32_t offset = 0;
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                NodeState& state = m_States[i];
                state.jobs = jobs;
                state.nestedCount = nodes[i].nestedCount;
                state.work = nodes[i].work;
                state.childJobs = nodes[i].childJobs;
                state.childCount = nodes[i].childCount;
                state.childGrain = nodes[i].childGrain;
                state.visits = countVisits ? m_Visits.get() + offset : nullptr;
                offset += countVisits ? nodes[i].count : 0;
            }
        }

        // Schedules node `i`; `extra` (optional) is one more dependency, `stale` a finished job's handle.
        JobHandle ScheduleNode(IJobSystem* jobs, size_t i, JobHandle extra, JobHandle stale)
        {
            const GraphNode& node = nodes[i];
            std::vector<JobHandle>& dependencies = m_Scratch;
            dependencies.clear();
            for (uint32_t dependency : node.dependencies)
                dependencies.push_back(m_Handles[dependency]);
            for (uint32_t k = 0; k < node.nullDependencies; ++k)
                dependencies.insert(dependencies.begin() + (k % (dependencies.size() + 1)), JobHandle::Null);
            for (uint32_t k = 0; k < node.staleDependencies; ++k)
                dependencies.push_back(stale);
            if (extra != JobHandle::Null)
                dependencies.push_back(extra);
            const JobDesc desc{ &NodeState::Run, &m_States[i], node.count, node.grainSize, "node" };
            m_Handles[i] = Schedule(jobs, desc, dependencies.data(), uint32_t(dependencies.size()));
            return m_Handles[i];
        }

        void ScheduleAll(IJobSystem* jobs, JobHandle gate = JobHandle::Null, JobHandle stale = JobHandle::Null)
        {
            for (size_t i = 0; i < nodes.size(); ++i)
                ScheduleNode(jobs, i, nodes[i].dependencies.empty() ? gate : JobHandle::Null, stale);
        }

        void WaitAll(IJobSystem* jobs)
        {
            for (size_t i = nodes.size(); i-- > 0;)
                jobs->Wait(m_Handles[i]);
        }

        JobHandle Handle(size_t i) const { return m_Handles[i]; }

        // Every job ran each of its ranges once, every index once, the nested ParallelFors and child jobs completely,
        // and every job started after all its dependencies had finished (through jobs without indices, transitively).
        // `jobs` may be null once the system is gone.
        void Verify(IJobSystem* jobs, const char* name) const
        {
            std::vector<uint64_t> finish(nodes.size(), 0);
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                const GraphNode& node = nodes[i];
                const NodeState& state = m_States[i];
                const uint32_t ranges = RangeCount(node.count, node.grainSize);
                if (state.ranges.load() != ranges || (jobs && !jobs->IsComplete(m_Handles[i])) ||
                    state.nestedIndices.load() != uint64_t(ranges) * node.nestedCount ||
                    state.childIndices.load() != uint64_t(ranges) * node.childJobs * node.childCount)
                    throw std::runtime_error(std::string(name) + ": node " + std::to_string(i) + " ran " +
                                             std::to_string(state.ranges.load()) + " of " + std::to_string(ranges) +
                                             " ranges");
                uint64_t ready = 0;
                for (uint32_t dependency : node.dependencies)
                    ready = std::max(ready, finish[dependency]);
                if (node.count != 0)
                {
                    if (state.firstStart.load() <= ready)
                        throw std::runtime_error(std::string(name) + ": node " + std::to_string(i) +
                                                 " started before a dependency finished");
                    finish[i] = state.lastEnd.load();
                }
                else
                    finish[i] = ready;
            }
            for (uint32_t i = 0; i < m_VisitCount; ++i)
            {
                if (m_Visits[i].load() != 1)
                    throw std::runtime_error(std::string(name) + ": index " + std::to_string(i) + " ran " +
                                             std::to_string(m_Visits[i].load()) + " times");
            }
        }

    private:
        std::unique_ptr<NodeState[]> m_States;
        std::vector<JobHandle> m_Handles;
        std::unique_ptr<std::atomic<uint8_t>[]> m_Visits;
        uint32_t m_VisitCount = 0;
        std::vector<JobHandle> m_Scratch;
    };

    Graph MakeChain(uint32_t length)
    {
        Graph graph;
        graph.nodes.resize(length);
        for (uint32_t i = 0; i < length; ++i)
        {
            GraphNode& node = graph.nodes[i];
            node.count = i % 17 == 5 ? 0 : (i % 7 == 0 ? 25 : 1);
            node.grainSize = i % 7 == 0 ? 4 : 0;
            if (i > 0)
                node.dependencies.push_back(i - 1);
        }
        return graph;
    }

    Graph MakeFan(uint32_t width)
    {
        Graph graph;
        graph.nodes.resize(width + 2);
        graph.nodes[0].count = 10;
        graph.nodes[0].grainSize = 1;
        for (uint32_t i = 1; i <= width; ++i)
        {
            graph.nodes[i].count = 50;
            graph.nodes[i].grainSize = 7;
            graph.nodes[i].dependencies.push_back(0);
        }
        for (uint32_t i = 1; i <= width; ++i)
            graph.nodes[width + 1].dependencies.push_back(i);
        return graph;
    }

    Graph MakeDiamonds(uint32_t layers, uint32_t width)
    {
        Graph graph;
        graph.nodes.resize(1 + size_t(layers) * width + 1);
        for (uint32_t layer = 0; layer < layers; ++layer)
        {
            for (uint32_t j = 0; j < width; ++j)
            {
                GraphNode& node = graph.nodes[1 + layer * width + j];
                node.count = (layer + j) % 5 == 0 ? 0 : 3 + (j % 4);
                node.grainSize = j % 3;
                node.nullDependencies = j == 1 ? 1 : 0;
                node.staleDependencies = j == 2 ? 1 : 0;
                if (layer == 0)
                    node.dependencies.push_back(0);
                else
                {
                    node.dependencies.push_back(1 + (layer - 1) * width + j);
                    node.dependencies.push_back(1 + (layer - 1) * width + (j + 1) % width);
                }
            }
        }
        GraphNode& sink = graph.nodes.back();
        for (uint32_t j = 0; j < width; ++j)
            sink.dependencies.push_back(1 + (layers - 1) * width + j);
        return graph;
    }

    // --------------------------------------------------------------------------------------- the watchdog

    std::atomic<int64_t> g_CaseStart{ 0 };
    std::atomic<const char*> g_CaseName{ nullptr };
    std::atomic<bool> g_AllDone{ false };
    constexpr int64_t c_CaseTimeLimitSeconds = 300;

    int64_t NowMilliseconds()
    {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    }

    void Watchdog()
    {
        while (!g_AllDone.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const int64_t start = g_CaseStart.load();
            if (start != 0 && NowMilliseconds() - start > c_CaseTimeLimitSeconds * 1000)
            {
                std::fflush(stdout);
                std::fprintf(stderr, "test_job_system: %s did not finish in %lld s (a deadlock?)\n", g_CaseName.load(),
                             (long long)c_CaseTimeLimitSeconds);
                std::fflush(stderr);
                std::_Exit(3);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------- the tests

// Creation, thread counts, worker indices of workers, of the calling thread, of nested calls and across two systems,
// and Schedule's argument checks.
static void test_creation_and_indices()
{
    JobSystemDesc desc;
    IJobSystem* none = nullptr;
    CHECK(CreateJobSystem(desc, nullptr) == nvrhi::FE_INVALID_ARGS);
    desc.maxExternalThreads = 0;
    CHECK(CreateJobSystem(desc, &none) == nvrhi::FE_INVALID_ARGS && none == nullptr);
    desc.maxExternalThreads = 65;
    CHECK(CreateJobSystem(desc, &none) == nvrhi::FE_INVALID_ARGS && none == nullptr);
    desc.maxExternalThreads = 2;
    desc.workerThreadCount = 257;
    CHECK(CreateJobSystem(desc, &none) == nvrhi::FE_INVALID_ARGS && none == nullptr);
    {
        nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(0);
        const uint32_t hardware = std::thread::hardware_concurrency();
        CHECK(jobs->GetThreadCount() == 2 + std::min(hardware > 1 ? hardware - 1 : 1u, 256u));
        CHECK(jobs->GetCurrentWorkerIndex() == c_NoWorker);
    }

    nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(3, 2);
    CHECK(jobs->GetThreadCount() == 5);
    CHECK(jobs->GetCurrentWorkerIndex() == c_NoWorker);

    struct Indices
    {
        IJobSystem* jobs = nullptr;
        IJobSystem* other = nullptr;    // a second system, driven from inside this one's jobs
        std::thread::id caller;
        std::atomic<uint32_t> callerIndex{ c_NoWorker };
        std::atomic<uint32_t> workerMask{ 0 };
        std::atomic<uint32_t> nestedRanges{ 0 };
        std::atomic<uint32_t> otherRanges{ 0 };
        uint32_t outerIndex = c_NoWorker;   // per call of Nested: set by the caller's range
    } indices;
    indices.jobs = jobs;
    indices.caller = std::this_thread::get_id();

    struct Tasks
    {
        static void Outer(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            Indices* indices = static_cast<Indices*>(context);
            JOB_CHECK(workerIndex < 5);
            JOB_CHECK(indices->jobs->GetCurrentWorkerIndex() == workerIndex);
            if (std::this_thread::get_id() == indices->caller)
            {
                JOB_CHECK(workerIndex < 2);
                uint32_t expected = c_NoWorker;
                if (!indices->callerIndex.compare_exchange_strong(expected, workerIndex))
                    JOB_CHECK(expected == workerIndex);
            }
            else
            {
                JOB_CHECK(workerIndex >= 2);
                indices->workerMask.fetch_or(1u << workerIndex);
            }
            Spin(20000);
        }

        static void Nested(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            Indices* indices = static_cast<Indices*>(context);
            JOB_CHECK(indices->jobs->GetCurrentWorkerIndex() == workerIndex);
            if (std::this_thread::get_id() == indices->caller)
                JOB_CHECK(workerIndex == indices->callerIndex.load());
            indices->nestedRanges.fetch_add(1);
        }

        static void OuterNesting(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            Indices* indices = static_cast<Indices*>(context);
            indices->jobs->ParallelFor(8, 1, &Nested, indices);
            JOB_CHECK(indices->jobs->GetCurrentWorkerIndex() == workerIndex);
        }

        // Runs on a thread of `jobs`; inside, the second system's ranges see their own indices and this system's.
        struct Cross
        {
            Indices* indices;
            uint32_t outer;
        };
        static void OtherInner(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            const Cross* cross = static_cast<const Cross*>(context);
            JOB_CHECK(workerIndex < 2);   // one external slot, one worker
            JOB_CHECK(cross->indices->other->GetCurrentWorkerIndex() == workerIndex);
            if (cross->indices->jobs->GetCurrentWorkerIndex() != c_NoWorker)
                JOB_CHECK(cross->indices->jobs->GetCurrentWorkerIndex() == cross->outer);
            cross->indices->otherRanges.fetch_add(1);
            Spin(5000);
        }
        static void OtherOuter(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            Indices* indices = static_cast<Indices*>(context);
            Cross cross{ indices, workerIndex };
            indices->other->ParallelFor(4, 1, &OtherInner, &cross);
            JOB_CHECK(indices->other->GetCurrentWorkerIndex() == c_NoWorker);
        }
    };

    for (uint32_t round = 0; round < 20; ++round)
        jobs->ParallelFor(64, 1, &Tasks::Outer, &indices);
    CheckJobs("worker indices");
    CHECK(indices.callerIndex.load() < 2);
    CHECK((indices.workerMask.load() & ~0x1cu) == 0);   // workers 2, 3 and 4 only
    CHECK(jobs->GetCurrentWorkerIndex() == c_NoWorker);

    jobs->ParallelFor(16, 1, &Tasks::OuterNesting, &indices);
    CheckJobs("nested worker indices");
    CHECK(indices.nestedRanges.load() == 16 * 8);

    // A second system with one external slot: this system's four threads take turns at it.
    nvrhi::AutoPtr<IJobSystem> other = CreateSystem(1, 1);
    indices.other = other;
    CHECK(other->GetThreadCount() == 2);
    jobs->ParallelFor(32, 1, &Tasks::OtherOuter, &indices);
    CheckJobs("two systems");
    CHECK(indices.otherRanges.load() == 32 * 4);

    // Schedule's arguments.
    JobHandle handle = JobHandle(123);
    const JobDesc noFunction{ nullptr, nullptr, 5, 1, "no function" };
    CHECK(jobs->Schedule(noFunction, nullptr, 0, handle) == nvrhi::FE_INVALID_ARGS);
    CHECK(handle == JobHandle::Null);
    Counter counter;
    const JobDesc counted{ &Counter::Run, &counter, 5, 1, "counted" };
    handle = JobHandle(123);
    CHECK(jobs->Schedule(counted, nullptr, 2, handle) == nvrhi::FE_INVALID_ARGS && handle == JobHandle::Null);
    const JobHandle notAJob[] = { JobHandle((uint64_t(1) << 32) | c_JobCapacity) };
    CHECK(jobs->Schedule(counted, notAJob, 1, handle) == nvrhi::FE_INVALID_ARGS && handle == JobHandle::Null);
    const JobHandle noGeneration[] = { JobHandle(5) };
    CHECK(jobs->Schedule(counted, noGeneration, 1, handle) == nvrhi::FE_INVALID_ARGS && handle == JobHandle::Null);
    CHECK(counter.value.load() == 0);

    // Null and out-of-range handles are complete; a job without indices and dependencies finishes in Schedule.
    CHECK(jobs->IsComplete(JobHandle::Null));
    CHECK(jobs->IsComplete(notAJob[0]));
    jobs->Wait(JobHandle::Null);
    jobs->Wait(notAJob[0]);
    const JobHandle empty = Schedule(jobs, JobDesc{ nullptr, nullptr, 0, 0, "empty" });
    CHECK(jobs->IsComplete(empty));
    jobs->Wait(empty);

    // A handle stays complete after its slot has run other jobs.
    const JobHandle first = Schedule(jobs, counted);
    jobs->Wait(first);
    for (uint32_t i = 0; i < 2 * c_JobCapacity; ++i)
        jobs->Wait(Schedule(jobs, counted));
    CHECK(jobs->IsComplete(first));
    jobs->Wait(first);
    CHECK(counter.value.load() == uint64_t(2 * c_JobCapacity + 1) * 5);
    CheckFullCapacity(jobs);
    CheckFullCapacity(other);
}

// ParallelFor, Schedule and JobSystemParallelFor cut [0, count) into ranges of grainSize, the last shorter, whatever
// the thread count: every range runs once with the expected bounds, every index once, and per-range results keyed by
// range equal a serial reference.
static void test_parallel_for_split()
{
    struct Split
    {
        uint32_t count = 0;
        uint32_t grain = 1;   // effective
        std::unique_ptr<std::atomic<uint32_t>[]> rangeCalls;
        std::unique_ptr<std::atomic<uint32_t>[]> rangeEnds;
        std::unique_ptr<std::atomic<uint64_t>[]> rangeValues;
        std::unique_ptr<std::atomic<uint8_t>[]> visits;

        void Reset(uint32_t count_, uint32_t grainSize)
        {
            count = count_;
            grain = grainSize ? grainSize : (count ? count : 1);
            const uint32_t ranges = RangeCount(count, grainSize);
            rangeCalls.reset(new std::atomic<uint32_t>[ranges + 1]);
            rangeEnds.reset(new std::atomic<uint32_t>[ranges + 1]);
            rangeValues.reset(new std::atomic<uint64_t>[ranges + 1]);
            visits.reset(new std::atomic<uint8_t>[count + 1]);
            for (uint32_t i = 0; i <= ranges; ++i)
            {
                rangeCalls[i].store(0);
                rangeEnds[i].store(0);
                rangeValues[i].store(0);
            }
            for (uint32_t i = 0; i <= count; ++i)
                visits[i].store(0);
        }

        static uint64_t Value(uint32_t begin, uint32_t end)
        {
            uint64_t hash = 0xcbf29ce484222325ull;
            for (uint32_t i = begin; i < end; ++i)
                hash = (hash ^ i) * 0x100000001b3ull;
            return hash ^ (uint64_t(end - begin) << 40);
        }

        static void Run(void* context, uint32_t begin, uint32_t end, uint32_t)
        {
            Split* split = static_cast<Split*>(context);
            JOB_CHECK(begin < end && end <= split->count && begin % split->grain == 0);
            if (!(begin < end && end <= split->count))
                return;
            const uint32_t range = begin / split->grain;
            JOB_CHECK(split->rangeCalls[range].fetch_add(1) == 0);
            split->rangeEnds[range].store(end);
            split->rangeValues[range].store(Value(begin, end));
            for (uint32_t i = begin; i < end; ++i)
                split->visits[i].fetch_add(1, std::memory_order_relaxed);
        }

        static void RunIndex(void* context, uint32_t index)
        {
            Split* split = static_cast<Split*>(context);
            Run(split, index, index + 1, 0);
        }

        void Verify(const char* how, uint32_t workers, uint32_t grainSize) const
        {
            const uint32_t ranges = RangeCount(count, grainSize);
            for (uint32_t range = 0; range < ranges; ++range)
            {
                const uint32_t begin = range * grain;
                const uint32_t end = begin + std::min(grain, count - begin);
                if (rangeCalls[range].load() != 1 || rangeEnds[range].load() != end ||
                    rangeValues[range].load() != Value(begin, end))
                    throw std::runtime_error(std::string("split (") + how + "): count " + std::to_string(count) +
                                             ", grain " + std::to_string(grainSize) + ", " + std::to_string(workers) +
                                             " workers: range " + std::to_string(range) + " ran " +
                                             std::to_string(rangeCalls[range].load()) + " times");
            }
            CHECK(rangeCalls[ranges].load() == 0);
            for (uint32_t i = 0; i < count; ++i)
                CHECK(visits[i].load() == 1);
        }
    };

    const uint32_t counts[] = { 0, 1, 2, 3, 7, 64, 65, 1000, 4097, 100003 };
    for (uint32_t workers : { 1u, 2u, 8u })
    {
        nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(workers);
        Split split;
        for (uint32_t count : counts)
        {
            const uint32_t grains[] = { 0, 1, 2, 3, 64, 1000, count, count + 1, 0xffffffffu };
            for (uint32_t grainSize : grains)
            {
                split.Reset(count, grainSize);
                jobs->ParallelFor(count, grainSize, &Split::Run, &split);
                CheckJobs("split: ParallelFor");
                split.Verify("ParallelFor", workers, grainSize);

                split.Reset(count, grainSize);
                const JobHandle handle = Schedule(jobs, JobDesc{ &Split::Run, &split, count, grainSize, "split" });
                jobs->Wait(handle);
                CHECK(jobs->IsComplete(handle));
                CheckJobs("split: Schedule");
                split.Verify("Schedule", workers, grainSize);
            }
            // The render graph's adapter: one index per task.
            split.Reset(count, 1);
            donut::engine::JobSystemParallelFor(jobs.Get(), count, &Split::RunIndex, &split);
            CheckJobs("split: JobSystemParallelFor");
            split.Verify("JobSystemParallelFor", workers, 1);
        }
        CheckFullCapacity(jobs);
    }
}

// Chains, fans and diamonds, scheduled from a thread without a slot (the main thread) and from inside a job, with all
// dependencies registered before any runs (a gate) and while they run; Null and stale dependencies are ignored; a long
// chain of jobs without indices finishes without recursion.
static void test_dependency_graphs()
{
    for (uint32_t workers : { 1u, 2u, 8u })
    {
        nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(workers);
        const JobHandle stale = Schedule(jobs, JobDesc{ nullptr, nullptr, 0, 0, "stale" });
        Graph graphs[] = { MakeChain(300), MakeFan(200), MakeDiamonds(50, 8) };
        const char* names[] = { "chain", "fan", "diamonds" };
        for (size_t g = 0; g < 3; ++g)
        {
            Graph& graph = graphs[g];
            for (uint32_t mode = 0; mode < 3; ++mode)
            {
                graph.Prepare(jobs, true);
                if (mode == 0)
                {
                    // From the main thread, behind a gate.
                    Gate gate;
                    const JobHandle gateJob = Schedule(jobs, gate.Desc());
                    graph.ScheduleAll(jobs, gateJob, stale);
                    CHECK(!jobs->IsComplete(graph.Handle(graph.nodes.size() - 1)));
                    gate.open.store(true);
                    graph.WaitAll(jobs);
                }
                else if (mode == 1)
                {
                    // From the main thread, while they run.
                    graph.ScheduleAll(jobs, JobHandle::Null, stale);
                    graph.WaitAll(jobs);
                }
                else
                {
                    // From inside a job, which holds a slot.
                    struct Driver
                    {
                        IJobSystem* jobs;
                        Graph* graph;
                        JobHandle stale;
                        static void Run(void* context, uint32_t, uint32_t, uint32_t)
                        {
                            Driver* driver = static_cast<Driver*>(context);
                            try
                            {
                                driver->graph->ScheduleAll(driver->jobs, JobHandle::Null, driver->stale);
                                driver->graph->WaitAll(driver->jobs);
                            }
                            catch (...)
                            {
                                JOB_CHECK(false);
                            }
                        }
                    } driver{ jobs, &graph, stale };
                    jobs->Wait(Schedule(jobs, JobDesc{ &Driver::Run, &driver, 1, 0, "driver" }));
                }
                CheckJobs(names[g]);
                graph.Verify(jobs, names[g]);
            }
        }

        // 3000 jobs without indices in a chain behind a gate, then one with.
        Graph empties;
        empties.nodes.resize(3001);
        for (uint32_t i = 0; i < 3001; ++i)
        {
            empties.nodes[i].count = i == 3000 ? 1 : 0;
            if (i > 0)
                empties.nodes[i].dependencies.push_back(i - 1);
        }
        empties.Prepare(jobs, true);
        Gate gate;
        const JobHandle gateJob = Schedule(jobs, gate.Desc());
        empties.ScheduleAll(jobs, gateJob);
        CHECK(!jobs->IsComplete(empties.Handle(0)));
        gate.open.store(true);
        jobs->Wait(empties.Handle(3000));
        // A job is marked finished after its dependents were released, so the last one can finish first: Verify asks
        // whether every job is complete.
        empties.WaitAll(jobs);
        CheckJobs("empty chain");
        empties.Verify(jobs, "empty chain");
        CheckFullCapacity(jobs);
    }
}

// Waits inside jobs: recursive ParallelFor trees, recursive Schedule and Wait, a job that waits for a job that cannot
// start yet, several threads waiting for one job; with spinning workers and with workers that sleep at once.
static void test_nested_waits()
{
    struct Tree
    {
        IJobSystem* jobs;
        std::atomic<uint32_t> leaves{ 0 };
        uint32_t depth;
    };
    struct TreeNode
    {
        Tree* tree;
        uint32_t depth;
        static void Run(void* context, uint32_t begin, uint32_t end, uint32_t)
        {
            const TreeNode* node = static_cast<const TreeNode*>(context);
            for (uint32_t i = begin; i < end; ++i)
            {
                if (node->depth == node->tree->depth)
                {
                    node->tree->leaves.fetch_add(1);
                    continue;
                }
                TreeNode child{ node->tree, node->depth + 1 };
                node->tree->jobs->ParallelFor(6, 1, &TreeNode::Run, &child);
            }
        }
    };

    struct Fibonacci
    {
        IJobSystem* jobs;
        uint32_t n;
        uint64_t result = 0;
        static void Run(void* context, uint32_t, uint32_t, uint32_t)
        {
            Fibonacci* f = static_cast<Fibonacci*>(context);
            if (f->n < 2)
            {
                f->result = f->n;
                return;
            }
            Fibonacci a{ f->jobs, f->n - 1 };
            Fibonacci b{ f->jobs, f->n - 2 };
            JobHandle ha = JobHandle::Null;
            JobHandle hb = JobHandle::Null;
            JOB_CHECK(f->jobs->Schedule(JobDesc{ &Run, &a, 1, 0, "fib" }, nullptr, 0, ha) == nvrhi::FS_OK);
            JOB_CHECK(f->jobs->Schedule(JobDesc{ &Run, &b, 1, 0, "fib" }, nullptr, 0, hb) == nvrhi::FS_OK);
            f->jobs->Wait(hb);
            f->jobs->Wait(ha);
            f->result = a.result + b.result;
        }
    };

    struct Waiter
    {
        IJobSystem* jobs;
        JobHandle target;
        const std::atomic<uint32_t>* targetDone;
        std::atomic<uint32_t> waited{ 0 };
        static void Run(void* context, uint32_t begin, uint32_t end, uint32_t)
        {
            Waiter* waiter = static_cast<Waiter*>(context);
            for (uint32_t i = begin; i < end; ++i)
            {
                waiter->jobs->Wait(waiter->target);
                JOB_CHECK(waiter->jobs->IsComplete(waiter->target));
                JOB_CHECK(waiter->targetDone->load() == 1);
                waiter->waited.fetch_add(1);
            }
        }
    };
    struct Done
    {
        std::atomic<uint32_t> value{ 0 };
        static void Run(void* context, uint32_t, uint32_t, uint32_t)
        {
            static_cast<Done*>(context)->value.fetch_add(1);
        }
    };

    for (uint32_t spinCount : { 4096u, 0u })
    {
        for (uint32_t workers : { 1u, 2u, 8u })
        {
            nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(workers, 2, spinCount, spinCount ? 30 : 0);

            Tree tree{ jobs, {}, 4 };
            TreeNode root{ &tree, 0 };
            jobs->ParallelFor(6, 1, &TreeNode::Run, &root);
            CheckJobs("ParallelFor tree");
            CHECK(tree.leaves.load() == 6 * 6 * 6 * 6 * 6);

            Fibonacci fibonacci{ jobs, 15 };
            jobs->Wait(Schedule(jobs, JobDesc{ &Fibonacci::Run, &fibonacci, 1, 0, "fib" }));
            CheckJobs("Schedule and Wait recursion");
            CHECK(fibonacci.result == 610);

            // A job waits for a job that waits for a gate the main thread opens later; a second thread opens it while
            // eight ranges and the main thread wait for the same job.
            for (uint32_t variant = 0; variant < 2; ++variant)
            {
                Gate gate;
                Done done;
                const JobHandle gateJob = Schedule(jobs, gate.Desc());
                const JobHandle target = Schedule(jobs, JobDesc{ &Done::Run, &done, 1, 0, "target" }, &gateJob, 1);
                Waiter waiter{ jobs, target, &done.value };
                if (variant == 0)
                {
                    const JobHandle waiting = Schedule(jobs, JobDesc{ &Waiter::Run, &waiter, 1, 0, "waiter" });
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    CHECK(!jobs->IsComplete(waiting) && !jobs->IsComplete(target));
                    gate.open.store(true);
                    jobs->Wait(waiting);
                    CHECK(waiter.waited.load() == 1);
                }
                else
                {
                    std::thread opener([&gate] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        gate.open.store(true);
                    });
                    jobs->ParallelFor(8, 1, &Waiter::Run, &waiter);
                    jobs->Wait(target);
                    opener.join();
                    CHECK(waiter.waited.load() == 8);
                }
                CheckJobs("waiting for a gated job");
                CHECK(done.value.load() == 1 && jobs->IsComplete(target));
            }
            CheckFullCapacity(jobs);
        }
    }
}

// More external threads than slots: callers wait for a slot and proceed; two threads never share a worker index at
// the same time; a caller's ranges on its own thread keep one external index for the call (other ranges run on
// workers or on other callers that help while they wait).
static void test_external_threads()
{
    nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(2, 2);
    struct Shared
    {
        IJobSystem* jobs;
        std::atomic<uint32_t> inUse[4];
        std::atomic<uint64_t> ranges{ 0 };
    } shared{ jobs, {}, {} };
    for (std::atomic<uint32_t>& use : shared.inUse)
        use.store(0);
    struct Call
    {
        Shared* shared;
        std::thread::id caller;
        uint32_t callerIndex;
        static void Run(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            Call* call = static_cast<Call*>(context);
            JOB_CHECK(workerIndex < 4);
            if (workerIndex >= 4)
                return;
            JOB_CHECK(call->shared->inUse[workerIndex].fetch_add(1) == 0);
            if (std::this_thread::get_id() == call->caller)
            {
                JOB_CHECK(workerIndex < 2);
                if (call->callerIndex == c_NoWorker)
                    call->callerIndex = workerIndex;
                JOB_CHECK(call->callerIndex == workerIndex);
            }
            // Elsewhere: a worker, or another caller helping while it waits, each with its own index.
            JOB_CHECK(call->shared->jobs->GetCurrentWorkerIndex() == workerIndex);
            Spin(2000);
            call->shared->ranges.fetch_add(1);
            call->shared->inUse[workerIndex].fetch_sub(1);
        }
    };
    static constexpr uint32_t c_Threads = 6;
    static constexpr uint32_t c_Calls = 150;
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < c_Threads; ++t)
    {
        threads.emplace_back([&shared] {
            for (uint32_t i = 0; i < c_Calls; ++i)
            {
                Call call{ &shared, std::this_thread::get_id(), c_NoWorker };
                shared.jobs->ParallelFor(32, 2, &Call::Run, &call);
                JOB_CHECK(shared.jobs->GetCurrentWorkerIndex() == c_NoWorker);
            }
        });
    }
    for (std::thread& thread : threads)
        thread.join();
    CheckJobs("external threads");
    CHECK(shared.ranges.load() == uint64_t(c_Threads) * c_Calls * 16);

    // One external slot: while thread A holds it (waiting inside a call for a gated job), thread B's call waits.
    nvrhi::AutoPtr<IJobSystem> single = CreateSystem(1, 1, 64);
    Gate gate;
    const JobHandle gated = Schedule(single, gate.Desc());
    struct Hold
    {
        IJobSystem* jobs;
        JobHandle gated;
        std::atomic<bool> inside{ false };
        static void Run(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            Hold* hold = static_cast<Hold*>(context);
            JOB_CHECK(workerIndex == 0);
            hold->inside.store(true);
            hold->jobs->Wait(hold->gated);
        }
    } hold{ single, gated };
    struct Second
    {
        std::atomic<bool> done{ false };
        std::atomic<uint32_t> index{ c_NoWorker };
        static void Run(void* context, uint32_t, uint32_t, uint32_t workerIndex)
        {
            static_cast<Second*>(context)->index.store(workerIndex);
        }
    } second;
    std::thread a([&] { single->ParallelFor(1, 0, &Hold::Run, &hold); });
    while (!hold.inside.load())
        std::this_thread::yield();
    std::thread b([&] {
        single->ParallelFor(1, 0, &Second::Run, &second);
        second.done.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(!second.done.load());   // B waits for A's slot
    gate.open.store(true);
    a.join();
    b.join();
    CheckJobs("one external slot");
    CHECK(second.done.load() && second.index.load() == 0);
    CheckFullCapacity(jobs);
    CheckFullCapacity(single);
}

// The job table and the edge pool: Schedule fails with FE_OUT_OF_MEMORY and a Null handle when either is exhausted,
// nothing of the failed call runs, the table takes exactly its capacity (FillTable), a finished job's handle stays
// complete while its slot holds a pending job, ParallelFor still covers its indices, and the capacity comes back.
static void test_table_full()
{
    nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(2);
    {
        Counter earlyCounter;
        const JobHandle early = Schedule(jobs, JobDesc{ &Counter::Run, &earlyCounter, 1, 0, "early" });
        jobs->Wait(early);
        Gate gate;
        Counter counter;
        std::vector<JobHandle> handles;
        const JobHandle gateJob = FillTable(jobs, gate, counter, handles);
        std::printf("test_job_system: the job table took a gate and %zu jobs\n", handles.size());
        // Every slot is taken, the early job's by a pending one: its handle stays complete.
        CHECK(jobs->IsComplete(early) && earlyCounter.value.load() == 1);
        jobs->Wait(early);

        // On a full table, ParallelFor runs its ranges on the calling thread.
        struct Visits
        {
            std::atomic<uint32_t> visits[100];
            static void Run(void* context, uint32_t begin, uint32_t end, uint32_t)
            {
                for (uint32_t i = begin; i < end; ++i)
                    static_cast<Visits*>(context)->visits[i].fetch_add(1);
            }
        } visits;
        for (std::atomic<uint32_t>& visit : visits.visits)
            visit.store(0);
        jobs->ParallelFor(100, 3, &Visits::Run, &visits);
        for (const std::atomic<uint32_t>& visit : visits.visits)
            CHECK(visit.load() == 1);

        CHECK(counter.value.load() == 0);
        gate.open.store(true);
        for (JobHandle handle : handles)
            jobs->Wait(handle);
        jobs->Wait(gateJob);   // its edges are back once it is finished
        CHECK(counter.value.load() == handles.size());
    }
    {
        // The edge pool: 20000 dependencies on one pending job.
        Counter counter;
        const JobDesc counted{ &Counter::Run, &counter, 1, 0, "fan-in" };
        Gate gate;
        const JobHandle gateJob = Schedule(jobs, gate.Desc());
        std::vector<JobHandle> dependencies(20000, gateJob);
        JobHandle handle = JobHandle(5);
        CHECK(jobs->Schedule(counted, dependencies.data(), 20000, handle) == nvrhi::FE_OUT_OF_MEMORY);
        CHECK(handle == JobHandle::Null);
        JobHandle many = JobHandle::Null;
        CHECK(jobs->Schedule(counted, dependencies.data(), 16000, many) == nvrhi::FS_OK);
        CHECK(jobs->Schedule(counted, dependencies.data(), 1000, handle) == nvrhi::FE_OUT_OF_MEMORY);
        JobHandle rest = JobHandle::Null;
        CHECK(jobs->Schedule(counted, dependencies.data(), c_EdgeCapacity - 16000, rest) == nvrhi::FS_OK);
        CHECK(jobs->Schedule(counted, dependencies.data(), 1, handle) == nvrhi::FE_OUT_OF_MEMORY);
        CHECK(!jobs->IsComplete(many) && !jobs->IsComplete(rest));
        gate.open.store(true);
        jobs->Wait(many);
        jobs->Wait(rest);
        jobs->Wait(gateJob);
        CHECK(counter.value.load() == 2);

        // Every edge came back as soon as the gate was finished: no retry needed here.
        Gate again;
        const JobHandle againJob = Schedule(jobs, again.Desc());
        dependencies.assign(c_EdgeCapacity, againJob);
        const JobHandle all = Schedule(jobs, counted, dependencies.data(), c_EdgeCapacity);
        again.open.store(true);
        jobs->Wait(all);
        jobs->Wait(againJob);
        CHECK(counter.value.load() == 3);
    }
    CheckFullCapacity(jobs);
}

// Releasing the last reference runs every scheduled job first, with workers asleep, spinning or busy.
static void test_shutdown()
{
    for (uint32_t workers : { 1u, 2u, 8u })
    {
        Graph graph = MakeDiamonds(20, 6);
        for (GraphNode& node : graph.nodes)
            node.work = 3000;
        struct Slow
        {
            static void Run(void*, uint32_t, uint32_t, uint32_t)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        };
        {
            nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(workers);
            graph.Prepare(nullptr, true);
            const JobHandle slow = Schedule(jobs, JobDesc{ &Slow::Run, nullptr, 1, 0, "slow" });
            graph.ScheduleAll(jobs, slow);
            CHECK(!jobs->IsComplete(graph.Handle(graph.nodes.size() - 1)));
            const int64_t start = NowMilliseconds();
            jobs = nullptr;
            std::printf("test_job_system: shutdown with %zu pending jobs, %u workers: %lld ms\n", graph.nodes.size(),
                        workers, (long long)(NowMilliseconds() - start));
        }
        CheckJobs("shutdown with pending work");
        // The handles are gone with the system; the record shows that every job ran once, in order.
        graph.Verify(nullptr, "shutdown");
    }
    for (uint32_t pause : { 0u, 50u })
    {
        nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(8, 2, 64);
        Counter counter;
        jobs->ParallelFor(1000, 10, &Counter::Run, &counter);
        CHECK(counter.value.load() == 1000);
        if (pause)
            std::this_thread::sleep_for(std::chrono::milliseconds(pause));   // the workers are asleep by now
        const int64_t start = NowMilliseconds();
        jobs = nullptr;
        const int64_t elapsed = NowMilliseconds() - start;
        CHECK(elapsed < 2000);
    }
}

// After warm-up, scheduling graphs, ParallelFor (nested, through the adapter, from a second external thread), waits,
// sleeping and waking allocate nothing.
static void test_steady_state_allocations()
{
    nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(4, 2, 16);   // short spins: workers sleep and wake every round
    constexpr uint32_t c_Nodes = 64;
    struct Fixed
    {
        NodeState states[c_Nodes];
        JobHandle handles[c_Nodes];
        uint32_t counts[c_Nodes];
        std::atomic<uint64_t> indices{ 0 };
    };
    std::unique_ptr<Fixed> fixed(new Fixed);
    for (uint32_t i = 0; i < c_Nodes; ++i)
    {
        fixed->states[i].jobs = jobs;
        fixed->states[i].nestedCount = i % 9 == 0 ? 12 : 0;
        fixed->counts[i] = i % 5 == 0 ? 0 : 1 + (i * 7) % 40;
    }
    struct Index
    {
        static void Run(void* context, uint32_t index) { static_cast<Fixed*>(context)->indices.fetch_add(index + 1); }
    };

    // A second external thread runs its own ParallelFor calls each round.
    std::atomic<uint32_t> round{ 0 };
    std::atomic<uint32_t> roundDone{ 0 };
    std::atomic<bool> stop{ false };
    Counter external;
    std::thread helper([&] {
        uint32_t seen = 0;
        while (!stop.load())
        {
            if (round.load() == seen)
            {
                std::this_thread::yield();
                continue;
            }
            seen = round.load();
            jobs->ParallelFor(512, 8, &Counter::Run, &external);
            jobs->ParallelFor(64, 1, &Counter::Run, &external);
            roundDone.store(seen);
        }
    });

    uint32_t failures = 0;
    auto runRound = [&](uint32_t number) {
        for (uint32_t i = 0; i < c_Nodes; ++i)
            fixed->states[i].Reset();
        round.store(number);
        for (uint32_t i = 0; i < c_Nodes; ++i)
        {
            JobHandle dependencies[3] = { JobHandle::Null, JobHandle::Null, JobHandle::Null };
            if (i > 0)
                dependencies[0] = fixed->handles[i - 1];
            if (i > 4)
                dependencies[1] = fixed->handles[i - 5];
            if (i > 16)
                dependencies[2] = fixed->handles[i / 2];
            const JobDesc desc{ &NodeState::Run, &fixed->states[i], fixed->counts[i], 3, "steady" };
            fixed->handles[i] = JobHandle::Null;
            if (jobs->Schedule(desc, dependencies, 3, fixed->handles[i]) != nvrhi::FS_OK)
                ++failures;
        }
        jobs->ParallelFor(256, 4, &Counter::Run, &external);
        donut::engine::JobSystemParallelFor(jobs.Get(), 16, &Index::Run, fixed.get());
        for (uint32_t i = c_Nodes; i-- > 0;)
            jobs->Wait(fixed->handles[i]);
        for (uint32_t i = 0; i < c_Nodes; ++i)
        {
            if (!jobs->IsComplete(fixed->handles[i]) ||
                fixed->states[i].ranges.load() != RangeCount(fixed->counts[i], 3))
                ++failures;
        }
        while (roundDone.load() != number)
            std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));   // the workers go to sleep
    };

    for (uint32_t i = 1; i <= 3; ++i)
        runRound(i);
    const uint64_t before = g_NewCount.load();
    for (uint32_t i = 4; i <= 13; ++i)
        runRound(i);
    const uint64_t after = g_NewCount.load();
    stop.store(true);
    helper.join();
    std::printf("test_job_system: steady state: %llu operator new in 10 rounds\n",
                (unsigned long long)(after - before));
    CheckJobs("steady state");
    CHECK(failures == 0);
    CHECK(after == before);
    CHECK(fixed->indices.load() == 13ull * (16 * 17 / 2));

    // The counter sees allocations when there are some.
    int* probe = new int[16];
    probe[0] = 1;
    const bool counted = g_NewCount.load() > after;
    delete[] probe;
    CHECK(counted);
    CheckFullCapacity(jobs);
}

// Random graphs from fixed seeds, driven by the main thread and two more external threads at once (one more than the
// slots), with nested ParallelFor, Null and stale dependencies and Waits midway, at 1, 2 and 8 workers.
static void test_stress()
{
    constexpr uint32_t c_Seeds = 12;
    constexpr uint32_t c_Nodes = 150;
    constexpr uint32_t c_Drivers = 3;
    for (uint32_t workers : { 1u, 2u, 8u })
    {
        nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(workers, 2, workers == 2 ? 0 : 512);
        const JobHandle stale = Schedule(jobs, JobDesc{ nullptr, nullptr, 0, 0, "stale" });
        uint64_t total = 0;
        for (uint32_t seed = 1; seed <= c_Seeds; ++seed)
        {
            Graph graphs[c_Drivers];
            for (uint32_t d = 0; d < c_Drivers; ++d)
            {
                Random random(seed * 1000 + d * 17 + workers);
                Graph& graph = graphs[d];
                graph.nodes.resize(c_Nodes);
                for (uint32_t i = 0; i < c_Nodes; ++i)
                {
                    GraphNode& node = graph.nodes[i];
                    node.count = random.Below(5) == 0 ? 0 : random.Below(200);
                    node.grainSize = random.Below(4) == 0 ? 0 : 1 + random.Below(40);
                    node.nestedCount = random.Below(10) == 0 ? 1 + random.Below(20) : 0;
                    node.work = random.Below(3) == 0 ? random.Below(4000) : 0;
                    const uint32_t dependencies = i ? random.Below(6) : 0;
                    for (uint32_t k = 0; k < dependencies; ++k)
                        node.dependencies.push_back(random.Below(i));
                    node.nullDependencies = random.Below(10) == 0 ? 1 : 0;
                    node.staleDependencies = random.Below(10) == 0 ? 1 : 0;
                    // Jobs scheduled from inside the node's ranges, onto the running thread's deque.
                    node.childJobs = random.Below(8) == 0 ? 1 + random.Below(NodeState::c_MaxChildJobs) : 0;
                    node.childCount = random.Below(64);
                    node.childGrain = random.Below(3) == 0 ? 0 : 1 + random.Below(8);
                    total += node.count;
                }
                graph.Prepare(jobs, true);
            }
            auto drive = [&](uint32_t d) {
                Random random(seed * 7919 + d);
                Graph& graph = graphs[d];
                for (size_t i = 0; i < graph.nodes.size(); ++i)
                {
                    graph.ScheduleNode(jobs, i, JobHandle::Null, stale);
                    if (random.Below(20) == 0)
                        jobs->Wait(graph.Handle(random.Below(uint32_t(i + 1))));
                }
                graph.WaitAll(jobs);
            };
            std::exception_ptr failures[c_Drivers];
            auto driveCaught = [&](uint32_t d) {
                try
                {
                    drive(d);
                }
                catch (...)
                {
                    failures[d] = std::current_exception();
                }
            };
            std::thread second(driveCaught, 1u);
            std::thread third(driveCaught, 2u);
            driveCaught(0);
            second.join();
            third.join();
            for (const std::exception_ptr& failure : failures)
            {
                if (failure)
                    std::rethrow_exception(failure);
            }
            CheckJobs("stress");
            for (uint32_t d = 0; d < c_Drivers; ++d)
                graphs[d].Verify(jobs, "stress");
        }
        std::printf("test_job_system: stress, %u workers: %u graphs, %llu indices\n", workers, c_Seeds * c_Drivers,
                    (unsigned long long)total);
        CheckFullCapacity(jobs);
    }
}

// A Wait for a job with a range running lower on the same thread's stack returns at once instead of deadlocking. The
// review's example: one worker; G is a gate a thread opens later, A waits for G, C waits for A, all scheduled from a
// thread without a slot. One thread spins in G; the other runs A, A's wait for G runs C, and C's wait for A would never
// end. Also a job that waits for itself. Guarded: a regression fails after 20 s instead of hanging.
static void test_wait_on_own_stack()
{
    struct Scenario
    {
        IJobSystem* jobs = nullptr;
        Gate gate;
        JobHandle g = JobHandle::Null;
        JobHandle a = JobHandle::Null;
        std::atomic<std::thread::id> aThread{};
        std::atomic<uint32_t> aDone{ 0 };
        std::atomic<uint32_t> cDone{ 0 };
        std::atomic<uint32_t> cNested{ 0 };      // C ran on A's thread while A was running
        std::atomic<uint32_t> cReturned{ 0 };    // ... and its Wait for A returned with A not finished

        static void A(void* context, uint32_t, uint32_t, uint32_t)
        {
            Scenario* s = static_cast<Scenario*>(context);
            s->aThread.store(std::this_thread::get_id());
            s->jobs->Wait(s->g);
            JOB_CHECK(s->jobs->IsComplete(s->g));
            s->aDone.fetch_add(1);
        }
        static void C(void* context, uint32_t, uint32_t, uint32_t)
        {
            Scenario* s = static_cast<Scenario*>(context);
            const bool nested = s->aThread.load() == std::this_thread::get_id() && s->aDone.load() == 0;
            s->jobs->Wait(s->a);
            if (nested)
            {
                s->cNested.fetch_add(1);
                if (!s->jobs->IsComplete(s->a))
                    s->cReturned.fetch_add(1);
            }
            s->cDone.fetch_add(1);
        }
    };
    struct Self
    {
        IJobSystem* jobs = nullptr;
        JobHandle self = JobHandle::Null;
        std::atomic<uint32_t> returned{ 0 };
        static void Run(void* context, uint32_t, uint32_t, uint32_t)
        {
            Self* s = static_cast<Self*>(context);
            s->jobs->Wait(s->self);
            JOB_CHECK(!s->jobs->IsComplete(s->self));
            s->returned.fetch_add(1);
        }
    };

    // Debug builds report each such Wait through log::warning; Release builds stay silent.
    static std::atomic<uint32_t> s_Warnings{ 0 };
    s_Warnings.store(0);
    donut::log::SetCallback([](donut::log::Severity severity, const char* message) {
        if (severity == donut::log::Severity::Warning && std::strstr(message, "returns without waiting"))
            s_Warnings.fetch_add(1);
    });

    uint32_t nested = 0;
    for (uint32_t spinCount : { 4096u, 0u })
    {
        nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(1, 2, spinCount, spinCount ? 30 : 0);
        RunGuarded("a Wait for a job lower on the same stack", 20, [&] {
            for (uint32_t round = 0; round < 5; ++round)
            {
                Scenario s;
                s.jobs = jobs;
                s.g = Schedule(jobs, s.gate.Desc());
                s.a = Schedule(jobs, JobDesc{ &Scenario::A, &s, 1, 0, "A" });
                const JobHandle c = Schedule(jobs, JobDesc{ &Scenario::C, &s, 1, 0, "C" });
                std::thread opener([&s] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    s.gate.open.store(true);
                });
                jobs->Wait(c);
                jobs->Wait(s.a);
                opener.join();
                CheckJobs("a wait lower on the stack");
                CHECK(s.aDone.load() == 1 && s.cDone.load() == 1);
                CHECK(s.cReturned.load() == s.cNested.load());
                nested += s.cNested.load();
            }

            Gate gate;
            Self self;
            self.jobs = jobs;
            const JobHandle gateJob = Schedule(jobs, gate.Desc());
            self.self = Schedule(jobs, JobDesc{ &Self::Run, &self, 1, 0, "self" }, &gateJob, 1);
            gate.open.store(true);
            jobs->Wait(self.self);
            CheckJobs("a job that waits for itself");
            CHECK(self.returned.load() == 1);
        });
        CheckFullCapacity(jobs);
    }
    donut::log::ResetCallback();
    std::printf("test_job_system: C ran nested inside A in %u of 10 rounds; %u warnings\n", nested,
                s_Warnings.load());
    CHECK(nested >= 5);
#ifdef NDEBUG
    CHECK(s_Warnings.load() == 0);
#else
    CHECK(s_Warnings.load() == nested + 2);   // the nested waits for A, and each system's job that waits for itself
#endif
}

// A Wait that waits for an external slot ends when its job finishes. One slot: thread A holds it in a ParallelFor
// range that runs until thread B's Wait for job K returns; B blocks on the slot; then K finishes. Guarded.
static void test_wait_for_slot_ends()
{
    nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(1, 1, 64);
    RunGuarded("a Wait blocked on an external slot", 20, [&] {
        struct Hold
        {
            std::atomic<bool> inside{ false };
            std::atomic<bool> release{ false };
            static void Run(void* context, uint32_t, uint32_t, uint32_t)
            {
                Hold* hold = static_cast<Hold*>(context);
                hold->inside.store(true);
                while (!hold->release.load())
                    std::this_thread::yield();
            }
        } hold;
        Gate k;
        const JobHandle kJob = Schedule(jobs, k.Desc());   // the worker spins in it
        std::thread a([&] { jobs->ParallelFor(1, 0, &Hold::Run, &hold); });
        while (!hold.inside.load())
            std::this_thread::yield();
        std::atomic<bool> waiting{ false };
        std::atomic<bool> done{ false };
        std::thread b([&] {
            waiting.store(true);
            jobs->Wait(kJob);
            done.store(true);
            hold.release.store(true);
        });
        while (!waiting.load())
            std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const bool doneEarly = done.load();   // B waits for the slot A holds
        k.open.store(true);
        b.join();
        a.join();
        CHECK(!doneEarly && done.load() && jobs->IsComplete(kJob));
    });
    CheckFullCapacity(jobs);
}

// One thread makes 600 jobs of 40 ranges runnable at once (they depend on a gate job that finishes on a worker): 9
// tickets each at 8 workers is 5,400, more than a deque's 4,096, so the later jobs get partial pushes and then only
// their first ticket, through the shared queue. Every other round holds the ranges until all jobs are runnable, so that
// nothing drains the deque meanwhile; the others run free, so jobs finish while tickets are still being pushed. Every
// index runs once, and the job table and edge pool are whole afterwards.
static void test_deque_overflow()
{
    constexpr uint32_t c_Jobs = 600;
    constexpr uint32_t c_Ranges = 40;
    struct Fan
    {
        std::atomic<bool> hold{ false };
        std::atomic<uint64_t> indices{ 0 };
        static void Run(void* context, uint32_t begin, uint32_t end, uint32_t)
        {
            Fan* fan = static_cast<Fan*>(context);
            while (fan->hold.load(std::memory_order_acquire))
                std::this_thread::yield();
            fan->indices.fetch_add(end - begin, std::memory_order_relaxed);
        }
    };
    nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(8);
    std::vector<JobHandle> handles(c_Jobs);
    for (uint32_t round = 0; round < 40; ++round)
    {
        const bool held = round % 2 == 0;
        Fan fan;
        fan.hold.store(held);
        Gate gate;
        // A lost slot shows as a failed Schedule here or in CheckFullCapacity; failures are counted, not thrown, so
        // that nothing stays pending behind the gate.
        uint32_t failed = 0;
        JobHandle root = JobHandle::Null;
        if (jobs->Schedule(gate.Desc(), nullptr, 0, root) != nvrhi::FS_OK)
            ++failed;
        uint64_t expected = 0;
        for (uint32_t i = 0; i < c_Jobs; ++i)
        {
            handles[i] = JobHandle::Null;
            if (jobs->Schedule(JobDesc{ &Fan::Run, &fan, c_Ranges, 1, "fan" }, &root, 1, handles[i]) == nvrhi::FS_OK)
                expected += c_Ranges;
            else
                ++failed;
        }
        gate.open.store(true);
        if (held)
        {
            // The root is finished once all 600 are runnable (and their tickets pushed or spilled).
            while (!jobs->IsComplete(root))
                std::this_thread::yield();
        }
        fan.hold.store(false);
        for (JobHandle handle : handles)
            jobs->Wait(handle);
        jobs->Wait(root);
        if (failed != 0)
            throw std::runtime_error("deque_overflow, round " + std::to_string(round) + ": " +
                                     std::to_string(failed) + " jobs did not fit the table: slots are lost");
        CHECK(fan.indices.load() == expected && expected == uint64_t(c_Jobs) * c_Ranges);
    }
    CheckFullCapacity(jobs);
}

// The CPU time of every thread of the process but the calling one: cycles on Windows (GetProcessTimes counts whole clock
// ticks), scaled by a busy loop of this thread, so that they are milliseconds at its speed.
namespace
{
#if defined(_WIN32)
    double CyclesPerMillisecond()
    {
        double best = 0.0;
        for (uint32_t attempt = 0; attempt < 5; ++attempt)
        {
            ULONG64 before = 0, after = 0;
            QueryThreadCycleTime(GetCurrentThread(), &before);
            const auto start = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(4))
                Spin(100);
            QueryThreadCycleTime(GetCurrentThread(), &after);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            best = std::max(best, double(after - before) / ms);   // a preempted attempt counts fewer
        }
        return best;
    }
    double OtherThreadsCpuMilliseconds()
    {
        static const double s_PerMillisecond = CyclesPerMillisecond();
        ULONG64 process = 0, own = 0;
        QueryThreadCycleTime(GetCurrentThread(), &own);
        QueryProcessCycleTime(GetCurrentProcess(), &process);
        return (double(process) - double(own)) / s_PerMillisecond;
    }
#else
    double OtherThreadsCpuMilliseconds()
    {
        timespec process{}, own{};
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &own);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process);
        return double(process.tv_sec - own.tv_sec) * 1e3 + double(process.tv_nsec - own.tv_nsec) * 1e-6;
    }
#endif

    // The workers' CPU time per worker and frame, in milliseconds. A frame: `run`, which wakes the workers, then a pause
    // in which they look for work and fall asleep. A sleeping worker costs nothing, so the time is the wake-up's and the
    // looks'.
    template <typename Frame>
    double IdleCpuPerWorkerAndFrame(uint32_t workers, uint32_t frames, Frame&& run)
    {
        OtherThreadsCpuMilliseconds();   // the scale
        const double before = OtherThreadsCpuMilliseconds();
        for (uint32_t frame = 0; frame < frames; ++frame)
        {
            run();
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        return (OtherThreadsCpuMilliseconds() - before) / (double(workers) * frames);
    }
}

// Idle workers sleep soon whatever a look for work costs (E5.8: the rounds of the spin took 1.5 ms with many slots, so
// that one small job per frame kept the workers busy for the whole frame). Only an upper bound on the CPU time, far
// above the spin's 30 us per worker and frame and far below a millisecond: a busy machine makes it smaller, not larger.
static void test_idle_workers_sleep()
{
    constexpr uint32_t c_Workers = 8;
    constexpr uint32_t c_Frames = 60;
    JobSystemDesc desc;   // the defaults of the shared system
    desc.debugName = "test_job_system";
    desc.workerThreadCount = c_Workers;
    desc.maxExternalThreads = 64;
    IJobSystem* system = nullptr;
    CHECK(CreateJobSystem(desc, &system) == nvrhi::FS_OK && system != nullptr);
    nvrhi::AutoPtr<IJobSystem> jobs = nvrhi::TakeOver(system);

    Counter counter;
    const double perWorkerFrame = IdleCpuPerWorkerAndFrame(c_Workers, c_Frames, [&] {
        jobs->ParallelFor(64, 1, &Counter::Run, &counter);   // a ticket for every worker, and next to nothing to do
    });
    std::printf("test_job_system: idle workers: %.3f ms of CPU per worker and frame\n", perWorkerFrame);
    CHECK(counter.value.load() == 64ull * c_Frames);
    CHECK(perWorkerFrame < 0.25);
    CheckFullCapacity(jobs);
}

// They look for work again after every sleep, whether or not they found any in between. A frame: one worker sleeps in a
// job, the caller sleeps in its Wait (spinCount 0), and the job's end wakes every worker. With idleSpinMicroseconds = 500
// they spend 0.5 ms of CPU each (a busy machine takes some of it away: the best of several attempts counts), workers that
// look only before their first sleep 0.03 ms, and workers that look again only after a job 0.1 ms.
static void test_idle_workers_spin_again()
{
    constexpr uint32_t c_Workers = 8;
    constexpr uint32_t c_Attempts = 20;
    constexpr double c_LowerBound = 0.15;
    nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(c_Workers, 2, 0, 500);

    struct SleepingJob
    {
        std::atomic<bool> started{ false };
        static void Run(void* context, uint32_t, uint32_t, uint32_t)
        {
            static_cast<SleepingJob*>(context)->started.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    auto run = [&] {
        SleepingJob sleeping;
        const JobHandle job = Schedule(jobs, JobDesc{ &SleepingJob::Run, &sleeping, 1, 0, "sleeping" });
        while (!sleeping.started.load())   // by a worker: this thread runs no job before its Wait
            std::this_thread::yield();
        jobs->Wait(job);
    };
    (void)IdleCpuPerWorkerAndFrame(c_Workers, 3, run);   // every worker has slept
    double best = 0.0;
    uint32_t attempts = 0;
    while (attempts < c_Attempts && best <= c_LowerBound)
    {
        best = std::max(best, IdleCpuPerWorkerAndFrame(c_Workers, 10, run));
        ++attempts;
    }
    std::printf("test_job_system: idle workers with a spin of 0.5 ms: %.3f ms of CPU per worker and frame, %u attempts\n",
                best, attempts);
    CHECK(best > c_LowerBound);
    CheckFullCapacity(jobs);
}

// The wake path: jobs that arrive while workers go to sleep or have just slept. Pauses from 0 to 60 us around the length
// of the spin, so that the last look for work, the announcement and the sleep all meet new tickets; each round a
// ParallelFor, a job scheduled by a thread without a slot and not waited for at first, and a dependency chain. A lost
// wake-up would leave the detached job unfinished: only a worker runs it, as nobody waits (a guard reports it).
static void test_wake_path()
{
    struct Spins
    {
        uint32_t spinCount, idleSpinMicroseconds;
    };
    for (const Spins spins : { Spins{ 4096, 30 }, Spins{ 1, 1 }, Spins{ 0, 0 } })
    {
        for (uint32_t workers : { 1u, 2u, 8u })
        {
            nvrhi::AutoPtr<IJobSystem> jobs = CreateSystem(workers, 2, spins.spinCount, spins.idleSpinMicroseconds);
            Counter counter;
            Random random(0x5eed + workers + spins.spinCount);
            uint64_t expected = 0;
            for (uint32_t round = 0; round < 2000; ++round)
            {
                const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(random.Below(61));
                while (std::chrono::steady_clock::now() < end)
                    Spin(10);
                // no Wait: the workers alone run these
                const JobHandle first = Schedule(jobs, JobDesc{ &Counter::Run, &counter, 4, 1, "detached" });
                const JobHandle second = Schedule(jobs, JobDesc{ &Counter::Run, &counter, 1, 0, "chained" }, &first, 1);
                expected += 5;
                if (round % 3 == 0)
                {
                    jobs->ParallelFor(16, 1, &Counter::Run, &counter);
                    expected += 16;
                }
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                // a job is marked finished after its dependents were released, so the second may be complete first
                while (!jobs->IsComplete(first) || !jobs->IsComplete(second))
                {
                    CHECK(std::chrono::steady_clock::now() < deadline);   // a lost wake-up
                    std::this_thread::yield();
                }
                CHECK(counter.value.load() == expected);
            }
            CheckFullCapacity(jobs);
        }
    }
}

int main(int, char**)
{
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "creation_and_indices", &test_creation_and_indices },
        { "parallel_for_split", &test_parallel_for_split },
        { "dependency_graphs", &test_dependency_graphs },
        { "nested_waits", &test_nested_waits },
        { "external_threads", &test_external_threads },
        { "table_full", &test_table_full },
        { "shutdown", &test_shutdown },
        { "steady_state_allocations", &test_steady_state_allocations },
        { "stress", &test_stress },
        { "wait_on_own_stack", &test_wait_on_own_stack },
        { "wait_for_slot_ends", &test_wait_for_slot_ends },
        { "deque_overflow", &test_deque_overflow },
        { "idle_workers_sleep", &test_idle_workers_sleep },
        { "idle_workers_spin_again", &test_idle_workers_spin_again },
        { "wake_path", &test_wake_path },
    };
    std::thread watchdog(&Watchdog);
    int result = 0;
    try
    {
        for (const Test& test : tests)
        {
            g_CaseName.store(test.name);
            g_CaseStart.store(NowMilliseconds());
            const int64_t start = NowMilliseconds();
            test.run();
            std::printf("test_job_system: %s passed (%lld ms)\n", test.name, (long long)(NowMilliseconds() - start));
        }
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "%s\n", err.what());
        result = 1;
    }
    g_AllDone.store(true);
    watchdog.join();
    return result;
}
