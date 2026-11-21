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

// rdg_bench: the device-free budgets of design 10.3.
//
//     rdg_bench [--quick] [--gate] [--profile d3d12|vulkan]
//     rdg_bench --parallel d3d12|vulkan [--quick] [--gate]
//
// Reports, as medians:
// - compile on a cache miss, 64 passes / 256 resources (budget 1 ms) and 1024 passes / 4096 resources (16 ms);
// - the frame cost on a cache hit: Reset, recording and the plan lookup at Compile, 64 passes (0.05 ms);
// - the frame cost without the cache (maxCachedPlans = 0: recording and a full compile), beside it, not gated;
// - heap allocations per steady-state frame (0).
// --gate exits with 1 when a gated budget is missed (Release builds; Debug numbers are not comparable).
// --parallel measures the parallel-recording budget on a device instead (rdg_bench_parallel.cpp).

#include "rdg_corpus.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

using namespace donut::render;
using namespace rdgtest;

int RunParallelBench(const char* backend, bool quick, bool gate);   // rdg_bench_parallel.cpp

static std::atomic<bool> g_CountAllocations{ false };
static std::atomic<uint64_t> g_Allocations{ 0 };

void* operator new(size_t size)
{
    if (g_CountAllocations.load(std::memory_order_relaxed))
        g_Allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size ? size : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](size_t size) { return operator new(size); }
void* operator new(size_t size, const std::nothrow_t&) noexcept
{
    if (g_CountAllocations.load(std::memory_order_relaxed))
        g_Allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size ? size : 1);
}
void* operator new[](size_t size, const std::nothrow_t& tag) noexcept { return operator new(size, tag); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

namespace
{
    using Clock = std::chrono::steady_clock;

    double Microseconds(Clock::time_point start, Clock::time_point end)
    {
        return std::chrono::duration<double, std::micro>(end - start).count();
    }

    double Median(std::vector<double> samples)
    {
        if (samples.empty())
            return 0;
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }

    struct Graph
    {
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;

        Graph()
        {
            IResourceRegistry* r = nullptr;
            CreateResourceRegistry(nullptr, &r);
            registry = nvrhi::TakeOver(r);
            IRenderGraph* g = nullptr;
            CreateRenderGraph(registry.Get(), &g);
            graph = nvrhi::TakeOver(g);
        }
    };

    bool CompileFrame(IRenderGraph* graph, const BackendProfile& profile, const CompileOptions& options)
    {
        ICompiledRenderGraph* plan = nullptr;
        const nvrhi::FRESULT result = graph->Compile(profile, options, &plan);
        if (plan)
            plan->Release();
        if (result != nvrhi::FS_OK)
        {
            nvrhi::string diagnostics;
            graph->GetDiagnostics(diagnostics);
            std::fprintf(stderr, "compile failed:\n%s\n", diagnostics.c_str());
            return false;
        }
        return true;
    }

    // Compile on a miss: the cache is cleared before each compile; recording is not timed.
    double CompileMiss(const BackendProfile& profile, uint32_t passes, uint32_t iterations)
    {
        Graph g;
        BenchSetup setup;
        setup.Init(passes, 4);
        std::vector<double> samples;
        for (uint32_t i = 0; i < iterations; ++i)
        {
            g.graph->Reset();
            RecordBenchFrame(g.graph, setup);
            g.registry->ClearPlanCache();
            const auto start = Clock::now();
            if (!CompileFrame(g.graph, profile, CompileOptions()))
                return -1;
            samples.push_back(Microseconds(start, Clock::now()));
        }
        return Median(samples);
    }

    // A whole frame of the graph: Reset, recording and Compile; with the cache, a hit.
    double FrameCost(const BackendProfile& profile, uint32_t passes, uint32_t iterations, uint32_t maxCachedPlans,
                     uint64_t* allocationsPerFrame)
    {
        Graph g;
        BenchSetup setup;
        setup.Init(passes, 4);
        CompileOptions options;
        options.maxCachedPlans = maxCachedPlans;
        for (uint32_t i = 0; i < 3; ++i)
        {
            g.graph->Reset();
            RecordBenchFrame(g.graph, setup);
            if (!CompileFrame(g.graph, profile, options))
                return -1;
        }
        std::vector<double> samples;
        samples.reserve(iterations);   // the counted loop must not grow it
        g_Allocations = 0;
        g_CountAllocations = allocationsPerFrame != nullptr;
        for (uint32_t i = 0; i < iterations; ++i)
        {
            const auto start = Clock::now();
            g.graph->Reset();
            RecordBenchFrame(g.graph, setup);
            CompileFrame(g.graph, profile, options);
            samples.push_back(Microseconds(start, Clock::now()));
        }
        g_CountAllocations = false;
        if (allocationsPerFrame)
            *allocationsPerFrame = iterations ? g_Allocations.load() / iterations : 0;
        return Median(samples);
    }
}

int main(int argc, char** argv)
{
    bool quick = false;
    bool gate = false;
    const char* parallelBackend = nullptr;
    ReferenceProfile kind = ReferenceProfile::D3D12Legacy;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--quick"))
            quick = true;
        else if (!std::strcmp(argv[i], "--gate"))
            gate = true;
        else if (!std::strcmp(argv[i], "--parallel") && i + 1 < argc)
            parallelBackend = argv[++i];
        else if (!std::strcmp(argv[i], "--profile") && i + 1 < argc)
        {
            ++i;
            kind = !std::strcmp(argv[i], "vulkan") ? ReferenceProfile::Vulkan : ReferenceProfile::D3D12Legacy;
        }
    }
    if (parallelBackend)
        return RunParallelBench(parallelBackend, quick, gate);

    BackendProfile profile;
    GetReferenceProfile(kind, &profile);

    const uint32_t many = quick ? 3 : 201;
    const uint32_t few = quick ? 1 : 21;

    const double miss64 = CompileMiss(profile, 64, many);
    const double miss1024 = CompileMiss(profile, 1024, few);
    uint64_t allocations = 0;
    const double hit64 = FrameCost(profile, 64, many, 8, &allocations);
    const double noCache64 = FrameCost(profile, 64, many, 0, nullptr);
    if (miss64 < 0 || miss1024 < 0 || hit64 < 0 || noCache64 < 0)
        return 1;

#ifdef NDEBUG
    const char* build = "Release";
#else
    const char* build = "Debug";
#endif
    std::printf("rdg_bench (%s build, %s profile, medians)\n", build, kind == ReferenceProfile::Vulkan ? "Vulkan" : "D3D12 legacy");
    std::printf("  %-58s %10.1f us   budget %8.1f us\n", "compile, cache miss, 64 passes / 256 resources", miss64, 1000.0);
    std::printf("  %-58s %10.1f us   budget %8.1f us\n", "compile, cache miss, 1024 passes / 4096 resources", miss1024, 16000.0);
    std::printf("  %-58s %10.1f us   budget %8.1f us\n", "frame on a cache hit (record + Compile), 64 passes", hit64, 50.0);
    std::printf("  %-58s %10.1f us   (reported, not gated)\n", "frame without the cache (record + full compile), 64 passes", noCache64);
    std::printf("  %-58s %10llu      budget %8d\n", "heap allocations per steady-state frame", (unsigned long long)allocations, 0);

    if (gate)
    {
        const bool pass = miss64 <= 1000.0 && miss1024 <= 16000.0 && hit64 <= 50.0 && allocations == 0;
        std::printf("%s\n", pass ? "all budgets met" : "BUDGET MISSED");
        return pass ? 0 : 1;
    }
    return allocations == 0 ? 0 : 1;
}
