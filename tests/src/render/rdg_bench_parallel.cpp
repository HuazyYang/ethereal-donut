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

// rdg_bench --parallel d3d12|vulkan: the parallel-recording budget of design 10.3 (aggregate plan, step P1.7). It
// needs a device, unlike the rest of rdg_bench.
//
// A frame is 64 passes: a cleared seed, 62 blits through BlitPass (one object each, sharing CommonRenderPasses) and a
// copy out. The recording time is RunStatistics::recordMilliseconds: patches, Prepare and the command recording. It is
// measured on one list (no scheduler, one segment) and on four threads (RenderGraphWorkerPool with three workers and the caller,
// 16 passes per segment). Protocol 10.4: Release, the NVIDIA adapter, no validation layers, 100 warm-up frames per
// mode, then 300 frames per mode in alternating blocks of 50; medians. The budget: four threads take at most 60 % of
// the single list's time. It is a budget for recording with awake workers: each frame waits for the device to be idle
// before the next one records, so the pool's workers get a longer idle spin than its default (c_IdleSpinMicroseconds).
// With the default of 30 us they sleep in between, and the wake-ups count as recording time.

#include "rdg_corpus.h"
#include <donut/render/RenderGraphWorkerPool.h>

#include <donut/core/vfs/VFS.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/render/BlitPass.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphLambdaPass.h>

#include <nvrhi/nvrhi.h>
#if RDG_BENCH_D3D12
#include <nvrhi/d3d12.h>
#endif
#if RDG_BENCH_VULKAN
#include <nvrhi/vulkan.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace donut::render;
using rdgtest::NoData;
using rdgtest::Read;
using rdgtest::Write;

namespace
{
    std::filesystem::path ExecutableDirectory()
    {
#ifdef _WIN32
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, path, MAX_PATH);
        return std::filesystem::path(path).parent_path();
#else
        return std::filesystem::current_path();
#endif
    }

    template <typename TFactory>
    int32_t FindNvidiaAdapter(TFactory* factory)
    {
        for (uint32_t index = 0; index < factory->getAdapterCount(); ++index)
        {
            nvrhi::AdapterInfo info;
            if (factory->getAdapterInfo(index, info) == nvrhi::FS_OK && std::string(info.name.c_str()).find("NVIDIA") != std::string::npos)
                return int32_t(index);
        }
        return -1;
    }

    struct Bench
    {
        nvrhi::IDevice* device = nullptr;
        nvrhi::AutoPtr<donut::vfs::RootFileSystem> rootFS;
        nvrhi::AutoPtr<donut::engine::ShaderFactory> shaderFactory;
        nvrhi::AutoPtr<donut::engine::CommonRenderPasses> commonPasses;
        std::vector<nvrhi::AutoPtr<BlitPass>> blits;
        nvrhi::AutoPtr<nvrhi::ITexture> result;
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;
        BackendProfile profile;

        nvrhi::TextureDesc TargetDesc(const char* name) const
        {
            return nvrhi::TextureDesc().setWidth(64).setHeight(64).setFormat(nvrhi::Format::RGBA8_UNORM).setIsRenderTarget(true)
                .setDebugName(name);
        }

        bool Init(nvrhi::IDevice* device_, const char* shaderType)
        {
            device = device_;
            rootFS = MAKE_RC_OBJ_PTR(donut::vfs::RootFileSystem);
            rootFS->mount("/shaders/donut", ExecutableDirectory() / "shaders/framework" / shaderType);
            shaderFactory = MAKE_RC_OBJ_PTR(donut::engine::ShaderFactory, device, rootFS, "/shaders");
            commonPasses = MAKE_RC_OBJ_PTR(donut::engine::CommonRenderPasses, device, shaderFactory);
            for (uint32_t i = 0; i < 62; ++i)
                blits.push_back(MAKE_RC_OBJ_PTR(BlitPass, device, commonPasses));
            device->createTexture(TargetDesc("result"), result.ReleaseAndGetAddressOf());
            nvrhi::AutoPtr<nvrhi::ICommandList> list;
            device->createCommandList(nvrhi::CommandListParameters(), list.ReleaseAndGetAddressOf());
            if (!result || !list)
                return false;
            // The import's entry state: CopySource.
            list->open();
            list->setEnableAutomaticBarriers(false);
            const nvrhi::ResourceBarrier entry = nvrhi::ResourceBarrier::Transition(result, nvrhi::ResourceStates::Unknown,
                                                                                    nvrhi::ResourceStates::CopySource);
            list->barriers(&entry, 1);
            list->close();
            device->executeCommandList(list);
            device->waitForIdle();
            IResourceRegistry* r = nullptr;
            IRenderGraph* g = nullptr;
            if (CreateResourceRegistry(device, &r) != nvrhi::FS_OK)
                return false;
            registry = nvrhi::TakeOver(r);
            if (CreateRenderGraph(r, &g) != nvrhi::FS_OK)
                return false;
            graph = nvrhi::TakeOver(g);
            return ComputeBackendProfile(device, &profile) == nvrhi::FS_OK;
        }

        // One frame; returns its recording time in microseconds, or a negative value on failure.
        double Frame(donut::render::RenderGraphWorkerPool* pool, uint32_t granularity)
        {
            graph->Reset();
            static const ClearValue c_Seed = { { 0.5f, 0.25f, 0.f, 1.f }, 0.f, 0 };
            ResourceId current = graph->CreateTexture(TargetDesc("Seed"), ResourceFlags::None);
            AddLambdaPass<NoData>(graph, PassDesc{ "Seed", PassType::Raster, PassFlags::None, 0 },
                [&current](ISetupContext* s, NoData&) {
                    current = s->SetColorTarget(0, current, nvrhi::SubresourceRegion(), AccessFlags::Clear, &c_Seed);
                },
                [](IPassContext*, const NoData&) { });
            for (BlitPass* blit : blits)
            {
                BlitGraphInputs inputs;
                inputs.source = current;
                inputs.target = graph->CreateTexture(TargetDesc("Blit"), ResourceFlags::None);
                inputs.sampler = donut::engine::BlitSampler::Point;
                current = blit->AddToGraph(graph, inputs).target;
            }
            ResourceId out = graph->ImportTexture(result, nvrhi::ResourceStates::CopySource, nvrhi::ResourceStates::CopySource);
            const ResourceId last = current;
            AddLambdaPass<NoData>(graph, PassDesc{ "Save", PassType::Copy, PassFlags::None, 0 },
                [&](ISetupContext* s, NoData&) {
                    Read(s, last, Access::CopySource);
                    out = Write(s, out, Access::CopyDest);
                },
                [last, out](IPassContext* context, const NoData&) {
                    context->GetCommandList()->copyTexture1(context->GetTexture(out), nvrhi::TextureSlice(), context->GetTexture(last),
                                                            nvrhi::TextureSlice());
                });

            CompileOptions options;
            options.recordGranularity = granularity;
            ICompiledRenderGraph* plan = nullptr;
            if (graph->Compile(profile, options, &plan) != nvrhi::FS_OK)
            {
                nvrhi::string diagnostics;
                graph->GetDiagnostics(diagnostics);
                std::fprintf(stderr, "compile failed:\n%s\n", diagnostics.c_str());
                return -1;
            }
            ExecuteDesc execute;
            if (pool)
            {
                execute.parallelFor = &donut::render::RenderGraphWorkerPool::ParallelFor;
                execute.schedulerContext = pool;
                execute.workerCount = pool->ThreadCount();
            }
            const nvrhi::FRESULT executed = graph->Execute(plan, execute);
            plan->Release();
            device->waitForIdle();
            device->runGarbageCollection();
            if (executed != nvrhi::FS_OK)
                return -1;
            RunStatistics run;
            registry->GetRunStatistics(run);
            return double(run.recordMilliseconds) * 1000.0;
        }
    };

    double Median(std::vector<double> samples)
    {
        std::sort(samples.begin(), samples.end());
        return samples.empty() ? 0.0 : samples[samples.size() / 2];
    }

    constexpr uint32_t c_IdleSpinMicroseconds = 480;

    int Measure(nvrhi::IDevice* device, const char* api, const char* shaderType, bool quick, bool gate)
    {
        Bench bench;
        if (!bench.Init(device, shaderType))
        {
            std::fprintf(stderr, "rdg_bench --parallel: setup failed\n");
            return 1;
        }
        donut::render::RenderGraphWorkerPool pool(3, c_IdleSpinMicroseconds);
        const uint32_t warmup = quick ? 5 : 100;
        const uint32_t blocks = quick ? 1 : 6;
        const uint32_t perBlock = quick ? 5 : 50;
        for (uint32_t i = 0; i < warmup; ++i)
        {
            if (bench.Frame(nullptr, 0) < 0 || bench.Frame(&pool, 16) < 0)
                return 1;
        }
        std::vector<double> single, parallel;
        for (uint32_t block = 0; block < blocks; ++block)
        {
            for (uint32_t i = 0; i < perBlock; ++i)
                single.push_back(bench.Frame(nullptr, 0));
            for (uint32_t i = 0; i < perBlock; ++i)
                parallel.push_back(bench.Frame(&pool, 16));
        }
        RunStatistics run;
        bench.registry->GetRunStatistics(run);
        const double singleMedian = Median(single);
        const double parallelMedian = Median(parallel);
        const double ratio = singleMedian > 0 ? parallelMedian / singleMedian : 0;
#ifdef NDEBUG
        const char* build = "Release";
#else
        const char* build = "Debug";
#endif
        std::printf("rdg_bench --parallel (%s build, %s, medians of %u frames per mode)\n", build, api, blocks * perBlock);
        std::printf("  %-58s %10.1f us\n", "recording, 64 passes, one list", singleMedian);
        std::printf("  %-58s %10.1f us   (%u segments, %u tasks)\n", "recording, 64 passes, 4 threads", parallelMedian, run.segments,
                    run.recordTasks);
        std::printf("  %-58s %10.1f %%    budget %8.1f %%\n", "4 threads against one list", ratio * 100.0, 60.0);
        if (!gate)
            return 0;
        const bool pass = ratio > 0 && ratio <= 0.6 && run.recordTasks == pool.ThreadCount();
        std::printf("%s\n", pass ? "parallel-recording budget met" : "BUDGET MISSED");
        return pass ? 0 : 1;
    }
}

int RunParallelBench(const char* backend, bool quick, bool gate)
{
    const std::string api = backend ? backend : "";
#if RDG_BENCH_D3D12
    if (api == "d3d12")
    {
        nvrhi::d3d12::GraphicsFactoryDesc factoryDesc;
        nvrhi::d3d12::GraphicsFactoryHandle factory = nvrhi::d3d12::createGraphicsFactory(factoryDesc);
        if (!factory)
            return 1;
        nvrhi::d3d12::DeviceCreationDesc deviceDesc;
        deviceDesc.enableComputeQueue = true;
        deviceDesc.enableCopyQueue = true;
        deviceDesc.adapterIndex = FindNvidiaAdapter(factory.Get());
        nvrhi::d3d12::DeviceHandle device;
        if (factory->createDeviceOnAdapter(deviceDesc, &device) != nvrhi::FS_OK)
            return 1;
        return Measure(device, "D3D12", "dxil", quick, gate);
    }
#endif
#if RDG_BENCH_VULKAN
    if (api == "vulkan")
    {
        nvrhi::vulkan::GraphicsFactoryDesc factoryDesc;
        factoryDesc.headless = true;
        nvrhi::vulkan::GraphicsFactoryHandle factory = nvrhi::vulkan::createGraphicsFactory(factoryDesc);
        if (!factory)
            return 1;
        nvrhi::vulkan::DeviceCreationDesc deviceDesc;
        deviceDesc.enableComputeQueue = true;
        deviceDesc.enableCopyQueue = true;
        deviceDesc.adapterIndex = FindNvidiaAdapter(factory.Get());
        nvrhi::vulkan::DeviceHandle device;
        if (factory->createDeviceOnAdapter(deviceDesc, &device) != nvrhi::FS_OK)
            return 1;
        const int result = Measure(device, "Vulkan", "spirv", quick, gate);
        device->waitForIdle();
        return result;
    }
#endif
    std::fprintf(stderr, "rdg_bench --parallel: backend '%s' is not built\n", api.c_str());
    return 1;
}
