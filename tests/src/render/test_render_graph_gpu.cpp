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

// Headless graphs executed on a device (aggregate plan, step P1.4), under the backend's validation:
//
//     test_render_graph_gpu d3d11 | d3d12 | vulkan | cuda
//
// The D3D12 debug layer with GPU-based validation and the Vulkan validation layer with synchronization validation are
// confirmed loaded; the nvrhi validation layer wraps the device. Copies and clears only, so that no shader is needed:
// the results are read back and checked, and every frame's barriers are under validation. The steady-state test runs
// on the device without the nvrhi validation layer and counts heap allocations against a hand-written baseline.

#include "rdg_corpus.h"

#include <nvrhi/nvrhi.h>
#include <nvrhi/factory.h>
#include <nvrhi/core/foundation.h>
#if TEST_D3D11
#include <nvrhi/d3d11.h>
#endif
#if TEST_D3D12
#include <nvrhi/d3d12.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#endif
#if TEST_VULKAN
#include <nvrhi/vulkan.h>
#endif
#if TEST_CUDA
#include <nvrhi/cuda.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

using namespace donut::render;
using namespace rdgtest;
using nvrhi::Format;
using nvrhi::ResourceStates;

// ---------------------------------------------------------------------------------------------- allocation hook

// Counts operator new in this module: the graph, the executor and nvrhi are linked in statically. The native runtimes
// and their debug layers allocate in their own modules and are not counted.
static std::atomic<bool> g_CountAllocations{ false };
static std::atomic<uint64_t> g_Allocations{ 0 };
// RDG_ALLOC_TRACE=1: the steady-state test prints the call stack of every allocation in one measured frame of each
// loop. A diagnostic for a failed allocation check.
static std::atomic<bool> g_TraceAllocations{ false };

#ifdef _WIN32
static void TraceAllocation(size_t size)
{
    static thread_local bool inside = false;
    if (inside)
        return;
    inside = true;
    void* frames[20];
    const USHORT count = CaptureStackBackTrace(2, 20, frames, nullptr);
    std::printf("  allocation of %zu bytes:", size);
    alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(storage);
    for (USHORT i = 0; i < count; ++i)
    {
        std::memset(storage, 0, sizeof(storage));
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 255;
        DWORD64 displacement = 0;
        if (SymFromAddr(GetCurrentProcess(), DWORD64(frames[i]), &displacement, symbol))
            std::printf("\n      %s", symbol->Name);
    }
    std::printf("\n");
    inside = false;
}
#else
static void TraceAllocation(size_t) { }
#endif

void* operator new(size_t size)
{
    if (g_CountAllocations.load(std::memory_order_relaxed))
    {
        g_Allocations.fetch_add(1, std::memory_order_relaxed);
        if (g_TraceAllocations.load(std::memory_order_relaxed))
            TraceAllocation(size);
    }
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
    void check(bool condition, const std::string& what)
    {
        if (!condition)
            throw std::runtime_error("check failed: " + what);
        std::printf("ok: %s\n", what.c_str());
    }

    struct Messages : nvrhi::ObjectImpl<nvrhi::IMessageCallback>
    {
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Messages)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IMessageCallback)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IRHIObject)
        NVRHI_END_INTERFACE_TABLE()

        unsigned errors = 0;
        unsigned expectedErrorsSeen = 0;
        bool expecting = false;

        void message(nvrhi::MessageSeverity severity, const char* text) noexcept override
        {
            const bool error = severity == nvrhi::MessageSeverity::Error || severity == nvrhi::MessageSeverity::Fatal;
            if (error && expecting)
            {
                ++expectedErrorsSeen;
                std::printf("  (expected) NVRHI: %s\n", text);
                return;
            }
            if (error)
                ++errors;
            if (severity != nvrhi::MessageSeverity::Info)
                std::printf("NVRHI: %s\n", text);
        }
    };

    struct Env
    {
        nvrhi::GraphicsAPI api = nvrhi::GraphicsAPI::D3D12;
        nvrhi::IDevice* device = nullptr;      // the nvrhi validation layer
        nvrhi::IDevice* rawDevice = nullptr;   // the backend device under it
        Messages* messages = nullptr;
        bool raster = true;                    // CUDA has no raster passes
        bool explicitScopes = false;           // D3D12 and Vulkan record real barriers in explicit-barrier scopes
        std::function<unsigned()> nativeErrors = [] { return 0u; };
    };

    void checkClean(Env& env, const std::string& phase)
    {
        env.device->waitForIdle();
        const unsigned native = env.nativeErrors();
        check(env.messages->errors == 0 && native == 0, phase + ": no NVRHI or debug-layer errors");
    }

    nvrhi::AutoPtr<nvrhi::ICommandList> CreateList(nvrhi::IDevice* device)
    {
        // D3D11 has only the immediate context; the other backends record deferred lists.
        const bool immediate = device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D11;
        nvrhi::AutoPtr<nvrhi::ICommandList> list;
        device->createCommandList(nvrhi::CommandListParameters().setEnableImmediateExecution(immediate), list.ReleaseAndGetAddressOf());
        check(list != nullptr, "command list");
        return list;
    }

    void Submit(nvrhi::IDevice* device, nvrhi::ICommandList* list)
    {
        nvrhi::ICommandList* lists[] = { list };
        device->submit(nvrhi::SubmitDesc().setCommandLists(lists, 1));
        device->waitForIdle();
        device->runGarbageCollection();
    }

    // Puts a new texture or buffer into `state` with an explicit barrier, so that its import's entry state is true.
    void InitialState(Env& env, nvrhi::IResource* resource, ResourceStates state)
    {
        nvrhi::AutoPtr<nvrhi::ICommandList> list = CreateList(env.device);
        list->open();
        list->setEnableAutomaticBarriers(false);
        const nvrhi::ResourceBarrier barrier = nvrhi::ResourceBarrier::Transition(resource, ResourceStates::Unknown, state);
        list->barriers(&barrier, 1);
        list->close();
        Submit(env.device, list);
    }

    // A registry and its graph on one device.
    struct GraphOnDevice
    {
        nvrhi::IDevice* device;
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;
        BackendProfile profile;

        explicit GraphOnDevice(nvrhi::IDevice* device_) : device(device_)
        {
            IResourceRegistry* r = nullptr;
            check(CreateResourceRegistry(device, &r) == nvrhi::FS_OK, "registry");
            registry = nvrhi::TakeOver(r);
            IRenderGraph* g = nullptr;
            check(CreateRenderGraph(r, &g) == nvrhi::FS_OK, "graph");
            graph = nvrhi::TakeOver(g);
            check(ComputeBackendProfile(device, &profile) == nvrhi::FS_OK, "backend profile");
        }

        uint64_t compileAllocations = 0;   // in the last Run, while the allocation hook counts
        uint64_t executeAllocations = 0;

        // Compiles and executes the recorded frame; returns the plan's statistics.
        PlanStatistics Run(const CompileOptions& options = CompileOptions())
        {
            ICompiledRenderGraph* plan = nullptr;
            const uint64_t start = g_Allocations;
            const nvrhi::FRESULT compiled = graph->Compile(profile, options, &plan);
            const uint64_t afterCompile = g_Allocations;
            if (compiled != nvrhi::FS_OK)
            {
                nvrhi::string diagnostics;
                graph->GetDiagnostics(diagnostics);
                throw std::runtime_error(std::string("compile failed:\n") + diagnostics.c_str());
            }
            const nvrhi::FRESULT executed = graph->Execute(plan, ExecuteDesc());
            compileAllocations = afterCompile - start;
            executeAllocations = g_Allocations - afterCompile;
            // The host's once-per-frame collection; the steady-state test collects outside its measurement.
            if (!g_CountAllocations)
                device->runGarbageCollection();
            PlanStatistics stats;
            plan->GetStatistics(stats);
            plan->Release();
            if (executed != nvrhi::FS_OK)
                throw std::runtime_error("execute failed: " + std::to_string(executed));
            return stats;
        }
    };

    nvrhi::BufferDesc GraphBufferDesc(const char* name, uint64_t size)
    {
        return nvrhi::BufferDesc().setByteSize(size).setCanHaveUAVs(true).setCanHaveRawViews(true).setDebugName(name);
    }

    nvrhi::AutoPtr<nvrhi::IBuffer> CreateReadback(nvrhi::IDevice* device)
    {
        nvrhi::AutoPtr<nvrhi::IBuffer> buffer;
        device->createBuffer(nvrhi::BufferDesc().setByteSize(256).setCpuAccess(nvrhi::CpuAccessMode::Read)
            .setInitialState(ResourceStates::CopyDest).setDebugName("readback"), buffer.ReleaseAndGetAddressOf());
        check(buffer != nullptr, "readback buffer");
        return buffer;
    }

    // A pass from a declaration lambda and a body; the body runs inside a noexcept Execute and must not throw.
    template <typename Setup, typename Body>
    PassId AddPass(IRenderGraph* graph, const char* name, PassType type, PassFlags flags, Setup&& setup, Body&& body)
    {
        PassId id = InvalidPassId;
        AddLambdaPass<NoData>(graph, PassDesc{ name, type, flags, 0 },
            [&setup](ISetupContext* context, NoData&) { setup(context); },
            [body](IPassContext* context, const NoData&) { body(context); },
            &id);
        return id;
    }

    // ------------------------------------------------------------------------------------------ buffers

    struct BufferChain
    {
        nvrhi::AutoPtr<nvrhi::IBuffer> readback;
        uint32_t pattern[32];
    };

    // Fill (writeBuffer) -> copy -> readback, with a cleared buffer read back beside it. `asyncFill` puts the fill on
    // the compute queue; `extraPass` adds a second copy that a mask switches off when `extraEnabled` is false.
    void RecordBufferChain(IRenderGraph* graph, BufferChain& chain, uint32_t frame, bool asyncFill, bool extraPass, bool extraEnabled)
    {
        for (uint32_t i = 0; i < 32; ++i)
            chain.pattern[i] = frame * 1000u + i;
        ResourceId a = graph->CreateBuffer(GraphBufferDesc("A", 128), ResourceFlags::None);
        ResourceId b = graph->CreateBuffer(GraphBufferDesc("B", 128), ResourceFlags::None);
        ResourceId c = graph->CreateBuffer(GraphBufferDesc("C", 128), ResourceFlags::None);
        ResourceId out = graph->ImportBuffer(chain.readback, ResourceStates::CopyDest, ResourceStates::CopyDest);
        const uint32_t* pattern = chain.pattern;

        AddPass(graph, "Fill", PassType::Compute, asyncFill ? PassFlags::AsyncCompute : PassFlags::None,
            [&](ISetupContext* s) { a = Write(s, a, Access::CopyDest); },
            [a, pattern](IPassContext* context) { context->GetCommandList()->writeBuffer(context->GetBuffer(a), pattern, 128); });
        AddPass(graph, "Copy", PassType::Copy, PassFlags::None,
            [&](ISetupContext* s) { Read(s, a, Access::CopySource); b = Write(s, b, Access::CopyDest); },
            [a, b](IPassContext* context) { context->GetCommandList()->copyBuffer(context->GetBuffer(b), 0, context->GetBuffer(a), 0, 128); });
        PassId extra = InvalidPassId;
        if (extraPass)
        {
            // Disabled, it passes B through: the readback still sees the first copy.
            extra = AddPass(graph, "CopyAgain", PassType::Copy, PassFlags::None,
                [&](ISetupContext* s) {
                    const ResourceId input = b;
                    Read(s, a, Access::CopySource);
                    b = Write(s, b, Access::CopyDest);
                    s->SetDisabledOutput(b, DisabledOutput::Passthrough, input);
                },
                [a, b](IPassContext* context) { context->GetCommandList()->copyBuffer(context->GetBuffer(b), 0, context->GetBuffer(a), 0, 128); });
        }
        AddPass(graph, "Clear", PassType::Compute, PassFlags::None,
            [&](ISetupContext* s) { c = Write(s, c, Access::UAVWrite, AccessFlags::Clear); },
            [](IPassContext*) { });
        AddPass(graph, "Readback", PassType::Copy, PassFlags::None,
            [&](ISetupContext* s) { Read(s, b, Access::CopySource); Read(s, c, Access::CopySource); out = Write(s, out, Access::CopyDest); },
            [b, c, out](IPassContext* context) {
                nvrhi::ICommandList* list = context->GetCommandList();
                list->copyBuffer(context->GetBuffer(out), 0, context->GetBuffer(b), 0, 128);
                list->copyBuffer(context->GetBuffer(out), 128, context->GetBuffer(c), 0, 128);
            });
        if (extraPass && !extraEnabled)
            graph->SetPassEnabled(extra, false);
    }

    void CheckReadback(nvrhi::IDevice* device, BufferChain& chain, const std::string& what)
    {
        device->waitForIdle();
        const uint32_t* data = static_cast<const uint32_t*>(device->mapBuffer(chain.readback, nvrhi::CpuAccessMode::Read));
        check(data != nullptr, "map readback");
        bool good = true;
        for (uint32_t i = 0; i < 32; ++i)
            good = good && data[i] == chain.pattern[i] && data[32 + i] == 0;
        if (!good)
            std::printf("  read %u %u ... %u, expected %u %u ... 0\n", data[0], data[1], data[32], chain.pattern[0], chain.pattern[1]);
        device->unmapBuffer(chain.readback);
        check(good, what);
    }

    void testBufferChains(Env& env)
    {
        for (int async = 0; async < 2; ++async)
        {
            const std::string prefix = async ? "async fill: " : "one queue: ";
            GraphOnDevice g(env.device);
            BufferChain chain;
            chain.readback = CreateReadback(env.device);
            for (uint32_t frame = 0; frame < 8; ++frame)
            {
                g.graph->Reset();
                RecordBufferChain(g.graph, chain, frame, async != 0, false, false);
                const PlanStatistics stats = g.Run();
                if (frame == 0)
                    std::printf("  %s%u barriers, %u segments, %u fence waits, %u transfers\n", prefix.c_str(), stats.barriers,
                                stats.segments, stats.fenceWaits, stats.queueTransfers);
                CheckReadback(env.device, chain, prefix + "buffer chain, frame " + std::to_string(frame));
            }
            checkClean(env, prefix + "buffer chain");
        }
    }

    // 6.12: masks and an async variant alternate; every switch patches states and waits for the previous frame.
    void testPlanSwitches(Env& env)
    {
        GraphOnDevice g(env.device);
        BufferChain chain;
        chain.readback = CreateReadback(env.device);
        for (uint32_t frame = 0; frame < 12; ++frame)
        {
            g.graph->Reset();
            RecordBufferChain(g.graph, chain, frame, (frame / 4) % 2 == 1, true, frame % 2 == 0);
            g.Run();
            // Every other frame is checked after a switch that was submitted without waiting for idle.
            if (frame % 2 == 1)
                CheckReadback(env.device, chain, "plan switch, frame " + std::to_string(frame));
        }
        RegistryStatistics stats;
        g.registry->GetRegistryStatistics(stats);
        check(stats.cachedPlans >= 4, "four plans cached (" + std::to_string(stats.cachedPlans) + ")");
        checkClean(env, "plan switches");
    }

    // 5.2, 6.12: a persistent buffer carries its contents and states across frames, plan switches, ReleaseResources,
    // and a ReleasePersistent that reallocates it. Each frame reads last frame's value, then writes its own.
    void testPersistent(Env& env)
    {
        GraphOnDevice g(env.device);
        BufferChain chain;
        chain.readback = CreateReadback(env.device);
        uint32_t previous[32] = {};
        for (uint32_t frame = 0; frame < 10; ++frame)
        {
            if (frame == 6)
                g.registry->ReleaseResources();
            if (frame == 8)
                check(g.registry->ReleasePersistent("Counter") == nvrhi::FS_OK, "ReleasePersistent");
            for (uint32_t i = 0; i < 32; ++i)
                chain.pattern[i] = 5000u + frame * 100u + i;
            g.graph->Reset();
            g.graph->PushMarker("Persistent");
            ResourceId counter = g.graph->GetPersistentBuffer("Counter", GraphBufferDesc("Counter", 128));
            ResourceId zero = g.graph->CreateBuffer(GraphBufferDesc("Zero", 128), ResourceFlags::None);
            ResourceId out = g.graph->ImportBuffer(chain.readback, ResourceStates::CopyDest, ResourceStates::CopyDest);
            const uint32_t* pattern = chain.pattern;
            bool isNew = false;
            AddPass(g.graph, "Zero", PassType::Compute, PassFlags::None,
                [&](ISetupContext* s) { zero = Write(s, zero, Access::UAVWrite, AccessFlags::Clear); },
                [](IPassContext*) { });
            AddPass(g.graph, "ReadPrevious", PassType::Copy, PassFlags::None,
                [&](ISetupContext* s) { Read(s, counter, Access::CopySource); Read(s, zero, Access::CopySource); out = Write(s, out, Access::CopyDest); },
                [counter, zero, out, &isNew](IPassContext* context) {
                    isNew = context->IsNewAllocation(counter);
                    nvrhi::ICommandList* list = context->GetCommandList();
                    list->copyBuffer(context->GetBuffer(out), 0, context->GetBuffer(counter), 0, 128);
                    list->copyBuffer(context->GetBuffer(out), 128, context->GetBuffer(zero), 0, 128);
                });
            // Odd frames add a pass that a mask switches off: a plan switch every frame.
            const PassId spare = AddPass(g.graph, "Spare", PassType::Copy, PassFlags::SideEffect,
                [&](ISetupContext* s) { Read(s, counter, Access::CopySource); },
                [](IPassContext*) { });
            g.graph->SetPassEnabled(spare, frame % 2 == 1);
            g.graph->PushMarker("Write");
            AddPass(g.graph, "WriteCurrent", PassType::Copy, PassFlags::None,
                [&](ISetupContext* s) { counter = Write(s, counter, Access::CopyDest); },
                [counter, pattern](IPassContext* context) { context->GetCommandList()->writeBuffer(context->GetBuffer(counter), pattern, 128); });
            g.graph->PopMarker();
            g.graph->PopMarker();
            g.Run();

            const bool expectNew = frame == 0 || frame == 8;
            check(isNew == expectNew, "persistent: IsNewAllocation " + std::to_string(isNew) + ", frame " + std::to_string(frame));
            if (!expectNew)
            {
                // The readback holds last frame's pattern in its first half.
                for (uint32_t i = 0; i < 32; ++i)
                    chain.pattern[i] = previous[i];
                CheckReadback(env.device, chain, "persistent: frame " + std::to_string(frame) + " reads frame " + std::to_string(frame - 1));
            }
            for (uint32_t i = 0; i < 32; ++i)
                previous[i] = 5000u + frame * 100u + i;
        }
        checkClean(env, "persistent");
    }

    // 13.1: Prepare records a static upload into the frame's first list; a Tracked body copies it into the graph.
    void testPrepare(Env& env)
    {
        GraphOnDevice g(env.device);
        BufferChain chain;
        chain.readback = CreateReadback(env.device);
        nvrhi::AutoPtr<nvrhi::IBuffer> staticData;
        env.device->createBuffer(nvrhi::BufferDesc().setByteSize(128).setInitialState(ResourceStates::CopyDest)
            .setKeepInitialState(true).setDebugName("static data"), staticData.ReleaseAndGetAddressOf());
        check(staticData != nullptr, "static data buffer");
        nvrhi::IBuffer* source = staticData;
        for (uint32_t frame = 0; frame < 3; ++frame)
        {
            for (uint32_t i = 0; i < 32; ++i)
                chain.pattern[i] = 9000u + frame * 10u + i;
            g.graph->Reset();
            ResourceId target = g.graph->CreateBuffer(GraphBufferDesc("Target", 128), ResourceFlags::None);
            ResourceId zero = g.graph->CreateBuffer(GraphBufferDesc("Zero", 128), ResourceFlags::None);
            ResourceId out = g.graph->ImportBuffer(chain.readback, ResourceStates::CopyDest, ResourceStates::CopyDest);
            const uint32_t* pattern = chain.pattern;
            AddLambdaPassWithPrepare<NoData>(g.graph, PassDesc{ "Upload", PassType::Copy, PassFlags::Tracked, 0 },
                [&](ISetupContext* s, NoData&) { target = Write(s, target, Access::CopyDest); },
                [source, pattern](IPrepareContext* context, NoData&) {
                    context->GetUploadCommandList()->writeBuffer(source, pattern, 128);
                },
                [source, target](IPassContext* context, const NoData&) {
                    context->GetCommandList()->copyBuffer(context->GetBuffer(target), 0, source, 0, 128);
                });
            AddPass(g.graph, "Zero", PassType::Compute, PassFlags::None,
                [&](ISetupContext* s) { zero = Write(s, zero, Access::UAVWrite, AccessFlags::Clear); },
                [](IPassContext*) { });
            AddPass(g.graph, "Readback", PassType::Copy, PassFlags::None,
                [&](ISetupContext* s) { Read(s, target, Access::CopySource); Read(s, zero, Access::CopySource); out = Write(s, out, Access::CopyDest); },
                [target, zero, out](IPassContext* context) {
                    nvrhi::ICommandList* list = context->GetCommandList();
                    list->copyBuffer(context->GetBuffer(out), 0, context->GetBuffer(target), 0, 128);
                    list->copyBuffer(context->GetBuffer(out), 128, context->GetBuffer(zero), 0, 128);
                });
            g.Run();
            CheckReadback(env.device, chain, "Prepare upload, frame " + std::to_string(frame));
        }
        checkClean(env, "Prepare");
    }

    // 10.3: no create* call and no heap allocation of the graph's own in steady-state frames. Measured on the device
    // without the nvrhi validation layer, against a baseline that records the same work by hand: the backend's own
    // allocations per command list and submit are in both.
    void testSteadyState(Env& env)
    {
        nvrhi::IDevice* device = env.rawDevice;
        const uint32_t warmup = 8, measured = 32;
        const bool trace = std::getenv("RDG_ALLOC_TRACE") != nullptr;
#ifdef _WIN32
        if (trace)
        {
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
            SymInitialize(GetCurrentProcess(), nullptr, TRUE);
        }
#endif

        GraphOnDevice g(device);
        BufferChain chain;
        chain.readback = CreateReadback(device);

        RegistryStatistics before{};
        std::string graphFrames, handFrames;
        std::vector<uint64_t> graphOwn, graphExecute, hand;
        graphOwn.reserve(measured);
        graphExecute.reserve(measured);
        hand.reserve(measured);
        uint64_t mark = 0;
        for (uint32_t frame = 0; frame < warmup + measured; ++frame)
        {
            if (frame == warmup)
            {
                g.registry->GetRegistryStatistics(before);
                g_Allocations = 0;
                g_CountAllocations = true;
            }
            mark = g_Allocations;
            g.graph->Reset();
            RecordBufferChain(g.graph, chain, frame, false, false, false);
            const uint64_t recorded = g_Allocations;
            g_TraceAllocations = trace && frame == warmup + 2;
            g.Run();
            g_TraceAllocations = false;
            if (frame >= warmup)
            {
                g_CountAllocations = false;
                graphFrames += " " + std::to_string(recorded - mark) + "+" + std::to_string(g.compileAllocations) + "+" +
                    std::to_string(g.executeAllocations);
                graphOwn.push_back(recorded - mark + g.compileAllocations);
                graphExecute.push_back(g.executeAllocations);
                g_CountAllocations = true;
            }
            if (frame % 4 == 3)
            {
                g_CountAllocations = false;
                device->waitForIdle();
                device->runGarbageCollection();
                g_CountAllocations = frame + 1 >= warmup;
            }
        }
        g_CountAllocations = false;
        const uint64_t graphAllocations = g_Allocations;
        CheckReadback(device, chain, "steady state: the last frame's results");
        RegistryStatistics after{};
        g.registry->GetRegistryStatistics(after);
        check(after.objectsCreated == before.objectsCreated, "steady state: no create* call in " + std::to_string(measured) +
              " frames (" + std::to_string(after.objectsCreated) + " objects in all)");

        // The baseline: the executor's nvrhi calls for this plan, made by hand. One list per frame with a marker per
        // pass, the barrier batches, an explicit-barrier scope around each body, and one submit with a fence signal.
        nvrhi::AutoPtr<nvrhi::IBuffer> a, b, c;
        device->createBuffer(GraphBufferDesc("A", 128), a.ReleaseAndGetAddressOf());
        device->createBuffer(GraphBufferDesc("B", 128), b.ReleaseAndGetAddressOf());
        device->createBuffer(GraphBufferDesc("C", 128), c.ReleaseAndGetAddressOf());
        nvrhi::AutoPtr<nvrhi::ICommandList> segment = CreateList(device);
        nvrhi::AutoPtr<nvrhi::IFence> fence;
        const bool fences = device->createFence(nvrhi::FenceDesc(), fence.ReleaseAndGetAddressOf()) == nvrhi::FS_OK;
        uint64_t fenceValue = 0;
        const bool planBarriers = env.api == nvrhi::GraphicsAPI::D3D12 || env.api == nvrhi::GraphicsAPI::VULKAN;
        const bool vulkan = env.api == nvrhi::GraphicsAPI::VULKAN;
        // Clears need CopyDest on Vulkan and a UAV state on D3D12 (6.9); Vulkan orders the readback's writes of
        // consecutive frames only with a barrier.
        const ResourceStates clearState = vulkan ? ResourceStates::CopyDest : ResourceStates::UnorderedAccess;
        const nvrhi::ResourceBarrier toWrite[] = {
            nvrhi::ResourceBarrier::Transition(a, ResourceStates::Unknown, ResourceStates::CopyDest),
            nvrhi::ResourceBarrier::Transition(b, ResourceStates::Unknown, ResourceStates::CopyDest),
            nvrhi::ResourceBarrier::Transition(c, ResourceStates::Unknown, clearState),
        };
        const nvrhi::ResourceBarrier toSource = nvrhi::ResourceBarrier::Transition(a, ResourceStates::CopyDest, ResourceStates::CopySource);
        const nvrhi::ResourceBarrier toRead[] = {
            nvrhi::ResourceBarrier::Transition(b, ResourceStates::CopyDest, ResourceStates::CopySource),
            nvrhi::ResourceBarrier::Transition(c, clearState, ResourceStates::CopySource),
            nvrhi::ResourceBarrier::Transition(chain.readback, ResourceStates::CopyDest, ResourceStates::CopyDest),
        };
        auto pass = [&](const char* name, const nvrhi::ResourceBarrier* barriers, uint32_t count, auto&& body) {
            segment->beginMarker(name);
            if (planBarriers && count)
                segment->barriers(barriers, count);
            if (planBarriers)
                segment->setExplicitBarrierScope(name);
            body();
            if (planBarriers)
                segment->setExplicitBarrierScope(nullptr);
            segment->endMarker();
        };
        for (uint32_t frame = 0; frame < warmup + measured; ++frame)
        {
            if (frame == warmup)
            {
                g_Allocations = 0;
                g_CountAllocations = true;
            }
            for (uint32_t i = 0; i < 32; ++i)
                chain.pattern[i] = frame * 1000u + i;
            mark = g_Allocations;
            g_TraceAllocations = trace && frame == warmup + 2;
            segment->open();
            if (planBarriers)
                segment->setEnableAutomaticBarriers(false);
            pass("Fill", toWrite, 2, [&] { segment->writeBuffer(a, chain.pattern, 128); });
            pass("Copy", &toSource, 1, [&] { segment->copyBuffer(b, 0, a, 0, 128); });
            pass("Clear", toWrite + 2, 1, [&] { segment->clearBufferUInt(c, 0); });
            pass("Readback", toRead, vulkan ? 3 : 2, [&] {
                segment->copyBuffer(chain.readback, 0, b, 0, 128);
                segment->copyBuffer(chain.readback, 128, c, 0, 128);
            });
            segment->close();
            nvrhi::ICommandList* lists[] = { segment };
            if (fences)
            {
                const nvrhi::FenceValue signal{ fence, ++fenceValue };
                device->submit(nvrhi::SubmitDesc().setCommandLists(lists, 1).setSignals(&signal, 1));
            }
            else
            {
                device->submit(nvrhi::SubmitDesc().setCommandLists(lists, 1));
            }
            g_TraceAllocations = false;
            if (frame >= warmup)
            {
                g_CountAllocations = false;
                handFrames += " " + std::to_string(g_Allocations - mark);
                hand.push_back(g_Allocations - mark);
                g_CountAllocations = true;
            }
            if (frame % 4 == 3)
            {
                g_CountAllocations = false;
                device->waitForIdle();
                device->runGarbageCollection();
                g_CountAllocations = frame + 1 >= warmup;
            }
        }
        g_CountAllocations = false;
        const uint64_t baselineAllocations = g_Allocations;
        device->waitForIdle();
        std::printf("  heap allocations in %u frames: through the graph %llu, recorded by hand %llu\n", measured,
                    (unsigned long long)graphAllocations, (unsigned long long)baselineAllocations);
        std::printf("  per frame, graph (record+compile+execute):%s\n  per frame, by hand:%s\n", graphFrames.c_str(), handFrames.c_str());
        // Recording and the cached compile allocate nothing; the executor's frames allocate what nvrhi allocates for the
        // same calls. nvrhi itself allocates more in an occasional frame (upload chunks), in either loop: medians.
        auto median = [](std::vector<uint64_t> values) {
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        };
        bool ownFree = true;
        for (uint64_t own : graphOwn)
            ownFree = ownFree && own == 0;
        check(ownFree, "steady state: recording and the cached compile allocate nothing");
        check(median(graphExecute) <= median(hand), "steady state: execution allocates no more than the same nvrhi calls by hand (" +
              std::to_string(median(graphExecute)) + " and " + std::to_string(median(hand)) + " per frame)");
        checkClean(env, "steady state");
    }

    // ------------------------------------------------------------------------------------------ textures

    // Texel (0, 0) of an RGBA8 texture in the CopySource state.
    uint32_t ReadTexel(Env& env, nvrhi::ITexture* texture)
    {
        nvrhi::TextureDesc stagingDesc = texture->getDesc();
        stagingDesc.isRenderTarget = false;
        stagingDesc.isUAV = false;
        stagingDesc.debugName = "staging";
        nvrhi::AutoPtr<nvrhi::IStagingTexture> staging;
        env.device->createStagingTexture(stagingDesc, nvrhi::CpuAccessMode::Read, staging.ReleaseAndGetAddressOf());
        check(staging != nullptr, "staging texture");
        nvrhi::AutoPtr<nvrhi::ICommandList> list = CreateList(env.device);
        list->open();
        list->beginTrackingTextureState(texture, nvrhi::AllSubresources, ResourceStates::CopySource);
        list->copyTexture2(staging, nvrhi::TextureSlice(), texture, nvrhi::TextureSlice());
        list->close();
        Submit(env.device, list);
        size_t pitch = 0;
        const void* data = env.device->mapStagingTexture(staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, pitch);
        check(data != nullptr, "map staging texture");
        uint32_t texel = 0;
        std::memcpy(&texel, data, sizeof(texel));
        env.device->unmapStagingTexture(staging);
        return texel;
    }

    nvrhi::TextureDesc ColorDesc(const char* name)
    {
        return nvrhi::TextureDesc().setWidth(16).setHeight(16).setFormat(Format::RGBA8_UNORM).setIsRenderTarget(true)
            .setIsUAV(true).setDebugName(name);
    }

    nvrhi::AutoPtr<nvrhi::ITexture> CreateImported(Env& env, const char* name)
    {
        nvrhi::AutoPtr<nvrhi::ITexture> texture;
        env.device->createTexture(ColorDesc(name), texture.ReleaseAndGetAddressOf());
        check(texture != nullptr, name);
        InitialState(env, texture, ResourceStates::CopySource);
        return texture;
    }

    std::string Hex(uint32_t value)
    {
        char text[16];
        std::snprintf(text, sizeof(text), "0x%08x", value);
        return text;
    }

    // A colour and a depth target cleared through the framebuffer's attachments, then copied to an import.
    void testTargets(Env& env, bool trackerFallback)
    {
        GraphOnDevice g(env.device);
        g.profile.explicitBarriers = !trackerFallback;
        nvrhi::AutoPtr<nvrhi::ITexture> result = CreateImported(env, "result");
        for (uint32_t frame = 0; frame < 4; ++frame)
        {
            g.graph->Reset();
            ResourceId color = g.graph->CreateTexture(ColorDesc("Color"), ResourceFlags::None);
            // Typeless, as D3D11 needs for a depth format with a shader-resource binding.
            ResourceId depth = g.graph->CreateTexture(nvrhi::TextureDesc().setWidth(16).setHeight(16).setFormat(Format::D32)
                .setIsRenderTarget(true).setIsTypeless(true).setDebugName("Depth"), ResourceFlags::None);
            ResourceId out = g.graph->ImportTexture(result, ResourceStates::CopySource, ResourceStates::CopySource);
            const uint32_t expected = (frame + 1) * 40u;
            const ClearValue clear = { { float(expected) / 255.0f, 0.5f, 0.25f, 1.0f }, 0.5f, 0 };
            bool framebuffer = false;
            AddPass(g.graph, "ClearTargets", PassType::Raster, PassFlags::None,
                [&](ISetupContext* s) {
                    color = s->SetColorTarget(0, color, nvrhi::SubresourceRegion(), AccessFlags::Clear, &clear);
                    depth = s->SetDepthTarget(depth, false, AccessFlags::Clear, &clear);
                },
                [&framebuffer](IPassContext* context) { framebuffer = context->GetFramebuffer() != nullptr; });
            AddPass(g.graph, "Resolve", PassType::Copy, PassFlags::None,
                [&](ISetupContext* s) { Read(s, color, Access::CopySource); out = Write(s, out, Access::CopyDest); },
                [color, out](IPassContext* context) {
                    context->GetCommandList()->copyTexture1(context->GetTexture(out), nvrhi::TextureSlice(), context->GetTexture(color), nvrhi::TextureSlice());
                });
            g.Run();
            check(framebuffer, "the raster pass has a framebuffer, frame " + std::to_string(frame));
            const uint32_t texel = ReadTexel(env, result);
            check((texel & 0xff) == expected, std::string(trackerFallback ? "tracker fallback: " : "") + "colour clear and copy, frame " +
                  std::to_string(frame) + " (texel " + Hex(texel) + ")");
        }
        checkClean(env, trackerFallback ? "tracker fallback: targets" : "targets");
    }

    // 9.1: a profile without ICommandList::barriers lowers every barrier through nvrhi's tracker, patches included.
    void testTrackerFallback(Env& env)
    {
        GraphOnDevice g(env.device);
        g.profile.explicitBarriers = false;
        BufferChain chain;
        chain.readback = CreateReadback(env.device);
        for (uint32_t frame = 0; frame < 8; ++frame)
        {
            g.graph->Reset();
            RecordBufferChain(g.graph, chain, frame, frame >= 4, true, frame % 2 == 0);
            g.Run();
            CheckReadback(env.device, chain, "tracker fallback: buffer chain, frame " + std::to_string(frame));
        }
        checkClean(env, "tracker fallback: buffer chain");
        if (env.raster)
            testTargets(env, true);
    }

    // 5.2: a three-layer history rotates, and IsNewAllocation reports each layer's first frames.
    void testHistory(Env& env)
    {
        GraphOnDevice g(env.device);
        nvrhi::AutoPtr<nvrhi::ITexture> layer1 = CreateImported(env, "layer 1");
        nvrhi::AutoPtr<nvrhi::ITexture> layer2 = CreateImported(env, "layer 2");
        for (uint32_t frame = 0; frame < 9; ++frame)
        {
            // The history survives ReleaseResources, with its rotated states exact (6.12).
            if (frame == 5)
                g.registry->ReleaseResources();
            g.graph->Reset();
            ResourceId history[3];
            check(g.graph->GetHistory("Accumulation", ColorDesc("History"), 3, history) == nvrhi::FS_OK, "GetHistory");
            ResourceId out1 = g.graph->ImportTexture(layer1, ResourceStates::CopySource, ResourceStates::CopySource);
            ResourceId out2 = g.graph->ImportTexture(layer2, ResourceStates::CopySource, ResourceStates::CopySource);
            const ClearValue clear = { { float((frame + 1) * 20u) / 255.0f, 0, 0, 1 }, 1.0f, 0 };
            bool newAllocation[3] = {};
            AddPass(g.graph, "Write", PassType::Raster, PassFlags::None,
                [&](ISetupContext* s) { history[0] = s->SetColorTarget(0, history[0], nvrhi::SubresourceRegion(), AccessFlags::Clear, &clear); },
                [](IPassContext*) { });
            AddPass(g.graph, "ReadOlder", PassType::Copy, PassFlags::None,
                [&](ISetupContext* s) {
                    Read(s, history[1], Access::CopySource);
                    Read(s, history[2], Access::CopySource);
                    out1 = Write(s, out1, Access::CopyDest);
                    out2 = Write(s, out2, Access::CopyDest);
                },
                [h0 = history[0], h1 = history[1], h2 = history[2], out1, out2, &newAllocation](IPassContext* context) {
                    newAllocation[0] = context->IsNewAllocation(h0);
                    newAllocation[1] = context->IsNewAllocation(h1);
                    newAllocation[2] = context->IsNewAllocation(h2);
                    nvrhi::ICommandList* list = context->GetCommandList();
                    list->copyTexture1(context->GetTexture(out1), nvrhi::TextureSlice(), context->GetTexture(h1), nvrhi::TextureSlice());
                    list->copyTexture1(context->GetTexture(out2), nvrhi::TextureSlice(), context->GetTexture(h2), nvrhi::TextureSlice());
                });
            g.Run();
            check(!newAllocation[0] && newAllocation[1] == (frame < 1) && newAllocation[2] == (frame < 2),
                  "IsNewAllocation per layer, frame " + std::to_string(frame) + " (" + std::to_string(newAllocation[0]) +
                  std::to_string(newAllocation[1]) + std::to_string(newAllocation[2]) + ")");
            if (frame >= 2)
            {
                const uint32_t one = ReadTexel(env, layer1) & 0xff;
                const uint32_t two = ReadTexel(env, layer2) & 0xff;
                check(one == frame * 20u && two == (frame - 1) * 20u, "history layers 1 and 2 hold frames " + std::to_string(frame - 1) +
                      " and " + std::to_string(frame - 2) + " (read " + std::to_string(one) + ", " + std::to_string(two) + ")");
            }
        }
        checkClean(env, "history");
    }

    // 5.2: an extracted transient outlives the graph; the pool reuses it once the caller has released it.
    void testExtract(Env& env)
    {
        GraphOnDevice g(env.device);
        RegistryStatistics afterFirst{};
        for (uint32_t frame = 0; frame < 5; ++frame)
        {
            g.graph->Reset();
            ResourceId color = g.graph->CreateTexture(ColorDesc("Extracted"), ResourceFlags::None);
            const uint32_t expected = (frame + 2) * 30u;
            const ClearValue clear = { { float(expected) / 255.0f, 0, 0, 1 }, 1.0f, 0 };
            AddPass(g.graph, "Produce", PassType::Raster, PassFlags::None,
                [&](ISetupContext* s) { color = s->SetColorTarget(0, color, nvrhi::SubresourceRegion(), AccessFlags::Clear, &clear); },
                [](IPassContext*) { });
            nvrhi::ITexture* extracted = nullptr;
            g.graph->Extract(color, ResourceStates::CopySource, &extracted);
            g.Run();
            check(extracted != nullptr, "Extract returns the texture, frame " + std::to_string(frame));
            const uint32_t texel = ReadTexel(env, extracted) & 0xff;
            check(texel == expected, "the extracted texture holds the frame's clear (" + std::to_string(texel) + ")");
            extracted->Release();
            if (frame == 0)
                g.registry->GetRegistryStatistics(afterFirst);
        }
        RegistryStatistics last{};
        g.registry->GetRegistryStatistics(last);
        check(last.objectsCreated == afterFirst.objectsCreated, "released extracted textures are reused (" +
              std::to_string(last.objectsCreated - afterFirst.objectsCreated) + " objects created after frame 0)");
        checkClean(env, "extract");
    }

    // ------------------------------------------------------------------------------------------ pass modes

    // 9.1: a Tracked pass relies on automatic barriers; a Restore pass leaves its ranges in their declared states.
    void testTrackedAndRestore(Env& env)
    {
        GraphOnDevice g(env.device);
        BufferChain chain;
        chain.readback = CreateReadback(env.device);
        for (uint32_t frame = 0; frame < 4; ++frame)
        {
            for (uint32_t i = 0; i < 32; ++i)
                chain.pattern[i] = 7u + frame * 100u + i;
            g.graph->Reset();
            ResourceId source = g.graph->CreateBuffer(GraphBufferDesc("Source", 128), ResourceFlags::None);
            ResourceId x = g.graph->CreateBuffer(GraphBufferDesc("X", 128), ResourceFlags::None);
            ResourceId y = g.graph->CreateBuffer(GraphBufferDesc("Y", 128), ResourceFlags::None);
            ResourceId zero = g.graph->CreateBuffer(GraphBufferDesc("Zero", 128), ResourceFlags::None);
            ResourceId out = g.graph->ImportBuffer(chain.readback, ResourceStates::CopyDest, ResourceStates::CopyDest);
            const uint32_t* pattern = chain.pattern;
            AddPass(g.graph, "TrackedFill", PassType::Copy, PassFlags::Tracked,
                [&](ISetupContext* s) { source = Write(s, source, Access::CopyDest); },
                [source, pattern](IPassContext* context) { context->GetCommandList()->writeBuffer(context->GetBuffer(source), pattern, 128); });
            // The body reads X after writing it: the automatic barriers move X to CopySource, and Restore returns it
            // to the declared CopyDest before the next pass.
            AddPass(g.graph, "RestoreCopies", PassType::Copy, PassFlags::Restore,
                [&](ISetupContext* s) {
                    Read(s, source, Access::CopySource);
                    x = Write(s, x, Access::CopyDest);
                    y = Write(s, y, Access::CopyDest);
                },
                [source, x, y](IPassContext* context) {
                    nvrhi::ICommandList* list = context->GetCommandList();
                    list->copyBuffer(context->GetBuffer(x), 0, context->GetBuffer(source), 0, 128);
                    list->copyBuffer(context->GetBuffer(y), 0, context->GetBuffer(x), 0, 128);
                });
            AddPass(g.graph, "Zero", PassType::Compute, PassFlags::None,
                [&](ISetupContext* s) { zero = Write(s, zero, Access::UAVWrite, AccessFlags::Clear); },
                [](IPassContext*) { });
            AddPass(g.graph, "Readback", PassType::Copy, PassFlags::None,
                [&](ISetupContext* s) {
                    Read(s, y, Access::CopySource);
                    Read(s, x, Access::CopySource);
                    Read(s, zero, Access::CopySource);
                    out = Write(s, out, Access::CopyDest);
                },
                [y, zero, out](IPassContext* context) {
                    nvrhi::ICommandList* list = context->GetCommandList();
                    list->copyBuffer(context->GetBuffer(out), 0, context->GetBuffer(y), 0, 128);
                    list->copyBuffer(context->GetBuffer(out), 128, context->GetBuffer(zero), 0, 128);
                });
            g.Run();
            CheckReadback(env.device, chain, "Tracked and Restore passes, frame " + std::to_string(frame));
        }
        checkClean(env, "Tracked and Restore");
    }

    // 8.2, 9.1: a state call by a Native body is reported by the nvrhi validation layer, naming the pass.
    void testNativeGuard(Env& env)
    {
        GraphOnDevice g(env.device);
        g.graph->Reset();
        ResourceId a = g.graph->CreateBuffer(GraphBufferDesc("Guarded", 128), ResourceFlags::None);
        AddPass(g.graph, "SneakyBody", PassType::Copy, PassFlags::SideEffect,
            [&](ISetupContext* s) { a = Write(s, a, Access::CopyDest); },
            [a](IPassContext* context) { context->GetCommandList()->setBufferState(context->GetBuffer(a), ResourceStates::CopySource); });
        env.messages->expecting = true;
        env.messages->expectedErrorsSeen = 0;
        g.Run();
        env.messages->expecting = false;
        check(env.messages->expectedErrorsSeen > 0, "the validation layer reports a state call inside a Native body");
        env.device->waitForIdle();
        env.nativeErrors();   // whatever the native layer made of the sneaked state call
    }

    // 11.4: the debug switches that change the plan run clean.
    void testDebugSwitches(Env& env)
    {
        GraphOnDevice g(env.device);
        BufferChain chain;
        chain.readback = CreateReadback(env.device);
        const DebugFlags switches[] = { DebugFlags::BarrierAfterEveryPass, DebugFlags::SerializeQueues, DebugFlags::FillTransients,
                                        DebugFlags::ForceDedicated, DebugFlags::NoAliasing, DebugFlags::StrictValidation };
        uint32_t frame = 0;
        for (DebugFlags flags : switches)
        {
            CompileOptions options;
            options.debugFlags = flags;
            for (uint32_t i = 0; i < 2; ++i, ++frame)
            {
                g.graph->Reset();
                RecordBufferChain(g.graph, chain, frame, true, false, false);
                g.Run(options);
                CheckReadback(env.device, chain, "debug switch " + std::to_string(uint32_t(flags)) + ", frame " + std::to_string(frame));
            }
        }
        checkClean(env, "debug switches");
    }

    // A deliberate write-after-write hazard: the clean runs then mean that synchronization validation was watching.
    void testSyncValidationIsLoaded(Env& env)
    {
        nvrhi::AutoPtr<nvrhi::IBuffer> buffer;
        env.device->createBuffer(nvrhi::BufferDesc().setByteSize(256).setDebugName("hazard"), buffer.ReleaseAndGetAddressOf());
        nvrhi::AutoPtr<nvrhi::ICommandList> list = CreateList(env.device);
        list->open();
        list->setEnableAutomaticBarriers(false);
        const nvrhi::ResourceBarrier state = nvrhi::ResourceBarrier::Transition(buffer, ResourceStates::Unknown, ResourceStates::CopyDest);
        list->barriers(&state, 1);
        const uint32_t pattern[4] = { 1, 2, 3, 4 };
        list->writeBuffer(buffer, pattern, sizeof(pattern));
        env.messages->expecting = true;
        env.messages->expectedErrorsSeen = 0;
        list->writeBuffer(buffer, pattern, sizeof(pattern));
        env.messages->expecting = false;
        list->close();
        Submit(env.device, list);
        check(env.messages->expectedErrorsSeen > 0, "synchronization validation reports a hazard (it is loaded)");
    }

    void runAll(Env& env)
    {
        if (env.api == nvrhi::GraphicsAPI::VULKAN)
            testSyncValidationIsLoaded(env);
        testBufferChains(env);
        testPlanSwitches(env);
        testTrackedAndRestore(env);
        testDebugSwitches(env);
        testPersistent(env);
        testPrepare(env);
        if (env.raster)
        {
            testTargets(env, false);
            testHistory(env);
            testExtract(env);
        }
        if (env.explicitScopes)
            testTrackerFallback(env);
        if (env.explicitScopes)
            testNativeGuard(env);
        checkClean(env, "all graphs");
        testSteadyState(env);
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
}

int main(int argc, char** argv)
{
    // Unbuffered, so that the output shows how far a run that dies got.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try
    {
        check(argc == 2, "specify d3d11, d3d12, vulkan or cuda");
        const std::string backend = argv[1];
        nvrhi::AutoPtr<Messages> messages = MAKE_RC_OBJ_PTR(Messages);
        Env env;
        env.messages = messages;

#if TEST_D3D11
        if (backend == "d3d11")
        {
            nvrhi::d3d11::GraphicsFactoryDesc factoryDesc;
            factoryDesc.messageCallback = messages;
            nvrhi::d3d11::GraphicsFactoryHandle factory = nvrhi::d3d11::createGraphicsFactory(factoryDesc);
            check(factory != nullptr, "d3d11 graphics factory");
            nvrhi::d3d11::DeviceCreationDesc deviceDesc;
            deviceDesc.messageCallback = messages;
            deviceDesc.adapterIndex = FindNvidiaAdapter(factory.Get());
            nvrhi::DeviceHandle device, validation;
            check(factory->createDeviceOnAdapter(deviceDesc, &device) == nvrhi::FS_OK, "createDeviceOnAdapter");
            check(factory->createValidationLayer(device, &validation) == nvrhi::FS_OK, "validation layer");
            env.api = nvrhi::GraphicsAPI::D3D11;
            env.device = validation;
            env.rawDevice = device;
            runAll(env);
        }
        else
#endif
#if TEST_D3D12
        if (backend == "d3d12")
        {
            nvrhi::d3d12::GraphicsFactoryDesc factoryDesc;
            factoryDesc.messageCallback = messages;
            factoryDesc.enableDebugRuntime = true;
            factoryDesc.enableGPUValidation = true;
            // The Agility SDK in .\D3D12\, when its exact version is given: it brings enhanced barriers to Windows 10.
            if (const char* sdkVersion = std::getenv("NVRHI_TEST_AGILITY_SDK_VERSION"))
            {
                factoryDesc.sdkVersion = uint32_t(std::strtoul(sdkVersion, nullptr, 10));
                factoryDesc.sdkPath = ".\\D3D12\\";
                std::printf("  Agility SDK %u\n", factoryDesc.sdkVersion);
            }
            nvrhi::d3d12::GraphicsFactoryHandle factory = nvrhi::d3d12::createGraphicsFactory(factoryDesc);
            check(factory != nullptr, "d3d12 graphics factory");
            nvrhi::d3d12::DeviceCreationDesc deviceDesc;
            deviceDesc.errorCB = messages;
            deviceDesc.enableComputeQueue = true;
            deviceDesc.enableCopyQueue = true;
            deviceDesc.adapterIndex = FindNvidiaAdapter(factory.Get());
            nvrhi::d3d12::DeviceHandle device;
            check(factory->createDeviceOnAdapter(deviceDesc, &device) == nvrhi::FS_OK, "createDeviceOnAdapter");
            nvrhi::DeviceHandle validation;
            check(factory->createValidationLayer(device, &validation) == nvrhi::FS_OK, "validation layer");

            // The debug layer is loaded when the device answers ID3D12InfoQueue.
            ID3D12Device* native = static_cast<ID3D12Device*>(device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device));
            nvrhi::AutoPtr<ID3D12InfoQueue> infoQueue;
            check(native && SUCCEEDED(native->QueryInterface(IID_PPV_ARGS(&infoQueue))), "the D3D12 debug layer is loaded (ID3D12InfoQueue)");
            infoQueue->SetMuteDebugOutput(FALSE);
            infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
            infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
            // Messages as they happen, where the runtime has ID3D12InfoQueue1: a run that dies still shows its last ones.
            nvrhi::AutoPtr<ID3D12InfoQueue1> infoQueue1;
            DWORD callbackCookie = 0;
            if (SUCCEEDED(infoQueue->QueryInterface(IID_PPV_ARGS(&infoQueue1))))
            {
                infoQueue1->RegisterMessageCallback([](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id,
                    LPCSTR description, void*)
                {
                    if (severity <= D3D12_MESSAGE_SEVERITY_WARNING)
                        std::printf("  D3D12 (live, severity %d, id %d): %s\n", int(severity), int(id), description);
                }, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &callbackCookie);
            }
            std::printf("  enhanced barriers: %d\n", int(device->queryFeatureSupport(nvrhi::Feature::EnhancedBarriers)));

            env.api = nvrhi::GraphicsAPI::D3D12;
            env.device = validation;
            env.rawDevice = device;
            env.explicitScopes = true;
            env.nativeErrors = [&]
            {
                unsigned errors = 0;
                const UINT64 count = infoQueue->GetNumStoredMessages();
                for (UINT64 index = 0; index < count; ++index)
                {
                    SIZE_T size = 0;
                    infoQueue->GetMessage(index, nullptr, &size);
                    std::vector<uint8_t> storage(size);
                    D3D12_MESSAGE* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                    if (FAILED(infoQueue->GetMessage(index, message, &size)))
                        continue;
                    if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
                    {
                        ++errors;
                        std::printf("D3D12: %s\n", message->pDescription);
                    }
                }
                infoQueue->ClearStoredMessages();
                return errors;
            };
            runAll(env);
        }
        else
#endif
#if TEST_VULKAN
        if (backend == "vulkan")
        {
            // Synchronization validation.
#ifdef _WIN32
            _putenv_s("VK_LAYER_VALIDATE_SYNC", "1");
#endif
            nvrhi::vulkan::GraphicsFactoryDesc factoryDesc;
            factoryDesc.messageCallback = messages;
            factoryDesc.enableDebugRuntime = true;
            factoryDesc.enableRayTracingValidation = true;   // creates the debug-utils messenger that reports errors as errors
            factoryDesc.headless = true;
            nvrhi::vulkan::GraphicsFactoryHandle factory = nvrhi::vulkan::createGraphicsFactory(factoryDesc);
            check(factory != nullptr, "vulkan graphics factory");
            bool layerEnabled = false;
            const uint32_t layerCount = factory->getEnabledLayers(nullptr, 0);
            std::vector<const char*> layers(layerCount);
            factory->getEnabledLayers(layers.data(), layerCount);
            for (const char* layer : layers)
                layerEnabled |= std::strcmp(layer, "VK_LAYER_KHRONOS_validation") == 0;
            check(layerEnabled, "the Vulkan validation layer is loaded");

            nvrhi::vulkan::DeviceCreationDesc deviceDesc;
            deviceDesc.errorCB = messages;
            deviceDesc.enableComputeQueue = true;
            deviceDesc.enableCopyQueue = true;
            deviceDesc.adapterIndex = FindNvidiaAdapter(factory.Get());
            nvrhi::vulkan::DeviceHandle device;
            check(factory->createDeviceOnAdapter(deviceDesc, &device) == nvrhi::FS_OK, "createDeviceOnAdapter");
            nvrhi::DeviceHandle validation;
            check(factory->createValidationLayer(device, &validation) == nvrhi::FS_OK, "validation layer");
            std::printf("  queue-ownership transfers: %d\n", int(device->queryFeatureSupport(nvrhi::Feature::QueueOwnershipTransfer)));

            env.api = nvrhi::GraphicsAPI::VULKAN;
            env.device = validation;
            env.rawDevice = device;
            env.explicitScopes = true;
            runAll(env);
            validation = nullptr;
            device->waitForIdle();
        }
        else
#endif
#if TEST_CUDA
        if (backend == "cuda")
        {
            nvrhi::cuda::GraphicsFactoryDesc factoryDesc;
            factoryDesc.messageCallback = messages;
            nvrhi::cuda::GraphicsFactoryHandle factory = nvrhi::cuda::createGraphicsFactory(factoryDesc);
            if (!factory)
            {
                std::printf("SKIP: no CUDA driver\n");
                return 0;
            }
            nvrhi::cuda::DeviceDesc deviceDesc;
            deviceDesc.messageCallback = messages;
            nvrhi::cuda::DeviceHandle device;
            check(factory->createDevice(deviceDesc, &device) == nvrhi::FS_OK, "createDevice");
            nvrhi::DeviceHandle validation;
            check(factory->createValidationLayer(device, &validation) == nvrhi::FS_OK, "validation layer");
            env.api = nvrhi::GraphicsAPI::CUDA;
            env.device = validation;
            env.rawDevice = device;
            env.raster = false;
            runAll(env);
        }
        else
#endif
        {
            check(false, "unknown or disabled backend");
        }

        check(messages->errors == 0, "no unexpected NVRHI errors");
        std::printf("PASS\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
