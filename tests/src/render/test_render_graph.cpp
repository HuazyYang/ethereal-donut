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

// The render graph's device-free tests (aggregate plan, step P1.3): recording rules, validation, culling, the plan
// cache, golden barrier lists for the frame of design 6.11, and the verifier corpus (11.1) over the reference
// profiles, with two-frame and plan-switch replays. Public interfaces only.
//
//     test_render_graph [filter]          runs the tests whose name contains `filter`
//     RDG_UPDATE_GOLDEN=1                 rewrites the golden files instead of comparing

#include "rdg_corpus.h"

#include <donut/render/RenderGraphFrameData.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace donut::render;
using namespace rdgtest;
using nvrhi::Format;
using nvrhi::ResourceStates;
using nvrhi::ShaderType;

// ---------------------------------------------------------------------------------------------- allocation hook

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

// ---------------------------------------------------------------------------------------------- harness

namespace
{
    struct Failure : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

#define EXPECT(condition, message)                                                                         \
    do {                                                                                                   \
        if (!(condition))                                                                                  \
            throw Failure(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + (message));     \
    } while (0)

    struct Fixture
    {
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;

        Fixture()
        {
            IResourceRegistry* r = nullptr;
            EXPECT(CreateResourceRegistry(nullptr, &r) == nvrhi::FS_OK, "CreateResourceRegistry");
            registry = nvrhi::TakeOver(r);
            IRenderGraph* g = nullptr;
            EXPECT(CreateRenderGraph(registry.Get(), &g) == nvrhi::FS_OK, "CreateRenderGraph");
            graph = nvrhi::TakeOver(g);
            graph->Reset();
        }
    };

    BackendProfile Profile(ReferenceProfile kind)
    {
        BackendProfile profile;
        GetReferenceProfile(kind, &profile);
        return profile;
    }

    const char* ProfileName(ReferenceProfile kind)
    {
        switch (kind)
        {
        case ReferenceProfile::D3D12Legacy: return "D3D12Legacy";
        case ReferenceProfile::D3D12Enhanced: return "D3D12Enhanced";
        case ReferenceProfile::Vulkan: return "Vulkan";
        case ReferenceProfile::VulkanSeparateComputeFamily: return "VulkanSeparateComputeFamily";
        case ReferenceProfile::D3D11: return "D3D11";
        case ReferenceProfile::CUDA: return "CUDA";
        default: return "?";
        }
    }

    const ReferenceProfile c_AllProfiles[] = {
        ReferenceProfile::D3D12Legacy, ReferenceProfile::D3D12Enhanced, ReferenceProfile::Vulkan,
        ReferenceProfile::VulkanSeparateComputeFamily, ReferenceProfile::D3D11, ReferenceProfile::CUDA,
    };

    std::string Diagnostics(IRenderGraph* graph)
    {
        nvrhi::string text;
        graph->GetDiagnostics(text);
        return text.c_str();
    }

    nvrhi::AutoPtr<ICompiledRenderGraph> Compile(IRenderGraph* graph, const BackendProfile& profile,
                                                 const CompileOptions& options = CompileOptions())
    {
        ICompiledRenderGraph* plan = nullptr;
        const nvrhi::FRESULT result = graph->Compile(profile, options, &plan);
        if (result != nvrhi::FS_OK)
            return nullptr;
        return nvrhi::TakeOver(plan);
    }

    std::string Dump(ICompiledRenderGraph* plan)
    {
        nvrhi::string text;
        plan->WriteDump(DumpFormat::Json, text);
        return text.c_str();
    }

    bool Contains(const std::string& text, const char* what) { return text.find(what) != std::string::npos; }

    // Compiles, expects failure, and returns the diagnostics.
    std::string ExpectCompileError(IRenderGraph* graph, ReferenceProfile kind = ReferenceProfile::D3D12Legacy)
    {
        ICompiledRenderGraph* plan = nullptr;
        const nvrhi::FRESULT result = graph->Compile(Profile(kind), CompileOptions(), &plan);
        if (plan)
            plan->Release();
        EXPECT(result != nvrhi::FS_OK, "Compile succeeded, but it should have failed");
        return Diagnostics(graph);
    }

    std::string StateName(ResourceStates states)
    {
        static const struct { ResourceStates bit; const char* name; } c_Names[] = {
            { ResourceStates::Common, "Common" }, { ResourceStates::ConstantBuffer, "CB" },
            { ResourceStates::VertexBuffer, "VB" }, { ResourceStates::IndexBuffer, "IB" },
            { ResourceStates::IndirectArgument, "Indirect" }, { ResourceStates::PixelShaderResource, "PixelSRV" },
            { ResourceStates::NonPixelShaderResource, "NonPixelSRV" }, { ResourceStates::UnorderedAccess, "UAV" },
            { ResourceStates::RenderTarget, "RT" }, { ResourceStates::DepthWrite, "DepthWrite" },
            { ResourceStates::DepthRead, "DepthRead" }, { ResourceStates::CopyDest, "CopyDest" },
            { ResourceStates::CopySource, "CopySource" }, { ResourceStates::Present, "Present" },
        };
        if (states == ResourceStates::Unknown)
            return "Unknown";
        std::string text;
        for (const auto& entry : c_Names)
        {
            if ((states & entry.bit) != 0)
            {
                if (!text.empty())
                    text += "|";
                text += entry.name;
            }
        }
        return text;
    }

    const char* ReasonText(BarrierReason reason)
    {
        static const char* const c_Names[] = { "RAW", "WAR", "WAW", "UavOrder", "ReadGroupChange", "Alias",
            "QueueTransfer", "Discard", "ExitState", "Persistent", "Debug" };
        return uint32_t(reason) < 11 ? c_Names[uint32_t(reason)] : "?";
    }

    const char* QueueText(nvrhi::CommandQueue queue)
    {
        return queue == nvrhi::CommandQueue::Graphics ? "graphics" : queue == nvrhi::CommandQueue::Compute ? "compute" : "copy";
    }

    std::string BarrierText(const Builder& builder, const PlanBarrier& barrier)
    {
        std::string text;
        const std::string name = barrier.resource == ~0u ? std::string("*") : builder.NameOf(barrier.resource);
        if (barrier.type == nvrhi::BarrierType::Uav)
            text = "UAV " + name;
        else if (barrier.type == nvrhi::BarrierType::Aliasing)
            text = "Alias -> " + name;
        else
        {
            text = name;
            const nvrhi::SubresourceRegion& region = barrier.region;
            if (region.mipCount != nvrhi::SubresourceRegion::AllMipLevels || region.sliceCount != nvrhi::SubresourceRegion::AllArraySlices ||
                region.planes != nvrhi::PlaneMask::All)
            {
                text += " [mips " + std::to_string(region.baseMip) + "+" + std::to_string(region.mipCount) + " slices " +
                    std::to_string(region.baseSlice) + "+" + std::to_string(region.sliceCount);
                if (region.planes == nvrhi::PlaneMask::Depth) text += " depth";
                if (region.planes == nvrhi::PlaneMask::Stencil) text += " stencil";
                text += "]";
            }
            text += " " + StateName(barrier.before) + " -> " + StateName(barrier.after);
            if ((barrier.flags & nvrhi::BarrierFlags::SplitBegin) != 0) text += " begin";
            if ((barrier.flags & nvrhi::BarrierFlags::SplitEnd) != 0) text += " end";
            if ((barrier.flags & nvrhi::BarrierFlags::Discard) != 0) text += " discard";
            if ((barrier.flags & nvrhi::BarrierFlags::QueueRelease) != 0) text += std::string(" release-to-") + QueueText(barrier.otherQueue);
            if ((barrier.flags & nvrhi::BarrierFlags::QueueAcquire) != 0) text += std::string(" acquire-from-") + QueueText(barrier.otherQueue);
        }
        return text + " (" + ReasonText(barrier.reason) + ")";
    }

    // The plan as text, through the public queries only.
    std::string Listing(const Builder& builder, ICompiledRenderGraph* plan)
    {
        std::string text;
        static const char* const c_Slots[] = { "prologue", "post-clear", "epilogue" };
        for (uint32_t i = 0; i < plan->GetScheduledPassCount(); ++i)
        {
            ScheduledPass pass;
            plan->GetScheduledPass(i, pass);
            text += std::to_string(i) + " " + pass.name + " (" + QueueText(pass.queue) + ", segment " +
                (pass.enabled ? std::to_string(pass.segment) : std::string("-")) + (pass.enabled ? "" : ", disabled") + ")\n";
            for (uint32_t slot = 0; slot < 3; ++slot)
            {
                BarrierBatch batch;
                plan->GetBatch(i, BatchSlot(slot), batch);
                for (uint32_t b = 0; b < batch.count; ++b)
                    text += std::string("  ") + c_Slots[slot] + ": " + BarrierText(builder, batch.barriers[b]) + "\n";
            }
        }
        PlanStatistics stats;
        plan->GetStatistics(stats);
        text += "barriers " + std::to_string(stats.barriers) + ", transitions " + std::to_string(stats.transitions) +
            ", split " + std::to_string(stats.splitBarriers) + ", uav " + std::to_string(stats.uavBarriers) +
            ", aliasing " + std::to_string(stats.aliasingBarriers) + ", transfers " + std::to_string(stats.queueTransfers) +
            ", segments " + std::to_string(stats.segments) + ", fence waits " + std::to_string(stats.fenceWaits) + "\n";
        return text;
    }

    // Transitions of one resource, in schedule order.
    std::vector<PlanBarrier> TransitionsOf(ICompiledRenderGraph* plan, const Builder& builder, const char* name)
    {
        std::vector<PlanBarrier> result;
        for (uint32_t i = 0; i < plan->GetScheduledPassCount(); ++i)
        {
            for (uint32_t slot = 0; slot < 3; ++slot)
            {
                BarrierBatch batch;
                plan->GetBatch(i, BatchSlot(slot), batch);
                for (uint32_t b = 0; b < batch.count; ++b)
                {
                    const PlanBarrier& barrier = batch.barriers[b];
                    if (barrier.type == nvrhi::BarrierType::Transition && barrier.resource != ~0u &&
                        builder.NameOf(barrier.resource) == name)
                        result.push_back(barrier);
                }
            }
        }
        return result;
    }

    std::string VerifyOrReport(ICompiledRenderGraph* plan, IRenderGraph* graph, ICompiledRenderGraph* previous = nullptr)
    {
        nvrhi::string report;
        const nvrhi::FRESULT result = VerifyPlan(plan, graph, previous, &report);
        if (result == nvrhi::FS_OK)
            return std::string();
        std::string text = report.c_str();
        return text.empty() ? std::string("verify failed without a report") : text;
    }

    // ------------------------------------------------------------------------------------------ recording rules

    void TestIds()
    {
        const ResourceId id = MakeResourceId(5, 7);
        EXPECT(GetResourceIndex(id) == 5 && GetResourceVersion(id) == 7, "id packing");
        EXPECT(GetResourceIndex(InvalidResourceId) == ~0u, "invalid id");

        Fixture f;
        Builder b(f.graph);
        ResourceId t = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        const ResourceId stale = t;
        ResourceId out = b.Import("Out", 64, 64, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        b.Pass("Writer", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { t = Write(c, t, Access::UAVWrite); });
        b.Pass("StaleReader", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            Read(c, stale, Access::ShaderRead);
            out = Write(c, out, Access::UAVWrite);
        });
        const std::string diagnostics = ExpectCompileError(f.graph);
        EXPECT(Contains(diagnostics, "StaleReader") && Contains(diagnostics, "Writer") &&
               Contains(diagnostics, "not its current version"), "the stale-id error names both passes: " + diagnostics);
    }

    void TestIdFromEarlierFrame()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId old = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        f.graph->Reset();
        Builder b2(f.graph);
        ResourceId t = b2.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        EXPECT(GetResourceIndex(t) == GetResourceIndex(old) && t != old, "the index is reused, the version is not");
        ResourceId out = b2.Import("Out", 64, 64, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        b2.Pass("UsesOld", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            Write(c, old, Access::UAVWrite);
            out = Write(c, out, Access::UAVWrite);
        });
        EXPECT(Contains(ExpectCompileError(f.graph), "not its current version"), "an id from an earlier frame is rejected");
    }

    void TestSetupContextClosed()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId t = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        ISetupContext* kept = nullptr;
        b.Pass("Keeps", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) {
            kept = c;
            t = Write(c, t, Access::UAVWrite);
        });
        EXPECT(kept->Write(t, Access::UAVWrite, ShaderType::None, {}, AccessFlags::FullWrite) == InvalidResourceId,
               "Write after Setup returns InvalidResourceId");
        EXPECT(Contains(ExpectCompileError(f.graph), "outside IGpuPass::Setup"), "a closed setup context is reported");
    }

    void TestNestedAddPass()
    {
        Fixture f;
        Builder b(f.graph);
        b.Pass("Outer", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext*) {
            AddLambdaPass<NoData>(f.graph, PassDesc{ "Inner", PassType::Compute, PassFlags::None, 0 },
                [](ISetupContext*, NoData&) { }, [](IPassContext*, const NoData&) { });
        });
        EXPECT(Contains(ExpectCompileError(f.graph), "from inside IGpuPass::Setup"), "AddPass inside Setup is rejected");
        EXPECT(f.graph->GetPassCount() == 1, "the nested pass was not added");
    }

    void TestFrozen()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId out = b.Import("Out", 64, 64, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        b.Pass("P", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { out = Write(c, out, Access::UAVWrite); });
        EXPECT(Compile(f.graph, Profile(ReferenceProfile::D3D12Legacy)), "the first compile succeeds");
        EXPECT(b.Texture("Late", 64, 64, Format::RGBA8_UNORM, false, true) == InvalidResourceId, "a declaration after Compile fails");
        EXPECT(Contains(ExpectCompileError(f.graph), "frozen"), "the next Compile reports it");
    }

    void TestHistoryWrite()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId history[2];
        b.History("H", 64, 64, Format::RGBA8_UNORM, 2, history);
        b.Pass("P", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) { Write(c, history[1], Access::UAVWrite); });
        EXPECT(Contains(ExpectCompileError(f.graph), "only the current layer"), "writing an older history layer is rejected");

        ResourceId ids[9];
        EXPECT(f.graph->GetHistory("Bad", nvrhi::TextureDesc(), 9, ids) == nvrhi::FE_INVALID_ARGS, "9 layers are rejected");
    }

    void TestPassLocalForeign()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId scratch = InvalidResourceId;
        nvrhi::TextureDesc desc;
        desc.width = 64;
        desc.height = 64;
        desc.format = Format::R32_FLOAT;
        desc.isUAV = true;
        b.Pass("Owner", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) {
            scratch = c->CreateTransientTexture(desc, ResourceFlags::None);
            scratch = Write(c, scratch, Access::UAVWrite);
        });
        b.Pass("Other", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) { Read(c, scratch, Access::ShaderRead); });
        EXPECT(Contains(ExpectCompileError(f.graph), "pass-local transient"), "another pass naming a pass-local transient is rejected");
    }

    void TestReadBeforeWrite()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId t = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        ResourceId out = b.Import("Out", 64, 64, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        b.Pass("Reader", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            Read(c, t, Access::ShaderRead);
            out = Write(c, out, Access::UAVWrite);
        });
        EXPECT(Contains(ExpectCompileError(f.graph), "before any pass writes it"), "a transient read before any write is rejected");
    }

    void TestPartialFirstWrite()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId t = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        ResourceId out = b.Import("Out", 64, 64, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        b.Pass("Partial", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { t = Write(c, t, Access::UAVWrite, AccessFlags::None); });
        b.Pass("Reader", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            Read(c, t, Access::ShaderRead);
            out = Write(c, out, Access::UAVWrite);
        });
        EXPECT(Contains(ExpectCompileError(f.graph), "must be a FullWrite or a Clear"), "a partial first write is rejected");
    }

    void TestOverlapConflict()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId t = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        b.Pass("Init", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { t = Write(c, t, Access::UAVWrite); });
        b.Pass("Both", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) {
            Read(c, t, Access::ShaderRead);
            t = Write(c, t, Access::UAVWrite, AccessFlags::None);
        });
        EXPECT(Contains(ExpectCompileError(f.graph), "overlap"), "a shader read overlapping a UAV write is rejected");
    }

    void TestExclusionConflict()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId out = b.Import("Out", 64, 64, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        b.Pass("A", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { out = Write(c, out, Access::UAVWrite); }, 3);
        b.Pass("B", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { out = Write(c, out, Access::UAVWrite); }, 3);
        EXPECT(Contains(ExpectCompileError(f.graph), "exclusion group"), "two enabled passes of one exclusion group are rejected");
    }

    void TestAsyncBackBuffer()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId back = b.BackBuffer("Back", 64, 64);
        b.Pass("AsyncTouch", PassType::Compute, PassFlags::AsyncCompute, [&](ISetupContext* c) { Read(c, back, Access::ShaderRead); });
        EXPECT(Contains(ExpectCompileError(f.graph), "AsyncCompute pass cannot touch it"), "the back buffer in an async pass is rejected");
    }

    void TestCudaRules()
    {
        {
            Fixture f;
            Builder b(f.graph);
            ResourceId rt = b.Texture("RT", 64, 64, Format::RGBA8_UNORM, true, false);
            b.Pass("Draw", PassType::Raster, PassFlags::SideEffect, [&](ISetupContext* c) {
                rt = c->SetColorTarget(0, rt, {}, AccessFlags::FullWrite, nullptr);
            });
            EXPECT(Contains(ExpectCompileError(f.graph, ReferenceProfile::CUDA), "no raster passes"), "raster on CUDA is rejected");
        }
        {
            Fixture f;
            Builder b(f.graph);
            ResourceId depth = b.Depth("D", 64, 64, Format::D32);
            b.Pass("ClearDepth", PassType::Copy, PassFlags::SideEffect, [&](ISetupContext* c) {
                depth = Write(c, depth, Access::DepthWrite, AccessFlags::Clear);
            });
            EXPECT(Contains(ExpectCompileError(f.graph, ReferenceProfile::CUDA), "no depth clears"), "depth clears on CUDA are rejected");
        }
    }

    void TestKeepInitialStateTransient()
    {
        Fixture f;
        nvrhi::TextureDesc desc;
        desc.width = 64;
        desc.height = 64;
        desc.format = Format::RGBA8_UNORM;
        desc.isUAV = true;
        desc.keepInitialState = true;
        desc.initialState = ResourceStates::ShaderResource;
        f.graph->CreateTexture(desc, ResourceFlags::None);
        EXPECT(Contains(ExpectCompileError(f.graph), "keepInitialState"), "a keepInitialState transient is rejected");
    }

    void TestQueueLegality()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId depth = b.Depth("D", 64, 64, Format::D32);
        b.Pass("Prepass", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
            depth = c->SetDepthTarget(depth, {}, false, AccessFlags::Clear, nullptr);
        });
        b.Pass("AsyncDepthTest", PassType::Compute, PassFlags::AsyncCompute | PassFlags::SideEffect, [&](ISetupContext* c) {
            Read(c, depth, Access::DepthRead);
        });
        EXPECT(Contains(ExpectCompileError(f.graph), "illegal on the compute queue"), "a depth read on the compute queue is rejected");
    }

    void TestRegistryOneGraph()
    {
        IResourceRegistry* registry = nullptr;
        CreateResourceRegistry(nullptr, &registry);
        IRenderGraph* first = nullptr;
        IRenderGraph* second = nullptr;
        EXPECT(CreateRenderGraph(registry, &first) == nvrhi::FS_OK, "the first graph");
        EXPECT(CreateRenderGraph(registry, &second) == nvrhi::FE_INVALID_ARGS && !second, "one graph per registry");
        first->Release();
        EXPECT(CreateRenderGraph(registry, &second) == nvrhi::FS_OK, "a new graph after the first is gone");
        second->Release();
        registry->Release();
    }

    void TestExecuteDeviceFree()
    {
        Fixture f;
        Builder b(f.graph);
        ResourceId t = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        b.Pass("P", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) { t = Write(c, t, Access::UAVWrite); });
        auto plan = Compile(f.graph, Profile(ReferenceProfile::D3D12Legacy));
        EXPECT(plan, "compile");
        EXPECT(f.graph->Execute(plan, ExecuteDesc()) == nvrhi::FE_UNSUPPORTED, "Execute without a device is unsupported");
    }

    void TestReferenceProfiles()
    {
        const BackendProfile cuda = Profile(ReferenceProfile::CUDA);
        EXPECT((cuda.passTypes & PassTypeMask::Raster) == PassTypeMask::None && !cuda.depthClears &&
               cuda.barrierModel == BarrierModel::Implicit, "CUDA profile");
        const BackendProfile d3d11 = Profile(ReferenceProfile::D3D11);
        EXPECT(!d3d11.computeQueue && d3d11.recordingMode == RecordingMode::ImmediateSerial, "D3D11 profile");
        const BackendProfile vk = Profile(ReferenceProfile::VulkanSeparateComputeFamily);
        EXPECT(vk.queueOwnershipTransfer && vk.planeModel == PlaneModel::AspectMasks, "Vulkan profile");
        BackendProfile none;
        EXPECT(GetReferenceProfile(ReferenceProfile(99), &none) == nvrhi::FE_INVALID_ARGS, "unknown profile");
    }

    // ------------------------------------------------------------------------------------------ culling and caching

    void TestCulling()
    {
        Fixture f;
        Builder b(f.graph);
        for (const CorpusGraph& graph : GetCorpus())
        {
            if (std::strcmp(graph.name, "culling") == 0)
                graph.build(b, 0);
        }
        auto plan = Compile(f.graph, Profile(ReferenceProfile::D3D12Legacy));
        EXPECT(plan, "compile");
        PlanStatistics stats;
        plan->GetStatistics(stats);
        EXPECT(stats.culledPasses == 1 && stats.scheduledPasses == 3, "the dead pass is culled, the side effect is kept");
    }

    void RecordSmall(Builder& b, uint32_t variant)
    {
        ResourceId t = b.Texture("T", 64, 64, Format::RGBA8_UNORM, false, true);
        ResourceId out = b.Import("Out", 64, 64, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        b.Pass("Write", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { t = Write(c, t, Access::UAVWrite); });
        for (uint32_t i = 0; i < variant; ++i)
            b.Pass("Extra", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { t = Write(c, t, Access::UAVRead | Access::UAVWrite, AccessFlags::None); });
        b.Pass("Read", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            Read(c, t, Access::ShaderRead);
            out = Write(c, out, Access::UAVWrite);
        });
    }

    void TestPlanCache()
    {
        Fixture f;
        const BackendProfile profile = Profile(ReferenceProfile::D3D12Legacy);
        CompileOptions options;
        options.maxCachedPlans = 2;

        { Builder b(f.graph); RecordSmall(b, 0); }
        auto first = Compile(f.graph, profile, options);
        f.graph->Reset();
        { Builder b(f.graph); RecordSmall(b, 0); }
        auto second = Compile(f.graph, profile, options);
        EXPECT(first.Get() == second.Get(), "the same structure hits the cache");
        PlanStatistics stats;
        second->GetStatistics(stats);
        EXPECT(stats.cacheHit, "the statistics report the hit");

        // Two more structures evict the first (least recently used).
        for (uint32_t variant = 1; variant <= 2; ++variant)
        {
            f.graph->Reset();
            Builder b(f.graph);
            RecordSmall(b, variant);
            EXPECT(Compile(f.graph, profile, options), "compile");
        }
        f.graph->Reset();
        { Builder b(f.graph); RecordSmall(b, 0); }
        auto third = Compile(f.graph, profile, options);
        RegistryStatistics registryStats;
        f.registry->GetRegistryStatistics(registryStats);
        EXPECT(registryStats.cachedPlans == 2 && registryStats.planHits == 1 && registryStats.planMisses == 4,
               "LRU eviction: hits " + std::to_string(registryStats.planHits) + ", misses " + std::to_string(registryStats.planMisses));

        // No cache: every frame compiles.
        options.maxCachedPlans = 0;
        f.graph->Reset();
        { Builder b(f.graph); RecordSmall(b, 0); }
        auto uncached = Compile(f.graph, profile, options);
        uncached->GetStatistics(stats);
        EXPECT(!stats.cacheHit, "maxCachedPlans = 0 compiles");

        // A pass-enable mask is not part of the structural hash (13.5): same plan hash, another mask.
        options.maxCachedPlans = 8;
        f.graph->Reset();
        Builder b(f.graph);
        RecordSmall(b, 1);
        auto enabled = Compile(f.graph, profile, options);
        f.graph->Reset();
        Builder b2(f.graph);
        RecordSmall(b2, 1);
        f.graph->SetPassEnabled(b2.Passes()[1], false);
        auto masked = Compile(f.graph, profile, options);
        EXPECT(enabled->GetPlanHash() == masked->GetPlanHash() && enabled->GetPlanEnableMaskHash() != masked->GetPlanEnableMaskHash(),
               "masks share the plan hash");
    }

    // ------------------------------------------------------------------------------------------ golden lists

    std::string GoldenPath(const char* name)
    {
        // Under src/: test_vfs expects the tests folder to hold only include/ and src/.
        return std::string(DONUT_TEST_SOURCE_DIR) + "/src/render/golden/" + name;
    }

    void TestGoldenDeferredFrame()
    {
        const ReferenceProfile profiles[] = { ReferenceProfile::D3D12Legacy, ReferenceProfile::D3D12Enhanced,
            ReferenceProfile::Vulkan, ReferenceProfile::VulkanSeparateComputeFamily };
        std::string listing;
        for (ReferenceProfile kind : profiles)
        {
            for (int async = 0; async < 2; ++async)
            {
                Fixture f;
                Builder b(f.graph);
                BuildDeferredFrame(b, async != 0);
                auto plan = Compile(f.graph, Profile(kind));
                EXPECT(plan, std::string("compile 6.11 on ") + ProfileName(kind) + ": " + Diagnostics(f.graph));
                const std::string report = VerifyOrReport(plan, f.graph);
                EXPECT(report.empty(), std::string("verify 6.11 on ") + ProfileName(kind) + ":\n" + report);
                listing += std::string("== ") + ProfileName(kind) + (async ? ", async SSAO" : "") + " ==\n" + Listing(b, plan) + "\n";

                // Design 6.11, depth: one read state for all readers without async compute; three groups with async
                // SSAO on D3D12 (the compute queue cannot hold DEPTH_READ); one layout on Vulkan with one family.
                std::vector<PlanBarrier> depth = TransitionsOf(plan, b, "Depth");
                const ResourceStates readStates = ResourceStates::DepthRead | ResourceStates::ShaderResource;
                const ResourceStates writeStates = ResourceStates::DepthWrite | ResourceStates::CopyDest |
                    ResourceStates::RenderTarget | ResourceStates::UnorderedAccess;
                uint32_t intoReads = 0;
                for (const PlanBarrier& barrier : depth)
                {
                    const bool acquire = (barrier.flags & nvrhi::BarrierFlags::QueueAcquire) != 0;
                    if ((barrier.flags & nvrhi::BarrierFlags::SplitBegin) == 0 && !acquire && barrier.before != barrier.after &&
                        (barrier.after & readStates) != 0 && (barrier.after & writeStates) == 0)
                        ++intoReads;
                }
                const bool d3d12 = kind == ReferenceProfile::D3D12Legacy || kind == ReferenceProfile::D3D12Enhanced;
                const uint32_t expected = (async && (d3d12 || kind == ReferenceProfile::VulkanSeparateComputeFamily)) ? 3u : 1u;
                if (!(async && kind == ReferenceProfile::VulkanSeparateComputeFamily))
                    EXPECT(intoReads == expected, std::string(ProfileName(kind)) + (async ? " async" : "") + ": depth read groups " +
                        std::to_string(intoReads) + ", expected " + std::to_string(expected) + "\n" + Listing(b, plan));
            }
        }

        // With a separate compute family the default LatticePolicy demotes async SSAO (its ownership pairs and read
        // groups cost more than the overlap gains, 6.8). Forced to stay async, the plan shows the transfers.
        {
            Fixture f;
            Builder b(f.graph);
            BuildDeferredFrame(b, true);
            BackendProfile profile = Profile(ReferenceProfile::VulkanSeparateComputeFamily);
            profile.lattice.asyncOverlapGain = 0xffff;
            auto plan = Compile(f.graph, profile);
            EXPECT(plan, "compile 6.11 with forced async SSAO: " + Diagnostics(f.graph));
            const std::string report = VerifyOrReport(plan, f.graph);
            EXPECT(report.empty(), "verify 6.11 with forced async SSAO:\n" + report);
            PlanStatistics stats;
            plan->GetStatistics(stats);
            EXPECT(stats.queueTransfers > 0, "ownership transfers between the families\n" + Listing(b, plan));
            listing += "== VulkanSeparateComputeFamily, async SSAO kept (asyncOverlapGain 65535) ==\n" + Listing(b, plan) + "\n";
        }

        const std::string path = GoldenPath("deferred_frame_6_11.txt");
        const char* update = std::getenv("RDG_UPDATE_GOLDEN");
        if (update && *update == '1')
        {
            std::ofstream file(path, std::ios::binary);
            file << listing;
            std::printf("  wrote %s\n", path.c_str());
            return;
        }
        std::ifstream file(path, std::ios::binary);
        EXPECT(file.good(), "golden file missing: " + path + " (run with RDG_UPDATE_GOLDEN=1)");
        std::stringstream golden;
        golden << file.rdbuf();
        std::string expected = golden.str();
        expected.erase(std::remove(expected.begin(), expected.end(), '\r'), expected.end());
        if (expected != listing)
        {
            std::ofstream actual(path + ".actual", std::ios::binary);
            actual << listing;
            EXPECT(false, "the 6.11 barrier lists differ from " + path + "; the actual lists are in " + path + ".actual");
        }
    }

    // ------------------------------------------------------------------------------------------ the corpus

    void TestCorpus()
    {
        uint32_t verified = 0;
        uint32_t switches = 0;
        uint32_t graphs = 0;
        std::string failures;
        for (const CorpusGraph& graph : GetCorpus())
        {
            ++graphs;
            for (ReferenceProfile kind : c_AllProfiles)
            {
                if (kind == ReferenceProfile::CUDA && graph.raster)
                    continue;
                const BackendProfile profile = Profile(kind);
                for (int placed = 0; placed < (profile.maxMemoryTier == MemoryTier::Placed ? 2 : 1); ++placed)
                {
                    CompileOptions options;
                    options.placedResources = placed != 0;
                    const std::string where = std::string(graph.name) + " on " + ProfileName(kind) + (placed ? " (placed)" : "");

                    Fixture f;
                    Builder b(f.graph);
                    graph.build(b, 0);
                    auto planA = Compile(f.graph, profile, options);
                    if (!planA)
                    {
                        failures += where + ": compile failed:\n" + Diagnostics(f.graph);
                        continue;
                    }
                    std::string report = VerifyOrReport(planA, f.graph);
                    if (!report.empty())
                    {
                        failures += where + ":\n" + report + Listing(b, planA);
                        continue;
                    }
                    ++verified;

                    // The plan switch: the variant's plan after this one (6.12).
                    f.graph->Reset();
                    Builder b2(f.graph);
                    graph.build(b2, 1);
                    auto planB = Compile(f.graph, profile, options);
                    if (!planB)
                    {
                        failures += where + " (variant): compile failed:\n" + Diagnostics(f.graph);
                        continue;
                    }
                    report = VerifyOrReport(planB, f.graph, planA);
                    if (!report.empty())
                    {
                        failures += where + " (switch):\n" + report + Listing(b2, planB);
                        continue;
                    }
                    ++switches;
                }
            }
        }
        std::printf("  %u graphs, %u verified plans, %u verified plan switches\n", graphs, verified, switches);
        EXPECT(failures.empty(), failures);
        EXPECT(graphs >= 40 && verified >= 160, "the corpus covers at least 40 graphs x 4 profiles");
    }

    void TestStrictValidation()
    {
        Fixture f;
        Builder b(f.graph);
        BuildDeferredFrame(b, true);
        CompileOptions options;
        options.debugFlags = DebugFlags::StrictValidation;
        EXPECT(Compile(f.graph, Profile(ReferenceProfile::D3D12Enhanced), options), "StrictValidation runs the verifier: " + Diagnostics(f.graph));
    }

    void TestSkipAliasingBarriersFails()
    {
        // 11.4: the negative test; dropping the aliasing barriers must make the verifier fail.
        for (int skip = 0; skip < 2; ++skip)
        {
            Fixture f;
            Builder b(f.graph);
            for (const CorpusGraph& graph : GetCorpus())
            {
                if (std::strcmp(graph.name, "aliasing_chain") == 0)
                    graph.build(b, 0);
            }
            CompileOptions options;
            options.placedResources = true;
            if (skip)
                options.debugFlags = DebugFlags::SkipAliasingBarriers;
            auto plan = Compile(f.graph, Profile(ReferenceProfile::D3D12Legacy), options);
            EXPECT(plan, "compile");
            PlanStatistics stats;
            plan->GetStatistics(stats);
            const std::string report = VerifyOrReport(plan, f.graph);
            if (skip)
                EXPECT(!report.empty() && Contains(report, "aliasing barrier"), "SkipAliasingBarriers is detected: " + report);
            else
                EXPECT(report.empty() && stats.aliasingBarriers > 0, "the aliasing chain aliases: " + report);
        }
    }

    void TestSplitBarriers()
    {
        auto record = [](Builder& b) {
            ResourceId a = b.Texture("A", 256, 256, Format::RGBA8_UNORM, true, false);
            ResourceId x = b.Texture("X", 256, 256, Format::RGBA8_UNORM, true, false);
            ResourceId y = b.Texture("Y", 256, 256, Format::RGBA8_UNORM, true, false);
            ResourceId out = b.Import("Out", 256, 256, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
            b.Pass("WriteA", PassType::Raster, PassFlags::None, [&](ISetupContext* c) { a = c->SetColorTarget(0, a, {}, AccessFlags::FullWrite, nullptr); });
            b.Pass("WriteX", PassType::Raster, PassFlags::None, [&](ISetupContext* c) { x = c->SetColorTarget(0, x, {}, AccessFlags::FullWrite, nullptr); });
            b.Pass("WriteY", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, x, Access::ShaderRead, ShaderType::Pixel);
                y = c->SetColorTarget(0, y, {}, AccessFlags::FullWrite, nullptr);
            });
            b.Pass("ReadA", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, a, Access::ShaderRead);
                Read(c, y, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        };
        for (uint32_t granularity = 0; granularity <= 2; granularity += 2)
        {
            Fixture f;
            Builder b(f.graph);
            record(b);
            CompileOptions options;
            options.recordGranularity = granularity;
            auto plan = Compile(f.graph, Profile(ReferenceProfile::D3D12Legacy), options);
            EXPECT(plan, "compile");
            EXPECT(VerifyOrReport(plan, f.graph).empty(), "verify");
            PlanStatistics stats;
            plan->GetStatistics(stats);
            if (granularity == 0)
                EXPECT(stats.splitBarriers == 1, "A splits from WriteA's epilogue to ReadA's prologue\n" + Listing(b, plan));
            else
                EXPECT(stats.splitBarriers == 0 && stats.segments == 2, "a split never crosses a command-list boundary\n" + Listing(b, plan));
        }
    }

    void TestUavOrder()
    {
        for (uint32_t variant = 0; variant < 2; ++variant)
        {
            Fixture f;
            Builder b(f.graph);
            for (const CorpusGraph& graph : GetCorpus())
            {
                if (std::strcmp(graph.name, "relaxed_uav_run") == 0)
                    graph.build(b, variant);
            }
            auto plan = Compile(f.graph, Profile(ReferenceProfile::D3D12Legacy));
            EXPECT(plan, "compile");
            PlanStatistics stats;
            plan->GetStatistics(stats);
            const uint32_t expected = variant ? 2 : 0;
            EXPECT(stats.uavBarriers == expected, "UAV barriers " + std::to_string(stats.uavBarriers) + ", expected " +
                   std::to_string(expected) + "\n" + Listing(b, plan));
        }
    }

    void TestPlanesAndMips()
    {
        const char* graphs[] = { "depth_stencil_planes", "bloom_mip_chain", "shadow_cascades_slices" };
        for (const char* name : graphs)
        {
            Fixture f;
            Builder b(f.graph);
            for (const CorpusGraph& graph : GetCorpus())
            {
                if (std::strcmp(graph.name, name) == 0)
                    graph.build(b, 0);
            }
            auto plan = Compile(f.graph, Profile(ReferenceProfile::D3D12Legacy));
            EXPECT(plan, "compile");
            bool partial = false;
            for (uint32_t i = 0; i < plan->GetScheduledPassCount(); ++i)
            {
                for (uint32_t slot = 0; slot < 3; ++slot)
                {
                    BarrierBatch batch;
                    plan->GetBatch(i, BatchSlot(slot), batch);
                    for (uint32_t k = 0; k < batch.count; ++k)
                    {
                        const nvrhi::SubresourceRegion& region = batch.barriers[k].region;
                        if (region.planes != nvrhi::PlaneMask::All || region.mipCount != nvrhi::SubresourceRegion::AllMipLevels ||
                            region.sliceCount != nvrhi::SubresourceRegion::AllArraySlices)
                            partial = true;
                    }
                }
            }
            EXPECT(partial, std::string(name) + ": transitions name planes, mips or slices\n" + Listing(b, plan));
        }
    }

    // ------------------------------------------------------------------------------------------ helpers

    struct Counter
    {
        int* count;
        explicit Counter(int* c) : count(c) { }
        Counter(const Counter& other) : count(other.count) { }
        ~Counter() { ++*count; }
    };

    void TestLambdaPassDestructors()
    {
        Fixture f;
        int destroyed = 0;
        {
            Counter counter(&destroyed);
            ResourceId out = InvalidResourceId;
            nvrhi::TextureDesc desc;
            desc.width = 8;
            desc.height = 8;
            desc.format = Format::RGBA8_UNORM;
            desc.isUAV = true;
            out = f.graph->ImportTextureDesc(desc, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
            AddLambdaPass<NoData>(f.graph, PassDesc{ "Captures", PassType::Compute, PassFlags::None, 0 },
                [&out](ISetupContext* c, NoData&) { out = Write(c, out, Access::UAVWrite); },
                [counter](IPassContext*, const NoData&) { (void)counter; });
        }
        const int before = destroyed;
        f.graph->Reset();
        EXPECT(destroyed == before + 1, "Reset runs the destructors of the lambdas' captures");
    }

    struct SceneTargets { ResourceId depth; ResourceId color; };
    struct AmbientOcclusion { ResourceId ao; };

    void TestFrameData()
    {
        Fixture f;
        FrameData frame;
        EXPECT(!frame.Find<SceneTargets>(), "empty");
        SceneTargets& scene = frame.Create<SceneTargets>(f.graph);
        scene.depth = MakeResourceId(1, 2);
        EXPECT(frame.Find<SceneTargets>() == &scene && frame.Get<SceneTargets>().depth == MakeResourceId(1, 2), "find");
        EXPECT(!frame.Find<AmbientOcclusion>(), "another type is not found");
        frame.Clear();
        EXPECT(!frame.Find<SceneTargets>(), "cleared");
    }

    void TestDumps()
    {
        Fixture f;
        Builder b(f.graph);
        BuildDeferredFrame(b, true);
        auto plan = Compile(f.graph, Profile(ReferenceProfile::Vulkan));
        EXPECT(plan, "compile");
        const std::string json = Dump(plan);
        EXPECT(Contains(json, "\"passes\"") && Contains(json, "DeferredLighting") && Contains(json, "\"statistics\""), "JSON dump");
        nvrhi::string graphviz;
        EXPECT(plan->WriteDump(DumpFormat::Graphviz, graphviz) == nvrhi::FS_OK && Contains(graphviz.c_str(), "digraph"), "Graphviz dump");
        ResourceLifetime lifetime;
        EXPECT(plan->GetResourceLifetime(0, lifetime) == nvrhi::FS_OK && lifetime.firstPass == 0, "lifetime of the depth buffer");
        LoadOp load;
        StoreOp store;
        EXPECT(plan->GetAttachmentOps(0, 8, load, store) == nvrhi::FS_OK && load == LoadOp::Clear, "the depth prepass clears");
    }

    void TestSteadyStateAllocations()
    {
        // 10.3: no heap allocation per steady-state frame (recording and a plan-cache hit).
        Fixture f;
        BenchSetup setup;
        setup.Init(64, 4);
        const BackendProfile profile = Profile(ReferenceProfile::D3D12Legacy);
        const CompileOptions options;
        for (uint32_t frame = 0; frame < 3; ++frame)
        {
            f.graph->Reset();
            RecordBenchFrame(f.graph, setup);
            ICompiledRenderGraph* plan = nullptr;
            EXPECT(f.graph->Compile(profile, options, &plan) == nvrhi::FS_OK, "compile: " + Diagnostics(f.graph));
            plan->Release();
        }
        g_Allocations = 0;
        g_CountAllocations = true;
        for (uint32_t frame = 0; frame < 10; ++frame)
        {
            f.graph->Reset();
            RecordBenchFrame(f.graph, setup);
            ICompiledRenderGraph* plan = nullptr;
            f.graph->Compile(profile, options, &plan);
            if (plan)
                plan->Release();
        }
        g_CountAllocations = false;
        EXPECT(g_Allocations == 0, std::to_string(g_Allocations.load()) + " heap allocations in 10 steady-state frames");
    }

    struct Test
    {
        const char* name;
        void (*run)();
    };

    const Test c_Tests[] = {
        { "ids", TestIds },
        { "id_from_earlier_frame", TestIdFromEarlierFrame },
        { "setup_context_closed", TestSetupContextClosed },
        { "nested_add_pass", TestNestedAddPass },
        { "frozen", TestFrozen },
        { "history_write", TestHistoryWrite },
        { "pass_local_foreign", TestPassLocalForeign },
        { "read_before_write", TestReadBeforeWrite },
        { "partial_first_write", TestPartialFirstWrite },
        { "overlap_conflict", TestOverlapConflict },
        { "exclusion_conflict", TestExclusionConflict },
        { "async_back_buffer", TestAsyncBackBuffer },
        { "cuda_rules", TestCudaRules },
        { "keep_initial_state_transient", TestKeepInitialStateTransient },
        { "queue_legality", TestQueueLegality },
        { "registry_one_graph", TestRegistryOneGraph },
        { "execute_device_free", TestExecuteDeviceFree },
        { "reference_profiles", TestReferenceProfiles },
        { "culling", TestCulling },
        { "plan_cache", TestPlanCache },
        { "golden_deferred_frame", TestGoldenDeferredFrame },
        { "corpus", TestCorpus },
        { "strict_validation", TestStrictValidation },
        { "skip_aliasing_barriers_fails", TestSkipAliasingBarriersFails },
        { "split_barriers", TestSplitBarriers },
        { "uav_order", TestUavOrder },
        { "planes_and_mips", TestPlanesAndMips },
        { "lambda_pass_destructors", TestLambdaPassDestructors },
        { "frame_data", TestFrameData },
        { "dumps", TestDumps },
        { "steady_state_allocations", TestSteadyStateAllocations },
    };
}

int main(int argc, char** argv)
{
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int failed = 0;
    int run = 0;
    for (const Test& test : c_Tests)
    {
        if (filter && !std::strstr(test.name, filter))
            continue;
        ++run;
        try
        {
            test.run();
            std::printf("[ OK ] %s\n", test.name);
        }
        catch (const std::exception& e)
        {
            ++failed;
            std::printf("[FAIL] %s\n%s\n", test.name, e.what());
        }
    }
    std::printf("%d of %d tests passed\n", run - failed, run);
    return failed ? 1 : 0;
}
