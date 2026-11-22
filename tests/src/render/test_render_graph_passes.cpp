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

// donut's passes in a render graph, device-free (aggregate plan, step P1.5): every AddToGraph of design 13.3 is
// recorded into one frame on a null device, compiled for the D3D12 reference profile and verified, and the accesses
// each use declared are compared against a golden list. The passes' graph constructions create nothing until
// Prepare, so no device is needed.
//
//     test_render_graph_passes [filter]
//     RDG_UPDATE_GOLDEN=1      rewrites the golden file instead of comparing

#include "rdg_donut_frame.h"

#include <donut/render/RenderGraphLambdaPass.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::render;
using nvrhi::Format;
using nvrhi::ResourceStates;

#define EXPECT(condition, message) \
    do { if (!(condition)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " + (message)); } while (0)

namespace
{
    struct Frame
    {
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;

        Frame()
        {
            IResourceRegistry* r = nullptr;
            EXPECT(CreateResourceRegistry(nullptr, &r) == nvrhi::FS_OK, "registry");
            registry = nvrhi::TakeOver(r);
            IRenderGraph* g = nullptr;
            EXPECT(CreateRenderGraph(r, &g) == nvrhi::FS_OK, "graph");
            graph = nvrhi::TakeOver(g);
            graph->Reset();
        }

        std::string Diagnostics()
        {
            nvrhi::string text;
            graph->GetDiagnostics(text);
            return text.c_str();
        }

        // Compiles for a reference profile and runs the device-free verifier.
        nvrhi::AutoPtr<ICompiledRenderGraph> Compile(ReferenceProfile reference = ReferenceProfile::D3D12Legacy)
        {
            BackendProfile profile;
            EXPECT(GetReferenceProfile(reference, &profile) == nvrhi::FS_OK, "profile");
            ICompiledRenderGraph* plan = nullptr;
            EXPECT(graph->Compile(profile, CompileOptions(), &plan) == nvrhi::FS_OK, "compile:\n" + Diagnostics());
            nvrhi::AutoPtr<ICompiledRenderGraph> result = nvrhi::TakeOver(plan);
            nvrhi::string report;
            EXPECT(VerifyPlan(plan, graph, nullptr, &report) == nvrhi::FS_OK, std::string("verify:\n") + report.c_str());
            return result;
        }
    };

    nvrhi::AutoPtr<engine::PlanarView> MakeView(float x0, float x1, float width, float height, int slice = 0)
    {
        nvrhi::AutoPtr<engine::PlanarView> view = MAKE_RC_OBJ_PTR(engine::PlanarView);
        view->SetViewport(nvrhi::Viewport(x0, x1, 0.f, height, 0.f, 1.f));
        view->SetMatrices(dm::affine3::identity(), dm::perspProjD3DStyleReverse(dm::radians(60.f), width / height, 0.1f));
        view->SetArraySlice(slice);
        view->UpdateCache();
        return view;
    }

    nvrhi::TextureDesc Texture(uint32_t width, uint32_t height, Format format, bool renderTarget, bool uav, uint32_t mips = 1,
                               uint32_t slices = 1)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.isRenderTarget = renderTarget;
        desc.isUAV = uav;
        desc.mipLevels = mips;
        desc.arraySize = slices;
        if (slices > 1)
            desc.dimension = nvrhi::TextureDimension::Texture2DArray;
        return desc;
    }

    // The quoted strings of a JSON array that starts at `from` (at its '[').
    std::vector<std::string> Strings(const std::string& line, size_t from)
    {
        std::vector<std::string> values;
        size_t i = from + 1;
        while (i < line.size() && line[i] != ']')
        {
            if (line[i] != '"')
            {
                ++i;
                continue;
            }
            std::string value;
            for (++i; i < line.size() && line[i] != '"'; ++i)
            {
                if (line[i] == '\\' && i + 1 < line.size())
                    ++i;
                value += line[i];
            }
            values.push_back(value);
            ++i;
        }
        return values;
    }

    std::string Field(const std::string& line, const char* name)
    {
        const std::string key = std::string("\"") + name + "\": \"";
        const size_t at = line.find(key);
        if (at == std::string::npos)
            return std::string();
        const size_t begin = at + key.size();
        return line.substr(begin, line.find('"', begin) - begin);
    }

    // The passes of a plan's JSON dump in schedule order, with the uses they declared.
    std::string AccessListing(ICompiledRenderGraph* plan)
    {
        nvrhi::string dump;
        EXPECT(plan->WriteDump(DumpFormat::Json, dump) == nvrhi::FS_OK, "dump");
        std::istringstream lines(dump.c_str());
        std::string line;
        std::string listing;
        bool inPasses = false;
        while (std::getline(lines, line))
        {
            if (line.find("\"passes\": [") != std::string::npos)
            {
                inPasses = true;
                continue;
            }
            if (inPasses && line.find("  ],") == 0)
                break;
            if (!inPasses)
                continue;
            listing += Field(line, "name") + " [" + Field(line, "type") + ", " + Field(line, "mode") + ", " + Field(line, "queue") + "]\n";
            const size_t uses = line.find("\"uses\": [");
            EXPECT(uses != std::string::npos, "the dump lists the uses");
            for (const std::string& use : Strings(line, uses + 8))   // at the [
                listing += "    " + use + "\n";
        }
        return listing;
    }

    std::string GoldenPath(const char* name)
    {
        return std::string(DONUT_TEST_SOURCE_DIR) + "/src/render/golden/" + name;
    }

    void CompareGolden(const char* name, const std::string& listing)
    {
        const std::string path = GoldenPath(name);
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
            EXPECT(false, std::string("the access lists differ from ") + path + "; the actual lists are in " + path + ".actual");
        }
    }

    // ------------------------------------------------------------------------------------------ the frame

    // Every graph-capable donut pass in one frame on a null device.
    void TestGoldenPassAccesses()
    {
        Frame f;
        rdgtest::DonutScene scene;
        rdgtest::DonutPasses passes;
        passes.Create(nullptr, nullptr, nullptr);
        rdgtest::DonutImports imports;
        imports.shadowMap = f.graph->ImportTextureDesc(rdgtest::ShadowMapDesc(), ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        imports.environmentMap = f.graph->ImportTextureDesc(rdgtest::EnvironmentMapDesc(), ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        imports.mipChain = f.graph->ImportTextureDesc(rdgtest::MipChainDesc(), ResourceStates::ShaderResource, ResourceStates::ShaderResource);
        imports.backBuffer = f.graph->ImportTextureDesc(rdgtest::BackBufferDesc(scene), ResourceStates::RenderTarget, ResourceStates::RenderTarget);
        rdgtest::RecordDonutFrame(f.graph, passes, scene, imports);

        auto plan = f.Compile();
        PlanStatistics stats;
        plan->GetStatistics(stats);
        EXPECT(stats.culledPasses == 0, "no pass of the frame is culled (" + std::to_string(stats.culledPasses) + ")");
        CompareGolden("pass_accesses.txt", AccessListing(plan));

        // The same frame verifies on the other reference profiles. RDG_DUMP_PASSES=1 prints their plans.
        const ReferenceProfile others[] = { ReferenceProfile::D3D12Enhanced, ReferenceProfile::Vulkan,
                                            ReferenceProfile::VulkanSeparateComputeFamily, ReferenceProfile::D3D11 };
        for (ReferenceProfile other : others)
        {
            auto otherPlan = f.Compile(other);
            if (std::getenv("RDG_DUMP_PASSES"))
            {
                nvrhi::string dump;
                otherPlan->WriteDump(DumpFormat::Json, dump);
                std::printf("== profile %d\n%s\n", int(other), dump.c_str());
            }
        }
    }

    // 13.2: per-view uses of one pass over a composite view have disjoint transients, which share memory.
    void TestPerViewFolding()
    {
        Frame f;
        IRenderGraph* g = f.graph;
        nvrhi::AutoPtr<engine::CompositeView> stereo = MAKE_RC_OBJ_PTR(engine::CompositeView);
        nvrhi::AutoPtr<engine::PlanarView> left = MakeView(0.f, 64.f, 64.f, 64.f);
        nvrhi::AutoPtr<engine::PlanarView> right = MakeView(64.f, 128.f, 64.f, 64.f);
        stereo->AddView(left);
        stereo->AddView(right);
        nvrhi::AutoPtr<BloomPass> bloomPass = MAKE_RC_OBJ_PTR(BloomPass, nullptr, nullptr, nullptr);
        ResourceId color = g->ImportTextureDesc(Texture(128, 64, Format::RGBA16_FLOAT, true, false),
                                                ResourceStates::RenderTarget, ResourceStates::RenderTarget);
        BloomGraphInputs bloom;
        bloom.view = stereo;
        bloom.color = color;
        EXPECT(bloomPass->AddToGraph(g, bloom).color != color, "bloom writes the colour");
        EXPECT(g->GetPassCount() == 10, "five uses per view (" + std::to_string(g->GetPassCount()) + ")");
        auto plan = f.Compile();

        uint32_t transients = 0;
        for (uint32_t r = 0; r < g->GetResourceCount(); ++r)
        {
            ResourceLifetime lifetime;
            if (plan->GetResourceLifetime(r, lifetime) != nvrhi::FS_OK || lifetime.firstPass == ~0u || r == GetResourceIndex(color))
                continue;
            ++transients;
        }
        nvrhi::string dump;
        plan->WriteDump(DumpFormat::Json, dump);
        std::set<std::string> physicals;
        std::istringstream lines(dump.c_str());
        std::string line;
        while (std::getline(lines, line))
        {
            if (line.find("\"kind\": \"transient\"") == std::string::npos)
                continue;
            const size_t at = line.find("\"physical\": ");
            physicals.insert(line.substr(at + 12, line.find(',', at) - at - 12));
        }
        EXPECT(transients == 8, "four transients per view (" + std::to_string(transients) + ")");
        EXPECT(!physicals.empty() && physicals.size() < transients,
               "the views share physical textures (" + std::to_string(physicals.size()) + " for " + std::to_string(transients) + ")");
    }

    // 13.2: a pass that cannot run in a graph fails AddPass, and Compile names it.
    void TestUnsupportedPass()
    {
        struct Unsupported : nvrhi::ObjectImpl<IGpuPass>
        {
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Unsupported)
            NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
            NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
            NVRHI_END_INTERFACE_TABLE()
            nvrhi::FRESULT Setup(ISetupContext*, void*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
            nvrhi::FRESULT Prepare(IPrepareContext*, void*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
            nvrhi::FRESULT Execute(IPassContext*, const void*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
            void OnResourcesReleased() noexcept override { }
        };
        Frame f;
        nvrhi::AutoPtr<Unsupported> pass = MAKE_RC_OBJ_PTR(Unsupported);
        EXPECT(f.graph->AddPass(PassDesc{ "PixelReadback", PassType::Copy, PassFlags::None, 0 }, pass, nullptr, nullptr) != nvrhi::FS_OK,
               "AddPass fails");
        ICompiledRenderGraph* plan = nullptr;
        BackendProfile profile;
        GetReferenceProfile(ReferenceProfile::D3D12Legacy, &profile);
        EXPECT(f.graph->Compile(profile, CompileOptions(), &plan) != nvrhi::FS_OK, "Compile fails");
        EXPECT(f.Diagnostics().find("PixelReadback") != std::string::npos, "the diagnostic names the pass:\n" + f.Diagnostics());
    }

    // ECS sub-plan item E5.3 (open point 5): GeometryGraphInputs::bufferReads. A geometry pass over two views reads the
    // scene's buffers after the passes that write them (an upload and a skinning pass, as lambda passes): a buffer made
    // for vertex input as a vertex-buffer and SRV read, any other as an SRV read, in the vertex and pixel stages; its
    // uses are Restore; the geometry buffer and an idle skinned vertex buffer, read in their resting states, get no
    // barrier; the plan verifies on every reference profile. Without buffer reads (the golden frame above) the pass
    // declares none and stays Tracked.
    void TestGeometryBufferReads()
    {
        struct NoData
        {
            uint32_t unused;
        };
        Frame f;
        IRenderGraph* g = f.graph;
        nvrhi::AutoPtr<engine::CompositeView> stereo = MAKE_RC_OBJ_PTR(engine::CompositeView);
        nvrhi::AutoPtr<engine::PlanarView> left = MakeView(0.f, 64.f, 64.f, 64.f);
        nvrhi::AutoPtr<engine::PlanarView> right = MakeView(64.f, 128.f, 64.f, 64.f);
        stereo->AddView(left);
        stereo->AddView(right);

        auto buffer = [](const char* name, bool vertexBuffer)
        {
            nvrhi::BufferDesc desc;
            desc.byteSize = 4096;
            desc.isVertexBuffer = vertexBuffer;
            desc.canHaveRawViews = true;
            desc.canHaveUAVs = true;
            desc.debugName = name;
            return desc;
        };
        // the resting states of the graph path (engine::c_GpuSceneRestingState, c_SkinnedVertexBufferRestingState)
        const ResourceStates sceneResting = ResourceStates::ShaderResource;
        const ResourceStates skinnedResting = ResourceStates::VertexBuffer | ResourceStates::ShaderResource;
        ResourceId instances = g->ImportBufferDesc(buffer("InstanceBuffer", true), sceneResting, sceneResting);
        const ResourceId geometries = g->ImportBufferDesc(buffer("GeometryBuffer", false), sceneResting, sceneResting);
        ResourceId skinned = g->ImportBufferDesc(buffer("SkinnedVertices", true), skinnedResting, skinnedResting);
        const ResourceId idle = g->ImportBufferDesc(buffer("IdleSkinnedVertices", true), skinnedResting, skinnedResting);
        AddLambdaPass<NoData>(g, PassDesc{ "Upload", PassType::Compute, PassFlags::None, 0 },
            [&](ISetupContext* setup, NoData&)
            {
                instances = setup->Write(instances, Access::UAVWrite, nvrhi::ShaderType::Compute, {}, AccessFlags::None);
            },
            [](IPassContext*, const NoData&) { });
        AddLambdaPass<NoData>(g, PassDesc{ "Skinning", PassType::Compute, PassFlags::None, 0 },
            [&](ISetupContext* setup, NoData&)
            {
                skinned = setup->Write(skinned, Access::UAVWrite, nvrhi::ShaderType::Compute, {}, AccessFlags::None);
            },
            [](IPassContext*, const NoData&) { });

        const ResourceId color = g->ImportTextureDesc(Texture(128, 64, Format::RGBA8_UNORM, true, false),
                                                      ResourceStates::RenderTarget, ResourceStates::RenderTarget);
        nvrhi::TextureDesc depthDesc = Texture(128, 64, Format::D32, true, false);
        depthDesc.debugName = "Depth";
        const ResourceId depth = g->ImportTextureDesc(depthDesc, ResourceStates::DepthWrite, ResourceStates::DepthWrite);
        nvrhi::AutoPtr<GBufferFillPass> gbuffer = MAKE_RC_OBJ_PTR(GBufferFillPass, nullptr, nullptr);
        GBufferFillPass::Context context;
        nvrhi::AutoPtr<PassthroughDrawStrategy> strategy = MAKE_RC_OBJ_PTR(PassthroughDrawStrategy);
        const ResourceId reads[] = { instances, geometries, skinned, idle, InvalidResourceId };
        GeometryGraphInputs inputs;
        inputs.name = "GBufferFill";
        inputs.view = stereo;
        inputs.drawStrategy = strategy;
        inputs.passContext = &context;
        inputs.colorTargets[0] = color;
        inputs.colorTargetCount = 1;
        inputs.depth = depth;
        inputs.clearTargets = true;
        inputs.bufferReads = reads;
        inputs.bufferReadCount = uint32_t(std::size(reads));
        gbuffer->AddToGraph(g, inputs);
        EXPECT(g->GetPassCount() == 4, "the writers and one use per view (" + std::to_string(g->GetPassCount()) + ")");

        auto plan = f.Compile();
        PlanStatistics stats;
        plan->GetStatistics(stats);
        EXPECT(stats.culledPasses == 0, "nothing culled");
        const std::string listing = AccessListing(plan);
        const std::string expected =
            "Upload [compute, native, graphics]\n"
            "    InstanceBuffer UAVWrite stages=0x0020 mips=0+1 slices=0+1\n"
            "Skinning [compute, native, graphics]\n"
            "    SkinnedVertices UAVWrite stages=0x0020 mips=0+1 slices=0+1\n";
        EXPECT(listing.compare(0, expected.size(), expected) == 0, "the writers first:\n" + listing);
        size_t uses = 0;
        for (size_t at = listing.find("GBufferFill [raster, restore, graphics]"); at != std::string::npos;
             at = listing.find("GBufferFill [raster, restore, graphics]", at + 1))
            ++uses;
        EXPECT(uses == 2, "two Restore uses after the writers:\n" + listing);
        auto count = [&listing](const std::string& text)
        {
            size_t n = 0;
            for (size_t at = listing.find(text); at != std::string::npos; at = listing.find(text, at + 1))
                ++n;
            return n;
        };
        EXPECT(count("InstanceBuffer VertexBuffer|ShaderRead stages=0x0011") == 2 &&
               count("GeometryBuffer ShaderRead stages=0x0011") == 2 &&
               count("    SkinnedVertices VertexBuffer|ShaderRead stages=0x0011") == 2 &&
               count("IdleSkinnedVertices VertexBuffer|ShaderRead stages=0x0011") == 2,
               "each use reads the four buffers in the vertex and pixel stages:\n" + listing);

        // Barriers: the written buffers move to the read state before the first use and back to rest at the end; the
        // buffers read in their resting states get none.
        nvrhi::string dump;
        EXPECT(plan->WriteDump(DumpFormat::Json, dump) == nvrhi::FS_OK, "dump");
        std::istringstream lines(dump.c_str());
        std::string line;
        uint32_t passLines = 0;
        size_t instanceBarriers = 0, skinnedBarriers = 0;
        while (std::getline(lines, line))
        {
            // a pass's line: its uses, then its barrier lists (prologue, postClear, epilogue)
            const size_t barriers = line.find("\"prologue\": [");
            if (line.find("\"uses\": [") == std::string::npos || barriers == std::string::npos)
                continue;
            ++passLines;
            const std::string lists = line.substr(barriers);
            EXPECT(lists.find("GeometryBuffer") == std::string::npos, "a barrier for the geometry buffer:\n" + line);
            EXPECT(lists.find("IdleSkinnedVertices") == std::string::npos, "a barrier for the idle buffer:\n" + line);
            for (size_t at = lists.find("InstanceBuffer"); at != std::string::npos; at = lists.find("InstanceBuffer", at + 1))
                ++instanceBarriers;
            for (size_t at = lists.find("SkinnedVertices"); at != std::string::npos; at = lists.find("SkinnedVertices", at + 1))
                ++skinnedBarriers;
        }
        EXPECT(passLines == 4, "four passes in the dump (" + std::to_string(passLines) + ")");
        // into the write and from the write to the reads; the instance buffer also back to rest at the end of the frame
        // (its read state is wider than ShaderResource), the skinned one is read in its resting state
        EXPECT(instanceBarriers >= 3 && skinnedBarriers >= 2,
               "the written buffers move between their states (" + std::to_string(instanceBarriers) + ", " +
                   std::to_string(skinnedBarriers) + ")");

        const ReferenceProfile others[] = { ReferenceProfile::D3D12Enhanced, ReferenceProfile::Vulkan,
                                            ReferenceProfile::VulkanSeparateComputeFamily, ReferenceProfile::D3D11 };
        for (ReferenceProfile other : others)
            f.Compile(other);
    }

    struct Test
    {
        const char* name;
        void (*run)();
    };

    const Test c_Tests[] = {
        { "golden_pass_accesses", TestGoldenPassAccesses },
        { "per_view_folding", TestPerViewFolding },
        { "unsupported_pass", TestUnsupportedPass },
        { "geometry_buffer_reads", TestGeometryBufferReads },
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
