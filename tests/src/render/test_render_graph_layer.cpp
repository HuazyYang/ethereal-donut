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

// app::RenderGraphLayer on a headless DeviceManager (aggregate plan, step P1.5): the layer resets, builds, compiles
// and executes each frame, applies pass toggles by name from the next frame, and releases its resources on a resize,
// under the debug runtime and the nvrhi validation layer.
//
//     test_render_graph_layer d3d11 | d3d12 | vulkan

#include <donut/app/RenderGraphLayer.h>
#include <donut/core/log.h>
#include <donut/render/RenderGraphLambdaPass.h>

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace donut;
using namespace donut::render;

namespace
{
    std::atomic<uint32_t> g_Errors{ 0 };

    void check(bool condition, const std::string& what)
    {
        if (!condition)
            throw std::runtime_error("check failed: " + what);
        std::printf("ok: %s\n", what.c_str());
    }

    struct NoData
    {
        uint32_t unused;
    };

    // A buffer filled, copied twice (the second copy can be toggled off and passes through) and read back.
    class TestLayer : public app::RenderGraphLayer
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        explicit TestLayer(app::DeviceManager* deviceManager) : RenderGraphLayer(deviceManager)
        {
            GetDevice()->createBuffer(nvrhi::BufferDesc().setByteSize(64).setCpuAccess(nvrhi::CpuAccessMode::Read)
                .setInitialState(nvrhi::ResourceStates::CopyDest).setDebugName("readback"), readback.ReleaseAndGetAddressOf());
        }

        void BuildGraph(IRenderGraph* graph, FrameData&, ResourceId) override
        {
            ++frames;
            for (uint32_t i = 0; i < 16; ++i)
                pattern[i] = frames * 100u + i;
            const nvrhi::BufferDesc desc = nvrhi::BufferDesc().setByteSize(64).setCanHaveUAVs(true);
            ResourceId a = graph->CreateBuffer(desc, ResourceFlags::None);
            ResourceId b = graph->CreateBuffer(desc, ResourceFlags::None);
            ResourceId out = graph->ImportBuffer(readback, nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::CopyDest);
            const uint32_t* data = pattern;
            AddLambdaPass<NoData>(graph, PassDesc{ "Fill", PassType::Copy, PassFlags::None, 0 },
                [&](ISetupContext* s, NoData&) { a = s->Write(a, Access::CopyDest, nvrhi::ShaderType::None, {}, AccessFlags::FullWrite); },
                [a, data](IPassContext* c, const NoData&) { c->GetCommandList()->writeBuffer(c->GetBuffer(a), data, 64); });
            AddLambdaPass<NoData>(graph, PassDesc{ "Copy", PassType::Copy, PassFlags::None, 0 },
                [&](ISetupContext* s, NoData&) {
                    s->Read(a, Access::CopySource, nvrhi::ShaderType::None, {});
                    b = s->Write(b, Access::CopyDest, nvrhi::ShaderType::None, {}, AccessFlags::FullWrite);
                },
                [a, b](IPassContext* c, const NoData&) { c->GetCommandList()->copyBuffer(c->GetBuffer(b), 0, c->GetBuffer(a), 0, 64); });
            AddLambdaPass<NoData>(graph, PassDesc{ "Extra", PassType::Copy, PassFlags::None, 0 },
                [&](ISetupContext* s, NoData&) {
                    const ResourceId input = b;
                    s->Read(a, Access::CopySource, nvrhi::ShaderType::None, {});
                    b = s->Write(b, Access::CopyDest, nvrhi::ShaderType::None, {}, AccessFlags::None);
                    s->SetDisabledOutput(b, DisabledOutput::Passthrough, input);
                },
                [a, b](IPassContext* c, const NoData&) { c->GetCommandList()->copyBuffer(c->GetBuffer(b), 0, c->GetBuffer(a), 0, 64); });
            AddLambdaPass<NoData>(graph, PassDesc{ "Readback", PassType::Copy, PassFlags::None, 0 },
                [&](ISetupContext* s, NoData&) {
                    s->Read(b, Access::CopySource, nvrhi::ShaderType::None, {});
                    out = s->Write(out, Access::CopyDest, nvrhi::ShaderType::None, {}, AccessFlags::FullWrite);
                },
                [b, out](IPassContext* c, const NoData&) { c->GetCommandList()->copyBuffer(c->GetBuffer(out), 0, c->GetBuffer(b), 0, 64); });
        }

        nvrhi::AutoPtr<nvrhi::IBuffer> readback;
        uint32_t pattern[16] = {};
        uint32_t frames = 0;
    };

    bool PassEnabled(ICompiledRenderGraph* plan, const char* name)
    {
        for (uint32_t index = 0; index < plan->GetScheduledPassCount(); ++index)
        {
            ScheduledPass pass;
            if (plan->GetScheduledPass(index, pass) == nvrhi::FS_OK && pass.name && std::string(pass.name) == name)
                return pass.enabled;
        }
        return false;
    }

    void CheckReadback(TestLayer& layer, const std::string& what)
    {
        nvrhi::IDevice* device = layer.GetDevice();
        device->waitForIdle();
        const uint32_t* data = static_cast<const uint32_t*>(device->mapBuffer(layer.readback, nvrhi::CpuAccessMode::Read));
        bool good = data != nullptr;
        for (uint32_t i = 0; i < 16 && good; ++i)
            good = data[i] == layer.pattern[i];
        if (data)
            device->unmapBuffer(layer.readback);
        check(good, what);
    }
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    log::SetCallback([](log::Severity severity, const char* text) {
        if (severity >= log::Severity::Error)
            ++g_Errors;
        if (severity >= log::Severity::Warning)
            std::printf("LOG: %s\n", text);
    });
    try
    {
        check(argc == 2, "specify d3d11, d3d12 or vulkan");
        const std::string backend = argv[1];
        const nvrhi::GraphicsAPI api = backend == "d3d11" ? nvrhi::GraphicsAPI::D3D11
            : backend == "d3d12" ? nvrhi::GraphicsAPI::D3D12 : nvrhi::GraphicsAPI::VULKAN;
        nvrhi::AutoPtr<app::DeviceManager> deviceManager = nvrhi::TakeOver(app::DeviceManager::Create(api));
        check(deviceManager != nullptr, "device manager");
        app::DeviceCreationParameters params;
        params.headlessDevice = true;
        params.enableDebugRuntime = true;
        params.enableGPUValidation = api == nvrhi::GraphicsAPI::D3D12;
        params.enableWarningsAsErrors = false;
        params.enableNvrhiValidationLayer = true;
        params.backBufferWidth = 64;
        params.backBufferHeight = 64;
        check(deviceManager->CreateHeadlessDevice(params), "headless device");

        {
            nvrhi::AutoPtr<TestLayer> layer = MAKE_RC_OBJ_PTR(TestLayer, deviceManager);
            for (uint32_t frame = 0; frame < 3; ++frame)
            {
                layer->Render(nullptr);
                check(layer->GetLastPlan() != nullptr, "frame " + std::to_string(frame) + " compiled and executed");
                CheckReadback(*layer, "frame " + std::to_string(frame) + " read back");
            }
            check(PassEnabled(layer->GetLastPlan(), "Extra"), "the toggleable pass runs");

            // 13.5: a toggle by name takes effect from the next frame and passes the buffer through.
            layer->SetPassEnabled("Extra", false);
            layer->Render(nullptr);
            layer->Render(nullptr);
            check(!PassEnabled(layer->GetLastPlan(), "Extra"), "the toggle turned the pass off");
            CheckReadback(*layer, "the disabled pass passes the buffer through");
            layer->SetPassEnabled("Extra", true);
            layer->Render(nullptr);
            check(PassEnabled(layer->GetLastPlan(), "Extra"), "the toggle turned it back on");

            // A resize releases the pooled memory; the next frame recreates it.
            layer->BackBufferResizing();
            layer->Render(nullptr);
            CheckReadback(*layer, "the frame after a resize");
            RunStatistics run;
            layer->GetRegistry()->GetRunStatistics(run);
            check(run.passes == 4, "run statistics: " + std::to_string(run.passes) + " passes");
        }

        deviceManager->Shutdown();
        deviceManager = nullptr;
        check(g_Errors == 0, "no errors were logged");
        std::printf("PASS\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
