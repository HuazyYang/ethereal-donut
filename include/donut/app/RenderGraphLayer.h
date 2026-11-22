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

#pragma once

#include <donut/app/DeviceManager.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>

#include <string>
#include <vector>

namespace donut::app
{
    // An app layer that renders through a render graph (design 13.4). It owns the registry, its graph and a
    // FrameData blackboard. Each frame it resets the graph and clears the blackboard, imports the current back buffer,
    // calls BuildGraph, applies the pass toggles, then compiles and executes. A resize releases the pooled memory and
    // the passes' caches; the next frame's new descriptions recompile, and no pass is recreated.
    class RenderGraphLayer : public IAppLayer
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        explicit RenderGraphLayer(DeviceManager* deviceManager);
        ~RenderGraphLayer() override;

        // Records the frame. `backBuffer` is the imported back buffer: host-kept where the swap chain keeps its
        // initial state, so it ends the frame in that state.
        virtual void BuildGraph(render::IRenderGraph* graph, render::FrameData& frameData, render::ResourceId backBuffer) = 0;
        // Called once per Render after BuildGraph, when the frame is over: after Execute, whether it succeeded or not, or
        // after a Compile that failed (`executed` false then, and after a failed Execute). The place for per-frame
        // cleanup that must follow execution, such as render::RenderService::EndFrame. The default does nothing.
        virtual void OnGraphExecuted(render::IRenderGraph* graph, bool executed) { (void)graph; (void)executed; }

        void Render(nvrhi::IFramebuffer* framebuffer) override;
        void BackBufferResizing() override;

        [[nodiscard]] render::IRenderGraph* GetGraph() const { return m_Graph; }
        [[nodiscard]] render::IResourceRegistry* GetRegistry() const { return m_Registry; }
        // The plan the last frame executed, for diagnostics (RenderGraphVisualizer); nullptr before the first frame.
        [[nodiscard]] render::ICompiledRenderGraph* GetLastPlan() const { return m_LastPlan; }
        [[nodiscard]] const render::BackendProfile& GetProfile() const { return m_Profile; }
        // The device's profile, which an application may narrow (split barriers off, its LatticePolicy) between
        // frames; the plan cache keys on it.
        render::BackendProfile& GetProfile() { return m_Profile; }
        render::CompileOptions& GetCompileOptions() { return m_CompileOptions; }
        render::ExecuteDesc& GetExecuteDesc() { return m_ExecuteDesc; }

        // Pass toggles by pass name (13.5), applied with SetPassEnabled to every use of that name. A name is found
        // through the last plan, so a toggle takes effect from the frame after it is set.
        void SetPassEnabled(const char* name, bool enabled);
        [[nodiscard]] bool IsPassEnabled(const char* name) const;

    private:
        nvrhi::AutoPtr<render::IResourceRegistry> m_Registry;
        nvrhi::AutoPtr<render::IRenderGraph> m_Graph;
        nvrhi::AutoPtr<render::ICompiledRenderGraph> m_LastPlan;
        render::FrameData m_FrameData;
        render::BackendProfile m_Profile;
        render::CompileOptions m_CompileOptions;
        render::ExecuteDesc m_ExecuteDesc;
        std::vector<std::string> m_DisabledPasses;
        uint64_t m_LastStructuralHash = 0;   // of the graph m_LastPlan was compiled from
    };
}
