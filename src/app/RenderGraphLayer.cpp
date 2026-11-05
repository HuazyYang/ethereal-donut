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

#include <donut/app/RenderGraphLayer.h>
#include <donut/core/log.h>

#include <algorithm>
#include <cstring>

using namespace donut::app;
using namespace donut::render;

RenderGraphLayer::RenderGraphLayer(DeviceManager* deviceManager)
    : IAppLayer(deviceManager)
{
    IResourceRegistry* registry = nullptr;
    if (CreateResourceRegistry(GetDevice(), &registry) == nvrhi::FS_OK)
        m_Registry = nvrhi::TakeOver(registry);
    IRenderGraph* graph = nullptr;
    if (m_Registry && CreateRenderGraph(m_Registry, &graph) == nvrhi::FS_OK)
        m_Graph = nvrhi::TakeOver(graph);
    if (ComputeBackendProfile(GetDevice(), &m_Profile) != nvrhi::FS_OK)
        log::error("RenderGraphLayer: no backend profile for this device");
}

RenderGraphLayer::~RenderGraphLayer()
{
    // The registry waits for the frames in flight when it goes; the graph and the plan go first.
    m_LastPlan = nullptr;
    m_Graph = nullptr;
    m_Registry = nullptr;
}

void RenderGraphLayer::Render(nvrhi::IFramebuffer*)
{
    if (!m_Graph)
        return;

    m_Graph->Reset();
    m_FrameData.Clear();

    // 13.4: the swap chain keeps its back buffers' initial state, so the graph imports them host-kept.
    nvrhi::ITexture* backBufferTexture = GetDeviceManager()->GetCurrentBackBuffer();
    ResourceId backBuffer = InvalidResourceId;
    if (backBufferTexture)
    {
        const nvrhi::TextureDesc& desc = backBufferTexture->getDesc();
        backBuffer = desc.keepInitialState
            ? m_Graph->ImportHostKeptTexture(backBufferTexture)
            : m_Graph->ImportTexture(backBufferTexture, nvrhi::ResourceStates::Present, nvrhi::ResourceStates::Present);
    }

    BuildGraph(m_Graph, m_FrameData, backBuffer);

    // The toggles: through the last plan, which names every pass use of the same structure by its PassId.
    const uint64_t structuralHash = m_Graph->GetStructuralHash();
    if (!m_DisabledPasses.empty() && m_LastPlan && m_LastStructuralHash == structuralHash)
    {
        for (uint32_t index = 0; index < m_LastPlan->GetScheduledPassCount(); ++index)
        {
            ScheduledPass scheduled;
            if (m_LastPlan->GetScheduledPass(index, scheduled) != nvrhi::FS_OK || !scheduled.name)
                continue;
            if (std::find(m_DisabledPasses.begin(), m_DisabledPasses.end(), scheduled.name) != m_DisabledPasses.end())
                m_Graph->SetPassEnabled(scheduled.pass, false);
        }
    }

    ICompiledRenderGraph* plan = nullptr;
    if (m_Graph->Compile(m_Profile, m_CompileOptions, &plan) != nvrhi::FS_OK)
    {
        nvrhi::string diagnostics;
        m_Graph->GetDiagnostics(diagnostics);
        log::error("The render graph does not compile:\n%s", diagnostics.c_str());
        return;
    }
    m_LastPlan = nvrhi::TakeOver(plan);
    m_LastStructuralHash = structuralHash;
    if (m_Graph->Execute(m_LastPlan, m_ExecuteDesc) != nvrhi::FS_OK)
    {
        nvrhi::string diagnostics;
        m_Graph->GetDiagnostics(diagnostics);
        log::error("The render graph failed to execute:\n%s", diagnostics.c_str());
    }
}

void RenderGraphLayer::BackBufferResizing()
{
    // The old sizes' memory and the passes' framebuffers and binding sets go; the new descriptions recompile.
    m_LastPlan = nullptr;
    if (m_Registry)
        m_Registry->ReleaseResources();
}

void RenderGraphLayer::SetPassEnabled(const char* name, bool enabled)
{
    if (!name)
        return;
    auto found = std::find(m_DisabledPasses.begin(), m_DisabledPasses.end(), name);
    if (enabled && found != m_DisabledPasses.end())
        m_DisabledPasses.erase(found);
    else if (!enabled && found == m_DisabledPasses.end())
        m_DisabledPasses.push_back(name);
}

bool RenderGraphLayer::IsPassEnabled(const char* name) const
{
    return !name || std::find(m_DisabledPasses.begin(), m_DisabledPasses.end(), name) == m_DisabledPasses.end();
}
