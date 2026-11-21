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

// One frame of every graph-capable donut pass (aggregate plan, step P1.5), for the device-free golden access lists and
// for the headless test that executes it on each backend. The scene is empty: the geometry passes draw nothing, and
// every other pass runs in full.

#pragma once

#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/View.h>
#include <donut/render/BlitPass.h>
#include <donut/render/BloomPass.h>
#include <donut/render/DeferredLightingPass.h>
#include <donut/render/DepthPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/EnvironmentMapPass.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/GBufferFillPass.h>
#include <donut/render/MipMapGenPass.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/SkyPass.h>
#include <donut/render/SsaoPass.h>
#include <donut/render/TemporalAntiAliasingPass.h>
#include <donut/render/ToneMappingPasses.h>

#include <vector>

namespace rdgtest
{
    using namespace donut::render;

    struct DonutScene
    {
        uint32_t width = 64;
        uint32_t height = 64;
        nvrhi::AutoPtr<donut::engine::PlanarView> view;
        nvrhi::AutoPtr<donut::engine::CompositeView> cascades;   // four shadow cascades, one slice each
        std::vector<nvrhi::AutoPtr<donut::engine::PlanarView>> cascadeViews;
        nvrhi::AutoPtr<donut::engine::DirectionalLight> light;
        nvrhi::AutoPtr<PassthroughDrawStrategy> drawStrategy;
        DepthPass::Context depthContext;
        GBufferFillPass::Context gbufferContext;
        ForwardShadingPass::Context forwardContext;

        DonutScene();
    };

    struct DonutPasses
    {
        nvrhi::AutoPtr<DepthPass> depth;
        nvrhi::AutoPtr<GBufferFillPass> gbuffer;
        nvrhi::AutoPtr<ForwardShadingPass> forward;
        nvrhi::AutoPtr<SsaoPass> ssao;
        nvrhi::AutoPtr<DeferredLightingPass> lighting;
        nvrhi::AutoPtr<SkyPass> sky;
        nvrhi::AutoPtr<EnvironmentMapPass> environment;
        nvrhi::AutoPtr<BloomPass> bloom;
        nvrhi::AutoPtr<TemporalAntiAliasingPass> taa;
        nvrhi::AutoPtr<ToneMappingPass> toneMapping;
        nvrhi::AutoPtr<MipMapGenPass> mipMap;
        nvrhi::AutoPtr<BlitPass> blit;

        // Graph constructions; with a device, the passes that need Init get it.
        void Create(nvrhi::IDevice* device, donut::engine::ShaderFactory* shaderFactory,
                    donut::engine::CommonRenderPasses* commonPasses);
    };

    // The frame's imports.
    struct DonutImports
    {
        ResourceId shadowMap;        // D32, 32x32, 4 slices, render target; ShaderResource between frames
        ResourceId environmentMap;   // RGBA16_FLOAT cube, 16x16; ShaderResource
        ResourceId mipChain;         // R32_FLOAT 64x64, 7 mips, UAV; ShaderResource
        ResourceId backBuffer;       // RGBA8_UNORM width x height, render target; RenderTarget
    };

    // The descriptions of those imports.
    nvrhi::TextureDesc ShadowMapDesc();
    nvrhi::TextureDesc EnvironmentMapDesc();
    nvrhi::TextureDesc MipChainDesc();
    nvrhi::TextureDesc BackBufferDesc(const DonutScene& scene);

    // Shadow cascades, G-buffer, SSAO, deferred lighting, sky, environment map, forward shading, bloom, TAA, tone
    // mapping, a blit to the back buffer, and a mip chain.
    void RecordDonutFrame(IRenderGraph* graph, DonutPasses& passes, DonutScene& scene, const DonutImports& imports);
}
