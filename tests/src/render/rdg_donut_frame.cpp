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

#include "rdg_donut_frame.h"

using namespace donut;
using nvrhi::Format;

namespace rdgtest
{
    namespace
    {
        nvrhi::AutoPtr<engine::PlanarView> MakeView(float width, float height, int slice)
        {
            nvrhi::AutoPtr<engine::PlanarView> view = MAKE_RC_OBJ_PTR(engine::PlanarView);
            view->SetViewport(nvrhi::Viewport(width, height));
            view->SetMatrices(dm::affine3::identity(), dm::perspProjD3DStyleReverse(dm::radians(60.f), width / height, 0.1f));
            view->SetArraySlice(slice);
            view->UpdateCache();
            return view;
        }

        nvrhi::TextureDesc Texture(uint32_t width, uint32_t height, Format format, bool renderTarget, bool uav)
        {
            nvrhi::TextureDesc desc;
            desc.width = width;
            desc.height = height;
            desc.format = format;
            desc.isRenderTarget = renderTarget;
            desc.isUAV = uav;
            return desc;
        }
    }

    DonutScene::DonutScene()
    {
        view = MakeView(float(width), float(height), 0);
        cascades = MAKE_RC_OBJ_PTR(engine::CompositeView);
        for (int slice = 0; slice < 4; ++slice)
        {
            cascadeViews.push_back(MakeView(32.f, 32.f, slice));
            cascades->AddView(cascadeViews.back());
        }
        light = MAKE_RC_OBJ_PTR(engine::DirectionalLight);
        drawStrategy = MAKE_RC_OBJ_PTR(PassthroughDrawStrategy);
    }

    void DonutPasses::Create(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory, engine::CommonRenderPasses* commonPasses)
    {
        depth = MAKE_RC_OBJ_PTR(DepthPass, device, commonPasses);
        gbuffer = MAKE_RC_OBJ_PTR(GBufferFillPass, device, commonPasses);
        forward = MAKE_RC_OBJ_PTR(ForwardShadingPass, device, commonPasses);
        lighting = MAKE_RC_OBJ_PTR(DeferredLightingPass, device, commonPasses);
        if (device)
        {
            depth->Init(*shaderFactory, DepthPass::CreateParameters());
            GBufferFillPass::CreateParameters gbufferParams;
            gbufferParams.enableMotionVectors = false;
            gbuffer->Init(*shaderFactory, gbufferParams);
            forward->Init(*shaderFactory, ForwardShadingPass::CreateParameters());
            lighting->Init(shaderFactory);
        }
        ssao = MAKE_RC_OBJ_PTR(SsaoPass, device, shaderFactory, commonPasses);
        sky = MAKE_RC_OBJ_PTR(SkyPass, device, shaderFactory, commonPasses);
        environment = MAKE_RC_OBJ_PTR(EnvironmentMapPass, device, shaderFactory, commonPasses);
        bloom = MAKE_RC_OBJ_PTR(BloomPass, device, shaderFactory, commonPasses);
        taa = MAKE_RC_OBJ_PTR(TemporalAntiAliasingPass, device, shaderFactory, commonPasses, TemporalAntiAliasingPass::CreateParameters());
        toneMapping = MAKE_RC_OBJ_PTR(ToneMappingPass, device, shaderFactory, commonPasses, ToneMappingPass::CreateParameters());
        mipMap = MAKE_RC_OBJ_PTR(MipMapGenPass, device, shaderFactory, MipMapGenPass::MODE_MAX);
        blit = MAKE_RC_OBJ_PTR(BlitPass, device, commonPasses);
    }

    nvrhi::TextureDesc ShadowMapDesc()
    {
        nvrhi::TextureDesc desc = Texture(32, 32, Format::D32, true, false);
        desc.arraySize = 4;
        desc.dimension = nvrhi::TextureDimension::Texture2DArray;
        desc.isTypeless = true;
        desc.debugName = "ShadowMap";
        return desc;
    }

    nvrhi::TextureDesc EnvironmentMapDesc()
    {
        nvrhi::TextureDesc desc = Texture(16, 16, Format::RGBA16_FLOAT, false, false);
        desc.arraySize = 6;
        desc.dimension = nvrhi::TextureDimension::TextureCube;
        desc.debugName = "EnvironmentMap";
        return desc;
    }

    nvrhi::TextureDesc MipChainDesc()
    {
        nvrhi::TextureDesc desc = Texture(64, 64, Format::R32_FLOAT, false, true);
        desc.mipLevels = 7;
        desc.debugName = "MipChain";
        return desc;
    }

    nvrhi::TextureDesc BackBufferDesc(const DonutScene& scene)
    {
        nvrhi::TextureDesc desc = Texture(scene.width, scene.height, Format::RGBA8_UNORM, true, false);
        desc.debugName = "BackBuffer";
        return desc;
    }

    void RecordDonutFrame(IRenderGraph* g, DonutPasses& passes, DonutScene& scene, const DonutImports& imports)
    {
        const uint32_t w = scene.width, h = scene.height;

        // Shadow cascades: DepthPass once per slice.
        GeometryGraphInputs shadow;
        shadow.name = "ShadowDepth";
        shadow.view = scene.cascades;
        shadow.drawStrategy = scene.drawStrategy;
        shadow.passContext = &scene.depthContext;
        shadow.depth = imports.shadowMap;
        shadow.clearTargets = true;
        shadow.depthClear.depth = 0.f;
        const ResourceId shadowMap = passes.depth->AddToGraph(g, shadow).depth;

        // G-buffer.
        nvrhi::TextureDesc depthDesc = Texture(w, h, Format::D24S8, true, false);
        depthDesc.isTypeless = true;
        GeometryGraphInputs gbuffer;
        gbuffer.name = "GBufferFill";
        gbuffer.view = scene.view;
        gbuffer.viewPrev = scene.view;
        gbuffer.drawStrategy = scene.drawStrategy;
        gbuffer.passContext = &scene.gbufferContext;
        gbuffer.colorTargets[0] = g->CreateTexture(Texture(w, h, Format::SRGBA8_UNORM, true, false), ResourceFlags::None);
        gbuffer.colorTargets[1] = g->CreateTexture(Texture(w, h, Format::SRGBA8_UNORM, true, false), ResourceFlags::None);
        gbuffer.colorTargets[2] = g->CreateTexture(Texture(w, h, Format::RGBA16_SNORM, true, false), ResourceFlags::None);
        gbuffer.colorTargets[3] = g->CreateTexture(Texture(w, h, Format::RGBA16_FLOAT, true, false), ResourceFlags::None);
        gbuffer.colorTargetCount = 4;
        gbuffer.depth = g->CreateTexture(depthDesc, ResourceFlags::None);
        gbuffer.clearTargets = true;
        const GeometryGraphOutputs targets = passes.gbuffer->AddToGraph(g, gbuffer);

        SsaoGraphInputs ssao;
        ssao.view = scene.view;
        ssao.depth = targets.depth;
        ssao.normals = targets.colorTargets[2];
        ssao.output = g->CreateTexture(Texture(w, h, Format::R8_UNORM, false, true), ResourceFlags::None);
        const ResourceId ambientOcclusion = passes.ssao->AddToGraph(g, ssao).output;

        DeferredLightingGraphInputs lighting;
        lighting.view = scene.view;
        lighting.depth = targets.depth;
        lighting.gbufferDiffuse = targets.colorTargets[0];
        lighting.gbufferSpecular = targets.colorTargets[1];
        lighting.gbufferNormals = targets.colorTargets[2];
        lighting.gbufferEmissive = targets.colorTargets[3];
        lighting.ambientOcclusion = ambientOcclusion;
        lighting.shadowMap = shadowMap;
        lighting.output = g->CreateTexture(Texture(w, h, Format::RGBA16_FLOAT, true, true), ResourceFlags::None);
        ResourceId hdr = passes.lighting->AddToGraph(g, lighting).output;

        SkyGraphInputs sky;
        sky.view = scene.view;
        sky.color = hdr;
        sky.depth = targets.depth;
        sky.light = scene.light;
        hdr = passes.sky->AddToGraph(g, sky).color;

        EnvironmentMapGraphInputs environment;
        environment.view = scene.view;
        environment.color = hdr;
        environment.depth = targets.depth;
        environment.environmentMap = imports.environmentMap;
        hdr = passes.environment->AddToGraph(g, environment).color;

        GeometryGraphInputs forward;
        forward.name = "ForwardShading";
        forward.view = scene.view;
        forward.drawStrategy = scene.drawStrategy;
        forward.passContext = &scene.forwardContext;
        forward.colorTargets[0] = hdr;
        forward.colorTargetCount = 1;
        forward.depth = targets.depth;
        forward.depthWrite = false;
        forward.shaderReads[0] = shadowMap;
        forward.shaderReadCount = 1;
        hdr = passes.forward->AddToGraph(g, forward).colorTargets[0];

        BloomGraphInputs bloom;
        bloom.view = scene.view;
        bloom.color = hdr;
        hdr = passes.bloom->AddToGraph(g, bloom).color;

        TemporalAntiAliasingGraphInputs taa;
        taa.view = scene.view;
        taa.viewPrevious = scene.view;
        taa.depth = targets.depth;
        taa.motionVectors = g->CreateTexture(Texture(w, h, Format::RG16_FLOAT, true, false), ResourceFlags::None);
        taa.unresolvedColor = hdr;
        taa.resolvedColor = g->CreateTexture(Texture(w, h, Format::RGBA16_FLOAT, false, true), ResourceFlags::None);
        const ResourceId resolved = passes.taa->AddToGraph(g, taa).resolvedColor;

        ToneMappingGraphInputs toneMapping;
        toneMapping.view = scene.view;
        toneMapping.source = resolved;
        toneMapping.output = g->CreateTexture(Texture(w, h, Format::RGBA8_UNORM, true, false), ResourceFlags::None);
        const ResourceId ldr = passes.toneMapping->AddToGraph(g, toneMapping).output;

        BlitGraphInputs blit;
        blit.source = ldr;
        blit.target = imports.backBuffer;
        passes.blit->AddToGraph(g, blit);

        passes.mipMap->AddToGraph(g, MipMapGenGraphInputs{ imports.mipChain, -1 });
    }
}
