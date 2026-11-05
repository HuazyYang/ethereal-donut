/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <donut/render/BloomPass.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>
#include <utility>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/bloom_ps.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/bloom_ps.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/bloom_ps.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/bloom_cb.h>

using namespace donut::engine;
using namespace donut::render;

void BloomPass::CreateCommonObjects(ShaderFactory* shaderFactory)
{
    m_BloomBlurPixelShader = shaderFactory->CreateAutoShader("donut/passes/bloom_ps.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_bloom_ps), nullptr, nvrhi::ShaderType::Pixel);

    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(BloomConstants);
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.debugName = "BloomConstantsH";
    constantBufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
    m_Device->createBuffer(constantBufferDesc, &m_BloomHBlurCB);
    constantBufferDesc.debugName = "BloomConstantsV";
    m_Device->createBuffer(constantBufferDesc, &m_BloomVBlurCB);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::Pixel;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Texture_SRV(0)
    };
    m_Device->createBindingLayout(layoutDesc, &m_BloomBlurBindingLayout);

}

BloomPass::BloomPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    CommonRenderPasses* commonPasses,
    FramebufferFactory* framebufferFactory,
    const ICompositeView& compositeView)
    : m_CommonPasses(commonPasses)
    , m_FramebufferFactory(framebufferFactory)
    , m_Device(device)
    , m_BindingCache(device)
{
    CreateCommonObjects(shaderFactory);

    m_PerViewData.resize(compositeView.GetNumChildViews(ViewType::PLANAR));
    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);
        
        nvrhi::IFramebuffer* sampleFramebuffer = m_FramebufferFactory->GetFramebuffer(*view);
        PerViewData& perViewData = m_PerViewData[viewIndex];

        nvrhi::Rect viewExtent = view->GetViewExtent();
        int viewportWidth = viewExtent.maxX - viewExtent.minX;
        int viewportHeight = viewExtent.maxY - viewExtent.minY;

        // temporary textures for downscaling
        nvrhi::TextureDesc downscaleTextureDesc;
        downscaleTextureDesc.format = sampleFramebuffer->getFramebufferInfo().colorFormats[0];
        downscaleTextureDesc.width = (uint32_t)ceil(viewportWidth / 2.f);
        downscaleTextureDesc.height = (uint32_t)ceil(viewportHeight / 2.f);
        downscaleTextureDesc.mipLevels = 1;
        downscaleTextureDesc.isRenderTarget = true;
        downscaleTextureDesc.debugName = "bloom src mip1";
        downscaleTextureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        downscaleTextureDesc.keepInitialState = true;
        m_Device->createTexture(downscaleTextureDesc, &perViewData.textureDownscale1);
        m_Device->createFramebuffer(nvrhi::FramebufferDesc()
            .addColorAttachment(perViewData.textureDownscale1), &perViewData.framebufferDownscale1);

        downscaleTextureDesc.debugName = "bloom src mip2";
        downscaleTextureDesc.width = (uint32_t)ceil(downscaleTextureDesc.width / 2.f);
        downscaleTextureDesc.height = (uint32_t)ceil(downscaleTextureDesc.height / 2.f);
        m_Device->createTexture(downscaleTextureDesc, &perViewData.textureDownscale2);
        m_Device->createFramebuffer(nvrhi::FramebufferDesc()
            .addColorAttachment(perViewData.textureDownscale2), &perViewData.framebufferDownscale2);

        // intermediate textures for accumulating blur
        nvrhi::TextureDesc intermediateTextureDesc2;
        intermediateTextureDesc2.format = downscaleTextureDesc.format;
        intermediateTextureDesc2.width = downscaleTextureDesc.width;
        intermediateTextureDesc2.height = downscaleTextureDesc.height;
        intermediateTextureDesc2.mipLevels = 1;
        intermediateTextureDesc2.isRenderTarget = true;

        intermediateTextureDesc2.debugName = "bloom accumulation pass1";
        intermediateTextureDesc2.initialState = nvrhi::ResourceStates::ShaderResource;
        intermediateTextureDesc2.keepInitialState = true;
        m_Device->createTexture(intermediateTextureDesc2, &perViewData.texturePass1Blur);
        m_Device->createFramebuffer(nvrhi::FramebufferDesc()
            .addColorAttachment(perViewData.texturePass1Blur), &perViewData.framebufferPass1Blur);

        intermediateTextureDesc2.debugName = "bloom accumulation pass2";
        m_Device->createTexture(intermediateTextureDesc2, &perViewData.texturePass2Blur);
        m_Device->createFramebuffer(nvrhi::FramebufferDesc()
            .addColorAttachment(perViewData.texturePass2Blur), &perViewData.framebufferPass2Blur);

        nvrhi::GraphicsPipelineDesc graphicsPipelineDesc;
        graphicsPipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
        graphicsPipelineDesc.VS = m_CommonPasses->m_FullscreenVS;
        graphicsPipelineDesc.PS = m_BloomBlurPixelShader;
        graphicsPipelineDesc.bindingLayouts = { m_BloomBlurBindingLayout };
        graphicsPipelineDesc.renderState.rasterState.setCullNone();
        graphicsPipelineDesc.renderState.depthStencilState.depthTestEnable = false;
        graphicsPipelineDesc.renderState.depthStencilState.stencilEnable = false;
        device->createGraphicsPipeline1(graphicsPipelineDesc,
            perViewData.framebufferPass1Blur->getFramebufferInfo().getInfo(), &perViewData.bloomBlurPso);
        
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_BloomHBlurCB),
            nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearClampSampler),
            nvrhi::BindingSetItem::Texture_SRV(0, perViewData.textureDownscale2)
        };
        m_Device->createBindingSet(bindingSetDesc, m_BloomBlurBindingLayout, &perViewData.bloomBlurBindingSetPass1);

        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_BloomVBlurCB),
            nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearClampSampler),
            nvrhi::BindingSetItem::Texture_SRV(0, perViewData.texturePass1Blur)
        };
        m_Device->createBindingSet(bindingSetDesc, m_BloomBlurBindingLayout, &perViewData.bloomBlurBindingSetPass2);
    }
}

void BloomPass::Render(
	nvrhi::ICommandList* commandList,
    FramebufferFactory* framebufferFactory,
    const ICompositeView* compositeView,
	nvrhi::ITexture* sourceDestTexture,
    float sigmaInPixels,
    float blendFactor)
{
    float effectiveSigma = clamp(sigmaInPixels * 0.25f, 1.f, 100.f);

    commandList->beginMarker("Bloom");

    nvrhi::DrawArguments fullscreenquadargs;
    fullscreenquadargs.instanceCount = 1;
    fullscreenquadargs.vertexCount = 4;

    for (uint viewIndex = 0; viewIndex < compositeView->GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView->GetChildView(ViewType::PLANAR, viewIndex);
        nvrhi::IFramebuffer* framebuffer = framebufferFactory->GetFramebuffer(*view);
        PerViewData& perViewData = m_PerViewData[viewIndex];

        nvrhi::ViewportState viewportState = view->GetViewportState();
        const nvrhi::Rect& scissorRect = viewportState.scissorRects[0];
        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

        // downscale
        {
            commandList->beginMarker("Downscale");

            dm::box2 uvSrcRect = box2(
                float2(
                    float(scissorRect.minX) / (float)fbinfo.width,
                    float(scissorRect.minY) / (float)fbinfo.height),
                float2(
                    float(scissorRect.maxX) / (float)fbinfo.width,
                    float(scissorRect.maxY) / (float)fbinfo.height)
            );

            // half-scale down

            BlitParameters blitParams1;
            blitParams1.targetFramebuffer = perViewData.framebufferDownscale1;
            blitParams1.sourceTexture = sourceDestTexture;
            blitParams1.sourceBox = uvSrcRect;
            m_CommonPasses->BlitTexture(commandList, blitParams1, &m_BindingCache);

            // half-scale again down to quarter-scale

            BlitParameters blitParams2;
            blitParams2.targetFramebuffer = perViewData.framebufferDownscale2;
            blitParams2.sourceTexture = perViewData.textureDownscale1;
            m_CommonPasses->BlitTexture(commandList, blitParams2, &m_BindingCache);

            commandList->endMarker(); // "Downscale"
        }

        // apply blur
        {
            commandList->beginMarker("Blur");
            nvrhi::Viewport viewport;

            nvrhi::GraphicsState state;
            state.pipeline = perViewData.bloomBlurPso;
            viewport = nvrhi::Viewport(float(perViewData.texturePass1Blur->getDesc().width), float(perViewData.texturePass1Blur->getDesc().height));
            state.viewport.addViewport(viewport);
            state.viewport.addScissorRect(nvrhi::Rect(viewport));
            state.framebuffer = perViewData.framebufferPass1Blur;
            state.bindings = { perViewData.bloomBlurBindingSetPass1 };

            BloomConstants bloomHorizonal = {};
            bloomHorizonal.pixstep.x = 1.f / perViewData.texturePass1Blur->getDesc().width;
            bloomHorizonal.pixstep.y = 0.f;
            bloomHorizonal.argumentScale = -1.f / (2 * effectiveSigma * effectiveSigma);
            bloomHorizonal.normalizationScale = 1.f / (sqrtf(2 * PI_f) * effectiveSigma);
            bloomHorizonal.numSamples = ::round(effectiveSigma * 4.f);
            BloomConstants bloomVertical = bloomHorizonal;
            bloomVertical.pixstep.x = 0.f;
            bloomVertical.pixstep.y = 1.f / perViewData.texturePass1Blur->getDesc().height;
            commandList->writeBuffer(m_BloomHBlurCB, &bloomHorizonal, sizeof(bloomHorizonal));
            commandList->writeBuffer(m_BloomVBlurCB, &bloomVertical, sizeof(bloomVertical));

            commandList->setGraphicsState(state);
            commandList->draw(fullscreenquadargs); // blur to m_TexturePass1Blur or m_TexturePass3Blur

            viewport = nvrhi::Viewport(float(perViewData.texturePass2Blur->getDesc().width), float(perViewData.texturePass2Blur->getDesc().height));
            state.viewport.viewports[0] = viewport;
            state.viewport.scissorRects[0] = nvrhi::Rect(viewport);
            state.framebuffer = perViewData.framebufferPass2Blur;
            state.bindings = { perViewData.bloomBlurBindingSetPass2 };

            commandList->setGraphicsState(state);
            commandList->draw(fullscreenquadargs); // blur to m_TexturePass2Blur

            commandList->endMarker(); // "Blur"
        }

        // composite
        {
            commandList->beginMarker("Apply");

            BlitParameters blitParams3;
            blitParams3.targetFramebuffer = framebuffer;
            blitParams3.targetViewport = viewportState.viewports[0];
            blitParams3.sourceTexture = perViewData.texturePass2Blur;
            blitParams3.blendState.setBlendEnable(true)
                .setSrcBlend(nvrhi::BlendFactor::ConstantColor)
                .setDestBlend(nvrhi::BlendFactor::InvConstantColor)
                .setSrcBlendAlpha(nvrhi::BlendFactor::Zero)
                .setDestBlendAlpha(nvrhi::BlendFactor::One);
            blitParams3.blendConstantColor = nvrhi::Color(blendFactor);
            m_CommonPasses->BlitTexture(commandList, blitParams3, &m_BindingCache);

            commandList->endMarker(); // "Apply"
        }
    }

    commandList->endMarker(); // "Bloom"
}

// ------------------------------------------------------------------------------------------ render graph (13.3)

BloomPass::BloomPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    CommonRenderPasses* commonPasses)
    : m_CommonPasses(commonPasses)
    , m_Device(device)
    , m_BindingCache(device)
    , m_ShaderFactory(shaderFactory)
{
}

namespace
{
    enum class BloomStage : uint32_t { Downscale1, Downscale2, BlurHorizontal, BlurVertical, Apply };

    struct BloomUse
    {
        BloomStage stage;
        const IView* view;
        ResourceId source;
        ResourceId target;
        dm::box2 sourceBox;
        float sigmaInPixels;
        float blendFactor;
        nvrhi::SubresourceRegion region;
    };

    const char* const c_BloomStageNames[] = { "Bloom/Downscale1", "Bloom/Downscale2", "Bloom/BlurH", "Bloom/BlurV", "Bloom/Apply" };
}

BloomGraphOutputs BloomPass::AddToGraph(IRenderGraph* graph, const BloomGraphInputs& inputs)
{
    BloomGraphOutputs outputs{ inputs.color };
    nvrhi::TextureDesc colorDesc;
    if (!inputs.view || graph->GetTextureDesc(inputs.color, colorDesc) != nvrhi::FS_OK)
        return outputs;

    const uint32_t viewCount = inputs.view->GetNumChildViews(ViewType::PLANAR);
    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        const IView* view = inputs.view->GetChildView(ViewType::PLANAR, viewIndex);
        const nvrhi::Rect viewExtent = view->GetViewExtent();
        const nvrhi::Rect scissor = view->GetViewportState().scissorRects[0];

        // The chain's transients, without debug names so that recording allocates nothing.
        nvrhi::TextureDesc desc;
        desc.format = colorDesc.format;
        desc.isRenderTarget = true;
        desc.width = uint32_t(ceil((viewExtent.maxX - viewExtent.minX) / 2.f));
        desc.height = uint32_t(ceil((viewExtent.maxY - viewExtent.minY) / 2.f));
        const ResourceId downscale1 = graph->CreateTexture(desc, ResourceFlags::None);
        desc.width = uint32_t(ceil(desc.width / 2.f));
        desc.height = uint32_t(ceil(desc.height / 2.f));
        const ResourceId downscale2 = graph->CreateTexture(desc, ResourceFlags::None);
        const ResourceId blur1 = graph->CreateTexture(desc, ResourceFlags::None);
        const ResourceId blur2 = graph->CreateTexture(desc, ResourceFlags::None);

        const nvrhi::TextureSubresourceSet subresources = view->GetSubresources();
        const nvrhi::SubresourceRegion viewRegion = nvrhi::SubresourceRegion().setMips(subresources.baseMipLevel, 1)
            .setSlices(subresources.baseArraySlice, subresources.numArraySlices);
        const dm::box2 sourceBox(
            float2(float(scissor.minX) / float(colorDesc.width), float(scissor.minY) / float(colorDesc.height)),
            float2(float(scissor.maxX) / float(colorDesc.width), float(scissor.maxY) / float(colorDesc.height)));

        const struct { BloomStage stage; ResourceId source; ResourceId target; } chain[] = {
            { BloomStage::Downscale1, outputs.color, downscale1 },
            { BloomStage::Downscale2, downscale1, downscale2 },
            { BloomStage::BlurHorizontal, downscale2, blur1 },
            { BloomStage::BlurVertical, blur1, blur2 },
            { BloomStage::Apply, blur2, outputs.color },
        };
        ResourceId written[5] = {};
        for (uint32_t i = 0; i < 5; ++i)
        {
            BloomUse* use = NewFrameData<BloomUse>(graph);
            if (!use)
                return outputs;
            // Each stage reads what the stage before it wrote.
            const ResourceId source = i == 0 ? chain[i].source : written[i - 1];
            const ResourceId target = i == 4 ? outputs.color : chain[i].target;
            *use = BloomUse{ chain[i].stage, view, source, target, sourceBox, inputs.sigmaInPixels, inputs.blendFactor,
                             i == 4 ? viewRegion : nvrhi::SubresourceRegion() };
            graph->AddPass(PassDesc{ c_BloomStageNames[i], PassType::Raster, PassFlags::None, 0 }, this, use, nullptr);
            written[i] = use->target;
        }
        outputs.color = written[4];
    }
    return outputs;
}

nvrhi::FRESULT BloomPass::Setup(ISetupContext* context, void* data) noexcept
{
    BloomUse& use = *static_cast<BloomUse*>(data);
    context->Read(use.source, Access::ShaderRead, nvrhi::ShaderType::Pixel, nvrhi::SubresourceRegion());
    // The apply blends onto the colour target and keeps what is there; the other stages write all of theirs.
    const AccessFlags flags = use.stage == BloomStage::Apply ? AccessFlags::None : AccessFlags::FullWrite;
    use.target = context->SetColorTarget(0, use.target, use.region, flags, nullptr);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT BloomPass::Prepare(IPrepareContext* context, void* data) noexcept
{
    const BloomUse& use = *static_cast<const BloomUse*>(data);
    try
    {
        if (!m_BloomBlurPixelShader)
        {
            if (!m_Device || !m_ShaderFactory)
                return nvrhi::FE_INVALID_ARGS;
            CreateCommonObjects(m_ShaderFactory);
        }
        if (use.stage == BloomStage::BlurHorizontal || use.stage == BloomStage::BlurVertical)
        {
            const nvrhi::FramebufferInfo framebufferInfo = context->GetFramebufferInfo().getInfo();
            if (!m_GraphBlurPso || m_GraphBlurPso->getFramebufferInfo() != framebufferInfo)
            {
                nvrhi::GraphicsPipelineDesc graphicsPipelineDesc;
                graphicsPipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
                graphicsPipelineDesc.VS = m_CommonPasses->m_FullscreenVS;
                graphicsPipelineDesc.PS = m_BloomBlurPixelShader;
                graphicsPipelineDesc.bindingLayouts = { m_BloomBlurBindingLayout };
                graphicsPipelineDesc.renderState.rasterState.setCullNone();
                graphicsPipelineDesc.renderState.depthStencilState.depthTestEnable = false;
                graphicsPipelineDesc.renderState.depthStencilState.stencilEnable = false;
                m_GraphBlurPso = nullptr;
                m_Device->createGraphicsPipeline1(graphicsPipelineDesc, framebufferInfo, &m_GraphBlurPso);
            }
            if (!m_GraphBlurPso)
                return nvrhi::FE_GENERIC_ERROR;
        }
        return nvrhi::FS_OK;
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
}

nvrhi::FRESULT BloomPass::Execute(IPassContext* context, const void* data) noexcept
{
    const BloomUse& use = *static_cast<const BloomUse*>(data);
    nvrhi::ICommandList* commandList = context->GetCommandList();
    nvrhi::IFramebuffer* framebuffer = context->GetFramebuffer();
    nvrhi::ITexture* source = context->GetTexture(use.source);
    if (!framebuffer || !source)
        return nvrhi::FE_INVALID_ARGS;
    try
    {
        switch (use.stage)
        {
        case BloomStage::Downscale1:
        case BloomStage::Downscale2:
        {
            BlitParameters blitParams;
            blitParams.targetFramebuffer = framebuffer;
            blitParams.sourceTexture = source;
            if (use.stage == BloomStage::Downscale1)
                blitParams.sourceBox = use.sourceBox;
            m_CommonPasses->BlitTexture(commandList, blitParams, &m_BindingCache);
            break;
        }
        case BloomStage::BlurHorizontal:
        case BloomStage::BlurVertical:
        {
            const bool horizontal = use.stage == BloomStage::BlurHorizontal;
            const nvrhi::FramebufferInfoEx& info = framebuffer->getFramebufferInfo();
            const float effectiveSigma = clamp(use.sigmaInPixels * 0.25f, 1.f, 100.f);
            BloomConstants constants = {};
            constants.pixstep.x = horizontal ? 1.f / float(info.width) : 0.f;
            constants.pixstep.y = horizontal ? 0.f : 1.f / float(info.height);
            constants.argumentScale = -1.f / (2 * effectiveSigma * effectiveSigma);
            constants.normalizationScale = 1.f / (sqrtf(2 * PI_f) * effectiveSigma);
            constants.numSamples = ::round(effectiveSigma * 4.f);
            nvrhi::IBuffer* constantBuffer = horizontal ? m_BloomHBlurCB.Get() : m_BloomVBlurCB.Get();
            commandList->writeBuffer(constantBuffer, &constants, sizeof(constants));

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, constantBuffer),
                nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearClampSampler),
                nvrhi::BindingSetItem::Texture_SRV(0, source)
            };
            nvrhi::GraphicsState state;
            state.pipeline = m_GraphBlurPso;
            state.framebuffer = framebuffer;
            state.bindings = { m_BindingCache.GetOrCreateBindingSet(bindingSetDesc, m_BloomBlurBindingLayout) };
            const nvrhi::Viewport viewport(float(info.width), float(info.height));
            state.viewport.addViewport(viewport);
            state.viewport.addScissorRect(nvrhi::Rect(viewport));
            commandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.instanceCount = 1;
            args.vertexCount = 4;
            commandList->draw(args);
            break;
        }
        case BloomStage::Apply:
        {
            BlitParameters blitParams;
            blitParams.targetFramebuffer = framebuffer;
            blitParams.targetViewport = use.view->GetViewportState().viewports[0];
            blitParams.sourceTexture = source;
            blitParams.blendState.setBlendEnable(true)
                .setSrcBlend(nvrhi::BlendFactor::ConstantColor)
                .setDestBlend(nvrhi::BlendFactor::InvConstantColor)
                .setSrcBlendAlpha(nvrhi::BlendFactor::Zero)
                .setDestBlendAlpha(nvrhi::BlendFactor::One);
            blitParams.blendConstantColor = nvrhi::Color(use.blendFactor);
            m_CommonPasses->BlitTexture(commandList, blitParams, &m_BindingCache);
            break;
        }
        }
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

void BloomPass::OnResourcesReleased() noexcept
{
    m_BindingCache.Clear();
}
