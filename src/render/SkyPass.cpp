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

#include <donut/render/SkyPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/sky_ps.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/sky_ps.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/sky_ps.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

using namespace donut::engine;
using namespace donut::render;

void SkyPass::CreateCommonObjects(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory)
{
    m_PixelShader = shaderFactory->CreateAutoShader("donut/passes/sky_ps.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_sky_ps), nullptr, nvrhi::ShaderType::Pixel);

    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(SkyConstants);
    constantBufferDesc.debugName = "SkyConstants";
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
    device->createBuffer(constantBufferDesc, &m_SkyCB);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::Pixel;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0)
    };
    device->createBindingLayout(layoutDesc, &m_RenderBindingLayout);

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_SkyCB)
    };
    device->createBindingSet(bindingSetDesc, m_RenderBindingLayout, &m_RenderBindingSet);
}

nvrhi::GraphicsPipelineHandle SkyPass::CreatePipeline(nvrhi::IDevice* device, const nvrhi::FramebufferInfo& framebufferInfo,
                                                      bool reverseDepth) const
{
    nvrhi::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
    pipelineDesc.VS = reverseDepth ? m_CommonPasses->m_FullscreenVS : m_CommonPasses->m_FullscreenAtOneVS;
    pipelineDesc.PS = m_PixelShader;
    pipelineDesc.bindingLayouts = { m_RenderBindingLayout };

    pipelineDesc.renderState.rasterState.setCullNone();
    pipelineDesc.renderState.depthStencilState
        .enableDepthTest()
        .disableDepthWrite()
        .disableStencil()
        .setDepthFunc(reverseDepth
            ? nvrhi::ComparisonFunc::GreaterOrEqual
            : nvrhi::ComparisonFunc::LessOrEqual);

    nvrhi::GraphicsPipelineHandle pipeline;
    device->createGraphicsPipeline1(pipelineDesc, framebufferInfo, &pipeline);
    return pipeline;
}

SkyPass::SkyPass(
    nvrhi::IDevice* device,
    engine::ShaderFactory *shaderFactory,
    engine::CommonRenderPasses* commonPasses,
    engine::FramebufferFactory* framebufferFactory,
    const ICompositeView& compositeView)
    : m_FramebufferFactory(framebufferFactory)
    , m_CommonPasses(commonPasses)
{
    CreateCommonObjects(device, shaderFactory);

    const IView* sampleView = compositeView.GetChildView(ViewType::PLANAR, 0);
    m_RenderPso = CreatePipeline(device, m_FramebufferFactory->GetFramebufferInfo(), sampleView->IsReverseDepth());
}

SkyPass::SkyPass(
    nvrhi::IDevice* device,
    engine::ShaderFactory* shaderFactory,
    engine::CommonRenderPasses* commonPasses)
    : m_CommonPasses(commonPasses)
    , m_Device(device)
    , m_ShaderFactory(shaderFactory)
{
}

void SkyPass::RenderView(nvrhi::ICommandList* commandList, const IView& view, nvrhi::IFramebuffer* framebuffer,
                         nvrhi::IGraphicsPipeline* pipeline, const DirectionalLight& light, const SkyParameters& params) const
{
    nvrhi::GraphicsState state;
    state.pipeline = pipeline;
    state.framebuffer = framebuffer;
    state.bindings = { m_RenderBindingSet };
    state.viewport = view.GetViewportState();

    dm::affine viewToWorld = view.GetInverseViewMatrix();
    viewToWorld.m_translation = 0.f;
    dm::float4x4 clipToTranslatedWorld = view.GetInverseProjectionMatrix(true) * affineToHomogeneous(viewToWorld);

    SkyConstants skyConstants{};
    skyConstants.matClipToTranslatedWorld = clipToTranslatedWorld;
    FillShaderParameters(light, params, skyConstants.params);
    commandList->writeBuffer(m_SkyCB, &skyConstants, sizeof(skyConstants));

    commandList->setGraphicsState(state);

    nvrhi::DrawArguments args;
    args.instanceCount = 1;
    args.vertexCount = 4;
    commandList->draw(args);
}

void SkyPass::Render(
    nvrhi::ICommandList* commandList,
    const ICompositeView& compositeView,
    const DirectionalLight& light,
    const SkyParameters& params) const
{
    commandList->beginMarker("Sky");

    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);
        RenderView(commandList, *view, m_FramebufferFactory->GetFramebuffer(*view), m_RenderPso, light, params);
    }

    commandList->endMarker();
}

void SkyPass::FillShaderParameters(const engine::DirectionalLight& light, const SkyParameters& input, ProceduralSkyShaderParameters& output)
{
    float lightAngularSize = dm::radians(clamp(light.angularSize, 0.1f, 90.f));
    float lightSolidAngle = 4 * dm::PI_f * square(sinf(lightAngularSize * 0.5f));
    float lightRadiance = light.irradiance / lightSolidAngle;
    if (input.maxLightRadiance > 0.f)
        lightRadiance = min(lightRadiance, input.maxLightRadiance);

    output.directionToLight = float3(normalize(-light.GetDirection()));
    output.angularSizeOfLight = lightAngularSize;
    output.lightColor = lightRadiance * light.color;
    output.glowSize = dm::radians(dm::clamp(input.glowSize, 0.f, 90.f));
    output.skyColor = input.skyColor * input.brightness;
    output.glowIntensity = dm::clamp(input.glowIntensity, 0.f, 1.f);
    output.horizonColor = input.horizonColor * input.brightness;
    output.horizonSize = dm::radians(dm::clamp(input.horizonSize, 0.f, 90.f));
    output.groundColor = input.groundColor * input.brightness;
    output.glowSharpness = dm::clamp(input.glowSharpness, 1.f, 10.f);
    output.directionUp = normalize(input.directionUp);
}


// ------------------------------------------------------------------------------------------ render graph (13.3)

namespace
{
    struct SkyUse
    {
        const IView* view;
        ResourceId color;
        ResourceId depth;
        const DirectionalLight* light;
        SkyParameters params;
        nvrhi::SubresourceRegion region;
    };
}

SkyGraphOutputs SkyPass::AddToGraph(IRenderGraph* graph, const SkyGraphInputs& inputs)
{
    SkyGraphOutputs outputs{ inputs.color };
    if (!inputs.view || !inputs.light)
        return outputs;
    const uint32_t viewCount = inputs.view->GetNumChildViews(ViewType::PLANAR);
    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        SkyUse* use = NewFrameData<SkyUse>(graph);
        if (!use)
            break;
        use->view = inputs.view->GetChildView(ViewType::PLANAR, viewIndex);
        use->color = outputs.color;
        use->depth = inputs.depth;
        use->light = inputs.light;
        use->params = inputs.params;
        const nvrhi::TextureSubresourceSet subresources = use->view->GetSubresources();
        use->region = nvrhi::SubresourceRegion().setMips(subresources.baseMipLevel, 1)
            .setSlices(subresources.baseArraySlice, subresources.numArraySlices);
        graph->AddPass(PassDesc{ "Sky", PassType::Raster, PassFlags::None, 0 }, this, use, nullptr);
        outputs.color = use->color;
    }
    return outputs;
}

nvrhi::FRESULT SkyPass::Setup(ISetupContext* context, void* data) noexcept
{
    SkyUse& use = *static_cast<SkyUse*>(data);
    use.color = context->SetColorTarget(0, use.color, use.region, AccessFlags::None, nullptr);
    if (use.depth != InvalidResourceId)
        context->SetDepthTarget(use.depth, use.region, true, AccessFlags::None, nullptr);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT SkyPass::Prepare(IPrepareContext* context, void* data) noexcept
{
    const SkyUse& use = *static_cast<const SkyUse*>(data);
    try
    {
        if (!m_PixelShader)
        {
            if (!m_Device || !m_ShaderFactory)
                return nvrhi::FE_INVALID_ARGS;
            CreateCommonObjects(m_Device, m_ShaderFactory);
        }
        const nvrhi::FramebufferInfo framebufferInfo = context->GetFramebufferInfo().getInfo();
        const bool reverseDepth = use.view->IsReverseDepth();
        nvrhi::GraphicsPipelineHandle& pipeline = m_GraphPso[reverseDepth ? 1 : 0];
        if (!pipeline || pipeline->getFramebufferInfo() != framebufferInfo)
            pipeline = CreatePipeline(m_Device ? m_Device.Get() : context->GetDevice(), framebufferInfo, reverseDepth);
        return pipeline ? nvrhi::FS_OK : nvrhi::FE_GENERIC_ERROR;
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
}

nvrhi::FRESULT SkyPass::Execute(IPassContext* context, const void* data) noexcept
{
    const SkyUse& use = *static_cast<const SkyUse*>(data);
    nvrhi::IGraphicsPipeline* pipeline = m_GraphPso[use.view->IsReverseDepth() ? 1 : 0];
    nvrhi::IFramebuffer* framebuffer = context->GetFramebuffer();
    if (!pipeline || !framebuffer)
        return nvrhi::FE_INVALID_ARGS;
    RenderView(context->GetCommandList(), *use.view, framebuffer, pipeline, *use.light, use.params);
    return nvrhi::FS_OK;
}

void SkyPass::OnResourcesReleased() noexcept
{
}
