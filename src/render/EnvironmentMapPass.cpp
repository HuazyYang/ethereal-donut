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

#include <donut/render/EnvironmentMapPass.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>
#include <donut/core/math/math.h>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/environment_map_ps.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/environment_map_ps.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/environment_map_ps.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

using namespace donut::engine;
using namespace donut::render;

EnvironmentMapPass::EnvironmentMapPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    CommonRenderPasses* commonPasses,
    FramebufferFactory *framebufferFactory,
    const ICompositeView& compositeView,
    nvrhi::ITexture* environmentMap)
    : m_CommonPasses(commonPasses)
    , m_FramebufferFactory(framebufferFactory)
{
    nvrhi::TextureDimension envMapDimension = environmentMap->getDesc().dimension;
    bool isCubeMap = (envMapDimension == nvrhi::TextureDimension::TextureCube) || 
        (envMapDimension == nvrhi::TextureDimension::TextureCubeArray);

    std::vector<engine::ShaderMacro> PSMacros;
    PSMacros.push_back(engine::ShaderMacro("LATLONG_TEXTURE", isCubeMap ? "0" : "1"));

    m_PixelShader = shaderFactory->CreateAutoShader("donut/passes/environment_map_ps.hlsl", "main", 
        DONUT_MAKE_PLATFORM_SHADER(g_environment_map_ps), &PSMacros, nvrhi::ShaderType::Pixel);

    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(SkyConstants);
    constantBufferDesc.debugName = "SkyConstants";
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
    device->createBuffer(constantBufferDesc, &m_SkyCB);

    const IView* sampleView = compositeView.GetChildView(ViewType::PLANAR, 0);

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Pixel;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Sampler(0)
        };
        device->createBindingLayout(layoutDesc, &m_RenderBindingLayout);

        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_SkyCB),
            nvrhi::BindingSetItem::Texture_SRV(0, environmentMap),
            nvrhi::BindingSetItem::Sampler(0, commonPasses->m_LinearWrapSampler)
        };
        device->createBindingSet(bindingSetDesc, m_RenderBindingLayout, &m_RenderBindingSet);

        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
        pipelineDesc.VS = sampleView->IsReverseDepth() ? m_CommonPasses->m_FullscreenVS : m_CommonPasses->m_FullscreenAtOneVS;
        pipelineDesc.PS = m_PixelShader;
        pipelineDesc.bindingLayouts = { m_RenderBindingLayout };

        pipelineDesc.renderState.rasterState.setCullNone();
        pipelineDesc.renderState.depthStencilState
            .enableDepthTest()
            .disableDepthWrite()
            .disableStencil()
            .setDepthFunc(sampleView->IsReverseDepth()
                ? nvrhi::ComparisonFunc::GreaterOrEqual
                : nvrhi::ComparisonFunc::LessOrEqual);

        device->createGraphicsPipeline1(pipelineDesc, m_FramebufferFactory->GetFramebufferInfo(), &m_RenderPso);
    }
}

void EnvironmentMapPass::Render(
    nvrhi::ICommandList* commandList,
    const ICompositeView& compositeView)
{
    commandList->beginMarker("Environment Map");

    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);
        DrawView(commandList, *view, m_FramebufferFactory->GetFramebuffer(*view), m_RenderPso, m_RenderBindingSet);
    }

    commandList->endMarker();
}

void EnvironmentMapPass::DrawView(nvrhi::ICommandList* commandList, const IView& view, nvrhi::IFramebuffer* framebuffer,
                                  nvrhi::IGraphicsPipeline* pipeline, nvrhi::IBindingSet* bindingSet)
{
    nvrhi::GraphicsState state;
    state.pipeline = pipeline;
    state.framebuffer = framebuffer;
    state.bindings = { bindingSet };
    state.viewport = view.GetViewportState();

    SkyConstants skyConstants = {};
    skyConstants.matClipToTranslatedWorld = view.GetInverseViewProjectionMatrix() * affineToHomogeneous(translation(-view.GetViewOrigin()));
    commandList->writeBuffer(m_SkyCB, &skyConstants, sizeof(skyConstants));

    commandList->setGraphicsState(state);

    nvrhi::DrawArguments args;
    args.instanceCount = 1;
    args.vertexCount = 4;
    commandList->draw(args);
}

// ------------------------------------------------------------------------------------------ render graph (13.3)

EnvironmentMapPass::EnvironmentMapPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    CommonRenderPasses* commonPasses)
    : m_CommonPasses(commonPasses)
    , m_Device(device)
    , m_ShaderFactory(shaderFactory)
    , m_GraphBindings(device)
{
}

namespace
{
    struct EnvironmentMapUse
    {
        const IView* view;
        ResourceId color;
        ResourceId depth;
        ResourceId environmentMap;
        nvrhi::SubresourceRegion region;
    };
}

EnvironmentMapGraphOutputs EnvironmentMapPass::AddToGraph(IRenderGraph* graph, const EnvironmentMapGraphInputs& inputs)
{
    EnvironmentMapGraphOutputs outputs{ inputs.color };
    if (!inputs.view || inputs.environmentMap == InvalidResourceId)
        return outputs;
    const uint32_t viewCount = inputs.view->GetNumChildViews(ViewType::PLANAR);
    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        const IView* view = inputs.view->GetChildView(ViewType::PLANAR, viewIndex);
        const nvrhi::TextureSubresourceSet subresources = view->GetSubresources();
        EnvironmentMapUse* use = NewFrameData<EnvironmentMapUse>(graph, view, outputs.color, inputs.depth, inputs.environmentMap,
            nvrhi::SubresourceRegion().setMips(subresources.baseMipLevel, 1)
                .setSlices(subresources.baseArraySlice, subresources.numArraySlices));
        if (!use)
            break;
        graph->AddPass(PassDesc{ "EnvironmentMap", PassType::Raster, PassFlags::None, 0 }, this, use, nullptr);
        outputs.color = use->color;
    }
    return outputs;
}

nvrhi::FRESULT EnvironmentMapPass::Setup(ISetupContext* context, void* data) noexcept
{
    EnvironmentMapUse& use = *static_cast<EnvironmentMapUse*>(data);
    context->Read(use.environmentMap, Access::ShaderRead, nvrhi::ShaderType::Pixel, nvrhi::SubresourceRegion());
    use.color = context->SetColorTarget(0, use.color, use.region, AccessFlags::None, nullptr);
    if (use.depth != InvalidResourceId)
        context->SetDepthTarget(use.depth, use.region, true, AccessFlags::None, nullptr);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT EnvironmentMapPass::Prepare(IPrepareContext* context, void* data) noexcept
{
    const EnvironmentMapUse& use = *static_cast<const EnvironmentMapUse*>(data);
    if (!m_Device || !m_ShaderFactory)
        return nvrhi::FE_INVALID_ARGS;   // the immediate construction has no graph path
    try
    {
        const nvrhi::TextureDesc* mapDesc = context->GetTextureDesc(use.environmentMap);
        if (!mapDesc)
            return nvrhi::FE_INVALID_ARGS;
        const bool isCubeMap = mapDesc->dimension == nvrhi::TextureDimension::TextureCube ||
            mapDesc->dimension == nvrhi::TextureDimension::TextureCubeArray;
        if (!m_SkyCB)
        {
            nvrhi::BufferDesc constantBufferDesc;
            constantBufferDesc.byteSize = sizeof(SkyConstants);
            constantBufferDesc.debugName = "SkyConstants";
            constantBufferDesc.isConstantBuffer = true;
            constantBufferDesc.isVolatile = true;
            constantBufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
            m_Device->createBuffer(constantBufferDesc, &m_SkyCB);

            nvrhi::BindingLayoutDesc layoutDesc;
            layoutDesc.visibility = nvrhi::ShaderType::Pixel;
            layoutDesc.bindings = {
                nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
                nvrhi::BindingLayoutItem::Texture_SRV(0),
                nvrhi::BindingLayoutItem::Sampler(0)
            };
            m_Device->createBindingLayout(layoutDesc, &m_RenderBindingLayout);
        }
        if (m_GraphCubeMap != int(isCubeMap))
        {
            std::vector<engine::ShaderMacro> macros;
            macros.push_back(engine::ShaderMacro("LATLONG_TEXTURE", isCubeMap ? "0" : "1"));
            m_PixelShader = m_ShaderFactory->CreateAutoShader("donut/passes/environment_map_ps.hlsl", "main",
                DONUT_MAKE_PLATFORM_SHADER(g_environment_map_ps), &macros, nvrhi::ShaderType::Pixel);
            m_GraphPso[0] = m_GraphPso[1] = nullptr;
            m_GraphCubeMap = int(isCubeMap);
        }

        const nvrhi::FramebufferInfo framebufferInfo = context->GetFramebufferInfo().getInfo();
        const bool reverseDepth = use.view->IsReverseDepth();
        nvrhi::GraphicsPipelineHandle& pipeline = m_GraphPso[reverseDepth ? 1 : 0];
        if (!pipeline || pipeline->getFramebufferInfo() != framebufferInfo)
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
            pipeline = nullptr;
            m_Device->createGraphicsPipeline1(pipelineDesc, framebufferInfo, &pipeline);
        }
        return pipeline ? nvrhi::FS_OK : nvrhi::FE_GENERIC_ERROR;
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
}

nvrhi::FRESULT EnvironmentMapPass::Execute(IPassContext* context, const void* data) noexcept
{
    const EnvironmentMapUse& use = *static_cast<const EnvironmentMapUse*>(data);
    nvrhi::IGraphicsPipeline* pipeline = m_GraphPso[use.view->IsReverseDepth() ? 1 : 0];
    nvrhi::IFramebuffer* framebuffer = context->GetFramebuffer();
    nvrhi::ITexture* environmentMap = context->GetTexture(use.environmentMap);
    if (!pipeline || !framebuffer || !environmentMap)
        return nvrhi::FE_INVALID_ARGS;
    try
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_SkyCB),
            nvrhi::BindingSetItem::Texture_SRV(0, environmentMap),
            nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearWrapSampler)
        };
        DrawView(context->GetCommandList(), *use.view, framebuffer, pipeline,
                 m_GraphBindings.GetOrCreateBindingSet(bindingSetDesc, m_RenderBindingLayout));
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

void EnvironmentMapPass::OnResourcesReleased() noexcept
{
    m_GraphBindings.Clear();
}
