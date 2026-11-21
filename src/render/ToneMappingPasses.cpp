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

#include <donut/render/ToneMappingPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>
#include <atomic>
#include <sstream>
#include <string>
#include <assert.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/core/log.h>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/exposure_cs.dxbc.h"
#include "compiled_shaders/passes/histogram_cs.dxbc.h"
#include "compiled_shaders/passes/tonemapping_ps.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/exposure_cs.dxil.h"
#include "compiled_shaders/passes/histogram_cs.dxil.h"
#include "compiled_shaders/passes/tonemapping_ps.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/exposure_cs.spirv.h"
#include "compiled_shaders/passes/histogram_cs.spirv.h"
#include "compiled_shaders/passes/tonemapping_ps.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/tonemapping_cb.h>

using namespace donut::engine;
using namespace donut::render;

// The objects both constructions share: shaders, the constant buffer, the colour LUT, the compute pipelines and the
// render binding layout.
void ToneMappingPass::CreateObjects(nvrhi::IDevice* device, ShaderFactory* shaderFactory, const CreateParameters& params)
{
    {
        std::vector<ShaderMacro> Macros;

        std::stringstream ss;
        ss << params.histogramBins;
        Macros.push_back(ShaderMacro("HISTOGRAM_BINS", ss.str()));
        Macros.push_back(ShaderMacro("SOURCE_ARRAY", params.isTextureArray ? "1" : "0"));

        m_HistogramComputeShader = shaderFactory->CreateAutoShader("donut/passes/histogram_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_histogram_cs), &Macros, nvrhi::ShaderType::Compute);
        m_ExposureComputeShader = shaderFactory->CreateAutoShader("donut/passes/exposure_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_exposure_cs), &Macros, nvrhi::ShaderType::Compute);
        m_PixelShader = shaderFactory->CreateAutoShader("donut/passes/tonemapping_ps.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_tonemapping_ps), &Macros, nvrhi::ShaderType::Pixel);
    }

    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(ToneMappingConstants);
    constantBufferDesc.debugName = "ToneMappingConstants";
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.maxVersions = params.numConstantBufferVersions;
    device->createBuffer(constantBufferDesc, &m_ToneMappingCB);

    m_ColorLUT = m_CommonPasses->m_BlackTexture;

    if (params.colorLUT)
    {
        const nvrhi::TextureDesc& desc = params.colorLUT->getDesc();
        m_ColorLUTSize = float(desc.height);

        if (desc.width != desc.height * desc.height)
        {
            log::error("Color LUT texture size must be: width = (n*n), height = (n)");
            m_ColorLUTSize = 0.f;
        }
        else
        {
            m_ColorLUT = params.colorLUT;
        }
    }

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0)
        };
        device->createBindingLayout(layoutDesc, &m_HistogramBindingLayout);

        nvrhi::ComputePipelineDesc computePipelineDesc;
        computePipelineDesc.CS = m_HistogramComputeShader;
        computePipelineDesc.bindingLayouts = { m_HistogramBindingLayout };
        device->createComputePipeline(computePipelineDesc, &m_HistogramPso);
    }

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0)
        };
        device->createBindingLayout(layoutDesc, &m_ExposureBindingLayout);

        nvrhi::ComputePipelineDesc computePipelineDesc;
        computePipelineDesc.CS = m_ExposureComputeShader;
        computePipelineDesc.bindingLayouts = { m_ExposureBindingLayout };
        device->createComputePipeline(computePipelineDesc, &m_ExposurePso);
    }

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Pixel;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(1),
            nvrhi::BindingLayoutItem::Texture_SRV(2),
            nvrhi::BindingLayoutItem::Sampler(0)
        };
        device->createBindingLayout(layoutDesc, &m_RenderBindingLayout);
    }
}

static nvrhi::GraphicsPipelineHandle CreateToneMappingPipeline(nvrhi::IDevice* device, nvrhi::IShader* vertexShader,
    nvrhi::IShader* pixelShader, nvrhi::IBindingLayout* layout, const nvrhi::FramebufferInfo& framebufferInfo)
{
    nvrhi::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
    pipelineDesc.VS = vertexShader;
    pipelineDesc.PS = pixelShader;
    pipelineDesc.bindingLayouts = { layout };

    pipelineDesc.renderState.rasterState.setCullNone();
    pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
    pipelineDesc.renderState.depthStencilState.stencilEnable = false;

    nvrhi::GraphicsPipelineHandle pipeline;
    device->createGraphicsPipeline1(pipelineDesc, framebufferInfo, &pipeline);
    return pipeline;
}

ToneMappingPass::ToneMappingPass(
    nvrhi::IDevice* device,
    ShaderFactory *shaderFactory,
    CommonRenderPasses* commonPasses,
    FramebufferFactory* framebufferFactory,
    const ICompositeView& compositeView,
    const CreateParameters& params)
    : m_Device(device)
    , m_HistogramBins(params.histogramBins)
    , m_CommonPasses(commonPasses)
    , m_FramebufferFactory(framebufferFactory)
    , m_GraphBindings(device)
{
    assert(params.histogramBins <= 256);

    CreateObjects(device, shaderFactory, params);

    nvrhi::BufferDesc storageBufferDesc;
    storageBufferDesc.byteSize = sizeof(uint) * m_HistogramBins;
    storageBufferDesc.format = nvrhi::Format::R32_UINT;
    storageBufferDesc.canHaveUAVs = true;
    storageBufferDesc.debugName = "HistogramBuffer";
    storageBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    storageBufferDesc.keepInitialState = true;
    storageBufferDesc.canHaveTypedViews = true;
    device->createBuffer(storageBufferDesc, &m_HistogramBuffer);

    if (params.exposureBufferOverride)
    {
        m_ExposureBuffer = params.exposureBufferOverride;
    }
    else
    {
        storageBufferDesc.byteSize = sizeof(uint);
        storageBufferDesc.format = nvrhi::Format::R32_UINT;
        storageBufferDesc.debugName = "ExposureBuffer";
        device->createBuffer(storageBufferDesc, &m_ExposureBuffer);
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_ToneMappingCB),
        nvrhi::BindingSetItem::TypedBuffer_SRV(0, m_HistogramBuffer),
        nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_ExposureBuffer)
    };
    device->createBindingSet(bindingSetDesc, m_ExposureBindingLayout, &m_ExposureBindingSet);

    (void)compositeView;
    m_RenderPso = CreateToneMappingPipeline(device, m_CommonPasses->m_FullscreenVS, m_PixelShader, m_RenderBindingLayout,
                                            m_FramebufferFactory->GetFramebufferInfo());
}

ToneMappingPass::ToneMappingPass(
    nvrhi::IDevice* device,
    ShaderFactory *shaderFactory,
    CommonRenderPasses* commonPasses,
    const CreateParameters& params)
    : m_Device(device)
    , m_HistogramBins(params.histogramBins)
    , m_CommonPasses(commonPasses)
    , m_ShaderFactory(shaderFactory)
    , m_CreateParams(params)
    , m_GraphBindings(device)
{
    assert(params.histogramBins <= 256);

    // The graph's buffers, described once so that recording allocates nothing (10.3).
    m_HistogramDesc = nvrhi::BufferDesc()
        .setByteSize(sizeof(uint32_t) * m_HistogramBins)
        .setFormat(nvrhi::Format::R32_UINT)
        .setCanHaveUAVs(true)
        .setCanHaveTypedViews(true)
        .setDebugName("ToneMapping/Histogram");
    m_ExposureDesc = nvrhi::BufferDesc()
        .setByteSize(sizeof(uint32_t))
        .setFormat(nvrhi::Format::R32_UINT)
        .setCanHaveUAVs(true)
        .setCanHaveTypedViews(true)
        .setDebugName("ToneMapping/Exposure");
    static std::atomic<uint32_t> s_Instances{ 0 };
    m_ExposureName = "donut.ToneMappingPass.Exposure#" + std::to_string(++s_Instances);
}

void ToneMappingPass::Render(
    nvrhi::ICommandList* commandList,
    const ToneMappingParameters& params,
    const ICompositeView& compositeView,
    nvrhi::ITexture* sourceTexture)
{
    nvrhi::BindingSetHandle& bindingSet = m_RenderBindingSets[sourceTexture];
    if (!bindingSet)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ToneMappingCB),
            nvrhi::BindingSetItem::Texture_SRV(0, sourceTexture),
            nvrhi::BindingSetItem::TypedBuffer_SRV(1, m_ExposureBuffer),
            nvrhi::BindingSetItem::Texture_SRV(2, m_ColorLUT),
            nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearClampSampler)
        };
        m_Device->createBindingSet(bindingSetDesc, m_RenderBindingLayout, &bindingSet);
    }

    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);
        DrawView(commandList, params, *view, m_FramebufferFactory->GetFramebuffer(*view), m_RenderPso, bindingSet);
    }
}

void ToneMappingPass::DrawView(nvrhi::ICommandList* commandList, const ToneMappingParameters& params, const IView& view,
                               nvrhi::IFramebuffer* framebuffer, nvrhi::IGraphicsPipeline* pipeline, nvrhi::IBindingSet* bindingSet)
{
    nvrhi::GraphicsState state;
    state.pipeline = pipeline;
    state.framebuffer = framebuffer;
    state.bindings = { bindingSet };
    state.viewport = view.GetViewportState();

    bool enableColorLUT = params.enableColorLUT && m_ColorLUTSize > 0;

    ToneMappingConstants toneMappingConstants = {};
    toneMappingConstants.exposureScale = ::exp2f(params.exposureBias);
    toneMappingConstants.whitePointInvSquared = 1.f / powf(params.whitePoint, 2.f);
    toneMappingConstants.minAdaptedLuminance = params.minAdaptedLuminance;
    toneMappingConstants.maxAdaptedLuminance = params.maxAdaptedLuminance;
    toneMappingConstants.sourceSlice = view.GetSubresources().baseArraySlice;
    toneMappingConstants.colorLUTTextureSize = enableColorLUT ? float2(m_ColorLUTSize * m_ColorLUTSize, m_ColorLUTSize) : float2(0.f);
    toneMappingConstants.colorLUTTextureSizeInv = enableColorLUT ? 1.f / toneMappingConstants.colorLUTTextureSize : float2(0.f);
    commandList->writeBuffer(m_ToneMappingCB, &toneMappingConstants, sizeof(toneMappingConstants));

    commandList->setGraphicsState(state);

    nvrhi::DrawArguments args;
    args.instanceCount = 1;
    args.vertexCount = 4;
    commandList->draw(args);
}

void ToneMappingPass::SimpleRender(
    nvrhi::ICommandList* commandList, 
    const ToneMappingParameters& params, 
    const ICompositeView& compositeView, nvrhi
    ::ITexture* sourceTexture)
{
    commandList->beginMarker("ToneMapping");
    ResetHistogram(commandList);
    AddFrameToHistogram(commandList, compositeView, sourceTexture);
    ComputeExposure(commandList, params);
    Render(commandList, params, compositeView, sourceTexture);
    commandList->endMarker();
}

nvrhi::BufferHandle ToneMappingPass::GetExposureBuffer()
{
    return m_ExposureBuffer;
}

void ToneMappingPass::AdvanceFrame(float frameTime)
{
    m_FrameTime = frameTime;
}

void ToneMappingPass::ResetExposure(nvrhi::ICommandList* commandList, float initialExposure)
{
    uint32_t uintValue = *(uint32_t*)&initialExposure;
    commandList->clearBufferUInt(m_ExposureBuffer, uintValue);
}

void ToneMappingPass::ResetHistogram(nvrhi::ICommandList* commandList)
{
    commandList->clearBufferUInt(m_HistogramBuffer, 0);
}

static const float g_minLogLuminance = -10; // TODO: figure out how to set these properly
static const float g_maxLogLuminamce = 4;

void ToneMappingPass::AddFrameToHistogram(nvrhi::ICommandList* commandList, const ICompositeView& compositeView, nvrhi::ITexture* sourceTexture)
{
    nvrhi::BindingSetHandle& bindingSet = m_HistogramBindingSets[sourceTexture];
    if (!bindingSet)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ToneMappingCB),
            nvrhi::BindingSetItem::Texture_SRV(0, sourceTexture),
            nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_HistogramBuffer)
        };

        m_Device->createBindingSet(bindingSetDesc, m_HistogramBindingLayout, &bindingSet);
    }

    DispatchHistogram(commandList, compositeView, bindingSet);
}

void ToneMappingPass::DispatchHistogram(nvrhi::ICommandList* commandList, const ICompositeView& compositeView,
                                        nvrhi::IBindingSet* bindingSet)
{
    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);

        nvrhi::ViewportState viewportState = view->GetViewportState();

        for (uint viewportIndex = 0; viewportIndex < viewportState.scissorRects.size(); viewportIndex++)
        {
            ToneMappingConstants toneMappingConstants = {};
            toneMappingConstants.logLuminanceScale = 1.0f / (g_maxLogLuminamce - g_minLogLuminance);
            toneMappingConstants.logLuminanceBias = -g_minLogLuminance * toneMappingConstants.logLuminanceScale;

            nvrhi::Rect& scissor = viewportState.scissorRects[viewportIndex];
            toneMappingConstants.viewOrigin = uint2(scissor.minX, scissor.minY);
            toneMappingConstants.viewSize = uint2(scissor.maxX - scissor.minX, scissor.maxY - scissor.minY);
            toneMappingConstants.sourceSlice = view->GetSubresources().baseArraySlice;

            commandList->writeBuffer(m_ToneMappingCB, &toneMappingConstants, sizeof(toneMappingConstants));

            nvrhi::ComputeState state;
            state.pipeline = m_HistogramPso;
            state.bindings = { bindingSet };
            commandList->setComputeState(state);

            uint2 numGroups = (toneMappingConstants.viewSize + uint2(15)) / uint2(16);
            commandList->dispatch(numGroups.x, numGroups.y, 1);
        }
    }
}

void ToneMappingPass::ComputeExposure(nvrhi::ICommandList* commandList, const ToneMappingParameters& params)
{
    DispatchExposure(commandList, params, m_FrameTime, m_ExposureBindingSet);
}

void ToneMappingPass::DispatchExposure(nvrhi::ICommandList* commandList, const ToneMappingParameters& params, float frameTime,
                                       nvrhi::IBindingSet* bindingSet)
{
    ToneMappingConstants toneMappingConstants = {};
    toneMappingConstants.logLuminanceScale = g_maxLogLuminamce - g_minLogLuminance;
    toneMappingConstants.logLuminanceBias = g_minLogLuminance;
    toneMappingConstants.histogramLowPercentile = std::min(0.99f, std::max(0.f, params.histogramLowPercentile));
    toneMappingConstants.histogramHighPercentile = std::min(1.f, std::max(toneMappingConstants.histogramLowPercentile, params.histogramHighPercentile));
    toneMappingConstants.eyeAdaptationSpeedUp = params.eyeAdaptationSpeedUp;
    toneMappingConstants.eyeAdaptationSpeedDown = params.eyeAdaptationSpeedDown;
    toneMappingConstants.minAdaptedLuminance = params.minAdaptedLuminance;
    toneMappingConstants.maxAdaptedLuminance = params.maxAdaptedLuminance;
    toneMappingConstants.frameTime = frameTime;

    commandList->writeBuffer(m_ToneMappingCB, &toneMappingConstants, sizeof(toneMappingConstants));

    nvrhi::ComputeState state;
    state.pipeline = m_ExposurePso;
    state.bindings = { bindingSet };
    commandList->setComputeState(state);

    commandList->dispatch(1);
}

// ------------------------------------------------------------------------------------------ render graph (13.3)

namespace
{
    enum class ToneMappingStage : uint32_t { Histogram, Exposure, Resolve };

    struct ToneMappingUse
    {
        ToneMappingStage stage;
        const ICompositeView* compositeView;   // Histogram
        const IView* view;                     // Resolve
        ResourceId source;
        ResourceId histogram;
        ResourceId exposure;
        ResourceId output;
        ToneMappingParameters params;
        float frameTime;
        nvrhi::SubresourceRegion region;
    };
}

ToneMappingGraphOutputs ToneMappingPass::AddToGraph(IRenderGraph* graph, const ToneMappingGraphInputs& inputs)
{
    ToneMappingGraphOutputs outputs{ inputs.output, InvalidResourceId };
    if (!inputs.view || inputs.source == InvalidResourceId || inputs.output == InvalidResourceId)
        return outputs;

    ResourceId histogram = graph->CreateBuffer(m_HistogramDesc, ResourceFlags::None);
    ResourceId exposure = graph->GetPersistentBuffer(m_ExposureName.c_str(), m_ExposureDesc);

    ToneMappingUse* use = NewFrameData<ToneMappingUse>(graph);
    if (!use)
        return outputs;
    *use = ToneMappingUse{ ToneMappingStage::Histogram, inputs.view, nullptr, inputs.source, histogram, exposure,
                           InvalidResourceId, inputs.params, inputs.frameTime, nvrhi::SubresourceRegion() };
    graph->AddPass(PassDesc{ "ToneMapping/Histogram", PassType::Compute, PassFlags::None, 0 }, this, use, nullptr);
    histogram = use->histogram;

    use = NewFrameData<ToneMappingUse>(graph);
    if (!use)
        return outputs;
    *use = ToneMappingUse{ ToneMappingStage::Exposure, nullptr, nullptr, inputs.source, histogram, exposure,
                           InvalidResourceId, inputs.params, inputs.frameTime, nvrhi::SubresourceRegion() };
    graph->AddPass(PassDesc{ "ToneMapping/Exposure", PassType::Compute, PassFlags::None, 0 }, this, use, nullptr);
    exposure = use->exposure;
    outputs.exposure = exposure;

    const uint32_t viewCount = inputs.view->GetNumChildViews(ViewType::PLANAR);
    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        use = NewFrameData<ToneMappingUse>(graph);
        if (!use)
            break;
        const IView* view = inputs.view->GetChildView(ViewType::PLANAR, viewIndex);
        const nvrhi::TextureSubresourceSet subresources = view->GetSubresources();
        *use = ToneMappingUse{ ToneMappingStage::Resolve, nullptr, view, inputs.source, histogram, exposure, outputs.output,
                               inputs.params, inputs.frameTime,
                               nvrhi::SubresourceRegion().setMips(subresources.baseMipLevel, 1)
                                   .setSlices(subresources.baseArraySlice, subresources.numArraySlices) };
        graph->AddPass(PassDesc{ "ToneMapping", PassType::Raster, PassFlags::None, 0 }, this, use, nullptr);
        outputs.output = use->output;
    }
    return outputs;
}

nvrhi::FRESULT ToneMappingPass::Setup(ISetupContext* context, void* data) noexcept
{
    ToneMappingUse& use = *static_cast<ToneMappingUse*>(data);
    switch (use.stage)
    {
    case ToneMappingStage::Histogram:
        context->Read(use.source, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
        use.histogram = context->Write(use.histogram, Access::UAVWrite, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion(),
                                       AccessFlags::Clear);
        break;
    case ToneMappingStage::Exposure:
        context->Read(use.histogram, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
        use.exposure = context->Write(use.exposure, Access::UAVWrite, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion(),
                                      AccessFlags::None);
        break;
    case ToneMappingStage::Resolve:
        context->Read(use.source, Access::ShaderRead, nvrhi::ShaderType::Pixel, nvrhi::SubresourceRegion());
        context->Read(use.exposure, Access::ShaderRead, nvrhi::ShaderType::Pixel, nvrhi::SubresourceRegion());
        use.output = context->SetColorTarget(0, use.output, use.region, AccessFlags::FullWrite, nullptr);
        break;
    }
    return nvrhi::FS_OK;
}

nvrhi::FRESULT ToneMappingPass::Prepare(IPrepareContext* context, void* data) noexcept
{
    const ToneMappingUse& use = *static_cast<const ToneMappingUse*>(data);
    try
    {
        if (!m_HistogramPso)
        {
            if (!m_Device || !m_ShaderFactory)
                return nvrhi::FE_INVALID_ARGS;
            CreateObjects(m_Device, m_ShaderFactory, m_CreateParams);
        }
        if (use.stage == ToneMappingStage::Resolve)
        {
            const nvrhi::FramebufferInfo framebufferInfo = context->GetFramebufferInfo().getInfo();
            if (!m_GraphRenderPso || m_GraphRenderPso->getFramebufferInfo() != framebufferInfo)
                m_GraphRenderPso = CreateToneMappingPipeline(m_Device, m_CommonPasses->m_FullscreenVS, m_PixelShader,
                                                             m_RenderBindingLayout, framebufferInfo);
            if (!m_GraphRenderPso)
                return nvrhi::FE_GENERIC_ERROR;
        }
        return nvrhi::FS_OK;
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
}

nvrhi::FRESULT ToneMappingPass::Execute(IPassContext* context, const void* data) noexcept
{
    const ToneMappingUse& use = *static_cast<const ToneMappingUse*>(data);
    nvrhi::ICommandList* commandList = context->GetCommandList();
    try
    {
        switch (use.stage)
        {
        case ToneMappingStage::Histogram:
        {
            nvrhi::BindingSetDesc desc;
            desc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ToneMappingCB),
                nvrhi::BindingSetItem::Texture_SRV(0, context->GetTexture(use.source)),
                nvrhi::BindingSetItem::TypedBuffer_UAV(0, context->GetBuffer(use.histogram))
            };
            nvrhi::BindingSetHandle bindingSet = m_GraphBindings.GetOrCreateBindingSet(desc, m_HistogramBindingLayout);
            DispatchHistogram(commandList, *use.compositeView, bindingSet);
            break;
        }
        case ToneMappingStage::Exposure:
        {
            nvrhi::BindingSetDesc desc;
            desc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ToneMappingCB),
                nvrhi::BindingSetItem::TypedBuffer_SRV(0, context->GetBuffer(use.histogram)),
                nvrhi::BindingSetItem::TypedBuffer_UAV(0, context->GetBuffer(use.exposure))
            };
            nvrhi::BindingSetHandle bindingSet = m_GraphBindings.GetOrCreateBindingSet(desc, m_ExposureBindingLayout);
            // A new exposure holds nothing: no adaptation from it, the shader writes the target exposure.
            ToneMappingParameters params = use.params;
            if (context->IsNewAllocation(use.exposure))
                params.eyeAdaptationSpeedUp = params.eyeAdaptationSpeedDown = 0.f;
            DispatchExposure(commandList, params, use.frameTime, bindingSet);
            break;
        }
        case ToneMappingStage::Resolve:
        {
            nvrhi::BindingSetDesc desc;
            desc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ToneMappingCB),
                nvrhi::BindingSetItem::Texture_SRV(0, context->GetTexture(use.source)),
                nvrhi::BindingSetItem::TypedBuffer_SRV(1, context->GetBuffer(use.exposure)),
                nvrhi::BindingSetItem::Texture_SRV(2, m_ColorLUT),
                nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearClampSampler)
            };
            nvrhi::BindingSetHandle bindingSet = m_GraphBindings.GetOrCreateBindingSet(desc, m_RenderBindingLayout);
            nvrhi::IFramebuffer* framebuffer = context->GetFramebuffer();
            if (!framebuffer || !m_GraphRenderPso)
                return nvrhi::FE_INVALID_ARGS;
            DrawView(commandList, use.params, *use.view, framebuffer, m_GraphRenderPso, bindingSet);
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

void ToneMappingPass::OnResourcesReleased() noexcept
{
    m_GraphBindings.Clear();
}
