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

#include <donut/render/TemporalAntiAliasingPass.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/motion_vectors_ps.dxbc.h"
#include "compiled_shaders/passes/taa_cs.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/motion_vectors_ps.dxil.h"
#include "compiled_shaders/passes/taa_cs.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/motion_vectors_ps.spirv.h"
#include "compiled_shaders/passes/taa_cs.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/taa_cb.h>

#include <nvrhi/utils.h>

#include <assert.h>
#include <atomic>
#include <string>
#include <random>

using namespace donut::engine;
using namespace donut::render;

TemporalAntiAliasingPass::TemporalAntiAliasingPass(
    nvrhi::IDevice* device,
    ShaderFactory *shaderFactory, 
    CommonRenderPasses *commonPasses,
    const ICompositeView& compositeView,
    const CreateParameters& params)
    : m_CommonPasses(commonPasses)
    , m_FrameIndex(0)
    , m_StencilMask(params.motionVectorStencilMask)
    , m_R2Jitter(0.0f, 0.0f)
    , m_Jitter(TemporalAntiAliasingJitter::MSAA)
    , m_GraphBindings(device)
{
    const nvrhi::TextureDesc& unresolvedColorDesc = params.unresolvedColor->getDesc();
    const nvrhi::TextureDesc& resolvedColorDesc = params.resolvedColor->getDesc();
#ifdef _DEBUG
    const nvrhi::TextureDesc& feedback1Desc = params.feedback1->getDesc();
    const nvrhi::TextureDesc& feedback2Desc = params.feedback2->getDesc();

    assert(feedback1Desc.width == feedback2Desc.width);
    assert(feedback1Desc.height == feedback2Desc.height);
    assert(feedback1Desc.format == feedback2Desc.format);
    assert(feedback1Desc.isUAV);
    assert(feedback2Desc.isUAV);
    assert(resolvedColorDesc.isUAV);
#endif

    bool useStencil = false;
    nvrhi::Format stencilFormat = nvrhi::Format::UNKNOWN;
    if (params.motionVectorStencilMask)
    {
        useStencil = true;

        nvrhi::Format depthFormat = params.sourceDepth->getDesc().format;

        if (depthFormat == nvrhi::Format::D24S8)
            stencilFormat = nvrhi::Format::X24G8_UINT;
        else if (depthFormat == nvrhi::Format::D32S8)
            stencilFormat = nvrhi::Format::X32G8_UINT;
        else
            assert(!"the format of sourceDepth texture doesn't have a stencil plane");
    }

    std::vector<ShaderMacro> MotionVectorMacros;
    MotionVectorMacros.push_back(ShaderMacro("USE_STENCIL", useStencil ? "1" : "0"));
    m_MotionVectorPS = shaderFactory->CreateAutoShader("donut/passes/motion_vectors_ps.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_motion_vectors_ps), &MotionVectorMacros, nvrhi::ShaderType::Pixel);
    
    std::vector<ShaderMacro> ResolveMacros;
    ResolveMacros.push_back(ShaderMacro("SAMPLE_COUNT", std::to_string(unresolvedColorDesc.sampleCount)));
    ResolveMacros.push_back(ShaderMacro("USE_CATMULL_ROM_FILTER", params.useCatmullRomFilter ? "1" : "0"));
    m_TemporalAntiAliasingCS = shaderFactory->CreateAutoShader("donut/passes/taa_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_taa_cs), &ResolveMacros, nvrhi::ShaderType::Compute);

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.addressU = samplerDesc.addressV = samplerDesc.addressW = nvrhi::SamplerAddressMode::Border;
    samplerDesc.borderColor = nvrhi::Color(0.0f);
    device->createSampler(samplerDesc, &m_BilinearSampler);

    m_ResolvedColorSize = float2(float(resolvedColorDesc.width), float(resolvedColorDesc.height));

    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(TemporalAntiAliasingConstants);
    constantBufferDesc.debugName = "TemporalAntiAliasingConstants";
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.maxVersions = params.numConstantBufferVersions;
    device->createBuffer(constantBufferDesc, &m_TemporalAntiAliasingCB);

    if(params.sourceDepth)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Pixel;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0)
        };

        if (useStencil)
        {
            layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(1));
        }

        device->createBindingLayout(layoutDesc, &m_MotionVectorsBindingLayout);

        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_TemporalAntiAliasingCB),
            nvrhi::BindingSetItem::Texture_SRV(0, params.sourceDepth),
        };
        if (useStencil)
        {
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(1, params.sourceDepth, stencilFormat));
        }
        device->createBindingSet(bindingSetDesc, m_MotionVectorsBindingLayout, &m_MotionVectorsBindingSet);

        m_MotionVectorsFramebufferFactory = MAKE_RC_OBJ_PTR(FramebufferFactory, device);
        m_MotionVectorsFramebufferFactory->RenderTargets = { params.motionVectors };

        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
        pipelineDesc.VS = m_CommonPasses->m_FullscreenVS;
        pipelineDesc.PS = m_MotionVectorPS;
        pipelineDesc.bindingLayouts = { m_MotionVectorsBindingLayout };

        pipelineDesc.renderState.rasterState.setCullNone();
        pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
        pipelineDesc.renderState.depthStencilState.stencilEnable = false;

        nvrhi::FramebufferInfo framebufferInfo = m_MotionVectorsFramebufferFactory->GetFramebufferInfo();

        device->createGraphicsPipeline1(pipelineDesc, framebufferInfo, &m_MotionVectorsPso);
    }

    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_TemporalAntiAliasingCB),
            nvrhi::BindingSetItem::Sampler(0, m_BilinearSampler),
            nvrhi::BindingSetItem::Texture_SRV(0, params.unresolvedColor),
            nvrhi::BindingSetItem::Texture_SRV(1, params.motionVectors),
            nvrhi::BindingSetItem::Texture_SRV(2, params.feedback1),
            nvrhi::BindingSetItem::Texture_UAV(0, params.resolvedColor),
            nvrhi::BindingSetItem::Texture_UAV(1, params.feedback2)
        };
        
        m_HasHistoryClampRelaxTexture = params.historyClampRelax != nullptr;
        if (params.historyClampRelax != nullptr)
        {
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(3, params.historyClampRelax));
        }
        else
        {
            // No relax mask, but we need to bind something to match the shader binding slots
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(3, params.unresolvedColor));
        }

        nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_ResolveBindingLayout, m_ResolveBindingSet);
     
        // Swap resolvedColor and resolvedColorPrevious (t2 and u0)
        bindingSetDesc.bindings[4].resourceHandle = params.feedback2;
        bindingSetDesc.bindings[6].resourceHandle = params.feedback1;
        device->createBindingSet(bindingSetDesc, m_ResolveBindingLayout, &m_ResolveBindingSetPrevious);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.CS = m_TemporalAntiAliasingCS;
        pipelineDesc.bindingLayouts = { m_ResolveBindingLayout };

        device->createComputePipeline(pipelineDesc, &m_ResolvePso);
    }
}

void TemporalAntiAliasingPass::RenderMotionVectors(
    nvrhi::ICommandList* commandList,
    const ICompositeView& compositeView,
    const ICompositeView& compositeViewPrevious,
    dm::float3 preViewTranslationDifference)
{
    assert(compositeView.GetNumChildViews(ViewType::PLANAR) == compositeViewPrevious.GetNumChildViews(ViewType::PLANAR));
    assert(m_MotionVectorsPso);

    commandList->beginMarker("MotionVectors");

    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);
        const IView* viewPrevious = compositeViewPrevious.GetChildView(ViewType::PLANAR, viewIndex);
        DrawMotionVectors(commandList, *view, *viewPrevious, m_MotionVectorsFramebufferFactory->GetFramebuffer(*view),
                          m_MotionVectorsPso, m_MotionVectorsBindingSet, preViewTranslationDifference);
    }

    commandList->endMarker();
}

void TemporalAntiAliasingPass::DrawMotionVectors(nvrhi::ICommandList* commandList, const IView& view, const IView& viewPrevious,
                                                 nvrhi::IFramebuffer* framebuffer, nvrhi::IGraphicsPipeline* pipeline,
                                                 nvrhi::IBindingSet* bindingSet, dm::float3 preViewTranslationDifference)
{
    const nvrhi::ViewportState viewportState = view.GetViewportState();

    // This pass only works for planar, single-viewport views
    assert(viewportState.viewports.size() == 1);

    const nvrhi::Viewport& inputViewport = viewportState.viewports[0];

    TemporalAntiAliasingConstants taaConstants = {};
    affine3 viewReprojection = inverse(view.GetViewMatrix()) * translation(-preViewTranslationDifference) * viewPrevious.GetViewMatrix();
    taaConstants.reprojectionMatrix = inverse(view.GetProjectionMatrix(false)) * affineToHomogeneous(viewReprojection) * viewPrevious.GetProjectionMatrix(false);
    taaConstants.inputViewOrigin = float2(inputViewport.minX, inputViewport.minY);
    taaConstants.inputViewSize = float2(inputViewport.width(), inputViewport.height());
    taaConstants.stencilMask = m_StencilMask;
    commandList->writeBuffer(m_TemporalAntiAliasingCB, &taaConstants, sizeof(taaConstants));

    nvrhi::GraphicsState state;
    state.pipeline = pipeline;
    state.framebuffer = framebuffer;
    state.bindings = { bindingSet };
    state.viewport = viewportState;
    commandList->setGraphicsState(state);

    nvrhi::DrawArguments args;
    args.instanceCount = 1;
    args.vertexCount = 4;
    commandList->draw(args);
}

void TemporalAntiAliasingPass::TemporalResolve(
    nvrhi::ICommandList* commandList,
    const TemporalAntiAliasingParameters& params,
    bool feedbackIsValid,
    const ICompositeView& compositeViewInput,
    const ICompositeView& compositeViewOutput)
{
    commandList->beginMarker("TemporalAA");
    ResolveViews(commandList, params, feedbackIsValid, compositeViewInput, compositeViewOutput, m_ResolveBindingSet,
                 m_ResolvedColorSize, m_HasHistoryClampRelaxTexture);
    commandList->endMarker();
}

void TemporalAntiAliasingPass::ResolveViews(nvrhi::ICommandList* commandList, const TemporalAntiAliasingParameters& params,
                                            bool feedbackIsValid, const ICompositeView& compositeViewInput,
                                            const ICompositeView& compositeViewOutput, nvrhi::IBindingSet* bindingSet,
                                            dm::float2 resolvedColorSize, bool hasHistoryClampRelax)
{
    assert(compositeViewInput.GetNumChildViews(ViewType::PLANAR) == compositeViewOutput.GetNumChildViews(ViewType::PLANAR));

    for (uint viewIndex = 0; viewIndex < compositeViewInput.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* viewInput = compositeViewInput.GetChildView(ViewType::PLANAR, viewIndex);
        const IView* viewOutput = compositeViewOutput.GetChildView(ViewType::PLANAR, viewIndex);

        const nvrhi::Viewport viewportInput = viewInput->GetViewportState().viewports[0];
        const nvrhi::Viewport viewportOutput = viewOutput->GetViewportState().viewports[0];

        TemporalAntiAliasingConstants taaConstants = {};
        taaConstants.inputViewOrigin = float2(viewportInput.minX, viewportInput.minY);
        taaConstants.inputViewSize = float2(viewportInput.width(), viewportInput.height());
        taaConstants.outputViewOrigin = float2(viewportOutput.minX, viewportOutput.minY);
        taaConstants.outputViewSize = float2(viewportOutput.width(), viewportOutput.height());
        taaConstants.inputPixelOffset = viewInput->GetPixelOffset();
        taaConstants.outputTextureSizeInv = 1.0f / resolvedColorSize;
        taaConstants.inputOverOutputViewSize = taaConstants.inputViewSize / taaConstants.outputViewSize;
        taaConstants.outputOverInputViewSize = taaConstants.outputViewSize / taaConstants.inputViewSize;
        taaConstants.clampingFactor = params.enableHistoryClamping ? params.clampingFactor : -1.f;
        taaConstants.newFrameWeight = feedbackIsValid ? params.newFrameWeight : 1.f;
        taaConstants.pqC = dm::clamp(params.maxRadiance, 1e-4f, 1e8f);
        taaConstants.invPqC = 1.f / taaConstants.pqC;
        taaConstants.useHistoryClampRelax = (params.useHistoryClampRelax && hasHistoryClampRelax) ? 1 : 0;
        commandList->writeBuffer(m_TemporalAntiAliasingCB, &taaConstants, sizeof(taaConstants));

        int2 viewportSize = int2(taaConstants.outputViewSize);
        int2 gridSize = (viewportSize + 15) / 16;

        nvrhi::ComputeState state;
        state.pipeline = m_ResolvePso;
        state.bindings = { bindingSet };
        commandList->setComputeState(state);

        commandList->dispatch(gridSize.x, gridSize.y, 1);
    }
}

void TemporalAntiAliasingPass::AdvanceFrame()
{
    m_FrameIndex++;

    std::swap(m_ResolveBindingSet, m_ResolveBindingSetPrevious);

    if (m_Jitter == TemporalAntiAliasingJitter::R2)
    {
        // Advance R2 jitter sequence
        // http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

        static const float g = 1.32471795724474602596f;
        static const float a1 = 1.0f / g;
        static const float a2 = 1.0f / (g * g);
        m_R2Jitter[0] = fmodf(m_R2Jitter[0] + a1, 1.0f);
        m_R2Jitter[1] = fmodf(m_R2Jitter[1] + a2, 1.0f);
    }
}

static float VanDerCorput(size_t base, size_t index)
{
    float ret = 0.0f;
    float denominator = float(base);
    while (index > 0)
    {
        size_t multiplier = index % base;
        ret += float(multiplier) / denominator;
        index = index / base;
        denominator *= base;
    }
    return ret;
}

dm::float2 TemporalAntiAliasingPass::GetCurrentPixelOffset()
{
    switch (m_Jitter)
    {
        default:
        case TemporalAntiAliasingJitter::MSAA:
        {
            // This is the standard 8x MSAA sample pattern, source can be found e.g. here:
            // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_standard_multisample_quality_levels
            const float2 offsets[] = {
                float2(0.0625f, -0.1875f), float2(-0.0625f, 0.1875f), float2(0.3125f, 0.0625f), float2(-0.1875f, -0.3125f),
                float2(-0.3125f, 0.3125f), float2(-0.4375f, -0.0625f), float2(0.1875f, 0.4375f), float2(0.4375f, -0.4375f)
            };

            return offsets[m_FrameIndex % 8];
        }
        case TemporalAntiAliasingJitter::Halton:
        {
            uint32_t index = (m_FrameIndex % 16) + 1;
            return float2{ VanDerCorput(2, index), VanDerCorput(3, index) } - 0.5f;
        }
        case TemporalAntiAliasingJitter::R2:
        {
            return m_R2Jitter - 0.5f;
        }
        case TemporalAntiAliasingJitter::WhiteNoise:
        {
            std::mt19937 rng(m_FrameIndex);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            return float2{ dist(rng), dist(rng) };
        }
    }
}

void donut::render::TemporalAntiAliasingPass::SetJitter(TemporalAntiAliasingJitter jitter)
{
    m_Jitter = jitter;
}

// ------------------------------------------------------------------------------------------ render graph (13.3)

TemporalAntiAliasingPass::TemporalAntiAliasingPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    CommonRenderPasses* commonPasses,
    const CreateParameters& params)
    : m_CommonPasses(commonPasses)
    , m_FrameIndex(0)
    , m_StencilMask(params.motionVectorStencilMask)
    , m_R2Jitter(0.0f, 0.0f)
    , m_Jitter(TemporalAntiAliasingJitter::MSAA)
    , m_HasHistoryClampRelaxTexture(false)
    , m_Device(device)
    , m_ShaderFactory(shaderFactory)
    , m_UseCatmullRomFilter(params.useCatmullRomFilter)
    , m_NumConstantBufferVersions(params.numConstantBufferVersions)
    , m_GraphBindings(device)
{
    static std::atomic<uint32_t> s_Instances{ 0 };
    m_FeedbackName = "donut.TemporalAntiAliasingPass.Feedback#" + std::to_string(++s_Instances);
}

namespace
{
    enum class TaaStage : uint32_t { MotionVectors, Resolve };

    struct TaaUse
    {
        TaaStage stage;
        const IView* view;                    // MotionVectors
        const IView* viewPrevious;
        const ICompositeView* input;          // Resolve
        const ICompositeView* output;
        ResourceId depth;
        ResourceId motionVectors;
        ResourceId unresolved;
        ResourceId resolved;
        ResourceId current;
        ResourceId previous;
        ResourceId relax;
        TemporalAntiAliasingParameters params;
        dm::float3 preViewTranslationDifference;
        nvrhi::SubresourceRegion region;
    };

    nvrhi::Format StencilViewFormat(nvrhi::Format depthFormat)
    {
        if (depthFormat == nvrhi::Format::D24S8)
            return nvrhi::Format::X24G8_UINT;
        if (depthFormat == nvrhi::Format::D32S8)
            return nvrhi::Format::X32G8_UINT;
        return nvrhi::Format::UNKNOWN;
    }
}

TemporalAntiAliasingGraphOutputs TemporalAntiAliasingPass::AddToGraph(IRenderGraph* graph, const TemporalAntiAliasingGraphInputs& inputs)
{
    TemporalAntiAliasingGraphOutputs outputs{ inputs.motionVectors, inputs.resolvedColor };
    nvrhi::TextureDesc resolvedDesc;
    if (!inputs.view || inputs.motionVectors == InvalidResourceId || inputs.unresolvedColor == InvalidResourceId ||
        graph->GetTextureDesc(inputs.resolvedColor, resolvedDesc) != nvrhi::FS_OK)
        return outputs;
    TemporalAntiAliasingGraphInputs resolve = inputs;
    resolve.motionVectors = outputs.motionVectors = AddMotionVectorsToGraph(graph, inputs);
    outputs.resolvedColor = AddResolveToGraph(graph, resolve);
    return outputs;
}

ResourceId TemporalAntiAliasingPass::AddMotionVectorsToGraph(IRenderGraph* graph, const TemporalAntiAliasingGraphInputs& inputs)
{
    TemporalAntiAliasingGraphOutputs outputs{ inputs.motionVectors, inputs.resolvedColor };
    if (inputs.view && inputs.motionVectors != InvalidResourceId && inputs.viewPrevious && inputs.depth != InvalidResourceId)
    {
        const uint32_t viewCount = inputs.view->GetNumChildViews(ViewType::PLANAR);
        for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
        {
            TaaUse* use = NewFrameData<TaaUse>(graph);
            if (!use)
                return outputs.motionVectors;
            const IView* view = inputs.view->GetChildView(ViewType::PLANAR, viewIndex);
            const nvrhi::TextureSubresourceSet subresources = view->GetSubresources();
            *use = TaaUse{ TaaStage::MotionVectors, view, inputs.viewPrevious->GetChildView(ViewType::PLANAR, viewIndex),
                           nullptr, nullptr, inputs.depth, outputs.motionVectors, InvalidResourceId, InvalidResourceId,
                           InvalidResourceId, InvalidResourceId, InvalidResourceId, inputs.params,
                           inputs.preViewTranslationDifference,
                           nvrhi::SubresourceRegion().setMips(subresources.baseMipLevel, 1)
                               .setSlices(subresources.baseArraySlice, subresources.numArraySlices) };
            graph->AddPass(PassDesc{ "MotionVectors", PassType::Raster, PassFlags::None, 0 }, this, use, nullptr);
            outputs.motionVectors = use->motionVectors;
        }
    }
    return outputs.motionVectors;
}

ResourceId TemporalAntiAliasingPass::AddResolveToGraph(IRenderGraph* graph, const TemporalAntiAliasingGraphInputs& inputs)
{
    TemporalAntiAliasingGraphOutputs outputs{ inputs.motionVectors, inputs.resolvedColor };
    nvrhi::TextureDesc resolvedDesc;
    if (!inputs.view || inputs.motionVectors == InvalidResourceId || inputs.unresolvedColor == InvalidResourceId ||
        graph->GetTextureDesc(inputs.resolvedColor, resolvedDesc) != nvrhi::FS_OK)
        return outputs.resolvedColor;

    // The feedback: no debug name, so that the description allocates nothing; the registry names it.
    nvrhi::TextureDesc feedbackDesc;
    feedbackDesc.width = resolvedDesc.width;
    feedbackDesc.height = resolvedDesc.height;
    feedbackDesc.format = inputs.feedbackFormat;
    feedbackDesc.isUAV = true;
    ResourceId current = InvalidResourceId;
    ResourceId previous = InvalidResourceId;
    if (GetHistoryPair(graph, m_FeedbackName.c_str(), feedbackDesc, &current, &previous) != nvrhi::FS_OK)
        return outputs.resolvedColor;

    TaaUse* use = NewFrameData<TaaUse>(graph);
    if (!use)
        return outputs.resolvedColor;
    *use = TaaUse{ TaaStage::Resolve, nullptr, nullptr, inputs.view, inputs.viewOutput ? inputs.viewOutput : inputs.view,
                   InvalidResourceId, outputs.motionVectors, inputs.unresolvedColor, outputs.resolvedColor, current, previous,
                   inputs.historyClampRelax, inputs.params, inputs.preViewTranslationDifference, nvrhi::SubresourceRegion() };
    graph->AddPass(PassDesc{ "TemporalAA", PassType::Compute, PassFlags::None, 0 }, this, use, nullptr);
    return use->resolved;
}

nvrhi::FRESULT TemporalAntiAliasingPass::Setup(ISetupContext* context, void* data) noexcept
{
    TaaUse& use = *static_cast<TaaUse*>(data);
    if (use.stage == TaaStage::MotionVectors)
    {
        context->Read(use.depth, Access::ShaderRead, nvrhi::ShaderType::Pixel, nvrhi::SubresourceRegion());
        use.motionVectors = context->SetColorTarget(0, use.motionVectors, use.region, AccessFlags::FullWrite, nullptr);
        return nvrhi::FS_OK;
    }
    context->Read(use.unresolved, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
    context->Read(use.motionVectors, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
    context->Read(use.previous, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
    if (use.relax != InvalidResourceId)
        context->Read(use.relax, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
    use.resolved = context->Write(use.resolved, Access::UAVWrite, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion(),
                                  AccessFlags::FullWrite);
    use.current = context->Write(use.current, Access::UAVWrite, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion(),
                                 AccessFlags::FullWrite);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT TemporalAntiAliasingPass::Prepare(IPrepareContext* context, void* data) noexcept
{
    const TaaUse& use = *static_cast<const TaaUse*>(data);
    if (!m_Device || !m_ShaderFactory)
        return nvrhi::FE_INVALID_ARGS;   // the immediate construction has no graph path
    try
    {
        if (!m_TemporalAntiAliasingCB)
        {
            nvrhi::SamplerDesc samplerDesc;
            samplerDesc.addressU = samplerDesc.addressV = samplerDesc.addressW = nvrhi::SamplerAddressMode::Border;
            samplerDesc.borderColor = nvrhi::Color(0.0f);
            m_Device->createSampler(samplerDesc, &m_BilinearSampler);

            nvrhi::BufferDesc constantBufferDesc;
            constantBufferDesc.byteSize = sizeof(TemporalAntiAliasingConstants);
            constantBufferDesc.debugName = "TemporalAntiAliasingConstants";
            constantBufferDesc.isConstantBuffer = true;
            constantBufferDesc.isVolatile = true;
            constantBufferDesc.maxVersions = m_NumConstantBufferVersions;
            m_Device->createBuffer(constantBufferDesc, &m_TemporalAntiAliasingCB);
        }

        if (use.stage == TaaStage::MotionVectors)
        {
            const nvrhi::TextureDesc* depthDesc = context->GetTextureDesc(use.depth);
            const bool useStencil = m_StencilMask && depthDesc && StencilViewFormat(depthDesc->format) != nvrhi::Format::UNKNOWN;
            if (!m_MotionVectorPS || m_GraphUseStencil != int(useStencil))
            {
                std::vector<ShaderMacro> macros;
                macros.push_back(ShaderMacro("USE_STENCIL", useStencil ? "1" : "0"));
                m_MotionVectorPS = m_ShaderFactory->CreateAutoShader("donut/passes/motion_vectors_ps.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_motion_vectors_ps), &macros, nvrhi::ShaderType::Pixel);

                nvrhi::BindingLayoutDesc layoutDesc;
                layoutDesc.visibility = nvrhi::ShaderType::Pixel;
                layoutDesc.bindings = {
                    nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
                    nvrhi::BindingLayoutItem::Texture_SRV(0)
                };
                if (useStencil)
                    layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(1));
                m_Device->createBindingLayout(layoutDesc, &m_MotionVectorsBindingLayout);
                m_GraphMotionVectorsPso = nullptr;
                m_GraphUseStencil = int(useStencil);
            }
            const nvrhi::FramebufferInfo framebufferInfo = context->GetFramebufferInfo().getInfo();
            if (!m_GraphMotionVectorsPso || m_GraphMotionVectorsPso->getFramebufferInfo() != framebufferInfo)
            {
                nvrhi::GraphicsPipelineDesc pipelineDesc;
                pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
                pipelineDesc.VS = m_CommonPasses->m_FullscreenVS;
                pipelineDesc.PS = m_MotionVectorPS;
                pipelineDesc.bindingLayouts = { m_MotionVectorsBindingLayout };
                pipelineDesc.renderState.rasterState.setCullNone();
                pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
                pipelineDesc.renderState.depthStencilState.stencilEnable = false;
                m_GraphMotionVectorsPso = nullptr;
                m_Device->createGraphicsPipeline1(pipelineDesc, framebufferInfo, &m_GraphMotionVectorsPso);
            }
            return m_GraphMotionVectorsPso ? nvrhi::FS_OK : nvrhi::FE_GENERIC_ERROR;
        }

        const nvrhi::TextureDesc* unresolvedDesc = context->GetTextureDesc(use.unresolved);
        const uint32_t sampleCount = unresolvedDesc ? unresolvedDesc->sampleCount : 1;
        if (!m_ResolvePso || m_GraphSampleCount != sampleCount)
        {
            std::vector<ShaderMacro> macros;
            macros.push_back(ShaderMacro("SAMPLE_COUNT", std::to_string(sampleCount)));
            macros.push_back(ShaderMacro("USE_CATMULL_ROM_FILTER", m_UseCatmullRomFilter ? "1" : "0"));
            m_TemporalAntiAliasingCS = m_ShaderFactory->CreateAutoShader("donut/passes/taa_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_taa_cs), &macros, nvrhi::ShaderType::Compute);

            nvrhi::BindingLayoutDesc layoutDesc;
            layoutDesc.visibility = nvrhi::ShaderType::Compute;
            layoutDesc.bindings = {
                nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
                nvrhi::BindingLayoutItem::Sampler(0),
                nvrhi::BindingLayoutItem::Texture_SRV(0),
                nvrhi::BindingLayoutItem::Texture_SRV(1),
                nvrhi::BindingLayoutItem::Texture_SRV(2),
                nvrhi::BindingLayoutItem::Texture_UAV(0),
                nvrhi::BindingLayoutItem::Texture_UAV(1),
                nvrhi::BindingLayoutItem::Texture_SRV(3)
            };
            m_ResolveBindingLayout = nullptr;
            m_Device->createBindingLayout(layoutDesc, &m_ResolveBindingLayout);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.CS = m_TemporalAntiAliasingCS;
            pipelineDesc.bindingLayouts = { m_ResolveBindingLayout };
            m_ResolvePso = nullptr;
            m_Device->createComputePipeline(pipelineDesc, &m_ResolvePso);
            m_GraphSampleCount = sampleCount;
        }
        return m_ResolvePso ? nvrhi::FS_OK : nvrhi::FE_GENERIC_ERROR;
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
}

nvrhi::FRESULT TemporalAntiAliasingPass::Execute(IPassContext* context, const void* data) noexcept
{
    const TaaUse& use = *static_cast<const TaaUse*>(data);
    nvrhi::ICommandList* commandList = context->GetCommandList();
    try
    {
        if (use.stage == TaaStage::MotionVectors)
        {
            nvrhi::ITexture* depth = context->GetTexture(use.depth);
            nvrhi::IFramebuffer* framebuffer = context->GetFramebuffer();
            if (!depth || !framebuffer || !m_GraphMotionVectorsPso)
                return nvrhi::FE_INVALID_ARGS;
            nvrhi::BindingSetDesc desc;
            desc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_TemporalAntiAliasingCB),
                nvrhi::BindingSetItem::Texture_SRV(0, depth)
            };
            if (m_GraphUseStencil == 1)
                desc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(1, depth, StencilViewFormat(depth->getDesc().format)));
            nvrhi::BindingSetHandle bindingSet = m_GraphBindings.GetOrCreateBindingSet(desc, m_MotionVectorsBindingLayout);
            DrawMotionVectors(commandList, *use.view, *use.viewPrevious, framebuffer, m_GraphMotionVectorsPso, bindingSet,
                              use.preViewTranslationDifference);
            return nvrhi::FS_OK;
        }

        nvrhi::ITexture* unresolved = context->GetTexture(use.unresolved);
        nvrhi::ITexture* resolved = context->GetTexture(use.resolved);
        nvrhi::ITexture* relax = context->GetTexture(use.relax);
        if (!unresolved || !resolved || !m_ResolvePso)
            return nvrhi::FE_INVALID_ARGS;
        nvrhi::BindingSetDesc desc;
        desc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_TemporalAntiAliasingCB),
            nvrhi::BindingSetItem::Sampler(0, m_BilinearSampler),
            nvrhi::BindingSetItem::Texture_SRV(0, unresolved),
            nvrhi::BindingSetItem::Texture_SRV(1, context->GetTexture(use.motionVectors)),
            nvrhi::BindingSetItem::Texture_SRV(2, context->GetTexture(use.previous)),
            nvrhi::BindingSetItem::Texture_UAV(0, resolved),
            nvrhi::BindingSetItem::Texture_UAV(1, context->GetTexture(use.current)),
            nvrhi::BindingSetItem::Texture_SRV(3, relax ? relax : unresolved)
        };
        nvrhi::BindingSetHandle bindingSet = m_GraphBindings.GetOrCreateBindingSet(desc, m_ResolveBindingLayout);
        const nvrhi::TextureDesc& resolvedDesc = resolved->getDesc();
        // 13.3: the history is valid unless the previous layer was allocated this frame.
        const bool feedbackIsValid = !context->IsNewAllocation(use.previous);
        ResolveViews(commandList, use.params, feedbackIsValid, *use.input, *use.output, bindingSet,
                     float2(float(resolvedDesc.width), float(resolvedDesc.height)), relax != nullptr);
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

void TemporalAntiAliasingPass::OnResourcesReleased() noexcept
{
    m_GraphBindings.Clear();
}
