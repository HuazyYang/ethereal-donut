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

#include <donut/render/SsaoPass.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/render/DrawStrategy.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/ssao_blur_cs.dxbc.h"
#include "compiled_shaders/passes/ssao_compute_cs.dxbc.h"
#include "compiled_shaders/passes/ssao_deinterleave_cs.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/ssao_blur_cs.dxil.h"
#include "compiled_shaders/passes/ssao_compute_cs.dxil.h"
#include "compiled_shaders/passes/ssao_deinterleave_cs.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/ssao_blur_cs.spirv.h"
#include "compiled_shaders/passes/ssao_compute_cs.spirv.h"
#include "compiled_shaders/passes/ssao_deinterleave_cs.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/ssao_cb.h>

#include <sstream>
#include <assert.h>

using namespace donut::engine;
using namespace donut::render;


// The three compute stages of one shader variant: shaders, binding layouts and pipelines.
void SsaoPass::CreatePipelines(ShaderFactory* shaderFactory, bool inputLinearDepth, bool octEncodedNormals,
                               bool directionalOcclusion)
{
    {
        std::vector<engine::ShaderMacro> macros = {
            { "LINEAR_DEPTH", inputLinearDepth ? "1" : "0" }
        };
        m_Deinterleave.Shader = shaderFactory->CreateAutoShader("donut/passes/ssao_deinterleave_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_ssao_deinterleave_cs), &macros, nvrhi::ShaderType::Compute);

        nvrhi::BindingLayoutDesc DeinterleaveBindings;
        DeinterleaveBindings.visibility = nvrhi::ShaderType::Compute;
        DeinterleaveBindings.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
        };
        m_Deinterleave.BindingLayout = nullptr;
        m_Device->createBindingLayout(DeinterleaveBindings, &m_Deinterleave.BindingLayout);

        nvrhi::ComputePipelineDesc DeinterleavePipelineDesc;
        DeinterleavePipelineDesc.bindingLayouts = { m_Deinterleave.BindingLayout };
        DeinterleavePipelineDesc.CS = m_Deinterleave.Shader;
        m_Deinterleave.Pipeline = nullptr;
        m_Device->createComputePipeline(DeinterleavePipelineDesc, &m_Deinterleave.Pipeline);
    }

    {
        std::vector<engine::ShaderMacro> macros = {
            { "OCT_ENCODED_NORMALS", octEncodedNormals ? "1" : "0" },
            { "DIRECTIONAL_OCCLUSION", directionalOcclusion ? "1" : "0" }
        };
        m_Compute.Shader = shaderFactory->CreateAutoShader("donut/passes/ssao_compute_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_ssao_compute_cs), &macros, nvrhi::ShaderType::Compute);

        nvrhi::BindingLayoutDesc ComputeBindings;
        ComputeBindings.visibility = nvrhi::ShaderType::Compute;
        ComputeBindings.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
        };
        m_Compute.BindingLayout = nullptr;
        m_Device->createBindingLayout(ComputeBindings, &m_Compute.BindingLayout);

        nvrhi::ComputePipelineDesc ComputePipeline;
        ComputePipeline.bindingLayouts = { m_Compute.BindingLayout };
        ComputePipeline.CS = m_Compute.Shader;
        m_Compute.Pipeline = nullptr;
        m_Device->createComputePipeline(ComputePipeline, &m_Compute.Pipeline);
    }

    {
        std::vector<engine::ShaderMacro> macros = {
            { "DIRECTIONAL_OCCLUSION", directionalOcclusion ? "1" : "0" }
        };
        m_Blur.Shader = shaderFactory->CreateAutoShader("donut/passes/ssao_blur_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_ssao_blur_cs), &macros, nvrhi::ShaderType::Compute);

        nvrhi::BindingLayoutDesc BlurBindings;
        BlurBindings.visibility = nvrhi::ShaderType::Compute;
        BlurBindings.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
        };
        m_Blur.BindingLayout = nullptr;
        m_Device->createBindingLayout(BlurBindings, &m_Blur.BindingLayout);

        nvrhi::ComputePipelineDesc BlurPipeline;
        BlurPipeline.bindingLayouts = { m_Blur.BindingLayout };
        BlurPipeline.CS = m_Blur.Shader;
        m_Blur.Pipeline = nullptr;
        m_Device->createComputePipeline(BlurPipeline, &m_Blur.Pipeline);
    }
}

SsaoPass::SsaoPass(
    nvrhi::IDevice* device,
    ShaderFactory *shaderFactory,
    CommonRenderPasses *commonPasses,
    const CreateParameters& params)
    : m_Device(device)
    , m_CommonPasses(commonPasses)
    , m_GraphBindings(device)
{
    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(SsaoConstants);
    constantBufferDesc.debugName = "SsaoConstants";
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
    device->createBuffer(constantBufferDesc, &m_ConstantBuffer);

    nvrhi::TextureDesc DeinterleavedTextureDesc;
    DeinterleavedTextureDesc.width = (params.dimensions.x + 3) / 4;
    DeinterleavedTextureDesc.height = (params.dimensions.y + 3) / 4;
    DeinterleavedTextureDesc.arraySize = 16;
    DeinterleavedTextureDesc.dimension = nvrhi::TextureDimension::Texture2DArray;
    DeinterleavedTextureDesc.isUAV = true;
    DeinterleavedTextureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    DeinterleavedTextureDesc.keepInitialState = true;
    DeinterleavedTextureDesc.debugName = "SSAO/DeinterleavedDepth";
    DeinterleavedTextureDesc.format = nvrhi::Format::R32_FLOAT;
    device->createTexture(DeinterleavedTextureDesc, &m_DeinterleavedDepth);

    m_QuantizedGbufferTextureSize = float2(float(DeinterleavedTextureDesc.width), float(DeinterleavedTextureDesc.height)) * 4.f;

    DeinterleavedTextureDesc.debugName = "SSAO/DeinterleavedOcclusion";
    DeinterleavedTextureDesc.format = params.directionalOcclusion ? nvrhi::Format::RGBA16_FLOAT : nvrhi::Format::R8_UNORM;
    device->createTexture(DeinterleavedTextureDesc, &m_DeinterleavedOcclusion);

    CreatePipelines(shaderFactory, params.inputLinearDepth, params.octEncodedNormals, params.directionalOcclusion);

    m_Deinterleave.BindingSets.resize(params.numBindingSets);
    m_Compute.BindingSets.resize(params.numBindingSets);
    m_Blur.BindingSets.resize(params.numBindingSets);
}

// Backwards compatibility constructor
SsaoPass::SsaoPass(
    nvrhi::IDevice* device,
    ShaderFactory *shaderFactory,
    CommonRenderPasses* commonPasses,
    nvrhi::ITexture* gbufferDepth,
    nvrhi::ITexture* gbufferNormals,
    nvrhi::ITexture* destinationTexture)
    : SsaoPass(device, shaderFactory, commonPasses, CreateParameters{ dm::int2(gbufferDepth->getDesc().width, gbufferDepth->getDesc().height), false, false, false, 1 })
{
#ifdef _DEBUG
    const nvrhi::TextureDesc& depthDesc = gbufferDepth->getDesc();
    const nvrhi::TextureDesc& normalsDesc = gbufferNormals->getDesc();
    assert(depthDesc.sampleCount == normalsDesc.sampleCount);
    assert(depthDesc.sampleCount == 1); // more is currently unsupported
    assert(depthDesc.dimension == nvrhi::TextureDimension::Texture2D); // arrays are currently unsupported
#endif

    CreateBindingSet(gbufferDepth, gbufferNormals, destinationTexture, 0);
}

void SsaoPass::CreateBindingSet(
    nvrhi::ITexture* gbufferDepth,
    nvrhi::ITexture* gbufferNormals,
    nvrhi::ITexture* destinationTexture,
    int bindingSetIndex)
{
    nvrhi::BindingSetDesc DeinterleaveBindings;
    DeinterleaveBindings.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
        nvrhi::BindingSetItem::Texture_SRV(0, gbufferDepth),
        nvrhi::BindingSetItem::Texture_UAV(0, m_DeinterleavedDepth)
    };
    m_Device->createBindingSet(DeinterleaveBindings, m_Deinterleave.BindingLayout, &m_Deinterleave.BindingSets[bindingSetIndex]);
    
    nvrhi::BindingSetDesc ComputeBindings;
    ComputeBindings.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
        nvrhi::BindingSetItem::Texture_SRV(0, m_DeinterleavedDepth),
        nvrhi::BindingSetItem::Texture_SRV(1, gbufferNormals),
        nvrhi::BindingSetItem::Texture_UAV(0, m_DeinterleavedOcclusion)
    };
    m_Device->createBindingSet(ComputeBindings, m_Compute.BindingLayout, &m_Compute.BindingSets[bindingSetIndex]);

    nvrhi::BindingSetDesc BlurBindings;
    BlurBindings.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
        nvrhi::BindingSetItem::Texture_SRV(0, m_DeinterleavedDepth),
        nvrhi::BindingSetItem::Texture_SRV(1, m_DeinterleavedOcclusion),
        nvrhi::BindingSetItem::Texture_UAV(0, destinationTexture),
        nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_PointClampSampler)
    };
    m_Device->createBindingSet(BlurBindings, m_Blur.BindingLayout, &m_Blur.BindingSets[bindingSetIndex]);
}

void SsaoPass::Render(
    nvrhi::ICommandList* commandList,
    const SsaoParameters& params,
    const ICompositeView& compositeView,
    int bindingSetIndex)
{
    assert(m_Deinterleave.BindingSets[bindingSetIndex]);
    assert(m_Compute.BindingSets[bindingSetIndex]);
    assert(m_Blur.BindingSets[bindingSetIndex]);

    commandList->beginMarker("SSAO");
    DispatchViews(commandList, params, compositeView, m_Deinterleave.BindingSets[bindingSetIndex],
                  m_Compute.BindingSets[bindingSetIndex], m_Blur.BindingSets[bindingSetIndex], m_QuantizedGbufferTextureSize);
    commandList->endMarker();
}

void SsaoPass::DispatchViews(nvrhi::ICommandList* commandList, const SsaoParameters& params, const ICompositeView& compositeView,
                             nvrhi::IBindingSet* deinterleave, nvrhi::IBindingSet* compute, nvrhi::IBindingSet* blur,
                             dm::float2 quantizedGbufferTextureSize)
{
    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);

        const float4x4 projectionMatrix = view->GetProjectionMatrix(false);

        nvrhi::Rect viewExtent = view->GetViewExtent();
        nvrhi::Rect quarterResExtent = viewExtent;
        quarterResExtent.minX /= 4;
        quarterResExtent.minY /= 4;
        quarterResExtent.maxX = (quarterResExtent.maxX + 3) / 4;
        quarterResExtent.maxY = (quarterResExtent.maxY + 3) / 4;

        SsaoConstants ssaoConstants = {};
        view->FillPlanarViewConstants(ssaoConstants.view);

        ssaoConstants.clipToView = float2(
            projectionMatrix[2].w / projectionMatrix[0].x,
            projectionMatrix[2].w / projectionMatrix[1].y);
        ssaoConstants.invQuantizedGbufferSize = 1.f / quantizedGbufferTextureSize;
        ssaoConstants.quantizedViewportOrigin = int2(quarterResExtent.minX, quarterResExtent.minY) * 4;
        ssaoConstants.amount = params.amount;
        ssaoConstants.invBackgroundViewDepth = (params.backgroundViewDepth > 0.f) ? 1.f / params.backgroundViewDepth : 0.f;
        ssaoConstants.radiusWorld = params.radiusWorld;
        ssaoConstants.surfaceBias = params.surfaceBias;
        ssaoConstants.powerExponent = params.powerExponent;
        ssaoConstants.radiusToScreen = 0.5f * ssaoConstants.view.viewportSize.y * abs(projectionMatrix[1].y);
        commandList->writeBuffer(m_ConstantBuffer, &ssaoConstants, sizeof(ssaoConstants));

        uint32_t dispatchWidth = (quarterResExtent.width() + 7) / 8;
        uint32_t dispatchHeight = (quarterResExtent.height() + 7) / 8;

        nvrhi::ComputeState state;
        state.pipeline = m_Deinterleave.Pipeline;
        state.bindings = { deinterleave };
        commandList->setComputeState(state);
        commandList->dispatch(dispatchWidth, dispatchHeight, 1);

        state.pipeline = m_Compute.Pipeline;
        state.bindings = { compute };
        commandList->setComputeState(state);
        commandList->dispatch(dispatchWidth, dispatchHeight, 16);

        dispatchWidth = (viewExtent.width() + 15) / 16;
        dispatchHeight = (viewExtent.height() + 15) / 16;

        state.pipeline = m_Blur.Pipeline;
        state.bindings = { blur };
        commandList->setComputeState(state);
        commandList->dispatch(dispatchWidth, dispatchHeight, 1);
    }
}

// ------------------------------------------------------------------------------------------ render graph (13.3)

SsaoPass::SsaoPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    CommonRenderPasses* commonPasses)
    : m_Device(device)
    , m_CommonPasses(commonPasses)
    , m_ShaderFactory(shaderFactory)
    , m_GraphBindings(device)
{
}

namespace
{
    struct SsaoUse
    {
        const ICompositeView* view;
        ResourceId depth;
        ResourceId normals;
        ResourceId output;
        ResourceId deinterleavedDepth;
        ResourceId deinterleavedOcclusion;
        SsaoParameters params;
        uint32_t quarterWidth;
        uint32_t quarterHeight;
        bool inputLinearDepth;
        bool octEncodedNormals;
        bool directionalOcclusion;
    };
}

SsaoGraphOutputs SsaoPass::AddToGraph(IRenderGraph* graph, const SsaoGraphInputs& inputs)
{
    SsaoGraphOutputs outputs{ inputs.output };
    nvrhi::TextureDesc depthDesc;
    if (!inputs.view || inputs.normals == InvalidResourceId || inputs.output == InvalidResourceId ||
        graph->GetTextureDesc(inputs.depth, depthDesc) != nvrhi::FS_OK)
        return outputs;
    SsaoUse* use = NewFrameData<SsaoUse>(graph);
    if (!use)
        return outputs;
    *use = SsaoUse{ inputs.view, inputs.depth, inputs.normals, inputs.output, InvalidResourceId, InvalidResourceId,
                    inputs.params, (depthDesc.width + 3) / 4, (depthDesc.height + 3) / 4, inputs.inputLinearDepth,
                    inputs.octEncodedNormals, inputs.directionalOcclusion };
    // Restore: the body moves the arrays between UAV and SRV states through the automatic barriers.
    graph->AddPass(PassDesc{ "SSAO", PassType::Compute, PassFlags::Restore, 0 }, this, use, nullptr);
    outputs.output = use->output;
    return outputs;
}

nvrhi::FRESULT SsaoPass::Setup(ISetupContext* context, void* data) noexcept
{
    SsaoUse& use = *static_cast<SsaoUse*>(data);
    nvrhi::TextureDesc desc;
    desc.width = use.quarterWidth;
    desc.height = use.quarterHeight;
    desc.arraySize = 16;
    desc.dimension = nvrhi::TextureDimension::Texture2DArray;
    desc.isUAV = true;
    desc.format = nvrhi::Format::R32_FLOAT;
    use.deinterleavedDepth = context->CreateTransientTexture(desc, ResourceFlags::None);
    desc.format = use.directionalOcclusion ? nvrhi::Format::RGBA16_FLOAT : nvrhi::Format::R8_UNORM;
    use.deinterleavedOcclusion = context->CreateTransientTexture(desc, ResourceFlags::None);

    context->Read(use.depth, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
    context->Read(use.normals, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
    use.deinterleavedDepth = context->Write(use.deinterleavedDepth, Access::UAVWrite, nvrhi::ShaderType::Compute,
                                            nvrhi::SubresourceRegion(), AccessFlags::FullWrite);
    use.deinterleavedOcclusion = context->Write(use.deinterleavedOcclusion, Access::UAVWrite, nvrhi::ShaderType::Compute,
                                                nvrhi::SubresourceRegion(), AccessFlags::FullWrite);
    use.output = context->Write(use.output, Access::UAVWrite, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion(),
                                AccessFlags::FullWrite);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT SsaoPass::Prepare(IPrepareContext*, void* data) noexcept
{
    const SsaoUse& use = *static_cast<const SsaoUse*>(data);
    if (!m_Device || !m_ShaderFactory)
        return nvrhi::FE_INVALID_ARGS;   // the immediate construction has no graph path
    try
    {
        if (!m_ConstantBuffer)
        {
            nvrhi::BufferDesc constantBufferDesc;
            constantBufferDesc.byteSize = sizeof(SsaoConstants);
            constantBufferDesc.debugName = "SsaoConstants";
            constantBufferDesc.isConstantBuffer = true;
            constantBufferDesc.isVolatile = true;
            constantBufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
            m_Device->createBuffer(constantBufferDesc, &m_ConstantBuffer);
        }
        const int variant = int(use.inputLinearDepth) | (int(use.octEncodedNormals) << 1) | (int(use.directionalOcclusion) << 2);
        if (m_GraphVariant != variant)
        {
            CreatePipelines(m_ShaderFactory, use.inputLinearDepth, use.octEncodedNormals, use.directionalOcclusion);
            m_GraphBindings.Clear();
            m_GraphVariant = variant;
        }
        return m_Deinterleave.Pipeline && m_Compute.Pipeline && m_Blur.Pipeline ? nvrhi::FS_OK : nvrhi::FE_GENERIC_ERROR;
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
}

nvrhi::FRESULT SsaoPass::Execute(IPassContext* context, const void* data) noexcept
{
    const SsaoUse& use = *static_cast<const SsaoUse*>(data);
    nvrhi::ITexture* depth = context->GetTexture(use.depth);
    nvrhi::ITexture* normals = context->GetTexture(use.normals);
    nvrhi::ITexture* output = context->GetTexture(use.output);
    nvrhi::ITexture* deinterleavedDepth = context->GetTexture(use.deinterleavedDepth);
    nvrhi::ITexture* deinterleavedOcclusion = context->GetTexture(use.deinterleavedOcclusion);
    if (!depth || !normals || !output || !deinterleavedDepth || !deinterleavedOcclusion)
        return nvrhi::FE_INVALID_ARGS;
    try
    {
        nvrhi::BindingSetDesc deinterleave;
        deinterleave.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
            nvrhi::BindingSetItem::Texture_SRV(0, depth),
            nvrhi::BindingSetItem::Texture_UAV(0, deinterleavedDepth)
        };
        nvrhi::BindingSetDesc compute;
        compute.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
            nvrhi::BindingSetItem::Texture_SRV(0, deinterleavedDepth),
            nvrhi::BindingSetItem::Texture_SRV(1, normals),
            nvrhi::BindingSetItem::Texture_UAV(0, deinterleavedOcclusion)
        };
        nvrhi::BindingSetDesc blur;
        blur.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
            nvrhi::BindingSetItem::Texture_SRV(0, deinterleavedDepth),
            nvrhi::BindingSetItem::Texture_SRV(1, deinterleavedOcclusion),
            nvrhi::BindingSetItem::Texture_UAV(0, output),
            nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_PointClampSampler)
        };
        nvrhi::BindingSetHandle deinterleaveSet = m_GraphBindings.GetOrCreateBindingSet(deinterleave, m_Deinterleave.BindingLayout);
        nvrhi::BindingSetHandle computeSet = m_GraphBindings.GetOrCreateBindingSet(compute, m_Compute.BindingLayout);
        nvrhi::BindingSetHandle blurSet = m_GraphBindings.GetOrCreateBindingSet(blur, m_Blur.BindingLayout);
        DispatchViews(context->GetCommandList(), use.params, *use.view, deinterleaveSet, computeSet, blurSet,
                      float2(float(use.quarterWidth), float(use.quarterHeight)) * 4.f);
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

void SsaoPass::OnResourcesReleased() noexcept
{
    m_GraphBindings.Clear();
}
