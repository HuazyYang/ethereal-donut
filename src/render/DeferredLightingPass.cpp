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

#include <donut/render/DeferredLightingPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/GBuffer.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include <utility>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/deferred_lighting_cs.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/deferred_lighting_cs.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/deferred_lighting_cs.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/deferred_lighting_cb.h>

using namespace donut::engine;
using namespace donut::render;

void DeferredLightingPass::Inputs::SetGBuffer(const GBufferRenderTargets& targets)
{
    depth = targets.Depth;
    gbufferNormals = targets.GBufferNormals;
    gbufferDiffuse = targets.GBufferDiffuse;
    gbufferSpecular = targets.GBufferSpecular;
    gbufferEmissive = targets.GBufferEmissive;
}

DeferredLightingPass::DeferredLightingPass(
    nvrhi::IDevice* device,
    CommonRenderPasses* commonPasses)
    : m_Device(device)
    , m_BindingSets(device)
    , m_CommonPasses(commonPasses)
{
}

void donut::render::DeferredLightingPass::Init(engine::ShaderFactory* shaderFactory)
{
    auto samplerDesc = nvrhi::SamplerDesc()
        .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
        .setBorderColor(1.0f);
    m_Device->createSampler(samplerDesc, &m_ShadowSampler);

    samplerDesc.setReductionType(nvrhi::SamplerReductionType::Comparison);
    m_Device->createSampler(samplerDesc, &m_ShadowSamplerComparison);

    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(DeferredLightingConstants);
    constantBufferDesc.debugName = "DeferredLightingConstants";
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.maxVersions = c_MaxRenderPassConstantBufferVersions;
    m_Device->createBuffer(constantBufferDesc, &m_DeferredLightingCB);
    
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::Texture_SRV(2),
            nvrhi::BindingLayoutItem::Texture_SRV(3),
            nvrhi::BindingLayoutItem::Texture_SRV(8),
            nvrhi::BindingLayoutItem::Texture_SRV(9),
            nvrhi::BindingLayoutItem::Texture_SRV(10),
            nvrhi::BindingLayoutItem::Texture_SRV(11),
            nvrhi::BindingLayoutItem::Texture_SRV(12),
            nvrhi::BindingLayoutItem::Texture_SRV(14),
            nvrhi::BindingLayoutItem::Texture_SRV(15),
            nvrhi::BindingLayoutItem::Texture_SRV(16),
            nvrhi::BindingLayoutItem::Texture_SRV(17),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::Sampler(1),
            nvrhi::BindingLayoutItem::Sampler(2),
            nvrhi::BindingLayoutItem::Sampler(3)
        };
        m_Device->createBindingLayout(layoutDesc, &m_BindingLayout);
        
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.CS = CreateComputeShader(*shaderFactory);
        pipelineDesc.bindingLayouts = { m_BindingLayout };
        
        m_Device->createComputePipeline(pipelineDesc, &m_Pso);
    }
}

nvrhi::ShaderHandle DeferredLightingPass::CreateComputeShader(ShaderFactory& shaderFactory)
{
    return shaderFactory.CreateAutoShader("donut/passes/deferred_lighting_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_deferred_lighting_cs), nullptr, nvrhi::ShaderType::Compute);
}

void DeferredLightingPass::Render(
    nvrhi::ICommandList* commandList,
    const ICompositeView& compositeView,
    const Inputs& inputs,
    dm::float2 randomOffset)
{
    assert(inputs.depth);
    assert(inputs.gbufferNormals);
    assert(inputs.gbufferDiffuse);
    assert(inputs.gbufferSpecular);
    assert(inputs.gbufferEmissive);
    assert(inputs.output);

    commandList->beginMarker("DeferredLighting");

    DeferredLightingConstants deferredConstants = {};
    deferredConstants.randomOffset = randomOffset;
    deferredConstants.noisePattern[0] = float4(0.059f, 0.529f, 0.176f, 0.647f);
    deferredConstants.noisePattern[1] = float4(0.765f, 0.294f, 0.882f, 0.412f);
    deferredConstants.noisePattern[2] = float4(0.235f, 0.706f, 0.118f, 0.588f);
    deferredConstants.noisePattern[3] = float4(0.941f, 0.471f, 0.824f, 0.353f);
    deferredConstants.ambientColorTop = float4(inputs.ambientColorTop, 0.f);
    deferredConstants.ambientColorBottom = float4(inputs.ambientColorBottom, 0.f);
    deferredConstants.enableAmbientOcclusion = (inputs.ambientOcclusion != nullptr);
    deferredConstants.indirectDiffuseScale = 1.f;
    deferredConstants.indirectSpecularScale = 1.f;

    nvrhi::ITexture* shadowMapTexture = nullptr;

    int numShadows = 0;

    if (inputs.lights)
    {
        for (const auto& light : *inputs.lights)
        {
            if (light->shadowMap)
            {
                if (!shadowMapTexture)
                {
                    shadowMapTexture = light->shadowMap->GetTexture();
                    deferredConstants.shadowMapTextureSize = float2(light->shadowMap->GetTextureSize());
                }
                else
                {
                    if (shadowMapTexture != light->shadowMap->GetTexture())
                    {
                        log::error("All lights submitted to DeferredLightingPass::Render(...) must use the same shadow map textures");
                        return;
                    }
                }
            }

            if (deferredConstants.numLights >= DEFERRED_MAX_LIGHTS)
            {
                log::warning("Maximum number of active lights (%d) exceeded in DeferredLightingPass",
                    DEFERRED_MAX_LIGHTS);
                break;
            }

            LightConstants& lightConstants = deferredConstants.lights[deferredConstants.numLights];
            light->FillLightConstants(lightConstants);

            if (light->shadowMap)
            {
                for (uint32_t cascade = 0; cascade < light->shadowMap->GetNumberOfCascades(); cascade++)
                {
                    if (numShadows < DEFERRED_MAX_SHADOWS)
                    {
                        light->shadowMap->GetCascade(cascade)->FillShadowConstants(deferredConstants.shadows[numShadows]);
                        lightConstants.shadowCascades[cascade] = numShadows;
                        ++numShadows;
                    }
                }

                for (uint32_t perObjectShadow = 0; perObjectShadow < light->shadowMap->GetNumberOfPerObjectShadows(); perObjectShadow++)
                {
                    if (numShadows < DEFERRED_MAX_SHADOWS)
                    {
                        light->shadowMap->GetPerObjectShadow(perObjectShadow)->FillShadowConstants(deferredConstants.shadows[numShadows]);
                        lightConstants.perObjectShadows[perObjectShadow] = numShadows;
                        ++numShadows;
                    }
                }
            }

            ++deferredConstants.numLights;
        }
    }

    nvrhi::ITexture* lightProbeDiffuse = nullptr;
    nvrhi::ITexture* lightProbeSpecular = nullptr;
    nvrhi::ITexture* lightProbeEnvironmentBrdf = nullptr;

    if (inputs.lightProbes)
    {
        for (const auto& probe : *inputs.lightProbes)
        {
            if (!probe->IsActive())
                continue;

            LightProbeConstants& lightProbeConstants = deferredConstants.lightProbes[deferredConstants.numLightProbes];
            probe->FillLightProbeConstants(lightProbeConstants);

            ++deferredConstants.numLightProbes;

            if (deferredConstants.numLightProbes >= DEFERRED_MAX_LIGHT_PROBES)
            {
                log::warning("Maximum number of active light probes (%d) exceeded in DeferredLightingPass",
                    DEFERRED_MAX_LIGHT_PROBES);
                break;
            }

            if (lightProbeDiffuse == nullptr || lightProbeSpecular == nullptr || lightProbeEnvironmentBrdf == nullptr)
            {
                lightProbeDiffuse = probe->diffuseMap;
                lightProbeSpecular = probe->specularMap;
                lightProbeEnvironmentBrdf = probe->environmentBrdf;
            }
            else
            {
                if (lightProbeDiffuse != probe->diffuseMap || lightProbeSpecular != probe->specularMap || lightProbeEnvironmentBrdf != probe->environmentBrdf)
                {
                    log::error("All light probes submitted to DeferredLightingPass::Render(...) must use the same set of textures");
                    return;
                }
            }
        }
    }


    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);
        auto viewSubresources = view->GetSubresources();

        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_DeferredLightingCB),
            nvrhi::BindingSetItem::Texture_SRV(0, shadowMapTexture ? shadowMapTexture : m_CommonPasses->m_BlackDepthStencilTexture2DArray.Get()),
            nvrhi::BindingSetItem::Texture_SRV(1, lightProbeDiffuse ? lightProbeDiffuse : m_CommonPasses->m_BlackCubeMapArray.Get()),
            nvrhi::BindingSetItem::Texture_SRV(2, lightProbeSpecular ? lightProbeSpecular : m_CommonPasses->m_BlackCubeMapArray.Get()),
            nvrhi::BindingSetItem::Texture_SRV(3, lightProbeEnvironmentBrdf ? lightProbeEnvironmentBrdf : m_CommonPasses->m_BlackTexture.Get()),
            nvrhi::BindingSetItem::Texture_SRV(8, inputs.depth, nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_SRV(9, inputs.gbufferDiffuse, nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_SRV(10, inputs.gbufferSpecular, nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_SRV(11, inputs.gbufferNormals, nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_SRV(12, inputs.gbufferEmissive, nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_SRV(14, inputs.indirectDiffuse ? inputs.indirectDiffuse : m_CommonPasses->m_BlackTexture.Get(), nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_SRV(15, inputs.indirectSpecular ? inputs.indirectSpecular : m_CommonPasses->m_BlackTexture.Get(), nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_SRV(16, inputs.shadowChannels ? inputs.shadowChannels : m_CommonPasses->m_BlackTexture.Get()),
            nvrhi::BindingSetItem::Texture_SRV(17, inputs.ambientOcclusion ? inputs.ambientOcclusion : m_CommonPasses->m_WhiteTexture.Get(), nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Texture_UAV(0, inputs.output, nvrhi::Format::UNKNOWN, viewSubresources),
            nvrhi::BindingSetItem::Sampler(0, m_ShadowSampler),
            nvrhi::BindingSetItem::Sampler(1, m_ShadowSamplerComparison),
            nvrhi::BindingSetItem::Sampler(2, m_CommonPasses->m_LinearWrapSampler),
            nvrhi::BindingSetItem::Sampler(3, m_CommonPasses->m_LinearClampSampler)
        };

        nvrhi::BindingSetHandle bindingSet = m_BindingSets.GetOrCreateBindingSet(bindingSetDesc, m_BindingLayout);
    
        view->FillPlanarViewConstants(deferredConstants.view);
        commandList->writeBuffer(m_DeferredLightingCB, &deferredConstants, sizeof(deferredConstants));

        nvrhi::ComputeState state;
        state.pipeline = m_Pso;
        state.bindings = { bindingSet };
        commandList->setComputeState(state);

        auto viewExtent = view->GetViewExtent();
        commandList->dispatch(
            dm::div_ceil(viewExtent.width(), 16),
            dm::div_ceil(viewExtent.height(), 16));
    }

    commandList->endMarker();
}

void DeferredLightingPass::ResetBindingCache()
{
    m_BindingSets.Clear();
}


// ------------------------------------------------------------------------------------------ render graph (13.2)

namespace
{
    struct DeferredLightingUse
    {
        DeferredLightingGraphInputs inputs;
    };
}

DeferredLightingGraphOutputs DeferredLightingPass::AddToGraph(IRenderGraph* graph, const DeferredLightingGraphInputs& inputs)
{
    DeferredLightingUse* use = NewFrameData<DeferredLightingUse>(graph, inputs);
    if (!use || graph->AddPass(PassDesc{ "DeferredLighting", PassType::Compute, PassFlags::None, 0 }, this, use, nullptr) != nvrhi::FS_OK)
        return DeferredLightingGraphOutputs{ InvalidResourceId };
    return DeferredLightingGraphOutputs{ use->inputs.output };
}

nvrhi::FRESULT DeferredLightingPass::Setup(ISetupContext* context, void* data) noexcept
{
    DeferredLightingGraphInputs& inputs = static_cast<DeferredLightingUse*>(data)->inputs;
    if (!inputs.view)
        return nvrhi::FE_INVALID_ARGS;
    const ResourceId reads[] = { inputs.depth, inputs.gbufferNormals, inputs.gbufferDiffuse, inputs.gbufferSpecular,
                                 inputs.gbufferEmissive, inputs.indirectDiffuse, inputs.indirectSpecular,
                                 inputs.shadowChannels, inputs.ambientOcclusion, inputs.shadowMap };
    for (ResourceId id : reads)
    {
        if (id != InvalidResourceId)
            context->Read(id, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
    }
    inputs.output = context->Write(inputs.output, Access::UAVWrite, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion(),
                                   AccessFlags::FullWrite);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT DeferredLightingPass::Prepare(IPrepareContext*, void*) noexcept
{
    return m_Pso ? nvrhi::FS_OK : nvrhi::FE_INVALID_ARGS;   // Init has not run
}

nvrhi::FRESULT DeferredLightingPass::Execute(IPassContext* context, const void* data) noexcept
{
    const DeferredLightingGraphInputs& use = static_cast<const DeferredLightingUse*>(data)->inputs;
    Inputs inputs;
    inputs.depth = context->GetTexture(use.depth);
    inputs.gbufferNormals = context->GetTexture(use.gbufferNormals);
    inputs.gbufferDiffuse = context->GetTexture(use.gbufferDiffuse);
    inputs.gbufferSpecular = context->GetTexture(use.gbufferSpecular);
    inputs.gbufferEmissive = context->GetTexture(use.gbufferEmissive);
    inputs.indirectDiffuse = context->GetTexture(use.indirectDiffuse);
    inputs.indirectSpecular = context->GetTexture(use.indirectSpecular);
    inputs.shadowChannels = context->GetTexture(use.shadowChannels);
    inputs.ambientOcclusion = context->GetTexture(use.ambientOcclusion);
    inputs.output = context->GetTexture(use.output);
    inputs.lights = use.lights;
    inputs.lightProbes = use.lightProbes;
    inputs.ambientColorTop = use.ambientColorTop;
    inputs.ambientColorBottom = use.ambientColorBottom;
    if (!inputs.depth || !inputs.gbufferNormals || !inputs.gbufferDiffuse || !inputs.gbufferSpecular ||
        !inputs.gbufferEmissive || !inputs.output)
        return nvrhi::FE_INVALID_ARGS;
    try
    {
        Render(context->GetCommandList(), *use.view, inputs, use.randomOffset);
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

void DeferredLightingPass::OnResourcesReleased() noexcept
{
    ResetBindingCache();
}
