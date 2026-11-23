/*
* Copyright (c) 2014-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <donut/render/ForwardShadingPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/MaterialBindingCache.h>
#include <donut/core/log.h>
#include <nvrhi/utils.h>
#include <utility>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/cubemap_gs.dxbc.h"
#include "compiled_shaders/passes/forward_ps.dxbc.h"
#include "compiled_shaders/passes/forward_vs_input_assembler.dxbc.h"
#include "compiled_shaders/passes/forward_vs_buffer_loads.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/cubemap_gs.dxil.h"
#include "compiled_shaders/passes/forward_ps.dxil.h"
#include "compiled_shaders/passes/forward_vs_input_assembler.dxil.h"
#include "compiled_shaders/passes/forward_vs_buffer_loads.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/cubemap_gs.spirv.h"
#include "compiled_shaders/passes/forward_ps.spirv.h"
#include "compiled_shaders/passes/forward_vs_input_assembler.spirv.h"
#include "compiled_shaders/passes/forward_vs_buffer_loads.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/forward_cb.h>

#include "LightingConstants.h"


using namespace donut::engine;
using namespace donut::render;

ForwardShadingPass::~ForwardShadingPass() {}

ForwardShadingPass::ForwardShadingPass(
    nvrhi::IDevice* device,
    CommonRenderPasses *commonPasses)
    : m_Device(device)
    , m_CommonPasses(commonPasses)
{
    m_IsDX11 = m_Device && m_Device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D11;
}

void ForwardShadingPass::Init(ShaderFactory& shaderFactory, const CreateParameters& params)
{
    m_UseInputAssembler = params.useInputAssembler;

    m_SupportedViewTypes = ViewType::PLANAR;
    if (params.singlePassCubemap)
        m_SupportedViewTypes = ViewType::CUBEMAP;
    
    m_VertexShader = CreateVertexShader(shaderFactory, params);
    m_InputLayout = CreateInputLayout(m_VertexShader, params);
    m_GeometryShader = CreateGeometryShader(shaderFactory, params);
    m_PixelShader = CreatePixelShader(shaderFactory, params, false);
    m_PixelShaderTransmissive = CreatePixelShader(shaderFactory, params, true);

    if (params.materialBindings)
        m_MaterialBindings = params.materialBindings;
    else
        m_MaterialBindings = CreateMaterialBindingCache(*m_CommonPasses);

    auto samplerDesc = nvrhi::SamplerDesc()
        .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
        .setBorderColor(1.0f);
    m_Device->createSampler(samplerDesc, &m_ShadowSampler);

    m_Device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(ForwardShadingViewConstants), "ForwardShadingViewConstants", params.numConstantBufferVersions), &m_ForwardViewCB);
    m_Device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(ForwardShadingLightConstants), "ForwardShadingLightConstants", params.numConstantBufferVersions), &m_ForwardLightCB);

    m_ViewBindingLayout = CreateViewBindingLayout();
    m_ViewBindingSet = CreateViewBindingSet();
    m_ShadingBindingLayout = CreateShadingBindingLayout();
    m_InputBindingLayout = CreateInputBindingLayout();
}

void ForwardShadingPass::ResetBindingCache()
{
    m_MaterialBindings->Clear();
    m_ShadingBindingSets.clear();
    m_InputBindingSets.clear();
}

nvrhi::ShaderHandle ForwardShadingPass::CreateVertexShader(ShaderFactory& shaderFactory, const CreateParameters& params)
{
    char const* sourceFileName = "donut/passes/forward_vs.hlsl";

    if (params.useInputAssembler)
    {
        return shaderFactory.CreateAutoShader(sourceFileName, "input_assembler",
            DONUT_MAKE_PLATFORM_SHADER(g_forward_vs_input_assembler), nullptr, nvrhi::ShaderType::Vertex);
    }
    else
    {
        return shaderFactory.CreateAutoShader(sourceFileName, "buffer_loads",
            DONUT_MAKE_PLATFORM_SHADER(g_forward_vs_buffer_loads), nullptr, nvrhi::ShaderType::Vertex);
    }
}

nvrhi::ShaderHandle ForwardShadingPass::CreateGeometryShader(ShaderFactory& shaderFactory, const CreateParameters& params)
{
    if (params.singlePassCubemap)
    {
        auto desc = nvrhi::ShaderDesc()
            .setShaderType(nvrhi::ShaderType::Geometry)
            .setFastGSFlags(nvrhi::FastGeometryShaderFlags(
                nvrhi::FastGeometryShaderFlags::ForceFastGS |
                nvrhi::FastGeometryShaderFlags::UseViewportMask |
                nvrhi::FastGeometryShaderFlags::OffsetTargetIndexByViewportIndex))
            .setCoordinateSwizzling(CubemapView::GetCubemapCoordinateSwizzle());

        return shaderFactory.CreateAutoShader("donut/passes/cubemap_gs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_cubemap_gs), nullptr, desc);
    }

    return nullptr;
}

nvrhi::ShaderHandle ForwardShadingPass::CreatePixelShader(ShaderFactory& shaderFactory, const CreateParameters& params, bool transmissiveMaterial)
{
    std::vector<ShaderMacro> Macros;
    Macros.push_back(ShaderMacro("TRANSMISSIVE_MATERIAL", transmissiveMaterial ? "1" : "0"));

    return shaderFactory.CreateAutoShader("donut/passes/forward_ps.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_forward_ps), &Macros, nvrhi::ShaderType::Pixel);
}

nvrhi::InputLayoutHandle ForwardShadingPass::CreateInputLayout(nvrhi::IShader* vertexShader, const CreateParameters& params)
{
    if (params.useInputAssembler)
    {
        const nvrhi::VertexAttributeDesc inputDescs[] =
        {
            GetVertexAttributeDesc(VertexAttribute::Position, "POS", 0),
            GetVertexAttributeDesc(VertexAttribute::PrevPosition, "PREV_POS", 1),
            GetVertexAttributeDesc(VertexAttribute::TexCoord1, "TEXCOORD", 2),
            GetVertexAttributeDesc(VertexAttribute::Normal, "NORMAL", 3),
            GetVertexAttributeDesc(VertexAttribute::Tangent, "TANGENT", 4),
            GetVertexAttributeDesc(VertexAttribute::Transform, "TRANSFORM", 5),
        };

        nvrhi::InputLayoutHandle inputLayout;
        m_Device->createInputLayout(inputDescs, uint32_t(std::size(inputDescs)), vertexShader, &inputLayout);
        return inputLayout;
    }
    
    return nullptr;
}

nvrhi::BindingLayoutHandle ForwardShadingPass::CreateViewBindingLayout()
{
    auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
        .setRegisterSpaceAndDescriptorSet(FORWARD_SPACE_VIEW)
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(FORWARD_BINDING_VIEW_CONSTANTS));

    nvrhi::BindingLayoutHandle bindingLayout;
    m_Device->createBindingLayout(bindingLayoutDesc, &bindingLayout);
    return bindingLayout;
}


nvrhi::BindingSetHandle ForwardShadingPass::CreateViewBindingSet()
{
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .setTrackLiveness(m_TrackLiveness)
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_VIEW_CONSTANTS, m_ForwardViewCB));

    nvrhi::BindingSetHandle bindingSet;
    m_Device->createBindingSet(bindingSetDesc, m_ViewBindingLayout, &bindingSet);
    return bindingSet;
}

nvrhi::BindingLayoutHandle ForwardShadingPass::CreateShadingBindingLayout()
{
    auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Pixel)
        .setRegisterSpaceAndDescriptorSet(FORWARD_SPACE_SHADING)
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(FORWARD_BINDING_LIGHT_CONSTANTS))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_SHADOW_MAP_TEXTURE))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_DIFFUSE_LIGHT_PROBE_TEXTURE))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_SPECULAR_LIGHT_PROBE_TEXTURE))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_ENVIRONMENT_BRDF_TEXTURE))
        .addItem(nvrhi::BindingLayoutItem::Sampler(FORWARD_BINDING_MATERIAL_SAMPLER))
        .addItem(nvrhi::BindingLayoutItem::Sampler(FORWARD_BINDING_SHADOW_MAP_SAMPLER))
        .addItem(nvrhi::BindingLayoutItem::Sampler(FORWARD_BINDING_LIGHT_PROBE_SAMPLER))
        .addItem(nvrhi::BindingLayoutItem::Sampler(FORWARD_BINDING_ENVIRONMENT_BRDF_SAMPLER));

    nvrhi::BindingLayoutHandle bindingLayout;
    m_Device->createBindingLayout(bindingLayoutDesc, &bindingLayout);
    return bindingLayout;
}

nvrhi::BindingSetHandle ForwardShadingPass::CreateShadingBindingSet(nvrhi::ITexture* shadowMapTexture,
    nvrhi::ITexture* diffuse, nvrhi::ITexture* specular, nvrhi::ITexture* environmentBrdf)
{
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .setTrackLiveness(m_TrackLiveness)
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_LIGHT_CONSTANTS, m_ForwardLightCB))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(FORWARD_BINDING_SHADOW_MAP_TEXTURE,
            shadowMapTexture ? shadowMapTexture : m_CommonPasses->m_BlackTexture2DArray.Get()))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(FORWARD_BINDING_DIFFUSE_LIGHT_PROBE_TEXTURE,
            diffuse ? diffuse : m_CommonPasses->m_BlackCubeMapArray.Get()))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(FORWARD_BINDING_SPECULAR_LIGHT_PROBE_TEXTURE,
            specular ? specular : m_CommonPasses->m_BlackCubeMapArray.Get()))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(FORWARD_BINDING_ENVIRONMENT_BRDF_TEXTURE,
            environmentBrdf ? environmentBrdf : m_CommonPasses->m_BlackTexture.Get()))
        .addItem(nvrhi::BindingSetItem::Sampler(FORWARD_BINDING_MATERIAL_SAMPLER,
            m_CommonPasses->m_AnisotropicWrapSampler))
        .addItem(nvrhi::BindingSetItem::Sampler(FORWARD_BINDING_SHADOW_MAP_SAMPLER,
            m_ShadowSampler))
        .addItem(nvrhi::BindingSetItem::Sampler(FORWARD_BINDING_LIGHT_PROBE_SAMPLER,
            m_CommonPasses->m_LinearWrapSampler))
        .addItem(nvrhi::BindingSetItem::Sampler(FORWARD_BINDING_ENVIRONMENT_BRDF_SAMPLER,
            m_CommonPasses->m_LinearClampSampler));

    nvrhi::BindingSetHandle bindingSet;
    m_Device->createBindingSet(bindingSetDesc, m_ShadingBindingLayout, &bindingSet);
    return bindingSet;
}


nvrhi::GraphicsPipelineHandle ForwardShadingPass::CreateGraphicsPipeline(ForwardShadingPassPipelineKey const& key,
    nvrhi::FramebufferInfo const& framebufferInfo)
{
    nvrhi::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.inputLayout = m_InputLayout;
    pipelineDesc.VS = m_VertexShader;
    pipelineDesc.GS = m_GeometryShader;
    pipelineDesc.renderState.rasterState.frontCounterClockwise = key.frontCounterClockwise;
    pipelineDesc.renderState.rasterState.setCullMode(key.cullMode);
    pipelineDesc.renderState.blendState.alphaToCoverageEnable = false;
    pipelineDesc.shadingRateState = key.shadingRateState;
    pipelineDesc.bindingLayouts = { m_MaterialBindings->GetLayout(), m_ViewBindingLayout, m_ShadingBindingLayout };
    if (!m_UseInputAssembler)
        pipelineDesc.bindingLayouts.push_back(m_InputBindingLayout);

    bool const framebufferUsesMSAA = framebufferInfo.sampleCount > 1;

    pipelineDesc.renderState.depthStencilState
        .setDepthFunc(key.reverseDepth
            ? nvrhi::ComparisonFunc::GreaterOrEqual
            : nvrhi::ComparisonFunc::LessOrEqual);
    
    switch (key.domain)
    {
    case MaterialDomain::Opaque:
        pipelineDesc.PS = m_PixelShader;
        break;

    case MaterialDomain::AlphaTested:
        pipelineDesc.PS = m_PixelShader;
        pipelineDesc.renderState.blendState.alphaToCoverageEnable = framebufferUsesMSAA;
        break;

    case MaterialDomain::AlphaBlended: {
        pipelineDesc.PS = m_PixelShader;
        pipelineDesc.renderState.blendState.targets[0]
            .enableBlend()
            .setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
            .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
            .setSrcBlendAlpha(nvrhi::BlendFactor::Zero)
            .setDestBlendAlpha(nvrhi::BlendFactor::One);
        
        pipelineDesc.renderState.depthStencilState.disableDepthWrite();
        break;
    }

    case MaterialDomain::Transmissive:
    case MaterialDomain::TransmissiveAlphaTested:
    case MaterialDomain::TransmissiveAlphaBlended: {
        pipelineDesc.PS = m_PixelShaderTransmissive;
        pipelineDesc.renderState.blendState.targets[0]
            .enableBlend()
            .setSrcBlend(nvrhi::BlendFactor::One)
            .setDestBlend(nvrhi::BlendFactor::Src1Color)
            .setSrcBlendAlpha(nvrhi::BlendFactor::Zero)
            .setDestBlendAlpha(nvrhi::BlendFactor::One);

        pipelineDesc.renderState.depthStencilState.disableDepthWrite();
        break;
    }
    default:
        return nullptr;
    }

    nvrhi::GraphicsPipelineHandle pipeline;
    m_Device->createGraphicsPipeline1(pipelineDesc, framebufferInfo, &pipeline);
    return pipeline;
}

nvrhi::AutoPtr<MaterialBindingCache> ForwardShadingPass::CreateMaterialBindingCache(CommonRenderPasses& commonPasses)
{
    std::vector<MaterialResourceBinding> materialBindings = {
        { MaterialResource::ConstantBuffer,         FORWARD_BINDING_MATERIAL_CONSTANTS },
        { MaterialResource::DiffuseTexture,         FORWARD_BINDING_MATERIAL_DIFFUSE_TEXTURE },
        { MaterialResource::SpecularTexture,        FORWARD_BINDING_MATERIAL_SPECULAR_TEXTURE },
        { MaterialResource::NormalTexture,          FORWARD_BINDING_MATERIAL_NORMAL_TEXTURE },
        { MaterialResource::EmissiveTexture,        FORWARD_BINDING_MATERIAL_EMISSIVE_TEXTURE },
        { MaterialResource::OcclusionTexture,       FORWARD_BINDING_MATERIAL_OCCLUSION_TEXTURE },
        { MaterialResource::TransmissionTexture,    FORWARD_BINDING_MATERIAL_TRANSMISSION_TEXTURE },
        { MaterialResource::OpacityTexture,         FORWARD_BINDING_MATERIAL_OPACITY_TEXTURE }
    };

    return MAKE_RC_OBJ_PTR(MaterialBindingCache,
        m_Device,
        nvrhi::ShaderType::Pixel,
        /* registerSpace = */ (uint32_t)FORWARD_SPACE_MATERIAL,
        /* registerSpaceIsDescriptorSet = */ true,
        materialBindings,
        commonPasses.m_AnisotropicWrapSampler,
        commonPasses.m_GrayTexture);
}

void ForwardShadingPass::SetupView(
    GeometryPassContext& abstractContext,
    nvrhi::ICommandList* commandList,
    const IView* view,
    const IView* viewPrev)
{
    auto& context = static_cast<Context&>(abstractContext);
    
    ForwardShadingViewConstants viewConstants = {};
    view->FillPlanarViewConstants(viewConstants.view);
    commandList->writeBuffer(m_ForwardViewCB, &viewConstants, sizeof(viewConstants));

    context.keyTemplate.frontCounterClockwise = view->IsMirrored();
    context.keyTemplate.reverseDepth = view->IsReverseDepth();
    context.keyTemplate.shadingRateState = view->GetVariableRateShadingState();
}

namespace
{
    // The body of PrepareLights and PrepareExtractedLights, for both input forms (LightingConstants.h): the light
    // probes' textures checked, the light constants (FillForwardLights), the shading binding set for the first shadow
    // map's texture and the probes (the pass's cache, `bindingSets` under `mutex`, made by `createBindingSet` when
    // missing) into `outBindingSet`, then the probe constants, all written to `lightBuffer` on `commandList`. False,
    // with nothing bound or written, when the enabled probes use different textures (the caller reports it).
    template <typename Lights, typename BindingSets, typename CreateBindingSet>
    bool PrepareForwardLights(const Lights& lights, const std::vector<nvrhi::AutoPtr<LightProbe>>& lightProbes,
                              dm::float3 ambientColorTop, dm::float3 ambientColorBottom, std::mutex& mutex,
                              BindingSets& bindingSets, CreateBindingSet&& createBindingSet,
                              nvrhi::BindingSetHandle& outBindingSet, nvrhi::ICommandList* commandList,
                              nvrhi::IBuffer* lightBuffer)
    {
        nvrhi::ITexture* lightProbeDiffuse = nullptr;
        nvrhi::ITexture* lightProbeSpecular = nullptr;
        nvrhi::ITexture* lightProbeEnvironmentBrdf = nullptr;

        for (const auto& probe : lightProbes)
        {
            if (!probe->enabled)
                continue;

            if (lightProbeDiffuse == nullptr || lightProbeSpecular == nullptr || lightProbeEnvironmentBrdf == nullptr)
            {
                lightProbeDiffuse = probe->diffuseMap;
                lightProbeSpecular = probe->specularMap;
                lightProbeEnvironmentBrdf = probe->environmentBrdf;
            }
            else
            {
                if (lightProbeDiffuse != probe->diffuseMap || lightProbeSpecular != probe->specularMap || lightProbeEnvironmentBrdf != probe->environmentBrdf)
                    return false;
            }
        }

        ForwardShadingLightConstants constants = {};
        nvrhi::ITexture* shadowMapTexture = nullptr;
        lighting::FillForwardLights(lights, constants, FORWARD_MAX_LIGHTS, FORWARD_MAX_SHADOWS, shadowMapTexture);

        {
            std::lock_guard<std::mutex> lockGuard(mutex);

            nvrhi::BindingSetHandle& shadingBindings = bindingSets[std::make_pair(shadowMapTexture, lightProbeDiffuse)];

            if (!shadingBindings)
            {
                shadingBindings = createBindingSet(shadowMapTexture, lightProbeDiffuse, lightProbeSpecular, lightProbeEnvironmentBrdf);
            }

            outBindingSet = shadingBindings;
        }

        constants.ambientColorTop = float4(ambientColorTop, 0.f);
        constants.ambientColorBottom = float4(ambientColorBottom, 0.f);

        for (const auto& probe : lightProbes)
        {
            if (!probe->IsActive())
                continue;

            LightProbeConstants& lightProbeConstants = constants.lightProbes[constants.numLightProbes];
            probe->FillLightProbeConstants(lightProbeConstants);

            ++constants.numLightProbes;

            if (constants.numLightProbes >= FORWARD_MAX_LIGHT_PROBES)
                break;
        }

        commandList->writeBuffer(lightBuffer, &constants, sizeof(constants));
        return true;
    }
}

void ForwardShadingPass::PrepareLights(
    Context& context,
    nvrhi::ICommandList* commandList,
    const std::vector<nvrhi::AutoPtr<Light>>& lights,
    dm::float3 ambientColorTop,
    dm::float3 ambientColorBottom,
    const std::vector<nvrhi::AutoPtr<LightProbe>>& lightProbes)
{
    auto createBindingSet = [this](nvrhi::ITexture* shadowMapTexture, nvrhi::ITexture* diffuse, nvrhi::ITexture* specular,
                                   nvrhi::ITexture* environmentBrdf)
    {
        return CreateShadingBindingSet(shadowMapTexture, diffuse, specular, environmentBrdf);
    };
    if (!PrepareForwardLights(lighting::LeafLights{ &lights }, lightProbes, ambientColorTop, ambientColorBottom, m_Mutex,
                              m_ShadingBindingSets, createBindingSet, context.shadingBindingSet, commandList,
                              m_ForwardLightCB))
    {
        log::error("All lights probe submitted to ForwardShadingPass::PrepareLights(...) must use the same set of textures");
    }
}

// The snapshot's lights (plan step E5, sub-plan item E5.6): PrepareLights' body (one helper above, so that both bind
// and write alike), each light's shadow map from the extracted light (the render service's, or the leaf's own without
// service shadows). The shadow texture is bound as PrepareLights binds it: an application in a render graph declares
// the read of ExtractedLight::shadowTexture (GeometryGraphInputs::shaderReads, or its own pass's reads).
void ForwardShadingPass::PrepareExtractedLights(
    Context& context,
    nvrhi::ICommandList* commandList,
    const ExtractedLight* lights,
    uint32_t lightCount,
    float3 ambientColorTop,
    float3 ambientColorBottom,
    const std::vector<nvrhi::AutoPtr<LightProbe>>& lightProbes)
{
    auto createBindingSet = [this](nvrhi::ITexture* shadowMapTexture, nvrhi::ITexture* diffuse, nvrhi::ITexture* specular,
                                   nvrhi::ITexture* environmentBrdf)
    {
        return CreateShadingBindingSet(shadowMapTexture, diffuse, specular, environmentBrdf);
    };
    if (!PrepareForwardLights(lighting::SnapshotLights{ lights, lightCount }, lightProbes, ambientColorTop,
                              ambientColorBottom, m_Mutex, m_ShadingBindingSets, createBindingSet,
                              context.shadingBindingSet, commandList, m_ForwardLightCB))
    {
        log::warning("ForwardShadingPass::PrepareExtractedLights: the light probes must use the same set of textures");
    }
}

ViewType::Enum ForwardShadingPass::GetSupportedViewTypes() const
{
    return m_SupportedViewTypes;
}

bool ForwardShadingPass::SetupMaterial(GeometryPassContext& abstractContext, const Material* material,
    nvrhi::RasterCullMode cullMode, nvrhi::GraphicsState& state)
{
    auto& context = static_cast<Context&>(abstractContext);

    nvrhi::IBindingSet* materialBindingSet = m_MaterialBindings->GetMaterialBindingSet(material);

    if (!materialBindingSet)
        return false;

    if (material->domain >= MaterialDomain::Count || cullMode > nvrhi::RasterCullMode::None)
    {
        assert(false);
        return false;
    }

    ForwardShadingPassPipelineKey key = context.keyTemplate;
    key.cullMode = cullMode;
    key.domain = material->domain;

    nvrhi::GraphicsPipelineHandle& pipeline = m_Pipelines[key];
    nvrhi::FramebufferInfo const framebufferInfo = state.framebuffer->getFramebufferInfo().getInfo();

    // The framebuffer's formats are part of the cache key (design 13.3): a target of another format, such as a
    // render graph's, replaces the pipeline instead of asserting.
    if (!pipeline || pipeline->getFramebufferInfo() != framebufferInfo)
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);

        if (!pipeline || pipeline->getFramebufferInfo() != framebufferInfo)
            pipeline = CreateGraphicsPipeline(key, framebufferInfo);

        if (!pipeline)
            return false;
    }

    state.pipeline = pipeline;
    state.bindings = { materialBindingSet, m_ViewBindingSet, context.shadingBindingSet };
    
    if (!m_UseInputAssembler)
        state.bindings.push_back(context.inputBindingSet);

    return true;
}

void ForwardShadingPass::SetupInputBuffers(GeometryPassContext& abstractContext, const BufferGroup* buffers, nvrhi::GraphicsState& state)
{
    auto& context = static_cast<Context&>(abstractContext);
    
    state.indexBuffer = { buffers->indexBuffer, nvrhi::Format::R32_UINT, 0 };
    
    if (m_UseInputAssembler)
    {
        state.vertexBuffers = {
            { buffers->vertexBuffer, 0, buffers->getVertexBufferRange(VertexAttribute::Position).byteOffset },
            { buffers->vertexBuffer, 1, buffers->getVertexBufferRange(VertexAttribute::PrevPosition).byteOffset },
            { buffers->vertexBuffer, 2, buffers->getVertexBufferRange(VertexAttribute::TexCoord1).byteOffset },
            { buffers->vertexBuffer, 3, buffers->getVertexBufferRange(VertexAttribute::Normal).byteOffset },
            { buffers->vertexBuffer, 4, buffers->getVertexBufferRange(VertexAttribute::Tangent).byteOffset },
            { buffers->instanceBuffer, 5, 0 }
        };
    }
    else
    {
        context.inputBindingSet = GetOrCreateInputBindingSet(buffers);
        // SetupMaterial appends the input set to the bindings, and RenderView calls SetupInputBuffers alone when the
        // buffers change and the material does not: replace it here, or the next draws read the previous buffer
        // group's vertex and instance buffers at this group's offsets.
        if (!state.bindings.empty() && state.bindings.back() &&
            state.bindings.back()->getLayout() == m_InputBindingLayout.Get())
            state.bindings.back() = context.inputBindingSet;
        context.positionOffset = uint32_t(buffers->getVertexBufferRange(VertexAttribute::Position).byteOffset);
        context.texCoordOffset = uint32_t(buffers->getVertexBufferRange(VertexAttribute::TexCoord1).byteOffset);
        context.normalOffset = uint32_t(buffers->getVertexBufferRange(VertexAttribute::Normal).byteOffset);
        context.tangentOffset = uint32_t(buffers->getVertexBufferRange(VertexAttribute::Tangent).byteOffset);
    }
}

nvrhi::BindingLayoutHandle ForwardShadingPass::CreateInputBindingLayout()
{
    if (m_UseInputAssembler)
        return nullptr;

    auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Vertex)
        .setRegisterSpaceAndDescriptorSet(FORWARD_SPACE_INPUT)
        .addItem(m_IsDX11
            ? nvrhi::BindingLayoutItem::RawBuffer_SRV(FORWARD_BINDING_INSTANCE_BUFFER)
            : nvrhi::BindingLayoutItem::StructuredBuffer_SRV(FORWARD_BINDING_INSTANCE_BUFFER))
        .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(FORWARD_BINDING_VERTEX_BUFFER))
        .addItem(nvrhi::BindingLayoutItem::PushConstants(FORWARD_BINDING_PUSH_CONSTANTS, sizeof(ForwardPushConstants)));
        
    nvrhi::BindingLayoutHandle bindingLayout;
    m_Device->createBindingLayout(bindingLayoutDesc, &bindingLayout);
    return bindingLayout;
}

nvrhi::BindingSetHandle ForwardShadingPass::CreateInputBindingSet(const BufferGroup* bufferGroup)
{
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(m_IsDX11
            ? nvrhi::BindingSetItem::RawBuffer_SRV(FORWARD_BINDING_INSTANCE_BUFFER, bufferGroup->instanceBuffer)
            : nvrhi::BindingSetItem::StructuredBuffer_SRV(FORWARD_BINDING_INSTANCE_BUFFER, bufferGroup->instanceBuffer))
        .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(FORWARD_BINDING_VERTEX_BUFFER, bufferGroup->vertexBuffer))
        .addItem(nvrhi::BindingSetItem::PushConstants(FORWARD_BINDING_PUSH_CONSTANTS, sizeof(ForwardPushConstants)));

    nvrhi::BindingSetHandle bindingSet;
    m_Device->createBindingSet(bindingSetDesc, m_InputBindingLayout, &bindingSet);
    return bindingSet;
}

nvrhi::BindingSetHandle ForwardShadingPass::GetOrCreateInputBindingSet(const BufferGroup* bufferGroup)
{
    auto it = m_InputBindingSets.find(bufferGroup);
    if (it == m_InputBindingSets.end())
    {
        auto bindingSet = CreateInputBindingSet(bufferGroup);
        m_InputBindingSets[bufferGroup] = bindingSet;
        return bindingSet;
    }

    return it->second;
}

void ForwardShadingPass::SetPushConstants(
    donut::render::GeometryPassContext& abstractContext,
    nvrhi::ICommandList* commandList,
    nvrhi::GraphicsState& state,
    nvrhi::DrawArguments& args)
{
    if (m_UseInputAssembler)
        return;
        
    auto& context = static_cast<Context&>(abstractContext);

    ForwardPushConstants constants;
    constants.startInstanceLocation = args.startInstanceLocation;
    constants.startVertexLocation = args.startVertexLocation;
    constants.positionOffset = context.positionOffset;
    constants.texCoordOffset = context.texCoordOffset;
    constants.normalOffset = context.normalOffset;
    constants.tangentOffset = context.tangentOffset;

    commandList->setPushConstants(&constants, sizeof(constants));

    args.startInstanceLocation = 0;
    args.startVertexLocation = 0;
}