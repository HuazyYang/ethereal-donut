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

#include <donut/render/MipMapGenPass.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/passes/mipmapgen_cs.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/passes/mipmapgen_cs.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/passes/mipmapgen_cs.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/mipmapgen_cb.h>

#include <cassert>
#include <mutex>

using namespace donut::engine;
using namespace donut::render;

// The compute shader reduces 'NUM_LODS' mip-levels at a time into an
// array of NUM_LODS bound UAVs. For textures that have a number
// of mip levels that is not a multiple of NUM_LODS, we need to bind
// "something" to the UAV slots : a set of static dummy NullTextures.
//
// Each MipMapGen pass object owns its set of NullTextures.
//

// In the format of the texture whose mips are generated: Vulkan requires a storage image view of the format the
// shader declares for it.
static nvrhi::TextureHandle createNullTexture(nvrhi::DeviceHandle device, nvrhi::Format format)
{
    nvrhi::TextureDesc desc;
    desc.width = 1;
    desc.height = 1;
    desc.isRenderTarget = false;
    desc.useClearValue = false;
    desc.sampleCount = 1;
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.keepInitialState = true;
    desc.arraySize = 1;
    desc.isUAV = true;
    desc.format = format;

    nvrhi::TextureHandle texture;
    device->createTexture(desc, &texture);
    return texture;
}

struct MipMapGenPass::NullTextures: nvrhi::ObjectImpl<nvrhi::IObject> {
    NVRHI_BEGIN_INTERFACE_TABLE_INLINE(NullTextures)
    NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
    NVRHI_END_INTERFACE_TABLE()


    nvrhi::TextureHandle lod[NUM_LODS];

    // One set per pass object. A process-wide set outlived its device (released after the device at exit) and served
    // the textures of the first device to every other one.
    static nvrhi::AutoPtr<NullTextures> get(nvrhi::DeviceHandle device, nvrhi::Format format)
    {
        nvrhi::AutoPtr<NullTextures> nullTextures = MAKE_RC_OBJ_PTR(NullTextures);
        for (int i = 0; i < NUM_LODS; ++i)
            nullTextures->lod[i] = createNullTexture(device, format);
        return nullTextures;
    }
};

void MipMapGenPass::CreateObjects(ShaderFactory* shaderFactory)
{

    std::vector<ShaderMacro> macros = { {"MODE", std::to_string(m_Mode)} };
    m_Shader = shaderFactory->CreateAutoShader(
        "donut/passes/mipmapgen_cs.hlsl", "main", DONUT_MAKE_PLATFORM_SHADER(g_mipmapgen_cs), &macros, nvrhi::ShaderType::Compute);

    // Constants
    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(MipmmapGenConstants);
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.debugName = "MipMapGenPass/Constants";
    constantBufferDesc.maxVersions = c_MaxRenderPassConstantBufferVersions;
    m_Device->createBuffer(constantBufferDesc, &m_ConstantBuffer);

    // BindingLayout
    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::Compute;
    layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0));
    layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(0));
    layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(0).setSize(NUM_LODS));
    m_Device->createBindingLayout(layoutDesc, &m_BindingLayout);


    nvrhi::ComputePipelineDesc computePipelineDesc;
    computePipelineDesc.CS = m_Shader;
    computePipelineDesc.bindingLayouts = { m_BindingLayout };
    m_Pso = nullptr;
    m_Device->createComputePipeline(computePipelineDesc, &m_Pso);
}

MipMapGenPass::MipMapGenPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    nvrhi::TextureHandle input, 
    Mode mode)
    : m_Device(device)
    , m_Texture(input)
    , m_BindingSets(MAX_PASSES)
    , m_BindingCache(device)
{
    assert(m_Texture);

    m_NullTextures = NullTextures::get(m_Device, m_Texture->getDesc().format);

    uint nmipLevels = m_Texture->getDesc().mipLevels;

    // Shader
    assert(mode>=0 && mode <= MODE_MINMAX);
    m_Mode = mode;
    CreateObjects(shaderFactory);

    // BindingSets
    m_BindingSets.resize(MAX_PASSES);
    nvrhi::BindingSetDesc setDesc;
    for (uint i = 0; i < (uint)m_BindingSets.size(); ++i)
    {
        // Create a unique binding set for each compute pass
        if (i * NUM_LODS >= nmipLevels)
            break;

        nvrhi::BindingSetHandle & set = m_BindingSets[i];

        nvrhi::BindingSetDesc setDesc;
        setDesc.bindings.push_back(nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer));
        setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(0, m_Texture, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(i*NUM_LODS, 1, 0, 1)));
        for (uint mipLevel = 1; mipLevel <= NUM_LODS; ++mipLevel)
        {   // output UAVs start after the mip-level UAV that was computed last
            if (i * NUM_LODS + mipLevel < nmipLevels)
            {
                setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(0, m_Texture)
                    .setArrayElement(mipLevel - 1)
                    .setSubresources(nvrhi::TextureSubresourceSet(i*NUM_LODS + mipLevel, 1, 0, 1)));
            }
            else
            {
                setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(0, m_NullTextures->lod[mipLevel-1])
                    .setArrayElement(mipLevel - 1));
            }
        }
        m_Device->createBindingSet(setDesc, m_BindingLayout, &set);
    }

}

void MipMapGenPass::Dispatch(nvrhi::ICommandList* commandList, int maxLOD) 
{
    assert(m_Texture);

    commandList->beginMarker("MipMapGen::Dispatch");

    uint nmipLevels = m_Texture->getDesc().mipLevels;
    if (maxLOD > 0 && maxLOD < (int)nmipLevels)
        nmipLevels = maxLOD+1;

    uint width = m_Texture->getDesc().width,
         height = m_Texture->getDesc().height;

    width = (width + GROUP_SIZE - 1) / GROUP_SIZE;
    height = (height + GROUP_SIZE - 1) / GROUP_SIZE;
 
    for (uint i=0; i < MAX_PASSES; ++i)
    {
        if (i * NUM_LODS >= nmipLevels)
            break;

        MipmmapGenConstants constants = {};
        constants.numLODs = std::min(nmipLevels - i*NUM_LODS -1, (uint32_t)NUM_LODS);
        constants.dispatch = i;
        commandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));

        nvrhi::ComputeState state;
        state.pipeline = m_Pso;
        state.bindings = { m_BindingSets[i] };
        commandList->setComputeState(state);
        commandList->dispatch(width, height);
    }

    commandList->endMarker(); // "MipMapGen::Dispatch"
}


void MipMapGenPass::Display(donut::engine::CommonRenderPasses* commonPasses, nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* target)
{
    assert(m_Texture);
    
    commandList->beginMarker("MipMapGen::Display");
    
    nvrhi::Viewport viewport = nvrhi::Viewport((float)target->getFramebufferInfo().width, (float)target->getFramebufferInfo().height);

    float2 size = { m_Texture->getDesc().width / 2.f, m_Texture->getDesc().height / 2.f };
    float2 corner = { 10.f, uint(viewport.maxY) - 10.f };
 
    for (uint level = 0; level < m_Texture->getDesc().mipLevels-1; ++level)
    {
        BlitParameters blitParams;
        blitParams.targetFramebuffer = target;
        blitParams.sourceTexture = m_Texture;
        blitParams.sourceMip = level + 1;
        blitParams.targetViewport = nvrhi::Viewport(
            corner.x,
            corner.x + size.x,
            corner.y - size.y,
            corner.y, 0.f, 1.f
        );

        commonPasses->BlitTexture(commandList, blitParams, &m_BindingCache);

        // spiral pattern
        switch (level % 4)
        {
            case 0: corner += float2(size.x + 10.f, 0.f); break;
            case 1: corner += float2(size.x / 2.f, -(size.y + 10.f)); break;
            case 2: corner += float2(-size.x / 2.f - 10.f, -size.y / 2.f); break;
            case 3: corner += float2(0.f, size.y); break;
        }
        size = { size.x / 2.f, size.y / 2.f };
    }
    commandList->endMarker(); // "MipMapGen::Display"
}

MipMapGenPass::~MipMapGenPass() = default;

// ------------------------------------------------------------------------------------------ render graph (13.3)

MipMapGenPass::MipMapGenPass(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    Mode mode)
    : m_Device(device)
    , m_BindingSets(MAX_PASSES)
    , m_BindingCache(device)
    , m_ShaderFactory(shaderFactory)
    , m_Mode(mode)
{
}

namespace
{
    struct MipMapGenUse
    {
        ResourceId texture;
        uint32_t mipLevels;   // levels written and read: 0 to mipLevels - 1
        uint32_t width;
        uint32_t height;
    };
}

MipMapGenGraphOutputs MipMapGenPass::AddToGraph(IRenderGraph* graph, const MipMapGenGraphInputs& inputs)
{
    MipMapGenGraphOutputs outputs{ inputs.texture };
    nvrhi::TextureDesc desc;
    if (graph->GetTextureDesc(inputs.texture, desc) != nvrhi::FS_OK || desc.mipLevels < 2)
        return outputs;
    uint32_t mipLevels = desc.mipLevels;
    if (inputs.maxLOD > 0 && inputs.maxLOD < int(mipLevels))
        mipLevels = uint32_t(inputs.maxLOD) + 1;
    MipMapGenUse* use = NewFrameData<MipMapGenUse>(graph, inputs.texture, mipLevels, desc.width, desc.height);
    if (!use)
        return outputs;
    // Restore: the body reads mips it wrote in an earlier dispatch, through the automatic barriers.
    graph->AddPass(PassDesc{ "MipMapGen", PassType::Compute, PassFlags::Restore, 0 }, this, use, nullptr);
    outputs.texture = use->texture;
    return outputs;
}

nvrhi::FRESULT MipMapGenPass::Setup(ISetupContext* context, void* data) noexcept
{
    MipMapGenUse& use = *static_cast<MipMapGenUse*>(data);
    context->Read(use.texture, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion().setMips(0, 1));
    use.texture = context->Write(use.texture, Access::UAVWrite, nvrhi::ShaderType::Compute,
                                 nvrhi::SubresourceRegion().setMips(1, use.mipLevels - 1), AccessFlags::FullWrite);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT MipMapGenPass::Prepare(IPrepareContext* context, void* data) noexcept
{
    const MipMapGenUse& use = *static_cast<const MipMapGenUse*>(data);
    if (!m_Device || (!m_Pso && !m_ShaderFactory))
        return nvrhi::FE_INVALID_ARGS;
    try
    {
        const nvrhi::TextureDesc* desc = context->GetTextureDesc(use.texture);
        const nvrhi::Format format = desc ? desc->format : nvrhi::Format::RGBA8_UNORM;
        if (!m_NullTextures || m_NullFormat != format)
        {
            m_NullTextures = NullTextures::get(m_Device, format);
            m_NullFormat = format;
            m_BindingCache.Clear();
        }
        if (!m_Pso)
            CreateObjects(m_ShaderFactory);
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return m_Pso ? nvrhi::FS_OK : nvrhi::FE_GENERIC_ERROR;
}

nvrhi::FRESULT MipMapGenPass::Execute(IPassContext* context, const void* data) noexcept
{
    const MipMapGenUse& use = *static_cast<const MipMapGenUse*>(data);
    nvrhi::ITexture* texture = context->GetTexture(use.texture);
    if (!texture || !m_Pso)
        return nvrhi::FE_INVALID_ARGS;
    nvrhi::ICommandList* commandList = context->GetCommandList();
    try
    {
        const uint32_t width = (use.width + GROUP_SIZE - 1) / GROUP_SIZE;
        const uint32_t height = (use.height + GROUP_SIZE - 1) / GROUP_SIZE;
        for (uint32_t i = 0; i < MAX_PASSES && i * NUM_LODS < use.mipLevels; ++i)
        {
            nvrhi::BindingSetDesc setDesc;
            setDesc.bindings.push_back(nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer));
            setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(0, texture, nvrhi::Format::UNKNOWN,
                nvrhi::TextureSubresourceSet(i * NUM_LODS, 1, 0, 1)));
            for (uint32_t mipLevel = 1; mipLevel <= NUM_LODS; ++mipLevel)
            {
                if (i * NUM_LODS + mipLevel < use.mipLevels)
                    setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(0, texture)
                        .setArrayElement(mipLevel - 1)
                        .setSubresources(nvrhi::TextureSubresourceSet(i * NUM_LODS + mipLevel, 1, 0, 1)));
                else
                    setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(0, m_NullTextures->lod[mipLevel - 1])
                        .setArrayElement(mipLevel - 1));
            }

            MipmmapGenConstants constants = {};
            constants.numLODs = std::min(use.mipLevels - i * NUM_LODS - 1, (uint32_t)NUM_LODS);
            constants.dispatch = i;
            commandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));

            nvrhi::ComputeState state;
            state.pipeline = m_Pso;
            state.bindings = { m_BindingCache.GetOrCreateBindingSet(setDesc, m_BindingLayout) };
            commandList->setComputeState(state);
            commandList->dispatch(width, height);
        }
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

void MipMapGenPass::OnResourcesReleased() noexcept
{
    m_BindingCache.Clear();
}
