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

#include <donut/render/BlitPass.h>

using namespace donut::engine;
using namespace donut::render;

namespace
{
    struct BlitUse
    {
        BlitGraphInputs inputs;
    };
}

BlitPass::BlitPass(nvrhi::IDevice* device, CommonRenderPasses* commonPasses)
    : m_CommonPasses(commonPasses)
    , m_BindingCache(device)
{
}

BlitGraphOutputs BlitPass::AddToGraph(IRenderGraph* graph, const BlitGraphInputs& inputs)
{
    BlitUse* use = NewFrameData<BlitUse>(graph, inputs);
    if (!use || graph->AddPass(PassDesc{ "Blit", PassType::Raster, PassFlags::None, 0 }, this, use, nullptr) != nvrhi::FS_OK)
        return BlitGraphOutputs{ inputs.target };
    return BlitGraphOutputs{ use->inputs.target };
}

nvrhi::FRESULT BlitPass::Setup(ISetupContext* context, void* data) noexcept
{
    BlitGraphInputs& inputs = static_cast<BlitUse*>(data)->inputs;
    context->Read(inputs.source, Access::ShaderRead, nvrhi::ShaderType::Pixel,
                  nvrhi::SubresourceRegion().setMips(inputs.sourceMip, 1).setSlices(inputs.sourceArraySlice, 1));
    inputs.target = context->SetColorTarget(0, inputs.target, inputs.targetRegion,
                                            inputs.fullWrite ? AccessFlags::FullWrite : AccessFlags::None, nullptr);
    return nvrhi::FS_OK;
}

nvrhi::FRESULT BlitPass::Prepare(IPrepareContext*, void*) noexcept
{
    // CommonRenderPasses creates and caches its pipelines per framebuffer when it blits.
    return m_CommonPasses ? nvrhi::FS_OK : nvrhi::FE_INVALID_ARGS;
}

nvrhi::FRESULT BlitPass::Execute(IPassContext* context, const void* data) noexcept
{
    const BlitGraphInputs& inputs = static_cast<const BlitUse*>(data)->inputs;
    nvrhi::IFramebuffer* framebuffer = context->GetFramebuffer();
    nvrhi::ITexture* source = context->GetTexture(inputs.source);
    if (!framebuffer || !source)
        return nvrhi::FE_INVALID_ARGS;
    BlitParameters params;
    params.targetFramebuffer = framebuffer;
    params.targetBox = inputs.targetBox;
    params.sourceTexture = source;
    params.sourceMip = inputs.sourceMip;
    params.sourceArraySlice = inputs.sourceArraySlice;
    params.sourceBox = inputs.sourceBox;
    params.sourceFormat = inputs.sourceFormat;
    params.sampler = inputs.sampler;
    try
    {
        m_CommonPasses->BlitTexture(context->GetCommandList(), params, &m_BindingCache);
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

void BlitPass::OnResourcesReleased() noexcept
{
    m_BindingCache.Clear();
}
