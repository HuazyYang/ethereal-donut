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

#pragma once

#include <donut/core/math/math.h>
#include <map>
#include <nvrhi/nvrhi.h>
#include <nvrhi/core/foundation.h>
#include <donut/render/RenderGraph.h>


namespace donut::engine
{
    class ShaderFactory;
    class CommonRenderPasses;
    class FramebufferFactory;
    class ICompositeView;
}

namespace donut::render
{
    class PixelReadbackPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(PixelReadbackPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        // Not in a render graph yet (design 13.3): AddPass fails with a diagnostic naming the pass, and the caller
        // runs the pass outside the graph and imports its results.
        nvrhi::FRESULT Setup(ISetupContext*, void*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
        nvrhi::FRESULT Prepare(IPrepareContext*, void*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
        nvrhi::FRESULT Execute(IPassContext*, const void*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
        void OnResourcesReleased() noexcept override { }

    private:
    private:
        nvrhi::DeviceHandle m_Device;
        nvrhi::ShaderHandle m_Shader;
        nvrhi::ComputePipelineHandle m_Pipeline;
        nvrhi::BindingLayoutHandle m_BindingLayout;
        nvrhi::BindingSetHandle m_BindingSet;
        nvrhi::BufferHandle m_ConstantBuffer;
        nvrhi::BufferHandle m_IntermediateBuffer;
        nvrhi::BufferHandle m_ReadbackBuffer;

    public:
        PixelReadbackPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            nvrhi::ITexture* inputTexture,
            nvrhi::Format format,
            uint32_t arraySlice = 0,
            uint32_t mipLevel = 0);

        void Capture(nvrhi::ICommandList* commandList, dm::uint2 pixelPosition);

        dm::float4 ReadFloats();
        dm::uint4 ReadUInts();
        dm::int4 ReadInts();
    };
}