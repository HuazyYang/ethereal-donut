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
#include <nvrhi/core/autoptr.h>
#include <donut/engine/BindingCache.h>
#include <donut/core/math/math.h>
#include <nvrhi/core/foundation.h>
#include <donut/render/RenderGraph.h>
#include <nvrhi/nvrhi.h>
#include <vector>

namespace donut::engine
{
    class ShaderFactory;
    class CommonRenderPasses;
}

// Compute reduction pass to generate mipmap levels

namespace donut::render
{

    // The pass in a render graph (design 13.3): one compute use in Restore mode that reads mip 0 of `texture` and
    // writes mips 1 to `maxLOD` (all of them when negative); the body reads the mips it has just written.
    struct MipMapGenGraphInputs
    {
        ResourceId texture = InvalidResourceId;
        int maxLOD = -1;
    };

    struct MipMapGenGraphOutputs
    {
        ResourceId texture = InvalidResourceId;
    };

    class MipMapGenPass : public nvrhi::ObjectImpl<IGpuPass> {

    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MipMapGenPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        enum Mode : uint8_t {
            MODE_COLOR = 0,  // bilinear reduction of RGB channels
            MODE_MIN = 1,    // min() reduction of R channel
            MODE_MAX = 2,    // max() reduction of R channel
            MODE_MINMAX = 3, // min() and max() reductions of R channel into RG channels
        };

        // note : 'texture' must have been allocated with some mip levels
        MipMapGenPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            nvrhi::TextureHandle texture,
            Mode mode = Mode::MODE_MAX);

        // Dispatches reduction kernel : reads LOD 0 and populates
        // LOD 1 and up
        void Dispatch(nvrhi::ICommandList* commandList, int maxLOD=-1);

        // debug : blits mip-map levels in spiral pattern to 'target'
        // (assumes 'target' texture resolution is high enough...)
        void Display(
            engine::CommonRenderPasses* commonPasses,
            nvrhi::ICommandList* commandList, 
            nvrhi::IFramebuffer* target);

        // Out of line: NullTextures is only declared here, and the reference-counted object is destroyed elsewhere.
        ~MipMapGenPass();

        // For a render graph: the texture is chosen per AddToGraph; Prepare creates the shader and pipeline.
        MipMapGenPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            Mode mode = Mode::MODE_MAX);

        MipMapGenGraphOutputs AddToGraph(IRenderGraph* graph, const MipMapGenGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:

        nvrhi::DeviceHandle m_Device;
        nvrhi::ShaderHandle m_Shader;
        nvrhi::TextureHandle m_Texture;
        nvrhi::BufferHandle m_ConstantBuffer;
        nvrhi::BindingLayoutHandle m_BindingLayout;
        std::vector<nvrhi::BindingSetHandle> m_BindingSets;
        nvrhi::ComputePipelineHandle m_Pso;

        // Set of unique dummy textures - see details in class implementation
        struct NullTextures;
        nvrhi::AutoPtr<NullTextures> m_NullTextures;

        engine::BindingCache m_BindingCache;

        // Graph construction.
        void CreateObjects(engine::ShaderFactory* shaderFactory);
        nvrhi::AutoPtr<engine::ShaderFactory> m_ShaderFactory;
        Mode m_Mode = Mode::MODE_MAX;
        nvrhi::Format m_NullFormat = nvrhi::Format::UNKNOWN;

    };

} // end namespace donut::render
