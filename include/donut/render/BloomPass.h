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
#include <nvrhi/core/foundation.h>
#include <nvrhi/core/autoptr.h>
#include <donut/engine/BindingCache.h>
#include <donut/render/RenderGraph.h>
#include <nvrhi/nvrhi.h>
#include <unordered_map>


namespace donut::engine
{
    class ShaderFactory;
    class ShadowMap;
    class CommonRenderPasses;
    class FramebufferFactory;
    class ICompositeView;
}

namespace donut::render
{
    // The pass in a render graph (design 13.3), graph-native: per planar child view of `view`, two downscales, a
    // horizontal and a vertical blur into transients, and a blended apply onto `color`. `view` must stay valid
    // until the graph has executed.
    struct BloomGraphInputs
    {
        const engine::ICompositeView* view = nullptr;
        ResourceId color = InvalidResourceId;
        float sigmaInPixels = 4.f;
        float blendFactor = 0.05f;
    };

    struct BloomGraphOutputs
    {
        ResourceId color = InvalidResourceId;
    };

    class BloomPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(BloomPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:
        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;
        nvrhi::AutoPtr<engine::FramebufferFactory> m_FramebufferFactory;

        nvrhi::DeviceHandle m_Device;

        struct PerViewData
        {
            nvrhi::GraphicsPipelineHandle bloomBlurPso;

            nvrhi::TextureHandle textureDownscale1;
            nvrhi::FramebufferHandle framebufferDownscale1;
            nvrhi::TextureHandle textureDownscale2;
            nvrhi::FramebufferHandle framebufferDownscale2;

            nvrhi::TextureHandle texturePass1Blur;
            nvrhi::FramebufferHandle framebufferPass1Blur;
            nvrhi::TextureHandle texturePass2Blur;
            nvrhi::FramebufferHandle framebufferPass2Blur;

            nvrhi::BindingSetHandle bloomBlurBindingSetPass1;
            nvrhi::BindingSetHandle bloomBlurBindingSetPass2;
            nvrhi::BindingSetHandle bloomBlurBindingSetPass3;
            nvrhi::BindingSetHandle blitFromDownscale1BindingSet;
            nvrhi::BindingSetHandle compositeBlitBindingSet;
        };

        std::vector<PerViewData> m_PerViewData;
        nvrhi::BufferHandle m_BloomHBlurCB;
        nvrhi::BufferHandle m_BloomVBlurCB;
        nvrhi::ShaderHandle m_BloomBlurPixelShader;
        nvrhi::BindingLayoutHandle m_BloomBlurBindingLayout;
        nvrhi::BindingLayoutHandle m_BloomApplyBindingLayout;

        engine::BindingCache m_BindingCache;

    public:
        BloomPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory *shaderFactory,
            engine::CommonRenderPasses *commonPasses,
            engine::FramebufferFactory *framebufferFactory,
            const engine::ICompositeView &compositeView);

        void Render(
            nvrhi::ICommandList* commandList,
            engine::FramebufferFactory *framebufferFactory,
            const engine::ICompositeView* compositeView,
            nvrhi::ITexture* sourceDestTexture,
            float sigmaInPixels,
            float blendFactor);

        // For a render graph: Prepare creates the shader and pipelines; the chain's textures are transients.
        BloomPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory *shaderFactory,
            engine::CommonRenderPasses *commonPasses);

        BloomGraphOutputs AddToGraph(IRenderGraph* graph, const BloomGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:
        void CreateCommonObjects(engine::ShaderFactory* shaderFactory);

        nvrhi::AutoPtr<engine::ShaderFactory> m_ShaderFactory;
        nvrhi::GraphicsPipelineHandle m_GraphBlurPso;
    };
}
