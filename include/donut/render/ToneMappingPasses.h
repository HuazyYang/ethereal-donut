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
#include <nvrhi/nvrhi.h>
#include <donut/engine/BindingCache.h>
#include <donut/render/RenderGraph.h>
#include <string>
#include <unordered_map>

namespace donut::engine
{
    class ShaderFactory;
    class CommonRenderPasses;
    class FramebufferFactory;
    class ICompositeView;
    class IView;
}

namespace donut::render
{
    struct ToneMappingParameters
    {
        float histogramLowPercentile = 0.8f;
        float histogramHighPercentile = 0.95f;
        float eyeAdaptationSpeedUp = 1.f;
        float eyeAdaptationSpeedDown = 0.5f;
        float minAdaptedLuminance = 0.02f;
        float maxAdaptedLuminance = 0.5f;
        float exposureBias = -0.5f;
        float whitePoint = 3.f;
        bool enableColorLUT = true;
    };

    // The pass in a render graph (design 13.3): a histogram use (the histogram is a transient buffer), an exposure
    // use (the exposure is a persistent buffer of this pass object), and one raster use per planar child view of
    // `view` that writes all of that view's subresources of `output`. `view` must stay valid until the graph has
    // executed. The exposure starts at the first frame's target, without adaptation.
    struct ToneMappingGraphInputs
    {
        const engine::ICompositeView* view = nullptr;
        ResourceId source = InvalidResourceId;
        ResourceId output = InvalidResourceId;
        ToneMappingParameters params;
        float frameTime = 0.f;
    };

    struct ToneMappingGraphOutputs
    {
        ResourceId output = InvalidResourceId;
        ResourceId exposure = InvalidResourceId;
    };

    class ToneMappingPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(ToneMappingPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:

        nvrhi::DeviceHandle m_Device;
        nvrhi::ShaderHandle m_PixelShader;
        nvrhi::ShaderHandle m_HistogramComputeShader;
        nvrhi::ShaderHandle m_ExposureComputeShader;
        uint32_t m_HistogramBins;

        nvrhi::BufferHandle m_ToneMappingCB;
        nvrhi::BufferHandle m_HistogramBuffer;
        nvrhi::BufferHandle m_ExposureBuffer;
        float m_FrameTime = 0.f;

        nvrhi::TextureHandle m_ColorLUT;
        float m_ColorLUTSize = 0.f;

        nvrhi::BindingLayoutHandle m_HistogramBindingLayout;
        nvrhi::ComputePipelineHandle m_HistogramPso;

        nvrhi::BindingLayoutHandle m_ExposureBindingLayout;
        nvrhi::BindingSetHandle m_ExposureBindingSet;
        nvrhi::ComputePipelineHandle m_ExposurePso;

        nvrhi::BindingLayoutHandle m_RenderBindingLayout;
        nvrhi::GraphicsPipelineHandle m_RenderPso;

        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;
        nvrhi::AutoPtr<engine::FramebufferFactory> m_FramebufferFactory;

        std::unordered_map<nvrhi::ITexture*, nvrhi::BindingSetHandle> m_HistogramBindingSets;
        std::unordered_map<nvrhi::ITexture*, nvrhi::BindingSetHandle> m_RenderBindingSets;

    public:
        struct CreateParameters
        {
            bool isTextureArray = false;
            uint32_t histogramBins = 256;
            uint32_t numConstantBufferVersions = 16;
            nvrhi::IBuffer* exposureBufferOverride = nullptr;
            nvrhi::ITexture* colorLUT = nullptr;
        };

        ToneMappingPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory *shaderFactory,
            engine::CommonRenderPasses *commonPasses,
            engine::FramebufferFactory *framebufferFactory,
            const engine::ICompositeView& compositeView,
            const CreateParameters& params);

        // For a render graph: no target is bound and no buffer is made; Prepare creates the shaders and pipelines.
        // params.exposureBufferOverride is ignored.
        ToneMappingPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory *shaderFactory,
            engine::CommonRenderPasses *commonPasses,
            const CreateParameters& params);

        void Render(
            nvrhi::ICommandList* commandList,
            const ToneMappingParameters& params,
            const engine::ICompositeView& compositeView,
            nvrhi::ITexture* sourceTexture);

        void SimpleRender(
            nvrhi::ICommandList* commandList,
            const ToneMappingParameters& params,
            const engine::ICompositeView& compositeView,
            nvrhi::ITexture* sourceTexture);

        nvrhi::BufferHandle GetExposureBuffer();
        
        void AdvanceFrame(float frameTime);

        void ResetExposure(nvrhi::ICommandList* commandList, float initialExposure = 0.f);
        void ResetHistogram(nvrhi::ICommandList* commandList);
        void AddFrameToHistogram(nvrhi::ICommandList* commandList, const engine::ICompositeView& compositeView, nvrhi::ITexture* sourceTexture);
        void ComputeExposure(nvrhi::ICommandList* commandList, const ToneMappingParameters& params);

        ToneMappingGraphOutputs AddToGraph(IRenderGraph* graph, const ToneMappingGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:
        void CreateObjects(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory, const CreateParameters& params);
        void DispatchHistogram(nvrhi::ICommandList* commandList, const engine::ICompositeView& compositeView,
                               nvrhi::IBindingSet* bindingSet);
        void DispatchExposure(nvrhi::ICommandList* commandList, const ToneMappingParameters& params, float frameTime,
                              nvrhi::IBindingSet* bindingSet);
        void DrawView(nvrhi::ICommandList* commandList, const ToneMappingParameters& params, const engine::IView& view,
                      nvrhi::IFramebuffer* framebuffer, nvrhi::IGraphicsPipeline* pipeline, nvrhi::IBindingSet* bindingSet);

        // Graph construction.
        nvrhi::AutoPtr<engine::ShaderFactory> m_ShaderFactory;
        CreateParameters m_CreateParams;
        nvrhi::GraphicsPipelineHandle m_GraphRenderPso;
        engine::BindingCache m_GraphBindings;
        nvrhi::BufferDesc m_HistogramDesc;
        nvrhi::BufferDesc m_ExposureDesc;
        std::string m_ExposureName;
    };
}