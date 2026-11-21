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
#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <donut/render/RenderGraph.h>
#include <nvrhi/nvrhi.h>
#include <string>

namespace donut::engine
{
    class ShaderFactory;
    class ShadowMap;
    class CommonRenderPasses;
    class FramebufferFactory;
    class ICompositeView;
    class IView;
}

namespace donut::render
{
    enum class TemporalAntiAliasingJitter
    {
        MSAA,
        Halton,
        R2,
        WhiteNoise
    };

    struct TemporalAntiAliasingParameters
    {
        float newFrameWeight = 0.1f;
        float clampingFactor = 1.0f;
        float maxRadiance = 10000.f;
        bool enableHistoryClamping = true;

        // Requires CreateParameters::historyClampRelax single channel [0, 1] mask to be provided. 
        // For texels with mask value of 0 the behavior is unchanged; for texels with mask value > 0, 
        // 'newFrameWeight' will be reduced and 'clampingFactor' will be increased proportionally. 
        bool useHistoryClampRelax = false;
    };

    // The pass in a render graph (design 13.3). With `viewPrevious` and `depth`, one raster use per planar child
    // view of `view` writes that view's motion vectors; then one compute use resolves `unresolvedColor` into
    // `resolvedColor`. The feedback is a two-layer history of this pass object (GetHistoryPair), and a new
    // previous layer resolves without history. The views must stay valid until the graph has executed.
    struct TemporalAntiAliasingGraphInputs
    {
        const engine::ICompositeView* view = nullptr;           // the input (render) resolution
        const engine::ICompositeView* viewPrevious = nullptr;   // for motion vectors
        const engine::ICompositeView* viewOutput = nullptr;     // the output resolution; nullptr: view
        ResourceId depth = InvalidResourceId;
        ResourceId motionVectors = InvalidResourceId;
        ResourceId unresolvedColor = InvalidResourceId;
        ResourceId resolvedColor = InvalidResourceId;
        ResourceId historyClampRelax = InvalidResourceId;
        nvrhi::Format feedbackFormat = nvrhi::Format::RGBA16_FLOAT;
        TemporalAntiAliasingParameters params;
        dm::float3 preViewTranslationDifference = 0.f;
    };

    struct TemporalAntiAliasingGraphOutputs
    {
        ResourceId motionVectors = InvalidResourceId;
        ResourceId resolvedColor = InvalidResourceId;
    };

    class TemporalAntiAliasingPass: public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(TemporalAntiAliasingPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:
        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;

        nvrhi::ShaderHandle m_MotionVectorPS;
        nvrhi::ShaderHandle m_TemporalAntiAliasingCS;
        nvrhi::SamplerHandle m_BilinearSampler;
        nvrhi::BufferHandle m_TemporalAntiAliasingCB;

        nvrhi::BindingLayoutHandle m_MotionVectorsBindingLayout;
        nvrhi::BindingSetHandle m_MotionVectorsBindingSet;
        nvrhi::GraphicsPipelineHandle m_MotionVectorsPso;
        nvrhi::AutoPtr<engine::FramebufferFactory> m_MotionVectorsFramebufferFactory;

        nvrhi::BindingLayoutHandle m_ResolveBindingLayout;
        nvrhi::BindingSetHandle m_ResolveBindingSet;
        nvrhi::BindingSetHandle m_ResolveBindingSetPrevious;
        nvrhi::ComputePipelineHandle m_ResolvePso;

        uint32_t m_FrameIndex;
        uint32_t m_StencilMask;
        dm::float2 m_ResolvedColorSize;

        dm::float2 m_R2Jitter;
        TemporalAntiAliasingJitter m_Jitter;

        bool m_HasHistoryClampRelaxTexture;

    public:
        struct CreateParameters
        {
            nvrhi::ITexture* sourceDepth = nullptr;
            nvrhi::ITexture* motionVectors = nullptr;
            nvrhi::ITexture* unresolvedColor = nullptr;
            nvrhi::ITexture* resolvedColor = nullptr;
            nvrhi::ITexture* feedback1 = nullptr;
            nvrhi::ITexture* feedback2 = nullptr;
            nvrhi::ITexture* historyClampRelax = nullptr;
            bool useCatmullRomFilter = true;
            uint32_t motionVectorStencilMask = 0;
            uint32_t numConstantBufferVersions = 16;
        };

        TemporalAntiAliasingPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses,
            const engine::ICompositeView& compositeView,
            const CreateParameters& params);

        // For a render graph: the textures of `params` are ignored; Prepare creates the shaders and pipelines.
        TemporalAntiAliasingPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses,
            const CreateParameters& params);

        void RenderMotionVectors(
            nvrhi::ICommandList* commandList,
            const engine::ICompositeView& compositeView,
            const engine::ICompositeView& compositeViewPrevious,
            dm::float3 preViewTranslationDifference = dm::float3::zero());

        void TemporalResolve(
            nvrhi::ICommandList* commandList,
            const TemporalAntiAliasingParameters& params,
            bool feedbackIsValid,
            const engine::ICompositeView& compositeViewInput,
            const engine::ICompositeView& compositeViewOutput);

        void AdvanceFrame();
        void SetJitter(TemporalAntiAliasingJitter jitter);
        dm::float2 GetCurrentPixelOffset();

        TemporalAntiAliasingGraphOutputs AddToGraph(IRenderGraph* graph, const TemporalAntiAliasingGraphInputs& inputs);
        // AddToGraph's two stages, for a frame that uses the motion vectors before the resolve (variable-rate shading
        // reads them before the scene is drawn). Each returns the new version of what it writes; the resolve reads
        // inputs.motionVectors as given.
        ResourceId AddMotionVectorsToGraph(IRenderGraph* graph, const TemporalAntiAliasingGraphInputs& inputs);
        ResourceId AddResolveToGraph(IRenderGraph* graph, const TemporalAntiAliasingGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:
        void DrawMotionVectors(nvrhi::ICommandList* commandList, const engine::IView& view, const engine::IView& viewPrevious,
                               nvrhi::IFramebuffer* framebuffer, nvrhi::IGraphicsPipeline* pipeline,
                               nvrhi::IBindingSet* bindingSet, dm::float3 preViewTranslationDifference);
        void ResolveViews(nvrhi::ICommandList* commandList, const TemporalAntiAliasingParameters& params, bool feedbackIsValid,
                          const engine::ICompositeView& compositeViewInput, const engine::ICompositeView& compositeViewOutput,
                          nvrhi::IBindingSet* bindingSet, dm::float2 resolvedColorSize, bool hasHistoryClampRelax);

        // Graph construction: shader variants follow the graph's textures.
        nvrhi::DeviceHandle m_Device;
        nvrhi::AutoPtr<engine::ShaderFactory> m_ShaderFactory;
        bool m_UseCatmullRomFilter = true;
        uint32_t m_NumConstantBufferVersions = 16;
        uint32_t m_GraphSampleCount = 0;
        int m_GraphUseStencil = -1;
        nvrhi::GraphicsPipelineHandle m_GraphMotionVectorsPso;
        engine::BindingCache m_GraphBindings;
        std::string m_FeedbackName;
    };
}
