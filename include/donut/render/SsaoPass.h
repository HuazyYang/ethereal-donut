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
    struct SsaoParameters
    {
        float amount = 2.f;
        float backgroundViewDepth = 100.f;
        float radiusWorld = 0.5f;
        float surfaceBias = 0.1f;
        float powerExponent = 2.f;
        bool enableBlur = true;
        float blurSharpness = 16.f;
    };

    // The pass in a render graph (design 13.3): one compute use in Restore mode. The deinterleaved quarter-resolution
    // arrays are pass-local transients sized from `depth`, so the graph pools and aliases them. `view` must stay
    // valid until the graph has executed.
    struct SsaoGraphInputs
    {
        const engine::ICompositeView* view = nullptr;
        ResourceId depth = InvalidResourceId;
        ResourceId normals = InvalidResourceId;
        ResourceId output = InvalidResourceId;
        SsaoParameters params;
        bool inputLinearDepth = false;
        bool octEncodedNormals = false;
        bool directionalOcclusion = false;
    };

    struct SsaoGraphOutputs
    {
        ResourceId output = InvalidResourceId;
    };

    class SsaoPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SsaoPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:
        struct SubPass
        {
            nvrhi::ShaderHandle Shader;
            nvrhi::BindingLayoutHandle BindingLayout;
            std::vector<nvrhi::BindingSetHandle> BindingSets;
            nvrhi::ComputePipelineHandle Pipeline;
        };

        SubPass m_Deinterleave;
        SubPass m_Compute;
        SubPass m_Blur;

        nvrhi::DeviceHandle m_Device;
        nvrhi::BufferHandle m_ConstantBuffer;

        nvrhi::TextureHandle m_DeinterleavedDepth;
        nvrhi::TextureHandle m_DeinterleavedOcclusion;
        dm::float2 m_QuantizedGbufferTextureSize;
                
        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;

    public:
        struct CreateParameters
        {
            dm::int2 dimensions = 0;
            bool inputLinearDepth = false;
            bool octEncodedNormals = false;
            bool directionalOcclusion = false;
            int numBindingSets = 1;
        };

        SsaoPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses,
            const CreateParameters& params);

        SsaoPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses,
            nvrhi::ITexture* gbufferDepth,
            nvrhi::ITexture* gbufferNormals,
            nvrhi::ITexture* destinationTexture);

        void CreateBindingSet(
            nvrhi::ITexture* gbufferDepth,
            nvrhi::ITexture* gbufferNormals,
            nvrhi::ITexture* destinationTexture,
            int bindingSetIndex = 0);

        void Render(
            nvrhi::ICommandList* commandList,
            const SsaoParameters& params,
            const engine::ICompositeView& compositeView,
            int bindingSetIndex = 0);

        // For a render graph: Prepare creates the shaders and pipelines of each use's variant.
        SsaoPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses);

        SsaoGraphOutputs AddToGraph(IRenderGraph* graph, const SsaoGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:
        void CreatePipelines(engine::ShaderFactory* shaderFactory, bool inputLinearDepth, bool octEncodedNormals,
                             bool directionalOcclusion);
        void DispatchViews(nvrhi::ICommandList* commandList, const SsaoParameters& params,
                           const engine::ICompositeView& compositeView, nvrhi::IBindingSet* deinterleave,
                           nvrhi::IBindingSet* compute, nvrhi::IBindingSet* blur, dm::float2 quantizedGbufferTextureSize);

        // Graph construction.
        nvrhi::AutoPtr<engine::ShaderFactory> m_ShaderFactory;
        int m_GraphVariant = -1;
        engine::BindingCache m_GraphBindings;
    };
}