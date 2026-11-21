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
#include <nvrhi/nvrhi.h>
#include <unordered_map>
#include <donut/engine/BindingCache.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraph.h>

namespace donut::engine
{
    class ShaderFactory;
    class Light;
    class CommonRenderPasses;
    class FramebufferFactory;
    class ICompositeView;
    class IView;
    struct LightProbe;
}

namespace donut::render
{
    class GBufferRenderTargets;

    // The pass in a render graph (design 13.2): one compute use over every planar child view of `view`. The ids
    // are textures of the graph; the optional ones may stay invalid and then bind the same defaults as Render.
    // `view`, `lights` and `lightProbes` must stay valid until the graph has executed. The lights' shadow map is
    // read through the lights, as in Render: import it and pass its id here so that the graph orders the read.
    struct DeferredLightingGraphInputs
    {
        const engine::ICompositeView* view = nullptr;
        ResourceId depth = InvalidResourceId;
        ResourceId gbufferNormals = InvalidResourceId;
        ResourceId gbufferDiffuse = InvalidResourceId;
        ResourceId gbufferSpecular = InvalidResourceId;
        ResourceId gbufferEmissive = InvalidResourceId;
        ResourceId indirectDiffuse = InvalidResourceId;
        ResourceId indirectSpecular = InvalidResourceId;
        ResourceId shadowChannels = InvalidResourceId;
        ResourceId ambientOcclusion = InvalidResourceId;
        ResourceId shadowMap = InvalidResourceId;
        ResourceId output = InvalidResourceId;
        const std::vector<nvrhi::AutoPtr<engine::Light>>* lights = nullptr;
        const std::vector<nvrhi::AutoPtr<engine::LightProbe>>* lightProbes = nullptr;
        dm::float3 ambientColorTop = 0.f;
        dm::float3 ambientColorBottom = 0.f;
        dm::float2 randomOffset = dm::float2::zero();
    };

    struct DeferredLightingGraphOutputs
    {
        ResourceId output = InvalidResourceId;
    };

    class DeferredLightingPass: public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DeferredLightingPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:
        nvrhi::DeviceHandle m_Device;

        nvrhi::ShaderHandle m_ComputeShader;
        nvrhi::SamplerHandle m_ShadowSampler;
        nvrhi::SamplerHandle m_ShadowSamplerComparison;
        nvrhi::BufferHandle m_DeferredLightingCB;
        nvrhi::ComputePipelineHandle m_Pso;

        nvrhi::BindingLayoutHandle m_BindingLayout;
        engine::BindingCache m_BindingSets;

        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;

    protected:

        virtual nvrhi::ShaderHandle CreateComputeShader(
            engine::ShaderFactory& shaderFactory);

    public:
        struct Inputs
        {
            nvrhi::ITexture* depth = nullptr;
            nvrhi::ITexture* gbufferNormals = nullptr;
            nvrhi::ITexture* gbufferDiffuse = nullptr;
            nvrhi::ITexture* gbufferSpecular = nullptr;
            nvrhi::ITexture* gbufferEmissive = nullptr;
            nvrhi::ITexture* indirectDiffuse = nullptr;
            nvrhi::ITexture* indirectSpecular = nullptr;
            nvrhi::ITexture* shadowChannels = nullptr;
            nvrhi::ITexture* ambientOcclusion = nullptr;
            nvrhi::ITexture* output = nullptr;

            const std::vector<nvrhi::AutoPtr<engine::Light>>* lights = nullptr;
            const std::vector<nvrhi::AutoPtr<engine::LightProbe>>* lightProbes = nullptr;

            dm::float3 ambientColorTop = 0.f;
            dm::float3 ambientColorBottom = 0.f;

            // Fills the GBuffer-related textures (depth, normals, etc.) from the provided structure.
            void SetGBuffer(const GBufferRenderTargets& targets);
        };

        DeferredLightingPass(
            nvrhi::IDevice* device,
            engine::CommonRenderPasses *commonPasses);

        virtual void Init(engine::ShaderFactory *shaderFactory);

        void Render(
            nvrhi::ICommandList* commandList,
            const engine::ICompositeView& compositeView,
            const Inputs& inputs,
            dm::float2 randomOffset = dm::float2::zero());

        void ResetBindingCache();

        // Render graph (13.2). Init must have run.
        DeferredLightingGraphOutputs AddToGraph(IRenderGraph* graph, const DeferredLightingGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;
    };
}
