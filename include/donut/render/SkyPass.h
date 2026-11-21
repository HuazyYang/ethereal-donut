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
#include <donut/render/RenderGraph.h>
#include <nvrhi/nvrhi.h>


namespace donut::engine
{
    class ShaderFactory;
    class ShadowMap;
    class CommonRenderPasses;
    class FramebufferFactory;
    class ICompositeView;
    class IView;
    class DirectionalLight;
}

struct ProceduralSkyShaderParameters;

namespace donut::render
{
    struct SkyParameters
    {
        dm::float3 skyColor{ 0.17f, 0.37f, 0.65f };
        dm::float3 horizonColor{ 0.50f, 0.70f, 0.92f };
        dm::float3 groundColor{ 0.62f, 0.59f, 0.55f };
        dm::float3 directionUp{ 0.f, 1.f, 0.f };
        float brightness = 0.1f; // scaler for sky brightness
        float horizonSize = 30.f; // +/- degrees
        float glowSize = 5.f; // degrees, starting from the edge of the light disk
        float glowIntensity = 0.1f; // [0-1] relative to light intensity
        float glowSharpness = 4.f; // [1-10] is the glow power exponent
        float maxLightRadiance = 100.f; // clamp for light radiance derived from its angular size, 0 = no clamp
    };

    // The pass in a render graph (design 13.3): one raster use per planar child view of `view`, drawing into that
    // view's subresources of `color`, depth-tested against `depth` (read-only) when it is valid. `view` and `light`
    // must stay valid until the graph has executed.
    struct SkyGraphInputs
    {
        const engine::ICompositeView* view = nullptr;
        ResourceId color = InvalidResourceId;
        ResourceId depth = InvalidResourceId;
        const engine::DirectionalLight* light = nullptr;
        SkyParameters params;
    };

    struct SkyGraphOutputs
    {
        ResourceId color = InvalidResourceId;
    };

    class SkyPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SkyPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:
        nvrhi::ShaderHandle m_PixelShader;
        nvrhi::BufferHandle m_SkyCB;
        nvrhi::BindingLayoutHandle m_RenderBindingLayout;
        nvrhi::BindingSetHandle m_RenderBindingSet;
        nvrhi::GraphicsPipelineHandle m_RenderPso;

        nvrhi::AutoPtr<engine::FramebufferFactory> m_FramebufferFactory;
        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;

        // Graph construction: the objects are made in Prepare, one pipeline per depth convention.
        nvrhi::DeviceHandle m_Device;
        nvrhi::AutoPtr<engine::ShaderFactory> m_ShaderFactory;
        nvrhi::GraphicsPipelineHandle m_GraphPso[2];

        void CreateCommonObjects(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory);
        nvrhi::GraphicsPipelineHandle CreatePipeline(nvrhi::IDevice* device, const nvrhi::FramebufferInfo& framebufferInfo,
                                                     bool reverseDepth) const;
        void RenderView(nvrhi::ICommandList* commandList, const engine::IView& view, nvrhi::IFramebuffer* framebuffer,
                        nvrhi::IGraphicsPipeline* pipeline, const engine::DirectionalLight& light,
                        const SkyParameters& params) const;

    public:
        SkyPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses,
            engine::FramebufferFactory* framebufferFactory,
            const engine::ICompositeView& compositeView);

        // For a render graph: no target is bound; Prepare creates the objects from the graph's framebuffer.
        SkyPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses);

        void Render(
            nvrhi::ICommandList* commandList,
            const engine::ICompositeView& compositeView,
            const engine::DirectionalLight& light,
            const SkyParameters& params) const;

        static void FillShaderParameters(
            const engine::DirectionalLight& light,
            const SkyParameters& input,
            ProceduralSkyShaderParameters& output);

        SkyGraphOutputs AddToGraph(IRenderGraph* graph, const SkyGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;
    };
}