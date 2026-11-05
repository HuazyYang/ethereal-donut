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
#include <nvrhi/nvrhi.h>
#include <donut/engine/BindingCache.h>
#include <nvrhi/core/foundation.h>
#include <donut/render/RenderGraph.h>

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
    // The pass in a render graph (design 13.3): one raster use per planar child view of `view`, drawing
    // `environmentMap` (a cube map or a lat-long texture, imported) into that view's subresources of `color`,
    // depth-tested against `depth` (read-only) when it is valid. `view` must stay valid until the graph has executed.
    struct EnvironmentMapGraphInputs
    {
        const engine::ICompositeView* view = nullptr;
        ResourceId color = InvalidResourceId;
        ResourceId depth = InvalidResourceId;
        ResourceId environmentMap = InvalidResourceId;
    };

    struct EnvironmentMapGraphOutputs
    {
        ResourceId color = InvalidResourceId;
    };

    class EnvironmentMapPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(EnvironmentMapPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:
        nvrhi::ShaderHandle m_PixelShader;
        nvrhi::BufferHandle m_SkyCB;
        nvrhi::BindingLayoutHandle m_RenderBindingLayout;
        nvrhi::BindingSetHandle m_RenderBindingSet;
        nvrhi::GraphicsPipelineHandle m_RenderPso;

        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;
        nvrhi::AutoPtr<engine::FramebufferFactory> m_FramebufferFactory;

    public:
        EnvironmentMapPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory *shaderFactory,
            engine::CommonRenderPasses *commonPasses,
            engine::FramebufferFactory *framebufferFactory,
            const engine::ICompositeView& compositeView,
            nvrhi::ITexture* environmentMap);

        void Render(
            nvrhi::ICommandList* commandList,
            const engine::ICompositeView& compositeView);

        // For a render graph: the map is chosen per AddToGraph; Prepare creates the shader and pipelines.
        EnvironmentMapPass(
            nvrhi::IDevice* device,
            engine::ShaderFactory* shaderFactory,
            engine::CommonRenderPasses* commonPasses);

        EnvironmentMapGraphOutputs AddToGraph(IRenderGraph* graph, const EnvironmentMapGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:
        void DrawView(nvrhi::ICommandList* commandList, const engine::IView& view, nvrhi::IFramebuffer* framebuffer,
                      nvrhi::IGraphicsPipeline* pipeline, nvrhi::IBindingSet* bindingSet);

        // Graph construction: one pixel shader per map kind, one pipeline per depth convention.
        nvrhi::DeviceHandle m_Device;
        nvrhi::AutoPtr<engine::ShaderFactory> m_ShaderFactory;
        int m_GraphCubeMap = -1;
        nvrhi::GraphicsPipelineHandle m_GraphPso[2];
        engine::BindingCache m_GraphBindings{ nullptr };
    };

}