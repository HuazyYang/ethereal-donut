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

#include <donut/engine/View.h>
#include <donut/render/RenderGraph.h>
#include <nvrhi/nvrhi.h>

namespace donut::engine
{
    class SceneGraphNode;
    struct MeshInfo;
    struct MeshGeometry;
    class MeshInstance;
    struct Material;
    struct BufferGroup;
    class FramebufferFactory;
}

namespace donut::render
{
    class IDrawStrategy;

    // DrawItem::instanceSlot of an item that takes its slot from `instance` (MeshInstance::GetInstanceIndex).
    constexpr uint32_t c_DrawItemSlotFromInstance = ~0u;

    struct DrawItem
    {
        const engine::MeshInstance* instance;   // null for a runtime-only mesh instance (SceneComponents.h)
        const engine::MeshInfo* mesh;
        const engine::MeshGeometry* geometry;
        const engine::Material* material;
        const engine::BufferGroup* buffers;
        float distanceToCamera;
        nvrhi::RasterCullMode cullMode;
        void* userData;
        // The instance's row in the instance buffer (GpuInstanceSlot::instance), which RenderView draws with: the
        // packet adapters set it from the snapshot for every item, so that recording reads no leaf (design 7.5; plan
        // step E7, item E7.2). Items made by hand or by walking strategies keep the default and take the slot from
        // `instance`, which must then be set.
        uint32_t instanceSlot = c_DrawItemSlotFromInstance;
    };

    class GeometryPassContext
    {
    };

    // A geometry pass in a render graph (design 13.2, 13.3): one use per child view of `view`, drawn into that
    // view's subresources of the targets. The targets are contiguous colour slots from 0 and an optional depth
    // target. `view`, `viewPrev`, `rootNode`, `drawStrategy` and `passContext` must stay valid until the graph
    // has executed.
    struct GeometryGraphInputs
    {
        const char* name = "Geometry";
        const engine::ICompositeView* view = nullptr;
        const engine::ICompositeView* viewPrev = nullptr;
        engine::SceneGraphNode* rootNode = nullptr;
        // Prepared for each child view in turn. Strategies are stateful: give each IGeometryPass object its own (the
        // uses of two pass objects may record on two threads at once; DrawStrategy.h).
        IDrawStrategy* drawStrategy = nullptr;
        GeometryPassContext* passContext = nullptr;
        ResourceId colorTargets[8] = { InvalidResourceId, InvalidResourceId, InvalidResourceId, InvalidResourceId,
                                       InvalidResourceId, InvalidResourceId, InvalidResourceId, InvalidResourceId };
        uint32_t colorTargetCount = 0;
        ResourceId depth = InvalidResourceId;
        bool depthWrite = true;
        // Clears every target of each use first (the targets of transients must be cleared or fully written).
        bool clearTargets = false;
        ClearValue colorClear = { { 0.f, 0.f, 0.f, 0.f }, 0.f, 0 };
        ClearValue depthClear = { { 0.f, 0.f, 0.f, 0.f }, 0.f, 0 };
        // Textures the shaders sample (a shadow map for forward shading), read in the pixel shader.
        ResourceId shaderReads[4] = { InvalidResourceId, InvalidResourceId, InvalidResourceId, InvalidResourceId };
        uint32_t shaderReadCount = 0;
        bool materialEvents = false;
        // Buffers the draws read (plan step E5, sub-plan item E5.3, open point 5): the scene's instance, geometry
        // and material buffers and the skinned vertex buffers (SceneFrame::bufferReads), declared as reads in the
        // vertex and pixel stages (a buffer made with isVertexBuffer as a vertex-buffer and SRV read, any other as an
        // SRV read), so that the graph orders the pass after the upload and skinning passes that write them. With
        // buffer reads the uses are Tracked and Restore: the body's tracker may move a buffer from the combined read
        // state into the one state a binding asks for, and Restore returns it (an id that names no buffer is skipped,
        // and without any read declared the uses stay Tracked). Read only while AddToGraph runs; none by default.
        const ResourceId* bufferReads = nullptr;
        uint32_t bufferReadCount = 0;
    };

    struct GeometryGraphOutputs
    {
        ResourceId colorTargets[8];
        ResourceId depth;
    };

    // The base of the geometry passes. Its IGpuPass implementation draws one view per use, in Tracked mode: the
    // scene's buffers and material textures stay under nvrhi's automatic barriers (13.3).
    class IGeometryPass: public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(IGeometryPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        GeometryGraphOutputs AddToGraph(IRenderGraph* graph, const GeometryGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

        [[nodiscard]] virtual engine::ViewType::Enum GetSupportedViewTypes() const = 0;
        virtual void SetupView(GeometryPassContext& context, nvrhi::ICommandList* commandList, const engine::IView* view, const engine::IView* viewPrev) = 0;
        virtual bool SetupMaterial(GeometryPassContext& context, const engine::Material* material, nvrhi::RasterCullMode cullMode, nvrhi::GraphicsState& state) = 0;
        virtual void SetupInputBuffers(GeometryPassContext& context, const engine::BufferGroup* buffers, nvrhi::GraphicsState& state) = 0;
        virtual void SetPushConstants(GeometryPassContext& context, nvrhi::ICommandList* commandList, nvrhi::GraphicsState& state, nvrhi::DrawArguments& args) = 0;
        virtual ~IGeometryPass() = default;
    };

    void RenderView(
        nvrhi::ICommandList* commandList, 
        const engine::IView* view, 
        const engine::IView* viewPrev, 
        nvrhi::IFramebuffer* framebuffer,
        IDrawStrategy& drawStrategy,
        IGeometryPass& pass,
        GeometryPassContext& passContext,
        bool materialEvents = false);

    void RenderCompositeView(
        nvrhi::ICommandList* commandList,
        const engine::ICompositeView* compositeView,
        const engine::ICompositeView* compositeViewPrev,
        engine::FramebufferFactory* framebufferFactory,
        engine::SceneGraphNode* rootNode,
        IDrawStrategy& drawStrategy,
        IGeometryPass& pass,
        GeometryPassContext& passContext,
        const char* passEvent = nullptr,
        bool materialEvents = false);
}