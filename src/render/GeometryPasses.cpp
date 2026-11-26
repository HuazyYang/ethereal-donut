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

#include <donut/render/GeometryPasses.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/render/DrawStrategy.h>

#include <algorithm>
#include <type_traits>

using namespace donut::math;
using namespace donut::engine;
using namespace donut::render;

void donut::render::RenderView(
    nvrhi::ICommandList* commandList, 
    const IView* view, 
    const IView* viewPrev,
    nvrhi::IFramebuffer* framebuffer,
    IDrawStrategy& drawStrategy,
    IGeometryPass& pass,
    GeometryPassContext& passContext,
    bool materialEvents)
{
    pass.SetupView(passContext, commandList, view, viewPrev);

    const Material* lastMaterial = nullptr;
    const BufferGroup* lastBuffers = nullptr;
    nvrhi::RasterCullMode lastCullMode = nvrhi::RasterCullMode::Back;

    bool drawMaterial = true;
    bool stateValid = false;

    const Material* eventMaterial = nullptr;

    nvrhi::GraphicsState graphicsState;
    graphicsState.framebuffer = framebuffer;
    graphicsState.viewport = view->GetViewportState();
    graphicsState.shadingRateState = view->GetVariableRateShadingState();

    nvrhi::DrawArguments currentDraw;
    currentDraw.instanceCount = 0;

    auto flushDraw = [commandList, materialEvents, &graphicsState, &currentDraw, &eventMaterial, &pass, &passContext](const Material* material)
    {
        if (currentDraw.instanceCount == 0)
            return;

        if (materialEvents && material != eventMaterial)
        {
            if (eventMaterial)
                commandList->endMarker();

            if (material->name.empty())
            {
                eventMaterial = nullptr;
            }
            else
            {
                commandList->beginMarker(material->name.c_str());
                eventMaterial = material;
            }
        }

        pass.SetPushConstants(passContext, commandList, graphicsState, currentDraw);

        commandList->drawIndexed(currentDraw);
        currentDraw.instanceCount = 0;
    };
    
    while (const DrawItem* item = drawStrategy.GetNextItem())
    {
        if (item->material == nullptr)
            continue;


        bool newBuffers = item->buffers != lastBuffers;
        bool newMaterial = item->material != lastMaterial || item->cullMode != lastCullMode;

        if (newBuffers || newMaterial)
        {
            flushDraw(lastMaterial);
        }

        if (newBuffers)
        {
            pass.SetupInputBuffers(passContext, item->buffers, graphicsState);

            lastBuffers = item->buffers;
            stateValid = false;
        }

        if (newMaterial)
        {
            drawMaterial = pass.SetupMaterial(passContext, item->material, item->cullMode, graphicsState);

            lastMaterial = item->material;
            lastCullMode = item->cullMode;
            stateValid = false;
        }

        if (drawMaterial)
        {
            if (!stateValid)
            {
                commandList->setGraphicsState(graphicsState);
                stateValid = true;
            }

            nvrhi::DrawArguments args;
            args.vertexCount = item->geometry->numIndices;
            args.instanceCount = 1;
            args.startVertexLocation = item->mesh->vertexOffset + item->geometry->vertexOffsetInMesh;
            args.startIndexLocation = item->mesh->indexOffset + item->geometry->indexOffsetInMesh;
            // The packet adapters give the slot (and no leaf for a runtime-only row); hand-made and walked items
            // keep c_DrawItemSlotFromInstance and take it from their leaf.
            args.startInstanceLocation = item->instanceSlot != c_DrawItemSlotFromInstance
                ? item->instanceSlot : uint32_t(item->instance->GetInstanceIndex());

            if (currentDraw.instanceCount > 0 && 
                currentDraw.startIndexLocation == args.startIndexLocation && 
                currentDraw.startInstanceLocation + currentDraw.instanceCount == args.startInstanceLocation)
            {
                currentDraw.instanceCount += 1;
            }
            else
            {
                flushDraw(item->material);

                currentDraw = args;
            }
        }
    }

    flushDraw(lastMaterial);

    if (materialEvents && eventMaterial)
        commandList->endMarker();
}

void donut::render::RenderCompositeView(
    nvrhi::ICommandList* commandList, 
    const ICompositeView* compositeView, 
    const ICompositeView* compositeViewPrev, 
    FramebufferFactory* framebufferFactory,
    engine::SceneGraphNode* rootNode,
    IDrawStrategy& drawStrategy,
    IGeometryPass& pass,
    GeometryPassContext& passContext,
    const char* passEvent, 
    bool materialEvents)
{
    if (passEvent)
        commandList->beginMarker(passEvent);

    ViewType::Enum supportedViewTypes = pass.GetSupportedViewTypes();

    if (compositeViewPrev)
    {
        // the views must have the same topology
        assert(compositeView->GetNumChildViews(supportedViewTypes) == compositeViewPrev->GetNumChildViews(supportedViewTypes));
    }
    
    for (uint viewIndex = 0; viewIndex < compositeView->GetNumChildViews(supportedViewTypes); viewIndex++)
    {
        const IView* view = compositeView->GetChildView(supportedViewTypes, viewIndex);
        const IView* viewPrev = compositeViewPrev ? compositeViewPrev->GetChildView(supportedViewTypes, viewIndex) : nullptr;

        assert(view != nullptr);

        drawStrategy.PrepareForView(rootNode, *view);

        nvrhi::IFramebuffer* framebuffer = framebufferFactory->GetFramebuffer(*view);

        RenderView(commandList, view, viewPrev, framebuffer, drawStrategy, pass, passContext, materialEvents);
    }

    if (passEvent)
        commandList->endMarker();
}

// ------------------------------------------------------------------------------------------ render graph (13.2)

namespace
{
    // One use: one child view. Trivially destructible: it lives in the graph's frame arena.
    struct GeometryGraphUse
    {
        const IView* view;
        const IView* viewPrev;
        SceneGraphNode* rootNode;
        IDrawStrategy* drawStrategy;
        GeometryPassContext* passContext;
        const char* name;
        donut::render::ResourceId colorTargets[8];
        uint32_t colorTargetCount;
        donut::render::ResourceId depth;
        donut::render::ResourceId shaderReads[4];
        uint32_t shaderReadCount;
        // GeometryGraphInputs::bufferReads and the access of each (none for an id that names no buffer); read only by
        // Setup, which AddPass calls at once.
        const donut::render::ResourceId* bufferReads;
        const donut::render::Access* bufferAccesses;
        uint32_t bufferReadCount;
        nvrhi::SubresourceRegion region;
        donut::render::ClearValue colorClear;
        donut::render::ClearValue depthClear;
        bool depthWrite;
        bool clearTargets;
        bool materialEvents;
    };
    static_assert(std::is_trivially_destructible_v<GeometryGraphUse>);
}

donut::render::GeometryGraphOutputs IGeometryPass::AddToGraph(IRenderGraph* graph, const GeometryGraphInputs& inputs)
{
    GeometryGraphOutputs outputs;
    for (uint32_t i = 0; i < 8; ++i)
        outputs.colorTargets[i] = inputs.colorTargets[i];
    outputs.depth = inputs.depth;
    if (!inputs.view || !inputs.drawStrategy || !inputs.passContext)
        return outputs;

    // Buffer reads (open point 5): a buffer made for vertex input is read as a vertex buffer and as an SRV (the body
    // binds it either way: input assembler or vertex pulling), any other buffer as an SRV. The declared states are then
    // the resting states of the graph path's buffers: VertexBuffer | ShaderResource for the skinned vertex buffers
    // (engine::c_SkinnedVertexBufferRestingState), ShaderResource for the geometry and material buffers
    // (engine::c_GpuSceneRestingState); the instance buffer, a vertex buffer too, is read in the combined state.
    Access* bufferAccesses = nullptr;
    uint32_t bufferReadCount = 0;
    uint32_t declaredReads = 0;
    if (inputs.bufferReads && inputs.bufferReadCount)
    {
        bufferAccesses = static_cast<Access*>(
            graph->AllocateFrameData(sizeof(Access) * inputs.bufferReadCount, alignof(Access)));
        if (!bufferAccesses)
            return outputs;
        bufferReadCount = inputs.bufferReadCount;
        for (uint32_t i = 0; i < bufferReadCount; ++i)
        {
            const ResourceId id = inputs.bufferReads[i];
            nvrhi::BufferDesc desc;
            bufferAccesses[i] = Access::None;
            if (id != InvalidResourceId && graph->GetBufferDesc(id, desc) == nvrhi::FS_OK)
            {
                bufferAccesses[i] = desc.isVertexBuffer ? Access::VertexBuffer | Access::ShaderRead : Access::ShaderRead;
                ++declaredReads;
            }
        }
    }
    // The body binds these buffers through nvrhi's tracker, which asks for one exact state per binding (an SRV, or a
    // vertex buffer) and so moves a buffer out of the combined state the plan put it in. Restore (Tracked, and the
    // declared ranges returned to the plan's states after the body) keeps the plan's view of every state true; it also
    // turns off the Debug check of Tracked bodies for the use. Without a buffer read declared the pass stays Tracked,
    // as before. (Declaring the exact state the body binds, asked of the pass, would keep it Tracked: a follow-up.)
    const PassFlags passFlags = declaredReads ? PassFlags::Tracked | PassFlags::Restore : PassFlags::Tracked;

    const ViewType::Enum supportedViewTypes = GetSupportedViewTypes();
    const uint32_t viewCount = inputs.view->GetNumChildViews(supportedViewTypes);
    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        GeometryGraphUse* use = NewFrameData<GeometryGraphUse>(graph);
        if (!use)
            break;
        use->view = inputs.view->GetChildView(supportedViewTypes, viewIndex);
        use->viewPrev = inputs.viewPrev ? inputs.viewPrev->GetChildView(supportedViewTypes, viewIndex) : nullptr;
        use->rootNode = inputs.rootNode;
        use->drawStrategy = inputs.drawStrategy;
        use->passContext = inputs.passContext;
        use->name = inputs.name;
        use->colorTargetCount = std::min<uint32_t>(inputs.colorTargetCount, 8);
        for (uint32_t i = 0; i < 8; ++i)
            use->colorTargets[i] = outputs.colorTargets[i];
        use->depth = outputs.depth;
        use->shaderReadCount = std::min<uint32_t>(inputs.shaderReadCount, 4);
        for (uint32_t i = 0; i < 4; ++i)
            use->shaderReads[i] = inputs.shaderReads[i];
        use->bufferReads = inputs.bufferReads;
        use->bufferAccesses = bufferAccesses;
        use->bufferReadCount = bufferReadCount;
        const nvrhi::TextureSubresourceSet subresources = use->view->GetSubresources();
        use->region = nvrhi::SubresourceRegion().setMips(subresources.baseMipLevel, 1)
            .setSlices(subresources.baseArraySlice, subresources.numArraySlices);
        use->colorClear = inputs.colorClear;
        use->depthClear = inputs.depthClear;
        use->depthWrite = inputs.depthWrite;
        use->clearTargets = inputs.clearTargets;
        use->materialEvents = inputs.materialEvents;

        // AddPass calls Setup, which writes the new versions into the use; the next view continues from them.
        graph->AddPass(PassDesc{ inputs.name, PassType::Raster, passFlags, 0 }, this, use, nullptr);
        for (uint32_t i = 0; i < use->colorTargetCount; ++i)
            outputs.colorTargets[i] = use->colorTargets[i];
        outputs.depth = use->depth;
    }
    return outputs;
}

nvrhi::FRESULT IGeometryPass::Setup(ISetupContext* context, void* data) noexcept
{
    GeometryGraphUse& use = *static_cast<GeometryGraphUse*>(data);
    const AccessFlags clearFlag = use.clearTargets ? AccessFlags::Clear : AccessFlags::None;
    for (uint32_t i = 0; i < use.shaderReadCount; ++i)
    {
        if (use.shaderReads[i] != InvalidResourceId)
            context->Read(use.shaderReads[i], Access::ShaderRead, nvrhi::ShaderType::Pixel, nvrhi::SubresourceRegion());
    }
    for (uint32_t i = 0; i < use.bufferReadCount; ++i)
    {
        if (use.bufferAccesses[i] != Access::None)
        {
            context->Read(use.bufferReads[i], use.bufferAccesses[i],
                          nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel, nvrhi::SubresourceRegion());
        }
    }
    for (uint32_t i = 0; i < use.colorTargetCount; ++i)
    {
        use.colorTargets[i] = context->SetColorTarget(i, use.colorTargets[i], use.region, clearFlag,
                                                      use.clearTargets ? &use.colorClear : nullptr);
    }
    if (use.depth != InvalidResourceId)
    {
        const bool clearDepth = use.clearTargets && use.depthWrite;
        use.depth = context->SetDepthTarget(use.depth, use.region, !use.depthWrite,
                                            clearDepth ? AccessFlags::Clear : AccessFlags::None,
                                            clearDepth ? &use.depthClear : nullptr);
    }
    return nvrhi::FS_OK;
}

nvrhi::FRESULT IGeometryPass::Prepare(IPrepareContext*, void*) noexcept
{
    return nvrhi::FS_OK;
}

nvrhi::FRESULT IGeometryPass::Execute(IPassContext* context, const void* data) noexcept
{
    const GeometryGraphUse& use = *static_cast<const GeometryGraphUse*>(data);
    nvrhi::IFramebuffer* framebuffer = context->GetFramebuffer();
    if (!framebuffer)
        return nvrhi::FE_INVALID_ARGS;
    use.drawStrategy->PrepareForView(use.rootNode, *use.view);
    RenderView(context->GetCommandList(), use.view, use.viewPrev, framebuffer, *use.drawStrategy, *this,
               *use.passContext, use.materialEvents);
    return nvrhi::FS_OK;
}

void IGeometryPass::OnResourcesReleased() noexcept
{
}
