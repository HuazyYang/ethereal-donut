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

#include <donut/engine/SceneGraph.h>
#include <donut/render/GeometryPasses.h>
#include <atomic>
#include <cstdint>
#include <vector>

namespace donut::engine
{
    class IView;
}

namespace donut::render
{
    struct RenderSnapshot;
    enum class DrawPacketPass : uint8_t;   // RenderExtract.h

    // Draw strategies are stateful (PrepareForView, then GetNextItem until null), so one object must not be used from
    // two threads at once. Give each IGeometryPass object its own strategy object: a render graph records the uses of
    // one pass object on one thread, but the uses of two pass objects may record in parallel. In Debug builds this
    // check logs one warning per object when PrepareForView and GetNextItem calls of two threads interleave.
    class DrawStrategyUseCheck
    {
    public:
        void Prepare() noexcept
        {
#ifndef NDEBUG
            OnPrepare();
#endif
        }
        void Next() noexcept
        {
#ifndef NDEBUG
            OnNext();
#endif
        }

    private:
        void OnPrepare() noexcept;
        void OnNext() noexcept;
        std::atomic<uint64_t> m_Owner{ 0 };     // the thread of the last Prepare
        std::atomic<uint32_t> m_Preparing{ 0 };
        std::atomic<bool> m_Warned{ false };
    };

    // DrawItems from the draw packets of one pass of a render snapshot, for the view a PrepareForView names, in key
    // order (plan step E5, sub-plan item E5.3; design 7.3: the strategies become adapters over packets). The view is
    // found by its address among the snapshot's ExtractedView::view, so the child views of a composite view, which
    // IGeometryPass::AddToGraph prepares in turn on one strategy object, each get their own packets. `instance` is the
    // packet's leaf, `mesh` the instance's MeshInfo from the snapshot (VisibleInstances::meshInfos), `geometry` and
    // `material` by index into the mesh's geometries (read when the item is made), `buffers` the mesh's,
    // `distanceToCamera` from a transparent key. Recording reads the snapshot and the objects it points to; RenderView
    // also calls the leaf's GetInstanceIndex() (design 7.5: pipelining must keep those objects unchanged until the
    // frame has recorded). Each item is made when Next returns it, into one item of this object: its pointer is valid
    // until the next call.
    //
    // The snapshot lives in the render graph's frame arena and dies at the next IRenderGraph::Reset: bind again every
    // frame, after RenderService::Extract. Debug builds fill the arena with a pattern at Reset, and Prepare refuses a
    // snapshot whose view count is not plausible (one warning), which catches most strategies left bound to an old
    // frame.
    class PacketDrawItems
    {
    public:
        // Null `snapshot` unbinds.
        void Bind(const RenderSnapshot* snapshot, DrawPacketPass pass) noexcept;
        [[nodiscard]] bool IsBound() const noexcept { return m_Snapshot != nullptr; }
        // Selects the packets of `view` (PrepareForView). A view the snapshot does not hold gives no item and logs a
        // warning, once per view and Bind.
        void Prepare(const engine::IView& view);
        // Null after the last item.
        const DrawItem* Next() noexcept;

    private:
        const RenderSnapshot* m_Snapshot = nullptr;
        DrawPacketPass m_Pass = DrawPacketPass(0);
        uint32_t m_Next = 0;
        uint32_t m_End = 0;
        DrawItem m_Item = {};
        std::vector<const engine::IView*> m_WarnedViews;
        DrawStrategyUseCheck m_UseCheck;
    };

    class IDrawStrategy: public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(IDrawStrategy)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        virtual void PrepareForView(
            engine::SceneGraphNode* rootNode,
            const engine::IView& view) = 0;

        virtual const DrawItem* GetNextItem() = 0;

        virtual ~IDrawStrategy() = default;
    };

    class PassthroughDrawStrategy : public IDrawStrategy
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    private:
        const DrawItem* m_Data = nullptr;
        size_t m_Count = 0;

    public:
        void PrepareForView(
            engine::SceneGraphNode* rootNode,
            const engine::IView& view) override { }

        const DrawItem* GetNextItem() override;

        void SetData(const DrawItem* data, size_t count);
    };
    
    class InstancedOpaqueDrawStrategy : public IDrawStrategy
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    private:
        dm::frustum m_ViewFrustum;
        engine::SceneGraphWalker m_Walker;
        std::vector<DrawItem> m_InstanceChunk;
        std::vector<const DrawItem*> m_InstancePtrChunk;
        size_t m_ReadPtr = 0;
        size_t m_ChunkSize = 128;

        void FillChunk();

    public:

        void PrepareForView(
            engine::SceneGraphNode* rootNode,
            const engine::IView& view) override;

        const DrawItem* GetNextItem() override;

        [[nodiscard]] size_t GetChunkSize() const { return m_ChunkSize; }
        void SetChunkSize(size_t size) { m_ChunkSize = std::max<size_t>(size, 1u); }

        // Binds the strategy to the opaque packets of `snapshot` (RenderExtract.h): PrepareForView then ignores its
        // root node and takes the packets of the view it is given (found by address). Null returns it to the graph walk.
        // The snapshot dies at the next IRenderGraph::Reset: bind again every frame, after RenderService::Extract.
        void SetPackets(const RenderSnapshot* snapshot);

    private:
        PacketDrawItems m_Packets;
        DrawStrategyUseCheck m_UseCheck;
    };

    class TransparentDrawStrategy : public IDrawStrategy
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    private:
        std::vector<DrawItem> m_InstancesToDraw;
        std::vector<const DrawItem*> m_InstancePtrsToDraw;
        size_t m_ReadPtr = 0;
        PacketDrawItems m_Packets;
        DrawStrategyUseCheck m_UseCheck;

    public:
        // Bound to packets, the setting is ExtractDesc::drawDoubleSidedMaterialsSeparately instead.
        bool DrawDoubleSidedMaterialsSeparately = true;
        
        void PrepareForView(
            engine::SceneGraphNode* rootNode,
            const engine::IView& view) override;

        const DrawItem* GetNextItem() override;

        // Binds the strategy to the transparent packets of `snapshot` (RenderExtract.h): PrepareForView then ignores
        // its root node and takes the packets of the view it is given (found by address). Null returns it to the graph
        // walk. The snapshot dies at the next IRenderGraph::Reset: bind again every frame, after RenderService::Extract.
        void SetPackets(const RenderSnapshot* snapshot);
    };

    // An IDrawStrategy over the packets of one pass of a snapshot (plan step E5, sub-plan item E5.3). PrepareForView
    // ignores its root node and takes the packets of the view it is given (found by address); unbound, it draws nothing.
    class PacketDrawStrategy : public IDrawStrategy
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    private:
        PacketDrawItems m_Packets;

    public:
        // The snapshot dies at the next IRenderGraph::Reset: bind again every frame, after RenderService::Extract.
        void SetPackets(const RenderSnapshot* snapshot, DrawPacketPass pass);

        void PrepareForView(
            engine::SceneGraphNode* rootNode,
            const engine::IView& view) override;

        const DrawItem* GetNextItem() override;
    };
}