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

#include <donut/render/DrawStrategy.h>
#include <donut/render/GeometryPasses.h>
#include <donut/render/RenderExtract.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>

#include <algorithm>
#include <functional>
#include <thread>

using namespace donut::math;
using namespace donut::engine;
using namespace donut::render;

// ------------------------------------------------------------------------- packet adapters (plan step E5, item E5.3)

namespace
{
    // More views than any frame extracts: a larger count is a snapshot that died with its frame (Debug builds fill the
    // arena with 0xDD at IRenderGraph::Reset).
    constexpr uint32_t c_MaxPlausibleSnapshotViews = 1u << 16;

    uint64_t CurrentThreadKey() noexcept
    {
        // never 0, the key of "no thread yet"
        return uint64_t(std::hash<std::thread::id>()(std::this_thread::get_id())) | 1u;
    }

    // A walking strategy in a pipelined frame (E7.8; DrawStrategy.h): true, with one warning per strategy object, while
    // a pipelined frame is open, that is while any graph has a Defer scope open (part I7, backlog I6-N5). Set on the
    // main thread at Extract, before the hand-off, and cleared after the join: no graph is read here, so a root detached
    // or re-attached by the main thread while the frame records is not raced, and a detached root is refused too.
    bool RefuseWalk(bool& warned, const char* strategy) noexcept
    {
        if (!SceneGraph::IsAnyRenderReadDeferring())
            return false;
        if (!warned)
        {
            warned = true;
            donut::log::warning("%s: the scene graph walk is refused while a pipelined frame records (a Defer "
                                "render-read scope is open); bind the strategy to the frame's packets (SetPackets); "
                                "no draw items", strategy);
        }
        return true;
    }
}

void DrawStrategyUseCheck::OnPrepare() noexcept
{
    const bool overlapping = m_Preparing.fetch_add(1, std::memory_order_acq_rel) != 0;
    m_Owner.store(CurrentThreadKey(), std::memory_order_release);
    if (overlapping && !m_Warned.exchange(true))
        log::warning("A draw strategy was prepared on two threads at once: give each IGeometryPass object its own "
                     "strategy object (DrawStrategy.h)");
    m_Preparing.fetch_sub(1, std::memory_order_acq_rel);
}

void DrawStrategyUseCheck::OnNext() noexcept
{
    const uint64_t owner = m_Owner.load(std::memory_order_acquire);
    if (owner != 0 && owner != CurrentThreadKey() && !m_Warned.exchange(true))
        log::warning("A draw strategy was drained on another thread than the one that prepared it last: give each "
                     "IGeometryPass object its own strategy object (DrawStrategy.h)");
}

void PacketDrawItems::Bind(const RenderSnapshot* snapshot, DrawPacketPass pass) noexcept
{
    m_Snapshot = snapshot;
    m_Pass = pass;
    m_Next = 0;
    m_End = 0;
    m_WarnedViews.clear();
}

void PacketDrawItems::Prepare(const IView& view)
{
    m_UseCheck.Prepare();
    m_Next = 0;
    m_End = 0;
    if (!m_Snapshot || uint32_t(m_Pass) >= c_DrawPacketPassCount)
        return;

    auto warnOnce = [this](const IView* key, const char* message)
    {
        if (std::find(m_WarnedViews.begin(), m_WarnedViews.end(), key) != m_WarnedViews.end())
            return;
        m_WarnedViews.push_back(key);
        log::warning("PacketDrawItems: %s (view %p)", message, static_cast<const void*>(key));
    };
    if (m_Snapshot->viewCount > c_MaxPlausibleSnapshotViews)
    {
        warnOnce(nullptr, "the bound snapshot is not of this frame (it dies at IRenderGraph::Reset); bind the strategy "
                          "again after each RenderService::Extract; no draw items");
        return;
    }

    // The view by address: the child views of a composite view, prepared in turn, each get their own packets.
    const ExtractedView* extracted = nullptr;
    for (uint32_t v = 0; v < m_Snapshot->viewCount; ++v)
    {
        if (m_Snapshot->views[v].view == &view)
        {
            extracted = &m_Snapshot->views[v];
            break;
        }
    }
    if (!extracted)
    {
        warnOnce(&view, "the view was not extracted into the bound snapshot (ExtractDesc::views); it gets no draw items");
        return;
    }

    const SnapshotRange range = extracted->packets[uint32_t(m_Pass)];
    if (range.first > m_Snapshot->packets.count || range.count > m_Snapshot->packets.count - range.first)
    {
        warnOnce(&view, "the view's packet range is outside the bound snapshot; no draw items");
        return;
    }
    m_Next = range.first;
    m_End = range.first + range.count;
}

const DrawItem* PacketDrawItems::Next() noexcept
{
    m_UseCheck.Next();
    if (m_Next >= m_End || !m_Snapshot)   // m_End is 0 unless Prepare selected packets of the bound snapshot
        return nullptr;
    const DrawPackets& packets = m_Snapshot->packets;
    const VisibleInstances& instances = m_Snapshot->instances;
    while (m_Next < m_End)
    {
        const uint32_t i = m_Next++;
        const uint32_t visible = packets.visibleInstances[i];
        if (visible >= instances.count)
            continue;
        const MeshInfo* mesh = instances.meshInfos[visible];
        const uint32_t geometryIndex = packets.geometryIndices[i];
        // The mesh and its geometries were taken at extraction; a mesh edited in place since then (a structural change
        // inside the render-read scope) may have fewer geometries: such a packet is dropped, not read out of range.
        if (!mesh || geometryIndex >= mesh->geometries.size())
            continue;
        const MeshGeometry* geometry = mesh->geometries[geometryIndex].Get();

        // The leaf (null for a runtime-only row) and the slot from the snapshot: RenderView draws with the slot and
        // reads no leaf (plan step E7, item E7.2).
        m_Item.instance = instances.leaves[visible];
        m_Item.instanceSlot = instances.instanceSlots[visible];
        m_Item.mesh = mesh;
        m_Item.geometry = geometry;
        m_Item.material = geometry->material.Get();
        m_Item.buffers = mesh->buffers.Get();
        m_Item.distanceToCamera =
            m_Pass == DrawPacketPass::Transparent ? GetTransparentDrawKeyDistance(packets.keys[i]) : 0.f;
        m_Item.cullMode = nvrhi::RasterCullMode(packets.cullModes[i]);
        m_Item.userData = nullptr;
        return &m_Item;
    }
    return nullptr;
}

void PacketDrawStrategy::SetPackets(const RenderSnapshot* snapshot, DrawPacketPass pass)
{
    m_Packets.Bind(snapshot, pass);
}

void PacketDrawStrategy::PrepareForView(engine::SceneGraphNode*, const engine::IView& view)
{
    m_Packets.Prepare(view);
}

const DrawItem* PacketDrawStrategy::GetNextItem()
{
    return m_Packets.Next();
}

// ------------------------------------------------------------------------------------------ legacy strategies

const DrawItem* PassthroughDrawStrategy::GetNextItem()
{
    if (m_Count > 0)
    {
        --m_Count;
        return m_Data++;
    }

    m_Data = nullptr;
    return nullptr;
}

void PassthroughDrawStrategy::SetData(const DrawItem* data, size_t count)
{
    m_Data = data;
    m_Count = count;
}

static int CompareDrawItemsOpaque(const DrawItem* a, const DrawItem* b)
{
    if (a->material != b->material)
        return a->material < b->material;

    if (a->buffers != b->buffers)
        return a->buffers < b->buffers;

    if (a->mesh != b->mesh)
        return a->mesh < b->mesh;

    return a->instance < b->instance;
}

void InstancedOpaqueDrawStrategy::FillChunk()
{
    m_InstanceChunk.resize(m_ChunkSize);

    DrawItem* writePtr = m_InstanceChunk.data();
    size_t itemCount = 0;

    while (m_Walker && itemCount < m_ChunkSize)
    {
        auto relevantContentFlags = SceneContentFlags::OpaqueMeshes | SceneContentFlags::AlphaTestedMeshes;
        bool subgraphContentRelevant = (m_Walker->GetSubgraphContentFlags() & relevantContentFlags) != 0;
        bool nodeContentsRelevant = (m_Walker->GetLeafContentFlags() & relevantContentFlags) != 0;

        bool nodeVisible = false;
        if (subgraphContentRelevant)
        {
            nodeVisible = m_ViewFrustum.intersectsWith(m_Walker->GetGlobalBoundingBox());

            if (nodeVisible && nodeContentsRelevant)
            {
                auto meshInstance = query_cast<MeshInstance>(m_Walker->GetLeaf());
                if (meshInstance)
                {
                    const engine::MeshInfo* mesh = meshInstance->GetMesh();

                    size_t requiredChunkSize = itemCount + mesh->geometries.size();
                    if (m_InstanceChunk.size() < requiredChunkSize)
                    {
                        m_InstanceChunk.resize(requiredChunkSize);
                        writePtr = m_InstanceChunk.data() + itemCount;
                    }

                    for (const auto& geometry : mesh->geometries)
                    {
                        auto domain = geometry->material->domain;
                        if (domain != MaterialDomain::Opaque && domain != MaterialDomain::AlphaTested)
                            continue;
                        
                        if (mesh->geometries.size() > 1 && !mesh->skinPrototype)
                        {
                            dm::box3 geometryGlobalBoundingBox = geometry->objectSpaceBounds * m_Walker->GetLocalToWorldTransformFloat();
                            if (!m_ViewFrustum.intersectsWith(geometryGlobalBoundingBox))
                                continue;
                        }

                        DrawItem& item = *writePtr;
                        item.instance = meshInstance;
                        item.mesh = mesh;
                        item.geometry = geometry;
                        item.material = geometry->material;
                        item.buffers = item.mesh->buffers;
                        item.cullMode = (item.material->doubleSided) ? nvrhi::RasterCullMode::None : nvrhi::RasterCullMode::Back;
                        item.distanceToCamera = 0; // don't care
                        item.userData = nullptr;
                        
                        ++writePtr;
                        ++itemCount;
                    }
                }
            }
        }

        m_Walker.Next(nodeVisible);
    }

    m_InstanceChunk.resize(itemCount);
    m_InstancePtrChunk.resize(itemCount);

    for (size_t i = 0; i < itemCount; i++)
    {
        m_InstancePtrChunk[i] = &m_InstanceChunk[i];
    }
    
    if (itemCount > 1)
    {
        std::sort(m_InstancePtrChunk.data(), m_InstancePtrChunk.data() + m_InstancePtrChunk.size(), CompareDrawItemsOpaque);
    }

    m_ReadPtr = 0;
}

void donut::render::InstancedOpaqueDrawStrategy::PrepareForView(engine::SceneGraphNode *rootNode, const engine::IView& view)
{
    if (m_Packets.IsBound())
    {
        m_Packets.Prepare(view);   // checks its own use
        return;
    }

    m_UseCheck.Prepare();
    if (RefuseWalk(m_WarnedPipelined, "InstancedOpaqueDrawStrategy"))
    {
        m_Walker = SceneGraphWalker();
        m_InstanceChunk.clear();
        m_InstancePtrChunk.clear();
        m_ReadPtr = 0;
        return;
    }
    m_Walker = SceneGraphWalker(rootNode);
    m_ViewFrustum = view.GetViewFrustum();
    m_InstanceChunk.clear();
    m_InstancePtrChunk.clear();   // the last view's sorted pointers point into the items just cleared (backlog I6-N2)
    m_ReadPtr = 0;
}

void InstancedOpaqueDrawStrategy::SetPackets(const RenderSnapshot* snapshot)
{
    m_Packets.Bind(snapshot, DrawPacketPass::Opaque);
    // Either mode starts from PrepareForView: nothing of the other mode's last view is drained.
    m_Walker = SceneGraphWalker();
    m_InstanceChunk.clear();
    m_InstancePtrChunk.clear();
    m_ReadPtr = 0;
}

const DrawItem* InstancedOpaqueDrawStrategy::GetNextItem()
{
    if (m_Packets.IsBound())
        return m_Packets.Next();

    m_UseCheck.Next();
    if (m_ReadPtr >= m_InstancePtrChunk.size())
        FillChunk();

    if (m_InstancePtrChunk.empty())
        return nullptr;

    return m_InstancePtrChunk[m_ReadPtr++];
}


static int CompareDrawItemsTransparent(const DrawItem* a, const DrawItem* b)
{
    if (a->instance == b->instance)
        return a->cullMode > b->cullMode;

    return a->distanceToCamera > b->distanceToCamera;
}

void TransparentDrawStrategy::SetPackets(const RenderSnapshot* snapshot)
{
    m_Packets.Bind(snapshot, DrawPacketPass::Transparent);
    m_InstancesToDraw.clear();
    m_InstancePtrsToDraw.clear();
    m_ReadPtr = 0;
}

void TransparentDrawStrategy::PrepareForView(engine::SceneGraphNode* rootNode, const IView& view)
{
    if (m_Packets.IsBound())
    {
        m_Packets.Prepare(view);   // checks its own use
        return;
    }

    m_UseCheck.Prepare();
    m_ReadPtr = 0;

    m_InstancesToDraw.clear();
    m_InstancePtrsToDraw.clear();
    if (RefuseWalk(m_WarnedPipelined, "TransparentDrawStrategy"))
        return;

    float3 viewOrigin = view.GetViewOrigin();
    auto viewFrustum = view.GetViewFrustum();

    SceneGraphWalker walker(rootNode);
    while (walker)
    {
        auto relevantContentFlags = SceneContentFlags::BlendedMeshes;
        bool subgraphContentRelevant = (walker->GetSubgraphContentFlags() & relevantContentFlags) != 0;
        bool nodeContentsRelevant = (walker->GetLeafContentFlags() & relevantContentFlags) != 0;

        bool nodeVisible = false;
        if (subgraphContentRelevant)
        {
            nodeVisible = viewFrustum.intersectsWith(walker->GetGlobalBoundingBox());

            if (nodeVisible && nodeContentsRelevant)
            {
                auto meshInstance = query_cast<MeshInstance>(walker->GetLeaf());
                if (meshInstance)
                {
                    const engine::MeshInfo* mesh = meshInstance->GetMesh();
                    for (const auto& geometry : mesh->geometries)
                    {
                        const auto& material = geometry->material;
                        if (material->domain == MaterialDomain::Opaque || material->domain == MaterialDomain::AlphaTested)
                            continue;

                        dm::box3 geometryGlobalBoundingBox;
                        if (mesh->geometries.size() > 1 && mesh->skinPrototype)
                        {
                            geometryGlobalBoundingBox = geometry->objectSpaceBounds * walker->GetLocalToWorldTransformFloat();
                            if (!viewFrustum.intersectsWith(geometryGlobalBoundingBox))
                                continue;
                        }
                        else
                        {
                            geometryGlobalBoundingBox = walker->GetGlobalBoundingBox();
                        }

                        DrawItem item{};
                        item.instance = meshInstance;
                        item.mesh = mesh;
                        item.geometry = geometry;
                        item.material = geometry->material;
                        item.buffers = mesh->buffers;
                        item.distanceToCamera = length(geometryGlobalBoundingBox.center() - viewOrigin);
                        item.userData = nullptr;
                        if (material->doubleSided)
                        {
                            if (DrawDoubleSidedMaterialsSeparately)
                            {
                                item.cullMode = nvrhi::RasterCullMode::Front;
                                m_InstancesToDraw.push_back(item);
                                item.cullMode = nvrhi::RasterCullMode::Back;
                                m_InstancesToDraw.push_back(item);
                            }
                            else
                            {
                                item.cullMode = nvrhi::RasterCullMode::None;
                                m_InstancesToDraw.push_back(item);
                            }
                        }
                        else
                        {
                            item.cullMode = nvrhi::RasterCullMode::Back;
                            m_InstancesToDraw.push_back(item);
                        }
                    }
                }
            }
        }

        walker.Next(nodeVisible);
    }

    if (m_InstancesToDraw.empty())
        return;

    m_InstancePtrsToDraw.resize(m_InstancesToDraw.size());

    for (size_t i = 0; i < m_InstancesToDraw.size(); i++)
    {
        m_InstancePtrsToDraw[i] = &m_InstancesToDraw[i];
    }

    if (m_InstancePtrsToDraw.size() > 1)
    {
        std::sort(m_InstancePtrsToDraw.data(), m_InstancePtrsToDraw.data() + m_InstancePtrsToDraw.size(), CompareDrawItemsTransparent);
    }
}

const DrawItem* TransparentDrawStrategy::GetNextItem()
{
    if (m_Packets.IsBound())
        return m_Packets.Next();

    m_UseCheck.Next();
    if (m_ReadPtr >= m_InstancePtrsToDraw.size())
    {
        m_InstancesToDraw.clear();
        m_InstancePtrsToDraw.clear();
        return nullptr;
    }

    return m_InstancePtrsToDraw[m_ReadPtr++];
}
