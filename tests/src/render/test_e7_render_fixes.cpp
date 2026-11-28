/*
* Copyright (c) 2026, ethereal contributors. All rights reserved.
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

// Regression tests of the E7 fix track (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md, stage F), render side:
// one case per backlog item, named after it. Each fails without its fix. Device-free, but with `--gpu d3d12|vulkan`
// (ctest test_e7_render_gpu_<backend>, a GPU program: one at a time) the GPU cases on a headless device instead.
//   e72_r5_runtime_rows_checked  Debug's leaf check compares runtime-only rows with their mesh (Debug only: Release
//                                builds do not check)
//   i6_n2_walk_prepared_again    a walking strategy prepared again mid-drain hands out the new view's items only
//   i6_n3_service_closes_its_scope  EndFrame closes the service's own render-read scope, not the innermost one
//   i6_n3_scopes_by_identity     render-read scopes closed by identity, under the deferring rules
//   f4_n1_identity_beyond_64     beyond 64 open scopes: no identity; EndRenderRead(0) closes the innermost
//   f4_n1_zero_closes_only_without_identity  0 closes only a scope without identity; a pipelined Extract needs one
//   t5_n1_compact_geometry_cap   with compact records, a mesh's geometries past the cap get no packets (one warning)
//   t8_4_scope_identity_wrap     the scope identities skip 0 at the wrap past 2^32 - 1
//   s2_first_packet_of_a_pass    a pass's first packet in a range that a view already sees grows no scratch
//   f10_compact_records_default  compact packet records are on by default (stage B passed them; F10)
//   i7_n1_service_destroyed_while_recording  the service's state is kept, with a warning (last: the process defers)
// GPU:
//   s1_permanent_state_from_prepare  a permanent state that a Prepare sets on the upload list holds in the frame's lists
//   s1_mesh_after_load           a mesh acquired after Scene::FinishedLoading is usable in the frame that makes it

#include <donut/app/ApplicationBase.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/render/RenderGraphLambdaPass.h>
#include <donut/tests/e7_switches.h>
#include <donut/tests/utils.h>

#include "RenderVisibility.h"   // its test seam for the compact geometry cap (src/render)
#include "SceneGraphStore.h"    // its test seam for the scope identities (src/engine)

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;
using namespace donut::render;

namespace
{
    std::atomic<uint32_t> g_Warnings{ 0 };   // log::warning calls; a case that provokes some takes them back
    uint32_t TakeWarnings() { return g_Warnings.exchange(0); }

    nvrhi::AutoPtr<MeshInfo> MakeMesh(nvrhi::AutoPtr<Material>& material)
    {
        material = MAKE_RC_OBJ_PTR(Material);
        material->domain = MaterialDomain::Opaque;
        auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
        geometry->material = material;
        geometry->numIndices = 3;
        geometry->numVertices = 3;
        geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->type = MeshType::Triangles;
        mesh->geometries.push_back(geometry);
        mesh->objectSpaceBounds = geometry->objectSpaceBounds;
        return mesh;
    }

    // A graph, a render service and a device-free render graph; one view that sees the rows.
    struct Rig
    {
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        nvrhi::AutoPtr<RenderService> service;
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> renderGraph;
        FrameData data;
        nvrhi::AutoPtr<PlanarView> view = MAKE_RC_OBJ_PTR(PlanarView);
        const IView* views[1] = {};
        uint32_t frame = 0;

        Rig()
        {
            graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            RenderServiceDesc desc;
            desc.sceneGraph = graph;
            service = MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
            CHECK(CreateResourceRegistry(nullptr, &registry) == nvrhi::FS_OK);
            CHECK(CreateRenderGraph(registry, &renderGraph) == nvrhi::FS_OK);
            view->SetViewport(nvrhi::Viewport(1280.f, 720.f));
            const dm::affine3 eye = dm::translation(dm::float3(0.f, 0.f, -20.f));
            view->SetMatrices(dm::inverse(eye), dm::perspProjD3DStyleReverse(dm::radians(60.f), 16.f / 9.f, 0.1f));
            view->UpdateCache();
            views[0] = view.Get();
        }

        void Frame()
        {
            graph->Refresh(frame);
            renderGraph->Reset();
            data.Clear();
            ExtractDesc desc;
            desc.views = views;
            desc.viewCount = 1;
            desc.frameIndex = frame++;
            SceneFrame* sceneFrame = nullptr;
            CHECK(service->Extract(renderGraph, &data, desc, &sceneFrame) == nvrhi::FS_OK);
            service->EndFrame(true);
        }

        Entity Spawn(MeshId mesh)
        {
            IWorld* world = graph->GetWorld();
            std::vector<ComponentId> ids(graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
            graph->GetRuntimeMeshInstanceComponents(ids.data(), uint32_t(ids.size()));
            const LocalTransform local{ dm::double3(0.0), dm::dquat::identity(), dm::double3(1.0) };
            const MeshRef ref{ mesh };
            std::vector<const void*> values(ids.size(), nullptr);
            for (size_t i = 0; i < ids.size(); ++i)
            {
                if (ids[i] == FindComponent<LocalTransform>(world))
                    values[i] = &local;
                else if (ids[i] == FindComponent<MeshRef>(world))
                    values[i] = &ref;
            }
            EntityCreateDesc desc{};
            desc.components = ids.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(ids.size());
            desc.entityCount = 1;
            Entity entity = Entity::Null;
            CHECK(world->CreateEntities(desc, &entity) == nvrhi::FS_OK);
            return entity;
        }
    };
}

// ------------------------------------------------------------------------------------------------ E72-R5

// A runtime-only row has no leaf: its LocalBounds and RenderFlags come from its mesh when it joins. An in-place edit of
// the mesh's box, or of a geometry's material domain, leaves them stale; Debug's leaf check (every 64th extraction)
// reports it, once per stale row and check, as it does for a leaf. 64 extractions hold exactly one check.
void test_e72_r5_runtime_rows_checked()
{
#ifndef NDEBUG
    const uint32_t expected = 1;
#else
    const uint32_t expected = 0;
#endif
    nvrhi::AutoPtr<Material> material;
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh(material);
    Rig rig;
    const MeshId id = rig.graph->AcquireRuntimeMesh(mesh);
    CHECK(id != MeshId::Invalid);
    rig.Spawn(id);
    for (int f = 0; f < 64; ++f)
        rig.Frame();
    CHECK(TakeWarnings() == 0);   // as it joined: no warning

    mesh->objectSpaceBounds = dm::box3(dm::float3(-3.f), dm::float3(3.f));   // edited in place
    for (int f = 0; f < 64; ++f)
        rig.Frame();
    CHECK(TakeWarnings() == expected);
    mesh->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
    for (int f = 0; f < 64; ++f)
        rig.Frame();
    CHECK(TakeWarnings() == 0);

    material->domain = MaterialDomain::AlphaTested;   // the content flags change with it
    for (int f = 0; f < 64; ++f)
        rig.Frame();
    CHECK(TakeWarnings() == expected);
    material->domain = MaterialDomain::Opaque;
    for (int f = 0; f < 64; ++f)
        rig.Frame();
    CHECK(TakeWarnings() == 0);
    rig.graph->ReleaseRuntimeMesh(id);
}

// ------------------------------------------------------------------------------------------------ I6-N2

// InstancedOpaqueDrawStrategy prepared for a view, drained in part, then prepared for a view that sees nothing: the
// next item is null, not one of the first view's items (PrepareForView left the sorted item pointers of the last
// chunk, which pointed into the cleared items).
void test_i6_n2_walk_prepared_again()
{
    nvrhi::AutoPtr<Material> material;
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh(material);
    Rig rig;
    for (int i = 0; i < 4; ++i)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetTranslation(dm::double3(double(i) * 2.0 - 3.0, 0.0, 0.0));
        node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
        rig.graph->Attach(rig.graph->GetRootNode(), node);
    }
    rig.graph->Refresh(0);
    auto away = MAKE_RC_OBJ_PTR(PlanarView);   // behind the nodes, looking further away from them
    away->SetViewport(nvrhi::Viewport(1280.f, 720.f));
    away->SetMatrices(dm::inverse(dm::translation(dm::float3(0.f, 0.f, 100.f))),
                      dm::perspProjD3DStyleReverse(dm::radians(60.f), 16.f / 9.f, 0.1f));
    away->UpdateCache();

    auto strategy = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
    strategy->PrepareForView(rig.graph->GetRootNode(), *rig.view);
    uint32_t seen = 0;
    while (strategy->GetNextItem())
        ++seen;
    CHECK(seen == 4);   // the rig's view sees the four
    strategy->PrepareForView(rig.graph->GetRootNode(), *rig.view);
    CHECK(strategy->GetNextItem() != nullptr);   // one taken, three left of this chunk
    strategy->PrepareForView(rig.graph->GetRootNode(), *away);
    CHECK(strategy->GetNextItem() == nullptr);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------ I6-N3

// The service closes its own scope by its identity (SceneGraph::BeginRenderRead's), wherever it is among the open
// ones: a scope the application opened after Extract stays open; one the application opened before Extract and closes
// in between (below the service's) does not move the service's out of reach. Closing the innermost instead closed the
// application's scope and left the Defer one open; closing by depth left it open in the second case (the graph
// deferred for good). An application EndRenderRead() that closes the service's scope before EndFrame: one warning.
void test_i6_n3_service_closes_its_scope()
{
    Rig rig;
    RenderServiceDesc desc;
    desc.sceneGraph = rig.graph;
    desc.pipelined = true;
    rig.service = nullptr;
    rig.service = MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
    rig.graph->Refresh(0);
    ExtractDesc extract;
    extract.views = rig.views;
    extract.viewCount = 1;
    auto extractFrame = [&]()
    {
        rig.renderGraph->Reset();
        rig.data.Clear();
        SceneFrame* sceneFrame = nullptr;
        CHECK(rig.service->Extract(rig.renderGraph, &rig.data, extract, &sceneFrame) == nvrhi::FS_OK);
        CHECK(rig.graph->IsRenderReadDeferring());
    };

    // a scope opened after Extract: it stays open, the service's Defer scope closes
    extractFrame();
    const uint32_t after = rig.graph->BeginRenderRead(SceneGraph::RenderReadMode::Warn);
    CHECK(after != 0);
    rig.service->EndFrame(true);
    CHECK(TakeWarnings() == 0);
    CHECK(rig.graph->IsInRenderRead());   // the application's
    CHECK(!rig.graph->IsRenderReadDeferring() && !SceneGraph::IsAnyRenderReadDeferring());
    rig.graph->EndRenderRead(after);
    CHECK(!rig.graph->IsInRenderRead());

    // a scope opened before Extract and closed in between, below the service's
    const uint32_t before = rig.graph->BeginRenderRead(SceneGraph::RenderReadMode::Warn);
    extractFrame();
    rig.graph->EndRenderRead(before);
    CHECK(rig.graph->IsRenderReadDeferring());
    rig.service->EndFrame(true);
    CHECK(TakeWarnings() == 0);
    CHECK(!rig.graph->IsInRenderRead() && !SceneGraph::IsAnyRenderReadDeferring());

    // the application closes the service's scope (EndRenderRead without its BeginRenderRead): EndFrame warns once
    extractFrame();
    rig.graph->EndRenderRead();
    CHECK(!rig.graph->IsInRenderRead());
    rig.service->EndFrame(true);
    CHECK(TakeWarnings() == 1);
    CHECK(!rig.graph->IsInRenderRead());

    // the usual order: no warning
    extractFrame();
    rig.service->EndFrame(true);
    CHECK(TakeWarnings() == 0);
    CHECK(!rig.graph->IsInRenderRead());
}

// Two Defer scopes, a drawn node detached inside them and a mesh edited in place: closing the lower one (by identity)
// keeps the node alive and the reconciliation waiting; closing the last runs and releases them. A scope that is not
// open: one warning, nothing closes. Beyond 64 nested scopes: no identity (0), a Defer scope acts as Warn (one
// warning), and EndRenderRead(0) closes the innermost.
void test_i6_n3_scopes_by_identity()
{
    nvrhi::AutoPtr<Material> material;
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh(material);
    nvrhi::AutoPtr<MeshInfo> edited = MakeMesh(material);
    Rig rig;
    nvrhi::AutoPtr<SceneGraphNode> drawn = MAKE_RC_OBJ_PTR(SceneGraphNode);
    drawn->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
    rig.graph->Attach(rig.graph->GetRootNode(), drawn);
    auto other = MAKE_RC_OBJ_PTR(SceneGraphNode);
    other->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, edited));
    rig.graph->Attach(rig.graph->GetRootNode(), other);
    rig.graph->Refresh(0);
    const uint32_t editedSlot = MeshTable::SlotOf(rig.graph->GetMeshes().Find(edited));
    CHECK(rig.graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 1);
    nvrhi::AutoPtr<nvrhi::IWeakReference> weak;
    drawn->GetWeakReference(weak.GetAddressOf());
    CHECK(weak);

    const uint32_t lower = rig.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
    const uint32_t upper = rig.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
    CHECK(lower != 0 && upper != 0 && lower != upper);
    (void)rig.graph->Detach(drawn);
    drawn = nullptr;
    auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);   // an in-place edit, then a structural change
    geometry->material = material;
    edited->geometries.push_back(geometry);
    rig.graph->Attach(rig.graph->GetRootNode(), MAKE_RC_OBJ_PTR(SceneGraphNode));
    rig.graph->Refresh(1);
    CHECK(!weak->IsExpired());
    CHECK(rig.graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 1);   // waits for the last Defer scope

    rig.graph->EndRenderRead(lower);
    CHECK(rig.graph->IsRenderReadDeferring());
    CHECK(!weak->IsExpired());
    CHECK(rig.graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 1);
    rig.graph->EndRenderRead(lower);   // not open any more
    CHECK(TakeWarnings() == 1);
    CHECK(rig.graph->IsRenderReadDeferring());
    rig.graph->EndRenderRead(upper);
    CHECK(!rig.graph->IsInRenderRead() && !rig.graph->IsRenderReadDeferring());
    CHECK(weak->IsExpired());
    CHECK(rig.graph->GetMeshes().GetMeshRow(editedSlot).geometryCount == 2);
    CHECK(TakeWarnings() == 0);

    // deeper than 64
    for (int i = 0; i < 64; ++i)
        CHECK(rig.graph->BeginRenderRead() != 0);
    CHECK(rig.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer) == 0);
    CHECK(TakeWarnings() == 1);
    CHECK(!rig.graph->IsRenderReadDeferring());
    rig.graph->EndRenderRead(0);   // the innermost
    for (int i = 0; i < 64; ++i)
    {
        CHECK(rig.graph->IsInRenderRead());
        rig.graph->EndRenderRead();
    }
    CHECK(!rig.graph->IsInRenderRead());
    CHECK(TakeWarnings() == 0);
}

// More than 64 scopes open: closing one in the middle by identity moves the scopes above it down, and the one that
// comes to the 64th place has no identity (0), as it had none; EndRenderRead(0) closes the innermost scope whatever
// its identity (F4-N1). 66 open, one closed in the middle, one by 0: 64 left.
void test_f4_n1_identity_beyond_64()
{
    Rig rig;
    uint32_t ids[66] = {};
    for (uint32_t i = 0; i < 66; ++i)
        ids[i] = rig.graph->BeginRenderRead();
    CHECK(ids[63] != 0 && ids[64] == 0 && ids[65] == 0);
    rig.graph->EndRenderRead(ids[10]);
    rig.graph->EndRenderRead(ids[10]);   // closed already: a warning, nothing closes
    CHECK(TakeWarnings() == 1);
    rig.graph->EndRenderRead(0);
    uint32_t left = 0;
    while (rig.graph->IsInRenderRead())
    {
        rig.graph->EndRenderRead();
        ++left;
    }
    CHECK(left == 64);
    CHECK(TakeWarnings() == 0);
}

// EndRenderRead(0) closes the innermost scope only when it has no identity (the F5 review): a caller that got 0
// closes its own scope, or nothing (a warning). And a pipelined service whose scope gets no identity (64 scopes open:
// its Defer request acts as Warn, with no retire list) fails its Extract with a warning, closing that scope; its
// BeginPipelinedRecording then has no frame to hand over.
void test_f4_n1_zero_closes_only_without_identity()
{
    Rig rig;
    const uint32_t first = rig.graph->BeginRenderRead();
    CHECK(first != 0);
    rig.graph->EndRenderRead(0);   // the innermost has an identity: nothing closes
    CHECK(TakeWarnings() == 1);
    CHECK(rig.graph->IsInRenderRead());
    rig.graph->EndRenderRead(first);
    CHECK(!rig.graph->IsInRenderRead());

    RenderServiceDesc desc;
    desc.sceneGraph = rig.graph;
    desc.pipelined = true;
    rig.service = nullptr;
    rig.service = MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
    rig.graph->Refresh(0);
    for (int i = 0; i < 64; ++i)
        rig.graph->BeginRenderRead();
    ExtractDesc extract;
    extract.views = rig.views;
    extract.viewCount = 1;
    SceneFrame* sceneFrame = nullptr;
    CHECK(rig.service->Extract(rig.renderGraph, &rig.data, extract, &sceneFrame) == nvrhi::FE_INVALID_ARGS);
    CHECK(sceneFrame == nullptr);
    CHECK(TakeWarnings() == 2);   // the graph's (a Defer scope that deep acts as Warn) and the service's
    CHECK(!rig.graph->IsRenderReadDeferring() && !SceneGraph::IsAnyRenderReadDeferring());
    CHECK(rig.service->BeginPipelinedRecording() == nvrhi::FE_INVALID_ARGS);   // no frame open
    CHECK(TakeWarnings() == 1);
    uint32_t left = 0;
    while (rig.graph->IsInRenderRead())
    {
        rig.graph->EndRenderRead();
        ++left;
    }
    CHECK(left == 64);   // the service's scope was closed
    rig.service->EndFrame(false);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------ T5-N1

// Compact packet records hold a geometry index of 26 bits: with the switch on, a mesh's geometries from the cap on
// get no packets, with one warning per process. The cap lowered to 2 through the module's test seam: a mesh of four
// opaque geometries gives half the packets it gives with the default cap; the switch off ignores the cap.
void test_t5_n1_compact_geometry_cap()
{
    nvrhi::AutoPtr<Material> material;
    nvrhi::AutoPtr<MeshInfo> mesh = MakeMesh(material);
    for (int i = 0; i < 3; ++i)
    {
        auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
        geometry->material = material;
        geometry->numIndices = 3;
        geometry->numVertices = 3;
        geometry->objectSpaceBounds = mesh->objectSpaceBounds;
        mesh->geometries.push_back(geometry);
    }
    Rig rig;
    auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
    node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
    rig.graph->Attach(rig.graph->GetRootNode(), node);
    auto packets = [&]()
    {
        rig.graph->Refresh(rig.frame);
        rig.renderGraph->Reset();
        rig.data.Clear();
        ExtractDesc desc;
        desc.views = rig.views;
        desc.viewCount = 1;
        desc.frameIndex = rig.frame++;
        SceneFrame* sceneFrame = nullptr;
        CHECK(rig.service->Extract(rig.renderGraph, &rig.data, desc, &sceneFrame) == nvrhi::FS_OK);
        CHECK(sceneFrame && sceneFrame->snapshot);
        const uint32_t count = sceneFrame->snapshot->packets.count;
        rig.service->EndFrame(true);
        return count;
    };
    const bool compact = RenderService::GetCompactPacketRecords();
    RenderService::SetCompactPacketRecords(true);
    const uint32_t full = packets();
    CHECK(full >= 4 && full % 4 == 0);
    CHECK(TakeWarnings() == 0);
    visibility::InstanceVisibility::SetCompactGeometryLimit(2);
    CHECK(packets() == full / 2);
    CHECK(TakeWarnings() == 1);
    CHECK(packets() == full / 2);
    CHECK(TakeWarnings() == 0);   // once per process
    RenderService::SetCompactPacketRecords(false);
    CHECK(packets() == full);     // the cap is the compact records'
    visibility::InstanceVisibility::SetCompactGeometryLimit(visibility::InstanceVisibility::c_CompactGeometryLimit);
    RenderService::SetCompactPacketRecords(compact);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------ S-2

// A range's scratch for a view reserves every pass's packets when the view first sees rows there (backlog S-2: in
// ecs_scale at a million rows, a pass's first packet in such a range was one of extraction's last allocations, at frames
// 100 to 260). Opaque nodes for two frames; then a node of a blended mesh joins their chunk, so that the transparent
// pass gets its first packet in that range: the ranges' scratch does not grow (InstanceVisibility's test seam), with
// the compact records off and on.
void test_s2_first_packet_of_a_pass()
{
    const bool compact = RenderService::GetCompactPacketRecords();
    for (const bool records : { false, true })
    {
        RenderService::SetCompactPacketRecords(records);
        nvrhi::AutoPtr<Material> opaque;
        nvrhi::AutoPtr<MeshInfo> solid = MakeMesh(opaque);
        nvrhi::AutoPtr<Material> blended;
        nvrhi::AutoPtr<MeshInfo> glass = MakeMesh(blended);
        blended->domain = MaterialDomain::AlphaBlended;
        Rig rig;
        auto attach = [&](MeshInfo* mesh, double x)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(dm::double3(x, 0.0, 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
            CHECK(rig.graph->Attach(rig.graph->GetRootNode(), node));
        };
        for (int i = 0; i < 4; ++i)
            attach(solid, double(i) - 2.0);
        rig.Frame();
        rig.Frame();
        const uint64_t growths = visibility::InstanceVisibility::GetRangeScratchGrowths();
        attach(glass, 0.5);
        rig.Frame();
        CHECK(visibility::InstanceVisibility::GetRangeScratchGrowths() == growths);
        rig.Frame();
        CHECK(visibility::InstanceVisibility::GetRangeScratchGrowths() == growths);
    }
    RenderService::SetCompactPacketRecords(compact);
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------ F10

// Compact packet records are on by default since stage B measured them (F10): a process that sets nothing extracts
// with them (the switches-on runs set them on too, DONUT_E7_SWITCHES=off off).
void test_f10_compact_records_default()
{
    CHECK(RenderService::GetCompactPacketRecords() == donut::tests::E7SwitchExpected("compact", true));
}

// ------------------------------------------------------------------------------------------------ T8-4

// A graph whose identities start near the wrap (the store's test seam): 2^32 - 1, then 1, never 0 (0 means "no
// identity"); both close by identity.
void test_t8_4_scope_identity_wrap()
{
    SceneGraphStore::SetFirstRenderReadScope(0xfffffffeu);
    Rig rig;
    SceneGraphStore::SetFirstRenderReadScope(0);
    const uint32_t last = rig.graph->BeginRenderRead();
    const uint32_t wrapped = rig.graph->BeginRenderRead();
    CHECK(last == 0xffffffffu && wrapped == 1);
    rig.graph->EndRenderRead(last);
    CHECK(rig.graph->IsInRenderRead());
    rig.graph->EndRenderRead(wrapped);
    CHECK(!rig.graph->IsInRenderRead());
    CHECK(TakeWarnings() == 0);
}

// ------------------------------------------------------------------------------------------------ I7-N1

// A pipelined service destroyed while its frame records (BeginPipelinedRecording without the EndFrame after the
// join): one warning, and its state is kept (leaked) so that the recording reads nothing freed: the graph's Defer
// scope stays open and the graph alive. Last of the cases: the process defers from then on.
void test_i7_n1_service_destroyed_while_recording()
{
    nvrhi::AutoPtr<nvrhi::IWeakReference> weakGraph;
    {
        Rig rig;
        RenderServiceDesc desc;
        desc.sceneGraph = rig.graph;
        desc.pipelined = true;
        rig.service = nullptr;
        rig.service = MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
        rig.graph->Refresh(0);
        ExtractDesc extract;
        extract.views = rig.views;
        extract.viewCount = 1;
        SceneFrame* sceneFrame = nullptr;
        CHECK(rig.service->Extract(rig.renderGraph, &rig.data, extract, &sceneFrame) == nvrhi::FS_OK);
        CHECK(rig.service->BeginPipelinedRecording() == nvrhi::FS_OK);
        rig.graph->GetWeakReference(weakGraph.GetAddressOf());
        CHECK(weakGraph);
        rig.service = nullptr;   // the frame still records
        CHECK(TakeWarnings() == 1);
        CHECK(rig.graph->IsRenderReadDeferring());
    }
    CHECK(!weakGraph->IsExpired());   // the leaked service holds it
    CHECK(SceneGraph::IsAnyRenderReadDeferring());
}

// ------------------------------------------------------------------------------------------- S-1 (GPU)

namespace
{
    std::atomic<uint32_t> g_Errors{ 0 };   // log::error calls (nvrhi's messages among them): the GPU cases

    struct GpuNoData
    {
        uint32_t unused;
    };

    // A buffer the graph imports in CopyDest and leaves there; `cpuRead`: host-readable.
    nvrhi::BufferHandle MakeTarget(nvrhi::IDevice* device, uint64_t bytes, bool cpuRead)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = bytes;
        desc.cpuAccess = cpuRead ? nvrhi::CpuAccessMode::Read : nvrhi::CpuAccessMode::None;
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        desc.debugName = "S1Target";
        nvrhi::BufferHandle buffer;
        device->createBuffer(desc, &buffer);
        CHECK(buffer);
        return buffer;
    }

    std::vector<uint8_t> ReadHost(nvrhi::IDevice* device, nvrhi::IBuffer* buffer)
    {
        std::vector<uint8_t> bytes(size_t(buffer->getDesc().byteSize));
        const void* data = device->mapBuffer(buffer, nvrhi::CpuAccessMode::Read);
        CHECK(data != nullptr);
        std::memcpy(bytes.data(), data, bytes.size());
        device->unmapBuffer(buffer);
        return bytes;
    }

    // A registry, a graph and the device's profile; Run compiles (no diagnostics) and executes, then waits.
    struct GpuRig
    {
        nvrhi::IDevice* device;
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;
        BackendProfile profile;

        explicit GpuRig(nvrhi::IDevice* d) : device(d)
        {
            CHECK(CreateResourceRegistry(device, &registry) == nvrhi::FS_OK);
            CHECK(CreateRenderGraph(registry, &graph) == nvrhi::FS_OK);
            CHECK(ComputeBackendProfile(device, &profile) == nvrhi::FS_OK);
        }

        void Run()
        {
            nvrhi::AutoPtr<ICompiledRenderGraph> plan;
            CHECK(graph->Compile(profile, CompileOptions(), &plan) == nvrhi::FS_OK);
            nvrhi::string diagnostics;
            graph->GetDiagnostics(diagnostics);
            if (!diagnostics.empty())
                std::fprintf(stderr, "%s\n", diagnostics.c_str());
            CHECK(diagnostics.empty());
            CHECK(graph->Execute(plan, ExecuteDesc()) == nvrhi::FS_OK);
            device->waitForIdle();
        }
    };
}

// The executor's contract behind S-1: a permanent state that a Prepare sets on the upload list holds in the frame's
// lists while they record (the executor submits the upload list before it records them; nvrhi applies a permanent
// state only when its list is submitted). One pass: its Prepare writes a fresh buffer (no permanent state yet, no graph
// resource) on the upload list and makes CopySource | ShaderResource its permanent state; its Execute copies it into a
// graph buffer on a segment list, where the tracker must find that state. No error, and the bytes arrive (the upload
// ran first on the GPU too). With the upload list submitted after the recording: "Unknown prior state of buffer".
void test_s1_permanent_state_from_prepare(nvrhi::IDevice* device)
{
    const uint32_t errors = g_Errors.load();
    constexpr uint32_t c_Bytes = 256;
    nvrhi::BufferDesc desc;
    desc.byteSize = c_Bytes;
    desc.debugName = "S1Fresh";
    nvrhi::BufferHandle fresh;
    device->createBuffer(desc, &fresh);
    CHECK(fresh);
    nvrhi::BufferHandle target = MakeTarget(device, c_Bytes, true);
    std::vector<uint8_t> bytes(c_Bytes);
    for (uint32_t i = 0; i < c_Bytes; ++i)
        bytes[i] = uint8_t(i * 7 + 3);
    {
        GpuRig rig(device);
        ResourceId id = rig.graph->ImportBuffer(target, nvrhi::ResourceStates::CopyDest,
                                                nvrhi::ResourceStates::CopyDest);
        nvrhi::IBuffer* source = fresh;
        AddLambdaPassWithPrepare<GpuNoData>(rig.graph,
            PassDesc{ "S1PermanentState", PassType::Copy, PassFlags::Tracked, 0 },
            [&](ISetupContext* setup, GpuNoData&)
            {
                id = setup->Write(id, render::Access::CopyDest, nvrhi::ShaderType::None, {}, AccessFlags::FullWrite);
            },
            [&](IPrepareContext* prepare, GpuNoData&)
            {
                nvrhi::ICommandList* upload = prepare->GetUploadCommandList();
                upload->beginTrackingBufferState(source, nvrhi::ResourceStates::Common);
                upload->writeBuffer(source, bytes.data(), bytes.size());
                upload->setPermanentBufferState(source, nvrhi::ResourceStates::CopySource |
                                                        nvrhi::ResourceStates::ShaderResource);
                upload->commitBarriers();
            },
            [&](IPassContext* context, const GpuNoData&)
            {
                context->GetCommandList()->copyBuffer(context->GetBuffer(id), 0, source, 0, c_Bytes);
            });
        rig.Run();
    }
    CHECK(ReadHost(device, target) == bytes);
    CHECK(g_Errors.load() == errors);
}

// S-1 as the sample met it: a graph-path scene loaded (Scene::FinishedLoading), then a mesh acquired for runtime-only
// rows (SceneGraph::AcquireRuntimeMesh). Its buffers are made at the next RenderService::Extract
// (Scene::CreateMeshBufferObjects) and written, with their permanent states, by the upload pass's Prepare; a later pass
// of that frame uses them on its own list, as a draw does (here: their index and vertex buffer states required). No
// error, and the vertex buffer holds the mesh's positions.
void test_s1_mesh_after_load(nvrhi::IDevice* device, nvrhi::GraphicsAPI api)
{
    const uint32_t errors = g_Errors.load();
    auto nativeFS = MAKE_RC_OBJ_PTR(vfs::NativeFileSystem);
    auto rootFS = MAKE_RC_OBJ_PTR(vfs::RootFileSystem);
    rootFS->mount("/shaders/donut", app::GetDirectoryWithExecutable() / "shaders/framework" /
                                        app::GetShaderTypeName(api));
    auto shaderFactory = MAKE_RC_OBJ_PTR(ShaderFactory, device, rootFS, "/shaders");
    auto textureCache = MAKE_RC_OBJ_PTR(TextureCache, device, nativeFS, nullptr);
    auto scene = MAKE_RC_OBJ_PTR(Scene, device, shaderFactory, nativeFS, textureCache, nullptr, nullptr);
    CHECK(scene->SetGpuSceneUploadMode(GpuSceneUploadMode::Scatter) == nvrhi::FS_OK);
    SceneGraph* graph = scene->CreateSceneGraph();
    graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
    RenderServiceDesc serviceDesc;
    serviceDesc.scene = scene;
    serviceDesc.shaderFactory = shaderFactory;
    auto service = MAKE_RC_OBJ_PTR(RenderService, device, serviceDesc);
    scene->FinishedLoading(0);

    // the mesh, after the load: three vertices, one triangle
    auto material = MAKE_RC_OBJ_PTR(Material);
    material->name = "S1";
    auto buffers = MAKE_RC_OBJ_PTR(BufferGroup);
    buffers->positionData = { dm::float3(0.f, 0.f, 1.f), dm::float3(1.f, 2.f, 3.f), dm::float3(-4.f, 5.f, 6.f) };
    buffers->indexData = { 0, 1, 2 };
    const std::vector<dm::float3> positions = buffers->positionData;
    auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
    geometry->material = material;
    geometry->numIndices = 3;
    geometry->numVertices = 3;
    geometry->objectSpaceBounds = dm::box3(dm::float3(-4.f, 0.f, 1.f), dm::float3(1.f, 5.f, 6.f));
    auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
    mesh->name = "S1";
    mesh->type = MeshType::Triangles;
    mesh->buffers = buffers;
    mesh->geometries.push_back(geometry);
    mesh->totalIndices = 3;
    mesh->totalVertices = 3;
    mesh->objectSpaceBounds = geometry->objectSpaceBounds;
    CHECK(graph->AcquireRuntimeMesh(mesh) != MeshId::Invalid);

    nvrhi::BufferHandle target = MakeTarget(device, 16, false);
    {
        GpuRig rig(device);
        scene->RefreshSceneGraph(1);
        ExtractDesc extractDesc;
        extractDesc.frameIndex = 1;
        SceneFrame* frame = nullptr;
        CHECK(service->Extract(rig.graph, nullptr, extractDesc, &frame) == nvrhi::FS_OK && frame);
        CHECK(buffers->vertexBuffer && buffers->indexBuffer);   // made at this extraction
        CHECK(service->AddUploadPasses(rig.graph, frame) == nvrhi::FS_OK);
        ResourceId id = rig.graph->ImportBuffer(target, nvrhi::ResourceStates::CopyDest,
                                                nvrhi::ResourceStates::CopyDest);
        nvrhi::IBuffer* vertexBuffer = buffers->vertexBuffer;
        nvrhi::IBuffer* indexBuffer = buffers->indexBuffer;
        AddLambdaPass<GpuNoData>(rig.graph, PassDesc{ "S1UsesTheMesh", PassType::Copy, PassFlags::Tracked, 0 },
            [&](ISetupContext* setup, GpuNoData&)
            {
                id = setup->Write(id, render::Access::CopyDest, nvrhi::ShaderType::None, {}, AccessFlags::FullWrite);
            },
            [&](IPassContext* context, const GpuNoData&)
            {
                nvrhi::ICommandList* list = context->GetCommandList();
                list->setBufferState(vertexBuffer, nvrhi::ResourceStates::VertexBuffer);
                list->setBufferState(indexBuffer, nvrhi::ResourceStates::IndexBuffer);
                const uint32_t marker[4] = { 1, 2, 3, 4 };
                list->writeBuffer(context->GetBuffer(id), marker, sizeof(marker));
            });
        rig.Run();
        service->EndFrame(true);
    }

    // the positions, through a copy outside the graph (explicit transitions: the tracker cannot leave a permanent state)
    nvrhi::BufferHandle readback = MakeTarget(device, buffers->vertexBuffer->getDesc().byteSize, true);
    nvrhi::ResourceStates permanent = nvrhi::ResourceStates::VertexBuffer | nvrhi::ResourceStates::ShaderResource;
    if (buffers->vertexBuffer->getDesc().isAccelStructBuildInput)
        permanent = permanent | nvrhi::ResourceStates::AccelStructBuildInput;
    nvrhi::CommandListHandle list;
    device->createCommandList(nvrhi::CommandListParameters(), &list);
    list->open();
    list->setEnableAutomaticBarriers(false);
    const nvrhi::ResourceBarrier before =
        nvrhi::ResourceBarrier::Transition(buffers->vertexBuffer, permanent, nvrhi::ResourceStates::CopySource);
    list->barriers(&before, 1);
    list->copyBuffer(readback, 0, buffers->vertexBuffer, 0, readback->getDesc().byteSize);
    const nvrhi::ResourceBarrier after =
        nvrhi::ResourceBarrier::Transition(buffers->vertexBuffer, nvrhi::ResourceStates::CopySource, permanent);
    list->barriers(&after, 1);
    list->setEnableAutomaticBarriers(true);
    list->close();
    device->executeCommandList(list);
    device->waitForIdle();
    const std::vector<uint8_t> vertices = ReadHost(device, readback);
    const nvrhi::BufferRange& range = buffers->getVertexBufferRange(VertexAttribute::Position);
    CHECK(range.byteOffset + positions.size() * sizeof(dm::float3) <= vertices.size());
    CHECK(std::memcmp(vertices.data() + range.byteOffset, positions.data(), positions.size() * sizeof(dm::float3)) == 0);
    CHECK(g_Errors.load() == errors);
    service = nullptr;
    scene = nullptr;
}

// --gpu d3d12|vulkan: a headless device with the debug runtime and the nvrhi validation layer (an NVIDIA adapter when
// there is one), then the GPU cases; every log::error (nvrhi's messages among them) fails the run.
int RunGpuCases(const char* backend)
{
    const bool vulkan = std::strcmp(backend, "vulkan") == 0;
    if (!vulkan && std::strcmp(backend, "d3d12") != 0)
    {
        std::fprintf(stderr, "--gpu d3d12 | vulkan\n");
        return 1;
    }
    const nvrhi::GraphicsAPI api = vulkan ? nvrhi::GraphicsAPI::VULKAN : nvrhi::GraphicsAPI::D3D12;
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Error)
            g_Errors.fetch_add(1);
        else if (severity >= log::Severity::Warning)
            g_Warnings.fetch_add(1);
        if (severity >= log::Severity::Warning)
            std::fprintf(stderr, "%s\n", message);
    });
    int failed = 0;
    try
    {
        nvrhi::AutoPtr<app::DeviceManager> deviceManager = nvrhi::TakeOver(app::DeviceManager::Create(api));
        CHECK(deviceManager != nullptr);
        app::DeviceCreationParameters params;
        params.headlessDevice = true;
        params.enableDebugRuntime = true;
        params.enableGPUValidation = false;
        params.enableWarningsAsErrors = false;
        params.enableNvrhiValidationLayer = true;
        params.backBufferWidth = 64;
        params.backBufferHeight = 64;
        CHECK(deviceManager->CreateInstance(params));
        {
            std::vector<app::AdapterInfo> adapters;
            if (deviceManager->EnumerateAdapters(adapters))
                for (int i = 0; i < int(adapters.size()); ++i)
                    if (adapters[i].name.find("NVIDIA") != std::string::npos)
                    {
                        params.adapterIndex = i;
                        break;
                    }
        }
        CHECK(deviceManager->CreateHeadlessDevice(params));
        std::printf("device: %s\n", deviceManager->GetRendererString());
        nvrhi::IDevice* device = deviceManager->GetDevice();
        struct GpuCase { const char* name; void (*run)(nvrhi::IDevice*, nvrhi::GraphicsAPI); };
        const GpuCase cases[] = {
            { "s1_permanent_state_from_prepare",
              [](nvrhi::IDevice* d, nvrhi::GraphicsAPI) { test_s1_permanent_state_from_prepare(d); } },
            { "s1_mesh_after_load", test_s1_mesh_after_load },
        };
        for (const GpuCase& c : cases)
        {
            try
            {
                c.run(device, api);
                std::printf("%s: passed\n", c.name);
            }
            catch (const std::runtime_error& err)
            {
                std::fprintf(stderr, "%s: FAILED at %s\n", c.name, err.what());
                ++failed;
            }
        }
        deviceManager->Shutdown();
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "the device: FAILED at %s\n", err.what());
        ++failed;
    }
    if (g_Errors.load() != 0)
    {
        std::fprintf(stderr, "%u errors were logged\n", g_Errors.load());
        ++failed;
    }
    return failed ? 1 : 0;
}

int main(int argc, char** argv)
{
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        { "e72_r5_runtime_rows_checked", test_e72_r5_runtime_rows_checked },
        { "i6_n2_walk_prepared_again", test_i6_n2_walk_prepared_again },
        { "i6_n3_service_closes_its_scope", test_i6_n3_service_closes_its_scope },
        { "i6_n3_scopes_by_identity", test_i6_n3_scopes_by_identity },
        { "f4_n1_identity_beyond_64", test_f4_n1_identity_beyond_64 },
        { "f4_n1_zero_closes_only_without_identity", test_f4_n1_zero_closes_only_without_identity },
        { "t5_n1_compact_geometry_cap", test_t5_n1_compact_geometry_cap },
        { "t8_4_scope_identity_wrap", test_t8_4_scope_identity_wrap },
        { "s2_first_packet_of_a_pass", test_s2_first_packet_of_a_pass },
        { "f10_compact_records_default", test_f10_compact_records_default },
        { "i7_n1_service_destroyed_while_recording", test_i7_n1_service_destroyed_while_recording },   // last
    };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 2 && std::strcmp(argv[1], "--gpu") == 0)
        return RunGpuCases(argv[2]);
    int failed = 0;
    for (const Case& c : cases)
    {
        if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
            continue;
        try
        {
            c.run();
            CHECK(TakeWarnings() == 0);
            std::printf("%s: passed\n", c.name);
        }
        catch (const std::runtime_error& err)
        {
            std::fprintf(stderr, "%s: FAILED at %s\n", c.name, err.what());
            TakeWarnings();
            ++failed;
        }
    }
    return failed ? 1 : 0;
}
