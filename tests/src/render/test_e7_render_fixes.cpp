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
// one case per backlog item, named after it, device-free. Each fails without its fix.
//   e72_r5_runtime_rows_checked  Debug's leaf check compares runtime-only rows with their mesh (Debug only: Release
//                                builds do not check)
//   i6_n2_walk_prepared_again    a walking strategy prepared again mid-drain hands out the new view's items only
//   i6_n3_service_closes_its_scope  EndFrame closes the service's own render-read scope, not the innermost one
//   i6_n3_scopes_by_identity     render-read scopes closed by identity, under the deferring rules
//   f4_n1_identity_beyond_64     beyond 64 open scopes: no identity; EndRenderRead(0) closes the innermost
//   f4_n1_zero_closes_only_without_identity  0 closes only a scope without identity; a pipelined Extract needs one
//   i7_n1_service_destroyed_while_recording  the service's state is kept, with a warning (last: the process defers)

#include <donut/core/log.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/tests/utils.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
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
