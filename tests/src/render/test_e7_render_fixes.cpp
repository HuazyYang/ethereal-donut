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

#include <donut/core/log.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
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

int main(int argc, char** argv)
{
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        { "e72_r5_runtime_rows_checked", test_e72_r5_runtime_rows_checked },
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
