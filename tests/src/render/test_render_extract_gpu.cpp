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

// The render-graph path of the GPU scene on a device (aggregate docs/plans/2026-11-21-donut-ecs-e5.md, item E5.4),
// under the debug runtime and the nvrhi validation layer:
//
//     test_render_extract_gpu d3d11 | d3d12 | vulkan
//
// A Scene on the graph path (SetGpuSceneUploadMode(Scatter) before anything is made) is driven as the render service
// drives it at extraction (CreateMeshBufferObjects, GpuScene::Update, CreateBuffers, BindMeshInstanceBuffers), its
// buffers imported, and GpuSceneUploadPass added to a render graph with readback passes after it. The script, 16
// frames: moves of instances and of parents, a detach and re-attach, growth past the 1,024-row granularity, a mesh
// attached after the graph's refresh, idle frames, a frame with nothing but a material edit (asset rows only), SetLeaf to
// a curve mesh and to a mesh of another geometry count (keep-previous row records), a material domain change, a frame
// whose Prepare ran (simulated on a command list that is never executed) but that is not executed, with growth, a mesh
// attached and a material edited, and the resend after it, with growth again and no growth copy. After every executed
// frame the instance, geometry and material buffers read back from the GPU equal the images of a CPU-mode GpuScene
// updated from the same graph, byte for byte (every row CPU mode has written; after the dropped frame the live rows, with
// prevTransform = transform in the resend's frame); every material's constant buffer equals its image row; the index and
// vertex buffers of the meshes made in the late frames, and of the loaded ones, hold their data. Every plan passes
// VerifyPlan. Bindless buffers off, then on (not D3D11); each driven by hand (the dispatch narrowed to 2 groups in X, so
// that it wraps into Y), then through RenderService::Extract / AddUploadPasses / EndFrame after Scene::FinishedLoading
// (its load-time upload: the first frame is no resend), with item E5.1's device-only checks. Then the skinning pass of
// item E5.5 against Scene::UpdateSkinnedMeshes (RunSkinning): skinned vertex buffers and joint buffers byte-identical
// to the CPU path's after every executed frame, the first frame, a skinned instance attached at runtime, a detach and
// re-attach, and a dropped frame that keeps a first frame for the next one. A GPU program: one at a time.

#include <donut/app/ApplicationBase.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/render/GpuSceneUploadPass.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphLambdaPass.h>

using namespace donut::math;
#include <donut/shaders/bindless.h>
#include <donut/shaders/gpu_scene_scatter_cb.h>
#include <donut/shaders/material_cb.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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
    std::atomic<uint32_t> g_Errors{ 0 };
    std::atomic<uint32_t> g_Warnings{ 0 };

    void check(bool condition, const std::string& what)
    {
        if (!condition)
            throw std::runtime_error("check failed: " + what);
    }

    struct NoData
    {
        uint32_t unused;
    };

    // A mesh with its own buffers' CPU data, and a copy of that data (the scene frees it once written).
    struct TestMesh
    {
        nvrhi::AutoPtr<MeshInfo> mesh;
        std::vector<float3> positions;
        std::vector<float2> texcoords;
        std::vector<uint32_t> indices;
    };

    TestMesh MakeMesh(const std::string& name, uint32_t geometryCount, Material* a, Material* b,
                      MeshType type = MeshType::Triangles)
    {
        auto buffers = MAKE_RC_OBJ_PTR(BufferGroup);
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->name = name;
        mesh->type = type;
        mesh->buffers = buffers;
        for (uint32_t g = 0; g < geometryCount; ++g)
        {
            const uint32_t base = uint32_t(buffers->positionData.size());
            buffers->positionData.push_back(float3(0.f, 0.f, float(g)));
            buffers->positionData.push_back(float3(1.f, float(name.size()), float(g)));
            buffers->positionData.push_back(float3(0.f, 1.f, float(g) + 0.5f));
            buffers->texcoord1Data.insert(buffers->texcoord1Data.end(), { float2(0.f), float2(1.f, 0.f), float2(0.f, 1.f) });
            for (uint32_t i = 0; i < 3; ++i)
                buffers->indexData.push_back(base + i);
            auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
            geometry->material = (g & 1) ? b : a;
            geometry->indexOffsetInMesh = 3 * g;
            geometry->vertexOffsetInMesh = base;
            geometry->numIndices = 3;
            geometry->numVertices = 3;
            geometry->objectSpaceBounds = box3(float3(0.f), float3(1.f));
            mesh->geometries.push_back(geometry);
        }
        mesh->totalIndices = 3 * geometryCount;
        mesh->totalVertices = 3 * geometryCount;
        mesh->objectSpaceBounds = box3(float3(0.f), float3(1.f, 1.f, float(geometryCount)));
        return TestMesh{ mesh, buffers->positionData, buffers->texcoord1Data, buffers->indexData };
    }

    nvrhi::BufferHandle MakeReadback(nvrhi::IDevice* device, uint64_t bytes, bool hostKept = false)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = bytes;
        desc.cpuAccess = nvrhi::CpuAccessMode::Read;
        desc.initialState = nvrhi::ResourceStates::CopyDest;   // imported in it (not host-kept), or host-kept in it
        desc.keepInitialState = hostKept;
        desc.debugName = "GpuSceneReadback";
        nvrhi::BufferHandle buffer;
        device->createBuffer(desc, &buffer);
        return buffer;
    }

    // A copy of `source` into a host-readable buffer, as a graph pass.
    void AddReadback(IRenderGraph* graph, ResourceId source, nvrhi::IBuffer* readback, uint64_t bytes, const char* name)
    {
        ResourceId target = graph->ImportBuffer(readback, nvrhi::ResourceStates::CopyDest, nvrhi::ResourceStates::CopyDest);
        AddLambdaPass<NoData>(graph, PassDesc{ name, PassType::Copy, PassFlags::None, 0 },
            [&](ISetupContext* setup, NoData&)
            {
                setup->Read(source, render::Access::CopySource, nvrhi::ShaderType::None, {});
                target = setup->Write(target, render::Access::CopyDest, nvrhi::ShaderType::None, {}, AccessFlags::FullWrite);
            },
            [source, target, bytes](IPassContext* context, const NoData&)
            {
                context->GetCommandList()->copyBuffer(context->GetBuffer(target), 0, context->GetBuffer(source), 0, bytes);
            });
    }

    template <typename T>
    std::vector<T> Read(nvrhi::IDevice* device, nvrhi::IBuffer* readback)
    {
        std::vector<T> rows(size_t(readback->getDesc().byteSize / sizeof(T)));
        const void* data = device->mapBuffer(readback, nvrhi::CpuAccessMode::Read);
        check(data != nullptr, "map a readback buffer");
        std::memcpy(rows.data(), data, rows.size() * sizeof(T));
        device->unmapBuffer(readback);
        return rows;
    }

    // The bytes of a buffer outside the graph. `permanent`: the buffer has that permanent state (the mesh buffers),
    // which the tracker cannot leave: explicit transitions to CopySource and back, with automatic barriers off.
    std::vector<uint8_t> ReadBuffer(nvrhi::IDevice* device, nvrhi::IBuffer* buffer,
                                    nvrhi::ResourceStates permanent = nvrhi::ResourceStates::Unknown)
    {
        const uint64_t bytes = buffer->getDesc().byteSize;
        nvrhi::BufferHandle readback = MakeReadback(device, bytes, true);
        nvrhi::CommandListHandle commandList;
        device->createCommandList(nvrhi::CommandListParameters(), &commandList);
        commandList->open();
        if (permanent != nvrhi::ResourceStates::Unknown)
        {
            commandList->setEnableAutomaticBarriers(false);
            const nvrhi::ResourceBarrier before =
                nvrhi::ResourceBarrier::Transition(buffer, permanent, nvrhi::ResourceStates::CopySource);
            commandList->barriers(&before, 1);
            commandList->copyBuffer(readback, 0, buffer, 0, bytes);
            const nvrhi::ResourceBarrier after =
                nvrhi::ResourceBarrier::Transition(buffer, nvrhi::ResourceStates::CopySource, permanent);
            commandList->barriers(&after, 1);
            commandList->setEnableAutomaticBarriers(true);
        }
        else
            commandList->copyBuffer(readback, 0, buffer, 0, bytes);
        commandList->close();
        device->executeCommandList(commandList);
        device->waitForIdle();
        return Read<uint8_t>(device, readback);
    }

    // The mesh's index and vertex buffers hold its data (positions and first texture coordinates).
    void CheckMeshData(nvrhi::IDevice* device, const TestMesh& test, const std::string& where)
    {
        const BufferGroup* buffers = test.mesh->buffers;
        check(buffers && buffers->indexBuffer && buffers->vertexBuffer, where + ": " + test.mesh->name + " has buffers");
        auto permanent = [](nvrhi::IBuffer* buffer, nvrhi::ResourceStates state)
        {
            return buffer->getDesc().isAccelStructBuildInput ? state | nvrhi::ResourceStates::AccelStructBuildInput : state;
        };
        const std::vector<uint8_t> indices = ReadBuffer(device, buffers->indexBuffer,
            permanent(buffers->indexBuffer, nvrhi::ResourceStates::IndexBuffer | nvrhi::ResourceStates::ShaderResource));
        check(indices.size() >= test.indices.size() * sizeof(uint32_t) &&
              std::memcmp(indices.data(), test.indices.data(), test.indices.size() * sizeof(uint32_t)) == 0,
              where + ": " + test.mesh->name + " index buffer");
        const std::vector<uint8_t> vertices = ReadBuffer(device, buffers->vertexBuffer,
            permanent(buffers->vertexBuffer, nvrhi::ResourceStates::VertexBuffer | nvrhi::ResourceStates::ShaderResource));
        const nvrhi::BufferRange& positions = buffers->getVertexBufferRange(VertexAttribute::Position);
        const nvrhi::BufferRange& texcoords = buffers->getVertexBufferRange(VertexAttribute::TexCoord1);
        check(positions.byteOffset + test.positions.size() * sizeof(float3) <= vertices.size() &&
              std::memcmp(vertices.data() + positions.byteOffset, test.positions.data(),
                          test.positions.size() * sizeof(float3)) == 0, where + ": " + test.mesh->name + " positions");
        check(texcoords.byteOffset + test.texcoords.size() * sizeof(float2) <= vertices.size() &&
              std::memcmp(vertices.data() + texcoords.byteOffset, test.texcoords.data(),
                          test.texcoords.size() * sizeof(float2)) == 0, where + ": " + test.mesh->name + " texcoords");
    }

    // Every material's own constant buffer equals its row of the CPU-mode image (the value with this graph's slot).
    void CheckMaterialConstants(nvrhi::IDevice* device, SceneGraph* graph, const GpuScene& reference,
                                const std::string& where)
    {
        for (Material* material : graph->GetMaterials())
        {
            check(material->materialConstants != nullptr, where + ": " + material->name + " has a constant buffer");
            const std::vector<uint8_t> bytes = ReadBuffer(device, material->materialConstants);
            const uint32_t slot = uint32_t(material->materialID);
            check(slot < reference.GetMaterialRowCount() && bytes.size() >= sizeof(MaterialConstants) &&
                  std::memcmp(bytes.data(), &reference.GetMaterialRows()[slot], sizeof(MaterialConstants)) == 0,
                  where + ": " + material->name + "'s constant buffer");
        }
    }

    enum class Compared { Written, Live };

    struct Fixture
    {
        nvrhi::IDevice* device = nullptr;
        nvrhi::AutoPtr<ShaderFactory> shaderFactory;
        nvrhi::BindingLayoutHandle bindlessLayout;
        nvrhi::AutoPtr<DescriptorTableManager> descriptorTable;
        nvrhi::AutoPtr<TextureCache> textureCache;
        nvrhi::AutoPtr<vfs::NativeFileSystem> nativeFS;

        Fixture(nvrhi::IDevice* d, nvrhi::GraphicsAPI api, bool bindless) : device(d)
        {
            nativeFS = MAKE_RC_OBJ_PTR(vfs::NativeFileSystem);
            auto rootFS = MAKE_RC_OBJ_PTR(vfs::RootFileSystem);
            rootFS->mount("/shaders/donut",
                          app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(api));
            shaderFactory = MAKE_RC_OBJ_PTR(ShaderFactory, device, rootFS, "/shaders");
            if (bindless)
            {
                nvrhi::BindlessLayoutDesc layoutDesc;
                layoutDesc.visibility = nvrhi::ShaderType::All;
                layoutDesc.maxCapacity = 1024;
                layoutDesc.registerSpaces = { nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
                                              nvrhi::BindingLayoutItem::Texture_SRV(2) };
                device->createBindlessLayout(layoutDesc, &bindlessLayout);
                descriptorTable = MAKE_RC_OBJ_PTR(DescriptorTableManager, device, bindlessLayout);
            }
            textureCache = MAKE_RC_OBJ_PTR(TextureCache, device, nativeFS, descriptorTable);
        }

        nvrhi::AutoPtr<Scene> MakeScene(bool graphPath)
        {
            auto scene = MAKE_RC_OBJ_PTR(Scene, device, shaderFactory, nativeFS, textureCache, descriptorTable, nullptr);
            if (graphPath)
                check(scene->SetGpuSceneUploadMode(GpuSceneUploadMode::Scatter) == nvrhi::FS_OK, "graph path");
            SceneGraph* graph = scene->CreateSceneGraph();
            auto root = MAKE_RC_OBJ_PTR(SceneGraphNode);
            root->SetName("root");
            graph->SetRootNode(root);
            return scene;
        }
    };

    void Run(app::DeviceManager* deviceManager, nvrhi::GraphicsAPI api, bool bindless, bool useService)
    {
        nvrhi::IDevice* device = deviceManager->GetDevice();
        const std::string tag = std::string(app::GetShaderTypeName(api)) + (bindless ? ", bindless" : "") +
                                (useService ? ", service" : "");
        const uint32_t warningsBefore = g_Warnings.load();
        Fixture fixture(device, api, bindless);
        nvrhi::AutoPtr<Scene> scene = fixture.MakeScene(true);
        check(scene->IsOnGraphPath(), "IsOnGraphPath");
        SceneGraph* graph = scene->GetSceneGraph();
        SceneGraphNode* root = graph->GetRootNode();

        auto stone = MAKE_RC_OBJ_PTR(Material);
        stone->name = "stone";
        auto glass = MAKE_RC_OBJ_PTR(Material);
        glass->name = "glass";
        glass->domain = MaterialDomain::AlphaBlended;
        std::vector<TestMesh> meshes = { MakeMesh("a", 1, stone, glass), MakeMesh("bb", 2, stone, glass),
                                         MakeMesh("ccc", 3, glass, stone) };
        TestMesh lateMesh = MakeMesh("late", 2, glass, stone);
        TestMesh lostMesh = MakeMesh("written by a lost Prepare", 2, stone, glass);
        TestMesh curveMesh = MakeMesh("curve", 1, stone, stone, MeshType::CurveLinearSweptSpheres);

        // 3,000 instances: 2,000 under the root, 1,000 under 10 parents
        std::vector<SceneGraphNode*> instances;
        std::vector<SceneGraphNode*> parents;
        auto attach = [&](SceneGraphNode* parent, MeshInfo* mesh, double x, double y)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(double3(x, y, 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh));
            instances.push_back(graph->Attach(parent, node));
        };
        for (int i = 0; i < 2000; ++i)
            attach(root, meshes[i % 3].mesh, double(i % 50), double(i / 50));
        for (int p = 0; p < 10; ++p)
        {
            auto parent = MAKE_RC_OBJ_PTR(SceneGraphNode);
            parent->SetTranslation(double3(0.0, 0.0, double(p)));
            parents.push_back(graph->Attach(root, parent));
            for (int i = 0; i < 100; ++i)
                attach(parents.back(), meshes[(i + p) % 3].mesh, double(i), double(p));
        }

        GpuSceneDesc referenceDesc;
        referenceDesc.geometryAndMaterialBuffers = bindless;
        referenceDesc.structuredInstanceBuffer = api != nvrhi::GraphicsAPI::D3D11;
        referenceDesc.rowGranularity = 1024;
        GpuScene reference(nullptr, referenceDesc);

        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> renderGraph;
        check(CreateResourceRegistry(device, &registry) == nvrhi::FS_OK, "registry");
        check(CreateRenderGraph(registry, &renderGraph) == nvrhi::FS_OK, "render graph");
        BackendProfile profile;
        check(ComputeBackendProfile(device, &profile) == nvrhi::FS_OK, "profile");
        auto uploadPass = MAKE_RC_OBJ_PTR(GpuSceneUploadPass, device, fixture.shaderFactory);
        uploadPass->SetMaxDispatchGroupsX(2);   // the scatter's dispatches wrap into Y
        nvrhi::AutoPtr<RenderService> service;
        if (useService)
        {
            RenderServiceDesc serviceDesc;
            serviceDesc.scene = scene;
            serviceDesc.shaderFactory = fixture.shaderFactory;
            service = MAKE_RC_OBJ_PTR(RenderService, device, serviceDesc);
            // the load: mesh data and the first update on FinishedLoading's command list (deviation 5 of the note)
            scene->FinishedLoading(0);
            check(reference.Update(graph) == nvrhi::FS_OK, tag + ": reference update at the load");
            check(scene->GetGpuScene() && !scene->GetGpuScene()->IsScatterPending(), tag + ": the load is acknowledged");
        }

        nvrhi::AutoPtr<SceneGraphNode> detached;
        nvrhi::CommandListHandle lost;   // frame 12's Prepare, never submitted
        Compared compared = Compared::Written;
        uint32_t lateGeometryFrame = 0;
        uint32_t keepRecords = 0;
        uint32_t wrappedDispatches = 0;
        const uint32_t frames = 16;
        for (uint32_t frame = 1; frame <= frames; ++frame)
        {
            const std::string where = tag + ", frame " + std::to_string(frame);
            // the script; frames 7, 8 and 9 are idle
            const bool idle = frame >= 7 && frame <= 9;
            if (!idle)
            {
                for (size_t i = frame % 7; i < instances.size(); i += 7)
                    instances[i]->SetTranslation(double3(double(i % 50), double(i / 50), 0.01 * frame));
                parents[frame % parents.size()]->SetTranslation(double3(0.1 * frame, 0.0, double(frame % parents.size())));
            }
            if (frame == 3)
                detached = graph->Detach(parents[3]);
            if (frame == 4)   // growth: 3,072 rows to 4,096
                for (int i = 0; i < 300; ++i)
                    attach(root, meshes[i % 3].mesh, double(i % 50), 100.0 + double(i / 50));
            if (frame == 5)
            {
                graph->Attach(root, detached);
                detached = nullptr;
            }
            if (frame == 9)   // nothing but a material edit: asset rows and a constant buffer, no instance record
            {
                glass->roughness = 0.25f;
                glass->dirty = true;
            }
            if (frame == 10)   // the same entities, other heads: keep-previous row records
            {
                instances[5]->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, curveMesh.mesh.Get()));   // flags
                instances[6]->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, meshes[2].mesh.Get()));   // geometry count
            }
            if (frame == 11)   // a material's domain: the instances' content flags, not their rows
            {
                stone->domain = MaterialDomain::AlphaTested;
                stone->dirty = true;
                instances[0]->InvalidateContent();
            }
            if (frame == 12)   // the frame whose Prepare is lost: growth to 5,120 rows and a material edit
            {
                for (int i = 0; i < 900; ++i)
                    attach(root, meshes[i % 3].mesh, double(i % 50), 200.0 + double(i / 50));
                stone->metalness = 0.5f;
                stone->dirty = true;
            }
            if (frame == 13)   // the resend grows again: 6,144 rows, no growth copy
                for (int i = 0; i < 1000; ++i)
                    attach(root, meshes[i % 3].mesh, double(i % 50), 400.0 + double(i / 50));
            scene->RefreshSceneGraph(frame);
            if (frame == 6)   // after the graph's refresh: made and filled at this frame's extraction
                attach(root, lateMesh.mesh, 0.0, 300.0);
            if (frame == 12)
                attach(root, lostMesh.mesh, 0.0, 310.0);

            // extraction: as RenderService::Extract does it, or through the service
            renderGraph->Reset();
            SceneFrame* sceneFrame = nullptr;
            if (useService)
            {
                ExtractDesc extractDesc;
                extractDesc.frameIndex = frame;
                check(service->Extract(renderGraph, nullptr, extractDesc, &sceneFrame) == nvrhi::FS_OK && sceneFrame,
                      where + ": RenderService::Extract");
            }
            else
            {
                check(scene->CreateMeshBufferObjects() == nvrhi::FS_OK, where + ": CreateMeshBufferObjects");
                GpuScene* made = scene->EnsureGpuScene();
                check(made && made->Update(graph) == nvrhi::FS_OK, where + ": Update");
                check(made->CreateBuffers() == nvrhi::FS_OK, where + ": CreateBuffers");
                scene->BindMeshInstanceBuffers();
            }
            GpuScene* gpu = scene->GetGpuScene();
            check(gpu && gpu->GetUploadMode() == GpuSceneUploadMode::Scatter, "a scatter-mode GpuScene");
            check(reference.Update(graph) == nvrhi::FS_OK, where + ": reference Update");
            GpuSceneScatterRecords records;
            gpu->GetScatterRecords(records);
            const uint32_t recordCount = records.settleCount + records.transformCount + records.rowCount;
            const bool executed = frame != 12;
            const bool resend = (frame == 1 && !useService) || frame == 13;
            check(records.resend == resend, where + ": resend");
            const GpuSceneBufferChanges& changes = gpu->GetBufferChanges();
            if (frame == 4 || frame == 12 || frame == 13)
                check(changes.instancesResized, where + ": the instance buffer grows");
            if (frame == 4)
                check(changes.previousInstanceBuffer != nullptr && changes.previousInstanceRows == 3072,
                      where + ": the old buffer is kept for the growth copy");
            if (frame == 13)
                check(changes.previousInstanceBuffer == nullptr && changes.previousInstanceRows == 0,
                      where + ": no growth copy after a resend");
            if (frame == 9)
                check(recordCount == 0 && gpu->HasPendingAssets(), where + ": asset rows only");
            for (uint32_t i = 0; i < records.rowCount; ++i)
                keepRecords += (records.rows[i].flags & GPU_SCENE_ROW_FLAG_KEEP_PREVIOUS) ? 1u : 0u;
            if (frame == 10)
                check(keepRecords >= 2, where + ": keep-previous row records");
            // BindMeshInstanceBuffers after CreateBuffers: every mesh names this frame's instance buffer
            for (MeshInfo* mesh : graph->GetMeshes())
                check(mesh->buffers && mesh->buffers->instanceBuffer.Get() == gpu->GetInstanceBuffer(),
                      where + ": " + mesh->name + "'s instance buffer");

            // the graph
            GpuSceneUploadGraphOutputs outputs;
            if (useService)
            {
                const bool replaced = changes.instancesResized || changes.geometriesResized || changes.materialsResized;
                check(sceneFrame->sceneBuffersReplaced == replaced, where + ": sceneBuffersReplaced");
                if (frame == 4 || frame == 13)
                    check(sceneFrame->sceneBuffersReplaced, where + ": sceneBuffersReplaced on growth");
                check(service->AddUploadPasses(renderGraph, sceneFrame) == nvrhi::FS_OK, where + ": AddUploadPasses");
                const uint32_t passes = renderGraph->GetPassCount();
                check(service->AddUploadPasses(renderGraph, sceneFrame) == nvrhi::FS_OK &&
                      renderGraph->GetPassCount() == passes, where + ": a second AddUploadPasses adds nothing");
                outputs = { sceneFrame->instanceBuffer, sceneFrame->geometryBuffer, sceneFrame->materialBuffer };
                check((outputs.geometryBuffer != InvalidResourceId) == bindless &&
                      (outputs.materialBuffer != InvalidResourceId) == bindless, where + ": the asset imports");
                const ResourceId expected[] = { outputs.instanceBuffer, outputs.geometryBuffer, outputs.materialBuffer };
                uint32_t at = 0;
                for (ResourceId id : expected)
                    if (id != InvalidResourceId)
                        check(at < sceneFrame->bufferReadCount && sceneFrame->bufferReads[at++] == id,
                              where + ": SceneFrame::bufferReads");
                check(at == sceneFrame->bufferReadCount, where + ": SceneFrame::bufferReads count");
            }
            else
            {
                GpuSceneUploadGraphInputs inputs;
                inputs.gpuScene = gpu;
                inputs.scene = scene;
                inputs.bufferChanges = &changes;
                inputs.instanceBuffer = renderGraph->ImportBuffer(gpu->GetInstanceBuffer(), c_GpuSceneRestingState,
                                                                  c_GpuSceneRestingState);
                if (bindless)
                {
                    inputs.geometryBuffer = renderGraph->ImportBuffer(gpu->GetGeometryBuffer(),
                        c_GpuSceneRestingState, c_GpuSceneRestingState);
                    inputs.materialBuffer = renderGraph->ImportBuffer(gpu->GetMaterialBuffer(),
                        c_GpuSceneRestingState, c_GpuSceneRestingState);
                }
                outputs = uploadPass->AddToGraph(renderGraph, inputs);
                if (recordCount > 2 * GPU_SCENE_SCATTER_GROUP_SIZE)
                    ++wrappedDispatches;
            }
            check(outputs.instanceBuffer != InvalidResourceId, where + ": AddToGraph");
            const uint64_t instanceBytes = gpu->GetInstanceBuffer()->getDesc().byteSize;
            nvrhi::BufferHandle instanceReadback = MakeReadback(device, instanceBytes);
            AddReadback(renderGraph, outputs.instanceBuffer, instanceReadback, instanceBytes, "ReadInstances");
            nvrhi::BufferHandle geometryReadback, materialReadback;
            if (bindless)
            {
                geometryReadback = MakeReadback(device, gpu->GetGeometryBuffer()->getDesc().byteSize);
                materialReadback = MakeReadback(device, gpu->GetMaterialBuffer()->getDesc().byteSize);
                AddReadback(renderGraph, outputs.geometryBuffer, geometryReadback,
                            gpu->GetGeometryBuffer()->getDesc().byteSize, "ReadGeometries");
                AddReadback(renderGraph, outputs.materialBuffer, materialReadback,
                            gpu->GetMaterialBuffer()->getDesc().byteSize, "ReadMaterials");
            }
            nvrhi::AutoPtr<ICompiledRenderGraph> plan;
            const nvrhi::FRESULT compiled = renderGraph->Compile(profile, CompileOptions(), &plan);
            nvrhi::string diagnostics;
            renderGraph->GetDiagnostics(diagnostics);
            check(compiled == nvrhi::FS_OK, where + ": Compile: " + std::string(diagnostics.c_str()));
            check(diagnostics.empty(), where + ": diagnostics: " + std::string(diagnostics.c_str()));
            nvrhi::string report;
            const nvrhi::FRESULT verified = VerifyPlan(plan, renderGraph, nullptr, &report);
            check(verified == nvrhi::FS_OK, where + ": VerifyPlan: " + std::string(report.c_str()));
            if (!executed)
            {
                // The upload pass's Prepare ran (its writes, on a command list that is never executed), the frame did
                // not: the mesh data, the asset rows and the material's constant buffer must be written again.
                check(scene->HasMeshBufferDataPending() && gpu->HasPendingAssets(), where + ": writes due");
                // D3D12, Vulkan: a list that is never submitted (not immediate, so that the validation layer lets it be
                // abandoned). D3D11 records on the immediate context: its writes land at once, and the list is executed.
                const bool d3d11 = api == nvrhi::GraphicsAPI::D3D11;
                device->createCommandList(nvrhi::CommandListParameters().setEnableImmediateExecution(d3d11), &lost);
                lost->open();
                scene->WriteMeshBufferData(lost);
                check(gpu->UploadAssets(lost) == nvrhi::FS_OK, where + ": the lost UploadAssets");
                lost->close();
                if (d3d11)
                {
                    device->executeCommandList(lost);
                    lost = nullptr;
                }
                check(!scene->HasMeshBufferDataPending() && !gpu->HasPendingAssets() && !stone->dirty,
                      where + ": the lost Prepare cleared what it wrote");
                if (service)
                    service->EndFrame(false);
                check(gpu->IsScatterPending(), where + ": the records stay pending");
                compared = Compared::Live;
                std::printf("%s: Prepare ran, not executed (%u records)\n", where.c_str(), recordCount);
                continue;
            }
            check(renderGraph->Execute(plan, ExecuteDesc()) == nvrhi::FS_OK, where + ": Execute");
            if (service)
                service->EndFrame(true);
            check(!gpu->IsScatterPending(), where + ": the records are acknowledged");
            device->waitForIdle();

            // the instance rows
            const std::vector<InstanceData> rows = Read<InstanceData>(device, instanceReadback);
            check(rows.size() == reference.GetInstanceRowCount(), where + ": instance buffer size");
            const InstanceData* expectedRows = reference.GetInstanceRows();
            uint32_t comparedRows = 0;
            auto compareRow = [&](uint32_t slot, bool resent)
            {
                InstanceData expected = expectedRows[slot];
                if (resent)
                    expected.prevTransform = expected.transform;
                if (std::memcmp(&rows[slot], &expected, sizeof(expected)) != 0)
                    throw std::runtime_error(where + ": instance slot " + std::to_string(slot) + " differs");
                ++comparedRows;
            };
            if (compared == Compared::Written && !(resend && frame != 1))
            {
                for (uint32_t slot = 0; slot < reference.GetInstanceRowCount(); ++slot)
                    if (reference.GetInstanceSlotEntity(slot) != Entity::Null)
                        compareRow(slot, false);
            }
            else
            {
                for (const auto& instance : graph->GetMeshInstances())
                    compareRow(uint32_t(instance->GetInstanceIndex()), resend);
            }

            // the asset rows, the constant buffers and the mesh data
            if (bindless)
            {
                const std::vector<GeometryData> geometries = Read<GeometryData>(device, geometryReadback);
                check(geometries.size() >= reference.GetGeometryRowCount(), where + ": geometry buffer size");
                check(std::memcmp(geometries.data(), reference.GetGeometryRows(),
                                  reference.GetGeometryRowCount() * sizeof(GeometryData)) == 0, where + ": geometry rows");
                const std::vector<MaterialConstants> materials = Read<MaterialConstants>(device, materialReadback);
                check(materials.size() >= reference.GetMaterialRowCount(), where + ": material buffer size");
                check(std::memcmp(materials.data(), reference.GetMaterialRows(),
                                  reference.GetMaterialRowCount() * sizeof(MaterialConstants)) == 0, where + ": material rows");
                // the late mesh's geometry rows name its buffers in the frame it was attached
                if (frame >= 6 && !lateGeometryFrame)
                {
                    const int slot = lateMesh.mesh->geometries[0]->globalGeometryIndex;
                    const BufferGroup* buffers = lateMesh.mesh->buffers;
                    if (slot >= 0 && size_t(slot) < geometries.size() && buffers->indexBufferDescriptor &&
                        buffers->vertexBufferDescriptor &&
                        geometries[slot].indexBufferIndex == int(buffers->indexBufferDescriptor->Get()) &&
                        geometries[slot].vertexBufferIndex == int(buffers->vertexBufferDescriptor->Get()))
                        lateGeometryFrame = frame;
                }
            }
            CheckMaterialConstants(device, graph, reference, where);
            if (frame == 1)
                for (const TestMesh& mesh : meshes)
                    CheckMeshData(device, mesh, where);
            if (frame == 6)
                CheckMeshData(device, lateMesh, where);
            if (frame == 10)
                CheckMeshData(device, curveMesh, where);
            if (frame == 13)
                CheckMeshData(device, lostMesh, where);
            std::printf("ok: %s: %u records (%u settle, %u transform, %u row%s), %u rows equal\n", where.c_str(),
                        recordCount, records.settleCount, records.transformCount, records.rowCount,
                        records.resend ? ", resend" : "", comparedRows);
            device->runGarbageCollection();
        }
        if (bindless)
        {
            check(lateGeometryFrame == 6, tag + ": the late mesh's geometry rows are complete in its own frame (frame " +
                                              std::to_string(lateGeometryFrame) + ")");
            std::printf("ok: %s: late attach: geometry rows complete in frame %u\n", tag.c_str(), lateGeometryFrame);
        }
        if (!useService)
            check(wrappedDispatches > 0, tag + ": dispatches wrapped into Y");
        std::printf("ok: %s: %u keep-previous row records, %u dispatches wrapped into Y\n", tag.c_str(), keepRecords,
                    wrappedDispatches);
        check(g_Warnings.load() == warningsBefore, tag + ": no warning");

        // Vulkan: the lost list is reused once, as the render graph's executor reuses its lists. A list that is never
        // submitted keeps its command buffer, which holds the list (nvrhi's Vulkan backend): both would leak. Submitting
        // it empty commits the permanent states of its lost recording, which the rewrite has given the buffers since.
        // The nvrhi validation layer warns once that the list was not executed before it was opened again (expected).
        if (lost && api == nvrhi::GraphicsAPI::VULKAN)
        {
            lost->open();
            lost->close();
            device->executeCommandList(lost);
            device->waitForIdle();
            lost = nullptr;
        }

        if (useService)
        {
            // E5.1: a scene whose graph was replaced after the first Extract is refused
            scene->CreateSceneGraph();
            renderGraph->Reset();
            ExtractDesc extractDesc;
            extractDesc.frameIndex = frames + 1;
            check(service->Extract(renderGraph, nullptr, extractDesc, nullptr) == nvrhi::FE_INVALID_ARGS,
                  tag + ": a replaced graph is refused");
            service->EndFrame(false);
            std::printf("ok: %s: a replaced scene graph is refused\n", tag.c_str());
        }
        check(uploadPass->ReleaseStagingBuffer(registry) == (useService ? nvrhi::FE_NOT_FOUND : nvrhi::FS_OK),
              tag + ": ReleaseStagingBuffer");

        service = nullptr;
        uploadPass = nullptr;
        renderGraph = nullptr;
        registry = nullptr;
        scene = nullptr;
        device->waitForIdle();
        device->runGarbageCollection();
    }

    // ------------------------------------------------------------------------------------------ skinning (E5.5)

    // A skin prototype of `vertices` vertices (one triangle per three): positions, normals, first texture coordinates,
    // joint indices into `joints` joints and weights (some zero), tangents and second texture coordinates when asked.
    nvrhi::AutoPtr<MeshInfo> MakeSkinPrototype(const std::string& name, uint32_t vertices, uint32_t joints,
                                               bool tangents, bool texcoord2, Material* material)
    {
        auto buffers = MAKE_RC_OBJ_PTR(BufferGroup);
        for (uint32_t v = 0; v < vertices; ++v)
        {
            const float t = float(v) * 0.37f;
            buffers->positionData.push_back(float3(std::sin(t), float(v) * 0.01f, std::cos(t)));
            buffers->normalData.push_back(vectorToSnorm8(normalize(float3(std::cos(t), 0.5f, std::sin(t)))));
            if (tangents)
                buffers->tangentData.push_back(vectorToSnorm8(float4(-std::sin(t), 0.f, std::cos(t), 1.f)));
            buffers->texcoord1Data.push_back(float2(float(v) / float(vertices), 0.25f));
            if (texcoord2)
                buffers->texcoord2Data.push_back(float2(0.5f, float(v) / float(vertices)));
            buffers->jointData.push_back(dm::vector<uint16_t, 4>(uint16_t(v % joints), uint16_t((v + 1) % joints),
                                                                 uint16_t((v + 2) % joints),
                                                                 uint16_t((v + 3) % joints)));
            float4 weights(1.f + float(v % 3), float(v % 2), (v % 5) ? 0.5f : 0.f, (v % 7) ? 0.f : 0.25f);
            weights /= weights.x + weights.y + weights.z + weights.w;
            buffers->weightData.push_back(weights);
            buffers->indexData.push_back(v);
        }
        auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
        geometry->material = material;
        geometry->numIndices = vertices;
        geometry->numVertices = vertices;
        geometry->objectSpaceBounds = box3(float3(-1.f), float3(1.f, float(vertices) * 0.01f, 1.f));
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->name = name;
        mesh->buffers = buffers;
        mesh->geometries.push_back(geometry);
        mesh->totalIndices = vertices;
        mesh->totalVertices = vertices;
        mesh->objectSpaceBounds = geometry->objectSpaceBounds;
        mesh->isSkinPrototype = true;
        return mesh;
    }

    // A character as an importer builds it, off the graph: a root, a chain of joints (SkinnedMeshReference leaves) and
    // the skin (a SkinnedMeshInstance with the joints and their inverse bind matrices).
    struct Character
    {
        std::string name;
        uint32_t id = 0;   // the same in both scenes: the pose's phase
        nvrhi::AutoPtr<SceneGraphNode> root;
        std::vector<SceneGraphNode*> joints;
        nvrhi::AutoPtr<SkinnedMeshInstance> skinned;
    };

    Character MakeCharacter(SceneGraph* graph, SceneTypeFactory* factory, MeshInfo* prototype, uint32_t jointCount,
                            const std::string& name, double x)
    {
        Character character;
        character.name = name;
        character.root = MAKE_RC_OBJ_PTR(SceneGraphNode);
        character.root->SetName(name);
        character.root->SetTranslation(double3(x, 0.0, 0.0));
        character.skinned = factory->CreateSkinnedMeshInstance(factory, prototype);
        SceneGraphNode* parent = character.root;
        for (uint32_t j = 0; j < jointCount; ++j)
        {
            auto joint = MAKE_RC_OBJ_PTR(SceneGraphNode);
            joint->SetName(name + "_joint" + std::to_string(j));
            joint->SetTranslation(double3(0.0, j ? 0.5 : 0.0, 0.0));
            joint->SetLeaf(MAKE_RC_OBJ_PTR(SkinnedMeshReference, character.skinned.Get()));
            graph->Attach(parent, joint);
            const float4x4 inverseBind = affineToHomogeneous(translation(float3(0.1f, -0.5f * float(j), 0.f)));
            character.skinned->joints.push_back(
                SkinnedMeshJoint{ nvrhi::WeakPtr<SceneGraphNode>(joint.Get()), inverseBind });
            character.joints.push_back(joint);
            parent = joint;
        }
        auto skin = MAKE_RC_OBJ_PTR(SceneGraphNode);
        skin->SetName(name + "_skin");
        skin->SetLeaf(character.skinned);
        graph->Attach(character.root, skin);
        return character;
    }

    // The pose of a character's joints in a frame: the same values in both scenes.
    void Pose(Character& character, uint32_t frame)
    {
        for (size_t j = 0; j < character.joints.size(); ++j)
        {
            const double angle = 0.3 * std::sin(0.7 * double(frame) + double(j) + 1.3 * double(character.id));
            character.joints[j]->SetRotation(rotationQuat(double3(0.0, 0.0, 1.0), angle));
        }
    }

    // One scene of the skinning comparison: the render-graph path (the service) or the CPU path (the reference).
    struct SkinningScene
    {
        nvrhi::AutoPtr<Scene> scene;
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        nvrhi::AutoPtr<Material> material = MAKE_RC_OBJ_PTR(Material);
        nvrhi::AutoPtr<MeshInfo> prototypes[3];
        std::vector<Character> characters;   // in creation order: A, B, C (frame 4), D (frame 7), E (13), F (16)

        SkinningScene(Fixture& fixture, bool graphPath)
        {
            scene = fixture.MakeScene(graphPath);
            material->name = "skin";
            prototypes[0] = MakeSkinPrototype("p0", 700, 6, true, false, material);   // three dispatch groups
            prototypes[1] = MakeSkinPrototype("p1", 300, 4, false, true, material);
            prototypes[2] = MakeSkinPrototype("p2", 512, 5, true, true, material);
        }

        void Add(uint32_t prototype, uint32_t joints, const std::string& name, double x)
        {
            SceneGraph* graph = scene->GetSceneGraph();
            characters.push_back(MakeCharacter(graph, factory, prototypes[prototype], joints, name, x));
            characters.back().id = uint32_t(name[0] - 'A');
            graph->Attach(graph->GetRootNode(), characters.back().root);
        }

        Character* Find(const SkinnedMeshInstance* leaf)
        {
            for (Character& character : characters)
                if (character.skinned.Get() == leaf)
                    return &character;
            return nullptr;
        }

        Character* Find(const std::string& name)
        {
            for (Character& character : characters)
                if (character.name == name)
                    return &character;
            return nullptr;
        }

        // Detaches a character and drops every reference the test holds to it.
        void Remove(const std::string& name)
        {
            for (auto it = characters.begin(); it != characters.end(); ++it)
            {
                if (it->name == name)
                {
                    nvrhi::AutoPtr<SceneGraphNode> detached = scene->GetSceneGraph()->Detach(it->root);
                    characters.erase(it);
                    return;
                }
            }
        }
    };

    // The skinning pass against Scene::UpdateSkinnedMeshes. Two scenes built alike, one on the graph path driven
    // through RenderService::Extract / AddUploadPasses / AddSkinningPasses / EndFrame, one on the CPU path
    // (Scene::Refresh): after every executed frame every skinned vertex buffer (read back through graph passes from
    // SceneFrame's ids) and every joint buffer equal the CPU path's byte for byte. Both load through FinishedLoading,
    // which skins the loaded characters A and B. The script (a frame the graph path drops is skipped by the CPU path):
    //   frames 1-12   A animated in 1-9, B in 1-3, 5 (while detached) and 8-9, the others in their frames up to 9;
    //                 B detached in 5 and attached again in 6; C attached in 4 with a prototype new to the scene; D
    //                 attached in 7 with A's prototype; frame 7 dropped: D's first frame is not consumed and frame 8
    //                 skins it as a first frame; in frame 9, C is detached and released on the graph path between
    //                 AddSkinningPasses and the execution (it was dispatched; review fix 2); 10-12 idle
    //   frames 13-15  A moves in 13; E attached in 13; 13 and 14 dropped (review fix 1: E is never skinned and A's pose
    //                 is pending): 15 skins both
    //   frames 16-20  A moves in 16; F attached in 16; 16 dropped; then a frame-index gap (17 and 18 do not exist);
    //                 19 skins A and F, 20 settles
    //   frames 21-90  idle: every call of AddSkinningPasses ages the binding-set cache, which ends empty (review fix 4)
    // AddSkinningPasses before AddUploadPasses is refused (review fix 3).
    void RunSkinning(app::DeviceManager* deviceManager, nvrhi::GraphicsAPI api, bool bindless)
    {
        nvrhi::IDevice* device = deviceManager->GetDevice();
        const std::string tag =
            std::string(app::GetShaderTypeName(api)) + (bindless ? ", bindless" : "") + ", skinning";
        const uint32_t warningsBefore = g_Warnings.load();
        uint32_t expectedWarnings = 0;
        Fixture fixture(device, api, bindless);
        SkinningScene graphScene(fixture, true);
        SkinningScene cpuScene(fixture, false);
        for (SkinningScene* side : { &graphScene, &cpuScene })
        {
            side->Add(0, 6, "A", 0.0);
            side->Add(1, 4, "B", 3.0);
        }

        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> renderGraph;
        check(CreateResourceRegistry(device, &registry) == nvrhi::FS_OK, "registry");
        check(CreateRenderGraph(registry, &renderGraph) == nvrhi::FS_OK, "render graph");
        BackendProfile profile;
        check(ComputeBackendProfile(device, &profile) == nvrhi::FS_OK, "profile");
        RenderServiceDesc serviceDesc;
        serviceDesc.scene = graphScene.scene;
        serviceDesc.shaderFactory = fixture.shaderFactory;
        auto service = MAKE_RC_OBJ_PTR(RenderService, device, serviceDesc);
        graphScene.scene->FinishedLoading(0);
        cpuScene.scene->FinishedLoading(0);
        device->waitForIdle();
        for (SkinningScene* side : { &graphScene, &cpuScene })
            for (const Character& character : side->characters)
                check(character.skinned->skinningInitialized, tag + ": " + character.name + " skinned at the load");

        auto animated = [](const std::string& name, uint32_t frame)
        {
            if (name == "A")
                return frame <= 9 || frame == 13 || frame == 16;
            if (name == "B")
                return frame <= 3 || frame == 5 || frame == 8 || frame == 9;
            return frame <= 9;
        };
        std::vector<uint32_t> frames;
        for (uint32_t frame = 1; frame <= 16; ++frame)
            frames.push_back(frame);
        for (uint32_t frame = 19; frame <= 90; ++frame)   // 17 and 18: a frame-index gap
            frames.push_back(frame);
        auto dropped = [](uint32_t frame) { return frame == 7 || frame == 13 || frame == 14 || frame == 16; };

        nvrhi::CommandListHandle commandList;
        device->createCommandList(nvrhi::CommandListParameters(), &commandList);
        nvrhi::AutoPtr<SceneGraphNode> detachedB[2];
        uint32_t dispatchedFrames = 0, firstFrameDispatches = 0, othersImported = 0, setsAtFrame20 = 0;
        std::vector<uint8_t> lastA;
        for (const uint32_t frame : frames)
        {
            const std::string where = tag + ", frame " + std::to_string(frame);
            const bool executed = !dropped(frame);
            const bool compared = frame <= 20 || frame == frames.back();
            for (SkinningScene* side : { &graphScene, &cpuScene })
            {
                const size_t s = side == &graphScene ? 0 : 1;
                SceneGraph* graph = side->scene->GetSceneGraph();
                if (frame == 4)
                    side->Add(2, 5, "C", 6.0);
                if (frame == 7)
                    side->Add(0, 6, "D", 9.0);
                if (frame == 13)
                    side->Add(1, 4, "E", 12.0);
                if (frame == 16)
                    side->Add(2, 5, "F", 15.0);
                if (frame == 5)
                    detachedB[s] = graph->Detach(side->Find("B")->root);
                if (frame == 6)
                {
                    graph->Attach(graph->GetRootNode(), detachedB[s]);
                    detachedB[s] = nullptr;
                }
                if (frame == 9 && side == &cpuScene)   // the graph path removes C mid-frame, below
                    side->Remove("C");
                for (Character& character : side->characters)
                {
                    if (animated(character.name, frame))
                        Pose(character, frame);
                }
            }

            // the CPU path: Scene::Refresh; the dropped frames do not exist there
            if (executed)
            {
                cpuScene.scene->RefreshSceneGraph(frame);
                commandList->open();
                cpuScene.scene->RefreshBuffers(commandList, frame);
                commandList->close();
                device->executeCommandList(commandList);
            }

            // the graph path
            SceneGraph* graph = graphScene.scene->GetSceneGraph();
            graphScene.scene->RefreshSceneGraph(frame);
            renderGraph->Reset();
            SceneFrame* sceneFrame = nullptr;
            ExtractDesc extractDesc;
            extractDesc.frameIndex = frame;
            check(service->Extract(renderGraph, nullptr, extractDesc, &sceneFrame) == nvrhi::FS_OK && sceneFrame,
                  where + ": Extract");
            const RenderSnapshot& snapshot = *sceneFrame->snapshot;
            // the legacy rule's instances and those never skinned are all selected (and those pending from a dropped
            // frame, which the CPU path cannot tell)
            for (const auto& leaf : graph->GetSkinnedMeshInstances())
            {
                if (leaf->GetLastUpdateFrameIndex() + 1 >= frame || !leaf->skinningInitialized)
                {
                    bool found = false;
                    for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
                        found = found || snapshot.skinnedInstances[i].leaf == leaf.Get();
                    check(found, where + ": the selection");
                }
            }
            if (frame == 1)
            {
                // review fix 3: skinning before the upload pass is refused, with a warning
                check(service->AddSkinningPasses(renderGraph, sceneFrame) == nvrhi::FE_INVALID_ARGS &&
                      renderGraph->GetPassCount() == 0, where + ": AddSkinningPasses before AddUploadPasses");
                ++expectedWarnings;
            }
            check(service->AddUploadPasses(renderGraph, sceneFrame) == nvrhi::FS_OK, where + ": AddUploadPasses");
            const uint32_t passesBefore = renderGraph->GetPassCount();
            check(service->AddSkinningPasses(renderGraph, sceneFrame) == nvrhi::FS_OK, where + ": AddSkinningPasses");
            const uint32_t skinningPasses = renderGraph->GetPassCount() - passesBefore;
            check(skinningPasses == (snapshot.skinnedInstanceCount ? 1u : 0u), where + ": one compute use");
            check(service->AddSkinningPasses(renderGraph, sceneFrame) == nvrhi::FS_OK &&
                  renderGraph->GetPassCount() == passesBefore + skinningPasses, where + ": a second call adds nothing");
            if (frame == 20)
                setsAtFrame20 = service->GetStats().skinningBindingSets;

            // SceneFrame's ids: the snapshot's instances (written), then the others in the graph's order (imported)
            const auto& skinnedLeaves = graph->GetSkinnedMeshInstances();
            check(sceneFrame->skinnedVertexBufferCount == skinnedLeaves.size(),
                  where + ": one id per skinned instance");
            std::vector<const SkinnedMeshInstance*> order;
            for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
                order.push_back(snapshot.skinnedInstances[i].leaf);
            for (const auto& leaf : skinnedLeaves)
                if (std::find(order.begin(), order.end(), leaf.Get()) == order.end())
                    order.push_back(leaf.Get());
            for (uint32_t i = 0; i < sceneFrame->skinnedVertexBufferCount; ++i)
            {
                const ResourceId id = sceneFrame->skinnedVertexBuffers[i];
                check(id != InvalidResourceId, where + ": a skinned buffer imported");
                othersImported += i < snapshot.skinnedInstanceCount ? 0u : 1u;
            }
            uint32_t reads = 0;
            for (uint32_t i = 0; i < sceneFrame->bufferReadCount; ++i)
                for (uint32_t k = 0; k < sceneFrame->skinnedVertexBufferCount; ++k)
                    reads += sceneFrame->bufferReads[i] == sceneFrame->skinnedVertexBuffers[k] ? 1u : 0u;
            check(reads == sceneFrame->skinnedVertexBufferCount,
                  where + ": SceneFrame::bufferReads has the skinned buffers");
            for (uint32_t i = 0; executed && i < snapshot.skinnedInstanceCount; ++i)
                firstFrameDispatches += snapshot.skinnedInstances[i].firstFrame ? 1u : 0u;
            dispatchedFrames += executed && snapshot.skinnedInstanceCount ? 1u : 0u;

            std::vector<nvrhi::BufferHandle> readbacks;
            for (uint32_t i = 0; i < sceneFrame->skinnedVertexBufferCount; ++i)
            {
                const uint64_t bytes = order[i]->GetMesh()->buffers->vertexBuffer->getDesc().byteSize;
                readbacks.push_back(MakeReadback(device, bytes));
                AddReadback(renderGraph, sceneFrame->skinnedVertexBuffers[i], readbacks.back(), bytes, "ReadSkinned");
            }

            // review fix 2: C, dispatched this frame, is detached and released before the graph compiles and executes
            const SkinnedMeshInstance* removed = nullptr;
            if (frame == 9)
            {
                const Character* c = graphScene.Find("C");
                check(c != nullptr && c->skinned->skinningInitialized, where + ": C is no first frame");
                removed = c->skinned.Get();
                bool dispatched = false;
                for (uint32_t i = 0; i < snapshot.skinnedInstanceCount; ++i)
                    dispatched = dispatched || snapshot.skinnedInstances[i].leaf == removed;
                check(dispatched, where + ": C is dispatched");
                graphScene.Remove("C");
#ifndef NDEBUG
                ++expectedWarnings;   // a scene-graph write inside the render-read scope
#endif
            }

            nvrhi::AutoPtr<ICompiledRenderGraph> plan;
            const nvrhi::FRESULT compiled = renderGraph->Compile(profile, CompileOptions(), &plan);
            nvrhi::string diagnostics;
            renderGraph->GetDiagnostics(diagnostics);
            check(compiled == nvrhi::FS_OK, where + ": Compile: " + std::string(diagnostics.c_str()));
            check(diagnostics.empty(), where + ": diagnostics: " + std::string(diagnostics.c_str()));
            nvrhi::string report;
            check(VerifyPlan(plan, renderGraph, nullptr, &report) == nvrhi::FS_OK,
                  where + ": VerifyPlan: " + std::string(report.c_str()));
            if (!executed)
            {
                service->EndFrame(false);
                const char* first = frame == 7 ? "D" : frame == 16 ? "F" : "E";
                const Character* character = graphScene.Find(first);
                check(character && !character->skinned->skinningInitialized,
                      where + ": a dropped first frame is not consumed (" + first + ")");
                std::printf("ok: %s: compiled, not executed; %u skinned instances selected\n", where.c_str(),
                            snapshot.skinnedInstanceCount);
                continue;
            }
            check(renderGraph->Execute(plan, ExecuteDesc()) == nvrhi::FS_OK, where + ": Execute");
            service->EndFrame(true);
            device->waitForIdle();
            for (const auto& leaf : skinnedLeaves)
                check(leaf->skinningInitialized, where + ": skinningInitialized after an executed frame");

            // every skinned buffer and every joint buffer against the CPU path's
            check(cpuScene.scene->GetSceneGraph()->GetSkinnedMeshInstances().size() == skinnedLeaves.size(),
                  where + ": the same skinned instances");
            for (uint32_t i = 0; compared && i < sceneFrame->skinnedVertexBufferCount; ++i)
            {
                if (order[i] == removed)
                    continue;   // released; the CPU path did not have it this frame
                const Character* character = graphScene.Find(order[i]);
                check(character != nullptr, where + ": a known skinned instance");
                const Character* reference = cpuScene.Find(character->name);
                check(reference != nullptr, where + ": " + character->name + " on the CPU path");
                const std::vector<uint8_t> graphBytes = Read<uint8_t>(device, readbacks[i]);
                const std::vector<uint8_t> cpuBytes =
                    ReadBuffer(device, reference->skinned->GetMesh()->buffers->vertexBuffer);
                check(graphBytes.size() == cpuBytes.size() &&
                      std::memcmp(graphBytes.data(), cpuBytes.data(), cpuBytes.size()) == 0,
                      where + ": " + character->name + "'s skinned vertex buffer");
                check(std::any_of(graphBytes.begin(), graphBytes.end(), [](uint8_t b) { return b != 0; }),
                      where + ": " + character->name + "'s skinned vertex buffer has data");
                // A's positions move in every animated frame, and its previous positions follow one frame behind
                if (character->name == "A")
                {
                    if (!lastA.empty() && frame <= 9)
                        check(graphBytes != lastA, where + ": A's skinned vertices moved");
                    lastA = graphBytes;
                }
                const std::vector<uint8_t> graphJoints = ReadBuffer(device, character->skinned->jointBuffer);
                const std::vector<uint8_t> cpuJoints = ReadBuffer(device, reference->skinned->jointBuffer);
                check(graphJoints == cpuJoints, where + ": " + character->name + "'s joint buffer");
            }
            if (compared)
                std::printf("ok: %s: %u of %zu skinned instances dispatched, every skinned and joint buffer equal\n",
                            where.c_str(), snapshot.skinnedInstanceCount, skinnedLeaves.size());
            device->runGarbageCollection();
        }
        // first-frame dispatches in executed frames: C (4), D (8), E (15), F (19)
        check(dispatchedFrames >= 12 && firstFrameDispatches == 4 && othersImported > 0,
              tag + ": frames with dispatches " + std::to_string(dispatchedFrames) + ", first-frame dispatches " +
                  std::to_string(firstFrameDispatches));
        const uint32_t setsAtEnd = service->GetStats().skinningBindingSets;
        check(setsAtFrame20 >= 5 && setsAtEnd == 0, tag + ": binding-set cache " + std::to_string(setsAtFrame20) +
                                                        " at frame 20, " + std::to_string(setsAtEnd) + " at the end");
        std::printf("ok: %s: binding-set cache %u at frame 20, %u after 70 idle frames\n", tag.c_str(), setsAtFrame20,
                    setsAtEnd);
        check(g_Warnings.load() == warningsBefore + expectedWarnings, tag + ": the expected warnings only");

        service = nullptr;
        renderGraph = nullptr;
        registry = nullptr;
        commandList = nullptr;
        device->waitForIdle();
        device->runGarbageCollection();
    }

    // E5.1's device-only checks: a scene not on the graph path is refused; an empty scene on it adds nothing and warns
    // about nothing.
    void ServiceChecks(app::DeviceManager* deviceManager, nvrhi::GraphicsAPI api)
    {
        nvrhi::IDevice* device = deviceManager->GetDevice();
        const std::string tag = std::string(app::GetShaderTypeName(api)) + ", service checks";
        Fixture fixture(device, api, false);
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> renderGraph;
        check(CreateResourceRegistry(device, &registry) == nvrhi::FS_OK, "registry");
        check(CreateRenderGraph(registry, &renderGraph) == nvrhi::FS_OK, "render graph");
        ExtractDesc extractDesc;
        extractDesc.frameIndex = 1;
        {
            nvrhi::AutoPtr<Scene> cpuScene = fixture.MakeScene(false);
            RenderServiceDesc desc;
            desc.scene = cpuScene;
            desc.shaderFactory = fixture.shaderFactory;
            auto service = MAKE_RC_OBJ_PTR(RenderService, device, desc);
            renderGraph->Reset();
            check(service->Extract(renderGraph, nullptr, extractDesc, nullptr) == nvrhi::FE_INVALID_ARGS,
                  tag + ": a scene not on the graph path is refused");
            service->EndFrame(false);
        }
        {
            nvrhi::AutoPtr<Scene> empty = fixture.MakeScene(true);
            RenderServiceDesc desc;
            desc.scene = empty;
            desc.shaderFactory = fixture.shaderFactory;
            auto service = MAKE_RC_OBJ_PTR(RenderService, device, desc);
            const uint32_t warnings = g_Warnings.load();
            for (uint32_t frame = 1; frame <= 2; ++frame)
            {
                empty->RefreshSceneGraph(frame);
                renderGraph->Reset();
                SceneFrame* sceneFrame = nullptr;
                extractDesc.frameIndex = frame;
                check(service->Extract(renderGraph, nullptr, extractDesc, &sceneFrame) == nvrhi::FS_OK && sceneFrame,
                      tag + ": Extract of an empty scene");
                check(service->AddUploadPasses(renderGraph, sceneFrame) == nvrhi::FS_OK, tag + ": AddUploadPasses");
                check(sceneFrame->instanceBuffer == InvalidResourceId && renderGraph->GetPassCount() == 0,
                      tag + ": an empty scene adds nothing");
                service->EndFrame(true);
                check(empty->GetGpuScene() && !empty->GetGpuScene()->IsScatterPending(), tag + ": acknowledged at once");
            }
            check(g_Warnings.load() == warnings, tag + ": no warning for an empty scene");
        }
        renderGraph = nullptr;
        registry = nullptr;
        device->waitForIdle();
        std::printf("ok: %s: a CPU-mode scene is refused; an empty scene adds nothing\n", tag.c_str());
    }
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    log::SetCallback([](log::Severity severity, const char* text) {
        if (severity >= log::Severity::Error)
            ++g_Errors;
        else if (severity == log::Severity::Warning)
            ++g_Warnings;
        if (severity >= log::Severity::Warning)
            std::printf("LOG: %s\n", text);
    });
    try
    {
        check(argc == 2, "specify d3d11, d3d12 or vulkan");
        const std::string backend = argv[1];
        const nvrhi::GraphicsAPI api = backend == "d3d11" ? nvrhi::GraphicsAPI::D3D11
            : backend == "d3d12" ? nvrhi::GraphicsAPI::D3D12 : nvrhi::GraphicsAPI::VULKAN;
        nvrhi::AutoPtr<app::DeviceManager> deviceManager = nvrhi::TakeOver(app::DeviceManager::Create(api));
        check(deviceManager != nullptr, "device manager");
        app::DeviceCreationParameters params;
        params.headlessDevice = true;
        params.enableDebugRuntime = true;
        params.enableGPUValidation = api == nvrhi::GraphicsAPI::D3D12;
        params.enableWarningsAsErrors = false;
        params.enableNvrhiValidationLayer = true;
        params.backBufferWidth = 64;
        params.backBufferHeight = 64;
        check(deviceManager->CreateInstance(params), "graphics instance");
        {
            // released before the device: the adapters hold DXGI objects that the debug layer reports at shutdown
            std::vector<app::AdapterInfo> adapters;
            if (deviceManager->EnumerateAdapters(adapters))
            {
                for (int i = 0; i < int(adapters.size()); ++i)
                {
                    if (adapters[i].name.find("NVIDIA") != std::string::npos)
                    {
                        params.adapterIndex = i;
                        break;
                    }
                }
            }
        }
        check(deviceManager->CreateHeadlessDevice(params), "headless device");
        std::printf("device: %s\n", deviceManager->GetRendererString());

        for (bool useService : { false, true })
        {
            Run(deviceManager, api, false, useService);
            if (api != nvrhi::GraphicsAPI::D3D11)
                Run(deviceManager, api, true, useService);
        }
        ServiceChecks(deviceManager, api);
        RunSkinning(deviceManager, api, false);
        if (api != nvrhi::GraphicsAPI::D3D11)
            RunSkinning(deviceManager, api, true);

        deviceManager->Shutdown();
        deviceManager = nullptr;
        check(g_Errors == 0, "no errors were logged (" + std::to_string(g_Errors.load()) + ")");
        std::printf("PASS\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
