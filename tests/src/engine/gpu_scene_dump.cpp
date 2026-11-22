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

// The A/B check of plan step E4, "buffers equal to the E0 build's after slot remapping" (design 11.2): loads a scene
// through donut's Scene on a headless device, runs a scripted sequence of frames (animations, a moving node, a subgraph
// detached and attached again), reads the instance, geometry and material buffers back from the GPU after
// Scene::Refresh and writes them in the canonical form of GpuSceneDump.h. It uses only API that the E0 tree has too, so
// the same source builds against both trees (the E0 build from outside that tree); the two outputs must be equal.
//
//     gpu_scene_dump --scene <file> --shaders <framework shader dir of the API> --out <file> [--api d3d12|vulkan]
//                    [--adapter <name substring>] [--frames N] [--bindless] [--move <node path>] [--detach <node path>]
//     gpu_scene_dump ... --bindless --check-late-attach
//         attaches a mesh between Scene::RefreshSceneGraph and RefreshBuffers, and checks that the next frame's
//         geometry row has the mesh's buffer descriptors (review fix 1 of step E4); exit code 0 when it does.
//     gpu_scene_dump ... --path graph
//         (plan step E5, sub-plan item E5.4; this tree only, GPU_SCENE_DUMP_GRAPH_PATH) the render-graph path: the
//         scene in scatter mode (Scene::SetGpuSceneUploadMode before loading), driven as the render service drives it
//         at extraction, the buffers imported into a render graph, GpuSceneUploadPass, and the readbacks as graph passes
//         after it. Its dumps must equal --path immediate (the default, Scene::Refresh). With --check-late-attach the
//         mesh attached after the graph's refresh must have its geometry rows in the same frame.
//
// A GPU program: one at a time. Not a ctest test.

#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#if GPU_SCENE_DUMP_GRAPH_PATH
#include <donut/engine/GpuScene.h>
#include <donut/render/GpuSceneUploadPass.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphLambdaPass.h>
#endif

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>
#include "GpuSceneDump.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    struct Options
    {
        nvrhi::GraphicsAPI api = nvrhi::GraphicsAPI::D3D12;
        std::string adapter = "NVIDIA";
        std::string scene;
        std::string shaders;
        std::string out;
        std::string move;
        std::string detach;
        uint32_t frames = 8;
        bool bindless = false;
        bool checkLateAttach = false;
        bool graphPath = false;   // --path graph
    };

    // Copies `bytes` of `source` into a CPU-readable buffer on `commandList`; read it after the list ran.
    nvrhi::BufferHandle Readback(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, nvrhi::IBuffer* source)
    {
        if (!source)
            return nullptr;
        nvrhi::BufferDesc desc;
        desc.byteSize = source->getDesc().byteSize;
        desc.cpuAccess = nvrhi::CpuAccessMode::Read;
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        desc.keepInitialState = true;
        desc.debugName = "GpuSceneDumpReadback";
        nvrhi::BufferHandle staging;
        if (NVRHI_FAILED(device->createBuffer(desc, &staging)))
            return nullptr;
        commandList->copyBuffer(staging, 0, source, 0, desc.byteSize);
        return staging;
    }

    template <typename T>
    std::vector<T> Read(nvrhi::IDevice* device, nvrhi::IBuffer* staging)
    {
        std::vector<T> rows;
        if (!staging)
            return rows;
        const size_t bytes = size_t(staging->getDesc().byteSize);
        rows.resize(bytes / sizeof(T));
        if (const void* data = device->mapBuffer(staging, nvrhi::CpuAccessMode::Read))
        {
            std::memcpy(rows.data(), data, rows.size() * sizeof(T));
            device->unmapBuffer(staging);
        }
        return rows;
    }

#if GPU_SCENE_DUMP_GRAPH_PATH
    // The render-graph path of one frame: extraction as RenderService::Extract does it, the GpuScene's buffers imported
    // (not host-kept in scatter mode: engine::c_GpuSceneRestingState), GpuSceneUploadPass, then graph passes that copy
    // the instance, geometry and material buffers into readback buffers. Returns them (null where a buffer is absent).
    struct GraphPath
    {
        nvrhi::IDevice* device = nullptr;
        nvrhi::AutoPtr<render::IResourceRegistry> registry;
        nvrhi::AutoPtr<render::IRenderGraph> graph;
        nvrhi::AutoPtr<render::GpuSceneUploadPass> pass;
        render::BackendProfile profile;

        struct NoData
        {
            uint32_t unused;
        };

        bool Init(nvrhi::IDevice* d, ShaderFactory* shaderFactory)
        {
            device = d;
            if (NVRHI_FAILED(render::CreateResourceRegistry(device, &registry)) ||
                NVRHI_FAILED(render::CreateRenderGraph(registry, &graph)) ||
                NVRHI_FAILED(render::ComputeBackendProfile(device, &profile)))
                return false;
            pass = MAKE_RC_OBJ_PTR(render::GpuSceneUploadPass, device, shaderFactory);
            return true;
        }

        nvrhi::BufferHandle AddReadback(render::ResourceId source)
        {
            if (source == render::InvalidResourceId)
                return nullptr;
            nvrhi::BufferDesc sourceDesc;
            graph->GetBufferDesc(source, sourceDesc);
            nvrhi::BufferDesc desc;
            desc.byteSize = sourceDesc.byteSize;
            desc.cpuAccess = nvrhi::CpuAccessMode::Read;
            desc.initialState = nvrhi::ResourceStates::CopyDest;   // imported in it: not host-kept
            desc.debugName = "GpuSceneDumpReadback";
            nvrhi::BufferHandle readback;
            if (NVRHI_FAILED(device->createBuffer(desc, &readback)))
                return nullptr;
            render::ResourceId target = graph->ImportBuffer(readback, nvrhi::ResourceStates::CopyDest,
                                                            nvrhi::ResourceStates::CopyDest);
            const uint64_t bytes = desc.byteSize;
            render::AddLambdaPass<NoData>(graph, render::PassDesc{ "GpuSceneDumpReadback", render::PassType::Copy,
                                                                   render::PassFlags::None, 0 },
                [&](render::ISetupContext* setup, NoData&)
                {
                    setup->Read(source, render::Access::CopySource, nvrhi::ShaderType::None, {});
                    target = setup->Write(target, render::Access::CopyDest, nvrhi::ShaderType::None, {},
                                          render::AccessFlags::FullWrite);
                },
                [source, target, bytes](render::IPassContext* context, const NoData&)
                {
                    context->GetCommandList()->copyBuffer(context->GetBuffer(target), 0, context->GetBuffer(source), 0,
                                                          bytes);
                });
            return readback;
        }

        // False when the frame could not be built or run (the diagnostics are printed).
        bool Frame(Scene* scene, nvrhi::BufferHandle& instances, nvrhi::BufferHandle& geometries,
                   nvrhi::BufferHandle& materials)
        {
            SceneGraph* sceneGraph = scene->GetSceneGraph();
            scene->CreateMeshBufferObjects();
            GpuScene* gpu = scene->EnsureGpuScene();
            if (!gpu || NVRHI_FAILED(gpu->Update(sceneGraph)) || NVRHI_FAILED(gpu->CreateBuffers()))
                return false;
            scene->BindMeshInstanceBuffers();

            graph->Reset();
            render::GpuSceneUploadGraphInputs inputs;
            inputs.gpuScene = gpu;
            inputs.scene = scene;
            inputs.bufferChanges = &gpu->GetBufferChanges();
            auto import = [&](nvrhi::IBuffer* buffer)
            {
                return buffer ? graph->ImportBuffer(buffer, c_GpuSceneRestingState, c_GpuSceneRestingState)
                              : render::InvalidResourceId;
            };
            inputs.instanceBuffer = import(gpu->GetInstanceBuffer());
            inputs.geometryBuffer = import(gpu->GetGeometryBuffer());
            inputs.materialBuffer = import(gpu->GetMaterialBuffer());
            const render::GpuSceneUploadGraphOutputs outputs = pass->AddToGraph(graph, inputs);
            instances = AddReadback(outputs.instanceBuffer);
            geometries = AddReadback(outputs.geometryBuffer);
            materials = AddReadback(outputs.materialBuffer);

            nvrhi::AutoPtr<render::ICompiledRenderGraph> plan;
            const nvrhi::FRESULT compiled = graph->Compile(profile, render::CompileOptions(), &plan);
            nvrhi::string diagnostics;
            graph->GetDiagnostics(diagnostics);
            if (!diagnostics.empty())
                std::fprintf(stderr, "render graph: %s\n", diagnostics.c_str());
            if (NVRHI_FAILED(compiled) || NVRHI_FAILED(graph->Execute(plan, render::ExecuteDesc())))
                return false;
            return !gpu->IsScatterPending();
        }

        void Release()
        {
            pass = nullptr;
            graph = nullptr;
            registry = nullptr;
        }
    };
#endif
}

int main(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (arg == "--api")
        {
            const std::string api = next();
            options.api = api == "vulkan" || api == "vk" ? nvrhi::GraphicsAPI::VULKAN : nvrhi::GraphicsAPI::D3D12;
        }
        else if (arg == "--adapter") options.adapter = next();
        else if (arg == "--scene") options.scene = next();
        else if (arg == "--shaders") options.shaders = next();
        else if (arg == "--out") options.out = next();
        else if (arg == "--move") options.move = next();
        else if (arg == "--detach") options.detach = next();
        else if (arg == "--frames") options.frames = uint32_t(std::stoul(next()));
        else if (arg == "--bindless") options.bindless = true;
        else if (arg == "--check-late-attach") options.checkLateAttach = true;
        else if (arg == "--path") options.graphPath = next() == "graph";
    }
#if !GPU_SCENE_DUMP_GRAPH_PATH
    if (options.graphPath)
    {
        std::fprintf(stderr, "--path graph needs a build with GPU_SCENE_DUMP_GRAPH_PATH\n");
        return 2;
    }
#endif
    if (options.scene.empty() || options.shaders.empty() || (options.out.empty() && !options.checkLateAttach))
    {
        std::fprintf(stderr, "usage: gpu_scene_dump --scene <file> --shaders <dir> --out <file> [options]\n");
        return 2;
    }

    nvrhi::AutoPtr<app::DeviceManager> deviceManager = app::DeviceManager::Create(options.api);
    app::DeviceCreationParameters params;
    params.headlessDevice = true;
    if (!deviceManager->CreateInstance(params))
    {
        std::fprintf(stderr, "cannot create the graphics instance\n");
        return 1;
    }
    std::vector<app::AdapterInfo> adapters;
    if (deviceManager->EnumerateAdapters(adapters))
    {
        for (int i = 0; i < int(adapters.size()); ++i)
        {
            if (adapters[i].name.find(options.adapter) != std::string::npos)
            {
                params.adapterIndex = i;
                break;
            }
        }
    }
    if (!deviceManager->CreateHeadlessDevice(params))
    {
        std::fprintf(stderr, "cannot create the device\n");
        return 1;
    }
    nvrhi::IDevice* device = deviceManager->GetDevice();

    auto nativeFS = MAKE_RC_OBJ_PTR(vfs::NativeFileSystem);
    auto rootFS = MAKE_RC_OBJ_PTR(vfs::RootFileSystem);
    rootFS->mount("/shaders/donut", options.shaders);
    auto shaderFactory = MAKE_RC_OBJ_PTR(ShaderFactory, device, rootFS, "/shaders");

    nvrhi::BindingLayoutHandle bindlessLayout;
    nvrhi::AutoPtr<DescriptorTableManager> descriptorTable;
    if (options.bindless)
    {
        nvrhi::BindlessLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::All;
        layoutDesc.firstSlot = 0;
        layoutDesc.maxCapacity = 1024;
        layoutDesc.registerSpaces = { nvrhi::BindingLayoutItem::RawBuffer_SRV(1), nvrhi::BindingLayoutItem::Texture_SRV(2) };
        device->createBindlessLayout(layoutDesc, &bindlessLayout);
        descriptorTable = MAKE_RC_OBJ_PTR(DescriptorTableManager, device, bindlessLayout);
    }
    auto textureCache = MAKE_RC_OBJ_PTR(TextureCache, device, nativeFS, descriptorTable);
    auto scene = MAKE_RC_OBJ_PTR(Scene, device, shaderFactory, nativeFS, textureCache, descriptorTable, nullptr);
#if GPU_SCENE_DUMP_GRAPH_PATH
    GraphPath graphPath;
    if (options.graphPath)
    {
        // the GpuScene's mode is chosen before anything makes it
        if (NVRHI_FAILED(scene->SetGpuSceneUploadMode(GpuSceneUploadMode::Scatter)) ||
            !graphPath.Init(device, shaderFactory))
        {
            std::fprintf(stderr, "cannot set up the render-graph path\n");
            return 1;
        }
    }
#endif
    if (!scene->Load(options.scene))
    {
        std::fprintf(stderr, "cannot load %s\n", options.scene.c_str());
        return 1;
    }
    scene->FinishedLoading(0);
    device->waitForIdle();

    SceneGraph* graph = scene->GetSceneGraph();
    nvrhi::AutoPtr<SceneGraphNode> moving = options.move.empty() ? nullptr : graph->FindNode(options.move);
    nvrhi::AutoPtr<SceneGraphNode> detached;
    nvrhi::AutoPtr<SceneGraphNode> detachTarget = options.detach.empty() ? nullptr : graph->FindNode(options.detach);
    const dm::double3 home = moving ? moving->GetTranslation() : dm::double3(0.0);

    nvrhi::CommandListHandle commandList;
    device->createCommandList(nvrhi::CommandListParameters(), &commandList);

    if (options.checkLateAttach)
    {
        // A triangle with its own buffers, attached after this frame's RefreshSceneGraph: its buffers are made by the
        // next frame's CreateMeshBuffers, after which its geometry row must name them.
        auto material = MAKE_RC_OBJ_PTR(Material);
        material->name = "late";
        auto buffers = MAKE_RC_OBJ_PTR(BufferGroup);
        buffers->indexData = { 0, 1, 2 };
        buffers->positionData = { dm::float3(0.f), dm::float3(1.f, 0.f, 0.f), dm::float3(0.f, 1.f, 0.f) };
        auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
        geometry->material = material;
        geometry->numIndices = 3;
        geometry->numVertices = 3;
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->name = "late";
        mesh->buffers = buffers;
        mesh->geometries.push_back(geometry);
        mesh->totalIndices = 3;
        mesh->totalVertices = 3;
        mesh->objectSpaceBounds = dm::box3(dm::float3(0.f), dm::float3(1.f));

        int rowIndex = -1, rowVertex = -1;
        uint32_t rowPosition = 0;
        uint32_t completeFrame = 0;
        for (uint32_t frame = 1; frame <= 2; ++frame)
        {
            scene->RefreshSceneGraph(frame);
            if (frame == 1)
                graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
            nvrhi::BufferHandle geometries;
#if GPU_SCENE_DUMP_GRAPH_PATH
            if (options.graphPath)
            {
                nvrhi::BufferHandle instances, materials;
                if (!graphPath.Frame(scene, instances, geometries, materials))
                {
                    std::fprintf(stderr, "frame %u: the render graph did not run\n", frame);
                    return 1;
                }
                device->waitForIdle();
            }
            else
#endif
            {
                commandList->open();
                scene->RefreshBuffers(commandList, frame);
                geometries = Readback(device, commandList, scene->GetGeometryBuffer());
                commandList->close();
                device->executeCommandList(commandList);
                device->waitForIdle();
            }
            const std::vector<GeometryData> rows = Read<GeometryData>(device, geometries);
            const int slot = geometry->globalGeometryIndex;
            if (slot >= 0 && size_t(slot) < rows.size())
            {
                rowIndex = rows[slot].indexBufferIndex;
                rowVertex = rows[slot].vertexBufferIndex;
                rowPosition = rows[slot].positionOffset;
            }
            std::printf("frame %u: geometry slot %d: indexBufferIndex %d vertexBufferIndex %d positionOffset %x\n",
                        frame, slot, rowIndex, rowVertex, rowPosition);
            const bool complete = rowIndex >= 0 && rowVertex >= 0 && rowPosition != ~0u &&
                buffers->indexBufferDescriptor && rowIndex == int(buffers->indexBufferDescriptor->Get()) &&
                buffers->vertexBufferDescriptor && rowVertex == int(buffers->vertexBufferDescriptor->Get());
            if (complete && !completeFrame)
                completeFrame = frame;
        }
        // the graph path makes the mesh's buffers at the extraction of the frame it was attached in, before the
        // GpuScene's update: its rows are complete at once
        const bool ok = completeFrame != 0 && (!options.graphPath || completeFrame == 1) && rowIndex >= 0;
        std::printf("late attach: %s (rows complete in frame %u)\n", ok ? "ok" : "FAILED", completeFrame);
#if GPU_SCENE_DUMP_GRAPH_PATH
        graphPath.Release();
#endif
        scene = nullptr;
        textureCache = nullptr;
        descriptorTable = nullptr;
        commandList = nullptr;
        device->waitForIdle();
        deviceManager->Shutdown();
        return ok ? 0 : 1;
    }

    FILE* out = std::fopen(options.out.c_str(), "wb");
    if (!out)
        return 1;

    for (uint32_t frame = 1; frame <= options.frames; ++frame)
    {
        // a fixed time step; every animation, then the scripted edits
        const float time = float(frame) / 60.f;
        for (const auto& animation : graph->GetAnimations())
        {
            const float duration = animation->GetDuration();
            (void)animation->Apply(duration > 0.f ? std::fmod(time, duration) : 0.f);
        }
        if (moving && frame % 3 != 0)   // moves two frames of three: the third shows the previous transforms settle
            moving->SetTranslation(home + dm::double3(0.25 * frame, 0.0, 0.0));
        if (detachTarget && frame == 3)
            detached = graph->Detach(detachTarget);
        if (detached && frame == 5)
        {
            graph->Attach(graph->GetRootNode(), detached);
            detached = nullptr;
        }

        nvrhi::BufferHandle instances, geometries, materials;
#if GPU_SCENE_DUMP_GRAPH_PATH
        if (options.graphPath)
        {
            scene->RefreshSceneGraph(frame);
            if (!graphPath.Frame(scene, instances, geometries, materials))
            {
                std::fprintf(stderr, "frame %u: the render graph did not run\n", frame);
                return 1;
            }
            commandList->open();   // the constant buffers below (host-kept)
        }
        else
#endif
        {
            commandList->open();
            scene->Refresh(commandList, frame);
            instances = Readback(device, commandList, scene->GetInstanceBuffer());
            geometries = Readback(device, commandList, scene->GetGeometryBuffer());
            materials = Readback(device, commandList, scene->GetMaterialBuffer());
        }
        // Without bindless buffers, each material's own constant buffer, by its slot.
        std::vector<std::pair<int, nvrhi::BufferHandle>> constants;
        if (!materials)
        {
            for (const auto& material : graph->GetMaterials())
                constants.emplace_back(material->materialID, Readback(device, commandList, material->materialConstants));
        }
        commandList->close();
        device->executeCommandList(commandList);
        device->waitForIdle();

        const std::vector<InstanceData> instanceRows = Read<InstanceData>(device, instances);
        const std::vector<GeometryData> geometryRows = Read<GeometryData>(device, geometries);
        std::vector<MaterialConstants> materialRows = Read<MaterialConstants>(device, materials);
        for (const auto& [slot, buffer] : constants)
        {
            const std::vector<MaterialConstants> row = Read<MaterialConstants>(device, buffer);
            if (slot < 0 || row.empty())
                continue;
            if (size_t(slot) >= materialRows.size())
                materialRows.resize(size_t(slot) + 1, MaterialConstants());
            materialRows[slot] = row[0];
        }
        gpu_scene_dump::Dump(out, frame, graph, instanceRows.data(), instanceRows.size(), geometryRows.data(),
                             geometryRows.size(), materialRows.data(), materialRows.size());
        device->runGarbageCollection();
    }
    std::fclose(out);

#if GPU_SCENE_DUMP_GRAPH_PATH
    graphPath.Release();
#endif
    scene = nullptr;
    textureCache = nullptr;
    descriptorTable = nullptr;
    commandList = nullptr;
    device->waitForIdle();
    deviceManager->Shutdown();
    std::printf("wrote %s\n", options.out.c_str());
    return 0;
}
