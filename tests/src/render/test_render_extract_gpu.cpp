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
//     test_render_extract_gpu d3d11 | d3d12 | vulkan [steady]      steady: the allocation counts alone
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
// re-attach, and a dropped frame that keeps a first frame for the next one. Then, on a second device without the nvrhi
// validation layer, the steady-state heap allocations of extraction, upload, skinning and recording (RunSteadyState).
// A GPU program: one at a time.

#include <donut/app/ApplicationBase.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <donut/render/DepthPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/GBufferFillPass.h>
#include <donut/render/GeometryPasses.h>
#include <donut/render/GpuSceneUploadPass.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphLambdaPass.h>
#include <donut/render/RenderGraphWorkerPool.h>

#include "../../../src/engine/EcsWorldImpl.h"   // ecs::GetHeapAllocationCount

using namespace donut::math;
#include <donut/shaders/bindless.h>
#include <donut/shaders/gbuffer_cb.h>
#include <donut/shaders/gpu_scene_scatter_cb.h>
#include <donut/shaders/material_cb.h>
#include <donut/shaders/skinning_cb.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <malloc.h>
#endif

using namespace donut;
using namespace donut::engine;
using namespace donut::render;

// ------------------------------------------------------------------------------- allocation counter (RunSteadyState)

namespace allocation_counter
{
    enum Kind : uint32_t { OperatorNew, NvrhiAllocator, KindCount };
    const char* const c_KindNames[KindCount] = { "operator new", "nvrhi allocator" };

    std::atomic<bool> g_Counting{ false };
    std::atomic<uint32_t> g_Counts[KindCount];
    struct Record
    {
        Kind kind;
        uint32_t depth;
        void* stack[60];
    };
    std::atomic<uint32_t> g_RecordCount{ 0 };
    Record g_Records[2048];   // of one scope

    void Count(Kind kind) noexcept
    {
        if (!g_Counting.load(std::memory_order_relaxed))
            return;
        g_Counts[kind].fetch_add(1, std::memory_order_relaxed);
        const uint32_t index = g_RecordCount.fetch_add(1, std::memory_order_relaxed);
        if (index >= std::size(g_Records))
            return;
        Record& record = g_Records[index];
        record.kind = kind;
#ifdef _WIN32
        record.depth = CaptureStackBackTrace(0, DWORD(std::size(record.stack)), record.stack, nullptr);
#endif
    }

    // Counts every thread's allocations while it lives.
    struct Scope
    {
        Scope()
        {
            for (std::atomic<uint32_t>& count : g_Counts)
                count = 0;
            g_RecordCount = 0;
            g_Counting = true;
        }
        ~Scope() { g_Counting = false; }
    };

    // Counts nvrhi's default allocator while it lives: RC objects (MAKE_RC_OBJ), nvrhi's containers and the ECS storage
    // allocate through it, in nvrhi_core.dll, which the operator new below does not reach. The allocator is one
    // stateless object per process, called through its virtual functions: its vtable pointer is exchanged for that of a
    // forwarding class.
    class AllocatorHook
    {
        static inline std::atomic<void*> s_Original{ nullptr };   // an object with the original's vtable pointer alone

        struct Forwarding final : nvrhi::IMemoryAllocator   // its `this` is nvrhi's object: no members
        {
            static nvrhi::IMemoryAllocator* Original() noexcept
            {
                return reinterpret_cast<nvrhi::IMemoryAllocator*>(&s_Original);
            }
            void* Allocate(size_t size) noexcept override
            {
                Count(NvrhiAllocator);
                return Original()->Allocate(size);
            }
            void Free(void* memory) noexcept override { Original()->Free(memory); }
            void* AllocateAligned(size_t size, size_t alignment) noexcept override
            {
                Count(NvrhiAllocator);
                return Original()->AllocateAligned(size, alignment);
            }
            void FreeAligned(void* memory) noexcept override { Original()->FreeAligned(memory); }
        };
        Forwarding m_Forwarding;

        static std::atomic<void*>& VtableOf(nvrhi::IMemoryAllocator* allocator) noexcept
        {
            static_assert(sizeof(std::atomic<void*>) == sizeof(void*) && sizeof(Forwarding) == sizeof(void*));
            return *reinterpret_cast<std::atomic<void*>*>(allocator);
        }

    public:
        AllocatorHook()
        {
            s_Original = VtableOf(nvrhi::GetDefaultMemAllocator()).load();
            VtableOf(nvrhi::GetDefaultMemAllocator()) = VtableOf(&m_Forwarding).load();
        }
        ~AllocatorHook() { VtableOf(nvrhi::GetDefaultMemAllocator()) = s_Original.load(); }
        AllocatorHook(const AllocatorHook&) = delete;
        AllocatorHook& operator=(const AllocatorHook&) = delete;
    };
}

// This module: donut, the render graph, nvrhi and its backends, which are static libraries. The native runtimes and
// their debug layers allocate in their own modules and are not counted. The library's other forms (array, nothrow,
// sized) call these four.
void* operator new(size_t size)
{
    allocation_counter::Count(allocation_counter::OperatorNew);
    if (void* memory = std::malloc(size ? size : 1))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, std::align_val_t alignment)
{
    allocation_counter::Count(allocation_counter::OperatorNew);
#ifdef _WIN32
    if (void* memory = _aligned_malloc(size ? size : 1, size_t(alignment)))
        return memory;
#else
    if (void* memory = std::aligned_alloc(size_t(alignment), (size + size_t(alignment) - 1) & ~(size_t(alignment) - 1)))
        return memory;
#endif
    throw std::bad_alloc();
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::align_val_t) noexcept
{
#ifdef _WIN32
    _aligned_free(memory);
#else
    std::free(memory);
#endif
}

namespace
{
    std::atomic<uint32_t> g_Errors{ 0 };
    std::atomic<uint32_t> g_Warnings{ 0 };
    // The Khronos layer's performance warning about GBufferFillPass with the input assembler and without motion vectors
    // (its input layout has the previous positions, which that vertex shader does not read): older than step E5, and
    // counted apart (RunGeometry).
    std::atomic<uint32_t> g_UnconsumedAttributeWarnings{ 0 };

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

    // ------------------------------------------------------------------------------- geometry-pass helpers

    // A mesh of quads side by side along x, one geometry per material, with normals, tangents and texture coordinates,
    // appended to `buffers` (meshes may share a group, as an importer makes the meshes of one file); the indices are
    // relative to each geometry's first vertex.
    nvrhi::AutoPtr<MeshInfo> MakeQuadMesh(const std::string& name, const std::vector<Material*>& materials,
                                          BufferGroup* buffers)
    {
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->name = name;
        mesh->buffers = buffers;
        mesh->vertexOffset = uint32_t(buffers->positionData.size());
        mesh->indexOffset = uint32_t(buffers->indexData.size());
        box3 bounds = box3::empty();
        for (uint32_t g = 0; g < uint32_t(materials.size()); ++g)
        {
            const uint32_t firstVertex = uint32_t(buffers->positionData.size());
            const uint32_t firstIndex = uint32_t(buffers->indexData.size());
            const float x = 1.2f * float(g);
            const float3 corners[4] = { float3(x, 0.f, 0.f), float3(x + 1.f, 0.f, 0.f), float3(x + 1.f, 1.f, 0.f),
                                        float3(x, 1.f, 0.f) };
            for (const float3& corner : corners)
            {
                buffers->positionData.push_back(corner);
                buffers->normalData.push_back(vectorToSnorm8(float3(0.f, 0.f, -1.f)));
                buffers->tangentData.push_back(vectorToSnorm8(float4(1.f, 0.f, 0.f, 1.f)));
                buffers->texcoord1Data.push_back(float2(corner.x - x, corner.y));
            }
            for (uint32_t index : { 0u, 2u, 1u, 0u, 3u, 2u })
                buffers->indexData.push_back(index);
            auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
            geometry->material = materials[g];
            geometry->indexOffsetInMesh = firstIndex - mesh->indexOffset;
            geometry->vertexOffsetInMesh = firstVertex - mesh->vertexOffset;
            geometry->numIndices = 6;
            geometry->numVertices = 4;
            geometry->objectSpaceBounds = box3(float3(x, 0.f, -0.01f), float3(x + 1.f, 1.f, 0.01f));
            bounds |= geometry->objectSpaceBounds;
            mesh->geometries.push_back(geometry);
        }
        mesh->totalIndices = uint32_t(buffers->indexData.size()) - mesh->indexOffset;
        mesh->totalVertices = uint32_t(buffers->positionData.size()) - mesh->vertexOffset;
        mesh->objectSpaceBounds = bounds;
        return mesh;
    }

    // The state a target rests in between frames (its import's entry and exit).
    nvrhi::ResourceStates TargetState(nvrhi::Format format)
    {
        return nvrhi::getFormatInfo(format).hasDepth ? nvrhi::ResourceStates::DepthWrite
                                                     : nvrhi::ResourceStates::RenderTarget;
    }

    // A render target imported into the graph (not host-kept), moved into its resting state once.
    nvrhi::TextureHandle MakeTarget(nvrhi::IDevice* device, nvrhi::Format format, const char* name)
    {
        nvrhi::TextureDesc desc;
        desc.width = 192;
        desc.height = 128;
        desc.format = format;
        desc.isRenderTarget = true;
        desc.isTypeless = nvrhi::getFormatInfo(format).hasDepth;   // D3D11: a depth format with a shader view
        desc.debugName = name;
        nvrhi::TextureHandle texture;
        device->createTexture(desc, &texture);
        check(texture != nullptr, std::string("create ") + name);
        nvrhi::CommandListHandle commandList;
        device->createCommandList(nvrhi::CommandListParameters(), &commandList);
        commandList->open();
        commandList->setEnableAutomaticBarriers(false);
        const nvrhi::ResourceBarrier barrier =
            nvrhi::ResourceBarrier::Transition(texture, nvrhi::ResourceStates::Unknown, TargetState(format));
        commandList->barriers(&barrier, 1);
        commandList->setEnableAutomaticBarriers(true);
        commandList->close();
        device->executeCommandList(commandList);
        device->waitForIdle();
        return texture;
    }

    // The texels of a target outside the graph, rows packed.
    std::vector<uint8_t> ReadTexture(nvrhi::IDevice* device, nvrhi::ITexture* texture)
    {
        const nvrhi::TextureDesc& desc = texture->getDesc();
        const nvrhi::ResourceStates state = TargetState(desc.format);
        nvrhi::TextureDesc stagingDesc = desc;
        stagingDesc.isRenderTarget = false;
        stagingDesc.debugName = "ReadTexture";
        nvrhi::StagingTextureHandle staging;
        device->createStagingTexture(stagingDesc, nvrhi::CpuAccessMode::Read, &staging);
        check(staging != nullptr, "a staging texture");
        nvrhi::CommandListHandle commandList;
        device->createCommandList(nvrhi::CommandListParameters(), &commandList);
        commandList->open();
        commandList->beginTrackingTextureState(texture, nvrhi::AllSubresources, state);
        commandList->copyTexture2(staging, nvrhi::TextureSlice(), texture, nvrhi::TextureSlice());
        commandList->setTextureState(texture, nvrhi::AllSubresources, state);
        commandList->commitBarriers();
        commandList->close();
        device->executeCommandList(commandList);
        device->waitForIdle();
        size_t pitch = 0;
        const auto* data = static_cast<const uint8_t*>(
            device->mapStagingTexture(staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, pitch));
        check(data != nullptr, "map a staging texture");
        const size_t rowBytes = size_t(desc.width) * nvrhi::getFormatInfo(desc.format).bytesPerBlock;
        std::vector<uint8_t> texels(rowBytes * desc.height);
        for (uint32_t y = 0; y < desc.height; ++y)
            std::memcpy(texels.data() + rowBytes * y, data + pitch * y, rowBytes);
        device->unmapStagingTexture(staging);
        return texels;
    }

    // ------------------------------------------------------------------------------- buffer-group switch (older than E5)

    // The input binding set across a change of buffer group without a change of material, with vertex pulling (the
    // passes' default): RenderView calls SetupInputBuffers alone then, and DepthPass, GBufferFillPass and
    // ForwardShadingPass used to keep the previous group's input set (its vertex and instance buffers) in the graphics
    // state, so a draw read another group's vertex buffer at its own group's offsets. Found by item E5.3's GPU test;
    // fixed in the three passes' SetupInputBuffers. A Scene on the CPU path (host-kept buffers, drawn outside a graph):
    // a quad mesh in one buffer group, a second geometry of a mesh in another group (its vertices start past the first
    // group's), and two skinned instances of one prototype in different poses (each has its own skinned buffer group and
    // the prototype's material). All four draws share one material. Each pass draws them in one RenderView, and again
    // one RenderView per draw (each starts from SetupMaterial) into a second set of targets: the targets are equal byte
    // for byte, and not empty. Without the fix the depth, G-buffer and forward images differ.
    void RunBufferGroupSwitch(app::DeviceManager* deviceManager, nvrhi::GraphicsAPI api)
    {
        nvrhi::IDevice* device = deviceManager->GetDevice();
        const std::string tag = std::string(app::GetShaderTypeName(api)) + ", buffer-group switch";
        const uint32_t warningsBefore = g_Warnings.load();
        Fixture fixture(device, api, false);
        nvrhi::AutoPtr<Scene> scene = fixture.MakeScene(false);
        SceneGraph* graph = scene->GetSceneGraph();
        SceneGraphNode* root = graph->GetRootNode();

        auto material = MAKE_RC_OBJ_PTR(Material);
        material->name = "shared";
        material->doubleSided = true;
        material->baseOrDiffuseColor = float3(0.9f, 0.4f, 0.2f);
        auto groupA = MAKE_RC_OBJ_PTR(BufferGroup);
        auto groupB = MAKE_RC_OBJ_PTR(BufferGroup);
        nvrhi::AutoPtr<MeshInfo> meshA = MakeQuadMesh("a", { material }, groupA);
        nvrhi::AutoPtr<MeshInfo> meshB = MakeQuadMesh("b", { material, material }, groupB);
        auto attach = [&](MeshInfo* mesh, double3 position)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(position);
            auto leaf = MAKE_RC_OBJ_PTR(MeshInstance, mesh);
            node->SetLeaf(leaf);
            graph->Attach(root, node);
            return leaf;
        };
        nvrhi::AutoPtr<MeshInstance> instanceA = attach(meshA, double3(-2.6, 0.2, 5.0));
        nvrhi::AutoPtr<MeshInstance> instanceB = attach(meshB, double3(-0.6, 0.2, 5.5));
        nvrhi::AutoPtr<MeshInfo> prototype = MakeSkinPrototype("p", 300, 4, true, false, material);
        auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        Character characters[2] = { MakeCharacter(graph, factory, prototype, 4, "C", 0.0),
                                    MakeCharacter(graph, factory, prototype, 4, "D", 0.0) };
        for (uint32_t c = 0; c < 2; ++c)
        {
            characters[c].id = 2 * c;
            characters[c].root->SetTranslation(double3(c ? 2.0 : -1.5, -2.5, 6.0));
            Pose(characters[c], 5);
            graph->Attach(root, characters[c].root);
        }
        scene->FinishedLoading(0);
        device->waitForIdle();

        // one material, four buffer groups, in this order
        std::vector<DrawItem> items;
        auto add = [&items](const MeshInstance* instance, uint32_t geometry)
        {
            const MeshInfo* mesh = instance->GetMesh();
            DrawItem item{};
            item.instance = instance;
            item.mesh = mesh;
            item.geometry = mesh->geometries[geometry];
            item.material = item.geometry->material;
            item.buffers = mesh->buffers;
            item.cullMode = nvrhi::RasterCullMode::None;
            items.push_back(item);
        };
        add(instanceA, 0);
        add(instanceB, 1);
        add(characters[0].skinned, 0);
        add(characters[1].skinned, 0);
        check(items[0].buffers != items[1].buffers && items[2].buffers != items[3].buffers &&
              items[0].material == items[3].material, tag + ": one material, distinct buffer groups");

        auto view = MAKE_RC_OBJ_PTR(PlanarView);
        view->SetViewport(nvrhi::Viewport(192.f, 128.f));
        view->SetMatrices(affine3::identity(), perspProjD3DStyleReverse(radians(70.f), 1.5f, 0.1f));
        view->UpdateCache();
        nvrhi::AutoPtr<CommonRenderPasses> commonPasses = MAKE_RC_OBJ_PTR(CommonRenderPasses, device, fixture.shaderFactory);
        auto depthPass = MAKE_RC_OBJ_PTR(DepthPass, device, commonPasses);
        depthPass->Init(*fixture.shaderFactory, DepthPass::CreateParameters());
        auto gbufferPass = MAKE_RC_OBJ_PTR(GBufferFillPass, device, commonPasses);
        gbufferPass->Init(*fixture.shaderFactory, GBufferFillPass::CreateParameters());
        auto forwardPass = MAKE_RC_OBJ_PTR(ForwardShadingPass, device, commonPasses);
        forwardPass->Init(*fixture.shaderFactory, ForwardShadingPass::CreateParameters());

        struct Case
        {
            const char* name;
            IGeometryPass* pass;
            std::vector<nvrhi::Format> formats;   // colour targets, then depth
        };
        const Case cases[] = {
            { "DepthPass", depthPass, { nvrhi::Format::D32 } },
            { "GBufferFillPass", gbufferPass,
              { nvrhi::Format::SRGBA8_UNORM, nvrhi::Format::SRGBA8_UNORM, nvrhi::Format::RGBA16_SNORM,
                nvrhi::Format::RGBA16_FLOAT, nvrhi::Format::D32 } },
            { "ForwardShadingPass", forwardPass, { nvrhi::Format::RGBA16_FLOAT, nvrhi::Format::D32 } },
        };
        nvrhi::CommandListHandle commandList;
        device->createCommandList(nvrhi::CommandListParameters(), &commandList);
        for (const Case& c : cases)
        {
            const std::string where = tag + ", " + c.name;
            if (api == nvrhi::GraphicsAPI::D3D11 && c.pass == depthPass.Get())
                continue;   // depth only, which D3D11 does not read back here
            std::vector<nvrhi::TextureHandle> targets[2];
            nvrhi::FramebufferHandle framebuffers[2];
            for (uint32_t side = 0; side < 2; ++side)
            {
                nvrhi::FramebufferDesc framebufferDesc;
                for (nvrhi::Format format : c.formats)
                {
                    targets[side].push_back(MakeTarget(device, format, side ? "SeparateDraws" : "OneRenderView"));
                    if (nvrhi::getFormatInfo(format).hasDepth)
                        framebufferDesc.setDepthAttachment(targets[side].back());
                    else
                        framebufferDesc.addColorAttachment(targets[side].back());
                }
                device->createFramebuffer(framebufferDesc, &framebuffers[side]);
                check(framebuffers[side] != nullptr, where + ": framebuffer");
            }

            commandList->open();
            for (uint32_t side = 0; side < 2; ++side)
            {
                for (nvrhi::ITexture* texture : targets[side])
                {
                    const nvrhi::Format format = texture->getDesc().format;
                    commandList->beginTrackingTextureState(texture, nvrhi::AllSubresources, TargetState(format));
                    if (nvrhi::getFormatInfo(format).hasDepth)
                        commandList->clearDepthStencilTexture(texture, nvrhi::AllSubresources, true, 0.f, false, 0);
                    else
                        commandList->clearTextureFloat(texture, nvrhi::AllSubresources, nvrhi::Color(0.f));
                }
            }
            DepthPass::Context depthContext;
            GBufferFillPass::Context gbufferContext;
            ForwardShadingPass::Context forwardContext;
            GeometryPassContext* context = c.pass == depthPass.Get() ? static_cast<GeometryPassContext*>(&depthContext)
                : c.pass == gbufferPass.Get() ? static_cast<GeometryPassContext*>(&gbufferContext)
                : static_cast<GeometryPassContext*>(&forwardContext);
            if (c.pass == forwardPass.Get())
                forwardPass->PrepareLights(forwardContext, commandList, {}, float3(0.6f), float3(0.2f), {});
            auto strategy = MAKE_RC_OBJ_PTR(PassthroughDrawStrategy);
            strategy->SetData(items.data(), items.size());
            RenderView(commandList, view, view, framebuffers[0], *strategy, *c.pass, *context);
            for (const DrawItem& item : items)
            {
                strategy->SetData(&item, 1);
                RenderView(commandList, view, view, framebuffers[1], *strategy, *c.pass, *context);
            }
            for (uint32_t side = 0; side < 2; ++side)
                for (nvrhi::ITexture* texture : targets[side])
                    commandList->setTextureState(texture, nvrhi::AllSubresources, TargetState(texture->getDesc().format));
            commandList->commitBarriers();
            commandList->close();
            device->executeCommandList(commandList);
            device->waitForIdle();

            for (size_t t = 0; t < c.formats.size(); ++t)
            {
                if (api == nvrhi::GraphicsAPI::D3D11 && nvrhi::getFormatInfo(c.formats[t]).hasDepth)
                    continue;   // D3D11 reads a depth target back only whole (see RunGeometry); colour is compared
                const std::vector<uint8_t> one = ReadTexture(device, targets[0][t]);
                const std::vector<uint8_t> separate = ReadTexture(device, targets[1][t]);
                size_t differing = 0;
                const size_t texel = nvrhi::getFormatInfo(c.formats[t]).bytesPerBlock;
                for (size_t i = 0; i + texel <= one.size(); i += texel)
                    differing += std::memcmp(one.data() + i, separate.data() + i, texel) != 0 ? 1 : 0;
                check(differing == 0, where + ": target " + std::to_string(t) + ": " + std::to_string(differing) +
                                          " texels differ between one RenderView and one per draw");
                if (c.formats[t] != nvrhi::Format::RGBA16_FLOAT || c.pass == forwardPass.Get())
                    check(std::any_of(one.begin(), one.end(), [](uint8_t b) { return b != 0; }),
                          where + ": target " + std::to_string(t) + " has content");
            }
            std::printf("ok: %s: four draws of one material over four buffer groups equal the draws made one by one\n",
                        where.c_str());
        }
        check(g_Warnings.load() == warningsBefore, tag + ": no warning");

        commandList = nullptr;
        scene = nullptr;
        device->waitForIdle();
        device->runGarbageCollection();
    }

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

    // ------------------------------------------------------------------------------------------ geometry passes (E5.3)
    // One draw call of a geometry pass, as RenderView issues it.
    struct DrawRecord
    {
        uint32_t startIndex;
        uint32_t startVertex;
        uint32_t startInstance;
        uint32_t instanceCount;
    };

    struct RecordingContext : GBufferFillPass::Context
    {
        std::vector<std::vector<DrawRecord>> views;   // the draws of each use (child view), in recording order
    };

    // GBufferFillPass, recording its draw calls into each use's context.
    class RecordingGBufferPass : public GBufferFillPass
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        using GBufferFillPass::GBufferFillPass;

        void SetupView(GeometryPassContext& context, nvrhi::ICommandList* commandList, const IView* view,
                       const IView* viewPrev) override
        {
            static_cast<RecordingContext&>(context).views.emplace_back();
            GBufferFillPass::SetupView(context, commandList, view, viewPrev);
        }

        void SetPushConstants(GeometryPassContext& context, nvrhi::ICommandList* commandList, nvrhi::GraphicsState& state,
                              nvrhi::DrawArguments& args) override
        {
            static_cast<RecordingContext&>(context).views.back().push_back(
                DrawRecord{ args.startIndexLocation, args.startVertexLocation, args.startInstanceLocation,
                            args.instanceCount });
            GBufferFillPass::SetPushConstants(context, commandList, state, args);
        }
    };

    // (first index, instance) of every instance a list of draws covers, in draw order.
    std::vector<std::pair<uint32_t, uint32_t>> Expand(const std::vector<DrawRecord>& draws)
    {
        std::vector<std::pair<uint32_t, uint32_t>> drawn;
        for (const DrawRecord& draw : draws)
            for (uint32_t i = 0; i < draw.instanceCount; ++i)
                drawn.emplace_back(draw.startIndex, draw.startInstance + i);
        return drawn;
    }

    // Sub-plan item E5.3 on a device: the G-buffer drawn twice in one graph per frame, from the same scene on the graph
    // path, once through InstancedOpaqueDrawStrategy and TransparentDrawStrategy bound to the snapshot (SetPackets) and
    // once through the same kinds walking the graph, each pass with the frame's buffer reads (SceneFrame::bufferReads:
    // the instance buffer, the skinned vertex buffers, with bindless the geometry and material buffers). A
    // GBufferFillPass (input assembler or vertex pulling) draws the opaque packets into four colour targets and a depth
    // target, then the transparent ones into the same targets (GBufferFillPass takes every domain), for a composite
    // view of two children side by side (one use each; the bound strategy finds each child's packets). The scene: 180
    // quad instances of four meshes of 1-3 geometries, each mesh in a buffer group of its own (one material spans
    // groups), opaque, alpha-tested, double-sided, blended and double-sided blended materials, in runs of 45 so that one
    // mesh's instance slots are consecutive, at distinct depths, a third turned away (parallel planes: no depth ties,
    // so the images do not depend on the draw order; quads turned at random angles intersect, and a few texels on the
    // intersection lines then follow the draw order); nodes far outside the view with a child inside it (the walk
    // keeps them through their subgraph box only) and nodes with a child far outside; two skinned copies of one
    // prototype in front of every quad. Frames: 1 (load), 2 (a fifth of the quads moved: the upload pass writes), 3
    // (the characters posed: the skinning pass writes), 4 (idle: every buffer read in its resting state). Per frame: VerifyPlan and no diagnostics; after the execution the bound targets equal the walked ones
    // byte for byte and are not empty (but the emissive one, as no material is emissive; D3D11: the colour targets);
    // the bound opaque draws are no more than the walk's, and every instance they draw the walk draws too; the
    // transparent draws cover the walk's instances (less its subgraph-only ones) in the same order, one instance's
    // geometries in any order. Validation (debug runtime, GPU-based validation on D3D12, the Khronos layer, nvrhi's)
    // reports nothing but a known Khronos-layer performance warning (g_UnconsumedAttributeWarnings).
    void RunGeometry(app::DeviceManager* deviceManager, nvrhi::GraphicsAPI api, bool bindless, bool inputAssembler)
    {
        nvrhi::IDevice* device = deviceManager->GetDevice();
        const std::string tag = std::string(app::GetShaderTypeName(api)) + (bindless ? ", bindless" : "") +
                                (inputAssembler ? ", input assembler" : ", vertex pulling") + ", geometry passes";
        const uint32_t warningsBefore = g_Warnings.load();
        const uint32_t unconsumedBefore = g_UnconsumedAttributeWarnings.load();
        Fixture fixture(device, api, bindless);
        nvrhi::AutoPtr<Scene> scene = fixture.MakeScene(true);
        SceneGraph* graph = scene->GetSceneGraph();
        SceneGraphNode* root = graph->GetRootNode();

        auto material = [](const char* name, MaterialDomain domain, bool doubleSided, float3 color)
        {
            auto m = MAKE_RC_OBJ_PTR(Material);
            m->name = name;
            m->domain = domain;
            m->doubleSided = doubleSided;
            m->baseOrDiffuseColor = color;
            m->opacity = domain == MaterialDomain::AlphaBlended ? 0.5f : 1.f;
            return m;
        };
        nvrhi::AutoPtr<Material> red = material("red", MaterialDomain::Opaque, false, float3(1.f, 0.f, 0.f));
        nvrhi::AutoPtr<Material> green = material("green", MaterialDomain::AlphaTested, false, float3(0.f, 1.f, 0.f));
        nvrhi::AutoPtr<Material> blue = material("blue", MaterialDomain::Opaque, true, float3(0.f, 0.f, 1.f));
        nvrhi::AutoPtr<Material> yellow = material("yellow", MaterialDomain::AlphaBlended, false, float3(1.f, 1.f, 0.f));
        nvrhi::AutoPtr<Material> cyan = material("cyan", MaterialDomain::AlphaBlended, true, float3(0.f, 1.f, 1.f));
        nvrhi::AutoPtr<MeshInfo> meshes[4] = {
            MakeQuadMesh("quad", { red }, MAKE_RC_OBJ_PTR(BufferGroup)),
            MakeQuadMesh("pair", { green, blue }, MAKE_RC_OBJ_PTR(BufferGroup)),
            MakeQuadMesh("triple", { red, yellow, cyan }, MAKE_RC_OBJ_PTR(BufferGroup)),
            MakeQuadMesh("glass", { cyan }, MAKE_RC_OBJ_PTR(BufferGroup)),
        };

        std::vector<SceneGraphNode*> nodes;
        for (uint32_t i = 0; i < 180; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetName("quad" + std::to_string(i));
            node->SetTranslation(double3(-6.0 + 0.9 * double(i % 14), -3.8 + 0.85 * double((i / 14) % 9),
                                         4.0 + 0.31 * double((i * 37) % 61) + 0.001 * double(i)));
            if (i % 3 == 0)   // facing away: parallel planes at distinct depths, so no two surfaces tie in depth
                node->SetRotation(rotationQuat(double3(0.0, 1.0, 0.0), dm::PI_d));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, meshes[(i / 45) % 4].Get()));   // runs of consecutive slots
            nodes.push_back(graph->Attach(root, node));
            if (i % 23 == 5)
            {
                // a child far to the side: the node's subgraph box reaches it, its own box stays in the view
                auto child = MAKE_RC_OBJ_PTR(SceneGraphNode);
                child->SetName("far" + std::to_string(i));
                child->SetTranslation(double3(60.0, 0.0, 0.0));
                child->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, meshes[(i + 1) % 4].Get()));
                graph->Attach(nodes.back(), child);
            }
            if (i % 31 == 7)
            {
                // a node far outside whose child is in the view: the walk keeps it through its subgraph box
                auto outside = MAKE_RC_OBJ_PTR(SceneGraphNode);
                outside->SetName("outside" + std::to_string(i));
                outside->SetTranslation(double3(-40.0, 0.0, 10.0));
                outside->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, meshes[(i + 2) % 4].Get()));
                auto inside = MAKE_RC_OBJ_PTR(SceneGraphNode);
                inside->SetName("inside" + std::to_string(i));
                inside->SetTranslation(double3(40.0 + 0.3 * double(i % 5), 1.0, 2.0005 + 0.4 * double(i / 31)));
                inside->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, meshes[(i + 3) % 4].Get()));
                graph->Attach(root, outside);
                graph->Attach(outside, inside);
            }
        }
        auto skinMaterial = material("skin", MaterialDomain::Opaque, true, float3(0.8f, 0.5f, 0.3f));
        nvrhi::AutoPtr<MeshInfo> prototype = MakeSkinPrototype("p0", 300, 4, true, false, skinMaterial);
        auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        // two skinned copies of one prototype: their own skinned buffer groups, the prototype's material
        Character characters[2] = { MakeCharacter(graph, factory, prototype, 4, "A", 0.0),
                                    MakeCharacter(graph, factory, prototype, 4, "B", 0.0) };
        for (uint32_t c = 0; c < 2; ++c)
        {
            characters[c].id = c;
            characters[c].root->SetTranslation(double3(c ? 1.6 : -1.2, -1.5, c ? 3.0 : 2.5));   // in front of every quad
            graph->Attach(root, characters[c].root);
        }

        auto view = MAKE_RC_OBJ_PTR(CompositeView);
        nvrhi::AutoPtr<PlanarView> children[2];
        for (uint32_t c = 0; c < 2; ++c)
        {
            children[c] = MAKE_RC_OBJ_PTR(PlanarView);
            children[c]->SetViewport(nvrhi::Viewport(96.f * float(c), 96.f * float(c + 1), 0.f, 128.f, 0.f, 1.f));
            children[c]->SetMatrices(inverse(translation(float3(c ? 0.15f : -0.15f, 0.f, 0.f))),
                                     perspProjD3DStyleReverse(radians(70.f), 0.75f, 0.1f));
            children[c]->UpdateCache();
            view->AddView(children[c]);
        }
        const IView* views[] = { children[0].Get(), children[1].Get() };

        nvrhi::AutoPtr<CommonRenderPasses> commonPasses = MAKE_RC_OBJ_PTR(CommonRenderPasses, device, fixture.shaderFactory);
        auto gbuffer = MAKE_RC_OBJ_PTR(RecordingGBufferPass, device, commonPasses);
        GBufferFillPass::CreateParameters gbufferParams;
        gbufferParams.useInputAssembler = inputAssembler;
        gbuffer->Init(*fixture.shaderFactory, gbufferParams);
        auto boundOpaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        auto boundTransparent = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
        auto walkOpaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        auto walkTransparent = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);

        const nvrhi::Format formats[5] = { nvrhi::Format::SRGBA8_UNORM, nvrhi::Format::SRGBA8_UNORM,
                                           nvrhi::Format::RGBA16_SNORM, nvrhi::Format::RGBA16_FLOAT, nvrhi::Format::D32 };
        nvrhi::TextureHandle targets[2][5];
        for (uint32_t side = 0; side < 2; ++side)
            for (uint32_t t = 0; t < 5; ++t)
                targets[side][t] = MakeTarget(device, formats[t], side ? "WalkTarget" : "BoundTarget");

        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> renderGraph;
        check(CreateResourceRegistry(device, &registry) == nvrhi::FS_OK, "registry");
        check(CreateRenderGraph(registry, &renderGraph) == nvrhi::FS_OK, "render graph");
        BackendProfile profile;
        check(ComputeBackendProfile(device, &profile) == nvrhi::FS_OK, "profile");
        RenderServiceDesc serviceDesc;
        serviceDesc.scene = scene;
        serviceDesc.shaderFactory = fixture.shaderFactory;
        auto service = MAKE_RC_OBJ_PTR(RenderService, device, serviceDesc);
        scene->FinishedLoading(0);
        device->waitForIdle();

        uint32_t subgraphOnly = 0, mergedDraws = 0;
        for (uint32_t frame = 1; frame <= 4; ++frame)
        {
            const std::string where = tag + ", frame " + std::to_string(frame);
            if (frame == 2)
                for (size_t i = 0; i < nodes.size(); i += 5)
                    nodes[i]->SetTranslation(nodes[i]->GetTranslation() + double3(0.05, 0.02, 0.5003));
            if (frame == 3)
                for (Character& character : characters)
                    Pose(character, frame);
            scene->RefreshSceneGraph(frame);
            renderGraph->Reset();
            SceneFrame* sceneFrame = nullptr;
            ExtractDesc extractDesc;
            extractDesc.views = views;
            extractDesc.viewCount = 2;
            extractDesc.frameIndex = frame;
            check(service->Extract(renderGraph, nullptr, extractDesc, &sceneFrame) == nvrhi::FS_OK && sceneFrame,
                  where + ": Extract");
            check(service->AddUploadPasses(renderGraph, sceneFrame) == nvrhi::FS_OK, where + ": AddUploadPasses");
            check(service->AddSkinningPasses(renderGraph, sceneFrame) == nvrhi::FS_OK, where + ": AddSkinningPasses");
            check(sceneFrame->bufferReadCount >= (bindless ? 4u : 2u), where + ": buffer reads");
            const uint32_t passesBefore = renderGraph->GetPassCount();

            boundOpaque->SetPackets(sceneFrame->snapshot);
            boundTransparent->SetPackets(sceneFrame->snapshot);
            RecordingContext contexts[2][2];   // [bound, walk][opaque, transparent]
            for (uint32_t side = 0; side < 2; ++side)
            {
                GeometryGraphInputs inputs;
                inputs.name = side ? "GBufferWalk" : "GBufferBound";
                inputs.view = view;
                inputs.viewPrev = view;
                inputs.rootNode = side ? root : nullptr;
                inputs.colorTargetCount = 4;
                for (uint32_t t = 0; t < 4; ++t)
                    inputs.colorTargets[t] = renderGraph->ImportTexture(targets[side][t], TargetState(formats[t]),
                                                                        TargetState(formats[t]));
                inputs.depth = renderGraph->ImportTexture(targets[side][4], TargetState(formats[4]),
                                                          TargetState(formats[4]));
                inputs.clearTargets = true;
                inputs.bufferReads = sceneFrame->bufferReads;
                inputs.bufferReadCount = sceneFrame->bufferReadCount;
                inputs.drawStrategy = side ? static_cast<IDrawStrategy*>(walkOpaque.Get()) : boundOpaque.Get();
                inputs.passContext = &contexts[side][0];
                const GeometryGraphOutputs opaque = gbuffer->AddToGraph(renderGraph, inputs);
                for (uint32_t t = 0; t < 4; ++t)
                    inputs.colorTargets[t] = opaque.colorTargets[t];
                inputs.depth = opaque.depth;
                inputs.clearTargets = false;
                inputs.drawStrategy = side ? static_cast<IDrawStrategy*>(walkTransparent.Get()) : boundTransparent.Get();
                inputs.passContext = &contexts[side][1];
                gbuffer->AddToGraph(renderGraph, inputs);
            }
            check(renderGraph->GetPassCount() == passesBefore + 8, where + ": a use per child view of four geometry passes");

            nvrhi::AutoPtr<ICompiledRenderGraph> plan;
            const nvrhi::FRESULT compiled = renderGraph->Compile(profile, CompileOptions(), &plan);
            nvrhi::string diagnostics;
            renderGraph->GetDiagnostics(diagnostics);
            check(compiled == nvrhi::FS_OK, where + ": Compile: " + std::string(diagnostics.c_str()));
            check(diagnostics.empty(), where + ": diagnostics: " + std::string(diagnostics.c_str()));
            nvrhi::string report;
            check(VerifyPlan(plan, renderGraph, nullptr, &report) == nvrhi::FS_OK,
                  where + ": VerifyPlan: " + std::string(report.c_str()));
            check(renderGraph->Execute(plan, ExecuteDesc()) == nvrhi::FS_OK, where + ": Execute");
            service->EndFrame(true);
            device->waitForIdle();

            // the images
            // D3D11 copies a depth texture only whole (CopyResource), which nvrhi's staging copy does not do: its depth
            // target is not read back there (the colour targets are)
            const uint32_t readTargets = api == nvrhi::GraphicsAPI::D3D11 ? 4 : 5;
            std::vector<uint8_t> images[2][5];
            for (uint32_t side = 0; side < 2; ++side)
                for (uint32_t t = 0; t < readTargets; ++t)
                    images[side][t] = ReadTexture(device, targets[side][t]);
            for (uint32_t t = 0; t < readTargets; ++t)
            {
                const std::vector<uint8_t>& bound = images[0][t];
                const std::vector<uint8_t>& walked = images[1][t];
                if (bound != walked)
                {
                    // where, and the depths there (a tie: equal depths, different surfaces)
                    const size_t texel = nvrhi::getFormatInfo(formats[t]).bytesPerBlock;
                    size_t differing = 0;
                    for (size_t i = 0; i + texel <= bound.size(); i += texel)
                    {
                        if (std::memcmp(bound.data() + i, walked.data() + i, texel) == 0)
                            continue;
                        if (++differing <= 4)
                        {
                            float depths[2] = {};
                            if (!images[0][4].empty())
                            {
                                std::memcpy(&depths[0], images[0][4].data() + (i / texel) * 4, 4);
                                std::memcpy(&depths[1], images[1][4].data() + (i / texel) * 4, 4);
                            }
                            std::printf("%s: target %u texel (%zu, %zu): depth %.9g bound, %.9g walked; bytes",
                                        where.c_str(), t, (i / texel) % 192, (i / texel) / 192, depths[0], depths[1]);
                            for (size_t b = 0; b < texel; ++b)
                                std::printf(" %02x/%02x", bound[i + b], walked[i + b]);
                            std::printf("\n");
                        }
                    }
                    std::printf("%s: target %u: %zu of %zu texels differ\n", where.c_str(), t, differing,
                                bound.size() / texel);
                }
                check(bound == walked, where + ": target " + std::to_string(t) + " equal");
                if (t != 3)   // no material is emissive
                    check(std::any_of(bound.begin(), bound.end(), [](uint8_t b) { return b != 0; }),
                          where + ": target " + std::to_string(t) + " has content");
            }

            // the draws, per child view: opaque no more than the walk's, every instance drawn the walk draws too;
            // transparent in the walk's order, less its subgraph-only instances, one instance's geometries in any order
            size_t draws[2][2] = {};   // [bound, walk][opaque, transparent]
            for (uint32_t side = 0; side < 2; ++side)
                for (uint32_t pass = 0; pass < 2; ++pass)
                    check(contexts[side][pass].views.size() == 2, where + ": two uses of each geometry pass");
            for (uint32_t v = 0; v < 2; ++v)
            {
                const std::string inView = where + ", view " + std::to_string(v);
                const std::vector<DrawRecord>& boundOpaqueDraws = contexts[0][0].views[v];
                const std::vector<DrawRecord>& walkOpaqueDraws = contexts[1][0].views[v];
                check(!boundOpaqueDraws.empty() && boundOpaqueDraws.size() <= walkOpaqueDraws.size(),
                      inView + ": " + std::to_string(boundOpaqueDraws.size()) + " opaque draws bound, " +
                          std::to_string(walkOpaqueDraws.size()) + " walked");
                auto boundDrawn = Expand(boundOpaqueDraws);
                auto walkDrawn = Expand(walkOpaqueDraws);
                std::sort(boundDrawn.begin(), boundDrawn.end());
                std::sort(walkDrawn.begin(), walkDrawn.end());
                check(std::includes(walkDrawn.begin(), walkDrawn.end(), boundDrawn.begin(), boundDrawn.end()),
                      inView + ": the bound opaque instances are the walk's");
                subgraphOnly += uint32_t(walkDrawn.size() - boundDrawn.size());
                mergedDraws += uint32_t(walkOpaqueDraws.size() - boundOpaqueDraws.size());

                const auto boundOrder = Expand(contexts[0][1].views[v]);
                auto boundSet = boundOrder;
                std::sort(boundSet.begin(), boundSet.end());
                std::vector<std::pair<uint32_t, uint32_t>> walkOrder;
                for (const auto& drawn : Expand(contexts[1][1].views[v]))
                {
                    if (std::binary_search(boundSet.begin(), boundSet.end(), drawn))
                        walkOrder.push_back(drawn);
                    else
                        ++subgraphOnly;
                }
                check(walkOrder.size() == boundOrder.size() && !boundOrder.empty(), inView + ": the transparent instances");
                for (size_t i = 0; i < boundOrder.size();)
                {
                    size_t j = i;
                    while (j < boundOrder.size() && boundOrder[j].second == boundOrder[i].second)
                        ++j;
                    std::vector<std::pair<uint32_t, uint32_t>> a(boundOrder.begin() + i, boundOrder.begin() + j);
                    std::vector<std::pair<uint32_t, uint32_t>> b(walkOrder.begin() + i, walkOrder.begin() + j);
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    check(a == b, inView + ": the transparent order");
                    i = j;
                }
                for (uint32_t side = 0; side < 2; ++side)
                    for (uint32_t pass = 0; pass < 2; ++pass)
                        draws[side][pass] += contexts[side][pass].views[v].size();
            }
            std::printf("ok: %s: images equal; two views: opaque draws %zu bound, %zu walked; transparent draws %zu "
                        "bound, %zu walked\n", where.c_str(), draws[0][0], draws[1][0], draws[0][1], draws[1][1]);
            device->runGarbageCollection();
        }
        check(subgraphOnly > 0, tag + ": the walk drew instances through subgraph boxes only");
        check(mergedDraws > 0, tag + ": the packets merged draws the walk did not");
        check(g_Warnings.load() == warningsBefore, tag + ": no warning");
        const uint32_t unconsumed = g_UnconsumedAttributeWarnings.load() - unconsumedBefore;
        check(unconsumed == 0 || (api == nvrhi::GraphicsAPI::VULKAN && inputAssembler),
              tag + ": the unconsumed-attribute warning only from the input assembler on Vulkan");
        if (unconsumed)
            std::printf("ok: %s: %u known Khronos-layer warnings (an input attribute the vertex shader does not read)\n",
                        tag.c_str(), unconsumed);

        service = nullptr;
        renderGraph = nullptr;
        registry = nullptr;
        device->waitForIdle();
        device->runGarbageCollection();
    }

    // ---------------------------------------------------------------------- steady-state allocations (E5's target)

#ifdef _WIN32
    // The functions of this executable by address, from the linker's map beside it (the target links with /MAP): the
    // functions of the static libraries and the internal ones too, with or without debug information. Nothing is asked of
    // a symbol path or server. /OPT:NOICF: no function has the name of another one it was folded with.
    class FunctionMap
    {
        struct Function
        {
            std::string decorated;
            std::string object;
        };
        std::map<uint64_t, Function> m_Functions;   // by relative address
        std::map<void*, std::string> m_Names;
        uint64_t m_Base = 0;
        uint64_t m_Size = 0;

    public:
        FunctionMap()
        {
            char path[MAX_PATH] = {};
            GetModuleFileNameA(nullptr, path, MAX_PATH);
            m_Base = uint64_t(GetModuleHandleA(nullptr));
            const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m_Base);
            const auto* headers = reinterpret_cast<const IMAGE_NT_HEADERS*>(m_Base + dos->e_lfanew);
            m_Size = headers->OptionalHeader.SizeOfImage;
            const std::string mapPath = std::string(path, std::strrchr(path, '.')) + ".map";
            std::FILE* file = std::fopen(mapPath.c_str(), "r");
            check(file != nullptr, "the linker's map of this executable: " + mapPath);
            std::vector<char> line(64 * 1024), name(line.size()), object(line.size());
            unsigned timestamp = 0, section = 0, offset = 0;
            unsigned long long base = 0, address = 0;
            while (std::fgets(line.data(), int(line.size()), file))
            {
                std::sscanf(line.data(), " Timestamp is %x", &timestamp);
                std::sscanf(line.data(), " Preferred load address is %llx", &base);
                // section:offset, name, address, f (a function), i (an inline one), library:object
                if (std::sscanf(line.data(), " %x:%x %s %llx f i %s", &section, &offset, name.data(), &address,
                                object.data()) == 5 ||
                    std::sscanf(line.data(), " %x:%x %s %llx f %s", &section, &offset, name.data(), &address,
                                object.data()) == 5)
                    m_Functions.emplace(address - base, Function{ name.data(), object.data() });
            }
            std::fclose(file);
            check(timestamp == headers->FileHeader.TimeDateStamp && m_Functions.size() > 1000,
                  mapPath + " is the map of this executable");
        }

        // The qualified name of the function at a return address and, in brackets, its object file. Empty outside this
        // executable (the system's modules, nvrhi_core.dll).
        const std::string& NameOf(void* returnAddress)
        {
            auto [it, inserted] = m_Names.try_emplace(returnAddress);
            const uint64_t relative = uint64_t(returnAddress) - m_Base - 1;   // in the call
            auto function = m_Functions.upper_bound(relative);
            if (!inserted || relative >= m_Size || function == m_Functions.begin())
                return it->second;
            --function;
            char name[4096] = {};
            if (!UnDecorateSymbolName(function->second.decorated.c_str(), name, sizeof(name), UNDNAME_NAME_ONLY))
                std::snprintf(name, sizeof(name), "%s", function->second.decorated.c_str());
            return it->second = std::string(name) + " [" + function->second.object + "]";
        }
    };
#endif

    bool StartsWith(const std::string& text, const char* prefix) { return text.rfind(prefix, 0) == 0; }

    // nvrhi's own code: its backends, and the state tracker in their command lists.
    bool IsNvrhi(const std::string& name)
    {
        return StartsWith(name, "nvrhi::d3d11::") || StartsWith(name, "nvrhi::d3d12::") ||
               StartsWith(name, "nvrhi::vulkan::") || StartsWith(name, "nvrhi::CommandListResourceStateTracker::");
    }

    // Functions that allocate for their callers: the rest of nvrhi (containers, descriptors, RC objects), the standard
    // library, lambdas (their names have no scope), and the counter.
    bool IsTransparent(const std::string& name)
    {
        return name.empty() || (StartsWith(name, "nvrhi::") && !IsNvrhi(name)) || StartsWith(name, "std::") ||
               StartsWith(name, "operator ") || StartsWith(name, "<lambda_") || StartsWith(name, "allocation_counter::");
    }

    // One scope's allocations: how many of each kind, and how many of each site. A site is the innermost function that
    // is not transparent, and its innermost caller outside nvrhi.
    struct Allocations
    {
        uint32_t counts[allocation_counter::KindCount] = {};
        std::map<std::string, uint32_t> sites;
        std::string first;       // the first one's site and stack
        std::string foreign;     // those of the first one not made by nvrhi, or made to create an object
        uint32_t growths = 0;    // of vulkan's TrackedCommandBuffer::referencedResources, not among the sites

        uint32_t Total() const { return counts[0] + counts[1]; }
    };

    // The allocations of the scope that closed last.
    Allocations TakeAllocations()
    {
        using namespace allocation_counter;
        Allocations result;
        for (uint32_t k = 0; k < KindCount; ++k)
            result.counts[k] = g_Counts[k];
        check(g_RecordCount <= std::size(g_Records), "room for every allocation's stack");
#ifdef _WIN32
        static FunctionMap functions;
        for (uint32_t r = 0; r < g_RecordCount; ++r)
        {
            const Record& record = g_Records[r];
            std::string owner, caller, stack;
            bool creation = false, growth = false;
            for (uint32_t i = 0; i < record.depth; ++i)
            {
                const std::string& name = functions.NameOf(record.stack[i]);
                stack += name.empty() ? "" : "\n        " + name;
                // Device::create* ends in a tail call of utils::ReturnObject
                creation = creation || StartsWith(name, "nvrhi::utils::ReturnObject") ||
                           (IsNvrhi(name) && name.find("::Device::create") != std::string::npos);
                growth = growth || (owner.empty() && StartsWith(name, "std::vector<nvrhi::AutoPtr<nvrhi::IRHIObject>"));
                if (owner.empty() && !IsTransparent(name))
                    owner = name;
                if (caller.empty() && !IsTransparent(name) && !IsNvrhi(name))
                    caller = name;
            }
            const std::string site = std::string(c_KindNames[record.kind]) + ": " + owner + " from " + caller;
            if (r == 0)
                result.first = site + stack;
            if (result.foreign.empty() && (creation || !IsNvrhi(owner)))
                result.foreign = site + stack;
            if (growth && StartsWith(owner, "nvrhi::vulkan::CommandList::"))
                ++result.growths;
            else
                ++result.sites[site];
        }
#endif
        return result;
    }

    // Runs `body` counted.
    template <typename F>
    Allocations CountAllocations(F&& body)
    {
        {
            allocation_counter::Scope scope;
            body();
        }
        return TakeAllocations();
    }

    // The counters see each form of both allocators, exactly once: without them every count would be 0.
    void CheckCounters(const std::string& tag)
    {
        void* volatile memory = nullptr;
        nvrhi::IMemoryAllocator* allocator = nvrhi::GetDefaultMemAllocator();
        auto counted = [&tag](const char* what, uint32_t news, uint32_t allocations, auto&& body)
        {
            {
                allocation_counter::Scope scope;
                body();
            }
            check(allocation_counter::g_Counts[allocation_counter::OperatorNew] == news &&
                  allocation_counter::g_Counts[allocation_counter::NvrhiAllocator] == allocations,
                  tag + ": positive control, " + what);
        };
        counted("operator new", 1, 0, [&] { memory = ::operator new(16); });
        ::operator delete(memory);
        counted("nothrow array new", 1, 0, [&] { memory = ::operator new[](16, std::nothrow); });
        ::operator delete[](memory);
        counted("aligned new", 1, 0, [&] { memory = ::operator new(16, std::align_val_t(64)); });
        ::operator delete(memory, std::align_val_t(64));
        counted("nvrhi's default allocator", 0, 1, [&] { memory = allocator->Allocate(16); });
        allocator->Free(memory);
        counted("nvrhi's default allocator, aligned", 0, 1, [&] { memory = allocator->AllocateAligned(16, 64); });
        allocator->FreeAligned(memory);
    }

    // GBufferFillPass, counting its draws and, while `capture` is set, keeping them by use (the pass holds the pipelines
    // and binding sets their states name).
    class CapturingGBufferPass : public GBufferFillPass
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        using GBufferFillPass::GBufferFillPass;
        using Draws = std::vector<std::pair<nvrhi::GraphicsState, nvrhi::DrawArguments>>;
        std::vector<Draws>* capture = nullptr;
        uint32_t draws = 0;

        nvrhi::IBuffer* GetConstantBuffer() const noexcept { return m_GBufferCB; }

        void SetupView(GeometryPassContext& context, nvrhi::ICommandList* commandList, const IView* view,
                       const IView* viewPrev) override
        {
            GBufferFillPass::SetupView(context, commandList, view, viewPrev);
            if (capture)
                capture->emplace_back();
        }

        void SetPushConstants(GeometryPassContext& context, nvrhi::ICommandList* commandList, nvrhi::GraphicsState& state,
                              nvrhi::DrawArguments& args) override
        {
            GBufferFillPass::SetPushConstants(context, commandList, state, args);
            ++draws;
            if (capture)
                capture->back().emplace_back(state, args);
        }
    };

    // The steady-state allocations of step E5's device part (the sub-plan's target: 0 per frame in extraction, upload,
    // skinning and recording; device-free: test_render_extract, test_gpu_scene). The scene: 180 quad instances of which
    // a fifth moves every frame (36 transform records, 36 settle slots), and two skinned characters posed in every
    // frame (the scripts' period is 10 frames); two views, each drawn by a GBufferFillPass of its own through
    // strategies bound to the packets, opaque then transparent. 32 frames to warm up, then 32 counted: operator new of
    // this module (donut, the render graph, nvrhi and its backends) and nvrhi's default allocator (RC objects, nvrhi
    // containers, the ECS storage). The device has the debug runtime but not the nvrhi validation layer, which allocates.
    // Serial recording, and (D3D12, Vulkan) parallel recording, one segment per pass on 4 threads. Every counted frame
    // has the same records, skinned instances, draws, segments and passes, and a cached plan.
    //
    // - The scene's refresh, Extract, AddUploadPasses, AddSkinningPasses, the geometry passes' AddToGraph, Reset and
    //   Compile, and EndFrame allocate nothing in any frame; no ECS storage block and no graph object is made.
    // - Execute records through nvrhi, whose backends allocate (state-tracking nodes, upload and descriptor
    //   bookkeeping, command-list instances, submissions). Windows: every allocation is attributed by its stack, and one
    //   that nvrhi's backends did not make (a pass body, donut, the graph, also through nvrhi's inline helpers), or that
    //   they made to create an object, fails the case. The linker's map must be there.
    // - All platforms: it allocates exactly as much as the same nvrhi calls recorded by hand, without donut or the graph
    //   (the plan's lists, markers, barriers, tracked states, clears, writes, dispatches and draws, the draws with the
    //   graphics states the geometry passes set), in the median; each loop's frames all allocate the same at the same
    //   sites. The exception is Vulkan with parallel recording: a queue's command buffers go first in, first out to the
    //   lists in the order the threads open them, so that a buffer can meet a list with more pipelines, framebuffers and
    //   binding sets than any it recorded before, and the growth of its vector of them (TrackedCommandBuffer::
    //   referencedResources, pushed to at vulkan-graphics.cpp:554) falls into a counted frame: those growths only, in
    //   fewer than half of the frames.
    //
    // Not covered: the shadow passes, lights, the Forward, Depth and Deferred passes, the input assembler, growth of the
    // scene's buffers, dropped frames, parallel extraction (the scene is too small), and nvrhi's garbage collection,
    // which runs every fourth frame outside the counted code. E5_ALLOC_TRACE=1 prints the sites of Execute.
    void RunSteadyState(app::DeviceManager* deviceManager, nvrhi::GraphicsAPI api, bool parallel)
    {
        nvrhi::IDevice* device = deviceManager->GetDevice();
        const bool bindless = api != nvrhi::GraphicsAPI::D3D11;
        const std::string tag = std::string(app::GetShaderTypeName(api)) + ", steady state, " +
                                (parallel ? "parallel" : "serial") + " recording";
        const uint32_t warningsBefore = g_Warnings.load();
        Fixture fixture(device, api, bindless);
        nvrhi::AutoPtr<Scene> scene = fixture.MakeScene(true);
        SceneGraph* graph = scene->GetSceneGraph();
        SceneGraphNode* root = graph->GetRootNode();

        auto material = [](const char* name, MaterialDomain domain, bool doubleSided)
        {
            auto m = MAKE_RC_OBJ_PTR(Material);
            m->name = name;
            m->domain = domain;
            m->doubleSided = doubleSided;
            return m;
        };
        nvrhi::AutoPtr<Material> red = material("red", MaterialDomain::Opaque, false);
        nvrhi::AutoPtr<Material> green = material("green", MaterialDomain::AlphaTested, false);
        nvrhi::AutoPtr<Material> cyan = material("cyan", MaterialDomain::AlphaBlended, true);
        nvrhi::AutoPtr<MeshInfo> meshes[3] = {
            MakeQuadMesh("quad", { red }, MAKE_RC_OBJ_PTR(BufferGroup)),
            MakeQuadMesh("pair", { green, red }, MAKE_RC_OBJ_PTR(BufferGroup)),
            MakeQuadMesh("glass", { cyan, green }, MAKE_RC_OBJ_PTR(BufferGroup)),
        };
        std::vector<SceneGraphNode*> nodes;
        std::vector<double3> positions;
        for (uint32_t i = 0; i < 180; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            positions.push_back(double3(-6.0 + 0.9 * double(i % 14), -3.8 + 0.85 * double((i / 14) % 9),
                                        4.0 + 0.31 * double((i * 37) % 61)));
            node->SetTranslation(positions.back());
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, meshes[(i / 60) % 3].Get()));
            nodes.push_back(graph->Attach(root, node));
        }
        nvrhi::AutoPtr<Material> skinMaterial = material("skin", MaterialDomain::Opaque, true);
        nvrhi::AutoPtr<MeshInfo> prototype = MakeSkinPrototype("p0", 300, 4, true, false, skinMaterial);
        auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        Character characters[2] = { MakeCharacter(graph, factory, prototype, 4, "A", 0.0),
                                    MakeCharacter(graph, factory, prototype, 4, "B", 0.0) };
        for (uint32_t c = 0; c < 2; ++c)
        {
            characters[c].id = c;
            characters[c].root->SetTranslation(double3(c ? 1.6 : -1.2, -1.5, 2.5));
            graph->Attach(root, characters[c].root);
        }

        // two views side by side, each drawn by a pass object of its own (the unit of parallel recording), opaque and
        // transparent packets
        constexpr uint32_t viewCount = 2;
        nvrhi::AutoPtr<PlanarView> viewObjects[viewCount];
        const IView* views[viewCount];
        auto commonPasses = MAKE_RC_OBJ_PTR(CommonRenderPasses, device, fixture.shaderFactory);
        nvrhi::AutoPtr<CapturingGBufferPass> gbuffers[viewCount];
        nvrhi::AutoPtr<InstancedOpaqueDrawStrategy> opaque[viewCount];
        nvrhi::AutoPtr<TransparentDrawStrategy> transparent[viewCount];
        const nvrhi::Format formats[5] = { nvrhi::Format::SRGBA8_UNORM, nvrhi::Format::SRGBA8_UNORM,
                                           nvrhi::Format::RGBA16_SNORM, nvrhi::Format::RGBA16_FLOAT, nvrhi::Format::D32 };
        nvrhi::TextureHandle targets[viewCount][5];
        for (uint32_t v = 0; v < viewCount; ++v)
        {
            viewObjects[v] = MAKE_RC_OBJ_PTR(PlanarView);
            viewObjects[v]->SetViewport(nvrhi::Viewport(192.f, 128.f));
            viewObjects[v]->SetMatrices(inverse(translation(float3(v ? 0.15f : -0.15f, 0.f, 0.f))),
                                        perspProjD3DStyleReverse(radians(70.f), 1.5f, 0.1f));
            viewObjects[v]->UpdateCache();
            views[v] = viewObjects[v];
            gbuffers[v] = MAKE_RC_OBJ_PTR(CapturingGBufferPass, device, commonPasses);
            GBufferFillPass::CreateParameters gbufferParams;
            gbufferParams.numConstantBufferVersions = 64;
            gbuffers[v]->Init(*fixture.shaderFactory, gbufferParams);
            opaque[v] = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
            transparent[v] = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
            for (uint32_t t = 0; t < 5; ++t)
                targets[v][t] = MakeTarget(device, formats[t], "SteadyStateTarget");
        }

        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> renderGraph;
        check(CreateResourceRegistry(device, &registry) == nvrhi::FS_OK, "registry");
        check(CreateRenderGraph(registry, &renderGraph) == nvrhi::FS_OK, "render graph");
        BackendProfile profile;
        check(ComputeBackendProfile(device, &profile) == nvrhi::FS_OK, "profile");
        for (uint8_t& lanes : profile.queueLanes)
            lanes = 1;
        RenderServiceDesc serviceDesc;
        serviceDesc.scene = scene;
        serviceDesc.shaderFactory = fixture.shaderFactory;
        auto service = MAKE_RC_OBJ_PTR(RenderService, device, serviceDesc);
        scene->FinishedLoading(0);
        device->waitForIdle();
        RenderGraphWorkerPool pool(parallel ? 3 : 0);
        CompileOptions options;
        options.recordGranularity = parallel ? 1 : 0;
        ExecuteDesc executeDesc;
        if (parallel)
        {
            executeDesc.parallelFor = &RenderGraphWorkerPool::ParallelFor;
            executeDesc.schedulerContext = &pool;
            executeDesc.workerCount = pool.ThreadCount();
        }

        const uint32_t warmup = 32, measured = 32, passCount = 7;
        const uint32_t segmentCount = parallel ? passCount : 1;
        const bool planBarriers = profile.barrierModel != BarrierModel::Implicit;
        const uint32_t trackedPerPass = 8 + (bindless ? 2 : 0);   // 5 targets, the instance and 2 skinned buffers
        GBufferFillPass::Context contexts[viewCount][2];
        std::vector<CapturingGBufferPass::Draws> captured[viewCount];
        std::vector<Allocations> graphFrames, handFrames;
        GpuSceneScatterRecords records;
        RegistryStatistics registryBefore{}, registryAfter{};
        uint64_t storageBlocks = 0;
        uint32_t draws = 0;

        allocation_counter::AllocatorHook hook;
        CheckCounters(tag);
        for (uint32_t i = 0; i < warmup + measured; ++i)
        {
            const uint32_t frame = i + 1;
            const std::string where = tag + ", frame " + std::to_string(frame);
            const bool counted = i >= warmup;
            if (i == warmup)
            {
                registry->GetRegistryStatistics(registryBefore);
                storageBlocks = ecs::GetHeapAllocationCount();
            }
            for (uint32_t v = 0; v < viewCount; ++v)
            {
                gbuffers[v]->capture = i + 1 == warmup ? &captured[v] : nullptr;
                gbuffers[v]->draws = 0;
            }
            // a phase that allocates nothing in a counted frame
            auto phase = [&](const char* name, auto&& body)
            {
                if (!counted)
                    return body();
                const Allocations allocations = CountAllocations(body);
                check(allocations.Total() == 0, where + ": " + name + " makes " + std::to_string(allocations.Total()) +
                                                    " allocations; " + allocations.first);
            };
            SceneFrame* sceneFrame = nullptr;
            nvrhi::AutoPtr<ICompiledRenderGraph> plan;
            nvrhi::FRESULT extracted = nvrhi::FE_GENERIC_ERROR, uploads = extracted, skinning = extracted;
            nvrhi::FRESULT compiled = extracted, executed = extracted;
            phase("refresh", [&]
            {
                for (size_t n = frame % 5; n < nodes.size(); n += 5)
                    nodes[n]->SetTranslation(positions[n] + double3(0.0, 0.0, 0.05 * double((frame / 5) % 2)));
                for (Character& character : characters)
                    Pose(character, frame % 10);
                scene->RefreshSceneGraph(frame);
            });
            ExtractDesc extractDesc;
            extractDesc.views = views;
            extractDesc.viewCount = viewCount;
            extractDesc.frameIndex = frame;
            phase("Reset", [&] { renderGraph->Reset(); });
            phase("Extract", [&] { extracted = service->Extract(renderGraph, nullptr, extractDesc, &sceneFrame); });
            check(extracted == nvrhi::FS_OK && sceneFrame, where + ": Extract");
            phase("AddUploadPasses", [&] { uploads = service->AddUploadPasses(renderGraph, sceneFrame); });
            phase("AddSkinningPasses", [&] { skinning = service->AddSkinningPasses(renderGraph, sceneFrame); });
            check(uploads == nvrhi::FS_OK && skinning == nvrhi::FS_OK, where + ": AddUploadPasses, AddSkinningPasses");
            phase("the geometry passes' AddToGraph", [&]
            {
                for (uint32_t v = 0; v < viewCount; ++v)
                {
                    opaque[v]->SetPackets(sceneFrame->snapshot);
                    transparent[v]->SetPackets(sceneFrame->snapshot);
                    GeometryGraphInputs inputs;
                    inputs.name = "GBuffer";
                    inputs.view = viewObjects[v];
                    inputs.viewPrev = viewObjects[v];
                    inputs.colorTargetCount = 4;
                    for (uint32_t t = 0; t < 4; ++t)
                        inputs.colorTargets[t] = renderGraph->ImportTexture(targets[v][t], TargetState(formats[t]),
                                                                            TargetState(formats[t]));
                    inputs.depth = renderGraph->ImportTexture(targets[v][4], TargetState(formats[4]),
                                                              TargetState(formats[4]));
                    inputs.clearTargets = true;
                    inputs.bufferReads = sceneFrame->bufferReads;
                    inputs.bufferReadCount = sceneFrame->bufferReadCount;
                    inputs.drawStrategy = opaque[v];
                    inputs.passContext = &contexts[v][0];
                    const GeometryGraphOutputs outputs = gbuffers[v]->AddToGraph(renderGraph, inputs);
                    for (uint32_t t = 0; t < 4; ++t)
                        inputs.colorTargets[t] = outputs.colorTargets[t];
                    inputs.depth = outputs.depth;
                    inputs.clearTargets = false;
                    inputs.drawStrategy = transparent[v];
                    inputs.passContext = &contexts[v][1];
                    gbuffers[v]->AddToGraph(renderGraph, inputs);
                }
            });
            scene->GetGpuScene()->GetScatterRecords(records);
            phase("Compile", [&] { compiled = renderGraph->Compile(profile, options, &plan); });
            check(compiled == nvrhi::FS_OK, where + ": Compile");
            auto execute = [&] { executed = renderGraph->Execute(plan, executeDesc); };
            if (counted)
                graphFrames.push_back(CountAllocations(execute));
            else
                execute();
            check(executed == nvrhi::FS_OK, where + ": Execute");
            phase("EndFrame", [&] { service->EndFrame(true); });

            // the same work in every counted frame
            RunStatistics run{};
            registry->GetRunStatistics(run);
            PlanStatistics statistics{};
            plan->GetStatistics(statistics);
            if (i + 1 == warmup)
                draws = gbuffers[0]->draws;
            if (counted)
            {
                check(records.rowCount == 0 && records.transformCount == 36 && records.settleCount == 36 &&
                      sceneFrame->snapshot->skinnedInstanceCount == 2, where + ": the moved and posed instances");
                check(draws != 0 && gbuffers[0]->draws == draws && gbuffers[1]->draws == draws, where + ": the draws");
                check(statistics.cacheHit && !run.planSwitched && run.passes == passCount &&
                      run.segments == segmentCount && run.recordTasks == (parallel ? 4u : 1u) && run.submissions == 1 &&
                      run.stateCalls == (planBarriers ? 4 * 2 * trackedPerPass : 0),
                      where + ": the plan and its recording");
                check(graphFrames.back().foreign.empty(), where + ": Execute allocates outside nvrhi, or an object; " +
                                                              graphFrames.back().foreign);
            }
            if (i % 4 == 3)
            {
                device->waitForIdle();
                device->runGarbageCollection();
            }
        }
        registry->GetRegistryStatistics(registryAfter);
        check(registryAfter.objectsCreated == registryBefore.objectsCreated, tag + ": no graph object is made");
        check(ecs::GetHeapAllocationCount() == storageBlocks, tag + ": no ECS storage block is made");
        for (uint32_t v = 0; v < viewCount; ++v)
            check(captured[v].size() == 2 && !captured[v][0].empty() && !captured[v][1].empty(),
                  tag + ": captured draws");

        // the baseline: the executor's lists (the upload list, then a list per segment) and their contents
        const bool vulkan = api == nvrhi::GraphicsAPI::VULKAN;
        std::vector<nvrhi::CommandListHandle> lists(1 + segmentCount);
        std::vector<nvrhi::ICommandList*> segments;
        for (nvrhi::CommandListHandle& list : lists)
        {
            device->createCommandList(nvrhi::CommandListParameters().setEnableImmediateExecution(!planBarriers), &list);
            check(list != nullptr, tag + ": a command list");
            if (&list != &lists[0])
                segments.push_back(list);
        }
        nvrhi::FenceHandle fence;
        const bool fences = device->createFence(nvrhi::FenceDesc(), &fence) == nvrhi::FS_OK;
        uint64_t fenceValue = 0;
        nvrhi::BufferDesc stagingDesc;
        stagingDesc.byteSize = 64 * 1024;
        stagingDesc.canHaveRawViews = true;
        stagingDesc.debugName = "BaselineRecords";
        nvrhi::BufferHandle staging;
        device->createBuffer(stagingDesc, &staging);
        GpuScene* gpuScene = scene->GetGpuScene();
        nvrhi::IBuffer* instanceBuffer = gpuScene->GetInstanceBuffer();
        nvrhi::IBuffer* skinnedBuffers[2];
        SkinningConstants skinningConstants[2];
        GpuSceneScatterConstants scatterConstants = {};   // no records: the dispatch writes nothing
        scatterConstants.instanceRowStride = uint32_t(sizeof(InstanceData));
        scatterConstants.groupsX = 1;
        nvrhi::ShaderHandle shaders[2];
        nvrhi::BindingLayoutHandle layouts[2];
        nvrhi::ComputePipelineHandle pipelines[2];   // scatter, skinning
        for (uint32_t k = 0; k < 2; ++k)
        {
            shaders[k] = fixture.shaderFactory->CreateShader(k ? "donut/skinning_cs" : "donut/gpu_scene_scatter_cs",
                                                             "main", nullptr, nvrhi::ShaderType::Compute);
            auto layoutDesc = nvrhi::BindingLayoutDesc().setVisibility(nvrhi::ShaderType::Compute)
                .addItem(nvrhi::BindingLayoutItem::PushConstants(0, k ? sizeof(SkinningConstants)
                                                                      : sizeof(GpuSceneScatterConstants)))
                .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(0)).addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(0));
            if (k)
                layoutDesc.addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(1));
            device->createBindingLayout(layoutDesc, &layouts[k]);
            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.CS = shaders[k];
            pipelineDesc.bindingLayouts = { layouts[k] };
            device->createComputePipeline(pipelineDesc, &pipelines[k]);
            check(pipelines[k] != nullptr, tag + ": the baseline's pipelines");
        }
        nvrhi::BindingSetHandle scatterSet, skinningSets[2];
        device->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(GpuSceneScatterConstants)))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, staging))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(0, instanceBuffer)), layouts[0], &scatterSet);
        for (uint32_t k = 0; k < 2; ++k)
        {
            const SkinnedMeshInstance* leaf = characters[k].skinned;
            skinnedBuffers[k] = leaf->GetMesh()->buffers->vertexBuffer;
            device->createBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(SkinningConstants)))
                .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, leaf->GetPrototypeMesh()->buffers->vertexBuffer))
                .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(1, leaf->jointBuffer))
                .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(0, skinnedBuffers[k])), layouts[1], &skinningSets[k]);
            FillSkinningConstants(*leaf->GetPrototypeMesh(), *leaf->GetMesh(), false, skinningConstants[k]);
        }
        const std::vector<uint8_t> bytes(64 * 1024);
        const GBufferFillConstants viewConstants = {};
        const GBufferPushConstants drawConstants = {};

        using nvrhi::ResourceStates;
        using Barriers = std::vector<nvrhi::ResourceBarrier>;
        auto to = [](nvrhi::IResource* resource, ResourceStates before, ResourceStates after)
        {
            return nvrhi::ResourceBarrier::Transition(resource, before, after);
        };
        const ResourceStates drawState = ResourceStates::VertexBuffer | ResourceStates::ShaderResource;
        const ResourceStates uav = ResourceStates::UnorderedAccess, copyDest = ResourceStates::CopyDest;
        const Barriers beforeRecords = { to(staging, ResourceStates::Unknown, copyDest) };
        const Barriers beforeScatter = { to(staging, copyDest, ResourceStates::ShaderResource),
                                         to(instanceBuffer, c_GpuSceneRestingState, uav) };
        const Barriers beforeSkinning = { to(skinnedBuffers[0], c_SkinnedVertexBufferRestingState, uav),
                                          to(skinnedBuffers[1], c_SkinnedVertexBufferRestingState, uav) };
        const Barriers beforeGeometry = { to(instanceBuffer, uav, drawState), to(skinnedBuffers[0], uav, drawState),
                                          to(skinnedBuffers[1], uav, drawState) };
        const Barriers afterGeometry = { to(instanceBuffer, drawState, c_GpuSceneRestingState) };
        Barriers toClear[viewCount], toDraw[viewCount], drawToDraw[viewCount];   // Vulkan
        for (uint32_t v = 0; v < viewCount; ++v)
        {
            for (uint32_t t = 0; t < 5; ++t)
            {
                toClear[v].push_back(to(targets[v][t], TargetState(formats[t]), copyDest));
                toDraw[v].push_back(to(targets[v][t], copyDest, TargetState(formats[t])));
                drawToDraw[v].push_back(to(targets[v][t], TargetState(formats[t]), TargetState(formats[t])));
            }
        }
        std::vector<std::pair<nvrhi::IBuffer*, ResourceStates>> trackedBuffers = {
            { instanceBuffer, drawState }, { skinnedBuffers[0], drawState }, { skinnedBuffers[1], drawState } };
        if (bindless)
        {
            trackedBuffers.emplace_back(gpuScene->GetGeometryBuffer(), ResourceStates::ShaderResource);
            trackedBuffers.emplace_back(gpuScene->GetMaterialBuffer(), ResourceStates::ShaderResource);
        }
        auto barriers = [&](nvrhi::ICommandList* list, const Barriers& batch)
        {
            if (planBarriers)
                list->barriers(batch.data(), uint32_t(batch.size()));
        };
        auto nativePass = [&](nvrhi::ICommandList* list, const char* name, const Barriers& batch, auto&& body)
        {
            list->beginMarker(name);
            barriers(list, batch);
            if (planBarriers)
                list->setExplicitBarrierScope(name);
            body();
            if (planBarriers)
                list->setExplicitBarrierScope(nullptr);
            list->endMarker();
        };
        auto dispatch = [&](nvrhi::ICommandList* list, uint32_t kernel, nvrhi::IBindingSet* set, const void* constants,
                            size_t size, uint32_t groups)
        {
            nvrhi::ComputeState state;
            state.pipeline = pipelines[kernel];
            state.bindings = { set };
            list->setComputeState(state);
            list->setPushConstants(constants, size);
            list->dispatch(groups);
        };
        auto geometryPass = [&](nvrhi::ICommandList* list, uint32_t v, uint32_t use)
        {
            list->beginMarker("GBuffer");
            if (v == 0 && use == 0)
                barriers(list, beforeGeometry);
            if (use == 0)
            {
                if (vulkan)
                    barriers(list, toClear[v]);
                for (uint32_t t = 0; t < 4; ++t)
                    list->clearTextureFloat(targets[v][t], nvrhi::AllSubresources, nvrhi::Color(0.f));
                list->clearDepthStencilTexture(targets[v][4], nvrhi::AllSubresources, true, 0.f, false, 0);
            }
            if (vulkan)
                barriers(list, use ? drawToDraw[v] : toDraw[v]);
            if (planBarriers)
            {
                for (uint32_t t = 0; t < 5; ++t)
                    list->beginTrackingTextureState(targets[v][t], nvrhi::AllSubresources, TargetState(formats[t]));
                for (const auto& [buffer, state] : trackedBuffers)
                    list->beginTrackingBufferState(buffer, state);
                list->setEnableAutomaticBarriers(true);
            }
            list->writeBuffer(gbuffers[v]->GetConstantBuffer(), &viewConstants, sizeof(viewConstants));
            const nvrhi::GraphicsState* last = nullptr;
            for (const auto& [state, arguments] : captured[v][use])
            {
                // RenderView sets the state at a change of material, cull mode or buffers
                if (!last || last->pipeline != state.pipeline || last->indexBuffer.buffer != state.indexBuffer.buffer ||
                    !std::equal(last->bindings.begin(), last->bindings.end(), state.bindings.begin(),
                                state.bindings.end()))
                    list->setGraphicsState(state);
                last = &state;
                list->setPushConstants(&drawConstants, sizeof(drawConstants));
                list->drawIndexed(arguments);
            }
            if (planBarriers)
            {
                for (uint32_t t = 0; t < 5; ++t)
                    list->setTextureState(targets[v][t], nvrhi::AllSubresources, TargetState(formats[t]));
                for (const auto& [buffer, state] : trackedBuffers)
                    list->setBufferState(buffer, state);
                list->commitBarriers();
                list->setEnableAutomaticBarriers(false);
            }
            if (v + 1 == viewCount && use == 1)
                barriers(list, afterGeometry);
            list->endMarker();
        };
        auto recordByHand = [&]
        {
            lists[0]->open();
            for (const Character& character : characters)
                lists[0]->writeBuffer(character.skinned->jointBuffer, bytes.data(),
                                      size_t(character.skinned->jointBuffer->getDesc().byteSize));
            lists[0]->close();
            for (uint32_t pass = 0; pass < passCount; ++pass)
            {
                nvrhi::ICommandList* list = segments[parallel ? pass : 0];
                if (parallel || pass == 0)
                {
                    list->open();
                    if (planBarriers)
                        list->setEnableAutomaticBarriers(false);
                }
                if (pass == 0)
                    nativePass(list, "GpuSceneRecords", beforeRecords, [&]
                    {
                        const size_t transforms = records.transformCount * sizeof(GpuSceneTransformRecord);
                        list->writeBuffer(staging, bytes.data(), transforms);
                        list->writeBuffer(staging, bytes.data(), records.settleCount * sizeof(uint32_t), transforms);
                    });
                else if (pass == 1)
                    nativePass(list, "GpuSceneScatter", beforeScatter, [&]
                    {
                        dispatch(list, 0, scatterSet, &scatterConstants, sizeof(scatterConstants), 2);
                    });
                else if (pass == 2)
                    nativePass(list, "Skinning", beforeSkinning, [&]
                    {
                        for (uint32_t k = 0; k < 2; ++k)
                        {
                            list->beginMarker(characters[k].name.c_str());
                            dispatch(list, 1, skinningSets[k], &skinningConstants[k], sizeof(skinningConstants[k]),
                                     dm::div_ceil(skinningConstants[k].numVertices, 256u));
                            list->endMarker();
                        }
                    });
                else
                    geometryPass(list, (pass - 3) / 2, (pass - 3) % 2);
                if (parallel || pass + 1 == passCount)
                    list->close();
            }
            nvrhi::ICommandList* const upload[] = { lists[0] };
            if (!fences)
            {
                device->executeCommandList(lists[0]);
                device->executeCommandList(segments[0]);
                return;
            }
            const nvrhi::FenceValue first{ fence, ++fenceValue }, second{ fence, ++fenceValue };
            device->submit(nvrhi::SubmitDesc().setCommandLists(upload, 1).setSignals(&first, 1));
            device->submit(nvrhi::SubmitDesc().setCommandLists(segments.data(), segmentCount).setSignals(&second, 1));
        };
        device->waitForIdle();
        device->runGarbageCollection();
        for (uint32_t i = 0; i < warmup + measured; ++i)
        {
            if (i < warmup)
                recordByHand();
            else
                handFrames.push_back(CountAllocations(recordByHand));
            if (i % 4 == 3)
            {
                device->waitForIdle();
                device->runGarbageCollection();
            }
        }
        device->waitForIdle();
        CheckCounters(tag);

        // a loop's median; its frames differ in growths alone
        uint32_t growingFrames = 0;
        auto median = [&](const char* loop, const std::vector<Allocations>& frames)
        {
            std::vector<uint32_t> totals;
            for (const Allocations& frame : frames)
            {
                totals.push_back(frame.Total());
                growingFrames += frame.growths ? 1 : 0;
                const auto differs = std::mismatch(frame.sites.begin(), frame.sites.end(), frames[0].sites.begin(),
                                                   frames[0].sites.end()).first;
                check(frame.sites == frames[0].sites && (frame.growths == 0 || (vulkan && parallel)),
                      tag + ": " + loop + ": every frame allocates at the same sites; " +
                          (differs == frame.sites.end() ? std::string() : differs->first));
            }
            std::sort(totals.begin(), totals.end());
            return totals[totals.size() / 2];
        };
        const uint32_t throughAllocator = graphFrames[0].counts[allocation_counter::NvrhiAllocator];
        const uint32_t byGraph = median("the graph", graphFrames), byHand = median("by hand", handFrames);
        std::printf("%s: %u frames of %u draws per view: execute %u allocations per frame (operator new %u, nvrhi "
                    "allocator %u), the same calls by hand %u; frames with a growth %u; every other phase 0\n",
                    tag.c_str(), measured, draws, byGraph, byGraph - throughAllocator, throughAllocator, byHand,
                    growingFrames);
        if (std::getenv("E5_ALLOC_TRACE"))
            for (const auto& [site, count] : graphFrames[0].sites)
                std::printf("  %u x %s\n", count, site.c_str());
        check(handFrames[0].foreign.empty(), tag + ": the baseline allocates outside nvrhi; " + handFrames[0].foreign);
        check(growingFrames < measured / 2, tag + ": growths in " + std::to_string(growingFrames) + " frames");
        check(byGraph == byHand, tag + ": execute allocates as much as the same nvrhi calls by hand (" +
                                     std::to_string(byGraph) + " and " + std::to_string(byHand) + ")");
        check(g_Warnings.load() == warningsBefore, tag + ": no warning");
        std::printf("ok: %s\n", tag.c_str());

        service = nullptr;
        renderGraph = nullptr;
        registry = nullptr;
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
        else if (severity == log::Severity::Warning && std::strstr(text, "not consumed by vertex shader"))
            ++g_UnconsumedAttributeWarnings;
        else if (severity == log::Severity::Warning)
            ++g_Warnings;
        if (severity >= log::Severity::Warning)
            std::printf("LOG: %s\n", text);
    });
    try
    {
        const bool steadyOnly = argc == 3 && std::string(argv[2]) == "steady";
        check(argc == 2 || steadyOnly, "specify d3d11, d3d12 or vulkan (and steady, for that case alone)");
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

        if (!steadyOnly)
        {
            RunBufferGroupSwitch(deviceManager, api);

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
            for (bool inputAssembler : { false, true })
                RunGeometry(deviceManager, api, api != nvrhi::GraphicsAPI::D3D11, inputAssembler);
        }

        deviceManager->Shutdown();
        deviceManager = nullptr;

        // the allocation counts, on a device without the nvrhi validation layer (it allocates; the cases above have it)
        deviceManager = nvrhi::TakeOver(app::DeviceManager::Create(api));
        check(deviceManager != nullptr, "device manager");
        params.enableGPUValidation = false;
        params.enableNvrhiValidationLayer = false;
        check(deviceManager->CreateInstance(params), "graphics instance");
        check(deviceManager->CreateHeadlessDevice(params), "headless device");
        std::printf("device: %s\n", deviceManager->GetRendererString());
        for (bool parallel : { false, true })
            if (!parallel || api != nvrhi::GraphicsAPI::D3D11)
                RunSteadyState(deviceManager, api, parallel);
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
