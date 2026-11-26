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
#include <nvrhi/core/foundation.h>
#include <nvrhi/core/autoptr.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/GpuScene.h>
#include <nvrhi/nvrhi.h>
#include <vector>
#include <mutex>
#include <filesystem>

namespace Json
{
    class Value;
}

namespace donut::vfs
{
    class IFileSystem;
}

struct GeometryData;
struct SkinningConstants;   // donut/shaders/skinning_cb.h

namespace donut::engine
{
    class ShaderFactory;
    struct SceneImportResult;
    class TextureCache;
    class ThreadPool;
    class DescriptorTableManager;
    struct ISceneImporter;
    struct MeshInfo;

    // The skinning dispatch's constants of a skinned mesh made from `prototype`: Scene::UpdateSkinnedMeshes and the
    // render graph's SkinningPass both call it (plan step E5, sub-plan item E5.5).
    void FillSkinningConstants(const MeshInfo& prototype, const MeshInfo& skinned, bool firstFrame,
                               SkinningConstants& constants);

    // The graph path (sub-plan item E5.5): the skinned meshes' vertex buffers are not host-kept (the render graph's
    // ImportBuffer rejects keepInitialState buffers); between frames they rest in this state, which the render service
    // imports them in and returns them to: vertex fetch and bindless or SRV reads need no transition. The CPU path
    // keeps them host-kept in VertexBuffer, as before. The joint buffers stay host-kept (ShaderResource) on both paths:
    // the skinning pass writes them on the graph's upload list, with automatic barriers, as UpdateSkinnedMeshes does.
    constexpr nvrhi::ResourceStates c_SkinnedVertexBufferRestingState = nvrhi::ResourceStates(
        uint32_t(nvrhi::ResourceStates::VertexBuffer) | uint32_t(nvrhi::ResourceStates::ShaderResource));

    class Scene: public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Scene)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    protected:
        nvrhi::AutoPtr<vfs::IFileSystem> m_fs;
        nvrhi::AutoPtr<SceneTypeFactory> m_SceneTypeFactory;
        nvrhi::AutoPtr<TextureCache> m_TextureCache;
        nvrhi::AutoPtr<DescriptorTableManager> m_DescriptorTable;
        nvrhi::AutoPtr<SceneGraph> m_SceneGraph;
        nvrhi::AutoPtr<ISceneImporter> m_SceneImporter;
        std::vector<SceneImportResult> m_Models;
        bool m_EnableBindlessResources = false;
        bool m_UseResourceDescriptorHeapBindless = false;

        // The instance, geometry and material buffers and their CPU images (plan step E4): made at the first
        // RefreshBuffers, from the flags above.
        nvrhi::AutoPtr<GpuScene> m_GpuScene;
        // Scatter: the render-graph path (plan step E5; SetGpuSceneUploadMode).
        GpuSceneUploadMode m_GpuSceneUploadMode = GpuSceneUploadMode::CpuRows;
        // Buffers that CreateMeshBufferObjects made and whose data WriteMeshBufferData has not written yet.
        struct PendingMeshBufferData
        {
            nvrhi::AutoPtr<BufferGroup> buffers;
            bool indexData = false;
            bool vertexData = false;
        };
        std::vector<PendingMeshBufferData> m_PendingMeshBufferData;
        // The graph path: buffers WriteMeshBufferData wrote, whose CPU data is kept until ConfirmMeshBufferData says
        // that the list holding the writes is submitted; CreateMeshBufferObjects makes the writes due again when it
        // was not.
        std::vector<PendingMeshBufferData> m_UnconfirmedMeshBufferData;
        // What CreateMeshBufferObjects scanned last: it scans again only when the graph, its mesh table's serial or the
        // number of skinned instances changed.
        const SceneGraph* m_MeshScanGraph = nullptr;
        uint64_t m_MeshScanSerial = ~uint64_t(0);
        size_t m_MeshScanSkinnedInstances = ~size_t(0);
        // Pipelined rendering (E7.8): BeginPipelinedRecording to EndPipelinedRecording.
        PipelinedRecordingCheck m_PipelinedRecording;

        nvrhi::DeviceHandle m_Device;
        nvrhi::ShaderHandle m_SkinningShader;
        nvrhi::ComputePipelineHandle m_SkinningPipeline;
        nvrhi::BindingLayoutHandle m_SkinningBindingLayout;

        bool m_RayTracingSupported = false;
        bool m_SceneStructureChanged = false;

        void LoadModelAsync(
            uint32_t index,
            const std::filesystem::path& fileName,
            ThreadPool* threadPool);

        void LoadModels(
            const Json::Value& modelList, 
            const std::filesystem::path& scenePath, 
            ThreadPool* threadPool);

        void LoadSceneGraph(const Json::Value& nodeList, SceneGraphNode* parent);
        void LoadAnimations(const Json::Value& nodeList);

        void UpdateSkinnedMeshes(nvrhi::ICommandList* commandList, uint32_t frameIndex);

        virtual void CreateMeshBuffers(nvrhi::ICommandList* commandList);
        // The objects of CreateMeshBuffers, of every mesh and skinned instance that has none (may throw std::bad_alloc).
        // Returns whether it made anything.
        bool ScanMeshBufferObjects();

        virtual bool LoadCustomData(Json::Value& rootNode, const std::filesystem::path& scenePath, ThreadPool* threadPool);
    public:
        ~Scene();

        Scene(
            nvrhi::IDevice* device,
            ShaderFactory *shaderFactory,
            vfs::IFileSystem *fs,
            TextureCache *textureCache,
            DescriptorTableManager *descriptorTable,
            SceneTypeFactory *sceneTypeFactory = nullptr);
        
        void FinishedLoading(uint32_t frameIndex);

        // Processes animations, transforms, bounding boxes etc.
        void RefreshSceneGraph(uint32_t frameIndex);

        // Creates missing buffers, uploads vertex buffers, instance data, materials, etc. The instance, geometry and
        // material rows are GpuScene's: only the rows that changed are written (plan step E4).
        // InstanceData::prevTransform is the transform a row was uploaded with by the last RefreshBuffers of an earlier
        // frame (a frame being a SceneGraph::Refresh call): call it every frame that refreshes the graph, as before, or
        // the first frame after a gap gets the transforms of the last frame it ran, not of the previous one.
        // Does nothing on the render-graph path (IsOnGraphPath, below): the render service updates the GpuScene there.
        void RefreshBuffers(nvrhi::ICommandList* commandList, uint32_t frameIndex);

        // A combination of RefreshSceneGraph and RefreshBuffers
        void Refresh(nvrhi::ICommandList* commandList, uint32_t frameIndex);

        bool Load(const std::filesystem::path& jsonFileName);

        virtual bool LoadWithThreadPool(const std::filesystem::path& sceneFileName, ThreadPool* threadPool);

        static const SceneLoadingStats& GetLoadingStats();

        // Creates an empty scene graph for hand-built scenes
        nvrhi::AutoPtr<SceneGraph> CreateSceneGraph();

        [[nodiscard]] SceneGraph* GetSceneGraph() const { return m_SceneGraph; }
        [[nodiscard]] nvrhi::IDescriptorTable* GetDescriptorTable() const { return m_DescriptorTable ? m_DescriptorTable->GetDescriptorTable() : nullptr; }
        [[nodiscard]] nvrhi::IBuffer* GetMaterialBuffer() const { return m_GpuScene ? m_GpuScene->GetMaterialBuffer() : nullptr; }
        [[nodiscard]] nvrhi::IBuffer* GetGeometryBuffer() const { return m_GpuScene ? m_GpuScene->GetGeometryBuffer() : nullptr; }
        [[nodiscard]] nvrhi::IBuffer* GetInstanceBuffer() const { return m_GpuScene ? m_GpuScene->GetInstanceBuffer() : nullptr; }
        [[nodiscard]] GpuScene* GetGpuScene() const { return m_GpuScene; }

        // Can return nullptr if not yet created (by Scene::RefreshBuffers)
        GeometryData* GetGeometryData(const MeshGeometry& geometry) const;

        // ------------------------------------------------------------------------------------ the render-graph path
        //
        // Plan step E5 (sub-plan item E5.4, open point 9): a donut::render::RenderService drives the GpuScene in its
        // frame instead of RefreshBuffers. The scene keeps owning the GpuScene and the mesh buffers. Module-private.

        // The GpuScene's upload mode, a creation option: set it before Load / LoadWithThreadPool (before the GpuScene is
        // made), so that nothing is replaced later; FE_INVALID_ARGS once a GpuScene of another mode exists. Scatter puts
        // the scene on the graph path: RefreshBuffers then does nothing (FinishedLoading still makes and writes the
        // loaded meshes' buffers, and uploads the loaded instance, geometry and material rows on its command list, as
        // RefreshBuffers does on the CPU path, so that the first frame's previous transforms are the loaded ones), and
        // the service calls the functions below each frame. Applications call
        // RefreshSceneGraph each frame, as before.
        nvrhi::FRESULT SetGpuSceneUploadMode(GpuSceneUploadMode mode) noexcept
        {
            if (m_GpuScene && m_GpuScene->GetUploadMode() != mode)
                return nvrhi::FE_INVALID_ARGS;
            m_GpuSceneUploadMode = mode;
            return nvrhi::FS_OK;
        }
        [[nodiscard]] GpuSceneUploadMode GetGpuSceneUploadMode() const noexcept { return m_GpuSceneUploadMode; }
        [[nodiscard]] bool IsOnGraphPath() const noexcept { return m_GpuSceneUploadMode == GpuSceneUploadMode::Scatter; }
        // The GpuScene in the scene's mode, made from the scene's flags as RefreshBuffers makes it; null when it cannot be.
        GpuScene* EnsureGpuScene() noexcept;
        // The last RefreshSceneGraph saw structural changes: mesh buffers may be due and the meshes' instance buffer
        // must be bound again (BindMeshInstanceBuffers).
        [[nodiscard]] bool HasStructureChanged() const noexcept { return m_SceneStructureChanged; }
        // At extraction, without a command list: creates the buffer objects and descriptors of meshes that have none,
        // and the skinned instances' BufferGroups, vertex buffers, joint buffers and skinning binding sets, as
        // CreateMeshBuffers would, without writing any data; then GpuScene::InvalidateAssets when it made anything (their
        // data is due: WriteMeshBufferData, HasMeshBufferDataPending). It scans only when the graph, its mesh table's
        // serial or its number of skinned instances changed. First, the writes of a WriteMeshBufferData that no
        // ConfirmMeshBufferData followed (the frame's Prepare ran, its lists were never submitted) are due again.
        // FE_OUT_OF_MEMORY when an allocation failed (what was made stays; the next call scans again). The virtual
        // CreateMeshBuffers is not called on the graph path (no class in the tree overrides it).
        nvrhi::FRESULT CreateMeshBufferObjects() noexcept;
        // In the upload pass's Prepare, on the graph's upload list: writes the data of the buffers CreateMeshBufferObjects
        // made and gives them their permanent states, as CreateMeshBuffers does. On the CPU path the CPU data is freed
        // at once; on the graph path it is kept until ConfirmMeshBufferData.
        void WriteMeshBufferData(nvrhi::ICommandList* commandList);
        // The command list of the last WriteMeshBufferData will be submitted (the upload pass's records use calls it
        // from its Execute: the graph's executor submits the upload list with the lists it records): frees the CPU
        // data of what it wrote. A write it never confirms is made again by the next CreateMeshBufferObjects; one that
        // was submitted is never written again (the buffers have their permanent states then).
        void ConfirmMeshBufferData() noexcept;
        // CreateMeshBufferObjects made buffers whose data WriteMeshBufferData has not written yet (the upload pass adds
        // a use for it).
        [[nodiscard]] bool HasMeshBufferDataPending() const noexcept
        {
            m_PipelinedRecording.Check("Scene::HasMeshBufferDataPending");
            return !m_PendingMeshBufferData.empty();
        }
        // Points every mesh's BufferGroup::instanceBuffer at the GpuScene's instance buffer (after it was made or
        // replaced, or after structural changes), before any recording reads it. A replaced buffer is reported to the
        // application (SceneFrame::sceneBuffersReplaced): passes that cache binding sets need ResetBindingCache.
        void BindMeshInstanceBuffers();
        // Pipelined rendering (design 7.5; plan step E7, item E7.8; PipelinedRecordingCheck, GpuScene.h): while a
        // pipelined frame records on another thread, the upload pass writes the mesh data there (WriteMeshBufferData in
        // its Prepare, ConfirmMeshBufferData in its Execute) and the main thread touches neither the pending mesh writes
        // nor the GpuScene until the join. BeginPipelinedRecording, on the main thread when it hands the frame over,
        // also begins the GpuScene's; EndPipelinedRecording, after the join, ends both. In between, Debug builds warn
        // once about a call of CreateMeshBufferObjects, CreateMeshBuffers, WriteMeshBufferData, ConfirmMeshBufferData,
        // HasMeshBufferDataPending, BindMeshInstanceBuffers or EnsureGpuScene made on the main thread (and the GpuScene
        // about its own). Never begun (serial frames), nothing is checked.
        void BeginPipelinedRecording() noexcept;
        void EndPipelinedRecording() noexcept;
        [[nodiscard]] bool IsPipelinedRecording() const noexcept { return m_PipelinedRecording.IsRecording(); }

    };
}
