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

namespace donut::engine
{
    class ShaderFactory;
    struct SceneImportResult;
    class TextureCache;
    class ThreadPool;
    class DescriptorTableManager;
    struct ISceneImporter;

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

    };
}
