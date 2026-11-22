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

// The GpuScene upload as an IGpuPass with a compute scatter (aggregate docs/ecs-design.md 7.2, 7.4; plan step E5,
// sub-plan docs/plans/2026-11-21-donut-ecs-e5.md, items E5.0 and E5.4).
//
// The GpuScene is in scatter mode (GpuSceneDesc::uploadMode, Scene::SetGpuSceneUploadMode before loading). At
// RenderService::Extract, in this order: Scene::CreateMeshBufferObjects (buffer objects, descriptors and skinned
// BufferGroups of new meshes, no data), GpuScene::Update, GpuScene::CreateBuffers, Scene::BindMeshInstanceBuffers; then
// the buffers are imported. AddToGraph adds, in order:
//
//     growth   (copy, only when GpuSceneUploadGraphInputs::bufferChanges names a replaced instance buffer and the
//              records are not a resend) the old instance buffer's rows into the new one
//     records  (copy) the scatter records of the last Update (GpuScene::GetScatterRecords) into a registry-persistent
//              staging buffer with writeBuffer; in its Prepare, on the upload list: the data of the mesh buffers made at
//              extraction (Scene::WriteMeshBufferData), then the geometry and material rows and the per-material
//              constant buffers (GpuScene::UploadAssets)
//     scatter  (compute) one thread per record (gpu_scene_scatter_cs.hlsl, gpu_scene_scatter_cb.h): settle records copy
//              transform into prevTransform, transform records copy the old transform into prevTransform before they
//              write the new one, row records write whole rows; uint words only, byte-address views (D3D11 too). Its
//              Execute acknowledges the records (GpuScene::CommitScatter with GpuSceneScatterRecords::serial); records
//              of a frame that never executes are sent again in full by the next Update.
//
// The records are read when the uses record (Execute), not at AddToGraph: pipelined rendering (step E7) must keep
// them, and the replaced buffer, until then.
//
// Module-private C++, like GpuScene.h.

#pragma once

#include <donut/render/RenderGraph.h>
#include <nvrhi/core/foundation.h>
#include <nvrhi/nvrhi.h>

#include <memory>

namespace donut::engine
{
    class GpuScene;
    struct GpuSceneBufferChanges;
    class Scene;
    class ShaderFactory;
}

namespace donut::render
{
    // The GpuScene's buffers must stay valid until the graph has executed.
    struct GpuSceneUploadGraphInputs
    {
        engine::GpuScene* gpuScene = nullptr;      // scatter mode; Update and CreateBuffers ran this frame
        // Optional: the scene whose mesh buffers made at extraction get their data in Prepare (WriteMeshBufferData).
        engine::Scene* scene = nullptr;
        // What this frame's CreateBuffers did (GpuScene::GetBufferChanges; it stays valid until the next one): the
        // growth copy's source. Null: no growth copy.
        const engine::GpuSceneBufferChanges* bufferChanges = nullptr;
        // GpuScene's buffers, imported this frame. The geometry and material buffers are optional (no bindless
        // buffers); the instance buffer is required.
        ResourceId instanceBuffer = InvalidResourceId;
        ResourceId geometryBuffer = InvalidResourceId;
        ResourceId materialBuffer = InvalidResourceId;
    };

    // The versions the uses wrote (or the inputs when nothing wrote them).
    struct GpuSceneUploadGraphOutputs
    {
        ResourceId instanceBuffer = InvalidResourceId;
        ResourceId geometryBuffer = InvalidResourceId;
        ResourceId materialBuffer = InvalidResourceId;
    };

    class GpuSceneUploadPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(GpuSceneUploadPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        // Construction allocates nothing; the first Prepare makes the scatter pipeline from `shaderFactory`.
        GpuSceneUploadPass(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory) noexcept;
        ~GpuSceneUploadPass();

        GpuSceneUploadPass(const GpuSceneUploadPass&) = delete;
        GpuSceneUploadPass& operator=(const GpuSceneUploadPass&) = delete;

        // Render graph (13.2). When the last Update recorded nothing to write (no record, no asset row, no mesh data),
        // adds nothing, acknowledges the empty records at once and returns the inputs.
        GpuSceneUploadGraphOutputs AddToGraph(IRenderGraph* graph, const GpuSceneUploadGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:
        struct Impl;
        nvrhi::IDevice* m_Device;                  // not a reference, as in GpuScene: the scene holds the device
        engine::ShaderFactory* m_ShaderFactory;
        std::unique_ptr<Impl> m_Impl;
    };
}
