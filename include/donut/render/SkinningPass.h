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

// Skinning as an IGpuPass (aggregate docs/ecs-design.md 7.4, D16; plan step E5, sub-plan
// docs/plans/2026-11-21-donut-ecs-e5.md, items E5.0 and E5.5).
//
// One compute use over the snapshot's skinned instances (RenderSnapshot::skinnedInstances: the instances whose joints
// moved this frame or the previous one, with their joint matrices computed at extraction). Per instance, as
// Scene::UpdateSkinnedMeshes does: write the joint matrices into the leaf's joint buffer, then dispatch
// skinning_cs.hlsl with the constants of engine::FillSkinningConstants (Scene.h), which both paths call. The skinned
// meshes' vertex buffers are imported by the caller and written here (UAVWrite); no pass body writes a leaf (the
// first-frame flag was taken at extraction).
//
// Module-private C++, like GpuScene.h.

#pragma once

#include <donut/render/RenderGraph.h>
#include <nvrhi/core/foundation.h>
#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <memory>

namespace donut::engine
{
    class ShaderFactory;
}

namespace donut::render
{
    struct RenderSnapshot;

    // `snapshot` and `vertexBuffers` must stay valid until the graph has executed (the snapshot does; the ids come
    // from the arena).
    struct SkinningGraphInputs
    {
        const RenderSnapshot* snapshot = nullptr;
        // One per RenderSnapshot::skinnedInstances entry: the skinned mesh's vertex buffer, imported this frame.
        const ResourceId* vertexBuffers = nullptr;
        uint32_t vertexBufferCount = 0;
    };

    struct SkinningGraphOutputs
    {
        // The versions written, one per input (an arena array); the inputs when nothing was added.
        const ResourceId* vertexBuffers = nullptr;
        uint32_t vertexBufferCount = 0;
    };

    class SkinningPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SkinningPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        // Construction allocates nothing; the first Prepare makes the pipeline from `shaderFactory`.
        SkinningPass(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory) noexcept;
        ~SkinningPass();

        SkinningPass(const SkinningPass&) = delete;
        SkinningPass& operator=(const SkinningPass&) = delete;

        // Render graph (13.2). Adds nothing when the snapshot has no skinned instance.
        SkinningGraphOutputs AddToGraph(IRenderGraph* graph, const SkinningGraphInputs& inputs);

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
