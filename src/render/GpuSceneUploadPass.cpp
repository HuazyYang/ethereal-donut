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

// The GpuScene upload pass (plan step E5, sub-plan item E5.4); see GpuSceneUploadPass.h.

#include <donut/render/GpuSceneUploadPass.h>
#include <donut/core/log.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>

#include <algorithm>
#include <cstdio>
#include <new>
#include <string>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/gpu_scene_scatter_cs.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/gpu_scene_scatter_cs.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/gpu_scene_scatter_cs.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/bindless.h>
#include <donut/shaders/gpu_scene_scatter_cb.h>

namespace donut::render
{
    namespace
    {
        enum class UploadUseKind : uint32_t
        {
            Growth,    // copy: the replaced instance buffer's rows into the new one
            Records,   // copy: the records into the staging buffer; Prepare: mesh data and asset rows
            Scatter,   // compute: the kernel
        };

        // One use's frame-arena data.
        struct UploadUse
        {
            UploadUseKind kind;
            engine::GpuScene* gpuScene;
            engine::Scene* scene;
            ResourceId source;           // Growth: the old instance buffer; Scatter: the staging buffer
            ResourceId target;           // Growth, Scatter: the instance buffer; Records: the staging buffer. Setup
                                         // replaces it with the version the use wrote.
            uint64_t bytes;              // Growth: the bytes to copy. Records: 1 when its Execute acknowledges the records
                                         // (no scatter use)
            uint64_t serial;             // Records, Scatter: GpuSceneScatterRecords::serial at AddToGraph
            GpuSceneScatterConstants constants;   // Records, Scatter: the staging layout and the record counts
        };
        static_assert(std::is_trivially_destructible_v<UploadUse>);

        constexpr uint64_t c_MinStagingBytes = 64 * 1024;

        // The records in the staging buffer: row records first (128-byte aligned), then transform records, then the
        // settle slots.
        GpuSceneScatterConstants Layout(const engine::GpuSceneScatterRecords& records, uint64_t& bytes)
        {
            GpuSceneScatterConstants constants = {};
            constants.rowRecordOffset = 0;
            constants.rowRecordCount = records.rowCount;
            constants.transformRecordOffset = records.rowCount * uint32_t(sizeof(GpuSceneRowRecord));
            constants.transformRecordCount = records.transformCount;
            constants.settleRecordOffset = constants.transformRecordOffset +
                                           records.transformCount * uint32_t(sizeof(GpuSceneTransformRecord));
            constants.settleRecordCount = records.settleCount;
            constants.instanceRowStride = uint32_t(sizeof(InstanceData));
            constants.groupsX = 0;   // AddToGraph sets it
            bytes = uint64_t(constants.settleRecordOffset) + uint64_t(records.settleCount) * sizeof(uint32_t);
            return constants;
        }

        // The records of the use's update, or null when the GpuScene has been updated since AddToGraph (the records
        // are read at Execute; E7's pipelining must keep them until then).
        bool CurrentRecords(const UploadUse& use, engine::GpuSceneScatterRecords& records)
        {
            use.gpuScene->GetScatterRecords(records);
            if (records.serial == use.serial && records.rowCount == use.constants.rowRecordCount &&
                records.transformCount == use.constants.transformRecordCount &&
                records.settleCount == use.constants.settleRecordCount)
                return true;
            log::warning("GpuSceneUploadPass: the GpuScene was updated after AddToGraph; its records are not written");
            return false;
        }
    }

    struct GpuSceneUploadPass::Impl
    {
        std::string stagingName;          // the registry-persistent staging buffer, named after the GpuScene
        const engine::GpuScene* stagingOwner = nullptr;
        uint64_t stagingBytes = 0;        // its size: grows to a power of two, never shrinks
        nvrhi::ShaderHandle shader;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::ComputePipelineHandle pipeline;
        // The binding set of the last scatter, with the buffers it names (it holds references to them, so a pointer
        // is never reused while it is cached).
        nvrhi::BindingSetHandle bindingSet;
        nvrhi::IBuffer* boundRecords = nullptr;
        nvrhi::IBuffer* boundInstances = nullptr;
    };

    GpuSceneUploadPass::GpuSceneUploadPass(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory) noexcept
        : m_Device(device), m_ShaderFactory(shaderFactory)
    {
    }

    GpuSceneUploadPass::~GpuSceneUploadPass() = default;

    GpuSceneUploadGraphOutputs GpuSceneUploadPass::AddToGraph(IRenderGraph* graph, const GpuSceneUploadGraphInputs& inputs)
    {
        GpuSceneUploadGraphOutputs outputs{ inputs.instanceBuffer, inputs.geometryBuffer, inputs.materialBuffer };
        engine::GpuScene* gpuScene = inputs.gpuScene;
        if (!graph || !gpuScene || gpuScene->GetUploadMode() != engine::GpuSceneUploadMode::Scatter)
        {
            log::warning("GpuSceneUploadPass::AddToGraph: needs a GpuScene in scatter mode");
            return outputs;
        }

        engine::GpuSceneScatterRecords records;
        gpuScene->GetScatterRecords(records);
        const uint32_t recordCount = records.settleCount + records.transformCount + records.rowCount;
        const engine::GpuSceneBufferChanges* changes = inputs.bufferChanges;
        const bool growth = changes && changes->previousInstanceBuffer && changes->previousInstanceRows != 0 &&
                            !records.resend;
        const bool assets = gpuScene->HasPendingAssets();
        const bool meshData = inputs.scene && inputs.scene->HasMeshBufferDataPending();
        if (recordCount == 0 && !growth && !assets && !meshData)
        {
            gpuScene->CommitScatter(records.serial);   // nothing to write
            return outputs;
        }
        // Without the instance buffer, nothing is scattered and the records stay unacknowledged: the next update sends
        // every row again.
        nvrhi::BufferDesc instanceDesc;
        if ((recordCount != 0 || growth) &&
            (inputs.instanceBuffer == InvalidResourceId ||
             graph->GetBufferDesc(inputs.instanceBuffer, instanceDesc) != nvrhi::FS_OK ||
             instanceDesc.byteSize < uint64_t(gpuScene->GetInstanceRowCount()) * sizeof(InstanceData)))
        {
            log::warning("GpuSceneUploadPass::AddToGraph: the instance buffer is missing or too small; nothing is scattered");
            return outputs;
        }

        if (!m_Impl)
        {
            m_Impl.reset(new (std::nothrow) Impl());
            if (!m_Impl)
                return outputs;
        }
        Impl& impl = *m_Impl;
        if (impl.stagingOwner != gpuScene)
        {
            // One staging buffer per GpuScene, kept by the registry: a pass made again for it reuses the buffer.
            char name[64];
            std::snprintf(name, sizeof(name), "GpuSceneScatterRecords/%p", static_cast<const void*>(gpuScene));
            impl.stagingName = name;
            impl.stagingOwner = gpuScene;
        }

        ResourceId instances = inputs.instanceBuffer;
        if (growth)
        {
            const ResourceId previous = graph->ImportBuffer(changes->previousInstanceBuffer,
                                                            engine::c_GpuSceneRestingState,
                                                            engine::c_GpuSceneRestingState);
            const uint64_t bytes = std::min<uint64_t>(uint64_t(changes->previousInstanceRows) * sizeof(InstanceData),
                                                      std::min(instanceDesc.byteSize,
                                                               changes->previousInstanceBuffer->getDesc().byteSize));
            UploadUse* use = NewFrameData<UploadUse>(graph, UploadUse{ UploadUseKind::Growth, gpuScene, nullptr, previous,
                                                                       instances, bytes, 0, {} });
            if (!use || graph->AddPass(PassDesc{ "GpuSceneGrowth", PassType::Copy, PassFlags::None, 0 }, this, use,
                                       nullptr) != nvrhi::FS_OK)
                return outputs;
            instances = use->target;
        }

        uint64_t recordBytes = 0;
        GpuSceneScatterConstants constants = Layout(records, recordBytes);
        const uint32_t maxGroupsX = m_MaxDispatchGroupsX ? m_MaxDispatchGroupsX : uint32_t(GPU_SCENE_SCATTER_MAX_GROUPS_X);
        const uint32_t groups = (recordCount + GPU_SCENE_SCATTER_GROUP_SIZE - 1) / GPU_SCENE_SCATTER_GROUP_SIZE;
        constants.groupsX = std::max(1u, std::min(groups, maxGroupsX));
        const uint64_t needed = std::max(recordBytes, c_MinStagingBytes);
        if (impl.stagingBytes < needed)
        {
            uint64_t bytes = c_MinStagingBytes;
            while (bytes < needed)
                bytes *= 2;
            impl.stagingBytes = bytes;
        }
        nvrhi::BufferDesc stagingDesc;
        stagingDesc.byteSize = impl.stagingBytes;
        stagingDesc.canHaveRawViews = true;
        stagingDesc.debugName = "GpuSceneScatterRecords";
        const ResourceId staging = graph->GetPersistentBuffer(impl.stagingName.c_str(), stagingDesc);

        // The records' copy; its Prepare writes the mesh data and the asset rows on the upload list. Without records,
        // its Execute acknowledges them (after the growth copy, earlier in plan order): an acknowledgement only ever
        // comes from an Execute, so that what a Prepare wrote in a frame that never executes is written again.
        UploadUse* recordUse = NewFrameData<UploadUse>(graph, UploadUse{ UploadUseKind::Records, gpuScene, inputs.scene,
                                                                         InvalidResourceId, staging, 0, records.serial,
                                                                         constants });
        if (!recordUse || graph->AddPass(PassDesc{ "GpuSceneRecords", PassType::Copy, PassFlags::None, 0 }, this,
                                         recordUse, nullptr) != nvrhi::FS_OK)
            return outputs;

        if (recordCount == 0)
        {
            recordUse->bytes = 1;
            outputs.instanceBuffer = instances;
            return outputs;
        }
        UploadUse* scatterUse = NewFrameData<UploadUse>(graph, UploadUse{ UploadUseKind::Scatter, gpuScene, nullptr,
                                                                          recordUse->target, instances, 0, records.serial,
                                                                          constants });
        if (!scatterUse || graph->AddPass(PassDesc{ "GpuSceneScatter", PassType::Compute, PassFlags::None, 0 }, this,
                                          scatterUse, nullptr) != nvrhi::FS_OK)
            return outputs;
        outputs.instanceBuffer = scatterUse->target;
        return outputs;
    }

    nvrhi::FRESULT GpuSceneUploadPass::Setup(ISetupContext* context, void* data) noexcept
    {
        UploadUse& use = *static_cast<UploadUse*>(data);
        switch (use.kind)
        {
        case UploadUseKind::Growth:
            context->Read(use.source, Access::CopySource, nvrhi::ShaderType::None, nvrhi::SubresourceRegion());
            use.target = context->Write(use.target, Access::CopyDest, nvrhi::ShaderType::None, nvrhi::SubresourceRegion(),
                                        AccessFlags::None);
            break;
        case UploadUseKind::Records:
            use.target = context->Write(use.target, Access::CopyDest, nvrhi::ShaderType::None, nvrhi::SubresourceRegion(),
                                        AccessFlags::None);
            break;
        case UploadUseKind::Scatter:
            context->Read(use.source, Access::ShaderRead, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion());
            use.target = context->Write(use.target, Access::UAVWrite, nvrhi::ShaderType::Compute,
                                        nvrhi::SubresourceRegion(), AccessFlags::None);
            break;
        }
        return use.target != InvalidResourceId ? nvrhi::FS_OK : nvrhi::FE_INVALID_ARGS;
    }

    nvrhi::FRESULT GpuSceneUploadPass::Prepare(IPrepareContext* context, void* data) noexcept
    {
        UploadUse& use = *static_cast<UploadUse*>(data);
        if (use.kind == UploadUseKind::Records)
        {
            // The data of the mesh buffers made at extraction, then the geometry and material rows that name them.
            nvrhi::ICommandList* commandList = context->GetUploadCommandList();
            if (!commandList)
                return nvrhi::FE_INVALID_ARGS;
            try
            {
                if (use.scene)
                    use.scene->WriteMeshBufferData(commandList);
            }
            catch (const std::bad_alloc&)
            {
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            const nvrhi::FRESULT result = use.gpuScene->UploadAssets(commandList);
            if (NVRHI_FAILED(result))
                log::warning("GpuSceneUploadPass: the geometry and material rows were not all written");
            return nvrhi::FS_OK;
        }
        if (use.kind != UploadUseKind::Scatter || (m_Impl && m_Impl->pipeline))
            return nvrhi::FS_OK;

        // The scatter pipeline, once.
        if (!m_Impl || !m_ShaderFactory || !m_Device)
            return nvrhi::FE_INVALID_ARGS;
        Impl& impl = *m_Impl;
        try
        {
            if (!impl.shader)
                impl.shader = m_ShaderFactory->CreateAutoShader("donut/gpu_scene_scatter_cs", "main",
                                                                DONUT_MAKE_PLATFORM_SHADER(g_gpu_scene_scatter_cs), nullptr,
                                                                nvrhi::ShaderType::Compute);
            if (!impl.shader)
            {
                log::warning("GpuSceneUploadPass: cannot load gpu_scene_scatter_cs");
                return nvrhi::FE_NOT_FOUND;
            }
            if (!impl.bindingLayout)
            {
                nvrhi::BindingLayoutDesc layoutDesc;
                layoutDesc.visibility = nvrhi::ShaderType::Compute;
                layoutDesc.bindings = {
                    nvrhi::BindingLayoutItem::PushConstants(0, sizeof(GpuSceneScatterConstants)),
                    nvrhi::BindingLayoutItem::RawBuffer_SRV(0),
                    nvrhi::BindingLayoutItem::RawBuffer_UAV(0)
                };
                if (NVRHI_FAILED(m_Device->createBindingLayout(layoutDesc, &impl.bindingLayout)))
                    return nvrhi::FE_OUT_OF_MEMORY;
            }
            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.CS = impl.shader;
            pipelineDesc.bindingLayouts = { impl.bindingLayout };
            if (NVRHI_FAILED(m_Device->createComputePipeline(pipelineDesc, &impl.pipeline)))
                return nvrhi::FE_OUT_OF_MEMORY;
        }
        catch (const std::bad_alloc&)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT GpuSceneUploadPass::Execute(IPassContext* context, const void* data) noexcept
    {
        const UploadUse& use = *static_cast<const UploadUse*>(data);
        nvrhi::ICommandList* commandList = context->GetCommandList();
        switch (use.kind)
        {
        case UploadUseKind::Growth:
        {
            nvrhi::IBuffer* source = context->GetBuffer(use.source);
            nvrhi::IBuffer* target = context->GetBuffer(use.target);
            if (!source || !target)
                return nvrhi::FE_INVALID_ARGS;
            if (use.bytes)
                commandList->copyBuffer(target, 0, source, 0, use.bytes);
            return nvrhi::FS_OK;
        }
        case UploadUseKind::Records:
        {
            // The upload list that Prepare wrote the mesh data on is submitted with the lists recorded now (the executor
            // records every pass, failed or not, and then submits).
            if (use.scene)
                use.scene->ConfirmMeshBufferData();
            engine::GpuSceneScatterRecords records;
            if (!CurrentRecords(use, records))
                return nvrhi::FE_INVALID_ARGS;
            nvrhi::IBuffer* staging = context->GetBuffer(use.target);
            if (!staging)
                return nvrhi::FE_INVALID_ARGS;
            const GpuSceneScatterConstants& constants = use.constants;
            if (records.rowCount)
                commandList->writeBuffer(staging, records.rows, size_t(records.rowCount) * sizeof(GpuSceneRowRecord),
                                         constants.rowRecordOffset);
            if (records.transformCount)
                commandList->writeBuffer(staging, records.transforms,
                                         size_t(records.transformCount) * sizeof(GpuSceneTransformRecord),
                                         constants.transformRecordOffset);
            if (records.settleCount)
                commandList->writeBuffer(staging, records.settleSlots, size_t(records.settleCount) * sizeof(uint32_t),
                                         constants.settleRecordOffset);
            if (use.bytes)
                use.gpuScene->CommitScatter(use.serial);   // no scatter use: recorded here
            return nvrhi::FS_OK;
        }
        case UploadUseKind::Scatter:
        {
            engine::GpuSceneScatterRecords records;
            if (!m_Impl || !m_Impl->pipeline || !CurrentRecords(use, records))
                return nvrhi::FE_INVALID_ARGS;
            Impl& impl = *m_Impl;
            nvrhi::IBuffer* staging = context->GetBuffer(use.source);
            nvrhi::IBuffer* instances = context->GetBuffer(use.target);
            if (!staging || !instances)
                return nvrhi::FE_INVALID_ARGS;
            if (!impl.bindingSet || impl.boundRecords != staging || impl.boundInstances != instances)
            {
                nvrhi::BindingSetDesc setDesc;
                setDesc.bindings = {
                    nvrhi::BindingSetItem::PushConstants(0, sizeof(GpuSceneScatterConstants)),
                    nvrhi::BindingSetItem::RawBuffer_SRV(0, staging),
                    nvrhi::BindingSetItem::RawBuffer_UAV(0, instances)
                };
                impl.bindingSet = nullptr;
                try
                {
                    if (NVRHI_FAILED(m_Device->createBindingSet(setDesc, impl.bindingLayout, &impl.bindingSet)))
                        return nvrhi::FE_OUT_OF_MEMORY;
                }
                catch (const std::bad_alloc&)
                {
                    return nvrhi::FE_OUT_OF_MEMORY;
                }
                impl.boundRecords = staging;
                impl.boundInstances = instances;
            }
            nvrhi::ComputeState state;
            state.pipeline = impl.pipeline;
            state.bindings = { impl.bindingSet };
            commandList->setComputeState(state);
            commandList->setPushConstants(&use.constants, sizeof(GpuSceneScatterConstants));
            const uint32_t recordCount = records.settleCount + records.transformCount + records.rowCount;
            const uint32_t groups = (recordCount + GPU_SCENE_SCATTER_GROUP_SIZE - 1) / GPU_SCENE_SCATTER_GROUP_SIZE;
            const uint32_t groupsX = use.constants.groupsX;
            const uint32_t groupsY = (groups + groupsX - 1) / groupsX;
            commandList->dispatch(groupsX, groupsY);
            // Recorded: the records reach the GPU with this frame.
            use.gpuScene->CommitScatter(use.serial);
            return nvrhi::FS_OK;
        }
        }
        return nvrhi::FE_INVALID_ARGS;
    }

    nvrhi::FRESULT GpuSceneUploadPass::ReleaseStagingBuffer(IResourceRegistry* registry) noexcept
    {
        if (!registry)
            return nvrhi::FE_INVALID_ARGS;
        if (!m_Impl || m_Impl->stagingName.empty())
            return nvrhi::FE_NOT_FOUND;
        OnResourcesReleased();   // the cached binding set names the buffer
        return registry->ReleasePersistent(m_Impl->stagingName.c_str());
    }

    void GpuSceneUploadPass::OnResourcesReleased() noexcept
    {
        if (!m_Impl)
            return;
        m_Impl->bindingSet = nullptr;
        m_Impl->boundRecords = nullptr;
        m_Impl->boundInstances = nullptr;
    }
}
