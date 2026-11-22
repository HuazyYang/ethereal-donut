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

// The skinning pass (plan step E5, sub-plan item E5.5); see SkinningPass.h.
//
// One compute use per frame. Its Setup writes (UAVWrite) the imported vertex buffer of every skinned instance it
// dispatches. Its Prepare makes the pipeline once, the binding sets of new instances, and writes each dispatched
// instance's joint matrices from the snapshot into the leaf's joint buffer on the graph's upload list: the joint
// buffers are host-kept (ShaderResource), and the upload list records with automatic barriers, so that they are written
// exactly as Scene::UpdateSkinnedMeshes writes them, and the list is submitted before the frame's other lists. Its
// Execute, with the plan's barriers only, binds and dispatches per instance in snapshot order with
// engine::FillSkinningConstants, as Scene::UpdateSkinnedMeshes does. The prototype's vertex buffer, read through a raw
// SRV, has its permanent state from the upload list (Scene::WriteMeshBufferData) and is not a graph resource.

#include <donut/render/SkinningPass.h>
#include <donut/core/log.h>
#include <donut/engine/Scene.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/render/RenderExtract.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/skinning_cs.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/skinning_cs.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/skinning_cs.spirv.h"
#endif
#endif

using namespace donut::math;
#include <donut/shaders/skinning_cb.h>

namespace donut::render
{
    namespace
    {
        // One dispatch, resolved at AddToGraph (review fix 2): the pass reads no leaf after it, so that a leaf detached
        // and released, or given other buffers, while the frame records changes nothing. The buffers are held by the
        // pass until its next AddToGraph.
        struct SkinningDispatch
        {
            uint32_t input;                       // its index in SkinningGraphInputs::vertexBuffers
            nvrhi::IBuffer* prototypeVertices;
            nvrhi::IBuffer* joints;
            nvrhi::IBuffer* vertices;             // the imported buffer
            const float4x4* jointMatrices;        // in the snapshot
            uint32_t jointBytes;
            SkinningConstants constants;          // engine::FillSkinningConstants with the snapshot's first-frame flag
            const char* name;                     // the leaf's name (the marker), in the arena; null when empty
            nvrhi::IBindingSet* bindingSet;       // Prepare: the cached set
        };
        static_assert(std::is_trivially_destructible_v<SkinningDispatch>);

        // The use's frame-arena data.
        struct SkinningUse
        {
            uint32_t count;
            ResourceId* vertexBuffers;            // one per input; Setup puts in the version each dispatch writes
            SkinningDispatch* dispatches;
            uint32_t dispatchCount;
        };
        static_assert(std::is_trivially_destructible_v<SkinningUse>);

        // The objects one dispatch needs, or false when the instance has not all of them yet (its buffers are made
        // by Scene::CreateMeshBufferObjects at extraction, before the RenderExtract phase).
        bool Dispatchable(const engine::SkinnedMeshInstance* leaf) noexcept
        {
            if (!leaf || !leaf->jointBuffer || leaf->joints.empty())
                return false;
            const engine::MeshInfo* prototype = leaf->GetPrototypeMesh();
            const engine::MeshInfo* skinned = leaf->GetMesh();
            return prototype && skinned && prototype->buffers && prototype->buffers->vertexBuffer && skinned->buffers &&
                   skinned->buffers->vertexBuffer && prototype->totalVertices != 0;
        }
    }

    struct SkinningPass::Impl
    {
        nvrhi::ShaderHandle shader;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::ComputePipelineHandle pipeline;

        // The binding sets, keyed by the skinned vertex buffer (the set holds a reference to it, so a key is never a
        // reused address while it is cached); the other two buffers are checked. Entries unused for c_KeepCalls
        // AddToGraph calls are dropped; every call ages them, with or without skinning (review fix 4).
        struct CachedSet
        {
            nvrhi::BindingSetHandle set;
            nvrhi::IBuffer* prototypeVertices = nullptr;
            nvrhi::IBuffer* joints = nullptr;
            uint64_t lastUse = 0;
        };
        static constexpr uint64_t c_KeepCalls = 64;
        std::unordered_map<nvrhi::IBuffer*, CachedSet> sets;
        uint64_t calls = 0;
        // The buffers of the last AddToGraph's dispatches, held until the next one (capacity kept).
        std::vector<nvrhi::BufferHandle> frameBuffers;
    };

    SkinningPass::SkinningPass(nvrhi::IDevice* device, engine::ShaderFactory* shaderFactory) noexcept
        : m_Device(device), m_ShaderFactory(shaderFactory)
    {
    }

    SkinningPass::~SkinningPass() = default;

    SkinningGraphOutputs SkinningPass::AddToGraph(IRenderGraph* graph, const SkinningGraphInputs& inputs)
    {
        SkinningGraphOutputs outputs{ inputs.vertexBuffers, inputs.vertexBufferCount };
        if (!m_Impl)
        {
            m_Impl.reset(new (std::nothrow) Impl());
            if (!m_Impl)
                return outputs;
        }
        Impl& impl = *m_Impl;
        // Every call ages the cache and releases the previous frame's buffers.
        const uint64_t now = ++impl.calls;
        for (auto it = impl.sets.begin(); it != impl.sets.end();)
        {
            if (it->second.lastUse + Impl::c_KeepCalls < now)
                it = impl.sets.erase(it);
            else
                ++it;
        }
        impl.frameBuffers.clear();

        const RenderSnapshot* snapshot = inputs.snapshot;
        if (!graph || !snapshot || snapshot->skinnedInstanceCount == 0)
            return outputs;
        const uint32_t count = snapshot->skinnedInstanceCount;
        if (inputs.vertexBufferCount < count || !inputs.vertexBuffers)
        {
            log::warning("SkinningPass::AddToGraph: one vertex buffer per skinned instance of the snapshot is needed");
            return outputs;
        }

        uint32_t dispatches = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            if (inputs.vertexBuffers[i] != InvalidResourceId && Dispatchable(snapshot->skinnedInstances[i].leaf))
                ++dispatches;
        }
        if (dispatches == 0)
            return outputs;

        // The arena arrays: the ids (copied: the outputs do not alias the inputs) and the dispatches.
        auto* ids = static_cast<ResourceId*>(graph->AllocateFrameData(sizeof(ResourceId) * count, alignof(ResourceId)));
        auto* resolved = static_cast<SkinningDispatch*>(
            graph->AllocateFrameData(sizeof(SkinningDispatch) * dispatches, alignof(SkinningDispatch)));
        if (!ids || !resolved)
            return outputs;
        impl.frameBuffers.reserve(size_t(dispatches) * 3);
        uint32_t at = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            ids[i] = inputs.vertexBuffers[i];
            const ExtractedSkinnedInstance& instance = snapshot->skinnedInstances[i];
            const engine::SkinnedMeshInstance* leaf = instance.leaf;
            if (ids[i] == InvalidResourceId || !Dispatchable(leaf))
                continue;
            SkinningDispatch& dispatch = resolved[at++];
            dispatch.input = i;
            dispatch.prototypeVertices = leaf->GetPrototypeMesh()->buffers->vertexBuffer;
            dispatch.joints = leaf->jointBuffer;
            dispatch.vertices = leaf->GetMesh()->buffers->vertexBuffer;
            dispatch.jointMatrices = snapshot->jointMatrices + instance.jointMatrices.first;
            dispatch.jointBytes = uint32_t(std::min<uint64_t>(uint64_t(instance.jointMatrices.count) * sizeof(float4x4),
                                                              dispatch.joints->getDesc().byteSize));
            engine::FillSkinningConstants(*leaf->GetPrototypeMesh(), *leaf->GetMesh(), instance.firstFrame,
                                          dispatch.constants);
            dispatch.name = nullptr;
            const std::string& name = leaf->GetName();
            if (!name.empty())
            {
                if (char* copy = static_cast<char*>(graph->AllocateFrameData(name.size() + 1, 1)))
                {
                    std::memcpy(copy, name.c_str(), name.size() + 1);
                    dispatch.name = copy;
                }
            }
            dispatch.bindingSet = nullptr;
            impl.frameBuffers.push_back(dispatch.prototypeVertices);
            impl.frameBuffers.push_back(dispatch.joints);
            impl.frameBuffers.push_back(dispatch.vertices);
        }

        SkinningUse* use = NewFrameData<SkinningUse>(graph, SkinningUse{ count, ids, resolved, dispatches });
        if (!use || graph->AddPass(PassDesc{ "Skinning", PassType::Compute, PassFlags::None, 0 }, this, use, nullptr) !=
                        nvrhi::FS_OK)
            return outputs;
        outputs.vertexBuffers = ids;
        outputs.vertexBufferCount = count;
        return outputs;
    }

    nvrhi::FRESULT SkinningPass::Setup(ISetupContext* context, void* data) noexcept
    {
        SkinningUse& use = *static_cast<SkinningUse*>(data);
        for (uint32_t d = 0; d < use.dispatchCount; ++d)
        {
            // The kernel reads the current positions (the previous ones of this frame) and writes every attribute it
            // has: no FullWrite, the contents are kept.
            ResourceId& id = use.vertexBuffers[use.dispatches[d].input];
            id = context->Write(id, Access::UAVWrite, nvrhi::ShaderType::Compute, nvrhi::SubresourceRegion(),
                                AccessFlags::None);
            if (id == InvalidResourceId)
                return nvrhi::FE_INVALID_ARGS;
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT SkinningPass::Prepare(IPrepareContext* context, void* data) noexcept
    {
        SkinningUse& use = *static_cast<SkinningUse*>(data);
        nvrhi::ICommandList* upload = context->GetUploadCommandList();
        if (!upload || !m_Device || !m_ShaderFactory || !m_Impl)
            return nvrhi::FE_INVALID_ARGS;
        Impl& impl = *m_Impl;
        try
        {
            // The pipeline, once.
            if (!impl.shader)
                impl.shader = m_ShaderFactory->CreateAutoShader("donut/skinning_cs", "main",
                                                                DONUT_MAKE_PLATFORM_SHADER(g_skinning_cs), nullptr,
                                                                nvrhi::ShaderType::Compute);
            if (!impl.shader)
            {
                log::warning("SkinningPass: cannot load skinning_cs");
                return nvrhi::FE_NOT_FOUND;
            }
            if (!impl.bindingLayout)
            {
                nvrhi::BindingLayoutDesc layoutDesc;
                layoutDesc.visibility = nvrhi::ShaderType::Compute;
                layoutDesc.bindings = {
                    nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SkinningConstants)),
                    nvrhi::BindingLayoutItem::RawBuffer_SRV(0),
                    nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
                    nvrhi::BindingLayoutItem::RawBuffer_UAV(0)
                };
                if (NVRHI_FAILED(m_Device->createBindingLayout(layoutDesc, &impl.bindingLayout)))
                    return nvrhi::FE_OUT_OF_MEMORY;
            }
            if (!impl.pipeline)
            {
                nvrhi::ComputePipelineDesc pipelineDesc;
                pipelineDesc.CS = impl.shader;
                pipelineDesc.bindingLayouts = { impl.bindingLayout };
                if (NVRHI_FAILED(m_Device->createComputePipeline(pipelineDesc, &impl.pipeline)))
                    return nvrhi::FE_OUT_OF_MEMORY;
            }

            for (uint32_t d = 0; d < use.dispatchCount; ++d)
            {
                SkinningDispatch& dispatch = use.dispatches[d];

                // The binding set: cached by the skinned vertex buffer.
                Impl::CachedSet& cached = impl.sets[dispatch.vertices];
                if (!cached.set || cached.prototypeVertices != dispatch.prototypeVertices ||
                    cached.joints != dispatch.joints)
                {
                    nvrhi::BindingSetDesc setDesc;
                    setDesc.bindings = {
                        nvrhi::BindingSetItem::PushConstants(0, sizeof(SkinningConstants)),
                        nvrhi::BindingSetItem::RawBuffer_SRV(0, dispatch.prototypeVertices),
                        nvrhi::BindingSetItem::RawBuffer_SRV(1, dispatch.joints),
                        nvrhi::BindingSetItem::RawBuffer_UAV(0, dispatch.vertices)
                    };
                    cached.set = nullptr;
                    if (NVRHI_FAILED(m_Device->createBindingSet(setDesc, impl.bindingLayout, &cached.set)))
                    {
                        impl.sets.erase(dispatch.vertices);
                        return nvrhi::FE_OUT_OF_MEMORY;
                    }
                    cached.prototypeVertices = dispatch.prototypeVertices;
                    cached.joints = dispatch.joints;
                }
                cached.lastUse = impl.calls;
                dispatch.bindingSet = cached.set;

                // The joint matrices of the snapshot (the legacy formula, E5.1), into the host-kept joint buffer: the
                // upload list's automatic barriers take it to CopyDest and back to ShaderResource, as on the CPU path.
                if (dispatch.jointBytes != 0)
                    upload->writeBuffer(dispatch.joints, dispatch.jointMatrices, dispatch.jointBytes);
            }
        }
        catch (const std::bad_alloc&)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT SkinningPass::Execute(IPassContext* context, const void* data) noexcept
    {
        const SkinningUse& use = *static_cast<const SkinningUse*>(data);
        if (!m_Impl || !m_Impl->pipeline)
            return nvrhi::FE_INVALID_ARGS;
        nvrhi::ICommandList* commandList = context->GetCommandList();
        for (uint32_t d = 0; d < use.dispatchCount; ++d)
        {
            const SkinningDispatch& dispatch = use.dispatches[d];
            if (!dispatch.bindingSet)
                return nvrhi::FE_INVALID_ARGS;
            if (dispatch.name)
                commandList->beginMarker(dispatch.name);

            nvrhi::ComputeState state;
            state.pipeline = m_Impl->pipeline;
            state.bindings = { dispatch.bindingSet };
            commandList->setComputeState(state);
            commandList->setPushConstants(&dispatch.constants, sizeof(dispatch.constants));
            commandList->dispatch(dm::div_ceil(dispatch.constants.numVertices, 256u));

            if (dispatch.name)
                commandList->endMarker();
        }
        return nvrhi::FS_OK;
    }

    void SkinningPass::OnResourcesReleased() noexcept
    {
        if (m_Impl)
            m_Impl->sets.clear();
    }

    uint32_t SkinningPass::GetCachedBindingSetCount() const noexcept
    {
        return m_Impl ? uint32_t(m_Impl->sets.size()) : 0;
    }
}
