/*
* Copyright (c) 2014-2024, NVIDIA CORPORATION. All rights reserved.
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

/*
License for JsonCpp

JsonCpp is Public Domain

The JsonCpp library's source code, including accompanying documentation, 
tests and demonstration applications, are licensed under the following
conditions...

Baptiste Lepilleur and The JsonCpp Authors explicitly disclaim copyright in all 
jurisdictions which recognize such a disclaimer. In such jurisdictions, 
this software is released into the Public Domain.
*/

#include <donut/engine/Scene.h>
#include "SceneImporterImpl.h"
#include <donut/engine/ThreadPool.h>
#include <donut/core/json.h>
#include <donut/core/log.h>
#include <donut/core/string_utils.h>
#include <nvrhi/common/misc.h>
#include <json/json-forwards.h>

#include "donut/engine/ShaderFactory.h"
#include <donut/engine/TextureCache.h>
#include <donut/core/vfs/VFS.h>

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
#include <donut/shaders/material_cb.h>
#include <donut/shaders/skinning_cb.h>
#include <donut/shaders/bindless.h>

using namespace donut::vfs;
using namespace donut::engine;

static SceneLoadingStats g_LoadingStats;

const SceneLoadingStats& Scene::GetLoadingStats()
{
    return g_LoadingStats;
}

Scene::Scene(
    nvrhi::IDevice* device,
    ShaderFactory* shaderFactory,
    vfs::IFileSystem *fs,
    TextureCache* textureCache,
    DescriptorTableManager* descriptorTable,
    SceneTypeFactory* sceneTypeFactory)
    : m_fs(fs)
    , m_SceneTypeFactory(sceneTypeFactory)
    , m_TextureCache(textureCache)
    , m_DescriptorTable(descriptorTable)
    , m_Device(device)
{
    if (!m_SceneTypeFactory)
        m_SceneTypeFactory = MAKE_RC_OBJ_PTR(SceneTypeFactory);

    m_EnableBindlessResources = !!m_DescriptorTable;
    m_RayTracingSupported = m_Device->queryFeatureSupport(nvrhi::Feature::RayTracingAccelStruct);

    m_SkinningShader = shaderFactory->CreateAutoShader("donut/skinning_cs", "main", DONUT_MAKE_PLATFORM_SHADER(g_skinning_cs), nullptr, nvrhi::ShaderType::Compute);

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SkinningConstants)),
            nvrhi::BindingLayoutItem::RawBuffer_SRV(0),
            nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
            nvrhi::BindingLayoutItem::RawBuffer_UAV(0)
        };

        m_Device->createBindingLayout(layoutDesc, &m_SkinningBindingLayout);
    }

    {
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_SkinningBindingLayout };
        pipelineDesc.CS = m_SkinningShader;
        m_Device->createComputePipeline(pipelineDesc, &m_SkinningPipeline);
    }
}

bool Scene::Load(const std::filesystem::path& jsonFileName)
{
    ThreadPool threadPool;
    return LoadWithThreadPool(jsonFileName, &threadPool);
}

bool Scene::LoadWithThreadPool(const std::filesystem::path& sceneFileName, ThreadPool* threadPool)
{
    g_LoadingStats.ObjectsLoaded = 0;
    g_LoadingStats.ObjectsTotal = 0;
    
    m_SceneGraph = m_SceneTypeFactory->CreateGraph();

    std::string sceneFileName2 = sceneFileName.generic_string();
    int len = static_cast<int>(sceneFileName2.length());

    if (!(len > 11 && strcasecmp(sceneFileName2.data() + len - 11, ".scene.json")) == 0) {
        ++g_LoadingStats.ObjectsTotal;
        m_Models.resize(1);
        LoadModelAsync(0, sceneFileName, threadPool);

        if (threadPool)
            threadPool->WaitForTasks();

        auto modelResult = m_Models[0];
        if (!modelResult.rootNode)
            return false;

        m_SceneGraph->SetRootNode(modelResult.rootNode);
    } else {
        auto rootNode = MAKE_RC_OBJ_PTR(SceneGraphNode);
        rootNode->SetName("SceneRoot");
        m_SceneGraph->SetRootNode(rootNode);

        std::filesystem::path scenePath = sceneFileName.parent_path();

        Json::Value documentRoot;
        if (!json::LoadFromFile(*m_fs, sceneFileName, documentRoot))
            return false;

        if (documentRoot.isObject())
        {
            if (!LoadCustomData(documentRoot, scenePath, threadPool))
                return false;

            LoadModels(documentRoot["models"], scenePath, threadPool);
            LoadSceneGraph(documentRoot["graph"], rootNode);
            LoadAnimations(documentRoot["animations"]);
        }
        else
        {
            log::error("Unrecognized structure of the scene description file.");
            return false;
        }
    }

    return true;
}

void Scene::LoadModelAsync(
    uint32_t index,
    const std::filesystem::path& fileName,
    ThreadPool* threadPool)
{
    auto ext = fileName.extension().generic_string();
    if (strcasecmp(ext.data(), ".glb") == 0 || strcasecmp(ext.data(), ".gltf") == 0) {
        if (!m_SceneImporter || NVRHI_FAILED(m_SceneImporter->QueryInterface(nvrhi::uuid_of<GltfImporter>(), nullptr)))
            m_SceneImporter = MAKE_RC_OBJ_PTR(GltfImporter, m_fs, m_SceneTypeFactory);
    } else {
        log::error("Donut scene loader does not support this scene archive type");
        NVRHI_ASSERT(0 && "Unsupported scene archive type");
        return;
#if 0
        if(!m_SceneImporter || NVRHI_FAILED(m_SceneImporter->QueryInterface(nvrhi::uuid_of<AssimpSceneImporter>(), nullptr)))
            m_SceneImporter = MAKE_RC_OBJ_PTR(AssimpSceneImporter, m_fs, m_SceneTypeFactory);
#endif
    }

    if (threadPool)
    {
        threadPool->AddTask([this, index, threadPool, fileName]()
        {
            SceneImportResult result;
            m_SceneImporter->Load(fileName, *m_TextureCache, g_LoadingStats, threadPool, result);
            ++g_LoadingStats.ObjectsLoaded;
            m_Models[index] = result;
        });
    }
    else
    {
        SceneImportResult result;
        m_SceneImporter->Load(fileName, *m_TextureCache, g_LoadingStats, threadPool, result);
        ++g_LoadingStats.ObjectsLoaded;
        m_Models[index] = result;
    }
}

void Scene::LoadModels(
    const Json::Value& modelList,
    const std::filesystem::path& scenePath,
    ThreadPool* threadPool)
{
    if (!modelList.isArray())
    {
        return;
    }

    m_Models.resize(modelList.size());
    uint32_t index = 0;
    for (const auto& model : modelList)
    {
        ++g_LoadingStats.ObjectsTotal;

        std::filesystem::path fileName = scenePath / std::filesystem::path(model.asString());

        LoadModelAsync(index, fileName, threadPool);

        ++index;
    }

    if (threadPool)
        threadPool->WaitForTasks();
}

void Scene::LoadSceneGraph(const Json::Value& nodeList, SceneGraphNode* parent)
{
    for (const auto& src : nodeList)
    {
        if (!src.isObject())
        {
            log::warning("Non-object node in the scene graph definition.");
            continue;
        }

        std::string nodeName;
        const auto& name = src["name"];
        if (name.isString())
        {
            nodeName = name.asString();
        }

        nvrhi::AutoPtr<SceneGraphNode> customParent = parent;
        const auto& parentNode = src["parent"];
        if (parentNode.isString())
        {
            customParent = m_SceneGraph->FindNode(parentNode.asString());
            if (!customParent)
            {
                log::warning("Custom parent '%s' specified for node '%s' not found, skipping the node.",
                    parentNode.asCString(), nodeName.c_str());
                continue;
            }
        }
        else if (!parentNode.isNull())
        {
            log::warning("Custom parent specification for node '%s' is not a string, ignoring.",
                nodeName.c_str());
        }

        nvrhi::AutoPtr<SceneGraphNode> dst;

        const auto& modelNode = src["model"];
        if (!modelNode.isNull())
        {
            if (!modelNode.isIntegral())
            {
                log::warning("Model references in the scene graph must be indices into the model array.");
                continue;
            }

            int modelIndex = modelNode.asInt();
            if (modelIndex < 0 || modelIndex >= int(m_Models.size()))
            {
                log::warning("Referenced model %d is not defined in the model array.", modelIndex);
                continue;
            }

            const auto& loadedModel = m_Models[modelIndex];
            if (!loadedModel.rootNode)
            {
                continue;
            }

            dst = loadedModel.rootNode;
        }
        else
        {
            dst = MAKE_RC_OBJ_PTR(SceneGraphNode);
        }

        dst = m_SceneGraph->Attach(customParent, dst);

        dst->SetName(nodeName);
        
        const auto& translation = src["translation"];
        if (!translation.isNull())
        {
            double3 value = double3::zero();
            translation >> value;
            dst->SetTranslation(value);
        }

        const auto& rotation = src["rotation"];
        if (!rotation.isNull())
        {
            double4 value = double4(0.0, 0.0, 0.0, 1.0);
            rotation >> value;
            dst->SetRotation(dm::dquat::fromXYZW(value));
        }
        else
        {
            const auto& euler = src["euler"];
            if (!euler.isNull())
            {
                double3 value = double3::zero();
                euler >> value;
                dst->SetRotation(rotationQuat(value));
            }
        }

        const auto& scaling = src["scaling"];
        if (!scaling.isNull())
        {
            double3 value = double3(1.0);
            scaling >> value;
            dst->SetScaling(value);
        }

        const auto& children = src["children"];
        if (!children.isNull())
        {
            LoadSceneGraph(children, dst);
        }

        const auto& leafTypeNode = src["type"];
        if (leafTypeNode.isString())
        {
            auto leaf = m_SceneTypeFactory->CreateLeaf(leafTypeNode.asString());
            if (leaf)
            {
                dst->SetLeaf(leaf);
                leaf->Load(src);
            }
            else
            {
                log::warning("Unknown leaf type '%s' for node '%s', skipping.",
                    leafTypeNode.asCString(), dst->GetName().c_str());
            }
        }
        else if (!leafTypeNode.isNull())
        {
            log::warning("Leaf type specification for node '%s' is not a string, skipping.",
                dst->GetName().c_str());
        }
    }
}

static dm::float4 ReadUpToFloat4(const Json::Value& node)
{
    if (node.isNumeric())
        return dm::float4(node.asFloat());

    if (node.isArray())
    {
        float4 result = float4::zero();
        for (int i = 0; i < std::min(4, int(node.size())); i++)
        {
            result[i] = node[i].asFloat();
        }
        return result;
    }

    return float4::zero();
}

void Scene::LoadAnimations(const Json::Value& nodeList)
{
    nvrhi::AutoPtr<SceneGraphNode> animationContainer;

    for (const auto& animationNode : nodeList)
    {
        const auto& animation = MAKE_RC_OBJ_PTR(SceneGraphAnimation);

        const auto& sceneAnimationNode = MAKE_RC_OBJ_PTR(SceneGraphNode);
        sceneAnimationNode->SetLeaf(animation);

        const auto& nameNode = animationNode["name"];
        if (nameNode.isString())
        {
            animation->SetName(nameNode.asString());
        }

        const auto& channelsNode = animationNode["channels"];
        if (channelsNode.isArray())
        {
            int channelIndex = -1;
            for (const auto& channelSrc : channelsNode)
            {
                // Increment the index in the beginning because there are 'continue' statements below
                ++channelIndex;

                const auto& sampler = MAKE_RC_OBJ_PTR(animation::Sampler);

                const auto& modeNode = channelSrc["mode"];
                if (modeNode.isString())
                {
                    if (modeNode.asString() == "step")
                        sampler->SetInterpolationMode(animation::InterpolationMode::Step);
                    else if (modeNode.asString() == "linear")
                        sampler->SetInterpolationMode(animation::InterpolationMode::Linear);
                    else if (modeNode.asString() == "slerp")
                        sampler->SetInterpolationMode(animation::InterpolationMode::Slerp);
                    else if (modeNode.asString() == "hermite")
                        sampler->SetInterpolationMode(animation::InterpolationMode::HermiteSpline);
                    else if (modeNode.asString() == "catmull-rom")
                        sampler->SetInterpolationMode(animation::InterpolationMode::CatmullRomSpline);
                    else
                        log::warning("Unknown interpolation mode '%s' specified for animation '%s' channel %d. "
                            "Valid interpolation modes are: step, linear, hermite, catmull-rom.",
                            modeNode.asCString(), animation->GetName().c_str(), channelIndex);
                }
                else
                {
                    sampler->SetInterpolationMode(animation::InterpolationMode::Step);
                    log::warning("Interpolation mode is not specified for animation '%s' channel %d, using step.",
                        animation->GetName().c_str(), channelIndex);
                }

                const auto& attributeNode = channelSrc["attribute"];
                AnimationAttribute attribute = AnimationAttribute::Undefined;
                if (attributeNode.isString() && !attributeNode.asString().empty())
                {
                    if (attributeNode.asString() == "translation")
                        attribute = AnimationAttribute::Translation;
                    else if (attributeNode.asString() == "rotation")
                        attribute = AnimationAttribute::Rotation;
                    else if (attributeNode.asString() == "scaling")
                        attribute = AnimationAttribute::Scaling;
                    else
                        attribute = AnimationAttribute::LeafProperty;
                }
                else
                {
                    log::warning("Attribute is not specified for animation '%s' channel %d, ignoring.",
                        animation->GetName().c_str(), channelIndex);
                    continue;
                }

                int keyframeIndex = -1;
                for (const auto& dataPoint : channelSrc["data"])
                {
                    ++keyframeIndex;

                    const auto& timeNode = dataPoint["time"];
                    if (!timeNode.isNumeric())
                    {
                        log::warning("Invalid keyframe %d in animation '%s' channel %d: time is not specified or is not numeric.",
                            keyframeIndex, animation->GetName().c_str(), channelIndex);
                        continue;
                    }

                    animation::Keyframe keyframe;
                    keyframe.time = timeNode.asFloat();
                    keyframe.value = ReadUpToFloat4(dataPoint["value"]);
                    keyframe.inTangent = ReadUpToFloat4(dataPoint["inTangent"]);
                    keyframe.outTangent = ReadUpToFloat4(dataPoint["outTangent"]);

                    sampler->AddKeyframe(keyframe);
                }

                auto processTargetNode = [this, &animation, &sampler, attribute, &attributeNode, channelIndex](const Json::Value& targetNode)
                {
                    if (targetNode.isString())
                    {
                        std::string targetName = targetNode.asString();
                        if (donut::string_utils::starts_with(targetName, "material:"))
                        {
                            targetName = targetName.substr(9);

                            nvrhi::AutoPtr<Material> material;
                            for (const auto& it : m_SceneGraph->GetMaterials())
                            {
                                if (it->name == targetName)
                                {
                                    material = it;
                                    break;
                                }
                            }

                            if (material)
                            {
                                const auto& channel = MAKE_RC_OBJ_PTR(SceneGraphAnimationChannel, sampler, material);
                                channel->SetLeafProperyName(attributeNode.asString());
                                animation->AddChannel(channel);
                            }
                            else
                            {
                                log::warning("Target material '%s' specified for animation '%s' channel %d not found, ignoring.",
                                    std::string(targetName).c_str(), animation->GetName().c_str(), channelIndex);
                            }
                        }
                        else
                        {
                            const auto& target = m_SceneGraph->FindNode(targetNode.asString());
                            if (target)
                            {
                                const auto& channel = MAKE_RC_OBJ_PTR(SceneGraphAnimationChannel, sampler, target, attribute);
                                if (attribute == AnimationAttribute::LeafProperty)
                                    channel->SetLeafProperyName(attributeNode.asString());
                                animation->AddChannel(channel);
                            }
                            else
                            {
                                log::warning("Target node '%s' specified for animation '%s' channel %d not found, ignoring.",
                                    targetNode.asCString(), animation->GetName().c_str(), channelIndex);
                            }
                        }
                    }
                    else if (!targetNode.isNull())
                    {
                        log::warning("Target node specification for animation '%s' channel %d is not a string, ignoring.",
                            animation->GetName().c_str(), channelIndex);
                    }
                };

                const auto& targetNode = channelSrc["target"];
                if (!targetNode.isNull())
                {
                    processTargetNode(targetNode);
                }
                else
                {
                    const auto& targetsNode = channelSrc["targets"];
                    if (targetsNode.isArray())
                    {
                        for (const auto& targetArrayItem : targetsNode)
                        {
                            processTargetNode(targetArrayItem);
                        }
                    }
                }
            }
        }

        if (!animation->GetChannels().empty())
        {
            if (!animationContainer)
            {
                animationContainer = MAKE_RC_OBJ_PTR(SceneGraphNode);
                animationContainer->SetName("Animations");
                m_SceneGraph->Attach(m_SceneGraph->GetRootNode(), animationContainer);
            }
            
            m_SceneGraph->Attach(animationContainer, sceneAnimationNode);
        }
        else
        {
            log::warning("Animation '%s' processed with no valid channels, ignoring.",
                animation->GetName().c_str());
        }
    }
}

bool Scene::LoadCustomData(Json::Value& rootNode, const std::filesystem::path& scenePath, ThreadPool* threadPool)
{
    // Reserved for derived classes
    return true;
}

Scene::~Scene() {}

nvrhi::AutoPtr<SceneGraph> Scene::CreateSceneGraph()
{
    return m_SceneGraph = m_SceneTypeFactory->CreateGraph();
}

void Scene::FinishedLoading(uint32_t frameIndex)
{
    nvrhi::CommandListHandle commandList;
    m_Device->createCommandList(nvrhi::CommandListParameters(), &commandList);
    commandList->open();
    
    CreateMeshBuffers(commandList);
    Refresh(commandList, frameIndex);

    // The graph path (plan step E5): the loaded scene's rows are uploaded here, as RefreshBuffers uploads them on the
    // CPU path, so that the first rendered frame's previous transforms are the loaded ones there too. The first update
    // is a resend: its rows are complete in the image (prevTransform = transform), which is written whole, as the
    // records would write it; the records are then acknowledged.
    if (IsOnGraphPath())
    {
        GpuScene* gpuScene = EnsureGpuScene();
        GpuSceneScatterRecords records;
        if (gpuScene && NVRHI_SUCCEEDED(gpuScene->Update(m_SceneGraph)) && NVRHI_SUCCEEDED(gpuScene->CreateBuffers()))
        {
            gpuScene->GetScatterRecords(records);
            nvrhi::IBuffer* instances = gpuScene->GetInstanceBuffer();
            if (records.resend && instances)
            {
                BindMeshInstanceBuffers();
                commandList->beginTrackingBufferState(instances, c_GpuSceneRestingState);
                commandList->writeBuffer(instances, gpuScene->GetInstanceRows(),
                                         size_t(gpuScene->GetInstanceRowCount()) * sizeof(InstanceData));
                commandList->setBufferState(instances, c_GpuSceneRestingState);
                commandList->commitBarriers();
                if (NVRHI_FAILED(gpuScene->UploadAssets(commandList)))
                    log::warning("Scene::FinishedLoading: the geometry and material rows were not all written");
                gpuScene->CommitScatter(records.serial);
            }
        }

        // The loaded skinned instances are skinned on this list too, as Refresh skins them on the CPU path (sub-plan
        // item E5.5), so that the first rendered frame is not their first skinning there either. Their vertex buffers
        // are not host-kept on the graph path: tracked from their resting state and returned to it.
        const auto& skinnedInstances = m_SceneGraph->GetSkinnedMeshInstances();
        for (const auto& skinnedInstance : skinnedInstances)
            commandList->beginTrackingBufferState(skinnedInstance->GetMesh()->buffers->vertexBuffer,
                                                  c_SkinnedVertexBufferRestingState);
        UpdateSkinnedMeshes(commandList, frameIndex);
        for (const auto& skinnedInstance : skinnedInstances)
            commandList->setBufferState(skinnedInstance->GetMesh()->buffers->vertexBuffer,
                                        c_SkinnedVertexBufferRestingState);
        commandList->commitBarriers();
    }

    commandList->close();
    m_Device->executeCommandList(commandList);
    ConfirmMeshBufferData();   // the graph path keeps the data until its list is submitted
}

void Scene::RefreshSceneGraph(uint32_t frameIndex)
{
    // GpuScene keeps the previous transforms per instance slot (plan step E4) and finds what changed by itself.
    m_SceneStructureChanged = m_SceneGraph->HasPendingStructureChanges();
    m_SceneGraph->Refresh(frameIndex);
}

void Scene::RefreshBuffers(nvrhi::ICommandList* commandList, uint32_t frameIndex)
{
    // On the graph path (plan step E5) the render service updates the GpuScene and skins; an Update here would consume
    // the scatter records.
    if (IsOnGraphPath())
        return;

    if (m_SceneStructureChanged)
    {
        CreateMeshBuffers(commandList);
        // The buffers and descriptors of meshes attached after the last update are made only now: their geometry rows
        // are filled again (a mesh attached between RefreshSceneGraph and RefreshBuffers got rows without them).
        if (m_GpuScene)
            m_GpuScene->InvalidateAssets();
    }

    // The instance, geometry and material rows: GpuScene writes the ones that changed (plan step E4).
    if (!EnsureGpuScene())
    {
        log::warning("Scene::RefreshBuffers: the GPU scene cannot be made");
        return;
    }
    nvrhi::IBuffer* previousInstanceBuffer = m_GpuScene->GetInstanceBuffer();
    if (NVRHI_FAILED(m_GpuScene->Update(m_SceneGraph)) || NVRHI_FAILED(m_GpuScene->Upload(commandList)))
        log::warning("Scene::RefreshBuffers: the GPU scene was not updated");

    if (m_SceneStructureChanged || m_GpuScene->GetInstanceBuffer() != previousInstanceBuffer)
    {
        for (const auto& mesh : m_SceneGraph->GetMeshes())
        {
            if (mesh->buffers)
                mesh->buffers->instanceBuffer = m_GpuScene->GetInstanceBuffer();
        }
    }

    UpdateSkinnedMeshes(commandList, frameIndex);
}

void donut::engine::FillSkinningConstants(const MeshInfo& prototype, const MeshInfo& skinned, bool firstFrame,
                                          SkinningConstants& constants)
{
    const uint32_t vertexOffset = prototype.vertexOffset;
    const BufferGroup* prototypeBuffers = prototype.buffers;
    const BufferGroup* skinnedBuffers = skinned.buffers;

    constants = SkinningConstants{};
    constants.numVertices = prototype.totalVertices;

    constants.flags = 0;
    if (prototypeBuffers->hasAttribute(VertexAttribute::Normal)) constants.flags |= SkinningFlag_Normals;
    if (prototypeBuffers->hasAttribute(VertexAttribute::Tangent)) constants.flags |= SkinningFlag_Tangents;
    if (prototypeBuffers->hasAttribute(VertexAttribute::TexCoord1)) constants.flags |= SkinningFlag_TexCoord1;
    if (prototypeBuffers->hasAttribute(VertexAttribute::TexCoord2)) constants.flags |= SkinningFlag_TexCoord2;
    if (firstFrame) constants.flags |= SkinningFlag_FirstFrame;

    constants.inputPositionOffset = uint32_t(prototypeBuffers->getVertexBufferRange(VertexAttribute::Position).byteOffset + vertexOffset * sizeof(float3));
    constants.inputNormalOffset = uint32_t(prototypeBuffers->getVertexBufferRange(VertexAttribute::Normal).byteOffset + vertexOffset * sizeof(uint32_t));
    constants.inputTangentOffset = uint32_t(prototypeBuffers->getVertexBufferRange(VertexAttribute::Tangent).byteOffset + vertexOffset * sizeof(uint32_t));
    constants.inputTexCoord1Offset = uint32_t(prototypeBuffers->getVertexBufferRange(VertexAttribute::TexCoord1).byteOffset + vertexOffset * sizeof(float2));
    constants.inputTexCoord2Offset = uint32_t(prototypeBuffers->getVertexBufferRange(VertexAttribute::TexCoord2).byteOffset + vertexOffset * sizeof(float2));
    constants.inputJointIndexOffset = uint32_t(prototypeBuffers->getVertexBufferRange(VertexAttribute::JointIndices).byteOffset + vertexOffset * sizeof(uint2));
    constants.inputJointWeightOffset = uint32_t(prototypeBuffers->getVertexBufferRange(VertexAttribute::JointWeights).byteOffset + vertexOffset * sizeof(float4));
    constants.outputPositionOffset = uint32_t(skinnedBuffers->getVertexBufferRange(VertexAttribute::Position).byteOffset);
    constants.outputPrevPositionOffset = uint32_t(skinnedBuffers->getVertexBufferRange(VertexAttribute::PrevPosition).byteOffset);
    constants.outputNormalOffset = uint32_t(skinnedBuffers->getVertexBufferRange(VertexAttribute::Normal).byteOffset);
    constants.outputTangentOffset = uint32_t(skinnedBuffers->getVertexBufferRange(VertexAttribute::Tangent).byteOffset);
    constants.outputTexCoord1Offset = uint32_t(skinnedBuffers->getVertexBufferRange(VertexAttribute::TexCoord1).byteOffset);
    constants.outputTexCoord2Offset = uint32_t(skinnedBuffers->getVertexBufferRange(VertexAttribute::TexCoord2).byteOffset);
}

void Scene::UpdateSkinnedMeshes(nvrhi::ICommandList* commandList, uint32_t frameIndex)
{
    bool skinningMarkerPlaced = false;

    std::vector<dm::float4x4> jointMatrices;
    for (const auto& skinnedInstance : m_SceneGraph->GetSkinnedMeshInstances())
    {
        // Only process the groups that were updated on this or previous frame.
        // Previous frame updates should be processed to copy the current positions to the previous buffer.
        if (skinnedInstance->GetLastUpdateFrameIndex() + 1 < frameIndex)
            continue;

        if (!skinningMarkerPlaced)
        {
            commandList->beginMarker("Skinning");
            skinningMarkerPlaced = true;
        }

        const auto& groupName = skinnedInstance->GetName();
        if (!groupName.empty())
            commandList->beginMarker(groupName.c_str());

        jointMatrices.resize(skinnedInstance->joints.size());
        dm::daffine3 worldToRoot = inverse(skinnedInstance->GetNode()->GetLocalToWorldTransform());

        for (size_t i = 0; i < skinnedInstance->joints.size(); i++)
        {
            auto jointNode = skinnedInstance->joints[i].node.Lock();

            dm::float4x4 jointMatrix = dm::affineToHomogeneous(dm::affine3(jointNode->GetLocalToWorldTransform() * worldToRoot));
            jointMatrix = skinnedInstance->joints[i].inverseBindMatrix * jointMatrix;
            jointMatrices[i] = jointMatrix;
        }

        commandList->writeBuffer(skinnedInstance->jointBuffer, jointMatrices.data(), jointMatrices.size() * sizeof(float4x4));

        nvrhi::ComputeState state;
        state.pipeline = m_SkinningPipeline;
        state.bindings = { skinnedInstance->skinningBindingSet };
        commandList->setComputeState(state);

        SkinningConstants constants{};
        FillSkinningConstants(*skinnedInstance->GetPrototypeMesh(), *skinnedInstance->GetMesh(),
                              !skinnedInstance->skinningInitialized, constants);
        skinnedInstance->skinningInitialized = true;
        commandList->setPushConstants(&constants, sizeof(constants));

        commandList->dispatch(dm::div_ceil(constants.numVertices, 256u));

        if (!groupName.empty())
            commandList->endMarker();
    }

    if (skinningMarkerPlaced)
    {
        commandList->endMarker();
    }
}

void Scene::Refresh(nvrhi::ICommandList* commandList, uint32_t frameIndex)
{
    RefreshSceneGraph(frameIndex);
    RefreshBuffers(commandList, frameIndex);
}


nvrhi::BufferHandle CreateMaterialConstantBuffer(nvrhi::IDevice* device, const std::string& debugName, bool isVirtual)
{
    nvrhi::BufferDesc bufferDesc;
    bufferDesc.byteSize = sizeof(MaterialConstants);
    bufferDesc.debugName = debugName;
    bufferDesc.isConstantBuffer = true;
    bufferDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
    bufferDesc.keepInitialState = true;
    bufferDesc.isVirtual = isVirtual;

    nvrhi::BufferHandle buffer;
    device->createBuffer(bufferDesc, &buffer);
    return buffer;
}


inline void AppendBufferRange(nvrhi::BufferRange& range, size_t size, uint64_t& currentBufferSize)
{
    range.byteOffset = currentBufferSize;
    range.byteSize = nvrhi::align(size, size_t(16));
    currentBufferSize += range.byteSize;
}

void Scene::CreateMeshBuffers(nvrhi::ICommandList* commandList)
{
    m_PipelinedRecording.Check("Scene::CreateMeshBuffers");
    // The objects first, then their data (plan step E5: the graph path makes them at extraction and writes the data in
    // the upload pass's Prepare). The objects and descriptors are made in the same order as before.
    ScanMeshBufferObjects();
    WriteMeshBufferData(commandList);
}

namespace
{
    // The CPU copies of a buffer group's data, once written (the vectors' memory too).
    void FreeMeshCpuData(BufferGroup& buffers, bool indexData, bool vertexData)
    {
        if (indexData)
            std::vector<uint32_t>().swap(buffers.indexData);
        if (vertexData)
        {
            std::vector<float3>().swap(buffers.positionData);
            std::vector<uint32_t>().swap(buffers.normalData);
            std::vector<uint32_t>().swap(buffers.tangentData);
            std::vector<float2>().swap(buffers.texcoord1Data);
            std::vector<float2>().swap(buffers.texcoord2Data);
            std::vector<float4>().swap(buffers.weightData);
            std::vector<vector<uint16_t, 4>>().swap(buffers.jointData);
            std::vector<float>().swap(buffers.radiusData);
        }
    }
}

nvrhi::FRESULT Scene::CreateMeshBufferObjects() noexcept
{
    m_PipelinedRecording.Check("Scene::CreateMeshBufferObjects");
    if (!m_SceneGraph)
        return nvrhi::FE_INVALID_ARGS;
    try
    {
        // What a Prepare wrote without a confirmation: its lists were never submitted, so the buffers are still
        // without data and without their permanent states; written again.
        if (!m_UnconfirmedMeshBufferData.empty())
        {
            m_PendingMeshBufferData.reserve(m_PendingMeshBufferData.size() + m_UnconfirmedMeshBufferData.size());
            for (PendingMeshBufferData& entry : m_UnconfirmedMeshBufferData)
                m_PendingMeshBufferData.push_back(std::move(entry));
            m_UnconfirmedMeshBufferData.clear();
        }

        // New meshes and skinned instances change the mesh table's serial or the skinned-instance count.
        const uint64_t serial = m_SceneGraph->GetMeshes().GetSerial();
        const size_t skinned = m_SceneGraph->GetSkinnedMeshInstances().size();
        if (m_MeshScanGraph == m_SceneGraph.Get() && m_MeshScanSerial == serial && m_MeshScanSkinnedInstances == skinned)
            return nvrhi::FS_OK;
        m_MeshScanGraph = nullptr;   // scanned again after a failure
        ScanMeshBufferObjects();
        m_MeshScanGraph = m_SceneGraph.Get();
        m_MeshScanSerial = serial;
        m_MeshScanSkinnedInstances = skinned;
    }
    catch (const std::bad_alloc&)
    {
        log::warning("Scene::CreateMeshBufferObjects: out of memory");
        return nvrhi::FE_OUT_OF_MEMORY;
    }
    return nvrhi::FS_OK;
}

bool Scene::ScanMeshBufferObjects()
{
    bool made = false;
    for (const auto& mesh : m_SceneGraph->GetMeshes())
    {
        auto buffers = mesh->buffers;

        if (!buffers)
            continue;

        PendingMeshBufferData pending;

        if (!buffers->indexData.empty() && !buffers->indexBuffer)
        {
            nvrhi::BufferDesc bufferDesc;
            bufferDesc.isIndexBuffer = true;
            bufferDesc.byteSize = buffers->indexData.size() * sizeof(uint32_t);
            bufferDesc.debugName = "IndexBuffer";
            bufferDesc.canHaveTypedViews = true;
            bufferDesc.canHaveRawViews = true;
            bufferDesc.format = nvrhi::Format::R32_UINT;
            bufferDesc.isAccelStructBuildInput = m_RayTracingSupported;

            m_Device->createBuffer(bufferDesc, &buffers->indexBuffer);

            if (m_DescriptorTable)
            {
                buffers->indexBufferDescriptor = MAKE_RC_OBJ_PTR(DescriptorHandle, m_DescriptorTable->CreateDescriptorHandle(
                    nvrhi::BindingSetItem::RawBuffer_SRV(0, buffers->indexBuffer)));
            }

            pending.indexData = buffers->indexBuffer != nullptr;
        }

        if (!buffers->vertexBuffer)
        {
            nvrhi::BufferDesc bufferDesc;
            bufferDesc.isVertexBuffer = true;
            bufferDesc.byteSize = 0;
            bufferDesc.debugName = "VertexBuffer";
            bufferDesc.canHaveTypedViews = true;
            bufferDesc.canHaveRawViews = true;
            bufferDesc.isAccelStructBuildInput = m_RayTracingSupported;

            if (!buffers->positionData.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::Position),
                    buffers->positionData.size() * sizeof(buffers->positionData[0]), bufferDesc.byteSize);
            }

            if (!buffers->normalData.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::Normal),
                    buffers->normalData.size() * sizeof(buffers->normalData[0]), bufferDesc.byteSize);
            }

            if (!buffers->tangentData.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::Tangent),
                    buffers->tangentData.size() * sizeof(buffers->tangentData[0]), bufferDesc.byteSize);
            }

            if (!buffers->texcoord1Data.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::TexCoord1),
                    buffers->texcoord1Data.size() * sizeof(buffers->texcoord1Data[0]), bufferDesc.byteSize);
            }

            if (!buffers->texcoord2Data.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::TexCoord2),
                    buffers->texcoord2Data.size() * sizeof(buffers->texcoord2Data[0]), bufferDesc.byteSize);
            }

            if (!buffers->weightData.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::JointWeights),
                    buffers->weightData.size() * sizeof(buffers->weightData[0]), bufferDesc.byteSize);
            }

            if (!buffers->jointData.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::JointIndices),
                    buffers->jointData.size() * sizeof(buffers->jointData[0]), bufferDesc.byteSize);
            }

            if (!buffers->radiusData.empty())
            {
                AppendBufferRange(buffers->getVertexBufferRange(VertexAttribute::CurveRadius),
                    buffers->radiusData.size() * sizeof(buffers->radiusData[0]), bufferDesc.byteSize);
            }

            if (bufferDesc.byteSize != 0)
            {
                m_Device->createBuffer(bufferDesc, &buffers->vertexBuffer);
                if (m_DescriptorTable)
                {
                    buffers->vertexBufferDescriptor =
                        MAKE_RC_OBJ_PTR(DescriptorHandle, m_DescriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, buffers->vertexBuffer)));
                }

                pending.vertexData = buffers->vertexBuffer != nullptr;
            }
        }

        if (pending.indexData || pending.vertexData)
        {
            pending.buffers = buffers;
            m_PendingMeshBufferData.push_back(std::move(pending));
            made = true;
        }
    }

    for (const auto& skinnedInstance : m_SceneGraph->GetSkinnedMeshInstances())
    {
        const auto& skinnedMesh = skinnedInstance->GetMesh();

        if (!skinnedMesh->buffers)
        {
            made = true;
            skinnedMesh->buffers =MAKE_RC_OBJ_PTR(BufferGroup);

            uint32_t totalVertices = skinnedMesh->totalVertices;

            skinnedMesh->buffers->indexBuffer = skinnedInstance->GetPrototypeMesh()->buffers->indexBuffer;
            skinnedMesh->buffers->indexBufferDescriptor = skinnedInstance->GetPrototypeMesh()->buffers->indexBufferDescriptor;

            const auto& prototypeBuffers = skinnedInstance->GetPrototypeMesh()->buffers;
            const auto& skinnedBuffers = skinnedMesh->buffers;

            size_t skinnedVertexBufferSize = 0;
            assert(prototypeBuffers->hasAttribute(VertexAttribute::Position));

            AppendBufferRange(skinnedBuffers->getVertexBufferRange(VertexAttribute::Position),
                totalVertices * sizeof(float3), skinnedVertexBufferSize);

            AppendBufferRange(skinnedBuffers->getVertexBufferRange(VertexAttribute::PrevPosition),
                totalVertices * sizeof(float3), skinnedVertexBufferSize);

            if(prototypeBuffers->hasAttribute(VertexAttribute::Normal))
            {
                AppendBufferRange(skinnedBuffers->getVertexBufferRange(VertexAttribute::Normal),
                    totalVertices * sizeof(uint32_t), skinnedVertexBufferSize);
            }

            if (prototypeBuffers->hasAttribute(VertexAttribute::Tangent))
            {
                AppendBufferRange(skinnedBuffers->getVertexBufferRange(VertexAttribute::Tangent),
                    totalVertices * sizeof(uint32_t), skinnedVertexBufferSize);
            }

            if (prototypeBuffers->hasAttribute(VertexAttribute::TexCoord1))
            {
                AppendBufferRange(skinnedBuffers->getVertexBufferRange(VertexAttribute::TexCoord1),
                    totalVertices * sizeof(float2), skinnedVertexBufferSize);
            }

            if (prototypeBuffers->hasAttribute(VertexAttribute::TexCoord2))
            {
                AppendBufferRange(skinnedBuffers->getVertexBufferRange(VertexAttribute::TexCoord2),
                    totalVertices * sizeof(float2), skinnedVertexBufferSize);
            }

            nvrhi::BufferDesc bufferDesc;
            bufferDesc.isVertexBuffer = true;
            bufferDesc.byteSize = skinnedVertexBufferSize;
            bufferDesc.debugName = "SkinnedVertexBuffer";
            bufferDesc.canHaveTypedViews = true;
            bufferDesc.canHaveRawViews = true;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.isAccelStructBuildInput = m_RayTracingSupported;
            if (IsOnGraphPath())
            {
                // The render service imports it each frame (ImportBuffer rejects host-kept buffers): sub-plan E5.5.
                bufferDesc.keepInitialState = false;
                bufferDesc.initialState = c_SkinnedVertexBufferRestingState;
            }
            else
            {
                bufferDesc.keepInitialState = true;
                bufferDesc.initialState = nvrhi::ResourceStates::VertexBuffer;
            }

            m_Device->createBuffer(bufferDesc, &skinnedBuffers->vertexBuffer);

            if (m_DescriptorTable)
            {
                skinnedBuffers->vertexBufferDescriptor =
                    MAKE_RC_OBJ_PTR(DescriptorHandle, m_DescriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, skinnedBuffers->vertexBuffer)));
            }
        }

        if (!skinnedInstance->jointBuffer)
        {
            made = true;
            nvrhi::BufferDesc jointBufferDesc;
            jointBufferDesc.debugName = "JointBuffer";
            jointBufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            jointBufferDesc.keepInitialState = true;
            jointBufferDesc.canHaveRawViews = true;
            jointBufferDesc.byteSize = sizeof(dm::float4x4) * skinnedInstance->joints.size();
            m_Device->createBuffer(jointBufferDesc, &skinnedInstance->jointBuffer);
        }

        if (!skinnedInstance->skinningBindingSet)
        {
            made = true;
            const auto& prototypeBuffers = skinnedInstance->GetPrototypeMesh()->buffers;
            const auto& skinnedBuffers = skinnedInstance->GetMesh()->buffers;

            nvrhi::BindingSetDesc setDesc;
            setDesc.bindings = {
                nvrhi::BindingSetItem::PushConstants(0, sizeof(SkinningConstants)),
                nvrhi::BindingSetItem::RawBuffer_SRV(0, prototypeBuffers->vertexBuffer),
                nvrhi::BindingSetItem::RawBuffer_SRV(1, skinnedInstance->jointBuffer),
                nvrhi::BindingSetItem::RawBuffer_UAV(0, skinnedBuffers->vertexBuffer)
            };

            m_Device->createBindingSet(setDesc, m_SkinningBindingLayout, &skinnedInstance->skinningBindingSet);
        }
    }

    // The geometry rows of these meshes are filled again with the new buffers' descriptors and offsets.
    if (made && m_GpuScene)
        m_GpuScene->InvalidateAssets();
    return made;
}

void Scene::WriteMeshBufferData(nvrhi::ICommandList* commandList)
{
    m_PipelinedRecording.Check("Scene::WriteMeshBufferData");
    // The graph path keeps the CPU data until the frame is acknowledged (CreateMeshBufferObjects).
    const bool keep = IsOnGraphPath();
    if (keep)
        m_UnconfirmedMeshBufferData.reserve(m_UnconfirmedMeshBufferData.size() + m_PendingMeshBufferData.size());
    for (const PendingMeshBufferData& pending : m_PendingMeshBufferData)
    {
        BufferGroup* buffers = pending.buffers;

        if (pending.indexData && buffers->indexBuffer)
        {
            commandList->beginTrackingBufferState(buffers->indexBuffer, nvrhi::ResourceStates::Common);

            commandList->writeBuffer(buffers->indexBuffer, buffers->indexData.data(), buffers->indexData.size() * sizeof(uint32_t));

            nvrhi::ResourceStates state = nvrhi::ResourceStates::IndexBuffer | nvrhi::ResourceStates::ShaderResource;

            if (buffers->indexBuffer->getDesc().isAccelStructBuildInput)
                state = state | nvrhi::ResourceStates::AccelStructBuildInput;

            commandList->setPermanentBufferState(buffers->indexBuffer, state);
            commandList->commitBarriers();
        }

        if (pending.vertexData && buffers->vertexBuffer)
        {
            commandList->beginTrackingBufferState(buffers->vertexBuffer, nvrhi::ResourceStates::Common);

            if (!buffers->positionData.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::Position);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->positionData.data(), range.byteSize, range.byteOffset);
            }

            if (!buffers->normalData.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::Normal);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->normalData.data(), range.byteSize, range.byteOffset);
            }

            if (!buffers->tangentData.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::Tangent);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->tangentData.data(), range.byteSize, range.byteOffset);
            }

            if (!buffers->texcoord1Data.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::TexCoord1);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->texcoord1Data.data(), range.byteSize, range.byteOffset);
            }

            if (!buffers->texcoord2Data.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::TexCoord2);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->texcoord2Data.data(), range.byteSize, range.byteOffset);
            }

            if (!buffers->weightData.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::JointWeights);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->weightData.data(), range.byteSize, range.byteOffset);
            }

            if (!buffers->jointData.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::JointIndices);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->jointData.data(), range.byteSize, range.byteOffset);
            }

            if (!buffers->radiusData.empty())
            {
                const auto& range = buffers->getVertexBufferRange(VertexAttribute::CurveRadius);
                commandList->writeBuffer(buffers->vertexBuffer, buffers->radiusData.data(), range.byteSize, range.byteOffset);
            }

            nvrhi::ResourceStates state = nvrhi::ResourceStates::VertexBuffer | nvrhi::ResourceStates::ShaderResource;

            if (buffers->vertexBuffer->getDesc().isAccelStructBuildInput)
                state = state | nvrhi::ResourceStates::AccelStructBuildInput;

            commandList->setPermanentBufferState(buffers->vertexBuffer, state);
            commandList->commitBarriers();
        }
    }
    for (PendingMeshBufferData& pending : m_PendingMeshBufferData)
    {
        if (keep)
            m_UnconfirmedMeshBufferData.push_back(std::move(pending));
        else
            FreeMeshCpuData(*pending.buffers, pending.indexData, pending.vertexData);
    }
    m_PendingMeshBufferData.clear();
}

void Scene::ConfirmMeshBufferData() noexcept
{
    m_PipelinedRecording.Check("Scene::ConfirmMeshBufferData");
    for (const PendingMeshBufferData& entry : m_UnconfirmedMeshBufferData)
        FreeMeshCpuData(*entry.buffers, entry.indexData, entry.vertexData);
    m_UnconfirmedMeshBufferData.clear();
}

GpuScene* Scene::EnsureGpuScene() noexcept
{
    m_PipelinedRecording.Check("Scene::EnsureGpuScene");
    if (m_GpuScene || !m_Device)
        return m_GpuScene;
    GpuSceneDesc desc;
    desc.geometryAndMaterialBuffers = m_EnableBindlessResources;
    desc.resourceDescriptorHeapBindless = m_UseResourceDescriptorHeapBindless;
    // On DX11, a buffer cannot be both structured and vertex.
    // On other APIs, a structured instance buffer can be used for rasterization.
    desc.structuredInstanceBuffer = m_Device->getGraphicsAPI() != nvrhi::GraphicsAPI::D3D11;
    desc.rowGranularity = 1024;
    desc.uploadMode = m_GpuSceneUploadMode;
    try
    {
        m_GpuScene = MAKE_RC_OBJ_PTR(GpuScene, m_Device, desc);
    }
    catch (const std::bad_alloc&)
    {
        m_GpuScene = nullptr;
    }
    return m_GpuScene;
}

void Scene::BeginPipelinedRecording() noexcept
{
    m_PipelinedRecording.Begin();
    if (m_GpuScene)
        m_GpuScene->BeginPipelinedRecording();
}

void Scene::EndPipelinedRecording() noexcept
{
    if (m_GpuScene)
        m_GpuScene->EndPipelinedRecording();
    m_PipelinedRecording.End();
}

void Scene::BindMeshInstanceBuffers()
{
    m_PipelinedRecording.Check("Scene::BindMeshInstanceBuffers");
    nvrhi::IBuffer* instanceBuffer = m_GpuScene ? m_GpuScene->GetInstanceBuffer() : nullptr;
    for (const auto& mesh : m_SceneGraph->GetMeshes())
    {
        if (mesh->buffers && mesh->buffers->instanceBuffer.Get() != instanceBuffer)
            mesh->buffers->instanceBuffer = instanceBuffer;
    }
}

GeometryData* Scene::GetGeometryData(const MeshGeometry& geometry) const
{
    if (!m_GpuScene || geometry.globalGeometryIndex < 0)
        return nullptr;

    return m_GpuScene->GetGeometryRow(uint32_t(geometry.globalGeometryIndex));
}
