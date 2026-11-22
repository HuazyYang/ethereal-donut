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

// The render service: extraction into the frame snapshot (RenderExtract.h; aggregate docs/ecs-design.md 7.1, 7.4, 7.5,
// D8; plan step E5, sub-plan item E5.1).
//
// Extract runs the scene graph's RenderExtract phase, in which the service's systems fill the snapshot. They form one
// chain (Views, Lights, Cameras, SkinnedInstances, Instances; runAfter), all MainThread systems, so that only one of
// them touches the frame arena at a time: the arena is not thread-safe (open point 2). Each system sizes its arrays
// from an upper bound and allocates them serially; Instances then fills its arrays in parallel by chunk at offsets
// from a prefix sum over the chunks' counts, so that the rows are in chunk order at any worker count. The systems
// write no world component (only arena data) and declare only reads.
//
// Instances runs the per-view visibility and the draw packets of E5.2 (RenderVisibility.h): a chunk test, a row test
// and the legacy strategies' per-geometry rules, each view's packets of each pass radix-sorted by key. The service
// shadows (cascade views, LightShadowTable requests) are E5.6's, the skinning pass E5.5's; AddUploadPasses is a thin
// call into E5.4's GpuSceneUploadPass.

#include <donut/render/RenderExtract.h>
#include <donut/render/GpuSceneUploadPass.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/render/SkinningPass.h>
#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/Scene.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>

#include "RenderVisibility.h"

using namespace donut::math;
#include <donut/shaders/light_cb.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <new>
#include <thread>
#include <vector>

namespace donut::render
{
    using namespace donut::engine;

    namespace
    {
        // The RenderExtract systems, in their order (each runs after the one before it).
        enum ExtractSystem : uint32_t
        {
            SystemViews,
            SystemLights,
            SystemCameras,
            SystemSkinnedInstances,
            SystemInstances,
            SystemCount
        };

        const char* const c_SystemNames[SystemCount] = {
            "RenderService.Views",
            "RenderService.Lights",
            "RenderService.Cameras",
            "RenderService.SkinnedInstances",
            "RenderService.Instances",
        };

        // Chunks per task of the parallel passes: about four tasks per thread. The output does not depend on it.
        uint32_t ChunksPerTask(IJobSystem* jobs, uint32_t chunkCount) noexcept
        {
            uint32_t threads = jobs ? jobs->GetThreadCount() : 1;
            const uint32_t hardware = std::thread::hardware_concurrency();
            if (hardware != 0)
                threads = std::min(threads, hardware);
            return std::max(1u, chunkCount / (std::max(1u, threads) * 4));
        }
    }

    struct RenderService::Impl
    {
        using InstanceQuery = visibility::InstanceQuery;
        using SkinnedQuery = Query<With<SkinnedMeshInstanceLeaf>, Read<LeafObject>>;

        nvrhi::AutoPtr<SceneGraph> graph;   // claimed (SceneGraph::ClaimRenderService) while set
        nvrhi::AutoPtr<Scene> scene;
        InstanceQuery instanceQuery;
        SkinnedQuery skinnedQuery;

        // The graph's scheduler generation the systems are registered on (SceneGraph::GetSchedulerGeneration).
        // SceneGraph::SetJobSystem replaces the scheduler, and the systems die with the old one; Extract registers them
        // again on the new one. No reference to a scheduler is held, so the old one and its job system go at once.
        uint64_t schedulerGeneration = 0;
        bool registered = false;
        SystemId systems[SystemCount] = { SystemId::Invalid, SystemId::Invalid, SystemId::Invalid, SystemId::Invalid,
                                          SystemId::Invalid };

        // The frame being extracted, set for the RenderExtract phase only (a phase run elsewhere finds no snapshot).
        IRenderGraph* renderGraph = nullptr;
        const ExtractDesc* desc = nullptr;
        RenderSnapshot* snapshot = nullptr;
        ExtractedView* views = nullptr;
        nvrhi::FRESULT failure = nvrhi::FS_OK;
        uint64_t bytes = 0;

        // Between a successful Extract and its EndFrame.
        bool frameOpen = false;
        // The graph's render-read scope (a flag per service, not a count; the graph counts the scopes).
        bool scopeOpen = false;
        // The leaves' own shadow maps the snapshot names (ExtractedLight::shadowMap), held until EndFrame: the
        // application may replace Light::shadowMap while the frame records. Capacity kept.
        std::vector<nvrhi::AutoPtr<IShadowMap>> frameShadowMaps;
        // The first-frame skinned instances whose first dispatch this frame's skinning passes consumed
        // (ConsumeFirstFrame, from AddSkinningPasses, E5.5); EndFrame(true) sets their skinningInitialized. Held, so
        // that a write during the frame that takes one out of the graph leaves a valid leaf. Capacity kept.
        std::vector<nvrhi::AutoPtr<SkinnedMeshInstance>> consumedFirstFrames;
        // AddUploadPasses ran for the open frame (a second call adds nothing).
        bool uploadAdded = false;
        // AddSkinningPasses ran for the open frame (a second call adds nothing).
        bool skinningAdded = false;
        // AddSkinningPasses: the snapshot's skinned leaves, sorted, to find the graph's other skinned instances.
        // Capacity kept.
        std::vector<const SkinnedMeshInstance*> selectedSkinned;
        // The skinning pass (E5.5), made at the first AddSkinningPasses.
        nvrhi::AutoPtr<SkinningPass> skinningPass;
        // The skinned leaves this frame's snapshot selected, held until EndFrame (E5.5 review fix 1): when the frame
        // does not execute they become pending. Capacity kept.
        std::vector<nvrhi::AutoPtr<SkinnedMeshInstance>> frameSkinned;
        // Leaves a frame that did not execute selected: selected again until two executed frames have selected them
        // (the dispatch, then the settle of the previous positions that the legacy rule's second frame gives), so that
        // dropped frames and frame-index gaps lose no pose. Ordered by address (the SkinnedInstances system looks them
        // up). Capacity kept.
        struct PendingSkin
        {
            nvrhi::AutoPtr<SkinnedMeshInstance> leaf;
            uint32_t remaining = 0;
        };
        std::vector<PendingSkin> pendingSkins;
        // Visibility and draw packets (E5.2): the scratch of the Instances system, capacity kept.
        visibility::InstanceVisibility instanceVisibility;

        LightShadowTable shadows;
        RenderServiceStats stats;
        // The GpuScene upload (E5.4), made at the first AddUploadPasses.
        nvrhi::AutoPtr<GpuSceneUploadPass> uploadPass;

        Impl() noexcept = default;
        ~Impl()
        {
            EndFrame(false);
            UnregisterSystems();
            if (graph)
                graph->ReleaseRenderService();
        }

        Impl(const Impl&) = delete;
        Impl& operator=(const Impl&) = delete;

        // ------------------------------------------------------------------------------------ frame arena

        template <typename T>
        T* Allocate(uint32_t count) noexcept
        {
            static_assert(std::is_trivially_destructible_v<T>, "the frame arena runs no destructors");
            if (count == 0)
                return nullptr;
            void* memory = renderGraph->AllocateFrameData(sizeof(T) * size_t(count), alignof(T));
            if (!memory)
            {
                failure = nvrhi::FE_OUT_OF_MEMORY;
                return nullptr;
            }
            bytes += uint64_t(sizeof(T)) * count;
            return static_cast<T*>(memory);
        }

        // Value-initialised elements (the snapshot structs' member initialisers).
        template <typename T>
        T* AllocateConstructed(uint32_t count) noexcept
        {
            T* items = Allocate<T>(count);
            if (items)
            {
                for (uint32_t i = 0; i < count; ++i)
                    new (items + i) T();
            }
            return items;
        }

        // ---------------------------------------------------------------------------------- set-up

        nvrhi::FRESULT ClaimGraphOf(const RenderServiceDesc& serviceDesc) noexcept
        {
            if (graph)
                return nvrhi::FS_OK;

            SceneGraph* target = serviceDesc.sceneGraph;
            if (!target && serviceDesc.scene)
                target = serviceDesc.scene->GetSceneGraph();
            if (!target)
            {
                log::warning("RenderService: no scene graph (RenderServiceDesc::scene or sceneGraph)");
                return nvrhi::FE_INVALID_ARGS;
            }
            if (serviceDesc.scene && serviceDesc.scene->GetSceneGraph() != target)
            {
                log::warning("RenderService: RenderServiceDesc::sceneGraph is not the scene's graph");
                return nvrhi::FE_INVALID_ARGS;
            }
            if (serviceDesc.scene && !serviceDesc.scene->IsOnGraphPath())
            {
                log::warning("RenderService: the scene is not on the render-graph path (call "
                             "Scene::SetGpuSceneUploadMode(GpuSceneUploadMode::Scatter) before it loads)");
                return nvrhi::FE_INVALID_ARGS;
            }

            if (!target->ClaimRenderService())
            {
                log::warning("RenderService: another render service is registered on this scene graph (one per graph)");
                return nvrhi::FE_INVALID_ARGS;
            }

            IWorld* world = target->GetWorld();
            nvrhi::FRESULT result = instanceQuery.Create(world, "RenderService.Instances");
            if (NVRHI_SUCCEEDED(result))
                result = skinnedQuery.Create(world, "RenderService.SkinnedInstances");
            if (NVRHI_FAILED(result))
            {
                instanceQuery = InstanceQuery();
                skinnedQuery = SkinnedQuery();
                target->ReleaseRenderService();
                return result;
            }
            graph = target;
            scene = serviceDesc.scene;
            return nvrhi::FS_OK;
        }

        // Removes the systems from the graph's scheduler while it is the one they were added to; an older one took
        // them along when it went.
        void UnregisterSystems() noexcept
        {
            IScheduler* current = graph && registered && graph->GetSchedulerGeneration() == schedulerGeneration
                                      ? graph->GetScheduler() : nullptr;
            for (SystemId& id : systems)
            {
                if (current && id != SystemId::Invalid)
                    current->RemoveSystem(id);
                id = SystemId::Invalid;
            }
            registered = false;
        }

        template <typename F>
        nvrhi::FRESULT AddSystem(IScheduler* target, uint32_t index, const ComponentAccess* accesses,
                                 uint32_t accessCount, F&& update) noexcept
        {
            SystemDesc systemDesc{};
            systemDesc.name = c_SystemNames[index];
            systemDesc.accesses = accessCount != 0 ? accesses : nullptr;
            systemDesc.accessCount = accessCount;
            systemDesc.phase = Phase::RenderExtract;
            if (index != 0)
            {
                systemDesc.runAfter = &c_SystemNames[index - 1];
                systemDesc.runAfterCount = 1;
            }
            // On the thread that runs the phase: the arena is not thread-safe, and Instances splits its own work.
            systemDesc.flags = SystemFlags::MainThread;

            nvrhi::AutoPtr<ISystem> system;
            nvrhi::FRESULT result = CreateLambdaSystem(systemDesc, std::forward<F>(update), &system);
            if (NVRHI_SUCCEEDED(result))
                result = target->AddSystem(system, systems[index]);
            return result;
        }

        nvrhi::FRESULT RegisterSystems() noexcept
        {
            IScheduler* current = graph->GetScheduler();
            if (!current)
            {
                log::warning("RenderService: the scene graph has no scheduler");
                return nvrhi::FE_INVALID_ARGS;
            }
            const uint64_t generation = graph->GetSchedulerGeneration();
            if (registered && generation == schedulerGeneration)
                return nvrhi::FS_OK;

            UnregisterSystems();
            IWorld* world = graph->GetWorld();
            const ComponentId leafObject = RegisterComponent<LeafObject>(world);
            const ComponentId worldTransform = RegisterComponent<WorldTransform>(world);
            const ComponentId instanceSlot = RegisterComponent<GpuInstanceSlot>(world);
            const ComponentId meshRef = RegisterComponent<MeshRef>(world);
            const ComponentId worldBounds = RegisterComponent<WorldBounds>(world);
            const ComponentId renderFlags = RegisterComponent<RenderFlags>(world);
            const ComponentId chunkBounds = RegisterComponent<ChunkWorldBounds>(world);
            const ComponentId sceneLinks = RegisterComponent<SceneLinks>(world);
            const ComponentId sceneNode = RegisterComponent<SceneNode>(world);
            const ComponentId localBounds = RegisterComponent<LocalBounds>(world);
            if (leafObject == ComponentId::Invalid || worldTransform == ComponentId::Invalid ||
                instanceSlot == ComponentId::Invalid || meshRef == ComponentId::Invalid ||
                worldBounds == ComponentId::Invalid || renderFlags == ComponentId::Invalid ||
                chunkBounds == ComponentId::Invalid || sceneLinks == ComponentId::Invalid ||
                sceneNode == ComponentId::Invalid || localBounds == ComponentId::Invalid)
                return nvrhi::FE_INVALID_ARGS;

            // Lights, cameras and skinned instances read their leaves and, through the node getters, the world
            // transforms; Instances reads its query's columns, and through SceneGraphNode::GetGlobalBoundingBox (the
            // legacy transparent distance of a node with children) the subgraph's links, leaves and transforms.
            // Nothing writes: no ambiguity with any other system.
            const ComponentAccess leafReads[] = {
                { leafObject, engine::Access::Read, {} },
                { worldTransform, engine::Access::Read, {} },
            };
            const ComponentAccess instanceReads[] = {
                { instanceSlot, engine::Access::Read, {} },
                { meshRef, engine::Access::Read, {} },
                { renderFlags, engine::Access::Read, {} },
                { worldBounds, engine::Access::Read, {} },
                { worldTransform, engine::Access::Read, {} },
                { sceneLinks, engine::Access::Read, {} },
                { sceneNode, engine::Access::Read, {} },
                { leafObject, engine::Access::Read, {} },
                { localBounds, engine::Access::Read, {} },   // the Debug leaf check
                { chunkBounds, engine::Access::Read, {} },
            };

            registered = true;   // so that a failure below removes what was added
            schedulerGeneration = generation;
            Impl* self = this;
            nvrhi::FRESULT result = AddSystem(current, SystemViews, nullptr, 0,
                [self](ISystemContext* context) noexcept { return self->RunViews(context); });
            if (NVRHI_SUCCEEDED(result))
                result = AddSystem(current, SystemLights, leafReads, uint32_t(std::size(leafReads)),
                    [self](ISystemContext* context) noexcept { return self->RunLights(context); });
            if (NVRHI_SUCCEEDED(result))
                result = AddSystem(current, SystemCameras, leafReads, uint32_t(std::size(leafReads)),
                    [self](ISystemContext* context) noexcept { return self->RunCameras(context); });
            if (NVRHI_SUCCEEDED(result))
                result = AddSystem(current, SystemSkinnedInstances, leafReads, uint32_t(std::size(leafReads)),
                    [self](ISystemContext* context) noexcept { return self->RunSkinnedInstances(context); });
            if (NVRHI_SUCCEEDED(result))
                result = AddSystem(current, SystemInstances, instanceReads, uint32_t(std::size(instanceReads)),
                    [self](ISystemContext* context) noexcept { return self->RunInstances(context); });
            if (NVRHI_FAILED(result))
                UnregisterSystems();
            return result;
        }

        // The graph path's GpuScene, before the phase (RenderExtract.h): mesh buffer objects, the update, the buffers,
        // the meshes' instance buffer.
        nvrhi::FRESULT UpdateGpuScene(nvrhi::IDevice* device, bool& outReplaced) noexcept
        {
            outReplaced = false;
            GpuScene* gpuScene = scene->EnsureGpuScene();
            if (!gpuScene)
            {
                log::warning("RenderService: the scene has no GpuScene");
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            // Without a device CreateBuffers does the bookkeeping only (sizes, GetBufferChanges).
            nvrhi::FRESULT result = scene->CreateMeshBufferObjects();
            if (NVRHI_SUCCEEDED(result))
                result = gpuScene->Update(graph);
            if (NVRHI_SUCCEEDED(result))
                result = gpuScene->CreateBuffers();
            if (NVRHI_FAILED(result))
            {
                log::warning("RenderService: the GpuScene was not updated");
                return result;
            }
            if (device)
            {
                const GpuSceneBufferChanges& changes = gpuScene->GetBufferChanges();
                outReplaced = changes.instancesResized || changes.geometriesResized || changes.materialsResized;
            }
            scene->BindMeshInstanceBuffers();
            return nvrhi::FS_OK;
        }

        // ------------------------------------------------------------------------------- the systems

        // An exhausted arena: Extract returns the failure. The system itself returns FS_OK, so that the scheduler's
        // diagnostics do not report it a second time; the systems after it skip their work.
        nvrhi::FRESULT Fail() noexcept
        {
            if (NVRHI_SUCCEEDED(failure))
                failure = nvrhi::FE_OUT_OF_MEMORY;
            return nvrhi::FS_OK;
        }

        nvrhi::FRESULT RunViews(ISystemContext*) noexcept
        {
            if (!snapshot || NVRHI_FAILED(failure))
                return nvrhi::FS_OK;   // the phase ran outside Extract
            const uint32_t count = desc->viewCount;
            ExtractedView* out = AllocateConstructed<ExtractedView>(count);
            if (count != 0 && !out)
                return Fail();
            for (uint32_t i = 0; i < count; ++i)
            {
                const IView* view = desc->views[i];
                out[i].view = view;
                out[i].frustum = view->GetViewFrustum();
                out[i].viewOrigin = view->GetViewOrigin();
            }
            views = out;
            snapshot->views = out;
            snapshot->viewCount = count;
            snapshot->descViewCount = count;
            return nvrhi::FS_OK;
        }

        // GetLights() order: the lighting shaders sum lights in array order. One FillLightConstants per light (D16),
        // into zeroed constants, as the snapshot is compared byte for byte.
        nvrhi::FRESULT RunLights(ISystemContext*) noexcept
        {
            if (!snapshot || NVRHI_FAILED(failure))
                return nvrhi::FS_OK;
            const auto& lights = graph->GetLights();
            const uint32_t count = uint32_t(lights.size());
            ExtractedLight* out = AllocateConstructed<ExtractedLight>(count);
            LightConstants* constants = Allocate<LightConstants>(count);
            if (count != 0 && (!out || !constants))
                return Fail();
            for (uint32_t i = 0; i < count; ++i)
            {
                const Light* light = lights[i].Get();
                std::memset(&constants[i], 0, sizeof(LightConstants));
                light->FillLightConstants(constants[i]);
                ExtractedLight& extracted = out[i];
                extracted.constants = &constants[i];
                extracted.entity = graph->GetEntity(light->GetNode());
                extracted.leaf = light;
                // Without service shadows (E5.6) the light keeps its leaf's own map, which the application renders;
                // held until EndFrame, so that the application may replace Light::shadowMap meanwhile.
                extracted.shadowMap = light->shadowMap.Get();
                if (light->shadowMap)
                {
                    try
                    {
                        frameShadowMaps.push_back(light->shadowMap);
                    }
                    catch (...)
                    {
                        return Fail();
                    }
                }
                extracted.shadowTexture = InvalidResourceId;
                extracted.serviceShadow = false;
            }
            snapshot->lights = out;
            snapshot->lightConstants = constants;
            snapshot->lightCount = count;
            return nvrhi::FS_OK;
        }

        // GetCameras() order. The render origin is zero in E5 (open point 10), so viewToWorld is the leaf's.
        nvrhi::FRESULT RunCameras(ISystemContext*) noexcept
        {
            if (!snapshot || NVRHI_FAILED(failure))
                return nvrhi::FS_OK;
            const auto& cameras = graph->GetCameras();
            const uint32_t count = uint32_t(cameras.size());
            ExtractedCamera* out = AllocateConstructed<ExtractedCamera>(count);
            if (count != 0 && !out)
                return Fail();
            for (uint32_t i = 0; i < count; ++i)
            {
                const SceneCamera* camera = cameras[i].Get();
                out[i].entity = graph->GetEntity(camera->GetNode());
                out[i].leaf = camera;
                out[i].viewToWorld = camera->GetViewToWorldMatrix();
            }
            snapshot->cameras = out;
            snapshot->cameraCount = count;
            return nvrhi::FS_OK;
        }

        // Skinned instances whose joints moved this frame or the previous one (the legacy rule of
        // Scene::UpdateSkinnedMeshes), never skinned yet, or pending from a frame that did not execute (E5.5 review fix
        // 1), in chunk order, with their joint matrices by the legacy formula:
        // inverseBindMatrix * affineToHomogeneous(affine3(jointWorld * inverse(instanceWorld))), doubles until the
        // conversion. The first-frame flag is the leaf's skinningInitialized at extraction; EndFrame(true) sets it for
        // the instances a skinning dispatch consumed (ConsumeFirstFrame).
        nvrhi::FRESULT RunSkinnedInstances(ISystemContext* context) noexcept
        {
            if (!snapshot || NVRHI_FAILED(failure))
                return nvrhi::FS_OK;
            const uint32_t frameIndex = desc->frameIndex;
            auto selected = [this, frameIndex](const SkinnedMeshInstance* instance) noexcept
            {
                return instance && (!(instance->GetLastUpdateFrameIndex() + 1 < frameIndex) ||
                                    !instance->skinningInitialized || IsPendingSkin(instance));
            };

            uint32_t count = 0;
            uint64_t jointCount = 0;
            skinnedQuery.ForEachChunk(context, [&](const ChunkView& chunk, const LeafObject* leaves)
            {
                for (uint32_t row = 0; row < chunk.count; ++row)
                {
                    const auto* instance = static_cast<const SkinnedMeshInstance*>(leaves[row].object);
                    if (selected(instance))
                    {
                        ++count;
                        jointCount += instance->joints.size();
                    }
                }
            });
            if (jointCount > UINT32_MAX)
                return Fail();

            ExtractedSkinnedInstance* out = AllocateConstructed<ExtractedSkinnedInstance>(count);
            dm::float4x4* matrices = Allocate<dm::float4x4>(uint32_t(jointCount));
            if ((count != 0 && !out) || (jointCount != 0 && !matrices))
                return Fail();

            uint32_t next = 0;
            uint32_t nextJoint = 0;
            skinnedQuery.ForEachChunk(context, [&](const ChunkView& chunk, const LeafObject* leaves)
            {
                for (uint32_t row = 0; row < chunk.count; ++row)
                {
                    const auto* instance = static_cast<const SkinnedMeshInstance*>(leaves[row].object);
                    if (!selected(instance))
                        continue;
                    ExtractedSkinnedInstance& extracted = out[next++];
                    const uint32_t joints = uint32_t(instance->joints.size());
                    extracted.entity = chunk.entities[row];
                    extracted.leaf = instance;
                    extracted.jointMatrices = SnapshotRange{ nextJoint, joints };
                    extracted.firstFrame = !instance->skinningInitialized;

                    const SceneGraphNode* node = instance->GetNode();
                    const dm::daffine3 worldToRoot =
                        node ? dm::inverse(node->GetLocalToWorldTransform()) : dm::daffine3::identity();
                    for (uint32_t j = 0; j < joints; ++j)
                    {
                        const SkinnedMeshJoint& joint = instance->joints[j];
                        const nvrhi::AutoPtr<SceneGraphNode> jointNode = joint.node.Lock();
                        if (jointNode)
                        {
                            const dm::daffine3 jointToRoot = jointNode->GetLocalToWorldTransform() * worldToRoot;
                            const dm::float4x4 jointMatrix = dm::affineToHomogeneous(dm::affine3(jointToRoot));
                            matrices[nextJoint + j] = joint.inverseBindMatrix * jointMatrix;
                        }
                        else
                        {
                            // a joint node that is gone (the legacy path would dereference null)
                            matrices[nextJoint + j] = dm::float4x4::identity();
                        }
                    }
                    nextJoint += joints;
                }
            });

            snapshot->skinnedInstances = out;
            snapshot->skinnedInstanceCount = count;
            snapshot->jointMatrices = matrices;
            snapshot->jointMatrixCount = uint32_t(jointCount);
            return nvrhi::FS_OK;
        }

        // Visibility per view and the draw packets (E5.2, RenderVisibility.h): each view's visible instances in chunk
        // order, its packets of each pass sorted by key; the arrays in the arena, allocated serially.
        nvrhi::FRESULT RunInstances(ISystemContext* context) noexcept
        {
            if (!snapshot || NVRHI_FAILED(failure))
                return nvrhi::FS_OK;
            visibility::VisibilityFrame frame;
            frame.context = context;
            frame.graph = graph;
            frame.renderGraph = renderGraph;
            frame.views = views;
            frame.viewCount = snapshot->viewCount;
            frame.drawDoubleSidedMaterialsSeparately = desc->drawDoubleSidedMaterialsSeparately;
            if (NVRHI_FAILED(instanceVisibility.Run(instanceQuery, frame, *snapshot, bytes)))
                return Fail();
            return nvrhi::FS_OK;
        }

        // ------------------------------------------------------------------------------- frame end

        void CloseScope() noexcept
        {
            if (scopeOpen)
            {
                graph->EndRenderRead();
                scopeOpen = false;
            }
        }

        // The contract for AddSkinningPasses (E5.5): for each ExtractedSkinnedInstance with firstFrame set whose
        // dispatch it added (with SkinningFlag_FirstFrame), it calls this once; EndFrame(true) then sets the leaf's
        // skinningInitialized, and a frame that does not execute leaves it for the next frame's first-frame dispatch.
        // Instances nobody dispatched keep the flag. False when out of memory (the flag then stays unset).
        bool ConsumeFirstFrame(const SkinnedMeshInstance* instance) noexcept
        {
            try
            {
                consumedFirstFrames.push_back(const_cast<SkinnedMeshInstance*>(instance));
            }
            catch (...)
            {
                return false;
            }
            return true;
        }

        // The frame's references (the leaves' shadow maps, the consumed first frames, the selected skinned leaves),
        // capacity kept.
        void ReleaseFrameReferences() noexcept
        {
            frameShadowMaps.clear();
            consumedFirstFrames.clear();
            frameSkinned.clear();
        }

        static bool PendingLess(const PendingSkin& entry, const SkinnedMeshInstance* leaf) noexcept
        {
            return entry.leaf.Get() < leaf;
        }

        bool IsPendingSkin(const SkinnedMeshInstance* leaf) const noexcept
        {
            const auto it = std::lower_bound(pendingSkins.begin(), pendingSkins.end(), leaf, PendingLess);
            return it != pendingSkins.end() && it->leaf.Get() == leaf;
        }

        // An executed frame counts one of the two selections of every pending leaf; one that did not execute makes
        // its selected leaves pending for two more.
        void UpdatePendingSkins(bool executed) noexcept
        {
            if (executed)
            {
                for (PendingSkin& entry : pendingSkins)
                    entry.remaining = entry.remaining ? entry.remaining - 1 : 0;
                pendingSkins.erase(std::remove_if(pendingSkins.begin(), pendingSkins.end(),
                                                  [](const PendingSkin& entry) { return entry.remaining == 0; }),
                                   pendingSkins.end());
                return;
            }
            for (const auto& leaf : frameSkinned)
            {
                const auto it = std::lower_bound(pendingSkins.begin(), pendingSkins.end(), leaf.Get(), PendingLess);
                if (it != pendingSkins.end() && it->leaf.Get() == leaf.Get())
                {
                    it->remaining = 2;
                    continue;
                }
                try
                {
                    pendingSkins.insert(it, PendingSkin{ leaf, 2 });
                }
                catch (...)
                {
                    log::warning("RenderService: out of memory; a skinned instance of a dropped frame may keep its "
                                 "pose");
                }
            }
        }

        // Idempotent: the scope closes once, and the first-frame flags are set at most once, for an executed frame.
        void EndFrame(bool executed) noexcept
        {
            CloseScope();
            if (frameOpen && executed)
            {
                for (const auto& leaf : consumedFirstFrames)
                    leaf->skinningInitialized = true;
            }
            if (frameOpen)
                UpdatePendingSkins(executed);
            frameOpen = false;
            uploadAdded = false;
            skinningAdded = false;
            ReleaseFrameReferences();
        }
    };

    RenderService::RenderService(nvrhi::IDevice* device, const RenderServiceDesc& desc) noexcept
        : m_Device(device)
        , m_Desc(desc)
    {
    }

    RenderService::~RenderService() = default;

    nvrhi::FRESULT RenderService::Extract(IRenderGraph* graph, FrameData* frameData, const ExtractDesc& desc,
                                          SceneFrame** outFrame) noexcept
    {
        if (outFrame)
            *outFrame = nullptr;
        if (!m_Impl)
        {
            m_Impl.reset(new (std::nothrow) Impl());
            if (!m_Impl)
                return nvrhi::FE_OUT_OF_MEMORY;
        }
        Impl& s = *m_Impl;
        // A scope still open from the previous frame (EndFrame not called) closes; that frame counts as not executed.
        s.EndFrame(false);

        if (!graph || (desc.viewCount != 0 && !desc.views))
            return nvrhi::FE_INVALID_ARGS;
        for (uint32_t v = 0; v < desc.viewCount; ++v)
        {
            if (!desc.views[v])
                return nvrhi::FE_INVALID_ARGS;
        }

        nvrhi::FRESULT result = s.ClaimGraphOf(m_Desc);
        if (NVRHI_FAILED(result))
            return result;
        // A service serves the graph it claimed: when the scene's graph was replaced (Scene::CreateSceneGraph, a new
        // load), Extract refuses; the application makes a new service for the new graph.
        if (s.scene && s.scene->GetSceneGraph() != s.graph.Get())
        {
            log::warning("RenderService: the scene's graph is not the one this service was made for; make a new "
                         "service");
            return nvrhi::FE_INVALID_ARGS;
        }
        result = s.RegisterSystems();
        if (NVRHI_FAILED(result))
            return result;

        bool sceneBuffersReplaced = false;
        if (s.scene)
        {
            result = s.UpdateGpuScene(m_Device, sceneBuffersReplaced);
            if (NVRHI_FAILED(result))
                return result;
        }

        RenderSnapshot* snapshot = NewFrameData<RenderSnapshot>(graph);
        if (!snapshot)
            return nvrhi::FE_OUT_OF_MEMORY;
        snapshot->graphFrameIndex = graph->GetFrameIndex();
        snapshot->frameIndex = desc.frameIndex;
        snapshot->worldVersion = uint32_t(s.graph->GetWorld()->GetVersion());
        snapshot->renderOrigin = dm::double3(0.0);   // open point 10

        // The phase: the systems fill the snapshot (they find the frame through these members).
        s.renderGraph = graph;
        s.desc = &desc;
        s.snapshot = snapshot;
        s.views = nullptr;
        s.failure = nvrhi::FS_OK;
        s.bytes = sizeof(RenderSnapshot);
        s.graph->BeginRenderRead();
        s.scopeOpen = true;
        // Through the graph, which accounts for the read-only phase: the next Refresh may still skip.
        result = s.graph->RunReadOnlyPhase(Phase::RenderExtract);
        if (NVRHI_SUCCEEDED(result))
            result = s.failure;
        s.renderGraph = nullptr;
        s.desc = nullptr;
        s.snapshot = nullptr;
        s.views = nullptr;
        if (NVRHI_FAILED(result))
        {
            s.CloseScope();
            s.ReleaseFrameReferences();
            return result;
        }
        // The selected skinned leaves, held for the frame: EndFrame(false) makes them pending (E5.5 review fix 1).
        try
        {
            for (uint32_t i = 0; i < snapshot->skinnedInstanceCount; ++i)
                s.frameSkinned.push_back(const_cast<SkinnedMeshInstance*>(snapshot->skinnedInstances[i].leaf));
        }
        catch (...)
        {
            s.CloseScope();
            s.ReleaseFrameReferences();
            return nvrhi::FE_OUT_OF_MEMORY;
        }

        // The blackboard's entry: this frame's (a second Extract in one frame overwrites it) or a new one.
        SceneFrame* frame = frameData ? frameData->TryCreate<SceneFrame>(graph) : NewFrameData<SceneFrame>(graph);
        if (!frame)
        {
            s.CloseScope();
            s.ReleaseFrameReferences();
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        *frame = SceneFrame{};
        frame->snapshot = snapshot;
        frame->sceneBuffersReplaced = sceneBuffersReplaced;
        s.frameOpen = true;

        RenderServiceStats& stats = s.stats;
        stats = RenderServiceStats{};
        stats.views = snapshot->descViewCount;
        stats.cascadeViews = snapshot->viewCount - snapshot->descViewCount;
        stats.visibleInstances = snapshot->instances.count;
        for (uint32_t v = 0; v < snapshot->viewCount; ++v)
        {
            for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
                stats.packets[p] += snapshot->views[v].packets[p].count;
        }
        stats.lights = snapshot->lightCount;
        stats.cameras = snapshot->cameraCount;
        stats.skinnedInstances = snapshot->skinnedInstanceCount;
        stats.jointMatrices = snapshot->jointMatrixCount;
        stats.serviceShadowMaps = s.shadows.GetCount();
        stats.snapshotBytes = s.bytes;

        if (outFrame)
            *outFrame = frame;
        return nvrhi::FS_OK;
    }

    namespace
    {
        // SceneFrame::bufferReads: every valid scene buffer id of `frame` in its latest version, in the arena.
        nvrhi::FRESULT RebuildBufferReads(IRenderGraph* graph, SceneFrame* frame) noexcept
        {
            const ResourceId scene[] = { frame->instanceBuffer, frame->geometryBuffer, frame->materialBuffer };
            uint32_t count = 0;
            for (ResourceId id : scene)
                count += id != InvalidResourceId ? 1u : 0u;
            for (uint32_t i = 0; i < frame->skinnedVertexBufferCount; ++i)
                count += frame->skinnedVertexBuffers[i] != InvalidResourceId ? 1u : 0u;
            frame->bufferReads = nullptr;
            frame->bufferReadCount = 0;
            if (count == 0)
                return nvrhi::FS_OK;
            auto* reads =
                static_cast<ResourceId*>(graph->AllocateFrameData(sizeof(ResourceId) * count, alignof(ResourceId)));
            if (!reads)
                return nvrhi::FE_OUT_OF_MEMORY;
            uint32_t at = 0;
            for (ResourceId id : scene)
            {
                if (id != InvalidResourceId)
                    reads[at++] = id;
            }
            for (uint32_t i = 0; i < frame->skinnedVertexBufferCount; ++i)
            {
                if (frame->skinnedVertexBuffers[i] != InvalidResourceId)
                    reads[at++] = frame->skinnedVertexBuffers[i];
            }
            frame->bufferReads = reads;
            frame->bufferReadCount = count;
            return nvrhi::FS_OK;
        }
    }

    // A thin call into GpuSceneUploadPass (E5.4): the GpuScene's buffers imported in their resting state
    // (engine::c_GpuSceneRestingState), the pass added with this frame's buffer changes, the written versions stored in
    // `frame`. FE_INVALID_ARGS when no frame is open (Extract did not run or failed, or EndFrame came). Nothing for a
    // service without a scene, device or shader factory (extraction only), nor for a second call in the same frame.
    // A scene without an instance buffer yet (empty, or still loading) still gets its asset rows and mesh data: the
    // pass writes them without scattering.
    nvrhi::FRESULT RenderService::AddUploadPasses(IRenderGraph* graph, SceneFrame* frame) noexcept
    {
        if (!graph || !frame)
            return nvrhi::FE_INVALID_ARGS;
        if (!m_Impl || !m_Impl->frameOpen)
            return nvrhi::FE_INVALID_ARGS;
        Impl& s = *m_Impl;
        if (!s.scene || !m_Device || !m_Desc.shaderFactory || s.uploadAdded)
            return nvrhi::FS_OK;
        GpuScene* gpuScene = s.scene->GetGpuScene();
        if (!gpuScene)
            return nvrhi::FE_INVALID_ARGS;
        if (!s.uploadPass)
        {
            try
            {
                s.uploadPass = MAKE_RC_OBJ_PTR(GpuSceneUploadPass, m_Device, m_Desc.shaderFactory);
            }
            catch (...)
            {
            }
            if (!s.uploadPass)
                return nvrhi::FE_OUT_OF_MEMORY;
        }

        constexpr nvrhi::ResourceStates state = engine::c_GpuSceneRestingState;
        GpuSceneUploadGraphInputs inputs;
        inputs.gpuScene = gpuScene;
        inputs.scene = s.scene;
        inputs.bufferChanges = &gpuScene->GetBufferChanges();
        if (nvrhi::IBuffer* buffer = gpuScene->GetInstanceBuffer())
        {
            inputs.instanceBuffer = graph->ImportBuffer(buffer, state, state);
            if (inputs.instanceBuffer == InvalidResourceId)
                return nvrhi::FE_OUT_OF_MEMORY;
        }
        if (nvrhi::IBuffer* buffer = gpuScene->GetGeometryBuffer())
        {
            inputs.geometryBuffer = graph->ImportBuffer(buffer, state, state);
            if (inputs.geometryBuffer == InvalidResourceId)
                return nvrhi::FE_OUT_OF_MEMORY;
        }
        if (nvrhi::IBuffer* buffer = gpuScene->GetMaterialBuffer())
        {
            inputs.materialBuffer = graph->ImportBuffer(buffer, state, state);
            if (inputs.materialBuffer == InvalidResourceId)
                return nvrhi::FE_OUT_OF_MEMORY;
        }
        s.uploadAdded = true;

        GpuSceneUploadGraphOutputs outputs;
        try
        {
            outputs = s.uploadPass->AddToGraph(graph, inputs);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        frame->instanceBuffer = outputs.instanceBuffer;
        frame->geometryBuffer = outputs.geometryBuffer;
        frame->materialBuffer = outputs.materialBuffer;
        return RebuildBufferReads(graph, frame);
    }

    // Skinning (E5.5). Every skinned instance's vertex buffer is imported in its resting state
    // (engine::c_SkinnedVertexBufferRestingState): first the snapshot's skinned instances, in snapshot order, which the
    // SkinningPass writes; then the graph's other skinned instances, in SceneGraph::GetSkinnedMeshInstances() order,
    // only imported, so that SceneFrame::bufferReads names every skinned buffer a geometry pass may draw (they are not
    // host-kept on the graph path). Each first-frame instance the pass dispatches is marked (ConsumeFirstFrame), so
    // that EndFrame(true) sets its skinningInitialized; one it does not dispatch keeps its first frame. FE_INVALID_ARGS
    // when no frame is open. Nothing for a service without a scene, device or shader factory (extraction only), nor for
    // a second call in the same frame.
    nvrhi::FRESULT RenderService::AddSkinningPasses(IRenderGraph* graph, SceneFrame* frame) noexcept
    {
        if (!graph || !frame || !m_Impl || !m_Impl->frameOpen)
            return nvrhi::FE_INVALID_ARGS;
        Impl& s = *m_Impl;
        if (!s.scene || !m_Device || !m_Desc.shaderFactory || s.skinningAdded)
            return nvrhi::FS_OK;
        const RenderSnapshot* snapshot = frame->snapshot;
        if (!snapshot)
            return nvrhi::FE_INVALID_ARGS;
        // The prototypes' vertex data is written by the upload pass's Prepare (E5.5 review fix 3).
        if (!s.uploadAdded)
        {
            log::warning("RenderService::AddSkinningPasses: call AddUploadPasses first (it writes the prototypes' "
                         "data)");
            return nvrhi::FE_INVALID_ARGS;
        }
        s.skinningAdded = true;

        // The snapshot's leaves, sorted, to tell the other skinned instances of the graph apart.
        const uint32_t selected = snapshot->skinnedInstanceCount;
        const auto& skinnedLeaves = s.graph->GetSkinnedMeshInstances();
        try
        {
            s.selectedSkinned.clear();
            for (uint32_t i = 0; i < selected; ++i)
                s.selectedSkinned.push_back(snapshot->skinnedInstances[i].leaf);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        std::sort(s.selectedSkinned.begin(), s.selectedSkinned.end());
        auto isSelected = [&s](const SkinnedMeshInstance* leaf) noexcept
        {
            return std::binary_search(s.selectedSkinned.begin(), s.selectedSkinned.end(), leaf);
        };
        uint32_t others = 0;
        for (const auto& leaf : skinnedLeaves)
            others += isSelected(leaf.Get()) ? 0u : 1u;
        const uint32_t total = selected + others;
        if (total == 0)
        {
            // Nothing to skin; the pass still ages its binding-set cache (E5.5 review fix 4).
            if (s.skinningPass)
            {
                try
                {
                    (void)s.skinningPass->AddToGraph(graph, SkinningGraphInputs{ snapshot, nullptr, 0 });
                }
                catch (...)
                {
                    return nvrhi::FE_OUT_OF_MEMORY;
                }
                s.stats.skinningBindingSets = s.skinningPass->GetCachedBindingSetCount();
            }
            return nvrhi::FS_OK;
        }

        auto* ids = static_cast<ResourceId*>(graph->AllocateFrameData(sizeof(ResourceId) * total, alignof(ResourceId)));
        if (!ids)
            return nvrhi::FE_OUT_OF_MEMORY;
        // A skinned instance's vertex buffer, imported; invalid when it has none yet, or a host-kept one (CPU path).
        bool importFailed = false;
        auto import = [&](const SkinnedMeshInstance* leaf) noexcept
        {
            const MeshInfo* mesh = leaf ? leaf->GetMesh() : nullptr;
            nvrhi::IBuffer* buffer = mesh && mesh->buffers ? mesh->buffers->vertexBuffer.Get() : nullptr;
            if (!buffer || buffer->getDesc().keepInitialState)
                return InvalidResourceId;
            const ResourceId id = graph->ImportBuffer(buffer, c_SkinnedVertexBufferRestingState,
                                                      c_SkinnedVertexBufferRestingState);
            importFailed = importFailed || id == InvalidResourceId;
            return id;
        };
        for (uint32_t i = 0; i < selected; ++i)
            ids[i] = import(snapshot->skinnedInstances[i].leaf);
        uint32_t at = selected;
        for (const auto& leaf : skinnedLeaves)
        {
            if (!isSelected(leaf.Get()))
                ids[at++] = import(leaf.Get());
        }
        if (importFailed)
            return nvrhi::FE_OUT_OF_MEMORY;

        nvrhi::FRESULT result = nvrhi::FS_OK;
        if (selected != 0 || s.skinningPass)   // the pass ages its cache at every call (review fix 4)
        {
            if (!s.skinningPass)
            {
                try
                {
                    s.skinningPass = MAKE_RC_OBJ_PTR(SkinningPass, m_Device, m_Desc.shaderFactory);
                }
                catch (...)
                {
                }
                if (!s.skinningPass)
                    return nvrhi::FE_OUT_OF_MEMORY;
            }
            SkinningGraphOutputs outputs;
            try
            {
                outputs = s.skinningPass->AddToGraph(graph, SkinningGraphInputs{ snapshot, ids, selected });
            }
            catch (...)
            {
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            // A dispatched instance's buffer has a new version; its first frame is consumed if the frame executes.
            for (uint32_t i = 0; i < selected && outputs.vertexBuffers && i < outputs.vertexBufferCount; ++i)
            {
                const ExtractedSkinnedInstance& instance = snapshot->skinnedInstances[i];
                const bool dispatched = outputs.vertexBuffers[i] != ids[i];
                ids[i] = outputs.vertexBuffers[i];
                if (dispatched && instance.firstFrame && !s.ConsumeFirstFrame(instance.leaf))
                    result = nvrhi::FE_OUT_OF_MEMORY;   // the flag stays unset: the next dispatch is a first one again
            }
            s.stats.skinningBindingSets = s.skinningPass->GetCachedBindingSetCount();
        }
        frame->skinnedVertexBuffers = ids;
        frame->skinnedVertexBufferCount = total;
        const nvrhi::FRESULT reads = RebuildBufferReads(graph, frame);
        return NVRHI_FAILED(reads) ? reads : result;
    }

    nvrhi::FRESULT RenderService::AddShadowPasses(IRenderGraph* graph, SceneFrame* frame) noexcept
    {
        if (!graph || !frame || !m_Impl || !m_Impl->frameOpen)
            return nvrhi::FE_INVALID_ARGS;
        // TODO(E5.6): a DepthPass use per cascade of each service shadow (RenderServiceDesc::serviceShadows).
        return nvrhi::FS_OK;
    }

    void RenderService::EndFrame(bool executed) noexcept
    {
        if (m_Impl)
            m_Impl->EndFrame(executed);
    }

    const RenderServiceStats& RenderService::GetStats() const noexcept
    {
        static const RenderServiceStats s_None{};
        return m_Impl ? m_Impl->stats : s_None;
    }

    const LightShadowTable* RenderService::GetLightShadows() const noexcept
    {
        return m_Impl ? &m_Impl->shadows : nullptr;
    }
}
