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
// and the legacy strategies' per-geometry rules, each view's packets of each pass radix-sorted by key. The skinning pass
// is E5.5's; AddUploadPasses is a thin call into E5.4's GpuSceneUploadPass.
//
// Service shadows (E5.6, LightShadows.h; RenderServiceDesc::serviceShadows): the Lights system reads each light's hint;
// the first light with a cascaded hint (one shadow texture in the lighting passes) requests its LightShadow from the
// table keyed by entity, makes a CascadedShadowMap for a new or reconfigured entry,
// copies the hint's lit-out-of-bounds and falloff onto it, sets its cascades up for the shadow view, and appends one
// ExtractedView per cascade after the desc's views, so that Instances extracts their packets too. A sweep in entity
// order then retires the maps of entries no light requested; EndFrame releases them. AddShadowPasses adds one DepthPass
// use per cascade view (a DepthPass and a PacketDrawStrategy per cascade, pooled) into the map's imported texture.

#include <donut/render/RenderExtract.h>
#include <donut/render/CascadedShadowMap.h>
#include <donut/render/DepthPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/GeometryPasses.h>
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
#include <donut/core/query_cast.h>

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
        // The graph's render-read scope (a flag per service, not a count; the graph counts the scopes). A Defer scope
        // for a pipelined service (RenderServiceDesc::pipelined), a Warn scope otherwise; `scopeId` is its identity
        // (SceneGraph::BeginRenderRead's), by which EndFrame closes it.
        bool scopeOpen = false;
        uint32_t scopeId = 0;
        // Pipelined rendering (part I7): between BeginPipelinedRecording (the hand-off of the open frame to a recording
        // thread) and the EndFrame after the join. The scene's owner-thread checks are begun meanwhile.
        bool recording = false;
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

        // Service shadows (E5.6). The service's desc and device, for the Lights system (set by Extract).
        const RenderServiceDesc* serviceDesc = nullptr;
        nvrhi::IDevice* device = nullptr;
        LightShadowTable shadows;
        // The Lights system's served lights of the frame: (light index, its map), capacity kept.
        struct ServedShadow
        {
            uint32_t light = 0;
            CascadedShadowMap* map = nullptr;
        };
        std::vector<ServedShadow> servedShadows;
        // The frame's hints the service does not serve: another kind than a cascaded directional one, and cascaded
        // ones after the first (one service shadow: the lighting passes bind one shadow texture).
        uint32_t unsupportedHints = 0;
        uint32_t withheldHints = 0;
        // One warning per service each.
        bool shadowViewWarned = false;
        bool shadowSettingsWarned = false;
        bool unsupportedWarned = false;
        bool withheldWarned = false;
        bool shadowConfigWarned = false;
        // AddShadowPasses ran for the open frame (a second call adds nothing).
        bool shadowsAdded = false;
        // AddUploadPasses and AddSkinningPasses were called for the open frame (even when they added nothing): the
        // order AddShadowPasses checks.
        bool uploadCalled = false;
        bool skinningCalled = false;
        // A frame replaced the scene's buffers (SceneFrame::sceneBuffersReplaced) since the shadow depth passes' binding
        // caches were last reset: set by Extract (and by AddShadowPasses from its frame), cleared only by the
        // AddShadowPasses call that resets every pooled DepthPass, so that a frame whose AddShadowPasses did not run or
        // was refused does not lose the reset.
        bool shadowBindingResetPending = false;
        // One per cascade view of a frame, pooled: a DepthPass object per cascade (a graph records one pass object's
        // uses on one thread), its bound strategy (strategies are stateful) and context.
        struct ShadowSlot
        {
            nvrhi::AutoPtr<DepthPass> pass;
            nvrhi::AutoPtr<PacketDrawStrategy> strategy;
            DepthPass::Context context;
        };
        std::vector<std::unique_ptr<ShadowSlot>> shadowSlots;
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
            // The instance query no longer names MeshInstanceLeaf (plan step E7, decision 8); the tag is registered
            // first, as the query's first term did, so that a world whose service comes before its first mesh leaf
            // keeps the component ids it had.
            nvrhi::FRESULT result = RegisterComponent<MeshInstanceLeaf>(world) != ComponentId::Invalid
                                        ? nvrhi::FS_OK : nvrhi::FE_INVALID_ARGS;
            if (NVRHI_SUCCEEDED(result))
                result = instanceQuery.Create(world, "RenderService.Instances");
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
            const bool service = serviceDesc && serviceDesc->serviceShadows;
            servedShadows.clear();
            unsupportedHints = 0;
            withheldHints = 0;
            // A failure before the Sweep forgets this extraction's requests, so that the next Sweep sees only its own
            // (a light removed meanwhile does not keep its map an extra frame).
            auto fail = [&]() noexcept
            {
                if (service)
                    shadows.CancelRequests();
                return Fail();
            };
            bool claimed = false;   // a light took the frame's one service shadow
            for (uint32_t i = 0; i < count; ++i)
            {
                const Light* light = lights[i].Get();
                std::memset(&constants[i], 0, sizeof(LightConstants));
                light->FillLightConstants(constants[i]);
                ExtractedLight& extracted = out[i];
                extracted.constants = &constants[i];
                extracted.entity = graph->GetEntity(light->GetNode());
                extracted.leaf = light;
                extracted.shadowTexture = InvalidResourceId;
                extracted.serviceShadow = false;
                if (service)
                {
                    CascadedShadowMap* map = nullptr;
                    bool withheld = false;
                    const nvrhi::FRESULT result = ServeShadow(i, light, extracted.entity, claimed, map, withheld);
                    if (NVRHI_FAILED(result))
                        return fail();
                    if (map)
                    {
                        // the service's map, held by the table until a later extraction retires it
                        extracted.shadowMap = map;
                        extracted.serviceShadow = true;
                        continue;
                    }
                    if (withheld)
                    {
                        // a cascaded hint after the first: no shadow (one shadow texture in the lighting passes), and
                        // the constants of a light without a map (FillLightConstants: lit outside any cascade), or
                        // the shaders would take the hint's out-of-bounds value for the whole light
                        extracted.shadowMap = nullptr;
                        constants[i].outOfBoundsShadow = 1.f;
                        continue;
                    }
                }
                // Without a service shadow the light keeps its leaf's own map, which the application renders; held
                // until EndFrame, so that the application may replace Light::shadowMap meanwhile.
                extracted.shadowMap = light->shadowMap.Get();
                if (light->shadowMap)
                {
                    try
                    {
                        frameShadowMaps.push_back(light->shadowMap);
                    }
                    catch (...)
                    {
                        return fail();
                    }
                }
            }
            snapshot->lights = out;
            snapshot->lightConstants = constants;
            snapshot->lightCount = count;
            if (service)
            {
                // the entries no light requested this extraction, in entity order: their maps retire
                shadows.Sweep();
                if (unsupportedHints != 0 && !unsupportedWarned)
                {
                    unsupportedWarned = true;
                    log::warning("RenderService: %u light(s) have a shadow-map hint the service does not serve (only "
                                 "a DirectionalLight with a CascadedShadowMap); they keep their own map",
                                 unsupportedHints);
                }
                if (withheldHints != 0 && !withheldWarned)
                {
                    withheldWarned = true;
                    log::warning("RenderService: %u more light(s) have a CascadedShadowMap hint; the service serves the "
                                 "first light's only (the lighting passes bind one shadow texture), the others get no "
                                 "shadow", withheldHints);
                }
                return AppendCascadeViews();
            }
            return nvrhi::FS_OK;
        }

        // ------------------------------------------------------------------------------- service shadows (E5.6)

        // A new map for `shadow`: its texture from the desc's seam, else the device. Null when neither can make it.
        CascadedShadowMap* CreateShadowMap(const LightShadow& shadow) noexcept
        {
            nvrhi::TextureDesc textureDesc = CascadedShadowMap::GetTextureDesc(
                int(shadow.resolution), int(shadow.cascadeCount), 0, shadow.format);
            textureDesc.debugName = "ServiceShadowMap";
            nvrhi::TextureHandle texture;
            if (serviceDesc->createShadowTexture)
                texture = nvrhi::TakeOver(serviceDesc->createShadowTexture(serviceDesc->createShadowTextureContext,
                                                                           textureDesc));
            else if (device)
                device->createTexture(textureDesc, &texture);
            if (!texture)
                return nullptr;
            try
            {
                nvrhi::AutoPtr<CascadedShadowMap> map =
                    MAKE_RC_OBJ_PTR(CascadedShadowMap, device, texture.Get(), int(shadow.cascadeCount), 0);
                return map.Detach();
            }
            catch (...)
            {
                return nullptr;
            }
        }

        // The view the frame's service shadow is set up for, or null when the desc names none or its ShadowSettings
        // are not valid for it (AreShadowSettingsValid: what SetupForPlanarView asserts); one warning per service each.
        const IView* GetShadowView() noexcept
        {
            const IView* view = desc->shadowViewIndex < desc->viewCount ? desc->views[desc->shadowViewIndex] : nullptr;
            if (!view)
            {
                if (!shadowViewWarned)
                {
                    shadowViewWarned = true;
                    log::warning("RenderService: ExtractDesc::shadowViewIndex %u names no view; the service shadows "
                                 "are not set up", desc->shadowViewIndex);
                }
                return nullptr;
            }
            if (!AreShadowSettingsValid(view, desc->shadows))
            {
                if (!shadowSettingsWarned)
                {
                    shadowSettingsWarned = true;
                    log::warning("RenderService: ExtractDesc::shadows is not valid for the shadow view (an exponent "
                                 "above 1, and a maxShadowDistance above 0 or a view with a far plane); the service "
                                 "shadows are not set up");
                }
                return nullptr;
            }
            return view;
        }

        // One light's service shadow. The first light whose hint the service serves (`claimed` still false) takes the
        // frame's one service shadow: its entry is requested, made a map when it has none, the hint's values copied
        // onto the map and the cascades set up for the shadow view; `outMap` is then the map, or null when it cannot
        // be served this frame (no map, no valid shadow view: the light keeps its leaf's map). A later light with such
        // a hint is withheld (`outWithheld`: no shadow). Another hint is unsupported (the light keeps its leaf's map).
        // Fails only when the table cannot grow.
        nvrhi::FRESULT ServeShadow(uint32_t index, const Light* light, Entity entity, bool& claimed,
                                   CascadedShadowMap*& outMap, bool& outWithheld) noexcept
        {
            outMap = nullptr;
            outWithheld = false;
            LightShadow shadow;
            LightShadowKind kind = LightShadowKind::Cascaded;
            if (!GetLightShadowRequest(light, shadow, kind) || entity == Entity::Null)
            {
                unsupportedHints += light->shadowMap ? 1u : 0u;
                return nvrhi::FS_OK;
            }
            if (claimed)
            {
                ++withheldHints;
                outWithheld = true;
                return nvrhi::FS_OK;
            }
            claimed = true;
            LightShadowEntry* entry = shadows.Request(entity, shadow, kind);
            if (!entry)
                return nvrhi::FE_OUT_OF_MEMORY;
            if (!entry->map)
                entry->map = CreateShadowMap(shadow);   // the table's reference
            if (!entry->map)
                return nvrhi::FS_OK;   // no device and no seam, or out of memory: the next extraction tries again

            const IView* shadowView = GetShadowView();
            if (!shadowView)
                return nvrhi::FS_OK;

            auto* map = static_cast<CascadedShadowMap*>(entry->map);
            const CascadedShadowMap* hint = query_cast<CascadedShadowMap>(light->shadowMap.Get());
            map->SetLitOutOfBounds(hint->IsLitOutOfBounds());
            map->SetFalloffDistance(hint->GetFalloffDistance());
            if (!SetupLightShadowCascades(map, static_cast<const DirectionalLight*>(light), shadowView, desc->shadows,
                                          dm::float3(snapshot->renderOrigin)))
                return nvrhi::FS_OK;
            try
            {
                servedShadows.push_back(ServedShadow{ index, map });
            }
            catch (...)
            {
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            outMap = map;
            return nvrhi::FS_OK;
        }

        // The served maps' cascades as views after the desc's (ExtractedView::light, cascade), in light order, then
        // cascade order. The views array grows into a new arena array (the old one stays unused until Reset).
        nvrhi::FRESULT AppendCascadeViews() noexcept
        {
            uint32_t cascades = 0;
            for (const ServedShadow& served : servedShadows)
                cascades += served.map->GetNumberOfCascades();
            if (cascades == 0)
                return nvrhi::FS_OK;
            const uint32_t descCount = snapshot->viewCount;
            ExtractedView* out = AllocateConstructed<ExtractedView>(descCount + cascades);
            if (!out)
                return Fail();
            for (uint32_t v = 0; v < descCount; ++v)
                out[v] = views[v];
            uint32_t at = descCount;
            for (const ServedShadow& served : servedShadows)
            {
                for (uint32_t c = 0; c < served.map->GetNumberOfCascades(); ++c)
                {
                    const IView* view = served.map->GetCascadeView(c);
                    ExtractedView& extracted = out[at++];
                    extracted.view = view;
                    extracted.frustum = view->GetViewFrustum();
                    extracted.viewOrigin = view->GetViewOrigin();
                    extracted.light = served.light;
                    extracted.cascade = c;
                }
            }
            views = out;
            snapshot->views = out;
            snapshot->viewCount = descCount + cascades;
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
            frame.compactPacketRecords = RenderService::GetCompactPacketRecords();
            if (NVRHI_FAILED(instanceVisibility.Run(instanceQuery, frame, *snapshot, bytes)))
                return Fail();
            return nvrhi::FS_OK;
        }

        // ------------------------------------------------------------------------------- frame end

        // Closes the service's own scope, by its identity (backlog I6-N3): a scope opened after Extract and still open
        // stays open, and the mode that closes is the service's. One that an EndRenderRead without its BeginRenderRead
        // closed already gets the graph's warning, and nothing else closes.
        void CloseScope() noexcept
        {
            if (!scopeOpen)
                return;
            scopeOpen = false;
            graph->EndRenderRead(scopeId);
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

        // Pipelined rendering: the recording has joined; the scene's owner-thread checks end (before the scope closes,
        // whose last EndRenderRead runs the deferred reconciliation and releases the retire list on this thread).
        void EndRecording() noexcept
        {
            if (!recording)
                return;
            if (scene)
                scene->EndPipelinedRecording();
            recording = false;
        }

        // Idempotent: the scope closes once, and the first-frame flags are set at most once, for an executed frame.
        void EndFrame(bool executed) noexcept
        {
            EndRecording();
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
            shadowsAdded = false;
            uploadCalled = false;
            skinningCalled = false;
            // Retired at this frame's extraction: no pass of this frame draws into or reads them.
            shadows.ReleaseRetired();
            ReleaseFrameReferences();
        }
    };

    RenderService::RenderService(nvrhi::IDevice* device, const RenderServiceDesc& desc) noexcept
        : m_Device(device)
        , m_Desc(desc)
    {
    }

    RenderService::~RenderService()
    {
        if (m_Impl && m_Impl->recording)
        {
            log::warning("RenderService: destroyed while its frame records (BeginPipelinedRecording without the "
                         "EndFrame after the join); its state, the graph's Defer scope and the graph are kept for good "
                         "(leaked: the graph stays claimed and deferring, its retire list grows, walking strategies "
                         "draw nothing) so that the recording reads nothing freed; join the recording and call "
                         "EndFrame first");
            (void)m_Impl.release();
        }
    }

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
        // A pipelined frame handed to its recording thread and not yet joined (EndFrame not called after
        // BeginPipelinedRecording): closing its Defer scope here would release what the recording still reads and run
        // the deferred reconciliation under it (backlog I6-R4). Refused; nothing changes.
        if (s.recording)
        {
            log::warning("RenderService::Extract: the previous frame is still recording (BeginPipelinedRecording "
                         "without its EndFrame); join it and call EndFrame first; the frame is not extracted");
            return nvrhi::FE_INVALID_ARGS;
        }
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
            // The meshes point at the new buffers from here on, even when this Extract fails below: the shadow depth
            // passes reset their binding caches at the next AddShadowPasses that runs.
            if (sceneBuffersReplaced)
                s.shadowBindingResetPending = true;
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
        s.serviceDesc = &m_Desc;
        s.device = m_Device;
        s.renderGraph = graph;
        s.desc = &desc;
        s.snapshot = snapshot;
        s.views = nullptr;
        s.failure = nvrhi::FS_OK;
        s.bytes = sizeof(RenderSnapshot);
        // Pipelined (part I7): the deferring scope, whose retire list keeps what the main thread drops while the frame
        // records on another thread; serial: the warning scope, as before.
        s.scopeId = s.graph->BeginRenderRead(m_Desc.pipelined ? SceneGraph::RenderReadMode::Defer
                                                              : SceneGraph::RenderReadMode::Warn);
        s.scopeOpen = true;
        if (m_Desc.pipelined && s.scopeId == 0)
        {
            // Nested beyond 64 scopes, the Defer request acts as Warn: no retire list would keep what the main thread
            // drops while the frame records (F5 review, F4-N1). The scope is the innermost now: it closes.
            log::warning("RenderService::Extract: the graph has 64 render-read scopes open; a pipelined frame's scope "
                         "would not defer; the frame is not extracted");
            s.CloseScope();
            s.renderGraph = nullptr;
            s.desc = nullptr;
            s.snapshot = nullptr;
            return nvrhi::FE_INVALID_ARGS;
        }
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
        for (uint32_t e = 0; e < s.shadows.GetCount(); ++e)
            stats.serviceShadowMaps += s.shadows.GetEntries()[e].map ? 1u : 0u;
        stats.unservedShadowHints = s.unsupportedHints + s.withheldHints;
        stats.retiredShadowMaps = s.shadows.GetRetiredCount();
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
        s.uploadCalled = true;   // the order AddShadowPasses checks
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
        s.skinningCalled = true;   // the order AddShadowPasses checks
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

    // Service shadows (E5.6): for each served light (ExtractedLight::serviceShadow), its map's texture imported
    // host-kept (CascadedShadowMap makes it keepInitialState, ShaderResource) and one DepthPass use per cascade view of
    // the snapshot (ExtractedView::light), in cascade order, each clearing its slice and drawing the view's opaque
    // packets through its own bound PacketDrawStrategy with the frame's buffer reads. The last version goes into the
    // extracted light's shadowTexture. The DepthPass objects are pooled per cascade slot and made with the desc's bias;
    // their binding caches are reset by the first call that runs after a frame replaced the scene's buffers (the
    // pending reset survives frames whose call did not run or was refused). With a device the passes are initialised,
    // which needs the shader factory and the common passes; without one (device-free tests, the seam) they are not.
    nvrhi::FRESULT RenderService::AddShadowPasses(IRenderGraph* graph, SceneFrame* frame) noexcept
    {
        if (!graph || !frame || !m_Impl || !m_Impl->frameOpen)
            return nvrhi::FE_INVALID_ARGS;
        Impl& s = *m_Impl;
        if (!m_Desc.serviceShadows)
            return nvrhi::FS_OK;
        if (frame->sceneBuffersReplaced)
            s.shadowBindingResetPending = true;
        if (s.shadowsAdded)
            return nvrhi::FS_OK;
        const RenderSnapshot* snapshot = frame->snapshot;
        if (!snapshot)
            return nvrhi::FE_INVALID_ARGS;
        if (m_Device && (!m_Desc.shaderFactory || !m_Desc.commonPasses))
        {
            if (!s.shadowConfigWarned)
            {
                s.shadowConfigWarned = true;
                log::warning("RenderService::AddShadowPasses: with a device the depth passes need "
                             "RenderServiceDesc::shaderFactory and commonPasses; no shadow pass is added");
            }
            return nvrhi::FE_INVALID_ARGS;
        }
        // The depth passes read SceneFrame::bufferReads, which the upload and skinning passes fill: the calls must come
        // first (checked for every service; without a scene or device they add nothing but still record the call), and
        // on the GPU path they must have added their passes.
        const bool gpu = s.scene && m_Device;
        if (!s.uploadCalled || !s.skinningCalled || (gpu && (!s.uploadAdded || !s.skinningAdded)))
        {
            log::warning("RenderService::AddShadowPasses: call AddUploadPasses and AddSkinningPasses first (the depth "
                         "passes read their buffers)");
            return nvrhi::FE_INVALID_ARGS;
        }
        s.shadowsAdded = true;

        if (s.shadowBindingResetPending)
        {
            for (const auto& slot : s.shadowSlots)
                slot->pass->ResetBindingCache();
            s.stats.shadowBindingCacheResets = uint32_t(s.shadowSlots.size());
            s.shadowBindingResetPending = false;
        }

        uint32_t slotIndex = 0;
        for (uint32_t i = 0; i < snapshot->lightCount; ++i)
        {
            ExtractedLight& light = const_cast<ExtractedLight&>(snapshot->lights[i]);
            if (!light.serviceShadow || !light.shadowMap)
                continue;
            ResourceId texture = graph->ImportHostKeptTexture(light.shadowMap->GetTexture());
            if (texture == InvalidResourceId)
                return nvrhi::FE_OUT_OF_MEMORY;
            for (uint32_t v = snapshot->descViewCount; v < snapshot->viewCount; ++v)
            {
                const ExtractedView& view = snapshot->views[v];
                if (view.light != i)
                    continue;
                if (slotIndex == s.shadowSlots.size())
                {
                    try
                    {
                        auto slot = std::make_unique<Impl::ShadowSlot>();
                        slot->pass = MAKE_RC_OBJ_PTR(DepthPass, m_Device, m_Desc.commonPasses);
                        slot->strategy = MAKE_RC_OBJ_PTR(PacketDrawStrategy);
                        if (m_Device && m_Desc.shaderFactory && m_Desc.commonPasses)
                        {
                            DepthPass::CreateParameters params;
                            params.depthBias = m_Desc.shadowDepthBias;
                            params.depthBiasClamp = m_Desc.shadowDepthBiasClamp;
                            params.slopeScaledDepthBias = m_Desc.shadowSlopeScaledDepthBias;
                            slot->pass->Init(*m_Desc.shaderFactory, params);
                        }
                        s.shadowSlots.push_back(std::move(slot));
                    }
                    catch (...)
                    {
                        return nvrhi::FE_OUT_OF_MEMORY;
                    }
                }
                Impl::ShadowSlot& slot = *s.shadowSlots[slotIndex++];
                slot.strategy->SetPackets(snapshot, DrawPacketPass::Opaque);

                GeometryGraphInputs inputs;
                inputs.name = "ShadowDepth";
                inputs.view = view.view;
                inputs.drawStrategy = slot.strategy;
                inputs.passContext = &slot.context;
                inputs.depth = texture;
                inputs.clearTargets = true;   // the cascade's slice, as CascadedShadowMap::Clear
                inputs.depthClear.depth = 1.f;
                inputs.bufferReads = frame->bufferReads;
                inputs.bufferReadCount = frame->bufferReadCount;
                GeometryGraphOutputs outputs;
                try
                {
                    outputs = slot.pass->AddToGraph(graph, inputs);
                }
                catch (...)
                {
                    return nvrhi::FE_OUT_OF_MEMORY;
                }
                if (outputs.depth == texture)
                    return nvrhi::FE_OUT_OF_MEMORY;   // the use was not added
                texture = outputs.depth;
            }
            light.shadowTexture = texture;
        }
        return nvrhi::FS_OK;
    }

    void RenderService::EndFrame(bool executed) noexcept
    {
        if (m_Impl)
            m_Impl->EndFrame(executed);
    }

    nvrhi::FRESULT RenderService::BeginPipelinedRecording() noexcept
    {
        if (!m_Desc.pipelined)
        {
            log::warning("RenderService::BeginPipelinedRecording: the service is not pipelined "
                         "(RenderServiceDesc::pipelined); record the frame serially");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!m_Impl || !m_Impl->frameOpen)
        {
            log::warning("RenderService::BeginPipelinedRecording: no frame is open (Extract did not run or failed, or "
                         "EndFrame came)");
            return nvrhi::FE_INVALID_ARGS;
        }
        Impl& s = *m_Impl;
        if (s.recording)
            return nvrhi::FS_OK;
        if (s.scene)
            s.scene->BeginPipelinedRecording();
        s.recording = true;
        return nvrhi::FS_OK;
    }

    void RenderService::HandOverPipelinedRecording() noexcept
    {
        if (m_Impl && m_Impl->recording && m_Impl->scene)
            m_Impl->scene->HandOverPipelinedRecording();
    }

    bool RenderService::IsPipelinedRecording() const noexcept
    {
        return m_Impl && m_Impl->recording;
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

    namespace
    {
        std::atomic<bool> g_CompactPacketRecords{ false };   // E7.4 remedy 2: off until measured
    }

    void RenderService::SetCompactPacketRecords(bool enabled) noexcept
    {
        g_CompactPacketRecords.store(enabled, std::memory_order_relaxed);
    }

    bool RenderService::GetCompactPacketRecords() noexcept
    {
        return g_CompactPacketRecords.load(std::memory_order_relaxed);
    }
}
