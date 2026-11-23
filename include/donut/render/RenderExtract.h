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

// The render service (aggregate docs/ecs-design.md 7.1-7.6, D8, D11, D13; plan step E5, sub-plan
// docs/plans/2026-11-21-donut-ecs-e5.md). Each frame, from the top of RenderGraphLayer::BuildGraph:
//
//     RenderService::Extract        runs the scene graph's RenderExtract phase (design 6.3) into a RenderSnapshot in
//                                   the render graph's frame arena: views, visible instances and their draw packets
//                                   sorted per view and pass, lights, cameras, skinned instances and joint matrices
//                                   (D8, 7.1, 7.3); sets the service's shadow cascades up and extracts their views too.
//                                   In this order: Scene::CreateMeshBufferObjects (buffer objects, descriptors and
//                                   skinned BufferGroups of meshes new to the graph; their data is written later, in
//                                   the upload pass's Prepare), GpuScene::Update, GpuScene::CreateBuffers,
//                                   Scene::BindMeshInstanceBuffers, then the RenderExtract phase. Publishes a
//                                   SceneFrame in the FrameData.
//     AddUploadPasses               the GpuScene upload (GpuSceneUploadPass.h): imports the buffers, scatters the rows
//                                   that changed (7.2, 7.4)
//     AddSkinningPasses             skinning as an IGpuPass (SkinningPass.h, 7.4)
//     AddShadowPasses               a DepthPass use per cascade of each service shadow (LightShadows.h, 7.6, D13)
//     ... the application's geometry and lighting passes read the snapshot: bound draw strategies (DrawStrategy.h),
//         GeometryGraphInputs::bufferReads, the extracted-light inputs of the lighting passes
//     RenderService::EndFrame       after the graph executed (or failed to compile or execute): closes the Debug
//                                   render-read scope. app::RenderGraphLayer::OnGraphExecuted is the place; without
//                                   the layer, call it after IRenderGraph::Execute.
//
// The snapshot is immutable once Extract returns, except ExtractedLight::shadowTexture, which AddShadowPasses fills
// before any pass reads it. Its arrays are trivially destructible and released with the frame (IRenderGraph::Reset).
// It copies values; the object pointers it keeps (leaves, meshes, and through the meshes geometries, materials and
// buffer groups; shadow maps) stay valid while no structural scene-graph call happens, which the Debug render-read
// scope checks between Extract and EndFrame (5.6, 7.5; pipelining in step E7 replaces the guard with deferred
// releases). Recording reads the snapshot and those objects, never the world.
//
// One service per scene graph: its systems run in the graph's RenderExtract phase and it publishes the frame's one
// SceneFrame, so a second service on the same graph is refused (its first Extract returns FE_INVALID_ARGS).
//
// Module-private C++, like GpuScene.h: it names scene-graph implementation classes, and the ABI lint does not cover it.
// The service follows the ABI conventions anyway: noexcept, FRESULT, raw pointers in parameters, no smart-pointer
// parameters.

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/EcsTypes.h>
#include <donut/render/LightShadows.h>
#include <donut/render/RenderGraph.h>
#include <nvrhi/core/foundation.h>
#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

struct LightConstants;   // donut/shaders/light_cb.h

namespace donut::engine
{
    enum class MeshId : uint32_t;   // AssetTables.h
    struct MeshInfo;
    class IView;
    class IShadowMap;
    class Light;
    class MeshInstance;
    class SceneCamera;
    class SkinnedMeshInstance;
    class SceneGraph;
    class Scene;
    class ShaderFactory;
    class CommonRenderPasses;
}

namespace donut::render
{
    class FrameData;

    // ---------------------------------------------------------------------------------- draw packets (7.3, D11)

    // The passes packets are sorted for, matching the two legacy strategies.
    enum class DrawPacketPass : uint8_t
    {
        Opaque,        // MaterialDomain::Opaque and AlphaTested: InstancedOpaqueDrawStrategy
        Transparent,   // the other domains: TransparentDrawStrategy
        Count
    };
    constexpr uint32_t c_DrawPacketPassCount = uint32_t(DrawPacketPass::Count);

    // DrawPackets::flags.
    constexpr uint8_t c_DrawPacketFlagOpaquePass = 0x01;
    constexpr uint8_t c_DrawPacketFlagTransparentPass = 0x02;
    constexpr uint8_t c_DrawPacketFlagDoubleSided = 0x04;        // the material is double-sided
    constexpr uint8_t c_DrawPacketFlagGeometryCulled = 0x08;     // the per-geometry frustum test of the legacy rules ran

    // The 64-bit sort key (open point 6). Keys are radix-sorted per view and pass, stably, so equal keys keep chunk
    // order. Fields are given as (shift, bits); a value wider than its field is masked. In an opaque key that only
    // costs batching (the packets carry the full slots; opaque draws are depth-tested). In a transparent key the
    // instance slot only breaks ties between equal distances, but from 2^24 slots on a masked slot changes which of two
    // equally distant instances blends first.
    //
    //     all keys      [63:62] pass (DrawPacketPass)
    //     opaque        [61:59] material domain   [58:57] cull mode   [56:39] material slot   [38:20] geometry slot
    //                   [19:0] instance slot
    //                   Material first, then mesh and geometry (geometry slots are ranges of their mesh's row), then
    //                   the instance slot, so that RenderView batches by material and buffers and merges consecutive
    //                   instance slots as before. The depth field of design 7.3 has no bits here (it would be zero):
    //                   depth before the instance slot breaks instance merging; front-to-back is a later change.
    //     transparent   [61:30] distance, descending (the bits of the non-negative float, inverted)
    //                   [29:6] instance slot   [5:4] cull order (None, Front, Back: the legacy order of a
    //                   double-sided material's two draws)   [3:0] zero
    constexpr uint32_t c_DrawKeyPassShift = 62, c_DrawKeyPassBits = 2;

    constexpr uint32_t c_OpaqueKeyDomainShift = 59, c_OpaqueKeyDomainBits = 3;
    constexpr uint32_t c_OpaqueKeyCullShift = 57, c_OpaqueKeyCullBits = 2;
    constexpr uint32_t c_OpaqueKeyMaterialShift = 39, c_OpaqueKeyMaterialBits = 18;
    constexpr uint32_t c_OpaqueKeyGeometryShift = 20, c_OpaqueKeyGeometryBits = 19;
    constexpr uint32_t c_OpaqueKeyInstanceShift = 0, c_OpaqueKeyInstanceBits = 20;

    constexpr uint32_t c_TransparentKeyDistanceShift = 30, c_TransparentKeyDistanceBits = 32;
    constexpr uint32_t c_TransparentKeyInstanceShift = 6, c_TransparentKeyInstanceBits = 24;
    constexpr uint32_t c_TransparentKeyCullShift = 4, c_TransparentKeyCullBits = 2;

    static_assert(c_DrawKeyPassShift + c_DrawKeyPassBits == 64);
    static_assert(c_OpaqueKeyDomainShift + c_OpaqueKeyDomainBits == c_DrawKeyPassShift);
    static_assert(c_OpaqueKeyCullShift + c_OpaqueKeyCullBits == c_OpaqueKeyDomainShift);
    static_assert(c_OpaqueKeyMaterialShift + c_OpaqueKeyMaterialBits == c_OpaqueKeyCullShift);
    static_assert(c_OpaqueKeyGeometryShift + c_OpaqueKeyGeometryBits == c_OpaqueKeyMaterialShift);
    static_assert(c_OpaqueKeyInstanceShift + c_OpaqueKeyInstanceBits == c_OpaqueKeyGeometryShift);
    static_assert(c_TransparentKeyDistanceShift + c_TransparentKeyDistanceBits == c_DrawKeyPassShift);
    static_assert(c_TransparentKeyInstanceShift + c_TransparentKeyInstanceBits == c_TransparentKeyDistanceShift);
    static_assert(c_TransparentKeyCullShift + c_TransparentKeyCullBits == c_TransparentKeyInstanceShift);

    constexpr uint64_t MakeDrawKeyField(uint64_t value, uint32_t shift, uint32_t bits) noexcept
    {
        return (value & ((uint64_t(1) << bits) - 1)) << shift;
    }

    // `domain` is engine::MaterialDomain, `cullMode` nvrhi::RasterCullMode, as integers.
    constexpr uint64_t MakeOpaqueDrawKey(uint32_t domain, uint32_t cullMode, uint32_t materialSlot, uint32_t geometrySlot,
                                         uint32_t instanceSlot) noexcept
    {
        return MakeDrawKeyField(uint32_t(DrawPacketPass::Opaque), c_DrawKeyPassShift, c_DrawKeyPassBits)
            | MakeDrawKeyField(domain, c_OpaqueKeyDomainShift, c_OpaqueKeyDomainBits)
            | MakeDrawKeyField(cullMode, c_OpaqueKeyCullShift, c_OpaqueKeyCullBits)
            | MakeDrawKeyField(materialSlot, c_OpaqueKeyMaterialShift, c_OpaqueKeyMaterialBits)
            | MakeDrawKeyField(geometrySlot, c_OpaqueKeyGeometryShift, c_OpaqueKeyGeometryBits)
            | MakeDrawKeyField(instanceSlot, c_OpaqueKeyInstanceShift, c_OpaqueKeyInstanceBits);
    }

    // The legacy distance: length(box.center() - viewOrigin) in float. A value with the sign bit set (-0, or anything
    // negative) is taken as 0; a NaN (positive, as length gives it) orders above every finite distance, so it sorts
    // first. `cullMode` is nvrhi::RasterCullMode (Back 0, Front 1, None 2); its order field is 2 - cullMode, so Front
    // sorts before Back.
    inline uint64_t MakeTransparentDrawKey(float distance, uint32_t instanceSlot, uint32_t cullMode) noexcept
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &distance, sizeof(bits));
        if (bits & 0x80000000u)
            bits = 0;
        const uint32_t descending = ~bits;
        return MakeDrawKeyField(uint32_t(DrawPacketPass::Transparent), c_DrawKeyPassShift, c_DrawKeyPassBits)
            | MakeDrawKeyField(descending, c_TransparentKeyDistanceShift, c_TransparentKeyDistanceBits)
            | MakeDrawKeyField(instanceSlot, c_TransparentKeyInstanceShift, c_TransparentKeyInstanceBits)
            | MakeDrawKeyField(2u - (cullMode & 3u), c_TransparentKeyCullShift, c_TransparentKeyCullBits);
    }

    // DrawItem::distanceToCamera of a transparent packet.
    inline float GetTransparentDrawKeyDistance(uint64_t key) noexcept
    {
        const uint32_t bits = ~uint32_t(key >> c_TransparentKeyDistanceShift);
        float distance = 0.f;
        std::memcpy(&distance, &bits, sizeof(distance));
        return distance;
    }

    // A range of one of the snapshot's arrays.
    struct SnapshotRange
    {
        uint32_t first = 0;
        uint32_t count = 0;
    };

    // The packets of every view and pass, SoA (7.3); each view's packets of one pass are a range
    // (ExtractedView::packets), sorted by key.
    struct DrawPackets
    {
        const uint64_t* keys = nullptr;
        const uint32_t* visibleInstances = nullptr;   // index into VisibleInstances
        const uint32_t* geometryIndices = nullptr;    // index into the mesh's MeshInfo::geometries
        const uint32_t* geometrySlots = nullptr;      // MeshGeometry::globalGeometryIndex
        const uint32_t* materialSlots = nullptr;      // the material's slot in this graph's table
        const uint8_t* cullModes = nullptr;           // nvrhi::RasterCullMode
        const uint8_t* flags = nullptr;               // c_DrawPacketFlag*
        uint32_t count = 0;
    };

    // The instances visible in each view, SoA; each view's are a range (ExtractedView::instances), in chunk order.
    struct VisibleInstances
    {
        const engine::Entity* entities = nullptr;
        const uint32_t* instanceSlots = nullptr;              // GpuInstanceSlot::instance
        const engine::MeshId* meshes = nullptr;               // MeshRef::mesh
        const uint32_t* firstGeometryInstanceSlots = nullptr; // GpuInstanceSlot::firstGeometryInstance
        const uint32_t* geometryCounts = nullptr;             // GpuInstanceSlot::geometryCount
        const engine::MeshInstance* const* leaves = nullptr;  // DrawItem::instance (MeshInstanceLeaf guarantees the class)
        // The instance's mesh, taken at extraction (MeshInstance::GetMesh()): the adapters read it from here, so that
        // recording reads the snapshot and not the leaf (D8, 7.5).
        const engine::MeshInfo* const* meshInfos = nullptr;
        const dm::box3* worldBounds = nullptr;                // WorldBounds, float
        uint32_t count = 0;
    };

    // ----------------------------------------------------------------------------------------------- the snapshot

    constexpr uint32_t c_NoExtractedLight = ~0u;

    // One planar view: one of ExtractDesc::views, or a cascade of a service shadow (light != c_NoExtractedLight).
    struct ExtractedView
    {
        // The service's cascade views stay valid until the next Extract. The bound draw strategies find a view's packets
        // by this pointer (the IView& their PrepareForView gets), so every view appears once.
        const engine::IView* view = nullptr;
        dm::frustum frustum;                   // view->GetViewFrustum()
        dm::float3 viewOrigin = 0.f;           // view->GetViewOrigin(), the transparent distances' origin
        SnapshotRange instances;               // into RenderSnapshot::instances
        SnapshotRange packets[c_DrawPacketPassCount];   // into RenderSnapshot::packets
        uint32_t light = c_NoExtractedLight;   // the light (RenderSnapshot::lights) whose cascade this view is
        uint32_t cascade = 0;
    };

    // One light, in SceneGraph::GetLights() order (the lighting shaders sum lights in array order).
    struct ExtractedLight
    {
        const LightConstants* constants = nullptr;   // FillLightConstants, once per light (D16); in lightConstants
        engine::Entity entity = engine::Entity::Null;
        const engine::Light* leaf = nullptr;         // its parameters stay in the leaf (D16)
        // The light's shadow: the service's map when it has a service shadow (serviceShadow), else the leaf's own
        // Light::shadowMap (rendered by the application, as before), or null. With service shadows, null too for a
        // light whose cascaded hint the service withholds (one after the first: LightShadows.h); its constants are
        // then those of a light without a map (outOfBoundsShadow 1).
        const engine::IShadowMap* shadowMap = nullptr;
        // The service shadow's texture in this frame's graph: InvalidResourceId until AddShadowPasses sets it (the
        // version its cascades wrote last); always invalid for a leaf's own map.
        ResourceId shadowTexture = InvalidResourceId;
        bool serviceShadow = false;
    };

    // One camera, in SceneGraph::GetCameras() order. The projection parameters stay in the leaf (D16).
    struct ExtractedCamera
    {
        engine::Entity entity = engine::Entity::Null;
        const engine::SceneCamera* leaf = nullptr;
        dm::affine3 viewToWorld = dm::affine3::identity();   // relative to the render origin
    };

    // A skinned instance to skin this frame: one skinning dispatch. Selected when its joints moved this frame or the
    // previous one (the legacy rule, GetLastUpdateFrameIndex() + 1 >= frameIndex), when it was never skinned
    // (skinningInitialized unset), or when a frame that did not execute selected it, until two executed frames have
    // (its dispatch and the settle of its previous positions): dropped frames and frame-index gaps lose no pose.
    struct ExtractedSkinnedInstance
    {
        engine::Entity entity = engine::Entity::Null;
        const engine::SkinnedMeshInstance* leaf = nullptr;   // its GPU objects (joint buffer, binding set) and meshes
        SnapshotRange jointMatrices;                         // into RenderSnapshot::jointMatrices
        // SkinningFlag_FirstFrame: the leaf's skinningInitialized was unset at extraction. RenderService::EndFrame of an
        // executed frame sets it, so no pass body writes a leaf and a frame that never ran keeps the flag for the next.
        // The instance's BufferGroup, vertex and joint buffers and binding set exist by then (CreateMeshBufferObjects
        // runs before the RenderExtract phase).
        bool firstFrame = false;
    };

    struct RenderSnapshot
    {
        uint64_t graphFrameIndex = 0;   // IRenderGraph::GetFrameIndex
        uint32_t frameIndex = 0;        // ExtractDesc::frameIndex
        uint32_t worldVersion = 0;      // the scene graph world's IWorld::GetVersion at extraction
        dm::double3 renderOrigin = 0.;  // D4; zero in step E5 (open point 10)

        const ExtractedView* views = nullptr;   // ExtractDesc::views in order, then the service's cascade views
        uint32_t viewCount = 0;
        uint32_t descViewCount = 0;             // the first descViewCount views are ExtractDesc's
        VisibleInstances instances;
        DrawPackets packets;

        const ExtractedLight* lights = nullptr;
        const LightConstants* lightConstants = nullptr;   // one per light, contiguous
        uint32_t lightCount = 0;
        const ExtractedCamera* cameras = nullptr;
        uint32_t cameraCount = 0;
        const ExtractedSkinnedInstance* skinnedInstances = nullptr;
        uint32_t skinnedInstanceCount = 0;
        const dm::float4x4* jointMatrices = nullptr;      // the legacy formula, converted to float last
        uint32_t jointMatrixCount = 0;
    };

    static_assert(std::is_trivially_destructible_v<ExtractedView>);
    static_assert(std::is_trivially_destructible_v<ExtractedLight>);
    static_assert(std::is_trivially_destructible_v<ExtractedCamera>);
    static_assert(std::is_trivially_destructible_v<ExtractedSkinnedInstance>);
    static_assert(std::is_trivially_destructible_v<RenderSnapshot>);

    // ------------------------------------------------------------------------------------------- frame and service

    // The FrameData entry the service publishes at Extract (design 7.4: "light and camera arrays travel in the frame
    // blackboard"). The ids are this frame's; each Add*Passes call stores the versions it wrote back here.
    struct SceneFrame
    {
        const RenderSnapshot* snapshot = nullptr;
        // GpuScene's buffers, imported at AddUploadPasses (InvalidResourceId before, and the geometry and material
        // buffers without bindless buffers).
        ResourceId instanceBuffer = InvalidResourceId;
        ResourceId geometryBuffer = InvalidResourceId;
        ResourceId materialBuffer = InvalidResourceId;
        // The skinned meshes' vertex buffers, imported at AddSkinningPasses (an arena array): first one per
        // RenderSnapshot::skinnedInstances entry, in its order, in the version the skinning pass wrote
        // (InvalidResourceId for an instance without a vertex buffer yet); then those of the graph's other skinned
        // instances, in SceneGraph::GetSkinnedMeshInstances() order, imported only, so that the geometry passes declare
        // reads of every skinned buffer they may draw (they are not host-kept on the graph path).
        const ResourceId* skinnedVertexBuffers = nullptr;
        uint32_t skinnedVertexBufferCount = 0;
        // Every buffer above that is valid, in their latest versions: GeometryGraphInputs::bufferReads of the geometry
        // passes added after the upload and skinning passes (an arena array, rebuilt by each Add*Passes call).
        const ResourceId* bufferReads = nullptr;
        uint32_t bufferReadCount = 0;
        // The GpuScene replaced a buffer this frame (growth), and Scene::BindMeshInstanceBuffers pointed the meshes'
        // BufferGroup::instanceBuffer at the new one. Passes that cache binding sets by BufferGroup* or by buffer
        // (ForwardShadingPass, GBufferFillPass, DepthPass and the application's own sets made from
        // Scene::GetInstanceBuffer) must drop them, ResetBindingCache, before they record this frame, as after a growth
        // in Scene::RefreshBuffers.
        bool sceneBuffersReplaced = false;
    };
    static_assert(std::is_trivially_destructible_v<SceneFrame>);

    struct ExtractDesc
    {
        // The planar views to extract (GetChildView of the application's composite views). They must stay valid until
        // the graph has executed.
        const engine::IView* const* views = nullptr;
        uint32_t viewCount = 0;
        // The frame index given to Scene::RefreshSceneGraph (SceneGraph::Refresh) this frame: skinning selection.
        uint32_t frameIndex = 0;
        // TransparentDrawStrategy::DrawDoubleSidedMaterialsSeparately: a double-sided transparent material gives two
        // packets (front faces, then back faces) instead of one without culling.
        bool drawDoubleSidedMaterialsSeparately = true;
        // Service shadows (RenderServiceDesc::serviceShadows): the view their cascades are set up for (an index into
        // `views`; out of range: no light is served this frame, with a warning), and how (LightShadows.h; settings
        // that fail AreShadowSettingsValid for that view: no light is served this frame, with a warning).
        uint32_t shadowViewIndex = 0;
        ShadowSettings shadows;
    };

    struct RenderServiceDesc
    {
        // The scene whose graph, GpuScene and mesh buffers the service uses. It must be on the graph path: created with
        // Scene::SetGpuSceneUploadMode(GpuSceneUploadMode::Scatter) called before it loaded (Extract returns
        // FE_INVALID_ARGS otherwise). Null with sceneGraph set: extraction only, without upload, skinning or shadow
        // passes (device-free tests).
        engine::Scene* scene = nullptr;
        // Null: the scene's graph. The service registers its RenderExtract systems on this graph's scheduler and holds
        // a reference to the graph. One service per graph.
        engine::SceneGraph* sceneGraph = nullptr;
        // The upload, skinning and depth passes' shaders; null: no GPU passes.
        engine::ShaderFactory* shaderFactory = nullptr;
        // The depth passes of the service shadows.
        engine::CommonRenderPasses* commonPasses = nullptr;
        // Shadow maps owned by the service (D13; opt-in, open point 8). Off: every light keeps its leaf's map, and
        // applications that render it themselves (VXGI, nv_asteroids) are unaffected. One light is served, the first
        // with a cascaded hint; its map takes its format, cascade count, resolution and channel from the hint
        // (LightShadows.h).
        bool serviceShadows = false;
        // The rasterizer state of the service's shadow depth passes (DepthPass::CreateParameters; its defaults).
        int shadowDepthBias = 0;
        float shadowDepthBiasClamp = 0.f;
        float shadowSlopeScaledDepthBias = 0.f;
        // A seam for device-free tests: makes the service shadows' textures when set (else the device does; without
        // either, served lights get a table entry but no map and keep their leaf's map).
        CreateShadowTextureFunction createShadowTexture = nullptr;
        void* createShadowTextureContext = nullptr;
    };

    // The last frame's counts.
    struct RenderServiceStats
    {
        uint32_t views = 0;
        uint32_t cascadeViews = 0;
        uint32_t visibleInstances = 0;   // summed over the views: an instance visible in two views counts twice
        uint32_t packets[c_DrawPacketPassCount] = {};   // per pass, summed over the views
        uint32_t lights = 0;
        uint32_t cameras = 0;
        uint32_t skinnedInstances = 0;
        uint32_t jointMatrices = 0;
        uint32_t serviceShadowMaps = 0;     // the table's entries that hold a map
        // With service shadows: lights whose hint the service does not serve (another kind, or a cascaded hint after
        // the first light's: LightShadows.h).
        uint32_t unservedShadowHints = 0;
        // Maps the extraction retired (reconfigured, or their light gone), released at EndFrame. A retired map's
        // texture can stay alive in the binding caches of DeferredLightingPass and ForwardShadingPass (their cached
        // sets name it) until the application calls their ResetBindingCache: a non-zero count is the signal to do so.
        uint32_t retiredShadowMaps = 0;
        // The shadow depth passes whose binding caches the last AddShadowPasses reset (after a frame replaced the
        // scene's buffers: SceneFrame::sceneBuffersReplaced, of this frame or of one whose AddShadowPasses did not
        // run or was refused).
        uint32_t shadowBindingCacheResets = 0;
        uint64_t snapshotBytes = 0;   // frame-arena bytes of the snapshot
        uint32_t skinningBindingSets = 0;   // the skinning pass's cached binding sets, aged by AddSkinningPasses
    };

    class RenderService : public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(RenderService)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        // `device` may be null (extraction only). Construction allocates nothing; the first Extract makes the state
        // and registers the RenderExtract systems: FE_OUT_OF_MEMORY if it cannot, FE_INVALID_ARGS when another service
        // is registered on the same scene graph (one per graph) or the scene is not on the graph path. The registration
        // is released with the service.
        RenderService(nvrhi::IDevice* device, const RenderServiceDesc& desc) noexcept;
        ~RenderService();

        RenderService(const RenderService&) = delete;
        RenderService& operator=(const RenderService&) = delete;

        // Called at the top of BuildGraph, after IRenderGraph::Reset and FrameData::Clear, and after the scene graph's
        // Refresh: sets the frame context and runs the graph scheduler's RenderExtract phase into a snapshot in
        // `graph`'s arena (open points 1 and 2). `frameData` (may be null) gets the SceneFrame entry; `outFrame` (may
        // be null) receives it. Opens the Debug render-read scope (open point 12) until EndFrame; a scope still open
        // from the previous frame (EndFrame not called) is closed first. A service serves the graph it was made for:
        // FE_INVALID_ARGS once the scene's graph was replaced (Scene::CreateSceneGraph, a new load); make a new
        // service then.
        nvrhi::FRESULT Extract(IRenderGraph* graph, FrameData* frameData, const ExtractDesc& desc,
                               SceneFrame** outFrame) noexcept;
        // The GpuScene upload (GpuSceneUploadPass); fills the buffer ids of `frame`. When it fails, or the frame never
        // executes, the records stay unacknowledged and the next frame sends every row again (GpuScene::CommitScatter).
        nvrhi::FRESULT AddUploadPasses(IRenderGraph* graph, SceneFrame* frame) noexcept;
        // Skinning (SkinningPass); fills the skinned vertex buffer ids of `frame`. Call it after AddUploadPasses, which
        // writes the skinned meshes' prototype data in its Prepare: without it, FE_INVALID_ARGS with a warning (for a
        // service with a scene, device and shader factory; one without adds nothing either way).
        nvrhi::FRESULT AddSkinningPasses(IRenderGraph* graph, SceneFrame* frame) noexcept;
        // A DepthPass use per cascade of the service shadow (one light: LightShadows.h), drawing the cascade view's
        // opaque packets through a bound adapter (PacketDrawStrategy) with `frame`'s bufferReads (the uses are then
        // Restore, GeometryPasses.h); one DepthPass object per cascade, so that a parallel recorder records cascades on
        // separate threads. Imports the shadow texture (host-kept, as CascadedShadowMap makes it), clears each
        // cascade's slice, and stores the version the last cascade wrote in the extracted light's shadowTexture, which
        // the lighting passes then read. Call it after AddUploadPasses and AddSkinningPasses (bufferReads) and before
        // the lighting passes: otherwise FE_INVALID_ARGS with a warning (checked for every service: those calls record
        // that they ran even when they add nothing; on the GPU path they must also have added their passes). With a
        // device, FE_INVALID_ARGS with a warning (once per service) unless RenderServiceDesc::shaderFactory and
        // commonPasses are both set; without a device the uses are added over uninitialised passes (device-free tests).
        // The pooled passes' binding caches are reset by the first call that runs after a frame replaced the scene's
        // buffers (RenderServiceStats::shadowBindingCacheResets). Nothing without RenderServiceDesc::serviceShadows, nor
        // for a second call in the frame. FE_INVALID_ARGS when no frame is open.
        nvrhi::FRESULT AddShadowPasses(IRenderGraph* graph, SceneFrame* frame) noexcept;
        // After the graph has executed (`executed`), or failed to compile or execute: closes the render-read scope (a
        // flag per service, not a count). Only an executed frame sets the leaves' skinningInitialized for its first-frame
        // skinned instances; otherwise the next frame dispatches them as first frames again. Releases the shadow maps
        // this frame's extraction retired (the frames that drew into them were submitted before it; nvrhi keeps a
        // texture alive for the command lists that reference it). The service's references go; a retired texture can
        // still be held by the binding caches of DeferredLightingPass and ForwardShadingPass, whose cached sets name the
        // shadow texture, until the application calls their ResetBindingCache (RenderServiceStats::retiredShadowMaps
        // non-zero is the signal; the service adds no mechanism for it). Idempotent per frame.
        // app::RenderGraphLayer::OnGraphExecuted is the place to call it; without the layer, call it after
        // IRenderGraph::Execute. A frame whose EndFrame never came counts as not executed. The snapshot dies at the
        // next Reset.
        void EndFrame(bool executed) noexcept;

        [[nodiscard]] const RenderServiceStats& GetStats() const noexcept;
        // The service shadows, keyed by light entity; null before the first Extract.
        [[nodiscard]] const LightShadowTable* GetLightShadows() const noexcept;

    private:
        struct Impl;
        nvrhi::IDevice* m_Device;   // not a reference: the scene the service serves holds the device
        RenderServiceDesc m_Desc;
        std::unique_ptr<Impl> m_Impl;
    };
}
