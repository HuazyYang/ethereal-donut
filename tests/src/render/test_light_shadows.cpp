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

// Shadow maps owned by the render service, keyed by light entity, without a device (aggregate
// docs/plans/2026-11-21-donut-ecs-e5.md, item E5.6; docs/ecs-design.md 7.6 and D13). The service makes its maps over
// stand-in textures (RenderServiceDesc::createShadowTexture), so that maps, cascade views, packets and the depth passes'
// uses exist without a GPU:
//
//   table_requests    the table alone: a failed extraction's requests cancelled (the next Sweep sees only its own); a
//                     changed channel keeps the map, a changed resolution, cascade count or format retires it
//   table_lifecycle   a light attached gets a map; it keeps it while nothing changes, also across a channel change; a
//                     reconfigured hint (another resolution) gets a new map and retires the old one; a cleared hint,
//                     Detach, SetLeaf to a mesh and destruction (the entity and its objects gone) free the map at the
//                     next extraction, and EndFrame releases it; of three suns with cascaded hints the first is served
//                     and the others get no shadow (one warning); an entity index recycled with a new generation does
//                     not inherit the map; at most one entry per entity, in entity order; hints the service does not
//                     serve (a spot light, a directional light whose hint is not a CascadedShadowMap) keep the leaf's
//                     map, with one warning; a service without the seam and device keeps entries without maps; service
//                     shadows off request nothing; a shadow view index out of range, or settings the view cannot take,
//                     set nothing up (one warning each); the service never writes a leaf or its hint
//   constants         the deferred-lighting and forward-shading light constants from the extracted lights equal, byte
//                     for byte, those from the leaf vector, and those equal a verbatim copy of the passes' loops before
//                     the factoring (light order, cascade order, numShadows, per-object indices, the light and shadow
//                     limits, a second shadow texture refused); with service shadows, the extracted lights' constants
//                     equal the leaf vector's whose hint was set up as the application sets its own map up; with two
//                     cascaded hints the deferred constants are produced (the leaf vector's are refused) and equal the
//                     leaf vector's without the second map
//   cascades          the service map's cascades equal a CascadedShadowMap set up by the application's call
//                     (SetupForPlanarView, and SetupForPlanarViewStable with settings.stable) over 8 frames of camera
//                     and sun moves: view frusta and every cascade's shadow constants, byte for byte; the copied
//                     lit-out-of-bounds and falloff distance; the cascade views' opaque packets equal the legacy
//                     InstancedOpaqueDrawStrategy walk of the legacy map's cascade views, as multisets
//   shadow_passes     AddShadowPasses: one Tracked DepthPass use per cascade, clearing and writing its slice of the
//                     imported texture; the version stored in the extracted light; DeferredLightingPass declaring the
//                     read of that texture (the dump's uses, and its transition where the profile records barriers)
//                     after every cascade, and no such read without extracted lights; Compile and VerifyPlan on every
//                     reference profile; a second call adds nothing; no open frame: FE_INVALID_ARGS; service shadows
//                     off: nothing
//   binding_resets    a buffer replacement in a frame whose AddShadowPasses was refused: the next call resets every
//                     pooled depth pass (uninitialised ones included)
//   gpu_pass_shape    the cascade uses with the frame's buffer reads (stand-in upload and skinning writers): Restore,
//                     the reads declared, after both writers; AddShadowPasses before AddUploadPasses or
//                     AddSkinningPasses refused
//   steady_state      10 frames after warm-up with moves, Extract, the add-pass calls: no operator new
//
//     test_light_shadows [case]

#include <donut/render/CascadedShadowMap.h>
#include <donut/render/DeferredLightingPass.h>
#include <donut/render/DepthPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/GeometryPasses.h>
#include <donut/render/LightShadows.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/render/RenderGraphLambdaPass.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include <donut/tests/utils.h>

using namespace donut::math;
#include <donut/shaders/deferred_lighting_cb.h>
#include <donut/shaders/forward_cb.h>

#include "LightingConstants.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <tuple>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <malloc.h>
#include <stdlib.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

namespace
{
    std::atomic<uint64_t> g_NewCount{ 0 };

    void* CountedAllocate(size_t size) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
        return std::malloc(size ? size : 1);
    }

    void* CountedAllocateAligned(size_t size, size_t alignment) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
        return _aligned_malloc(size ? size : 1, alignment);
#else
        return std::aligned_alloc(alignment, ((size ? size : 1) + alignment - 1) & ~(alignment - 1));
#endif
    }

    void CountedFreeAligned(void* memory) noexcept
    {
#if defined(_MSC_VER)
        _aligned_free(memory);
#else
        std::free(memory);
#endif
    }
}

void* operator new(size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new[](size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new(size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }
void operator delete(void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete[](void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete(void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }

using namespace donut;
using namespace donut::engine;
using namespace donut::render;

namespace
{
    std::atomic<uint32_t> g_Warnings{ 0 };

    // ------------------------------------------------------------------------------------------ stand-ins

    // A texture without a device: a description and a reference count. Counts the live ones.
    std::atomic<int> g_LiveTextures{ 0 };
    std::atomic<uint32_t> g_TexturesMade{ 0 };

    class StandInTexture : public nvrhi::ObjectImpl<nvrhi::ITexture>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(StandInTexture)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::ITexture)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IResource)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IRHIObject)
        NVRHI_END_INTERFACE_TABLE()

        explicit StandInTexture(const nvrhi::TextureDesc& desc) : m_Desc(desc) { ++g_LiveTextures; }
        ~StandInTexture() { --g_LiveTextures; }

        const nvrhi::TextureDesc& getDesc() const noexcept override { return m_Desc; }
        nvrhi::NativeObject getNativeView(nvrhi::ObjectType, nvrhi::Format, const nvrhi::TextureSubresourceSet&,
                                          nvrhi::TextureDimension, bool, const nvrhi::ComponentMapping*) noexcept override
        {
            return nullptr;
        }
        nvrhi::FRESULT queryHeapBinding(nvrhi::HeapBinding&) const noexcept override { return nvrhi::FE_NOT_FOUND; }

    private:
        nvrhi::TextureDesc m_Desc;
    };

    nvrhi::ITexture* CreateStandInTexture(void*, const nvrhi::TextureDesc& desc) noexcept
    {
        try
        {
            nvrhi::AutoPtr<StandInTexture> texture = MAKE_RC_OBJ_PTR(StandInTexture, desc);
            ++g_TexturesMade;
            return texture.Detach();
        }
        catch (...)
        {
            return nullptr;
        }
    }

    // An application's map over a stand-in texture, as CascadedShadowMap's device constructor makes it.
    nvrhi::AutoPtr<CascadedShadowMap> MakeHint(int resolution, int cascades, int perObject,
                                               nvrhi::Format format = nvrhi::Format::D32)
    {
        nvrhi::AutoPtr<StandInTexture> texture =
            MAKE_RC_OBJ_PTR(StandInTexture, CascadedShadowMap::GetTextureDesc(resolution, cascades, perObject, format));
        return MAKE_RC_OBJ_PTR(CascadedShadowMap, nullptr, texture.Get(), cascades, perObject);
    }

    // A shadow map that is not a CascadedShadowMap (the service does not serve it), with distinct constants per cascade.
    class OtherShadowMap : public IShadowMap
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        OtherShadowMap(uint32_t cascades, uint32_t perObject, float seed, nvrhi::ITexture* texture)
            : m_Cascades(cascades), m_PerObject(perObject), m_Seed(seed), m_Texture(texture) { }
        dm::float4x4 GetWorldToUvzwMatrix() const override { return dm::float4x4::identity(); }
        const ICompositeView& GetView() const override { return *m_View; }
        nvrhi::ITexture* GetTexture() const override { return m_Texture; }
        uint32_t GetNumberOfCascades() const override { return m_Cascades; }
        const IShadowMap* GetCascade(uint32_t) const override { return this; }
        uint32_t GetNumberOfPerObjectShadows() const override { return m_PerObject; }
        const IShadowMap* GetPerObjectShadow(uint32_t) const override { return this; }
        dm::int2 GetTextureSize() const override { return dm::int2(512, 256); }
        dm::box2 GetUVRange() const override { return dm::box2(dm::float2(0.f), dm::float2(1.f)); }
        dm::float2 GetFadeRangeInTexels() const override { return dm::float2(2.f); }
        bool IsLitOutOfBounds() const override { return true; }
        void FillShadowConstants(ShadowConstants& constants) const override
        {
            constants.matWorldToUvzwShadow = dm::float4x4::diagonal(m_Seed);
            constants.shadowFadeScale = dm::float2(m_Seed, 2.f * m_Seed);
            constants.shadowMapArrayIndex = int(m_Seed);
            constants.shadowFalloffDistance = 0.5f * m_Seed;
        }
    private:
        uint32_t m_Cascades;
        uint32_t m_PerObject;
        float m_Seed;
        nvrhi::ITexture* m_Texture;
        nvrhi::AutoPtr<PlanarView> m_View = MAKE_RC_OBJ_PTR(PlanarView);
    };

    // ------------------------------------------------------------------------------------------ the scene

    nvrhi::AutoPtr<SceneGraphNode> MakeNode(const std::string& name)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName(name);
        return node;
    }

    // A flat field of boxes (no node has children, so the legacy walk keeps nothing through a subgraph box), a few
    // transparent ones, a sun with a cascaded hint, and a camera view looking over the field.
    struct ShadowScene
    {
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        nvrhi::AutoPtr<Material> opaque = MAKE_RC_OBJ_PTR(Material);
        nvrhi::AutoPtr<Material> blended = MAKE_RC_OBJ_PTR(Material);
        nvrhi::AutoPtr<MeshInfo> box = MAKE_RC_OBJ_PTR(MeshInfo);
        nvrhi::AutoPtr<MeshInfo> glass = MAKE_RC_OBJ_PTR(MeshInfo);
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        nvrhi::AutoPtr<SceneGraphNode> sunNode;
        nvrhi::AutoPtr<DirectionalLight> sun;
        nvrhi::AutoPtr<CascadedShadowMap> sunHint;
        nvrhi::AutoPtr<PlanarView> view = MAKE_RC_OBJ_PTR(PlanarView);
        const IView* views[1] = {};

        ShadowScene()
        {
            opaque->name = "opaque";
            blended->name = "blended";
            blended->domain = MaterialDomain::AlphaBlended;
            auto makeMesh = [](MeshInfo* mesh, const char* name, Material* material)
            {
                mesh->name = name;
                mesh->buffers = MAKE_RC_OBJ_PTR(BufferGroup);
                auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
                geometry->material = material;
                geometry->numIndices = 36;
                geometry->numVertices = 8;
                geometry->objectSpaceBounds = dm::box3(dm::float3(-0.5f), dm::float3(0.5f, 2.f, 0.5f));
                mesh->geometries.push_back(geometry);
                mesh->objectSpaceBounds = geometry->objectSpaceBounds;
            };
            makeMesh(box, "box", opaque);
            makeMesh(glass, "glass", blended);

            graph->SetRootNode(MakeNode("root"));
            for (int i = 0; i < 400; ++i)
            {
                auto node = MakeNode("n" + std::to_string(i));
                node->SetTranslation(dm::double3(-30.0 + 3.0 * double(i % 20), 0.0, -10.0 + 3.0 * double(i / 20)));
                node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, i % 17 == 3 ? glass.Get() : box.Get()));
                graph->Attach(graph->GetRootNode(), node);
                nodes.push_back(node);
            }

            sunNode = MakeNode("sun");
            sun = MAKE_RC_OBJ_PTR(DirectionalLight);
            sun->angularSize = 0.53f;
            sun->irradiance = 2.f;
            sunNode->SetLeaf(sun);
            graph->Attach(graph->GetRootNode(), sunNode);
            sun->SetName("Sun");   // a leaf's name is its node's: set once attached
            sun->SetDirection(dm::double3(0.1, -1.0, 0.15));
            sunHint = MakeHint(1024, 4, 0);
            sun->shadowMap = sunHint;

            view->SetViewport(nvrhi::Viewport(1280.f, 720.f));
            views[0] = view.Get();
            Look(dm::float3(-8.f, 1.8f, 0.f), dm::float3(1.f, 2.5f, 0.f));
        }

        void Look(dm::float3 eye, dm::float3 at)
        {
            const dm::affine3 viewToWorld = dm::lookatZ(dm::normalize(at - eye)) * dm::translation(eye);
            view->SetMatrices(dm::inverse(viewToWorld),
                              dm::perspProjD3DStyleReverse(dm::PI_f * 0.25f, 1280.f / 720.f, 0.1f));
            view->UpdateCache();
        }

        Entity EntityOf(const SceneGraphLeaf* leaf) const { return graph->GetEntity(leaf->GetNode()); }

        ExtractDesc Desc(uint32_t frameIndex, const ShadowSettings& shadows = { 40.f, 40.f, 40.f, 4.f, false }) const
        {
            ExtractDesc desc;
            desc.views = views;
            desc.viewCount = 1;
            desc.frameIndex = frameIndex;
            desc.shadowViewIndex = 0;
            desc.shadows = shadows;
            return desc;
        }
    };

    struct RenderFrame
    {
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;
        FrameData data;

        RenderFrame()
        {
            CHECK(CreateResourceRegistry(nullptr, &registry) == nvrhi::FS_OK);
            CHECK(CreateRenderGraph(registry, &graph) == nvrhi::FS_OK);
        }

        void Begin()
        {
            graph->Reset();
            data.Clear();
        }
    };

    nvrhi::AutoPtr<RenderService> MakeService(SceneGraph* graph, bool serviceShadows, bool seam = true)
    {
        RenderServiceDesc desc;
        desc.sceneGraph = graph;
        desc.serviceShadows = serviceShadows;
        desc.shadowDepthBias = 100;
        desc.shadowSlopeScaledDepthBias = 4.f;
        if (seam)
            desc.createShadowTexture = CreateStandInTexture;
        return MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
    }

    SceneFrame* Extract(RenderService* service, RenderFrame& frame, const ExtractDesc& desc)
    {
        frame.Begin();
        SceneFrame* sceneFrame = nullptr;
        CHECK(service->Extract(frame.graph, &frame.data, desc, &sceneFrame) == nvrhi::FS_OK);
        CHECK(sceneFrame && sceneFrame->snapshot);
        return sceneFrame;
    }

    const ExtractedLight* FindLight(const RenderSnapshot& snapshot, const Light* leaf, uint32_t* index = nullptr)
    {
        for (uint32_t i = 0; i < snapshot.lightCount; ++i)
        {
            if (snapshot.lights[i].leaf == leaf)
            {
                if (index)
                    *index = i;
                return &snapshot.lights[i];
            }
        }
        return nullptr;
    }

    // The snapshot's cascade views of light `index`, in order.
    std::vector<const ExtractedView*> CascadeViews(const RenderSnapshot& snapshot, uint32_t index)
    {
        std::vector<const ExtractedView*> out;
        for (uint32_t v = snapshot.descViewCount; v < snapshot.viewCount; ++v)
        {
            if (snapshot.views[v].light == index)
                out.push_back(&snapshot.views[v]);
        }
        return out;
    }

    template <typename T>
    bool BytesEqual(const T& a, const T& b)
    {
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }

    uint32_t ReferenceCount(nvrhi::IObject* object)
    {
        object->AddRef();
        return uint32_t(object->Release());
    }

    // One pass of a compiled plan's JSON dump (RenderGraphDump.cpp), in schedule order: its name, mode ("native",
    // "tracked", "restore"), its declared uses and the barriers before its body, as the dump's text.
    struct DumpedPass
    {
        std::string name;
        std::string mode;
        std::string uses;
        std::string prologue;
    };

    std::string DumpField(const std::string& line, const char* name)
    {
        const std::string key = std::string("\"") + name + "\": \"";
        const size_t at = line.find(key);
        if (at == std::string::npos)
            return std::string();
        const size_t begin = at + key.size();
        return line.substr(begin, line.find('"', begin) - begin);
    }

    std::string DumpSection(const std::string& line, const char* from, const char* to)
    {
        const size_t begin = line.find(from);
        if (begin == std::string::npos)
            return std::string();
        const size_t end = line.find(to, begin);
        return line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    }

    std::vector<DumpedPass> DumpPasses(ICompiledRenderGraph* plan)
    {
        nvrhi::string dump;
        CHECK(plan->WriteDump(DumpFormat::Json, dump) == nvrhi::FS_OK);
        const std::string text(dump.c_str());
        const size_t at = text.find("\"passes\": [");
        CHECK(at != std::string::npos);
        const size_t end = text.find("\n  ],", at);
        std::vector<DumpedPass> passes;
        for (size_t line = text.find('\n', at) + 1; line < end;)
        {
            const size_t lineEnd = std::min(text.find('\n', line), end);
            const std::string l = text.substr(line, lineEnd - line);
            DumpedPass pass;
            pass.name = DumpField(l, "name");
            pass.mode = DumpField(l, "mode");
            pass.uses = DumpSection(l, "\"uses\": [", "\"prologue\": [");
            pass.prologue = DumpSection(l, "\"prologue\": [", "\"postClear\": [");
            passes.push_back(pass);
            line = lineEnd + 1;
        }
        return passes;
    }

    // ------------------------------------------------------------------------------------------ table_requests

    // The table's primitives without a service: an extraction that fails before its Sweep cancels its requests, so that
    // a light gone by the next extraction does not keep its map an extra frame; a changed channel keeps the map; a
    // changed resolution, cascade count or format retires it.
    void test_table_requests()
    {
        const int texturesAtStart = g_LiveTextures.load();
        LightShadowTable table;
        const LightShadow shadow{ 0, 4, 1024, nvrhi::Format::D32 };
        const Entity a = MakeEntity(3, 1), b = MakeEntity(5, 1);
        auto request = [&](Entity light, const LightShadow& wanted) -> LightShadowEntry*
        {
            LightShadowEntry* entry = table.Request(light, wanted, LightShadowKind::Cascaded);
            CHECK(entry && entry->light == light);
            if (!entry->map)
                entry->map = MakeHint(int(wanted.resolution), int(wanted.cascadeCount), 0, wanted.format).Detach();
            return entry;
        };
        request(a, shadow);
        request(b, shadow);
        CHECK(table.Sweep() == 0 && table.GetCount() == 2);
        const IShadowMap* mapA = table.Find(a)->map;

        // a failed extraction that wanted both, cancelled; the next wants `a` only: `b` leaves at once
        request(a, shadow);
        request(b, shadow);
        table.CancelRequests();
        request(a, shadow);
        CHECK(table.Sweep() == 1 && table.GetCount() == 1 && table.Find(b) == nullptr && table.GetRetiredCount() == 1);
        // both wanted again; then, for contrast, the same failure without the cancel: the failed extraction's request
        // keeps `b` one Sweep longer
        request(b, shadow);
        request(a, shadow);
        CHECK(table.Sweep() == 0 && table.GetCount() == 2);
        request(a, shadow);
        request(b, shadow);   // the failed extraction: no Sweep, no cancel
        request(a, shadow);   // the next wants `a` only
        CHECK(table.Sweep() == 0 && table.GetCount() == 2);   // `b` kept
        table.ReleaseRetired();
        CHECK(table.Find(a)->map == mapA);
        mapA = table.Find(a)->map;

        // another channel: the same map, the channel in place; another resolution, cascade count or format: a new map
        LightShadow other = shadow;
        other.channel = 3;
        CHECK(request(a, other)->map == mapA && table.Find(a)->shadow.channel == 3 && table.GetRetiredCount() == 0);
        other.resolution = 512;
        CHECK(request(a, other)->map != mapA && table.GetRetiredCount() == 1);
        mapA = table.Find(a)->map;
        other.cascadeCount = 2;
        CHECK(request(a, other)->map != mapA && table.GetRetiredCount() == 2);
        mapA = table.Find(a)->map;
        other.format = nvrhi::Format::D24S8;
        CHECK(request(a, other)->map != mapA && table.GetRetiredCount() == 3);
        request(b, shadow);
        CHECK(table.Sweep() == 0);
        table.ReleaseRetired();
        CHECK(g_LiveTextures.load() == texturesAtStart + 2);   // the two entries' maps
    }

    // ------------------------------------------------------------------------------------------ table_lifecycle

    void test_table_lifecycle()
    {
        ShadowScene scene;
        RenderFrame frame;
        nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
        const int texturesAtStart = g_LiveTextures.load();   // the hint's
        uint32_t frameIndex = 0;

        auto extract = [&]() -> const RenderSnapshot&
        {
            scene.graph->Refresh(++frameIndex);
            return *Extract(service, frame, scene.Desc(frameIndex))->snapshot;
        };
        auto mapOf = [&](Entity entity) -> const IShadowMap*
        {
            const LightShadowEntry* entry = service->GetLightShadows()->Find(entity);
            return entry ? entry->map : nullptr;
        };
        auto entriesOrdered = [&]()
        {
            const LightShadowTable* table = service->GetLightShadows();
            for (uint32_t i = 1; i < table->GetCount(); ++i)
            {
                const Entity a = table->GetEntries()[i - 1].light, b = table->GetEntries()[i].light;
                if (!(GetEntityIndex(a) < GetEntityIndex(b) ||
                      (GetEntityIndex(a) == GetEntityIndex(b) && GetEntityGeneration(a) < GetEntityGeneration(b))))
                    return false;
            }
            return true;
        };

        // 1: the sun gets a map of its hint's configuration
        const RenderSnapshot* snapshot = &extract();
        const Entity sunEntity = scene.EntityOf(scene.sun);
        uint32_t sunIndex = 0;
        const ExtractedLight* light = FindLight(*snapshot, scene.sun, &sunIndex);
        CHECK(light && light->serviceShadow && light->shadowMap && light->shadowMap != scene.sunHint.Get());
        CHECK(light->entity == sunEntity && light->shadowTexture == InvalidResourceId);
        const IShadowMap* first = mapOf(sunEntity);
        CHECK(first == light->shadowMap && service->GetLightShadows()->GetCount() == 1);
        const LightShadowEntry* entry = service->GetLightShadows()->Find(sunEntity);
        CHECK(entry->kind == LightShadowKind::Cascaded && entry->shadow.cascadeCount == 4 &&
              entry->shadow.resolution == 1024 && entry->shadow.format == nvrhi::Format::D32 &&
              entry->shadow.channel == scene.sun->shadowChannel);
        CHECK(first->GetTexture() != scene.sunHint->GetTexture());
        const nvrhi::TextureDesc& serviceDesc = first->GetTexture()->getDesc();
        const nvrhi::TextureDesc legacyDesc = CascadedShadowMap::GetTextureDesc(1024, 4, 0, nvrhi::Format::D32);
        CHECK(serviceDesc.width == legacyDesc.width && serviceDesc.height == legacyDesc.height &&
              serviceDesc.arraySize == legacyDesc.arraySize && serviceDesc.format == legacyDesc.format &&
              serviceDesc.dimension == legacyDesc.dimension && serviceDesc.isRenderTarget &&
              serviceDesc.isTypeless && serviceDesc.keepInitialState &&
              serviceDesc.initialState == nvrhi::ResourceStates::ShaderResource);
        CHECK(snapshot->viewCount == 5 && snapshot->descViewCount == 1);
        const auto cascades = CascadeViews(*snapshot, sunIndex);
        CHECK(cascades.size() == 4);
        for (uint32_t c = 0; c < 4; ++c)
        {
            CHECK(cascades[c]->cascade == c);
            CHECK(cascades[c]->view == static_cast<const CascadedShadowMap*>(first)->GetCascade(c)->GetView()
                                           .GetChildView(ViewType::PLANAR, 0));
        }
        CHECK(service->GetStats().serviceShadowMaps == 1 && service->GetStats().cascadeViews == 4);
        CHECK(g_LiveTextures.load() == texturesAtStart + 1);
        // the service never sets the hint up, nor changes the leaf
        CHECK(scene.sunHint->GetNumberOfCascades() == 0 && scene.sun->shadowMap.Get() == scene.sunHint.Get());
        service->EndFrame(true);

        // 2: nothing changed: the same map, nothing retired
        snapshot = &extract();
        CHECK(mapOf(sunEntity) == first && service->GetStats().retiredShadowMaps == 0);
        CHECK(service->GetStats().unservedShadowHints == 0);
        service->EndFrame(true);

        // 2b: another channel only: the same map and texture, the entry's channel updated in place, nothing retired
        {
            const int channel = scene.sun->shadowChannel;
            const uint32_t made = g_TexturesMade.load();
            for (int next : { channel + 3, channel })
            {
                scene.sun->shadowChannel = next;
                snapshot = &extract();
                CHECK(mapOf(sunEntity) == first && service->GetLightShadows()->Find(sunEntity)->shadow.channel == next);
                CHECK(service->GetStats().retiredShadowMaps == 0 && g_TexturesMade.load() == made);
                CHECK(FindLight(*snapshot, scene.sun)->constants->shadowChannel.x == next);
                service->EndFrame(true);
            }
        }

        // 3: a hint of another resolution: a new map, the old one retired, then released at EndFrame
        const uint32_t madeBefore = g_TexturesMade.load();
        scene.sun->shadowMap = MakeHint(512, 4, 0);
        snapshot = &extract();
        const IShadowMap* second = mapOf(sunEntity);
        CHECK(second && second->GetTextureSize().x == 512 && g_TexturesMade.load() == madeBefore + 1);
        CHECK(service->GetStats().retiredShadowMaps == 1 && service->GetLightShadows()->GetRetiredCount() == 1);
        CHECK(FindLight(*snapshot, scene.sun)->shadowMap == second);
        service->EndFrame(true);
        CHECK(service->GetLightShadows()->GetRetiredCount() == 0);
        CHECK(g_LiveTextures.load() == texturesAtStart + 2);   // the 512 hint and the service's 512 map
        scene.sun->shadowMap = scene.sunHint;
        snapshot = &extract();   // back to the first configuration: a third map
        CHECK(mapOf(sunEntity) && mapOf(sunEntity)->GetTextureSize().x == 1024);
        service->EndFrame(true);
        CHECK(g_LiveTextures.load() == texturesAtStart + 1);

        // 4: the hint cleared: no service shadow; the entry leaves at this extraction
        scene.sun->shadowMap = nullptr;
        snapshot = &extract();
        light = FindLight(*snapshot, scene.sun);
        CHECK(light && !light->serviceShadow && light->shadowMap == nullptr && snapshot->viewCount == 1);
        CHECK(service->GetLightShadows()->GetCount() == 0 && service->GetStats().retiredShadowMaps == 1);
        service->EndFrame(true);
        CHECK(g_LiveTextures.load() == texturesAtStart);
        scene.sun->shadowMap = scene.sunHint;

        // 5: three suns with cascaded hints (one hint shared by two): the first in GetLights() order takes the one
        // service shadow, the others are withheld (no shadow in the snapshot, counted, one warning per service); then
        // Detach, SetLeaf to a mesh and destruction of the served light each free its entry at the next extraction, and
        // the next sun takes the shadow. At most one entry per entity, in entity order.
        auto secondNode = MakeNode("sun2");
        auto secondSun = MAKE_RC_OBJ_PTR(DirectionalLight);
        secondNode->SetLeaf(secondSun);
        scene.graph->Attach(scene.graph->GetRootNode(), secondNode);
        secondSun->SetDirection(dm::double3(-0.3, -1.0, 0.2));
        secondSun->shadowMap = scene.sunHint;   // the sun's hint
        auto thirdNode = MakeNode("sun3");
        auto thirdSun = MAKE_RC_OBJ_PTR(DirectionalLight);
        thirdNode->SetLeaf(thirdSun);
        scene.graph->Attach(scene.graph->GetRootNode(), thirdNode);
        thirdSun->SetDirection(dm::double3(0.0, -1.0, 0.0));
        thirdSun->shadowMap = MakeHint(256, 2, 0);   // held by the leaf only
        {
            const uint32_t before = g_Warnings.load();
            snapshot = &extract();
            CHECK(g_Warnings.load() == before + 1);
            g_Warnings -= 1;
        }
        CHECK(snapshot->lightCount == 3 && snapshot->lights[0].leaf == scene.sun.Get());   // GetLights() order
        CHECK(service->GetLightShadows()->GetCount() == 1 && mapOf(sunEntity) != nullptr && entriesOrdered());
        CHECK(snapshot->viewCount == 1 + 4 && service->GetStats().serviceShadowMaps == 1);
        CHECK(service->GetStats().unservedShadowHints == 2 && FindLight(*snapshot, scene.sun)->serviceShadow);
        for (const Light* withheld : { static_cast<const Light*>(secondSun.Get()), static_cast<const Light*>(thirdSun.Get()) })
            CHECK(!FindLight(*snapshot, withheld)->serviceShadow && FindLight(*snapshot, withheld)->shadowMap == nullptr);
        service->EndFrame(true);
        {
            const uint32_t before = g_Warnings.load();
            snapshot = &extract();   // the warning came once
            CHECK(g_Warnings.load() == before && service->GetStats().unservedShadowHints == 2);
            service->EndFrame(true);
        }

        // Detach: the served sun's node detached (and released; its leaf stays): its entry leaves, the second sun
        // takes the shadow with a map of its own
        const Entity oldSun = sunEntity;
        const Entity secondEntity = scene.EntityOf(secondSun);
        scene.graph->Detach(scene.sunNode);
        scene.sunNode = nullptr;
        snapshot = &extract();
        CHECK(FindLight(*snapshot, scene.sun) == nullptr && service->GetLightShadows()->Find(oldSun) == nullptr);
        CHECK(service->GetLightShadows()->GetCount() == 1 && mapOf(secondEntity) != nullptr);
        CHECK(FindLight(*snapshot, secondSun)->serviceShadow && FindLight(*snapshot, secondSun)->shadowMap == mapOf(secondEntity));
        CHECK(service->GetStats().retiredShadowMaps == 1 && service->GetStats().unservedShadowHints == 1);
        CHECK(snapshot->viewCount == 5);
        service->EndFrame(true);

        // SetLeaf to a mesh: the second sun's entry leaves, the third sun takes the shadow (two cascades)
        secondNode->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, scene.box.Get()));
        const Entity thirdEntity = scene.EntityOf(thirdSun);
        snapshot = &extract();
        CHECK(service->GetLightShadows()->GetCount() == 1 && service->GetLightShadows()->Find(secondEntity) == nullptr);
        CHECK(mapOf(thirdEntity) != nullptr && mapOf(thirdEntity)->GetTextureSize().x == 256);
        CHECK(service->GetStats().retiredShadowMaps == 1 && service->GetStats().unservedShadowHints == 0);
        CHECK(snapshot->viewCount == 1 + 2);
        service->EndFrame(true);

        // destruction: the third sun's node detached and every reference to the node and its leaf released, so that
        // the entity is destroyed with its objects (its hint, held by the leaf only, goes with them); the entry leaves
        // at the next extraction and its map is released at EndFrame
        IWorld* world = scene.graph->GetWorld();
        CHECK(world->IsAlive(thirdEntity));
        scene.graph->Detach(thirdNode);
        thirdSun = nullptr;
        CHECK(ReferenceCount(thirdNode) == 1);   // ours: releasing it destroys the node and the leaf
        thirdNode = nullptr;
        CHECK(!world->IsAlive(thirdEntity));
        snapshot = &extract();
        CHECK(service->GetLightShadows()->GetCount() == 0 && service->GetStats().retiredShadowMaps == 1);
        CHECK(snapshot->lightCount == 0 && snapshot->viewCount == 1);   // the second sun's node holds a mesh now
        service->EndFrame(true);
        CHECK(g_LiveTextures.load() == texturesAtStart);   // the leaf's hint and the service's map are gone

        // 6: an entity index recycled with a new generation does not inherit a map. The sun, with a map, is detached;
        // within the same frame lights without a hint are attached until one takes the sun's freed index (the order
        // in which the world hands out free indices is its own), and that one gets the sun's hint.
        scene.sunNode = MakeNode("sun");
        scene.sunNode->SetLeaf(scene.sun);
        scene.graph->Attach(scene.graph->GetRootNode(), scene.sunNode);
        scene.sun->SetDirection(dm::double3(0.1, -1.0, 0.15));
        snapshot = &extract();
        const Entity reattached = scene.EntityOf(scene.sun);
        CHECK(reattached != oldSun && mapOf(reattached) != nullptr);
        service->EndFrame(true);
        const uint32_t made = g_TexturesMade.load();
        scene.graph->Detach(scene.sunNode);
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> fillers;
        nvrhi::AutoPtr<SceneGraphNode> otherNode;
        nvrhi::AutoPtr<DirectionalLight> other;
        for (uint32_t attempt = 0; attempt < 256 && !other; ++attempt)
        {
            auto node = MakeNode("other" + std::to_string(attempt));
            auto candidate = MAKE_RC_OBJ_PTR(DirectionalLight);
            node->SetLeaf(candidate);
            scene.graph->Attach(scene.graph->GetRootNode(), node);
            candidate->SetDirection(dm::double3(0.1, -1.0, 0.15));
            if (GetEntityIndex(scene.graph->GetEntity(node)) == GetEntityIndex(reattached))
            {
                otherNode = node;
                other = candidate;
            }
            else
                fillers.push_back(node);
        }
        CHECK(other);
        other->shadowMap = scene.sunHint;   // the same configuration as the sun's
        snapshot = &extract();
        const Entity recycled = scene.EntityOf(other);
        std::printf("  detached sun %u:%u, the next light %u:%u\n", GetEntityIndex(reattached),
                    GetEntityGeneration(reattached), GetEntityIndex(recycled), GetEntityGeneration(recycled));
        CHECK(GetEntityIndex(recycled) == GetEntityIndex(reattached));   // the index is recycled...
        CHECK(GetEntityGeneration(recycled) != GetEntityGeneration(reattached));   // ...with a new generation
        CHECK(service->GetLightShadows()->GetCount() == 1 && service->GetLightShadows()->Find(reattached) == nullptr);
        CHECK(mapOf(recycled) != nullptr && g_TexturesMade.load() == made + 1);   // a map of its own
        CHECK(service->GetStats().retiredShadowMaps == 1);
        service->EndFrame(true);
        scene.graph->Detach(otherNode);
        for (const auto& node : fillers)
            scene.graph->Detach(node);

        // 7: hints the service does not serve: a spot light, a directional light whose hint is not cascaded
        scene.graph->Attach(scene.graph->GetRootNode(), scene.sunNode);
        auto spotNode = MakeNode("spot");
        auto spot = MAKE_RC_OBJ_PTR(SpotLight);
        spotNode->SetLeaf(spot);
        scene.graph->Attach(scene.graph->GetRootNode(), spotNode);
        spot->shadowMap = MakeHint(256, 1, 0);
        auto planarNode = MakeNode("planar");
        auto planar = MAKE_RC_OBJ_PTR(DirectionalLight);
        planarNode->SetLeaf(planar);
        scene.graph->Attach(scene.graph->GetRootNode(), planarNode);
        planar->SetDirection(dm::double3(0.0, -1.0, 0.0));
        auto otherMap = MAKE_RC_OBJ_PTR(OtherShadowMap, 1u, 0u, 1.f, scene.sunHint->GetTexture());
        planar->shadowMap = otherMap;
        const uint32_t warnings = g_Warnings.load();
        snapshot = &extract();
        CHECK(g_Warnings.load() == warnings + 1);   // once per service
        CHECK(service->GetStats().unservedShadowHints == 2 && service->GetLightShadows()->GetCount() == 1);
        CHECK(FindLight(*snapshot, spot)->shadowMap == spot->shadowMap.Get() && !FindLight(*snapshot, spot)->serviceShadow);
        CHECK(FindLight(*snapshot, planar)->shadowMap == otherMap.Get());
        CHECK(FindLight(*snapshot, scene.sun)->serviceShadow);
        service->EndFrame(true);
        snapshot = &extract();
        CHECK(g_Warnings.load() == warnings + 1);
        service->EndFrame(true);
        g_Warnings -= 1;
        scene.graph->Detach(spotNode);
        scene.graph->Detach(planarNode);

        // 8: a shadow view index out of range: entries and maps kept, nothing set up, the leaf's map carried
        {
            scene.graph->Refresh(++frameIndex);
            ExtractDesc desc = scene.Desc(frameIndex);
            desc.shadowViewIndex = 3;
            const uint32_t before = g_Warnings.load();
            const RenderSnapshot& s = *Extract(service, frame, desc)->snapshot;
            CHECK(g_Warnings.load() == before + 1 && s.viewCount == 1);
            CHECK(!FindLight(s, scene.sun)->serviceShadow && FindLight(s, scene.sun)->shadowMap == scene.sunHint.Get());
            CHECK(service->GetLightShadows()->GetCount() == 1 && mapOf(scene.EntityOf(scene.sun)) != nullptr);
            service->EndFrame(true);
            g_Warnings -= 1;
        }

        // 8b: settings the shadow view cannot take (what SetupForPlanarView asserts): an exponent of 1 or NaN, and no
        // maxShadowDistance with a view whose far plane is at infinity, plain and stable. Nothing set up, no assert
        // (Debug), entries and maps kept, the leaf's map carried, one warning per service; then served again.
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            const ShadowSettings invalid[] = { { 40.f, 40.f, 40.f, 1.f, false }, { 40.f, 40.f, 40.f, nan, true },
                                               { 0.f, 40.f, 40.f, 4.f, false }, { 0.f, 40.f, 40.f, 4.f, true } };
            const uint32_t before = g_Warnings.load();
            for (const ShadowSettings& settings : invalid)
            {
                CHECK(!AreShadowSettingsValid(scene.view, settings));
                scene.graph->Refresh(++frameIndex);
                const RenderSnapshot& s = *Extract(service, frame, scene.Desc(frameIndex, settings))->snapshot;
                CHECK(s.viewCount == 1 && !FindLight(s, scene.sun)->serviceShadow);
                CHECK(FindLight(s, scene.sun)->shadowMap == scene.sunHint.Get());
                CHECK(service->GetLightShadows()->GetCount() == 1 && mapOf(scene.EntityOf(scene.sun)) != nullptr);
                service->EndFrame(true);
            }
            CHECK(g_Warnings.load() == before + 1);
            g_Warnings -= 1;
            // a finite far plane needs no maxShadowDistance; the defaults are valid; no view is not
            auto finite = MAKE_RC_OBJ_PTR(PlanarView);
            finite->SetViewport(nvrhi::Viewport(1280.f, 720.f));
            finite->SetMatrices(scene.view->GetViewMatrix(),
                                dm::perspProjD3DStyle(dm::PI_f * 0.25f, 1280.f / 720.f, 0.1f, 200.f));
            finite->UpdateCache();
            CHECK(AreShadowSettingsValid(finite, { 0.f, 40.f, 40.f, 4.f, false }));
            CHECK(AreShadowSettingsValid(finite, { 0.f, 40.f, 40.f, 4.f, true }));
            CHECK(AreShadowSettingsValid(scene.view, ShadowSettings{}) && !AreShadowSettingsValid(nullptr, ShadowSettings{}));
            snapshot = &extract();
            CHECK(FindLight(*snapshot, scene.sun)->serviceShadow && snapshot->viewCount == 5);
            service->EndFrame(true);
        }

        // 9: the service released: every map goes (the hints made since the start released first)
        spot->shadowMap = nullptr;
        service = nullptr;
        CHECK(g_LiveTextures.load() == texturesAtStart);

        // without the seam and a device: entries, no maps, the leaf's map carried
        {
            nvrhi::AutoPtr<RenderService> plain = MakeService(scene.graph, true, false);
            scene.graph->Refresh(++frameIndex);
            const RenderSnapshot& s = *Extract(plain, frame, scene.Desc(frameIndex))->snapshot;
            CHECK(plain->GetLightShadows()->GetCount() == 1 && plain->GetLightShadows()->GetEntries()[0].map == nullptr);
            CHECK(!FindLight(s, scene.sun)->serviceShadow && FindLight(s, scene.sun)->shadowMap == scene.sunHint.Get());
            CHECK(s.viewCount == 1 && plain->GetStats().serviceShadowMaps == 0);
            plain->EndFrame(true);
        }
        // service shadows off: nothing requested, the leaf's map carried
        {
            nvrhi::AutoPtr<RenderService> off = MakeService(scene.graph, false);
            scene.graph->Refresh(++frameIndex);
            const RenderSnapshot& s = *Extract(off, frame, scene.Desc(frameIndex))->snapshot;
            CHECK(off->GetLightShadows()->GetCount() == 0 && s.viewCount == 1);
            CHECK(!FindLight(s, scene.sun)->serviceShadow && FindLight(s, scene.sun)->shadowMap == scene.sunHint.Get());
            CHECK(g_LiveTextures.load() == texturesAtStart);
            off->EndFrame(true);
        }
    }

    // ------------------------------------------------------------------------------------------ constants

    // DeferredLightingPass::Render's light loop before the factoring (ethereal-donut 8b94988), verbatim but for the
    // early return's error log (counted as a refusal here).
    bool LegacyDeferredLights(const std::vector<nvrhi::AutoPtr<Light>>* lights, DeferredLightingConstants& deferredConstants,
                              nvrhi::ITexture*& shadowMapTexture)
    {
        int numShadows = 0;

        if (lights)
        {
            for (const auto& light : *lights)
            {
                if (light->shadowMap)
                {
                    if (!shadowMapTexture)
                    {
                        shadowMapTexture = light->shadowMap->GetTexture();
                        deferredConstants.shadowMapTextureSize = float2(light->shadowMap->GetTextureSize());
                    }
                    else
                    {
                        if (shadowMapTexture != light->shadowMap->GetTexture())
                            return false;
                    }
                }

                if (deferredConstants.numLights >= DEFERRED_MAX_LIGHTS)
                {
                    log::warning("Maximum number of active lights (%d) exceeded in DeferredLightingPass",
                        DEFERRED_MAX_LIGHTS);
                    break;
                }

                LightConstants& lightConstants = deferredConstants.lights[deferredConstants.numLights];
                light->FillLightConstants(lightConstants);

                if (light->shadowMap)
                {
                    for (uint32_t cascade = 0; cascade < light->shadowMap->GetNumberOfCascades(); cascade++)
                    {
                        if (numShadows < DEFERRED_MAX_SHADOWS)
                        {
                            light->shadowMap->GetCascade(cascade)->FillShadowConstants(deferredConstants.shadows[numShadows]);
                            lightConstants.shadowCascades[cascade] = numShadows;
                            ++numShadows;
                        }
                    }

                    for (uint32_t perObjectShadow = 0; perObjectShadow < light->shadowMap->GetNumberOfPerObjectShadows(); perObjectShadow++)
                    {
                        if (numShadows < DEFERRED_MAX_SHADOWS)
                        {
                            light->shadowMap->GetPerObjectShadow(perObjectShadow)->FillShadowConstants(deferredConstants.shadows[numShadows]);
                            lightConstants.perObjectShadows[perObjectShadow] = numShadows;
                            ++numShadows;
                        }
                    }
                }

                ++deferredConstants.numLights;
            }
        }
        return true;
    }

    // ForwardShadingPass::PrepareLights' light loop before the factoring (ethereal-donut 8b94988), verbatim.
    void LegacyForwardLights(const std::vector<nvrhi::AutoPtr<Light>>& lights, ForwardShadingLightConstants& constants,
                             nvrhi::ITexture*& shadowMapTexture)
    {
        int2 shadowMapTextureSize = 0;
        for (const auto& light : lights)
        {
            if (light->shadowMap)
            {
                shadowMapTexture = light->shadowMap->GetTexture();
                shadowMapTextureSize = light->shadowMap->GetTextureSize();
                break;
            }
        }

        constants.shadowMapTextureSize = float2(shadowMapTextureSize);
        constants.shadowMapTextureSizeInv = 1.f / constants.shadowMapTextureSize;

        int numShadows = 0;

        for (int nLight = 0; nLight < std::min(static_cast<int>(lights.size()), FORWARD_MAX_LIGHTS); nLight++)
        {
            const auto& light = lights[nLight];

            LightConstants& lightConstants = constants.lights[constants.numLights];
            light->FillLightConstants(lightConstants);

            if (light->shadowMap)
            {
                for (uint32_t cascade = 0; cascade < light->shadowMap->GetNumberOfCascades(); cascade++)
                {
                    if (numShadows < FORWARD_MAX_SHADOWS)
                    {
                        light->shadowMap->GetCascade(cascade)->FillShadowConstants(constants.shadows[numShadows]);
                        lightConstants.shadowCascades[cascade] = numShadows;
                        ++numShadows;
                    }
                }

                for (uint32_t perObjectShadow = 0; perObjectShadow < light->shadowMap->GetNumberOfPerObjectShadows(); perObjectShadow++)
                {
                    if (numShadows < FORWARD_MAX_SHADOWS)
                    {
                        light->shadowMap->GetPerObjectShadow(perObjectShadow)->FillShadowConstants(constants.shadows[numShadows]);
                        lightConstants.perObjectShadows[perObjectShadow] = numShadows;
                        ++numShadows;
                    }
                }
            }

            ++constants.numLights;
        }
    }

    // The three forms of each pass's constants for the graph's lights and the snapshot's: equal bytes; the deferred
    // refusal of a second texture in all three. Returns whether the deferred forms were refused.
    bool CompareConstants(const std::vector<nvrhi::AutoPtr<Light>>& lights, const RenderSnapshot& snapshot,
                          const char* what)
    {
        CHECK(snapshot.lightCount == lights.size());
        for (uint32_t i = 0; i < snapshot.lightCount; ++i)
            CHECK(snapshot.lights[i].leaf == lights[i].Get());   // GetLights() order

        static DeferredLightingConstants deferred[3];
        std::memset(deferred, 0, sizeof(deferred));
        nvrhi::ITexture* deferredTextures[3] = {};
        const bool legacy = LegacyDeferredLights(&lights, deferred[0], deferredTextures[0]);
        const bool leaf = lighting::FillDeferredLights(lighting::LeafLights{ &lights }, deferred[1], DEFERRED_MAX_LIGHTS,
                                                       DEFERRED_MAX_SHADOWS, deferredTextures[1]);
        const bool extracted = lighting::FillDeferredLights(lighting::SnapshotLights{ snapshot.lights, snapshot.lightCount },
                                                            deferred[2], DEFERRED_MAX_LIGHTS, DEFERRED_MAX_SHADOWS,
                                                            deferredTextures[2]);
        CHECK(legacy == leaf && leaf == extracted);
        if (legacy)
        {
            CHECK(BytesEqual(deferred[0], deferred[1]));
            CHECK(BytesEqual(deferred[1], deferred[2]));
            CHECK(deferredTextures[0] == deferredTextures[1] && deferredTextures[1] == deferredTextures[2]);
        }

        static ForwardShadingLightConstants forward[3];
        std::memset(forward, 0, sizeof(forward));
        nvrhi::ITexture* forwardTextures[3] = {};
        LegacyForwardLights(lights, forward[0], forwardTextures[0]);
        lighting::FillForwardLights(lighting::LeafLights{ &lights }, forward[1], FORWARD_MAX_LIGHTS, FORWARD_MAX_SHADOWS,
                                    forwardTextures[1]);
        lighting::FillForwardLights(lighting::SnapshotLights{ snapshot.lights, snapshot.lightCount }, forward[2],
                                    FORWARD_MAX_LIGHTS, FORWARD_MAX_SHADOWS, forwardTextures[2]);
        CHECK(BytesEqual(forward[0], forward[1]));
        CHECK(BytesEqual(forward[1], forward[2]));
        CHECK(forwardTextures[0] == forwardTextures[1] && forwardTextures[1] == forwardTextures[2]);
        std::printf("  %s: %u lights, deferred %s (%u lights), forward %u lights\n", what, snapshot.lightCount,
                    legacy ? "equal" : "refused in every form", deferred[0].numLights, forward[0].numLights);
        return !legacy;
    }

    void test_constants()
    {
        ShadowScene scene;
        RenderFrame frame;
        uint32_t frameIndex = 0;
        SceneGraphNode* root = scene.graph->GetRootNode();
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> lightNodes;
        std::vector<nvrhi::AutoPtr<Light>> extra;
        auto add = [&](nvrhi::AutoPtr<Light> light, dm::double3 position, dm::double3 direction)
        {
            auto node = MakeNode("light" + std::to_string(lightNodes.size()));
            node->SetTranslation(position);
            node->SetLeaf(light);
            scene.graph->Attach(root, node);
            light->SetDirection(direction);
            light->color = dm::float3(0.1f * float(lightNodes.size()), 0.5f, 1.f);
            light->shadowChannel = int(lightNodes.size() % 5) - 1;
            lightNodes.push_back(node);
            extra.push_back(light);
        };

        // the sun's hint shared by a second directional light (one texture), a cascaded hint with per-object shadows
        // on the same texture, spot and point lights, one with a shadow map of another kind on that texture too
        nvrhi::AutoPtr<CascadedShadowMap> shared = MakeHint(1024, 4, 2);
        shared->SetupForPlanarView(*scene.sun, scene.view->GetViewFrustum(), 40.f, 40.f, 40.f);
        for (uint32_t i = 0; i < 2; ++i)
            shared->SetupPerObjectShadow(*scene.sun, i, dm::box3(dm::float3(float(i)), dm::float3(float(i) + 3.f)));
        shared->SetLitOutOfBounds(true);
        scene.sun->shadowMap = shared;
        for (uint32_t i = 0; i < 17; ++i)
        {
            const dm::double3 position(double(i), 3.0, -double(i));
            const dm::double3 direction = dm::normalize(dm::double3(0.1 * double(i), -1.0, 0.2));
            nvrhi::AutoPtr<Light> light;
            switch (i % 4)
            {
            case 0: { auto d = MAKE_RC_OBJ_PTR(DirectionalLight); d->irradiance = float(i); light = d; break; }
            case 1: { auto s = MAKE_RC_OBJ_PTR(SpotLight); s->intensity = float(i); s->radius = 0.5f; s->range = 9.f;
                      s->innerAngle = 10.f; s->outerAngle = 30.f; light = s; break; }
            case 2: { auto p = MAKE_RC_OBJ_PTR(PointLight); p->intensity = float(i); p->range = 4.f; light = p; break; }
            default: { auto d = MAKE_RC_OBJ_PTR(DirectionalLight); d->angularSize = 1.f; light = d; break; }
            }
            if (i % 4 == 0)
                light->shadowMap = shared;
            if (i == 5)
                light->shadowMap = MAKE_RC_OBJ_PTR(OtherShadowMap, 3u, 1u, 7.f, shared->GetTexture());
            add(light, position, direction);
        }

        auto run = [&](bool serviceShadows, const char* what)
        {
            nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, serviceShadows);
            scene.graph->Refresh(++frameIndex);
            const RenderSnapshot& snapshot = *Extract(service, frame, scene.Desc(frameIndex))->snapshot;
            const uint32_t warnings = g_Warnings.load();
            const bool refused = CompareConstants(scene.graph->GetLights(), snapshot, what);
            g_Warnings = warnings;   // the max-lights warnings of the three deferred forms
            service->EndFrame(true);
            return refused;
        };

        // 18 lights, more than 16: the light and the shadow limits; one texture
        CHECK(!run(false, "service shadows off, one texture"));

        // a second texture: refused by the deferred pass in every form; forward takes the first
        extra[6]->shadowMap = MakeHint(256, 1, 0);
        CHECK(run(false, "service shadows off, two textures"));
        extra[6]->shadowMap = nullptr;

        // few lights, under the limits, in another order of attachment
        for (size_t i = 8; i < lightNodes.size(); ++i)
            scene.graph->Detach(lightNodes[i]);
        CHECK(!run(false, "service shadows off, 9 lights"));

        // service shadows: the sun alone with a cascaded hint (no per-object shadows: the service makes none), the
        // others without hints; the leaf vector's hint set up as the application would: equal constants
        for (size_t i = 0; i < 8; ++i)
            scene.graph->Detach(lightNodes[i]);
        nvrhi::AutoPtr<CascadedShadowMap> hint = MakeHint(2048, 4, 0);
        hint->SetLitOutOfBounds(true);
        hint->SetFalloffDistance(3.5f);
        scene.sun->shadowMap = hint;
        auto point = MAKE_RC_OBJ_PTR(PointLight);
        auto pointNode = MakeNode("point");
        pointNode->SetLeaf(point);
        scene.graph->Attach(root, pointNode);
        {
            nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
            for (uint32_t f = 0; f < 3; ++f)
            {
                scene.Look(dm::float3(-8.f + 2.f * float(f), 1.8f, float(f)), dm::float3(1.f, 2.5f - float(f), 4.f));
                scene.sun->SetDirection(dm::normalize(dm::double3(0.1 + 0.2 * f, -1.0, 0.15)));
                scene.graph->Refresh(++frameIndex);
                const RenderSnapshot& snapshot = *Extract(service, frame, scene.Desc(frameIndex))->snapshot;
                const ExtractedLight* sun = FindLight(snapshot, scene.sun);
                CHECK(sun->serviceShadow && sun->shadowMap != hint.Get());
                // the application's set-up of its own map (async_compute_rdg: SetupForPlanarView(sun, frustum, 40...))
                hint->SetupForPlanarView(*scene.sun, scene.view->GetViewFrustum(), 40.f, 40.f, 40.f);
                const auto& lights = scene.graph->GetLights();
                static DeferredLightingConstants deferred[2];
                std::memset(deferred, 0, sizeof(deferred));
                nvrhi::ITexture* textures[2] = {};
                CHECK(lighting::FillDeferredLights(lighting::LeafLights{ &lights }, deferred[0], DEFERRED_MAX_LIGHTS,
                                                   DEFERRED_MAX_SHADOWS, textures[0]));
                CHECK(lighting::FillDeferredLights(lighting::SnapshotLights{ snapshot.lights, snapshot.lightCount },
                                                   deferred[1], DEFERRED_MAX_LIGHTS, DEFERRED_MAX_SHADOWS, textures[1]));
                CHECK(BytesEqual(deferred[0], deferred[1]) && deferred[0].numLights == 2);
                CHECK(textures[0] == hint->GetTexture() && textures[1] == sun->shadowMap->GetTexture());
                static ForwardShadingLightConstants forward[2];
                std::memset(forward, 0, sizeof(forward));
                lighting::FillForwardLights(lighting::LeafLights{ &lights }, forward[0], FORWARD_MAX_LIGHTS,
                                            FORWARD_MAX_SHADOWS, textures[0]);
                lighting::FillForwardLights(lighting::SnapshotLights{ snapshot.lights, snapshot.lightCount }, forward[1],
                                            FORWARD_MAX_LIGHTS, FORWARD_MAX_SHADOWS, textures[1]);
                CHECK(BytesEqual(forward[0], forward[1]));
                // the copied values: lit out of bounds (outOfBoundsShadow 1) and the falloff of every cascade
                CHECK(sun->constants->outOfBoundsShadow == 1.f && deferred[1].numLights == 2);
                for (uint32_t c = 0; c < 4; ++c)
                    CHECK(deferred[1].shadows[c].shadowFalloffDistance == 3.5f);
                service->EndFrame(true);
            }
            std::printf("  service shadows: deferred and forward constants equal the application-set-up hint's, 3 "
                        "frames\n");
        }

        // Two suns with cascaded hints of different textures: the leaf vector is refused by the deferred pass (two
        // textures). With service shadows the first takes the one service shadow and the second is withheld (no
        // shadow), so the deferred constants are produced, and they equal the leaf vector's where that is equal: the
        // first hint set up as the application would, the second light without a map.
        {
            auto sun2 = MAKE_RC_OBJ_PTR(DirectionalLight);
            auto sun2Node = MakeNode("sun2");
            sun2Node->SetLeaf(sun2);
            scene.graph->Attach(root, sun2Node);
            sun2->SetDirection(dm::normalize(dm::double3(-0.4, -1.0, 0.3)));
            sun2->irradiance = 3.f;
            sun2->shadowChannel = 1;
            nvrhi::AutoPtr<CascadedShadowMap> hint2 = MakeHint(1024, 2, 0);
            hint2->SetLitOutOfBounds(false);   // a withheld light still gets a map-less light's constants: lit
            sun2->shadowMap = hint2;
            nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
            scene.graph->Refresh(++frameIndex);
            const uint32_t warnings = g_Warnings.load();
            const RenderSnapshot& snapshot = *Extract(service, frame, scene.Desc(frameIndex))->snapshot;
            CHECK(g_Warnings.load() == warnings + 1);   // the withheld hint, once
            g_Warnings = warnings;
            CHECK(service->GetStats().unservedShadowHints == 1 && service->GetLightShadows()->GetCount() == 1);
            const ExtractedLight* first = FindLight(snapshot, scene.sun);
            const ExtractedLight* second = FindLight(snapshot, sun2);
            CHECK(first && first->serviceShadow && second && !second->serviceShadow && second->shadowMap == nullptr);
            CHECK(second->constants->outOfBoundsShadow == 1.f);
            const auto& lights = scene.graph->GetLights();
            CHECK(lights.size() == 3 && lights[0].Get() == scene.sun.Get() && lights[2].Get() == sun2.Get());

            static DeferredLightingConstants deferred[3];
            std::memset(deferred, 0, sizeof(deferred));
            nvrhi::ITexture* textures[3] = {};
            CHECK(!lighting::FillDeferredLights(lighting::LeafLights{ &lights }, deferred[0], DEFERRED_MAX_LIGHTS,
                                                DEFERRED_MAX_SHADOWS, textures[0]));   // two textures: refused
            CHECK(lighting::FillDeferredLights(lighting::SnapshotLights{ snapshot.lights, snapshot.lightCount },
                                               deferred[1], DEFERRED_MAX_LIGHTS, DEFERRED_MAX_SHADOWS, textures[1]));
            hint->SetupForPlanarView(*scene.sun, scene.view->GetViewFrustum(), 40.f, 40.f, 40.f);
            sun2->shadowMap = nullptr;
            CHECK(lighting::FillDeferredLights(lighting::LeafLights{ &lights }, deferred[2], DEFERRED_MAX_LIGHTS,
                                               DEFERRED_MAX_SHADOWS, textures[2]));
            CHECK(BytesEqual(deferred[1], deferred[2]) && deferred[1].numLights == 3);
            CHECK(textures[1] == first->shadowMap->GetTexture() && textures[2] == hint->GetTexture());

            static ForwardShadingLightConstants forward[2];
            std::memset(forward, 0, sizeof(forward));
            nvrhi::ITexture* forwardTextures[2] = {};
            lighting::FillForwardLights(lighting::SnapshotLights{ snapshot.lights, snapshot.lightCount }, forward[0],
                                        FORWARD_MAX_LIGHTS, FORWARD_MAX_SHADOWS, forwardTextures[0]);
            lighting::FillForwardLights(lighting::LeafLights{ &lights }, forward[1], FORWARD_MAX_LIGHTS,
                                        FORWARD_MAX_SHADOWS, forwardTextures[1]);
            CHECK(BytesEqual(forward[0], forward[1]) && forwardTextures[0] == first->shadowMap->GetTexture());
            sun2->shadowMap = hint2;
            service->EndFrame(true);
            scene.graph->Detach(sun2Node);
            std::printf("  two cascaded hints with service shadows: the first served, the second withheld; deferred "
                        "constants produced (the leaf vector's refused) and equal to the leaf vector without the "
                        "second map\n");
        }
        scene.sun->shadowMap = scene.sunHint;
        scene.graph->Detach(pointNode);
    }

    // ------------------------------------------------------------------------------------------ cascades

    void CompareCascades(const CascadedShadowMap& service, const CascadedShadowMap& legacy, const RenderSnapshot& snapshot,
                         uint32_t lightIndex)
    {
        CHECK(service.GetNumberOfCascades() == legacy.GetNumberOfCascades() && legacy.GetNumberOfCascades() == 4);
        const auto views = CascadeViews(snapshot, lightIndex);
        CHECK(views.size() == 4);
        for (uint32_t c = 0; c < 4; ++c)
        {
            ShadowConstants a{}, b{};
            std::memset(&a, 0, sizeof(a));
            std::memset(&b, 0, sizeof(b));
            service.GetCascade(c)->FillShadowConstants(a);
            legacy.GetCascade(c)->FillShadowConstants(b);
            CHECK(BytesEqual(a, b));
            const IView* serviceView = service.GetCascade(c)->GetView().GetChildView(ViewType::PLANAR, 0);
            const IView* legacyView = legacy.GetCascade(c)->GetView().GetChildView(ViewType::PLANAR, 0);
            CHECK(serviceView->GetViewMatrix() == legacyView->GetViewMatrix());
            CHECK(dm::all(serviceView->GetProjectionMatrix(false) == legacyView->GetProjectionMatrix(false)));
            const dm::frustum fa = views[c]->frustum, fb = legacyView->GetViewFrustum();
            CHECK(std::memcmp(&fa, &fb, sizeof(fa)) == 0 && views[c]->view == serviceView);
        }
    }

    using Item = std::tuple<const MeshInstance*, const MeshGeometry*, uint32_t>;

    std::vector<Item> Drain(IDrawStrategy& strategy, SceneGraphNode* root, const IView& view)
    {
        std::vector<Item> items;
        strategy.PrepareForView(root, view);
        while (const DrawItem* item = strategy.GetNextItem())
            items.emplace_back(item->instance, item->geometry, uint32_t(item->cullMode));
        std::sort(items.begin(), items.end());
        return items;
    }

    void test_cascades()
    {
        ShadowScene scene;
        RenderFrame frame;
        uint32_t frameIndex = 0;
        nvrhi::AutoPtr<CascadedShadowMap> legacy = MakeHint(1024, 4, 0);
        scene.sunHint->SetLitOutOfBounds(true);
        scene.sunHint->SetFalloffDistance(2.f);
        auto bound = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        auto walk = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        uint64_t packets = 0;
        uint32_t nonEmpty = 0;
        for (bool stable : { false, true })
        {
            nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
            const ShadowSettings settings{ stable ? 60.f : 40.f, 40.f, stable ? 30.f : 40.f, stable ? 3.f : 4.f, stable };
            for (uint32_t f = 0; f < 8; ++f)
            {
                scene.Look(dm::float3(-8.f + 3.f * float(f), 1.8f + 0.5f * float(f), -2.f + float(f)),
                           dm::float3(1.f + float(f), 2.5f - 0.4f * float(f), 6.f));
                scene.sun->SetDirection(dm::normalize(dm::double3(0.1 + 0.15 * f, -1.0, 0.15 - 0.05 * f)));
                for (size_t i = f; i < scene.nodes.size(); i += 7)
                    scene.nodes[i]->SetTranslation(scene.nodes[i]->GetTranslation() + dm::double3(0.3, 0.0, -0.2));
                scene.graph->Refresh(++frameIndex);
                SceneFrame* sceneFrame = Extract(service, frame, scene.Desc(frameIndex, settings));
                const RenderSnapshot& snapshot = *sceneFrame->snapshot;
                uint32_t sunIndex = 0;
                const ExtractedLight* sun = FindLight(snapshot, scene.sun, &sunIndex);
                CHECK(sun && sun->serviceShadow);
                const auto* map = static_cast<const CascadedShadowMap*>(sun->shadowMap);
                CHECK(map->IsLitOutOfBounds() && map->GetFalloffDistance() == 2.f);

                // the application's call for its own map
                if (stable)
                    legacy->SetupForPlanarViewStable(*scene.sun, scene.view->GetProjectionFrustum(),
                                                     scene.view->GetInverseViewMatrix(), settings.maxShadowDistance,
                                                     settings.lightSpaceZUp, settings.lightSpaceZDown, settings.exponent);
                else
                    legacy->SetupForPlanarView(*scene.sun, scene.view->GetViewFrustum(), settings.maxShadowDistance,
                                               settings.lightSpaceZUp, settings.lightSpaceZDown, settings.exponent);
                legacy->SetLitOutOfBounds(true);
                legacy->SetFalloffDistance(2.f);
                CompareCascades(*map, *legacy, snapshot, sunIndex);

                // each cascade view's opaque packets: the legacy strategy's walk of the legacy cascade view
                bound->SetPackets(&snapshot);
                for (uint32_t c = 0; c < 4; ++c)
                {
                    const IView& serviceView = *map->GetCascade(c)->GetView().GetChildView(ViewType::PLANAR, 0);
                    const IView& legacyView = *legacy->GetCascade(c)->GetView().GetChildView(ViewType::PLANAR, 0);
                    const auto fromPackets = Drain(*bound, nullptr, serviceView);
                    const auto walked = Drain(*walk, scene.graph->GetRootNode(), legacyView);
                    if (fromPackets != walked)
                        std::fprintf(stderr, "  frame %u cascade %u: %zu packets, %zu walked\n", f, c,
                                     fromPackets.size(), walked.size());
                    CHECK(fromPackets == walked);   // a near cascade may hold nothing
                    packets += fromPackets.size();
                    nonEmpty += walked.empty() ? 0u : 1u;
                }
                bound->SetPackets(nullptr);
                service->EndFrame(true);
            }
        }
        std::printf("  16 frames: cascades equal the application's set-up (plain and stable); %llu opaque packets in "
                    "the cascade views (%u of 64 not empty), equal to the legacy walk\n", (unsigned long long)packets,
                    nonEmpty);
        CHECK(packets != 0 && nonEmpty >= 32);
    }

    // ------------------------------------------------------------------------------------------ shadow_passes

    void test_shadow_passes()
    {
        ShadowScene scene;
        RenderFrame frame;
        uint32_t frameIndex = 0;
        nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
        auto lighting = MAKE_RC_OBJ_PTR(DeferredLightingPass, nullptr, nullptr);

        // no open frame
        SceneFrame none{};
        CHECK(service->AddShadowPasses(frame.graph, &none) == nvrhi::FE_INVALID_ARGS);

        auto standIn = [&](nvrhi::Format format, bool uav)
        {
            nvrhi::TextureDesc desc;
            desc.width = 1280;
            desc.height = 720;
            desc.format = format;
            desc.isRenderTarget = !uav;
            desc.isUAV = uav;
            desc.debugName = "StandIn";
            return frame.graph->ImportTextureDesc(desc, nvrhi::ResourceStates::ShaderResource,
                                                  nvrhi::ResourceStates::ShaderResource);
        };

        // Frames 0-2 light from the extracted lights; frame 3 is the negative control: the lighting pass without them
        // declares no read of the shadow texture.
        for (uint32_t f = 0; f < 4; ++f)
        {
            const bool withLights = f < 3;
            scene.graph->Refresh(++frameIndex);
            SceneFrame* sceneFrame = Extract(service, frame, scene.Desc(frameIndex));
            CHECK(service->AddUploadPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            CHECK(service->AddSkinningPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            const uint32_t passesBefore = frame.graph->GetPassCount();
            CHECK(service->AddShadowPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            CHECK(frame.graph->GetPassCount() == passesBefore + 4);
            CHECK(service->AddShadowPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);   // a second call: nothing
            CHECK(frame.graph->GetPassCount() == passesBefore + 4);
            const RenderSnapshot& snapshot = *sceneFrame->snapshot;
            const ExtractedLight* sun = FindLight(snapshot, scene.sun);
            CHECK(sun->shadowTexture != InvalidResourceId);

            DeferredLightingGraphInputs inputs;
            inputs.view = scene.view;
            inputs.depth = standIn(nvrhi::Format::D32, false);
            inputs.gbufferNormals = standIn(nvrhi::Format::RGBA16_SNORM, false);
            inputs.gbufferDiffuse = standIn(nvrhi::Format::SRGBA8_UNORM, false);
            inputs.gbufferSpecular = standIn(nvrhi::Format::SRGBA8_UNORM, false);
            inputs.gbufferEmissive = standIn(nvrhi::Format::RGBA16_FLOAT, false);
            inputs.output = standIn(nvrhi::Format::RGBA16_FLOAT, true);
            inputs.extractedLights = withLights ? snapshot.lights : nullptr;
            inputs.extractedLightCount = withLights ? snapshot.lightCount : 0;
            CHECK(lighting->AddToGraph(frame.graph, inputs).output != InvalidResourceId);

            for (ReferenceProfile reference : { ReferenceProfile::D3D12Legacy, ReferenceProfile::D3D12Enhanced,
                                                ReferenceProfile::Vulkan, ReferenceProfile::D3D11 })
            {
                BackendProfile profile;
                CHECK(GetReferenceProfile(reference, &profile) == nvrhi::FS_OK);
                ICompiledRenderGraph* compiled = nullptr;
                const nvrhi::FRESULT result = frame.graph->Compile(profile, CompileOptions(), &compiled);
                nvrhi::string diagnostics;
                frame.graph->GetDiagnostics(diagnostics);
                if (result != nvrhi::FS_OK || !diagnostics.empty())
                    std::fprintf(stderr, "%s\n", diagnostics.c_str());
                CHECK(result == nvrhi::FS_OK && diagnostics.empty());
                nvrhi::AutoPtr<ICompiledRenderGraph> plan = nvrhi::TakeOver(compiled);
                nvrhi::string report;
                if (VerifyPlan(plan, frame.graph, nullptr, &report) != nvrhi::FS_OK)
                {
                    std::fprintf(stderr, "%s\n", report.c_str());
                    CHECK(false);
                }
                // every cascade before the lighting, all enabled
                uint32_t cascades = 0, lightingAt = ~0u, lastCascade = 0;
                for (uint32_t i = 0; i < plan->GetScheduledPassCount(); ++i)
                {
                    ScheduledPass pass;
                    CHECK(plan->GetScheduledPass(i, pass) == nvrhi::FS_OK && (pass.enabled || !withLights));
                    if (pass.name && std::strcmp(pass.name, "ShadowDepth") == 0)
                    {
                        ++cascades;
                        lastCascade = i;
                    }
                    else if (pass.name && std::strcmp(pass.name, "DeferredLighting") == 0)
                        lightingAt = i;
                }
                CHECK(cascades == 4 && lightingAt != ~0u && (lastCascade < lightingAt || !withLights));

                // The ordering is declared, not incidental: each cascade use writes its slice of the service map's
                // texture (Tracked: no buffer reads without a scene), and the lighting use reads that texture, with
                // the transition into the read before it on the profiles that record barriers; the negative control
                // has neither.
                const std::vector<DumpedPass> passes = DumpPasses(plan);
                uint32_t writers = 0, readers = 0, transitions = 0;
                for (const DumpedPass& pass : passes)
                {
                    if (pass.name == "ShadowDepth")
                    {
                        CHECK(pass.mode == "tracked");
                        writers += pass.uses.find("ServiceShadowMap DepthWrite") != std::string::npos ? 1u : 0u;
                    }
                    else if (pass.name == "DeferredLighting")
                    {
                        readers += pass.uses.find("ServiceShadowMap ShaderRead") != std::string::npos ? 1u : 0u;
                        transitions += pass.prologue.find("ServiceShadowMap") != std::string::npos ? 1u : 0u;
                    }
                }
                const bool barriers = reference != ReferenceProfile::D3D11;
                if (writers != 4 || readers != (withLights ? 1u : 0u) ||
                    transitions != (withLights && barriers ? 1u : 0u))
                {
                    nvrhi::string dump;
                    plan->WriteDump(DumpFormat::Json, dump);
                    std::fprintf(stderr, "writers %u readers %u transitions %u\n%s\n", writers, readers, transitions,
                                 dump.c_str());
                }
                CHECK(writers == 4 && readers == (withLights ? 1u : 0u));
                CHECK(transitions == (withLights && barriers ? 1u : 0u));
            }
            service->EndFrame(false);
        }
        // after EndFrame: no open frame
        SceneFrame closed{};
        CHECK(service->AddShadowPasses(frame.graph, &closed) == nvrhi::FE_INVALID_ARGS);

        // service shadows off: nothing
        service = nullptr;
        nvrhi::AutoPtr<RenderService> off = MakeService(scene.graph, false);
        scene.graph->Refresh(++frameIndex);
        SceneFrame* sceneFrame = Extract(off, frame, scene.Desc(frameIndex));
        CHECK(off->AddShadowPasses(frame.graph, sceneFrame) == nvrhi::FS_OK && frame.graph->GetPassCount() == 0);
        off->EndFrame(true);
        std::printf("  3 frames: 4 cascade uses before the lighting use, which reads their texture (and transitions "
                    "it where barriers are recorded); without extracted lights no read; compiled and verified on 4 "
                    "reference profiles\n");
    }

    // ------------------------------------------------------------------------------------------ binding_resets

    // The pooled depth passes' binding caches after a buffer replacement (SceneFrame::sceneBuffersReplaced): a frame
    // that replaced the buffers but whose AddShadowPasses was refused leaves the reset pending, and the next call that
    // runs resets every pass (device-free, the passes are not initialised: the reset must not touch what Init makes).
    void test_binding_resets()
    {
        ShadowScene scene;
        RenderFrame frame;
        uint32_t frameIndex = 0;
        nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
        auto next = [&]()
        {
            scene.graph->Refresh(++frameIndex);
            return Extract(service, frame, scene.Desc(frameIndex));
        };
        auto addAll = [&](SceneFrame* sceneFrame)
        {
            CHECK(service->AddUploadPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            CHECK(service->AddSkinningPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            return service->AddShadowPasses(frame.graph, sceneFrame);
        };
        auto resets = [&]() { return service->GetStats().shadowBindingCacheResets; };

        // the four cascade passes are made; nothing to reset
        CHECK(addAll(next()) == nvrhi::FS_OK && resets() == 0 && frame.graph->GetPassCount() == 4);
        service->EndFrame(true);

        // replaced, and AddShadowPasses refused (called before AddUploadPasses): nothing added, the reset waits
        SceneFrame* sceneFrame = next();
        sceneFrame->sceneBuffersReplaced = true;
        const uint32_t warnings = g_Warnings.load();
        CHECK(service->AddShadowPasses(frame.graph, sceneFrame) == nvrhi::FE_INVALID_ARGS);
        CHECK(g_Warnings.load() == warnings + 1 && resets() == 0 && frame.graph->GetPassCount() == 0);
        g_Warnings -= 1;
        service->EndFrame(false);

        // the next frame replaced nothing: its call runs the pending reset over every pooled pass
        CHECK(addAll(next()) == nvrhi::FS_OK && resets() == 4);
        service->EndFrame(true);
        CHECK(addAll(next()) == nvrhi::FS_OK && resets() == 0);   // done
        service->EndFrame(true);

        // replaced in a frame whose call runs: at once
        sceneFrame = next();
        sceneFrame->sceneBuffersReplaced = true;
        CHECK(addAll(sceneFrame) == nvrhi::FS_OK && resets() == 4);
        service->EndFrame(true);

        // a DepthPass never initialised (no device): ResetBindingCache has nothing to clear
        auto uninitialised = MAKE_RC_OBJ_PTR(DepthPass, nullptr, nullptr);
        uninitialised->ResetBindingCache();
        std::printf("  a reset pending across a refused AddShadowPasses runs at the next call (4 passes)\n");
    }

    // ------------------------------------------------------------------------------------------ gpu_pass_shape

    // The cascade uses as on the GPU path, device-free. A Scene needs a device, so the frame's scene buffers are
    // stand-ins written by two lambda passes in the places of the upload and skinning passes, and the SceneFrame names
    // their versions in bufferReads as AddUploadPasses and AddSkinningPasses would: each cascade use then declares the
    // reads (VertexBuffer and ShaderRead where the buffer is a vertex buffer) and is Restore, scheduled after both
    // writers; the plan verifies. AddShadowPasses before AddUploadPasses, or before AddSkinningPasses, is refused
    // (FE_INVALID_ARGS, a warning each) and adds nothing.
    void test_gpu_pass_shape()
    {
        struct NoData
        {
            uint32_t unused;
        };
        ShadowScene scene;
        RenderFrame frame;
        nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
        scene.graph->Refresh(1);
        SceneFrame* sceneFrame = Extract(service, frame, scene.Desc(1));
        IRenderGraph* g = frame.graph;

        const uint32_t warnings = g_Warnings.load();
        CHECK(service->AddShadowPasses(g, sceneFrame) == nvrhi::FE_INVALID_ARGS && g->GetPassCount() == 0);
        CHECK(service->AddUploadPasses(g, sceneFrame) == nvrhi::FS_OK);
        CHECK(service->AddShadowPasses(g, sceneFrame) == nvrhi::FE_INVALID_ARGS && g->GetPassCount() == 0);
        CHECK(service->AddSkinningPasses(g, sceneFrame) == nvrhi::FS_OK);
        CHECK(g_Warnings.load() == warnings + 2);
        g_Warnings -= 2;

        auto buffer = [](const char* name, bool vertexBuffer)
        {
            nvrhi::BufferDesc desc;
            desc.byteSize = 4096;
            desc.isVertexBuffer = vertexBuffer;
            desc.canHaveRawViews = true;
            desc.canHaveUAVs = true;
            desc.debugName = name;
            return desc;
        };
        // the graph path's resting states (engine::c_GpuSceneRestingState, c_SkinnedVertexBufferRestingState)
        const nvrhi::ResourceStates sceneResting = nvrhi::ResourceStates::ShaderResource;
        const nvrhi::ResourceStates skinnedResting =
            nvrhi::ResourceStates::VertexBuffer | nvrhi::ResourceStates::ShaderResource;
        ResourceId instances = g->ImportBufferDesc(buffer("InstanceBuffer", true), sceneResting, sceneResting);
        const ResourceId geometries = g->ImportBufferDesc(buffer("GeometryBuffer", false), sceneResting, sceneResting);
        ResourceId skinned = g->ImportBufferDesc(buffer("SkinnedVertices", true), skinnedResting, skinnedResting);
        AddLambdaPass<NoData>(g, PassDesc{ "Upload", PassType::Compute, PassFlags::None, 0 },
            [&](ISetupContext* setup, NoData&)
            {
                instances = setup->Write(instances, render::Access::UAVWrite, nvrhi::ShaderType::Compute, {}, AccessFlags::None);
            },
            [](IPassContext*, const NoData&) { });
        AddLambdaPass<NoData>(g, PassDesc{ "Skinning", PassType::Compute, PassFlags::None, 0 },
            [&](ISetupContext* setup, NoData&)
            {
                skinned = setup->Write(skinned, render::Access::UAVWrite, nvrhi::ShaderType::Compute, {}, AccessFlags::None);
            },
            [](IPassContext*, const NoData&) { });
        auto* reads = static_cast<ResourceId*>(g->AllocateFrameData(sizeof(ResourceId) * 3, alignof(ResourceId)));
        reads[0] = instances;
        reads[1] = geometries;
        reads[2] = skinned;
        sceneFrame->bufferReads = reads;
        sceneFrame->bufferReadCount = 3;

        CHECK(service->AddShadowPasses(g, sceneFrame) == nvrhi::FS_OK && g->GetPassCount() == 2 + 4);

        BackendProfile profile;
        CHECK(GetReferenceProfile(ReferenceProfile::D3D12Legacy, &profile) == nvrhi::FS_OK);
        ICompiledRenderGraph* compiled = nullptr;
        const nvrhi::FRESULT result = g->Compile(profile, CompileOptions(), &compiled);
        nvrhi::string diagnostics;
        g->GetDiagnostics(diagnostics);
        if (result != nvrhi::FS_OK || !diagnostics.empty())
            std::fprintf(stderr, "%s\n", diagnostics.c_str());
        CHECK(result == nvrhi::FS_OK && diagnostics.empty());
        nvrhi::AutoPtr<ICompiledRenderGraph> plan = nvrhi::TakeOver(compiled);
        nvrhi::string report;
        if (VerifyPlan(plan, g, nullptr, &report) != nvrhi::FS_OK)
        {
            std::fprintf(stderr, "%s\n", report.c_str());
            CHECK(false);
        }

        const std::vector<DumpedPass> passes = DumpPasses(plan);
        uint32_t uploadAt = ~0u, skinningAt = ~0u, cascades = 0;
        bool ordered = true;
        for (uint32_t i = 0; i < passes.size(); ++i)
        {
            const DumpedPass& pass = passes[i];
            if (pass.name == "Upload")
                uploadAt = i;
            else if (pass.name == "Skinning")
                skinningAt = i;
            else if (pass.name == "ShadowDepth")
            {
                ++cascades;
                ordered = ordered && uploadAt < i && skinningAt < i;
                const bool shape = pass.mode == "restore" &&
                    pass.uses.find("InstanceBuffer VertexBuffer|ShaderRead") != std::string::npos &&
                    pass.uses.find("GeometryBuffer ShaderRead") != std::string::npos &&
                    pass.uses.find("SkinnedVertices VertexBuffer|ShaderRead") != std::string::npos &&
                    pass.uses.find("ServiceShadowMap DepthWrite") != std::string::npos;
                if (!shape)
                    std::fprintf(stderr, "ShadowDepth [%s]: %s\n", pass.mode.c_str(), pass.uses.c_str());
                CHECK(shape);
            }
        }
        CHECK(cascades == 4 && ordered && uploadAt != ~0u && skinningAt != ~0u);
        service->EndFrame(false);
        std::printf("  4 Restore cascade uses reading the scene's buffers after the upload and skinning writers; "
                    "AddShadowPasses before either refused\n");
    }

    // ------------------------------------------------------------------------------------------ steady_state

    void test_steady_state()
    {
        ShadowScene scene;
        RenderFrame frame;
        nvrhi::AutoPtr<RenderService> service = MakeService(scene.graph, true);
        uint32_t frameIndex = 0;
        uint64_t allocations = 0;
        // The moves repeat every 10 frames (each fifth of the nodes moves one way, then back five frames later), so
        // the warm-up is one period: every visible count of the measured frames has been seen.
        for (uint32_t f = 0; f < 20; ++f)
        {
            for (size_t i = f % 5; i < scene.nodes.size(); i += 5)
                scene.nodes[i]->SetTranslation(scene.nodes[i]->GetTranslation() + dm::double3(f & 1 ? 0.4 : -0.4, 0, 0));
            scene.Look(dm::float3(-8.f + float(f & 1), 1.8f, 0.f), dm::float3(1.f, 2.5f, 0.f));
            scene.graph->Refresh(++frameIndex);
            const uint64_t before = g_NewCount.load();
            frame.Begin();
            SceneFrame* sceneFrame = nullptr;
            CHECK(service->Extract(frame.graph, &frame.data, scene.Desc(frameIndex), &sceneFrame) == nvrhi::FS_OK);
            CHECK(service->AddUploadPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            CHECK(service->AddSkinningPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            CHECK(service->AddShadowPasses(frame.graph, sceneFrame) == nvrhi::FS_OK);
            service->EndFrame(true);
            if (f >= 10)
                allocations += g_NewCount.load() - before;
        }
        std::printf("  10 frames after warm-up with moves, Extract and the add-pass calls: %llu operator new\n",
                    (unsigned long long)allocations);
        CHECK(allocations == 0);
    }

    struct Case
    {
        const char* name;
        void (*run)();
    };
}

int main(int argc, char** argv)
{
#if defined(_MSC_VER)
    _set_error_mode(_OUT_TO_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });

    const Case cases[] = {
        { "table_requests", test_table_requests },
        { "table_lifecycle", test_table_lifecycle },
        { "constants", test_constants },
        { "cascades", test_cascades },
        { "shadow_passes", test_shadow_passes },
        { "binding_resets", test_binding_resets },
        { "gpu_pass_shape", test_gpu_pass_shape },
        { "steady_state", test_steady_state },
    };
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
                continue;
            std::printf("%s\n", c.name);
            const uint32_t warnings = g_Warnings.load();
            c.run();
            CHECK(g_Warnings.load() == warnings);
            std::printf("%s: passed\n", c.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    std::printf("light shadows: ok\n");
    return 0;
}
