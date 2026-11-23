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

// Shadow maps owned by the render service, keyed by light entity (aggregate docs/ecs-design.md 7.6 and D13; plan step
// E5, sub-plan docs/plans/2026-11-21-donut-ecs-e5.md, items E5.0 and E5.6, open point 8).
//
// The leaf fields Light::shadowMap and Light::shadowChannel are authoring hints: at each extraction the service reads
// them (lights are few, D16) and turns a light whose shadowMap is set into a request for a LightShadow of that map's
// cascade count, resolution and texture format and the light's channel. The service makes and keeps its own map per
// light entity in a LightShadowTable; it never writes a leaf. A change of the cascade count, resolution or format makes
// a new map; a change of the channel is taken in place. The values a
// map can take without being made again are copied from the hint onto the service's map at every extraction:
// lit-out-of-bounds (IShadowMap::IsLitOutOfBounds) and the falloff distance (CascadedShadowMap::GetFalloffDistance, the
// value FillShadowConstants gives each cascade). Service shadows are opt-in per service
// (RenderServiceDesc::serviceShadows, which also holds the depth passes' bias): without them the extracted lights carry
// the leaf's own map, which the application renders as before.
//
// What the service serves (sub-plan item E5.6): the first light in SceneGraph::GetLights() order that is a
// DirectionalLight whose hint is a CascadedShadowMap, with the hint's cascades (its texture's slices less its
// per-object shadows), resolution and format, set up for the shadow view of ExtractDesc with ShadowSettings
// (SetupForPlanarView, or SetupForPlanarViewStable); no per-object shadows (DirectionalLight::perObjectShadows has no
// user in the tree). One light only: the lighting passes bind one shadow texture (DeferredLightingPass refuses a
// second, ForwardShadingPass samples the first). A later light with such a hint is not served and carries no shadow in
// the snapshot (its hint is the service's to render, and a second texture would stop the deferred lighting); any other
// hint (a spot or point light, a PlanarShadowMap hint, a map without a texture) is not served either, and the light
// keeps its leaf's map, which the application renders as before. RenderServiceStats::unservedShadowHints counts both.
// donut sets a PlanarShadowMap up only for directional lights, so LightShadowKind::Planar is reserved for a spot-light
// shadow that sets up its own perspective view.
//
// Module-private C++, like GpuScene.h: the table holds shadow-map objects, which are implementation classes.

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/EcsTypes.h>
#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <vector>

namespace donut::engine
{
    class IShadowMap;
    class IView;
    class Light;
    class DirectionalLight;
}

namespace donut::render
{
    // What a light's shadow is (D13). Derived from the leaf's hints each extraction.
    struct LightShadow
    {
        int channel = -1;               // Light::shadowChannel
        uint32_t cascadeCount = 0;      // the hint's cascades (CascadedShadowMap::GetCascadeCapacity), 1 to 4
        uint32_t resolution = 0;        // texels per side of one cascade (the hint map's GetTextureSize().x)
        nvrhi::Format format = nvrhi::Format::UNKNOWN;   // the hint map's texture format (GetTexture()->getDesc())

        [[nodiscard]] bool operator==(const LightShadow& other) const noexcept
        {
            return channel == other.channel && cascadeCount == other.cascadeCount && resolution == other.resolution
                && format == other.format;
        }
        [[nodiscard]] bool operator!=(const LightShadow& other) const noexcept { return !(*this == other); }
        // The fields that shape the map's texture: a change of these makes a new map; the channel is updated in place.
        [[nodiscard]] bool HasSameTexture(const LightShadow& other) const noexcept
        {
            return cascadeCount == other.cascadeCount && resolution == other.resolution && format == other.format;
        }
    };

    // The cascade distribution of the service's directional shadows: per service and view, not per light (open
    // point 8; D13's LightShadow has no place for it). The values of CascadedShadowMap::SetupForPlanarView(Stable).
    struct ShadowSettings
    {
        float maxShadowDistance = 100.f;
        float lightSpaceZUp = 100.f;
        float lightSpaceZDown = 100.f;
        float exponent = 4.f;
        // SetupForPlanarViewStable instead of SetupForPlanarView.
        bool stable = false;
    };

    enum class LightShadowKind : uint8_t
    {
        Cascaded,   // a DirectionalLight: CascadedShadowMap
        Planar      // reserved: a SpotLight's PlanarShadowMap (not served in E5; see above)
    };

    // One service shadow. `map` is the table's (one reference, released when the entry leaves); its texture is
    // imported into each frame's graph by RenderService::AddShadowPasses.
    struct LightShadowEntry
    {
        engine::Entity light = engine::Entity::Null;   // index and generation: a recycled index is another light
        LightShadow shadow;
        LightShadowKind kind = LightShadowKind::Cascaded;
        engine::IShadowMap* map = nullptr;             // null until the service made it
    };

    // The service's shadows, one entry per light entity, ordered by entity (index, then generation), so that every
    // walk over it is deterministic. Data with primitive operations; RenderService decides what to request.
    class LightShadowTable
    {
    public:
        LightShadowTable() noexcept = default;
        ~LightShadowTable();   // releases every map, the retired ones included

        LightShadowTable(const LightShadowTable&) = delete;
        LightShadowTable& operator=(const LightShadowTable&) = delete;

        [[nodiscard]] uint32_t GetCount() const noexcept { return uint32_t(m_Entries.size()); }
        [[nodiscard]] const LightShadowEntry* GetEntries() const noexcept { return m_Entries.data(); }
        // The entry of exactly this entity; null when it has none (also for the same index with another generation).
        // GetEntries and Find point into the table: valid until the next Request or Sweep.
        [[nodiscard]] const LightShadowEntry* Find(engine::Entity light) const noexcept;

        // Marks `light` as wanted by this extraction and returns its entry: a new one (map null), or the existing one;
        // when the fields that shape its texture (LightShadow::HasSameTexture) or its kind changed, the old map is
        // retired and `map` is null again; a changed channel alone is updated in place (the map is kept). Null when out
        // of memory. The pointer is valid until the next Request or Sweep (a later Request may insert and move the
        // entries).
        LightShadowEntry* Request(engine::Entity light, const LightShadow& shadow, LightShadowKind kind) noexcept;
        // Ends an extraction: the entries not requested since the last Sweep leave the table in entity order (a light
        // that is gone, detached, replaced by SetLeaf, or whose hint was cleared), and their maps are retired. Returns
        // how many left.
        uint32_t Sweep() noexcept;
        // Abandons an extraction that failed before its Sweep: the requests since the last Sweep are forgotten (nothing
        // leaves the table), so that the next extraction's Sweep sees only its own requests.
        void CancelRequests() noexcept;

        // Retired maps are kept until ReleaseRetired. The service calls it at the EndFrame of the extraction that
        // retired them (RenderService::EndFrame): no pass of that frame uses a retired map, the frames that drew into it
        // or read it were submitted before, and nvrhi's command lists hold their textures until the GPU has completed
        // them. Not a registry retirement (IResourceRegistry has none for application objects); E7's pipelining must
        // keep retired maps until the frames in flight that name them have completed.
        [[nodiscard]] uint32_t GetRetiredCount() const noexcept { return uint32_t(m_Retired.size()); }
        void ReleaseRetired() noexcept;

    private:
        std::vector<LightShadowEntry> m_Entries;
        std::vector<uint8_t> m_Requested;            // per entry, since the last Sweep
        std::vector<engine::IShadowMap*> m_Retired;  // one reference each
    };

    class CascadedShadowMap;

    // Makes a service shadow's texture instead of the device (RenderServiceDesc::createShadowTexture, a seam for the
    // device-free tests): a new reference to a texture of `desc` (CascadedShadowMap::GetTextureDesc), or null.
    using CreateShadowTextureFunction = nvrhi::ITexture* (*)(void* context, const nvrhi::TextureDesc& desc) noexcept;

    // The service's reading of a light's hint: true, with the LightShadow and kind, when the service serves it (a
    // DirectionalLight whose shadowMap is a CascadedShadowMap with a texture of 1 to 4 cascades); false otherwise,
    // also for a light without a hint.
    [[nodiscard]] bool GetLightShadowRequest(const engine::Light* light, LightShadow& outShadow,
                                             LightShadowKind& outKind) noexcept;

    // Sets a service map's cascades up for `view` as an application sets its own map up: SetupForPlanarView
    // (view->GetViewFrustum()) or, with settings.stable, SetupForPlanarViewStable (view->GetProjectionFrustum(),
    // view->GetInverseViewMatrix()), every cascade, with `preViewTranslation` (the render origin, zero in E5). False
    // when an argument is null, the light is not attached, or the settings are not valid for the view
    // (AreShadowSettingsValid); nothing is set up then.
    bool SetupLightShadowCascades(CascadedShadowMap* map, const engine::DirectionalLight* light,
                                  const engine::IView* view, const ShadowSettings& settings,
                                  dm::float3 preViewTranslation) noexcept;

    // What CascadedShadowMap::SetupForPlanarView(Stable) asserts, checked: a finite exponent above 1; the view's
    // frustum (its projection frustum with settings.stable) with a near plane; and a finite maxShadowDistance above 0,
    // or else a far plane (an infinite projection has none). False for a null view. The render service checks it at
    // each extraction and serves no light in a frame whose settings fail (one warning per service).
    [[nodiscard]] bool AreShadowSettingsValid(const engine::IView* view, const ShadowSettings& settings) noexcept;
}
