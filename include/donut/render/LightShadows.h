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
// light entity in a LightShadowTable; it never writes a leaf. A change of those values makes a new map. The values a
// map can take without being made again are copied from the hint onto the service's map at every extraction:
// lit-out-of-bounds (IShadowMap::IsLitOutOfBounds) and the falloff distance (ShadowConstants::shadowFalloffDistance of
// the hint's first cascade, as FillShadowConstants gives it). Service shadows are opt-in per service
// (RenderServiceDesc::serviceShadows, which also holds the depth passes' bias): without them the extracted lights carry
// the leaf's own map, which the application renders as before.
//
// Module-private C++, like GpuScene.h: the table holds shadow-map objects, which are implementation classes.

#pragma once

#include <donut/engine/EcsTypes.h>
#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <vector>

namespace donut::engine
{
    class IShadowMap;
}

namespace donut::render
{
    // What a light's shadow is (D13). Derived from the leaf's hints each extraction.
    struct LightShadow
    {
        int channel = -1;               // Light::shadowChannel
        uint32_t cascadeCount = 0;      // the hint map's GetNumberOfCascades(); 1 for a spot light's planar map
        uint32_t resolution = 0;        // texels per side of one cascade (the hint map's GetTextureSize().x)
        nvrhi::Format format = nvrhi::Format::UNKNOWN;   // the hint map's texture format (GetTexture()->getDesc())

        [[nodiscard]] bool operator==(const LightShadow& other) const noexcept
        {
            return channel == other.channel && cascadeCount == other.cascadeCount && resolution == other.resolution
                && format == other.format;
        }
        [[nodiscard]] bool operator!=(const LightShadow& other) const noexcept { return !(*this == other); }
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
        Planar      // a SpotLight: PlanarShadowMap
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
        // when its LightShadow or kind changed, the old map is retired and `map` is null again. Null when out of memory.
        // The pointer is valid until the next Request or Sweep (a later Request may insert and move the entries).
        LightShadowEntry* Request(engine::Entity light, const LightShadow& shadow, LightShadowKind kind) noexcept;
        // Ends an extraction: the entries not requested since the last Sweep leave the table in entity order (a light
        // that is gone, detached, replaced by SetLeaf, or whose hint was cleared), and their maps are retired. Returns
        // how many left.
        uint32_t Sweep() noexcept;

        // Retired maps are kept until no frame in flight can still use their textures; the service releases them
        // through the registry's retirement (ReleaseRetired once those frames have completed).
        [[nodiscard]] uint32_t GetRetiredCount() const noexcept { return uint32_t(m_Retired.size()); }
        void ReleaseRetired() noexcept;

    private:
        std::vector<LightShadowEntry> m_Entries;
        std::vector<uint8_t> m_Requested;            // per entry, since the last Sweep
        std::vector<engine::IShadowMap*> m_Retired;  // one reference each
    };
}
