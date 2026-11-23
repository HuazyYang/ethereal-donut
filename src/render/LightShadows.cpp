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

// The render service's shadow table (LightShadows.h; plan step E5, sub-plan items E5.1 and E5.6): primitive operations
// on entries ordered by light entity, and the service's reading of a light's hint and set-up of its cascades. The
// service decides what to request (RenderExtract.cpp).

#include <donut/render/LightShadows.h>
#include <donut/render/CascadedShadowMap.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include <donut/core/query_cast.h>

#include <algorithm>
#include <cmath>

namespace donut::render
{
    namespace
    {
        // Index, then generation: a recycled index sorts next to its earlier light, and every walk is deterministic.
        bool EntityLess(engine::Entity a, engine::Entity b) noexcept
        {
            const uint32_t indexA = engine::GetEntityIndex(a);
            const uint32_t indexB = engine::GetEntityIndex(b);
            if (indexA != indexB)
                return indexA < indexB;
            return engine::GetEntityGeneration(a) < engine::GetEntityGeneration(b);
        }

        struct EntryLess
        {
            bool operator()(const LightShadowEntry& entry, engine::Entity light) const noexcept
            {
                return EntityLess(entry.light, light);
            }
        };

        // Keeps one reference until ReleaseRetired. Out of memory: released at once (nvrhi keeps a texture alive for
        // the command lists that still reference it).
        void Retire(std::vector<engine::IShadowMap*>& retired, engine::IShadowMap* map) noexcept
        {
            if (!map)
                return;
            try
            {
                retired.push_back(map);
            }
            catch (...)
            {
                log::warning("LightShadowTable: out of memory; a retired shadow map is released at once");
                map->Release();
            }
        }
    }

    LightShadowTable::~LightShadowTable()
    {
        for (LightShadowEntry& entry : m_Entries)
        {
            if (entry.map)
                entry.map->Release();
            entry.map = nullptr;
        }
        ReleaseRetired();
    }

    const LightShadowEntry* LightShadowTable::Find(engine::Entity light) const noexcept
    {
        const auto it = std::lower_bound(m_Entries.begin(), m_Entries.end(), light, EntryLess());
        return it != m_Entries.end() && it->light == light ? &*it : nullptr;
    }

    LightShadowEntry* LightShadowTable::Request(engine::Entity light, const LightShadow& shadow,
                                                LightShadowKind kind) noexcept
    {
        if (light == engine::Entity::Null)
            return nullptr;

        const auto it = std::lower_bound(m_Entries.begin(), m_Entries.end(), light, EntryLess());
        const size_t index = size_t(it - m_Entries.begin());
        if (it != m_Entries.end() && it->light == light)
        {
            LightShadowEntry& entry = m_Entries[index];
            if (!entry.shadow.HasSameTexture(shadow) || entry.kind != kind)
            {
                // another texture: a new map, made by the service
                Retire(m_Retired, entry.map);
                entry.map = nullptr;
                entry.kind = kind;
            }
            entry.shadow = shadow;   // the channel, in place
            m_Requested[index] = 1;
            return &entry;
        }

        try
        {
            m_Entries.insert(m_Entries.begin() + ptrdiff_t(index), LightShadowEntry{ light, shadow, kind, nullptr });
        }
        catch (...)
        {
            return nullptr;
        }
        try
        {
            m_Requested.insert(m_Requested.begin() + ptrdiff_t(index), uint8_t(1));
        }
        catch (...)
        {
            m_Entries.erase(m_Entries.begin() + ptrdiff_t(index));
            return nullptr;
        }
        return &m_Entries[index];
    }

    uint32_t LightShadowTable::Sweep() noexcept
    {
        uint32_t removed = 0;
        size_t kept = 0;
        for (size_t i = 0; i < m_Entries.size(); ++i)
        {
            if (m_Requested[i])
            {
                if (kept != i)
                    m_Entries[kept] = m_Entries[i];
                m_Requested[kept] = 0;   // not requested yet by the next extraction
                ++kept;
            }
            else
            {
                Retire(m_Retired, m_Entries[i].map);
                ++removed;
            }
        }
        // shrinking never allocates
        m_Entries.resize(kept);
        m_Requested.resize(kept);
        return removed;
    }

    void LightShadowTable::CancelRequests() noexcept
    {
        std::fill(m_Requested.begin(), m_Requested.end(), uint8_t(0));
    }

    void LightShadowTable::ReleaseRetired() noexcept
    {
        for (engine::IShadowMap* map : m_Retired)
        {
            if (map)
                map->Release();
        }
        m_Retired.clear();
    }

    bool GetLightShadowRequest(const engine::Light* light, LightShadow& outShadow, LightShadowKind& outKind) noexcept
    {
        if (!light || !light->shadowMap || light->GetLightType() != LightType_Directional)
            return false;
        const CascadedShadowMap* hint = query_cast<CascadedShadowMap>(light->shadowMap.Get());
        const nvrhi::ITexture* texture = hint ? hint->GetTexture() : nullptr;
        if (!texture)
            return false;
        const uint32_t cascades = hint->GetCascadeCapacity();
        if (cascades == 0 || cascades > 4)
            return false;
        const nvrhi::TextureDesc& desc = texture->getDesc();
        outShadow.channel = light->shadowChannel;
        outShadow.cascadeCount = cascades;
        outShadow.resolution = desc.width;
        outShadow.format = desc.format;
        outKind = LightShadowKind::Cascaded;
        return true;
    }

    bool SetupLightShadowCascades(CascadedShadowMap* map, const engine::DirectionalLight* light,
                                  const engine::IView* view, const ShadowSettings& settings,
                                  dm::float3 preViewTranslation) noexcept
    {
        if (!map || !light || !view || !light->GetNode() || !AreShadowSettingsValid(view, settings))
            return false;
        try
        {
            if (settings.stable)
            {
                map->SetupForPlanarViewStable(*light, view->GetProjectionFrustum(), view->GetInverseViewMatrix(),
                                              settings.maxShadowDistance, settings.lightSpaceZUp,
                                              settings.lightSpaceZDown, settings.exponent, preViewTranslation);
            }
            else
            {
                map->SetupForPlanarView(*light, view->GetViewFrustum(), settings.maxShadowDistance,
                                        settings.lightSpaceZUp, settings.lightSpaceZDown, settings.exponent,
                                        preViewTranslation);
            }
        }
        catch (...)
        {
            return false;
        }
        return true;
    }

    bool AreShadowSettingsValid(const engine::IView* view, const ShadowSettings& settings) noexcept
    {
        if (!view || !(settings.exponent > 1.f) || !std::isfinite(settings.exponent))
            return false;
        const dm::frustum frustum = settings.stable ? view->GetProjectionFrustum() : view->GetViewFrustum();
        if (!(dm::length(frustum.planes[dm::frustum::NEAR_PLANE].normal) > 0.f))
            return false;
        if (settings.maxShadowDistance > 0.f)
            return std::isfinite(settings.maxShadowDistance);
        return dm::length(frustum.planes[dm::frustum::FAR_PLANE].normal) > 0.f;
    }
}
