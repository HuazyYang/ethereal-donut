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

// The light loops of DeferredLightingPass::Render and ForwardShadingPass::PrepareLights, shared by their two input
// forms (plan step E5, sub-plan item E5.6): the scene graph's leaf vector (std::vector<AutoPtr<Light>>) and the
// snapshot's extracted lights (ExtractedLight, RenderExtract.h). Both forms give the same bytes: an extracted light's
// LightConstants were filled by FillLightConstants into zeroed memory at extraction, as the passes fill them into
// their zeroed constant structs, and the shadow loop reads the light's shadow map (the leaf's own, or the render
// service's) in the same order. Module-private; included by the two passes and by the device-free test that compares
// the forms. The constant structs are template parameters, so that this header needs none of the shader headers
// (they need `using namespace donut::math` before them).

#pragma once

#include <donut/core/log.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShadowMap.h>
#include <donut/render/RenderExtract.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace donut::render::lighting
{
    // The leaf vector: constants and shadow map read from each light at the call.
    struct LeafLights
    {
        const std::vector<nvrhi::AutoPtr<engine::Light>>* lights = nullptr;

        [[nodiscard]] uint32_t Count() const { return lights ? uint32_t(lights->size()) : 0u; }
        [[nodiscard]] const engine::IShadowMap* ShadowMap(uint32_t i) const { return (*lights)[i]->shadowMap.Get(); }
        template <typename LightConstantsT>
        void Fill(uint32_t i, LightConstantsT& constants) const { (*lights)[i]->FillLightConstants(constants); }
    };

    // The snapshot's lights: constants copied byte for byte, the extracted shadow map (the service's or the leaf's).
    struct SnapshotLights
    {
        const ExtractedLight* lights = nullptr;
        uint32_t count = 0;

        [[nodiscard]] uint32_t Count() const { return lights ? count : 0u; }
        [[nodiscard]] const engine::IShadowMap* ShadowMap(uint32_t i) const { return lights[i].shadowMap; }
        template <typename LightConstantsT>
        void Fill(uint32_t i, LightConstantsT& constants) const
        {
            // the bytes copied are the extracted LightConstants (light_cb.h), the passes' constant element type
            static_assert(std::is_same_v<LightConstantsT, ::LightConstants>,
                          "the lighting passes' light constants are the snapshot's LightConstants");
            if (lights[i].constants)
                std::memcpy(&constants, lights[i].constants, sizeof(LightConstantsT));
        }
    };

    // A light's shadows into the pass's shadow array while there is room: its cascades, then its per-object shadows,
    // their indices into the light's constants.
    template <typename LightConstantsT, typename ShadowConstantsT>
    void AppendShadows(const engine::IShadowMap* shadowMap, LightConstantsT& light, ShadowConstantsT* shadows,
                       int maxShadows, int& numShadows)
    {
        if (!shadowMap)
            return;
        for (uint32_t cascade = 0; cascade < shadowMap->GetNumberOfCascades(); cascade++)
        {
            if (numShadows < maxShadows)
            {
                shadowMap->GetCascade(cascade)->FillShadowConstants(shadows[numShadows]);
                light.shadowCascades[cascade] = numShadows;
                ++numShadows;
            }
        }

        for (uint32_t perObjectShadow = 0; perObjectShadow < shadowMap->GetNumberOfPerObjectShadows(); perObjectShadow++)
        {
            if (numShadows < maxShadows)
            {
                shadowMap->GetPerObjectShadow(perObjectShadow)->FillShadowConstants(shadows[numShadows]);
                light.perObjectShadows[perObjectShadow] = numShadows;
                ++numShadows;
            }
        }
    }

    // DeferredLightingPass::Render's loop into `constants` (zeroed by the caller): the lights in order, up to
    // `maxLights` (a warning beyond, as before), each with its shadows; `shadowMapTexture` and
    // constants.shadowMapTextureSize from the first shadowed light. False when two shadowed lights use different
    // textures (the pass binds one): the caller reports it and draws nothing.
    template <typename Lights, typename DeferredConstantsT>
    bool FillDeferredLights(const Lights& lights, DeferredConstantsT& constants, int maxLights, int maxShadows,
                            nvrhi::ITexture*& shadowMapTexture)
    {
        int numShadows = 0;
        for (uint32_t i = 0; i < lights.Count(); ++i)
        {
            const engine::IShadowMap* shadowMap = lights.ShadowMap(i);
            if (shadowMap)
            {
                if (!shadowMapTexture)
                {
                    shadowMapTexture = shadowMap->GetTexture();
                    constants.shadowMapTextureSize = dm::float2(shadowMap->GetTextureSize());
                }
                else if (shadowMapTexture != shadowMap->GetTexture())
                    return false;
            }

            if (int(constants.numLights) >= maxLights)
            {
                log::warning("Maximum number of active lights (%d) exceeded in DeferredLightingPass", maxLights);
                break;
            }

            auto& lightConstants = constants.lights[constants.numLights];
            lights.Fill(i, lightConstants);
            AppendShadows(shadowMap, lightConstants, constants.shadows, maxShadows, numShadows);
            ++constants.numLights;
        }
        return true;
    }

    // ForwardShadingPass::PrepareLights' loop into `constants` (zeroed by the caller): the shadow texture and its size
    // from the first light with a shadow map (of all the lights), then the first `maxLights` lights with their shadows.
    template <typename Lights, typename ForwardConstantsT>
    void FillForwardLights(const Lights& lights, ForwardConstantsT& constants, int maxLights, int maxShadows,
                           nvrhi::ITexture*& shadowMapTexture)
    {
        dm::int2 shadowMapTextureSize = 0;
        for (uint32_t i = 0; i < lights.Count(); ++i)
        {
            if (const engine::IShadowMap* shadowMap = lights.ShadowMap(i))
            {
                shadowMapTexture = shadowMap->GetTexture();
                shadowMapTextureSize = shadowMap->GetTextureSize();
                break;
            }
        }

        constants.shadowMapTextureSize = dm::float2(shadowMapTextureSize);
        constants.shadowMapTextureSizeInv = 1.f / constants.shadowMapTextureSize;

        int numShadows = 0;
        const int count = std::min(int(lights.Count()), maxLights);
        for (int i = 0; i < count; ++i)
        {
            auto& lightConstants = constants.lights[constants.numLights];
            lights.Fill(uint32_t(i), lightConstants);
            AppendShadows(lights.ShadowMap(uint32_t(i)), lightConstants, constants.shadows, maxShadows, numShadows);
            ++constants.numLights;
        }
    }
}
