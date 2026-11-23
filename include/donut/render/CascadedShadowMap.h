/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/View.h>
#include <nvrhi/nvrhi.h>

namespace donut::render
{
    class PlanarShadowMap;

    // A class ID, so that a light's shadow map can be recognised without dynamic_cast (nvrhi ADR 0006; query_cast):
    // the render service serves a directional light whose Light::shadowMap hint is a CascadedShadowMap (LightShadows.h).
    NVRHI_CLASS_CLSID(CascadedShadowMap, "ddc2b9ba-16a4-4777-a316-67540a428138")
    class CascadedShadowMap : public engine::IShadowMap
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(CascadedShadowMap)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(CascadedShadowMap)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_IMPLEMENTS_CLASS(CascadedShadowMap)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(engine::IShadowMap)
        NVRHI_END_INTERFACE_TABLE()
    private:
        nvrhi::TextureHandle m_ShadowMapTexture;
        std::vector<nvrhi::AutoPtr<PlanarShadowMap>> m_Cascades;
        std::vector<nvrhi::AutoPtr<PlanarShadowMap>> m_PerObjectShadows;
        engine::CompositeView m_CompositeView;
        int m_NumberOfCascades;

        void Init(nvrhi::IDevice* device, nvrhi::ITexture* texture, int numCascades, int numPerObjectShadows);

    public:
        CascadedShadowMap(
            nvrhi::IDevice* device,
            int resolution,
            int numCascades,
            int numPerObjectShadows,
            nvrhi::Format format,
            bool isUAV = false);

        // Over an existing texture array of numCascades + numPerObjectShadows slices (GetTextureDesc), cascades first.
        // `device` may be null: it is not used (the render service's device-free tests make maps over stand-in textures).
        CascadedShadowMap(
            nvrhi::IDevice* device,
            nvrhi::ITexture* texture,
            int numCascades,
            int numPerObjectShadows);

        // The description of the texture the first constructor makes.
        [[nodiscard]] static nvrhi::TextureDesc GetTextureDesc(
            int resolution,
            int numCascades,
            int numPerObjectShadows,
            nvrhi::Format format,
            bool isUAV = false);

        // Computes the cascade projections based on the view frustum, shadow distance, and the distribution exponent.
        bool SetupForPlanarView(
            const engine::DirectionalLight& light, 
            dm::frustum viewFrustum, 
            float maxShadowDistance, 
            float lightSpaceZUp, 
            float lightSpaceZDown, 
            float exponent = 4.f,
            dm::float3 preViewTranslation = 0.f,
            int numberOfCascades = -1);

        // Similar to SetupForPlanarView, but the size of the cascades does not depend on orientation, and therefore 
        // the shadow map texels have the same world space projections when the camera turns or moves.
        // The downside of this algorithm is that the cascades are often larger than necessary.
        bool SetupForPlanarViewStable(
            const engine::DirectionalLight& light, 
            dm::frustum projectionFrustum, 
            dm::affine3 inverseViewMatrix, 
            float maxShadowDistance, 
            float lightSpaceZUp, 
            float lightSpaceZDown, 
            float exponent = 4.f, 
            dm::float3 preViewTranslation = 0.f,
            int numberOfCascades = -1);

        // Computes the cascade projections to cover an omnidirectional view from a given point. The cascades are all centered on that point.
        bool SetupForCubemapView(
            const engine::DirectionalLight& light, 
            dm::float3 center, 
            float maxShadowDistance, float lightSpaceZUp, 
            float lightSpaceZDown, 
            float exponent = 4.f,
            int numberOfCascades = -1);

        // Computes a simple directional shadow projection that covers a given world space box.
        bool SetupPerObjectShadow(const engine::DirectionalLight& light, uint32_t object, const dm::box3& objectBounds);

        void SetupProxyViews();

        void Clear(nvrhi::ICommandList* commandList);

        void SetLitOutOfBounds(bool litOutOfBounds);
        void SetFalloffDistance(float distance);
		void SetNumberOfCascadesUnsafe(int cascades);
        // The first cascade's falloff distance (SetFalloffDistance sets every cascade's).
        [[nodiscard]] float GetFalloffDistance() const;
        // The cascades the texture holds (the constructor's numCascades); GetNumberOfCascades is the number set up.
        [[nodiscard]] uint32_t GetCascadeCapacity() const;

        engine::PlanarView* GetCascadeView(uint32_t cascade);
        engine::PlanarView* GetPerObjectView(uint32_t object);

        virtual dm::float4x4 GetWorldToUvzwMatrix() const override;
        virtual const engine::ICompositeView& GetView() const override;
        virtual nvrhi::ITexture* GetTexture() const override;
        virtual uint32_t GetNumberOfCascades() const override;
        virtual const IShadowMap* GetCascade(uint32_t index) const override;
        virtual uint32_t GetNumberOfPerObjectShadows() const override;
        virtual const IShadowMap* GetPerObjectShadow(uint32_t index) const override;
        virtual dm::int2 GetTextureSize() const override;
        virtual dm::box2 GetUVRange() const override;
        virtual dm::float2 GetFadeRangeInTexels() const override;
        virtual bool IsLitOutOfBounds() const override;
        virtual void FillShadowConstants(ShadowConstants& constants) const override;
    };
}