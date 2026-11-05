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

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <nvrhi/core/foundation.h>
#include <donut/render/RenderGraph.h>
#include <nvrhi/nvrhi.h>

namespace donut::render
{
    // CommonRenderPasses::BlitTexture in a render graph (aggregate plan, step P1.5): one raster use that samples a
    // mip and slice of `source` and draws into `targetRegion` (one mip; the slices of a layered target) of `target`.
    struct BlitGraphInputs
    {
        ResourceId source = InvalidResourceId;
        uint32_t sourceMip = 0;
        uint32_t sourceArraySlice = 0;
        dm::box2 sourceBox = dm::box2(0.f, 1.f);
        nvrhi::Format sourceFormat = nvrhi::Format::UNKNOWN;
        ResourceId target = InvalidResourceId;
        nvrhi::SubresourceRegion targetRegion = nvrhi::SubresourceRegion().setMips(0, 1);
        dm::box2 targetBox = dm::box2(0.f, 1.f);
        engine::BlitSampler sampler = engine::BlitSampler::Linear;
        // The blit covers the whole target region (the default boxes): the graph may discard what was there.
        bool fullWrite = true;
    };

    struct BlitGraphOutputs
    {
        ResourceId target = InvalidResourceId;
    };

    class BlitPass : public nvrhi::ObjectImpl<IGpuPass>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(BlitPass)
        NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        BlitPass(nvrhi::IDevice* device, engine::CommonRenderPasses* commonPasses);

        BlitGraphOutputs AddToGraph(IRenderGraph* graph, const BlitGraphInputs& inputs);

        nvrhi::FRESULT Setup(ISetupContext* context, void* data) noexcept override;
        nvrhi::FRESULT Prepare(IPrepareContext* context, void* data) noexcept override;
        nvrhi::FRESULT Execute(IPassContext* context, const void* data) noexcept override;
        void OnResourcesReleased() noexcept override;

    private:
        nvrhi::AutoPtr<engine::CommonRenderPasses> m_CommonPasses;
        engine::BindingCache m_BindingCache;
    };
}
