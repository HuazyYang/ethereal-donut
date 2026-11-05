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

// Device-free graphs for the render graph's tests and rdg_bench (aggregate plan, step P1.3). Each graph records one
// frame into an IRenderGraph through the public interface only.

#pragma once

#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphLambdaPass.h>

#include <functional>
#include <string>
#include <vector>

namespace rdgtest
{
    using namespace donut::render;

    struct NoData
    {
        uint32_t unused;
    };

    // Short helpers over ISetupContext.
    inline void Read(ISetupContext* context, ResourceId id, Access access,
                     nvrhi::ShaderType stages = nvrhi::ShaderType::None,
                     const nvrhi::SubresourceRegion& region = nvrhi::SubresourceRegion())
    {
        context->Read(id, access, stages, region);
    }

    inline ResourceId Write(ISetupContext* context, ResourceId id, Access access,
                            AccessFlags flags = AccessFlags::FullWrite,
                            nvrhi::ShaderType stages = nvrhi::ShaderType::None,
                            const nvrhi::SubresourceRegion& region = nvrhi::SubresourceRegion())
    {
        return context->Write(id, access, stages, region, flags);
    }

    inline nvrhi::SubresourceRegion Mip(uint32_t mip)
    {
        return nvrhi::SubresourceRegion().setMips(mip, 1);
    }

    inline nvrhi::SubresourceRegion Slice(uint32_t slice)
    {
        return nvrhi::SubresourceRegion().setSlices(slice, 1);
    }

    inline nvrhi::SubresourceRegion Planes(nvrhi::PlaneMask planes)
    {
        return nvrhi::SubresourceRegion().setPlanes(planes);
    }

    // Records resources and passes; names the resources for golden listings.
    class Builder
    {
    public:
        explicit Builder(IRenderGraph* graph) : m_Graph(graph) { }

        IRenderGraph* Graph() const { return m_Graph; }

        ResourceId Texture(const char* name, uint32_t width, uint32_t height, nvrhi::Format format, bool renderTarget,
                           bool uav, uint32_t mips = 1, uint32_t slices = 1, ResourceFlags flags = ResourceFlags::None);
        ResourceId Depth(const char* name, uint32_t width, uint32_t height, nvrhi::Format format = nvrhi::Format::D24S8,
                         uint32_t slices = 1);
        ResourceId Buffer(const char* name, uint64_t size, bool uav, bool vertex = false, bool index = false,
                          bool indirect = false);
        ResourceId Import(const char* name, uint32_t width, uint32_t height, nvrhi::Format format,
                          nvrhi::ResourceStates entry, nvrhi::ResourceStates exit, bool renderTarget = true, bool uav = true);
        ResourceId ImportBuffer(const char* name, uint64_t size, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit,
                                bool uav = true);
        // A host-kept stand-in: keepInitialState, initial state Present.
        ResourceId BackBuffer(const char* name, uint32_t width, uint32_t height);
        ResourceId Persistent(const char* name, uint32_t width, uint32_t height, nvrhi::Format format);
        void History(const char* name, uint32_t width, uint32_t height, nvrhi::Format format, uint32_t layers, ResourceId* ids);

        template <typename Setup>
        PassId Pass(const char* name, PassType type, PassFlags flags, Setup&& setup, uint32_t exclusionGroup = 0)
        {
            PassId id = InvalidPassId;
            PassDesc desc{ name, type, flags, exclusionGroup };
            AddLambdaPass<NoData>(m_Graph, desc,
                [&setup](ISetupContext* context, NoData&) { setup(context); },
                [](IPassContext*, const NoData&) { },
                &id);
            m_Passes.push_back(id);
            return id;
        }

        const std::vector<PassId>& Passes() const { return m_Passes; }
        const std::string& NameOf(uint32_t resource) const;

    private:
        void Remember(ResourceId id, const char* name);

        IRenderGraph* m_Graph;
        std::vector<PassId> m_Passes;
        std::vector<std::string> m_Names;
    };

    struct CorpusGraph
    {
        const char* name;
        bool raster;     // has raster passes: not for CUDA
        bool async;      // has AsyncCompute passes
        // variant 0 is the frame; variant 1 is the frame after a plan switch (another mask or another graph).
        std::function<void(Builder&, uint32_t variant)> build;
    };

    const std::vector<CorpusGraph>& GetCorpus();

    // The worked example of design 6.11.
    void BuildDeferredFrame(Builder& builder, bool asyncSsao);

    // A random valid graph; computeOnly for profiles without raster passes.
    void BuildRandomGraph(Builder& builder, uint32_t seed, uint32_t passCount, bool computeOnly, bool async);

    // The rdg_bench graph: `passCount` passes, `resourcesPerPass` transients each, every pass reading one resource of
    // each of the four passes before it. Recording allocates nothing once the setup and the graph's arena have grown:
    // the descriptions are made once, and the ids live in a reserved vector.
    struct BenchSetup
    {
        uint32_t passCount = 0;
        uint32_t resourcesPerPass = 0;
        nvrhi::TextureDesc descs[2];
        nvrhi::TextureDesc outputDesc;
        std::vector<ResourceId> live;

        void Init(uint32_t passes, uint32_t perPass);
    };
    void RecordBenchFrame(IRenderGraph* graph, BenchSetup& setup);
}
