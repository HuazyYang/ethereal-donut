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

#include "rdg_corpus.h"

#include <algorithm>
#include <cstdio>
#include <random>

namespace rdgtest
{
    using nvrhi::Format;
    using nvrhi::ResourceStates;
    using nvrhi::ShaderType;

    void Builder::Remember(ResourceId id, const char* name)
    {
        if (id == InvalidResourceId)
            return;
        const uint32_t index = GetResourceIndex(id);
        if (m_Names.size() <= index)
            m_Names.resize(index + 1);
        m_Names[index] = name ? name : "";
    }

    const std::string& Builder::NameOf(uint32_t resource) const
    {
        static const std::string s_Unknown = "?";
        return resource < m_Names.size() ? m_Names[resource] : s_Unknown;
    }

    ResourceId Builder::Texture(const char* name, uint32_t width, uint32_t height, Format format, bool renderTarget,
                                bool uav, uint32_t mips, uint32_t slices, ResourceFlags flags)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.mipLevels = mips;
        desc.arraySize = slices;
        desc.dimension = slices > 1 ? nvrhi::TextureDimension::Texture2DArray : nvrhi::TextureDimension::Texture2D;
        desc.format = format;
        desc.isRenderTarget = renderTarget;
        desc.isUAV = uav;
        desc.debugName = name;
        const ResourceId id = m_Graph->CreateTexture(desc, flags);
        Remember(id, name);
        return id;
    }

    ResourceId Builder::Depth(const char* name, uint32_t width, uint32_t height, Format format, uint32_t slices)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.arraySize = slices;
        desc.dimension = slices > 1 ? nvrhi::TextureDimension::Texture2DArray : nvrhi::TextureDimension::Texture2D;
        desc.format = format;
        desc.isRenderTarget = true;
        desc.debugName = name;
        const ResourceId id = m_Graph->CreateTexture(desc, ResourceFlags::None);
        Remember(id, name);
        return id;
    }

    ResourceId Builder::Buffer(const char* name, uint64_t size, bool uav, bool vertex, bool index, bool indirect)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = size;
        desc.canHaveUAVs = uav;
        desc.canHaveRawViews = true;
        desc.isVertexBuffer = vertex;
        desc.isIndexBuffer = index;
        desc.isDrawIndirectArgs = indirect;
        desc.debugName = name;
        const ResourceId id = m_Graph->CreateBuffer(desc, ResourceFlags::None);
        Remember(id, name);
        return id;
    }

    ResourceId Builder::Import(const char* name, uint32_t width, uint32_t height, Format format, ResourceStates entry,
                               ResourceStates exit, bool renderTarget, bool uav)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.isRenderTarget = renderTarget;
        desc.isUAV = uav;
        desc.debugName = name;
        const ResourceId id = m_Graph->ImportTextureDesc(desc, entry, exit);
        Remember(id, name);
        return id;
    }

    ResourceId Builder::ImportBuffer(const char* name, uint64_t size, ResourceStates entry, ResourceStates exit, bool uav)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = size;
        desc.canHaveUAVs = uav;
        desc.canHaveRawViews = true;
        desc.debugName = name;
        const ResourceId id = m_Graph->ImportBufferDesc(desc, entry, exit);
        Remember(id, name);
        return id;
    }

    ResourceId Builder::BackBuffer(const char* name, uint32_t width, uint32_t height)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = Format::RGBA8_UNORM;
        desc.isRenderTarget = true;
        desc.debugName = name;
        desc.initialState = ResourceStates::Present;
        desc.keepInitialState = true;
        const ResourceId id = m_Graph->ImportTextureDesc(desc, ResourceStates::Present, ResourceStates::Present);
        Remember(id, name);
        return id;
    }

    ResourceId Builder::Persistent(const char* name, uint32_t width, uint32_t height, Format format)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.isUAV = true;
        desc.isRenderTarget = true;
        desc.debugName = name;
        const ResourceId id = m_Graph->GetPersistentTexture(name, desc);
        Remember(id, name);
        return id;
    }

    void Builder::History(const char* name, uint32_t width, uint32_t height, Format format, uint32_t layers, ResourceId* ids)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.isUAV = true;
        desc.isRenderTarget = true;
        desc.debugName = name;
        m_Graph->GetHistory(name, desc, layers, ids);
        for (uint32_t layer = 0; layer < layers; ++layer)
            Remember(ids[layer], name);
    }

    // ---------------------------------------------------------------------------------------------- graphs

    static const ClearValue c_Clear = { { 0, 0, 0, 0 }, 1.0f, 0 };

    void BuildDeferredFrame(Builder& b, bool asyncSsao)
    {
        const uint32_t w = 1920, h = 1080;
        ResourceId depth = b.Depth("Depth", w, h);
        ResourceId gb0 = b.Texture("GB0", w, h, Format::RGBA8_UNORM, true, false);
        ResourceId gb1 = b.Texture("GB1", w, h, Format::RGBA8_UNORM, true, false);
        ResourceId normals = b.Texture("Normals", w, h, Format::RGBA16_FLOAT, true, false);
        ResourceId gb3 = b.Texture("GB3", w, h, Format::RGBA8_UNORM, true, false);
        ResourceId mv = b.Texture("MV", w, h, Format::RG16_FLOAT, true, false);
        ResourceId ao = b.Texture("AO", w, h, Format::R16_FLOAT, false, true);
        ResourceId shadow = b.Depth("ShadowMap", 2048, 2048, Format::D32);
        ResourceId hdr = b.Texture("HDR", w, h, Format::RGBA16_FLOAT, true, true);
        ResourceId history[2];
        b.History("TAA", w, h, Format::RGBA16_FLOAT, 2, history);
        ResourceId back = b.BackBuffer("BackBuffer", w, h);

        b.Pass("DepthPrepass", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
            depth = c->SetDepthTarget(depth, false, AccessFlags::None, &c_Clear);
        });
        b.Pass("GBuffer", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
            c->SetDepthTarget(depth, true, AccessFlags::None, nullptr);
            gb0 = c->SetColorTarget(0, gb0, {}, AccessFlags::None, &c_Clear);
            gb1 = c->SetColorTarget(1, gb1, {}, AccessFlags::None, &c_Clear);
            normals = c->SetColorTarget(2, normals, {}, AccessFlags::None, &c_Clear);
            gb3 = c->SetColorTarget(3, gb3, {}, AccessFlags::None, &c_Clear);
            mv = c->SetColorTarget(4, mv, {}, AccessFlags::None, &c_Clear);
        });
        b.Pass("SSAO", PassType::Compute, asyncSsao ? PassFlags::AsyncCompute : PassFlags::None, [&](ISetupContext* c) {
            Read(c, depth, Access::ShaderRead);
            Read(c, normals, Access::ShaderRead);
            ao = Write(c, ao, Access::UAVWrite);
        });
        b.Pass("Shadow", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
            shadow = c->SetDepthTarget(shadow, false, AccessFlags::None, &c_Clear);
        });
        b.Pass("DeferredLighting", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            Read(c, depth, Access::ShaderRead);
            Read(c, gb0, Access::ShaderRead);
            Read(c, gb1, Access::ShaderRead);
            Read(c, normals, Access::ShaderRead);
            Read(c, gb3, Access::ShaderRead);
            Read(c, ao, Access::ShaderRead);
            Read(c, shadow, Access::ShaderRead);
            hdr = Write(c, hdr, Access::UAVWrite);
        });
        b.Pass("ForwardTransparent", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
            c->SetDepthTarget(depth, true, AccessFlags::None, nullptr);
            hdr = c->SetColorTarget(0, hdr, {}, AccessFlags::None, nullptr);
        });
        b.Pass("TAA", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            Read(c, hdr, Access::ShaderRead);
            Read(c, mv, Access::ShaderRead);
            Read(c, history[1], Access::ShaderRead);
            history[0] = Write(c, history[0], Access::UAVWrite);
        });
        b.Pass("ToneMap", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
            Read(c, history[0], Access::ShaderRead, ShaderType::Pixel);
            back = c->SetColorTarget(0, back, {}, AccessFlags::FullWrite, nullptr);
        });
    }

    void BuildRandomGraph(Builder& b, uint32_t seed, uint32_t passCount, bool computeOnly, bool async)
    {
        std::mt19937 rng(seed * 7919u + 13u);
        auto pick = [&](uint32_t n) { return uint32_t(rng() % std::max(1u, n)); };

        struct Entry { ResourceId id; bool buffer; };
        std::vector<Entry> written;
        static const char* const c_Names[] = { "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10",
            "R11", "R12", "R13", "R14", "R15" };
        uint32_t created = 0;
        const Format formats[] = { Format::RGBA16_FLOAT, Format::RGBA8_UNORM, Format::R32_FLOAT };
        const uint32_t sizes[] = { 256, 512, 1024 };
        ResourceId output = b.Import("Output", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                     ResourceStates::ShaderResource);
        ResourceId persistent = (seed % 3 == 0) ? b.Persistent("Accum", 512, 512, Format::RGBA16_FLOAT) : InvalidResourceId;
        bool persistentWritten = false;

        for (uint32_t p = 0; p < passCount; ++p)
        {
            const bool raster = !computeOnly && pick(3) == 0;
            const bool asyncPass = async && !raster && pick(3) == 0;
            std::vector<uint32_t> reads;
            for (uint32_t r = 0, count = pick(3); r < count && !written.empty(); ++r)
            {
                const uint32_t candidate = pick(uint32_t(written.size()));
                if (std::find(reads.begin(), reads.end(), candidate) == reads.end())
                    reads.push_back(candidate);
            }

            // Either a new transient, or a full rewrite of one that this pass does not read.
            int32_t target = -1;
            if (!written.empty() && pick(3) == 0)
            {
                const uint32_t candidate = pick(uint32_t(written.size()));
                if (std::find(reads.begin(), reads.end(), candidate) == reads.end() && !written[candidate].buffer)
                    target = int32_t(candidate);
            }
            if (target < 0)
            {
                const bool buffer = !raster && pick(4) == 0;
                const char* name = c_Names[created % 16];
                ++created;
                ResourceId id = buffer ? b.Buffer(name, 65536u * (1 + pick(4)), true)
                                       : b.Texture(name, sizes[pick(3)], sizes[pick(3)], formats[pick(3)], true, true);
                written.push_back({ id, buffer });
                target = int32_t(written.size() - 1);
            }
            const bool rmwPersistent = persistent != InvalidResourceId && !raster && pick(4) == 0;
            const bool readPersistent = persistent != InvalidResourceId && persistentWritten && !rmwPersistent && pick(4) == 0;

            static const char* const c_PassNames[] = { "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
                "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23",
                "P24", "P25", "P26", "P27", "P28", "P29", "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37",
                "P38", "P39" };
            b.Pass(c_PassNames[p % 40], raster ? PassType::Raster : PassType::Compute,
                   asyncPass ? PassFlags::AsyncCompute : PassFlags::None, [&](ISetupContext* c) {
                for (uint32_t r : reads)
                    Read(c, written[r].id, Access::ShaderRead);
                Entry& out = written[target];
                if (raster)
                    out.id = c->SetColorTarget(0, out.id, {}, AccessFlags::None, &c_Clear);
                else
                    out.id = Write(c, out.id, Access::UAVWrite);
                if (rmwPersistent)
                {
                    persistent = Write(c, persistent, Access::UAVRead | Access::UAVWrite, AccessFlags::None);
                    persistentWritten = true;
                }
                if (readPersistent)
                    Read(c, persistent, Access::ShaderRead);
            });
        }

        b.Pass("Resolve", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
            for (uint32_t r = 0; r < std::min<uint32_t>(4, uint32_t(written.size())); ++r)
                Read(c, written[written.size() - 1 - r].id, Access::ShaderRead);
            if (persistentWritten)
                Read(c, persistent, Access::ShaderRead);
            output = Write(c, output, Access::UAVWrite);
        });
    }

    void BenchSetup::Init(uint32_t passes, uint32_t perPass)
    {
        passCount = passes;
        resourcesPerPass = perPass;
        for (uint32_t i = 0; i < 2; ++i)
        {
            descs[i] = nvrhi::TextureDesc();
            descs[i].width = 512u << i;
            descs[i].height = 512;
            descs[i].format = i ? Format::RGBA8_UNORM : Format::RGBA16_FLOAT;
            descs[i].isRenderTarget = true;
            descs[i].isUAV = true;
            descs[i].debugName = i ? "B" : "A";
        }
        outputDesc = nvrhi::TextureDesc();
        outputDesc.width = 1024;
        outputDesc.height = 1024;
        outputDesc.format = Format::RGBA16_FLOAT;
        outputDesc.isUAV = true;
        outputDesc.debugName = "Out";
        live.clear();
        live.reserve(size_t(passes) * perPass);
    }

    void RecordBenchFrame(IRenderGraph* graph, BenchSetup& setup)
    {
        setup.live.clear();
        ResourceId output = graph->ImportTextureDesc(setup.outputDesc, ResourceStates::ShaderResource,
                                                     ResourceStates::ShaderResource);
        const uint32_t perPass = setup.resourcesPerPass;
        for (uint32_t p = 0; p < setup.passCount; ++p)
        {
            const bool raster = (p % 3) == 1;
            const size_t base = setup.live.size();
            for (uint32_t r = 0; r < perPass; ++r)
                setup.live.push_back(graph->CreateTexture(setup.descs[r % 2], ResourceFlags::None));
            const PassDesc desc{ "Bench", raster ? PassType::Raster : PassType::Compute, PassFlags::None, 0 };
            AddLambdaPass<NoData>(graph, desc,
                [&setup, &output, p, base, raster, perPass](ISetupContext* c, NoData&) {
                    for (uint32_t r = 0; r < 4 && base >= size_t(r + 1) * perPass; ++r)
                        Read(c, setup.live[base - size_t(r + 1) * perPass + (p % perPass)], Access::ShaderRead);
                    for (uint32_t r = 0; r < perPass; ++r)
                    {
                        ResourceId& id = setup.live[base + r];
                        if (raster && r < 8)
                            id = c->SetColorTarget(r, id, {}, AccessFlags::FullWrite, nullptr);
                        else
                            id = Write(c, id, Access::UAVWrite);
                    }
                    if (p + 1 == setup.passCount)
                        output = Write(c, output, Access::UAVWrite);
                },
                [](IPassContext*, const NoData&) { });
        }
    }

    // ---------------------------------------------------------------------------------------------- the corpus

    static std::vector<CorpusGraph> MakeCorpus()
    {
        std::vector<CorpusGraph> corpus;

        corpus.push_back({ "deferred", true, false, [](Builder& b, uint32_t variant) {
            BuildDeferredFrame(b, false);
            if (variant)
                b.Graph()->SetPassEnabled(b.Passes()[5], false);   // no transparent pass
        } });
        corpus.push_back({ "deferred_async_ssao", true, true, [](Builder& b, uint32_t variant) {
            BuildDeferredFrame(b, variant == 0);
        } });

        corpus.push_back({ "bloom_mip_chain", false, false, [](Builder& b, uint32_t variant) {
            const uint32_t mips = 6;
            ResourceId hdr = b.Import("HDR", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            ResourceId bloom = b.Texture("Bloom", 512, 512, Format::RGBA16_FLOAT, false, true, mips);
            ResourceId out = b.Import("Composite", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::UnorderedAccess,
                                      ResourceStates::ShaderResource);
            for (uint32_t m = 0; m < mips; ++m)
            {
                b.Pass("Down", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    if (m == 0)
                        Read(c, hdr, Access::ShaderRead);
                    else
                        Read(c, bloom, Access::ShaderRead, ShaderType::None, Mip(m - 1));
                    bloom = Write(c, bloom, Access::UAVWrite, AccessFlags::FullWrite, ShaderType::None, Mip(m));
                });
            }
            for (uint32_t m = mips - 1; m-- > (variant ? 2u : 0u);)
            {
                b.Pass("Up", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    Read(c, bloom, Access::ShaderRead, ShaderType::None, Mip(m + 1));
                    bloom = Write(c, bloom, Access::UAVRead | Access::UAVWrite, AccessFlags::None, ShaderType::None, Mip(m));
                });
            }
            b.Pass("Composite", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, hdr, Access::ShaderRead);
                Read(c, bloom, Access::ShaderRead, ShaderType::None, Mip(0));
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "hiz_chain", false, false, [](Builder& b, uint32_t variant) {
            const uint32_t mips = variant ? 6 : 8;
            ResourceId depth = b.Import("SceneDepth", 1024, 1024, Format::R32_FLOAT, ResourceStates::ShaderResource,
                                        ResourceStates::ShaderResource, false, false);
            ResourceId hiz = b.Texture("HiZ", 512, 512, Format::R32_FLOAT, false, true, mips);
            ResourceId visible = b.ImportBuffer("Visible", 65536, ResourceStates::UnorderedAccess, ResourceStates::ShaderResource);
            for (uint32_t m = 0; m < mips; ++m)
            {
                b.Pass("HiZ", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    if (m == 0)
                        Read(c, depth, Access::ShaderRead);
                    else
                        Read(c, hiz, Access::ShaderRead, ShaderType::None, Mip(m - 1));
                    hiz = Write(c, hiz, Access::UAVWrite, AccessFlags::FullWrite, ShaderType::None, Mip(m));
                });
            }
            b.Pass("Cull", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, hiz, Access::ShaderRead);
                visible = Write(c, visible, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "uav_ping_pong", false, false, [](Builder& b, uint32_t variant) {
            ResourceId a = b.Buffer("PingA", 1 << 20, true);
            ResourceId bb = b.Buffer("PingB", 1 << 20, true);
            ResourceId out = b.ImportBuffer("Result", 1 << 20, ResourceStates::Common, ResourceStates::ShaderResource);
            b.Pass("Init", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { a = Write(c, a, Access::UAVWrite); });
            for (uint32_t i = 0; i < (variant ? 3u : 6u); ++i)
            {
                b.Pass("Step", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    if (i % 2 == 0) { Read(c, a, Access::ShaderRead); bb = Write(c, bb, Access::UAVWrite); }
                    else { Read(c, bb, Access::ShaderRead); a = Write(c, a, Access::UAVWrite); }
                });
            }
            b.Pass("Copy", PassType::Copy, PassFlags::None, [&](ISetupContext* c) {
                Read(c, (variant ? bb : a), Access::CopySource);
                out = Write(c, out, Access::CopyDest);
            });
        } });

        corpus.push_back({ "relaxed_uav_run", false, false, [](Builder& b, uint32_t variant) {
            ResourceId acc = b.Buffer("Atomics", 1 << 16, true);
            ResourceId out = b.ImportBuffer("Out", 1 << 16, ResourceStates::UnorderedAccess, ResourceStates::UnorderedAccess);
            b.Pass("Clear", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                acc = Write(c, acc, Access::UAVWrite, AccessFlags::FullWrite | AccessFlags::RelaxedOrder);
            });
            for (uint32_t i = 0; i < 3; ++i)
            {
                const AccessFlags flags = (variant && i == 1) ? AccessFlags::None : AccessFlags::RelaxedOrder;
                b.Pass("Accumulate", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    acc = Write(c, acc, Access::UAVRead | Access::UAVWrite, flags);
                });
            }
            b.Pass("Reduce", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, acc, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "copy_upload_draw", true, false, [](Builder& b, uint32_t variant) {
            ResourceId staging = b.ImportBuffer("Staging", 1 << 20, ResourceStates::CopySource, ResourceStates::CopySource, false);
            ResourceId vertices = b.Buffer("Vertices", 1 << 20, false, true);
            ResourceId target = b.Import("Target", 1024, 1024, Format::RGBA8_UNORM, ResourceStates::RenderTarget,
                                         ResourceStates::ShaderResource);
            b.Pass("Upload", PassType::Copy, variant ? PassFlags::AsyncCompute : PassFlags::None, [&](ISetupContext* c) {
                Read(c, staging, Access::CopySource);
                vertices = Write(c, vertices, Access::CopyDest);
            });
            b.Pass("Draw", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, vertices, Access::VertexBuffer);
                target = c->SetColorTarget(0, target, {}, AccessFlags::None, &c_Clear);
            });
        } });

        corpus.push_back({ "persistent_accumulation", false, false, [](Builder& b, uint32_t variant) {
            ResourceId input = b.Import("Input", 512, 512, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                        ResourceStates::ShaderResource, false, false);
            ResourceId accum = b.Persistent("Accum", 512, 512, Format::RGBA16_FLOAT);
            ResourceId out = b.Import("Out", 512, 512, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("Accumulate", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, input, Access::ShaderRead);
                accum = Write(c, accum, Access::UAVRead | Access::UAVWrite, AccessFlags::None);
            });
            b.Pass("Show", variant ? PassType::Compute : PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, accum, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "taa_history_pair", true, false, [](Builder& b, uint32_t variant) {
            ResourceId color = b.Texture("Color", 1280, 720, Format::RGBA16_FLOAT, true, false);
            ResourceId history[2];
            b.History("TAA", 1280, 720, Format::RGBA16_FLOAT, 2, history);
            ResourceId back = b.BackBuffer("BackBuffer", 1280, 720);
            b.Pass("Scene", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                color = c->SetColorTarget(0, color, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("TAA", variant ? PassType::Raster : PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, color, Access::ShaderRead);
                Read(c, history[1], Access::ShaderRead);
                if (variant)
                    history[0] = c->SetColorTarget(0, history[0], {}, AccessFlags::FullWrite, nullptr);
                else
                    history[0] = Write(c, history[0], Access::UAVWrite);
            });
            b.Pass("Blit", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, history[0], Access::ShaderRead, ShaderType::Pixel);
                back = c->SetColorTarget(0, back, {}, AccessFlags::FullWrite, nullptr);
            });
        } });

        corpus.push_back({ "history_three_layers", false, false, [](Builder& b, uint32_t variant) {
            ResourceId noisy = b.Import("Noisy", 512, 512, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                        ResourceStates::ShaderResource, false, false);
            ResourceId history[3];
            b.History("Denoise", 512, 512, Format::RGBA16_FLOAT, 3, history);
            ResourceId out = b.Import("Out", 512, 512, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("Temporal", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, noisy, Access::ShaderRead);
                Read(c, history[1], Access::ShaderRead);
                if (!variant)
                    Read(c, history[2], Access::ShaderRead);
                history[0] = Write(c, history[0], Access::UAVWrite);
            });
            b.Pass("Output", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, history[0], Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "import_exit_states", true, false, [](Builder& b, uint32_t variant) {
            ResourceId target = b.Import("Probe", 256, 256, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                         variant ? ResourceStates::RenderTarget : ResourceStates::ShaderResource);
            ResourceId out = b.ImportBuffer("SH", 4096, ResourceStates::UnorderedAccess, ResourceStates::ShaderResource);
            b.Pass("RenderProbe", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                target = c->SetColorTarget(0, target, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("Project", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, target, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "backbuffer_ui", true, false, [](Builder& b, uint32_t variant) {
            ResourceId scene = b.Texture("Scene", 1280, 720, Format::RGBA16_FLOAT, true, false);
            ResourceId back = b.BackBuffer("BackBuffer", 1280, 720);
            b.Pass("Scene", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                scene = c->SetColorTarget(0, scene, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("ToneMap", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, scene, Access::ShaderRead, ShaderType::Pixel);
                back = c->SetColorTarget(0, back, {}, AccessFlags::FullWrite, nullptr);
            });
            if (!variant)
            {
                b.Pass("UI", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                    back = c->SetColorTarget(0, back, {}, AccessFlags::None, nullptr);
                });
            }
        } });

        corpus.push_back({ "depth_stencil_planes", true, false, [](Builder& b, uint32_t variant) {
            ResourceId ds = b.Depth("DepthStencil", 1024, 1024, Format::D24S8);
            ResourceId out = b.Import("Out", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("Prepass", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                ds = c->SetDepthTarget(ds, false, AccessFlags::None, &c_Clear);
            });
            b.Pass("SampleDepth", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) {
                Read(c, ds, Access::ShaderRead, ShaderType::None, Planes(nvrhi::PlaneMask::Depth));
            });
            b.Pass("StencilMark", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                // Writes stencil while testing and sampling depth (5.3).
                Read(c, ds, Access::DepthRead | Access::ShaderRead, ShaderType::Pixel, Planes(nvrhi::PlaneMask::Depth));
                ds = Write(c, ds, Access::StencilWrite, AccessFlags::None, ShaderType::None, Planes(nvrhi::PlaneMask::Stencil));
            });
            b.Pass("Resolve", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, ds, Access::ShaderRead, ShaderType::None, Planes(variant ? nvrhi::PlaneMask::All : nvrhi::PlaneMask::Stencil));
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "shadow_cascades_slices", true, false, [](Builder& b, uint32_t variant) {
            const uint32_t cascades = variant ? 2 : 4;
            ResourceId shadow = b.Depth("Cascades", 2048, 2048, Format::D32, 4);
            ResourceId out = b.Import("Lit", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            for (uint32_t i = 0; i < 4; ++i)
            {
                b.Pass("Cascade", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                    shadow = Write(c, shadow, Access::DepthWrite, i < cascades ? AccessFlags::Clear : AccessFlags::FullWrite,
                                   ShaderType::None, Slice(i));
                });
            }
            b.Pass("Light", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, shadow, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "pass_local_scratch", false, false, [](Builder& b, uint32_t variant) {
            ResourceId depth = b.Import("Depth", 1024, 1024, Format::R32_FLOAT, ResourceStates::ShaderResource,
                                        ResourceStates::ShaderResource, false, false);
            ResourceId ao = b.Texture("AO", 1024, 1024, Format::R16_FLOAT, false, true);
            ResourceId out = b.Import("Out", 1024, 1024, Format::R16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            nvrhi::TextureDesc scratchDesc;
            scratchDesc.width = 256;
            scratchDesc.height = 256;
            scratchDesc.arraySize = 16;
            scratchDesc.dimension = nvrhi::TextureDimension::Texture2DArray;
            scratchDesc.format = Format::R32_FLOAT;
            scratchDesc.isUAV = true;
            scratchDesc.debugName = "Deinterleaved";
            b.Pass("SSAO", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                ResourceId scratch = c->CreateTransientTexture(scratchDesc, ResourceFlags::None);
                Read(c, depth, Access::ShaderRead);
                Write(c, scratch, Access::UAVRead | Access::UAVWrite);
                ao = Write(c, ao, Access::UAVWrite);
            });
            b.Pass("Blur", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                ResourceId scratch = c->CreateTransientTexture(scratchDesc, ResourceFlags::None);
                Write(c, scratch, Access::UAVRead | Access::UAVWrite);
                Read(c, ao, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
            if (variant)
                b.Graph()->SetPassEnabled(b.Passes()[1], false);
        } });

        corpus.push_back({ "exclusion_group_ao", false, false, [](Builder& b, uint32_t variant) {
            ResourceId depth = b.Import("Depth", 1024, 1024, Format::R32_FLOAT, ResourceStates::ShaderResource,
                                        ResourceStates::ShaderResource, false, false);
            ResourceId ao = b.Texture("AO", 1024, 1024, Format::R16_FLOAT, false, true);
            ResourceId out = b.Import("Out", 1024, 1024, Format::R16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            ResourceId hbaoOut = InvalidResourceId;
            nvrhi::TextureDesc scratch;
            scratch.width = 512;
            scratch.height = 512;
            scratch.format = Format::R32_FLOAT;
            scratch.isUAV = true;
            scratch.debugName = "AOScratch";
            const PassId hbao = b.Pass("HBAO", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                ResourceId s = c->CreateTransientTexture(scratch, ResourceFlags::None);
                Read(c, depth, Access::ShaderRead);
                Write(c, s, Access::UAVRead | Access::UAVWrite);
                ao = Write(c, ao, Access::UAVWrite);
                hbaoOut = ao;
            }, 1);
            const PassId ssao = b.Pass("SSAO", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                ResourceId s = c->CreateTransientTexture(scratch, ResourceFlags::None);
                Read(c, depth, Access::ShaderRead);
                Write(c, s, Access::UAVRead | Access::UAVWrite);
                ao = Write(c, ao, Access::UAVWrite);
                c->SetDisabledOutput(ao, DisabledOutput::Passthrough, hbaoOut);
            }, 1);
            b.Pass("Composite", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, ao, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
            b.Graph()->SetPassEnabled(variant ? ssao : hbao, false);
        } });

        corpus.push_back({ "disabled_substitute", false, false, [](Builder& b, uint32_t variant) {
            ResourceId depth = b.Import("Depth", 1024, 1024, Format::R32_FLOAT, ResourceStates::ShaderResource,
                                        ResourceStates::ShaderResource, false, false);
            nvrhi::TextureDesc aoDesc;
            aoDesc.width = 1024;
            aoDesc.height = 1024;
            aoDesc.format = Format::R16_FLOAT;
            aoDesc.isUAV = true;
            aoDesc.debugName = "AO";
            ResourceId ao = b.Graph()->CreateTexture(aoDesc, ResourceFlags::None);
            aoDesc.debugName = "WhiteAO";
            ResourceId white = b.Graph()->ImportTextureDesc(aoDesc, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
            ResourceId out = b.Import("Lit", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            const PassId ssao = b.Pass("SSAO", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, depth, Access::ShaderRead);
                ao = Write(c, ao, Access::UAVWrite);
                c->SetDisabledOutput(ao, DisabledOutput::Substitute, white);
            });
            b.Pass("Lighting", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, ao, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
            if (variant)
                b.Graph()->SetPassEnabled(ssao, false);
        } });

        corpus.push_back({ "disabled_passthrough_bloom", false, false, [](Builder& b, uint32_t variant) {
            nvrhi::TextureDesc desc;
            desc.width = 1024;
            desc.height = 1024;
            desc.format = Format::RGBA16_FLOAT;
            desc.isUAV = true;
            desc.debugName = "HDR";
            ResourceId hdr = b.Graph()->CreateTexture(desc, ResourceFlags::None);
            desc.debugName = "Bloomed";
            ResourceId bloomed = b.Graph()->CreateTexture(desc, ResourceFlags::None);
            ResourceId out = b.Import("LDR", 1024, 1024, Format::RGBA8_UNORM, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("Lighting", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { hdr = Write(c, hdr, Access::UAVWrite); });
            const PassId bloom = b.Pass("Bloom", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, hdr, Access::ShaderRead);
                bloomed = Write(c, bloomed, Access::UAVWrite);
                c->SetDisabledOutput(bloomed, DisabledOutput::Passthrough, hdr);
            });
            b.Pass("ToneMap", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, bloomed, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
            if (variant)
                b.Graph()->SetPassEnabled(bloom, false);
        } });

        corpus.push_back({ "disabled_cull_cascade", false, false, [](Builder& b, uint32_t variant) {
            ResourceId main = b.Texture("Main", 512, 512, Format::RGBA8_UNORM, false, true);
            ResourceId debug = b.Texture("Debug", 512, 512, Format::RGBA8_UNORM, false, true);
            ResourceId overlay = b.Import("Overlay", 512, 512, Format::RGBA8_UNORM, ResourceStates::ShaderResource,
                                          ResourceStates::ShaderResource);
            ResourceId out = b.Import("Out", 512, 512, Format::RGBA8_UNORM, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("Main", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { main = Write(c, main, Access::UAVWrite); });
            const PassId viz = b.Pass("DebugViz", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, main, Access::ShaderRead);
                debug = Write(c, debug, Access::UAVWrite);
            });
            b.Pass("DebugOverlay", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, debug, Access::ShaderRead);
                overlay = Write(c, overlay, Access::UAVWrite);
            });
            b.Pass("Output", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, main, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
            if (variant)
                b.Graph()->SetPassEnabled(viz, false);
        } });

        corpus.push_back({ "aliasing_chain", true, false, [](Builder& b, uint32_t variant) {
            const Format formats[] = { Format::RGBA16_FLOAT, Format::RGBA8_UNORM, Format::RG16_FLOAT, Format::R32_FLOAT };
            static const char* const names[] = { "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7" };
            ResourceId previous = InvalidResourceId;
            ResourceId out = b.Import("Out", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            const uint32_t count = variant ? 6 : 8;
            for (uint32_t i = 0; i < count; ++i)
            {
                ResourceId t = b.Texture(names[i], 1024 >> (i % 2), 1024, formats[i % 4], true, true);
                b.Pass("Stage", i % 2 ? PassType::Raster : PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    if (previous != InvalidResourceId)
                        Read(c, previous, Access::ShaderRead);
                    if (i % 2)
                        t = c->SetColorTarget(0, t, {}, AccessFlags::None, &c_Clear);
                    else
                        t = Write(c, t, Access::UAVWrite);
                });
                previous = t;
            }
            b.Pass("Final", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, previous, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "pooled_reuse", false, false, [](Builder& b, uint32_t variant) {
            ResourceId out = b.Import("Out", 512, 512, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            ResourceId previous = InvalidResourceId;
            for (uint32_t i = 0; i < (variant ? 4u : 6u); ++i)
            {
                ResourceId t = b.Texture("Ping", 512, 512, Format::RGBA16_FLOAT, false, true);
                b.Pass("Filter", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    if (previous != InvalidResourceId)
                        Read(c, previous, Access::ShaderRead);
                    t = Write(c, t, Access::UAVWrite);
                });
                previous = t;
            }
            b.Pass("Final", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, previous, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "cross_queue_transient", true, true, [](Builder& b, uint32_t variant) {
            ResourceId t = b.Texture("Volume", 256, 256, Format::RGBA16_FLOAT, false, true);
            ResourceId scene = b.Texture("Scene", 1024, 1024, Format::RGBA16_FLOAT, true, false);
            ResourceId back = b.BackBuffer("BackBuffer", 1024, 1024);
            b.Pass("Simulate", PassType::Compute, variant ? PassFlags::None : PassFlags::AsyncCompute, [&](ISetupContext* c) {
                t = Write(c, t, Access::UAVWrite);
            });
            b.Pass("Scene", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                scene = c->SetColorTarget(0, scene, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("Composite", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, t, Access::ShaderRead, ShaderType::Pixel);
                Read(c, scene, Access::ShaderRead, ShaderType::Pixel);
                back = c->SetColorTarget(0, back, {}, AccessFlags::FullWrite, nullptr);
            });
        } });

        corpus.push_back({ "async_chain", true, true, [](Builder& b, uint32_t variant) {
            ResourceId t1 = b.Texture("T1", 512, 512, Format::RGBA16_FLOAT, false, true);
            ResourceId t2 = b.Texture("T2", 512, 512, Format::RGBA16_FLOAT, false, true);
            ResourceId g = b.Texture("G", 1024, 1024, Format::RGBA8_UNORM, true, false);
            ResourceId out = b.Import("Out", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("A", PassType::Compute, PassFlags::AsyncCompute, [&](ISetupContext* c) { t1 = Write(c, t1, Access::UAVWrite); });
            b.Pass("D", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                g = c->SetColorTarget(0, g, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("B", PassType::Compute, variant ? PassFlags::None : PassFlags::AsyncCompute, [&](ISetupContext* c) {
                Read(c, t1, Access::ShaderRead);
                t2 = Write(c, t2, Access::UAVWrite);
            });
            b.Pass("C", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, t2, Access::ShaderRead);
                Read(c, g, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "extract_transient", false, false, [](Builder& b, uint32_t variant) {
            ResourceId t = b.Texture("Exported", 256, 256, Format::RGBA8_UNORM, false, true);
            b.Pass("Produce", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { t = Write(c, t, Access::UAVWrite); });
            b.Graph()->Extract(t, variant ? ResourceStates::CopySource : ResourceStates::ShaderResource, nullptr);
        } });

        corpus.push_back({ "gpu_driven_draw", true, false, [](Builder& b, uint32_t variant) {
            ResourceId vb = b.Buffer("VB", 1 << 20, true, true);
            ResourceId ib = b.Buffer("IB", 1 << 18, true, false, true);
            ResourceId args = b.Buffer("Args", 4096, true, false, false, true);
            ResourceId target = b.Import("Target", 1024, 1024, Format::RGBA8_UNORM, ResourceStates::ShaderResource,
                                         ResourceStates::ShaderResource);
            b.Pass("Generate", PassType::Compute, variant ? PassFlags::AsyncCompute : PassFlags::None, [&](ISetupContext* c) {
                vb = Write(c, vb, Access::UAVWrite);
                ib = Write(c, ib, Access::UAVWrite);
                args = Write(c, args, Access::UAVWrite);
            });
            b.Pass("Draw", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, vb, Access::VertexBuffer);
                Read(c, ib, Access::IndexBuffer);
                Read(c, args, Access::IndirectArgs);
                target = c->SetColorTarget(0, target, {}, AccessFlags::None, &c_Clear);
            });
        } });

        corpus.push_back({ "clears", true, false, [](Builder& b, uint32_t variant) {
            ResourceId rt = b.Texture("RT", 1024, 1024, Format::RGBA8_UNORM, true, false);
            ResourceId depth = b.Depth("Depth", 1024, 1024, Format::D32);
            ResourceId counters = b.Texture("Counters", 64, 64, Format::R32_UINT, false, true);
            ResourceId out = b.Import("Out", 1024, 1024, Format::RGBA8_UNORM, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("Draw", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                rt = c->SetColorTarget(0, rt, {}, AccessFlags::None, &c_Clear);
                depth = c->SetDepthTarget(depth, false, AccessFlags::None, &c_Clear);
            });
            b.Pass("Count", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, rt, Access::ShaderRead);
                counters = Write(c, counters, Access::UAVRead | Access::UAVWrite, variant ? AccessFlags::FullWrite : AccessFlags::Clear);
            });
            b.Pass("Show", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, counters, Access::ShaderRead);
                Read(c, depth, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "sampling_and_copy_readers", true, false, [](Builder& b, uint32_t variant) {
            ResourceId t = b.Texture("Shared", 512, 512, Format::RGBA8_UNORM, true, false);
            ResourceId readback = b.ImportBuffer("Readback", 1 << 20, ResourceStates::CopyDest, ResourceStates::CopyDest, false);
            ResourceId a = b.Import("A", 512, 512, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
            b.Pass("Render", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                t = c->SetColorTarget(0, t, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("SampleA", PassType::Compute, PassFlags::SideEffect, [&](ISetupContext* c) { Read(c, t, Access::ShaderRead); });
            b.Pass("Readback", PassType::Copy, PassFlags::None, [&](ISetupContext* c) {
                Read(c, t, Access::CopySource);
                readback = Write(c, readback, Access::CopyDest);
            });
            b.Pass("SampleB", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, t, Access::ShaderRead);
                a = Write(c, a, Access::UAVWrite);
            });
            if (variant)
                b.Graph()->SetPassEnabled(b.Passes()[1], false);
        } });

        corpus.push_back({ "sampling_and_uav_readers", false, false, [](Builder& b, uint32_t variant) {
            ResourceId t = b.Texture("Field", 512, 512, Format::RGBA16_FLOAT, false, true);
            ResourceId out = b.Import("Out", 512, 512, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            ResourceId out2 = b.Import("Out2", 512, 512, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                       ResourceStates::ShaderResource);
            b.Pass("Write", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { t = Write(c, t, Access::UAVWrite); });
            b.Pass("Sample", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, t, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
            b.Pass("Load", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, t, Access::UAVRead);
                out2 = Write(c, out2, Access::UAVWrite);
            });
            if (!variant)
            {
                b.Pass("SampleAgain", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                    Read(c, t, Access::ShaderRead);
                    out = Write(c, out, Access::UAVWrite);
                });
            }
        } });

        corpus.push_back({ "depth_test_and_sampling", true, false, [](Builder& b, uint32_t variant) {
            ResourceId depth = b.Depth("Depth", 1024, 1024, Format::D32);
            ResourceId color = b.Texture("Color", 1024, 1024, Format::RGBA16_FLOAT, true, false);
            ResourceId out = b.Import("Out", 1024, 1024, Format::RGBA16_FLOAT, ResourceStates::ShaderResource,
                                      ResourceStates::ShaderResource);
            b.Pass("Depth", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                depth = c->SetDepthTarget(depth, false, AccessFlags::None, &c_Clear);
            });
            b.Pass("Opaque", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                c->SetDepthTarget(depth, true, AccessFlags::None, nullptr);
                color = c->SetColorTarget(0, color, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("Fog", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, depth, Access::ShaderRead, ShaderType::Pixel);
                color = c->SetColorTarget(0, color, {}, AccessFlags::None, nullptr);
            });
            b.Pass("Particles", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                c->SetDepthTarget(depth, true, AccessFlags::None, nullptr);
                Read(c, depth, Access::ShaderRead, ShaderType::Pixel);
                color = c->SetColorTarget(0, color, {}, AccessFlags::None, nullptr);
            });
            b.Pass("Post", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, color, Access::ShaderRead);
                if (!variant)
                    Read(c, depth, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "culling", false, false, [](Builder& b, uint32_t variant) {
            ResourceId unused = b.Texture("Unused", 256, 256, Format::RGBA8_UNORM, false, true);
            ResourceId used = b.Texture("Used", 256, 256, Format::RGBA8_UNORM, false, true);
            ResourceId query = b.Buffer("Query", 4096, true);
            ResourceId out = b.Import("Out", 256, 256, Format::RGBA8_UNORM, ResourceStates::ShaderResource, ResourceStates::ShaderResource);
            b.Pass("Dead", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { unused = Write(c, unused, Access::UAVWrite); });
            b.Pass("Live", PassType::Compute, PassFlags::None, [&](ISetupContext* c) { used = Write(c, used, Access::UAVWrite); });
            b.Pass("Readback", PassType::Compute, variant ? PassFlags::None : PassFlags::SideEffect, [&](ISetupContext* c) {
                query = Write(c, query, Access::UAVWrite);
            });
            b.Pass("Out", PassType::Compute, PassFlags::None, [&](ISetupContext* c) {
                Read(c, used, Access::ShaderRead);
                out = Write(c, out, Access::UAVWrite);
            });
        } });

        corpus.push_back({ "async_exit_on_graphics", true, true, [](Builder& b, uint32_t variant) {
            ResourceId volume = b.Import("Volume", 256, 256, Format::RGBA16_FLOAT, ResourceStates::PixelShaderResource,
                                         ResourceStates::PixelShaderResource);
            ResourceId scene = b.Texture("Scene", 512, 512, Format::RGBA8_UNORM, true, false);
            ResourceId back = b.BackBuffer("BackBuffer", 512, 512);
            b.Pass("Scene", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                scene = c->SetColorTarget(0, scene, {}, AccessFlags::None, &c_Clear);
            });
            b.Pass("Present", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                Read(c, scene, Access::ShaderRead, ShaderType::Pixel);
                back = c->SetColorTarget(0, back, {}, AccessFlags::FullWrite, nullptr);
            });
            b.Pass("UpdateVolume", PassType::Compute, variant ? PassFlags::None : PassFlags::AsyncCompute, [&](ISetupContext* c) {
                volume = Write(c, volume, Access::UAVWrite);
            });
            // The import's exit state (pixel-shader resource) is illegal on the compute queue: the exit transition
            // goes to this graphics pass, which waits for the async one (6.8).
            b.Pass("Overlay", PassType::Raster, PassFlags::None, [&](ISetupContext* c) {
                back = c->SetColorTarget(0, back, {}, AccessFlags::None, nullptr);
            });
        } });

        // Random graphs.
        for (uint32_t seed = 1; seed <= 20; ++seed)
        {
            static char names[20][16];
            std::snprintf(names[seed - 1], sizeof(names[seed - 1]), "random_%02u", seed);
            const bool computeOnly = seed % 4 == 0;
            const bool async = seed % 2 == 1;
            corpus.push_back({ names[seed - 1], !computeOnly, async, [seed, computeOnly, async](Builder& b, uint32_t variant) {
                BuildRandomGraph(b, seed + variant * 1000, 6 + seed % 16, computeOnly, async);
            } });
        }
        return corpus;
    }

    const std::vector<CorpusGraph>& GetCorpus()
    {
        static const std::vector<CorpusGraph> s_Corpus = MakeCorpus();
        return s_Corpus;
    }
}
