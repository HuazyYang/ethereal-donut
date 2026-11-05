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

// Diagnostics (design 11.3): text forms of states and barriers, and the JSON and Graphviz dumps of a plan. Off the
// hot path: they allocate.

#include "RenderGraphInternal.h"

#include <cstdio>

namespace donut::render::rdg
{
    using nvrhi::ResourceStates;

    std::string StatesToString(ResourceStates states)
    {
        static const struct { ResourceStates bit; const char* name; } c_Names[] = {
            { ResourceStates::Common, "Common" },
            { ResourceStates::ConstantBuffer, "ConstantBuffer" },
            { ResourceStates::VertexBuffer, "VertexBuffer" },
            { ResourceStates::IndexBuffer, "IndexBuffer" },
            { ResourceStates::IndirectArgument, "IndirectArgument" },
            { ResourceStates::PixelShaderResource, "PixelShaderResource" },
            { ResourceStates::NonPixelShaderResource, "NonPixelShaderResource" },
            { ResourceStates::UnorderedAccess, "UnorderedAccess" },
            { ResourceStates::RenderTarget, "RenderTarget" },
            { ResourceStates::DepthWrite, "DepthWrite" },
            { ResourceStates::DepthRead, "DepthRead" },
            { ResourceStates::StreamOut, "StreamOut" },
            { ResourceStates::CopyDest, "CopyDest" },
            { ResourceStates::CopySource, "CopySource" },
            { ResourceStates::ResolveDest, "ResolveDest" },
            { ResourceStates::ResolveSource, "ResolveSource" },
            { ResourceStates::Present, "Present" },
            { ResourceStates::AccelStructRead, "AccelStructRead" },
            { ResourceStates::AccelStructWrite, "AccelStructWrite" },
            { ResourceStates::AccelStructBuildInput, "AccelStructBuildInput" },
            { ResourceStates::AccelStructBuildBlas, "AccelStructBuildBlas" },
            { ResourceStates::ShadingRateSurface, "ShadingRateSurface" },
        };
        if (states == ResourceStates::Unknown)
            return "Unknown";
        std::string text;
        for (const auto& entry : c_Names)
        {
            if ((states & entry.bit) == 0)
                continue;
            if (!text.empty())
                text += '|';
            text += entry.name;
        }
        return text.empty() ? std::to_string(uint32_t(states)) : text;
    }

    std::string AccessToString(Access access)
    {
        static const struct { Access bit; const char* name; } c_Names[] = {
            { Access::VertexBuffer, "VertexBuffer" }, { Access::IndexBuffer, "IndexBuffer" },
            { Access::IndirectArgs, "IndirectArgs" }, { Access::ConstantBuffer, "ConstantBuffer" },
            { Access::ShaderRead, "ShaderRead" }, { Access::UAVRead, "UAVRead" },
            { Access::ColorTargetRead, "ColorTargetRead" }, { Access::DepthRead, "DepthRead" },
            { Access::StencilRead, "StencilRead" }, { Access::CopySource, "CopySource" },
            { Access::ResolveSource, "ResolveSource" }, { Access::ShadingRateRead, "ShadingRateRead" },
            { Access::AccelStructRead, "AccelStructRead" }, { Access::AccelStructBuildInput, "AccelStructBuildInput" },
            { Access::Present, "Present" }, { Access::HostRead, "HostRead" },
            { Access::UAVWrite, "UAVWrite" }, { Access::ColorTarget, "ColorTarget" },
            { Access::DepthWrite, "DepthWrite" }, { Access::StencilWrite, "StencilWrite" },
            { Access::CopyDest, "CopyDest" }, { Access::ResolveDest, "ResolveDest" },
            { Access::AccelStructWrite, "AccelStructWrite" },
        };
        std::string text;
        for (const auto& entry : c_Names)
        {
            if ((access & entry.bit) == Access::None)
                continue;
            if (!text.empty())
                text += '|';
            text += entry.name;
        }
        return text.empty() ? std::string("None") : text;
    }

    const char* QueueName(nvrhi::CommandQueue queue) noexcept
    {
        switch (queue)
        {
        case nvrhi::CommandQueue::Graphics: return "graphics";
        case nvrhi::CommandQueue::Compute:  return "compute";
        case nvrhi::CommandQueue::Copy:     return "copy";
        default:                            return "unknown";
        }
    }

    const char* ReasonName(BarrierReason reason) noexcept
    {
        switch (reason)
        {
        case BarrierReason::RAW:             return "RAW";
        case BarrierReason::WAR:             return "WAR";
        case BarrierReason::WAW:             return "WAW";
        case BarrierReason::UavOrder:        return "UavOrder";
        case BarrierReason::ReadGroupChange: return "ReadGroupChange";
        case BarrierReason::Alias:           return "Alias";
        case BarrierReason::QueueTransfer:   return "QueueTransfer";
        case BarrierReason::Discard:         return "Discard";
        case BarrierReason::ExitState:       return "ExitState";
        case BarrierReason::Persistent:      return "Persistent";
        case BarrierReason::Debug:           return "Debug";
        default:                             return "?";
        }
    }

    const char* KindName(ResourceKind kind) noexcept
    {
        switch (kind)
        {
        case ResourceKind::Transient:  return "transient";
        case ResourceKind::PassLocal:  return "pass-local";
        case ResourceKind::Imported:   return "imported";
        case ResourceKind::HostKept:   return "host-kept";
        case ResourceKind::Persistent: return "persistent";
        case ResourceKind::History:    return "history";
        default:                       return "?";
        }
    }

    static std::string ResourceLabel(const CompiledRenderGraph& plan, uint32_t index)
    {
        if (index == c_None)
            return "*";
        if (index >= plan.m_Resources.size())
            return "#" + std::to_string(index);
        const PlanResource& resource = plan.m_Resources[index];
        std::string name = plan.GetName(resource.nameOffset);
        if (name.empty())
            name = "#" + std::to_string(index);
        if (resource.kind == ResourceKind::History)
            name += "[" + std::to_string(resource.historyLayer) + "]";
        return name;
    }

    static std::string RegionToString(const nvrhi::SubresourceRegion& region)
    {
        if (region.baseMip == 0 && region.mipCount == nvrhi::SubresourceRegion::AllMipLevels && region.baseSlice == 0 &&
            region.sliceCount == nvrhi::SubresourceRegion::AllArraySlices && region.planes == nvrhi::PlaneMask::All)
            return "";
        std::string text = " mips " + std::to_string(region.baseMip) + "+" + std::to_string(region.mipCount) +
            " slices " + std::to_string(region.baseSlice) + "+" + std::to_string(region.sliceCount);
        if (region.planes != nvrhi::PlaneMask::All)
            text += region.planes == nvrhi::PlaneMask::Depth ? " depth" : region.planes == nvrhi::PlaneMask::Stencil ? " stencil" : " planes";
        return text;
    }

    std::string DescribeBarrier(const CompiledRenderGraph& plan, const PlanBarrier& barrier)
    {
        std::string text;
        switch (barrier.type)
        {
        case nvrhi::BarrierType::Uav:
            text = "UAV " + ResourceLabel(plan, barrier.resource);
            break;
        case nvrhi::BarrierType::Aliasing:
            text = "Alias " + ResourceLabel(plan, barrier.aliasBefore) + " -> " + ResourceLabel(plan, barrier.resource);
            break;
        default:
            text = ResourceLabel(plan, barrier.resource) + RegionToString(barrier.region) + " " +
                StatesToString(barrier.before) + " -> " + StatesToString(barrier.after);
            if ((barrier.flags & nvrhi::BarrierFlags::SplitBegin) != 0) text += " [begin]";
            if ((barrier.flags & nvrhi::BarrierFlags::SplitEnd) != 0) text += " [end]";
            if ((barrier.flags & nvrhi::BarrierFlags::Discard) != 0) text += " [discard]";
            if ((barrier.flags & nvrhi::BarrierFlags::QueueRelease) != 0)
                text += std::string(" [release to ") + QueueName(barrier.otherQueue) + "]";
            if ((barrier.flags & nvrhi::BarrierFlags::QueueAcquire) != 0)
                text += std::string(" [acquire from ") + QueueName(barrier.otherQueue) + "]";
            break;
        }
        text += " (";
        text += ReasonName(barrier.reason);
        text += ")";
        return text;
    }

    static std::string JsonString(const std::string& text)
    {
        std::string out = "\"";
        for (char c : text)
        {
            switch (c)
            {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default:
                if (uint8_t(c) < 0x20)
                {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", unsigned(uint8_t(c)));
                    out += buffer;
                }
                else
                    out += c;
            }
        }
        return out + "\"";
    }

    static const char* PassTypeName(PassType type)
    {
        switch (type)
        {
        case PassType::Raster: return "raster";
        case PassType::Compute: return "compute";
        case PassType::Copy: return "copy";
        case PassType::RayTracing: return "rayTracing";
        }
        return "?";
    }

    // One declared use: "<resource> <access>", then the stages, flags, region and attachment that differ from the
    // defaults. The golden access lists of donut's passes compare these (plan step P1.5).
    static std::string DescribeUse(const CompiledRenderGraph& plan, const PlanUse& use)
    {
        std::string text = ResourceLabel(plan, use.resource) + " " + AccessToString(use.access);
        if (use.stages != nvrhi::ShaderType::None)
        {
            char stages[16];
            std::snprintf(stages, sizeof(stages), "0x%04x", uint32_t(use.stages));
            text += std::string(" stages=") + stages;
        }
        const struct { AccessFlags flag; const char* name; } flags[] = {
            { AccessFlags::Clear, "Clear" }, { AccessFlags::FullWrite, "FullWrite" },
            { AccessFlags::RelaxedOrder, "RelaxedOrder" }, { AccessFlags::Preserve, "Preserve" } };
        for (const auto& flag : flags)
        {
            if ((use.flags & flag.flag) != AccessFlags::None)
                text += std::string(" ") + flag.name;
        }
        const nvrhi::SubresourceRegion& region = use.region;
        text += " mips=" + std::to_string(region.baseMip) + "+" + std::to_string(region.mipCount) +
            " slices=" + std::to_string(region.baseSlice) + "+" + std::to_string(region.sliceCount);
        if (use.attachment != c_NoAttachment)
            text += use.attachment == c_DepthAttachment ? " depthTarget" : " colorTarget" + std::to_string(use.attachment);
        return text;
    }

    static void WriteJson(const CompiledRenderGraph& plan, std::string& out)
    {
        static const char* const c_SlotNames[c_SlotCount] = { "prologue", "postClear", "epilogue" };
        out += "{\n  \"planHash\": \"" + std::to_string(plan.m_Key) + "\",\n";
        out += "  \"enableMaskHash\": \"" + std::to_string(plan.m_MaskHash) + "\",\n";

        out += "  \"passes\": [\n";
        for (size_t s = 0; s < plan.m_Passes.size(); ++s)
        {
            const PlanPass& pass = plan.m_Passes[s];
            out += "    { \"index\": " + std::to_string(s) + ", \"pass\": " + std::to_string(pass.program) +
                ", \"name\": " + JsonString(plan.GetName(pass.nameOffset)) + ", \"type\": \"" + PassTypeName(pass.type) +
                "\", \"queue\": \"" + QueueName(pass.queue) +
                "\", \"segment\": " + (pass.segment == c_None ? std::string("null") : std::to_string(pass.segment)) +
                ", \"enabled\": " + (pass.enabled ? "true" : "false");
            const PassFlags modes = pass.flags & (PassFlags::Tracked | PassFlags::Restore | PassFlags::AsyncCompute);
            out += std::string(", \"mode\": \"") +
                ((pass.flags & PassFlags::Restore) != PassFlags::None ? "restore" :
                 (pass.flags & PassFlags::Tracked) != PassFlags::None ? "tracked" : "native") +
                ((modes & PassFlags::AsyncCompute) != PassFlags::None ? "+async" : "") + "\"";
            out += ", \"uses\": [";
            for (uint32_t u = 0; u < pass.useCount; ++u)
                out += (u ? ", " : "") + JsonString(DescribeUse(plan, plan.m_Uses[pass.useBegin + u]));
            out += "]";
            for (uint32_t slot = 0; slot < c_SlotCount; ++slot)
            {
                out += std::string(", \"") + c_SlotNames[slot] + "\": [";
                for (uint32_t b = 0; b < pass.batchCount[slot]; ++b)
                {
                    const PlanBarrier& barrier = plan.m_Barriers[pass.batchBegin[slot] + b];
                    out += (b ? ", " : "") + JsonString(DescribeBarrier(plan, barrier));
                }
                out += "]";
            }
            out += ", \"waits\": [";
            for (uint32_t w = 0; w < pass.waitCount; ++w)
            {
                const PassWait& wait = plan.m_PassWaits[pass.waitBegin + w];
                out += std::string(w ? ", " : "") + "{ \"pass\": " + std::to_string(wait.producer) + ", \"frame\": " +
                    std::to_string(int(wait.frameDelta)) + " }";
            }
            out += "] }";
            out += s + 1 < plan.m_Passes.size() ? ",\n" : "\n";
        }
        out += "  ],\n";

        out += "  \"segments\": [\n";
        for (size_t i = 0; i < plan.m_Segments.size(); ++i)
        {
            const PlanSegment& segment = plan.m_Segments[i];
            out += "    { \"queue\": \"" + std::string(QueueName(segment.queue)) + "\", \"firstPass\": " +
                std::to_string(segment.firstPass) + ", \"passCount\": " + std::to_string(segment.passCount) +
                ", \"signals\": " + (segment.signals ? "true" : "false") + ", \"waits\": [";
            for (uint32_t w = 0; w < segment.waitCount; ++w)
            {
                const SegmentWait& wait = plan.m_SegmentWaits[segment.waitBegin + w];
                out += std::string(w ? ", " : "") + "{ \"queue\": \"" + QueueName(wait.queue) + "\", \"segment\": " +
                    std::to_string(wait.segment) + ", \"frame\": " + std::to_string(int(wait.frameDelta)) + " }";
            }
            out += "] }";
            out += i + 1 < plan.m_Segments.size() ? ",\n" : "\n";
        }
        out += "  ],\n";

        out += "  \"resources\": [\n";
        for (size_t r = 0; r < plan.m_Resources.size(); ++r)
        {
            const PlanResource& resource = plan.m_Resources[r];
            out += "    { \"index\": " + std::to_string(r) + ", \"name\": " + JsonString(ResourceLabel(plan, uint32_t(r))) +
                ", \"kind\": \"" + KindName(resource.kind) + "\", \"buffer\": " + (resource.isBuffer ? "true" : "false") +
                ", \"first\": " + (resource.firstPass == c_None ? std::string("null") : std::to_string(resource.firstPass)) +
                ", \"last\": " + (resource.lastPass == c_None ? std::string("null") : std::to_string(resource.lastPass)) +
                ", \"physical\": " + (resource.physical == c_None ? std::string("null") : std::to_string(resource.physical)) +
                ", \"heap\": " + (resource.heap == c_None ? std::string("null") : std::to_string(resource.heap)) +
                ", \"offset\": " + std::to_string(resource.heapOffset) + ", \"size\": " + std::to_string(resource.size) +
                ", \"aliased\": " + (resource.aliased ? "true" : "false") + ", \"crossQueue\": " +
                (resource.crossQueue ? "true" : "false") + ", \"entry\": " + JsonString(StatesToString(resource.entry)) + " }";
            out += r + 1 < plan.m_Resources.size() ? ",\n" : "\n";
        }
        out += "  ],\n";

        const PlanStatistics& stats = plan.m_Statistics;
        out += "  \"statistics\": { \"scheduledPasses\": " + std::to_string(stats.scheduledPasses) +
            ", \"culledPasses\": " + std::to_string(stats.culledPasses) + ", \"segments\": " + std::to_string(stats.segments) +
            ", \"barriers\": " + std::to_string(stats.barriers) + ", \"transitions\": " + std::to_string(stats.transitions) +
            ", \"splitBarriers\": " + std::to_string(stats.splitBarriers) + ", \"uavBarriers\": " +
            std::to_string(stats.uavBarriers) + ", \"aliasingBarriers\": " + std::to_string(stats.aliasingBarriers) +
            ", \"queueTransfers\": " + std::to_string(stats.queueTransfers) + ", \"fenceWaits\": " +
            std::to_string(stats.fenceWaits) + ", \"fenceSignals\": " + std::to_string(stats.fenceSignals) +
            ", \"heaps\": " + std::to_string(stats.heaps) + ", \"poolableTransientBytes\": " +
            std::to_string(stats.poolableTransientBytes) + ", \"placedPeakBytes\": " + std::to_string(stats.placedPeakBytes) +
            ", \"compileMicroseconds\": " + std::to_string(stats.compileMicroseconds) + " }\n}\n";
    }

    static void WriteGraphviz(const CompiledRenderGraph& plan, std::string& out)
    {
        out += "digraph RenderGraph {\n  rankdir=LR;\n  node [shape=box, fontname=\"Consolas\"];\n";
        for (uint32_t q = 0; q < 3; ++q)
        {
            out += "  subgraph cluster_" + std::to_string(q) + " {\n    label=\"" +
                QueueName(nvrhi::CommandQueue(q)) + "\";\n";
            for (size_t s = 0; s < plan.m_Passes.size(); ++s)
            {
                const PlanPass& pass = plan.m_Passes[s];
                if (uint32_t(pass.queue) != q)
                    continue;
                out += "    p" + std::to_string(s) + " [label=" + JsonString(std::to_string(s) + ": " +
                    plan.GetName(pass.nameOffset)) + (pass.enabled ? "" : ", style=dashed") + "];\n";
            }
            out += "  }\n";
        }
        // Resource flow: each use's producer, the previous writer of the resource in the schedule.
        std::vector<uint32_t> lastWriter(plan.m_Resources.size(), c_None);
        for (size_t s = 0; s < plan.m_Passes.size(); ++s)
        {
            const PlanPass& pass = plan.m_Passes[s];
            for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount; ++u)
            {
                const PlanUse& use = plan.m_Uses[u];
                const uint32_t writer = lastWriter[use.resource];
                if (writer != c_None && writer != s)
                    out += "  p" + std::to_string(writer) + " -> p" + std::to_string(s) + " [label=" +
                        JsonString(ResourceLabel(plan, use.resource)) + "];\n";
            }
            for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount; ++u)
            {
                if (plan.m_Uses[u].isWrite)
                    lastWriter[plan.m_Uses[u].resource] = uint32_t(s);
            }
            for (uint32_t w = 0; w < pass.waitCount; ++w)
            {
                const PassWait& wait = plan.m_PassWaits[pass.waitBegin + w];
                if (wait.frameDelta == 0)
                    out += "  p" + std::to_string(wait.producer) + " -> p" + std::to_string(s) +
                        " [style=dotted, color=red, label=\"fence\"];\n";
            }
        }
        out += "}\n";
    }

    nvrhi::FRESULT CompiledRenderGraph::WriteDump(DumpFormat format, nvrhi::string& out) noexcept
    {
        try
        {
            std::string text;
            if (format == DumpFormat::Json)
                WriteJson(*this, text);
            else if (format == DumpFormat::Graphviz)
                WriteGraphviz(*this, text);
            else
                return nvrhi::FE_INVALID_ARGS;
            out = text.c_str();
            return nvrhi::FS_OK;
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
    }
}
