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

#include <donut/app/RenderGraphVisualizer.h>

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace donut::app;
using namespace donut::render;

namespace
{
    const char* QueueName(nvrhi::CommandQueue queue)
    {
        switch (queue)
        {
        case nvrhi::CommandQueue::Graphics: return "graphics";
        case nvrhi::CommandQueue::Compute:  return "compute";
        case nvrhi::CommandQueue::Copy:     return "copy";
        default:                            return "?";
        }
    }

    const char* ReasonName(BarrierReason reason)
    {
        static const char* const c_Names[] = { "RAW", "WAR", "WAW", "UavOrder", "ReadGroupChange", "Alias",
                                               "QueueTransfer", "Discard", "ExitState", "Persistent", "Debug" };
        const uint32_t index = uint32_t(reason);
        return index < sizeof(c_Names) / sizeof(c_Names[0]) ? c_Names[index] : "?";
    }

    void DescribeBarrier(const PlanBarrier& barrier, char* text, size_t size)
    {
        switch (barrier.type)
        {
        case nvrhi::BarrierType::Uav:
            std::snprintf(text, size, "UAV #%d (%s)", barrier.resource == ~0u ? -1 : int(barrier.resource), ReasonName(barrier.reason));
            break;
        case nvrhi::BarrierType::Aliasing:
            std::snprintf(text, size, "aliasing #%d after #%d (%s)", int(barrier.resource),
                          barrier.aliasBefore == ~0u ? -1 : int(barrier.aliasBefore), ReasonName(barrier.reason));
            break;
        default:
            std::snprintf(text, size, "#%u mips %u+%u slices %u+%u: 0x%x -> 0x%x%s%s%s (%s)", barrier.resource,
                          barrier.region.baseMip, barrier.region.mipCount, barrier.region.baseSlice, barrier.region.sliceCount,
                          uint32_t(barrier.before), uint32_t(barrier.after),
                          (barrier.flags & nvrhi::BarrierFlags::Discard) != 0 ? " discard" : "",
                          (barrier.flags & nvrhi::BarrierFlags::SplitBegin) != 0 ? " split-begin" :
                              (barrier.flags & nvrhi::BarrierFlags::SplitEnd) != 0 ? " split-end" : "",
                          (barrier.flags & nvrhi::BarrierFlags::QueueRelease) != 0 ? " release" :
                              (barrier.flags & nvrhi::BarrierFlags::QueueAcquire) != 0 ? " acquire" : "",
                          ReasonName(barrier.reason));
            break;
        }
    }
}

RenderGraphVisualizer::RenderGraphVisualizer(DeviceManager* deviceManager, RenderGraphLayer* graphLayer)
    : ImGuiLayer(deviceManager)
    , m_GraphLayer(graphLayer)
{
}

void RenderGraphVisualizer::BuildUI()
{
    DrawWindow();
}

void RenderGraphVisualizer::DrawWindow()
{
    if (!open || !m_GraphLayer)
        return;
    if (!ImGui::Begin("Render Graph", &open))
    {
        ImGui::End();
        return;
    }

    ICompiledRenderGraph* plan = m_GraphLayer->GetLastPlan();
    IResourceRegistry* registry = m_GraphLayer->GetRegistry();
    IRenderGraph* graph = m_GraphLayer->GetGraph();
    if (!plan || !registry || !graph)
    {
        ImGui::TextUnformatted("No plan yet.");
        ImGui::End();
        return;
    }

    // Statistics.
    PlanStatistics stats;
    plan->GetStatistics(stats);
    RunStatistics run;
    registry->GetRunStatistics(run);
    RegistryStatistics registryStats;
    registry->GetRegistryStatistics(registryStats);
    ImGui::Text("%u passes (%u culled), %u segments, %u fence waits", stats.scheduledPasses, stats.culledPasses, stats.segments,
                stats.fenceWaits);
    ImGui::Text("%u barriers: %u transitions, %u split, %u UAV, %u aliasing, %u queue transfers", stats.barriers,
                stats.transitions, stats.splitBarriers, stats.uavBarriers, stats.aliasingBarriers, stats.queueTransfers);
    ImGui::Text("compile %.3f ms%s; record %.3f ms, submit %.3f ms%s", double(stats.compileMicroseconds) / 1000.0,
                stats.cacheHit ? " (cached)" : "", double(run.recordMilliseconds), double(run.submitMilliseconds),
                run.planSwitched ? ", plan switch" : "");
    ImGui::Text("transients %.2f MB, placed peak %.2f MB; %u plans cached, %llu objects created",
                double(stats.poolableTransientBytes) / 1048576.0, double(stats.placedPeakBytes) / 1048576.0,
                registryStats.cachedPlans, (unsigned long long)registryStats.objectsCreated);

    // Switches.
    ImGui::Checkbox("GPU timers", &m_GraphLayer->GetExecuteDesc().timers);
    if (ImGui::TreeNode("Debug switches (11.4)"))
    {
        CompileOptions& options = m_GraphLayer->GetCompileOptions();
        const struct { DebugFlags flag; const char* name; } switches[] = {
            { DebugFlags::ForceDedicated, "ForceDedicated" }, { DebugFlags::NoAliasing, "NoAliasing" },
            { DebugFlags::BarrierAfterEveryPass, "BarrierAfterEveryPass" }, { DebugFlags::FillTransients, "FillTransients" },
            { DebugFlags::SerializeQueues, "SerializeQueues" }, { DebugFlags::StrictValidation, "StrictValidation" } };
        for (const auto& entry : switches)
        {
            bool on = (options.debugFlags & entry.flag) != DebugFlags::None;
            if (ImGui::Checkbox(entry.name, &on))
                options.debugFlags = on ? (options.debugFlags | entry.flag) : (options.debugFlags & ~entry.flag);
        }
        ImGui::TreePop();
    }

    // Passes: toggles, queue and segment, GPU time, barriers with reasons.
    if (ImGui::CollapsingHeader("Passes", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const uint32_t timingCount = registry->GetPassTimingCount();
        uint32_t timingCursor = 0;
        char text[256];
        for (uint32_t index = 0; index < plan->GetScheduledPassCount(); ++index)
        {
            ScheduledPass pass;
            if (plan->GetScheduledPass(index, pass) != nvrhi::FS_OK)
                continue;
            const char* name = pass.name ? pass.name : "?";
            ImGui::PushID(int(index));
            bool enabled = m_GraphLayer->IsPassEnabled(name);
            if (ImGui::Checkbox("##enabled", &enabled))
                m_GraphLayer->SetPassEnabled(name, enabled);
            ImGui::SameLine();
            float milliseconds = -1.f;
            if (pass.enabled && pass.queue != nvrhi::CommandQueue::Copy && timingCursor < timingCount)
            {
                PassTiming timing;
                if (registry->GetPassTiming(timingCursor, timing) == nvrhi::FS_OK && timing.name && std::strcmp(timing.name, name) == 0)
                {
                    milliseconds = timing.gpuMilliseconds;
                    ++timingCursor;
                }
            }
            if (milliseconds >= 0.f)
                std::snprintf(text, sizeof(text), "%s  [%s, segment %u]  %.3f ms###pass", name, QueueName(pass.queue), pass.segment,
                              double(milliseconds));
            else
                std::snprintf(text, sizeof(text), "%s  [%s, segment %u]%s###pass", name, QueueName(pass.queue), pass.segment,
                              pass.enabled ? "" : "  (off)");
            if (ImGui::TreeNode(text))
            {
                static const char* const c_Slots[] = { "before", "after the clears", "after" };
                for (uint32_t slot = 0; slot < 3; ++slot)
                {
                    BarrierBatch batch;
                    if (plan->GetBatch(index, BatchSlot(slot), batch) != nvrhi::FS_OK || batch.count == 0)
                        continue;
                    ImGui::Text("%s:", c_Slots[slot]);
                    for (uint32_t b = 0; b < batch.count; ++b)
                    {
                        DescribeBarrier(batch.barriers[b], text, sizeof(text));
                        ImGui::BulletText("%s", text);
                    }
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    // Lifetimes over the schedule, with heap placement.
    if (ImGui::CollapsingHeader("Resource lifetimes"))
    {
        const uint32_t passCount = std::max(plan->GetScheduledPassCount(), 1u);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        const float width = std::max(ImGui::GetContentRegionAvail().x - 220.f, 100.f);
        for (uint32_t resource = 0; resource < graph->GetResourceCount(); ++resource)
        {
            ResourceLifetime lifetime;
            if (plan->GetResourceLifetime(resource, lifetime) != nvrhi::FS_OK || lifetime.firstPass == ~0u)
                continue;
            if (lifetime.heap != ~0u)
                ImGui::Text("#%-4u heap %u +%llu, %llu KB%s", resource, lifetime.heap, (unsigned long long)lifetime.heapOffset,
                            (unsigned long long)(lifetime.size / 1024), lifetime.aliased ? ", aliased" : "");
            else
                ImGui::Text("#%-4u %llu KB%s", resource, (unsigned long long)(lifetime.size / 1024), lifetime.crossQueue ? ", cross-queue" : "");
            ImGui::SameLine(220.f);
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const float height = ImGui::GetTextLineHeight();
            const float x0 = origin.x + width * float(lifetime.firstPass) / float(passCount);
            const float x1 = origin.x + width * float(lifetime.lastPass + 1) / float(passCount);
            drawList->AddRectFilled(ImVec2(x0, origin.y), ImVec2(x1, origin.y + height),
                                    lifetime.aliased ? IM_COL32(220, 140, 60, 255) : IM_COL32(90, 150, 220, 255));
            ImGui::Dummy(ImVec2(width, height));
        }
    }

    ImGui::End();
}
