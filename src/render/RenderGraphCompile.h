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

// The compile state that the compiler's phases share (design 7): RenderGraphCompiler.cpp runs phases 1-5 and 7,
// RenderGraphTransitions.cpp runs phase 6.

#pragma once

#include "RenderGraphInternal.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace donut::render::rdg
{
    // One unit of a resource's subresource space (6.2): a run of array slices of one mip of one plane, or the whole
    // resource (plane c_WholeUnit).
    constexpr uint8_t c_WholeUnit = 0xff;

    struct UnitKey
    {
        uint8_t plane;
        uint32_t mip;
        uint32_t slice0;
        uint32_t slice1;
    };

    struct EmittedBarrier
    {
        uint32_t sched;
        uint8_t slot;              // BatchSlot
        nvrhi::CommandQueue queue;
        PlanBarrier barrier;       // region filled in when the units are merged
        UnitKey unit;
        uint32_t pairId;           // split pairs: both halves carry the pair's index; c_None otherwise
    };

    struct WaitEdge
    {
        uint32_t waiter;           // schedule index
        uint32_t producer;         // schedule index, in this frame or the previous one
        int8_t frameDelta;
    };

    struct SplitPair
    {
        uint32_t begin;            // schedule index of the begin half's epilogue
        uint32_t end;              // schedule index of the end half's prologue
        bool joined;               // set by the segment phase when a command-list boundary falls between them
    };

    struct CompileState
    {
        CompileState(const RenderGraph& graph_, const BackendProfile& profile_, const CompileOptions& options_)
            : graph(graph_), profile(profile_), options(options_) { }

        const RenderGraph& graph;
        const BackendProfile& profile;
        const CompileOptions& options;
        const StructurePlan* structure = nullptr;
        bool superset = false;     // compiling for the structure: every pass enabled, no policies
        bool failed = false;
        bool quiet = false;        // cost estimates: no diagnostics
        std::string* diagnostics = nullptr;

        // Per program pass.
        std::vector<uint8_t> enabled;       // after SetPassEnabled and the Cull policies
        std::vector<uint8_t> live;          // enabled and not culled
        std::vector<uint32_t> scheduleOf;   // schedule index, or c_None

        // Per access: the resource it names after Substitute and Passthrough.
        std::vector<uint32_t> accessResource;

        // The schedule: live passes and disabled ones (listed, with no work), in program order.
        std::vector<uint32_t> schedule;
        std::vector<uint8_t> scheduledLive;
        std::vector<uint32_t> useBegin;     // per schedule index
        std::vector<uint32_t> useCount;
        std::vector<PlanUse> uses;
        std::vector<nvrhi::ResourceStates> useState;
        std::vector<nvrhi::ShaderType> useStages;   // shader stages for sync scopes; None without a shader access
        std::vector<nvrhi::CommandQueue> schedQueue;
        std::vector<uint8_t> schedLane;     // per schedule index: the lane of its queue (StructurePlan::passLanes)

        // Per resource (mask level).
        std::vector<uint32_t> first;
        std::vector<uint32_t> last;
        std::vector<uint8_t> queueMask;
        std::vector<uint32_t> streamMask;   // per resource: the (queue, lane) streams that use it (StreamOf bits)
        std::vector<nvrhi::ResourceStates> entryState;  // the state the plan assumes at frame start

        // Phase 6 output.
        std::vector<EmittedBarrier> emitted;
        std::vector<WaitEdge> waits;
        std::vector<SplitPair> splits;
        std::vector<std::array<LoadOp, 9>> load;     // per schedule index
        std::vector<std::array<StoreOp, 9>> store;
        std::vector<uint32_t> clearMask;
        struct EntryUnit
        {
            uint32_t resource;              // the physical resource's first occupant, or the history layer
            UnitKey unit;
            nvrhi::ResourceStates state;
        };
        std::vector<EntryUnit> entryUnits;  // frame-start state of every unit the plan touches
        struct TrackedUnit
        {
            uint32_t sched;
            uint32_t resource;
            UnitKey unit;
            nvrhi::ResourceStates state;
        };
        std::vector<TrackedUnit> trackedUnits;   // per Tracked or Restore pass: the states its body finds (9.1)
        uint32_t transitionCount = 0;
        uint32_t transferCount = 0;

        void Error(const std::string& message)
        {
            failed = true;
            if (diagnostics && !quiet)
            {
                *diagnostics += "error: ";
                *diagnostics += message;
                *diagnostics += '\n';
            }
        }

        void Warning(const std::string& message)
        {
            if (diagnostics && !quiet)
            {
                *diagnostics += "warning: ";
                *diagnostics += message;
                *diagnostics += '\n';
            }
        }

        std::string PassName(uint32_t program) const
        {
            const PassRecord& pass = graph.GetPass(program);
            return "pass '" + std::string(pass.name ? pass.name : "") + "' (" + std::to_string(program) + ")";
        }

        std::string ResourceName(uint32_t resource) const
        {
            const ResourceRecord& record = graph.GetResource(resource);
            std::string text = record.name ? "'" + std::string(record.name) + "'" : std::string("resource");
            text += " (#" + std::to_string(resource);
            if (record.kind == ResourceKind::History)
                text += ", history layer " + std::to_string(record.historyLayer);
            return text + ")";
        }

        bool IsImplicit() const { return profile.barrierModel == BarrierModel::Implicit; }
        uint32_t StreamOfSched(uint32_t sched) const { return StreamOf(schedQueue[sched], schedLane[sched]); }
    };

    // The planes of a resource that units distinguish: 2 for a depth-stencil format where the profile tracks planes,
    // else 1.
    inline uint32_t UnitPlaneCount(const ResourceRecord& record, const BackendProfile& profile) noexcept
    {
        return (!record.isBuffer && record.hasDepth && record.hasStencil && profile.planeModel != PlaneModel::Ignored &&
            profile.barrierModel != BarrierModel::Implicit) ? 2u : 1u;
    }

    // The plane mask of the planes a resource has, in the profile's terms.
    inline nvrhi::PlaneMask ResourcePlanes(const ResourceRecord& record, const BackendProfile& profile) noexcept
    {
        if (record.isBuffer || UnitPlaneCount(record, profile) == 1)
            return nvrhi::PlaneMask::All;
        return nvrhi::PlaneMask::Depth | nvrhi::PlaneMask::Stencil;
    }

    inline uint32_t MipCount(const ResourceRecord& record) noexcept
    {
        return record.isBuffer ? 1u : std::max(1u, record.texture.mipLevels);
    }

    inline uint32_t SliceCount(const ResourceRecord& record) noexcept
    {
        if (record.isBuffer || record.texture.dimension == nvrhi::TextureDimension::Texture3D)
            return 1u;
        return std::max(1u, record.texture.arraySize);
    }

    // Phase 6 (RenderGraphTransitions.cpp).
    void RunTransitions(CompileState& state);
}
