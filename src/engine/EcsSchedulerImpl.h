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

// The scheduler (aggregate docs/ecs-design.md 6.2-6.5, 8.4, 11.3; plan step E2.2). Module-private
// (com-interface-abi.md section 6): what the queries' access checks and the tests read of it. The scheduler itself is
// in EcsScheduler.cpp.
//
// Order. Per phase, the systems in registration order; runBefore/runAfter give explicit edges; the schedule order is
// their topological order with ties broken by registration order. Two systems conflict when one writes a component or
// resource that the other reads or writes, or when either is Exclusive; a conflicting pair is ordered by the schedule
// order, and it is an ambiguity when no chain of explicit edges orders it. Disabled systems keep their place, so the
// order and the reports do not depend on which systems are enabled.
//
// Execution. The enabled systems of a phase form a job graph: each depends on its nearest predecessors in the
// transitive reduction of (explicit edges plus conflict edges). Main-thread and exclusive systems run on the thread
// that called RunPhase; a job system is scheduled once every main-thread system it depends on has run, so no job ever
// waits for one. One job without indices (a sync job) waits for the phase's last jobs. Versions are handed out in
// schedule order before the phase; one more after it, which the sync point's playback stamps.

#pragma once

#include "EcsWorldImpl.h"

#include <donut/engine/EcsScheduler.h>

namespace donut::engine::ecs
{
    // What the Debug access checks (design 11.3) know of the running system: QueryImpl::ValidateRequest reads it
    // through ChunkRequest::system.
    struct SystemAccessInfo
    {
        const char* name;
        const ComponentAccess* accesses;   // sorted by component, one entry per component
        uint32_t accessCount;
        bool exclusive;                    // runs alone, so any access is allowed
        uint8_t reserved[3];
        uint64_t version;                  // this run's version: it names the run
        IWorld* world;
    };

    // False for a null context, or one that this module's scheduler did not make.
    bool GetSystemAccessInfo(ISystemContext* context, SystemAccessInfo& out) noexcept;
    // The access the system declared to `component`: Filter when it declared none.
    Access FindDeclaredAccess(const SystemAccessInfo& info, ComponentId component) noexcept;

    // Counters for tests and the inspector.
    struct SchedulerCounters
    {
        uint64_t orderBuilds;          // the per-phase orders and reports, after a system or phase was added or removed
        uint64_t executionBuilds;      // the job graphs, after the order or an enabled state changed
        uint64_t phaseRuns;
        uint64_t systemRuns;
        uint64_t jobsScheduled;        // system jobs
        uint64_t syncJobs;             // jobs without indices that end a phase
        uint64_t inlineRuns;           // systems run on the calling thread because Schedule failed
        uint64_t syncPoints;
        uint64_t playbacks;            // IWorld::Playback calls, one per system with commands
        uint64_t edges;                // dependency edges of the current job graphs, all phases
        uint64_t fixedSteps;           // ISimulationDomain::Step calls
    };

    // False for a scheduler of another module.
    bool GetSchedulerCounters(IScheduler* scheduler, SchedulerCounters& out) noexcept;
    // A phase's systems in schedule order (Build first): writes up to `capacity` ids, returns how many there are.
    uint32_t GetScheduleOrder(IScheduler* scheduler, Phase phase, SystemId* out, uint32_t capacity) noexcept;
    // A system's command buffers (not AddRef'd): one per worker index of the job system, then the home buffer of the
    // thread that runs its Update. Writes up to `capacity`, returns how many there are. A buffer takes its first arena
    // page when it first records, so which of them a system has warmed depends on which workers ran it; tests warm
    // them all through this before they count allocations.
    uint32_t GetSystemCommandBuffers(IScheduler* scheduler, SystemId system, ICommandBuffer** out,
                                     uint32_t capacity) noexcept;
}
