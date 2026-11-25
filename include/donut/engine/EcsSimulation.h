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

// Simulation domains (aggregate docs/ecs-design.md section 8): an engine instance behind ISimulationDomain, which may
// live in another module (DLL), so these interfaces are on the ABI surface. A domain is added with
// IScheduler::AddSimulationDomain. It never sees an entity in its hot path: it exchanges batched, index-addressed
// column views (8.2).
//
// Threads. A domain belongs to one scheduler and is called by one thread at a time; different domains may be called at
// once. GatherInputs and ScatterOutputs of a Cpu domain run on any thread of the job system, with the accesses of the
// descriptor; Step runs on any thread too, at the same time as the steps of other domains, unless the domain is
// ChangesStructure: then it runs alone, on the thread that runs the phase. A Graphics, Compute or Cuda domain is
// called only on the thread that runs the phase, one domain after the other in schedule order, because an engine
// records into one command list on one thread. An engine that wants more threads splits its step on
// ISimulationContext::GetJobSystem. The lifetime callbacks run where the change is made: in the immediate structural
// change, or at the sync point.
//
// Memory. A domain allocates on entity events (index arrays, staging), never in a frame without them.

#pragma once

#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>

// The flag enum keeps nvrhi's `None`; X11 defines it as a macro.
#if defined(None)
#define DONUT_ECS_SIMULATION_RESTORE_NONE
#pragma push_macro("None")
#undef None
#endif

namespace donut::engine
{
    // Where a domain runs (8.1): see the threads above. A GPU domain records into the list that
    // IScheduler::SetSimulationCommandList handed over, or into its engine's own.
    enum class SimulationQueue : uint8_t { Cpu, Graphics, Compute, Cuda };

    enum class SimulationDomainFlags : uint32_t
    {
        None             = 0,
        Deterministic    = 1u << 0,   // same inputs give the same outputs on any thread count (8.5)
        ChangesStructure = 1u << 1,   // records structural commands during steps: a sync point follows each step
    };
    template <> struct EnableEcsFlagOperators<SimulationDomainFlags> : std::true_type { };

    // Every pointer must stay valid while the domain is added.
    struct SimulationDomainDesc
    {
        const char* name;
        const ComponentAccess* inputs;      // read by GatherInputs (Write for inputs the domain consumes, e.g. teleports)
        // Used by ScatterOutputs: Write for what it writes, Read for what it only reads (SimBodyRef, to map a row to the
        // engine's index).
        const ComponentAccess* outputs;
        // The domain's entities. OnEntitiesAdded and OnEntitiesRemoved follow their OnMatch and OnUnmatch (EcsWorld.h);
        // the query's data terms are the batch columns, and its chunk order is the order of every column view.
        QueryDesc lifetime;
        uint32_t inputCount;
        uint32_t outputCount;
        uint32_t substeps;                  // per fixed step, inside the domain
        uint32_t maxStepsPerFrame;          // the accumulator drops time beyond this (8.4)
        double fixedTimeStep;               // seconds, positive and finite
        SimulationQueue queue;
        uint8_t reserved[3];
        SimulationDomainFlags flags;
    };
    static_assert(sizeof(void*) != 8 || sizeof(SimulationDomainDesc) == 80, "SimulationDomainDesc layout");

    // ------------------------------------------------------------------------------------------------ time (8.4)

    // The domains of one scheduler; AddSimulationDomain gives FE_OUT_OF_MEMORY beyond.
    constexpr uint32_t c_MaxSimulationDomains = 16;

    // The fixed-step time of one domain, for systems that read it (interpolation: alpha = accumulator / fixedTimeStep).
    struct FixedStepEntry
    {
        double accumulator;      // seconds not simulated yet
        double fixedTimeStep;    // 0: no domain has this id
        uint64_t stepIndex;      // steps completed since the domain was added
        uint32_t stepsPlanned;   // the next FixedSimulation phase takes this many
        uint32_t stepsTaken;     // since the last SetFrameTime
        uint32_t stepsDropped;   // since the last SetFrameTime: beyond maxStepsPerFrame, or not taken after a failure
        uint32_t reserved;
    };
    static_assert(sizeof(FixedStepEntry) == 40, "FixedStepEntry layout");

    // A world resource (4.9), by SimulationDomainId. A copy for readers: the scheduler's working state is its own, and
    // the thread that drives the scheduler publishes it here where no system runs (when a domain is added or removed,
    // in SetFrameTime, before and after each FixedSimulation phase, after LoadSimulationState). Systems declare Read
    // and never Write. NoSerialize: snapshots and world hashes leave it out; SaveSimulationState writes the working
    // state. One scheduler with domains per world: a scheduler made later on the same world starts it at zero.
    struct FixedStepState
    {
        FixedStepEntry domains[c_MaxSimulationDomains];
    };
    static_assert(sizeof(FixedStepState) == 640, "FixedStepState layout");
    DONUT_ECS_COMPONENT_EX(FixedStepState, "99e1e90e-f7ac-45df-88bb-39bfdee76b14",
                           ::donut::engine::ComponentFlags::Resource | ::donut::engine::ComponentFlags::NoSerialize, 0,
                           DONUT_ECS_FIELDS(domains));

    // ------------------------------------------------------------------------------------------------ interfaces

    // What a domain's calls get. Valid during the call, on the thread that makes it; not to be kept. Nothing it returns
    // is AddRef'd.
    NVRHI_IID(ISimulationContext, "91c91391-2718-4a16-9f98-258e8f2ab916")
    struct ISimulationContext : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ISimulationContext)
        virtual IWorld* GetWorld() noexcept = 0;
        virtual SimulationDomainId GetDomainId() noexcept = 0;
        // One input or output component's columns over the domain's entities: one view per chunk of the lifetime query,
        // in its chunk order, paged like IQuery::GetChunks, so that views of different components line up by chunkIndex
        // (ColumnView). `sinceVersion` skips chunks whose column did not change. GatherInputs gets its declared inputs,
        // ScatterOutputs its declared outputs: read views, or write views of the entries declared Write, which stamp
        // their chunks with the running system's version when they are handed out. Elsewhere there are no views: in
        // Step and in the lifetime callbacks both calls return 0, so steps declare nothing and independent domains step
        // in parallel. Both also return 0, with one validation report, for a component that is not declared, not
        // Relocatable (D14; a construct hook is allowed, the other hooks exclude Relocatable), or without one value per
        // row (tags, buffers, shared, chunk and sparse components). A domain without lifetime terms has no entities: it
        // gets 0, without a report. Views carry no enabled bits: a domain that needs them uses a query of its own
        // (GetSystemContext).
        virtual uint32_t GetColumnCount(ComponentId component) noexcept = 0;
        virtual uint32_t GetColumns(ComponentId component, uint32_t first, uint64_t sinceVersion, uint32_t capacity,
                                    _Out_writes_to_(capacity, return) ColumnView* views) noexcept = 0;
        virtual uint64_t GetLastGatherVersion() noexcept = 0;             // sinceVersion for incremental gathers
        // The open list this frame's work goes into (IScheduler::SetSimulationCommandList); the domain does not close
        // it. Null when none was handed over (the engine uses its own), and always for Cpu domains.
        virtual nvrhi::ICommandList* GetCommandList() noexcept = 0;
        // In a phase call its system's buffer (a ChangesStructure domain's is played back after each step). In the
        // lifetime callbacks the observer's, or, for the deliveries of AddSimulationDomain and RemoveSimulationDomain,
        // one of the scheduler's, which is played back before that call returns. Null elsewhere.
        virtual ICommandBuffer* GetCommandBuffer() noexcept = 0;
        virtual uint64_t GetStepIndex() noexcept = 0;                     // fixed steps since the domain was added
        // In GatherInputs the steps that FixedSimulation will take; in Step all steps planned for this frame, the ones
        // to come included; in ScatterOutputs the steps taken (fewer after a failed one). Not meaningful in the
        // lifetime callbacks.
        virtual uint32_t GetStepsThisFrame() noexcept = 0;
        // Inside Step: 0 for the frame's first step, GetStepsThisFrame() - 1 for its last planned one. 0 elsewhere.
        virtual uint32_t GetStepInFrame() noexcept = 0;
        // The scheduler's job system, for CPU engines that split a step.
        virtual IJobSystem* GetJobSystem() noexcept = 0;
        // The running system's context in GatherInputs, Step and ScatterOutputs; null in the lifetime callbacks and
        // outside phases. A domain that needs a query of its own in a phase passes it to the query (MakeChunkRequest),
        // so that the access check and the run's versions hold; everything the query touches must be among the accesses
        // the descriptor declares for that call (inputs in GatherInputs, outputs in ScatterOutputs, nothing in Step
        // unless the domain is ChangesStructure).
        virtual ISystemContext* GetSystemContext() noexcept = 0;
    };

    NVRHI_IID(ISimulationDomain, "67b8f7cf-57b6-4232-b4be-ed18dbd8648b")
    struct ISimulationDomain : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ISimulationDomain)
        // Read once, by IScheduler::AddSimulationDomain.
        virtual SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept = 0;
        // When entities start or stop matching the lifetime query, batched per chunk run (8.1): create or destroy
        // engine objects and store their engine indices in the batch's write columns. Engine indices are assigned in
        // batch order, which is deterministic (rule 3.3.6). Observer re-entrancy rules apply (EcsWorld.h). Entities
        // that match when the domain is added are delivered then, and the ones that still match when it is removed
        // then (AddSimulationDomain, RemoveSimulationDomain); the scheduler's destruction delivers nothing.
        virtual nvrhi::FRESULT OnEntitiesAdded(const EntityBatch& batch, ISimulationContext* context) noexcept = 0;
        virtual nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch& batch, ISimulationContext* context) noexcept = 0;
        virtual nvrhi::FRESULT GatherInputs(ISimulationContext* context) noexcept = 0;      // PreSimulation
        // FixedSimulation, once per planned step. A failure is recorded as the step system's
        // (IScheduler::GetDiagnostics; counted once per frame, logged the first time) and ends this domain's steps for
        // the frame: the failed step is not counted in stepIndex and stepsTaken, and the planned steps that were not
        // taken, the failed one included, are discarded and counted in stepsDropped. Other domains and the frame go on,
        // ScatterOutputs included. The next frame tries again; a domain that cannot recover latches its own error and
        // fails again.
        virtual nvrhi::FRESULT Step(double dt, ISimulationContext* context) noexcept = 0;
        virtual nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept = 0;    // PostSimulation
        // The engine's state, for IScheduler::SaveSimulationState and LoadSimulationState (8.5), outside phases.
        // FE_UNSUPPORTED: the engine cannot (its state stays on the GPU). RestoreState reads what CaptureState wrote;
        // FE_NOT_FOUND when the stream names state that the domain does not hold (of another process or domain, or
        // dropped since).
        virtual nvrhi::FRESULT CaptureState(IStateWriter* writer) noexcept = 0;
        virtual nvrhi::FRESULT RestoreState(IStateReader* reader) noexcept = 0;
    };
}

#if defined(DONUT_ECS_SIMULATION_RESTORE_NONE)
#undef DONUT_ECS_SIMULATION_RESTORE_NONE
#pragma pop_macro("None")
#endif
