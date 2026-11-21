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
// column views (8.2). Declarations only in step E1; implemented in E6.

#pragma once

#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>

// The flag enum keeps nvrhi's `None`; X11 defines it as a macro.
#if defined(None)
#define DONUT_ECS_SIMULATION_RESTORE_NONE
#pragma push_macro("None")
#undef None
#endif

namespace nvrhi
{
    struct ICommandList;   // GPU domains record into one (8.1); the ECS headers do not include nvrhi.h
}

namespace donut::engine
{
    // Where a domain runs (8.1). GPU domains get a command list for each step.
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
        const ComponentAccess* outputs;     // written by ScatterOutputs
        // The domain's entities. OnEntitiesAdded and OnEntitiesRemoved follow their OnMatch and OnUnmatch (EcsWorld.h);
        // the query's data terms are the batch columns, and its chunk order is the order of every column view.
        QueryDesc lifetime;
        uint32_t inputCount;
        uint32_t outputCount;
        uint32_t substeps;                  // per fixed step, inside the domain
        uint32_t maxStepsPerFrame;          // the accumulator drops time beyond this (8.4)
        double fixedTimeStep;               // seconds
        SimulationQueue queue;
        uint8_t reserved[3];
        SimulationDomainFlags flags;
    };
    static_assert(sizeof(void*) != 8 || sizeof(SimulationDomainDesc) == 80, "SimulationDomainDesc layout");

    NVRHI_IID(ISimulationContext, "91c91391-2718-4a16-9f98-258e8f2ab916")
    struct ISimulationContext : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ISimulationContext)
        virtual IWorld* GetWorld() noexcept = 0;                          // not AddRef'd
        virtual SimulationDomainId GetDomainId() noexcept = 0;
        // One input or output component's columns over the domain's entities: one view per chunk of the lifetime query,
        // in its chunk order, paged like IQuery::GetChunks, so that views of different components line up by
        // chunkIndex (ColumnView). `sinceVersion` skips chunks whose column did not change. Output views stamp their
        // chunks with the scatter step's version.
        virtual uint32_t GetColumnCount(ComponentId component) noexcept = 0;
        virtual uint32_t GetColumns(ComponentId component, uint32_t first, uint64_t sinceVersion, uint32_t capacity,
                                    _Out_writes_to_(capacity, return) ColumnView* views) noexcept = 0;
        virtual uint64_t GetLastGatherVersion() noexcept = 0;             // sinceVersion for incremental gathers
        virtual nvrhi::ICommandList* GetCommandList() noexcept = 0;       // null for CPU domains; not AddRef'd
        virtual ICommandBuffer* GetCommandBuffer() noexcept = 0;          // for ChangesStructure domains
        virtual uint64_t GetStepIndex() noexcept = 0;                     // fixed steps since the domain was added
    };

    NVRHI_IID(ISimulationDomain, "67b8f7cf-57b6-4232-b4be-ed18dbd8648b")
    struct ISimulationDomain : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ISimulationDomain)
        // Read once, by IScheduler::AddSimulationDomain.
        virtual SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept = 0;
        // When entities start or stop matching the lifetime query, batched per chunk run (8.1): create or destroy
        // engine objects and store their engine indices in the batch's write columns. Engine indices are assigned in
        // batch order, which is deterministic (rule 3.3.6). Observer re-entrancy rules apply (EcsWorld.h).
        virtual nvrhi::FRESULT OnEntitiesAdded(const EntityBatch& batch, ISimulationContext* context) noexcept = 0;
        virtual nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch& batch, ISimulationContext* context) noexcept = 0;
        virtual nvrhi::FRESULT GatherInputs(ISimulationContext* context) noexcept = 0;      // PreSimulation
        virtual nvrhi::FRESULT Step(double dt, ISimulationContext* context) noexcept = 0;   // FixedSimulation, per step
        virtual nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept = 0;    // PostSimulation
        virtual nvrhi::FRESULT CaptureState(IStateWriter* writer) noexcept = 0;             // with SaveSnapshot (8.5)
        virtual nvrhi::FRESULT RestoreState(IStateReader* reader) noexcept = 0;
    };
}

#if defined(DONUT_ECS_SIMULATION_RESTORE_NONE)
#undef DONUT_ECS_SIMULATION_RESTORE_NONE
#pragma pop_macro("None")
#endif
