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

// Systems, phases and the scheduler (aggregate docs/ecs-design.md 6.2-6.5, 8.1, 8.4, 11.3; plan step E2.2). See
// EcsSchedulerImpl.h for the ordering and execution rules.
//
// A phase run:
// 1. Versions: each enabled system, in schedule order, takes IWorld::AdvanceVersion; its previous one becomes its
//    last-run version (the sinceVersion of its Changed and Added terms).
// 2. BeginPhase, then the phase's script: job systems are scheduled with their predecessors' handles as dependencies;
//    main-thread and exclusive systems run on the calling thread after waiting for their predecessors (an exclusive
//    one inside BeginExclusive/EndExclusive). A system whose Schedule fails (FE_OUT_OF_MEMORY) runs on the calling
//    thread after its predecessors. A sync job over the jobs without job successors ends the phase; EndPhase.
// 3. One more version, for writes outside phases. The systems that recorded commands join the pending list.
// 4. At a sync point (after PostSimulation and RenderExtract, design 6.3) each pending system's buffers are played back
//    with one IWorld::Playback, systems in (phase run, schedule order) order. A phase whose systems still have pending
//    commands from an earlier run gets a sync point before it runs again, so that every playback holds one run's
//    commands and sort keys stay unique per system. Commands recorded after a frame's last sync point (in
//    TransformPropagation, Bounds and EndFrame) stay pending until the next frame's PostSimulation: a snapshot taken
//    between frames does not hold them (a concern for the rewinds of design 8.5 and 11.4; E6/E7 may add a flush).
//
// Simulation domains (8.1, 8.4): a domain adds three systems, `<name>.gather` (PreSimulation, its inputs),
// `<name>.step` (FixedSimulation, no column access; Exclusive when the domain changes structure) and `<name>.scatter`
// (PostSimulation, its outputs). SetFrameTime adds the frame's delta to each domain's accumulator; the step system runs
// floor(accumulated / fixedTimeStep) steps, at most maxStepsPerFrame, keeps the remainder fraction and drops whole
// steps beyond the limit. After each step of a ChangesStructure domain a sync point plays back, in order, the pending
// systems, the phase's systems that ran before the step, and the step's own commands. Column views, the lifetime
// query's observers and GPU queues are E6 (the seams are marked below).

#include "EcsSchedulerImpl.h"

#include <donut/core/log.h>
#include <donut/engine/EcsSimulation.h>
#include <nvrhi/core/autoptr.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>

namespace donut::engine::ecs
{
    namespace
    {
        constexpr uint32_t c_None = 0xffffffffu;
        constexpr uint32_t c_MainBit = 0x80000000u;    // a script entry that runs on the calling thread
        constexpr double c_StatsSmoothing = 0.125;     // weight of the newest run in SystemStats::averageMilliseconds
        constexpr Phase c_NoPhase = Phase(0xffffffffu);
        constexpr SystemFlags c_KnownSystemFlags =
            SystemFlags::Exclusive | SystemFlags::MainThread | SystemFlags::StartDisabled;
        constexpr SimulationDomainFlags c_KnownDomainFlags =
            SimulationDomainFlags::Deterministic | SimulationDomainFlags::ChangesStructure;

        const char* const c_BuiltinPhaseNames[uint32_t(Phase::Count)] = {
            "BeginFrame", "PreSimulation", "FixedSimulation", "PostSimulation",
            "TransformPropagation", "Bounds", "RenderExtract", "EndFrame",
        };

        bool Has(SystemFlags flags, SystemFlags flag) noexcept { return !!(flags & flag); }
        bool Has(SimulationDomainFlags flags, SimulationDomainFlags flag) noexcept { return !!(flags & flag); }

        void AppendFormat(nvrhi::string& out, const char* format, ...) noexcept
        {
            char line[1024];
            va_list args;
            va_start(args, format);
            vsnprintf(line, sizeof(line), format, args);
            va_end(args);
            out.append(line);
        }

        // Sorts by component and keeps one entry per component, with the strongest access (Write > Read > Filter).
        void NormalizeAccesses(Vector<ComponentAccess>& accesses) noexcept
        {
            std::sort(accesses.begin(), accesses.end(), [](const ComponentAccess& a, const ComponentAccess& b)
            {
                return uint32_t(a.component) < uint32_t(b.component);
            });
            size_t kept = 0;
            for (size_t i = 0; i < accesses.size(); ++i)
            {
                if (kept != 0 && accesses[kept - 1].component == accesses[i].component)
                {
                    if (uint8_t(accesses[i].access) > uint8_t(accesses[kept - 1].access))
                        accesses[kept - 1].access = accesses[i].access;
                    continue;
                }
                accesses[kept] = accesses[i];
                accesses[kept].reserved[0] = accesses[kept].reserved[1] = accesses[kept].reserved[2] = 0;
                ++kept;
            }
            accesses.resize(kept);
        }

        // Rows of bits over a phase's systems: reachability in schedule order.
        class BitMatrix
        {
        public:
            void Reset(uint32_t rows, uint32_t columns) noexcept
            {
                m_Words = (columns + 63) / 64;
                m_Bits.clear();
                m_Bits.resize(size_t(rows) * m_Words, 0);
            }
            uint32_t Words() const noexcept { return m_Words; }
            uint64_t* Row(uint32_t row) noexcept { return m_Bits.data() + size_t(row) * m_Words; }
            const uint64_t* Row(uint32_t row) const noexcept { return m_Bits.data() + size_t(row) * m_Words; }
            bool Test(uint32_t row, uint32_t column) const noexcept
            {
                return ((Row(row)[column >> 6] >> (column & 63)) & 1) != 0;
            }
            void Set(uint32_t row, uint32_t column) noexcept { Row(row)[column >> 6] |= 1ull << (column & 63); }
            // row |= {column} | other row `column`
            void Absorb(uint32_t row, uint32_t column) noexcept
            {
                Set(row, column);
                uint64_t* dst = Row(row);
                const uint64_t* src = Row(column);
                for (uint32_t w = 0; w < m_Words; ++w)
                    dst[w] |= src[w];
            }

        private:
            Vector<uint64_t> m_Bits;
            uint32_t m_Words = 0;
        };
    }

    class SchedulerImpl;
    class SystemContextImpl;
    struct DomainRecord;

    enum class RunKind : uint8_t
    {
        Job,          // a job on the job system
        MainThread,   // on the thread that called RunPhase
        Exclusive,    // on that thread, alone, inside BeginExclusive/EndExclusive
    };

    // ---------------------------------------------------------------------------------------------- records

    struct SystemRecord
    {
        SchedulerImpl* scheduler = nullptr;
        nvrhi::AutoPtr<ISystem> system;               // released by RemoveSystem
        nvrhi::AutoPtr<SystemContextImpl> context;
        nvrhi::string name;
        Vector<ComponentAccess> accesses;             // NormalizeAccesses
        Vector<nvrhi::string> runBefore;
        Vector<nvrhi::string> runAfter;
        // One buffer per worker index of the job system, then the home buffer: the thread that runs Update records into
        // home (also in the ranges of its own ParallelFor that it runs itself), other workers into their own. So a
        // system that records from one thread uses one buffer whichever worker runs it, and each buffer is used by one
        // thread at a time.
        Vector<ICommandBuffer*> buffers;
        DomainRecord* domain = nullptr;               // the domain whose gather, step or scatter system this is
        uint64_t registration = 0;                    // registration order: ties in the schedule order
        SystemId id = SystemId::Invalid;
        Phase phase = Phase::BeginFrame;
        SystemFlags flags = SystemFlags::None;
        bool enabled = true;
        bool removed = false;                         // removed, kept until its pending commands are played back
        bool pending = false;                         // in the scheduler's pending list
        bool reportedFailure = false;

        // Written by the calling thread before the phase's jobs are scheduled.
        uint64_t version = 0;
        uint64_t lastRunVersion = c_AllVersions;
        uint32_t runPosition = c_None;                // index in its phase's run list

        // Written by the thread that runs the system. updateWorker: its worker index (~0u without a slot), written
        // before Update, so before any range of its ParallelFor reads it.
        uint32_t updateWorker = c_None;
        SystemStats stats{};
        uint64_t updateFailures = 0;
        nvrhi::FRESULT lastUpdateFailure = nvrhi::FS_OK;
        uint64_t playbackFailures = 0;
        nvrhi::FRESULT lastPlaybackFailure = nvrhi::FS_OK;

        ~SystemRecord()
        {
            for (ICommandBuffer* buffer : buffers)
                if (buffer)
                    buffer->Release();
        }

        bool HasCommands() const noexcept
        {
            for (ICommandBuffer* buffer : buffers)
                if (buffer->GetCommandCount() != 0)
                    return true;
            return false;
        }
    };

    struct RunEntry
    {
        SystemRecord* record;
        uint32_t firstPredecessor;   // in PhaseRecord::predecessors
        uint32_t predecessorCount;
        RunKind kind;
        uint8_t reserved[3];
    };

    struct PhaseRecord
    {
        nvrhi::string name;
        Phase id = Phase::BeginFrame;
        bool syncAfter = false;
        bool valid = true;               // no ordering cycle

        // The order: every system of the phase in schedule order, and reach[p][q] when q runs after p (explicit and
        // conflict edges, transitively), by schedule position.
        Vector<SystemRecord*> order;
        BitMatrix reach;

        // The job graph of the enabled systems, in schedule order.
        Vector<RunEntry> run;
        Vector<uint32_t> predecessors;   // run indices: each entry's nearest predecessors
        Vector<uint32_t> script;         // run indices in the order the calling thread handles them; c_MainBit: run
                                         // on the calling thread
        Vector<uint32_t> sinks;          // run indices of the jobs that no job depends on
        Vector<JobHandle> handles;       // this run's jobs, by run index

        uint32_t pendingCount = 0;       // systems of this phase in the pending list
    };

    struct DomainRecord
    {
        SchedulerImpl* scheduler = nullptr;
        nvrhi::AutoPtr<ISimulationDomain> domain;
        nvrhi::AutoPtr<ISimulationContext> context;
        nvrhi::string name;
        Vector<ComponentAccess> inputs;
        Vector<ComponentAccess> outputs;
        double fixedTimeStep = 0.0;
        double accumulator = 0.0;        // seconds not yet stepped
        uint64_t stepIndex = 0;          // fixed steps since the domain was added
        uint32_t maxStepsPerFrame = 1;
        uint32_t substeps = 1;
        SimulationDomainFlags flags = SimulationDomainFlags::None;
        SimulationDomainId id = SimulationDomainId::Invalid;
        SystemId systems[3] = { SystemId::Invalid, SystemId::Invalid, SystemId::Invalid };   // gather, step, scatter
        SystemRecord* gather = nullptr;
        ISystemContext* current = nullptr;   // the running domain system's context, for GetCommandBuffer
    };

    // ---------------------------------------------------------------------------------------------- the context

    NVRHI_CLASS_CLSID(SystemContextImpl, "724e51e8-e99c-4492-9364-78e72c7bf925")
    class SystemContextImpl final : public nvrhi::ObjectImpl<ISystemContext>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SystemContextImpl)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SystemContextImpl)
        NVRHI_IMPLEMENTS_INTERFACE(ISystemContext)
        NVRHI_IMPLEMENTS_CLASS(SystemContextImpl)
        NVRHI_END_INTERFACE_TABLE()

        explicit SystemContextImpl(SystemRecord* record) noexcept : m_Record(record) { }

        IWorld* GetWorld() noexcept override;
        ICommandBuffer* GetCommandBuffer() noexcept override;
        IJobSystem* GetJobSystem() noexcept override;
        uint64_t GetVersion() noexcept override { return m_Record->version; }
        uint64_t GetLastRunVersion() noexcept override { return m_Record->lastRunVersion; }
        const FrameTime& GetFrameTime() noexcept override;
        SystemId GetSystemId() noexcept override { return m_Record->id; }

        SystemRecord* GetRecord() const noexcept { return m_Record; }

    private:
        SystemRecord* m_Record;
    };

    // ------------------------------------------------------------------------------------------ the scheduler

    NVRHI_CLASS_CLSID(SchedulerImpl, "610a3947-39da-4ec1-a1c4-df5d44aecdae")
    class SchedulerImpl final : public nvrhi::ObjectImpl<IScheduler>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SchedulerImpl)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SchedulerImpl)
        NVRHI_IMPLEMENTS_INTERFACE(IScheduler)
        NVRHI_IMPLEMENTS_CLASS(SchedulerImpl)
        NVRHI_END_INTERFACE_TABLE()

        SchedulerImpl(WorldImpl* world, IJobSystem* jobs) noexcept;
        ~SchedulerImpl();
        nvrhi::FRESULT Initialize() noexcept;

        // IScheduler
        nvrhi::FRESULT AddPhase(const char* name, Phase after, Phase& outPhase) noexcept override;
        nvrhi::FRESULT AddSystem(ISystem* system, SystemId& outId) noexcept override;
        nvrhi::FRESULT RemoveSystem(SystemId system) noexcept override;
        nvrhi::FRESULT SetSystemEnabled(SystemId system, bool enabled) noexcept override;
        nvrhi::FRESULT AddSimulationDomain(ISimulationDomain* domain, SimulationDomainId& outId) noexcept override;
        nvrhi::FRESULT RemoveSimulationDomain(SimulationDomainId domain) noexcept override;
        nvrhi::FRESULT Build() noexcept override;
        uint32_t GetAmbiguityCount() noexcept override;
        void GetDiagnostics(nvrhi::string& out) noexcept override;
        void SetFrameTime(const FrameTime& time) noexcept override;
        nvrhi::FRESULT RunPhase(Phase phase) noexcept override;
        nvrhi::FRESULT RunPhases(Phase first, Phase last) noexcept override;
        IWorld* GetWorld() noexcept override { return m_World; }
        IJobSystem* GetJobSystem() noexcept override { return m_Jobs.Get(); }
        nvrhi::FRESULT GetSystemStats(SystemId system, SystemStats& out) noexcept override;

        // ------------------------------------------------------------------------------- module-internal access

        WorldImpl& GetWorldImpl() const noexcept { return *m_World; }
        IJobSystem& GetJobs() const noexcept { return *m_Jobs; }
        // The buffer the calling thread records into for `record` (SystemRecord::buffers).
        uint32_t GetBufferSlot(const SystemRecord& record) noexcept
        {
            const uint32_t worker = m_Jobs->GetCurrentWorkerIndex();
            return worker == record.updateWorker || worker >= m_ThreadCount ? m_ThreadCount : worker;
        }
        uint32_t GetCurrentWorker() noexcept { return m_Jobs->GetCurrentWorkerIndex(); }
        const FrameTime& GetFrameTimeValue() const noexcept { return *m_FrameTime; }
        void RunSystem(SystemRecord& record) noexcept;
        // The sync point after each step of a ChangesStructure domain, from its exclusive step system.
        void SyncInsideStep(SystemRecord& step) noexcept;
        void CountFixedStep() noexcept { m_FixedSteps.fetch_add(1, std::memory_order_relaxed); }
        SchedulerCounters GetCounters() const noexcept
        {
            SchedulerCounters counters = m_Counters;
            counters.systemRuns = m_SystemRuns.load(std::memory_order_relaxed);
            counters.fixedSteps = m_FixedSteps.load(std::memory_order_relaxed);
            return counters;
        }
        uint32_t GetScheduleOrder(Phase phase, SystemId* out, uint32_t capacity) noexcept;
        bool PhaseDeclaresWrites(Phase phase) noexcept;
        uint32_t GetCommandBuffers(SystemId system, ICommandBuffer** out, uint32_t capacity) const noexcept
        {
            const SystemRecord* record = FindSystem(system);
            if (!record)
                return 0;
            for (uint32_t i = 0; i < record->buffers.size() && i < capacity && out; ++i)
                out[i] = record->buffers[i];
            return uint32_t(record->buffers.size());
        }

    private:
        PhaseRecord* FindPhase(Phase phase) const noexcept
        {
            return uint32_t(phase) < m_Phases.size() ? m_Phases[uint32_t(phase)] : nullptr;
        }
        SystemRecord* FindSystem(SystemId id) const noexcept
        {
            const uint32_t index = uint32_t(id);
            SystemRecord* record = index < m_Systems.size() ? m_Systems[index] : nullptr;
            return record && !record->removed ? record : nullptr;
        }
        uint32_t FindPhasePosition(Phase phase) const noexcept;
        bool IsRunning() const noexcept { return m_Running.load(std::memory_order_acquire); }
        bool RefuseWhileRunning(const char* operation) noexcept;

        nvrhi::FRESULT AddSystemInternal(ISystem* system, DomainRecord* domain, SystemId& outId) noexcept;
        void RemoveSystemInternal(SystemRecord& record) noexcept;
        void FreeRecord(SystemRecord* record) noexcept;

        // Build: the orders and reports (when systems or phases changed), then the job graphs.
        nvrhi::FRESULT EnsureBuilt() noexcept;
        void BuildOrder(PhaseRecord& phase) noexcept;
        void BuildExecution(PhaseRecord& phase) noexcept;
        bool Conflict(const SystemRecord& a, const SystemRecord& b, ComponentId& component,
                      const SystemRecord*& writer) const noexcept;
        const char* ComponentName(ComponentId component) const noexcept;

        // Running.
        void RunPhaseInternal(PhaseRecord& phase) noexcept;
        void ScheduleSystem(PhaseRecord& phase, uint32_t index) noexcept;
        void RunOnCallingThread(PhaseRecord& phase, uint32_t index) noexcept;
        void FinishJobs(PhaseRecord& phase) noexcept;
        void CollectPending(PhaseRecord& phase) noexcept;
        void SyncPoint() noexcept;
        void PlayPending() noexcept;
        void PlaySystem(SystemRecord& record) noexcept;

        static void RunSystemJob(void* context, uint32_t begin, uint32_t end, uint32_t workerIndex) noexcept;
        // WorldImpl::WriteAccessCheck: GetComponentForWrite and ResizeBuffer inside a phase, with validation.
        static void CheckWriteAccess(void* context, ComponentId component, uint64_t writeVersion,
                                     const char* operation) noexcept;

        nvrhi::AutoPtr<IWorld> m_WorldReference;   // released last
        nvrhi::AutoPtr<IJobSystem> m_Jobs;
        WorldImpl* m_World;
        uint32_t m_ThreadCount = 0;                // IJobSystem::GetThreadCount; buffers per system are one more
        ComponentId m_FrameTimeId = ComponentId::Invalid;
        FrameTime* m_FrameTime = nullptr;          // the world's resource

        Vector<SystemRecord*> m_Systems;           // by SystemId; null for free ids
        Vector<uint32_t> m_FreeSystemIds;
        uint64_t m_NextRegistration = 1;
        uint32_t m_LiveSystemCount = 0;
        Vector<PhaseRecord*> m_Phases;             // by Phase value
        Vector<Phase> m_PhaseOrder;                // the run order, named phases included
        Vector<DomainRecord*> m_Domains;           // by SimulationDomainId; null for free ids
        Vector<uint32_t> m_FreeDomainIds;

        bool m_OrderDirty = true;
        bool m_ExecutionDirty = true;
        nvrhi::FRESULT m_BuildResult = nvrhi::FS_OK;
        uint32_t m_AmbiguityCount = 0;
        nvrhi::string m_Report;                    // the last order build's diagnostics

        Vector<SystemRecord*> m_Pending;           // systems with commands, in playback order
        Vector<ICommandBuffer*> m_PlaybackScratch;
        Vector<JobHandle> m_DependencyScratch;
        PhaseRecord* m_CurrentPhase = nullptr;
        bool m_ExclusiveRunning = false;           // an exclusive system runs: alone, so its writes are not checked
        std::atomic<bool> m_Running{ false };

        SchedulerCounters m_Counters{};
        std::atomic<uint64_t> m_SystemRuns{ 0 };
        std::atomic<uint64_t> m_FixedSteps{ 0 };
    };

    // --------------------------------------------------------------------------------------- context methods

    IWorld* SystemContextImpl::GetWorld() noexcept
    {
        return &m_Record->scheduler->GetWorldImpl();
    }

    ICommandBuffer* SystemContextImpl::GetCommandBuffer() noexcept
    {
        return m_Record->buffers[m_Record->scheduler->GetBufferSlot(*m_Record)];
    }

    IJobSystem* SystemContextImpl::GetJobSystem() noexcept
    {
        return &m_Record->scheduler->GetJobs();
    }

    const FrameTime& SystemContextImpl::GetFrameTime() noexcept
    {
        return m_Record->scheduler->GetFrameTimeValue();
    }

    // ------------------------------------------------------------------------------------ simulation domains

    // The context a domain's calls receive (EcsSimulation.h). E6 fills in the column views and GPU command lists.
    NVRHI_CLASS_CLSID(SimulationContextImpl, "297c7982-d1d9-4e5a-bd51-9d0b5899eb16")
    class SimulationContextImpl final : public nvrhi::ObjectImpl<ISimulationContext>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SimulationContextImpl)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SimulationContextImpl)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationContext)
        NVRHI_IMPLEMENTS_CLASS(SimulationContextImpl)
        NVRHI_END_INTERFACE_TABLE()

        explicit SimulationContextImpl(DomainRecord* domain) noexcept : m_Domain(domain) { }

        IWorld* GetWorld() noexcept override { return &m_Domain->scheduler->GetWorldImpl(); }
        SimulationDomainId GetDomainId() noexcept override { return m_Domain->id; }
        // E6: views over the lifetime query's chunks.
        uint32_t GetColumnCount(ComponentId) noexcept override { return 0; }
        uint32_t GetColumns(ComponentId, uint32_t, uint64_t, uint32_t, ColumnView*) noexcept override { return 0; }
        uint64_t GetLastGatherVersion() noexcept override
        {
            return m_Domain->gather ? m_Domain->gather->lastRunVersion : c_AllVersions;
        }
        // E6: GPU queues. AddSimulationDomain accepts CPU domains only.
        nvrhi::ICommandList* GetCommandList() noexcept override { return nullptr; }
        ICommandBuffer* GetCommandBuffer() noexcept override
        {
            return m_Domain->current ? m_Domain->current->GetCommandBuffer() : nullptr;
        }
        uint64_t GetStepIndex() noexcept override { return m_Domain->stepIndex; }

    private:
        DomainRecord* m_Domain;
    };

    enum class DomainRole : uint8_t { Gather, Step, Scatter };

    // The gather, step and scatter systems of a domain. They hold the record, which the scheduler owns.
    class DomainSystem final : public nvrhi::ObjectImpl<ISystem>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DomainSystem)
        NVRHI_IMPLEMENTS_INTERFACE(ISystem)
        NVRHI_END_INTERFACE_TABLE()

        DomainSystem(DomainRecord* domain, DomainRole role) noexcept : m_Domain(domain), m_Role(role)
        {
            static const char* const suffixes[3] = { ".gather", ".step", ".scatter" };
            m_Name = domain->name;
            m_Name.append(suffixes[uint32_t(role)]);
            if (role == DomainRole::Gather)
                for (const ComponentAccess& access : domain->inputs)
                    m_Accesses.push_back(access);
            else if (role == DomainRole::Scatter)
                for (const ComponentAccess& access : domain->outputs)
                    m_Accesses.push_back(access);
            // The step declares nothing while ISimulationContext::GetColumns returns no views; once E6 gives the step
            // column views, it must declare the domain's column accesses.
        }

        SystemDesc& GetDesc(SystemDesc& retVal) noexcept override
        {
            retVal = SystemDesc{};
            retVal.name = m_Name.c_str();
            retVal.accesses = m_Accesses.data();
            retVal.accessCount = uint32_t(m_Accesses.size());
            retVal.phase = m_Role == DomainRole::Gather ? Phase::PreSimulation
                         : m_Role == DomainRole::Step ? Phase::FixedSimulation : Phase::PostSimulation;
            // Structural commands of a step are played back before the next step, which needs the phase to itself.
            const bool changesStructure = Has(m_Domain->flags, SimulationDomainFlags::ChangesStructure);
            retVal.flags = m_Role == DomainRole::Step && changesStructure ? SystemFlags::Exclusive : SystemFlags::None;
            return retVal;
        }

        nvrhi::FRESULT Update(ISystemContext* context) noexcept override
        {
            DomainRecord& domain = *m_Domain;
            ISimulationContext* simulation = domain.context.Get();
            domain.current = context;
            nvrhi::FRESULT result = nvrhi::FS_OK;
            switch (m_Role)
            {
            case DomainRole::Gather:
                result = domain.domain->GatherInputs(simulation);
                break;
            case DomainRole::Scatter:
                result = domain.domain->ScatterOutputs(simulation);
                break;
            case DomainRole::Step:
                result = Step(context, simulation);
                break;
            }
            domain.current = nullptr;
            return result;
        }

    private:
        nvrhi::FRESULT Step(ISystemContext* context, ISimulationContext* simulation) noexcept
        {
            DomainRecord& domain = *m_Domain;
            const double dt = domain.fixedTimeStep;
            // floor(accumulated / dt) steps, at most maxStepsPerFrame; the remainder fraction stays, whole steps beyond
            // the limit are dropped (8.4).
            const double whole = domain.accumulator >= dt ? std::floor(domain.accumulator / dt) : 0.0;
            domain.accumulator = std::max(0.0, domain.accumulator - whole * dt);
            const uint32_t steps = whole >= double(domain.maxStepsPerFrame) ? domain.maxStepsPerFrame : uint32_t(whole);
            const bool sync = Has(domain.flags, SimulationDomainFlags::ChangesStructure);
            SystemRecord* record = static_cast<SystemContextImpl*>(context)->GetRecord();
            nvrhi::FRESULT first = nvrhi::FS_OK;
            for (uint32_t s = 0; s < steps; ++s)
            {
                const nvrhi::FRESULT result = domain.domain->Step(dt, simulation);
                if (NVRHI_FAILED(result) && first == nvrhi::FS_OK)
                    first = result;
                ++domain.stepIndex;
                domain.scheduler->CountFixedStep();
                if (sync)
                    domain.scheduler->SyncInsideStep(*record);
            }
            return first;
        }

        DomainRecord* m_Domain;
        DomainRole m_Role;
        nvrhi::string m_Name;
        Vector<ComponentAccess> m_Accesses;
    };

    // --------------------------------------------------------------------------------------- construction

    SchedulerImpl::SchedulerImpl(WorldImpl* world, IJobSystem* jobs) noexcept
        : m_WorldReference(static_cast<IWorld*>(world))
        , m_Jobs(jobs)
        , m_World(world)
    {
    }

    SchedulerImpl::~SchedulerImpl()
    {
        // Pending commands are dropped with their buffers.
        for (SystemRecord* record : m_Systems)
            delete record;
        m_Systems.clear();
        for (DomainRecord* domain : m_Domains)
            delete domain;
        m_Domains.clear();
        for (PhaseRecord* phase : m_Phases)
            delete phase;
        m_Phases.clear();
    }

    nvrhi::FRESULT SchedulerImpl::Initialize() noexcept
    {
        m_ThreadCount = m_Jobs->GetThreadCount();
        nvrhi::FRESULT result = m_World->RegisterResource(ComponentTraits<FrameTime>::Desc(), nullptr, m_FrameTimeId);
        if (NVRHI_FAILED(result))
            return result;
        m_FrameTime = static_cast<FrameTime*>(m_World->GetResource(m_FrameTimeId));
        if (!m_FrameTime)
            return nvrhi::FE_INVALID_ARGS;

        for (uint32_t p = 0; p < uint32_t(Phase::Count); ++p)
        {
            PhaseRecord* phase = new (std::nothrow) PhaseRecord();
            if (!phase)
                return nvrhi::FE_OUT_OF_MEMORY;
            phase->name = c_BuiltinPhaseNames[p];
            phase->id = Phase(p);
            phase->syncAfter = Phase(p) == Phase::PostSimulation || Phase(p) == Phase::RenderExtract;
            m_Phases.push_back(phase);
            m_PhaseOrder.push_back(Phase(p));
        }
        m_PlaybackScratch.reserve(m_ThreadCount + 1);
        return nvrhi::FS_OK;
    }

    bool SchedulerImpl::RefuseWhileRunning(const char* operation) noexcept
    {
        if (!IsRunning())
            return false;
        log::warning("ECS scheduler: %s while a phase runs; refused", operation);
        return true;
    }

    uint32_t SchedulerImpl::FindPhasePosition(Phase phase) const noexcept
    {
        for (uint32_t i = 0; i < m_PhaseOrder.size(); ++i)
            if (m_PhaseOrder[i] == phase)
                return i;
        return c_None;
    }

    const char* SchedulerImpl::ComponentName(ComponentId component) const noexcept
    {
        const ComponentType* type = m_World->GetType(component);
        return type && type->desc.name ? type->desc.name : "(unnamed)";
    }

    // ------------------------------------------------------------------------------------ phases and systems

    nvrhi::FRESULT SchedulerImpl::AddPhase(const char* name, Phase after, Phase& outPhase) noexcept
    {
        outPhase = c_NoPhase;
        if (RefuseWhileRunning("AddPhase"))
            return nvrhi::FE_INVALID_ARGS;
        const uint32_t position = FindPhasePosition(after);
        if (!name || !*name || position == c_None)
        {
            log::warning("ECS scheduler: AddPhase('%s'): %s", name ? name : "", position == c_None
                         ? "`after` is not a phase" : "a phase needs a name");
            return nvrhi::FE_INVALID_ARGS;
        }
        for (const PhaseRecord* phase : m_Phases)
        {
            if (std::strcmp(phase->name.c_str(), name) == 0)
            {
                log::warning("ECS scheduler: AddPhase('%s'): the name is taken", name);
                return nvrhi::FE_INVALID_ARGS;
            }
        }
        PhaseRecord* phase = new (std::nothrow) PhaseRecord();
        if (!phase)
            return nvrhi::FE_OUT_OF_MEMORY;
        phase->name = name;
        phase->id = Phase(uint32_t(m_Phases.size()));   // from Phase::Count up
        m_Phases.push_back(phase);
        // Right after `after`, before whatever ran after it until now.
        m_PhaseOrder.insert(m_PhaseOrder.begin() + position + 1, phase->id);
        m_OrderDirty = true;
        outPhase = phase->id;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT SchedulerImpl::AddSystem(ISystem* system, SystemId& outId) noexcept
    {
        outId = SystemId::Invalid;
        if (RefuseWhileRunning("AddSystem"))
            return nvrhi::FE_INVALID_ARGS;
        return AddSystemInternal(system, nullptr, outId);
    }

    nvrhi::FRESULT SchedulerImpl::AddSystemInternal(ISystem* system, DomainRecord* domain, SystemId& outId) noexcept
    {
        outId = SystemId::Invalid;
        if (!system)
        {
            log::warning("ECS scheduler: AddSystem: the system is null");
            return nvrhi::FE_INVALID_ARGS;
        }
        SystemDesc desc{};
        system->GetDesc(desc);
        const char* name = desc.name ? desc.name : "";
        auto fail = [&](const char* reason) noexcept
        {
            log::warning("ECS scheduler: AddSystem('%s'): %s", name, reason);
            return nvrhi::FE_INVALID_ARGS;
        };
        if (!*name)
            return fail("a system needs a name");
        for (const SystemRecord* other : m_Systems)
            if (other && !other->removed && std::strcmp(other->name.c_str(), name) == 0)
                return fail("the name is taken");
        if (!FindPhase(desc.phase))
            return fail("the phase does not exist");
        if (!!(desc.flags & ~c_KnownSystemFlags))
            return fail("unknown flags");
        if (desc.accessCount != 0 && !desc.accesses)
            return fail("accessCount without an access array");
        for (uint32_t i = 0; i < desc.accessCount; ++i)
        {
            if (!m_World->GetType(desc.accesses[i].component))
                return fail("an access names a component that is not registered");
            if (uint8_t(desc.accesses[i].access) > uint8_t(Access::Write))
                return fail("an access is neither Filter, Read nor Write");
        }
        if ((desc.runBeforeCount != 0 && !desc.runBefore) || (desc.runAfterCount != 0 && !desc.runAfter))
            return fail("an ordering count without its array");
        for (uint32_t i = 0; i < desc.runBeforeCount; ++i)
            if (!desc.runBefore[i])
                return fail("a null runBefore name");
        for (uint32_t i = 0; i < desc.runAfterCount; ++i)
            if (!desc.runAfter[i])
                return fail("a null runAfter name");

        SystemRecord* record = new (std::nothrow) SystemRecord();
        if (!record)
            return nvrhi::FE_OUT_OF_MEMORY;
        record->scheduler = this;
        record->system = system;
        record->name = name;
        record->phase = desc.phase;
        record->flags = desc.flags;
        record->enabled = !Has(desc.flags, SystemFlags::StartDisabled);
        record->domain = domain;
        for (uint32_t i = 0; i < desc.accessCount; ++i)
            record->accesses.push_back(desc.accesses[i]);
        NormalizeAccesses(record->accesses);
        for (uint32_t i = 0; i < desc.runBeforeCount; ++i)
            record->runBefore.push_back(nvrhi::string(desc.runBefore[i]));
        for (uint32_t i = 0; i < desc.runAfterCount; ++i)
            record->runAfter.push_back(nvrhi::string(desc.runAfter[i]));

        SystemContextImpl* context = nullptr;
        try
        {
            context = MAKE_RC_OBJ(SystemContextImpl, record);
        }
        catch (...)
        {
            context = nullptr;
        }
        if (!context)
        {
            delete record;
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        record->context.Attach(context);

        record->buffers.resize(m_ThreadCount + 1, nullptr);
        for (ICommandBuffer*& buffer : record->buffers)
        {
            const nvrhi::FRESULT result = m_World->CreateCommandBuffer(&buffer);
            if (NVRHI_FAILED(result))
            {
                delete record;
                return result;
            }
        }

        uint32_t index;
        if (!m_FreeSystemIds.empty())
        {
            index = m_FreeSystemIds.back();
            m_FreeSystemIds.pop_back();
            m_Systems[index] = record;
        }
        else
        {
            index = uint32_t(m_Systems.size());
            m_Systems.push_back(record);
        }
        record->id = SystemId(index);
        record->registration = m_NextRegistration++;
        ++m_LiveSystemCount;
        // Each record is in the pending list at most once.
        m_Pending.reserve(m_Systems.size());
        m_OrderDirty = true;
        outId = record->id;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT SchedulerImpl::RemoveSystem(SystemId system) noexcept
    {
        if (RefuseWhileRunning("RemoveSystem"))
            return nvrhi::FE_INVALID_ARGS;
        SystemRecord* record = FindSystem(system);
        if (!record)
            return nvrhi::FE_INVALID_ARGS;
        if (record->domain)
        {
            log::warning("ECS scheduler: RemoveSystem('%s'): the system belongs to a simulation domain; remove the "
                         "domain", record->name.c_str());
            return nvrhi::FE_INVALID_ARGS;
        }
        RemoveSystemInternal(*record);
        return nvrhi::FS_OK;
    }

    void SchedulerImpl::RemoveSystemInternal(SystemRecord& record) noexcept
    {
        record.system = nullptr;
        record.removed = true;
        --m_LiveSystemCount;
        m_OrderDirty = true;
        // Commands it recorded are still played back at the next sync point, in their place.
        if (!record.pending)
            FreeRecord(&record);
    }

    void SchedulerImpl::FreeRecord(SystemRecord* record) noexcept
    {
        const uint32_t index = uint32_t(record->id);
        m_Systems[index] = nullptr;
        m_FreeSystemIds.push_back(index);
        delete record;
    }

    nvrhi::FRESULT SchedulerImpl::SetSystemEnabled(SystemId system, bool enabled) noexcept
    {
        if (RefuseWhileRunning("SetSystemEnabled"))
            return nvrhi::FE_INVALID_ARGS;
        SystemRecord* record = FindSystem(system);
        if (!record)
            return nvrhi::FE_INVALID_ARGS;
        if (record->enabled != enabled)
        {
            record->enabled = enabled;
            m_ExecutionDirty = true;
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT SchedulerImpl::GetSystemStats(SystemId system, SystemStats& out) noexcept
    {
        // The threads that run systems write the statistics during a phase: read them between phases.
        if (IsRunning())
        {
            m_World->ReportValidation("IScheduler::GetSystemStats",
                                      "called while a phase runs; read it between phases");
            out = SystemStats{};
            return nvrhi::FE_INVALID_ARGS;
        }
        const SystemRecord* record = FindSystem(system);
        if (!record)
        {
            out = SystemStats{};
            return nvrhi::FE_INVALID_ARGS;
        }
        out = record->stats;
        return nvrhi::FS_OK;
    }

    // ------------------------------------------------------------------------------------ simulation domains

    nvrhi::FRESULT SchedulerImpl::AddSimulationDomain(ISimulationDomain* domain, SimulationDomainId& outId) noexcept
    {
        outId = SimulationDomainId::Invalid;
        if (RefuseWhileRunning("AddSimulationDomain"))
            return nvrhi::FE_INVALID_ARGS;
        if (!domain)
            return nvrhi::FE_INVALID_ARGS;
        SimulationDomainDesc desc{};
        domain->GetDesc(desc);
        const char* name = desc.name ? desc.name : "";
        auto fail = [&](nvrhi::FRESULT result, const char* reason) noexcept
        {
            log::warning("ECS scheduler: AddSimulationDomain('%s'): %s", name, reason);
            return result;
        };
        if (!*name)
            return fail(nvrhi::FE_INVALID_ARGS, "a domain needs a name");
        if (!(desc.fixedTimeStep > 0.0) || !std::isfinite(desc.fixedTimeStep))
            return fail(nvrhi::FE_INVALID_ARGS, "fixedTimeStep must be a positive number of seconds");
        if (desc.maxStepsPerFrame == 0)
            return fail(nvrhi::FE_INVALID_ARGS, "maxStepsPerFrame must be at least 1");
        if ((desc.inputCount != 0 && !desc.inputs) || (desc.outputCount != 0 && !desc.outputs))
            return fail(nvrhi::FE_INVALID_ARGS, "an access count without its array");
        if (!!(desc.flags & ~c_KnownDomainFlags))
            return fail(nvrhi::FE_INVALID_ARGS, "unknown flags");
        if (desc.queue != SimulationQueue::Cpu)
            return fail(nvrhi::FE_UNSUPPORTED, "GPU queues come with plan step E6; CPU domains only");

        DomainRecord* record = new (std::nothrow) DomainRecord();
        if (!record)
            return nvrhi::FE_OUT_OF_MEMORY;
        record->scheduler = this;
        record->domain = domain;
        record->name = name;
        for (uint32_t i = 0; i < desc.inputCount; ++i)
            record->inputs.push_back(desc.inputs[i]);
        for (uint32_t i = 0; i < desc.outputCount; ++i)
            record->outputs.push_back(desc.outputs[i]);
        record->fixedTimeStep = desc.fixedTimeStep;
        record->maxStepsPerFrame = desc.maxStepsPerFrame;
        record->substeps = desc.substeps;
        record->flags = desc.flags;

        uint32_t index;
        if (!m_FreeDomainIds.empty())
        {
            index = m_FreeDomainIds.back();
            m_FreeDomainIds.pop_back();
            m_Domains[index] = record;
        }
        else
        {
            index = uint32_t(m_Domains.size());
            m_Domains.push_back(record);
        }
        record->id = SimulationDomainId(index);

        auto discard = [&](nvrhi::FRESULT result) noexcept
        {
            for (SystemId& id : record->systems)
            {
                if (SystemRecord* system = FindSystem(id))
                    RemoveSystemInternal(*system);
                id = SystemId::Invalid;
            }
            m_Domains[index] = nullptr;
            m_FreeDomainIds.push_back(index);
            delete record;
            return result;
        };

        try
        {
            record->context = MAKE_RC_OBJ_PTR(SimulationContextImpl, record);
        }
        catch (...)
        {
        }
        if (!record->context)
            return discard(nvrhi::FE_OUT_OF_MEMORY);
        for (uint32_t role = 0; role < 3; ++role)
        {
            nvrhi::AutoPtr<ISystem> system;
            try
            {
                system = MAKE_RC_OBJ_PTR(DomainSystem, record, DomainRole(role));
            }
            catch (...)
            {
            }
            if (!system)
                return discard(nvrhi::FE_OUT_OF_MEMORY);
            const nvrhi::FRESULT result = AddSystemInternal(system.Get(), record, record->systems[role]);
            if (NVRHI_FAILED(result))
                return discard(result);
        }
        record->gather = FindSystem(record->systems[0]);
        // E6: observers on desc.lifetime (OnMatch/OnUnmatch) that call OnEntitiesAdded/OnEntitiesRemoved.
        outId = record->id;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT SchedulerImpl::RemoveSimulationDomain(SimulationDomainId domain) noexcept
    {
        if (RefuseWhileRunning("RemoveSimulationDomain"))
            return nvrhi::FE_INVALID_ARGS;
        const uint32_t index = uint32_t(domain);
        DomainRecord* record = index < m_Domains.size() ? m_Domains[index] : nullptr;
        if (!record)
            return nvrhi::FE_INVALID_ARGS;
        for (SystemId id : record->systems)
        {
            if (SystemRecord* system = FindSystem(id))
            {
                // A pending system outlives the domain until its commands are played back.
                system->domain = nullptr;
                RemoveSystemInternal(*system);
            }
        }
        m_Domains[index] = nullptr;
        m_FreeDomainIds.push_back(index);
        delete record;
        return nvrhi::FS_OK;
    }

    // ------------------------------------------------------------------------------------------------ build

    nvrhi::FRESULT SchedulerImpl::Build() noexcept
    {
        if (RefuseWhileRunning("Build"))
            return nvrhi::FE_INVALID_ARGS;
        return EnsureBuilt();
    }

    uint32_t SchedulerImpl::GetAmbiguityCount() noexcept
    {
        if (!IsRunning())
            EnsureBuilt();
        return m_AmbiguityCount;
    }

    void SchedulerImpl::GetDiagnostics(nvrhi::string& out) noexcept
    {
        // The failure counts are written by the threads that run systems: refused (empty) while a phase runs.
        if (IsRunning())
        {
            m_World->ReportValidation("IScheduler::GetDiagnostics",
                                      "called while a phase runs; read it between phases");
            out.clear();
            return;
        }
        EnsureBuilt();
        out = m_Report;
        for (const SystemRecord* record : m_Systems)
        {
            if (!record || (record->updateFailures == 0 && record->playbackFailures == 0))
                continue;
            AppendFormat(out, "system '%s': %llu failed updates (last 0x%08x), %llu failed playbacks (last 0x%08x)\n",
                         record->name.c_str(), (unsigned long long)record->updateFailures,
                         uint32_t(record->lastUpdateFailure), (unsigned long long)record->playbackFailures,
                         uint32_t(record->lastPlaybackFailure));
        }
    }

    nvrhi::FRESULT SchedulerImpl::EnsureBuilt() noexcept
    {
        if (m_OrderDirty)
        {
            m_Report.clear();
            m_AmbiguityCount = 0;
            m_BuildResult = nvrhi::FS_OK;
            AppendFormat(m_Report, "ECS scheduler: %u systems in %u phases\n", m_LiveSystemCount,
                         uint32_t(m_PhaseOrder.size()));
            for (Phase id : m_PhaseOrder)
                BuildOrder(*m_Phases[uint32_t(id)]);
            ++m_Counters.orderBuilds;
            m_OrderDirty = false;
            m_ExecutionDirty = true;
        }
        if (m_ExecutionDirty)
        {
            m_Counters.edges = 0;
            size_t dependencies = 1;
            for (PhaseRecord* phase : m_Phases)
            {
                BuildExecution(*phase);
                for (const RunEntry& entry : phase->run)
                    dependencies = std::max<size_t>(dependencies, entry.predecessorCount);
                dependencies = std::max<size_t>(dependencies, phase->sinks.size());
            }
            m_DependencyScratch.resize(dependencies, JobHandle::Null);
            ++m_Counters.executionBuilds;
            m_ExecutionDirty = false;
        }
        return m_BuildResult;
    }

    bool SchedulerImpl::Conflict(const SystemRecord& a, const SystemRecord& b, ComponentId& component,
                                 const SystemRecord*& writer) const noexcept
    {
        component = ComponentId::Invalid;
        writer = nullptr;
        if (Has(a.flags, SystemFlags::Exclusive) || Has(b.flags, SystemFlags::Exclusive))
            return true;
        size_t i = 0;
        size_t j = 0;
        while (i < a.accesses.size() && j < b.accesses.size())
        {
            const ComponentAccess& x = a.accesses[i];
            const ComponentAccess& y = b.accesses[j];
            if (uint32_t(x.component) < uint32_t(y.component))
                ++i;
            else if (uint32_t(y.component) < uint32_t(x.component))
                ++j;
            else
            {
                // A write conflicts with any other access; Filter accesses never conflict (EcsTypes.h).
                const bool xWrites = x.access == Access::Write && y.access != Access::Filter;
                const bool yWrites = y.access == Access::Write && x.access != Access::Filter;
                if (xWrites || yWrites)
                {
                    component = x.component;
                    writer = xWrites ? &a : &b;
                    return true;
                }
                ++i;
                ++j;
            }
        }
        return false;
    }

    void SchedulerImpl::BuildOrder(PhaseRecord& phase) noexcept
    {
        const bool validation = m_World->IsValidationEnabled();
        const char* phaseName = phase.name.c_str();

        // The phase's systems in registration order.
        Vector<SystemRecord*> systems;
        for (SystemRecord* record : m_Systems)
            if (record && !record->removed && record->phase == phase.id)
                systems.push_back(record);
        std::sort(systems.begin(), systems.end(), [](const SystemRecord* a, const SystemRecord* b)
        {
            return a->registration < b->registration;
        });
        const uint32_t n = uint32_t(systems.size());
        phase.order.clear();
        phase.valid = true;
        phase.reach.Reset(n, n);
        if (n == 0)
            return;

        // Explicit edges, by registration index.
        BitMatrix explicitEdges;
        explicitEdges.Reset(n, n);
        auto resolve = [&](const SystemRecord& from, const nvrhi::string& target, const char* relation) noexcept
        {
            for (uint32_t j = 0; j < n; ++j)
                if (std::strcmp(systems[j]->name.c_str(), target.c_str()) == 0)
                    return j;
            const SystemRecord* elsewhere = nullptr;
            for (const SystemRecord* record : m_Systems)
                if (record && !record->removed && std::strcmp(record->name.c_str(), target.c_str()) == 0)
                    elsewhere = record;
            if (elsewhere)
            {
                AppendFormat(m_Report, "phase '%s': '%s' %s '%s', a system of phase '%s'; ignored\n", phaseName,
                             from.name.c_str(), relation, target.c_str(),
                             m_Phases[uint32_t(elsewhere->phase)]->name.c_str());
                log::warning("ECS scheduler: '%s' %s '%s', which runs in another phase; ignored", from.name.c_str(),
                             relation, target.c_str());
            }
            else
            {
                AppendFormat(m_Report, "phase '%s': '%s' %s '%s', which matches no system; ignored\n", phaseName,
                             from.name.c_str(), relation, target.c_str());
                log::warning("ECS scheduler: '%s' %s '%s', which matches no system; ignored", from.name.c_str(),
                             relation, target.c_str());
            }
            return c_None;
        };
        for (uint32_t i = 0; i < n; ++i)
        {
            for (const nvrhi::string& target : systems[i]->runBefore)
            {
                const uint32_t j = resolve(*systems[i], target, "runs before");
                if (j != c_None)
                    explicitEdges.Set(i, j);
            }
            for (const nvrhi::string& target : systems[i]->runAfter)
            {
                const uint32_t j = resolve(*systems[i], target, "runs after");
                if (j != c_None)
                    explicitEdges.Set(j, i);
            }
        }

        // Topological order, ties by registration (Kahn's algorithm, always taking the earliest registered system).
        Vector<uint32_t> inDegree;
        inDegree.resize(n, 0);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j)
                inDegree[j] += explicitEdges.Test(i, j) ? 1 : 0;
        Vector<uint8_t> emitted;
        emitted.resize(n, 0);
        Vector<uint32_t> position;   // registration index -> schedule position
        position.resize(n, c_None);
        Vector<uint32_t> local;      // schedule position -> registration index
        for (uint32_t step = 0; step < n; ++step)
        {
            uint32_t next = c_None;
            for (uint32_t i = 0; i < n && next == c_None; ++i)
                if (!emitted[i] && inDegree[i] == 0)
                    next = i;
            if (next == c_None)
                break;
            emitted[next] = 1;
            position[next] = uint32_t(local.size());
            local.push_back(next);
            for (uint32_t j = 0; j < n; ++j)
                if (explicitEdges.Test(next, j))
                    --inDegree[j];
        }

        if (local.size() != n)
        {
            // A cycle: walk predecessors among the systems left until one repeats.
            Vector<uint32_t> path;
            Vector<uint32_t> onPath;
            onPath.resize(n, c_None);
            uint32_t current = c_None;
            for (uint32_t i = 0; i < n && current == c_None; ++i)
                if (!emitted[i])
                    current = i;
            while (onPath[current] == c_None)
            {
                onPath[current] = uint32_t(path.size());
                path.push_back(current);
                uint32_t predecessor = c_None;
                for (uint32_t i = 0; i < n && predecessor == c_None; ++i)
                    if (!emitted[i] && explicitEdges.Test(i, current))
                        predecessor = i;
                current = predecessor;
            }
            // path[k + 1] runs before path[k]: in running order the cycle is path[end], ..., path[first]. It is printed
            // from its earliest registered member.
            const size_t first = onPath[current];
            const size_t length = path.size() - first;
            size_t earliest = 0;
            for (size_t k = 1; k < length; ++k)
                if (path[path.size() - 1 - k] < path[path.size() - 1 - earliest])
                    earliest = k;
            nvrhi::string cycle;
            for (size_t k = 0; k <= length; ++k)
                AppendFormat(cycle, k ? " -> '%s'" : "'%s'",
                             systems[path[path.size() - 1 - (earliest + k) % length]]->name.c_str());
            AppendFormat(m_Report, "phase '%s': ordering cycle %s; the phase does not run until it is broken\n",
                         phaseName, cycle.c_str());
            log::warning("ECS scheduler: phase '%s' has an ordering cycle: %s", phaseName, cycle.c_str());
            phase.valid = false;
            m_BuildResult = nvrhi::FE_INVALID_ARGS;
            for (uint32_t i = 0; i < n; ++i)
                phase.order.push_back(systems[i]);
            return;
        }

        for (uint32_t p = 0; p < n; ++p)
            phase.order.push_back(systems[local[p]]);

        // Reachability through explicit edges only, then through explicit and conflict edges; every edge goes forward
        // in the schedule order, so one pass from the back closes each.
        BitMatrix explicitReach;
        explicitReach.Reset(n, n);
        for (uint32_t p = n; p-- > 0;)
            for (uint32_t q = p + 1; q < n; ++q)
                if (explicitEdges.Test(local[p], local[q]))
                    explicitReach.Absorb(p, q);

        BitMatrix& reach = phase.reach;
        uint32_t ambiguities = 0;
        for (uint32_t p = n; p-- > 0;)
        {
            for (uint32_t q = p + 1; q < n; ++q)
            {
                ComponentId component;
                const SystemRecord* writer;
                const bool conflict = Conflict(*phase.order[p], *phase.order[q], component, writer);
                if (!conflict && !explicitEdges.Test(local[p], local[q]))
                    continue;
                if (conflict && !explicitReach.Test(p, q))
                {
                    ++ambiguities;
                    const char* first = phase.order[p]->name.c_str();
                    const char* second = phase.order[q]->name.c_str();
                    if (component == ComponentId::Invalid)
                        AppendFormat(m_Report, "phase '%s': ambiguity: '%s' and '%s' ('%s' is exclusive) have no "
                                     "order; "
                                     "they run in registration order\n", phaseName, first, second,
                                     Has(phase.order[p]->flags, SystemFlags::Exclusive) ? first : second);
                    else
                        AppendFormat(m_Report, "phase '%s': ambiguity: '%s' and '%s' both access '%s' ('%s' writes) "
                                     "and have no order; they run in registration order\n", phaseName, first, second,
                                     ComponentName(component), writer->name.c_str());
                    if (validation)
                        log::warning("ECS scheduler: phase '%s': '%s' and '%s' conflict and have no order (6.4)",
                                     phaseName, first, second);
                }
                reach.Absorb(p, q);
            }
        }
        m_AmbiguityCount += ambiguities;

        AppendFormat(m_Report, "phase '%s':", phaseName);
        for (uint32_t p = 0; p < n; ++p)
            AppendFormat(m_Report, "%s '%s'%s", p ? "," : "", phase.order[p]->name.c_str(),
                         Has(phase.order[p]->flags, SystemFlags::Exclusive) ? " (exclusive)"
                         : Has(phase.order[p]->flags, SystemFlags::MainThread) ? " (main thread)" : "");
        AppendFormat(m_Report, "\n");
    }

    void SchedulerImpl::BuildExecution(PhaseRecord& phase) noexcept
    {
        phase.run.clear();
        phase.predecessors.clear();
        phase.script.clear();
        phase.sinks.clear();
        for (SystemRecord* record : phase.order)
            record->runPosition = c_None;
        if (!phase.valid)
        {
            phase.handles.clear();
            return;
        }

        const uint32_t n = uint32_t(phase.order.size());
        Vector<uint32_t> positions;   // run index -> schedule position
        for (uint32_t p = 0; p < n; ++p)
        {
            SystemRecord* record = phase.order[p];
            if (!record->enabled)
                continue;
            RunEntry entry{};
            entry.record = record;
            entry.kind = Has(record->flags, SystemFlags::Exclusive) ? RunKind::Exclusive
                       : Has(record->flags, SystemFlags::MainThread) ? RunKind::MainThread : RunKind::Job;
            record->runPosition = uint32_t(phase.run.size());
            phase.run.push_back(entry);
            positions.push_back(p);
        }
        const uint32_t count = uint32_t(phase.run.size());
        phase.handles.clear();
        phase.handles.resize(count, JobHandle::Null);

        // Each entry's nearest predecessors: the enabled systems it runs after, minus those that run before another of
        // them (the transitive reduction). Order through a disabled system still counts.
        const uint32_t words = phase.reach.Words();
        Vector<uint64_t> candidates;
        candidates.resize(words, 0);
        Vector<uint32_t> gate;              // the last calling-thread entry an entry's job waits for, or c_None
        gate.resize(count, c_None);
        Vector<uint8_t> hasJobSuccessor;
        hasJobSuccessor.resize(count, 0);
        for (uint32_t r = 0; r < count; ++r)
        {
            const uint32_t p = positions[r];
            std::fill(candidates.begin(), candidates.end(), 0ull);
            for (uint32_t s = 0; s < r; ++s)
                if (phase.reach.Test(positions[s], p))
                    candidates[positions[s] >> 6] |= 1ull << (positions[s] & 63);
            RunEntry& entry = phase.run[r];
            entry.firstPredecessor = uint32_t(phase.predecessors.size());
            uint32_t latestMain = c_None;
            for (uint32_t s = 0; s < r; ++s)
            {
                const uint32_t ps = positions[s];
                if (((candidates[ps >> 6] >> (ps & 63)) & 1) == 0)
                    continue;
                const uint64_t* later = phase.reach.Row(ps);
                bool redundant = false;
                for (uint32_t w = 0; w < words && !redundant; ++w)
                    redundant = (later[w] & candidates[w]) != 0;
                if (redundant)
                    continue;
                phase.predecessors.push_back(s);
                const uint32_t main = phase.run[s].kind != RunKind::Job ? s : gate[s];
                if (main != c_None && (latestMain == c_None || main > latestMain))
                    latestMain = main;
                if (entry.kind == RunKind::Job)
                    hasJobSuccessor[s] = 1;
            }
            entry.predecessorCount = uint32_t(phase.predecessors.size()) - entry.firstPredecessor;
            gate[r] = entry.kind == RunKind::Job ? latestMain : c_None;
        }

        // The script: the jobs that wait for no calling-thread system, then each calling-thread system followed by the
        // jobs whose last such predecessor it is. A job's predecessors are always handled before it.
        for (uint32_t r = 0; r < count; ++r)
            if (phase.run[r].kind == RunKind::Job && gate[r] == c_None)
                phase.script.push_back(r);
        for (uint32_t m = 0; m < count; ++m)
        {
            if (phase.run[m].kind == RunKind::Job)
                continue;
            phase.script.push_back(m | c_MainBit);
            for (uint32_t r = m + 1; r < count; ++r)
                if (phase.run[r].kind == RunKind::Job && gate[r] == m)
                    phase.script.push_back(r);
        }
        for (uint32_t r = 0; r < count; ++r)
            if (phase.run[r].kind == RunKind::Job && !hasJobSuccessor[r])
                phase.sinks.push_back(r);
        m_Counters.edges += phase.predecessors.size() + (phase.sinks.size() > 1 ? phase.sinks.size() : 0);
    }

    uint32_t SchedulerImpl::GetScheduleOrder(Phase phase, SystemId* out, uint32_t capacity) noexcept
    {
        if (!IsRunning())
            EnsureBuilt();
        const PhaseRecord* record = FindPhase(phase);
        if (!record)
            return 0;
        const uint32_t count = uint32_t(record->order.size());
        for (uint32_t i = 0; i < count && i < capacity && out; ++i)
            out[i] = record->order[i]->id;
        return count;
    }

    bool SchedulerImpl::PhaseDeclaresWrites(Phase phase) noexcept
    {
        if (!IsRunning())
            EnsureBuilt();
        const PhaseRecord* record = FindPhase(phase);
        if (!record)
            return false;
        for (const SystemRecord* system : record->order)
        {
            if (!system || !system->enabled)
                continue;
            if (Has(system->flags, SystemFlags::Exclusive))
                return true;
            for (const ComponentAccess& access : system->accesses)
            {
                if (access.access == Access::Write)
                    return true;
            }
        }
        return false;
    }

    // -------------------------------------------------------------------------------------------- running

    void SchedulerImpl::SetFrameTime(const FrameTime& time) noexcept
    {
        if (RefuseWhileRunning("SetFrameTime"))
            return;
        *m_FrameTime = time;
        const double delta = std::isfinite(time.deltaTime) && time.deltaTime > 0.0 ? time.deltaTime : 0.0;
        for (DomainRecord* domain : m_Domains)
            if (domain)
                domain->accumulator += delta;
    }

    nvrhi::FRESULT SchedulerImpl::RunPhase(Phase phase) noexcept
    {
        return RunPhases(phase, phase);
    }

    nvrhi::FRESULT SchedulerImpl::RunPhases(Phase first, Phase last) noexcept
    {
        const uint32_t begin = FindPhasePosition(first);
        const uint32_t end = FindPhasePosition(last);
        if (begin == c_None || end == c_None || begin > end)
        {
            log::warning("ECS scheduler: RunPhases: %s", begin == c_None || end == c_None
                         ? "not a phase" : "`first` runs after `last`");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (m_Running.exchange(true, std::memory_order_acq_rel))
        {
            log::warning("ECS scheduler: RunPhase while a phase runs (from a system?); refused");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (m_World->IsInPhase())
        {
            m_Running.store(false, std::memory_order_release);
            log::warning("ECS scheduler: RunPhase while the world is in a phase of another scheduler; refused");
            return nvrhi::FE_INVALID_ARGS;
        }
        EnsureBuilt();
        // Nothing runs unless every phase of the range can.
        for (uint32_t i = begin; i <= end; ++i)
        {
            if (!m_Phases[uint32_t(m_PhaseOrder[i])]->valid)
            {
                m_Running.store(false, std::memory_order_release);
                return nvrhi::FE_INVALID_ARGS;
            }
        }
        for (uint32_t i = begin; i <= end; ++i)
        {
            PhaseRecord& phase = *m_Phases[uint32_t(m_PhaseOrder[i])];
            // A system plays back once per run: commands still pending from an earlier run of this phase go first.
            if (phase.pendingCount != 0)
                SyncPoint();
            RunPhaseInternal(phase);
        }
        m_Running.store(false, std::memory_order_release);
        return nvrhi::FS_OK;
    }

    void SchedulerImpl::RunPhaseInternal(PhaseRecord& phase) noexcept
    {
        WorldImpl& world = *m_World;
        // Versions in schedule order, before anything runs (EcsScheduler.h).
        for (RunEntry& entry : phase.run)
        {
            SystemRecord& record = *entry.record;
            record.lastRunVersion = record.version;
            record.version = world.AdvanceVersion();
        }

        m_CurrentPhase = &phase;
        if (world.IsValidationEnabled())
            world.SetWriteAccessCheck(&SchedulerImpl::CheckWriteAccess, this);
        world.BeginPhase();
        std::fill(phase.handles.begin(), phase.handles.end(), JobHandle::Null);
        for (uint32_t step : phase.script)
        {
            if (step & c_MainBit)
                RunOnCallingThread(phase, step & ~c_MainBit);
            else
                ScheduleSystem(phase, step);
        }
        FinishJobs(phase);
        world.EndPhase();
        world.SetWriteAccessCheck(nullptr, nullptr);
        m_CurrentPhase = nullptr;

        // For writes outside phases, and what the sync point's playback stamps: newer than every run of the phase.
        world.AdvanceVersion();
        ++m_Counters.phaseRuns;
        CollectPending(phase);
        if (phase.syncAfter)
            SyncPoint();
    }

    void SchedulerImpl::ScheduleSystem(PhaseRecord& phase, uint32_t index) noexcept
    {
        const RunEntry& entry = phase.run[index];
        uint32_t count = 0;
        for (uint32_t i = 0; i < entry.predecessorCount; ++i)
        {
            const JobHandle handle = phase.handles[phase.predecessors[entry.firstPredecessor + i]];
            if (handle != JobHandle::Null)
                m_DependencyScratch[count++] = handle;
        }
        const JobDesc desc{ &SchedulerImpl::RunSystemJob, entry.record, 1, 1, entry.record->name.c_str() };
        JobHandle handle = JobHandle::Null;
        if (NVRHI_SUCCEEDED(m_Jobs->Schedule(desc, m_DependencyScratch.data(), count, handle)))
        {
            ++m_Counters.jobsScheduled;
            phase.handles[index] = handle;
            return;
        }
        // The job table or the dependency pool is full (FE_OUT_OF_MEMORY): run it here once its predecessors are done.
        // Its successors are scheduled later and find it finished.
        for (uint32_t i = 0; i < count; ++i)
            m_Jobs->Wait(m_DependencyScratch[i]);
        ++m_Counters.inlineRuns;
        RunSystem(*entry.record);
        phase.handles[index] = JobHandle::Null;
    }

    void SchedulerImpl::RunOnCallingThread(PhaseRecord& phase, uint32_t index) noexcept
    {
        const RunEntry& entry = phase.run[index];
        for (uint32_t i = 0; i < entry.predecessorCount; ++i)
        {
            const JobHandle handle = phase.handles[phase.predecessors[entry.firstPredecessor + i]];
            if (handle != JobHandle::Null)
                m_Jobs->Wait(handle);
        }
        if (entry.kind == RunKind::Exclusive)
        {
            // Everything before it has finished and nothing after it is scheduled yet: it runs alone. Its immediate
            // structural changes and the playbacks of its sync points (SyncInsideStep) stamp its own version, not the
            // phase's last one: the systems after it see them in this run and not again in their next.
            m_World->BeginExclusive();
            m_World->SetStampVersion(entry.record->version);
            m_ExclusiveRunning = true;
            RunSystem(*entry.record);
            m_ExclusiveRunning = false;
            m_World->SetStampVersion(0);
            m_World->EndExclusive();
        }
        else
            RunSystem(*entry.record);
    }

    void SchedulerImpl::FinishJobs(PhaseRecord& phase) noexcept
    {
        uint32_t count = 0;
        for (uint32_t index : phase.sinks)
            if (phase.handles[index] != JobHandle::Null)
                m_DependencyScratch[count++] = phase.handles[index];
        if (count == 0)
            return;
        if (count == 1)
        {
            m_Jobs->Wait(m_DependencyScratch[0]);
            return;
        }
        // One job without indices instead of waiting for each.
        const JobDesc sync{ nullptr, nullptr, 0, 0, phase.name.c_str() };
        JobHandle handle = JobHandle::Null;
        if (NVRHI_SUCCEEDED(m_Jobs->Schedule(sync, m_DependencyScratch.data(), count, handle)))
        {
            ++m_Counters.syncJobs;
            m_Jobs->Wait(handle);
            return;
        }
        for (uint32_t i = 0; i < count; ++i)
            m_Jobs->Wait(m_DependencyScratch[i]);
    }

    void SchedulerImpl::RunSystemJob(void* context, uint32_t, uint32_t, uint32_t) noexcept
    {
        SystemRecord& record = *static_cast<SystemRecord*>(context);
        record.scheduler->RunSystem(record);
    }

    void SchedulerImpl::CheckWriteAccess(void* context, ComponentId component, uint64_t writeVersion,
                                         const char* operation) noexcept
    {
        const SchedulerImpl& self = *static_cast<const SchedulerImpl*>(context);
        const PhaseRecord* phase = self.m_CurrentPhase;
        // An exclusive system, and the observers of its sync points, run alone: any write is safe.
        if (!phase || phase->run.empty() || self.m_ExclusiveRunning)
            return;
        if (writeVersion == c_WorldVersion)
        {
            self.m_World->ReportValidation(operation, "'%s': c_WorldVersion is for writes outside phases; a system "
                                           "passes its own version (ISystemContext::GetVersion)",
                                           self.ComponentName(component));
            return;
        }
        // The phase's versions were handed out one per run entry, consecutively (RunPhaseInternal).
        const uint64_t first = phase->run[0].record->version;
        if (writeVersion < first || writeVersion - first >= phase->run.size())
            return;   // not a version of this phase's systems: nothing to check it against
        const SystemRecord& record = *phase->run[size_t(writeVersion - first)].record;
        if (record.version != writeVersion || Has(record.flags, SystemFlags::Exclusive))
            return;
        SystemAccessInfo info{};
        info.accesses = record.accesses.data();
        info.accessCount = uint32_t(record.accesses.size());
        const Access declared = FindDeclaredAccess(info, component);
        if (declared != Access::Write)
            self.m_World->ReportValidation(operation, "system '%s' writes '%s', but declared %s (11.3)",
                                           record.name.c_str(), self.ComponentName(component),
                                           declared == Access::Read ? "Read access only" : "no access to it");
    }

    void SchedulerImpl::RunSystem(SystemRecord& record) noexcept
    {
        record.updateWorker = GetCurrentWorker();
        const auto start = std::chrono::steady_clock::now();
        const nvrhi::FRESULT result = record.system->Update(record.context.Get());
        const double milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        SystemStats& stats = record.stats;
        stats.lastMilliseconds = milliseconds;
        stats.averageMilliseconds = stats.runCount == 0 ? milliseconds
            : stats.averageMilliseconds + c_StatsSmoothing * (milliseconds - stats.averageMilliseconds);
        ++stats.runCount;
        m_SystemRuns.fetch_add(1, std::memory_order_relaxed);
        if (NVRHI_FAILED(result))
        {
            // Recorded in the diagnostics; the phase goes on (EcsScheduler.h).
            ++record.updateFailures;
            record.lastUpdateFailure = result;
            if (!record.reportedFailure)
            {
                record.reportedFailure = true;
                log::warning("ECS scheduler: system '%s' failed (0x%08x); see IScheduler::GetDiagnostics",
                             record.name.c_str(), uint32_t(result));
            }
        }
    }

    // ----------------------------------------------------------------------------------------- sync points

    void SchedulerImpl::CollectPending(PhaseRecord& phase) noexcept
    {
        for (const RunEntry& entry : phase.run)
        {
            SystemRecord& record = *entry.record;
            if (record.pending || !record.HasCommands())
                continue;
            record.pending = true;
            m_Pending.push_back(&record);
            ++phase.pendingCount;
        }
    }

    void SchedulerImpl::PlaySystem(SystemRecord& record) noexcept
    {
        m_PlaybackScratch.clear();
        for (ICommandBuffer* buffer : record.buffers)
            if (buffer->GetCommandCount() != 0)
                m_PlaybackScratch.push_back(buffer);
        if (m_PlaybackScratch.empty())
            return;
        // One playback per system: its buffers merge by sort key (EcsCommandBuffer.h).
        const nvrhi::FRESULT result = m_World->Playback(m_PlaybackScratch.data(), uint32_t(m_PlaybackScratch.size()));
        ++m_Counters.playbacks;
        if (NVRHI_FAILED(result))
        {
            ++record.playbackFailures;
            record.lastPlaybackFailure = result;
        }
    }

    void SchedulerImpl::PlayPending() noexcept
    {
        for (size_t i = 0; i < m_Pending.size(); ++i)
        {
            SystemRecord* record = m_Pending[i];
            PlaySystem(*record);
            record->pending = false;
            m_Phases[uint32_t(record->phase)]->pendingCount = 0;
            if (record->removed)
                FreeRecord(record);
        }
        m_Pending.clear();
    }

    void SchedulerImpl::SyncPoint() noexcept
    {
        ++m_Counters.syncPoints;
        PlayPending();
    }

    void SchedulerImpl::SyncInsideStep(SystemRecord& step) noexcept
    {
        // The step system is exclusive: every system of the phase before it has finished, none after it has started,
        // and the world allows structural changes. Playback order stays (phase run, schedule order).
        ++m_Counters.syncPoints;
        PlayPending();
        if (PhaseRecord* phase = m_CurrentPhase)
            for (uint32_t r = 0; r < step.runPosition && r < phase->run.size(); ++r)
                PlaySystem(*phase->run[r].record);
        PlaySystem(step);
    }

    // ----------------------------------------------------------------------------------- module-internal

    bool GetSystemAccessInfo(ISystemContext* context, SystemAccessInfo& out) noexcept
    {
        SystemContextImpl* impl = donut::query_cast<SystemContextImpl>(context);
        if (!impl)
            return false;
        const SystemRecord& record = *impl->GetRecord();
        out = SystemAccessInfo{};
        out.name = record.name.c_str();
        out.accesses = record.accesses.data();
        out.accessCount = uint32_t(record.accesses.size());
        out.exclusive = Has(record.flags, SystemFlags::Exclusive);
        out.version = record.version;
        out.world = &record.scheduler->GetWorldImpl();
        return true;
    }

    Access FindDeclaredAccess(const SystemAccessInfo& info, ComponentId component) noexcept
    {
        const ComponentAccess* begin = info.accesses;
        const ComponentAccess* end = info.accesses + info.accessCount;
        const ComponentAccess* found = std::lower_bound(begin, end, component,
            [](const ComponentAccess& access, ComponentId id) { return uint32_t(access.component) < uint32_t(id); });
        return found != end && found->component == component ? found->access : Access::Filter;
    }

    bool GetSchedulerCounters(IScheduler* scheduler, SchedulerCounters& out) noexcept
    {
        SchedulerImpl* impl = donut::query_cast<SchedulerImpl>(scheduler);
        if (!impl)
            return false;
        out = impl->GetCounters();
        return true;
    }

    uint32_t GetScheduleOrder(IScheduler* scheduler, Phase phase, SystemId* out, uint32_t capacity) noexcept
    {
        SchedulerImpl* impl = donut::query_cast<SchedulerImpl>(scheduler);
        return impl ? impl->GetScheduleOrder(phase, out, capacity) : 0;
    }

    bool PhaseDeclaresWrites(IScheduler* scheduler, Phase phase) noexcept
    {
        SchedulerImpl* impl = donut::query_cast<SchedulerImpl>(scheduler);
        return impl ? impl->PhaseDeclaresWrites(phase) : true;
    }

    uint32_t GetSystemCommandBuffers(IScheduler* scheduler, SystemId system, ICommandBuffer** out,
                                     uint32_t capacity) noexcept
    {
        SchedulerImpl* impl = donut::query_cast<SchedulerImpl>(scheduler);
        return impl ? impl->GetCommandBuffers(system, out, capacity) : 0;
    }
}

namespace donut::engine
{
    nvrhi::FRESULT CreateScheduler(IWorld* world, IJobSystem* jobs, IScheduler** ppScheduler) noexcept
    {
        if (!ppScheduler)
            return nvrhi::FE_INVALID_ARGS;
        *ppScheduler = nullptr;
        ecs::WorldImpl* impl = ecs::GetWorldImpl(world);
        if (!impl || !jobs)
        {
            log::warning("CreateScheduler: %s", !jobs ? "the job system is null"
                         : "the world is null or not a world of this module");
            return nvrhi::FE_INVALID_ARGS;
        }
        ecs::SchedulerImpl* scheduler = nullptr;
        try
        {
            scheduler = MAKE_RC_OBJ(ecs::SchedulerImpl, impl, jobs);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        if (!scheduler)
            return nvrhi::FE_OUT_OF_MEMORY;
        const nvrhi::FRESULT result = scheduler->Initialize();
        if (NVRHI_FAILED(result))
        {
            scheduler->Release();
            return result;
        }
        *ppScheduler = scheduler;
        return nvrhi::FS_OK;
    }
}
