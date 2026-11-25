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

// Systems, phases and the scheduler (aggregate docs/ecs-design.md 6.2-6.4). Each phase's systems form a dependency
// graph built from their declared accesses and ordering constraints; conflicting systems without an order are
// ambiguities (6.4). Interfaces follow nvrhi's ABI rules (com-interface-abi.md).
//
// Schedule order: a phase's systems in dependency order, ties broken by registration order (never SystemId, which may
// be reused). It decides two things that must not depend on threads (rule 3.3.6):
// - Versions: before a phase runs, every system gets its version from IWorld::AdvanceVersion in schedule order; after
//   the phase one more version is taken, for writes made outside phases.
// - Playback: at a sync point each system's buffers are played back once, systems in schedule order (EcsCommandBuffer.h).

#pragma once

#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <nvrhi/core/containers.h>
#include <nvrhi/core/foundation.h>

#include <new>
#include <type_traits>
#include <utility>

// The flag enum keeps nvrhi's `None`; X11 defines it as a macro.
#if defined(None)
#define DONUT_ECS_SCHEDULER_RESTORE_NONE
#pragma push_macro("None")
#undef None
#endif

namespace nvrhi
{
    struct ICommandList;   // GPU simulation domains record into one (8.1); the ECS headers do not include nvrhi.h
}

namespace donut::engine
{
    struct ISimulationDomain;   // EcsSimulation.h

    // The built-in phases, in order (6.3). IScheduler::AddPhase inserts named phases between them; those get values
    // from Count up.
    enum class Phase : uint32_t
    {
        BeginFrame,             // time, input, view list, asset-table maintenance
        PreSimulation,          // animation sampling, scripts, simulation input gathering
        FixedSimulation,        // zero or more fixed steps per domain (8.4)
        PostSimulation,         // scatter results, interpolation; sync point after
        TransformPropagation,   // local to world, by depth level
        Bounds,                 // world bounds, chunk bounds
        RenderExtract,          // frame snapshot; sync point after
        EndFrame,               // deferred destruction, statistics
        Count
    };

    enum class SystemId : uint32_t { Invalid = 0xffffffffu };
    enum class SimulationDomainId : uint32_t { Invalid = 0xffffffffu };

    enum class SystemFlags : uint32_t
    {
        None          = 0,
        Exclusive     = 1u << 0,   // runs alone and may change structure immediately (rule 3.3.3)
        MainThread    = 1u << 1,   // runs on the thread that called RunPhase
        StartDisabled = 1u << 2,
    };
    template <> struct EnableEcsFlagOperators<SystemFlags> : std::true_type { };

    // Access to a component or a resource. A Write conflicts with any other access to the same id (6.4).
    struct ComponentAccess
    {
        ComponentId component;
        Access access;
        uint8_t reserved[3];
    };
    static_assert(sizeof(ComponentAccess) == 8, "ComponentAccess layout");

    // Every pointer must stay valid while the system is added. runBefore and runAfter name other systems of the
    // same phase; a name that matches no system is ignored and reported by Build.
    struct SystemDesc
    {
        const char* name;                      // unique per scheduler
        const ComponentAccess* accesses;
        uint32_t accessCount;
        Phase phase;
        const char* const* runBefore;
        const char* const* runAfter;
        uint32_t runBeforeCount;
        uint32_t runAfterCount;
        SystemFlags flags;
        uint32_t reserved;
    };
    static_assert(sizeof(void*) != 8 || sizeof(SystemDesc) == 56, "SystemDesc layout");

    // A world resource (4.9): the scheduler registers it and IScheduler::SetFrameTime writes it. Systems that read it
    // declare Read access to it like any other resource.
    struct FrameTime
    {
        double time;          // seconds since the scheduler started
        double deltaTime;     // seconds since the previous frame
        uint64_t frameIndex;
    };
    static_assert(sizeof(FrameTime) == 24, "FrameTime layout");
    DONUT_ECS_COMPONENT_EX(FrameTime, "5b0b6a3e-2f7c-4d1e-9a64-8c3e7f21d905", ::donut::engine::ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(time, deltaTime, frameIndex));

    struct SystemStats
    {
        uint64_t runCount;
        double lastMilliseconds;
        double averageMilliseconds;   // exponential moving average
    };
    static_assert(sizeof(SystemStats) == 24, "SystemStats layout");

    // Valid during ISystem::Update, on any thread that works for the system.
    NVRHI_IID(ISystemContext, "161e575e-334c-4040-9bf4-938b192e2f5e")
    struct ISystemContext : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ISystemContext)
        virtual IWorld* GetWorld() noexcept = 0;                    // not AddRef'd
        // This system's buffer for the calling thread's worker index (JobSystem.h).
        virtual ICommandBuffer* GetCommandBuffer() noexcept = 0;
        virtual IJobSystem* GetJobSystem() noexcept = 0;            // not AddRef'd
        virtual uint64_t GetVersion() noexcept = 0;                 // this run's version: what its writes stamp
        virtual uint64_t GetLastRunVersion() noexcept = 0;          // the sinceVersion for Changed and Added terms
        virtual const FrameTime& GetFrameTime() noexcept = 0;
        virtual SystemId GetSystemId() noexcept = 0;
    };

    NVRHI_IID(ISystem, "0f87e54d-c29c-4545-8842-6add4fee3767")
    struct ISystem : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ISystem)
        // Read once, by AddSystem.
        virtual SystemDesc& GetDesc(SystemDesc& retVal) noexcept = 0;
        // Called once per run of its phase. A failure is recorded in the diagnostics; the phase goes on.
        virtual nvrhi::FRESULT Update(ISystemContext* context) noexcept = 0;
    };

    NVRHI_IID(IScheduler, "7c35923f-243e-413b-9546-021f10b0e930")
    struct IScheduler : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IScheduler)

        // A named phase that runs right after `after` and before whatever ran after `after` until now.
        virtual nvrhi::FRESULT AddPhase(const char* name, Phase after, Phase& outPhase) noexcept = 0;
        // Holds a reference to the system until RemoveSystem or the scheduler's destruction.
        virtual nvrhi::FRESULT AddSystem(ISystem* system, SystemId& outId) noexcept = 0;
        virtual nvrhi::FRESULT RemoveSystem(SystemId system) noexcept = 0;
        virtual nvrhi::FRESULT SetSystemEnabled(SystemId system, bool enabled) noexcept = 0;

        // Adds the domain's gather, step and scatter systems to PreSimulation, FixedSimulation and PostSimulation with
        // the accesses of its descriptor, observers on its lifetime query, and its fixed-step accumulator (8.1, 8.4).
        // Holds a reference to the domain until RemoveSimulationDomain or the scheduler's destruction. A domain may
        // hold the world (a query of its own does) until it is destroyed, so nothing that the world holds, such as a
        // component value or an observer, may hold the domain or the scheduler; the scheduler's own observers hold
        // neither. Entities that match the lifetime query already are delivered before the call returns, in chunk
        // order, one batch per chunk. A domain without lifetime terms has no entities. FE_OUT_OF_MEMORY beyond
        // c_MaxSimulationDomains. One scheduler with domains per world (FixedStepState). Outside phases.
        virtual nvrhi::FRESULT AddSimulationDomain(ISimulationDomain* domain, SimulationDomainId& outId) noexcept = 0;
        // Delivers the removal of the entities that still match, the last chunk first, then releases the domain.
        // Outside phases.
        virtual nvrhi::FRESULT RemoveSimulationDomain(SimulationDomainId domain) noexcept = 0;
        // The open list a GPU domain records into this frame (null: the engine's own). Not AddRef'd: it must stay open
        // and alive until the frame's simulation phases have run; the scheduler forgets it after PostSimulation. On
        // the thread that runs the phases. FE_INVALID_ARGS inside a phase, for a Cpu domain or an unknown id.
        virtual nvrhi::FRESULT SetSimulationCommandList(SimulationDomainId domain,
                                                        nvrhi::ICommandList* list) noexcept = 0;
        // What a rewind of the simulation needs, of every domain (8.5): its time, its entities, the Relocatable columns
        // of the outputs and inputs declared Write with their enabled bits, and its CaptureState. Pending commands are
        // played back first. Outside phases. FE_UNSUPPORTED, with nothing written, when a domain cannot capture.
        // Authored state (animation, kinematic parents, inputs declared Read, the application's commands) is not in it:
        // a replay sets the authored values of the frames it repeats itself.
        virtual nvrhi::FRESULT SaveSimulationState(IStateWriter* writer) noexcept = 0;
        // Restores the saved state: the times and columns of every domain, with a new version, then the engines
        // (RestoreState), domain by domain. Needs the domains and the entities of the save: FE_OUT_OF_DATE, with
        // nothing written, when they differ. The first failure of a RestoreState ends the call and is returned, with
        // the state partly restored: the caller loads another state or removes the domain. Outside phases.
        virtual nvrhi::FRESULT LoadSimulationState(IStateReader* reader) noexcept = 0;

        // Orders every phase and validates it. FE_INVALID_ARGS on an ordering cycle. Ambiguities do not fail the build;
        // they are counted and described in the diagnostics. RunPhase builds when anything changed since.
        virtual nvrhi::FRESULT Build() noexcept = 0;
        virtual uint32_t GetAmbiguityCount() noexcept = 0;
        virtual void GetDiagnostics(nvrhi::string& out) noexcept = 0;

        // Called once per frame, before the first phase. Writes the world's FrameTime resource and adds the delta to
        // the domains' accumulators (FixedStepState).
        virtual void SetFrameTime(const FrameTime& time) noexcept = 0;
        // Runs one phase, then its sync point if it has one (6.3). The calling thread works too, in an external slot of
        // the job system.
        virtual nvrhi::FRESULT RunPhase(Phase phase) noexcept = 0;
        // Runs the phases from `first` to `last` inclusive, in phase order, named phases included.
        virtual nvrhi::FRESULT RunPhases(Phase first, Phase last) noexcept = 0;

        virtual IWorld* GetWorld() noexcept = 0;                    // not AddRef'd
        virtual IJobSystem* GetJobSystem() noexcept = 0;            // not AddRef'd
        virtual nvrhi::FRESULT GetSystemStats(SystemId system, SystemStats& out) noexcept = 0;
    };

    // The scheduler keeps references to the world and the job system.
    nvrhi::FRESULT CreateScheduler(IWorld* world, IJobSystem* jobs, IScheduler** ppScheduler) noexcept;

    // ----------------------------------------- systems from lambdas (header-only, outside the ABI surface, 6.2)

    namespace detail
    {
        template <typename F>
        class LambdaSystem final : public nvrhi::ObjectImpl<ISystem>
        {
        public:
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(LambdaSystem)
            NVRHI_IMPLEMENTS_INTERFACE(ISystem)
            NVRHI_END_INTERFACE_TABLE()

            LambdaSystem(const SystemDesc& desc, F&& update)
                : m_Update(std::forward<F>(update))
                , m_Name(desc.name ? desc.name : "")
                , m_Phase(desc.phase)
                , m_Flags(desc.flags)
            {
                for (uint32_t i = 0; i < desc.accessCount; ++i)
                    m_Accesses.push_back(desc.accesses[i]);
                for (uint32_t i = 0; i < desc.runBeforeCount; ++i)
                    m_Before.push_back(nvrhi::string(desc.runBefore[i]));
                for (uint32_t i = 0; i < desc.runAfterCount; ++i)
                    m_After.push_back(nvrhi::string(desc.runAfter[i]));
                for (const nvrhi::string& name : m_Before)
                    m_BeforePointers.push_back(name.c_str());
                for (const nvrhi::string& name : m_After)
                    m_AfterPointers.push_back(name.c_str());
            }

            SystemDesc& GetDesc(SystemDesc& retVal) noexcept override
            {
                retVal = SystemDesc{};
                retVal.name = m_Name.c_str();
                retVal.accesses = m_Accesses.data();
                retVal.accessCount = uint32_t(m_Accesses.size());
                retVal.phase = m_Phase;
                retVal.runBefore = m_BeforePointers.data();
                retVal.runAfter = m_AfterPointers.data();
                retVal.runBeforeCount = uint32_t(m_BeforePointers.size());
                retVal.runAfterCount = uint32_t(m_AfterPointers.size());
                retVal.flags = m_Flags;
                return retVal;
            }

            nvrhi::FRESULT Update(ISystemContext* context) noexcept override
            {
                if constexpr (std::is_same_v<decltype(m_Update(context)), nvrhi::FRESULT>)
                    return m_Update(context);
                else
                {
                    m_Update(context);
                    return nvrhi::FS_OK;
                }
            }

        private:
            std::decay_t<F> m_Update;
            nvrhi::string m_Name;
            Phase m_Phase;
            SystemFlags m_Flags;
            nvrhi::vector<ComponentAccess> m_Accesses;
            nvrhi::vector<nvrhi::string> m_Before;
            nvrhi::vector<nvrhi::string> m_After;
            nvrhi::vector<const char*> m_BeforePointers;
            nvrhi::vector<const char*> m_AfterPointers;
        };
    }

    // update(ISystemContext*) returns void or FRESULT. The descriptor's strings and arrays are copied. FE_OUT_OF_MEMORY
    // when the object or its copies cannot be allocated.
    template <typename F>
    nvrhi::FRESULT CreateLambdaSystem(const SystemDesc& desc, F&& update, ISystem** ppSystem) noexcept
    {
        *ppSystem = nullptr;
        try
        {
            *ppSystem = MAKE_RC_OBJ(detail::LambdaSystem<F>, desc, std::forward<F>(update));
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        return *ppSystem ? nvrhi::FS_OK : nvrhi::FE_OUT_OF_MEMORY;
    }
}

#if defined(DONUT_ECS_SCHEDULER_RESTORE_NONE)
#undef DONUT_ECS_SCHEDULER_RESTORE_NONE
#pragma pop_macro("None")
#endif
