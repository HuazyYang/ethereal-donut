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

// The domain core (aggregate docs/plans/2026-11-21-donut-ecs-e6.md, item E6.1; design 8.1, 8.2, 8.4, 8.5): the column
// views of a domain's gather and scatter (alignment over mixed archetypes, paging, versions, refusals), its lifetime
// callbacks (every source of events, their context, the commands they record), references between world, scheduler and
// domain, GPU queues, planned, dropped and failed steps, the published FixedStepState, parallel steps, captures and
// rewinds (what is in them, streams that are refused, engines that fail), equal engine indices at 1, 2 and 8 workers,
// and an allocation counter over steady frames. Internals are read through the private headers (src/engine/Ecs*Impl.h).

#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsObserverImpl.h"
#include "EcsSchedulerImpl.h"
#include "EcsSnapshot.h"
#include "EcsWorldImpl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test, and donut_engine, which links statically) is counted. The storage's
// own blocks, the command buffers' arenas included, come from nvrhi_core's allocator, which ecs::GetHeapAllocationCount
// counts.
namespace
{
    std::atomic<uint64_t> g_NewCount{ 0 };

    void* CountedAllocate(size_t size) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
        return std::malloc(size ? size : 1);
    }

    void* CountedAllocateAligned(size_t size, size_t alignment) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
        return _aligned_malloc(size ? size : 1, alignment);
#else
        return std::aligned_alloc(alignment, ((size ? size : 1) + alignment - 1) & ~(alignment - 1));
#endif
    }

    void CountedFreeAligned(void* memory) noexcept
    {
#if defined(_MSC_VER)
        _aligned_free(memory);
#else
        std::free(memory);
#endif
    }
}

void* operator new(size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new[](size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new(size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }
void operator delete(void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete[](void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete(void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }

// ------------------------------------------------------------------------------------------ test components

namespace sim_test
{
    using donut::engine::ComponentFlags;

    struct Pos { double x, y; };
    DONUT_ECS_COMPONENT(Pos, "d107da14-f045-4985-bf47-6bc6995d638a", DONUT_ECS_FIELDS(x, y));

    struct Vel { double x, y; };
    DONUT_ECS_COMPONENT(Vel, "2fa79ec5-1f8f-4e7f-a418-6933d1989b2c", DONUT_ECS_FIELDS(x, y));

    struct Mass { double value; };
    DONUT_ECS_COMPONENT(Mass, "8b495610-fc29-4584-938e-33446227f0aa", DONUT_ECS_FIELDS(value));

    // Like SimBodyRef: Relocatable with a construct hook.
    struct Body { uint32_t domain = ~0u; uint32_t index = ~0u; };
    DONUT_ECS_COMPONENT(Body, "8eab1413-d030-43f8-a0f6-865e2489344e", DONUT_ECS_FIELDS(domain, index));

    // Like SimTeleport: a consumed input whose enabled bit is the request.
    struct Push { double x, y; };
    DONUT_ECS_COMPONENT_EX(Push, "2ead8b48-e235-45bc-b5cb-53b233837c38", ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(x, y));

    struct Ball { };
    DONUT_ECS_TAG(Ball, "6583600d-314d-4f4a-899a-b2fd06092b5c");

    struct Blue { };
    DONUT_ECS_TAG(Blue, "70a90a13-3210-44b9-af69-bf3316c5dc97");

    struct Awake { };
    DONUT_ECS_TAG_EX(Awake, "b4f891ce-9b06-404a-a003-0eeb686222a3", ComponentFlags::Enableable);

    struct Red { uint32_t value; };
    DONUT_ECS_COMPONENT(Red, "f27552a1-982a-42a0-9a12-861a66b4cf8d", DONUT_ECS_FIELDS(value));

    struct Pad { uint64_t words[40]; };
    DONUT_ECS_COMPONENT(Pad, "3f5a9ee9-50ea-42e1-b8d2-54490ce43f12", DONUT_ECS_FIELDS());

    struct Group { uint32_t id; };
    DONUT_ECS_COMPONENT_EX(Group, "12491cd2-80e4-4dc2-8635-8227003028e0", ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(id));

    struct Span { double low, high; };
    DONUT_ECS_COMPONENT_EX(Span, "c000e7e3-36cb-41cb-800c-702c3626e441", ComponentFlags::Chunk, 0,
                           DONUT_ECS_FIELDS(low, high));

    struct Item { uint32_t value; };
    DONUT_ECS_BUFFER(Item, "54e8af53-24f4-4c9c-afbe-7528677c86db", 2, DONUT_ECS_FIELDS(value));

    struct Chosen { };
    DONUT_ECS_TAG_EX(Chosen, "f8b563d6-5ecf-45c2-b344-784bda700211", ComponentFlags::Sparse);

    struct Gauge { uint64_t value; };
    DONUT_ECS_COMPONENT_EX(Gauge, "a4de9035-8ddf-4549-a6a7-ad53177b0cfe", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(value));

    // Not Relocatable.
    struct Hooked
    {
        uint32_t value = 0;
        Hooked() noexcept = default;
        Hooked(const Hooked& other) noexcept : value(other.value) { }
        Hooked& operator=(const Hooked& other) noexcept { value = other.value; return *this; }
        ~Hooked() { }
    };
    DONUT_ECS_COMPONENT(Hooked, "6d39713e-ff33-492b-8ab4-f650d038d174", DONUT_ECS_FIELDS(value));
}

using namespace donut::engine;
using namespace sim_test;
namespace ecs = donut::engine::ecs;

namespace
{
    // Domain calls are noexcept, so a failed expectation in one is counted, and main checks the count after each case.
    std::atomic<int> g_Errors{ 0 };
    void Expect(bool condition, int line, const char* text) noexcept
    {
        if (condition)
            return;
        g_Errors.fetch_add(1);
        fprintf(stderr, "test_ecs_simulation.cpp:%d: expected %s\n", line, text);
    }
#define EXPECT(condition) Expect(bool(condition), __LINE__, #condition)

    ComponentAccess R(ComponentId id) { return ComponentAccess{ id, Access::Read, {} }; }
    ComponentAccess W(ComponentId id) { return ComponentAccess{ id, Access::Write, {} }; }
    ComponentAccess Filter(ComponentId id) { return ComponentAccess{ id, Access::Filter, {} }; }
    QueryTerm Term(ComponentId id, Access access, TermOp op = TermOp::All) { return QueryTerm{ id, op, access, {} }; }

    class MemoryWriter final : public nvrhi::ObjectImpl<IStateWriter>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MemoryWriter)
        NVRHI_IMPLEMENTS_INTERFACE(IStateWriter)
        NVRHI_END_INTERFACE_TABLE()

        std::vector<uint8_t> bytes;

        nvrhi::FRESULT Write(const void* data, uint64_t size) noexcept override
        {
            const uint8_t* begin = static_cast<const uint8_t*>(data);
            bytes.insert(bytes.end(), begin, begin + size);
            return nvrhi::FS_OK;
        }
    };

    class MemoryReader final : public nvrhi::ObjectImpl<IStateReader>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MemoryReader)
        NVRHI_IMPLEMENTS_INTERFACE(IStateReader)
        NVRHI_END_INTERFACE_TABLE()

        std::vector<uint8_t> bytes;
        size_t position = 0;

        nvrhi::FRESULT Read(uint64_t size, void* data) noexcept override
        {
            if (size > bytes.size() - position)
                return nvrhi::FE_INVALID_ARGS;
            std::memcpy(data, bytes.data() + position, size_t(size));
            position += size_t(size);
            return nvrhi::FS_OK;
        }
    };

    struct Env
    {
        nvrhi::AutoPtr<IWorld> world;
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<IScheduler> scheduler;
        ComponentId pos, vel, mass, body, push, ball, blue, awake, red, pad, group, span, item, chosen, gauge, hooked;
        ComponentId fixedStep;
        uint64_t frame = 0;
        double time = 0.0;

        explicit Env(uint32_t workers = 2)
        {
            WorldDesc worldDesc;
            worldDesc.debugName = "simulation test";
            worldDesc.validation = true;
            CHECK(CreateWorld(worldDesc, &world) == nvrhi::FS_OK);
            JobSystemDesc jobDesc;
            jobDesc.debugName = "simulation test";
            jobDesc.workerThreadCount = workers;
            CHECK(CreateJobSystem(jobDesc, &jobs) == nvrhi::FS_OK);
            CHECK(CreateScheduler(world.Get(), jobs.Get(), &scheduler) == nvrhi::FS_OK);
            pos = RegisterComponent<Pos>(world.Get());
            vel = RegisterComponent<Vel>(world.Get());
            mass = RegisterComponent<Mass>(world.Get());
            body = RegisterComponent<Body>(world.Get());
            push = RegisterComponent<Push>(world.Get());
            ball = RegisterComponent<Ball>(world.Get());
            blue = RegisterComponent<Blue>(world.Get());
            awake = RegisterComponent<Awake>(world.Get());
            red = RegisterComponent<Red>(world.Get());
            pad = RegisterComponent<Pad>(world.Get());
            group = RegisterComponent<Group>(world.Get());
            span = RegisterComponent<Span>(world.Get());
            item = RegisterComponent<BufferOf<Item>>(world.Get());
            chosen = RegisterComponent<Chosen>(world.Get());
            CHECK(world->RegisterResource(ComponentTraits<Gauge>::Desc(), nullptr, gauge) == nvrhi::FS_OK);
            hooked = RegisterComponent<Hooked>(world.Get());
            fixedStep = FindComponent<FixedStepState>(world.Get());
            CHECK(fixedStep != ComponentId::Invalid);
        }

        ecs::WorldImpl& Impl() const { return *ecs::GetWorldImpl(world.Get()); }
        uint64_t Reports() const { return Impl().GetValidationErrorCount(); }
        ecs::SchedulerCounters Counters() const
        {
            ecs::SchedulerCounters counters{};
            CHECK(ecs::GetSchedulerCounters(scheduler.Get(), counters));
            return counters;
        }
        uint32_t ObserverCount() const
        {
            return Impl().GetObserverRegistry() ? Impl().GetObserverRegistry()->GetObserverCount() : 0;
        }
        uint32_t SystemCount(Phase phase) const
        {
            return ecs::GetScheduleOrder(scheduler.Get(), phase, nullptr, 0);
        }
        std::string Diagnostics() const
        {
            nvrhi::string text;
            scheduler->GetDiagnostics(text);
            return text.c_str();
        }
        const FixedStepEntry& Step(SimulationDomainId id) const
        {
            return GetResource<FixedStepState>(world.Get())->domains[uint32_t(id)];
        }

        std::vector<Entity> Create(const std::vector<ComponentId>& components, uint32_t count,
                                   const std::vector<const void*>& values = {}) const
        {
            EntityCreateDesc desc{};
            desc.components = components.data();
            desc.values = values.empty() ? nullptr : values.data();
            desc.componentCount = uint32_t(components.size());
            desc.entityCount = count;
            std::vector<Entity> entities(count);
            CHECK(world->CreateEntities(desc, entities.data()) == nvrhi::FS_OK);
            return entities;
        }
        template <typename T> T* Edit(Entity entity) const
        {
            T* value = GetComponentForWrite<T>(world.Get(), entity);
            CHECK(value);
            return value;
        }
        template <typename T> const T* Get(Entity entity) const { return GetComponent<T>(world.Get(), entity); }

        // The version of the chunk column that holds the entity's component.
        uint64_t ColumnVersion(Entity entity, ComponentId component) const
        {
            const ecs::EntityRecord* record = Impl().FindAlive(entity);
            CHECK(record);
            const ecs::Archetype& archetype = Impl().GetArchetypeStorage(record->archetype);
            const uint32_t column = archetype.FindColumn(component);
            CHECK(column != ecs::c_NoColumn);
            return archetype.Versions(record->chunk)[column];
        }

        void SetTime(double delta)
        {
            time += delta;
            scheduler->SetFrameTime(FrameTime{ time, delta, frame++ });
        }
        void Frame(double delta, Phase first = Phase::BeginFrame, Phase last = Phase::EndFrame)
        {
            SetTime(delta);
            CHECK(scheduler->RunPhases(first, last) == nvrhi::FS_OK);
        }

        template <typename Function>
        SystemId Add(const char* name, Phase phase, const std::vector<ComponentAccess>& accesses, Function&& update,
                     const std::vector<const char*>& after = {}, const std::vector<const char*>& before = {},
                     SystemFlags flags = SystemFlags::None) const
        {
            SystemDesc desc{};
            desc.name = name;
            desc.accesses = accesses.data();
            desc.accessCount = uint32_t(accesses.size());
            desc.phase = phase;
            desc.runAfter = after.data();
            desc.runAfterCount = uint32_t(after.size());
            desc.runBefore = before.data();
            desc.runBeforeCount = uint32_t(before.size());
            desc.flags = flags;
            nvrhi::AutoPtr<ISystem> system;
            CHECK(CreateLambdaSystem(desc, std::forward<Function>(update), &system) == nvrhi::FS_OK);
            SystemId id = SystemId::Invalid;
            CHECK(scheduler->AddSystem(system.Get(), id) == nvrhi::FS_OK);
            return id;
        }

        std::vector<uint8_t> Save(nvrhi::FRESULT expected = nvrhi::FS_OK) const
        {
            nvrhi::AutoPtr<MemoryWriter> writer = MAKE_RC_OBJ_PTR(MemoryWriter);
            CHECK(scheduler->SaveSimulationState(writer.Get()) == expected);
            return writer->bytes;
        }
        nvrhi::FRESULT Load(const std::vector<uint8_t>& bytes, size_t* position = nullptr) const
        {
            nvrhi::AutoPtr<MemoryReader> reader = MAKE_RC_OBJ_PTR(MemoryReader);
            reader->bytes = bytes;
            const nvrhi::FRESULT result = scheduler->LoadSimulationState(reader.Get());
            if (position)
                *position = reader->position;
            return result;
        }
    };

    // A domain whose calls are the test's lambdas.
    class ScriptDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(ScriptDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        std::string name;
        std::vector<ComponentAccess> inputs;
        std::vector<ComponentAccess> outputs;
        std::vector<QueryTerm> terms;
        SimulationDomainDesc desc{};
        std::function<nvrhi::FRESULT(const EntityBatch&, ISimulationContext*)> added;
        std::function<nvrhi::FRESULT(const EntityBatch&, ISimulationContext*)> removed;
        std::function<nvrhi::FRESULT(ISimulationContext*)> gather;
        std::function<nvrhi::FRESULT(ISimulationContext*)> step;
        std::function<nvrhi::FRESULT(ISimulationContext*)> scatter;
        std::function<nvrhi::FRESULT(IStateWriter*)> capture;
        std::function<nvrhi::FRESULT(IStateReader*)> restore;
        nvrhi::AutoPtr<IQuery> heldQuery;      // holds the world, as a domain's own queries do
        uint32_t addedEntities = 0;
        uint32_t removedEntities = 0;
        uint32_t gathers = 0;
        uint32_t steps = 0;
        uint32_t scatters = 0;

        ScriptDomain(const char* domainName, double dt, uint32_t maxSteps) : name(domainName)
        {
            desc.fixedTimeStep = dt;
            desc.maxStepsPerFrame = maxSteps;
            desc.substeps = 1;
        }

        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override
        {
            retVal = desc;
            retVal.name = name.c_str();
            retVal.inputs = inputs.data();
            retVal.inputCount = uint32_t(inputs.size());
            retVal.outputs = outputs.data();
            retVal.outputCount = uint32_t(outputs.size());
            retVal.lifetime.terms = terms.data();
            retVal.lifetime.termCount = uint32_t(terms.size());
            return retVal;
        }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch& batch, ISimulationContext* context) noexcept override
        {
            addedEntities += batch.count;
            return added ? added(batch, context) : nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch& batch, ISimulationContext* context) noexcept override
        {
            removedEntities += batch.count;
            return removed ? removed(batch, context) : nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext* context) noexcept override
        {
            ++gathers;
            return gather ? gather(context) : nvrhi::FS_OK;
        }
        nvrhi::FRESULT Step(double dt, ISimulationContext* context) noexcept override
        {
            EXPECT(dt == desc.fixedTimeStep);
            ++steps;
            return step ? step(context) : nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept override
        {
            ++scatters;
            return scatter ? scatter(context) : nvrhi::FS_OK;
        }
        nvrhi::FRESULT CaptureState(IStateWriter* writer) noexcept override
        {
            return capture ? capture(writer) : nvrhi::FS_OK;
        }
        nvrhi::FRESULT RestoreState(IStateReader* reader) noexcept override
        {
            return restore ? restore(reader) : nvrhi::FS_OK;
        }
    };

    nvrhi::AutoPtr<ScriptDomain> MakeDomain(const char* name, double dt = 1.0, uint32_t maxSteps = 4)
    {
        return MAKE_RC_OBJ_PTR(ScriptDomain, name, dt, maxSteps);
    }

    SimulationDomainId AddDomain(Env& env, ISimulationDomain* domain, nvrhi::FRESULT expected = nvrhi::FS_OK)
    {
        SimulationDomainId id = SimulationDomainId(7777);
        CHECK(env.scheduler->AddSimulationDomain(domain, id) == expected);
        CHECK((id != SimulationDomainId::Invalid) == (expected == nvrhi::FS_OK));
        return id;
    }

    long ReferenceCount(nvrhi::IObject* object)
    {
        object->AddRef();
        return long(object->Release());
    }

    // Every view of a component, taken in pages.
    std::vector<ColumnView> Views(ISimulationContext* context, ComponentId component,
                                  uint64_t sinceVersion = c_AllVersions, uint32_t page = 3)
    {
        std::vector<ColumnView> views;
        std::vector<ColumnView> buffer(page);
        const uint32_t count = context->GetColumnCount(component);
        for (uint32_t first = 0; first < count; first += page)
        {
            const uint32_t written = context->GetColumns(component, first, sinceVersion, page, buffer.data());
            EXPECT(written <= page);
            views.insert(views.end(), buffer.begin(), buffer.begin() + std::min(written, page));
        }
        return views;
    }

    // The chunks of a query: what the views must line up with.
    struct Chunk
    {
        std::vector<Entity> entities;
    };
    std::vector<Chunk> Chunks(IQuery* query)
    {
        std::vector<Chunk> chunks;
        ChunkView view;
        for (uint32_t i = 0; i < query->GetChunkCount(); ++i)
            if (query->GetChunks(i, ChunkRequest{}, 1, &view) == 1)
                chunks.push_back(Chunk{ std::vector<Entity>(view.entities, view.entities + view.count) });
        return chunks;
    }
    std::vector<Entity> Flatten(const std::vector<Chunk>& chunks)
    {
        std::vector<Entity> entities;
        for (const Chunk& chunk : chunks)
            entities.insert(entities.end(), chunk.entities.begin(), chunk.entities.end());
        return entities;
    }

    bool Near(double a, double b) { return std::abs(a - b) < 1e-12; }

    // ------------------------------------------------------------------------------------------ a toy engine

    // Balls in dense arrays, addressed by the index that OnEntitiesAdded stores in Body: what an engine behind a domain
    // looks like. Inputs: Body (to map rows), Push (consumed: added to the velocity once, by its enabled bit) and Mass;
    // outputs: Body (Read), Pos and Vel.
    class BallDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(BallDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        struct State { double x, y, vx, vy, mass; uint64_t entity; };

        ComponentId pos, vel, mass, body, push;
        std::vector<ComponentAccess> inputs;
        std::vector<ComponentAccess> outputs;
        std::vector<QueryTerm> terms;
        std::vector<State> states;
        std::vector<uint32_t> freeIndices;
        std::vector<uint64_t> log;             // entity << 32 | index of every event, additions and removals
        nvrhi::AutoPtr<IQuery> pushes;
        double dt = 0.25;
        uint32_t maxSteps = 4;
        SimulationDomainFlags flags = SimulationDomainFlags::Deterministic;
        bool keepLog = true;

        explicit BallDomain(const Env& env)
            : pos(env.pos), vel(env.vel), mass(env.mass), body(env.body), push(env.push)
        {
            inputs = { R(body), W(push), R(mass) };
            outputs = { W(pos), W(vel), R(body) };
            terms = { Term(env.ball, Access::Filter), Term(body, Access::Write), Term(pos, Access::Read) };
            const QueryTerm own[2] = { Term(body, Access::Read), Term(push, Access::Write) };
            CHECK(env.world->CreateQuery(QueryDesc{ own, 2, 0, "pushes" }, &pushes) == nvrhi::FS_OK);
        }

        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override
        {
            retVal = SimulationDomainDesc{};
            retVal.name = "balls";
            retVal.inputs = inputs.data();
            retVal.inputCount = uint32_t(inputs.size());
            retVal.outputs = outputs.data();
            retVal.outputCount = uint32_t(outputs.size());
            retVal.lifetime.terms = terms.data();
            retVal.lifetime.termCount = uint32_t(terms.size());
            retVal.substeps = 1;
            retVal.maxStepsPerFrame = maxSteps;
            retVal.fixedTimeStep = dt;
            retVal.flags = flags;
            return retVal;
        }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch& batch, ISimulationContext* context) noexcept override
        {
            Body* bodies = static_cast<Body*>(batch.columns[0]);
            const Pos* positions = static_cast<const Pos*>(batch.columns[1]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                uint32_t index = uint32_t(states.size());
                if (!freeIndices.empty())
                {
                    index = freeIndices.back();
                    freeIndices.pop_back();
                }
                else
                    states.emplace_back();
                states[index] = State{ positions[i].x, positions[i].y, 0.0, 0.0, 1.0, uint64_t(batch.entities[i]) };
                bodies[i].domain = uint32_t(context->GetDomainId());
                bodies[i].index = index;
                if (keepLog)
                    log.push_back((uint64_t(GetEntityIndex(batch.entities[i])) << 32) | index);
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch& batch, ISimulationContext*) noexcept override
        {
            Body* bodies = static_cast<Body*>(batch.columns[0]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                const uint32_t index = bodies[i].index;
                EXPECT(index < states.size() && states[index].entity == uint64_t(batch.entities[i]));
                states[index] = State{};
                freeIndices.push_back(index);
                bodies[i] = Body{};
                if (keepLog)
                    log.push_back((uint64_t(GetEntityIndex(batch.entities[i])) << 32) | index | 0x80000000u);
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext* context) noexcept override
        {
            ColumnView bodies[4];
            ColumnView masses[4];
            const uint32_t count = context->GetColumnCount(body);
            for (uint32_t first = 0; first < count; first += 4)
            {
                const uint32_t n = context->GetColumns(body, first, c_AllVersions, 4, bodies);
                const uint32_t m = context->GetColumns(mass, first, c_AllVersions, 4, masses);
                uint32_t next = 0;
                for (uint32_t v = 0; v < n; ++v)
                {
                    while (next < m && masses[next].chunkIndex < bodies[v].chunkIndex)
                        ++next;
                    if (next >= m || masses[next].chunkIndex != bodies[v].chunkIndex)
                        continue;
                    for (uint32_t row = 0; row < bodies[v].count; ++row)
                        states[static_cast<const Body*>(bodies[v].base)[row].index].mass =
                            static_cast<const Mass*>(masses[next].base)[row].value;
                }
            }
            // The requests, by their enabled bits: a query of the domain's own.
            ISystemContext* system = context->GetSystemContext();
            ChunkView view;
            for (uint32_t i = 0; i < pushes->GetChunkCount(); ++i)
            {
                if (pushes->GetChunks(i, MakeChunkRequest(system), 1, &view) != 1)
                    continue;
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    if (!view.IsRowEnabled(1, row))
                        continue;
                    const Body& bodyRef = static_cast<const Body*>(view.columns[0])[row];
                    const Push& request = static_cast<const Push*>(view.columns[1])[row];
                    if (bodyRef.index < states.size())
                    {
                        states[bodyRef.index].vx += request.x;
                        states[bodyRef.index].vy += request.y;
                    }
                    EXPECT(context->GetWorld()->SetEnabled(view.entities[row], push, false) == nvrhi::FS_OK);
                }
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT Step(double step, ISimulationContext*) noexcept override
        {
            for (State& s : states)
            {
                s.vy -= 8.0 * step * s.mass;
                s.x += s.vx * step;
                s.y += s.vy * step;
                if (s.y < 0.0)
                {
                    s.y = -s.y;
                    s.vy = -s.vy * 0.75;
                }
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept override
        {
            ColumnView bodies[4];
            ColumnView positions[4];
            ColumnView velocities[4];
            const uint32_t count = context->GetColumnCount(body);
            for (uint32_t first = 0; first < count; first += 4)
            {
                const uint32_t n = context->GetColumns(body, first, c_AllVersions, 4, bodies);
                const uint32_t p = context->GetColumns(pos, first, c_AllVersions, 4, positions);
                const uint32_t v = context->GetColumns(vel, first, c_AllVersions, 4, velocities);
                EXPECT(p == n);
                uint32_t next = 0;
                for (uint32_t c = 0; c < n; ++c)
                {
                    EXPECT(positions[c].chunkIndex == bodies[c].chunkIndex && positions[c].count == bodies[c].count);
                    while (next < v && velocities[next].chunkIndex < bodies[c].chunkIndex)
                        ++next;
                    const bool hasVelocity = next < v && velocities[next].chunkIndex == bodies[c].chunkIndex;
                    for (uint32_t row = 0; row < bodies[c].count; ++row)
                    {
                        const State& s = states[static_cast<const Body*>(bodies[c].base)[row].index];
                        static_cast<Pos*>(positions[c].base)[row] = Pos{ s.x, s.y };
                        if (hasVelocity)
                            static_cast<Vel*>(velocities[next].base)[row] = Vel{ s.vx, s.vy };
                    }
                }
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT CaptureState(IStateWriter* writer) noexcept override
        {
            const uint64_t sizes[2] = { states.size(), freeIndices.size() };
            nvrhi::FRESULT result = writer->Write(sizes, sizeof(sizes));
            if (NVRHI_SUCCEEDED(result))
                result = writer->Write(states.data(), states.size() * sizeof(State));
            if (NVRHI_SUCCEEDED(result))
                result = writer->Write(freeIndices.data(), freeIndices.size() * sizeof(uint32_t));
            return result;
        }
        nvrhi::FRESULT RestoreState(IStateReader* reader) noexcept override
        {
            uint64_t sizes[2] = {};
            nvrhi::FRESULT result = reader->Read(sizeof(sizes), sizes);
            if (NVRHI_FAILED(result))
                return result;
            states.resize(size_t(sizes[0]));
            freeIndices.resize(size_t(sizes[1]));
            result = reader->Read(states.size() * sizeof(State), states.data());
            return NVRHI_SUCCEEDED(result) ? reader->Read(freeIndices.size() * sizeof(uint32_t), freeIndices.data())
                                           : result;
        }
    };

    // Everything a rewind must bring back, as bytes: the rows of the balls (values and enabled bits), the engine, the
    // published time.
    std::vector<uint8_t> BallImage(Env& env, BallDomain* domain, SimulationDomainId id)
    {
        nvrhi::AutoPtr<MemoryWriter> out = MAKE_RC_OBJ_PTR(MemoryWriter);
        const QueryTerm terms[1] = { Term(env.ball, Access::Filter) };
        nvrhi::AutoPtr<IQuery> query;
        CHECK(env.world->CreateQuery(QueryDesc{ terms, 1, 0, "image" }, &query) == nvrhi::FS_OK);
        for (Entity entity : Flatten(Chunks(query.Get())))
        {
            out->Write(&entity, sizeof(entity));
            const std::pair<ComponentId, size_t> columns[4] = {
                { env.pos, sizeof(Pos) }, { env.vel, sizeof(Vel) }, { env.push, sizeof(Push) },
                { env.body, sizeof(Body) } };
            for (const auto& [component, size] : columns)
            {
                const bool has = env.world->HasComponent(entity, component);
                out->Write(&has, 1);
                if (has)
                    out->Write(env.world->GetComponent(entity, component), size);
            }
            const bool enabled = env.world->HasComponent(entity, env.push) && env.world->IsEnabled(entity, env.push);
            out->Write(&enabled, 1);
        }
        CHECK(domain->CaptureState(out.Get()) == nvrhi::FS_OK);
        out->Write(&env.Step(id), sizeof(FixedStepEntry));
        return out->bytes;
    }
}

// ---------------------------------------------------------------------------------------------------- views

// Views of several components over mixed archetypes: each names the chunk of the lifetime query it belongs to, a chunk
// without the column has none, and rows of views with one chunkIndex are the same entities.
static void test_views_align()
{
    Env env(4);
    uint32_t next = 0;
    auto make = [&](const std::vector<ComponentId>& components, uint32_t count, uint32_t groupId = 0)
    {
        std::vector<const void*> values(components.size(), nullptr);
        const std::vector<Group> groups(count, Group{ groupId });
        for (size_t i = 0; i < components.size(); ++i)
            if (components[i] == env.group)
                values[i] = groups.data();
        for (Entity entity : env.Create(components, count, values))
        {
            ++next;
            const double key = double(GetEntityIndex(entity));
            if (env.world->HasComponent(entity, env.pos))
                *env.Edit<Pos>(entity) = Pos{ key, 1.0 };
            if (env.world->HasComponent(entity, env.vel))
                *env.Edit<Vel>(entity) = Vel{ -key, 2.0 };
            if (env.world->HasComponent(entity, env.body))
                env.Edit<Body>(entity)->index = GetEntityIndex(entity);
        }
    };
    make({ env.ball, env.body, env.pos, env.vel }, 900);
    make({ env.ball, env.body, env.pos, env.vel, env.blue }, 10);
    make({ env.ball, env.body, env.pos, env.pad }, 500);        // no Vel
    make({ env.ball, env.body, env.vel }, 400);                 // no Pos
    make({ env.body, env.pos, env.vel }, 50);                   // not in the domain
    make({ env.ball, env.body, env.pos, env.vel, env.group }, 30, 1);
    make({ env.ball, env.body, env.pos, env.vel, env.group }, 30, 2);

    Query<With<Ball>> reference(env.world.Get(), "reference");
    const std::vector<Chunk> chunks = Chunks(reference.Get());
    CHECK(chunks.size() > 8);

    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("views");
    domain->terms = { Term(env.ball, Access::Filter), Term(env.body, Access::Write) };
    domain->inputs = { R(env.pos), R(env.vel), R(env.body) };
    domain->outputs = { W(env.pos), W(env.vel), R(env.body) };
    uint32_t checkedRows = 0;
    auto check = [&](ISimulationContext* context)
    {
        auto lines = [&](ComponentId component, uint32_t stride)
        {
            EXPECT(context->GetColumnCount(component) == chunks.size());
            const std::vector<ColumnView> views = Views(context, component);
            size_t v = 0;
            for (uint32_t c = 0; c < chunks.size(); ++c)
            {
                if (!env.world->HasComponent(chunks[c].entities[0], component))
                    continue;
                if (v >= views.size())
                {
                    EXPECT(false);
                    break;
                }
                const ColumnView& view = views[v++];
                EXPECT(view.chunkIndex == c && view.count == chunks[c].entities.size() && view.stride == stride &&
                       view.component == component);
                EXPECT(view.base == env.world->GetComponent(chunks[c].entities[0], component));
                for (uint32_t row = 0; row < view.count && row < chunks[c].entities.size(); ++row)
                    EXPECT(static_cast<uint8_t*>(view.base) + size_t(row) * view.stride ==
                           env.world->GetComponent(chunks[c].entities[row], component));
            }
            EXPECT(v == views.size());
            return views;
        };
        const std::vector<ColumnView> positions = lines(env.pos, sizeof(Pos));
        const std::vector<ColumnView> velocities = lines(env.vel, sizeof(Vel));
        const std::vector<ColumnView> bodies = lines(env.body, sizeof(Body));
        EXPECT(bodies.size() == chunks.size() && positions.size() < bodies.size() && velocities.size() < bodies.size());
        // Rows by chunkIndex: the same entity in each.
        for (const ColumnView& b : bodies)
        {
            for (const ColumnView& p : positions)
                for (uint32_t row = 0; p.chunkIndex == b.chunkIndex && row < b.count; ++row, ++checkedRows)
                    EXPECT(static_cast<const Pos*>(p.base)[row].x == double(static_cast<const Body*>(b.base)[row].index));
            for (const ColumnView& v : velocities)
                for (uint32_t row = 0; v.chunkIndex == b.chunkIndex && row < b.count; ++row, ++checkedRows)
                    EXPECT(static_cast<const Vel*>(v.base)[row].x ==
                           -double(static_cast<const Body*>(b.base)[row].index));
        }
        return nvrhi::FS_OK;
    };
    domain->gather = check;
    domain->scatter = check;
    AddDomain(env, domain.Get());
    env.Frame(1.0);
    CHECK(domain->gathers == 1 && domain->scatters == 1);
    CHECK(checkedRows == 2 * (900 * 2 + 10 * 2 + 500 + 400 + 60 * 2));
    CHECK(env.Reports() == 0 && env.scheduler->GetAmbiguityCount() == 0);
}

static void test_paging()
{
    Env env;
    env.Create({ env.ball, env.pos, env.body }, 3000);
    env.Create({ env.ball, env.body }, 1500);                   // chunks without Pos between those with it
    env.Create({ env.ball, env.pos, env.body, env.blue }, 2000);
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("paging");
    domain->terms = { Term(env.ball, Access::Filter) };
    domain->inputs = { R(env.pos) };
    domain->gather = [&](ISimulationContext* context)
    {
        const uint32_t count = context->GetColumnCount(env.pos);
        const std::vector<ColumnView> all = Views(context, env.pos, c_AllVersions, 64);
        EXPECT(count >= 8 && all.size() < count && all.size() >= 6);
        for (uint32_t page : { 1u, 2u, 3u, 5u, 7u })
        {
            const std::vector<ColumnView> paged = Views(context, env.pos, c_AllVersions, page);
            EXPECT(paged.size() == all.size() &&
                   std::memcmp(paged.data(), all.data(), all.size() * sizeof(ColumnView)) == 0);
        }
        // A page of one chunk gives that chunk's view or none.
        ColumnView view;
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t written = context->GetColumns(env.pos, i, c_AllVersions, 1, &view);
            EXPECT(written == 0 || view.chunkIndex == i);
            uint32_t expected = 0;
            for (const ColumnView& v : all)
                expected += v.chunkIndex == i ? 1 : 0;
            EXPECT(written == expected);
        }
        EXPECT(context->GetColumns(env.pos, count, c_AllVersions, 1, &view) == 0);
        EXPECT(context->GetColumns(env.pos, 0xffffffffu, c_AllVersions, 1, &view) == 0);
        EXPECT(context->GetColumns(env.pos, 0, c_AllVersions, 0, &view) == 0);
        EXPECT(context->GetColumns(env.pos, 0, c_AllVersions, 4, nullptr) == 0);
        return nvrhi::FS_OK;
    };
    AddDomain(env, domain.Get());
    env.Frame(0.0);
    CHECK(domain->gathers == 1 && env.Reports() == 0);
}

static void test_change_versions()
{
    Env env;
    const std::vector<Entity> entities = env.Create({ env.ball, env.body, env.pos, env.mass, env.push }, 1000);
    Query<With<Ball>> reference(env.world.Get(), "reference");
    const std::vector<Chunk> chunks = Chunks(reference.Get());
    CHECK(chunks.size() >= 3);
    const uint32_t chunkCount = uint32_t(chunks.size());

    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("versions");
    domain->terms = { Term(env.ball, Access::Filter) };
    domain->inputs = { R(env.mass), W(env.push), R(env.body) };
    domain->outputs = { W(env.pos), R(env.body) };
    uint64_t sinceGather = 0;
    uint64_t gatherVersion = 0;
    uint64_t scatterVersion = 0;
    std::vector<ColumnView> masses;
    std::vector<ColumnView> pushes;
    bool takePushes = true;
    uint32_t scatterFirst = 0;
    uint32_t scatterCount = 0;
    domain->gather = [&](ISimulationContext* context)
    {
        sinceGather = context->GetLastGatherVersion();
        gatherVersion = context->GetSystemContext()->GetVersion();
        masses = Views(context, env.mass, sinceGather);
        pushes = takePushes ? Views(context, env.push, sinceGather) : std::vector<ColumnView>();
        // Without a filter every chunk comes, whatever changed, and reading stamps nothing.
        EXPECT(Views(context, env.mass).size() == chunkCount && Views(context, env.body).size() == chunkCount);
        return nvrhi::FS_OK;
    };
    domain->scatter = [&](ISimulationContext* context)
    {
        scatterVersion = context->GetSystemContext()->GetVersion();
        ColumnView view;
        for (uint32_t i = scatterFirst; i < scatterFirst + scatterCount; ++i)
        {
            EXPECT(context->GetColumns(env.pos, i, c_AllVersions, 1, &view) == 1 && view.version == scatterVersion);
            EXPECT(context->GetColumns(env.body, i, c_AllVersions, 1, &view) == 1 && view.version < scatterVersion);
        }
        // A chunk that the filter leaves out is not stamped.
        for (uint32_t i = scatterFirst + scatterCount; i < chunkCount; ++i)
            EXPECT(context->GetColumns(env.pos, i, scatterVersion, 1, &view) == 0);
        return nvrhi::FS_OK;
    };
    Query<Changed<Pos>> changedPos(env.world.Get(), "changed pos");
    uint32_t changed = 0;
    env.Add("watch", Phase::TransformPropagation, { R(env.pos) }, [&](ISystemContext* context)
    {
        changed = 0;
        changedPos.ForEachChunk(context, [&](const ChunkView&, const Pos*) { ++changed; });
    });
    AddDomain(env, domain.Get());
    auto versions = [&](ComponentId component)
    {
        std::vector<uint64_t> out;
        for (const Chunk& chunk : chunks)
            out.push_back(env.ColumnVersion(chunk.entities[0], component));
        return out;
    };

    // Frame 1: everything is new; the consumed input is stamped, the read ones are not.
    const std::vector<uint64_t> massBefore = versions(env.mass);
    const std::vector<uint64_t> bodyBefore = versions(env.body);
    env.Frame(0.0);
    CHECK(sinceGather == c_AllVersions && masses.size() == chunkCount && pushes.size() == chunkCount);
    for (const ColumnView& view : pushes)
        CHECK(view.version == gatherVersion);
    for (const ColumnView& view : masses)
        CHECK(view.version == massBefore[view.chunkIndex] && view.version < gatherVersion);
    CHECK(versions(env.push) == std::vector<uint64_t>(chunkCount, gatherVersion));
    CHECK(versions(env.mass) == massBefore && versions(env.body) == bodyBefore);
    CHECK(changed == chunkCount);   // new rows

    // Frame 2: nothing changed. The domain does not see its own stamps.
    const uint64_t firstGather = gatherVersion;
    env.Frame(0.0);
    CHECK(sinceGather == firstGather && masses.empty() && pushes.empty() && changed == 0);
    CHECK(versions(env.push) == std::vector<uint64_t>(chunkCount, firstGather));   // nothing handed out, nothing stamped

    // Frame 3: one chunk of Mass, another of Push changed outside.
    *env.Edit<Mass>(chunks[1].entities[3]) = Mass{ 5.0 };
    *env.Edit<Push>(chunks[2].entities[0]) = Push{ 1.0, 1.0 };
    const std::vector<uint64_t> pushBefore = versions(env.push);
    env.Frame(0.0);
    CHECK(masses.size() == 1 && masses[0].chunkIndex == 1 && masses[0].version == versions(env.mass)[1]);
    CHECK(pushes.size() == 1 && pushes[0].chunkIndex == 2 && pushes[0].version == gatherVersion);
    CHECK(versions(env.push)[2] == gatherVersion && versions(env.push)[0] == pushBefore[0]);

    // Frames 4 and 5: the output is stamped chunk by chunk as it is handed out.
    scatterFirst = 1;
    scatterCount = 1;
    const std::vector<uint64_t> posBefore = versions(env.pos);
    env.Frame(0.0);
    CHECK(changed == 1 && versions(env.pos)[1] == scatterVersion && versions(env.pos)[0] == posBefore[0] &&
          versions(env.pos)[2] == posBefore[2]);
    CHECK(versions(env.body) == bodyBefore);   // an output declared Read
    scatterCount = 0;
    env.Frame(0.0);
    CHECK(changed == 0);
    CHECK(env.Reports() == 0);
}

static void test_refused_views()
{
    Env env;
    const std::vector<ComponentId> all = { env.ball, env.body, env.pos, env.vel, env.mass, env.hooked, env.awake,
                                           env.item, env.span, env.group, env.push };
    const std::vector<Entity> entities = env.Create(all, 20);
    for (Entity entity : entities)
        CHECK(env.world->AddComponent(entity, env.chosen, nullptr) == nvrhi::FS_OK);

    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("refused");
    domain->terms = { Term(env.ball, Access::Filter) };
    domain->inputs = { R(env.pos), R(env.body), W(env.push), R(env.hooked), R(env.ball), R(env.awake), R(env.item),
                       R(env.group), R(env.span), R(env.chosen), R(env.gauge), Filter(env.vel) };
    domain->outputs = { W(env.mass), R(env.body), W(env.hooked), W(env.span) };
    auto refused = [&](ISimulationContext* context, ComponentId component, uint64_t reports)
    {
        ColumnView view{};
        view.count = 77;
        const uint64_t before = env.Reports();
        EXPECT(context->GetColumnCount(component) == 0);
        EXPECT(env.Reports() == before + reports);
        EXPECT(context->GetColumns(component, 0, c_AllVersions, 1, &view) == 0);
        EXPECT(env.Reports() == before + 2 * reports && view.count == 77);
    };
    domain->gather = [&](ISimulationContext* context)
    {
        const uint64_t before = env.Reports();
        EXPECT(Views(context, env.pos).size() == 1 && Views(context, env.push).size() == 1);
        EXPECT(Views(context, env.body).size() == 1);   // a construct hook is allowed
        EXPECT(env.Reports() == before);
        for (ComponentId component : { env.mass, env.hooked, env.ball, env.awake, env.item, env.group, env.span,
                                       env.chosen, env.gauge, env.vel, env.red, ComponentId(5000),
                                       ComponentId::Invalid })
            refused(context, component, 1);
        return nvrhi::FS_OK;
    };
    domain->step = [&](ISimulationContext* context)
    {
        // Declared or not: no views in Step, and no report.
        refused(context, env.pos, 0);
        refused(context, env.mass, 0);
        refused(context, env.red, 0);
        return nvrhi::FS_OK;
    };
    domain->scatter = [&](ISimulationContext* context)
    {
        EXPECT(Views(context, env.mass).size() == 1 && Views(context, env.body).size() == 1);
        for (ComponentId component : { env.pos, env.push, env.hooked, env.span, env.red })
            refused(context, component, 1);
        return nvrhi::FS_OK;
    };
    uint32_t callbacks = 0;
    domain->added = [&](const EntityBatch&, ISimulationContext* context)
    {
        ++callbacks;
        refused(context, env.pos, 0);
        refused(context, env.red, 0);
        return nvrhi::FS_OK;
    };
    domain->removed = domain->added;
    AddDomain(env, domain.Get());
    env.Frame(1.0);
    CHECK(domain->gathers == 1 && domain->steps == 1 && domain->scatters == 1);
    const uint64_t reports = env.Reports();
    CHECK(env.world->DestroyEntities(entities.data(), 1) == nvrhi::FS_OK);
    CHECK(callbacks == 2 && env.Reports() == reports);

    // A domain without lifetime terms has no entities, so it has no views, whatever it declares; nothing is reported.
    Env other;
    other.Create({ other.ball, other.pos }, 5);
    nvrhi::AutoPtr<ScriptDomain> none = MakeDomain("none");
    none->inputs = { R(other.pos) };
    none->gather = [&](ISimulationContext* context)
    {
        ColumnView view;
        EXPECT(context->GetColumnCount(other.pos) == 0 && context->GetColumns(other.pos, 0, 0, 1, &view) == 0);
        return nvrhi::FS_OK;
    };
    AddDomain(other, none.Get());
    other.Frame(0.0);
    CHECK(none->gathers == 1 && none->addedEntities == 0 && other.Reports() == 0 && other.ObserverCount() == 0);
}

// A domain's own queries, through the running system's context: checked against the accesses of that call.
static void test_own_query()
{
    for (uint32_t workers : { 1u, 4u })
    {
        Env env(workers);
        env.Create({ env.ball, env.pos, env.mass, env.vel }, 300);
        Query<Read<Pos>, With<Ball>> readPos(env.world.Get(), "read pos");
        Query<Write<Pos>, With<Ball>> writePos(env.world.Get(), "write pos");
        Query<Read<Vel>, With<Ball>> readVel(env.world.Get(), "read vel");
        Query<Write<Vel>, With<Ball>> writeVel(env.world.Get(), "write vel");
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("own");
        domain->terms = { Term(env.ball, Access::Filter) };
        domain->inputs = { R(env.pos) };
        domain->outputs = { W(env.vel) };
        // The number of reports that a query makes through the running system's context.
        auto run = [&](ISimulationContext* context, const auto& query)
        {
            const uint64_t before = env.Reports();
            uint32_t seen = 0;
            ChunkView view;
            for (uint32_t i = 0; i < query.Get()->GetChunkCount(); ++i)
                if (query.Get()->GetChunks(i, MakeChunkRequest(context->GetSystemContext()), 1, &view) == 1)
                    seen += view.count;
            EXPECT(seen == 300);
            return env.Reports() - before;
        };
        uint64_t versions[3] = {};
        domain->gather = [&](ISimulationContext* context)
        {
            versions[0] = context->GetSystemContext()->GetVersion();
            EXPECT(run(context, readPos) == 0 && run(context, writePos) == 1 && run(context, readVel) == 1);
            return nvrhi::FS_OK;
        };
        domain->step = [&](ISimulationContext* context)
        {
            versions[1] = context->GetSystemContext()->GetVersion();
            EXPECT(run(context, readPos) == 1 && run(context, readVel) == 1);   // a step declares nothing
            return nvrhi::FS_OK;
        };
        domain->scatter = [&](ISimulationContext* context)
        {
            versions[2] = context->GetSystemContext()->GetVersion();
            EXPECT(run(context, writeVel) == 0 && run(context, readVel) == 0 && run(context, readPos) == 1);
            return nvrhi::FS_OK;
        };
        bool nullInCallback = false;
        domain->added = [&](const EntityBatch&, ISimulationContext* context)
        {
            nullInCallback = context->GetSystemContext() == nullptr;
            return nvrhi::FS_OK;
        };
        AddDomain(env, domain.Get());
        CHECK(nullInCallback);
        env.Frame(1.0);
        CHECK(versions[0] != 0 && versions[0] < versions[1] && versions[1] < versions[2]);
        // The own query's writes carry the system's version.
        Query<With<Ball>> balls(env.world.Get(), "balls");
        CHECK(env.ColumnVersion(Chunks(balls.Get())[0].entities[0], env.vel) == versions[2]);
        CHECK(env.Reports() == 5);
    }
}

// ------------------------------------------------------------------------------------------------- lifetime

namespace
{
    struct Event
    {
        bool added;
        uint32_t archetype;
        std::vector<Entity> entities;
    };

    // Records the batches and checks their columns against the world: Body (Write), Vel (Optional, null where absent),
    // Blue (Optional tag, always null). It stores an index in Body.
    struct Recorder
    {
        Env& env;
        nvrhi::AutoPtr<ScriptDomain> domain;
        std::vector<Event> events;
        uint32_t nextIndex = 100;
        std::map<Entity, uint32_t> indices;
        std::vector<std::string>* trace = nullptr;

        Recorder(Env& environment, const char* name) : env(environment), domain(MakeDomain(name))
        {
            domain->terms = { Term(env.ball, Access::Filter), Term(env.body, Access::Write),
                              Term(env.vel, Access::Read, TermOp::Optional),
                              Term(env.blue, Access::Read, TermOp::Optional) };
            domain->added = [this](const EntityBatch& batch, ISimulationContext* context)
            {
                Check(batch, context);
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    EXPECT(indices.count(batch.entities[i]) == 0);   // never twice
                    Body& body = static_cast<Body*>(batch.columns[0])[i];
                    EXPECT(body.index == ~0u);
                    body.index = nextIndex++;
                    indices[batch.entities[i]] = body.index;
                }
                events.push_back(Event{ true, batch.archetypeIndex, Entities(batch) });
                if (trace)
                    trace->push_back("added " + std::to_string(batch.count));
                return nvrhi::FS_OK;
            };
            domain->removed = [this](const EntityBatch& batch, ISimulationContext* context)
            {
                Check(batch, context);
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    const auto found = indices.find(batch.entities[i]);
                    EXPECT(found != indices.end());
                    // The row is still there with its values.
                    EXPECT(env.world->IsAlive(batch.entities[i]) && found != indices.end() &&
                           static_cast<const Body*>(batch.columns[0])[i].index == found->second);
                    static_cast<Body*>(batch.columns[0])[i].index = ~0u;
                    indices.erase(batch.entities[i]);
                }
                events.push_back(Event{ false, batch.archetypeIndex, Entities(batch) });
                if (trace)
                    trace->push_back("removed " + std::to_string(batch.count));
                return nvrhi::FS_OK;
            };
        }
        static std::vector<Entity> Entities(const EntityBatch& batch)
        {
            return std::vector<Entity>(batch.entities, batch.entities + batch.count);
        }
        void Check(const EntityBatch& batch, ISimulationContext* context)
        {
            EXPECT(batch.count != 0 && context && context->GetSystemContext() == nullptr);
            const ecs::EntityRecord* record = env.Impl().FindAlive(batch.entities[0]);
            EXPECT(record && record->archetype == batch.archetypeIndex);
            EXPECT(batch.columns[0] == env.world->GetComponent(batch.entities[0], env.body));
            EXPECT(batch.columns[1] == (env.world->HasComponent(batch.entities[0], env.vel)
                                        ? env.world->GetComponent(batch.entities[0], env.vel) : nullptr));
            EXPECT(batch.columns[2] == nullptr);
            // What a callback may write is stamped.
            EXPECT(env.ColumnVersion(batch.entities[0], env.body) == env.world->GetVersion());
        }
        std::vector<Entity> Flat(size_t first = 0) const
        {
            std::vector<Entity> out;
            for (size_t i = first; i < events.size(); ++i)
                out.insert(out.end(), events[i].entities.begin(), events[i].entities.end());
            return out;
        }
    };
}

static void test_lifetime_batches()
{
    Env env;
    Query<With<Ball>> balls(env.world.Get(), "balls");
    env.Create({ env.ball, env.body, env.vel }, 900);
    env.Create({ env.ball, env.body, env.blue }, 10);
    env.Create({ env.body }, 7);
    const uint32_t existing = uint32_t(Flatten(Chunks(balls.Get())).size());
    CHECK(existing == 910);
    // Time passes before the domain comes: what is delivered then must be stamped with a newer version.
    env.Frame(0.0);
    const uint64_t systemVersionBefore = env.world->GetVersion();

    Recorder recorder(env, "recorder");
    const SimulationDomainId id = AddDomain(env, recorder.domain.Get());
    CHECK(env.ObserverCount() == 2 && env.world->GetVersion() == systemVersionBefore);

    // Entities that match already: one batch per chunk, in chunk order.
    std::vector<Chunk> chunks = Chunks(balls.Get());
    CHECK(chunks.size() >= 3 && recorder.events.size() == chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i)
        CHECK(recorder.events[i].added && recorder.events[i].entities == chunks[i].entities);
    for (const auto& [entity, index] : recorder.indices)
        CHECK(env.Get<Body>(entity)->index == index);

    // A batch creation: runs of rows.
    recorder.events.clear();
    const std::vector<Entity> created = env.Create({ env.ball, env.body, env.vel, env.pos }, 1000);
    CHECK(recorder.events.size() >= 2 && recorder.Flat() == created);
    env.Create({ env.body, env.pos }, 5);
    CHECK(recorder.Flat() == created);

    // Matching by an added component, unmatching by a removed one; a move that keeps the match is no event.
    recorder.events.clear();
    const Entity plain = env.Create({ env.body, env.pos }, 1)[0];
    CHECK(recorder.events.empty());
    CHECK(env.world->AddComponent(plain, env.ball, nullptr) == nvrhi::FS_OK);
    CHECK(recorder.events.size() == 1 && recorder.events[0].added && recorder.Flat() == std::vector<Entity>{ plain });
    CHECK(env.world->AddComponent(plain, env.red, nullptr) == nvrhi::FS_OK);
    CHECK(recorder.events.size() == 1);
    CHECK(env.world->RemoveComponent(plain, env.ball) == nvrhi::FS_OK);
    CHECK(recorder.events.size() == 2 && !recorder.events[1].added && env.world->IsAlive(plain));

    // Destruction: one batch per entity, in the list's order, before the rows go.
    recorder.events.clear();
    const Entity doomed[3] = { created[10], created[500], created[11] };
    CHECK(env.world->DestroyEntities(doomed, 3) == nvrhi::FS_OK);
    CHECK(recorder.events.size() == 3 && recorder.Flat() == std::vector<Entity>(doomed, doomed + 3));

    // Commands of a system: at the sync point after PostSimulation, not before.
    recorder.events.clear();
    size_t eventsInScatter = 1000;
    recorder.domain->scatter = [&](ISimulationContext*)
    {
        eventsInScatter = recorder.events.size();
        return nvrhi::FS_OK;
    };
    bool record = true;
    env.Add("commands", Phase::PreSimulation, {}, [&](ISystemContext* context)
    {
        if (!record)
            return;
        const ComponentId list[2] = { env.ball, env.body };
        EntityCreateDesc desc{};
        desc.components = list;
        desc.componentCount = 2;
        desc.entityCount = 5;
        Entity placeholders[5];
        context->GetCommandBuffer()->CreateEntities(desc, 0, placeholders);
        context->GetCommandBuffer()->DestroyEntity(created[20], 1);
        context->GetCommandBuffer()->RemoveComponent(created[21], env.ball, 2);
    });
    env.Frame(0.0);
    record = false;
    CHECK(eventsInScatter == 0 && recorder.events.size() == 3);
    CHECK(recorder.events[0].added && recorder.events[0].entities.size() == 5);
    CHECK(!recorder.events[1].added && recorder.events[1].entities[0] == created[20]);
    CHECK(!recorder.events[2].added && recorder.events[2].entities[0] == created[21]);

    // A snapshot load removes everything and adds what it holds.
    nvrhi::AutoPtr<MemoryWriter> snapshot = MAKE_RC_OBJ_PTR(MemoryWriter);
    CHECK(env.world->SaveSnapshot(snapshot.Get()) == nvrhi::FS_OK);
    const size_t alive = recorder.indices.size();
    CHECK(alive == Flatten(Chunks(balls.Get())).size());
    recorder.events.clear();
    nvrhi::AutoPtr<MemoryReader> reader = MAKE_RC_OBJ_PTR(MemoryReader);
    reader->bytes = snapshot->bytes;
    // The loaded rows hold the indices of the save; the recorder expects fresh ones.
    auto added = recorder.domain->added;
    recorder.domain->added = [&](const EntityBatch& batch, ISimulationContext* context)
    {
        for (uint32_t i = 0; i < batch.count; ++i)
            static_cast<Body*>(batch.columns[0])[i].index = ~0u;
        return added(batch, context);
    };
    CHECK(env.world->LoadSnapshot(reader.Get()) == nvrhi::FS_OK);
    size_t removedCount = 0;
    size_t addedCount = 0;
    for (const Event& event : recorder.events)
        (event.added ? addedCount : removedCount) += event.entities.size();
    CHECK(removedCount == alive && addedCount == alive && recorder.indices.size() == alive);

    // Removal of the domain: the last chunk first, one batch per chunk; then nothing more.
    recorder.events.clear();
    chunks = Chunks(balls.Get());
    CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
    CHECK(recorder.events.size() == chunks.size() && recorder.indices.empty());
    for (size_t i = 0; i < chunks.size(); ++i)
        CHECK(!recorder.events[i].added && recorder.events[i].entities == chunks[chunks.size() - 1 - i].entities);
    CHECK(env.ObserverCount() == 0);
    env.Create({ env.ball, env.body }, 3);
    CHECK(recorder.events.size() == chunks.size());
    CHECK(env.Reports() == 0);
}

// The steps of a ChangesStructure domain: what a step records is played back after it, and the domain hears it before
// its next step. The calls inside see neither the step system's context nor its buffer.
static void test_changes_structure()
{
    Env env(4);
    std::vector<std::string> trace;
    Recorder recorder(env, "builder");
    recorder.trace = &trace;
    recorder.domain->desc.flags = SimulationDomainFlags::ChangesStructure;
    Query<With<Ball>> balls(env.world.Get(), "balls");
    ICommandBuffer* stepBuffer = nullptr;
    ISystemContext* stepContext = nullptr;
    const std::thread::id thread = std::this_thread::get_id();
    recorder.domain->step = [&](ISimulationContext* context)
    {
        trace.push_back("step " + std::to_string(context->GetStepInFrame()));
        EXPECT(std::this_thread::get_id() == thread);
        EXPECT(context->GetStepsThisFrame() == 3);
        stepBuffer = context->GetCommandBuffer();
        stepContext = context->GetSystemContext();
        EXPECT(stepBuffer && stepContext);
        EXPECT(balls.CountEntities() == 2 * context->GetStepInFrame());
        const ComponentId list[2] = { env.ball, env.body };
        EntityCreateDesc desc{};
        desc.components = list;
        desc.componentCount = 2;
        desc.entityCount = 2;
        Entity placeholders[2];
        stepBuffer->CreateEntities(desc, 0, placeholders);
        return nvrhi::FS_OK;
    };
    auto added = recorder.domain->added;
    recorder.domain->added = [&](const EntityBatch& batch, ISimulationContext* context)
    {
        EXPECT(std::this_thread::get_id() == thread);
        EXPECT(context->GetCommandBuffer() && context->GetCommandBuffer() != stepBuffer);
        EXPECT(context->GetCommandBuffer() == env.Impl().GetObserverRegistry()->GetRecordingBuffer());
        EXPECT(context->GetColumnCount(env.body) == 0 && context->GetStepInFrame() == 0);
        // A cascade: played back before the next step.
        context->GetCommandBuffer()->AddComponents(batch.entities, batch.count, env.red, nullptr, 0);
        return added(batch, context);
    };
    AddDomain(env, recorder.domain.Get());
    CHECK(env.ObserverCount() == 2 && recorder.events.empty());
    env.Frame(3.0, Phase::PreSimulation, Phase::FixedSimulation);
    const std::vector<std::string> expected = { "step 0", "added 2", "step 1", "added 2", "step 2", "added 2" };
    CHECK(trace == expected);
    CHECK(balls.CountEntities() == 6);
    for (const auto& [entity, index] : recorder.indices)
        CHECK(env.world->HasComponent(entity, env.red) && env.Get<Body>(entity)->index == index);
    CHECK(env.Reports() == 0 && env.Counters().fixedSteps == 3);
}

// What a lifetime callback sees and may do, wherever it comes from.
static void test_callback_context()
{
    Env env;
    const std::vector<Entity> before = env.Create({ env.ball, env.pos }, 4);
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("context");
    domain->terms = { Term(env.ball, Access::Filter) };
    domain->inputs = { R(env.pos) };
    domain->outputs = { W(env.pos) };
    SimulationDomainId id = SimulationDomainId::Invalid;
    nvrhi::AutoPtr<ScriptDomain> second = MakeDomain("second");
    std::vector<ICommandBuffer*> buffers;
    std::vector<uint8_t> saved;   // a stream that fits the domain, once it is there
    uint32_t calls = 0;
    auto callback = [&](const EntityBatch& batch, ISimulationContext* context)
    {
        ++calls;
        EXPECT(context->GetWorld() == env.world.Get() && context->GetSystemContext() == nullptr);
        EXPECT(context->GetJobSystem() == env.jobs.Get());
        EXPECT(id == SimulationDomainId::Invalid || context->GetDomainId() == id);
        EXPECT(context->GetStepInFrame() == 0 && context->GetStepIndex() == domain->steps);
        EXPECT(context->GetCommandList() == nullptr);
        ColumnView view;
        EXPECT(context->GetColumnCount(env.pos) == 0 && context->GetColumns(env.pos, 0, 0, 1, &view) == 0);
        buffers.push_back(context->GetCommandBuffer());
        EXPECT(context->GetCommandBuffer() != nullptr);

        // Observer rules: no immediate change, no change of the scheduler's domains.
        const uint64_t reports = env.Reports();
        EXPECT(env.world->AddComponent(batch.entities[0], env.red, nullptr) == nvrhi::FE_INVALID_ARGS);
        EXPECT(env.world->DestroyEntities(batch.entities, 1) == nvrhi::FE_INVALID_ARGS);
        EXPECT(env.Reports() == reports + 2);
        SimulationDomainId other;
        EXPECT(env.scheduler->AddSimulationDomain(second.Get(), other) == nvrhi::FE_INVALID_ARGS);
        EXPECT(env.scheduler->RemoveSimulationDomain(context->GetDomainId()) == nvrhi::FE_INVALID_ARGS);
        nvrhi::AutoPtr<MemoryWriter> writer = MAKE_RC_OBJ_PTR(MemoryWriter);
        EXPECT(env.scheduler->SaveSimulationState(writer.Get()) == nvrhi::FE_INVALID_ARGS && writer->bytes.empty());
        nvrhi::AutoPtr<MemoryReader> reader = MAKE_RC_OBJ_PTR(MemoryReader);
        reader->bytes = saved;
        EXPECT(env.scheduler->LoadSimulationState(reader.Get()) == nvrhi::FE_INVALID_ARGS && reader->position == 0);
        // Nor does a phase run, which would call this domain in the middle of its own callback.
        const uint32_t called[3] = { domain->gathers, domain->steps, domain->scatters };
        EXPECT(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FE_INVALID_ARGS);
        EXPECT(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FE_INVALID_ARGS);
        EXPECT(called[0] == domain->gathers && called[1] == domain->steps && called[2] == domain->scatters);
        EXPECT(context->GetCommandBuffer() == buffers.back() && context->GetSystemContext() == nullptr);
        EXPECT(env.Reports() == reports + 2);
        return nvrhi::FS_OK;
    };
    domain->added = callback;
    domain->removed = callback;
    uint32_t phaseCalls = 0;
    auto inPhase = [&](ISimulationContext* context)
    {
        ++phaseCalls;
        EXPECT(context->GetSystemContext() && context->GetCommandBuffer() &&
               context->GetCommandBuffer() == context->GetSystemContext()->GetCommandBuffer());
        EXPECT(context->GetJobSystem() == env.jobs.Get() && context->GetDomainId() == id);
        return nvrhi::FS_OK;
    };
    domain->gather = inPhase;
    domain->step = inPhase;
    domain->scatter = inPhase;
    id = AddDomain(env, domain.Get());                                                   // by the scheduler
    CHECK(calls == 1 && domain->addedEntities == 4);
    saved = env.Save();
    CHECK(env.Load(saved) == nvrhi::FS_OK);
    env.Create({ env.ball, env.pos }, 1);                                                // immediate
    CHECK(env.world->DestroyEntities(before.data(), 1) == nvrhi::FS_OK);
    CHECK(calls == 3);
    env.Add("late", Phase::BeginFrame, {}, [&](ISystemContext* context)
    {
        context->GetCommandBuffer()->DestroyEntity(before[1], 0);
    });
    env.Frame(1.0);                                                                      // a sync point
    CHECK(calls == 4 && phaseCalls == 3);
    CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);                    // by the scheduler
    CHECK(calls == 5);
    // The scheduler's own buffer for add and remove, the observers' otherwise.
    CHECK(buffers.size() == 5 && buffers[0] == buffers[4] && buffers[0] != buffers[1] && buffers[0] != buffers[3]);
    CHECK(env.SystemCount(Phase::FixedSimulation) == 0 && env.ObserverCount() == 0);
}

// Commands that the callbacks of AddSimulationDomain and RemoveSimulationDomain record apply before those return.
static void test_commands_in_add_and_remove()
{
    Env env;
    Query<With<Ball>> balls(env.world.Get(), "balls");
    const std::vector<Entity> existing = env.Create({ env.ball, env.body }, 6);
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("commands");
    domain->terms = { Term(env.ball, Access::Filter), Term(env.body, Access::Write) };
    bool created = false;
    domain->added = [&](const EntityBatch& batch, ISimulationContext* context)
    {
        ICommandBuffer* commands = context->GetCommandBuffer();
        commands->AddComponents(batch.entities, batch.count, env.red, nullptr, 0);
        if (!created)
        {
            created = true;
            const ComponentId list[2] = { env.ball, env.body };
            EntityCreateDesc desc{};
            desc.components = list;
            desc.componentCount = 2;
            desc.entityCount = 2;
            Entity placeholders[2];
            commands->CreateEntities(desc, 1, placeholders);
        }
        return nvrhi::FS_OK;
    };
    // An entity that leaves the query by the commands of its own removal must not be removed again.
    domain->removed = [&](const EntityBatch& batch, ISimulationContext* context)
    {
        context->GetCommandBuffer()->RemoveComponents(batch.entities, batch.count, env.body, 0);
        context->GetCommandBuffer()->AddComponents(batch.entities, batch.count, env.blue, nullptr, 1);
        return nvrhi::FS_OK;
    };
    const SimulationDomainId id = AddDomain(env, domain.Get());
    CHECK(balls.CountEntities() == 8 && domain->addedEntities == 8);
    for (Entity entity : Flatten(Chunks(balls.Get())))
        CHECK(env.world->HasComponent(entity, env.red));
    CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
    CHECK(domain->removedEntities == 8 && domain->addedEntities == 8);
    for (Entity entity : Flatten(Chunks(balls.Get())))
        CHECK(!env.world->HasComponent(entity, env.body) && env.world->HasComponent(entity, env.blue));

    // A failing callback undoes nothing: the domain is added, all its entities are delivered, the rest of the calls too.
    env.Create({ env.ball, env.body }, 2500);
    uint32_t batches = 0;
    domain->added = [&](const EntityBatch&, ISimulationContext*)
    {
        ++batches;
        return nvrhi::FE_GENERIC_ERROR;
    };
    domain->addedEntities = 0;
    const SimulationDomainId failing = AddDomain(env, domain.Get());
    CHECK(batches >= 3 && domain->addedEntities == 2500 && env.Reports() == 0);
    domain->removed = domain->added;
    domain->removedEntities = 0;
    CHECK(env.scheduler->RemoveSimulationDomain(failing) == nvrhi::FS_OK && domain->removedEntities == 2500);
}

// ----------------------------------------------------------------------------------------------- references

// The last reference to the scheduler goes away in user code that the scheduler itself called: it lasts until its call
// has returned, with everything it owns.
static void test_references_dropped_inside()
{
    enum Where { InAdd, InRemove, InSync, InSystem, InAddSystem, InRemoveSystem, InCapture, InRestore, Count };
    for (uint32_t where = 0; where < Count; ++where)
    {
        Env env;
        const std::vector<Entity> entities = env.Create({ env.ball, env.pad }, 400);   // several chunks
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("dropped");
        domain->terms = { Term(env.ball, Access::Filter) };
        uint32_t calls = 0;
        auto drop = [&]()
        {
            ++calls;
            env.scheduler = nullptr;
            EXPECT(ReferenceCount(domain.Get()) == 2);   // the scheduler's
        };
        auto callback = [&](const EntityBatch&, ISimulationContext* context)
        {
            drop();
            EXPECT(context->GetJobSystem() == env.jobs.Get() && context->GetWorld() == env.world.Get());
            return nvrhi::FS_OK;
        };
        IScheduler* scheduler = env.scheduler.Get();
        SimulationDomainId id = SimulationDomainId::Invalid;
        if (where == InAdd)
            domain->added = callback;
        else if (where == InRemove || where == InSync)
            domain->removed = callback;
        else if (where == InCapture)
            domain->capture = [&](IStateWriter* writer) { drop(); return writer->Write(&calls, sizeof(calls)); };
        else if (where == InRestore)
            domain->restore = [&](IStateReader* reader)
            {
                drop();
                uint32_t value = 0;
                return reader->Read(sizeof(value), &value) == nvrhi::FS_OK ? nvrhi::FE_GENERIC_ERROR : nvrhi::FS_OK;
            };
        struct Dropping final : nvrhi::ObjectImpl<ISystem>
        {
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Dropping)
            NVRHI_IMPLEMENTS_INTERFACE(ISystem)
            NVRHI_END_INTERFACE_TABLE()
            std::function<void()> atDesc;
            std::function<void()> atEnd;
            ~Dropping()
            {
                if (atEnd)
                    atEnd();
            }
            SystemDesc& GetDesc(SystemDesc& retVal) noexcept override
            {
                if (atDesc)
                    atDesc();
                retVal = SystemDesc{};
                retVal.name = "dropping";
                return retVal;
            }
            nvrhi::FRESULT Update(ISystemContext*) noexcept override { return nvrhi::FS_OK; }
        };
        SystemId systemId = SystemId::Invalid;
        nvrhi::AutoPtr<Dropping> system = MAKE_RC_OBJ_PTR(Dropping);
        if (where == InRemoveSystem)
        {
            system->atEnd = drop;
            CHECK(scheduler->AddSystem(system.Get(), systemId) == nvrhi::FS_OK);
            system = nullptr;
        }
        uint32_t later = 0;
        if (where == InSystem)
            env.Add("drop", Phase::PreSimulation, {}, [&](ISystemContext*) { drop(); });
        else if (where == InSync)
            env.Add("destroy", Phase::PreSimulation, {}, [&](ISystemContext* context)
            {
                context->GetCommandBuffer()->DestroyEntities(entities.data(), 3, 0);
            });
        env.Add("later", Phase::EndFrame, {}, [&](ISystemContext*) { ++later; });

        CHECK(scheduler->AddSimulationDomain(domain.Get(), id) == nvrhi::FS_OK && id == SimulationDomainId(0));
        const uint32_t chunks = 400 / 160 + 1;
        switch (where)
        {
        case InAdd:
            CHECK(calls == chunks && domain->addedEntities == 400);
            break;
        case InRemove:
            CHECK(scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
            CHECK(calls == chunks && domain->removedEntities == 400);
            break;
        case InSync:
            // The callbacks after the first, and the phases after it, as if nothing had happened.
            CHECK(scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
            CHECK(calls == 3 && domain->removedEntities == 3 && later == 1);
            break;
        case InSystem:
            CHECK(scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
            CHECK(calls == 1 && domain->gathers == 1 && domain->scatters == 1 && later == 1);
            break;
        case InAddSystem:
            system->atDesc = drop;
            CHECK(scheduler->AddSystem(system.Get(), systemId) == nvrhi::FS_OK && systemId != SystemId::Invalid);
            CHECK(calls == 1 && ReferenceCount(system.Get()) == 1);
            break;
        case InRemoveSystem:
            CHECK(scheduler->RemoveSystem(systemId) == nvrhi::FS_OK);
            CHECK(calls == 1);
            break;
        default:
        {
            nvrhi::AutoPtr<MemoryWriter> writer = MAKE_RC_OBJ_PTR(MemoryWriter);
            CHECK(scheduler->SaveSimulationState(writer.Get()) == nvrhi::FS_OK && !writer->bytes.empty());
            if (where == InRestore)
            {
                nvrhi::AutoPtr<MemoryReader> reader = MAKE_RC_OBJ_PTR(MemoryReader);
                reader->bytes = writer->bytes;
                CHECK(scheduler->LoadSimulationState(reader.Get()) == nvrhi::FS_OK);
            }
            CHECK(calls == 1);
            break;
        }
        }
        // Gone now, observers and all.
        CHECK(!env.scheduler && ReferenceCount(domain.Get()) == 1 && env.ObserverCount() == 0);
        const uint32_t removed = domain->removedEntities;
        domain->removed = nullptr;
        CHECK(env.world->DestroyEntities(&entities[10], 1) == nvrhi::FS_OK && domain->removedEntities == removed);
        CHECK(env.Reports() == 0);
    }
    // The scheduler's destruction releases domains whose destructors change the world: the others hear nothing.
    struct Killer final : nvrhi::ObjectImpl<ISimulationDomain>
    {
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Killer)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()
        IWorld* world = nullptr;
        Entity victim = Entity::Null;
        ~Killer() { world->DestroyEntities(&victim, 1); }
        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override
        {
            retVal = SimulationDomainDesc{};
            retVal.name = "killer";
            retVal.fixedTimeStep = 1.0;
            retVal.maxStepsPerFrame = 1;
            return retVal;
        }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch&, ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch&, ISimulationContext*) noexcept override
        {
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT Step(double, ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT CaptureState(IStateWriter*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT RestoreState(IStateReader*) noexcept override { return nvrhi::FS_OK; }
    };
    Env env;
    const std::vector<Entity> entities = env.Create({ env.ball }, 4);
    nvrhi::AutoPtr<Killer> killers[2] = { MAKE_RC_OBJ_PTR(Killer), MAKE_RC_OBJ_PTR(Killer) };
    nvrhi::AutoPtr<ScriptDomain> witness = MakeDomain("witness");
    witness->terms = { Term(env.ball, Access::Filter) };
    killers[0]->world = env.world.Get();
    killers[0]->victim = entities[1];
    AddDomain(env, killers[0].Get());
    AddDomain(env, witness.Get());
    killers[1]->world = env.world.Get();
    killers[1]->victim = entities[2];
    killers[1] = nullptr;   // the second one only shows that the destruction is heard otherwise
    CHECK(witness->removedEntities == 1);
    killers[0] = nullptr;
    env.scheduler = nullptr;
    CHECK(!env.world->IsAlive(entities[1]) && witness->removedEntities == 1 && ReferenceCount(witness.Get()) == 1);
}

// The world holds its observers and a domain's query holds the world: the scheduler's observers hold nothing.
static void test_references()
{
    // Counts: the scheduler's one reference to the domain, none to itself.
    {
        Env env;
        env.Create({ env.ball }, 10);
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("counted");
        domain->terms = { Term(env.ball, Access::Filter) };
        const long schedulerReferences = ReferenceCount(env.scheduler.Get());
        const long worldReferences = ReferenceCount(env.world.Get());
        CHECK(ReferenceCount(domain.Get()) == 1);
        const SimulationDomainId id = AddDomain(env, domain.Get());
        CHECK(ReferenceCount(domain.Get()) == 2 && ReferenceCount(env.scheduler.Get()) == schedulerReferences);
        CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
        CHECK(ReferenceCount(domain.Get()) == 1 && domain->removedEntities == 10);
        // The buffer of the lifetime callbacks stays with the scheduler and holds the world; the query is gone.
        CHECK(ReferenceCount(env.world.Get()) == worldReferences + 1);
        CHECK(env.ObserverCount() == 0);
    }
    // The scheduler's destruction: no delivery, no observer left; the domain with its own query may outlive it, and
    // events after it reach nobody.
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    {
        Env env;
        const std::vector<Entity> entities = env.Create({ env.ball }, 10);
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("outliving");
        domain->terms = { Term(env.ball, Access::Filter) };
        const QueryTerm terms[1] = { Term(env.ball, Access::Filter) };
        CHECK(env.world->CreateQuery(QueryDesc{ terms, 1, 0, "held" }, &domain->heldQuery) == nvrhi::FS_OK);
        AddDomain(env, domain.Get());
        CHECK(domain->addedEntities == 10 && env.ObserverCount() == 2);
        env.scheduler = nullptr;
        CHECK(domain->removedEntities == 0 && ReferenceCount(domain.Get()) == 1 && env.ObserverCount() == 0);
        CHECK(env.world->DestroyEntities(entities.data(), 5) == nvrhi::FS_OK);
        env.Create({ env.ball }, 3);
        CHECK(domain->removedEntities == 0 && domain->addedEntities == 10 && env.Reports() == 0);
    }
    CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    // The same, released in the other order: the world is dropped first, the scheduler holds it.
    {
        Env env;
        env.Create({ env.ball }, 10);
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("last");
        domain->terms = { Term(env.ball, Access::Filter) };
        AddDomain(env, domain.Get());
        env.world = nullptr;
        env.scheduler = nullptr;
        CHECK(domain->removedEntities == 0);
    }
    CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    // The last reference to the scheduler goes away inside a callback: its observers cannot be removed there, so they
    // stay in the world, disconnected and without a reference to anything.
    for (bool change : { false, true })
    {
        Env env;
        const std::vector<Entity> entities = env.Create({ env.ball, env.pos }, 10);
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("dropped");
        domain->terms = { Term(env.ball, Access::Filter) };
        const QueryTerm terms[1] = { Term(env.ball, Access::Filter) };
        CHECK(env.world->CreateQuery(QueryDesc{ terms, 1, 0, "held" }, &domain->heldQuery) == nvrhi::FS_OK);
        AddDomain(env, domain.Get());
        uint32_t calls = 0;
        domain->removed = [&](const EntityBatch&, ISimulationContext*)
        {
            ++calls;
            env.scheduler = nullptr;
            // The scheduler, and its reference to this domain, last until the callback has returned.
            EXPECT(ReferenceCount(domain.Get()) == 2);
            return nvrhi::FS_OK;
        };
        CHECK(env.world->DestroyEntities(entities.data(), change ? 1 : 4) == nvrhi::FS_OK);
        CHECK(calls == 1 && env.ObserverCount() == 2 && ReferenceCount(domain.Get()) == 1);
        env.Create({ env.ball }, 3);
        CHECK(calls == 1 && domain->addedEntities == 10 && env.Reports() == 0);
    }
    CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    test_references_dropped_inside();
    CHECK(ecs::GetHeapLiveBlockCount() == blocks);
}

// Domains that cannot be added leave nothing behind, and a scheduler has c_MaxSimulationDomains.
static void test_limits()
{
    Env env;
    env.Create({ env.ball, env.push }, 10);
    nvrhi::AutoPtr<ScriptDomain> bad = MakeDomain("bad");
    auto refused = [&](nvrhi::FRESULT result)
    {
        AddDomain(env, bad.Get(), result);
        CHECK(ReferenceCount(bad.Get()) == 1 && env.ObserverCount() == 0 && bad->addedEntities == 0);
        CHECK(env.SystemCount(Phase::PreSimulation) == 0 && env.SystemCount(Phase::FixedSimulation) == 0 &&
              env.SystemCount(Phase::PostSimulation) == 0);
        CHECK(env.Step(SimulationDomainId(0)).fixedTimeStep == 0.0);
    };
    // Terms that observers cannot filter by.
    const uint64_t reports = env.Reports();
    bad->terms = { Term(env.ball, Access::Filter), Term(env.push, Access::Read, TermOp::Enabled) };
    refused(nvrhi::FE_INVALID_ARGS);
    bad->terms = { Term(env.ball, Access::Filter), Term(env.push, Access::Read, TermOp::Changed) };
    refused(nvrhi::FE_INVALID_ARGS);
    bad->terms = { Term(env.ball, Access::Filter), Term(env.chosen, Access::Filter) };
    refused(nvrhi::FE_INVALID_ARGS);
    bad->terms = { Term(env.group, Access::Write) };
    refused(nvrhi::FE_INVALID_ARGS);
    CHECK(env.Reports() > reports);
    bad->terms = { Term(env.ball, Access::Filter) };
    bad->inputs = { R(ComponentId(5000)) };
    refused(nvrhi::FE_INVALID_ARGS);
    bad->inputs = {};
    bad->desc.queue = SimulationQueue(9);
    refused(nvrhi::FE_INVALID_ARGS);
    bad->desc.queue = SimulationQueue::Cpu;

    std::vector<nvrhi::AutoPtr<ScriptDomain>> domains;
    std::vector<SimulationDomainId> ids;
    for (uint32_t i = 0; i < c_MaxSimulationDomains; ++i)
    {
        domains.push_back(MakeDomain(("domain " + std::to_string(i)).c_str(), 1.0 + i));
        domains.back()->terms = { Term(env.ball, Access::Filter) };
        ids.push_back(AddDomain(env, domains.back().Get()));
        CHECK(uint32_t(ids.back()) < c_MaxSimulationDomains && env.Step(ids.back()).fixedTimeStep == 1.0 + i);
    }
    const FixedStepState state = *GetResource<FixedStepState>(env.world.Get());
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    AddDomain(env, bad.Get(), nvrhi::FE_OUT_OF_MEMORY);
    CHECK(ReferenceCount(bad.Get()) == 1 && bad->addedEntities == 0 && ecs::GetHeapLiveBlockCount() == blocks);
    CHECK(env.SystemCount(Phase::FixedSimulation) == c_MaxSimulationDomains &&
          env.ObserverCount() == 2 * c_MaxSimulationDomains);
    CHECK(std::memcmp(&state, GetResource<FixedStepState>(env.world.Get()), sizeof(state)) == 0);
    env.Frame(1.0);
    // A free place is used again, from zero.
    CHECK(env.scheduler->RemoveSimulationDomain(ids[5]) == nvrhi::FS_OK && domains[5]->removedEntities == 10);
    CHECK(env.Step(ids[5]).fixedTimeStep == 0.0 && env.Step(ids[0]).stepIndex == 1);
    CHECK(AddDomain(env, bad.Get()) == ids[5] && bad->addedEntities == 10);
    CHECK(env.Step(ids[5]).fixedTimeStep == 1.0 && env.Step(ids[5]).stepIndex == 0 &&
          env.Step(ids[5]).accumulator == 0.0);
}

// --------------------------------------------------------------------------------------------------- queues

static void test_queues()
{
    const std::thread::id thread = std::this_thread::get_id();
    nvrhi::ICommandList* const listA = reinterpret_cast<nvrhi::ICommandList*>(uintptr_t(0x1000));   // never dereferenced
    nvrhi::ICommandList* const listB = reinterpret_cast<nvrhi::ICommandList*>(uintptr_t(0x2000));

    // Every call of a GPU domain is made on the thread that runs the phase, and no job runs for it.
    for (SimulationQueue queue : { SimulationQueue::Graphics, SimulationQueue::Compute, SimulationQueue::Cuda })
    {
        Env env(4);
        env.Create({ env.ball, env.pos }, 10);
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("gpu");
        domain->desc.queue = queue;
        domain->terms = { Term(env.ball, Access::Filter) };
        domain->inputs = { R(env.pos) };
        domain->outputs = { W(env.pos) };
        uint32_t calls = 0;
        auto call = [&](ISimulationContext*)
        {
            ++calls;
            EXPECT(std::this_thread::get_id() == thread);
            return nvrhi::FS_OK;
        };
        domain->gather = domain->step = domain->scatter = call;
        const uint64_t jobs = env.Counters().jobsScheduled;
        AddDomain(env, domain.Get());
        for (uint32_t frame = 0; frame < 20; ++frame)
            env.Frame(2.0);
        CHECK(calls == 20 * 4 && env.Counters().jobsScheduled == jobs);
        CHECK(env.Diagnostics().find("'gpu.step' (main thread)") != std::string::npos);

        // A ChangesStructure one is exclusive in its step, as a CPU domain is.
        nvrhi::AutoPtr<ScriptDomain> exclusive = MakeDomain("exclusive");
        exclusive->desc.queue = queue;
        exclusive->desc.flags = SimulationDomainFlags::ChangesStructure;
        exclusive->step = call;
        AddDomain(env, exclusive.Get());
        CHECK(env.Diagnostics().find("'exclusive.step' (exclusive)") != std::string::npos);
        CHECK(env.Diagnostics().find("'exclusive.gather' (main thread)") != std::string::npos);
    }
    // The systems of a CPU domain are jobs.
    {
        Env env(4);
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("cpu");
        AddDomain(env, domain.Get());
        const uint64_t jobs = env.Counters().jobsScheduled;
        env.Frame(1.0);
        CHECK(env.Counters().jobsScheduled == jobs + 3);
    }

    // The order of the calls is the schedule order, and each domain gets its own list for the frame.
    Env env(4);
    env.Create({ env.ball, env.pos }, 10);
    std::vector<std::string> order;
    nvrhi::ICommandList* seen[3][3] = {};
    nvrhi::ICommandList* seenInCallback = nullptr;
    nvrhi::AutoPtr<ScriptDomain> domains[3] = { MakeDomain("a"), MakeDomain("b"), MakeDomain("cpu") };
    SimulationDomainId ids[3];
    for (uint32_t d = 0; d < 3; ++d)
    {
        ScriptDomain* domain = domains[d].Get();
        domain->desc.queue = d == 0 ? SimulationQueue::Graphics : d == 1 ? SimulationQueue::Cuda : SimulationQueue::Cpu;
        auto call = [&, d](uint32_t role, const char* text)
        {
            return [&, d, role, text](ISimulationContext* context)
            {
                seen[d][role] = context->GetCommandList();
                if (d < 2)
                    order.push_back(std::string(text) + (d == 0 ? " a" : " b"));
                return nvrhi::FS_OK;
            };
        };
        domain->gather = call(0, "gather");
        domain->step = call(1, "step");
        domain->scatter = call(2, "scatter");
    }
    domains[0]->terms = { Term(env.ball, Access::Filter) };
    domains[0]->added = [&](const EntityBatch&, ISimulationContext* context)
    {
        seenInCallback = context->GetCommandList();
        return nvrhi::FS_OK;
    };
    for (uint32_t d = 0; d < 3; ++d)
        ids[d] = AddDomain(env, domains[d].Get());
    CHECK(seenInCallback == nullptr);
    bool spawn = false;
    nvrhi::FRESULT inPhase = nvrhi::FS_OK;
    env.Add("spawn", Phase::PreSimulation, {}, [&](ISystemContext* context)
    {
        inPhase = env.scheduler->SetSimulationCommandList(ids[0], listB);
        if (!spawn)
            return;
        const ComponentId list[1] = { env.ball };
        EntityCreateDesc desc{};
        desc.components = list;
        desc.componentCount = 1;
        desc.entityCount = 1;
        Entity placeholder;
        context->GetCommandBuffer()->CreateEntities(desc, 0, &placeholder);
    });
    CHECK(env.scheduler->SetSimulationCommandList(ids[0], listA) == nvrhi::FS_OK);
    CHECK(env.scheduler->SetSimulationCommandList(ids[1], listB) == nvrhi::FS_OK);
    CHECK(env.scheduler->SetSimulationCommandList(ids[2], listA) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->SetSimulationCommandList(SimulationDomainId(3), listA) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->SetSimulationCommandList(SimulationDomainId::Invalid, listA) == nvrhi::FE_INVALID_ARGS);
    spawn = true;   // a lifetime callback at the sync point after PostSimulation still has the list
    env.Frame(1.0);
    spawn = false;
    CHECK(inPhase == nvrhi::FE_INVALID_ARGS);
    const std::vector<std::string> expected = { "gather a", "gather b", "step a", "step b", "scatter a", "scatter b" };
    CHECK(order == expected);
    for (uint32_t role = 0; role < 3; ++role)
        CHECK(seen[0][role] == listA && seen[1][role] == listB && seen[2][role] == nullptr);
    CHECK(domains[0]->addedEntities == 11 && seenInCallback == listA);
    // Forgotten after PostSimulation.
    env.Frame(1.0);
    for (uint32_t role = 0; role < 3; ++role)
        CHECK(seen[0][role] == nullptr && seen[1][role] == nullptr);
    // Null takes it back.
    CHECK(env.scheduler->SetSimulationCommandList(ids[0], listA) == nvrhi::FS_OK);
    CHECK(env.scheduler->SetSimulationCommandList(ids[0], nullptr) == nvrhi::FS_OK);
    env.Frame(1.0);
    CHECK(seen[0][1] == nullptr);
    // PreSimulation and FixedSimulation do not take it.
    CHECK(env.scheduler->SetSimulationCommandList(ids[1], listB) == nvrhi::FS_OK);
    env.Frame(1.0, Phase::PreSimulation, Phase::FixedSimulation);
    CHECK(seen[1][1] == listB);
    CHECK(env.scheduler->RunPhase(Phase::PostSimulation) == nvrhi::FS_OK && seen[1][2] == listB);
    // An ordering cycle keeps PostSimulation from running: the frame's list is forgotten all the same, but not by a range
    // without PostSimulation that cannot run.
    const SystemId first = env.Add("first", Phase::BeginFrame, {}, [](ISystemContext*) { }, { "second" });
    const SystemId second = env.Add("second", Phase::BeginFrame, {}, [](ISystemContext*) { }, { "first" });
    CHECK(env.scheduler->SetSimulationCommandList(ids[1], listB) == nvrhi::FS_OK);
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::FixedSimulation) == nvrhi::FE_INVALID_ARGS);
    env.SetTime(1.0);
    CHECK(env.scheduler->RunPhases(Phase::PreSimulation, Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(seen[1][1] == listB);
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK && seen[1][0] == nullptr);
    CHECK(env.scheduler->RemoveSystem(first) == nvrhi::FS_OK && env.scheduler->RemoveSystem(second) == nvrhi::FS_OK);
    CHECK(env.Reports() == 0);
}

// ----------------------------------------------------------------------------------------------------- time

static void test_planned_steps()
{
    Env env;
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("time", 0.25, 3);
    uint32_t gatherPlan = 99;
    uint32_t scatterTaken = 99;
    std::vector<uint32_t> inStep;
    std::vector<uint64_t> indices;
    domain->gather = [&](ISimulationContext* context)
    {
        gatherPlan = context->GetStepsThisFrame();
        EXPECT(context->GetStepInFrame() == 0);
        return nvrhi::FS_OK;
    };
    domain->step = [&](ISimulationContext* context)
    {
        inStep.push_back(context->GetStepsThisFrame() * 10 + context->GetStepInFrame());
        indices.push_back(context->GetStepIndex());
        return nvrhi::FS_OK;
    };
    domain->scatter = [&](ISimulationContext* context)
    {
        scatterTaken = context->GetStepsThisFrame();
        EXPECT(context->GetStepInFrame() == 0);
        return nvrhi::FS_OK;
    };
    const SimulationDomainId id = AddDomain(env, domain.Get());
    auto entry = [&](double accumulator, uint64_t index, uint32_t planned, uint32_t taken, uint32_t dropped)
    {
        const FixedStepEntry& e = env.Step(id);
        return e.accumulator == accumulator && e.fixedTimeStep == 0.25 && e.stepIndex == index &&
               e.stepsPlanned == planned && e.stepsTaken == taken && e.stepsDropped == dropped && e.reserved == 0;
    };
    // Published when the domain is added.
    CHECK(entry(0.0, 0, 0, 0, 0));

    // Planned by SetFrameTime, which consumes nothing; consumed by the step system.
    env.SetTime(0.625);
    CHECK(entry(0.625, 0, 2, 0, 0) && domain->steps == 0);
    CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
    CHECK(gatherPlan == 2 && entry(0.625, 0, 2, 0, 0));
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(entry(0.125, 2, 0, 2, 0) && domain->steps == 2);
    CHECK((inStep == std::vector<uint32_t>{ 20, 21 }) && (indices == std::vector<uint64_t>{ 0, 1 }));
    // A second FixedSimulation takes no step and keeps the counts.
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(entry(0.125, 2, 0, 2, 0) && domain->steps == 2);
    CHECK(env.scheduler->RunPhase(Phase::PostSimulation) == nvrhi::FS_OK && scatterTaken == 2);

    // SetFrameTime twice before a run plans the sum.
    env.SetTime(0.125);
    CHECK(entry(0.25, 2, 1, 0, 0));
    env.SetTime(0.25);
    CHECK(entry(0.5, 2, 2, 0, 0));
    inStep.clear();
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(entry(0.0, 4, 0, 2, 0) && (inStep == std::vector<uint32_t>{ 20, 21 }));

    // Beyond the limit: the plan is the limit, the rest of the whole steps is dropped when they are consumed.
    env.SetTime(1.875);
    CHECK(entry(1.875, 4, 3, 0, 0));
    env.SetTime(0.0);   // still the whole accumulator
    CHECK(entry(1.875, 4, 3, 0, 0));
    CHECK(env.scheduler->RunPhases(Phase::PreSimulation, Phase::PostSimulation) == nvrhi::FS_OK);
    CHECK(gatherPlan == 3 && scatterTaken == 3 && entry(0.125, 7, 0, 3, 4));
    // The next SetFrameTime starts the counts again.
    env.SetTime(0.0);
    CHECK(entry(0.125, 7, 0, 0, 0));
    CHECK(env.scheduler->RunPhases(Phase::PreSimulation, Phase::PostSimulation) == nvrhi::FS_OK);
    CHECK(gatherPlan == 0 && scatterTaken == 0 && entry(0.125, 7, 0, 0, 0) && domain->steps == 7);

    // Deltas that are no time.
    env.SetTime(-1.0);
    CHECK(entry(0.125, 7, 0, 0, 0));
    env.SetTime(std::nan(""));
    CHECK(entry(0.125, 7, 0, 0, 0));
    env.SetTime(1e300);
    CHECK(env.Step(id).stepsPlanned == 3);
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(env.Step(id).stepsTaken == 3 && env.Step(id).stepsDropped == 0xffffffffu && env.Step(id).stepIndex == 10);
    CHECK(env.Step(id).accumulator == 0.0);

    // A time beyond what a double resolves leaves no steps behind (1e30 - floor(1e30 / 0.7) * 0.7 is 1.4e14), and a sum
    // that overflows stays a number.
    nvrhi::AutoPtr<ScriptDomain> coarse = MakeDomain("seven tenths", 0.7, 3);
    const SimulationDomainId coarseId = AddDomain(env, coarse.Get());
    for (double huge : { 1e17, 5e29, 1e300, std::numeric_limits<double>::max() })
    {
        coarse->steps = 0;
        env.SetTime(huge);
        env.SetTime(huge);
        CHECK(std::isfinite(env.Step(coarseId).accumulator) && env.Step(coarseId).stepsPlanned == 3);
        CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK && coarse->steps == 3);
        CHECK(env.Step(coarseId).accumulator == 0.0 && env.Step(coarseId).stepsDropped == 0xffffffffu);
        env.SetTime(0.0);
        CHECK(env.Step(coarseId).stepsPlanned == 0);
        const std::vector<uint8_t> times = env.Save();
        CHECK(env.Load(times) == nvrhi::FS_OK && env.Save() == times);
    }
    env.SetTime(std::numeric_limits<double>::max());
    env.SetTime(std::numeric_limits<double>::max());
    CHECK(env.Load(env.Save()) == nvrhi::FS_OK);   // what a save wrote can be loaded
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(env.scheduler->RemoveSimulationDomain(coarseId) == nvrhi::FS_OK);

    // A domain added between SetFrameTime and the run has no time yet.
    env.SetTime(1.0);
    nvrhi::AutoPtr<ScriptDomain> late = MakeDomain("late", 0.25, 3);
    const SimulationDomainId lateId = AddDomain(env, late.Get());
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK && late->steps == 0);
    env.Frame(0.25);
    CHECK(late->steps == 1 && env.Step(lateId).stepIndex == 1);
    // Removed: nothing has the id.
    CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
    const FixedStepEntry none{};
    CHECK(std::memcmp(&env.Step(id), &none, sizeof(none)) == 0 && env.Step(lateId).fixedTimeStep == 0.25);
}

// What readers of the resource see, and when it is published.
static void test_publication()
{
    Env env(4);
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("published", 0.5, 4);
    // A ChangesStructure step runs alone on the calling thread, and nothing is published inside the phase all the same.
    nvrhi::AutoPtr<ScriptDomain> builder = MakeDomain("builder", 0.5, 4);
    builder->desc.flags = SimulationDomainFlags::ChangesStructure;
    const SimulationDomainId id = AddDomain(env, domain.Get());
    const SimulationDomainId builderId = AddDomain(env, builder.Get());
    FixedStepEntry inPre{}, inFixed{}, inBuilder[2] = {}, inPost{};
    builder->step = [&](ISimulationContext* context)
    {
        if (context->GetStepInFrame() < 2)
            inBuilder[context->GetStepInFrame()] = env.Step(builderId);
        return nvrhi::FS_OK;
    };
    env.Add("pre", Phase::PreSimulation, { R(env.fixedStep) }, [&](ISystemContext*) { inPre = env.Step(id); });
    env.Add("fixed", Phase::FixedSimulation, { R(env.fixedStep) }, [&](ISystemContext*) { inFixed = env.Step(id); },
            { "published.step" }, { "builder.step" });
    env.Add("post", Phase::PostSimulation, { R(env.fixedStep) }, [&](ISystemContext*) { inPost = env.Step(id); });
    CHECK(env.scheduler->Build() == nvrhi::FS_OK && env.scheduler->GetAmbiguityCount() == 0);

    env.SetTime(1.25);
    CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
    CHECK(inPre.stepsPlanned == 2 && inPre.accumulator == 1.25 && inPre.stepsTaken == 0);
    // FixedSimulation starts from the scheduler's state, whatever was written over the copy.
    FixedStepState* resource = GetResource<FixedStepState>(env.world.Get());
    std::memset(resource, 0x5a, sizeof(FixedStepEntry) * 2);
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
    CHECK(inFixed.stepsPlanned == 2 && inFixed.accumulator == 1.25 && inFixed.stepsTaken == 0 && inFixed.stepIndex == 0);
    for (const FixedStepEntry& entry : inBuilder)
        CHECK(entry.stepsPlanned == 2 && entry.accumulator == 1.25 && entry.stepsTaken == 0 && entry.stepIndex == 0);
    CHECK(env.Step(id).stepsTaken == 2 && env.Step(id).accumulator == 0.25 && env.Step(id).stepIndex == 2);
    CHECK(env.Step(builderId).stepsTaken == 2 && env.Step(builderId).stepIndex == 2);
    CHECK(env.scheduler->RunPhase(Phase::PostSimulation) == nvrhi::FS_OK);
    CHECK(inPost.stepsTaken == 2 && inPost.stepsPlanned == 0 && inPost.accumulator == 0.25);
    // Not part of the world's content.
    const uint64_t hash = env.world->ComputeHash();
    resource->domains[3].accumulator = 17.0;
    CHECK(env.world->ComputeHash() == hash && env.Reports() == 0);
}

// Steps of independent domains run at once, beside a system that reads FixedStepState: no ambiguity, and the reader has
// the values of the phase's start also after the steps ended.
static void test_parallel_steps()
{
    Env env(4);
    std::atomic<uint32_t> arrived{ 0 };
    std::atomic<uint32_t> finished{ 0 };
    auto wait = [](const std::atomic<uint32_t>& value, uint32_t target)
    {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (value.load() < target)
        {
            if (std::chrono::steady_clock::now() > end)
                return false;
            std::this_thread::yield();
        }
        return true;
    };
    nvrhi::AutoPtr<ScriptDomain> domains[2] = { MakeDomain("left", 0.5, 4), MakeDomain("right", 0.25, 4) };
    SimulationDomainId ids[2];
    for (uint32_t d = 0; d < 2; ++d)
    {
        // Both write the same column in scatter, which the gate below orders: their steps have nothing to do with it.
        domains[d]->inputs = { R(env.pos) };
        domains[d]->outputs = { W(env.pos) };
        ScriptDomain* domain = domains[d].Get();
        domain->step = [&, domain](ISimulationContext* context)
        {
            if (context->GetStepInFrame() == 0)
            {
                ++arrived;
                EXPECT(wait(arrived, 3));   // the other step and the reader are running
            }
            if (context->GetStepInFrame() + 1 == context->GetStepsThisFrame())
                ++finished;
            return nvrhi::FS_OK;
        };
        ids[d] = AddDomain(env, domain);
    }
    env.Add("gate", Phase::PostSimulation, {}, [](ISystemContext*) { }, { "left.scatter" }, { "right.scatter" });
    FixedStepEntry during[2] = {}, after[2] = {};
    env.Add("reader", Phase::FixedSimulation, { R(env.fixedStep) }, [&](ISystemContext*)
    {
        ++arrived;
        EXPECT(wait(arrived, 3));
        for (uint32_t d = 0; d < 2; ++d)
            during[d] = env.Step(ids[d]);
        EXPECT(wait(finished, 2));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // the step systems return
        for (uint32_t d = 0; d < 2; ++d)
            after[d] = env.Step(ids[d]);
    });
    CHECK(env.scheduler->Build() == nvrhi::FS_OK && env.scheduler->GetAmbiguityCount() == 0);
    for (uint32_t frame = 0; frame < 3; ++frame)
    {
        arrived = 0;
        finished = 0;
        env.SetTime(1.0);
        CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
        for (uint32_t d = 0; d < 2; ++d)
        {
            const uint32_t steps = d == 0 ? 2 : 4;
            for (const FixedStepEntry* seen : { &during[d], &after[d] })
                CHECK(seen->accumulator == 1.0 && seen->stepsPlanned == steps && seen->stepsTaken == 0 &&
                      seen->stepIndex == uint64_t(frame) * steps);
            CHECK(env.Step(ids[d]).stepsTaken == steps && env.Step(ids[d]).accumulator == 0.0);
        }
        CHECK(arrived == 3 && finished == 2);
    }
    CHECK(env.Reports() == 0);
}

// A failed step ends the domain's steps of the frame; everything else goes on.
static void test_failed_steps()
{
    for (bool changesStructure : { false, true })
    {
        for (uint32_t failAt : { 0u, 1u, 2u })
        {
            Env env(2);
            Query<With<Ball>> balls(env.world.Get(), "balls");
            nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("failing", 0.5, 3);
            nvrhi::AutoPtr<ScriptDomain> other = MakeDomain("other", 0.5, 8);
            if (changesStructure)
                domain->desc.flags = SimulationDomainFlags::ChangesStructure;
            bool fail = true;
            uint32_t scatterSteps = 99;
            std::vector<uint64_t> stepIndices;
            domain->step = [&](ISimulationContext* context)
            {
                stepIndices.push_back(context->GetStepIndex());
                if (changesStructure)
                {
                    const ComponentId list[1] = { env.ball };
                    EntityCreateDesc desc{};
                    desc.components = list;
                    desc.componentCount = 1;
                    desc.entityCount = 1;
                    Entity placeholder;
                    context->GetCommandBuffer()->CreateEntities(desc, 0, &placeholder);
                }
                return fail && context->GetStepInFrame() == failAt ? nvrhi::FE_NOT_FOUND : nvrhi::FS_OK;
            };
            domain->scatter = [&](ISimulationContext* context)
            {
                scatterSteps = context->GetStepsThisFrame();
                return nvrhi::FS_OK;
            };
            const SimulationDomainId id = AddDomain(env, domain.Get());
            const SimulationDomainId otherId = AddDomain(env, other.Get());
            uint32_t laterSystems = 0;
            env.Add("later", Phase::TransformPropagation, {}, [&](ISystemContext*) { ++laterSystems; });

            // 5 whole steps and a fraction, at most 3 planned.
            env.SetTime(2.625);
            CHECK(env.Step(id).stepsPlanned == 3);
            CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
            CHECK(domain->steps == failAt + 1 && other->steps == 5);
            const FixedStepEntry failed = env.Step(id);
            CHECK(failed.stepIndex == failAt && failed.stepsTaken == failAt && failed.stepsPlanned == 0 &&
                  failed.stepsDropped == 5 - failAt && failed.accumulator == 0.125);
            CHECK(env.Step(otherId).stepIndex == 5 && env.Step(otherId).stepsDropped == 0);
            // What the failed step recorded is played back like any step's.
            CHECK(balls.CountEntities() == (changesStructure ? failAt + 1 : 0));
            CHECK(env.scheduler->RunPhases(Phase::PostSimulation, Phase::EndFrame) == nvrhi::FS_OK);
            CHECK(domain->scatters == 1 && scatterSteps == failAt && laterSystems == 1);
            std::string text = env.Diagnostics();
            CHECK(text.find("system 'failing.step': 1 failed updates (last 0x") != std::string::npos);
            CHECK(text.find("'other.step':") == std::string::npos);
            CHECK(text.find("'failing.scatter':") == std::string::npos);

            // The next frame tries again, from the same step index.
            fail = false;
            stepIndices.clear();
            env.Frame(0.5);
            CHECK(domain->steps == failAt + 2 && (stepIndices == std::vector<uint64_t>{ failAt }));
            CHECK(env.Step(id).stepIndex == failAt + 1 && env.Step(id).stepsTaken == 1 && env.Step(id).stepsDropped == 0);
            CHECK(env.Diagnostics().find("'failing.step': 1 failed updates") != std::string::npos);
            fail = true;
            env.Frame(1.5);
            CHECK(env.Diagnostics().find("'failing.step': 2 failed updates") != std::string::npos);
            CHECK(env.Step(id).stepsDropped == 3 - failAt && env.Step(id).stepIndex == 2 * failAt + 1);
            CHECK(env.Counters().fixedSteps == domain->steps + other->steps);
        }
    }
}

// A scheduler made later on the same world starts FixedStepState at zero, and its domains from nothing.
static void test_scheduler_recreated()
{
    Env env;
    env.Create({ env.ball, env.pos, env.body, env.mass, env.push }, 20);
    nvrhi::AutoPtr<BallDomain> domain = MAKE_RC_OBJ_PTR(BallDomain, env);
    SimulationDomainId id = AddDomain(env, domain.Get());
    nvrhi::AutoPtr<ScriptDomain> second = MakeDomain("second", 0.5, 4);
    const SimulationDomainId secondId = AddDomain(env, second.Get());
    env.Frame(1.1);
    CHECK(env.Step(id).stepIndex == 4 && env.Step(secondId).stepIndex == 2);
    const FixedStepState zero{};
    // Not inside a phase, where a system may be reading the copy; nor does another scheduler's time enter then.
    const FixedStepState before = *GetResource<FixedStepState>(env.world.Get());
    nvrhi::AutoPtr<IScheduler> refused;
    nvrhi::AutoPtr<IScheduler> bystander;
    CHECK(CreateScheduler(env.world.Get(), env.jobs.Get(), &bystander) == nvrhi::FS_OK);
    *GetResource<FixedStepState>(env.world.Get()) = before;
    nvrhi::FRESULT created[2] = {};
    FixedStepState seen{};
    const SystemId make = env.Add("make", Phase::PreSimulation, {}, [&](ISystemContext*)
    {
        created[0] = CreateScheduler(env.world.Get(), env.jobs.Get(), &refused);
        EXPECT(!refused);
        bystander->SetFrameTime(FrameTime{ 1000.0, 1000.0, 77 });
    });
    const SystemId read = env.Add("read", Phase::PreSimulation, { R(env.fixedStep) },
        [&](ISystemContext*) { seen = *GetResource<FixedStepState>(env.world.Get()); }, { "make", "balls.gather" });
    const SystemId alone = env.Add("alone", Phase::BeginFrame, {}, [&](ISystemContext*)
    {
        created[1] = CreateScheduler(env.world.Get(), env.jobs.Get(), &refused);
        EXPECT(!refused);
    }, {}, {}, SystemFlags::Exclusive);
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::PreSimulation) == nvrhi::FS_OK);
    CHECK(created[0] == nvrhi::FE_INVALID_ARGS && created[1] == nvrhi::FE_INVALID_ARGS);
    CHECK(std::memcmp(&seen, &before, sizeof(before)) == 0);
    CHECK(GetResource<FrameTime>(env.world.Get())->frameIndex != 77);
    for (SystemId system : { make, read, alone })
        CHECK(env.scheduler->RemoveSystem(system) == nvrhi::FS_OK);
    bystander = nullptr;
    // The new scheduler may be there before the old one goes.
    nvrhi::AutoPtr<IScheduler> next;
    CHECK(CreateScheduler(env.world.Get(), env.jobs.Get(), &next) == nvrhi::FS_OK);
    CHECK(std::memcmp(GetResource<FixedStepState>(env.world.Get()), &zero, sizeof(zero)) == 0);
    env.scheduler = nullptr;
    env.scheduler = next;
    CHECK(std::memcmp(GetResource<FixedStepState>(env.world.Get()), &zero, sizeof(zero)) == 0);
    CHECK(domain->states.size() == 20);   // nothing was delivered
    domain->states.clear();
    id = AddDomain(env, domain.Get());
    CHECK(domain->states.size() == 20 && env.Step(id).stepIndex == 0 && env.Step(id).fixedTimeStep == 0.25);
    env.Frame(0.3);
    CHECK(env.Step(id).stepIndex == 1 && Near(env.Step(id).accumulator, 0.05));
    CHECK(env.Step(SimulationDomainId(1)).fixedTimeStep == 0.0 && env.Reports() == 0);
}

// -------------------------------------------------------------------------------------------------- capture

// Balls in several archetypes; requests, some enabled.
static std::vector<Entity> MakeBalls(Env& env, uint32_t count)
{
    std::vector<Entity> all;
    auto add = [&](const std::vector<ComponentId>& components, uint32_t n)
    {
        for (Entity entity : env.Create(components, n))
        {
            const uint32_t k = uint32_t(all.size());
            *env.Edit<Pos>(entity) = Pos{ double(k % 17), 1.0 + double(k % 5) };
            env.Edit<Mass>(entity)->value = 1.0 + 0.125 * (k % 4);
            *env.Edit<Push>(entity) = Push{ double(k % 3) - 1.0, double(k % 7) };
            CHECK(env.world->SetEnabled(entity, env.push, k % 3 == 0) == nvrhi::FS_OK);
            all.push_back(entity);
        }
    };
    add({ env.ball, env.body, env.pos, env.mass, env.push, env.vel }, count / 2);
    add({ env.ball, env.body, env.pos, env.mass, env.push }, count / 4);   // no Vel
    add({ env.ball, env.body, env.pos, env.mass, env.push, env.vel, env.blue }, count - count / 2 - count / 4);
    return all;
}

// Save, 20 steps, load, 20 steps: the same state after every frame, engine bytes and enabled bits included.
static void test_capture()
{
    Env env(4);
    const std::vector<Entity> balls = MakeBalls(env, 1200);
    nvrhi::AutoPtr<BallDomain> domain = MAKE_RC_OBJ_PTR(BallDomain, env);
    const SimulationDomainId id = AddDomain(env, domain.Get());
    env.Frame(0.3);
    auto script = [&](uint32_t frame)
    {
        for (uint32_t i = frame; i < balls.size(); i += 37)
        {
            *env.Edit<Push>(balls[i]) = Push{ 0.5 * frame, 1.0 };
            CHECK(env.world->SetEnabled(balls[i], env.push, true) == nvrhi::FS_OK);
        }
        env.Frame(frame % 4 == 3 ? 0.55 : 0.2);   // 20 frames, at least 16 steps
    };
    for (uint32_t frame = 0; frame < 3; ++frame)
        script(frame);
    // Pending requests and half a step are in the save: their enabled bits and the accumulator must come back.
    for (uint32_t i = 5; i < balls.size(); i += 11)
        CHECK(env.world->SetEnabled(balls[i], env.push, true) == nvrhi::FS_OK);
    env.SetTime(0.3);   // planned, not run
    const std::vector<uint8_t> image = BallImage(env, domain.Get(), id);
    const std::vector<uint8_t> saved = env.Save();
    CHECK(BallImage(env, domain.Get(), id) == image);   // a save changes nothing
    const uint64_t savedSteps = env.Step(id).stepIndex;
    CHECK(env.Step(id).stepsPlanned >= 1 && env.Step(id).accumulator > 0.25);

    std::vector<std::vector<uint8_t>> images;
    CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
    for (uint32_t frame = 0; frame < 20; ++frame)
    {
        script(frame);
        images.push_back(BallImage(env, domain.Get(), id));
    }
    CHECK(env.Step(id).stepIndex >= savedSteps + 20 && images.back() != image);
    for (uint32_t i = 0; i < balls.size(); i += 13)
        CHECK(env.world->SetEnabled(balls[i], env.push, true) == nvrhi::FS_OK);
    for (uint32_t i = 5; i < balls.size(); i += 11)
        CHECK(env.world->SetEnabled(balls[i], env.push, false) == nvrhi::FS_OK);

    for (uint32_t repeat = 0; repeat < 2; ++repeat)
    {
        size_t position = 0;
        std::vector<uint8_t> longer = saved;
        longer.insert(longer.end(), 100, uint8_t(0xee));   // what follows in the stream is not read
        CHECK(env.Load(longer, &position) == nvrhi::FS_OK && position == saved.size());
        CHECK(BallImage(env, domain.Get(), id) == image);
        // What a save after the load writes is what was loaded.
        CHECK(env.Save() == saved);
        // The plan of the save is run.
        CHECK(env.scheduler->RunPhases(Phase::BeginFrame, Phase::EndFrame) == nvrhi::FS_OK);
        for (uint32_t frame = 0; frame < 20; ++frame)
        {
            script(frame);
            CHECK(BallImage(env, domain.Get(), id) == images[frame]);
        }
    }
    CHECK(env.Reports() == 0);
}

// What is in a capture and what is not: write columns of outputs and inputs, with a new version, and the bits of
// Enableable tags; not what is declared Read, not what is not Relocatable, not entities outside the domain.
static void test_captured_columns()
{
    Env env;
    const std::vector<Entity> entities =
        env.Create({ env.ball, env.pos, env.vel, env.mass, env.body, env.push, env.awake, env.hooked, env.red }, 700);
    const Entity outside = env.Create({ env.pos }, 1)[0];
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("columns");
    domain->terms = { Term(env.ball, Access::Filter) };
    domain->inputs = { R(env.mass), W(env.push), W(env.awake), Filter(env.red) };
    domain->outputs = { W(env.pos), R(env.body), W(env.hooked), W(env.span), W(env.gauge), W(env.push) };
    const SimulationDomainId id = AddDomain(env, domain.Get());
    env.Frame(6.5);   // 4 steps taken, 2 dropped, half a step left
    const FixedStepEntry time = env.Step(id);
    CHECK(time.stepsTaken == 4 && time.stepsDropped == 2 && time.accumulator == 0.5 && time.stepIndex == 4);
    auto set = [&](Entity entity, double value, bool enabled)
    {
        *env.Edit<Pos>(entity) = Pos{ value, -value };
        *env.Edit<Vel>(entity) = Vel{ value, 0.0 };
        env.Edit<Mass>(entity)->value = value;
        env.Edit<Body>(entity)->index = uint32_t(value);
        *env.Edit<Push>(entity) = Push{ value, value };
        env.Edit<Hooked>(entity)->value = uint32_t(value);
        env.Edit<Red>(entity)->value = uint32_t(value);
        CHECK(env.world->SetEnabled(entity, env.push, enabled) == nvrhi::FS_OK);
        CHECK(env.world->SetEnabled(entity, env.awake, !enabled) == nvrhi::FS_OK);
    };
    for (uint32_t i = 0; i < entities.size(); ++i)
        set(entities[i], 1.0 + i, i % 2 == 0);
    *env.Edit<Pos>(outside) = Pos{ 5.0, 5.0 };
    GetResource<Gauge>(env.world.Get())->value = 11;
    const std::vector<uint8_t> saved = env.Save();
    // The ids, Pos and Push (declared twice, captured once), and bits.
    CHECK(saved.size() > 700 * (8 + 16 + 16) && saved.size() < 700 * (8 + 16 + 16) + 1000);
    for (uint32_t i = 0; i < entities.size(); ++i)
        set(entities[i], 10000.0 + i, i % 2 != 0);
    *env.Edit<Pos>(outside) = Pos{ 6.0, 6.0 };
    GetResource<Gauge>(env.world.Get())->value = 12;

    Query<Changed<Pos>> changedPos(env.world.Get(), "changed pos");
    Query<Changed<Push>> changedPush(env.world.Get(), "changed push");
    Query<Changed<Mass>> changedMass(env.world.Get(), "changed mass");
    uint32_t changed[3] = {};
    env.Add("watch", Phase::EndFrame, { R(env.pos), R(env.push), R(env.mass) }, [&](ISystemContext* context)
    {
        changed[0] = changed[1] = changed[2] = 0;
        changedPos.ForEachChunk(context, [&](const ChunkView&, const Pos*) { ++changed[0]; });
        changedPush.ForEachChunk(context, [&](const ChunkView&, const Push*) { ++changed[1]; });
        changedMass.ForEachChunk(context, [&](const ChunkView&, const Mass*) { ++changed[2]; });
    });
    Query<With<Ball>> balls(env.world.Get(), "balls");
    const uint32_t chunkCount = balls.Get()->GetChunkCount();
    CHECK(chunkCount >= 2);
    env.Frame(0.0);
    env.Frame(0.0);
    CHECK(changed[0] == 0 && changed[1] == 0 && changed[2] == 0);
    const uint64_t version = env.world->GetVersion();
    CHECK(std::memcmp(&time, &env.Step(id), sizeof(time)) != 0);
    CHECK(env.Load(saved) == nvrhi::FS_OK);
    CHECK(std::memcmp(&time, &env.Step(id), sizeof(time)) == 0);
    CHECK(env.world->GetVersion() == version + 1 && env.ColumnVersion(entities[0], env.pos) == version + 1 &&
          env.ColumnVersion(entities[0], env.awake) == version + 1 && env.ColumnVersion(entities[0], env.mass) < version);
    for (uint32_t i = 0; i < entities.size(); ++i)
    {
        const Entity entity = entities[i];
        CHECK(env.Get<Pos>(entity)->x == 1.0 + i && env.Get<Push>(entity)->y == 1.0 + i);
        CHECK(env.world->IsEnabled(entity, env.push) == (i % 2 == 0) &&
              env.world->IsEnabled(entity, env.awake) == (i % 2 != 0));
        CHECK(env.Get<Mass>(entity)->value == 10000.0 + i && env.Get<Body>(entity)->index == 10000 + i &&
              env.Get<Vel>(entity)->x == 10000.0 + i && env.Get<Hooked>(entity)->value == 10000 + i &&
              env.Get<Red>(entity)->value == 10000 + i);
    }
    CHECK(env.Get<Pos>(outside)->x == 6.0 && GetResource<Gauge>(env.world.Get())->value == 12);
    env.Frame(0.0);
    CHECK(changed[0] == chunkCount && changed[1] == chunkCount && changed[2] == 0);

    // The same entities in the same order match, also when they lie in other chunks: all of them move to an archetype
    // with larger rows.
    const uint32_t chunksBefore = balls.Get()->GetChunkCount();
    for (Entity entity : entities)
        CHECK(env.world->AddComponent(entity, env.pad, nullptr) == nvrhi::FS_OK);
    CHECK(Flatten(Chunks(balls.Get())) == entities && balls.Get()->GetChunkCount() != chunksBefore);
    for (uint32_t i = 0; i < entities.size(); ++i)
        set(entities[i], 20000.0 + i, i % 3 == 0);
    CHECK(env.Load(saved) == nvrhi::FS_OK);
    for (uint32_t i = 0; i < entities.size(); ++i)
        CHECK(env.Get<Pos>(entities[i])->x == 1.0 + i && env.Get<Pos>(entities[i])->y == -1.0 - i &&
              env.world->IsEnabled(entities[i], env.push) == (i % 2 == 0) &&
              env.Get<Mass>(entities[i])->value == 20000.0 + i);
    CHECK(env.Reports() == 0);
}

// Streams that do not fit the world, and broken ones, change nothing.
static void test_refused_loads()
{
    Env env;
    const std::vector<Entity> balls = MakeBalls(env, 40);
    nvrhi::AutoPtr<BallDomain> domain = MAKE_RC_OBJ_PTR(BallDomain, env);
    const SimulationDomainId id = AddDomain(env, domain.Get());
    nvrhi::AutoPtr<ScriptDomain> second = MakeDomain("second", 0.5, 4);
    uint32_t restores = 0;
    uint64_t engine = 42;
    second->capture = [&](IStateWriter* writer) { return writer->Write(&engine, sizeof(engine)); };
    second->restore = [&](IStateReader* reader)
    {
        ++restores;
        return reader->Read(sizeof(engine), &engine);
    };
    SimulationDomainId secondId = AddDomain(env, second.Get());
    for (uint32_t frame = 0; frame < 3; ++frame)
        env.Frame(0.3);
    const std::vector<uint8_t> saved = env.Save();
    engine = 43;
    for (uint32_t frame = 0; frame < 5; ++frame)
        env.Frame(0.4);
    auto state = [&]()
    {
        std::vector<uint8_t> bytes = BallImage(env, domain.Get(), id);
        auto append = [&bytes](const void* data, size_t size)
        {
            const uint8_t* begin = static_cast<const uint8_t*>(data);
            bytes.insert(bytes.end(), begin, begin + size);
        };
        const uint64_t version = env.world->GetVersion();
        append(GetResource<FixedStepState>(env.world.Get()), sizeof(FixedStepState));
        append(&engine, sizeof(engine));
        append(&version, sizeof(version));
        return bytes;
    };
    std::vector<uint8_t> before = state();
    auto refused = [&](const std::vector<uint8_t>& bytes, nvrhi::FRESULT expected)
    {
        const nvrhi::FRESULT result = env.Load(bytes);
        CHECK(result == expected);
        CHECK(state() == before && restores == 0);
    };
    // The working time as well, not only its copy: another SetFrameTime gives what it would have given.
    auto planOf = [&]() { env.SetTime(0.0); return env.Step(id); };
    const FixedStepEntry plan = planOf();
    before = state();

    // Truncated at every length; a byte changed.
    for (size_t length = 0; length < saved.size(); length += length < 300 ? 1 : 97)
        refused(std::vector<uint8_t>(saved.begin(), saved.begin() + length), nvrhi::FE_INVALID_ARGS);
    for (size_t at = 0; at < saved.size(); at += at < 300 ? 1 : 61)
    {
        std::vector<uint8_t> corrupt = saved;
        corrupt[at] ^= 0x10;
        refused(corrupt, at >= 8 && at < 12 ? nvrhi::FE_UNSUPPORTED : nvrhi::FE_INVALID_ARGS);
    }
    const FixedStepEntry planAfter = planOf();
    CHECK(std::memcmp(&plan, &planAfter, sizeof(plan)) == 0);

    // Another entity set: one more, one less, the same set in another order, another set of the same size.
    const Entity extra = env.Create({ env.ball, env.body, env.pos, env.mass, env.push }, 1)[0];
    before = state();
    refused(saved, nvrhi::FE_OUT_OF_DATE);
    CHECK(env.world->DestroyEntities(&extra, 1) == nvrhi::FS_OK);
    before = state();
    CHECK(env.Load(saved) == nvrhi::FS_OK && restores == 1 && engine == 42);   // the set of the save again
    restores = 0;
    CHECK(env.world->AddComponent(balls[2], env.red, nullptr) == nvrhi::FS_OK);   // in another archetype, so elsewhere
    before = state();
    refused(saved, nvrhi::FE_OUT_OF_DATE);
    CHECK(env.world->DestroyEntities(&balls[2], 1) == nvrhi::FS_OK);
    before = state();
    refused(saved, nvrhi::FE_OUT_OF_DATE);
    env.Create({ env.ball, env.body, env.pos, env.mass, env.push }, 1);
    before = state();
    refused(saved, nvrhi::FE_OUT_OF_DATE);

    // Other domains.
    const std::vector<uint8_t> newer = env.Save();
    nvrhi::AutoPtr<ScriptDomain> third = MakeDomain("third", 0.5, 4);
    const SimulationDomainId thirdId = AddDomain(env, third.Get());
    before = state();
    refused(newer, nvrhi::FE_OUT_OF_DATE);                       // one domain more
    CHECK(env.scheduler->RemoveSimulationDomain(secondId) == nvrhi::FS_OK);
    before = state();
    refused(newer, nvrhi::FE_OUT_OF_DATE);                       // as many, another id
    CHECK(env.scheduler->RemoveSimulationDomain(thirdId) == nvrhi::FS_OK);
    before = state();
    refused(newer, nvrhi::FE_OUT_OF_DATE);                       // one less
    // The same domain again is the same, under its id only.
    nvrhi::AutoPtr<ScriptDomain> other = MakeDomain("other", 0.5, 4);
    SimulationDomainId otherId = AddDomain(env, other.Get());
    CHECK(otherId != secondId);                                  // the id freed last is taken first
    const SimulationDomainId again = AddDomain(env, second.Get());
    CHECK(again == secondId && env.scheduler->RemoveSimulationDomain(otherId) == nvrhi::FS_OK);
    CHECK(env.Load(newer) == nvrhi::FS_OK && restores == 1);
    restores = 0;
    CHECK(env.scheduler->RemoveSimulationDomain(again) == nvrhi::FS_OK);
    otherId = AddDomain(env, other.Get());
    CHECK(otherId == secondId && AddDomain(env, second.Get()) != secondId);
    CHECK(env.scheduler->RemoveSimulationDomain(otherId) == nvrhi::FS_OK);
    before = state();
    refused(newer, nvrhi::FE_OUT_OF_DATE);
}

// Identity of the domains and shape of the rows, one difference at a time.
static void test_refused_loads_identity()
{
    auto run = [](const std::function<void(ScriptDomain&, Env&)>& change, nvrhi::FRESULT expected)
    {
        Env env;
        const std::vector<Entity> entities = env.Create({ env.ball, env.pos, env.vel, env.mass }, 60);
        nvrhi::AutoPtr<ScriptDomain> first = MakeDomain("first", 0.5, 4);
        first->terms = { Term(env.ball, Access::Filter) };
        first->outputs = { W(env.pos) };
        const SimulationDomainId firstId = AddDomain(env, first.Get());
        nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("domain", 0.5, 4);
        domain->terms = { Term(env.ball, Access::Filter) };
        domain->outputs = { W(env.vel) };
        const SimulationDomainId id = AddDomain(env, domain.Get());
        env.Frame(0.75);
        const std::vector<uint8_t> saved = env.Save();
        env.Frame(0.75);
        for (Entity entity : entities)
            *env.Edit<Pos>(entity) = Pos{ 3.0, 4.0 };
        CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
        change(*domain.Get(), env);
        CHECK(AddDomain(env, domain.Get()) == id);
        const FixedStepEntry before = env.Step(firstId);
        CHECK(env.Load(saved) == expected);
        const bool same = std::memcmp(&before, &env.Step(firstId), sizeof(before)) == 0;
        CHECK(same == (expected != nvrhi::FS_OK));
        for (Entity entity : entities)
            CHECK(!env.world->IsAlive(entity) || (env.Get<Pos>(entity)->x == 3.0) == (expected != nvrhi::FS_OK));
        // The working time of the first domain, not only its copy.
        env.SetTime(0.0);
        CHECK(env.Step(firstId).stepIndex == (expected == nvrhi::FS_OK ? 1 : 3));
    };
    run([](ScriptDomain&, Env&) { }, nvrhi::FS_OK);
    run([](ScriptDomain& domain, Env&) { domain.name = "nomain"; }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain& domain, Env&) { domain.name = "domain2"; }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain& domain, Env&) { domain.desc.fixedTimeStep = 0.25; }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain& domain, Env& env) { domain.outputs = { W(env.pos) }; }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain& domain, Env& env) { domain.outputs = { W(env.pos), W(env.vel) }; }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain& domain, Env& env) { domain.outputs = { R(env.vel) }; }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain& domain, Env& env) { domain.terms = { Term(env.blue, Access::Filter) }; }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain& domain, Env&) { domain.terms.clear(); }, nvrhi::FE_OUT_OF_DATE);
    // The same entities in the same order without the captured column.
    run([](ScriptDomain&, Env& env)
    {
        Query<With<Ball>> balls(env.world.Get(), "balls");
        for (Entity entity : Flatten(Chunks(balls.Get())))
            CHECK(env.world->RemoveComponent(entity, env.vel) == nvrhi::FS_OK);
    }, nvrhi::FE_OUT_OF_DATE);
    // ... and only the last ten without it: they move to a later archetype in their order.
    run([](ScriptDomain&, Env& env)
    {
        Query<With<Ball>> balls(env.world.Get(), "balls");
        const std::vector<Entity> entities = Flatten(Chunks(balls.Get()));
        for (size_t i = 50; i < entities.size(); ++i)
            CHECK(env.world->RemoveComponent(entities[i], env.vel) == nvrhi::FS_OK);
        CHECK(Flatten(Chunks(balls.Get())) == entities);
    }, nvrhi::FE_OUT_OF_DATE);
    // As many entities of the same shape, but not the same ones; the same ones in another order.
    run([](ScriptDomain&, Env& env)
    {
        Query<With<Ball>> balls(env.world.Get(), "balls");
        const Entity last = Flatten(Chunks(balls.Get())).back();
        CHECK(env.world->DestroyEntities(&last, 1) == nvrhi::FS_OK);
        env.Create({ env.ball, env.pos, env.vel, env.mass }, 1);
        for (Entity entity : Flatten(Chunks(balls.Get())))
            *env.Edit<Pos>(entity) = Pos{ 3.0, 4.0 };
    }, nvrhi::FE_OUT_OF_DATE);
    run([](ScriptDomain&, Env& env)
    {
        Query<With<Ball>> balls(env.world.Get(), "balls");
        const std::vector<Entity> entities = Flatten(Chunks(balls.Get()));
        CHECK(env.world->AddComponent(entities[0], env.red, nullptr) == nvrhi::FS_OK);
        CHECK(env.world->RemoveComponent(entities[0], env.red) == nvrhi::FS_OK);
        std::vector<Entity> moved = Flatten(Chunks(balls.Get()));
        CHECK(moved != entities);
        std::sort(moved.begin(), moved.end());
        CHECK(std::is_permutation(moved.begin(), moved.end(), entities.begin()));
    }, nvrhi::FE_OUT_OF_DATE);
    // The step limit and the accesses that are not captured may differ.
    run([](ScriptDomain& domain, Env& env)
    {
        domain.desc.maxStepsPerFrame = 9;
        domain.inputs = { R(env.mass) };
    }, nvrhi::FS_OK);
}

// A domain that cannot capture fails the save before anything is written; a failing RestoreState ends the load.
static void test_failing_captures()
{
    Env env;
    const std::vector<Entity> entities = env.Create({ env.ball, env.pos, env.vel }, 30);
    nvrhi::AutoPtr<ScriptDomain> domains[3] = { MakeDomain("a", 0.5), MakeDomain("b", 0.5), MakeDomain("c", 0.5) };
    SimulationDomainId ids[3];
    uint32_t values[3] = { 1, 2, 3 };
    nvrhi::FRESULT captureResult = nvrhi::FS_OK;
    nvrhi::FRESULT restoreResult[3] = { nvrhi::FS_OK, nvrhi::FS_OK, nvrhi::FS_OK };
    std::vector<uint32_t> restored;
    nvrhi::FRESULT pastEnd = nvrhi::FS_OK;
    nvrhi::FRESULT nested = nvrhi::FS_OK;
    nvrhi::AutoPtr<IStateWriter> kept;
    nvrhi::AutoPtr<IStateReader> keptReader;
    nvrhi::FRESULT nestedInRestore[3] = {};
    std::vector<uint8_t> saved;
    for (uint32_t d = 0; d < 3; ++d)
    {
        domains[d]->terms = { Term(env.ball, Access::Filter) };
        domains[d]->outputs = { W(d == 0 ? env.pos : env.vel) };
        domains[d]->capture = [&, d](IStateWriter* writer)
        {
            if (d == 2 && NVRHI_FAILED(captureResult))
                return captureResult;
            if (d == 0)
            {
                kept = writer;
                nested = env.scheduler->RemoveSimulationDomain(ids[0]);
            }
            return writer->Write(&values[d], sizeof(values[d]));
        };
        domains[d]->restore = [&, d](IStateReader* reader)
        {
            restored.push_back(d);
            if (d == 0)
            {
                nestedInRestore[0] = env.scheduler->RemoveSimulationDomain(ids[2]);
                nestedInRestore[1] = env.scheduler->RunPhase(Phase::EndFrame);
                nestedInRestore[2] = env.Load(saved);
            }
            keptReader = reader;
            if (NVRHI_FAILED(restoreResult[d]))
                return restoreResult[d];
            const nvrhi::FRESULT result = reader->Read(sizeof(values[d]), &values[d]);
            uint8_t more = 0;
            pastEnd = reader->Read(1, &more);
            return result;
        };
        ids[d] = AddDomain(env, domains[d].Get());
    }
    env.SetTime(1.0);   // planned, not run
    const uint64_t hash = env.world->ComputeHash();
    for (nvrhi::FRESULT failure : { nvrhi::FE_UNSUPPORTED, nvrhi::FE_GENERIC_ERROR })
    {
        captureResult = failure;
        CHECK(env.Save(failure).empty());
    }
    captureResult = nvrhi::FS_OK;
    saved = env.Save();
    CHECK(!saved.empty() && env.world->ComputeHash() == hash && nested == nvrhi::FE_INVALID_ARGS);
    // A writer that was kept writes nowhere.
    CHECK(kept && kept->Write(&hash, sizeof(hash)) == nvrhi::FE_INVALID_ARGS);
    kept = nullptr;
    CHECK(env.scheduler->SaveSimulationState(nullptr) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.scheduler->LoadSimulationState(nullptr) == nvrhi::FE_INVALID_ARGS);

    for (uint32_t failing = 0; failing < 3; ++failing)
    {
        for (uint32_t d = 0; d < 3; ++d)
        {
            restoreResult[d] = d == failing ? nvrhi::FE_NOT_FOUND : nvrhi::FS_OK;
            values[d] = 100 + d;
        }
        for (Entity entity : entities)
        {
            *env.Edit<Pos>(entity) = Pos{ 9.0, 9.0 };
            *env.Edit<Vel>(entity) = Vel{ 9.0, 9.0 };
        }
        env.Frame(0.5);
        CHECK(env.Step(ids[0]).stepIndex >= 1);
        restored.clear();
        CHECK(env.Load(saved) == nvrhi::FE_NOT_FOUND);
        for (nvrhi::FRESULT result : nestedInRestore)
            CHECK(result == nvrhi::FE_INVALID_ARGS);
        // A reader that was kept reads nothing; the failing domain had not read its bytes.
        uint32_t stale = 0;
        CHECK(keptReader->Read(sizeof(stale), &stale) == nvrhi::FE_INVALID_ARGS && stale == 0);
        // The engines up to the failing one, in order; the columns and times of all of them.
        CHECK(restored.size() == failing + 1);
        for (uint32_t d = 0; d < 3; ++d)
        {
            CHECK(values[d] == (d < failing ? d + 1 : 100 + d));
            CHECK(env.Step(ids[d]).stepIndex == 0 && env.Step(ids[d]).stepsPlanned == 2 &&
                  env.Step(ids[d]).accumulator == 1.0);
        }
        for (Entity entity : entities)
            CHECK(env.Get<Pos>(entity)->x == 0.0 && env.Get<Vel>(entity)->x == 0.0);
    }
    restoreResult[2] = nvrhi::FS_OK;
    restoreResult[0] = restoreResult[1] = nvrhi::FS_OK;
    CHECK(env.Load(saved) == nvrhi::FS_OK && values[0] == 1 && values[1] == 2 && values[2] == 3);
    CHECK(pastEnd == nvrhi::FE_INVALID_ARGS);   // a domain reads its own bytes only
    // Refused inside a phase.
    nvrhi::FRESULT inPhase[4] = {};
    env.Add("inside", Phase::EndFrame, {}, [&](ISystemContext*)
    {
        nvrhi::AutoPtr<MemoryWriter> writer = MAKE_RC_OBJ_PTR(MemoryWriter);
        inPhase[0] = env.scheduler->SaveSimulationState(writer.Get());
        nvrhi::AutoPtr<MemoryReader> reader = MAKE_RC_OBJ_PTR(MemoryReader);
        reader->bytes = saved;
        inPhase[1] = env.scheduler->LoadSimulationState(reader.Get());
        EXPECT(reader->position == 0);
        SimulationDomainId none;
        inPhase[2] = env.scheduler->AddSimulationDomain(domains[0].Get(), none);
        inPhase[3] = env.scheduler->RemoveSimulationDomain(ids[0]);
    }, {}, {}, SystemFlags::Exclusive);
    env.Frame(0.0);
    for (nvrhi::FRESULT result : inPhase)
        CHECK(result == nvrhi::FE_INVALID_ARGS);
    // ... and of another scheduler on the world.
    nvrhi::AutoPtr<IScheduler> other;
    CHECK(CreateScheduler(env.world.Get(), env.jobs.Get(), &other) == nvrhi::FS_OK);
    CHECK(env.Load(saved) == nvrhi::FS_OK);   // it fits
    nvrhi::FRESULT inOther[5] = {};
    nvrhi::AutoPtr<ScriptDomain> late = MakeDomain("late");
    SystemDesc desc{};
    desc.name = "outer";
    desc.phase = Phase::EndFrame;
    desc.flags = SystemFlags::Exclusive;
    nvrhi::AutoPtr<ISystem> system;
    auto outer = [&](ISystemContext*)
    {
        SimulationDomainId none;
        nvrhi::AutoPtr<MemoryReader> reader = MAKE_RC_OBJ_PTR(MemoryReader);
        reader->bytes = saved;
        nvrhi::AutoPtr<MemoryWriter> writer = MAKE_RC_OBJ_PTR(MemoryWriter);
        inOther[0] = env.scheduler->RemoveSimulationDomain(ids[0]);
        inOther[1] = env.scheduler->AddSimulationDomain(late.Get(), none);
        inOther[2] = env.scheduler->SaveSimulationState(writer.Get());
        inOther[3] = env.scheduler->LoadSimulationState(reader.Get());
        inOther[4] = env.scheduler->RunPhase(Phase::BeginFrame);
        EXPECT(reader->position == 0 && writer->bytes.empty());
    };
    CHECK(CreateLambdaSystem(desc, outer, &system) == nvrhi::FS_OK);
    SystemId systemId;
    CHECK(other->AddSystem(system.Get(), systemId) == nvrhi::FS_OK);
    CHECK(other->RunPhase(Phase::EndFrame) == nvrhi::FS_OK);
    for (nvrhi::FRESULT result : inOther)
        CHECK(result == nvrhi::FE_INVALID_ARGS);
    other->RemoveSystem(systemId);
    CHECK(ReferenceCount(late.Get()) == 1);
}

// Commands recorded after a frame's last sync point are played back before a save and before a load.
static void test_pending_commands()
{
    Env env;
    Query<With<Ball>> balls(env.world.Get(), "balls");
    const std::vector<Entity> entities = env.Create({ env.ball, env.pos }, 10);
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("pending");
    domain->terms = { Term(env.ball, Access::Filter) };
    domain->outputs = { W(env.pos) };
    AddDomain(env, domain.Get());
    enum { None, Spawn, Move } action = None;
    env.Add("late", Phase::EndFrame, {}, [&](ISystemContext* context)
    {
        if (action == Spawn)
        {
            const ComponentId list[2] = { env.ball, env.pos };
            EntityCreateDesc desc{};
            desc.components = list;
            desc.componentCount = 2;
            desc.entityCount = 3;
            Entity placeholders[3];
            context->GetCommandBuffer()->CreateEntities(desc, 0, placeholders);
        }
        else if (action == Move)
        {
            const Pos value{ 77.0, 77.0 };
            for (Entity entity : entities)
                context->GetCommandBuffer()->SetComponent(entity, env.pos, &value, GetEntityIndex(entity));
        }
    });
    action = Spawn;
    env.Frame(1.0);
    action = None;
    CHECK(balls.CountEntities() == 10 && domain->addedEntities == 10);
    const std::vector<uint8_t> saved = env.Save();
    CHECK(balls.CountEntities() == 13 && domain->addedEntities == 13);
    env.Frame(1.0);
    CHECK(balls.CountEntities() == 13);
    CHECK(env.Load(saved) == nvrhi::FS_OK);   // the entities of the save are in it

    action = Move;
    env.Frame(1.0);
    action = None;
    CHECK(env.Get<Pos>(entities[0])->x == 0.0);
    CHECK(env.Load(saved) == nvrhi::FS_OK);
    env.Frame(1.0);                           // its sync point has nothing of the rewound frame left
    CHECK(env.Get<Pos>(entities[0])->x == 0.0);

    // A broken stream changes nothing, pending commands included.
    action = Move;
    env.Frame(1.0);
    action = None;
    std::vector<uint8_t> corrupt = saved;
    corrupt[saved.size() / 2] ^= 1;
    CHECK(env.Load(corrupt) == nvrhi::FE_INVALID_ARGS && env.Get<Pos>(entities[0])->x == 0.0);
    CHECK(env.Load({ saved.begin(), saved.end() - 1 }) == nvrhi::FE_INVALID_ARGS);
    CHECK(env.Get<Pos>(entities[0])->x == 0.0);
    env.Frame(1.0);
    CHECK(env.Get<Pos>(entities[0])->x == 77.0);
    CHECK(env.Reports() == 0);
}

// Streams with a right checksum that no save wrote, and times that do not fit the domain of now.
static void test_crafted_streams()
{
    Env env;
    nvrhi::AutoPtr<ScriptDomain> domain = MakeDomain("d", 0.5, 4);
    uint32_t restores = 0;
    domain->capture = [](IStateWriter* writer) { const uint32_t value = 5; return writer->Write(&value, sizeof(value)); };
    domain->restore = [&](IStateReader*) { ++restores; return nvrhi::FS_OK; };
    SimulationDomainId id = AddDomain(env, domain.Get());
    // The header (24), id, name length, the name, the time step; then the accumulator, the step index, the plan.
    constexpr size_t c_Accumulator = 24 + 4 + 4 + 1 + 8;
    constexpr size_t c_Planned = c_Accumulator + 8 + 8;
    auto patched = [](std::vector<uint8_t> bytes, size_t at, const void* value, size_t size)
    {
        std::memcpy(bytes.data() + at, value, size);
        const uint64_t checksum = ecs::Xxh64(bytes.data(), bytes.size() - 8);
        std::memcpy(bytes.data() + bytes.size() - 8, &checksum, 8);
        return bytes;
    };
    env.SetTime(3.25);   // 6 whole steps, 4 planned
    const std::vector<uint8_t> saved = env.Save();
    double accumulator = 0.0;
    uint32_t planned = 0;
    std::memcpy(&accumulator, saved.data() + c_Accumulator, 8);
    std::memcpy(&planned, saved.data() + c_Planned, 4);
    CHECK(accumulator == 3.25 && planned == 4 && patched(saved, c_Planned, &planned, 4) == saved);
    CHECK(env.Load(saved) == nvrhi::FS_OK && env.Save() == saved && restores == 1);
    restores = 0;
    env.Frame(0.0);
    CHECK(domain->steps == 4 && env.Step(id).accumulator == 0.25);
    const FixedStepEntry now = env.Step(id);
    auto refused = [&](const std::vector<uint8_t>& bytes, nvrhi::FRESULT expected)
    {
        CHECK(env.Load(bytes) == expected && restores == 0);
        CHECK(std::memcmp(&now, &env.Step(id), sizeof(now)) == 0);
    };
    for (double value : { -0.5, std::nan(""), std::numeric_limits<double>::infinity(), -1e300 })
        refused(patched(saved, c_Accumulator, &value, 8), nvrhi::FE_INVALID_ARGS);
    refused(patched(saved, 0, "NOTASIM!", 8), nvrhi::FE_INVALID_ARGS);
    const uint32_t version = 2;
    refused(patched(saved, 8, &version, 4), nvrhi::FE_UNSUPPORTED);
    // Payload behind the last domain.
    std::vector<uint8_t> longer = saved;
    longer.insert(longer.end() - 8, 16, uint8_t(0xee));
    const uint64_t payload = longer.size() - 24 - 8;
    refused(patched(longer, 16, &payload, 8), nvrhi::FE_INVALID_ARGS);

    // A plan that the accumulator does not hold is cut to it.
    const double little = 0.75;
    std::vector<uint8_t> crafted = patched(saved, c_Accumulator, &little, 8);
    planned = 0xffffffffu;
    crafted = patched(crafted, c_Planned, &planned, 4);
    domain->steps = 0;
    CHECK(env.Load(crafted) == nvrhi::FS_OK && env.Step(id).stepsPlanned == 1);
    CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK && domain->steps == 1);
    CHECK(env.Step(id).accumulator == 0.25 && env.Step(id).stepsDropped == 0);

    // A smaller limit than at the save: the plan is the limit, the rest is dropped. A larger one: the plan of the save.
    for (uint32_t limit : { 1u, 3u, 9u })
    {
        CHECK(env.scheduler->RemoveSimulationDomain(id) == nvrhi::FS_OK);
        domain->desc.maxStepsPerFrame = limit;
        id = AddDomain(env, domain.Get());
        domain->steps = 0;
        CHECK(env.Load(saved) == nvrhi::FS_OK);
        const uint32_t steps = std::min(limit, 4u);
        CHECK(env.Step(id).stepsPlanned == steps);
        CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK && domain->steps == steps);
        CHECK(env.Step(id).stepsTaken == steps && env.Step(id).stepsDropped == 6 - steps &&
              env.Step(id).accumulator == 0.25);
    }
    // A save and a load give the same bytes at every point of frames, with a time step that is no binary fraction.
    nvrhi::AutoPtr<ScriptDomain> tenth = MakeDomain("tenth", 0.1, 3);
    AddDomain(env, tenth.Get());
    for (uint32_t frame = 0; frame < 200; ++frame)
    {
        env.SetTime((frame % 7) * 0.1 / 3.0 + (frame % 5 == 0 ? 0.5 : 0.0));
        for (uint32_t phase = 0; phase < 2; ++phase)
        {
            const std::vector<uint8_t> bytes = env.Save();
            CHECK(env.Load(bytes) == nvrhi::FS_OK && env.Save() == bytes);
            CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
        }
    }
    CHECK(env.Reports() == 0);
}

// ---------------------------------------------------------------------------------------------- determinism

// The same events, indices and results at 1, 2 and 8 workers: creations, destructions and component changes recorded by
// parallel chunk loops, immediate ones between frames, requests, and the domain's own steps.
static void test_worker_counts()
{
    std::vector<uint64_t> referenceLog;
    std::vector<uint8_t> referenceImage;
    for (uint32_t workers : { 1u, 2u, 8u, 8u, 2u, 1u })
    {
        Env env(workers);
        MakeBalls(env, 1000);
        nvrhi::AutoPtr<BallDomain> domain = MAKE_RC_OBJ_PTR(BallDomain, env);
        const SimulationDomainId id = AddDomain(env, domain.Get());
        Query<Read<Pos>, With<Ball>> positions(env.world.Get(), "positions");
        uint32_t frame = 0;
        env.Add("churn", Phase::PreSimulation, { R(env.pos) }, [&](ISystemContext* context)
        {
            positions.ParallelForChunks(context, 1, [&](const ChunkView& view, const Pos* pos)
            {
                ICommandBuffer* commands = context->GetCommandBuffer();
                std::vector<Entity> gone;
                for (uint32_t row = 0; row < view.count; ++row)
                    if ((uint64_t(int64_t(pos[row].x * 16.0)) + GetEntityIndex(view.entities[row]) + frame) % 23 == 0)
                        gone.push_back(view.entities[row]);
                if (!gone.empty())
                    commands->DestroyEntities(gone.data(), uint32_t(gone.size()), MakeSortKey(0, view.chunkIndex, 0));
                const ComponentId list[5] = { env.ball, env.body, env.pos, env.mass, env.push };
                EntityCreateDesc desc{};
                desc.components = list;
                desc.componentCount = 5;
                desc.entityCount = 5 + view.chunkIndex % 3;
                Entity placeholders[8];
                commands->CreateEntities(desc, MakeSortKey(1, view.chunkIndex, 0), placeholders);
                commands->AddComponent(placeholders[0], env.vel, nullptr, MakeSortKey(2, view.chunkIndex, 0));
            });
        });
        for (; frame < 30; ++frame)
        {
            env.Create({ env.ball, env.body, env.pos, env.mass, env.push, env.blue }, 3);
            env.Frame(0.1 + 0.05 * (frame % 5));
        }
        const std::vector<uint8_t> image = BallImage(env, domain.Get(), id);
        CHECK(env.Reports() == 0 && env.scheduler->GetAmbiguityCount() == 0);
        CHECK(domain->log.size() > 1500);
        if (referenceLog.empty())
        {
            referenceLog = domain->log;
            referenceImage = image;
        }
        CHECK(domain->log == referenceLog && image == referenceImage);
    }
}

// ---------------------------------------------------------------------------------------------- allocations

static void test_steady_state_allocations()
{
    Env env(4);
    MakeBalls(env, 1000);
    nvrhi::AutoPtr<BallDomain> domain = MAKE_RC_OBJ_PTR(BallDomain, env);
    const SimulationDomainId id = AddDomain(env, domain.Get());
    nvrhi::AutoPtr<ScriptDomain> gpu = MakeDomain("gpu", 0.5, 4);
    gpu->desc.queue = SimulationQueue::Cuda;
    AddDomain(env, gpu.Get());
    uint64_t planned = 0;
    env.Add("reader", Phase::FixedSimulation, { R(env.fixedStep) },
            [&](ISystemContext*) { planned += env.Step(id).stepsPlanned; });
    for (uint32_t frame = 0; frame < 4; ++frame)
        env.Frame(0.5);
    const uint64_t news = g_NewCount.load();
    const uint64_t blocks = ecs::GetHeapAllocationCount();
    const uint64_t steps = env.Step(id).stepIndex;
    for (uint32_t frame = 0; frame < 10; ++frame)
        env.Frame(0.5);
    const uint64_t newsAfter = g_NewCount.load();
    const uint64_t blocksAfter = ecs::GetHeapAllocationCount();
    printf("test_ecs_simulation: steady state: %llu operator new, %llu ECS heap blocks in 10 frames (gather, 2 steps, "
           "scatter of %u entities)\n", (unsigned long long)(newsAfter - news),
           (unsigned long long)(blocksAfter - blocks), uint32_t(domain->states.size()));
    CHECK(newsAfter == news && blocksAfter == blocks);
    CHECK(env.Step(id).stepIndex == steps + 20 && planned == 28 && env.Reports() == 0);
    // The counter sees allocations when there are some.
    int* probe = new int[16];
    CHECK(g_NewCount.load() > newsAfter);
    delete[] probe;
}

// Not a ctest case: the cost of the views (--time-views).
static void time_views()
{
    Env env(8);
    MakeBalls(env, 100000);
    nvrhi::AutoPtr<BallDomain> domain = MAKE_RC_OBJ_PTR(BallDomain, env);
    AddDomain(env, domain.Get());
    std::vector<double> gather, scatter;
    for (uint32_t frame = 0; frame < 210; ++frame)
    {
        env.SetTime(0.25);
        const auto a = std::chrono::steady_clock::now();
        CHECK(env.scheduler->RunPhase(Phase::PreSimulation) == nvrhi::FS_OK);
        const auto b = std::chrono::steady_clock::now();
        CHECK(env.scheduler->RunPhase(Phase::FixedSimulation) == nvrhi::FS_OK);
        const auto c = std::chrono::steady_clock::now();
        CHECK(env.scheduler->RunPhase(Phase::PostSimulation) == nvrhi::FS_OK);
        const auto d = std::chrono::steady_clock::now();
        if (frame < 10)
            continue;
        gather.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        scatter.push_back(std::chrono::duration<double, std::milli>(d - c).count());
    }
    std::sort(gather.begin(), gather.end());
    std::sort(scatter.begin(), scatter.end());
    printf("time_views: 100000 entities, medians of 200 frames: gather %.3f ms, scatter %.3f ms\n",
           gather[gather.size() / 2], scatter[scatter.size() / 2]);
}

int main(int argc, char** argv)
{
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "views_align", &test_views_align },
        { "paging", &test_paging },
        { "change_versions", &test_change_versions },
        { "refused_views", &test_refused_views },
        { "own_query", &test_own_query },
        { "lifetime_batches", &test_lifetime_batches },
        { "changes_structure", &test_changes_structure },
        { "callback_context", &test_callback_context },
        { "commands_in_add_and_remove", &test_commands_in_add_and_remove },
        { "references", &test_references },
        { "limits", &test_limits },
        { "queues", &test_queues },
        { "planned_steps", &test_planned_steps },
        { "publication", &test_publication },
        { "parallel_steps", &test_parallel_steps },
        { "failed_steps", &test_failed_steps },
        { "scheduler_recreated", &test_scheduler_recreated },
        { "capture", &test_capture },
        { "captured_columns", &test_captured_columns },
        { "refused_loads", &test_refused_loads },
        { "refused_loads_identity", &test_refused_loads_identity },
        { "failing_captures", &test_failing_captures },
        { "pending_commands", &test_pending_commands },
        { "crafted_streams", &test_crafted_streams },
        { "worker_counts", &test_worker_counts },
        { "steady_state_allocations", &test_steady_state_allocations },
    };
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    try
    {
        if (argc > 1 && std::strcmp(argv[1], "--time-views") == 0)
        {
            time_views();
            return 0;
        }
        for (const Test& test : tests)
        {
            if (argc > 1 && std::strcmp(argv[1], test.name) != 0)
                continue;
            test.run();
            CHECK(g_Errors.load() == 0);
            CHECK(ecs::GetHeapLiveBlockCount() == blocks);   // every world, buffer, query and scheduler is gone
            printf("test_ecs_simulation: %s passed\n", test.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
