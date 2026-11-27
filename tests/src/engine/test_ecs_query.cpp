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

// ECS queries (aggregate docs/plans/2026-11-21-donut-ecs.md, step E1.2; design 4.4, 4.5, 6.1): term validation,
// matching and chunk order against a brute-force model over seeded random worlds, change detection, enabled bits,
// sparse filters, Shared and Chunk components, buffers, resources, the cached query's lifetime, every typed term and
// ForEach form, determinism, steady-state allocations and a performance smoke. The parallel forms run on a test job
// system that splits ranges as JobSystem.h specifies (E2 brings the real one). The world's internals are read through
// the private headers (src/engine/EcsWorldImpl.h, EcsQueryImpl.h).

#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/tests/e7_switches.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsQueryImpl.h"
#include "EcsWorldImpl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test, and donut_engine, which links statically) is counted. The storage's
// and the queries' own blocks come from nvrhi_core's allocator, which ecs::GetHeapAllocationCount counts.
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

namespace query_test
{
    using donut::engine::ComponentFlags;
    using donut::engine::Entity;

    struct Position { float x, y, z; };
    DONUT_ECS_COMPONENT(Position, "399df5cc-893c-4743-9b11-2182ee5b37c5", DONUT_ECS_FIELDS(x, y, z));

    struct Velocity { float x, y, z; };
    DONUT_ECS_COMPONENT(Velocity, "6784c843-6a7d-4e8c-9b6e-b4cd96e6c8c4", DONUT_ECS_FIELDS(x, y, z));

    struct Id { uint64_t value; };
    DONUT_ECS_COMPONENT(Id, "80fea1d0-a68f-4b8e-a9da-1ff32c7af78a", DONUT_ECS_FIELDS(value));

    // A row over 128 bytes: the large chunk (D6).
    struct Big { uint8_t bytes[200]; };
    DONUT_ECS_COMPONENT(Big, "d41d1639-b918-4d0b-b4a5-763fee41e0a2", DONUT_ECS_FIELDS(bytes));

    struct Frozen { };
    DONUT_ECS_TAG(Frozen, "095c95e3-159b-49c8-9ded-8d9aee0710d7");

    struct Layer { uint32_t id; };
    DONUT_ECS_COMPONENT_EX(Layer, "cec9b4c1-ccde-4ebc-af2b-aa615e9560da", ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(id));

    struct ChunkBounds { float lo, hi; };
    DONUT_ECS_COMPONENT_EX(ChunkBounds, "d8edadc9-2174-49f6-9b4c-a85c5e971133", ComponentFlags::Chunk, 0,
                           DONUT_ECS_FIELDS(lo, hi));

    struct Selected { };
    DONUT_ECS_TAG_EX(Selected, "3f0f4ae9-b78b-45f2-8bd0-98e798bde62f", ComponentFlags::Sparse);

    struct DebugColor { uint32_t rgba; };
    DONUT_ECS_COMPONENT_EX(DebugColor, "cd325a26-0263-4ae1-b0f1-c7938b2e844a", ComponentFlags::Sparse, 0,
                           DONUT_ECS_FIELDS(rgba));

    struct Toggle { int32_t value; };
    DONUT_ECS_COMPONENT_EX(Toggle, "aedd76de-655d-45f1-b7f9-e11a58fc6573", ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(value));

    struct Hidden { };
    DONUT_ECS_TAG_EX(Hidden, "d2ba19c3-faaa-46e8-b996-34bec4a79604", ComponentFlags::Enableable);

    struct Joint { Entity target; float weight; uint32_t pad; };
    DONUT_ECS_BUFFER(Joint, "b77df597-eb0d-4209-b2ba-b3d5e0daf92d", 4, DONUT_ECS_FIELDS(target, weight, pad));

    struct Settings { float gravity; uint32_t steps; };
    DONUT_ECS_COMPONENT_EX(Settings, "3ef1f7a2-e194-4115-b0f7-54918526cda3", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(gravity, steps));

    struct PerfA { float x, y, z, w; };
    DONUT_ECS_COMPONENT(PerfA, "fc855236-268d-4a40-939b-b3f86149dca1", DONUT_ECS_FIELDS(x, y, z, w));

    struct PerfB { float x, y, z, w; };
    DONUT_ECS_COMPONENT(PerfB, "59e744b9-a31b-4777-b1ca-9c98cb11b28b", DONUT_ECS_FIELDS(x, y, z, w));
}

using namespace donut::engine;
using namespace query_test;
namespace ecs = donut::engine::ecs;

namespace
{
    // xorshift64*: explicit seeds, no std:: distributions (plan conventions).
    struct Xorshift
    {
        uint64_t state;
        uint64_t Next()
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 2685821657736338717ull;
        }
        uint32_t Below(uint32_t n) { return uint32_t(Next() % n); }
    };

    struct Fnv
    {
        uint64_t hash = 0xcbf29ce484222325ull;
        void Add(const void* data, size_t size)
        {
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; ++i)
            {
                hash ^= bytes[i];
                hash *= 0x100000001b3ull;
            }
        }
        template <typename T> void Add(const T& value) { Add(&value, sizeof(value)); }
    };

    nvrhi::AutoPtr<IWorld> MakeWorld(bool validation, uint32_t chunkSize = 16 * 1024,
                                     uint32_t largeChunkSize = 64 * 1024, uint32_t initialEntityCapacity = 1024)
    {
        WorldDesc desc;
        desc.debugName = "test_ecs_query";
        desc.validation = validation;
        desc.chunkSize = chunkSize;
        desc.largeChunkSize = largeChunkSize;
        desc.initialEntityCapacity = initialEntityCapacity;
        nvrhi::AutoPtr<IWorld> world;
        CHECK(CreateWorld(desc, &world) == nvrhi::FS_OK);
        return world;
    }

    ecs::WorldImpl& Impl(IWorld* world)
    {
        ecs::WorldImpl* impl = ecs::GetWorldImpl(world);
        CHECK(impl != nullptr);
        return *impl;
    }

    ecs::QueryImpl& QImpl(IQuery* query)
    {
        ecs::QueryImpl* impl = donut::query_cast<ecs::QueryImpl>(query);
        CHECK(impl != nullptr);
        return *impl;
    }

    const ecs::EntityRecord& Record(IWorld* world, Entity entity)
    {
        const ecs::EntityRecord* record = Impl(world).FindAlive(entity);
        CHECK(record != nullptr);
        return *record;
    }

    // (archetype, chunk) of a live entity, ordered as queries order chunks.
    uint64_t KeyOf(IWorld* world, Entity entity)
    {
        const ecs::EntityRecord& record = Record(world, entity);
        return (uint64_t(record.archetype) << 32) | record.chunk;
    }

    uint64_t ColumnVersion(IWorld* world, Entity entity, ComponentId component)
    {
        const ecs::EntityRecord& record = Record(world, entity);
        const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(record.archetype);
        const uint32_t column = archetype.FindColumn(component);
        CHECK(column != ecs::c_NoColumn);
        return archetype.Versions(record.chunk)[column];
    }

    QueryTerm Term(ComponentId component, TermOp op, Access access)
    {
        return QueryTerm{ component, op, access, {} };
    }

    nvrhi::AutoPtr<IQuery> MakeQuery(IWorld* world, const std::vector<QueryTerm>& terms, const char* name = "test")
    {
        const QueryDesc desc{ terms.data(), uint32_t(terms.size()), 0, name };
        nvrhi::AutoPtr<IQuery> query;
        CHECK(world->CreateQuery(desc, &query) == nvrhi::FS_OK && query);
        return query;
    }

    // CreateQuery's result; the query, if any, is released at once.
    nvrhi::FRESULT TryQuery(IWorld* world, const std::vector<QueryTerm>& terms)
    {
        const QueryDesc desc{ terms.empty() ? nullptr : terms.data(), uint32_t(terms.size()), 0, "try" };
        IQuery* query = reinterpret_cast<IQuery*>(uintptr_t(64));   // must be overwritten
        const nvrhi::FRESULT result = world->CreateQuery(desc, &query);
        if (NVRHI_FAILED(result))
        {
            CHECK(query == nullptr);
        }
        else
        {
            CHECK(query != nullptr);
            query->Release();
        }
        return result;
    }

    template <typename... Terms>
    std::vector<QueryTerm> TermsOf(IWorld* world)
    {
        return { donut::engine::detail::MakeTerm<Terms>(world)... };
    }

    void CreateMany(IWorld* world, std::initializer_list<ComponentId> components, uint32_t count, Entity* out,
                    std::initializer_list<const void*> values = {})
    {
        const std::vector<ComponentId> list(components);
        const std::vector<const void*> valueList(values);
        CHECK(valueList.empty() || valueList.size() == list.size());
        EntityCreateDesc desc{};
        desc.components = list.data();
        desc.componentCount = uint32_t(list.size());
        desc.values = valueList.empty() ? nullptr : valueList.data();
        desc.entityCount = count;
        desc.transfer = ValueTransfer::Copy;
        CHECK(world->CreateEntities(desc, out) == nvrhi::FS_OK);
    }

    // The chunks a request hands out, as (archetype, chunk) keys in the order handed out.
    std::vector<uint64_t> Handed(IWorld* world, IQuery* query, const ChunkRequest& request)
    {
        std::vector<uint64_t> keys;
        ChunkView views[16];
        const uint32_t count = query->GetChunkCount();
        for (uint32_t first = 0; first < count; first += 16)
        {
            const uint32_t written = query->GetChunks(first, request, 16, views);
            for (uint32_t i = 0; i < written; ++i)
                keys.push_back(KeyOf(world, views[i].entities[0]));
        }
        return keys;
    }

    std::vector<uint64_t> Keys(IWorld* world, std::initializer_list<Entity> entities)
    {
        std::vector<uint64_t> keys;
        for (const Entity entity : entities)
            keys.push_back(KeyOf(world, entity));
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }

    std::vector<Entity> Sorted(std::vector<Entity> entities)
    {
        std::sort(entities.begin(), entities.end());
        return entities;
    }

    bool MaskBit(const uint64_t* mask, uint32_t row)
    {
        return ((mask[row >> 6] >> (row & 63)) & 1) != 0;
    }

    // ------------------------------------------------------------------------------------- test doubles

    // A job system for the parallel typed forms (IJobSystem comes in E2): the split of JobSystem.h (ranges of
    // grainSize, the last one shorter), run by `threads` threads that take ranges in turn, the caller among them.
    class TestJobSystem final : public nvrhi::ObjectImpl<IJobSystem>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(TestJobSystem)
        NVRHI_IMPLEMENTS_INTERFACE(IJobSystem)
        NVRHI_END_INTERFACE_TABLE()

        explicit TestJobSystem(uint32_t threads) : m_Threads(threads) { }

        uint32_t GetThreadCount() noexcept override { return m_Threads; }
        uint32_t GetCurrentWorkerIndex() noexcept override { return ~0u; }
        nvrhi::FRESULT Schedule(const JobDesc&, const JobHandle*, uint32_t, JobHandle& outJob) noexcept override
        {
            outJob = JobHandle::Null;
            return nvrhi::FE_NOT_IMPLEMENT;
        }
        void Wait(JobHandle) noexcept override { }
        bool IsComplete(JobHandle) noexcept override { return true; }
        void ParallelFor(uint32_t count, uint32_t grainSize, PFN_JobFunction function, void* context) noexcept override
        {
            const uint32_t grain = grainSize != 0 ? grainSize : (count != 0 ? count : 1);
            const uint32_t ranges = (count + grain - 1) / grain;
            std::atomic<uint32_t> next{ 0 };
            auto work = [&](uint32_t worker)
            {
                for (;;)
                {
                    const uint32_t range = next.fetch_add(1);
                    if (range >= ranges)
                        break;
                    const uint32_t begin = range * grain;
                    const uint32_t end = count - begin < grain ? count : begin + grain;
                    function(context, begin, end, worker);
                }
            };
            std::vector<std::thread> threads;
            for (uint32_t worker = 1; worker < m_Threads; ++worker)
                threads.emplace_back(work, worker);
            work(0);
            for (std::thread& thread : threads)
                thread.join();
            ++parallelForCalls;
        }

        uint32_t parallelForCalls = 0;

    private:
        uint32_t m_Threads;
    };

    // The context of a running system, with versions the test sets.
    class TestContext final : public nvrhi::ObjectImpl<ISystemContext>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(TestContext)
        NVRHI_IMPLEMENTS_INTERFACE(ISystemContext)
        NVRHI_END_INTERFACE_TABLE()

        TestContext(IWorld* world, IJobSystem* jobs) : m_World(world), m_Jobs(jobs) { }

        IWorld* GetWorld() noexcept override { return m_World; }
        ICommandBuffer* GetCommandBuffer() noexcept override { return nullptr; }
        IJobSystem* GetJobSystem() noexcept override { return m_Jobs; }
        uint64_t GetVersion() noexcept override { return version; }
        uint64_t GetLastRunVersion() noexcept override { return lastRunVersion; }
        const FrameTime& GetFrameTime() noexcept override { return m_Time; }
        SystemId GetSystemId() noexcept override { return SystemId(0); }

        // A new run: the last run's version becomes the since version.
        void NextRun()
        {
            lastRunVersion = version;
            version = m_World->AdvanceVersion();
        }

        uint64_t version = 0;
        uint64_t lastRunVersion = 0;

    private:
        IWorld* m_World;
        IJobSystem* m_Jobs;
        FrameTime m_Time{};
    };

    // ------------------------------------------------------------------------------------ registration

    struct Ids
    {
        ComponentId position, velocity, id, big, frozen, layer, chunkBounds, selected, color, toggle, hidden, joint;
        ComponentId settings;
    };

    Ids RegisterAll(IWorld* world)
    {
        Ids ids{};
        ids.position = RegisterComponent<Position>(world);
        ids.velocity = RegisterComponent<Velocity>(world);
        ids.id = RegisterComponent<Id>(world);
        ids.big = RegisterComponent<Big>(world);
        ids.frozen = RegisterComponent<Frozen>(world);
        ids.layer = RegisterComponent<Layer>(world);
        ids.chunkBounds = RegisterComponent<ChunkBounds>(world);
        ids.selected = RegisterComponent<Selected>(world);
        ids.color = RegisterComponent<DebugColor>(world);
        ids.toggle = RegisterComponent<Toggle>(world);
        ids.hidden = RegisterComponent<Hidden>(world);
        ids.joint = RegisterComponent<Joint>(world);
        const Settings initial{ 9.81f, 4 };
        CHECK(world->RegisterResource(ComponentTraits<Settings>::Desc(), &initial, ids.settings) == nvrhi::FS_OK);
        for (const ComponentId id : { ids.position, ids.velocity, ids.id, ids.big, ids.frozen, ids.layer,
                                      ids.chunkBounds, ids.selected, ids.color, ids.toggle, ids.hidden, ids.joint,
                                      ids.settings })
            CHECK(id != ComponentId::Invalid);
        return ids;
    }

    // -------------------------------------------------------------------------------------------- the model

    ComponentFlags FlagsOf(IWorld* world, ComponentId component)
    {
        const ComponentTypeDesc* desc = world->GetComponentTypeDesc(component);
        CHECK(desc != nullptr);
        return desc->flags;
    }

    bool IsSparse(IWorld* world, ComponentId component)
    {
        return !!(FlagsOf(world, component) & ComponentFlags::Sparse);
    }

    bool IsRequiredOp(TermOp op)
    {
        return op == TermOp::All || op == TermOp::Changed || op == TermOp::Added || op == TermOp::Enabled ||
               op == TermOp::Disabled;
    }

    // The rules of CreateQuery (EcsQuery.h, EcsTypes.h and the E1.2 decisions), spelled out on their own.
    bool ExpectValid(IWorld* world, const std::vector<QueryTerm>& terms)
    {
        if (terms.empty() || terms.size() > ecs::c_MaxQueryTerms)
            return false;
        for (const QueryTerm& term : terms)
        {
            if (uint8_t(term.op) > uint8_t(TermOp::Disabled) || uint8_t(term.access) > uint8_t(Access::Write))
                return false;
            const ComponentTypeDesc* desc = world->GetComponentTypeDesc(term.component);
            if (!desc)
                return false;
            const ComponentFlags flags = desc->flags;
            if (!!(flags & ComponentFlags::Resource))
                return false;
            if (term.op == TermOp::Without && term.access == Access::Write)
                return false;
            if (term.access == Access::Write && !!(flags & ComponentFlags::Shared))
                return false;
            if (!!(flags & ComponentFlags::Sparse) &&
                (IsDataTerm(term) || term.op == TermOp::Changed || term.op == TermOp::Added || term.op == TermOp::Any))
                return false;
            if ((term.op == TermOp::Enabled || term.op == TermOp::Disabled) && !(flags & ComponentFlags::Enableable))
                return false;
        }
        for (size_t i = 0; i < terms.size(); ++i)
        {
            for (size_t j = i + 1; j < terms.size(); ++j)
            {
                if (terms[i].component != terms[j].component)
                    continue;
                const TermOp a = terms[i].op;
                const TermOp b = terms[j].op;
                if ((a == TermOp::Without && IsRequiredOp(b)) || (b == TermOp::Without && IsRequiredOp(a)))
                    return false;
                if ((a == TermOp::Enabled && b == TermOp::Disabled) || (a == TermOp::Disabled && b == TermOp::Enabled))
                    return false;
            }
        }
        return true;
    }

    // Whether the entity's archetype matches, from HasComponent alone.
    bool ModelEntityMatches(IWorld* world, Entity entity, const std::vector<QueryTerm>& terms)
    {
        bool hasAny = false;
        bool anyPresent = false;
        for (const QueryTerm& term : terms)
        {
            if (IsSparse(world, term.component))
                continue;
            const bool has = world->HasComponent(entity, term.component);
            switch (term.op)
            {
            case TermOp::Optional:
                break;
            case TermOp::Without:
                if (has)
                    return false;
                break;
            case TermOp::Any:
                hasAny = true;
                anyPresent = anyPresent || has;
                break;
            default:
                if (!has)
                    return false;
                break;
            }
        }
        return !hasAny || anyPresent;
    }

    // Whether an archetype matches, from its component list.
    bool ModelArchetypeMatches(IWorld* world, const ecs::Archetype& archetype, const std::vector<QueryTerm>& terms)
    {
        bool hasAny = false;
        bool anyPresent = false;
        for (const QueryTerm& term : terms)
        {
            if (IsSparse(world, term.component))
                continue;
            const bool has =
                std::find(archetype.components.begin(), archetype.components.end(), term.component) !=
                archetype.components.end();
            if (term.op == TermOp::Without && has)
                return false;
            if (term.op == TermOp::Any)
            {
                hasAny = true;
                anyPresent = anyPresent || has;
            }
            else if (IsRequiredOp(term.op) && !has)
                return false;
        }
        return !hasAny || anyPresent;
    }

    // Changed and Added, from the entity's chunk versions in the storage.
    bool ModelChunkPasses(IWorld* world, Entity entity, const std::vector<QueryTerm>& terms, uint64_t since)
    {
        if (since == c_AllVersions)
            return true;
        const ecs::EntityRecord& record = Record(world, entity);
        const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(record.archetype);
        const uint64_t* versions = archetype.Versions(record.chunk);
        for (const QueryTerm& term : terms)
        {
            if (term.op == TermOp::Changed && versions[archetype.FindColumn(term.component)] <= since)
                return false;
            if (term.op == TermOp::Added && versions[archetype.AddedVersionSlot()] <= since)
                return false;
        }
        return true;
    }

    // The row filters of the typed ForEach, from IsEnabled and HasComponent.
    bool ModelRowPasses(IWorld* world, Entity entity, const std::vector<QueryTerm>& terms)
    {
        for (const QueryTerm& term : terms)
        {
            if (term.op == TermOp::Enabled && !world->IsEnabled(entity, term.component))
                return false;
            if (term.op == TermOp::Disabled && world->IsEnabled(entity, term.component))
                return false;
            if (IsSparse(world, term.component))
            {
                const bool has = world->HasComponent(entity, term.component);
                if (term.op == TermOp::All && !has)
                    return false;
                if (term.op == TermOp::Without && has)
                    return false;
            }
        }
        return true;
    }

    std::vector<Entity> ModelVisit(IWorld* world, const std::vector<Entity>& alive, const std::vector<QueryTerm>& terms,
                                   uint64_t since)
    {
        std::vector<Entity> visited;
        for (const Entity entity : alive)
            if (ModelEntityMatches(world, entity, terms) && ModelChunkPasses(world, entity, terms, since) &&
                ModelRowPasses(world, entity, terms))
                visited.push_back(entity);
        return Sorted(visited);
    }

    // The row filters as the typed layer applies them, from the view's masks.
    bool ViewRowPasses(IWorld* world, const std::vector<QueryTerm>& terms, const ChunkView& view, uint32_t row)
    {
        for (uint32_t t = 0; t < terms.size(); ++t)
        {
            const QueryTerm& term = terms[t];
            if (term.op == TermOp::Enabled && !view.IsRowEnabled(t, row))
                return false;
            if (term.op == TermOp::Disabled && view.IsRowEnabled(t, row))
                return false;
            if (IsSparse(world, term.component))
            {
                if (term.op == TermOp::All && !view.IsRowPresent(t, row))
                    return false;
                if (term.op == TermOp::Without && view.presenceMasks[t] && view.IsRowPresent(t, row))
                    return false;
            }
        }
        return true;
    }

    // Every column pointer and mask of one row, against the per-entity accessors.
    void CheckRow(IWorld* world, const std::vector<QueryTerm>& terms, const ChunkView& view, uint32_t row)
    {
        const Entity entity = view.entities[row];
        uint32_t data = 0;
        for (uint32_t t = 0; t < terms.size(); ++t)
        {
            const QueryTerm& term = terms[t];
            const ComponentTypeDesc& desc = *world->GetComponentTypeDesc(term.component);
            const bool sparse = !!(desc.flags & ComponentFlags::Sparse);
            const bool has = world->HasComponent(entity, term.component);
            if (IsDataTerm(term))
            {
                const uint8_t* column = static_cast<const uint8_t*>(view.columns[data++]);
                const void* value = world->GetComponent(entity, term.component);
                if (!has)
                {
                    CHECK(column == nullptr && value == nullptr);
                }
                else if (!!(desc.flags & (ComponentFlags::Shared | ComponentFlags::Chunk | ComponentFlags::Tag)))
                {
                    CHECK(column == value);   // the chunk's one value; a tag's address
                }
                else
                {
                    const size_t stride = !!(desc.flags & ComponentFlags::Buffer) ? sizeof(BufferHeader) : desc.size;
                    CHECK(column != nullptr && column + size_t(row) * stride == value);
                }
            }
            const uint64_t* enabled = view.enabledMasks[t];
            const bool enableable = has && !sparse && !!(desc.flags & ComponentFlags::Enableable);
            CHECK(enableable ? enabled != nullptr && MaskBit(enabled, row) == world->IsEnabled(entity, term.component)
                             : enabled == nullptr);
            const uint64_t* presence = view.presenceMasks[t];
            CHECK(sparse ? presence != nullptr && MaskBit(presence, row) == has : presence == nullptr);
        }
        CHECK(data == uint32_t(std::count_if(terms.begin(), terms.end(), [](const QueryTerm& term) {
            return IsDataTerm(term);
        })));
    }

    struct ChunkVersions
    {
        uint64_t key;
        std::vector<uint64_t> slots;
    };

    std::vector<ChunkVersions> SnapshotVersions(IWorld* world)
    {
        ecs::WorldImpl& impl = Impl(world);
        std::vector<ChunkVersions> out;
        for (uint32_t a = 0; a < impl.GetArchetypeStorageCount(); ++a)
        {
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(a);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const uint64_t* versions = archetype.Versions(c);
                out.push_back({ (uint64_t(a) << 32) | c,
                                std::vector<uint64_t>(versions, versions + archetype.columns.size() + 1) });
            }
        }
        return out;
    }

    // After a request: handed-out chunks have their Write columns stamped with max(old, writeVersion); nothing else
    // changed anywhere.
    void CheckStamps(IWorld* world, const std::vector<ChunkVersions>& before, std::vector<uint64_t> handed,
                     const std::vector<QueryTerm>& terms, uint64_t writeVersion)
    {
        std::sort(handed.begin(), handed.end());
        const std::vector<ChunkVersions> after = SnapshotVersions(world);
        CHECK(after.size() == before.size());
        ecs::WorldImpl& impl = Impl(world);
        for (size_t i = 0; i < after.size(); ++i)
        {
            CHECK(after[i].key == before[i].key);
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(uint32_t(after[i].key >> 32));
            const bool wasHanded = std::binary_search(handed.begin(), handed.end(), after[i].key);
            for (size_t s = 0; s < after[i].slots.size(); ++s)
            {
                uint64_t expected = before[i].slots[s];
                bool written = false;
                if (wasHanded && s < archetype.columns.size())
                    for (const QueryTerm& term : terms)
                        written = written ||
                                  (term.access == Access::Write && term.component == archetype.components[s]);
                if (written)
                    expected = std::max(expected, writeVersion);
                CHECK(after[i].slots[s] == expected);
            }
        }
    }

    // The non-empty chunks of the matched archetypes, in archetype-creation and chunk order.
    std::vector<uint64_t> ModelChunkList(IWorld* world, const std::vector<QueryTerm>& terms, uint32_t* outRows)
    {
        ecs::WorldImpl& impl = Impl(world);
        std::vector<uint64_t> keys;
        uint32_t rows = 0;
        for (uint32_t a = 0; a < impl.GetArchetypeStorageCount(); ++a)
        {
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(a);
            if (!ModelArchetypeMatches(world, archetype, terms))
                continue;
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                if (archetype.chunks[c].count == 0)
                    continue;
                keys.push_back((uint64_t(a) << 32) | c);
                rows += archetype.chunks[c].count;
            }
        }
        *outRows = rows;
        return keys;
    }

    // One raw query against the model: matching, chunk order and indices, Changed and Added, every pointer and mask,
    // the row filters, CountEntities, and the stamps of its Write terms.
    void CheckRawQuery(IWorld* world, const std::vector<Entity>& alive, const std::vector<QueryTerm>& terms,
                       uint64_t since, Xorshift& rng)
    {
        nvrhi::AutoPtr<IQuery> query = MakeQuery(world, terms, "model");
        CHECK(query->GetTermCount() == terms.size());
        CHECK(query->GetDataTermCount() == uint32_t(std::count_if(terms.begin(), terms.end(), [](const QueryTerm& t) {
            return IsDataTerm(t);
        })));

        // Expectations first: handing chunks out stamps them.
        const std::vector<Entity> expected = ModelVisit(world, alive, terms, since);
        uint32_t modelRows = 0;
        const std::vector<uint64_t> chunkList = ModelChunkList(world, terms, &modelRows);
        uint32_t matchedEntities = 0;
        for (const Entity entity : alive)
            matchedEntities += ModelEntityMatches(world, entity, terms) ? 1 : 0;
        CHECK(modelRows == matchedEntities);
        CHECK(query->CountEntities() == matchedEntities);
        CHECK(query->GetChunkCount() == chunkList.size());
        // Handed out: the chunks that pass Changed and Added and have a row that passes the row filters.
        std::vector<uint64_t> expectedHanded;
        for (const uint64_t key : chunkList)
        {
            const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(uint32_t(key >> 32));
            const uint32_t chunk = uint32_t(key);
            const Entity* rows = archetype.Entities(chunk);
            bool anyRow = false;
            for (uint32_t row = 0; row < archetype.chunks[chunk].count && !anyRow; ++row)
                anyRow = ModelRowPasses(world, rows[row], terms);
            if (anyRow && ModelChunkPasses(world, rows[0], terms, since))
                expectedHanded.push_back(key);
        }
        const std::vector<ChunkVersions> before = SnapshotVersions(world);
        const uint64_t writeVersion = world->GetVersion();

        const ChunkRequest request{ since, c_WorldVersion, nullptr };
        std::vector<Entity> visited;
        std::vector<uint64_t> handed;
        uint32_t previous = ~0u;
        ChunkView views[8];
        const uint32_t chunkCount = query->GetChunkCount();
        for (uint32_t first = 0; first < chunkCount;)
        {
            const uint32_t capacity = 1 + rng.Below(8);
            const uint32_t written = query->GetChunks(first, request, capacity, views);
            CHECK(written <= capacity);
            for (uint32_t i = 0; i < written; ++i)
            {
                const ChunkView& view = views[i];
                CHECK(view.chunkIndex >= first && view.chunkIndex < first + capacity);
                CHECK(previous == ~0u || view.chunkIndex > previous);
                previous = view.chunkIndex;
                CHECK(view.count > 0);
                const uint64_t key = KeyOf(world, view.entities[0]);
                CHECK(chunkList[view.chunkIndex] == key);   // the position among the matched chunks
                handed.push_back(key);
                const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(uint32_t(key >> 32));
                CHECK(view.count == archetype.chunks[uint32_t(key)].count);
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    const ecs::EntityRecord& record = Record(world, view.entities[row]);
                    CHECK(((uint64_t(record.archetype) << 32) | record.chunk) == key && record.row == row);
                    CheckRow(world, terms, view, row);
                    if (ViewRowPasses(world, terms, view, row))
                        visited.push_back(view.entities[row]);
                }
            }
            first += capacity;
        }
        CHECK(handed == expectedHanded);
        CHECK(Sorted(visited) == expected);
        CheckStamps(world, before, handed, terms, writeVersion);
    }

    // ------------------------------------------------------------------------------------- random worlds

    struct ScriptWorld
    {
        nvrhi::AutoPtr<IWorld> world;
        Ids ids{};
        std::vector<Entity> alive;
        uint64_t nextId = 1;
    };

    void RandomCreate(ScriptWorld& script, Xorshift& rng)
    {
        const Ids& ids = script.ids;
        const ComponentId candidates[] = { ids.position, ids.velocity, ids.id,     ids.big,    ids.frozen,
                                           ids.layer,    ids.chunkBounds, ids.toggle, ids.hidden, ids.joint };
        std::vector<ComponentId> components;
        for (const ComponentId component : candidates)
            if (rng.Below(100) < 35)
                components.push_back(component);
        const uint32_t count = 1 + rng.Below(24);
        std::vector<Position> positions(count);
        std::vector<Id> idValues(count);
        std::vector<Layer> layers(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            positions[i] = Position{ float(rng.Below(1000)), float(rng.Below(1000)), float(i) };
            idValues[i] = Id{ script.nextId++ };
            layers[i] = Layer{ rng.Below(3) };
        }
        std::vector<const void*> values(components.size(), nullptr);
        for (size_t i = 0; i < components.size(); ++i)
        {
            if (components[i] == ids.position)
                values[i] = positions.data();
            else if (components[i] == ids.id)
                values[i] = idValues.data();
            else if (components[i] == ids.layer)
                values[i] = layers.data();
        }
        EntityCreateDesc desc{};
        desc.components = components.data();
        desc.componentCount = uint32_t(components.size());
        desc.values = values.data();
        desc.entityCount = count;
        std::vector<Entity> created(count);
        CHECK(script.world->CreateEntities(desc, created.data()) == nvrhi::FS_OK);
        script.alive.insert(script.alive.end(), created.begin(), created.end());
    }

    // One random immediate operation; every one of them is valid, so validation stays silent.
    void RandomStep(ScriptWorld& script, Xorshift& rng)
    {
        IWorld* world = script.world;
        const Ids& ids = script.ids;
        std::vector<Entity>& alive = script.alive;
        const uint32_t operation = rng.Below(14);
        if (alive.empty() || operation < 2)
        {
            RandomCreate(script, rng);
            return;
        }
        const uint32_t pick = rng.Below(uint32_t(alive.size()));
        const Entity entity = alive[pick];
        const ComponentId archetypeKinds[] = { ids.position, ids.velocity, ids.id,     ids.big,    ids.frozen,
                                               ids.layer,    ids.chunkBounds, ids.toggle, ids.hidden, ids.joint };
        switch (operation)
        {
        case 2:
        {
            CHECK(world->DestroyEntities(&entity, 1) == nvrhi::FS_OK);
            alive[pick] = alive.back();
            alive.pop_back();
            break;
        }
        case 3:
        case 4:
        {
            const ComponentId component = archetypeKinds[rng.Below(10)];
            const Position position{ 1, 2, 3 };
            const Layer layer{ rng.Below(3) };
            const void* value = component == ids.position ? static_cast<const void*>(&position)
                              : component == ids.layer ? static_cast<const void*>(&layer) : nullptr;
            CHECK(world->AddComponent(entity, component, value) == nvrhi::FS_OK);
            break;
        }
        case 5:
        {
            const ComponentId component = archetypeKinds[rng.Below(10)];
            if (world->HasComponent(entity, component))
                CHECK(world->RemoveComponent(entity, component) == nvrhi::FS_OK);
            break;
        }
        case 6:
        {
            const ComponentId component = rng.Below(2) ? ids.toggle : ids.hidden;
            if (world->HasComponent(entity, component))
                CHECK(world->SetEnabled(entity, component, rng.Below(2) != 0) == nvrhi::FS_OK);
            break;
        }
        case 7:
        {
            const DebugColor color{ rng.Below(256) };
            const bool tag = rng.Below(2) != 0;
            CHECK(world->AddComponent(entity, tag ? ids.selected : ids.color, tag ? nullptr : &color) == nvrhi::FS_OK);
            break;
        }
        case 8:
        {
            const ComponentId component = rng.Below(2) ? ids.selected : ids.color;
            if (world->HasComponent(entity, component))
                CHECK(world->RemoveComponent(entity, component) == nvrhi::FS_OK);
            break;
        }
        case 9:
        case 10:
        {
            // A write at a version that has been handed out.
            const ComponentId writable[] = { ids.position, ids.velocity, ids.id,
                                             ids.toggle, ids.chunkBounds, ids.joint };
            const ComponentId component = writable[rng.Below(6)];
            const uint64_t version = 1 + rng.Next() % world->GetVersion();
            if (void* value = world->GetComponentForWrite(entity, component, version))
                (void)value;
            break;
        }
        case 11:
            world->AdvanceVersion();
            break;
        case 12:
            if (world->HasComponent(entity, ids.joint))
                CHECK(world->ResizeBuffer(entity, ids.joint, rng.Below(7), c_WorldVersion) == nvrhi::FS_OK);
            break;
        default:
        {
            Entity copies[3];
            const uint32_t count = 1 + rng.Below(3);
            CHECK(world->Instantiate(entity, count, copies) == nvrhi::FS_OK);
            alive.insert(alive.end(), copies, copies + count);
            break;
        }
        }
    }

    // A random term: mostly plausible for its component, sometimes anything at all.
    QueryTerm RandomTerm(IWorld* world, const Ids& ids, Xorshift& rng)
    {
        const ComponentId components[] = { ids.position, ids.velocity, ids.id, ids.big, ids.frozen, ids.layer,
                                           ids.chunkBounds, ids.selected, ids.color, ids.toggle, ids.hidden,
                                           ids.joint };
        if (rng.Below(8) == 0)
        {
            const uint32_t which = rng.Below(16);
            const ComponentId component = which == 0 ? ids.settings
                                        : which == 1 ? ComponentId(500 + rng.Below(4)) : components[rng.Below(12)];
            return Term(component, TermOp(rng.Below(9)), Access(rng.Below(4)));   // includes out-of-range values
        }
        const ComponentId component = components[rng.Below(12)];
        const ComponentFlags flags = FlagsOf(world, component);
        TermOp op;
        Access access;
        if (!!(flags & ComponentFlags::Sparse))
        {
            const TermOp ops[] = { TermOp::All, TermOp::Without, TermOp::Optional };
            op = ops[rng.Below(3)];
            access = Access::Filter;
        }
        else if (!!(flags & ComponentFlags::Enableable))
        {
            op = TermOp(rng.Below(8));
            access = Access(rng.Below(3));
        }
        else
        {
            const TermOp ops[] = { TermOp::All, TermOp::Any, TermOp::Optional, TermOp::Without, TermOp::Changed,
                                   TermOp::Added };
            op = ops[rng.Below(6)];
            access = Access(rng.Below(!!(flags & ComponentFlags::Shared) ? 2u : 3u));
        }
        if (op == TermOp::Without && access == Access::Write)
            access = Access::Filter;
        return Term(component, op, access);
    }

    // The typed queries of the model check: every term type, through ForEach and ForEachChunk.
    void CheckTypedQueries(IWorld* world, const Ids& ids, const std::vector<Entity>& alive, uint64_t since)
    {
        const ChunkRequest all{ c_AllVersions, c_WorldVersion, nullptr };
        const ChunkRequest recent{ since, c_WorldVersion, nullptr };
        {
            using Q = Query<Read<Position>, Write<Velocity>, Without<Frozen>>;
            Q query(world, "typed_1");
            CHECK(query);
            const std::vector<Entity> expected = ModelVisit(world, alive, TermsOf<Read<Position>, Write<Velocity>,
                                                                                  Without<Frozen>>(world), 0);
            std::vector<Entity> visited;
            query.ForEach(all, [&](Entity e, const Position& p, Velocity& v) {
                CHECK(&p == world->GetComponent(e, ids.position) && &v == world->GetComponent(e, ids.velocity));
                visited.push_back(e);
            });
            CHECK(Sorted(visited) == expected);
            CHECK(query.CountEntities() == expected.size());
        }
        {
            using Q = Query<Optional<Id>, With<Selected>, Without<DebugColor>>;
            Q query(world, "typed_2");
            const std::vector<Entity> expected =
                ModelVisit(world, alive, TermsOf<Optional<Id>, With<Selected>, Without<DebugColor>>(world), 0);
            std::vector<Entity> visited;
            query.ForEach(all, [&](Entity e, const Id* id) {
                CHECK(id == world->GetComponent(e, ids.id));
                visited.push_back(e);
            });
            CHECK(Sorted(visited) == expected);
        }
        {
            using Q = Query<Enabled<Toggle>, Disabled<Hidden>, Read<Position>>;
            Q query(world, "typed_3");
            const std::vector<Entity> expected =
                ModelVisit(world, alive, TermsOf<Enabled<Toggle>, Disabled<Hidden>, Read<Position>>(world), 0);
            std::vector<Entity> visited;
            query.ForEach(all, [&](Entity e, const Toggle& toggle, const Hidden&, const Position& p) {
                CHECK(&toggle == world->GetComponent(e, ids.toggle) && &p == world->GetComponent(e, ids.position));
                CHECK(world->IsEnabled(e, ids.toggle) && !world->IsEnabled(e, ids.hidden));
                visited.push_back(e);
            });
            CHECK(Sorted(visited) == expected);
        }
        {
            using Q = Query<ReadShared<Layer>, ReadChunk<ChunkBounds>, Read<BufferOf<Joint>>>;
            Q query(world, "typed_4");
            const std::vector<Entity> expected =
                ModelVisit(world, alive,
                           TermsOf<ReadShared<Layer>, ReadChunk<ChunkBounds>, Read<BufferOf<Joint>>>(world), 0);
            std::vector<Entity> visited;
            query.ForEach(all, [&](Entity e, const Layer& layer, const ChunkBounds& bounds,
                                   const BufferHeader& joints) {
                CHECK(&layer == world->GetComponent(e, ids.layer));
                CHECK(&bounds == world->GetComponent(e, ids.chunkBounds));
                CHECK(&joints == world->GetComponent(e, ids.joint) && joints.count <= joints.capacity);
                visited.push_back(e);
            });
            CHECK(Sorted(visited) == expected);
        }
        {
            using Q = Query<Changed<Position>, Added<Velocity>>;
            Q query(world, "typed_5");
            const std::vector<Entity> expected =
                ModelVisit(world, alive, TermsOf<Changed<Position>, Added<Velocity>>(world), since);
            std::vector<Entity> visited;
            query.ForEach(recent, [&](Entity e, const Position& p, const Velocity& v) {
                CHECK(&p == world->GetComponent(e, ids.position) && &v == world->GetComponent(e, ids.velocity));
                visited.push_back(e);
            });
            CHECK(Sorted(visited) == expected);
        }
        {
            using Q = Query<WriteChunk<ChunkBounds>, With<Big>>;
            Q query(world, "typed_6");
            const std::vector<Entity> expected =
                ModelVisit(world, alive, TermsOf<WriteChunk<ChunkBounds>, With<Big>>(world), 0);
            std::vector<Entity> visited;
            query.ForEachChunk(all, [&](const ChunkView& view, ChunkBounds* bounds) {
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    CHECK(bounds == world->GetComponent(view.entities[row], ids.chunkBounds));
                    CHECK(Q::RowPasses(view, row));
                    visited.push_back(view.entities[row]);
                }
            });
            CHECK(Sorted(visited) == expected);
        }
        {
            using Q = Query<Write<BufferOf<Joint>>, Optional<Toggle>, With<Selected>>;
            Q query(world, "typed_7");
            const std::vector<Entity> expected =
                ModelVisit(world, alive, TermsOf<Write<BufferOf<Joint>>, Optional<Toggle>, With<Selected>>(world), 0);
            std::vector<Entity> visited;
            query.ForEachChunk(all, [&](const ChunkView& view, BufferHeader* joints, const Toggle* toggles) {
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    const Entity e = view.entities[row];
                    CHECK(Q::RowPasses(view, row) == world->HasComponent(e, ids.selected));
                    if (!Q::RowPasses(view, row))
                        continue;
                    CHECK(joints + row == world->GetComponent(e, ids.joint));
                    const void* toggle = toggles ? static_cast<const void*>(toggles + row) : nullptr;
                    CHECK(toggle == world->GetComponent(e, ids.toggle));
                    visited.push_back(e);
                }
            });
            CHECK(Sorted(visited) == expected);
        }
    }
}

// ------------------------------------------------------------------------------------------------ validation

void test_validation()
{
    for (const bool validation : { true, false })
    {
        auto world = MakeWorld(validation);
        ecs::WorldImpl& impl = Impl(world);
        const Ids ids = RegisterAll(world);
        uint64_t reports = 0;
        auto rejected = [&](const std::vector<QueryTerm>& terms) {
            CHECK(TryQuery(world, terms) == nvrhi::FE_INVALID_ARGS);
            reports += validation ? 1 : 0;
            CHECK(impl.GetValidationErrorCount() == reports);
        };
        auto accepted = [&](const std::vector<QueryTerm>& terms) {
            CHECK(TryQuery(world, terms) == nvrhi::FS_OK);
            CHECK(impl.GetValidationErrorCount() == reports);
        };

        // The descriptor.
        rejected({});
        {
            const QueryDesc desc{ nullptr, 2, 0, "null terms" };
            IQuery* query = nullptr;
            CHECK(world->CreateQuery(desc, &query) == nvrhi::FE_INVALID_ARGS && !query);
            const QueryTerm term = Term(ids.position, TermOp::All, Access::Read);
            const QueryDesc one{ &term, 1, 0, nullptr };
            CHECK(world->CreateQuery(one, nullptr) == nvrhi::FE_INVALID_ARGS);
            reports += validation ? 2 : 0;
            CHECK(impl.GetValidationErrorCount() == reports);
        }
        rejected(std::vector<QueryTerm>(ecs::c_MaxQueryTerms + 1, Term(ids.position, TermOp::All, Access::Read)));
        accepted(std::vector<QueryTerm>(ecs::c_MaxQueryTerms, Term(ids.position, TermOp::All, Access::Read)));

        // Terms on their own.
        rejected({ Term(ComponentId(9999), TermOp::All, Access::Read) });
        rejected({ Term(ComponentId::Invalid, TermOp::All, Access::Read) });
        rejected({ Term(ids.position, TermOp(8), Access::Read) });
        rejected({ Term(ids.position, TermOp::All, Access(3)) });
        rejected({ Term(ids.settings, TermOp::All, Access::Read) });       // a resource
        rejected({ Term(ids.settings, TermOp::Optional, Access::Filter) });
        rejected({ Term(ids.position, TermOp::Without, Access::Write) });
        rejected({ Term(ids.layer, TermOp::All, Access::Write) });         // Write on Shared
        rejected({ Term(ids.layer, TermOp::Optional, Access::Write) });
        rejected({ Term(ids.layer, TermOp::Any, Access::Write) });
        rejected({ Term(ids.selected, TermOp::All, Access::Read) });       // data on Sparse
        rejected({ Term(ids.color, TermOp::Optional, Access::Read) });
        rejected({ Term(ids.color, TermOp::All, Access::Write) });
        rejected({ Term(ids.color, TermOp::Changed, Access::Filter) });    // Sparse has no versions
        rejected({ Term(ids.selected, TermOp::Added, Access::Filter) });
        rejected({ Term(ids.selected, TermOp::Any, Access::Filter) });     // Any only in archetypes
        rejected({ Term(ids.position, TermOp::Enabled, Access::Read) });   // not Enableable
        rejected({ Term(ids.position, TermOp::Disabled, Access::Filter) });
        rejected({ Term(ids.layer, TermOp::Enabled, Access::Read) });
        rejected({ Term(ids.selected, TermOp::Enabled, Access::Filter) });
        // Contradictions.
        rejected({ Term(ids.position, TermOp::All, Access::Read),
                   Term(ids.position, TermOp::Without, Access::Filter) });
        rejected({ Term(ids.velocity, TermOp::Without, Access::Filter),
                   Term(ids.velocity, TermOp::Changed, Access::Read) });
        rejected({ Term(ids.toggle, TermOp::Enabled, Access::Read), Term(ids.toggle, TermOp::Disabled, Access::Read) });
        rejected({ Term(ids.hidden, TermOp::Disabled, Access::Filter),
                   Term(ids.hidden, TermOp::Without, Access::Filter) });
        rejected({ Term(ids.selected, TermOp::All, Access::Filter),
                   Term(ids.selected, TermOp::Without, Access::Filter) });

        // Edge cases that are valid.
        accepted({ Term(ids.frozen, TermOp::All, Access::Read) });         // a tag as data: an address, no bytes
        accepted({ Term(ids.frozen, TermOp::All, Access::Write) });
        accepted({ Term(ids.position, TermOp::Changed, Access::Write) });
        accepted({ Term(ids.color, TermOp::Optional, Access::Filter) });
        accepted({ Term(ids.layer, TermOp::Changed, Access::Read) });
        accepted({ Term(ids.chunkBounds, TermOp::Added, Access::Write) });
        accepted({ Term(ids.position, TermOp::Without, Access::Read) });   // no data (IsDataTerm)
        accepted({ Term(ids.position, TermOp::All, Access::Read), Term(ids.position, TermOp::All, Access::Write) });
        accepted({ Term(ids.position, TermOp::Any, Access::Read), Term(ids.velocity, TermOp::Any, Access::Read),
                   Term(ids.position, TermOp::Without, Access::Filter) });
        accepted({ Term(ids.hidden, TermOp::Enabled, Access::Filter) });
        accepted({ Term(ids.toggle, TermOp::Enabled, Access::Read), Term(ids.hidden, TermOp::Disabled, Access::Read) });
        accepted({ Term(ids.position, TermOp::Optional, Access::Write),
                   Term(ids.selected, TermOp::Without, Access::Filter) });

        // The typed layer: a resource term fails Create and leaves the query empty.
        {
            Query<Read<Settings>> typed;
            CHECK(typed.Create(world, "resource") == nvrhi::FE_INVALID_ARGS && !typed && typed.Get() == nullptr);
            reports += validation ? 1 : 0;
            CHECK(impl.GetValidationErrorCount() == reports);
        }

        // Requests.
        nvrhi::AutoPtr<IQuery> writer = MakeQuery(world, { Term(ids.position, TermOp::All, Access::Write) }, "writer");
        nvrhi::AutoPtr<IQuery> reader = MakeQuery(world, { Term(ids.position, TermOp::All, Access::Read) }, "reader");
        Entity entities[10];
        CreateMany(world, { ids.position }, 10, entities);
        ChunkView views[4];
        const uint64_t now = world->GetVersion();
        CHECK(writer->GetChunks(0, ChunkRequest{ c_AllVersions, now + 1, nullptr }, 4, views) == 1);
        CHECK(ColumnVersion(world, entities[0], ids.position) == now + 1);   // stamped all the same
        reports += validation ? 1 : 0;
        CHECK(reader->GetChunks(0, ChunkRequest{ now + 5, c_WorldVersion, nullptr }, 4, views) == 1);
        reports += validation ? 1 : 0;
        CHECK(impl.GetValidationErrorCount() == reports);
        CHECK(writer->GetChunks(0, ChunkRequest{}, 0, nullptr) == 0);      // nothing asked for
        CHECK(impl.GetValidationErrorCount() == reports);
        CHECK(writer->GetChunks(0, ChunkRequest{}, 4, nullptr) == 0);
        reports += validation ? 1 : 0;
        CHECK(impl.GetValidationErrorCount() == reports);
        world->AdvanceVersion();
        impl.BeginPhase();
        CHECK(writer->GetChunks(0, ChunkRequest{}, 4, views) == 1);        // c_WorldVersion inside a phase
        reports += validation ? 1 : 0;
        CHECK(reader->GetChunks(0, ChunkRequest{}, 4, views) == 1);        // read-only: no version to stamp
        CHECK(writer->GetChunks(0, ChunkRequest{ now, world->GetVersion(), nullptr }, 4, views) == 1);
        impl.EndPhase();
        CHECK(impl.GetValidationErrorCount() == reports);
        CHECK(writer->GetChunks(5, ChunkRequest{}, 4, views) == 0);        // past the end
        CHECK(writer->GetChunks(0xffffff00u, ChunkRequest{}, 0x200u, views) == 0);
        CHECK(impl.GetValidationErrorCount() == reports);
    }
}

// ------------------------------------------------------------------------------------ matching and order

void test_matching_and_order()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);
    nvrhi::AutoPtr<IQuery> positions = MakeQuery(world, { Term(ids.position, TermOp::All, Access::Read) });

    // Archetypes in this creation order: [], [velocity], [position], [position, frozen], [position, velocity],
    // [position, big] (64 KiB chunks).
    Entity none[3];
    CreateMany(world, {}, 3, none);
    std::vector<Entity> v(5), p(2000), pf(10), pv(700), pb(100);
    CreateMany(world, { ids.velocity }, 5, v.data());
    CreateMany(world, { ids.position }, 2000, p.data());
    CreateMany(world, { ids.position, ids.frozen }, 10, pf.data());
    CreateMany(world, { ids.velocity, ids.position }, 700, pv.data());
    CreateMany(world, { ids.position, ids.big }, 100, pb.data());
    const uint32_t aNone = Record(world, none[0]).archetype;
    const uint32_t aV = Record(world, v[0]).archetype;
    const uint32_t aP = Record(world, p[0]).archetype;
    const uint32_t aPF = Record(world, pf[0]).archetype;
    const uint32_t aPV = Record(world, pv[0]).archetype;
    const uint32_t aPB = Record(world, pb[0]).archetype;
    CHECK(aNone == 0 && aNone < aV && aV < aP && aP < aPF && aPF < aPV && aPV < aPB);
    CHECK(impl.GetArchetypeStorage(aP).GetChunkCount() >= 2);
    CHECK(impl.GetArchetypeStorage(aPB).chunkSize == 64 * 1024 && impl.GetArchetypeStorage(aPB).GetChunkCount() == 1);

    // Matched in creation order; chunks in archetype and chunk order; chunk indices count them.
    ecs::QueryImpl& q = QImpl(positions);
    CHECK(positions->CountEntities() == 2810);
    CHECK(q.GetMatchedArchetypeCount() == 4);
    CHECK(q.GetMatchedArchetype(0) == aP && q.GetMatchedArchetype(1) == aPF && q.GetMatchedArchetype(2) == aPV &&
          q.GetMatchedArchetype(3) == aPB);
    CHECK(positions->GetTermCount() == 1 && positions->GetDataTermCount() == 1);
    std::vector<uint64_t> expectedKeys;
    for (const uint32_t a : { aP, aPF, aPV, aPB })
        for (uint32_t c = 0; c < impl.GetArchetypeStorage(a).GetChunkCount(); ++c)
            expectedKeys.push_back((uint64_t(a) << 32) | c);
    CHECK(positions->GetChunkCount() == expectedKeys.size());
    CHECK(Handed(world, positions, ChunkRequest{}) == expectedKeys);
    // Any batch size gives the same views.
    for (const uint32_t capacity : { 1u, 2u, 3u, 64u })
    {
        std::vector<ChunkView> views(capacity);
        uint32_t index = 0;
        for (uint32_t first = 0; first < positions->GetChunkCount(); first += capacity)
        {
            const uint32_t written = positions->GetChunks(first, ChunkRequest{}, capacity, views.data());
            for (uint32_t i = 0; i < written; ++i, ++index)
            {
                CHECK(views[i].chunkIndex == index && KeyOf(world, views[i].entities[0]) == expectedKeys[index]);
                const ecs::Archetype& archetype = impl.GetArchetypeStorage(uint32_t(expectedKeys[index] >> 32));
                const uint32_t chunk = uint32_t(expectedKeys[index]);
                CHECK(views[i].entities == archetype.Entities(chunk));
                CHECK(views[i].count == archetype.chunks[chunk].count);
                CHECK(views[i].columns[0] == archetype.ColumnData(chunk, archetype.FindColumn(ids.position)));
                CHECK(views[i].enabledMasks[0] == nullptr && views[i].presenceMasks[0] == nullptr);
            }
        }
        CHECK(index == expectedKeys.size());
    }

    // Without, Any, Optional, tags as data, and a query of exclusions only (the archetype without components too).
    nvrhi::AutoPtr<IQuery> notFrozen = MakeQuery(world, { Term(ids.position, TermOp::All, Access::Read),
                                                          Term(ids.frozen, TermOp::Without, Access::Filter) });
    CHECK(notFrozen->CountEntities() == 2800 && QImpl(notFrozen).GetMatchedArchetypeCount() == 3);
    nvrhi::AutoPtr<IQuery> any = MakeQuery(world, { Term(ids.velocity, TermOp::Any, Access::Read),
                                                    Term(ids.frozen, TermOp::Any, Access::Read),
                                                    Term(ids.position, TermOp::All, Access::Read) });
    CHECK(any->CountEntities() == 710 && any->GetDataTermCount() == 3);
    {
        ChunkView views[64];
        const uint32_t written = any->GetChunks(0, ChunkRequest{}, 64, views);
        CHECK(written == any->GetChunkCount());
        for (uint32_t i = 0; i < written; ++i)
        {
            const bool frozen = Record(world, views[i].entities[0]).archetype == aPF;
            CHECK((views[i].columns[0] == nullptr) == frozen);                          // absent Any: null
            CHECK((views[i].columns[1] == ecs::g_TagValue) == frozen);                  // a tag: GetComponent's address
            CHECK(views[i].columns[2] != nullptr);
        }
    }
    nvrhi::AutoPtr<IQuery> optional = MakeQuery(world, { Term(ids.position, TermOp::All, Access::Read),
                                                         Term(ids.velocity, TermOp::Optional, Access::Read) });
    {
        ChunkView views[64];
        const uint32_t written = optional->GetChunks(0, ChunkRequest{}, 64, views);
        CHECK(written == positions->GetChunkCount());
        for (uint32_t i = 0; i < written; ++i)
            CHECK((views[i].columns[1] != nullptr) == (Record(world, views[i].entities[0]).archetype == aPV));
    }
    nvrhi::AutoPtr<IQuery> exclusions = MakeQuery(world, { Term(ids.position, TermOp::Without, Access::Filter),
                                                           Term(ids.velocity, TermOp::Without, Access::Read) });
    CHECK(exclusions->CountEntities() == 3 && QImpl(exclusions).GetMatchedArchetype(0) == aNone);
    CHECK(exclusions->GetDataTermCount() == 0);
    {
        ChunkView view{};
        CHECK(exclusions->GetChunks(0, ChunkRequest{}, 1, &view) == 1 && view.columns != nullptr && view.count == 3);
    }

    // Views are stable until the structure changes.
    ChunkView first[2];
    ChunkView again[2];
    CHECK(positions->GetChunks(0, ChunkRequest{}, 2, first) == 2);
    CHECK(positions->GetChunks(0, ChunkRequest{}, 2, again) == 2);
    CHECK(std::memcmp(first, again, sizeof(first)) == 0);
    CHECK(impl.GetValidationErrorCount() == 0);
}

// -------------------------------------------------------------------------------------------- the model

void test_model()
{
    uint64_t checks = 0;
    uint64_t invalid = 0;
    for (uint64_t seed = 1; seed <= 8; ++seed)
    {
        // Odd seeds use 1 KiB chunks (4 KiB large ones): a few rows per chunk, many chunk boundaries.
        ScriptWorld script;
        script.world = (seed & 1) ? MakeWorld(true, 1024, 4096) : MakeWorld(true);
        IWorld* world = script.world;
        script.ids = RegisterAll(world);
        Xorshift rng{ 0x9e3779b97f4a7c15ull * seed };
        uint64_t expectedReports = 0;
        for (uint32_t step = 0; step < 420; ++step)
        {
            RandomStep(script, rng);
            if (step % 30 != 29)
                continue;
            for (uint32_t q = 0; q < 6; ++q)
            {
                std::vector<QueryTerm> terms;
                const uint32_t termCount = rng.Below(24) == 0 ? 0 : 1 + rng.Below(4);
                for (uint32_t t = 0; t < termCount; ++t)
                    terms.push_back(RandomTerm(world, script.ids, rng));
                const uint64_t since = rng.Below(3) == 0 ? c_AllVersions : 1 + rng.Next() % world->GetVersion();
                if (!ExpectValid(world, terms))
                {
                    CHECK(TryQuery(world, terms) == nvrhi::FE_INVALID_ARGS);
                    ++expectedReports;
                    ++invalid;
                    continue;
                }
                CheckRawQuery(world, script.alive, terms, since, rng);
                ++checks;
            }
            CheckTypedQueries(world, script.ids, script.alive, 1 + rng.Next() % world->GetVersion());
        }
        CHECK(Impl(world).GetValidationErrorCount() == expectedReports);
    }
    printf("test_ecs_query: model: %llu raw queries checked, %llu invalid ones refused\n", (unsigned long long)checks,
           (unsigned long long)invalid);
    CHECK(checks > 200 && invalid > 20);
}

// --------------------------------------------------------------------------------------- change detection

void test_change_detection()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);
    std::vector<Entity> plain(3000), moving(1500);
    CreateMany(world, { ids.position }, 3000, plain.data());
    CreateMany(world, { ids.position, ids.velocity }, 1500, moving.data());
    CHECK(world->GetVersion() == 1);   // every chunk is at version 1

    nvrhi::AutoPtr<IQuery> changed = MakeQuery(world, { Term(ids.position, TermOp::Changed, Access::Read) }, "changed");
    nvrhi::AutoPtr<IQuery> added = MakeQuery(world, { Term(ids.position, TermOp::Added, Access::Read) }, "added");
    const std::vector<uint64_t> allChunks = Handed(world, changed, ChunkRequest{});
    CHECK(allChunks.size() >= 6);
    auto since = [](uint64_t version) { return ChunkRequest{ version, c_WorldVersion, nullptr }; };
    CHECK(Handed(world, changed, since(1)).empty() && Handed(world, added, since(1)).empty());

    // GetComponentForWrite stamps its entity's chunk with the version given, and only that chunk.
    const uint64_t v2 = world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(plain[1500], ids.position, v2) != nullptr);
    CHECK(Handed(world, changed, since(1)) == Keys(world, { plain[1500] }));
    CHECK(Handed(world, changed, since(v2)).empty());
    CHECK(Handed(world, added, since(1)).empty());
    // c_WorldVersion is the world's version.
    CHECK(world->GetComponentForWrite(plain[0], ids.position, c_WorldVersion) != nullptr);
    CHECK(ColumnVersion(world, plain[0], ids.position) == v2);

    // A system that processes changed positions and writes them: its own stamps are not changes for its next run.
    nvrhi::AutoPtr<IQuery> system = MakeQuery(world, { Term(ids.position, TermOp::Changed, Access::Write) }, "system");
    const uint64_t v3 = world->AdvanceVersion();
    CHECK(Handed(world, system, ChunkRequest{ c_AllVersions, v3, nullptr }) == allChunks);   // the first run
    for (const Entity e : { plain[0], plain[2999], moving[0], moving[1499] })
        CHECK(ColumnVersion(world, e, ids.position) == v3);
    CHECK(Handed(world, changed, since(v2)) == allChunks);
    const uint64_t v4 = world->AdvanceVersion();
    CHECK(Handed(world, system, ChunkRequest{ v3, v4, nullptr }).empty());   // nothing changed since its last run
    CHECK(Handed(world, changed, since(v3)).empty());                         // and nothing was stamped
    const uint64_t v5 = world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(plain[10], ids.position, v5) != nullptr);
    const uint64_t v6 = world->AdvanceVersion();
    CHECK(Handed(world, system, ChunkRequest{ v4, v6, nullptr }) == Keys(world, { plain[10] }));
    CHECK(ColumnVersion(world, plain[10], ids.position) == v6 && ColumnVersion(world, plain[2999], ids.position) == v3);

    // Chunks that the filters skip are not stamped.
    nvrhi::AutoPtr<IQuery> follow = MakeQuery(world, { Term(ids.velocity, TermOp::Changed, Access::Read),
                                                       Term(ids.position, TermOp::All, Access::Write) }, "follow");
    const uint64_t v7 = world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(moving[1400], ids.velocity, v7) != nullptr);
    const uint64_t v8 = world->AdvanceVersion();
    CHECK(Keys(world, { moving[0] }) != Keys(world, { moving[1400] }));
    CHECK(Handed(world, follow, ChunkRequest{ v6, v8, nullptr }) == Keys(world, { moving[1400] }));
    CHECK(ColumnVersion(world, moving[1400], ids.position) == v8);
    CHECK(ColumnVersion(world, moving[0], ids.position) == v3);

    // Several Changed terms: every one of them must be newer (each term is a requirement, EcsTypes.h).
    nvrhi::AutoPtr<IQuery> both = MakeQuery(world, { Term(ids.position, TermOp::Changed, Access::Read),
                                                     Term(ids.velocity, TermOp::Changed, Access::Read) }, "both");
    const uint64_t v9 = world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(moving[0], ids.position, v9) != nullptr);
    CHECK(world->GetComponentForWrite(moving[1400], ids.velocity, v9) != nullptr);
    CHECK(Handed(world, both, since(v8)).empty());
    CHECK(world->GetComponentForWrite(moving[1], ids.velocity, v9) != nullptr);
    CHECK(Handed(world, both, since(v8)) == Keys(world, { moving[0] }));

    // Stamps are a maximum: an older version never lowers a column, from GetComponentForWrite or from a query.
    const uint64_t v10 = world->AdvanceVersion();
    const uint64_t v11 = world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(plain[0], ids.position, v11) != nullptr);
    CHECK(world->GetComponentForWrite(plain[0], ids.position, v10) != nullptr);
    CHECK(ColumnVersion(world, plain[0], ids.position) == v11);
    nvrhi::AutoPtr<IQuery> writer = MakeQuery(world, { Term(ids.position, TermOp::All, Access::Write) }, "writer");
    CHECK(Handed(world, writer, ChunkRequest{ c_AllVersions, v10, nullptr }) == allChunks);
    CHECK(ColumnVersion(world, plain[0], ids.position) == v11);
    CHECK(ColumnVersion(world, plain[2999], ids.position) == v10);
    // c_WorldVersion in a request is the world's version too.
    const uint64_t v12 = world->AdvanceVersion();
    CHECK(Handed(world, writer, ChunkRequest{}) == allChunks);
    CHECK(ColumnVersion(world, plain[2999], ids.position) == v12);

    // A structural move stamps every column and the added version of the target chunk, and of the source chunk when
    // a row was swapped into the hole (chunk granularity over-reports, 4.5).
    const uint64_t v13 = world->AdvanceVersion();
    const Entity mover = plain[100];
    const std::vector<uint64_t> source = Keys(world, { mover });
    const Velocity velocity{ 1, 2, 3 };
    CHECK(world->AddComponent(mover, ids.velocity, &velocity) == nvrhi::FS_OK);
    std::vector<uint64_t> expected = source;
    expected.push_back(KeyOf(world, mover));
    CHECK(Handed(world, changed, since(v13 - 1)) == expected);
    CHECK(Handed(world, added, since(v13 - 1)) == expected);
    // The archetype's last row moves without a swap: only its target is stamped.
    const uint64_t v14 = world->AdvanceVersion();
    const ecs::Archetype& plainArchetype = impl.GetArchetypeStorage(Record(world, plain[0]).archetype);
    const uint32_t lastChunk = plainArchetype.GetChunkCount() - 1;
    const Entity last = plainArchetype.Entities(lastChunk)[plainArchetype.chunks[lastChunk].count - 1];
    CHECK(world->AddComponent(last, ids.velocity, &velocity) == nvrhi::FS_OK);
    CHECK(Handed(world, changed, since(v14 - 1)) == Keys(world, { last }));
    CHECK(Handed(world, added, since(v14 - 1)) == Keys(world, { last }));

    // Added sees creations and instantiations, in the chunks that got the rows.
    const uint64_t v15 = world->AdvanceVersion();
    Entity fresh[10];
    CreateMany(world, { ids.position }, 10, fresh);
    std::vector<uint64_t> freshKeys;
    for (const Entity e : fresh)
        freshKeys.push_back(KeyOf(world, e));
    std::sort(freshKeys.begin(), freshKeys.end());
    freshKeys.erase(std::unique(freshKeys.begin(), freshKeys.end()), freshKeys.end());
    CHECK(Handed(world, added, since(v15 - 1)) == freshKeys);
    const uint64_t v16 = world->AdvanceVersion();
    Entity copies[5];
    CHECK(world->Instantiate(moving[3], 5, copies) == nvrhi::FS_OK);
    CHECK(Handed(world, added, since(v16 - 1)) == Keys(world, { copies[0], copies[4] }));
    // Added on a component the move added: the typed Added<Velocity> sees the target chunk.
    const uint64_t v17 = world->AdvanceVersion();
    CHECK(world->AddComponent(plain[200], ids.velocity, &velocity) == nvrhi::FS_OK);
    Query<Added<Velocity>> addedVelocity(world, "added velocity");
    std::vector<Entity> visited;
    addedVelocity.ForEach(since(v17 - 1), [&](Entity e, const Velocity&) { visited.push_back(e); });
    CHECK(std::find(visited.begin(), visited.end(), plain[200]) != visited.end());
    for (const Entity e : visited)
        CHECK(KeyOf(world, e) == KeyOf(world, plain[200]));

    // The context forms use the context's versions.
    auto context = MAKE_RC_OBJ_PTR(TestContext, world.Get(), nullptr);
    context->NextRun();
    context->NextRun();
    Query<Changed<Position>, Write<Velocity>> contextQuery(world, "context");
    CHECK(world->GetComponentForWrite(moving[700], ids.position, context->version) != nullptr);
    uint32_t chunks = 0;
    contextQuery.ForEachChunk(context.Get(), [&](const ChunkView& view, const Position*, Velocity*) {
        ++chunks;
        CHECK(ColumnVersion(world, view.entities[0], ids.velocity) == context->version);
    });
    CHECK(chunks == 1);   // only moving[700]'s chunk changed since the context's last run
    CHECK(impl.GetValidationErrorCount() == 0);
}

// ------------------------------------------------------------------------------------------- enabled bits

void test_enabled_bits()
{
    auto world = MakeWorld(true);
    const Ids ids = RegisterAll(world);
    std::vector<Entity> e(3000);
    CreateMany(world, { ids.position, ids.toggle, ids.hidden }, 3000, e.data());
    for (uint32_t i = 0; i < 3000; ++i)
    {
        if (i % 3 == 0)
            CHECK(world->SetEnabled(e[i], ids.toggle, false) == nvrhi::FS_OK);
        if (i % 5 == 0)
            CHECK(world->SetEnabled(e[i], ids.hidden, false) == nvrhi::FS_OK);
    }
    auto expect = [&](auto predicate) {
        std::vector<Entity> out;
        for (uint32_t i = 0; i < 3000; ++i)
            if (predicate(i))
                out.push_back(e[i]);
        return Sorted(out);
    };
    const ChunkRequest all{};

    Query<Enabled<Toggle>, Read<Position>> enabled(world);
    Query<Disabled<Toggle>> disabled(world);
    Query<Enabled<Toggle>, Disabled<Hidden>> both(world);
    std::vector<Entity> visited;
    enabled.ForEach(all, [&](Entity entity, const Toggle& toggle, const Position&) {
        CHECK(&toggle == world->GetComponent(entity, ids.toggle));
        visited.push_back(entity);
    });
    CHECK(Sorted(visited) == expect([](uint32_t i) { return i % 3 != 0; }));
    visited.clear();
    disabled.ForEach(all, [&](Entity entity, const Toggle&) { visited.push_back(entity); });
    CHECK(Sorted(visited) == expect([](uint32_t i) { return i % 3 == 0; }));
    visited.clear();
    // A tag argument refers to the tag sentinel at its row's offset, inside the sentinel for every row of a chunk.
    uint32_t highestRow = 0;
    both.ForEach(all, [&](Entity entity, const Toggle&, const Hidden& hidden) {
        const uint8_t* address = reinterpret_cast<const uint8_t*>(&hidden);
        const uint32_t row = Record(world, entity).row;
        CHECK(address >= ecs::g_TagValue && address < ecs::g_TagValue + sizeof(ecs::g_TagValue));
        CHECK(size_t(address - ecs::g_TagValue) == size_t(row) * sizeof(Hidden));
        highestRow = std::max(highestRow, row);
        visited.push_back(entity);
    });
    CHECK(Sorted(visited) == expect([](uint32_t i) { return i % 3 != 0 && i % 5 == 0; }));
    CHECK(highestRow >= 64);   // past the 64 bytes the sentinel used to have

    // ForEachChunk hands every row out; RowPasses and the masks tell them apart.
    uint32_t rows = 0;
    enabled.ForEachChunk(all, [&](const ChunkView& view, const Toggle*, const Position*) {
        CHECK(view.enabledMasks[0] != nullptr && view.enabledMasks[1] == nullptr);
        for (uint32_t row = 0; row < view.count; ++row, ++rows)
            CHECK(decltype(enabled)::RowPasses(view, row) == world->IsEnabled(view.entities[row], ids.toggle));
    });
    CHECK(rows == 3000);

    // Enabled bits follow archetype moves.
    const Velocity velocity{ 0, 0, 0 };
    CHECK(world->AddComponent(e[3], ids.velocity, &velocity) == nvrhi::FS_OK);
    visited.clear();
    disabled.ForEach(all, [&](Entity entity, const Toggle&) { visited.push_back(entity); });
    CHECK(Sorted(visited) == expect([](uint32_t i) { return i % 3 == 0; }));

    // SetEnabled stamps the column with the world's version: Changed sees the chunk.
    nvrhi::AutoPtr<IQuery> changed = MakeQuery(world, { Term(ids.toggle, TermOp::Changed, Access::Read) });
    const uint64_t v = world->AdvanceVersion();
    CHECK(world->SetEnabled(e[7], ids.toggle, false) == nvrhi::FS_OK);
    CHECK(Handed(world, changed, ChunkRequest{ v - 1, c_WorldVersion, nullptr }) == Keys(world, { e[7] }));
    CHECK(Impl(world).GetValidationErrorCount() == 0);
}

// ------------------------------------------------------------------------------------------ sparse filters

void test_sparse_filters()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);
    std::vector<Entity> e(2000);
    std::vector<Position> values(2000);
    for (uint32_t i = 0; i < 2000; ++i)
        values[i] = Position{ float(i), 0, 0 };
    CreateMany(world, { ids.position }, 2000, e.data(), { values.data() });
    const ChunkRequest all{};

    using SelectedQuery = Query<Read<Position>, With<Selected>>;
    SelectedQuery selected(world, "selected");
    Query<Read<Position>, Without<Selected>> unselected(world, "unselected");
    Query<With<DebugColor>, With<Selected>> colored(world, "colored");
    Query<Read<Position>> plain(world, "plain");
    auto visit = [&](auto& query) {
        std::vector<Entity> out;
        query.ForEach(all, [&](Entity entity, auto&&...) { out.push_back(entity); });
        return Sorted(out);
    };
    auto expect = [&](auto predicate) {
        std::vector<Entity> out;
        for (uint32_t i = 0; i < 2000; ++i)
            if (world->IsAlive(e[i]) && predicate(i))
                out.push_back(e[i]);
        return Sorted(out);
    };

    // Nothing selected yet: no chunk has a row for With<Selected>, so none is handed out; Without<Selected> gets every
    // chunk, with its all-zero presence mask.
    CHECK(visit(selected).empty() && visit(unselected) == Sorted(e) && visit(plain).size() == 2000);
    uint32_t chunks = 0;
    selected.ForEachChunk(all, [&](const ChunkView&, const Position*) { ++chunks; });
    CHECK(chunks == 0);
    unselected.ForEachChunk(all, [&](const ChunkView& view, const Position* positions) {
        CHECK(view.presenceMasks[0] == nullptr && view.presenceMasks[1] == ecs::g_ZeroMask);
        CHECK(positions == world->GetComponent(view.entities[0], ids.position));
        ++chunks;
    });
    CHECK(chunks >= 2 && chunks == unselected.Get()->GetChunkCount());

    // The first Selected row of a chunk allocates the chunk's mask; queries built before see it (they rebuild on the
    // presence-mask version), and queries without Sparse terms do not rebuild for it.
    const uint64_t selectedRebuilds = QImpl(selected.Get()).GetCounters().rebuilds;
    const uint64_t plainRebuilds = QImpl(plain.Get()).GetCounters().rebuilds;
    const uint64_t maskVersion = impl.GetPresenceMaskVersion();
    CHECK(world->AddComponent(e[5], ids.selected, nullptr) == nvrhi::FS_OK);
    CHECK(impl.GetPresenceMaskVersion() == maskVersion + 1);
    CHECK(visit(selected) == expect([](uint32_t i) { return i == 5; }));
    CHECK(QImpl(selected.Get()).GetCounters().rebuilds == selectedRebuilds + 1);
    CHECK(visit(plain).size() == 2000 && QImpl(plain.Get()).GetCounters().rebuilds == plainRebuilds);
    // The mask of that chunk exists now: another row of it needs no rebuild.
    CHECK(world->AddComponent(e[6], ids.selected, nullptr) == nvrhi::FS_OK);
    CHECK(impl.GetPresenceMaskVersion() == maskVersion + 1);
    CHECK(visit(selected) == expect([](uint32_t i) { return i == 5 || i == 6; }));
    CHECK(QImpl(selected.Get()).GetCounters().rebuilds == selectedRebuilds + 1);

    for (uint32_t i = 0; i < 2000; i += 7)
        CHECK(world->AddComponent(e[i], ids.selected, nullptr) == nvrhi::FS_OK);
    for (uint32_t i = 0; i < 2000; i += 3)
    {
        const DebugColor color{ i };
        CHECK(world->AddComponent(e[i], ids.color, &color) == nvrhi::FS_OK);
    }
    auto isSelected = [](uint32_t i) { return i % 7 == 0 || i == 5 || i == 6; };
    CHECK(visit(selected) == expect(isSelected));
    CHECK(visit(unselected) == expect([&](uint32_t i) { return !isSelected(i); }));
    CHECK(visit(colored) == expect([&](uint32_t i) { return isSelected(i) && i % 3 == 0; }));

    // Removal, an archetype move (the presence follows the row) and destruction.
    CHECK(world->RemoveComponent(e[7], ids.selected) == nvrhi::FS_OK);
    const Velocity velocity{ 1, 1, 1 };
    CHECK(world->AddComponent(e[14], ids.velocity, &velocity) == nvrhi::FS_OK);
    CHECK(world->DestroyEntities(&e[21], 1) == nvrhi::FS_OK);
    auto isSelectedNow = [&](uint32_t i) { return isSelected(i) && i != 7; };
    CHECK(visit(selected) == expect(isSelectedNow));
    CHECK(visit(unselected) == expect([&](uint32_t i) { return !isSelectedNow(i); }));

    // Sparse values are read per entity.
    CHECK(static_cast<const DebugColor*>(world->GetComponent(e[9], ids.color))->rgba == 9);
    CHECK(GetComponent<DebugColor>(world, e[10]) == nullptr);
    CHECK(impl.GetValidationErrorCount() == 0);
}

// --------------------------------------------------------------------------- chunks without a passing row

// GetChunks skips a chunk in which no row passes the row filters: no view, no stamp (E1.2 review). One selected entity
// must not make a query with Write<T> and With<Selected> mark every chunk of T as changed.
void test_row_filter_skips()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);
    std::vector<Entity> e(6000);
    CreateMany(world, { ids.position, ids.toggle }, 6000, e.data());
    const ecs::Archetype& archetype = impl.GetArchetypeStorage(Record(world, e[0]).archetype);
    nvrhi::AutoPtr<IQuery> changed = MakeQuery(world, { Term(ids.position, TermOp::Changed, Access::Read) }, "changed");
    const std::vector<uint64_t> allChunks = Handed(world, changed, ChunkRequest{});
    CHECK(allChunks.size() >= 6);
    auto stampedSince = [&](uint64_t version) {
        return Handed(world, changed, ChunkRequest{ version, c_WorldVersion, nullptr });
    };
    auto allBut = [&](uint64_t key) {
        std::vector<uint64_t> keys = allChunks;
        keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
        return keys;
    };

    // With on a Sparse component: one selected row among 6000 gives one view and one stamped chunk.
    CHECK(world->AddComponent(e[4321], ids.selected, nullptr) == nvrhi::FS_OK);
    using SelectedWriter = Query<Write<Position>, With<Selected>>;
    SelectedWriter selectedWriter(world, "selected writer");
    uint64_t v = world->AdvanceVersion();
    std::vector<Entity> visited;
    uint32_t views = 0;
    selectedWriter.ForEachChunk(ChunkRequest{ c_AllVersions, v, nullptr }, [&](const ChunkView& view, Position*) {
        ++views;
        for (uint32_t row = 0; row < view.count; ++row)
            if (SelectedWriter::RowPasses(view, row))
                visited.push_back(view.entities[row]);
    });
    CHECK(views == 1 && visited == std::vector<Entity>{ e[4321] });
    CHECK(stampedSince(v - 1) == Keys(world, { e[4321] }));

    // Without on a Sparse component: a chunk whose rows are all selected is skipped.
    const uint32_t firstRows = archetype.chunks[0].count;
    for (uint32_t row = 0; row < firstRows; ++row)
        CHECK(world->AddComponent(archetype.Entities(0)[row], ids.selected, nullptr) == nvrhi::FS_OK);
    Query<Write<Position>, Without<Selected>> unselectedWriter(world, "unselected writer");
    v = world->AdvanceVersion();
    uint32_t rows = 0;
    unselectedWriter.ForEach(ChunkRequest{ c_AllVersions, v, nullptr }, [&](Entity, Position&) { ++rows; });
    CHECK(rows == 6000 - firstRows - 1);
    CHECK(stampedSince(v - 1) == allBut(allChunks.front()));

    // Disabled: one disabled row gives one view and one stamped chunk.
    CHECK(world->SetEnabled(e[100], ids.toggle, false) == nvrhi::FS_OK);
    Query<Write<Position>, Disabled<Toggle>> disabledWriter(world, "disabled writer");
    v = world->AdvanceVersion();
    visited.clear();
    disabledWriter.ForEach(ChunkRequest{ c_AllVersions, v, nullptr }, [&](Entity entity, Position&, const Toggle&) {
        visited.push_back(entity);
    });
    CHECK(visited == std::vector<Entity>{ e[100] });
    CHECK(stampedSince(v - 1) == Keys(world, { e[100] }));

    // Enabled: a chunk whose rows are all disabled is skipped.
    const uint32_t lastChunk = uint32_t(allChunks.back());
    for (uint32_t row = 0; row < archetype.chunks[lastChunk].count; ++row)
        CHECK(world->SetEnabled(archetype.Entities(lastChunk)[row], ids.toggle, false) == nvrhi::FS_OK);
    Query<Write<Position>, Enabled<Toggle>> enabledWriter(world, "enabled writer");
    v = world->AdvanceVersion();
    rows = 0;
    enabledWriter.ForEach(ChunkRequest{ c_AllVersions, v, nullptr }, [&](Entity, Position&, const Toggle&) { ++rows; });
    CHECK(rows == 6000 - 1 - archetype.chunks[lastChunk].count);
    CHECK(stampedSince(v - 1) == allBut(allChunks.back()));

    // Without row filters every chunk is handed out, whatever the rows' bits.
    Query<Write<Position>, Read<Toggle>> writer(world, "writer");
    v = world->AdvanceVersion();
    views = 0;
    writer.ForEachChunk(ChunkRequest{ c_AllVersions, v, nullptr }, [&](const ChunkView&, Position*, const Toggle*) {
        ++views;
    });
    CHECK(views == allChunks.size() && stampedSince(v - 1) == allChunks);
    CHECK(impl.GetValidationErrorCount() == 0);
}

// ------------------------------------------------------------------------------ Shared and Chunk components

void test_shared_and_chunk_components()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);
    const ChunkRequest all{};

    // Three partitions of one component set.
    std::vector<Entity> e(3000);
    std::vector<Layer> layers(3000);
    for (uint32_t i = 0; i < 3000; ++i)
        layers[i] = Layer{ i % 3 };
    CreateMany(world, { ids.position, ids.layer }, 3000, e.data(), { nullptr, layers.data() });
    Query<ReadShared<Layer>, Read<Position>> layered(world, "layered");
    CHECK(layered.CountEntities() == 3000 && QImpl(layered.Get()).GetMatchedArchetypeCount() == 3);
    uint32_t rows = 0;
    layered.ForEachChunk(all, [&](const ChunkView& view, const Layer* layer, const Position*) {
        for (uint32_t row = 0; row < view.count; ++row, ++rows)
            CHECK(layer == world->GetComponent(view.entities[row], ids.layer));   // one value per chunk
    });
    CHECK(rows == 3000);
    std::vector<uint32_t> seen(3, 0);
    layered.ForEach(all, [&](Entity entity, const Layer& layer, const Position&) {
        CHECK(&layer == world->GetComponent(entity, ids.layer) && layer.id < 3);
        ++seen[layer.id];
    });
    CHECK(seen[0] == 1000 && seen[1] == 1000 && seen[2] == 1000);
    // A new Shared value moves the entity to its partition.
    const Layer two{ 2 };
    CHECK(world->AddComponent(e[0], ids.layer, &two) == nvrhi::FS_OK);
    uint32_t movedVisits = 0;
    layered.ForEach(all, [&](Entity entity, const Layer& layer, const Position&) {
        if (entity != e[0])
            return;
        CHECK(layer.id == 2);
        ++movedVisits;
    });
    CHECK(movedVisits == 1);
    Query<With<Layer>> withLayer(world);
    Query<Read<Position>, Without<Layer>> withoutLayer(world);
    CHECK(withLayer.CountEntities() == 3000 && withoutLayer.CountEntities() == 0);

    // Chunk components: WriteChunk writes the chunk's value, ReadChunk and GetComponent read it.
    std::vector<Entity> c(3000);
    std::vector<Position> positions(3000);
    for (uint32_t i = 0; i < 3000; ++i)
        positions[i] = Position{ float(i), 0, 0 };
    CreateMany(world, { ids.position, ids.chunkBounds }, 3000, c.data(), { positions.data(), nullptr });
    Query<WriteChunk<ChunkBounds>, Read<Position>> bound(world, "bound");
    Query<ReadChunk<ChunkBounds>> bounds(world, "bounds");
    const uint64_t v = world->AdvanceVersion();
    uint32_t chunks = 0;
    bound.ForEachChunk(ChunkRequest{ c_AllVersions, v, nullptr }, [&](const ChunkView& view, ChunkBounds* b,
                                                                      const Position* p) {
        b->lo = p[0].x;
        b->hi = p[view.count - 1].x;
        ++chunks;
    });
    CHECK(chunks >= 2 && chunks == bounds.Get()->GetChunkCount());
    bounds.ForEach(all, [&](Entity entity, const ChunkBounds& b) {
        CHECK(&b == world->GetComponent(entity, ids.chunkBounds));
        const float x = static_cast<const Position*>(world->GetComponent(entity, ids.position))->x;
        CHECK(b.lo <= x && x <= b.hi);
    });
    // WriteChunk stamped the column: a Changed term on the Chunk component sees every chunk, then none.
    nvrhi::AutoPtr<IQuery> changed = MakeQuery(world, { Term(ids.chunkBounds, TermOp::Changed, Access::Read) });
    CHECK(Handed(world, changed, ChunkRequest{ v - 1, c_WorldVersion, nullptr }).size() == chunks);
    CHECK(Handed(world, changed, ChunkRequest{ v, c_WorldVersion, nullptr }).empty());
    // GetComponentForWrite of a Chunk component stamps it too.
    const uint64_t v2 = world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(c[2000], ids.chunkBounds, v2) != nullptr);
    CHECK(Handed(world, changed, ChunkRequest{ v, c_WorldVersion, nullptr }) == Keys(world, { c[2000] }));
    CHECK(impl.GetValidationErrorCount() == 0);
}

// ------------------------------------------------------------------------------------------------- buffers

void test_buffers()
{
    auto world = MakeWorld(true);
    const Ids ids = RegisterAll(world);
    std::vector<Entity> e(500);
    CreateMany(world, { ids.position, ids.joint }, 500, e.data());
    for (uint32_t i = 0; i < 500; ++i)
    {
        CHECK(world->ResizeBuffer(e[i], ids.joint, i % 5, c_WorldVersion) == nvrhi::FS_OK);
        const BufferHeader* header = static_cast<const BufferHeader*>(world->GetComponent(e[i], ids.joint));
        for (uint32_t k = 0; k < header->count; ++k)
            GetBufferData<Joint>(*header)[k] = Joint{ e[i], float(i), 0 };
    }
    const ChunkRequest all{};
    Query<Read<BufferOf<Joint>>> reader(world, "reader");
    Query<Write<BufferOf<Joint>>, Read<Position>> writer(world, "writer");
    uint32_t visited = 0;
    reader.ForEach(all, [&](Entity entity, const BufferHeader& header) {
        CHECK(&header == world->GetComponent(entity, ids.joint));
        const uint32_t i = uint32_t(std::find(e.begin(), e.end(), entity) - e.begin());
        CHECK(header.count == i % 5);
        for (uint32_t k = 0; k < header.count; ++k)
            CHECK(GetBufferData<Joint>(header)[k].target == entity &&
                  GetBufferData<Joint>(header)[k].weight == float(i));
        ++visited;
    });
    CHECK(visited == 500);
    const uint64_t v = world->AdvanceVersion();
    writer.ForEach(ChunkRequest{ c_AllVersions, v, nullptr }, [&](Entity, BufferHeader& header, const Position&) {
        for (uint32_t k = 0; k < header.count; ++k)
            GetBufferData<Joint>(header)[k].weight += 1000.0f;
    });
    CHECK(ColumnVersion(world, e[0], ids.joint) == v);
    reader.ForEachChunk(all, [&](const ChunkView& view, const BufferHeader* headers) {
        for (uint32_t row = 0; row < view.count; ++row)
            for (uint32_t k = 0; k < headers[row].count; ++k)
                CHECK(GetBufferData<Joint>(headers[row])[k].weight >= 1000.0f);
    });

    // ResizeBuffer stamps the column; a spilled buffer is read the same way.
    Query<Changed<BufferOf<Joint>>> changed(world, "changed");
    const uint64_t v2 = world->AdvanceVersion();
    CHECK(world->ResizeBuffer(e[42], ids.joint, 9, v2) == nvrhi::FS_OK);
    std::vector<Entity> changedRows;
    changed.ForEach(ChunkRequest{ v2 - 1, c_WorldVersion, nullptr }, [&](Entity entity, const BufferHeader& header) {
        changedRows.push_back(entity);
        if (entity == e[42])
            CHECK(header.count == 9 && header.capacity >= 9);
    });
    CHECK(std::find(changedRows.begin(), changedRows.end(), e[42]) != changedRows.end());
    for (const Entity entity : changedRows)
        CHECK(KeyOf(world, entity) == KeyOf(world, e[42]));
    CHECK(Impl(world).GetValidationErrorCount() == 0);
}

// ----------------------------------------------------------------------------------------------- resources

void test_resources()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);
    Settings* settings = GetResource<Settings>(world);
    CHECK(settings && settings->gravity == 9.81f && settings->steps == 4);
    settings->steps = 8;
    CHECK(static_cast<const Settings*>(world->GetResource(ids.settings))->steps == 8);
    // Registering again keeps the value; resources share the component id space.
    ComponentId again = ComponentId::Invalid;
    const Settings other{ 1, 1 };
    CHECK(world->RegisterResource(ComponentTraits<Settings>::Desc(), &other, again) == nvrhi::FS_OK);
    CHECK(again == ids.settings);
    CHECK(GetResource<Settings>(world)->steps == 8);
    CHECK(uint32_t(ids.settings) < world->GetComponentTypeCount());
    CHECK(!!(world->GetComponentTypeDesc(ids.settings)->flags & ComponentFlags::Resource));

    // FrameTime is a resource the scheduler writes (EcsScheduler.h).
    const ComponentId frameTime = RegisterComponent<FrameTime>(world);
    CHECK(frameTime != ComponentId::Invalid);
    FrameTime* time = GetResource<FrameTime>(world);
    CHECK(time && time->frameIndex == 0 && time->deltaTime == 0);
    time->frameIndex = 7;
    CHECK(GetResource<FrameTime>(world)->frameIndex == 7);

    // Resources are never on entities, and queries refuse them.
    Entity entity = Entity::Null;
    CreateMany(world, { ids.position }, 1, &entity);
    CHECK(!world->HasComponent(entity, ids.settings) && world->GetComponent(entity, ids.settings) == nullptr);
    const uint64_t reports = impl.GetValidationErrorCount();
    Query<Read<FrameTime>> query(world);
    CHECK(!query);
    CHECK(TryQuery(world, { Term(ids.settings, TermOp::All, Access::Filter) }) == nvrhi::FE_INVALID_ARGS);
    CHECK(impl.GetValidationErrorCount() == reports + 2);
}

// ---------------------------------------------------------------------------------------------- lifetime

void test_query_lifetime()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);

    // A query made before anything matches.
    Query<Read<Position>> query(world, "lifetime");
    ecs::QueryImpl& q = QImpl(query.Get());
    CHECK(query.Get()->GetChunkCount() == 0 && query.CountEntities() == 0);
    const uint64_t rebuilds = q.GetCounters().rebuilds;
    CHECK(rebuilds == 1);
    CHECK(query.Get()->GetChunkCount() == 0 && q.GetCounters().rebuilds == rebuilds);   // nothing changed

    // Archetypes created after it: matched at the next call after the immediate change.
    std::vector<Entity> p(10), v(10), pv(2000), pt(10);
    CreateMany(world, { ids.position, ids.toggle, ids.joint }, 10, pt.data());
    CHECK(query.Get()->GetChunkCount() == 1 && query.CountEntities() == 10 && q.GetMatchedArchetypeCount() == 1);
    CHECK(q.GetCounters().rebuilds == rebuilds + 1);
    CreateMany(world, { ids.velocity }, 10, v.data());
    CHECK(query.CountEntities() == 10 && q.GetMatchedArchetypeCount() == 1 && q.GetCounters().rebuilds == rebuilds + 2);
    CreateMany(world, { ids.position, ids.velocity }, 2000, pv.data());
    CreateMany(world, { ids.position }, 10, p.data());
    CHECK(query.CountEntities() == 2020 && q.GetMatchedArchetypeCount() == 3);
    CHECK(q.GetMatchedArchetype(0) < q.GetMatchedArchetype(1) && q.GetMatchedArchetype(1) < q.GetMatchedArchetype(2));
    CHECK(q.GetCounters().archetypesTested == impl.GetArchetypeStorageCount());   // each archetype tested once

    // Operations that change no structure do not rebuild the list.
    const uint64_t settled = q.GetCounters().rebuilds;
    const Entity e = pt[3];
    CHECK(world->GetComponentForWrite(e, ids.position, c_WorldVersion) != nullptr);
    CHECK(world->SetEnabled(e, ids.toggle, false) == nvrhi::FS_OK);
    CHECK(world->ResizeBuffer(e, ids.joint, 2, c_WorldVersion) == nvrhi::FS_OK);
    world->AdvanceVersion();
    CHECK(world->AddComponent(e, ids.selected, nullptr) == nvrhi::FS_OK);   // Sparse: no row moves
    CHECK(query.CountEntities() == 2020 && q.GetCounters().rebuilds == settled);

    // Inside a phase the list does not change; an exclusive system's immediate change shows at the next call.
    impl.BeginPhase();
    Entity late = Entity::Null;
    {
        const ComponentId only[] = { ids.position, ids.frozen };
        EntityCreateDesc desc{};
        desc.components = only;
        desc.componentCount = 2;
        desc.entityCount = 1;
        CHECK(world->CreateEntities(desc, &late) == nvrhi::FE_INVALID_ARGS);
    }
    CHECK(query.CountEntities() == 2020 && q.GetCounters().rebuilds == settled);
    Query<Read<Position>, With<Frozen>> inPhase(world, "made in a phase");
    CHECK(inPhase && inPhase.CountEntities() == 0);
    impl.BeginExclusive();
    Entity frozen[3];
    CreateMany(world, { ids.position, ids.frozen }, 3, frozen);
    impl.EndExclusive();
    CHECK(query.CountEntities() == 2023 && inPhase.CountEntities() == 3);
    impl.EndPhase();
    const uint64_t validationReports = impl.GetValidationErrorCount();
    CHECK(validationReports == 1);   // the refused creation

    // Destruction empties chunks and archetypes; the list follows.
    CHECK(world->DestroyEntities(pv.data(), 2000) == nvrhi::FS_OK);
    CHECK(query.CountEntities() == 23 && query.Get()->GetChunkCount() == 3);

    // A query holds its world: released last, the world frees everything with it.
    {
        const int64_t liveBlocks = ecs::GetHeapLiveBlockCount();
        auto other = MakeWorld(false);
        const Ids otherIds = RegisterAll(other);
        std::vector<Entity> rows(500);
        CreateMany(other, { otherIds.position }, 500, rows.data());
        IQuery* raw = nullptr;
        const QueryTerm term = Term(otherIds.position, TermOp::All, Access::Read);
        CHECK(other->CreateQuery(QueryDesc{ &term, 1, 0, "outlives" }, &raw) == nvrhi::FS_OK);
        other = nullptr;
        CHECK(raw->CountEntities() == 500 && raw->GetChunkCount() == 1);
        ChunkView view{};
        CHECK(raw->GetChunks(0, ChunkRequest{}, 1, &view) == 1 && view.count == 500);
        raw->Release();
        CHECK(ecs::GetHeapLiveBlockCount() == liveBlocks);
    }

    // A query that the world itself owns (E1.4's observer filters) holds no reference, so that the two do not keep
    // each other alive; released after its world, it touches nothing of it.
    {
        const int64_t liveBlocks = ecs::GetHeapLiveBlockCount();
        auto other = MakeWorld(false);
        const Ids otherIds = RegisterAll(other);
        Entity rows[10];
        CreateMany(other, { otherIds.position }, 10, rows);
        const QueryTerm term = Term(otherIds.position, TermOp::All, Access::Read);
        auto references = [&]() {
            const nvrhi::FLONG count = other->AddRef();
            other->Release();
            return count - 1;
        };
        CHECK(references() == 1);
        IQuery* held = nullptr;
        IQuery* owned = nullptr;
        CHECK(ecs::CreateQueryObject(Impl(other), QueryDesc{ &term, 1, 0, "held" }, ecs::QueryWorldReference::Held,
                                     &held) == nvrhi::FS_OK);
        CHECK(ecs::CreateQueryObject(Impl(other), QueryDesc{ &term, 1, 0, "owned" },
                                     ecs::QueryWorldReference::NotHeld, &owned) == nvrhi::FS_OK);
        CHECK(references() == 2);   // the held query's reference only
        held->Release();
        CHECK(references() == 1 && owned->CountEntities() == 10);
        other = nullptr;            // the last reference: the world is destroyed with the query still alive
        owned->Release();
        CHECK(ecs::GetHeapLiveBlockCount() == liveBlocks);
    }
}

// ----------------------------------------------------------------------------------------- the typed layer

void test_typed_forms()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const Ids ids = RegisterAll(world);
    std::vector<Entity> e(6000);
    std::vector<Position> positions(6000);
    for (uint32_t i = 0; i < 6000; ++i)
        positions[i] = Position{ float(i), 1, 2 };
    CreateMany(world, { ids.position, ids.velocity }, 4000, e.data(), { positions.data(), nullptr });
    CreateMany(world, { ids.position, ids.velocity, ids.toggle }, 2000, e.data() + 4000,
               { positions.data() + 4000, nullptr, nullptr });
    for (uint32_t i = 4000; i < 6000; i += 2)
        CHECK(world->SetEnabled(e[i], ids.toggle, false) == nvrhi::FS_OK);
    const uint32_t tableSize = impl.GetEntityTableSize();

    auto jobs = MAKE_RC_OBJ_PTR(TestJobSystem, 4u);
    auto context = MAKE_RC_OBJ_PTR(TestContext, world.Get(), jobs.Get());
    context->NextRun();

    // Two-step creation, Get and the bool conversion.
    Query<Read<Position>, Write<Velocity>, Optional<Toggle>> query;
    CHECK(!query && query.Get() == nullptr);
    CHECK(query.Create(world, "typed") == nvrhi::FS_OK && query && query.Get() != nullptr);
    CHECK(query.CountEntities() == 6000 && query.Get()->GetDataTermCount() == 3);

    // ParallelForEach on four threads: every row once, writes stamped with the context's version.
    std::unique_ptr<std::atomic<uint32_t>[]> hits(new std::atomic<uint32_t>[tableSize]());
    std::atomic<uint32_t> failures{ 0 };
    query.ParallelForEach(context.Get(), 2, [&](Entity entity, const Position& p, Velocity& v, const Toggle* toggle) {
        hits[GetEntityIndex(entity)].fetch_add(1);
        v.x = p.x + 1;
        if ((toggle != nullptr) != (p.x >= 4000))
            failures.fetch_add(1);
    });
    CHECK(jobs->parallelForCalls == 1 && failures.load() == 0);
    for (uint32_t i = 0; i < 6000; ++i)
    {
        CHECK(hits[GetEntityIndex(e[i])].load() == 1);
        CHECK(static_cast<const Velocity*>(world->GetComponent(e[i], ids.velocity))->x == float(i) + 1);
    }
    CHECK(ColumnVersion(world, e[0], ids.velocity) == context->version);
    CHECK(ColumnVersion(world, e[5999], ids.velocity) == context->version);

    // ParallelForChunks: every chunk once, with its chunk index.
    const uint32_t chunkCount = query.Get()->GetChunkCount();
    std::unique_ptr<std::atomic<uint32_t>[]> chunkHits(new std::atomic<uint32_t>[chunkCount]());
    context->NextRun();
    query.ParallelForChunks(context.Get(), 3, [&](const ChunkView& view, const Position* p, Velocity* v,
                                                  const Toggle*) {
        chunkHits[view.chunkIndex].fetch_add(1);
        for (uint32_t row = 0; row < view.count; ++row)
            v[row].y = p[row].y;
    });
    for (uint32_t i = 0; i < chunkCount; ++i)
        CHECK(chunkHits[i].load() == 1);

    // The Enabled filter in the parallel row form; the Changed filter with the context's since version.
    Query<Enabled<Toggle>, Read<Position>> enabled(world);
    std::atomic<uint32_t> enabledRows{ 0 };
    enabled.ParallelForEach(context.Get(), 1, [&](Entity, const Toggle&, const Position& p) {
        if (uint32_t(p.x) % 2 == 0)
            failures.fetch_add(1);
        enabledRows.fetch_add(1);
    });
    CHECK(enabledRows.load() == 1000 && failures.load() == 0);
    Query<Changed<Velocity>> changedVelocity(world);
    auto observer = MAKE_RC_OBJ_PTR(TestContext, world.Get(), jobs.Get());
    observer->lastRunVersion = context->version - 1;   // ParallelForChunks stamped Velocity with context->version
    observer->version = world->GetVersion();
    uint32_t changedChunks = 0;
    changedVelocity.ForEachChunk(observer.Get(), [&](const ChunkView&, const Velocity*) { ++changedChunks; });
    CHECK(changedChunks == chunkCount);   // another system sees every chunk that run wrote
    observer->NextRun();                  // its last run is now after the writes
    changedChunks = 0;
    changedVelocity.ForEach(observer.Get(), [&](Entity, const Velocity&) { ++changedChunks; });
    CHECK(changedChunks == 0);

    // Queries without data terms, in every serial form.
    Query<With<Toggle>> tagged(world);
    uint32_t rows = 0;
    tagged.ForEach(ChunkRequest{}, [&](Entity) { ++rows; });
    tagged.ForEach(context.Get(), [&](Entity) { ++rows; });
    tagged.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view) { rows += view.count; });
    tagged.ForEachChunk(context.Get(), [&](const ChunkView& view) { rows += view.count; });
    CHECK(rows == 8000);

    // Concurrent first calls after a structural change: one rebuild, every thread sees the whole list.
    Query<Read<Position>> reader(world);
    CHECK(reader.CountEntities() == 6000);
    Entity more[700];
    CreateMany(world, { ids.position, ids.velocity }, 700, more);
    const uint64_t rebuilds = QImpl(reader.Get()).GetCounters().rebuilds;
    std::atomic<uint32_t> start{ 0 };
    std::vector<uint32_t> totals(8, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < 8; ++t)
    {
        threads.emplace_back([&, t]() {
            while (start.load() == 0)
                std::this_thread::yield();
            ChunkView views[4];
            uint32_t total = 0;
            for (uint32_t first = 0;; first += 4)
            {
                const uint32_t written = reader.Get()->GetChunks(first, ChunkRequest{}, 4, views);
                if (written == 0)
                    break;
                for (uint32_t i = 0; i < written; ++i)
                    total += views[i].count;
            }
            totals[t] = total;
        });
    }
    start.store(1);
    for (std::thread& thread : threads)
        thread.join();
    for (const uint32_t total : totals)
        CHECK(total == 6700);
    CHECK(QImpl(reader.Get()).GetCounters().rebuilds == rebuilds + 1);
    CHECK(impl.GetValidationErrorCount() == 0);
}

// ------------------------------------------------------------------------------------------- determinism

namespace
{
    // A seeded script of immediate operations with queries along the way; the hash covers the order of every view and
    // row the queries visit.
    uint64_t RunQueryScript(uint64_t seed, bool smallChunks)
    {
        ScriptWorld script;
        script.world = smallChunks ? MakeWorld(false, 1024, 4096) : MakeWorld(false);
        IWorld* world = script.world;
        script.ids = RegisterAll(world);
        const Ids& ids = script.ids;
        Xorshift rng{ seed };
        Fnv fnv;
        const std::vector<std::vector<QueryTerm>> queries = {
            { Term(ids.position, TermOp::All, Access::Read), Term(ids.velocity, TermOp::Optional, Access::Write) },
            { Term(ids.toggle, TermOp::Enabled, Access::Read), Term(ids.selected, TermOp::Without, Access::Filter) },
            { Term(ids.layer, TermOp::All, Access::Read), Term(ids.joint, TermOp::Any, Access::Read),
              Term(ids.chunkBounds, TermOp::Any, Access::Write) },
            { Term(ids.position, TermOp::Changed, Access::Read), Term(ids.id, TermOp::Added, Access::Read) },
        };
        std::vector<nvrhi::AutoPtr<IQuery>> made;
        for (const std::vector<QueryTerm>& terms : queries)
            made.push_back(MakeQuery(world, terms, "script"));
        uint64_t lastRun = 0;
        for (uint32_t step = 0; step < 600; ++step)
        {
            RandomStep(script, rng);
            if (step % 20 != 19)
                continue;
            const uint64_t version = world->AdvanceVersion();
            for (size_t q = 0; q < made.size(); ++q)
            {
                ChunkView views[5];
                const uint32_t count = made[q]->GetChunkCount();
                fnv.Add(count);
                fnv.Add(made[q]->CountEntities());
                for (uint32_t first = 0; first < count; first += 5)
                {
                    const ChunkRequest request{ lastRun, version, nullptr };
                    const uint32_t written = made[q]->GetChunks(first, request, 5, views);
                    for (uint32_t i = 0; i < written; ++i)
                    {
                        fnv.Add(views[i].chunkIndex);
                        fnv.Add(views[i].count);
                        fnv.Add(views[i].entities, views[i].count * sizeof(Entity));
                        for (uint32_t row = 0; row < views[i].count; ++row)
                            fnv.Add(ViewRowPasses(world, queries[q], views[i], row));
                    }
                }
            }
            lastRun = version;
        }
        return fnv.hash;
    }
}

void test_determinism()
{
    const uint64_t first = RunQueryScript(0x5eed1234abcdull, true);
    // Other heap addresses for the second run.
    std::vector<std::unique_ptr<uint8_t[]>> perturb;
    for (uint32_t i = 0; i < 257; ++i)
        perturb.emplace_back(new uint8_t[16 + i * 37]);
    std::vector<void*> blocks;
    for (uint32_t i = 0; i < 33; ++i)
        blocks.push_back(ecs::HeapAllocate(64 + i * 128, 64));
    const uint64_t second = RunQueryScript(0x5eed1234abcdull, true);
    for (void* block : blocks)
        ecs::HeapFree(block);
    const uint64_t large = RunQueryScript(0x5eed1234abcdull, false);
    const uint64_t otherSeed = RunQueryScript(0x5eed1235abcdull, true);
    printf("test_ecs_query: determinism: visit hash %016llx (16 KiB chunks %016llx)\n", (unsigned long long)first,
           (unsigned long long)large);
    CHECK(first == second);
    CHECK(first != otherSeed && first != large);
}

// ------------------------------------------------------------------------------- steady-state allocations

void test_steady_state_allocations()
{
    auto world = MakeWorld(false, 16 * 1024, 64 * 1024, 8192);
    const Ids ids = RegisterAll(world);
    std::vector<Entity> base(2000);
    const Layer one{ 1 };
    std::vector<Layer> layers(1000, one);
    CreateMany(world, { ids.position, ids.velocity, ids.toggle, ids.joint, ids.chunkBounds }, 1000, base.data());
    CreateMany(world, { ids.position, ids.hidden, ids.layer }, 1000, base.data() + 1000,
               { nullptr, nullptr, layers.data() });
    for (uint32_t i = 0; i < 2000; i += 3)
        CHECK(world->AddComponent(base[i], ids.selected, nullptr) == nvrhi::FS_OK);
    auto context = MAKE_RC_OBJ_PTR(TestContext, world.Get(), nullptr);

    using Move = Query<Read<Position>, Write<Velocity>>;
    Move move(world);
    Query<Enabled<Toggle>, Optional<Velocity>, Without<Frozen>> toggled(world);
    using Layered = Query<ReadShared<Layer>, Read<Position>, With<Selected>>;
    Layered layered(world);
    Query<WriteChunk<ChunkBounds>, Read<BufferOf<Joint>>> bounds(world);
    Query<Changed<Position>, Added<Velocity>> changed(world);
    Query<Disabled<Hidden>, Write<Position>> hidden(world);
    nvrhi::AutoPtr<IQuery> raw = MakeQuery(world, {
        Term(ids.position, TermOp::All, Access::Read), Term(ids.velocity, TermOp::Optional, Access::Write),
        Term(ids.toggle, TermOp::Any, Access::Read), Term(ids.hidden, TermOp::Any, Access::Read),
        Term(ids.selected, TermOp::Without, Access::Filter), Term(ids.frozen, TermOp::Without, Access::Filter) });

    Entity batch[100];
    std::vector<Position> batchPositions(100);
    const ComponentId batchComponents[] = { ids.position, ids.velocity, ids.toggle, ids.joint, ids.chunkBounds };
    const void* batchValues[] = { batchPositions.data(), nullptr, nullptr, nullptr, nullptr };
    EntityCreateDesc batchDesc{};
    batchDesc.components = batchComponents;
    batchDesc.values = batchValues;
    batchDesc.componentCount = 5;
    batchDesc.entityCount = 100;
    ChunkView views[8];
    uint64_t sink = 0;

    auto round = [&]()
    {
        // Structural changes in existing archetypes and chunks: the queries rebuild within their capacity.
        CHECK(world->CreateEntities(batchDesc, batch) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < 50; ++i)
            CHECK(world->AddComponent(base[i], ids.frozen, nullptr) == nvrhi::FS_OK);
        context->NextRun();
        move.ForEachChunk(context.Get(), [&](const ChunkView& view, const Position* p, Velocity* v) {
            for (uint32_t row = 0; row < view.count; ++row)
                v[row].x = p[row].x;
        });
        move.ForEach(context.Get(), [&](Entity, const Position& p, Velocity& v) { v.y += p.y; });
        toggled.ForEach(ChunkRequest{}, [&](Entity, const Toggle& t, const Velocity* v) {
            sink += uint64_t(t.value) + (v ? 1 : 0);
        });
        layered.ForEachChunk(ChunkRequest{}, [&](const ChunkView& view, const Layer* layer, const Position*) {
            for (uint32_t row = 0; row < view.count; ++row)
                sink += Layered::RowPasses(view, row) ? layer->id : 0;
        });
        bounds.ForEach(context.Get(), [&](Entity, ChunkBounds& b, const BufferHeader& h) { b.hi += float(h.count); });
        changed.ForEachChunk(context.Get(), [&](const ChunkView& view, const Position*, const Velocity*) {
            sink += view.count;
        });
        hidden.ForEach(context.Get(), [&](Entity, const Hidden&, Position& p) { p.z += 1; });
        const ChunkRequest request = MakeChunkRequest(context.Get());
        const uint32_t count = raw->GetChunkCount();
        for (uint32_t first = 0; first < count; first += 8)
        {
            const uint32_t written = raw->GetChunks(first, request, 8, views);
            for (uint32_t i = 0; i < written; ++i)
                sink += views[i].count + (views[i].columns[1] ? 1 : 0);
        }
        sink += move.CountEntities() + raw->CountEntities() + toggled.Get()->GetChunkCount();
        CHECK(world->DestroyEntities(batch, 100) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < 50; ++i)
            CHECK(world->RemoveComponent(base[i], ids.frozen) == nvrhi::FS_OK);
        sink += move.CountEntities();
    };

    for (uint32_t i = 0; i < 3; ++i)
        round();
    const uint64_t news = g_NewCount.load();
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount();
    const uint64_t rebuilds = QImpl(raw).GetCounters().rebuilds;
    for (uint32_t i = 0; i < 5; ++i)
        round();
    const uint64_t newsAfter = g_NewCount.load();
    const uint64_t heapBlocksAfter = ecs::GetHeapAllocationCount();
    printf("test_ecs_query: steady state: %llu operator new, %llu storage blocks in 5 rounds, %llu rebuilds "
           "(sink %llu)\n",
           (unsigned long long)(newsAfter - news), (unsigned long long)(heapBlocksAfter - heapBlocks),
           (unsigned long long)(QImpl(raw).GetCounters().rebuilds - rebuilds), (unsigned long long)(sink & 1));
    CHECK(newsAfter == news);
    CHECK(heapBlocksAfter == heapBlocks);
    CHECK(QImpl(raw).GetCounters().rebuilds > rebuilds);   // the rounds did rebuild the lists

    // Both counters see allocations when there are some.
    int* probe = new int[16];
    sink += uintptr_t(probe) & 1;
    delete[] probe;
    CHECK(g_NewCount.load() > newsAfter);
    Entity probeEntity = Entity::Null;
    CreateMany(world, { ids.position, ids.big }, 1, &probeEntity);
    CHECK(move.CountEntities() > 0 && ecs::GetHeapAllocationCount() > heapBlocksAfter);
}

// ------------------------------------------------------------------------------------- performance smoke

void test_performance_smoke()
{
#if defined(NDEBUG)
    uint32_t count = 1000000;
#else
    uint32_t count = 100000;   // Debug: the same code, fewer rows
#endif
    if (const char* override = std::getenv("ECS_QUERY_PERF_ENTITIES"))   // other sizes, by hand
        if (const uint32_t requested = uint32_t(std::strtoul(override, nullptr, 10)))
            count = requested;
    auto world = MakeWorld(false, 16 * 1024, 64 * 1024, count + 16);
    const ComponentId a = RegisterComponent<PerfA>(world);
    const ComponentId b = RegisterComponent<PerfB>(world);
    std::vector<PerfA> as(count);
    std::vector<PerfB> bs(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        as[i] = PerfA{ float(i & 1023), 1, 2, 3 };
        bs[i] = PerfB{ 0, 0, 0, 0 };
    }
    std::vector<Entity> entities(count);
    CreateMany(world, { a, b }, count, entities.data(), { as.data(), bs.data() });
    Query<Read<PerfA>, Write<PerfB>> query(world, "perf");
    CHECK(query.CountEntities() == count);
    const ChunkRequest request{};

    auto plain = [&]() {
        const PerfA* pa = as.data();
        PerfB* pb = bs.data();
        for (uint32_t i = 0; i < count; ++i)
        {
            pb[i].x += pa[i].x * 0.5f;
            pb[i].y += pa[i].y * 0.5f;
            pb[i].z += pa[i].z * 0.5f;
            pb[i].w += pa[i].w * 0.5f;
        }
    };
    // The same chunk memory without the query: what the layout costs on its own.
    struct ChunkColumns
    {
        const PerfA* a;
        PerfB* b;
        uint32_t count;
    };
    std::vector<ChunkColumns> chunkColumns;
    query.ForEachChunk(request, [&](const ChunkView& view, const PerfA* pa, PerfB* pb) {
        chunkColumns.push_back(ChunkColumns{ pa, pb, view.count });
    });
    auto chunked = [&]() {
        for (const ChunkColumns& chunk : chunkColumns)
        {
            const PerfA* pa = chunk.a;
            PerfB* pb = chunk.b;
            const uint32_t n = chunk.count;
            for (uint32_t i = 0; i < n; ++i)
            {
                pb[i].x += pa[i].x * 0.5f;
                pb[i].y += pa[i].y * 0.5f;
                pb[i].z += pa[i].z * 0.5f;
                pb[i].w += pa[i].w * 0.5f;
            }
        }
    };
    auto forEachChunk = [&]() {
        query.ForEachChunk(request, [](const ChunkView& view, const PerfA* pa, PerfB* pb) {
            const uint32_t n = view.count;
            for (uint32_t i = 0; i < n; ++i)
            {
                pb[i].x += pa[i].x * 0.5f;
                pb[i].y += pa[i].y * 0.5f;
                pb[i].z += pa[i].z * 0.5f;
                pb[i].w += pa[i].w * 0.5f;
            }
        });
    };
    auto forEach = [&]() {
        query.ForEach(request, [](Entity, const PerfA& pa, PerfB& pb) {
            pb.x += pa.x * 0.5f;
            pb.y += pa.y * 0.5f;
            pb.z += pa.z * 0.5f;
            pb.w += pa.w * 0.5f;
        });
    };
    // The query's own cost: every chunk handed out (and its Write column stamped), no row touched.
    uint64_t rowsSeen = 0;
    auto iterationOnly = [&]() {
        query.ForEachChunk(request, [&](const ChunkView& view, const PerfA*, PerfB*) { rowsSeen += view.count; });
    };
    auto time = [](auto&& f) {
        const auto start = std::chrono::steady_clock::now();
        f();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    std::vector<double> tp, tk, tc, te, ti;
    for (uint32_t round = 0; round < 9; ++round)
    {
        tp.push_back(time(plain));
        tk.push_back(time(chunked));
        tc.push_back(time(forEachChunk));
        te.push_back(time(forEach));
        ti.push_back(time(iterationOnly));
    }
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    double checksum = 0;
    for (uint32_t i = 0; i < count; i += 4099)
        checksum += bs[i].x + static_cast<const PerfB*>(world->GetComponent(entities[i], b))->x;
    CHECK(rowsSeen == uint64_t(count) * 9);
    const double p = median(tp);
    const double k = median(tk);
    const double c = median(tc);
    const double e = median(te);
    const double i = median(ti);
    printf("test_ecs_query: performance, %u entities in %u chunks, Read<A> Write<B> (16 bytes each), median of 9:\n"
           "    plain arrays %.3f ms; the chunks without the query %.3f ms (x%.2f); ForEachChunk %.3f ms (x%.2f); "
           "ForEach %.3f ms (x%.2f); handing out the chunks alone %.3f ms (checksum %.0f)\n",
           count, query.Get()->GetChunkCount(), p, k, k / p, c, c / p, e, e / p, i, checksum);
}

// RowTracked (D9, plan step E7, item E7.5): ChunkView::rowVersions, and the chunk-only writer rule. A write that stamps
// only the chunk makes every row of it changed, also when a row write follows it before anyone reads; rows that a
// writer stamps through its view stay apart.
void test_row_versions_chunk_only_writer()
{
    auto world = MakeWorld(true);
    ComponentTypeDesc desc = ComponentTraits<Position>::Desc();
    desc.flags |= ComponentFlags::RowTracked;
    ComponentId position = ComponentId::Invalid;
    CHECK(world->RegisterComponent(desc, position) == nvrhi::FS_OK);
    const ComponentId id = RegisterComponent<Id>(world);
    Entity e[4];
    CreateMany(world, { position, id }, 4, e);   // one chunk, rows 0-3, at version 1
    nvrhi::AutoPtr<IQuery> writer = MakeQuery(world, { Term(position, TermOp::All, Access::Write) }, "writer");
    nvrhi::AutoPtr<IQuery> reader = MakeQuery(world, { Term(position, TermOp::Changed, Access::Read),
                                                       Term(id, TermOp::All, Access::Read) }, "reader");
    // The rows a reader of changed positions takes since `since` (IsRowChanged); none when the chunk is not handed out.
    auto changedRows = [&](uint64_t since)
    {
        std::vector<uint32_t> rows;
        ChunkView view{};
        if (reader->GetChunks(0, ChunkRequest{ since, c_WorldVersion, nullptr }, 1, &view) == 0)
            return rows;
        CHECK(view.count == 4 && view.rowVersions[0] != nullptr && view.rowVersions[1] == nullptr);
        for (uint32_t r = 0; r < view.count; ++r)
            if (view.IsRowChanged(0, r, since))
                rows.push_back(r);
        return rows;
    };
    const std::vector<uint32_t> all = { 0, 1, 2, 3 };
    CHECK(changedRows(c_AllVersions) == all);
    CHECK(changedRows(1).empty());
    const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(Record(world, e[0]).archetype);
    const uint32_t column = archetype.FindColumn(position);
    const uint64_t* rowVersions = archetype.RowVersions(0, column);

    // A Write term whose rows are not stamped: the chunk is newer than its rows, so it is wholly changed; the
    // reader's hand-out stamps every row with the chunk's version first.
    const uint64_t v2 = world->AdvanceVersion();
    ChunkView view{};
    CHECK(writer->GetChunks(0, ChunkRequest{ c_AllVersions, v2, nullptr }, 1, &view) == 1);
    CHECK(view.rowVersions[0] == rowVersions && rowVersions[1] == 1);
    CHECK(ecs::ChangedRowVersions(archetype, 0, column) == nullptr);
    CHECK(changedRows(1) == all);
    CHECK(ecs::ChangedRowVersions(archetype, 0, column) == rowVersions);
    for (uint32_t r = 0; r < 4; ++r)
        CHECK(rowVersions[r] == v2);

    // A chunk-only write, then a row write before any reader: the chunk-only write stays seen.
    const uint64_t v3 = world->AdvanceVersion();
    CHECK(writer->GetChunks(0, ChunkRequest{ c_AllVersions, v3, nullptr }, 1, &view) == 1);
    world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(e[2], position, c_WorldVersion) != nullptr);
    CHECK(changedRows(v2) == all);
    CHECK(changedRows(v3) == std::vector<uint32_t>{ 2 });

    // A writer that stamps the rows it wrote through its view: only those rows changed.
    const uint64_t v4 = world->GetVersion();
    const uint64_t v5 = world->AdvanceVersion();
    CHECK(writer->GetChunks(0, ChunkRequest{ c_AllVersions, v5, nullptr }, 1, &view) == 1);
    view.StampRow(0, 1, v5);
    CHECK(rowVersions[-1] == v5);
    CHECK(changedRows(v4) == std::vector<uint32_t>{ 1 });

    // A row write, then a chunk-only write of the same version (a setter, then a c_WorldVersion query, outside
    // phases): the chunk is not newer than its rows, so the hand-out stamps every row.
    world->AdvanceVersion();
    CHECK(world->GetComponentForWrite(e[0], position, c_WorldVersion) != nullptr);
    CHECK(writer->GetChunks(0, ChunkRequest{ c_AllVersions, c_WorldVersion, nullptr }, 1, &view) == 1);
    CHECK(changedRows(v5) == all);
    CHECK(Impl(world).GetValidationErrorCount() == 0);
}

// ------------------------------------------------------------ row versions through every path (stage T4, part I4)

namespace
{
    // An IStateWriter / IStateReader over memory, for the snapshot path.
    struct RowMemoryWriter final : IStateWriter
    {
        std::vector<uint8_t> bytes;
        nvrhi::FRESULT QueryInterface(nvrhi::FREFIID, void** ppInterface) noexcept override
        {
            if (ppInterface)
                *ppInterface = nullptr;
            return nvrhi::FE_NOINTERFACE;
        }
        nvrhi::FLONG AddRef() noexcept override { return 1; }
        nvrhi::FLONG Release() noexcept override { return 1; }
        nvrhi::FRESULT Write(const void* data, uint64_t size) noexcept override
        {
            const uint8_t* p = static_cast<const uint8_t*>(data);
            bytes.insert(bytes.end(), p, p + size);
            return nvrhi::FS_OK;
        }
    };
    struct RowMemoryReader final : IStateReader
    {
        const std::vector<uint8_t>& bytes;
        size_t position = 0;
        explicit RowMemoryReader(const std::vector<uint8_t>& b) : bytes(b) { }
        nvrhi::FRESULT QueryInterface(nvrhi::FREFIID, void** ppInterface) noexcept override
        {
            if (ppInterface)
                *ppInterface = nullptr;
            return nvrhi::FE_NOINTERFACE;
        }
        nvrhi::FLONG AddRef() noexcept override { return 1; }
        nvrhi::FLONG Release() noexcept override { return 1; }
        nvrhi::FRESULT Read(uint64_t size, void* data) noexcept override
        {
            if (size > bytes.size() - position)
                return nvrhi::FE_INVALID_ARGS;
            std::memcpy(data, bytes.data() + position, size_t(size));
            position += size_t(size);
            return nvrhi::FS_OK;
        }
    };

    ComponentId RegisterTracked(IWorld* world, ComponentTypeDesc desc)
    {
        desc.flags |= ComponentFlags::RowTracked;
        ComponentId id = ComponentId::Invalid;
        CHECK(world->RegisterComponent(desc, id) == nvrhi::FS_OK);
        return id;
    }

    // The version a reader of changed rows sees for a row (the ChunkView rule): its row version, or the chunk's column
    // version while a chunk-only write is pending (ChangedRowVersions null).
    uint64_t EffectiveRowVersion(IWorld* world, Entity entity, ComponentId component)
    {
        const ecs::EntityRecord& record = Record(world, entity);
        const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(record.archetype);
        const uint32_t column = archetype.FindColumn(component);
        CHECK(column != ecs::c_NoColumn && archetype.RowVersions(record.chunk, column) != nullptr);
        const uint64_t* rows = ecs::ChangedRowVersions(archetype, record.chunk, column);
        return rows ? rows[record.row] : archetype.Versions(record.chunk)[column];
    }

    // Every RowTracked column in use: no row newer than the column's newest row version, which is not newer than the
    // chunk's column version; columns without the flag have no row versions; rowsPerChunk rows fit each block.
    uint64_t CheckRowVersionStorage(IWorld* world)
    {
        ecs::WorldImpl& impl = Impl(world);
        uint64_t rowsChecked = 0;
        for (uint32_t a = 0; a < impl.GetArchetypeStorageCount(); ++a)
        {
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(a);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const uint32_t count = archetype.chunks[c].count;
                CHECK(count <= archetype.rowsPerChunk);
                for (uint32_t k = 0; k < uint32_t(archetype.columns.size()); ++k)
                {
                    const uint64_t* rows = archetype.RowVersions(c, k);
                    const ComponentTypeDesc* desc = world->GetComponentTypeDesc(archetype.components[k]);
                    CHECK((rows != nullptr) == (desc->size != 0 && !!(desc->flags & ComponentFlags::RowTracked)));
                    if (!rows || count == 0)
                        continue;
                    CHECK(rows[-1] <= archetype.Versions(c)[k]);
                    for (uint32_t r = 0; r < count; ++r)
                        CHECK(rows[r] <= rows[-1]);
                    rowsChecked += count;
                }
            }
        }
        return rowsChecked;
    }
}

// Row versions through every structural path, against a model of each live row's last write (stage T4; revision 1,
// E7.5's tests; I4 as built, deviations 1 and 2). A world of small chunks with two RowTracked columns (Position in
// every archetype, Velocity in some) and an untracked one, a tag that moves rows between archetypes. A seeded script
// of 600 steps, each after a new version or (one in four) at the same one: creates, destroys (swap-removes; chunks
// released and their blocks taken again), moves (a component or the tag added and removed), Instantiate, copies from
// another world, row writes (GetComponentForWrite), chunk-only writes (a Write hand-out that stamps nothing, at the
// world's version or a later one), writers that stamp the rows they wrote through the view, Changed readers, and
// snapshot save and load. The model: a row created, moved into another archetype or into a hole (its place changed),
// copied or loaded gets the version of the change; a row write its version; a chunk-only write every row of its
// chunks. After every step each live row's effective version (ChangedRowVersions, or the chunk's while a chunk-only
// write is pending) equals the model's, the storage invariants hold, and every few steps a Changed reader at several
// `since` versions takes exactly the rows the model says changed.
void test_row_versions_model()
{
    auto world = MakeWorld(true, 4096, 16 * 1024, 1024);
    const ComponentId position = RegisterTracked(world, ComponentTraits<Position>::Desc());
    const ComponentId velocity = RegisterTracked(world, ComponentTraits<Velocity>::Desc());
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId frozen = RegisterComponent<Frozen>(world);
    const ComponentId tracked[2] = { position, velocity };

    // Another world for CopyEntitiesFrom: the same types (its Position RowTracked, its Velocity not).
    auto other = MakeWorld(false, 4096, 16 * 1024, 1024);
    const ComponentId otherPosition = RegisterTracked(other, ComponentTraits<Position>::Desc());
    RegisterComponent<Velocity>(other);
    const ComponentId otherId = RegisterComponent<Id>(other);
    std::vector<Entity> sources(40);
    CreateMany(other, { otherPosition, otherId }, 40, sources.data());

    nvrhi::AutoPtr<IQuery> writers[2] = { MakeQuery(world, { Term(position, TermOp::All, Access::Write) }, "w pos"),
                                          MakeQuery(world, { Term(velocity, TermOp::All, Access::Write) }, "w vel") };
    nvrhi::AutoPtr<IQuery> readers[2] = { MakeQuery(world, { Term(position, TermOp::Changed, Access::Read) }, "r pos"),
                                          MakeQuery(world, { Term(velocity, TermOp::Changed, Access::Read) }, "r vel") };

    std::map<Entity, uint64_t> model[2];   // per tracked component: each live row's last write
    std::vector<Entity> alive;
    struct Place
    {
        uint32_t archetype, chunk, row;
        bool operator==(const Place& o) const { return archetype == o.archetype && chunk == o.chunk && row == o.row; }
    };
    auto placeOf = [&](Entity e)
    {
        const ecs::EntityRecord& record = Record(world, e);
        return Place{ record.archetype, record.chunk, record.row };
    };
    // Around a structural call: rows whose place changed were moved (into a hole or another archetype) and get the
    // version of the change; rows that left a component leave the model.
    auto structural = [&](auto&& call)
    {
        std::map<Entity, Place> before;
        for (Entity e : alive)
            before[e] = placeOf(e);
        call();
        const uint64_t v = world->GetVersion();
        for (Entity e : alive)
        {
            const auto it = before.find(e);
            if (it == before.end() || !(it->second == placeOf(e)))
                for (uint32_t t = 0; t < 2; ++t)
                    if (world->HasComponent(e, tracked[t]))
                        model[t][e] = v;
            for (uint32_t t = 0; t < 2; ++t)
                if (!world->HasComponent(e, tracked[t]))
                    model[t].erase(e);
        }
    };
    auto checkModel = [&]()
    {
        for (uint32_t t = 0; t < 2; ++t)
        {
            uint32_t present = 0;
            for (Entity e : alive)
            {
                if (!world->HasComponent(e, tracked[t]))
                    continue;
                ++present;
                const auto it = model[t].find(e);
                CHECK(it != model[t].end());
                if (EffectiveRowVersion(world, e, tracked[t]) != it->second)
                {
                    printf("  row version of %llu (component %u): %llu, the model's %llu\n", (unsigned long long)e,
                           t, (unsigned long long)EffectiveRowVersion(world, e, tracked[t]),
                           (unsigned long long)it->second);
                    CHECK(false);
                }
            }
            CHECK(present == model[t].size());
        }
        CheckRowVersionStorage(world);
    };
    // A Changed reader at `since`: the rows it takes are exactly the model's rows newer than `since`.
    uint64_t readerRows = 0;
    auto checkReaders = [&](uint64_t since)
    {
        for (uint32_t t = 0; t < 2; ++t)
        {
            std::vector<Entity> seen, expected;
            ChunkView views[16];
            const uint32_t chunks = readers[t]->GetChunkCount();
            for (uint32_t first = 0; first < chunks; first += 16)
            {
                const uint32_t n = readers[t]->GetChunks(first, ChunkRequest{ since, c_WorldVersion, nullptr }, 16,
                                                         views);
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t r = 0; r < views[i].count; ++r)
                        if (views[i].IsRowChanged(0, r, since))
                            seen.push_back(views[i].entities[r]);
            }
            for (const auto& [e, v] : model[t])
                if (v > since)
                    expected.push_back(e);
            std::sort(seen.begin(), seen.end());
            CHECK(seen == expected);
            readerRows += seen.size();
        }
    };

    Xorshift rng{ 0x7ab1e5eedull };
    uint32_t counts[12] = {};
    std::vector<uint64_t> versions;
    for (uint32_t step = 0; step < 600; ++step)
    {
        if (rng.Below(4) != 0)
            world->AdvanceVersion();
        const uint64_t v = world->GetVersion();
        versions.push_back(v);
        const uint32_t op = alive.size() < 50 ? 0 : rng.Below(12);
        ++counts[op];
        switch (op)
        {
        case 0:   // create, with or without Velocity, some frozen
        case 1:
        {
            const uint32_t n = 1 + rng.Below(60);
            std::vector<Entity> made(n);
            const uint32_t shape = rng.Below(3);
            structural([&]()
            {
                if (shape == 0)
                    CreateMany(world, { position, id }, n, made.data());
                else if (shape == 1)
                    CreateMany(world, { position, velocity, id }, n, made.data());
                else
                    CreateMany(world, { position, velocity, id, frozen }, n, made.data());
                alive.insert(alive.end(), made.begin(), made.end());
            });
            break;
        }
        case 2:   // destroy: swap-removes; emptied chunks are released, their blocks kept for the next chunk there
        case 3:
        {
            const uint32_t n = 1 + rng.Below(alive.size() > 400 ? 120 : 30);
            std::vector<Entity> victims;
            for (uint32_t k = 0; k < n && !alive.empty(); ++k)
            {
                const uint32_t i = rng.Below(uint32_t(alive.size()));
                victims.push_back(alive[i]);
                alive[i] = alive.back();
                alive.pop_back();
            }
            structural([&]()
            {
                CHECK(world->DestroyEntities(victims.data(), uint32_t(victims.size())) == nvrhi::FS_OK);
            });
            for (Entity e : victims)
                for (uint32_t t = 0; t < 2; ++t)
                    model[t].erase(e);
            break;
        }
        case 4:   // moves: Velocity or the tag added or removed
        {
            for (uint32_t k = 0, n = 1 + rng.Below(20); k < n; ++k)
            {
                const Entity e = alive[rng.Below(uint32_t(alive.size()))];
                const ComponentId c = rng.Below(2) ? velocity : frozen;
                structural([&]()
                {
                    const bool had = world->HasComponent(e, c);
                    CHECK((had ? world->RemoveComponent(e, c) : world->AddComponent(e, c, nullptr)) == nvrhi::FS_OK);
                });
            }
            break;
        }
        case 5:   // Instantiate
        {
            const Entity prefab = alive[rng.Below(uint32_t(alive.size()))];
            const uint32_t n = 1 + rng.Below(30);
            std::vector<Entity> made(n);
            structural([&]()
            {
                CHECK(world->Instantiate(prefab, n, made.data()) == nvrhi::FS_OK);
                alive.insert(alive.end(), made.begin(), made.end());
            });
            break;
        }
        case 6:   // copies from another world
        {
            const uint32_t n = 1 + rng.Below(uint32_t(sources.size()));
            std::vector<Entity> made(n);
            structural([&]()
            {
                CHECK(world->CopyEntitiesFrom(other, sources.data(), n, made.data()) == nvrhi::FS_OK);
                alive.insert(alive.end(), made.begin(), made.end());
            });
            break;
        }
        case 7:   // row writes
        case 8:
        {
            for (uint32_t k = 0, n = 1 + rng.Below(40); k < n; ++k)
            {
                const Entity e = alive[rng.Below(uint32_t(alive.size()))];
                const uint32_t t = rng.Below(2);
                if (!world->HasComponent(e, tracked[t]))
                    continue;
                CHECK(world->GetComponentForWrite(e, tracked[t], c_WorldVersion) != nullptr);
                model[t][e] = v;
            }
            break;
        }
        case 9:   // a chunk-only write of a range of chunks: c_WorldVersion, or a new version named explicitly
        {
            const uint32_t t = rng.Below(2);
            const uint64_t writeVersion = rng.Below(2) ? c_WorldVersion : world->AdvanceVersion();
            const uint64_t stamp = writeVersion == c_WorldVersion ? v : writeVersion;
            const uint32_t chunks = writers[t]->GetChunkCount();
            if (chunks == 0)
                break;
            const uint32_t first = rng.Below(chunks);
            ChunkView views[16];
            const uint32_t n = writers[t]->GetChunks(first, ChunkRequest{ c_AllVersions, writeVersion, nullptr },
                                                     1 + rng.Below(16), views);
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t r = 0; r < views[i].count; ++r)
                    model[t][views[i].entities[r]] = stamp;
            break;
        }
        case 10:   // a writer that stamps the rows it wrote, at a new version (not the same one: deviation 2 there)
        {
            const uint32_t t = rng.Below(2);
            const uint64_t writeVersion = world->AdvanceVersion();
            const uint32_t chunks = writers[t]->GetChunkCount();
            if (chunks == 0)
                break;
            ChunkView view{};
            if (writers[t]->GetChunks(rng.Below(chunks), ChunkRequest{ c_AllVersions, writeVersion, nullptr }, 1,
                                      &view) != 1)
                break;
            CHECK(view.rowVersions[0] != nullptr);
            uint32_t stamped = 0;
            for (uint32_t r = 0; r < view.count; ++r)
            {
                if (rng.Below(3) != 0)
                    continue;
                view.StampRow(0, r, writeVersion);
                model[t][view.entities[r]] = writeVersion;
                ++stamped;
            }
            if (stamped == 0)   // a hand-out that stamps no row is a chunk-only write
                for (uint32_t r = 0; r < view.count; ++r)
                    model[t][view.entities[r]] = writeVersion;
            break;
        }
        case 11:   // snapshot: save, then load; every row loaded gets the load's version
        {
            if (rng.Below(4) != 0)
            {
                checkReaders(versions[rng.Below(uint32_t(versions.size()))]);
                break;
            }
            RowMemoryWriter writer;
            CHECK(world->SaveSnapshot(&writer) == nvrhi::FS_OK);
            RowMemoryReader reader(writer.bytes);
            CHECK(world->LoadSnapshot(&reader) == nvrhi::FS_OK);
            const uint64_t loaded = EffectiveRowVersion(world, alive.front(), position);
            CHECK(loaded >= v);
            for (uint32_t t = 0; t < 2; ++t)
                for (auto& [e, version] : model[t])
                    version = loaded;
            break;
        }
        }
        checkModel();
        if (step % 25 == 24)
        {
            checkReaders(c_AllVersions);
            checkReaders(versions[versions.size() / 2]);
            checkReaders(versions[versions.size() - 2]);
        }
    }
    const uint64_t rows = CheckRowVersionStorage(world);
    printf("  600 steps (creates %u, destroys %u, moves %u, instantiates %u, copies %u, row writes %u, chunk-only "
           "writes %u, stamping writers %u, snapshot or reader steps %u): %zu live rows, %llu tracked rows checked, "
           "%llu rows taken by readers\n", counts[0] + counts[1], counts[2] + counts[3], counts[4], counts[5], counts[6],
           counts[7] + counts[8], counts[9], counts[10], counts[11], alive.size(), (unsigned long long)rows,
           (unsigned long long)readerRows);
    CHECK(Impl(world).GetValidationErrorCount() == 0);
}

// The chunk-only writer rule's two deviations (I4 as built, deviations 1 and 2), step by step on one chunk, for both
// orders of each pair and through every kind of stamp: a pending chunk-only write is spread before a row write, a
// create into the chunk, a swap-remove and a move out of it stamp anything (deviation 1); a chunk-only write of a
// version some row already has stamps every row at once (deviation 2), whether that row got it from a row write, a
// create or a move.
void test_row_versions_chunk_only_rule()
{
    auto world = MakeWorld(true);
    const ComponentId position = RegisterTracked(world, ComponentTraits<Position>::Desc());
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId frozen = RegisterComponent<Frozen>(world);
    nvrhi::AutoPtr<IQuery> writer = MakeQuery(world, { Term(position, TermOp::All, Access::Write),
                                                       Term(frozen, TermOp::Without, Access::Filter) }, "writer");
    // A Changed reader's rows since `since` in the unfrozen archetype, by entity.
    nvrhi::AutoPtr<IQuery> reader = MakeQuery(world, { Term(position, TermOp::Changed, Access::Read),
                                                       Term(frozen, TermOp::Without, Access::Filter) }, "reader");
    auto changed = [&](uint64_t since)
    {
        std::vector<Entity> rows;
        ChunkView view{};
        if (reader->GetChunkCount() && reader->GetChunks(0, ChunkRequest{ since, c_WorldVersion, nullptr }, 1, &view))
            for (uint32_t r = 0; r < view.count; ++r)
                if (view.IsRowChanged(0, r, since))
                    rows.push_back(view.entities[r]);
        std::sort(rows.begin(), rows.end());
        return rows;
    };
    auto sorted = [](std::vector<Entity> list)
    {
        std::sort(list.begin(), list.end());
        return list;
    };
    auto chunkOnly = [&](uint64_t writeVersion)
    {
        ChunkView view{};
        CHECK(writer->GetChunkCount() == 1);
        CHECK(writer->GetChunks(0, ChunkRequest{ c_AllVersions, writeVersion, nullptr }, 1, &view) == 1);
    };

    std::vector<Entity> e(6);
    CreateMany(world, { position, id }, 6, e.data());
    const uint32_t cases = 5;
    for (uint32_t k = 0; k < cases; ++k)
    {
        // Deviation 1: a chunk-only write at u, then (before any reader) a stamp at w > u of one kind.
        const uint64_t before = world->GetVersion();
        const uint64_t u = world->AdvanceVersion();
        chunkOnly(c_WorldVersion);
        const uint64_t w = world->AdvanceVersion();
        std::map<Entity, uint64_t> placesBefore;   // chunk and row: a row whose place changes was moved at w
        for (Entity x : e)
            placesBefore[x] = (uint64_t(Record(world, x).chunk) << 32) | Record(world, x).row;
        std::vector<Entity> rowsAtW;
        switch (k)
        {
        case 0:   // a row write
            CHECK(world->GetComponentForWrite(e[1], position, c_WorldVersion) != nullptr);
            rowsAtW = { e[1] };
            break;
        case 1:   // a create into the chunk
        {
            Entity made{};
            CreateMany(world, { position, id }, 1, &made);
            e.push_back(made);
            rowsAtW = { made };
            break;
        }
        case 2:   // a swap-remove: the chunk's last row moves into the hole
            CHECK(world->DestroyEntities(&e[0], 1) == nvrhi::FS_OK);
            e.erase(e.begin());
            break;
        case 3:   // a move out of the archetype (the last row fills the hole) and back (appended as the last row)
            CHECK(world->AddComponent(e[2], frozen, nullptr) == nvrhi::FS_OK);
            CHECK(world->RemoveComponent(e[2], frozen) == nvrhi::FS_OK);
            rowsAtW = { e[2] };
            break;
        case 4:   // a writer that stamps one row through its view
        {
            ChunkView view{};
            CHECK(writer->GetChunks(0, ChunkRequest{ c_AllVersions, w, nullptr }, 1, &view) == 1);
            view.StampRow(0, 3, w);
            rowsAtW = { view.entities[3] };
            break;
        }
        }
        for (Entity x : e)
        {
            const auto it = placesBefore.find(x);
            const uint64_t place = (uint64_t(Record(world, x).chunk) << 32) | Record(world, x).row;
            if (it != placesBefore.end() && it->second != place &&
                std::find(rowsAtW.begin(), rowsAtW.end(), x) == rowsAtW.end())
                rowsAtW.push_back(x);
        }
        CHECK(k == 0 || k == 1 || k == 4 || rowsAtW.size() >= 1);
        CHECK(changed(before) == sorted(e));      // the chunk-only write at u is still seen by every row
        CHECK(changed(u) == sorted(rowsAtW));     // and only the stamped rows are newer
        for (Entity x : e)
            CHECK(EffectiveRowVersion(world, x, position) ==
                  (std::find(rowsAtW.begin(), rowsAtW.end(), x) != rowsAtW.end() ? w : u));
    }
    for (uint32_t k = 0; k < 3; ++k)
    {
        // Deviation 2: a row gets version v (a row write, a create, a move into the chunk), then a chunk-only write of
        // the same version v (c_WorldVersion, no new version between): every row is stamped v at once.
        const uint64_t before = world->GetVersion();
        const uint64_t v = world->AdvanceVersion();
        switch (k)
        {
        case 0:
            CHECK(world->GetComponentForWrite(e[0], position, c_WorldVersion) != nullptr);
            break;
        case 1:
        {
            Entity made{};
            CreateMany(world, { position, id }, 1, &made);
            e.push_back(made);
            break;
        }
        case 2:
            CHECK(world->DestroyEntities(&e[1], 1) == nvrhi::FS_OK);   // the last row moves into the hole at v
            e.erase(e.begin() + 1);
            break;
        }
        chunkOnly(c_WorldVersion);
        CHECK(changed(before) == sorted(e));
        for (Entity x : e)
            CHECK(EffectiveRowVersion(world, x, position) == v);
    }
    CheckRowVersionStorage(world);
    CHECK(Impl(world).GetValidationErrorCount() == 0);
    printf("  deviation 1 through 5 kinds of stamp, deviation 2 through 3 kinds of row version\n");
}

// Snapshots leave row versions out (I4 as built; stage T8, the killer of a mutant that records RowTracked in the
// component table): two worlds made and edited alike, one with Position and Velocity RowTracked, give the same snapshot
// bytes and hash; each snapshot loads into the other world, and the loaded tracked world's rows are stamped.
void test_row_versions_snapshot_bytes()
{
    nvrhi::AutoPtr<IWorld> worlds[2] = { MakeWorld(true, 4096), MakeWorld(true, 4096) };
    std::vector<uint8_t> bytes[2];
    for (uint32_t tracked = 0; tracked < 2; ++tracked)
    {
        IWorld* world = worlds[tracked];
        const ComponentId position = tracked ? RegisterTracked(world, ComponentTraits<Position>::Desc())
                                             : RegisterComponent<Position>(world);
        const ComponentId velocity = tracked ? RegisterTracked(world, ComponentTraits<Velocity>::Desc())
                                             : RegisterComponent<Velocity>(world);
        const ComponentId id = RegisterComponent<Id>(world);
        std::vector<Entity> e(300);
        std::vector<Position> positions(300);
        for (uint32_t i = 0; i < 300; ++i)
            positions[i] = Position{ float(i), 1, 2 };
        CreateMany(world, { position, velocity, id }, 300, e.data(), { positions.data(), nullptr, nullptr });
        world->AdvanceVersion();
        for (uint32_t i = 0; i < 300; i += 7)
            static_cast<Position*>(world->GetComponentForWrite(e[i], position, c_WorldVersion))->y = 5;
        CHECK(world->DestroyEntities(e.data() + 100, 20) == nvrhi::FS_OK);
        RowMemoryWriter writer;
        CHECK(world->SaveSnapshot(&writer) == nvrhi::FS_OK);
        bytes[tracked] = writer.bytes;
    }
    CHECK(bytes[0] == bytes[1] && !bytes[0].empty());
    CHECK(worlds[0]->ComputeHash() == worlds[1]->ComputeHash());
    for (uint32_t into = 0; into < 2; ++into)
    {
        RowMemoryReader reader(bytes[1 - into]);
        CHECK(worlds[into]->LoadSnapshot(&reader) == nvrhi::FS_OK);
    }
    CHECK(worlds[0]->ComputeHash() == worlds[1]->ComputeHash());
    CheckRowVersionStorage(worlds[1]);
    CHECK(Impl(worlds[0]).GetValidationErrorCount() == 0 && Impl(worlds[1]).GetValidationErrorCount() == 0);
    printf("  snapshots of a tracked and an untracked world: %zu bytes, equal; loaded into each other\n",
           bytes[0].size());
}

// --------------------------------------------------------------------- chunk prefetch (E7.4 remedy 1, part I5)

// SetQueryChunkPrefetch: off by default; every typed form visits the same chunks, rows and values, in the same order,
// with the same stamps, with it on and off. Small chunks, so that batches of 16 views are full and the next chunk is
// prefetched across many of them; an Optional term absent in one archetype, a per-chunk term, and a form without data.
// One world of the case below: `plain` rows of Position and Velocity (with Big when `big`, rows of 224 bytes that take
// the large chunk size), `toggled` with Toggle too, `bounded` with a per-chunk ChunkBounds; returns the chunk count.
uint32_t run_chunk_prefetch(uint32_t chunkSize, uint32_t largeChunkSize, uint32_t plain, uint32_t toggled,
                            uint32_t bounded, bool big)
{
    const bool prefetchFound = GetQueryChunkPrefetch();
    auto world = MakeWorld(true, chunkSize, largeChunkSize);
    const Ids ids = RegisterAll(world);
    const uint32_t total = plain + toggled + bounded;
    std::vector<Entity> e(total);
    std::vector<Position> positions(total);
    for (uint32_t i = 0; i < total; ++i)
        positions[i] = Position{ float(i), float(i % 7), 2 };
    if (plain && big)
        CreateMany(world, { ids.position, ids.velocity, ids.big }, plain, e.data(), { positions.data(), nullptr,
                                                                                     nullptr });
    else if (plain)
        CreateMany(world, { ids.position, ids.velocity }, plain, e.data(), { positions.data(), nullptr });
    if (toggled)
        CreateMany(world, { ids.position, ids.velocity, ids.toggle }, toggled, e.data() + plain,
                   { positions.data() + plain, nullptr, nullptr });
    if (bounded)
        CreateMany(world, { ids.position, ids.velocity, ids.chunkBounds }, bounded, e.data() + plain + toggled,
                   { positions.data() + plain + toggled, nullptr, nullptr });
    auto jobs = MAKE_RC_OBJ_PTR(TestJobSystem, 4u);
    auto context = MAKE_RC_OBJ_PTR(TestContext, world.Get(), jobs.Get());
    Query<Read<Position>, Write<Velocity>, Optional<Toggle>> query(world, "prefetch");
    Query<ReadChunk<ChunkBounds>, Read<Position>> perChunk(world, "prefetch per chunk");
    Query<With<Toggle>> withoutData(world, "prefetch without data");
    const uint32_t chunkCount = query.Get()->GetChunkCount();
    CHECK(chunkCount > 0);
    const uint32_t tableSize = Impl(world).GetEntityTableSize();

    // Everything the forms see and write, as numbers; stamps relative to the run's version.
    auto trace = [&](bool prefetch)
    {
        SetQueryChunkPrefetch(prefetch);
        CHECK(GetQueryChunkPrefetch() == prefetch);
        std::vector<uint64_t> out;
        context->NextRun();
        query.ForEachChunk(context.Get(), [&](const ChunkView& view, const Position* p, Velocity* v,
                                              const Toggle* t) {
            out.push_back(view.chunkIndex);
            out.push_back(view.count);
            out.push_back(t != nullptr);
            for (uint32_t row = 0; row < view.count; ++row)
            {
                out.push_back(GetEntityIndex(view.entities[row]));
                v[row] = Velocity{ p[row].x * 2, 0, 0 };
            }
        });
        query.ForEach(context.Get(), [&](Entity entity, const Position& p, Velocity& v, const Toggle* t) {
            out.push_back(GetEntityIndex(entity));
            out.push_back(t != nullptr);
            v.y = p.y + 1;
        });
        perChunk.ForEach(context.Get(), [&](Entity entity, const ChunkBounds& bounds, const Position& p) {
            out.push_back(GetEntityIndex(entity));
            out.push_back(uint64_t(p.x));
            out.push_back(&bounds == world->GetComponent(entity, ids.chunkBounds));
        });
        uint64_t tagged = 0;
        withoutData.ForEach(context.Get(), [&](Entity entity) { tagged += GetEntityIndex(entity); });
        out.push_back(tagged);
        context->NextRun();
        std::unique_ptr<std::atomic<uint32_t>[]> chunkHits(new std::atomic<uint32_t>[chunkCount]());
        query.ParallelForChunks(context.Get(), 3, [&](const ChunkView& view, const Position* p, Velocity* v,
                                                      const Toggle*) {
            chunkHits[view.chunkIndex].fetch_add(1);
            for (uint32_t row = 0; row < view.count; ++row)
                v[row].z = p[row].x + float(view.chunkIndex);
        });
        std::unique_ptr<std::atomic<uint32_t>[]> rowHits(new std::atomic<uint32_t>[tableSize]());
        query.ParallelForEach(context.Get(), 2, [&](Entity entity, const Position&, Velocity& v, const Toggle*) {
            rowHits[GetEntityIndex(entity)].fetch_add(1);
            v.x += 1;
        });
        for (uint32_t c = 0; c < chunkCount; ++c)
            out.push_back(chunkHits[c].load());
        for (const Entity entity : e)
        {
            const Velocity* v = static_cast<const Velocity*>(world->GetComponent(entity, ids.velocity));
            out.push_back(rowHits[GetEntityIndex(entity)].load());
            out.push_back(uint64_t(v->x) * 1000003 + uint64_t(v->y) * 1009 + uint64_t(v->z));
            out.push_back(context->version - ColumnVersion(world, entity, ids.velocity));
        }
        return out;
    };
    const std::vector<uint64_t> off = trace(false);
    const std::vector<uint64_t> on = trace(true);
    SetQueryChunkPrefetch(false);
    CHECK(off.size() > 3 * total && off == on);
    CHECK(!GetQueryChunkPrefetch());
    SetQueryChunkPrefetch(prefetchFound);
    CHECK(Impl(world).GetValidationErrorCount() == 0);
    return chunkCount;
}

void test_chunk_prefetch()
{
    // Stage T1: on in the switches-on runs (tests/src/e7_switches.cpp); the value found is put back at the end.
    CHECK(GetQueryChunkPrefetch() == donut::tests::E7SwitchRequested("prefetch"));
    CHECK(run_chunk_prefetch(1024, 4096, 1800, 700, 500, false) > 40);
}

// Stage T5: the same over many chunk sizes and counts: 1 KiB to 256 KiB chunks, small rows and rows that take the
// large chunk size, worlds of one row, of 15, 16 and 17 chunks (a batch of 16 views and its edges), and of many.
void test_chunk_prefetch_sizes()
{
    uint32_t worlds = 0, chunks = 0;
    for (const uint32_t chunkSize : { 1024u, 4096u, 16u * 1024, 64u * 1024, 256u * 1024 })
    {
        for (const bool big : { false, true })
        {
            const uint32_t largeChunkSize = std::max(chunkSize, 64u * 1024);
            // rows per chunk of the plain archetype, from a first world of one chunk's worth
            const uint32_t perChunk = std::max(1u, (big ? largeChunkSize : chunkSize) / (big ? 240u : 40u));
            for (const uint32_t plain : { 1u, 15 * perChunk, 16 * perChunk, 17 * perChunk + 3, 40 * perChunk + 1 })
            {
                if (plain > 200000)
                    continue;   // the largest chunks: the batch edges are enough
                chunks += run_chunk_prefetch(chunkSize, largeChunkSize, plain, plain / 3 + 1, plain / 5 + 1, big);
                ++worlds;
            }
        }
    }
    printf("  %u worlds, %u chunks: every typed form equal with and without prefetch\n", worlds, chunks);
}

int main(int, char**)
{
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "validation", &test_validation },
        { "matching_and_order", &test_matching_and_order },
        { "model", &test_model },
        { "change_detection", &test_change_detection },
        { "enabled_bits", &test_enabled_bits },
        { "sparse_filters", &test_sparse_filters },
        { "row_filter_skips", &test_row_filter_skips },
        { "shared_and_chunk_components", &test_shared_and_chunk_components },
        { "buffers", &test_buffers },
        { "resources", &test_resources },
        { "query_lifetime", &test_query_lifetime },
        { "typed_forms", &test_typed_forms },
        { "determinism", &test_determinism },
        { "row_versions_chunk_only_writer", &test_row_versions_chunk_only_writer },
        { "row_versions_model", &test_row_versions_model },
        { "row_versions_chunk_only_rule", &test_row_versions_chunk_only_rule },
        { "row_versions_snapshot_bytes", &test_row_versions_snapshot_bytes },
        { "chunk_prefetch", &test_chunk_prefetch },
        { "chunk_prefetch_sizes", &test_chunk_prefetch_sizes },
        { "steady_state_allocations", &test_steady_state_allocations },
        { "performance_smoke", &test_performance_smoke },
    };
    try
    {
        for (const Test& test : tests)
        {
            test.run();
            printf("test_ecs_query: %s passed\n", test.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
