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

// ECS observers (aggregate docs/plans/2026-11-21-donut-ecs.md, step E1.4; design 6.6): registration and its validation,
// every event on every storage kind, batches and where they end, a brute-force model that predicts every callback of
// seeded random immediate operations and playbacks, cascades and their limit, re-entrancy, snapshot loads, lifetime,
// and an allocation counter that proves steady-state delivery allocates nothing. The internals are read through the
// private headers (src/engine/EcsObserverImpl.h, EcsCommandBufferImpl.h, EcsWorldImpl.h).

#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsWorld.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsCommandBufferImpl.h"
#include "EcsObserverImpl.h"
#include "EcsWorldImpl.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test, and donut_engine, which links statically) is counted. The storage's
// own blocks come from nvrhi_core's allocator, which ecs::GetHeapAllocationCount counts.
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

namespace obs_test
{
    using donut::engine::ComponentFlags;

    struct A { uint32_t v; };
    DONUT_ECS_COMPONENT(A, "9a3470e0-2d93-4dc8-97e2-8f3331ce428f", DONUT_ECS_FIELDS(v));

    struct B { uint32_t v; };
    DONUT_ECS_COMPONENT(B, "41d0d81a-5e0a-4e9e-9861-e7c7f2874978", DONUT_ECS_FIELDS(v));

    struct C { };
    DONUT_ECS_TAG(C, "dfc5670e-d6b0-4ce5-88c3-9504e20f9082");

    struct Sh { uint32_t key; };
    DONUT_ECS_COMPONENT_EX(Sh, "c3add5a1-45ae-4456-a26c-e9fa48633651", ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(key));

    struct S { uint32_t v; };
    DONUT_ECS_COMPONENT_EX(S, "46836c93-8894-456f-8c95-bfb586d592c2", ComponentFlags::Sparse, 0, DONUT_ECS_FIELDS(v));

    struct Sel { };
    DONUT_ECS_TAG_EX(Sel, "d55bfeca-ad17-4dba-ba1b-d2f031a1892b", ComponentFlags::Sparse);

    struct Item { uint32_t v; };
    DONUT_ECS_BUFFER(Item, "ea1f5709-e44b-4676-9774-91e229e97555", 2, DONUT_ECS_FIELDS(v));

    struct Ch { uint32_t v; };
    DONUT_ECS_COMPONENT_EX(Ch, "8f1c8c4f-218a-4480-a04d-5cdc778498e0", ComponentFlags::Chunk, 0, DONUT_ECS_FIELDS(v));

    struct Tog { uint32_t v; };
    DONUT_ECS_COMPONENT_EX(Tog, "91dc6cf4-f4a5-4dc5-be9a-3227d569d83f", ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(v));

    struct Res { uint32_t v; };
    DONUT_ECS_COMPONENT_EX(Res, "7d865077-cf88-44ee-a673-0763f8a565ed", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(v));

    // A type registered only inside an observer (refused there).
    struct Late { uint32_t v; };
    DONUT_ECS_COMPONENT(Late, "13475f1c-bf87-4ef0-ad64-7c7a52645405", DONUT_ECS_FIELDS(v));

    // Non-trivial: a counted reference to a payload, so that OnRemove can prove the value is still alive.
    struct Payload { int64_t refs = 0; };

    struct Handle
    {
        Payload* payload = nullptr;

        Handle() noexcept = default;
        explicit Handle(Payload* p) noexcept : payload(p)
        {
            if (payload)
                ++payload->refs;
        }
        Handle(const Handle& other) noexcept : payload(other.payload)
        {
            if (payload)
                ++payload->refs;
        }
        Handle(Handle&& other) noexcept : payload(other.payload) { other.payload = nullptr; }
        ~Handle()
        {
            if (payload)
                --payload->refs;
        }
        Handle& operator=(const Handle&) = delete;
        Handle& operator=(Handle&&) = delete;
    };
    DONUT_ECS_COMPONENT(Handle, "57363813-e08e-418a-a98d-e20875e810e7", DONUT_ECS_FIELDS());
}

using namespace donut::engine;
using namespace obs_test;
namespace ecs = donut::engine::ecs;

// Inside an observer (a noexcept callback) a failed check cannot throw: EXPECT notes the first failing line, and
// CHECK_EXPECTATIONS throws it afterwards.
namespace
{
    int g_FailedLine = 0;
}
#define EXPECT(condition)                                   \
    do                                                      \
    {                                                       \
        if (!(condition) && g_FailedLine == 0)              \
            g_FailedLine = __LINE__;                        \
    } while (false)
#define CHECK_EXPECTATIONS()                                                                                          \
    do                                                                                                                \
    {                                                                                                                 \
        const int line = g_FailedLine;                                                                                \
        g_FailedLine = 0;                                                                                             \
        if (line != 0)                                                                                                \
            throw std::runtime_error(std::string(__FILE__) + ": expectation inside an observer failed at line " +   \
                                     std::to_string(line));                                                           \
    } while (false)

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

    struct Ids
    {
        ComponentId a, b, c, sh, s, sel, item, ch, tog, res, handle;
    };

    Ids Register(IWorld* world)
    {
        Ids ids{};
        ids.a = RegisterComponent<A>(world);
        ids.b = RegisterComponent<B>(world);
        ids.c = RegisterComponent<C>(world);
        ids.sh = RegisterComponent<Sh>(world);
        ids.s = RegisterComponent<S>(world);
        ids.sel = RegisterComponent<Sel>(world);
        ids.item = RegisterComponent<Item>(world);
        ids.ch = RegisterComponent<Ch>(world);
        ids.tog = RegisterComponent<Tog>(world);
        ids.res = RegisterComponent<Res>(world);
        ids.handle = RegisterComponent<Handle>(world);
        CHECK(ids.handle != ComponentId::Invalid && ids.res != ComponentId::Invalid);
        return ids;
    }

    nvrhi::AutoPtr<IWorld> MakeWorld(bool validation = false, uint32_t chunkSize = 16 * 1024)
    {
        WorldDesc desc;
        desc.debugName = "test_ecs_observers";
        desc.validation = validation;
        desc.chunkSize = chunkSize;
        desc.initialEntityCapacity = 4096;
        nvrhi::AutoPtr<IWorld> world;
        CHECK(CreateWorld(desc, &world) == nvrhi::FS_OK);
        return world;
    }

    nvrhi::AutoPtr<ICommandBuffer> MakeBuffer(IWorld* world)
    {
        nvrhi::AutoPtr<ICommandBuffer> buffer;
        CHECK(world->CreateCommandBuffer(&buffer) == nvrhi::FS_OK && buffer);
        return buffer;
    }

    ecs::WorldImpl& Impl(IWorld* world)
    {
        ecs::WorldImpl* impl = ecs::GetWorldImpl(world);
        CHECK(impl != nullptr);
        return *impl;
    }

    ecs::ObserverRegistry& Registry(IWorld* world)
    {
        ecs::ObserverRegistry* registry = Impl(world).GetObserverRegistry();
        CHECK(registry != nullptr);
        return *registry;
    }

    uint64_t Errors(IWorld* world) { return Impl(world).GetValidationErrorCount(); }

    template <typename T>
    const T& Get(IWorld* world, Entity entity)
    {
        const T* value = static_cast<const T*>(world->GetComponent(entity, FindComponent<T>(world)));
        CHECK(value != nullptr);
        return *value;
    }

    uint32_t ArchetypeOf(IWorld* world, Entity entity)
    {
        const ecs::EntityRecord* record = Impl(world).FindAlive(entity);
        CHECK(record != nullptr);
        return record->archetype;
    }

    // An EntityCreateDesc that owns its component and value arrays.
    class CreateDesc
    {
    public:
        CreateDesc(std::initializer_list<ComponentId> components, std::initializer_list<const void*> values,
                   uint32_t count)
        {
            CHECK(components.size() <= 16 && (values.size() == 0 || values.size() == components.size()));
            m_Components.assign(components.begin(), components.end());
            m_Values.assign(values.begin(), values.end());
            Finish(count);
        }
        CreateDesc(const std::vector<ComponentId>& components, const std::vector<const void*>& values, uint32_t count)
            : m_Components(components), m_Values(values)
        {
            Finish(count);
        }
        CreateDesc(const CreateDesc&) = delete;
        CreateDesc& operator=(const CreateDesc&) = delete;
        operator const EntityCreateDesc&() const { return m_Desc; }

    private:
        void Finish(uint32_t count)
        {
            m_Desc = EntityCreateDesc{};
            m_Desc.components = m_Components.data();
            m_Desc.componentCount = uint32_t(m_Components.size());
            m_Desc.values = m_Values.empty() ? nullptr : m_Values.data();
            m_Desc.entityCount = count;
            m_Desc.transfer = ValueTransfer::Copy;
        }
        std::vector<ComponentId> m_Components;
        std::vector<const void*> m_Values;
        EntityCreateDesc m_Desc{};
    };

    std::vector<Entity> Create(IWorld* world, const EntityCreateDesc& desc)
    {
        std::vector<Entity> entities(desc.entityCount);
        CHECK(world->CreateEntities(desc, entities.data()) == nvrhi::FS_OK);
        return entities;
    }

    // ----------------------------------------------------------------------------------------- the observer

    using ObserverFunction = std::function<nvrhi::FRESULT(ObserverEvent, const EntityBatch&, ICommandBuffer*)>;

    class TestObserver final : public nvrhi::ObjectImpl<IObserver>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(TestObserver)
        NVRHI_IMPLEMENTS_INTERFACE(IObserver)
        NVRHI_END_INTERFACE_TABLE()

        explicit TestObserver(ObserverFunction function, bool* destroyed = nullptr)
            : m_Function(std::move(function)), m_Destroyed(destroyed)
        {
            if (m_Destroyed)
                *m_Destroyed = false;
        }
        ~TestObserver()
        {
            if (m_Destroyed)
                *m_Destroyed = true;
        }
        nvrhi::FRESULT OnEntities(ObserverEvent event, const EntityBatch& batch,
                                  ICommandBuffer* commands) noexcept override
        {
            return m_Function(event, batch, commands);
        }

    private:
        ObserverFunction m_Function;
        bool* m_Destroyed;
    };

    ObserverDesc MakeDesc(ObserverEvent event, ComponentId component, Access access,
                          const std::vector<QueryTerm>& terms, const char* name = "test")
    {
        ObserverDesc desc{};
        desc.debugName = name;
        desc.filterTerms = terms.empty() ? nullptr : terms.data();
        desc.filterTermCount = uint32_t(terms.size());
        desc.component = component;
        desc.event = event;
        desc.access = access;
        return desc;
    }

    QueryTerm Term(ComponentId component, TermOp op, Access access = Access::Filter)
    {
        QueryTerm term{};
        term.component = component;
        term.op = op;
        term.access = access;
        return term;
    }

    ObserverId AddObserver(IWorld* world, const ObserverDesc& desc, ObserverFunction function,
                           bool* destroyed = nullptr)
    {
        auto observer = MAKE_RC_OBJ_PTR(TestObserver, std::move(function), destroyed);
        ObserverId id = ObserverId::Invalid;
        CHECK(world->AddObserver(desc, observer.Get(), id) == nvrhi::FS_OK);
        CHECK(id != ObserverId::Invalid);
        return id;
    }

    // What one callback saw.
    struct Seen
    {
        uint32_t tag;
        ObserverEvent event;
        uint32_t archetype;
        std::vector<Entity> entities;
        std::vector<uint64_t> values;
        std::vector<const void*> columns;
    };

    using Reader = std::function<uint64_t(const EntityBatch&, uint32_t)>;

    // A recording observer: every batch into `log`, with `reader`'s value per row and the first `columnCount` columns.
    ObserverId Record(IWorld* world, std::vector<Seen>& log, uint32_t tag, const ObserverDesc& desc, Reader reader,
                      uint32_t columnCount)
    {
        return AddObserver(world, desc, [&log, tag, reader, columnCount](ObserverEvent event, const EntityBatch& batch,
                                                                         ICommandBuffer*) {
            Seen seen{ tag, event, batch.archetypeIndex, { batch.entities, batch.entities + batch.count }, {}, {} };
            for (uint32_t i = 0; i < batch.count; ++i)
                seen.values.push_back(reader ? reader(batch, i) : 0);
            for (uint32_t c = 0; c < columnCount; ++c)
                seen.columns.push_back(batch.columns[c]);
            log.push_back(std::move(seen));
            return nvrhi::FS_OK;
        });
    }

    template <typename T>
    Reader ColumnValue(uint32_t column)
    {
        return [column](const EntityBatch& batch, uint32_t i) -> uint64_t {
            const T* values = static_cast<const T*>(batch.columns[column]);
            return values ? uint64_t(values[i].v) : 0xffffffffull;
        };
    }

    std::vector<Entity> Entities(std::initializer_list<Entity> list) { return std::vector<Entity>(list); }

    // An IStateWriter into memory.
    struct MemoryWriter final : IStateWriter
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

    // An IStateReader over memory.
    struct MemoryReader final : IStateReader
    {
        const uint8_t* data;
        uint64_t size;
        uint64_t position = 0;

        MemoryReader(const uint8_t* d, uint64_t s) : data(d), size(s) { }

        nvrhi::FRESULT QueryInterface(nvrhi::FREFIID, void** ppInterface) noexcept override
        {
            if (ppInterface)
                *ppInterface = nullptr;
            return nvrhi::FE_NOINTERFACE;
        }
        nvrhi::FLONG AddRef() noexcept override { return 1; }
        nvrhi::FLONG Release() noexcept override { return 1; }
        nvrhi::FRESULT Read(uint64_t count, void* out) noexcept override
        {
            if (count > size - position)
                return nvrhi::FE_INVALID_ARGS;
            std::memcpy(out, data + position, size_t(count));
            position += count;
            return nvrhi::FS_OK;
        }
    };
}

// ---------------------------------------------------------------------------------------------- registration

void test_registration()
{
    for (const bool validation : { false, true })
    {
        auto world = MakeWorld(validation);
        const Ids ids = Register(world);
        auto observer = MAKE_RC_OBJ_PTR(TestObserver,
            [](ObserverEvent, const EntityBatch&, ICommandBuffer*) { return nvrhi::FS_OK; });

        // Every refusal: FE_INVALID_ARGS with or without validation, an Invalid id, and with validation a report.
        auto refused = [&](const ObserverDesc& desc, IObserver* callback, uint32_t reports = 1)
        {
            const uint64_t errors = Errors(world);
            ObserverId id = ObserverId(5);
            CHECK(world->AddObserver(desc, callback, id) == nvrhi::FE_INVALID_ARGS);
            CHECK(id == ObserverId::Invalid);
            CHECK(Errors(world) - errors == (validation ? reports : 0));
        };
        const std::vector<QueryTerm> none;
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, none), nullptr);
        refused(MakeDesc(ObserverEvent(9), ids.a, Access::Read, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access(7), none), observer.Get());
        ObserverDesc dangling = MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, none);
        dangling.filterTermCount = 2;
        refused(dangling, observer.Get());
        const std::vector<QueryTerm> withA = { Term(ids.a, TermOp::All) };
        refused(MakeDesc(ObserverEvent::OnMatch, ids.a, Access::Read, withA), observer.Get());   // component set
        refused(MakeDesc(ObserverEvent::OnUnmatch, ComponentId::Invalid, Access::Read, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ComponentId(999), Access::Read, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ComponentId::Invalid, Access::Read, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnRemove, ids.res, Access::Read, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnSet, ids.item, Access::Read, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnSet, ids.ch, Access::Read, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.sh, Access::Write, none), observer.Get());
        // Filters: the query rules (reported by CreateQuery), then the observers' own.
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ComponentId(999), TermOp::All) }),
                observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ids.sh, TermOp::All, Access::Write) }),
                observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read,
                         { Term(ids.b, TermOp::All), Term(ids.b, TermOp::Without) }), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ids.s, TermOp::All) }), observer.Get());
        refused(MakeDesc(ObserverEvent::OnMatch, ComponentId::Invalid, Access::Read,
                         { Term(ids.a, TermOp::All), Term(ids.sel, TermOp::Without) }), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ids.b, TermOp::Changed) }), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ids.b, TermOp::Added) }), observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ids.tog, TermOp::Enabled) }),
                observer.Get());
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ids.tog, TermOp::Disabled) }),
                observer.Get());
        refused(MakeDesc(ObserverEvent::OnRemove, ids.a, Access::Read, { Term(ids.a, TermOp::Without) }),
                observer.Get());
        // Filter access hands out no column: refused for components with data (a zero-initialised desc has it).
        refused(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Filter, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnRemove, ids.s, Access::Filter, none), observer.Get());
        refused(MakeDesc(ObserverEvent::OnRemove, ids.item, Access::Filter, none), observer.Get());
        ObserverDesc zeroed{};
        zeroed.component = ids.b;
        refused(zeroed, observer.Get());
        CHECK(Registry(world).GetObserverCount() == 0);

        // Accepted: ids in registration order, never reused.
        auto accept = [&](const ObserverDesc& desc)
        {
            ObserverId id = ObserverId::Invalid;
            CHECK(world->AddObserver(desc, observer.Get(), id) == nvrhi::FS_OK);
            return id;
        };
        const uint64_t errors = Errors(world);
        const ObserverId first = accept(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, none));
        const ObserverId second = accept(MakeDesc(ObserverEvent::OnRemove, ids.sel, Access::Filter, none));   // tag
        const ObserverId third = accept(MakeDesc(ObserverEvent::OnMatch, ComponentId::Invalid, Access::Read,
            { Term(ids.a, TermOp::All, Access::Write), Term(ids.b, TermOp::Optional, Access::Read),
              Term(ids.c, TermOp::Any), Term(ids.sh, TermOp::Any, Access::Read), Term(ids.item, TermOp::Without) }));
        const ObserverId fourth = accept(MakeDesc(ObserverEvent::OnSet, ids.sel, Access::Read, none));
        const ObserverId fifth = accept(MakeDesc(ObserverEvent::OnAdd, ids.ch, Access::Write,
                                                  { Term(ids.a, TermOp::All, Access::Read) }));
        CHECK(Errors(world) == errors);
        CHECK(uint32_t(first) == 0 && uint32_t(second) == 1 && uint32_t(third) == 2 && uint32_t(fourth) == 3 &&
              uint32_t(fifth) == 4);
        CHECK(Registry(world).GetObserverCount() == 5);
        CHECK(world->RemoveObserver(second) == nvrhi::FS_OK);
        CHECK(Registry(world).GetObserverCount() == 4);
        const uint64_t before = Errors(world);
        CHECK(world->RemoveObserver(second) == nvrhi::FE_INVALID_ARGS);              // removed already
        CHECK(world->RemoveObserver(ObserverId(77)) == nvrhi::FE_INVALID_ARGS);      // never added
        CHECK(world->RemoveObserver(ObserverId::Invalid) == nvrhi::FE_INVALID_ARGS);
        CHECK(Errors(world) - before == (validation ? 3u : 0u));
        CHECK(uint32_t(accept(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, none))) == 5);

        // Phases: refused, unless an exclusive system runs.
        ecs::WorldImpl& impl = Impl(world);
        impl.BeginPhase();
        ObserverId id = ObserverId::Invalid;
        CHECK(world->AddObserver(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, none), observer.Get(), id) ==
              nvrhi::FE_INVALID_ARGS);
        CHECK(world->RemoveObserver(first) == nvrhi::FE_INVALID_ARGS);
        impl.BeginExclusive();
        CHECK(world->AddObserver(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, none), observer.Get(), id) ==
              nvrhi::FS_OK);
        CHECK(world->RemoveObserver(id) == nvrhi::FS_OK);
        impl.EndExclusive();
        impl.EndPhase();

        // A world without observers delivers nothing and has no registry.
        auto plain = MakeWorld(validation);
        CHECK(Impl(plain).GetObserverRegistry() == nullptr);
        CHECK(plain->RemoveObserver(ObserverId(0)) == nvrhi::FE_INVALID_ARGS);
    }

    // The world holds a reference to each observer until it is removed.
    auto world = MakeWorld();
    const Ids ids = Register(world);
    bool destroyed = false;
    const ObserverId id = AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}),
        [](ObserverEvent, const EntityBatch&, ICommandBuffer*) { return nvrhi::FS_OK; }, &destroyed);
    CHECK(!destroyed);
    CHECK(world->RemoveObserver(id) == nvrhi::FS_OK);
    CHECK(destroyed);
}

// -------------------------------------------------------------------------------------------- events

void test_events()
{
    for (const bool validation : { false, true })
    {
        Payload payload;   // outlives the world
        auto world = MakeWorld(validation);
        const Ids ids = Register(world);
        std::vector<Seen> log;
        int64_t refsSeen = -1;
        bool aliveSeen = false;
        bool sparseSeen = false;

        Record(world, log, 0, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}), ColumnValue<A>(0), 1);
        Record(world, log, 1, MakeDesc(ObserverEvent::OnRemove, ids.a, Access::Read,
                                       { Term(ids.b, TermOp::All, Access::Read) }),
               [](const EntityBatch& batch, uint32_t i) {
                   return (uint64_t(static_cast<const A*>(batch.columns[0])[i].v) << 32) |
                          static_cast<const B*>(batch.columns[1])[i].v;
               }, 2);
        Record(world, log, 2, MakeDesc(ObserverEvent::OnAdd, ids.c, Access::Read, {}), nullptr, 1);
        Record(world, log, 3, MakeDesc(ObserverEvent::OnAdd, ids.sh, Access::Read, {}),
               [](const EntityBatch& batch, uint32_t) {
                   return uint64_t(static_cast<const Sh*>(batch.columns[0])->key);
               }, 1);
        Record(world, log, 4, MakeDesc(ObserverEvent::OnAdd, ids.ch, Access::Read, {}), nullptr, 1);
        Record(world, log, 5, MakeDesc(ObserverEvent::OnRemove, ids.item, Access::Read, {}),
               [](const EntityBatch& batch, uint32_t i) {
                   const BufferHeader& header = static_cast<const BufferHeader*>(batch.columns[0])[i];
                   uint64_t sum = 0;
                   for (uint32_t k = 0; k < header.count; ++k)
                       sum += GetBufferData<Item>(header)[k].v;
                   return (uint64_t(header.count) << 32) | sum;
               }, 1);
        Record(world, log, 6, MakeDesc(ObserverEvent::OnAdd, ids.s, Access::Read, {}), ColumnValue<S>(0), 1);
        Record(world, log, 7, MakeDesc(ObserverEvent::OnRemove, ids.s, Access::Read, {}), ColumnValue<S>(0), 1);
        const std::vector<QueryTerm> filter = { Term(ids.a, TermOp::All, Access::Read), Term(ids.c, TermOp::Without) };
        Record(world, log, 8, MakeDesc(ObserverEvent::OnMatch, ComponentId::Invalid, Access::Read, filter),
               ColumnValue<A>(0), 1);
        Record(world, log, 9, MakeDesc(ObserverEvent::OnUnmatch, ComponentId::Invalid, Access::Read, filter),
               ColumnValue<A>(0), 1);
        IWorld* raw = world.Get();
        AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.handle, Access::Read, {}),
            [&, raw](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                const Payload* seen = static_cast<const Handle*>(batch.columns[0])[0].payload;
                refsSeen = seen ? seen->refs : -1;
                aliveSeen = raw->IsAlive(batch.entities[0]);
                return nvrhi::FS_OK;
            });
        AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.sel, Access::Read, {}),
            [&, raw](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                // Removals are delivered before the change: the entity still has the rest of its components.
                sparseSeen = batch.columns[0] == nullptr && raw->HasComponent(batch.entities[0], ids.sel) &&
                             raw->IsAlive(batch.entities[0]) && raw->HasComponent(batch.entities[0], ids.a);
                return nvrhi::FS_OK;
            });

        // Creation: OnAdd for every component, OnMatch for the filter, one batch per run.
        const A as[3] = { { 1 }, { 2 }, { 3 } };
        const B bs[3] = { { 10 }, { 20 }, { 30 } };
        const std::vector<Entity> e = Create(world, CreateDesc({ ids.a, ids.b }, { as, bs }, 3));
        CHECK(log.size() == 2);
        CHECK(log[0].tag == 0 && log[0].event == ObserverEvent::OnAdd && log[0].entities == e);
        CHECK(log[0].values == std::vector<uint64_t>({ 1, 2, 3 }));
        CHECK(log[0].archetype == ArchetypeOf(world, e[0]));
        CHECK(log[0].columns[0] == world->GetComponent(e[0], ids.a));
        CHECK(log[1].tag == 8 && log[1].event == ObserverEvent::OnMatch && log[1].entities == e);
        CHECK(log[1].values == std::vector<uint64_t>({ 1, 2, 3 }));
        log.clear();

        // A move adds and removes exactly what the archetypes do not share.
        CHECK(world->AddComponent(e[1], ids.c, nullptr) == nvrhi::FS_OK);
        CHECK(log.size() == 2);
        CHECK(log[0].tag == 9 && log[0].event == ObserverEvent::OnUnmatch && log[0].entities == Entities({ e[1] }));
        CHECK(log[0].values == std::vector<uint64_t>({ 2 }));
        CHECK(log[1].tag == 2 && log[1].entities == Entities({ e[1] }) && log[1].columns[0] == nullptr);   // a tag
        CHECK(log[1].archetype == ArchetypeOf(world, e[1]));
        log.clear();
        CHECK(world->RemoveComponent(e[0], ids.a) == nvrhi::FS_OK);
        CHECK(log.size() == 2);
        CHECK(log[0].tag == 1 && log[0].values == std::vector<uint64_t>({ (1ull << 32) | 10 }));
        CHECK(log[1].tag == 9 && log[1].values == std::vector<uint64_t>({ 1 }));
        log.clear();
        CHECK(world->RemoveComponent(e[0], ids.a) == nvrhi::FS_OK);   // absent: nothing
        CHECK(world->AddComponent(e[2], ids.a, &as[0]) == nvrhi::FS_OK);   // present: overwritten, no event
        CHECK(log.empty());

        // Shared: added with its value; a value change moves the entity, but adds and removes nothing.
        const Sh key7{ 7 };
        const Sh key8{ 8 };
        CHECK(world->AddComponent(e[2], ids.sh, &key7) == nvrhi::FS_OK);
        CHECK(log.size() == 1 && log[0].tag == 3 && log[0].values == std::vector<uint64_t>({ 7 }));
        CHECK(log[0].columns[0] == world->GetComponent(e[2], ids.sh));
        log.clear();
        CHECK(world->AddComponent(e[2], ids.sh, &key8) == nvrhi::FS_OK);
        CHECK(log.empty());

        // Sparse: OnAdd with the dense value; overwriting is no addition.
        const S s5{ 5 };
        CHECK(world->AddComponent(e[2], ids.s, &s5) == nvrhi::FS_OK);
        CHECK(log.size() == 1 && log[0].tag == 6 && log[0].values == std::vector<uint64_t>({ 5 }));
        CHECK(log[0].columns[0] == world->GetComponent(e[2], ids.s));
        CHECK(log[0].archetype == ArchetypeOf(world, e[2]));
        log.clear();
        CHECK(world->AddComponent(e[2], ids.s, &s5) == nvrhi::FS_OK);
        CHECK(world->AddComponent(e[2], ids.sel, nullptr) == nvrhi::FS_OK);
        CHECK(log.empty());

        // Destruction: the rows' events, then the Sparse ones, each before the values go.
        CHECK(world->DestroyEntities(&e[2], 1) == nvrhi::FS_OK);
        CHECK(log.size() == 3);
        CHECK(log[0].tag == 1 && log[0].values == std::vector<uint64_t>({ (1ull << 32) | 30 }));
        CHECK(log[1].tag == 9 && log[1].values == std::vector<uint64_t>({ 1 }));
        CHECK(log[2].tag == 7 && log[2].values == std::vector<uint64_t>({ 5 }));
        CHECK(sparseSeen);
        log.clear();

        // Chunk components: the chunk's value. Buffers and non-trivial values: alive while OnRemove reads them.
        const std::vector<Entity> f = Create(world, CreateDesc({ ids.a, ids.ch, ids.item }, {}, 1));
        CHECK(log.size() == 3);
        CHECK(log[0].tag == 0 && log[1].tag == 4 && log[2].tag == 8);
        CHECK(log[1].columns[0] == world->GetComponent(f[0], ids.ch));
        log.clear();
        CHECK(world->ResizeBuffer(f[0], ids.item, 5, c_WorldVersion) == nvrhi::FS_OK);   // on the heap
        const BufferHeader* header =
            static_cast<const BufferHeader*>(world->GetComponentForWrite(f[0], ids.item, c_WorldVersion));
        for (uint32_t k = 0; k < 5; ++k)
            GetBufferData<Item>(*header)[k].v = k + 1;
        {
            Handle handle(&payload);
            CHECK(world->AddComponent(f[0], ids.handle, &handle) == nvrhi::FS_OK);   // a copy
        }
        CHECK(log.empty());
        CHECK(payload.refs == 1);
        log.clear();
        CHECK(world->DestroyEntities(f.data(), 1) == nvrhi::FS_OK);
        CHECK(refsSeen == 1 && aliveSeen);
        CHECK(payload.refs == 0);
        bool sawItems = false;
        for (const Seen& seen : log)
            if (seen.tag == 5)
                sawItems = seen.values == std::vector<uint64_t>({ (5ull << 32) | 15 });
        CHECK(sawItems);
        log.clear();
        CHECK(Errors(world) == 0);
    }
}

// Write access stamps the column it hands out, Read access does not.
void test_write_access()
{
    auto world = MakeWorld();
    const Ids ids = Register(world);
    // e[1] is its chunk's last row, so its removal stamps nothing in the source chunk itself; e[0] keeps the chunk.
    const std::vector<Entity> e = Create(world, CreateDesc({ ids.a, ids.b, ids.c }, {}, 2));
    AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.c, Access::Read,
                                { Term(ids.a, TermOp::All, Access::Write), Term(ids.b, TermOp::All, Access::Read) }),
        [](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
            static_cast<A*>(batch.columns[1])[0].v = 99;
            return nvrhi::FS_OK;
        });
    const uint32_t archetype = ArchetypeOf(world, e[0]);
    const ecs::Archetype& storage = Impl(world).GetArchetypeStorage(archetype);
    const uint32_t columnA = storage.FindColumn(ids.a);
    const uint32_t columnB = storage.FindColumn(ids.b);
    world->AdvanceVersion();
    world->AdvanceVersion();
    const uint64_t version = world->GetVersion();
    CHECK(storage.Versions(0)[columnA] < version && storage.Versions(0)[columnB] < version);
    CHECK(world->RemoveComponent(e[1], ids.c) == nvrhi::FS_OK);
    CHECK(storage.GetChunkCount() == 1 && storage.chunks[0].count == 1);
    CHECK(storage.Versions(0)[columnA] == version);
    CHECK(storage.Versions(0)[columnB] < version);
    CHECK(Get<A>(world, e[1]).v == 99);   // moved with the row
}

// ------------------------------------------------------------------------------------------------ batching

void test_batching()
{
    auto world = MakeWorld();
    const Ids ids = Register(world);
    std::vector<Seen> log;
    Record(world, log, 0, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}), ColumnValue<A>(0), 1);
    Record(world, log, 1, MakeDesc(ObserverEvent::OnAdd, ids.c, Access::Read, {}), nullptr, 0);
    Record(world, log, 2, MakeDesc(ObserverEvent::OnSet, ids.a, Access::Read, {}), ColumnValue<A>(0), 1);
    Record(world, log, 3, MakeDesc(ObserverEvent::OnAdd, ids.s, Access::Read, {}), ColumnValue<S>(0), 1);
    Record(world, log, 4, MakeDesc(ObserverEvent::OnAdd, ids.item, Access::Read, {}),
           [](const EntityBatch& batch, uint32_t i) {
               return uint64_t(static_cast<const BufferHeader*>(batch.columns[0])[i].count);
           }, 1);
    IWorld* raw = world.Get();
    // A batch is a run of rows of one chunk, in order.
    auto checkRuns = [&]()
    {
        for (const Seen& seen : log)
        {
            const ecs::EntityRecord* first = Impl(raw).FindAlive(seen.entities[0]);
            CHECK(first && first->archetype == seen.archetype);
            for (uint32_t i = 0; i < seen.entities.size(); ++i)
            {
                const ecs::EntityRecord* record = Impl(raw).FindAlive(seen.entities[i]);
                CHECK(record && record->archetype == first->archetype && record->chunk == first->chunk &&
                      record->row == first->row + i);
            }
        }
    };
    auto chunksOf = [&](const std::vector<Entity>& entities)
    {
        std::set<uint64_t> chunks;
        for (const Entity entity : entities)
        {
            const ecs::EntityRecord* record = Impl(raw).FindAlive(entity);
            chunks.insert((uint64_t(record->archetype) << 32) | record->chunk);
        }
        return uint32_t(chunks.size());
    };
    auto flatten = [&](uint32_t tag)
    {
        std::vector<Entity> entities;
        for (const Seen& seen : log)
            if (seen.tag == tag)
                entities.insert(entities.end(), seen.entities.begin(), seen.entities.end());
        return entities;
    };
    auto batches = [&](uint32_t tag)
    {
        return uint32_t(std::count_if(log.begin(), log.end(), [tag](const Seen& seen) { return seen.tag == tag; }));
    };

    // One CreateEntities: one batch per chunk of the run (12-byte rows: about 1300 per 16 KiB chunk).
    const std::vector<Entity> base = Create(world, CreateDesc({ ids.a }, {}, 3000));
    CHECK(flatten(0) == base && batches(0) == chunksOf(base));
    CHECK(batches(0) > 1);
    checkRuns();
    log.clear();

    // Two Create commands of one playback extend one batch across the calls.
    auto buffer = MakeBuffer(world);
    std::vector<Entity> placeholders(200);
    buffer->CreateEntities(CreateDesc({ ids.a }, {}, 100), 0, placeholders.data());
    buffer->CreateEntities(CreateDesc({ ids.a }, {}, 100), 1, placeholders.data() + 100);
    ICommandBuffer* list[1] = { buffer.Get() };
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    std::vector<Entity> created;
    for (const Entity placeholder : placeholders)
        created.push_back(world->ResolvePlaceholder(placeholder));
    CHECK(flatten(0) == created && batches(0) == chunksOf(created));
    checkRuns();
    log.clear();

    // Coalesced moves: one OnAdd batch per target chunk.
    std::vector<Entity> all = base;
    all.insert(all.end(), created.begin(), created.end());
    buffer->AddComponents(all.data(), 1600, ids.c, nullptr, 0);
    buffer->AddComponents(all.data() + 1600, uint32_t(all.size()) - 1600, ids.c, nullptr, 1);
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    CHECK(flatten(1) == all && batches(1) == chunksOf(all));
    CHECK(batches(1) > 1);
    checkRuns();
    log.clear();

    // Sets: one batch per chunk; a set of the same entity twice in a row shows the newer value once.
    std::vector<A> values(50);
    for (uint32_t i = 0; i < 50; ++i)
        values[i].v = 1000 + i;
    std::vector<Entity> fifty(all.begin(), all.begin() + 50);
    buffer->SetComponents(fifty.data(), 50, ids.a, values.data(), 0);
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    CHECK(flatten(2) == fifty && batches(2) == chunksOf(fifty));
    CHECK(log[0].values[0] == 1000 && log.back().values.back() == 1049);
    log.clear();
    const Entity twice[2] = { all[0], all[0] };
    const A twiceValues[2] = { { 1 }, { 2 } };
    buffer->SetComponents(twice, 2, ids.a, twiceValues, 0);
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    // Two sets, two callbacks; the first already shows the newer value (a set writes before it reports).
    CHECK(log.size() == 2 && log[0].entities == Entities({ all[0] }) && log[1].entities == Entities({ all[0] }));
    CHECK(log[0].values == std::vector<uint64_t>({ 2 }) && log[1].values == std::vector<uint64_t>({ 2 }));
    log.clear();
    // The same entity set again after another entity: three sets, as they happened.
    const Entity apart[3] = { all[0], all[700], all[0] };
    const A apartValues[3] = { { 3 }, { 4 }, { 5 } };
    buffer->SetComponents(apart, 3, ids.a, apartValues, 0);
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    CHECK(log.size() == 3 && log[0].values[0] == 3 && log[1].values[0] == 4 && log[2].values[0] == 5);
    log.clear();
    // The same with adjacent rows, in two commands that playback coalesces into one run: e0, e1 extend one batch, the
    // second e0 is no continuation, so it gets a callback of its own after them (three events, in order).
    const Entity adjacent[2] = { all[0], all[1] };
    const A adjacentValues[2] = { { 3 }, { 4 } };
    const A again{ 5 };
    buffer->SetComponents(adjacent, 2, ids.a, adjacentValues, 0);
    buffer->SetComponent(all[0], ids.a, &again, 1);
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    CHECK(log.size() == 2 && log[0].entities == Entities({ all[0], all[1] }) &&
          log[1].entities == Entities({ all[0] }));
    CHECK(log[0].values == std::vector<uint64_t>({ 5, 4 }) && log[1].values == std::vector<uint64_t>({ 5 }));
    log.clear();

    // Sparse adds: consecutive rows with consecutive dense values form one batch.
    std::vector<S> sparse(10);
    for (uint32_t i = 0; i < 10; ++i)
        sparse[i].v = 50 + i;
    buffer->AddComponents(all.data(), 10, ids.s, sparse.data(), 0);
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    CHECK(flatten(3) == std::vector<Entity>(all.begin(), all.begin() + 10));
    CHECK(batches(3) == chunksOf(std::vector<Entity>(all.begin(), all.begin() + 10)));
    CHECK(log[0].values[0] == 50 && log[0].values[1] == 51);
    checkRuns();
    log.clear();
    // A held batch of Sparse values is delivered before a removal of that component that nobody observes moves the
    // set's last value into the hole.
    for (uint32_t i = 0; i < 5; ++i)
        sparse[i].v = 70 + i;
    buffer->AddComponents(all.data() + 10, 5, ids.s, sparse.data(), 0);   // dense 10..14
    buffer->RemoveComponent(all[11], ids.s, 1);                           // dense 14 moves to 11
    CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
    CHECK(log.size() == 1 && log[0].values == std::vector<uint64_t>({ 70, 71, 72, 73, 74 }));
    CHECK(Get<S>(world, all[14]).v == 74 && !world->HasComponent(all[11], ids.s));
    log.clear();

    // A held batch is delivered before a batch operation writes its values without a report: the OnAdd sees the
    // value of the creation, not the later overwrite; the buffer before the append.
    {
        Entity p[2];
        const A one{ 1 };
        const A five{ 5 };
        buffer->CreateEntities(CreateDesc({ ids.a, ids.item }, { &one, nullptr }, 1), 0, &p[0]);
        buffer->AddComponent(p[0], ids.a, &five, 1);
        const Item items[3] = { { 1 }, { 2 }, { 3 } };
        buffer->AppendToBuffer(p[0], ids.item, items, 3, 2);
        CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
        CHECK(log.size() == 2);
        CHECK(log[0].tag == 0 && log[0].values == std::vector<uint64_t>({ 1 }));
        CHECK(log[1].tag == 4 && log[1].values == std::vector<uint64_t>({ 0 }));
        const Entity entity = world->ResolvePlaceholder(p[0]);
        CHECK(Get<A>(world, entity).v == 5);
        log.clear();
    }

    // A held batch is delivered before a removal moves rows of its archetype, even one that nobody observes.
    {
        std::vector<Entity> p(5);
        bool allAlive = true;
        std::vector<Entity> seen;
        const ObserverId check = AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.b, Access::Read, {}),
            [&, raw](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                for (uint32_t i = 0; i < batch.count; ++i)
                    allAlive = allAlive && raw->IsAlive(batch.entities[i]);
                seen.insert(seen.end(), batch.entities, batch.entities + batch.count);
                return nvrhi::FS_OK;
            });
        buffer->CreateEntities(CreateDesc({ ids.b }, {}, 5), 0, p.data());
        buffer->DestroyEntity(p[0], 1);   // the swap-remove would move p[4] into p[0]'s row
        CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
        const ecs::PlaceholderTable& table = Impl(world).GetCommandPlayback().placeholders;
        std::vector<Entity> created5;
        for (const Entity placeholder : p)
            created5.push_back(table.Resolve(placeholder));
        CHECK(allAlive && seen == created5);
        CHECK(world->RemoveObserver(check) == nvrhi::FS_OK);
        log.clear();
    }

    // With a removal that someone takes between them, moves deliver entity by entity, in order.
    {
        std::vector<Seen> moves;
        auto other = MakeWorld();
        const Ids o = Register(other);
        const std::vector<Entity> e = Create(other, CreateDesc({ o.a, o.c }, {}, 6));
        const std::vector<QueryTerm> filter = { Term(o.a, TermOp::All), Term(o.c, TermOp::Without) };
        Record(other, moves, 0, MakeDesc(ObserverEvent::OnRemove, o.c, Access::Read, {}), nullptr, 0);
        Record(other, moves, 1, MakeDesc(ObserverEvent::OnMatch, ComponentId::Invalid, Access::Read, filter), nullptr,
               0);
        auto otherBuffer = MakeBuffer(other);
        otherBuffer->RemoveComponents(e.data(), 6, o.c, 0);
        ICommandBuffer* otherList[1] = { otherBuffer.Get() };
        CHECK(other->Playback(otherList, 1) == nvrhi::FS_OK);
        CHECK(moves.size() == 12);
        for (uint32_t i = 0; i < 6; ++i)
        {
            CHECK(moves[2 * i].tag == 0 && moves[2 * i].entities == Entities({ e[i] }));
            CHECK(moves[2 * i + 1].tag == 1 && moves[2 * i + 1].entities == Entities({ e[i] }));
        }
        moves.clear();
        // Without a taker for the removals the additions extend one batch.
        CHECK(other->RemoveObserver(ObserverId(0)) == nvrhi::FS_OK);
        for (const Entity entity : e)
            CHECK(other->AddComponent(entity, o.c, nullptr) == nvrhi::FS_OK);
        otherBuffer->RemoveComponents(e.data(), 6, o.c, 0);
        CHECK(other->Playback(otherList, 1) == nvrhi::FS_OK);
        CHECK(moves.size() == 1 && moves[0].entities == e);
        CHECK(Registry(other).GetCounters().extended >= 5);
    }
}

// -------------------------------------------------------------------------------------------------- model

namespace
{
    // The model test's components as bits.
    constexpr uint32_t kA = 1, kB = 2, kC = 4, kSh = 8, kS = 16, kItem = 32;
    constexpr uint32_t kArchetypeBits = kA | kB | kC | kSh | kItem;
    constexpr uint32_t c_ModelObservers = 15;

    struct ModelEntity
    {
        uint32_t mask = 0;
        uint32_t a = 0, b = 0, s = 0, sh = 0;
        std::vector<uint32_t> items;
    };

    struct Expected
    {
        uint64_t entity;   // an Entity, or a placeholder until its playback
        uint64_t value;
        uint64_t sequence;
        uint32_t reads;                 // the component bit `value` is of (A or S), for the allowance below
        std::vector<uint64_t> later;    // values written to it later in the same coalesced run: also accepted
    };

    uint64_t ItemsValue(const std::vector<uint32_t>& items)
    {
        uint64_t sum = 0;
        for (const uint32_t item : items)
            sum += item;
        return (uint64_t(items.size()) << 32) | (sum & 0xffffffffull);
    }

    // The callbacks of the model's observers, predicted from the storage's report order (EcsObserverImpl.h): per
    // report, the observers in registration order; a creation reports its rows, then its Sparse components; a move
    // reports the removal, then the addition; a destruction its rows, then its Sparse components.
    //
    // A playback may name an entity twice in one coalesced run (one batch operation). Its Add or Set then writes A or S
    // values before (or without) a report, so an earlier event of that entity in the run may already show a later value
    // of the run (EcsObserverImpl.h): such an event accepts its own value or any value written to that component of the
    // entity later in the run.
    struct Model
    {
        struct Write
        {
            uint64_t entity;
            uint32_t bit;
            uint64_t value;
            uint64_t sequence;
        };

        std::map<uint64_t, ModelEntity> entities;
        std::vector<Expected> expected[c_ModelObservers];
        uint64_t sequence = 0;
        bool inRun = false;
        size_t runStart[c_ModelObservers] = {};
        std::vector<Write> runWrites;

        void Emit(uint32_t observer, uint64_t entity, uint64_t value, uint32_t reads = 0)
        {
            expected[observer].push_back(Expected{ entity, value, sequence++, reads, {} });
        }
        void Written(uint64_t entity, uint32_t bit, uint64_t value)
        {
            if (inRun && (bit == kA || bit == kS))
                runWrites.push_back(Write{ entity, bit, value, sequence++ });
        }
        void BeginRun()
        {
            EndRun();
            inRun = true;
            for (uint32_t o = 0; o < c_ModelObservers; ++o)
                runStart[o] = expected[o].size();
        }
        void EndRun()
        {
            if (!inRun)
                return;
            for (uint32_t o = 0; o < c_ModelObservers; ++o)
                for (size_t i = runStart[o]; i < expected[o].size(); ++i)
                {
                    Expected& e = expected[o][i];
                    for (const Write& w : runWrites)
                        if (e.reads != 0 && w.entity == e.entity && w.bit == e.reads && w.sequence > e.sequence)
                            e.later.push_back(w.value);
                }
            runWrites.clear();
            inRun = false;
        }
        static bool Filter(uint32_t mask) { return (mask & kA) && !(mask & kC); }   // observers 6 and 7

        void RowsAdded(uint64_t id, uint32_t from, bool created)
        {
            ModelEntity& m = entities[id];
            const uint32_t to = m.mask & kArchetypeBits;
            auto added = [&](uint32_t bit) { return (to & bit) && (created || !(from & bit)); };
            if (added(kA))
                Emit(0, id, m.a, kA);
            if (Filter(to) && (created || !Filter(from)))
                Emit(6, id, m.a, kA);
            if (added(kC))
                Emit(8, id, (to & kB) ? m.b : 0xffffffffull);
            if (added(kSh))
                Emit(10, id, m.sh);
            if (added(kB))
            {
                Emit(14, id, m.b);
                m.b += 1000;   // observer 14 writes
            }
        }
        void RowsRemoving(uint64_t id, uint32_t to, bool destroyed)
        {
            ModelEntity& m = entities[id];
            const uint32_t from = m.mask & kArchetypeBits;
            auto removed = [&](uint32_t bit) { return (from & bit) && (destroyed || !(to & bit)); };
            if (removed(kA) && (from & kB))
                Emit(1, id, (uint64_t(m.a) << 32) | m.b);
            if (Filter(from) && (destroyed || !Filter(to)))
                Emit(7, id, m.a);
            if (removed(kItem))
                Emit(9, id, ItemsValue(m.items));
            if (removed(kSh))
                Emit(11, id, m.sh);
            if (removed(kC) && (from & (kA | kB)))
                Emit(13, id, 0);
        }

        // A creation of entities that are already in `entities` with their values.
        void Created(const std::vector<uint64_t>& created)
        {
            for (const uint64_t id : created)
                RowsAdded(id, 0, true);
            for (const uint64_t id : created)
                if (entities[id].mask & kS)
                    Emit(3, id, entities[id].s, kS);
        }
        void Destroy(uint64_t id)
        {
            RowsRemoving(id, 0, true);
            if (entities[id].mask & kS)
                Emit(4, id, entities[id].s);
            entities.erase(id);
        }
        // AddComponent: `bit` with `value` (ignored for C and Item).
        void Add(uint64_t id, uint32_t bit, uint32_t value)
        {
            ModelEntity& m = entities[id];
            if (bit == kS)
            {
                const bool present = (m.mask & kS) != 0;
                m.mask |= kS;
                m.s = value;
                if (!present)
                    Emit(3, id, value, kS);
                else
                    Written(id, kS, value);
                return;
            }
            if (m.mask & bit)
            {
                // Present: a Data value is overwritten, a Shared value moves the entity; neither is reported.
                Written(id, bit, value);
                if (bit == kA) m.a = value;
                if (bit == kB) m.b = value;
                if (bit == kSh) m.sh = value;
                return;
            }
            const uint32_t from = m.mask & kArchetypeBits;
            RowsRemoving(id, from | bit, false);
            m.mask |= bit;
            if (bit == kA) m.a = value;
            if (bit == kB) m.b = value;
            if (bit == kSh) m.sh = value;
            if (bit == kItem) m.items.clear();
            RowsAdded(id, from, false);
        }
        void Remove(uint64_t id, uint32_t bit)
        {
            ModelEntity& m = entities[id];
            if (!(m.mask & bit))
                return;
            if (bit == kS)
            {
                Emit(4, id, m.s);
                m.mask &= ~kS;
                return;
            }
            const uint32_t from = m.mask & kArchetypeBits;
            RowsRemoving(id, from & ~bit, false);
            m.mask &= ~bit;
            if (bit == kItem) m.items.clear();
            RowsAdded(id, from, false);
        }
        void Set(uint64_t id, uint32_t bit, uint32_t value)
        {
            ModelEntity& m = entities[id];
            Written(id, bit, value);
            if (bit == kA)
            {
                m.a = value;
                Emit(2, id, value, kA);
            }
            else if (bit == kS)
            {
                m.s = value;
                Emit(5, id, value, kS);
            }
            else
            {
                m.sh = value;
                Emit(12, id, value);
            }
        }
        // Renames a placeholder after its playback.
        void Rename(uint64_t from, uint64_t to)
        {
            auto found = entities.find(from);
            if (found != entities.end())
            {
                ModelEntity m = found->second;
                entities.erase(found);
                entities[to] = m;
            }
            for (std::vector<Expected>& list : expected)
                for (Expected& e : list)
                    if (e.entity == from)
                        e.entity = to;
        }
    };

    struct Actual
    {
        uint64_t entity;
        uint64_t value;
        uint64_t callback;   // global callback index
    };
}

void test_model()
{
    uint64_t totalCallbacks = 0;
    uint64_t totalEvents = 0;
    uint64_t totalSteps = 0;
    uint64_t repeatedShown = 0;   // events of a repeated entity that showed a later value of their run
    uint64_t extended = 0;        // reports that extended a held batch
    for (uint64_t seed = 1; seed <= 8; ++seed)
    {
        const bool validation = (seed & 1) != 0;
        const uint32_t chunkSize = (seed & 2) != 0 ? 1024 : 16 * 1024;
        auto world = MakeWorld(validation, chunkSize);
        const Ids ids = Register(world);
        IWorld* raw = world.Get();
        Xorshift rng{ 0x9e3779b97f4a7c15ull * seed + 17 };
        Model model;
        std::vector<Actual> actual[c_ModelObservers];
        uint64_t callbacks = 0;
        bool wellFormed = true;

        // Observer i records (entity, value) per row and checks that its batch is one run of rows of one chunk whose
        // observed column lines up with the entities.
        auto observe = [&](uint32_t index, ObserverEvent event, ComponentId component, Access access,
                           std::vector<QueryTerm> terms, Reader reader, size_t valueSize)
        {
            AddObserver(world, MakeDesc(event, component, access, terms),
                [&, index, component, reader, valueSize, raw](ObserverEvent, const EntityBatch& batch,
                                                             ICommandBuffer*) {
                    const ecs::EntityRecord* first = Impl(raw).FindAlive(batch.entities[0]);
                    wellFormed = wellFormed && first && first->archetype == batch.archetypeIndex && batch.count != 0;
                    for (uint32_t i = 0; i < batch.count; ++i)
                    {
                        const ecs::EntityRecord* record = Impl(raw).FindAlive(batch.entities[i]);
                        wellFormed = wellFormed && record && first && record->chunk == first->chunk &&
                                     record->row == first->row + i && record->archetype == first->archetype;
                        if (valueSize != 0 && batch.columns[0])
                            wellFormed = wellFormed && static_cast<const uint8_t*>(batch.columns[0]) + i * valueSize ==
                                                       raw->GetComponent(batch.entities[i], component);
                        actual[index].push_back(Actual{ uint64_t(batch.entities[i]), reader(batch, i), callbacks });
                    }
                    ++callbacks;
                    return nvrhi::FS_OK;
                });
        };
        auto zero = [](const EntityBatch&, uint32_t) -> uint64_t { return 0; };
        observe(0, ObserverEvent::OnAdd, ids.a, Access::Read, {}, ColumnValue<A>(0), sizeof(A));
        observe(1, ObserverEvent::OnRemove, ids.a, Access::Read, { Term(ids.b, TermOp::All, Access::Read) },
            [](const EntityBatch& batch, uint32_t i) {
                return (uint64_t(static_cast<const A*>(batch.columns[0])[i].v) << 32) |
                       static_cast<const B*>(batch.columns[1])[i].v;
            }, sizeof(A));
        observe(2, ObserverEvent::OnSet, ids.a, Access::Read, {}, ColumnValue<A>(0), sizeof(A));
        observe(3, ObserverEvent::OnAdd, ids.s, Access::Read, {}, ColumnValue<S>(0), sizeof(S));
        observe(4, ObserverEvent::OnRemove, ids.s, Access::Read, {}, ColumnValue<S>(0), sizeof(S));
        observe(5, ObserverEvent::OnSet, ids.s, Access::Read, {}, ColumnValue<S>(0), sizeof(S));
        const std::vector<QueryTerm> filter = { Term(ids.a, TermOp::All, Access::Read), Term(ids.c, TermOp::Without) };
        observe(6, ObserverEvent::OnMatch, ComponentId::Invalid, Access::Read, filter, ColumnValue<A>(0), 0);
        observe(7, ObserverEvent::OnUnmatch, ComponentId::Invalid, Access::Read, filter, ColumnValue<A>(0), 0);
        observe(8, ObserverEvent::OnAdd, ids.c, Access::Read, { Term(ids.b, TermOp::Optional, Access::Read) },
                ColumnValue<B>(1), 0);
        observe(9, ObserverEvent::OnRemove, ids.item, Access::Read, {},
            [](const EntityBatch& batch, uint32_t i) {
                const BufferHeader& header = static_cast<const BufferHeader*>(batch.columns[0])[i];
                std::vector<uint32_t> items;
                for (uint32_t k = 0; k < header.count; ++k)
                    items.push_back(GetBufferData<Item>(header)[k].v);
                return ItemsValue(items);
            }, sizeof(BufferHeader));
        auto key = [](const EntityBatch& batch, uint32_t) {
            return uint64_t(static_cast<const Sh*>(batch.columns[0])->key);
        };
        observe(10, ObserverEvent::OnAdd, ids.sh, Access::Read, {}, key, 0);
        observe(11, ObserverEvent::OnRemove, ids.sh, Access::Read, {}, key, 0);
        observe(12, ObserverEvent::OnSet, ids.sh, Access::Read, {}, key, 0);
        observe(13, ObserverEvent::OnRemove, ids.c, Access::Read,
                { Term(ids.a, TermOp::Any), Term(ids.b, TermOp::Any) }, zero, 0);
        observe(14, ObserverEvent::OnAdd, ids.b, Access::Write, {},
            [](const EntityBatch& batch, uint32_t i) {
                B& b = static_cast<B*>(batch.columns[0])[i];
                const uint64_t seen = b.v;
                b.v += 1000;
                return seen;
            }, sizeof(B));

        const ComponentId componentOf[6] = { ids.a, ids.b, ids.c, ids.sh, ids.s, ids.item };
        const uint32_t bitOf[6] = { kA, kB, kC, kSh, kS, kItem };
        auto alive = [&]() {
            std::vector<uint64_t> list;
            for (const auto& entry : model.entities)
                list.push_back(entry.first);
            return list;
        };
        auto pick = [&](const std::vector<uint64_t>& list) { return list[rng.Below(uint32_t(list.size()))]; };
        auto value = [&]() { return rng.Below(1000); };

        // A creation's components and values; the model entity it describes.
        struct Creation
        {
            std::vector<ComponentId> components;
            std::vector<const void*> values;
            std::vector<A> a;
            std::vector<B> b;
            std::vector<Sh> sh;
            std::vector<S> s;
            uint32_t mask = 0;
        };
        auto makeCreation = [&](uint32_t count, Creation& c)
        {
            c.mask = rng.Below(64);
            c.a.resize(count);
            c.b.resize(count);
            c.sh.resize(count);
            c.s.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                c.a[i].v = value();
                c.b[i].v = value();
                c.sh[i].key = 1 + rng.Below(3);
                c.s[i].v = value();
            }
            const uint32_t key = 1 + rng.Below(3);
            for (uint32_t i = 0; i < count; ++i)
                c.sh[i].key = key;   // one Shared value per creation
            for (uint32_t k = 0; k < 6; ++k)
            {
                if (!(c.mask & bitOf[k]))
                    continue;
                c.components.push_back(componentOf[k]);
                const void* values = nullptr;
                if (bitOf[k] == kA) values = c.a.data();
                if (bitOf[k] == kB) values = c.b.data();
                if (bitOf[k] == kSh) values = c.sh.data();
                if (bitOf[k] == kS) values = c.s.data();
                c.values.push_back(values);
            }
        };
        auto modelCreation = [&](const Creation& c, const std::vector<uint64_t>& created)
        {
            for (uint32_t i = 0; i < created.size(); ++i)
            {
                ModelEntity m;
                m.mask = c.mask;
                m.a = (c.mask & kA) ? c.a[i].v : 0;
                m.b = (c.mask & kB) ? c.b[i].v : 0;
                m.sh = (c.mask & kSh) ? c.sh[i].key : 0;
                m.s = (c.mask & kS) ? c.s[i].v : 0;
                model.entities[created[i]] = m;
            }
            model.Created(created);
        };
        auto copyOf = [&](uint64_t source, const std::vector<uint64_t>& copies)
        {
            const ModelEntity m = model.entities[source];
            for (const uint64_t copy : copies)
                model.entities[copy] = m;
            model.Created(copies);
        };

        // Compares the callbacks with the prediction: per observer, the same (entity, value) sequence; across
        // observers, the batches come in the order of their earliest event.
        auto compare = [&]()
        {
            std::vector<std::pair<uint64_t, uint64_t>> order;   // (callback, sequence)
            for (uint32_t o = 0; o < c_ModelObservers; ++o)
            {
                CHECK(actual[o].size() == model.expected[o].size());
                for (size_t i = 0; i < actual[o].size(); ++i)
                {
                    CHECK(actual[o][i].entity == model.expected[o][i].entity);
                    const Expected& e = model.expected[o][i];
                    const bool later = std::find(e.later.begin(), e.later.end(), actual[o][i].value) != e.later.end();
                    CHECK(actual[o][i].value == e.value || later);
                    repeatedShown += later && actual[o][i].value != e.value ? 1 : 0;
                    order.emplace_back(actual[o][i].callback, model.expected[o][i].sequence);
                }
                totalEvents += actual[o].size();
                actual[o].clear();
                model.expected[o].clear();
            }
            std::sort(order.begin(), order.end());
            uint64_t previousMinimum = 0;
            for (size_t i = 0; i < order.size();)
            {
                size_t end = i;
                uint64_t minimum = order[i].second;
                while (end < order.size() && order[end].first == order[i].first)
                    minimum = std::min(minimum, order[end++].second);
                CHECK(minimum >= previousMinimum);
                previousMinimum = minimum;
                i = end;
            }
            CHECK(wellFormed);
            CHECK(Registry(raw).GetHeld().kind == ecs::ReportKind::None);
        };

        auto buffer = MakeBuffer(world);
        ICommandBuffer* list[1] = { buffer.Get() };
        for (uint32_t step = 0; step < 300; ++step)
        {
            std::vector<uint64_t> live = alive();
            if (live.size() < 8 || rng.Below(100) < 60)
            {
                // ------------------------------------------------------------------------ an immediate operation
                const uint32_t op = live.size() < 8 ? 0 : rng.Below(8);
                switch (op)
                {
                case 0:
                {
                    const uint32_t count = 1 + rng.Below(24);
                    Creation c;
                    makeCreation(count, c);
                    const CreateDesc desc(c.components, c.values, count);
                    std::vector<Entity> e(count);
                    CHECK(world->CreateEntities(desc, e.data()) == nvrhi::FS_OK);
                    modelCreation(c, std::vector<uint64_t>(reinterpret_cast<uint64_t*>(e.data()),
                                                           reinterpret_cast<uint64_t*>(e.data()) + count));
                    break;
                }
                case 1:
                {
                    std::vector<uint64_t> victims;
                    const uint32_t count = 1 + rng.Below(6);
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        const uint64_t id = pick(live);
                        if (std::find(victims.begin(), victims.end(), id) == victims.end())
                            victims.push_back(id);
                    }
                    CHECK(world->DestroyEntities(reinterpret_cast<const Entity*>(victims.data()),
                                                 uint32_t(victims.size())) == nvrhi::FS_OK);
                    for (const uint64_t id : victims)
                        model.Destroy(id);
                    break;
                }
                case 2:
                case 3:
                {
                    const uint64_t id = pick(live);
                    const uint32_t k = rng.Below(6);
                    const uint32_t v = (bitOf[k] == kSh) ? 1 + rng.Below(3) : value();
                    if (op == 2)
                    {
                        const void* pointer = nullptr;
                        A a{ v };
                        B b{ v };
                        Sh sh{ v };
                        S s{ v };
                        if (bitOf[k] == kA) pointer = &a;
                        if (bitOf[k] == kB) pointer = &b;
                        if (bitOf[k] == kSh) pointer = &sh;
                        if (bitOf[k] == kS) pointer = &s;
                        CHECK(world->AddComponent(Entity(id), componentOf[k], pointer) == nvrhi::FS_OK);
                        model.Add(id, bitOf[k], v);
                    }
                    else
                    {
                        CHECK(world->RemoveComponent(Entity(id), componentOf[k]) == nvrhi::FS_OK);
                        model.Remove(id, bitOf[k]);
                    }
                    break;
                }
                case 4:
                {
                    const uint64_t prefab = pick(live);
                    const uint32_t count = 1 + rng.Below(8);
                    std::vector<Entity> e(count);
                    CHECK(world->Instantiate(Entity(prefab), count, e.data()) == nvrhi::FS_OK);
                    copyOf(prefab, std::vector<uint64_t>(reinterpret_cast<uint64_t*>(e.data()),
                                                         reinterpret_cast<uint64_t*>(e.data()) + count));
                    break;
                }
                case 5:
                {
                    // Copies within the world: per copy, rows then Sparse components, like a creation.
                    const uint32_t count = 1 + rng.Below(4);
                    std::vector<uint64_t> sources;
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        const uint64_t id = pick(live);
                        if (std::find(sources.begin(), sources.end(), id) == sources.end())
                            sources.push_back(id);
                    }
                    std::vector<Entity> copies(sources.size());
                    CHECK(world->CopyEntitiesFrom(world, reinterpret_cast<const Entity*>(sources.data()),
                                                  uint32_t(sources.size()), copies.data()) == nvrhi::FS_OK);
                    std::vector<uint64_t> created;
                    for (size_t i = 0; i < sources.size(); ++i)
                    {
                        model.entities[uint64_t(copies[i])] = model.entities[sources[i]];
                        created.push_back(uint64_t(copies[i]));
                    }
                    model.Created(created);
                    break;
                }
                case 6:
                {
                    // Not structural: nothing is reported, the model only follows the values.
                    std::vector<uint64_t> withItems;
                    for (const uint64_t id : live)
                        if (model.entities[id].mask & kItem)
                            withItems.push_back(id);
                    if (withItems.empty())
                        break;
                    const uint64_t id = pick(withItems);
                    const uint32_t count = rng.Below(6);
                    CHECK(world->ResizeBuffer(Entity(id), ids.item, count, c_WorldVersion) == nvrhi::FS_OK);
                    const BufferHeader* header = static_cast<const BufferHeader*>(
                        world->GetComponentForWrite(Entity(id), ids.item, c_WorldVersion));
                    model.entities[id].items.assign(count, 0);
                    for (uint32_t k = 0; k < count; ++k)
                    {
                        const uint32_t v = value();
                        GetBufferData<Item>(*header)[k].v = v;
                        model.entities[id].items[k] = v;
                    }
                    break;
                }
                default:
                {
                    std::vector<uint64_t> withA;
                    for (const uint64_t id : live)
                        if (model.entities[id].mask & kA)
                            withA.push_back(id);
                    if (withA.empty())
                        break;
                    const uint64_t id = pick(withA);
                    const uint32_t v = value();
                    static_cast<A*>(world->GetComponentForWrite(Entity(id), ids.a, c_WorldVersion))->v = v;
                    model.entities[id].a = v;
                    break;
                }
                }
                compare();
                ++totalSteps;
                continue;
            }

            // ------------------------------------------------------------------------------------- a playback
            // The model applies each command as it is recorded. A run of adjacent commands of one operation and
            // component (which playback coalesces into one batch operation) may name an entity again (Model's
            // allowance), except a Destroy run and an Add run of B, whose observer 14 writes what a repeat overwrites.
            std::vector<uint64_t> placeholders;
            uint32_t runOp = ~0u;
            ComponentId runComponent = ComponentId::Invalid;
            std::set<uint64_t> runEntities;
            std::vector<std::unique_ptr<Creation>> creations;   // values must outlive recording only, but keep them
            const uint32_t commands = 1 + rng.Below(12);
            uint64_t key = 0;
            for (uint32_t c = 0; c < commands; ++c)
            {
                live = alive();
                if (live.empty())
                    break;
                const uint32_t op = rng.Below(7);
                const uint32_t k = rng.Below(6);
                const uint32_t setChoice = rng.Below(3);   // Set: A, S or Sh
                const uint32_t setBit = setChoice == 0 ? kA : (setChoice == 1 ? kS : kSh);
                const ComponentId setComponent = setChoice == 0 ? ids.a : (setChoice == 1 ? ids.s : ids.sh);
                ComponentId runKey = ComponentId::Invalid;
                if (op == 2 || op == 3)
                    runKey = componentOf[k];
                else if (op == 5)
                    runKey = setComponent;
                if (op != runOp || runKey != runComponent)
                {
                    runEntities.clear();
                    runOp = op;
                    runComponent = runKey;
                    model.BeginRun();
                }
                const bool repeats = op != 1 && !(op == 2 && componentOf[k] == ids.b);
                auto pickRun = [&](const std::vector<uint64_t>& candidates, uint32_t count)
                {
                    std::vector<uint64_t> chosen;
                    for (uint32_t i = 0; i < count * 2 && chosen.size() < count; ++i)
                    {
                        uint64_t id = pick(candidates);
                        if (repeats && !runEntities.empty() && rng.Below(3) == 0)
                        {
                            auto it = runEntities.begin();
                            std::advance(it, rng.Below(uint32_t(runEntities.size())));
                            id = *it;   // again in the run, maybe in the same command
                        }
                        else if (!repeats &&
                                 (runEntities.count(id) || std::find(chosen.begin(), chosen.end(), id) != chosen.end()))
                        {
                            continue;
                        }
                        chosen.push_back(id);
                        runEntities.insert(id);
                    }
                    return chosen;
                };
                switch (op)
                {
                case 0:
                {
                    const uint32_t count = 1 + rng.Below(10);
                    creations.push_back(std::make_unique<Creation>());
                    Creation& creation = *creations.back();
                    makeCreation(count, creation);
                    const CreateDesc desc(creation.components, creation.values, count);
                    std::vector<Entity> p(count);
                    buffer->CreateEntities(desc, key++, p.data());
                    std::vector<uint64_t> created;
                    for (const Entity entity : p)
                    {
                        created.push_back(uint64_t(entity));
                        placeholders.push_back(uint64_t(entity));
                    }
                    modelCreation(creation, created);
                    break;
                }
                case 1:
                {
                    const std::vector<uint64_t> victims = pickRun(live, 1 + rng.Below(4));
                    if (victims.empty())
                        break;
                    buffer->DestroyEntities(reinterpret_cast<const Entity*>(victims.data()), uint32_t(victims.size()),
                                            key++);
                    for (const uint64_t id : victims)
                        model.Destroy(id);
                    break;
                }
                case 2:
                {
                    const std::vector<uint64_t> targets = pickRun(live, 1 + rng.Below(6));
                    if (targets.empty())
                        break;
                    const uint32_t n = uint32_t(targets.size());
                    const uint32_t bit = bitOf[k];
                    std::vector<A> a(n);
                    std::vector<B> b(n);
                    std::vector<Sh> sh(n);
                    std::vector<S> s(n);
                    const void* values = nullptr;
                    const bool withValues = bit != kC && bit != kItem && rng.Below(4) != 0;
                    std::vector<uint32_t> v(n);
                    for (uint32_t i = 0; i < n; ++i)
                    {
                        v[i] = bit == kSh ? 1 + rng.Below(3) : value();
                        a[i].v = b[i].v = s[i].v = v[i];
                        sh[i].key = v[i];
                    }
                    if (withValues)
                    {
                        if (bit == kA) values = a.data();
                        if (bit == kB) values = b.data();
                        if (bit == kSh) values = sh.data();
                        if (bit == kS) values = s.data();
                    }
                    buffer->AddComponents(reinterpret_cast<const Entity*>(targets.data()), n, componentOf[k], values,
                                          key++);
                    for (uint32_t i = 0; i < n; ++i)
                    {
                        const ModelEntity& m = model.entities[targets[i]];
                        if (!withValues)
                        {
                            // Defaults: a zero value for an absent component, nothing for a present one (a Shared
                            // default is the registered default, zero).
                            if (!(m.mask & bit))
                                model.Add(targets[i], bit, 0);
                        }
                        else
                        {
                            model.Add(targets[i], bit, v[i]);
                        }
                    }
                    break;
                }
                case 3:
                {
                    const std::vector<uint64_t> targets = pickRun(live, 1 + rng.Below(6));
                    if (targets.empty())
                        break;
                    buffer->RemoveComponents(reinterpret_cast<const Entity*>(targets.data()), uint32_t(targets.size()),
                                             componentOf[k], key++);
                    for (const uint64_t id : targets)
                        model.Remove(id, bitOf[k]);
                    break;
                }
                case 4:
                {
                    const uint64_t prefab = pick(live);
                    const uint32_t count = 1 + rng.Below(6);
                    std::vector<Entity> p(count);
                    buffer->Instantiate(Entity(prefab), count, key++, p.data());
                    std::vector<uint64_t> created;
                    for (const Entity entity : p)
                    {
                        created.push_back(uint64_t(entity));
                        placeholders.push_back(uint64_t(entity));
                    }
                    copyOf(prefab, created);
                    break;
                }
                case 5:
                {
                    // Sets of A, S or Sh on entities that have them.
                    const uint32_t bit = setBit;
                    const ComponentId component = setComponent;
                    std::vector<uint64_t> candidates;
                    for (const uint64_t id : live)
                        if (model.entities[id].mask & bit)
                            candidates.push_back(id);
                    if (candidates.empty())
                        break;
                    const std::vector<uint64_t> targets = pickRun(candidates, 1 + rng.Below(6));
                    if (targets.empty())
                        break;
                    const uint32_t n = uint32_t(targets.size());
                    std::vector<A> a(n);
                    std::vector<S> s(n);
                    std::vector<Sh> sh(n);
                    std::vector<uint32_t> v(n);
                    for (uint32_t i = 0; i < n; ++i)
                    {
                        v[i] = bit == kSh ? 1 + rng.Below(3) : value();
                        a[i].v = s[i].v = v[i];
                        sh[i].key = v[i];
                    }
                    const void* values = bit == kA ? static_cast<const void*>(a.data())
                                                   : (bit == kS ? static_cast<const void*>(s.data()) : sh.data());
                    buffer->SetComponents(reinterpret_cast<const Entity*>(targets.data()), n, component, values,
                                          key++);
                    for (uint32_t i = 0; i < n; ++i)
                        model.Set(targets[i], bit, v[i]);
                    break;
                }
                default:
                {
                    std::vector<uint64_t> withItems;
                    for (const uint64_t id : live)
                        if (model.entities[id].mask & kItem)
                            withItems.push_back(id);
                    if (withItems.empty())
                        break;
                    const uint64_t id = pick(withItems);
                    const uint32_t count = 1 + rng.Below(3);
                    std::vector<Item> items(count);
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        items[i].v = value();
                        model.entities[id].items.push_back(items[i].v);
                    }
                    buffer->AppendToBuffer(Entity(id), ids.item, items.data(), count, key++);
                    break;
                }
                }
            }
            model.EndRun();
            CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
            const ecs::PlaceholderTable& table = Impl(world).GetCommandPlayback().placeholders;
            for (const uint64_t placeholder : placeholders)
            {
                const Entity entity = table.Resolve(Entity(placeholder));
                CHECK(entity != Entity::Null);
                model.Rename(placeholder, uint64_t(entity));
            }
            compare();
            ++totalSteps;
        }

        // The world holds what the model holds.
        for (const auto& entry : model.entities)
        {
            const Entity entity = Entity(entry.first);
            const ModelEntity& m = entry.second;
            CHECK(world->IsAlive(entity));
            for (uint32_t k = 0; k < 6; ++k)
                CHECK(world->HasComponent(entity, componentOf[k]) == ((m.mask & bitOf[k]) != 0));
            if (m.mask & kA) CHECK(Get<A>(world, entity).v == m.a);
            if (m.mask & kB) CHECK(Get<B>(world, entity).v == m.b);
            if (m.mask & kS) CHECK(Get<S>(world, entity).v == m.s);
            if (m.mask & kSh) CHECK(Get<Sh>(world, entity).key == m.sh);
        }
        CHECK(Errors(world) == 0);
        totalCallbacks += callbacks;
        extended += Registry(world).GetCounters().extended;
    }
    printf("test_ecs_observers: model: %llu steps, %llu callbacks, %llu events, %llu extensions, %llu repeated "
           "entities showing a later value of their run\n", (unsigned long long)totalSteps,
           (unsigned long long)totalCallbacks, (unsigned long long)totalEvents, (unsigned long long)extended,
           (unsigned long long)repeatedShown);
    CHECK(extended > 0 && repeatedShown > 0);   // the merges and the repeats are exercised
}

// ------------------------------------------------------------------------------------------------ cascades

void test_cascades()
{
    for (const bool validation : { false, true })
    {
        // Each round's commands trigger the next round: A -> B -> C.
        auto world = MakeWorld(validation);
        const Ids ids = Register(world);
        std::vector<uint32_t> order;   // observer per callback
        uint32_t cCount = 0;
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) {
                order.push_back(0);
                std::vector<B> values(batch.count);
                for (uint32_t i = 0; i < batch.count; ++i)
                    values[i].v = static_cast<const A*>(batch.columns[0])[i].v + 1;
                commands->AddComponents(batch.entities, batch.count, ids.b, values.data(), 0);
                return nvrhi::FS_OK;
            });
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.b, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) {
                order.push_back(1);
                commands->AddComponents(batch.entities, batch.count, ids.c, nullptr, 0);
                return nvrhi::FS_OK;
            });
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.c, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                order.push_back(2);
                cCount += batch.count;
                return nvrhi::FS_OK;
            });
        std::vector<A> values(10);
        for (uint32_t i = 0; i < 10; ++i)
            values[i].v = i;
        WorldStats before;
        world->GetStats(before);
        const uint64_t rounds = Registry(world).GetCounters().rounds;
        const std::vector<Entity> e = Create(world, CreateDesc({ ids.a }, { values.data() }, 10));
        CHECK(order == std::vector<uint32_t>({ 0, 1, 2 }) && cCount == 10);
        CHECK(Registry(world).GetCounters().rounds - rounds == 2);
        WorldStats after;
        world->GetStats(after);
        CHECK(after.commandsPlayedBack - before.commandsPlayedBack == 2);
        CHECK(after.observerBatches - before.observerBatches == 3);
        for (uint32_t i = 0; i < 10; ++i)
            CHECK(Get<B>(world, e[i]).v == i + 1 && world->HasComponent(e[i], ids.c));

        // The same from a playback: the observers' commands follow the system's.
        order.clear();
        auto buffer = MakeBuffer(world);
        Entity p[3];
        buffer->CreateEntities(CreateDesc({ ids.a }, { values.data() }, 3), 0, p);
        ICommandBuffer* list[1] = { buffer.Get() };
        CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
        CHECK(order == std::vector<uint32_t>({ 0, 1, 2 }));
        for (const Entity placeholder : p)
            CHECK(world->HasComponent(world->ResolvePlaceholder(placeholder), ids.c));
        CHECK(Errors(world) == 0);
    }

    // Placeholders of one round resolve for the commands after them in the cascade.
    {
        auto world = MakeWorld(true);
        const Ids ids = Register(world);
        std::vector<Entity> made;
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.c, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) {
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    Entity placeholder;
                    commands->CreateEntities(CreateDesc({ ids.b }, {}, 1), 0, &placeholder);
                    const A a{ 7 };
                    commands->AddComponent(placeholder, ids.a, &a, 0);
                }
                return nvrhi::FS_OK;
            });
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, { Term(ids.b, TermOp::All) }),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                made.insert(made.end(), batch.entities, batch.entities + batch.count);
                return nvrhi::FS_OK;
            });
        Create(world, CreateDesc({ ids.c }, {}, 4));
        CHECK(made.size() == 4);
        for (const Entity entity : made)
            CHECK(Get<A>(world, entity).v == 7 && world->HasComponent(entity, ids.b));
        // Their table is the cascade's: the public one is not grown by immediate calls.
        CHECK(Impl(world).GetCommandPlayback().placeholders.ranges.empty());
        CHECK(Errors(world) == 0);
    }

    // The limit: an observer that triggers itself forever stops after 64 rounds.
    for (const bool validation : { false, true })
    {
        auto world = MakeWorld(validation);
        const Ids ids = Register(world);
        uint32_t calls = 0;
        const ObserverId self = AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch&, ICommandBuffer* commands) {
                ++calls;
                Entity placeholder;
                commands->CreateEntities(CreateDesc({ ids.a }, {}, 1), 0, &placeholder);
                return nvrhi::FS_OK;
            });
        const uint64_t errors = Errors(world);
        const uint64_t rounds = Registry(world).GetCounters().rounds;
        Entity first;
        CHECK(world->CreateEntities(CreateDesc({ ids.a }, {}, 1), &first) == nvrhi::FE_GENERIC_ERROR);
        CHECK(calls == ecs::c_MaxObserverRounds + 1);   // the triggering change and 64 rounds
        CHECK(Registry(world).GetCounters().rounds - rounds == ecs::c_MaxObserverRounds);
        CHECK(Registry(world).GetCounters().cascadeFailures == 1);
        CHECK(Errors(world) - errors == (validation ? 1u : 0u));
        CHECK(Registry(world).GetRecordingBuffer()->GetCommandCount() == 0);   // the last round's command dropped
        WorldStats stats;
        CHECK(world->GetStats(stats).entityCount == ecs::c_MaxObserverRounds + 1);
        // The next call starts afresh.
        CHECK(world->RemoveObserver(self) == nvrhi::FS_OK);
        CHECK(world->CreateEntities(CreateDesc({ ids.a }, {}, 1), &first) == nvrhi::FS_OK);
        CHECK(world->GetStats(stats).entityCount == ecs::c_MaxObserverRounds + 2);
    }

    // An observer's failure is the call's result; the other observers still run.
    for (const bool validation : { false, true })
    {
        auto world = MakeWorld(validation);
        const Ids ids = Register(world);
        uint32_t second = 0;
        AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.a, Access::Read, {}),
            [](ObserverEvent, const EntityBatch&, ICommandBuffer*) { return nvrhi::FE_UNSUPPORTED; });
        AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.a, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                second += batch.count;
                return nvrhi::FS_OK;
            });
        const std::vector<Entity> e = Create(world, CreateDesc({ ids.a }, {}, 3));
        const uint64_t errors = Errors(world);
        CHECK(world->DestroyEntities(e.data(), 3) == nvrhi::FE_UNSUPPORTED);
        CHECK(second == 3 && !world->IsAlive(e[0]) && !world->IsAlive(e[2]));
        CHECK(Errors(world) - errors == (validation ? 3u : 0u));
        CHECK(world->DestroyEntities(nullptr, 0) == nvrhi::FS_OK);   // the failure was returned once
    }
}

// --------------------------------------------------------------------------------------------- re-entrancy

void test_reentrancy()
{
    auto world = MakeWorld(true);
    const Ids ids = Register(world);
    IWorld* raw = world.Get();
    auto other = MakeWorld();
    Register(other);
    const std::vector<Entity> sources = Create(other, CreateDesc({ FindComponent<A>(other) }, {}, 1));
    nvrhi::AutoPtr<IQuery> query;
    const QueryTerm aTerm = Term(ids.a, TermOp::All, Access::Read);
    QueryDesc queryDesc{};
    queryDesc.terms = &aTerm;
    queryDesc.termCount = 1;
    CHECK(world->CreateQuery(queryDesc, &query) == nvrhi::FS_OK);
    auto outsideBuffer = MakeBuffer(world);
    const std::vector<Entity> bystander = Create(world, CreateDesc({ ids.a, ids.tog, ids.item }, {}, 1));

    ObserverId first = ObserverId::Invalid;
    ObserverId second = ObserverId::Invalid;
    uint32_t checks = 0;
    uint32_t secondCalls = 0;
    auto tryEverything = [&](const EntityBatch& batch)
    {
        Entity out[2];
        EXPECT(raw->CreateEntities(CreateDesc({ ids.a }, {}, 1), out) == nvrhi::FE_INVALID_ARGS);
        EXPECT(raw->DestroyEntities(batch.entities, 1) == nvrhi::FE_INVALID_ARGS);
        EXPECT(raw->AddComponent(batch.entities[0], ids.b, nullptr) == nvrhi::FE_INVALID_ARGS);
        EXPECT(raw->RemoveComponent(batch.entities[0], ids.a) == nvrhi::FE_INVALID_ARGS);
        EXPECT(raw->Instantiate(batch.entities[0], 1, out) == nvrhi::FE_INVALID_ARGS);
        EXPECT(raw->ReserveEntities(1, out) == nvrhi::FE_INVALID_ARGS);
        EXPECT(raw->CopyEntitiesFrom(other, sources.data(), 1, out) == nvrhi::FE_INVALID_ARGS);
        ICommandBuffer* list[1] = { outsideBuffer.Get() };
        EXPECT(raw->Playback(list, 1) == ecs::WorldImpl::c_WrongTimeResult);
        MemoryWriter writer;
        EXPECT(raw->SaveSnapshot(&writer) == nvrhi::FE_INVALID_ARGS);
        MemoryReader reader(nullptr, 0);
        EXPECT(raw->LoadSnapshot(&reader) == nvrhi::FE_INVALID_ARGS);
        ObserverId id = ObserverId::Invalid;
        EXPECT(raw->AddObserver(MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}),
                               static_cast<IObserver*>(nullptr), id) == nvrhi::FE_INVALID_ARGS);
        // Removal during delivery is refused: both observers stay, and the other one still gets this batch.
        EXPECT(raw->RemoveObserver(second) == nvrhi::FE_INVALID_ARGS);
        EXPECT(raw->RemoveObserver(first) == nvrhi::FE_INVALID_ARGS);
        // A new type is refused; looking up a registered one is not.
        ComponentId late = ComponentId::Invalid;
        EXPECT(raw->RegisterComponent(ComponentTraits<Late>::Desc(), late) == ecs::WorldImpl::c_WrongTimeResult);
        EXPECT(RegisterComponent<A>(raw) == ids.a);
        // Reads, writes of values and recording are allowed.
        EXPECT(raw->IsAlive(batch.entities[0]) && raw->HasComponent(batch.entities[0], ids.a));
        EXPECT(raw->GetComponent(batch.entities[0], ids.a) != nullptr);
        EXPECT(raw->GetResource(ids.res) != nullptr);
        EXPECT(query->CountEntities() >= 1);
        static_cast<A*>(raw->GetComponentForWrite(bystander[0], ids.a, c_WorldVersion))->v = 42;
        EXPECT(raw->SetEnabled(bystander[0], ids.tog, false) == nvrhi::FS_OK);
        EXPECT(raw->ResizeBuffer(bystander[0], ids.item, 3, c_WorldVersion) == nvrhi::FS_OK);
        ++checks;
    };
    first = AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.b, Access::Read, {}),
        [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
            tryEverything(batch);
            return nvrhi::FS_OK;
        });
    second = AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.b, Access::Read, {}),
        [&](ObserverEvent, const EntityBatch&, ICommandBuffer*) {
            ++secondCalls;
            return nvrhi::FS_OK;
        });
    const ObserverId third = AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.b, Access::Read, {}),
        [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
            tryEverything(batch);
            return nvrhi::FS_OK;
        });

    const std::vector<Entity> e = Create(world, CreateDesc({ ids.a, ids.b }, {}, 2));   // OnAdd, held
    CHECK_EXPECTATIONS();
    CHECK(checks == 1);
    CHECK(world->DestroyEntities(e.data(), 1) == nvrhi::FS_OK);                         // OnRemove, at once
    CHECK_EXPECTATIONS();
    CHECK(checks == 2 && secondCalls == 1);
    CHECK(Get<A>(world, bystander[0]).v == 42 && !world->IsEnabled(bystander[0], ids.tog));
    // Outside delivery the same calls work.
    CHECK(world->RemoveObserver(second) == nvrhi::FS_OK);
    CHECK(world->RemoveObserver(first) == nvrhi::FS_OK);
    CHECK(world->RemoveObserver(third) == nvrhi::FS_OK);
    CHECK(world->DestroyEntities(e.data() + 1, 1) == nvrhi::FS_OK);
    CHECK(checks == 2 && secondCalls == 1);
    ComponentId late = ComponentId::Invalid;
    CHECK(world->RegisterComponent(ComponentTraits<Late>::Desc(), late) == nvrhi::FS_OK);
}

// --------------------------------------------------------------------------------------------- snapshots

void test_snapshots()
{
    for (const bool validation : { false, true })
    {
        auto world = MakeWorld(validation, 1024);
        const Ids ids = Register(world);
        IWorld* raw = world.Get();

        // What gets saved: A in two archetypes, Sparse values on some.
        std::vector<A> values(120);
        for (uint32_t i = 0; i < 120; ++i)
            values[i].v = 100 + i;
        const std::vector<Entity> saved1 = Create(world, CreateDesc({ ids.a }, { values.data() }, 70));
        const std::vector<Entity> saved2 = Create(world, CreateDesc({ ids.a, ids.b }, { values.data() + 70, nullptr },
                                                                    50));
        for (uint32_t i = 0; i < 120; i += 7)
        {
            const S s{ i };
            CHECK(world->AddComponent(i < 70 ? saved1[i] : saved2[i - 70], ids.s, &s) == nvrhi::FS_OK);
        }
        MemoryWriter writer;
        CHECK(world->SaveSnapshot(&writer) == nvrhi::FS_OK);

        // Observers that allocate nothing while they run: fixed arrays, reserved vectors.
        constexpr uint32_t c_Capacity = 4096;
        struct Tally
        {
            std::vector<Entity> entities;
            uint64_t sum = 0;
            uint32_t batches = 0;
            bool wholeChunks = true;
        };
        Tally removedA, addedA, removedS, addedS, unmatched, matched;
        for (Tally* tally : { &removedA, &addedA, &removedS, &addedS, &unmatched, &matched })
            tally->entities.reserve(c_Capacity);
        uint64_t firstNews = 0, lastNews = 0, firstBlocks = 0, lastBlocks = 0;
        bool firstCall = true;
        auto note = [&]()
        {
            const uint64_t news = g_NewCount.load();
            const uint64_t blocks = ecs::GetHeapAllocationCount();
            if (firstCall)
            {
                firstNews = news;
                firstBlocks = blocks;
                firstCall = false;
            }
            lastNews = news;
            lastBlocks = blocks;
        };
        auto tallyInto = [&](Tally& tally, bool valued, bool chunks)
        {
            return [&tally, &note, valued, chunks, raw](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                note();
                ++tally.batches;
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    tally.entities.push_back(batch.entities[i]);
                    if (valued)
                        tally.sum += static_cast<const A*>(batch.columns[0])[i].v;
                }
                if (chunks)
                {
                    // A load removes whole chunks: the batch starts at row 0 and holds every row of its chunk.
                    const ecs::EntityRecord* record = Impl(raw).FindAlive(batch.entities[0]);
                    const ecs::Archetype& archetype = Impl(raw).GetArchetypeStorage(batch.archetypeIndex);
                    tally.wholeChunks = tally.wholeChunks && record && record->row == 0 &&
                                        archetype.chunks[record->chunk].count == batch.count;
                }
                return nvrhi::FS_OK;
            };
        };
        AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.a, Access::Read, {}), tallyInto(removedA, true, true));
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}), tallyInto(addedA, true, false));
        AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.s, Access::Read, {}),
                    tallyInto(removedS, false, false));
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.s, Access::Read, {}), tallyInto(addedS, false, false));
        const std::vector<QueryTerm> filter = { Term(ids.b, TermOp::All) };
        AddObserver(world, MakeDesc(ObserverEvent::OnUnmatch, ComponentId::Invalid, Access::Read, filter),
                    tallyInto(unmatched, false, true));
        AddObserver(world, MakeDesc(ObserverEvent::OnMatch, ComponentId::Invalid, Access::Read, filter),
                    tallyInto(matched, false, false));

        auto entitiesWith = [&](ComponentId component)
        {
            std::vector<Entity> list;
            const uint32_t size = Impl(raw).GetEntityTableSize();
            for (uint32_t index = 1; index < size; ++index)
            {
                const Entity entity = MakeEntity(index, Impl(raw).GetEntityRecord(index).generation);
                if (raw->IsAlive(entity) && raw->HasComponent(entity, component))
                    list.push_back(entity);
            }
            std::sort(list.begin(), list.end());
            return list;
        };
        auto sorted = [](std::vector<Entity> list) {
            std::sort(list.begin(), list.end());
            return list;
        };
        auto sumOfA = [&]() {
            uint64_t sum = 0;
            for (const Entity entity : entitiesWith(ids.a))
                sum += Get<A>(raw, entity).v;
            return sum;
        };
        const std::vector<Entity> savedA = entitiesWith(ids.a);
        const std::vector<Entity> savedS = entitiesWith(ids.s);
        const std::vector<Entity> savedB = entitiesWith(ids.b);
        const uint64_t savedSum = sumOfA();

        for (uint32_t round = 0; round < 2; ++round)
        {
            // Change the world: destroy some, make others, move some.
            for (uint32_t i = 0; i < 30; i += 3)
                CHECK(world->DestroyEntities(&saved1[i], 1) == nvrhi::FS_OK || round == 1);
            const std::vector<Entity> fresh =
                Create(world, CreateDesc({ ids.a, ids.c }, { values.data(), nullptr }, 25));
            const S s{ 9 };
            for (uint32_t i = 0; i < 25; i += 2)
                CHECK(world->AddComponent(fresh[i], ids.s, &s) == nvrhi::FS_OK);
            CHECK(world->AddComponent(fresh[0], ids.b, nullptr) == nvrhi::FS_OK);

            const std::vector<Entity> oldA = entitiesWith(ids.a);
            const std::vector<Entity> oldS = entitiesWith(ids.s);
            const std::vector<Entity> oldB = entitiesWith(ids.b);
            const uint64_t oldSum = sumOfA();
            for (Tally* tally : { &removedA, &addedA, &removedS, &addedS, &unmatched, &matched })
            {
                tally->entities.clear();
                tally->sum = 0;
                tally->batches = 0;
                tally->wholeChunks = true;
            }
            firstCall = true;

            MemoryReader reader(writer.bytes.data(), writer.bytes.size());
            CHECK(world->LoadSnapshot(&reader) == nvrhi::FS_OK);

            // A load is destroy-all, then create-all.
            CHECK(sorted(removedA.entities) == oldA && removedA.sum == oldSum && removedA.wholeChunks);
            CHECK(sorted(addedA.entities) == savedA && addedA.sum == savedSum);
            CHECK(sorted(removedS.entities) == oldS && sorted(addedS.entities) == savedS);
            CHECK(sorted(unmatched.entities) == oldB && unmatched.wholeChunks);
            CHECK(sorted(matched.entities) == savedB);
            CHECK(entitiesWith(ids.a) == savedA);
            // The load's commit (between the first and the last callback) allocates nothing: delivery holds at most
            // one batch and queues nothing. The second round loads into pools and containers the first one grew.
            if (round == 1)
            {
                printf("test_ecs_observers: snapshot load: %llu operator new, %llu storage blocks between the first "
                       "and the last callback\n", (unsigned long long)(lastNews - firstNews),
                       (unsigned long long)(lastBlocks - firstBlocks));
                CHECK(lastNews == firstNews);
                CHECK(lastBlocks == firstBlocks);
            }
        }
        CHECK(Errors(world) == 0);
    }

    // Once the content is replaced the load succeeds: an observer's failure and the cascade limit after it are reported
    // through validation, not returned. Commands recorded by OnRemove observers during the clear run after the load,
    // against the loaded entities (ids are kept, so here the same ids).
    for (const bool validation : { false, true })
    {
        auto world = MakeWorld(validation);
        const Ids ids = Register(world);
        const std::vector<Entity> saved = Create(world, CreateDesc({ ids.a }, {}, 3));
        MemoryWriter writer;
        CHECK(world->SaveSnapshot(&writer) == nvrhi::FS_OK);
        auto load = [&]() {
            MemoryReader reader(writer.bytes.data(), writer.bytes.size());
            return world->LoadSnapshot(&reader);
        };
        WorldStats stats;

        const ObserverId failing = AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}),
            [](ObserverEvent, const EntityBatch&, ICommandBuffer*) { return nvrhi::FE_UNSUPPORTED; });
        uint64_t errors = Errors(world);
        CHECK(load() == nvrhi::FS_OK);
        CHECK(world->GetStats(stats).entityCount == 3);
        CHECK(Errors(world) - errors == (validation ? 2u : 0u));   // the observer's report and the load's
        CHECK(world->RemoveObserver(failing) == nvrhi::FS_OK);

        const ObserverId forever = AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch&, ICommandBuffer* commands) {
                Entity placeholder;
                commands->CreateEntities(CreateDesc({ ids.a }, {}, 1), 0, &placeholder);
                return nvrhi::FS_OK;
            });
        errors = Errors(world);
        CHECK(load() == nvrhi::FS_OK);
        CHECK(Registry(world).GetCounters().cascadeFailures == 1);
        CHECK(world->GetStats(stats).entityCount == 3 + ecs::c_MaxObserverRounds);
        CHECK(Errors(world) - errors == (validation ? 2u : 0u));   // the cascade limit and the load
        CHECK(world->RemoveObserver(forever) == nvrhi::FS_OK);

        AddObserver(world, MakeDesc(ObserverEvent::OnRemove, ids.a, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) {
                commands->AddComponents(batch.entities, batch.count, ids.c, nullptr, 0);
                return nvrhi::FS_OK;
            });
        CHECK(load() == nvrhi::FS_OK);
        CHECK(world->GetStats(stats).entityCount == 3);
        for (const Entity entity : saved)
            CHECK(world->IsAlive(entity) && world->HasComponent(entity, ids.c));
    }

    // The observers' commands run after the load, and their placeholders resolve within the cascade.
    {
        auto world = MakeWorld();
        const Ids ids = Register(world);
        Create(world, CreateDesc({ ids.a }, {}, 5));
        MemoryWriter writer;
        CHECK(world->SaveSnapshot(&writer) == nvrhi::FS_OK);
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read, {}),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) {
                commands->AddComponents(batch.entities, batch.count, ids.c, nullptr, 0);
                return nvrhi::FS_OK;
            });
        MemoryReader reader(writer.bytes.data(), writer.bytes.size());
        CHECK(world->LoadSnapshot(&reader) == nvrhi::FS_OK);
        WorldStats stats;
        CHECK(world->GetStats(stats).entityCount == 5);
        const QueryTerm terms[2] = { Term(ids.a, TermOp::All), Term(ids.c, TermOp::All) };
        QueryDesc desc{};
        desc.terms = terms;
        desc.termCount = 2;
        nvrhi::AutoPtr<IQuery> query;
        CHECK(world->CreateQuery(desc, &query) == nvrhi::FS_OK);
        CHECK(query->CountEntities() == 5);
    }
}

// ---------------------------------------------------------------------------------------------- lifetime

void test_lifetime()
{
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    bool destroyed[3] = { false, false, false };
    nvrhi::AutoPtr<TestObserver> kept;
    {
        auto world = MakeWorld(true);
        const Ids ids = Register(world);
        IWorld* raw = world.Get();
        const nvrhi::FLONG references = raw->AddRef();
        raw->Release();
        AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.a, Access::Read,
                                    { Term(ids.b, TermOp::Optional, Access::Read), Term(ids.c, TermOp::Without) }),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) {
                commands->AddComponents(batch.entities, batch.count, ids.b, nullptr, 0);
                return nvrhi::FS_OK;
            }, &destroyed[0]);
        AddObserver(world, MakeDesc(ObserverEvent::OnMatch, ComponentId::Invalid, Access::Read,
                                    { Term(ids.a, TermOp::All), Term(ids.b, TermOp::All, Access::Read) }),
            [](ObserverEvent, const EntityBatch&, ICommandBuffer*) { return nvrhi::FS_OK; }, &destroyed[1]);
        kept = MAKE_RC_OBJ_PTR(TestObserver,
            [](ObserverEvent, const EntityBatch&, ICommandBuffer*) { return nvrhi::FS_OK; }, &destroyed[2]);
        ObserverId id = ObserverId::Invalid;
        CHECK(world->AddObserver(MakeDesc(ObserverEvent::OnRemove, ids.handle, Access::Read, {}), kept.Get(), id) ==
              nvrhi::FS_OK);
        // Neither the observers' buffers nor their filters hold the world.
        CHECK(raw->AddRef() == references);
        raw->Release();
        Payload payload;
        {
            const Handle handles[3] = { Handle(&payload), Handle(&payload), Handle(&payload) };
            const void* values[2] = { nullptr, handles };
            Create(world, CreateDesc(std::vector<ComponentId>({ ids.a, ids.handle }),
                                     std::vector<const void*>(values, values + 2), 3));
        }
        CHECK(payload.refs == 3);
        // Torn down with observers registered and entities alive: no callback, every block freed, every observer
        // released. Teardown destroys the values (their hooks run) but delivers nothing.
        world = nullptr;
        CHECK(payload.refs == 0);
    }
    CHECK(destroyed[0] && destroyed[1] && !destroyed[2]);
    kept = nullptr;
    CHECK(destroyed[2]);
    CHECK(ecs::GetHeapLiveBlockCount() == blocks);
}

// ----------------------------------------------------------------------------------------- steady state

void test_steady_state_allocations()
{
    auto world = MakeWorld(false);
    const Ids ids = Register(world);
    IWorld* raw = world.Get();
    uint64_t sink = 0;
    // Observers that allocate nothing themselves; two of them record commands (a one-round cascade, and creations
    // whose placeholders resolve in the cascade's table).
    auto add = [&](ObserverEvent event, ComponentId component, std::vector<QueryTerm> terms)
    {
        AddObserver(world, MakeDesc(event, component, Access::Read, terms),
            [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer*) {
                sink += batch.count;
                return nvrhi::FS_OK;
            });
    };
    add(ObserverEvent::OnAdd, ids.a, {});
    add(ObserverEvent::OnRemove, ids.a, { Term(ids.b, TermOp::All, Access::Read) });
    add(ObserverEvent::OnSet, ids.a, {});
    add(ObserverEvent::OnAdd, ids.s, {});
    add(ObserverEvent::OnRemove, ids.s, {});
    add(ObserverEvent::OnMatch, ComponentId::Invalid, { Term(ids.a, TermOp::All), Term(ids.c, TermOp::Without) });
    add(ObserverEvent::OnUnmatch, ComponentId::Invalid, { Term(ids.a, TermOp::All), Term(ids.c, TermOp::Without) });
    add(ObserverEvent::OnRemove, ids.tog, {});
    const CreateDesc togDesc({ ids.tog }, {}, 1);
    AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.c, Access::Read, {}),
        [&](ObserverEvent, const EntityBatch& batch, ICommandBuffer* commands) {
            commands->RemoveComponents(batch.entities, batch.count, ids.c, 0);
            return nvrhi::FS_OK;
        });
    AddObserver(world, MakeDesc(ObserverEvent::OnAdd, ids.b, Access::Read, {}),
        [&](ObserverEvent, const EntityBatch&, ICommandBuffer* commands) {
            Entity placeholder;
            commands->CreateEntities(togDesc, 0, &placeholder);
            return nvrhi::FS_OK;
        });

    auto buffer = MakeBuffer(world);
    ICommandBuffer* list[1] = { buffer.Get() };
    const QueryTerm togTerm = Term(ids.tog, TermOp::All);
    QueryDesc togQuery{};
    togQuery.terms = &togTerm;
    togQuery.termCount = 1;
    nvrhi::AutoPtr<IQuery> query;
    CHECK(world->CreateQuery(togQuery, &query) == nvrhi::FS_OK);

    std::vector<A> values(64);
    std::vector<S> sparse(64);
    for (uint32_t i = 0; i < 64; ++i)
    {
        values[i].v = i;
        sparse[i].v = i;
    }
    const CreateDesc desc({ ids.a, ids.b }, { values.data(), nullptr }, 64);
    std::vector<Entity> e(64);
    std::vector<Entity> togs(256);
    ChunkView views[16];
    WorldStats stats;

    auto round = [&]()
    {
        CHECK(world->CreateEntities(desc, e.data()) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < 16; ++i)
            CHECK(world->AddComponent(e[i], ids.c, nullptr) == nvrhi::FS_OK);
        buffer->SetComponents(e.data(), 64, ids.a, values.data(), 0);
        buffer->AddComponents(e.data(), 64, ids.s, sparse.data(), 1);
        buffer->RemoveComponents(e.data(), 64, ids.s, 2);
        buffer->DestroyEntities(e.data(), 64, 4);
        CHECK(world->Playback(list, 1) == nvrhi::FS_OK);
        // The entities the observers made, destroyed again.
        uint32_t count = 0;
        const ChunkRequest request{};
        const uint32_t chunks = query->GetChunks(0, request, 16, views);
        for (uint32_t c = 0; c < chunks; ++c)
            for (uint32_t r = 0; r < views[c].count; ++r)
                togs[count++] = views[c].entities[r];
        CHECK(count == 1);
        CHECK(world->DestroyEntities(togs.data(), count) == nvrhi::FS_OK);
        world->AdvanceVersion();
        world->GetStats(stats);
    };

    for (uint32_t r = 0; r < 3; ++r)
        round();
    const uint64_t news = g_NewCount.load();
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount();
    const uint64_t batchesBefore = stats.observerBatches;
    for (uint32_t r = 0; r < 5; ++r)
        round();
    const uint64_t newsAfter = g_NewCount.load();
    const uint64_t heapBlocksAfter = ecs::GetHeapAllocationCount();
    printf("test_ecs_observers: steady state: %llu operator new, %llu storage blocks in 5 rounds, %llu observer "
           "batches (sink %llu)\n", (unsigned long long)(newsAfter - news),
           (unsigned long long)(heapBlocksAfter - heapBlocks),
           (unsigned long long)(stats.observerBatches - batchesBefore), (unsigned long long)(sink & 1));
    CHECK(newsAfter == news);
    CHECK(heapBlocksAfter == heapBlocks);
    CHECK(stats.observerBatches > batchesBefore);
    CHECK(stats.entityCount == 0);

    // Both counters see allocations when there are some.
    int* probe = new int[16];
    sink += uintptr_t(probe) & 1;
    delete[] probe;
    CHECK(g_NewCount.load() > newsAfter);
    std::vector<Entity> wide(100000, e[0]);
    buffer->DestroyEntities(wide.data(), uint32_t(wide.size()), 0);
    CHECK(ecs::GetHeapAllocationCount() > heapBlocksAfter);
    buffer->Reset();
    (void)raw;
}

// An argument runs only the cases whose name it names.
int main(int argc, char** argv)
{
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "registration", &test_registration },
        { "events", &test_events },
        { "write_access", &test_write_access },
        { "batching", &test_batching },
        { "model", &test_model },
        { "cascades", &test_cascades },
        { "reentrancy", &test_reentrancy },
        { "snapshots", &test_snapshots },
        { "lifetime", &test_lifetime },
        { "steady_state_allocations", &test_steady_state_allocations },
    };
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    try
    {
        for (const Test& test : tests)
        {
            if (argc > 1 && std::strcmp(argv[1], test.name) != 0)
                continue;
            test.run();
            printf("test_ecs_observers: %s passed\n", test.name);
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);   // every world, observer and buffer freed every block
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
