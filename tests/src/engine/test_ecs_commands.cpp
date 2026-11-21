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

// ECS command buffers (aggregate docs/plans/2026-11-21-donut-ecs.md, step E1.3; design 3.3.6, 6.5): recording of every
// command, playback merged in sort-key order, batched moves, placeholders and their resolution, value ownership on
// every path, validation, a brute-force model that applies the same commands immediately, determinism across buffers,
// interleavings and threads, and an allocation counter that proves steady-state recording and playback allocate
// nothing. The internals are read through the private headers (src/engine/EcsCommandBufferImpl.h, EcsWorldImpl.h).

#include <donut/engine/EcsWorld.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsCommandBufferImpl.h"
#include "EcsWorldImpl.h"

#include <algorithm>
#include <atomic>
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

namespace cmd_test
{
    using donut::engine::ComponentFlags;
    using donut::engine::Entity;

    struct Position { float x, y, z; };
    DONUT_ECS_COMPONENT(Position, "97c60158-03d1-4595-9e36-41119bda9a4e", DONUT_ECS_FIELDS(x, y, z));

    struct Velocity { float x, y, z; };
    DONUT_ECS_COMPONENT(Velocity, "565916b0-b991-42de-9cb8-8d71764a002e", DONUT_ECS_FIELDS(x, y, z));

    struct Id { uint64_t value; };
    DONUT_ECS_COMPONENT(Id, "caab978c-3217-49b5-85a1-d0a561216400", DONUT_ECS_FIELDS(value));

    struct Link { Entity other; Entity outside; };
    DONUT_ECS_COMPONENT(Link, "e10b19ee-cd12-4616-aced-63b3e270b274", DONUT_ECS_FIELDS(other, outside));

    // Three bytes of padding after `id`: never part of the key.
    struct Layer { uint8_t id; uint32_t mask; };
    DONUT_ECS_COMPONENT_EX(Layer, "534f267e-ac45-47db-9136-8830cfa65e11", ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(id, mask));

    struct Toggle { int32_t value; };
    DONUT_ECS_COMPONENT_EX(Toggle, "ed26b0ed-2e8d-4cbb-b551-39df3f1ec088", ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(value));

    struct Hidden { };
    DONUT_ECS_TAG_EX(Hidden, "a66cabfc-aa68-4990-a7eb-98dc1b34cd02", ComponentFlags::Enableable);

    struct Frozen { };
    DONUT_ECS_TAG(Frozen, "9907326a-1dde-4bc7-8086-502cfc1eb602");

    struct Selected { };
    DONUT_ECS_TAG_EX(Selected, "547ad487-b01b-4351-8a9f-6dc2809b09ca", ComponentFlags::Sparse);

    struct DebugColor { uint32_t rgba; };
    DONUT_ECS_COMPONENT_EX(DebugColor, "288d00d2-5f03-47a2-a8a3-3c8a238f46ed", ComponentFlags::Sparse, 0,
                           DONUT_ECS_FIELDS(rgba));

    struct Joint { Entity target; float weight; uint32_t pad; };
    DONUT_ECS_BUFFER(Joint, "4da6615e-7825-43ea-b96c-32f8136539a3", 4, DONUT_ECS_FIELDS(target, weight));

    struct ChunkBounds { float lo, hi; };
    DONUT_ECS_COMPONENT_EX(ChunkBounds, "7a513f1b-7d3f-422c-838e-2dbecf684ee0", ComponentFlags::Chunk, 0,
                           DONUT_ECS_FIELDS(lo, hi));

    struct Gravity { float g; };
    DONUT_ECS_COMPONENT_EX(Gravity, "bd35f27f-5ca1-4ff8-afa9-222465fd8887", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(g));

    // Non-trivial values that hold a reference to a counted payload; every hook call is counted. Atomic, because
    // threads record such values in the determinism case.
    struct HookCounts
    {
        std::atomic<int64_t> constructs{ 0 };
        std::atomic<int64_t> destructs{ 0 };
        std::atomic<int64_t> copies{ 0 };
        std::atomic<int64_t> moves{ 0 };
        int64_t Live() const { return constructs + copies + moves - destructs; }
    };
    HookCounts g_Hooks;

    struct Payload { std::atomic<int64_t> refs{ 0 }; };

    struct Handle
    {
        Payload* payload = nullptr;

        Handle() noexcept { ++g_Hooks.constructs; }
        explicit Handle(Payload* p) noexcept : payload(p)
        {
            ++g_Hooks.constructs;
            if (payload)
                ++payload->refs;
        }
        Handle(const Handle& other) noexcept : payload(other.payload)
        {
            ++g_Hooks.copies;
            if (payload)
                ++payload->refs;
        }
        Handle(Handle&& other) noexcept : payload(other.payload)
        {
            ++g_Hooks.moves;
            other.payload = nullptr;
        }
        ~Handle()
        {
            ++g_Hooks.destructs;
            if (payload)
                --payload->refs;
        }
        Handle& operator=(const Handle&) = delete;
        Handle& operator=(Handle&&) = delete;
    };
    DONUT_ECS_COMPONENT(Handle, "80073bb3-5cb9-40c5-b774-1611fbc80882", DONUT_ECS_FIELDS());

    // The same as a Buffer element: two inline, then on the heap.
    struct RefElement
    {
        Payload* payload = nullptr;

        RefElement() noexcept { ++g_Hooks.constructs; }
        explicit RefElement(Payload* p) noexcept : payload(p)
        {
            ++g_Hooks.constructs;
            if (payload)
                ++payload->refs;
        }
        RefElement(const RefElement& other) noexcept : payload(other.payload)
        {
            ++g_Hooks.copies;
            if (payload)
                ++payload->refs;
        }
        RefElement(RefElement&& other) noexcept : payload(other.payload)
        {
            ++g_Hooks.moves;
            other.payload = nullptr;
        }
        ~RefElement()
        {
            ++g_Hooks.destructs;
            if (payload)
                --payload->refs;
        }
        RefElement& operator=(const RefElement&) = delete;
        RefElement& operator=(RefElement&&) = delete;
    };
    DONUT_ECS_BUFFER(RefElement, "e978c02e-55c7-4943-bf98-e9acbf80bb2b", 2, DONUT_ECS_FIELDS());

    // Movable, not copyable: relocate hook, no copy hook.
    struct MoveOnly
    {
        int32_t value = 0;
        MoveOnly() = default;
        explicit MoveOnly(int32_t v) : value(v) { }
        MoveOnly(MoveOnly&& other) noexcept : value(other.value) { other.value = -1; }
        MoveOnly(const MoveOnly&) = delete;
        MoveOnly& operator=(const MoveOnly&) = delete;
        MoveOnly& operator=(MoveOnly&&) = delete;
        ~MoveOnly() { value = -2; }
    };
    DONUT_ECS_COMPONENT(MoveOnly, "4afd62fe-668d-489b-8a1a-a1a0d76edfcc", DONUT_ECS_FIELDS(value));
}

using namespace donut::engine;
using namespace cmd_test;
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
    };

    struct Ids
    {
        ComponentId position, velocity, id, link, layer, toggle, hidden, frozen, selected, color, joint, bounds,
            gravity, handle, refs, moveOnly;
    };

    Ids Register(IWorld* world)
    {
        Ids ids{};
        ids.position = RegisterComponent<Position>(world);
        ids.velocity = RegisterComponent<Velocity>(world);
        ids.id = RegisterComponent<Id>(world);
        ids.link = RegisterComponent<Link>(world);
        ids.layer = RegisterComponent<Layer>(world);
        ids.toggle = RegisterComponent<Toggle>(world);
        ids.hidden = RegisterComponent<Hidden>(world);
        ids.frozen = RegisterComponent<Frozen>(world);
        ids.selected = RegisterComponent<Selected>(world);
        ids.color = RegisterComponent<DebugColor>(world);
        ids.joint = RegisterComponent<Joint>(world);
        ids.bounds = RegisterComponent<ChunkBounds>(world);
        ids.gravity = RegisterComponent<Gravity>(world);
        ids.handle = RegisterComponent<Handle>(world);
        ids.refs = RegisterComponent<RefElement>(world);
        ids.moveOnly = RegisterComponent<MoveOnly>(world);
        CHECK(ids.moveOnly != ComponentId::Invalid && ids.gravity != ComponentId::Invalid);
        return ids;
    }

    nvrhi::AutoPtr<IWorld> MakeWorld(bool validation = false, uint32_t chunkSize = 16 * 1024)
    {
        WorldDesc desc;
        desc.debugName = "test_ecs_commands";
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

    ecs::CommandPlayback& State(IWorld* world) { return Impl(world).GetCommandPlayback(); }

    ecs::CommandBufferImpl& BufferImpl(ICommandBuffer* buffer)
    {
        ecs::CommandBufferImpl* impl = donut::query_cast<ecs::CommandBufferImpl>(buffer);
        CHECK(impl != nullptr);
        return *impl;
    }

    uint64_t Errors(IWorld* world) { return Impl(world).GetValidationErrorCount(); }

    template <typename T>
    const T& Get(IWorld* world, Entity entity)
    {
        const T* value = static_cast<const T*>(world->GetComponent(entity, FindComponent<T>(world)));
        CHECK(value != nullptr);
        return *value;
    }

    const BufferHeader& HeaderOf(IWorld* world, Entity entity, ComponentId buffer)
    {
        const BufferHeader* header = static_cast<const BufferHeader*>(world->GetComponent(entity, buffer));
        CHECK(header != nullptr);
        return *header;
    }

    const ecs::Archetype& ArchetypeOf(IWorld* world, Entity entity)
    {
        const ecs::EntityRecord* record = Impl(world).FindAlive(entity);
        CHECK(record != nullptr);
        return Impl(world).GetArchetypeStorage(record->archetype);
    }

    // An EntityCreateDesc that owns its component and value arrays.
    class CreateDesc
    {
    public:
        CreateDesc(std::initializer_list<ComponentId> components, std::initializer_list<const void*> values,
                   uint32_t count, ValueTransfer transfer = ValueTransfer::Copy)
        {
            CHECK(components.size() <= 16 && (values.size() == 0 || values.size() == components.size()));
            std::copy(components.begin(), components.end(), m_Components);
            std::copy(values.begin(), values.end(), m_Values);
            m_Desc.components = m_Components;
            m_Desc.componentCount = uint32_t(components.size());
            m_Desc.values = values.size() ? m_Values : nullptr;
            m_Desc.entityCount = count;
            m_Desc.transfer = transfer;
        }
        CreateDesc(const CreateDesc&) = delete;
        CreateDesc& operator=(const CreateDesc&) = delete;
        operator const EntityCreateDesc&() const { return m_Desc; }
        EntityCreateDesc* operator->() { return &m_Desc; }

    private:
        ComponentId m_Components[16] = {};
        const void* m_Values[16] = {};
        EntityCreateDesc m_Desc{};
    };

    nvrhi::FRESULT Play(IWorld* world, std::initializer_list<ICommandBuffer*> buffers)
    {
        ICommandBuffer* list[16] = {};
        CHECK(buffers.size() <= 16);
        std::copy(buffers.begin(), buffers.end(), list);
        return world->Playback(list, uint32_t(buffers.size()));
    }

    // Every entity of the table with its record (archetype, chunk, row: the placement) and, for live ones, every
    // component's presence, enabled bit and field bytes (buffers element by element; the payload of a counted value).
    std::vector<uint64_t> Dump(IWorld* world, const Ids& ids)
    {
        ecs::WorldImpl& impl = Impl(world);
        std::vector<uint64_t> out;
        const uint32_t size = impl.GetEntityTableSize();
        out.push_back(size);
        const uint32_t types = world->GetComponentTypeCount();
        auto addValue = [&](const ecs::ComponentType& type, ComponentId component, const uint8_t* bytes)
        {
            if (component == ids.handle)
                out.push_back(uint64_t(uintptr_t(reinterpret_cast<const Handle*>(bytes)->payload)));
            else if (component == ids.refs)
                out.push_back(uint64_t(uintptr_t(reinterpret_cast<const RefElement*>(bytes)->payload)));
            else
            {
                Fnv fnv;
                for (const ecs::ByteRange& range : type.fieldBytes)
                    fnv.Add(bytes + range.offset, range.size);
                out.push_back(fnv.hash);
            }
        };
        for (uint32_t index = 1; index < size; ++index)
        {
            const ecs::EntityRecord& record = impl.GetEntityRecord(index);
            out.push_back(record.generation);
            out.push_back(record.archetype);
            out.push_back(record.chunk);
            if (record.archetype >= ecs::c_ClaimedSlot)
                continue;
            out.push_back(record.row);
            const Entity entity = MakeEntity(index, record.generation);
            for (uint32_t t = 0; t < types; ++t)
            {
                const ComponentId component = ComponentId(t);
                const ecs::ComponentType& type = *impl.GetType(component);
                if (type.kind == ecs::StorageKind::Resource)
                    continue;
                const bool has = world->HasComponent(entity, component);
                out.push_back(has ? 1 : 0);
                if (!has)
                    continue;
                out.push_back(world->IsEnabled(entity, component) ? 1 : 0);
                const uint8_t* value = static_cast<const uint8_t*>(world->GetComponent(entity, component));
                if (type.kind == ecs::StorageKind::Buffer)
                {
                    const BufferHeader& header = *reinterpret_cast<const BufferHeader*>(value);
                    out.push_back(header.count);
                    const uint8_t* elements = static_cast<const uint8_t*>(header.data);
                    for (uint32_t k = 0; k < header.count; ++k)
                        addValue(type, component, elements + size_t(k) * type.desc.size);
                }
                else if (type.desc.size != 0)
                {
                    addValue(type, component, value);
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------------- generated command sequences

    enum class Op : uint8_t { Create, Destroy, Add, Remove, Set, SetEnabled, Instantiate, Append };

    // An entity a command names: entity `index` of creating command `command` of the same round (a placeholder when
    // recorded), or a plain id (alive, stale, null, or a placeholder of an earlier round).
    struct Ref
    {
        int32_t command = -1;
        uint32_t index = 0;
        Entity entity = Entity::Null;
    };

    // A value before it is written as a component (or Buffer element).
    struct Spec
    {
        float f[3] = {};
        uint64_t number = 0;
        Ref a;
        Ref b;
        uint32_t payload = 0;   // index in the payload pool; c_NoPayload for none
    };
    constexpr uint32_t c_PayloadCount = 3;
    constexpr uint32_t c_NoPayload = c_PayloadCount;

    struct Logical
    {
        Op op = Op::Create;
        uint64_t key = 0;
        uint32_t buffer = 0;
        ComponentId component = ComponentId::Invalid;
        bool enabled = false;
        bool values = false;                        // Add: with values
        ValueTransfer transfer = ValueTransfer::Copy;   // Create
        uint32_t count = 0;                         // Create, Instantiate: entities; Append: elements
        std::vector<Ref> targets;                   // the entity list; Instantiate: the prefab; Append: the entity
        std::vector<ComponentId> components;        // Create
        std::vector<std::vector<Spec>> specs;       // Create: per component (empty: defaults); Add, Set, Append: [0]
        std::vector<uint32_t> dependencies;         // creating commands whose placeholders it names
    };

    bool Creates(const Logical& command) { return command.op == Op::Create || command.op == Op::Instantiate; }

    // Constructs the value `spec` describes in raw storage; `entityOf` maps the refs.
    template <typename EntityOf>
    void ConstructValue(const Ids& ids, ComponentId c, void* dst, const Spec& s, Payload* payloads, EntityOf&& entityOf)
    {
        Payload* payload = s.payload < c_PayloadCount && payloads ? &payloads[s.payload] : nullptr;
        if (c == ids.position)
            new (dst) Position{ s.f[0], s.f[1], s.f[2] };
        else if (c == ids.velocity)
            new (dst) Velocity{ s.f[1], s.f[2], s.f[0] };
        else if (c == ids.id)
            new (dst) Id{ s.number };
        else if (c == ids.link)
            new (dst) Link{ entityOf(s.a), entityOf(s.b) };
        else if (c == ids.layer)
        {
            Layer layer;
            std::memset(&layer, 0, sizeof(layer));
            layer.id = uint8_t(1 + s.number % 3);
            layer.mask = uint32_t(s.number >> 8) & 3;
            std::memcpy(dst, &layer, sizeof(layer));
        }
        else if (c == ids.toggle)
            new (dst) Toggle{ int32_t(s.number & 0xffff) };
        else if (c == ids.color)
            new (dst) DebugColor{ uint32_t(s.number) };
        else if (c == ids.joint)
            new (dst) Joint{ entityOf(s.a), s.f[0], 0 };
        else if (c == ids.handle)
            new (dst) Handle(payload);
        else if (c == ids.refs)
            new (dst) RefElement(payload);
        else if (c == ids.moveOnly)
            new (dst) MoveOnly(int32_t(s.number & 0xffff));
    }

    void DestroyValue(const Ids& ids, ComponentId c, void* dst)
    {
        if (c == ids.handle)
            static_cast<Handle*>(dst)->~Handle();
        else if (c == ids.refs)
            static_cast<RefElement*>(dst)->~RefElement();
        else if (c == ids.moveOnly)
            static_cast<MoveOnly*>(dst)->~MoveOnly();
    }

    // Raw storage for `count` values of one component, built from specs and destroyed with the block unless given up
    // (ValueTransfer::Relocate: the callee owns them).
    class ValueBlock
    {
    public:
        template <typename EntityOf>
        ValueBlock(IWorld* world, const Ids& ids, ComponentId component, const std::vector<Spec>& specs,
                   Payload* payloads, EntityOf&& entityOf)
            : m_Ids(ids)
            , m_Component(component)
            , m_Size(world->GetComponentTypeDesc(component)->size)
            , m_Storage((specs.size() * m_Size + 7) / 8 + 1)
        {
            for (const Spec& spec : specs)
                ConstructValue(ids, component, At(m_Count++), spec, payloads, entityOf);
        }
        ValueBlock(const ValueBlock&) = delete;
        ValueBlock& operator=(const ValueBlock&) = delete;
        ~ValueBlock()
        {
            for (uint32_t i = 0; i < m_Count; ++i)
                DestroyValue(m_Ids, m_Component, At(i));
        }
        void* Data() { return m_Storage.data(); }
        void GiveUp() { m_Count = 0; }

    private:
        void* At(uint32_t i) { return reinterpret_cast<uint8_t*>(m_Storage.data()) + size_t(i) * m_Size; }

        const Ids& m_Ids;
        ComponentId m_Component;
        uint32_t m_Size;
        uint32_t m_Count = 0;
        std::vector<uint64_t> m_Storage;
    };

    // Seeded random rounds of commands over a set of known ids.
    //
    // A dirty round names anything: ids from before the round (alive or not), placeholders of earlier creating
    // commands (in recording order, so perhaps later in key order), null and stale ids. A clean round names only
    // `known` ids, which it keeps alive with Position, Id, Toggle and Joint, and placeholders of earlier keys; it
    // destroys only `victim` creations that nothing else names. So a clean round succeeds, apart from the `poison`
    // commands it may add, which fail with known codes: a value that cannot be copied (FE_UNSUPPORTED when recorded), a
    // stale id (FE_INVALID_ARGS), an Instantiate of a prefab that cannot be copied (FE_UNSUPPORTED when played back).
    struct Generator
    {
        Xorshift rng;
        const Ids& ids;
        bool payloads = true;           // Handle and RefElement values
        uint32_t bufferCount = 1;
        bool clean = false;
        bool clash = false;             // a repeated key may go to another buffer: a validation failure
        std::vector<Entity> known;
        Entity movable = Entity::Null;  // clean rounds: an entity with MoveOnly, for the poison Instantiate
        std::vector<Logical> commands;
        std::vector<bool> victim;                   // per command: a clean creation that only Destroy names
        std::vector<bool> poison;                   // per command: added to fail
        std::vector<std::vector<bool>> destroyed;   // per command and entity: a victim already destroyed

        Ref PickRef(uint64_t key)
        {
            std::vector<uint32_t> creating;
            for (uint32_t i = 0; i < commands.size(); ++i)
                if (Creates(commands[i]) && (!clean || (!victim[i] && !poison[i] && commands[i].key < key)))
                    creating.push_back(i);
            const uint32_t r = rng.Below(100);
            if (r < 45 && !creating.empty())
            {
                const uint32_t j = creating[rng.Below(uint32_t(creating.size()))];
                Ref ref;
                ref.command = int32_t(j);
                ref.index = rng.Below(commands[j].count);
                return ref;
            }
            Ref ref;
            if (clean || (r < 95 && !known.empty()))
                ref.entity = known[rng.Below(uint32_t(known.size()))];
            else if (r < 97)
                ref.entity = Entity::Null;
            else
                ref.entity = MakeEntity(1 + rng.Below(80), 77777);   // a generation no index reaches here
            return ref;
        }

        Ref PickKnown()
        {
            Ref ref;
            ref.entity = known[rng.Below(uint32_t(known.size()))];
            return ref;
        }

        Spec PickSpec(uint64_t key)
        {
            Spec spec;
            for (float& f : spec.f)
                f = float(int32_t(rng.Below(200)) - 100);
            spec.number = rng.Next();
            spec.a = PickRef(key);
            spec.b = PickRef(key);
            spec.payload = payloads ? rng.Below(c_PayloadCount + 1) : c_NoPayload;
            return spec;
        }

        bool TakesValues(ComponentId c) const
        {
            return c != ids.hidden && c != ids.frozen && c != ids.selected && c != ids.joint && c != ids.bounds &&
                   c != ids.refs;
        }

        ComponentId Pick(std::initializer_list<ComponentId> choices)
        {
            std::vector<ComponentId> list;
            for (ComponentId c : choices)
                if (payloads || (c != ids.handle && c != ids.refs))
                    list.push_back(c);
            return list[rng.Below(uint32_t(list.size()))];
        }

        // Clean rounds: up to two victim entities created at earlier keys and not destroyed yet.
        bool PickVictims(Logical& command)
        {
            std::vector<Ref> candidates;
            for (uint32_t j = 0; j < commands.size(); ++j)
            {
                if (!victim[j] || commands[j].key >= command.key)
                    continue;
                for (uint32_t t = 0; t < commands[j].count; ++t)
                {
                    if (destroyed[j][t])
                        continue;
                    Ref ref;
                    ref.command = int32_t(j);
                    ref.index = t;
                    candidates.push_back(ref);
                }
            }
            if (candidates.empty())
                return false;
            const uint32_t n = 1 + rng.Below(candidates.size() > 1 ? 2 : 1);
            for (uint32_t k = 0; k < n; ++k)
            {
                const uint32_t pick = k + rng.Below(uint32_t(candidates.size()) - k);
                std::swap(candidates[k], candidates[pick]);
                destroyed[candidates[k].command][candidates[k].index] = true;
                command.targets.push_back(candidates[k]);
            }
            return true;
        }

        void Fill(Logical& command)
        {
            command.targets.clear();
            command.specs.clear();
            command.components.clear();
            const uint64_t key = command.key;
            if (clean && command.op == Op::Destroy && !PickVictims(command))
            {
                command.op = Op::Add;   // nothing to destroy yet
                command.component = ids.frozen;
                command.values = false;
            }
            switch (command.op)
            {
            case Op::Create:
            {
                command.count = 1 + rng.Below(6);
                command.transfer = rng.Below(100) < 40 ? ValueTransfer::Relocate : ValueTransfer::Copy;
                const ComponentId all[] = { ids.position, ids.velocity, ids.id, ids.link, ids.layer, ids.toggle,
                                            ids.hidden, ids.frozen, ids.selected, ids.color, ids.joint, ids.handle,
                                            ids.bounds, ids.refs };
                for (ComponentId c : all)
                {
                    if (!payloads && (c == ids.handle || c == ids.refs))
                        continue;
                    if (rng.Below(100) >= 35)
                        continue;
                    command.components.push_back(c);
                    std::vector<Spec> specs;
                    if (TakesValues(c) && rng.Below(100) < 60)
                        for (uint32_t t = 0; t < command.count; ++t)
                            specs.push_back(PickSpec(key));
                    command.specs.push_back(specs);
                }
                break;
            }
            case Op::Instantiate:
                command.count = 1 + rng.Below(3);
                command.targets.push_back(PickRef(key));
                break;
            case Op::Append:
            {
                command.count = 1 + rng.Below(6);
                command.targets.push_back(clean ? PickKnown() : PickRef(key));
                std::vector<Spec> specs;
                for (uint32_t t = 0; t < command.count; ++t)
                    specs.push_back(PickSpec(key));
                command.specs.push_back(specs);
                break;
            }
            default:
            {
                if (!(clean && command.op == Op::Destroy))
                {
                    // Clean rounds set and toggle only what the known ids keep.
                    const bool knownOnly = clean && (command.op == Op::Set || command.op == Op::SetEnabled);
                    const uint32_t count = 1 + rng.Below(5);
                    for (uint32_t t = 0; t < count; ++t)
                        command.targets.push_back(knownOnly ? PickKnown() : PickRef(key));
                }
                const uint32_t n = uint32_t(command.targets.size());
                const bool sized = command.op == Op::Set ? TakesValues(command.component) :
                                   command.op == Op::Add && command.values;
                if (sized)
                {
                    std::vector<Spec> specs;
                    for (uint32_t t = 0; t < n; ++t)
                        specs.push_back(PickSpec(key));
                    command.specs.push_back(specs);
                }
                break;
            }
            }
            // The creating commands it names, for recording orders that must respect them.
            command.dependencies.clear();
            auto note = [&](const Ref& ref)
            {
                if (ref.command >= 0)
                    command.dependencies.push_back(uint32_t(ref.command));
            };
            for (const Ref& ref : command.targets)
                note(ref);
            for (const std::vector<Spec>& specs : command.specs)
                for (const Spec& spec : specs)
                {
                    note(spec.a);
                    note(spec.b);
                }
        }

        void Push(Logical& command, bool isVictim, bool isPoison, std::map<uint64_t, uint32_t>& bufferOfKey)
        {
            auto found = bufferOfKey.find(command.key);
            if (found == bufferOfKey.end())
                found = bufferOfKey.emplace(command.key, rng.Below(bufferCount)).first;
            command.buffer = found->second;
            if (clash && bufferCount > 1 && rng.Below(3) == 0)
                command.buffer = rng.Below(bufferCount);   // perhaps not the buffer of the key's first command
            victim.push_back(isVictim);
            poison.push_back(isPoison);
            destroyed.emplace_back(Creates(command) ? command.count : 0u, false);
            commands.push_back(command);
        }

        // A round of `n` commands. Some continue the previous command's operation at the next keys (a system's loop
        // over chunks, recorded by several workers); some repeat a key, which then stays in one buffer unless `clash`.
        void Generate(uint32_t n)
        {
            commands.clear();
            victim.clear();
            poison.clear();
            destroyed.clear();
            std::map<uint64_t, uint32_t> bufferOfKey;
            for (uint32_t i = 0; i < n; ++i)
            {
                Logical command;
                const bool group = i > 0 && rng.Below(100) < 45 && commands.back().op != Op::Create &&
                                   commands.back().op != Op::Instantiate;
                if (group)
                {
                    const Logical& previous = commands.back();
                    command.op = previous.op;
                    command.component = previous.component;
                    command.enabled = previous.enabled;
                    command.values = previous.values;
                    command.key = previous.key + (rng.Below(100) < 10 ? 0 : 1 + rng.Below(2));
                }
                else
                {
                    const uint32_t r = rng.Below(100);
                    command.op = r < 18 ? Op::Create : r < 30 ? Op::Destroy : r < 48 ? Op::Add : r < 62 ? Op::Remove :
                                 r < 76 ? Op::Set : r < 84 ? Op::SetEnabled : r < 92 ? Op::Instantiate : Op::Append;
                    switch (command.op)
                    {
                    case Op::Add:
                        command.component = Pick({ ids.velocity, ids.id, ids.link, ids.layer, ids.toggle, ids.hidden,
                                                   ids.frozen, ids.selected, ids.color, ids.joint, ids.handle,
                                                   ids.bounds, ids.refs });
                        command.values = TakesValues(command.component) && rng.Below(100) < 60;
                        break;
                    case Op::Remove:
                        command.component = clean ?
                            Pick({ ids.velocity, ids.link, ids.layer, ids.hidden, ids.frozen, ids.selected, ids.color,
                                   ids.handle, ids.bounds, ids.refs }) :
                            Pick({ ids.position, ids.velocity, ids.id, ids.link, ids.layer, ids.toggle, ids.hidden,
                                   ids.frozen, ids.selected, ids.color, ids.joint, ids.handle, ids.bounds, ids.refs });
                        break;
                    case Op::Set:
                        command.component = clean ? Pick({ ids.position, ids.id, ids.toggle }) :
                            Pick({ ids.position, ids.velocity, ids.id, ids.link, ids.layer, ids.toggle, ids.color,
                                   ids.handle, ids.frozen, ids.selected });
                        break;
                    case Op::SetEnabled:
                        command.component = clean || rng.Below(2) ? ids.toggle : ids.hidden;
                        command.enabled = rng.Below(2) != 0;
                        break;
                    case Op::Append:
                        command.component = !clean && payloads && rng.Below(3) == 0 ? ids.refs : ids.joint;
                        break;
                    default:
                        break;
                    }
                    command.key = MakeSortKey(rng.Below(3), rng.Below(40), rng.Below(64));
                }
                Fill(command);
                const bool isVictim = clean && command.op == Op::Create && rng.Below(100) < 30;
                Push(command, isVictim, false, bufferOfKey);
            }
            if (!clean || rng.Below(3) == 0)
                return;
            // Poison: commands that fail with known codes, at random keys; nothing names them.
            const uint32_t count = 1 + rng.Below(3);
            for (uint32_t i = 0; i < count; ++i)
            {
                Logical command;
                command.key = MakeSortKey(rng.Below(3), rng.Below(40), rng.Below(64));
                const uint32_t kind = rng.Below(3);
                if (kind == 0)
                {
                    command.op = Op::Add;   // MoveOnly has no copy hook: refused when recorded
                    command.component = ids.moveOnly;
                    command.values = true;
                    command.targets.push_back(PickKnown());
                    command.specs.push_back(std::vector<Spec>(1, Spec{}));
                }
                else if (kind == 1)
                {
                    command.op = Op::Destroy;
                    Ref stale;
                    stale.entity = MakeEntity(1 + rng.Below(80), 77777);
                    command.targets.push_back(stale);
                }
                else
                {
                    command.op = Op::Instantiate;   // the prefab's MoveOnly cannot be copied
                    command.count = 1;
                    Ref prefab;
                    prefab.entity = movable;
                    command.targets.push_back(prefab);
                }
                Push(command, false, true, bufferOfKey);
            }
        }
    };

    // Records generated commands; placeholders[i] receives what command i handed out.
    struct Recorder
    {
        IWorld* world;
        const Ids& ids;
        Payload* payloads;
        std::vector<std::vector<Entity>>& placeholders;

        Entity Named(const Ref& ref) const
        {
            return ref.command >= 0 ? placeholders[ref.command][ref.index] : ref.entity;
        }

        void Record(ICommandBuffer* buffer, const Logical& command, uint32_t index)
        {
            auto named = [this](const Ref& ref) { return Named(ref); };
            std::vector<Entity> list;
            for (const Ref& ref : command.targets)
                list.push_back(Named(ref));
            switch (command.op)
            {
            case Op::Create:
            {
                std::vector<std::unique_ptr<ValueBlock>> blocks;
                std::vector<const void*> values(command.components.size(), nullptr);
                for (size_t i = 0; i < command.components.size(); ++i)
                {
                    if (command.specs[i].empty())
                        continue;
                    blocks.emplace_back(new ValueBlock(world, ids, command.components[i], command.specs[i], payloads,
                                                       named));
                    values[i] = blocks.back()->Data();
                }
                EntityCreateDesc desc{};
                desc.components = command.components.data();
                desc.componentCount = uint32_t(command.components.size());
                desc.values = values.data();
                desc.entityCount = command.count;
                desc.transfer = command.transfer;
                placeholders[index].assign(command.count, Entity::Null);
                buffer->CreateEntities(desc, command.key, placeholders[index].data());
                if (command.transfer == ValueTransfer::Relocate)
                    for (auto& block : blocks)
                        block->GiveUp();
                break;
            }
            case Op::Destroy:
                buffer->DestroyEntities(list.data(), uint32_t(list.size()), command.key);
                break;
            case Op::Add:
            case Op::Set:
            {
                std::unique_ptr<ValueBlock> block;
                if (!command.specs.empty())
                    block.reset(new ValueBlock(world, ids, command.component, command.specs[0], payloads, named));
                if (command.op == Op::Add)
                    buffer->AddComponents(list.data(), uint32_t(list.size()), command.component,
                                          block ? block->Data() : nullptr, command.key);
                else
                    buffer->SetComponents(list.data(), uint32_t(list.size()), command.component,
                                          block ? block->Data() : nullptr, command.key);
                break;
            }
            case Op::Remove:
                buffer->RemoveComponents(list.data(), uint32_t(list.size()), command.component, command.key);
                break;
            case Op::SetEnabled:
                buffer->SetEnabled(list.data(), uint32_t(list.size()), command.component, command.enabled,
                                   command.key);
                break;
            case Op::Instantiate:
                placeholders[index].assign(command.count, Entity::Null);
                buffer->Instantiate(list[0], command.count, command.key, placeholders[index].data());
                break;
            case Op::Append:
            {
                ValueBlock block(world, ids, command.component, command.specs[0], payloads, named);
                buffer->AppendToBuffer(list[0], command.component, block.Data(), command.count, command.key);
                break;
            }
            }
        }
    };

    // The same commands applied immediately, one by one in playback order, through the public immediate operations
    // (the batch operations where IWorld has no immediate form: SetComponents, AppendToBuffer).
    struct Model
    {
        IWorld* world;
        const Ids& ids;
        Payload* payloads;
        const std::vector<std::vector<Entity>>& placeholders;   // what recording handed out
        std::vector<std::vector<Entity>> created;
        std::vector<bool> applied;
        nvrhi::FRESULT result = nvrhi::FS_OK;

        void Keep(nvrhi::FRESULT next)
        {
            if (NVRHI_SUCCEEDED(result) && NVRHI_FAILED(next))
                result = next;
        }

        // What playback names: the entity a placeholder became, or the placeholder itself (stale) before its command
        // ran or when it failed.
        Entity Target(const Ref& ref) const
        {
            if (ref.command < 0)
                return ref.entity;
            if (applied[ref.command] && created[ref.command][ref.index] != Entity::Null)
                return created[ref.command][ref.index];
            return placeholders[ref.command][ref.index];
        }

        void Apply(const Logical& command, uint32_t index)
        {
            bool unresolved = false;
            auto field = [&](const Ref& ref)
            {
                const Entity entity = Target(ref);
                if (!IsPlaceholder(entity))
                    return entity;
                unresolved = true;
                return Entity::Null;
            };
            std::vector<Entity> list;
            for (const Ref& ref : command.targets)
                list.push_back(Target(ref));
            ecs::WorldImpl& impl = Impl(world);
            switch (command.op)
            {
            case Op::Create:
            {
                std::vector<std::unique_ptr<ValueBlock>> blocks;
                std::vector<const void*> values(command.components.size(), nullptr);
                for (size_t i = 0; i < command.components.size(); ++i)
                {
                    if (command.specs[i].empty())
                        continue;
                    blocks.emplace_back(new ValueBlock(world, ids, command.components[i], command.specs[i], payloads,
                                                       field));
                    values[i] = blocks.back()->Data();
                }
                if (unresolved)
                    Keep(nvrhi::FE_INVALID_ARGS);
                EntityCreateDesc desc{};
                desc.components = command.components.data();
                desc.componentCount = uint32_t(command.components.size());
                desc.values = values.data();
                desc.entityCount = command.count;
                desc.transfer = ValueTransfer::Copy;
                created[index].assign(command.count, Entity::Null);
                Keep(world->CreateEntities(desc, created[index].data()));
                break;
            }
            case Op::Destroy:
                Keep(world->DestroyEntities(list.data(), uint32_t(list.size())));
                break;
            case Op::Add:
            case Op::Set:
            {
                std::unique_ptr<ValueBlock> block;
                const uint32_t size = world->GetComponentTypeDesc(command.component)->size;
                if (!command.specs.empty())
                    block.reset(new ValueBlock(world, ids, command.component, command.specs[0], payloads, field));
                if (unresolved)
                    Keep(nvrhi::FE_INVALID_ARGS);
                for (size_t t = 0; t < list.size(); ++t)
                {
                    const void* value = block ? static_cast<const uint8_t*>(block->Data()) + t * size : nullptr;
                    if (command.op == Op::Add)
                        Keep(world->AddComponent(list[t], command.component, value));
                    else
                        Keep(impl.SetComponentsInternal(&list[t], 1, command.component, value, ValueTransfer::Copy,
                                                        world->GetVersion()));
                }
                break;
            }
            case Op::Remove:
                for (const Entity entity : list)
                    Keep(world->RemoveComponent(entity, command.component));
                break;
            case Op::SetEnabled:
                for (const Entity entity : list)
                    Keep(world->SetEnabled(entity, command.component, command.enabled));
                break;
            case Op::Instantiate:
                created[index].assign(command.count, Entity::Null);
                Keep(world->Instantiate(list[0], command.count, created[index].data()));
                break;
            case Op::Append:
            {
                ValueBlock block(world, ids, command.component, command.specs[0], payloads, field);
                if (unresolved)
                    Keep(nvrhi::FE_INVALID_ARGS);
                Keep(impl.AppendToBufferInternal(list[0], command.component, block.Data(), command.count,
                                                 ValueTransfer::Copy, world->GetVersion()));
                break;
            }
            }
            applied[index] = true;
        }
    };

    // Playback order: by key, then buffer position, then recording order (generation order within a buffer).
    std::vector<uint32_t> PlaybackOrder(const std::vector<Logical>& commands)
    {
        std::vector<uint32_t> order(commands.size());
        for (uint32_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b)
        {
            if (commands[a].key != commands[b].key)
                return commands[a].key < commands[b].key;
            return commands[a].buffer < commands[b].buffer;
        });
        return order;
    }

    // The same starting entities in every world of a comparison.
    // Every entity has Position, Id, Toggle and Joint (what clean rounds set, toggle and append to); some disabled,
    // some with a Handle.
    std::vector<Entity> Populate(IWorld* world, const Ids& ids, Xorshift rng, Payload* payloads, uint32_t count)
    {
        std::vector<Entity> entities;
        for (uint32_t i = 0; i < count; ++i)
        {
            Entity entity = Entity::Null;
            const Position position{ float(i), float(rng.Below(10)), 0 };
            const Id id{ rng.Next() };
            const CreateDesc desc({ ids.position, ids.id, ids.toggle, ids.joint }, { &position, &id, nullptr, nullptr },
                                  1);
            CHECK(world->CreateEntities(desc, &entity) == nvrhi::FS_OK);
            if (rng.Below(3) == 0)
                CHECK(world->SetEnabled(entity, ids.toggle, false) == nvrhi::FS_OK);
            if (payloads && rng.Below(4) == 0)
            {
                const Handle handle(&payloads[rng.Below(c_PayloadCount)]);
                CHECK(world->AddComponent(entity, ids.handle, &handle) == nvrhi::FS_OK);
            }
            entities.push_back(entity);
        }
        return entities;
    }
}

// ------------------------------------------------------------------------------------------------- basics

void test_basics()
{
    auto world = MakeWorld(true);
    const Ids ids = Register(world);
    CHECK(world->CreateCommandBuffer(nullptr) == nvrhi::FE_INVALID_ARGS);
    auto a = MakeBuffer(world);
    auto b = MakeBuffer(world);
    const uint64_t errors = Errors(world);

    Entity base[3];
    const Position zero{ 0, 0, 0 };
    CHECK(world->CreateEntities(CreateDesc({ ids.position }, { &zero }, 3), base) == nvrhi::FS_OK);

    // Placeholders, unique across the world's buffers.
    const Position positions[2] = { { 1, 2, 3 }, { 4, 5, 6 } };
    const Id idValues[2] = { { 11 }, { 12 } };
    const CreateDesc desc({ ids.position, ids.id }, { positions, idValues }, 2);
    Entity pa[2];
    Entity pb[2];
    a->CreateEntities(desc, MakeSortKey(0, 0, 0), pa);
    b->CreateEntities(desc, MakeSortKey(0, 1, 0), pb);
    const Entity all[4] = { pa[0], pa[1], pb[0], pb[1] };
    for (uint32_t i = 0; i < 4; ++i)
    {
        CHECK(IsPlaceholder(all[i]));
        for (uint32_t j = 0; j < i; ++j)
            CHECK(all[i] != all[j]);
    }

    // Later commands of either buffer name them; Entity fields of values hold them.
    const Velocity velocity{ 7, 8, 9 };
    b->AddComponent(pa[0], ids.velocity, &velocity, MakeSortKey(1, 0, 0));
    const Link link{ pb[1], base[0] };
    a->AddComponent(pa[1], ids.link, &link, MakeSortKey(1, 0, 1));
    const Position moved{ 9, 9, 9 };
    a->SetComponent(pb[1], ids.position, &moved, MakeSortKey(1, 0, 2));
    b->AddComponent(pb[1], ids.joint, nullptr, MakeSortKey(1, 1, 0));
    const Joint joints[2] = { { pa[0], 1.0f, 0 }, { base[1], 2.0f, 0 } };
    b->AppendToBuffer(pb[1], ids.joint, joints, 2, MakeSortKey(1, 1, 1));
    b->AddComponent(pb[1], ids.toggle, nullptr, MakeSortKey(1, 1, 2));
    b->SetEnabled(&pb[1], 1, ids.toggle, false, MakeSortKey(1, 1, 3));
    b->AddComponent(pa[1], ids.selected, nullptr, MakeSortKey(1, 1, 4));
    Entity instances[3];
    a->Instantiate(pa[0], 3, MakeSortKey(2, 0, 0), instances);
    b->DestroyEntity(pb[0], MakeSortKey(3, 0, 0));
    b->RemoveComponent(pa[1], ids.id, MakeSortKey(3, 0, 1));
    CHECK(a->GetCommandCount() == 4 && b->GetCommandCount() == 9);

    // Empty commands record nothing.
    a->CreateEntities(CreateDesc({ ids.position }, {}, 0), 0, nullptr);
    a->DestroyEntities(base, 0, 0);
    a->AddComponents(base, 0, ids.velocity, nullptr, 0);
    a->RemoveComponents(base, 0, ids.velocity, 0);
    a->SetComponents(base, 0, ids.position, nullptr, 0);
    a->SetEnabled(base, 0, ids.toggle, true, 0);
    a->Instantiate(base[0], 0, 0, nullptr);
    a->AppendToBuffer(base[0], ids.joint, nullptr, 0, 0);
    CHECK(a->GetCommandCount() == 4);

    WorldStats before;
    world->GetStats(before);
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
    CHECK(a->GetCommandCount() == 0 && b->GetCommandCount() == 0);
    WorldStats after;
    world->GetStats(after);
    CHECK(after.commandsPlayedBack == before.commandsPlayedBack + 13);
    CHECK(Errors(world) == errors);

    const Entity ea0 = world->ResolvePlaceholder(pa[0]);
    const Entity ea1 = world->ResolvePlaceholder(pa[1]);
    const Entity eb1 = world->ResolvePlaceholder(pb[1]);
    CHECK(world->IsAlive(ea0) && world->IsAlive(ea1) && world->IsAlive(eb1));
    CHECK(world->ResolvePlaceholder(pb[0]) == Entity::Null);   // destroyed
    // Ids in key order: a's creation (chunk 0) before b's (chunk 1).
    CHECK(GetEntityIndex(ea0) < GetEntityIndex(ea1) && GetEntityIndex(ea1) < GetEntityIndex(eb1));
    CHECK(Get<Velocity>(world, ea0).y == 8 && Get<Position>(world, ea0).x == 1 && Get<Id>(world, ea0).value == 11);
    CHECK(Get<Link>(world, ea1).other == eb1 && Get<Link>(world, ea1).outside == base[0]);
    CHECK(!world->HasComponent(ea1, ids.id) && world->HasComponent(ea1, ids.selected));
    CHECK(Get<Position>(world, eb1).x == 9 && !world->IsEnabled(eb1, ids.toggle));
    const BufferHeader& header = HeaderOf(world, eb1, ids.joint);
    CHECK(header.count == 2);
    CHECK(GetBufferData<Joint>(header)[0].target == ea0 && GetBufferData<Joint>(header)[1].target == base[1]);
    for (const Entity instance : instances)
    {
        const Entity entity = world->ResolvePlaceholder(instance);
        CHECK(world->IsAlive(entity) && Get<Velocity>(world, entity).z == 9);   // the prefab had its velocity by then
    }
    CHECK(world->ResolvePlaceholder(base[0]) == Entity::Null);   // not a placeholder

    // Valid until the next Playback starts.
    CHECK(world->Playback(nullptr, 0) == nvrhi::FS_OK);
    CHECK(world->ResolvePlaceholder(pa[0]) == Entity::Null);

    // Sort keys order the commands, not recording; equal keys in one buffer keep recording order.
    const Position p10{ 10, 0, 0 }, p20{ 20, 0, 0 }, p30{ 30, 0, 0 }, p31{ 31, 0, 0 };
    a->SetComponent(base[0], ids.position, &p20, MakeSortKey(0, 0, 20));
    a->SetComponent(base[0], ids.position, &p10, MakeSortKey(0, 0, 10));
    a->SetComponent(base[1], ids.position, &p30, MakeSortKey(0, 0, 5));
    a->SetComponent(base[1], ids.position, &p31, MakeSortKey(0, 0, 5));
    CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
    CHECK(Get<Position>(world, base[0]).x == 20 && Get<Position>(world, base[1]).x == 31);

    // Reset drops the commands; their placeholders never resolve.
    const uint32_t alive = world->GetStats(after).entityCount;
    Entity dropped;
    a->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &dropped);
    CHECK(a->GetCommandCount() == 1);
    a->Reset();
    CHECK(a->GetCommandCount() == 0);
    CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
    CHECK(world->ResolvePlaceholder(dropped) == Entity::Null && world->GetStats(after).entityCount == alive);

    // The arena keeps its pages.
    const size_t capacity = BufferImpl(a.Get()).GetArena().GetCapacity();
    CHECK(capacity > 0);
    for (uint32_t i = 0; i < 3; ++i)
    {
        a->SetComponent(base[0], ids.position, &p10, 0);
        a->Reset();
    }
    CHECK(BufferImpl(a.Get()).GetArena().GetCapacity() == capacity);

    // Playback stamps the world's version: the writes of a sync point come after every system's run.
    world->AdvanceVersion();
    const uint64_t version = world->GetVersion();
    a->AddComponent(base[2], ids.velocity, &velocity, 0);
    CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
    const ecs::EntityRecord& record = *Impl(world).FindAlive(base[2]);
    const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(record.archetype);
    CHECK(archetype.Versions(record.chunk)[archetype.FindColumn(ids.velocity)] == version);
    CHECK(archetype.Versions(record.chunk)[archetype.AddedVersionSlot()] == version);
    CHECK(Errors(world) == errors);
}

// A placeholder range that wraps past 2^32 is split, and every placeholder still resolves.
void test_placeholder_wrap()
{
    auto world = MakeWorld(true);
    const Ids ids = Register(world);
    ecs::CommandPlayback& state = State(world);
    const uint32_t current = state.AllocatePlaceholders(0);
    state.AllocatePlaceholders(0xfffffff0u - current);
    auto a = MakeBuffer(world);
    Id values[32];
    for (uint32_t i = 0; i < 32; ++i)
        values[i].value = i;
    Entity p[32];
    a->CreateEntities(CreateDesc({ ids.id }, { values }, 32), 0, p);
    CHECK(GetEntityIndex(p[0]) == 0xfffffff0u && GetEntityIndex(p[16]) == 0 && GetEntityIndex(p[31]) == 15);
    Link links[32];
    for (uint32_t i = 0; i < 32; ++i)
        links[i] = Link{ p[31 - i], Entity::Null };
    a->AddComponents(p, 32, ids.link, links, 1);
    CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
    CHECK(state.placeholders.ranges.size() == 2);
    std::vector<Entity> entities;
    for (uint32_t i = 0; i < 32; ++i)
    {
        const Entity entity = world->ResolvePlaceholder(p[i]);
        CHECK(world->IsAlive(entity) && Get<Id>(world, entity).value == i);
        for (const Entity other : entities)
            CHECK(other != entity);
        entities.push_back(entity);
    }
    for (uint32_t i = 0; i < 32; ++i)
        CHECK(Get<Link>(world, entities[i]).other == entities[31 - i]);
    CHECK(Errors(world) == 0);
}

// ----------------------------------------------------------------------------------------------- batching

void test_batching()
{
    auto world = MakeWorld(false, 1024);
    const Ids ids = Register(world);
    std::vector<Entity> base(300);
    std::vector<Position> positions(300);
    for (uint32_t i = 0; i < 300; ++i)
        positions[i] = Position{ float(i), 0, 0 };
    CHECK(world->CreateEntities(CreateDesc({ ids.position }, { positions.data() }, 300), base.data()) ==
          nvrhi::FS_OK);
    const ecs::Archetype& from = ArchetypeOf(world, base[0]);
    const uint32_t chunkCount = from.GetChunkCount();
    CHECK(chunkCount >= 6);
    std::vector<std::vector<Entity>> chunks(chunkCount);
    for (uint32_t c = 0; c < chunkCount; ++c)
        chunks[c].assign(from.Entities(c), from.Entities(c) + from.chunks[c].count);

    auto a = MakeBuffer(world);
    auto b = MakeBuffer(world);
    const ecs::CommandPlayback::Counters& counters = State(world).GetCounters();
    auto bufferOf = [&](uint32_t c) { return c % 2 ? b.Get() : a.Get(); };

    // One AddComponents per chunk, from two workers, at adjacent keys: one batch through AddComponentsInternal.
    std::vector<std::vector<Velocity>> velocities(chunkCount);
    for (uint32_t c = 0; c < chunkCount; ++c)
    {
        velocities[c].assign(chunks[c].size(), Velocity{ float(c), 0, 0 });
        bufferOf(c)->AddComponents(chunks[c].data(), uint32_t(chunks[c].size()), ids.velocity, velocities[c].data(),
                                   MakeSortKey(0, c, 0));
    }
    uint64_t batches = counters.batches;
    uint64_t coalesced = counters.coalesced;
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
    CHECK(counters.batches - batches == 1 && counters.coalesced - coalesced == chunkCount - 1);
    // The moved rows are in key order in the target archetype, with their values.
    const ecs::Archetype& to = ArchetypeOf(world, base[0]);
    CHECK(to.index != from.index && to.entityCount == 300);
    uint32_t position = 0;
    for (uint32_t c = 0; c < chunkCount; ++c)
    {
        for (const Entity entity : chunks[c])
        {
            const ecs::EntityRecord& record = *Impl(world).FindAlive(entity);
            CHECK(record.chunk * to.rowsPerChunk + record.row == position);
            CHECK(Get<Velocity>(world, entity).x == float(c));
            ++position;
        }
    }

    // Runs break where the operation, the component, the value mode or the flag changes.
    const std::vector<Id> idValues(400, Id{ 5 });
    a->RemoveComponents(chunks[0].data(), uint32_t(chunks[0].size()), ids.velocity, MakeSortKey(1, 0, 0));
    b->RemoveComponents(chunks[1].data(), uint32_t(chunks[1].size()), ids.velocity, MakeSortKey(1, 1, 0));   // joins
    a->AddComponents(chunks[0].data(), uint32_t(chunks[0].size()), ids.id, idValues.data(), MakeSortKey(1, 2, 0));
    b->AddComponents(chunks[1].data(), uint32_t(chunks[1].size()), ids.id, nullptr, MakeSortKey(1, 3, 0));
    a->AddComponents(chunks[2].data(), uint32_t(chunks[2].size()), ids.frozen, nullptr, MakeSortKey(1, 4, 0));
    b->AddComponents(chunks[3].data(), uint32_t(chunks[3].size()), ids.frozen, nullptr, MakeSortKey(1, 5, 0));
    const std::vector<Position> moved(400, Position{ -1, -1, -1 });
    a->SetComponents(chunks[2].data(), uint32_t(chunks[2].size()), ids.position, moved.data(), MakeSortKey(1, 6, 0));
    b->SetComponents(chunks[3].data(), uint32_t(chunks[3].size()), ids.position, moved.data(), MakeSortKey(1, 7, 0));
    const uint32_t half = uint32_t(chunks[4].size() / 2);
    b->DestroyEntities(chunks[4].data(), half, MakeSortKey(1, 8, 0));
    a->DestroyEntities(chunks[4].data() + half, uint32_t(chunks[4].size()) - half, MakeSortKey(1, 9, 0));   // joins
    batches = counters.batches;
    coalesced = counters.coalesced;
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
    CHECK(counters.batches - batches == 6 && counters.coalesced - coalesced == 4);
    CHECK(Get<Id>(world, chunks[0][0]).value == 5 && Get<Id>(world, chunks[1][0]).value == 0);
    CHECK(world->HasComponent(chunks[3].back(), ids.frozen) && Get<Position>(world, chunks[3].back()).x == -1);
    CHECK(!world->IsAlive(chunks[4][0]) && !world->IsAlive(chunks[4].back()));

    // SetEnabled: the same flag joins, the other does not.
    for (uint32_t c = 0; c < 2; ++c)
        for (const Entity entity : chunks[c])
            CHECK(world->AddComponent(entity, ids.toggle, nullptr) == nvrhi::FS_OK);
    a->SetEnabled(chunks[0].data(), uint32_t(chunks[0].size()), ids.toggle, false, MakeSortKey(2, 0, 0));
    b->SetEnabled(chunks[1].data(), uint32_t(chunks[1].size()), ids.toggle, false, MakeSortKey(2, 1, 0));
    a->SetEnabled(chunks[0].data(), 3, ids.toggle, true, MakeSortKey(2, 2, 0));
    batches = counters.batches;
    coalesced = counters.coalesced;
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
    CHECK(counters.batches - batches == 2 && counters.coalesced - coalesced == 1);
    CHECK(world->IsEnabled(chunks[0][0], ids.toggle) && !world->IsEnabled(chunks[0][3], ids.toggle));
    CHECK(!world->IsEnabled(chunks[1][0], ids.toggle));

    // A failed command or another operation between two runs separates them; Create, Instantiate and Append are one
    // batch each, a failed command none.
    a->AddComponents(chunks[2].data(), uint32_t(chunks[2].size()), ids.hidden, nullptr, MakeSortKey(3, 0, 0));
    a->AddComponents(chunks[2].data(), 1, ComponentId(999), nullptr, MakeSortKey(3, 1, 0));
    b->AddComponents(chunks[3].data(), uint32_t(chunks[3].size()), ids.hidden, nullptr, MakeSortKey(3, 2, 0));
    Entity created[2];
    b->CreateEntities(CreateDesc({ ids.position }, {}, 2), MakeSortKey(3, 3, 0), created);
    a->AddComponents(chunks[5].data(), uint32_t(chunks[5].size()), ids.hidden, nullptr, MakeSortKey(3, 4, 0));
    Entity instances[4];
    b->Instantiate(chunks[5][0], 4, MakeSortKey(3, 5, 0), instances);
    b->AddComponent(chunks[5][1], ids.joint, nullptr, MakeSortKey(3, 6, 0));
    const Joint joint{ created[0], 1, 0 };
    a->AppendToBuffer(chunks[5][1], ids.joint, &joint, 1, MakeSortKey(3, 7, 0));
    batches = counters.batches;
    coalesced = counters.coalesced;
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);   // the unknown component
    CHECK(counters.batches - batches == 7 && counters.coalesced - coalesced == 0);
    CHECK(world->HasComponent(chunks[3][0], ids.hidden) && world->HasComponent(chunks[5][0], ids.hidden));
    CHECK(world->HasComponent(world->ResolvePlaceholder(instances[3]), ids.hidden));
    CHECK(world->IsAlive(world->ResolvePlaceholder(created[1])));
    CHECK(GetBufferData<Joint>(HeaderOf(world, chunks[5][1], ids.joint))[0].target ==
          world->ResolvePlaceholder(created[0]));
}

// ---------------------------------------------------------------------------------------------- ownership

void test_ownership()
{
    const int64_t baseLive = g_Hooks.Live();
    Payload payload;
    {
        auto world = MakeWorld(true);
        const Ids ids = Register(world);
        auto a = MakeBuffer(world);
        auto b = MakeBuffer(world);
        Entity e[6];
        CHECK(world->CreateEntities(CreateDesc({ ids.position }, {}, 6), e) == nvrhi::FS_OK);
        auto live = [&]() { return g_Hooks.Live() - baseLive; };

        // Applied: the buffer copies at recording and moves its copies into the world; the caller keeps its own.
        {
            const Handle handles[2] = { Handle(&payload), Handle(&payload) };
            a->AddComponents(e, 2, ids.handle, handles, 0);
            CHECK(payload.refs == 4 && live() == 4);
        }
        CHECK(payload.refs == 2 && live() == 2);
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
        CHECK(payload.refs == 2 && live() == 2);
        CHECK(Get<Handle>(world, e[0]).payload == &payload && Get<Handle>(world, e[1]).payload == &payload);

        // Skipped as stale, inside a batch gathered from two buffers: that value is destroyed once.
        CHECK(world->DestroyEntities(&e[1], 1) == nvrhi::FS_OK);
        CHECK(payload.refs == 1);
        {
            const Handle handle(&payload);
            b->AddComponent(e[2], ids.handle, &handle, 0);
            a->AddComponent(e[1], ids.handle, &handle, 1);
            b->AddComponent(e[3], ids.handle, &handle, 2);
        }
        CHECK(payload.refs == 4);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(payload.refs == 3 && live() == 3);

        // Failed in the operation: SetComponents on an entity without the component.
        {
            const Handle handle(&payload);
            a->SetComponent(e[4], ids.handle, &handle, 0);
        }
        CHECK(payload.refs == 4);
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(payload.refs == 3 && live() == 3);

        // Set on an entity that has it: the old value is destroyed, the new one moved in.
        {
            const Handle handle(&payload);
            a->SetComponent(e[0], ids.handle, &handle, 0);
        }
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
        CHECK(payload.refs == 3 && live() == 3);

        // Failed when recorded: nothing is copied.
        {
            const Handle handle(&payload);
            a->AddComponents(nullptr, 2, ids.handle, &handle, 0);
            CHECK(payload.refs == 4 && a->GetCommandCount() == 1);
        }
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(payload.refs == 3 && live() == 3);

        // Reset without playback.
        {
            const Handle handle(&payload);
            a->SetComponent(e[0], ids.handle, &handle, 0);
            a->AddComponent(e[4], ids.handle, &handle, 1);
        }
        CHECK(payload.refs == 5);
        a->Reset();
        CHECK(payload.refs == 3 && live() == 3);

        // The buffer destroyed without playback.
        {
            auto c = MakeBuffer(world);
            const Handle handle(&payload);
            c->AddComponent(e[4], ids.handle, &handle, 0);
            CHECK(payload.refs == 5);
        }
        CHECK(payload.refs == 3 && live() == 3);

        // An unresolved placeholder as the target: skipped as stale, the value destroyed.
        {
            Entity p;
            b->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &p);
            const Handle handle(&payload);
            a->AddComponent(p, ids.handle, &handle, 0);
        }
        CHECK(payload.refs == 4);
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(payload.refs == 3 && live() == 3);
        b->Reset();

        // Relocated creation: the caller gives its storage up when recording.
        alignas(Handle) unsigned char storage[3 * sizeof(Handle)];
        Handle* handles = reinterpret_cast<Handle*>(storage);
        CreateDesc relocated({ ids.position, ids.handle }, { nullptr, handles }, 3, ValueTransfer::Relocate);
        Entity p[3];
        for (uint32_t i = 0; i < 3; ++i)
            new (handles + i) Handle(&payload);
        a->CreateEntities(relocated, 0, p);
        CHECK(payload.refs == 6 && live() == 6);
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
        CHECK(payload.refs == 6 && live() == 6);
        for (uint32_t i = 0; i < 3; ++i)
            CHECK(Get<Handle>(world, world->ResolvePlaceholder(p[i])).payload == &payload);
        // ... recorded, then reset.
        for (uint32_t i = 0; i < 3; ++i)
            new (handles + i) Handle(&payload);
        a->CreateEntities(relocated, 0, p);
        CHECK(payload.refs == 9);
        a->Reset();
        CHECK(payload.refs == 6 && live() == 6);
        // ... rejected when recorded (reservedEntities set): destroyed at once.
        for (uint32_t i = 0; i < 3; ++i)
            new (handles + i) Handle(&payload);
        Entity reserved[3];
        CHECK(world->ReserveEntities(3, reserved) == nvrhi::FS_OK);
        relocated->reservedEntities = reserved;
        a->CreateEntities(relocated, 0, p);
        CHECK(payload.refs == 6 && live() == 6 && a->GetCommandCount() == 1);
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(world->DestroyEntities(reserved, 3) == nvrhi::FS_OK);   // gives the reservations back
        relocated->reservedEntities = nullptr;
        // ... one of the created entities destroyed later in the same playback.
        for (uint32_t i = 0; i < 3; ++i)
            new (handles + i) Handle(&payload);
        a->CreateEntities(relocated, 0, p);
        b->DestroyEntity(p[1], 1);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
        CHECK(payload.refs == 8 && live() == 8);

        // A type without a copy hook: Copy fails (the caller keeps its value), Relocate moves it.
        {
            MoveOnly value(5);
            a->CreateEntities(CreateDesc({ ids.moveOnly }, { &value }, 1), 0, p);
            a->AddComponents(&e[5], 1, ids.moveOnly, &value, 1);
            CHECK(a->GetCommandCount() == 2 && value.value == 5);
            CHECK(Play(world, { a.Get() }) == nvrhi::FE_UNSUPPORTED);
            CHECK(world->ResolvePlaceholder(p[0]) == Entity::Null && !world->HasComponent(e[5], ids.moveOnly));
            a->CreateEntities(CreateDesc({ ids.moveOnly }, { &value }, 1, ValueTransfer::Relocate), 0, p);
            CHECK(value.value == -2);   // moved from and destroyed by the relocate hook
            new (&value) MoveOnly();
            CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
            CHECK(Get<MoveOnly>(world, world->ResolvePlaceholder(p[0])).value == 5);
        }

        // Buffer elements: appended, appended to a stale entity, and reset.
        CHECK(world->AddComponent(e[0], ids.refs, nullptr) == nvrhi::FS_OK);
        {
            const RefElement elements[3] = { RefElement(&payload), RefElement(&payload), RefElement(nullptr) };
            a->AppendToBuffer(e[0], ids.refs, elements, 3, 0);
            b->AppendToBuffer(e[1], ids.refs, elements, 3, 1);
            CHECK(payload.refs == 8 + 2 + 2 + 2);
        }
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(payload.refs == 10 && live() == 8 + 3);
        CHECK(HeaderOf(world, e[0], ids.refs).count == 3);
        {
            const RefElement element(&payload);
            a->AppendToBuffer(e[0], ids.refs, &element, 1, 0);
        }
        a->Reset();
        CHECK(payload.refs == 10 && live() == 11);

        // A coalesced Set batch, with a stale entity in the middle, gathered from three buffers.
        auto c = MakeBuffer(world);
        {
            const Handle handle(&payload);
            a->SetComponent(e[0], ids.handle, &handle, 0);
            b->SetComponent(e[1], ids.handle, &handle, 1);
            c->SetComponent(e[2], ids.handle, &handle, 2);
        }
        CHECK(payload.refs == 13);
        CHECK(Play(world, { c.Get(), b.Get(), a.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(payload.refs == 10 && live() == 11);
        CHECK(Get<Handle>(world, e[0]).payload == &payload && Get<Handle>(world, e[2]).payload == &payload);
    }
    CHECK(payload.refs == 0 && g_Hooks.Live() == baseLive);
}

// ------------------------------------------------------------------------------------------------- model

void test_model()
{
    uint64_t coalesced = 0;
    uint64_t batches = 0;
    uint64_t commands = 0;
    std::map<nvrhi::FRESULT, uint32_t> results;   // playback results by code
    uint32_t clashes = 0;
    for (uint32_t seed = 1; seed <= 6; ++seed)
    {
        for (uint32_t config = 0; config < 8; ++config)
        {
            const uint32_t chunkSize = config & 1 ? 16 * 1024 : 1024;
            const bool validation = (config & 2) != 0;
            const uint32_t bufferCount = config & 4 ? 3 : 1;
            const int64_t hooksLive = g_Hooks.Live();
            Payload payloads[c_PayloadCount];
            {
                auto world = MakeWorld(validation, chunkSize);
                auto model = MakeWorld(validation, chunkSize);
                const Ids ids = Register(world);
                CHECK(Register(model).refs == ids.refs);
                const Xorshift populate{ 0x9e3779b97f4a7c15ull * seed };
                const std::vector<Entity> stable = Populate(world, ids, populate, payloads, 60);
                CHECK(Populate(model, ids, populate, payloads, 60) == stable);
                // An entity whose MoveOnly component cannot be copied: the poison Instantiate of clean rounds.
                Entity movable = Entity::Null;
                Entity movableModel = Entity::Null;
                CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.moveOnly }, {}, 1), &movable) ==
                      nvrhi::FS_OK);
                CHECK(model->CreateEntities(CreateDesc({ ids.position, ids.moveOnly }, {}, 1), &movableModel) ==
                      nvrhi::FS_OK);
                CHECK(movable == movableModel);
                std::vector<Entity> known = stable;

                std::vector<nvrhi::AutoPtr<ICommandBuffer>> buffers;
                for (uint32_t b = 0; b < bufferCount; ++b)
                    buffers.push_back(MakeBuffer(world));
                Generator generator{ Xorshift{ 0x2545f4914f6cdd1dull * (seed * 16 + config) }, ids };
                generator.bufferCount = bufferCount;
                generator.movable = movable;

                // Clean rounds first, while the known ids keep what clean rounds rely on; then dirty rounds. The
                // second of each kind repeats keys across buffers.
                for (uint32_t round = 0; round < 4; ++round)
                {
                    generator.clean = round < 2;
                    generator.clash = round % 2 == 1;
                    generator.known = generator.clean ? stable : known;
                    generator.Generate(120);
                    const std::vector<Logical>& cmds = generator.commands;
                    std::vector<std::vector<Entity>> placeholders(cmds.size());
                    Recorder recorder{ world, ids, payloads, placeholders };
                    for (uint32_t i = 0; i < cmds.size(); ++i)
                        recorder.Record(buffers[cmds[i].buffer].Get(), cmds[i], i);
                    uint32_t recorded = 0;
                    for (auto& buffer : buffers)
                        recorded += buffer->GetCommandCount();
                    CHECK(recorded == cmds.size());

                    const ecs::CommandPlayback::Counters before = State(world).GetCounters();
                    WorldStats stats;
                    const uint64_t played = world->GetStats(stats).commandsPlayedBack;
                    std::vector<ICommandBuffer*> list;
                    for (auto& buffer : buffers)
                        list.push_back(buffer.Get());
                    const nvrhi::FRESULT result = world->Playback(list.data(), uint32_t(list.size()));
                    CHECK(world->GetStats(stats).commandsPlayedBack == played + cmds.size());
                    const ecs::CommandPlayback::Counters& after = State(world).GetCounters();
                    batches += after.batches - before.batches;
                    coalesced += after.coalesced - before.coalesced;
                    commands += cmds.size();

                    // The model applies the commands one by one in playback order. With validation, equal keys in two
                    // buffers fail at the first command with that key.
                    const std::vector<uint32_t> order = PlaybackOrder(cmds);
                    size_t clashAt = order.size();
                    for (size_t p = 0; p < order.size() && clashAt == order.size();)
                    {
                        size_t q = p + 1;
                        bool mixed = false;
                        while (q < order.size() && cmds[order[q]].key == cmds[order[p]].key)
                        {
                            mixed = mixed || cmds[order[q]].buffer != cmds[order[p]].buffer;
                            ++q;
                        }
                        if (mixed)
                            clashAt = p;
                        p = q;
                    }
                    clashes += clashAt != order.size() ? 1 : 0;
                    Model applied{ model, ids, payloads, placeholders };
                    applied.created.resize(cmds.size());
                    applied.applied.assign(cmds.size(), false);
                    for (size_t p = 0; p < order.size(); ++p)
                    {
                        if (validation && p == clashAt)
                            applied.Keep(nvrhi::FE_INVALID_ARGS);
                        applied.Apply(cmds[order[p]], order[p]);
                    }
                    CHECK(result == applied.result);
                    ++results[result];
                    if (generator.clean && !generator.clash)
                    {
                        // Only the poison fails a clean round.
                        bool poisoned = false;
                        for (const bool isPoison : generator.poison)
                            poisoned = poisoned || isPoison;
                        CHECK(poisoned == NVRHI_FAILED(result));
                    }

                    CHECK(world->ComputeHash() == model->ComputeHash());
                    CHECK(Dump(world, ids) == Dump(model, ids));
                    for (uint32_t i = 0; i < cmds.size(); ++i)
                    {
                        if (!Creates(cmds[i]))
                            continue;
                        for (uint32_t t = 0; t < cmds[i].count; ++t)
                        {
                            const Entity entity = applied.created[i][t];
                            const Entity expected = model->IsAlive(entity) ? entity : Entity::Null;
                            CHECK(world->ResolvePlaceholder(placeholders[i][t]) == expected);
                            known.push_back(entity);
                        }
                        if (i % 7 == 0)
                            known.push_back(placeholders[i][0]);   // a placeholder of an earlier round: never resolves
                    }
                }
            }
            for (const Payload& payload : payloads)
                CHECK(payload.refs == 0);
            CHECK(g_Hooks.Live() == hooksLive);
        }
    }
    printf("test_ecs_commands: model: %llu commands in %llu batches (%llu joined a batch); results:",
           (unsigned long long)commands, (unsigned long long)batches, (unsigned long long)coalesced);
    for (const auto& entry : results)
        printf(" %d x%u", int(entry.first), entry.second);
    printf("; %u rounds with equal keys across buffers\n", clashes);
    CHECK(coalesced > commands / 20 && clashes > 0);
    CHECK(results[nvrhi::FS_OK] > 0 && results[nvrhi::FE_INVALID_ARGS] > 0 && results[nvrhi::FE_UNSUPPORTED] > 0);
}

// -------------------------------------------------------------------------------------------- determinism

namespace
{
    struct Outcome
    {
        nvrhi::FRESULT result = nvrhi::FS_OK;
        uint64_t hash = 0;
        std::vector<uint64_t> dump;
        std::vector<std::vector<Entity>> resolved;   // per command: what its placeholders became
    };

    enum class Recording { InOrder, Shuffled, Threads };

    // A distribution of the commands over `bufferCount` workers; commands with one key stay in one buffer.
    std::vector<Logical> Assign(std::vector<Logical> commands, uint32_t bufferCount, uint64_t seed)
    {
        Xorshift rng{ seed };
        std::map<uint64_t, uint32_t> bufferOfKey;
        for (Logical& command : commands)
        {
            auto found = bufferOfKey.find(command.key);
            if (found == bufferOfKey.end())
                found = bufferOfKey.emplace(command.key, rng.Below(bufferCount)).first;
            command.buffer = found->second;
        }
        return commands;
    }

    // Records the commands into fresh buffers of a fresh world and plays them back: in generation order, in a shuffled
    // interleaving of the buffers (each buffer in its own order), or from one thread per buffer.
    Outcome Run(const std::vector<Logical>& commands, uint32_t bufferCount, Recording mode, uint64_t seed,
                bool validation, Payload* payloads)
    {
        auto world = MakeWorld(validation, 1024);
        const Ids ids = Register(world);
        Populate(world, ids, Xorshift{ 0x5851f42d4c957f2dull }, payloads, 80);
        std::vector<nvrhi::AutoPtr<ICommandBuffer>> buffers;
        for (uint32_t b = 0; b < bufferCount; ++b)
            buffers.push_back(MakeBuffer(world));
        const uint32_t n = uint32_t(commands.size());
        std::vector<std::vector<Entity>> placeholders(n);
        for (uint32_t i = 0; i < n; ++i)
            if (Creates(commands[i]))
                placeholders[i].assign(commands[i].count, Entity::Null);
        Recorder recorder{ world, ids, payloads, placeholders };
        std::vector<std::vector<uint32_t>> queues(bufferCount);
        for (uint32_t i = 0; i < n; ++i)
            queues[commands[i].buffer].push_back(i);

        if (mode == Recording::InOrder)
        {
            for (uint32_t i = 0; i < n; ++i)
                recorder.Record(buffers[commands[i].buffer].Get(), commands[i], i);
        }
        else if (mode == Recording::Shuffled)
        {
            Xorshift rng{ seed };
            std::vector<bool> recorded(n, false);
            std::vector<size_t> heads(bufferCount, 0);
            uint32_t done = 0;
            while (done < n)
            {
                const uint32_t b = rng.Below(bufferCount);
                if (heads[b] == queues[b].size())
                    continue;
                const uint32_t i = queues[b][heads[b]];
                bool ready = true;
                for (const uint32_t dependency : commands[i].dependencies)
                    ready = ready && recorded[dependency];
                if (!ready)
                    continue;
                recorder.Record(buffers[b].Get(), commands[i], i);
                recorded[i] = true;
                ++heads[b];
                ++done;
            }
        }
        else
        {
            // A command waits until the creating commands it names were recorded by their threads.
            std::unique_ptr<std::atomic<uint32_t>[]> recorded(new std::atomic<uint32_t>[n]);
            for (uint32_t i = 0; i < n; ++i)
                recorded[i].store(0, std::memory_order_relaxed);
            std::vector<std::thread> threads;
            for (uint32_t b = 0; b < bufferCount; ++b)
            {
                threads.emplace_back([&, b]()
                {
                    for (const uint32_t i : queues[b])
                    {
                        for (const uint32_t dependency : commands[i].dependencies)
                            while (recorded[dependency].load(std::memory_order_acquire) == 0)
                                std::this_thread::yield();
                        recorder.Record(buffers[b].Get(), commands[i], i);
                        recorded[i].store(1, std::memory_order_release);
                    }
                });
            }
            for (std::thread& thread : threads)
                thread.join();
        }

        std::vector<ICommandBuffer*> list;
        for (auto& buffer : buffers)
            list.push_back(buffer.Get());
        Outcome outcome;
        outcome.result = world->Playback(list.data(), uint32_t(list.size()));
        outcome.hash = world->ComputeHash();
        outcome.dump = Dump(world, ids);
        outcome.resolved.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            for (const Entity placeholder : placeholders[i])
                outcome.resolved[i].push_back(world->ResolvePlaceholder(placeholder));
        return outcome;
    }
}

void test_determinism()
{
    uint32_t runs = 0;
    const int64_t hooksLive = g_Hooks.Live();
    // One payload pool for every run, so that the dumps (which hold payload addresses) compare across runs. Threads
    // record counted values too: the counters are atomic.
    Payload pool[c_PayloadCount];
    auto balanced = [&]()
    {
        for (const Payload& payload : pool)
            CHECK(payload.refs == 0);
        CHECK(g_Hooks.Live() == hooksLive);
    };
    for (uint32_t seed = 1; seed <= 3; ++seed)
    {
        std::vector<Entity> known;
        {
            auto scratch = MakeWorld();
            const Ids ids = Register(scratch);
            known = Populate(scratch, ids, Xorshift{ 0x5851f42d4c957f2dull }, pool, 80);
        }
        auto ids = [] { auto world = MakeWorld(); return Register(world); }();
        Generator generator{ Xorshift{ 0xda942042e4dd58b5ull * seed }, ids };
        generator.payloads = true;
        generator.known = known;
        generator.Generate(400);
        const std::vector<Logical> commands = generator.commands;

        const Outcome reference = Run(Assign(commands, 1, 0), 1, Recording::InOrder, 0, true, pool);
        balanced();
        uint32_t alive = 0;
        for (const std::vector<Entity>& entities : reference.resolved)
            for (const Entity entity : entities)
                alive += entity != Entity::Null ? 1 : 0;
        CHECK(alive > 20);

        for (const uint32_t bufferCount : { 2u, 3u, 8u })
        {
            for (uint32_t variant = 0; variant < 3; ++variant)
            {
                const uint64_t assignment = seed * 1000 + bufferCount * 10 + variant;
                const std::vector<Logical> assigned = Assign(commands, bufferCount, assignment);
                for (const Recording mode : { Recording::InOrder, Recording::Shuffled, Recording::Threads })
                {
                    const uint32_t repeats = mode == Recording::Threads ? 3 : 1;
                    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
                    {
                        // With validation, threads included: its reports are logged under donut's log mutex.
                        const Outcome outcome = Run(assigned, bufferCount, mode, assignment + repeat, true, pool);
                        balanced();
                        CHECK(outcome.result == reference.result);
                        CHECK(outcome.hash == reference.hash);
                        CHECK(outcome.dump == reference.dump);
                        CHECK(outcome.resolved == reference.resolved);
                        ++runs;
                    }
                }
            }
        }
    }
    printf("test_ecs_commands: determinism: %u runs identical to their single-buffer reference\n", runs);
}

// --------------------------------------------------------------------------------------------- validation

void test_validation()
{
    for (const bool validation : { true, false })
    {
        auto world = MakeWorld(validation);
        const Ids ids = Register(world);
        auto a = MakeBuffer(world);
        auto b = MakeBuffer(world);
        Entity e[4];
        CHECK(world->CreateEntities(CreateDesc({ ids.position }, {}, 4), e) == nvrhi::FS_OK);
        auto reported = [&](uint64_t before, uint64_t count)
        {
            return validation ? Errors(world) == before + count : Errors(world) == 0;
        };
        const Velocity v{ 1, 1, 1 };
        uint64_t before = 0;

        // Equal keys in two buffers are reported; both commands still apply. In one buffer they are fine.
        before = Errors(world);
        a->AddComponent(e[0], ids.velocity, &v, 5);
        b->AddComponent(e[1], ids.velocity, &v, 5);
        CHECK(Play(world, { a.Get(), b.Get() }) == (validation ? nvrhi::FE_INVALID_ARGS : nvrhi::FS_OK));
        CHECK(reported(before, 1));
        CHECK(world->HasComponent(e[0], ids.velocity) && world->HasComponent(e[1], ids.velocity));
        before = Errors(world);
        a->RemoveComponent(e[0], ids.velocity, 6);
        a->RemoveComponent(e[1], ids.velocity, 6);
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK && reported(before, 0));

        // A placeholder of a buffer that is not played back with it (another system's).
        Entity p = Entity::Null;
        a->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &p);
        b->AddComponent(p, ids.velocity, &v, 1);
        before = Errors(world);
        CHECK(Play(world, { b.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(Errors(world) >= before + (validation ? 1 : 0));
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
        CHECK(world->IsAlive(world->ResolvePlaceholder(p)));
        CHECK(!world->HasComponent(world->ResolvePlaceholder(p), ids.velocity));

        // Named before its creation in key order.
        a->CreateEntities(CreateDesc({ ids.position }, {}, 1), 10, &p);
        b->AddComponent(p, ids.velocity, &v, 1);
        before = Errors(world);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(Errors(world) >= before + (validation ? 1 : 0));
        CHECK(world->IsAlive(world->ResolvePlaceholder(p)));
        CHECK(!world->HasComponent(world->ResolvePlaceholder(p), ids.velocity));

        // An Entity field that names a placeholder that does not resolve is written as null.
        Entity elsewhere;
        b->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &elsewhere);
        const Link link{ elsewhere, e[0] };
        a->AddComponent(e[2], ids.link, &link, 0);
        before = Errors(world);
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS && reported(before, 1));
        CHECK(Get<Link>(world, e[2]).other == Entity::Null && Get<Link>(world, e[2]).outside == e[0]);
        b->Reset();

        // A placeholder of an earlier playback: resolvable until the next playback starts, never named again.
        a->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &p);
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
        const Entity real = world->ResolvePlaceholder(p);
        CHECK(world->IsAlive(real));
        a->AddComponent(p, ids.velocity, &v, 0);
        before = Errors(world);
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(Errors(world) >= before + (validation ? 1 : 0));
        CHECK(!world->HasComponent(real, ids.velocity) && world->ResolvePlaceholder(p) == Entity::Null);

        // Unknown components and resources, in every command: reported when recorded, returned by playback, which
        // still applies the rest.
        const ComponentId unknown = ComponentId(999);
        before = Errors(world);
        Entity unplayed;
        a->AddComponent(e[0], unknown, nullptr, 0);
        a->RemoveComponent(e[0], unknown, 1);
        a->SetComponent(e[0], unknown, &v, 2);
        a->SetEnabled(e, 1, unknown, true, 3);
        a->AppendToBuffer(e[0], unknown, &v, 1, 4);
        a->CreateEntities(CreateDesc({ ids.position, unknown }, {}, 1), 5, &unplayed);
        a->AddComponent(e[0], ids.gravity, nullptr, 6);
        a->AddComponent(e[1], ids.velocity, &v, 7);
        CHECK(reported(before, 7) && a->GetCommandCount() == 8 && IsPlaceholder(unplayed));
        before = Errors(world);
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS && reported(before, 0));
        CHECK(world->HasComponent(e[1], ids.velocity) && world->ResolvePlaceholder(unplayed) == Entity::Null);

        // Arguments the command cannot take: reported when recorded.
        Entity reserved;
        CHECK(world->ReserveEntities(1, &reserved) == nvrhi::FS_OK);
        CreateDesc withReserved({ ids.position }, {}, 1);
        withReserved->reservedEntities = &reserved;
        Entity p6 = Entity::Null;
        Entity p7 = Entity::Null;
        Entity p8 = Entity::Null;
        const Joint joint{ e[0], 1, 0 };
        const ChunkBounds bounds{ 0, 1 };
        const Position zero{ 0, 0, 0 };
        before = Errors(world);
        a->CreateEntities(withReserved, 0, &p6);
        a->CreateEntities(CreateDesc({ ids.joint }, { &joint }, 1), 1, &p7);   // a Buffer takes no initial value
        a->AddComponent(e[0], ids.bounds, &bounds, 2);                         // nor a Chunk component
        a->SetComponent(e[0], ids.joint, &joint, 3);                           // a Buffer is not set
        a->AppendToBuffer(e[0], ids.position, &zero, 1, 4);                    // not a Buffer
        a->SetComponents(e, 1, ids.position, nullptr, 5);                      // no values
        a->DestroyEntities(nullptr, 2, 6);                                     // no list
        a->AppendToBuffer(e[0], ids.joint, nullptr, 1, 7);                     // no elements
        CreateDesc noComponents({ ids.position }, {}, 1);
        noComponents->components = nullptr;
        a->CreateEntities(noComponents, 8, &p8);
        CHECK(reported(before, 9) && a->GetCommandCount() == 9);
        CHECK(IsPlaceholder(p6) && IsPlaceholder(p7) && IsPlaceholder(p8));
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(world->ResolvePlaceholder(p6) == Entity::Null && world->ResolvePlaceholder(p7) == Entity::Null);
        CHECK(world->DestroyEntities(&reserved, 1) == nvrhi::FS_OK);

        // Values the buffer cannot copy.
        MoveOnly moveOnly(3);
        before = Errors(world);
        a->AddComponent(e[0], ids.moveOnly, &moveOnly, 0);
        CHECK(reported(before, 1));
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_UNSUPPORTED && moveOnly.value == 3);

        // At the wrong time (inside a phase, outside an exclusive system), Playback does nothing and says so with its
        // own code: the buffer keeps its commands, the resolutions stay.
        a->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &p);
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
        const Entity resolved = world->ResolvePlaceholder(p);
        a->AddComponent(e[3], ids.velocity, &v, 0);
        Impl(world).BeginPhase();
        before = Errors(world);
        CHECK(Play(world, { a.Get() }) == nvrhi::FE_OUT_OF_DATE && reported(before, 1));
        CHECK(world->Playback(nullptr, 0) == nvrhi::FE_OUT_OF_DATE);
        CHECK(a->GetCommandCount() == 1 && !world->HasComponent(e[3], ids.velocity));
        CHECK(world->ResolvePlaceholder(p) == resolved);
        Impl(world).BeginExclusive();
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);
        Impl(world).EndExclusive();
        Impl(world).EndPhase();
        CHECK(world->HasComponent(e[3], ids.velocity));

        // Entries that cannot be played back (another world's buffer, a buffer listed again, null) are skipped and
        // left as they are; the other buffers still apply and are reset; the call fails.
        auto other = MakeWorld(validation);
        Register(other);
        auto foreign = MakeBuffer(other);
        foreign->DestroyEntities(e, 1, 0);
        a->AddComponent(e[3], ids.frozen, nullptr, 0);
        before = Errors(world);
        CHECK(Play(world, { a.Get(), foreign.Get() }) == nvrhi::FE_INVALID_ARGS && reported(before, 1));
        CHECK(world->HasComponent(e[3], ids.frozen) && a->GetCommandCount() == 0);
        CHECK(foreign->GetCommandCount() == 1 && world->IsAlive(e[0]));
        a->AddComponent(e[2], ids.frozen, nullptr, 0);
        const Position moved{ 5, 5, 5 };
        a->SetComponent(e[2], ids.position, &moved, 1);
        CHECK(Play(world, { a.Get(), a.Get() }) == nvrhi::FE_INVALID_ARGS);   // played once
        CHECK(world->HasComponent(e[2], ids.frozen) && Get<Position>(world, e[2]).x == 5 && a->GetCommandCount() == 0);
        a->AddComponent(e[1], ids.frozen, nullptr, 0);
        CHECK(Play(world, { nullptr, a.Get() }) == nvrhi::FE_INVALID_ARGS && world->HasComponent(e[1], ids.frozen));
        CHECK(world->Playback(nullptr, 1) == nvrhi::FE_INVALID_ARGS);
        foreign->Reset();

        // One thread at a time. While another thread records, a call records nothing (its relocated values are
        // consumed) and makes the buffer's next playback fail; playback skips a buffer in use and leaves it as it is.
        // Without validation nothing is checked.
        ecs::CommandBufferImpl& impl = BufferImpl(a.Get());
        Payload payload;
        impl.SimulateConcurrentUseForTest(true);
        before = Errors(world);
        a->AddComponent(e[0], ids.hidden, nullptr, 0);
        Entity lost = Entity::Null;
        alignas(Handle) unsigned char storage[sizeof(Handle)];
        Handle* handle = new (storage) Handle(&payload);
        a->CreateEntities(CreateDesc({ ids.handle }, { handle }, 1, ValueTransfer::Relocate), 1, &lost);
        CHECK(IsPlaceholder(lost) && payload.refs == (validation ? 0 : 1));
        CHECK(reported(before, 2) && a->GetCommandCount() == (validation ? 0u : 2u));
        if (validation)
        {
            a->Reset();   // refused as well: it would race with the other thread
            CHECK(reported(before, 3));
        }
        b->AddComponent(e[1], ids.hidden, nullptr, 0);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);
        CHECK(world->HasComponent(e[1], ids.hidden) && b->GetCommandCount() == 0);
        CHECK(a->GetCommandCount() == (validation ? 0u : 2u) && !world->HasComponent(e[0], ids.hidden));
        impl.SimulateConcurrentUseForTest(false);
        CHECK(Play(world, { a.Get() }) == (validation ? nvrhi::FE_INVALID_ARGS : nvrhi::FS_OK));
        CHECK(a->GetCommandCount() == 0 && world->HasComponent(e[0], ids.hidden) == !validation);
        const Entity created = world->ResolvePlaceholder(lost);
        CHECK(world->IsAlive(created) == !validation);
        CHECK(Play(world, { a.Get() }) == nvrhi::FS_OK);   // the lost call fails one playback
        if (!validation)
            CHECK(world->DestroyEntities(&created, 1) == nvrhi::FS_OK);
        CHECK(payload.refs == 0);

        if (!validation)
            CHECK(Errors(world) == 0);
    }
}

// ------------------------------------------------------------------------------------------ first failure

// Playback returns the failure of the earliest failing command in merged order, as if the commands ran one by one: a
// key clash fails at its key, and an Entity field that does not resolve fails its own command, not its batch's first.
void test_first_failure()
{
    auto world = MakeWorld(true, 1024);
    const Ids ids = Register(world);
    auto a = MakeBuffer(world);
    auto b = MakeBuffer(world);
    auto elsewhere = MakeBuffer(world);
    const ecs::CommandPlayback::Counters& counters = State(world).GetCounters();
    Entity e[6];
    CHECK(world->CreateEntities(CreateDesc({ ids.position }, {}, 6), e) == nvrhi::FS_OK);

    // A key clash fails at its key: after an earlier failure, before a later one.
    MoveOnly moveOnly(1);
    a->AddComponent(e[0], ids.moveOnly, &moveOnly, 1);   // FE_UNSUPPORTED when recorded
    a->AddComponent(e[1], ids.frozen, nullptr, 5);
    b->AddComponent(e[2], ids.frozen, nullptr, 5);
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_UNSUPPORTED);
    a->AddComponent(e[1], ids.hidden, nullptr, 5);
    b->AddComponent(e[2], ids.hidden, nullptr, 5);
    a->AddComponent(e[0], ids.moveOnly, &moveOnly, 9);
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->HasComponent(e[1], ids.hidden) && world->HasComponent(e[2], ids.hidden));

    // Inside a run of AddComponents, on fresh entities. f[0] and f[1] move to {Position, Link}, which exists; f[2] and
    // f[3] (with Velocity) to {Position, Velocity, Link}, which does not, and whose creation fails. Everything a
    // playback would allocate first is warmed, so that the failure hits that archetype.
    Entity f[6];
    CHECK(world->CreateEntities(CreateDesc({ ids.position }, {}, 6), f) == nvrhi::FS_OK);
    CHECK(world->AddComponent(f[5], ids.link, nullptr) == nvrhi::FS_OK);
    for (uint32_t i = 2; i < 4; ++i)
        CHECK(world->AddComponent(f[i], ids.velocity, nullptr) == nvrhi::FS_OK);
    Entity layout;
    CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.velocity, ids.id }, {}, 1), &layout) == nvrhi::FS_OK);
    const Position p{ 1, 1, 1 };
    a->SetComponent(f[0], ids.position, &p, 0);
    a->SetComponent(f[1], ids.position, &p, 1);
    b->SetComponent(f[4], ids.position, &p, 2);   // a gather block of 36 bytes
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
    Entity unplayed;
    elsewhere->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &unplayed);
    const Link good{ f[5], Entity::Null };
    const Link bad{ unplayed, Entity::Null };   // never resolves: written as null, the command fails

    // The batch's first command fails in the operation, the second when prepared: the first failure wins.
    a->AddComponent(f[2], ids.link, &good, 10);
    b->AddComponent(f[0], ids.link, &bad, 11);
    ecs::SetHeapFailureCountdown(0);
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_OUT_OF_MEMORY);
    ecs::SetHeapFailureCountdown(-1);
    CHECK(!world->HasComponent(f[2], ids.link) && Get<Link>(world, f[0]).other == Entity::Null);

    // The first command fails when prepared, the second in the operation, in one batch: the first failure wins.
    uint64_t batches = counters.batches;
    a->AddComponent(f[1], ids.link, &bad, 20);
    b->AddComponent(f[3], ids.link, &good, 21);
    ecs::SetHeapFailureCountdown(0);
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);
    ecs::SetHeapFailureCountdown(-1);
    CHECK(counters.batches - batches == 1);
    CHECK(world->HasComponent(f[1], ids.link) && !world->HasComponent(f[3], ids.link));

    // Only the second command of a would-be batch fails: it runs as a batch of its own, and its failure is returned.
    batches = counters.batches;
    a->SetComponent(f[0], ids.link, &good, 30);
    b->SetComponent(f[1], ids.link, &bad, 31);
    CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_INVALID_ARGS);
    CHECK(counters.batches - batches == 2);
    CHECK(Get<Link>(world, f[0]).other == f[5] && Get<Link>(world, f[1]).other == Entity::Null);
    elsewhere->Reset();
}

// ------------------------------------------------------------------------------------------------- versions

// Playback stamps the world's version for writes outside phases: every command of a sync point, whichever system
// recorded it, gets the version taken after the phase, newer than every system's own; columns a frame leaves alone keep
// their versions.
void test_versions()
{
    auto world = MakeWorld(true);
    const Ids ids = Register(world);
    ecs::WorldImpl& impl = Impl(world);
    auto systemA = MakeBuffer(world);
    auto systemB = MakeBuffer(world);
    Entity e[8];
    CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.toggle, ids.joint }, {}, 8), e) == nvrhi::FS_OK);
    auto version = [&](Entity entity, ComponentId component)
    {
        const ecs::EntityRecord& record = *impl.FindAlive(entity);
        const ecs::Archetype& archetype = impl.GetArchetypeStorage(record.archetype);
        return archetype.Versions(record.chunk)[archetype.FindColumn(component)];
    };
    auto added = [&](Entity entity)
    {
        const ecs::EntityRecord& record = *impl.FindAlive(entity);
        const ecs::Archetype& archetype = impl.GetArchetypeStorage(record.archetype);
        return archetype.Versions(record.chunk)[archetype.AddedVersionSlot()];
    };

    uint64_t firstSync = 0;
    for (uint32_t frame = 0; frame < 2; ++frame)
    {
        const uint64_t runA = world->AdvanceVersion();   // the systems' versions, in schedule order
        const uint64_t runB = world->AdvanceVersion();
        impl.BeginPhase();
        const Position moved{ float(frame + 1), 0, 0 };
        Entity created = Entity::Null;
        if (frame == 0)
        {
            systemA->AddComponent(e[0], ids.velocity, nullptr, 0);
            systemA->SetEnabled(&e[1], 1, ids.toggle, false, 1);
            const Joint joint{ e[2], 1, 0 };
            systemA->AppendToBuffer(e[2], ids.joint, &joint, 1, 2);
            systemA->CreateEntities(CreateDesc({ ids.position, ids.velocity }, {}, 1), 3, &created);
            systemB->SetComponent(e[3], ids.position, &moved, 0);
            systemB->RemoveComponent(e[4], ids.toggle, 1);
        }
        else
        {
            systemB->SetComponent(e[5], ids.position, &moved, 0);
        }
        impl.EndPhase();
        const uint64_t sync = world->AdvanceVersion();   // for writes outside phases
        CHECK(runA < runB && runB < sync);
        CHECK(Play(world, { systemA.Get() }) == nvrhi::FS_OK);
        const Entity createdEntity = world->ResolvePlaceholder(created);   // until system B's playback starts
        CHECK(Play(world, { systemB.Get() }) == nvrhi::FS_OK);
        if (frame == 0)
        {
            firstSync = sync;
            CHECK(version(e[0], ids.velocity) == sync && added(e[0]) == sync);   // a move
            CHECK(version(e[1], ids.toggle) == sync);                            // SetEnabled
            CHECK(version(e[2], ids.joint) == sync);                             // AppendToBuffer
            CHECK(added(createdEntity) == sync);                                 // CreateEntities
            CHECK(version(e[3], ids.position) == sync);                          // system B's SetComponents
            CHECK(added(e[4]) == sync);                                          // RemoveComponents moved it
        }
        else
        {
            CHECK(version(e[5], ids.position) == sync);
            CHECK(version(e[5], ids.joint) == firstSync && version(e[5], ids.toggle) == firstSync);
            CHECK(version(e[0], ids.velocity) == firstSync);
        }
    }
}

// ------------------------------------------------------------------------------------- buffers the world owns

// A NotHeld buffer (E1.4's observer buffer) does not keep its world alive, plays back like any buffer of its world, and
// through PlayCommandBuffers resolves placeholders in the table it is given, not in IWorld::ResolvePlaceholder's.
void test_not_held()
{
    auto world = MakeWorld(true);
    const Ids ids = Register(world);
    auto references = [&]()
    {
        const auto count = world->AddRef();
        world->Release();
        return count;
    };
    const auto before = references();
    ICommandBuffer* notHeld = nullptr;
    CHECK(ecs::CreateCommandBufferObject(Impl(world), ecs::CommandBufferWorldReference::NotHeld, &notHeld) ==
          nvrhi::FS_OK && notHeld);
    CHECK(references() == before);
    {
        auto held = MakeBuffer(world);
        CHECK(references() == before + 1);
    }
    CHECK(references() == before);

    Entity e;
    CHECK(world->CreateEntities(CreateDesc({ ids.position }, {}, 1), &e) == nvrhi::FS_OK);
    notHeld->AddComponent(e, ids.velocity, nullptr, 0);
    CHECK(world->Playback(&notHeld, 1) == nvrhi::FS_OK && world->HasComponent(e, ids.velocity));

    ecs::CommandPlayback& state = State(world);
    Entity p;
    notHeld->CreateEntities(CreateDesc({ ids.position }, {}, 1), 0, &p);
    const Link link{ p, e };
    notHeld->AddComponent(e, ids.link, &link, 1);
    state.observerPlaceholders.Clear();
    ecs::CommandBufferImpl* buffer = &BufferImpl(notHeld);
    uint64_t commands = 0;
    CHECK(ecs::PlayCommandBuffers(Impl(world), &buffer, 1, world->GetVersion(), state.observerPlaceholders, commands) ==
          nvrhi::FS_OK && commands == 2);
    const Entity created = state.observerPlaceholders.Resolve(p);
    CHECK(world->IsAlive(created) && Get<Link>(world, e).other == created);
    CHECK(state.placeholders.Resolve(p) == Entity::Null && world->ResolvePlaceholder(p) == Entity::Null);
    state.observerPlaceholders.Clear();
    CHECK(state.observerPlaceholders.Resolve(p) == Entity::Null);
    notHeld->Release();   // before the world, which it does not keep alive
    CHECK(references() == before);
}

// --------------------------------------------------------------------------------------------- registration

// Recording workers read the registry, so a new component type is refused while systems may run (inside a phase, unless
// an exclusive system runs) and while an observer runs; finding a registered type is allowed everywhere.
void test_registration_in_phases()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId position = RegisterComponent<Position>(world);
    CHECK(position != ComponentId::Invalid);
    impl.BeginPhase();
    const uint64_t errors = Errors(world);
    CHECK(RegisterComponent<Position>(world) == position);   // registered: a lookup
    CHECK(RegisterComponent<Velocity>(world) == ComponentId::Invalid);
    ComponentId out = ComponentId(0);
    CHECK(world->RegisterComponent(ComponentTraits<Velocity>::Desc(), out) == nvrhi::FE_OUT_OF_DATE);
    CHECK(out == ComponentId::Invalid);
    CHECK(world->RegisterResource(ComponentTraits<Gravity>::Desc(), nullptr, out) == nvrhi::FE_OUT_OF_DATE);
    CHECK(Errors(world) == errors + 3);
    CHECK(FindComponent<Velocity>(world) == ComponentId::Invalid && world->GetComponentTypeCount() == 1);
    impl.BeginExclusive();
    const ComponentId velocity = RegisterComponent<Velocity>(world);
    CHECK(velocity != ComponentId::Invalid);
    impl.EndExclusive();
    CHECK(RegisterComponent<Velocity>(world) == velocity);
    impl.EndPhase();
    CHECK(RegisterComponent<Id>(world) != ComponentId::Invalid);
    CHECK(world->RegisterResource(ComponentTraits<Gravity>::Desc(), nullptr, out) == nvrhi::FS_OK);
    CHECK(Errors(world) == errors + 3);
}

// ------------------------------------------------------------------------------------------- out of memory

// An arena page when recording, the gather block, a creation that fails partway, a batch split by a failure: nothing
// leaks, every relocated value is destroyed once, and the world equals one where the same commands ran immediately
// under the same failure. Each failure is aimed with ecs::SetHeapFailureCountdown after warming everything else that
// the operation would allocate first.
void test_out_of_memory()
{
    const int64_t baseLive = g_Hooks.Live();
    Payload payload;
    {
        auto world = MakeWorld(true, 1024);
        auto reference = MakeWorld(true, 1024);
        const Ids ids = Register(world);
        CHECK(Register(reference).moveOnly == ids.moveOnly);
        ecs::WorldImpl& referenceImpl = Impl(reference);
        auto a = MakeBuffer(world);
        auto b = MakeBuffer(world);
        const ecs::CommandPlayback::Counters& counters = State(world).GetCounters();
        auto same = [&]()
        {
            CHECK(world->ComputeHash() == reference->ComputeHash());
            CHECK(Dump(world, ids) == Dump(reference, ids));
        };
        auto setPositions = [&](const Entity* entities, uint32_t count, const Position& value)
        {
            for (uint32_t i = 0; i < count; ++i)
                CHECK(referenceImpl.SetComponentsInternal(&entities[i], 1, ids.position, &value, ValueTransfer::Copy,
                                                          reference->GetVersion()) == nvrhi::FS_OK);
        };
        Entity e[16];
        Entity r[16];
        CHECK(world->CreateEntities(CreateDesc({ ids.position }, {}, 16), e) == nvrhi::FS_OK);
        CHECK(reference->CreateEntities(CreateDesc({ ids.position }, {}, 16), r) == nvrhi::FS_OK);
        CHECK(std::equal(e, e + 16, r));
        Entity layout;
        CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.velocity, ids.id }, {}, 1), &layout) ==
              nvrhi::FS_OK);
        CHECK(reference->CreateEntities(CreateDesc({ ids.position, ids.velocity, ids.id }, {}, 1), &layout) ==
              nvrhi::FS_OK);

        // 1. An arena page fails when recording: a Failed command, nothing copied, relocated values destroyed.
        {
            auto c = MakeBuffer(world);
            {
                const Handle handle(&payload);
                ecs::SetHeapFailureCountdown(0);
                c->AddComponent(e[0], ids.handle, &handle, 0);
                ecs::SetHeapFailureCountdown(-1);
                CHECK(c->GetCommandCount() == 1 && BufferImpl(c.Get()).GetArena().GetPageCount() == 0);
                CHECK(payload.refs == 1);
            }
            alignas(Handle) unsigned char storage[2 * sizeof(Handle)];
            Handle* handles = reinterpret_cast<Handle*>(storage);
            new (handles) Handle(&payload);
            new (handles + 1) Handle(&payload);
            Entity p[2];
            ecs::SetHeapFailureCountdown(0);
            const CreateDesc relocated({ ids.position, ids.handle }, { nullptr, handles }, 2, ValueTransfer::Relocate);
            c->CreateEntities(relocated, 1, p);
            ecs::SetHeapFailureCountdown(-1);
            CHECK(payload.refs == 0 && g_Hooks.Live() == baseLive);
            CHECK(IsPlaceholder(p[1]) && c->GetCommandCount() == 2);
            // The values block fails after the entity list took the first page. The allocations: the first page, the
            // arena's page list (a container, which would stop the process), then the second page, which fails.
            {
                const std::vector<Entity> many(1500, e[1]);
                const std::vector<Handle> values(1500, Handle(&payload));
                ecs::SetHeapFailureCountdown(2);
                c->AddComponents(many.data(), 1500, ids.handle, values.data(), 2);
                ecs::SetHeapFailureCountdown(-1);
                CHECK(c->GetCommandCount() == 3 && BufferImpl(c.Get()).GetArena().GetPageCount() == 1);
                CHECK(payload.refs == 1500);   // the caller's only
            }
            CHECK(Play(world, { c.Get() }) == nvrhi::FE_OUT_OF_MEMORY);
            CHECK(world->ResolvePlaceholder(p[0]) == Entity::Null);
            same();
        }
        CHECK(payload.refs == 0 && g_Hooks.Live() == baseLive);

        // 2. The gather block cannot grow: the run is applied one command at a time, with the same result.
        const Position p1{ 1, 2, 3 };
        a->SetComponent(e[0], ids.position, &p1, 0);
        b->SetComponent(e[1], ids.position, &p1, 1);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
        setPositions(r, 2, p1);
        CHECK(State(world).GetGatherSize() == 24);
        {
            const Handle handles[4] = { Handle(&payload), Handle(&payload), Handle(&payload), Handle(nullptr) };
            a->AddComponents(e + 2, 2, ids.handle, handles, 0);
            b->AddComponents(e + 4, 2, ids.handle, handles + 2, 1);
            for (uint32_t i = 0; i < 4; ++i)
                CHECK(reference->AddComponent(r[2 + i], ids.handle, &handles[i]) == nvrhi::FS_OK);
        }
        uint64_t batches = counters.batches;
        uint64_t coalesced = counters.coalesced;
        ecs::SetHeapFailureCountdown(0);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
        ecs::SetHeapFailureCountdown(-1);
        CHECK(counters.batches - batches == 2 && counters.coalesced == coalesced);
        CHECK(State(world).GetGatherSize() == 0 && payload.refs == 6);
        same();
        // The gather block at least doubles when it grows.
        const Position p2{ 4, 5, 6 };
        const Position twoValues[2] = { p2, p2 };
        a->SetComponents(e, 2, ids.position, twoValues, 0);
        b->SetComponents(e + 2, 1, ids.position, twoValues, 1);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK && State(world).GetGatherSize() == 36);
        a->SetComponents(e, 2, ids.position, twoValues, 0);
        b->SetComponents(e + 2, 2, ids.position, twoValues, 1);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK && State(world).GetGatherSize() == 72);
        setPositions(r, 4, p2);
        same();

        // 3. A creation fails partway: the entities before the failure exist, the others' placeholders stay null, and
        // a later command that names them fails after the creation's failure.
        Layer one;
        Layer three;
        std::memset(&one, 0, sizeof(one));
        std::memset(&three, 0, sizeof(three));
        one.id = 1;
        three.id = 3;
        {
            const Handle handle(&payload);
            Entity seed;
            const CreateDesc seedDesc({ ids.position, ids.layer, ids.handle }, { nullptr, &one, &handle }, 1);
            CHECK(world->CreateEntities(seedDesc, &seed) == nvrhi::FS_OK);
            CHECK(reference->CreateEntities(seedDesc, &seed) == nvrhi::FS_OK);
        }
        Entity warm[4];
        a->CreateEntities(CreateDesc({ ids.position }, {}, 4), 0, warm);   // four slots in the placeholder table
        b->DestroyEntities(warm, 4, 1);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK);
        CHECK(reference->CreateEntities(CreateDesc({ ids.position }, {}, 4), warm) == nvrhi::FS_OK);
        CHECK(reference->DestroyEntities(warm, 4) == nvrhi::FS_OK);
        {
            const Layer layers[4] = { one, one, three, three };
            const Handle handles[4] = { Handle(&payload), Handle(&payload), Handle(&payload), Handle(&payload) };
            const CreateDesc desc({ ids.position, ids.layer, ids.handle }, { nullptr, layers, handles }, 4);
            Entity p[4];
            a->CreateEntities(desc, 0, p);
            b->AddComponents(p, 4, ids.velocity, nullptr, 1);
            ecs::SetHeapFailureCountdown(0);
            CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_OUT_OF_MEMORY);
            ecs::SetHeapFailureCountdown(-1);
            const Entity c0 = world->ResolvePlaceholder(p[0]);
            const Entity c1 = world->ResolvePlaceholder(p[1]);
            CHECK(world->IsAlive(c0) && world->IsAlive(c1) && world->HasComponent(c1, ids.velocity));
            CHECK(world->ResolvePlaceholder(p[2]) == Entity::Null && world->ResolvePlaceholder(p[3]) == Entity::Null);
            Entity out[4] = {};
            ecs::SetHeapFailureCountdown(0);
            CHECK(reference->CreateEntities(desc, out) == nvrhi::FE_OUT_OF_MEMORY);
            ecs::SetHeapFailureCountdown(-1);
            CHECK(out[0] == c0 && out[1] == c1 && out[2] == Entity::Null);
            for (uint32_t i = 0; i < 2; ++i)
                CHECK(reference->AddComponent(out[i], ids.velocity, nullptr) == nvrhi::FS_OK);
        }
        same();
        CHECK(payload.refs == 2 * (3 + 1 + 2));   // per world: e[2..4], the seed, two created

        // 4. A batch split by a failure. Ten entities move to {Position, Velocity}, whose last chunk has three free
        // rows, while the chunk pools have no free page: the fourth move needs a new chunk and fails, the others apply.
        Entity target;
        CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.velocity }, {}, 1), &target) == nvrhi::FS_OK);
        CHECK(reference->CreateEntities(CreateDesc({ ids.position, ids.velocity }, {}, 1), &target) == nvrhi::FS_OK);
        ArchetypeInfo info{};
        CHECK(world->GetArchetypeInfo(Impl(world).FindAlive(target)->archetype, info) == nvrhi::FS_OK);
        CHECK(info.rowsPerChunk > 8 && info.entityCount == 1);
        std::vector<Entity> rows(info.rowsPerChunk - 4);
        CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.velocity }, {}, uint32_t(rows.size())),
                                    rows.data()) == nvrhi::FS_OK);
        CHECK(reference->CreateEntities(CreateDesc({ ids.position, ids.velocity }, {}, uint32_t(rows.size())),
                                        rows.data()) == nvrhi::FS_OK);
        // The add and remove edges between {Position} and the target, cached before the failure is armed.
        for (IWorld* w : { world.Get(), reference.Get() })
        {
            CHECK(w->AddComponent(e[0], ids.velocity, nullptr) == nvrhi::FS_OK);
            CHECK(w->RemoveComponent(e[0], ids.velocity) == nvrhi::FS_OK);
        }
        const std::vector<Position> tenPositions(10, p1);
        a->SetComponents(e + 6, 5, ids.position, tenPositions.data(), 0);   // a gather block for ten values
        b->SetComponents(e + 11, 5, ids.position, tenPositions.data(), 1);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FS_OK && State(world).GetGatherSize() >= 120);
        setPositions(r + 6, 10, p1);
        for (IWorld* w : { world.Get(), reference.Get() })
        {
            Entity filler;
            while (Impl(w).GetFreeChunkPageCount() > 0)
                CHECK(w->CreateEntities(CreateDesc({ ids.id }, {}, 1), &filler) == nvrhi::FS_OK);
        }
        same();
        std::vector<Velocity> velocities(10);
        for (uint32_t i = 0; i < 10; ++i)
            velocities[i] = Velocity{ float(i), 0, 0 };
        a->AddComponents(e + 6, 5, ids.velocity, velocities.data(), 0);
        b->AddComponents(e + 11, 5, ids.velocity, velocities.data() + 5, 1);
        batches = counters.batches;
        ecs::SetHeapFailureCountdown(0);
        CHECK(Play(world, { a.Get(), b.Get() }) == nvrhi::FE_OUT_OF_MEMORY);
        ecs::SetHeapFailureCountdown(-1);
        CHECK(counters.batches - batches == 1);
        for (uint32_t i = 6; i < 16; ++i)
            CHECK(world->HasComponent(e[i], ids.velocity) == (i != 9));
        CHECK(Get<Velocity>(world, e[10]).x == 4);
        nvrhi::FRESULT result = nvrhi::FS_OK;
        ecs::SetHeapFailureCountdown(0);
        for (uint32_t i = 6; i < 16; ++i)
        {
            const nvrhi::FRESULT added = reference->AddComponent(r[i], ids.velocity, &velocities[i - 6]);
            if (NVRHI_SUCCEEDED(result))
                result = added;
        }
        ecs::SetHeapFailureCountdown(-1);
        CHECK(result == nvrhi::FE_OUT_OF_MEMORY);
        same();
    }
    CHECK(payload.refs == 0 && g_Hooks.Live() == baseLive);
}

// ----------------------------------------------------------------------------------------------- the arena

// A request goes to the current page, else to the first kept page with room: the same sizes in another order need no
// new page here.
void test_arena_reuse()
{
    auto world = MakeWorld();
    Register(world);
    auto a = MakeBuffer(world);
    const ecs::CommandArena& arena = BufferImpl(a.Get()).GetArena();
    const std::vector<Entity> list(3750, Entity::Null);
    a->DestroyEntities(list.data(), 1000, 0);   // 8,000 bytes: the first page (16 KiB)
    a->DestroyEntities(list.data(), 3750, 1);   // 30,000: a second page (32 KiB), now the current one
    CHECK(arena.GetPageCount() == 2);
    a->DestroyEntities(list.data(), 625, 2);    // 5,000: no room in the current page, room in the first
    CHECK(arena.GetPageCount() == 2);
    a->Reset();
    a->DestroyEntities(list.data(), 3750, 0);
    a->DestroyEntities(list.data(), 1000, 1);
    a->DestroyEntities(list.data(), 625, 2);
    CHECK(arena.GetPageCount() == 2 && arena.GetCapacity() == 48 * 1024);
    a->Reset();
}

// ------------------------------------------------------------------------------- steady-state allocations

void test_steady_state_allocations()
{
    auto world = MakeWorld(false);
    const Ids ids = Register(world);
    constexpr uint32_t c_Workers = 4;
    std::vector<Entity> base(c_Workers * 64);
    CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.id, ids.toggle, ids.joint }, {}, c_Workers * 64),
                                base.data()) == nvrhi::FS_OK);
    Entity prefab;
    CHECK(world->CreateEntities(CreateDesc({ ids.position, ids.id }, {}, 1), &prefab) == nvrhi::FS_OK);
    std::vector<nvrhi::AutoPtr<ICommandBuffer>> buffers;
    ICommandBuffer* list[c_Workers];
    for (uint32_t w = 0; w < c_Workers; ++w)
    {
        buffers.push_back(MakeBuffer(world));
        list[w] = buffers.back().Get();
    }

    // Everything a round needs is allocated here and in the warm-up rounds, never in the measured ones.
    Entity created[c_Workers][16];
    Entity instances[c_Workers][4];
    std::vector<Position> positions(64, Position{ 1, 2, 3 });
    std::vector<Velocity> velocities(64, Velocity{ 4, 5, 6 });
    std::vector<Id> idValues(16, Id{ 7 });
    Layer firsts[16];
    Layer seconds[16];
    std::memset(firsts, 0, sizeof(firsts));
    std::memset(seconds, 0, sizeof(seconds));
    for (uint32_t i = 0; i < 16; ++i)
    {
        firsts[i].id = 1;
        seconds[i].id = 2;
    }
    Joint joints[2] = { { Entity::Null, 1, 0 }, { Entity::Null, 2, 0 } };
    const CreateDesc desc({ ids.position, ids.id }, { positions.data(), idValues.data() }, 16);
    WorldStats stats;
    uint64_t sink = 0;

    auto round = [&](uint32_t r)
    {
        for (uint32_t w = 0; w < c_Workers; ++w)
        {
            ICommandBuffer* buffer = list[w];
            const Entity* slice = base.data() + w * 64;
            buffer->CreateEntities(desc, MakeSortKey(0, w, 0), created[w]);
            buffer->Instantiate(prefab, 4, MakeSortKey(0, w, 1), instances[w]);
            buffer->AddComponents(slice, 64, ids.velocity, velocities.data(), MakeSortKey(1, w, 0));
            buffer->SetComponents(slice, 64, ids.position, positions.data(), MakeSortKey(2, w, 0));
            buffer->SetEnabled(slice, 64, ids.toggle, (r & 1) != 0, MakeSortKey(3, w, 0));
            joints[0].target = created[w][0];
            buffer->AppendToBuffer(slice[0], ids.joint, joints, 2, MakeSortKey(4, w, 0));
            buffer->AddComponents(slice, 16, ids.layer, seconds, MakeSortKey(5, w, 0));
            buffer->AddComponents(slice, 16, ids.layer, firsts, MakeSortKey(6, w, 0));
            buffer->AddComponents(slice, 64, ids.selected, nullptr, MakeSortKey(7, w, 0));
            buffer->RemoveComponents(slice, 64, ids.selected, MakeSortKey(8, w, 0));
            buffer->RemoveComponents(slice, 64, ids.velocity, MakeSortKey(9, w, 0));
            buffer->DestroyEntities(created[w], 16, MakeSortKey(10, w, 0));
            buffer->DestroyEntities(instances[w], 4, MakeSortKey(10, w, 1));
        }
        CHECK(world->Playback(list, c_Workers) == nvrhi::FS_OK);
        for (uint32_t w = 0; w < c_Workers; ++w)
        {
            sink += GetBufferData<Joint>(HeaderOf(world, base[w * 64], ids.joint))[0].weight == 1 ? 1 : 0;
            CHECK(world->ResizeBuffer(base[w * 64], ids.joint, 0, c_WorldVersion) == nvrhi::FS_OK);
            sink += uint64_t(world->ResolvePlaceholder(created[w][0]));   // destroyed again: null
        }
        world->AdvanceVersion();
        world->GetStats(stats);
    };

    for (uint32_t r = 0; r < 3; ++r)
        round(r);
    const uint64_t news = g_NewCount.load();
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount();
    for (uint32_t r = 3; r < 8; ++r)
        round(r);
    const uint64_t newsAfter = g_NewCount.load();
    const uint64_t heapBlocksAfter = ecs::GetHeapAllocationCount();
    printf("test_ecs_commands: steady state: %llu operator new, %llu storage blocks in 5 rounds of %u commands "
           "(sink %llu)\n", (unsigned long long)(newsAfter - news), (unsigned long long)(heapBlocksAfter - heapBlocks),
           c_Workers * 13, (unsigned long long)(sink & 1));
    CHECK(newsAfter == news);
    CHECK(heapBlocksAfter == heapBlocks);
    CHECK(stats.entityCount == c_Workers * 64 + 1);

    // Both counters see allocations when there are some: an operator new, and a buffer's arena growing.
    int* probe = new int[16];
    sink += uintptr_t(probe) & 1;
    delete[] probe;
    CHECK(g_NewCount.load() > newsAfter);
    std::vector<Entity> wide(100000, base[0]);
    list[0]->DestroyEntities(wide.data(), uint32_t(wide.size()), 0);   // 800 KB: a page larger than any so far
    CHECK(ecs::GetHeapAllocationCount() > heapBlocksAfter);
    list[0]->Reset();
}

int main(int, char**)
{
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "basics", &test_basics },
        { "placeholder_wrap", &test_placeholder_wrap },
        { "batching", &test_batching },
        { "ownership", &test_ownership },
        { "model", &test_model },
        { "determinism", &test_determinism },
        { "validation", &test_validation },
        { "first_failure", &test_first_failure },
        { "versions", &test_versions },
        { "not_held", &test_not_held },
        { "registration_in_phases", &test_registration_in_phases },
        { "out_of_memory", &test_out_of_memory },
        { "arena_reuse", &test_arena_reuse },
        { "steady_state_allocations", &test_steady_state_allocations },
    };
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash keeps the names of the cases that passed
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    try
    {
        for (const Test& test : tests)
        {
            test.run();
            printf("test_ecs_commands: %s passed\n", test.name);
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);   // every world and buffer freed every block
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
