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

// The ECS world's storage (aggregate docs/plans/2026-11-21-donut-ecs.md, step E1.1; design 4.1-4.5, 4.9, 4.10): the
// registry, the entity table, archetypes and chunks, sparse sets, dynamic buffers, hooks, Instantiate,
// CopyEntitiesFrom, resources, determinism, and an allocation counter that proves steady-state operations allocate
// nothing. The world's internals are read through its private header (src/engine/EcsWorldImpl.h).

#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsWorldImpl.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <new>
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

namespace ecs_test
{
    using donut::engine::ComponentFlags;
    using donut::engine::Entity;

    struct Position { float x, y, z; };
    DONUT_ECS_COMPONENT(Position, "d0f1863f-66f0-4e69-9d38-8af8de219938", DONUT_ECS_FIELDS(x, y, z));

    struct Velocity { float x, y, z; };
    DONUT_ECS_COMPONENT(Velocity, "82e2f9e0-7a0e-4bc1-8ddb-e871d606f569", DONUT_ECS_FIELDS(x, y, z));

    struct Id { uint64_t value; };
    DONUT_ECS_COMPONENT(Id, "186d8233-6ed3-420e-89ad-c1235b73a216", DONUT_ECS_FIELDS(value));

    // A row over 128 bytes: 64 KiB chunks (D6).
    struct Big { uint8_t bytes[200]; };
    DONUT_ECS_COMPONENT(Big, "fa205067-710f-42ee-87b7-b48097c80a88", DONUT_ECS_FIELDS(bytes));

    struct alignas(64) Wide { float v[16]; };
    DONUT_ECS_COMPONENT(Wide, "595df89e-58c3-4527-af94-cdc270a648ba", DONUT_ECS_FIELDS(v));

    struct Frozen { };
    DONUT_ECS_TAG(Frozen, "22dd1e49-1210-4faa-b8d3-cb3b2259d12c");

    // Three bytes of padding after `id`: never part of the key.
    struct Layer { uint8_t id; uint32_t mask; };
    DONUT_ECS_COMPONENT_EX(Layer, "6a9cb1cc-bd80-4ebf-8cc4-11e247620445", ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(id, mask));

    struct ChunkBounds { float lo, hi; };
    DONUT_ECS_COMPONENT_EX(ChunkBounds, "199b3b4f-62f2-45dd-9ed4-6f5de11cf080", ComponentFlags::Chunk, 0,
                           DONUT_ECS_FIELDS(lo, hi));

    struct Selected { };
    DONUT_ECS_TAG_EX(Selected, "a5d30c84-a3d6-4c93-9297-798984155120", ComponentFlags::Sparse);

    struct DebugColor { uint32_t rgba; };
    DONUT_ECS_COMPONENT_EX(DebugColor, "a89a9b1c-f68d-4dce-b24a-168a7febfa3c", ComponentFlags::Sparse, 0,
                           DONUT_ECS_FIELDS(rgba));

    struct Toggle { int32_t value; };
    DONUT_ECS_COMPONENT_EX(Toggle, "99e082fb-f342-4d4e-8d50-c0ae969ebc6c", ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(value));

    struct Hidden { };
    DONUT_ECS_TAG_EX(Hidden, "14b7646e-cf0e-48bd-a569-b5abcd63a6a9", ComponentFlags::Enableable);

    struct Joint { Entity target; float weight; uint32_t pad; };
    DONUT_ECS_BUFFER(Joint, "2c0f2285-f5b1-4335-80b7-94e2ae5c72d1", 4, DONUT_ECS_FIELDS(target, weight));

    struct Link { Entity other; Entity outside; };
    DONUT_ECS_COMPONENT(Link, "6ccb8140-7b66-4dfc-b026-c061c63f74f9", DONUT_ECS_FIELDS(other, outside));

    // A non-trivial component that holds a reference to a counted payload; every hook call is counted.
    struct HookCounts
    {
        int64_t constructs = 0;
        int64_t destructs = 0;
        int64_t copies = 0;
        int64_t moves = 0;
        int64_t Live() const { return constructs + copies + moves - destructs; }
    };
    HookCounts g_Hooks;

    struct Payload { int64_t refs = 0; };

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
    DONUT_ECS_COMPONENT(Handle, "ba4b1505-0cd1-4d1a-86ec-d6ba8048b122", DONUT_ECS_FIELDS());

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
    DONUT_ECS_COMPONENT(MoveOnly, "ebf8a8d8-fab8-4501-a85a-3065571fe1ef", DONUT_ECS_FIELDS(value));

    // Trivially copyable, so Relocatable, yet with a construct hook (the default member initialiser).
    struct Defaulted { int32_t value = 7; };
    DONUT_ECS_COMPONENT(Defaulted, "7a0939b1-e7de-464d-9690-775be8c2169e", DONUT_ECS_FIELDS(value));
}

using namespace donut::engine;
using namespace ecs_test;
namespace ecs = donut::engine::ecs;

#define CHECK_HOOKS(base, c, cp, mv, d)                    \
    CHECK(g_Hooks.constructs - (base).constructs == (c));  \
    CHECK(g_Hooks.copies - (base).copies == (cp));         \
    CHECK(g_Hooks.moves - (base).moves == (mv));           \
    CHECK(g_Hooks.destructs - (base).destructs == (d))

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

    nvrhi::AutoPtr<IWorld> MakeWorld(bool validation = false, uint32_t initialEntityCapacity = 1024)
    {
        WorldDesc desc;
        desc.debugName = "test_ecs_world";
        desc.validation = validation;
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

    const ecs::EntityRecord& Record(IWorld* world, Entity entity)
    {
        const ecs::EntityRecord* record = Impl(world).FindAlive(entity);
        CHECK(record != nullptr);
        return *record;
    }

    uint32_t ArchetypeOf(IWorld* world, Entity entity)
    {
        return Record(world, entity).archetype;
    }

    template <typename T>
    const T& Get(IWorld* world, Entity entity)
    {
        const T* value = static_cast<const T*>(world->GetComponent(entity, FindComponent<T>(world)));
        CHECK(value != nullptr);
        return *value;
    }

    // An EntityCreateDesc that owns its component and value arrays, so that it may outlive the braced lists it was
    // built from.
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

    Entity Create(IWorld* world, std::initializer_list<ComponentId> components,
                  std::initializer_list<const void*> values = {})
    {
        const CreateDesc desc(components, values, 1);
        Entity entity = Entity::Null;
        CHECK(world->CreateEntities(desc, &entity) == nvrhi::FS_OK);
        return entity;
    }

    ArchetypeInfo InfoOf(IWorld* world, Entity entity)
    {
        ArchetypeInfo info{};
        CHECK(world->GetArchetypeInfo(ArchetypeOf(world, entity), info) == nvrhi::FS_OK);
        return info;
    }

    const BufferHeader& HeaderOf(IWorld* world, Entity entity, ComponentId buffer)
    {
        const BufferHeader* header = static_cast<const BufferHeader*>(world->GetComponent(entity, buffer));
        CHECK(header != nullptr);
        return *header;
    }

    uint8_t* InlineOf(IWorld* world, Entity entity, ComponentId buffer)
    {
        const ecs::EntityRecord& record = Record(world, entity);
        const ecs::Archetype& archetype = Impl(world).GetArchetypeStorage(record.archetype);
        return archetype.InlineData(record.chunk, archetype.FindColumn(buffer), record.row);
    }

    const uint64_t* VersionsOf(IWorld* world, Entity entity)
    {
        const ecs::EntityRecord& record = Record(world, entity);
        return Impl(world).GetArchetypeStorage(record.archetype).Versions(record.chunk);
    }

    uint32_t ColumnOf(IWorld* world, Entity entity, ComponentId component)
    {
        const uint32_t column = Impl(world).GetArchetypeStorage(ArchetypeOf(world, entity)).FindColumn(component);
        CHECK(column != ecs::c_NoColumn);
        return column;
    }

    constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // The layout rule of design 4.3 for archetypes of Data columns only, by brute force: the Entity column, each column
    // 64-byte aligned, then one 64-bit version per column plus the added version.
    uint32_t ExpectedRows(uint32_t chunkSize, std::initializer_list<uint32_t> strides)
    {
        uint32_t best = 0;
        for (uint32_t rows = 1; rows <= 65536; ++rows)
        {
            uint64_t offset = uint64_t(rows) * 8;
            for (uint32_t stride : strides)
                offset = AlignUp(offset, 64) + uint64_t(rows) * stride;
            offset = AlignUp(offset, 8) + (strides.size() + 1) * 8;
            if (offset > chunkSize)
                break;
            best = rows;
        }
        return best;
    }

    // `count` Handles that reference `payload`, in raw storage that a Relocate transfer gives up: nothing here destroys
    // them, the world must.
    struct RawHandles
    {
        alignas(Handle) unsigned char bytes[8 * sizeof(Handle)];

        RawHandles(Payload* payload, uint32_t count)
        {
            CHECK(count <= 8);
            for (uint32_t i = 0; i < count; ++i)
                new (bytes + i * sizeof(Handle)) Handle(payload);
        }
        const void* Get() const { return bytes; }
    };

    // 944 bytes: one row of it alone fills a 1 KiB chunk (64 before it, 16 bytes of versions after), so that a 1 KiB
    // world fits nothing more beside it.
    ComponentTypeDesc FillDesc()
    {
        ComponentTypeDesc desc{};
        desc.uuid = "b57433a2-9cc7-41b5-9f73-34d1b41e4fbd"_nvrhi_guid;
        desc.name = "Fill944";
        desc.size = 944;
        desc.alignment = 8;
        desc.flags = ComponentFlags::Relocatable;
        return desc;
    }

    nvrhi::AutoPtr<IWorld> MakeSizedWorld(uint32_t chunkSize, uint32_t largeChunkSize)
    {
        WorldDesc desc;
        desc.debugName = "test_ecs_world_sized";
        desc.chunkSize = chunkSize;
        desc.largeChunkSize = largeChunkSize;
        desc.validation = true;
        nvrhi::AutoPtr<IWorld> world;
        CHECK(CreateWorld(desc, &world) == nvrhi::FS_OK);
        return world;
    }

    bool Aligned(const void* pointer, size_t alignment)
    {
        return (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
    }
}

// ------------------------------------------------------------------------------------------------ registry

void test_registration()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId position = RegisterComponent<Position>(world);
    const ComponentId velocity = RegisterComponent<Velocity>(world);
    CHECK(position == ComponentId(0) && velocity == ComponentId(1));   // dense, in registration order
    CHECK(RegisterComponent<Position>(world) == position);              // the same GUID again
    CHECK(FindComponent<Velocity>(world) == velocity);
    CHECK(world->FindComponent("e4511dbb-6ddb-470b-a9d6-3916e5fe3a7b"_nvrhi_guid) == ComponentId::Invalid);
    CHECK(world->GetComponentTypeCount() == 2);

    // The world keeps its own copy of the descriptor.
    const ComponentTypeDesc* desc = world->GetComponentTypeDesc(position);
    CHECK(desc && desc != &ComponentTraits<Position>::Desc());
    CHECK(desc->size == sizeof(Position) && desc->fieldCount == 3 && std::strcmp(desc->name, "Position") == 0);
    CHECK(!!(desc->flags & ComponentFlags::Relocatable));
    CHECK(world->GetComponentTypeDesc(ComponentId(7)) == nullptr);

    // The GUID of Position with another layout.
    const uint64_t errors = impl.GetValidationErrorCount();
    ComponentTypeDesc clash = ComponentTraits<Position>::Desc();
    clash.size = 16;
    ComponentId id = ComponentId(0);
    CHECK(world->RegisterComponent(clash, id) == nvrhi::FE_INVALID_ARGS && id == ComponentId::Invalid);
    clash = ComponentTraits<Position>::Desc();
    clash.flags |= ComponentFlags::Enableable;
    CHECK(world->RegisterComponent(clash, id) == nvrhi::FE_INVALID_ARGS);
    clash = ComponentTraits<Position>::Desc();
    FieldDesc moved[3] = { clash.fields[0], clash.fields[1], clash.fields[2] };
    moved[2].offset = 4;
    clash.fields = moved;
    CHECK(world->RegisterComponent(clash, id) == nvrhi::FE_INVALID_ARGS);
    CHECK(impl.GetValidationErrorCount() == errors + 3);

    // Invalid descriptors: rejected, nothing registered.
    ComponentTypeDesc base = ComponentTraits<Velocity>::Desc();
    base.uuid = "e4511dbb-6ddb-470b-a9d6-3916e5fe3a7b"_nvrhi_guid;
    auto reject = [&](const ComponentTypeDesc& d)
    {
        ComponentId out = ComponentId(0);
        CHECK(world->RegisterComponent(d, out) == nvrhi::FE_INVALID_ARGS && out == ComponentId::Invalid);
    };
    { ComponentTypeDesc d = base; d.uuid = nvrhi::GUID{}; reject(d); }
    { ComponentTypeDesc d = base; d.alignment = 3; reject(d); }
    { ComponentTypeDesc d = base; d.alignment = 128; d.size = 128; d.fieldCount = 0; reject(d); }
    { ComponentTypeDesc d = base; d.size = 0; d.fieldCount = 0; reject(d); }                   // size 0 without Tag
    { ComponentTypeDesc d = base; d.flags |= ComponentFlags::Tag; reject(d); }                  // a Tag with data
    {
        ComponentTypeDesc d = base;   // a Buffer of tags
        d.size = 0;
        d.fieldCount = 0;
        d.flags = ComponentFlags::Tag | ComponentFlags::Buffer;
        reject(d);
    }
    { ComponentTypeDesc d = base; d.flags |= ComponentFlags::Shared | ComponentFlags::Sparse; reject(d); }
    { ComponentTypeDesc d = base; d.flags |= ComponentFlags::Shared; d.fieldCount = 0; reject(d); }
    { ComponentTypeDesc d = base; d.flags |= ComponentFlags::Enableable | ComponentFlags::Chunk; reject(d); }
    { ComponentTypeDesc d = base; d.flags = ComponentFlags::GpuMirrored; reject(d); }           // not Relocatable
    { ComponentTypeDesc d = base; d.bufferInlineCapacity = 4; reject(d); }                      // not a Buffer
    { ComponentTypeDesc d = base; d.flags = ComponentFlags(1u << 20); reject(d); }              // unknown flag
    {
        const FieldDesc outside[] = { { "x", 8, 2, FieldType::Float, FieldShape::Vector, {} } };
        ComponentTypeDesc d = base;
        d.fields = outside;
        d.fieldCount = 1;
        reject(d);
    }
    CHECK(world->GetComponentTypeCount() == 2);
    CHECK(impl.GetValidationErrorCount() == errors + 3 + 13);

    // GpuMirrored with Relocatable is accepted.
    {
        ComponentTypeDesc d = base;
        d.uuid = "28bf395d-90cf-4008-b5a0-db50628f5cc5"_nvrhi_guid;
        d.flags |= ComponentFlags::GpuMirrored;
        ComponentId out = ComponentId::Invalid;
        CHECK(world->RegisterComponent(d, out) == nvrhi::FS_OK && out == ComponentId(2));
    }

    // The field bytes that Shared keys compare: merged and ascending, padding excluded.
    const ComponentId layer = RegisterComponent<Layer>(world);
    const ecs::ComponentType* layerType = impl.GetType(layer);
    CHECK(layerType->kind == ecs::StorageKind::Shared && layerType->fieldBytes.size() == 2);
    CHECK(layerType->fieldBytes[0].offset == 0 && layerType->fieldBytes[0].size == 1);
    CHECK(layerType->fieldBytes[1].offset == 4 && layerType->fieldBytes[1].size == 4);
    const ComponentId link = RegisterComponent<Link>(world);
    const ecs::ComponentType* linkType = impl.GetType(link);
    CHECK(linkType->fieldBytes.size() == 1 && linkType->fieldBytes[0].size == 16);
    CHECK(linkType->entityOffsets.size() == 2 && linkType->entityOffsets[0] == 0 && linkType->entityOffsets[1] == 8);
}

// ------------------------------------------------------------------------------------------- entity table

void test_entity_lifecycle()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);

    const Id values[3] = { { 1 }, { 2 }, { 3 } };
    Entity e[3];
    const CreateDesc desc({ id }, { values }, 3);
    CHECK(world->CreateEntities(desc, e) == nvrhi::FS_OK);
    CHECK(e[0] == MakeEntity(1, ecs::c_FirstGeneration));   // index 0 is never handed out
    CHECK(e[1] == MakeEntity(2, ecs::c_FirstGeneration) && e[2] == MakeEntity(3, ecs::c_FirstGeneration));
    CHECK(world->IsAlive(e[0]) && world->IsAlive(e[2]));
    CHECK(!world->IsAlive(Entity::Null) && !world->IsAlive(MakeEntity(0, 1)) && !world->IsAlive(MakeEntity(1, 2)));
    CHECK(!world->IsAlive(MakeEntity(99, 1)) && !world->IsAlive(MakeEntity(1, c_PlaceholderGeneration)));
    CHECK(Get<Id>(world, e[1]).value == 2);
    CHECK(impl.GetEntityRecord(0).archetype == ecs::c_FreeSlot);

    const Entity stale = e[1];
    CHECK(world->DestroyEntities(&stale, 1) == nvrhi::FS_OK);
    CHECK(!world->IsAlive(stale));

    // A stale id fails everywhere, and validation reports each use.
    const uint64_t errors = impl.GetValidationErrorCount();
    CHECK(world->GetComponent(stale, id) == nullptr);
    CHECK(!world->HasComponent(stale, id));
    CHECK(world->GetComponentForWrite(stale, id, c_WorldVersion) == nullptr);
    CHECK(world->AddComponent(stale, position, nullptr) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->RemoveComponent(stale, id) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->DestroyEntities(&stale, 1) == nvrhi::FE_INVALID_ARGS);
    CHECK(!world->IsEnabled(stale, id));
    CHECK(impl.GetValidationErrorCount() == errors + 7);

    // A batch with a stale id still destroys the others.
    const Entity batch[3] = { e[0], stale, e[2] };
    CHECK(world->DestroyEntities(batch, 3) == nvrhi::FE_INVALID_ARGS);
    CHECK(!world->IsAlive(e[0]) && !world->IsAlive(e[2]));
    WorldStats stats;
    CHECK(world->GetStats(stats).entityCount == 0);

    // An entity without components lives in archetype 0.
    const Entity empty = Create(world, {});
    CHECK(ArchetypeOf(world, empty) == 0 && world->GetStats(stats).entityCount == 1);
}

void test_free_list_and_generations()
{
    CHECK(ecs::NextGeneration(1) == 2);
    CHECK(ecs::NextGeneration(c_PlaceholderGeneration - 1) == ecs::c_FirstGeneration);   // skips the placeholder
    CHECK(ecs::NextGeneration(0) == 1);

    auto world = MakeWorld();
    const ComponentId id = RegisterComponent<Id>(world);
    Entity e[4];
    const CreateDesc desc({ id }, {}, 4);
    CHECK(world->CreateEntities(desc, e) == nvrhi::FS_OK);

    // FIFO: indices come back in the order they were freed, with the next generation.
    const Entity order[3] = { e[1], e[3], e[0] };
    CHECK(world->DestroyEntities(order, 3) == nvrhi::FS_OK);
    Entity f[4];
    CHECK(world->CreateEntities(desc, f) == nvrhi::FS_OK);
    CHECK(f[0] == MakeEntity(2, 2) && f[1] == MakeEntity(4, 2) && f[2] == MakeEntity(1, 2));
    CHECK(f[3] == MakeEntity(5, 1));   // the free list ran out: a new index
    CHECK(!world->IsAlive(e[1]) && world->IsAlive(f[0]));

    // An index freed later waits behind the ones freed before it.
    CHECK(world->DestroyEntities(&f[0], 1) == nvrhi::FS_OK);
    CHECK(world->DestroyEntities(&f[2], 1) == nvrhi::FS_OK);
    Entity g = Entity::Null;
    CHECK(world->CreateEntities(CreateDesc({ id }, {}, 1), &g) == nvrhi::FS_OK);
    CHECK(g == MakeEntity(2, 3));
    CHECK(world->CreateEntities(CreateDesc({ id }, {}, 1), &g) == nvrhi::FS_OK);
    CHECK(g == MakeEntity(1, 3));
}

void test_reserved_entities()
{
    auto world = MakeWorld(true);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId link = RegisterComponent<Link>(world);

    Entity r[4];
    CHECK(world->ReserveEntities(4, r) == nvrhi::FS_OK);
    for (const Entity entity : r)
        CHECK(!world->IsAlive(entity));
    WorldStats stats;
    CHECK(world->GetStats(stats).entityCount == 0);

    // Two rows that name each other, created in one call.
    const Id ids[2] = { { 10 }, { 11 } };
    const Link links[2] = { { r[1], Entity::Null }, { r[0], Entity::Null } };
    CreateDesc desc({ id, link }, { ids, links }, 2);
    desc->reservedEntities = r;
    CHECK(world->CreateEntities(desc, nullptr) == nvrhi::FS_OK);
    CHECK(world->IsAlive(r[0]) && world->IsAlive(r[1]));
    CHECK(Get<Link>(world, r[0]).other == r[1] && Get<Link>(world, r[1]).other == r[0]);
    CHECK(Get<Id>(world, r[1]).value == 11);

    // A used id cannot be used again; the good ids of a failed call stay reserved.
    const Entity again[2] = { r[0], r[2] };
    desc->reservedEntities = again;
    CHECK(world->CreateEntities(desc, nullptr) == nvrhi::FE_INVALID_ARGS);
    const Entity twice[2] = { r[2], r[2] };
    desc->reservedEntities = twice;
    desc->values = nullptr;
    CHECK(world->CreateEntities(desc, nullptr) == nvrhi::FE_INVALID_ARGS);
    CHECK(!world->IsAlive(r[2]));
    CreateDesc one({ id }, {}, 1);
    one->reservedEntities = &r[2];
    Entity out = Entity::Null;
    CHECK(world->CreateEntities(one, &out) == nvrhi::FS_OK && out == r[2] && world->IsAlive(r[2]));

    // A reservation given back cannot be created, and its index is reused with the next generation.
    CHECK(world->DestroyEntities(&r[3], 1) == nvrhi::FS_OK);
    one->reservedEntities = &r[3];
    CHECK(world->CreateEntities(one, nullptr) == nvrhi::FE_INVALID_ARGS);
    const Entity fresh = Create(world, { id });
    CHECK(fresh == MakeEntity(GetEntityIndex(r[3]), GetEntityGeneration(r[3]) + 1));
}

// ------------------------------------------------------------------------------------------- archetypes

void test_archetype_moves_and_edges()
{
    auto world = MakeWorld();
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);
    const ComponentId velocity = RegisterComponent<Velocity>(world);
    const ComponentId frozen = RegisterComponent<Frozen>(world);
    CHECK(world->GetArchetypeCount() == 1);   // the archetype without components

    const Id va{ 10 };
    const Entity a = Create(world, { id }, { &va });
    const uint32_t idOnly = ArchetypeOf(world, a);
    const Position p{ 1, 2, 3 };
    CHECK(world->AddComponent(a, position, &p) == nvrhi::FS_OK);
    const uint32_t withPosition = ArchetypeOf(world, a);
    CHECK(idOnly == 1 && withPosition == 2 && world->GetArchetypeCount() == 3);   // creation order
    CHECK(Get<Id>(world, a).value == 10 && Get<Position>(world, a).z == 3);

    // The move cached the edge both ways.
    const ecs::Archetype& from = impl.GetArchetypeStorage(idOnly);
    const ecs::Archetype& to = impl.GetArchetypeStorage(withPosition);
    CHECK(from.addEdges.size() == 1 && from.addEdges[0].component == position);
    CHECK(from.addEdges[0].archetype == withPosition);
    CHECK(to.removeEdges.size() == 1 && to.removeEdges[0].component == position);
    CHECK(to.removeEdges[0].archetype == idOnly);

    // A second entity takes the cached edges: no archetype lookup.
    const Id vb{ 20 };
    const Entity b = Create(world, { id }, { &vb });
    const uint64_t lookups = impl.GetCounters().archetypeLookups;
    const uint64_t hits = impl.GetCounters().edgeHits;
    CHECK(world->AddComponent(b, position, nullptr) == nvrhi::FS_OK);
    CHECK(impl.GetCounters().archetypeLookups == lookups && impl.GetCounters().edgeHits == hits + 1);
    CHECK(ArchetypeOf(world, b) == withPosition && world->GetArchetypeCount() == 3);
    CHECK(Get<Position>(world, b).x == 0 && Get<Position>(world, b).y == 0);   // zero-filled without a hook
    CHECK(world->RemoveComponent(b, position) == nvrhi::FS_OK);
    CHECK(impl.GetCounters().archetypeLookups == lookups && impl.GetCounters().edgeHits == hits + 2);
    CHECK(ArchetypeOf(world, b) == idOnly && Get<Id>(world, b).value == 20);

    // A tag joins the component set without a column.
    CHECK(world->AddComponent(a, frozen, nullptr) == nvrhi::FS_OK);
    CHECK(world->HasComponent(a, frozen) && world->GetComponent(a, frozen) != nullptr);
    CHECK(Get<Position>(world, a).y == 2 && world->GetArchetypeCount() == 4);

    // Adding a component the entity has writes the value; removing one it lacks does nothing.
    const uint32_t current = ArchetypeOf(world, a);
    const uint64_t structural = world->GetStructuralVersion();
    const Position q{ 4, 5, 6 };
    CHECK(world->AddComponent(a, position, &q) == nvrhi::FS_OK);
    CHECK(ArchetypeOf(world, a) == current && Get<Position>(world, a).x == 4);
    CHECK(world->RemoveComponent(a, velocity) == nvrhi::FS_OK && ArchetypeOf(world, a) == current);
    CHECK(world->GetStructuralVersion() == structural);

    ArchetypeInfo info{};
    CHECK(world->GetArchetypeInfo(current, info) == nvrhi::FS_OK);
    CHECK(info.componentCount == 3 && info.entityCount == 1 && info.chunkCount == 1);
    CHECK(info.components[0] == id && info.components[1] == position && info.components[2] == frozen);
    CHECK(world->GetArchetypeInfo(999, info) == nvrhi::FE_INVALID_ARGS);
}

// Swap-remove keeps every chunk but the last full, and the row order follows from the order of changes alone.
void test_swap_remove_density()
{
    auto world = MakeWorld();
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);

    std::vector<Entity> model;       // the archetype's rows, in order
    std::vector<uint64_t> modelIds;
    uint64_t serial = 0;
    uint32_t archetype = ecs::c_NoArchetype;

    auto create = [&](uint32_t count)
    {
        std::vector<Id> ids(count);
        std::vector<Position> positions(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            ids[i].value = serial++;
            positions[i] = Position{ float(ids[i].value), 0, 0 };
        }
        std::vector<Entity> out(count);
        const CreateDesc desc({ id, position }, { ids.data(), positions.data() }, count);
        CHECK(world->CreateEntities(desc, out.data()) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < count; ++i)
        {
            model.push_back(out[i]);
            modelIds.push_back(ids[i].value);
        }
        archetype = ArchetypeOf(world, out[0]);
    };
    auto destroy = [&](std::vector<uint32_t> positions)
    {
        std::vector<Entity> list;
        for (uint32_t p : positions)
            list.push_back(model[p]);
        CHECK(world->DestroyEntities(list.data(), uint32_t(list.size())) == nvrhi::FS_OK);
        for (const Entity entity : list)
        {
            const size_t p = size_t(std::find(model.begin(), model.end(), entity) - model.begin());
            model[p] = model.back();
            modelIds[p] = modelIds.back();
            model.pop_back();
            modelIds.pop_back();
        }
    };
    auto verify = [&]()
    {
        const ecs::Archetype& a = impl.GetArchetypeStorage(archetype);
        const uint32_t rows = a.rowsPerChunk;
        CHECK(a.entityCount == model.size());
        CHECK(a.GetChunkCount() == (model.size() + rows - 1) / rows);
        for (uint32_t c = 0; c + 1 < a.GetChunkCount(); ++c)
            CHECK(a.chunks[c].count == rows);
        for (size_t i = 0; i < model.size(); ++i)
        {
            const uint32_t chunk = uint32_t(i / rows);
            const uint32_t row = uint32_t(i % rows);
            CHECK(a.Entities(chunk)[row] == model[i]);
            const ecs::EntityRecord& record = Record(world, model[i]);
            CHECK(record.archetype == archetype && record.chunk == chunk && record.row == row);
            CHECK(Get<Id>(world, model[i]).value == modelIds[i]);
            CHECK(Get<Position>(world, model[i]).x == float(modelIds[i]));
        }
    };

    create(2000);
    verify();
    Xorshift rng{ 0x1234567887654321ull };
    for (uint32_t step = 0; step < 1500; ++step)
    {
        const uint32_t r = rng.Below(10);
        if (r < 3 || model.size() < 10)
        {
            create(1 + rng.Below(40));
        }
        else
        {
            // One to three distinct rows, destroyed in one call, in list order.
            std::vector<uint32_t> positions;
            const uint32_t count = 1 + rng.Below(3);
            while (positions.size() < count)
            {
                const uint32_t p = rng.Below(uint32_t(model.size()));
                if (std::find(positions.begin(), positions.end(), p) == positions.end())
                    positions.push_back(p);
            }
            destroy(positions);
        }
        if (step % 100 == 0)
            verify();
    }
    verify();
    while (!model.empty())
        destroy({ 0 });
    verify();
    CHECK(impl.GetArchetypeStorage(archetype).GetChunkCount() == 0);   // the last chunk went with the last row
}

void test_chunk_sizes_and_alignment()
{
    auto world = MakeWorld();
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);
    const ComponentId big = RegisterComponent<Big>(world);
    const ComponentId wide = RegisterComponent<Wide>(world);
    const ComponentId joint = RegisterComponent<Joint>(world);
    const ComponentId toggle = RegisterComponent<Toggle>(world);

    // D6: 16 KiB, or 64 KiB for rows over 128 bytes.
    ArchetypeInfo info = InfoOf(world, Create(world, { position }));
    CHECK(info.chunkSize == 16384 && info.rowSize == 8 + 12);
    CHECK(info.rowsPerChunk == 816 && info.rowsPerChunk == ExpectedRows(16384, { 12 }));
    info = InfoOf(world, Create(world, { big }));
    CHECK(info.chunkSize == 65536 && info.rowSize == 8 + 200);
    CHECK(info.rowsPerChunk == 314 && info.rowsPerChunk == ExpectedRows(65536, { 200 }));
    info = InfoOf(world, Create(world, { id, position }));
    CHECK(info.chunkSize == 16384 && info.rowsPerChunk == ExpectedRows(16384, { 8, 12 }));
    CHECK(world->GetArchetypeInfo(0, info) == nvrhi::FS_OK);
    CHECK(info.chunkSize == 16384 && info.rowSize == 8 && info.rowsPerChunk == 2047);   // 8 bytes a row, one version

    // The threshold: a row of exactly 128 bytes keeps the small chunk.
    ComponentTypeDesc blob{};
    blob.uuid = "82638eec-34ab-49aa-a86d-a4360639b9da"_nvrhi_guid;
    blob.name = "Blob120";
    blob.size = 120;
    blob.alignment = 8;
    blob.flags = ComponentFlags::Relocatable;
    ComponentId blob120 = ComponentId::Invalid;
    CHECK(world->RegisterComponent(blob, blob120) == nvrhi::FS_OK);
    blob.uuid = "f0a719e3-3e00-4c85-98d7-67a5969ac009"_nvrhi_guid;
    blob.name = "Blob128";
    blob.size = 128;
    ComponentId blob128 = ComponentId::Invalid;
    CHECK(world->RegisterComponent(blob, blob128) == nvrhi::FS_OK);
    info = InfoOf(world, Create(world, { blob120 }));
    CHECK(info.rowSize == 128 && info.chunkSize == 16384 && info.rowsPerChunk == ExpectedRows(16384, { 120 }));
    info = InfoOf(world, Create(world, { blob128 }));
    CHECK(info.rowSize == 136 && info.chunkSize == 65536 && info.rowsPerChunk == ExpectedRows(65536, { 128 }));

    // Other chunk sizes from the world descriptor; invalid ones are refused.
    {
        WorldDesc desc;
        desc.chunkSize = 4096;
        desc.largeChunkSize = 8192;
        desc.largeRowThreshold = 64;
        nvrhi::AutoPtr<IWorld> small;
        CHECK(CreateWorld(desc, &small) == nvrhi::FS_OK);
        const ComponentId p = RegisterComponent<Position>(small);
        const ComponentId b = RegisterComponent<Big>(small);
        info = InfoOf(small, Create(small, { p }));
        CHECK(info.chunkSize == 4096 && info.rowsPerChunk == ExpectedRows(4096, { 12 }));
        info = InfoOf(small, Create(small, { b }));
        CHECK(info.chunkSize == 8192 && info.rowsPerChunk == ExpectedRows(8192, { 200 }));
        IWorld* refused = nullptr;
        desc.chunkSize = 3000;
        CHECK(CreateWorld(desc, &refused) == nvrhi::FE_INVALID_ARGS && !refused);
        desc.chunkSize = 16384;
        desc.largeChunkSize = 8192;
        CHECK(CreateWorld(desc, &refused) == nvrhi::FE_INVALID_ARGS && !refused);
    }

    // Every column and every inline buffer storage of every chunk is 64-byte aligned.
    const Entity first = Create(world, { id, position, wide, joint, toggle });
    info = InfoOf(world, first);
    CHECK(info.rowSize == 8 + sizeof(Id) + sizeof(Position) + sizeof(Wide) + sizeof(BufferHeader) + 4 * sizeof(Joint) +
                         sizeof(Toggle));
    CHECK(info.chunkSize == 65536);
    std::vector<Entity> rows(2 * info.rowsPerChunk + 5);
    const CreateDesc mixed({ id, position, wide, joint, toggle }, {}, uint32_t(rows.size()));
    CHECK(world->CreateEntities(mixed, rows.data()) == nvrhi::FS_OK);
    const ecs::Archetype& a = Impl(world).GetArchetypeStorage(ArchetypeOf(world, first));
    CHECK(a.GetChunkCount() == 3);
    for (uint32_t chunk = 0; chunk < a.GetChunkCount(); ++chunk)
    {
        CHECK(Aligned(a.chunks[chunk].memory, 64));
        for (uint32_t c = 0; c < a.columns.size(); ++c)
        {
            const ecs::ArchetypeColumn& column = a.columns[c];
            if (column.kind == ecs::StorageKind::Data || column.kind == ecs::StorageKind::Buffer)
                CHECK(Aligned(a.ColumnData(chunk, c), 64));
            if (column.kind == ecs::StorageKind::Buffer)
                CHECK(Aligned(a.InlineData(chunk, c, 0), 64));
            if (column.enableable)
                CHECK(Aligned(a.EnabledMask(chunk, c), 8));
        }
        CHECK(Aligned(a.Versions(chunk), 8));
    }
    for (const Entity entity : rows)
        CHECK(Aligned(world->GetComponent(entity, wide), 64));
}

// ------------------------------------------------------------------------------------- change versions

void test_versions()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);
    CHECK(world->GetVersion() == 1);

    const Entity e = Create(world, { id, position });
    const uint32_t idColumn = ColumnOf(world, e, id);
    const uint32_t posColumn = ColumnOf(world, e, position);
    const uint32_t added = Impl(world).GetArchetypeStorage(ArchetypeOf(world, e)).AddedVersionSlot();
    CHECK(VersionsOf(world, e)[idColumn] == 1 && VersionsOf(world, e)[posColumn] == 1);
    CHECK(VersionsOf(world, e)[added] == 1);

    CHECK(world->AdvanceVersion() == 2 && world->GetVersion() == 2);
    CHECK(world->GetComponent(e, position) != nullptr);
    CHECK(VersionsOf(world, e)[posColumn] == 1);   // reading never stamps
    CHECK(world->GetComponentForWrite(e, position, c_WorldVersion) != nullptr);
    CHECK(VersionsOf(world, e)[posColumn] == 2 && VersionsOf(world, e)[idColumn] == 1);
    CHECK(VersionsOf(world, e)[added] == 1);
    CHECK(world->GetComponentForWrite(e, position, 1) != nullptr);
    CHECK(VersionsOf(world, e)[posColumn] == 2);   // an older version never lowers it

    world->AdvanceVersion();
    world->AdvanceVersion();
    CHECK(world->AdvanceVersion() == 5);
    CHECK(world->GetComponentForWrite(e, id, 4) != nullptr);   // an explicit version
    CHECK(VersionsOf(world, e)[idColumn] == 4);

    // New rows stamp their chunk: every column and the added version.
    const Entity f = Create(world, { id, position });
    CHECK(VersionsOf(world, f)[idColumn] == 5 && VersionsOf(world, f)[posColumn] == 5);
    CHECK(VersionsOf(world, f)[added] == 5);

    // A removal that moves the last row into a hole stamps the hole's chunk.
    CHECK(world->AdvanceVersion() == 6);
    CHECK(world->DestroyEntities(&e, 1) == nvrhi::FS_OK);
    CHECK(Record(world, f).row == 0);
    CHECK(VersionsOf(world, f)[idColumn] == 6 && VersionsOf(world, f)[added] == 6);

    // The structural version follows structure only.
    const uint64_t structural = world->GetStructuralVersion();
    CHECK(world->GetComponentForWrite(f, id, c_WorldVersion) != nullptr);
    world->AdvanceVersion();
    CHECK(world->GetStructuralVersion() == structural);
    Create(world, { id });
    CHECK(world->GetStructuralVersion() > structural);

    // A version not handed out yet is stamped, and validation reports it.
    const uint64_t errors = impl.GetValidationErrorCount();
    CHECK(world->GetComponentForWrite(f, position, 100) != nullptr);
    CHECK(VersionsOf(world, f)[posColumn] == 100 && impl.GetValidationErrorCount() == errors + 1);

    WorldStats stats;
    CHECK(world->GetStats(stats).version == world->GetVersion());
    CHECK(stats.structuralVersion == world->GetStructuralVersion());
}

// ------------------------------------------------------------------------------------ enableable bits

void test_enableable()
{
    auto world = MakeWorld(true);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId toggle = RegisterComponent<Toggle>(world);
    const ComponentId hidden = RegisterComponent<Hidden>(world);
    const ComponentId velocity = RegisterComponent<Velocity>(world);

    const Id ids[3] = { { 0 }, { 1 }, { 2 } };
    const Toggle toggles[3] = { { 100 }, { 101 }, { 102 } };
    Entity e[3];
    CHECK(world->CreateEntities(CreateDesc({ id, toggle }, { ids, toggles }, 3), e) == nvrhi::FS_OK);
    for (const Entity entity : e)
    {
        CHECK(world->IsEnabled(entity, toggle));      // new rows start enabled
        CHECK(world->IsEnabled(entity, id));          // present, not Enableable
        CHECK(!world->IsEnabled(entity, velocity));   // absent
    }

    const uint64_t version = world->AdvanceVersion();
    CHECK(world->SetEnabled(e[2], toggle, false) == nvrhi::FS_OK);
    CHECK(!world->IsEnabled(e[2], toggle) && world->IsEnabled(e[0], toggle) && world->IsEnabled(e[1], toggle));
    CHECK(VersionsOf(world, e[2])[ColumnOf(world, e[2], toggle)] == version);   // toggling is a change
    CHECK(world->SetEnabled(e[0], id, false) == nvrhi::FE_INVALID_ARGS);        // not Enableable
    CHECK(world->SetEnabled(e[0], hidden, false) == nvrhi::FE_INVALID_ARGS);    // absent

    // The last row moves into a hole with its bit.
    CHECK(world->DestroyEntities(&e[0], 1) == nvrhi::FS_OK);
    CHECK(Record(world, e[2]).row == 0);
    CHECK(!world->IsEnabled(e[2], toggle) && world->IsEnabled(e[1], toggle));
    CHECK(Get<Toggle>(world, e[2]).value == 102);

    // Bits carry over moves between archetypes.
    CHECK(world->AddComponent(e[2], velocity, nullptr) == nvrhi::FS_OK);
    CHECK(!world->IsEnabled(e[2], toggle) && Get<Toggle>(world, e[2]).value == 102);
    CHECK(world->RemoveComponent(e[2], velocity) == nvrhi::FS_OK);
    CHECK(!world->IsEnabled(e[2], toggle));

    // An Enableable tag.
    CHECK(world->AddComponent(e[1], hidden, nullptr) == nvrhi::FS_OK);
    CHECK(world->IsEnabled(e[1], hidden) && world->GetComponent(e[1], hidden) != nullptr);
    CHECK(world->SetEnabled(e[1], hidden, false) == nvrhi::FS_OK && !world->IsEnabled(e[1], hidden));
    CHECK(world->IsEnabled(e[1], toggle));

    // A new row starts enabled where a disabled row was before.
    CHECK(world->DestroyEntities(&e[2], 1) == nvrhi::FS_OK);
    const Entity g = Create(world, { id, toggle });
    CHECK(Record(world, g).row == 0 && world->IsEnabled(g, toggle));
}

// --------------------------------------------------------------------------------- shared and chunk components

void test_shared_partitions()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId layer = RegisterComponent<Layer>(world);

    Layer a;
    std::memset(&a, 0xaa, sizeof(a));
    a.id = 1;
    a.mask = 0xff;
    Layer b;
    std::memset(&b, 0x55, sizeof(b));
    b.id = 1;
    b.mask = 0xff;
    Layer c;
    std::memset(&c, 0, sizeof(c));
    c.id = 2;
    c.mask = 0xff;
    const Id idA{ 1 }, idB{ 2 }, idC{ 3 };

    const uint32_t before = world->GetArchetypeCount();
    const Entity ea = Create(world, { id, layer }, { &idA, &a });
    const Entity eb = Create(world, { id, layer }, { &idB, &b });
    CHECK(ArchetypeOf(world, ea) == ArchetypeOf(world, eb));   // padding is not part of the key
    CHECK(world->GetArchetypeCount() == before + 1);
    CHECK(world->GetComponent(ea, layer) == world->GetComponent(eb, layer));                               // one value
    const Entity ec = Create(world, { id, layer }, { &idC, &c });
    CHECK(ArchetypeOf(world, ec) != ArchetypeOf(world, ea) && world->GetArchetypeCount() == before + 2);

    // A batch: runs of equal values go to one archetype each, in order.
    Layer layers[6];
    const uint8_t layerIds[6] = { 1, 1, 2, 2, 1, 3 };
    Id ids[6];
    for (uint32_t i = 0; i < 6; ++i)
    {
        std::memset(&layers[i], int(i * 17), sizeof(Layer));
        layers[i].id = layerIds[i];
        layers[i].mask = 0xff;
        ids[i].value = 10 + i;
    }
    Entity out[6];
    CHECK(world->CreateEntities(CreateDesc({ id, layer }, { ids, layers }, 6), out) == nvrhi::FS_OK);
    CHECK(ArchetypeOf(world, out[0]) == ArchetypeOf(world, ea) && ArchetypeOf(world, out[1]) == ArchetypeOf(world, ea));
    CHECK(ArchetypeOf(world, out[2]) == ArchetypeOf(world, ec) && ArchetypeOf(world, out[3]) == ArchetypeOf(world, ec));
    CHECK(ArchetypeOf(world, out[4]) == ArchetypeOf(world, ea) && world->GetArchetypeCount() == before + 3);
    CHECK(Get<Layer>(world, out[5]).id == 3);
    const ecs::Archetype& one = impl.GetArchetypeStorage(ArchetypeOf(world, ea));
    const Entity expected[5] = { ea, eb, out[0], out[1], out[4] };
    CHECK(one.entityCount == 5);
    for (uint32_t row = 0; row < 5; ++row)
        CHECK(one.Entities(0)[row] == expected[row]);

    // Another value moves the entity; the same value does nothing.
    CHECK(world->AddComponent(ea, layer, &c) == nvrhi::FS_OK);
    CHECK(ArchetypeOf(world, ea) == ArchetypeOf(world, ec));
    CHECK(Get<Layer>(world, ea).id == 2 && Get<Id>(world, ea).value == 1);
    const uint64_t structural = world->GetStructuralVersion();
    CHECK(world->AddComponent(eb, layer, &a) == nvrhi::FS_OK);
    CHECK(world->GetStructuralVersion() == structural);

    // Shared values are read-only per entity.
    const uint64_t errors = impl.GetValidationErrorCount();
    CHECK(world->GetComponentForWrite(eb, layer, c_WorldVersion) == nullptr);
    CHECK(impl.GetValidationErrorCount() == errors + 1);

    // Added without a value: the default value. Removed: back to the archetype without it.
    const Entity plain = Create(world, { id });
    CHECK(world->AddComponent(plain, layer, nullptr) == nvrhi::FS_OK);
    CHECK(Get<Layer>(world, plain).id == 0 && Get<Layer>(world, plain).mask == 0);
    CHECK(world->RemoveComponent(ec, layer) == nvrhi::FS_OK && !world->HasComponent(ec, layer));
    CHECK(ArchetypeOf(world, ec) == ArchetypeOf(world, Create(world, { id })) && Get<Id>(world, ec).value == 3);
}

void test_chunk_components()
{
    auto world = MakeWorld(true);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId bounds = RegisterComponent<ChunkBounds>(world);

    const Entity probe = Create(world, { id, bounds });
    const uint32_t rows = InfoOf(world, probe).rowsPerChunk;
    std::vector<Entity> e(rows);
    CHECK(world->CreateEntities(CreateDesc({ id, bounds }, {}, rows), e.data()) == nvrhi::FS_OK);   // two chunks
    CHECK(InfoOf(world, probe).chunkCount == 2);

    const uint64_t version = world->AdvanceVersion();
    ChunkBounds* first = static_cast<ChunkBounds*>(world->GetComponentForWrite(probe, bounds, c_WorldVersion));
    CHECK(first && first->lo == 0 && first->hi == 0);   // zero-filled per chunk
    first->lo = 1;
    first->hi = 2;
    CHECK(world->GetComponent(e[0], bounds) == first);   // one value per chunk
    const ChunkBounds* last = static_cast<const ChunkBounds*>(world->GetComponent(e[rows - 1], bounds));
    CHECK(last != first && last->lo == 0);
    CHECK(VersionsOf(world, probe)[ColumnOf(world, probe, bounds)] == version);
    CHECK(VersionsOf(world, e[rows - 1])[ColumnOf(world, probe, bounds)] < version);

    // Chunk values come only from the chunk: no initial values.
    const ChunkBounds value{ 5, 6 };
    const CreateDesc withValue({ id, bounds }, { nullptr, &value }, 1);
    CHECK(world->CreateEntities(withValue, e.data()) == nvrhi::FE_INVALID_ARGS);
    const Entity plain = Create(world, { id });
    CHECK(world->AddComponent(plain, bounds, &value) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->AddComponent(plain, bounds, nullptr) == nvrhi::FS_OK && world->HasComponent(plain, bounds));

    // A non-trivial chunk component: constructed per chunk, destroyed with the chunk.
    const HookCounts start = g_Hooks;
    {
        auto hooked = MakeWorld(true);
        const ComponentId hid = RegisterComponent<Id>(hooked);
        ComponentTypeDesc desc = ComponentTraits<Handle>::Desc();
        desc.uuid = "f0af2f4b-dcd6-49e8-a679-2d2ed2188666"_nvrhi_guid;
        desc.name = "HandleChunk";
        desc.flags = ComponentFlags::Chunk;
        ComponentId handleChunk = ComponentId::Invalid;
        CHECK(hooked->RegisterComponent(desc, handleChunk) == nvrhi::FS_OK);
        const Entity h0 = Create(hooked, { hid, handleChunk });
        const uint32_t hrows = InfoOf(hooked, h0).rowsPerChunk;
        std::vector<Entity> h(hrows);
        CHECK(hooked->CreateEntities(CreateDesc({ hid, handleChunk }, {}, hrows), h.data()) == nvrhi::FS_OK);
        CHECK_HOOKS(start, 2, 0, 0, 0);
        CHECK(hooked->DestroyEntities(&h[hrows - 1], 1) == nvrhi::FS_OK);   // the second chunk's only row
        CHECK_HOOKS(start, 2, 0, 0, 1);
        CHECK(InfoOf(hooked, h0).chunkCount == 1);
    }
    CHECK_HOOKS(start, 2, 0, 0, 2);
}

// ------------------------------------------------------------------------------------------- sparse sets

void test_sparse_sets()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId velocity = RegisterComponent<Velocity>(world);
    const ComponentId selected = RegisterComponent<Selected>(world);
    const ComponentId color = RegisterComponent<DebugColor>(world);

    // More entities than a sparse page holds.
    const uint32_t n = 5000;
    std::vector<Id> ids(n);
    for (uint32_t i = 0; i < n; ++i)
        ids[i].value = i;
    std::vector<Entity> e(n);
    CHECK(world->CreateEntities(CreateDesc({ id }, { ids.data() }, n), e.data()) == nvrhi::FS_OK);
    const uint32_t archetypes = world->GetArchetypeCount();
    const uint32_t home = ArchetypeOf(world, e[0]);

    std::vector<uint8_t> alive(n, 1), hasSelected(n, 0);
    std::vector<int64_t> colorOf(n, -1);
    for (uint32_t i = 0; i < n; i += 3)
    {
        CHECK(world->AddComponent(e[i], selected, nullptr) == nvrhi::FS_OK);
        hasSelected[i] = 1;
    }
    for (uint32_t i = 0; i < n; i += 5)
    {
        const DebugColor value{ 0x1000 + i };
        CHECK(world->AddComponent(e[i], color, &value) == nvrhi::FS_OK);
        colorOf[i] = 0x1000 + i;
    }
    CHECK(world->GetArchetypeCount() == archetypes && ArchetypeOf(world, e[3]) == home);   // no archetype moves
    WorldStats stats;
    CHECK(world->GetStats(stats).sparseSetCount == 2);

    auto verify = [&]()
    {
        for (uint32_t i = 0; i < n; ++i)
        {
            if (!alive[i])
                continue;
            CHECK(world->HasComponent(e[i], selected) == (hasSelected[i] != 0));
            CHECK(world->HasComponent(e[i], color) == (colorOf[i] >= 0));
            if (colorOf[i] >= 0)
                CHECK(Get<DebugColor>(world, e[i]).rgba == uint32_t(colorOf[i]));
            CHECK(Get<Id>(world, e[i]).value == i);
        }
        // The presence masks match the sets row by row, and rows past the end are clear.
        for (const ComponentId component : { selected, color })
        {
            const ecs::SparseSet* set = impl.GetSparseSet(component);
            CHECK(set != nullptr);
            uint32_t present = 0;
            for (uint32_t ai = 0; ai < impl.GetArchetypeStorageCount(); ++ai)
            {
                const ecs::Archetype& a = impl.GetArchetypeStorage(ai);
                for (uint32_t chunk = 0; chunk < a.GetChunkCount(); ++chunk)
                {
                    const uint64_t* mask = impl.GetSparsePresence(*set, ai, chunk);
                    for (uint32_t row = 0; row < a.rowsPerChunk; ++row)
                    {
                        const bool bit = ecs::GetMaskBit(mask, row);
                        if (row >= a.chunks[chunk].count)
                        {
                            CHECK(!bit);
                            continue;
                        }
                        CHECK(bit == world->HasComponent(a.Entities(chunk)[row], component));
                        present += bit ? 1 : 0;
                    }
                }
            }
            CHECK(present == set->GetCount());
        }
    };
    verify();

    for (uint32_t i = 0; i < n; i += 6)
    {
        CHECK(world->RemoveComponent(e[i], selected) == nvrhi::FS_OK);
        hasSelected[i] = 0;
    }
    CHECK(world->RemoveComponent(e[1], selected) == nvrhi::FS_OK);   // absent: nothing to do
    verify();

    // Destroyed entities leave the sets; swap-removed rows take their bits along.
    std::vector<Entity> doomed;
    for (uint32_t i = 0; i < n; i += 7)
    {
        doomed.push_back(e[i]);
        alive[i] = 0;
    }
    CHECK(world->DestroyEntities(doomed.data(), uint32_t(doomed.size())) == nvrhi::FS_OK);
    verify();

    // Archetype moves take the bits along too.
    for (uint32_t i = 1; i < n; i += 4)
        if (alive[i])
            CHECK(world->AddComponent(e[i], velocity, nullptr) == nvrhi::FS_OK);
    verify();

    // Overwritten through AddComponent; created with values.
    const DebugColor overwrite{ 77 };
    CHECK(world->AddComponent(e[5], color, &overwrite) == nvrhi::FS_OK);
    colorOf[5] = 77;
    verify();
    const Id createdIds[2] = { { 9000 }, { 9001 } };
    const DebugColor createdColors[2] = { { 7 }, { 8 } };
    Entity created[2];
    CHECK(world->CreateEntities(CreateDesc({ color, id }, { createdColors, createdIds }, 2), created) == nvrhi::FS_OK);
    CHECK(Get<DebugColor>(world, created[0]).rgba == 7 && Get<DebugColor>(world, created[1]).rgba == 8);
    CHECK(ArchetypeOf(world, created[0]) == home);
    const uint32_t before = impl.GetSparseSet(color)->GetCount();
    CHECK(world->DestroyEntities(created, 2) == nvrhi::FS_OK);
    CHECK(impl.GetSparseSet(color)->GetCount() == before - 2);
    verify();
}

// --------------------------------------------------------------------------------------- dynamic buffers

void test_dynamic_buffers()
{
    auto world = MakeWorld(true);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId joint = RegisterComponent<Joint>(world);
    const ComponentId velocity = RegisterComponent<Velocity>(world);
    WorldStats stats;

    const Entity e = Create(world, { id, joint });
    CHECK(HeaderOf(world, e, joint).data == InlineOf(world, e, joint));
    CHECK(HeaderOf(world, e, joint).count == 0 && HeaderOf(world, e, joint).capacity == 4);

    const uint64_t version = world->AdvanceVersion();
    CHECK(world->ResizeBuffer(e, joint, 3, c_WorldVersion) == nvrhi::FS_OK);
    CHECK(VersionsOf(world, e)[ColumnOf(world, e, joint)] == version);
    BufferHeader* header = static_cast<BufferHeader*>(world->GetComponentForWrite(e, joint, c_WorldVersion));
    CHECK(header->count == 3 && header->data == InlineOf(world, e, joint));
    for (uint32_t k = 0; k < 3; ++k)
    {
        CHECK(GetBufferData<Joint>(*header)[k].weight == 0);   // constructed (zero-filled)
        GetBufferData<Joint>(*header)[k] = Joint{ MakeEntity(k + 1, 1), float(k), 0 };
    }

    // Spill to the heap.
    CHECK(world->GetStats(stats).bufferHeapBytes == 0);
    CHECK(world->ResizeBuffer(e, joint, 10, c_WorldVersion) == nvrhi::FS_OK);
    const BufferHeader* spilled = &HeaderOf(world, e, joint);
    CHECK(spilled->data != InlineOf(world, e, joint) && spilled->count == 10 && spilled->capacity >= 10);
    for (uint32_t k = 0; k < 10; ++k)
        CHECK(GetBufferData<Joint>(*spilled)[k].weight == (k < 3 ? float(k) : 0.f));
    CHECK(world->GetStats(stats).bufferHeapBytes == uint64_t(spilled->capacity) * sizeof(Joint));

    // A heap buffer moves with its row.
    void* heap = spilled->data;
    CHECK(world->AddComponent(e, velocity, nullptr) == nvrhi::FS_OK);
    CHECK(HeaderOf(world, e, joint).data == heap && HeaderOf(world, e, joint).count == 10);

    // Back into the row once it fits.
    CHECK(world->ResizeBuffer(e, joint, 2, c_WorldVersion) == nvrhi::FS_OK);
    CHECK(HeaderOf(world, e, joint).data == InlineOf(world, e, joint) && HeaderOf(world, e, joint).capacity == 4);
    CHECK(GetBufferData<Joint>(HeaderOf(world, e, joint))[1].weight == 1.f);
    CHECK(GetBufferData<Joint>(HeaderOf(world, e, joint))[1].target == MakeEntity(2, 1));
    CHECK(world->GetStats(stats).bufferHeapBytes == 0);

    // Inline elements follow their row through swap-removes and archetype moves.
    const Entity f = Create(world, { id, joint, velocity });
    CHECK(world->ResizeBuffer(f, joint, 4, c_WorldVersion) == nvrhi::FS_OK);
    BufferHeader* fh = static_cast<BufferHeader*>(world->GetComponentForWrite(f, joint, c_WorldVersion));
    for (uint32_t k = 0; k < 4; ++k)
        GetBufferData<Joint>(*fh)[k].weight = float(10 + k);
    CHECK(world->DestroyEntities(&e, 1) == nvrhi::FS_OK);
    CHECK(Record(world, f).row == 0 && HeaderOf(world, f, joint).data == InlineOf(world, f, joint));
    CHECK(world->RemoveComponent(f, velocity) == nvrhi::FS_OK);
    CHECK(HeaderOf(world, f, joint).data == InlineOf(world, f, joint) && HeaderOf(world, f, joint).count == 4);
    for (uint32_t k = 0; k < 4; ++k)
        CHECK(GetBufferData<Joint>(HeaderOf(world, f, joint))[k].weight == float(10 + k));

    CHECK(world->ResizeBuffer(f, id, 3, c_WorldVersion) == nvrhi::FE_INVALID_ARGS);        // not a Buffer
    CHECK(world->ResizeBuffer(Create(world, { id }), joint, 3, c_WorldVersion) == nvrhi::FE_INVALID_ARGS);   // absent
    Entity rejected = Entity::Null;
    CHECK(world->CreateEntities(CreateDesc({ id, joint }, { nullptr, &stats }, 1), &rejected) ==
          nvrhi::FE_INVALID_ARGS);   // Buffer components start empty

    // Destroying the entity frees its spilled storage.
    CHECK(world->ResizeBuffer(f, joint, 20, c_WorldVersion) == nvrhi::FS_OK);
    CHECK(world->GetStats(stats).bufferHeapBytes > 0);
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    CHECK(world->DestroyEntities(&f, 1) == nvrhi::FS_OK);
    CHECK(world->GetStats(stats).bufferHeapBytes == 0 && ecs::GetHeapLiveBlockCount() == blocks - 1);

    // Non-trivial elements: the hooks run per element.
    ComponentTypeDesc desc = ComponentTraits<Handle>::Desc();
    desc.uuid = "eaa66465-8b97-428c-95ae-a53a5de3a0af"_nvrhi_guid;
    desc.name = "HandleBuffer";
    desc.flags = ComponentFlags::Buffer;
    desc.bufferInlineCapacity = 2;
    ComponentId handles = ComponentId::Invalid;
    CHECK(world->RegisterComponent(desc, handles) == nvrhi::FS_OK);
    const Entity h = Create(world, { id, handles });
    const HookCounts start = g_Hooks;
    CHECK(world->ResizeBuffer(h, handles, 2, c_WorldVersion) == nvrhi::FS_OK);
    CHECK_HOOKS(start, 2, 0, 0, 0);
    CHECK(world->ResizeBuffer(h, handles, 5, c_WorldVersion) == nvrhi::FS_OK);   // spill: relocate 2, construct 3
    CHECK_HOOKS(start, 5, 0, 2, 2);
    CHECK(world->ResizeBuffer(h, handles, 1, c_WorldVersion) == nvrhi::FS_OK);   // destroy 4, relocate 1 back
    CHECK_HOOKS(start, 5, 0, 3, 7);
    CHECK(world->DestroyEntities(&h, 1) == nvrhi::FS_OK);
    CHECK_HOOKS(start, 5, 0, 3, 8);
}

// ---------------------------------------------------------------------------------------- lifecycle hooks

void test_hooks()
{
    Payload payload;
    const HookCounts outset = g_Hooks;
    {
        Handle source[3] = { Handle(&payload), Handle(&payload), Handle(&payload) };
        {
            auto world = MakeWorld(true);
            const ComponentId handle = RegisterComponent<Handle>(world);
            const ComponentId id = RegisterComponent<Id>(world);
            CHECK(payload.refs == 3);

            HookCounts base = g_Hooks;
            Entity d[4];
            CHECK(world->CreateEntities(CreateDesc({ handle }, {}, 4), d) == nvrhi::FS_OK);   // default-constructed
            CHECK_HOOKS(base, 4, 0, 0, 0);

            base = g_Hooks;
            Entity c[3];
            CHECK(world->CreateEntities(CreateDesc({ handle }, { source }, 3), c) == nvrhi::FS_OK);   // copied
            CHECK_HOOKS(base, 0, 3, 0, 0);
            CHECK(payload.refs == 6);

            alignas(Handle) unsigned char raw[2 * sizeof(Handle)];
            new (raw) Handle(&payload);
            new (raw + sizeof(Handle)) Handle(&payload);
            base = g_Hooks;
            Entity r[2];
            const CreateDesc relocated({ handle }, { raw }, 2, ValueTransfer::Relocate);
            CHECK(world->CreateEntities(relocated, r) == nvrhi::FS_OK);
            CHECK_HOOKS(base, 0, 0, 2, 2);   // moved in, sources destroyed
            CHECK(payload.refs == 8);
            // rows: d0 d1 d2 d3 c0 c1 c2 r0 r1

            base = g_Hooks;
            CHECK(world->DestroyEntities(&d[0], 1) == nvrhi::FS_OK);   // destroy row 0, relocate r1 into it
            CHECK_HOOKS(base, 0, 0, 1, 2);
            base = g_Hooks;
            CHECK(world->DestroyEntities(&r[0], 1) == nvrhi::FS_OK);   // the last row: no relocation
            CHECK_HOOKS(base, 0, 0, 0, 1);
            CHECK(payload.refs == 7);
            // rows: r1 d1 d2 d3 c0 c1 c2

            base = g_Hooks;
            CHECK(world->AddComponent(c[0], id, nullptr) == nvrhi::FS_OK);   // relocated out, c2 into the hole
            CHECK_HOOKS(base, 0, 0, 2, 2);
            CHECK(Get<Handle>(world, c[0]).payload == &payload && payload.refs == 7);
            // rows: r1 d1 d2 d3 c2 c1

            base = g_Hooks;
            Entity instances[5];
            CHECK(world->Instantiate(c[1], 5, instances) == nvrhi::FS_OK);
            CHECK_HOOKS(base, 0, 5, 0, 0);
            CHECK(payload.refs == 12);

            base = g_Hooks;
            CHECK(world->RemoveComponent(c[1], handle) == nvrhi::FS_OK);   // destroyed, the last instance relocated
            CHECK_HOOKS(base, 0, 0, 1, 2);
            CHECK(payload.refs == 11);

            // Every live handle is a constructed, copied or moved one that was not destroyed.
            CHECK(g_Hooks.Live() - outset.Live() == 3 + 11);

            // Move-only components: copies are refused, moves work.
            const ComponentId moveOnly = RegisterComponent<MoveOnly>(world);
            CHECK(!world->GetComponentTypeDesc(moveOnly)->copy && world->GetComponentTypeDesc(moveOnly)->relocate);
            const MoveOnly copySource[1] = {};
            Entity m = Entity::Null;
            CHECK(world->CreateEntities(CreateDesc({ moveOnly }, { copySource }, 1), &m) == nvrhi::FE_UNSUPPORTED);
            alignas(MoveOnly) unsigned char moveSource[sizeof(MoveOnly)];
            new (moveSource) MoveOnly(42);
            CHECK(world->CreateEntities(CreateDesc({ moveOnly }, { moveSource }, 1, ValueTransfer::Relocate), &m) ==
                  nvrhi::FS_OK);
            CHECK(Get<MoveOnly>(world, m).value == 42);
            CHECK(world->Instantiate(m, 2, instances) == nvrhi::FE_UNSUPPORTED);
            CHECK(world->AddComponent(m, id, nullptr) == nvrhi::FS_OK && Get<MoveOnly>(world, m).value == 42);
            const Entity other = Create(world, { id });
            CHECK(world->AddComponent(other, moveOnly, copySource) == nvrhi::FE_UNSUPPORTED);
            CHECK(world->AddComponent(other, moveOnly, nullptr) == nvrhi::FS_OK);
            CHECK(Get<MoveOnly>(world, other).value == 0);
        }
        // The world destroyed the 11 handles it held.
        CHECK(payload.refs == 3);
        CHECK(g_Hooks.Live() - outset.Live() == 3);
    }
    CHECK(payload.refs == 0 && g_Hooks.Live() == outset.Live());
}

// ------------------------------------------------------------------------------------------------ copies

void test_instantiate()
{
    auto world = MakeWorld(true);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);
    const ComponentId toggle = RegisterComponent<Toggle>(world);
    const ComponentId joint = RegisterComponent<Joint>(world);
    const ComponentId frozen = RegisterComponent<Frozen>(world);
    const ComponentId layer = RegisterComponent<Layer>(world);
    const ComponentId color = RegisterComponent<DebugColor>(world);
    const ComponentId selected = RegisterComponent<Selected>(world);

    const Id i{ 42 };
    const Position p{ 1, 2, 3 };
    const Toggle t{ 9 };
    Layer l;
    std::memset(&l, 0, sizeof(l));
    l.id = 3;
    l.mask = 7;
    const Entity prefab =
        Create(world, { id, position, toggle, joint, frozen, layer }, { &i, &p, &t, nullptr, nullptr, &l });
    CHECK(world->SetEnabled(prefab, toggle, false) == nvrhi::FS_OK);
    CHECK(world->ResizeBuffer(prefab, joint, 6, c_WorldVersion) == nvrhi::FS_OK);   // spilled to the heap
    BufferHeader* header = static_cast<BufferHeader*>(world->GetComponentForWrite(prefab, joint, c_WorldVersion));
    for (uint32_t k = 0; k < 6; ++k)
        GetBufferData<Joint>(*header)[k] = Joint{ prefab, float(k), 0 };
    const DebugColor c{ 0xabcdef };
    CHECK(world->AddComponent(prefab, color, &c) == nvrhi::FS_OK);
    CHECK(world->AddComponent(prefab, selected, nullptr) == nvrhi::FS_OK);

    const uint32_t archetypes = world->GetArchetypeCount();
    std::vector<Entity> out(1000);
    CHECK(world->Instantiate(prefab, 1000, out.data()) == nvrhi::FS_OK);
    CHECK(world->GetArchetypeCount() == archetypes);
    std::vector<Entity> sorted = out;
    std::sort(sorted.begin(), sorted.end());
    CHECK(std::unique(sorted.begin(), sorted.end()) == sorted.end());
    for (const Entity entity : out)
    {
        CHECK(world->IsAlive(entity) && ArchetypeOf(world, entity) == ArchetypeOf(world, prefab));
        CHECK(Get<Id>(world, entity).value == 42 && Get<Position>(world, entity).y == 2);
        CHECK(!world->IsEnabled(entity, toggle) && Get<Toggle>(world, entity).value == 9);
        CHECK(world->HasComponent(entity, frozen));
        CHECK(world->GetComponent(entity, layer) == world->GetComponent(prefab, layer));
        const BufferHeader& copy = HeaderOf(world, entity, joint);
        CHECK(copy.count == 6 && copy.data != header->data);   // a deep copy
        for (uint32_t k = 0; k < 6; ++k)
            CHECK(GetBufferData<Joint>(copy)[k].weight == float(k) && GetBufferData<Joint>(copy)[k].target == prefab);
        CHECK(Get<DebugColor>(world, entity).rgba == 0xabcdef && world->HasComponent(entity, selected));
    }
    WorldStats stats;
    CHECK(world->GetStats(stats).entityCount == 1001);
    CHECK(stats.bufferHeapBytes == uint64_t(header->capacity) * sizeof(Joint) + 1000ull * 6 * sizeof(Joint));

    const Entity stale = MakeEntity(GetEntityIndex(prefab), GetEntityGeneration(prefab) + 1);
    CHECK(world->Instantiate(stale, 1, out.data()) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->Instantiate(prefab, 0, nullptr) == nvrhi::FS_OK);
}

void test_copy_entities_from()
{
    auto source = MakeWorld(true);
    auto target = MakeWorld(true);
    // Registered in other orders, so that the ids differ.
    const ComponentId sId = RegisterComponent<Id>(source);
    const ComponentId sLink = RegisterComponent<Link>(source);
    const ComponentId sPosition = RegisterComponent<Position>(source);
    const ComponentId sJoint = RegisterComponent<Joint>(source);
    const ComponentId sColor = RegisterComponent<DebugColor>(source);
    const ComponentId sLayer = RegisterComponent<Layer>(source);
    RegisterComponent<Position>(target);
    RegisterComponent<Layer>(target);
    RegisterComponent<Joint>(target);
    RegisterComponent<Link>(target);
    RegisterComponent<DebugColor>(target);
    const ComponentId tId = RegisterComponent<Id>(target);
    CHECK(sId != tId);

    const Id ids[5] = { { 0 }, { 1 }, { 2 }, { 3 }, { 99 } };
    const Entity outside = Create(source, { sId }, { &ids[4] });
    Entity a[4];
    a[0] = Create(source, { sId, sLink }, { &ids[0], nullptr });
    a[1] = Create(source, { sId, sLink }, { &ids[1], nullptr });
    a[2] = Create(source, { sId, sJoint }, { &ids[2], nullptr });
    Layer layer;
    std::memset(&layer, 0, sizeof(layer));
    layer.id = 5;
    layer.mask = 1;
    const Position position{ 7, 8, 9 };
    a[3] = Create(source, { sId, sPosition, sLayer }, { &ids[3], &position, &layer });
    const DebugColor color{ 0x123456 };
    CHECK(source->AddComponent(a[3], sColor, &color) == nvrhi::FS_OK);
    *static_cast<Link*>(source->GetComponentForWrite(a[0], sLink, c_WorldVersion)) = Link{ a[1], outside };
    *static_cast<Link*>(source->GetComponentForWrite(a[1], sLink, c_WorldVersion)) = Link{ a[0], Entity::Null };
    CHECK(source->ResizeBuffer(a[2], sJoint, 6, c_WorldVersion) == nvrhi::FS_OK);
    BufferHeader* joints = static_cast<BufferHeader*>(source->GetComponentForWrite(a[2], sJoint, c_WorldVersion));
    const Entity targets[6] = { a[0], outside, a[3], a[2], Entity::Null, a[1] };
    for (uint32_t k = 0; k < 6; ++k)
        GetBufferData<Joint>(*joints)[k] = Joint{ targets[k], float(k), 0 };

    Entity b[4];
    CHECK(target->CopyEntitiesFrom(source, a, 4, b) == nvrhi::FS_OK);
    for (uint32_t i = 0; i < 4; ++i)
        CHECK(target->IsAlive(b[i]) && Get<Id>(target, b[i]).value == i);
    // Entity fields that name a copied entity name its copy; others stay.
    CHECK(Get<Link>(target, b[0]).other == b[1] && Get<Link>(target, b[0]).outside == outside);
    CHECK(Get<Link>(target, b[1]).other == b[0] && Get<Link>(target, b[1]).outside == Entity::Null);
    const Entity remapped[6] = { b[0], outside, b[3], b[2], Entity::Null, b[1] };
    const BufferHeader& copied = HeaderOf(target, b[2], FindComponent<Joint>(target));
    CHECK(copied.count == 6);
    for (uint32_t k = 0; k < 6; ++k)
    {
        CHECK(GetBufferData<Joint>(copied)[k].target == remapped[k]);
        CHECK(GetBufferData<Joint>(copied)[k].weight == float(k));
    }
    CHECK(Get<Position>(target, b[3]).z == 9 && Get<Layer>(target, b[3]).id == 5);
    CHECK(Get<DebugColor>(target, b[3]).rgba == 0x123456);
    WorldStats stats;
    CHECK(source->GetStats(stats).entityCount == 5 && target->GetStats(stats).entityCount == 4);

    // A component type the target lacks: FE_NOT_FOUND, nothing created.
    auto lacking = MakeWorld(true);
    RegisterComponent<Id>(lacking);
    RegisterComponent<Link>(lacking);
    CHECK(lacking->CopyEntitiesFrom(source, a, 4, b) == nvrhi::FE_NOT_FOUND);
    CHECK(lacking->GetStats(stats).entityCount == 0);

    // The same GUID with another layout.
    auto clash = MakeWorld(true);
    RegisterComponent<Id>(clash);
    RegisterComponent<Link>(clash);
    ComponentTypeDesc other = ComponentTraits<Id>::Desc();
    CHECK(clash->FindComponent(other.uuid) != ComponentId::Invalid);
    ComponentTypeDesc wrongColor = ComponentTraits<DebugColor>::Desc();
    wrongColor.size = 8;
    ComponentId wrong = ComponentId::Invalid;
    CHECK(clash->RegisterComponent(wrongColor, wrong) == nvrhi::FS_OK);
    RegisterComponent<Position>(clash);
    RegisterComponent<Joint>(clash);
    RegisterComponent<Layer>(clash);
    CHECK(clash->CopyEntitiesFrom(source, a, 4, b) == nvrhi::FE_INVALID_ARGS);
    CHECK(clash->GetStats(stats).entityCount == 0);

    // Within one world; a source listed twice maps to its first copy.
    Entity c[3];
    const Entity list[3] = { a[1], a[0], a[1] };
    CHECK(source->CopyEntitiesFrom(source, list, 3, c) == nvrhi::FS_OK);
    CHECK(Get<Link>(source, c[1]).other == c[0] && Get<Link>(source, c[0]).other == c[1]);
    CHECK(Get<Link>(source, c[2]).other == c[1]);

    // A stale source entity.
    const Entity dead = Create(source, { sId });
    CHECK(source->DestroyEntities(&dead, 1) == nvrhi::FS_OK);
    CHECK(target->CopyEntitiesFrom(source, &dead, 1, b) == nvrhi::FE_INVALID_ARGS);
}

// ---------------------------------------------------------------------------------------------- resources

void test_resources()
{
    auto world = MakeWorld(true);
    const ComponentId position = RegisterComponent<Position>(world);

    const FrameTime initial{ 1.5, 0.25, 7 };
    ComponentId frame = ComponentId::Invalid;
    CHECK(world->RegisterResource(ComponentTraits<FrameTime>::Desc(), &initial, frame) == nvrhi::FS_OK);
    CHECK(frame == ComponentId(1));   // resources share the component id space
    CHECK(FindComponent<FrameTime>(world) == frame && RegisterComponent<FrameTime>(world) == frame);
    FrameTime* time = GetResource<FrameTime>(world);
    CHECK(time && time->time == 1.5 && time->deltaTime == 0.25 && time->frameIndex == 7);

    // Registered again: the same id, the value unchanged.
    const FrameTime later{ 9, 9, 9 };
    ComponentId again = ComponentId::Invalid;
    CHECK(world->RegisterResource(ComponentTraits<FrameTime>::Desc(), &later, again) == nvrhi::FS_OK);
    CHECK(again == frame && time->time == 1.5);
    CHECK(world->GetResource(position) == nullptr && world->GetResource(ComponentId(55)) == nullptr);
    CHECK(world->RegisterResource(ComponentTraits<Velocity>::Desc(), nullptr, again) == nvrhi::FE_INVALID_ARGS);

    // Never on an entity.
    const Entity e = Create(world, { position });
    CHECK(!world->HasComponent(e, frame) && world->GetComponent(e, frame) == nullptr);
    CHECK(world->AddComponent(e, frame, nullptr) == nvrhi::FE_INVALID_ARGS);
    Entity out = Entity::Null;
    CHECK(world->CreateEntities(CreateDesc({ position, frame }, {}, 1), &out) == nvrhi::FE_INVALID_ARGS);

    // A non-trivial resource is copied in and destroyed with the world.
    Payload payload;
    {
        const Handle handle(&payload);
        const HookCounts base = g_Hooks;
        {
            auto other = MakeWorld();
            ComponentTypeDesc desc = ComponentTraits<Handle>::Desc();
            desc.uuid = "32389834-720c-4699-9c8e-a44144eb5d1a"_nvrhi_guid;
            desc.name = "HandleResource";
            desc.flags = ComponentFlags::Resource;
            ComponentId resource = ComponentId::Invalid;
            CHECK(other->RegisterResource(desc, &handle, resource) == nvrhi::FS_OK);
            CHECK(static_cast<Handle*>(other->GetResource(resource))->payload == &payload && payload.refs == 2);
            CHECK_HOOKS(base, 0, 1, 0, 0);
        }
        CHECK_HOOKS(base, 0, 1, 0, 1);
        CHECK(payload.refs == 1);
    }
    CHECK(payload.refs == 0);
}

// ---------------------------------------------------------------------------- validation, phases, seams

void test_validation_and_phases()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);
    Entity e = Create(world, { id });
    CHECK(!world->IsInPhase());

    // Inside a phase, immediate structural changes are refused (rule 3.3.3) unless an exclusive system runs.
    impl.BeginPhase();
    CHECK(world->IsInPhase());
    const uint64_t errors = impl.GetValidationErrorCount();
    Entity out = Entity::Null;
    CHECK(world->CreateEntities(CreateDesc({ id }, {}, 1), &out) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->DestroyEntities(&e, 1) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->AddComponent(e, position, nullptr) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->RemoveComponent(e, id) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->Instantiate(e, 1, &out) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->CopyEntitiesFrom(world, &e, 1, &out) == nvrhi::FE_INVALID_ARGS);
    Entity reserved = Entity::Null;
    CHECK(world->ReserveEntities(1, &reserved) == nvrhi::FE_INVALID_ARGS && reserved == Entity::Null);
    CHECK(impl.GetValidationErrorCount() == errors + 7);
    CHECK(world->GetComponentForWrite(e, id, c_WorldVersion) != nullptr);   // not structural
    impl.BeginExclusive();
    CHECK(world->CreateEntities(CreateDesc({ id }, {}, 1), &out) == nvrhi::FS_OK && world->IsAlive(out));
    CHECK(world->ReserveEntities(1, &reserved) == nvrhi::FS_OK && !world->IsAlive(reserved));
    impl.EndExclusive();
    impl.EndPhase();
    CHECK(!world->IsInPhase());
    CHECK(world->AddComponent(e, position, nullptr) == nvrhi::FS_OK);

    // Without validation the same calls fail the same way, and nothing is reported.
    auto quiet = MakeWorld(false);
    const ComponentId qid = RegisterComponent<Id>(quiet);
    const Entity q = Create(quiet, { qid });
    CHECK(quiet->DestroyEntities(&q, 1) == nvrhi::FS_OK);
    CHECK(quiet->GetComponent(q, qid) == nullptr && quiet->DestroyEntities(&q, 1) == nvrhi::FE_INVALID_ARGS);
    CHECK(Impl(quiet).GetValidationErrorCount() == 0);
}

// The batch operations that command-buffer playback (E1.3) builds on; not reachable through IWorld.
void test_batch_internals()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);
    const ComponentId layer = RegisterComponent<Layer>(world);
    const ComponentId color = RegisterComponent<DebugColor>(world);
    const ComponentId joint = RegisterComponent<Joint>(world);
    const uint64_t version = world->GetVersion();

    const Id ids[4] = { { 0 }, { 1 }, { 2 }, { 3 } };
    Entity e[4];
    CHECK(world->CreateEntities(CreateDesc({ id }, { ids }, 4), e) == nvrhi::FS_OK);

    // A batch add: one archetype lookup for the run of entities that start in the same archetype.
    const Position added[4] = { { 1, 0, 0 }, { 2, 0, 0 }, { 3, 0, 0 }, { 4, 0, 0 } };
    const uint64_t lookups = impl.GetCounters().archetypeLookups;
    CHECK(impl.AddComponentsInternal(e, 4, position, added, ValueTransfer::Copy, version) == nvrhi::FS_OK);
    CHECK(impl.GetCounters().archetypeLookups == lookups + 1);
    for (uint32_t i = 0; i < 4; ++i)
        CHECK(Get<Position>(world, e[i]).x == float(i + 1) && Get<Id>(world, e[i]).value == i);

    // Sets: Data values in place, a Sparse value, a Shared value (a move); absent components fail.
    const Position set[2] = { { 10, 0, 0 }, { 20, 0, 0 } };
    const uint32_t archetype = ArchetypeOf(world, e[0]);
    CHECK(impl.SetComponentsInternal(e, 2, position, set, ValueTransfer::Copy, version) == nvrhi::FS_OK);
    CHECK(Get<Position>(world, e[1]).x == 20 && ArchetypeOf(world, e[0]) == archetype);
    const DebugColor first{ 1 }, second{ 2 };
    CHECK(impl.SetComponentsInternal(e, 1, color, &first, ValueTransfer::Copy, version) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->AddComponent(e[0], color, &first) == nvrhi::FS_OK);
    CHECK(impl.SetComponentsInternal(e, 1, color, &second, ValueTransfer::Copy, version) == nvrhi::FS_OK);
    CHECK(Get<DebugColor>(world, e[0]).rgba == 2);
    Layer layers[2];
    std::memset(layers, 0, sizeof(layers));
    layers[0].id = 1;
    layers[1].id = 1;
    CHECK(impl.AddComponentsInternal(e, 2, layer, layers, ValueTransfer::Copy, version) == nvrhi::FS_OK);
    CHECK(ArchetypeOf(world, e[0]) == ArchetypeOf(world, e[1]));
    layers[0].id = 2;
    CHECK(impl.SetComponentsInternal(e, 1, layer, layers, ValueTransfer::Copy, version) == nvrhi::FS_OK);
    CHECK(Get<Layer>(world, e[0]).id == 2 && ArchetypeOf(world, e[0]) != ArchetypeOf(world, e[1]));
    CHECK(Get<Position>(world, e[0]).x == 10 && Get<DebugColor>(world, e[0]).rgba == 2);

    // Appends: inline, then spilled.
    CHECK(world->AddComponent(e[2], joint, nullptr) == nvrhi::FS_OK);
    Joint joints[6];
    for (uint32_t k = 0; k < 6; ++k)
        joints[k] = Joint{ e[k % 4], float(k), 0 };
    CHECK(impl.AppendToBufferInternal(e[2], joint, joints, 3, ValueTransfer::Copy, version) == nvrhi::FS_OK);
    CHECK(HeaderOf(world, e[2], joint).data == InlineOf(world, e[2], joint));
    CHECK(impl.AppendToBufferInternal(e[2], joint, joints + 3, 3, ValueTransfer::Copy, version) == nvrhi::FS_OK);
    const BufferHeader& header = HeaderOf(world, e[2], joint);
    CHECK(header.count == 6 && header.data != InlineOf(world, e[2], joint));
    for (uint32_t k = 0; k < 6; ++k)
    {
        CHECK(GetBufferData<Joint>(header)[k].weight == float(k));
        CHECK(GetBufferData<Joint>(header)[k].target == e[k % 4]);
    }
    // A count that would overflow 32 bits is refused and changes nothing (6 + 0xfffffffa wraps to 0).
    CHECK(impl.AppendToBufferInternal(e[2], joint, joints, 0xfffffffau, ValueTransfer::Copy, version) ==
          nvrhi::FE_INVALID_ARGS);
    CHECK(impl.AppendToBufferInternal(e[2], joint, joints, 0xffffffffu, ValueTransfer::Copy, version) ==
          nvrhi::FE_INVALID_ARGS);
    CHECK(HeaderOf(world, e[2], joint).count == 6 && GetBufferData<Joint>(HeaderOf(world, e[2], joint))[5].weight == 5);

    // A batch removal from entities in different archetypes.
    CHECK(impl.RemoveComponentsInternal(e, 4, position, version) == nvrhi::FS_OK);
    for (const Entity entity : e)
        CHECK(!world->HasComponent(entity, position) && world->HasComponent(entity, id));

    // Reserved ids for placeholders.
    Entity reserved[2];
    CHECK(impl.AllocateEntityIds(2, reserved) == nvrhi::FS_OK);
    CHECK(!world->IsAlive(reserved[0]));
    CHECK(impl.GetEntityRecord(GetEntityIndex(reserved[1])).archetype == ecs::c_ReservedSlot);
    CreateDesc desc({ id }, {}, 2);
    desc->reservedEntities = reserved;
    CHECK(world->CreateEntities(desc, nullptr) == nvrhi::FS_OK && world->IsAlive(reserved[1]));
}

// Review item 1: sizes are checked in 64 bits, and one row of a component alone must fit the world's largest chunk.
void test_registration_limits()
{
    auto world = MakeWorld(true);   // 16 and 64 KiB chunks
    ecs::WorldImpl& impl = Impl(world);
    const ComponentTypeDesc joint = ComponentTraits<Joint>::Desc();   // 16-byte elements
    ComponentId out = ComponentId::Invalid;

    // 16 * 2^28 is 2^32, 0 in 32 bits; 16 * (2^28 + 1) wraps to 16.
    ComponentTypeDesc d = joint;
    d.uuid = "331ddd9c-1538-41ce-b75f-4f50294e09ba"_nvrhi_guid;
    d.bufferInlineCapacity = 0x10000000u;
    CHECK(world->RegisterComponent(d, out) == nvrhi::FE_INVALID_ARGS);
    d.uuid = "a1cf3536-4c84-4549-b551-8bc4e200f6f7"_nvrhi_guid;
    d.bufferInlineCapacity = 0x10000001u;
    CHECK(world->RegisterComponent(d, out) == nvrhi::FE_INVALID_ARGS);

    // The largest inline storage of which one row fits 64 KiB: 128 bytes before it, 16 bytes of versions after.
    d.uuid = "fbc062f1-d501-4c3c-be5f-bdd373a239b3"_nvrhi_guid;
    d.bufferInlineCapacity = 4088;
    CHECK(world->RegisterComponent(d, out) == nvrhi::FE_INVALID_ARGS);
    d.uuid = "d6604f89-dad9-4b0d-8844-ed8f22dcc378"_nvrhi_guid;
    d.bufferInlineCapacity = 4087;
    ComponentId largest = ComponentId::Invalid;
    CHECK(world->RegisterComponent(d, largest) == nvrhi::FS_OK);
    const Entity e = Create(world, { largest });
    const ArchetypeInfo info = InfoOf(world, e);
    CHECK(info.chunkSize == 65536 && info.rowsPerChunk == 1);
    CHECK(world->ResizeBuffer(e, largest, 4087, c_WorldVersion) == nvrhi::FS_OK);
    CHECK(HeaderOf(world, e, largest).data == InlineOf(world, e, largest));
    CHECK(world->ResizeBuffer(e, largest, 4088, c_WorldVersion) == nvrhi::FS_OK);
    CHECK(HeaderOf(world, e, largest).data != InlineOf(world, e, largest));
    CHECK(world->ResizeBuffer(e, largest, 1, c_WorldVersion) == nvrhi::FS_OK);
    CHECK(HeaderOf(world, e, largest).data == InlineOf(world, e, largest));

    // A world with smaller chunks refuses what fits 64 KiB.
    {
        auto small = MakeSizedWorld(4096, 8192);
        CHECK(small->RegisterComponent(d, out) == nvrhi::FE_INVALID_ARGS);
    }

    // Data and Chunk components alike.
    ComponentTypeDesc blob{};
    blob.name = "Blob";
    blob.alignment = 8;
    blob.flags = ComponentFlags::Relocatable;
    ComponentTypeDesc b = blob;
    b.uuid = "1f1f4fd4-79fe-4e71-b638-d9f51df85a73"_nvrhi_guid;
    b.size = 65464;
    CHECK(world->RegisterComponent(b, out) == nvrhi::FE_INVALID_ARGS);
    b.uuid = "4ef960a0-3756-45d5-8d0e-5a5ba0ec059f"_nvrhi_guid;
    b.size = 65456;
    CHECK(world->RegisterComponent(b, out) == nvrhi::FS_OK);
    b.uuid = "2db56a6e-fdea-40dc-a087-85ea601acc94"_nvrhi_guid;
    b.size = 65520;
    b.flags = ComponentFlags::Relocatable | ComponentFlags::Chunk;
    CHECK(world->RegisterComponent(b, out) == nvrhi::FE_INVALID_ARGS);
    b.uuid = "1facb4f6-94e9-42d4-812a-4f524c366d00"_nvrhi_guid;
    b.size = 65512;
    CHECK(world->RegisterComponent(b, out) == nvrhi::FS_OK);

    // Two that fit one by one but not together: FE_INVALID_ARGS, and no archetype is made.
    b = blob;
    b.size = 40000;
    b.uuid = "9e7fd7fe-b3dc-48c9-bede-601d6bd49b73"_nvrhi_guid;
    ComponentId halfA = ComponentId::Invalid;
    CHECK(world->RegisterComponent(b, halfA) == nvrhi::FS_OK);
    b.uuid = "f9666a26-9817-4110-be9d-3378ed2b438f"_nvrhi_guid;
    ComponentId halfB = ComponentId::Invalid;
    CHECK(world->RegisterComponent(b, halfB) == nvrhi::FS_OK);
    const ComponentId pair[2] = { halfA, halfB };
    CHECK(impl.RowFits(pair, 1) && !impl.RowFits(pair, 2));
    const uint32_t archetypes = world->GetArchetypeCount();
    Entity refused = Entity::Null;
    CHECK(world->CreateEntities(CreateDesc({ halfA, halfB }, {}, 1), &refused) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->GetArchetypeCount() == archetypes);
}

// Review item 2: hooks and Relocatable agree (D14), and a type that may not be memcpy'd is never copied with memcpy.
void test_hook_consistency()
{
    auto world = MakeWorld(true);
    ecs::WorldImpl& impl = Impl(world);

    // Every type the macros declare in this test registers under the rules.
    const ComponentId declared[] = {
        RegisterComponent<Position>(world), RegisterComponent<Velocity>(world), RegisterComponent<Id>(world),
        RegisterComponent<Big>(world), RegisterComponent<Wide>(world), RegisterComponent<Frozen>(world),
        RegisterComponent<Layer>(world), RegisterComponent<ChunkBounds>(world), RegisterComponent<Selected>(world),
        RegisterComponent<DebugColor>(world), RegisterComponent<Toggle>(world), RegisterComponent<Hidden>(world),
        RegisterComponent<Joint>(world), RegisterComponent<Link>(world), RegisterComponent<Handle>(world),
        RegisterComponent<MoveOnly>(world), RegisterComponent<FrameTime>(world),
    };
    for (const ComponentId component : declared)
        CHECK(component != ComponentId::Invalid);

    // Relocatable with a construct hook is fine.
    const ComponentId defaulted = RegisterComponent<Defaulted>(world);
    const ComponentTypeDesc* defaultedDesc = world->GetComponentTypeDesc(defaulted);
    CHECK(defaultedDesc->construct && !!(defaultedDesc->flags & ComponentFlags::Relocatable));
    CHECK(Get<Defaulted>(world, Create(world, { defaulted })).value == 7);

    const ComponentTypeDesc& handle = ComponentTraits<Handle>::Desc();
    const uint64_t errors = impl.GetValidationErrorCount();
    auto reject = [&](const ComponentTypeDesc& desc)
    {
        ComponentId out = ComponentId(0);
        CHECK(world->RegisterComponent(desc, out) == nvrhi::FE_INVALID_ARGS && out == ComponentId::Invalid);
    };
    // Relocatable and a destruct, relocate or copy hook: a snapshot's memcpy would release a reference twice.
    ComponentTypeDesc plain = ComponentTraits<Position>::Desc();
    plain.uuid = "9606f526-7c52-45e3-837c-db9bcfd4bccf"_nvrhi_guid;
    { ComponentTypeDesc d = plain; d.destruct = handle.destruct; reject(d); }
    { ComponentTypeDesc d = plain; d.relocate = handle.relocate; reject(d); }
    { ComponentTypeDesc d = plain; d.copy = handle.copy; reject(d); }
    // A destruct hook without a relocate hook.
    ComponentTypeDesc owning = handle;
    owning.uuid = "04ba90d2-830c-420e-b1ef-1a7ec7f8a087"_nvrhi_guid;
    { ComponentTypeDesc d = owning; d.relocate = nullptr; d.copy = nullptr; reject(d); }
    { ComponentTypeDesc d = owning; d.relocate = nullptr; reject(d); }
    // A Shared component that cannot be copied into its archetype's key.
    {
        ComponentTypeDesc d = ComponentTraits<MoveOnly>::Desc();
        d.uuid = "bcce2606-520a-4d6a-8bca-806a3c97a84e"_nvrhi_guid;
        d.flags = ComponentFlags::Shared;
        reject(d);
    }
    CHECK(impl.GetValidationErrorCount() == errors + 6);

    // Neither Relocatable nor a copy hook: registered and moved (null relocate: memcpy), but never copied.
    ComponentTypeDesc raw{};
    raw.uuid = "8c34e35e-57ea-43ee-88cc-c30886e1cd69"_nvrhi_guid;
    raw.name = "NotRelocatable";
    raw.size = 8;
    raw.alignment = 8;
    ComponentId opaque = ComponentId::Invalid;
    CHECK(world->RegisterComponent(raw, opaque) == nvrhi::FS_OK && !impl.GetType(opaque)->copyable);
    const uint64_t value = 42;
    Entity o = Entity::Null;
    CHECK(world->CreateEntities(CreateDesc({ opaque }, { &value }, 1), &o) == nvrhi::FE_UNSUPPORTED);
    CHECK(world->CreateEntities(CreateDesc({ opaque }, { &value }, 1, ValueTransfer::Relocate), &o) == nvrhi::FS_OK);
    CHECK(*static_cast<const uint64_t*>(world->GetComponent(o, opaque)) == 42);
    Entity copies[2];
    CHECK(world->Instantiate(o, 2, copies) == nvrhi::FE_UNSUPPORTED);
    CHECK(world->CopyEntitiesFrom(world, &o, 1, copies) == nvrhi::FE_UNSUPPORTED);
    const Entity other = Create(world, { FindComponent<Id>(world) });
    CHECK(world->AddComponent(other, opaque, &value) == nvrhi::FE_UNSUPPORTED);
    CHECK(world->AddComponent(other, opaque, nullptr) == nvrhi::FS_OK);
    CHECK(*static_cast<const uint64_t*>(world->GetComponent(other, opaque)) == 0);
    // As a resource it takes no initial value to copy.
    raw.uuid = "4e9268a5-4907-4a3a-98e0-c1ada84186c7"_nvrhi_guid;
    raw.flags = ComponentFlags::Resource;
    ComponentId resource = ComponentId::Invalid;
    CHECK(world->RegisterResource(raw, &value, resource) == nvrhi::FE_UNSUPPORTED);
    CHECK(world->RegisterResource(raw, nullptr, resource) == nvrhi::FS_OK);
}

// Review item 4: with ValueTransfer::Relocate an operation consumes every value it is given, also those it cannot
// place; with Copy the caller keeps them. Counted with the reference-counted payload: every path ends balanced.
void test_relocate_ownership()
{
    Payload payload;
    const HookCounts start = g_Hooks;
    // The handles alive are those the world holds, which all reference the payload, plus SharedHandle's default key
    // value (constructed at registration, without a payload).
    int64_t defaults = 0;
    auto held = [&](int64_t inWorld)
    {
        return payload.refs == inWorld && g_Hooks.Live() - start.Live() == inWorld + defaults;
    };
    {
        // 1 KiB chunks: Fill944 fills one, so that nothing fits beside it.
        auto world = MakeSizedWorld(1024, 1024);
        ecs::WorldImpl& impl = Impl(world);
        const uint64_t version = world->GetVersion();
        const ComponentId id = RegisterComponent<Id>(world);
        const ComponentId handle = RegisterComponent<Handle>(world);
        ComponentId fill = ComponentId::Invalid;
        CHECK(world->RegisterComponent(FillDesc(), fill) == nvrhi::FS_OK);
        const FieldDesc pointerField[] = { { "payload", 0, 8, FieldType::Opaque, FieldShape::Scalar, {} } };
        ComponentTypeDesc variant = ComponentTraits<Handle>::Desc();
        variant.uuid = "fcf03c10-89d5-46b3-aa59-d29f5347f098"_nvrhi_guid;
        variant.name = "SparseHandle";
        variant.flags = ComponentFlags::Sparse;
        ComponentId sparseHandle = ComponentId::Invalid;
        CHECK(world->RegisterComponent(variant, sparseHandle) == nvrhi::FS_OK);
        variant.uuid = "a210454e-3259-4238-9766-1a420fbf1e1a"_nvrhi_guid;
        variant.name = "SharedHandle";
        variant.flags = ComponentFlags::Shared;
        variant.fields = pointerField;
        variant.fieldCount = 1;
        ComponentId sharedHandle = ComponentId::Invalid;
        CHECK(world->RegisterComponent(variant, sharedHandle) == nvrhi::FS_OK);
        variant.uuid = "71ccb9d3-0646-436c-8474-752ae03b38f7"_nvrhi_guid;
        variant.name = "BufferHandle";
        variant.flags = ComponentFlags::Buffer;
        variant.bufferInlineCapacity = 2;
        variant.fields = nullptr;
        variant.fieldCount = 0;
        ComponentId bufferHandle = ComponentId::Invalid;
        CHECK(world->RegisterComponent(variant, bufferHandle) == nvrhi::FS_OK);
        defaults = g_Hooks.Live() - start.Live();
        CHECK(defaults == 1 && payload.refs == 0);

        const Entity plain = Create(world, { id });
        const Entity filled = Create(world, { fill });
        const Entity stale = MakeEntity(40, 1);   // never created
        Entity out[2];

        // CreateEntities refused as a whole: a row that does not fit, a component listed twice, an id that is not
        // reserved, a value for a Buffer, a phase.
        {
            RawHandles raw(&payload, 2);
            const CreateDesc desc({ handle, fill }, { raw.Get(), nullptr }, 2, ValueTransfer::Relocate);
            CHECK(world->CreateEntities(desc, out) == nvrhi::FE_INVALID_ARGS);
            CHECK(held(0));
        }
        {
            RawHandles a(&payload, 1);
            RawHandles b(&payload, 1);
            const CreateDesc desc({ handle, handle }, { a.Get(), b.Get() }, 1, ValueTransfer::Relocate);
            CHECK(world->CreateEntities(desc, out) == nvrhi::FE_INVALID_ARGS);
            CHECK(held(0));
        }
        {
            RawHandles raw(&payload, 1);
            CreateDesc desc({ handle }, { raw.Get() }, 1, ValueTransfer::Relocate);
            const Entity notReserved = MakeEntity(41, 1);
            desc->reservedEntities = &notReserved;
            CHECK(world->CreateEntities(desc, nullptr) == nvrhi::FE_INVALID_ARGS);
            CHECK(held(0));
        }
        {
            RawHandles raw(&payload, 2);
            CHECK(world->CreateEntities(CreateDesc({ bufferHandle }, { raw.Get() }, 2, ValueTransfer::Relocate), out) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(0));
        }
        {
            RawHandles raw(&payload, 1);
            impl.BeginPhase();
            CHECK(world->CreateEntities(CreateDesc({ handle }, { raw.Get() }, 1, ValueTransfer::Relocate), out) ==
                  nvrhi::FE_INVALID_ARGS);
            impl.EndPhase();
            CHECK(held(0));
        }
        // Created: Data and Sparse values move in; Shared values are destroyed (the archetype keeps one copy).
        Entity created[2];
        {
            RawHandles data(&payload, 2);
            RawHandles sparse(&payload, 2);
            RawHandles shared(&payload, 2);
            const CreateDesc desc({ id, handle, sparseHandle, sharedHandle },
                                  { nullptr, data.Get(), sparse.Get(), shared.Get() }, 2, ValueTransfer::Relocate);
            CHECK(world->CreateEntities(desc, created) == nvrhi::FS_OK);
            CHECK(held(2 + 2 + 1));
        }

        // AddComponentsInternal: a stale entity and a row that does not fit lose their values; `plain` takes its.
        {
            RawHandles raw(&payload, 3);
            const Entity targets[3] = { stale, plain, filled };
            CHECK(impl.AddComponentsInternal(targets, 3, handle, raw.Get(), ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(world->HasComponent(plain, handle) && !world->HasComponent(filled, handle));
            CHECK(held(6));
        }
        {
            // Shared: no fit beside Fill944, and the partition the entity is in already.
            RawHandles raw(&payload, 2);
            const Entity targets[2] = { filled, created[0] };
            CHECK(impl.AddComponentsInternal(targets, 2, sharedHandle, raw.Get(), ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(6));
        }
        {
            RawHandles raw(&payload, 1);
            CHECK(impl.AddComponentsInternal(&stale, 1, sparseHandle, raw.Get(), ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(6));
        }
        {
            RawHandles raw(&payload, 1);
            CHECK(impl.AddComponentsInternal(&plain, 1, bufferHandle, raw.Get(), ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(6));
        }

        // SetComponentsInternal: stale, absent Data and absent Sparse lose their values; present ones take them.
        {
            RawHandles raw(&payload, 2);
            const Entity targets[2] = { stale, filled };
            CHECK(impl.SetComponentsInternal(targets, 2, handle, raw.Get(), ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(6));
        }
        {
            RawHandles raw(&payload, 1);
            CHECK(impl.SetComponentsInternal(&plain, 1, sparseHandle, raw.Get(), ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(6));
        }
        {
            RawHandles raw(&payload, 2);
            const Entity targets[2] = { plain, created[1] };
            CHECK(impl.SetComponentsInternal(targets, 2, handle, raw.Get(), ValueTransfer::Relocate, version) ==
                  nvrhi::FS_OK);
            CHECK(held(6));   // the old values destroyed, the new ones moved in
        }
        {
            RawHandles raw(&payload, 1);
            CHECK(impl.SetComponentsInternal(&created[0], 1, sharedHandle, raw.Get(), ValueTransfer::Relocate,
                                             version) == nvrhi::FS_OK);
            CHECK(held(6));   // the same partition: the value is destroyed
        }

        // AppendToBufferInternal: a stale entity and an absent buffer lose their elements; a buffer takes them.
        const Entity buffered = Create(world, { id, bufferHandle });
        {
            RawHandles raw(&payload, 2);
            CHECK(impl.AppendToBufferInternal(stale, bufferHandle, raw.Get(), 2, ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(6));
        }
        {
            RawHandles raw(&payload, 2);
            CHECK(impl.AppendToBufferInternal(plain, bufferHandle, raw.Get(), 2, ValueTransfer::Relocate, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(held(6));
        }
        {
            RawHandles raw(&payload, 3);
            CHECK(impl.AppendToBufferInternal(buffered, bufferHandle, raw.Get(), 3, ValueTransfer::Relocate, version) ==
                  nvrhi::FS_OK);
            CHECK(HeaderOf(world, buffered, bufferHandle).count == 3);
            CHECK(held(9));
        }

        // Copy: the caller keeps its values, whatever happens.
        {
            const Handle kept[2] = { Handle(&payload), Handle(&payload) };
            CHECK(impl.AddComponentsInternal(&stale, 1, handle, kept, ValueTransfer::Copy, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(impl.SetComponentsInternal(&filled, 1, handle, kept, ValueTransfer::Copy, version) ==
                  nvrhi::FE_INVALID_ARGS);
            CHECK(payload.refs == 9 + 2 && kept[0].payload == &payload && kept[1].payload == &payload);
        }
        CHECK(held(9));
    }
    CHECK(payload.refs == 0 && g_Hooks.Live() == start.Live());
}

// Review items 5 and 9: running out of memory returns FE_OUT_OF_MEMORY (a row that does not fit is FE_INVALID_ARGS),
// changes nothing, reports nothing to observers, and consumes relocated values.
void test_out_of_memory()
{
    Payload payload;
    const HookCounts start = g_Hooks;
    {
        // 256 KiB chunks: one page per slab, so that a chunk the pool cannot recycle is a new heap block.
        auto world = MakeSizedWorld(256 * 1024, 256 * 1024);
        ecs::WorldImpl& impl = Impl(world);
        const ComponentId id = RegisterComponent<Id>(world);
        const ComponentId handle = RegisterComponent<Handle>(world);
        const ComponentId position = RegisterComponent<Position>(world);
        const ComponentId velocity = RegisterComponent<Velocity>(world);
        const ComponentId big = RegisterComponent<Big>(world);

        // {Id, Handle}, reached from {Id} through a cached edge, without a chunk; no free page in the pool. The
        // scratch storage of three-component operations is warm.
        const Entity e = Create(world, { id });
        Create(world, { id, position, velocity });
        const Entity x = Create(world, { id });
        CHECK(world->AddComponent(x, handle, nullptr) == nvrhi::FS_OK);
        const uint32_t withHandle = ArchetypeOf(world, x);
        CHECK(world->DestroyEntities(&x, 1) == nvrhi::FS_OK);
        Create(world, { velocity });   // takes the page {Id, Handle} gave back
        CHECK(impl.GetFreeChunkPageCount() == 0 && impl.GetArchetypeStorage(withHandle).GetChunkCount() == 0);

        // A move whose target needs a chunk: nothing is reported, nothing moves, the value is consumed.
        const uint64_t added = impl.GetCounters().rowsAddedReports;
        const uint64_t removing = impl.GetCounters().rowsRemovingReports;
        const uint32_t home = ArchetypeOf(world, e);
        {
            RawHandles raw(&payload, 1);
            ecs::SetHeapFailureCountdown(0);
            CHECK(impl.AddComponentsInternal(&e, 1, handle, raw.Get(), ValueTransfer::Relocate, world->GetVersion()) ==
                  nvrhi::FE_OUT_OF_MEMORY);
        }
        CHECK(payload.refs == 0 && g_Hooks.Live() == start.Live());
        CHECK(impl.GetCounters().rowsRemovingReports == removing && impl.GetCounters().rowsAddedReports == added);
        CHECK(ArchetypeOf(world, e) == home && impl.GetArchetypeStorage(withHandle).GetChunkCount() == 0);
        // The failure was injected once: the same move succeeds, and is reported.
        CHECK(world->AddComponent(e, handle, nullptr) == nvrhi::FS_OK && ArchetypeOf(world, e) == withHandle);
        CHECK(impl.GetCounters().rowsRemovingReports == removing + 1);
        CHECK(impl.GetCounters().rowsAddedReports == added + 1);

        // A new archetype that cannot be allocated: FE_OUT_OF_MEMORY, not "does not fit".
        const uint32_t archetypes = world->GetArchetypeCount();
        WorldStats stats;
        const uint32_t entities = world->GetStats(stats).entityCount;
        {
            RawHandles raw(&payload, 2);
            const CreateDesc desc({ id, handle, position }, { nullptr, raw.Get(), nullptr }, 2,
                                  ValueTransfer::Relocate);
            Entity out[2];
            ecs::SetHeapFailureCountdown(0);
            CHECK(world->CreateEntities(desc, out) == nvrhi::FE_OUT_OF_MEMORY);
        }
        CHECK(payload.refs == 0 && world->GetArchetypeCount() == archetypes);
        CHECK(world->GetStats(stats).entityCount == entities);
        ecs::SetHeapFailureCountdown(0);
        CHECK(world->AddComponent(e, big, nullptr) == nvrhi::FE_OUT_OF_MEMORY);
        CHECK(world->GetArchetypeCount() == archetypes && !world->HasComponent(e, big));

        // A full chunk and no free page: the creation is refused and its values consumed.
        const ArchetypeInfo info = InfoOf(world, e);   // {Id, Handle}: e alone in its first chunk
        CHECK(info.entityCount == 1 && info.chunkCount == 1);
        std::vector<Entity> rows(info.rowsPerChunk - 1);
        const CreateDesc fill({ id, handle }, {}, uint32_t(rows.size()));
        CHECK(world->CreateEntities(fill, rows.data()) == nvrhi::FS_OK);
        Create(world, { id });   // takes the page {Id} gave back when e left it
        CHECK(InfoOf(world, e).chunkCount == 1 && impl.GetFreeChunkPageCount() == 0);
        {
            RawHandles raw(&payload, 2);
            const CreateDesc desc({ id, handle }, { nullptr, raw.Get() }, 2, ValueTransfer::Relocate);
            Entity out[2];
            ecs::SetHeapFailureCountdown(0);
            CHECK(world->CreateEntities(desc, out) == nvrhi::FE_OUT_OF_MEMORY);
        }
        CHECK(payload.refs == 0);
        CHECK(InfoOf(world, e).entityCount == info.rowsPerChunk && InfoOf(world, e).chunkCount == 1);

        // With memory, the same calls work.
        Create(world, { id, handle, position });
        CHECK(world->AddComponent(e, big, nullptr) == nvrhi::FS_OK && world->HasComponent(e, big));
        ecs::SetHeapFailureCountdown(-1);
    }
    CHECK(payload.refs == 0 && g_Hooks.Live() == start.Live());

    // Sparse presence masks (E1.2 review): a change that needs a chunk's first mask and cannot allocate it fails with
    // FE_OUT_OF_MEMORY and changes nothing; it used to succeed and leave the row out of the mask, so queries with a
    // With term on the component never saw it. The presence vectors are sized beforehand, so that the mask is the
    // block that fails.
    {
        auto world = MakeWorld(true);
        ecs::WorldImpl& impl = Impl(world);
        const ComponentId position = RegisterComponent<Position>(world);
        const ComponentId velocity = RegisterComponent<Velocity>(world);
        const ComponentId selected = RegisterComponent<Selected>(world);
        const ecs::SparseSet& set = *impl.GetSparseSet(selected);
        auto present = [&](Entity entity) {
            const ecs::EntityRecord& record = Record(world, entity);
            return ecs::GetMaskBit(impl.GetSparsePresence(set, record.archetype, record.chunk), record.row);
        };
        auto lastRow = [&](uint32_t archetype) {
            const ecs::Archetype& a = impl.GetArchetypeStorage(archetype);
            const uint32_t chunk = a.GetChunkCount() - 1;
            return a.Entities(chunk)[a.chunks[chunk].count - 1];
        };

        // {Position}: four chunks; a selected row in the last one sizes the vectors and leaves only that mask.
        const Entity first = Create(world, { position });
        const uint32_t rows = InfoOf(world, first).rowsPerChunk;
        const uint32_t p = ArchetypeOf(world, first);
        std::vector<Entity> plain(3 * rows + 10);
        CHECK(world->CreateEntities(CreateDesc({ position }, {}, uint32_t(plain.size())), plain.data()) ==
              nvrhi::FS_OK);
        CHECK(InfoOf(world, first).chunkCount == 4);
        CHECK(world->AddComponent(plain.back(), selected, nullptr) == nvrhi::FS_OK);
        CHECK(world->RemoveComponent(plain.back(), selected) == nvrhi::FS_OK);

        // Adding the component.
        const Entity a = plain[0];
        const uint64_t masks = impl.GetPresenceMaskVersion();
        ecs::SetHeapFailureCountdown(0);
        CHECK(world->AddComponent(a, selected, nullptr) == nvrhi::FE_OUT_OF_MEMORY);
        CHECK(!world->HasComponent(a, selected) && set.GetCount() == 0 && Record(world, a).sparseCount == 0);
        CHECK(impl.GetSparsePresence(set, p, 0) == ecs::g_ZeroMask && impl.GetPresenceMaskVersion() == masks);
        CHECK(world->AddComponent(a, selected, nullptr) == nvrhi::FS_OK && present(a));

        // A move into a chunk without the mask. {Position, Velocity}: a selected row once in its second chunk, then
        // only part of the first chunk left; the edge from {Position} is cached by a move without Sparse components.
        const Entity pvFirst = Create(world, { position, velocity });
        const uint32_t pvRows = InfoOf(world, pvFirst).rowsPerChunk;
        const uint32_t pv = ArchetypeOf(world, pvFirst);
        std::vector<Entity> moved(pvRows + 4);
        CHECK(world->CreateEntities(CreateDesc({ position, velocity }, {}, uint32_t(moved.size())), moved.data()) ==
              nvrhi::FS_OK);
        CHECK(InfoOf(world, pvFirst).chunkCount == 2);
        CHECK(world->AddComponent(moved.back(), selected, nullptr) == nvrhi::FS_OK);
        CHECK(world->RemoveComponent(moved.back(), selected) == nvrhi::FS_OK);
        CHECK(world->DestroyEntities(moved.data() + pvRows - 11, 15) == nvrhi::FS_OK);
        CHECK(world->AddComponent(plain[5], velocity, nullptr) == nvrhi::FS_OK);
        CHECK(InfoOf(world, pvFirst).chunkCount == 1 && InfoOf(world, pvFirst).entityCount == pvRows - 9);
        const uint64_t removing = impl.GetCounters().rowsRemovingReports;
        ecs::SetHeapFailureCountdown(0);
        CHECK(world->AddComponent(a, velocity, nullptr) == nvrhi::FE_OUT_OF_MEMORY);
        CHECK(ArchetypeOf(world, a) == p && world->HasComponent(a, selected) && present(a));
        CHECK(impl.GetCounters().rowsRemovingReports == removing);
        CHECK(world->AddComponent(a, velocity, nullptr) == nvrhi::FS_OK);
        CHECK(ArchetypeOf(world, a) == pv && Record(world, a).chunk == 0 && present(a));

        // A move whose swap-remove brings a selected last row into a chunk without the mask.
        const Entity last = lastRow(p);
        CHECK(Record(world, last).chunk == 3 && world->AddComponent(last, selected, nullptr) == nvrhi::FS_OK);
        const Entity mover = plain[rows + 3];
        CHECK(Record(world, mover).chunk == 1);
        ecs::SetHeapFailureCountdown(0);
        CHECK(world->AddComponent(mover, velocity, nullptr) == nvrhi::FE_OUT_OF_MEMORY);
        CHECK(ArchetypeOf(world, mover) == p && Record(world, last).chunk == 3 && present(last));
        CHECK(impl.GetCounters().rowsRemovingReports == removing + 1);
        CHECK(world->AddComponent(mover, velocity, nullptr) == nvrhi::FS_OK);
        CHECK(Record(world, last).chunk == 1 && present(last));

        // A destruction whose swap-remove does the same.
        const Entity last2 = lastRow(p);
        CHECK(Record(world, last2).chunk == 3 && world->AddComponent(last2, selected, nullptr) == nvrhi::FS_OK);
        const Entity victim = plain[2 * rows + 3];
        CHECK(Record(world, victim).chunk == 2);
        ecs::SetHeapFailureCountdown(0);
        CHECK(world->DestroyEntities(&victim, 1) == nvrhi::FE_OUT_OF_MEMORY);
        CHECK(world->IsAlive(victim) && Record(world, last2).chunk == 3 && present(last2));
        CHECK(world->DestroyEntities(&victim, 1) == nvrhi::FS_OK);
        CHECK(!world->IsAlive(victim) && Record(world, last2).chunk == 2 && present(last2));

        // Every member of the set is in its chunk's mask.
        CHECK(set.GetCount() == 3);
        for (const Entity entity : set.entities)
            CHECK(present(entity));
        ecs::SetHeapFailureCountdown(-1);
        CHECK(impl.GetValidationErrorCount() == 0);
    }
}

// Review items 6 and 7: CopyEntitiesFrom checks that every copy fits before it takes an id or makes a row, and its
// output may be its input list.
void test_copy_prepass_and_alias()
{
    auto source = MakeWorld(true);
    const ComponentId sId = RegisterComponent<Id>(source);
    const ComponentId sLink = RegisterComponent<Link>(source);
    ComponentId sFill = ComponentId::Invalid;
    CHECK(source->RegisterComponent(FillDesc(), sFill) == nvrhi::FS_OK);
    const Id i0{ 0 }, i1{ 1 };
    const Entity list[2] = { Create(source, { sId }, { &i0 }), Create(source, { sId, sFill }, { &i1, nullptr }) };

    // A target with 1 KiB chunks, where {Id, Fill944} does not fit; the first copy would.
    auto target = MakeSizedWorld(1024, 1024);
    const ComponentId tId = RegisterComponent<Id>(target);
    RegisterComponent<Link>(target);
    ComponentId tFill = ComponentId::Invalid;
    CHECK(target->RegisterComponent(FillDesc(), tFill) == nvrhi::FS_OK);
    const uint32_t archetypes = target->GetArchetypeCount();
    Entity out[2] = { Entity::Null, Entity::Null };
    CHECK(target->CopyEntitiesFrom(source, list, 2, out) == nvrhi::FE_INVALID_ARGS);
    WorldStats stats;
    CHECK(out[0] == Entity::Null && out[1] == Entity::Null);
    CHECK(target->GetArchetypeCount() == archetypes && target->GetStats(stats).entityCount == 0);
    CHECK(Create(target, { tId }) == MakeEntity(1, ecs::c_FirstGeneration));   // no id was taken

    // The output may be the list itself, within one world or across two.
    const Id ids[2] = { { 10 }, { 11 } };
    Entity a[2];
    CHECK(source->CreateEntities(CreateDesc({ sId, sLink }, { ids, nullptr }, 2), a) == nvrhi::FS_OK);
    static_cast<Link*>(source->GetComponentForWrite(a[0], sLink, c_WorldVersion))->other = a[1];
    static_cast<Link*>(source->GetComponentForWrite(a[1], sLink, c_WorldVersion))->other = a[0];
    Entity inPlace[2] = { a[0], a[1] };
    CHECK(source->CopyEntitiesFrom(source, inPlace, 2, inPlace) == nvrhi::FS_OK);
    CHECK(inPlace[0] != a[0] && inPlace[1] != a[1]);
    CHECK(Get<Id>(source, inPlace[0]).value == 10 && Get<Id>(source, inPlace[1]).value == 11);
    CHECK(Get<Link>(source, inPlace[0]).other == inPlace[1] && Get<Link>(source, inPlace[1]).other == inPlace[0]);
    auto other = MakeWorld(true);
    RegisterComponent<Link>(other);
    RegisterComponent<Id>(other);
    Entity across[2] = { a[0], a[1] };
    CHECK(other->CopyEntitiesFrom(source, across, 2, across) == nvrhi::FS_OK);
    CHECK(Get<Id>(other, across[0]).value == 10 && Get<Link>(other, across[0]).other == across[1]);
    CHECK(Get<Link>(other, across[1]).other == across[0]);
}

void test_unimplemented_seams()
{
    auto world = MakeWorld();
    const QueryDesc queryDesc{};
    IQuery* query = nullptr;
    CHECK(world->CreateQuery(queryDesc, &query) == nvrhi::FE_INVALID_ARGS && !query);   // E1.2: no terms
    ICommandBuffer* buffer = nullptr;
    CHECK(world->CreateCommandBuffer(&buffer) == nvrhi::FE_NOT_IMPLEMENT && !buffer);
    CHECK(world->Playback(nullptr, 0) == nvrhi::FE_NOT_IMPLEMENT);
    const ObserverDesc observerDesc{};
    ObserverId observer = ObserverId(0);
    CHECK(world->AddObserver(observerDesc, nullptr, observer) == nvrhi::FE_NOT_IMPLEMENT);
    CHECK(observer == ObserverId::Invalid);
    CHECK(world->RemoveObserver(ObserverId(0)) == nvrhi::FE_NOT_IMPLEMENT);
    CHECK(world->SaveSnapshot(nullptr) == nvrhi::FE_INVALID_ARGS);   // E1.5: test_ecs_snapshot
    CHECK(world->LoadSnapshot(nullptr) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->ComputeHash() != 0);
    CHECK(world->ResolvePlaceholder(MakeEntity(1, c_PlaceholderGeneration)) == Entity::Null);
}

void test_stats_and_leaks()
{
    const int64_t blocks = ecs::GetHeapLiveBlockCount();
    {
        auto world = MakeWorld();
        const ComponentId id = RegisterComponent<Id>(world);
        const ComponentId position = RegisterComponent<Position>(world);
        const ComponentId big = RegisterComponent<Big>(world);
        const ComponentId selected = RegisterComponent<Selected>(world);
        const ComponentId joint = RegisterComponent<Joint>(world);

        std::vector<Entity> e(1000);
        CHECK(world->CreateEntities(CreateDesc({ id, position }, {}, 1000), e.data()) == nvrhi::FS_OK);
        std::vector<Entity> b(10);
        CHECK(world->CreateEntities(CreateDesc({ big }, {}, 10), b.data()) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < 100; ++i)
            CHECK(world->AddComponent(e[i], selected, nullptr) == nvrhi::FS_OK);
        const Entity buffered = Create(world, { id, joint });
        CHECK(world->ResizeBuffer(buffered, joint, 9, c_WorldVersion) == nvrhi::FS_OK);

        WorldStats stats;
        world->GetStats(stats);
        const ArchetypeInfo small = InfoOf(world, e[0]);
        const ArchetypeInfo large = InfoOf(world, b[0]);
        const ArchetypeInfo withBuffer = InfoOf(world, buffered);
        CHECK(stats.entityCount == 1011 && stats.entityCapacity == 1024);
        CHECK(stats.componentTypeCount == 5 && stats.archetypeCount == 4 && stats.sparseSetCount == 1);
        CHECK(stats.chunkCount == small.chunkCount + large.chunkCount + withBuffer.chunkCount);
        CHECK(small.chunkCount == (1000 + small.rowsPerChunk - 1) / small.rowsPerChunk);
        CHECK(stats.chunkBytes == uint64_t(small.chunkCount) * 16384 + uint64_t(large.chunkCount) * 65536 +
                                  uint64_t(withBuffer.chunkCount) * withBuffer.chunkSize);
        CHECK(stats.rowBytes == 1000ull * small.rowSize + 10ull * large.rowSize + withBuffer.rowSize);
        CHECK(stats.bufferHeapBytes == uint64_t(HeaderOf(world, buffered, joint).capacity) * sizeof(Joint));
        CHECK(stats.commandsPlayedBack == 0 && stats.observerBatches == 0);
        CHECK(ecs::GetHeapLiveBlockCount() > blocks);
    }
    CHECK(ecs::GetHeapLiveBlockCount() == blocks);   // a released world frees every block
}

// ------------------------------------------------------------------------------------------- determinism

namespace
{
    struct ScriptIds
    {
        ComponentId id, position, velocity, frozen, toggle, layer, joint, selected, color, big;
    };

    ScriptIds RegisterScriptTypes(IWorld* world)
    {
        ScriptIds c;
        c.id = RegisterComponent<Id>(world);
        c.position = RegisterComponent<Position>(world);
        c.velocity = RegisterComponent<Velocity>(world);
        c.frozen = RegisterComponent<Frozen>(world);
        c.toggle = RegisterComponent<Toggle>(world);
        c.layer = RegisterComponent<Layer>(world);
        c.joint = RegisterComponent<Joint>(world);
        c.selected = RegisterComponent<Selected>(world);
        c.color = RegisterComponent<DebugColor>(world);
        c.big = RegisterComponent<Big>(world);
        return c;
    }

    // A seeded sequence of every immediate operation.
    void RunScript(IWorld* world, uint64_t seed, std::vector<Entity>& alive)
    {
        const ScriptIds c = RegisterScriptTypes(world);
        Xorshift rng{ seed };
        uint64_t serial = 0;
        for (uint32_t step = 0; step < 3000; ++step)
        {
            const uint32_t r = rng.Below(100);
            if (r < 30 || alive.empty())
            {
                const uint32_t count = 1 + rng.Below(8);
                Id ids[8];
                Position positions[8];
                Toggle toggles[8];
                Layer layers[8];
                std::memset(layers, 0, sizeof(layers));
                for (uint32_t k = 0; k < count; ++k)
                {
                    ids[k].value = serial++;
                    positions[k] = Position{ float(rng.Below(1000)), float(k), float(step) };
                    toggles[k].value = int32_t(k);
                    layers[k].id = uint8_t(rng.Below(3));
                    layers[k].mask = 1;
                }
                ComponentId components[10];
                const void* values[10];
                uint32_t n = 0;
                components[n] = c.id; values[n++] = ids;
                if (rng.Below(2)) { components[n] = c.position; values[n++] = positions; }
                if (rng.Below(4) == 0) { components[n] = c.velocity; values[n++] = nullptr; }
                if (rng.Below(4) == 0) { components[n] = c.frozen; values[n++] = nullptr; }
                if (rng.Below(3) == 0) { components[n] = c.toggle; values[n++] = toggles; }
                if (rng.Below(3) == 0) { components[n] = c.layer; values[n++] = layers; }
                if (rng.Below(4) == 0) { components[n] = c.joint; values[n++] = nullptr; }
                if (rng.Below(5) == 0) { components[n] = c.color; values[n++] = nullptr; }
                if (rng.Below(10) == 0) { components[n] = c.big; values[n++] = nullptr; }
                EntityCreateDesc desc{};
                desc.components = components;
                desc.values = values;
                desc.componentCount = n;
                desc.entityCount = count;
                Entity out[8];
                CHECK(world->CreateEntities(desc, out) == nvrhi::FS_OK);
                alive.insert(alive.end(), out, out + count);
                continue;
            }
            const uint32_t pick = rng.Below(uint32_t(alive.size()));
            const Entity e = alive[pick];
            const ComponentId choices[7] = { c.velocity, c.frozen, c.toggle, c.layer, c.joint, c.selected, c.color };
            if (r < 45)
            {
                CHECK(world->DestroyEntities(&e, 1) == nvrhi::FS_OK);
                alive[pick] = alive.back();
                alive.pop_back();
            }
            else if (r < 60)
            {
                const ComponentId component = choices[rng.Below(7)];
                const Toggle toggle{ int32_t(step) };
                Layer layer;
                std::memset(&layer, 0, sizeof(layer));
                layer.id = uint8_t(rng.Below(3));
                layer.mask = 2;
                const DebugColor color{ step };
                const void* value = component == c.toggle ? static_cast<const void*>(&toggle)
                                  : component == c.layer ? static_cast<const void*>(&layer)
                                  : component == c.color ? static_cast<const void*>(&color) : nullptr;
                CHECK(world->AddComponent(e, component, value) == nvrhi::FS_OK);
            }
            else if (r < 72)
            {
                CHECK(world->RemoveComponent(e, choices[rng.Below(7)]) == nvrhi::FS_OK);
            }
            else if (r < 80)
            {
                (void)world->SetEnabled(e, c.toggle, rng.Below(2) != 0);
            }
            else if (r < 88)
            {
                (void)world->ResizeBuffer(e, c.joint, rng.Below(11), c_WorldVersion);
            }
            else if (r < 94)
            {
                Entity out[4];
                const uint32_t count = 1 + rng.Below(4);
                CHECK(world->Instantiate(e, count, out) == nvrhi::FS_OK);
                alive.insert(alive.end(), out, out + count);
            }
            else
            {
                if (Position* p = static_cast<Position*>(world->GetComponentForWrite(e, c.position, c_WorldVersion)))
                    p->x += 1;
                world->AdvanceVersion();
            }
        }
        // The next ids show the free list's order.
        Entity reserved[8];
        CHECK(world->ReserveEntities(8, reserved) == nvrhi::FS_OK);
        alive.insert(alive.end(), reserved, reserved + 8);
    }

    // Everything the layout is made of: archetypes in order, rows in order with entities and field bytes, enabled bits,
    // buffers, chunk and shared values, versions, sparse sets in dense order, the entity table with its free list.
    std::vector<uint64_t> DumpWorld(IWorld* world)
    {
        ecs::WorldImpl& impl = Impl(world);
        std::vector<uint64_t> out;
        auto fieldHash = [](const ecs::ComponentType& type, const void* value, uint32_t count)
        {
            Fnv h;
            for (uint32_t k = 0; k < count; ++k)
                for (const ecs::ByteRange& range : type.fieldBytes)
                    h.Add(static_cast<const uint8_t*>(value) + size_t(k) * type.desc.size + range.offset, range.size);
            return h.hash;
        };
        out.push_back(impl.GetArchetypeStorageCount());
        for (uint32_t ai = 0; ai < impl.GetArchetypeStorageCount(); ++ai)
        {
            const ecs::Archetype& a = impl.GetArchetypeStorage(ai);
            out.push_back(a.components.size());
            for (const ComponentId component : a.components)
                out.push_back(uint32_t(component));
            out.push_back(a.entityCount);
            out.push_back(a.GetChunkCount());
            out.push_back(a.rowsPerChunk);
            for (uint32_t c = 0; c < a.columns.size(); ++c)
                if (a.columns[c].kind == ecs::StorageKind::Shared)
                    out.push_back(fieldHash(*a.columns[c].type, a.sharedValues + a.columns[c].offset, 1));
            for (uint32_t chunk = 0; chunk < a.GetChunkCount(); ++chunk)
            {
                for (uint32_t v = 0; v <= a.columns.size(); ++v)
                    out.push_back(a.Versions(chunk)[v]);
                for (uint32_t row = 0; row < a.chunks[chunk].count; ++row)
                {
                    out.push_back(uint64_t(a.Entities(chunk)[row]));
                    for (uint32_t c = 0; c < a.columns.size(); ++c)
                    {
                        const ecs::ArchetypeColumn& column = a.columns[c];
                        const uint8_t* base = static_cast<const uint8_t*>(a.ColumnData(chunk, c));
                        if (column.kind == ecs::StorageKind::Data)
                        {
                            out.push_back(fieldHash(*column.type, base + size_t(row) * column.stride, 1));
                        }
                        else if (column.kind == ecs::StorageKind::Buffer)
                        {
                            const BufferHeader& header = reinterpret_cast<const BufferHeader*>(base)[row];
                            out.push_back(header.count);
                            out.push_back(header.capacity);
                            out.push_back(fieldHash(*column.type, header.data, header.count));
                        }
                        if (column.enableable)
                            out.push_back(ecs::GetMaskBit(a.EnabledMask(chunk, c), row) ? 1 : 0);
                    }
                }
            }
        }
        for (uint32_t t = 0; t < world->GetComponentTypeCount(); ++t)
        {
            const ecs::SparseSet* set = impl.GetSparseSet(ComponentId(t));
            if (!set)
                continue;
            out.push_back(set->GetCount());
            for (uint32_t d = 0; d < set->GetCount(); ++d)
            {
                out.push_back(uint64_t(set->entities[d]));
                if (set->type->desc.size != 0)
                    out.push_back(fieldHash(*set->type, set->Value(d), 1));
            }
        }
        for (uint32_t index = 0; index < impl.GetEntityTableSize(); ++index)
        {
            const ecs::EntityRecord& record = impl.GetEntityRecord(index);
            out.push_back(record.generation);
            out.push_back(record.archetype);
            out.push_back(record.chunk);
            out.push_back(record.row);
            out.push_back(record.sparseCount);
        }
        return out;
    }
}

void test_determinism()
{
    const uint64_t seed = 0x9e3779b97f4a7c15ull;
    std::vector<uint64_t> first;
    std::vector<uint64_t> second;
    size_t aliveCount = 0;
    {
        auto world = MakeWorld();
        std::vector<Entity> alive;
        RunScript(world, seed, alive);
        first = DumpWorld(world);
        aliveCount = alive.size();
        WorldStats stats;
        CHECK(world->GetStats(stats).entityCount + 8 == alive.size());   // the 8 reserved ids are not alive
        CHECK(stats.archetypeCount > 20 && stats.entityCount > 500);
        for (size_t i = 0; i + 8 < alive.size(); ++i)
            CHECK(world->IsAlive(alive[i]));
    }

    // The second run sees other heap addresses: blocks of the process heap and of nvrhi's allocator in between.
    std::vector<std::unique_ptr<uint8_t[]>> noise;
    for (uint32_t i = 0; i < 1000; ++i)
        noise.emplace_back(new uint8_t[17 + (i * 131) % 4096]);
    void* blocks[64];
    for (uint32_t i = 0; i < 64; ++i)
        blocks[i] = ecs::HeapAllocate(1024 + i * 4096, 64);
    {
        auto world = MakeWorld();
        std::vector<Entity> alive;
        RunScript(world, seed, alive);
        second = DumpWorld(world);
        CHECK(alive.size() == aliveCount);
    }
    for (void* block : blocks)
        ecs::HeapFree(block);

    CHECK(first == second);
    Fnv hash;
    hash.Add(first.data(), first.size() * sizeof(uint64_t));
    printf("test_ecs_world: determinism: %zu words, layout hash %016llx\n", first.size(),
           (unsigned long long)hash.hash);
}

// ------------------------------------------------------------------------------- steady-state allocations

void test_steady_state_allocations()
{
    auto source = MakeWorld();
    {
        const ComponentId id = RegisterComponent<Id>(source);
        const ComponentId position = RegisterComponent<Position>(source);
        const ComponentId link = RegisterComponent<Link>(source);
        (void)id;
        (void)position;
        (void)link;
    }
    Entity sourceEntities[8];
    CHECK(source->CreateEntities(CreateDesc({ FindComponent<Id>(source), FindComponent<Position>(source),
                                            FindComponent<Link>(source) }, {}, 8), sourceEntities) == nvrhi::FS_OK);
    const ComponentId sourceLink = FindComponent<Link>(source);
    for (uint32_t i = 0; i < 8; ++i)
    {
        Link* link = static_cast<Link*>(source->GetComponentForWrite(sourceEntities[i], sourceLink, c_WorldVersion));
        link->other = sourceEntities[(i + 1) % 8];
    }

    auto world = MakeWorld(false, 4096);
    const ComponentId id = RegisterComponent<Id>(world);
    const ComponentId position = RegisterComponent<Position>(world);
    const ComponentId velocity = RegisterComponent<Velocity>(world);
    const ComponentId toggle = RegisterComponent<Toggle>(world);
    const ComponentId joint = RegisterComponent<Joint>(world);
    const ComponentId selected = RegisterComponent<Selected>(world);
    const ComponentId color = RegisterComponent<DebugColor>(world);
    const ComponentId layer = RegisterComponent<Layer>(world);
    RegisterComponent<Link>(world);

    std::vector<Entity> base(1000);
    CHECK(world->CreateEntities(CreateDesc({ id, position, toggle, joint }, {}, 1000), base.data()) == nvrhi::FS_OK);

    // Everything a round needs is allocated here and in the warm-up rounds, never in the measured ones.
    std::vector<Entity> batch(100);
    std::vector<Id> batchIds(100);
    std::vector<Position> batchPositions(100);
    Entity instances[20];
    Entity copies[8];
    Layer first;
    std::memset(&first, 0, sizeof(first));
    first.id = 1;
    Layer second = first;
    second.id = 2;
    const Velocity v{ 1, 2, 3 };
    const DebugColor tint{ 5 };
    WorldStats stats;
    ArchetypeInfo info;
    uint64_t sink = 0;

    // The presence masks of every chunk that a sparse component will touch.
    for (const Entity entity : base)
    {
        CHECK(world->AddComponent(entity, selected, nullptr) == nvrhi::FS_OK);
        CHECK(world->AddComponent(entity, color, &tint) == nvrhi::FS_OK);
    }
    for (const Entity entity : base)
    {
        CHECK(world->RemoveComponent(entity, selected) == nvrhi::FS_OK);
        CHECK(world->RemoveComponent(entity, color) == nvrhi::FS_OK);
    }

    auto round = [&]()
    {
        // Creation and destruction in existing archetypes and chunks.
        const CreateDesc desc({ id, position }, { batchIds.data(), batchPositions.data() }, 100);
        CHECK(world->CreateEntities(desc, batch.data()) == nvrhi::FS_OK);
        CHECK(world->DestroyEntities(batch.data(), 100) == nvrhi::FS_OK);
        // Archetype moves through the cached edges, and Shared values between existing partitions.
        for (uint32_t i = 0; i < 50; ++i)
            CHECK(world->AddComponent(base[i], velocity, &v) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < 50; ++i)
            CHECK(world->RemoveComponent(base[i], velocity) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < 20; ++i)
            CHECK(world->AddComponent(base[i], layer, &second) == nvrhi::FS_OK);
        for (uint32_t i = 0; i < 20; ++i)
            CHECK(world->AddComponent(base[i], layer, &first) == nvrhi::FS_OK);
        // Sparse components.
        for (uint32_t i = 0; i < 50; ++i)
        {
            CHECK(world->AddComponent(base[i * 7], selected, nullptr) == nvrhi::FS_OK);
            CHECK(world->AddComponent(base[i * 7], color, &tint) == nvrhi::FS_OK);
        }
        for (uint32_t i = 0; i < 50; ++i)
        {
            CHECK(world->RemoveComponent(base[i * 7], selected) == nvrhi::FS_OK);
            CHECK(world->RemoveComponent(base[i * 7], color) == nvrhi::FS_OK);
        }
        // Reads, writes, versions, enabled bits, buffers within their inline capacity.
        for (uint32_t i = 0; i < 1000; ++i)
        {
            sink += uint64_t(static_cast<const Position*>(world->GetComponent(base[i], position))->x);
            static_cast<Position*>(world->GetComponentForWrite(base[i], position, c_WorldVersion))->x += 1;
            CHECK(world->SetEnabled(base[i], toggle, (i & 1) != 0) == nvrhi::FS_OK);
            sink += world->IsEnabled(base[i], toggle) ? 1 : 0;
            CHECK(world->ResizeBuffer(base[i], joint, i % 5, c_WorldVersion) == nvrhi::FS_OK);
            sink += world->HasComponent(base[i], velocity) ? 1 : 0;
        }
        world->AdvanceVersion();
        // Instantiation and copies from another world, destroyed again.
        CHECK(world->Instantiate(base[500], 20, instances) == nvrhi::FS_OK);
        CHECK(world->DestroyEntities(instances, 20) == nvrhi::FS_OK);
        CHECK(world->CopyEntitiesFrom(source, sourceEntities, 8, copies) == nvrhi::FS_OK);
        CHECK(world->DestroyEntities(copies, 8) == nvrhi::FS_OK);
        world->GetStats(stats);
        CHECK(world->GetArchetypeInfo(1, info) == nvrhi::FS_OK);
    };

    for (uint32_t i = 0; i < 3; ++i)
        round();
    const uint64_t news = g_NewCount.load();
    const uint64_t heapBlocks = ecs::GetHeapAllocationCount();
    for (uint32_t i = 0; i < 5; ++i)
        round();
    const uint64_t newsAfter = g_NewCount.load();
    const uint64_t heapBlocksAfter = ecs::GetHeapAllocationCount();
    printf("test_ecs_world: steady state: %llu operator new, %llu storage blocks in 5 rounds (sink %llu)\n",
           (unsigned long long)(newsAfter - news), (unsigned long long)(heapBlocksAfter - heapBlocks),
           (unsigned long long)(sink & 1));
    CHECK(newsAfter == news);
    CHECK(heapBlocksAfter == heapBlocks);

    // Both counters see allocations when there are some: an operator new, and the storage of a new archetype.
    int* probe = new int[16];
    sink += uintptr_t(probe) & 1;
    delete[] probe;
    CHECK(g_NewCount.load() > newsAfter);
    const Entity probeEntity = Create(world, { id, RegisterComponent<Big>(world) });
    CHECK(world->IsAlive(probeEntity) && ecs::GetHeapAllocationCount() > heapBlocksAfter);
}

int main(int, char**)
{
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "registration", &test_registration },
        { "entity_lifecycle", &test_entity_lifecycle },
        { "free_list_and_generations", &test_free_list_and_generations },
        { "reserved_entities", &test_reserved_entities },
        { "archetype_moves_and_edges", &test_archetype_moves_and_edges },
        { "swap_remove_density", &test_swap_remove_density },
        { "chunk_sizes_and_alignment", &test_chunk_sizes_and_alignment },
        { "versions", &test_versions },
        { "enableable", &test_enableable },
        { "shared_partitions", &test_shared_partitions },
        { "chunk_components", &test_chunk_components },
        { "sparse_sets", &test_sparse_sets },
        { "dynamic_buffers", &test_dynamic_buffers },
        { "hooks", &test_hooks },
        { "instantiate", &test_instantiate },
        { "copy_entities_from", &test_copy_entities_from },
        { "resources", &test_resources },
        { "validation_and_phases", &test_validation_and_phases },
        { "batch_internals", &test_batch_internals },
        { "registration_limits", &test_registration_limits },
        { "hook_consistency", &test_hook_consistency },
        { "relocate_ownership", &test_relocate_ownership },
        { "out_of_memory", &test_out_of_memory },
        { "copy_prepass_and_alias", &test_copy_prepass_and_alias },
        { "unimplemented_seams", &test_unimplemented_seams },
        { "stats_and_leaks", &test_stats_and_leaks },
        { "determinism", &test_determinism },
        { "steady_state_allocations", &test_steady_state_allocations },
    };
    try
    {
        for (const Test& test : tests)
        {
            test.run();
            printf("test_ecs_world: %s passed\n", test.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
