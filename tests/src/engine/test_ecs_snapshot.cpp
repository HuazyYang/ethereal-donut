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

// World snapshots and the world hash (aggregate docs/plans/2026-11-21-donut-ecs.md, step E1.5; design 11.1, 11.4):
// XXH64 against reference vectors, round trips that keep entity ids, generations and the free-list order, every
// component kind, failures that change nothing (corruption and truncation at every byte, structural corruption behind
// a valid checksum, unknown and mismatched components, versions, phases, out of memory at every allocation), the
// properties of the hash, a golden corpus of world hashes, and GetStats.
//
// Golden file, in DONUT_TEST_GOLDEN_DIR (set in tests/test-engine.cmake): ecs_world_hashes.txt, one line per seeded
// script: the world hash, the XXH64 of its snapshot, the snapshot's size and the entity count. ECS_UPDATE_GOLDEN=1
// rewrites it. Values are integers or exact binary fractions, so Debug and Release (/arch:AVX2) agree.

#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsSnapshot.h"
#include "EcsWorldImpl.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#ifndef DONUT_TEST_GOLDEN_DIR
#error "DONUT_TEST_GOLDEN_DIR is defined for this test in tests/test-engine.cmake"
#endif

// ------------------------------------------------------------------------------------------ test components

namespace snapshot_test
{
    using donut::engine::ComponentFlags;
    using donut::engine::Entity;

    struct Position { float x, y, z; };
    DONUT_ECS_COMPONENT(Position, "efd06311-aed1-4f85-bae6-9aace6de7752", DONUT_ECS_FIELDS(x, y, z));

    // Padding at bytes 1-3 and 10-15.
    struct Padded { uint8_t a; uint32_t b; uint16_t c; double d; };
    DONUT_ECS_COMPONENT(Padded, "06e1c8f2-08b0-4ac3-8009-c98600b6e4f2", DONUT_ECS_FIELDS(a, b, c, d));

    // Four bytes of tail padding.
    struct Link { Entity other; uint32_t tag; };
    DONUT_ECS_COMPONENT(Link, "ca653151-1e4b-495e-a676-753d5f0c2358", DONUT_ECS_FIELDS(other, tag));

    struct Frozen { };
    DONUT_ECS_TAG(Frozen, "afd373d6-ec9e-413f-bafe-5698263f7daa");

    // Three bytes of padding after `id`.
    struct Layer { uint8_t id; uint32_t mask; };
    DONUT_ECS_COMPONENT_EX(Layer, "430791fe-1550-4116-81e6-7c8594e441c1", ComponentFlags::Shared, 0,
                           DONUT_ECS_FIELDS(id, mask));

    struct Bounds { float lo, hi; };
    DONUT_ECS_COMPONENT_EX(Bounds, "81305184-a050-49cf-979c-0861ad5d9e48", ComponentFlags::Chunk, 0,
                           DONUT_ECS_FIELDS(lo, hi));

    // `pad` is not a field.
    struct Joint { Entity target; float weight; uint32_t pad; };
    DONUT_ECS_BUFFER(Joint, "47b7bff7-43ea-47c0-9420-259964c10fed", 4, DONUT_ECS_FIELDS(target, weight));

    struct Selected { };
    DONUT_ECS_TAG_EX(Selected, "39c4f410-9748-4ed5-a631-8b05bed0f86e", ComponentFlags::Sparse);

    struct Color { uint32_t rgba; };
    DONUT_ECS_COMPONENT_EX(Color, "813e0d21-94d9-429f-8f80-9886b3cef0c1", ComponentFlags::Sparse, 0,
                           DONUT_ECS_FIELDS(rgba));

    struct Toggle { int32_t value; };
    DONUT_ECS_COMPONENT_EX(Toggle, "5c371043-6e94-469d-a2d8-0366af988e02", ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(value));

    struct Hidden { };
    DONUT_ECS_TAG_EX(Hidden, "3800c55b-8f58-4740-b775-9263a52d0787", ComponentFlags::Enableable);

    // No fields: snapshots keep its bytes as they are, hashes see its presence only.
    struct Opaque { uint8_t raw[12]; };
    DONUT_ECS_COMPONENT(Opaque, "c98c32c2-55ac-4c15-add5-be0dfc7f1e02", DONUT_ECS_FIELDS());

    // A row over 128 bytes: 64 KiB chunks.
    struct Big { uint8_t bytes[200]; };
    DONUT_ECS_COMPONENT(Big, "0c2954dc-c17d-486a-9d82-5ba4e4f2df07", DONUT_ECS_FIELDS(bytes));

    struct alignas(64) Wide { float v[16]; };
    DONUT_ECS_COMPONENT(Wide, "36458943-7cf2-422c-8dde-2b14dad56f37", DONUT_ECS_FIELDS(v));

    // Relocatable, with a construct hook.
    struct Tracked { int32_t value = 7; };
    DONUT_ECS_COMPONENT(Tracked, "d1a35ee9-8a96-4716-a75d-b8a175a9e362", DONUT_ECS_FIELDS(value));

    // Not serialisable: NoSerialize ...
    struct Cache { uint32_t value; };
    DONUT_ECS_COMPONENT_EX(Cache, "ea0cc3c1-2c7d-4edc-8bbc-8c39fb9c802f", ComponentFlags::NoSerialize, 0,
                           DONUT_ECS_FIELDS(value));

    struct Note { uint32_t value; };
    DONUT_ECS_COMPONENT_EX(Note, "3c3a0f71-0d1c-4a16-934d-7bed46f2e364",
                           ComponentFlags::Sparse | ComponentFlags::NoSerialize, 0, DONUT_ECS_FIELDS(value));

    // ... or not Relocatable (hooks). Every live value is counted.
    int64_t g_OwnersLive = 0;
    struct Owner
    {
        uint32_t value = 0;
        Owner() noexcept { ++g_OwnersLive; }
        Owner(const Owner& other) noexcept : value(other.value) { ++g_OwnersLive; }
        Owner(Owner&& other) noexcept : value(other.value) { ++g_OwnersLive; }
        ~Owner() { --g_OwnersLive; }
        Owner& operator=(const Owner&) = delete;
        Owner& operator=(Owner&&) = delete;
    };
    DONUT_ECS_COMPONENT(Owner, "b8686637-fd28-4346-b204-132c294268ca", DONUT_ECS_FIELDS(value));

    // Resources: four bytes of tail padding.
    struct Gravity { double g; uint32_t steps; };
    DONUT_ECS_COMPONENT_EX(Gravity, "09bbdd47-581f-42dc-bf05-b5b10111ee0d", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(g, steps));

    struct Transient { uint64_t value; };
    DONUT_ECS_COMPONENT_EX(Transient, "ea575e37-b74a-4ba3-b68c-7930175adcec",
                           ComponentFlags::Resource | ComponentFlags::NoSerialize, 0, DONUT_ECS_FIELDS(value));

    // Registered by some worlds only.
    struct Extra { uint32_t value; };
    DONUT_ECS_COMPONENT_EX(Extra, "f6632366-c278-44dd-a08f-a51268361a53", ComponentFlags::Resource, 0,
                           DONUT_ECS_FIELDS(value));

    // Two of them do not fit one 1 KiB chunk.
    struct Huge { uint8_t bytes[900]; };
    DONUT_ECS_COMPONENT(Huge, "35a2ddb1-fc51-4a18-856a-347f0572098c", DONUT_ECS_FIELDS(bytes));
    struct Huge2 { uint8_t bytes[900]; };
    DONUT_ECS_COMPONENT(Huge2, "132d6142-6983-410d-ad0e-145f5e8a13c0", DONUT_ECS_FIELDS(bytes));
}

using namespace donut::engine;
using namespace snapshot_test;
namespace ecs = donut::engine::ecs;

namespace
{
    // ------------------------------------------------------------------------------------------- helpers

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

    bool Has(ComponentFlags flags, ComponentFlags flag) { return !!(flags & flag); }

    // An IStateWriter into memory; Write fails with FE_UNSUPPORTED once the bytes would pass `limit`.
    struct MemoryWriter final : IStateWriter
    {
        std::vector<uint8_t> bytes;
        uint64_t limit = ~uint64_t(0);
        uint32_t writes = 0;

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
            ++writes;
            if (bytes.size() + size > limit)
                return nvrhi::FE_UNSUPPORTED;
            const uint8_t* p = static_cast<const uint8_t*>(data);
            bytes.insert(bytes.end(), p, p + size);
            return nvrhi::FS_OK;
        }
    };

    // An IStateReader over memory; `failure` is returned when fewer bytes remain than asked for.
    struct MemoryReader final : IStateReader
    {
        const uint8_t* data;
        uint64_t size;
        uint64_t position = 0;
        nvrhi::FRESULT failure = nvrhi::FE_INVALID_ARGS;

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
                return failure;
            std::memcpy(out, data + position, size_t(count));
            position += count;
            return nvrhi::FS_OK;
        }
    };

    using Bytes = std::vector<uint8_t>;

    Bytes Save(IWorld* world)
    {
        MemoryWriter writer;
        CHECK(world->SaveSnapshot(&writer) == nvrhi::FS_OK);
        return writer.bytes;
    }

    nvrhi::FRESULT Load(IWorld* world, const Bytes& bytes, uint64_t* consumed = nullptr)
    {
        MemoryReader reader(bytes.data(), bytes.size());
        const nvrhi::FRESULT result = world->LoadSnapshot(&reader);
        if (consumed)
            *consumed = reader.position;
        return result;
    }

    // Recomputes the checksum of a snapshot that a test changed on purpose.
    void FixChecksum(Bytes& bytes)
    {
        const uint64_t checksum = ecs::Xxh64(bytes.data(), bytes.size() - ecs::c_SnapshotChecksumBytes);
        std::memcpy(bytes.data() + bytes.size() - ecs::c_SnapshotChecksumBytes, &checksum, sizeof(checksum));
    }

    nvrhi::AutoPtr<IWorld> MakeWorld(bool validation = false, uint32_t chunkSize = 16 * 1024,
                                     uint32_t largeChunkSize = 64 * 1024)
    {
        WorldDesc desc;
        desc.debugName = "test_ecs_snapshot";
        desc.validation = validation;
        desc.chunkSize = chunkSize;
        desc.largeChunkSize = largeChunkSize;
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

    // An entity without components.
    Entity Create(IWorld* world)
    {
        EntityCreateDesc desc{};
        desc.entityCount = 1;
        Entity entity = Entity::Null;
        CHECK(world->CreateEntities(desc, &entity) == nvrhi::FS_OK);
        return entity;
    }

    struct Ids
    {
        ComponentId position, padded, link, frozen, layer, bounds, joint, selected, color, toggle, hidden, opaque, big,
            wide, tracked, cache, note, owner;
        ComponentId gravity, frameTime, transient;
    };

    // Every test type, in a fixed order; resources last.
    std::vector<const ComponentTypeDesc*> AllTypes()
    {
        return {
            &ComponentTraits<Position>::Desc(), &ComponentTraits<Padded>::Desc(), &ComponentTraits<Link>::Desc(),
            &ComponentTraits<Frozen>::Desc(), &ComponentTraits<Layer>::Desc(), &ComponentTraits<Bounds>::Desc(),
            &ComponentTraits<Joint>::Desc(), &ComponentTraits<Selected>::Desc(), &ComponentTraits<Color>::Desc(),
            &ComponentTraits<Toggle>::Desc(), &ComponentTraits<Hidden>::Desc(), &ComponentTraits<Opaque>::Desc(),
            &ComponentTraits<Big>::Desc(), &ComponentTraits<Wide>::Desc(), &ComponentTraits<Tracked>::Desc(),
            &ComponentTraits<Cache>::Desc(), &ComponentTraits<Note>::Desc(), &ComponentTraits<Owner>::Desc(),
            &ComponentTraits<Gravity>::Desc(), &ComponentTraits<FrameTime>::Desc(), &ComponentTraits<Transient>::Desc(),
        };
    }

    // Registers every test type, in the fixed order or reversed (other component ids).
    Ids Register(IWorld* world, bool reverse = false)
    {
        std::vector<const ComponentTypeDesc*> types = AllTypes();
        if (reverse)
            std::reverse(types.begin(), types.end());
        for (const ComponentTypeDesc* desc : types)
        {
            ComponentId id = ComponentId::Invalid;
            CHECK(world->RegisterComponent(*desc, id) == nvrhi::FS_OK);
        }
        Ids ids;
        ids.position = FindComponent<Position>(world);
        ids.padded = FindComponent<Padded>(world);
        ids.link = FindComponent<Link>(world);
        ids.frozen = FindComponent<Frozen>(world);
        ids.layer = FindComponent<Layer>(world);
        ids.bounds = FindComponent<Bounds>(world);
        ids.joint = FindComponent<Joint>(world);
        ids.selected = FindComponent<Selected>(world);
        ids.color = FindComponent<Color>(world);
        ids.toggle = FindComponent<Toggle>(world);
        ids.hidden = FindComponent<Hidden>(world);
        ids.opaque = FindComponent<Opaque>(world);
        ids.big = FindComponent<Big>(world);
        ids.wide = FindComponent<Wide>(world);
        ids.tracked = FindComponent<Tracked>(world);
        ids.cache = FindComponent<Cache>(world);
        ids.note = FindComponent<Note>(world);
        ids.owner = FindComponent<Owner>(world);
        ids.gravity = FindComponent<Gravity>(world);
        ids.frameTime = FindComponent<FrameTime>(world);
        ids.transient = FindComponent<Transient>(world);
        return ids;
    }

    uint32_t ScalarSize(FieldType type)
    {
        switch (type)
        {
        case FieldType::Int16: case FieldType::UInt16: return 2;
        case FieldType::Int32: case FieldType::UInt32: case FieldType::Float: return 4;
        case FieldType::Int64: case FieldType::UInt64: case FieldType::Double: case FieldType::Entity: return 8;
        default: return 1;
        }
    }

    bool IsSerialisableDesc(const ComponentTypeDesc& desc)
    {
        return Has(desc.flags, ComponentFlags::Relocatable) && !Has(desc.flags, ComponentFlags::NoSerialize);
    }

    void AppendHex(std::string& out, const uint8_t* bytes, size_t size)
    {
        static const char digits[] = "0123456789abcdef";
        const size_t start = out.size();
        out.resize(start + 2 * size);
        for (size_t i = 0; i < size; ++i)
        {
            out[start + 2 * i] = digits[bytes[i] >> 4];
            out[start + 2 * i + 1] = digits[bytes[i] & 15];
        }
    }

    // The field bytes of a value, or all of its bytes when it lists no field (snapshots keep those as they are).
    void AppendValue(std::string& out, const ComponentTypeDesc& desc, const uint8_t* value)
    {
        if (desc.fieldCount == 0)
        {
            AppendHex(out, value, desc.size);
            return;
        }
        for (uint32_t f = 0; f < desc.fieldCount; ++f)
        {
            const FieldDesc& field = desc.fields[f];
            AppendHex(out, value + field.offset, size_t(ScalarSize(field.type)) * field.scalarCount);
            out += '.';
        }
    }

    // What the public per-entity API reads of the serialisable components, per entity index, plus generations, states
    // and resources: independent of ComputeHash.
    std::string Dump(IWorld* world, bool chunkValues = true)
    {
        ecs::WorldImpl& impl = Impl(world);
        std::string out;
        char text[64];
        for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
        {
            const ecs::EntityRecord& record = impl.GetEntityRecord(index);
            const char state = record.archetype == ecs::c_FreeSlot ? 'f'
                             : record.archetype >= ecs::c_ClaimedSlot ? 'r' : 'a';
            std::snprintf(text, sizeof(text), "%u:%u%c", index, record.generation, state);
            out += text;
            if (state == 'a')
            {
                const Entity entity = MakeEntity(index, record.generation);
                CHECK(world->IsAlive(entity));
                for (const ComponentTypeDesc* desc : AllTypes())
                {
                    const ComponentId id = world->FindComponent(desc->uuid);
                    if (id == ComponentId::Invalid || !IsSerialisableDesc(*desc) ||
                        Has(desc->flags, ComponentFlags::Resource) || !world->HasComponent(entity, id))
                        continue;
                    out += ' ';
                    out += desc->name;
                    out += '=';
                    const uint8_t* value = static_cast<const uint8_t*>(world->GetComponent(entity, id));
                    CHECK(value != nullptr);
                    if (Has(desc->flags, ComponentFlags::Buffer))
                    {
                        const BufferHeader& header = *reinterpret_cast<const BufferHeader*>(value);
                        std::snprintf(text, sizeof(text), "[%u]", header.count);
                        out += text;
                        for (uint32_t i = 0; i < header.count; ++i)
                            AppendValue(out, *desc, static_cast<const uint8_t*>(header.data) + size_t(i) * desc->size);
                    }
                    else if (desc->size != 0 && (chunkValues || !Has(desc->flags, ComponentFlags::Chunk)))
                    {
                        AppendValue(out, *desc, value);
                    }
                    if (Has(desc->flags, ComponentFlags::Enableable))
                        out += world->IsEnabled(entity, id) ? '+' : '-';
                }
            }
            out += '\n';
        }
        for (const ComponentTypeDesc* desc : AllTypes())
        {
            const ComponentId id = world->FindComponent(desc->uuid);
            if (id == ComponentId::Invalid || !Has(desc->flags, ComponentFlags::Resource) || !IsSerialisableDesc(*desc))
                continue;
            out += desc->name;
            out += '=';
            AppendValue(out, *desc, static_cast<const uint8_t*>(world->GetResource(id)));
            out += '\n';
        }
        return out;
    }

    // Everything a load could change, for "a refused load changes nothing": the serialisable content (Dump, Chunk values
    // included), the non-serialisable components and resources, the versions, every archetype's chunks with their
    // change versions and rows, and the sparse sets' dense order.
    std::string FullState(IWorld* world)
    {
        ecs::WorldImpl& impl = Impl(world);
        std::string out = Dump(world, true);
        auto raw = [&out](const void* data, size_t size) { out.append(static_cast<const char*>(data), size); };
        for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
        {
            const ecs::EntityRecord& record = impl.GetEntityRecord(index);
            if (record.archetype >= ecs::c_ClaimedSlot)
                continue;
            const Entity entity = MakeEntity(index, record.generation);
            for (const ComponentTypeDesc* desc : AllTypes())
            {
                const ComponentId id = world->FindComponent(desc->uuid);
                if (id == ComponentId::Invalid || IsSerialisableDesc(*desc) || Has(desc->flags, ComponentFlags::Resource) ||
                    !world->HasComponent(entity, id))
                    continue;
                out += desc->name;
                AppendValue(out, *desc, static_cast<const uint8_t*>(world->GetComponent(entity, id)));
            }
        }
        for (const ComponentTypeDesc* desc : AllTypes())
        {
            const ComponentId id = world->FindComponent(desc->uuid);
            if (id != ComponentId::Invalid && Has(desc->flags, ComponentFlags::Resource))
                AppendValue(out, *desc, static_cast<const uint8_t*>(world->GetResource(id)));
        }
        const uint64_t versions[2] = { world->GetVersion(), world->GetStructuralVersion() };
        raw(versions, sizeof(versions));
        const uint32_t archetypes = impl.GetArchetypeStorageCount();
        raw(&archetypes, sizeof(archetypes));
        for (uint32_t a = 0; a < archetypes; ++a)
        {
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(a);
            const uint32_t sizes[2] = { archetype.entityCount, archetype.GetChunkCount() };
            raw(sizes, sizeof(sizes));
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                raw(&archetype.chunks[c].count, sizeof(uint32_t));
                raw(archetype.Versions(c), (archetype.columns.size() + 1) * sizeof(uint64_t));
                raw(archetype.Entities(c), size_t(archetype.chunks[c].count) * sizeof(Entity));
            }
        }
        for (uint32_t t = 0; t < world->GetComponentTypeCount(); ++t)
            if (const ecs::SparseSet* set = impl.GetSparseSet(ComponentId(t)))
                raw(set->entities.data(), size_t(set->GetCount()) * sizeof(Entity));
        return out;
    }

    // Whether every alive entity reads the same field bytes for each Chunk component in both worlds (the same ids).
    bool ChunkValuesAgree(IWorld* a, IWorld* b)
    {
        ecs::WorldImpl& impl = Impl(a);
        for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
        {
            const ecs::EntityRecord& record = impl.GetEntityRecord(index);
            if (record.archetype >= ecs::c_ClaimedSlot)
                continue;
            const Entity entity = MakeEntity(index, record.generation);
            for (const ComponentTypeDesc* desc : AllTypes())
            {
                if (!Has(desc->flags, ComponentFlags::Chunk) || !IsSerialisableDesc(*desc))
                    continue;
                const ComponentId idA = a->FindComponent(desc->uuid);
                const ComponentId idB = b->FindComponent(desc->uuid);
                if (!a->HasComponent(entity, idA))
                    continue;
                std::string x;
                std::string y;
                AppendValue(x, *desc, static_cast<const uint8_t*>(a->GetComponent(entity, idA)));
                AppendValue(y, *desc, static_cast<const uint8_t*>(b->GetComponent(entity, idB)));
                if (x != y)
                    return false;
            }
        }
        return true;
    }

    // The non-empty archetypes in index order: their serialisable components (by name), rows per chunk and rows.
    std::string Layout(IWorld* world)
    {
        ecs::WorldImpl& impl = Impl(world);
        std::string out;
        char text[64];
        for (uint32_t a = 0; a < impl.GetArchetypeStorageCount(); ++a)
        {
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(a);
            if (archetype.entityCount == 0)
                continue;
            std::vector<std::string> names;
            for (const ecs::ArchetypeColumn& column : archetype.columns)
                if (IsSerialisableDesc(column.type->desc))
                    names.push_back(column.type->desc.name);
            std::sort(names.begin(), names.end());
            for (const std::string& name : names)
                out += name + ',';
            std::snprintf(text, sizeof(text), " rpc %u:", archetype.rowsPerChunk);
            out += text;
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                {
                    std::snprintf(text, sizeof(text), " %llx", (unsigned long long)archetype.Entities(c)[r]);
                    out += text;
                }
            out += '\n';
        }
        return out;
    }

    // The storage's invariants, through the private header.
    void CheckIntegrity(IWorld* world)
    {
        ecs::WorldImpl& impl = Impl(world);
        uint32_t alive = 0;
        for (uint32_t a = 0; a < impl.GetArchetypeStorageCount(); ++a)
        {
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(a);
            uint32_t rows = 0;
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const uint32_t count = archetype.chunks[c].count;
                CHECK(count != 0 && count <= archetype.rowsPerChunk);
                CHECK(c + 1 == archetype.GetChunkCount() || count == archetype.rowsPerChunk);
                for (uint32_t r = 0; r < count; ++r)
                {
                    const Entity entity = archetype.Entities(c)[r];
                    const ecs::EntityRecord* record = impl.FindAlive(entity);
                    CHECK(record && record->archetype == a && record->chunk == c && record->row == r);
                    for (uint32_t k = 0; k < archetype.columns.size(); ++k)
                    {
                        const ecs::ArchetypeColumn& column = archetype.columns[k];
                        if (column.kind != ecs::StorageKind::Buffer)
                            continue;
                        const BufferHeader& header = static_cast<const BufferHeader*>(archetype.ColumnData(c, k))[r];
                        const void* inlineData = column.inlineStride != 0 ? archetype.InlineData(c, k, r) : nullptr;
                        CHECK(header.count <= header.capacity || header.count == 0);
                        CHECK((header.data == inlineData) == (header.count <= column.type->desc.bufferInlineCapacity));
                    }
                }
                rows += count;
            }
            CHECK(rows == archetype.entityCount);
            alive += rows;
        }
        WorldStats stats;
        CHECK(world->GetStats(stats).entityCount == alive);
        uint32_t records = 0;
        for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
            if (impl.GetEntityRecord(index).archetype < ecs::c_ClaimedSlot)
                ++records;
        CHECK(records == alive);

        // Sparse sets: dense entities alive and indexed, presence bits exactly where the entity has the component.
        std::vector<uint32_t> sparseCounts(impl.GetEntityTableSize(), 0);
        for (uint32_t t = 0; t < world->GetComponentTypeCount(); ++t)
        {
            const ecs::SparseSet* set = impl.GetSparseSet(ComponentId(t));
            if (!set)
                continue;
            for (uint32_t i = 0; i < set->GetCount(); ++i)
            {
                const Entity entity = set->entities[i];
                CHECK(world->IsAlive(entity) && set->Find(GetEntityIndex(entity)) == i);
                ++sparseCounts[GetEntityIndex(entity)];
            }
            for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
            {
                const ecs::EntityRecord& record = impl.GetEntityRecord(index);
                if (record.archetype >= ecs::c_ClaimedSlot)
                    continue;
                const bool member = set->Find(index) != ecs::HashIndex::c_NotFound;
                CHECK(ecs::GetMaskBit(impl.GetSparsePresence(*set, record.archetype, record.chunk), record.row) == member);
            }
        }
        for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
            if (impl.GetEntityRecord(index).archetype < ecs::c_ClaimedSlot)
                CHECK(impl.GetEntityRecord(index).sparseCount == sparseCounts[index]);
    }

    // Moves the heap's state, so that addresses differ between two builds of a world.
    void PerturbHeap(uint64_t seed)
    {
        Xorshift rng{ seed };
        std::vector<void*> blocks;
        std::vector<std::vector<uint8_t>> vectors;
        for (uint32_t i = 0; i < 200; ++i)
        {
            blocks.push_back(ecs::HeapAllocate(16 + rng.Below(5000), 16));
            vectors.emplace_back(1 + rng.Below(3000), uint8_t(i));
        }
        for (uint32_t i = 0; i < blocks.size(); i += 2)
            ecs::HeapFree(blocks[i]);
        for (uint32_t i = 1; i < blocks.size(); i += 2)
            ecs::HeapFree(blocks[i]);
    }

    // ----------------------------------------------------------------------------------- seeded scripts

    enum Feature : uint32_t
    {
        F_Data = 1u << 0,        // Position, Padded, Tracked
        F_Tags = 1u << 1,        // Frozen
        F_Shared = 1u << 2,      // Layer
        F_Chunk = 1u << 3,       // Bounds
        F_Buffers = 1u << 4,     // Joint
        F_Sparse = 1u << 5,      // Selected, Color
        F_Enable = 1u << 6,      // Toggle, Hidden
        F_Large = 1u << 7,       // Big, Wide, Opaque
        F_NonSerial = 1u << 8,   // Cache, Owner, Note
        F_Reserve = 1u << 9,     // reserved ids
        F_Instance = 1u << 10,   // Instantiate
        F_Links = 1u << 11,      // Link
        F_Resources = 1u << 12,  // Gravity, FrameTime, Transient
        F_All = (1u << 13) - 1,
        F_Serial = F_All & ~F_NonSerial,
    };

    struct ScriptSpec
    {
        const char* name;
        uint64_t seed;
        uint32_t features;
        uint32_t steps;
        uint32_t maxBatch;
    };

    // Builds a world through the public API only, deterministically from the seed.
    class Script
    {
    public:
        Script(IWorld* world, const Ids& ids, const ScriptSpec& spec)
            : m_World(world), m_Ids(ids), m_Spec(spec), m_Rng{ spec.seed * 0x9E3779B97F4A7C15ull + 1 }
        {
            auto add = [&](uint32_t feature, std::initializer_list<ComponentId> list, std::vector<ComponentId>& pool)
            {
                if (spec.features & feature)
                    pool.insert(pool.end(), list);
            };
            add(F_Data, { ids.position, ids.padded, ids.tracked }, m_Row);
            add(F_Tags, { ids.frozen }, m_Row);
            add(F_Shared, { ids.layer }, m_Row);
            add(F_Chunk, { ids.bounds }, m_Row);
            add(F_Buffers, { ids.joint }, m_Row);
            add(F_Enable, { ids.toggle, ids.hidden }, m_Row);
            add(F_Large, { ids.big, ids.wide, ids.opaque }, m_Row);
            add(F_NonSerial, { ids.cache, ids.owner }, m_Row);
            add(F_Links, { ids.link }, m_Row);
            add(F_Sparse, { ids.selected, ids.color }, m_Sparse);
            add(F_NonSerial, { ids.note }, m_Sparse);
        }

        void Run()
        {
            for (uint32_t step = 0; step < m_Spec.steps; ++step)
            {
                const uint32_t op = m_Rng.Below(16);
                switch (op)
                {
                case 0: case 1: case 2: Create(); break;
                case 3: case 4: Destroy(); break;
                case 5: AddOne(); break;
                case 6: RemoveOne(); break;
                case 7: WriteOne(); break;
                case 8: ChangeShared(); break;
                case 9: ToggleOne(); break;
                case 10: ResizeOne(); break;
                case 11: SparseOne(); break;
                case 12: ChunkOne(); break;
                case 13: ReserveOne(); break;
                case 14: InstantiateOne(); break;
                default: WriteResources(step); break;
                }
            }
            WriteResources(m_Spec.steps);
        }

        const std::vector<Entity>& Alive() const { return m_Alive; }
        const std::vector<Entity>& Reserved() const { return m_Reserved; }

    private:
        Entity Pick()
        {
            return m_Alive[m_Rng.Below(uint32_t(m_Alive.size()))];
        }
        Entity PickOrNull()
        {
            return m_Alive.empty() || m_Rng.Below(4) == 0 ? Entity::Null : Pick();
        }
        float Real() { return float(int32_t(m_Rng.Below(4000)) - 2000) * 0.25f; }

        // Values for the components that take one at creation or AddComponent (not tags, Buffers, Chunk, Owner).
        bool TakesValue(ComponentId id) const
        {
            return id == m_Ids.position || id == m_Ids.padded || id == m_Ids.tracked || id == m_Ids.layer ||
                   id == m_Ids.toggle || id == m_Ids.big || id == m_Ids.wide || id == m_Ids.opaque ||
                   id == m_Ids.cache || id == m_Ids.link || id == m_Ids.color || id == m_Ids.note;
        }

        // Into `destination`, which may be unaligned: the value is built in aligned storage first.
        void RandomValue(ComponentId id, uint8_t* destination)
        {
            const uint32_t size = m_World->GetComponentTypeDesc(id)->size;
            alignas(64) uint8_t out[256];
            CHECK(size <= sizeof(out));
            std::memset(out, 0xcd, size);   // padding left as garbage on purpose: neither hashed nor saved
            FillValue(id, out, size);
            std::memcpy(destination, out, size);
        }

        void FillValue(ComponentId id, uint8_t* out, uint32_t size)
        {
            if (id == m_Ids.position)
            {
                Position* p = reinterpret_cast<Position*>(out);
                p->x = Real();
                p->y = Real();
                p->z = Real();
            }
            else if (id == m_Ids.padded)
            {
                Padded* p = reinterpret_cast<Padded*>(out);
                p->a = uint8_t(m_Rng.Below(256));
                p->b = uint32_t(m_Rng.Next());
                p->c = uint16_t(m_Rng.Below(65536));
                p->d = double(m_Rng.Below(1u << 20)) / 8;
            }
            else if (id == m_Ids.tracked || id == m_Ids.toggle || id == m_Ids.cache || id == m_Ids.color ||
                     id == m_Ids.note)
            {
                const uint32_t value = uint32_t(m_Rng.Next());
                std::memcpy(out, &value, sizeof(value));
            }
            else if (id == m_Ids.layer)
            {
                Layer* layer = reinterpret_cast<Layer*>(out);
                layer->id = uint8_t(m_Rng.Below(4));
                layer->mask = m_Rng.Below(3);
            }
            else if (id == m_Ids.link)
            {
                Link* link = reinterpret_cast<Link*>(out);
                link->other = PickOrNull();
                link->tag = m_Rng.Below(1u << 30);
            }
            else if (id == m_Ids.wide)
            {
                Wide* wide = reinterpret_cast<Wide*>(out);
                for (float& v : wide->v)
                    v = Real();
            }
            else
            {
                for (uint32_t i = 0; i < size; ++i)   // Big, Opaque
                    out[i] = uint8_t(m_Rng.Next());
            }
        }

        void Create()
        {
            const uint32_t count = 1 + m_Rng.Below(m_Spec.maxBatch);
            std::vector<ComponentId> components;
            for (const ComponentId id : m_Row)
                if (m_Rng.Below(2) != 0)
                    components.push_back(id);
            for (const ComponentId id : m_Sparse)
                if (m_Rng.Below(4) == 0)
                    components.push_back(id);
            std::vector<std::vector<uint8_t>> storage(components.size());
            std::vector<const void*> values(components.size(), nullptr);
            const bool oneValue = m_Rng.Below(2) != 0;   // one Shared partition, or runs of several
            for (size_t i = 0; i < components.size(); ++i)
            {
                const ComponentId id = components[i];
                if (!TakesValue(id) || m_Rng.Below(4) == 0)
                    continue;
                const uint32_t size = m_World->GetComponentTypeDesc(id)->size;
                storage[i].resize(size_t(size) * count);
                for (uint32_t k = 0; k < count; ++k)
                {
                    if (k != 0 && oneValue && id == m_Ids.layer)
                        std::memcpy(storage[i].data() + size_t(k) * size, storage[i].data(), size);
                    else
                        RandomValue(id, storage[i].data() + size_t(k) * size);
                }
                values[i] = storage[i].data();
            }
            EntityCreateDesc desc{};
            desc.components = components.data();
            desc.componentCount = uint32_t(components.size());
            desc.values = values.data();
            desc.entityCount = count;
            desc.transfer = ValueTransfer::Copy;
            std::vector<Entity> out(count);
            if ((m_Spec.features & F_Reserve) && m_Reserved.size() >= count && m_Rng.Below(2) == 0)
            {
                desc.reservedEntities = m_Reserved.data() + (m_Reserved.size() - count);
                CHECK(m_World->CreateEntities(desc, out.data()) == nvrhi::FS_OK);
                m_Reserved.resize(m_Reserved.size() - count);
            }
            else
            {
                CHECK(m_World->CreateEntities(desc, out.data()) == nvrhi::FS_OK);
            }
            m_Alive.insert(m_Alive.end(), out.begin(), out.end());
        }

        void Destroy()
        {
            if (m_Alive.empty())
                return;
            const uint32_t i = m_Rng.Below(uint32_t(m_Alive.size()));
            const Entity entity = m_Alive[i];
            CHECK(m_World->DestroyEntities(&entity, 1) == nvrhi::FS_OK);
            m_Alive[i] = m_Alive.back();
            m_Alive.pop_back();
        }

        ComponentId PickComponent()
        {
            const uint32_t total = uint32_t(m_Row.size() + m_Sparse.size());
            if (total == 0)
                return ComponentId::Invalid;
            const uint32_t k = m_Rng.Below(total);
            return k < m_Row.size() ? m_Row[k] : m_Sparse[k - m_Row.size()];
        }

        void AddOne()
        {
            if (m_Alive.empty())
                return;
            const Entity entity = Pick();
            const ComponentId id = PickComponent();
            if (id == ComponentId::Invalid)
                return;
            uint8_t value[256];
            const bool withValue = TakesValue(id) && m_Rng.Below(2) != 0;
            if (withValue)
                RandomValue(id, value);
            CHECK(m_World->AddComponent(entity, id, withValue ? value : nullptr) == nvrhi::FS_OK);
        }

        void RemoveOne()
        {
            if (m_Alive.empty())
                return;
            const Entity entity = Pick();
            const ComponentId id = PickComponent();
            if (id != ComponentId::Invalid)
                CHECK(m_World->RemoveComponent(entity, id) == nvrhi::FS_OK);
        }

        void WriteOne()
        {
            if (m_Alive.empty() || m_Row.empty())
                return;
            const Entity entity = Pick();
            const ComponentId id = m_Row[m_Rng.Below(uint32_t(m_Row.size()))];
            if (!m_World->HasComponent(entity, id))
                return;
            if (id == m_Ids.owner)
            {
                static_cast<Owner*>(m_World->GetComponentForWrite(entity, id, c_WorldVersion))->value = m_Rng.Below(100);
                return;
            }
            if (!TakesValue(id) || id == m_Ids.layer)
                return;
            uint8_t value[256];
            RandomValue(id, value);
            void* target = m_World->GetComponentForWrite(entity, id, c_WorldVersion);
            CHECK(target != nullptr);
            std::memcpy(target, value, m_World->GetComponentTypeDesc(id)->size);
        }

        void ChangeShared()
        {
            if (m_Alive.empty() || !(m_Spec.features & F_Shared))
                return;
            Layer layer;
            RandomValue(m_Ids.layer, reinterpret_cast<uint8_t*>(&layer));
            CHECK(m_World->AddComponent(Pick(), m_Ids.layer, &layer) == nvrhi::FS_OK);
        }

        void ToggleOne()
        {
            if (m_Alive.empty() || !(m_Spec.features & F_Enable))
                return;
            const Entity entity = Pick();
            const ComponentId id = m_Rng.Below(2) != 0 ? m_Ids.toggle : m_Ids.hidden;
            if (m_World->HasComponent(entity, id))
                CHECK(m_World->SetEnabled(entity, id, m_Rng.Below(2) != 0) == nvrhi::FS_OK);
        }

        void ResizeOne()
        {
            if (m_Alive.empty() || !(m_Spec.features & F_Buffers))
                return;
            const Entity entity = Pick();
            if (!m_World->HasComponent(entity, m_Ids.joint))
                return;
            const uint32_t count = m_Rng.Below(11);   // inline up to 4, spilled above
            CHECK(m_World->ResizeBuffer(entity, m_Ids.joint, count, c_WorldVersion) == nvrhi::FS_OK);
            BufferHeader* header =
                static_cast<BufferHeader*>(m_World->GetComponentForWrite(entity, m_Ids.joint, c_WorldVersion));
            CHECK(header && header->count == count);
            Joint* joints = GetBufferData<Joint>(*header);
            for (uint32_t i = 0; i < count; ++i)
            {
                joints[i].target = PickOrNull();
                joints[i].weight = Real();
                joints[i].pad = uint32_t(m_Rng.Next());   // not a field
            }
        }

        void SparseOne()
        {
            if (m_Alive.empty() || m_Sparse.empty())
                return;
            const Entity entity = Pick();
            const ComponentId id = m_Sparse[m_Rng.Below(uint32_t(m_Sparse.size()))];
            if (m_Rng.Below(3) == 0)
            {
                CHECK(m_World->RemoveComponent(entity, id) == nvrhi::FS_OK);
                return;
            }
            uint8_t value[16];
            const bool withValue = TakesValue(id);
            if (withValue)
                RandomValue(id, value);
            CHECK(m_World->AddComponent(entity, id, withValue ? value : nullptr) == nvrhi::FS_OK);
        }

        void ChunkOne()
        {
            if (m_Alive.empty() || !(m_Spec.features & F_Chunk))
                return;
            const Entity entity = Pick();
            if (!m_World->HasComponent(entity, m_Ids.bounds))
                return;
            Bounds* bounds = static_cast<Bounds*>(m_World->GetComponentForWrite(entity, m_Ids.bounds, c_WorldVersion));
            bounds->lo = Real();
            bounds->hi = Real();
        }

        void ReserveOne()
        {
            if (!(m_Spec.features & F_Reserve))
                return;
            if (m_Rng.Below(3) != 0 || m_Reserved.empty())
            {
                const uint32_t count = 1 + m_Rng.Below(3);
                const size_t first = m_Reserved.size();
                m_Reserved.resize(first + count);
                CHECK(m_World->ReserveEntities(count, m_Reserved.data() + first) == nvrhi::FS_OK);
            }
            else
            {
                CHECK(m_World->DestroyEntities(&m_Reserved.back(), 1) == nvrhi::FS_OK);   // given back
                m_Reserved.pop_back();
            }
        }

        void InstantiateOne()
        {
            if (m_Alive.empty() || !(m_Spec.features & F_Instance))
                return;
            const Entity prefab = Pick();
            const uint32_t count = 1 + m_Rng.Below(3);
            std::vector<Entity> out(count);
            CHECK(m_World->Instantiate(prefab, count, out.data()) == nvrhi::FS_OK);
            m_Alive.insert(m_Alive.end(), out.begin(), out.end());
        }

        void WriteResources(uint32_t step)
        {
            if (!(m_Spec.features & F_Resources))
                return;
            Gravity* gravity = GetResource<Gravity>(m_World);
            std::memset(gravity, 0xcd, sizeof(*gravity));
            gravity->g = double(m_Rng.Below(1000)) / 16;
            gravity->steps = step;
            FrameTime* time = GetResource<FrameTime>(m_World);
            time->time = double(step) / 64;
            time->deltaTime = 1.0 / 64;
            time->frameIndex = step;
            GetResource<Transient>(m_World)->value = m_Rng.Next();
        }

        IWorld* m_World;
        Ids m_Ids;
        ScriptSpec m_Spec;
        Xorshift m_Rng;
        std::vector<ComponentId> m_Row;
        std::vector<ComponentId> m_Sparse;
        std::vector<Entity> m_Alive;
        std::vector<Entity> m_Reserved;
    };

    nvrhi::AutoPtr<IWorld> BuildScripted(const ScriptSpec& spec, Ids* outIds = nullptr)
    {
        auto world = MakeWorld();
        const Ids ids = Register(world);
        Script script(world, ids, spec);
        script.Run();
        if (outIds)
            *outIds = ids;
        return world;
    }

    // A small world with every kind, for the byte-by-byte failure tests.
    nvrhi::AutoPtr<IWorld> BuildSmall()
    {
        auto world = MakeWorld();
        const Ids ids = Register(world);
        // Two entities: two values each.
        const Position position[2] = { { 1.5f, -2.0f, 3.25f }, { 0.5f, 8.0f, -1.0f } };
        const Padded padded[2] = { { 9, 0x01020304u, 0x0506, 7.5 }, { 1, 2, 3, 4.0 } };
        const Layer layer[2] = { { 2, 5 }, { 2, 5 } };
        const Color color[2] = { { 0xff00ff00u }, { 0x12345678u } };
        const Toggle toggle[2] = { { -4 }, { 12 } };
        Entity e[8];
        {
            const ComponentId components[] = { ids.position, ids.padded, ids.toggle, ids.layer, ids.bounds, ids.joint,
                                               ids.color };
            const void* values[] = { position, padded, toggle, layer, nullptr, nullptr, color };
            EntityCreateDesc desc{};
            desc.components = components;
            desc.values = values;
            desc.componentCount = 7;
            desc.entityCount = 2;
            CHECK(world->CreateEntities(desc, e) == nvrhi::FS_OK);
        }
        CHECK(world->SetEnabled(e[1], ids.toggle, false) == nvrhi::FS_OK);
        CHECK(world->ResizeBuffer(e[0], ids.joint, 2, c_WorldVersion) == nvrhi::FS_OK);
        CHECK(world->ResizeBuffer(e[1], ids.joint, 6, c_WorldVersion) == nvrhi::FS_OK);
        GetComponentForWrite<Bounds>(world, e[0])->hi = 4.0f;
        {
            const ComponentId components[] = { ids.position, ids.frozen, ids.hidden };
            EntityCreateDesc desc{};
            desc.components = components;
            desc.componentCount = 3;
            desc.entityCount = 1;
            CHECK(world->CreateEntities(desc, &e[2]) == nvrhi::FS_OK);
        }
        CHECK(world->SetEnabled(e[2], ids.hidden, false) == nvrhi::FS_OK);
        const Link link{ e[0], 77 };
        Opaque opaque;
        for (uint32_t i = 0; i < sizeof(opaque.raw); ++i)
            opaque.raw[i] = uint8_t(i * 17 + 3);
        {
            const ComponentId components[] = { ids.link, ids.selected, ids.opaque };
            const void* values[] = { &link, nullptr, &opaque };
            EntityCreateDesc desc{};
            desc.components = components;
            desc.values = values;
            desc.componentCount = 3;
            desc.entityCount = 1;
            CHECK(world->CreateEntities(desc, &e[3]) == nvrhi::FS_OK);
        }
        {
            const ComponentId components[] = { ids.big, ids.wide };
            EntityCreateDesc desc{};
            desc.components = components;
            desc.componentCount = 2;
            desc.entityCount = 1;
            CHECK(world->CreateEntities(desc, &e[4]) == nvrhi::FS_OK);
        }
        GetComponentForWrite<Big>(world, e[4])->bytes[100] = 42;
        EntityCreateDesc empty{};
        empty.entityCount = 2;
        CHECK(world->CreateEntities(empty, &e[5]) == nvrhi::FS_OK);
        CHECK(world->AddComponent(e[6], ids.tracked, nullptr) == nvrhi::FS_OK);
        CHECK(world->DestroyEntities(&e[5], 1) == nvrhi::FS_OK);
        CHECK(world->ReserveEntities(1, &e[7]) == nvrhi::FS_OK);
        *GetResource<Gravity>(world) = Gravity{ 9.75, 3 };
        *GetResource<FrameTime>(world) = FrameTime{ 0.5, 0.25, 2 };
        return world;
    }

    // ------------------------------------------------------------------------------------------- tests

    void test_xxh64()
    {
        // Reference values from xxHash 0.8.4 (xxhash.h, XXH64).
        const struct { const char* text; uint64_t seed0; uint64_t seedGolden; } strings[] = {
            { "", 0xef46db3751d8e999ull, 0xc4349fc93c010000ull },
            { "a", 0xd24ec4f1a98c6e5bull, 0x9a7c6d2ea45568c9ull },
            { "abc", 0x44bc2cf5ad770999ull, 0x2ed0f59d6b43ac8bull },
            { "The quick brown fox jumps over the lazy dog", 0x0b242d361fda71bcull, 0x7cfac66832f66b74ull },
            { "0123456789abcdef0123456789abcdef", 0x642a94958e71e6c5ull, 0x91c9c9c721ea6985ull },
            { "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef!", 0x2020b26dbc09cee8ull,
              0x06a25d8aaa9c3cc0ull },
        };
        for (const auto& s : strings)
        {
            CHECK(ecs::Xxh64(s.text, std::strlen(s.text)) == s.seed0);
            CHECK(ecs::Xxh64(s.text, std::strlen(s.text), 0x9E3779B97F4A7C15ull) == s.seedGolden);
        }
        uint8_t pattern[1000];
        for (uint32_t i = 0; i < 1000; ++i)
            pattern[i] = uint8_t(i * 31 + 7);
        const struct { uint32_t length; uint64_t hash; } patterns[] = {
            { 1, 0xa96c7f0ce858bbb7ull }, { 3, 0x56e6957632a487f9ull }, { 4, 0xc60d15b1e3ff8f04ull },
            { 7, 0xafbefc3d6c6f9a8eull }, { 8, 0x3da5c7aa269683e0ull }, { 15, 0xae2a37eb9357caa7ull },
            { 16, 0xa19ad429b02bc413ull }, { 31, 0x4a74f3a1a39ad4a1ull }, { 32, 0x8d57d6a4671cc43dull },
            { 33, 0x62c9fd21ed857664ull }, { 63, 0x5c320a0d2707057full }, { 64, 0x7bbabbc45729d17eull },
            { 65, 0xf3980c34bae65dc1ull }, { 100, 0xefa0ad2d3e70c151ull }, { 1000, 0x99594f4828043d35ull },
        };
        for (const auto& p : patterns)
        {
            CHECK(ecs::Xxh64(pattern, p.length) == p.hash);
            // Any split of the stream gives the same digest.
            for (uint32_t step = 1; step <= 37; step += 6)
            {
                ecs::Hasher64 hasher;
                for (uint32_t at = 0; at < p.length; at += step)
                    hasher.Update(pattern + at, std::min(step, p.length - at));
                CHECK(hasher.Digest() == p.hash);
            }
        }
        ecs::Hasher64 integers;
        integers.U32(0x04030201u);
        integers.U64(0x0c0b0a0908070605ull);
        integers.U8(0x0d);
        const uint8_t bytes[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
        CHECK(integers.Digest() == ecs::Xxh64(bytes, sizeof(bytes)));
    }

    const ScriptSpec c_RoundTrip = { "round_trip", 11, F_Serial, 900, 6 };

    void test_round_trip()
    {
        const int64_t blocks = ecs::GetHeapLiveBlockCount();
        {
            Ids ids;
            auto source = BuildScripted(c_RoundTrip, &ids);
            // A long free list, out of index order: a scattered quarter of the entities destroyed.
            std::vector<Entity> alive;
            for (uint32_t index = 1; index < Impl(source).GetEntityTableSize(); ++index)
            {
                const ecs::EntityRecord& record = Impl(source).GetEntityRecord(index);
                if (record.archetype < ecs::c_ClaimedSlot)
                    alive.push_back(MakeEntity(index, record.generation));
            }
            for (uint32_t i = 0; i < alive.size() / 4; ++i)
            {
                const Entity victim = alive[(size_t(i) * 37) % alive.size()];
                if (source->IsAlive(victim))
                    CHECK(source->DestroyEntities(&victim, 1) == nvrhi::FS_OK);
            }
            CheckIntegrity(source);
            const uint64_t hash = source->ComputeHash();
            const std::string dump = Dump(source);
            const std::string layout = Layout(source);
            const Bytes bytes = Save(source);
            CHECK(source->ComputeHash() == hash);   // saving changes nothing
            WorldStats stats;
            printf("test_ecs_snapshot: round trip: %u entities, %u archetypes, %zu bytes, hash %016llx\n",
                   source->GetStats(stats).entityCount, source->GetArchetypeCount(), bytes.size(),
                   (unsigned long long)hash);

            // Into a fresh world: the same hash, values, generations, states, archetype order and row order; the same
            // snapshot again.
            auto loaded = MakeWorld(true);
            Register(loaded);
            uint64_t consumed = 0;
            CHECK(Load(loaded, bytes, &consumed) == nvrhi::FS_OK && consumed == bytes.size());
            CheckIntegrity(loaded);
            // The observer seam (E1.4): one report per chunk run of loaded rows; nothing was removed.
            auto chunkCount = [](IWorld* world)
            {
                uint64_t chunks = 0;
                for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
                {
                    ArchetypeInfo info{};
                    CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
                    chunks += info.chunkCount;
                }
                return chunks;
            };
            CHECK(Impl(loaded).GetCounters().rowsAddedReports == chunkCount(loaded));
            CHECK(Impl(loaded).GetCounters().rowsRemovingReports == 0);
            CHECK(loaded->ComputeHash() == hash);
            CHECK(Dump(loaded) == dump);
            CHECK(Layout(loaded) == layout);
            CHECK(Save(loaded) == bytes);
            CHECK(Impl(loaded).GetValidationErrorCount() == 0);

            // Ids created afterwards, through the free list and past the end of the table, are the same.
            const uint32_t count = Impl(source).GetEntityTableSize() + 8;
            std::vector<Entity> fromSource(count);
            std::vector<Entity> fromLoaded(count);
            EntityCreateDesc desc{};
            desc.components = &ids.position;
            desc.componentCount = 1;
            desc.entityCount = count;
            CHECK(source->CreateEntities(desc, fromSource.data()) == nvrhi::FS_OK);
            CHECK(loaded->CreateEntities(desc, fromLoaded.data()) == nvrhi::FS_OK);
            CHECK(fromSource == fromLoaded);
            CHECK(source->ComputeHash() == loaded->ComputeHash());

            // Back into the source: its state at the save. Every chunk of the old content is reported as removed.
            const uint64_t removing = Impl(source).GetCounters().rowsRemovingReports;
            const uint64_t oldChunks = chunkCount(source);
            CHECK(Load(source, bytes) == nvrhi::FS_OK);
            CHECK(Impl(source).GetCounters().rowsRemovingReports - removing == oldChunks);
            CHECK(source->ComputeHash() == hash && Dump(source) == dump);
            CheckIntegrity(source);
        }
        {
            // Another registration order (other component ids): matched by GUID; the same hash and values.
            auto source = BuildScripted(c_RoundTrip);
            const Bytes bytes = Save(source);
            auto reversed = MakeWorld(true);
            Register(reversed, true);
            CHECK(Load(reversed, bytes) == nvrhi::FS_OK);
            CheckIntegrity(reversed);
            // Column order follows component ids, so rows per chunk, and with them an entity's Chunk values, may
            // differ; the hash differs exactly then.
            CHECK(Dump(reversed, false) == Dump(source, false));
            CHECK((reversed->ComputeHash() == source->ComputeHash()) == ChunkValuesAgree(reversed, source));

            // Into a world with content of its own, non-serialisable components included: replaced entirely.
            const int64_t owners = g_OwnersLive;
            auto busy = BuildScripted(ScriptSpec{ "busy", 5, F_All, 400, 4 });
            CHECK(g_OwnersLive > owners);
            CHECK(Load(busy, bytes) == nvrhi::FS_OK);
            CHECK(g_OwnersLive == owners);   // every Owner of the old content was destroyed
            CheckIntegrity(busy);
            CHECK(busy->ComputeHash() == source->ComputeHash());
            CHECK(Dump(busy) == Dump(source));
            WorldStats stats;
            CHECK(busy->GetStats(stats).entityCount == source->GetStats(stats).entityCount);

            // The stream may go on after the snapshot: the load reads exactly the snapshot, twice in a row.
            Bytes stream = bytes;
            auto secondWorld = BuildScripted(ScriptSpec{ "second", 8, F_Serial, 200, 3 });
            const Bytes second = Save(secondWorld);
            stream.insert(stream.end(), second.begin(), second.end());
            stream.push_back(0x7f);
            MemoryReader reader(stream.data(), stream.size());
            auto twice = MakeWorld();
            Register(twice);
            CHECK(twice->LoadSnapshot(&reader) == nvrhi::FS_OK && reader.position == bytes.size());
            CHECK(twice->ComputeHash() == source->ComputeHash());
            CHECK(twice->LoadSnapshot(&reader) == nvrhi::FS_OK && reader.position == stream.size() - 1);
            CHECK(twice->ComputeHash() == secondWorld->ComputeHash());
            CHECK(Dump(twice, false) == Dump(secondWorld, false));
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }

    void test_coverage()
    {
        const int64_t blocks = ecs::GetHeapLiveBlockCount();
        const int64_t owners = g_OwnersLive;
        {
            auto source = MakeWorld();
            const Ids ids = Register(source);
            // Rows of several chunks: Position, Bounds (Chunk), Layer (Shared), Toggle (Enableable), Joint (Buffer).
            std::vector<Entity> rows(1500);
            const std::vector<Layer> layers(rows.size(), Layer{ 3, 9 });   // a value per entity
            const ComponentId components[] = { ids.position, ids.bounds, ids.layer, ids.toggle, ids.joint };
            const void* values[] = { nullptr, nullptr, layers.data(), nullptr, nullptr };
            EntityCreateDesc desc{};
            desc.components = components;
            desc.values = values;
            desc.componentCount = 5;
            desc.entityCount = uint32_t(rows.size());
            CHECK(source->CreateEntities(desc, rows.data()) == nvrhi::FS_OK);
            const ecs::Archetype& archetype =
                Impl(source).GetArchetypeStorage(Impl(source).FindAlive(rows[0])->archetype);
            CHECK(archetype.GetChunkCount() >= 3);
            for (uint32_t i = 0; i < rows.size(); ++i)
            {
                GetComponentForWrite<Position>(source, rows[i])->x = float(i);
                CHECK(source->SetEnabled(rows[i], ids.toggle, i % 3 != 0) == nvrhi::FS_OK);
                const uint32_t joints = i % 9;   // up to 4 inline, then spilled
                CHECK(source->ResizeBuffer(rows[i], ids.joint, joints, c_WorldVersion) == nvrhi::FS_OK);
            }
            for (uint32_t i = 0; i < rows.size(); ++i)
            {
                BufferHeader* header =
                    static_cast<BufferHeader*>(source->GetComponentForWrite(rows[i], ids.joint, c_WorldVersion));
                for (uint32_t j = 0; j < header->count; ++j)
                    GetBufferData<Joint>(*header)[j] = Joint{ rows[(i + j) % rows.size()], float(j), 0xabcdu };
            }
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                GetComponentForWrite<Bounds>(source, archetype.Entities(c)[0])->lo = float(100 + c);
            // Sparse components, in a dense order that is not the creation order.
            for (uint32_t i = 0; i < 300; ++i)
            {
                const Color color{ 0x1000u + i };
                CHECK(source->AddComponent(rows[(i * 7) % rows.size()], ids.color, &color) == nvrhi::FS_OK);
                CHECK(source->AddComponent(rows[(i * 11) % rows.size()], ids.selected, nullptr) == nvrhi::FS_OK);
            }
            for (uint32_t i = 0; i < 300; i += 5)
                CHECK(source->RemoveComponent(rows[(i * 7) % rows.size()], ids.color) == nvrhi::FS_OK);
            // Non-serialisable components: an entity that has only them, others that have them besides (Note is
            // Sparse, so rows[6] stays where it is).
            const Owner owner;
            const Cache cache{ 5 };
            const Note note{ 6 };
            Entity onlyOwner = Entity::Null;
            {
                const ComponentId list[] = { ids.owner, ids.note };
                const void* listValues[] = { &owner, &note };
                EntityCreateDesc d{};
                d.components = list;
                d.values = listValues;
                d.componentCount = 2;
                d.entityCount = 1;
                CHECK(source->CreateEntities(d, &onlyOwner) == nvrhi::FS_OK);
            }
            Entity cached = Entity::Null;
            {
                const ComponentId list[] = { ids.position, ids.cache };
                const void* listValues[] = { nullptr, &cache };
                EntityCreateDesc d{};
                d.components = list;
                d.values = listValues;
                d.componentCount = 2;
                d.entityCount = 1;
                CHECK(source->CreateEntities(d, &cached) == nvrhi::FS_OK);
            }
            CHECK(source->AddComponent(rows[6], ids.note, &note) == nvrhi::FS_OK);
            // Opaque, Tracked (construct hook), a disabled tag, Big and Wide.
            Opaque opaque;
            for (uint32_t i = 0; i < sizeof(opaque.raw); ++i)
                opaque.raw[i] = uint8_t(0xf0 - i);
            Entity misc = Entity::Null;
            {
                const ComponentId list[] = { ids.opaque, ids.tracked, ids.hidden, ids.big, ids.wide };
                const void* listValues[] = { &opaque, nullptr, nullptr, nullptr, nullptr };
                EntityCreateDesc d{};
                d.components = list;
                d.values = listValues;
                d.componentCount = 5;
                d.entityCount = 1;
                CHECK(source->CreateEntities(d, &misc) == nvrhi::FS_OK);
            }
            CHECK(source->SetEnabled(misc, ids.hidden, false) == nvrhi::FS_OK);
            GetComponentForWrite<Wide>(source, misc)->v[15] = 2.5f;
            *GetResource<Gravity>(source) = Gravity{ 1.25, 8 };
            *GetResource<FrameTime>(source) = FrameTime{ 3.0, 0.5, 6 };
            GetResource<Transient>(source)->value = 111;
            const Bytes bytes = Save(source);

            auto target = MakeWorld(true);
            const Ids local = Register(target, true);
            uint32_t extraValue = 0x5eed;
            ComponentId extra = ComponentId::Invalid;
            CHECK(target->RegisterResource(ComponentTraits<Extra>::Desc(), &extraValue, extra) == nvrhi::FS_OK);
            GetResource<Transient>(target)->value = 222;
            CHECK(Load(target, bytes) == nvrhi::FS_OK);
            CheckIntegrity(target);
            CHECK(target->ComputeHash() != source->ComputeHash());   // Extra is a serialisable resource of the target
            CHECK(Dump(target) == Dump(source));

            // Data, Chunk per chunk (the same partition), Shared, Enableable, Buffers inline and spilled.
            const ecs::Archetype& loadedArchetype =
                Impl(target).GetArchetypeStorage(Impl(target).FindAlive(rows[0])->archetype);
            CHECK(loadedArchetype.GetChunkCount() == archetype.GetChunkCount());
            uint64_t spilledBytes = 0;
            for (uint32_t i = 0; i < rows.size(); ++i)
            {
                CHECK(GetComponent<Position>(target, rows[i])->x == float(i));
                CHECK(target->IsEnabled(rows[i], local.toggle) == (i % 3 != 0));
                CHECK(GetComponent<Layer>(target, rows[i])->id == 3 && GetComponent<Layer>(target, rows[i])->mask == 9);
                const BufferHeader& header = *static_cast<const BufferHeader*>(target->GetComponent(rows[i], local.joint));
                CHECK(header.count == i % 9);
                const ecs::EntityRecord& record = *Impl(target).FindAlive(rows[i]);
                const uint32_t column = loadedArchetype.FindColumn(local.joint);
                const bool isInline = header.data == loadedArchetype.InlineData(record.chunk, column, record.row);
                CHECK(isInline == (header.count <= 4));
                if (!isInline)
                {
                    CHECK(header.capacity == header.count);
                    spilledBytes += uint64_t(header.count) * sizeof(Joint);
                }
                for (uint32_t j = 0; j < header.count; ++j)
                {
                    const Joint& joint = GetBufferData<Joint>(header)[j];
                    CHECK(joint.target == rows[(i + j) % rows.size()] && joint.weight == float(j));
                    CHECK(joint.pad == 0);   // not a field: saved as 0
                }
            }
            WorldStats stats;
            CHECK(target->GetStats(stats).bufferHeapBytes == spilledBytes);
            for (uint32_t c = 0; c < loadedArchetype.GetChunkCount(); ++c)
            {
                CHECK(GetComponent<Bounds>(target, loadedArchetype.Entities(c)[0])->lo == float(100 + c));
                CHECK(loadedArchetype.chunks[c].count == archetype.chunks[c].count);
            }
            // Sparse: values and dense order.
            const ecs::SparseSet* sourceColors = Impl(source).GetSparseSet(ids.color);
            const ecs::SparseSet* loadedColors = Impl(target).GetSparseSet(local.color);
            CHECK(sourceColors->GetCount() == loadedColors->GetCount() && loadedColors->GetCount() == 240);
            for (uint32_t i = 0; i < loadedColors->GetCount(); ++i)
            {
                CHECK(loadedColors->entities[i] == sourceColors->entities[i]);
                CHECK(std::memcmp(loadedColors->Value(i), sourceColors->Value(i), sizeof(Color)) == 0);
            }
            const ecs::SparseSet* sourceSelected = Impl(source).GetSparseSet(ids.selected);
            const ecs::SparseSet* loadedSelected = Impl(target).GetSparseSet(local.selected);
            CHECK(sourceSelected->GetCount() == 300 && loadedSelected->GetCount() == 300);
            for (uint32_t i = 0; i < 300; ++i)
                CHECK(loadedSelected->entities[i] == sourceSelected->entities[i]);
            // Not serialisable: gone; an entity that had only such components is alive without any.
            CHECK(target->IsAlive(onlyOwner) && Impl(target).FindAlive(onlyOwner)->archetype == 0);
            CHECK(!target->HasComponent(onlyOwner, local.owner) && !target->HasComponent(onlyOwner, local.note));
            CHECK(!target->HasComponent(cached, local.cache) && target->HasComponent(cached, local.position));
            CHECK(!target->HasComponent(rows[6], local.note) && target->HasComponent(rows[6], local.position));
            CHECK(Impl(target).GetSparseSet(local.note)->GetCount() == 0);
            // Opaque bytes as they are; Tracked; the disabled tag; Big and Wide.
            CHECK(std::memcmp(GetComponent<Opaque>(target, misc)->raw, opaque.raw, sizeof(opaque.raw)) == 0);
            CHECK(GetComponent<Tracked>(target, misc)->value == 7);
            CHECK(target->HasComponent(misc, local.hidden) && !target->IsEnabled(misc, local.hidden));
            CHECK(GetComponent<Wide>(target, misc)->v[15] == 2.5f);
            CHECK((reinterpret_cast<uintptr_t>(GetComponent<Wide>(target, misc)) & 63) == 0);
            // Resources: the snapshot's; Transient (NoSerialize) and Extra (not in the snapshot) keep theirs.
            CHECK(GetResource<Gravity>(target)->g == 1.25 && GetResource<Gravity>(target)->steps == 8);
            CHECK(GetResource<FrameTime>(target)->frameIndex == 6 && GetResource<FrameTime>(target)->deltaTime == 0.5);
            CHECK(GetResource<Transient>(target)->value == 222);
            CHECK(*static_cast<uint32_t*>(target->GetResource(extra)) == 0x5eed);
            // Versions do not go back: the rows are stamped with the target's version.
            const uint64_t version = target->GetVersion();
            CHECK(loadedArchetype.Versions(0)[loadedArchetype.FindColumn(local.position)] == version);
            CHECK(Impl(target).GetValidationErrorCount() == 0);

            // Reserved ids stay reserved: CreateEntities takes them in the loaded world.
            Entity reserved = Entity::Null;
            CHECK(source->ReserveEntities(1, &reserved) == nvrhi::FS_OK);
            auto again = MakeWorld();
            Register(again);
            CHECK(Load(again, Save(source)) == nvrhi::FS_OK);
            CHECK(!again->IsAlive(reserved));
            EntityCreateDesc fromReserved{};
            fromReserved.reservedEntities = &reserved;
            fromReserved.entityCount = 1;
            CHECK(again->CreateEntities(fromReserved, nullptr) == nvrhi::FS_OK && again->IsAlive(reserved));
            CHECK(source->CreateEntities(fromReserved, nullptr) == nvrhi::FS_OK);
            CHECK(again->ComputeHash() == source->ComputeHash());
        }
        {
            // Records that differ only in non-serialisable components share one archetype on load: their rows in
            // record order, and each chunk takes the Chunk value of the saved chunk that held its first row.
            auto source = MakeWorld();
            const Ids ids = Register(source);
            std::vector<Entity> a(10);
            std::vector<Entity> b(10);
            const ComponentId withCache[] = { ids.position, ids.bounds, ids.cache };
            const ComponentId without[] = { ids.position, ids.bounds };
            EntityCreateDesc desc{};
            desc.components = withCache;
            desc.componentCount = 3;
            desc.entityCount = 10;
            CHECK(source->CreateEntities(desc, a.data()) == nvrhi::FS_OK);
            desc.components = without;
            desc.componentCount = 2;
            CHECK(source->CreateEntities(desc, b.data()) == nvrhi::FS_OK);
            GetComponentForWrite<Bounds>(source, a[0])->lo = 1.0f;
            GetComponentForWrite<Bounds>(source, b[0])->lo = 2.0f;
            auto target = MakeWorld();
            Register(target);
            CHECK(Load(target, Save(source)) == nvrhi::FS_OK);
            CheckIntegrity(target);
            // b's rows now read a's chunk value: another per-entity read, so another hash.
            CHECK(!ChunkValuesAgree(target, source) && target->ComputeHash() != source->ComputeHash());
            const ecs::EntityRecord& first = *Impl(target).FindAlive(a[0]);
            const ecs::Archetype& merged = Impl(target).GetArchetypeStorage(first.archetype);
            CHECK(merged.entityCount == 20 && merged.GetChunkCount() == 1);
            for (uint32_t i = 0; i < 10; ++i)
            {
                CHECK(merged.Entities(0)[i] == a[i] && merged.Entities(0)[10 + i] == b[i]);
            }
            CHECK(GetComponent<Bounds>(target, b[0])->lo == 1.0f);
            CHECK(Dump(target, false) == Dump(source, false));

            // Other chunk sizes: the partition differs, each chunk takes the value of the saved chunk of its first row.
            std::vector<Entity> many(1000);
            desc.entityCount = 1000;
            CHECK(source->CreateEntities(desc, many.data()) == nvrhi::FS_OK);
            const ecs::Archetype& sourceArchetype = Impl(source).GetArchetypeStorage(Impl(source).FindAlive(many[0])->archetype);
            for (uint32_t c = 0; c < sourceArchetype.GetChunkCount(); ++c)
                GetComponentForWrite<Bounds>(source, sourceArchetype.Entities(c)[0])->hi = float(c + 1);
            auto small = MakeWorld(false, 4096, 4096);
            Register(small);
            CHECK(Load(small, Save(source)) == nvrhi::FS_OK);
            CheckIntegrity(small);
            CHECK(!ChunkValuesAgree(small, source) && small->ComputeHash() != source->ComputeHash());
            CHECK(Dump(small, false) == Dump(source, false));
            const ecs::Archetype& smaller = Impl(small).GetArchetypeStorage(Impl(small).FindAlive(many[0])->archetype);
            CHECK(smaller.rowsPerChunk < sourceArchetype.rowsPerChunk);
            // The loaded rows are [a, b, many]; a chunk's value is the one its first row had in the saved world.
            CHECK(smaller.entityCount == 1020 && smaller.GetChunkCount() > 1);
            for (uint32_t c = 0; c < smaller.GetChunkCount(); ++c)
            {
                const Entity head = smaller.Entities(c)[0];
                CHECK(GetComponent<Bounds>(small, head)->lo == GetComponent<Bounds>(source, head)->lo);
                CHECK(GetComponent<Bounds>(small, head)->hi == GetComponent<Bounds>(source, head)->hi);
            }

            // Another partition with the same per-entity Chunk values (all chunks alike) hashes equal.
            auto uniform = MakeWorld();
            const Ids u = Register(uniform);
            std::vector<Entity> rows(1000);
            EntityCreateDesc d{};
            const ComponentId list[] = { u.position, u.bounds };
            d.components = list;
            d.componentCount = 2;
            d.entityCount = 1000;
            CHECK(uniform->CreateEntities(d, rows.data()) == nvrhi::FS_OK);
            const ecs::Archetype& wide = Impl(uniform).GetArchetypeStorage(Impl(uniform).FindAlive(rows[0])->archetype);
            for (uint32_t c = 0; c < wide.GetChunkCount(); ++c)
                *GetComponentForWrite<Bounds>(uniform, wide.Entities(c)[0]) = Bounds{ -1.0f, 1.0f };
            auto narrow = MakeWorld(false, 4096, 4096);
            Register(narrow);
            CHECK(Load(narrow, Save(uniform)) == nvrhi::FS_OK);
            const ecs::Archetype& narrowArchetype =
                Impl(narrow).GetArchetypeStorage(Impl(narrow).FindAlive(rows[0])->archetype);
            CHECK(narrowArchetype.GetChunkCount() > wide.GetChunkCount());
            CHECK(ChunkValuesAgree(narrow, uniform) && narrow->ComputeHash() == uniform->ComputeHash());
            GetComponentForWrite<Bounds>(narrow, narrowArchetype.Entities(1)[0])->hi = 2.0f;
            CHECK(!ChunkValuesAgree(narrow, uniform) && narrow->ComputeHash() != uniform->ComputeHash());
        }
        CHECK(g_OwnersLive == owners);
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }

    void test_failures()
    {
        const int64_t blocks = ecs::GetHeapLiveBlockCount();
        {
            auto small = BuildSmall();
            const Bytes bytes = Save(small);
            printf("test_ecs_snapshot: the small snapshot has %zu bytes\n", bytes.size());
            CHECK(bytes.size() < 4096);

            // The target holds other content, non-serialisable components and Chunk values included. A refused load
            // must leave it exactly as it is: FullState compares the content, the versions and the placement.
            auto target = BuildScripted(ScriptSpec{ "target", 3, F_All, 150, 3 });
            const Bytes targetBytes = Save(target);
            std::string state = FullState(target);

            // Corruption at every byte, and truncation at every length.
            for (size_t at = 0; at < bytes.size(); ++at)
            {
                for (const uint8_t flip : { uint8_t(0x01), uint8_t(0xff) })
                {
                    Bytes corrupt = bytes;
                    corrupt[at] ^= flip;
                    const nvrhi::FRESULT result = Load(target, corrupt);
                    CHECK(NVRHI_FAILED(result));
                    CHECK(FullState(target) == state);
                }
                const Bytes truncated(bytes.begin(), bytes.begin() + at);
                CHECK(NVRHI_FAILED(Load(target, truncated)));
                CHECK(FullState(target) == state);
            }
            CheckIntegrity(target);

            // Behind a valid checksum, a corrupt structure is refused and changes nothing; a load that is accepted (a
            // changed value, generation or Shared key) gives a consistent world that round-trips. After each accepted
            // load the target is reloaded from its own snapshot, which restores the same content and Chunk values, and
            // given non-serialisable content again (a Sparse component and a resource: no row moves).
            const ComponentId note = FindComponent<Note>(target);
            auto refill = [&]()
            {
                CHECK(Load(target, targetBytes) == nvrhi::FS_OK);
                ecs::WorldImpl& impl = Impl(target);
                for (uint32_t index = 1; index < impl.GetEntityTableSize(); index += 3)
                {
                    const ecs::EntityRecord& record = impl.GetEntityRecord(index);
                    const Note value{ index * 7 };
                    if (record.archetype < ecs::c_ClaimedSlot)
                        CHECK(target->AddComponent(MakeEntity(index, record.generation), note, &value) == nvrhi::FS_OK);
                }
                GetResource<Transient>(target)->value = 0x5eed;
                return FullState(target);
            };
            state = refill();
            const uint64_t hash = target->ComputeHash();
            uint32_t accepted = 0;
            uint32_t refused = 0;
            for (size_t at = ecs::c_SnapshotHeaderBytes; at + ecs::c_SnapshotChecksumBytes < bytes.size(); ++at)
            {
                for (const int variant : { 0, 1, 2, 3 })
                {
                    Bytes corrupt = bytes;
                    switch (variant)
                    {
                    case 0: corrupt[at] ^= 0x01; break;
                    case 1: corrupt[at] ^= 0x80; break;
                    case 2: corrupt[at] = 0x00; break;
                    default: corrupt[at] = 0xff; break;
                    }
                    if (corrupt == bytes)
                        continue;
                    FixChecksum(corrupt);
                    const nvrhi::FRESULT result = Load(target, corrupt);
                    if (NVRHI_FAILED(result))
                    {
                        ++refused;
                        CHECK(result == nvrhi::FE_INVALID_ARGS || result == nvrhi::FE_NOT_FOUND);
                        CHECK(FullState(target) == state);
                        continue;
                    }
                    ++accepted;
                    CheckIntegrity(target);
                    auto copy = MakeWorld();
                    Register(copy);
                    CHECK(Load(copy, Save(target)) == nvrhi::FS_OK);
                    CHECK(copy->ComputeHash() == target->ComputeHash());
                    state = refill();
                    CHECK(target->ComputeHash() == hash);
                }
            }
            printf("test_ecs_snapshot: structural corruption: %u refused, %u accepted as other values\n", refused,
                   accepted);
            CHECK(refused > 0 && accepted > 0);

            // The header: magic, version, flags.
            Bytes badMagic = bytes;
            badMagic[0] = 'X';
            FixChecksum(badMagic);
            CHECK(Load(target, badMagic) == nvrhi::FE_INVALID_ARGS);
            Bytes newer = bytes;
            newer[ecs::c_SnapshotVersionOffset] = uint8_t(ecs::c_SnapshotVersion + 1);
            FixChecksum(newer);
            CHECK(Load(target, newer) == nvrhi::FE_UNSUPPORTED);
            Bytes flagged = bytes;
            flagged[ecs::c_SnapshotVersionOffset + 4] = 1;
            FixChecksum(flagged);
            CHECK(Load(target, flagged) == nvrhi::FE_INVALID_ARGS);
            CHECK(FullState(target) == state);

            // A reader's own failure code is returned.
            MemoryReader failing(bytes.data(), bytes.size() / 2);
            failing.failure = nvrhi::FE_GENERIC_ERROR;
            CHECK(target->LoadSnapshot(&failing) == nvrhi::FE_GENERIC_ERROR);
            CHECK(FullState(target) == state);

            // Components: unknown here, another size, other fields, other flags.
            auto lacking = MakeWorld(true);
            RegisterComponent<Position>(lacking);
            Create(lacking);
            const std::string lackingState = FullState(lacking);
            CHECK(Load(lacking, bytes) == nvrhi::FE_NOT_FOUND);
            CHECK(FullState(lacking) == lackingState && Impl(lacking).GetValidationErrorCount() == 1);

            auto mismatch = [&](const ComponentTypeDesc& positionDesc)
            {
                auto world = MakeWorld(true);
                std::vector<const ComponentTypeDesc*> types = AllTypes();
                for (const ComponentTypeDesc* desc : types)
                {
                    ComponentId id = ComponentId::Invalid;
                    CHECK(world->RegisterComponent(desc->uuid == positionDesc.uuid ? positionDesc : *desc, id) ==
                          nvrhi::FS_OK);
                }
                const Entity existing = Create(world);
                const std::string before = FullState(world);
                CHECK(Load(world, bytes) == nvrhi::FE_INVALID_ARGS);
                CHECK(FullState(world) == before && world->IsAlive(existing));
                CHECK(Impl(world).GetValidationErrorCount() == 1);
            };
            const ComponentTypeDesc& position = ComponentTraits<Position>::Desc();
            ComponentTypeDesc wider = position;
            wider.size = 16;
            mismatch(wider);
            static const FieldDesc otherFields[] = {
                { "xy", 0, 2, FieldType::Float, FieldShape::Vector, {} },
                { "z", 8, 1, FieldType::Float, FieldShape::Scalar, {} },
            };
            ComponentTypeDesc reshaped = position;
            reshaped.fields = otherFields;
            reshaped.fieldCount = 2;
            mismatch(reshaped);
            static const FieldDesc intFields[] = {
                { "x", 0, 1, FieldType::Int32, FieldShape::Scalar, {} },
                { "y", 4, 1, FieldType::Float, FieldShape::Scalar, {} },
                { "z", 8, 1, FieldType::Float, FieldShape::Scalar, {} },
            };
            ComponentTypeDesc retyped = position;
            retyped.fields = intFields;
            mismatch(retyped);
            ComponentTypeDesc enableable = position;
            enableable.flags = enableable.flags | ComponentFlags::Enableable;
            mismatch(enableable);

            // Rows that fit the saved world's chunks but not the target's.
            auto huge = MakeWorld();
            const ComponentId h1 = RegisterComponent<Huge>(huge);
            const ComponentId h2 = RegisterComponent<Huge2>(huge);
            const ComponentId pair[] = { h1, h2 };
            EntityCreateDesc desc{};
            desc.components = pair;
            desc.componentCount = 2;
            desc.entityCount = 1;
            Entity hugeEntity = Entity::Null;
            CHECK(huge->CreateEntities(desc, &hugeEntity) == nvrhi::FS_OK);
            auto tiny = MakeWorld(true, 1024, 1024);
            RegisterComponent<Huge>(tiny);
            RegisterComponent<Huge2>(tiny);
            const Entity resident = Create(tiny);
            const std::string tinyState = FullState(tiny);
            const uint32_t tinyArchetypes = tiny->GetArchetypeCount();
            CHECK(Load(tiny, Save(huge)) == nvrhi::FE_INVALID_ARGS);
            // Refused before any archetype is made: the structural version and the archetypes are as they were.
            CHECK(FullState(tiny) == tinyState && tiny->IsAlive(resident) && tiny->GetArchetypeCount() == tinyArchetypes);
            // Also when an earlier record's archetype would be new here: {Huge} fits and comes first, {Huge, Huge2} not.
            auto ordered = MakeWorld();
            const ComponentId o1 = RegisterComponent<Huge>(ordered);
            const ComponentId o2 = RegisterComponent<Huge2>(ordered);
            const Entity single = Create(ordered);
            CHECK(ordered->AddComponent(single, o1, nullptr) == nvrhi::FS_OK);
            const Entity both = Create(ordered);
            CHECK(ordered->AddComponent(both, o1, nullptr) == nvrhi::FS_OK);
            CHECK(ordered->AddComponent(both, o2, nullptr) == nvrhi::FS_OK);
            CHECK(Load(tiny, Save(ordered)) == nvrhi::FE_INVALID_ARGS);
            CHECK(FullState(tiny) == tinyState && tiny->GetArchetypeCount() == tinyArchetypes);

            // Phases: refused inside one, allowed for an exclusive system; ComputeHash reports, and still hashes.
            auto phased = MakeWorld(true);
            Register(phased);
            CHECK(Load(phased, bytes) == nvrhi::FS_OK);
            const uint64_t phasedHash = phased->ComputeHash();
            ecs::WorldImpl& impl = Impl(phased);
            impl.BeginPhase();
            MemoryWriter writer;
            CHECK(phased->SaveSnapshot(&writer) == nvrhi::FE_INVALID_ARGS && writer.bytes.empty());
            MemoryReader reader(targetBytes.data(), targetBytes.size());
            const std::string phasedState = FullState(phased);
            CHECK(phased->LoadSnapshot(&reader) == nvrhi::FE_INVALID_ARGS && reader.position == 0);
            CHECK(FullState(phased) == phasedState);
            const uint64_t errors = impl.GetValidationErrorCount();
            CHECK(phased->ComputeHash() == phasedHash && impl.GetValidationErrorCount() == errors + 1);
            impl.BeginExclusive();
            CHECK(Save(phased) == bytes);
            CHECK(Load(phased, targetBytes) == nvrhi::FS_OK && phased->ComputeHash() == hash);
            impl.EndExclusive();
            impl.EndPhase();

            // A writer's failure is returned, wherever it happens.
            for (const uint64_t limit : { uint64_t(0), uint64_t(10), uint64_t(ecs::c_SnapshotHeaderBytes),
                                          uint64_t(bytes.size() - 9), uint64_t(bytes.size() - 1) })
            {
                MemoryWriter limited;
                limited.limit = limit;
                CHECK(small->SaveSnapshot(&limited) == nvrhi::FE_UNSUPPORTED);
            }
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }

    void test_out_of_memory()
    {
        const int64_t blocks = ecs::GetHeapLiveBlockCount();
        {
            auto source = BuildScripted(ScriptSpec{ "oom", 21, F_Serial, 500, 5 });
            const Bytes bytes = Save(source);

            // A cold world: the first allocation of the load fails.
            auto cold = MakeWorld();
            Register(cold);
            Create(cold);
            const std::string coldState = FullState(cold);
            ecs::SetHeapFailureCountdown(0);
            const nvrhi::FRESULT coldResult = Load(cold, bytes);
            ecs::SetHeapFailureCountdown(-1);
            CHECK(coldResult == nvrhi::FE_OUT_OF_MEMORY && FullState(cold) == coldState);

            // A warm world, whose containers and pools already hold what the load needs: every allocation of the load
            // is a storage block that may fail softly. Each one in turn fails, and changes nothing.
            CHECK(Load(cold, bytes) == nvrhi::FS_OK);
            auto& warm = cold;
            *GetResource<Gravity>(warm) = Gravity{ -1.0, 99 };   // a change that makes no block
            CHECK(warm->ComputeHash() != source->ComputeHash());
            const std::string changed = FullState(warm);
            uint32_t failures = 0;
            for (int64_t k = 0;; ++k)
            {
                ecs::SetHeapFailureCountdown(k);
                const nvrhi::FRESULT result = Load(warm, bytes);
                ecs::SetHeapFailureCountdown(-1);
                if (result == nvrhi::FS_OK)
                    break;
                CHECK(result == nvrhi::FE_OUT_OF_MEMORY);
                CHECK(FullState(warm) == changed);
                ++failures;
                CHECK(failures < 100000);
            }
            printf("test_ecs_snapshot: out of memory: %u allocations of a warm load failed one by one\n", failures);
            CHECK(failures >= 4);   // the payload, the image's arrays, spilled buffers
            CheckIntegrity(warm);
            CHECK(warm->ComputeHash() == source->ComputeHash());
            CHECK(Dump(warm) == Dump(source));
        }
        {
            // The pools must hold every page the commit takes (review finding): the target has one row in a small and
            // one in a large chunk (slabs of 16 and 4 pages, partly used), and the snapshot needs 17 small and 6 large
            // chunks, more than the pools hold free and in use, so the load must grow them before the old content
            // goes. The containers the load grows are grown first, so that every allocation of the load is a soft
            // one: each fails in turn and changes nothing, Sparse components and their presence masks included.
            auto source = MakeWorld();
            const Ids ids = Register(source);
            auto fill = [&](ComponentId component, uint32_t chunks, std::vector<Entity>& rows)
            {
                EntityCreateDesc desc{};
                desc.components = &component;
                desc.componentCount = 1;
                desc.entityCount = 1;
                Entity first = Entity::Null;
                CHECK(source->CreateEntities(desc, &first) == nvrhi::FS_OK);
                ArchetypeInfo info{};
                CHECK(source->GetArchetypeInfo(Impl(source).FindAlive(first)->archetype, info) == nvrhi::FS_OK);
                rows.assign(size_t(chunks - 1) * info.rowsPerChunk + 1, Entity::Null);
                desc.entityCount = uint32_t(rows.size());
                CHECK(source->CreateEntities(desc, rows.data()) == nvrhi::FS_OK);
                CHECK(source->GetArchetypeInfo(Impl(source).FindAlive(first)->archetype, info) == nvrhi::FS_OK);
                CHECK(info.chunkCount == chunks);
            };
            std::vector<Entity> smallRows;
            std::vector<Entity> largeRows;
            fill(ids.position, 17, smallRows);
            fill(ids.big, 6, largeRows);
            for (uint32_t i = 0; i < 20; ++i)
            {
                const Color color{ 0xc0de0000u + i };
                CHECK(source->AddComponent(smallRows[i], ids.color, &color) == nvrhi::FS_OK);
                CHECK(source->AddComponent(largeRows[i], ids.color, &color) == nvrhi::FS_OK);
            }
            const Bytes bytes = Save(source);

            WorldDesc desc;
            desc.debugName = "test_ecs_snapshot pages";
            desc.initialEntityCapacity = 32768;   // the entity table does not grow during the load
            nvrhi::AutoPtr<IWorld> target;
            CHECK(CreateWorld(desc, &target) == nvrhi::FS_OK);
            const Ids t = Register(target);
            const Color color{ 7 };
            Entity resident[2] = {};
            EntityCreateDesc one{};
            one.componentCount = 1;
            one.entityCount = 1;
            one.components = &t.position;
            CHECK(target->CreateEntities(one, &resident[0]) == nvrhi::FS_OK);
            one.components = &t.big;
            CHECK(target->CreateEntities(one, &resident[1]) == nvrhi::FS_OK);
            CHECK(target->AddComponent(resident[0], t.color, &color) == nvrhi::FS_OK);
            CHECK(target->AddComponent(resident[1], t.color, &color) == nvrhi::FS_OK);
            ecs::WorldImpl& impl = Impl(target);
            for (uint32_t a = 0; a < impl.GetArchetypeStorageCount(); ++a)
                impl.GetArchetypeStorage(a).chunks.reserve(64);
            ecs::SparseSet& colors = const_cast<ecs::SparseSet&>(*impl.GetSparseSet(t.color));
            colors.entities.reserve(256);
            colors.pages.resize(16, nullptr);
            CHECK(impl.GetFreeChunkPageCount() + 2 < 17 + 6);   // free and in use: short of what the load needs

            const std::string before = FullState(target);
            uint32_t failures = 0;
            for (int64_t k = 0;; ++k)
            {
                ecs::SetHeapFailureCountdown(k);
                const nvrhi::FRESULT result = Load(target, bytes);
                ecs::SetHeapFailureCountdown(-1);
                if (result == nvrhi::FS_OK)
                    break;
                CHECK(result == nvrhi::FE_OUT_OF_MEMORY);
                CHECK(FullState(target) == before);
                CheckIntegrity(target);
                ++failures;
                CHECK(failures < 100000);
            }
            printf("test_ecs_snapshot: out of memory: %u allocations of a load into short pools failed one by one\n",
                   failures);
            CHECK(failures >= 6);   // besides the image: slabs of both pools, sparse values, an index page
            CheckIntegrity(target);
            CHECK(target->ComputeHash() == source->ComputeHash());
            CHECK(Dump(target) == Dump(source));
            CHECK(impl.GetSparseSet(t.color)->GetCount() == 40);
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }

    void test_hash_properties()
    {
        const int64_t blocks = ecs::GetHeapLiveBlockCount();
        {
            auto world = BuildSmall();
            const Ids ids = Register(world);
            const uint64_t base = world->ComputeHash();
            const Bytes baseBytes = Save(world);
            ecs::WorldImpl& impl = Impl(world);

            // Every field byte of every value (Chunk values included, as entities read them), element and resource
            // changes the hash; padding never does.
            auto sweep = [&](uint8_t* value, const ComponentTypeDesc& desc)
            {
                std::vector<bool> field(desc.size, false);
                for (uint32_t f = 0; f < desc.fieldCount; ++f)
                    for (uint32_t b = 0; b < ScalarSize(desc.fields[f].type) * desc.fields[f].scalarCount; ++b)
                        field[desc.fields[f].offset + b] = true;
                for (uint32_t b = 0; b < desc.size; ++b)
                {
                    const uint8_t old = value[b];
                    value[b] ^= 0x10;
                    const bool changed = world->ComputeHash() != base;
                    CHECK(changed == field[b]);
                    if (!field[b])
                        CHECK(Save(world) == baseBytes || desc.fieldCount == 0);   // padding is saved as 0
                    value[b] = old;
                }
                CHECK(world->ComputeHash() == base);
            };
            for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
            {
                const ecs::EntityRecord& record = impl.GetEntityRecord(index);
                if (record.archetype >= ecs::c_ClaimedSlot)
                    continue;
                const Entity entity = MakeEntity(index, record.generation);
                for (const ComponentTypeDesc* desc : AllTypes())
                {
                    const ComponentId id = world->FindComponent(desc->uuid);
                    if (Has(desc->flags, ComponentFlags::Resource) || !world->HasComponent(entity, id) ||
                        desc->size == 0 || Has(desc->flags, ComponentFlags::Shared))
                        continue;
                    uint8_t* value = static_cast<uint8_t*>(const_cast<void*>(world->GetComponent(entity, id)));
                    if (Has(desc->flags, ComponentFlags::Buffer))
                    {
                        const BufferHeader& header = *reinterpret_cast<const BufferHeader*>(value);
                        for (uint32_t i = 0; i < header.count; ++i)
                            sweep(static_cast<uint8_t*>(header.data) + size_t(i) * desc->size, *desc);
                    }
                    else if (desc->fieldCount == 0)
                    {
                        // Opaque: presence only.
                        value[0] ^= 1;
                        CHECK(world->ComputeHash() == base && Save(world) != baseBytes);
                        value[0] ^= 1;
                    }
                    else
                    {
                        sweep(value, *desc);
                    }
                }
            }
            sweep(static_cast<uint8_t*>(world->GetResource(ids.gravity)), ComponentTraits<Gravity>::Desc());
            sweep(static_cast<uint8_t*>(world->GetResource(ids.frameTime)), ComponentTraits<FrameTime>::Desc());
            CHECK(world->ComputeHash() == base);

            // A Shared key that differs only in padding is the same partition: same hash.
            Entity withLayer = Entity::Null;
            for (uint32_t index = 1; index < impl.GetEntityTableSize() && withLayer == Entity::Null; ++index)
            {
                const ecs::EntityRecord& record = impl.GetEntityRecord(index);
                const Entity entity = MakeEntity(index, record.generation);
                if (record.archetype < ecs::c_ClaimedSlot && world->HasComponent(entity, ids.layer))
                    withLayer = entity;
            }
            CHECK(withLayer != Entity::Null);
            Layer layer;
            std::memcpy(&layer, GetComponent<Layer>(world, withLayer), sizeof(layer));
            std::memset(reinterpret_cast<uint8_t*>(&layer) + 1, 0x77, 3);
            CHECK(world->AddComponent(withLayer, ids.layer, &layer) == nvrhi::FS_OK);
            CHECK(world->ComputeHash() == base);
            layer.mask ^= 1;
            CHECK(world->AddComponent(withLayer, ids.layer, &layer) == nvrhi::FS_OK);
            const uint64_t otherPartition = world->ComputeHash();
            CHECK(otherPartition != base);
            layer.mask ^= 1;
            CHECK(world->AddComponent(withLayer, ids.layer, &layer) == nvrhi::FS_OK);
            CHECK(world->ComputeHash() == base);

            // Versions, reads and non-serialisable components leave it (on an entity without Chunk components).
            Entity plain = Entity::Null;
            for (uint32_t index = 1; index < impl.GetEntityTableSize() && plain == Entity::Null; ++index)
            {
                const ecs::EntityRecord& record = impl.GetEntityRecord(index);
                const Entity entity = MakeEntity(index, record.generation);
                if (record.archetype < ecs::c_ClaimedSlot && world->HasComponent(entity, ids.position) &&
                    !world->HasComponent(entity, ids.bounds))
                    plain = entity;
            }
            CHECK(plain != Entity::Null);
            world->AdvanceVersion();
            CHECK(GetComponentForWrite<Position>(world, plain) != nullptr);
            const Cache cache{ 1 };
            CHECK(world->AddComponent(plain, ids.cache, &cache) == nvrhi::FS_OK);
            CHECK(world->AddComponent(plain, ids.owner, nullptr) == nvrhi::FS_OK);
            const Note note{ 2 };
            CHECK(world->AddComponent(plain, ids.note, &note) == nvrhi::FS_OK);
            GetResource<Transient>(world)->value ^= 0xffff;
            CHECK(world->ComputeHash() == base);
            CHECK(world->RemoveComponent(plain, ids.cache) == nvrhi::FS_OK);
            CHECK(world->RemoveComponent(plain, ids.owner) == nvrhi::FS_OK);
            CHECK(world->RemoveComponent(plain, ids.note) == nvrhi::FS_OK);
            CHECK(world->ComputeHash() == base);
            // A move into another chunk changes the hash exactly when it gives the entity another Chunk value: the
            // hash follows what the entity reads.
            const Bounds bounds = *GetComponent<Bounds>(world, withLayer);
            CHECK(world->AddComponent(withLayer, ids.cache, &cache) == nvrhi::FS_OK);
            CHECK(GetComponent<Bounds>(world, withLayer)->hi != bounds.hi && world->ComputeHash() != base);
            *GetComponentForWrite<Bounds>(world, withLayer) = bounds;
            CHECK(world->ComputeHash() == base);
            CHECK(world->RemoveComponent(withLayer, ids.cache) == nvrhi::FS_OK);
            CHECK(world->ComputeHash() == base);

            // Membership, enabled bits, buffer counts, sparse components, entities, reservations: each changes it.
            auto changes = [&](auto&& change, auto&& undo)
            {
                change();
                CHECK(world->ComputeHash() != base);
                undo();
                CHECK(world->ComputeHash() == base);
            };
            changes([&] { CHECK(world->AddComponent(withLayer, ids.frozen, nullptr) == nvrhi::FS_OK); },
                    [&] { CHECK(world->RemoveComponent(withLayer, ids.frozen) == nvrhi::FS_OK); });
            changes([&] { CHECK(world->AddComponent(withLayer, ids.selected, nullptr) == nvrhi::FS_OK); },
                    [&] { CHECK(world->RemoveComponent(withLayer, ids.selected) == nvrhi::FS_OK); });
            const bool enabled = world->IsEnabled(withLayer, ids.toggle);
            changes([&] { CHECK(world->SetEnabled(withLayer, ids.toggle, !enabled) == nvrhi::FS_OK); },
                    [&] { CHECK(world->SetEnabled(withLayer, ids.toggle, enabled) == nvrhi::FS_OK); });
            const uint32_t joints = static_cast<const BufferHeader*>(world->GetComponent(withLayer, ids.joint))->count;
            changes([&] { CHECK(world->ResizeBuffer(withLayer, ids.joint, joints + 1, c_WorldVersion) == nvrhi::FS_OK); },
                    [&] { CHECK(world->ResizeBuffer(withLayer, ids.joint, joints, c_WorldVersion) == nvrhi::FS_OK); });

            // A new entity, a destroyed one, a reservation, another generation: none can be undone, so compare with
            // the saved world each time.
            const uint64_t before = world->ComputeHash();
            const Entity added = Create(world);
            CHECK(world->ComputeHash() != before);
            const uint64_t withAdded = world->ComputeHash();
            CHECK(world->DestroyEntities(&added, 1) == nvrhi::FS_OK);
            CHECK(world->ComputeHash() != withAdded && world->ComputeHash() != before);   // a generation and the free list
            Entity reserved = Entity::Null;
            const uint64_t beforeReserve = world->ComputeHash();
            CHECK(world->ReserveEntities(1, &reserved) == nvrhi::FS_OK);
            CHECK(world->ComputeHash() != beforeReserve);
        }
        {
            // The same content in another placement hashes equal: row order, chunk sizes, registration order, sparse
            // dense order, archetype creation order. (No Chunk components here: their values follow the partition,
            // and the hash sees them per entity, test_coverage.)
            auto first = MakeWorld();
            auto second = MakeWorld(false, 4096, 8192);
            const Ids a = Register(first);
            const Ids b = Register(second, true);
            Entity x[2];
            Entity y[2];
            const ComponentId justPosition[] = { a.position };
            EntityCreateDesc desc{};
            desc.components = justPosition;
            desc.componentCount = 1;
            desc.entityCount = 2;
            CHECK(first->CreateEntities(desc, x) == nvrhi::FS_OK);   // rows [x0, x1]
            const ComponentId frozenPosition[] = { b.position, b.frozen };
            desc.components = frozenPosition;
            desc.componentCount = 2;
            desc.entityCount = 1;
            CHECK(second->CreateEntities(desc, &y[0]) == nvrhi::FS_OK);
            desc.components = &b.position;
            desc.componentCount = 1;
            CHECK(second->CreateEntities(desc, &y[1]) == nvrhi::FS_OK);
            CHECK(second->RemoveComponent(y[0], b.frozen) == nvrhi::FS_OK);   // rows [y1, y0]
            CHECK(x[0] == y[0] && x[1] == y[1]);
            CHECK(Impl(second).FindAlive(y[0])->row == 1);
            const Color c0{ 1 };
            const Color c1{ 2 };
            CHECK(first->AddComponent(x[0], a.color, &c0) == nvrhi::FS_OK);
            CHECK(first->AddComponent(x[1], a.color, &c1) == nvrhi::FS_OK);
            CHECK(second->AddComponent(y[1], b.color, &c1) == nvrhi::FS_OK);
            CHECK(second->AddComponent(y[0], b.color, &c0) == nvrhi::FS_OK);
            CHECK(first->ComputeHash() == second->ComputeHash());
            CHECK(Save(first) != Save(second));   // the snapshot keeps placement

            // Generations and the free-list order are part of it.
            auto third = MakeWorld();
            auto fourth = MakeWorld();
            Register(third);
            Register(fourth);
            Entity t[3];
            Entity f[3];
            desc.components = nullptr;
            desc.componentCount = 0;
            desc.entityCount = 3;
            CHECK(third->CreateEntities(desc, t) == nvrhi::FS_OK);
            CHECK(fourth->CreateEntities(desc, f) == nvrhi::FS_OK);
            CHECK(third->ComputeHash() == fourth->ComputeHash());
            CHECK(third->DestroyEntities(&t[0], 1) == nvrhi::FS_OK && third->DestroyEntities(&t[1], 1) == nvrhi::FS_OK);
            CHECK(fourth->DestroyEntities(&f[1], 1) == nvrhi::FS_OK && fourth->DestroyEntities(&f[0], 1) == nvrhi::FS_OK);
            CHECK(third->ComputeHash() != fourth->ComputeHash());
            auto fifth = MakeWorld();
            auto sixth = MakeWorld();
            Register(fifth);
            Register(sixth);
            Entity e5 = Entity::Null;
            Entity e6 = Entity::Null;
            desc.entityCount = 1;
            CHECK(fifth->CreateEntities(desc, &e5) == nvrhi::FS_OK);
            CHECK(fifth->DestroyEntities(&e5, 1) == nvrhi::FS_OK);
            CHECK(fifth->CreateEntities(desc, &e5) == nvrhi::FS_OK);   // index 1, generation 2
            CHECK(sixth->CreateEntities(desc, &e6) == nvrhi::FS_OK);   // index 1, generation 1
            CHECK(GetEntityIndex(e5) == GetEntityIndex(e6) && e5 != e6);
            CHECK(fifth->ComputeHash() != sixth->ComputeHash());
        }
        {
            // Determinism: the same script, the heap perturbed in between.
            const ScriptSpec spec = { "determinism", 31, F_All, 1200, 5 };
            auto one = BuildScripted(spec);
            PerturbHeap(77);
            auto two = BuildScripted(spec);
            CHECK(one->ComputeHash() == two->ComputeHash());
            CHECK(Save(one) == Save(two));
            CHECK(one->ComputeHash() == one->ComputeHash());
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }

    const ScriptSpec c_Corpus[] = {
        { "empty", 1, F_Resources, 0, 1 },
        { "flat", 2, F_Data, 60, 200 },
        { "churn", 3, F_Data | F_Tags | F_Reserve, 3000, 2 },
        { "partitions", 4, F_Data | F_Tags | F_Shared | F_Chunk | F_Enable, 1500, 8 },
        { "buffers", 5, F_Data | F_Buffers | F_Links, 1500, 4 },
        { "sparse", 6, F_Data | F_Sparse | F_Reserve | F_Tags, 1500, 6 },
        { "large_rows", 7, F_Data | F_Large | F_Chunk, 800, 6 },
        { "mixed", 8, F_All, 3000, 5 },
    };

    std::string ReadText(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            return std::string();
        std::ostringstream text;
        text << file.rdbuf();
        return text.str();
    }

    void test_golden_corpus()
    {
        const int64_t blocks = ecs::GetHeapLiveBlockCount();
        const char* update = std::getenv("ECS_UPDATE_GOLDEN");
        const bool rewrite = update && std::strcmp(update, "1") == 0;
        std::string actual = "# ECS world-hash corpus (test_ecs_snapshot.cpp; ECS_UPDATE_GOLDEN=1 rewrites it)\n"
                             "# script world-hash snapshot-xxh64 snapshot-bytes entities\n";
        for (const ScriptSpec& spec : c_Corpus)
        {
            auto world = BuildScripted(spec);
            CheckIntegrity(world);
            const uint64_t hash = world->ComputeHash();
            const Bytes bytes = Save(world);
            PerturbHeap(spec.seed);
            auto again = BuildScripted(spec);
            CHECK(again->ComputeHash() == hash && Save(again) == bytes);
            auto loaded = MakeWorld();
            Register(loaded);
            CHECK(Load(loaded, bytes) == nvrhi::FS_OK);
            CheckIntegrity(loaded);
            CHECK(Dump(loaded, !(spec.features & F_NonSerial)) == Dump(world, !(spec.features & F_NonSerial)));
            // Without non-serialisable components the partition, and so every Chunk value, is the same; with them,
            // merged archetypes may give an entity another Chunk value, and the hash differs exactly then.
            CHECK((loaded->ComputeHash() == hash) == ChunkValuesAgree(loaded, world));
            if (!(spec.features & F_NonSerial))
                CHECK(loaded->ComputeHash() == hash && Save(loaded) == bytes);
            WorldStats stats;
            char line[160];
            std::snprintf(line, sizeof(line), "%s %016llx %016llx %zu %u\n", spec.name, (unsigned long long)hash,
                          (unsigned long long)ecs::Xxh64(bytes.data(), bytes.size()), bytes.size(),
                          world->GetStats(stats).entityCount);
            actual += line;
        }
        const std::filesystem::path path = std::filesystem::path(DONUT_TEST_GOLDEN_DIR) / "ecs_world_hashes.txt";
        if (rewrite)
        {
            std::ofstream file(path, std::ios::binary);
            file << actual;
            CHECK(file.good());
            printf("test_ecs_snapshot: wrote %s\n", path.generic_string().c_str());
        }
        else
        {
            const std::string expected = ReadText(path);
            if (expected != actual)
            {
                printf("test_ecs_snapshot: the corpus differs from %s (ECS_UPDATE_GOLDEN=1 rewrites it)\n"
                       "expected:\n%sactual:\n%s", path.generic_string().c_str(), expected.c_str(), actual.c_str());
                CHECK(expected == actual);
            }
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }

    // GetStats against numbers computed from the storage, before and after a load.
    void CheckStats(IWorld* world)
    {
        ecs::WorldImpl& impl = Impl(world);
        WorldStats stats;
        world->GetStats(stats);
        uint32_t alive = 0;
        for (uint32_t index = 1; index < impl.GetEntityTableSize(); ++index)
            alive += impl.GetEntityRecord(index).archetype < ecs::c_ClaimedSlot ? 1 : 0;
        uint32_t chunks = 0;
        uint64_t chunkBytes = 0;
        uint64_t rowBytes = 0;
        uint64_t heapBytes = 0;
        for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
        {
            ArchetypeInfo info{};
            CHECK(world->GetArchetypeInfo(a, info) == nvrhi::FS_OK);
            chunks += info.chunkCount;
            chunkBytes += uint64_t(info.chunkCount) * info.chunkSize;
            rowBytes += uint64_t(info.entityCount) * info.rowSize;
            const ecs::Archetype& archetype = impl.GetArchetypeStorage(a);
            for (uint32_t k = 0; k < archetype.columns.size(); ++k)
            {
                const ecs::ArchetypeColumn& column = archetype.columns[k];
                if (column.kind != ecs::StorageKind::Buffer)
                    continue;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                    for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                    {
                        const BufferHeader& header = static_cast<const BufferHeader*>(archetype.ColumnData(c, k))[r];
                        if (header.count > column.type->desc.bufferInlineCapacity)
                            heapBytes += uint64_t(header.capacity) * column.type->desc.size;
                    }
            }
        }
        uint32_t sparseSets = 0;
        for (uint32_t t = 0; t < world->GetComponentTypeCount(); ++t)
            sparseSets += impl.GetSparseSet(ComponentId(t)) ? 1 : 0;
        CHECK(stats.entityCount == alive);
        CHECK(stats.entityCapacity >= impl.GetEntityTableSize() - 1);
        CHECK(stats.componentTypeCount == world->GetComponentTypeCount());
        CHECK(stats.archetypeCount == world->GetArchetypeCount());
        CHECK(stats.chunkCount == chunks && stats.chunkBytes == chunkBytes && stats.rowBytes == rowBytes);
        CHECK(stats.bufferHeapBytes == heapBytes);
        CHECK(stats.sparseSetCount == sparseSets);
        CHECK(stats.version == world->GetVersion() && stats.structuralVersion == world->GetStructuralVersion());
        CHECK(stats.commandsPlayedBack == 0 && stats.observerBatches == 0);
    }

    void test_stats()
    {
        const int64_t blocks = ecs::GetHeapLiveBlockCount();
        {
            auto source = BuildScripted(c_Corpus[7]);
            CheckStats(source);
            auto loaded = MakeWorld();
            Register(loaded);
            const uint64_t structural = loaded->GetStructuralVersion();
            CHECK(Load(loaded, Save(source)) == nvrhi::FS_OK);
            CHECK(loaded->GetStructuralVersion() > structural);
            CheckStats(loaded);
            WorldStats a;
            WorldStats b;
            source->GetStats(a);
            loaded->GetStats(b);
            CHECK(a.entityCount == b.entityCount && a.componentTypeCount == b.componentTypeCount);
            CHECK(a.sparseSetCount == b.sparseSetCount);
            CHECK(b.bufferHeapBytes <= a.bufferHeapBytes);   // loaded buffers spill with their exact count
        }
        CHECK(ecs::GetHeapLiveBlockCount() == blocks);
    }
}

int main(int, char**)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // progress survives a crash
    struct Test
    {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        { "xxh64", &test_xxh64 },
        { "round_trip", &test_round_trip },
        { "coverage", &test_coverage },
        { "failures", &test_failures },
        { "out_of_memory", &test_out_of_memory },
        { "hash_properties", &test_hash_properties },
        { "golden_corpus", &test_golden_corpus },
        { "stats", &test_stats },
    };
    try
    {
        for (const Test& test : tests)
        {
            test.run();
            printf("test_ecs_snapshot: %s passed\n", test.name);
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
