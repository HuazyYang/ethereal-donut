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

// Two storage additions of plan step E3 (aggregate docs/plans/2026-11-21-donut-ecs.md), private to donut_engine:
// - Archetype::rowMoves counts removals of rows and released chunks, so that a remembered (chunk, row) of an entity is
//   its place while the count is unchanged (the scene graph's row hints). Checked against the entity table over a
//   seeded script of creations, destructions, moves and a snapshot load.
// - PagePool::ShareSlabs (WorldImpl::ShareChunkPages): a world's chunk slabs come from a process-wide cache and go back
//   to it when the world is destroyed; worlds that do not share never touch it; ReleaseSharedSlabs empties it.

#include <donut/engine/EcsWorld.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "EcsWorldImpl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    struct Position { float x, y, z; };
    DONUT_ECS_COMPONENT(Position, "e605fb33-a49d-4ac5-bfc7-7c982cdb3169", DONUT_ECS_FIELDS(x, y, z))
    struct Mass { double value; };
    DONUT_ECS_COMPONENT(Mass, "c62ec65a-d039-403b-bbb1-4ed82a61165e", DONUT_ECS_FIELDS(value))

    class Rng
    {
    public:
        explicit Rng(uint64_t seed) : m_State(seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull) { }
        uint32_t Below(uint32_t bound)
        {
            m_State ^= m_State >> 12;
            m_State ^= m_State << 25;
            m_State ^= m_State >> 27;
            return bound ? uint32_t((m_State * 0x2545F4914F6CDD1Dull) % bound) : 0;
        }

    private:
        uint64_t m_State;
    };

    nvrhi::AutoPtr<IWorld> MakeWorld(bool share)
    {
        nvrhi::AutoPtr<IWorld> world;
        WorldDesc desc;
        desc.chunkSize = 4096;   // small chunks: many of them
        desc.largeChunkSize = 4096;
        CHECK(NVRHI_SUCCEEDED(CreateWorld(desc, &world)));
        if (share)
            ecs::GetWorldImpl(world)->ShareChunkPages();
        return world;
    }

    std::vector<Entity> Create(IWorld* world, uint32_t count, bool withMass, float base)
    {
        const ComponentId ids[2] = { RegisterComponent<Position>(world), RegisterComponent<Mass>(world) };
        std::vector<Position> positions(count);
        std::vector<Mass> masses(count);
        for (uint32_t i = 0; i < count; i++)
        {
            positions[i] = Position{ base + float(i), 0.f, 0.f };
            masses[i] = Mass{ double(i) };
        }
        const void* values[2] = { positions.data(), masses.data() };
        EntityCreateDesc desc{};
        desc.components = ids;
        desc.values = values;
        desc.componentCount = withMass ? 2 : 1;
        desc.entityCount = count;
        desc.transfer = ValueTransfer::Copy;
        std::vector<Entity> entities(count);
        CHECK(NVRHI_SUCCEEDED(world->CreateEntities(desc, entities.data())));
        return entities;
    }

    struct Hint
    {
        Entity entity;
        uint32_t archetype, chunk, row;
        uint64_t moves;
    };

    Hint Remember(ecs::WorldImpl* impl, Entity entity)
    {
        const ecs::EntityRecord* record = impl->FindAlive(entity);
        CHECK(record);
        return Hint{ entity, record->archetype, record->chunk, record->row,
                     impl->GetArchetypeStorage(record->archetype).rowMoves };
    }

    // A hint whose count is unchanged names the entity's row; the converse is not required.
    void CheckHint(ecs::WorldImpl* impl, const Hint& hint, uint64_t& validHints)
    {
        const ecs::Archetype& archetype = impl->GetArchetypeStorage(hint.archetype);
        if (archetype.rowMoves != hint.moves)
            return;
        ++validHints;
        CHECK(hint.chunk < archetype.GetChunkCount() && hint.row < archetype.chunks[hint.chunk].count);
        CHECK(archetype.Entities(hint.chunk)[hint.row] == hint.entity);
        const ecs::EntityRecord* record = impl->FindAlive(hint.entity);
        CHECK(record && record->archetype == hint.archetype && record->chunk == hint.chunk && record->row == hint.row);
    }

    struct MemoryWriter final : IStateWriter
    {
        std::vector<uint8_t> bytes;
        nvrhi::FRESULT QueryInterface(nvrhi::FREFIID, void** out) noexcept override
        {
            if (out)
                *out = nullptr;
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

    struct MemoryReader final : IStateReader
    {
        const std::vector<uint8_t>& bytes;
        uint64_t position = 0;
        explicit MemoryReader(const std::vector<uint8_t>& b) : bytes(b) { }
        nvrhi::FRESULT QueryInterface(nvrhi::FREFIID, void** out) noexcept override
        {
            if (out)
                *out = nullptr;
            return nvrhi::FE_NOINTERFACE;
        }
        nvrhi::FLONG AddRef() noexcept override { return 1; }
        nvrhi::FLONG Release() noexcept override { return 1; }
        nvrhi::FRESULT Read(uint64_t count, void* out) noexcept override
        {
            if (count > bytes.size() - position)
                return nvrhi::FE_INVALID_ARGS;
            std::memcpy(out, bytes.data() + position, size_t(count));
            position += count;
            return nvrhi::FS_OK;
        }
    };
}

// Appends keep every hint; removals, moves and released chunks change the count of the archetypes they touch.
void test_row_moves()
{
    auto world = MakeWorld(false);
    ecs::WorldImpl* impl = ecs::GetWorldImpl(world);
    const ComponentId mass = RegisterComponent<Mass>(world);

    // appends move nothing
    auto first = Create(world, 300, false, 0.f);
    const Hint early = Remember(impl, first[5]);
    Create(world, 500, false, 1000.f);
    uint64_t valid = 0;
    CheckHint(impl, early, valid);
    CHECK(valid == 1);

    // a destruction swap-removes: the count of that archetype changes, not of others
    auto other = Create(world, 10, true, 0.f);
    const Hint otherHint = Remember(impl, other[3]);
    const uint64_t before = impl->GetArchetypeStorage(early.archetype).rowMoves;
    CHECK(NVRHI_SUCCEEDED(world->DestroyEntities(&first[7], 1)));
    CHECK(impl->GetArchetypeStorage(early.archetype).rowMoves != before);
    valid = 0;
    CheckHint(impl, otherHint, valid);
    CHECK(valid == 1);

    // a move out of an archetype counts there; the target only appends
    const Hint moved = Remember(impl, first[9]);
    const Hint target = Remember(impl, other[0]);
    CHECK(NVRHI_SUCCEEDED(world->AddComponent(first[9], mass, nullptr)));
    CHECK(impl->GetArchetypeStorage(moved.archetype).rowMoves != moved.moves);
    valid = 0;
    CheckHint(impl, target, valid);
    CHECK(valid == 1);

    // a seeded script: every hint whose count is unchanged still names its entity's row
    Rng rng(17);
    std::vector<Entity> alive = first;
    alive.erase(alive.begin() + 7);
    std::vector<Hint> hints;
    uint64_t checked = 0, holding = 0;
    for (int step = 0; step < 2000; step++)
    {
        const uint32_t op = rng.Below(10);
        if (op < 3)
        {
            auto created = Create(world, 1 + rng.Below(40), rng.Below(2) != 0, float(step));
            alive.insert(alive.end(), created.begin(), created.end());
        }
        else if (op < 6 && !alive.empty())
        {
            const uint32_t i = rng.Below(uint32_t(alive.size()));
            CHECK(NVRHI_SUCCEEDED(world->DestroyEntities(&alive[i], 1)));
            alive.erase(alive.begin() + i);
        }
        else if (op < 8 && !alive.empty())
        {
            const Entity e = alive[rng.Below(uint32_t(alive.size()))];
            const nvrhi::FRESULT result = world->HasComponent(e, mass) ? world->RemoveComponent(e, mass)
                                                                         : world->AddComponent(e, mass, nullptr);
            CHECK(NVRHI_SUCCEEDED(result));
        }
        else if (!alive.empty())
            hints.push_back(Remember(impl, alive[rng.Below(uint32_t(alive.size()))]));
        for (const Hint& hint : hints)
        {
            if (!world->IsAlive(hint.entity))
                continue;
            ++checked;
            CheckHint(impl, hint, holding);
        }
    }
    CHECK(checked > 10000 && holding > 100);

    // a snapshot load replaces every row: every count changes
    MemoryWriter writer;
    CHECK(NVRHI_SUCCEEDED(world->SaveSnapshot(&writer)));
    const Hint loaded = Remember(impl, alive.front());
    MemoryReader reader(writer.bytes);
    CHECK(NVRHI_SUCCEEDED(world->LoadSnapshot(&reader)));
    CHECK(impl->GetArchetypeStorage(loaded.archetype).rowMoves != loaded.moves);
    std::printf("row moves: %llu hints checked, %llu still valid\n", (unsigned long long)checked,
                (unsigned long long)holding);
}

// Worlds that share take their slabs from the cache and give them back; others never touch it.
void test_shared_slabs()
{
    ecs::ReleaseSharedSlabs();
    const int64_t liveBefore = ecs::GetHeapLiveBlockCount();
    CHECK(ecs::GetSharedSlabBytes() == 0);

    // a world that does not share leaves the cache alone and frees everything
    {
        auto world = MakeWorld(false);
        Create(world, 5000, true, 0.f);
    }
    CHECK(ecs::GetSharedSlabBytes() == 0);
    CHECK(ecs::GetHeapLiveBlockCount() == liveBefore);

    // a sharing world gives its slabs back on destruction
    uint64_t cached = 0;
    uint64_t firstAllocations = 0;
    {
        auto world = MakeWorld(true);
        const uint64_t before = ecs::GetHeapAllocationCount();
        Create(world, 5000, true, 0.f);
        firstAllocations = ecs::GetHeapAllocationCount() - before;
    }
    cached = ecs::GetSharedSlabBytes();
    CHECK(cached >= uint64_t(5000) * (sizeof(Position) + sizeof(Mass)));
    CHECK(cached % ecs::PagePool::c_SlabBytes == 0);

    // the next sharing world takes them: the same content allocates exactly the slabs fewer
    {
        auto world = MakeWorld(true);
        const uint64_t before = ecs::GetHeapAllocationCount();
        auto entities = Create(world, 5000, true, 7.f);
        const uint64_t allocations = ecs::GetHeapAllocationCount() - before;
        CHECK(ecs::GetSharedSlabBytes() == 0);
        CHECK(allocations + cached / ecs::PagePool::c_SlabBytes == firstAllocations);
        const Position* p = GetComponent<Position>(world, entities[4321]);
        CHECK(p && p->x == 7.f + 4321.f);
    }
    CHECK(ecs::GetSharedSlabBytes() == cached);

    // a world without sharing next to it does not take them
    {
        auto world = MakeWorld(false);
        Create(world, 5000, true, 0.f);
        CHECK(ecs::GetSharedSlabBytes() == cached);
    }

    ecs::ReleaseSharedSlabs();
    CHECK(ecs::GetSharedSlabBytes() == 0);
    CHECK(ecs::GetHeapLiveBlockCount() == liveBefore);
}

int main(int, char**)
{
    try
    {
        test_row_moves();
        test_shared_slabs();
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    return 0;
}
