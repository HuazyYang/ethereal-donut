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

// Asset tables and stable slots (aggregate docs/plans/2026-11-21-donut-ecs.md, step E4; docs/ecs-design.md 4.8, 7.2):
// the slot allocator against a reference model over a seeded script, the material and mesh tables (references, slots
// written into the objects, generations, iteration in slot order, the events behind SceneGraph's callbacks, the mesh
// row's materials and skin prototype, Reconcile after meshes are edited in place).

#include <donut/engine/AssetTables.h>
#include <donut/tests/utils.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    // xorshift64* with an explicit seed (plan conventions: no std:: distributions).
    struct Rng
    {
        uint64_t state;
        uint64_t Next()
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 0x2545F4914F6CDD1Dull;
        }
        uint32_t Below(uint32_t bound) { return uint32_t(Next() % bound); }
    };

    nvrhi::AutoPtr<Material> make_material(const char* name)
    {
        auto material = MAKE_RC_OBJ_PTR(Material);
        material->name = name;
        return material;
    }

    nvrhi::AutoPtr<MeshInfo> make_mesh(const char* name, std::initializer_list<Material*> materials)
    {
        auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
        mesh->name = name;
        for (Material* material : materials)
        {
            auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
            geometry->material = material;
            mesh->geometries.push_back(geometry);
        }
        return mesh;
    }

    // "+m:name", "+t:name", "-m:name", "-t:name" for the events of a Changes list.
    std::string describe(const MeshTable::Changes& changes)
    {
        std::string text;
        for (const MeshTable::Change& change : changes.events)
        {
            text += change.added ? '+' : '-';
            text += change.mesh ? "m:" + change.mesh->name : "t:" + change.material->name;
            text += ' ';
        }
        return text;
    }

    // The reference model of SlotAllocator: occupancy per slot, first fit from the lowest slot; the end grows by what
    // does not fit, starting at the highest free run that reaches it.
    struct ModelAllocator
    {
        std::vector<bool> used;
        uint32_t Allocate(uint32_t count)
        {
            if (count == 0)
                return 0;
            uint32_t run = 0;
            for (uint32_t slot = 0; slot < used.size(); ++slot)
            {
                run = used[slot] ? 0 : run + 1;
                if (run == count)
                {
                    const uint32_t first = slot + 1 - count;
                    for (uint32_t i = first; i <= slot; ++i)
                        used[i] = true;
                    return first;
                }
            }
            // the free run at the end (`run` slots) is extended
            const uint32_t first = uint32_t(used.size()) - run;
            used.resize(size_t(first) + count, false);
            for (uint32_t i = first; i < first + count; ++i)
                used[i] = true;
            return first;
        }
        void Free(uint32_t first, uint32_t count)
        {
            for (uint32_t i = first; i < first + count; ++i)
            {
                CHECK(used[i]);
                used[i] = false;
            }
        }
    };
}

void test_slot_allocator_basics()
{
    SlotAllocator slots;
    CHECK(slots.Allocate(0) == 0 && slots.GetEnd() == 0 && slots.GetUsed() == 0);
    for (uint32_t i = 0; i < 8; ++i)
        CHECK(slots.Allocate(1) == i);
    CHECK(slots.GetEnd() == 8 && slots.GetUsed() == 8);

    // lowest first, and frees coalesce
    slots.Free(5, 1);
    slots.Free(2, 1);
    slots.Free(3, 1);
    CHECK(slots.GetFreeRangeCount() == 2);   // [2, 4) and [5, 6)
    slots.Free(4, 1);
    CHECK(slots.GetFreeRangeCount() == 1);   // [2, 6)
    CHECK(slots.Allocate(1) == 2);
    CHECK(slots.Allocate(3) == 3);           // first fit: [3, 6)
    CHECK(slots.GetFreeRangeCount() == 0 && slots.GetUsed() == 8 && slots.GetEnd() == 8);

    // a range that fits nowhere goes past the end; one at the end grows the free run that reaches it
    slots.Free(6, 2);
    CHECK(slots.Allocate(4) == 6 && slots.GetEnd() == 10);
    CHECK(slots.Allocate(2) == 10 && slots.GetEnd() == 12);
    // the end is a high-water mark
    slots.Free(6, 6);
    CHECK(slots.GetEnd() == 12 && slots.GetUsed() == 6);
    CHECK(slots.Allocate(5) == 6);

    slots.Reset();
    CHECK(slots.GetEnd() == 0 && slots.GetUsed() == 0 && slots.Allocate(1) == 0);
}

void test_slot_allocator_against_model()
{
    for (uint64_t seed : { 1ull, 7ull, 0x9e3779b97f4a7c15ull })
    {
        std::vector<uint32_t> trace[2];
        for (int run = 0; run < 2; ++run)
        {
            Rng rng{ seed };
            SlotAllocator slots;
            ModelAllocator model;
            struct Live { uint32_t first, count; };
            std::vector<Live> live;
            uint32_t used = 0;
            for (int step = 0; step < 6000; ++step)
            {
                const bool allocate = live.empty() || rng.Below(100) < 55;
                if (allocate)
                {
                    const uint32_t count = rng.Below(4) == 0 ? rng.Below(9) : 1;   // mostly single slots
                    const uint32_t first = slots.Allocate(count);
                    CHECK(first == model.Allocate(count));
                    trace[run].push_back(first);
                    if (count != 0)
                        live.push_back(Live{ first, count });
                    used += count;
                }
                else
                {
                    const uint32_t index = rng.Below(uint32_t(live.size()));
                    slots.Free(live[index].first, live[index].count);
                    model.Free(live[index].first, live[index].count);
                    used -= live[index].count;
                    live[index] = live.back();
                    live.pop_back();
                }
                CHECK(slots.GetUsed() == used);
                CHECK(slots.GetEnd() >= model.used.size());
            }
        }
        CHECK(trace[0] == trace[1]);   // deterministic
    }
}

void test_material_table()
{
    MaterialTable table;
    auto a = make_material("a");
    auto b = make_material("b");
    auto c = make_material("c");
    bool added = false;

    const MaterialId idA = table.Acquire(a, added);
    CHECK(added && a->materialID == 0 && table.Get(idA) == a.Get());
    CHECK(table.Acquire(a, added) == idA && !added);
    const MaterialId idB = table.Acquire(b, added);
    CHECK(added && b->materialID == 1);
    CHECK(table.size() == 2 && table.Find(b) == idB && table.Find(c) == MaterialId::Invalid);
    CHECK(table.Acquire(nullptr, added) == MaterialId::Invalid && !added);

    // a holds two references: the first release keeps it
    CHECK(!table.Release(idA));
    CHECK(table.Get(idA) == a.Get());
    nvrhi::AutoPtr<Material> released = table.Release(idA);
    CHECK(released.Get() == a.Get() && table.Get(idA) == nullptr && table.size() == 1);

    // the slot goes to the next material, with another generation: the old id names nothing
    const MaterialId idC = table.Acquire(c, added);
    CHECK(added && c->materialID == 0 && MaterialTable::SlotOf(idC) == 0 && idC != idA);
    CHECK(table.Get(idA) == nullptr && table.Get(idC) == c.Get());

    // iteration in slot order, over the rows in use
    std::vector<Material*> order;
    for (Material* material : table)
        order.push_back(material);
    CHECK(order.size() == 2 && order[0] == c.Get() && order[1] == b.Get());
    CHECK(table.GetRow(0).serial > table.GetRow(1).serial && table.GetSerial() == 3);
}

void test_mesh_table()
{
    MeshTable meshes;
    MaterialTable materials;
    MeshTable::Changes changes;
    auto stone = make_material("stone");
    auto glass = make_material("glass");
    auto house = make_mesh("house", { stone, glass, stone });
    auto rock = make_mesh("rock", { stone });

    const MeshId houseId = meshes.Acquire(house, materials, changes);
    CHECK(describe(changes) == "+m:house +t:stone +t:glass ");
    CHECK(changes.geometriesAdded == 3);
    CHECK(house->globalMeshIndex == 0);
    CHECK(house->geometries[0]->globalGeometryIndex == 0 && house->geometries[2]->globalGeometryIndex == 2);
    CHECK(stone->materialID == 0 && glass->materialID == 1);
    CHECK(meshes.GetMeshRow(0).firstGeometry == 0 && meshes.GetMeshRow(0).geometryCount == 3);
    CHECK(meshes.GetGeometryEnd() == 3 && meshes.GetGeometryCount() == 3);

    // a second reference adds nothing
    changes.Clear();
    CHECK(meshes.Acquire(house, materials, changes) == houseId && changes.events.empty());

    const MeshId rockId = meshes.Acquire(rock, materials, changes);
    CHECK(describe(changes) == "+m:rock ");   // stone is there: the rock row only references it
    CHECK(rock->globalMeshIndex == 1 && rock->geometries[0]->globalGeometryIndex == 3);

    // house leaves with its last reference: its materials lose the references of its row, glass goes
    changes.Clear();
    meshes.Release(houseId, materials, changes);
    CHECK(changes.events.empty() && meshes.size() == 2);
    meshes.Release(houseId, materials, changes);
    CHECK(describe(changes) == "-m:house -t:glass ");
    CHECK(meshes.size() == 1 && materials.size() == 1 && meshes.GetGeometryCount() == 1);

    // a mesh of two geometries takes the first free range that fits ([0, 3) is free)
    auto pair = make_mesh("pair", { glass, glass });
    changes.Clear();
    meshes.Acquire(pair, materials, changes);
    CHECK(describe(changes) == "+m:pair +t:glass ");
    CHECK(pair->globalMeshIndex == 0 && pair->geometries[0]->globalGeometryIndex == 0);
    CHECK(glass->materialID == 1);   // glass's old slot was free again, and the lowest
    CHECK(meshes.GetGeometryEnd() == 4);

    // a skinned mesh: the mesh, its materials, then its prototype (RegisterLeaf's order); released likewise
    auto prototype = make_mesh("prototype", { stone });
    auto skinned = make_mesh("skinned", { stone });
    skinned->skinPrototype = prototype;
    changes.Clear();
    const MeshId skinnedId = meshes.Acquire(skinned, materials, changes);
    CHECK(describe(changes) == "+m:skinned +m:prototype ");
    CHECK(changes.geometriesAdded == 2);
    CHECK(meshes.Get(meshes.GetMeshRow(MeshTable::SlotOf(skinnedId)).prototype) == prototype.Get());
    changes.Clear();
    meshes.Release(skinnedId, materials, changes);
    CHECK(describe(changes) == "-m:skinned -m:prototype ");

    // iteration in slot order
    std::vector<std::string> names;
    for (MeshInfo* mesh : meshes)
        names.push_back(mesh->name);
    CHECK(names.size() == 2 && names[0] == "pair" && names[1] == "rock");
    (void)rockId;
}

void test_mesh_table_reconcile()
{
    MeshTable meshes;
    MaterialTable materials;
    MeshTable::Changes changes;
    auto stone = make_material("stone");
    auto glass = make_material("glass");
    auto a = make_mesh("a", { stone, stone });
    auto b = make_mesh("b", { stone });
    meshes.Acquire(a, materials, changes);
    meshes.Acquire(b, materials, changes);
    const uint32_t slotA = MeshTable::SlotOf(meshes.Find(a));

    // nothing edited: nothing changes, not even the serial
    changes.Clear();
    const uint64_t serial = meshes.GetRow(slotA).serial;
    CHECK(!meshes.Reconcile(slotA, materials, changes) && changes.events.empty());
    CHECK(meshes.GetRow(slotA).serial == serial);

    // geometries replaced by new objects of the same count: their slots are written again
    auto replacement = make_mesh("tmp", { stone, glass });
    a->geometries = replacement->geometries;
    CHECK(!meshes.Reconcile(slotA, materials, changes));
    CHECK(a->geometries[0]->globalGeometryIndex == 0 && a->geometries[1]->globalGeometryIndex == 1);
    CHECK(describe(changes) == "+t:glass ");   // the new geometry's material is referenced by the row
    CHECK(meshes.GetRow(slotA).serial > serial);

    // a third geometry: new geometry slots (the old range is too small and b's lies after it)
    changes.Clear();
    auto third = MAKE_RC_OBJ_PTR(MeshGeometry);
    third->material = stone;
    a->geometries.push_back(third);
    CHECK(meshes.Reconcile(slotA, materials, changes));
    CHECK(meshes.GetMeshRow(slotA).geometryCount == 3 && meshes.GetMeshRow(slotA).firstGeometry == 3);
    CHECK(a->geometries[2]->globalGeometryIndex == 5);
    CHECK(meshes.GetGeometryCount() == 4);

    // materials swapped away: glass loses its last reference
    changes.Clear();
    a->geometries[1]->material = stone;
    CHECK(!meshes.Reconcile(slotA, materials, changes));
    CHECK(describe(changes) == "-t:glass ");
    CHECK(materials.size() == 1);
}

int main(int, char**)
{
    try
    {
        test_slot_allocator_basics();
        test_slot_allocator_against_model();
        test_material_table();
        test_mesh_table();
        test_mesh_table_reconcile();
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    std::printf("asset tables: ok\n");
    return 0;
}
