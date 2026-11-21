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

// Asset tables and stable slots (aggregate docs/ecs-design.md 4.8 and 7.2; plan step E4). Meshes and materials are not
// entities: each scene graph keeps them in a table whose rows hold the authoring object (the only CPU home of its
// parameters), a reference count and a stable slot. The slot is the asset's GPU index (MeshInfo::globalMeshIndex,
// MeshGeometry::globalGeometryIndex, Material::materialID) for as long as the asset is in the table, so GPU indices no
// longer change when the structure changes and no longer depend on object addresses: they follow the order in which
// assets were first referenced, and freed slots are reused lowest first.
//
// Module-private C++ (it holds implementation classes and std:: containers), like ResourceTracker before it.
//
//     MeshInstance on an entity    MeshRef { MeshId }  ->  MeshTable row: MeshInfo, references, geometry slots
//                                                              geometry i: MaterialTable row (one reference per row)
//
// Reference counts: a MeshRef column holds one reference to its mesh; a mesh row holds one reference to the material
// of each of its geometries and one to its skin prototype. So a material stays while any mesh that uses it does, as
// it stayed while any instance of such a mesh did before.

#pragma once

#include <donut/engine/SceneTypes.h>
#include <nvrhi/core/autoptr.h>

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace donut::engine
{
    // Generational handles: the slot in the low 24 bits, a generation (1-255) above it, so that 0 is never a valid id
    // and an id kept after its row was freed does not name the slot's next asset.
    enum class MeshId : uint32_t { Invalid = 0 };
    enum class MaterialId : uint32_t { Invalid = 0 };

    // Stable slots: ranges of consecutive slots, first fit from the lowest free slot, coalesced when freed. The end is
    // a high-water mark: one past the highest slot ever allocated, the size of an array indexed by slot. Allocation and
    // freeing are deterministic (they depend only on the order of the calls). A range of 0 slots allocates nothing and
    // starts at 0.
    class SlotAllocator
    {
    public:
        uint32_t Allocate(uint32_t count);
        void Free(uint32_t first, uint32_t count);
        [[nodiscard]] uint32_t GetEnd() const noexcept { return m_End; }
        [[nodiscard]] uint32_t GetUsed() const noexcept { return m_Used; }
        [[nodiscard]] uint32_t GetFreeRangeCount() const noexcept { return uint32_t(m_Free.size()); }
        void Reset() noexcept;

    private:
        struct Range
        {
            uint32_t first;
            uint32_t count;
        };
        std::vector<Range> m_Free;   // descending by `first`: the lowest free range is the last element
        uint32_t m_End = 0;
        uint32_t m_Used = 0;
    };

    // A dense table of assets with stable slots and reference counts. Iteration is in slot order over the rows in use
    // and yields T*, as ResourceTracker did (in its hash order).
    template <typename T, typename Id>
    class AssetTable
    {
    public:
        static constexpr uint32_t c_SlotBits = 24;
        static constexpr uint32_t c_MaxSlots = 1u << c_SlotBits;

        struct Row
        {
            nvrhi::AutoPtr<T> object;      // null while the slot is free
            uint32_t references = 0;
            uint32_t generation = 0;       // of the slot's current or last asset
            uint64_t serial = 0;           // the table's serial when the slot got its asset
        };

        class ConstIterator
        {
        public:
            ConstIterator(const std::vector<Row>* rows, uint32_t slot) noexcept : m_Rows(rows), m_Slot(slot) { Skip(); }
            ConstIterator& operator++() noexcept { ++m_Slot; Skip(); return *this; }
            ConstIterator operator++(int) noexcept { ConstIterator previous = *this; ++*this; return previous; }
            bool operator==(const ConstIterator& other) const noexcept { return m_Slot == other.m_Slot; }
            bool operator!=(const ConstIterator& other) const noexcept { return m_Slot != other.m_Slot; }
            T* operator*() const noexcept { return (*m_Rows)[m_Slot].object.Get(); }
            [[nodiscard]] uint32_t GetSlot() const noexcept { return m_Slot; }

        private:
            void Skip() noexcept
            {
                while (m_Slot < m_Rows->size() && !(*m_Rows)[m_Slot].object)
                    ++m_Slot;
            }
            const std::vector<Row>* m_Rows;
            uint32_t m_Slot;
        };

        [[nodiscard]] ConstIterator begin() const noexcept { return ConstIterator(&m_Rows, 0); }
        [[nodiscard]] ConstIterator end() const noexcept { return ConstIterator(&m_Rows, uint32_t(m_Rows.size())); }
        [[nodiscard]] size_t size() const noexcept { return m_Count; }
        [[nodiscard]] bool empty() const noexcept { return m_Count == 0; }

        static uint32_t SlotOf(Id id) noexcept { return uint32_t(id) & (c_MaxSlots - 1); }
        // The asset an id names; null for Invalid and for an id whose row was freed since.
        [[nodiscard]] T* Get(Id id) const noexcept
        {
            const uint32_t slot = SlotOf(id);
            if (id == Id::Invalid || slot >= m_Rows.size())
                return nullptr;
            const Row& row = m_Rows[slot];
            return row.generation == (uint32_t(id) >> c_SlotBits) ? row.object.Get() : nullptr;
        }
        // The id of an asset in the table; Invalid when it is not there.
        [[nodiscard]] Id Find(const T* object) const
        {
            auto it = m_Index.find(object);
            return it != m_Index.end() ? MakeId(it->second) : Id::Invalid;
        }
        // Rows by slot, [0, GetSlotEnd()): free rows have no object.
        [[nodiscard]] uint32_t GetSlotEnd() const noexcept { return uint32_t(m_Rows.size()); }
        [[nodiscard]] const Row& GetRow(uint32_t slot) const noexcept { return m_Rows[slot]; }
        // Counts up whenever a slot gets an asset; Row::serial says when.
        [[nodiscard]] uint64_t GetSerial() const noexcept { return m_Serial; }

    protected:
        Id MakeId(uint32_t slot) const noexcept
        {
            return Id((m_Rows[slot].generation << c_SlotBits) | slot);
        }

        // Adds a reference to `object`; `added` says whether it took a new row (its first reference).
        Id AcquireRow(T* object, bool& added)
        {
            added = false;
            if (!object)
                return Id::Invalid;
            auto it = m_Index.find(object);
            if (it != m_Index.end())
            {
                ++m_Rows[it->second].references;
                return MakeId(it->second);
            }
            const uint32_t slot = m_Slots.Allocate(1);
            assert(slot < c_MaxSlots);
            if (slot >= m_Rows.size())
                m_Rows.resize(size_t(slot) + 1);
            Row& row = m_Rows[slot];
            row.object = object;
            row.references = 1;
            row.generation = row.generation % 255u + 1u;
            row.serial = ++m_Serial;
            m_Index.emplace(object, slot);
            ++m_Count;
            added = true;
            return MakeId(slot);
        }

        // Removes a reference; returns the asset when that was its last one (its row is free now), else null.
        nvrhi::AutoPtr<T> ReleaseRow(Id id)
        {
            if (!Get(id))
            {
                assert(!"AssetTable: releasing an id that is not in the table");
                return nullptr;
            }
            const uint32_t slot = SlotOf(id);
            Row& row = m_Rows[slot];
            if (--row.references != 0)
                return nullptr;
            nvrhi::AutoPtr<T> object = std::move(row.object);
            m_Index.erase(object.Get());
            m_Slots.Free(slot, 1);
            --m_Count;
            return object;
        }

        std::vector<Row> m_Rows;
        std::unordered_map<const T*, uint32_t> m_Index;   // looked up, never iterated: no order depends on addresses
        SlotAllocator m_Slots;
        size_t m_Count = 0;
        uint64_t m_Serial = 0;
    };

    // Materials: the slot is Material::materialID, written when the material gets its row.
    class MaterialTable : public AssetTable<Material, MaterialId>
    {
    public:
        MaterialId Acquire(Material* material, bool& added);
        nvrhi::AutoPtr<Material> Release(MaterialId id) { return ReleaseRow(id); }
        // Writes every row's slot into its material again (a material shared with another graph may hold that
        // graph's slot).
        void WriteSlots() const;
    };

    // Meshes: the slot is MeshInfo::globalMeshIndex. Each mesh row also owns a range of geometry slots, one per
    // geometry, which are the geometries' globalGeometryIndex (InstanceData::firstGeometryIndex is the first), the
    // materials of its geometries and its skin prototype.
    class MeshTable : public AssetTable<MeshInfo, MeshId>
    {
    public:
        struct MeshRow
        {
            uint32_t firstGeometry = 0;
            uint32_t geometryCount = 0;
            MeshId prototype = MeshId::Invalid;       // the skin prototype's row, when the mesh has one
            std::vector<MaterialId> materials;        // per geometry; Invalid for a geometry without a material
        };

        // What Acquire, Release and Reconcile added or removed, in order: a mesh, then its materials, then its skin
        // prototype, as RegisterLeaf and UnregisterLeaf raised SceneGraph's callbacks. The events hold their objects,
        // so that a removed asset lives until its callback ran.
        struct Change
        {
            bool added;
            nvrhi::AutoPtr<MeshInfo> mesh;         // a mesh event, or
            nvrhi::AutoPtr<Material> material;     // a material event
        };
        struct Changes
        {
            std::vector<Change> events;
            size_t geometriesAdded = 0;   // geometries of the meshes added (RegisterLeaf's count for the max per mesh)
            void Clear() noexcept { events.clear(); geometriesAdded = 0; }
        };

        // Adds a reference to `mesh`. A new row gets geometry slots, references to its geometries' materials and to
        // its skin prototype (recursively), and writes the slots into the objects.
        MeshId Acquire(MeshInfo* mesh, MaterialTable& materials, Changes& changes);
        // Removes a reference; the last one frees the row, its geometry slots, and its references.
        void Release(MeshId id, MaterialTable& materials, Changes& changes);

        // Brings a row in line with its mesh after the mesh was edited in place: a changed geometry count gets new
        // geometry slots (returns true), replaced geometries get their slots written, and changed materials are
        // referenced instead of the old ones.
        bool Reconcile(uint32_t slot, MaterialTable& materials, Changes& changes);

        [[nodiscard]] const MeshRow& GetMeshRow(uint32_t slot) const noexcept { return m_MeshRows[slot]; }
        // One past the highest geometry slot in use or used before: the size of an array indexed by globalGeometryIndex.
        [[nodiscard]] uint32_t GetGeometryEnd() const noexcept { return m_Geometries.GetEnd(); }
        [[nodiscard]] uint32_t GetGeometryCount() const noexcept { return m_Geometries.GetUsed(); }

    private:
        void WriteGeometrySlots(uint32_t slot) const;
        std::vector<MeshRow> m_MeshRows;   // by slot, beside m_Rows
        SlotAllocator m_Geometries;
    };
}
