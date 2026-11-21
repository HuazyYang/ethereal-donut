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

// Asset tables and stable slots (plan step E4); see AssetTables.h.

#include <donut/engine/AssetTables.h>

#include <algorithm>

namespace donut::engine
{
    // ------------------------------------------------------------------------------------------ slots

    uint32_t SlotAllocator::Allocate(uint32_t count)
    {
        if (count == 0)
            return 0;
        m_Used += count;
        // First fit from the lowest free range, which is the last element.
        for (size_t i = m_Free.size(); i-- > 0;)
        {
            Range& range = m_Free[i];
            if (range.count < count)
                continue;
            const uint32_t first = range.first;
            range.first += count;
            range.count -= count;
            if (range.count == 0)
                m_Free.erase(m_Free.begin() + ptrdiff_t(i));
            return first;
        }
        // No free range is large enough: the highest one grows past the end when it reaches it.
        if (!m_Free.empty() && m_Free.front().first + m_Free.front().count == m_End)
        {
            const uint32_t first = m_Free.front().first;
            m_Free.erase(m_Free.begin());
            m_End = first + count;
            return first;
        }
        const uint32_t first = m_End;
        m_End += count;
        return first;
    }

    void SlotAllocator::Free(uint32_t first, uint32_t count)
    {
        if (count == 0)
            return;
        assert(first + count <= m_End && count <= m_Used);
        m_Used -= count;
        // `below` is the first range at or below `first` (descending order); the one before it lies above.
        auto below = std::lower_bound(m_Free.begin(), m_Free.end(), first,
                                      [](const Range& range, uint32_t value) { return range.first > value; });
        assert(below == m_Free.end() || below->first + below->count <= first);       // freed twice
        assert(below == m_Free.begin() || first + count <= (below - 1)->first);
        const bool joinBelow = below != m_Free.end() && below->first + below->count == first;
        const bool joinAbove = below != m_Free.begin() && first + count == (below - 1)->first;
        if (joinBelow && joinAbove)
        {
            below->count += count + (below - 1)->count;
            m_Free.erase(below - 1);
        }
        else if (joinBelow)
            below->count += count;
        else if (joinAbove)
        {
            (below - 1)->first = first;
            (below - 1)->count += count;
        }
        else
            m_Free.insert(below, Range{ first, count });
    }

    void SlotAllocator::Reset() noexcept
    {
        m_Free.clear();
        m_End = 0;
        m_Used = 0;
    }

    // -------------------------------------------------------------------------------------- materials

    MaterialId MaterialTable::Acquire(Material* material, bool& added)
    {
        const MaterialId id = AcquireRow(material, added);
        if (added)
            material->materialID = int(SlotOf(id));
        return id;
    }

    void MaterialTable::WriteSlots() const
    {
        for (uint32_t slot = 0; slot < uint32_t(m_Rows.size()); ++slot)
        {
            if (Material* material = m_Rows[slot].object)
                material->materialID = int(slot);
        }
    }

    // ----------------------------------------------------------------------------------------- meshes

    namespace
    {
        // Writes the mesh's and its geometries' slots; true when any of them changed.
        bool WriteSlots(MeshInfo* mesh, uint32_t slot, uint32_t firstGeometry)
        {
            bool changed = mesh->globalMeshIndex != int(slot);
            mesh->globalMeshIndex = int(slot);
            for (size_t i = 0; i < mesh->geometries.size(); ++i)
            {
                MeshGeometry* geometry = mesh->geometries[i];
                if (!geometry)
                    continue;
                const int index = int(firstGeometry + uint32_t(i));
                changed |= geometry->globalGeometryIndex != index;
                geometry->globalGeometryIndex = index;
            }
            return changed;
        }
    }

    void MeshTable::WriteGeometrySlots(uint32_t slot) const
    {
        (void)WriteSlots(m_Rows[slot].object, slot, m_MeshRows[slot].firstGeometry);
    }

    MeshId MeshTable::Acquire(MeshInfo* mesh, MaterialTable& materials, Changes& changes)
    {
        bool added = false;
        const MeshId id = AcquireRow(mesh, added);
        if (!added)
            return id;

        const uint32_t slot = SlotOf(id);
        if (slot >= m_MeshRows.size())
            m_MeshRows.resize(size_t(slot) + 1);
        const uint32_t count = uint32_t(mesh->geometries.size());
        {
            MeshRow& row = m_MeshRows[slot];
            row.geometryCount = count;
            row.firstGeometry = m_Geometries.Allocate(count);
            row.prototype = MeshId::Invalid;
            row.materials.assign(count, MaterialId::Invalid);
        }
        WriteGeometrySlots(slot);
        changes.events.push_back(Change{ true, mesh, nullptr });
        changes.geometriesAdded += count;

        for (uint32_t i = 0; i < count; ++i)
        {
            Material* material = mesh->geometries[i] ? mesh->geometries[i]->material.Get() : nullptr;
            if (!material)
                continue;
            bool materialAdded = false;
            m_MeshRows[slot].materials[i] = materials.Acquire(material, materialAdded);
            if (materialAdded)
                changes.events.push_back(Change{ true, nullptr, material });
        }

        if (mesh->skinPrototype)
        {
            const MeshId prototype = Acquire(mesh->skinPrototype, materials, changes);   // may grow m_MeshRows
            m_MeshRows[slot].prototype = prototype;
        }
        return id;
    }

    void MeshTable::Release(MeshId id, MaterialTable& materials, Changes& changes)
    {
        if (id == MeshId::Invalid)
            return;
        const uint32_t slot = SlotOf(id);
        nvrhi::AutoPtr<MeshInfo> mesh = ReleaseRow(id);
        if (!mesh)
            return;

        MeshRow& row = m_MeshRows[slot];
        m_Geometries.Free(row.firstGeometry, row.geometryCount);
        changes.events.push_back(Change{ false, std::move(mesh), nullptr });
        for (MaterialId material : row.materials)
        {
            if (material == MaterialId::Invalid)
                continue;
            nvrhi::AutoPtr<Material> removed = materials.Release(material);
            if (removed)
                changes.events.push_back(Change{ false, nullptr, std::move(removed) });
        }
        const MeshId prototype = row.prototype;
        row.firstGeometry = 0;
        row.geometryCount = 0;
        row.prototype = MeshId::Invalid;
        row.materials.clear();
        Release(prototype, materials, changes);
    }

    bool MeshTable::Reconcile(uint32_t slot, MaterialTable& materials, Changes& changes)
    {
        MeshInfo* mesh = m_Rows[slot].object;
        if (!mesh)
            return false;
        bool changed = false;
        bool reranged = false;

        const uint32_t count = uint32_t(mesh->geometries.size());
        if (count != m_MeshRows[slot].geometryCount)
        {
            MeshRow& row = m_MeshRows[slot];
            m_Geometries.Free(row.firstGeometry, row.geometryCount);
            row.firstGeometry = m_Geometries.Allocate(count);
            row.geometryCount = count;
            reranged = true;
            changed = true;
        }
        changed |= WriteSlots(mesh, slot, m_MeshRows[slot].firstGeometry);

        // Materials: the new ones are referenced before the old ones are released, so that one that stays is never
        // removed in between.
        bool sameMaterials = m_MeshRows[slot].materials.size() == count;
        for (uint32_t i = 0; sameMaterials && i < count; ++i)
        {
            Material* material = mesh->geometries[i] ? mesh->geometries[i]->material.Get() : nullptr;
            sameMaterials = materials.Get(m_MeshRows[slot].materials[i]) == material;
        }
        if (!sameMaterials)
        {
            std::vector<MaterialId> previous = std::move(m_MeshRows[slot].materials);
            std::vector<MaterialId> current(count, MaterialId::Invalid);
            for (uint32_t i = 0; i < count; ++i)
            {
                Material* material = mesh->geometries[i] ? mesh->geometries[i]->material.Get() : nullptr;
                if (!material)
                    continue;
                bool materialAdded = false;
                current[i] = materials.Acquire(material, materialAdded);
                if (materialAdded)
                    changes.events.push_back(Change{ true, nullptr, material });
            }
            for (MaterialId material : previous)
            {
                if (material == MaterialId::Invalid)
                    continue;
                nvrhi::AutoPtr<Material> removed = materials.Release(material);
                if (removed)
                    changes.events.push_back(Change{ false, nullptr, std::move(removed) });
            }
            m_MeshRows[slot].materials = std::move(current);
            changed = true;
        }

        if (Get(m_MeshRows[slot].prototype) != mesh->skinPrototype.Get())
        {
            const MeshId previous = m_MeshRows[slot].prototype;
            MeshId current = MeshId::Invalid;
            if (mesh->skinPrototype)
                current = Acquire(mesh->skinPrototype, materials, changes);
            m_MeshRows[slot].prototype = current;
            Release(previous, materials, changes);
            changed = true;
        }

        if (changed)
            m_Rows[slot].serial = ++m_Serial;
        return reranged;
    }
}
