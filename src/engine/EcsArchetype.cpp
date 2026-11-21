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

// Archetypes, chunks, rows, dynamic buffers and sparse sets (aggregate docs/ecs-design.md 4.3, 4.4).

#include "EcsWorldImpl.h"

#include <new>

namespace donut::engine::ecs
{
    namespace
    {
        constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment) noexcept
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        BufferHeader& HeaderAt(uint8_t* chunkMemory, const ArchetypeColumn& column, uint32_t row) noexcept
        {
            return reinterpret_cast<BufferHeader*>(chunkMemory + column.offset)[row];
        }

        uint8_t* InlineAt(uint8_t* chunkMemory, const ArchetypeColumn& column, uint32_t row) noexcept
        {
            return chunkMemory + column.inlineOffset + size_t(row) * column.inlineStride;
        }

        uint8_t* ValueAt(uint8_t* chunkMemory, const ArchetypeColumn& column, uint32_t row) noexcept
        {
            return chunkMemory + column.offset + size_t(row) * column.stride;
        }

        uint64_t* MaskAt(uint8_t* chunkMemory, const ArchetypeColumn& column) noexcept
        {
            return reinterpret_cast<uint64_t*>(chunkMemory + column.enabledOffset);
        }

        void AddEdge(Vector<ArchetypeEdge>& edges, ComponentId component, uint32_t target) noexcept
        {
            for (const ArchetypeEdge& edge : edges)
                if (edge.component == component)
                    return;
            edges.push_back(ArchetypeEdge{ component, target });
        }

        void AddHeapBytes(std::atomic<uint64_t>& counter, uint64_t bytes, bool add) noexcept
        {
            if (add)
                counter.fetch_add(bytes, std::memory_order_relaxed);
            else
                counter.fetch_sub(bytes, std::memory_order_relaxed);
        }
    }

    // --------------------------------------------------------------------------------------------- values

    void ConstructValues(const ComponentType& type, void* dst, uint32_t count) noexcept
    {
        if (count == 0 || type.desc.size == 0)
            return;
        if (type.desc.construct)
            type.desc.construct(dst, count);
        else
            std::memset(dst, 0, size_t(type.desc.size) * count);
    }

    void DestructValues(const ComponentType& type, void* dst, uint32_t count) noexcept
    {
        if (count != 0 && type.desc.size != 0 && type.desc.destruct)
            type.desc.destruct(dst, count);
    }

    void RelocateValues(const ComponentType& type, void* dst, void* src, uint32_t count) noexcept
    {
        if (count == 0 || type.desc.size == 0)
            return;
        if (type.desc.relocate)
            type.desc.relocate(dst, src, count);
        else
            std::memcpy(dst, src, size_t(type.desc.size) * count);
    }

    void CopyValues(const ComponentType& type, void* dst, const void* src, uint32_t count) noexcept
    {
        if (count == 0 || type.desc.size == 0)
            return;
        if (type.desc.copy)
            type.desc.copy(dst, src, count);
        else
            std::memcpy(dst, src, size_t(type.desc.size) * count);
    }

    void TransferValues(const ComponentType& type, void* dst, const void* src, uint32_t count,
                        ValueTransfer transfer) noexcept
    {
        if (transfer == ValueTransfer::Relocate)
            RelocateValues(type, dst, const_cast<void*>(src), count);
        else
            CopyValues(type, dst, src, count);
    }

    bool FieldBytesEqual(const ComponentType& type, const void* a, const void* b) noexcept
    {
        const uint8_t* x = static_cast<const uint8_t*>(a);
        const uint8_t* y = static_cast<const uint8_t*>(b);
        for (const ByteRange& range : type.fieldBytes)
            if (std::memcmp(x + range.offset, y + range.offset, range.size) != 0)
                return false;
        return true;
    }

    uint64_t HashFieldBytes(const ComponentType& type, const void* value, uint64_t hash) noexcept
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(value);
        for (const ByteRange& range : type.fieldBytes)
            hash = HashBytes(bytes + range.offset, range.size, hash);
        return hash;
    }

    // ------------------------------------------------------------------------------------------ archetype

    uint32_t Archetype::FindColumn(ComponentId component) const noexcept
    {
        uint32_t lo = 0;
        uint32_t hi = uint32_t(components.size());
        while (lo < hi)
        {
            const uint32_t mid = (lo + hi) / 2;
            if (components[mid] < component)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo < components.size() && components[lo] == component ? lo : c_NoColumn;
    }

    void* Archetype::ColumnData(uint32_t chunk, uint32_t column) const noexcept
    {
        const ArchetypeColumn& c = columns[column];
        switch (c.kind)
        {
        case StorageKind::Data:
        case StorageKind::Buffer:
        case StorageKind::Chunk:
            return chunks[chunk].memory + c.offset;
        case StorageKind::Shared:
            return sharedValues + c.offset;
        default:
            return nullptr;
        }
    }

    uint64_t* Archetype::EnabledMask(uint32_t chunk, uint32_t column) const noexcept
    {
        const ArchetypeColumn& c = columns[column];
        return c.enableable ? MaskAt(chunks[chunk].memory, c) : nullptr;
    }

    uint32_t WorldImpl::LayoutChunk(Archetype& archetype, uint32_t rows) noexcept
    {
        uint64_t offset = uint64_t(rows) * sizeof(Entity);
        for (ArchetypeColumn& column : archetype.columns)
        {
            if (column.kind == StorageKind::Data || column.kind == StorageKind::Buffer)
            {
                offset = AlignUp(offset, c_ColumnAlignment);
                column.offset = uint32_t(offset);
                offset += uint64_t(rows) * column.stride;
            }
        }
        for (ArchetypeColumn& column : archetype.columns)
        {
            if (column.kind == StorageKind::Buffer)
            {
                offset = AlignUp(offset, c_ColumnAlignment);
                column.inlineOffset = uint32_t(offset);
                offset += uint64_t(rows) * column.inlineStride;
            }
        }
        offset = AlignUp(offset, sizeof(uint64_t));
        const uint32_t maskWords = (rows + 63) / 64;
        for (ArchetypeColumn& column : archetype.columns)
        {
            if (column.enableable)
            {
                column.enabledOffset = uint32_t(offset);
                offset += uint64_t(maskWords) * sizeof(uint64_t);
            }
        }
        archetype.versionsOffset = uint32_t(offset);
        offset += (uint64_t(archetype.columns.size()) + 1) * sizeof(uint64_t);
        for (ArchetypeColumn& column : archetype.columns)
        {
            if (column.kind == StorageKind::Chunk)
            {
                offset = AlignUp(offset, column.type->desc.alignment);
                column.offset = uint32_t(offset);
                offset += column.stride;
            }
        }
        archetype.maskWords = maskWords;
        return offset > 0xffffffffull ? 0xffffffffu : uint32_t(offset);
    }

    void WorldImpl::DescribeColumns(Archetype& archetype, const ComponentId* components, uint32_t count) noexcept
    {
        // Sizes in 64 bits: registration bounds each component by the largest chunk, but a row adds them up.
        uint64_t rowSize = sizeof(Entity);
        uint64_t sharedSize = 0;
        archetype.columns.clear();
        archetype.destructRows = false;
        for (uint32_t i = 0; i < count; ++i)
        {
            const ComponentType& type = *m_Types[uint32_t(components[i])];
            ArchetypeColumn column{};
            column.type = &type;
            column.component = components[i];
            column.kind = type.kind;
            column.enableable = type.enableable;
            switch (type.kind)
            {
            case StorageKind::Data:
                column.stride = type.desc.size;
                rowSize += type.desc.size;
                if (type.desc.destruct)
                    archetype.destructRows = true;
                break;
            case StorageKind::Buffer:
                column.stride = sizeof(BufferHeader);
                column.inlineStride = uint32_t(uint64_t(type.desc.size) * type.desc.bufferInlineCapacity);
                rowSize += column.stride + uint64_t(column.inlineStride);
                archetype.destructRows = true;
                break;
            case StorageKind::Shared:
                column.stride = type.desc.size;
                sharedSize = AlignUp(sharedSize, type.desc.alignment);
                column.offset = uint32_t(sharedSize);
                sharedSize += type.desc.size;
                break;
            case StorageKind::Chunk:
                column.stride = type.desc.size;
                break;
            default:
                break;
            }
            archetype.columns.push_back(column);
        }
        archetype.rowSize = rowSize > 0xffffffffull ? 0xffffffffu : uint32_t(rowSize);
        archetype.sharedSize = sharedSize > 0xffffffffull ? 0xffffffffu : uint32_t(sharedSize);
    }

    void WorldImpl::PlanChunks(Archetype& archetype) const noexcept
    {
        // D6: rows over the threshold get the large chunk, so that a chunk keeps about 128 rows or more.
        auto capacity = [&archetype](uint32_t chunkSize) noexcept
        {
            uint32_t lo = 0;
            uint32_t hi = c_MaxRowsPerChunk;
            while (lo < hi)
            {
                const uint32_t mid = lo + (hi - lo + 1) / 2;
                if (LayoutChunk(archetype, mid) <= chunkSize)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            return lo;
        };
        uint32_t chunkSize = archetype.rowSize > m_Desc.largeRowThreshold ? m_Desc.largeChunkSize : m_Desc.chunkSize;
        uint32_t rows = capacity(chunkSize);
        if (rows == 0 && chunkSize < m_Desc.largeChunkSize)
        {
            chunkSize = m_Desc.largeChunkSize;
            rows = capacity(chunkSize);
        }
        if (rows != 0)
            LayoutChunk(archetype, rows);
        archetype.chunkSize = chunkSize;
        archetype.rowsPerChunk = rows;
    }

    bool WorldImpl::RowFits(const ComponentId* components, uint32_t count) noexcept
    {
        DescribeColumns(m_ScratchArchetype, components, count);
        PlanChunks(m_ScratchArchetype);
        return m_ScratchArchetype.rowsPerChunk != 0;
    }

    uint32_t WorldImpl::CreateArchetype(const ComponentId* components, uint32_t count, const void* const* shared,
                                        uint64_t hash) noexcept
    {
        // The layout first: a row that fits no chunk needs no memory.
        if (!RowFits(components, count))
            return c_ArchetypeTooLarge;

        void* memory = HeapAllocate(sizeof(Archetype), alignof(Archetype));
        if (!memory)
            return c_ArchetypeOutOfMemory;
        Archetype* archetype = new (memory) Archetype();
        archetype->index = uint32_t(m_Archetypes.size());
        archetype->hash = hash;
        archetype->components.assign(components, components + count);
        archetype->columns.reserve(count);
        DescribeColumns(*archetype, components, count);
        PlanChunks(*archetype);
        NVRHI_ASSERT(archetype->rowsPerChunk != 0);
        const uint32_t sharedSize = archetype->sharedSize;

        if (sharedSize != 0)
        {
            archetype->sharedValues = static_cast<uint8_t*>(HeapAllocate(size_t(sharedSize), c_ColumnAlignment));
            if (!archetype->sharedValues)
            {
                archetype->~Archetype();
                HeapFree(archetype);
                return c_ArchetypeOutOfMemory;
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                const ArchetypeColumn& column = archetype->columns[i];
                if (column.kind == StorageKind::Shared)
                    CopyValues(*column.type, archetype->sharedValues + column.offset, shared[i], 1);
            }
        }

        m_Archetypes.push_back(archetype);
        m_ArchetypeIndex.Insert(hash, archetype->index);
        ++m_StructuralVersion;
        return archetype->index;
    }

    uint32_t WorldImpl::FindOrCreateArchetype(const ComponentId* components, uint32_t count,
                                              const void* const* shared) noexcept
    {
        uint64_t hash = c_HashSeed;
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t id = uint32_t(components[i]);
            hash = HashBytes(&id, sizeof(id), hash);
            const ComponentType& type = *m_Types[id];
            if (type.kind == StorageKind::Shared)
                hash = HashFieldBytes(type, shared[i], hash);
        }
        ++m_Counters.archetypeLookups;
        const uint32_t found = m_ArchetypeIndex.Find(hash, [&](uint32_t index) noexcept
        {
            const Archetype& archetype = *m_Archetypes[index];
            if (archetype.components.size() != count)
                return false;
            for (uint32_t i = 0; i < count; ++i)
                if (archetype.components[i] != components[i])
                    return false;
            for (uint32_t i = 0; i < count; ++i)
            {
                const ArchetypeColumn& column = archetype.columns[i];
                if (column.kind == StorageKind::Shared &&
                    !FieldBytesEqual(*column.type, archetype.sharedValues + column.offset, shared[i]))
                    return false;
            }
            return true;
        });
        if (found != HashIndex::c_NotFound)
            return found;
        return CreateArchetype(components, count, shared, hash);
    }

    uint32_t WorldImpl::GetEdge(const Vector<ArchetypeEdge>& edges, ComponentId component) noexcept
    {
        for (const ArchetypeEdge& edge : edges)
        {
            if (edge.component == component)
            {
                ++m_Counters.edgeHits;
                return edge.archetype;
            }
        }
        return c_NoArchetype;
    }

    uint32_t WorldImpl::GetAddTarget(uint32_t archetype, ComponentId component) noexcept
    {
        Archetype& from = *m_Archetypes[archetype];
        uint32_t target = GetEdge(from.addEdges, component);
        if (target != c_NoArchetype)
            return target;

        m_ScratchIds.clear();
        m_ScratchShared.clear();
        bool inserted = false;
        for (uint32_t i = 0; i < from.components.size(); ++i)
        {
            if (!inserted && component < from.components[i])
            {
                m_ScratchIds.push_back(component);
                m_ScratchShared.push_back(nullptr);
                inserted = true;
            }
            const ArchetypeColumn& column = from.columns[i];
            m_ScratchIds.push_back(from.components[i]);
            m_ScratchShared.push_back(column.kind == StorageKind::Shared ? from.sharedValues + column.offset : nullptr);
        }
        if (!inserted)
        {
            m_ScratchIds.push_back(component);
            m_ScratchShared.push_back(nullptr);
        }
        target = FindOrCreateArchetype(m_ScratchIds.data(), uint32_t(m_ScratchIds.size()), m_ScratchShared.data());
        if (IsArchetypeError(target))
            return target;
        AddEdge(from.addEdges, component, target);
        AddEdge(m_Archetypes[target]->removeEdges, component, archetype);
        return target;
    }

    uint32_t WorldImpl::GetSharedTarget(uint32_t archetype, ComponentId component, const void* value) noexcept
    {
        // No cache: the value decides the target (4.4).
        const Archetype& from = *m_Archetypes[archetype];
        m_ScratchIds.clear();
        m_ScratchShared.clear();
        bool placed = false;
        for (uint32_t i = 0; i < from.components.size(); ++i)
        {
            const ArchetypeColumn& column = from.columns[i];
            if (!placed && component <= from.components[i])
            {
                m_ScratchIds.push_back(component);
                m_ScratchShared.push_back(value);
                placed = true;
                if (component == from.components[i])
                    continue;
            }
            m_ScratchIds.push_back(from.components[i]);
            m_ScratchShared.push_back(column.kind == StorageKind::Shared ? from.sharedValues + column.offset : nullptr);
        }
        if (!placed)
        {
            m_ScratchIds.push_back(component);
            m_ScratchShared.push_back(value);
        }
        return FindOrCreateArchetype(m_ScratchIds.data(), uint32_t(m_ScratchIds.size()), m_ScratchShared.data());
    }

    uint32_t WorldImpl::GetRemoveTarget(uint32_t archetype, ComponentId component) noexcept
    {
        Archetype& from = *m_Archetypes[archetype];
        uint32_t target = GetEdge(from.removeEdges, component);
        if (target != c_NoArchetype)
            return target;

        m_ScratchIds.clear();
        m_ScratchShared.clear();
        for (uint32_t i = 0; i < from.components.size(); ++i)
        {
            if (from.components[i] == component)
                continue;
            const ArchetypeColumn& column = from.columns[i];
            m_ScratchIds.push_back(from.components[i]);
            m_ScratchShared.push_back(column.kind == StorageKind::Shared ? from.sharedValues + column.offset : nullptr);
        }
        target = FindOrCreateArchetype(m_ScratchIds.data(), uint32_t(m_ScratchIds.size()), m_ScratchShared.data());
        if (IsArchetypeError(target))
            return target;
        AddEdge(from.removeEdges, component, target);
        if (m_Types[uint32_t(component)]->kind != StorageKind::Shared)
            AddEdge(m_Archetypes[target]->addEdges, component, archetype);
        return target;
    }

    // ---------------------------------------------------------------------------------------------- chunks

    PagePool& WorldImpl::PoolFor(uint32_t chunkSize) noexcept
    {
        return chunkSize == m_SmallPages.GetPageSize() ? m_SmallPages : m_LargePages;
    }

    bool WorldImpl::AppendChunk(Archetype& archetype, uint64_t version) noexcept
    {
        uint8_t* memory = static_cast<uint8_t*>(PoolFor(archetype.chunkSize).Allocate());
        if (!memory)
            return false;
        for (const ArchetypeColumn& column : archetype.columns)
        {
            if (column.enableable)
                std::memset(memory + column.enabledOffset, 0, size_t(archetype.maskWords) * sizeof(uint64_t));
            if (column.kind == StorageKind::Chunk)
                ConstructValues(*column.type, memory + column.offset, 1);
        }
        uint64_t* versions = reinterpret_cast<uint64_t*>(memory + archetype.versionsOffset);
        for (uint32_t i = 0; i <= archetype.columns.size(); ++i)
            versions[i] = version;
        archetype.chunks.push_back(ChunkRecord{ memory, 0, 0 });
        return true;
    }

    bool WorldImpl::ReserveRows(Archetype& archetype, uint32_t count, uint64_t version) noexcept
    {
        const uint32_t rows = archetype.rowsPerChunk;
        const uint64_t needed = (uint64_t(archetype.entityCount) + count + rows - 1) / rows;
        const uint32_t existing = archetype.GetChunkCount();
        for (uint64_t c = existing; c < needed; ++c)
        {
            if (!AppendChunk(archetype, version))
            {
                while (archetype.GetChunkCount() > existing)
                    ReleaseLastChunk(archetype);
                return false;
            }
        }
        return true;
    }

    void WorldImpl::ReleaseLastChunk(Archetype& archetype) noexcept
    {
        const ChunkRecord& chunk = archetype.chunks.back();
        NVRHI_ASSERT(chunk.count == 0);
        for (const ArchetypeColumn& column : archetype.columns)
            if (column.kind == StorageKind::Chunk)
                DestructValues(*column.type, chunk.memory + column.offset, 1);
        PoolFor(archetype.chunkSize).Free(chunk.memory);
        archetype.chunks.pop_back();
    }

    void WorldImpl::StampChunk(const Archetype& archetype, uint32_t chunk, uint64_t version) noexcept
    {
        uint64_t* versions = archetype.Versions(chunk);
        for (uint32_t i = 0; i <= archetype.columns.size(); ++i)
            StampVersion(versions + i, version);
    }

    // ------------------------------------------------------------------------------------------------ rows

    void WorldImpl::RemoveRow(Archetype& archetype, uint32_t chunk, uint32_t row, uint64_t version) noexcept
    {
        NVRHI_ASSERT(archetype.entityCount > 0);
        const uint32_t last = archetype.entityCount - 1;
        const uint32_t lastChunk = last / archetype.rowsPerChunk;
        const uint32_t lastRow = last % archetype.rowsPerChunk;
        if (chunk != lastChunk || row != lastRow)
        {
            MoveRowWithin(archetype, lastChunk, lastRow, chunk, row);
            StampChunk(archetype, chunk, version);
        }
        --archetype.chunks[lastChunk].count;
        --archetype.entityCount;
        if (archetype.chunks[lastChunk].count == 0)
            ReleaseLastChunk(archetype);
        ++m_StructuralVersion;
    }

    void WorldImpl::MoveRowWithin(Archetype& archetype, uint32_t srcChunk, uint32_t srcRow, uint32_t dstChunk,
                                  uint32_t dstRow) noexcept
    {
        uint8_t* src = archetype.chunks[srcChunk].memory;
        uint8_t* dst = archetype.chunks[dstChunk].memory;
        const Entity entity = reinterpret_cast<Entity*>(src)[srcRow];
        reinterpret_cast<Entity*>(dst)[dstRow] = entity;
        for (const ArchetypeColumn& column : archetype.columns)
        {
            if (column.kind == StorageKind::Data)
                RelocateValues(*column.type, ValueAt(dst, column, dstRow), ValueAt(src, column, srcRow), 1);
            else if (column.kind == StorageKind::Buffer)
                RelocateBuffer(column, HeaderAt(dst, column, dstRow), InlineAt(dst, column, dstRow),
                               HeaderAt(src, column, srcRow), InlineAt(src, column, srcRow));
            if (column.enableable)
                SetMaskBit(MaskAt(dst, column), dstRow, GetMaskBit(MaskAt(src, column), srcRow));
        }
        EntityRecord& record = m_Entities[GetEntityIndex(entity)];
        record.chunk = dstChunk;
        record.row = uint16_t(dstRow);
        if (record.sparseCount != 0)
            MoveSparsePresence(entity, archetype.index, srcChunk, srcRow, archetype.index, dstChunk, dstRow);
    }

    void WorldImpl::InitializeColumn(Archetype& archetype, uint32_t column, uint32_t chunk, uint32_t firstRow,
                                     uint32_t count, const void* values, ValueTransfer transfer) noexcept
    {
        const ArchetypeColumn& c = archetype.columns[column];
        uint8_t* memory = archetype.chunks[chunk].memory;
        if (c.kind == StorageKind::Data)
        {
            if (values)
                TransferValues(*c.type, ValueAt(memory, c, firstRow), values, count, transfer);
            else
                ConstructValues(*c.type, ValueAt(memory, c, firstRow), count);
        }
        else if (c.kind == StorageKind::Buffer)
        {
            for (uint32_t r = firstRow; r < firstRow + count; ++r)
                InitializeBuffer(HeaderAt(memory, c, r), c, InlineAt(memory, c, r));
        }
        if (c.enableable)
            SetMaskRange(MaskAt(memory, c), firstRow, count, true);
    }

    void WorldImpl::DestructRow(Archetype& archetype, uint32_t chunk, uint32_t row) noexcept
    {
        if (!archetype.destructRows)
            return;
        uint8_t* memory = archetype.chunks[chunk].memory;
        for (const ArchetypeColumn& column : archetype.columns)
        {
            if (column.kind == StorageKind::Data)
                DestructValues(*column.type, ValueAt(memory, column, row), 1);
            else if (column.kind == StorageKind::Buffer)
                DestroyBuffer(column, HeaderAt(memory, column, row), InlineAt(memory, column, row));
        }
    }

    void WorldImpl::OverwriteValue(void* dst, const ComponentType& type, const void* value,
                                   ValueTransfer transfer) noexcept
    {
        if (dst == value)
            return;
        DestructValues(type, dst, 1);
        TransferValues(type, dst, value, 1, transfer);
    }

    nvrhi::FRESULT WorldImpl::MoveEntity(Entity entity, EntityRecord& record, uint32_t target, const NewValue& newValue,
                                         uint64_t version) noexcept
    {
        Archetype& from = *m_Archetypes[record.archetype];
        Archetype& to = *m_Archetypes[target];
        const uint32_t srcChunk = record.chunk;
        const uint32_t srcRow = record.row;

        // The target's chunk is secured first, so that nothing is reported for a move that cannot happen; the report
        // comes before any row changes, and the append below cannot fail. So are the presence masks of the entity's
        // Sparse components in its new chunk, and of the row that the swap-remove moves into the hole it leaves.
        if (!SecureRowPresence(entity, target, to.entityCount / to.rowsPerChunk) || !SecureHoleFill(from, srcChunk))
            return nvrhi::FE_OUT_OF_MEMORY;
        if (!ReserveRows(to, 1, version))
            return nvrhi::FE_OUT_OF_MEMORY;
        ObserveRowsRemoving(from.index, srcChunk, srcRow, 1, to.index);

        uint32_t dstChunk = 0;
        uint32_t dstRow = 0;
        const nvrhi::FRESULT result = AppendRows(to, 1, version, [&](uint32_t chunk, uint32_t row, uint32_t, uint32_t)
        {
            dstChunk = chunk;
            dstRow = row;
        });
        NVRHI_ASSERT(NVRHI_SUCCEEDED(result));
        (void)result;

        uint8_t* src = from.chunks[srcChunk].memory;
        uint8_t* dst = to.chunks[dstChunk].memory;
        reinterpret_cast<Entity*>(dst)[dstRow] = entity;

        // Both component lists are sorted: one merge walk pairs the columns.
        const uint32_t fromCount = uint32_t(from.columns.size());
        const uint32_t toCount = uint32_t(to.columns.size());
        uint32_t i = 0;
        uint32_t j = 0;
        while (i < fromCount || j < toCount)
        {
            if (j == toCount || (i < fromCount && from.components[i] < to.components[j]))
            {
                // Only the old archetype has it: destroyed.
                const ArchetypeColumn& column = from.columns[i];
                if (column.kind == StorageKind::Data)
                    DestructValues(*column.type, ValueAt(src, column, srcRow), 1);
                else if (column.kind == StorageKind::Buffer)
                    DestroyBuffer(column, HeaderAt(src, column, srcRow), InlineAt(src, column, srcRow));
                ++i;
            }
            else if (i == fromCount || to.components[j] < from.components[i])
            {
                // Only the new archetype has it: constructed, or the new value.
                const void* value = to.components[j] == newValue.component ? newValue.value : nullptr;
                InitializeColumn(to, j, dstChunk, dstRow, 1, value, newValue.transfer);
                ++j;
            }
            else
            {
                const ArchetypeColumn& s = from.columns[i];
                const ArchetypeColumn& d = to.columns[j];
                if (d.kind == StorageKind::Data)
                    RelocateValues(*d.type, ValueAt(dst, d, dstRow), ValueAt(src, s, srcRow), 1);
                else if (d.kind == StorageKind::Buffer)
                    RelocateBuffer(d, HeaderAt(dst, d, dstRow), InlineAt(dst, d, dstRow), HeaderAt(src, s, srcRow),
                                   InlineAt(src, s, srcRow));
                if (d.enableable)
                    SetMaskBit(MaskAt(dst, d), dstRow, GetMaskBit(MaskAt(src, s), srcRow));
                ++i;
                ++j;
            }
        }

        record.archetype = target;
        record.chunk = dstChunk;
        record.row = uint16_t(dstRow);
        if (record.sparseCount != 0)
            MoveSparsePresence(entity, from.index, srcChunk, srcRow, target, dstChunk, dstRow);
        RemoveRow(from, srcChunk, srcRow, version);
        // The swap-remove changed only the old archetype: the entity's new row stands.
        ObserveRowsAdded(target, dstChunk, dstRow, 1, from.index);
        return nvrhi::FS_OK;
    }

    // ------------------------------------------------------------------------------------- dynamic buffers

    void WorldImpl::InitializeBuffer(BufferHeader& header, const ArchetypeColumn& column, uint8_t* inlineData) noexcept
    {
        header.data = column.inlineStride != 0 ? inlineData : nullptr;
        header.count = 0;
        header.capacity = column.type->desc.bufferInlineCapacity;
    }

    void WorldImpl::RelocateBuffer(const ArchetypeColumn& column, BufferHeader& dst, uint8_t* dstInline,
                                   BufferHeader& src, uint8_t* srcInline) noexcept
    {
        if (column.inlineStride != 0 && src.data == srcInline)
        {
            // Inline elements move with the row; the header follows them.
            RelocateValues(*column.type, dstInline, srcInline, src.count);
            dst.data = dstInline;
            dst.count = src.count;
            dst.capacity = src.capacity;
        }
        else
        {
            dst = src;
        }
    }

    void WorldImpl::DestroyBuffer(const ArchetypeColumn& column, BufferHeader& header, uint8_t* inlineData) noexcept
    {
        DestructValues(*column.type, header.data, header.count);
        if (header.data && !(column.inlineStride != 0 && header.data == inlineData))
        {
            HeapFree(header.data);
            AddHeapBytes(m_BufferHeapBytes, uint64_t(header.capacity) * column.type->desc.size, false);
        }
        header.data = nullptr;
        header.count = 0;
        header.capacity = 0;
    }

    bool WorldImpl::CopyBuffer(const ArchetypeColumn& column, BufferHeader& dst, uint8_t* dstInline,
                               const BufferHeader& src) noexcept
    {
        const ComponentType& type = *column.type;
        InitializeBuffer(dst, column, dstInline);
        if (src.count <= type.desc.bufferInlineCapacity)
        {
            CopyValues(type, dst.data, src.data, src.count);
            dst.count = src.count;
            return true;
        }
        void* memory = HeapAllocate(size_t(src.count) * type.desc.size, type.desc.alignment);
        if (!memory)
            return false;
        CopyValues(type, memory, src.data, src.count);
        AddHeapBytes(m_BufferHeapBytes, uint64_t(src.count) * type.desc.size, true);
        dst.data = memory;
        dst.count = src.count;
        dst.capacity = src.count;
        return true;
    }

    nvrhi::FRESULT WorldImpl::ResizeBufferRow(const ArchetypeColumn& column, BufferHeader& header, uint8_t* inlineData,
                                              uint32_t count) noexcept
    {
        const ComponentType& type = *column.type;
        const uint32_t size = type.desc.size;
        const uint32_t inlineCapacity = type.desc.bufferInlineCapacity;
        const bool onHeap = header.data && !(column.inlineStride != 0 && header.data == inlineData);

        if (count > header.capacity)
        {
            // Spill to the heap (or grow there), at least doubling.
            uint64_t capacity = uint64_t(header.capacity) * 2;
            if (capacity < count)
                capacity = count;
            if (capacity > 0xffffffffull)
                capacity = count;
            void* memory = HeapAllocate(size_t(capacity) * size, type.desc.alignment);
            if (!memory)
                return nvrhi::FE_OUT_OF_MEMORY;
            RelocateValues(type, memory, header.data, header.count);
            if (onHeap)
            {
                HeapFree(header.data);
                AddHeapBytes(m_BufferHeapBytes, uint64_t(header.capacity) * size, false);
            }
            AddHeapBytes(m_BufferHeapBytes, capacity * size, true);
            header.data = memory;
            header.capacity = uint32_t(capacity);
            ConstructValues(type, static_cast<uint8_t*>(memory) + size_t(header.count) * size, count - header.count);
            header.count = count;
            return nvrhi::FS_OK;
        }

        if (count < header.count)
        {
            DestructValues(type, static_cast<uint8_t*>(header.data) + size_t(count) * size, header.count - count);
            header.count = count;
            if (onHeap && count <= inlineCapacity)
            {
                // Back into the row: a buffer is on the heap exactly while it holds more than its inline capacity.
                uint8_t* target = column.inlineStride != 0 ? inlineData : nullptr;
                RelocateValues(type, target, header.data, count);
                HeapFree(header.data);
                AddHeapBytes(m_BufferHeapBytes, uint64_t(header.capacity) * size, false);
                header.data = target;
                header.capacity = inlineCapacity;
            }
            return nvrhi::FS_OK;
        }

        ConstructValues(type, static_cast<uint8_t*>(header.data) + size_t(header.count) * size, count - header.count);
        header.count = count;
        return nvrhi::FS_OK;
    }

    // ---------------------------------------------------------------------------------------- sparse sets

    void* WorldImpl::SparseInsert(SparseSet& set, Entity entity, EntityRecord& record) noexcept
    {
        // The row's presence mask first: running out of memory then changes nothing.
        if (!SecurePresence(set, record.archetype, record.chunk))
            return nullptr;
        const uint32_t index = GetEntityIndex(entity);
        const uint32_t page = index >> c_SparsePageBits;
        if (page >= set.pages.size())
            set.pages.resize(page + 1, nullptr);
        if (!set.pages[page])
        {
            uint32_t* words =
                static_cast<uint32_t*>(HeapAllocate(c_SparsePageSize * sizeof(uint32_t), c_ColumnAlignment));
            if (!words)
                return nullptr;
            std::memset(words, 0, c_SparsePageSize * sizeof(uint32_t));
            set.pages[page] = words;
        }
        const uint32_t dense = set.GetCount();
        const uint32_t size = set.type->desc.size;
        if (size != 0 && dense == set.capacity)
        {
            if (set.capacity >= 0x80000000u)
                return nullptr;   // the doubled capacity would not fit 32 bits
            const uint32_t capacity = set.capacity ? set.capacity * 2 : 16;
            uint8_t* values = static_cast<uint8_t*>(HeapAllocate(size_t(capacity) * size, set.type->desc.alignment));
            if (!values)
                return nullptr;
            RelocateValues(*set.type, values, set.values, dense);
            HeapFree(set.values);
            set.values = values;
            set.capacity = capacity;
        }
        set.entities.push_back(entity);
        set.pages[page][index & (c_SparsePageSize - 1)] = dense + 1;
        ++record.sparseCount;
        SetPresence(set, record.archetype, record.chunk, record.row, true);
        return size != 0 ? set.Value(dense) : static_cast<void*>(g_TagValue);
    }

    void WorldImpl::SparseErase(SparseSet& set, Entity entity, EntityRecord& record) noexcept
    {
        const uint32_t index = GetEntityIndex(entity);
        const uint32_t dense = set.Find(index);
        if (dense == HashIndex::c_NotFound)
            return;
        const uint32_t last = set.GetCount() - 1;
        if (set.type->desc.size != 0)
        {
            DestructValues(*set.type, set.Value(dense), 1);
            if (dense != last)
                RelocateValues(*set.type, set.Value(dense), set.Value(last), 1);
        }
        if (dense != last)
        {
            const Entity moved = set.entities[last];
            const uint32_t movedIndex = GetEntityIndex(moved);
            set.entities[dense] = moved;
            set.pages[movedIndex >> c_SparsePageBits][movedIndex & (c_SparsePageSize - 1)] = dense + 1;
        }
        set.entities.pop_back();
        set.pages[index >> c_SparsePageBits][index & (c_SparsePageSize - 1)] = 0;
        --record.sparseCount;
        SetPresence(set, record.archetype, record.chunk, record.row, false);
    }

    bool WorldImpl::SecurePresence(SparseSet& set, uint32_t archetype, uint32_t chunk) noexcept
    {
        if (archetype >= set.presence.size())
            set.presence.resize(archetype + 1);
        Vector<uint64_t*>& chunks = set.presence[archetype];
        if (chunk >= chunks.size())
            chunks.resize(chunk + 1, nullptr);
        uint64_t*& mask = chunks[chunk];
        if (!mask)
        {
            // The chunk need not exist yet: a mask outlives its chunk and serves the next chunk at that index.
            const size_t bytes = size_t(m_Archetypes[archetype]->maskWords) * sizeof(uint64_t);
            mask = static_cast<uint64_t*>(HeapAllocate(bytes, c_ColumnAlignment));
            if (!mask)
                return false;
            std::memset(mask, 0, bytes);
            ++m_PresenceMaskVersion;   // GetSparsePresence returns another pointer for this chunk now
        }
        return true;
    }

    bool WorldImpl::SecureRowPresence(Entity entity, uint32_t archetype, uint32_t chunk) noexcept
    {
        const uint32_t index = GetEntityIndex(entity);
        if (m_Entities[index].sparseCount == 0)
            return true;
        for (SparseSet* set : m_SparseSets)
            if (set->Find(index) != HashIndex::c_NotFound && !SecurePresence(*set, archetype, chunk))
                return false;
        return true;
    }

    bool WorldImpl::SecureHoleFill(const Archetype& archetype, uint32_t chunk) noexcept
    {
        const uint32_t last = archetype.entityCount - 1;
        const uint32_t lastChunk = last / archetype.rowsPerChunk;
        if (lastChunk == chunk)
            return true;   // the last row stays in its chunk, whose masks it already has
        return SecureRowPresence(archetype.Entities(lastChunk)[last % archetype.rowsPerChunk], archetype.index, chunk);
    }

    void WorldImpl::SetPresence(SparseSet& set, uint32_t archetype, uint32_t chunk, uint32_t row, bool present) noexcept
    {
        if (present)
        {
            // The structural changes secure the mask before they change anything; this allocation cannot fail for
            // them. Others (LoadSnapshot) check the mask afterwards.
            if (SecurePresence(set, archetype, chunk))
                SetMaskBit(set.presence[archetype][chunk], row, true);
            return;
        }
        if (archetype >= set.presence.size())
            return;
        const Vector<uint64_t*>& chunks = set.presence[archetype];
        if (chunk < chunks.size() && chunks[chunk])
            SetMaskBit(chunks[chunk], row, false);
    }

    void WorldImpl::MoveSparsePresence(Entity entity, uint32_t fromArchetype, uint32_t fromChunk, uint32_t fromRow,
                                       uint32_t toArchetype, uint32_t toChunk, uint32_t toRow) noexcept
    {
        const uint32_t index = GetEntityIndex(entity);
        for (SparseSet* set : m_SparseSets)
        {
            if (set->Find(index) == HashIndex::c_NotFound)
                continue;
            SetPresence(*set, fromArchetype, fromChunk, fromRow, false);
            SetPresence(*set, toArchetype, toChunk, toRow, true);
        }
    }

    const SparseSet* WorldImpl::GetSparseSet(ComponentId component) const noexcept
    {
        const ComponentType* type = GetType(component);
        return type && type->kind == StorageKind::Sparse ? m_SparseSets[type->sparseSet] : nullptr;
    }

    const uint64_t* WorldImpl::GetSparsePresence(const SparseSet& set, uint32_t archetype,
                                                 uint32_t chunk) const noexcept
    {
        if (archetype < set.presence.size())
        {
            const Vector<uint64_t*>& chunks = set.presence[archetype];
            if (chunk < chunks.size() && chunks[chunk])
                return chunks[chunk];
        }
        return g_ZeroMask;
    }
}
