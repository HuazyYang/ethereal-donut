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

// The world's storage (aggregate docs/ecs-design.md 4.1-4.5, 4.9, 4.10; plan step E1.1): the registry, the entity
// table, archetypes and their chunks, sparse sets, dynamic buffers, resources, and the immediate operations of IWorld.
// Module-private (com-interface-abi.md section 6): none of these types crosses a module boundary.
//
// Seams for the later steps of plan E1, all on WorldImpl:
// - E1.2 queries: archetypes are only ever appended (indices in creation order) and never destroyed, so a cached
//   query matches archetypes [its count, GetArchetypeStorageCount()) when it refreshes. Per archetype it finds its
//   terms with Archetype::FindColumn and reads chunks through Archetype::Entities/ColumnData/EnabledMask/Versions;
//   Sparse terms read GetSparsePresence. GetStructuralVersion changes with every structural change.
// - E1.3 playback: the *Internal batch operations take an explicit version and do no phase or observer checks; entity
//   lists may come from the caller's buffers. AllocateEntityIds hands out reserved ids for placeholders.
// - E1.4 observers: every structural change reports its rows through the Observe* functions (EcsObserver.cpp) and the
//   IWorld entry points call FlushObservers before they return. m_ObserverDepth makes structural calls fail while an
//   observer runs.
// - E1.5 snapshots and hashes: ComponentType::fieldBytes are the bytes to hash; archetypes, chunks and rows, sparse
//   sets (registry order, dense order) and the entity table (with its FIFO free list) are all in deterministic order.

#pragma once

#include <donut/core/query_cast.h>
#include <donut/engine/EcsWorld.h>
#include <nvrhi/core/containers.h>
#include <nvrhi/core/foundation.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace donut::engine::ecs
{
    // ---------------------------------------------------------------------------------------------------------- memory

    // Every heap block of the storage (chunk pages, containers, sparse values, spilled buffers, resources) comes from
    // nvrhi's default allocator through these functions, which count the blocks process-wide. Tests read the counts to
    // prove that steady-state operations allocate nothing and that a destroyed world frees everything.
    void* HeapAllocate(size_t bytes, size_t alignment) noexcept;   // null on failure; alignment is a power of two
    void HeapFree(void* memory) noexcept;
    uint64_t GetHeapAllocationCount() noexcept;   // blocks allocated since the process started
    int64_t GetHeapLiveBlockCount() noexcept;     // blocks allocated and not freed
    // Tests of out-of-memory paths: after `allocations` more successful blocks, the next HeapAllocate fails once.
    // Negative disarms. Process-wide and not synchronised: single-threaded tests only. Containers do not survive a
    // failed allocation (NVRHI_VERIFY), so a test aims it at a page, an archetype or a buffer.
    void SetHeapFailureCountdown(int64_t allocations) noexcept;

    // A stateless nvrhi container allocator over HeapAllocate: every container of the storage is an ecs::Vector.
    class Allocator
    {
    public:
        void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) noexcept
        {
            void* memory = HeapAllocate(bytes, alignment);
            NVRHI_VERIFY(memory != nullptr, "ecs::Allocator: out of memory");
            return memory;
        }
        void deallocate(void* memory, size_t, size_t = alignof(std::max_align_t)) noexcept { HeapFree(memory); }

        friend bool operator==(const Allocator&, const Allocator&) noexcept { return true; }
        friend bool operator!=(const Allocator&, const Allocator&) noexcept { return false; }
    };

    template <typename T>
    using Vector = nvrhi::vector<T, Allocator>;

    // Fixed-size pages for chunks, carved from slabs of c_SlabBytes and kept on a free list: a chunk freed by one
    // structural change is reused by the next, so that steady-state changes allocate nothing (10.3). Growing the pool
    // fails softly: the slab and the slab list are plain HeapAllocate blocks.
    //
    // A pool that shares its slabs (ShareSlabs, before its first page; plan step E3) takes them from a process-wide
    // cache first and gives them back to it on Release, so that a world made after another one was destroyed (a scene
    // graph per loaded scene) reuses memory that is already committed instead of faulting fresh pages in. The cache
    // keeps at most c_MaxSharedSlabBytes and frees the rest; worlds that do not share never touch it.
    class PagePool
    {
    public:
        static constexpr uint32_t c_SlabBytes = 256 * 1024;
        static constexpr uint64_t c_MaxSharedSlabBytes = uint64_t(256) << 20;

        void ShareSlabs() noexcept { m_Shared = true; }

        void Initialize(uint32_t pageSize) noexcept;
        void* Allocate() noexcept;              // null when out of memory
        void Free(void* page) noexcept;
        // Adds slabs until at least `freePages` pages are free, so that as many Allocate calls cannot fail; false when
        // out of memory (the slabs added stay free).
        bool Reserve(uint32_t freePages) noexcept;
        void Release() noexcept;                // frees every slab; no page may be in use
        uint32_t GetPageSize() const noexcept { return m_PageSize; }
        uint32_t GetPagesInUse() const noexcept { return m_PagesInUse; }
        uint32_t GetFreePageCount() const noexcept { return m_SlabCount * m_PagesPerSlab - m_PagesInUse; }

    private:
        struct FreePage { FreePage* next; };
        bool AddSlab() noexcept;                // false, with nothing changed, when out of memory
        FreePage* m_FreeList = nullptr;
        void** m_Slabs = nullptr;
        uint32_t m_SlabCount = 0;
        uint32_t m_SlabCapacity = 0;
        uint32_t m_PageSize = 0;
        uint32_t m_PagesPerSlab = 0;
        uint32_t m_PagesInUse = 0;
        bool m_Shared = false;
    };

    // The process-wide cache of shared slabs (PagePool::ShareSlabs): the bytes it holds, and freeing all of it.
    uint64_t GetSharedSlabBytes() noexcept;
    void ReleaseSharedSlabs() noexcept;

    // ------------------------------------------------------------------------------------------- chunk words

    // Change versions and enabled bits may be written from parallel systems for different rows of one chunk
    // (GetComponentForWrite, ResizeBuffer, SetEnabled), so they are updated atomically.

    // *slot = max(*slot, version), a release when it writes: a reader that acquires the newest row version of a column
    // (LoadVersionWord) sees the row versions stamped before it (backlog I4-R7). On x86 and x64 the same instruction as
    // a relaxed one; MSVC's _InterlockedCompareExchange64 is a full barrier on every target.
    inline void StampVersion(uint64_t* slot, uint64_t version) noexcept
    {
#if defined(_MSC_VER)
        volatile long long* target = reinterpret_cast<volatile long long*>(slot);
        long long current = *target;
        while (uint64_t(current) < version)
        {
            const long long previous = _InterlockedCompareExchange64(target, (long long)version, current);
            if (previous == current)
                break;
            current = previous;
        }
#else
        uint64_t current = __atomic_load_n(slot, __ATOMIC_RELAXED);
        while (current < version &&
               !__atomic_compare_exchange_n(slot, &current, version, true, __ATOMIC_RELEASE, __ATOMIC_RELAXED))
        {
        }
#endif
    }

    inline bool GetMaskBit(const uint64_t* words, uint32_t row) noexcept
    {
        return ((words[row >> 6] >> (row & 63)) & 1) != 0;
    }

    inline void SetMaskBit(uint64_t* words, uint32_t row, bool value) noexcept
    {
        const uint64_t bit = uint64_t(1) << (row & 63);
#if defined(_MSC_VER)
        volatile long long* word = reinterpret_cast<volatile long long*>(words + (row >> 6));
        if (value)
            _InterlockedOr64(word, (long long)bit);
        else
            _InterlockedAnd64(word, (long long)~bit);
#else
        if (value)
            __atomic_fetch_or(words + (row >> 6), bit, __ATOMIC_RELAXED);
        else
            __atomic_fetch_and(words + (row >> 6), ~bit, __ATOMIC_RELAXED);
#endif
    }

    // Sets or clears the bits [first, first + count); structural changes only (not atomic).
    void SetMaskRange(uint64_t* words, uint32_t first, uint32_t count, bool value) noexcept;

    // What GetComponent returns for a tag the entity has, and a query's column for a tag data term: an address, but no
    // bytes to read or write. The typed layer forms `column + row` for its rows, so the array spans every row of a
    // chunk (c_MaxSortKeyRow + 1) times c_MaxTagTypeSize, the size of an empty C++ type aligned to the component
    // limit of 64. Zero-initialised and never written: it costs address space, not memory.
    constexpr size_t c_MaxTagTypeSize = 64;
    alignas(64) inline uint8_t g_TagValue[(size_t(c_MaxSortKeyRow) + 1) * c_MaxTagTypeSize] = {};
    // The presence mask of a chunk in which no row has a Sparse component.
    alignas(64) inline const uint64_t g_ZeroMask[c_MaxSortKeyRow / 64 + 1] = {};

    // ---------------------------------------------------------------------------------------------------- hashing

    constexpr uint64_t c_HashSeed = 0xcbf29ce484222325ull;   // FNV-1a 64

    uint64_t HashBytes(const void* data, size_t size, uint64_t hash = c_HashSeed) noexcept;

    // An open-addressing table from 64-bit hashes to dense indices. It is only looked up, never iterated, so no order
    // in the world depends on it (rule 3.3.6).
    class HashIndex
    {
    public:
        static constexpr uint32_t c_NotFound = 0xffffffffu;

        template <typename Equal>
        uint32_t Find(uint64_t hash, Equal&& equal) const noexcept
        {
            if (m_Slots.empty())
                return c_NotFound;
            const size_t mask = m_Slots.size() - 1;
            for (size_t i = size_t(hash) & mask;; i = (i + 1) & mask)
            {
                const Slot& slot = m_Slots[i];
                if (slot.value == c_NotFound)
                    return c_NotFound;
                if (slot.hash == hash && equal(slot.value))
                    return slot.value;
            }
        }

        // `value` must not be in the table yet.
        void Insert(uint64_t hash, uint32_t value) noexcept;

    private:
        struct Slot
        {
            uint64_t hash;
            uint32_t value;   // c_NotFound when empty
            uint32_t reserved;
        };

        void Place(uint64_t hash, uint32_t value) noexcept;

        Vector<Slot> m_Slots;
        uint32_t m_Count = 0;
    };

    // ---------------------------------------------------------------------------------------------- registry (4.2)

    enum class StorageKind : uint8_t
    {
        Data,       // a column of values, one per row
        Tag,        // presence only: in the archetype, no column
        Buffer,     // a column of BufferHeaders, plus inline element storage per row
        Shared,     // one value per archetype, part of its key (4.4)
        Chunk,      // one value per chunk (4.4)
        Sparse,     // in a sparse set, outside the archetype (4.3)
        Resource,   // one value per world (4.9)
    };

    struct ByteRange
    {
        uint32_t offset;
        uint32_t size;
    };

    struct ComponentType
    {
        ComponentTypeDesc desc;             // the world's copy; name, fields and hooks point at the registering module
        ComponentId id = ComponentId::Invalid;
        StorageKind kind = StorageKind::Data;
        bool enableable = false;
        bool copyable = false;              // a copy hook, Relocatable (memcpy copies it), or a tag; else no copies
        std::atomic<bool> warnedRowTracked{ false };   // the one warning that RowTracked was asked again (Data)
        uint32_t sparseSet = 0xffffffffu;   // Sparse: index in WorldImpl::m_SparseSets
        void* value = nullptr;              // Resource: the world's value. Shared: the default value
        Vector<ByteRange> fieldBytes;       // bytes the fields cover, merged and ascending: compared and hashed (4.4)
        Vector<uint32_t> entityOffsets;     // offset of every Entity scalar of a value (of an element, for a Buffer)
    };

    // Value operations: a null hook is the trivial operation (EcsTypes.h). For a Buffer, the values are elements.
    void ConstructValues(const ComponentType& type, void* dst, uint32_t count) noexcept;
    void DestructValues(const ComponentType& type, void* dst, uint32_t count) noexcept;
    void RelocateValues(const ComponentType& type, void* dst, void* src, uint32_t count) noexcept;
    void CopyValues(const ComponentType& type, void* dst, const void* src, uint32_t count) noexcept;
    void TransferValues(const ComponentType& type, void* dst, const void* src, uint32_t count,
                        ValueTransfer transfer) noexcept;
    bool FieldBytesEqual(const ComponentType& type, const void* a, const void* b) noexcept;
    uint64_t HashFieldBytes(const ComponentType& type, const void* value, uint64_t hash) noexcept;

    // ------------------------------------------------------------------------------------------- archetypes (4.3)

    constexpr uint32_t c_NoColumn = 0xffffffffu;
    constexpr uint32_t c_NoArchetype = 0xffffffffu;           // none: a new or a destroyed entity, an uncached edge
    // What FindOrCreateArchetype and the Get*Target functions return instead of an archetype index when they fail.
    constexpr uint32_t c_ArchetypeTooLarge = 0xfffffffeu;     // one row does not fit the largest chunk: FE_INVALID_ARGS
    constexpr uint32_t c_ArchetypeOutOfMemory = 0xfffffffdu;  // FE_OUT_OF_MEMORY
    constexpr bool IsArchetypeError(uint32_t archetype) noexcept { return archetype >= c_ArchetypeOutOfMemory; }
    constexpr nvrhi::FRESULT ArchetypeErrorResult(uint32_t archetype) noexcept
    {
        return archetype == c_ArchetypeOutOfMemory ? nvrhi::FE_OUT_OF_MEMORY : nvrhi::FE_INVALID_ARGS;
    }
    constexpr uint32_t c_ColumnAlignment = 64;
    // Row indices fit the 16 bits of a command sort key (EcsCommandBuffer.h) and of EntityRecord::row.
    constexpr uint32_t c_MaxRowsPerChunk = c_MaxSortKeyRow + 1;

    // ArchetypeColumn::rowVersionSlot of a column without row versions.
    constexpr uint16_t c_NoRowVersions = 0xffff;

    struct ArchetypeColumn
    {
        const ComponentType* type;
        ComponentId component;
        StorageKind kind;           // Data, Tag, Buffer, Shared or Chunk
        bool enableable;
        uint16_t rowVersionSlot;    // RowTracked Data: the column's array in Archetype::rowVersionBlocks; else
                                    // c_NoRowVersions
        uint32_t offset;            // Data, Buffer: the column in the chunk. Chunk: the value in the chunk. Shared:
                                    // the value in Archetype::sharedValues
        uint32_t stride;            // Data: the value size; Buffer: sizeof(BufferHeader); Shared, Chunk: the value size
        uint32_t inlineOffset;      // Buffer: the inline element storage in the chunk,
        uint32_t inlineStride;      //         inlineStride bytes per row (bufferInlineCapacity elements)
        uint32_t enabledOffset;     // Enableable: the row mask in the chunk (Archetype::maskWords words)
    };

    struct ChunkRecord
    {
        uint8_t* memory;
        uint32_t count;             // rows in use
        uint32_t reserved;
        // The version at which the chunk last lost a row that was not replaced (RemoveRow: the archetype's last row,
        // removed or moved into a hole elsewhere), 0 if never. No column version says so (the chunk's rows that stay
        // are unchanged): a per-chunk aggregate of the rows (the scene graph's ChunkWorldBounds) reads it to know that
        // it is stale (backlog T4-1).
        uint64_t shrunkVersion;
    };

    struct ArchetypeEdge
    {
        ComponentId component;
        uint32_t archetype;
    };

    // A cached target of removing and adding several components at once (WorldImpl::PrepareChange, E6.3).
    struct ArchetypeTransition
    {
        uint32_t firstId;           // in Archetype::transitionIds: the removed components, then the added ones, as the
        uint32_t removedCount;      // caller listed them
        uint32_t addedCount;
        uint32_t archetype;
        bool addsPresent;           // some component to add is in the source already (and not removed): its value is
    };                              // written there

    // A sorted set of components plus the values of its Shared components (4.3). Chunk layout: the Entity column; each
    // Data and Buffer column, 64-byte aligned; the inline storage of each Buffer column, 64-byte aligned; the enabled
    // masks of the Enableable columns; the change versions, one per column in column order and the added version last;
    // the values of the Chunk components. Rows are dense over the archetype: every chunk but the last is full, and a
    // removal moves the archetype's last row into the hole (swap-remove), so row order follows only from the order of
    // structural changes (rule 3.3.6).
    struct Archetype
    {
        uint32_t index = 0;             // in creation order
        uint32_t chunkSize = 0;
        uint32_t rowsPerChunk = 0;
        uint32_t rowSize = 0;           // the Entity, the Data and Buffer columns and the inline buffer storage
        uint32_t maskWords = 0;         // per enabled mask: ceil(rowsPerChunk / 64)
        uint32_t versionsOffset = 0;
        uint32_t entityCount = 0;
        uint32_t sharedSize = 0;
        uint64_t hash = 0;
        uint8_t* sharedValues = nullptr;
        // Counts the removals of rows (a swap-remove moves the archetype's last row into the hole) and the chunks
        // released. A (chunk, row) of one of its entities, remembered with this count, is still that entity's place
        // while the count is unchanged: appending rows moves nothing (plan step E3, the scene graph's row hints).
        uint64_t rowMoves = 0;
        bool destructRows = false;      // some column needs work to destroy a row (hooks, spilled buffers)
        uint8_t reserved[3] = {};
        // RowTracked columns (D9; plan step E7, item E7.5): their row versions live outside the chunk, so that they do
        // not lower rowsPerChunk. Per chunk one block of rowVersionSlots arrays, each the column's newest row version
        // followed by rowsPerChunk row versions. Blocks are kept by chunk index: a released chunk's block waits for
        // the next chunk at that index, and the world frees them when it dies (steady state allocates nothing). They
        // are the world's own heap blocks (HeapAllocate), one per chunk index, not shared through the process-wide
        // slab cache of the chunk pages (ShareChunkPages): their size is per archetype. WorldStats::rowVersionBytes
        // counts them (backlog I4-R5).
        uint32_t rowVersionSlots = 0;
        Vector<uint64_t*> rowVersionBlocks;
        Vector<ComponentId> components; // sorted
        Vector<ArchetypeColumn> columns;// one per component, in the same order
        Vector<ChunkRecord> chunks;
        Vector<ArchetypeEdge> addEdges;     // cached targets of adding a component (not for Shared: the value decides)
        Vector<ArchetypeEdge> removeEdges;  // cached targets of removing one
        Vector<ArchetypeTransition> transitions;   // cached targets of several changes at once; one per pair of lists
        Vector<ComponentId> transitionIds;         // that was asked for, and kept as the edges are

        uint32_t FindColumn(ComponentId component) const noexcept;
        uint32_t GetChunkCount() const noexcept { return uint32_t(chunks.size()); }
        uint32_t AddedVersionSlot() const noexcept { return uint32_t(columns.size()); }

        Entity* Entities(uint32_t chunk) const noexcept { return reinterpret_cast<Entity*>(chunks[chunk].memory); }
        // Data, Buffer: the column's first row. Chunk: the chunk's value. Shared: the archetype's value. Tag: null.
        void* ColumnData(uint32_t chunk, uint32_t column) const noexcept;
        // The row mask of an Enableable column; null for other columns.
        uint64_t* EnabledMask(uint32_t chunk, uint32_t column) const noexcept;
        // Versions(chunk)[column] is the column's change version; [AddedVersionSlot()] the version at which rows were
        // last added.
        uint64_t* Versions(uint32_t chunk) const noexcept
        {
            return reinterpret_cast<uint64_t*>(chunks[chunk].memory + versionsOffset);
        }
        uint8_t* InlineData(uint32_t chunk, uint32_t column, uint32_t row) const noexcept
        {
            const ArchetypeColumn& c = columns[column];
            return chunks[chunk].memory + c.inlineOffset + size_t(row) * c.inlineStride;
        }
        // The row versions of a RowTracked column in a chunk (ChunkView::rowVersions): [-1] is the newest row
        // version, [row] each row's. Null for other columns.
        uint64_t* RowVersions(uint32_t chunk, uint32_t column) const noexcept
        {
            const uint32_t slot = columns[column].rowVersionSlot;
            return slot == c_NoRowVersions ? nullptr
                                           : rowVersionBlocks[chunk] + size_t(slot) * (size_t(rowsPerChunk) + 1) + 1;
        }
    };

    // ------------------------------------------------------------------------------- row versions (D9, E7.5)

    // A version slot or row version written by other threads (StampVersion), read as a whole, with acquire: after a
    // load of a column's newest row version ([-1]) the rows stamped before it are seen (the spread of the chunk-only
    // writer rule writes the rows, then [-1]; backlog I4-R7). On x86 and x64 a plain load (the fence only stops the
    // compiler); on weakly ordered CPUs a load and a barrier.
    inline uint64_t LoadVersionWord(const uint64_t* slot) noexcept
    {
#if defined(_MSC_VER)
        const uint64_t value = uint64_t(*reinterpret_cast<const volatile long long*>(slot));
        std::atomic_thread_fence(std::memory_order_acquire);
        return value;
#else
        return __atomic_load_n(slot, __ATOMIC_ACQUIRE);
#endif
    }

    // The chunk-only writer rule (ChunkView): a RowTracked column whose chunk version is newer than its newest row
    // version had a write that stamped only the chunk. Before anything stamps that column's rows or chunk again, every
    // row in use is stamped with the chunk version, so that the write stays seen. Nothing when the versions agree;
    // nothing for a column without row versions. Rows, then the newest version, with StampVersion: parallel readers of
    // the column may do it at once.
    void SpreadChunkWrite(const Archetype& archetype, uint32_t chunk, uint32_t column) noexcept;

    // A write that stamps the chunk only (a query's Write term, a domain's column view). On a RowTracked column whose
    // newest row version is this version or a later one (a row write of the same version came first, such as a node
    // setter and a c_WorldVersion query outside phases), the chunk version would not be newer than the rows, so the
    // rule above could not see the write: every row in use is stamped with the version at once.
    void StampChunkWrite(const Archetype& archetype, uint32_t chunk, uint32_t column, uint64_t version) noexcept;

    // A write of rows [firstRow, firstRow + count) of a column: their row versions, the newest one, then the chunk's.
    // For a column without row versions, the chunk's only.
    void StampRowWrites(const Archetype& archetype, uint32_t chunk, uint32_t column, uint32_t firstRow, uint32_t count,
                        uint64_t version) noexcept;
    inline void StampRowWrite(const Archetype& archetype, uint32_t chunk, uint32_t column, uint32_t row,
                              uint64_t version) noexcept
    {
        if (archetype.columns[column].rowVersionSlot != c_NoRowVersions)
            StampRowWrites(archetype, chunk, column, row, 1, version);
        else
            StampVersion(archetype.Versions(chunk) + column, version);
    }

    // For readers that test chunk versions themselves (the scene graph's systems, GpuScene): the row versions of a
    // column in a chunk when they tell its changed rows apart, else null: no row versions, or a chunk-only write made
    // the chunk wholly changed. With them, a chunk whose column version is newer than sinceVersion changed in exactly
    // its rows whose version is newer. Reads only.
    inline const uint64_t* ChangedRowVersions(const Archetype& archetype, uint32_t chunk, uint32_t column) noexcept
    {
        const uint64_t* rows = archetype.RowVersions(chunk, column);
        if (!rows || LoadVersionWord(archetype.Versions(chunk) + column) > LoadVersionWord(rows - 1))
            return nullptr;
        return rows;
    }

    // ------------------------------------------------------------------------------------------ sparse sets (4.3)

    constexpr uint32_t c_SparsePageBits = 12;
    constexpr uint32_t c_SparsePageSize = 1u << c_SparsePageBits;

    struct SparseSet
    {
        const ComponentType* type = nullptr;
        Vector<uint32_t*> pages;                // by entity index: dense index + 1, 0 when absent
        Vector<Entity> entities;                // dense
        uint8_t* values = nullptr;              // dense, desc.size bytes each; none for a tag
        uint32_t capacity = 0;
        Vector<Vector<uint64_t*>> presence;     // [archetype][chunk]: the rows whose entity is in the set; allocated
                                                // on first use

        uint32_t Find(uint32_t entityIndex) const noexcept   // dense index, or HashIndex::c_NotFound
        {
            const uint32_t page = entityIndex >> c_SparsePageBits;
            if (page >= pages.size() || !pages[page])
                return HashIndex::c_NotFound;
            return pages[page][entityIndex & (c_SparsePageSize - 1)] - 1;
        }
        uint32_t GetCount() const noexcept { return uint32_t(entities.size()); }
        void* Value(uint32_t dense) const noexcept { return values + size_t(dense) * type->desc.size; }
    };

    // ----------------------------------------------------------------------------------------- entity table (4.1)

    constexpr uint32_t c_FreeSlot = 0xffffffffu;       // EntityRecord::archetype of a free index
    constexpr uint32_t c_ReservedSlot = 0xfffffffeu;   // reserved (ReserveEntities), not alive yet
    constexpr uint32_t c_ClaimedSlot = 0xfffffffdu;    // reserved and being created (CreateEntities)
    constexpr uint32_t c_FirstGeneration = 1;          // so that MakeEntity(index, 0) is never alive

    struct EntityRecord
    {
        uint32_t generation;
        uint32_t archetype;     // the archetype of a live entity; c_FreeSlot, c_ReservedSlot or c_ClaimedSlot otherwise
        uint32_t chunk;         // free: the next index of the FIFO free list, 0 at its end
        uint16_t row;
        uint16_t sparseCount;   // the entity's Sparse components
    };
    static_assert(sizeof(EntityRecord) == 16, "EntityRecord layout");

    // Generations skip the placeholder generation and 0 when they wrap.
    constexpr uint32_t NextGeneration(uint32_t generation) noexcept
    {
        const uint32_t next = generation + 1;
        return next == c_PlaceholderGeneration || next == 0 ? c_FirstGeneration : next;
    }

    // -------------------------------------------------------------------------------------------------- the world

    // The world's command-buffer state (EcsCommandBufferImpl.h, plan step E1.3): the placeholder counter, the
    // placeholder tables, the playback's scratch. WorldImpl::Initialize creates it, ~WorldImpl destroys it.
    class CommandPlayback;
    CommandPlayback* CreateCommandPlayback() noexcept;   // null when out of memory
    void DestroyCommandPlayback(CommandPlayback* playback) noexcept;

    // The world's observers (EcsObserverImpl.h, plan step E1.4): registrations, the batch held for merging, the
    // observers' command buffers. The first AddObserver makes it; ~WorldImpl destroys it before anything else.
    class ObserverRegistry;
    void DestroyObserverRegistry(ObserverRegistry* registry) noexcept;

    NVRHI_CLASS_CLSID(WorldImpl, "9dd22807-f2d4-43cc-a6f8-3ebf910603af")
    class WorldImpl final : public nvrhi::ObjectImpl<IWorld>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(WorldImpl)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(WorldImpl)
        NVRHI_IMPLEMENTS_INTERFACE(IWorld)
        NVRHI_IMPLEMENTS_CLASS(WorldImpl)
        NVRHI_END_INTERFACE_TABLE()

        explicit WorldImpl(const WorldDesc& desc) noexcept;
        ~WorldImpl();
        nvrhi::FRESULT Initialize() noexcept;

        // IWorld
        nvrhi::FRESULT RegisterComponent(const ComponentTypeDesc& desc, ComponentId& outId) noexcept override;
        ComponentId FindComponent(const nvrhi::GUID& uuid) noexcept override;
        const ComponentTypeDesc* GetComponentTypeDesc(ComponentId component) noexcept override;
        uint32_t GetComponentTypeCount() noexcept override;
        nvrhi::FRESULT RegisterResource(const ComponentTypeDesc& desc, const void* initialValue,
                                        ComponentId& outId) noexcept override;
        void* GetResource(ComponentId resource) noexcept override;
        nvrhi::FRESULT ReserveEntities(uint32_t count, Entity* outEntities) noexcept override;
        nvrhi::FRESULT CreateEntities(const EntityCreateDesc& desc, Entity* outEntities) noexcept override;
        nvrhi::FRESULT DestroyEntities(const Entity* entities, uint32_t count) noexcept override;
        nvrhi::FRESULT AddComponent(Entity entity, ComponentId component, const void* value) noexcept override;
        nvrhi::FRESULT RemoveComponent(Entity entity, ComponentId component) noexcept override;
        nvrhi::FRESULT Instantiate(Entity prefab, uint32_t count, Entity* outEntities) noexcept override;
        nvrhi::FRESULT CopyEntitiesFrom(IWorld* source, const Entity* entities, uint32_t count,
                                        Entity* outEntities) noexcept override;
        Entity ResolvePlaceholder(Entity placeholder) noexcept override;
        bool IsAlive(Entity entity) noexcept override;
        bool HasComponent(Entity entity, ComponentId component) noexcept override;
        const void* GetComponent(Entity entity, ComponentId component) noexcept override;
        void* GetComponentForWrite(Entity entity, ComponentId component, uint64_t writeVersion) noexcept override;
        nvrhi::FRESULT SetEnabled(Entity entity, ComponentId component, bool enabled) noexcept override;
        bool IsEnabled(Entity entity, ComponentId component) noexcept override;
        nvrhi::FRESULT ResizeBuffer(Entity entity, ComponentId component, uint32_t count,
                                    uint64_t writeVersion) noexcept override;
        nvrhi::FRESULT CreateQuery(const QueryDesc& desc, IQuery** ppQuery) noexcept override;
        nvrhi::FRESULT CreateCommandBuffer(ICommandBuffer** ppBuffer) noexcept override;
        nvrhi::FRESULT Playback(ICommandBuffer* const* buffers, uint32_t count) noexcept override;
        nvrhi::FRESULT AddObserver(const ObserverDesc& desc, IObserver* observer, ObserverId& outId) noexcept override;
        nvrhi::FRESULT RemoveObserver(ObserverId observer) noexcept override;
        uint64_t GetVersion() noexcept override { return GetStampVersion(); }
        uint64_t AdvanceVersion() noexcept override { return ++m_Version; }
        uint64_t GetStructuralVersion() noexcept override { return m_StructuralVersion; }
        bool IsInPhase() noexcept override { return m_InPhase; }
        nvrhi::FRESULT SaveSnapshot(IStateWriter* writer) noexcept override;
        nvrhi::FRESULT LoadSnapshot(IStateReader* reader) noexcept override;
        uint64_t ComputeHash() noexcept override;
        WorldStats& GetStats(WorldStats& retVal) noexcept override;
        uint32_t GetArchetypeCount() noexcept override { return uint32_t(m_Archetypes.size()); }
        nvrhi::FRESULT GetArchetypeInfo(uint32_t archetype, ArchetypeInfo& out) noexcept override;
        nvrhi::FRESULT GetChunkInfo(uint32_t archetype, uint32_t chunk, ChunkInfo& out) noexcept override;

        // ------------------------------------------------------------------------------- module-internal access

        const WorldDesc& GetDesc() const noexcept { return m_Desc; }
        bool IsValidationEnabled() const noexcept { return m_Desc.validation; }
        // Chunk pages from the process-wide slab cache (PagePool::ShareSlabs); before the first entity is created.
        void ShareChunkPages() noexcept
        {
            m_SmallPages.ShareSlabs();
            m_LargePages.ShareSlabs();
        }
        uint64_t GetValidationErrorCount() const noexcept
        {
            return m_ValidationErrors.load(std::memory_order_relaxed);
        }
        // With validation on: counts the failure and reports it through log::warning. For the world's operations and
        // the objects beside it (queries, command buffers, observers); printf format.
        void ReportValidation(const char* operation, const char* format, ...) noexcept;
        // The version a write stamps: `writeVersion`, or the world's version for c_WorldVersion.
        uint64_t ResolveWriteVersion(uint64_t writeVersion) const noexcept
        {
            return writeVersion == c_WorldVersion ? GetStampVersion() : writeVersion;
        }
        // What immediate structural changes, playback and c_WorldVersion writes stamp: the world's version, or, while
        // the scheduler runs an exclusive system inside a phase (its sync points included), that system's own version,
        // so that the systems after it in the phase see its changes once, not again at their next run (E2.2).
        uint64_t GetStampVersion() const noexcept { return m_StampVersion != 0 ? m_StampVersion : m_Version; }
        void SetStampVersion(uint64_t version) noexcept { m_StampVersion = version; }   // 0: the world's version

        const ComponentType* GetType(ComponentId component) const noexcept
        {
            return uint32_t(component) < m_Types.size() ? m_Types[uint32_t(component)] : nullptr;
        }
        uint32_t GetArchetypeStorageCount() const noexcept { return uint32_t(m_Archetypes.size()); }
        Archetype& GetArchetypeStorage(uint32_t archetype) const noexcept { return *m_Archetypes[archetype]; }
        // The record of a live entity; null for null, stale, free and reserved ids.
        const EntityRecord* FindAlive(Entity entity) const noexcept;
        // The entity table, index 0 included (never used); for snapshots and diagnostics.
        uint32_t GetEntityTableSize() const noexcept { return uint32_t(m_Entities.size()); }
        const EntityRecord& GetEntityRecord(uint32_t index) const noexcept { return m_Entities[index]; }
        const SparseSet* GetSparseSet(ComponentId component) const noexcept;
        // The presence mask of a Sparse component over a chunk's rows; an all-zero mask when no row has it.
        const uint64_t* GetSparsePresence(const SparseSet& set, uint32_t archetype, uint32_t chunk) const noexcept;
        // Changes whenever a sparse set allocates a chunk's presence mask, after which GetSparsePresence returns
        // another pointer for that chunk. It happens only in structural changes; queries rebuild their chunk lists on
        // it.
        uint64_t GetPresenceMaskVersion() const noexcept { return m_PresenceMaskVersion; }
        // Command buffers (E1.3): placeholders, resolutions, playback scratch and counters. Recording threads take
        // placeholders from it.
        CommandPlayback& GetCommandPlayback() const noexcept { return *m_CommandPlayback; }

        // Counters for tests and the inspector.
        struct Counters
        {
            uint64_t archetypeLookups = 0;      // hash lookups of an archetype key (edge-cache misses, Shared values)
            uint64_t edgeHits = 0;              // structural moves whose target came from an edge cache
            uint64_t rowsAddedReports = 0;      // ObserveRowsAdded calls
            uint64_t rowsRemovingReports = 0;   // ObserveRowsRemoving calls
        };
        const Counters& GetCounters() const noexcept { return m_Counters; }
        // Pages the chunk pools hold for reuse: a new chunk needs a new heap block only when this is 0.
        uint32_t GetFreeChunkPageCount() const noexcept
        {
            return m_SmallPages.GetFreePageCount() +
                   (m_LargePages.GetPageSize() != m_SmallPages.GetPageSize() ? m_LargePages.GetFreePageCount() : 0);
        }

        // ------------------------------------------------------------------------- phases (IsInPhase, rule 3.3.3)

        // The scheduler (E2) brackets each phase, and each exclusive system inside it. Immediate structural changes
        // fail with FE_INVALID_ARGS inside a phase unless an exclusive system runs, and while an observer runs.
        void BeginPhase() noexcept { m_InPhase = true; }
        void EndPhase() noexcept { m_InPhase = false; }
        void BeginExclusive() noexcept { ++m_ExclusiveDepth; }
        void EndExclusive() noexcept { --m_ExclusiveDepth; }
        // The scheduler delivers the entities that a simulation domain finds or leaves behind by itself (E6.1): its
        // callbacks run under an observer's rules, so no structure changes while it walks the chunks.
        void BeginObserver() noexcept { ++m_ObserverDepth; }
        void EndObserver() noexcept { --m_ObserverDepth; }
        bool CanChangeStructure() const noexcept
        {
            return m_ObserverDepth == 0 && (!m_InPhase || m_ExclusiveDepth > 0);
        }
        // What IWorld::Playback and the registration of a new component type return when !CanChangeStructure():
        // nothing was done, the call belongs at a sync point (EcsWorld.h). Distinct from what a played command returns.
        static constexpr nvrhi::FRESULT c_WrongTimeResult = nvrhi::FE_OUT_OF_DATE;
        // The Debug access checks of 11.3 (E2.2): with validation, GetComponentForWrite and ResizeBuffer inside a phase
        // hand the component and the write version, which names the system run, to the check that the scheduler sets
        // for the duration of the phase.
        using WriteAccessCheck = void (*)(void* context, ComponentId component, uint64_t writeVersion,
                                          const char* operation) noexcept;
        void SetWriteAccessCheck(WriteAccessCheck check, void* context) noexcept
        {
            m_WriteAccessCheck = check;
            m_WriteAccessContext = context;
        }
        void CheckWriteAccess(ComponentId component, uint64_t writeVersion, const char* operation) noexcept
        {
            if (m_Desc.validation && m_InPhase && m_WriteAccessCheck)
                m_WriteAccessCheck(m_WriteAccessContext, component, writeVersion, operation);
        }

        // ------------------------------------------------------------------------- observer seam (E1.4)

        // Every structural change reports its rows here, in the order it applies them, and only once it is certain
        // to happen (no report precedes a failure).
        // - ObserveRowsAdded: the rows are complete in `archetype` (values in place, Sparse components of a creation
        //   included). `fromArchetype` is c_NoArchetype for new entities, else the archetype the rows came from: OnAdd
        //   for the components only `archetype` has, OnMatch for queries only it matches.
        // - ObserveRowsRemoving: before the rows' values are destroyed or moved. `toArchetype` is c_NoArchetype when
        //   the entities are destroyed, else their next archetype: OnRemove and OnUnmatch likewise. For a move, the
        //   target's chunk is already secured, so the target may end with an empty chunk at that moment.
        // - ObserveSparseAdded / ObserveSparseRemoving: a Sparse component after it was added, before it is destroyed.
        // - ObserveSet: a value written by a set (the OnSet of ICommandBuffer::SetComponents).
        // - ObserveValueAccess: at the start of a batch operation that writes values before (or without) a report, or
        //   copies existing values (Instantiate): delivers the batch the observers hold for merging, so that it reads
        //   its rows as their reports left them and what its Write observers fill in is there before it is copied.
        // The rows that a swap-remove moves to fill a hole are not reported: their component sets do not change.
        // E1.4 (EcsObserver.cpp): while no observer is registered, each is a test of m_ObserverCount.
        void ObserveRowsAdded(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                              uint32_t fromArchetype) noexcept
        {
            ++m_Counters.rowsAddedReports;
            if (m_ObserverCount != 0)
                ReportRowsAdded(archetype, chunk, firstRow, count, fromArchetype);
        }
        void ObserveRowsRemoving(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                                 uint32_t toArchetype) noexcept
        {
            ++m_Counters.rowsRemovingReports;
            if (m_ObserverCount != 0)
                ReportRowsRemoving(archetype, chunk, firstRow, count, toArchetype);
        }
        void ObserveSparseAdded(ComponentId component, Entity entity) noexcept
        {
            if (m_ObserverCount != 0)
                ReportSparse(component, entity, true);
        }
        void ObserveSparseRemoving(ComponentId component, Entity entity) noexcept
        {
            if (m_ObserverCount != 0)
                ReportSparse(component, entity, false);
        }
        void ObserveSet(ComponentId component, Entity entity) noexcept
        {
            if (m_ObserverCount != 0)
                ReportSet(component, entity);
        }
        void ObserveValueAccess() noexcept
        {
            if (m_ObserverCount != 0 && m_ObserverDepth == 0)
                DeliverHeldBatch();
        }
        // Called by every IWorld entry point that changed structure, before it returns: delivers the observers'
        // batches and plays back their command buffers (the cascade of EcsWorld.h). Returns the first failure.
        nvrhi::FRESULT FlushObservers() noexcept
        {
            return m_ObserverCount != 0 ? RunObserverCascade() : nvrhi::FS_OK;
        }
        // The observers' state; null before the first AddObserver.
        ObserverRegistry* GetObserverRegistry() const noexcept { return m_ObserverRegistry; }

        // ------------------------------------------------------------------------------- row hooks (E6.3)

        // For a caller in this module whose values have their home outside the world while no row has it (the scene
        // graph's leaf records, design 9.2), so that every value has one home at any time and observers find it in the
        // row. Passed to a batch operation below; null: nothing is called.
        // - rowsAdded: the rows of the first `count` entities of the operation exist with their initial values, and no
        //   observer has seen them: values may be moved into them.
        // - rowRemoving: every observer of the row's removal has run, and the values that leave with it are destroyed
        //   next: they may be moved out, each replaced by a constructed value.
        // Neither may change structure or throw.
        struct RowHooks
        {
            void* context;
            void (*rowsAdded)(void* context, uint32_t count) noexcept;
            void (*rowRemoving)(void* context, Entity entity, const Archetype& archetype, uint32_t chunk,
                                uint32_t row) noexcept;
        };
        // A value the move of an entity writes into a column that only its new archetype has.
        struct NewValue
        {
            ComponentId component = ComponentId::Invalid;
            const void* value = nullptr;
            ValueTransfer transfer = ValueTransfer::Copy;
        };
        // Components that one move of a row removes and adds (PrepareChange, ChangeComponentsInternal). Not Sparse and
        // Shared ones; none in both lists. The lists are the caller's and stay as they are between the two calls.
        struct ComponentChange
        {
            const ComponentId* removed = nullptr;
            const NewValue* added = nullptr;
            uint32_t removedCount = 0;
            uint32_t addedCount = 0;
            // PrepareChange's
            uint32_t source = c_NoArchetype;
            uint32_t target = c_NoArchetype;
            bool addsPresent = false;
        };

        // ------------------------------------------------------------------------- batch operations (E1.3)

        // The work of the IWorld entry points without their phase check and observer flush (validation reports stay).
        // `version` is what the changed columns and the added version of the touched chunks are stamped with.
        // Entity lists must not point into the world's storage (DestroyEntities copies its list first). A failure for
        // one entity (a stale id) skips it and is returned; the others still apply. Argument errors that apply to all
        // entities (a bad component id, a non-copyable value) are detected before anything changes.
        //
        // Value ownership. ValueTransfer::Copy: the caller keeps its values, whatever happens. ValueTransfer::Relocate:
        // the operation consumes every value it was given, whatever happens: a value that is not moved into the world
        // (a stale entity, an absent component, an archetype that cannot be made, out of memory, a rejected call) is
        // destroyed exactly once, with the destruct hook of the component it was given for. The one exception is a
        // value for a component id that is not registered, which cannot be destroyed. Shared values are copied into
        // the archetype key, so a relocated Shared value is always destroyed.
        nvrhi::FRESULT CreateEntitiesInternal(const EntityCreateDesc& desc, Entity* outEntities, uint64_t version,
                                              const RowHooks* hooks = nullptr) noexcept;
        nvrhi::FRESULT DestroyEntitiesInternal(const Entity* entities, uint32_t count, uint64_t version,
                                               const RowHooks* hooks = nullptr) noexcept;
        // `values`: `count` consecutive values, or null to default-construct. An entity that has the component gets
        // the value written (a Shared value moves it to the archetype of the new value); a null value leaves it.
        nvrhi::FRESULT AddComponentsInternal(const Entity* entities, uint32_t count, ComponentId component,
                                             const void* values, ValueTransfer transfer, uint64_t version) noexcept;
        nvrhi::FRESULT RemoveComponentsInternal(const Entity* entities, uint32_t count, ComponentId component,
                                                uint64_t version) noexcept;
        // Removes and adds components of one entity in one move, straight to its last archetype, so that the observers
        // of all of them see one complete row before it and one after it (E6.3). Components to remove that the entity
        // lacks are skipped. A component to add that it has gets the value written, as AddComponentsInternal does (a null
        // value leaves it). Every value stays the caller's when the call fails; nothing changed then. `change` need not
        // be prepared, or may be prepared for another archetype: it then is.
        nvrhi::FRESULT ChangeComponentsInternal(Entity entity, ComponentChange& change, uint64_t version,
                                                const RowHooks* hooks) noexcept;
        // Writes `count` consecutive values into components the entities have (FE_INVALID_ARGS where one is absent).
        // Not structural, except for Shared components, which move the entity.
        nvrhi::FRESULT SetComponentsInternal(const Entity* entities, uint32_t count, ComponentId component,
                                             const void* values, ValueTransfer transfer, uint64_t version) noexcept;
        nvrhi::FRESULT SetEnabledInternal(const Entity* entities, uint32_t count, ComponentId component, bool enabled,
                                          uint64_t version) noexcept;
        nvrhi::FRESULT InstantiateInternal(Entity prefab, uint32_t count, Entity* outEntities,
                                           uint64_t version) noexcept;
        nvrhi::FRESULT ResizeBufferInternal(Entity entity, ComponentId component, uint32_t count,
                                            uint64_t version) noexcept;
        // FE_INVALID_ARGS when the new count would not fit 32 bits.
        nvrhi::FRESULT AppendToBufferInternal(Entity entity, ComponentId component, const void* elements,
                                              uint32_t count, ValueTransfer transfer, uint64_t version) noexcept;
        // Ids in the reserved state, as ReserveEntities hands out (placeholders resolve to them at playback). No phase
        // check: playback runs it at sync points.
        nvrhi::FRESULT AllocateEntityIds(uint32_t count, Entity* outEntities) noexcept;
        // Relocate transfer: destroys `count` values of `component` (nothing for Copy, null values, unknown ids).
        void DiscardValues(ComponentId component, const void* values, uint32_t count, ValueTransfer transfer) noexcept;
        // Relocate transfer: destroys the values of the entities [first, entityCount) of a creation.
        void DiscardCreateValues(const EntityCreateDesc& desc, uint32_t first) noexcept;

        // The archetype graph. `components` sorted and unique, archetype kinds only; `shared[i]` the value of
        // components[i] if it is Shared (else ignored). Returns c_ArchetypeTooLarge when one row does not fit the
        // largest chunk, c_ArchetypeOutOfMemory when it runs out of memory (IsArchetypeError, ArchetypeErrorResult).
        uint32_t FindOrCreateArchetype(const ComponentId* components, uint32_t count,
                                       const void* const* shared) noexcept;
        uint32_t GetAddTarget(uint32_t archetype, ComponentId component) noexcept;              // not Shared
        uint32_t GetSharedTarget(uint32_t archetype, ComponentId component, const void* value) noexcept;
        uint32_t GetRemoveTarget(uint32_t archetype, ComponentId component) noexcept;
        // Finds the archetype without the removed and with the added components, cached per source archetype and lists,
        // and changes no row: a caller that must not stop half way asks first. Fails, with a validation report, when a
        // row does not fit, out of memory, or with components that a change cannot take.
        nvrhi::FRESULT PrepareChange(uint32_t archetype, ComponentChange& change) noexcept;
        // Whether one row of these components fits the largest chunk, without making the archetype.
        bool RowFits(const ComponentId* components, uint32_t count) noexcept;

    private:
        // SaveSnapshot, LoadSnapshot and ComputeHash (EcsSnapshot.cpp) read and rebuild the whole storage.
        friend class WorldSnapshot;
        // Observers (EcsObserver.cpp) run with m_ObserverDepth raised and count batches and played commands.
        friend class ObserverRegistry;

        // The observer seams' work while observers are registered (EcsObserver.cpp).
        void ReportRowsAdded(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                             uint32_t fromArchetype) noexcept;
        void ReportRowsRemoving(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                                uint32_t toArchetype) noexcept;
        void ReportSparse(ComponentId component, Entity entity, bool added) noexcept;
        void ReportSet(ComponentId component, Entity entity) noexcept;
        void DeliverHeldBatch() noexcept;
        nvrhi::FRESULT RunObserverCascade() noexcept;

        struct RowRun
        {
            uint32_t archetype;
            uint32_t chunk;
            uint32_t firstRow;
            uint32_t count;
        };

        // Registry (EcsWorld.cpp).
        nvrhi::FRESULT ValidateTypeDesc(const ComponentTypeDesc& desc) noexcept;
        nvrhi::FRESULT CheckRowTrackedAgain(ComponentType& type, const ComponentTypeDesc& desc,
                                            const char* operation) noexcept;
        nvrhi::FRESULT AddType(const ComponentTypeDesc& desc, const void* initialValue, ComponentId& outId) noexcept;
        uint32_t FindType(const nvrhi::GUID& uuid) const noexcept;

        // Entity table (EcsWorld.cpp).
        EntityRecord* FindAliveRecord(Entity entity) noexcept;
        uint32_t AllocateIndex() noexcept;   // 0 when the table is exhausted
        void ReleaseIndex(uint32_t index) noexcept;
        void ReportStale(const char* operation, Entity entity) noexcept;
        void* ComponentPointer(Entity entity, const EntityRecord& record, const ComponentType& type,
                               uint32_t* outColumn) const noexcept;

        // Archetypes and chunks (EcsArchetype.cpp).
        uint32_t CreateArchetype(const ComponentId* components, uint32_t count, const void* const* shared,
                                 uint64_t hash) noexcept;
        // Fills the columns, the row size and the Shared block size of an archetype of these components.
        void DescribeColumns(Archetype& archetype, const ComponentId* components, uint32_t count) noexcept;
        // Chooses the chunk size (D6) and the rows per chunk, and lays the chunk out; rowsPerChunk 0 when one row fits
        // no chunk.
        void PlanChunks(Archetype& archetype) const noexcept;
        static uint32_t LayoutChunk(Archetype& archetype, uint32_t rows) noexcept;   // bytes; sets the offsets
        PagePool& PoolFor(uint32_t chunkSize) noexcept;
        bool AppendChunk(Archetype& archetype, uint64_t version) noexcept;
        void ReleaseLastChunk(Archetype& archetype) noexcept;
        static void StampChunk(const Archetype& archetype, uint32_t chunk, uint64_t version) noexcept;
        // Rows placed by a structural change (appended, or moved into a hole): their row versions in every RowTracked
        // column, after the chunk-only writer rule; StampChunk then stamps the chunk.
        static void StampPlacedRows(const Archetype& archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                                    uint64_t version) noexcept;
        // Makes sure that chunk indices [0, chunkCount) have their row-version blocks, so that appending chunks up to
        // that count cannot fail for them (LoadSnapshot). False when out of memory.
        bool ReserveRowVersionBlocks(Archetype& archetype, uint32_t chunkCount) noexcept;
        // Allocates the chunks that `count` more rows need (empty, at the end), so that the next AppendRows of at most
        // `count` rows cannot fail; false, with nothing changed, when out of memory.
        bool ReserveRows(Archetype& archetype, uint32_t count, uint64_t version) noexcept;
        // Appends `count` rows and calls fill(chunk, firstRow, runCount, done) for each run of them in one chunk, in
        // order (`done`: rows of the earlier runs). The new rows have neither entity nor values yet.
        template <typename Fill>
        nvrhi::FRESULT AppendRows(Archetype& archetype, uint32_t count, uint64_t version, Fill&& fill) noexcept;
        // Removes a row whose values were destroyed or moved away: the archetype's last row moves into it.
        void RemoveRow(Archetype& archetype, uint32_t chunk, uint32_t row, uint64_t version) noexcept;
        void MoveRowWithin(Archetype& archetype, uint32_t srcChunk, uint32_t srcRow, uint32_t dstChunk,
                           uint32_t dstRow) noexcept;
        // Moves a live entity to another archetype: shared columns relocate, columns only the old one has are
        // destroyed, columns only the new one has are constructed (or take their new value), enabled bits carry over.
        // Reports ObserveRowsRemoving before and ObserveRowsAdded after the move.
        nvrhi::FRESULT MoveEntity(Entity entity, EntityRecord& record, uint32_t target, const NewValue& newValue,
                                  uint64_t version) noexcept
        {
            return MoveEntity(entity, record, target, &newValue, 1, version, nullptr);
        }
        nvrhi::FRESULT MoveEntity(Entity entity, EntityRecord& record, uint32_t target, const NewValue* newValues,
                                  uint32_t newValueCount, uint64_t version, const RowHooks* hooks) noexcept;
        void InitializeColumn(Archetype& archetype, uint32_t column, uint32_t chunk, uint32_t firstRow, uint32_t count,
                              const void* values, ValueTransfer transfer) noexcept;
        void DestructRow(Archetype& archetype, uint32_t chunk, uint32_t row) noexcept;
        void OverwriteValue(void* dst, const ComponentType& type, const void* value, ValueTransfer transfer) noexcept;
        uint32_t GetEdge(const Vector<ArchetypeEdge>& edges, ComponentId component) noexcept;

        // Dynamic buffers (EcsArchetype.cpp).
        void InitializeBuffer(BufferHeader& header, const ArchetypeColumn& column, uint8_t* inlineData) noexcept;
        void RelocateBuffer(const ArchetypeColumn& column, BufferHeader& dst, uint8_t* dstInline, BufferHeader& src,
                            uint8_t* srcInline) noexcept;
        void DestroyBuffer(const ArchetypeColumn& column, BufferHeader& header, uint8_t* inlineData) noexcept;
        bool CopyBuffer(const ArchetypeColumn& column, BufferHeader& dst, uint8_t* dstInline,
                        const BufferHeader& src) noexcept;
        nvrhi::FRESULT ResizeBufferRow(const ArchetypeColumn& column, BufferHeader& header, uint8_t* inlineData,
                                       uint32_t count) noexcept;

        // Sparse sets (EcsArchetype.cpp).
        void* SparseInsert(SparseSet& set, Entity entity, EntityRecord& record) noexcept;   // uninitialized slot
        void SparseErase(SparseSet& set, Entity entity, EntityRecord& record) noexcept;    // destroys the value
        // Setting a presence bit allocates the chunk's mask on first use. Structural changes secure every mask they
        // will set before they change anything, so that running out of memory fails them with nothing changed:
        // SecurePresence for one set, SecureRowPresence for every Sparse component of an entity in a chunk, and
        // SecureHoleFill for the archetype's last row, which RemoveRow moves into a hole in `chunk`. False when out of
        // memory.
        bool SecurePresence(SparseSet& set, uint32_t archetype, uint32_t chunk) noexcept;
        bool SecureRowPresence(Entity entity, uint32_t archetype, uint32_t chunk) noexcept;
        bool SecureHoleFill(const Archetype& archetype, uint32_t chunk) noexcept;
        void SetPresence(SparseSet& set, uint32_t archetype, uint32_t chunk, uint32_t row, bool present) noexcept;
        void MoveSparsePresence(Entity entity, uint32_t fromArchetype, uint32_t fromChunk, uint32_t fromRow,
                                uint32_t toArchetype, uint32_t toChunk, uint32_t toRow) noexcept;

        // CopyEntitiesFrom (EcsWorld.cpp).
        void RemapEntities(const ComponentType& type, uint8_t* value, uint32_t count) const noexcept;
        uint8_t* ScratchBytes(size_t bytes) noexcept;

        WorldDesc m_Desc;
        nvrhi::string m_DebugName;

        // Registry.
        Vector<ComponentType*> m_Types;
        HashIndex m_TypeIndex;
        Vector<SparseSet*> m_SparseSets;

        // Entity table: index 0 is never used.
        Vector<EntityRecord> m_Entities;
        uint32_t m_FreeHead = 0;
        uint32_t m_FreeTail = 0;
        uint32_t m_AliveCount = 0;

        // Archetypes, in creation order; never destroyed. Index 0 is the archetype without components.
        Vector<Archetype*> m_Archetypes;
        HashIndex m_ArchetypeIndex;
        PagePool m_SmallPages;
        PagePool m_LargePages;

        // Versions and state.
        uint64_t m_Version = 1;
        uint64_t m_StampVersion = 0;             // E2.2: SetStampVersion
        uint64_t m_StructuralVersion = 1;
        uint64_t m_PresenceMaskVersion = 0;      // GetPresenceMaskVersion
        bool m_InPhase = false;
        uint32_t m_ExclusiveDepth = 0;
        WriteAccessCheck m_WriteAccessCheck = nullptr;   // E2.2: SetWriteAccessCheck
        void* m_WriteAccessContext = nullptr;
        uint32_t m_ObserverDepth = 0;            // E1.4: observers running
        std::atomic<uint64_t> m_ValidationErrors{ 0 };  // reports may come from parallel systems
        std::atomic<uint64_t> m_BufferHeapBytes{ 0 };   // ResizeBuffer may run in parallel
        uint64_t m_CommandsPlayedBack = 0;       // E1.3
        uint64_t m_ObserverBatches = 0;          // E1.4
        Counters m_Counters;
        CommandPlayback* m_CommandPlayback = nullptr;   // E1.3: GetCommandPlayback
        ObserverRegistry* m_ObserverRegistry = nullptr; // E1.4: made by the first AddObserver
        uint32_t m_ObserverCount = 0;            // E1.4: registered observers; the seams do nothing while 0

        // Scratch storage, kept between calls so that steady-state operations allocate nothing.
        struct SortedTerm
        {
            ComponentId component;
            uint32_t source;   // index in the caller's list
        };
        Vector<SortedTerm> m_ScratchTerms;
        Vector<ComponentId> m_ScratchIds;
        Vector<const void*> m_ScratchShared;
        Vector<uint32_t> m_ScratchValueIndex;
        Vector<RowRun> m_ScratchRuns;
        Vector<Entity> m_ScratchEntities;
        Vector<uint32_t> m_ScratchTypeMap;
        struct RemapEntry
        {
            Entity source;
            Entity copy;
            uint32_t order;   // position in the caller's list: a source listed twice maps to its first copy
            uint32_t reserved;
        };
        Vector<RemapEntry> m_ScratchRemap;
        Archetype m_ScratchArchetype;   // RowFits lays out a row here without making an archetype
        uint8_t* m_ScratchBlock = nullptr;
        size_t m_ScratchBlockSize = 0;
    };

    // The world behind an IWorld of this module; null for a world of another module.
    inline WorldImpl* GetWorldImpl(IWorld* world) noexcept { return donut::query_cast<WorldImpl>(world); }

    template <typename Fill>
    nvrhi::FRESULT WorldImpl::AppendRows(Archetype& archetype, uint32_t count, uint64_t version, Fill&& fill) noexcept
    {
        if (count == 0)
            return nvrhi::FS_OK;
        // Every chunk is allocated before a row is written, so that running out of memory changes nothing.
        if (!ReserveRows(archetype, count, version))
            return nvrhi::FE_OUT_OF_MEMORY;
        const uint32_t rows = archetype.rowsPerChunk;
        uint32_t position = archetype.entityCount;
        uint32_t done = 0;
        while (done < count)
        {
            const uint32_t chunk = position / rows;
            const uint32_t row = position % rows;
            const uint32_t run = count - done < rows - row ? count - done : rows - row;
            if (archetype.rowVersionSlots != 0)
                StampPlacedRows(archetype, chunk, row, run, version);
            archetype.chunks[chunk].count += run;
            StampChunk(archetype, chunk, version);
            fill(chunk, row, run, done);
            done += run;
            position += run;
        }
        archetype.entityCount += count;
        ++m_StructuralVersion;
        return nvrhi::FS_OK;
    }
}
