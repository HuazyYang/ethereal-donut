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

// World snapshots and the world hash (aggregate docs/ecs-design.md 11.1, 11.4; plan step E1.5).
//
// Both cover the serialisable components: Relocatable and not NoSerialize (EcsWorld.h), resources included. Other
// components are invisible to them: an entity's archetype, as they see it, is its serialisable components.
//
// The world hash (IWorld::ComputeHash) is XXH64 with seed 0 over the canonical byte stream below; integers are
// little-endian, a GUID is the 16 bytes of nvrhi::GUID as stored, and "field bytes" are the byte ranges a component's
// fields cover, merged and ascending (ComponentType::fieldBytes): padding never takes part, and a component that lists
// no field contributes its presence only. Lists of components are in GUID order (memcmp of the 16 bytes).
//
//     the 16 bytes of c_WorldHashTag
//     u32 entity table size (index 0 included; it is never used)
//     for every index from 1, ascending:
//         u32 generation, u8 state (0 free, 1 reserved, 2 alive)
//         alive only:
//             u64 archetype digest: XXH64 of { u32 count; per serialisable component: GUID, then the field bytes of
//                 its value if it is Shared }
//             per serialisable component of the archetype that has a value per row (Data, Buffer) or per chunk
//             (Chunk), or is Enableable:
//                 Data: the field bytes of the value
//                 Buffer: u32 element count, then the field bytes of every element
//                 Chunk: the field bytes of the value of the entity's chunk, which GetComponent returns for it
//                 Enableable: u8 enabled (after the value, if any)
//             u32 count of the entity's serialisable Sparse components; per component: GUID, field bytes of the value
//     the free list from head to tail as u32 indices, then u32 0
//     u32 count of serialisable resources; per resource: GUID, field bytes of the value
//
// So the hash is a function of what per-entity reads return, plus the order in which ids will be reused. Archetype
// indices and creation order, chunk sizes, row order and sparse dense order are not in it, nor are component ids
// (registration order), versions, buffer capacities and non-serialisable components. Two worlds whose rows are placed
// in other chunks hash differently exactly when that gives an entity another Chunk value.
//
// The snapshot (SaveSnapshot, LoadSnapshot), version 1, little-endian:
//
//     header      "DONUTECS", u32 version (1), u32 flags (0), u64 payload bytes
//     payload     components   u32 count; per component, in GUID order: GUID, u32 flags, u32 size, u32 alignment,
//                              u32 bufferInlineCapacity, u32 fieldCount, per field { u32 offset, u32 scalarCount,
//                              u8 type, u8 shape, u16 0 }
//                 entities     u32 table size; per index from 1: u32 generation, u32 state (0 free, 1 reserved,
//                              2 alive); u32 free count; the free list as u32 indices, head to tail
//                 archetypes   u32 count; per non-empty archetype, in creation order:
//                                  u32 component count, u32 component-table indices (ascending)
//                                  u32 rows, u32 rows per chunk (in the saved world)
//                                  u64 entity per row, in chunk and row order
//                                  per component: Shared: the value; Data: a value per row; Chunk: a value per chunk;
//                                  Buffer: u32 count per row, then every row's elements; Tag: nothing; and then, if
//                                  it is Enableable, u8 enabled per row
//                 sparse       u32 count; per non-empty Sparse component, in table order: u32 table index, u32 count,
//                              u64 entities and the values in dense order
//                 resources    u32 count; per serialisable resource, in table order: u32 table index, the value
//     checksum    u64 XXH64 of the header and the payload
//
// A value is `size` bytes, with the bytes no field covers written as 0 when the component lists fields (a component
// without fields is written as it is). The component table lists the components the content uses. A snapshot is a
// function of the world's content and its placement (archetype order, row order, sparse dense order), not of
// registration order or addresses.
//
// LoadSnapshot reads exactly the snapshot's bytes, so a stream may carry more after it (simulation states, 8.5). It
// validates everything before it changes anything: the checksum, the structure, every component against the one
// registered here with its GUID (flags, size, alignment, inline capacity, fields), every entity id against the
// table, that rows fit this world's chunks. Then it prepares every allocation that may fail (archetypes; the chunk
// pages the pools lack, reserved as free pages; spilled buffers; sparse storage) and grows the containers, and only
// then replaces the content, which allocates nothing and can no longer fail. So a failed load changes no content and
// no version (out of memory may leave new empty archetypes, with a new structural version, and zeroed sparse pages
// and presence masks behind); only the containers of the storage stop the process when they run out of memory, as
// everywhere in it.

#include "EcsSnapshot.h"
#include "EcsWorldImpl.h"

#include <algorithm>
#include <cstring>
#include <type_traits>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "ECS snapshots and world hashes copy component bytes as they are: big-endian hosts are not supported"
#endif

namespace donut::engine::ecs
{
    // ------------------------------------------------------------------------------------------------- XXH64

    namespace
    {
        constexpr uint64_t c_Prime1 = 0x9E3779B185EBCA87ull;
        constexpr uint64_t c_Prime2 = 0xC2B2AE3D27D4EB4Full;
        constexpr uint64_t c_Prime3 = 0x165667B19E3779F9ull;
        constexpr uint64_t c_Prime4 = 0x85EBCA77C2B2AE63ull;
        constexpr uint64_t c_Prime5 = 0x27D4EB2F165667C5ull;

        inline uint64_t Rotl(uint64_t x, int r) noexcept { return (x << r) | (x >> (64 - r)); }

        inline uint32_t LoadU32(const uint8_t* p) noexcept
        {
            uint32_t value;
            std::memcpy(&value, p, sizeof(value));
            return value;
        }

        inline uint64_t LoadU64(const uint8_t* p) noexcept
        {
            uint64_t value;
            std::memcpy(&value, p, sizeof(value));
            return value;
        }

        inline void StoreU32(uint8_t* p, uint32_t value) noexcept { std::memcpy(p, &value, sizeof(value)); }
        inline void StoreU64(uint8_t* p, uint64_t value) noexcept { std::memcpy(p, &value, sizeof(value)); }

        inline uint64_t XxhRound(uint64_t accumulator, uint64_t input) noexcept
        {
            accumulator += input * c_Prime2;
            accumulator = Rotl(accumulator, 31);
            return accumulator * c_Prime1;
        }

        inline uint64_t XxhMerge(uint64_t accumulator, uint64_t lane) noexcept
        {
            accumulator ^= XxhRound(0, lane);
            return accumulator * c_Prime1 + c_Prime4;
        }
    }

    Hasher64::Hasher64(uint64_t seed) noexcept
        : m_Seed(seed)
    {
        m_Lanes[0] = seed + c_Prime1 + c_Prime2;
        m_Lanes[1] = seed + c_Prime2;
        m_Lanes[2] = seed;
        m_Lanes[3] = seed - c_Prime1;
        std::memset(m_Buffer, 0, sizeof(m_Buffer));
    }

    void Hasher64::Update(const void* data, size_t size) noexcept
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        m_Total += size;
        if (m_Buffered + size < sizeof(m_Buffer))
        {
            if (size != 0)
                std::memcpy(m_Buffer + m_Buffered, bytes, size);
            m_Buffered += uint32_t(size);
            return;
        }
        if (m_Buffered != 0)
        {
            const size_t fill = sizeof(m_Buffer) - m_Buffered;
            std::memcpy(m_Buffer + m_Buffered, bytes, fill);
            for (int lane = 0; lane < 4; ++lane)
                m_Lanes[lane] = XxhRound(m_Lanes[lane], LoadU64(m_Buffer + 8 * lane));
            bytes += fill;
            size -= fill;
            m_Buffered = 0;
        }
        while (size >= sizeof(m_Buffer))
        {
            for (int lane = 0; lane < 4; ++lane)
                m_Lanes[lane] = XxhRound(m_Lanes[lane], LoadU64(bytes + 8 * lane));
            bytes += sizeof(m_Buffer);
            size -= sizeof(m_Buffer);
        }
        if (size != 0)
            std::memcpy(m_Buffer, bytes, size);
        m_Buffered = uint32_t(size);
    }

    uint64_t Hasher64::Digest() const noexcept
    {
        uint64_t hash;
        if (m_Total >= sizeof(m_Buffer))
        {
            hash = Rotl(m_Lanes[0], 1) + Rotl(m_Lanes[1], 7) + Rotl(m_Lanes[2], 12) + Rotl(m_Lanes[3], 18);
            for (int lane = 0; lane < 4; ++lane)
                hash = XxhMerge(hash, m_Lanes[lane]);
        }
        else
        {
            hash = m_Seed + c_Prime5;
        }
        hash += m_Total;

        const uint8_t* bytes = m_Buffer;
        uint32_t left = m_Buffered;
        while (left >= 8)
        {
            hash ^= XxhRound(0, LoadU64(bytes));
            hash = Rotl(hash, 27) * c_Prime1 + c_Prime4;
            bytes += 8;
            left -= 8;
        }
        if (left >= 4)
        {
            hash ^= uint64_t(LoadU32(bytes)) * c_Prime1;
            hash = Rotl(hash, 23) * c_Prime2 + c_Prime3;
            bytes += 4;
            left -= 4;
        }
        while (left > 0)
        {
            hash ^= uint64_t(*bytes) * c_Prime5;
            hash = Rotl(hash, 11) * c_Prime1;
            ++bytes;
            --left;
        }
        hash ^= hash >> 33;
        hash *= c_Prime2;
        hash ^= hash >> 29;
        hash *= c_Prime3;
        hash ^= hash >> 32;
        return hash;
    }

    void Hasher64::U32(uint32_t value) noexcept
    {
        uint8_t bytes[4];
        StoreU32(bytes, value);
        Update(bytes, sizeof(bytes));
    }

    void Hasher64::U64(uint64_t value) noexcept
    {
        uint8_t bytes[8];
        StoreU64(bytes, value);
        Update(bytes, sizeof(bytes));
    }

    uint64_t Xxh64(const void* data, size_t size, uint64_t seed) noexcept
    {
        Hasher64 hasher(seed);
        hasher.Update(data, size);
        return hasher.Digest();
    }

    // --------------------------------------------------------------------------------------------- helpers

    namespace
    {
        bool GuidLess(const nvrhi::GUID& a, const nvrhi::GUID& b) noexcept
        {
            return std::memcmp(&a, &b, sizeof(nvrhi::GUID)) < 0;
        }

        bool TypeGuidLess(const ComponentType* a, const ComponentType* b) noexcept
        {
            return GuidLess(a->desc.uuid, b->desc.uuid);
        }

        bool Serialisable(const ComponentType& type) noexcept
        {
            return ecs::IsSerialisable(type.desc.flags);
        }

        // Every byte of a value is a field byte: values can be hashed and written in bulk.
        bool IsDense(const ComponentType& type) noexcept
        {
            return type.fieldBytes.size() == 1 && type.fieldBytes[0].offset == 0 &&
                   type.fieldBytes[0].size == type.desc.size;
        }

        void HashFieldBytes(Hasher64& hasher, const ComponentType& type, const uint8_t* values, uint32_t count) noexcept
        {
            if (count == 0 || type.fieldBytes.empty())
                return;
            if (IsDense(type))
            {
                hasher.Update(values, size_t(type.desc.size) * count);
                return;
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                const uint8_t* value = values + size_t(i) * type.desc.size;
                for (const ByteRange& range : type.fieldBytes)
                    hasher.Update(value + range.offset, range.size);
            }
        }

        SnapshotEntityState StateOf(const EntityRecord& record) noexcept
        {
            if (record.archetype == c_FreeSlot)
                return SnapshotEntityState::Free;
            if (record.archetype >= c_ClaimedSlot)
                return SnapshotEntityState::Reserved;   // a claimed id is reserved but for the CreateEntities using it
            return SnapshotEntityState::Alive;
        }

        // A growable array of trivially copyable values whose growth fails softly, unlike ecs::Vector's (NVRHI_VERIFY):
        // a load keeps its scratch here, so that it can fail with FE_OUT_OF_MEMORY and change nothing.
        template <typename T>
        class SoftArray
        {
            static_assert(std::is_trivially_copyable_v<T>, "SoftArray holds trivially copyable values");

        public:
            SoftArray() noexcept = default;
            SoftArray(const SoftArray&) = delete;
            SoftArray& operator=(const SoftArray&) = delete;
            ~SoftArray() { HeapFree(m_Data); }

            bool Reserve(size_t capacity) noexcept
            {
                if (capacity <= m_Capacity)
                    return true;
                if (capacity > SIZE_MAX / sizeof(T))
                    return false;
                T* data = static_cast<T*>(HeapAllocate(capacity * sizeof(T), alignof(T)));
                if (!data)
                    return false;
                if (m_Size != 0)
                    std::memcpy(static_cast<void*>(data), m_Data, m_Size * sizeof(T));
                HeapFree(m_Data);
                m_Data = data;
                m_Capacity = capacity;
                return true;
            }
            bool Assign(size_t size, const T& value) noexcept
            {
                if (!Reserve(size))
                    return false;
                for (size_t i = 0; i < size; ++i)
                    m_Data[i] = value;
                m_Size = size;
                return true;
            }
            bool Push(const T& value) noexcept
            {
                if (m_Size == m_Capacity && !Reserve(m_Capacity < 8 ? 16 : m_Capacity * 2))
                    return false;
                m_Data[m_Size++] = value;
                return true;
            }
            void Clear() noexcept { m_Size = 0; }
            // Within the capacity; the new values are whatever the storage holds.
            void SetSize(size_t size) noexcept
            {
                NVRHI_ASSERT(size <= m_Capacity);
                m_Size = size;
            }

            T* Data() noexcept { return m_Data; }
            size_t Size() const noexcept { return m_Size; }
            size_t Capacity() const noexcept { return m_Capacity; }
            T& operator[](size_t i) noexcept { return m_Data[i]; }
            const T& operator[](size_t i) const noexcept { return m_Data[i]; }
            T* begin() noexcept { return m_Data; }
            T* end() noexcept { return m_Data + m_Size; }

        private:
            T* m_Data = nullptr;
            size_t m_Size = 0;
            size_t m_Capacity = 0;
        };

        // Bounds-checked reads of the payload. Counts are checked against the bytes that remain before they are used,
        // so that no corrupt count can make the load allocate or read beyond the snapshot.
        class Cursor
        {
        public:
            Cursor(const uint8_t* data, uint64_t size) noexcept : m_Position(data), m_End(data + size) { }

            uint64_t Remaining() const noexcept { return uint64_t(m_End - m_Position); }
            bool AtEnd() const noexcept { return m_Position == m_End; }

            bool Take(uint64_t bytes, const uint8_t*& out) noexcept
            {
                if (bytes > Remaining())
                    return false;
                out = m_Position;
                m_Position += bytes;
                return true;
            }
            bool TakeArray(uint64_t count, uint64_t elementBytes, const uint8_t*& out) noexcept
            {
                if (elementBytes != 0 && count > Remaining() / elementBytes)
                    return false;
                return Take(count * elementBytes, out);
            }
            bool U32(uint32_t& value) noexcept
            {
                const uint8_t* bytes = nullptr;
                if (!Take(sizeof(value), bytes))
                    return false;
                value = LoadU32(bytes);
                return true;
            }

        private:
            const uint8_t* m_Position;
            const uint8_t* m_End;
        };

        // The first pass of a save: the payload's size, for the header.
        struct CountingSink
        {
            uint64_t bytes = 0;
            void Put(const void*, size_t size) noexcept { bytes += size; }
        };

        // The second pass: buffered writes, and the checksum of every byte that passes.
        class WriterSink
        {
        public:
            WriterSink(IStateWriter* writer, uint8_t* buffer, size_t capacity) noexcept
                : m_Writer(writer), m_Buffer(buffer), m_Capacity(capacity) { }

            void Put(const void* data, size_t size) noexcept
            {
                if (size == 0 || NVRHI_FAILED(m_Result))
                    return;
                m_Hasher.Update(data, size);
                m_Bytes += size;
                if (size >= m_Capacity)
                {
                    Flush();
                    if (NVRHI_SUCCEEDED(m_Result))
                        m_Result = m_Writer->Write(data, size);
                    return;
                }
                if (m_Used + size > m_Capacity)
                    Flush();
                std::memcpy(m_Buffer + m_Used, data, size);
                m_Used += size;
            }
            void Flush() noexcept
            {
                if (m_Used != 0 && NVRHI_SUCCEEDED(m_Result))
                    m_Result = m_Writer->Write(m_Buffer, m_Used);
                m_Used = 0;
            }
            uint64_t Digest() const noexcept { return m_Hasher.Digest(); }
            uint64_t Bytes() const noexcept { return m_Bytes; }
            nvrhi::FRESULT Result() const noexcept { return m_Result; }

        private:
            IStateWriter* m_Writer;
            uint8_t* m_Buffer;
            size_t m_Capacity;
            size_t m_Used = 0;
            uint64_t m_Bytes = 0;
            Hasher64 m_Hasher;
            nvrhi::FRESULT m_Result = nvrhi::FS_OK;
        };

        template <typename Sink>
        void PutU8(Sink& sink, uint8_t value) noexcept { sink.Put(&value, 1); }

        template <typename Sink>
        void PutU32(Sink& sink, uint32_t value) noexcept
        {
            uint8_t bytes[4];
            StoreU32(bytes, value);
            sink.Put(bytes, sizeof(bytes));
        }

        // `count` consecutive values; the bytes no field covers are written as 0 (through `scratch`, one value long),
        // unless the component lists no field.
        template <typename Sink>
        void PutValues(Sink& sink, const ComponentType& type, const uint8_t* values, uint32_t count,
                       uint8_t* scratch) noexcept
        {
            const uint32_t size = type.desc.size;
            if (count == 0 || size == 0)
                return;
            if (type.fieldBytes.empty() || IsDense(type))
            {
                sink.Put(values, size_t(size) * count);
                return;
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                const uint8_t* value = values + size_t(i) * size;
                std::memset(scratch, 0, size);
                for (const ByteRange& range : type.fieldBytes)
                    std::memcpy(scratch + range.offset, value + range.offset, range.size);
                sink.Put(scratch, size);
            }
        }

        constexpr uint32_t c_NotInTable = 0xffffffffu;
        constexpr uint32_t c_WriteBufferBytes = 64 * 1024;
        constexpr uint64_t c_ReadPieceBytes = 1024 * 1024;
        constexpr uint64_t c_FieldEntryBytes = 12;
        constexpr uint64_t c_TypeEntryBytes = 36;   // GUID and five u32, without the fields

        // Load scratch: views into the payload, and what the preparation allocated.
        struct LoadColumn
        {
            const ComponentType* type;
            const uint8_t* values;      // Shared: the value. Data: one per row. Chunk: one per saved chunk. Buffer: the
                                        // elements of every row, in row order
            const uint8_t* counts;      // Buffer: u32 per row
            const uint8_t* enabled;     // Enableable: u8 per row
            uint32_t column;            // in the target archetype
            uint32_t firstSpill;        // Buffer: the first of its rows' heap blocks in LoadImage::spills
            uint64_t elementCursor;     // Buffer, while rows are filled: elements consumed
            uint32_t spillCursor;       // Buffer, while rows are filled: heap blocks consumed
            uint32_t reserved;
        };

        struct LoadRecord
        {
            uint32_t firstColumn;
            uint32_t columnCount;
            uint32_t rowCount;
            uint32_t rowsPerChunk;      // of the saved world: where its Chunk values belong
            const uint8_t* entities;
            uint32_t target;            // the archetype that receives the rows
            uint32_t firstRow;          // where they start in it, once the load emptied it
        };

        struct LoadSparse
        {
            const ComponentType* type;
            SparseSet* set;
            const uint8_t* entities;
            const uint8_t* values;
            uint8_t* newValues;         // a larger value array, when the set's is too small; the commit installs it
            uint32_t count;
            uint32_t newCapacity;
        };

        struct LoadResource
        {
            const ComponentType* type;
            const uint8_t* value;
        };

        struct LoadImage
        {
            SoftArray<const ComponentType*> types;   // the snapshot's component table, as the local types
            uint32_t tableSize = 0;
            const uint8_t* table = nullptr;          // (u32 generation, u32 state) per index from 1
            const uint8_t* freeList = nullptr;
            uint32_t freeCount = 0;
            uint32_t aliveCount = 0;
            SoftArray<LoadRecord> records;
            SoftArray<LoadColumn> columns;
            SoftArray<LoadSparse> sparse;
            SoftArray<LoadResource> resources;
            SoftArray<uint8_t> marks;                // per entity index, during validation
            SoftArray<uint32_t> archetypeRows;       // per archetype: rows it receives
            SoftArray<uint32_t> targetOf;            // per entity index: the archetype that receives its row
            SoftArray<uint32_t> positionOf;          // per entity index: the row's position in that archetype
            SoftArray<void*> spills;                 // heap blocks of the buffers that hold more than their inline
                                                     // capacity, in record, column and row order
            SoftArray<ComponentId> ids;              // FindOrCreateArchetype's arguments
            SoftArray<const void*> shared;
            SoftArray<uint32_t> order;

            uint32_t Generation(uint32_t index) const noexcept { return LoadU32(table + size_t(index - 1) * 8); }
            SnapshotEntityState State(uint32_t index) const noexcept
            {
                return SnapshotEntityState(LoadU32(table + size_t(index - 1) * 8 + 4));
            }
        };

        uint64_t ChunksFor(uint64_t rows, uint32_t rowsPerChunk) noexcept
        {
            return (rows + rowsPerChunk - 1) / rowsPerChunk;
        }
    }

    // ----------------------------------------------------------------------------------------- WorldSnapshot

    // The hash, save and load of a world; a friend of WorldImpl (EcsWorldImpl.h).
    class WorldSnapshot
    {
    public:
        static uint64_t Hash(const WorldImpl& world) noexcept;
        static nvrhi::FRESULT Save(const WorldImpl& world, IStateWriter* writer) noexcept;
        static nvrhi::FRESULT Load(WorldImpl& world, IStateReader* reader) noexcept;

    private:
        struct SaveTables
        {
            Vector<const ComponentType*> types;     // the component table, in GUID order
            Vector<uint32_t> tableIndex;            // by component id; c_NotInTable
            Vector<uint32_t> records;               // the non-empty archetypes, in creation order
            Vector<uint32_t> recordColumns;         // per record: its serialisable columns, in table order
            Vector<uint32_t> recordColumnStart;     // per record, and one past the last
            Vector<const SparseSet*> sparse;        // the non-empty serialisable sparse sets, in table order
            Vector<const ComponentType*> resources; // in table order
            uint32_t freeCount = 0;
            uint32_t scratchBytes = 0;              // the largest value written through the padding scratch
        };

        static void BuildSaveTables(const WorldImpl& world, SaveTables& tables) noexcept;
        template <typename Sink>
        static void WritePayload(const WorldImpl& world, const SaveTables& tables, Sink& sink,
                                 uint8_t* scratch) noexcept;

        static nvrhi::FRESULT ReadSnapshot(WorldImpl& world, IStateReader* reader, SoftArray<uint8_t>& payload,
                                           uint64_t& payloadBytes) noexcept;
        static nvrhi::FRESULT Parse(WorldImpl& world, const uint8_t* payload, uint64_t payloadBytes,
                                    LoadImage& image) noexcept;
        static nvrhi::FRESULT ParseTypes(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept;
        static nvrhi::FRESULT ParseEntities(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept;
        static nvrhi::FRESULT ParseArchetypes(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept;
        static nvrhi::FRESULT ParseSparseAndResources(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept;
        static nvrhi::FRESULT Prepare(WorldImpl& world, LoadImage& image) noexcept;
        static nvrhi::FRESULT PreparePages(WorldImpl& world, LoadImage& image) noexcept;
        static nvrhi::FRESULT PrepareSparse(WorldImpl& world, LoadImage& image) noexcept;
        static void ReleasePrepared(LoadImage& image) noexcept;
        static void ClearContent(WorldImpl& world) noexcept;
        static nvrhi::FRESULT Commit(WorldImpl& world, LoadImage& image) noexcept;
        static void FillRows(WorldImpl& world, LoadImage& image, LoadRecord& record, uint32_t chunk, uint32_t row,
                             uint32_t run, uint32_t done) noexcept;

        static nvrhi::FRESULT Corrupt(WorldImpl& world, const char* what) noexcept
        {
            world.ReportValidation("LoadSnapshot", "the snapshot is corrupt: %s", what);
            return nvrhi::FE_INVALID_ARGS;
        }
    };

    // ------------------------------------------------------------------------------------------------- hash

    uint64_t WorldSnapshot::Hash(const WorldImpl& world) noexcept
    {
        // Per archetype: the digest of its serialisable components, and those of them with a value or a bit per row.
        struct ArchetypeHash
        {
            uint64_t digest;
            uint32_t firstColumn;
            uint32_t columnCount;
        };
        Vector<ArchetypeHash> archetypes;
        Vector<uint32_t> rowColumns;
        Vector<uint32_t> sorted;
        archetypes.reserve(world.m_Archetypes.size());
        for (const Archetype* archetype : world.m_Archetypes)
        {
            sorted.clear();
            for (uint32_t c = 0; c < archetype->columns.size(); ++c)
                if (Serialisable(*archetype->columns[c].type))
                    sorted.push_back(c);
            std::sort(sorted.begin(), sorted.end(), [archetype](uint32_t a, uint32_t b)
            {
                return TypeGuidLess(archetype->columns[a].type, archetype->columns[b].type);
            });
            Hasher64 digest;
            digest.U32(uint32_t(sorted.size()));
            ArchetypeHash entry{ 0, uint32_t(rowColumns.size()), 0 };
            for (const uint32_t c : sorted)
            {
                const ArchetypeColumn& column = archetype->columns[c];
                digest.Update(&column.type->desc.uuid, sizeof(nvrhi::GUID));
                if (column.kind == StorageKind::Shared)
                    HashFieldBytes(digest, *column.type, archetype->sharedValues + column.offset, 1);
                if (column.kind == StorageKind::Data || column.kind == StorageKind::Buffer ||
                    column.kind == StorageKind::Chunk || column.enableable)
                    rowColumns.push_back(c);
            }
            entry.digest = digest.Digest();
            entry.columnCount = uint32_t(rowColumns.size()) - entry.firstColumn;
            archetypes.push_back(entry);
        }

        Vector<const SparseSet*> sparse;
        for (const SparseSet* set : world.m_SparseSets)
            if (Serialisable(*set->type))
                sparse.push_back(set);
        std::sort(sparse.begin(), sparse.end(), [](const SparseSet* a, const SparseSet* b)
        {
            return TypeGuidLess(a->type, b->type);
        });
        Vector<const ComponentType*> resources;
        for (const ComponentType* type : world.m_Types)
            if (type->kind == StorageKind::Resource && Serialisable(*type))
                resources.push_back(type);
        std::sort(resources.begin(), resources.end(), TypeGuidLess);

        Hasher64 hasher;
        hasher.Update(c_WorldHashTag, sizeof(c_WorldHashTag));
        const uint32_t tableSize = uint32_t(world.m_Entities.size());
        hasher.U32(tableSize);
        for (uint32_t index = 1; index < tableSize; ++index)
        {
            const EntityRecord& record = world.m_Entities[index];
            const SnapshotEntityState state = StateOf(record);
            hasher.U32(record.generation);
            hasher.U8(uint8_t(state));
            if (state != SnapshotEntityState::Alive)
                continue;

            const Archetype& archetype = *world.m_Archetypes[record.archetype];
            const ArchetypeHash& entry = archetypes[record.archetype];
            hasher.U64(entry.digest);
            for (uint32_t k = 0; k < entry.columnCount; ++k)
            {
                const uint32_t c = rowColumns[entry.firstColumn + k];
                const ArchetypeColumn& column = archetype.columns[c];
                const uint8_t* data = static_cast<const uint8_t*>(archetype.ColumnData(record.chunk, c));
                if (column.kind == StorageKind::Data)
                {
                    HashFieldBytes(hasher, *column.type, data + size_t(record.row) * column.stride, 1);
                }
                else if (column.kind == StorageKind::Buffer)
                {
                    const BufferHeader& header = reinterpret_cast<const BufferHeader*>(data)[record.row];
                    hasher.U32(header.count);
                    HashFieldBytes(hasher, *column.type, static_cast<const uint8_t*>(header.data), header.count);
                }
                else if (column.kind == StorageKind::Chunk)
                {
                    HashFieldBytes(hasher, *column.type, data, 1);   // the chunk's value, as GetComponent reads it
                }
                if (column.enableable)
                    hasher.U8(GetMaskBit(archetype.EnabledMask(record.chunk, c), record.row) ? 1 : 0);
            }

            uint32_t sparseCount = 0;
            if (record.sparseCount != 0)
                for (const SparseSet* set : sparse)
                    if (set->Find(index) != HashIndex::c_NotFound)
                        ++sparseCount;
            hasher.U32(sparseCount);
            if (sparseCount != 0)
            {
                for (const SparseSet* set : sparse)
                {
                    const uint32_t dense = set->Find(index);
                    if (dense == HashIndex::c_NotFound)
                        continue;
                    hasher.Update(&set->type->desc.uuid, sizeof(nvrhi::GUID));
                    if (set->type->desc.size != 0)
                        HashFieldBytes(hasher, *set->type, static_cast<const uint8_t*>(set->Value(dense)), 1);
                }
            }
        }

        uint32_t steps = 0;
        for (uint32_t index = world.m_FreeHead; index != 0 && steps < tableSize; index = world.m_Entities[index].chunk)
        {
            hasher.U32(index);
            ++steps;
        }
        hasher.U32(0);

        hasher.U32(uint32_t(resources.size()));
        for (const ComponentType* type : resources)
        {
            hasher.Update(&type->desc.uuid, sizeof(nvrhi::GUID));
            HashFieldBytes(hasher, *type, static_cast<const uint8_t*>(type->value), 1);
        }
        return hasher.Digest();
    }

    // ------------------------------------------------------------------------------------------------- save

    void WorldSnapshot::BuildSaveTables(const WorldImpl& world, SaveTables& tables) noexcept
    {
        // The components the content uses: in a non-empty archetype, a non-empty sparse set, or a resource.
        const uint32_t typeCount = uint32_t(world.m_Types.size());
        tables.tableIndex.assign(typeCount, c_NotInTable);
        Vector<uint8_t> used;
        used.assign(typeCount, 0);
        for (const Archetype* archetype : world.m_Archetypes)
            if (archetype->entityCount != 0)
                for (const ArchetypeColumn& column : archetype->columns)
                    used[uint32_t(column.component)] = 1;
        for (const SparseSet* set : world.m_SparseSets)
            if (set->GetCount() != 0)
                used[uint32_t(set->type->id)] = 1;
        for (const ComponentType* type : world.m_Types)
        {
            if (type->kind == StorageKind::Resource)
                used[uint32_t(type->id)] = 1;
            if (used[uint32_t(type->id)] && Serialisable(*type))
                tables.types.push_back(type);
        }
        std::sort(tables.types.begin(), tables.types.end(), TypeGuidLess);
        for (uint32_t i = 0; i < tables.types.size(); ++i)
        {
            const ComponentType& type = *tables.types[i];
            tables.tableIndex[uint32_t(type.id)] = i;
            if (!type.fieldBytes.empty() && !IsDense(type))
                tables.scratchBytes = std::max(tables.scratchBytes, type.desc.size);
            if (type.kind == StorageKind::Sparse && world.m_SparseSets[type.sparseSet]->GetCount() != 0)
                tables.sparse.push_back(world.m_SparseSets[type.sparseSet]);
            else if (type.kind == StorageKind::Resource)
                tables.resources.push_back(&type);
        }

        for (const Archetype* archetype : world.m_Archetypes)
        {
            if (archetype->entityCount == 0)
                continue;
            tables.records.push_back(archetype->index);
            tables.recordColumnStart.push_back(uint32_t(tables.recordColumns.size()));
            const size_t first = tables.recordColumns.size();
            for (uint32_t c = 0; c < archetype->columns.size(); ++c)
                if (tables.tableIndex[uint32_t(archetype->components[c])] != c_NotInTable)
                    tables.recordColumns.push_back(c);
            std::sort(tables.recordColumns.begin() + first, tables.recordColumns.end(), [&](uint32_t a, uint32_t b)
            {
                return tables.tableIndex[uint32_t(archetype->components[a])] <
                       tables.tableIndex[uint32_t(archetype->components[b])];
            });
        }
        tables.recordColumnStart.push_back(uint32_t(tables.recordColumns.size()));

        for (uint32_t index = 1; index < world.m_Entities.size(); ++index)
            if (world.m_Entities[index].archetype == c_FreeSlot)
                ++tables.freeCount;
    }

    template <typename Sink>
    void WorldSnapshot::WritePayload(const WorldImpl& world, const SaveTables& tables, Sink& sink,
                                     uint8_t* scratch) noexcept
    {
        // The component table.
        PutU32(sink, uint32_t(tables.types.size()));
        for (const ComponentType* type : tables.types)
        {
            const ComponentTypeDesc& desc = type->desc;
            sink.Put(&desc.uuid, sizeof(nvrhi::GUID));
            // RowTracked is the world's storage option (EcsTypes.h): row versions are versions, not values (E7.5).
            PutU32(sink, uint32_t(desc.flags & ~ComponentFlags::RowTracked));
            PutU32(sink, desc.size);
            PutU32(sink, desc.alignment);
            PutU32(sink, desc.bufferInlineCapacity);
            PutU32(sink, desc.fieldCount);
            for (uint32_t f = 0; f < desc.fieldCount; ++f)
            {
                const FieldDesc& field = desc.fields[f];
                PutU32(sink, field.offset);
                PutU32(sink, field.scalarCount);
                PutU8(sink, uint8_t(field.type));
                PutU8(sink, uint8_t(field.shape));
                PutU8(sink, 0);
                PutU8(sink, 0);
            }
        }

        // The entity table and the free list.
        const uint32_t tableSize = uint32_t(world.m_Entities.size());
        PutU32(sink, tableSize);
        for (uint32_t index = 1; index < tableSize; ++index)
        {
            const EntityRecord& record = world.m_Entities[index];
            PutU32(sink, record.generation);
            PutU32(sink, uint32_t(StateOf(record)));
        }
        PutU32(sink, tables.freeCount);
        uint32_t steps = 0;
        for (uint32_t index = world.m_FreeHead; index != 0 && steps < tables.freeCount;
             index = world.m_Entities[index].chunk)
        {
            PutU32(sink, index);
            ++steps;
        }

        // The archetypes' rows.
        PutU32(sink, uint32_t(tables.records.size()));
        for (uint32_t r = 0; r < tables.records.size(); ++r)
        {
            const Archetype& archetype = *world.m_Archetypes[tables.records[r]];
            const uint32_t firstColumn = tables.recordColumnStart[r];
            const uint32_t columnCount = tables.recordColumnStart[r + 1] - firstColumn;
            PutU32(sink, columnCount);
            for (uint32_t k = 0; k < columnCount; ++k)
            {
                const uint32_t c = tables.recordColumns[firstColumn + k];
                PutU32(sink, tables.tableIndex[uint32_t(archetype.components[c])]);
            }
            PutU32(sink, archetype.entityCount);
            PutU32(sink, archetype.rowsPerChunk);
            const uint32_t chunkCount = archetype.GetChunkCount();
            for (uint32_t chunk = 0; chunk < chunkCount; ++chunk)
                sink.Put(archetype.Entities(chunk), size_t(archetype.chunks[chunk].count) * sizeof(Entity));

            for (uint32_t k = 0; k < columnCount; ++k)
            {
                const uint32_t c = tables.recordColumns[firstColumn + k];
                const ArchetypeColumn& column = archetype.columns[c];
                const ComponentType& type = *column.type;
                switch (column.kind)
                {
                case StorageKind::Shared:
                    PutValues(sink, type, archetype.sharedValues + column.offset, 1, scratch);
                    break;
                case StorageKind::Data:
                    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk)
                        PutValues(sink, type, static_cast<const uint8_t*>(archetype.ColumnData(chunk, c)),
                                  archetype.chunks[chunk].count, scratch);
                    break;
                case StorageKind::Chunk:
                    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk)
                        PutValues(sink, type, static_cast<const uint8_t*>(archetype.ColumnData(chunk, c)), 1, scratch);
                    break;
                case StorageKind::Buffer:
                    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk)
                    {
                        const BufferHeader* headers = static_cast<const BufferHeader*>(archetype.ColumnData(chunk, c));
                        for (uint32_t row = 0; row < archetype.chunks[chunk].count; ++row)
                            PutU32(sink, headers[row].count);
                    }
                    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk)
                    {
                        const BufferHeader* headers = static_cast<const BufferHeader*>(archetype.ColumnData(chunk, c));
                        for (uint32_t row = 0; row < archetype.chunks[chunk].count; ++row)
                            PutValues(sink, type, static_cast<const uint8_t*>(headers[row].data), headers[row].count,
                                      scratch);
                    }
                    break;
                default:
                    break;   // a Tag: presence only
                }
                if (column.enableable)
                    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk)
                    {
                        const uint64_t* mask = archetype.EnabledMask(chunk, c);
                        for (uint32_t row = 0; row < archetype.chunks[chunk].count; ++row)
                            PutU8(sink, GetMaskBit(mask, row) ? 1 : 0);
                    }
            }
        }

        // Sparse components, in dense order.
        PutU32(sink, uint32_t(tables.sparse.size()));
        for (const SparseSet* set : tables.sparse)
        {
            PutU32(sink, tables.tableIndex[uint32_t(set->type->id)]);
            PutU32(sink, set->GetCount());
            sink.Put(set->entities.data(), size_t(set->GetCount()) * sizeof(Entity));
            PutValues(sink, *set->type, set->values, set->GetCount(), scratch);
        }

        // Resources.
        PutU32(sink, uint32_t(tables.resources.size()));
        for (const ComponentType* type : tables.resources)
        {
            PutU32(sink, tables.tableIndex[uint32_t(type->id)]);
            PutValues(sink, *type, static_cast<const uint8_t*>(type->value), 1, scratch);
        }
    }

    nvrhi::FRESULT WorldSnapshot::Save(const WorldImpl& world, IStateWriter* writer) noexcept
    {
        SaveTables tables;
        BuildSaveTables(world, tables);

        uint8_t* scratch = nullptr;
        if (tables.scratchBytes != 0)
        {
            scratch = static_cast<uint8_t*>(HeapAllocate(tables.scratchBytes, c_ColumnAlignment));
            if (!scratch)
                return nvrhi::FE_OUT_OF_MEMORY;
        }
        CountingSink counter;
        WritePayload(world, tables, counter, scratch);

        uint8_t* buffer = static_cast<uint8_t*>(HeapAllocate(c_WriteBufferBytes, c_ColumnAlignment));
        if (!buffer)
        {
            HeapFree(scratch);
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        WriterSink sink(writer, buffer, c_WriteBufferBytes);
        uint8_t header[c_SnapshotHeaderBytes];
        std::memcpy(header, c_SnapshotMagic, sizeof(c_SnapshotMagic));
        StoreU32(header + c_SnapshotVersionOffset, c_SnapshotVersion);
        StoreU32(header + c_SnapshotVersionOffset + 4, 0);
        StoreU64(header + c_SnapshotPayloadBytesOffset, counter.bytes);
        sink.Put(header, sizeof(header));
        WritePayload(world, tables, sink, scratch);
        sink.Flush();
        NVRHI_ASSERT(NVRHI_FAILED(sink.Result()) || sink.Bytes() == c_SnapshotHeaderBytes + counter.bytes);

        nvrhi::FRESULT result = sink.Result();
        if (NVRHI_SUCCEEDED(result))
        {
            uint8_t checksum[c_SnapshotChecksumBytes];
            StoreU64(checksum, sink.Digest());
            result = writer->Write(checksum, sizeof(checksum));
        }
        HeapFree(buffer);
        HeapFree(scratch);
        return result;
    }

    // ------------------------------------------------------------------------------------------- load: read

    nvrhi::FRESULT WorldSnapshot::ReadSnapshot(WorldImpl& world, IStateReader* reader, SoftArray<uint8_t>& payload,
                                               uint64_t& payloadBytes) noexcept
    {
        const char* operation = "LoadSnapshot";
        uint8_t header[c_SnapshotHeaderBytes];
        nvrhi::FRESULT result = reader->Read(sizeof(header), header);
        if (NVRHI_FAILED(result))
        {
            world.ReportValidation(operation, "the snapshot is truncated (header)");
            return result;
        }
        if (std::memcmp(header, c_SnapshotMagic, sizeof(c_SnapshotMagic)) != 0)
            return Corrupt(world, "not a world snapshot");
        const uint32_t version = LoadU32(header + c_SnapshotVersionOffset);
        if (version != c_SnapshotVersion)
        {
            world.ReportValidation(operation, "snapshot version %u is not supported (this build reads version %u)",
                                   version, c_SnapshotVersion);
            return nvrhi::FE_UNSUPPORTED;
        }
        if (LoadU32(header + c_SnapshotVersionOffset + 4) != 0)
            return Corrupt(world, "unknown header flags");
        payloadBytes = LoadU64(header + c_SnapshotPayloadBytesOffset);

        // In pieces, growing the buffer as data arrives: a corrupt size cannot make the load allocate more than the
        // stream holds, and the load never reads past the snapshot's end.
        uint64_t read = 0;
        while (read < payloadBytes)
        {
            const uint64_t piece = std::min(payloadBytes - read, c_ReadPieceBytes);
            if (read + piece > payload.Capacity())
            {
                const uint64_t grown = std::min(payloadBytes, std::max(read + piece, uint64_t(payload.Capacity()) * 2));
                if (grown > SIZE_MAX || !payload.Reserve(size_t(grown)))
                    return nvrhi::FE_OUT_OF_MEMORY;
            }
            result = reader->Read(piece, payload.Data() + read);
            if (NVRHI_FAILED(result))
            {
                world.ReportValidation(operation, "the snapshot is truncated (payload)");
                return result;
            }
            read += piece;
            payload.SetSize(size_t(read));   // what a later growth copies
        }
        uint8_t stored[c_SnapshotChecksumBytes];
        result = reader->Read(sizeof(stored), stored);
        if (NVRHI_FAILED(result))
        {
            world.ReportValidation(operation, "the snapshot is truncated (checksum)");
            return result;
        }
        Hasher64 checksum;
        checksum.Update(header, sizeof(header));
        checksum.Update(payload.Data(), size_t(payloadBytes));
        if (checksum.Digest() != LoadU64(stored))
            return Corrupt(world, "checksum mismatch");
        return nvrhi::FS_OK;
    }

    // ---------------------------------------------------------------------------------------- load: validate

    nvrhi::FRESULT WorldSnapshot::ParseTypes(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept
    {
        uint32_t typeCount = 0;
        if (!cursor.U32(typeCount) || typeCount > cursor.Remaining() / c_TypeEntryBytes)
            return Corrupt(world, "component table");
        if (!image.types.Reserve(typeCount))
            return nvrhi::FE_OUT_OF_MEMORY;
        nvrhi::GUID previous{};
        for (uint32_t i = 0; i < typeCount; ++i)
        {
            const uint8_t* entry = nullptr;
            const uint8_t* fields = nullptr;
            if (!cursor.Take(c_TypeEntryBytes, entry))
                return Corrupt(world, "component table");
            nvrhi::GUID uuid;
            std::memcpy(&uuid, entry, sizeof(uuid));
            const uint32_t flags = LoadU32(entry + 16);
            const uint32_t size = LoadU32(entry + 20);
            const uint32_t alignment = LoadU32(entry + 24);
            const uint32_t inlineCapacity = LoadU32(entry + 28);
            const uint32_t fieldCount = LoadU32(entry + 32);
            if (!cursor.TakeArray(fieldCount, c_FieldEntryBytes, fields))
                return Corrupt(world, "component table");
            if (i != 0 && !GuidLess(previous, uuid))
                return Corrupt(world, "the component table is not in GUID order");
            previous = uuid;
            if (!IsSerialisable(ComponentFlags(flags)))
                return Corrupt(world, "the component table lists a component that is not serialisable");

            const uint32_t local = world.FindType(uuid);
            if (local == HashIndex::c_NotFound)
            {
                world.ReportValidation("LoadSnapshot",
                                       "component %08x-%04x-%04x-... (in the snapshot) is not registered in this world",
                                       uuid.Data1, uuid.Data2, uuid.Data3);
                return nvrhi::FE_NOT_FOUND;
            }
            const ComponentType& type = *world.m_Types[local];
            const ComponentTypeDesc& desc = type.desc;
            const uint32_t rowTracked = uint32_t(ComponentFlags::RowTracked);   // not a layout flag (EcsTypes.h)
            bool same = (uint32_t(desc.flags) & ~rowTracked) == (flags & ~rowTracked) &&
                        desc.size == size && desc.alignment == alignment &&
                        desc.bufferInlineCapacity == inlineCapacity && desc.fieldCount == fieldCount;
            for (uint32_t f = 0; same && f < fieldCount; ++f)
            {
                const uint8_t* field = fields + size_t(f) * c_FieldEntryBytes;
                same = desc.fields[f].offset == LoadU32(field) && desc.fields[f].scalarCount == LoadU32(field + 4) &&
                       uint8_t(desc.fields[f].type) == field[8] && uint8_t(desc.fields[f].shape) == field[9] &&
                       field[10] == 0 && field[11] == 0;
            }
            if (!same)
            {
                world.ReportValidation("LoadSnapshot",
                                       "%s has another layout in the snapshot (flags, size, alignment, inline capacity "
                                       "or fields)", desc.name ? desc.name : "(unnamed)");
                return nvrhi::FE_INVALID_ARGS;
            }
            image.types.Push(&type);
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldSnapshot::ParseEntities(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept
    {
        uint32_t tableSize = 0;
        if (!cursor.U32(tableSize) || tableSize == 0 || !cursor.TakeArray(tableSize - 1, 8, image.table))
            return Corrupt(world, "entity table");
        image.tableSize = tableSize;
        if (!image.marks.Assign(tableSize, 0))
            return nvrhi::FE_OUT_OF_MEMORY;
        uint32_t freeSlots = 0;
        for (uint32_t index = 1; index < tableSize; ++index)
        {
            const uint32_t generation = image.Generation(index);
            const uint32_t state = LoadU32(image.table + size_t(index - 1) * 8 + 4);
            if (generation == 0 || generation == c_PlaceholderGeneration || state > uint32_t(SnapshotEntityState::Alive))
                return Corrupt(world, "entity table");
            if (state == uint32_t(SnapshotEntityState::Free))
                ++freeSlots;
            else if (state == uint32_t(SnapshotEntityState::Alive))
                ++image.aliveCount;
        }
        // Every free index exactly once.
        if (!cursor.U32(image.freeCount) || image.freeCount != freeSlots ||
            !cursor.TakeArray(image.freeCount, 4, image.freeList))
            return Corrupt(world, "free list");
        for (uint32_t i = 0; i < image.freeCount; ++i)
        {
            const uint32_t index = LoadU32(image.freeList + size_t(i) * 4);
            if (index == 0 || index >= tableSize || image.State(index) != SnapshotEntityState::Free ||
                image.marks[index] != 0)
                return Corrupt(world, "free list");
            image.marks[index] = 1;
        }
        for (uint32_t i = 0; i < image.freeCount; ++i)
            image.marks[LoadU32(image.freeList + size_t(i) * 4)] = 0;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldSnapshot::ParseArchetypes(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept
    {
        // Every alive entity has exactly one row; marks[index] becomes 1 when it is found.
        uint32_t recordCount = 0;
        if (!cursor.U32(recordCount) || recordCount > cursor.Remaining() / 20)   // the smallest record has 20 bytes
            return Corrupt(world, "archetypes");
        if (!image.records.Reserve(recordCount))
            return nvrhi::FE_OUT_OF_MEMORY;
        const uint32_t typeCount = uint32_t(image.types.Size());
        uint64_t rows = 0;
        for (uint32_t r = 0; r < recordCount; ++r)
        {
            LoadRecord record{};
            uint32_t componentCount = 0;
            const uint8_t* indices = nullptr;
            if (!cursor.U32(componentCount) || componentCount > typeCount ||
                !cursor.TakeArray(componentCount, 4, indices) || !cursor.U32(record.rowCount) ||
                !cursor.U32(record.rowsPerChunk) || record.rowCount == 0 || record.rowsPerChunk == 0 ||
                record.rowsPerChunk > c_MaxRowsPerChunk || !cursor.TakeArray(record.rowCount, 8, record.entities))
                return Corrupt(world, "archetype record");
            for (uint32_t i = 0; i < record.rowCount; ++i)
            {
                const Entity entity = Entity(LoadU64(record.entities + size_t(i) * 8));
                const uint32_t index = GetEntityIndex(entity);
                if (index == 0 || index >= image.tableSize || image.State(index) != SnapshotEntityState::Alive ||
                    image.Generation(index) != GetEntityGeneration(entity) || image.marks[index] != 0)
                    return Corrupt(world, "a row names an entity that is not alive, or one that has another row");
                image.marks[index] = 1;
            }
            rows += record.rowCount;

            const uint64_t chunkCount = ChunksFor(record.rowCount, record.rowsPerChunk);
            record.firstColumn = uint32_t(image.columns.Size());
            record.columnCount = componentCount;
            for (uint32_t k = 0; k < componentCount; ++k)
            {
                const uint32_t tableIndex = LoadU32(indices + size_t(k) * 4);
                if (tableIndex >= typeCount || (k != 0 && tableIndex <= LoadU32(indices + size_t(k - 1) * 4)))
                    return Corrupt(world, "archetype components");
                LoadColumn column{};
                column.type = image.types[tableIndex];
                const uint32_t size = column.type->desc.size;
                bool ok = true;
                switch (column.type->kind)
                {
                case StorageKind::Shared:
                    ok = cursor.Take(size, column.values);
                    break;
                case StorageKind::Data:
                    ok = cursor.TakeArray(record.rowCount, size, column.values);
                    break;
                case StorageKind::Chunk:
                    ok = cursor.TakeArray(chunkCount, size, column.values);
                    break;
                case StorageKind::Buffer:
                {
                    ok = cursor.TakeArray(record.rowCount, 4, column.counts);
                    uint64_t elements = 0;
                    for (uint32_t i = 0; ok && i < record.rowCount; ++i)
                    {
                        elements += LoadU32(column.counts + size_t(i) * 4);
                        ok = elements <= cursor.Remaining() / size;
                    }
                    ok = ok && cursor.TakeArray(elements, size, column.values);
                    break;
                }
                case StorageKind::Tag:
                    break;
                default:
                    return Corrupt(world, "an archetype lists a Sparse component or a resource");
                }
                if (ok && column.type->enableable)
                {
                    ok = cursor.TakeArray(record.rowCount, 1, column.enabled);
                    for (uint32_t i = 0; ok && i < record.rowCount; ++i)
                        ok = column.enabled[i] <= 1;
                }
                if (!ok)
                    return Corrupt(world, "archetype values");
                if (!image.columns.Push(column))
                    return nvrhi::FE_OUT_OF_MEMORY;
            }
            image.records.Push(record);
        }
        if (rows != image.aliveCount)
            return Corrupt(world, "an alive entity has no row");
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldSnapshot::ParseSparseAndResources(WorldImpl& world, Cursor& cursor, LoadImage& image) noexcept
    {
        const uint32_t typeCount = uint32_t(image.types.Size());
        uint32_t setCount = 0;
        if (!cursor.U32(setCount) || setCount > cursor.Remaining() / 16)   // index, count and one entity
            return Corrupt(world, "sparse components");
        if (!image.sparse.Reserve(setCount))
            return nvrhi::FE_OUT_OF_MEMORY;
        uint32_t previous = 0;
        for (uint32_t s = 0; s < setCount; ++s)
        {
            LoadSparse sparse{};
            uint32_t tableIndex = 0;
            if (!cursor.U32(tableIndex) || tableIndex >= typeCount || (s != 0 && tableIndex <= previous) ||
                !cursor.U32(sparse.count) || sparse.count == 0)
                return Corrupt(world, "sparse components");
            previous = tableIndex;
            sparse.type = image.types[tableIndex];
            if (sparse.type->kind != StorageKind::Sparse ||
                !cursor.TakeArray(sparse.count, 8, sparse.entities) ||
                !cursor.TakeArray(sparse.count, sparse.type->desc.size, sparse.values))
                return Corrupt(world, "sparse components");
            sparse.set = world.m_SparseSets[sparse.type->sparseSet];
            // Alive (marked 1 by the rows) and once per set (marked 2 while the set is checked, 1 again after it).
            for (uint32_t i = 0; i < sparse.count; ++i)
            {
                const Entity entity = Entity(LoadU64(sparse.entities + size_t(i) * 8));
                const uint32_t index = GetEntityIndex(entity);
                if (index == 0 || index >= image.tableSize || image.marks[index] != 1 ||
                    image.Generation(index) != GetEntityGeneration(entity))
                    return Corrupt(world, "a sparse component names an entity that is not alive, or names one twice");
                image.marks[index] = 2;
            }
            for (uint32_t i = 0; i < sparse.count; ++i)
                image.marks[GetEntityIndex(Entity(LoadU64(sparse.entities + size_t(i) * 8)))] = 1;
            image.sparse.Push(sparse);
        }

        uint32_t resourceCount = 0;
        if (!cursor.U32(resourceCount) || resourceCount > cursor.Remaining() / 4)
            return Corrupt(world, "resources");
        if (!image.resources.Reserve(resourceCount))
            return nvrhi::FE_OUT_OF_MEMORY;
        uint32_t previousResource = 0;
        for (uint32_t i = 0; i < resourceCount; ++i)
        {
            uint32_t tableIndex = 0;
            LoadResource resource{};
            if (!cursor.U32(tableIndex) || tableIndex >= typeCount || (i != 0 && tableIndex <= previousResource))
                return Corrupt(world, "resources");
            previousResource = tableIndex;
            resource.type = image.types[tableIndex];
            if (resource.type->kind != StorageKind::Resource || !cursor.Take(resource.type->desc.size, resource.value))
                return Corrupt(world, "resources");
            image.resources.Push(resource);
        }
        if (!cursor.AtEnd())
            return Corrupt(world, "bytes after the resources");
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldSnapshot::Parse(WorldImpl& world, const uint8_t* payload, uint64_t payloadBytes,
                                        LoadImage& image) noexcept
    {
        Cursor cursor(payload, payloadBytes);
        nvrhi::FRESULT result = ParseTypes(world, cursor, image);
        if (NVRHI_SUCCEEDED(result))
            result = ParseEntities(world, cursor, image);
        if (NVRHI_SUCCEEDED(result))
            result = ParseArchetypes(world, cursor, image);
        if (NVRHI_SUCCEEDED(result))
            result = ParseSparseAndResources(world, cursor, image);
        return result;
    }

    // ----------------------------------------------------------------------------------------- load: prepare

    nvrhi::FRESULT WorldSnapshot::Prepare(WorldImpl& world, LoadImage& image) noexcept
    {
        // A record's components in ascending local ids, with their Shared values.
        auto sortRecord = [&](const LoadRecord& record) noexcept
        {
            image.ids.Clear();
            image.shared.Clear();
            image.order.Clear();
            for (uint32_t k = 0; k < record.columnCount; ++k)
                if (!image.order.Push(k))
                    return false;
            const LoadColumn* columns = &image.columns[record.firstColumn];
            std::sort(image.order.begin(), image.order.end(), [columns](uint32_t a, uint32_t b)
            {
                return columns[a].type->id < columns[b].type->id;
            });
            for (const uint32_t k : image.order)
            {
                const LoadColumn& column = columns[k];
                if (!image.ids.Push(column.type->id) ||
                    !image.shared.Push(column.type->kind == StorageKind::Shared ? column.values : nullptr))
                    return false;
            }
            return true;
        };

        // Every row must fit this world's chunks: checked for all records before any archetype is made.
        for (const LoadRecord& record : image.records)
        {
            if (!sortRecord(record))
                return nvrhi::FE_OUT_OF_MEMORY;
            if (!world.RowFits(image.ids.Data(), record.columnCount))
            {
                world.ReportValidation("LoadSnapshot", "a row of the snapshot does not fit this world's chunks");
                return nvrhi::FE_INVALID_ARGS;
            }
        }

        // The target archetypes, made now if they are new (an empty archetype changes no content; running out of
        // memory may leave some made). Records with the same components and Shared values share one: rows that
        // differed only in non-serialisable components.
        for (LoadRecord& record : image.records)
        {
            if (!sortRecord(record))
                return nvrhi::FE_OUT_OF_MEMORY;
            const uint32_t target = world.FindOrCreateArchetype(image.ids.Data(), record.columnCount,
                                                                image.shared.Data());
            if (IsArchetypeError(target))
            {
                if (target == c_ArchetypeTooLarge)
                    world.ReportValidation("LoadSnapshot", "a row of the snapshot does not fit this world's chunks");
                return ArchetypeErrorResult(target);
            }
            record.target = target;
            const Archetype& archetype = *world.m_Archetypes[target];
            for (uint32_t k = 0; k < record.columnCount; ++k)
            {
                LoadColumn& column = image.columns[record.firstColumn + k];
                column.column = archetype.FindColumn(column.type->id);
                NVRHI_ASSERT(column.column != c_NoColumn);
            }
        }

        // Placement: the commit empties every archetype, then appends the records' rows in order.
        const uint32_t archetypeCount = uint32_t(world.m_Archetypes.size());
        if (!image.archetypeRows.Assign(archetypeCount, 0))
            return nvrhi::FE_OUT_OF_MEMORY;
        for (LoadRecord& record : image.records)
        {
            record.firstRow = image.archetypeRows[record.target];
            image.archetypeRows[record.target] += record.rowCount;
        }

        nvrhi::FRESULT result = PreparePages(world, image);
        if (NVRHI_FAILED(result))
            return result;

        // Containers grow now, so that the commit allocates nothing; so do the blocks of the row versions (the commit
        // stamps every row with the load version).
        for (uint32_t a = 0; a < archetypeCount; ++a)
            if (image.archetypeRows[a] != 0)
            {
                Archetype& archetype = *world.m_Archetypes[a];
                const uint64_t chunks = ChunksFor(image.archetypeRows[a], archetype.rowsPerChunk);
                archetype.chunks.reserve(size_t(chunks));
                archetype.shrunkVersions.reserve(size_t(chunks));
                if (!world.ReserveRowVersionBlocks(archetype, uint32_t(chunks)))
                    return nvrhi::FE_OUT_OF_MEMORY;
            }
        world.m_Entities.reserve(image.tableSize);

        // The heap blocks of the buffers that hold more than their inline capacity, per column in row order.
        for (LoadRecord& record : image.records)
        {
            for (uint32_t k = 0; k < record.columnCount; ++k)
            {
                LoadColumn& column = image.columns[record.firstColumn + k];
                if (column.type->kind != StorageKind::Buffer)
                    continue;
                column.firstSpill = uint32_t(image.spills.Size());
                const ComponentTypeDesc& desc = column.type->desc;
                for (uint32_t i = 0; i < record.rowCount; ++i)
                {
                    const uint32_t count = LoadU32(column.counts + size_t(i) * 4);
                    if (count <= desc.bufferInlineCapacity)
                        continue;
                    void* block = HeapAllocate(size_t(count) * desc.size, desc.alignment);
                    if (!block)
                        return nvrhi::FE_OUT_OF_MEMORY;
                    if (!image.spills.Push(block))
                    {
                        HeapFree(block);
                        return nvrhi::FE_OUT_OF_MEMORY;
                    }
                }
            }
        }

        return PrepareSparse(world, image);
    }

    nvrhi::FRESULT WorldSnapshot::PreparePages(WorldImpl& world, LoadImage& image) noexcept
    {
        // Per pool: the pages the new rows need, and those the commit gives back when it empties the archetypes. The
        // pool reserves the difference as free pages now (adding slabs), so that it holds every page the commit takes
        // once the old content is gone, and no chunk allocation of the commit reaches the heap.
        PagePool* pools[2] = { &world.m_SmallPages, &world.m_LargePages };
        uint64_t need[2] = { 0, 0 };
        uint64_t released[2] = { 0, 0 };
        for (uint32_t a = 0; a < world.m_Archetypes.size(); ++a)
        {
            const Archetype& archetype = *world.m_Archetypes[a];
            const uint32_t p = &world.PoolFor(archetype.chunkSize) == pools[0] ? 0 : 1;
            released[p] += archetype.GetChunkCount();
            if (image.archetypeRows[a] != 0)
                need[p] += ChunksFor(image.archetypeRows[a], archetype.rowsPerChunk);
        }
        for (uint32_t p = 0; p < 2; ++p)
        {
            if (need[p] <= released[p])
                continue;
            const uint64_t free = need[p] - released[p];
            if (free > 0xffffffffull || !pools[p]->Reserve(uint32_t(free)))
                return nvrhi::FE_OUT_OF_MEMORY;
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldSnapshot::PrepareSparse(WorldImpl& world, LoadImage& image) noexcept
    {
        if (image.sparse.Size() == 0)
            return nvrhi::FS_OK;
        // Where each entity's row will be, for the presence masks.
        if (!image.targetOf.Assign(image.tableSize, c_NoArchetype) || !image.positionOf.Assign(image.tableSize, 0))
            return nvrhi::FE_OUT_OF_MEMORY;
        for (const LoadRecord& record : image.records)
            for (uint32_t i = 0; i < record.rowCount; ++i)
            {
                const uint32_t index = GetEntityIndex(Entity(LoadU64(record.entities + size_t(i) * 8)));
                image.targetOf[index] = record.target;
                image.positionOf[index] = record.firstRow + i;
            }

        // Value arrays that are too small are replaced by the commit; index pages and presence masks are added now
        // (zeroed, they mean what their absence means).
        for (LoadSparse& sparse : image.sparse)
        {
            SparseSet& set = *sparse.set;
            const ComponentTypeDesc& desc = sparse.type->desc;
            if (desc.size != 0 && set.capacity < sparse.count)
            {
                sparse.newCapacity = std::max(sparse.count, 16u);
                sparse.newValues = static_cast<uint8_t*>(HeapAllocate(size_t(sparse.newCapacity) * desc.size,
                                                                      desc.alignment));
                if (!sparse.newValues)
                    return nvrhi::FE_OUT_OF_MEMORY;
            }
            set.entities.reserve(sparse.count);
            for (uint32_t i = 0; i < sparse.count; ++i)
            {
                const uint32_t index = GetEntityIndex(Entity(LoadU64(sparse.entities + size_t(i) * 8)));
                const uint32_t page = index >> c_SparsePageBits;
                if (page >= set.pages.size())
                    set.pages.resize(page + 1, nullptr);
                if (!set.pages[page])
                {
                    uint32_t* words = static_cast<uint32_t*>(
                        HeapAllocate(c_SparsePageSize * sizeof(uint32_t), c_ColumnAlignment));
                    if (!words)
                        return nvrhi::FE_OUT_OF_MEMORY;
                    std::memset(words, 0, c_SparsePageSize * sizeof(uint32_t));
                    set.pages[page] = words;
                }

                // A missing mask is made by SetPresence (which tells queries, GetPresenceMaskVersion): setting and
                // clearing a bit of a mask that did not exist touches no row's presence.
                const uint32_t archetype = image.targetOf[index];
                const uint32_t rowsPerChunk = world.m_Archetypes[archetype]->rowsPerChunk;
                const uint32_t chunk = image.positionOf[index] / rowsPerChunk;
                const uint32_t row = image.positionOf[index] % rowsPerChunk;
                auto hasMask = [&]() noexcept
                {
                    return archetype < set.presence.size() && chunk < set.presence[archetype].size() &&
                           set.presence[archetype][chunk] != nullptr;
                };
                if (!hasMask())
                {
                    world.SetPresence(set, archetype, chunk, row, true);
                    world.SetPresence(set, archetype, chunk, row, false);
                    if (!hasMask())
                        return nvrhi::FE_OUT_OF_MEMORY;
                }
            }
        }
        return nvrhi::FS_OK;
    }

    void WorldSnapshot::ReleasePrepared(LoadImage& image) noexcept
    {
        for (void* block : image.spills)
            HeapFree(block);
        image.spills.Clear();
        for (LoadSparse& sparse : image.sparse)
        {
            HeapFree(sparse.newValues);
            sparse.newValues = nullptr;
        }
    }

    // ------------------------------------------------------------------------------------------ load: commit

    void WorldSnapshot::ClearContent(WorldImpl& world) noexcept
    {
        // Reported first, as DestroyEntities reports, while the values can still be read.
        for (const Archetype* archetype : world.m_Archetypes)
            for (uint32_t chunk = 0; chunk < archetype->GetChunkCount(); ++chunk)
                if (archetype->chunks[chunk].count != 0)
                    world.ObserveRowsRemoving(archetype->index, chunk, 0, archetype->chunks[chunk].count,
                                              c_NoArchetype);
        for (const SparseSet* set : world.m_SparseSets)
            for (const Entity entity : set->entities)
                world.ObserveSparseRemoving(set->type->id, entity);

        for (SparseSet* set : world.m_SparseSets)
        {
            for (const Entity entity : set->entities)
            {
                const uint32_t index = GetEntityIndex(entity);
                set->pages[index >> c_SparsePageBits][index & (c_SparsePageSize - 1)] = 0;
                const EntityRecord& record = world.m_Entities[index];
                world.SetPresence(*set, record.archetype, record.chunk, record.row, false);
            }
            DestructValues(*set->type, set->values, set->GetCount());
            set->entities.clear();
        }
        for (Archetype* archetype : world.m_Archetypes)
        {
            for (uint32_t chunk = 0; chunk < archetype->GetChunkCount(); ++chunk)
                for (uint32_t row = 0; row < archetype->chunks[chunk].count; ++row)
                    world.DestructRow(*archetype, chunk, row);
            while (!archetype->chunks.empty())
            {
                archetype->chunks.back().count = 0;
                world.ReleaseLastChunk(*archetype);
            }
            archetype->entityCount = 0;
        }
        world.m_AliveCount = 0;
    }

    void WorldSnapshot::FillRows(WorldImpl& world, LoadImage& image, LoadRecord& record, uint32_t chunk, uint32_t row,
                                 uint32_t run, uint32_t done) noexcept
    {
        Archetype& archetype = *world.m_Archetypes[record.target];
        Entity* entities = archetype.Entities(chunk);
        for (uint32_t t = 0; t < run; ++t)
        {
            const Entity entity = Entity(LoadU64(record.entities + size_t(done + t) * 8));
            entities[row + t] = entity;
            EntityRecord& entry = world.m_Entities[GetEntityIndex(entity)];
            entry.archetype = record.target;
            entry.chunk = chunk;
            entry.row = uint16_t(row + t);
            entry.sparseCount = 0;
        }

        for (uint32_t k = 0; k < record.columnCount; ++k)
        {
            LoadColumn& column = image.columns[record.firstColumn + k];
            const ArchetypeColumn& target = archetype.columns[column.column];
            const ComponentType& type = *column.type;
            const uint32_t size = type.desc.size;
            uint8_t* data = static_cast<uint8_t*>(archetype.ColumnData(chunk, column.column));
            switch (type.kind)
            {
            case StorageKind::Data:
                // Relocatable: the rows' raw storage takes the bytes as they are.
                std::memcpy(data + size_t(row) * size, column.values + size_t(done) * size, size_t(run) * size);
                break;
            case StorageKind::Chunk:
                // A chunk takes the value of the saved chunk that held its first row; later rows (of a record merged
                // into the same archetype) leave it.
                if (row == 0)
                    std::memcpy(data, column.values + size_t(done / record.rowsPerChunk) * size, size);
                break;
            case StorageKind::Buffer:
            {
                BufferHeader* headers = reinterpret_cast<BufferHeader*>(data);
                const uint32_t inlineCapacity = type.desc.bufferInlineCapacity;
                for (uint32_t t = 0; t < run; ++t)
                {
                    BufferHeader& header = headers[row + t];
                    const uint32_t count = LoadU32(column.counts + size_t(done + t) * 4);
                    const uint8_t* elements = column.values + size_t(column.elementCursor) * size;
                    column.elementCursor += count;
                    if (count <= inlineCapacity)
                    {
                        header.data = target.inlineStride != 0 ? archetype.InlineData(chunk, column.column, row + t)
                                                               : nullptr;
                        header.capacity = inlineCapacity;
                    }
                    else
                    {
                        header.data = image.spills[column.firstSpill + column.spillCursor++];
                        header.capacity = count;
                        world.m_BufferHeapBytes.fetch_add(uint64_t(count) * size, std::memory_order_relaxed);
                    }
                    header.count = count;
                    if (count != 0)
                        std::memcpy(header.data, elements, size_t(count) * size);
                }
                break;
            }
            default:
                break;   // Tag: nothing; Shared: the archetype holds the value
            }
            if (target.enableable)
            {
                uint64_t* mask = archetype.EnabledMask(chunk, column.column);
                for (uint32_t t = 0; t < run; ++t)
                    SetMaskBit(mask, row + t, column.enabled[done + t] != 0);
            }
        }
    }

    nvrhi::FRESULT WorldSnapshot::Commit(WorldImpl& world, LoadImage& image) noexcept
    {
        const uint64_t version = world.m_Version;
        ClearContent(world);

        // The prepared value arrays of the sparse sets.
        for (LoadSparse& sparse : image.sparse)
        {
            if (!sparse.newValues)
                continue;
            HeapFree(sparse.set->values);
            sparse.set->values = sparse.newValues;
            sparse.set->capacity = sparse.newCapacity;
            sparse.newValues = nullptr;
        }

        // The entity table, with the snapshot's generations, states and free-list order.
        world.m_Entities.resize(image.tableSize);
        world.m_Entities[0] = EntityRecord{ 0, c_FreeSlot, 0, 0, 0 };
        for (uint32_t index = 1; index < image.tableSize; ++index)
        {
            const SnapshotEntityState state = image.State(index);
            world.m_Entities[index] = EntityRecord{ image.Generation(index),
                                                    state == SnapshotEntityState::Reserved ? c_ReservedSlot : c_FreeSlot,
                                                    0, 0, 0 };
        }
        world.m_FreeHead = 0;
        world.m_FreeTail = 0;
        for (uint32_t i = 0; i < image.freeCount; ++i)
        {
            const uint32_t index = LoadU32(image.freeList + size_t(i) * 4);
            if (world.m_FreeTail != 0)
                world.m_Entities[world.m_FreeTail].chunk = index;
            else
                world.m_FreeHead = index;
            world.m_FreeTail = index;
        }
        world.m_AliveCount = image.aliveCount;

        // The rows, record by record. The pools hold every page they need (PreparePages), so no append reaches the
        // heap or fails.
        for (LoadRecord& record : image.records)
        {
            Archetype& archetype = *world.m_Archetypes[record.target];
            const nvrhi::FRESULT appended = world.AppendRows(archetype, record.rowCount, version,
                [&](uint32_t chunk, uint32_t row, uint32_t run, uint32_t done) noexcept
            {
                FillRows(world, image, record, chunk, row, run, done);
            });
            NVRHI_ASSERT(NVRHI_SUCCEEDED(appended));
            (void)appended;
        }

        // Sparse components, in the snapshot's dense order.
        for (const LoadSparse& sparse : image.sparse)
        {
            SparseSet& set = *sparse.set;
            const uint32_t size = sparse.type->desc.size;
            for (uint32_t i = 0; i < sparse.count; ++i)
            {
                const Entity entity = Entity(LoadU64(sparse.entities + size_t(i) * 8));
                const uint32_t index = GetEntityIndex(entity);
                set.entities.push_back(entity);
                set.pages[index >> c_SparsePageBits][index & (c_SparsePageSize - 1)] = i + 1;
                if (size != 0)
                    std::memcpy(set.Value(i), sparse.values + size_t(i) * size, size);
                EntityRecord& record = world.m_Entities[index];
                NVRHI_ASSERT(record.archetype < c_ClaimedSlot);   // the rows were all placed
                ++record.sparseCount;
                world.SetPresence(set, record.archetype, record.chunk, record.row, true);   // its mask was prepared
            }
        }

        // Resources in the snapshot; the others keep their values.
        for (const LoadResource& resource : image.resources)
            std::memcpy(world.m_Types[uint32_t(resource.type->id)]->value, resource.value, resource.type->desc.size);

        // Reported once the rows are complete, Sparse components included.
        for (const LoadRecord& record : image.records)
        {
            const uint32_t rowsPerChunk = world.m_Archetypes[record.target]->rowsPerChunk;
            uint32_t position = record.firstRow;
            const uint32_t end = record.firstRow + record.rowCount;
            while (position < end)
            {
                const uint32_t row = position % rowsPerChunk;
                const uint32_t run = std::min(end - position, rowsPerChunk - row);
                world.ObserveRowsAdded(record.target, position / rowsPerChunk, row, run, c_NoArchetype);
                position += run;
            }
        }
        for (const LoadSparse& sparse : image.sparse)
            for (uint32_t i = 0; i < sparse.count; ++i)
                world.ObserveSparseAdded(sparse.type->id, Entity(LoadU64(sparse.entities + size_t(i) * 8)));

        ++world.m_StructuralVersion;
        // The content is replaced: the load succeeded. What the observers and their cascade fail at afterwards is
        // reported through validation, not returned (EcsWorld.h).
        const nvrhi::FRESULT observers = world.FlushObservers();
        if (NVRHI_FAILED(observers))
            world.ReportValidation("LoadSnapshot", "the load succeeded; its observers failed with 0x%08x",
                                   uint32_t(observers));
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldSnapshot::Load(WorldImpl& world, IStateReader* reader) noexcept
    {
        SoftArray<uint8_t> payload;
        uint64_t payloadBytes = 0;
        nvrhi::FRESULT result = ReadSnapshot(world, reader, payload, payloadBytes);
        if (NVRHI_FAILED(result))
            return result;
        LoadImage image;
        result = Parse(world, payload.Data(), payloadBytes, image);
        if (NVRHI_SUCCEEDED(result))
            result = Prepare(world, image);
        if (NVRHI_FAILED(result))
        {
            ReleasePrepared(image);
            return result;
        }
        return Commit(world, image);
    }

    // ------------------------------------------------------------------------------------------------ IWorld

    nvrhi::FRESULT WorldImpl::SaveSnapshot(IStateWriter* writer) noexcept
    {
        // A read of every column: it races with parallel systems, so it runs where structural changes may.
        if (!CanChangeStructure())
        {
            ReportValidation("SaveSnapshot", "snapshots are taken outside phases (or by an exclusive system) and "
                             "outside observers");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!writer)
            return nvrhi::FE_INVALID_ARGS;
        return WorldSnapshot::Save(*this, writer);
    }

    nvrhi::FRESULT WorldImpl::LoadSnapshot(IStateReader* reader) noexcept
    {
        if (!CanChangeStructure())
        {
            ReportValidation("LoadSnapshot", "structural changes are deferred inside phases and observers (3.3.3)");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!reader)
            return nvrhi::FE_INVALID_ARGS;
        return WorldSnapshot::Load(*this, reader);
    }

    uint64_t WorldImpl::ComputeHash() noexcept
    {
        if (m_InPhase && m_ExclusiveDepth == 0)
            ReportValidation("ComputeHash", "the hash reads every column while parallel systems may write them");
        return WorldSnapshot::Hash(*this);
    }
}
