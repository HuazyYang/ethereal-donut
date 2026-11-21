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

// Cached queries (aggregate docs/ecs-design.md 6.1; plan step E1.2): IQuery over the world's archetypes. Module-private
// (com-interface-abi.md section 6).
//
// Matching is per archetype and incremental: archetypes are only appended (EcsWorldImpl.h), so a query tests the ones
// created since it last looked and keeps the matches in creation order. The chunk list (every non-empty chunk of the
// matched archetypes, in archetype-creation and chunk order) and each chunk's term arrays (column pointers, enabled
// masks, presence masks) are rebuilt when the world's structural version or presence-mask version has changed since
// the last rebuild, on the first call that needs them. Structural changes happen only outside phases, in exclusive
// systems and at sync points (rule 3.3.3), so the list never changes while a phase's parallel systems run; GetChunks
// then only reads it, tests versions and stamps them, and may run on several threads at once. It skips a chunk that
// fails Changed or Added, and, for queries with per-row filters (Enabled, Disabled, Sparse With and Without), a chunk
// in which no row passes them, so that such a query neither visits nor stamps it.

#pragma once

#include "EcsWorldImpl.h"

#include <donut/engine/EcsQuery.h>

#include <atomic>

namespace donut::engine::ecs
{
    // An implementation limit: CreateQuery refuses more terms (FE_INVALID_ARGS).
    constexpr uint32_t c_MaxQueryTerms = 1024;

    // Whether a query keeps its world alive. The queries IWorld::CreateQuery hands out do. A query that the world owns
    // itself must not, or the two would keep each other alive: E1.4's observers, which the world holds, match their
    // filter terms with a NotHeld query. Such a query must not outlive its world, and its owner releases it first.
    enum class QueryWorldReference : uint8_t
    {
        Held,
        NotHeld,
    };

    // Validates the terms and creates the query (FE_INVALID_ARGS, FE_OUT_OF_MEMORY). IWorld::CreateQuery passes Held.
    nvrhi::FRESULT CreateQueryObject(WorldImpl& world, const QueryDesc& desc, QueryWorldReference reference,
                                     IQuery** ppQuery) noexcept;

    NVRHI_CLASS_CLSID(QueryImpl, "00b75e20-8343-4fbc-b45f-c2b1e90c4aa6")
    class QueryImpl final : public nvrhi::ObjectImpl<IQuery>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(QueryImpl)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(QueryImpl)
        NVRHI_IMPLEMENTS_INTERFACE(IQuery)
        NVRHI_IMPLEMENTS_CLASS(QueryImpl)
        NVRHI_END_INTERFACE_TABLE()

        // Holds a reference to the world for QueryWorldReference::Held.
        QueryImpl(WorldImpl* world, QueryWorldReference reference) noexcept;
        // Validates and copies the terms; FE_INVALID_ARGS with or without validation.
        nvrhi::FRESULT Initialize(const QueryDesc& desc) noexcept;

        // IQuery
        uint32_t GetChunkCount() noexcept override;
        uint32_t GetChunks(uint32_t first, const ChunkRequest& request, uint32_t capacity,
                           ChunkView* views) noexcept override;
        uint32_t CountEntities() noexcept override;
        uint32_t GetTermCount() noexcept override { return uint32_t(m_Terms.size()); }
        uint32_t GetDataTermCount() noexcept override { return m_DataTermCount; }

        // ------------------------------------------------------------------------------- module-internal access

        struct Counters
        {
            uint64_t rebuilds = 0;            // chunk-list rebuilds
            uint64_t archetypesTested = 0;    // archetypes matched against the terms (each once)
        };
        const Counters& GetCounters() const noexcept { return m_Counters; }
        // The matched archetypes in creation order, as of the last rebuild.
        uint32_t GetMatchedArchetypeCount() const noexcept { return uint32_t(m_Matched.size()); }
        uint32_t GetMatchedArchetype(uint32_t index) const noexcept { return m_Matched[index].archetype; }

    private:
        struct Term
        {
            ComponentId component;
            TermOp op;
            Access access;
            StorageKind kind;
            bool enableable;
            bool data;                  // IsDataTerm: a column pointer in ChunkView::columns
            uint8_t reserved[3];
            uint32_t dataIndex;         // data terms: the index in ChunkView::columns
            const SparseSet* sparse;    // Sparse terms: the set whose presence masks the views carry
        };

        struct MatchedArchetype
        {
            uint32_t archetype;         // the world's index
            uint32_t firstColumn;       // m_TermColumns: one archetype column per term, c_NoColumn where absent
            uint32_t firstCheck;        // m_Slots: version slots that Changed and Added terms test
            uint32_t checkCount;
            uint32_t firstWrite;        // m_Slots: version slots that Write terms stamp
            uint32_t writeCount;
        };

        struct ChunkEntry
        {
            const Entity* entities;
            uint64_t* versions;
            uint32_t count;
            uint32_t matched;           // index in m_Matched
        };

        // A per-row filter, as the typed layer applies it (EcsQuery.h): the rows a term lets through are the set bits
        // of its mask, or the clear ones.
        struct RowFilter
        {
            uint32_t term;
            bool presence;              // the term's presence mask (Sparse); else its enabled mask
            bool clear;                 // Disabled, or Without on a Sparse term
            uint8_t reserved[2];
        };

        bool Matches(const Archetype& archetype) const noexcept;
        void AddMatched(const Archetype& archetype) noexcept;
        void AddSlot(uint32_t first, uint32_t slot) noexcept;   // appends to m_Slots unless [first, end) has it
        // Rebuilds the chunk list if the world's structure changed since the last rebuild; thread-safe.
        void EnsureCurrent() noexcept;
        void Rebuild() noexcept;
        // Whether some row of chunk `index` passes every row filter: an AND of mask words, rows / 64 of them.
        bool AnyRowPasses(size_t index, uint32_t rows) const noexcept;
        void ValidateRequest(const ChunkRequest& request, uint32_t capacity, const ChunkView* views) noexcept;

        nvrhi::AutoPtr<IWorld> m_WorldReference;   // null for QueryWorldReference::NotHeld
        WorldImpl* m_World;
        nvrhi::string m_DebugName;

        Vector<Term> m_Terms;
        Vector<RowFilter> m_RowFilters;  // empty for most queries: GetChunks then tests no row
        uint32_t m_DataTermCount = 0;
        bool m_HasEnableable = false;   // some term's component is Enableable: per-chunk enabled masks
        bool m_HasSparse = false;       // some term is Sparse: per-chunk presence masks
        bool m_HasWrite = false;
        uint8_t m_Reserved = 0;

        // Matching: archetypes [0, m_TestedArchetypes) have been tested.
        uint32_t m_TestedArchetypes = 0;
        Vector<MatchedArchetype> m_Matched;
        Vector<uint32_t> m_TermColumns;
        Vector<uint32_t> m_Slots;

        // The chunk list, rebuilt by Rebuild. Chunk i's arrays start at i * GetDataTermCount() in m_Columns and at
        // i * GetTermCount() in m_EnabledMasks and m_PresenceMasks; queries without Enableable or Sparse terms point
        // every view at m_NullMasks instead.
        Vector<ChunkEntry> m_Chunks;
        Vector<void*> m_Columns;
        Vector<const uint64_t*> m_EnabledMasks;
        Vector<const uint64_t*> m_PresenceMasks;
        Vector<const uint64_t*> m_NullMasks;
        uint32_t m_EntityCount = 0;

        // What the chunk list was built from. Published with release stores after a rebuild, read with acquire loads.
        std::atomic<uint64_t> m_BuiltStructuralVersion{ 0 };
        std::atomic<uint64_t> m_BuiltPresenceVersion{ ~0ull };
        std::atomic<bool> m_RebuildLock{ false };
        Counters m_Counters;
    };
}
