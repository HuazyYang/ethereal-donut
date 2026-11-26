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

// Cached queries (aggregate docs/ecs-design.md 4.5, 6.1, 11.3; plan step E1.2). See EcsQueryImpl.h.

#include "EcsQueryImpl.h"
#include "EcsSchedulerImpl.h"

#include <atomic>
#include <thread>

namespace donut::engine::ecs
{
    namespace
    {
        // The ChunkView::columns of a query without data terms: never indexed, but not null.
        void* const g_NoColumns[1] = { nullptr };

        // Version slots and enabled-mask words may be written by other threads (StampVersion, SetMaskBit); read them
        // as a whole.
        uint64_t LoadWord(const uint64_t* slot) noexcept
        {
#if defined(_MSC_VER)
            return uint64_t(*reinterpret_cast<const volatile long long*>(slot));
#else
            return __atomic_load_n(slot, __ATOMIC_RELAXED);
#endif
        }

        // Terms that the entity must have: the archetype must have the component (Sparse ones: the row).
        bool IsRequired(TermOp op) noexcept
        {
            return op == TermOp::All || op == TermOp::Changed || op == TermOp::Added || op == TermOp::Enabled ||
                   op == TermOp::Disabled;
        }

        const char* NameOf(const ComponentType& type) noexcept
        {
            return type.desc.name ? type.desc.name : "(unnamed)";
        }
    }

    // ------------------------------------------------------------------------------------------- creation

    QueryImpl::QueryImpl(WorldImpl* world, QueryWorldReference reference) noexcept
        : m_WorldReference(reference == QueryWorldReference::Held ? static_cast<IWorld*>(world) : nullptr)
        , m_World(world)
    {
    }

    nvrhi::FRESULT QueryImpl::Initialize(const QueryDesc& desc) noexcept
    {
        const char* operation = "CreateQuery";
        WorldImpl& world = *m_World;
        m_DebugName = desc.debugName ? desc.debugName : "";
        const char* name = m_DebugName.c_str();

        if (!desc.terms || desc.termCount == 0)
        {
            world.ReportValidation(operation, "query '%s': a query needs at least one term", name);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (desc.termCount > c_MaxQueryTerms)
        {
            world.ReportValidation(operation, "query '%s': %u terms; a query takes at most %u", name, desc.termCount,
                                   c_MaxQueryTerms);
            return nvrhi::FE_INVALID_ARGS;
        }

        // Each term on its own.
        for (uint32_t i = 0; i < desc.termCount; ++i)
        {
            const QueryTerm& term = desc.terms[i];
            auto fail = [&](const char* reason, const char* component) noexcept
            {
                world.ReportValidation(operation, "query '%s', term %u (%s): %s", name, i, component, reason);
                return nvrhi::FE_INVALID_ARGS;
            };
            if (uint8_t(term.op) > uint8_t(TermOp::Disabled))
                return fail("unknown operator", "?");
            if (uint8_t(term.access) > uint8_t(Access::Write))
                return fail("unknown access", "?");
            const ComponentType* type = world.GetType(term.component);
            if (!type)
                return fail("the component is not registered", "?");
            const char* component = NameOf(*type);
            if (type->kind == StorageKind::Resource)
                return fail("a resource is not on entities; read it with IWorld::GetResource", component);
            if (term.op == TermOp::Without && term.access == Access::Write)
                return fail("a Without term has nothing to write", component);
            if (term.access == Access::Write && type->kind == StorageKind::Shared)
                return fail("Shared values are read-only in queries (4.4); change them with AddComponent", component);
            if (type->kind == StorageKind::Sparse)
            {
                if (IsDataTerm(term))
                    return fail("Sparse components are filters only (With, Without); read them with "
                                "IWorld::GetComponent", component);
                if (term.op == TermOp::Changed || term.op == TermOp::Added)
                    return fail("Sparse components have no change versions (D9)", component);
                if (term.op == TermOp::Any)
                    return fail("Any terms name components stored in archetypes, not Sparse ones", component);
            }
            if ((term.op == TermOp::Enabled || term.op == TermOp::Disabled) && !type->enableable)
                return fail("Enabled and Disabled need an Enableable component", component);
        }

        // Terms that contradict each other: the query could never visit a row.
        for (uint32_t i = 0; i < desc.termCount; ++i)
        {
            for (uint32_t j = i + 1; j < desc.termCount; ++j)
            {
                const QueryTerm& a = desc.terms[i];
                const QueryTerm& b = desc.terms[j];
                if (a.component != b.component)
                    continue;
                const bool excluded = (a.op == TermOp::Without && IsRequired(b.op)) ||
                                      (b.op == TermOp::Without && IsRequired(a.op));
                const bool bothStates = (a.op == TermOp::Enabled && b.op == TermOp::Disabled) ||
                                        (a.op == TermOp::Disabled && b.op == TermOp::Enabled);
                if (excluded || bothStates)
                {
                    world.ReportValidation(operation, "query '%s', terms %u and %u (%s): %s", name, i, j,
                                           NameOf(*world.GetType(a.component)),
                                           excluded ? "the component is both required and excluded"
                                                    : "the component is both Enabled and Disabled");
                    return nvrhi::FE_INVALID_ARGS;
                }
            }
        }

        m_Terms.reserve(desc.termCount);
        for (uint32_t i = 0; i < desc.termCount; ++i)
        {
            const QueryTerm& source = desc.terms[i];
            const ComponentType& type = *world.GetType(source.component);
            Term term{};
            term.component = source.component;
            term.op = source.op;
            term.access = source.access;
            term.kind = type.kind;
            term.enableable = type.enableable;
            term.data = IsDataTerm(source);
            term.dataIndex = term.data ? m_DataTermCount++ : 0;
            term.sparse = type.kind == StorageKind::Sparse ? world.GetSparseSet(source.component) : nullptr;
            m_HasEnableable = m_HasEnableable || type.enableable;
            m_HasSparse = m_HasSparse || term.sparse != nullptr;
            m_HasWrite = m_HasWrite || source.access == Access::Write;
            m_Terms.push_back(term);

            // The row filters of the typed layer (EcsQuery.h, RowPasses), in term order.
            RowFilter filter{ i, false, false, {} };
            if (term.op == TermOp::Enabled || term.op == TermOp::Disabled)
            {
                filter.clear = term.op == TermOp::Disabled;
                m_RowFilters.push_back(filter);
            }
            else if (term.sparse && (term.op == TermOp::All || term.op == TermOp::Without))
            {
                filter.presence = true;
                filter.clear = term.op == TermOp::Without;
                m_RowFilters.push_back(filter);
            }
        }
        m_NullMasks.resize(desc.termCount, nullptr);
        m_NullRowVersions.resize(desc.termCount, nullptr);
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT CreateQueryObject(WorldImpl& world, const QueryDesc& desc, QueryWorldReference reference,
                                     IQuery** ppQuery) noexcept
    {
        if (!ppQuery)
        {
            world.ReportValidation("CreateQuery", "ppQuery is null");
            return nvrhi::FE_INVALID_ARGS;
        }
        *ppQuery = nullptr;
        QueryImpl* query = nullptr;
        try
        {
            query = MAKE_RC_OBJ(QueryImpl, &world, reference);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        if (!query)
            return nvrhi::FE_OUT_OF_MEMORY;
        const nvrhi::FRESULT result = query->Initialize(desc);
        if (NVRHI_FAILED(result))
        {
            query->Release();
            return result;
        }
        *ppQuery = query;
        return nvrhi::FS_OK;
    }

    // ------------------------------------------------------------------------------------------- matching

    bool QueryImpl::Matches(const Archetype& archetype) const noexcept
    {
        bool hasAny = false;
        bool anyPresent = false;
        for (const Term& term : m_Terms)
        {
            if (term.kind == StorageKind::Sparse)
                continue;   // outside the archetype: a row filter only
            const bool present = archetype.FindColumn(term.component) != c_NoColumn;
            switch (term.op)
            {
            case TermOp::Optional:
                break;
            case TermOp::Without:
                if (present)
                    return false;
                break;
            case TermOp::Any:
                hasAny = true;
                anyPresent = anyPresent || present;
                break;
            default:
                if (!present)
                    return false;
                break;
            }
        }
        return !hasAny || anyPresent;
    }

    void QueryImpl::AddSlot(uint32_t first, uint32_t slot) noexcept
    {
        for (size_t i = first; i < m_Slots.size(); ++i)
            if (m_Slots[i] == slot)
                return;
        m_Slots.push_back(slot);
    }

    void QueryImpl::AddMatched(const Archetype& archetype) noexcept
    {
        MatchedArchetype matched{};
        matched.archetype = archetype.index;
        matched.firstColumn = uint32_t(m_TermColumns.size());
        for (const Term& term : m_Terms)
            m_TermColumns.push_back(term.kind == StorageKind::Sparse ? c_NoColumn
                                                                      : archetype.FindColumn(term.component));
        const uint32_t* columns = m_TermColumns.data() + matched.firstColumn;

        // Changed tests the column's version, Added the chunk's added version (4.5). Both are required terms, so the
        // column is present.
        matched.firstCheck = uint32_t(m_Slots.size());
        for (size_t t = 0; t < m_Terms.size(); ++t)
        {
            if (m_Terms[t].op == TermOp::Changed)
                AddSlot(matched.firstCheck, columns[t]);
            else if (m_Terms[t].op == TermOp::Added)
                AddSlot(matched.firstCheck, archetype.AddedVersionSlot());
        }
        matched.checkCount = uint32_t(m_Slots.size()) - matched.firstCheck;

        // Write terms stamp their columns where the archetype has them (an Optional or Any term may be absent).
        matched.firstWrite = uint32_t(m_Slots.size());
        for (size_t t = 0; t < m_Terms.size(); ++t)
            if (m_Terms[t].access == Access::Write && columns[t] != c_NoColumn)
                AddSlot(matched.firstWrite, columns[t]);
        matched.writeCount = uint32_t(m_Slots.size()) - matched.firstWrite;

        // RowTracked columns of data terms (E7.5): views carry their row versions.
        matched.firstTracked = uint32_t(m_Slots.size());
        for (size_t t = 0; t < m_Terms.size(); ++t)
            if (m_Terms[t].data && columns[t] != c_NoColumn &&
                archetype.columns[columns[t]].rowVersionSlot != c_NoRowVersions)
                AddSlot(matched.firstTracked, columns[t]);
        matched.trackedCount = uint32_t(m_Slots.size()) - matched.firstTracked;

        m_Matched.push_back(matched);
    }

    // ----------------------------------------------------------------------------------------- the chunk list

    void QueryImpl::EnsureCurrent() noexcept
    {
        const uint64_t structural = m_World->GetStructuralVersion();
        const uint64_t presence = m_HasSparse ? m_World->GetPresenceMaskVersion() : 0;
        if (m_BuiltStructuralVersion.load(std::memory_order_acquire) == structural &&
            m_BuiltPresenceVersion.load(std::memory_order_acquire) == presence)
            return;

        // Only the first caller after a structural change rebuilds; others that arrive meanwhile wait for it.
        while (m_RebuildLock.exchange(true, std::memory_order_acquire))
        {
            while (m_RebuildLock.load(std::memory_order_relaxed))
                std::this_thread::yield();
        }
        if (m_BuiltStructuralVersion.load(std::memory_order_relaxed) != structural ||
            m_BuiltPresenceVersion.load(std::memory_order_relaxed) != presence)
        {
            Rebuild();
            ++m_Counters.rebuilds;
            m_BuiltPresenceVersion.store(presence, std::memory_order_release);
            m_BuiltStructuralVersion.store(structural, std::memory_order_release);
        }
        m_RebuildLock.store(false, std::memory_order_release);
    }

    void QueryImpl::Rebuild() noexcept
    {
        WorldImpl& world = *m_World;

        // New archetypes, in creation order.
        const uint32_t archetypeCount = world.GetArchetypeStorageCount();
        if (archetypeCount < m_TestedArchetypes)
        {
            // Archetypes are never destroyed (EcsWorldImpl.h); should the storage ever be replaced, start over.
            m_Matched.clear();
            m_TermColumns.clear();
            m_Slots.clear();
            m_TestedArchetypes = 0;
        }
        for (uint32_t a = m_TestedArchetypes; a < archetypeCount; ++a)
        {
            const Archetype& archetype = world.GetArchetypeStorage(a);
            ++m_Counters.archetypesTested;
            if (Matches(archetype))
                AddMatched(archetype);
        }
        m_TestedArchetypes = archetypeCount;

        // Every non-empty chunk of the matched archetypes, in archetype-creation and chunk order.
        size_t chunkCount = 0;
        m_HasRowVersions = false;
        for (const MatchedArchetype& matched : m_Matched)
        {
            const Archetype& archetype = world.GetArchetypeStorage(matched.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                chunkCount += archetype.chunks[c].count != 0 ? 1 : 0;
            m_HasRowVersions = m_HasRowVersions || matched.trackedCount != 0;
        }
        const size_t termCount = m_Terms.size();
        m_Chunks.clear();
        m_Chunks.reserve(chunkCount);
        m_Columns.resize(chunkCount * m_DataTermCount);
        m_EnabledMasks.resize(m_HasEnableable ? chunkCount * termCount : 0);
        m_PresenceMasks.resize(m_HasSparse ? chunkCount * termCount : 0);
        m_RowVersions.resize(m_HasRowVersions ? chunkCount * termCount : 0);
        uint64_t entityCount = 0;

        for (uint32_t m = 0; m < m_Matched.size(); ++m)
        {
            const MatchedArchetype& matched = m_Matched[m];
            const Archetype& archetype = world.GetArchetypeStorage(matched.archetype);
            const uint32_t* termColumns = m_TermColumns.data() + matched.firstColumn;
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const uint32_t rows = archetype.chunks[c].count;
                if (rows == 0)
                    continue;
                const size_t index = m_Chunks.size();
                m_Chunks.push_back(ChunkEntry{ archetype.Entities(c), archetype.Versions(c), rows, m, c, 0 });
                entityCount += rows;

                void** columns = m_Columns.data() + index * m_DataTermCount;
                const uint64_t** enabled = m_HasEnableable ? m_EnabledMasks.data() + index * termCount : nullptr;
                const uint64_t** presence = m_HasSparse ? m_PresenceMasks.data() + index * termCount : nullptr;
                uint64_t** rowVersions = m_HasRowVersions ? m_RowVersions.data() + index * termCount : nullptr;
                for (size_t t = 0; t < termCount; ++t)
                {
                    const Term& term = m_Terms[t];
                    const uint32_t column = termColumns[t];
                    if (term.data)
                    {
                        void* pointer = nullptr;
                        if (column != c_NoColumn)
                        {
                            // A tag has no bytes; it gets the address GetComponent gives it.
                            pointer = archetype.columns[column].kind == StorageKind::Tag
                                          ? static_cast<void*>(g_TagValue)
                                          : archetype.ColumnData(c, column);
                        }
                        columns[term.dataIndex] = pointer;
                    }
                    if (enabled)
                        enabled[t] = column != c_NoColumn ? archetype.EnabledMask(c, column) : nullptr;
                    if (presence)
                        presence[t] = term.sparse ? world.GetSparsePresence(*term.sparse, matched.archetype, c)
                                                  : nullptr;
                    if (rowVersions)
                        rowVersions[t] = term.data && column != c_NoColumn ? archetype.RowVersions(c, column) : nullptr;
                }
            }
        }
        m_EntityCount = entityCount > 0xffffffffull ? 0xffffffffu : uint32_t(entityCount);
    }

    // ------------------------------------------------------------------------------------------------ IQuery

    uint32_t QueryImpl::GetChunkCount() noexcept
    {
        EnsureCurrent();
        return uint32_t(m_Chunks.size());
    }

    uint32_t QueryImpl::CountEntities() noexcept
    {
        EnsureCurrent();
        return m_EntityCount;
    }

    bool QueryImpl::AnyRowPasses(size_t index, uint32_t rows) const noexcept
    {
        const size_t termCount = m_Terms.size();
        const uint64_t* const* enabled = m_HasEnableable ? m_EnabledMasks.data() + index * termCount : nullptr;
        const uint64_t* const* presence = m_HasSparse ? m_PresenceMasks.data() + index * termCount : nullptr;
        const uint32_t words = (rows + 63) / 64;
        for (uint32_t w = 0; w < words; ++w)
        {
            // The chunk's rows in this word; an Enabled or Disabled term's mask exists (the term is required and
            // Enableable), and every Sparse term has a presence mask (g_ZeroMask where no row has the component).
            uint64_t bits = w + 1 < words || (rows & 63) == 0 ? ~0ull : (1ull << (rows & 63)) - 1;
            for (const RowFilter& filter : m_RowFilters)
            {
                const uint64_t* mask = filter.presence ? presence[filter.term] : enabled[filter.term];
                const uint64_t word = LoadWord(mask + w);
                bits &= filter.clear ? ~word : word;
            }
            if (bits != 0)
                return true;
        }
        return false;
    }

    void QueryImpl::ValidateRequest(const ChunkRequest& request, uint32_t capacity, const ChunkView* views) noexcept
    {
        WorldImpl& world = *m_World;
        const char* operation = "IQuery::GetChunks";
        const uint64_t current = world.GetVersion();
        if (request.sinceVersion > current)
            world.ReportValidation(operation, "query '%s': since version %llu has not been handed out yet (the world "
                                   "is at %llu)", m_DebugName.c_str(), (unsigned long long)request.sinceVersion,
                                   (unsigned long long)current);
        if (request.writeVersion > current)
            world.ReportValidation(operation, "query '%s': write version %llu has not been handed out yet (the world "
                                   "is at %llu)", m_DebugName.c_str(), (unsigned long long)request.writeVersion,
                                   (unsigned long long)current);
        if (m_HasWrite && request.writeVersion == c_WorldVersion && world.IsInPhase())
            world.ReportValidation(operation, "query '%s': c_WorldVersion is for writes outside phases; a system "
                                   "passes its own version (MakeChunkRequest)", m_DebugName.c_str());
        if (!views && capacity != 0)
            world.ReportValidation(operation, "query '%s': views is null", m_DebugName.c_str());
        if (request.system)
            ValidateSystemAccess(request.system);
    }

    void QueryImpl::ValidateSystemAccess(ISystemContext* system) noexcept
    {
        // Contexts of this module's scheduler only; an exclusive system runs alone and may access anything.
        SystemAccessInfo info{};
        if (!GetSystemAccessInfo(system, info) || info.exclusive || info.world != static_cast<IWorld*>(m_World))
            return;
        // Once per query and system run (the run's version names it), not per batch of chunks.
        if (m_AccessCheckedVersion.exchange(info.version, std::memory_order_relaxed) == info.version)
            return;
        for (const Term& term : m_Terms)
        {
            if (!term.data)
                continue;   // filters (With, Without) read no column
            const Access declared = FindDeclaredAccess(info, term.component);
            if (declared == Access::Write || (declared == Access::Read && term.access == Access::Read))
                continue;
            m_World->ReportValidation("IQuery::GetChunks", "system '%s', query '%s': %s '%s', but the system declared "
                                      "%s (11.3)", info.name, m_DebugName.c_str(),
                                      term.access == Access::Write ? "writes" : "reads",
                                      NameOf(*m_World->GetType(term.component)),
                                      declared == Access::Read ? "Read access only" : "no access to it");
        }
    }

    uint32_t QueryImpl::GetChunks(uint32_t first, const ChunkRequest& request, uint32_t capacity,
                                  ChunkView* views) noexcept
    {
        if (m_World->IsValidationEnabled())
            ValidateRequest(request, capacity, views);
        if (!views || capacity == 0)
            return 0;
        EnsureCurrent();

        const uint64_t chunkCount = m_Chunks.size();
        const uint64_t end = uint64_t(first) + capacity < chunkCount ? uint64_t(first) + capacity : chunkCount;
        const uint64_t since = request.sinceVersion;
        const uint64_t writeVersion = m_World->ResolveWriteVersion(request.writeVersion);
        const size_t termCount = m_Terms.size();
        const uint32_t* slots = m_Slots.data();
        uint32_t written = 0;
        for (uint64_t i = first; i < end; ++i)
        {
            const ChunkEntry& entry = m_Chunks[size_t(i)];
            const MatchedArchetype& matched = m_Matched[entry.matched];

            // Changed and Added, per chunk (D9): every slot newer than sinceVersion. Versions start at 1, so
            // c_AllVersions passes every chunk.
            if (since != c_AllVersions)
            {
                bool passes = true;
                for (uint32_t s = 0; s < matched.checkCount; ++s)
                {
                    if (LoadWord(entry.versions + slots[matched.firstCheck + s]) <= since)
                    {
                        passes = false;
                        break;
                    }
                }
                if (!passes)
                    continue;
            }

            // A chunk in which no row passes the row filters is neither handed out nor stamped: one selected entity
            // must not make a Write<T>, With<Selected> query mark every chunk of T as changed.
            if (!m_RowFilters.empty() && !AnyRowPasses(size_t(i), entry.count))
                continue;

            // Handing the chunk out stamps its Write columns (4.5). With RowTracked data terms: a chunk-only write is
            // spread onto the rows before the view shows them, and this hand-out's stamp is a chunk-only write itself
            // until the writer stamps its rows (the chunk-only writer rule, EcsWorldImpl.h).
            if (matched.trackedCount != 0)
            {
                const Archetype& archetype = m_World->GetArchetypeStorage(matched.archetype);
                for (uint32_t s = 0; s < matched.trackedCount; ++s)
                    SpreadChunkWrite(archetype, entry.chunk, slots[matched.firstTracked + s]);
                for (uint32_t s = 0; s < matched.writeCount; ++s)
                    StampChunkWrite(archetype, entry.chunk, slots[matched.firstWrite + s], writeVersion);
            }
            else
            {
                for (uint32_t s = 0; s < matched.writeCount; ++s)
                    StampVersion(entry.versions + slots[matched.firstWrite + s], writeVersion);
            }

            ChunkView& view = views[written++];
            view.entities = entry.entities;
            view.columns = m_DataTermCount != 0 ? m_Columns.data() + size_t(i) * m_DataTermCount : g_NoColumns;
            view.enabledMasks = m_HasEnableable ? m_EnabledMasks.data() + size_t(i) * termCount : m_NullMasks.data();
            view.presenceMasks = m_HasSparse ? m_PresenceMasks.data() + size_t(i) * termCount : m_NullMasks.data();
            view.count = entry.count;
            view.chunkIndex = uint32_t(i);
            view.rowVersions = m_HasRowVersions ? m_RowVersions.data() + size_t(i) * termCount
                                                : m_NullRowVersions.data();
        }
        return written;
    }
}

namespace donut::engine
{
    namespace
    {
        std::atomic<bool> g_QueryChunkPrefetch{ false };   // E7.4 remedy 1: off until measured
    }

    void SetQueryChunkPrefetch(bool enabled) noexcept
    {
        g_QueryChunkPrefetch.store(enabled, std::memory_order_relaxed);
    }

    bool GetQueryChunkPrefetch() noexcept
    {
        return g_QueryChunkPrefetch.load(std::memory_order_relaxed);
    }
}
