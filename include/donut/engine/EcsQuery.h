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

// Queries (aggregate docs/ecs-design.md 6.1): IQuery on the ABI surface, and the header-only typed layer on top.
//
//     Query<Read<LocalTransform>, Write<WorldTransform>, Without<Parent>> roots(world);
//     roots.ForEachChunk(context, [](const ChunkView& chunk, const LocalTransform* local, WorldTransform* world)
//     {
//         for (uint32_t i = 0; i < chunk.count; ++i)
//             world[i] = ComposeAffine(local[i]);
//     });
//     roots.ParallelForChunks(context, /* chunksPerTask */ 4, ...);
//     roots.ForEach(context, [](Entity e, const LocalTransform& local, WorldTransform& world) { ... });
//
// The typed layer calls IQuery once per batch of chunks, never per entity (rule 3.3.4), and hoists each chunk's column
// pointers before its row loop, so the loop is plain array code (10.3).

#pragma once

#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <nvrhi/core/autoptr.h>

#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace donut::engine
{
    // What a chunk request is for. Versions are explicit, never inferred from the calling thread (EcsScheduler.h).
    struct ChunkRequest
    {
        uint64_t sinceVersion = c_AllVersions;   // the Changed and Added filters
        uint64_t writeVersion = c_WorldVersion;  // what handing out a chunk stamps on its Write columns (4.5)
        ISystemContext* system = nullptr;        // optional: the running system, for the access checks of 11.3
    };
    static_assert(sizeof(void*) != 8 || sizeof(ChunkRequest) == 24, "ChunkRequest layout");

    // The request of a running system: its last run's version, its own version, and itself.
    inline ChunkRequest MakeChunkRequest(ISystemContext* context) noexcept
    {
        return ChunkRequest{ context->GetLastRunVersion(), context->GetVersion(), context };
    }

    // A cached query: the set of matching archetypes is updated as archetypes are created. CreateQuery rejects Write
    // terms on Shared components and data terms on Sparse components (FE_INVALID_ARGS).
    NVRHI_IID(IQuery, "3600c8b4-c1c1-4e71-ac53-d16271a7c78d")
    struct IQuery : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IQuery)

        // The chunks of the matching archetypes, in archetype-creation and chunk order. The list is refreshed at sync
        // points and immediate structural changes; it does not change while a phase runs.
        virtual uint32_t GetChunkCount() noexcept = 0;
        // Examines the matched chunks [first, first + capacity) and writes, in order, the views of those that pass the
        // Changed and Added filters and have at least one row that passes the row filters (Enabled, Disabled, and
        // With and Without on Sparse components); returns how many it wrote. Advance `first` by `capacity`. Handing out
        // a chunk stamps its Write columns with request.writeVersion; a chunk that is not handed out is not stamped.
        // Within a chunk handed out, rows are not filtered (ChunkView).
        virtual uint32_t GetChunks(uint32_t first, const ChunkRequest& request, uint32_t capacity,
                                   _Out_writes_to_(capacity, return) ChunkView* views) noexcept = 0;
        // Rows in the matched chunks; ignores the Changed and Added filters and the row filters.
        virtual uint32_t CountEntities() noexcept = 0;
        virtual uint32_t GetTermCount() noexcept = 0;
        virtual uint32_t GetDataTermCount() noexcept = 0;
    };

    // Chunk prefetch in the typed loops (plan step E7, item E7.4, remedy 1; sub-plan revision 2, part I5): while the
    // callback runs on a chunk, the first bytes of the next chunk's data columns (and its entities, in the row forms) are
    // fetched, the next chunk of the same IQuery::GetChunks batch. A process-wide switch (one per module that links
    // donut_engine statically), off by default, which stays off until measured (revision 2, stage B). ForEachChunk,
    // ForEach, ParallelForChunks and ParallelForEach read it once per call; a call keeps what it read. It changes no
    // result (the same chunks, rows, order and stamps), only memory traffic. Set it on any thread.
    void SetQueryChunkPrefetch(bool enabled) noexcept;
    [[nodiscard]] bool GetQueryChunkPrefetch() noexcept;

    // ------------------------------------------------------------ typed terms (header-only, outside the ABI)

    namespace detail
    {
        template <typename T, TermOp Op, Access A, bool PerChunk = false, ComponentFlags Required = ComponentFlags{}>
        struct TermBase
        {
            using Component = T;
            static constexpr TermOp op = Op;
            static constexpr Access access = A;
            static constexpr bool perChunk = PerChunk;           // one value per chunk: Shared or Chunk components
            static constexpr ComponentFlags required = Required;
        };
    }

    // Per-row terms. Shared and Chunk components use the per-chunk terms below; Sparse components only With/Without.
    template <typename T> struct Read     : detail::TermBase<T, TermOp::All,      Access::Read>   { };
    template <typename T> struct Write    : detail::TermBase<T, TermOp::All,      Access::Write>  { };
    template <typename T> struct With     : detail::TermBase<T, TermOp::All,      Access::Filter> { };
    template <typename T> struct Without  : detail::TermBase<T, TermOp::Without,  Access::Filter> { };
    template <typename T> struct Optional : detail::TermBase<T, TermOp::Optional, Access::Read>   { };   // may be null
    template <typename T> struct Changed  : detail::TermBase<T, TermOp::Changed,  Access::Read>   { };
    template <typename T> struct Added    : detail::TermBase<T, TermOp::Added,    Access::Read>   { };
    template <typename T> struct Enabled  : detail::TermBase<T, TermOp::Enabled,  Access::Read>   { };
    template <typename T> struct Disabled : detail::TermBase<T, TermOp::Disabled, Access::Read>   { };

    // Per-chunk terms: the chunk's one value. Shared values are read-only (they partition chunks, 4.4).
    template <typename T> struct ReadShared : detail::TermBase<T, TermOp::All, Access::Read,  true, ComponentFlags::Shared> { };
    template <typename T> struct ReadChunk  : detail::TermBase<T, TermOp::All, Access::Read,  true, ComponentFlags::Chunk>  { };
    template <typename T> struct WriteChunk : detail::TermBase<T, TermOp::All, Access::Write, true, ComponentFlags::Chunk>  { };

    namespace detail
    {
        template <typename T> struct ColumnPointer { using Const = const T*; using Mutable = T*; };
        template <typename E> struct ColumnPointer<BufferOf<E>> { using Const = const BufferHeader*; using Mutable = BufferHeader*; };
        template <typename T> struct IsBufferOf : std::false_type { };
        template <typename E> struct IsBufferOf<BufferOf<E>> : std::true_type { };

        // Row filters the typed layer applies: 1 Enabled, 2 Disabled, 3 a Sparse component present, 4 one absent.
        enum : uint8_t { RowFilterNone = 0, RowFilterEnabled = 1, RowFilterDisabled = 2, RowFilterPresent = 3, RowFilterAbsent = 4 };

        template <typename Term>
        struct TermTraits
        {
            using Component = typename Term::Component;
            static constexpr ComponentFlags flags = ComponentTraits<Component>::flags;
            static constexpr bool isSparse = !!(flags & ComponentFlags::Sparse);
            static constexpr bool isPerChunkType = !!(flags & (ComponentFlags::Shared | ComponentFlags::Chunk));
            static constexpr bool hasData = Term::access != Access::Filter && Term::op != TermOp::Without;
            static constexpr bool optional = Term::op == TermOp::Optional;
            static constexpr bool perChunk = Term::perChunk;
            static constexpr uint8_t rowFilter =
                Term::op == TermOp::Enabled ? RowFilterEnabled :
                Term::op == TermOp::Disabled ? RowFilterDisabled :
                isSparse && Term::op == TermOp::All ? RowFilterPresent :
                isSparse && Term::op == TermOp::Without ? RowFilterAbsent : RowFilterNone;
            using Pointer = std::conditional_t<Term::access == Access::Write,
                                               typename ColumnPointer<Component>::Mutable,
                                               typename ColumnPointer<Component>::Const>;

            static_assert(!hasData || !isSparse,
                          "Sparse components are filters only (With, Without); read them with IWorld::GetComponent");
            static_assert(!hasData || perChunk || !isPerChunkType,
                          "use ReadShared, ReadChunk or WriteChunk for Shared and Chunk components");
            static_assert(!perChunk || (flags & Term::required) == Term::required,
                          "ReadShared needs a Shared component, ReadChunk and WriteChunk a Chunk component");
            static_assert(!hasData || IsBufferOf<Component>::value == !!(flags & ComponentFlags::Buffer),
                          "use BufferOf<T> for Buffer components, and only for them");
        };

        template <size_t I, typename T, typename... Ts> struct NthType { using Type = typename NthType<I - 1, Ts...>::Type; };
        template <typename T, typename... Ts> struct NthType<0, T, Ts...> { using Type = T; };

        template <typename... Terms>
        struct TermList
        {
            static constexpr uint32_t count = uint32_t(sizeof...(Terms));
            static constexpr bool hasData[sizeof...(Terms)] = { TermTraits<Terms>::hasData... };
            static constexpr uint8_t rowFilter[sizeof...(Terms)] = { TermTraits<Terms>::rowFilter... };
            static constexpr uint32_t dataCount = (0u + ... + uint32_t(TermTraits<Terms>::hasData));
            static constexpr bool anyRowFilter = (false || ... || (TermTraits<Terms>::rowFilter != RowFilterNone));

            // The term index of the k-th data term.
            static constexpr uint32_t DataTermIndex(uint32_t k) noexcept
            {
                for (uint32_t i = 0; i < count; ++i)
                    if (hasData[i] && k-- == 0)
                        return i;
                return count;
            }
        };

        template <typename Term>
        QueryTerm MakeTerm(IWorld* world) noexcept
        {
            return QueryTerm{ RegisterComponent<typename Term::Component>(world), Term::op, Term::access, {} };
        }

        // SetQueryChunkPrefetch: a hint per cache line, and at most c_PrefetchBytesPerColumn of each column (the start of
        // its stream, where the hardware prefetcher has no history yet; the next chunk is another page).
        constexpr size_t c_PrefetchLineBytes = 64;
        constexpr size_t c_PrefetchBytesPerColumn = 1024;

        inline void PrefetchLine(const void* address) noexcept
        {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(address);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
            _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#else
            (void)address;
#endif
        }

        inline void PrefetchBytes(const void* address, size_t bytes) noexcept
        {
            const char* const first = static_cast<const char*>(address);
            const size_t count = bytes < c_PrefetchBytesPerColumn ? bytes : c_PrefetchBytesPerColumn;
            for (size_t offset = 0; offset < count; offset += c_PrefetchLineBytes)
                PrefetchLine(first + offset);
        }
    }

    template <typename... Terms>
    class Query
    {
        static_assert(sizeof...(Terms) > 0, "a query needs at least one term");
        using List = detail::TermList<Terms...>;
        template <size_t K>
        using DataTerm = detail::TermTraits<typename detail::NthType<List::DataTermIndex(uint32_t(K)), Terms...>::Type>;

    public:
        Query() noexcept = default;
        explicit Query(IWorld* world, const char* debugName = nullptr) noexcept { Create(world, debugName); }

        // Registers the component types the terms name, if needed, and creates the query.
        nvrhi::FRESULT Create(IWorld* world, const char* debugName = nullptr) noexcept
        {
            const QueryTerm terms[List::count] = { detail::MakeTerm<Terms>(world)... };
            for (const QueryTerm& term : terms)
                if (term.component == ComponentId::Invalid)
                    return nvrhi::FE_INVALID_ARGS;
            const QueryDesc desc{ terms, List::count, 0, debugName };
            return world->CreateQuery(desc, &m_Query);
        }

        [[nodiscard]] IQuery* Get() const noexcept { return m_Query.Get(); }
        explicit operator bool() const noexcept { return m_Query.Get() != nullptr; }
        [[nodiscard]] uint32_t CountEntities() const noexcept { return m_Query->CountEntities(); }

        // f(const ChunkView&, Pointer...) for each chunk IQuery::GetChunks hands out, in chunk order. One pointer
        // per data term: const T* for reads, T* for Write, BufferHeader* for BufferOf terms, and the chunk's one value
        // for per-chunk terms. Rows are not filtered: test RowPasses(view, row) where the query has Enabled, Disabled
        // or Sparse terms.
        template <typename F>
        void ForEachChunk(ISystemContext* context, F&& f) const
        {
            ForEachChunk(MakeChunkRequest(context), f);
        }

        template <typename F>
        void ForEachChunk(const ChunkRequest& request, F&& f) const
        {
            if (GetQueryChunkPrefetch())
                RunSerial<false, true>(request, f);
            else
                RunSerial<false, false>(request, f);
        }

        // f(Entity, Ref...) for each row that passes the row filters. One argument per data term: const T& for reads,
        // T& for Write, const T* (null when absent) for Optional, the chunk's one value for per-chunk terms.
        template <typename F>
        void ForEach(ISystemContext* context, F&& f) const
        {
            ForEach(MakeChunkRequest(context), f);
        }

        template <typename F>
        void ForEach(const ChunkRequest& request, F&& f) const
        {
            if (GetQueryChunkPrefetch())
                RunSerial<true, true>(request, f);
            else
                RunSerial<true, false>(request, f);
        }

        // Splits the matched chunks into tasks of `chunksPerTask` consecutive chunks on the context's job system and
        // returns when all have run. The split is deterministic (JobSystem.h); f runs concurrently for different
        // chunks, and commands it records use MakeSortKey(loop, chunk.chunkIndex, row).
        template <typename F>
        void ParallelForChunks(ISystemContext* context, uint32_t chunksPerTask, F&& f) const
        {
            RunParallel<false>(context, chunksPerTask, f);
        }

        // ForEach, split like ParallelForChunks.
        template <typename F>
        void ParallelForEach(ISystemContext* context, uint32_t chunksPerTask, F&& f) const
        {
            RunParallel<true>(context, chunksPerTask, f);
        }

        // Whether a row passes the Enabled, Disabled and Sparse terms; for callbacks of ForEachChunk.
        static bool RowPasses(const ChunkView& view, uint32_t row) noexcept
        {
            if constexpr (List::anyRowFilter)
            {
                for (uint32_t t = 0; t < List::count; ++t)
                {
                    switch (List::rowFilter[t])
                    {
                    case detail::RowFilterEnabled:
                        if (!view.IsRowEnabled(t, row))
                            return false;
                        break;
                    case detail::RowFilterDisabled:
                        if (view.IsRowEnabled(t, row))
                            return false;
                        break;
                    case detail::RowFilterPresent:
                        if (!view.IsRowPresent(t, row))
                            return false;
                        break;
                    case detail::RowFilterAbsent:
                        if (view.presenceMasks[t] && view.IsRowPresent(t, row))
                            return false;
                        break;
                    default:
                        break;
                    }
                }
            }
            (void)view;
            (void)row;
            return true;
        }

    private:
        static constexpr uint32_t c_ChunkBatch = 16;

        template <typename F, size_t... K>
        static void InvokeChunk(F& f, const ChunkView& view, std::index_sequence<K...>)
        {
            f(view, static_cast<typename DataTerm<K>::Pointer>(view.columns[K])...);
        }

        template <typename F, size_t... K>
        static void InvokeRows(F& f, const ChunkView& view, std::index_sequence<K...> sequence)
        {
            RowLoop(f, view, sequence, static_cast<typename DataTerm<K>::Pointer>(view.columns[K])...);
        }

        // The column pointers arrive once per chunk; the loop indexes them directly.
        template <typename F, size_t... K, typename... Pointers>
        static void RowLoop(F& f, const ChunkView& view, std::index_sequence<K...>, Pointers... columns)
        {
            const Entity* const entities = view.entities;
            const uint32_t count = view.count;
            for (uint32_t row = 0; row < count; ++row)
            {
                if constexpr (List::anyRowFilter)
                {
                    if (!RowPasses(view, row))
                        continue;
                }
                f(entities[row], RowArgument<K>(columns, row)...);
            }
        }

        template <size_t K, typename Pointer>
        static decltype(auto) RowArgument(Pointer column, uint32_t row) noexcept
        {
            using Term = DataTerm<K>;
            if constexpr (Term::perChunk)
                return *column;
            else if constexpr (Term::optional)
                return column ? column + row : column;
            else
                return column[row];
        }

        // The chunks of one GetChunks batch, each to f (the chunk form or the row form); with Prefetch, the next chunk
        // of the batch is prefetched before f runs on this one. Without, the plain loop.
        template <bool PerRow, bool Prefetch, typename F>
        static void RunBatch(F& f, const ChunkView* views, uint32_t written)
        {
            for (uint32_t i = 0; i < written; ++i)
            {
                if constexpr (Prefetch)
                {
                    if (i + 1 < written)
                        PrefetchChunk<PerRow>(views[i + 1], std::make_index_sequence<List::dataCount>());
                }
                if constexpr (PerRow)
                    InvokeRows(f, views[i], std::make_index_sequence<List::dataCount>());
                else
                    InvokeChunk(f, views[i], std::make_index_sequence<List::dataCount>());
            }
        }

        template <bool PerRow, size_t... K>
        static void PrefetchChunk(const ChunkView& view, std::index_sequence<K...>) noexcept
        {
            if constexpr (PerRow)
                detail::PrefetchBytes(view.entities, size_t(view.count) * sizeof(Entity));
            (PrefetchColumn<K>(view), ...);
            (void)view;
        }

        template <size_t K>
        static void PrefetchColumn(const ChunkView& view) noexcept
        {
            using Term = DataTerm<K>;
            const void* const column = view.columns[K];
            if (!column)
                return;   // an Optional term the chunk does not have
            if constexpr (Term::perChunk)
                detail::PrefetchLine(column);
            else
                detail::PrefetchBytes(column, size_t(view.count) *
                                                  sizeof(std::remove_pointer_t<typename Term::Pointer>));
        }

        template <bool PerRow, bool Prefetch, typename F>
        void RunSerial(const ChunkRequest& request, F& f) const
        {
            ChunkView views[c_ChunkBatch];
            const uint32_t chunkCount = m_Query->GetChunkCount();
            for (uint32_t first = 0; first < chunkCount; first += c_ChunkBatch)
            {
                const uint32_t written = m_Query->GetChunks(first, request, c_ChunkBatch, views);
                RunBatch<PerRow, Prefetch>(f, views, written);
            }
        }

        // A task of RunParallel (a member template: a local class cannot have one).
        template <bool PerRow, bool Prefetch, typename F>
        struct ParallelTask
        {
            IQuery* query;
            F* function;
            ChunkRequest request;

            static void Run(void* taskContext, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                const ParallelTask* task = static_cast<const ParallelTask*>(taskContext);
                ChunkView views[c_ChunkBatch];
                for (uint32_t first = begin; first < end; first += c_ChunkBatch)
                {
                    const uint32_t capacity = end - first < c_ChunkBatch ? end - first : c_ChunkBatch;
                    const uint32_t written = task->query->GetChunks(first, task->request, capacity, views);
                    RunBatch<PerRow, Prefetch>(*task->function, views, written);
                }
            }
        };

        template <bool PerRow, typename F>
        void RunParallel(ISystemContext* context, uint32_t chunksPerTask, F& f) const
        {
            if (GetQueryChunkPrefetch())
                RunParallelTasks<PerRow, true>(context, chunksPerTask, f);
            else
                RunParallelTasks<PerRow, false>(context, chunksPerTask, f);
        }

        template <bool PerRow, bool Prefetch, typename F>
        void RunParallelTasks(ISystemContext* context, uint32_t chunksPerTask, F& f) const
        {
            using Task = ParallelTask<PerRow, Prefetch, F>;
            Task task{ m_Query.Get(), &f, MakeChunkRequest(context) };
            context->GetJobSystem()->ParallelFor(m_Query->GetChunkCount(), chunksPerTask, &Task::Run, &task);
        }

        nvrhi::AutoPtr<IQuery> m_Query;
    };
}
