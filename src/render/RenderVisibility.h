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

// Visibility and draw packets of the render service (RenderExtract.h; aggregate docs/ecs-design.md 7.3, D11, 10.1;
// sub-plan docs/plans/2026-11-21-donut-ecs-e5.md, item E5.2). Module-private to donut_render: included by
// RenderExtract.cpp, whose RenderService.Instances system runs it, and by the device-free test of the packets (the
// sort's own cases). A header, not a translation unit of its own, so that it builds wherever RenderExtract.cpp does.
//
// Per view, in parallel by chunk ranges of the instance query: a chunk test against ChunkWorldBounds, then a row test
// against WorldBounds, both with dm::frustum::intersectsWith, the legacy strategies' test (same planes from
// IView::GetViewFrustum, same box test; an empty box, FLT_MAX to -FLT_MAX, fails every plane with a normal). The chunk
// test only skips work: the plane distance of a union box is at most that of any of its boxes (each product and sum is
// monotonic), so a chunk that fails holds no row that passes. The per-geometry rules are the legacy ones (open point
// 7): InstancedOpaqueDrawStrategy tests each geometry of a mesh with more than one geometry and no skin prototype,
// TransparentDrawStrategy each geometry of a mesh with more than one geometry and a skin prototype; the transparent
// distance is length(box.center() - viewOrigin) in float, the box being that geometry's, or else the node's subgraph
// box (SceneGraphNode::GetGlobalBoundingBox, read only for a node with children, where it is a lock-free read of the
// values Refresh stored in the node; for a leaf node it equals the row's WorldBounds). The pass of each geometry is its
// material's domain now (no gate by RenderFlags::contentFlags, so that a domain edited without InvalidateContent moves
// the geometry to the other pass rather than dropping it). Slots come from this graph's tables (MeshTable rows: the
// mesh's first geometry slot and its materials' ids), not from MeshGeometry::globalGeometryIndex and
// Material::materialID, which a mesh or material shared with another graph holds for that graph.
//
// Work and memory (open point 2: the frame arena is not thread-safe):
//   1. in parallel, one task per range of `grain` chunks: each range lists its visible instances and its packets per
//      view and pass in its own scratch (heap, capacity kept), in chunk and row order;
//   2. serially: prefix sums over the ranges, then the snapshot's arrays in the arena;
//   3. in parallel by range: the instances into the arena, the packets' keys and records into the sort's input;
//   4. per view and pass, a stable LSD radix sort of the keys, so equal keys keep chunk order: the views' and passes'
//      sorts side by side, a large one on the job system inside too;
//   5. in parallel: the packets gathered into the arena's arrays in key order.
// With compact packet records (RenderService::SetCompactPacketRecords; plan step E7, item E7.4, remedy 2; off by
// default) a range lists 24-byte records (CompactPacketRecord) instead of 32-byte ones; step 2 also prepares each
// segment's sort; step 3 copies the records into one array and writes the sort's words itself (the key's varying
// bits and the position), so that the sort reads no keys; step 5 gathers from that array. The output is the same.
// The output does not depend on the worker count or the range size. Arena arrays are sized by high-water marks with a
// quarter of headroom (the arena reuses its blocks only while a frame asks for no more than before), heap scratch keeps
// its capacity and grows with the same headroom: once the marks have settled, no allocation.

#pragma once

#include <donut/render/RenderExtract.h>
#include <donut/engine/AssetTables.h>
#include <donut/engine/EcsQuery.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace donut::render::visibility
{
    // ------------------------------------------------------------------------------------------ scratch storage

    // A growable array of trivially copyable records: capacity kept, elements not initialised. It grows through
    // operator new (nothrow), so the tests' allocation counters see it like any other heap block.
    template <typename T>
    class PodBuffer
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>, "plain records only");

    public:
        PodBuffer() noexcept = default;
        PodBuffer(PodBuffer&& other) noexcept
            : m_Data(other.m_Data)
            , m_Capacity(other.m_Capacity)
        {
            other.m_Data = nullptr;
            other.m_Capacity = 0;
        }
        PodBuffer& operator=(PodBuffer&& other) noexcept
        {
            std::swap(m_Data, other.m_Data);
            std::swap(m_Capacity, other.m_Capacity);
            return *this;
        }
        PodBuffer(const PodBuffer&) = delete;
        PodBuffer& operator=(const PodBuffer&) = delete;
        ~PodBuffer() { ::operator delete(m_Data); }

        // At least `capacity` elements; the first `keep` survive a growth. A growth takes a quarter more than asked,
        // so that a count that wanders around its high-water mark does not grow the buffer again and again.
        bool Reserve(size_t capacity, size_t keep = 0) noexcept
        {
            if (capacity <= m_Capacity)
                return true;
            const size_t grown = capacity + capacity / 4 + 64;
            if (grown > SIZE_MAX / sizeof(T))
                return false;
            T* data = static_cast<T*>(::operator new(grown * sizeof(T), std::nothrow));
            if (!data)
                return false;
            if (keep && m_Data)
                std::memcpy(static_cast<void*>(data), m_Data, std::min(keep, m_Capacity) * sizeof(T));
            ::operator delete(m_Data);
            m_Data = data;
            m_Capacity = grown;
            return true;
        }

        [[nodiscard]] T* Data() const noexcept { return m_Data; }
        [[nodiscard]] size_t Capacity() const noexcept { return m_Capacity; }

    private:
        T* m_Data = nullptr;
        size_t m_Capacity = 0;
    };

    // Tasks of a parallel loop: about four per thread, as the service's chunk loops.
    inline uint32_t ThreadsOf(engine::IJobSystem* jobs) noexcept
    {
        uint32_t threads = jobs ? jobs->GetThreadCount() : 1;
        const uint32_t hardware = std::thread::hardware_concurrency();
        if (hardware != 0)
            threads = std::min(threads, hardware);
        return std::max(1u, threads);
    }

    // ------------------------------------------------------------------------------------------ the radix sort

    // The bits of a mask as runs, low to high. Compact() keeps the bits under the mask, in their order: for keys whose
    // bits outside the mask are all equal (the mask covers every bit that varies), compacted keys order as the keys do.
    struct BitRuns
    {
        uint8_t shift[64] = {};
        uint8_t length[64] = {};
        uint32_t count = 0;
        uint32_t bits = 0;

        explicit BitRuns(uint64_t mask) noexcept
        {
            uint32_t bit = 0;
            while (bit < 64)
            {
                if (!((mask >> bit) & 1))
                {
                    ++bit;
                    continue;
                }
                const uint32_t start = bit;
                while (bit < 64 && ((mask >> bit) & 1))
                    ++bit;
                shift[count] = uint8_t(start);
                length[count] = uint8_t(bit - start);
                ++count;
                bits += bit - start;
            }
        }

        static uint64_t Low(uint32_t n) noexcept { return n >= 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1; }

        uint64_t Compact(uint64_t key) const noexcept
        {
            uint64_t out = 0;
            uint32_t at = 0;
            for (uint32_t i = 0; i < count; ++i)
            {
                out |= ((key >> shift[i]) & Low(length[i])) << at;
                at += length[i];
            }
            return out;
        }
    };

    // A stable LSD radix sort of 64-bit keys that gives the order, not the keys. Only the bits that vary are sorted:
    // the keys are compacted to them (BitRuns), and when the compacted key and the element's position fit 64 bits
    // together, one word per element carries both (the position below the key, which also makes the order stable).
    // Digits of up to 12 bits. Serially, one read of the input fills the elements and every pass's histogram. A large
    // input sorts on the job system: per task a histogram of its block, then the blocks' offsets in task order, then
    // each block scatters; that is the serial pass's result exactly.
    class RadixSorter
    {
    public:
        static constexpr uint32_t c_MaxDigitBits = 12;
        static constexpr uint32_t c_MaxPasses = (64 + c_MaxDigitBits - 1) / c_MaxDigitBits;
        static constexpr uint32_t c_ParallelThreshold = 1u << 14;   // elements
        static constexpr uint32_t c_ElementsPerTask = 1u << 12;
        static constexpr uint32_t c_MaxTasks = 32;

        RadixSorter() noexcept = default;
        RadixSorter(RadixSorter&&) noexcept = default;
        RadixSorter& operator=(RadixSorter&&) noexcept = default;

        // order[i] = base + the position in `keys` of the i-th smallest key; equal keys keep their positions' order.
        // `varying`: a superset of the bits that differ between the keys (the OR of all keys XOR their AND). `jobs` may
        // be null (serial). False when out of memory (order is then unspecified).
        bool Sort(const uint64_t* keys, uint32_t count, uint64_t varying, uint32_t base, uint32_t* order,
                  engine::IJobSystem* jobs) noexcept
        {
            return Prepare(count, varying) && Order(keys, base, order, jobs);
        }

        // Sort in two steps, for a caller that writes the sort words itself (the compact packet records: the words
        // are written in the copy step, so that the keys are read once). Prepare, then for each element i with key k:
        // Words()[i] = Word(k, i), and Indices()[i] = i when Indices() is not null; then Finish, whose order is Sort's
        // exactly. Words() is null when the order is the identity (fewer than two elements, or no varying bit): then
        // nothing is written. Between Prepare and Finish, different elements may be written from different threads.
        bool Prepare(uint32_t count, uint64_t varying) noexcept
        {
            m_Count = count;
            m_Runs = BitRuns(varying);
            m_Identity = count <= 1 || m_Runs.bits == 0;
            m_Packed = true;
            m_PositionBits = 0;
            m_IndexBits = 0;
            if (m_Identity)
                return true;
            uint32_t indexBits = 0;
            while (indexBits < 32 && (uint64_t(count - 1) >> indexBits) != 0)
                ++indexBits;
            m_IndexBits = indexBits;
            m_Packed = m_Runs.bits + indexBits <= 64;
            m_PositionBits = m_Packed ? indexBits : 0;
            if (!m_A.Reserve(count) || !m_B.Reserve(count))
                return false;
            if (!m_Packed && (!m_IndexA.Reserve(count) || !m_IndexB.Reserve(count)))
                return false;
            return true;
        }

        [[nodiscard]] uint64_t* Words() const noexcept { return m_Identity ? nullptr : m_A.Data(); }
        [[nodiscard]] uint32_t* Indices() const noexcept { return m_Identity || m_Packed ? nullptr : m_IndexA.Data(); }
        [[nodiscard]] const BitRuns& Runs() const noexcept { return m_Runs; }
        // The position's bits in a word: those of count - 1 when packed (Indices() null), else 0.
        [[nodiscard]] uint32_t PositionBits() const noexcept { return m_PositionBits; }
        // The word of a key: its compacted bits, with the element's position below them when packed.
        [[nodiscard]] uint64_t Word(uint64_t key, uint32_t position) const noexcept
        {
            return m_Packed ? (m_Runs.Compact(key) << m_PositionBits) | position : m_Runs.Compact(key);
        }

        bool Finish(uint32_t base, uint32_t* order, engine::IJobSystem* jobs) noexcept
        {
            return Order(nullptr, base, order, jobs);
        }

    private:
        // The passes over the prepared elements; with `keys`, the elements are filled from them first (Sort).
        bool Order(const uint64_t* keys, uint32_t base, uint32_t* order, engine::IJobSystem* jobs) noexcept
        {
            const uint32_t count = m_Count;
            if (count == 0)
                return true;
            if (m_Identity)
            {
                for (uint32_t i = 0; i < count; ++i)
                    order[i] = base + i;
                return true;
            }

            uint32_t tasks = 1;
            if (jobs && count >= c_ParallelThreshold)
                tasks = std::min(std::min(ThreadsOf(jobs), c_MaxTasks), std::max(1u, count / c_ElementsPerTask));
            if (tasks > 1 && !m_Histograms.Reserve(size_t(tasks) << c_MaxDigitBits))
                return false;

            Job job;
            job.keys = keys;
            job.count = count;
            job.runs = &m_Runs;
            job.indexBits = m_PositionBits;
            job.tasks = tasks;
            job.block = (count + tasks - 1) / tasks;
            job.a = m_A.Data();
            job.b = m_B.Data();
            job.indexA = m_Packed ? nullptr : m_IndexA.Data();
            job.indexB = m_Packed ? nullptr : m_IndexB.Data();
            job.passes = (m_Runs.bits + c_MaxDigitBits - 1) / c_MaxDigitBits;
            job.width = (m_Runs.bits + job.passes - 1) / job.passes;

            if (tasks > 1)
            {
                job.histograms = m_Histograms.Data();
                if (keys)
                    jobs->ParallelFor(tasks, 1, &Job::Fill, &job);
                for (uint32_t p = 0; p < job.passes; ++p)
                {
                    job.Pass(p);
                    jobs->ParallelFor(tasks, 1, &Job::Histogram, &job);
                    job.Offsets();
                    jobs->ParallelFor(tasks, 1, &Job::Scatter, &job);
                    job.Swap();
                }
            }
            else
            {
                // every pass's histogram in the one read of the input; the global counts do not depend on the order
                if (!m_Histograms.Reserve(size_t(job.passes) << c_MaxDigitBits))
                    return false;
                uint32_t* histograms = m_Histograms.Data();
                std::fill(histograms, histograms + (size_t(job.passes) << c_MaxDigitBits), 0u);
                if (keys)
                    job.FillRange(0, count);
                job.Count(histograms);
                for (uint32_t p = 0; p < job.passes; ++p)
                {
                    job.Pass(p);
                    job.histograms = histograms + (size_t(p) << c_MaxDigitBits);
                    job.Offsets();
                    Job::Scatter(&job, 0, 1, 0);
                    job.Swap();
                }
            }

            if (m_Packed)
            {
                const uint64_t mask = BitRuns::Low(m_IndexBits);
                const uint64_t* a = job.a;
                for (uint32_t i = 0; i < count; ++i)
                    order[i] = base + uint32_t(a[i] & mask);
            }
            else
            {
                const uint32_t* index = job.indexA;
                for (uint32_t i = 0; i < count; ++i)
                    order[i] = base + index[i];
            }
            return true;
        }

        struct Job
        {
            const uint64_t* keys = nullptr;
            uint32_t count = 0;
            const BitRuns* runs = nullptr;
            uint32_t indexBits = 0;
            uint32_t tasks = 1;
            uint32_t block = 0;
            uint32_t passes = 0;
            uint32_t width = 0;
            uint64_t* a = nullptr;
            uint64_t* b = nullptr;
            uint32_t* indexA = nullptr;
            uint32_t* indexB = nullptr;
            uint32_t* histograms = nullptr;   // per task a row of 2^c_MaxDigitBits (serially: the pass's histogram)
            uint32_t shift = 0;
            uint32_t digitBits = 0;

            void Pass(uint32_t p) noexcept
            {
                shift = indexBits + p * width;
                digitBits = std::min(width, runs->bits - p * width);
            }

            void Swap() noexcept
            {
                std::swap(a, b);
                std::swap(indexA, indexB);
            }

            void Bounds(uint32_t task, uint32_t& begin, uint32_t& end) const noexcept
            {
                begin = std::min(count, task * block);
                end = std::min(count, begin + block);
            }

            // The compacted key of element i, with the position below it when packed.
            void FillRange(uint32_t begin, uint32_t end) noexcept
            {
                const BitRuns local = *runs;
                const uint64_t* in = keys;
                uint64_t* out = a;
                if (!indexA)
                {
                    const uint32_t positionBits = indexBits;
                    for (uint32_t i = begin; i < end; ++i)
                        out[i] = (local.Compact(in[i]) << positionBits) | i;
                }
                else
                {
                    uint32_t* index = indexA;
                    for (uint32_t i = begin; i < end; ++i)
                    {
                        out[i] = local.Compact(in[i]);
                        index[i] = i;
                    }
                }
            }

            static void Fill(void* context, uint32_t first, uint32_t last, uint32_t) noexcept
            {
                Job& job = *static_cast<Job*>(context);
                for (uint32_t t = first; t < last; ++t)
                {
                    uint32_t begin, end;
                    job.Bounds(t, begin, end);
                    job.FillRange(begin, end);
                }
            }

            // Serially: every pass's digit counts, in one read of the elements.
            void Count(uint32_t* passHistograms) noexcept
            {
                const uint64_t* elements = a;
                uint32_t shifts[c_MaxPasses] = {};
                uint64_t masks[c_MaxPasses] = {};
                for (uint32_t p = 0; p < passes; ++p)
                {
                    shifts[p] = indexBits + p * width;
                    masks[p] = BitRuns::Low(std::min(width, runs->bits - p * width));
                }
                const uint32_t n = count;
                const uint32_t passCount = passes;
                for (uint32_t i = 0; i < n; ++i)
                {
                    const uint64_t element = elements[i];
                    for (uint32_t p = 0; p < passCount; ++p)
                        ++passHistograms[(size_t(p) << c_MaxDigitBits) + ((element >> shifts[p]) & masks[p])];
                }
            }

            static void Histogram(void* context, uint32_t first, uint32_t last, uint32_t) noexcept
            {
                Job& job = *static_cast<Job*>(context);
                const uint64_t mask = BitRuns::Low(job.digitBits);
                const uint32_t shift = job.shift;
                const uint64_t* elements = job.a;
                for (uint32_t t = first; t < last; ++t)
                {
                    uint32_t begin, end;
                    job.Bounds(t, begin, end);
                    uint32_t* histogram = job.histograms + (size_t(t) << c_MaxDigitBits);
                    std::fill(histogram, histogram + (size_t(1) << job.digitBits), 0u);
                    for (uint32_t i = begin; i < end; ++i)
                        ++histogram[(elements[i] >> shift) & mask];
                }
            }

            // Exclusive prefix over (digit, task): each task's first position per digit.
            void Offsets() noexcept
            {
                uint32_t running = 0;
                const uint32_t digits = 1u << digitBits;
                for (uint32_t d = 0; d < digits; ++d)
                {
                    for (uint32_t t = 0; t < tasks; ++t)
                    {
                        uint32_t& slot = histograms[(size_t(t) << c_MaxDigitBits) + d];
                        const uint32_t n = slot;
                        slot = running;
                        running += n;
                    }
                }
            }

            static void Scatter(void* context, uint32_t first, uint32_t last, uint32_t) noexcept
            {
                Job& job = *static_cast<Job*>(context);
                const uint64_t mask = BitRuns::Low(job.digitBits);
                const uint32_t shift = job.shift;
                const uint64_t* from = job.a;
                uint64_t* to = job.b;
                for (uint32_t t = first; t < last; ++t)
                {
                    uint32_t begin, end;
                    job.Bounds(t, begin, end);
                    uint32_t* positions = job.histograms + (size_t(t) << c_MaxDigitBits);
                    if (job.indexA)
                    {
                        const uint32_t* indexFrom = job.indexA;
                        uint32_t* indexTo = job.indexB;
                        for (uint32_t i = begin; i < end; ++i)
                        {
                            const uint64_t element = from[i];
                            const uint32_t at = positions[(element >> shift) & mask]++;
                            to[at] = element;
                            indexTo[at] = indexFrom[i];
                        }
                    }
                    else
                    {
                        for (uint32_t i = begin; i < end; ++i)
                        {
                            const uint64_t element = from[i];
                            to[positions[(element >> shift) & mask]++] = element;
                        }
                    }
                }
            }
        };

        PodBuffer<uint64_t> m_A, m_B;
        PodBuffer<uint32_t> m_IndexA, m_IndexB;
        PodBuffer<uint32_t> m_Histograms;   // per task (parallel) or per pass (serial), 2^c_MaxDigitBits each
        // Prepare's: the varying bits, the elements, and the words' form.
        BitRuns m_Runs{ 0 };
        uint32_t m_Count = 0;
        uint32_t m_IndexBits = 0;      // the bits of a position (count - 1)
        uint32_t m_PositionBits = 0;   // the position's bits in a word: m_IndexBits when packed, else 0
        bool m_Packed = true;          // the position rides in the word; else in the index arrays
        bool m_Identity = true;        // fewer than two elements or no varying bit: no passes
    };

    // ------------------------------------------------------------------------------------------ visibility

    // The instance query of the service (RenderService.Instances). Only mesh-instance rows carry MeshRef and
    // GpuInstanceSlot, so the Read terms select the rows of both kinds (plan step E7, decision 8): a node's mesh
    // instance and a runtime-only one (SceneComponents.h, RuntimeMeshInstance), which has no SceneLinks, SceneNode
    // and LeafObject: those columns are Optional, null in such a chunk. Its data terms, in ChunkView::columns order:
    using InstanceQuery = engine::Query<engine::Read<engine::GpuInstanceSlot>,
                                        engine::Read<engine::MeshRef>, engine::Read<engine::RenderFlags>,
                                        engine::Read<engine::WorldBounds>, engine::Read<engine::WorldTransform>,
                                        engine::Optional<engine::SceneLinks>, engine::Optional<engine::SceneNode>,
                                        engine::Optional<engine::LeafObject>, engine::Read<engine::LocalBounds>,
                                        engine::ReadChunk<engine::ChunkWorldBounds>>;
    enum InstanceColumn : uint32_t
    {
        ColumnSlot,
        ColumnMeshRef,
        ColumnRenderFlags,
        ColumnWorldBounds,
        ColumnWorld,
        ColumnLinks,
        ColumnNode,
        ColumnLeaf,
        ColumnLocalBounds,
        ColumnChunkBounds,
    };

    // What a frame's visibility needs from the service.
    struct VisibilityFrame
    {
        engine::ISystemContext* context = nullptr;
        const engine::SceneGraph* graph = nullptr;
        IRenderGraph* renderGraph = nullptr;            // the frame arena
        ExtractedView* views = nullptr;                 // the snapshot's; Run sets their instance and packet ranges
        uint32_t viewCount = 0;
        bool drawDoubleSidedMaterialsSeparately = true;
        bool compactPacketRecords = false;              // RenderService::GetCompactPacketRecords, read per extraction
    };

    class InstanceVisibility
    {
    public:
        InstanceVisibility() noexcept = default;
        InstanceVisibility(const InstanceVisibility&) = delete;
        InstanceVisibility& operator=(const InstanceVisibility&) = delete;

        // Fills the snapshot's instances and packets and the views' ranges. `bytes` grows by the arena bytes taken.
        // FE_OUT_OF_MEMORY when the arena or the scratch cannot grow (the snapshot's arrays are then empty).
        nvrhi::FRESULT Run(const InstanceQuery& query, const VisibilityFrame& frame, RenderSnapshot& snapshot,
                           uint64_t& bytes) noexcept
        {
            snapshot.instances = VisibleInstances{};
            snapshot.packets = DrawPackets{};
            ClearRanges(frame);
            ++m_Runs;
            if (frame.viewCount == 0)
                return nvrhi::FS_OK;

            engine::IQuery* iquery = query.Get();
            engine::IJobSystem* jobs = frame.context->GetJobSystem();
            const uint32_t chunkCount = iquery->GetChunkCount();
            const uint32_t viewCount = frame.viewCount;
            const uint32_t grain = std::max(1u, chunkCount / (ThreadsOf(jobs) * 4));
            const uint32_t rangeCount = chunkCount == 0 ? 0 : (chunkCount + grain - 1) / grain;
            const uint32_t segmentCount = viewCount * c_DrawPacketPassCount;

            // Scratch per range and view; vectors grow only (their buffers are kept).
            try
            {
                if (m_Scratch.size() < size_t(rangeCount) * viewCount)
                    m_Scratch.resize(size_t(rangeCount) * viewCount);
                if (m_Segments.size() < segmentCount)
                    m_Segments.resize(segmentCount);
                if (m_Sorters.size() < segmentCount)
                    m_Sorters.resize(segmentCount);
            }
            catch (...)
            {
                return nvrhi::FE_OUT_OF_MEMORY;
            }

            // 1. Visibility and packets per range. The ranges and their output are the same either way.
            const bool parallelRanges = jobs && rangeCount > 1 &&
                uint64_t(iquery->CountEntities()) * viewCount >= s_SerialRowTests.load(std::memory_order_relaxed);
            RangeJob job;
            job.self = this;
            job.query = iquery;
            job.request = engine::MakeChunkRequest(frame.context);
            job.meshes = &frame.graph->GetMeshes();
            job.views = frame.views;
            job.viewCount = viewCount;
            job.grain = grain;
            job.separateDoubleSided = frame.drawDoubleSidedMaterialsSeparately;
            const bool compact = frame.compactPacketRecords;
            const engine::PFN_JobFunction rangeRun = compact ? &RangeJob::Run<true> : &RangeJob::Run<false>;
            if (parallelRanges)
                jobs->ParallelFor(chunkCount, grain, rangeRun, &job);
            else if (chunkCount != 0)
                rangeRun(&job, 0, chunkCount, 0);
            if (job.failed.load(std::memory_order_relaxed))
                return nvrhi::FE_OUT_OF_MEMORY;

            // 2. Offsets: instances view-major, packets by view then pass; each range's part in range order.
            uint32_t instanceTotal = 0;
            uint32_t packetTotal = 0;
            for (uint32_t v = 0; v < viewCount; ++v)
            {
                ExtractedView& view = frame.views[v];
                view.instances.first = instanceTotal;
                for (uint32_t r = 0; r < rangeCount; ++r)
                {
                    RangeView& scratch = m_Scratch[size_t(r) * viewCount + v];
                    scratch.instanceOffset = instanceTotal;
                    instanceTotal += scratch.instanceCount;
                }
                view.instances.count = instanceTotal - view.instances.first;
                for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
                {
                    Segment& segment = m_Segments[v * c_DrawPacketPassCount + p];
                    segment.first = packetTotal;
                    segment.keyOr = 0;
                    segment.keyAnd = ~uint64_t(0);
                    for (uint32_t r = 0; r < rangeCount; ++r)
                    {
                        RangeView& scratch = m_Scratch[size_t(r) * viewCount + v];
                        scratch.packetOffset[p] = packetTotal;
                        packetTotal += scratch.packetCount[p];
                        if (scratch.packetCount[p] != 0)
                        {
                            segment.keyOr |= scratch.keyOr[p];
                            segment.keyAnd &= scratch.keyAnd[p];
                        }
                    }
                    segment.count = packetTotal - segment.first;
                    view.packets[p] = SnapshotRange{ segment.first, segment.count };
                }
            }

            // The arena's arrays, serially, at their high-water sizes (counts below them are exact).
            const uint32_t instanceCapacity = HighWater(m_InstanceCapacity, instanceTotal);
            const uint32_t packetCapacity = HighWater(m_PacketCapacity, packetTotal);
            Arena arena{ frame.renderGraph, bytes };
            Output out;
            out.entities = arena.Allocate<engine::Entity>(instanceCapacity);
            out.instanceSlots = arena.Allocate<uint32_t>(instanceCapacity);
            out.meshes = arena.Allocate<engine::MeshId>(instanceCapacity);
            out.firstGeometryInstanceSlots = arena.Allocate<uint32_t>(instanceCapacity);
            out.geometryCounts = arena.Allocate<uint32_t>(instanceCapacity);
            out.leaves = arena.Allocate<const engine::MeshInstance*>(instanceCapacity);
            out.meshInfos = arena.Allocate<const engine::MeshInfo*>(instanceCapacity);
            out.worldBounds = arena.Allocate<dm::box3>(instanceCapacity);
            out.keys = arena.Allocate<uint64_t>(packetCapacity);
            out.visibleInstances = arena.Allocate<uint32_t>(packetCapacity);
            out.geometryIndices = arena.Allocate<uint32_t>(packetCapacity);
            out.geometrySlots = arena.Allocate<uint32_t>(packetCapacity);
            out.materialSlots = arena.Allocate<uint32_t>(packetCapacity);
            out.cullModes = arena.Allocate<uint8_t>(packetCapacity);
            out.flags = arena.Allocate<uint8_t>(packetCapacity);
            if (arena.failed ||
                !(compact ? m_CompactRecords.Reserve(packetTotal) && m_Order.Reserve(packetTotal) &&
                                PrepareSorts(segmentCount)
                          : m_SortKeys.Reserve(packetTotal) && m_SortRecords.Reserve(packetTotal) &&
                                m_Order.Reserve(packetTotal)))
            {
                ClearRanges(frame);
                return nvrhi::FE_OUT_OF_MEMORY;
            }

            // 3. The instances into the arena, the packets into the sort's input.
            const uint32_t serialRecords =
                (parallelRanges ? s_SerialRecords : s_SerialRecordsAlone).load(std::memory_order_relaxed);
            const bool parallelRecords = jobs && instanceTotal + packetTotal >= serialRecords;
            CopyJob copy;
            copy.self = this;
            copy.out = &out;
            copy.viewCount = viewCount;
            const engine::PFN_JobFunction copyRun = compact ? &CopyJob::Run<true> : &CopyJob::Run<false>;
            if (rangeCount > 1 && parallelRecords)
                jobs->ParallelFor(rangeCount, 1, copyRun, &copy);
            else if (rangeCount != 0)
                copyRun(&copy, 0, rangeCount, 0);

            // 4. Each view's packets of each pass sorted by key, stably: the segments side by side, a large one on the
            // job system inside too.
            SortJob sort;
            sort.self = this;
            sort.jobs = jobs;
            const engine::PFN_JobFunction sortRun = compact ? &SortJob::Run<true> : &SortJob::Run<false>;
            if (segmentCount > 1 && parallelRecords)
                jobs->ParallelFor(segmentCount, 1, sortRun, &sort);
            else
                sortRun(&sort, 0, segmentCount, 0);
            if (sort.failed.load(std::memory_order_relaxed))
            {
                ClearRanges(frame);
                return nvrhi::FE_OUT_OF_MEMORY;
            }

            // 5. The packets in key order.
            GatherJob gather;
            gather.self = this;
            gather.out = &out;
            gather.count = packetTotal;
            const uint32_t gatherGrain = 8192;
            const engine::PFN_JobFunction gatherRun = compact ? &GatherJob::Run<true> : &GatherJob::Run<false>;
            if (packetTotal > gatherGrain && jobs)
                jobs->ParallelFor(packetTotal, gatherGrain, gatherRun, &gather);
            else if (packetTotal != 0)
                gatherRun(&gather, 0, packetTotal, 0);

            VisibleInstances& instances = snapshot.instances;
            instances.entities = out.entities;
            instances.instanceSlots = out.instanceSlots;
            instances.meshes = out.meshes;
            instances.firstGeometryInstanceSlots = out.firstGeometryInstanceSlots;
            instances.geometryCounts = out.geometryCounts;
            instances.leaves = out.leaves;
            instances.meshInfos = out.meshInfos;
            instances.worldBounds = out.worldBounds;
            instances.count = instanceTotal;
            DrawPackets& packets = snapshot.packets;
            packets.keys = out.keys;
            packets.visibleInstances = out.visibleInstances;
            packets.geometryIndices = out.geometryIndices;
            packets.geometrySlots = out.geometrySlots;
            packets.materialSlots = out.materialSlots;
            packets.cullModes = out.cullModes;
            packets.flags = out.flags;
            packets.count = packetTotal;

#ifndef NDEBUG
            CheckLeaves(query, frame);
#endif
            return nvrhi::FS_OK;
        }

        // Less work than this runs on the calling thread: in the frames of a small scene the workers sleep, and waking
        // them can take longer than the work. Row tests (the query's rows times the views) decide for the ranges, records
        // (instances and packets) for the copy and the sort, with a limit of their own after serial ranges. Measured with
        // sleeping workers and with all of them awake and looking for work, 1, 2, 8 and 15 of them, over scenes with
        // every instance visible (1.75 and 6 packets each), five views, and mostly culled rows (aggregate plan, "E5.8
        // fix", the sweep; us per extraction):
        // - 512 to 640 row tests: parallel takes from 23 less to 3 more than serial (asleep), 54 less to 1 more (awake).
        //   Below 512: 4 less to 5 more, and 28 less to 14 more. Row tests are a rough measure of the work: with six
        //   packets per instance the two are even (asleep, 8 and 15 workers) from 192 to 384 row tests, and where every
        //   chunk is culled, parallel takes up to 10 more (asleep) at any size.
        // - The serial copy and sort below 1,536 records, after parallel ranges: from 17 less to 2 more than in parallel
        //   (asleep; 5 less on average), 4 less to 6 more (awake).
        // - The parallel copy and sort after serial ranges: from 2 less to 51 more than serial (asleep; more in 123 of
        //   128 points), 19 less to 50 more (awake; more in 96 of 128), for 69 to 11,264 records.
        // Tests set the limits, to run every combination. They are per module: an executable or DLL that links this code
        // statically has its own, so a test reaches only the extractions of its own module.
        static void SetSerialLimits(uint32_t rowTests, uint32_t records, uint32_t recordsAlone) noexcept
        {
            s_SerialRowTests.store(rowTests, std::memory_order_relaxed);
            s_SerialRecords.store(records, std::memory_order_relaxed);
            s_SerialRecordsAlone.store(recordsAlone, std::memory_order_relaxed);
        }
        static constexpr uint32_t c_SerialRowTests = 512;
        static constexpr uint32_t c_SerialRecords = 1536;                                  // after parallel ranges
        static constexpr uint32_t c_SerialRecordsAlone = RadixSorter::c_ParallelThreshold;   // after serial ranges

        // Extractions so far (the Debug sampled check runs at every 64th).
        [[nodiscard]] uint64_t GetRunCount() const noexcept { return m_Runs; }

        // Debug builds check the leaves of c_LeafCheckChunks chunks of the query at every c_LeafCheckInterval-th
        // extraction, the chunks after the last ones checked, so that every leaf is seen in turn; at most
        // c_LeafCheckWarnings warnings per check.
        static constexpr uint64_t c_LeafCheckInterval = 64;
        static constexpr uint32_t c_LeafCheckChunks = 4;
        static constexpr uint32_t c_LeafCheckWarnings = 4;
        // A slot a packet has no value for (a geometry beyond its mesh row, a geometry without a material).
        static constexpr uint32_t c_NoSlot = ~0u;
        // Compact packet records hold the geometry index in 26 bits: a mesh's geometries from this one on get no packet
        // with them (a warning, once per process).
        static constexpr uint32_t c_CompactGeometryLimit = 1u << 26;

    private:
        // One visible instance as a range lists it (the snapshot's VisibleInstances, one record).
        struct InstanceRecord
        {
            dm::box3 bounds;
            engine::Entity entity;
            const engine::MeshInstance* leaf;   // null for a runtime-only row
            const engine::MeshInfo* mesh;
            uint32_t instanceSlot;
            uint32_t firstGeometryInstance;
            uint32_t geometryCount;
            engine::MeshId meshId;
        };

        // One packet as a range lists it; `instance` is the range's instance index, made global before the sort.
        struct PacketRecord
        {
            uint64_t key;
            uint32_t instance;
            uint32_t geometryIndex;
            uint32_t geometrySlot;
            uint32_t materialSlot;
            uint8_t cullMode;
            uint8_t flags;
        };

        // The same packet in a compact record (RenderService::SetCompactPacketRecords): 24 bytes instead of 32.
        struct CompactPacketRecord
        {
            uint64_t key;
            uint32_t instance;
            uint32_t geometrySlot;
            uint32_t materialSlot;
            uint32_t geometryCullFlags;   // [25:0] geometry index, [27:26] cull mode, [31:28] flags
        };
        static_assert(sizeof(CompactPacketRecord) == 24, "CompactPacketRecord layout");
        static constexpr uint32_t c_CompactCullShift = 26;
        static constexpr uint32_t c_CompactFlagsShift = 28;
        static_assert(uint32_t(nvrhi::RasterCullMode::None) < 4, "the cull mode's two bits");
        static_assert((c_DrawPacketFlagOpaquePass | c_DrawPacketFlagTransparentPass | c_DrawPacketFlagDoubleSided |
                       c_DrawPacketFlagGeometryCulled) < 16, "the flags' four bits");

        static CompactPacketRecord MakeCompactRecord(const PacketRecord& packet) noexcept
        {
            return CompactPacketRecord{ packet.key, packet.instance, packet.geometrySlot, packet.materialSlot,
                                        packet.geometryIndex | uint32_t(packet.cullMode) << c_CompactCullShift |
                                            uint32_t(packet.flags) << c_CompactFlagsShift };
        }

        // A range's lists for one view. A cache line of its own: the range's task updates the counts and masks for
        // every packet, and neighbouring ranges run on other threads.
        struct alignas(64) RangeView
        {
            PodBuffer<InstanceRecord> instances;
            PodBuffer<PacketRecord> packets[c_DrawPacketPassCount];
            PodBuffer<CompactPacketRecord> compactPackets[c_DrawPacketPassCount];   // with compact records
            uint32_t instanceCount = 0;
            uint32_t packetCount[c_DrawPacketPassCount] = {};
            uint64_t keyOr[c_DrawPacketPassCount] = {};
            uint64_t keyAnd[c_DrawPacketPassCount] = {};
            uint32_t instanceOffset = 0;                       // in the snapshot's instances
            uint32_t packetOffset[c_DrawPacketPassCount] = {};  // in the snapshot's packets

            void Reset() noexcept
            {
                instanceCount = 0;
                for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
                {
                    packetCount[p] = 0;
                    keyOr[p] = 0;
                    keyAnd[p] = ~uint64_t(0);
                }
            }
        };

        // One view's packets of one pass.
        struct Segment
        {
            uint32_t first = 0;
            uint32_t count = 0;
            uint64_t keyOr = 0;
            uint64_t keyAnd = 0;
        };

        struct Output
        {
            engine::Entity* entities = nullptr;
            uint32_t* instanceSlots = nullptr;
            engine::MeshId* meshes = nullptr;
            uint32_t* firstGeometryInstanceSlots = nullptr;
            uint32_t* geometryCounts = nullptr;
            const engine::MeshInstance** leaves = nullptr;
            const engine::MeshInfo** meshInfos = nullptr;
            dm::box3* worldBounds = nullptr;
            uint64_t* keys = nullptr;
            uint32_t* visibleInstances = nullptr;
            uint32_t* geometryIndices = nullptr;
            uint32_t* geometrySlots = nullptr;
            uint32_t* materialSlots = nullptr;
            uint8_t* cullModes = nullptr;
            uint8_t* flags = nullptr;
        };

        struct Arena
        {
            IRenderGraph* graph;
            uint64_t& bytes;
            bool failed = false;

            template <typename T>
            T* Allocate(uint32_t count) noexcept
            {
                static_assert(std::is_trivially_destructible_v<T>, "the frame arena runs no destructors");
                if (count == 0 || failed)
                    return nullptr;
                void* memory = graph->AllocateFrameData(sizeof(T) * size_t(count), alignof(T));
                if (!memory)
                {
                    failed = true;
                    return nullptr;
                }
                bytes += uint64_t(sizeof(T)) * count;
                return static_cast<T*>(memory);
            }
        };

        // The arena size of an array: the high-water mark with a quarter of headroom, so that the arena, which reuses
        // its blocks only while a frame asks for no more than before, takes no new block when visibility wanders. It
        // shrinks when the need falls below a quarter of it (a smaller request never takes a new block).
        static uint32_t HighWater(uint32_t& mark, uint32_t need) noexcept
        {
            if (need > mark || need < mark / 4)
            {
                const uint64_t grown = uint64_t(need) + need / 4 + 256;
                mark = uint32_t(std::min<uint64_t>(grown, UINT32_MAX));
            }
            return need == 0 ? 0 : mark;
        }

        // ---------------------------------------------------------------- step 1: one task per range of chunks

        struct RangeJob
        {
            InstanceVisibility* self = nullptr;
            engine::IQuery* query = nullptr;
            engine::ChunkRequest request{};
            const engine::MeshTable* meshes = nullptr;
            const ExtractedView* views = nullptr;
            uint32_t viewCount = 0;
            uint32_t grain = 1;
            bool separateDoubleSided = true;
            std::atomic<bool> failed{ false };

            template <bool Compact>
            static void Run(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                RangeJob& job = *static_cast<RangeJob*>(context);
                for (uint32_t rangeBegin = begin; rangeBegin < end; rangeBegin += job.grain)
                {
                    const uint32_t rangeEnd = std::min(end, rangeBegin + job.grain);
                    RangeView* scratch = &job.self->m_Scratch[size_t(rangeBegin / job.grain) * job.viewCount];
                    for (uint32_t v = 0; v < job.viewCount; ++v)
                        scratch[v].Reset();
                    engine::ChunkView chunks[16];
                    for (uint32_t first = rangeBegin; first < rangeEnd; first += 16)
                    {
                        const uint32_t capacity = std::min(16u, rangeEnd - first);
                        const uint32_t written = job.query->GetChunks(first, job.request, capacity, chunks);
                        for (uint32_t i = 0; i < written; ++i)
                            job.Chunk<Compact>(chunks[i], scratch);
                    }
                }
            }

            void Fail() noexcept { failed.store(true, std::memory_order_relaxed); }

            template <typename T>
            static const T* ColumnOf(const engine::ChunkView& chunk, InstanceColumn column) noexcept
            {
                return static_cast<const T*>(chunk.columns[column]);
            }

            template <bool Compact>
            void Chunk(const engine::ChunkView& chunk, RangeView* scratch) noexcept
            {
                using namespace engine;
                const auto* slots = ColumnOf<GpuInstanceSlot>(chunk, ColumnSlot);
                const auto* bounds = ColumnOf<WorldBounds>(chunk, ColumnWorldBounds);
                const auto* chunkBounds = ColumnOf<ChunkWorldBounds>(chunk, ColumnChunkBounds);
                for (uint32_t v = 0; v < viewCount; ++v)
                {
                    const dm::frustum& frustum = views[v].frustum;
                    if (!frustum.intersectsWith(chunkBounds->bounds))
                        continue;
                    RangeView& out = scratch[v];
                    if (!out.instances.Reserve(size_t(out.instanceCount) + chunk.count, out.instanceCount))
                    {
                        Fail();
                        return;
                    }
                    for (uint32_t row = 0; row < chunk.count; ++row)
                    {
                        if (slots[row].instance == c_NoInstanceSlot)
                            continue;
                        if (!frustum.intersectsWith(bounds[row].bounds))
                            continue;
                        Row<Compact>(chunk, row, views[v], out);
                    }
                }
            }

            template <bool Compact>
            bool Push(RangeView& out, uint32_t pass, const PacketRecord& packet) noexcept
            {
                uint32_t& count = out.packetCount[pass];
                if constexpr (Compact)
                {
                    PodBuffer<CompactPacketRecord>& buffer = out.compactPackets[pass];
                    if (!buffer.Reserve(size_t(count) + 1, count))
                    {
                        Fail();
                        return false;
                    }
                    buffer.Data()[count++] = MakeCompactRecord(packet);
                }
                else
                {
                    PodBuffer<PacketRecord>& buffer = out.packets[pass];
                    if (!buffer.Reserve(size_t(count) + 1, count))
                    {
                        Fail();
                        return false;
                    }
                    buffer.Data()[count++] = packet;
                }
                out.keyOr[pass] |= packet.key;
                out.keyAnd[pass] &= packet.key;
                return true;
            }

            // A visible row: its instance record, then its packets by the legacy rules.
            template <bool Compact>
            void Row(const engine::ChunkView& chunk, uint32_t row, const ExtractedView& view, RangeView& out) noexcept
            {
                using namespace engine;
                const GpuInstanceSlot& slot = ColumnOf<GpuInstanceSlot>(chunk, ColumnSlot)[row];
                const MeshRef& meshRef = ColumnOf<MeshRef>(chunk, ColumnMeshRef)[row];
                // A node's row has its leaf object, a MeshInstance (its LeafTypeDesc gives it MeshRef and the slot); a
                // runtime-only row has none: its record's leaf is null.
                const LeafObject* leaves = ColumnOf<LeafObject>(chunk, ColumnLeaf);
                const SceneGraphLeaf* leafObject = leaves ? leaves[row].object : nullptr;
                const auto* leaf = static_cast<const MeshInstance*>(leafObject);
                // The mesh from the graph's table (MeshRef names the leaf's mesh while it is attached), so that the
                // leaf object is not read.
                const MeshInfo* mesh = meshes->Get(meshRef.mesh);
                if (!mesh && leaf)
                    mesh = leaf->GetMesh();

                const uint32_t local = out.instanceCount++;
                InstanceRecord& record = out.instances.Data()[local];
                record.bounds = ColumnOf<WorldBounds>(chunk, ColumnWorldBounds)[row].bounds;
                record.entity = chunk.entities[row];
                record.leaf = leaf;
                record.mesh = mesh;
                record.instanceSlot = slot.instance;
                record.firstGeometryInstance = slot.firstGeometryInstance;
                record.geometryCount = slot.geometryCount;
                record.meshId = meshRef.mesh;
                if (!mesh)
                    return;

                // This graph's slots: the mesh row's geometry range and materials.
                const MeshTable::MeshRow* meshRow = meshRef.mesh != MeshId::Invalid
                    ? &meshes->GetMeshRow(MeshTable::SlotOf(meshRef.mesh)) : nullptr;

                size_t geometryCount = mesh->geometries.size();
                const bool multiple = geometryCount > 1;
                if constexpr (Compact)
                {
                    if (geometryCount > c_CompactGeometryLimit)
                    {
                        geometryCount = c_CompactGeometryLimit;
                        WarnGeometryLimit();
                    }
                }
                const bool skinned = mesh->skinPrototype != nullptr;
                // The node's float world transform (SceneGraphNode::GetLocalToWorldTransformFloat), when a geometry
                // test needs it; the transparent box of the instance, when a distance needs it.
                dm::affine3 world;
                bool haveWorld = false;
                dm::box3 instanceBox;
                bool haveInstanceBox = false;
                auto worldOf = [&]() -> const dm::affine3&
                {
                    if (!haveWorld)
                    {
                        world = dm::affine3(ColumnOf<WorldTransform>(chunk, ColumnWorld)[row].transform);
                        haveWorld = true;
                    }
                    return world;
                };
                auto instanceBoxOf = [&]() -> const dm::box3&
                {
                    if (!haveInstanceBox)
                    {
                        // The legacy walker's GetGlobalBoundingBox: the subgraph's box. A node without children has
                        // its own box, the row's WorldBounds, and so has a runtime-only row (no node, no links, no
                        // subgraph).
                        // After Refresh it is the value Refresh stored in the node; a structural change not yet
                        // refreshed computes it under the graph's lock, which can throw: then the own box.
                        const SceneNode* nodes = ColumnOf<SceneNode>(chunk, ColumnNode);
                        const SceneLinks* links = ColumnOf<SceneLinks>(chunk, ColumnLinks);
                        const SceneGraphNode* node = nodes ? nodes[row].object : nullptr;
                        instanceBox = record.bounds;
                        if (node && links && links[row].firstChild != Entity::Null)
                        {
                            try
                            {
                                instanceBox = node->GetGlobalBoundingBox();
                            }
                            catch (...)
                            {
                            }
                        }
                        haveInstanceBox = true;
                    }
                    return instanceBox;
                };

                for (size_t g = 0; g < geometryCount; ++g)
                {
                    const MeshGeometry* geometry = mesh->geometries[g].Get();
                    const Material* material = geometry ? geometry->material.Get() : nullptr;
                    if (!material)
                        continue;
                    const MaterialDomain domain = material->domain;
                    PacketRecord packet;
                    packet.instance = local;
                    packet.geometryIndex = uint32_t(g);
                    // a geometry beyond the row's (the mesh grew in place, not reconciled yet) has no slot
                    const bool inRow = meshRow && g < meshRow->geometryCount && g < meshRow->materials.size();
                    packet.geometrySlot = inRow ? meshRow->firstGeometry + uint32_t(g) : c_NoSlot;
                    packet.materialSlot = inRow && meshRow->materials[g] != MaterialId::Invalid
                        ? MaterialTable::SlotOf(meshRow->materials[g]) : c_NoSlot;
                    if (domain == MaterialDomain::Opaque || domain == MaterialDomain::AlphaTested)
                    {
                        // InstancedOpaqueDrawStrategy
                        packet.flags = c_DrawPacketFlagOpaquePass;
                        if (multiple && !skinned)
                        {
                            const dm::box3 geometryBox = geometry->objectSpaceBounds * worldOf();
                            if (!view.frustum.intersectsWith(geometryBox))
                                continue;
                            packet.flags |= c_DrawPacketFlagGeometryCulled;
                        }
                        const nvrhi::RasterCullMode cull =
                            material->doubleSided ? nvrhi::RasterCullMode::None : nvrhi::RasterCullMode::Back;
                        if (material->doubleSided)
                            packet.flags |= c_DrawPacketFlagDoubleSided;
                        packet.cullMode = uint8_t(cull);
                        packet.key = MakeOpaqueDrawKey(uint32_t(domain), uint32_t(cull), packet.materialSlot,
                                                       packet.geometrySlot, slot.instance);
                        if (!Push<Compact>(out, uint32_t(DrawPacketPass::Opaque), packet))
                            return;
                        continue;
                    }

                    // TransparentDrawStrategy
                    packet.flags = c_DrawPacketFlagTransparentPass;
                    dm::box3 box;
                    if (multiple && skinned)
                    {
                        box = geometry->objectSpaceBounds * worldOf();
                        if (!view.frustum.intersectsWith(box))
                            continue;
                        packet.flags |= c_DrawPacketFlagGeometryCulled;
                    }
                    else
                        box = instanceBoxOf();
                    const float distance = dm::length(box.center() - view.viewOrigin);
                    auto push = [&](nvrhi::RasterCullMode cull) noexcept
                    {
                        packet.cullMode = uint8_t(cull);
                        packet.key = MakeTransparentDrawKey(distance, slot.instance, uint32_t(cull));
                        return Push<Compact>(out, uint32_t(DrawPacketPass::Transparent), packet);
                    };
                    if (material->doubleSided)
                    {
                        packet.flags |= c_DrawPacketFlagDoubleSided;
                        if (separateDoubleSided)
                        {
                            if (!push(nvrhi::RasterCullMode::Front) || !push(nvrhi::RasterCullMode::Back))
                                return;
                        }
                        else if (!push(nvrhi::RasterCullMode::None))
                            return;
                    }
                    else if (!push(nvrhi::RasterCullMode::Back))
                        return;
                }
            }
        };

        // ---------------------------------------------------------------- step 3: one task per range

        struct CopyJob
        {
            InstanceVisibility* self = nullptr;
            const Output* out = nullptr;
            uint32_t viewCount = 0;

            template <bool Compact>
            static void Run(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                CopyJob& job = *static_cast<CopyJob*>(context);
                InstanceVisibility& self = *job.self;
                const Output& out = *job.out;
                for (uint32_t r = begin; r < end; ++r)
                {
                    for (uint32_t v = 0; v < job.viewCount; ++v)
                    {
                        RangeView& scratch = self.m_Scratch[size_t(r) * job.viewCount + v];
                        const uint32_t base = scratch.instanceOffset;
                        const InstanceRecord* instances = scratch.instances.Data();
                        for (uint32_t i = 0; i < scratch.instanceCount; ++i)
                        {
                            const InstanceRecord& record = instances[i];
                            const uint32_t at = base + i;
                            out.entities[at] = record.entity;
                            out.instanceSlots[at] = record.instanceSlot;
                            out.meshes[at] = record.meshId;
                            out.firstGeometryInstanceSlots[at] = record.firstGeometryInstance;
                            out.geometryCounts[at] = record.geometryCount;
                            out.leaves[at] = record.leaf;
                            out.meshInfos[at] = record.mesh;
                            out.worldBounds[at] = record.bounds;
                        }
                        if constexpr (Compact)
                            CopyCompact(self, scratch, v);
                        else
                        {
                            uint64_t* keys = self.m_SortKeys.Data();
                            const PacketRecord** records = self.m_SortRecords.Data();
                            for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
                            {
                                PacketRecord* packets = scratch.packets[p].Data();
                                const uint32_t first = scratch.packetOffset[p];
                                for (uint32_t i = 0; i < scratch.packetCount[p]; ++i)
                                {
                                    packets[i].instance += base;   // the snapshot's visible-instance index
                                    keys[first + i] = packets[i].key;
                                    records[first + i] = &packets[i];
                                }
                            }
                        }
                    }
                }
            }

            // Compact records: each into the one array, its instance index made global, and its word into its
            // segment's sort (the key's varying bits, the position in the segment).
            static void CopyCompact(InstanceVisibility& self, const RangeView& scratch, uint32_t view) noexcept
            {
                const uint32_t base = scratch.instanceOffset;
                for (uint32_t p = 0; p < c_DrawPacketPassCount; ++p)
                {
                    const uint32_t count = scratch.packetCount[p];
                    if (count == 0)
                        continue;
                    const CompactPacketRecord* packets = scratch.compactPackets[p].Data();
                    const uint32_t first = scratch.packetOffset[p];
                    CompactPacketRecord* records = self.m_CompactRecords.Data() + first;
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        records[i] = packets[i];
                        records[i].instance += base;   // the snapshot's visible-instance index
                    }
                    const uint32_t segment = view * c_DrawPacketPassCount + p;
                    const RadixSorter& sorter = self.m_Sorters[segment];
                    uint64_t* const words = sorter.Words();
                    if (!words)
                        continue;   // the identity order
                    const uint32_t position = first - self.m_Segments[segment].first;
                    const BitRuns runs = sorter.Runs();
                    const uint32_t shift = sorter.PositionBits();
                    if (uint32_t* const indices = sorter.Indices())
                    {
                        for (uint32_t i = 0; i < count; ++i)
                        {
                            words[position + i] = runs.Compact(packets[i].key);
                            indices[position + i] = position + i;
                        }
                    }
                    else
                    {
                        for (uint32_t i = 0; i < count; ++i)
                            words[position + i] = (runs.Compact(packets[i].key) << shift) | (position + i);
                    }
                }
            }
        };

        // ---------------------------------------------------------------- step 4: one task per segment

        struct SortJob
        {
            InstanceVisibility* self = nullptr;
            engine::IJobSystem* jobs = nullptr;
            std::atomic<bool> failed{ false };

            template <bool Compact>
            static void Run(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                SortJob& job = *static_cast<SortJob*>(context);
                InstanceVisibility& self = *job.self;
                for (uint32_t s = begin; s < end; ++s)
                {
                    const Segment& segment = self.m_Segments[s];
                    bool sorted;
                    if constexpr (Compact)
                        sorted = self.m_Sorters[s].Finish(segment.first, self.m_Order.Data() + segment.first, job.jobs);
                    else
                        sorted = self.m_Sorters[s].Sort(self.m_SortKeys.Data() + segment.first, segment.count,
                                                        segment.keyOr ^ segment.keyAnd, segment.first,
                                                        self.m_Order.Data() + segment.first, job.jobs);
                    if (!sorted)
                        job.failed.store(true, std::memory_order_relaxed);
                }
            }
        };

        // Compact records: each segment's sort prepared (its buffers reserved, its word form known) before the copy
        // writes the words.
        bool PrepareSorts(uint32_t segmentCount) noexcept
        {
            for (uint32_t s = 0; s < segmentCount; ++s)
            {
                const Segment& segment = m_Segments[s];
                if (!m_Sorters[s].Prepare(segment.count, segment.keyOr ^ segment.keyAnd))
                    return false;
            }
            return true;
        }

        static void WarnGeometryLimit() noexcept
        {
            static std::atomic<bool> s_Warned{ false };
            if (!s_Warned.exchange(true, std::memory_order_relaxed))
                log::warning("RenderService: a mesh has more than %u geometries; with compact packet records the ones "
                             "beyond get no packets", c_CompactGeometryLimit);
        }

        static void ClearRanges(const VisibilityFrame& frame) noexcept
        {
            for (uint32_t v = 0; v < frame.viewCount; ++v)
            {
                frame.views[v].instances = SnapshotRange{};
                for (SnapshotRange& range : frame.views[v].packets)
                    range = SnapshotRange{};
            }
        }

        // ---------------------------------------------------------------- step 5: the packets in key order

        struct GatherJob
        {
            InstanceVisibility* self = nullptr;
            const Output* out = nullptr;
            uint32_t count = 0;

            template <bool Compact>
            static void Run(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                GatherJob& job = *static_cast<GatherJob*>(context);
                const Output& out = *job.out;
                const uint32_t* order = job.self->m_Order.Data();
                if constexpr (Compact)
                {
                    const CompactPacketRecord* records = job.self->m_CompactRecords.Data();
                    for (uint32_t i = begin; i < end; ++i)
                    {
                        const CompactPacketRecord& packet = records[order[i]];
                        out.keys[i] = packet.key;
                        out.visibleInstances[i] = packet.instance;
                        out.geometryIndices[i] = packet.geometryCullFlags & (c_CompactGeometryLimit - 1);
                        out.geometrySlots[i] = packet.geometrySlot;
                        out.materialSlots[i] = packet.materialSlot;
                        out.cullModes[i] = uint8_t((packet.geometryCullFlags >> c_CompactCullShift) & 3u);
                        out.flags[i] = uint8_t(packet.geometryCullFlags >> c_CompactFlagsShift);
                    }
                }
                else
                {
                    const PacketRecord* const* records = job.self->m_SortRecords.Data();
                    for (uint32_t i = begin; i < end; ++i)
                    {
                        const PacketRecord& packet = *records[order[i]];
                        out.keys[i] = packet.key;
                        out.visibleInstances[i] = packet.instance;
                        out.geometryIndices[i] = packet.geometryIndex;
                        out.geometrySlots[i] = packet.geometrySlot;
                        out.materialSlots[i] = packet.materialSlot;
                        out.cullModes[i] = packet.cullMode;
                        out.flags[i] = packet.flags;
                    }
                }
            }
        };

#ifndef NDEBUG
        // Debug: at every c_LeafCheckInterval-th extraction, every row of c_LeafCheckChunks chunks of the query (the
        // ones after the last checked, whether their instances are visible or not): LocalBounds against the leaf's
        // GetLocalBoundingBox and RenderFlags::contentFlags against its GetContentFlags. A difference means the leaf
        // changed without SceneGraphNode::InvalidateContent (E3: those columns follow the leaf only at attach, SetLeaf
        // and InvalidateContent; a leaf class with LeafTypeDesc::dynamicBounds keeps its box current by itself), so
        // culling used a stale box or pass selection stale flags. The warning names the node (by name: GetPath
        // allocates).
        void CheckLeaves(const InstanceQuery& query, const VisibilityFrame& frame) noexcept
        {
            if (m_Runs % c_LeafCheckInterval != 0)
                return;
            engine::IQuery* iquery = query.Get();
            const uint32_t chunkCount = iquery->GetChunkCount();
            if (chunkCount == 0)
                return;
            const engine::ChunkRequest request = engine::MakeChunkRequest(frame.context);
            uint32_t warnings = 0;
            for (uint32_t k = 0; k < std::min(c_LeafCheckChunks, chunkCount); ++k)
            {
                const uint32_t index = (m_LeafCheckCursor + k) % chunkCount;
                engine::ChunkView chunk;
                if (iquery->GetChunks(index, request, 1, &chunk) != 1)
                    continue;
                // Runtime-only rows have no leaf: the store writes their columns from the mesh when they join.
                const auto* leaves = static_cast<const engine::LeafObject*>(chunk.columns[ColumnLeaf]);
                if (!leaves)
                    continue;
                const auto* locals = static_cast<const engine::LocalBounds*>(chunk.columns[ColumnLocalBounds]);
                const auto* flags = static_cast<const engine::RenderFlags*>(chunk.columns[ColumnRenderFlags]);
                const auto* nodes = static_cast<const engine::SceneNode*>(chunk.columns[ColumnNode]);
                for (uint32_t row = 0; row < chunk.count && warnings < c_LeafCheckWarnings; ++row)
                {
                    engine::SceneGraphLeaf* leaf = leaves[row].object;
                    if (!leaf)
                        continue;
                    const dm::box3 box = leaf->GetLocalBoundingBox();
                    const bool staleBox = std::memcmp(&box, &locals[row].bounds, sizeof(box)) != 0;
                    const bool staleFlags = uint32_t(leaf->GetContentFlags()) != flags[row].contentFlags;
                    if (!staleBox && !staleFlags)
                        continue;
                    const engine::SceneGraphNode* node = nodes ? nodes[row].object : nullptr;
                    log::warning("RenderService: the %s of the leaf of node '%s' changed without "
                                 "SceneGraphNode::InvalidateContent; it is culled and sorted with the old one",
                                 staleBox ? "bounding box" : "content (material domains)",
                                 node ? node->GetName().c_str() : "");
                    ++warnings;
                }
            }
            m_LeafCheckCursor = (m_LeafCheckCursor + c_LeafCheckChunks) % chunkCount;
        }
#endif

        std::vector<RangeView> m_Scratch;            // range-major, viewCount per range
        std::vector<Segment> m_Segments;             // view-major, a segment per pass
        PodBuffer<uint64_t> m_SortKeys;              // the packets' keys, segment by segment, unsorted
        PodBuffer<const PacketRecord*> m_SortRecords;
        PodBuffer<CompactPacketRecord> m_CompactRecords;   // with compact records: every packet, unsorted
        PodBuffer<uint32_t> m_Order;                 // the sorted positions in m_SortKeys
        std::vector<RadixSorter> m_Sorters;          // one per segment, so that segments sort side by side
        uint32_t m_InstanceCapacity = 0;
        uint32_t m_PacketCapacity = 0;
        uint64_t m_Runs = 0;
        uint32_t m_LeafCheckCursor = 0;   // Debug: the next chunk CheckLeaves looks at

        inline static std::atomic<uint32_t> s_SerialRowTests{ c_SerialRowTests };
        inline static std::atomic<uint32_t> s_SerialRecords{ c_SerialRecords };
        inline static std::atomic<uint32_t> s_SerialRecordsAlone{ c_SerialRecordsAlone };
    };
}
