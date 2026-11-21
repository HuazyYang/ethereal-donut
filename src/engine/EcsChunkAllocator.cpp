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

// The memory of the ECS storage: counted heap blocks from nvrhi's default allocator, the chunk page pool, and the hash
// index (aggregate docs/ecs-design.md 4.3, 10.3).

#include "EcsWorldImpl.h"

#include <nvrhi/core/memory.h>

#include <atomic>
#include <mutex>

namespace donut::engine::ecs
{
    namespace
    {
        std::atomic<uint64_t> g_HeapAllocations{ 0 };
        std::atomic<int64_t> g_HeapLiveBlocks{ 0 };
        std::atomic<int64_t> g_FailureCountdown{ -1 };   // SetHeapFailureCountdown; negative when disarmed

        // The shared slabs, by size (PagePool::ShareSlabs). Never destroyed: worlds may be released during static
        // destruction. Its blocks stay counted as live heap blocks while it holds them.
        struct SharedSlab
        {
            void* memory;
            size_t bytes;
        };
        struct SlabCache
        {
            std::mutex mutex;
            SharedSlab* slabs = nullptr;   // a plain HeapAllocate block, so that the cache counts like the storage
            uint32_t count = 0;
            uint32_t capacity = 0;
            uint64_t bytes = 0;
        };
        SlabCache& GetSlabCache() noexcept
        {
            static SlabCache* const s_Cache = new SlabCache();
            return *s_Cache;
        }
    }

    // ---------------------------------------------------------------------------------------------------- heap

    void* HeapAllocate(size_t bytes, size_t alignment) noexcept
    {
        const int64_t countdown = g_FailureCountdown.load(std::memory_order_relaxed);
        if (countdown >= 0)
        {
            g_FailureCountdown.store(countdown - 1, std::memory_order_relaxed);
            if (countdown == 0)
                return nullptr;   // the injected failure; the countdown is now disarmed
        }
        if (bytes == 0)
            bytes = 1;
        if (alignment < alignof(std::max_align_t))
            alignment = alignof(std::max_align_t);
        void* memory = nvrhi::GetDefaultMemAllocator()->AllocateAligned(bytes, alignment);
        if (memory)
        {
            g_HeapAllocations.fetch_add(1, std::memory_order_relaxed);
            g_HeapLiveBlocks.fetch_add(1, std::memory_order_relaxed);
        }
        return memory;
    }

    void HeapFree(void* memory) noexcept
    {
        if (!memory)
            return;
        nvrhi::GetDefaultMemAllocator()->FreeAligned(memory);
        g_HeapLiveBlocks.fetch_sub(1, std::memory_order_relaxed);
    }

    uint64_t GetHeapAllocationCount() noexcept
    {
        return g_HeapAllocations.load(std::memory_order_relaxed);
    }

    int64_t GetHeapLiveBlockCount() noexcept
    {
        return g_HeapLiveBlocks.load(std::memory_order_relaxed);
    }

    void SetHeapFailureCountdown(int64_t allocations) noexcept
    {
        g_FailureCountdown.store(allocations < 0 ? -1 : allocations, std::memory_order_relaxed);
    }

    // ----------------------------------------------------------------------------------------------- page pool

    void PagePool::Initialize(uint32_t pageSize) noexcept
    {
        m_PageSize = pageSize;
        m_PagesPerSlab = pageSize < c_SlabBytes ? c_SlabBytes / pageSize : 1;
    }

    bool PagePool::AddSlab() noexcept
    {
        if (m_SlabCount == m_SlabCapacity)
        {
            const uint32_t capacity = m_SlabCapacity != 0 ? m_SlabCapacity * 2 : 8;
            void** slabs = static_cast<void**>(HeapAllocate(size_t(capacity) * sizeof(void*), alignof(void*)));
            if (!slabs)
                return false;
            if (m_SlabCount != 0)
                std::memcpy(slabs, m_Slabs, size_t(m_SlabCount) * sizeof(void*));
            HeapFree(m_Slabs);
            m_Slabs = slabs;
            m_SlabCapacity = capacity;
        }
        const size_t slabBytes = size_t(m_PageSize) * m_PagesPerSlab;
        uint8_t* slab = nullptr;
        if (m_Shared)
        {
            // the most recently given back slab of this size: the one most likely to be in the caches
            SlabCache& cache = GetSlabCache();
            std::lock_guard<std::mutex> lock(cache.mutex);
            for (uint32_t i = cache.count; i-- > 0;)
            {
                if (cache.slabs[i].bytes == slabBytes)
                {
                    slab = static_cast<uint8_t*>(cache.slabs[i].memory);
                    cache.slabs[i] = cache.slabs[--cache.count];
                    cache.bytes -= slabBytes;
                    break;
                }
            }
        }
        if (!slab)
            slab = static_cast<uint8_t*>(HeapAllocate(slabBytes, c_ColumnAlignment));
        if (!slab)
            return false;
        m_Slabs[m_SlabCount++] = slab;
        // Pushed in reverse, so that the slab's pages are handed out in address order.
        for (uint32_t i = m_PagesPerSlab; i-- > 0;)
        {
            FreePage* page = reinterpret_cast<FreePage*>(slab + size_t(i) * m_PageSize);
            page->next = m_FreeList;
            m_FreeList = page;
        }
        return true;
    }

    bool PagePool::Reserve(uint32_t freePages) noexcept
    {
        while (GetFreePageCount() < freePages)
            if (!AddSlab())
                return false;
        return true;
    }

    void* PagePool::Allocate() noexcept
    {
        if (!m_FreeList && !AddSlab())
            return nullptr;
        FreePage* page = m_FreeList;
        m_FreeList = page->next;
        ++m_PagesInUse;
        return page;
    }

    void PagePool::Free(void* page) noexcept
    {
        NVRHI_ASSERT(m_PagesInUse > 0);
        FreePage* freed = static_cast<FreePage*>(page);
        freed->next = m_FreeList;
        m_FreeList = freed;
        --m_PagesInUse;
    }

    void PagePool::Release() noexcept
    {
        NVRHI_ASSERT(m_PagesInUse == 0);
        uint32_t first = 0;
        if (m_Shared && m_SlabCount != 0)
        {
            // given back in reverse, so that the next pool takes the slabs in this pool's order
            const size_t slabBytes = size_t(m_PageSize) * m_PagesPerSlab;
            SlabCache& cache = GetSlabCache();
            std::lock_guard<std::mutex> lock(cache.mutex);
            const uint64_t room =
                cache.bytes < c_MaxSharedSlabBytes ? (c_MaxSharedSlabBytes - cache.bytes) / slabBytes : 0;
            const uint32_t kept = uint32_t(room < m_SlabCount ? room : m_SlabCount);
            if (cache.count + kept > cache.capacity)
            {
                uint32_t capacity = cache.capacity != 0 ? cache.capacity : 64;
                while (capacity < cache.count + kept)
                    capacity *= 2;
                SharedSlab* slabs =
                    static_cast<SharedSlab*>(HeapAllocate(capacity * sizeof(SharedSlab), alignof(SharedSlab)));
                if (slabs)
                {
                    if (cache.count != 0)
                        std::memcpy(slabs, cache.slabs, cache.count * sizeof(SharedSlab));
                    HeapFree(cache.slabs);
                    cache.slabs = slabs;
                    cache.capacity = capacity;
                }
            }
            const uint32_t fits = cache.capacity - cache.count < kept ? cache.capacity - cache.count : kept;
            for (uint32_t i = 0; i < fits; ++i)
                cache.slabs[cache.count++] = SharedSlab{ m_Slabs[m_SlabCount - 1 - i], slabBytes };
            cache.bytes += uint64_t(fits) * slabBytes;
            first = fits;
        }
        for (uint32_t i = first; i < m_SlabCount; ++i)
            HeapFree(m_Slabs[m_SlabCount - 1 - i]);
        HeapFree(m_Slabs);
        m_Slabs = nullptr;
        m_SlabCount = 0;
        m_SlabCapacity = 0;
        m_FreeList = nullptr;
        m_PagesInUse = 0;
    }

    uint64_t GetSharedSlabBytes() noexcept
    {
        SlabCache& cache = GetSlabCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        return cache.bytes;
    }

    void ReleaseSharedSlabs() noexcept
    {
        SlabCache& cache = GetSlabCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        for (uint32_t i = 0; i < cache.count; ++i)
            HeapFree(cache.slabs[i].memory);
        HeapFree(cache.slabs);
        cache.slabs = nullptr;
        cache.count = 0;
        cache.capacity = 0;
        cache.bytes = 0;
    }

    // ----------------------------------------------------------------------------------------------- chunk words

    void SetMaskRange(uint64_t* words, uint32_t first, uint32_t count, bool value) noexcept
    {
        // Structural changes only: no other thread touches the chunk, so plain stores do.
        uint32_t row = first;
        const uint32_t end = first + count;
        while (row < end)
        {
            const uint32_t bit = row & 63;
            const uint32_t span = end - row < 64 - bit ? end - row : 64 - bit;
            const uint64_t mask = (span == 64 ? ~uint64_t(0) : ((uint64_t(1) << span) - 1)) << bit;
            if (value)
                words[row >> 6] |= mask;
            else
                words[row >> 6] &= ~mask;
            row += span;
        }
    }

    // ----------------------------------------------------------------------------------------------- hashing

    uint64_t HashBytes(const void* data, size_t size, uint64_t hash) noexcept
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= bytes[i];
            hash *= 0x100000001b3ull;
        }
        return hash;
    }

    void HashIndex::Place(uint64_t hash, uint32_t value) noexcept
    {
        const size_t mask = m_Slots.size() - 1;
        size_t i = size_t(hash) & mask;
        while (m_Slots[i].value != c_NotFound)
            i = (i + 1) & mask;
        m_Slots[i].hash = hash;
        m_Slots[i].value = value;
    }

    void HashIndex::Insert(uint64_t hash, uint32_t value) noexcept
    {
        if (size_t(m_Count + 1) * 2 > m_Slots.size())
        {
            Vector<Slot> old(std::move(m_Slots));
            const size_t size = old.size() < 16 ? 16 : old.size() * 2;
            m_Slots = Vector<Slot>();
            m_Slots.assign(size, Slot{ 0, c_NotFound, 0 });
            for (const Slot& slot : old)
                if (slot.value != c_NotFound)
                    Place(slot.hash, slot.value);
        }
        Place(hash, value);
        ++m_Count;
    }
}
