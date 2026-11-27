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

// The job system (aggregate docs/ecs-design.md 6.4, decision D10; plan step E2.1), the implementation of JobSystem.h.
//
// - Jobs live in a fixed table, and each dependency on a job that has not finished takes an edge from a fixed pool.
//   Both are recycled through lock-free index stacks, so nothing is allocated after CreateJobSystem.
// - A runnable job is represented by tickets, one per thread that may join it: min(ranges, workers + 1). A thread that
//   takes a ticket claims ranges from the job's counter until none are left. The thread that makes the job runnable
//   pushes the tickets onto its own work-stealing deque (one per worker and per external slot), where idle threads
//   steal them, all at once. A thread without a slot puts one ticket into a bounded queue that every thread reads,
//   and the thread that takes it pushes the others. Only the first ticket of a job is needed: when a deque is full,
//   the first goes to the shared queue (which therefore holds at most one ticket per job) and the others are dropped.
// - The job's own reference (until it finishes) and every ticket hold its slot; the last one returns the slot to the
//   free stack. A handle carries the slot's generation, so it stays complete when the slot is used again.
// - A waiting thread runs tickets, of its job or others, until its job has finished. A Wait for a job with a range
//   running lower on the same thread's stack returns at once (it could never finish), and a Wait blocked on a free
//   external slot ends when its job finishes. An idle worker looks for work (pause, then yield between the looks) for
//   idleSpinMicroseconds, a waiting thread for spinCount rounds. Then it sleeps on an event count that every new ticket,
//   and every completion of a job that a sleeping thread waits for, signals.
// - An idle worker's spin is a time because rounds do not bound it: a round visits every slot's deque, and 4,096 rounds
//   took 1.5 ms with the shared system's 79 slots, so that a process that ran one small job per 2 ms frame kept 6 cores
//   busy (aggregate plan, "E5.8 fix", with the sweep behind the default of 30 us). A waiting thread keeps the rounds: it
//   has nothing else to do, and its wake-up wakes every sleeping worker (Complete).

#include <donut/engine/JobSystem.h>
#include "SharedJobSystem.h"

#include <donut/core/log.h>
#include <nvrhi/core/foundation.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <thread>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define DONUT_JOB_SYSTEM_PAUSE() _mm_pause()
#else
#define DONUT_JOB_SYSTEM_PAUSE() std::atomic_signal_fence(std::memory_order_seq_cst)
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace donut::engine::jobs
{
    constexpr uint32_t c_JobCapacity = 4096;                  // see Schedule in JobSystem.h
    constexpr uint32_t c_EdgeCapacity = 4 * c_JobCapacity;
    constexpr uint32_t c_MaxWorkerThreads = 256;
    constexpr uint32_t c_MaxExternalThreads = 64;             // a bit each in a 64-bit mask
    constexpr uint32_t c_None = ~0u;
    constexpr uint32_t c_PauseRounds = 64;                    // idle rounds that pause; the later ones yield
    constexpr uint32_t c_PausesPerRound = 8;
    // See Initialize: after c_StartupNoticeMs a slow start is noted (log::info), after c_StartupTimeoutMs the wait
    // gives up (log::warning). A loaded machine can take seconds to start a thread (T1-N1): only the give-up, which
    // ends the start-up guarantee, is a warning.
    constexpr uint32_t c_StartupNoticeMs = 2000;
    constexpr uint32_t c_StartupTimeoutMs = 30000;

    // Shared::stop: what an idle worker does. Leave: it leaves its loop (when its state goes from Running to
    // Leaving). Park: it sleeps until the process ends, and never runs thread-exit code (StopWorkers).
    constexpr uint32_t c_StopNone = 0;
    constexpr uint32_t c_StopLeave = 1;
    constexpr uint32_t c_StopPark = 2;
    // A worker's state; Running moves to Leaving (the worker) or to Parked (StopWorkers), whichever comes first.
    constexpr uint32_t c_WorkerRunning = 0;
    constexpr uint32_t c_WorkerLeaving = 1;
    constexpr uint32_t c_WorkerParked = 2;

    // The low half of JobSlot::status; the high half is the generation, which is never 0 in a handle.
    // Completing: the ranges are done and no dependency can be added, but the dependents and edges are still being
    // released; Wait and IsComplete see Finished only after that, so a waiter finds the job's edges back in the pool.
    constexpr uint32_t c_StateActive = 1;
    constexpr uint32_t c_StateFinished = 2;
    constexpr uint32_t c_StateCompleting = 3;

    inline void Relax(uint32_t round) noexcept
    {
        if (round < c_PauseRounds)
        {
            for (uint32_t i = 0; i < c_PausesPerRound; ++i)
                DONUT_JOB_SYSTEM_PAUSE();
        }
        else
            std::this_thread::yield();
    }

    inline uint32_t RangeCount(uint32_t count, uint32_t grainSize) noexcept
    {
        return count ? (count - 1) / grainSize + 1 : 0;
    }

    inline uint32_t LowestBit(uint64_t mask) noexcept
    {
#if defined(_MSC_VER)
        unsigned long index = 0;
        _BitScanForward64(&index, mask);
        return uint32_t(index);
#else
        return uint32_t(__builtin_ctzll(mask));
#endif
    }

    class SpinLock
    {
    public:
        void lock() noexcept
        {
            for (uint32_t round = 0;; ++round)
            {
                if (!m_Locked.load(std::memory_order_relaxed) && !m_Locked.exchange(true, std::memory_order_acquire))
                    return;
                Relax(round);
            }
        }
        void unlock() noexcept { m_Locked.store(false, std::memory_order_release); }

    private:
        std::atomic<bool> m_Locked{ false };
    };

    // A lock-free stack of the indices [0, capacity) (Treiber), linked through an array of its own. The head carries a
    // tag that every change advances, so a pop cannot succeed on a head that was popped and pushed back meanwhile
    // (ABA). A popped index owns its link until it is pushed back, so the caller may chain indices through Link().
    class IndexStack
    {
    public:
        ~IndexStack() { delete[] m_Next; }

        bool Initialize(uint32_t capacity) noexcept
        {
            m_Next = new (std::nothrow) std::atomic<uint32_t>[capacity];
            if (!m_Next)
                return false;
            for (uint32_t i = 0; i < capacity; ++i)
                m_Next[i].store(i + 1 < capacity ? i + 1 : c_None, std::memory_order_relaxed);
            m_Head.store(capacity ? 0 : c_None, std::memory_order_relaxed);
            return true;
        }

        uint32_t Pop() noexcept
        {
            uint64_t head = m_Head.load(std::memory_order_acquire);
            for (;;)
            {
                const uint32_t index = uint32_t(head);
                if (index == c_None)
                    return c_None;
                const uint32_t next = m_Next[index].load(std::memory_order_relaxed);
                const uint64_t newHead = (((head >> 32) + 1) << 32) | next;
                if (m_Head.compare_exchange_weak(head, newHead, std::memory_order_acquire, std::memory_order_acquire))
                    return index;
            }
        }

        // Pushes the chain from `first` to `last`, linked through Link().
        void PushChain(uint32_t first, uint32_t last) noexcept
        {
            uint64_t head = m_Head.load(std::memory_order_relaxed);
            for (;;)
            {
                m_Next[last].store(uint32_t(head), std::memory_order_relaxed);
                const uint64_t newHead = (((head >> 32) + 1) << 32) | first;
                if (m_Head.compare_exchange_weak(head, newHead, std::memory_order_release, std::memory_order_relaxed))
                    return;
            }
        }

        void Push(uint32_t index) noexcept { PushChain(index, index); }

        std::atomic<uint32_t>& Link(uint32_t index) noexcept { return m_Next[index]; }

    private:
        std::atomic<uint32_t>* m_Next = nullptr;
        alignas(64) std::atomic<uint64_t> m_Head{ c_None };
    };

    // A work-stealing deque of job indices with a fixed capacity (Chase and Lev; the memory orders of Le, Pop, Cohen
    // and Zappa Nardelli, 2013). The thread that holds the slot pushes and pops at the bottom; any thread steals at the
    // top.
    class alignas(64) WorkDeque
    {
    public:
        enum class StealResult { Empty, Lost, Taken };

        ~WorkDeque() { delete[] m_Buffer; }

        bool Initialize(uint32_t capacity) noexcept   // a power of two
        {
            m_Buffer = new (std::nothrow) std::atomic<uint32_t>[capacity];
            m_Mask = int64_t(capacity) - 1;
            return m_Buffer != nullptr;
        }

        bool Push(uint32_t value) noexcept
        {
            const int64_t bottom = m_Bottom.load(std::memory_order_relaxed);
            const int64_t top = m_Top.load(std::memory_order_acquire);
            if (bottom - top > m_Mask)
                return false;
            m_Buffer[bottom & m_Mask].store(value, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            m_Bottom.store(bottom + 1, std::memory_order_relaxed);
            return true;
        }

        bool Pop(uint32_t& value) noexcept
        {
            const int64_t bottom = m_Bottom.load(std::memory_order_relaxed) - 1;
            m_Bottom.store(bottom, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            int64_t top = m_Top.load(std::memory_order_relaxed);
            if (top > bottom)
            {
                m_Bottom.store(bottom + 1, std::memory_order_relaxed);
                return false;
            }
            value = m_Buffer[bottom & m_Mask].load(std::memory_order_relaxed);
            if (top < bottom)
                return true;
            // The last ticket: the thieves may race for it.
            const bool won =
                m_Top.compare_exchange_strong(top, top + 1, std::memory_order_seq_cst, std::memory_order_relaxed);
            m_Bottom.store(bottom + 1, std::memory_order_relaxed);
            return won;
        }

        StealResult Steal(uint32_t& value) noexcept
        {
            int64_t top = m_Top.load(std::memory_order_acquire);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const int64_t bottom = m_Bottom.load(std::memory_order_acquire);
            if (top >= bottom)
                return StealResult::Empty;
            value = m_Buffer[top & m_Mask].load(std::memory_order_relaxed);
            return m_Top.compare_exchange_strong(top, top + 1, std::memory_order_seq_cst, std::memory_order_relaxed)
                ? StealResult::Taken : StealResult::Lost;
        }

        // Read after a sequentially consistent fence, before a thread sleeps.
        bool MayHaveWork() const noexcept
        {
            return m_Bottom.load(std::memory_order_relaxed) > m_Top.load(std::memory_order_relaxed);
        }

        uint32_t random = 1;   // the slot holder's choice of victims (xorshift32)

    private:
        std::atomic<uint32_t>* m_Buffer = nullptr;
        int64_t m_Mask = 0;
        alignas(64) std::atomic<int64_t> m_Top{ 0 };
        alignas(64) std::atomic<int64_t> m_Bottom{ 0 };
    };

    // A bounded multi-producer, multi-consumer queue (Vyukov) for the first tickets of jobs that a thread without a
    // slot (or with a full deque) made runnable. Each job has at most one ticket here, so the job table's capacity is
    // enough.
    class InjectionQueue
    {
    public:
        ~InjectionQueue() { delete[] m_Cells; }

        bool Initialize(uint32_t capacity) noexcept   // a power of two
        {
            m_Cells = new (std::nothrow) Cell[capacity];
            if (!m_Cells)
                return false;
            m_Mask = capacity - 1;
            for (uint32_t i = 0; i < capacity; ++i)
                m_Cells[i].sequence.store(i, std::memory_order_relaxed);
            return true;
        }

        bool Enqueue(uint32_t value) noexcept
        {
            uint64_t position = m_EnqueuePosition.load(std::memory_order_relaxed);
            for (;;)
            {
                Cell& cell = m_Cells[position & m_Mask];
                const uint64_t sequence = cell.sequence.load(std::memory_order_acquire);
                const int64_t difference = int64_t(sequence - position);
                if (difference == 0)
                {
                    if (m_EnqueuePosition.compare_exchange_weak(position, position + 1, std::memory_order_relaxed))
                    {
                        cell.value = value;
                        cell.sequence.store(position + 1, std::memory_order_release);
                        return true;
                    }
                }
                else if (difference < 0)
                    return false;
                else
                    position = m_EnqueuePosition.load(std::memory_order_relaxed);
            }
        }

        bool Dequeue(uint32_t& value) noexcept
        {
            uint64_t position = m_DequeuePosition.load(std::memory_order_relaxed);
            for (;;)
            {
                Cell& cell = m_Cells[position & m_Mask];
                const uint64_t sequence = cell.sequence.load(std::memory_order_acquire);
                const int64_t difference = int64_t(sequence - (position + 1));
                if (difference == 0)
                {
                    if (m_DequeuePosition.compare_exchange_weak(position, position + 1, std::memory_order_relaxed))
                    {
                        value = cell.value;
                        cell.sequence.store(position + m_Mask + 1, std::memory_order_release);
                        return true;
                    }
                }
                else if (difference < 0)
                    return false;
                else
                    position = m_DequeuePosition.load(std::memory_order_relaxed);
            }
        }

        // Read after a sequentially consistent fence, before a thread sleeps.
        bool MayHaveWork() const noexcept
        {
            return m_EnqueuePosition.load(std::memory_order_relaxed) !=
                m_DequeuePosition.load(std::memory_order_relaxed);
        }

    private:
        struct Cell
        {
            std::atomic<uint64_t> sequence;
            uint32_t value;
        };
        Cell* m_Cells = nullptr;
        uint64_t m_Mask = 0;
        alignas(64) std::atomic<uint64_t> m_EnqueuePosition{ 0 };
        alignas(64) std::atomic<uint64_t> m_DequeuePosition{ 0 };
    };

    // Sleeping without lost wake-ups (an event count): a thread announces that it will sleep (PrepareWait), looks for
    // work once more, and sleeps only if no notification came after the announcement. Between the work a notifier
    // published and its check for sleepers, and between a sleeper's announcement and its last look, stand sequentially
    // consistent fences, so one of the two sees the other. While nobody sleeps, a notification is a fence and a load.
    class EventCount
    {
    public:
        uint32_t PrepareWait() noexcept
        {
            m_Waiters.fetch_add(1, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            return m_Epoch.load(std::memory_order_acquire);
        }

        void CancelWait() noexcept { m_Waiters.fetch_sub(1, std::memory_order_relaxed); }

        void Wait(uint32_t key) noexcept
        {
            {
                std::unique_lock<std::mutex> lock(m_Mutex);
                while (m_Epoch.load(std::memory_order_relaxed) == key)
                    m_Condition.wait(lock);
            }
            m_Waiters.fetch_sub(1, std::memory_order_relaxed);
        }

        // Wakes up to `count` sleeping threads.
        void Notify(uint32_t count) noexcept
        {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const uint32_t waiters = m_Waiters.load(std::memory_order_relaxed);
            if (waiters == 0 || count == 0)
                return;
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Epoch.fetch_add(1, std::memory_order_release);
            }
            if (count >= waiters)
                m_Condition.notify_all();
            else
            {
                for (uint32_t i = 0; i < count; ++i)
                    m_Condition.notify_one();
            }
        }

        // Shutdown: whether or not anybody sleeps.
        void WakeAll() noexcept
        {
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Epoch.fetch_add(1, std::memory_order_release);
            }
            m_Condition.notify_all();
        }

    private:
        alignas(64) std::atomic<uint32_t> m_Waiters{ 0 };
        std::atomic<uint32_t> m_Epoch{ 0 };
        std::mutex m_Mutex;
        std::condition_variable m_Condition;
    };

    struct alignas(64) JobSlot
    {
        // Set by Schedule before the job can run; read by the threads that run it.
        PFN_JobFunction function = nullptr;
        void* context = nullptr;
        const char* debugName = nullptr;
        uint32_t count = 0;
        uint32_t grainSize = 1;
        uint32_t rangeCount = 0;
        std::atomic<uint64_t> status{ c_StateFinished };   // generation << 32 | state; read by Wait and IsComplete

        // Changed while the job runs.
        alignas(64) std::atomic<uint64_t> nextRange{ 0 };  // the next range to claim; claims run past the end
        std::atomic<uint32_t> rangesLeft{ 0 };             // ranges not yet finished
        std::atomic<uint32_t> pendingDependencies{ 0 };    // plus one while Schedule links them
        std::atomic<uint32_t> references{ 0 };            // the job's own until it finishes, plus one per ticket
        std::atomic<uint32_t> hasWaiters{ 0 };             // a thread may sleep until it finishes
        SpinLock dependentsLock;
        uint32_t firstDependent = c_None;                  // edges to the jobs that wait for it, under the lock
        uint32_t nextReady = c_None;                       // Complete's list of jobs without ranges
    };

    // The slots a thread holds, innermost first: a worker's for its lifetime, an external slot for the duration of a
    // Wait or ParallelFor call (nested calls find it). A thread may hold slots of several job systems.
    struct ThreadFrame
    {
        const void* system;
        uint32_t index;
        ThreadFrame* previous;
    };
    static thread_local ThreadFrame* t_ThreadFrames = nullptr;

    // The jobs whose tickets this thread is running, innermost first. A Wait for one of them from further up the same
    // stack could never return: the job cannot finish before the range below returns.
    struct RunningFrame
    {
        const void* system;
        uint32_t index;
        uint32_t generation;
        RunningFrame* previous;
    };
    static thread_local RunningFrame* t_RunningJobs = nullptr;

    static void NameThread(const char* debugName, uint32_t worker) noexcept
    {
#if defined(_WIN32)
        typedef HRESULT(WINAPI* PFN_SetThreadDescription)(HANDLE, PCWSTR);
        HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
        const PFN_SetThreadDescription setThreadDescription = kernel ? reinterpret_cast<PFN_SetThreadDescription>(
            reinterpret_cast<void*>(GetProcAddress(kernel, "SetThreadDescription"))) : nullptr;
        if (!setThreadDescription)
            return;
        char name[96];
        std::snprintf(name, sizeof(name), "%s worker %u", debugName, worker);
        wchar_t wideName[96];
        if (MultiByteToWideChar(CP_UTF8, 0, name, -1, wideName, 96) > 0)
            setThreadDescription(GetCurrentThread(), wideName);
#else
        (void)debugName;
        (void)worker;
#endif
    }

    NVRHI_CLASS_CLSID(JobSystemImpl, "d77ebe6b-3c14-4e73-99a2-118f6b259e17")
    class JobSystemImpl final : public nvrhi::ObjectImpl<IJobSystem>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(JobSystemImpl)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(JobSystemImpl)
        NVRHI_IMPLEMENTS_INTERFACE(IJobSystem)
        NVRHI_IMPLEMENTS_CLASS(JobSystemImpl)
        NVRHI_END_INTERFACE_TABLE()

        JobSystemImpl() noexcept = default;
        ~JobSystemImpl();
        nvrhi::FRESULT Initialize(const JobSystemDesc& desc) noexcept;

        // IJobSystem
        uint32_t GetThreadCount() noexcept override { return m_ThreadCount; }
        uint32_t GetCurrentWorkerIndex() noexcept override { return FindThreadIndex(); }
        nvrhi::FRESULT Schedule(const JobDesc& job, const JobHandle* dependencies, uint32_t dependencyCount,
                                JobHandle& outJob) noexcept override;
        void Wait(JobHandle job) noexcept override;
        bool IsComplete(JobHandle job) noexcept override;
        void ParallelFor(uint32_t count, uint32_t grainSize, PFN_JobFunction function, void* context) noexcept override;

        // The shared system's exit handler (SharedJobSystem.h): the workers finish the tickets they can find, then
        // leave, and are joined. The workers still busy after `timeoutMs` are parked instead (they sleep once their
        // jobs return); so are all of them when the calling thread is inside a call or job of this system. Returns
        // whether every worker was joined. The system stays usable: Wait and ParallelFor run on the calling thread.
        bool StopWorkers(uint32_t timeoutMs, uint32_t& parkedCount) noexcept;

    private:
        // The state the threads share, allocated on its own so that its alignment holds.
        struct Shared
        {
            IndexStack freeJobs;
            IndexStack freeEdges;
            InjectionQueue injection;
            EventCount events;
            alignas(64) std::atomic<uint64_t> freeExternalSlots{ 0 };
            std::atomic<uint32_t> slotWaiters{ 0 };
            std::mutex slotMutex;
            std::condition_variable slotCondition;
            alignas(64) std::atomic<uint32_t> liveJobs{ 0 };   // scheduled and not finished
            std::atomic<uint32_t> stop{ c_StopNone };
            std::atomic<bool> warnedFull{ false };
            std::atomic<uint32_t> runningWorkers{ 0 };         // workers past start-up (Initialize waits for them)
        };

        // The calling thread's slot for the duration of a call: the one it holds, or a free external slot, waited for.
        // With an awaited job (Wait), the wait for a slot also ends when that job finishes; Index() is then c_None.
        class SlotScope
        {
        public:
            explicit SlotScope(JobSystemImpl* system, uint32_t awaitedIndex = c_None,
                               uint32_t awaitedGeneration = 0) noexcept
                : m_System(system)
                , m_Index(system->FindThreadIndex())
            {
                if (m_Index != c_None)
                    return;
                m_Index = system->AcquireExternalSlot(awaitedIndex, awaitedGeneration);
                if (m_Index == c_None)
                    return;
                m_Frame = { system, m_Index, t_ThreadFrames };
                t_ThreadFrames = &m_Frame;
                m_Owned = true;
            }
            ~SlotScope()
            {
                if (!m_Owned)
                    return;
                t_ThreadFrames = m_Frame.previous;
                m_System->ReleaseExternalSlot(m_Index);
            }
            SlotScope(const SlotScope&) = delete;
            SlotScope& operator=(const SlotScope&) = delete;

            uint32_t Index() const noexcept { return m_Index; }

        private:
            JobSystemImpl* m_System;
            uint32_t m_Index;
            ThreadFrame m_Frame{};
            bool m_Owned = false;
        };

        uint32_t FindThreadIndex() const noexcept;
        uint32_t AcquireExternalSlot(uint32_t awaitedIndex, uint32_t awaitedGeneration) noexcept;
        void NotifySlotWaiters() noexcept;
        void ReleaseExternalSlot(uint32_t index) noexcept;
        bool IsFinished(uint32_t index, uint32_t generation) const noexcept;
        nvrhi::FRESULT ScheduleInternal(const JobDesc& job, const JobHandle* dependencies, uint32_t dependencyCount,
                                        JobHandle& outJob, uint32_t self) noexcept;
        uint32_t TicketCount(uint32_t rangeCount) const noexcept { return std::min(rangeCount, m_WorkerCount + 1); }
        void MakeRunnable(uint32_t index, uint32_t self) noexcept;
        void PushTickets(uint32_t index, uint32_t self, uint32_t count) noexcept;
        void RunTicket(uint32_t index, uint32_t self, bool shared) noexcept;
        void Complete(uint32_t index, uint32_t self) noexcept;
        void ReleaseReferences(uint32_t index, uint32_t count) noexcept;
        bool FindWork(uint32_t self, uint32_t& ticket, bool& shared) noexcept;
        bool MayHaveWork() const noexcept;
        void WaitFor(uint32_t index, uint32_t generation, uint32_t self) noexcept;
        void Drain() noexcept;
        void WorkerMain(uint32_t self) noexcept;
        [[noreturn]] static void Park() noexcept;

        uint32_t m_ExternalCount = 0;
        uint32_t m_WorkerCount = 0;
        uint32_t m_ThreadCount = 0;
        uint32_t m_SpinCount = 0;                    // a waiting thread's rounds
        std::chrono::microseconds m_IdleSpin{ 0 };   // an idle worker's time
        char m_Name[64] = {};
        Shared* m_Shared = nullptr;
        JobSlot* m_Jobs = nullptr;
        uint32_t* m_EdgeJobs = nullptr;    // per edge: the job that waits
        WorkDeque* m_Deques = nullptr;     // per slot: external slots, then workers
        std::thread* m_Threads = nullptr;
        std::atomic<uint32_t>* m_WorkerStates = nullptr;   // per worker: c_WorkerRunning, Leaving or Parked
        uint32_t m_StartedThreads = 0;
    };

    nvrhi::FRESULT JobSystemImpl::Initialize(const JobSystemDesc& desc) noexcept
    {
        uint32_t workers = desc.workerThreadCount;
        if (workers == 0)
        {
            const uint32_t hardware = std::thread::hardware_concurrency();
            workers = std::min(hardware > 1 ? hardware - 1 : 1u, c_MaxWorkerThreads);
        }
        m_WorkerCount = workers;
        m_ExternalCount = desc.maxExternalThreads;
        m_ThreadCount = m_ExternalCount + m_WorkerCount;
        m_SpinCount = desc.spinCount;
        m_IdleSpin = std::chrono::microseconds(desc.idleSpinMicroseconds);
        std::snprintf(m_Name, sizeof(m_Name), "%s", desc.debugName ? desc.debugName : "JobSystem");

        m_Shared = new (std::nothrow) Shared;
        m_Jobs = new (std::nothrow) JobSlot[c_JobCapacity];
        m_EdgeJobs = new (std::nothrow) uint32_t[c_EdgeCapacity];
        m_Deques = new (std::nothrow) WorkDeque[m_ThreadCount];
        m_Threads = new (std::nothrow) std::thread[m_WorkerCount];
        m_WorkerStates = new (std::nothrow) std::atomic<uint32_t>[m_WorkerCount];
        if (!m_Shared || !m_Jobs || !m_EdgeJobs || !m_Deques || !m_Threads || !m_WorkerStates ||
            !m_Shared->freeJobs.Initialize(c_JobCapacity) || !m_Shared->freeEdges.Initialize(c_EdgeCapacity) ||
            !m_Shared->injection.Initialize(c_JobCapacity))
            return nvrhi::FE_OUT_OF_MEMORY;
        for (uint32_t i = 0; i < m_ThreadCount; ++i)
        {
            if (!m_Deques[i].Initialize(c_JobCapacity))
                return nvrhi::FE_OUT_OF_MEMORY;
            m_Deques[i].random = 0x9e3779b9u * (i + 1);
        }
        m_Shared->freeExternalSlots.store(m_ExternalCount == 64 ? ~0ull : (1ull << m_ExternalCount) - 1,
                                          std::memory_order_relaxed);
        for (uint32_t i = 0; i < m_WorkerCount; ++i)
            m_WorkerStates[i].store(c_WorkerRunning, std::memory_order_relaxed);

        for (uint32_t i = 0; i < m_WorkerCount; ++i)
        {
            try
            {
                m_Threads[i] = std::thread(&JobSystemImpl::WorkerMain, this, m_ExternalCount + i);
            }
            catch (...)
            {
                log::warning("CreateJobSystem: could not start worker thread %u of %u", i, m_WorkerCount);
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            ++m_StartedThreads;
        }
        // Start-up completes before CreateJobSystem returns. A worker still in thread start-up runs C runtime code,
        // and a process that ends meanwhile can tear that runtime down under it: exit, _Exit and quick_exit
        // uninitialize a static CRT before ExitProcess ends the threads (SharedJobSystem.h). Once started, an idle
        // worker touches no C runtime state (it spins, yields, or sleeps on an SRW-based condition variable).
        // Under the loader lock (DllMain, a DLL's static constructors and destructors) no thread can start: creation
        // there is not supported, and the wait gives up after c_StartupTimeoutMs (a note after c_StartupNoticeMs).
        const auto start = std::chrono::steady_clock::now();
        const auto notice = start + std::chrono::milliseconds(c_StartupNoticeMs);
        const auto deadline = start + std::chrono::milliseconds(c_StartupTimeoutMs);
        bool noted = false;
        for (uint32_t round = 0; m_Shared->runningWorkers.load(std::memory_order_acquire) < m_StartedThreads; ++round)
        {
            Relax(round);
            if (round < c_PauseRounds)
                continue;
            const auto now = std::chrono::steady_clock::now();
            if (!noted && now >= notice)
            {
                noted = true;
                log::info("CreateJobSystem('%s'): %u of %u workers have not started after %u ms; still waiting",
                          m_Name, m_StartedThreads - m_Shared->runningWorkers.load(), m_StartedThreads,
                          c_StartupNoticeMs);
            }
            if (now >= deadline)
            {
                log::warning("CreateJobSystem('%s'): %u of %u workers have not started after %u ms (created under the "
                             "loader lock?); it returns without waiting for them", m_Name,
                             m_StartedThreads - m_Shared->runningWorkers.load(), m_StartedThreads, c_StartupTimeoutMs);
                break;
            }
        }
        return nvrhi::FS_OK;
    }

    JobSystemImpl::~JobSystemImpl()
    {
        if (m_Shared && m_StartedThreads != 0)
        {
            if (FindThreadIndex() != c_None)
            {
                // A worker cannot join itself, and an external slot would be released into freed memory.
                log::warning("IJobSystem '%s': the last reference was released by a thread inside one of its calls or "
                             "jobs", m_Name);
                std::abort();
            }
            // Clean shutdown: every scheduled job runs, with this thread's help, before the workers stop.
            Drain();
            m_Shared->stop.store(c_StopLeave, std::memory_order_seq_cst);
            m_Shared->events.WakeAll();
            for (uint32_t i = 0; i < m_StartedThreads; ++i)
            {
                // A parked worker (StopWorkers) never leaves; its thread was detached.
                if (m_Threads[i].joinable() && m_WorkerStates[i].load(std::memory_order_acquire) != c_WorkerParked)
                    m_Threads[i].join();
            }
        }
        delete[] m_WorkerStates;
        delete[] m_Threads;
        delete[] m_Deques;
        delete[] m_EdgeJobs;
        delete[] m_Jobs;
        delete m_Shared;
    }

    // Exit runs this while the C runtime holds its exit lock (UCRT runs the atexit and at_quick_exit tables under
    // it), so a job that registers an exit function meanwhile (a function-local static with a destructor, built for
    // the first time) blocks until the process ends, and so does a job stuck on state the exit tore down: an
    // unbounded join would hang the exit. And a worker that left its loop during the C runtime's teardown would run
    // thread-exit code (the std::thread state, thread_local destructors, _endthreadex) against it. So the workers that
    // are not joined in time are parked instead: they sleep, once their jobs return, until ExitProcess ends them.
    bool JobSystemImpl::StopWorkers(uint32_t timeoutMs, uint32_t& parkedCount) noexcept
    {
        parkedCount = 0;
        if (!m_Shared || m_StartedThreads == 0)
            return true;
        if (FindThreadIndex() != c_None)
        {
            // A worker cannot join itself, and the others may wait for ranges on this thread's stack.
            m_Shared->stop.store(c_StopPark, std::memory_order_seq_cst);
            m_Shared->events.WakeAll();
            parkedCount = m_StartedThreads;
            return false;
        }
        // A worker leaves only when it finds no ticket after it has seen the stop (WorkerMain), so the jobs in flight,
        // and those scheduled before this call, finish first.
        m_Shared->stop.store(c_StopLeave, std::memory_order_seq_cst);
        m_Shared->events.WakeAll();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        for (uint32_t round = 0;; ++round)
        {
            uint32_t leaving = 0;
            for (uint32_t i = 0; i < m_StartedThreads; ++i)
                leaving += m_WorkerStates[i].load(std::memory_order_acquire) == c_WorkerLeaving ? 1 : 0;
            if (leaving == m_StartedThreads)
                break;
            if (round < c_PauseRounds)
                Relax(round);
            else if (std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            else
            {
                // The workers that finish their jobs later sleep instead of leaving: whichever of this thread and the
                // worker moves its state out of Running first decides (WorkerMain).
                m_Shared->stop.store(c_StopPark, std::memory_order_seq_cst);
                for (uint32_t i = 0; i < m_StartedThreads; ++i)
                {
                    uint32_t expected = c_WorkerRunning;
                    if (m_WorkerStates[i].compare_exchange_strong(expected, c_WorkerParked, std::memory_order_acq_rel))
                    {
                        m_Threads[i].detach();
                        ++parkedCount;
                    }
                }
                break;
            }
        }
        // The leaving workers are past their loop: each join ends once the thread's exit code has run.
        for (uint32_t i = 0; i < m_StartedThreads; ++i)
        {
            if (!m_Threads[i].joinable())
                continue;
            try
            {
                m_Threads[i].join();
            }
            catch (...)
            {
                ++parkedCount;
            }
        }
        return parkedCount == 0;
    }

    void JobSystemImpl::Park() noexcept
    {
        for (;;)
            std::this_thread::sleep_for(std::chrono::hours(24));
    }

    uint32_t JobSystemImpl::FindThreadIndex() const noexcept
    {
        for (const ThreadFrame* frame = t_ThreadFrames; frame; frame = frame->previous)
        {
            if (frame->system == this)
                return frame->index;
        }
        return c_None;
    }

    // Returns a free external slot, waiting for one if all are taken (the user's decision, E1.0). With an awaited job,
    // returns c_None as soon as that job has finished instead: Complete wakes the slot waiters of a job whose
    // hasWaiters is set. The waiter sets it and the job finishes, each before a sequentially consistent fence that
    // precedes the other's check, so either the waiter sees the job finished or Complete sees the flag (and the
    // incremented slotWaiters) and notifies under the mutex.
    uint32_t JobSystemImpl::AcquireExternalSlot(uint32_t awaitedIndex, uint32_t awaitedGeneration) noexcept
    {
        Shared& shared = *m_Shared;
        for (;;)
        {
            uint64_t mask = shared.freeExternalSlots.load(std::memory_order_acquire);
            while (mask != 0)
            {
                const uint32_t index = LowestBit(mask);
                const uint64_t taken = mask & ~(1ull << index);
                if (shared.freeExternalSlots.compare_exchange_weak(mask, taken, std::memory_order_acq_rel,
                                                                   std::memory_order_acquire))
                    return index;
            }
            bool finished = false;
            {
                std::unique_lock<std::mutex> lock(shared.slotMutex);
                shared.slotWaiters.fetch_add(1, std::memory_order_seq_cst);
                if (awaitedIndex != c_None)
                    m_Jobs[awaitedIndex].hasWaiters.store(1, std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                while (shared.freeExternalSlots.load(std::memory_order_seq_cst) == 0)
                {
                    if (awaitedIndex != c_None && IsFinished(awaitedIndex, awaitedGeneration))
                    {
                        finished = true;
                        break;
                    }
                    shared.slotCondition.wait(lock);
                }
                shared.slotWaiters.fetch_sub(1, std::memory_order_relaxed);
            }
            if (finished)
                return c_None;
        }
    }

    void JobSystemImpl::NotifySlotWaiters() noexcept
    {
        Shared& shared = *m_Shared;
        if (shared.slotWaiters.load(std::memory_order_seq_cst) != 0)
        {
            std::lock_guard<std::mutex> lock(shared.slotMutex);
            shared.slotCondition.notify_all();
        }
    }

    void JobSystemImpl::ReleaseExternalSlot(uint32_t index) noexcept
    {
        Shared& shared = *m_Shared;
        // Tickets of jobs without a range left only hold their slots: give them back. A ticket with ranges left stays
        // for the thieves (or the slot's next holder).
        WorkDeque& deque = m_Deques[index];
        uint32_t ticket;
        while (deque.Pop(ticket))
        {
            const JobSlot& slot = m_Jobs[ticket];
            if (slot.nextRange.load(std::memory_order_relaxed) < slot.rangeCount)
            {
                deque.Push(ticket);
                shared.events.Notify(1);
                break;
            }
            ReleaseReferences(ticket, 1);
        }
        shared.freeExternalSlots.fetch_or(1ull << index, std::memory_order_seq_cst);
        NotifySlotWaiters();
    }

    bool JobSystemImpl::IsFinished(uint32_t index, uint32_t generation) const noexcept
    {
        const uint64_t status = m_Jobs[index].status.load(std::memory_order_acquire);
        return uint32_t(status >> 32) != generation || uint32_t(status) == c_StateFinished;
    }

    nvrhi::FRESULT JobSystemImpl::Schedule(const JobDesc& job, const JobHandle* dependencies, uint32_t dependencyCount,
                                           JobHandle& outJob) noexcept
    {
        return ScheduleInternal(job, dependencies, dependencyCount, outJob, FindThreadIndex());
    }

    nvrhi::FRESULT JobSystemImpl::ScheduleInternal(const JobDesc& job, const JobHandle* dependencies,
                                                   uint32_t dependencyCount, JobHandle& outJob, uint32_t self) noexcept
    {
        outJob = JobHandle::Null;
        if (job.count != 0 && !job.function)
        {
            log::warning("IJobSystem::Schedule('%s'): a job with %u indices and no function",
                         job.debugName ? job.debugName : "", job.count);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (dependencyCount != 0 && !dependencies)
        {
            log::warning("IJobSystem::Schedule: %u dependencies and no array", dependencyCount);
            return nvrhi::FE_INVALID_ARGS;
        }
        uint32_t liveDependencies = 0;
        for (uint32_t i = 0; i < dependencyCount; ++i)
        {
            const uint64_t handle = uint64_t(dependencies[i]);
            if (handle == 0)
                continue;
            const uint32_t index = uint32_t(handle);
            const uint32_t generation = uint32_t(handle >> 32);
            if (index >= c_JobCapacity || generation == 0)
            {
                log::warning("IJobSystem::Schedule: dependency %u is not a job handle (0x%016llx)", i,
                             (unsigned long long)handle);
                return nvrhi::FE_INVALID_ARGS;
            }
            // A finished job stays finished, so only these can need an edge.
            if (!IsFinished(index, generation))
                ++liveDependencies;
        }

        Shared& shared = *m_Shared;
        const uint32_t index = shared.freeJobs.Pop();
        if (index == c_None)
            return nvrhi::FE_OUT_OF_MEMORY;
        // An edge per dependency that has not finished, chained through the edge links; given back when unused.
        uint32_t edges = c_None;
        uint32_t lastEdge = c_None;
        for (uint32_t i = 0; i < liveDependencies; ++i)
        {
            const uint32_t edge = shared.freeEdges.Pop();
            if (edge == c_None)
            {
                if (edges != c_None)
                    shared.freeEdges.PushChain(edges, lastEdge);
                shared.freeJobs.Push(index);
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            shared.freeEdges.Link(edge).store(edges, std::memory_order_relaxed);
            if (edges == c_None)
                lastEdge = edge;
            edges = edge;
        }

        JobSlot& slot = m_Jobs[index];
        uint32_t generation = uint32_t(slot.status.load(std::memory_order_relaxed) >> 32) + 1;
        if (generation == 0)
            generation = 1;
        const uint32_t grainSize = job.grainSize ? job.grainSize : std::max(job.count, 1u);
        slot.function = job.function;
        slot.context = job.context;
        slot.debugName = job.debugName;
        slot.count = job.count;
        slot.grainSize = grainSize;
        slot.rangeCount = RangeCount(job.count, grainSize);
        slot.nextRange.store(0, std::memory_order_relaxed);
        slot.rangesLeft.store(slot.rangeCount, std::memory_order_relaxed);
        slot.pendingDependencies.store(1, std::memory_order_relaxed);
        slot.references.store(1, std::memory_order_relaxed);
        slot.hasWaiters.store(0, std::memory_order_relaxed);
        slot.firstDependent = c_None;
        slot.nextReady = c_None;
        shared.liveJobs.fetch_add(1, std::memory_order_relaxed);
        slot.status.store((uint64_t(generation) << 32) | c_StateActive, std::memory_order_release);

        for (uint32_t i = 0; i < dependencyCount && edges != c_None; ++i)
        {
            const uint64_t handle = uint64_t(dependencies[i]);
            if (handle == 0)
                continue;
            const uint32_t dependencyGeneration = uint32_t(handle >> 32);
            JobSlot& dependency = m_Jobs[uint32_t(handle)];
            std::lock_guard<SpinLock> lock(dependency.dependentsLock);
            const uint64_t status = dependency.status.load(std::memory_order_acquire);
            if (uint32_t(status >> 32) != dependencyGeneration || uint32_t(status) != c_StateActive)
                continue;
            const uint32_t edge = edges;
            edges = shared.freeEdges.Link(edge).load(std::memory_order_relaxed);
            m_EdgeJobs[edge] = index;
            shared.freeEdges.Link(edge).store(dependency.firstDependent, std::memory_order_relaxed);
            dependency.firstDependent = edge;
            slot.pendingDependencies.fetch_add(1, std::memory_order_relaxed);
        }
        if (edges != c_None)
            shared.freeEdges.PushChain(edges, lastEdge);

        outJob = JobHandle((uint64_t(generation) << 32) | index);
        if (slot.pendingDependencies.fetch_sub(1, std::memory_order_acq_rel) == 1)
            MakeRunnable(index, self);
        return nvrhi::FS_OK;
    }

    void JobSystemImpl::MakeRunnable(uint32_t index, uint32_t self) noexcept
    {
        JobSlot& slot = m_Jobs[index];
        const uint32_t rangeCount = slot.rangeCount;
        if (rangeCount == 0)
        {
            Complete(index, self);
            return;
        }
        if (self != c_None)
        {
            PushTickets(index, self, TicketCount(rangeCount));
            return;
        }
        // A thread without a slot: one ticket in the shared queue; the thread that takes it pushes the others.
        slot.references.fetch_add(1, std::memory_order_relaxed);
        for (uint32_t round = 0; !m_Shared->injection.Enqueue(index); ++round)
            Relax(round);
        m_Shared->events.Notify(TicketCount(rangeCount));
    }

    // Pushes `count` tickets onto the slot's deque, each with a reference taken before any is published. The references
    // of tickets that do not fit are released like any other: once the first ticket is out, the job may run, finish
    // and lose every other reference before this thread gets here (MakeRunnable's caller holds none of its own), so
    // that release may be the last one.
    void JobSystemImpl::PushTickets(uint32_t index, uint32_t self, uint32_t count) noexcept
    {
        JobSlot& slot = m_Jobs[index];
        slot.references.fetch_add(count, std::memory_order_relaxed);
        WorkDeque& deque = m_Deques[self];
        uint32_t pushed = 0;
        while (pushed < count && deque.Push(index))
            ++pushed;
        if (pushed == 0)
        {
            // A full deque: the first ticket goes to the shared queue, which holds at most one ticket per job.
            for (uint32_t round = 0; !m_Shared->injection.Enqueue(index); ++round)
                Relax(round);
            pushed = 1;
        }
        m_Shared->events.Notify(pushed);
        if (pushed < count)
            ReleaseReferences(index, count - pushed);
    }

    // `shared`: the ticket came from the shared queue, so this thread pushes the job's other tickets.
    void JobSystemImpl::RunTicket(uint32_t index, uint32_t self, bool shared) noexcept
    {
        JobSlot& slot = m_Jobs[index];
        const uint32_t rangeCount = slot.rangeCount;
        // The ticket's reference keeps the slot, and so its generation, until the end.
        RunningFrame running{ this, index, uint32_t(slot.status.load(std::memory_order_relaxed) >> 32), t_RunningJobs };
        t_RunningJobs = &running;
        for (;;)
        {
            const uint64_t range = slot.nextRange.fetch_add(1, std::memory_order_relaxed);
            if (range >= rangeCount)
                break;
            if (shared)
            {
                shared = false;
                const uint32_t others = std::min(TicketCount(rangeCount) - 1, uint32_t(rangeCount - range - 1));
                if (others != 0)
                    PushTickets(index, self, others);
            }
            // The deterministic split: range r is [r * grainSize, min((r + 1) * grainSize, count)).
            const uint32_t begin = uint32_t(range) * slot.grainSize;
            slot.function(slot.context, begin, begin + std::min(slot.grainSize, slot.count - begin), self);
            if (slot.rangesLeft.fetch_sub(1, std::memory_order_acq_rel) == 1)
                Complete(index, self);
        }
        t_RunningJobs = running.previous;
        ReleaseReferences(index, 1);
    }

    // Finishes the job and every job without ranges that it makes runnable, in a loop rather than recursively.
    void JobSystemImpl::Complete(uint32_t index, uint32_t self) noexcept
    {
        Shared& shared = *m_Shared;
        m_Jobs[index].nextReady = c_None;
        uint32_t ready = index;
        while (ready != c_None)
        {
            const uint32_t current = ready;
            JobSlot& slot = m_Jobs[current];
            ready = slot.nextReady;
            const uint64_t generation = slot.status.load(std::memory_order_relaxed) & 0xffffffff00000000ull;
            uint32_t first;
            {
                // From here on Schedule links no dependent: it treats Completing like Finished.
                std::lock_guard<SpinLock> lock(slot.dependentsLock);
                slot.status.store(generation | c_StateCompleting, std::memory_order_release);
                first = slot.firstDependent;
                slot.firstDependent = c_None;
            }
            uint32_t last = c_None;
            for (uint32_t edge = first; edge != c_None;
                 edge = shared.freeEdges.Link(edge).load(std::memory_order_relaxed))
            {
                last = edge;
                const uint32_t dependent = m_EdgeJobs[edge];
                JobSlot& next = m_Jobs[dependent];
                if (next.pendingDependencies.fetch_sub(1, std::memory_order_acq_rel) != 1)
                    continue;
                if (next.rangeCount == 0)
                {
                    next.nextReady = ready;
                    ready = dependent;
                    continue;
                }
                MakeRunnable(dependent, self);
            }
            if (first != c_None)
                shared.freeEdges.PushChain(first, last);
            shared.liveJobs.fetch_sub(1, std::memory_order_release);
            // Finished only now, with the dependents released and the edges back. Pairs with the fences after a waiter
            // sets hasWaiters (WaitFor's PrepareWait, AcquireExternalSlot): either the waiter sees the job finished,
            // or this sees the flag and wakes it.
            slot.status.store(generation | c_StateFinished, std::memory_order_release);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (slot.hasWaiters.load(std::memory_order_relaxed) != 0)
            {
                shared.events.Notify(~0u);
                NotifySlotWaiters();
            }
            ReleaseReferences(current, 1);
        }
    }

    // Every release of a job's references goes through here: whichever drops the count to zero frees the slot.
    void JobSystemImpl::ReleaseReferences(uint32_t index, uint32_t count) noexcept
    {
        if (m_Jobs[index].references.fetch_sub(count, std::memory_order_acq_rel) == count)
            m_Shared->freeJobs.Push(index);
    }

    bool JobSystemImpl::FindWork(uint32_t self, uint32_t& ticket, bool& shared) noexcept
    {
        WorkDeque& own = m_Deques[self];
        shared = false;
        if (own.Pop(ticket))
            return true;
        if (m_Shared->injection.Dequeue(ticket))
        {
            shared = true;
            return true;
        }
        uint32_t random = own.random;
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        own.random = random;
        const uint32_t start = random % m_ThreadCount;
        for (uint32_t i = 0; i < m_ThreadCount; ++i)
        {
            uint32_t victim = start + i;
            if (victim >= m_ThreadCount)
                victim -= m_ThreadCount;
            // An unfenced look first: an idle scan then costs no fence per empty deque.
            if (victim == self || !m_Deques[victim].MayHaveWork())
                continue;
            for (;;)
            {
                const WorkDeque::StealResult result = m_Deques[victim].Steal(ticket);
                if (result == WorkDeque::StealResult::Taken)
                    return true;
                if (result == WorkDeque::StealResult::Empty)
                    break;
            }
        }
        return false;
    }

    bool JobSystemImpl::MayHaveWork() const noexcept
    {
        if (m_Shared->injection.MayHaveWork())
            return true;
        for (uint32_t i = 0; i < m_ThreadCount; ++i)
        {
            if (m_Deques[i].MayHaveWork())
                return true;
        }
        return false;
    }

    void JobSystemImpl::WaitFor(uint32_t index, uint32_t generation, uint32_t self) noexcept
    {
        EventCount& events = m_Shared->events;
        uint32_t idle = 0;
        uint32_t retries = 0;
        while (!IsFinished(index, generation))
        {
            uint32_t ticket;
            bool fromShared;
            if (FindWork(self, ticket, fromShared))
            {
                RunTicket(ticket, self, fromShared);
                idle = 0;
                retries = 0;
                continue;
            }
            if (idle < m_SpinCount)
            {
                Relax(idle++);
                continue;
            }
            // A slot that has moved on to another job only gets a spurious notification.
            m_Jobs[index].hasWaiters.store(1, std::memory_order_relaxed);
            const uint32_t key = events.PrepareWait();
            if (IsFinished(index, generation))
            {
                events.CancelWait();
                return;
            }
            if (MayHaveWork())
            {
                events.CancelWait();
                Relax(retries++);   // as in WorkerMain
                continue;
            }
            events.Wait(key);
            idle = 0;
            retries = 0;
        }
    }

    void JobSystemImpl::Wait(JobHandle job) noexcept
    {
        const uint64_t handle = uint64_t(job);
        const uint32_t index = uint32_t(handle);
        const uint32_t generation = uint32_t(handle >> 32);
        if (handle == 0 || index >= c_JobCapacity || IsFinished(index, generation))
            return;
        // The job has a range running lower on this thread's stack: it cannot finish while this call waits, so the
        // call returns without waiting (JobSystem.h). A walk of the thread's own short list.
        for (const RunningFrame* frame = t_RunningJobs; frame; frame = frame->previous)
        {
            if (frame->system == this && frame->index == index && frame->generation == generation)
            {
#ifndef NDEBUG
                const char* name = m_Jobs[index].debugName;
                log::warning("IJobSystem '%s': Wait for job '%s', which has a range running lower on the waiting "
                             "thread's stack, returns without waiting (it could never finish)", m_Name,
                             name ? name : "");
#endif
                return;
            }
        }
        SlotScope scope(this, index, generation);
        if (scope.Index() == c_None)
            return;   // it finished while this thread waited for an external slot
        WaitFor(index, generation, scope.Index());
    }

    bool JobSystemImpl::IsComplete(JobHandle job) noexcept
    {
        const uint64_t handle = uint64_t(job);
        const uint32_t index = uint32_t(handle);
        return handle == 0 || index >= c_JobCapacity || IsFinished(index, uint32_t(handle >> 32));
    }

    void JobSystemImpl::ParallelFor(uint32_t count, uint32_t grainSize, PFN_JobFunction function,
                                    void* context) noexcept
    {
        if (count == 0)
            return;
        if (!function)
        {
            log::warning("IJobSystem::ParallelFor: %u indices and no function", count);
            return;
        }
        const uint32_t grain = grainSize ? grainSize : count;
        const uint32_t rangeCount = RangeCount(count, grain);
        SlotScope scope(this);
        const uint32_t self = scope.Index();
        if (rangeCount > 1)
        {
            const JobDesc job = { function, context, count, grain, nullptr };
            JobHandle handle = JobHandle::Null;
            if (NVRHI_SUCCEEDED(ScheduleInternal(job, nullptr, 0, handle, self)))
            {
                WaitFor(uint32_t(uint64_t(handle)), uint32_t(uint64_t(handle) >> 32), self);
                return;
            }
            if (!m_Shared->warnedFull.exchange(true, std::memory_order_relaxed))
                log::warning("IJobSystem '%s': the job table is full; ParallelFor runs on the calling thread", m_Name);
        }
        // One range, or a full job table: the calling thread runs the ranges in order.
        for (uint32_t range = 0; range < rangeCount; ++range)
        {
            const uint32_t begin = range * grain;
            function(context, begin, begin + std::min(grain, count - begin), self);
        }
    }

    void JobSystemImpl::Drain() noexcept
    {
        Shared& shared = *m_Shared;
        if (shared.liveJobs.load(std::memory_order_acquire) == 0)
            return;
        SlotScope scope(this);
        const uint32_t self = scope.Index();
        for (uint32_t idle = 0; shared.liveJobs.load(std::memory_order_acquire) != 0;)
        {
            uint32_t ticket;
            bool fromShared;
            if (FindWork(self, ticket, fromShared))
            {
                RunTicket(ticket, self, fromShared);
                idle = 0;
            }
            else if (idle < m_SpinCount)
                Relax(idle++);
            else
                std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void JobSystemImpl::WorkerMain(uint32_t self) noexcept
    {
        NameThread(m_Name, self - m_ExternalCount);
        ThreadFrame frame{ this, self, t_ThreadFrames };
        t_ThreadFrames = &frame;
        Shared& shared = *m_Shared;
        shared.runningWorkers.fetch_add(1, std::memory_order_release);
        uint32_t idle = 0;                                 // the spin's rounds since the last work or sleep,
        std::chrono::steady_clock::time_point spinEnd{};   // when their time is up (set by the first of them),
        uint32_t retries = 0;                              // and the rounds after it that did not sleep either
        for (;;)
        {
            uint32_t ticket;
            bool fromShared;
            if (FindWork(self, ticket, fromShared))
            {
                RunTicket(ticket, self, fromShared);
                idle = 0;
                retries = 0;
                continue;
            }
            const uint32_t stop = shared.stop.load(std::memory_order_acquire);
            if (stop != c_StopNone)
            {
                // The look above may have come before a ticket that was published just before the stop (a job
                // scheduled right before exit). This acquire load makes every such ticket visible, so a worker that is
                // to leave looks once more. Without that look, when every worker missed the ticket (likely with one or
                // two workers), they all left and the job never ran (T7-3).
                if (stop == c_StopLeave && FindWork(self, ticket, fromShared))
                {
                    RunTicket(ticket, self, fromShared);
                    idle = 0;
                    retries = 0;
                    continue;
                }
                uint32_t expected = c_WorkerRunning;
                if (stop == c_StopLeave &&
                    m_WorkerStates[self - m_ExternalCount].compare_exchange_strong(expected, c_WorkerLeaving,
                                                                                   std::memory_order_acq_rel))
                    break;
                Park();   // StopWorkers parked this worker, or parks them all
            }
            if (m_IdleSpin.count() != 0)
            {
                // Reading the clock touches no C runtime state of the thread (Initialize).
                const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                if (idle == 0)
                    spinEnd = now + m_IdleSpin;
                if (now < spinEnd)
                {
                    Relax(idle++);
                    continue;
                }
            }
            const uint32_t key = shared.events.PrepareWait();
            if (MayHaveWork() || shared.stop.load(std::memory_order_relaxed) != c_StopNone)
            {
                // A ticket came after FindWork's look; the next look usually takes it, or finds it taken. But a thread
                // preempted between claiming a cell of the shared queue and publishing its ticket there leaves a queue
                // that looks full and yields nothing, and a loop without a pause could hold the core it needs. The
                // pause delays that next look as any look of the spin is delayed: 8 pause instructions in the first 64
                // such rounds, then a yield, which gives way only to a thread that is ready to run; a sleeping worker's
                // wake-up costs far more.
                shared.events.CancelWait();
                Relax(retries++);
                continue;
            }
            shared.events.Wait(key);
            idle = 0;
            retries = 0;
        }
        t_ThreadFrames = frame.previous;
    }
}

namespace donut::engine
{
    nvrhi::FRESULT CreateJobSystem(const JobSystemDesc& desc, IJobSystem** ppJobSystem) noexcept
    {
        if (!ppJobSystem)
            return nvrhi::FE_INVALID_ARGS;
        *ppJobSystem = nullptr;
        if (desc.maxExternalThreads == 0 || desc.maxExternalThreads > jobs::c_MaxExternalThreads ||
            desc.workerThreadCount > jobs::c_MaxWorkerThreads)
        {
            log::warning("CreateJobSystem: %u workers and %u external threads (1 to %u external threads, at most %u "
                         "workers)", desc.workerThreadCount, desc.maxExternalThreads, jobs::c_MaxExternalThreads,
                         jobs::c_MaxWorkerThreads);
            return nvrhi::FE_INVALID_ARGS;
        }
        jobs::JobSystemImpl* system = nullptr;
        try
        {
            system = MAKE_RC_OBJ(jobs::JobSystemImpl);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        if (!system)
            return nvrhi::FE_OUT_OF_MEMORY;
        const nvrhi::FRESULT result = system->Initialize(desc);
        if (NVRHI_FAILED(result))
        {
            system->Release();
            return result;
        }
        *ppJobSystem = system;
        return nvrhi::FS_OK;
    }

    namespace
    {
        // Set once the shared system exists; read by the exit handler. Trivial types: nothing to destroy at exit.
        std::atomic<jobs::JobSystemImpl*> g_SharedJobSystem{ nullptr };
        std::atomic<uint32_t> g_SharedExitResult{ uint32_t(SharedJobSystemExit::NotRun) };
        std::atomic<uint32_t> g_SharedExitTimeoutMs{ 5000 };
        std::atomic<bool> g_SharedExitStarted{ false };

        void StopSharedWorkersAtExit()
        {
            jobs::JobSystemImpl* system = g_SharedJobSystem.load(std::memory_order_acquire);
            if (!system || g_SharedExitStarted.exchange(true))
                return;
            const uint32_t timeoutMs = g_SharedExitTimeoutMs.load(std::memory_order_relaxed);
            uint32_t parked = 0;
            SharedJobSystemExit result = SharedJobSystemExit::Joined;
            if (!system->StopWorkers(timeoutMs, parked))
            {
                result = system->GetCurrentWorkerIndex() != ~0u ? SharedJobSystemExit::ParkedFromInside
                                                                : SharedJobSystemExit::Parked;
                if (result == SharedJobSystemExit::Parked)
                    log::warning("SceneGraph: %u workers of the shared job system still ran jobs %u ms into the "
                                 "process exit; they are left parked", parked, timeoutMs);
            }
            g_SharedExitResult.store(uint32_t(result), std::memory_order_release);
        }

        // The atexit handlers of a DLL run in its DLL_PROCESS_DETACH, where joining threads is not safe.
        bool IsLinkedIntoExecutable() noexcept
        {
#if defined(_WIN32)
            HMODULE module = nullptr;
            const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
            if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(reinterpret_cast<const void*>(
                                               &StopSharedWorkersAtExit)), &module))
                return false;
            return module == GetModuleHandleW(nullptr);
#else
            return true;
#endif
        }
    }

    uint32_t GetSharedJobWorkersOverride() noexcept
    {
        char* value = nullptr;
#if defined(_MSC_VER)
        size_t length = 0;
        if (_dupenv_s(&value, &length, "DONUT_JOB_WORKERS") != 0 || !value)
            return 0;
#else
        const char* found = std::getenv("DONUT_JOB_WORKERS");
        if (!found)
            return 0;
        value = const_cast<char*>(found);
#endif
        char* end = nullptr;
        const unsigned long workers = std::strtoul(value, &end, 10);
        const bool valid = end != value && *end == 0 && workers >= 1 && workers <= jobs::c_MaxWorkerThreads;
        if (!valid)
            log::warning("DONUT_JOB_WORKERS='%s' is not a worker count from 1 to %u; the shared job system takes its "
                         "default", value, jobs::c_MaxWorkerThreads);
#if defined(_MSC_VER)
        std::free(value);
#endif
        return valid ? uint32_t(workers) : 0u;
    }

    IJobSystem* GetSharedJobSystem() noexcept
    {
        static IJobSystem* const s_Jobs = []() noexcept
        {
            JobSystemDesc desc;
            desc.debugName = "SceneGraph";
            desc.maxExternalThreads = 64;   // every thread that may refresh a graph at the same time
            desc.workerThreadCount = GetSharedJobWorkersOverride();   // T7-2: 0 (unset) is the default
            IJobSystem* system = nullptr;
            if (NVRHI_FAILED(CreateJobSystem(desc, &system)))
            {
                log::warning("SceneGraph: cannot create the shared job system");
                return system;
            }
            // Registered after the system is complete, so the handlers run before the C runtime's teardown and the
            // destructors of the statics constructed before this one (SharedJobSystem.h).
            g_SharedJobSystem.store(static_cast<jobs::JobSystemImpl*>(system), std::memory_order_release);
            if (IsLinkedIntoExecutable() &&
                (std::atexit(&StopSharedWorkersAtExit) != 0 || std::at_quick_exit(&StopSharedWorkersAtExit) != 0))
                log::warning("SceneGraph: cannot register the exit handlers of the shared job system");
            return system;
        }();
        return s_Jobs;
    }

    SharedJobSystemExit GetSharedJobSystemExitResult() noexcept
    {
        return SharedJobSystemExit(g_SharedExitResult.load(std::memory_order_acquire));
    }

    void SetSharedJobSystemExitTimeout(uint32_t milliseconds) noexcept
    {
        g_SharedExitTimeoutMs.store(milliseconds, std::memory_order_relaxed);
    }
}
