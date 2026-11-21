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

// A host task scheduler for ExecuteDesc::parallelFor (design 10.1), for applications without a task system of their
// own: the samples, the render graph's tests and rdg_bench. Fixed worker threads, and the calling thread works too.
// A call allocates nothing. Workers spin briefly before they sleep, so that a frame's recording does not wait for a
// thread to wake.
//
//     RenderGraphWorkerPool pool(3);                      // three workers and the caller
//     execute.parallelFor = &RenderGraphWorkerPool::ParallelFor;
//     execute.schedulerContext = &pool;
//     execute.workerCount = pool.ThreadCount();

#pragma once

#include <donut/render/RenderGraph.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace donut::render
{
    class RenderGraphWorkerPool
    {
    public:
        // `threads` workers besides the caller.
        explicit RenderGraphWorkerPool(uint32_t threads)
        {
            for (uint32_t i = 0; i < threads; ++i)
                m_Threads.emplace_back([this] { Work(); });
        }

        ~RenderGraphWorkerPool()
        {
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Quit = true;
                m_Generation.fetch_add(1, std::memory_order_release);
            }
            m_Wake.notify_all();
            for (std::thread& thread : m_Threads)
                thread.join();
        }

        RenderGraphWorkerPool(const RenderGraphWorkerPool&) = delete;
        RenderGraphWorkerPool& operator=(const RenderGraphWorkerPool&) = delete;

        // The threads that run tasks: the workers and the caller.
        uint32_t ThreadCount() const { return uint32_t(m_Threads.size()) + 1; }

        // A ParallelForFunction; `pool` is the RenderGraphWorkerPool.
        static void ParallelFor(void* pool, uint32_t count, TaskFunction task, void* taskContext)
        {
            static_cast<RenderGraphWorkerPool*>(pool)->Run(count, task, taskContext);
        }

        void Run(uint32_t count, TaskFunction task, void* taskContext)
        {
            if (m_Threads.empty() || count < 2)
            {
                for (uint32_t i = 0; i < count; ++i)
                    task(taskContext, i);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Task = task;
                m_Context = taskContext;
                m_Count = count;
                m_Next.store(0, std::memory_order_relaxed);
                m_Active.store(uint32_t(m_Threads.size()), std::memory_order_relaxed);
                m_Generation.fetch_add(1, std::memory_order_release);
            }
            m_Wake.notify_all();
            Drain();
            // The workers finish the tasks they took.
            while (m_Active.load(std::memory_order_acquire) != 0)
                std::this_thread::yield();
        }

    private:
        void Drain()
        {
            for (;;)
            {
                const uint32_t index = m_Next.fetch_add(1, std::memory_order_relaxed);
                if (index >= m_Count)
                    return;
                m_Task(m_Context, index);
            }
        }

        void Work()
        {
            uint64_t seen = 0;
            for (;;)
            {
                // Spin a little: the next frame's call usually comes soon.
                uint64_t generation = m_Generation.load(std::memory_order_acquire);
                for (int spin = 0; generation == seen && spin < 4096; ++spin)
                {
                    std::this_thread::yield();
                    generation = m_Generation.load(std::memory_order_acquire);
                }
                if (generation == seen)
                {
                    std::unique_lock<std::mutex> lock(m_Mutex);
                    m_Wake.wait(lock, [&] { return m_Generation.load(std::memory_order_acquire) != seen; });
                    generation = m_Generation.load(std::memory_order_acquire);
                }
                seen = generation;
                {
                    std::lock_guard<std::mutex> lock(m_Mutex);   // the job's fields, published under the lock
                    if (m_Quit)
                        return;
                }
                Drain();
                m_Active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }

        std::vector<std::thread> m_Threads;
        std::mutex m_Mutex;
        std::condition_variable m_Wake;
        std::atomic<uint64_t> m_Generation{ 0 };
        std::atomic<uint32_t> m_Next{ 0 };
        std::atomic<uint32_t> m_Active{ 0 };
        TaskFunction m_Task = nullptr;
        void* m_Context = nullptr;
        uint32_t m_Count = 0;
        bool m_Quit = false;
    };
}
