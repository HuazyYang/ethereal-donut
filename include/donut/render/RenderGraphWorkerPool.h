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
// own: the samples, the render graph's tests and rdg_bench. It owns a job system (donut/engine/JobSystem.h; aggregate
// docs/ecs-design.md, D10) with `threads` workers and runs the graph's tasks on it through JobSystemParallelFor; the
// calling thread works too. A call allocates nothing. An idle worker looks for work for idleSpinMicroseconds before it
// sleeps (JobSystemDesc): with the default of 30 us the workers sleep between the frames of an application, which costs
// its recording a wake-up (async_compute_rdg, 4 threads: 10 and 12 us against a spin of 480 us, aggregate plan, "E5.8
// fix") and takes 28 to 35 % less CPU time, with a higher GPU clock and a frame 1 to 2 % shorter. An application that
// has an IJobSystem already sets ExecuteDesc::parallelFor = &donut::engine::JobSystemParallelFor and schedulerContext to
// it instead.
//
//     RenderGraphWorkerPool pool(3);                      // three workers and the caller
//     execute.parallelFor = &RenderGraphWorkerPool::ParallelFor;
//     execute.schedulerContext = &pool;
//     execute.workerCount = pool.ThreadCount();

#pragma once

#include <donut/engine/JobSystem.h>
#include <donut/render/RenderGraph.h>

#include <cstdint>

namespace donut::render
{
    class RenderGraphWorkerPool
    {
    public:
        // `threads` workers besides the caller. With none, or if the job system cannot start, the caller records alone.
        explicit RenderGraphWorkerPool(uint32_t threads, uint32_t idleSpinMicroseconds =
                                                             donut::engine::JobSystemDesc{}.idleSpinMicroseconds)
        {
            if (threads == 0)
                return;
            donut::engine::JobSystemDesc desc;
            desc.debugName = "RenderGraphWorkerPool";
            desc.workerThreadCount = threads;
            desc.idleSpinMicroseconds = idleSpinMicroseconds;
            if (donut::engine::CreateJobSystem(desc, &m_JobSystem) == nvrhi::FS_OK)
                m_Workers = threads;
        }

        ~RenderGraphWorkerPool()
        {
            if (m_JobSystem)
                m_JobSystem->Release();
        }

        RenderGraphWorkerPool(const RenderGraphWorkerPool&) = delete;
        RenderGraphWorkerPool& operator=(const RenderGraphWorkerPool&) = delete;

        // The threads that run tasks: the workers and the caller.
        uint32_t ThreadCount() const { return m_Workers + 1; }

        // A ParallelForFunction; `pool` is the RenderGraphWorkerPool.
        static void ParallelFor(void* pool, uint32_t count, TaskFunction task, void* taskContext)
        {
            static_cast<RenderGraphWorkerPool*>(pool)->Run(count, task, taskContext);
        }

        void Run(uint32_t count, TaskFunction task, void* taskContext)
        {
            if (!m_JobSystem || count < 2)
            {
                for (uint32_t i = 0; i < count; ++i)
                    task(taskContext, i);
                return;
            }
            donut::engine::JobSystemParallelFor(m_JobSystem, count, task, taskContext);
        }

    private:
        donut::engine::IJobSystem* m_JobSystem = nullptr;
        uint32_t m_Workers = 0;
    };
}
