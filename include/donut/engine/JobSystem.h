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

// The job system (aggregate docs/ecs-design.md 6.4, decision D10): fixed worker threads, work stealing, jobs with
// dependency counters, and ParallelFor with a deterministic split. The scheduler runs systems on it, and the render
// graph's parallel recording can drive it at the same time (design 7.5) through JobSystemParallelFor. It sits next to
// ThreadPool.h, which stays for the importers' asynchronous loads.

#pragma once

#include <nvrhi/core/types.h>
// SAL annotations, as nvrhi.h includes them: Salieri's fallbacks would clash with Windows SDK headers included later.
#if defined(_MSC_VER)
#include <sal.h>
#else
#include <nvrhi/core/salieri.h>
#endif

#include <cstdint>

namespace donut::engine
{
    // Runs the indices [begin, end) of a job. `workerIndex` identifies the calling thread (IJobSystem).
    typedef void (*PFN_JobFunction)(void* context, uint32_t begin, uint32_t end, uint32_t workerIndex);

    // A scheduled job: index in the low 32 bits, generation in the high 32. Null means "no job".
    enum class JobHandle : uint64_t { Null = 0 };

    struct JobDesc
    {
        PFN_JobFunction function;
        void* context;
        uint32_t count;        // indices [0, count)
        uint32_t grainSize;    // indices per task; 0 means count (one task)
        const char* debugName;
    };
    static_assert(sizeof(void*) != 8 || sizeof(JobDesc) == 32, "JobDesc layout");

    struct JobSystemDesc
    {
        const char* debugName = nullptr;
        uint32_t workerThreadCount = 0;    // 0: hardware threads minus one
        uint32_t maxExternalThreads = 2;   // threads that are not workers and drive the system at the same time
        uint32_t spinCount = 4096;         // spins before a worker sleeps
        uint32_t reserved = 0;
    };
    static_assert(sizeof(void*) != 8 || sizeof(JobSystemDesc) == 24, "JobSystemDesc layout");

    // Determinism: a job's indices are cut into ranges of grainSize, the last one shorter. The split depends only on
    // count and grainSize, never on the thread count or timing. Which thread runs a range is not deterministic, so
    // per-range results must be keyed by range (the chunk index of a command's sort key, 6.5).
    //
    // Worker indices span [0, GetThreadCount()): external slots [0, maxExternalThreads), then the workers. A thread that
    // is not a worker takes a free external slot for the duration of a Wait or ParallelFor call, keeps it for nested
    // calls, and waits for a slot when all are taken. Per-worker data (the scheduler's command buffers) is sized by
    // GetThreadCount().
    NVRHI_IID(IJobSystem, "6b260174-35a9-4b73-99fd-d0c3f8fbe2b8")
    struct IJobSystem : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(IJobSystem)

        // External slots plus workers: the range of worker indices.
        virtual uint32_t GetThreadCount() noexcept = 0;
        // The calling thread's worker index, or ~0u on a thread that holds no slot.
        virtual uint32_t GetCurrentWorkerIndex() noexcept = 0;

        // The job starts when every dependency has finished; Null dependencies are ignored. FE_OUT_OF_MEMORY when the
        // job table is full (nothing runs then); `outJob` is Null on failure. Capacity, allocated by CreateJobSystem:
        // 4096 jobs, each held from Schedule until it has finished and no thread holds a ticket of it, and 16384
        // dependencies on jobs that have not finished.
        virtual nvrhi::FRESULT Schedule(const JobDesc& job, _In_reads_(dependencyCount) const JobHandle* dependencies,
                                        uint32_t dependencyCount, JobHandle& outJob) noexcept = 0;
        // Runs tasks, of this job or others, until the job has finished. A finished job's handle is stale and returns
        // at once. Since a waiting thread runs other jobs on its own stack, a job may wait only for jobs it scheduled
        // (directly or through the jobs it scheduled) and for its own ParallelFor: waiting for a running job scheduled
        // elsewhere can deadlock. A Wait for a job with a range running lower on the calling thread's stack, which
        // could never finish, returns at once without waiting (Debug builds log a warning).
        virtual void Wait(JobHandle job) noexcept = 0;
        virtual bool IsComplete(JobHandle job) noexcept = 0;

        // Schedule and Wait in one call, without a handle: the calling thread works too.
        virtual void ParallelFor(uint32_t count, uint32_t grainSize, PFN_JobFunction function,
                                 void* context) noexcept = 0;
    };

    // FE_INVALID_ARGS unless maxExternalThreads is 1 to 64 and workerThreadCount at most 256. The last Release runs the
    // jobs still scheduled, with the releasing thread's help, then stops the workers; it must not come from inside a
    // call or a job of the same system.
    nvrhi::FRESULT CreateJobSystem(const JobSystemDesc& desc, IJobSystem** ppJobSystem) noexcept;

    // An adapter with the shape of the render graph's ParallelForFunction (rdg design 10.1): set
    // ExecuteDesc::parallelFor = &JobSystemParallelFor and schedulerContext = the IJobSystem.
    inline void JobSystemParallelFor(void* jobSystem, uint32_t count, void (*task)(void* taskContext, uint32_t index),
                                     void* taskContext) noexcept
    {
        struct Adapter
        {
            void (*task)(void*, uint32_t);
            void* taskContext;

            static void Run(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                const Adapter* self = static_cast<const Adapter*>(context);
                for (uint32_t i = begin; i < end; ++i)
                    self->task(self->taskContext, i);
            }
        };
        Adapter adapter{ task, taskContext };
        static_cast<IJobSystem*>(jobSystem)->ParallelFor(count, 1, &Adapter::Run, &adapter);
    }
}
