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

// For the tests of the extraction: a job system that counts what is handed to the one below it, and the combinations of
// serial and parallel steps that the limits of RenderVisibility.h choose between.

#pragma once

#include <donut/engine/JobSystem.h>
#include <nvrhi/core/foundation.h>

#include "RenderVisibility.h"   // SetSerialLimits

#include <atomic>
#include <cstdint>

namespace donut::tests
{
    class CountingJobSystem final : public nvrhi::ObjectImpl<donut::engine::IJobSystem>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(CountingJobSystem)
        NVRHI_IMPLEMENTS_INTERFACE(donut::engine::IJobSystem)
        NVRHI_END_INTERFACE_TABLE()

        explicit CountingJobSystem(donut::engine::IJobSystem* inner) : m_Inner(inner) { }

        uint32_t GetThreadCount() noexcept override { return m_Inner->GetThreadCount(); }
        uint32_t GetCurrentWorkerIndex() noexcept override { return m_Inner->GetCurrentWorkerIndex(); }
        nvrhi::FRESULT Schedule(const donut::engine::JobDesc& job, const donut::engine::JobHandle* dependencies,
                                uint32_t dependencyCount, donut::engine::JobHandle& outJob) noexcept override
        {
            calls.fetch_add(1);
            return m_Inner->Schedule(job, dependencies, dependencyCount, outJob);
        }
        void Wait(donut::engine::JobHandle job) noexcept override { m_Inner->Wait(job); }
        bool IsComplete(donut::engine::JobHandle job) noexcept override { return m_Inner->IsComplete(job); }
        void ParallelFor(uint32_t count, uint32_t grainSize, donut::engine::PFN_JobFunction function,
                         void* context) noexcept override
        {
            calls.fetch_add(1);
            m_Inner->ParallelFor(count, grainSize, function, context);
        }

        std::atomic<uint32_t> calls{ 0 };   // of Schedule and ParallelFor

    private:
        nvrhi::AutoPtr<donut::engine::IJobSystem> m_Inner;
    };

    // Which steps of an extraction run on the job system: the ranges, and the copy and the sort (one decision).
    struct ExtractionMode
    {
        bool parallelRanges;
        bool parallelRecords;

        // With at least two ranges and two segments, and segments below RadixSorter::c_ParallelThreshold, so that the
        // sort does not go parallel inside: what the extraction hands to the job system (the ranges; the copy, the sort).
        // The gather goes parallel on its own from 8,192 packets.
        uint32_t Calls(uint32_t packets) const
        {
            return (parallelRanges ? 1u : 0u) + (parallelRecords ? 2u : 0u) + (packets > 8192 ? 1u : 0u);
        }
        void Set() const
        {
            donut::render::visibility::InstanceVisibility::SetSerialLimits(parallelRanges ? 0 : UINT32_MAX,
                parallelRecords ? 0 : UINT32_MAX, parallelRecords ? 0 : UINT32_MAX);
        }
        static void Reset()
        {
            using Visibility = donut::render::visibility::InstanceVisibility;
            Visibility::SetSerialLimits(Visibility::c_SerialRowTests, Visibility::c_SerialRecords,
                                        Visibility::c_SerialRecordsAlone);
        }
    };
}
