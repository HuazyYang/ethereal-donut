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

// The shared job system at process exit (SharedJobSystem.h; plan Known issues). A process that returned from main
// while the shared system's workers were still starting crashed in Debug builds: exit() uninitialized the static C
// runtime (its locks, the per-thread data that thread start-up reads) before ExitProcess ended the workers that were
// running its code. The parent runs this executable as a child many times, in each mode, and requires a clean exit
// every time (before the fix, every `start`, `in_flight` and `abrupt` child crashed with 0xC0000005 here):
//
// - start: the shared system is created and main returns at once.
// - in_flight: main returns with a job of 256 ranges scheduled and not waited for; each range allocates (the debug
//   heap's lock). The job must have finished before the later exit handlers run.
// - graph: a scene graph refreshed on the shared system, with a chain 80 nodes deep (the deep bucket's walk); after
//   the workers stop, an exit handler moves two nodes and refreshes again (on the main thread, whose thread_locals
//   exit has destroyed), and a static graph is destroyed after that.
// - abrupt: std::_Exit right after the creation (no exit handler runs; the workers must be past start-up).
// - quick_exit: std::quick_exit with a job in flight (the at_quick_exit handlers).
// - touch_static: a job builds a function-local static with a destructor while exit runs the atexit handlers (its
//   registration blocks on the C runtime's exit lock): the exit must not hang.
// - stuck: a job that never returns: the exit handler parks its worker after the time limit (300 ms here).
// - exit_inside: exit() called from inside a ParallelFor range: every worker is parked, none leaves during teardown.
//
// In the modes with handlers, one registered before the shared system was created runs after the system's own and
// checks what it did (GetSharedJobSystemExitResult); after a join it also runs a ParallelFor on the stopped system
// (every index on the calling thread). A child reports a failure through its exit code. On Windows the children run
// in a job object that ends them with the parent, a child that has not ended after 30 s is killed, and a mode stops
// after its first hang. Elsewhere std::system waits without a limit: a hang there ends at ctest's TIMEOUT.
// Usage: test_job_system_exit [mode [runs]] (the parent), test_job_system_exit --child mode.

#include <donut/core/log.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>
#include <nvrhi/core/autoptr.h>

#include "SharedJobSystem.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace donut;
using namespace donut::engine;

namespace
{
    // Child exit codes.
    constexpr int c_NotStopped = 10;       // the exit handler did not do what the mode expects
    constexpr int c_NotUsable = 11;        // a ParallelFor on the stopped system missed an index or used another thread
    constexpr int c_JobUnfinished = 12;    // the job in flight had not finished when the workers stopped
    constexpr int c_GraphFailed = 13;      // the refresh after the stop did not move the nodes
    constexpr int c_SetupFailed = 14;

    constexpr uint32_t c_InFlightRanges = 256;
    constexpr uint32_t c_ChainDepth = 80;
    constexpr uint32_t c_ChildTimeoutMs = 30000;

    std::atomic<uint32_t> g_RangesDone{ 0 };
    std::atomic<uint32_t> g_UsableIndices{ 0 };
    std::atomic<bool> g_UsableOnOtherThread{ false };
    std::atomic<bool> g_Exiting{ false };
    std::atomic<bool> g_Never{ false };
    std::thread::id g_ExitThread;
    SharedJobSystemExit g_Expected = SharedJobSystemExit::Joined;

    // Constructed before main, so destroyed after every atexit handler that main and the shared system register.
    nvrhi::AutoPtr<SceneGraph> g_Graph;
    nvrhi::AutoPtr<SceneGraphNode> g_Moved;
    nvrhi::AutoPtr<SceneGraphNode> g_ChainRoot;
    nvrhi::AutoPtr<SceneGraphNode> g_ChainTail;
#if !defined(_WIN32)
    const char* g_Argv0 = nullptr;
#endif

    void Allocating(void*, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        for (uint32_t i = begin; i < end; ++i)
        {
            for (int n = 0; n < 200; ++n)
            {
                std::string text(64 + n, 'x');   // the heap, and its lock in Debug builds
                if (text.size() == 0)
                    std::abort();
            }
            g_RangesDone.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void Usable(void*, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        if (std::this_thread::get_id() != g_ExitThread)
            g_UsableOnOtherThread.store(true);
        g_UsableIndices.fetch_add(end - begin);
    }

    // Registered before the shared system exists, so it runs after the system's exit handler.
    void CheckAfterStop()
    {
        if (GetSharedJobSystemExitResult() != g_Expected)
            std::_Exit(c_NotStopped);
        if (g_Expected != SharedJobSystemExit::Joined)
            return;
        g_ExitThread = std::this_thread::get_id();
        GetSharedJobSystem()->ParallelFor(64, 1, &Usable, nullptr);
        if (g_UsableIndices.load() != 64 || g_UsableOnOtherThread.load())
            std::_Exit(c_NotUsable);
    }

    void CheckJobFinished()
    {
        if (g_RangesDone.load() != c_InFlightRanges)
            std::_Exit(c_JobUnfinished);
    }

    void RefreshAfterStop()
    {
        g_Moved->SetTranslation(dm::double3(7.0, 0.0, 0.0));
        g_ChainRoot->SetTranslation(dm::double3(0.0, 9.0, 0.0));
        g_Graph->Refresh(2);
        if (g_Moved->GetLocalToWorldTransform().m_translation.x != 7.0 ||
            g_ChainTail->GetLocalToWorldTransform().m_translation.y != 9.0 + double(c_ChainDepth - 1))
            std::_Exit(c_GraphFailed);
    }

    // A job that nobody waits for: the process ends while it runs.
    bool ScheduleInFlight(PFN_JobFunction function, uint32_t count)
    {
        IJobSystem* jobs = GetSharedJobSystem();
        const JobDesc job = { function, nullptr, count, 1, "in flight at exit" };
        JobHandle handle = JobHandle::Null;
        return jobs && NVRHI_SUCCEEDED(jobs->Schedule(job, nullptr, 0, handle));
    }

    struct Tracer
    {
        ~Tracer() { g_Never.store(false); }
    };

    // Waits for the exit, then builds a function-local static with a destructor: its registration takes the C
    // runtime's exit lock, which the exiting thread holds while it runs the atexit handlers.
    void TouchStatic(void*, uint32_t, uint32_t, uint32_t) noexcept
    {
        while (!g_Exiting.load())
            std::this_thread::yield();
        static Tracer s_Tracer;
        (void)s_Tracer;
    }

    void Stuck(void*, uint32_t, uint32_t, uint32_t) noexcept
    {
        while (!g_Never.load())
            std::this_thread::yield();
    }

    void ExitInside(void*, uint32_t begin, uint32_t, uint32_t) noexcept
    {
        if (begin == 40)
            std::exit(0);
    }

    // Registered after the shared system, so it runs before the system's exit handler.
    void SignalExiting()
    {
        g_Exiting.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // the job reaches the exit lock
    }

    int RunChild(const char* mode)
    {
        std::atexit(&CheckAfterStop);
        if (std::strcmp(mode, "start") == 0)
            return GetSharedJobSystem() ? 0 : c_SetupFailed;
        if (std::strcmp(mode, "abrupt") == 0)
            std::_Exit(GetSharedJobSystem() ? 0 : c_SetupFailed);
        if (std::strcmp(mode, "quick_exit") == 0)
        {
            std::at_quick_exit(&CheckAfterStop);
            std::at_quick_exit(&CheckJobFinished);
            if (!ScheduleInFlight(&Allocating, c_InFlightRanges))
                return c_SetupFailed;
            std::quick_exit(0);
        }
        if (std::strcmp(mode, "in_flight") == 0)
        {
            std::atexit(&CheckJobFinished);
            return ScheduleInFlight(&Allocating, c_InFlightRanges) ? 0 : c_SetupFailed;
        }
        if (std::strcmp(mode, "touch_static") == 0)
        {
            // The UCRT blocks the registration until the process ends: the worker is parked at the time limit.
            SetSharedJobSystemExitTimeout(300);
            g_Expected = SharedJobSystemExit::Parked;
            if (!ScheduleInFlight(&TouchStatic, 1))
                return c_SetupFailed;
            std::atexit(&SignalExiting);
            return 0;
        }
        if (std::strcmp(mode, "stuck") == 0)
        {
            SetSharedJobSystemExitTimeout(300);
            g_Expected = SharedJobSystemExit::Parked;
            return ScheduleInFlight(&Stuck, 1) ? 0 : c_SetupFailed;
        }
        if (std::strcmp(mode, "exit_inside") == 0)
        {
            g_Expected = SharedJobSystemExit::ParkedFromInside;
            IJobSystem* jobs = GetSharedJobSystem();
            if (!jobs)
                return c_SetupFailed;
            jobs->ParallelFor(64, 1, &ExitInside, nullptr);
            return c_SetupFailed;   // the range at 40 exits
        }
        if (std::strcmp(mode, "graph") == 0)
        {
            std::atexit(&RefreshAfterStop);
            g_Graph = MAKE_RC_OBJ_PTR(SceneGraph);   // creates the shared system
            g_Graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            for (int i = 0; i < 500; ++i)
            {
                auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                node->SetTranslation(dm::double3(double(i), 1.0, 2.0));
                g_Graph->Attach(g_Graph->GetRootNode(), node);
            }
            g_Moved = g_Graph->Attach(g_Graph->GetRootNode(), MAKE_RC_OBJ_PTR(SceneGraphNode));
            g_ChainRoot = g_Graph->Attach(g_Graph->GetRootNode(), MAKE_RC_OBJ_PTR(SceneGraphNode));
            nvrhi::AutoPtr<SceneGraphNode> parent = g_ChainRoot;
            for (uint32_t depth = 1; depth < c_ChainDepth; ++depth)
            {
                auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                node->SetTranslation(dm::double3(0.0, 1.0, 0.0));
                parent = g_Graph->Attach(parent, node);
            }
            g_ChainTail = parent;
            g_Graph->Refresh(0);
            g_Graph->Refresh(1);
            return 0;
        }
        return c_SetupFailed;
    }

#if defined(_WIN32)
    // Ends every child with the parent, whatever ends the parent.
    HANDLE ChildJobObject()
    {
        static const HANDLE s_Job = []()
        {
            HANDLE job = CreateJobObjectW(nullptr, nullptr);
            if (job)
            {
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {};
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
            }
            return job;
        }();
        return s_Job;
    }
#endif

    // Runs this executable with `--child mode`; returns its exit code, -1 when it could not start, -2 when it hung.
    int RunOnce(const char* mode)
    {
#if defined(_WIN32)
        wchar_t path[MAX_PATH];
        if (GetModuleFileNameW(nullptr, path, MAX_PATH) == 0)
            return -1;
        std::wstring command = L"\"" + std::wstring(path) + L"\" --child ";
        for (const char* c = mode; *c; ++c)
            command += wchar_t(*c);
        STARTUPINFOW startup = { sizeof(startup) };
        PROCESS_INFORMATION process = {};
        if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr,
                            &startup, &process))
            return -1;
        if (ChildJobObject())
            AssignProcessToJobObject(ChildJobObject(), process.hProcess);
        ResumeThread(process.hThread);
        CloseHandle(process.hThread);
        DWORD code = DWORD(-1);
        if (WaitForSingleObject(process.hProcess, c_ChildTimeoutMs) != WAIT_OBJECT_0)
        {
            TerminateProcess(process.hProcess, 1);
            WaitForSingleObject(process.hProcess, INFINITE);
            code = DWORD(-2);
        }
        else
            GetExitCodeProcess(process.hProcess, &code);
        CloseHandle(process.hProcess);
        return int(code);
#else
        const std::string command = std::string("\"") + g_Argv0 + "\" --child " + mode;
        const int status = std::system(command.c_str());
        return status == 0 ? 0 : (status > 0 && (status & 0x7f) == 0 ? (status >> 8) & 0xff : -1);
#endif
    }
}

int main(int argc, char** argv)
{
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0)
        return RunChild(argv[2]);
#if !defined(_WIN32)
    g_Argv0 = argv[0];
#endif

    struct Mode { const char* name; int runs; };
    std::vector<Mode> modes = { { "start", 40 }, { "in_flight", 10 }, { "graph", 10 }, { "abrupt", 20 },
                                { "quick_exit", 10 }, { "touch_static", 4 }, { "stuck", 3 }, { "exit_inside", 20 } };
    if (argc >= 2)
        modes = { { argv[1], argc >= 3 ? std::atoi(argv[2]) : 40 } };

    int failures = 0;
    for (const Mode& mode : modes)
    {
        const auto start = std::chrono::steady_clock::now();
        int failed = 0;
        int run = 0;
        for (; run < mode.runs; ++run)
        {
            const int code = RunOnce(mode.name);
            if (code == -2)
            {
                ++failed;
                std::fprintf(stderr, "test_job_system_exit: '%s' run %d hangs at exit; the mode stops here\n",
                             mode.name, run);
                ++run;
                break;
            }
            if (code != 0)
            {
                ++failed;
                std::fprintf(stderr, "test_job_system_exit: '%s' run %d exited with 0x%08x\n", mode.name, run,
                             unsigned(code));
            }
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("test_job_system_exit: %s %d/%d clean (%lld ms)\n", mode.name, run - failed, mode.runs,
                    (long long)ms);
        failures += failed + (mode.runs - run);
    }
    return failures == 0 ? 0 : 1;
}
