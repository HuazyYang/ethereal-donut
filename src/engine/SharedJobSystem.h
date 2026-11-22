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

// The process-wide job system that scene graphs share (SceneGraph::SetJobSystem; plan step E3, deviation 6), private
// to donut_engine. Implemented in JobSystem.cpp, beside the workers it stops at exit.

#pragma once

#include <donut/engine/JobSystem.h>

namespace donut::engine
{
    // Created on first use (hardware threads minus one workers, 64 external slots); null if it cannot start. It is
    // never released, and in an executable its workers stop at exit:
    //
    // - exit() (a return from main) runs the atexit handlers and static destructors, then uninitializes a static C
    //   runtime (it deletes the CRT's locks and frees every thread's CRT data, including the start-up parameters of a
    //   thread that has not reached its function yet), and only then calls ExitProcess, which ends the other threads.
    //   _Exit and quick_exit uninitialize it as well. A worker that ran C runtime code in that window crashed in Debug
    //   builds (test_job_system_exit; plan Known issues): one still in thread start-up, or a job that allocates.
    // - So CreateJobSystem returns only once every worker is past start-up (an idle worker touches no C runtime
    //   state), and creating the shared system registers an atexit and an at_quick_exit handler that stop and join
    //   its workers: they finish every ticket they can find (the jobs in flight, and the scheduled jobs that become
    //   runnable), then leave. The handlers run before the destructors of the statics constructed before the system,
    //   after those of the statics constructed after it. _Exit runs no handler: a job still running then can crash.
    // - The handler runs under the C runtime's exit lock, which a job blocks on if it registers an exit function then
    //   (a function-local static with a destructor, built for the first time). So it waits for the workers for 5 s at
    //   most (SetSharedJobSystemExitTimeout); the workers still busy are parked: they sleep, once their jobs return,
    //   until the process ends, and never run thread-exit code during the C runtime's teardown.
    // - A function-local static first built inside a job registers its destructor after the shared system's handler,
    //   so it is destroyed before that handler runs, while workers may still be running jobs that use it.
    // - Afterwards the system still works without workers: Wait and ParallelFor run the ranges on the calling thread
    //   (static destructors may still refresh a graph). A job scheduled then runs only when some thread waits for it.
    //   That thread's thread_local objects may be destroyed already (exit destroys the main thread's before the
    //   atexit handlers): jobs guard theirs (SceneGraphStore's scratch and deep-walk stacks).
    // - exit() called from inside a call or a job of the shared system cannot join (a worker cannot join itself, and
    //   the others may wait for ranges on the calling thread's stack): every worker is parked once idle.
    // - Windows: only when donut_engine is linked into the executable. A DLL's exit handlers run in its
    //   DLL_PROCESS_DETACH, under the loader lock, where a join deadlocks (FreeLibrary) or finds the threads already
    //   ended (ExitProcess). So a DLL that links donut_engine and is unloaded still gives its graphs a job system of
    //   its own (SetJobSystem) and releases it before it unloads. Creating a job system under the loader lock (DllMain,
    //   a DLL's static constructors or destructors) is not supported: no worker can start there, and CreateJobSystem
    //   stops waiting for them after 2 s.
    // - Elsewhere the handlers are registered in a shared object too: they run at its dlclose (glibc runs a shared
    //   object's atexit handlers then, through its __dso_handle). A dlclose from inside a job is not supported.
    // - ExitProcess and TerminateProcess called directly uninitialize nothing before the threads end.
    IJobSystem* GetSharedJobSystem() noexcept;

    // Tests: what the exit handler did.
    enum class SharedJobSystemExit : uint32_t
    {
        NotRun,             // no handler has run (yet)
        Joined,             // every worker left and was joined
        Parked,             // some workers were still busy at the time limit, and are parked
        ParkedFromInside,   // the exit came from inside a call or job of the system: every worker is parked
    };
    SharedJobSystemExit GetSharedJobSystemExitResult() noexcept;
    // How long the exit handler waits for busy workers before it parks them; 5000 ms unless a test sets it.
    void SetSharedJobSystemExitTimeout(uint32_t milliseconds) noexcept;
}
