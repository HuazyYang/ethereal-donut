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

// The process-wide switches of plan step E7 (sub-plan revision 2, parts I4 and I5; stage T1), set from the environment
// before any test runs. Linked into the device-free tests and smoke benches that build scene graphs, run queries or
// extract (tests/CMakeLists.txt, donut_tests_e7_switches); without the variable it does nothing, so the tests run as
// before, every switch off.
//
//   DONUT_E7_SWITCHES=all                 every switch on
//   DONUT_E7_SWITCHES=rowtracked,static   a list of: rowtracked (SceneGraph::SetRowTrackedTransforms, part I4),
//                                         prefetch (engine::SetQueryChunkPrefetch, I5 remedy 1),
//                                         compact (RenderService::SetCompactPacketRecords, remedy 2),
//                                         static (SceneGraph::SetStaticTag, remedy 3),
//                                         childless (SceneGraph::SetChildlessBoundsPath, remedy 4)
//   DONUT_E7_SWITCHES=off (or unset)      nothing
//
// ctest runs each listed test a second time as `<test>_switches_on` with DONUT_E7_SWITCHES=all. The switches are
// relaxed atomics with constant initialisers, so setting them from a dynamic initialiser is safe. The I6 Defer mode is
// per render-read scope, not a process switch; it is not set here. A test that toggles a switch for its own A/B sets
// both sides explicitly and puts back the value it found (the switches-on runs depend on it).

#include <donut/engine/EcsQuery.h>
#include <donut/engine/SceneGraph.h>
#include <donut/render/RenderExtract.h>
#include <donut/tests/e7_switches.h>

#include <cstdio>
#include <string>

namespace
{
    struct E7SwitchesFromEnvironment
    {
        E7SwitchesFromEnvironment()
        {
            const std::string list = donut::tests::E7SwitchesVariable();
            if (list.empty())
                return;
            std::string on;
            auto set = [&](const char* name, void (*setter)(bool) noexcept)
            {
                if (!donut::tests::E7SwitchRequested(name))
                    return;
                setter(true);
                on += on.empty() ? std::string(name) : std::string(",") + name;
            };
            set("rowtracked", &donut::engine::SceneGraph::SetRowTrackedTransforms);
            set("prefetch", &donut::engine::SetQueryChunkPrefetch);
            set("compact", &donut::render::RenderService::SetCompactPacketRecords);
            set("static", &donut::engine::SceneGraph::SetStaticTag);
            set("childless", &donut::engine::SceneGraph::SetChildlessBoundsPath);
            std::fprintf(stderr, "[e7-switches] DONUT_E7_SWITCHES=%s: on: %s\n", list.c_str(),
                on.empty() ? "none" : on.c_str());
        }
    };

    const E7SwitchesFromEnvironment g_E7SwitchesFromEnvironment;
}
