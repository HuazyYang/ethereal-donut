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

#pragma once

// The switches of plan step E7 (sub-plan revision 2, parts I4 and I5) as the environment asks for them (stage T1):
// DONUT_E7_SWITCHES=all, or a comma-separated list of rowtracked, prefetch, compact, static, childless; unset, empty
// or "off" asks for none. tests/src/e7_switches.cpp sets the switches from it before main; a test that checks a
// switch's default compares it with E7SwitchRequested instead of with false.

#include <cstdlib>
#include <cstring>
#include <string>

namespace donut::tests
{
    inline std::string E7SwitchesVariable()
    {
#ifdef _MSC_VER
        char* value = nullptr;
        size_t length = 0;
        if (_dupenv_s(&value, &length, "DONUT_E7_SWITCHES") != 0 || !value)
            return {};
        std::string list(value);
        std::free(value);
        return list;
#else
        const char* value = std::getenv("DONUT_E7_SWITCHES");
        return value ? std::string(value) : std::string();
#endif
    }

    // Whether the environment asks for the switch `name` (rowtracked, prefetch, compact, static, childless).
    inline bool E7SwitchRequested(const char* name)
    {
        const std::string list = E7SwitchesVariable();
        if (list.empty() || list == "off" || list == "0")
            return false;
        if (list == "all" || list == "1")
            return true;
        const size_t nameLength = std::strlen(name);
        size_t start = 0;
        while (start <= list.size())
        {
            size_t end = list.find(',', start);
            if (end == std::string::npos)
                end = list.size();
            if (end - start == nameLength && list.compare(start, nameLength, name) == 0)
                return true;
            start = end + 1;
        }
        return false;
    }
}
