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

// A typed blackboard for composing frames from optional features (design 12.2). Header-only and outside the ABI
// surface: the graph, the compiler and the plan never see it.
//
//     struct SceneTargets { ResourceId depth, color, motionVectors; };
//
//     auto& scene = frame.Create<SceneTargets>(graph);               // the depth feature
//     scene.depth = ...;
//
//     if (SceneTargets* s = frame.Find<SceneTargets>())              // an optional feature, elsewhere
//         frame.Create<AmbientOcclusion>(graph).ao = ...;
//
// Rules:
// - Keyed by C++ type, through the address of an inline per-type tag: no strings, no RTTI. Producers and consumers
//   of one type are built into one module, because an inline variable's address can differ across DLLs.
// - Values are trivially destructible structs of ids, in the graph's frame arena. A lookup is a few pointer
//   compares and allocates nothing.
// - Create<T> is the only call that makes an entry; Find<T> and Get<T> never insert one.
// - A write returns a new version: the feature that writes a resource stores the returned id back in the entry.
// - For composing frames only. donut's passes keep their typed AddToGraph inputs and never read the blackboard.

#pragma once

#include <donut/render/RenderGraph.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace donut::render
{
    namespace detail
    {
        template <typename T>
        struct FrameDataTag
        {
            static inline const char key = 0;
        };
    }

    class FrameData
    {
    public:
        static constexpr uint32_t c_MaxEntries = 32;

        // Right after IRenderGraph::Reset: the entries' memory went with the arena.
        void Clear() noexcept { m_Count = 0; }

        // Once per type per frame, in the graph's arena. A second Create of the same type in one frame asserts
        // and returns the existing entry.
        template <typename T>
        T& Create(IRenderGraph* graph)
        {
            static_assert(std::is_trivially_destructible_v<T>,
                          "frame data lives in the graph's arena, which runs no destructors");
            if (T* existing = Find<T>())
            {
                assert(!"FrameData::Create: this type already has an entry this frame");
                return *existing;
            }
            assert(m_Count < c_MaxEntries && "FrameData: too many entries");
            T* value = NewFrameData<T>(graph);
            assert(value && "FrameData: the frame arena is exhausted");
            if (m_Count < c_MaxEntries)
                m_Entries[m_Count++] = Entry{ &detail::FrameDataTag<T>::key, value };
            return *value;
        }

        // Create without asserting, for callers that handle failure: the entry this type already has this frame, or a
        // new one; nullptr when the arena is exhausted or the blackboard is full.
        template <typename T>
        T* TryCreate(IRenderGraph* graph) noexcept
        {
            static_assert(std::is_trivially_destructible_v<T>,
                          "frame data lives in the graph's arena, which runs no destructors");
            if (T* existing = Find<T>())
                return existing;
            if (m_Count >= c_MaxEntries)
                return nullptr;
            T* value = NewFrameData<T>(graph);
            if (value)
                m_Entries[m_Count++] = Entry{ &detail::FrameDataTag<T>::key, value };
            return value;
        }

        // nullptr if nothing created it this frame.
        template <typename T>
        T* Find() noexcept
        {
            const void* key = &detail::FrameDataTag<T>::key;
            for (uint32_t index = 0; index < m_Count; ++index)
            {
                if (m_Entries[index].key == key)
                    return static_cast<T*>(m_Entries[index].value);
            }
            return nullptr;
        }

        // Asserts that it exists.
        template <typename T>
        T& Get() noexcept
        {
            T* value = Find<T>();
            assert(value && "FrameData::Get: nothing created this type this frame");
            return *value;
        }

    private:
        struct Entry
        {
            const void* key;
            void* value;
        };

        Entry m_Entries[c_MaxEntries] = {};
        uint32_t m_Count = 0;
    };
}
