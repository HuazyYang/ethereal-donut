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

// Module-private parts of app::WorldInspector (plan step E7, item E7.7), for its tests: the formatter, and a test seam
// (a headless ImGui context has no mouse that could click the browser or type into a field, so a test stands in for
// the widgets). Not for applications.

#pragma once

#include <donut/engine/EcsTypes.h>

#include <cstdint>

namespace donut::app::detail
{
    // Formats one field of a component value (`value` is the component's first byte) from its FieldDesc into `out`
    // (at most `capacity` bytes with the terminator; cut where longer): scalars; vectors and quaternions as (a, b);
    // matrices as [row; row]; affine transforms as [linear rows] t (translation); boxes as min (...) max (...); arrays
    // as [first 16 scalars, ... (count)]; entity ids as #index:generation or null; opaque bytes as hex (the first 32).
    void FormatInspectorField(char* out, uint32_t capacity, const engine::FieldDesc& field, const void* value) noexcept;

    struct WorldInspectorTestHook
    {
        // When not Null, Draw selects this entity first, as a click in the browser would.
        engine::Entity select = engine::Entity::Null;
        // Draw opens every header and tree node it draws (so that one frame reaches every panel).
        bool openAll = false;
        // Called by Draw for each editable component value of the selected entity (`size` bytes at `value`, the copy
        // the widgets edit). Returning true commits the value as an edit, as a widget would; false leaves it.
        bool (*edit)(void* context, engine::ComponentId component, void* value, uint32_t size) = nullptr;
        // Called by Draw for each scalar of a number field it edits (field `field` of `component`, scalar `scalar`),
        // with the text the field shows (the editor's format) in `text` (`capacity` bytes): returning true takes the
        // text as typed into the field and committed, parsed as the widget parses it (T3-N1). Null: no typing.
        bool (*type)(void* context, engine::ComponentId component, uint32_t field, uint32_t scalar, char* text,
                     uint32_t capacity) = nullptr;
        void* context = nullptr;
    };

    // The hook every inspector's Draw uses; null clears it. Not copied: it must stay valid until cleared. One thread.
    void SetWorldInspectorTestHook(const WorldInspectorTestHook* hook) noexcept;
}
