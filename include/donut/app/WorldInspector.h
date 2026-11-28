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

// The world inspector (aggregate docs/ecs-design.md 11.4; plan step E7, item E7.7): ImGui panels over IWorld and
// IScheduler and their reflection. World totals (WorldStats); archetypes with their components by name, rows, chunks
// and the occupancy of each chunk; systems by phase with flags, enabled bits and timings (SystemStats); change versions
// per chunk and column; an entity browser paged by chunk, which reads only the page it shows, with values formatted
// from FieldDesc into fixed buffers of the inspector.
//
// Editing: fields of Relocatable, serialisable components (not NoSerialize) that are on neither donut's read-only list
// nor the application's (SetReadOnly). Donut's list is the world resource InspectorReadOnlyComponents that every
// scene graph's store registers in its world (SceneComponents.h): the hierarchy, derived, slot, reference and hooked
// columns and the simulation's SimPosePrevious, SimTeleport, SimPose and SimScale, whose free edit corrupts the graph,
// bypasses its setters or never reaches the domain's engine; and, on a simulation-driven row only, LocalTransform,
// which the graph ignores there (the list's per-row pairs). It applies to every inspector, not by the application's
// choice. Fields of type Opaque or Entity are shown, never edited. A commit keeps the whole value as the pending one of
// its (entity, component), at most 64 of them; Apply records one ICommandBuffer::SetComponents per pending value into
// the inspector's buffer and plays it back.
//
// Module-private C++ of donut_app, not on the ABI surface. It names no scene type: entity names come from the
// application (SetEntityNames). Between calls the inspector holds no scheduler, and a world only through pending edits:
// the command buffer of the edits comes from IWorld::CreateCommandBuffer at the first edit and holds a reference to
// its world until Apply plays it back, the edits are dropped, or the inspector is destroyed. Nothing allocates while
// its panels are closed and nothing is edited. Otherwise ImGui does, and so do the first edit after an Apply (the
// command buffer), Apply (the commands), and the Systems panel when the scheduler's order is built again
// (IScheduler::GetSystemInfo, after a change of its systems).

#pragma once

#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>

#include <cstdint>
#include <memory>

namespace donut::app
{
    class WorldInspector
    {
    public:
        WorldInspector();
        ~WorldInspector();
        WorldInspector(const WorldInspector&) = delete;
        WorldInspector& operator=(const WorldInspector&) = delete;

        // Draws the inspector's window into the current ImGui frame (ImGui::NewFrame has run; ImGuiLayer::buildUI is
        // the place). Outside phases: inside one (IWorld::IsInPhase), a warning and nothing drawn. `scheduler` may be
        // null (no systems panel); a scheduler of another world is not listed (a warning at each Draw that shows the
        // window, no systems panel). Pending edits belong to the world of their buffer (which holds it): a Draw of
        // another world drops them, with a warning, before it records new ones.
        void Draw(engine::IWorld* world, engine::IScheduler* scheduler);

        // Plays the pending edits back (IWorld::Playback), then IWorld::AdvanceVersion, so that SceneGraph::Refresh sees
        // a change (it runs when the version moved; writes outside phases stamp the world's version), and releases the
        // buffer and with it the world. Outside phases: FE_OUT_OF_DATE inside one, the edits kept. FS_OK when nothing is
        // pending; otherwise what Playback returns. `world` must be the buffer's world (the one it holds, so a world
        // made again at the same address is not taken for it); edits of another world are dropped with a warning and
        // FE_INVALID_ARGS returned.
        nvrhi::FRESULT Apply(engine::IWorld* world);

        // The application's entity names, for the browser and for entity fields: writes a null-terminated name of at
        // most `capacity` bytes (including the terminator) and returns true, or false for an unnamed entity (shown by
        // its id). Called only from Draw; null clears it. `context` is passed through and must outlive the inspector
        // or the next call.
        using NameFunction = bool (*)(void* context, engine::Entity entity, char* out, uint32_t capacity);
        void SetEntityNames(NameFunction function, void* context);

        // Components that the inspector shows but never edits, beyond donut's read-only list; copied. Replaces the
        // previous list; count 0 clears it.
        void SetReadOnly(const engine::ComponentId* components, uint32_t count);

    private:
        struct State;                     // WorldInspector.cpp (E7.7)
        std::unique_ptr<State> m_State;
    };

    // Donut's read-only list of `world` (the InspectorReadOnlyComponents resource; empty for a world of no scene graph):
    // writes at most `capacity` ids to `out` (null with capacity 0 to ask) and returns how many there are. The list's
    // per-row pairs are not among them: those components stay editable on other rows.
    uint32_t GetDonutReadOnlyComponents(engine::IWorld* world, engine::ComponentId* out, uint32_t capacity) noexcept;
}
