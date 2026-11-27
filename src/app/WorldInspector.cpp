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

// The world inspector (WorldInspector.h; plan step E7, item E7.7). Every string is formatted into fixed buffers of
// the inspector's state, which the constructor allocates once; pending edits keep their values in a fixed table, so
// that a field edited twice before Apply starts from its pending value. Draw reads only what its open panels show:
// the browser one chunk, the version table the rows the clipper lets through.

#include <donut/app/WorldInspector.h>
#include <donut/core/log.h>
#include <donut/engine/EcsCommandBuffer.h>
#include <donut/engine/SceneComponents.h>   // InspectorReadOnlyComponents: its GUID and layout, no scene type

#include "WorldInspectorDetail.h"

#include <nvrhi/core/autoptr.h>

#include <imgui.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace donut::app
{
    using engine::ArchetypeInfo;
    using engine::BufferHeader;
    using engine::ChunkInfo;
    using engine::ComponentFlags;
    using engine::ComponentId;
    using engine::ComponentTypeDesc;
    using engine::Entity;
    using engine::FieldDesc;
    using engine::FieldShape;
    using engine::FieldType;
    using engine::ICommandBuffer;
    using engine::IScheduler;
    using engine::IWorld;
    using engine::Phase;
    using engine::SystemFlags;
    using engine::SystemInfo;
    using engine::WorldStats;

    namespace
    {
        constexpr uint32_t c_MaxEditBytes = 256;        // larger values are shown, never edited
        constexpr uint32_t c_MaxPendingEdits = 64;      // distinct (entity, component) values between two Apply calls
        constexpr uint32_t c_MaxStripChunks = 4096;     // the occupancy strip draws at most this many chunks
        constexpr uint32_t c_MaxVersionColumns = 32;    // the version table shows at most this many components
        constexpr uint32_t c_MaxArrayScalars = 16;      // an array shows its first scalars
        constexpr uint32_t c_MaxOpaqueBytes = 32;       // an opaque field shows its first bytes

        const detail::WorldInspectorTestHook* g_TestHook = nullptr;

        // Appends to a fixed buffer; output past its end is cut, and the buffer stays null-terminated.
        struct Writer
        {
            char* p;
            char* end;   // one past the last byte, which is kept for the terminator

            Writer(char* buffer, size_t capacity) noexcept : p(buffer), end(buffer + capacity) { *p = 0; }

            void Append(const char* format, ...) noexcept
            {
                if (end - p <= 1)
                    return;
                va_list args;
                va_start(args, format);
                const int written = vsnprintf(p, size_t(end - p), format, args);
                va_end(args);
                if (written > 0)
                    p += std::min<ptrdiff_t>(written, end - p - 1);
            }
        };

        uint32_t ScalarSize(FieldType type) noexcept
        {
            switch (type)
            {
            case FieldType::Boolean: case FieldType::Int8: case FieldType::UInt8: case FieldType::Opaque:
                return 1;
            case FieldType::Int16: case FieldType::UInt16:
                return 2;
            case FieldType::Int32: case FieldType::UInt32: case FieldType::Float:
                return 4;
            default:
                return 8;
            }
        }

        template <typename T> T Load(const uint8_t* source) noexcept
        {
            T value;
            std::memcpy(&value, source, sizeof(T));
            return value;
        }

        void AppendEntityId(Writer& out, Entity entity) noexcept
        {
            if (entity == Entity::Null)
                out.Append("null");
            else
                out.Append("#%u:%u", engine::GetEntityIndex(entity), engine::GetEntityGeneration(entity));
        }

        void AppendScalar(Writer& out, FieldType type, const uint8_t* source) noexcept
        {
            switch (type)
            {
            case FieldType::Boolean: out.Append("%s", source[0] ? "true" : "false"); break;
            case FieldType::Int8: out.Append("%d", int(Load<int8_t>(source))); break;
            case FieldType::UInt8: out.Append("%u", unsigned(Load<uint8_t>(source))); break;
            case FieldType::Int16: out.Append("%d", int(Load<int16_t>(source))); break;
            case FieldType::UInt16: out.Append("%u", unsigned(Load<uint16_t>(source))); break;
            case FieldType::Int32: out.Append("%d", Load<int32_t>(source)); break;
            case FieldType::UInt32: out.Append("%u", Load<uint32_t>(source)); break;
            case FieldType::Int64: out.Append("%lld", (long long)Load<int64_t>(source)); break;
            case FieldType::UInt64: out.Append("%llu", (unsigned long long)Load<uint64_t>(source)); break;
            case FieldType::Float: out.Append("%.6g", double(Load<float>(source))); break;
            case FieldType::Double: out.Append("%.9g", Load<double>(source)); break;
            case FieldType::Entity: AppendEntityId(out, Load<Entity>(source)); break;
            case FieldType::Opaque: out.Append("%02x", unsigned(source[0])); break;
            }
        }

        // `count` scalars from `first`, separated by `separator`.
        void AppendScalars(Writer& out, FieldType type, const uint8_t* source, uint32_t first, uint32_t count,
                           const char* separator) noexcept
        {
            const uint32_t size = ScalarSize(type);
            for (uint32_t i = 0; i < count; ++i)
            {
                if (i)
                    out.Append("%s", separator);
                AppendScalar(out, type, source + size_t(first + i) * size);
            }
        }

        // The columns of a Matrix field of `count` scalars: square when it is one, else rows of 4 (3x4), else one row.
        uint32_t MatrixColumns(uint32_t count) noexcept
        {
            for (uint32_t n = 1; n <= 4; ++n)
                if (n * n == count)
                    return n;
            return count % 4 == 0 ? 4u : std::max(count, 1u);
        }

        // The dimension n of an Affine field of n * n + n scalars; 0 when the count is not one.
        uint32_t AffineDimension(uint32_t count) noexcept
        {
            for (uint32_t n = 1; n <= 4; ++n)
                if (n * n + n == count)
                    return n;
            return 0;
        }

        // Scalars per widget row of an editable field.
        uint32_t EditRowWidth(const FieldDesc& field) noexcept
        {
            switch (field.shape)
            {
            case FieldShape::Matrix: return std::min(MatrixColumns(field.scalarCount), 4u);
            case FieldShape::Affine: { const uint32_t n = AffineDimension(field.scalarCount); return n ? n : 4u; }
            case FieldShape::Box: return std::max(std::min(field.scalarCount / 2, 4u), 1u);
            default: return 4;
            }
        }
    }

    void detail::FormatInspectorField(char* out, uint32_t capacity, const FieldDesc& field, const void* value) noexcept
    {
        if (!out || capacity == 0)
            return;
        Writer writer(out, capacity);
        const uint8_t* source = static_cast<const uint8_t*>(value) + field.offset;
        const uint32_t count = field.scalarCount;
        if (field.type == FieldType::Opaque)
        {
            const uint32_t shown = std::min(count, c_MaxOpaqueBytes);
            for (uint32_t i = 0; i < shown; ++i)
                writer.Append("%02x", unsigned(source[i]));
            if (shown < count)
                writer.Append("... (%u bytes)", count);
            return;
        }
        switch (field.shape)
        {
        case FieldShape::Scalar:
            AppendScalar(writer, field.type, source);
            break;
        case FieldShape::Vector:
        case FieldShape::Quaternion:
            writer.Append("(");
            AppendScalars(writer, field.type, source, 0, count, ", ");
            writer.Append(")");
            break;
        case FieldShape::Matrix:
        {
            const uint32_t columns = MatrixColumns(count);
            writer.Append("[");
            for (uint32_t first = 0; first < count; first += columns)
            {
                if (first)
                    writer.Append("; ");
                AppendScalars(writer, field.type, source, first, std::min(columns, count - first), " ");
            }
            writer.Append("]");
            break;
        }
        case FieldShape::Affine:
        {
            const uint32_t n = AffineDimension(count);
            if (n == 0)
            {
                writer.Append("[");
                AppendScalars(writer, field.type, source, 0, count, " ");
                writer.Append("]");
                break;
            }
            writer.Append("[");
            for (uint32_t row = 0; row < n; ++row)
            {
                if (row)
                    writer.Append("; ");
                AppendScalars(writer, field.type, source, row * n, n, " ");
            }
            writer.Append("] t (");
            AppendScalars(writer, field.type, source, n * n, n, ", ");
            writer.Append(")");
            break;
        }
        case FieldShape::Box:
        {
            const uint32_t half = count / 2;
            writer.Append("min (");
            AppendScalars(writer, field.type, source, 0, half, ", ");
            writer.Append(") max (");
            AppendScalars(writer, field.type, source, half, count - half, ", ");
            writer.Append(")");
            break;
        }
        case FieldShape::Array:
        default:
        {
            const uint32_t shown = std::min(count, c_MaxArrayScalars);
            writer.Append("[");
            AppendScalars(writer, field.type, source, 0, shown, ", ");
            writer.Append(shown < count ? ", ... (%u)]" : "]", count);
            break;
        }
        }
    }

    // ------------------------------------------------------------------------------------------------- state

    namespace
    {
        struct PendingEdit
        {
            Entity entity;
            ComponentId component;
            uint32_t size;
            alignas(16) uint8_t value[c_MaxEditBytes];
        };

        ImGuiDataType DataType(FieldType type) noexcept
        {
            switch (type)
            {
            case FieldType::Int8: return ImGuiDataType_S8;
            case FieldType::UInt8: return ImGuiDataType_U8;
            case FieldType::Int16: return ImGuiDataType_S16;
            case FieldType::UInt16: return ImGuiDataType_U16;
            case FieldType::Int32: return ImGuiDataType_S32;
            case FieldType::UInt32: return ImGuiDataType_U32;
            case FieldType::Int64: return ImGuiDataType_S64;
            case FieldType::UInt64: return ImGuiDataType_U64;
            case FieldType::Float: return ImGuiDataType_Float;
            default: return ImGuiDataType_Double;
            }
        }

        bool Has(ComponentFlags flags, ComponentFlags flag) noexcept { return !!(flags & flag); }
        bool Has(SystemFlags flags, SystemFlags flag) noexcept { return !!(flags & flag); }

        bool OpenHeader(const char* label) noexcept
        {
            if (g_TestHook && g_TestHook->openAll)
                ImGui::SetNextItemOpen(true);
            return ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen);
        }

        // A tree node with a formatted label; TreePop when it returns true.
        bool OpenNode(const void* id, ImGuiTreeNodeFlags flags, const char* label) noexcept
        {
            if (g_TestHook && g_TestHook->openAll)
                ImGui::SetNextItemOpen(true);
            return ImGui::TreeNodeEx(id, flags, "%s", label);
        }
    }

    struct WorldInspector::State
    {
        // Pending edits: the buffer holds its world (CommandBufferWorldReference::Held), so `bufferWorld` is valid while
        // the buffer lives, and no other world can be made at its address meanwhile. It is made at the first edit and
        // stays empty until Apply records one command per pending value (backlog I3-R4: re-edits record nothing).
        nvrhi::AutoPtr<ICommandBuffer> buffer;
        IWorld* bufferWorld = nullptr;
        PendingEdit pending[c_MaxPendingEdits];
        uint32_t pendingCount = 0;

        NameFunction names = nullptr;
        void* namesContext = nullptr;
        std::vector<ComponentId> readOnly;   // the application's (SetReadOnly)
        ComponentId donutReadOnly[engine::c_MaxInspectorReadOnlyComponents] = {};
        uint32_t donutReadOnlyCount = 0;     // read at each Draw

        Entity selected = Entity::Null;
        int browserArchetype = 0;
        int browserChunk = 0;

        // The copy the widgets edit; while one of a value's widgets is active, `typing` keeps it from frame to frame (the
        // world's value is not read again until the widget lets go).
        alignas(16) uint8_t scratch[c_MaxEditBytes] = {};
        alignas(16) uint8_t typing[c_MaxEditBytes] = {};
        Entity typingEntity = Entity::Null;
        ComponentId typingComponent = ComponentId::Invalid;
        char text[512] = {};
        char name[96] = {};
        char label[160] = {};

        void DropEdits() noexcept
        {
            buffer = nullptr;
            bufferWorld = nullptr;
            pendingCount = 0;
        }

        PendingEdit* FindPending(Entity entity, ComponentId component) noexcept
        {
            for (uint32_t i = 0; i < pendingCount; ++i)
                if (pending[i].entity == entity && pending[i].component == component)
                    return &pending[i];
            return nullptr;
        }

        bool IsReadOnly(ComponentId component) const noexcept
        {
            for (uint32_t i = 0; i < donutReadOnlyCount; ++i)
                if (donutReadOnly[i] == component)
                    return true;
            return std::find(readOnly.begin(), readOnly.end(), component) != readOnly.end();
        }

        // Relocatable, serialisable values of a row (not tags, buffers, Shared or Chunk values, resources), small
        // enough for the scratch copy, on neither read-only list.
        bool IsEditable(ComponentId component, const ComponentTypeDesc& desc) const noexcept
        {
            const ComponentFlags excluded = ComponentFlags::Tag | ComponentFlags::Buffer | ComponentFlags::Shared |
                                            ComponentFlags::Chunk | ComponentFlags::NoSerialize |
                                            ComponentFlags::Resource;
            return Has(desc.flags, ComponentFlags::Relocatable) && !Has(desc.flags, excluded) && desc.size > 0 &&
                   desc.size <= c_MaxEditBytes && !IsReadOnly(component);
        }

        // The entity's name from the application, or its id; into `label`.
        const char* EntityLabel(Entity entity) noexcept
        {
            Writer writer(label, sizeof(label));
            name[0] = 0;
            if (entity != Entity::Null && names && names(namesContext, entity, name, uint32_t(sizeof(name))))
            {
                name[sizeof(name) - 1] = 0;
                writer.Append("%s  ", name);
            }
            AppendEntityId(writer, entity);
            return label;
        }

        // The value becomes the pending one of (entity, component); false, with a warning, when it is dropped (the
        // 65th value, no command buffer).
        bool RecordEdit(IWorld* world, Entity entity, ComponentId component, const void* value, uint32_t size) noexcept
        {
            PendingEdit* edit = FindPending(entity, component);
            if (!edit && pendingCount == c_MaxPendingEdits)
            {
                log::warning("WorldInspector: %u edits are pending already; call Apply first. The edit is dropped",
                             c_MaxPendingEdits);
                return false;
            }
            if (!buffer)
            {
                if (NVRHI_FAILED(world->CreateCommandBuffer(&buffer)) || !buffer)
                {
                    log::warning("WorldInspector: cannot create the command buffer of the edits. The edit is dropped");
                    buffer = nullptr;
                    return false;
                }
                bufferWorld = world;
            }
            if (!edit)
            {
                edit = &pending[pendingCount++];
                edit->entity = entity;
                edit->component = component;
                edit->size = size;
            }
            std::memcpy(edit->value, value, size);
            return true;
        }

        void DrawWorld(IWorld* world) noexcept;
        void DrawArchetypes(IWorld* world) noexcept;
        void DrawOccupancy(IWorld* world, uint32_t archetype, const ArchetypeInfo& info) noexcept;
        void DrawVersions(IWorld* world, uint32_t archetype, const ArchetypeInfo& info) noexcept;
        void DrawSystems(IScheduler* scheduler) noexcept;
        void DrawBrowser(IWorld* world) noexcept;
        void DrawSelected(IWorld* world) noexcept;
        void DrawComponent(IWorld* world, ComponentId component, const ComponentTypeDesc& desc) noexcept;
        bool DrawFieldEditor(const FieldDesc& field, bool& active) noexcept;
        void DrawFieldView(const FieldDesc& field, const void* value) noexcept;
    };

    // ---------------------------------------------------------------------------------------------- panels

    void WorldInspector::State::DrawWorld(IWorld* world) noexcept
    {
        WorldStats stats{};
        world->GetStats(stats);
        ImGui::Text("entities %u (capacity %u), component types %u, sparse sets %u", stats.entityCount,
                    stats.entityCapacity, stats.componentTypeCount, stats.sparseSetCount);
        ImGui::Text("archetypes %u, chunks %u: %.2f MB reserved, %.2f MB in rows (%.1f %%)", stats.archetypeCount,
                    stats.chunkCount, double(stats.chunkBytes) / (1024.0 * 1024.0),
                    double(stats.rowBytes) / (1024.0 * 1024.0),
                    stats.chunkBytes ? 100.0 * double(stats.rowBytes) / double(stats.chunkBytes) : 0.0);
        ImGui::Text("buffer heap %.2f MB; commands played back %llu, observer batches %llu",
                    double(stats.bufferHeapBytes) / (1024.0 * 1024.0), (unsigned long long)stats.commandsPlayedBack,
                    (unsigned long long)stats.observerBatches);
        ImGui::Text("version %llu, structural version %llu; pending edits %u", (unsigned long long)stats.version,
                    (unsigned long long)stats.structuralVersion, pendingCount);
    }

    void WorldInspector::State::DrawArchetypes(IWorld* world) noexcept
    {
        const uint32_t count = world->GetArchetypeCount();
        for (uint32_t a = 0; a < count; ++a)
        {
            ArchetypeInfo info{};
            if (NVRHI_FAILED(world->GetArchetypeInfo(a, info)))
                continue;
            Writer writer(text, sizeof(text));
            writer.Append("#%u: %u rows, %u chunks:", a, info.entityCount, info.chunkCount);
            for (uint32_t c = 0; c < info.componentCount; ++c)
            {
                const ComponentTypeDesc* desc = world->GetComponentTypeDesc(info.components[c]);
                writer.Append(" %s", desc && desc->name ? desc->name : "?");
            }
            if (!OpenNode(reinterpret_cast<const void*>(uintptr_t(a) + 1), ImGuiTreeNodeFlags_None, text))
                continue;
            ImGui::Text("row %u B, chunk %u B, %u rows per chunk", info.rowSize, info.chunkSize, info.rowsPerChunk);
            DrawOccupancy(world, a, info);
            if (OpenNode("versions", ImGuiTreeNodeFlags_None, "change versions per chunk and column"))
            {
                DrawVersions(world, a, info);
                ImGui::TreePop();
            }
            ImGui::TreePop();
        }
    }

    // One bar per chunk, filled to its occupancy; the tooltip names the chunk under the mouse.
    void WorldInspector::State::DrawOccupancy(IWorld* world, uint32_t archetype, const ArchetypeInfo& info) noexcept
    {
        const uint32_t chunks = std::min(info.chunkCount, c_MaxStripChunks);
        if (chunks == 0)
        {
            ImGui::TextDisabled("no chunks");
            return;
        }
        constexpr float barWidth = 2.f;
        constexpr float step = 3.f;
        constexpr float height = 12.f;
        constexpr float rowStep = height + 2.f;
        const float width = std::max(ImGui::GetContentRegionAvail().x, 64.f);
        const uint32_t perRow = std::max(uint32_t(width / step), 1u);
        const uint32_t rows = (chunks + perRow - 1) / perRow;
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        const ImU32 empty = ImGui::GetColorU32(ImGuiCol_FrameBg);
        const ImU32 full = ImGui::GetColorU32(ImGuiCol_PlotHistogram);
        for (uint32_t c = 0; c < chunks; ++c)
        {
            ChunkInfo chunk{};
            if (NVRHI_FAILED(world->GetChunkInfo(archetype, c, chunk)))
                continue;
            const float fill = chunk.capacity ? float(chunk.count) / float(chunk.capacity) : 0.f;
            const float x = origin.x + float(c % perRow) * step;
            const float y = origin.y + float(c / perRow) * rowStep;
            drawList->AddRectFilled(ImVec2(x, y), ImVec2(x + barWidth, y + height), empty);
            drawList->AddRectFilled(ImVec2(x, y + height * (1.f - fill)), ImVec2(x + barWidth, y + height), full);
        }
        ImGui::Dummy(ImVec2(float(perRow) * step, float(rows) * rowStep));
        if (ImGui::IsItemHovered())
        {
            const ImVec2 mouse = ImGui::GetIO().MousePos;
            const uint32_t column = uint32_t(std::max(mouse.x - origin.x, 0.f) / step);
            const uint32_t row = uint32_t(std::max(mouse.y - origin.y, 0.f) / rowStep);
            const uint32_t c = row * perRow + std::min(column, perRow - 1);
            ChunkInfo chunk{};
            if (c < chunks && NVRHI_SUCCEEDED(world->GetChunkInfo(archetype, c, chunk)))
                ImGui::SetTooltip("chunk %u: %u of %u rows", c, chunk.count, chunk.capacity);
        }
        if (chunks < info.chunkCount)
            ImGui::TextDisabled("the first %u of %u chunks", chunks, info.chunkCount);
    }

    void WorldInspector::State::DrawVersions(IWorld* world, uint32_t archetype, const ArchetypeInfo& info) noexcept
    {
        const uint32_t columns = std::min(info.componentCount, c_MaxVersionColumns);
        const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX |
                                      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
        if (!ImGui::BeginTable("versions", int(columns) + 3, flags, ImVec2(0.f, 220.f)))
            return;
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("chunk");
        ImGui::TableSetupColumn("rows");
        ImGui::TableSetupColumn("added");
        for (uint32_t c = 0; c < columns; ++c)
        {
            const ComponentTypeDesc* desc = world->GetComponentTypeDesc(info.components[c]);
            ImGui::TableSetupColumn(desc && desc->name ? desc->name : "?");
        }
        ImGui::TableHeadersRow();
        ImGuiListClipper clipper;
        clipper.Begin(int(info.chunkCount));
        while (clipper.Step())
        {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
            {
                ChunkInfo chunk{};
                if (NVRHI_FAILED(world->GetChunkInfo(archetype, uint32_t(i), chunk)))
                    continue;
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%d", i);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%u/%u", chunk.count, chunk.capacity);
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%llu", (unsigned long long)chunk.addedVersion);
                for (uint32_t c = 0; c < columns; ++c)
                {
                    ImGui::TableSetColumnIndex(int(c) + 3);
                    ImGui::Text("%llu", (unsigned long long)chunk.versions[c]);
                }
            }
        }
        ImGui::EndTable();
        if (columns < info.componentCount)
            ImGui::TextDisabled("the first %u of %u components", columns, info.componentCount);
    }

    void WorldInspector::State::DrawSystems(IScheduler* scheduler) noexcept
    {
        const uint32_t count = scheduler->GetSystemCount();
        const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
        if (!ImGui::BeginTable("systems", 6, flags))
            return;
        ImGui::TableSetupColumn("system");
        ImGui::TableSetupColumn("flags");
        ImGui::TableSetupColumn("enabled");
        ImGui::TableSetupColumn("runs");
        ImGui::TableSetupColumn("last ms");
        ImGui::TableSetupColumn("average ms");
        ImGui::TableHeadersRow();
        bool first = true;
        Phase phase = Phase::Count;
        for (uint32_t i = 0; i < count; ++i)
        {
            SystemInfo info{};
            if (NVRHI_FAILED(scheduler->GetSystemInfo(i, info)))
                continue;
            if (first || info.phase != phase)
            {
                first = false;
                phase = info.phase;
                const char* name = scheduler->GetPhaseName(phase);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextDisabled("phase %s", name ? name : "?");
            }
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("  %s", info.name ? info.name : "?");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s%s%s", Has(info.flags, SystemFlags::Exclusive) ? "exclusive " : "",
                        Has(info.flags, SystemFlags::MainThread) ? "main-thread " : "",
                        Has(info.flags, SystemFlags::StartDisabled) ? "start-disabled" : "");
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", info.enabled ? "yes" : "no");
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%llu", (unsigned long long)info.stats.runCount);
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.3f", info.stats.lastMilliseconds);
            ImGui::TableSetColumnIndex(5);
            ImGui::Text("%.3f", info.stats.averageMilliseconds);
        }
        ImGui::EndTable();
    }

    // One chunk of one archetype at a time: 1M rows cost what one page costs.
    void WorldInspector::State::DrawBrowser(IWorld* world) noexcept
    {
        const uint32_t archetypes = world->GetArchetypeCount();
        if (archetypes == 0)
        {
            ImGui::TextDisabled("no archetypes");
            return;
        }
        ImGui::InputInt("archetype", &browserArchetype);
        browserArchetype = std::clamp(browserArchetype, 0, int(archetypes) - 1);
        ArchetypeInfo info{};
        if (NVRHI_FAILED(world->GetArchetypeInfo(uint32_t(browserArchetype), info)) || info.chunkCount == 0)
        {
            ImGui::TextDisabled("no rows");
            return;
        }
        ImGui::InputInt("chunk", &browserChunk);
        browserChunk = std::clamp(browserChunk, 0, int(info.chunkCount) - 1);
        ChunkInfo chunk{};
        if (NVRHI_FAILED(world->GetChunkInfo(uint32_t(browserArchetype), uint32_t(browserChunk), chunk)))
            return;
        ImGui::Text("chunk %d of %u: %u of %u rows", browserChunk, info.chunkCount, chunk.count, chunk.capacity);
        if (ImGui::BeginChild("rows", ImVec2(0.f, 200.f), ImGuiChildFlags_Borders))
        {
            ImGuiListClipper clipper;
            clipper.Begin(int(chunk.count));
            while (clipper.Step())
            {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row)
                {
                    const Entity entity = chunk.entities[row];
                    ImGui::PushID(row);
                    if (ImGui::Selectable(EntityLabel(entity), entity == selected))
                        selected = entity;
                    ImGui::PopID();
                }
            }
        }
        ImGui::EndChild();
    }

    void WorldInspector::State::DrawSelected(IWorld* world) noexcept
    {
        if (selected == Entity::Null || !world->IsAlive(selected))
        {
            ImGui::TextDisabled("no entity selected: pick one in the browser");
            return;
        }
        ImGui::Text("%s", EntityLabel(selected));
        const uint32_t types = world->GetComponentTypeCount();
        for (uint32_t t = 0; t < types; ++t)
        {
            const ComponentId component = ComponentId(t);
            const ComponentTypeDesc* desc = world->GetComponentTypeDesc(component);
            if (!desc || Has(desc->flags, ComponentFlags::Resource) || !world->HasComponent(selected, component))
                continue;
            ImGui::PushID(int(t));
            DrawComponent(world, component, *desc);
            ImGui::PopID();
        }
    }

    void WorldInspector::State::DrawComponent(IWorld* world, ComponentId component,
                                              const ComponentTypeDesc& desc) noexcept
    {
        const bool editable = IsEditable(component, desc);
        const bool enableable = Has(desc.flags, ComponentFlags::Enableable);
        {
            Writer writer(text, sizeof(text));
            writer.Append("%s", desc.name ? desc.name : "?");
            if (!editable)
                writer.Append(" (read-only)");
            if (enableable && !world->IsEnabled(selected, component))
                writer.Append(" (disabled)");
        }
        if (!OpenNode("component", ImGuiTreeNodeFlags_DefaultOpen, text))
            return;
        if (Has(desc.flags, ComponentFlags::Tag))
            ImGui::TextDisabled("tag");
        else if (Has(desc.flags, ComponentFlags::Buffer))
        {
            const auto* header = static_cast<const BufferHeader*>(world->GetComponent(selected, component));
            ImGui::Text("buffer: %u elements (capacity %u)", header ? header->count : 0u, header ? header->capacity : 0u);
        }
        else if (const void* value = world->GetComponent(selected, component))
        {
            if (editable)
            {
                const PendingEdit* edit = FindPending(selected, component);
                const bool typed = typingEntity == selected && typingComponent == component;
                std::memcpy(scratch, typed ? typing : edit ? edit->value : value, desc.size);
                bool commit = false;
                bool active = false;
                for (uint32_t f = 0; f < desc.fieldCount; ++f)
                {
                    ImGui::PushID(int(f));
                    commit = DrawFieldEditor(desc.fields[f], active) || commit;
                    ImGui::PopID();
                }
                if (active)
                {
                    std::memcpy(typing, scratch, desc.size);
                    typingEntity = selected;
                    typingComponent = component;
                }
                else if (typed)
                {
                    typingEntity = Entity::Null;
                    typingComponent = ComponentId::Invalid;
                }
                const detail::WorldInspectorTestHook* hook = g_TestHook;
                if (hook && hook->edit && hook->edit(hook->context, component, scratch, desc.size))
                    commit = true;
                // pending before, or recorded now; not for a dropped edit (backlog I3-R3)
                bool pending = edit != nullptr;
                if (commit)
                    pending = RecordEdit(world, selected, component, scratch, desc.size) || pending;
                if (pending)
                    ImGui::TextDisabled("edited; Apply writes it");
            }
            else
            {
                for (uint32_t f = 0; f < desc.fieldCount; ++f)
                {
                    ImGui::PushID(int(f));
                    DrawFieldView(desc.fields[f], value);
                    ImGui::PopID();
                }
            }
            if (desc.fieldCount == 0)
            {
                const FieldDesc bytes{ "bytes", 0, desc.size, FieldType::Opaque, FieldShape::Array, {} };
                DrawFieldView(bytes, value);
            }
        }
        ImGui::TreePop();
    }

    // Widgets over the scratch copy, one per scalar; true when one committed: a checkbox clicked, or a number field let
    // go after an edit (Enter, Tab or a click elsewhere). `active` is set while a number field has the keyboard.
    bool WorldInspector::State::DrawFieldEditor(const FieldDesc& field, bool& active) noexcept
    {
        if (field.type == FieldType::Opaque || field.type == FieldType::Entity)
        {
            DrawFieldView(field, scratch);
            return false;
        }
        const char* name = field.name ? field.name : "?";
        uint8_t* source = scratch + field.offset;
        bool commit = false;
        if (field.type == FieldType::Boolean)
        {
            ImGui::TextUnformatted(name);
            for (uint32_t i = 0; i < field.scalarCount; ++i)
            {
                ImGui::SameLine();
                ImGui::PushID(int(i));
                bool value = source[i] != 0;
                if (ImGui::Checkbox("##value", &value))
                {
                    source[i] = value ? 1 : 0;
                    commit = true;
                }
                ImGui::PopID();
            }
            return commit;
        }
        const ImGuiDataType type = DataType(field.type);
        // every digit the type holds, so that an edit of one digit keeps the others (backlog I3-R5)
        const char* format = field.type == FieldType::Float ? "%.9g" : field.type == FieldType::Double ? "%.17g"
                                                                                                         : nullptr;
        const uint32_t size = ScalarSize(field.type);
        const uint32_t width = EditRowWidth(field);
        for (uint32_t i = 0; i < field.scalarCount; ++i)
        {
            if (i % width == 0)
                ImGui::TextUnformatted(i == 0 ? name : "");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(96.f);
            ImGui::PushID(int(i));
            ImGui::InputScalar("##value", type, source + size_t(i) * size, nullptr, nullptr, format);
            active = active || ImGui::IsItemActive();
            commit = commit || ImGui::IsItemDeactivatedAfterEdit();   // the value as typed when the widget let go
            ImGui::PopID();
        }
        return commit;
    }

    // A field as text; an entity is a link that selects it.
    void WorldInspector::State::DrawFieldView(const FieldDesc& field, const void* value) noexcept
    {
        const char* name = field.name ? field.name : "?";
        if (field.type == FieldType::Entity && field.shape == FieldShape::Scalar)
        {
            const Entity entity = Load<Entity>(static_cast<const uint8_t*>(value) + field.offset);
            ImGui::Text("%s:", name);
            ImGui::SameLine();
            if (entity == Entity::Null)
                ImGui::TextDisabled("null");
            else if (ImGui::SmallButton(EntityLabel(entity)))
                selected = entity;
            return;
        }
        detail::FormatInspectorField(text, uint32_t(sizeof(text)), field, value);
        ImGui::Text("%s: %s", name, text);
    }

    // ------------------------------------------------------------------------------------------- the inspector

    WorldInspector::WorldInspector()
        : m_State(std::make_unique<State>())
    {
    }

    WorldInspector::~WorldInspector() = default;

    void WorldInspector::Draw(IWorld* world, IScheduler* scheduler)
    {
        State& s = *m_State;
        if (!world)
        {
            log::warning("WorldInspector::Draw: no world; nothing drawn");
            return;
        }
        if (world->IsInPhase())
        {
            log::warning("WorldInspector::Draw: called inside a phase; nothing drawn");
            return;
        }
        if (s.buffer && s.bufferWorld != world)
        {
            log::warning("WorldInspector::Draw: %u pending edits of another world dropped", s.pendingCount);
            s.DropEdits();
        }
        if (g_TestHook && g_TestHook->select != Entity::Null)
            s.selected = g_TestHook->select;

        if (ImGui::Begin("World inspector"))
        {
            s.donutReadOnlyCount = std::min(GetDonutReadOnlyComponents(world, s.donutReadOnly,
                                                                        engine::c_MaxInspectorReadOnlyComponents),
                                            engine::c_MaxInspectorReadOnlyComponents);
            if (OpenHeader("World"))
                s.DrawWorld(world);
            if (OpenHeader("Archetypes"))
                s.DrawArchetypes(world);
            if (scheduler && scheduler->GetWorld() != world)
            {
                // its systems are another world's (backlog I3-R6)
                log::warning("WorldInspector::Draw: the scheduler is not the world's; the Systems panel is not drawn");
            }
            else if (scheduler && OpenHeader("Systems"))
                s.DrawSystems(scheduler);
            if (OpenHeader("Entities"))
                s.DrawBrowser(world);
            if (OpenHeader("Selected entity"))
                s.DrawSelected(world);
        }
        ImGui::End();
    }

    nvrhi::FRESULT WorldInspector::Apply(IWorld* world)
    {
        State& s = *m_State;
        if (world && world->IsInPhase())
        {
            log::warning("WorldInspector::Apply: called inside a phase; the edits are kept");
            return nvrhi::FE_OUT_OF_DATE;
        }
        if (!s.buffer)
            return nvrhi::FS_OK;
        if (world != s.bufferWorld)
        {
            log::warning("WorldInspector::Apply: %u pending edits belong to another world; dropped", s.pendingCount);
            s.DropEdits();
            return nvrhi::FE_INVALID_ARGS;
        }
        // One command per pending value, in the order of their first edits; the buffer keeps no earlier ones.
        for (uint32_t i = 0; i < s.pendingCount; ++i)
            s.buffer->SetComponent(s.pending[i].entity, s.pending[i].component, s.pending[i].value, i);
        const nvrhi::FRESULT result = world->Playback(s.buffer.GetAddressOf(), 1);
        if (result == nvrhi::FE_OUT_OF_DATE)
        {
            s.buffer->Reset();   // recorded again by the next Apply
            log::warning("WorldInspector::Apply: the world refuses playback now (an observer runs); the edits are kept");
            return result;
        }
        world->AdvanceVersion();
        s.DropEdits();
        return result;
    }

    void WorldInspector::SetEntityNames(NameFunction function, void* context)
    {
        m_State->names = function;
        m_State->namesContext = function ? context : nullptr;
    }

    void WorldInspector::SetReadOnly(const ComponentId* components, uint32_t count)
    {
        m_State->readOnly.clear();
        if (count && !components)
        {
            log::warning("WorldInspector::SetReadOnly: %u components but no list; the list is cleared", count);
            return;
        }
        m_State->readOnly.assign(components, components + count);
    }

    uint32_t GetDonutReadOnlyComponents(IWorld* world, ComponentId* out, uint32_t capacity) noexcept
    {
        if (!world)
            return 0;
        using ReadOnlyTraits = engine::ComponentTraits<engine::InspectorReadOnlyComponents>;
        const ComponentId id = world->FindComponent(ReadOnlyTraits::Desc().uuid);
        if (id == ComponentId::Invalid)
            return 0;
        const auto* list = static_cast<const engine::InspectorReadOnlyComponents*>(world->GetResource(id));
        if (!list)
            return 0;
        const uint32_t count = std::min(list->count, engine::c_MaxInspectorReadOnlyComponents);
        if (out)
            for (uint32_t i = 0; i < count && i < capacity; ++i)
                out[i] = list->components[i];
        return count;
    }

    void detail::SetWorldInspectorTestHook(const WorldInspectorTestHook* hook) noexcept
    {
        g_TestHook = hook;
    }
}
