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

// Deferred structural changes (aggregate docs/ecs-design.md, 6.5). Systems record into command buffers; the world
// plays them back at sync points so that the resulting chunk and row order does not depend on thread timing
// (rule 3.3.6):
// - The scheduler plays back once per system, systems in schedule order: the phase's dependency order, ties broken by
//   registration order (never by SystemId, which may be reused).
// - Within one system, IWorld::Playback merges the system's per-worker buffers by sort key. The merge is stable:
//   commands with equal keys in one buffer keep their recording order. Equal keys in different buffers would order by
//   scheduling, so a world with validation reports them (FE_INVALID_ARGS) and keys must be unique per system.
// - Recording is per batch of entities, so that a system records per chunk, not with a virtual call per entity
//   (rule 3.3.4).

#pragma once

#include <donut/engine/EcsTypes.h>

namespace donut::engine
{
    // The key for a command issued while iterating: loop (16 bits) | chunk index (32) | row (16). `loop` tells apart
    // passes of one system over the same chunks (several queries, or several loops), so that keys stay unique.
    constexpr uint32_t c_MaxSortKeyLoop = 0xffffu;
    constexpr uint32_t c_MaxSortKeyRow = 0xffffu;   // a chunk holds at most 64 KiB / 8 bytes of Entity = 8192 rows

    constexpr uint64_t MakeSortKey(uint32_t loop, uint32_t chunkIndex, uint32_t row) noexcept
    {
        return (uint64_t(loop & c_MaxSortKeyLoop) << 48) | (uint64_t(chunkIndex) << 16) | uint64_t(row & c_MaxSortKeyRow);
    }

    // One buffer is used by one thread at a time. Recording reports no error; IWorld::Playback validates the commands
    // and returns the first failure. Values are copied (or relocated, EntityCreateDesc::transfer) into the buffer when
    // recorded.
    //
    // Placeholders: CreateEntities and Instantiate return placeholders (IsPlaceholder) that are unique across all
    // buffers of the world until they are played back, so later commands of any buffer of the same system may name
    // them. At playback they become entities, and every Entity field (FieldType::Entity) of a recorded value that holds
    // a placeholder is rewritten to the entity it became. IWorld::ResolvePlaceholder maps them afterwards.
    NVRHI_IID(ICommandBuffer, "e78d5307-7cf1-4db5-8208-b342b28e2351")
    struct ICommandBuffer : nvrhi::IObject
    {
        NVRHI_DECLARE_UUID_TRAITS(ICommandBuffer)

        // desc.reservedEntities must be null (reserved ids are for immediate creation). Writes desc.entityCount
        // placeholders.
        virtual void CreateEntities(const EntityCreateDesc& desc, uint64_t sortKey,
                                    _Out_writes_(desc.entityCount) Entity* outEntities) noexcept = 0;
        virtual void DestroyEntities(_In_reads_(count) const Entity* entities, uint32_t count,
                                     uint64_t sortKey) noexcept = 0;
        // `values`: `count` consecutive values, or null to default-construct.
        virtual void AddComponents(_In_reads_(count) const Entity* entities, uint32_t count, ComponentId component,
                                   _In_opt_ const void* values, uint64_t sortKey) noexcept = 0;
        virtual void RemoveComponents(_In_reads_(count) const Entity* entities, uint32_t count, ComponentId component,
                                      uint64_t sortKey) noexcept = 0;
        // `count` copies of `prefab`'s row, in one archetype (4.10).
        virtual void Instantiate(Entity prefab, uint32_t count, uint64_t sortKey,
                                 _Out_writes_(count) Entity* outEntities) noexcept = 0;

        // Not structural, but ordered with the structural commands. SetComponents raises OnSet (6.6).
        virtual void SetComponents(_In_reads_(count) const Entity* entities, uint32_t count, ComponentId component,
                                   const void* values, uint64_t sortKey) noexcept = 0;
        virtual void SetEnabled(_In_reads_(count) const Entity* entities, uint32_t count, ComponentId component,
                                bool enabled, uint64_t sortKey) noexcept = 0;
        virtual void AppendToBuffer(Entity entity, ComponentId component, _In_reads_(count) const void* elements,
                                    uint32_t count, uint64_t sortKey) noexcept = 0;

        virtual uint32_t GetCommandCount() noexcept = 0;
        // Drops the recorded commands without playing them back; their placeholders never resolve.
        virtual void Reset() noexcept = 0;

        // Single-entity forms. Inline, not virtual.
        void DestroyEntity(Entity entity, uint64_t sortKey) noexcept { DestroyEntities(&entity, 1, sortKey); }
        void AddComponent(Entity entity, ComponentId component, _In_opt_ const void* value, uint64_t sortKey) noexcept
        {
            AddComponents(&entity, 1, component, value, sortKey);
        }
        void RemoveComponent(Entity entity, ComponentId component, uint64_t sortKey) noexcept
        {
            RemoveComponents(&entity, 1, component, sortKey);
        }
        void SetComponent(Entity entity, ComponentId component, const void* value, uint64_t sortKey) noexcept
        {
            SetComponents(&entity, 1, component, value, sortKey);
        }
    };
}
