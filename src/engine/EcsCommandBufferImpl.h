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

// Command buffers and their playback (aggregate docs/ecs-design.md 6.5; plan step E1.3). Module-private
// (com-interface-abi.md section 6).
//
// Recording: a buffer appends fixed-size Command records and copies (or relocates) every entity list and value into
// its own arena, whose pages never move (non-Relocatable values may live there) and are kept across Reset. A failure
// found while recording (an unknown component, a value that cannot be copied, out of memory) becomes a Failed command,
// so that Playback returns it at the command's place. Placeholders are indices from the world's CommandPlayback counter
// (one atomic add per creating command) with c_PlaceholderGeneration.
//
// Playback (PlayCommandBuffers): the commands of all buffers are ordered by (sort key, buffer position, recording
// order); placeholder ranges are registered; then the commands run in that order through the world's *Internal batch
// operations with the world's version. Adjacent Destroy commands, and adjacent Add, Remove, Set or SetEnabled commands
// of one component (and one value mode or flag) form one batch: one *Internal call over their concatenated entities
// and values. Values are handed over with ValueTransfer::Relocate, so the operation consumes them, whatever happens;
// the buffer destroys only values it never handed over (Reset, destruction). The result is the failure of the earliest
// failing command in that order, as if the commands were applied one by one.

#pragma once

#include "EcsWorldImpl.h"

#include <nvrhi/core/autoptr.h>

#include <atomic>

namespace donut::engine::ecs
{
    class CommandBufferImpl;
    struct PlaceholderTable;

    // Whether a buffer keeps its world alive. IWorld::CreateCommandBuffer hands out Held buffers. A buffer that the
    // world owns itself (E1.4's observer buffer) must be NotHeld, or the two would keep each other alive; its owner
    // releases it before the world's component types are destroyed (Reset destroys values with their hooks).
    enum class CommandBufferWorldReference : uint8_t
    {
        Held,
        NotHeld,
    };

    // FE_INVALID_ARGS for a null ppBuffer, FE_OUT_OF_MEMORY.
    nvrhi::FRESULT CreateCommandBufferObject(WorldImpl& world, CommandBufferWorldReference reference,
                                             ICommandBuffer** ppBuffer) noexcept;

    // Plays back `count` buffers of `world` in merged order, stamping `version`, and resets them. The caller checks the
    // list: no null entry, each buffer once, this world's, not being recorded. Each creating command's placeholders are
    // added to `placeholders` and resolve through it, for the later commands of this call and of later calls with the
    // same table until its owner clears it:
    // - IWorld::Playback passes CommandPlayback::placeholders, which it clears when it starts and which
    //   IWorld::ResolvePlaceholder answers from;
    // - E1.4's observer cascades pass CommandPlayback::observerPlaceholders, cleared when a cascade starts, so that
    //   flushes from immediate calls never grow the public table.
    // Does not flush observers. Returns the failure of the earliest failing command in merged order (a buffer whose
    // recording a second thread lost fails before every command); `outCommands` counts the commands played back.
    //
    // E1.4: every *Internal call it makes is one batch, in playback order (PlaybackRun::Apply, ApplyCreate,
    // ApplyBatch), and the Observe* reports of one call are contiguous. Not re-entrant: it uses CommandPlayback's
    // scratch, so observer buffers are played back after it returns (IWorld::Playback's FlushObservers), not from
    // inside a batch. A buffer must not be recorded into while it is played back.
    nvrhi::FRESULT PlayCommandBuffers(WorldImpl& world, CommandBufferImpl* const* buffers, uint32_t count,
                                      uint64_t version, PlaceholderTable& placeholders, uint64_t& outCommands) noexcept;

    // ------------------------------------------------------------------------------------------------ commands

    enum class CommandOp : uint8_t
    {
        Create,
        Destroy,
        Add,
        Remove,
        Set,
        SetEnabled,
        Instantiate,
        Append,
        Failed,     // rejected when recorded; `failure` is what playback returns
    };

    struct Command
    {
        uint64_t sortKey;
        CommandOp op;
        bool enabled;               // SetEnabled
        bool ownsValues;            // the arena holds values that nobody has taken yet
        bool prepared;              // playback resolved its placeholders (PlaybackRun::Prepare)
        ComponentId component;      // Add, Remove, Set, SetEnabled, Append
        uint32_t count;             // Create, Instantiate: entities; Append: elements; otherwise the entity list
        uint32_t componentCount;    // Create
        uint32_t placeholder;       // Create, Instantiate: the first placeholder index
        uint32_t resolved;          // Create, Instantiate: during playback, the offset of its entities in the table
        nvrhi::FRESULT failure;     // Failed: what playback returns. Others: a failure found when playback prepared
                                    // the command (an Entity field naming a placeholder that does not resolve)
        uint32_t reserved;
        Entity entity;              // Instantiate: the prefab; Append: the entity
        Entity* entities;           // Destroy, Add, Remove, Set, SetEnabled: in the arena
        void* values;               // Add, Set: `count` values or null; Append: the elements; Create: componentCount
                                    // pointers, each to entityCount values or null
        ComponentId* components;    // Create
    };

    // Pages for a buffer's lists and values. Pages never move, so values that are not Relocatable may live in them;
    // Reset keeps every page. A request goes to the current page, else to the first kept page with room, else to a new
    // page: a buffer that records the same sequence of sizes again allocates nothing, and other sequences converge as
    // pages accumulate (a different order of the same sizes may need one more page).
    class CommandArena
    {
    public:
        CommandArena() noexcept = default;
        CommandArena(const CommandArena&) = delete;
        CommandArena& operator=(const CommandArena&) = delete;
        ~CommandArena() { Release(); }

        // `alignment` is a power of two of at most c_ColumnAlignment. Null when out of memory.
        void* Allocate(size_t bytes, size_t alignment) noexcept;
        void Reset() noexcept;
        void Release() noexcept;
        size_t GetCapacity() const noexcept;
        uint32_t GetPageCount() const noexcept { return uint32_t(m_Pages.size()); }

    private:
        struct Page
        {
            uint8_t* memory;
            size_t size;
            size_t used;
        };

        static constexpr size_t c_FirstPageBytes = 16 * 1024;
        static constexpr size_t c_LargestPageBytes = 256 * 1024;   // a page for one larger request is its own size

        // The page's next free byte for this request, or null.
        static uint8_t* Fit(Page& page, size_t bytes, size_t alignment) noexcept;

        Vector<Page> m_Pages;
        uint32_t m_Current = 0;
    };

    // --------------------------------------------------------------------------------------------- the buffer

    NVRHI_CLASS_CLSID(CommandBufferImpl, "3946f3ce-567d-4a0c-918e-6eed25f85a1d")
    class CommandBufferImpl final : public nvrhi::ObjectImpl<ICommandBuffer>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(CommandBufferImpl)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(CommandBufferImpl)
        NVRHI_IMPLEMENTS_INTERFACE(ICommandBuffer)
        NVRHI_IMPLEMENTS_CLASS(CommandBufferImpl)
        NVRHI_END_INTERFACE_TABLE()

        CommandBufferImpl(WorldImpl* world, CommandBufferWorldReference reference) noexcept;
        ~CommandBufferImpl();

        // ICommandBuffer
        void CreateEntities(const EntityCreateDesc& desc, uint64_t sortKey, Entity* outEntities) noexcept override;
        void DestroyEntities(const Entity* entities, uint32_t count, uint64_t sortKey) noexcept override;
        void AddComponents(const Entity* entities, uint32_t count, ComponentId component, const void* values,
                           uint64_t sortKey) noexcept override;
        void RemoveComponents(const Entity* entities, uint32_t count, ComponentId component,
                              uint64_t sortKey) noexcept override;
        void Instantiate(Entity prefab, uint32_t count, uint64_t sortKey, Entity* outEntities) noexcept override;
        void SetComponents(const Entity* entities, uint32_t count, ComponentId component, const void* values,
                           uint64_t sortKey) noexcept override;
        void SetEnabled(const Entity* entities, uint32_t count, ComponentId component, bool enabled,
                        uint64_t sortKey) noexcept override;
        void AppendToBuffer(Entity entity, ComponentId component, const void* elements, uint32_t count,
                            uint64_t sortKey) noexcept override;
        uint32_t GetCommandCount() noexcept override { return uint32_t(m_Commands.size()); }
        void Reset() noexcept override;

        // ------------------------------------------------------------------------------- module-internal access

        WorldImpl* GetWorld() const noexcept { return m_World; }
        Command* GetCommands() noexcept { return m_Commands.data(); }
        const CommandArena& GetArena() const noexcept { return m_Arena; }
        // Destroys the values nobody took, drops the commands and keeps the capacity (Reset without the use check).
        void Clear() noexcept;
        // With validation: whether a thread is recording into the buffer right now.
        bool IsBusy() const noexcept { return m_Busy.load(std::memory_order_acquire) != 0; }
        // With validation: a second thread called the buffer while another recorded, and its call recorded nothing.
        // Playback reads and clears it, and fails with FE_INVALID_ARGS.
        bool TakeLostRecording() noexcept { return m_LostRecording.exchange(false, std::memory_order_acq_rel); }
        // Tests of the one-thread-at-a-time check: marks the buffer as being recorded by another thread.
        void SimulateConcurrentUseForTest(bool busy) noexcept
        {
            m_Busy.store(busy ? 1u : 0u, std::memory_order_release);
        }

    private:
        class RecordScope;

        // A Failed command: `result` at this key. ValueTransfer::Relocate values must have been dealt with.
        void RecordFailure(uint64_t sortKey, nvrhi::FRESULT result) noexcept;
        // The component of a recorded command, or null (a Failed command is recorded, with a report).
        const ComponentType* RequireComponent(const char* operation, ComponentId component, uint64_t sortKey) noexcept;
        // Copies `count` entities into the arena; null when out of memory (a Failed command is recorded).
        Entity* CopyEntities(const Entity* entities, uint32_t count, uint64_t sortKey) noexcept;
        // Copies `count` values into the arena; null when out of memory (a Failed command is recorded).
        void* CopyValues(const ComponentType& type, const void* values, uint32_t count, uint64_t sortKey) noexcept;
        // The common form of Add, Remove, Set and SetEnabled.
        void RecordEntityCommand(const char* operation, CommandOp op, const Entity* entities, uint32_t count,
                                 ComponentId component, const void* values, bool enabled, uint64_t sortKey) noexcept;
        // Placeholders for a creating command, written whatever happens to it.
        uint32_t HandOutPlaceholders(uint32_t count, Entity* outEntities) noexcept;
        void DestroyOwnedValues() noexcept;

        nvrhi::AutoPtr<IWorld> m_WorldReference;   // null for CommandBufferWorldReference::NotHeld; released last
        WorldImpl* m_World;
        bool m_Validation;
        bool m_OwnsDestructibleValues = false;     // some recorded value has a destruct hook
        std::atomic<uint32_t> m_Busy{ 0 };         // validation: a thread is recording
        std::atomic<bool> m_LostRecording{ false }; // validation: a second thread's call recorded nothing
        Vector<Command> m_Commands;
        CommandArena m_Arena;
    };

    // ------------------------------------------------------------------------------------ the world's state

    // The placeholders of one IWorld::Playback (CommandPlayback::placeholders) or of one observer cascade
    // (CommandPlayback::observerPlaceholders): each creating command's range of placeholder indices and the entities
    // they became. Clear keeps the capacity.
    struct PlaceholderTable
    {
        // Indices [first, first + count) became resolved[offset, offset + count).
        struct Range
        {
            uint32_t first;
            uint32_t count;
            uint32_t offset;
            uint32_t reserved;
        };

        Vector<Range> ranges;      // sorted by `first`
        Vector<Entity> resolved;   // Null until the command runs, and where it failed

        // The entity a placeholder became: Null when no command of the table created it, or the command has not run
        // yet or failed for that entity.
        Entity Resolve(Entity placeholder) const noexcept;
        void Clear() noexcept
        {
            ranges.clear();
            resolved.clear();
        }
    };

    // What the world keeps for command buffers (WorldImpl::GetCommandPlayback): the placeholder counter, the
    // placeholder tables, and the playback's scratch, kept between playbacks so that steady-state playback allocates
    // nothing.
    class CommandPlayback
    {
    public:
        struct Counters
        {
            uint64_t playbacks = 0;     // PlayCommandBuffers calls
            uint64_t commands = 0;      // commands played back
            uint64_t batches = 0;       // *Internal calls: one per Create, Instantiate and Append, one per run of
                                        // coalesced commands, none for a Failed command
            uint64_t coalesced = 0;     // commands that joined the batch of the command before them
            uint64_t sorts = 0;         // playbacks whose merged order needed a sort
        };

        // One command in merged order.
        struct Entry
        {
            uint64_t key;
            uint32_t buffer;     // position in the playback's buffer list
            uint32_t command;    // recording order within the buffer
        };

        CommandPlayback() noexcept = default;
        CommandPlayback(const CommandPlayback&) = delete;
        CommandPlayback& operator=(const CommandPlayback&) = delete;
        ~CommandPlayback() { HeapFree(m_Gather); }

        // `count` consecutive placeholder indices, unique among the placeholders not played back yet (the counter
        // wraps after 2^32, far beyond what buffers hold between playbacks). Safe from any thread.
        uint32_t AllocatePlaceholders(uint32_t count) noexcept
        {
            return uint32_t(m_NextPlaceholder.fetch_add(count, std::memory_order_relaxed));
        }
        // A block for the values of a coalesced batch, 64-byte aligned; null when out of memory. It at least doubles
        // when it grows.
        uint8_t* GatherBytes(size_t bytes) noexcept;
        size_t GetGatherSize() const noexcept { return m_GatherSize; }
        const Counters& GetCounters() const noexcept { return counters; }

        // IWorld::Playback's table: cleared when a Playback starts; IWorld::ResolvePlaceholder answers from it.
        PlaceholderTable placeholders;
        // E1.4's table for observer cascades: cleared when a cascade starts.
        PlaceholderTable observerPlaceholders;

        // The playback's scratch (EcsCommandBuffer.cpp).
        Vector<Entry> entries;
        Vector<CommandBufferImpl*> buffers;     // IWorld::Playback's checked list
        Vector<Entity> runEntities;             // the concatenated entity lists of a coalesced batch
        Counters counters;

    private:
        std::atomic<uint64_t> m_NextPlaceholder{ 1 };
        uint8_t* m_Gather = nullptr;
        size_t m_GatherSize = 0;
    };
}
