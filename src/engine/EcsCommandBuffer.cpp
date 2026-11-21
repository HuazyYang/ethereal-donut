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

// Command buffers and their playback (aggregate docs/ecs-design.md 3.3.6, 6.5; plan step E1.3). See
// EcsCommandBufferImpl.h.

#include "EcsCommandBufferImpl.h"

#include <donut/core/query_cast.h>

#include <algorithm>
#include <cstring>
#include <new>

namespace donut::engine::ecs
{
    namespace
    {
        void KeepFirstFailure(nvrhi::FRESULT& result, nvrhi::FRESULT next) noexcept
        {
            if (NVRHI_SUCCEEDED(result) && NVRHI_FAILED(next))
                result = next;
        }

        constexpr size_t AlignUp(size_t value, size_t alignment) noexcept
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        const char* NameOf(const ComponentType& type) noexcept
        {
            return type.desc.name ? type.desc.name : "(unnamed)";
        }

        const char* OpName(CommandOp op) noexcept
        {
            switch (op)
            {
            case CommandOp::Create: return "CreateEntities";
            case CommandOp::Destroy: return "DestroyEntities";
            case CommandOp::Add: return "AddComponents";
            case CommandOp::Remove: return "RemoveComponents";
            case CommandOp::Set: return "SetComponents";
            case CommandOp::SetEnabled: return "SetEnabled";
            case CommandOp::Instantiate: return "Instantiate";
            case CommandOp::Append: return "AppendToBuffer";
            default: return "a failed command";
            }
        }

        bool EntryLess(const CommandPlayback::Entry& a, const CommandPlayback::Entry& b) noexcept
        {
            if (a.key != b.key)
                return a.key < b.key;
            if (a.buffer != b.buffer)
                return a.buffer < b.buffer;
            return a.command < b.command;
        }

        // Commands whose adjacent runs are applied as one batch.
        bool IsCoalescable(CommandOp op) noexcept
        {
            return op == CommandOp::Destroy || op == CommandOp::Add || op == CommandOp::Remove ||
                   op == CommandOp::Set || op == CommandOp::SetEnabled;
        }

        // Whether `next` may join the batch that `lead` began: the same operation on the same component, the same
        // value mode (Add: values or defaults) and the same flag (SetEnabled).
        bool SameBatch(const Command& lead, const Command& next) noexcept
        {
            if (next.op != lead.op)
                return false;
            if (lead.op == CommandOp::Destroy)
                return true;
            if (next.component != lead.component)
                return false;
            if (lead.op == CommandOp::Add && (next.values != nullptr) != (lead.values != nullptr))
                return false;
            if (lead.op == CommandOp::SetEnabled && next.enabled != lead.enabled)
                return false;
            return true;
        }
    }

    // ----------------------------------------------------------------------------------------------- arena

    uint8_t* CommandArena::Fit(Page& page, size_t bytes, size_t alignment) noexcept
    {
        const size_t offset = AlignUp(page.used, alignment);
        if (offset > page.size || bytes > page.size - offset)
            return nullptr;
        page.used = offset + bytes;
        return page.memory + offset;
    }

    void* CommandArena::Allocate(size_t bytes, size_t alignment) noexcept
    {
        if (bytes == 0)
            bytes = 1;
        // The current page, else the first kept page with room, else a new page.
        const uint32_t pageCount = uint32_t(m_Pages.size());
        if (m_Current < pageCount)
        {
            if (uint8_t* memory = Fit(m_Pages[m_Current], bytes, alignment))
                return memory;
            for (uint32_t p = 0; p < pageCount; ++p)
            {
                if (p == m_Current)
                    continue;
                if (uint8_t* memory = Fit(m_Pages[p], bytes, alignment))
                {
                    m_Current = p;
                    return memory;
                }
            }
        }
        const size_t shift = pageCount < 4 ? pageCount : 4;
        size_t size = c_FirstPageBytes << shift;
        if (size > c_LargestPageBytes)
            size = c_LargestPageBytes;
        if (bytes > size)
            size = AlignUp(bytes, c_ColumnAlignment);
        uint8_t* memory = static_cast<uint8_t*>(HeapAllocate(size, c_ColumnAlignment));
        if (!memory)
            return nullptr;
        m_Pages.push_back(Page{ memory, size, bytes });
        m_Current = uint32_t(m_Pages.size() - 1);
        return memory;
    }

    void CommandArena::Reset() noexcept
    {
        for (Page& page : m_Pages)
            page.used = 0;
        m_Current = 0;
    }

    void CommandArena::Release() noexcept
    {
        for (const Page& page : m_Pages)
            HeapFree(page.memory);
        m_Pages.clear();
        m_Current = 0;
    }

    size_t CommandArena::GetCapacity() const noexcept
    {
        size_t bytes = 0;
        for (const Page& page : m_Pages)
            bytes += page.size;
        return bytes;
    }

    // ------------------------------------------------------------------------------------- the world's state

    CommandPlayback* CreateCommandPlayback() noexcept
    {
        void* memory = HeapAllocate(sizeof(CommandPlayback), alignof(CommandPlayback));
        return memory ? new (memory) CommandPlayback() : nullptr;
    }

    void DestroyCommandPlayback(CommandPlayback* playback) noexcept
    {
        if (!playback)
            return;
        playback->~CommandPlayback();
        HeapFree(playback);
    }

    Entity PlaceholderTable::Resolve(Entity placeholder) const noexcept
    {
        const uint32_t index = GetEntityIndex(placeholder);
        const Range* first = ranges.begin();
        const Range* last = ranges.end();
        const Range* found = std::upper_bound(first, last, index,
            [](uint32_t i, const Range& range) { return i < range.first; });
        if (found == first)
            return Entity::Null;
        --found;
        const uint32_t k = index - found->first;
        return k < found->count ? resolved[size_t(found->offset) + k] : Entity::Null;
    }

    uint8_t* CommandPlayback::GatherBytes(size_t bytes) noexcept
    {
        if (bytes > m_GatherSize)
        {
            const size_t size = std::max(bytes, m_GatherSize * 2);
            HeapFree(m_Gather);
            m_GatherSize = 0;
            m_Gather = static_cast<uint8_t*>(HeapAllocate(size, c_ColumnAlignment));
            if (m_Gather)
                m_GatherSize = size;
        }
        return m_Gather;
    }

    // ---------------------------------------------------------------------------------------------- buffers

    // With validation: the check of EcsCommandBuffer.h that one buffer is used by one thread at a time. A call that
    // finds another thread in the buffer records nothing (it would race with that thread) and makes the buffer's next
    // playback fail.
    class CommandBufferImpl::RecordScope
    {
    public:
        RecordScope(CommandBufferImpl& buffer, const char* operation) noexcept
            : m_Buffer(buffer)
        {
            if (!buffer.m_Validation)
                return;
            if (buffer.m_Busy.exchange(1, std::memory_order_acquire) != 0)
            {
                buffer.m_World->ReportValidation(operation, "the command buffer is used by two threads at once (one "
                                                 "thread at a time, EcsCommandBuffer.h): this call records nothing "
                                                 "and the buffer's next playback fails");
                buffer.m_LostRecording.store(true, std::memory_order_release);
                m_Refused = true;
            }
            else
            {
                m_Owned = true;
            }
        }
        RecordScope(const RecordScope&) = delete;
        RecordScope& operator=(const RecordScope&) = delete;
        ~RecordScope()
        {
            if (m_Owned)
                m_Buffer.m_Busy.store(0, std::memory_order_release);
        }
        bool Refused() const noexcept { return m_Refused; }

    private:
        CommandBufferImpl& m_Buffer;
        bool m_Owned = false;
        bool m_Refused = false;
    };

    nvrhi::FRESULT CreateCommandBufferObject(WorldImpl& world, CommandBufferWorldReference reference,
                                             ICommandBuffer** ppBuffer) noexcept
    {
        if (!ppBuffer)
        {
            world.ReportValidation("CreateCommandBuffer", "ppBuffer is null");
            return nvrhi::FE_INVALID_ARGS;
        }
        *ppBuffer = nullptr;
        CommandBufferImpl* buffer = nullptr;
        try
        {
            buffer = MAKE_RC_OBJ(CommandBufferImpl, &world, reference);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        if (!buffer)
            return nvrhi::FE_OUT_OF_MEMORY;
        *ppBuffer = buffer;
        return nvrhi::FS_OK;
    }

    CommandBufferImpl::CommandBufferImpl(WorldImpl* world, CommandBufferWorldReference reference) noexcept
        : m_WorldReference(reference == CommandBufferWorldReference::Held ? static_cast<IWorld*>(world) : nullptr)
        , m_World(world)
        , m_Validation(world->IsValidationEnabled())
    {
    }

    CommandBufferImpl::~CommandBufferImpl()
    {
        Clear();   // values never played back are destroyed with the buffer
        m_Arena.Release();
    }

    void CommandBufferImpl::RecordFailure(uint64_t sortKey, nvrhi::FRESULT result) noexcept
    {
        Command command{};
        command.sortKey = sortKey;
        command.op = CommandOp::Failed;
        command.failure = result;
        m_Commands.push_back(command);
    }

    const ComponentType* CommandBufferImpl::RequireComponent(const char* operation, ComponentId component,
                                                             uint64_t sortKey) noexcept
    {
        const ComponentType* type = m_World->GetType(component);
        if (type && type->kind != StorageKind::Resource)
            return type;
        m_World->ReportValidation(operation, "component %u is not a registered component (sort key 0x%016llx)",
                                  uint32_t(component), (unsigned long long)sortKey);
        RecordFailure(sortKey, nvrhi::FE_INVALID_ARGS);
        return nullptr;
    }

    Entity* CommandBufferImpl::CopyEntities(const Entity* entities, uint32_t count, uint64_t sortKey) noexcept
    {
        Entity* copy = static_cast<Entity*>(m_Arena.Allocate(size_t(count) * sizeof(Entity), alignof(Entity)));
        if (!copy)
        {
            RecordFailure(sortKey, nvrhi::FE_OUT_OF_MEMORY);
            return nullptr;
        }
        std::memcpy(copy, entities, size_t(count) * sizeof(Entity));
        return copy;
    }

    void* CommandBufferImpl::CopyValues(const ComponentType& type, const void* values, uint32_t count,
                                        uint64_t sortKey) noexcept
    {
        void* copy = m_Arena.Allocate(size_t(count) * type.desc.size, type.desc.alignment);
        if (!copy)
        {
            RecordFailure(sortKey, nvrhi::FE_OUT_OF_MEMORY);
            return nullptr;
        }
        ecs::CopyValues(type, copy, values, count);
        if (type.desc.destruct)
            m_OwnsDestructibleValues = true;
        return copy;
    }

    uint32_t CommandBufferImpl::HandOutPlaceholders(uint32_t count, Entity* outEntities) noexcept
    {
        const uint32_t first = m_World->GetCommandPlayback().AllocatePlaceholders(count);
        if (outEntities)
            for (uint32_t t = 0; t < count; ++t)
                outEntities[t] = MakeEntity(first + t, c_PlaceholderGeneration);
        return first;
    }

    void CommandBufferImpl::RecordEntityCommand(const char* operation, CommandOp op, const Entity* entities,
                                                uint32_t count, ComponentId component, const void* values,
                                                bool enabled, uint64_t sortKey) noexcept
    {
        if (count == 0)
            return;
        if (!entities)
        {
            m_World->ReportValidation(operation, "%u entities, but the list is null (sort key 0x%016llx)", count,
                                      (unsigned long long)sortKey);
            RecordFailure(sortKey, nvrhi::FE_INVALID_ARGS);
            return;
        }
        const ComponentType* type = nullptr;
        if (op != CommandOp::Destroy)
        {
            type = RequireComponent(operation, component, sortKey);
            if (!type)
                return;
        }

        // What the batch operation would reject before looking at an entity, and values that cannot be copied in.
        bool storeValues = false;
        if (op == CommandOp::Add || op == CommandOp::Set)
        {
            const bool noValues = type->kind == StorageKind::Buffer || type->kind == StorageKind::Chunk;
            const char* problem = nullptr;
            nvrhi::FRESULT failure = nvrhi::FE_INVALID_ARGS;
            if (noValues && (values || op == CommandOp::Set))
                problem = "takes no value here (Buffer and Chunk components)";
            else if (op == CommandOp::Set && type->desc.size != 0 && !values)
                problem = "needs values to be set";
            else
            {
                storeValues = values && type->desc.size != 0;
                if (storeValues && !type->copyable)
                {
                    problem = "cannot be copied into the buffer";
                    failure = nvrhi::FE_UNSUPPORTED;
                }
            }
            if (problem)
            {
                m_World->ReportValidation(operation, "%s %s (sort key 0x%016llx)", NameOf(*type), problem,
                                          (unsigned long long)sortKey);
                RecordFailure(sortKey, failure);
                return;
            }
        }

        Entity* list = CopyEntities(entities, count, sortKey);
        if (!list)
            return;
        void* stored = nullptr;
        if (storeValues)
        {
            stored = CopyValues(*type, values, count, sortKey);
            if (!stored)
                return;
        }
        Command command{};
        command.sortKey = sortKey;
        command.op = op;
        command.enabled = enabled;
        command.ownsValues = stored != nullptr;
        command.component = component;
        command.count = count;
        command.entities = list;
        command.values = stored;
        m_Commands.push_back(command);
    }

    void CommandBufferImpl::CreateEntities(const EntityCreateDesc& desc, uint64_t sortKey,
                                           Entity* outEntities) noexcept
    {
        const char* operation = "ICommandBuffer::CreateEntities";
        RecordScope scope(*this, operation);
        const uint32_t n = desc.entityCount;
        if (n == 0)
            return;
        // Placeholders are handed out whatever happens, so that the caller never reads undefined ids.
        const uint32_t first = HandOutPlaceholders(n, outEntities);

        // A rejected command consumes relocated values, as the batch operation would.
        auto reject = [&](nvrhi::FRESULT result) noexcept
        {
            m_World->DiscardCreateValues(desc, 0);
            RecordFailure(sortKey, result);
        };
        if (scope.Refused())
        {
            m_World->DiscardCreateValues(desc, 0);
            return;
        }
        if (desc.reservedEntities)
        {
            m_World->ReportValidation(operation, "reservedEntities must be null: reserved ids are for immediate "
                                      "creation (sort key 0x%016llx)", (unsigned long long)sortKey);
            reject(nvrhi::FE_INVALID_ARGS);
            return;
        }
        if ((desc.componentCount != 0 && !desc.components) ||
            (desc.transfer != ValueTransfer::Copy && desc.transfer != ValueTransfer::Relocate))
        {
            m_World->ReportValidation(operation, "invalid arguments (sort key 0x%016llx)", (unsigned long long)sortKey);
            reject(nvrhi::FE_INVALID_ARGS);
            return;
        }
        for (uint32_t i = 0; i < desc.componentCount; ++i)
        {
            const ComponentType* type = m_World->GetType(desc.components[i]);
            if (!type || type->kind == StorageKind::Resource)
            {
                m_World->ReportValidation(operation, "component %u is not a registered component (sort key 0x%016llx)",
                                          uint32_t(desc.components[i]), (unsigned long long)sortKey);
                reject(nvrhi::FE_INVALID_ARGS);
                return;
            }
            const void* value = desc.values ? desc.values[i] : nullptr;
            if (value && (type->kind == StorageKind::Buffer || type->kind == StorageKind::Chunk))
            {
                m_World->ReportValidation(operation, "%s takes no initial value (Buffer and Chunk components; sort key "
                                          "0x%016llx)", NameOf(*type), (unsigned long long)sortKey);
                reject(nvrhi::FE_INVALID_ARGS);
                return;
            }
            if (value && desc.transfer == ValueTransfer::Copy && !type->copyable && type->desc.size != 0)
            {
                m_World->ReportValidation(operation, "%s cannot be copied into the buffer (sort key 0x%016llx)",
                                          NameOf(*type), (unsigned long long)sortKey);
                reject(nvrhi::FE_UNSUPPORTED);   // Copy: the caller keeps its values
                return;
            }
        }

        // Every block first, so that running out of memory takes no value.
        const uint32_t c = desc.componentCount;
        ComponentId* components = nullptr;
        void** values = nullptr;
        if (c != 0)
        {
            components = static_cast<ComponentId*>(m_Arena.Allocate(sizeof(ComponentId) * c, alignof(ComponentId)));
            values = components ? static_cast<void**>(m_Arena.Allocate(sizeof(void*) * c, alignof(void*))) : nullptr;
            if (!components || !values)
            {
                reject(nvrhi::FE_OUT_OF_MEMORY);
                return;
            }
        }
        bool anyValues = false;
        for (uint32_t i = 0; i < c; ++i)
        {
            components[i] = desc.components[i];
            values[i] = nullptr;
            const ComponentType& type = *m_World->GetType(desc.components[i]);
            if (desc.values && desc.values[i] && type.desc.size != 0)
            {
                values[i] = m_Arena.Allocate(size_t(n) * type.desc.size, type.desc.alignment);
                if (!values[i])
                {
                    reject(nvrhi::FE_OUT_OF_MEMORY);
                    return;
                }
                anyValues = true;
            }
        }
        for (uint32_t i = 0; i < c; ++i)
        {
            if (!values[i])
                continue;
            const ComponentType& type = *m_World->GetType(desc.components[i]);
            TransferValues(type, values[i], desc.values[i], n, desc.transfer);
            if (type.desc.destruct)
                m_OwnsDestructibleValues = true;
        }

        Command command{};
        command.sortKey = sortKey;
        command.op = CommandOp::Create;
        command.ownsValues = anyValues;
        command.count = n;
        command.componentCount = c;
        command.placeholder = first;
        command.components = components;
        command.values = values;
        m_Commands.push_back(command);
    }

    void CommandBufferImpl::DestroyEntities(const Entity* entities, uint32_t count, uint64_t sortKey) noexcept
    {
        const char* operation = "ICommandBuffer::DestroyEntities";
        RecordScope scope(*this, operation);
        if (scope.Refused())
            return;
        RecordEntityCommand(operation, CommandOp::Destroy, entities, count, ComponentId::Invalid, nullptr, false,
                            sortKey);
    }

    void CommandBufferImpl::AddComponents(const Entity* entities, uint32_t count, ComponentId component,
                                          const void* values, uint64_t sortKey) noexcept
    {
        const char* operation = "ICommandBuffer::AddComponents";
        RecordScope scope(*this, operation);
        if (scope.Refused())
            return;
        RecordEntityCommand(operation, CommandOp::Add, entities, count, component, values, false, sortKey);
    }

    void CommandBufferImpl::RemoveComponents(const Entity* entities, uint32_t count, ComponentId component,
                                             uint64_t sortKey) noexcept
    {
        const char* operation = "ICommandBuffer::RemoveComponents";
        RecordScope scope(*this, operation);
        if (scope.Refused())
            return;
        RecordEntityCommand(operation, CommandOp::Remove, entities, count, component, nullptr, false, sortKey);
    }

    void CommandBufferImpl::SetComponents(const Entity* entities, uint32_t count, ComponentId component,
                                          const void* values, uint64_t sortKey) noexcept
    {
        const char* operation = "ICommandBuffer::SetComponents";
        RecordScope scope(*this, operation);
        if (scope.Refused())
            return;
        RecordEntityCommand(operation, CommandOp::Set, entities, count, component, values, false, sortKey);
    }

    void CommandBufferImpl::SetEnabled(const Entity* entities, uint32_t count, ComponentId component, bool enabled,
                                       uint64_t sortKey) noexcept
    {
        const char* operation = "ICommandBuffer::SetEnabled";
        RecordScope scope(*this, operation);
        if (scope.Refused())
            return;
        RecordEntityCommand(operation, CommandOp::SetEnabled, entities, count, component, nullptr, enabled, sortKey);
    }

    void CommandBufferImpl::Instantiate(Entity prefab, uint32_t count, uint64_t sortKey, Entity* outEntities) noexcept
    {
        RecordScope scope(*this, "ICommandBuffer::Instantiate");
        if (count == 0)
            return;
        const uint32_t first = HandOutPlaceholders(count, outEntities);
        if (scope.Refused())
            return;
        Command command{};
        command.sortKey = sortKey;
        command.op = CommandOp::Instantiate;
        command.count = count;
        command.placeholder = first;
        command.entity = prefab;
        m_Commands.push_back(command);
    }

    void CommandBufferImpl::AppendToBuffer(Entity entity, ComponentId component, const void* elements, uint32_t count,
                                           uint64_t sortKey) noexcept
    {
        const char* operation = "ICommandBuffer::AppendToBuffer";
        RecordScope scope(*this, operation);
        if (scope.Refused() || count == 0)
            return;
        if (!elements)
        {
            m_World->ReportValidation(operation, "%u elements, but the array is null (sort key 0x%016llx)", count,
                                      (unsigned long long)sortKey);
            RecordFailure(sortKey, nvrhi::FE_INVALID_ARGS);
            return;
        }
        const ComponentType* type = RequireComponent(operation, component, sortKey);
        if (!type)
            return;
        if (type->kind != StorageKind::Buffer || !type->copyable)
        {
            const bool buffer = type->kind == StorageKind::Buffer;
            m_World->ReportValidation(operation, "%s %s (sort key 0x%016llx)", NameOf(*type),
                                      buffer ? "cannot be copied into the buffer" : "is not a Buffer",
                                      (unsigned long long)sortKey);
            RecordFailure(sortKey, buffer ? nvrhi::FE_UNSUPPORTED : nvrhi::FE_INVALID_ARGS);
            return;
        }
        void* stored = CopyValues(*type, elements, count, sortKey);
        if (!stored)
            return;
        Command command{};
        command.sortKey = sortKey;
        command.op = CommandOp::Append;
        command.ownsValues = true;
        command.component = component;
        command.count = count;
        command.entity = entity;
        command.values = stored;
        m_Commands.push_back(command);
    }

    void CommandBufferImpl::Reset() noexcept
    {
        RecordScope scope(*this, "ICommandBuffer::Reset");
        if (scope.Refused())
            return;
        Clear();
    }

    void CommandBufferImpl::Clear() noexcept
    {
        DestroyOwnedValues();
        m_Commands.clear();
        m_Arena.Reset();
        m_OwnsDestructibleValues = false;
    }

    void CommandBufferImpl::DestroyOwnedValues() noexcept
    {
        if (!m_OwnsDestructibleValues)
            return;   // nothing recorded needs a destruct hook
        for (Command& command : m_Commands)
        {
            if (!command.ownsValues)
                continue;
            command.ownsValues = false;
            if (command.op == CommandOp::Create)
            {
                void** values = static_cast<void**>(command.values);
                for (uint32_t i = 0; i < command.componentCount; ++i)
                    if (values[i])
                        DestructValues(*m_World->GetType(command.components[i]), values[i], command.count);
            }
            else if (command.values)
            {
                DestructValues(*m_World->GetType(command.component), command.values, command.count);
            }
        }
    }

    // --------------------------------------------------------------------------------------------- playback

    namespace
    {
        // One PlayCommandBuffers call: merge, placeholder ranges, then the commands in order.
        //
        // The result is the failure of the earliest failing command in merged order, so that it equals applying the
        // commands one by one: each failure is kept with its command's position, and the smallest position wins (the
        // first recorded on a tie). A batch's failure takes the position of its first command, which is exact because
        // only its first command may carry a failure of its own (a command that failed when prepared starts a new run).
        class PlaybackRun
        {
        public:
            PlaybackRun(WorldImpl& world, CommandBufferImpl* const* buffers, uint32_t count, uint64_t version,
                        PlaceholderTable& placeholders) noexcept
                : m_World(world)
                , m_State(world.GetCommandPlayback())
                , m_Table(placeholders)
                , m_Buffers(buffers)
                , m_BufferCount(count)
                , m_Version(version)
            {
            }

            nvrhi::FRESULT Run(uint64_t& outCommands) noexcept
            {
                // A call that a second thread lost (validation) fails the buffer's playback, before every command.
                for (uint32_t b = 0; b < m_BufferCount; ++b)
                    if (m_Buffers[b]->TakeLostRecording())
                        Fail(nvrhi::FE_INVALID_ARGS, 0);
                Merge();
                RegisterPlaceholders();
                const size_t n = m_State.entries.size();
                size_t i = 0;
                while (i < n)
                {
                    Command& lead = At(i);
                    Prepare(lead);
                    size_t end = i + 1;
                    if (IsCoalescable(lead.op))
                    {
                        // A command that failed when prepared starts a run of its own, so that its failure keeps its
                        // place before the failures of the commands after it.
                        while (end < n)
                        {
                            Command& next = At(end);
                            if (!SameBatch(lead, next))
                                break;
                            Prepare(next);
                            if (NVRHI_FAILED(next.failure))
                                break;
                            ++end;
                        }
                    }
                    Apply(i, end);
                    i = end;
                }
                for (uint32_t b = 0; b < m_BufferCount; ++b)
                    m_Buffers[b]->Clear();
                ++m_State.counters.playbacks;
                m_State.counters.commands += n;
                outCommands = n;
                return m_Result;
            }

        private:
            Command& At(size_t entry) const noexcept
            {
                const CommandPlayback::Entry& e = m_State.entries[entry];
                return m_Buffers[e.buffer]->GetCommands()[e.command];
            }

            void Fail(nvrhi::FRESULT result, size_t position) noexcept
            {
                if (NVRHI_FAILED(result) && position < m_FailurePosition)
                {
                    m_Result = result;
                    m_FailurePosition = position;
                }
            }

            // Every command in (sort key, buffer position, recording order), which no thread timing can change once
            // the keys are unique per system (EcsCommandBuffer.h).
            void Merge() noexcept
            {
                Vector<CommandPlayback::Entry>& entries = m_State.entries;
                entries.clear();
                size_t total = 0;
                for (uint32_t b = 0; b < m_BufferCount; ++b)
                    total += m_Buffers[b]->GetCommandCount();
                entries.reserve(total);
                for (uint32_t b = 0; b < m_BufferCount; ++b)
                {
                    const Command* commands = m_Buffers[b]->GetCommands();
                    const uint32_t n = m_Buffers[b]->GetCommandCount();
                    for (uint32_t c = 0; c < n; ++c)
                        entries.push_back(CommandPlayback::Entry{ commands[c].sortKey, b, c });
                }
                bool sorted = true;
                for (size_t k = 1; k < entries.size() && sorted; ++k)
                    sorted = !EntryLess(entries[k], entries[k - 1]);
                if (!sorted)
                {
                    std::sort(entries.begin(), entries.end(), EntryLess);
                    ++m_State.counters.sorts;
                }

                // Validation: equal keys in two buffers. The failure takes the place of the first command with the key.
                if (!m_World.IsValidationEnabled())
                    return;
                uint32_t clashes = 0;
                size_t firstClash = 0;
                for (size_t k = 1; k < entries.size(); ++k)
                {
                    if (entries[k].key == entries[k - 1].key && entries[k].buffer != entries[k - 1].buffer)
                    {
                        if (clashes++ == 0)
                        {
                            firstClash = k - 1;
                            while (firstClash > 0 && entries[firstClash - 1].key == entries[k].key)
                                --firstClash;
                        }
                    }
                }
                if (clashes != 0)
                {
                    m_World.ReportValidation("Playback", "sort key 0x%016llx (and %u more) is used by commands of two "
                                             "buffers: their order would follow scheduling; keys must be unique per "
                                             "system (6.5)", (unsigned long long)entries[firstClash].key, clashes - 1);
                    Fail(nvrhi::FE_INVALID_ARGS, firstClash);
                }
            }

            // The placeholder ranges of the creating commands, each with its slots in the table, all Null until the
            // command runs.
            void RegisterPlaceholders() noexcept
            {
                const Vector<CommandPlayback::Entry>& entries = m_State.entries;
                size_t created = 0;
                for (size_t k = 0; k < entries.size(); ++k)
                {
                    const Command& command = At(k);
                    if (command.op == CommandOp::Create || command.op == CommandOp::Instantiate)
                        created += command.count;
                }
                if (created == 0)
                    return;
                size_t offset = m_Table.resolved.size();
                NVRHI_ASSERT(offset + created <= 0xffffffffull);
                m_Table.resolved.resize(offset + created);
                for (size_t k = 0; k < entries.size(); ++k)
                {
                    Command& command = At(k);
                    if (command.op != CommandOp::Create && command.op != CommandOp::Instantiate)
                        continue;
                    command.resolved = uint32_t(offset);
                    // A range that wraps past 2^32 is split in two.
                    const uint64_t end = uint64_t(command.placeholder) + command.count;
                    if (end <= 0x100000000ull)
                    {
                        m_Table.ranges.push_back(
                            PlaceholderTable::Range{ command.placeholder, command.count, uint32_t(offset), 0 });
                    }
                    else
                    {
                        const uint32_t head = uint32_t(0x100000000ull - command.placeholder);
                        m_Table.ranges.push_back(
                            PlaceholderTable::Range{ command.placeholder, head, uint32_t(offset), 0 });
                        m_Table.ranges.push_back(
                            PlaceholderTable::Range{ 0, command.count - head, uint32_t(offset + head), 0 });
                    }
                    offset += command.count;
                }
                std::sort(m_Table.ranges.begin(), m_Table.ranges.end(),
                          [](const PlaceholderTable::Range& a, const PlaceholderTable::Range& b)
                          { return a.first < b.first; });
                if (!m_World.IsValidationEnabled())
                    return;
                for (size_t k = 1; k < m_Table.ranges.size(); ++k)
                {
                    const PlaceholderTable::Range& a = m_Table.ranges[k - 1];
                    if (uint64_t(a.first) + a.count > m_Table.ranges[k].first)
                    {
                        m_World.ReportValidation("Playback", "placeholder %u is handed out twice: a buffer held it "
                                                 "while 2^32 more were recorded", m_Table.ranges[k].first);
                        Fail(nvrhi::FE_INVALID_ARGS, 0);
                        break;
                    }
                }
            }

            void ReportUnresolved(const Command& command, uint32_t count, bool inValues) noexcept
            {
                m_World.ReportValidation("Playback", inValues ?
                    "the values of %s at sort key 0x%016llx hold %u placeholder(s) that no earlier command of the "
                    "buffers played back together created; they are written as Entity::Null (6.5)" :
                    "%s at sort key 0x%016llx names %u placeholder(s) that no earlier command of the buffers played "
                    "back together created; they are skipped as stale (6.5)",
                    OpName(command.op), (unsigned long long)command.sortKey, count);
            }

            // Placeholders in an entity list become their entities; those that do not resolve stay placeholders,
            // which the batch operations skip as stale (and report).
            Entity ResolveEntity(Entity entity, uint32_t& unresolved) const noexcept
            {
                if (!IsPlaceholder(entity))
                    return entity;
                const Entity resolved = m_Table.Resolve(entity);
                if (resolved == Entity::Null)
                {
                    ++unresolved;
                    return entity;
                }
                return resolved;
            }

            // Entity fields of `count` values that hold placeholders: the entity it became, or Null, which fails the
            // command at its place.
            void RewriteValues(Command& command, const ComponentType& type, void* values, uint32_t count) noexcept
            {
                if (type.entityOffsets.empty() || !values)
                    return;
                uint32_t unresolved = 0;
                for (uint32_t k = 0; k < count; ++k)
                {
                    uint8_t* value = static_cast<uint8_t*>(values) + size_t(k) * type.desc.size;
                    for (const uint32_t offset : type.entityOffsets)
                    {
                        Entity entity;
                        std::memcpy(&entity, value + offset, sizeof(entity));
                        if (!IsPlaceholder(entity))
                            continue;
                        entity = m_Table.Resolve(entity);
                        if (entity == Entity::Null)
                            ++unresolved;
                        std::memcpy(value + offset, &entity, sizeof(entity));
                    }
                }
                if (unresolved != 0)
                {
                    ReportUnresolved(command, unresolved, true);
                    command.failure = nvrhi::FE_INVALID_ARGS;
                }
            }

            // Resolves the placeholders a command names, once, just before it runs: the table only changes when a
            // creating command runs, and a creating command never joins a batch.
            void Prepare(Command& command) noexcept
            {
                if (command.prepared || command.op == CommandOp::Failed)
                    return;
                command.prepared = true;
                uint32_t unresolved = 0;
                switch (command.op)
                {
                case CommandOp::Create:
                {
                    void** values = static_cast<void**>(command.values);
                    for (uint32_t i = 0; i < command.componentCount; ++i)
                        if (values[i])
                            RewriteValues(command, *m_World.GetType(command.components[i]), values[i], command.count);
                    return;
                }
                case CommandOp::Instantiate:
                case CommandOp::Append:
                    command.entity = ResolveEntity(command.entity, unresolved);
                    if (unresolved != 0)
                        ReportUnresolved(command, unresolved, false);
                    if (command.op == CommandOp::Append)
                        RewriteValues(command, *m_World.GetType(command.component), command.values, command.count);
                    return;
                default:
                    for (uint32_t t = 0; t < command.count; ++t)
                        command.entities[t] = ResolveEntity(command.entities[t], unresolved);
                    if (unresolved != 0)
                        ReportUnresolved(command, unresolved, false);
                    if (command.values)
                        RewriteValues(command, *m_World.GetType(command.component), command.values, command.count);
                    return;
                }
            }

            void Apply(size_t first, size_t end) noexcept
            {
                Command& lead = At(first);
                Fail(lead.failure, first);   // a Failed command, or a failure found when it was prepared
                switch (lead.op)
                {
                case CommandOp::Failed:
                    return;
                case CommandOp::Create:
                {
                    EntityCreateDesc desc{};
                    desc.components = lead.components;
                    desc.values = static_cast<void**>(lead.values);
                    desc.componentCount = lead.componentCount;
                    desc.entityCount = lead.count;
                    desc.transfer = ValueTransfer::Relocate;
                    lead.ownsValues = false;   // consumed by the operation, whatever happens
                    Batch();
                    Fail(m_World.CreateEntitiesInternal(desc, m_Table.resolved.data() + lead.resolved, m_Version),
                         first);
                    return;
                }
                case CommandOp::Instantiate:
                    Batch();
                    Fail(m_World.InstantiateInternal(lead.entity, lead.count, m_Table.resolved.data() + lead.resolved,
                                                     m_Version), first);
                    return;
                case CommandOp::Append:
                    lead.ownsValues = false;   // consumed by the operation, whatever happens
                    Batch();
                    Fail(m_World.AppendToBufferInternal(lead.entity, lead.component, lead.values, lead.count,
                                                        ValueTransfer::Relocate, m_Version), first);
                    return;
                default:
                    ApplyRun(first, end);
                    return;
                }
            }

            // A run of commands with the same operation, component, value mode and flag: one batch over their
            // concatenated entities and values, or one per command when the values cannot be gathered.
            void ApplyRun(size_t first, size_t end) noexcept
            {
                Command& lead = At(first);
                if (end - first == 1)
                {
                    lead.ownsValues = false;
                    ApplyBatch(lead, lead.entities, lead.count, lead.values, first);
                    return;
                }
                const ComponentType* type = lead.op == CommandOp::Destroy ? nullptr : m_World.GetType(lead.component);
                uint64_t total = 0;
                for (size_t k = first; k < end; ++k)
                    total += At(k).count;
                const size_t valueBytes = lead.values ? size_t(total) * type->desc.size : 0;
                uint8_t* gather = valueBytes != 0 ? m_State.GatherBytes(valueBytes) : nullptr;
                if (total > 0xffffffffull || (valueBytes != 0 && !gather))
                {
                    for (size_t k = first; k < end; ++k)
                    {
                        Command& command = At(k);
                        command.ownsValues = false;
                        ApplyBatch(command, command.entities, command.count, command.values, k);
                    }
                    return;
                }
                Vector<Entity>& entities = m_State.runEntities;
                entities.resize(size_t(total));
                size_t done = 0;
                for (size_t k = first; k < end; ++k)
                {
                    Command& command = At(k);
                    std::memcpy(entities.data() + done, command.entities, sizeof(Entity) * command.count);
                    if (gather)
                        RelocateValues(*type, gather + done * type->desc.size, command.values, command.count);
                    command.ownsValues = false;   // moved into the batch, which the operation consumes
                    done += command.count;
                }
                m_State.counters.coalesced += end - first - 1;
                ApplyBatch(lead, entities.data(), uint32_t(total), gather, first);
            }

            // One *Internal call. E1.4: the Observe* reports of one call are contiguous and in playback order.
            void ApplyBatch(const Command& lead, const Entity* entities, uint32_t count, void* values,
                            size_t position) noexcept
            {
                Batch();
                nvrhi::FRESULT result = nvrhi::FS_OK;
                switch (lead.op)
                {
                case CommandOp::Destroy:
                    result = m_World.DestroyEntitiesInternal(entities, count, m_Version);
                    break;
                case CommandOp::Add:
                    result = m_World.AddComponentsInternal(entities, count, lead.component, values,
                        values ? ValueTransfer::Relocate : ValueTransfer::Copy, m_Version);
                    break;
                case CommandOp::Remove:
                    result = m_World.RemoveComponentsInternal(entities, count, lead.component, m_Version);
                    break;
                case CommandOp::Set:
                    result = m_World.SetComponentsInternal(entities, count, lead.component, values,
                        ValueTransfer::Relocate, m_Version);
                    break;
                case CommandOp::SetEnabled:
                    result = m_World.SetEnabledInternal(entities, count, lead.component, lead.enabled, m_Version);
                    break;
                default:
                    NVRHI_ASSERT(false);
                    break;
                }
                Fail(result, position);
            }

            void Batch() noexcept { ++m_State.counters.batches; }

            WorldImpl& m_World;
            CommandPlayback& m_State;
            PlaceholderTable& m_Table;
            CommandBufferImpl* const* m_Buffers;
            uint32_t m_BufferCount;
            uint64_t m_Version;
            nvrhi::FRESULT m_Result = nvrhi::FS_OK;
            size_t m_FailurePosition = ~size_t(0);
        };
    }

    nvrhi::FRESULT PlayCommandBuffers(WorldImpl& world, CommandBufferImpl* const* buffers, uint32_t count,
                                      uint64_t version, PlaceholderTable& placeholders, uint64_t& outCommands) noexcept
    {
        PlaybackRun run(world, buffers, count, version, placeholders);
        return run.Run(outCommands);
    }

    // ------------------------------------------------------------------------------------ IWorld entry points

    nvrhi::FRESULT WorldImpl::CreateCommandBuffer(ICommandBuffer** ppBuffer) noexcept
    {
        return CreateCommandBufferObject(*this, CommandBufferWorldReference::Held, ppBuffer);
    }

    nvrhi::FRESULT WorldImpl::Playback(ICommandBuffer* const* buffers, uint32_t count) noexcept
    {
        // The rules of EcsWorld.h. Called at the wrong time, nothing is applied or reset.
        const char* operation = "Playback";
        if (!CanChangeStructure())
        {
            ReportValidation(operation, "structural changes are deferred inside phases and observers (3.3.3): play "
                             "command buffers back at a sync point");
            return c_WrongTimeResult;
        }
        if (count != 0 && !buffers)
        {
            ReportValidation(operation, "%u buffers, but the list is null", count);
            return nvrhi::FE_INVALID_ARGS;
        }
        // An entry that cannot be played back is skipped (and left as it is) and fails the call; the other buffers
        // still apply and are reset. A buffer listed twice is played once.
        CommandPlayback& state = *m_CommandPlayback;
        state.buffers.clear();
        nvrhi::FRESULT result = nvrhi::FS_OK;
        for (uint32_t i = 0; i < count; ++i)
        {
            CommandBufferImpl* buffer = donut::query_cast<CommandBufferImpl>(buffers[i]);
            const char* problem = nullptr;
            if (!buffer)
                problem = "is null or not a command buffer of this module";
            else if (buffer->GetWorld() != this)
                problem = "belongs to another world";
            else if (buffer->IsBusy())
                problem = "is being recorded by another thread";
            else if (std::find(state.buffers.begin(), state.buffers.end(), buffer) != state.buffers.end())
                problem = "is listed twice; it is played back once";
            if (problem)
            {
                ReportValidation(operation, "buffer %u %s", i, problem);
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                continue;
            }
            state.buffers.push_back(buffer);
        }

        state.placeholders.Clear();   // valid until the next Playback starts (EcsWorld.h)
        uint64_t commands = 0;
        KeepFirstFailure(result, PlayCommandBuffers(*this, state.buffers.data(), uint32_t(state.buffers.size()),
                                                    GetStampVersion(), state.placeholders, commands));
        m_CommandsPlayedBack += commands;
        // Once for the whole playback, as an immediate call flushes once: the observers' commands follow the system's.
        KeepFirstFailure(result, FlushObservers());
        return result;
    }

    Entity WorldImpl::ResolvePlaceholder(Entity placeholder) noexcept
    {
        if (!IsPlaceholder(placeholder))
            return Entity::Null;
        const Entity entity = m_CommandPlayback->placeholders.Resolve(placeholder);
        return FindAlive(entity) ? entity : Entity::Null;
    }
}
