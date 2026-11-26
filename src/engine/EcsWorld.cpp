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

// The world object: registry, entity table, resources and the immediate operations of IWorld (aggregate
// docs/ecs-design.md 4.1, 4.2, 4.9, 4.10, 9.1, 11.3; plan step E1.1).

#include "EcsWorldImpl.h"
#include "EcsQueryImpl.h"

#include <donut/core/log.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <new>

namespace donut::engine::ecs
{
    namespace
    {
        constexpr ComponentFlags c_KnownFlags = ComponentFlags::Tag | ComponentFlags::Sparse | ComponentFlags::Shared |
            ComponentFlags::Chunk | ComponentFlags::Enableable | ComponentFlags::Buffer | ComponentFlags::Relocatable |
            ComponentFlags::GpuMirrored | ComponentFlags::NoSerialize | ComponentFlags::Resource;

        constexpr uint32_t c_LayoutClash = 0xfffffffeu;   // in the CopyEntitiesFrom type map

        bool Has(ComponentFlags flags, ComponentFlags flag) noexcept { return !!(flags & flag); }

        bool IsPowerOfTwo(uint32_t value) noexcept { return value != 0 && (value & (value - 1)) == 0; }

        uint32_t ScalarSize(FieldType type) noexcept
        {
            switch (type)
            {
            case FieldType::Boolean:
            case FieldType::Int8:
            case FieldType::UInt8:
            case FieldType::Opaque:
                return 1;
            case FieldType::Int16:
            case FieldType::UInt16:
                return 2;
            case FieldType::Int32:
            case FieldType::UInt32:
            case FieldType::Float:
                return 4;
            case FieldType::Int64:
            case FieldType::UInt64:
            case FieldType::Double:
            case FieldType::Entity:
                return 8;
            default:
                return 0;
            }
        }

        uint64_t HashGuid(const nvrhi::GUID& uuid) noexcept { return HashBytes(&uuid, sizeof(uuid)); }

        // The same type, as two modules or two worlds see it: what decides the storage and the reflected fields. Names
        // and hook addresses differ between modules; whether a hook exists does not.
        bool SameLayout(const ComponentTypeDesc& a, const ComponentTypeDesc& b) noexcept
        {
            if (a.size != b.size || a.alignment != b.alignment || a.flags != b.flags ||
                a.bufferInlineCapacity != b.bufferInlineCapacity || a.fieldCount != b.fieldCount)
                return false;
            if (!a.construct != !b.construct || !a.destruct != !b.destruct || !a.relocate != !b.relocate ||
                !a.copy != !b.copy)
                return false;
            for (uint32_t i = 0; i < a.fieldCount; ++i)
            {
                const FieldDesc& x = a.fields[i];
                const FieldDesc& y = b.fields[i];
                if (x.offset != y.offset || x.scalarCount != y.scalarCount || x.type != y.type || x.shape != y.shape)
                    return false;
            }
            return true;
        }

        StorageKind KindOf(ComponentFlags flags) noexcept
        {
            if (Has(flags, ComponentFlags::Resource)) return StorageKind::Resource;
            if (Has(flags, ComponentFlags::Sparse)) return StorageKind::Sparse;
            if (Has(flags, ComponentFlags::Shared)) return StorageKind::Shared;
            if (Has(flags, ComponentFlags::Chunk)) return StorageKind::Chunk;
            if (Has(flags, ComponentFlags::Buffer)) return StorageKind::Buffer;
            if (Has(flags, ComponentFlags::Tag)) return StorageKind::Tag;
            return StorageKind::Data;
        }

        const char* NameOf(const ComponentTypeDesc& desc) noexcept { return desc.name ? desc.name : "(unnamed)"; }

        // Whether values of the type may be copied: a copy hook, or memcpy for a Relocatable type, or no bytes at all.
        // Never memcpy for any other type.
        bool IsCopyable(const ComponentTypeDesc& desc) noexcept
        {
            return desc.size == 0 || desc.copy != nullptr || Has(desc.flags, ComponentFlags::Relocatable);
        }

        constexpr uint64_t AlignUp64(uint64_t value, uint64_t alignment) noexcept
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        // The bytes of a chunk that holds one row of an archetype with only this component (EcsArchetype.cpp's
        // layout), in 64 bits.
        uint64_t OneRowBytes(const ComponentTypeDesc& desc) noexcept
        {
            const StorageKind kind = KindOf(desc.flags);
            uint64_t bytes = sizeof(Entity);
            if (kind == StorageKind::Data)
                bytes = AlignUp64(bytes, c_ColumnAlignment) + desc.size;
            else if (kind == StorageKind::Buffer)
                bytes = AlignUp64(AlignUp64(bytes, c_ColumnAlignment) + sizeof(BufferHeader), c_ColumnAlignment) +
                        uint64_t(desc.size) * desc.bufferInlineCapacity;
            bytes = AlignUp64(bytes, sizeof(uint64_t));
            if (Has(desc.flags, ComponentFlags::Enableable))
                bytes += sizeof(uint64_t);
            bytes += 2 * sizeof(uint64_t);   // the column's version and the added version
            if (kind == StorageKind::Chunk)
                bytes = AlignUp64(bytes, desc.alignment) + desc.size;
            return bytes;
        }

        void KeepFirstFailure(nvrhi::FRESULT& result, nvrhi::FRESULT next) noexcept
        {
            if (NVRHI_SUCCEEDED(result) && NVRHI_FAILED(next))
                result = next;
        }

        constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment) noexcept
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }
    }

    // ------------------------------------------------------------------------------------------- lifetime

    WorldImpl::WorldImpl(const WorldDesc& desc) noexcept
        : m_Desc(desc)
    {
        m_DebugName = desc.debugName ? desc.debugName : "";
        m_Desc.debugName = m_DebugName.c_str();
    }

    nvrhi::FRESULT WorldImpl::Initialize() noexcept
    {
        m_SmallPages.Initialize(m_Desc.chunkSize);
        m_LargePages.Initialize(m_Desc.largeChunkSize);
        m_Entities.reserve(size_t(m_Desc.initialEntityCapacity) + 1);
        m_Entities.push_back(EntityRecord{ 0, c_FreeSlot, 0, 0, 0 });   // index 0 is never handed out
        // Archetype 0 holds the entities without components.
        if (FindOrCreateArchetype(nullptr, 0, nullptr) != 0)
            return nvrhi::FE_OUT_OF_MEMORY;
        // Before any buffer exists: recording threads take placeholders from it (EcsCommandBuffer.cpp).
        m_CommandPlayback = CreateCommandPlayback();
        if (!m_CommandPlayback)
            return nvrhi::FE_OUT_OF_MEMORY;
        return nvrhi::FS_OK;
    }

    WorldImpl::~WorldImpl()
    {
        // First, while the component types exist: the observers' buffers destroy unplayed values with their hooks.
        DestroyObserverRegistry(m_ObserverRegistry);
        m_ObserverRegistry = nullptr;
        m_ObserverCount = 0;
        for (Archetype* archetype : m_Archetypes)
        {
            for (uint32_t chunk = 0; chunk < archetype->GetChunkCount(); ++chunk)
                for (uint32_t row = 0; row < archetype->chunks[chunk].count; ++row)
                    DestructRow(*archetype, chunk, row);
            while (!archetype->chunks.empty())
            {
                archetype->chunks.back().count = 0;
                ReleaseLastChunk(*archetype);
            }
            for (const ArchetypeColumn& column : archetype->columns)
                if (column.kind == StorageKind::Shared)
                    DestructValues(*column.type, archetype->sharedValues + column.offset, 1);
            HeapFree(archetype->sharedValues);
            archetype->~Archetype();
            HeapFree(archetype);
        }
        m_Archetypes.clear();

        for (SparseSet* set : m_SparseSets)
        {
            DestructValues(*set->type, set->values, set->GetCount());
            HeapFree(set->values);
            for (uint32_t* page : set->pages)
                HeapFree(page);
            for (const Vector<uint64_t*>& chunks : set->presence)
                for (uint64_t* mask : chunks)
                    HeapFree(mask);
            set->~SparseSet();
            HeapFree(set);
        }
        m_SparseSets.clear();

        for (ComponentType* type : m_Types)
        {
            if (type->value)
            {
                DestructValues(*type, type->value, 1);
                HeapFree(type->value);
            }
            type->~ComponentType();
            HeapFree(type);
        }
        m_Types.clear();

        m_SmallPages.Release();
        m_LargePages.Release();
        HeapFree(m_ScratchBlock);
        DestroyCommandPlayback(m_CommandPlayback);
    }

    // ------------------------------------------------------------------------------------------- validation

    void WorldImpl::ReportValidation(const char* operation, const char* format, ...) noexcept
    {
        if (!m_Desc.validation)
            return;
        m_ValidationErrors.fetch_add(1, std::memory_order_relaxed);
        char message[512];
        va_list args;
        va_start(args, format);
        vsnprintf(message, sizeof(message), format, args);
        va_end(args);
        log::warning("ECS world '%s', %s: %s", m_DebugName.c_str(), operation, message);
    }

    void WorldImpl::ReportStale(const char* operation, Entity entity) noexcept
    {
        if (!m_Desc.validation)
            return;
        ReportValidation(operation, "entity %u:%u is null, stale, or not created yet", GetEntityIndex(entity),
                         GetEntityGeneration(entity));
    }

    // --------------------------------------------------------------------------------------------- registry

    uint32_t WorldImpl::FindType(const nvrhi::GUID& uuid) const noexcept
    {
        return m_TypeIndex.Find(HashGuid(uuid),
                                [&](uint32_t index) noexcept { return m_Types[index]->desc.uuid == uuid; });
    }

    nvrhi::FRESULT WorldImpl::ValidateTypeDesc(const ComponentTypeDesc& desc) noexcept
    {
        const char* operation = "RegisterComponent";
        const ComponentFlags flags = desc.flags;
        auto fail = [&](const char* reason) noexcept
        {
            ReportValidation(operation, "%s: %s", NameOf(desc), reason);
            return nvrhi::FE_INVALID_ARGS;
        };

        if (desc.uuid == nvrhi::GUID{})
            return fail("the GUID is null");
        if (Has(flags, ~c_KnownFlags))
            return fail("unknown flags");
        if (!IsPowerOfTwo(desc.alignment) || desc.alignment > c_ColumnAlignment)
            return fail("the alignment must be a power of two up to 64");
        const bool tag = Has(flags, ComponentFlags::Tag);
        if (tag != (desc.size == 0))
            return fail("a Tag has size 0, and only a Tag");
        if (desc.size % desc.alignment != 0)
            return fail("the size must be a multiple of the alignment");
        const int storage = int(Has(flags, ComponentFlags::Sparse)) + int(Has(flags, ComponentFlags::Shared)) +
            int(Has(flags, ComponentFlags::Chunk)) + int(Has(flags, ComponentFlags::Buffer)) +
            int(Has(flags, ComponentFlags::Resource));
        if (storage > 1)
            return fail("Sparse, Shared, Chunk, Buffer and Resource exclude each other");
        if (tag && (Has(flags, ComponentFlags::Shared) || Has(flags, ComponentFlags::Chunk) ||
                    Has(flags, ComponentFlags::Buffer) || Has(flags, ComponentFlags::Resource)))
            return fail("a Tag cannot be Shared, Chunk, a Buffer or a Resource");
        if (Has(flags, ComponentFlags::Enableable) &&
            (Has(flags, ComponentFlags::Shared) || Has(flags, ComponentFlags::Chunk) ||
             Has(flags, ComponentFlags::Sparse) || Has(flags, ComponentFlags::Resource)))
            return fail("Enableable needs a component stored per row");
        if (Has(flags, ComponentFlags::GpuMirrored) && !Has(flags, ComponentFlags::Relocatable))
            return fail("GpuMirrored needs Relocatable (D14)");
        // Hooks and Relocatable agree (D14): memcpy moves and copies a Relocatable type, so it has no destruct,
        // relocate or copy hook; a type that owns something (a destruct hook) says how it moves; and a type that is
        // neither Relocatable nor has a copy hook is never copied (FE_UNSUPPORTED), never memcpy'd.
        if (Has(flags, ComponentFlags::Relocatable) && (desc.destruct || desc.relocate || desc.copy))
            return fail("a Relocatable type has no destruct, relocate or copy hook (memcpy moves and copies it)");
        if (desc.destruct && !desc.relocate)
            return fail("a type with a destruct hook needs a relocate hook");
        if (!Has(flags, ComponentFlags::Buffer) && desc.bufferInlineCapacity != 0)
            return fail("bufferInlineCapacity is for Buffer components");
        if (Has(flags, ComponentFlags::Shared) && desc.fieldCount == 0)
            return fail("a Shared component must list its fields (4.4)");
        if (Has(flags, ComponentFlags::Shared) && !IsCopyable(desc))
            return fail("a Shared component must be copyable: Relocatable, or a copy hook");
        // One row of the component alone must fit the largest chunk. In 64 bits: size * inline capacity can exceed 32.
        if (OneRowBytes(desc) > m_Desc.largeChunkSize)
            return fail("one row of it does not fit the largest chunk");
        if (desc.fieldCount != 0 && !desc.fields)
            return fail("fieldCount without fields");
        for (uint32_t i = 0; i < desc.fieldCount; ++i)
        {
            const FieldDesc& field = desc.fields[i];
            const uint32_t scalar = ScalarSize(field.type);
            if (scalar == 0 || field.scalarCount == 0 || uint8_t(field.shape) > uint8_t(FieldShape::Array))
                return fail("a field has an invalid type, shape or count");
            if (uint64_t(field.offset) + uint64_t(scalar) * field.scalarCount > desc.size)
                return fail("a field lies outside the value");
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldImpl::AddType(const ComponentTypeDesc& desc, const void* initialValue,
                                      ComponentId& outId) noexcept
    {
        void* memory = HeapAllocate(sizeof(ComponentType), alignof(ComponentType));
        if (!memory)
            return nvrhi::FE_OUT_OF_MEMORY;
        ComponentType* type = new (memory) ComponentType();
        type->desc = desc;
        type->id = ComponentId(uint32_t(m_Types.size()));
        type->kind = KindOf(desc.flags);
        type->enableable = Has(desc.flags, ComponentFlags::Enableable);
        type->copyable = IsCopyable(desc);

        // The bytes the fields cover, merged and ascending; the Entity scalars.
        for (uint32_t i = 0; i < desc.fieldCount; ++i)
        {
            const FieldDesc& field = desc.fields[i];
            const uint32_t scalar = ScalarSize(field.type);
            type->fieldBytes.push_back(ByteRange{ field.offset, scalar * field.scalarCount });
            if (field.type == FieldType::Entity)
                for (uint32_t k = 0; k < field.scalarCount; ++k)
                    type->entityOffsets.push_back(field.offset + k * uint32_t(sizeof(Entity)));
        }
        std::sort(type->fieldBytes.begin(), type->fieldBytes.end(), [](const ByteRange& a, const ByteRange& b)
        {
            return a.offset < b.offset || (a.offset == b.offset && a.size < b.size);
        });
        std::sort(type->entityOffsets.begin(), type->entityOffsets.end());
        uint32_t merged = 0;
        for (uint32_t i = 0; i < type->fieldBytes.size(); ++i)
        {
            const ByteRange range = type->fieldBytes[i];
            if (merged != 0)
            {
                ByteRange& last = type->fieldBytes[merged - 1];
                if (range.offset <= last.offset + last.size)
                {
                    const uint32_t end = std::max(last.offset + last.size, range.offset + range.size);
                    last.size = end - last.offset;
                    continue;
                }
            }
            type->fieldBytes[merged++] = range;
        }
        type->fieldBytes.resize(merged);

        if (type->kind == StorageKind::Sparse)
        {
            void* setMemory = HeapAllocate(sizeof(SparseSet), alignof(SparseSet));
            if (!setMemory)
            {
                type->~ComponentType();
                HeapFree(type);
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            SparseSet* set = new (setMemory) SparseSet();
            set->type = type;
            type->sparseSet = uint32_t(m_SparseSets.size());
            m_SparseSets.push_back(set);
        }
        else if (type->kind == StorageKind::Resource || type->kind == StorageKind::Shared)
        {
            // A resource's value, or a Shared component's default value (its key when it is added without one).
            type->value = HeapAllocate(desc.size, desc.alignment);
            if (!type->value)
            {
                type->~ComponentType();
                HeapFree(type);
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            if (initialValue)
                CopyValues(*type, type->value, initialValue, 1);
            else
                ConstructValues(*type, type->value, 1);
        }

        m_Types.push_back(type);
        m_TypeIndex.Insert(HashGuid(desc.uuid), uint32_t(type->id));
        outId = type->id;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldImpl::RegisterComponent(const ComponentTypeDesc& desc, ComponentId& outId) noexcept
    {
        outId = ComponentId::Invalid;
        const uint32_t existing = FindType(desc.uuid);
        if (existing != HashIndex::c_NotFound)
        {
            if (!SameLayout(m_Types[existing]->desc, desc))
            {
                ReportValidation("RegisterComponent", "%s: the GUID is registered with another layout", NameOf(desc));
                return nvrhi::FE_INVALID_ARGS;
            }
            outId = ComponentId(existing);
            return nvrhi::FS_OK;
        }
        // A new type grows the registry, which recording workers read during a phase (EcsCommandBuffer.cpp).
        if (!CanChangeStructure())
        {
            ReportValidation("RegisterComponent", "%s: a new type is registered outside phases and observers, or in an "
                             "exclusive system", NameOf(desc));
            return c_WrongTimeResult;
        }
        const nvrhi::FRESULT result = ValidateTypeDesc(desc);
        if (NVRHI_FAILED(result))
            return result;
        // A resource descriptor registers a resource with its default value (RegisterResource without a value).
        return AddType(desc, nullptr, outId);
    }

    nvrhi::FRESULT WorldImpl::RegisterResource(const ComponentTypeDesc& desc, const void* initialValue,
                                               ComponentId& outId) noexcept
    {
        outId = ComponentId::Invalid;
        if (!Has(desc.flags, ComponentFlags::Resource))
        {
            ReportValidation("RegisterResource", "%s: the descriptor lacks ComponentFlags::Resource", NameOf(desc));
            return nvrhi::FE_INVALID_ARGS;
        }
        const uint32_t existing = FindType(desc.uuid);
        if (existing != HashIndex::c_NotFound)
        {
            if (!SameLayout(m_Types[existing]->desc, desc))
            {
                ReportValidation("RegisterResource", "%s: the GUID is registered with another layout", NameOf(desc));
                return nvrhi::FE_INVALID_ARGS;
            }
            outId = ComponentId(existing);   // the value stays as it is
            return nvrhi::FS_OK;
        }
        if (!CanChangeStructure())
        {
            ReportValidation("RegisterResource", "%s: a new type is registered outside phases and observers, or in an "
                             "exclusive system", NameOf(desc));
            return c_WrongTimeResult;
        }
        const nvrhi::FRESULT result = ValidateTypeDesc(desc);
        if (NVRHI_FAILED(result))
            return result;
        if (initialValue && !IsCopyable(desc))
            return nvrhi::FE_UNSUPPORTED;
        return AddType(desc, initialValue, outId);
    }

    ComponentId WorldImpl::FindComponent(const nvrhi::GUID& uuid) noexcept
    {
        const uint32_t index = FindType(uuid);
        return index == HashIndex::c_NotFound ? ComponentId::Invalid : ComponentId(index);
    }

    const ComponentTypeDesc* WorldImpl::GetComponentTypeDesc(ComponentId component) noexcept
    {
        const ComponentType* type = GetType(component);
        return type ? &type->desc : nullptr;
    }

    uint32_t WorldImpl::GetComponentTypeCount() noexcept
    {
        return uint32_t(m_Types.size());
    }

    void* WorldImpl::GetResource(ComponentId resource) noexcept
    {
        const ComponentType* type = GetType(resource);
        return type && type->kind == StorageKind::Resource ? type->value : nullptr;
    }

    // ----------------------------------------------------------------------------------------- entity table

    const EntityRecord* WorldImpl::FindAlive(Entity entity) const noexcept
    {
        const uint32_t index = GetEntityIndex(entity);
        if (index == 0 || index >= m_Entities.size())
            return nullptr;
        const EntityRecord& record = m_Entities[index];
        if (record.archetype >= c_ClaimedSlot || record.generation != GetEntityGeneration(entity))
            return nullptr;
        return &record;
    }

    EntityRecord* WorldImpl::FindAliveRecord(Entity entity) noexcept
    {
        return const_cast<EntityRecord*>(FindAlive(entity));
    }

    uint32_t WorldImpl::AllocateIndex() noexcept
    {
        if (m_FreeHead != 0)
        {
            const uint32_t index = m_FreeHead;
            EntityRecord& record = m_Entities[index];
            m_FreeHead = record.chunk;
            if (m_FreeHead == 0)
                m_FreeTail = 0;
            record.chunk = 0;
            return index;
        }
        if (m_Entities.size() >= 0xffffffffull)
            return 0;
        m_Entities.push_back(EntityRecord{ c_FirstGeneration, c_FreeSlot, 0, 0, 0 });
        return uint32_t(m_Entities.size() - 1);
    }

    void WorldImpl::ReleaseIndex(uint32_t index) noexcept
    {
        // FIFO: a destroyed index is reused after every index freed before it, so a stale id stays stale for long.
        EntityRecord& record = m_Entities[index];
        record.generation = NextGeneration(record.generation);
        record.archetype = c_FreeSlot;
        record.chunk = 0;
        record.row = 0;
        record.sparseCount = 0;
        if (m_FreeTail != 0)
            m_Entities[m_FreeTail].chunk = index;
        else
            m_FreeHead = index;
        m_FreeTail = index;
    }

    nvrhi::FRESULT WorldImpl::AllocateEntityIds(uint32_t count, Entity* outEntities) noexcept
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t index = AllocateIndex();
            if (index == 0)
            {
                for (uint32_t j = 0; j < i; ++j)
                    ReleaseIndex(GetEntityIndex(outEntities[j]));
                return nvrhi::FE_OUT_OF_MEMORY;
            }
            EntityRecord& record = m_Entities[index];
            record.archetype = c_ReservedSlot;
            outEntities[i] = MakeEntity(index, record.generation);
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldImpl::ReserveEntities(uint32_t count, Entity* outEntities) noexcept
    {
        // An immediate structural change (EcsWorld.h): the entity table may grow, under readers in a phase.
        if (!CanChangeStructure())
        {
            ReportValidation("ReserveEntities", "structural changes are deferred inside phases and observers (3.3.3)");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (count != 0 && !outEntities)
            return nvrhi::FE_INVALID_ARGS;
        return AllocateEntityIds(count, outEntities);
    }

    bool WorldImpl::IsAlive(Entity entity) noexcept
    {
        return FindAlive(entity) != nullptr;
    }

    // ---------------------------------------------------------------------------------- per-entity access

    void* WorldImpl::ComponentPointer(Entity entity, const EntityRecord& record, const ComponentType& type,
                                      uint32_t* outColumn) const noexcept
    {
        if (outColumn)
            *outColumn = c_NoColumn;
        if (type.kind == StorageKind::Resource)
            return nullptr;
        if (type.kind == StorageKind::Sparse)
        {
            const SparseSet& set = *m_SparseSets[type.sparseSet];
            const uint32_t dense = set.Find(GetEntityIndex(entity));
            if (dense == HashIndex::c_NotFound)
                return nullptr;
            return type.desc.size != 0 ? set.Value(dense) : static_cast<void*>(g_TagValue);
        }
        const Archetype& archetype = *m_Archetypes[record.archetype];
        const uint32_t column = archetype.FindColumn(type.id);
        if (column == c_NoColumn)
            return nullptr;
        if (outColumn)
            *outColumn = column;
        const ArchetypeColumn& c = archetype.columns[column];
        uint8_t* memory = archetype.chunks[record.chunk].memory;
        switch (c.kind)
        {
        case StorageKind::Data:
        case StorageKind::Buffer:
            return memory + c.offset + size_t(record.row) * c.stride;
        case StorageKind::Shared:
            return archetype.sharedValues + c.offset;
        case StorageKind::Chunk:
            return memory + c.offset;
        default:
            return g_TagValue;
        }
    }

    bool WorldImpl::HasComponent(Entity entity, ComponentId component) noexcept
    {
        const EntityRecord* record = FindAlive(entity);
        if (!record)
        {
            ReportStale("HasComponent", entity);
            return false;
        }
        const ComponentType* type = GetType(component);
        if (!type)
            return false;
        if (type->kind == StorageKind::Sparse)
            return m_SparseSets[type->sparseSet]->Find(GetEntityIndex(entity)) != HashIndex::c_NotFound;
        if (type->kind == StorageKind::Resource)
            return false;
        return m_Archetypes[record->archetype]->FindColumn(component) != c_NoColumn;
    }

    const void* WorldImpl::GetComponent(Entity entity, ComponentId component) noexcept
    {
        const EntityRecord* record = FindAlive(entity);
        if (!record)
        {
            ReportStale("GetComponent", entity);
            return nullptr;
        }
        const ComponentType* type = GetType(component);
        return type ? ComponentPointer(entity, *record, *type, nullptr) : nullptr;
    }

    void* WorldImpl::GetComponentForWrite(Entity entity, ComponentId component, uint64_t writeVersion) noexcept
    {
        const char* operation = "GetComponentForWrite";
        const EntityRecord* record = FindAlive(entity);
        if (!record)
        {
            ReportStale(operation, entity);
            return nullptr;
        }
        const ComponentType* type = GetType(component);
        if (!type)
            return nullptr;
        if (type->kind == StorageKind::Shared)
        {
            ReportValidation(operation, "%s is Shared: its value partitions chunks, change it with AddComponent",
                             NameOf(type->desc));
            return nullptr;
        }
        if (writeVersion > m_Version)
            ReportValidation(operation, "version %llu has not been handed out yet (the world is at %llu)",
                             (unsigned long long)writeVersion, (unsigned long long)m_Version);
        CheckWriteAccess(component, writeVersion, operation);
        uint32_t column = c_NoColumn;
        void* pointer = ComponentPointer(entity, *record, *type, &column);
        if (pointer && column != c_NoColumn)
        {
            const Archetype& archetype = *m_Archetypes[record->archetype];
            const StorageKind kind = archetype.columns[column].kind;
            if (kind == StorageKind::Data || kind == StorageKind::Buffer || kind == StorageKind::Chunk)
                StampVersion(archetype.Versions(record->chunk) + column, ResolveWriteVersion(writeVersion));
        }
        return pointer;
    }

    nvrhi::FRESULT WorldImpl::SetEnabled(Entity entity, ComponentId component, bool enabled) noexcept
    {
        return SetEnabledInternal(&entity, 1, component, enabled, GetStampVersion());
    }

    nvrhi::FRESULT WorldImpl::SetEnabledInternal(const Entity* entities, uint32_t count, ComponentId component,
                                                 bool enabled, uint64_t version) noexcept
    {
        const char* operation = "SetEnabled";
        ObserveValueAccess();
        const ComponentType* type = GetType(component);
        if (!type || !type->enableable)
        {
            ReportValidation(operation, "component %u is not Enableable", uint32_t(component));
            return nvrhi::FE_INVALID_ARGS;
        }
        nvrhi::FRESULT result = nvrhi::FS_OK;
        for (uint32_t i = 0; i < count; ++i)
        {
            const EntityRecord* record = FindAlive(entities[i]);
            if (!record)
            {
                ReportStale(operation, entities[i]);
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                continue;
            }
            const Archetype& archetype = *m_Archetypes[record->archetype];
            const uint32_t column = archetype.FindColumn(component);
            if (column == c_NoColumn)
            {
                ReportValidation(operation, "entity %u:%u lacks %s", GetEntityIndex(entities[i]),
                                 GetEntityGeneration(entities[i]), NameOf(type->desc));
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                continue;
            }
            SetMaskBit(archetype.EnabledMask(record->chunk, column), record->row, enabled);
            StampVersion(archetype.Versions(record->chunk) + column, version);
        }
        return result;
    }

    bool WorldImpl::IsEnabled(Entity entity, ComponentId component) noexcept
    {
        const EntityRecord* record = FindAlive(entity);
        if (!record)
        {
            ReportStale("IsEnabled", entity);
            return false;
        }
        const ComponentType* type = GetType(component);
        if (!type)
            return false;
        if (type->kind == StorageKind::Sparse)
            return m_SparseSets[type->sparseSet]->Find(GetEntityIndex(entity)) != HashIndex::c_NotFound;
        if (type->kind == StorageKind::Resource)
            return false;
        const Archetype& archetype = *m_Archetypes[record->archetype];
        const uint32_t column = archetype.FindColumn(component);
        if (column == c_NoColumn)
            return false;
        const uint64_t* mask = archetype.EnabledMask(record->chunk, column);
        return !mask || GetMaskBit(mask, record->row);
    }

    nvrhi::FRESULT WorldImpl::ResizeBuffer(Entity entity, ComponentId component, uint32_t count,
                                           uint64_t writeVersion) noexcept
    {
        CheckWriteAccess(component, writeVersion, "ResizeBuffer");
        return ResizeBufferInternal(entity, component, count, ResolveWriteVersion(writeVersion));
    }

    nvrhi::FRESULT WorldImpl::ResizeBufferInternal(Entity entity, ComponentId component, uint32_t count,
                                                   uint64_t version) noexcept
    {
        const char* operation = "ResizeBuffer";
        const EntityRecord* record = FindAlive(entity);
        if (!record)
        {
            ReportStale(operation, entity);
            return nvrhi::FE_INVALID_ARGS;
        }
        const ComponentType* type = GetType(component);
        if (!type || type->kind != StorageKind::Buffer)
        {
            ReportValidation(operation, "component %u is not a Buffer", uint32_t(component));
            return nvrhi::FE_INVALID_ARGS;
        }
        const Archetype& archetype = *m_Archetypes[record->archetype];
        const uint32_t column = archetype.FindColumn(component);
        if (column == c_NoColumn)
        {
            ReportValidation(operation, "the entity lacks %s", NameOf(type->desc));
            return nvrhi::FE_INVALID_ARGS;
        }
        const ArchetypeColumn& c = archetype.columns[column];
        BufferHeader& header = static_cast<BufferHeader*>(archetype.ColumnData(record->chunk, column))[record->row];
        uint8_t* inlineData = archetype.InlineData(record->chunk, column, record->row);
        const nvrhi::FRESULT result = ResizeBufferRow(c, header, inlineData, count);
        if (NVRHI_SUCCEEDED(result))
            StampVersion(archetype.Versions(record->chunk) + column, version);
        return result;
    }

    nvrhi::FRESULT WorldImpl::AppendToBufferInternal(Entity entity, ComponentId component, const void* elements,
                                                     uint32_t count, ValueTransfer transfer, uint64_t version) noexcept
    {
        // `elements` must not point into the buffer itself: growing it may move the storage.
        const char* operation = "AppendToBuffer";
        ObserveValueAccess();
        if (count == 0)
            return nvrhi::FS_OK;
        if (!elements)
            return nvrhi::FE_INVALID_ARGS;
        auto fail = [&](nvrhi::FRESULT result) noexcept
        {
            DiscardValues(component, elements, count, transfer);   // Relocate: consumed even when not appended
            return result;
        };
        const EntityRecord* record = FindAlive(entity);
        if (!record)
        {
            ReportStale(operation, entity);
            return fail(nvrhi::FE_INVALID_ARGS);
        }
        const ComponentType* type = GetType(component);
        if (!type || type->kind != StorageKind::Buffer)
        {
            ReportValidation(operation, "component %u is not a Buffer", uint32_t(component));
            return fail(nvrhi::FE_INVALID_ARGS);
        }
        if (transfer == ValueTransfer::Copy && !type->copyable)
            return nvrhi::FE_UNSUPPORTED;   // Copy: the caller keeps its elements
        const Archetype& archetype = *m_Archetypes[record->archetype];
        const uint32_t column = archetype.FindColumn(component);
        if (column == c_NoColumn)
        {
            ReportValidation(operation, "the entity lacks %s", NameOf(type->desc));
            return fail(nvrhi::FE_INVALID_ARGS);
        }
        BufferHeader& header = static_cast<BufferHeader*>(archetype.ColumnData(record->chunk, column))[record->row];
        const uint32_t first = header.count;
        if (count > 0xffffffffu - first)
        {
            ReportValidation(operation, "%u more elements would overflow the count of %u", count, first);
            return fail(nvrhi::FE_INVALID_ARGS);
        }
        const nvrhi::FRESULT result = ResizeBufferRow(archetype.columns[column], header,
                                                      archetype.InlineData(record->chunk, column, record->row),
                                                      first + count);
        if (NVRHI_FAILED(result))
            return fail(result);
        // The resize constructed the new elements; they are replaced by the given ones.
        uint8_t* target = static_cast<uint8_t*>(header.data) + size_t(first) * type->desc.size;
        DestructValues(*type, target, count);
        TransferValues(*type, target, elements, count, transfer);
        StampVersion(archetype.Versions(record->chunk) + column, version);
        return nvrhi::FS_OK;
    }

    void WorldImpl::DiscardValues(ComponentId component, const void* values, uint32_t count,
                                  ValueTransfer transfer) noexcept
    {
        if (transfer != ValueTransfer::Relocate || !values || count == 0)
            return;
        const ComponentType* type = GetType(component);
        if (type)
            DestructValues(*type, const_cast<void*>(values), count);
    }

    void WorldImpl::DiscardCreateValues(const EntityCreateDesc& desc, uint32_t first) noexcept
    {
        if (desc.transfer != ValueTransfer::Relocate || !desc.values || !desc.components || first >= desc.entityCount)
            return;
        for (uint32_t i = 0; i < desc.componentCount; ++i)
        {
            const ComponentType* type = GetType(desc.components[i]);
            if (!type || !desc.values[i])
                continue;
            const uint8_t* values = static_cast<const uint8_t*>(desc.values[i]) + size_t(first) * type->desc.size;
            DiscardValues(desc.components[i], values, desc.entityCount - first, desc.transfer);
        }
    }

    // ----------------------------------------------------------------------------- structural operations

    nvrhi::FRESULT WorldImpl::CreateEntitiesInternal(const EntityCreateDesc& desc, Entity* outEntities,
                                                     uint64_t version, const RowHooks* hooks) noexcept
    {
        const char* operation = "CreateEntities";
        const uint32_t n = desc.entityCount;
        if (n == 0)
            return nvrhi::FS_OK;
        // A rejected call consumes relocated values too (the ownership rule of the batch operations).
        auto reject = [&](nvrhi::FRESULT result) noexcept
        {
            DiscardCreateValues(desc, 0);
            return result;
        };
        if ((desc.componentCount != 0 && !desc.components) || (!outEntities && !desc.reservedEntities) ||
            (desc.transfer != ValueTransfer::Copy && desc.transfer != ValueTransfer::Relocate))
        {
            ReportValidation(operation, "invalid arguments");
            return reject(nvrhi::FE_INVALID_ARGS);
        }

        // The components, sorted, with the index of each in the caller's list.
        m_ScratchTerms.clear();
        for (uint32_t i = 0; i < desc.componentCount; ++i)
        {
            const ComponentType* type = GetType(desc.components[i]);
            if (!type || type->kind == StorageKind::Resource)
            {
                ReportValidation(operation, "component %u is not a registered component", uint32_t(desc.components[i]));
                return reject(nvrhi::FE_INVALID_ARGS);
            }
            const void* value = desc.values ? desc.values[i] : nullptr;
            if (value && (type->kind == StorageKind::Buffer || type->kind == StorageKind::Chunk))
            {
                ReportValidation(operation, "%s takes no initial value (Buffer and Chunk components)",
                                 NameOf(type->desc));
                return reject(nvrhi::FE_INVALID_ARGS);
            }
            if (value && desc.transfer == ValueTransfer::Copy && !type->copyable && type->desc.size != 0)
            {
                ReportValidation(operation, "%s cannot be copied", NameOf(type->desc));
                return nvrhi::FE_UNSUPPORTED;
            }
            m_ScratchTerms.push_back(SortedTerm{ desc.components[i], i });
        }
        std::sort(m_ScratchTerms.begin(), m_ScratchTerms.end(),
                  [](const SortedTerm& a, const SortedTerm& b) { return a.component < b.component; });
        for (uint32_t i = 1; i < m_ScratchTerms.size(); ++i)
        {
            if (m_ScratchTerms[i].component == m_ScratchTerms[i - 1].component)
            {
                ReportValidation(operation, "component %u is listed twice", uint32_t(m_ScratchTerms[i].component));
                return reject(nvrhi::FE_INVALID_ARGS);
            }
        }

        // Reserved ids are claimed before anything changes, so that a bad or repeated one changes nothing.
        if (desc.reservedEntities)
        {
            for (uint32_t k = 0; k < n; ++k)
            {
                const Entity entity = desc.reservedEntities[k];
                const uint32_t index = GetEntityIndex(entity);
                const bool valid = index != 0 && index < m_Entities.size() &&
                    m_Entities[index].archetype == c_ReservedSlot &&
                    m_Entities[index].generation == GetEntityGeneration(entity);
                if (!valid)
                {
                    for (uint32_t j = 0; j < k; ++j)
                        m_Entities[GetEntityIndex(desc.reservedEntities[j])].archetype = c_ReservedSlot;
                    ReportValidation(operation, "entity %u:%u is not a reserved id", index,
                                     GetEntityGeneration(entity));
                    return reject(nvrhi::FE_INVALID_ARGS);
                }
                m_Entities[index].archetype = c_ClaimedSlot;
            }
        }

        // The archetype key: the archetype's components in order, the caller's index of each, and the Shared values.
        m_ScratchIds.clear();
        m_ScratchShared.clear();
        m_ScratchValueIndex.clear();
        bool sharedFromValues = false;
        for (const SortedTerm& term : m_ScratchTerms)
        {
            const ComponentType& type = *m_Types[uint32_t(term.component)];
            if (type.kind == StorageKind::Sparse)
                continue;
            const void* value = desc.values ? desc.values[term.source] : nullptr;
            m_ScratchIds.push_back(term.component);
            m_ScratchValueIndex.push_back(term.source);
            m_ScratchShared.push_back(type.kind == StorageKind::Shared ? (value ? value : type.value) : nullptr);
            if (type.kind == StorageKind::Shared && value)
                sharedFromValues = true;
        }
        const uint32_t keyCount = uint32_t(m_ScratchIds.size());

        auto entityAt = [&](uint32_t k) noexcept
        {
            return desc.reservedEntities ? desc.reservedEntities[k] : outEntities[k];
        };
        auto valueOf = [&](uint32_t column) noexcept -> const uint8_t*
        {
            return desc.values ? static_cast<const uint8_t*>(desc.values[m_ScratchValueIndex[column]]) : nullptr;
        };

        // Runs of entities with equal Shared values go to one archetype each, in order.
        nvrhi::FRESULT result = nvrhi::FS_OK;
        m_ScratchRuns.clear();
        uint32_t k = 0;
        while (k < n)
        {
            uint32_t end = n;
            if (sharedFromValues)
            {
                for (uint32_t c = 0; c < keyCount; ++c)
                {
                    const ComponentType& type = *m_Types[uint32_t(m_ScratchIds[c])];
                    const uint8_t* values = valueOf(c);
                    if (type.kind == StorageKind::Shared && values)
                        m_ScratchShared[c] = values + size_t(k) * type.desc.size;
                }
                end = k + 1;
                while (end < n)
                {
                    bool same = true;
                    for (uint32_t c = 0; c < keyCount && same; ++c)
                    {
                        const ComponentType& type = *m_Types[uint32_t(m_ScratchIds[c])];
                        const uint8_t* values = valueOf(c);
                        if (type.kind == StorageKind::Shared && values)
                            same = FieldBytesEqual(type, values + size_t(k) * type.desc.size,
                                                   values + size_t(end) * type.desc.size);
                    }
                    if (!same)
                        break;
                    ++end;
                }
            }

            const uint32_t archetypeIndex =
                FindOrCreateArchetype(m_ScratchIds.data(), keyCount, m_ScratchShared.data());
            if (IsArchetypeError(archetypeIndex))
            {
                if (archetypeIndex == c_ArchetypeTooLarge)
                    ReportValidation(operation, "one row of these components does not fit the largest chunk");
                result = ArchetypeErrorResult(archetypeIndex);
                break;
            }
            Archetype& archetype = *m_Archetypes[archetypeIndex];
            const uint32_t first = k;
            const nvrhi::FRESULT appended = AppendRows(archetype, end - k, version,
                [&](uint32_t chunk, uint32_t row, uint32_t run, uint32_t done) noexcept
            {
                Entity* entities = archetype.Entities(chunk);
                for (uint32_t t = 0; t < run; ++t)
                {
                    const uint32_t e = first + done + t;
                    Entity entity;
                    if (desc.reservedEntities)
                    {
                        entity = desc.reservedEntities[e];
                    }
                    else
                    {
                        const uint32_t index = AllocateIndex();
                        NVRHI_ASSERT(index != 0);
                        entity = MakeEntity(index, m_Entities[index].generation);
                        outEntities[e] = entity;
                    }
                    EntityRecord& record = m_Entities[GetEntityIndex(entity)];
                    record.archetype = archetypeIndex;
                    record.chunk = chunk;
                    record.row = uint16_t(row + t);
                    record.sparseCount = 0;
                    entities[row + t] = entity;
                    ++m_AliveCount;
                }
                if (desc.reservedEntities && outEntities)
                    for (uint32_t t = 0; t < run; ++t)
                        outEntities[first + done + t] = desc.reservedEntities[first + done + t];
                for (uint32_t c = 0; c < keyCount; ++c)
                {
                    const ArchetypeColumn& column = archetype.columns[c];
                    const uint8_t* values = column.kind == StorageKind::Data ? valueOf(c) : nullptr;
                    InitializeColumn(archetype, c, chunk, row, run,
                                     values ? values + size_t(first + done) * column.stride : nullptr, desc.transfer);
                }
                m_ScratchRuns.push_back(RowRun{ archetypeIndex, chunk, row, run });
            });
            if (NVRHI_FAILED(appended))
            {
                result = appended;
                break;
            }
            k = end;
        }

        // The entities that were not created give their claimed ids back.
        if (desc.reservedEntities)
            for (uint32_t j = k; j < n; ++j)
                m_Entities[GetEntityIndex(desc.reservedEntities[j])].archetype = c_ReservedSlot;
        const uint32_t created = k;

        // Sparse components, entity by entity, once the rows exist.
        for (const SortedTerm& term : m_ScratchTerms)
        {
            const ComponentType& type = *m_Types[uint32_t(term.component)];
            if (type.kind != StorageKind::Sparse)
                continue;
            SparseSet& set = *m_SparseSets[type.sparseSet];
            const uint8_t* values = desc.values ? static_cast<const uint8_t*>(desc.values[term.source]) : nullptr;
            for (uint32_t e = 0; e < created; ++e)
            {
                const Entity entity = entityAt(e);
                void* slot = SparseInsert(set, entity, m_Entities[GetEntityIndex(entity)]);
                if (!slot)
                {
                    KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);
                    if (values)
                        DiscardValues(term.component, values + size_t(e) * type.desc.size, 1, desc.transfer);
                    continue;
                }
                if (values)
                    TransferValues(type, slot, values + size_t(e) * type.desc.size, 1, desc.transfer);
                else
                    ConstructValues(type, slot, 1);
            }
        }

        // Relocate: what was not moved in is destroyed. The Data and Sparse values of the entities that were not
        // created, and every Shared value (the archetypes hold copies).
        if (desc.transfer == ValueTransfer::Relocate && desc.values)
        {
            for (const SortedTerm& term : m_ScratchTerms)
            {
                const ComponentType& type = *m_Types[uint32_t(term.component)];
                const uint8_t* values = static_cast<const uint8_t*>(desc.values[term.source]);
                if (!values)
                    continue;
                if (type.kind == StorageKind::Shared)
                    DiscardValues(term.component, values, n, desc.transfer);
                else if (created < n)
                    DiscardValues(term.component, values + size_t(created) * type.desc.size, n - created,
                                  desc.transfer);
            }
        }

        if (hooks && hooks->rowsAdded && created != 0)
            hooks->rowsAdded(hooks->context, created);
        for (const RowRun& run : m_ScratchRuns)
            ObserveRowsAdded(run.archetype, run.chunk, run.firstRow, run.count, c_NoArchetype);
        for (const SortedTerm& term : m_ScratchTerms)
        {
            const ComponentType& type = *m_Types[uint32_t(term.component)];
            if (type.kind != StorageKind::Sparse)
                continue;
            const SparseSet& set = *m_SparseSets[type.sparseSet];
            for (uint32_t e = 0; e < created; ++e)
                if (set.Find(GetEntityIndex(entityAt(e))) != HashIndex::c_NotFound)
                    ObserveSparseAdded(term.component, entityAt(e));
        }
        return result;
    }

    nvrhi::FRESULT WorldImpl::DestroyEntitiesInternal(const Entity* entities, uint32_t count, uint64_t version,
                                                      const RowHooks* hooks) noexcept
    {
        nvrhi::FRESULT result = nvrhi::FS_OK;
        for (uint32_t i = 0; i < count; ++i)
        {
            const Entity entity = entities[i];
            const uint32_t index = GetEntityIndex(entity);
            if (index != 0 && index < m_Entities.size() && m_Entities[index].archetype == c_ReservedSlot &&
                m_Entities[index].generation == GetEntityGeneration(entity))
            {
                ReleaseIndex(index);   // a reservation given back
                continue;
            }
            EntityRecord* record = FindAliveRecord(entity);
            if (!record)
            {
                ReportStale("DestroyEntities", entity);
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                continue;
            }
            Archetype& archetype = *m_Archetypes[record->archetype];
            const uint32_t chunk = record->chunk;
            const uint32_t row = record->row;
            if (!SecureHoleFill(archetype, chunk))
            {
                KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);   // nothing changed for this entity
                continue;
            }
            ObserveRowsRemoving(archetype.index, chunk, row, 1, c_NoArchetype);
            if (record->sparseCount != 0)
            {
                for (SparseSet* set : m_SparseSets)
                {
                    if (set->Find(index) == HashIndex::c_NotFound)
                        continue;
                    ObserveSparseRemoving(set->type->id, entity);
                    SparseErase(*set, entity, *record);
                }
            }
            if (hooks && hooks->rowRemoving)
                hooks->rowRemoving(hooks->context, entity, archetype, chunk, row);
            DestructRow(archetype, chunk, row);
            RemoveRow(archetype, chunk, row, version);
            ReleaseIndex(index);
            --m_AliveCount;
        }
        return result;
    }

    nvrhi::FRESULT WorldImpl::AddComponentsInternal(const Entity* entities, uint32_t count, ComponentId component,
                                                    const void* values, ValueTransfer transfer,
                                                    uint64_t version) noexcept
    {
        const char* operation = "AddComponent";
        ObserveValueAccess();   // a component the entity has is overwritten without a report
        const ComponentType* type = GetType(component);
        if (!type || type->kind == StorageKind::Resource || (count != 0 && !entities))
        {
            ReportValidation(operation, "component %u is not a registered component", uint32_t(component));
            DiscardValues(component, values, count, transfer);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (values && (type->kind == StorageKind::Buffer || type->kind == StorageKind::Chunk))
        {
            ReportValidation(operation, "%s takes no value (Buffer and Chunk components)", NameOf(type->desc));
            DiscardValues(component, values, count, transfer);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (values && transfer == ValueTransfer::Copy && !type->copyable && type->desc.size != 0)
        {
            ReportValidation(operation, "%s cannot be copied", NameOf(type->desc));
            return nvrhi::FE_UNSUPPORTED;
        }
        const uint32_t size = type->desc.size;
        nvrhi::FRESULT result = nvrhi::FS_OK;
        uint32_t cachedFrom = c_NoArchetype;
        uint32_t cachedTo = c_NoArchetype;
        for (uint32_t i = 0; i < count; ++i)
        {
            const Entity entity = entities[i];
            const uint8_t* value =
                values && size != 0 ? static_cast<const uint8_t*>(values) + size_t(i) * size : nullptr;
            EntityRecord* record = FindAliveRecord(entity);
            if (!record)
            {
                ReportStale(operation, entity);
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                DiscardValues(component, value, 1, transfer);
                continue;
            }

            if (type->kind == StorageKind::Sparse)
            {
                SparseSet& set = *m_SparseSets[type->sparseSet];
                const uint32_t dense = set.Find(GetEntityIndex(entity));
                if (dense != HashIndex::c_NotFound)
                {
                    if (value)
                        OverwriteValue(set.Value(dense), *type, value, transfer);
                    continue;
                }
                void* slot = SparseInsert(set, entity, *record);
                if (!slot)
                {
                    KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);
                    DiscardValues(component, value, 1, transfer);
                    continue;
                }
                if (value)
                    TransferValues(*type, slot, value, 1, transfer);
                else
                    ConstructValues(*type, slot, 1);
                ObserveSparseAdded(component, entity);
                continue;
            }

            const Archetype& from = *m_Archetypes[record->archetype];
            const uint32_t column = from.FindColumn(component);
            uint32_t target = c_NoArchetype;
            if (type->kind == StorageKind::Shared)
            {
                if (column != c_NoColumn &&
                    (!value || FieldBytesEqual(*type, from.sharedValues + from.columns[column].offset, value)))
                {
                    DiscardValues(component, value, 1, transfer);   // the same partition; the archetype has the value
                    continue;
                }
                target = GetSharedTarget(record->archetype, component, value ? value : type->value);
            }
            else if (column != c_NoColumn)
            {
                // Present already: the value is written, or nothing happens.
                if (value && type->kind == StorageKind::Data)
                {
                    uint8_t* dst = static_cast<uint8_t*>(from.ColumnData(record->chunk, column)) +
                        size_t(record->row) * from.columns[column].stride;
                    OverwriteValue(dst, *type, value, transfer);
                    StampVersion(from.Versions(record->chunk) + column, version);
                }
                continue;
            }
            else if (record->archetype == cachedFrom)
            {
                target = cachedTo;
                ++m_Counters.edgeHits;
            }
            else
            {
                cachedFrom = record->archetype;
                target = GetAddTarget(record->archetype, component);
                cachedTo = target;
            }
            if (IsArchetypeError(target))
            {
                if (target == c_ArchetypeTooLarge)
                    ReportValidation(operation, "one row with %s does not fit the largest chunk", NameOf(type->desc));
                KeepFirstFailure(result, ArchetypeErrorResult(target));
                DiscardValues(component, value, 1, transfer);
                continue;
            }
            NewValue newValue;
            newValue.component = component;
            newValue.value = type->kind == StorageKind::Data ? value : nullptr;
            newValue.transfer = transfer;
            const nvrhi::FRESULT moved = MoveEntity(entity, *record, target, newValue, version);
            KeepFirstFailure(result, moved);
            // A Shared value is always the caller's (the archetype keeps a copy); a Data value is the caller's when
            // the move failed before taking it.
            if (type->kind == StorageKind::Shared || NVRHI_FAILED(moved))
                DiscardValues(component, value, 1, transfer);
        }
        return result;
    }

    nvrhi::FRESULT WorldImpl::RemoveComponentsInternal(const Entity* entities, uint32_t count, ComponentId component,
                                                       uint64_t version) noexcept
    {
        const char* operation = "RemoveComponent";
        const ComponentType* type = GetType(component);
        if (!type || type->kind == StorageKind::Resource || (count != 0 && !entities))
        {
            ReportValidation(operation, "component %u is not a registered component", uint32_t(component));
            return nvrhi::FE_INVALID_ARGS;
        }
        nvrhi::FRESULT result = nvrhi::FS_OK;
        uint32_t cachedFrom = c_NoArchetype;
        uint32_t cachedTo = c_NoArchetype;
        for (uint32_t i = 0; i < count; ++i)
        {
            const Entity entity = entities[i];
            EntityRecord* record = FindAliveRecord(entity);
            if (!record)
            {
                ReportStale(operation, entity);
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                continue;
            }
            if (type->kind == StorageKind::Sparse)
            {
                SparseSet& set = *m_SparseSets[type->sparseSet];
                if (set.Find(GetEntityIndex(entity)) != HashIndex::c_NotFound)
                {
                    ObserveSparseRemoving(component, entity);
                    SparseErase(set, entity, *record);
                }
                continue;
            }
            if (m_Archetypes[record->archetype]->FindColumn(component) == c_NoColumn)
                continue;   // absent: nothing to do
            uint32_t target;
            if (record->archetype == cachedFrom)
            {
                target = cachedTo;
                ++m_Counters.edgeHits;
            }
            else
            {
                cachedFrom = record->archetype;
                target = GetRemoveTarget(record->archetype, component);
                cachedTo = target;
            }
            if (IsArchetypeError(target))
            {
                KeepFirstFailure(result, ArchetypeErrorResult(target));
                continue;
            }
            KeepFirstFailure(result, MoveEntity(entity, *record, target, NewValue{}, version));
        }
        return result;
    }

    nvrhi::FRESULT WorldImpl::ChangeComponentsInternal(Entity entity, ComponentChange& change, uint64_t version,
                                                       const RowHooks* hooks) noexcept
    {
        EntityRecord* record = FindAliveRecord(entity);
        if (!record)
        {
            ReportStale("ChangeComponents", entity);
            return nvrhi::FE_INVALID_ARGS;
        }
        const uint32_t source = record->archetype;
        if (change.source != source)
        {
            const nvrhi::FRESULT prepared = PrepareChange(source, change);
            if (NVRHI_FAILED(prepared))
                return prepared;
        }
        if (change.target != source)
        {
            const nvrhi::FRESULT moved = MoveEntity(entity, *record, change.target, change.added, change.addedCount,
                                                    version, hooks);
            if (NVRHI_FAILED(moved))
                return moved;
        }
        // The values of what the entity had already. The report of the move is held at most: no observer has run.
        for (uint32_t a = 0; change.addsPresent && a < change.addedCount; ++a)
        {
            const NewValue& value = change.added[a];
            if (!value.value || m_Archetypes[source]->FindColumn(value.component) == c_NoColumn)
                continue;
            const Archetype& archetype = *m_Archetypes[change.target];
            const uint32_t column = archetype.FindColumn(value.component);
            if (archetype.columns[column].kind != StorageKind::Data)
                continue;
            uint8_t* dst = static_cast<uint8_t*>(archetype.ColumnData(record->chunk, column)) +
                size_t(record->row) * archetype.columns[column].stride;
            OverwriteValue(dst, *archetype.columns[column].type, value.value, value.transfer);
            StampVersion(archetype.Versions(record->chunk) + column, version);
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldImpl::SetComponentsInternal(const Entity* entities, uint32_t count, ComponentId component,
                                                    const void* values, ValueTransfer transfer,
                                                    uint64_t version) noexcept
    {
        const char* operation = "SetComponents";
        ObserveValueAccess();   // each value is written before its report
        const ComponentType* type = GetType(component);
        if (!type || type->kind == StorageKind::Resource || type->kind == StorageKind::Buffer ||
            type->kind == StorageKind::Chunk || (count != 0 && !entities) || (!values && type->desc.size != 0))
        {
            ReportValidation(operation, "component %u cannot be set", uint32_t(component));
            DiscardValues(component, values, count, transfer);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (transfer == ValueTransfer::Copy && !type->copyable && type->desc.size != 0)
            return nvrhi::FE_UNSUPPORTED;
        const uint32_t size = type->desc.size;
        nvrhi::FRESULT result = nvrhi::FS_OK;
        for (uint32_t i = 0; i < count; ++i)
        {
            const Entity entity = entities[i];
            const uint8_t* value = size != 0 ? static_cast<const uint8_t*>(values) + size_t(i) * size : nullptr;
            EntityRecord* record = FindAliveRecord(entity);
            if (!record)
            {
                ReportStale(operation, entity);
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                DiscardValues(component, value, 1, transfer);
                continue;
            }
            if (type->kind == StorageKind::Sparse)
            {
                const SparseSet& set = *m_SparseSets[type->sparseSet];
                const uint32_t dense = set.Find(GetEntityIndex(entity));
                if (dense == HashIndex::c_NotFound)
                {
                    ReportValidation(operation, "the entity lacks %s", NameOf(type->desc));
                    KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                    DiscardValues(component, value, 1, transfer);
                    continue;
                }
                if (value)
                    OverwriteValue(set.Value(dense), *type, value, transfer);
                ObserveSet(component, entity);
                continue;
            }
            const Archetype& archetype = *m_Archetypes[record->archetype];
            const uint32_t column = archetype.FindColumn(component);
            if (column == c_NoColumn)
            {
                ReportValidation(operation, "the entity lacks %s", NameOf(type->desc));
                KeepFirstFailure(result, nvrhi::FE_INVALID_ARGS);
                DiscardValues(component, value, 1, transfer);
                continue;
            }
            if (type->kind == StorageKind::Shared)
            {
                nvrhi::FRESULT moved = nvrhi::FS_OK;
                if (!FieldBytesEqual(*type, archetype.sharedValues + archetype.columns[column].offset, value))
                {
                    const uint32_t target = GetSharedTarget(record->archetype, component, value);
                    moved = IsArchetypeError(target) ? ArchetypeErrorResult(target)
                                                     : MoveEntity(entity, *record, target, NewValue{}, version);
                    KeepFirstFailure(result, moved);
                }
                DiscardValues(component, value, 1, transfer);   // the archetype keeps its own copy
                if (NVRHI_FAILED(moved))
                    continue;
            }
            else if (type->kind == StorageKind::Data)
            {
                uint8_t* dst = static_cast<uint8_t*>(archetype.ColumnData(record->chunk, column)) +
                    size_t(record->row) * archetype.columns[column].stride;
                OverwriteValue(dst, *type, value, transfer);
                StampVersion(archetype.Versions(record->chunk) + column, version);
            }
            ObserveSet(component, entity);
        }
        return result;
    }

    nvrhi::FRESULT WorldImpl::InstantiateInternal(Entity prefab, uint32_t count, Entity* outEntities,
                                                  uint64_t version) noexcept
    {
        const char* operation = "Instantiate";
        ObserveValueAccess();   // the prefab's values are copied: what observers write into them comes first
        const EntityRecord* found = FindAlive(prefab);
        if (!found)
        {
            ReportStale(operation, prefab);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (count == 0)
            return nvrhi::FS_OK;
        if (!outEntities)
            return nvrhi::FE_INVALID_ARGS;
        // Copies: the entity table may grow below, so nothing holds a record across it.
        const uint32_t archetypeIndex = found->archetype;
        const uint32_t prefabChunk = found->chunk;
        const uint32_t prefabRow = found->row;
        const uint32_t prefabIndex = GetEntityIndex(prefab);
        const bool prefabSparse = found->sparseCount != 0;

        Archetype& archetype = *m_Archetypes[archetypeIndex];
        for (const ArchetypeColumn& column : archetype.columns)
        {
            if ((column.kind == StorageKind::Data || column.kind == StorageKind::Buffer) && !column.type->copyable)
            {
                ReportValidation(operation, "%s cannot be copied", NameOf(column.type->desc));
                return nvrhi::FE_UNSUPPORTED;
            }
        }
        if (prefabSparse)
            for (const SparseSet* set : m_SparseSets)
                if (set->Find(prefabIndex) != HashIndex::c_NotFound && !set->type->copyable)
                    return nvrhi::FE_UNSUPPORTED;

        uint8_t* prefabMemory = archetype.chunks[prefabChunk].memory;
        nvrhi::FRESULT result = nvrhi::FS_OK;
        m_ScratchRuns.clear();
        const nvrhi::FRESULT appended = AppendRows(archetype, count, version,
            [&](uint32_t chunk, uint32_t row, uint32_t run, uint32_t done) noexcept
        {
            uint8_t* memory = archetype.chunks[chunk].memory;
            Entity* entities = reinterpret_cast<Entity*>(memory);
            for (uint32_t t = 0; t < run; ++t)
            {
                const uint32_t index = AllocateIndex();
                NVRHI_ASSERT(index != 0);
                EntityRecord& record = m_Entities[index];
                record.archetype = archetypeIndex;
                record.chunk = chunk;
                record.row = uint16_t(row + t);
                record.sparseCount = 0;
                const Entity entity = MakeEntity(index, record.generation);
                entities[row + t] = entity;
                outEntities[done + t] = entity;
                ++m_AliveCount;
            }
            for (uint32_t c = 0; c < archetype.columns.size(); ++c)
            {
                const ArchetypeColumn& column = archetype.columns[c];
                if (column.kind == StorageKind::Data)
                {
                    const uint8_t* src = prefabMemory + column.offset + size_t(prefabRow) * column.stride;
                    for (uint32_t t = 0; t < run; ++t)
                        CopyValues(*column.type, memory + column.offset + size_t(row + t) * column.stride, src, 1);
                }
                else if (column.kind == StorageKind::Buffer)
                {
                    const BufferHeader& src =
                        reinterpret_cast<const BufferHeader*>(prefabMemory + column.offset)[prefabRow];
                    for (uint32_t t = 0; t < run; ++t)
                    {
                        BufferHeader& dst = reinterpret_cast<BufferHeader*>(memory + column.offset)[row + t];
                        if (!CopyBuffer(column, dst, archetype.InlineData(chunk, c, row + t), src))
                            KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);
                    }
                }
                if (column.enableable)
                    SetMaskRange(archetype.EnabledMask(chunk, c), row, run,
                                 GetMaskBit(archetype.EnabledMask(prefabChunk, c), prefabRow));
            }
            m_ScratchRuns.push_back(RowRun{ archetypeIndex, chunk, row, run });
        });
        if (NVRHI_FAILED(appended))
            return appended;

        if (prefabSparse)
        {
            for (SparseSet* set : m_SparseSets)
            {
                if (set->Find(prefabIndex) == HashIndex::c_NotFound)
                    continue;
                for (uint32_t t = 0; t < count; ++t)
                {
                    void* slot = SparseInsert(*set, outEntities[t], m_Entities[GetEntityIndex(outEntities[t])]);
                    if (!slot)
                    {
                        KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);
                        continue;
                    }
                    // Looked up after the insertion, which may have moved the values.
                    CopyValues(*set->type, slot, set->Value(set->Find(prefabIndex)), 1);
                }
            }
        }

        for (const RowRun& run : m_ScratchRuns)
            ObserveRowsAdded(run.archetype, run.chunk, run.firstRow, run.count, c_NoArchetype);
        if (prefabSparse)
            for (const SparseSet* set : m_SparseSets)
                if (set->Find(prefabIndex) != HashIndex::c_NotFound)
                    for (uint32_t t = 0; t < count; ++t)
                        ObserveSparseAdded(set->type->id, outEntities[t]);
        return result;
    }

    uint8_t* WorldImpl::ScratchBytes(size_t bytes) noexcept
    {
        if (bytes > m_ScratchBlockSize)
        {
            HeapFree(m_ScratchBlock);
            m_ScratchBlockSize = 0;
            m_ScratchBlock = static_cast<uint8_t*>(HeapAllocate(bytes, c_ColumnAlignment));
            if (m_ScratchBlock)
                m_ScratchBlockSize = bytes;
        }
        return m_ScratchBlock;
    }

    void WorldImpl::RemapEntities(const ComponentType& type, uint8_t* value, uint32_t count) const noexcept
    {
        if (type.entityOffsets.empty() || !value)
            return;
        for (uint32_t k = 0; k < count; ++k)
        {
            uint8_t* element = value + size_t(k) * type.desc.size;
            for (const uint32_t offset : type.entityOffsets)
            {
                Entity entity;
                std::memcpy(&entity, element + offset, sizeof(entity));
                const RemapEntry* first = m_ScratchRemap.begin();
                const RemapEntry* last = m_ScratchRemap.end();
                const RemapEntry* found = std::lower_bound(first, last, entity,
                    [](const RemapEntry& entry, Entity e) { return uint64_t(entry.source) < uint64_t(e); });
                if (found != last && found->source == entity)
                    std::memcpy(element + offset, &found->copy, sizeof(Entity));
            }
        }
    }

    // ----------------------------------------------------------------------------- IWorld structural calls

    nvrhi::FRESULT WorldImpl::CreateEntities(const EntityCreateDesc& desc, Entity* outEntities) noexcept
    {
        if (!CanChangeStructure())
        {
            ReportValidation("CreateEntities", "structural changes are deferred inside phases and observers (3.3.3)");
            DiscardCreateValues(desc, 0);   // ValueTransfer::Relocate: the caller gave the values up
            return nvrhi::FE_INVALID_ARGS;
        }
        nvrhi::FRESULT result = CreateEntitiesInternal(desc, outEntities, GetStampVersion());
        KeepFirstFailure(result, FlushObservers());
        return result;
    }

    nvrhi::FRESULT WorldImpl::DestroyEntities(const Entity* entities, uint32_t count) noexcept
    {
        if (!CanChangeStructure())
        {
            ReportValidation("DestroyEntities", "structural changes are deferred inside phases and observers (3.3.3)");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (count != 0 && !entities)
            return nvrhi::FE_INVALID_ARGS;
        // The list may be a chunk's Entity column, which the removals rewrite.
        m_ScratchEntities.assign(entities, entities + count);
        nvrhi::FRESULT result = DestroyEntitiesInternal(m_ScratchEntities.data(), count, GetStampVersion());
        KeepFirstFailure(result, FlushObservers());
        return result;
    }

    nvrhi::FRESULT WorldImpl::AddComponent(Entity entity, ComponentId component, const void* value) noexcept
    {
        if (!CanChangeStructure())
        {
            ReportValidation("AddComponent", "structural changes are deferred inside phases and observers (3.3.3)");
            return nvrhi::FE_INVALID_ARGS;
        }
        nvrhi::FRESULT result = AddComponentsInternal(&entity, 1, component, value, ValueTransfer::Copy,
                                                      GetStampVersion());
        KeepFirstFailure(result, FlushObservers());
        return result;
    }

    nvrhi::FRESULT WorldImpl::RemoveComponent(Entity entity, ComponentId component) noexcept
    {
        if (!CanChangeStructure())
        {
            ReportValidation("RemoveComponent", "structural changes are deferred inside phases and observers (3.3.3)");
            return nvrhi::FE_INVALID_ARGS;
        }
        nvrhi::FRESULT result = RemoveComponentsInternal(&entity, 1, component, GetStampVersion());
        KeepFirstFailure(result, FlushObservers());
        return result;
    }

    nvrhi::FRESULT WorldImpl::Instantiate(Entity prefab, uint32_t count, Entity* outEntities) noexcept
    {
        if (!CanChangeStructure())
        {
            ReportValidation("Instantiate", "structural changes are deferred inside phases and observers (3.3.3)");
            return nvrhi::FE_INVALID_ARGS;
        }
        nvrhi::FRESULT result = InstantiateInternal(prefab, count, outEntities, GetStampVersion());
        KeepFirstFailure(result, FlushObservers());
        return result;
    }

    nvrhi::FRESULT WorldImpl::CopyEntitiesFrom(IWorld* source, const Entity* entities, uint32_t count,
                                               Entity* outEntities) noexcept
    {
        const char* operation = "CopyEntitiesFrom";
        if (!CanChangeStructure())
        {
            ReportValidation(operation, "structural changes are deferred inside phases and observers (3.3.3)");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!source || (count != 0 && (!entities || !outEntities)))
            return nvrhi::FE_INVALID_ARGS;
        if (count == 0)
            return nvrhi::FS_OK;
        const uint64_t version = GetStampVersion();
        // The list is read after the copies' ids are written, and the output may be the list itself.
        m_ScratchEntities.assign(entities, entities + count);
        const Entity* list = m_ScratchEntities.data();

        // The source's component ids mapped to ours, by GUID. Only the public interface of the source is used, so
        // the source may be a world of another module.
        const uint32_t sourceTypeCount = source->GetComponentTypeCount();
        m_ScratchTypeMap.assign(sourceTypeCount, HashIndex::c_NotFound);
        for (uint32_t s = 0; s < sourceTypeCount; ++s)
        {
            const ComponentTypeDesc* desc = source->GetComponentTypeDesc(ComponentId(s));
            if (!desc || Has(desc->flags, ComponentFlags::Resource))
                continue;
            const uint32_t local = FindType(desc->uuid);
            if (local != HashIndex::c_NotFound)
                m_ScratchTypeMap[s] = SameLayout(m_Types[local]->desc, *desc) ? local : c_LayoutClash;
        }

        // Everything that can be checked is checked before anything changes: no id is taken, no row or archetype
        // made, when an entity is stale, a type is missing, clashes or cannot be copied, or a copy's row would not
        // fit a chunk of this world. Only running out of memory midway leaves the copies made before it.
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!source->IsAlive(list[i]))
            {
                ReportStale(operation, list[i]);
                return nvrhi::FE_INVALID_ARGS;
            }
            m_ScratchIds.clear();
            for (uint32_t s = 0; s < sourceTypeCount; ++s)
            {
                if (!source->HasComponent(list[i], ComponentId(s)))
                    continue;
                const uint32_t local = m_ScratchTypeMap[s];
                if (local == HashIndex::c_NotFound)
                {
                    ReportValidation(operation, "%s is not registered in this world",
                                     NameOf(*source->GetComponentTypeDesc(ComponentId(s))));
                    return nvrhi::FE_NOT_FOUND;
                }
                if (local == c_LayoutClash)
                {
                    ReportValidation(operation, "%s has another layout in this world",
                                     NameOf(*source->GetComponentTypeDesc(ComponentId(s))));
                    return nvrhi::FE_INVALID_ARGS;
                }
                if (!m_Types[local]->copyable)
                    return nvrhi::FE_UNSUPPORTED;
                if (m_Types[local]->kind != StorageKind::Sparse)
                    m_ScratchIds.push_back(ComponentId(local));
            }
            std::sort(m_ScratchIds.begin(), m_ScratchIds.end());
            if (!RowFits(m_ScratchIds.data(), uint32_t(m_ScratchIds.size())))
            {
                ReportValidation(operation, "the copy of entity %u:%u would not fit a chunk of this world",
                                 GetEntityIndex(list[i]), GetEntityGeneration(list[i]));
                return nvrhi::FE_INVALID_ARGS;
            }
        }

        // The copies' ids first, so that Entity fields can name them.
        nvrhi::FRESULT result = AllocateEntityIds(count, outEntities);
        if (NVRHI_FAILED(result))
            return result;
        m_ScratchRemap.clear();
        for (uint32_t i = 0; i < count; ++i)
            m_ScratchRemap.push_back(RemapEntry{ list[i], outEntities[i], i, 0 });
        std::sort(m_ScratchRemap.begin(), m_ScratchRemap.end(), [](const RemapEntry& a, const RemapEntry& b)
        {
            return uint64_t(a.source) < uint64_t(b.source) || (a.source == b.source && a.order < b.order);
        });

        m_ScratchRuns.clear();
        for (uint32_t i = 0; i < count; ++i)
        {
            const Entity entity = list[i];
            const Entity copy = outEntities[i];

            // The copy's archetype components (ascending local ids) with the source id of each.
            m_ScratchTerms.clear();
            for (uint32_t s = 0; s < sourceTypeCount; ++s)
            {
                const uint32_t local = m_ScratchTypeMap[s];
                if (local >= c_LayoutClash || m_Types[local]->kind == StorageKind::Sparse ||
                    !source->HasComponent(entity, ComponentId(s)))
                    continue;
                m_ScratchTerms.push_back(SortedTerm{ ComponentId(local), s });
            }
            std::sort(m_ScratchTerms.begin(), m_ScratchTerms.end(),
                      [](const SortedTerm& a, const SortedTerm& b) { return a.component < b.component; });

            // Shared values, with their Entity fields remapped where they have any.
            size_t sharedBytes = 0;
            for (const SortedTerm& term : m_ScratchTerms)
            {
                const ComponentType& type = *m_Types[uint32_t(term.component)];
                if (type.kind == StorageKind::Shared && !type.entityOffsets.empty())
                    sharedBytes = AlignUp(sharedBytes, c_ColumnAlignment) + type.desc.size;
            }
            uint8_t* scratch = sharedBytes != 0 ? ScratchBytes(sharedBytes) : nullptr;
            if (sharedBytes != 0 && !scratch)
            {
                KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);
                ReleaseIndex(GetEntityIndex(copy));
                outEntities[i] = Entity::Null;
                continue;
            }
            m_ScratchIds.clear();
            m_ScratchShared.clear();
            size_t scratchOffset = 0;
            for (const SortedTerm& term : m_ScratchTerms)
            {
                const ComponentType& type = *m_Types[uint32_t(term.component)];
                const void* shared = nullptr;
                if (type.kind == StorageKind::Shared)
                {
                    shared = source->GetComponent(entity, ComponentId(term.source));
                    if (!type.entityOffsets.empty())
                    {
                        scratchOffset = AlignUp(scratchOffset, c_ColumnAlignment);
                        uint8_t* local = scratch + scratchOffset;
                        scratchOffset += type.desc.size;
                        CopyValues(type, local, shared, 1);
                        RemapEntities(type, local, 1);
                        shared = local;
                    }
                }
                m_ScratchIds.push_back(term.component);
                m_ScratchShared.push_back(shared);
            }
            const uint32_t archetypeIndex =
                FindOrCreateArchetype(m_ScratchIds.data(), uint32_t(m_ScratchIds.size()), m_ScratchShared.data());
            scratchOffset = 0;
            for (const SortedTerm& term : m_ScratchTerms)
            {
                const ComponentType& type = *m_Types[uint32_t(term.component)];
                if (type.kind == StorageKind::Shared && !type.entityOffsets.empty())
                {
                    scratchOffset = AlignUp(scratchOffset, c_ColumnAlignment);
                    DestructValues(type, scratch + scratchOffset, 1);
                    scratchOffset += type.desc.size;
                }
            }
            if (IsArchetypeError(archetypeIndex))
            {
                // Out of memory: the fit was checked above.
                KeepFirstFailure(result, ArchetypeErrorResult(archetypeIndex));
                ReleaseIndex(GetEntityIndex(copy));
                outEntities[i] = Entity::Null;
                continue;
            }

            Archetype& archetype = *m_Archetypes[archetypeIndex];
            uint32_t chunk = 0;
            uint32_t row = 0;
            const nvrhi::FRESULT appended = AppendRows(archetype, 1, version,
                [&](uint32_t c, uint32_t r, uint32_t, uint32_t) noexcept
            {
                chunk = c;
                row = r;
            });
            if (NVRHI_FAILED(appended))
            {
                KeepFirstFailure(result, appended);
                ReleaseIndex(GetEntityIndex(copy));
                outEntities[i] = Entity::Null;
                continue;
            }
            uint8_t* memory = archetype.chunks[chunk].memory;
            reinterpret_cast<Entity*>(memory)[row] = copy;
            EntityRecord& record = m_Entities[GetEntityIndex(copy)];
            record.archetype = archetypeIndex;
            record.chunk = chunk;
            record.row = uint16_t(row);
            record.sparseCount = 0;
            ++m_AliveCount;

            for (uint32_t c = 0; c < archetype.columns.size(); ++c)
            {
                const ArchetypeColumn& column = archetype.columns[c];
                const ComponentId sourceId = ComponentId(m_ScratchTerms[c].source);
                if (column.kind == StorageKind::Data)
                {
                    uint8_t* dst = memory + column.offset + size_t(row) * column.stride;
                    CopyValues(*column.type, dst, source->GetComponent(entity, sourceId), 1);
                    RemapEntities(*column.type, dst, 1);
                }
                else if (column.kind == StorageKind::Buffer)
                {
                    BufferHeader& dst = reinterpret_cast<BufferHeader*>(memory + column.offset)[row];
                    const BufferHeader* src = static_cast<const BufferHeader*>(source->GetComponent(entity, sourceId));
                    if (!CopyBuffer(column, dst, archetype.InlineData(chunk, c, row), *src))
                        KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);
                    RemapEntities(*column.type, static_cast<uint8_t*>(dst.data), dst.count);
                }
                // A Chunk component is the copy's new chunk's own value; Tags and Shared values have no row.
                if (column.enableable)
                    SetMaskBit(archetype.EnabledMask(chunk, c), row, source->IsEnabled(entity, sourceId));
            }
            m_ScratchRuns.push_back(RowRun{ archetypeIndex, chunk, row, 1 });

            for (uint32_t s = 0; s < sourceTypeCount; ++s)
            {
                const uint32_t local = m_ScratchTypeMap[s];
                if (local >= c_LayoutClash || m_Types[local]->kind != StorageKind::Sparse ||
                    !source->HasComponent(entity, ComponentId(s)))
                    continue;
                const ComponentType& type = *m_Types[local];
                void* slot = SparseInsert(*m_SparseSets[type.sparseSet], copy, record);
                if (!slot)
                {
                    KeepFirstFailure(result, nvrhi::FE_OUT_OF_MEMORY);
                    continue;
                }
                if (type.desc.size != 0)
                {
                    // Looked up after the insertion, which may have moved the values (when source is this world).
                    CopyValues(type, slot, source->GetComponent(entity, ComponentId(s)), 1);
                    RemapEntities(type, static_cast<uint8_t*>(slot), 1);
                }
            }
        }

        for (const RowRun& run : m_ScratchRuns)
            ObserveRowsAdded(run.archetype, run.chunk, run.firstRow, run.count, c_NoArchetype);
        for (uint32_t i = 0; i < count; ++i)
        {
            if (outEntities[i] == Entity::Null)
                continue;
            for (uint32_t s = 0; s < sourceTypeCount; ++s)
            {
                const uint32_t local = m_ScratchTypeMap[s];
                if (local < c_LayoutClash && m_Types[local]->kind == StorageKind::Sparse &&
                    HasComponent(outEntities[i], ComponentId(local)))
                    ObserveSparseAdded(ComponentId(local), outEntities[i]);
            }
        }
        KeepFirstFailure(result, FlushObservers());
        return result;
    }

    // ------------------------------------------------------------------------------- later steps (E1.2-E1.5)

    nvrhi::FRESULT WorldImpl::CreateQuery(const QueryDesc& desc, IQuery** ppQuery) noexcept
    {
        return CreateQueryObject(*this, desc, QueryWorldReference::Held, ppQuery);   // EcsQuery.cpp
    }

    // CreateCommandBuffer, Playback and ResolvePlaceholder are in EcsCommandBuffer.cpp (E1.3).

    // AddObserver and RemoveObserver are in EcsObserver.cpp (E1.4).

    // SaveSnapshot, LoadSnapshot and ComputeHash are in EcsSnapshot.cpp (E1.5).

    // ----------------------------------------------------------------------------------------- diagnostics

    WorldStats& WorldImpl::GetStats(WorldStats& retVal) noexcept
    {
        retVal = WorldStats{};
        retVal.entityCount = m_AliveCount;
        retVal.entityCapacity = uint32_t(m_Entities.capacity() - 1);
        retVal.componentTypeCount = uint32_t(m_Types.size());
        retVal.archetypeCount = uint32_t(m_Archetypes.size());
        retVal.sparseSetCount = uint32_t(m_SparseSets.size());
        for (const Archetype* archetype : m_Archetypes)
        {
            retVal.chunkCount += archetype->GetChunkCount();
            retVal.chunkBytes += uint64_t(archetype->GetChunkCount()) * archetype->chunkSize;
            retVal.rowBytes += uint64_t(archetype->entityCount) * archetype->rowSize;
        }
        retVal.bufferHeapBytes = m_BufferHeapBytes.load(std::memory_order_relaxed);
        retVal.commandsPlayedBack = m_CommandsPlayedBack;
        retVal.observerBatches = m_ObserverBatches;
        retVal.version = m_Version;
        retVal.structuralVersion = m_StructuralVersion;
        return retVal;
    }

    nvrhi::FRESULT WorldImpl::GetArchetypeInfo(uint32_t archetype, ArchetypeInfo& out) noexcept
    {
        if (archetype >= m_Archetypes.size())
            return nvrhi::FE_INVALID_ARGS;
        const Archetype& a = *m_Archetypes[archetype];
        out.components = a.components.data();
        out.componentCount = uint32_t(a.components.size());
        out.entityCount = a.entityCount;
        out.chunkCount = a.GetChunkCount();
        out.chunkSize = a.chunkSize;
        out.rowSize = a.rowSize;
        out.rowsPerChunk = a.rowsPerChunk;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT WorldImpl::GetChunkInfo(uint32_t archetype, uint32_t chunk, ChunkInfo& out) noexcept
    {
        (void)archetype;
        (void)chunk;
        out = ChunkInfo{};
        return nvrhi::FE_NOT_IMPLEMENT;   // a stub of plan step E7, item E7.0; item E7.7 implements it
    }
}

namespace donut::engine
{
    nvrhi::FRESULT CreateWorld(const WorldDesc& desc, IWorld** ppWorld) noexcept
    {
        if (!ppWorld)
            return nvrhi::FE_INVALID_ARGS;
        *ppWorld = nullptr;
        auto isPowerOfTwo = [](uint32_t value) { return value != 0 && (value & (value - 1)) == 0; };
        if (!isPowerOfTwo(desc.chunkSize) || !isPowerOfTwo(desc.largeChunkSize) || desc.chunkSize < 1024 ||
            desc.largeChunkSize < desc.chunkSize || desc.largeChunkSize > (1u << 20))
            return nvrhi::FE_INVALID_ARGS;

        ecs::WorldImpl* world = nullptr;
        try
        {
            world = MAKE_RC_OBJ(ecs::WorldImpl, desc);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        if (!world)
            return nvrhi::FE_OUT_OF_MEMORY;
        const nvrhi::FRESULT result = world->Initialize();
        if (NVRHI_FAILED(result))
        {
            world->Release();
            return result;
        }
        *ppWorld = world;
        return nvrhi::FS_OK;
    }
}
