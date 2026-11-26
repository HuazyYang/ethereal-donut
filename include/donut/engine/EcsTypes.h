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

// Common types of the entity component system (aggregate docs/ecs-design.md, sections 4 and 9.1): entity and
// component ids, versions, component descriptors with reflected fields, query terms, chunk and column views, and the
// macros that give C++ types their descriptors. The structs follow nvrhi's ABI rules (com-interface-abi.md section 4).
// The Ecs* headers name no scene, render or simulation type (design 3.2).

#pragma once

#include <nvrhi/core/types.h>
// SAL annotations, as nvrhi.h includes them: Salieri's fallbacks would clash with Windows SDK headers included later.
#if defined(_MSC_VER)
#include <sal.h>
#else
#include <nvrhi/core/salieri.h>
#endif
#include <donut/core/math/math.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

// The flag enums keep nvrhi's `None`; X11 defines it as a macro. The declaration macros below never name it.
#if defined(None)
#define DONUT_ECS_TYPES_RESTORE_NONE
#pragma push_macro("None")
#undef None
#endif

namespace donut::engine
{
    // ------------------------------------------------------------------------------------------- flag operators

    // Bit operators for the flag enums of the ECS headers. An enum opts in by specialising this trait; the
    // operators are found by argument-dependent lookup.
    template <typename T> struct EnableEcsFlagOperators : std::false_type { };

    template <typename T, std::enable_if_t<EnableEcsFlagOperators<T>::value, int> = 0>
    constexpr T operator|(T a, T b) noexcept { return T(std::underlying_type_t<T>(a) | std::underlying_type_t<T>(b)); }
    template <typename T, std::enable_if_t<EnableEcsFlagOperators<T>::value, int> = 0>
    constexpr T operator&(T a, T b) noexcept { return T(std::underlying_type_t<T>(a) & std::underlying_type_t<T>(b)); }
    template <typename T, std::enable_if_t<EnableEcsFlagOperators<T>::value, int> = 0>
    constexpr T operator~(T a) noexcept { return T(~std::underlying_type_t<T>(a)); }
    template <typename T, std::enable_if_t<EnableEcsFlagOperators<T>::value, int> = 0>
    constexpr bool operator!(T a) noexcept { return std::underlying_type_t<T>(a) == 0; }
    template <typename T, std::enable_if_t<EnableEcsFlagOperators<T>::value, int> = 0>
    constexpr T& operator|=(T& a, T b) noexcept { return a = a | b; }
    template <typename T, std::enable_if_t<EnableEcsFlagOperators<T>::value, int> = 0>
    constexpr T& operator&=(T& a, T b) noexcept { return a = a & b; }

    // ------------------------------------------------------------------------------------------- entities (4.1)

    // Index in the low 32 bits, generation in the high 32. 0 is null, and index 0 is never handed out.
    enum class Entity : uint64_t { Null = 0 };

    // The generation of a command-buffer placeholder (6.5). Generations of live entities skip it when they wrap.
    constexpr uint32_t c_PlaceholderGeneration = 0xffffffffu;

    constexpr Entity MakeEntity(uint32_t index, uint32_t generation) noexcept
    {
        return Entity(uint64_t(index) | (uint64_t(generation) << 32));
    }
    constexpr uint32_t GetEntityIndex(Entity entity) noexcept { return uint32_t(uint64_t(entity)); }
    constexpr uint32_t GetEntityGeneration(Entity entity) noexcept { return uint32_t(uint64_t(entity) >> 32); }
    constexpr bool IsPlaceholder(Entity entity) noexcept { return GetEntityGeneration(entity) == c_PlaceholderGeneration; }

    // ------------------------------------------------------------------------------------- versions (4.5)

    // Versions are 64-bit and never wrap. The world hands them out in increasing order: one per system run, in schedule
    // order (EcsScheduler.h), and one more after each phase for writes made outside phases. 0 is never stamped.
    //
    // As a sinceVersion: every chunk counts as changed.
    constexpr uint64_t c_AllVersions = 0;
    // As a writeVersion: the world's current version (outside phases only).
    constexpr uint64_t c_WorldVersion = 0;

    // ------------------------------------------------------------------------------------ component types (4.2)

    // Dense per world, in registration order. Resources (4.9) share the id space, so that systems declare access to
    // both in one list.
    enum class ComponentId : uint32_t { Invalid = 0xffffffffu };

    enum class ComponentFlags : uint32_t
    {
        None        = 0,
        Tag         = 1u << 0,   // zero-size; presence only
        Sparse      = 1u << 1,   // stored in a sparse set: adding or removing it moves no entity; queries see it as a
                                 // per-row presence mask, never as data (4.3)
        Shared      = 1u << 2,   // one value per chunk that partitions chunks; read-only through queries (4.4)
        Chunk       = 1u << 3,   // one value per chunk, not a partition key (4.4)
        Enableable  = 1u << 4,   // a per-row enabled bit; toggling is not a structural change
        Buffer      = 1u << 5,   // a dynamic buffer: size and alignment describe one element (4.3)
        Relocatable = 1u << 6,   // can be moved with memcpy; required for GpuMirrored and simulation columns
        GpuMirrored = 1u << 7,   // the render service mirrors the column (7.2)
        NoSerialize = 1u << 8,   // skipped by snapshots and world hashes
        Resource    = 1u << 9,   // a world resource: one value per world, never on an entity (4.9)
        RowTracked  = 1u << 10,  // a change version per row beside the chunk's (D9): ChunkView::rowVersions. Only for
                                 // components with a value per row (not Tag, Sparse, Shared, Chunk, Buffer, Resource).
                                 // A storage option, not a layout: the world that registers the GUID first decides it,
                                 // a later registration with or without it gets the same id, and snapshots, hashes
                                 // and copies between worlds ignore it. The versions live outside the chunk's rows,
                                 // so rows per chunk do not change.
    };
    template <> struct EnableEcsFlagOperators<ComponentFlags> : std::true_type { };

    // Lifecycle hooks: C function pointers, so that a component type can be registered from any module. A null hook
    // means the trivial operation (zero-fill, nothing, memcpy, memcpy), where a memcpy copy needs a Relocatable type.
    // Registration enforces (D14): a Relocatable type has no destruct, relocate or copy hook; a type with a destruct
    // hook has a relocate hook; a type that is neither Relocatable nor has a copy hook is never copied (copies and
    // Instantiate fail with FE_UNSUPPORTED).
    typedef void (*PFN_ComponentConstruct)(void* dst, uint32_t count);
    typedef void (*PFN_ComponentDestruct)(void* dst, uint32_t count);
    typedef void (*PFN_ComponentRelocate)(void* dst, void* src, uint32_t count);   // move, then destroy the source
    typedef void (*PFN_ComponentCopy)(void* dst, const void* src, uint32_t count);

    // Reflection (4.2, 11.4): the scalar kind of a field, and how its scalars are arranged.
    enum class FieldType : uint8_t
    {
        Boolean, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float, Double,
        Entity,     // remapped where placeholders and snapshots need it
        Opaque,     // bytes: shown raw, copied, compared and hashed as they are, so keep them free of padding
    };
    enum class FieldShape : uint8_t { Scalar, Vector, Quaternion, Matrix, Affine, Box, Array };

    struct FieldDesc
    {
        const char* name;
        uint32_t offset;
        uint32_t scalarCount;   // scalars of `type` in the field (3 for a double3, 12 for a daffine3); bytes if Opaque
        FieldType type;
        FieldShape shape;
        uint8_t reserved[6];
    };
    static_assert(sizeof(void*) != 8 || sizeof(FieldDesc) == 24, "FieldDesc layout");

    // Lifetime: the world keeps a copy of the descriptor but not of what it points to. `name`, `fields` and the hooks
    // (the module that holds their code) must outlive every world the type is registered with.
    //
    // Values: a component's bytes are compared (Shared keys, 4.4) and hashed (IWorld::ComputeHash, 11.1) only over the
    // bytes its fields cover, so padding never takes part; a Shared component must list fields. A non-trivial
    // component may own memory only through nvrhi containers or COM references, because another module may destroy it
    // (com-interface-abi.md section 6).
    struct ComponentTypeDesc
    {
        nvrhi::GUID uuid;                       // identity across modules and worlds
        const char* name;
        uint32_t size;                          // 0 for a Tag; the element size for a Buffer
        uint32_t alignment;
        ComponentFlags flags;
        uint32_t bufferInlineCapacity;          // Buffer only: elements stored in the chunk before spilling to the heap
        PFN_ComponentConstruct construct;
        PFN_ComponentDestruct destruct;
        PFN_ComponentRelocate relocate;
        PFN_ComponentCopy copy;
        const FieldDesc* fields;
        uint32_t fieldCount;
        uint32_t reserved;
    };
    static_assert(sizeof(void*) != 8 || sizeof(ComponentTypeDesc) == 88, "ComponentTypeDesc layout");
    static_assert(offsetof(ComponentTypeDesc, name) == 16, "ComponentTypeDesc layout");

    // One row of a Buffer component's column (4.3). `data` addresses the row's inline storage, which the chunk keeps
    // apart from the column, or heap storage once the buffer outgrew it. Resize through IWorld::ResizeBuffer or
    // ICommandBuffer::AppendToBuffer, never by writing `count`.
    struct BufferHeader
    {
        void* data;
        uint32_t count;
        uint32_t capacity;
    };
    static_assert(sizeof(void*) != 8 || sizeof(BufferHeader) == 16, "BufferHeader layout");

    // ------------------------------------------------------------------------------------ creating entities

    enum class ValueTransfer : uint8_t
    {
        Copy,       // values are copied with the copy hook
        Relocate,   // values are moved in with the relocate hook and destroyed: the caller gives the storage up
    };

    // entityCount entities with the same components. `values[i]`, when present, points at entityCount consecutive
    // values of components[i]; a null array or entry default-constructs. Tags take no values, and Buffer components
    // start empty (their entry must be null).
    struct EntityCreateDesc
    {
        const ComponentId* components;
        const void* const* values;          // optional
        const Entity* reservedEntities;     // optional, immediate creation only: entityCount ids from ReserveEntities
        uint32_t componentCount;
        uint32_t entityCount;
        ValueTransfer transfer;
        uint8_t reserved[7];
    };
    static_assert(sizeof(void*) != 8 || sizeof(EntityCreateDesc) == 40, "EntityCreateDesc layout");

    // ------------------------------------------------------------------------------------------ queries (6.1)

    enum class Access : uint8_t
    {
        Filter,   // no data: a presence filter (a tag, a sparse component) or a Without term; neither conflicts nor
                  // gets a column
        Read,
        Write,    // rejected for Shared components
    };

    enum class TermOp : uint8_t
    {
        All,        // required
        Any,        // the Any terms of a query form one group: at least one of them is required
        Optional,   // never restricts matching; the column is null where the component is absent
        Without,    // excluded (named Without, not None: X11 defines None as a macro)
        Changed,    // required, and the chunk's column version is newer than the request's sinceVersion
        Added,      // required, and rows were added to the chunk after sinceVersion
        Enabled,    // required (Enableable); the typed layer skips rows where it is disabled
        Disabled,   // required (Enableable); the typed layer skips rows where it is enabled
    };

    struct QueryTerm
    {
        ComponentId component;
        TermOp op;
        Access access;
        uint8_t reserved[2];
    };
    static_assert(sizeof(QueryTerm) == 8, "QueryTerm layout");

    struct QueryDesc
    {
        const QueryTerm* terms;
        uint32_t termCount;
        uint32_t reserved;
        const char* debugName;
    };
    static_assert(sizeof(void*) != 8 || sizeof(QueryDesc) == 24, "QueryDesc layout");

    // A term with data: Read or Write, and not Without. Data terms on Sparse components are rejected.
    constexpr bool IsDataTerm(const QueryTerm& term) noexcept
    {
        return term.access != Access::Filter && term.op != TermOp::Without;
    }

    // One chunk as a query sees it. Valid until the next sync point. Row filters are not applied: rows that Enabled,
    // Disabled or Sparse terms exclude are still in [0, count); the typed layer's ForEach skips them.
    //
    // Row versions (RowTracked components, D9): rowVersions[term][row] is the version of the row's last write: rows
    // created or moved are stamped with the version of the change, IWorld::GetComponentForWrite, SetComponent and the
    // other writes of one entity stamp its row, and a writer that knows its rows stamps them with StampRow. A write
    // that stamps only the chunk (a Write term whose rows are not stamped, a simulation domain's column view) still
    // counts: a chunk whose version is newer than the newest row version is wholly changed, and the next time the
    // chunk is handed out (or a row of it is stamped) every row is stamped with that chunk version first. So a reader
    // that got the chunk from a Changed term takes the rows with IsRowChanged. rowVersions[term][-1] is the column's
    // newest row version in this chunk, which StampRow keeps.
    struct ChunkView
    {
        const Entity* entities;
        void* const* columns;                  // one per data term, in term order: the column's first row; for a Shared
                                               // or Chunk component the chunk's one value; null for an absent
                                               // Optional or Any term
        const uint64_t* const* enabledMasks;   // one per term: bit r of word r / 64 set when row r is enabled; null
                                               // where the component is not Enableable or is absent
        const uint64_t* const* presenceMasks;  // one per term: bit r set when row r has the Sparse component; null for
                                               // components stored in the archetype
        uint32_t count;
        uint32_t chunkIndex;                   // position among the query's matched chunks; stable until the next
                                               // sync point; part of command sort keys (6.5)
        uint64_t* const* rowVersions;          // one per term: the rows' change versions of a RowTracked component
                                               // (above); null for other terms and absent components

        bool IsRowEnabled(uint32_t term, uint32_t row) const noexcept
        {
            const uint64_t* mask = enabledMasks[term];
            return !mask || ((mask[row >> 6] >> (row & 63)) & 1) != 0;
        }
        bool IsRowPresent(uint32_t term, uint32_t row) const noexcept
        {
            const uint64_t* mask = presenceMasks[term];
            return !mask || ((mask[row >> 6] >> (row & 63)) & 1) != 0;
        }
        // Whether the row was written after sinceVersion; true for a term without row versions.
        bool IsRowChanged(uint32_t term, uint32_t row, uint64_t sinceVersion) const noexcept
        {
            const uint64_t* versions = rowVersions[term];
            return !versions || versions[row] > sinceVersion;
        }
        // For the writer of a Write term: stamps the row it wrote with its version (the request's writeVersion; a
        // c_WorldVersion request passes IWorld::GetVersion()). Nothing for a term without row versions. One thread
        // per chunk and column, as the scheduler runs writers.
        void StampRow(uint32_t term, uint32_t row, uint64_t version) const noexcept
        {
            uint64_t* versions = rowVersions[term];
            if (!versions)
                return;
            if (versions[row] < version)
                versions[row] = version;
            if (versions[-1] < version)
                versions[-1] = version;
        }
    };
    static_assert(sizeof(void*) != 8 || sizeof(ChunkView) == 48, "ChunkView layout");

    // One chunk's column of one component, as simulation domains exchange it (8.2). All views a domain gets follow the
    // chunk order of its lifetime query, so views of different components with the same chunkIndex line up row by row.
    struct ColumnView
    {
        void* base;
        uint32_t stride;
        uint32_t count;
        ComponentId component;
        uint32_t chunkIndex;
        uint64_t version;
    };
    static_assert(sizeof(void*) != 8 || sizeof(ColumnView) == 32, "ColumnView layout");

    // ------------------------------------------------------- C++ component types (header-only, outside the ABI)

    namespace detail
    {
        template <typename T>
        constexpr FieldType ScalarFieldType() noexcept
        {
            if constexpr (std::is_same_v<T, bool>)
                return FieldType::Boolean;
            else if constexpr (std::is_same_v<T, Entity>)
                return FieldType::Entity;
            else if constexpr (std::is_enum_v<T>)
                return ScalarFieldType<std::underlying_type_t<T>>();
            else if constexpr (std::is_integral_v<T>)
            {
                constexpr bool s = std::is_signed_v<T>;
                if constexpr (sizeof(T) == 1) return s ? FieldType::Int8 : FieldType::UInt8;
                else if constexpr (sizeof(T) == 2) return s ? FieldType::Int16 : FieldType::UInt16;
                else if constexpr (sizeof(T) == 4) return s ? FieldType::Int32 : FieldType::UInt32;
                else return s ? FieldType::Int64 : FieldType::UInt64;
            }
            else if constexpr (std::is_same_v<T, float>)
                return FieldType::Float;
            else if constexpr (std::is_same_v<T, double>)
                return FieldType::Double;
            else
                return FieldType::Opaque;
        }

        // How a field type is reflected. Scalars, enums, Entity, C arrays and the donut math types are known; any other
        // type is Opaque.
        template <typename T, typename = void>
        struct FieldTraits
        {
            static constexpr FieldType type = ScalarFieldType<T>();
            static constexpr FieldShape shape = FieldShape::Scalar;
            static constexpr uint32_t scalarCount = type == FieldType::Opaque ? uint32_t(sizeof(T)) : 1u;
        };
        template <typename T, size_t N>
        struct FieldTraits<T[N], void>
        {
            static constexpr FieldType type = FieldTraits<T>::type;
            static constexpr FieldShape shape = FieldShape::Array;
            static constexpr uint32_t scalarCount = FieldTraits<T>::scalarCount * uint32_t(N);
        };
        template <typename T, int n>
        struct FieldTraits<math::vector<T, n>, void>
        {
            static constexpr FieldType type = ScalarFieldType<T>();
            static constexpr FieldShape shape = FieldShape::Vector;
            static constexpr uint32_t scalarCount = uint32_t(n);
        };
        template <typename T>
        struct FieldTraits<math::quaternion<T>, void>
        {
            static constexpr FieldType type = ScalarFieldType<T>();
            static constexpr FieldShape shape = FieldShape::Quaternion;
            static constexpr uint32_t scalarCount = 4;
        };
        template <typename T, int rows, int cols>
        struct FieldTraits<math::matrix<T, rows, cols>, void>
        {
            static constexpr FieldType type = ScalarFieldType<T>();
            static constexpr FieldShape shape = FieldShape::Matrix;
            static constexpr uint32_t scalarCount = uint32_t(rows * cols);
        };
        template <typename T, int n>
        struct FieldTraits<math::affine<T, n>, void>
        {
            static constexpr FieldType type = ScalarFieldType<T>();
            static constexpr FieldShape shape = FieldShape::Affine;
            static constexpr uint32_t scalarCount = uint32_t(n * n + n);
        };
        template <typename T, int n>
        struct FieldTraits<math::box<T, n>, void>
        {
            static constexpr FieldType type = ScalarFieldType<T>();
            static constexpr FieldShape shape = FieldShape::Box;
            static constexpr uint32_t scalarCount = uint32_t(2 * n);
        };

        template <typename Field>
        constexpr FieldDesc MakeFieldDesc(const char* name, size_t offset) noexcept
        {
            using Traits = FieldTraits<std::remove_cv_t<Field>>;
            return FieldDesc{ name, uint32_t(offset), Traits::scalarCount, Traits::type, Traits::shape, {} };
        }

        template <typename T>
        constexpr FieldDesc TooManyFields() noexcept
        {
            static_assert(sizeof(T) == 0, "DONUT_ECS_FIELDS takes at most 16 fields");
            return FieldDesc{};
        }

        template <typename T> void ConstructComponents(void* dst, uint32_t count) noexcept
        {
            for (uint32_t i = 0; i < count; ++i)
                new (static_cast<T*>(dst) + i) T();
        }
        template <typename T> void DestructComponents(void* dst, uint32_t count) noexcept
        {
            for (uint32_t i = 0; i < count; ++i)
                (static_cast<T*>(dst) + i)->~T();
        }
        template <typename T> void RelocateComponents(void* dst, void* src, uint32_t count) noexcept
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                T* from = static_cast<T*>(src) + i;
                new (static_cast<T*>(dst) + i) T(std::move(*from));
                from->~T();
            }
        }
        template <typename T> void CopyComponents(void* dst, const void* src, uint32_t count) noexcept
        {
            for (uint32_t i = 0; i < count; ++i)
                new (static_cast<T*>(dst) + i) T(static_cast<const T*>(src)[i]);
        }

        // The descriptor that DONUT_ECS_COMPONENT builds. Hooks are null exactly where the trivial operation is right;
        // a type that is not trivially copyable is not Relocatable and gets every hook it supports.
        template <typename T>
        ComponentTypeDesc MakeComponentTypeDesc(const nvrhi::GUID& uuid, const char* name, ComponentFlags flags,
                                                uint32_t bufferInlineCapacity, const FieldDesc* fields,
                                                uint32_t fieldCount) noexcept
        {
            constexpr bool isTag = std::is_empty_v<T>;
            constexpr bool trivial = std::is_trivially_copyable_v<T>;
            ComponentTypeDesc desc{};
            desc.uuid = uuid;
            desc.name = name;
            desc.size = isTag ? 0u : uint32_t(sizeof(T));
            desc.alignment = isTag ? 1u : uint32_t(alignof(T));
            desc.flags = flags | (isTag ? ComponentFlags::Tag : ComponentFlags::None)
                               | (trivial ? ComponentFlags::Relocatable : ComponentFlags::None);
            desc.bufferInlineCapacity = bufferInlineCapacity;
            if constexpr (!isTag)
            {
                if constexpr (!std::is_trivially_default_constructible_v<T>)
                    desc.construct = &ConstructComponents<T>;
                if constexpr (!std::is_trivially_destructible_v<T>)
                    desc.destruct = &DestructComponents<T>;
                if constexpr (!trivial)
                {
                    desc.relocate = &RelocateComponents<T>;
                    if constexpr (std::is_copy_constructible_v<T>)
                        desc.copy = &CopyComponents<T>;   // otherwise copies and Instantiate fail with FE_UNSUPPORTED
                }
            }
            desc.fields = fields;
            desc.fieldCount = fieldCount;
            return desc;
        }
    }

    // Marks a Buffer component in the typed query layer: Read<BufferOf<SkinJoint>> hands out BufferHeaders.
    template <typename Element> struct BufferOf { };

    // The descriptor of a C++ component type. DONUT_ECS_COMPONENT defines DonutEcsComponentDesc and
    // DonutEcsComponentFlags next to the type, and argument-dependent lookup finds them, so a component may be
    // declared in any namespace. `flags` are the declared flags, usable in constant expressions.
    template <typename T>
    struct ComponentTraits
    {
        static constexpr ComponentFlags flags = DonutEcsComponentFlags(static_cast<const T*>(nullptr));
        static const ComponentTypeDesc& Desc() noexcept { return DonutEcsComponentDesc(static_cast<const T*>(nullptr)); }
    };
    template <typename Element>
    struct ComponentTraits<BufferOf<Element>>
    {
        static constexpr ComponentFlags flags = ComponentTraits<Element>::flags;
        static const ComponentTypeDesc& Desc() noexcept { return ComponentTraits<Element>::Desc(); }
    };

    template <typename Element>
    Element* GetBufferData(const BufferHeader& header) noexcept { return static_cast<Element*>(header.data); }
}

// ------------------------------------------------------------------------------------------- declaration macros
//
// At namespace scope, in the namespace of the type:
//
//     struct LocalTransform { dm::double3 translation; dm::dquat rotation; dm::double3 scaling; };
//     DONUT_ECS_COMPONENT(LocalTransform, "<guid>", DONUT_ECS_FIELDS(translation, rotation, scaling));
//
//     struct Static { };
//     DONUT_ECS_TAG(Static, "<guid>");
//
//     struct SkinJoint { donut::engine::Entity joint; dm::float4x4 inverseBind; };
//     DONUT_ECS_BUFFER(SkinJoint, "<guid>", 4, DONUT_ECS_FIELDS(joint, inverseBind));
//
//     DONUT_ECS_COMPONENT_EX(HierarchyDepth, "<guid>", donut::engine::ComponentFlags::Shared, 0,
//                            DONUT_ECS_FIELDS(depth));
//
// DONUT_ECS_FIELDS() lists no field (an opaque component: neither hashed nor compared); at most 16 fields. Tag,
// Relocatable and the hooks follow from the type; other flags are passed in.

#define DONUT_ECS_FIELDS(...) (__VA_ARGS__)

#define DONUT_ECS_COMPONENT(Type, StrGuid, Fields) \
    DONUT_ECS_COMPONENT_EX(Type, StrGuid, ::donut::engine::ComponentFlags{}, 0, Fields)

#define DONUT_ECS_BUFFER(Type, StrGuid, InlineCapacity, Fields) \
    DONUT_ECS_COMPONENT_EX(Type, StrGuid, ::donut::engine::ComponentFlags::Buffer, InlineCapacity, Fields)

#define DONUT_ECS_COMPONENT_EX(Type, StrGuid, Flags, BufferInlineCapacity, Fields)                                      \
    constexpr ::donut::engine::ComponentFlags DonutEcsComponentFlags(const Type*) noexcept { return (Flags); }          \
    inline const ::donut::engine::ComponentTypeDesc& DonutEcsComponentDesc(const Type*) noexcept                       \
    {                                                                                                                  \
        static const ::donut::engine::FieldDesc s_Fields[] = {                                                         \
            DONUT_ECS_DETAIL_FIELD_LIST(Type, DONUT_ECS_DETAIL_UNPAREN Fields) ::donut::engine::FieldDesc{} };         \
        static_assert(!((Flags) & ::donut::engine::ComponentFlags::Shared) || sizeof(s_Fields) > sizeof(s_Fields[0]), \
                      #Type ": a Shared component must list its fields; its key is compared over them (4.4)");        \
        static const ::donut::engine::ComponentTypeDesc s_Desc = ::donut::engine::detail::MakeComponentTypeDesc<Type>(  \
            StrGuid##_nvrhi_guid, #Type, (Flags), BufferInlineCapacity, s_Fields,                                      \
            uint32_t(sizeof(s_Fields) / sizeof(s_Fields[0]) - 1));                                                     \
        return s_Desc;                                                                                                 \
    }

#define DONUT_ECS_TAG(Type, StrGuid) DONUT_ECS_TAG_EX(Type, StrGuid, ::donut::engine::ComponentFlags{})

#define DONUT_ECS_TAG_EX(Type, StrGuid, Flags)                                                                          \
    constexpr ::donut::engine::ComponentFlags DonutEcsComponentFlags(const Type*) noexcept                             \
    {                                                                                                                  \
        return (Flags) | ::donut::engine::ComponentFlags::Tag;                                                         \
    }                                                                                                                  \
    inline const ::donut::engine::ComponentTypeDesc& DonutEcsComponentDesc(const Type*) noexcept                       \
    {                                                                                                                  \
        static_assert(std::is_empty_v<Type>, #Type " is declared as a tag but has data");                              \
        static const ::donut::engine::ComponentTypeDesc s_Desc = ::donut::engine::detail::MakeComponentTypeDesc<Type>(  \
            StrGuid##_nvrhi_guid, #Type, (Flags), 0, nullptr, 0);                                                      \
        return s_Desc;                                                                                                 \
    }

// Implementation of the field list. The EXPAND wrappers make MSVC's traditional preprocessor split __VA_ARGS__.
#define DONUT_ECS_DETAIL_EXPAND(x) x
#define DONUT_ECS_DETAIL_UNPAREN(...) __VA_ARGS__
#define DONUT_ECS_DETAIL_CAT(a, b) DONUT_ECS_DETAIL_CAT_I(a, b)
#define DONUT_ECS_DETAIL_CAT_I(a, b) a##b
#define DONUT_ECS_DETAIL_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
                               _20, _21, _22, _23, _24, N, ...) N
// 1..16, or TOO_MANY for 17..24 arguments.
#define DONUT_ECS_DETAIL_NARG(...) \
    DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_ARG_N(__VA_ARGS__, TOO_MANY, TOO_MANY, TOO_MANY, TOO_MANY, TOO_MANY,   \
        TOO_MANY, TOO_MANY, TOO_MANY, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
// 1 when the arguments contain a comma.
#define DONUT_ECS_DETAIL_HAS_COMMA(...) \
    DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_ARG_N(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        1, 1, 1, 1, 1, 1, 0))
#define DONUT_ECS_DETAIL_COMMA_IF_CALLED(...) ,
// Field names are identifiers, so the arguments are empty exactly when they contain no comma and the probe does.
#define DONUT_ECS_DETAIL_IS_EMPTY(...)                                                                                 \
    DONUT_ECS_DETAIL_CAT(DONUT_ECS_DETAIL_EMPTY_CASE_, DONUT_ECS_DETAIL_CAT(DONUT_ECS_DETAIL_HAS_COMMA(__VA_ARGS__), \
        DONUT_ECS_DETAIL_HAS_COMMA(DONUT_ECS_DETAIL_COMMA_IF_CALLED __VA_ARGS__ ())))
#define DONUT_ECS_DETAIL_EMPTY_CASE_00 0
#define DONUT_ECS_DETAIL_EMPTY_CASE_01 1
#define DONUT_ECS_DETAIL_EMPTY_CASE_10 0
#define DONUT_ECS_DETAIL_EMPTY_CASE_11 0
#define DONUT_ECS_DETAIL_FIELD_LIST(Type, ...) \
    DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_CAT(DONUT_ECS_DETAIL_FIELD_LIST_, DONUT_ECS_DETAIL_IS_EMPTY(__VA_ARGS__))(Type, __VA_ARGS__))
#define DONUT_ECS_DETAIL_FIELD_LIST_1(Type, ...)
#define DONUT_ECS_DETAIL_FIELD_LIST_0(Type, ...) \
    DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_CAT(DONUT_ECS_DETAIL_F, DONUT_ECS_DETAIL_NARG(__VA_ARGS__))(Type, __VA_ARGS__))
#define DONUT_ECS_DETAIL_FIELD(T, f) \
    ::donut::engine::detail::MakeFieldDesc<decltype(T::f)>(#f, offsetof(T, f)),
#define DONUT_ECS_DETAIL_FTOO_MANY(T, ...) ::donut::engine::detail::TooManyFields<T>(),
#define DONUT_ECS_DETAIL_F1(T, a) DONUT_ECS_DETAIL_FIELD(T, a)
#define DONUT_ECS_DETAIL_F2(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F1(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F3(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F2(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F4(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F3(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F5(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F4(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F6(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F5(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F7(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F6(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F8(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F7(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F9(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F8(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F10(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F9(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F11(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F10(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F12(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F11(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F13(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F12(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F14(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F13(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F15(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F14(T, __VA_ARGS__))
#define DONUT_ECS_DETAIL_F16(T, a, ...) DONUT_ECS_DETAIL_FIELD(T, a) DONUT_ECS_DETAIL_EXPAND(DONUT_ECS_DETAIL_F15(T, __VA_ARGS__))

#if defined(DONUT_ECS_TYPES_RESTORE_NONE)
#undef DONUT_ECS_TYPES_RESTORE_NONE
#pragma pop_macro("None")
#endif
