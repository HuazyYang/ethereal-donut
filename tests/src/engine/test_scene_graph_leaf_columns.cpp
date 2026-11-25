// Leaf columns with a home (LeafTypeDesc::leafRecord; aggregate docs/ecs-design.md 4.11, 5.3, 9.2; sub-plan item E6.3),
// and observers on leaf tags instead of RegisterLeaf overrides (5.5): the rule of Get and Edit records; the values
// through attach, detach, re-attach, SetLeaf, copies, refused attaches and teardown, with the reference counts of a
// hooked column; the rows complete when observers see them; records on worker threads; the E0 generator's edits through
// an override and through observers in one graph; ResourceTracker's order; zero allocations in steady state.
// Usage: test_scene_graph_leaf_columns [case]. `--child-mismatch` is the child process of the case `mismatch`.

#include "SceneGraphGenerator.h"

#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>
#include <donut/core/log.h>
#include <donut/core/query_cast.h>
#include <donut/tests/utils.h>

#include "EcsWorldImpl.h"
#include "SceneGraphStore.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <malloc.h>
#include <process.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test and donut_engine, which links statically). The world's own blocks and
// the leaf records come from nvrhi_core's allocator, which ecs::GetHeapAllocationCount counts.
namespace
{
    std::atomic<uint64_t> g_NewCount{ 0 };

    void* CountedAllocate(size_t size) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
        return std::malloc(size ? size : 1);
    }

    void* CountedAllocateAligned(size_t size, size_t alignment) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
        return _aligned_malloc(size ? size : 1, alignment);
#else
        return std::aligned_alloc(alignment, ((size ? size : 1) + alignment - 1) & ~(alignment - 1));
#endif
    }

    void CountedFreeAligned(void* memory) noexcept
    {
#if defined(_MSC_VER)
        _aligned_free(memory);
#else
        std::free(memory);
#endif
    }
}

void* operator new(size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new[](size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new(size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }
void operator delete(void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete[](void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete(void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }

// log::warning calls; main fails a case that leaves any (Debug worlds report validation errors through it). A case that
// provokes warnings on purpose checks them and takes them back.
std::atomic<uint32_t> g_Warnings{ 0 };

namespace
{
    // Errors seen where no exception may pass (observers, hooks, threads); main fails a case that leaves any.
    std::atomic<uint32_t> g_Errors{ 0 };
    void ReportError(const char* file, int line, const char* condition) noexcept
    {
        ++g_Errors;
        std::fprintf(stderr, "%s:%d: %s\n", file, line, condition);
    }

#define EXPECT(condition) \
    do { if (!(condition)) ReportError(__FILE__, __LINE__, #condition); } while (0)

    // ------------------------------------------------------------------------------------- columns

    // What a hooked column references.
    std::atomic<int32_t> g_LiveResources{ 0 };
    class Resource final : public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Resource)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        Resource() { ++g_LiveResources; }
        ~Resource() { --g_LiveResources; }
    };

    uint32_t References(nvrhi::IObject* object)
    {
        object->AddRef();
        return uint32_t(object->Release());
    }

    // A COM reference with hooks, as GvdbVolumeRef (E6.4). Counts the values alive, and the uses of dead ones.
    std::atomic<int32_t> g_LiveVolumeRefs{ 0 };
    struct VolumeRef
    {
        static constexpr uint32_t c_Alive = 0x414c4956u;
        Resource* object = nullptr;
        uint32_t state = c_Alive;

        VolumeRef() noexcept { ++g_LiveVolumeRefs; }
        explicit VolumeRef(Resource* resource) noexcept : object(resource)
        {
            ++g_LiveVolumeRefs;
            if (object)
                object->AddRef();
        }
        VolumeRef(const VolumeRef& other) noexcept : VolumeRef(other.object) { EXPECT(other.state == c_Alive); }
        VolumeRef(VolumeRef&& other) noexcept : object(other.object)
        {
            EXPECT(other.state == c_Alive);
            ++g_LiveVolumeRefs;
            other.object = nullptr;
        }
        VolumeRef& operator=(const VolumeRef& other) noexcept
        {
            EXPECT(state == c_Alive && other.state == c_Alive);
            if (other.object)
                other.object->AddRef();
            if (object)
                object->Release();
            object = other.object;
            return *this;
        }
        ~VolumeRef()
        {
            EXPECT(state == c_Alive);
            state = 0;
            --g_LiveVolumeRefs;
            if (object)
                object->Release();
        }
    };
    DONUT_ECS_COMPONENT(VolumeRef, "eaca865b-05e0-4144-9b48-644fd46db4b8", DONUT_ECS_FIELDS())

    // A default state that is not zero.
    struct Density
    {
        float density = 1.5f;
        uint32_t mode = 7;
    };
    DONUT_ECS_COMPONENT(Density, "4cf660a6-118c-488b-a941-cf74d3df9484", DONUT_ECS_FIELDS(density, mode))

    struct Extent
    {
        double values[3];
    };
    DONUT_ECS_COMPONENT(Extent, "4904ce54-88d5-4bde-9eaf-740f5436e948", DONUT_ECS_FIELDS(values))

    struct alignas(64) Wide
    {
        uint32_t words[16];
    };
    DONUT_ECS_COMPONENT(Wide, "84f337af-ef1e-47b1-a991-3349617ca050", DONUT_ECS_FIELDS(words))

    struct Block
    {
        uint8_t bytes[3000];
    };
    DONUT_ECS_COMPONENT(Block, "82e624ed-e40e-464a-9b64-8536807e1352", DONUT_ECS_FIELDS(bytes))

    // Two of them fit no chunk.
    struct Huge
    {
        uint8_t bytes[40000];
    };
    DONUT_ECS_COMPONENT(Huge, "20a4a49d-d8f7-4c6f-be88-9c2668a2c4ad", DONUT_ECS_FIELDS(bytes))
    struct Huge2
    {
        uint8_t bytes[40000];
    };
    DONUT_ECS_COMPONENT(Huge2, "2b2165a5-bb7a-4cda-8dd3-36c8e66595d2", DONUT_ECS_FIELDS(bytes))

    struct Sample
    {
        float value;
    };
    DONUT_ECS_BUFFER(Sample, "6e353168-b01a-491b-8558-79b88a268f1e", 2, DONUT_ECS_FIELDS(value))

    struct Switch
    {
        uint32_t value;
    };
    DONUT_ECS_COMPONENT_EX(Switch, "7f8b3eef-e171-47d6-acf2-d02c55e02103", ComponentFlags::Enableable, 0,
                           DONUT_ECS_FIELDS(value))

    struct VolumeLeafTag { };
    DONUT_ECS_TAG(VolumeLeafTag, "0377e3a0-6fa0-4546-b64b-2a32ed2ad15e")
    struct WideVolumeLeafTag { };
    DONUT_ECS_TAG(WideVolumeLeafTag, "9e32716d-d9a8-4ad6-bd28-0d4798cec096")
    struct OtherLeafTag { };
    DONUT_ECS_TAG(OtherLeafTag, "8b66a5a9-0c81-4474-bbd2-b949a058429e")
    struct TracedMeshLeafTag { };
    DONUT_ECS_TAG(TracedMeshLeafTag, "893bcc4f-ec0b-44a1-83cf-323155dd9098")
    struct MarkedLeafTag { };
    DONUT_ECS_TAG_EX(MarkedLeafTag, "0f998900-276d-4b18-8f7f-97f609fe0ab8", ComponentFlags::Sparse)

    // ------------------------------------------------------------------------------------- leaf classes

    template <typename... T>
    struct Types
    {
        static const ComponentTypeDesc* const* Get()
        {
            static const ComponentTypeDesc* const s_List[] = { &ComponentTraits<T>::Desc()... };
            return s_List;
        }
        static constexpr uint32_t count = uint32_t(sizeof...(T));
    };

    template <typename Leaf, typename Tags, typename Columns>
    const LeafTypeDesc& MakeDesc(bool leafRecord, bool dynamicBounds = false)
    {
        static const LeafTypeDesc s_Desc{ nvrhi::uuid_of<Leaf>(), Tags::Get(), Tags::count, Columns::Get(),
                                          Columns::count, dynamicBounds, leafRecord };
        return s_Desc;
    }

    // Get and Edit are protected.
    class ColumnLeaf : public SceneGraphLeaf
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        int id = -1;   // the test's, not a column

        template <typename T> const T* Read() const { return Get<T>(); }
        template <typename T> T* Write() { return Edit<T>(); }
    };

    class VolumeLeaf;
    NVRHI_CLASS_CLSID(VolumeLeaf, "4967b40b-3b73-4f12-ac6c-8bd28e932e7f")
    class VolumeLeaf : public ColumnLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(VolumeLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(VolumeLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(VolumeLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        bool throwBox = false;

        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<VolumeLeaf, Types<VolumeLeafTag>, Types<VolumeRef, Density, Extent>>(true);
        }
        // reads a column, as GVDB's and SPH's leaves do (E6.4)
        dm::box3 GetLocalBoundingBox() override
        {
            if (throwBox)
                throw std::runtime_error("box");
            const Density* density = Get<Density>();
            return density ? dm::box3(dm::float3(-density->density), dm::float3(density->density)) : dm::box3::empty();
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override
        {
            auto copy = MAKE_RC_OBJ_PTR(VolumeLeaf);
            CopyTo(copy);
            return copy;
        }
        void CopyTo(VolumeLeaf* copy) const
        {
            copy->id = id;
            if (const Density* density = Get<Density>())
            {
                *copy->Edit<Density>() = *density;
                *copy->Edit<VolumeRef>() = *Get<VolumeRef>();
                *copy->Edit<Extent>() = *Get<Extent>();
            }
        }
    };

    // A subclass with one more column, in another order.
    class WideVolumeLeaf;
    NVRHI_CLASS_CLSID(WideVolumeLeaf, "6f729df1-8914-40f3-9b36-a6f6473c0641")
    class WideVolumeLeaf final : public VolumeLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(WideVolumeLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(WideVolumeLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(WideVolumeLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(VolumeLeaf)
        NVRHI_END_INTERFACE_TABLE()

        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<WideVolumeLeaf, Types<VolumeLeafTag, WideVolumeLeafTag>,
                            Types<Wide, Extent, Density, VolumeRef>>(true);
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override
        {
            auto copy = MAKE_RC_OBJ_PTR(WideVolumeLeaf);
            CopyTo(copy);
            if (const Wide* wide = Get<Wide>())
                *copy->Write<Wide>() = *wide;
            return copy;
        }
    };

    // Another class that shares a column type, with a large one, and a Sparse tag.
    class OtherLeaf;
    NVRHI_CLASS_CLSID(OtherLeaf, "49ead7c4-d118-4ee3-9201-5077d2e1c216")
    class OtherLeaf final : public ColumnLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(OtherLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(OtherLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(OtherLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<OtherLeaf, Types<OtherLeafTag, MarkedLeafTag>, Types<Block, Density>>(true);
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override
        {
            auto copy = MAKE_RC_OBJ_PTR(OtherLeaf);
            copy->id = id;
            if (const Density* density = Get<Density>())
            {
                *copy->Edit<Density>() = *density;
                *copy->Edit<Block>() = *Get<Block>();
            }
            return copy;
        }
    };

    // Its row fits no chunk; the first column alone does.
    NVRHI_CLASS_CLSID(HugeLeaf, "c92fdbc9-a6b0-44a9-a5fc-2d69120e250d")
    class HugeLeaf final : public ColumnLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(HugeLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(HugeLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(HugeLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<HugeLeaf, Types<OtherLeafTag, MarkedLeafTag>, Types<Huge, Huge2>>(true);
        }
        dm::box3 GetLocalBoundingBox() override { return dm::box3(dm::float3(-77.f), dm::float3(77.f)); }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return MAKE_RC_OBJ_PTR(HugeLeaf); }
    };

    // Columns without a record (the E4 way), and a Sparse tag.
    class DerivedColumnLeaf;
    NVRHI_CLASS_CLSID(DerivedColumnLeaf, "ed23d126-0b44-4db3-9e1a-565559ab9b8c")
    class DerivedColumnLeaf final : public ColumnLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(DerivedColumnLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DerivedColumnLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(DerivedColumnLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<DerivedColumnLeaf, Types<MarkedLeafTag>, Types<Extent>>(false);
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return MAKE_RC_OBJ_PTR(DerivedColumnLeaf); }
    };

    // A class that edits in its constructor: allowed, no subclass adds columns. And columns that no record keeps.
    class EagerLeaf;
    NVRHI_CLASS_CLSID(EagerLeaf, "3d08907e-cb01-41d2-adf7-bda54f03ec1a")
    class EagerLeaf final : public ColumnLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(EagerLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(EagerLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(EagerLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        explicit EagerLeaf(float density) { Edit<Density>()->density = density; }
        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<EagerLeaf, Types<OtherLeafTag>, Types<Sample, Density, Switch>>(true);
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return MAKE_RC_OBJ_PTR(EagerLeaf, Get<Density>()->density); }
    };

    // The misuse: a base class edits in its constructor and a subclass adds a column, before the base's in its list.
    class EagerBaseLeaf;
    NVRHI_CLASS_CLSID(EagerBaseLeaf, "ee564e8b-b902-4f84-9756-97b4fcc38730")
    class EagerBaseLeaf : public ColumnLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(EagerBaseLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(EagerBaseLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(EagerBaseLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        explicit EagerBaseLeaf(Resource* resource)
        {
            Edit<Density>()->density = 4.f;
            *Edit<VolumeRef>() = VolumeRef(resource);
        }
        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<EagerBaseLeaf, Types<OtherLeafTag>, Types<Density, VolumeRef>>(true);
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return nullptr; }
    };
    class EagerSubLeaf;
    NVRHI_CLASS_CLSID(EagerSubLeaf, "c94757f2-4116-4bb9-a2df-2f1c298dcdbe")
    class EagerSubLeaf final : public EagerBaseLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(EagerSubLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(EagerSubLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(EagerSubLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(EagerBaseLeaf)
        NVRHI_END_INTERFACE_TABLE()

        using EagerBaseLeaf::EagerBaseLeaf;
        const LeafTypeDesc& GetTypeDesc() const override
        {
            // without VolumeRef: the record has a column too many, and Extent is missing in it
            return MakeDesc<EagerSubLeaf, Types<OtherLeafTag>, Types<Extent, Density>>(true);
        }
    };

    // A box that follows a column by itself (dynamicBounds): the Bounds system calls Get on job threads.
    class DynamicLeaf;
    NVRHI_CLASS_CLSID(DynamicLeaf, "00bf108a-a197-4750-9e7f-c27dac697317")
    class DynamicLeaf final : public ColumnLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(DynamicLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DynamicLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(DynamicLeaf)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<DynamicLeaf, Types<OtherLeafTag>, Types<Density>>(true, true);
        }
        dm::box3 GetLocalBoundingBox() override
        {
            const float size = Get<Density>()->density;
            return dm::box3(dm::float3(-size), dm::float3(size));
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override { return MAKE_RC_OBJ_PTR(DynamicLeaf); }
    };

    // A mesh instance whose class keeps a column of its own in a record, beside the derived E4 columns.
    class TracedMesh;
    NVRHI_CLASS_CLSID(TracedMesh, "7576160d-2585-4f24-ae3a-4407db149a42")
    class TracedMesh final : public MeshInstance
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(TracedMesh)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(TracedMesh)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(TracedMesh)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(MeshInstance)
        NVRHI_END_INTERFACE_TABLE()

        float value = 0.f;   // what the column must hold

        TracedMesh(MeshInfo* mesh, float v) : MeshInstance(mesh), value(v) { Edit<Density>()->density = v; }
        const LeafTypeDesc& GetTypeDesc() const override
        {
            return MakeDesc<TracedMesh, Types<MeshInstanceLeaf, TracedMeshLeafTag>,
                            Types<MeshRef, RenderFlags, GpuInstanceSlot, Density>>(true);
        }
        nvrhi::AutoPtr<SceneGraphLeaf> Clone() override
        {
            return MAKE_RC_OBJ_PTR(TracedMesh, m_Mesh.Get(), Get<Density>()->density);
        }
        bool HoldsValue() const { const Density* density = Get<Density>(); return density && density->density == value; }
    };

    // ------------------------------------------------------------------------------------- helpers

    nvrhi::AutoPtr<SceneGraphNode> MakeNode(SceneGraphLeaf* leaf = nullptr)
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        if (leaf)
            node->SetLeaf(leaf);
        return node;
    }

    nvrhi::AutoPtr<SceneGraph> MakeGraph()
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MakeNode());
        return graph;
    }

    nvrhi::AutoPtr<VolumeLeaf> MakeVolume(Resource* resource, float density, int id = -1)
    {
        auto leaf = MAKE_RC_OBJ_PTR(VolumeLeaf);
        leaf->id = id;
        leaf->Write<Density>()->density = density;
        *leaf->Write<VolumeRef>() = VolumeRef(resource);
        leaf->Write<Extent>()->values[2] = double(density) * 2.0;
        return leaf;
    }

    bool HasValues(const VolumeLeaf* leaf, const Resource* resource, float density)
    {
        const Density* d = leaf->Read<Density>();
        const VolumeRef* r = leaf->Read<VolumeRef>();
        const Extent* e = leaf->Read<Extent>();
        return d && r && e && d->density == density && r->object == resource && r->state == VolumeRef::c_Alive &&
            e->values[2] == double(density) * 2.0;
    }

    // The value is in the node's row.
    template <typename T>
    const T* RowOf(SceneGraph* graph, const SceneGraphNode* node)
    {
        return GetComponent<T>(graph->GetWorld(), graph->GetEntity(node));
    }

    // The change version of a column in the chunk of a node's row.
    template <typename T>
    uint64_t VersionOf(SceneGraph* graph, const SceneGraphNode* node)
    {
        ecs::WorldImpl* world = ecs::GetWorldImpl(graph->GetWorld());
        const ecs::EntityRecord* record = world->FindAlive(graph->GetEntity(node));
        CHECK(record);
        const ecs::Archetype& archetype = world->GetArchetypeStorage(record->archetype);
        const uint32_t column = archetype.FindColumn(FindComponent<T>(world));
        CHECK(column != ecs::c_NoColumn);
        return archetype.Versions(record->chunk)[column];
    }

    class Observer final : public nvrhi::ObjectImpl<IObserver>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Observer)
        NVRHI_IMPLEMENTS_INTERFACE(IObserver)
        NVRHI_END_INTERFACE_TABLE()

        using Callback = std::function<nvrhi::FRESULT(ObserverEvent, const EntityBatch&)>;
        explicit Observer(Callback callback) : m_Callback(std::move(callback)) { }
        nvrhi::FRESULT OnEntities(ObserverEvent event, const EntityBatch& batch, ICommandBuffer*) noexcept override
        {
            return m_Callback(event, batch);
        }

    private:
        Callback m_Callback;
    };

    // The observers of the design's replacement for an override: a class tag, LeafObject as a Read filter term.
    template <typename Tag>
    void ObserveTag(SceneGraph* graph, std::vector<ObserverId>& ids, Observer::Callback callback,
                    std::initializer_list<ComponentId> moreColumns = {})
    {
        IWorld* world = graph->GetWorld();
        std::vector<QueryTerm> terms{ QueryTerm{ RegisterComponent<LeafObject>(world), TermOp::All, Access::Read, {} } };
        for (ComponentId id : moreColumns)
            terms.push_back(QueryTerm{ id, TermOp::All, Access::Read, {} });
        auto observer = MAKE_RC_OBJ_PTR(Observer, std::move(callback));
        ObserverDesc desc{};
        desc.debugName = "test";
        desc.filterTerms = terms.data();
        desc.filterTermCount = uint32_t(terms.size());
        desc.component = RegisterComponent<Tag>(world);
        desc.access = Access::Filter;
        for (ObserverEvent event : { ObserverEvent::OnAdd, ObserverEvent::OnRemove })
        {
            desc.event = event;
            ObserverId id = ObserverId::Invalid;
            CHECK(NVRHI_SUCCEEDED(world->AddObserver(desc, observer, id)));
            ids.push_back(id);
        }
    }

    SceneGraphLeaf* LeafAt(const EntityBatch& batch, uint32_t row)
    {
        return static_cast<const LeafObject*>(batch.columns[1])[row].object;
    }

    // Everything the cases make is gone again: called by main after each case.
    void CheckNothingLeft(const char* name)
    {
        if (LeafRecord::GetLiveCount() != 0 || g_LiveVolumeRefs != 0 || g_LiveResources != 0)
        {
            throw std::runtime_error(std::string(name) + ": left over: " + std::to_string(LeafRecord::GetLiveCount()) +
                                     " records, " + std::to_string(g_LiveVolumeRefs.load()) + " references, " +
                                     std::to_string(g_LiveResources.load()) + " resources\n");
        }
    }
}

// ------------------------------------------------------------------------------------------ the world's part

namespace
{
    struct Counted
    {
        uint32_t added = 0, removing = 0;
        Entity removed = Entity::Null;
        uint32_t observersAtAdd = 0, observersAtRemove = 0;   // observer calls before the hook
        uint32_t observerCalls = 0;
    };
}

// WorldImpl::ChangeComponentsInternal and the row hooks: one report each way, the order around the observers, and what
// the call leaves alone.
void test_change_components()
{
    WorldDesc desc;
    desc.validation = true;
    nvrhi::AutoPtr<IWorld> world;
    CHECK(NVRHI_SUCCEEDED(CreateWorld(desc, &world)));
    ecs::WorldImpl* impl = ecs::GetWorldImpl(world);
    const ComponentId volume = RegisterComponent<VolumeRef>(world);
    const ComponentId density = RegisterComponent<Density>(world);
    const ComponentId extent = RegisterComponent<Extent>(world);
    const ComponentId tag = RegisterComponent<VolumeLeafTag>(world);
    const ComponentId sparse = RegisterComponent<MarkedLeafTag>(world);
    const ComponentId shared = RegisterComponent<HierarchyDepth>(world);

    Counted counted;
    const ecs::WorldImpl::RowHooks hooks{ &counted,
        [](void* context, uint32_t count) noexcept
        {
            Counted& c = *static_cast<Counted*>(context);
            c.added += count;
            c.observersAtAdd = c.observerCalls;
        },
        [](void* context, Entity entity, const ecs::Archetype& archetype, uint32_t chunk, uint32_t row) noexcept
        {
            Counted& c = *static_cast<Counted*>(context);
            ++c.removing;
            c.removed = entity;
            c.observersAtRemove = c.observerCalls;
            EXPECT(archetype.Entities(chunk)[row] == entity);
        } };
    auto observer = MAKE_RC_OBJ_PTR(Observer, [&](ObserverEvent, const EntityBatch& batch)
    {
        counted.observerCalls += batch.count;
        return nvrhi::FS_OK;
    });
    for (ComponentId id : { volume, density })
    {
        for (ObserverEvent event : { ObserverEvent::OnAdd, ObserverEvent::OnRemove })
        {
            ObserverDesc observed{};
            observed.component = id;
            observed.access = Access::Read;
            observed.event = event;
            ObserverId observerId = ObserverId::Invalid;
            CHECK(NVRHI_SUCCEEDED(world->AddObserver(observed, observer, observerId)));
        }
    }

    // creation: the hook once, before the observers
    Entity entities[3];
    const ComponentId components[] = { volume, extent };
    EntityCreateDesc create{};
    create.components = components;
    create.componentCount = 2;
    create.entityCount = 3;
    CHECK(NVRHI_SUCCEEDED(impl->CreateEntitiesInternal(create, entities, impl->GetStampVersion(), &hooks)));
    CHECK(NVRHI_SUCCEEDED(impl->FlushObservers()));
    CHECK(counted.added == 3 && counted.observersAtAdd == 0 && counted.observerCalls == 3 && counted.removing == 0);

    // a change: volume goes and density comes with a value; extent is there already and gets its value; what the entity
    // lacks is skipped
    GetComponentForWrite<Extent>(world, entities[1])->values[0] = 4.0;
    const Density value{ 9.f, 3 };
    const Extent other{ { 1.0, 1.0, 1.0 } };
    const ComponentId removed[] = { volume, tag };
    const ecs::WorldImpl::NewValue added[] = { { extent, &other, ValueTransfer::Copy },
                                               { density, &value, ValueTransfer::Copy } };
    ecs::WorldImpl::ComponentChange change;
    change.removed = removed;
    change.removedCount = 2;
    change.added = added;
    change.addedCount = 2;
    const uint32_t source = impl->FindAlive(entities[1])->archetype;
    const uint32_t archetypes = world->GetArchetypeCount();
    const Entity first = impl->GetArchetypeStorage(source).Entities(0)[0];

    // asked for first: the target, no row changed, no archetype between; asked for again: from the cache
    CHECK(NVRHI_SUCCEEDED(impl->PrepareChange(source, change)));
    CHECK(change.source == source && change.target != source && change.addsPresent);
    CHECK(world->GetArchetypeCount() == archetypes + 1);
    CHECK(impl->FindAlive(entities[1])->archetype == source);
    ecs::WorldImpl::ComponentChange again = change;
    again.source = again.target = ecs::c_NoArchetype;
    const uint64_t lookups = impl->GetCounters().archetypeLookups;
    const uint64_t blocks = ecs::GetHeapAllocationCount();
    CHECK(NVRHI_SUCCEEDED(impl->PrepareChange(source, again)) && again.target == change.target);
    CHECK(impl->GetCounters().archetypeLookups == lookups && ecs::GetHeapAllocationCount() == blocks);
    // other lists are other transitions
    ecs::WorldImpl::ComponentChange less;
    less.removed = removed;
    less.removedCount = 1;
    CHECK(NVRHI_SUCCEEDED(impl->PrepareChange(source, less)) && less.target != change.target && !less.addsPresent);
    less.removed = removed + 1;   // the tag, which it lacks
    CHECK(NVRHI_SUCCEEDED(impl->PrepareChange(source, less)) && less.target == source);

    world->AdvanceVersion();
    counted = Counted{};
    const auto before = impl->GetCounters();
    CHECK(NVRHI_SUCCEEDED(impl->ChangeComponentsInternal(entities[1], change, impl->GetStampVersion(), &hooks)));
    CHECK(NVRHI_SUCCEEDED(impl->FlushObservers()));
    CHECK(impl->GetCounters().rowsAddedReports == before.rowsAddedReports + 1);
    CHECK(impl->GetCounters().rowsRemovingReports == before.rowsRemovingReports + 1);
    CHECK(counted.removing == 1 && counted.removed == entities[1] && counted.observersAtRemove == 1);
    CHECK(counted.added == 1 && counted.observersAtAdd == 1 && counted.observerCalls == 2);
    CHECK(!world->HasComponent(entities[1], volume) && GetComponent<Density>(world, entities[1])->density == 9.f);
    CHECK(GetComponent<Extent>(world, entities[1])->values[0] == 1.0 && g_LiveVolumeRefs == 2);
    CHECK(first == entities[0] && GetComponent<Extent>(world, entities[2])->values[0] == 0.0);

    // no move: the values of what the entity has are written and stamped; a null value leaves them. `change` was prepared
    // for another archetype: it is prepared again.
    world->AdvanceVersion();
    const Density newer{ 11.f, 4 };
    const ecs::WorldImpl::NewValue present[] = { { extent, nullptr, ValueTransfer::Copy },
                                                 { density, &newer, ValueTransfer::Copy } };
    change.removed = nullptr;
    change.removedCount = 0;
    change.added = present;
    counted = Counted{};
    CHECK(NVRHI_SUCCEEDED(impl->ChangeComponentsInternal(entities[1], change, impl->GetStampVersion(), &hooks)));
    CHECK(impl->GetCounters().rowsAddedReports == before.rowsAddedReports + 1);
    CHECK(counted.added == 0 && counted.removing == 0);
    CHECK(GetComponent<Density>(world, entities[1])->density == 11.f);
    CHECK(GetComponent<Extent>(world, entities[1])->values[0] == 1.0);
    {
        const ecs::EntityRecord* record = impl->FindAlive(entities[1]);
        const ecs::Archetype& archetype = impl->GetArchetypeStorage(record->archetype);
        CHECK(archetype.Versions(record->chunk)[archetype.FindColumn(density)] == world->GetVersion());
        CHECK(archetype.Versions(record->chunk)[archetype.FindColumn(extent)] < world->GetVersion());
    }

    // refused, with nothing changed: Sparse and Shared components, a row that does not fit, an entity that is not alive
    const uint64_t structural = world->GetStructuralVersion();
    for (ComponentId bad : { sparse, shared, ComponentId::Invalid })
    {
        const ecs::WorldImpl::NewValue badValue[] = { { bad, nullptr, ValueTransfer::Copy } };
        ecs::WorldImpl::ComponentChange leaving, coming;
        leaving.removed = &bad;
        leaving.removedCount = 1;
        coming.added = badValue;
        coming.addedCount = 1;
        CHECK(impl->ChangeComponentsInternal(entities[1], leaving, 1, &hooks) == nvrhi::FE_INVALID_ARGS);
        CHECK(impl->ChangeComponentsInternal(entities[1], coming, 1, &hooks) == nvrhi::FE_INVALID_ARGS);
    }
    const ecs::WorldImpl::NewValue huge[] = { { RegisterComponent<Huge>(world), nullptr, ValueTransfer::Copy },
                                              { RegisterComponent<Huge2>(world), nullptr, ValueTransfer::Copy } };
    ecs::WorldImpl::ComponentChange tooLarge;
    tooLarge.added = huge;
    tooLarge.addedCount = 2;
    CHECK(impl->ChangeComponentsInternal(entities[1], tooLarge, 1, &hooks) == nvrhi::FE_INVALID_ARGS);
    CHECK(impl->ChangeComponentsInternal(Entity::Null, change, 1, &hooks) == nvrhi::FE_INVALID_ARGS);
    CHECK(world->GetStructuralVersion() == structural && counted.added == 0 && counted.removing == 0);
    CHECK(g_Warnings == 8);   // validation
    g_Warnings = 0;

    // destruction: the hook per entity, after its observers
    counted = Counted{};
    CHECK(NVRHI_SUCCEEDED(impl->DestroyEntitiesInternal(entities, 3, impl->GetStampVersion(), &hooks)));
    CHECK(NVRHI_SUCCEEDED(impl->FlushObservers()));
    CHECK(counted.removing == 3 && counted.removed == entities[2] && counted.observersAtRemove == 3);
    CHECK(g_LiveVolumeRefs == 0);
}

// ------------------------------------------------------------------------------------------ the record's rule

void test_record_rule()
{
    // out of memory: this class's size class has no pooled block yet (the first case that uses it)
    {
        auto leaf = MAKE_RC_OBJ_PTR(OtherLeaf);
        ecs::SetHeapFailureCountdown(0);
        CHECK(leaf->Write<Density>() == nullptr);
        ecs::SetHeapFailureCountdown(-1);
        CHECK(LeafRecord::GetLiveCount() == 0);
        CHECK(leaf->Write<Density>() != nullptr);
        CHECK(LeafRecord::GetLiveCount() == 1);
    }
    CHECK(LeafRecord::GetLiveCount() == 0);

    auto leaf = MAKE_RC_OBJ_PTR(VolumeLeaf);
    // Get makes no record, and neither does an Edit of a type that is not one of the columns
    CHECK(!leaf->Read<Density>() && !leaf->Read<VolumeRef>());
    CHECK(!leaf->Write<Block>() && !leaf->Write<WorldTransform>() && !leaf->Write<VolumeLeafTag>());
    CHECK(LeafRecord::GetLiveCount() == 0);
    // the first Edit makes it, with every column in its default value
    Density* density = leaf->Write<Density>();
    CHECK(density && density->density == 1.5f && density->mode == 7);
    CHECK(LeafRecord::GetLiveCount() == 1);
    CHECK(leaf->Read<VolumeRef>() && !leaf->Read<VolumeRef>()->object);
    CHECK(leaf->Read<Extent>()->values[0] == 0.0 && leaf->Read<Extent>()->values[2] == 0.0);
    CHECK(!leaf->Read<Block>() && !leaf->Read<VolumeLeafTag>() && !leaf->Read<WorldTransform>());
    density->density = 3.f;
    CHECK(leaf->Write<Density>() == density && leaf->Read<Density>() == density && density->density == 3.f);
    // the columns do not overlap: their addresses, their sizes
    leaf->Write<Extent>()->values[0] = 1.0;
    leaf->Write<Extent>()->values[2] = 1.0;
    CHECK(leaf->Read<Density>()->density == 3.f && leaf->Read<VolumeRef>()->state == VolumeRef::c_Alive);

    // a class without leafRecord has no record
    auto derived = MAKE_RC_OBJ_PTR(DerivedColumnLeaf);
    CHECK(!derived->Write<Extent>() && !derived->Read<Extent>());
    CHECK(LeafRecord::GetLiveCount() == 1);

    // a subclass: its own columns, aligned as declared
    auto wide = MAKE_RC_OBJ_PTR(WideVolumeLeaf);
    wide->Write<Wide>()->words[15] = 0x55u;
    CHECK(reinterpret_cast<uintptr_t>(wide->Read<Wide>()) % 64 == 0);
    CHECK(wide->Read<Density>()->density == 1.5f && wide->Read<Wide>()->words[15] == 0x55u);
    CHECK(wide->Read<Wide>()->words[0] == 0 && !wide->Read<VolumeRef>()->object);

    // the constructor of a class without subclasses; a Buffer column is in no record and no value of Get
    auto eager = MAKE_RC_OBJ_PTR(EagerLeaf, 2.5f);
    CHECK(eager->Read<Density>()->density == 2.5f && eager->Read<Switch>() && !eager->Read<Sample>());
    CHECK(!eager->Write<Sample>());
    CHECK(LeafRecord::GetLiveCount() == 3);
    auto graph = MakeGraph();
    auto node = graph->AttachLeafNode(graph->GetRootNode(), eager);
    CHECK(eager->Read<Density>()->density == 2.5f && !eager->Read<Sample>() && !eager->Write<Sample>());
    CHECK(RowOf<Sample>(graph, node) != nullptr);   // the row has the buffer column all the same
    // (the enabled bit of an Enableable column is not kept: SceneComponents.h)
    eager->Write<Switch>()->value = 9;
    CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->SetEnabled(graph->GetEntity(node), FindComponent<Switch>(graph->GetWorld()),
                                                        false)));
    graph->Detach(node);
    CHECK(eager->Read<Switch>()->value == 9);
    graph->Attach(graph->GetRootNode(), node);
    CHECK(eager->Read<Switch>()->value == 9);
    CHECK(graph->GetWorld()->IsEnabled(graph->GetEntity(node), FindComponent<Switch>(graph->GetWorld())));
}

// ------------------------------------------------------------------------------------------ attach, detach

void test_attach_detach()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    auto leaf = MakeVolume(resource, 2.f);
    CHECK(References(resource) == 2);   // ours and the record's
    auto graph = MakeGraph();
    auto node = MakeNode(leaf);
    graph->Attach(graph->GetRootNode(), node);

    // the row is the home, the record is gone
    CHECK(LeafRecord::GetLiveCount() == 0);
    CHECK(g_LiveVolumeRefs == 1 && References(resource) == 2);
    CHECK(leaf->Read<Density>() == RowOf<Density>(graph, node));
    CHECK(leaf->Read<VolumeRef>() == RowOf<VolumeRef>(graph, node) && leaf->Read<Extent>() == RowOf<Extent>(graph, node));
    CHECK(HasValues(leaf, resource, 2.f));
    CHECK(leaf->GetLocalBoundingBox().m_maxs.x == 2.f);
    CHECK(RowOf<LocalBounds>(graph, node)->bounds.m_maxs.x == 2.f);   // taken while the record was the home
    CHECK(!leaf->Read<Block>() && !leaf->Read<WorldTransform>() && !leaf->Write<LocalBounds>());

    // Edit stamps its column with the world's version, and only its column; Get stamps nothing
    IWorld* world = graph->GetWorld();
    world->AdvanceVersion();
    const uint64_t before = VersionOf<Density>(graph, node);
    const uint64_t extentBefore = VersionOf<Extent>(graph, node);
    CHECK(before < world->GetVersion());
    (void)leaf->Read<Density>();
    CHECK(VersionOf<Density>(graph, node) == before);
    leaf->Write<Density>()->mode = 12;
    CHECK(VersionOf<Density>(graph, node) == world->GetVersion());
    CHECK(VersionOf<Extent>(graph, node) == extentBefore);
    CHECK(LeafRecord::GetLiveCount() == 0);

    // detached: the record is the home
    auto detached = graph->Detach(node);
    CHECK(LeafRecord::GetLiveCount() == 1 && g_LiveVolumeRefs == 1 && References(resource) == 2);
    CHECK(HasValues(leaf, resource, 2.f) && leaf->Read<Density>()->mode == 12);

    // edited while in no graph, attached to another graph
    leaf->Write<Density>()->density = 5.f;
    leaf->Write<Extent>()->values[2] = 10.0;
    auto other = MakeGraph();
    other->Attach(other->GetRootNode(), node);
    CHECK(LeafRecord::GetLiveCount() == 0 && HasValues(leaf, resource, 5.f) && leaf->Read<Density>()->mode == 12);
    CHECK(leaf->Read<Density>() == RowOf<Density>(other, node));

    // the neighbours of a row that goes keep theirs (swap-remove)
    auto second = MAKE_RC_OBJ_PTR(Resource);
    auto third = MAKE_RC_OBJ_PTR(Resource);
    auto leaf2 = MakeVolume(second, 6.f);
    auto leaf3 = MakeVolume(third, 7.f);
    auto node2 = other->AttachLeafNode(other->GetRootNode(), leaf2);
    auto node3 = other->AttachLeafNode(other->GetRootNode(), leaf3);
    other->Detach(node);
    CHECK(HasValues(leaf, resource, 5.f) && HasValues(leaf2, second, 6.f) && HasValues(leaf3, third, 7.f));
    other->Detach(node3);
    CHECK(HasValues(leaf3, third, 7.f) && HasValues(leaf2, second, 6.f));
    CHECK(References(resource) == 2 && References(second) == 2 && References(third) == 2);

    // a leaf without a record gets the defaults, and a record with them when it leaves
    auto bare = MAKE_RC_OBJ_PTR(VolumeLeaf);
    auto bareNode = graph->AttachLeafNode(graph->GetRootNode(), bare);
    CHECK(bare->Read<Density>()->density == 1.5f && bare->Read<Density>()->mode == 7 && !bare->Read<VolumeRef>()->object);
    CHECK(bare->Read<Extent>()->values[1] == 0.0);
    const int64_t records = LeafRecord::GetLiveCount();
    graph->Detach(bareNode);
    CHECK(LeafRecord::GetLiveCount() == records + 1);
    CHECK(bare->Read<Density>()->mode == 7);

    // a leaf that leaves with nobody holding it gives everything back
    leaf2 = nullptr;
    CHECK(References(second) == 2);   // ours and the row's
    other->Detach(node2);
    CHECK(References(second) == 2);   // the node holds the leaf
    node2 = nullptr;
    CHECK(References(second) == 1);

    // a subtree: an orphan whose root has no leaf, moved as one
    auto top = MakeNode();
    std::vector<nvrhi::AutoPtr<VolumeLeaf>> leaves;
    for (int i = 0; i < 5; ++i)
    {
        leaves.push_back(MakeVolume(third, float(100 + i)));
        graph->Attach(top, MakeNode(leaves.back()));
    }
    graph->Attach(graph->GetRootNode(), top);
    for (int i = 0; i < 5; ++i)
        CHECK(HasValues(leaves[size_t(i)], third, float(100 + i)));
    CHECK(References(third) == 7);   // ours, leaf3's record, the five rows'
    graph->Detach(top);
    for (int i = 0; i < 5; ++i)
        CHECK(HasValues(leaves[size_t(i)], third, float(100 + i)));
    CHECK(References(third) == 7);
}

// ------------------------------------------------------------------------------------------ SetLeaf

void test_set_leaf()
{
    auto a = MAKE_RC_OBJ_PTR(Resource);
    auto b = MAKE_RC_OBJ_PTR(Resource);
    auto graph = MakeGraph();
    auto node = MakeNode();
    graph->Attach(graph->GetRootNode(), node);

    // from none
    auto leafA = MakeVolume(a, 1.f);
    node->SetLeaf(leafA);
    CHECK(LeafRecord::GetLiveCount() == 0 && HasValues(leafA, a, 1.f));
    CHECK(leafA->Read<Density>() == RowOf<Density>(graph, node) && References(a) == 2);
    CHECK(RowOf<LocalBounds>(graph, node)->bounds.m_maxs.x == 1.f);
    graph->Refresh(0);
    CHECK(RowOf<WorldBounds>(graph, node)->bounds.m_maxs.x == 1.f);

    // the same leaf: it leaves and comes back, with its values
    node->SetLeaf(leafA);
    CHECK(LeafRecord::GetLiveCount() == 0 && HasValues(leafA, a, 1.f) && References(a) == 2);
    CHECK(leafA->GetNode() == node.Get() && node->GetLeaf() == leafA.Get());
    CHECK(leafA->Read<Density>() == RowOf<Density>(graph, node));

    // another leaf of the class
    auto leafB = MakeVolume(b, 2.f);
    node->SetLeaf(leafB);
    CHECK(LeafRecord::GetLiveCount() == 1);
    CHECK(HasValues(leafA, a, 1.f) && HasValues(leafB, b, 2.f));
    CHECK(RowOf<Density>(graph, node)->density == 2.f && References(a) == 2 && References(b) == 2);
    CHECK(!leafA->GetNode() && RowOf<LocalBounds>(graph, node)->bounds.m_maxs.x == 2.f);

    // a subclass: the shared columns must come from its own record, by GUID
    auto wide = MAKE_RC_OBJ_PTR(WideVolumeLeaf);
    wide->Write<Density>()->density = 3.f;
    wide->Write<Wide>()->words[3] = 33;
    *wide->Write<VolumeRef>() = VolumeRef(a);
    wide->Write<Extent>()->values[2] = 6.0;
    node->SetLeaf(wide);
    CHECK(LeafRecord::GetLiveCount() == 2);
    CHECK(HasValues(wide, a, 3.f) && wide->Read<Wide>()->words[3] == 33 && HasValues(leafB, b, 2.f));
    CHECK(wide->Read<Wide>() == RowOf<Wide>(graph, node) && References(a) == 3);

    // another class that shares a column type
    auto other = MAKE_RC_OBJ_PTR(OtherLeaf);
    other->Write<Density>()->density = 4.f;
    other->Write<Block>()->bytes[2999] = 44;
    node->SetLeaf(other);
    CHECK(HasValues(wide, a, 3.f) && wide->Read<Wide>()->words[3] == 33);
    CHECK(other->Read<Density>()->density == 4.f && other->Read<Block>()->bytes[2999] == 44);
    CHECK(other->Read<Density>() == RowOf<Density>(graph, node) && !RowOf<VolumeRef>(graph, node));
    CHECK(!RowOf<Extent>(graph, node) && !RowOf<Wide>(graph, node));

    // a class without columns, one with columns and no record (and a Sparse tag), a mesh instance, and back
    node->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
    CHECK(other->Read<Density>()->density == 4.f && !RowOf<Density>(graph, node));
    auto derived = MAKE_RC_OBJ_PTR(DerivedColumnLeaf);
    node->SetLeaf(derived);
    CHECK(derived->Read<Extent>() && derived->Read<Extent>() == RowOf<Extent>(graph, node));
    CHECK(graph->GetWorld()->HasComponent(graph->GetEntity(node), FindComponent<MarkedLeafTag>(graph->GetWorld())));
    derived->Write<Extent>()->values[0] = 5.0;   // a column without a record is the row's, and dies with it
    auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
    node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get()));
    CHECK(!node->GetLeaf()->GetTypeDesc().leafRecord);   // E4's columns stay derived
    CHECK(!graph->GetWorld()->HasComponent(graph->GetEntity(node), FindComponent<MarkedLeafTag>(graph->GetWorld())));
    CHECK(!derived->Read<Extent>() && !derived->Write<Extent>() && graph->GetMeshInstances().size() == 1);
    node->SetLeaf(leafA);
    CHECK(graph->GetMeshInstances().empty() && HasValues(leafA, a, 1.f));
    CHECK(leafA->Read<Density>() == RowOf<Density>(graph, node));

    // no leaf: the row has no column left, the leaf its values
    node->SetLeaf(nullptr);
    CHECK(HasValues(leafA, a, 1.f) && !RowOf<Density>(graph, node) && !RowOf<VolumeRef>(graph, node));
    CHECK(!RowOf<LeafObject>(graph, node) && !RowOf<LocalBounds>(graph, node));
    CHECK(References(a) == 3 && References(b) == 2);

    // the last reference to a leaf goes with SetLeaf
    node->SetLeaf(MakeVolume(b, 8.f));
    CHECK(References(b) == 3);
    node->SetLeaf(nullptr);
    CHECK(References(b) == 2);

    // a node in no graph: the record stays the leaf's
    auto orphan = MakeNode(leafB);
    CHECK(HasValues(leafB, b, 2.f));
    orphan->SetLeaf(leafA);
    CHECK(HasValues(leafB, b, 2.f) && HasValues(leafA, a, 1.f));
    graph->Attach(graph->GetRootNode(), orphan);
    CHECK(HasValues(leafA, a, 1.f) && leafA->Read<Density>() == RowOf<Density>(graph, orphan));

    // moving a leaf from node to node
    orphan->SetLeaf(nullptr);
    node->SetLeaf(leafA);
    CHECK(HasValues(leafA, a, 1.f) && leafA->Read<Density>() == RowOf<Density>(graph, node));

    // misuse: a leaf with a record that is on another node is refused, in a graph and out of it; nothing changes
    auto otherOrphan = MakeNode(leafB);
    for (SceneGraphNode* wrong : { orphan.Get(), otherOrphan.Get() })
    {
        SceneGraphLeaf* before = wrong->GetLeaf();
        wrong->SetLeaf(leafA);
        CHECK(g_Warnings == 1 && wrong->GetLeaf() == before);
        g_Warnings = 0;
        CHECK(leafA->GetNode() == node.Get() && node->GetLeaf() == leafA.Get() && HasValues(leafA, a, 1.f));
        CHECK(leafA->Read<Density>() == RowOf<Density>(graph, node));
    }
    otherOrphan->SetLeaf(nullptr);
    CHECK(!leafB->GetNode() && HasValues(leafB, b, 2.f));
    // a class without a record is taken, as before (importers put one light on several nodes)
    auto light = MAKE_RC_OBJ_PTR(PointLight);
    orphan->SetLeaf(light);
    otherOrphan->SetLeaf(light);
    CHECK(orphan->GetLeaf() == light.Get() && otherOrphan->GetLeaf() == light.Get());
    CHECK(light->GetNode() == otherOrphan.Get());
    orphan->SetLeaf(nullptr);

    // misuse: LeafObject overwritten through the world: the row is the other one's, not the first leaf's
    auto graphNode = graph->AttachLeafNode(graph->GetRootNode(), MakeVolume(b, 12.f));
    nvrhi::AutoPtr<VolumeLeaf> first = static_cast<VolumeLeaf*>(graphNode->GetLeaf());
    auto intruder = MAKE_RC_OBJ_PTR(VolumeLeaf);
    LeafObject value(intruder);
    intruder->AddRef();   // the value holds one
    CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->AddComponent(graph->GetEntity(graphNode),
                                                          FindComponent<LeafObject>(graph->GetWorld()), &value)));
    CHECK(!first->Read<Density>() && !first->Write<Density>() && !intruder->Read<Density>());
    graph->Detach(graphNode);
    CHECK(!intruder->Read<Density>() && References(b) == 2);   // the row took the value along

    // a row that has the components already, through the world: the values are written, as AddComponent does
    auto early = MakeNode();
    graph->Attach(graph->GetRootNode(), early);
    const LocalBounds wrongBox{ dm::box3(dm::float3(-50.f), dm::float3(50.f)) };
    CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->AddComponent(graph->GetEntity(early),
                                                          FindComponent<LocalBounds>(graph->GetWorld()), &wrongBox)));
    early->SetLeaf(MakeVolume(b, 4.f));
    CHECK(RowOf<LocalBounds>(graph, early)->bounds.m_maxs.x == 4.f);
    graph->Refresh(1);
    CHECK(early->GetGlobalBoundingBox().m_maxs.x == 4.f);
    auto cleared = graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(PointLight));
    const LeafObject none;
    CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->AddComponent(graph->GetEntity(cleared),
                                                          FindComponent<LeafObject>(graph->GetWorld()), &none)));
    CHECK(!cleared->GetLeaf());
    auto afterNone = MakeVolume(b, 5.f);
    cleared->SetLeaf(afterNone);
    CHECK(cleared->GetLeaf() == afterNone.Get() && HasValues(afterNone, b, 5.f));
    CHECK(afterNone->Read<Density>() == RowOf<Density>(graph, cleared));
    CHECK(RowOf<LocalBounds>(graph, cleared)->bounds.m_maxs.x == 5.f);
    // The graph's vector still holds the light, which left without a removal (the world's overwrite).
    CHECK(graph->GetLights().size() == 1);
}

// SetLeaf makes no archetype beside the rows it makes: the one without a leaf and the new leaf's.
void test_archetypes()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
    auto makeLeaf = [&](int kind) -> nvrhi::AutoPtr<SceneGraphLeaf>
    {
        switch (kind)
        {
        case 0: return MAKE_RC_OBJ_PTR(MeshInstance, mesh.Get());
        case 1: return MakeVolume(resource, 2.f);
        case 2: return MAKE_RC_OBJ_PTR(SpotLight);
        default: return MAKE_RC_OBJ_PTR(OtherLeaf);   // with a Sparse tag
        }
    };
    for (int kind = 0; kind < 4; ++kind)
    {
        auto graph = MakeGraph();
        IWorld* world = graph->GetWorld();
        // every depth, with and without a local transform, is a row of its own
        SceneGraphNode* parent = graph->GetRootNode();
        for (int depth = 1; depth <= 3; ++depth)
        {
            auto plain = MakeNode();
            parent = graph->Attach(parent, plain);
            for (int local = 0; local < 2; ++local)
            {
                auto node = MakeNode(makeLeaf(kind));
                if (local)
                    node->SetTranslation(dm::double3(1.0, 2.0, 3.0));
                graph->Attach(parent, node);
                // the second of its class comes as the first does
                auto mate = MakeNode(makeLeaf(kind));
                if (local)
                    mate->SetTranslation(dm::double3(1.0, 2.0, 3.0));
                graph->Attach(parent, mate);
                uint32_t count = world->GetArchetypeCount();
                node->SetLeaf(makeLeaf(kind));
                CHECK(world->GetArchetypeCount() == count + 1);   // the row without the leaf
                node->SetLeaf(makeLeaf(kind));
                node->SetLeaf(nvrhi::AutoPtr<SceneGraphLeaf>(node->GetLeaf()));
                mate->SetLeaf(makeLeaf(kind));
                CHECK(world->GetArchetypeCount() == count + 1);
                node->SetLeaf(makeLeaf((kind + 1) % 4));          // another class: its row
                CHECK(world->GetArchetypeCount() == count + 2);
                node->SetLeaf(makeLeaf(kind));
                CHECK(world->GetArchetypeCount() == count + 2);
                count = world->GetArchetypeCount();
                node->SetLeaf(nullptr);                           // the row of a node, which may be there
                CHECK(world->GetArchetypeCount() <= count + 1);
                count = world->GetArchetypeCount();
                node->SetLeaf(makeLeaf(kind));
                mate->SetLeaf(nullptr);
                mate->SetLeaf(makeLeaf(kind));
                CHECK(world->GetArchetypeCount() == count);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------ copies

void test_copies()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    auto graph = MakeGraph();
    auto other = MakeGraph();
    auto top = MakeNode();
    auto leaf = MakeVolume(resource, 2.f);
    auto wide = MAKE_RC_OBJ_PTR(WideVolumeLeaf);
    wide->Write<Wide>()->words[0] = 5;
    *wide->Write<VolumeRef>() = VolumeRef(resource);
    graph->Attach(top, MakeNode(leaf));
    graph->Attach(top, MakeNode(wide));
    graph->Attach(graph->GetRootNode(), top);
    CHECK(References(resource) == 3);

    for (SceneGraph* target : { graph.Get(), other.Get() })
    {
        auto copy = target->Attach(target->GetRootNode(), top);
        CHECK(copy && copy.Get() != top.Get());
        auto copyLeaf = query_cast<VolumeLeaf>(copy->GetFirstChild()->GetLeaf());
        auto copyWide = query_cast<WideVolumeLeaf>(copy->GetLastChild()->GetLeaf());
        CHECK(copyLeaf && copyWide && copyLeaf != leaf.Get());
        CHECK(HasValues(copyLeaf, resource, 2.f) && copyWide->Read<Wide>()->words[0] == 5);
        CHECK(copyLeaf->Read<Density>() == RowOf<Density>(target, copy->GetFirstChild()));
        CHECK(LeafRecord::GetLiveCount() == 0);
        // a copy of its own
        copyLeaf->Write<Density>()->density = 20.f;
        copyWide->Write<Wide>()->words[0] = 50;
        CHECK(HasValues(leaf, resource, 2.f) && wide->Read<Wide>()->words[0] == 5);
        CHECK(References(resource) == 5);
        target->Detach(copy);
        CHECK(References(resource) == 5);
        copy = nullptr;
        CHECK(References(resource) == 3);
    }

    // AttachLeafNode of an attached leaf clones it
    auto cloneNode = other->AttachLeafNode(other->GetRootNode(), leaf);
    CHECK(cloneNode->GetLeaf() != leaf.Get());
    CHECK(HasValues(static_cast<VolumeLeaf*>(cloneNode->GetLeaf()), resource, 2.f) && References(resource) == 4);
}

// ------------------------------------------------------------------------------------------ observers

namespace
{
    struct ProbeState
    {
        std::vector<int> observed, observedGone, registered, unregistered, columnObserved;
        bool editOnRemove = false;
    };

    // A graph that checks the leaves' values in its override and in observers, and removes them in its destructor.
    class ProbeGraph final : public SceneGraph
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        explicit ProbeGraph(ProbeState* state) : m_State(state)
        {
            IWorld* world = GetWorld();
            ObserveTag<VolumeLeafTag>(this, m_Observers, [state](ObserverEvent event, const EntityBatch& batch)
            {
                const VolumeRef* refs = static_cast<const VolumeRef*>(batch.columns[2]);
                const Density* densities = static_cast<const Density*>(batch.columns[3]);
                EXPECT(batch.columns[0] == nullptr);   // a tag
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    auto* leaf = static_cast<VolumeLeaf*>(LeafAt(batch, i));
                    // the row is complete, and it is the leaf's home while the observers run, also at a removal
                    EXPECT(leaf->Read<Density>() == densities + i && leaf->Read<VolumeRef>() == refs + i);
                    EXPECT(densities[i].density == float(leaf->id) && refs[i].object != nullptr);
                    EXPECT(refs[i].state == VolumeRef::c_Alive);
                    (event == ObserverEvent::OnAdd ? state->observed : state->observedGone).push_back(leaf->id);
                    if (event == ObserverEvent::OnRemove && state->editOnRemove)
                        leaf->Write<Density>()->mode = 99;
                }
                return nvrhi::FS_OK;
            }, { RegisterComponent<VolumeRef>(world), RegisterComponent<Density>(world) });
            // an observer on the column itself sees the value, not the default the row was made with
            auto observer = MAKE_RC_OBJ_PTR(Observer, [state](ObserverEvent, const EntityBatch& batch)
            {
                const VolumeRef* refs = static_cast<const VolumeRef*>(batch.columns[0]);
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    EXPECT(refs[i].object != nullptr);
                    state->columnObserved.push_back(static_cast<VolumeLeaf*>(LeafAt(batch, i))->id);
                }
                return nvrhi::FS_OK;
            });
            const QueryTerm term{ RegisterComponent<LeafObject>(world), TermOp::All, Access::Read, {} };
            ObserverDesc desc{};
            desc.filterTerms = &term;
            desc.filterTermCount = 1;
            desc.component = RegisterComponent<VolumeRef>(world);
            desc.access = Access::Read;
            for (ObserverEvent event : { ObserverEvent::OnAdd, ObserverEvent::OnRemove })
            {
                desc.event = event;
                ObserverId id = ObserverId::Invalid;
                CHECK(NVRHI_SUCCEEDED(world->AddObserver(desc, observer, id)));
                m_Observers.push_back(id);
            }
        }
        ~ProbeGraph() override
        {
            for (ObserverId id : m_Observers)
                EXPECT(NVRHI_SUCCEEDED(GetWorld()->RemoveObserver(id)));
        }

    protected:
        void RegisterLeaf(SceneGraphLeaf* leaf) override
        {
            SceneGraph::RegisterLeaf(leaf);
            if (auto* volume = query_cast<VolumeLeaf>(leaf))
            {
                EXPECT(volume->Read<Density>() && volume->Read<Density>()->density == float(volume->id));
                EXPECT(volume->GetLocalBoundingBox().m_maxs.x == float(volume->id));
                EXPECT(volume->GetNode() && GetEntity(volume->GetNode()) != Entity::Null);
                m_State->registered.push_back(volume->id);
            }
        }
        void UnregisterLeaf(SceneGraphLeaf* leaf) override
        {
            SceneGraph::UnregisterLeaf(leaf);
            if (auto* volume = query_cast<VolumeLeaf>(leaf))
            {
                EXPECT(volume->Read<Density>() && volume->Read<Density>()->density == float(volume->id));
                EXPECT(volume->Read<VolumeRef>() && volume->Read<VolumeRef>()->object);
                m_State->unregistered.push_back(volume->id);
            }
        }

    private:
        ProbeState* m_State;
        std::vector<ObserverId> m_Observers;
    };
}

void test_observers_see_values()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    ProbeState state;
    auto graph = MAKE_RC_OBJ_PTR(ProbeGraph, &state);
    graph->SetRootNode(MakeNode());

    // a subgraph of three levels and several chunks: one call of the world per run, runs across depths and chunks
    auto top = MakeNode(MakeVolume(resource, 0.f, 0));
    std::vector<nvrhi::AutoPtr<VolumeLeaf>> leaves;
    std::vector<int> order{ 0 };
    for (int g = 0; g < 3; ++g)
    {
        auto groupLeaf = MakeVolume(resource, float(leaves.size() + 1), int(leaves.size() + 1));
        leaves.push_back(groupLeaf);
        order.push_back(groupLeaf->id);
        auto group = MakeNode(groupLeaf);
        for (int i = 0; i < 700; ++i)
        {
            leaves.push_back(MakeVolume(resource, float(leaves.size() + 1), int(leaves.size() + 1)));
            order.push_back(leaves.back()->id);
            graph->Attach(group, MakeNode(leaves.back()));
        }
        graph->Attach(top, group);
    }
    graph->Attach(graph->GetRootNode(), top);
    CHECK(g_Errors == 0);
    CHECK(state.registered == order && state.observed == order && state.columnObserved == order);
    CHECK(state.unregistered.empty() && state.observedGone.empty());
    CHECK(LeafRecord::GetLiveCount() == 0);

    // the removal's observers read the rows, and what they write is kept
    state.editOnRemove = true;
    graph->Detach(top);
    CHECK(g_Errors == 0);
    CHECK(state.unregistered == order && state.observedGone == order && state.columnObserved.size() == 2 * order.size());
    CHECK(LeafRecord::GetLiveCount() == int64_t(order.size()));
    for (const auto& leaf : leaves)
        CHECK(HasValues(leaf, resource, float(leaf->id)) && leaf->Read<Density>()->mode == 99);
    state.editOnRemove = false;

    // SetLeaf: complete rows too; a leaf of the same class, of a subclass, and the same leaf, are one removal and one
    // addition each
    state = ProbeState{};
    auto node = MakeNode();
    graph->Attach(graph->GetRootNode(), node);
    for (size_t taken : { 10, 11, 12, 20, 21 })
        leaves[taken]->GetNode()->SetLeaf(nullptr);   // off their nodes, which are in no graph
    node->SetLeaf(leaves[10]);
    node->SetLeaf(leaves[11]);
    node->SetLeaf(leaves[11]);
    auto wide = MAKE_RC_OBJ_PTR(WideVolumeLeaf);
    wide->id = 5000;
    wide->Write<Density>()->density = 5000.f;
    *wide->Write<VolumeRef>() = VolumeRef(resource);
    node->SetLeaf(wide);
    node->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
    node->SetLeaf(leaves[12]);
    node->SetLeaf(nullptr);
    const std::vector<int> added{ 11, 12, 12, 5000, 13 };
    CHECK(g_Errors == 0);
    CHECK(state.registered == added && state.observed == added);
    CHECK(state.unregistered == added && state.observedGone == added);
    CHECK(state.columnObserved.size() == 10);

    // it runs after the graph's own observer: a light is in GetLights when an observer on its tag sees it, and out of
    // it when the observer sees it go (not in a Detach, which takes its leaves out at its end)
    std::vector<ObserverId> ids;
    std::vector<size_t> lights;
    SceneGraph* raw = graph;
    ObserveTag<LightLeaf>(graph, ids, [&](ObserverEvent, const EntityBatch&)
    {
        lights.push_back(raw->GetLights().size());
        return nvrhi::FS_OK;
    });
    node->SetLeaf(MAKE_RC_OBJ_PTR(SpotLight));
    node->SetLeaf(MAKE_RC_OBJ_PTR(SpotLight));
    node->SetLeaf(nullptr);
    CHECK((lights == std::vector<size_t>{ 1, 0, 1, 0 }));
    for (ObserverId id : ids)
        CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->RemoveObserver(id)));

    // an observer that fails is reported (one warning per call; Debug: and one by validation per delivery), and undoes
    // nothing
    ids.clear();
    ObserveTag<VolumeLeafTag>(graph, ids, [&](ObserverEvent, const EntityBatch&) { return nvrhi::FE_GENERIC_ERROR; });
    auto failing = graph->AttachLeafNode(graph->GetRootNode(), leaves[20]);
    CHECK(failing && failing->GetGraph() == raw && HasValues(leaves[20], resource, 21.f));
    failing->SetLeaf(leaves[21]);
    CHECK(failing->GetLeaf() == leaves[21].Get() && HasValues(leaves[20], resource, 21.f));
    graph->Detach(failing);
    CHECK(!failing->GetGraph() && HasValues(leaves[21], resource, 22.f));
#ifdef NDEBUG
    CHECK(g_Warnings == 3);
#else
    CHECK(g_Warnings == 3 + 4);
#endif
    g_Warnings = 0;
    for (ObserverId id : ids)
        CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->RemoveObserver(id)));

    // observers' commands that never settle: the same
    class Endless final : public nvrhi::ObjectImpl<IObserver>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Endless)
        NVRHI_IMPLEMENTS_INTERFACE(IObserver)
        NVRHI_END_INTERFACE_TABLE()
        ComponentId wide = ComponentId::Invalid;
        nvrhi::FRESULT OnEntities(ObserverEvent event, const EntityBatch& batch,
                                  ICommandBuffer* commands) noexcept override
        {
            if (event == ObserverEvent::OnAdd)
                commands->RemoveComponents(batch.entities, batch.count, wide, 0);
            else
                commands->AddComponents(batch.entities, batch.count, wide, nullptr, 0);
            return nvrhi::FS_OK;
        }
    };
    auto endless = MAKE_RC_OBJ_PTR(Endless);
    endless->wide = RegisterComponent<Wide>(graph->GetWorld());
    ids.clear();
    for (ObserverEvent event : { ObserverEvent::OnAdd, ObserverEvent::OnRemove })
    {
        ObserverDesc desc{};
        desc.component = endless->wide;
        desc.access = Access::Read;
        desc.event = event;
        ObserverId id = ObserverId::Invalid;
        CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->AddObserver(desc, endless, id)));
        ids.push_back(id);
    }
    auto settling = graph->AttachLeafNode(graph->GetRootNode(), wide);   // WideVolumeLeaf: with the column Wide
    CHECK(settling && settling->GetGraph() == raw && settling->GetLeaf() == wide.Get());
#ifdef NDEBUG
    CHECK(g_Warnings == 1);
#else
    CHECK(g_Warnings == 2);
#endif
    g_Warnings = 0;
    for (ObserverId id : ids)
        CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->RemoveObserver(id)));
}

// A leaf class's box read from a column on job threads (dynamicBounds), and the misuse warnings of Debug builds.
void test_dynamic_bounds_and_guards()
{
    JobSystemDesc desc;
    desc.debugName = "leaf_columns";
    desc.workerThreadCount = 8;
    nvrhi::AutoPtr<IJobSystem> jobs;
    CHECK(NVRHI_SUCCEEDED(CreateJobSystem(desc, &jobs)));
    auto graph = MakeGraph();
    graph->SetJobSystem(jobs);
    std::vector<nvrhi::AutoPtr<DynamicLeaf>> leaves;
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 3000; ++i)
    {
        leaves.push_back(MAKE_RC_OBJ_PTR(DynamicLeaf));
        leaves.back()->Write<Density>()->density = float(i + 1);
        nodes.push_back(graph->AttachLeafNode(graph->GetRootNode(), leaves.back()));
    }
    graph->Refresh(0);
    graph->Refresh(1);
    for (int i = 0; i < 3000; ++i)
        CHECK(RowOf<WorldBounds>(graph, nodes[size_t(i)])->bounds.m_maxs.x == float(i + 1));
    // no InvalidateContent
    for (int i = 0; i < 3000; i += 7)
        leaves[size_t(i)]->Write<Density>()->density = float(5000 + i);
    graph->Refresh(2);
    for (int i = 0; i < 3000; ++i)
    {
        const float expected = i % 7 == 0 ? float(5000 + i) : float(i + 1);
        CHECK(RowOf<WorldBounds>(graph, nodes[size_t(i)])->bounds.m_maxs.x == expected);
    }
    CHECK(graph->GetRootNode()->GetGlobalBoundingBox().m_maxs.x == 7996.f);

    // Edit of an attached leaf is a write through the API: between BeginRenderRead and EndRenderRead it warns in Debug
    // builds and still applies. Get is none, and neither is the Edit of a leaf in no graph.
    graph->BeginRenderRead();
    leaves[0]->Write<Density>()->density = 2.f;
    (void)leaves[0]->Read<Density>();
    (void)MAKE_RC_OBJ_PTR(DynamicLeaf)->Write<Density>();
    graph->EndRenderRead();
    CHECK(leaves[0]->Read<Density>()->density == 2.f);
#ifndef NDEBUG
    CHECK(g_Warnings == 1);
    g_Warnings = 0;
#endif
    CHECK(g_Warnings == 0);
}

// ------------------------------------------------------------------------------------------ refusals

void test_refused_attach()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    auto graph = MakeGraph();
    auto top = MakeNode();   // a batch of its own
    std::vector<nvrhi::AutoPtr<VolumeLeaf>> leaves;
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 4; ++i)
    {
        leaves.push_back(MakeVolume(resource, float(i), i));
        nodes.push_back(MakeNode(leaves.back()));
        graph->Attach(top, nodes.back());
    }
    SceneGraphStore::FailCreateAfter(1);
    CHECK(!graph->Attach(graph->GetRootNode(), top));
    SceneGraphStore::FailCreateAfter(-1);
    CHECK(g_Warnings >= 1);
    g_Warnings = 0;
    CHECK(top->GetGraph() && LeafRecord::GetLiveCount() == 4 && References(resource) == 5);
    for (int i = 0; i < 4; ++i)
        CHECK(!nodes[size_t(i)]->GetGraph() && HasValues(leaves[size_t(i)], resource, float(i)));
    // they attach again
    for (const auto& node : nodes)
        graph->Attach(top, node);
    for (int i = 0; i < 4; ++i)
        CHECK(HasValues(leaves[size_t(i)], resource, float(i)) &&
              leaves[size_t(i)]->Read<Density>() == RowOf<Density>(graph, nodes[size_t(i)]));
    CHECK(LeafRecord::GetLiveCount() == 0);

    // A batch that the world makes in part: two leaves of one class at two depths are one call and two archetypes, and
    // the second has no page. The first one's values were in the row already.
    std::vector<int> events;
    std::vector<ObserverId> ids;
    ObserveTag<VolumeLeafTag>(graph, ids, [&](ObserverEvent event, const EntityBatch& batch)
    {
        for (uint32_t i = 0; i < batch.count; ++i)
        {
            auto* leaf = static_cast<VolumeLeaf*>(LeafAt(batch, i));
            EXPECT(HasValues(leaf, resource, float(leaf->id)));
            events.push_back(event == ObserverEvent::OnAdd ? leaf->id : -leaf->id);
        }
        return nvrhi::FS_OK;
    });
    auto upperLeaf = MakeVolume(resource, 50.f, 50);
    auto lowerLeaf = MakeVolume(resource, 51.f, 51);
    auto upper = MakeNode(upperLeaf);
    auto lower = MakeNode(lowerLeaf);
    graph->Attach(upper, lower);
    SceneGraphNode* parent = nodes[0];
    graph->Attach(parent, upper);   // the archetypes, the tables
    graph->Detach(upper);
    CHECK(!events.empty() && g_Errors == 0);
    events.clear();
    // Use up the pages but one: every new depth takes one.
    ecs::WorldImpl* world = ecs::GetWorldImpl(graph->GetWorld());
    ecs::ReleaseSharedSlabs();
    SceneGraphNode* deepest = graph->GetRootNode();
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> chain;
    while (world->GetFreeChunkPageCount() != 1)
    {
        CHECK(chain.size() < 60);
        chain.push_back(MakeNode());
        deepest = graph->Attach(deepest, chain.back());
    }
    ecs::ReleaseSharedSlabs();
    ecs::SetHeapFailureCountdown(0);
    const bool attached = graph->Attach(parent, upper) != nullptr;
    ecs::SetHeapFailureCountdown(-1);
    CHECK(!attached && g_Errors == 0 && g_Warnings >= 1);
    g_Warnings = 0;
    CHECK((events == std::vector<int>{ 50, -50 }));   // it came and went
    CHECK(LeafRecord::GetLiveCount() == 2);
    CHECK(HasValues(upperLeaf, resource, 50.f) && HasValues(lowerLeaf, resource, 51.f));
    CHECK(!upper->GetGraph() && !lower->GetGraph());
    graph->Attach(upper, lower);
    graph->Attach(parent, upper);
    CHECK(HasValues(upperLeaf, resource, 50.f) && HasValues(lowerLeaf, resource, 51.f));
    CHECK(LeafRecord::GetLiveCount() == 0);

    // SetLeaf that cannot move the row: the leaf stays, with its values (the archetype that the row would pass through
    // exists, and has no page)
    auto replacement = MakeVolume(resource, 60.f, 60);
    upper->SetLeaf(upperLeaf);
    while (world->GetFreeChunkPageCount() != 0)
    {
        CHECK(chain.size() < 120);
        chain.push_back(MakeNode());
        deepest = graph->Attach(deepest, chain.back());
    }
    events.clear();
    ecs::ReleaseSharedSlabs();
    ecs::SetHeapFailureCountdown(0);
    upper->SetLeaf(replacement);
    ecs::SetHeapFailureCountdown(-1);
    CHECK(g_Warnings == 1 && events.empty());
    g_Warnings = 0;
    CHECK(upper->GetLeaf() == upperLeaf.Get() && upperLeaf->GetNode() == upper.Get());
    CHECK(HasValues(upperLeaf, resource, 50.f) && upperLeaf->Read<Density>() == RowOf<Density>(graph, upper));
    CHECK(LeafRecord::GetLiveCount() == 1);   // the replacement's
    upper->SetLeaf(replacement);
    CHECK(HasValues(upperLeaf, resource, 50.f) && HasValues(replacement, resource, 60.f));

    // A leaf whose row fits no chunk: asked for before anything changes, so the old leaf stays, with its values, box and
    // events (none); the same at a node without a leaf.
    auto noDebris = [&](SceneGraphNode* at)
    {
        return !RowOf<LocalBounds>(graph, at) && !RowOf<WorldBounds>(graph, at) && !RowOf<LeafObject>(graph, at) &&
            !graph->GetWorld()->HasComponent(graph->GetEntity(at), FindComponent<MarkedLeafTag>(graph->GetWorld())) &&
            !graph->GetWorld()->HasComponent(graph->GetEntity(at), FindComponent<VolumeLeafTag>(graph->GetWorld()));
    };
    auto huge = MAKE_RC_OBJ_PTR(HugeLeaf);
    huge->Write<Huge>()->bytes[7] = 70;
    events.clear();
    const uint32_t entityArchetype = world->FindAlive(graph->GetEntity(upper))->archetype;
    upper->SetLeaf(huge);
    auto leafless = MakeNode();
    graph->Attach(parent, leafless);
    leafless->SetLeaf(huge);
#ifdef NDEBUG
    CHECK(g_Warnings == 2);
#else
    CHECK(g_Warnings == 4);   // and validation's
#endif
    g_Warnings = 0;
    CHECK(events.empty() && upper->GetLeaf() == replacement.Get() && replacement->GetNode() == upper.Get());
    CHECK(world->FindAlive(graph->GetEntity(upper))->archetype == entityArchetype);
    CHECK(HasValues(replacement, resource, 60.f) && RowOf<LocalBounds>(graph, upper)->bounds.m_maxs.x == 60.f);
    CHECK(!graph->GetWorld()->HasComponent(graph->GetEntity(upper), FindComponent<MarkedLeafTag>(graph->GetWorld())));
    CHECK(!huge->GetNode() && huge->Read<Huge>()->bytes[7] == 70 && !leafless->GetLeaf() && noDebris(leafless));
    upper->SetLeaf(upperLeaf);
    CHECK(RowOf<LocalBounds>(graph, upper)->bounds.m_maxs.x == 50.f);

    // Out of memory between the two moves: the old leaf is gone and the new one cannot come. No leaf, nothing of either
    // left in the row, both with their values. Two rows of the class: the old chunk gives no page back.
    auto mate = MakeNode(MakeVolume(resource, 61.f, 61));
    graph->Attach(parent, mate);
    auto otherLeaf = MAKE_RC_OBJ_PTR(OtherLeaf);
    otherLeaf->Write<Density>()->density = 62.f;
    mate->SetLeaf(otherLeaf);   // the archetypes, the sparse set's pages and masks
    mate->SetLeaf(nullptr);
    auto mateLeaf = MakeVolume(resource, 61.f, 61);
    mate->SetLeaf(mateLeaf);
    CHECK(world->FindAlive(graph->GetEntity(mate))->archetype == world->FindAlive(graph->GetEntity(upper))->archetype);
    while (world->GetFreeChunkPageCount() != 1)
    {
        CHECK(chain.size() < 120);
        chain.push_back(MakeNode());
        deepest = graph->Attach(deepest, chain.back());
    }
    events.clear();
    ecs::ReleaseSharedSlabs();
    ecs::SetHeapFailureCountdown(0);
    upper->SetLeaf(otherLeaf);
    ecs::SetHeapFailureCountdown(-1);
    CHECK(g_Warnings == 1 && (events == std::vector<int>{ -50 }));
    g_Warnings = 0;
    CHECK(!upper->GetLeaf() && !upperLeaf->GetNode() && !otherLeaf->GetNode() && noDebris(upper));
    CHECK(HasValues(upperLeaf, resource, 50.f) && otherLeaf->Read<Density>()->density == 62.f);
    graph->Refresh(3);
    CHECK(upper->GetLeafContentFlags() == SceneContentFlags::None);
    auto next = MakeVolume(resource, 3.f, 3);
    upper->SetLeaf(next);
    CHECK(RowOf<LocalBounds>(graph, upper)->bounds.m_maxs.x == 3.f && HasValues(next, resource, 3.f));
    CHECK(!graph->GetWorld()->HasComponent(graph->GetEntity(upper), FindComponent<MarkedLeafTag>(graph->GetWorld())));
    upper->SetLeaf(otherLeaf);
    CHECK(graph->GetWorld()->HasComponent(graph->GetEntity(upper), FindComponent<MarkedLeafTag>(graph->GetWorld())));
    CHECK(otherLeaf->Read<Density>() == RowOf<Density>(graph, upper) && otherLeaf->Read<Density>()->density == 62.f);

    // A box that throws: the same end, and the exception goes on.
    auto throwing = MakeVolume(resource, 63.f, 63);
    throwing->throwBox = true;
    bool caught = false;
    try
    {
        upper->SetLeaf(throwing);
    }
    catch (const std::runtime_error&)
    {
        caught = true;
    }
    CHECK(caught && !upper->GetLeaf() && !throwing->GetNode() && noDebris(upper));
    CHECK(HasValues(throwing, resource, 63.f) && otherLeaf->Read<Density>()->density == 62.f);
    throwing->throwBox = false;
    upper->SetLeaf(throwing);
    CHECK(HasValues(throwing, resource, 63.f) && RowOf<LocalBounds>(graph, upper)->bounds.m_maxs.x == 63.f);
    CHECK(upper->GetLocalToWorldTransform().m_translation.x == 0.0);   // the row hint is the row's
    for (ObserverId id : ids)
        CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->RemoveObserver(id)));
}

// An UnregisterLeaf callback that attaches the leaf to another graph: the leaf takes its values along, out of the row
// that it is leaving.
namespace
{
    class HandOverGraph final : public SceneGraph
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        SceneGraph* target = nullptr;

    protected:
        void UnregisterLeaf(SceneGraphLeaf* leaf) override
        {
            SceneGraph::UnregisterLeaf(leaf);
            if (target && !leaf->GetNode())   // SetLeaf: the leaf names no node any more
                target->AttachLeafNode(target->GetRootNode(), leaf);
        }
    };
}

void test_hand_over()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    auto graph = MAKE_RC_OBJ_PTR(HandOverGraph);
    graph->SetRootNode(MakeNode());
    auto other = MakeGraph();
    auto leaf = MakeVolume(resource, 3.f);
    auto node = graph->AttachLeafNode(graph->GetRootNode(), leaf);
    graph->target = other;
    // the observers of the removal that come after the callback see default values (SceneGraph.h)
    std::vector<ObserverId> ids;
    float seen = -1.f;
    ObserveTag<VolumeLeafTag>(graph, ids, [&](ObserverEvent event, const EntityBatch& batch)
    {
        if (event == ObserverEvent::OnRemove)
            seen = static_cast<const Density*>(batch.columns[2])->density;
        return nvrhi::FS_OK;
    }, { RegisterComponent<Density>(graph->GetWorld()) });
    node->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
    CHECK(seen == 1.5f);
    for (ObserverId id : ids)
        CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->RemoveObserver(id)));
    CHECK(leaf->GetNode() && leaf->GetNode()->GetGraph() == other.Get());
    CHECK(HasValues(leaf, resource, 3.f) && leaf->Read<Density>() == RowOf<Density>(other, leaf->GetNode()));
    CHECK(LeafRecord::GetLiveCount() == 0 && References(resource) == 2 && g_LiveVolumeRefs == 1);
}

// ------------------------------------------------------------------------------------------ equivalence

namespace
{
    // One event of one leaf: the class it was seen as, and whether it came.
    struct LeafEvent
    {
        SceneGraphLeaf* leaf;
        uint32_t kind;
        bool added;
        bool operator==(const LeafEvent& other) const
        {
            return leaf == other.leaf && kind == other.kind && added == other.added;
        }
    };

    enum Kind : uint32_t { KindMesh, KindCamera, KindLight, KindAnimation, KindSkinned, KindTraced, KindCount };

    struct LeafLog
    {
        std::vector<LeafEvent> events;                                  // the classes that exclude each other
        std::vector<LeafEvent> subclassEvents[2];                       // skinned, traced
        std::vector<nvrhi::AutoPtr<SceneGraphLeaf>> leaves[KindCount];  // as the graph's vectors

        void Record(uint32_t kind, SceneGraphLeaf* leaf, bool added)
        {
            (kind < KindSkinned ? events : subclassEvents[kind - KindSkinned]).push_back({ leaf, kind, added });
            auto& items = leaves[kind];
            if (added)
                items.push_back(leaf);
            else
            {
                auto it = std::find(items.begin(), items.end(), leaf);
                EXPECT(it != items.end());
                if (it != items.end())
                    items.erase(it);
            }
        }
    };

    // The same bookkeeping twice in one graph: from the overrides, and from observers on the class tags.
    class EquivalenceGraph final : public SceneGraph
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        LeafLog overridden;
        LeafLog observed;

        EquivalenceGraph()
        {
            Observe<MeshInstanceLeaf>(KindMesh);
            Observe<CameraLeaf>(KindCamera);
            Observe<LightLeaf>(KindLight);
            Observe<AnimationLeaf>(KindAnimation);
            Observe<SkinnedMeshInstanceLeaf>(KindSkinned);
            Observe<TracedMeshLeafTag>(KindTraced);
        }
        ~EquivalenceGraph() override
        {
            for (ObserverId id : m_Observers)
                EXPECT(NVRHI_SUCCEEDED(GetWorld()->RemoveObserver(id)));
        }

    protected:
        void RegisterLeaf(SceneGraphLeaf* leaf) override
        {
            SceneGraph::RegisterLeaf(leaf);
            Override(leaf, true);
        }
        void UnregisterLeaf(SceneGraphLeaf* leaf) override
        {
            SceneGraph::UnregisterLeaf(leaf);
            Override(leaf, false);
        }

    private:
        std::vector<ObserverId> m_Observers;

        void Override(SceneGraphLeaf* leaf, bool added)
        {
            if (query_cast<MeshInstance>(leaf)) overridden.Record(KindMesh, leaf, added);
            if (query_cast<SceneCamera>(leaf)) overridden.Record(KindCamera, leaf, added);
            if (query_cast<Light>(leaf)) overridden.Record(KindLight, leaf, added);
            if (query_cast<SceneGraphAnimation>(leaf)) overridden.Record(KindAnimation, leaf, added);
            if (query_cast<SkinnedMeshInstance>(leaf)) overridden.Record(KindSkinned, leaf, added);
            if (auto* traced = query_cast<TracedMesh>(leaf))
            {
                EXPECT(traced->HoldsValue());
                overridden.Record(KindTraced, leaf, added);
            }
        }
        template <typename Tag>
        void Observe(uint32_t kind)
        {
            ObserveTag<Tag>(this, m_Observers, [this, kind](ObserverEvent event, const EntityBatch& batch)
            {
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    SceneGraphLeaf* leaf = LeafAt(batch, i);
                    if (kind == KindTraced)
                        EXPECT(static_cast<TracedMesh*>(leaf)->HoldsValue());
                    observed.Record(kind, leaf, event == ObserverEvent::OnAdd);
                }
                return nvrhi::FS_OK;
            });
        }
    };

    // Every second mesh instance is one whose class keeps a column in a record.
    class TracedFactory final : public SceneTypeFactory
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        uint32_t made = 0;
        nvrhi::AutoPtr<MeshInstance> CreateMeshInstance(MeshInfo* mesh) override
        {
            if (++made % 2 == 0)
                return MAKE_RC_OBJ_PTR(MeshInstance, mesh);
            return MAKE_RC_OBJ_PTR(TracedMesh, mesh, float(made));
        }
    };

    template <typename T>
    bool SameLeaves(const std::vector<nvrhi::AutoPtr<SceneGraphLeaf>>& a, const std::vector<nvrhi::AutoPtr<T>>& b)
    {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (a[i].Get() != static_cast<SceneGraphLeaf*>(b[i].Get()))
                return false;
        }
        return true;
    }

    void CheckEquivalent(EquivalenceGraph* graph)
    {
        CHECK(g_Errors == 0);
        CHECK(graph->overridden.events == graph->observed.events);
        for (int k = 0; k < 2; ++k)
            CHECK(graph->overridden.subclassEvents[k] == graph->observed.subclassEvents[k]);
        for (int k = 0; k < KindCount; ++k)
            CHECK(graph->overridden.leaves[k] == graph->observed.leaves[k]);
        CHECK(SameLeaves(graph->observed.leaves[KindMesh], graph->GetMeshInstances()));
        CHECK(SameLeaves(graph->observed.leaves[KindSkinned], graph->GetSkinnedMeshInstances()));
        CHECK(SameLeaves(graph->observed.leaves[KindCamera], graph->GetCameras()));
        CHECK(SameLeaves(graph->observed.leaves[KindLight], graph->GetLights()));
        CHECK(SameLeaves(graph->observed.leaves[KindAnimation], graph->GetAnimations()));
        size_t attachedTraced = 0;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
        {
            if (auto* traced = query_cast<TracedMesh>(walker->GetLeaf()))
            {
                CHECK(traced->HoldsValue());
                ++attachedTraced;
            }
        }
        CHECK(attachedTraced == graph->observed.leaves[KindTraced].size());
    }
}

namespace
{
    // The leaves of the graph's nodes in pre-order, from the rows: independent of what any observer reported.
    std::vector<nvrhi::AutoPtr<SceneGraphLeaf>> AttachedLeaves(SceneGraph* graph)
    {
        std::vector<nvrhi::AutoPtr<SceneGraphLeaf>> leaves;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
        {
            if (SceneGraphLeaf* leaf = walker->GetLeaf())
                leaves.push_back(leaf);
        }
        return leaves;
    }

    // What both bookkeepings must have seen, built from the walks before and after an edit.
    struct ExpectedEvents
    {
        std::vector<LeafEvent> events;
        std::vector<LeafEvent> subclassEvents[2];

        void Add(SceneGraphLeaf* leaf, bool added)
        {
            if (query_cast<MeshInstance>(leaf)) events.push_back({ leaf, KindMesh, added });
            if (query_cast<SceneCamera>(leaf)) events.push_back({ leaf, KindCamera, added });
            if (query_cast<Light>(leaf)) events.push_back({ leaf, KindLight, added });
            if (query_cast<SceneGraphAnimation>(leaf)) events.push_back({ leaf, KindAnimation, added });
            if (query_cast<SkinnedMeshInstance>(leaf)) subclassEvents[0].push_back({ leaf, KindSkinned, added });
            if (query_cast<TracedMesh>(leaf)) subclassEvents[1].push_back({ leaf, KindTraced, added });
        }
    };

    bool EndsWith(const std::vector<LeafEvent>& log, size_t from, const std::vector<LeafEvent>& expected)
    {
        return log.size() == from + expected.size() && std::equal(expected.begin(), expected.end(), log.begin() + from);
    }
}

// The E0 generator's edits, one by one. Detach unregisters in pre-order, Attach registers in pre-order, SetLeaf does the
// one and then the other, SetRootNode with the old root under the new one all of them: the expected events come from the
// walks before and after each edit, and both bookkeepings must have exactly them, so that an event that everybody lost,
// or saw twice, fails too.
void test_equivalence()
{
    struct Spec
    {
        const char* name;
        uint64_t seed;
        const Profile* profile;
        void (*build)(Scene&, uint32_t);
        uint32_t size;
    };
    const Spec specs[] = {
        { "flat", 0x3001, &c_SmallProfile, BuildFlat, 40 },
        { "tree", 0x3002, &c_SmallProfile, BuildTree, 60 },
        { "crowd", 0x3003, &c_SmallProfile, BuildCrowd, 4 },
        { "mixed", 0x3004, &c_SmallProfile, BuildMixed, 120 },
        { "mixed_large", 0x3005, &c_LargeProfile, BuildMixed, 1200 },
        { "chain", 0x3006, &c_LargeProfile, BuildChain, 150 },
    };
    for (const Spec& spec : specs)
    {
        size_t events = 0, traced = 0, sameClass = 0, edits = 0;
        std::string sequence;
        for (int run = 0; run < 2; ++run)
        {
            Scene s(spec.seed, *spec.profile);
            auto graph = MAKE_RC_OBJ_PTR(EquivalenceGraph);
            s.graph = graph;
            s.factory = MAKE_RC_OBJ_PTR(TracedFactory);
            spec.build(s, spec.size);
            s.nodeLimit = CollectNodes(s.graph).size() * 3 / 2 + 64;
            CheckEquivalent(graph);

            // Every leaf ever seen is held, so that an address names one leaf, and numbered in the order of the walks.
            std::unordered_map<SceneGraphLeaf*, size_t> ids;
            std::vector<nvrhi::AutoPtr<SceneGraphLeaf>> kept;
            std::string text;
            auto number = [&](SceneGraphLeaf* leaf)
            {
                const auto [it, fresh] = ids.emplace(leaf, ids.size());
                if (fresh)
                    kept.push_back(leaf);
                return it->second;
            };
            auto check = [&](const ExpectedEvents& expected, const size_t (&from)[3])
            {
                for (LeafLog* log : { &graph->overridden, &graph->observed })
                {
                    CHECK(EndsWith(log->events, from[0], expected.events));
                    CHECK(EndsWith(log->subclassEvents[0], from[1], expected.subclassEvents[0]));
                    CHECK(EndsWith(log->subclassEvents[1], from[2], expected.subclassEvents[1]));
                }
                for (const LeafEvent& event : expected.events)
                    text += (event.added ? '+' : '-') + std::to_string(number(event.leaf)) + char('a' + event.kind);
                CheckEquivalent(graph);
            };
            // The build: each leaf of the graph came once, and none went.
            {
                ExpectedEvents built;
                for (const auto& leaf : AttachedLeaves(graph))
                {
                    number(leaf);
                    built.Add(leaf, true);
                }
                CHECK(graph->observed.events.size() == built.events.size());
                for (const LeafEvent& event : built.events)
                    CHECK(std::count(graph->observed.events.begin(), graph->observed.events.end(), event) == 1);
            }
            // `all`: the edit takes the whole graph out and in.
            auto edit = [&](bool all, auto&& call)
            {
                const auto before = AttachedLeaves(graph);
                for (const auto& leaf : before)
                    number(leaf);
                const size_t from[3] = { graph->observed.events.size(), graph->observed.subclassEvents[0].size(),
                                         graph->observed.subclassEvents[1].size() };
                call();
                const auto after = AttachedLeaves(graph);
                const auto contains = [](const auto& list, SceneGraphLeaf* leaf)
                {
                    return std::any_of(list.begin(), list.end(), [&](const auto& item) { return item.Get() == leaf; });
                };
                ExpectedEvents expected;
                for (const auto& leaf : before)
                    if (all || !contains(after, leaf))
                        expected.Add(leaf, false);
                for (const auto& leaf : after)
                    if (all || !contains(before, leaf))
                        expected.Add(leaf, true);
                check(expected, from);
                ++edits;
            };
            const Profile& p = s.profile;
            for (uint32_t frame = 1; frame < 40; ++frame)
            {
                for (const auto& node : s.movers)
                    Move(s, node);
                const bool first = frame == 1;
                if (first || s.rng.Percent(p.detach)) edit(false, [&] { EditDetach(s); });
                if (first || s.rng.Percent(p.setLeaf)) edit(false, [&] { EditSetLeaf(s); });
                if (first || s.rng.Percent(p.orphan)) edit(false, [&] { EditOrphan(s); });
                if (first || s.rng.Percent(p.leafNode)) edit(false, [&] { EditLeafNode(s); });
                if (first || s.rng.Percent(p.reattach)) edit(false, [&] { EditReattach(s); });
                if (first || s.rng.Percent(p.copy)) edit(false, [&] { EditCopy(s); });
                if (first || s.rng.Percent(p.setRoot)) edit(true, [&] { EditSetRoot(s); });
                s.graph->Refresh(frame);
                for (const auto& subgraph : s.detached)
                {
                    for (SceneGraphWalker walker(subgraph.Get()); walker; walker.Next(true))
                    {
                        if (auto* leaf = query_cast<TracedMesh>(walker->GetLeaf()))
                            CHECK(leaf->HoldsValue());
                    }
                }
                // SetLeaf between two leaves of one class, which the generator's random classes rarely give
                const auto& meshes = graph->GetMeshInstances();
                if (!meshes.empty())
                {
                    SceneGraphNode* node = meshes[frame % meshes.size()]->GetNode();
                    const bool wasTraced = query_cast<TracedMesh>(node->GetLeaf()) != nullptr;
                    edit(false, [&] { node->SetLeaf(s.factory->CreateMeshInstance(s.meshes[frame % s.meshes.size()])); });
                    sameClass += wasTraced == (query_cast<TracedMesh>(node->GetLeaf()) != nullptr);
                    // the same leaf goes and comes
                    nvrhi::AutoPtr<SceneGraphLeaf> same = node->GetLeaf();
                    const size_t from[3] = { graph->observed.events.size(), graph->observed.subclassEvents[0].size(),
                                             graph->observed.subclassEvents[1].size() };
                    node->SetLeaf(same);
                    ExpectedEvents expected;
                    expected.Add(same, false);
                    expected.Add(same, true);
                    check(expected, from);
                }
            }
            // the order does not come from addresses: a second run in this process gives the same leaves' events
            CHECK(run == 0 || text == sequence);
            sequence = text;
            events = graph->observed.events.size();
            traced = graph->observed.subclassEvents[1].size();
        }
        std::printf("  %s: %zu edits, %zu events, %zu of leaves with a record, %zu SetLeaf within a class\n", spec.name,
                    edits, events, traced, sameClass);
        CHECK(traced > 0 && sameClass > 0);
    }
}

// ------------------------------------------------------------------------------------------ teardown

void test_teardown()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);

    // attached leaves, one of them and one node referenced elsewhere; the subclass removes its observers: no callback
    {
        ProbeState state;
        auto graph = MAKE_RC_OBJ_PTR(ProbeGraph, &state);
        graph->SetRootNode(MakeNode());
        auto held = MakeVolume(resource, 1.f, 1);
        auto byNode = graph->AttachLeafNode(graph->GetRootNode(), MakeVolume(resource, 2.f, 2));
        auto child = MakeNode(MakeVolume(resource, 3.f, 3));
        graph->Attach(byNode, child);
        graph->AttachLeafNode(graph->GetRootNode(), held);
        for (int i = 10; i < 600; ++i)
            graph->AttachLeafNode(graph->GetRootNode(), MakeVolume(resource, float(i), i));
        CHECK(References(resource) == 594);
        const ProbeState before = state;
        graph = nullptr;
        CHECK(g_Errors == 0);
        CHECK(state.registered == before.registered && state.observed == before.observed);
        CHECK(state.unregistered.empty() && state.observedGone.empty());
        // the referenced leaf and the leaves of the referenced node's subgraph have their values; the others are gone
        CHECK(References(resource) == 4 && LeafRecord::GetLiveCount() == 3);
        CHECK(HasValues(held, resource, 1.f) && !held->GetNode());
        CHECK(HasValues(static_cast<VolumeLeaf*>(byNode->GetLeaf()), resource, 2.f));
        CHECK(HasValues(static_cast<VolumeLeaf*>(child->GetLeaf()), resource, 3.f));
        held->Write<Density>()->mode = 70;
        // and they go into a new graph
        auto next = MakeGraph();
        next->Attach(next->GetRootNode(), byNode);
        auto heldNode = next->AttachLeafNode(next->GetRootNode(), held);
        CHECK(heldNode->GetLeaf() == held.Get() && held->Read<Density>() == RowOf<Density>(next, heldNode));
        CHECK(held->Read<Density>()->mode == 70 && LeafRecord::GetLiveCount() == 0);
    }
    CHECK(References(resource) == 1 && LeafRecord::GetLiveCount() == 0);

    // a held world: its entities live on without a graph, and go later without a callback
    for (int how = 0; how < 2; ++how)
    {
        ProbeState state;
        auto graph = MAKE_RC_OBJ_PTR(ProbeGraph, &state);
        auto root = MakeNode();
        graph->SetRootNode(root);
        auto held = MakeVolume(resource, 1.f, 1);
        auto heldNode = graph->AttachLeafNode(graph->GetRootNode(), held);
        const Entity heldEntity = graph->GetEntity(heldNode);
        heldNode = nullptr;
        nvrhi::AutoPtr<VolumeLeaf> unreferenced = MakeVolume(resource, 2.f, 2);
        auto node = graph->AttachLeafNode(graph->GetRootNode(), unreferenced);
        const Entity entity = graph->GetEntity(node);
        const Entity rootEntity = graph->GetEntity(root);
        VolumeLeaf* weak = unreferenced;   // the world holds it
        unreferenced = nullptr;
        node = nullptr;
        root = nullptr;
        nvrhi::AutoPtr<IWorld> world = graph->GetWorld();
        const ProbeState before = state;
        graph = nullptr;
        CHECK(world->IsAlive(heldEntity) && world->IsAlive(entity) && world->IsAlive(rootEntity));
        CHECK(GetComponent<LeafObject>(world, entity)->object == weak);
        // the leaves are in no graph, so their records are the values' homes; the rows keep default values
        CHECK(HasValues(held, resource, 1.f) && HasValues(weak, resource, 2.f));
        CHECK(LeafRecord::GetLiveCount() == 2 && References(resource) == 3);
        CHECK(GetComponent<Density>(world, heldEntity)->density == 1.5f);
        CHECK(!GetComponent<VolumeRef>(world, entity)->object);
        CHECK(held->Read<Density>() != GetComponent<Density>(world, heldEntity));
        SceneGraphNode* heldIn = GetComponent<SceneNode>(world, heldEntity)->object;   // lives in the world, in no graph
        CHECK(!held->GetNode() && !heldIn->GetLeaf() && !heldIn->GetGraph());
        (void)heldIn->GetTranslation();
        held->Write<Density>()->density = 8.f;
        CHECK(held->Read<Density>()->density == 8.f && GetComponent<Density>(world, heldEntity)->density == 1.5f);
        if (how == 0)
        {
            const Entity entities[] = { rootEntity, heldEntity, entity };
            CHECK(NVRHI_SUCCEEDED(world->DestroyEntities(entities, 3)));
            CHECK(References(resource) == 2 && LeafRecord::GetLiveCount() == 1);   // the leaf that nobody held is gone
        }
        world = nullptr;
        CHECK(References(resource) == 2 && LeafRecord::GetLiveCount() == 1);
        CHECK(g_Errors == 0 && !held->GetNode());
        CHECK(state.registered == before.registered && state.observed == before.observed);
        CHECK(state.unregistered.empty() && state.observedGone.empty() && state.columnObserved == before.columnObserved);
    }

    // a graph of one leaf that the caller holds, without anything else
    auto leaf = MakeVolume(resource, 9.f);
    {
        auto graph = MakeGraph();
        graph->AttachLeafNode(graph->GetRootNode(), leaf);
        CHECK(LeafRecord::GetLiveCount() == 0);
    }
    CHECK(HasValues(leaf, resource, 9.f) && LeafRecord::GetLiveCount() == 1);
}

// ------------------------------------------------------------------------------------------ threads

void test_worker_threads()
{
    constexpr int threads = 8;
    constexpr int perThread = 400;
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> tops(threads);
    auto build = [&](int t)
    {
        tops[size_t(t)] = MakeNode();
        for (int i = 0; i < perThread; ++i)
        {
            const int id = t * perThread + i;
            auto leaf = MakeVolume(resource, float(id), id);
            auto drop = MakeVolume(resource, 1.f);   // records go back to the pool on this thread too
            EXPECT(HasValues(leaf, resource, float(id)) && HasValues(drop, resource, 1.f));
            auto node = MakeNode(leaf);
            SceneGraphStore::AppendOrphan(tops[size_t(t)], node);
        }
    };
    auto run = [&](auto&& body)
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back(body, t);
        for (std::thread& thread : pool)
            thread.join();
    };
    run(build);
    CHECK(g_Errors == 0 && LeafRecord::GetLiveCount() == threads * perThread);
    CHECK(References(resource) == threads * perThread + 1);

    auto graph = MakeGraph();
    for (const auto& top : tops)
        graph->Attach(graph->GetRootNode(), top);
    CHECK(LeafRecord::GetLiveCount() == 0);
    // readers on several threads, between structural changes
    run([&](int t)
    {
        for (SceneGraphWalker walker(tops[size_t(t)]); walker; walker.Next(true))
        {
            if (auto* leaf = query_cast<VolumeLeaf>(walker->GetLeaf()))
                EXPECT(HasValues(leaf, resource, float(leaf->id)));
        }
    });
    for (const auto& top : tops)
        graph->Detach(top);
    CHECK(g_Errors == 0 && LeafRecord::GetLiveCount() == threads * perThread);
    run([&](int t) { tops[size_t(t)] = nullptr; });
    CHECK(LeafRecord::GetLiveCount() == 0 && References(resource) == 1 && g_LiveVolumeRefs == 0);
}

// ------------------------------------------------------------------------------------------ random edits

// Every leaf beside what its columns must hold, through random calls on two graphs; after each call the values, the
// records alive (one per leaf that is in no graph and had a record) and the references are compared.
void test_random_edits()
{
    struct Model
    {
        nvrhi::AutoPtr<VolumeLeaf> leaf;
        float density;
        Resource* resource;
        uint32_t word;   // WideVolumeLeaf's
    };
    for (uint64_t seed = 1; seed <= 4; ++seed)
    {
        Rng rng(seed * 7919);
        std::vector<nvrhi::AutoPtr<Resource>> resources;
        for (int i = 0; i < 3; ++i)
            resources.push_back(MAKE_RC_OBJ_PTR(Resource));
        nvrhi::AutoPtr<SceneGraph> graphs[2] = { MakeGraph(), MakeGraph() };
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        std::vector<Model> models;
        for (int i = 0; i < 24; ++i)
        {
            nodes.push_back(MakeNode());
            Model model;
            model.density = float(i);
            model.resource = resources[rng.Below(3)];
            model.word = 1000u + uint32_t(i);
            if (i % 3 == 0)
            {
                auto wide = MAKE_RC_OBJ_PTR(WideVolumeLeaf);
                wide->Write<Wide>()->words[7] = model.word;
                model.leaf = wide;
            }
            else
                model.leaf = MAKE_RC_OBJ_PTR(VolumeLeaf);
            model.leaf->id = i;
            model.leaf->Write<Density>()->density = model.density;
            *model.leaf->Write<VolumeRef>() = VolumeRef(model.resource);
            models.push_back(model);
        }
        uint32_t attached = 0;
        for (int step = 0; step < 3000; ++step)
        {
            SceneGraphNode* node = nodes[rng.Below(nodes.size())];
            Model& model = models[rng.Below(models.size())];
            switch (rng.Below(9))
            {
            case 0:
            case 7:
            case 8:   // attach under a root or another node (never under itself)
                if (!node->GetGraph())
                {
                    SceneGraph* graph = graphs[rng.Below(2)];
                    SceneGraphNode* parent = nodes[rng.Below(nodes.size())];
                    if (parent->GetGraph())
                        parent->GetGraph()->Attach(parent, node);
                    else
                        graph->Attach(graph->GetRootNode(), node);
                }
                break;
            case 1:
                if (node->GetGraph())
                    node->GetGraph()->Detach(node);
                break;
            case 2:   // a leaf that no node holds
                if (!model.leaf->GetNode())
                    node->SetLeaf(model.leaf);
                break;
            case 3:   // the same leaf, or none
                node->SetLeaf(rng.Percent(50) ? nvrhi::AutoPtr<SceneGraphLeaf>(node->GetLeaf()).Get() : nullptr);
                break;
            case 4:
                model.density = float(step);
                model.leaf->Write<Density>()->density = model.density;
                break;
            case 5:
                model.resource = resources[rng.Below(3)];
                *model.leaf->Write<VolumeRef>() = VolumeRef(model.resource);
                break;
            default:   // a copy, dropped at once
                if (SceneGraphLeaf* leaf = node->GetLeaf(); leaf && node->GetGraph())
                {
                    SceneGraph* graph = graphs[rng.Below(2)];
                    auto copy = graph->Attach(graph->GetRootNode(), node);
                    auto* copied = copy ? static_cast<VolumeLeaf*>(copy->GetLeaf()) : nullptr;
                    const Model& source = models[size_t(static_cast<VolumeLeaf*>(leaf)->id)];
                    CHECK(copied && copied->Read<Density>()->density == source.density);
                    CHECK(copied->Read<VolumeRef>()->object == source.resource);
                    graph->Detach(copy);
                }
                break;
            }
            int64_t records = 0;
            uint32_t references[3] = { 1, 1, 1 };
            attached = 0;
            for (const Model& m : models)
            {
                const Density* density = m.leaf->Read<Density>();
                const VolumeRef* ref = m.leaf->Read<VolumeRef>();
                CHECK(density && ref && density->density == m.density && ref->object == m.resource);
                if (auto* wide = query_cast<WideVolumeLeaf>(m.leaf.Get()))
                    CHECK(wide->Read<Wide>()->words[7] == m.word);
                SceneGraphNode* holder = m.leaf->GetNode();
                if (holder && holder->GetGraph())
                {
                    CHECK(density == RowOf<Density>(holder->GetGraph(), holder));
                    ++attached;
                }
                else
                    ++records;
                for (int r = 0; r < 3; ++r)
                    references[r] += m.resource == resources[size_t(r)].Get();
            }
            CHECK(LeafRecord::GetLiveCount() == records);
            CHECK(g_LiveVolumeRefs == int32_t(models.size()) && g_Errors == 0);
            for (int r = 0; r < 3; ++r)
                CHECK(References(resources[size_t(r)]) == references[r]);
            if (step % 50 == 0)
            {
                graphs[0]->Refresh(uint32_t(step));
                graphs[1]->Refresh(uint32_t(step));
            }
        }
        std::printf("  seed %llu: %u of %zu leaves attached at the end\n", (unsigned long long)seed, attached,
                    models.size());
    }
}

// ------------------------------------------------------------------------------------------ the set of columns

namespace
{
    // A record made by a base class's constructor with other columns: the attach moves what it finds by GUID, gives the
    // rest its default and drops what the class does not have. Debug builds assert there.
    void AttachMismatch(bool check)
    {
        auto resource = MAKE_RC_OBJ_PTR(Resource);
        auto leaf = MAKE_RC_OBJ_PTR(EagerSubLeaf, resource.Get());
        leaf->Read<Density>();
        CHECK(!check || (leaf->Read<VolumeRef>()->object == resource.Get() && !leaf->Read<Extent>()));
        auto graph = MakeGraph();
        auto node = graph->AttachLeafNode(graph->GetRootNode(), leaf);
        CHECK(!check || (leaf->Read<Density>()->density == 4.f && leaf->Read<Density>()->mode == 7));
        CHECK(!check || (leaf->Read<Extent>() && leaf->Read<Extent>()->values[0] == 0.0));
        CHECK(!check || (!leaf->Read<VolumeRef>() && References(resource) == 1 && LeafRecord::GetLiveCount() == 0));
    }

    const char* g_Argv0 = "";
}

void test_mismatch()
{
#ifdef NDEBUG
    AttachMismatch(true);
#elif defined(_MSC_VER)
    // the assert ends the process: a child
    std::fflush(stdout);
    const intptr_t code = _spawnl(_P_WAIT, g_Argv0, "test_scene_graph_leaf_columns", "--child-mismatch",
                                  static_cast<const char*>(nullptr));
    std::printf("  the child ended with %lld\n", (long long)code);
    CHECK(code == 3);   // abort
#else
    const int code = std::system((std::string("\"") + g_Argv0 + "\" --child-mismatch").c_str());
    CHECK(code != 0);
#endif
}

// ------------------------------------------------------------------------------------------ ResourceTracker

// First-use order, whatever the addresses: GVDB's volume list (rule 3.3.6).
void test_resource_tracker()
{
    std::vector<nvrhi::AutoPtr<Resource>> resources;
    for (int i = 0; i < 64; ++i)
        resources.push_back(MAKE_RC_OBJ_PTR(Resource));
    Rng rng(42);
    for (int round = 0; round < 20; ++round)
    {
        ResourceTracker<Resource> tracker;
        std::vector<Resource*> model;               // in the order of the first reference
        std::vector<uint32_t> counts(64, 0);
        for (int step = 0; step < 500; ++step)
        {
            const size_t which = rng.Below(64);
            Resource* resource = resources[which];
            if (rng.Percent(55) || counts[which] == 0)
            {
                CHECK(tracker.AddRef(resource) == (counts[which] == 0));
                if (counts[which]++ == 0)
                    model.push_back(resource);
                CHECK(References(resource) == 2);   // ours and the tracker's
            }
            else
            {
                CHECK(tracker.Release(resource) == (counts[which] == 1));
                if (--counts[which] == 0)
                {
                    model.erase(std::find(model.begin(), model.end(), resource));
                    CHECK(References(resource) == 1);
                }
            }
            if (step % 10 == 0)
            {
                std::vector<Resource*> seen;
                for (Resource* item : tracker)
                    seen.push_back(item);
                CHECK(seen == model && tracker.size() == model.size() && tracker.empty() == model.empty());
            }
        }
        // a copy is a tracker of its own
        ResourceTracker<Resource> copy = tracker;
        if (!model.empty())
        {
            while (!tracker.Release(model.front())) { }
        }
        std::vector<Resource*> seen;
        for (auto it = copy.begin(); it != copy.end(); it++)
            seen.push_back(*it);
        CHECK(seen == model);
        CHECK(!tracker.AddRef(nullptr) && !tracker.Release(nullptr));
        // moved: the target iterates as the source did, the source is empty and can be used again
        ResourceTracker<Resource> moved = std::move(copy);
        CHECK(copy.empty() && copy.begin() == copy.end());
        ResourceTracker<Resource> assigned;
        assigned.AddRef(resources[0]);
        assigned = std::move(moved);
        CHECK(moved.empty() && moved.begin() == moved.end());
        seen.clear();
        for (Resource* item : assigned)
            seen.push_back(item);
        CHECK(seen == model);
        CHECK(copy.AddRef(resources[5]) && *copy.begin() == resources[5].Get() && copy.Release(resources[5]));
    }
}

// ------------------------------------------------------------------------------------------ allocations

void test_allocations()
{
    auto resource = MAKE_RC_OBJ_PTR(Resource);
    auto graph = MakeGraph();
    std::vector<ObserverId> ids;
    uint64_t seen = 0;
    ObserveTag<VolumeLeafTag>(graph, ids, [&](ObserverEvent, const EntityBatch& batch)
    {
        seen += batch.count;
        return nvrhi::FS_OK;
    });
    auto top = MakeNode();
    std::vector<nvrhi::AutoPtr<VolumeLeaf>> leaves;
    std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
    for (int i = 0; i < 1000; ++i)
    {
        leaves.push_back(MakeVolume(resource, float(i), i));
        nodes.push_back(MakeNode(leaves.back()));
        graph->Attach(top, nodes.back());
    }
    auto single = MakeNode();
    graph->Attach(graph->GetRootNode(), single);
    auto extraA = MakeVolume(resource, 1.f);
    auto extraB = MakeVolume(resource, 2.f);
    auto extraOther = MAKE_RC_OBJ_PTR(OtherLeaf);
    extraOther->Write<Density>();

    uint32_t frame = 0;
    auto structural = [&](bool refresh)
    {
        graph->Attach(graph->GetRootNode(), top);
        if (refresh)
            graph->Refresh(frame++);
        graph->Detach(top);
        if (refresh)
            graph->Refresh(frame++);
        for (int i = 0; i < 100; ++i)
        {
            graph->Attach(graph->GetRootNode(), nodes[size_t(i)]);   // out of the orphan `top`, one by one
            graph->Detach(nodes[size_t(i)]);
            graph->Attach(top, nodes[size_t(i)]);
        }
        single->SetLeaf(extraA);
        single->SetLeaf(extraB);
        single->SetLeaf(extraB);
        single->SetLeaf(extraOther);
        single->SetLeaf(nullptr);
    };
    auto count = [](auto&& body, uint64_t& news, uint64_t& blocks)
    {
        const uint64_t newsBefore = g_NewCount.load();
        const uint64_t blocksBefore = ecs::GetHeapAllocationCount();
        body();
        news = g_NewCount.load() - newsBefore;
        blocks = ecs::GetHeapAllocationCount() - blocksBefore;
    };
    uint64_t news = 0, blocks = 0;
    count([&] { structural(true); structural(true); structural(false); }, news, blocks);
    const uint64_t warmNews = news, warmBlocks = blocks;
    count([&] { for (int round = 0; round < 10; ++round) structural(false); }, news, blocks);
    std::printf("  10 rounds of attach, detach and SetLeaf: %llu operator new, %llu heap blocks (warm-up: %llu, %llu); "
                "%llu leaves observed\n", (unsigned long long)news, (unsigned long long)blocks,
                (unsigned long long)warmNews, (unsigned long long)warmBlocks, (unsigned long long)seen);
    CHECK(news == 0 && blocks == 0 && seen > 20000);
    CHECK(warmNews + warmBlocks > 0);   // the counters count
#ifdef NDEBUG
    // with Refresh between them too (in Debug builds each Refresh after a structural change makes one operator new, by
    // reading the iterator-debugging proxy of the local std::vector in ReconcileMeshes)
    count([&] { for (int round = 0; round < 10; ++round) structural(true); }, news, blocks);
    CHECK(news == 0 && blocks == 0);
#endif

    // Get and Edit, attached and not
    graph->Attach(graph->GetRootNode(), nodes[0]);
    float sum = 0.f;
    count([&]
    {
        for (int i = 0; i < 1000; ++i)
        {
            sum += leaves[0]->Read<Density>()->density + leaves[1]->Read<Density>()->density;
            leaves[0]->Write<Density>()->mode = uint32_t(i);
            leaves[1]->Write<Density>()->mode = uint32_t(i);
        }
    }, news, blocks);
    CHECK(news == 0 && blocks == 0 && sum == 1000.f);

    // records are recycled: the first Edit of a leaf takes a block that another gave back
    for (int i = 0; i < 64; ++i)
        (void)MakeVolume(resource, 1.f);
    uint64_t leafNews = 0;
    count([&] { for (int i = 0; i < 1000; ++i) { (void)MAKE_RC_OBJ_PTR(VolumeLeaf); } }, leafNews, blocks);
    count([&] { for (int i = 0; i < 1000; ++i) { (void)MakeVolume(resource, 1.f); } }, news, blocks);
    std::printf("  1000 leaves with a record made and dropped: %llu operator new (%llu without Edit), %llu heap blocks\n",
                (unsigned long long)news, (unsigned long long)leafNews, (unsigned long long)blocks);
    CHECK(news == leafNews && blocks == 0);
    for (ObserverId id : ids)
        CHECK(NVRHI_SUCCEEDED(graph->GetWorld()->RemoveObserver(id)));
}

int main(int argc, char** argv)
{
    g_Argv0 = argv[0];
    if (argc > 1 && std::strcmp(argv[1], "--child-mismatch") == 0)
    {
#if defined(_MSC_VER)
        _set_error_mode(_OUT_TO_STDERR);
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
        AttachMismatch(false);
        return 0;
    }

    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        { "record_rule", test_record_rule },
        { "change_components", test_change_components },
        { "attach_detach", test_attach_detach },
        { "set_leaf", test_set_leaf },
        { "archetypes", test_archetypes },
        { "copies", test_copies },
        { "observers", test_observers_see_values },
        { "dynamic_bounds", test_dynamic_bounds_and_guards },
        { "refused", test_refused_attach },
        { "hand_over", test_hand_over },
        { "equivalence", test_equivalence },
        { "teardown", test_teardown },
        { "threads", test_worker_threads },
        { "random_edits", test_random_edits },
        { "mismatch", test_mismatch },
        { "resource_tracker", test_resource_tracker },
        { "allocations", test_allocations },
    };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
                continue;
            c.run();
            CHECK(g_Warnings.load() == 0);
            CHECK(g_Errors.load() == 0);
            CheckNothingLeft(c.name);
            std::printf("%s: passed\n", c.name);
            std::fflush(stdout);
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
