/*
* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#pragma once
#include <nvrhi/core/foundation.h>
#include <nvrhi/core/autoptr.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/AssetTables.h>
#include <donut/engine/KeyframeAnimation.h>
#include <donut/core/math/math.h>
#include <donut/core/query_cast.h>
#include <unordered_map>
#include <utility>
#include <functional>
#include <filesystem>
#include <iterator>
#include <atomic>

namespace donut::engine
{
    class SceneGraph;
    class SceneGraphNode;
    class SceneTypeFactory;
    class SceneGraphStore;   // the graph's world and systems (src/engine/SceneGraphStore.h)
    class GpuScene;
    struct IWorld;
    struct IJobSystem;
    struct IScheduler;
    struct FrameTime;              // EcsScheduler.h
    enum class Phase : uint32_t;   // EcsScheduler.h

    enum struct SceneContentFlags : uint32_t
    {
        None = 0,
        OpaqueMeshes = 0x01,
        AlphaTestedMeshes = 0x02,
        BlendedMeshes = 0x04,
        Lights = 0x08,
        Cameras = 0x10,
        Animations = 0x20
    };

    NVRHI_CLASS_CLSID(SceneGraphLeaf, "c62d1eac-cc06-42ba-a1fe-dc5895517302")
    class SceneGraphLeaf: public nvrhi::WeakReferenceSourceImpl<nvrhi::IWeakReferenceSource>
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SceneGraphLeaf)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SceneGraphLeaf)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()

    private:
        friend class SceneGraphNode;
        friend class SceneGraph;
        friend class SceneGraphStore;
        nvrhi::WeakPtr<SceneGraphNode> m_Node;
        uint32_t m_RemovalPending = 0;   // unregistered this often in a Detach; SceneGraph compacts its vectors after it
        uint32_t m_RemovalTaken = 0;
        LeafRecord* m_LeafRecord = nullptr;   // while no row is the values' home; only classes that declare leafRecord

        [[nodiscard]] const void* GetColumn(const ComponentTypeDesc& type) const noexcept;
        [[nodiscard]] void* EditColumn(const ComponentTypeDesc& type) noexcept;

    protected:
        // The leaf's value of a column that its class declares (LeafTypeDesc::columns): the row while the leaf is
        // attached, its record otherwise. A leaf in no graph has a record once Edit made it or it left a graph
        // (leafRecord only). Null when there is neither, or T is not one of the class's columns, or not one with a value
        // per row (a tag, a buffer, a Shared, Chunk or Sparse component). A leaf that leaves a graph (Detach, SetLeaf)
        // has its values in the row until the last observer of the removal has run, UnregisterLeaf included: they call
        // Get and Edit as before. The pointer is valid until the next structural change of the graph, or until the leaf
        // enters or leaves one. Allocates nothing, makes no record. Several threads may call it between structural
        // changes; in a phase the caller's system declares the access.
        template <typename T> [[nodiscard]] const T* Get() const noexcept
        {
            return static_cast<const T*>(GetColumn(ComponentTraits<T>::Desc()));
        }
        // The same to write through: stamps the column with the world's version while attached, on the thread that uses
        // the graph's API, outside phases (design 5.6). A leaf in no graph may be edited on any one thread; its first
        // Edit makes the record, with every column of its class in its default value (null when out of memory), and is
        // the only call here that allocates. The record's columns are the ones of GetTypeDesc at that call. A
        // constructor runs with the GetTypeDesc of the class under construction: it may call Edit only in a class whose
        // subclasses add no columns.
        template <typename T> [[nodiscard]] T* Edit() noexcept
        {
            return static_cast<T*>(EditColumn(ComponentTraits<T>::Desc()));
        }

    public:
        using WeakReferenceSourceImpl::WeakReferenceSourceImpl;
        virtual ~SceneGraphLeaf();   // out of line: LeafRecord is only forward-declared here

        [[nodiscard]] SceneGraphNode* GetNode() const { return m_Node.Lock(); }
        [[nodiscard]] virtual dm::box3 GetLocalBoundingBox() { return dm::box3::empty(); }
        [[nodiscard]] virtual nvrhi::AutoPtr<SceneGraphLeaf> Clone() = 0;
        // What the class keeps in columns and the tags of its class chain (D19), read once when the leaf is attached.
        // The default is an object-only leaf without tags; a subclass that does not override it gets its base's.
        [[nodiscard]] virtual const LeafTypeDesc& GetTypeDesc() const;
        [[nodiscard]] virtual SceneContentFlags GetContentFlags() const { return SceneContentFlags::None; }
        [[nodiscard]] const std::string& GetName() const;
        void SetName(const std::string& name) const;
        virtual void Load(const Json::Value& node) { }
        virtual bool SetProperty(const std::string& name, const dm::float4& value) { return false; }

        // Non-copyable and non-movable
        SceneGraphLeaf(const SceneGraphLeaf&) = delete;
        SceneGraphLeaf(const SceneGraphLeaf&&) = delete;
        SceneGraphLeaf& operator=(const SceneGraphLeaf&) = delete;
        SceneGraphLeaf& operator=(const SceneGraphLeaf&&) = delete;
    };

    NVRHI_CLASS_CLSID(MeshInstance, "5c165f2e-7e3a-4e58-8ffd-b8612f5829b9")
    class MeshInstance : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(MeshInstance)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(MeshInstance)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(MeshInstance)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()
    protected:
        // The authoring reference, set before the leaf is attached and not changed while it is (no setter exists):
        // while attached, the instance's MeshRef column names the same mesh in the graph's mesh table, and its
        // GpuInstanceSlot column holds its stable slots (plan step E4).
        nvrhi::AutoPtr<MeshInfo> m_Mesh;

    public:
        explicit MeshInstance(MeshInfo* mesh)
            : m_Mesh(mesh)
        { }

        [[nodiscard]] MeshInfo* GetMesh() const { return m_Mesh; }
        // The stable slots of the attached instance (its GpuInstanceSlot column, design 7.2): its row in the instance
        // buffer, and the first of its geometry-instance slots, one per geometry. They stay the same until the leaf
        // leaves the graph; -1 while it is in none.
        [[nodiscard]] int GetInstanceIndex() const;
        [[nodiscard]] int GetGeometryInstanceIndex() const;
        [[nodiscard]] dm::box3 GetLocalBoundingBox() override { return m_Mesh->objectSpaceBounds; }
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        [[nodiscard]] SceneContentFlags GetContentFlags() const override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
        // The content flags of an instance of `mesh` (its geometries' material domains; None for null): what
        // MeshInstance::GetContentFlags gives (a subclass may override it), and what a runtime-only mesh instance of
        // `mesh` gets when it joins (SceneComponents.h).
        [[nodiscard]] static SceneContentFlags GetMeshContentFlags(const MeshInfo* mesh) noexcept;
    };

    struct SkinnedMeshJoint
    {
        nvrhi::WeakPtr<SceneGraphNode> node;
        dm::float4x4 inverseBindMatrix;
    };

    NVRHI_CLASS_CLSID(SkinnedMeshInstance, "f62efbc7-5386-4881-9692-186b85bf2040")
    class SkinnedMeshInstance : public MeshInstance
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SkinnedMeshInstance)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SkinnedMeshInstance)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(SkinnedMeshInstance)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(MeshInstance)
        NVRHI_END_INTERFACE_TABLE()
    protected:
        friend class SceneGraph;
        friend class SceneGraphStore;
        nvrhi::AutoPtr<MeshInfo> m_PrototypeMesh;
        uint32_t m_LastUpdateFrameIndex = 0;
        nvrhi::AutoPtr<SceneTypeFactory> m_SceneTypeFactory;

    public:
        std::vector<SkinnedMeshJoint> joints;
        nvrhi::BufferHandle jointBuffer;
        nvrhi::BindingSetHandle skinningBindingSet;
        bool skinningInitialized = false;

        explicit SkinnedMeshInstance(SceneTypeFactory* sceneTypeFactory, MeshInfo* prototypeMesh);

        [[nodiscard]] const MeshInfo* GetPrototypeMesh() const { return m_PrototypeMesh; }
        [[nodiscard]] uint32_t GetLastUpdateFrameIndex() const { return m_LastUpdateFrameIndex; }
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
    };

    // This leaf is attached to the joint nodes for a skeleton, and it makes them point at the mesh.
    // When the bones are updated, the mesh is flagged for rebuild.
    // Cannot do this through the graph because the skeleton can be separate from the mesh instance node.
    NVRHI_CLASS_CLSID(SkinnedMeshReference, "c6b937bf-9117-465f-be6b-8303d11eafd4")
    class SkinnedMeshReference : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SkinnedMeshReference)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SkinnedMeshReference)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(SkinnedMeshReference)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()
    private:
        friend class SceneGraph;
        friend class SceneGraphStore;
        nvrhi::WeakPtr<SkinnedMeshInstance> m_Instance;
    public:
       explicit SkinnedMeshReference(SkinnedMeshInstance* instance)
           : m_Instance(instance) {}
       [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
       [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
    };

    NVRHI_CLASS_CLSID(SceneCamera, "b030d3bf-c8c8-453c-9570-2eede9086ebf")
    class SceneCamera : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SceneCamera)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SceneCamera)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(SceneCamera)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()
    public:
        [[nodiscard]] SceneContentFlags GetContentFlags() const override { return SceneContentFlags::Cameras; }
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;

        [[nodiscard]] dm::affine3 GetViewToWorldMatrix() const;
        [[nodiscard]] dm::affine3 GetWorldToViewMatrix() const;

    protected:
       using SceneGraphLeaf::SceneGraphLeaf;
    };

    NVRHI_CLASS_CLSID(PerspectiveCamera, "5f410d0d-0596-4c2a-bb27-be68b7145106")
    class PerspectiveCamera : public SceneCamera
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(PerspectiveCamera)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(PerspectiveCamera)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(PerspectiveCamera)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneCamera)
        NVRHI_END_INTERFACE_TABLE()
    public:
        float zNear = 1.f;
        float verticalFov = 1.f; // in radians
        std::optional<float> zFar; // use reverse infinite projection if not specified
        std::optional<float> aspectRatio;

        using SceneCamera::SceneCamera;
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        void Load(const Json::Value& node) override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
    };

    NVRHI_CLASS_CLSID(OrthographicCamera, "bac6c248-a65c-41d7-a671-1dd1fbb4c67f")
    class OrthographicCamera : public SceneCamera
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(OrthographicCamera)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(OrthographicCamera)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(OrthographicCamera)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneCamera)
        NVRHI_END_INTERFACE_TABLE()
    public:
        float zNear = 0.f;
        float zFar = 1.f;
        float xMag = 1.f;
        float yMag = 1.f;

        using SceneCamera::SceneCamera;
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        void Load(const Json::Value& node) override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
    };

    class IShadowMap;

    NVRHI_CLASS_CLSID(Light, "b07e0ef2-7130-4d37-8dea-6f7ef7848177")
    class Light : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(Light)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Light)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(Light)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()
    public:
        nvrhi::AutoPtr<IShadowMap> shadowMap;
        int shadowChannel = -1;
        dm::float3 color = dm::colors::white;

        [[nodiscard]] SceneContentFlags GetContentFlags() const override { return SceneContentFlags::Lights; }
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;

        [[nodiscard]] virtual int GetLightType() const = 0;
        virtual void FillLightConstants(LightConstants& lightConstants) const;
        virtual void Store(Json::Value& node) const { }
        bool SetProperty(const std::string& name, const dm::float4& value) override;

        [[nodiscard]] dm::double3 GetPosition() const;
        [[nodiscard]] dm::double3 GetDirection() const;

        // Through the node's setters, over the parent's world as it is shown; on a node that a simulation drives (a
        // teleport request) over the parent's simulated world, as the node's getters are.
        void SetPosition(const dm::double3& position) const;
        void SetDirection(const dm::double3& direction) const;
    protected:
        Light();
    public:
        ~Light() override;   // out of line: IShadowMap is only forward-declared here
    };

    NVRHI_CLASS_CLSID(DirectionalLight, "48bc24b4-3827-49bd-a18d-331c1e8dc71f")
    class DirectionalLight : public Light
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(DirectionalLight)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DirectionalLight)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(DirectionalLight)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(Light)
        NVRHI_END_INTERFACE_TABLE()
    public:
        float irradiance = 1.f; // Target illuminance (lm/m2) of surfaces lit by this light; multiplied by `color`.
        float angularSize = 0.f; // Angular size of the light source, in degrees.
        std::vector<nvrhi::AutoPtr<IShadowMap>> perObjectShadows;

        DirectionalLight();
        ~DirectionalLight() override;   // out of line: IShadowMap is only forward-declared here
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] int GetLightType() const override { return LightType_Directional; }
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        void FillLightConstants(LightConstants& lightConstants) const override;
        void Load(const Json::Value& node) override;
        void Store(Json::Value& node) const override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
    };

    NVRHI_CLASS_CLSID(SpotLight, "f95dd469-8ee4-4aa5-9afa-34f424666ebb")
    class SpotLight : public Light
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SpotLight)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SpotLight)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(SpotLight)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(Light)
        NVRHI_END_INTERFACE_TABLE()
    public:
        float intensity = 1.f;  // Luminous intensity of the light (lm/sr) in its primary direction; multiplied by `color`.
        float radius = 0.f;     // Radius of the light sphere, in world units.
        float range = 0.f;      // Range of influence for the light. 0 means infinite range.
        float innerAngle = 180.f;    // Apex angle of the full-bright cone, in degrees; constant intensity inside the inner cone, smooth falloff between inside and outside.
        float outerAngle = 180.f;    // Apex angle of the light cone, in degrees - everything outside of that cone is dark.

        SpotLight();
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] int GetLightType() const override { return LightType_Spot; }
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        void FillLightConstants(LightConstants& lightConstants) const override;
        void Load(const Json::Value& node) override;
        void Store(Json::Value& node) const override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
    };

    NVRHI_CLASS_CLSID(PointLight, "ca2235fa-993c-4142-b51c-7ec4f949dedb")
    class PointLight : public Light
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(PointLight)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(PointLight)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(PointLight)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(Light)
        NVRHI_END_INTERFACE_TABLE()
    public:
        float intensity = 1.f;  // Luminous intensity of the light (lm/sr); multiplied by `color`.
        float radius = 0.f;    // Radius of the light sphere, in world units.
        float range = 0.f;     // Range of influence for the light. 0 means infinite range.

        PointLight();
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] int GetLightType() const override { return LightType_Point; }
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        void FillLightConstants(LightConstants& lightConstants) const override;
        void Load(const Json::Value& node) override;
        void Store(Json::Value& node) const override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
    };
    
    class SceneGraphNode final : public nvrhi::WeakReferenceSourceImpl<nvrhi::IWeakReferenceSource>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SceneGraphNode)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_END_INTERFACE_TABLE()

        // Deprecated (design 5.7): nodes no longer keep dirty flags. GetDirtyFlags answers from the graph's pending
        // changes (see there); kept so that existing code compiles.
        enum struct DirtyFlags : uint32_t
        {
            None                    = 0,
            LocalTransform          = 0x01,
            PrevTransform           = 0x02,
            Leaf                    = 0x04,
            SubgraphStructure       = 0x08,
            SubgraphTransforms      = 0x10,
            SubgraphPrevTransforms  = 0x20,
            SubgraphContentUpdate   = 0x40,
            SubgraphMask            = (SubgraphStructure | SubgraphTransforms | SubgraphPrevTransforms | SubgraphContentUpdate)
        };

    private:
        friend class SceneGraph;
        friend class SceneGraphStore;
        friend class SceneGraphWalker;   // its Next reads an attached node's links in one step

        // A node is a handle (design 5.2): while it is attached, its state lives in the columns of its entity in the
        // graph's world (SceneComponents.h) and the world holds it through SceneNode; while it is in no graph, its
        // state lives in an orphan record that it owns (5.3). A record also holds the node's leaf and, like the old
        // node, its children: a node owns its first child and every child owns its next sibling.
        SceneGraphStore* m_Store = nullptr;   // the graph's storage while attached
        Entity m_Entity = Entity::Null;       // valid while attached
        OrphanRecord* m_Record = nullptr;     // valid while in no graph (and while it is being attached)
        // Where the row was last found: valid while the archetype's count of row moves equals m_HintMoves (~0 when
        // unknown). Attach and the setters write it; getters only read it, so that threads may read nodes at once.
        uint64_t m_HintMoves = ~uint64_t(0);
        uint32_t m_HintArchetype = 0;
        uint32_t m_HintChunk = 0;
        uint32_t m_HintRow = 0;
        std::string m_Name;

        // Derived state (5.7): the subgraph's bounds and content flags and the leaf's content flags, valid while
        // m_CacheEpoch equals the graph's epoch. SceneGraph::Refresh fills the subgraph values of the nodes with
        // children (their own values stay in their rows); between a structural change and the next Refresh they are
        // computed on demand, under the store's lock, and published with release order, so that threads that walk the
        // graph at the same time (views rendered in parallel) may fill and read them.
        mutable std::atomic<uint64_t> m_CacheEpoch{ 0 };
        mutable dm::box3 m_CachedBounds = dm::box3::empty();
        mutable SceneContentFlags m_CachedLeafContent = SceneContentFlags::None;
        mutable SceneContentFlags m_CachedSubgraphContent = SceneContentFlags::None;

        OrphanRecord& EnsureRecord();

    public:
       SceneGraphNode() {}
       /* non-virtual */ ~SceneGraphNode();

       // Getters return by value (5.2): no reference into a chunk that a structural change moves.
       [[nodiscard]] dm::dquat GetRotation() const;
       [[nodiscard]] dm::double3 GetScaling() const;
       [[nodiscard]] dm::double3 GetTranslation() const;

       // scaling * rotation * translation of the local transform; the identity for a node without one
       [[nodiscard]] dm::daffine3 GetLocalToParentTransform() const;
        // As computed by the last SceneGraph::Refresh (or kept from the node's last graph while it is in none).
        [[nodiscard]] dm::daffine3 GetLocalToWorldTransform() const;
        [[nodiscard]] dm::affine3 GetLocalToWorldTransformFloat() const;
        // The union of the leaf's bounds and the subgraph's, computed on demand and cached until the graph changes.
        [[nodiscard]] dm::box3 GetGlobalBoundingBox() const;
        // Deprecated: an approximation from the graph's pending changes (DirtyFlags). Not marked [[deprecated]],
        // because test_scene_graph_nodes, which must stay unchanged through step E3, calls it.
        [[nodiscard]] DirtyFlags GetDirtyFlags() const;
        [[nodiscard]] SceneContentFlags GetLeafContentFlags() const;
        [[nodiscard]] SceneContentFlags GetSubgraphContentFlags() const;

        // Forward iteration over the children of a node, in order:
        //     for (SceneGraphNode* child : node->GetChildren()) ...
        // The loop body must not detach the child it is visiting.
        class ChildIterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = SceneGraphNode*;
            using difference_type = std::ptrdiff_t;
            using pointer = SceneGraphNode* const*;
            using reference = SceneGraphNode* const&;

            ChildIterator() = default;
            explicit ChildIterator(SceneGraphNode* node) : m_Node(node) { }

            reference operator*() const { return m_Node; }
            ChildIterator& operator++() { m_Node = m_Node->GetNextSibling(); return *this; }
            ChildIterator operator++(int) { ChildIterator previous = *this; ++*this; return previous; }
            bool operator==(const ChildIterator& other) const { return m_Node == other.m_Node; }
            bool operator!=(const ChildIterator& other) const { return m_Node != other.m_Node; }

        private:
            SceneGraphNode* m_Node = nullptr;
        };

        struct ChildRange
        {
            SceneGraphNode* first = nullptr;
            [[nodiscard]] ChildIterator begin() const { return ChildIterator(first); }
            [[nodiscard]] ChildIterator end() const { return ChildIterator(); }
        };

        // The links read SceneLinks and Parent and map entities back to nodes through SceneNode (5.2).
        [[nodiscard]] SceneGraphNode* GetParent() const;
        [[nodiscard]] ChildRange GetChildren() const { return ChildRange{ GetFirstChild() }; }
        [[nodiscard]] SceneGraphNode* GetFirstChild() const;
        [[nodiscard]] SceneGraphNode* GetLastChild() const;
        [[nodiscard]] SceneGraphNode* GetNextSibling() const;
        [[nodiscard]] SceneGraphNode* GetPrevSibling() const;
        [[nodiscard]] SceneGraphLeaf* GetLeaf() const;

        [[nodiscard]] const std::string& GetName() const { return m_Name; }
        [[nodiscard]] SceneGraph* GetGraph() const;

        [[nodiscard]] std::filesystem::path GetPath() const;

        void InvalidateContent();

        // Writes the parts that are given (not null). On a node that a simulation drives
        // (SceneGraph::SetSimulationDriven) it is a teleport request instead: neither the node's pose nor what the
        // getters return changes before the domain takes it, in the next Simulate with a step of it. The request is in
        // the space of the parent, where the simulation has that at the next Simulate (a pending request of the parent
        // itself does not count; a parent without an inverse counts as none: the request is then in the world), so
        // setting what the getters returned moves nothing. A part that is not given is that of the pending request, or
        // else where the simulation has the node now; a later request replaces the parts it gives. The rotation is
        // normalised; numbers that are not finite and a rotation of 0 are refused with a warning. Unbinding and Detach
        // drop a pending request; the request of a node of no domain waits for one.
        void SetTransform(const dm::double3* translation, const dm::dquat* rotation, const dm::double3* scaling);
        void SetScaling(const dm::double3& scaling);
        void SetRotation(const dm::dquat& rotation);
        void SetTranslation(const dm::double3& translation);
        // One node per leaf. A leaf of a class with LeafTypeDesc::leafRecord that is on another node is refused with a
        // warning: take it from that node first (SetLeaf(nullptr) there), or clone it. Other leaves are taken as they
        // always were (importers put one light or camera on several nodes): the leaf then names the last node only.
        // When the world cannot make the node's new row, nothing changes; when it runs out of memory between the old
        // leaf's going and the new one's coming, the node has none. Both warn.
        void SetLeaf(SceneGraphLeaf* leaf);
        void SetName(const std::string& name);

        // Non-copyable and non-movable
        SceneGraphNode(const SceneGraphNode&) = delete;
        SceneGraphNode(const SceneGraphNode&&) = delete;
        SceneGraphNode& operator=(const SceneGraphNode&) = delete;
        SceneGraphNode& operator=(const SceneGraphNode&&) = delete;
    };

    // Operators of the deprecated DirtyFlags.
    inline SceneGraphNode::DirtyFlags operator | (SceneGraphNode::DirtyFlags a, SceneGraphNode::DirtyFlags b) { return SceneGraphNode::DirtyFlags(uint32_t(a) | uint32_t(b)); }
    inline SceneGraphNode::DirtyFlags operator & (SceneGraphNode::DirtyFlags a, SceneGraphNode::DirtyFlags b) { return SceneGraphNode::DirtyFlags(uint32_t(a) & uint32_t(b)); }
    inline SceneGraphNode::DirtyFlags operator ~ (SceneGraphNode::DirtyFlags a) { return SceneGraphNode::DirtyFlags(~uint32_t(a)); }
    inline SceneGraphNode::DirtyFlags operator |= (SceneGraphNode::DirtyFlags& a, SceneGraphNode::DirtyFlags b) { a = SceneGraphNode::DirtyFlags(uint32_t(a) | uint32_t(b)); return a; }
    inline SceneGraphNode::DirtyFlags operator &= (SceneGraphNode::DirtyFlags& a, SceneGraphNode::DirtyFlags b) { a = SceneGraphNode::DirtyFlags(uint32_t(a) & uint32_t(b)); return a; }
    inline bool operator !(SceneGraphNode::DirtyFlags a) { return uint32_t(a) == 0; }
    inline bool operator ==(SceneGraphNode::DirtyFlags a, uint32_t b) { return uint32_t(a) == b; }
    inline bool operator !=(SceneGraphNode::DirtyFlags a, uint32_t b) { return uint32_t(a) != b; }

    inline SceneContentFlags operator | (SceneContentFlags a, SceneContentFlags b) { return SceneContentFlags(uint32_t(a) | uint32_t(b)); }
    inline SceneContentFlags operator & (SceneContentFlags a, SceneContentFlags b) { return SceneContentFlags(uint32_t(a) & uint32_t(b)); }
    inline SceneContentFlags operator ~ (SceneContentFlags a) { return SceneContentFlags(~uint32_t(a)); }
    inline SceneContentFlags operator |= (SceneContentFlags& a, SceneContentFlags b) { a = SceneContentFlags(uint32_t(a) | uint32_t(b)); return a; }
    inline SceneContentFlags operator &= (SceneContentFlags& a, SceneContentFlags b) { a = SceneContentFlags(uint32_t(a) & uint32_t(b)); return a; }
    inline bool operator !(SceneContentFlags a) { return uint32_t(a) == 0; }
    inline bool operator ==(SceneContentFlags a, uint32_t b) { return uint32_t(a) == b; }
    inline bool operator !=(SceneContentFlags a, uint32_t b) { return uint32_t(a) != b; }

    // Scene graph traversal helper. Similar to an iterator, but only goes forward.
    // Create a SceneGraphWalker from a node, and it will go over every node in the sub-tree of that node.
    // On each location, the walker can move either down (deeper) or right (siblings), depending on the needs.
    // A walker created with SceneGraphWalker(current, scope) continues the pre-order traversal of 'scope' from
    // 'current': after the sub-tree of 'current' it moves on to the following siblings of 'current' and of its
    // ancestors, up to 'scope' (or the root when scope is null).
    // In an attached subtree the walker steps through the graph's rows and keeps a pointer to the graph's storage: the
    // graph must outlive the walk (destroying it while a walker is in it is not supported). Changes the old graph
    // allowed during a walk (transforms, attaching under the current node, detaching nodes the walk has left) are seen
    // at the next step, as before; the loop body must not detach the node it is visiting.
    class SceneGraphWalker final
    {
    private:
        SceneGraphNode* m_Current;
        SceneGraphNode* m_Scope;
        // While the walk is in a graph: its store and m_Current's entity, so that Next steps through rows (SceneLinks,
        // Parent) and the store's NodeOf index without reading the node objects. Transient walk state, not scene state.
        SceneGraphStore* m_Store = nullptr;
        Entity m_Entity = Entity::Null;
    public:
        SceneGraphWalker() = default;

        explicit SceneGraphWalker(SceneGraphNode* scope)
            : m_Current(scope)
            , m_Scope(scope)
        { }

        SceneGraphWalker(SceneGraphNode* current, SceneGraphNode* scope)
            : m_Current(current)
            , m_Scope(scope)
        { }

        [[nodiscard]] SceneGraphNode* Get() const { return m_Current; }
        [[nodiscard]] operator bool() const { return m_Current != nullptr; }
        SceneGraphNode* operator->() const { return m_Current; }
        
        // Moves the pointer to the first child of the current node, if it exists, and if allowChildren = true.
        // Otherwise, moves the pointer to the next sibling of the current node, if it exists.
        // Otherwise, goes up and tries to find the next sibiling up the hierarchy.
        // Returns the depth of the new node relative to the current node.
        int Next(bool allowChildren);

        // Moves the pointer to the parent of the current node, up to the scope.
        // Note that using Up and Next together may result in an infinite loop.
        // Returns the depth of the new node relative to the current node.
        int Up();
    };

    enum class AnimationAttribute : uint32_t
    {
        Undefined,
        Scaling,
        Rotation,
        Translation,
        LeafProperty
    };

    class SceneGraphAnimationChannel: public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SceneGraphAnimationChannel)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    private:
        nvrhi::AutoPtr<animation::Sampler> m_Sampler;
        nvrhi::WeakPtr<SceneGraphNode> m_TargetNode;
        nvrhi::WeakPtr<Material> m_TargetMaterial;
        AnimationAttribute m_Attribute;
        std::string m_LeafPropertyName;

    public:
        SceneGraphAnimationChannel(animation::Sampler* sampler, SceneGraphNode* targetNode, AnimationAttribute attribute)
            : m_Sampler(sampler)
            , m_TargetNode(targetNode)
            , m_Attribute(attribute)
        { }
        SceneGraphAnimationChannel(animation::Sampler* sampler, Material* targetMaterial)
            : m_Sampler(sampler)
            , m_TargetMaterial(targetMaterial)
            , m_Attribute(AnimationAttribute::LeafProperty)
        { }

        [[nodiscard]] bool IsValid() const;
        [[nodiscard]] animation::Sampler* GetSampler() const { return m_Sampler; }
        [[nodiscard]] AnimationAttribute GetAttribute() const { return m_Attribute; }
        [[nodiscard]] SceneGraphNode* GetTargetNode() const { return m_TargetNode.Lock(); }
        [[nodiscard]] const std::string& GetLeafPropertyName() const { return m_LeafPropertyName; }
        void SetTargetNode(SceneGraphNode* node) { m_TargetNode = node; }
        void SetLeafProperyName(const std::string& name) { m_LeafPropertyName = name; }
        bool Apply(float time) const;  // NOLINT(modernize-use-nodiscard)
    };

    NVRHI_CLASS_CLSID(SceneGraphAnimation, "e0f48aff-538f-4d8f-a39d-892c7a81bb74")
    class SceneGraphAnimation : public SceneGraphLeaf
    {
    public:
        NVRHI_DECLARE_UUID_TRAITS(SceneGraphAnimation)
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SceneGraphAnimation)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_IMPLEMENTS_CLASS(SceneGraphAnimation)
        NVRHI_IMPLEMENTS_ROUTE_PARENT(SceneGraphLeaf)
        NVRHI_END_INTERFACE_TABLE()
    private:
        std::vector<nvrhi::AutoPtr<SceneGraphAnimationChannel>> m_Channels;
        float m_Duration = 0.f;

    public:
       using SceneGraphLeaf::SceneGraphLeaf;

       [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
       [[nodiscard]] SceneContentFlags GetContentFlags() const override {
           return SceneContentFlags::Animations;
       }
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<SceneGraphAnimationChannel>>& GetChannels() const { return m_Channels; }
        [[nodiscard]] float GetDuration() const { return m_Duration; }
        [[nodiscard]] bool IsVald() const;
        bool Apply(float time) const;  // NOLINT(modernize-use-nodiscard)
        void AddChannel(SceneGraphAnimationChannel* channel);
    };

    // A container that tracks unique resources of the same type used by some entity, for example unique meshes used in a scene graph.
    // It works by putting the resource shared pointers into a map and associating a reference count with each resource.
    // When the resource is added and released an equal number of times, its refrence count reaches zero, and it's removed from the container.
    // It iterates in the order in which the resources were added (a resource that was removed and added again comes
    // last), which no address decides (design 3.3.6). Nothing may be added or removed during an iteration.
    template<typename T>
    class ResourceTracker
    {
    public:
        ResourceTracker() = default;
        ResourceTracker(const ResourceTracker&) = default;
        ResourceTracker& operator=(const ResourceTracker&) = default;
        // A moved-from tracker is empty.
        ResourceTracker(ResourceTracker&& other) noexcept
            : m_Map(std::move(other.m_Map))
            , m_First(std::exchange(other.m_First, nullptr))
            , m_Last(std::exchange(other.m_Last, nullptr))
        {
            other.m_Map.clear();
        }
        ResourceTracker& operator=(ResourceTracker&& other) noexcept
        {
            if (this != &other)
            {
                m_Map = std::move(other.m_Map);
                other.m_Map.clear();
                m_First = std::exchange(other.m_First, nullptr);
                m_Last = std::exchange(other.m_Last, nullptr);
            }
            return *this;
        }

    private:
        // Keyed by the raw pointer, so that a lookup makes no AutoPtr (two atomic operations); the entry holds the
        // reference. The entries form a list in the order in which they were added, linked by key, so that a copied
        // tracker is a tracker too.
        struct Entry
        {
            nvrhi::AutoPtr<T> resource;
            uint32_t count = 0;
            T* prev = nullptr;
            T* next = nullptr;
        };
        std::unordered_map<T*, Entry> m_Map;
        T* m_First = nullptr;
        T* m_Last = nullptr;

    public:
        class ConstIterator
        {
        private:
            const ResourceTracker* m_Tracker;
            T* m_Resource;   // null at the end
        public:
            ConstIterator(const ResourceTracker* tracker, T* resource) : m_Tracker(tracker), m_Resource(resource) {}
            ConstIterator& operator++() { m_Resource = m_Tracker->m_Map.find(m_Resource)->second.next; return *this; }
            ConstIterator operator++(int) { ConstIterator res = *this; ++*this; return res; }
            bool operator==(ConstIterator other) const { return m_Resource == other.m_Resource; }
            bool operator!=(ConstIterator other) const { return !(*this == other); }
            T* operator*() { return m_Resource; }
        };

        // Adds a reference to the specified resource.
        // Returns true if this is the first reference, i.e. if the resource has just been added to the tracker.
        bool AddRef(const T *resource)
        {
            if (!resource) return false;
            T* key = const_cast<T *>(resource);
            Entry& entry = m_Map[key];
            if (entry.count++ == 0)
            {
                entry.resource = key;
                entry.prev = m_Last;
                (m_Last ? m_Map.find(m_Last)->second.next : m_First) = key;
                m_Last = key;
            }
            return (entry.count == 1);
        }

        // Removes a reference from the specified resource.
        // Returns true if this was the last reference, i.e. if the resource has just been removed from the tracker.
        bool Release(const T* resource)
        {
            if (!resource) return false;
            auto it = m_Map.find(const_cast<T *>(resource));
            if (it == m_Map.end())
            {
                assert(false); // trying to release an object not owned by this tracker
                return false;
            }

            if (it->second.count == 0)
                assert(false); // zero-reference entries should not be possible; might indicate concurrency issues
            else
                --it->second.count;

            if (it->second.count == 0)
            {
                const Entry& entry = it->second;
                (entry.prev ? m_Map.find(entry.prev)->second.next : m_First) = entry.next;
                (entry.next ? m_Map.find(entry.next)->second.prev : m_Last) = entry.prev;
                m_Map.erase(it);
                return true;
            }
            return false;
        }

        [[nodiscard]] ConstIterator begin() const { return ConstIterator(this, m_First); }
        [[nodiscard]] ConstIterator end() const { return ConstIterator(this, nullptr); }
        [[nodiscard]] bool empty() const { return m_Map.empty(); }
        [[nodiscard]] size_t size() const { return m_Map.size(); }
    };

    template<typename T>
    using SceneResourceCallback = std::function<void(const T*)>;
    
    class SceneGraph : public nvrhi::WeakReferenceSourceImpl<nvrhi::IWeakReferenceSource>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SceneGraph)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IWeakReferenceSource)
        NVRHI_END_INTERFACE_TABLE()

    private:
        friend class SceneGraphNode;
        friend class SceneGraphStore;
        friend class GpuScene;
        nvrhi::AutoPtr<SceneGraphNode> m_Root;
        SceneGraphStore* m_Store = nullptr;   // the world, its systems and observers (one world per graph, D18)
        // The asset tables (design 4.8): stable slots and reference counts, filled by the leaf observer.
        MaterialTable m_Materials;
        MeshTable m_Meshes;
        size_t m_MaxGeometryCountPerMesh = 0;
        std::vector<nvrhi::AutoPtr<MeshInstance>> m_MeshInstances;
        std::vector<nvrhi::AutoPtr<SkinnedMeshInstance>> m_SkinnedMeshInstances;
        std::vector<nvrhi::AutoPtr<SceneGraphAnimation>> m_Animations;
        std::vector<nvrhi::AutoPtr<SceneCamera>> m_Cameras;
        std::vector<nvrhi::AutoPtr<Light>> m_Lights;
        bool m_DeferLeafRemoval = false;   // set while a Detach unregisters a subgraph: removal is one pass at its end
        std::vector<nvrhi::AutoPtr<SceneGraphLeaf>> m_RemovalMarked;   // held: compaction may drop their last entry

        void CompactLeafVectors();

    protected:
        // Called by the graph's observer on LeafObject (design 5.5), in creation and destruction order, while the
        // world changes structure: an override may read the graph and the leaf's node and write transforms of nodes
        // that have one, but structural calls (Attach, Detach, SetLeaf, a first SetTransform) are refused there with a
        // warning. An exception thrown from an override, or from OnMeshAdded and the other callbacks, is caught at
        // the observer, logged, and the leaf skipped (the observer is noexcept). During a Detach the vectors keep the
        // leaves unregistered before the current one until the Detach ends.
        // New code adds an IObserver instead (GetWorld()->AddObserver): OnAdd and OnRemove on the tag of the leaf class
        // (LeafTypeDesc::tags; RegisterComponent makes it known to the world), with LeafObject as a Read filter term,
        // so that the batch has the leaves. It sees the leaves of that class and its subclasses in the order of these
        // calls, right after the graph's own observer, with complete rows (the columns of a class with leafRecord
        // hold the leaf's values), at Attach, Detach and SetLeaf, between two leaves of one class too. Exceptions: a
        // Sparse or Shared tag changes apart from the row in SetLeaf, where an observer on it with the LeafObject term
        // sees nothing (use archetype tags); and when a callback of a removal attaches the leaf to another graph, the
        // leaf takes its values along, so that the later observers of that removal see default ones. What an observer
        // returns undoes nothing (a warning). When a graph dies nothing is delivered and the world may outlive it, so a
        // graph class that adds observers removes them in its own destructor.
        virtual void RegisterLeaf(SceneGraphLeaf *leaf);
        virtual void UnregisterLeaf(SceneGraphLeaf *leaf);

    public:
       SceneGraph();
       ~SceneGraph() override;

       SceneResourceCallback<MeshInfo> OnMeshAdded;
       SceneResourceCallback<MeshInfo> OnMeshRemoved;
       SceneResourceCallback<Material> OnMaterialAdded;
       SceneResourceCallback<Material> OnMaterialRemoved;

       [[nodiscard]] SceneGraphNode* GetRootNode() { return m_Root; }
        // The asset tables, iterated in slot order (Material::materialID, MeshInfo::globalMeshIndex); their size is the
        // number of materials and meshes in use.
        [[nodiscard]] const MaterialTable& GetMaterials() const { return m_Materials; }
        [[nodiscard]] const MeshTable& GetMeshes() const { return m_Meshes; }
        // The size of an array indexed by MeshGeometry::globalGeometryIndex: one past the highest geometry slot used
        // (the number of geometries while no mesh has left the graph).
        [[nodiscard]] const size_t GetGeometryCount() const { return m_Meshes.GetGeometryEnd(); }
        [[nodiscard]] const size_t GetMaxGeometryCountPerMesh() const { return m_MaxGeometryCountPerMesh; }
        // The size of an array indexed by geometry-instance slot (MeshInstance::GetGeometryInstanceIndex), likewise.
        [[nodiscard]] const size_t GetGeometryInstancesCount() const;
        // The size of an array indexed by instance slot (MeshInstance::GetInstanceIndex), likewise.
        [[nodiscard]] size_t GetInstanceSlotCount() const;
        // The leaves: runtime-only mesh instances (RuntimeMeshInstance, SceneComponents.h) are not in it.
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<MeshInstance>>& GetMeshInstances() const { return m_MeshInstances; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<SkinnedMeshInstance>>& GetSkinnedMeshInstances() const { return m_SkinnedMeshInstances; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<SceneGraphAnimation>>& GetAnimations() const { return m_Animations; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<SceneCamera>>& GetCameras() const { return m_Cameras; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<Light>>& GetLights() const { return m_Lights; }
        // Attach, Detach, SetRootNode or SetLeaf since the last Refresh.
        [[nodiscard]] bool HasPendingStructureChanges() const;
        // A transform set (or nodes with transforms attached) since the last Refresh, or the last Refresh moved
        // something: then the next one is still pending for the previous-frame transforms. Only edits through this API
        // count: what a simulation moves (the driven nodes and what hangs below them) sets it neither before nor after
        // the Refresh that propagates it.
        [[nodiscard]] bool HasPendingTransformChanges() const;
        // A transform or structural edit through this API (the node's setters and SetLeaf; Attach, AttachLeafNode,
        // Detach, SetRootNode) that no run of transform propagation has processed: the world transforms are older than
        // the edits. The edits of HasPendingStructureChanges and HasPendingTransformChanges, but any run of propagation
        // clears it (Refresh, Simulate, the TransformPropagation phase run by hand), and it is not set in the frame
        // after a move. Edits of nodes in no graph, InvalidateContent, writes through GetWorld() and teleport requests
        // on a driven node do not set it; SetSimulationDriven(node, false) does. Read it on the thread that uses the
        // API, or in a system of a phase other than TransformPropagation.
        [[nodiscard]] bool HasStaleWorldTransforms() const;

        // One world per graph (D18). Nodes are entities of it while they are attached.
        [[nodiscard]] IWorld* GetWorld() const;
        [[nodiscard]] Entity GetEntity(const SceneGraphNode* node) const;   // Null when not attached to this graph
        [[nodiscard]] SceneGraphNode* GetNode(Entity entity) const;          // null for runtime-only entities; no AddRef
        // The scheduler that runs the TransformPropagation and Bounds phases in Refresh, on the graph's job system.
        [[nodiscard]] IScheduler* GetScheduler() const;
        // The job system the phases run on: by default one that the process shares (hardware threads minus one
        // workers), which is never released. In an executable, an atexit handler stops and joins its workers before
        // the C runtime tears down (the jobs in flight finish first, for 5 s at most; workers still busy then are
        // parked until the process ends; afterwards Refresh runs on the calling thread). On Windows a DLL's exit
        // handlers run under the loader lock, where no thread can be joined, so none is registered there: a DLL that
        // links donut_engine and is unloaded gives its graphs a job system of its own and releases it first.
        // Elsewhere a shared object registers the handler too, and it runs at dlclose (glibc runs a shared object's
        // atexit handlers then); a dlclose from inside a job is not supported. Set it between frames; null returns
        // to the shared one. Refused with a warning, nothing changed, inside a phase or a leaf callback, while the
        // graph's scheduler has simulation domains (the new one would not have them), and when the new scheduler or one
        // of the graph's phases and systems cannot be made: the graph keeps the scheduler and job system it has.
        void SetJobSystem(IJobSystem* jobSystem);

        // The render service's isolation guard (plan step E5, sub-plan item E5.1, open point 12), module-private. Between
        // BeginRenderRead and EndRenderRead (RenderService::Extract and EndFrame) a frame's snapshot points at this
        // graph's leaves and assets: in Debug builds every write through this API (structural calls, transforms,
        // SetLeaf, InvalidateContent) logs a warning there, beside the phase check of design 5.6, and still applies.
        // Release builds check nothing. Scopes nest.
        // RenderReadMode::Defer (plan step E7, item E7.8; RenderServiceDesc::pipelined) is the pipelined frame's scope:
        // writes are allowed and do not warn, and the last reference that the graph, its world or its tables drop to a
        // leaf, node, mesh or material while a scope of this mode is open goes to a retire list that EndRenderRead
        // releases, on the thread that calls it. Mixed scopes: while any Defer scope is open the graph defers, and writes
        // do not warn, also inside a Warn scope nested in it; the EndRenderRead that closes the last open Defer scope
        // releases the list. Scopes close in reverse order of opening. What is kept: a leaf that leaves the world (the
        // leaf observer's OnRemove: Detach, SetLeaf, a row destroyed through the world), each node of a detached
        // subgraph, and every mesh and material the asset tables remove; each one reference, released in that order.
        // Refresh's reconciliation of meshes edited in place (their geometries' and materials' slot fields, the mesh
        // table's ranges) waits for that EndRenderRead too, which runs it first and raises its asset callbacks. The
        // application's own objects (views, its shadow maps) and its in-place edits of assets are its own: it keeps
        // them, and orders such edits after the pipelined frame's join.
        enum class RenderReadMode : uint8_t
        {
            Warn,    // the serial frame's isolation guard, as above
            Defer,   // the pipelined frame's: deferred releases
        };
        void BeginRenderRead(RenderReadMode mode = RenderReadMode::Warn) noexcept;
        void EndRenderRead() noexcept;
        [[nodiscard]] bool IsInRenderRead() const noexcept;
        // A Defer scope is open on this graph: a pipelined frame records. Any thread may ask.
        [[nodiscard]] bool IsRenderReadDeferring() const noexcept;
        // Any graph of the process has a Defer scope open: a pipelined frame is open, from RenderService::Extract to
        // the EndFrame after its join (plan step E7, part I7). The walking draw strategies refuse while it holds, on any
        // thread and whatever graph their root node is in (fail-closed: a detached root has none to ask).
        [[nodiscard]] static bool IsAnyRenderReadDeferring() noexcept;
        // Runs a phase of GetScheduler() whose systems only read (the render service's RenderExtract, plan step E5.1),
        // and accounts for it: when nothing else ran on the world since the last Refresh, the version the phase took is
        // not a change, so the next Refresh still returns at once when nothing changed and the on-demand caches stay. A
        // phase with an enabled system that declares a write or is exclusive, or that played back commands or changed
        // structure, counts like any other phase (the next Refresh runs); Debug builds warn about such a system.
        // FE_INVALID_ARGS without a scheduler; otherwise what IScheduler::RunPhase returns. Only declared accesses,
        // commands and structural changes are seen: a system that declares reads only but writes through
        // IWorld::GetComponentForWrite is not (the world's validation mode reports such writes in Debug builds). A
        // read hand-out of a RowTracked column may write that column's row versions (the chunk-only writer rule of
        // ChunkView: rows stamped with the chunk's version, which they lacked); that records a change already made,
        // changes no value and no chunk version, and is not counted as a write (backlog I4-N4).
        nvrhi::FRESULT RunReadOnlyPhase(Phase phase) noexcept;
        // The render service's registration (one service per graph, plan step E5.1): ClaimRenderService returns false
        // when a service has claimed the graph already; ReleaseRenderService gives the claim back.
        bool ClaimRenderService() noexcept;
        void ReleaseRenderService() noexcept;
        // Counts the schedulers the graph made: SetJobSystem replaces GetScheduler() (and drops the systems on it), so a
        // caller that registered systems compares this instead of holding the scheduler.
        [[nodiscard]] uint64_t GetSchedulerGeneration() const noexcept;

        // RowTracked scene columns (plan step E7, item E7.5; design D9): a process-wide switch, off by default, that
        // stays off until measured (sub-plan revision 2, stage B). Graphs constructed while it is on register
        // LocalTransform and WorldTransform with ComponentFlags::RowTracked: the node setters, transform propagation
        // and the interpolation stamp the rows they write, propagation composes only the rows whose local transform
        // or parent changed, and GpuScene takes only the changed rows of a chunk. A graph keeps what it was made
        // with; the switch changes no output (snapshots, hashes, rows per chunk, uploaded rows), only the work. Set
        // it between graphs, on any thread; an A/B run makes one graph with it off and one with it on. It is a relaxed
        // atomic, read once by the graph's constructor: a graph made on another thread while it is set takes either
        // value, and keeps that one (backlog I4-N5).
        static void SetRowTrackedTransforms(bool enabled) noexcept;
        [[nodiscard]] static bool GetRowTrackedTransforms() noexcept;

        // The Static tag (SceneComponents.h; design 4.7; plan step E7, item E7.4, remedy 3): a process-wide switch, off
        // by default, that stays off until measured (sub-plan revision 2, stage B), read once by the graph's constructor
        // like SetRowTrackedTransforms; a graph keeps what it was made with. In a graph made while it is on, an archetype
        // of depth below c_MaxHierarchyDepth that carries Static, and that is not simulation-driven, leaves the
        // TransformPropagation work after a run that took it, and so does one that has no dynamic-bounds leaves the
        // Bounds work: a later run takes it again only when rows joined or left it (Bounds also after a structural
        // change through the API, SetLeaf or InvalidateContent, and after a propagation run that took it), and then
        // computes the chunks that rows joined, whole, from their local transforms and their parents' worlds at that
        // time. So every row that shares a chunk with a row that joined, or that a row's leaving moved (the swap-remove
        // of its hole), is composed again then, from its LocalTransform and its parent's current world, and a write of
        // its LocalTransform since the last run is taken. Otherwise their world transforms do not follow later writes
        // of their LocalTransform, nor moves of their parents; the rows below them are composed from those worlds, by
        // propagation and where the graph composes a world itself (SetSimulationDriven, teleports in parent space, the
        // local getters of driven nodes). A write after the run that took the row's chunk is refused in Debug builds,
        // with a warning: SceneGraphNode::SetTransform does not write; a write through the world is found by the next
        // propagation, which does not propagate it. Release builds do not check, and do not propagate it either. A
        // write before that run (the row joined since) is its initial value. The first transform of a Static node that
        // has none adds its LocalTransform, a structural change: the row joins another archetype and is taken as above,
        // in both builds.
        static void SetStaticTag(bool enabled) noexcept;
        [[nodiscard]] static bool GetStaticTag() noexcept;

        // The Bounds phase's path for chunks without children (plan step E7, item E7.4, remedy 4): a chunk whose rows
        // have no children and one parent (its subgraph summary says so; it changes only with a full subgraph pass) takes
        // its subgraph step from its summary and its new box, with one mark for the parent, instead of reading every
        // row's links, parent and content. A process-wide switch, off by default, that stays off until measured
        // (stage B), read at every run of the Bounds system; it changes no output (subgraph values, marks, summaries),
        // only the work. Set it on any thread.
        static void SetChildlessBoundsPath(bool enabled) noexcept;
        [[nodiscard]] static bool GetChildlessBoundsPath() noexcept;

        // Replaces the current root node of the graph with the new one.
        nvrhi::AutoPtr<SceneGraphNode> SetRootNode(SceneGraphNode *root);
        
        // Attaches a node and its subgraph to the parent.
        // If the node is already attached to this or other graph, a deep copy of the subgraph is made first.
        nvrhi::AutoPtr<SceneGraphNode> Attach(SceneGraphNode *parent, SceneGraphNode* child);
        
        // Creates a node holding the provided leaf and attaches it to the parent.
        nvrhi::AutoPtr<SceneGraphNode> AttachLeafNode(SceneGraphNode* parent,SceneGraphLeaf* leaf);
        
        // Removes the node and its subgraph from the graph.
        // The order of the node's remaining siblings is always preserved; preserveOrder is kept for source
        // compatibility and has no effect.
        nvrhi::AutoPtr<SceneGraphNode> Detach(SceneGraphNode* node, bool preserveOrder = false);

        // Finds a node whose path (sequence of nested node names) matches the provided path,
        // relative to the 'context' node or the root if 'context' is NULL.
        // If the path starts with / the search starts at the root, and the 'context' parameter is ignored.
        // Parent references with .. are supported.
        // If multiple nodes within one parent have the same name matching that component of the path, only the first node will be considered.
        [[nodiscard]] SceneGraphNode* FindNode(const std::filesystem::path& path, SceneGraphNode* context = nullptr) const;
        
        // Runs the TransformPropagation and Bounds phases (design 5.7) when anything changed since the last call. Slots
        // are stable (step E4): they are assigned when leaves are attached, not here; after structural changes it only
        // brings the mesh table in line with meshes whose geometries were edited in place (MeshTable::Reconcile).
        // "Changed" means: a call of this API (structure, transforms, InvalidateContent), a structural change of the
        // world, or a phase that ran on the world (its version moved). Writes into the graph's columns through
        // GetWorld() outside phases (GetComponentForWrite with c_WorldVersion) do not move the version and are not
        // seen: make them in a system of GetScheduler()'s phases, or through the node API. When the phases cannot
        // run (inside a phase), the changes stay pending for the next call. It also computes the subgraph bounds and
        // content flags (GetGlobalBoundingBox, GetSubgraphContentFlags) of the nodes with children whose subtree
        // changed. No other thread may read the graph while it runs; readers after it returned see its results.
        // Leaves with dynamic bounds (LeafTypeDesc::dynamicBounds) make every call run the Bounds phase.
        void Refresh(uint32_t frameIndex);

        // One simulation frame (design 8.4), outside phases: transform propagation when HasStaleWorldTransforms(), so
        // that kinematic inputs are this frame's; IScheduler::SetFrameTime(time); the phases BeginFrame to
        // PostSimulation; the interpolation of simulation-driven entities. The caller gives the time; nothing here
        // reads a clock. A Refresh after it is needed when a domain drives entities or the frame uses bounds that the
        // simulation changed; a Refresh before it is allowed. What HasPendingStructureChanges and
        // HasPendingTransformChanges report stays pending. A graph that never calls it runs no simulation.
        // Teleport requests (SceneGraphNode::SetTransform on a driven node) are resolved by a system of a named phase
        // "SimulationTeleports" right after BeginFrame, over the simulated poses as they are then: it sees what the
        // systems of BeginFrame wrote. A phase that the application adds after BeginFrame runs before it
        // (IScheduler::AddPhase puts a new phase right after the one it names), so it is seen too; a request made in
        // PreSimulation or later waits for the next frame's Simulate.
        // FE_INVALID_ARGS when the phases cannot run: inside a phase or a callback of the world, without a scheduler, or
        // with an ordering cycle. With a cycle in one of the phases BeginFrame to PostSimulation none of them runs, but
        // the time has been taken: the steps are taken once the phases run again, as far as maxStepsPerFrame allows. When
        // propagation is due and cannot run, nothing is done. The failures of systems and domains are not returned: the
        // frame goes on, and GetScheduler()->GetDiagnostics has them.
        nvrhi::FRESULT Simulate(const FrameTime& time) noexcept;
        // Binds a node of this graph to the simulation, or unbinds it (SceneComponents.h, simulation-driven entities).
        // Structural: outside phases and leaf callbacks. Binding takes both poses and the scale from the node's current
        // world transform: composed from its local transform and its ancestors' as they are now, propagated or not, and
        // below a driven ancestor over that one's simulated pose (SimPose), so that the engine's bodies are of one time
        // and binding and unbinding on a moving parent gives the local transform back. The node is shown there at once:
        // below a driven ancestor that the interpolation shows behind its pose, that is a jump of at most one fixed step
        // of the ancestor's motion. Shear and mirroring are refused with a warning; the domain whose lifetime query then
        // matches the entity adds it. While bound, the transform setters are teleport requests (SceneGraphNode::
        // SetTransform), and the local getters give the last simulated pose, in the space of the parent's simulated
        // world, not the place that is shown: whenever the interpolation shows the node between two poses (alpha below
        // 1), after Refresh too and under an unmoved parent, they differ from what GetLocalToWorldTransform gives, by at
        // most the motion of a fixed step. They cost O(depth) (about 600 ns below 8 levels, against 90 ns of a getter of
        // an authored node). Unbinding makes the same parts the local transform, and so does Detach (with a warning where
        // their matrix is not the pose: a parent that shears it, or has no inverse): a node that is attached again or
        // copied is authored. Binding a bound node, or unbinding another, does nothing. FE_INVALID_ARGS for a node that
        // is not attached to this graph, a refused transform (also a scale of 0, or numbers that are not finite), or a
        // wrong time; also, or FE_OUT_OF_MEMORY, when the world cannot make the node's new row (it fits no chunk; no
        // memory): a warning, and nothing changed. A failing observer or domain callback is a warning only: the call
        // stands.
        nvrhi::FRESULT SetSimulationDriven(SceneGraphNode* node, bool driven) noexcept;

        // Runtime-only mesh instances (design 5.8; plan step E7, item E7.1; RuntimeMeshInstance in SceneComponents.h).
        // Rows that the application or its systems create in GetWorld() with the components that
        // GetRuntimeMeshInstanceComponents lists, through IWorld::CreateEntities, Instantiate or command buffers, at sync
        // points or outside phases (rule 3.3.3). They are roots, render through the render service's packets and
        // simulate like a node's rows, and are invisible to SceneGraphWalker, GetMeshInstances, GetNode (null) and
        // the root's subgraph bounds. Only rows made in this graph's world: rows copied from another world
        // (IWorld::CopyEntitiesFrom) carry ids of another table. A Parent on such a row is not supported (a warning;
        // the row stays a root). A row made without GpuInstanceSlot (as a snapshot's are: it is not saved) gets it
        // through the graph's observer commands; one that the same playback destroys first makes that command fail:
        // nothing is held or leaked, the call still succeeds, and a world with validation (Debug) logs a warning per row.
        //
        // A reference to `mesh` in the graph's mesh table, held for the caller until ReleaseRuntimeMesh; raises
        // OnMeshAdded for a mesh new to the graph. Its id is what MeshRef of the caller's rows names: the graph's
        // observer takes one more reference per row while the row lives, so the caller may release its own once the
        // rows exist; a LoadSnapshot of the world re-adds rows by id, so a rewind needs the caller's reference to be
        // held across it. Outside phases and leaf callbacks; MeshId::Invalid, with a warning, for a null mesh, inside
        // a phase or out of memory.
        MeshId AcquireRuntimeMesh(MeshInfo* mesh) noexcept;
        // Returns a reference of AcquireRuntimeMesh; the last one of the mesh raises OnMeshRemoved. Outside phases and
        // leaf callbacks; a warning for an id that the caller holds no reference to.
        void ReleaseRuntimeMesh(MeshId mesh) noexcept;
        // The components of a runtime-only mesh instance (SceneComponents.h), with the simulation-driven ones when
        // `simulationDriven`, as ids of GetWorld() for EntityCreateDesc and command buffers. Writes at most `capacity`
        // ids to `out` (null with capacity 0 to ask) and returns how many there are; at most
        // c_MaxRuntimeMeshInstanceComponents.
        static constexpr uint32_t c_MaxRuntimeMeshInstanceComponents = 14;
        uint32_t GetRuntimeMeshInstanceComponents(ComponentId* out, uint32_t capacity,
                                                  bool simulationDriven = false) const noexcept;
        // The runtime-only mesh instances in the world that hold slots (the observer took them), for statistics.
        [[nodiscard]] uint32_t GetRuntimeMeshInstanceCount() const noexcept;
    };

    struct SceneImportResult
    {
        nvrhi::AutoPtr<SceneGraphNode> rootNode;
    };

    class SceneTypeFactory: public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(SceneTypeFactory)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        virtual ~SceneTypeFactory() = default;
        virtual nvrhi::AutoPtr<SceneGraph> CreateGraph();
        virtual nvrhi::AutoPtr<SceneGraphLeaf> CreateLeaf(const std::string& type);
        virtual nvrhi::AutoPtr<Material> CreateMaterial();
        virtual nvrhi::AutoPtr<MeshInfo> CreateMesh();
        virtual nvrhi::AutoPtr<MeshGeometry> CreateMeshGeometry();
        virtual nvrhi::AutoPtr<MeshInstance> CreateMeshInstance(MeshInfo* mesh);
        virtual nvrhi::AutoPtr<SkinnedMeshInstance> CreateSkinnedMeshInstance(SceneTypeFactory *sceneTypeFactory, MeshInfo *prototypeMesh);
    };

    void PrintSceneGraph(const SceneGraphNode* root);
}
