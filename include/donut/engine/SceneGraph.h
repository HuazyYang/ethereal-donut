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
    struct IWorld;
    struct IJobSystem;
    struct IScheduler;

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

    protected:

    public:
        using WeakReferenceSourceImpl::WeakReferenceSourceImpl;
        virtual ~SceneGraphLeaf() = default;

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
    private:
        friend class SceneGraph;
        int m_InstanceIndex = -1;
        int m_GeometryInstanceIndex = -1;

    protected:
        nvrhi::AutoPtr<MeshInfo> m_Mesh;

    public:
        explicit MeshInstance(MeshInfo* mesh)
            : m_Mesh(mesh)
        { }

        [[nodiscard]] MeshInfo* GetMesh() const { return m_Mesh; }
        [[nodiscard]] int GetInstanceIndex() const { return m_InstanceIndex; }
        [[nodiscard]] int GetGeometryInstanceIndex() const { return m_GeometryInstanceIndex; }
        [[nodiscard]] dm::box3 GetLocalBoundingBox() override { return m_Mesh->objectSpaceBounds; }
        [[nodiscard]] nvrhi::AutoPtr<SceneGraphLeaf> Clone() override;
        [[nodiscard]] const LeafTypeDesc& GetTypeDesc() const override;
        [[nodiscard]] SceneContentFlags GetContentFlags() const override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
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

        // Derived state computed on demand (5.7): the subgraph's bounds and content flags and the leaf's content
        // flags, valid while m_CacheEpoch equals the graph's epoch. Published with release order, so that threads that
        // walk the graph at the same time (views rendered in parallel) may fill and read it.
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

        void SetTransform(const dm::double3* translation, const dm::dquat* rotation, const dm::double3* scaling);
        void SetScaling(const dm::double3& scaling);
        void SetRotation(const dm::dquat& rotation);
        void SetTranslation(const dm::double3& translation);
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
    class SceneGraphWalker final
    {
    private:
        SceneGraphNode* m_Current;
        SceneGraphNode* m_Scope;
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
    template<typename T>
    class ResourceTracker
    {
    private:
        // Keyed by the raw pointer, so that a lookup makes no AutoPtr (two atomic operations); the entry holds the
        // reference. The hash of a pointer equals the hash of its AutoPtr, so the iteration order is unchanged.
        struct Entry
        {
            nvrhi::AutoPtr<T> resource;
            uint32_t count = 0;
        };
        std::unordered_map<T*, Entry> m_Map;
        using UnderlyingConstIterator = typename std::unordered_map<T*, Entry>::const_iterator;

    public:
        class ConstIterator
        {
        private:
            UnderlyingConstIterator m_Iter;
        public:
            ConstIterator(UnderlyingConstIterator iter) : m_Iter(std::move(iter)) {}
            ConstIterator& operator++() { ++m_Iter; return *this; }
            ConstIterator operator++(int) { ConstIterator res = *this; ++m_Iter; return res; }
            bool operator==(ConstIterator other) const { return m_Iter == other.m_Iter; }
            bool operator!=(ConstIterator other) const { return !(*this == other); }
            T* operator*() { return m_Iter->first; }
        };

        // Adds a reference to the specified resource.
        // Returns true if this is the first reference, i.e. if the resource has just been added to the tracker.
        bool AddRef(const T *resource)
        {
            if (!resource) return false;
            Entry& entry = m_Map[const_cast<T *>(resource)];
            if (entry.count++ == 0)
                entry.resource = const_cast<T *>(resource);
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
                m_Map.erase(it);
                return true;
            }
            return false;
        }

        [[nodiscard]] ConstIterator begin() const { return ConstIterator(m_Map.cbegin()); }
        [[nodiscard]] ConstIterator end() const { return ConstIterator(m_Map.cend()); }
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
        nvrhi::AutoPtr<SceneGraphNode> m_Root;
        SceneGraphStore* m_Store = nullptr;   // the world, its systems and observers (one world per graph, D18)
        ResourceTracker<Material> m_Materials;
        ResourceTracker<MeshInfo> m_Meshes;
        size_t m_GeometryCount = 0;
        size_t m_MaxGeometryCountPerMesh = 0;
        size_t m_GeometryInstancesCount = 0;
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
       [[nodiscard]] const ResourceTracker<Material>& GetMaterials() const {
           return m_Materials;
       }
        [[nodiscard]] const ResourceTracker<MeshInfo>& GetMeshes() const { return m_Meshes; }
        [[nodiscard]] const size_t GetGeometryCount() const { return m_GeometryCount; }
        [[nodiscard]] const size_t GetMaxGeometryCountPerMesh() const { return m_MaxGeometryCountPerMesh; }
        [[nodiscard]] const size_t GetGeometryInstancesCount() const { return m_GeometryInstancesCount; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<MeshInstance>>& GetMeshInstances() const { return m_MeshInstances; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<SkinnedMeshInstance>>& GetSkinnedMeshInstances() const { return m_SkinnedMeshInstances; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<SceneGraphAnimation>>& GetAnimations() const { return m_Animations; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<SceneCamera>>& GetCameras() const { return m_Cameras; }
        [[nodiscard]] const std::vector<nvrhi::AutoPtr<Light>>& GetLights() const { return m_Lights; }
        // Attach, Detach, SetRootNode or SetLeaf since the last Refresh.
        [[nodiscard]] bool HasPendingStructureChanges() const;
        // A transform set (or nodes with transforms attached) since the last Refresh, or the last Refresh moved
        // something: then the next one is still pending for the previous-frame transforms.
        [[nodiscard]] bool HasPendingTransformChanges() const;

        // One world per graph (D18). Nodes are entities of it while they are attached.
        [[nodiscard]] IWorld* GetWorld() const;
        [[nodiscard]] Entity GetEntity(const SceneGraphNode* node) const;   // Null when not attached to this graph
        [[nodiscard]] SceneGraphNode* GetNode(Entity entity) const;          // null for runtime-only entities; no AddRef
        // The scheduler that runs the TransformPropagation and Bounds phases in Refresh, on the graph's job system.
        [[nodiscard]] IScheduler* GetScheduler() const;
        // The job system the phases run on: by default one that the process shares (hardware threads minus one
        // workers), which is never released (its workers cannot be joined during static destruction): a DLL that
        // links donut_engine and is unloaded gives its graphs a job system of its own and releases it first. Set it
        // between frames; null returns to the shared one.
        void SetJobSystem(IJobSystem* jobSystem);

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
        
        // Runs the TransformPropagation and Bounds phases (design 5.7) when anything changed since the last call, and
        // numbers the mesh instances, geometries, meshes and materials after structural changes (until step E4).
        // "Changed" means: a call of this API (structure, transforms, InvalidateContent), a structural change of the
        // world, or a phase that ran on the world (its version moved). Writes into the graph's columns through
        // GetWorld() outside phases (GetComponentForWrite with c_WorldVersion) do not move the version and are not
        // seen: make them in a system of GetScheduler()'s phases, or through the node API. When the phases cannot
        // run (inside a phase), the changes stay pending for the next call.
        void Refresh(uint32_t frameIndex);
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
