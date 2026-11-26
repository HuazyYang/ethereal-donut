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

#include <donut/engine/SceneGraph.h>
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <xmmintrin.h>
#endif
#include <donut/core/log.h>
#include <donut/core/json.h>
#include <sstream>
#include <donut/engine/ShadowMap.h>

#include "SceneGraphStore.h"

#include <algorithm>
#include <cassert>

using namespace donut::engine;

// The render service's isolation guard (SceneGraph::BeginRenderRead; plan step E5, sub-plan item E5.1): Debug builds
// warn once per write through this API while a frame records; Release builds compile the check out.
#ifndef NDEBUG
#define DONUT_SCENE_GRAPH_WARN_RENDER_READ(store, operation) (store)->WarnRenderRead(operation)
#define DONUT_SCENE_GRAPH_RENDER_READ_QUIET(store) SceneGraphStore::RenderReadQuiet renderReadQuiet(*(store))
#else
#define DONUT_SCENE_GRAPH_WARN_RENDER_READ(store, operation) ((void)0)
#define DONUT_SCENE_GRAPH_RENDER_READ_QUIET(store) ((void)0)
#endif

SceneGraphLeaf::~SceneGraphLeaf()
{
    if (m_LeafRecord)
        LeafRecord::Destroy(m_LeafRecord);
}

const void* SceneGraphLeaf::GetColumn(const ComponentTypeDesc& type) const noexcept
{
    return SceneGraphStore::LeafColumn(this, type, false);
}

void* SceneGraphLeaf::EditColumn(const ComponentTypeDesc& type) noexcept
{
    if (!m_LeafRecord)
    {
        if (const SceneGraphStore* store = SceneGraphStore::StoreOf(this))
        {
            assert(!store->GetWorld()->IsInPhase());   // design 5.6: the API writes outside phases
            DONUT_SCENE_GRAPH_WARN_RENDER_READ(store, "Leaf::Edit");
            (void)store;
        }
        else
        {
            // in no graph: the first Edit of one of the class's columns makes the record
            const LeafTypeDesc& leafType = GetTypeDesc();
            if (!leafType.leafRecord || !SceneGraphStore::DeclaresColumn(leafType, type.uuid))
                return nullptr;
            m_LeafRecord = LeafRecord::Create(leafType);
            if (!m_LeafRecord)
                return nullptr;
        }
    }
    return SceneGraphStore::LeafColumn(this, type, true);
}

const std::string& SceneGraphLeaf::GetName() const
{
    auto node = GetNode();
    if (node)
        return node->GetName();

    static const std::string emptyString = "";
    return emptyString;
}

void SceneGraphLeaf::SetName(const std::string& name) const
{
    auto node = GetNode();
    if (node)
        node->SetName(name);
    else
        assert(!"The leaf must be attached in order to set its name");
}

SkinnedMeshInstance::SkinnedMeshInstance(SceneTypeFactory* sceneTypeFactory, MeshInfo *prototypeMesh)
    : MeshInstance(nullptr)
    , m_SceneTypeFactory(sceneTypeFactory)
{
    m_PrototypeMesh = std::move(prototypeMesh);
    
    auto skinnedMesh = m_SceneTypeFactory->CreateMesh();
    skinnedMesh->skinPrototype = m_PrototypeMesh;
    skinnedMesh->name = m_PrototypeMesh->name;
    skinnedMesh->objectSpaceBounds = m_PrototypeMesh->objectSpaceBounds;
    skinnedMesh->indexOffset = m_PrototypeMesh->indexOffset;
    skinnedMesh->totalVertices = m_PrototypeMesh->totalVertices;
    skinnedMesh->totalIndices = m_PrototypeMesh->totalIndices;
    skinnedMesh->geometries.reserve(m_PrototypeMesh->geometries.size());
    
    for (const auto& geometry : m_PrototypeMesh->geometries)
    {
        auto newGeometry = m_SceneTypeFactory->CreateMeshGeometry();
        *newGeometry = *geometry;
        skinnedMesh->geometries.push_back(newGeometry);
    }

    m_Mesh = skinnedMesh;
}

nvrhi::AutoPtr<SceneGraphLeaf> SkinnedMeshInstance::Clone()
{
    auto copy = MAKE_RC_OBJ_PTR(SkinnedMeshInstance, m_SceneTypeFactory, m_PrototypeMesh);

    for (const auto& joint : joints)
    {
        copy->joints.push_back(joint);
    }

    return copy;
}

nvrhi::AutoPtr<SceneGraphLeaf> SkinnedMeshReference::Clone()
{
    return MAKE_RC_OBJ_PTR(SkinnedMeshReference, m_Instance.Lock());
}

nvrhi::AutoPtr<SceneGraphLeaf> MeshInstance::Clone()
{
    return MAKE_RC_OBJ_PTR(MeshInstance, m_Mesh);
}

SceneContentFlags MeshInstance::GetContentFlags() const
{
    // one home for the rule: runtime-only mesh instances take their content flags from it too (SceneGraphStore.cpp)
    return GetMeshContentFlags(m_Mesh);
}

bool MeshInstance::SetProperty(const std::string& name, const dm::float4& value)
{
    if (m_Mesh && m_Mesh->geometries.size() == 1 && m_Mesh->geometries[0]->material) // TODO: what if we want to target a geometry other than 0?
        return m_Mesh->geometries[0]->material->SetProperty(name, value);

    return SceneGraphLeaf::SetProperty(name, value);
}

dm::affine3 SceneCamera::GetViewToWorldMatrix() const
{
    auto node = GetNode();
    if (!node)
        return dm::affine3::identity();

    return dm::scaling(dm::float3(1.f, 1.f, -1.f)) * dm::affine3(node->GetLocalToWorldTransform());
}

dm::affine3 SceneCamera::GetWorldToViewMatrix() const
{
    auto node = GetNode();
    if (!node)
        return dm::affine3::identity();

    return dm::affine3(inverse(node->GetLocalToWorldTransform())) * dm::scaling(dm::float3(1.f, 1.f, -1.f));
}

nvrhi::AutoPtr<SceneGraphLeaf> PerspectiveCamera::Clone()
{
    auto copy = MAKE_RC_OBJ_PTR(PerspectiveCamera);
    copy->zNear = zNear;
    copy->zFar = zFar;
    copy->verticalFov = verticalFov;
    copy->aspectRatio = aspectRatio;
    return copy;
}

void PerspectiveCamera::Load(const Json::Value& node)
{
    node["verticalFov"] >> verticalFov;
    node["aspectRatio"] >> aspectRatio;
    node["zNear"] >> zNear;
    node["zFar"] >> zFar;
}

bool PerspectiveCamera::SetProperty(const std::string& name, const dm::float4& value)
{
    if (name == "zNear")
    {
        zNear = value.x;
        return true;
    }

    if (name == "zFar")
    {
        zFar = value.x;
        return true;
    }

    if (name == "verticalFov")
    {
        verticalFov = value.x;
        return true;
    }

    if (name == "aspectRatio")
    {
        aspectRatio = value.x;
        return true;
    }

    return SceneGraphLeaf::SetProperty(name, value);
}

nvrhi::AutoPtr<SceneGraphLeaf> OrthographicCamera::Clone()
{
    auto copy = MAKE_RC_OBJ_PTR(OrthographicCamera);
    copy->zNear = zNear;
    copy->zFar = zFar;
    copy->xMag = xMag;
    copy->yMag = yMag;
    return copy;
}

void OrthographicCamera::Load(const Json::Value& node)
{
    node["xMag"] >> xMag;
    node["yMag"] >> yMag;
    node["zNear"] >> zNear;
    node["zFar"] >> zFar;
}

bool OrthographicCamera::SetProperty(const std::string& name, const dm::float4& value)
{
    if (name == "zNear")
    {
        zNear = value.x;
        return true;
    }

    if (name == "zFar")
    {
        zFar = value.x;
        return true;
    }

    if (name == "xMag")
    {
        xMag = value.x;
        return true;
    }

    if (name == "yMag")
    {
        yMag = value.x;
        return true;
    }

    return SceneGraphLeaf::SetProperty(name, value);
}

// ------------------------------------------------------------------------- leaf type descriptors (D19)

namespace
{
    // One descriptor per leaf class (the class is part of the template's identity, so classes with the same tags do
    // not share one).
    template <typename Leaf, typename... Tags>
    const LeafTypeDesc& MakeLeafTypeDesc() noexcept
    {
        static const ComponentTypeDesc* const s_Tags[] = { &ComponentTraits<Tags>::Desc()... };
        static const LeafTypeDesc s_Desc{ nvrhi::uuid_of<Leaf>(), s_Tags, uint32_t(sizeof...(Tags)), nullptr, 0 };
        return s_Desc;
    }

    // Mesh instances keep their mesh reference, render flags and GPU slots in columns (plan step E4), which the
    // graph's leaf observer fills when the leaf is attached.
    template <typename Leaf, typename... Tags>
    const LeafTypeDesc& MakeMeshLeafTypeDesc() noexcept
    {
        static const ComponentTypeDesc* const s_Tags[] = { &ComponentTraits<Tags>::Desc()... };
        static const ComponentTypeDesc* const s_Columns[] = { &ComponentTraits<MeshRef>::Desc(),
                                                              &ComponentTraits<RenderFlags>::Desc(),
                                                              &ComponentTraits<GpuInstanceSlot>::Desc() };
        static const LeafTypeDesc s_Desc{ nvrhi::uuid_of<Leaf>(), s_Tags, uint32_t(sizeof...(Tags)), s_Columns,
                                          uint32_t(std::size(s_Columns)) };
        return s_Desc;
    }
}

const LeafTypeDesc& SceneGraphLeaf::GetTypeDesc() const
{
    static const LeafTypeDesc s_Desc{ IID_SceneGraphLeaf, nullptr, 0, nullptr, 0 };
    return s_Desc;
}

const LeafTypeDesc& MeshInstance::GetTypeDesc() const
{
    return MakeMeshLeafTypeDesc<MeshInstance, MeshInstanceLeaf>();
}

const LeafTypeDesc& SkinnedMeshInstance::GetTypeDesc() const
{
    return MakeMeshLeafTypeDesc<SkinnedMeshInstance, MeshInstanceLeaf, SkinnedMeshInstanceLeaf>();
}

int MeshInstance::GetInstanceIndex() const
{
    const GpuInstanceSlot* slot = SceneGraphStore::InstanceSlotOf(this);
    return slot ? int(slot->instance) : -1;
}

int MeshInstance::GetGeometryInstanceIndex() const
{
    const GpuInstanceSlot* slot = SceneGraphStore::InstanceSlotOf(this);
    return slot ? int(slot->firstGeometryInstance) : -1;
}

const LeafTypeDesc& SkinnedMeshReference::GetTypeDesc() const
{
    return MakeLeafTypeDesc<SkinnedMeshReference, SkinnedMeshReferenceLeaf>();
}

const LeafTypeDesc& SceneCamera::GetTypeDesc() const
{
    return MakeLeafTypeDesc<SceneCamera, CameraLeaf>();
}

const LeafTypeDesc& PerspectiveCamera::GetTypeDesc() const
{
    return MakeLeafTypeDesc<PerspectiveCamera, CameraLeaf, PerspectiveCameraLeaf>();
}

const LeafTypeDesc& OrthographicCamera::GetTypeDesc() const
{
    return MakeLeafTypeDesc<OrthographicCamera, CameraLeaf, OrthographicCameraLeaf>();
}

const LeafTypeDesc& Light::GetTypeDesc() const
{
    return MakeLeafTypeDesc<Light, LightLeaf>();
}

const LeafTypeDesc& DirectionalLight::GetTypeDesc() const
{
    return MakeLeafTypeDesc<DirectionalLight, LightLeaf, DirectionalLightLeaf>();
}

const LeafTypeDesc& SpotLight::GetTypeDesc() const
{
    return MakeLeafTypeDesc<SpotLight, LightLeaf, SpotLightLeaf>();
}

const LeafTypeDesc& PointLight::GetTypeDesc() const
{
    return MakeLeafTypeDesc<PointLight, LightLeaf, PointLightLeaf>();
}

const LeafTypeDesc& SceneGraphAnimation::GetTypeDesc() const
{
    return MakeLeafTypeDesc<SceneGraphAnimation, AnimationLeaf>();
}

// ------------------------------------------------------------------------------- nodes as handles (5.2)

OrphanRecord& SceneGraphNode::EnsureRecord()
{
    assert(!m_Store);
    if (!m_Record)
        m_Record = OrphanRecord::Create();
    return *m_Record;
}

dm::dquat SceneGraphNode::GetRotation() const
{
    if (m_Record)
        return m_Record->local.rotation;
    if (m_Store)
    {
        if (const LocalTransform* local = m_Store->GetOf<LocalTransform>(this, SceneGraphStore::SlotLocal))
            return local->rotation;
        LocalTransform driven;
        if (m_Store->GetDrivenLocal(this, driven))
            return driven.rotation;
    }
    return dm::dquat::identity();
}

dm::double3 SceneGraphNode::GetScaling() const
{
    if (m_Record)
        return m_Record->local.scaling;
    if (m_Store)
    {
        if (const LocalTransform* local = m_Store->GetOf<LocalTransform>(this, SceneGraphStore::SlotLocal))
            return local->scaling;
        LocalTransform driven;
        if (m_Store->GetDrivenLocal(this, driven))
            return driven.scaling;
    }
    return dm::double3(1.0);
}

dm::double3 SceneGraphNode::GetTranslation() const
{
    if (m_Record)
        return m_Record->local.translation;
    if (m_Store)
    {
        if (const LocalTransform* local = m_Store->GetOf<LocalTransform>(this, SceneGraphStore::SlotLocal))
            return local->translation;
        LocalTransform driven;
        if (m_Store->GetDrivenLocal(this, driven))
            return driven.translation;
    }
    return dm::double3(0.0);
}

dm::daffine3 SceneGraphNode::GetLocalToParentTransform() const
{
    const LocalTransform* local = nullptr;
    if (m_Record)
        local = m_Record->hasLocal ? &m_Record->local : nullptr;
    else if (m_Store)
        local = m_Store->GetOf<LocalTransform>(this, SceneGraphStore::SlotLocal);
    if (!local)
    {
        dm::daffine3 driven;
        return !m_Record && m_Store && m_Store->GetDrivenLocal(this, driven) ? driven : dm::daffine3::identity();
    }
    dm::daffine3 transform = dm::scaling(local->scaling);
    transform *= local->rotation.toAffine();
    transform *= dm::translation(local->translation);
    return transform;
}

dm::daffine3 SceneGraphNode::GetLocalToWorldTransform() const
{
    if (m_Record)
        return m_Record->world.transform;
    if (m_Store)
    {
        if (const WorldTransform* world = m_Store->GetOf<WorldTransform>(this, SceneGraphStore::SlotWorld))
            return world->transform;
    }
    return dm::daffine3::identity();
}

dm::affine3 SceneGraphNode::GetLocalToWorldTransformFloat() const
{
    return dm::affine3(GetLocalToWorldTransform());
}

// While the values of the last Refresh are current (SceneGraphStore::IsSubgraphCurrent), a node without children answers
// from its row (WorldBounds, RenderFlags or its leaf) and a node with children from the subgraph values Refresh stored
// in it, without a lock; otherwise (a structural change, SetLeaf or InvalidateContent not yet refreshed) the on-demand
// caches answer, as before.
dm::box3 SceneGraphNode::GetGlobalBoundingBox() const
{
    if (m_Store && !m_Record)
    {
        if (m_Store->IsSubgraphCurrent())
        {
            SceneGraphStore::RowRef row;
            if (m_Store->LocateNode(this, row))
            {
                if (!m_Store->HasChildren(row))
                    return m_Store->OwnBounds(row);
                if (m_CacheEpoch.load(std::memory_order_acquire) == m_Store->GetEpoch())
                    return m_CachedBounds;
            }
        }
        if (m_CacheEpoch.load(std::memory_order_acquire) != m_Store->GetEpoch())
            m_Store->ComputeSubgraphCache(this);
        return m_CachedBounds;
    }
    // a node in no graph: computed on every call
    std::vector<SceneGraphStore::CacheFrame> stack;
    dm::box3 bounds;
    SceneContentFlags leafContent, content;
    SceneGraphStore::ComputeSubgraph(this, stack, 0, false, bounds, leafContent, content);
    return bounds;
}

SceneContentFlags SceneGraphNode::GetLeafContentFlags() const
{
    if (m_Store && !m_Record && m_Store->IsSubgraphCurrent())
    {
        SceneGraphStore::RowRef row;
        if (m_Store->LocateNode(this, row))
            return m_Store->OwnContent(row);
    }
    if (m_Store && !m_Record && m_CacheEpoch.load(std::memory_order_acquire) == m_Store->GetEpoch())
        return m_CachedLeafContent;
    SceneGraphLeaf* leaf = GetLeaf();
    return leaf ? leaf->GetContentFlags() : SceneContentFlags::None;
}

SceneContentFlags SceneGraphNode::GetSubgraphContentFlags() const
{
    if (m_Store && !m_Record)
    {
        if (m_Store->IsSubgraphCurrent())
        {
            SceneGraphStore::RowRef row;
            if (m_Store->LocateNode(this, row))
            {
                if (!m_Store->HasChildren(row))
                    return m_Store->OwnContent(row);
                if (m_CacheEpoch.load(std::memory_order_acquire) == m_Store->GetEpoch())
                    return m_CachedSubgraphContent;
            }
        }
        if (m_CacheEpoch.load(std::memory_order_acquire) != m_Store->GetEpoch())
            m_Store->ComputeSubgraphCache(this);
        return m_CachedSubgraphContent;
    }
    std::vector<SceneGraphStore::CacheFrame> stack;
    dm::box3 bounds;
    SceneContentFlags leafContent, content;
    SceneGraphStore::ComputeSubgraph(this, stack, 0, false, bounds, leafContent, content);
    return content;
}

SceneGraphNode::DirtyFlags SceneGraphNode::GetDirtyFlags() const
{
    if (m_Store && !m_Record)
    {
        DirtyFlags flags = DirtyFlags::None;
        if (m_Store->HasPendingStructureChanges())
            flags |= DirtyFlags::SubgraphStructure;
        if (m_Store->HasPendingTransformChanges())
            flags |= DirtyFlags::SubgraphTransforms;
        return flags;
    }
    // a node in no graph is new structure for the graph it joins
    DirtyFlags flags = DirtyFlags::SubgraphStructure;
    if (m_Record && m_Record->hasLocal)
        flags |= DirtyFlags::LocalTransform | DirtyFlags::SubgraphTransforms;
    return flags;
}

SceneGraphNode* SceneGraphNode::GetParent() const
{
    if (m_Record)
        return m_Record->parent;
    if (m_Store)
    {
        if (const Parent* parent = m_Store->GetOf<Parent>(this, SceneGraphStore::SlotParent))
            return m_Store->NodeOf(parent->entity);
    }
    return nullptr;
}

SceneGraphNode* SceneGraphNode::GetFirstChild() const
{
    if (m_Record)
        return m_Record->firstChild.Get();
    if (m_Store)
    {
        if (const SceneLinks* links = m_Store->GetOf<SceneLinks>(this, SceneGraphStore::SlotLinks))
            return m_Store->NodeOf(links->firstChild);
    }
    return nullptr;
}

SceneGraphNode* SceneGraphNode::GetLastChild() const
{
    if (m_Record)
        return m_Record->lastChild;
    if (m_Store)
    {
        if (const SceneLinks* links = m_Store->GetOf<SceneLinks>(this, SceneGraphStore::SlotLinks))
            return m_Store->NodeOf(links->lastChild);
    }
    return nullptr;
}

SceneGraphNode* SceneGraphNode::GetNextSibling() const
{
    if (m_Record)
        return m_Record->nextSibling.Get();
    if (m_Store)
    {
        if (const SceneLinks* links = m_Store->GetOf<SceneLinks>(this, SceneGraphStore::SlotLinks))
            return m_Store->NodeOf(links->nextSibling);
    }
    return nullptr;
}

SceneGraphNode* SceneGraphNode::GetPrevSibling() const
{
    if (m_Record)
        return m_Record->prevSibling;
    if (m_Store)
    {
        if (const SceneLinks* links = m_Store->GetOf<SceneLinks>(this, SceneGraphStore::SlotLinks))
            return m_Store->NodeOf(links->prevSibling);
    }
    return nullptr;
}

SceneGraphLeaf* SceneGraphNode::GetLeaf() const
{
    if (m_Record)
        return m_Record->leaf.Get();
    if (m_Store)
    {
        if (const LeafObject* leaf = m_Store->GetOf<LeafObject>(this, SceneGraphStore::SlotLeaf))
            return leaf->object;
    }
    return nullptr;
}

SceneGraph* SceneGraphNode::GetGraph() const
{
    return m_Store ? m_Store->GetGraph() : nullptr;
}

std::filesystem::path SceneGraphNode::GetPath() const
{
    std::filesystem::path path = GetName();

    SceneGraphWalker walker(GetParent(), nullptr);
    while (walker)
    {
        path = walker->GetName() / path;
        walker.Up();
    }

    return "/" / path;
}

void SceneGraphNode::InvalidateContent()
{
    if (m_Store && !m_Record)
    {
        DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "InvalidateContent");
        m_Store->InvalidateContent(this);
    }
}

void SceneGraphNode::SetTransform(const dm::double3* translation, const dm::dquat* rotation, const dm::double3* scaling)
{
    if (m_Store && !m_Record)
    {
        DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "SetTransform");
        m_Store->SetTransform(this, translation, rotation, scaling);
        return;
    }

    OrphanRecord& record = EnsureRecord();
    if (scaling) record.local.scaling = *scaling;
    if (rotation) record.local.rotation = *rotation;
    if (translation) record.local.translation = *translation;
    record.hasLocal = true;
}

void SceneGraphNode::SetScaling(const dm::double3& scaling)
{
    SetTransform(nullptr, nullptr, &scaling);
}

void SceneGraphNode::SetRotation(const dm::dquat& rotation)
{
    SetTransform(nullptr, &rotation, nullptr);
}

void SceneGraphNode::SetTranslation(const dm::double3& translation)
{
    SetTransform(&translation, nullptr, nullptr);
}

void SceneGraphNode::SetLeaf(SceneGraphLeaf* leaf)
{
    if (leaf && SceneGraphStore::IsOnAnotherNode(leaf, this))
    {
        log::warning("SceneGraphNode::SetLeaf: the leaf is on another node, and its class keeps its columns in one place "
                     "(LeafTypeDesc::leafRecord); nothing changed. Take it from that node first, or clone it");
        return;
    }
    if (m_Store && !m_Record)
    {
        DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "SetLeaf");
        m_Store->SetLeaf(this, leaf);
        return;
    }

    OrphanRecord& record = EnsureRecord();
    if (record.leaf)
        record.leaf->m_Node.Reset();
    record.leaf = leaf;
    if (leaf)
        leaf->m_Node = nvrhi::WeakPtr<SceneGraphNode>(this);
}

void SceneGraphNode::SetName(const std::string& name)
{
    m_Name = name;
}

SceneGraphNode::~SceneGraphNode()
{
    // A node dies attached only when its graph's world releases it; its state went with the row. A node in no graph
    // owns its children through its record.
    if (!m_Record)
        return;

    // Release the children one at a time, cutting every sibling link first, so that no destructor owns the
    // next sibling and the destructors never nest per sibling. Destroying a node with N children then costs
    // stack proportional to the depth of its subgraph, not to N (see NVIDIAGameWorks/donut#18).
    nvrhi::AutoPtr<SceneGraphNode> child = std::move(m_Record->firstChild);
    while (child)
    {
        // A child that someone else still references outlives this node as a root of its own subgraph.
        OrphanRecord& childRecord = *child->m_Record;
        childRecord.parent = nullptr;
        childRecord.prevSibling = nullptr;

        // Releases the previous child, which no longer owns a sibling
        child = std::move(childRecord.nextSibling);
    }
    OrphanRecord::Destroy(m_Record);
    m_Record = nullptr;
}

namespace
{
    inline void PrefetchObject(const void* object) noexcept
    {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
        _mm_prefetch(static_cast<const char*>(object), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(object);
#else
        (void)object;
#endif
    }
}

int SceneGraphWalker::Next(bool allowChildren)
{
    if (!m_Current)
        return 0;

    // An attached node: one read of its row's SceneLinks per step, through the entity (the entity table, then the
    // store's NodeOf index for the next node); Parent only when the walk climbs. The node objects are not read.
    if (!m_Store && m_Current->m_Store && !m_Current->m_Record)
    {
        m_Store = m_Current->m_Store;
        m_Entity = m_Current->m_Entity;
    }
    if (m_Store)
    {
        // The node the walk will most likely visit after the one it moves to (that one's first child, else its next
        // sibling): its object is fetched into the cache while the caller reads the current one, as a walk over node
        // objects linked by pointers had the hardware do.
        auto prefetchAfter = [this](Entity entity) noexcept
        {
            SceneGraphStore::RowRef next;
            if (!m_Store->Locate(entity, next))
                return;
            const SceneLinks* links =
                static_cast<const SceneLinks*>(m_Store->Column(next, SceneGraphStore::SlotLinks));
            const Entity after = !links ? Entity::Null
                                 : links->firstChild != Entity::Null ? links->firstChild : links->nextSibling;
            if (after != Entity::Null)
            {
                if (const SceneGraphNode* node = m_Store->NodeOf(after))
                    PrefetchObject(node);
            }
        };
        SceneGraphStore::RowRef row;
        if (m_Store->Locate(m_Entity, row))
        {
            const SceneLinks* links =
                static_cast<const SceneLinks*>(m_Store->Column(row, SceneGraphStore::SlotLinks));
            if (allowChildren && links && links->firstChild != Entity::Null)
            {
                m_Entity = links->firstChild;
                m_Current = m_Store->NodeOf(m_Entity);
                prefetchAfter(m_Entity);
                return 1;
            }
            int depth = 0;
            for (;;)
            {
                if (m_Current == m_Scope)
                {
                    m_Current = nullptr;
                    return depth;
                }
                if (links && links->nextSibling != Entity::Null)
                {
                    m_Entity = links->nextSibling;
                    m_Current = m_Store->NodeOf(m_Entity);
                    prefetchAfter(m_Entity);
                    return depth;
                }
                const Parent* parent = static_cast<const Parent*>(m_Store->Column(row, SceneGraphStore::SlotParent));
                --depth;
                if (!parent || !m_Store->Locate(parent->entity, row))
                {
                    m_Current = nullptr;
                    return depth;
                }
                m_Entity = parent->entity;
                m_Current = m_Store->NodeOf(m_Entity);
                links = static_cast<const SceneLinks*>(m_Store->Column(row, SceneGraphStore::SlotLinks));
            }
        }
        // not a row of the graph (cannot happen while the graph does not change): the general path
        m_Store = nullptr;
        m_Entity = Entity::Null;
    }

    // Try to move down to the first child of the current node
    if (allowChildren)
    {
        SceneGraphNode* firstChild = m_Current->GetFirstChild();
        if (firstChild)
        {
            m_Current = firstChild;
            return 1;
        }
    }

    int depth = 0;

    // No chlidren or not allowed to use them - try the next sibling or go up and try parent's sibling, etc.
    while (m_Current)
    {
        if (m_Current == m_Scope)
        {
            m_Current = nullptr;
            return depth;
        }

        SceneGraphNode* nextSibling = m_Current->GetNextSibling();
        if (nextSibling)
        {
            m_Current = nextSibling;
            return depth;
        }

        m_Current = m_Current->GetParent();
        --depth;
    }

    return depth;
}

int SceneGraphWalker::Up()
{
    m_Store = nullptr;   // Next takes the entity from the node again
    m_Entity = Entity::Null;
    if (!m_Current)
        return 0;

    if (m_Current == m_Scope)
    {
        m_Current = nullptr;
        return 0;
    }

    m_Current = m_Current->GetParent();
    return -1;
}

bool SceneGraphAnimationChannel::Apply(float time) const
{
    auto node = m_TargetNode.Lock();
    auto material = m_TargetMaterial.Lock();
    if ((!node && m_Attribute != AnimationAttribute::LeafProperty) || 
        (!material && !node && m_Attribute == AnimationAttribute::LeafProperty))
        return false;

    auto valueOption = m_Sampler->Evaluate(time, true);
    if (!valueOption.has_value())
        return false;

    auto value = valueOption.value();

    switch(m_Attribute)
    {
    case AnimationAttribute::Scaling:
        node->SetScaling(dm::double3(value.xyz()));
        break;

    case AnimationAttribute::Rotation: {
        dm::dquat quat = dm::dquat::fromXYZW(dm::double4(value));
        double len = length(quat);
        if (len == 0.0)
        {
            log::warning("Rotation quaternion interpolated to zero, ignoring.");
        }
        else
        {
            quat /= len;
            node->SetRotation(quat);
        }
        break;
    }

    case AnimationAttribute::Translation:
        node->SetTranslation(dm::double3(value.xyz()));
        break;

    case AnimationAttribute::LeafProperty: {
        if (material)
        {
            if (!material->SetProperty(m_LeafPropertyName, value))
            {
                log::warning("Cannot set property '%s' on material '%s': the material doesn't support this property.",
                    m_LeafPropertyName.c_str(), material->name.c_str());
            }
        }
        else
        {
            const auto& leaf = node->GetLeaf();
            if (leaf)
            {
                if (!leaf->SetProperty(m_LeafPropertyName, value))
                {
                    log::warning("Cannot set property '%s' on node '%s': the leaf doesn't support this property.",
                        m_LeafPropertyName.c_str(), node->GetName().c_str());
                }
            }
            else
            {
                log::warning("Cannot set property '%s' on node '%s' which has no leaf.",
                    m_LeafPropertyName.c_str(), node->GetName().c_str());
            }
        }
        break;
    }
        
    case AnimationAttribute::Undefined:
    default:
        log::warning("Unsupported animation target (%d), ignoring.", uint32_t(m_Attribute));
        return false;
    }

    return true;
}

nvrhi::AutoPtr<SceneGraphLeaf> SceneGraphAnimation::Clone()
{
    auto copy = MAKE_RC_OBJ_PTR(SceneGraphAnimation);
    for (const auto& channel : m_Channels)
    {
        auto channelCopy = MAKE_RC_OBJ_PTR(SceneGraphAnimationChannel,
            channel->GetSampler(), channel->GetTargetNode(), channel->GetAttribute());
        copy->AddChannel(channelCopy);
    }
    return copy;
}

bool SceneGraphAnimationChannel::IsValid() const
{
    return m_TargetNode.IsValid();
}

void SceneGraphAnimation::AddChannel(SceneGraphAnimationChannel *channel)
{
    m_Channels.push_back(channel);
    m_Duration = std::max(m_Duration, channel->GetSampler()->GetEndTime());
;}

bool SceneGraphAnimation::Apply(float time) const
{
    bool success = false;

    for (const auto& channel : m_Channels)
    {
        success = channel->Apply(time) && success;
    }

    return success;
}

bool SceneGraphAnimation::IsVald() const
{
    for (const auto& channel : m_Channels)
    {
        if (!channel->IsValid())
            return false;
    }

    return true;
}

SceneGraph::SceneGraph()
{
    m_Store = new SceneGraphStore(this);
}

SceneGraph::~SceneGraph()
{
    // The graph's own reference to its root goes first, so that the store sees which nodes are referenced elsewhere.
    m_Root.Reset();
    delete m_Store;
    m_Store = nullptr;
}

IWorld* SceneGraph::GetWorld() const
{
    return m_Store->GetWorld();
}

Entity SceneGraph::GetEntity(const SceneGraphNode* node) const
{
    return node && node->m_Store == m_Store && !node->m_Record ? node->m_Entity : Entity::Null;
}

SceneGraphNode* SceneGraph::GetNode(Entity entity) const
{
    return m_Store->NodeOf(entity);
}

IScheduler* SceneGraph::GetScheduler() const
{
    return m_Store->GetScheduler();
}

void SceneGraph::SetJobSystem(IJobSystem* jobSystem)
{
    m_Store->SetJobSystem(jobSystem);
}

bool SceneGraph::HasPendingStructureChanges() const
{
    return m_Store->HasPendingStructureChanges();
}

bool SceneGraph::HasPendingTransformChanges() const
{
    return m_Store->HasPendingTransformChanges();
}

bool SceneGraph::HasStaleWorldTransforms() const
{
    return m_Store->HasStaleWorldTransforms();
}

void SceneGraph::RegisterLeaf(SceneGraphLeaf *leaf)
{
    if (!leaf)
        return;

    // The asset tables were updated (and OnMeshAdded and OnMaterialAdded raised) by the leaf observer before this
    // call, from the instance's MeshRef column (plan step E4).
    auto meshInstance = query_cast<MeshInstance>(leaf);
    if (meshInstance)
    {
        m_MeshInstances.push_back(meshInstance);

        auto skinnedInstance = query_cast<SkinnedMeshInstance>(meshInstance);
        if (skinnedInstance)
        {
            m_SkinnedMeshInstances.push_back(skinnedInstance);
        }

        return;
    }

    auto animation = query_cast<SceneGraphAnimation>(leaf);
    if (animation)
    {
        m_Animations.push_back(animation);
        return;
    }

    auto camera = query_cast<SceneCamera>(leaf);
    if (camera)
    {
        m_Cameras.push_back(camera);
        return;
    }

    auto light = query_cast<Light>(leaf);
    if (light)
    {
        m_Lights.push_back(light);
        return;
    }
}

void SceneGraph::UnregisterLeaf(SceneGraphLeaf *leaf)
{
    if (!leaf)
        return;

    // Removes the leaf's first entry from a vector, keeping the order of the rest: at once, or, during a Detach, by
    // counting a mark on the leaf (once per call), so that CompactLeafVectors removes as many first entries from each
    // vector in one pass at the Detach's end. Until then the vectors still hold the leaves unregistered before it.
    if (m_DeferLeafRemoval)
    {
        if (leaf->m_RemovalPending++ == 0)
            m_RemovalMarked.push_back(leaf);
    }
    auto removeLeaf = [this](auto& items, auto* item)
    {
        if (m_DeferLeafRemoval)
            return;
        auto it = std::find(items.begin(), items.end(), item);
        if (it != items.end())
            items.erase(it);
    };

    // The leaf observer released the instance's mesh (and raised OnMeshRemoved and OnMaterialRemoved) before this call.
    auto meshInstance = query_cast<MeshInstance>(leaf);
    if (meshInstance)
    {
        // A Detach removes its leaves in one pass at its end (CompactLeafVectors); the order of the rest is kept.
        removeLeaf(m_MeshInstances, meshInstance);

        // a skinned instance is a mesh instance as well: RegisterLeaf put it into both vectors
        auto skinnedInstance = query_cast<SkinnedMeshInstance>(meshInstance);
        if (skinnedInstance)
            removeLeaf(m_SkinnedMeshInstances, skinnedInstance);

        return;
    }

    auto animation = query_cast<SceneGraphAnimation>(leaf);
    if (animation)
    {
        removeLeaf(m_Animations, animation);
        return;
    }

    auto camera = query_cast<SceneCamera>(leaf);
    if (camera)
    {
        removeLeaf(m_Cameras, camera);
        return;
    }

    auto light = query_cast<Light>(leaf);
    if (light)
    {
        removeLeaf(m_Lights, light);
        return;
    }
}

void SceneGraph::CompactLeafVectors()
{
    if (m_RemovalMarked.empty())
        return;
    // Drops, per vector, as many first entries of each marked leaf as it was unregistered (the old erase of the first
    // entry, once per call), keeping the order of the rest.
    auto compact = [this](auto& items)
    {
        size_t kept = 0;
        for (size_t i = 0; i < items.size(); ++i)
        {
            SceneGraphLeaf* leaf = items[i].Get();
            if (leaf->m_RemovalTaken < leaf->m_RemovalPending)
            {
                ++leaf->m_RemovalTaken;
                continue;
            }
            if (kept != i)
                items[kept] = std::move(items[i]);
            ++kept;
        }
        items.resize(kept);
        for (const auto& leaf : m_RemovalMarked)
            leaf->m_RemovalTaken = 0;
    };
    compact(m_MeshInstances);
    compact(m_SkinnedMeshInstances);
    compact(m_Animations);
    compact(m_Cameras);
    compact(m_Lights);
    for (const auto& leaf : m_RemovalMarked)
        leaf->m_RemovalPending = 0;
    m_RemovalMarked.clear();
}

void SceneGraph::BeginRenderRead(RenderReadMode mode) noexcept
{
    if (mode == RenderReadMode::Defer)   // E7.8 implements deferred releases
        log::warning("SceneGraph::BeginRenderRead: RenderReadMode::Defer is not implemented yet; the scope warns");
    m_Store->BeginRenderRead();
}

void SceneGraph::EndRenderRead() noexcept
{
    m_Store->EndRenderRead();
}

bool SceneGraph::IsInRenderRead() const noexcept
{
    return m_Store->IsInRenderRead();
}

nvrhi::FRESULT SceneGraph::RunReadOnlyPhase(Phase phase) noexcept
{
    return m_Store->RunReadOnlyPhase(phase);
}

bool SceneGraph::ClaimRenderService() noexcept
{
    return m_Store->ClaimRenderService();
}

void SceneGraph::ReleaseRenderService() noexcept
{
    m_Store->ReleaseRenderService();
}

uint64_t SceneGraph::GetSchedulerGeneration() const noexcept
{
    return m_Store->GetSchedulerGeneration();
}

nvrhi::AutoPtr<SceneGraphNode> SceneGraph::SetRootNode(SceneGraphNode *root)
{
    DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "SetRootNode");
    DONUT_SCENE_GRAPH_RENDER_READ_QUIET(m_Store);   // the Detach and Attach below are part of this call
    auto oldRoot = m_Root;
    if (m_Root)
        Detach(m_Root);

    Attach(nullptr, root);

    return oldRoot;
}

nvrhi::AutoPtr<SceneGraphNode> SceneGraph::Attach(SceneGraphNode* parent, SceneGraphNode* child) {
    SceneGraph* parentGraph = parent ? parent->GetGraph() : this;
    SceneGraph* childGraph = child->GetGraph();

    if (!parentGraph && !childGraph)
    {
        // operating on an orphaned subgraph - do not copy or register anything
        assert(parent);
        SceneGraphStore::AppendOrphan(parent, child);
        return child;
    }

    assert(parentGraph == this);
    DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "Attach");
    nvrhi::AutoPtr<SceneGraphNode> attachedChild;

    if (childGraph)
    {
        // attaching a subgraph that already belongs to a graph - this one or another: a deep copy (design 5.3)
        attachedChild = m_Store->AttachCopy(parent, child, childGraph->m_Store);
    }
    else
    {
        // attaching a subgraph that has been detached from another graph (or never attached): its records move into
        // rows, in pre-order
        attachedChild = m_Store->AttachOrphan(parent, child);
    }

    if (!parent && attachedChild)
        m_Root = attachedChild;

    return attachedChild;
}

nvrhi::AutoPtr<SceneGraphNode> SceneGraph::AttachLeafNode(SceneGraphNode *parent, SceneGraphLeaf *leaf)
{
    auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
    if (leaf->GetNode())
        node->SetLeaf(leaf->Clone());
    else
        node->SetLeaf(leaf);
    return Attach(parent, node);
}

nvrhi::AutoPtr<SceneGraphNode> SceneGraph::Detach(SceneGraphNode* node, bool /* preserveOrder */)
{
    nvrhi::AutoPtr<SceneGraphNode> detached;
    if (node->m_Store)
    {
        assert(node->m_Store == m_Store);
        DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "Detach");
        // the subgraph's rows move into records and its entities go; the leaf observer unregisters the leaves
        detached = m_Store->Detach(node);
    }
    else
    {
        // a node of an orphaned subgraph leaves its parent's child list, handing over the reference the list held
        detached = SceneGraphStore::UnlinkOrphan(node);
        if (!detached)
            detached = node;
    }

    if (m_Root == node && !node->m_Store)
    {
        m_Root = MAKE_RC_OBJ_PTR(SceneGraphNode);
    }

    return detached;
}

SceneGraphNode* SceneGraph::FindNode(const std::filesystem::path& path, SceneGraphNode* context) const
{
    auto pathComponent = path.begin();

    if (pathComponent == path.end())
        return nullptr;

    if (*pathComponent == "/")
    {
        context = m_Root;
        ++pathComponent;
    }

    if (!context)
    {
        log::error("Relative node queries with NULL context are not supported");
        return nullptr;
    }

    SceneGraphNode* current = context;
    
    while (current && pathComponent != path.end())
    {
        if (*pathComponent == "..")
        {
            current = current->GetParent();
            ++pathComponent;
            continue;
        }

        auto children = current->GetChildren();
        auto found = std::find_if(children.begin(), children.end(),
            [&pathComponent](SceneGraphNode* item) { return item->GetName() == *pathComponent; });

        if (found != children.end())
        {
            current = *found;
            ++pathComponent;
            continue;
        }

        return nullptr;
    }

    return current;
}

void SceneGraph::Refresh(uint32_t frameIndex)
{
    // TransformPropagation and Bounds (design 5.7), when anything changed since the last call.
    const bool structureDirty = m_Store->Refresh(frameIndex);

    // Slots are stable (step E4): nothing is renumbered. After structural changes, as the old renumbering did, the mesh
    // table follows meshes whose geometries were edited in place since they were added.
    if (structureDirty)
        m_Store->ReconcileMeshes();
}

nvrhi::FRESULT SceneGraph::Simulate(const FrameTime& time) noexcept
{
    DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "Simulate");
    return m_Store->Simulate(time);
}

nvrhi::FRESULT SceneGraph::SetSimulationDriven(SceneGraphNode* node, bool driven) noexcept
{
    if (GetEntity(node) == Entity::Null)
        return nvrhi::FE_INVALID_ARGS;   // null, in no graph, or in another one
    DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "SetSimulationDriven");
    try
    {
        return m_Store->SetSimulationDriven(node, driven);
    }
    catch (...)
    {
        return nvrhi::FE_OUT_OF_MEMORY;
    }
}

MeshId SceneGraph::AcquireRuntimeMesh(MeshInfo* mesh) noexcept
{
    DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "AcquireRuntimeMesh");
    return m_Store->AcquireRuntimeMesh(mesh);
}

void SceneGraph::ReleaseRuntimeMesh(MeshId mesh) noexcept
{
    DONUT_SCENE_GRAPH_WARN_RENDER_READ(m_Store, "ReleaseRuntimeMesh");
    m_Store->ReleaseRuntimeMesh(mesh);
}

uint32_t SceneGraph::GetRuntimeMeshInstanceComponents(ComponentId* out, uint32_t capacity,
                                                      bool simulationDriven) const noexcept
{
    return m_Store->GetRuntimeMeshInstanceComponents(out, capacity, simulationDriven);
}

uint32_t SceneGraph::GetRuntimeMeshInstanceCount() const noexcept
{
    return m_Store->GetRuntimeMeshInstanceCount();
}

const size_t SceneGraph::GetGeometryInstancesCount() const
{
    return m_Store->GetGeometryInstanceSlots().GetEnd();
}

size_t SceneGraph::GetInstanceSlotCount() const
{
    return m_Store->GetInstanceSlots().GetEnd();
}

nvrhi::AutoPtr<SceneGraph> SceneTypeFactory::CreateGraph()
{
    return MAKE_RC_OBJ_PTR(SceneGraph);
}

nvrhi::AutoPtr<SceneGraphLeaf> SceneTypeFactory::CreateLeaf(const std::string& type)
{
    if (type == "DirectionalLight")
    {
        return MAKE_RC_OBJ_PTR(DirectionalLight);
    }
    if (type == "PointLight")
    {
        return MAKE_RC_OBJ_PTR(PointLight);
    }
    if (type == "SpotLight")
    {
        return MAKE_RC_OBJ_PTR(SpotLight);
    }
    if (type == "PerspectiveCamera")
    {
        return MAKE_RC_OBJ_PTR(PerspectiveCamera);
    }
    if (type == "OrthographicCamera")
    {
        return MAKE_RC_OBJ_PTR(OrthographicCamera);
    }

    return nullptr;
}

nvrhi::AutoPtr<Material> SceneTypeFactory::CreateMaterial()
{
    return MAKE_RC_OBJ_PTR(Material);
}

nvrhi::AutoPtr<MeshInfo> SceneTypeFactory::CreateMesh()
{
    return MAKE_RC_OBJ_PTR(MeshInfo);
}

nvrhi::AutoPtr<MeshGeometry> SceneTypeFactory::CreateMeshGeometry()
{
    return MAKE_RC_OBJ_PTR(MeshGeometry);
}

nvrhi::AutoPtr<MeshInstance> SceneTypeFactory::CreateMeshInstance(MeshInfo* mesh)
{
    return MAKE_RC_OBJ_PTR(MeshInstance, mesh);
}

nvrhi::AutoPtr<SkinnedMeshInstance> SceneTypeFactory::CreateSkinnedMeshInstance(SceneTypeFactory *sceneTypeFactory, MeshInfo *prototypeMesh)
{
    return MAKE_RC_OBJ_PTR(SkinnedMeshInstance, sceneTypeFactory, prototypeMesh);
}

void donut::engine::PrintSceneGraph(const SceneGraphNode *root)
{
    SceneGraphWalker walker(const_cast<SceneGraphNode *>(root));
    int depth = 0;
    while(walker)
    {
        std::stringstream ss;

        for (int i = 0; i < depth; i++)
            ss << "  ";

        if (walker->GetName().empty())
            ss << "<Unnamed>";
        else
            ss << walker->GetName();

        bool hasTranslation = dm::any(walker->GetTranslation() != dm::double3::zero());
        bool hasRotation = dm::any(walker->GetRotation() != dm::dquat::identity());
        bool hasScaling = dm::any(walker->GetScaling() != dm::double3(1.0));

        if (hasTranslation || hasRotation || hasScaling)
        {
            ss << " (";
            if (hasTranslation) ss << "T";
            if (hasRotation) ss << "R";
            if (hasScaling) ss << "S";
            ss << ")";
        }

        auto bbox = walker->GetGlobalBoundingBox();
        if (!bbox.isempty())
        {
            ss << " [" << bbox.m_mins.x << ", " << bbox.m_mins.y << ", " << bbox.m_mins.z << " .. "
                      << bbox.m_maxs.x << ", " << bbox.m_maxs.y << ", " << bbox.m_maxs.z << "]";
        }

        if (walker->GetLeaf())
        {
            ss << ": ";

            if (auto meshInstance = query_cast<MeshInstance>(walker->GetLeaf()))
            {
                if (!meshInstance->GetMesh()->name.empty())
                    ss << meshInstance->GetMesh()->name;
                else
                    ss << "Unnamed Mesh";

                ss << " (" << meshInstance->GetMesh()->geometries.size() << " geometries)";

                auto skinnedInstance = query_cast<SkinnedMeshInstance>(meshInstance);
                if (skinnedInstance)
                {
                    ss << " - skinned, " << skinnedInstance->joints.size() << " joints";
                }
            }
            else if (auto animation = query_cast<SceneGraphAnimation>(walker->GetLeaf()))
            {
                ss << "Animation (" << animation->GetChannels().size() << " channels)";
                log::info("%s", ss.str().c_str());
                ss.str(std::string()); // clear

                for (const auto& channel : animation->GetChannels())
                {
                    for (int i = 0; i < depth + 1; i++)
                        ss << "  ";

                    auto targetNode = channel->GetTargetNode();
                    if (targetNode)
                        if (!targetNode->GetName().empty())
                            ss << targetNode->GetName();
                        else
                            ss << "<Unnamed Target>";
                    else
                        ss << "<No Target>";

                    ss << " (";
                    switch(channel->GetAttribute())
                    {
                    case AnimationAttribute::Scaling: ss << "Scaling"; break;
                    case AnimationAttribute::Rotation: ss << "Rotation"; break;
                    case AnimationAttribute::Translation: ss << "Translation"; break;
                    case AnimationAttribute::Undefined:
                    default:
                        ss << "Unknown Attribute";
                    }
                    ss << "): ";
                    auto keyframes = channel->GetSampler()->GetKeyframes();
                    ss << keyframes.size() << " keyframes";
                    if (!keyframes.empty())
                    {
                        ss << ", " << keyframes[0].time << "s - " << keyframes[keyframes.size() - 1].time << "s";
                    }

                    log::info("%s", ss.str().c_str());
                    ss.str(std::string()); // clear
                }
            }
            else if (auto camera = query_cast<SceneCamera>(walker->GetLeaf()))
            {
                auto perspective = query_cast<PerspectiveCamera>(camera);
                auto orthographic = query_cast<OrthographicCamera>(camera);

                if (perspective)
                {
                    ss << "Perspective Camera (yFov = " << perspective->verticalFov
                        << ", zNear = " << perspective->zNear << ")";
                }
                else if (orthographic)
                {
                    ss << "Orthographic Camera (xMag = " << orthographic->xMag
                        << ", yMag = " << orthographic->yMag
                        << ", zNear = " << orthographic->zNear << ")";
                }
                else
                {
                    ss << "Unknown Type Camera";
                }
            }
            else if (auto light = query_cast<Light>(walker->GetLeaf()))
            {
                auto directional = query_cast<DirectionalLight>(light);
                auto point = query_cast<PointLight>(light);
                auto spot = query_cast<SpotLight>(light);

                if (directional)
                {
                    ss << "Directional Light (irradiance = " << directional->irradiance
                        << ", r = " << directional->color.x
                        << ", g = " << directional->color.y
                        << ", g = " << directional->color.z
                        << ", angularSize = " << directional->angularSize
                        << ")";
                }
                else if (point)
                {
                    ss << "Point Light (intensity = " << point->intensity
                        << ", r = " << point->color.x
                        << ", g = " << point->color.y
                        << ", g = " << point->color.z
                        << ", radius = " << point->radius
                        << ", range = " << point->range
                        << ")";
                }
                else if (spot)
                {
                    ss << "Spot Light (intensity = " << spot->intensity
                        << ", r = " << spot->color.x
                        << ", g = " << spot->color.y
                        << ", g = " << spot->color.z
                        << ", radius = " << spot->radius
                        << ", range = " << spot->range
                        << ", innerAngle = " << spot->innerAngle
                        << ", outerAngle = " << spot->outerAngle
                        << ")";
                }
                else
                {
                    ss << "Unknown Type Light";
                }
            }
            else if (query_cast<SkinnedMeshReference>(walker->GetLeaf()))
            {
                ss << "Joint";
            }
            else
            {
                ss << "Unkwown Leaf Type";
            }
        }

        if (!ss.str().empty())
            log::info("%s", ss.str().c_str());

        depth += walker.Next(true);
    }
}
