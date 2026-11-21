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

// The scene graph's components (aggregate docs/ecs-design.md 4.6, 4.7, 4.11, 5.1 and 9.2; plan step E3). Every node
// attached to a SceneGraph is one entity of the graph's world (one world per graph, D18), and these are its columns.
// Module-private C++: SceneNode and LeafObject hold implementation classes, so nothing here crosses a module boundary.
//
//     node without a leaf or a transform   SceneNode, SceneLinks, (Parent), HierarchyDepth, WorldTransform
//     node with a transform                ... plus LocalTransform and its cache LocalMatrix
//     node with a leaf                     ... plus LeafObject, the leaf class's tags (LeafTypeDesc), LocalBounds,
//                                          WorldBounds and the chunk's ChunkWorldBounds
//
// SceneNode and LeafObject are strong references with hooks (D14); they are not Relocatable, so snapshots and world
// hashes leave them out (IWorld::ComputeHash covers the rest: links, depths, transforms, bounds, the tags).

#pragma once

#include <donut/engine/EcsTypes.h>
#include <donut/core/math/math.h>

namespace donut::engine
{
    class SceneGraphNode;
    class SceneGraphLeaf;

    // The node object of an entity. A strong reference: the hooks AddRef on copy and Release on destruction.
    struct SceneNode
    {
        SceneGraphNode* object = nullptr;

        SceneNode() noexcept = default;
        explicit SceneNode(SceneGraphNode* adopted) noexcept : object(adopted) { }   // takes over one reference
        SceneNode(const SceneNode& other) noexcept;
        SceneNode(SceneNode&& other) noexcept : object(other.object) { other.object = nullptr; }
        SceneNode& operator=(const SceneNode& other) noexcept;
        SceneNode& operator=(SceneNode&& other) noexcept;
        ~SceneNode();
    };
    DONUT_ECS_COMPONENT(SceneNode, "f8ab4712-6fca-4d72-b6bf-921c316834f9", DONUT_ECS_FIELDS())

    // The intrusive, ordered child list of scene-graph nodes (D20): append, detach and the next sibling are O(1).
    struct SceneLinks
    {
        Entity firstChild;
        Entity lastChild;
        Entity prevSibling;
        Entity nextSibling;
    };
    DONUT_ECS_COMPONENT(SceneLinks, "1c0129ff-516e-4da3-9364-b0d99586573a",
                        DONUT_ECS_FIELDS(firstChild, lastChild, prevSibling, nextSibling))

    // On every entity that has a parent; transform propagation reads the parent's WorldTransform through it.
    struct Parent
    {
        Entity entity;
    };
    DONUT_ECS_COMPONENT(Parent, "d0a5ee22-cd57-4215-bccf-8da7d5f9f42d", DONUT_ECS_FIELDS(entity))

    // The depth bucket of propagation (D5): 0 for roots, the parent's depth + 1 otherwise, capped at
    // c_MaxHierarchyDepth; deeper entities share the last bucket and take a serial path there.
    struct HierarchyDepth
    {
        uint32_t depth;
    };
    DONUT_ECS_COMPONENT_EX(HierarchyDepth, "0375a6be-bb69-4fcf-a1ff-66d3bc2f2cd6",
                           ::donut::engine::ComponentFlags::Shared, 0, DONUT_ECS_FIELDS(depth))
    constexpr uint32_t c_MaxHierarchyDepth = 63;

    // The local transform as SceneGraphNode::SetTransform writes it (80 B). A node that never set a transform has
    // none: its world transform is its parent's (today's m_HasLocalTransform).
    struct LocalTransform
    {
        dm::double3 translation;
        dm::dquat rotation;
        dm::double3 scaling;
    };
    DONUT_ECS_COMPONENT(LocalTransform, "878ac4de-6a90-45b0-86de-208dc125f353",
                        DONUT_ECS_FIELDS(translation, rotation, scaling))

    // scaling * rotation * translation of LocalTransform, on every entity that has one: a cache that transform
    // propagation refreshes in the chunks whose LocalTransform changed, so that a row whose parent moved but whose
    // own transform did not costs one multiplication. Derived, so snapshots and hashes leave it out.
    struct LocalMatrix
    {
        dm::daffine3 transform;
    };
    DONUT_ECS_COMPONENT_EX(LocalMatrix, "02942eba-2cf8-474e-8624-b1879dff99d9",
                           ::donut::engine::ComponentFlags::NoSerialize, 0, DONUT_ECS_FIELDS(transform))

    // Written by transform propagation (D4: double precision; float copies are made on the call).
    struct WorldTransform
    {
        dm::daffine3 transform;
    };
    DONUT_ECS_COMPONENT(WorldTransform, "50dc1ddf-7655-4b7b-bd19-92f24cb7f037", DONUT_ECS_FIELDS(transform))

    // The leaf's GetLocalBoundingBox, taken when the leaf is attached and by SceneGraphNode::InvalidateContent.
    struct LocalBounds
    {
        dm::box3 bounds;
    };
    DONUT_ECS_COMPONENT(LocalBounds, "9571d52d-6af4-4057-b5f9-9180f39a4bda", DONUT_ECS_FIELDS(bounds))

    // LocalBounds times the float world transform (empty for an empty local box); written by the Bounds system.
    struct WorldBounds
    {
        dm::box3 bounds;
    };
    DONUT_ECS_COMPONENT(WorldBounds, "6753835c-edcc-448c-b658-ccae15da15fc", DONUT_ECS_FIELDS(bounds))

    // The union of a chunk's WorldBounds; written by the Bounds system.
    struct ChunkWorldBounds
    {
        dm::box3 bounds;
    };
    DONUT_ECS_COMPONENT_EX(ChunkWorldBounds, "a9579fb4-f3a0-4039-a8ac-0caa535fc6ad",
                           ::donut::engine::ComponentFlags::Chunk, 0, DONUT_ECS_FIELDS(bounds))

    // The leaf object of an entity (4.11). A strong reference, like SceneNode.
    struct LeafObject
    {
        SceneGraphLeaf* object = nullptr;

        LeafObject() noexcept = default;
        explicit LeafObject(SceneGraphLeaf* adopted) noexcept : object(adopted) { }   // takes over one reference
        LeafObject(const LeafObject& other) noexcept;
        LeafObject(LeafObject&& other) noexcept : object(other.object) { other.object = nullptr; }
        LeafObject& operator=(const LeafObject& other) noexcept;
        LeafObject& operator=(LeafObject&& other) noexcept;
        ~LeafObject();
    };
    DONUT_ECS_COMPONENT(LeafObject, "636cbb39-e668-4872-b6b0-a5f7378e5e01", DONUT_ECS_FIELDS())

    // One tag per leaf class in the class chain (4.11), so that queries select leaf kinds without casts. A SpotLight
    // gets LightLeaf and SpotLightLeaf; SceneGraphLeaf itself is LeafObject.
    struct MeshInstanceLeaf { };
    DONUT_ECS_TAG(MeshInstanceLeaf, "4a208a7b-2b20-488e-9824-9b49a44aa969")
    struct SkinnedMeshInstanceLeaf { };
    DONUT_ECS_TAG(SkinnedMeshInstanceLeaf, "6fcecf2a-be0d-47df-89a5-bc0323d39211")
    struct SkinnedMeshReferenceLeaf { };
    DONUT_ECS_TAG(SkinnedMeshReferenceLeaf, "a1dd7c5b-526b-43a0-959a-71942c60f097")
    struct CameraLeaf { };
    DONUT_ECS_TAG(CameraLeaf, "c382fb5b-acd6-4102-ad00-c482fe1eeb6f")
    struct PerspectiveCameraLeaf { };
    DONUT_ECS_TAG(PerspectiveCameraLeaf, "c40bb0f3-084b-42ea-b0dd-adc42fe68518")
    struct OrthographicCameraLeaf { };
    DONUT_ECS_TAG(OrthographicCameraLeaf, "877a10ef-7d85-40b7-9ceb-ff747c23c6ef")
    struct LightLeaf { };
    DONUT_ECS_TAG(LightLeaf, "48b7a385-e8d4-47f6-84b8-d3ce437ae8de")
    struct DirectionalLightLeaf { };
    DONUT_ECS_TAG(DirectionalLightLeaf, "b6ae4b04-23cb-4f6f-bb90-b7176c0a4ec1")
    struct SpotLightLeaf { };
    DONUT_ECS_TAG(SpotLightLeaf, "c2d20964-14c9-4577-bd9e-844102660524")
    struct PointLightLeaf { };
    DONUT_ECS_TAG(PointLightLeaf, "7d2a8c83-fcab-4177-9c95-9e8fda778a3d")
    struct AnimationLeaf { };
    DONUT_ECS_TAG(AnimationLeaf, "7e13c6c4-00cc-42c3-a5aa-2befa3723396")

    // What a leaf class keeps in columns (D19). One static instance per leaf class, returned by
    // SceneGraphLeaf::GetTypeDesc(); read once per leaf when it is attached, never per entity in a system.
    struct LeafTypeDesc
    {
        nvrhi::GUID classId;                            // NVRHI_CLASS_CLSID of the leaf class
        const ComponentTypeDesc* const* tags;           // this class and its base classes (4.11)
        uint32_t tagCount;
        const ComponentTypeDesc* const* columns;        // state that lives in columns; empty for object-only leaves
        uint32_t columnCount;
    };

    // The state of an object that is in no graph (5.3): owned by the object, moved into rows when it is attached and
    // back when it is detached. Defined in the engine's private sources.
    class OrphanRecord;
}
