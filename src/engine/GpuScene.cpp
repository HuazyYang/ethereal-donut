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

// The GPU scene (plan step E4); see GpuScene.h. The instance pass reads the mesh-instance chunks through the archetype
// storage, as the scene graph's systems do (SceneGraphStore.h): it needs the chunk versions of several columns at once
// and must not stamp what it reads.

#include <donut/engine/GpuScene.h>
#include <donut/engine/SceneGraph.h>
#include <donut/core/log.h>

#include "SceneGraphStore.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>
#include <donut/shaders/gpu_scene_scatter_cb.h>

namespace donut::engine
{
    static_assert(InstanceFlags_CurveDisjointOrthogonalTriangleStrips == c_InstanceFlagCurveDisjointOrthogonalTriangleStrips,
                  "RenderFlags::instanceFlags follows bindless.h");
    static_assert(InstanceFlags_CurveLinearSweptSpheres == c_InstanceFlagCurveLinearSweptSpheres,
                  "RenderFlags::instanceFlags follows bindless.h");
    static_assert(sizeof(InstanceData) == 112 && sizeof(GeometryData) == 64, "layouts of bindless.h");
    // The scatter kernel's view of InstanceData (gpu_scene_scatter_cb.h).
    static_assert(sizeof(InstanceData) == GPU_SCENE_INSTANCE_ROW_WORDS * 4, "gpu_scene_scatter_cb.h row size");
    static_assert(offsetof(InstanceData, transform) == GPU_SCENE_ROW_TRANSFORM_WORD * 4, "gpu_scene_scatter_cb.h");
    static_assert(offsetof(InstanceData, prevTransform) == GPU_SCENE_ROW_PREV_TRANSFORM_WORD * 4, "gpu_scene_scatter_cb.h");
    static_assert(sizeof(float3x4) == GPU_SCENE_TRANSFORM_WORDS * 4, "gpu_scene_scatter_cb.h transform size");
    static_assert(sizeof(GpuSceneTransformRecord) == 64 && sizeof(GpuSceneRowRecord) == 128, "record sizes");

    namespace
    {
        using namespace ecs;

        constexpr uint32_t c_BufferCount = uint32_t(GpuSceneBuffer::Count);
        // The head of an InstanceData row: flags, firstGeometryInstanceIndex, firstGeometryIndex, numGeometries.
        constexpr size_t c_HeadBytes = offsetof(InstanceData, transform);
        // Scatter mode: candidate rows below this count are classified on the calling thread.
        constexpr uint32_t c_ScatterParallelRows = 4096;

        struct Run
        {
            uint32_t first;
            uint32_t count;
        };

        bool SameRow(const void* a, const void* b, size_t size) noexcept
        {
            return std::memcmp(a, b, size) == 0;
        }

        bool SameTransform(const float3x4& a, const float3x4& b) noexcept
        {
            return std::memcmp(&a, &b, sizeof(float3x4)) == 0;
        }

        // An array of trivial elements that keeps its capacity and does not initialise what it grows by (the scatter
        // records are rewritten every update; std::vector::resize would zero them first).
        template <typename T>
        class PodArray
        {
        public:
            static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
            // Throws std::bad_alloc; the contents up to the old size are kept.
            void Resize(size_t size)
            {
                if (size > m_Capacity)
                {
                    const size_t capacity = std::max(size, std::max<size_t>(m_Capacity * 2, 64));
                    std::unique_ptr<T[]> data(new T[capacity]);
                    if (m_Size)
                        std::memcpy(data.get(), m_Data.get(), m_Size * sizeof(T));
                    m_Data = std::move(data);
                    m_Capacity = capacity;
                }
                m_Size = size;
            }
            void Clear() noexcept { m_Size = 0; }
            void Swap(PodArray& other) noexcept
            {
                std::swap(m_Data, other.m_Data);
                std::swap(m_Size, other.m_Size);
                std::swap(m_Capacity, other.m_Capacity);
            }
            [[nodiscard]] T* Data() noexcept { return m_Data.get(); }
            [[nodiscard]] const T* Data() const noexcept { return m_Data.get(); }
            [[nodiscard]] size_t Size() const noexcept { return m_Size; }
            T& operator[](size_t i) noexcept { return m_Data[i]; }
            const T& operator[](size_t i) const noexcept { return m_Data[i]; }

        private:
            std::unique_ptr<T[]> m_Data;
            size_t m_Size = 0;
            size_t m_Capacity = 0;
        };

        // Scatter mode: the kind of record a candidate row became (ScatterMark).
        enum class ScatterKind : uint32_t
        {
            Transform = 1,   // only the transform changed: prevTransform = transform, then the new transform
            RowNew = 2,      // an entity new to its slot (or a resend): the whole row, prevTransform = transform
            RowKeep = 3,     // the same entity, its head changed: the whole row, prevTransform = the slot's transform
        };

        struct ScatterMark
        {
            uint32_t slot;
            ScatterKind kind;
        };

        // A changed mesh-instance chunk of a scatter-mode update, with its columns and its ranges of the outputs.
        struct ScatterChunk
        {
            const Entity* entities;
            const GpuInstanceSlot* slots;
            const MeshRef* refs;
            const RenderFlags* flags;
            const WorldTransform* worlds;
            const uint64_t* rowVersions;   // E7.5: only WorldTransform changed, by these rows (newer than `since`)
            uint64_t since;
            uint32_t count;
            bool headCheck;            // MeshRef, RenderFlags, GpuInstanceSlot or rows changed: compare the heads too
            uint32_t markOffset;       // its candidate rows' range in the marks
            uint32_t markCount;        // the records it made
            uint32_t transformCount;
            uint32_t rowCount;
            uint32_t keepCount;        // RowKeep records (their slots settle next frame, as the transform records')
            uint32_t transformOffset;
            uint32_t rowOffset;
            uint32_t settleOffset;     // in the next frame's settle candidates
        };
    }

    struct GpuScene::Impl
    {
        nvrhi::DeviceHandle device;
        GpuSceneDesc desc;

        // CPU images, by slot.
        std::vector<InstanceData> instances;
        std::vector<Entity> instanceEntities;   // the entity whose row a slot holds
        std::vector<uint64_t> instanceFrames;   // the frame (refresh-call count) of the slot's last rebuild
        std::vector<GeometryData> geometries;
        std::vector<MaterialConstants> materials;

        // What the next upload writes.
        std::vector<Run> pending[c_BufferCount];
        bool rewrite[c_BufferCount] = {};        // the image grew: the buffer is created again and written whole
        bool writeAll[c_BufferCount] = {};       // another graph: every row is written again into the same buffer
        uint32_t bufferRows[c_BufferCount] = {};
        // Dirty or new materials: the values for their own constant buffers, taken when the update saw them (with
        // this graph's slot as materialID), one entry per slot: an update that is not uploaded is overwritten by the
        // next, and a slot that another material takes before the upload names that material.
        struct PendingConstants
        {
            nvrhi::AutoPtr<Material> material;
            MaterialConstants value;
        };
        std::vector<PendingConstants> pendingConstants;
        std::vector<uint32_t> pendingBySlot;     // per material slot: its entry in pendingConstants, or ~0u
        bool assetsInvalid = false;              // InvalidateAssets: every material and geometry row is filled again
        // An update ran out of memory: the next one builds every row and every buffer is written whole, so that rows
        // changed in the image before the failure (and not yet marked) reach the buffers.
        bool recover = false;

        // Rows written with prevTransform != transform, by frame: settled by the first update of a later frame.
        std::vector<uint32_t> settle;
        std::vector<uint32_t> settleNow;

        // What the last update saw.
        uint64_t graphSerial = 0;
        uint64_t frame = ~uint64_t(0);
        uint64_t layoutSerial = 0;
        uint64_t since = 0;
        uint64_t structuralVersion = 0;
        uint64_t meshSerial = 0;
        uint64_t materialSerial = 0;

        nvrhi::BufferHandle buffers[c_BufferCount];
        GpuSceneStats stats;

        // ------------------------------------------------------------------- scatter mode
        // instanceFrames holds, per slot, the serial of the update that last gave it a record. slotTransforms holds the
        // transform each slot was given last, compact (48 bytes, not the image's 112-byte stride): the classification
        // reads it for every candidate row, and with WorldTransform it is that pass's memory traffic.
        std::vector<float3x4> slotTransforms;
        uint64_t scatterSerial = 0;              // of the last scatter-mode update
        uint64_t ackedSerial = 0;                // the last serial CommitScatter acknowledged
        bool scatterResend = false;              // the last update's records are a resend
        std::vector<ScatterChunk> scatterChunks;
        PodArray<ScatterMark> marks;
        PodArray<uint32_t> settleRecords;
        PodArray<GpuSceneTransformRecord> transformRecords;
        PodArray<GpuSceneRowRecord> rowRecords;
        // Slots whose prevTransform the last records may have left behind their transform (transform and RowKeep
        // records): the next frame's settle candidates.
        PodArray<uint32_t> settlePending;
        PodArray<uint32_t> settleNext;
        // CreateBuffers: what it did, the replaced instance buffer kept for the growth copy, and the geometry and
        // material buffers made since the last UploadAssetsTo (whose writer is told with ResizeBuffer).
        GpuSceneBufferChanges changes;
        nvrhi::BufferHandle previousInstanceBuffer;
        bool assetsResized[c_BufferCount] = {};

        void Reset() noexcept;
        void Ensure(GpuSceneBuffer buffer, uint32_t end);
        void Mark(GpuSceneBuffer buffer, uint32_t row);
        void UpdateMaterials(SceneGraph& graph, bool structureChanged, bool allConstants = false);
        void FillGeometries(const MeshTable& meshes, uint32_t slot);
        void UpdateGeometries(SceneGraph& graph, bool structureChanged);
        void UpdateInstances(SceneGraphStore& store, SceneGraph& graph, bool all);
        void UpdateScatter(SceneGraphStore& store, SceneGraph& graph, bool resend, bool layoutChanged);
        void ClassifyScatterChunk(const MeshTable& meshes, ScatterChunk& chunk, uint64_t serial, bool resend) noexcept;
        void WriteScatterChunk(const ScatterChunk& chunk) noexcept;
        bool WriteBuffer(IGpuSceneWriter& writer, GpuSceneBuffer buffer, const void* rows, size_t rowSize);
        void WriteRuns(IGpuSceneWriter& writer, GpuSceneBuffer buffer, const uint8_t* bytes, size_t rowSize);
        void WriteConstants(IGpuSceneWriter& writer);
        void PushConstants(Material* material, uint32_t slot, const MaterialConstants& value);
    };

    void GpuScene::Impl::Reset() noexcept
    {
        std::fill(instanceEntities.begin(), instanceEntities.end(), Entity::Null);
        std::fill(instanceFrames.begin(), instanceFrames.end(), ~uint64_t(0));
        for (uint32_t b = 0; b < c_BufferCount; ++b)
        {
            pending[b].clear();
            writeAll[b] = bufferRows[b] != 0;   // what the buffers hold belongs to another graph: written again
        }
        pendingConstants.clear();
        std::fill(pendingBySlot.begin(), pendingBySlot.end(), ~0u);
        settle.clear();
        settlePending.Clear();   // scatter mode: the next update is a resend (since is 0)
        frame = ~uint64_t(0);
        layoutSerial = 0;
        since = 0;
        structuralVersion = 0;
        meshSerial = 0;
        materialSerial = 0;
    }

    // The image of `buffer` holds at least `end` rows; when it grows, the buffer is created again and written whole.
    void GpuScene::Impl::Ensure(GpuSceneBuffer buffer, uint32_t end)
    {
        // The arrays of one image always have the same size: every one is reserved before any grows, so that running
        // out of memory leaves them as they were.
        size_t current = 0;
        switch (buffer)
        {
        case GpuSceneBuffer::Instances:
            current = std::min(instances.size(), std::min(instanceEntities.size(), instanceFrames.size()));
            if (desc.uploadMode == GpuSceneUploadMode::Scatter)
                current = std::min(current, slotTransforms.size());
            break;
        case GpuSceneBuffer::Geometries: current = geometries.size(); break;
        default: current = std::min(materials.size(), pendingBySlot.size()); break;
        }
        if (end <= current)
            return;
        const uint32_t granularity = std::max(1u, desc.rowGranularity);
        const size_t rows = (size_t(end) + granularity - 1) / granularity * granularity;
        switch (buffer)
        {
        case GpuSceneBuffer::Instances:
            instances.reserve(rows);
            instanceEntities.reserve(rows);
            instanceFrames.reserve(rows);
            if (desc.uploadMode == GpuSceneUploadMode::Scatter)
            {
                slotTransforms.reserve(rows);
                slotTransforms.resize(rows, float3x4(0.f));
            }
            instances.resize(rows, InstanceData());
            instanceEntities.resize(rows, Entity::Null);
            instanceFrames.resize(rows, ~uint64_t(0));
            break;
        case GpuSceneBuffer::Geometries: geometries.resize(rows, GeometryData()); break;
        default:
            materials.reserve(rows);
            pendingBySlot.reserve(rows);
            materials.resize(rows, MaterialConstants());
            pendingBySlot.resize(rows, ~0u);
            break;
        }
        // The rows already in the image stay valid; the buffer is made again and written whole from it (a material
        // image without a material buffer only feeds the per-material constant buffers).
        if (buffer != GpuSceneBuffer::Materials || desc.geometryAndMaterialBuffers)
            rewrite[uint32_t(buffer)] = true;
        pending[uint32_t(buffer)].clear();
    }

    void GpuScene::Impl::Mark(GpuSceneBuffer buffer, uint32_t row)
    {
        std::vector<Run>& runs = pending[uint32_t(buffer)];
        if (rewrite[uint32_t(buffer)] || writeAll[uint32_t(buffer)])
            return;   // the whole buffer is written anyway
        if (!runs.empty() && runs.back().first + runs.back().count == row)
            ++runs.back().count;
        else
            runs.push_back(Run{ row, 1 });
    }

    // Materials: the rows of new materials, of dirty ones, and after structural changes of all of them (as
    // RefreshBuffers refilled them, which also catches textures that arrived without a dirty flag); a row is kept for
    // the upload only when it changed. The material table polls the dirty flag (design 4.8: linear in materials).
    void GpuScene::Impl::UpdateMaterials(SceneGraph& graph, bool structureChanged, bool allConstants)
    {
        const MaterialTable& table = graph.GetMaterials();
        const bool all = structureChanged || allConstants;
        const uint32_t end = table.GetSlotEnd();
        for (uint32_t slot = 0; slot < end; ++slot)
        {
            const MaterialTable::Row& row = table.GetRow(slot);
            Material* material = row.object;
            if (!material)
                continue;
            // A dirty material, one new to its slot (its constant buffer may hold another slot), and one whose constant
            // buffer a device has not made yet get their constant buffers written.
            const bool added = row.serial > materialSerial;
            const bool constants = allConstants || material->dirty || added || (device && !material->materialConstants);
            if (!all && !constants)
                continue;
            MaterialConstants value = MaterialConstants();   // value-initialised: zero (the braces would leave its float3s)
            material->FillConstantBuffer(value, desc.resourceDescriptorHeapBindless);
            value.materialID = int(slot);   // this graph's slot, whatever another graph wrote into the material
            if (!SameRow(&value, &materials[slot], sizeof(value)))
            {
                materials[slot] = value;
                if (desc.geometryAndMaterialBuffers)
                    Mark(GpuSceneBuffer::Materials, slot);
                ++stats.materialRowsChanged;
            }
            if (constants)
                PushConstants(material, slot, value);
        }
        materialSerial = table.GetSerial();
    }

    void GpuScene::Impl::PushConstants(Material* material, uint32_t slot, const MaterialConstants& value)
    {
        uint32_t& entry = pendingBySlot[slot];
        if (entry == ~0u)
        {
            pendingConstants.push_back(PendingConstants{ material, value });
            entry = uint32_t(pendingConstants.size() - 1);
        }
        else
        {
            pendingConstants[entry].material = material;
            pendingConstants[entry].value = value;
        }
    }

    void GpuScene::Impl::FillGeometries(const MeshTable& meshes, uint32_t slot)
    {
        const MeshInfo* mesh = meshes.GetRow(slot).object;
        const MeshTable::MeshRow& meshRow = meshes.GetMeshRow(slot);
        const BufferGroup* buffers = mesh->buffers;
        for (uint32_t i = 0; i < meshRow.geometryCount && i < mesh->geometries.size(); ++i)
        {
            const MeshGeometry* geometry = mesh->geometries[i];
            const uint32_t index = meshRow.firstGeometry + i;
            // Scene::UpdateGeometry's row (TODO there: 64-bit buffer offsets); a mesh without buffers gets no offsets
            GeometryData row = GeometryData();
            const uint32_t indexOffset = mesh->indexOffset + geometry->indexOffsetInMesh;
            const uint32_t vertexOffset = mesh->vertexOffset + geometry->vertexOffsetInMesh;
            auto offset = [&](VertexAttribute attribute, size_t elementSize) -> uint32_t
            {
                return buffers && buffers->hasAttribute(attribute)
                    ? uint32_t(vertexOffset * elementSize + buffers->getVertexBufferRange(attribute).byteOffset)
                    : ~0u;
            };
            row.numIndices = geometry->numIndices;
            row.numVertices = geometry->numVertices;
            row.indexBufferIndex = buffers && buffers->indexBufferDescriptor ? buffers->indexBufferDescriptor->Get() : -1;
            row.indexOffset = indexOffset * uint32_t(sizeof(uint32_t));
            row.vertexBufferIndex = buffers && buffers->vertexBufferDescriptor ? buffers->vertexBufferDescriptor->Get() : -1;
            row.positionOffset = offset(VertexAttribute::Position, sizeof(float3));
            row.prevPositionOffset = offset(VertexAttribute::PrevPosition, sizeof(float3));
            row.texCoord1Offset = offset(VertexAttribute::TexCoord1, sizeof(float2));
            row.texCoord2Offset = offset(VertexAttribute::TexCoord2, sizeof(float2));
            row.normalOffset = offset(VertexAttribute::Normal, sizeof(uint32_t));
            row.tangentOffset = offset(VertexAttribute::Tangent, sizeof(uint32_t));
            row.curveRadiusOffset = offset(VertexAttribute::CurveRadius, sizeof(float));
            // this graph's slot of the material (MeshRow::materials), not the shared object's materialID
            const MaterialId material = i < meshRow.materials.size() ? meshRow.materials[i] : MaterialId::Invalid;
            row.materialIndex = material != MaterialId::Invalid ? MaterialTable::SlotOf(material) : ~0u;
            if (!SameRow(&row, &geometries[index], sizeof(row)))
            {
                geometries[index] = row;
                Mark(GpuSceneBuffer::Geometries, index);
                ++stats.geometryRowsChanged;
            }
        }
    }

    // Geometries: the rows of new or reconciled meshes, of all meshes after structural changes (their buffers'
    // descriptors are made then, by Scene::CreateMeshBuffers), and of meshes whose index counts changed (the poll
    // RefreshBuffers did every frame).
    void GpuScene::Impl::UpdateGeometries(SceneGraph& graph, bool structureChanged)
    {
        const MeshTable& meshes = graph.GetMeshes();
        const bool all = structureChanged;
        const uint32_t end = meshes.GetSlotEnd();
        for (uint32_t slot = 0; slot < end; ++slot)
        {
            const MeshTable::Row& row = meshes.GetRow(slot);
            const MeshInfo* mesh = row.object;
            if (!mesh)
                continue;
            bool fill = all || row.serial > meshSerial;
            if (!fill)
            {
                const MeshTable::MeshRow& meshRow = meshes.GetMeshRow(slot);
                for (uint32_t i = 0; i < meshRow.geometryCount && i < mesh->geometries.size() && !fill; ++i)
                    fill = mesh->geometries[i]->numIndices != geometries[meshRow.firstGeometry + i].numIndices;
            }
            if (fill)
                FillGeometries(meshes, slot);
        }
        meshSerial = meshes.GetSerial();
    }

    void GpuScene::Impl::UpdateInstances(SceneGraphStore& store, SceneGraph& graph, bool all)
    {
        const MeshTable& meshes = graph.GetMeshes();
        const uint64_t currentFrame = store.GetRefreshCalls();
        const bool newFrame = currentFrame != frame;
        if (newFrame)
        {
            settleNow.swap(settle);   // rows of earlier frames to settle after this pass
            settle.clear();
        }
        else
            settleNow.clear();
        // Taken now: an update that fails below is retried in the same frame with every row (Impl::recover), which
        // settles the rows of earlier frames by rebuilding them.
        frame = currentFrame;

        store.SyncArchetypes();
        WorldImpl& world = *store.m_Impl;
        const uint32_t archetypeCount = world.GetArchetypeStorageCount();
        for (uint32_t a = 0; a < archetypeCount; ++a)
        {
            const Archetype& archetype = world.GetArchetypeStorage(a);
            const uint8_t slotColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotInstance);
            const uint8_t refColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotMeshRef);
            const uint8_t flagsColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotRenderFlags);
            const uint8_t worldColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotWorld);
            if (slotColumn == SceneGraphStore::c_Absent || refColumn == SceneGraphStore::c_Absent ||
                flagsColumn == SceneGraphStore::c_Absent || worldColumn == SceneGraphStore::c_Absent)
                continue;
            const uint32_t added = archetype.AddedVersionSlot();
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                ++stats.chunksVisited;
                const uint64_t* versions = archetype.Versions(c);
                const bool head = all || versions[slotColumn] > since || versions[refColumn] > since ||
                                  versions[flagsColumn] > since || versions[added] > since;
                if (!head && versions[worldColumn] <= since)
                    continue;
                ++stats.chunksChanged;
                // E7.5: a chunk whose WorldTransform alone changed, with row versions: only the rows written since
                const uint64_t* rowVersions = head ? nullptr : ChangedRowVersions(archetype, c, worldColumn);

                uint8_t* memory = archetype.chunks[c].memory;
                const Entity* entities = archetype.Entities(c);
                const auto* slots = reinterpret_cast<const GpuInstanceSlot*>(memory + archetype.columns[slotColumn].offset);
                const auto* refs = reinterpret_cast<const MeshRef*>(memory + archetype.columns[refColumn].offset);
                const auto* flags = reinterpret_cast<const RenderFlags*>(memory + archetype.columns[flagsColumn].offset);
                const auto* worlds = reinterpret_cast<const WorldTransform*>(memory + archetype.columns[worldColumn].offset);
                const uint32_t count = archetype.chunks[c].count;
                for (uint32_t r = 0; r < count; ++r)
                {
                    if (rowVersions && rowVersions[r] <= since)
                        continue;   // as it was: the settle below covers last frame's move
                    const uint32_t slot = slots[r].instance;
                    if (slot >= instances.size())
                        continue;   // cannot happen: Ensure covered the allocator's end
                    ++stats.rowsBuilt;

                    // Scene::UpdateInstance's row, from the columns
                    InstanceData row = InstanceData();
                    row.flags = flags[r].instanceFlags;
                    row.firstGeometryInstanceIndex = slots[r].firstGeometryInstance;
                    const MeshTable::MeshRow* meshRow =
                        refs[r].mesh != MeshId::Invalid ? &meshes.GetMeshRow(MeshTable::SlotOf(refs[r].mesh)) : nullptr;
                    row.numGeometries = meshRow ? meshRow->geometryCount : 0;
                    row.firstGeometryIndex = row.numGeometries > 0 ? meshRow->firstGeometry : ~0u;
                    affineToColumnMajor(affine3(worlds[r].transform), row.transform);

                    // The previous transform: what the slot was uploaded with in an earlier frame, or what this frame
                    // gave it already; the current one for an entity new to the slot.
                    InstanceData& image = instances[slot];
                    const bool sameEntity = instanceEntities[slot] == entities[r];
                    if (!sameEntity)
                        row.prevTransform = row.transform;
                    else if (instanceFrames[slot] == currentFrame)
                        row.prevTransform = image.prevTransform;
                    else
                        row.prevTransform = image.transform;

                    if (instanceFrames[slot] != currentFrame && !SameTransform(row.prevTransform, row.transform))
                        settle.push_back(slot);   // may throw: the image row is not written yet
                    instanceEntities[slot] = entities[r];
                    instanceFrames[slot] = currentFrame;
                    if (!SameRow(&row, &image, sizeof(row)))
                    {
                        image = row;
                        Mark(GpuSceneBuffer::Instances, slot);
                        ++stats.rowsChanged;
                    }
                }
            }
        }

        // Rows that moved in an earlier frame and were not rebuilt now: their previous transform settles.
        if (newFrame)
        {
            for (uint32_t slot : settleNow)
            {
                if (slot >= instances.size() || instanceFrames[slot] == currentFrame)
                    continue;
                InstanceData& image = instances[slot];
                instanceFrames[slot] = currentFrame;
                if (SameTransform(image.prevTransform, image.transform))
                    continue;
                image.prevTransform = image.transform;
                Mark(GpuSceneBuffer::Instances, slot);
                ++stats.rowsSettled;
            }
            settleNow.clear();
        }
    }

    // ------------------------------------------------------------------------------------- scatter mode

    // One candidate row of a changed chunk becomes at most one record. The image keeps each slot's head and float
    // transform (not its prevTransform, which the GPU keeps), so that a row whose head and transform are unchanged
    // costs nothing; the per-slot entity tells a row new to its slot. Chunks write disjoint slots and marks, so the
    // chunks of one update can be classified in parallel.
    void GpuScene::Impl::ClassifyScatterChunk(const MeshTable& meshes, ScatterChunk& chunk, uint64_t serial,
                                              bool resend) noexcept
    {
        ScatterMark* out = marks.Data() + chunk.markOffset;
        uint32_t made = 0, transformCount = 0, rowCount = 0, keepCount = 0;
        const uint32_t slotEnd = uint32_t(instances.size());
        for (uint32_t r = 0; r < chunk.count; ++r)
        {
            if (chunk.rowVersions && chunk.rowVersions[r] <= chunk.since)
                continue;   // as it was: no record (a settle record covers last frame's move)
            const uint32_t slot = chunk.slots[r].instance;
            if (slot >= slotEnd)
                continue;   // cannot happen: Ensure covered the allocator's end

            // Scene::UpdateInstance's row, from the columns: the head, then the transform
            float3x4 transform;
            affineToColumnMajor(affine3(chunk.worlds[r].transform), transform);
            InstanceData& image = instances[slot];
            auto head = [&](InstanceData& row)
            {
                row.flags = chunk.flags[r].instanceFlags;
                row.firstGeometryInstanceIndex = chunk.slots[r].firstGeometryInstance;
                const MeshTable::MeshRow* meshRow = chunk.refs[r].mesh != MeshId::Invalid
                    ? &meshes.GetMeshRow(MeshTable::SlotOf(chunk.refs[r].mesh)) : nullptr;
                row.numGeometries = meshRow ? meshRow->geometryCount : 0;
                row.firstGeometryIndex = row.numGeometries > 0 ? meshRow->firstGeometry : ~0u;
            };

            ScatterKind kind = ScatterKind::Transform;
            if (resend || instanceEntities[slot] != chunk.entities[r])
            {
                // new to the slot: prevTransform = transform, as CPU mode gives an entity new to its slot
                head(image);
                image.transform = transform;
                image.prevTransform = transform;
                slotTransforms[slot] = transform;
                instanceEntities[slot] = chunk.entities[r];
                kind = ScatterKind::RowNew;
                ++rowCount;
            }
            else
            {
                bool headChanged = false;
                if (chunk.headCheck)
                {
                    InstanceData row;
                    head(row);
                    headChanged = !SameRow(&row, &image, c_HeadBytes);
                    if (headChanged)
                        std::memcpy(&image, &row, c_HeadBytes);
                }
                const bool moved = !SameTransform(transform, slotTransforms[slot]);
                if (moved)
                {
                    slotTransforms[slot] = transform;
                    image.transform = transform;
                }
                if (headChanged)
                {
                    kind = ScatterKind::RowKeep;
                    ++rowCount;
                    ++keepCount;
                }
                else if (moved)
                {
                    kind = ScatterKind::Transform;
                    ++transformCount;
                }
                else
                    continue;   // CPU mode rebuilds it to the same row; a settle record covers last frame's move
            }
            instanceFrames[slot] = serial;
            out[made++] = ScatterMark{ slot, kind };
        }
        chunk.markCount = made;
        chunk.transformCount = transformCount;
        chunk.rowCount = rowCount;
        chunk.keepCount = keepCount;
    }

    void GpuScene::Impl::WriteScatterChunk(const ScatterChunk& chunk) noexcept
    {
        const ScatterMark* in = marks.Data() + chunk.markOffset;
        GpuSceneTransformRecord* transformsOut = transformRecords.Data() + chunk.transformOffset;
        GpuSceneRowRecord* rowsOut = rowRecords.Data() + chunk.rowOffset;
        uint32_t* settleOut = settleNext.Data() + chunk.settleOffset;
        for (uint32_t i = 0; i < chunk.markCount; ++i)
        {
            const ScatterMark mark = in[i];
            const InstanceData& image = instances[mark.slot];
            if (mark.kind == ScatterKind::Transform)
            {
                GpuSceneTransformRecord& record = *transformsOut++;
                record.slot = mark.slot;
                record.padding0 = record.padding1 = record.padding2 = 0;
                std::memcpy(record.transform, &image.transform, sizeof(record.transform));
                *settleOut++ = mark.slot;
            }
            else
            {
                GpuSceneRowRecord& record = *rowsOut++;
                record.slot = mark.slot;
                record.flags = mark.kind == ScatterKind::RowKeep ? GPU_SCENE_ROW_FLAG_KEEP_PREVIOUS : 0u;
                record.padding0 = record.padding1 = 0;
                // the head and the transform; prevTransform = transform (the kernel ignores it for RowKeep)
                std::memcpy(record.row, &image, offsetof(InstanceData, prevTransform));
                std::memcpy(record.row + GPU_SCENE_ROW_PREV_TRANSFORM_WORD, &image.transform, sizeof(float3x4));
                if (mark.kind == ScatterKind::RowKeep)
                    *settleOut++ = mark.slot;
            }
        }
    }

    // Scatter mode's instance pass (GpuScene.h): candidate chunks serially, then the rows classified in parallel by
    // chunk, the settle records, the prefix sums of the per-chunk counts, and the records written in parallel at those
    // offsets, so the records are in chunk order at any worker count.
    void GpuScene::Impl::UpdateScatter(SceneGraphStore& store, SceneGraph& graph, bool resend, bool layoutChanged)
    {
        const MeshTable& meshes = graph.GetMeshes();
        const uint64_t currentFrame = store.GetRefreshCalls();
        const bool newFrame = currentFrame != frame;
        frame = currentFrame;
        const uint64_t serial = ++scatterSerial;
        scatterResend = resend;

        // The candidate chunks.
        scatterChunks.clear();
        store.SyncArchetypes();
        WorldImpl& world = *store.m_Impl;
        const uint32_t archetypeCount = world.GetArchetypeStorageCount();
        uint32_t candidateRows = 0;
        for (uint32_t a = 0; a < archetypeCount; ++a)
        {
            const Archetype& archetype = world.GetArchetypeStorage(a);
            const uint8_t slotColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotInstance);
            const uint8_t refColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotMeshRef);
            const uint8_t flagsColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotRenderFlags);
            const uint8_t worldColumn = store.ColumnIndex(archetype, a, SceneGraphStore::SlotWorld);
            if (slotColumn == SceneGraphStore::c_Absent || refColumn == SceneGraphStore::c_Absent ||
                flagsColumn == SceneGraphStore::c_Absent || worldColumn == SceneGraphStore::c_Absent)
                continue;
            const uint32_t added = archetype.AddedVersionSlot();
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                ++stats.chunksVisited;
                const uint32_t count = archetype.chunks[c].count;
                if (count == 0)
                    continue;
                const uint64_t* versions = archetype.Versions(c);
                const bool headCheck = resend || layoutChanged || versions[slotColumn] > since ||
                                       versions[refColumn] > since || versions[flagsColumn] > since ||
                                       versions[added] > since;
                if (!headCheck && versions[worldColumn] <= since)
                    continue;
                ++stats.chunksChanged;
                uint8_t* memory = archetype.chunks[c].memory;
                ScatterChunk chunk = {};
                chunk.entities = archetype.Entities(c);
                chunk.slots = reinterpret_cast<const GpuInstanceSlot*>(memory + archetype.columns[slotColumn].offset);
                chunk.refs = reinterpret_cast<const MeshRef*>(memory + archetype.columns[refColumn].offset);
                chunk.flags = reinterpret_cast<const RenderFlags*>(memory + archetype.columns[flagsColumn].offset);
                chunk.worlds = reinterpret_cast<const WorldTransform*>(memory + archetype.columns[worldColumn].offset);
                // E7.5: a chunk whose WorldTransform alone changed, with row versions: only the rows written since
                chunk.rowVersions = headCheck ? nullptr : ChangedRowVersions(archetype, c, worldColumn);
                chunk.since = since;
                chunk.count = count;
                chunk.headCheck = headCheck;
                chunk.markOffset = candidateRows;
                scatterChunks.push_back(chunk);
                candidateRows += count;
            }
        }
        stats.rowsBuilt = candidateRows;
        marks.Resize(candidateRows);

        // Classify, in parallel by chunk.
        struct Context
        {
            Impl* impl;
            const MeshTable* meshes;
            uint64_t serial;
            bool resend;
            static void Classify(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                const Context& ctx = *static_cast<const Context*>(context);
                for (uint32_t i = begin; i < end; ++i)
                    ctx.impl->ClassifyScatterChunk(*ctx.meshes, ctx.impl->scatterChunks[i], ctx.serial, ctx.resend);
            }
            static void Write(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                const Context& ctx = *static_cast<const Context*>(context);
                for (uint32_t i = begin; i < end; ++i)
                    ctx.impl->WriteScatterChunk(ctx.impl->scatterChunks[i]);
            }
        };
        Context ctx{ this, &meshes, serial, resend };
        const uint32_t chunkCount = uint32_t(scatterChunks.size());
        IJobSystem* jobs = store.m_Jobs;
        uint32_t grain = chunkCount;
        if (jobs && candidateRows >= c_ScatterParallelRows)
        {
            static const uint32_t s_Hardware = std::max(1u, std::thread::hardware_concurrency());
            const uint32_t threads = std::max(1u, std::min(jobs->GetThreadCount(), s_Hardware));
            grain = std::max(1u, chunkCount / (threads * 4));
        }
        auto run = [&](PFN_JobFunction function)
        {
            if (chunkCount == 0)
                return;
            if (grain >= chunkCount)
                function(&ctx, 0, chunkCount, 0);
            else
                jobs->ParallelFor(chunkCount, grain, function, &ctx);
        };
        run(&Context::Classify);

        // Settle records: last frame's candidates without a record now (CPU mode settles them in its first update of
        // a later frame). After a resend every row has prevTransform = transform.
        settleRecords.Clear();
        if (newFrame && !resend)
        {
            settleRecords.Resize(settlePending.Size());
            uint32_t count = 0;
            const uint32_t slotEnd = uint32_t(instances.size());
            for (size_t i = 0; i < settlePending.Size(); ++i)
            {
                const uint32_t slot = settlePending[i];
                if (slot >= slotEnd || instanceFrames[slot] == serial)
                    continue;
                instanceFrames[slot] = serial;   // a slot settles once
                settleRecords[count++] = slot;
            }
            settleRecords.Resize(count);
        }

        // Prefix sums, then the records.
        uint32_t transformTotal = 0, rowTotal = 0, settleTotal = 0;
        for (ScatterChunk& chunk : scatterChunks)
        {
            chunk.transformOffset = transformTotal;
            chunk.rowOffset = rowTotal;
            chunk.settleOffset = settleTotal;
            transformTotal += chunk.transformCount;
            rowTotal += chunk.rowCount;
            settleTotal += chunk.transformCount + chunk.keepCount;
        }
        // An update in the same frame keeps the frame's earlier candidates for the next frame.
        const size_t carried = !newFrame && !resend ? settlePending.Size() : 0;
        transformRecords.Resize(transformTotal);
        rowRecords.Resize(rowTotal);
        settleNext.Resize(settleTotal + carried);
        run(&Context::Write);
        if (carried)
            std::memcpy(settleNext.Data() + settleTotal, settlePending.Data(), carried * sizeof(uint32_t));
        settlePending.Swap(settleNext);

        stats.rowsChanged = transformTotal + rowTotal;
        stats.rowsSettled = uint32_t(settleRecords.Size());
        stats.settleRecords = uint32_t(settleRecords.Size());
        stats.transformRecords = transformTotal;
        stats.rowRecords = rowTotal;
        stats.uploadRuns = 0;
        stats.uploadBytes = uint64_t(settleRecords.Size()) * sizeof(uint32_t) +
                            uint64_t(transformTotal) * sizeof(GpuSceneTransformRecord) +
                            uint64_t(rowTotal) * sizeof(GpuSceneRowRecord);
        stats.buffersResized = 0;
    }

    // False when the buffer could not be made: everything stays to be written, and the next upload tries again.
    bool GpuScene::Impl::WriteBuffer(IGpuSceneWriter& writer, GpuSceneBuffer buffer, const void* rows, size_t rowSize)
    {
        const uint32_t b = uint32_t(buffer);
        uint32_t rowCount = 0;
        switch (buffer)
        {
        case GpuSceneBuffer::Instances: rowCount = uint32_t(instances.size()); break;
        case GpuSceneBuffer::Geometries: rowCount = uint32_t(geometries.size()); break;
        default: rowCount = uint32_t(materials.size()); break;
        }
        const uint8_t* bytes = static_cast<const uint8_t*>(rows);
        if (rewrite[b] || writeAll[b] || bufferRows[b] != rowCount)
        {
            if (rowCount != 0)
            {
                // A buffer is made again only when it is too small, as RefreshBuffers did: binding sets made from it
                // stay valid across a graph switch.
                if (bufferRows[b] < rowCount)
                {
                    if (!writer.ResizeBuffer(buffer, rowCount))
                    {
                        bufferRows[b] = 0;   // the old buffer is gone or too small: made again next time
                        rewrite[b] = true;
                        return false;
                    }
                    bufferRows[b] = rowCount;
                    ++stats.buffersResized;
                }
                writer.WriteRows(buffer, 0, rowCount, bytes, rowSize);
                ++stats.uploadRuns;
                stats.uploadBytes += uint64_t(rowCount) * rowSize;
            }
            rewrite[b] = false;
            writeAll[b] = false;
            pending[b].clear();
            return true;
        }
        WriteRuns(writer, buffer, bytes, rowSize);
        return true;
    }

    // The kept runs of `buffer`, sorted and merged.
    void GpuScene::Impl::WriteRuns(IGpuSceneWriter& writer, GpuSceneBuffer buffer, const uint8_t* bytes, size_t rowSize)
    {
        std::vector<Run>& runs = pending[uint32_t(buffer)];
        if (runs.empty())
            return;
        // Runs come in chunk order, mostly ascending: sort them, then merge overlapping and nearby ones.
        std::sort(runs.begin(), runs.end(), [](const Run& x, const Run& y) { return x.first < y.first; });
        Run current = runs.front();
        auto flush = [&](const Run& run)
        {
            writer.WriteRows(buffer, run.first, run.count, bytes + size_t(run.first) * rowSize, rowSize);
            ++stats.uploadRuns;
            stats.uploadBytes += uint64_t(run.count) * rowSize;
        };
        for (size_t i = 1; i < runs.size(); ++i)
        {
            const Run& next = runs[i];
            const uint32_t currentEnd = current.first + current.count;
            if (next.first <= currentEnd + desc.mergeGap)
                current.count = std::max(currentEnd, next.first + next.count) - current.first;
            else
            {
                flush(current);
                current = next;
            }
        }
        flush(current);
        runs.clear();
    }

    void GpuScene::Impl::WriteConstants(IGpuSceneWriter& writer)
    {
        for (const PendingConstants& entry : pendingConstants)
        {
            writer.WriteMaterialConstants(entry.material, entry.value);
            entry.material->dirty = false;
        }
        pendingConstants.clear();
        std::fill(pendingBySlot.begin(), pendingBySlot.end(), ~0u);
    }

    // ------------------------------------------------------------------------------------- the device

    namespace
    {
        // Scene::CreateInstanceBuffer, CreateGeometryBuffer and CreateMaterialBuffer. `handle` is replaced (the old
        // buffer is released through it, so a caller that keeps the old one holds its own reference).
        bool CreateSceneBuffer(nvrhi::IDevice* device, const GpuSceneDesc& desc, GpuSceneBuffer buffer, uint32_t rows,
                               nvrhi::BufferHandle& handle) noexcept
        {
            nvrhi::BufferDesc bufferDesc;
            bufferDesc.canHaveRawViews = true;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            // Scatter mode: the render graph imports them (ImportBuffer rejects keepInitialState buffers), with
            // ShaderResource as their state between frames (c_GpuSceneRestingState).
            bufferDesc.keepInitialState = desc.uploadMode != GpuSceneUploadMode::Scatter;
            switch (buffer)
            {
            case GpuSceneBuffer::Instances:
                bufferDesc.byteSize = sizeof(InstanceData) * uint64_t(rows);
                bufferDesc.debugName = "Instances";
                bufferDesc.structStride = desc.structuredInstanceBuffer ? sizeof(InstanceData) : 0;
                bufferDesc.isVertexBuffer = true;
                break;
            case GpuSceneBuffer::Geometries:
                bufferDesc.byteSize = sizeof(GeometryData) * uint64_t(rows);
                bufferDesc.debugName = "BindlessGeometry";
                bufferDesc.structStride = sizeof(GeometryData);
                break;
            default:
                bufferDesc.byteSize = sizeof(MaterialConstants) * uint64_t(rows);
                bufferDesc.debugName = "BindlessMaterials";
                bufferDesc.structStride = sizeof(MaterialConstants);
                break;
            }
            handle.Reset();
            if (NVRHI_FAILED(device->createBuffer(bufferDesc, &handle)))
            {
                log::warning("GpuScene: cannot create the %s buffer (%u rows)", bufferDesc.debugName.c_str(), rows);
                return false;
            }
            return true;
        }

        class DeviceWriter final : public IGpuSceneWriter
        {
        public:
            // `createBuffers` false (scatter mode's UploadAssets): CreateBuffers made them, ResizeBuffer only reports
            // whether the buffer exists.
            DeviceWriter(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, const GpuSceneDesc& desc,
                         nvrhi::BufferHandle* buffers, bool createBuffers = true) noexcept
                : m_Device(device), m_CommandList(commandList), m_Desc(desc), m_Buffers(buffers),
                  m_CreateBuffers(createBuffers) { }

            // Scene::CreateInstanceBuffer, CreateGeometryBuffer and CreateMaterialBuffer.
            bool ResizeBuffer(GpuSceneBuffer buffer, uint32_t rows) noexcept override
            {
                if (!m_CreateBuffers)
                    return m_Buffers[uint32_t(buffer)] != nullptr;
                return CreateSceneBuffer(m_Device, m_Desc, buffer, rows, m_Buffers[uint32_t(buffer)]);
            }

            void WriteRows(GpuSceneBuffer buffer, uint32_t firstRow, uint32_t rowCount, const void* data,
                           size_t rowSize) noexcept override
            {
                nvrhi::IBuffer* target = m_Buffers[uint32_t(buffer)];
                if (target)
                    m_CommandList->writeBuffer(target, data, size_t(rowCount) * rowSize, uint64_t(firstRow) * rowSize);
            }

            // Scene::CreateMaterialConstantBuffer, then the write RefreshBuffers made for a dirty material.
            void WriteMaterialConstants(Material* material, const MaterialConstants& constants) noexcept override
            {
                if (!material->materialConstants)
                {
                    nvrhi::BufferDesc bufferDesc;
                    bufferDesc.byteSize = sizeof(MaterialConstants);
                    bufferDesc.debugName = material->name;
                    bufferDesc.isConstantBuffer = true;
                    bufferDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
                    bufferDesc.keepInitialState = true;
                    if (NVRHI_FAILED(m_Device->createBuffer(bufferDesc, &material->materialConstants)))
                    {
                        log::warning("GpuScene: cannot create the constant buffer of material '%s'", material->name.c_str());
                        return;
                    }
                }
                m_CommandList->writeBuffer(material->materialConstants, &constants, sizeof(MaterialConstants));
            }

        private:
            nvrhi::IDevice* m_Device;
            nvrhi::ICommandList* m_CommandList;
            const GpuSceneDesc& m_Desc;
            nvrhi::BufferHandle* m_Buffers;
            bool m_CreateBuffers;
        };
    }

    // ------------------------------------------------------------------------------------- GpuScene

    GpuScene::GpuScene(nvrhi::IDevice* device, const GpuSceneDesc& desc) noexcept
        : m_Device(device), m_Desc(desc)
    {
    }

    GpuScene::~GpuScene() = default;

    // The state is made on first use, so that construction cannot fail; out of memory there is reported by the call.
    bool GpuScene::EnsureImpl() noexcept
    {
        if (m_Impl)
            return true;
        // nothrow covers the block; a member's constructor that allocates may still throw
        try
        {
            m_Impl.reset(new (std::nothrow) Impl());
        }
        catch (const std::bad_alloc&)
        {
            m_Impl.reset();
        }
        if (!m_Impl)
        {
            log::warning("GpuScene: out of memory");
            return false;
        }
        m_Impl->device = m_Device;
        m_Impl->desc = m_Desc;
        return true;
    }

    void GpuScene::InvalidateAssets() noexcept
    {
        m_Recording.Check("GpuScene::InvalidateAssets");
        if (m_Impl)
            m_Impl->assetsInvalid = true;
    }

    // ------------------------------------------------------------------- pipelined recording (E7.8)

    namespace
    {
        uint64_t CurrentThreadKey() noexcept
        {
            // never 0, the value of "no recording"
            return uint64_t(std::hash<std::thread::id>()(std::this_thread::get_id())) | 1u;
        }

        // PipelinedRecordingWork's depth on this thread (part I7, backlog I6-R1).
        thread_local uint32_t t_RecordingWork = 0;
    }

    PipelinedRecordingWork::PipelinedRecordingWork() noexcept
    {
        ++t_RecordingWork;
    }

    PipelinedRecordingWork::~PipelinedRecordingWork() noexcept
    {
        --t_RecordingWork;
    }

    bool PipelinedRecordingWork::IsActive() noexcept
    {
        return t_RecordingWork != 0;
    }

    void PipelinedRecordingCheck::Begin() noexcept
    {
        m_Warned.store(false, std::memory_order_relaxed);
        m_MainThread.store(CurrentThreadKey(), std::memory_order_release);
    }

    void PipelinedRecordingCheck::End() noexcept
    {
        m_MainThread.store(0, std::memory_order_release);
    }

    void PipelinedRecordingCheck::OnCheck(const char* operation) const noexcept
    {
        const uint64_t main = m_MainThread.load(std::memory_order_acquire);
        // Recording work the main thread runs while it waits in the shared job system is not the main thread's.
        if (main == 0 || main != CurrentThreadKey() || t_RecordingWork != 0 ||
            m_Warned.exchange(true, std::memory_order_relaxed))
            return;
        log::warning("%s on the main thread while a pipelined frame records (between BeginPipelinedRecording and "
                     "EndPipelinedRecording): the recording owns it until the join; the call runs", operation);
    }

    nvrhi::FRESULT GpuScene::Update(SceneGraph* graph) noexcept
    {
        m_Recording.Check("GpuScene::Update");
        if (!graph || !graph->m_Store)
            return nvrhi::FE_INVALID_ARGS;
        if (!EnsureImpl())
            return nvrhi::FE_OUT_OF_MEMORY;
        Impl& impl = *m_Impl;
        SceneGraphStore& store = *graph->m_Store;
        try
        {
            impl.stats.chunksVisited = impl.stats.chunksChanged = 0;
            impl.stats.rowsBuilt = impl.stats.rowsChanged = impl.stats.rowsSettled = 0;
            impl.stats.geometryRowsChanged = impl.stats.materialRowsChanged = 0;

            if (store.GetSerial() != impl.graphSerial)
            {
                impl.Reset();
                impl.graphSerial = store.GetSerial();
            }
            IWorld* world = store.GetWorld();
            // after structural changes every material and geometry row is filled again; InvalidateAssets asks for it
            // when the mesh buffers changed later than the graph (Scene::CreateMeshBuffers)
            const bool structureChanged = world->GetStructuralVersion() != impl.structuralVersion || impl.assetsInvalid ||
                                          impl.recover;
            const bool layoutChanged = store.GetLayoutSerial() != impl.layoutSerial;
            const bool firstUpdate = impl.since == 0;

            impl.Ensure(GpuSceneBuffer::Instances, store.GetInstanceSlots().GetEnd());
            impl.Ensure(GpuSceneBuffer::Materials, graph->GetMaterials().GetSlotEnd());
            if (impl.desc.geometryAndMaterialBuffers)
                impl.Ensure(GpuSceneBuffer::Geometries, graph->GetMeshes().GetGeometryEnd());

            const bool scatter = impl.desc.uploadMode == GpuSceneUploadMode::Scatter;
            // Scatter mode: records that the upload pass did not acknowledge are lost, and so may be what its Prepare
            // wrote (UploadAssets cleared the materials' dirty flags and pending constants, and may have made a
            // constant buffer it never wrote, if the frame then did not execute): every live row is sent again, the
            // asset buffers are written whole, and every material's constant buffer is written again.
            const bool unacknowledged = scatter && impl.scatterSerial != impl.ackedSerial;
            if (unacknowledged)
            {
                impl.writeAll[uint32_t(GpuSceneBuffer::Geometries)] = impl.desc.geometryAndMaterialBuffers;
                impl.writeAll[uint32_t(GpuSceneBuffer::Materials)] = impl.desc.geometryAndMaterialBuffers;
            }

            impl.UpdateMaterials(*graph, structureChanged, unacknowledged);
            if (impl.desc.geometryAndMaterialBuffers)
                impl.UpdateGeometries(*graph, structureChanged);
            if (scatter)
                impl.UpdateScatter(store, *graph, firstUpdate || impl.recover || unacknowledged, layoutChanged);
            else
                impl.UpdateInstances(store, *graph, firstUpdate || layoutChanged || impl.recover);

            // Every stamp after this is newer: phases advance the version, and the graph's structural operations and
            // InvalidateContent advance it before they stamp.
            impl.since = world->GetVersion();
            impl.structuralVersion = world->GetStructuralVersion();
            impl.layoutSerial = store.GetLayoutSerial();
            impl.assetsInvalid = false;
            impl.recover = false;
        }
        catch (const std::bad_alloc&)
        {
            // Rows may have changed in the images without being marked: the next update builds every row again and
            // the next upload writes every buffer whole (the images keep their sizes, Ensure is all or nothing).
            impl.recover = true;
            for (uint32_t b = 0; b < c_BufferCount; ++b)
                impl.writeAll[b] = true;
            // Scatter mode: no records (the next update is a resend); the pass acknowledges the empty set.
            impl.settleRecords.Clear();
            impl.transformRecords.Clear();
            impl.rowRecords.Clear();
            impl.settlePending.Clear();
            log::warning("GpuScene::Update: out of memory");
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT GpuScene::UploadTo(IGpuSceneWriter* writer) noexcept
    {
        m_Recording.Check("GpuScene::UploadTo");
        if (!writer)
            return nvrhi::FE_INVALID_ARGS;
        if (m_Desc.uploadMode == GpuSceneUploadMode::Scatter)
            return nvrhi::FE_UNSUPPORTED;   // the upload pass scatters; UploadAssetsTo writes the asset rows
        if (!m_Impl)
            return nvrhi::FS_OK;   // nothing was updated yet
        Impl& impl = *m_Impl;
        impl.stats.uploadRuns = 0;
        impl.stats.uploadBytes = 0;
        impl.stats.buffersResized = 0;
        bool ok = impl.WriteBuffer(*writer, GpuSceneBuffer::Instances, impl.instances.data(), sizeof(InstanceData));
        if (impl.desc.geometryAndMaterialBuffers)
        {
            ok &= impl.WriteBuffer(*writer, GpuSceneBuffer::Geometries, impl.geometries.data(), sizeof(GeometryData));
            ok &= impl.WriteBuffer(*writer, GpuSceneBuffer::Materials, impl.materials.data(), sizeof(MaterialConstants));
        }
        impl.WriteConstants(*writer);
        return ok ? nvrhi::FS_OK : nvrhi::FE_OUT_OF_MEMORY;
    }

    nvrhi::FRESULT GpuScene::Upload(nvrhi::ICommandList* commandList) noexcept
    {
        m_Recording.Check("GpuScene::Upload");
        if (!commandList || !m_Device)
            return nvrhi::FE_INVALID_ARGS;
        if (!m_Impl)
            return nvrhi::FS_OK;
        DeviceWriter writer(m_Impl->device, commandList, m_Impl->desc, m_Impl->buffers);
        return UploadTo(&writer);
    }

    // ------------------------------------------------------------------------------------- scatter mode

    nvrhi::FRESULT GpuScene::CreateBuffers() noexcept
    {
        m_Recording.Check("GpuScene::CreateBuffers");
        if (!m_Impl)
            return nvrhi::FS_OK;   // nothing was updated yet
        Impl& impl = *m_Impl;
        impl.changes = GpuSceneBufferChanges();
        // The buffer replaced by the last call: its rows were copied by the frame whose records were acknowledged, or
        // that frame was dropped and the update since then was a resend. The graph and the command lists that used it
        // hold their own references.
        impl.previousInstanceBuffer = nullptr;
        bool ok = true;
        for (uint32_t b = 0; b < c_BufferCount; ++b)
        {
            const GpuSceneBuffer buffer = GpuSceneBuffer(b);
            if (buffer != GpuSceneBuffer::Instances && !impl.desc.geometryAndMaterialBuffers)
                continue;
            uint32_t rows = 0;
            switch (buffer)
            {
            case GpuSceneBuffer::Instances: rows = uint32_t(impl.instances.size()); break;
            case GpuSceneBuffer::Geometries: rows = uint32_t(impl.geometries.size()); break;
            default: rows = uint32_t(impl.materials.size()); break;
            }
            if (rows == 0 || impl.bufferRows[b] >= rows)
                continue;   // never shrunk, as Upload does not shrink them
            nvrhi::BufferHandle old = impl.buffers[b];
            const uint32_t oldRows = impl.bufferRows[b];
            if (m_Device && !CreateSceneBuffer(m_Device, impl.desc, buffer, rows, impl.buffers[b]))
            {
                impl.buffers[b] = old;   // keep the smaller one; the pass does not scatter into it
                ok = false;
                continue;
            }
            impl.bufferRows[b] = rows;
            ++impl.stats.buffersResized;
            if (buffer == GpuSceneBuffer::Instances)
            {
                impl.changes.instancesResized = true;
                // A resend writes every live row: nothing to copy.
                if (!impl.scatterResend && oldRows != 0)
                {
                    impl.previousInstanceBuffer = old;
                    impl.changes.previousInstanceBuffer = old;
                    impl.changes.previousInstanceRows = oldRows;
                }
            }
            else
            {
                // written whole by the next UploadAssets, whose writer hears of the new size first
                impl.rewrite[b] = true;
                impl.pending[b].clear();
                impl.assetsResized[b] = true;
                if (buffer == GpuSceneBuffer::Geometries)
                    impl.changes.geometriesResized = true;
                else
                    impl.changes.materialsResized = true;
            }
        }
        if (!ok)
            log::warning("GpuScene::CreateBuffers: a buffer could not be made");
        return ok ? nvrhi::FS_OK : nvrhi::FE_OUT_OF_MEMORY;
    }

    const GpuSceneBufferChanges& GpuScene::GetBufferChanges() const noexcept
    {
        static const GpuSceneBufferChanges s_None;
        return m_Impl ? m_Impl->changes : s_None;
    }

    nvrhi::FRESULT GpuScene::UploadAssetsTo(IGpuSceneWriter* writer) noexcept
    {
        m_Recording.Check("GpuScene::UploadAssetsTo");
        if (!writer)
            return nvrhi::FE_INVALID_ARGS;
        if (m_Desc.uploadMode != GpuSceneUploadMode::Scatter)
            return nvrhi::FE_UNSUPPORTED;   // Upload writes everything in CPU mode
        if (!m_Impl)
            return nvrhi::FS_OK;
        Impl& impl = *m_Impl;
        bool ok = true;
        if (impl.desc.geometryAndMaterialBuffers)
        {
            for (GpuSceneBuffer buffer : { GpuSceneBuffer::Geometries, GpuSceneBuffer::Materials })
            {
                const uint32_t b = uint32_t(buffer);
                const bool geometries = buffer == GpuSceneBuffer::Geometries;
                const uint32_t rowCount = geometries ? uint32_t(impl.geometries.size()) : uint32_t(impl.materials.size());
                const size_t rowSize = geometries ? sizeof(GeometryData) : sizeof(MaterialConstants);
                const uint8_t* bytes = geometries ? reinterpret_cast<const uint8_t*>(impl.geometries.data())
                                                  : reinterpret_cast<const uint8_t*>(impl.materials.data());
                if (rowCount == 0)
                    continue;
                if (impl.bufferRows[b] < rowCount)
                {
                    ok = false;   // CreateBuffers has not made it (yet): everything stays to be written
                    continue;
                }
                if (impl.assetsResized[b])
                {
                    if (!writer->ResizeBuffer(buffer, impl.bufferRows[b]))
                    {
                        ok = false;
                        continue;
                    }
                    impl.assetsResized[b] = false;
                }
                if (impl.rewrite[b] || impl.writeAll[b])
                {
                    // the buffer may be larger than the image (an image never shrinks, but another graph's was larger)
                    writer->WriteRows(buffer, 0, rowCount, bytes, rowSize);
                    ++impl.stats.uploadRuns;
                    impl.stats.uploadBytes += uint64_t(rowCount) * rowSize;
                    impl.rewrite[b] = false;
                    impl.writeAll[b] = false;
                    impl.pending[b].clear();
                }
                else
                    impl.WriteRuns(*writer, buffer, bytes, rowSize);
            }
        }
        impl.WriteConstants(*writer);
        return ok ? nvrhi::FS_OK : nvrhi::FE_OUT_OF_MEMORY;
    }

    nvrhi::FRESULT GpuScene::UploadAssets(nvrhi::ICommandList* commandList) noexcept
    {
        m_Recording.Check("GpuScene::UploadAssets");
        if (!commandList || !m_Device)
            return nvrhi::FE_INVALID_ARGS;
        if (!m_Impl)
            return nvrhi::FS_OK;
        // The buffers are not host-kept in scatter mode: the list tracks them from the state they rest in between
        // frames (the graph imports them in it) and returns them to it.
        Impl& impl = *m_Impl;
        nvrhi::IBuffer* tracked[2] = {};
        if (impl.desc.geometryAndMaterialBuffers && HasPendingAssets())
        {
            tracked[0] = impl.buffers[uint32_t(GpuSceneBuffer::Geometries)];
            tracked[1] = impl.buffers[uint32_t(GpuSceneBuffer::Materials)];
        }
        for (nvrhi::IBuffer* buffer : tracked)
            if (buffer)
                commandList->beginTrackingBufferState(buffer, c_GpuSceneRestingState);
        DeviceWriter writer(impl.device, commandList, impl.desc, impl.buffers, false);
        const nvrhi::FRESULT result = UploadAssetsTo(&writer);
        for (nvrhi::IBuffer* buffer : tracked)
            if (buffer)
                commandList->setBufferState(buffer, c_GpuSceneRestingState);
        commandList->commitBarriers();
        return result;
    }

    bool GpuScene::HasPendingAssets() const noexcept
    {
        m_Recording.Check("GpuScene::HasPendingAssets");
        if (!m_Impl)
            return false;
        const Impl& impl = *m_Impl;
        if (!impl.pendingConstants.empty())
            return true;
        if (!impl.desc.geometryAndMaterialBuffers)
            return false;
        for (GpuSceneBuffer buffer : { GpuSceneBuffer::Geometries, GpuSceneBuffer::Materials })
        {
            const uint32_t b = uint32_t(buffer);
            if (!impl.pending[b].empty() || impl.rewrite[b] || impl.writeAll[b] || impl.assetsResized[b])
                return true;
        }
        return false;
    }

    void GpuScene::GetScatterRecords(GpuSceneScatterRecords& out) const noexcept
    {
        m_Recording.Check("GpuScene::GetScatterRecords");
        out = GpuSceneScatterRecords();
        if (!m_Impl || m_Desc.uploadMode != GpuSceneUploadMode::Scatter)
            return;
        const Impl& impl = *m_Impl;
        out.settleSlots = impl.settleRecords.Data();
        out.settleCount = uint32_t(impl.settleRecords.Size());
        out.transforms = impl.transformRecords.Data();
        out.transformCount = uint32_t(impl.transformRecords.Size());
        out.rows = impl.rowRecords.Data();
        out.rowCount = uint32_t(impl.rowRecords.Size());
        out.serial = impl.scatterSerial;
        out.resend = impl.scatterResend;
    }

    void GpuScene::CommitScatter(uint64_t serial) noexcept
    {
        m_Recording.Check("GpuScene::CommitScatter");
        if (m_Impl && serial == m_Impl->scatterSerial)
            m_Impl->ackedSerial = serial;
    }

    bool GpuScene::IsScatterPending() const noexcept
    {
        m_Recording.Check("GpuScene::IsScatterPending");
        return m_Impl && m_Impl->scatterSerial != m_Impl->ackedSerial;
    }

    nvrhi::IBuffer* GpuScene::GetInstanceBuffer() const noexcept
    {
        return m_Impl ? m_Impl->buffers[uint32_t(GpuSceneBuffer::Instances)].Get() : nullptr;
    }

    nvrhi::IBuffer* GpuScene::GetGeometryBuffer() const noexcept
    {
        return m_Impl ? m_Impl->buffers[uint32_t(GpuSceneBuffer::Geometries)].Get() : nullptr;
    }

    nvrhi::IBuffer* GpuScene::GetMaterialBuffer() const noexcept
    {
        return m_Impl ? m_Impl->buffers[uint32_t(GpuSceneBuffer::Materials)].Get() : nullptr;
    }

    const InstanceData* GpuScene::GetInstanceRows() const noexcept { return m_Impl ? m_Impl->instances.data() : nullptr; }
    uint32_t GpuScene::GetInstanceRowCount() const noexcept { return m_Impl ? uint32_t(m_Impl->instances.size()) : 0; }
    const GeometryData* GpuScene::GetGeometryRows() const noexcept { return m_Impl ? m_Impl->geometries.data() : nullptr; }
    uint32_t GpuScene::GetGeometryRowCount() const noexcept { return m_Impl ? uint32_t(m_Impl->geometries.size()) : 0; }
    const MaterialConstants* GpuScene::GetMaterialRows() const noexcept { return m_Impl ? m_Impl->materials.data() : nullptr; }
    uint32_t GpuScene::GetMaterialRowCount() const noexcept { return m_Impl ? uint32_t(m_Impl->materials.size()) : 0; }

    GeometryData* GpuScene::GetGeometryRow(uint32_t slot) noexcept
    {
        return m_Impl && slot < m_Impl->geometries.size() ? &m_Impl->geometries[slot] : nullptr;
    }

    Entity GpuScene::GetInstanceSlotEntity(uint32_t slot) const noexcept
    {
        return m_Impl && slot < m_Impl->instanceEntities.size() ? m_Impl->instanceEntities[slot] : Entity::Null;
    }

    const GpuSceneStats& GpuScene::GetStats() const noexcept
    {
        static const GpuSceneStats s_None;
        return m_Impl ? m_Impl->stats : s_None;
    }
}
