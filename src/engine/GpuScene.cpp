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
#include <cstring>
#include <new>
#include <vector>

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>

namespace donut::engine
{
    static_assert(InstanceFlags_CurveDisjointOrthogonalTriangleStrips == c_InstanceFlagCurveDisjointOrthogonalTriangleStrips,
                  "RenderFlags::instanceFlags follows bindless.h");
    static_assert(InstanceFlags_CurveLinearSweptSpheres == c_InstanceFlagCurveLinearSweptSpheres,
                  "RenderFlags::instanceFlags follows bindless.h");
    static_assert(sizeof(InstanceData) == 112 && sizeof(GeometryData) == 64, "layouts of bindless.h");

    namespace
    {
        using namespace ecs;

        constexpr uint32_t c_BufferCount = uint32_t(GpuSceneBuffer::Count);

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

        void Reset() noexcept;
        void Ensure(GpuSceneBuffer buffer, uint32_t end);
        void Mark(GpuSceneBuffer buffer, uint32_t row);
        void UpdateMaterials(SceneGraph& graph, bool structureChanged);
        void FillGeometries(const MeshTable& meshes, uint32_t slot);
        void UpdateGeometries(SceneGraph& graph, bool structureChanged);
        void UpdateInstances(SceneGraphStore& store, SceneGraph& graph, bool all);
        bool WriteBuffer(IGpuSceneWriter& writer, GpuSceneBuffer buffer, const void* rows, size_t rowSize);
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
    void GpuScene::Impl::UpdateMaterials(SceneGraph& graph, bool structureChanged)
    {
        const MaterialTable& table = graph.GetMaterials();
        const bool all = structureChanged;
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
            const bool constants = material->dirty || added || (device && !material->materialConstants);
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
                if (!all && versions[slotColumn] <= since && versions[refColumn] <= since &&
                    versions[flagsColumn] <= since && versions[worldColumn] <= since && versions[added] <= since)
                    continue;
                ++stats.chunksChanged;

                uint8_t* memory = archetype.chunks[c].memory;
                const Entity* entities = archetype.Entities(c);
                const auto* slots = reinterpret_cast<const GpuInstanceSlot*>(memory + archetype.columns[slotColumn].offset);
                const auto* refs = reinterpret_cast<const MeshRef*>(memory + archetype.columns[refColumn].offset);
                const auto* flags = reinterpret_cast<const RenderFlags*>(memory + archetype.columns[flagsColumn].offset);
                const auto* worlds = reinterpret_cast<const WorldTransform*>(memory + archetype.columns[worldColumn].offset);
                const uint32_t count = archetype.chunks[c].count;
                for (uint32_t r = 0; r < count; ++r)
                {
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

        std::vector<Run>& runs = pending[b];
        if (runs.empty())
            return true;
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
        return true;
    }

    // ------------------------------------------------------------------------------------- the device

    namespace
    {
        class DeviceWriter final : public IGpuSceneWriter
        {
        public:
            DeviceWriter(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, const GpuSceneDesc& desc,
                         nvrhi::BufferHandle* buffers) noexcept
                : m_Device(device), m_CommandList(commandList), m_Desc(desc), m_Buffers(buffers) { }

            // Scene::CreateInstanceBuffer, CreateGeometryBuffer and CreateMaterialBuffer.
            bool ResizeBuffer(GpuSceneBuffer buffer, uint32_t rows) noexcept override
            {
                nvrhi::BufferDesc bufferDesc;
                bufferDesc.canHaveRawViews = true;
                bufferDesc.canHaveUAVs = true;
                bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
                bufferDesc.keepInitialState = true;
                switch (buffer)
                {
                case GpuSceneBuffer::Instances:
                    bufferDesc.byteSize = sizeof(InstanceData) * uint64_t(rows);
                    bufferDesc.debugName = "Instances";
                    bufferDesc.structStride = m_Desc.structuredInstanceBuffer ? sizeof(InstanceData) : 0;
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
                nvrhi::BufferHandle& handle = m_Buffers[uint32_t(buffer)];
                handle.Reset();
                if (NVRHI_FAILED(m_Device->createBuffer(bufferDesc, &handle)))
                {
                    log::warning("GpuScene: cannot create the %s buffer (%u rows)", bufferDesc.debugName.c_str(), rows);
                    return false;
                }
                return true;
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
        if (m_Impl)
            m_Impl->assetsInvalid = true;
    }

    nvrhi::FRESULT GpuScene::Update(SceneGraph* graph) noexcept
    {
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

            impl.UpdateMaterials(*graph, structureChanged);
            if (impl.desc.geometryAndMaterialBuffers)
                impl.UpdateGeometries(*graph, structureChanged);
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
            log::warning("GpuScene::Update: out of memory");
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT GpuScene::UploadTo(IGpuSceneWriter* writer) noexcept
    {
        if (!writer)
            return nvrhi::FE_INVALID_ARGS;
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
        for (const Impl::PendingConstants& pending : impl.pendingConstants)
        {
            writer->WriteMaterialConstants(pending.material, pending.value);
            pending.material->dirty = false;
        }
        impl.pendingConstants.clear();
        std::fill(impl.pendingBySlot.begin(), impl.pendingBySlot.end(), ~0u);
        return ok ? nvrhi::FS_OK : nvrhi::FE_OUT_OF_MEMORY;
    }

    nvrhi::FRESULT GpuScene::Upload(nvrhi::ICommandList* commandList) noexcept
    {
        if (!commandList || !m_Device)
            return nvrhi::FE_INVALID_ARGS;
        if (!m_Impl)
            return nvrhi::FS_OK;
        DeviceWriter writer(m_Impl->device, commandList, m_Impl->desc, m_Impl->buffers);
        return UploadTo(&writer);
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
