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

// The GPU scene (aggregate docs/ecs-design.md 7.2; plan step E4): the persistent instance, geometry and material
// buffers of a scene graph, indexed by the stable slots of the graph's mesh instances and asset tables, and kept up to
// date by uploading only what changed. In donut_engine, beside Scene, which delegates RefreshBuffers to it (the plan's
// correction to design 3.2); the render-graph upload pass comes with step E5.
//
//     Update(graph)  device-free: walks the mesh-instance chunks whose WorldTransform, MeshRef, RenderFlags or
//                    GpuInstanceSlot changed (or that got rows) since the last update, builds their InstanceData rows
//                    into a CPU image indexed by instance slot, and keeps the rows that differ; does the same for the
//                    geometry rows of new or edited meshes and the rows of new or dirty materials.
//     Upload(...)    writes the kept rows, merged into runs, into the buffers (creating or growing them first), and
//                    the per-material constant buffers of dirty materials. The CPU scatter path.
//
// Scatter mode (GpuSceneDesc::uploadMode; plan step E5, sub-plan item E5.4, open points 3 and 9): the render-graph
// path. Update walks the changed chunks as in CPU mode, but a row whose only change is WorldTransform becomes a
// transform record (slot and float transform) without building or comparing its row; rows whose MeshRef, RenderFlags
// or GpuInstanceSlot changed, and rows new to their slot, become row records; the slots scattered last frame become
// settle records. The upload pass (donut/render/GpuSceneUploadPass.h) scatters them with a compute shader, which
// keeps prevTransform on the GPU (design 7.2): in this mode the CPU instance image is not kept up to date for
// transform-only rows. Buffer creation is separate from writing (CreateBuffers; the geometry and material rows with
// UploadAssets). A GpuScene keeps its mode for its lifetime (Scene chooses it before loading); graph-path scenes update
// once per frame.
//
// The records of an Update stay pending until the upload pass acknowledges them from its Execute (CommitScatter), that
// is, once they are recorded into the frame's command lists. An Update that finds the last records unacknowledged (the
// graph did not compile or execute, the pass was disabled, AddUploadPasses failed) sends every row again as a row
// record with prevTransform = transform, sends no settle record, and needs no growth copy (every row is written): a
// frame that was dropped costs one frame of motion vectors, never a stale row. The pass reads the records while it
// records (Execute); pipelined rendering (step E7) must keep them until then.
//
// Previous transforms (motion vectors) are kept per slot in the image: a row's prevTransform is the transform it was
// uploaded with by the last update of an earlier frame (a frame is a SceneGraph::Refresh call), so a second update in
// the same frame (VXGISample refreshes the buffers twice) keeps it; rows written with prevTransform != transform are
// visited once more in the next frame and settle to prevTransform == transform, as the old PrevTransform flags did. A
// slot that gets another entity starts with prevTransform == transform.
//
// A material shared with another graph has one constant buffer (Material::materialConstants) and one materialID: both
// hold what the GpuScene or graph that wrote last gave them (the slot of that graph). This GpuScene's material buffer
// and its geometry rows always use this graph's slots; a renderer that reads the per-material constant buffers of
// materials shared between scenes sees the last writer's materialID.
//
// Out of memory: an Update that fails leaves the images whole (the arrays of an image grow together or not at all) and
// the next Update builds every row again, after which the upload writes every buffer whole. An upload that cannot make
// a buffer leaves everything to write for the next one.
//
// Module-private C++: it takes the graph, an implementation class. The CPU images use the shader-side layouts of
// donut/shaders/bindless.h and material_cb.h, which this header only declares.

#pragma once

#include <donut/engine/EcsTypes.h>
#include <nvrhi/core/foundation.h>
#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <memory>

struct InstanceData;
struct GeometryData;
struct MaterialConstants;
struct GpuSceneTransformRecord;   // donut/shaders/gpu_scene_scatter_cb.h
struct GpuSceneRowRecord;

namespace donut::engine
{
    class SceneGraph;
    struct Material;

    enum class GpuSceneUploadMode : uint8_t
    {
        CpuRows,   // Update builds and compares rows; Upload writes the changed runs (Scene::RefreshBuffers; the reference)
        Scatter    // Update records transforms, rows and settles for the compute scatter (the render-graph path)
    };

    struct GpuSceneDesc
    {
        // The geometry and material buffers (bindless rendering; Scene: a descriptor table was given). Without them only
        // the instance buffer and the per-material constant buffers are kept.
        bool geometryAndMaterialBuffers = false;
        // Material::FillConstantBuffer's mode.
        bool resourceDescriptorHeapBindless = false;
        // D3D11 cannot make a buffer both structured and a vertex buffer: false there.
        bool structuredInstanceBuffer = true;
        // Buffers grow in steps of this many rows (Scene's allocation granularity).
        uint32_t rowGranularity = 1024;
        // Upload runs closer than this many rows are merged into one write (the rows between are rewritten unchanged).
        uint32_t mergeGap = 8;
        // For the lifetime of the GpuScene.
        GpuSceneUploadMode uploadMode = GpuSceneUploadMode::CpuRows;
    };

    // Scatter mode: the records of the last Update, in the order the kernel's constants describe. Valid until the next
    // Update. The records of one Update name every slot at most once.
    struct GpuSceneScatterRecords
    {
        const uint32_t* settleSlots = nullptr;                   // prevTransform = transform
        uint32_t settleCount = 0;
        const GpuSceneTransformRecord* transforms = nullptr;     // rows whose only change is WorldTransform
        uint32_t transformCount = 0;
        const GpuSceneRowRecord* rows = nullptr;                 // whole rows
        uint32_t rowCount = 0;
        // Increases with every scatter-mode Update; CommitScatter takes it back.
        uint64_t serial = 0;
        // Every live row is a row record (prevTransform = transform): the last records were not acknowledged, or this
        // is the first update. No settle record and no growth copy then.
        bool resend = false;
    };

    // What the last CreateBuffers did (GpuScene::GetBufferChanges). The replaced instance buffer is held until the
    // records of the Update it belongs to are acknowledged (CommitScatter), so that the upload pass can copy its rows
    // into the new one before the scatter (scatter mode keeps no full CPU image to rewrite); after a resend it is
    // dropped, as every row is written.
    struct GpuSceneBufferChanges
    {
        bool instancesResized = false;
        bool geometriesResized = false;
        bool materialsResized = false;
        nvrhi::IBuffer* previousInstanceBuffer = nullptr;   // null when the instance buffer was not replaced
        uint32_t previousInstanceRows = 0;                  // its size in rows
    };

    enum class GpuSceneBuffer : uint32_t
    {
        Instances,
        Geometries,
        Materials,
        Count
    };

    // Where Upload writes. A command list (UploadTo with the device's writer, inside Upload) or, for tests and dumps, a
    // CPU copy of the buffers.
    class IGpuSceneWriter
    {
    public:
        // The buffer is (re)created with `rows` rows: the rows written next fill it. False when it cannot be: the
        // upload writes nothing to it, reports FE_OUT_OF_MEMORY, and the next upload tries again.
        virtual bool ResizeBuffer(GpuSceneBuffer buffer, uint32_t rows) noexcept = 0;
        virtual void WriteRows(GpuSceneBuffer buffer, uint32_t firstRow, uint32_t rowCount,
                               const void* data, size_t rowSize) noexcept = 0;
        // A dirty or newly added material's own constant buffer (Material::materialConstants), with the value the last
        // update took for it (materialID: this scene's slot).
        virtual void WriteMaterialConstants(Material* material, const MaterialConstants& constants) noexcept = 0;
    protected:
        ~IGpuSceneWriter() = default;
    };

    struct GpuSceneStats
    {
        uint32_t chunksVisited = 0;        // mesh-instance chunks examined by the last Update
        uint32_t chunksChanged = 0;        // ... whose rows it rebuilt
        uint32_t rowsBuilt = 0;            // InstanceData rows rebuilt
        uint32_t rowsChanged = 0;          // ... that differed from the image (kept for the upload)
        uint32_t rowsSettled = 0;          // rows of the previous frame whose prevTransform settled
        uint32_t geometryRowsChanged = 0;
        uint32_t materialRowsChanged = 0;
        // The last Upload or UploadTo.
        uint32_t uploadRuns = 0;
        uint64_t uploadBytes = 0;
        uint32_t buffersResized = 0;
        // Scatter mode: the records of the last Update.
        uint32_t settleRecords = 0;
        uint32_t transformRecords = 0;
        uint32_t rowRecords = 0;
    };

    class GpuScene : public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(GpuScene)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        // `device` may be null: Update and UploadTo work without one (tests, benchmarks); Upload needs one. Construction
        // allocates nothing (the state is made by the first Update, which reports FE_OUT_OF_MEMORY if it cannot be).
        GpuScene(nvrhi::IDevice* device, const GpuSceneDesc& desc) noexcept;
        ~GpuScene();

        GpuScene(const GpuScene&) = delete;
        GpuScene& operator=(const GpuScene&) = delete;

        // Brings the CPU images up to date with `graph` (after its Refresh) and records what changed. A graph other
        // than the last one starts over: every row is built and the next upload writes every row again, into the same
        // buffers when they are large enough.
        nvrhi::FRESULT Update(SceneGraph* graph) noexcept;
        // The next Update fills every material and geometry row again: the mesh buffers or their descriptors changed
        // after the update that saw the meshes (Scene calls it after CreateMeshBuffers).
        void InvalidateAssets() noexcept;
        // Creates or grows the buffers and writes what Update recorded; creates missing per-material constant buffers.
        nvrhi::FRESULT Upload(nvrhi::ICommandList* commandList) noexcept;
        // The same writes, to `writer`; clears the materials' dirty flags as Upload does.
        nvrhi::FRESULT UploadTo(IGpuSceneWriter* writer) noexcept;

        // Scatter mode (the render-graph path). Upload and UploadTo return FE_UNSUPPORTED in it.
        [[nodiscard]] GpuSceneUploadMode GetUploadMode() const noexcept { return m_Desc.uploadMode; }
        // Creates or grows the buffers to the sizes the last Update needs, without writing any row (RenderService
        // calls it at extraction, after Update and before the buffers are imported). Needs a device.
        nvrhi::FRESULT CreateBuffers() noexcept;
        // What the last CreateBuffers did; valid until the next one (the upload pass's growth copy reads it).
        [[nodiscard]] const GpuSceneBufferChanges& GetBufferChanges() const noexcept;
        // Writes the kept geometry and material rows and the per-material constant buffers (the upload pass's Prepare,
        // on the graph's upload list); clears the materials' dirty flags. The mesh buffers and descriptors the rows name
        // were made at extraction, before Update (Scene::CreateMeshBufferObjects), so the rows are complete.
        nvrhi::FRESULT UploadAssets(nvrhi::ICommandList* commandList) noexcept;
        // The same writes, to `writer`.
        nvrhi::FRESULT UploadAssetsTo(IGpuSceneWriter* writer) noexcept;
        void GetScatterRecords(GpuSceneScatterRecords& out) const noexcept;
        // The upload pass's Execute: the records with this serial (GpuSceneScatterRecords::serial) are recorded. A
        // serial other than the last Update's is ignored. Unacknowledged records make the next Update a resend.
        void CommitScatter(uint64_t serial) noexcept;
        [[nodiscard]] bool IsScatterPending() const noexcept;

        [[nodiscard]] nvrhi::IBuffer* GetInstanceBuffer() const noexcept;
        [[nodiscard]] nvrhi::IBuffer* GetGeometryBuffer() const noexcept;
        [[nodiscard]] nvrhi::IBuffer* GetMaterialBuffer() const noexcept;

        // The CPU images, indexed by slot; rows of free slots hold what they held last. Valid until the next Update.
        [[nodiscard]] const InstanceData* GetInstanceRows() const noexcept;
        [[nodiscard]] uint32_t GetInstanceRowCount() const noexcept;   // the instance buffer's size in rows
        [[nodiscard]] const GeometryData* GetGeometryRows() const noexcept;
        [[nodiscard]] uint32_t GetGeometryRowCount() const noexcept;
        [[nodiscard]] const MaterialConstants* GetMaterialRows() const noexcept;
        [[nodiscard]] uint32_t GetMaterialRowCount() const noexcept;
        // The geometry row of a slot (MeshGeometry::globalGeometryIndex); null outside the image.
        [[nodiscard]] GeometryData* GetGeometryRow(uint32_t slot) noexcept;
        // The entity whose row an instance slot holds (Null when none was written there).
        [[nodiscard]] Entity GetInstanceSlotEntity(uint32_t slot) const noexcept;

        [[nodiscard]] const GpuSceneStats& GetStats() const noexcept;

    private:
        struct Impl;
        bool EnsureImpl() noexcept;
        nvrhi::IDevice* m_Device;   // not a reference: the scene that owns this holds the device
        GpuSceneDesc m_Desc;
        std::unique_ptr<Impl> m_Impl;
    };
}
