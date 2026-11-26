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
// transform record (slot and float transform) without building or comparing its row (only its float transform is
// compared with the one the slot holds, so that an unmoved row of a changed chunk costs no upload); rows whose MeshRef,
// RenderFlags or GpuInstanceSlot changed, and rows new to their slot, become row records; the slots whose
// prevTransform the last frame's records left behind their transform become settle records. The gather runs in
// parallel by chunk on the graph's job system, in chunk order at any worker count. The upload pass
// (donut/render/GpuSceneUploadPass.h) scatters them with a compute shader, which keeps prevTransform on the GPU
// (design 7.2): in this mode the CPU instance image holds every row's flags, geometry indices and transform, but its
// prevTransform only for rows new to their slot. Buffer creation is separate from writing (CreateBuffers; the
// geometry and material rows with UploadAssets). A GpuScene keeps its mode for its lifetime (Scene chooses it before
// loading). Graph-path scenes update once per frame (a SceneGraph::Refresh call): a second Update in one frame is
// treated as a later frame (its transform records move the frame's earlier transform into prevTransform; CPU mode
// keeps the earlier frame's), and its settle records wait for the next frame.
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

#include <atomic>
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

    // Scatter mode: the state of the instance, geometry and material buffers between frames. They are not host-kept
    // (keepInitialState false: the render graph's ImportBuffer rejects host-kept buffers), so the render graph imports
    // them with this as the entry and exit state, and UploadAssets returns them to it. CPU mode's buffers are host-kept
    // in it.
    constexpr nvrhi::ResourceStates c_GpuSceneRestingState = nvrhi::ResourceStates::ShaderResource;

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
        // Scatter mode: the records of the last Update. There rowsBuilt counts the candidate rows of the changed chunks,
        // rowsChanged the transform and row records, rowsSettled the settle records, and uploadBytes the records' bytes
        // (Update), then the asset rows' (UploadAssets); buffersResized counts CreateBuffers' new buffers.
        uint32_t settleRecords = 0;
        uint32_t transformRecords = 0;
        uint32_t rowRecords = 0;
    };

    // Pipelined rendering (design 7.5; plan step E7, item E7.8): while a pipelined frame records on another thread, the
    // objects its passes write on that thread (the GpuScene; the Scene's mesh-buffer data) belong to the recording, and
    // the main thread touches them only after the join (model 3 of the E7 sub-plan: GpuScene::Update runs in the next
    // Extract). Begin, called on the main thread when it hands the frame to the recording thread, names that thread;
    // End, after the join, ends the recording. Check, at the guarded calls: in Debug builds a call made on the named
    // thread in between, outside a PipelinedRecordingWork scope (below), logs one warning per recording (log::warning;
    // the call still runs); Release builds check
    // nothing. Never begun (serial frames), nothing is checked. Module-private.
    class PipelinedRecordingCheck
    {
    public:
        void Begin() noexcept;
        void End() noexcept;
        [[nodiscard]] bool IsRecording() const noexcept { return m_MainThread.load(std::memory_order_acquire) != 0; }
        void Check(const char* operation) const noexcept
        {
#ifndef NDEBUG
            if (m_MainThread.load(std::memory_order_relaxed) != 0)
                OnCheck(operation);
#else
            (void)operation;
#endif
        }

    private:
        void OnCheck(const char* operation) const noexcept;
        std::atomic<uint64_t> m_MainThread{ 0 };   // the thread that began the recording; 0 while none records
        mutable std::atomic<bool> m_Warned{ false };
    };

    // The recording's own work, on whatever thread runs it (part I7; backlog I6-R1). A thread that waits in a job
    // system runs other jobs, so the main thread, waiting in Simulate on the job system the recording shares, may run
    // a segment of the pipelined frame's recording: the calls made inside one of these scopes are the recording's and
    // PipelinedRecordingCheck does not flag them. The frame helper opens one around every recording task it hands to
    // the job system (and around the recording thread's own work). A thread_local depth: scopes nest, and a scope
    // must close on the thread that opened it. Release builds count too (two thread_local increments per task).
    class PipelinedRecordingWork
    {
    public:
        PipelinedRecordingWork() noexcept;
        ~PipelinedRecordingWork() noexcept;
        PipelinedRecordingWork(const PipelinedRecordingWork&) = delete;
        PipelinedRecordingWork& operator=(const PipelinedRecordingWork&) = delete;
        // The calling thread runs recording work now.
        [[nodiscard]] static bool IsActive() noexcept;
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
        // calls it at extraction, after Update and before the buffers are imported). Without a device it does the
        // bookkeeping only (sizes, GetBufferChanges without buffer objects; tests). A buffer is never shrunk. The
        // replaced instance buffer of the previous call is released here.
        nvrhi::FRESULT CreateBuffers() noexcept;
        // What the last CreateBuffers did; valid until the next one (the upload pass's growth copy reads it).
        [[nodiscard]] const GpuSceneBufferChanges& GetBufferChanges() const noexcept;
        // Writes the kept geometry and material rows and the per-material constant buffers (the upload pass's Prepare,
        // on the graph's upload list); clears the materials' dirty flags. The mesh buffers and descriptors the rows name
        // were made at extraction, before Update (Scene::CreateMeshBufferObjects), so the rows are complete.
        nvrhi::FRESULT UploadAssets(nvrhi::ICommandList* commandList) noexcept;
        // The same writes, to `writer`; ResizeBuffer is called for a geometry or material buffer that CreateBuffers
        // (re)created since the last asset upload, before its rows are written whole.
        nvrhi::FRESULT UploadAssetsTo(IGpuSceneWriter* writer) noexcept;
        // UploadAssets has something to write: geometry or material rows, or per-material constant buffers (the
        // upload pass adds no use when neither this, the records nor mesh data are due).
        [[nodiscard]] bool HasPendingAssets() const noexcept;
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

        // Pipelined rendering (E7.8; PipelinedRecordingCheck): while a pipelined frame records, its upload pass owns
        // this object (Prepare: UploadAssets; Execute: the records and CommitScatter). BeginPipelinedRecording on the
        // main thread when it hands the frame over, EndPipelinedRecording after the join. In between, Debug builds warn
        // once about a call of Update, InvalidateAssets, Upload, UploadTo, CreateBuffers, UploadAssets, UploadAssetsTo,
        // HasPendingAssets, GetScatterRecords, CommitScatter or IsScatterPending made on the main thread.
        void BeginPipelinedRecording() noexcept { m_Recording.Begin(); }
        void EndPipelinedRecording() noexcept { m_Recording.End(); }
        [[nodiscard]] bool IsPipelinedRecording() const noexcept { return m_Recording.IsRecording(); }

    private:
        struct Impl;
        bool EnsureImpl() noexcept;
        nvrhi::IDevice* m_Device;   // not a reference: the scene that owns this holds the device
        GpuSceneDesc m_Desc;
        std::unique_ptr<Impl> m_Impl;
        PipelinedRecordingCheck m_Recording;
    };
}
