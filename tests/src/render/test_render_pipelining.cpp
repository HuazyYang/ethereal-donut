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

// Pipelined recording against serial recording, device-free (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md,
// stage T6; revision 1's E7.8 tests; design 7.5, D8). A pipelined render service (RenderServiceDesc::pipelined: a
// Defer render-read scope from Extract to EndFrame, BeginPipelinedRecording at the hand-off) extracts frame k; a second
// thread records it while the main thread runs frame k+1's world work on the same job system: Simulate and Refresh with
// structural churn (moves, Detach and re-Attach, subgraphs dropped, SetLeaf, runtime-only rows spawned and despawned,
// a mesh and a material whose last references go, a runtime mesh released, light parameters, an in-place edit of a
// mesh the open frame does not name); then the join, EndFrame, and the edits that must wait for the join (an in-place
// edit of a drawn mesh's geometries: backlog I6-N1, part I6's item-4 choice). The recording, in tasks of the job
// system each inside a PipelinedRecordingWork scope (so the main thread may run some while it waits in Simulate):
// the draw items of every view and pass through PacketDrawStrategy, and the GpuScene work of the upload pass
// (Prepare: UploadAssetsTo, a CPU writer for the device's lists; Execute: the scatter records and CommitScatter) on a
// device-free scatter GpuScene whose owner-thread check is begun at the hand-off. Per frame, the snapshot, the draw
// items and the upload hash (no pointers: names, slots, ids, values) equal those of a serial run of the same script
// (record, EndFrame, then the world work); the snapshot is unchanged by the overlap; what the overlap drops lives until
// EndFrame and goes then; legitimate recording work logs no owner-check warning (backlog I6-R1, fixed in I7); a walking
// strategy on the recording thread is refused with its one warning. 200 iterations; Debug builds fill the frame arena
// with a pattern at Reset (RenderGraph.cpp), so a recording that read a reset arena would hash differently.
//
//     test_render_pipelining [case | --backlog]

#include <donut/render/DrawStrategy.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include <donut/tests/utils.h>

using namespace donut::math;
#include <donut/shaders/gpu_scene_scatter_cb.h>
#include <donut/shaders/light_cb.h>
#include <donut/shaders/material_cb.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <stdlib.h>
#endif

using namespace donut;
using namespace donut::engine;
using namespace donut::render;

namespace
{
    std::atomic<uint32_t> g_Warnings{ 0 };

    // xorshift64*: the same sequence everywhere.
    class Rng
    {
    public:
        explicit Rng(uint64_t seed) : m_State(seed ^ 0x9E3779B97F4A7C15ull) { if (m_State == 0) m_State = 1; }
        uint64_t Next()
        {
            m_State ^= m_State >> 12;
            m_State ^= m_State << 25;
            m_State ^= m_State >> 27;
            return m_State * 0x2545F4914F6CDD1Dull;
        }
        uint32_t Below(size_t n) { return uint32_t((Next() >> 32) % uint64_t(n)); }
        bool Percent(uint32_t percent) { return Below(100) < percent; }
        double Range(double lo, double hi) { return lo + (hi - lo) * double(Next() >> 11) * (1.0 / 9007199254740992.0); }

    private:
        uint64_t m_State;
    };

    struct Hasher
    {
        uint64_t value = 0xcbf29ce484222325ull;

        void Bytes(const void* data, size_t size)
        {
            const auto* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; ++i)
                value = (value ^ bytes[i]) * 0x100000001b3ull;
        }
        template <typename T> void Value(const T& v) { Bytes(&v, sizeof(v)); }
        template <typename T> void Array(const T* data, uint32_t count)
        {
            Value(count);
            if (count)
                Bytes(data, sizeof(T) * count);
        }
        void String(const std::string& s) { Array(s.data(), uint32_t(s.size())); }
    };

    // Meshes and materials whose destruction is counted (MeshInfo has no weak references).
    std::atomic<uint32_t> g_MeshesDestroyed{ 0 };
    struct CountedMesh : MeshInfo
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
        ~CountedMesh() override { g_MeshesDestroyed.fetch_add(1); }
    };

    nvrhi::AutoPtr<nvrhi::IWeakReference> WeakOf(nvrhi::IWeakReferenceSource* object)
    {
        nvrhi::AutoPtr<nvrhi::IWeakReference> weak;
        object->GetWeakReference(weak.GetAddressOf());
        CHECK(weak);
        return weak;
    }

    // A perspective view and an orthographic one (as in test_render_extract).
    struct TestViews
    {
        nvrhi::AutoPtr<PlanarView> perspective = MAKE_RC_OBJ_PTR(PlanarView);
        nvrhi::AutoPtr<PlanarView> orthographic = MAKE_RC_OBJ_PTR(PlanarView);
        const IView* list[2] = {};

        TestViews()
        {
            perspective->SetViewport(nvrhi::Viewport(1280.f, 720.f));
            const dm::affine3 eye = dm::translation(dm::float3(0.f, 2.f, -20.f));
            perspective->SetMatrices(dm::inverse(eye), dm::perspProjD3DStyleReverse(dm::radians(60.f), 16.f / 9.f, 0.1f));
            perspective->UpdateCache();
            orthographic->SetViewport(nvrhi::Viewport(1024.f, 1024.f));
            const dm::affine3 sun = dm::lookatZ(dm::normalize(dm::float3(0.3f, -1.f, 0.2f))) *
                                    dm::translation(dm::float3(0.f, 50.f, 0.f));
            orthographic->SetMatrices(dm::inverse(sun), dm::orthoProjD3DStyle(-40.f, 40.f, -40.f, 40.f, 0.f, 100.f));
            orthographic->UpdateCache();
            list[0] = perspective.Get();
            list[1] = orthographic.Get();
        }

        ExtractDesc Desc(uint32_t frameIndex) const
        {
            ExtractDesc desc;
            desc.views = list;
            desc.viewCount = 2;
            desc.frameIndex = frameIndex;
            return desc;
        }
    };

    // A device-free render graph and its blackboard; two of them alternate, one per frame in flight.
    struct RenderFrame
    {
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;
        FrameData data;

        RenderFrame()
        {
            CHECK(CreateResourceRegistry(nullptr, &registry) == nvrhi::FS_OK);
            CHECK(CreateRenderGraph(registry, &graph) == nvrhi::FS_OK);
        }
    };

    // ------------------------------------------------------------------------------------------------- the scene

    dm::double3 RandomVector(Rng& rng, double extent)
    {
        const double x = rng.Range(-extent, extent);
        const double y = rng.Range(-extent, extent);
        const double z = rng.Range(-extent, extent);
        return dm::double3(x, y, z);
    }

    // One run's world and its script. The same seed gives the same world and the same calls in both modes.
    struct Scene
    {
        Rng rng;
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        std::vector<nvrhi::AutoPtr<Material>> materials;
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;            // shared; meshes[0] is edited in place after joins
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> detached;    // detached subgraphs, attached again later
        std::vector<ComponentId> runtimeComponents;
        std::deque<std::pair<uint32_t, std::vector<Entity>>> runtimeRows;   // spawn frame, rows
        MeshId runtimeMesh = MeshId::Invalid;                    // the rows' current mesh (the caller's reference)
        nvrhi::AutoPtr<MeshInfo> hidden;                         // a runtime row's mesh no view sees
        nvrhi::AutoPtr<SceneGraphNode> ephemeral;                // a node whose mesh and material only it uses
        nvrhi::AutoPtr<nvrhi::IWeakReference> ephemeralMaterial;
        nvrhi::AutoPtr<nvrhi::IWeakReference> droppedMaterial;   // the dropped one, until its release is checked
        uint32_t meshesDestroyed = 0;                            // g_MeshesDestroyed after the last check
        uint32_t nextName = 0;
        double time = 0.0;

        Scene(uint64_t seed, uint32_t workers)
            : rng(seed)
        {
            JobSystemDesc desc;
            desc.debugName = "test_render_pipelining";
            desc.workerThreadCount = workers;
            desc.maxExternalThreads = 4;   // the main thread, the recording thread
            CHECK(CreateJobSystem(desc, &jobs) == nvrhi::FS_OK);
            graph->SetJobSystem(jobs);
        }

        nvrhi::AutoPtr<Material> NewMaterial(const std::string& name, MaterialDomain domain)
        {
            auto material = factory->CreateMaterial();
            material->name = name;
            material->domain = domain;
            return material;
        }

        void AddGeometry(MeshInfo* mesh, Material* material)
        {
            auto geometry = factory->CreateMeshGeometry();
            geometry->material = material;
            geometry->numIndices = 3 * uint32_t(mesh->geometries.size() + 1);
            geometry->numVertices = 3;
            geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f, 2.f, 1.f));
            mesh->geometries.push_back(geometry);
        }

        nvrhi::AutoPtr<MeshInfo> NewMesh(const std::string& name, std::initializer_list<Material*> list, bool counted)
        {
            nvrhi::AutoPtr<MeshInfo> mesh = counted ? nvrhi::AutoPtr<MeshInfo>(MAKE_RC_OBJ_PTR(CountedMesh))
                                                    : factory->CreateMesh();
            mesh->name = name;
            for (Material* material : list)
                AddGeometry(mesh, material);
            mesh->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f, 2.f, 1.f));
            return mesh;
        }

        nvrhi::AutoPtr<SceneGraphNode> NewNode()
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetName("n" + std::to_string(nextName++));
            node->SetTranslation(RandomVector(rng, 8.0));
            return node;
        }

        nvrhi::AutoPtr<SceneGraphLeaf> NewLeaf()
        {
            const uint32_t kind = rng.Below(12);
            if (kind == 10)
                return MAKE_RC_OBJ_PTR(PointLight);
            if (kind == 11)
                return MAKE_RC_OBJ_PTR(SpotLight);
            return factory->CreateMeshInstance(meshes[kind % meshes.size()]);
        }

        std::vector<SceneGraphNode*> Nodes() const
        {
            std::vector<SceneGraphNode*> nodes;
            for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
                nodes.push_back(walker.Get());
            return nodes;
        }

        // The nodes random edits may take: not the root, not the ephemeral node or below it.
        std::vector<SceneGraphNode*> Editable() const
        {
            std::vector<SceneGraphNode*> nodes;
            for (SceneGraphNode* node : Nodes())
            {
                bool excluded = node == graph->GetRootNode();
                for (SceneGraphNode* up = node; up && !excluded; up = up->GetParent())
                    excluded = up == ephemeral.Get();
                if (!excluded)
                    nodes.push_back(node);
            }
            return nodes;
        }

        void Build(uint32_t size)
        {
            materials.push_back(NewMaterial("opaque", MaterialDomain::Opaque));
            materials.push_back(NewMaterial("alpha_tested", MaterialDomain::AlphaTested));
            materials.push_back(NewMaterial("blended", MaterialDomain::AlphaBlended));
            meshes.push_back(NewMesh("box", { materials[0] }, false));
            meshes.push_back(NewMesh("two_part", { materials[0], materials[1] }, false));
            meshes.push_back(NewMesh("glass", { materials[2] }, false));
            meshes.push_back(NewMesh("mixed", { materials[1], materials[2] }, false));
            graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            std::vector<SceneGraphNode*> built = { graph->GetRootNode() };
            for (uint32_t i = 0; i < size; ++i)
            {
                auto node = NewNode();
                if (rng.Percent(80))
                    node->SetLeaf(NewLeaf());
                nvrhi::AutoPtr<SceneGraphNode> attached = graph->Attach(built[rng.Below(built.size())], node);
                built.push_back(attached);
            }
            auto sun = MAKE_RC_OBJ_PTR(DirectionalLight);
            graph->AttachLeafNode(graph->GetRootNode(), sun);
            graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(PerspectiveCamera));

            runtimeComponents.resize(graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
            graph->GetRuntimeMeshInstanceComponents(runtimeComponents.data(), uint32_t(runtimeComponents.size()));
            hidden = NewMesh("hidden", { materials[0] }, false);
            const MeshId hiddenId = graph->AcquireRuntimeMesh(hidden);
            CHECK(hiddenId != MeshId::Invalid);
            Spawn(hiddenId, { dm::double3(1.0e5, 1.0e5, 1.0e5) });   // never visible
            graph->ReleaseRuntimeMesh(hiddenId);
            graph->Refresh(0);
        }

        std::vector<Entity> Spawn(MeshId mesh, const std::vector<dm::double3>& positions)
        {
            IWorld* world = graph->GetWorld();
            std::vector<LocalTransform> locals;
            for (const dm::double3& p : positions)
                locals.push_back(LocalTransform{ p, dm::dquat::identity(), dm::double3(1.0) });
            const std::vector<MeshRef> refs(positions.size(), MeshRef{ mesh });
            std::vector<const void*> values(runtimeComponents.size(), nullptr);
            for (size_t i = 0; i < runtimeComponents.size(); ++i)
            {
                if (runtimeComponents[i] == FindComponent<LocalTransform>(world))
                    values[i] = locals.data();
                else if (runtimeComponents[i] == FindComponent<MeshRef>(world))
                    values[i] = refs.data();
            }
            EntityCreateDesc desc{};
            desc.components = runtimeComponents.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(runtimeComponents.size());
            desc.entityCount = uint32_t(positions.size());
            std::vector<Entity> rows(positions.size(), Entity::Null);
            CHECK(world->CreateEntities(desc, rows.data()) == nvrhi::FS_OK);
            return rows;
        }

        // Frame `frame`'s world work, which a pipelined run does while frame - 1 records. `pipelined`: the ephemeral
        // node dropped here was named by the recording frame, so its material must still be alive.
        void WorldWork(uint32_t frame, bool pipelined)
        {
            std::vector<SceneGraphNode*> nodes = Editable();
            // moves
            for (SceneGraphNode* node : nodes)
                if (rng.Percent(20))
                    node->SetTranslation(RandomVector(rng, 8.0));
            // Detach: kept for a later Attach, or dropped (nobody keeps it)
            if (nodes.size() > 4 && rng.Percent(60))
            {
                nvrhi::AutoPtr<SceneGraphNode> node = graph->Detach(nodes[rng.Below(nodes.size())]);
                if (rng.Percent(60))
                    detached.push_back(node);
            }
            nodes = Editable();
            if (!detached.empty() && rng.Percent(50))
            {
                graph->Attach(nodes.empty() ? graph->GetRootNode() : nodes[rng.Below(nodes.size())], detached.front());
                detached.erase(detached.begin());
            }
            nodes = Editable();
            // SetLeaf over a mesh instance or a light; a new node
            if (!nodes.empty() && rng.Percent(70))
                nodes[rng.Below(nodes.size())]->SetLeaf(NewLeaf());
            if (rng.Percent(50))
            {
                auto node = NewNode();
                node->SetLeaf(NewLeaf());
                graph->Attach(nodes.empty() ? graph->GetRootNode() : nodes[rng.Below(nodes.size())], node);
            }

            // The ephemeral mesh and material go with their node; a new one comes.
            if (ephemeral)
            {
                // Pipelined: the recording frame names them, so the retire list keeps them until EndFrame; serial:
                // they go at once.
                (void)graph->Detach(ephemeral);
                ephemeral = nullptr;
                CHECK(pipelined ? ephemeralMaterial->IsExpired() == 0 : ephemeralMaterial->IsExpired() != 0);
                CHECK(g_MeshesDestroyed.load() == meshesDestroyed + (pipelined ? 0u : 1u));
                if (pipelined)
                    droppedMaterial = ephemeralMaterial;
            }
            {
                auto material = NewMaterial("ephemeral_material_" + std::to_string(frame), MaterialDomain::Opaque);
                ephemeralMaterial = WeakOf(material);
                auto mesh = NewMesh("ephemeral_" + std::to_string(frame), { material }, true);
                auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                node->SetName("ephemeral");
                node->SetTranslation(dm::double3(0.0, 0.0, 2.0));
                node->SetLeaf(factory->CreateMeshInstance(mesh));
                ephemeral = graph->Attach(graph->GetRootNode(), node);
            }

            // Runtime-only rows: a new mesh every 10 frames (released by the caller at once, then by its last row),
            // two rows a frame, each kept for three frames.
            if (frame % 10 == 1)
            {
                if (runtimeMesh != MeshId::Invalid)
                    graph->ReleaseRuntimeMesh(runtimeMesh);
                const std::string name = "runtime_" + std::to_string(frame);
                auto material = NewMaterial(name + "_material", MaterialDomain::AlphaTested);
                auto mesh = NewMesh(name, { material }, true);
                runtimeMesh = graph->AcquireRuntimeMesh(mesh);
                CHECK(runtimeMesh != MeshId::Invalid);
            }
            runtimeRows.emplace_back(frame, Spawn(runtimeMesh, { RandomVector(rng, 6.0), RandomVector(rng, 6.0) }));
            while (runtimeRows.front().first + 3 <= frame)
            {
                const std::vector<Entity>& rows = runtimeRows.front().second;
                CHECK(graph->GetWorld()->DestroyEntities(rows.data(), uint32_t(rows.size())) == nvrhi::FS_OK);
                runtimeRows.pop_front();
            }

            // Light parameters (the recording reads the extracted constants, not the leaf).
            for (const auto& light : graph->GetLights())
            {
                const float red = float(rng.Range(0.0, 1.0));
                const float green = float(rng.Range(0.0, 1.0));
                light->color = dm::float3(red, green, 1.f);
            }

            // An in-place edit of a mesh no view of the open frame sees (the reconciliation waits for EndFrame).
            if (frame % 7 == 3)
            {
                if (hidden->geometries.size() < 4)
                    AddGeometry(hidden, materials[0]);
                else
                    hidden->geometries.pop_back();
                graph->AttachLeafNode(graph->GetRootNode(), MAKE_RC_OBJ_PTR(PointLight));   // a structural change
            }

            time += 1.0 / 60.0;
            CHECK(graph->Simulate(FrameTime{ time, 1.0 / 60.0, frame }) == nvrhi::FS_OK);
            graph->Refresh(frame);
            if (pipelined)   // the overlap released no counted mesh: they are all retired
                CHECK(g_MeshesDestroyed.load() == meshesDestroyed);
            meshesDestroyed = g_MeshesDestroyed.load();
        }

        // After the join (and EndFrame): what the application may change only then, the geometries of a drawn mesh
        // (backlog I6-N1), with a structural change so that Refresh reconciles the mesh table.
        void AfterJoin(uint32_t frame)
        {
            if (frame % 9 != 4)
                return;
            MeshInfo* mesh = meshes[0];
            if (mesh->geometries.size() < 4)
                AddGeometry(mesh, materials[0]);   // the same domain: a domain change needs InvalidateContent
            else
                mesh->geometries.resize(1);
            graph->Attach(graph->GetRootNode(), MAKE_RC_OBJ_PTR(SceneGraphNode));
            graph->Refresh(frame);
        }
    };

    // ------------------------------------------------------------------------------------------------- recording

    // The snapshot's values, no pointers (the runs' objects differ): ids, slots, packets, bounds, lights, cameras.
    uint64_t HashSnapshot(const RenderSnapshot& s)
    {
        Hasher h;
        h.Value(s.frameIndex);
        h.Value(s.viewCount);
        for (uint32_t v = 0; v < s.viewCount; ++v)
        {
            h.Value(s.views[v].instances);
            for (const SnapshotRange& range : s.views[v].packets)
                h.Value(range);
        }
        const VisibleInstances& in = s.instances;
        h.Array(in.entities, in.count);
        h.Array(in.instanceSlots, in.count);
        h.Array(in.meshes, in.count);
        h.Array(in.firstGeometryInstanceSlots, in.count);
        h.Array(in.geometryCounts, in.count);
        h.Array(in.worldBounds, in.count);
        for (uint32_t i = 0; i < in.count; ++i)
        {
            h.Value(uint8_t(in.leaves[i] != nullptr));
            h.String(in.meshInfos[i]->name);
        }
        const DrawPackets& p = s.packets;
        h.Array(p.keys, p.count);
        h.Array(p.visibleInstances, p.count);
        h.Array(p.geometryIndices, p.count);
        h.Array(p.geometrySlots, p.count);
        h.Array(p.materialSlots, p.count);
        h.Array(p.cullModes, p.count);
        h.Array(p.flags, p.count);
        h.Value(s.lightCount);
        for (uint32_t i = 0; i < s.lightCount; ++i)
        {
            h.Bytes(s.lights[i].constants, sizeof(LightConstants));
            h.Value(s.lights[i].entity);
        }
        h.Value(s.cameraCount);
        for (uint32_t i = 0; i < s.cameraCount; ++i)
        {
            h.Value(s.cameras[i].entity);
            h.Value(s.cameras[i].viewToWorld);
        }
        h.Value(s.skinnedInstanceCount);
        return h.value;
    }

    // The upload pass's writes, hashed: Prepare's asset rows and material constants.
    class HashWriter final : public IGpuSceneWriter
    {
    public:
        Hasher hash;
        bool ResizeBuffer(GpuSceneBuffer buffer, uint32_t rows) noexcept override
        {
            hash.Value(buffer);
            hash.Value(rows);
            return true;
        }
        void WriteRows(GpuSceneBuffer buffer, uint32_t firstRow, uint32_t rowCount, const void* data,
                       size_t rowSize) noexcept override
        {
            hash.Value(buffer);
            hash.Value(firstRow);
            hash.Value(rowCount);
            hash.Bytes(data, rowSize * rowCount);
        }
        void WriteMaterialConstants(Material* material, const MaterialConstants& constants) noexcept override
        {
            hash.String(material->name);
            hash.Bytes(&constants, sizeof(constants));
        }
    };

    void Spin(uint32_t microseconds)
    {
        const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(microseconds);
        while (microseconds && std::chrono::steady_clock::now() < end)
            std::this_thread::yield();
    }

    // A job that keeps the job system's one worker busy until `done` is set (or 200 ms pass): the main thread that
    // waits for it runs whatever is queued meanwhile, the recording's tasks included (backlog I6-R1's pattern).
    struct BusyWorker
    {
        const std::atomic<bool>* done = nullptr;

        static void Run(void* context, uint32_t, uint32_t, uint32_t) noexcept
        {
            const BusyWorker& self = *static_cast<const BusyWorker*>(context);
            const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
            while (!self.done->load() && std::chrono::steady_clock::now() < end)
                std::this_thread::yield();
        }
    };

    JobHandle ScheduleBusyWorker(IJobSystem* jobs, BusyWorker& busy)
    {
        JobDesc job{ &BusyWorker::Run, &busy, 1, 1, "busy worker" };
        JobHandle handle = JobHandle::Null;
        CHECK(jobs->Schedule(job, nullptr, 0, handle) == nvrhi::FS_OK);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));   // the worker takes it, not the main thread
        return handle;
    }

    constexpr uint32_t c_MaxViews = 4;
    constexpr uint32_t c_ItemSlots = c_MaxViews * c_DrawPacketPassCount;

    // One frame's recording: a task per view and pass (its draw items) and one for the GpuScene's upload work, run
    // through the job system; each task inside a PipelinedRecordingWork scope.
    struct Recorder
    {
        nvrhi::AutoPtr<PacketDrawStrategy> strategies[c_ItemSlots];
        nvrhi::AutoPtr<InstancedOpaqueDrawStrategy> walking = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        const RenderSnapshot* snapshot = nullptr;
        GpuScene* gpuScene = nullptr;
        IJobSystem* jobs = nullptr;
        SceneGraphNode* root = nullptr;   // read on the main thread before the hand-off; the walk must not use it
        bool walk = false;                // ask the walking strategy too (pipelined runs: refused)
        std::thread::id mainThread;

        // Outputs.
        uint64_t items[c_ItemSlots] = {};
        uint32_t itemCounts[c_ItemSlots] = {};
        uint64_t upload = 0;
        uint64_t snapshotAfter = 0;
        uint32_t walked = 0;
        std::atomic<uint32_t> tasksOnMainThread{ 0 };
        std::atomic<uint32_t> itemCountMismatches{ 0 };
        uint32_t spinMicroseconds = 0;    // each task takes at least this long (the stealing case)
        std::atomic<bool> finished{ false };

        Recorder()
        {
            for (auto& strategy : strategies)
                strategy = MAKE_RC_OBJ_PTR(PacketDrawStrategy);
        }

        void DrawItems(uint32_t slot)
        {
            const uint32_t view = slot / c_DrawPacketPassCount;
            const DrawPacketPass pass = DrawPacketPass(slot % c_DrawPacketPassCount);
            Hasher h;
            uint32_t count = 0;
            if (view < snapshot->viewCount)
            {
                PacketDrawStrategy& strategy = *strategies[slot];
                strategy.SetPackets(snapshot, pass);
                strategy.PrepareForView(nullptr, *snapshot->views[view].view);
                while (const DrawItem* item = strategy.GetNextItem())
                {
                    ++count;
                    h.Value(item->instanceSlot);
                    h.Value(item->distanceToCamera);
                    h.Value(item->cullMode);
                    h.Value(uint8_t(item->instance != nullptr));
                    h.String(item->mesh->name);
                    const auto& geometries = item->mesh->geometries;
                    const auto found = std::find_if(geometries.begin(), geometries.end(),
                                                    [item](const auto& g) { return g.Get() == item->geometry; });
                    h.Value(uint32_t(found - geometries.begin()));
                    h.Value(item->geometry->numIndices);
                    h.String(item->material ? item->material->name : std::string("(none)"));
                    h.Value(uint8_t(item->buffers == item->mesh->buffers.Get()));
                }
                strategy.SetPackets(nullptr, pass);
                if (count != snapshot->views[view].packets[uint32_t(pass)].count)
                    itemCountMismatches.fetch_add(1);
            }
            items[slot] = h.value;
            itemCounts[slot] = count;
        }

        // The upload pass's GpuScene work: Prepare (UploadAssets, here to a CPU writer), Execute (the records,
        // CommitScatter).
        void Upload()
        {
            HashWriter writer;
            writer.hash.Value(uint8_t(gpuScene->HasPendingAssets()));
            CHECK(gpuScene->UploadAssetsTo(&writer) == nvrhi::FS_OK);
            GpuSceneScatterRecords records;
            gpuScene->GetScatterRecords(records);
            Hasher& h = writer.hash;
            h.Value(uint8_t(records.resend));
            h.Array(records.settleSlots, records.settleCount);
            h.Value(records.transformCount);
            for (uint32_t i = 0; i < records.transformCount; ++i)
            {
                h.Value(records.transforms[i].slot);
                h.Value(records.transforms[i].transform);
            }
            h.Value(records.rowCount);
            for (uint32_t i = 0; i < records.rowCount; ++i)
            {
                h.Value(records.rows[i].slot);
                h.Value(records.rows[i].flags);
                h.Value(records.rows[i].row);
            }
            gpuScene->CommitScatter(records.serial);
            (void)gpuScene->IsScatterPending();
            upload = h.value;
        }

        static void Task(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
        {
            Recorder& self = *static_cast<Recorder*>(context);
            for (uint32_t i = begin; i < end; ++i)
            {
                PipelinedRecordingWork work;   // the recording's own work, on whatever thread runs it
                Spin(self.spinMicroseconds);
                if (std::this_thread::get_id() == self.mainThread)
                    self.tasksOnMainThread.fetch_add(1);
                try
                {
                    if (i < c_ItemSlots)
                        self.DrawItems(i);
                    else
                        self.Upload();
                }
                catch (const std::runtime_error& error)
                {
                    std::fprintf(stderr, "%s\n", error.what());
                    self.itemCountMismatches.fetch_add(1000);
                }
            }
        }

        void Record()
        {
            PipelinedRecordingWork work;
            jobs->ParallelFor(c_ItemSlots + 1, 1, &Task, this);
            if (walk)
            {
                walking->PrepareForView(root, *snapshot->views[0].view);
                while (walking->GetNextItem())
                    ++walked;
            }
            snapshotAfter = HashSnapshot(*snapshot);
            finished = true;
        }
    };

    // The recording thread (a thread that is not a worker: it takes an external slot of the job system), handed one
    // frame at a time.
    class RecordingThread
    {
    public:
        RecordingThread() : m_Thread([this]() { Run(); }) { }
        ~RecordingThread()
        {
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Stop = true;
            }
            m_Wake.notify_all();
            m_Thread.join();
        }
        void Start(Recorder* recorder)
        {
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Recorder = recorder;
                m_Done = false;
            }
            m_Wake.notify_all();
        }
        void Join()
        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            m_Wake.wait(lock, [this]() { return m_Done; });
        }

    private:
        void Run()
        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            for (;;)
            {
                m_Wake.wait(lock, [this]() { return m_Stop || m_Recorder != nullptr; });
                if (m_Stop)
                    return;
                Recorder* recorder = m_Recorder;
                lock.unlock();
                recorder->Record();
                lock.lock();
                m_Recorder = nullptr;
                m_Done = true;
                m_Wake.notify_all();
            }
        }

        std::mutex m_Mutex;
        std::condition_variable m_Wake;
        Recorder* m_Recorder = nullptr;
        bool m_Done = true;
        bool m_Stop = false;
        std::thread m_Thread;
    };

    struct FrameHashes
    {
        uint64_t snapshot = 0;
        uint64_t items[c_ItemSlots] = {};
        uint32_t itemCounts[c_ItemSlots] = {};
        uint64_t upload = 0;

        bool operator==(const FrameHashes& other) const
        {
            return snapshot == other.snapshot && upload == other.upload &&
                   std::equal(std::begin(items), std::end(items), std::begin(other.items)) &&
                   std::equal(std::begin(itemCounts), std::end(itemCounts), std::begin(other.itemCounts));
        }
    };

    struct RunResult
    {
        std::vector<FrameHashes> frames;
        uint32_t tasksOnMainThread = 0;
        uint32_t walked = 0;
        uint32_t packets = 0;
        uint32_t meshesDestroyed = 0;
    };

    // `frames` frames of the script, serial or pipelined, on a job system of `workers` workers.
    // `stealing`: recording tasks that take 1 ms, the one worker kept busy, and the main thread waiting in the job
    // system after its world work.
    RunResult Run(bool pipelined, uint32_t frames, uint32_t workers, bool stealing = false)
    {
        const uint32_t destroyedBefore = g_MeshesDestroyed.load();
        RunResult result;
        TestViews views;
        RenderFrame renderFrames[2];
        Scene scene(0x7e6, workers);
        scene.Build(160);
        scene.meshesDestroyed = g_MeshesDestroyed.load();
        GpuSceneDesc gpuDesc;
        gpuDesc.uploadMode = GpuSceneUploadMode::Scatter;
        gpuDesc.geometryAndMaterialBuffers = true;
        auto gpuScene = MAKE_RC_OBJ_PTR(GpuScene, nullptr, gpuDesc);
        RenderServiceDesc serviceDesc;
        serviceDesc.sceneGraph = scene.graph;
        serviceDesc.pipelined = pipelined;
        auto service = MAKE_RC_OBJ_PTR(RenderService, nullptr, serviceDesc);
        Recorder recorder;
        recorder.gpuScene = gpuScene;
        recorder.jobs = scene.jobs;
        recorder.walk = pipelined;
        recorder.spinMicroseconds = stealing ? 1000 : 0;
        recorder.mainThread = std::this_thread::get_id();
        RecordingThread thread;

        for (uint32_t frame = 0; frame < frames; ++frame)
        {
            RenderFrame& rf = renderFrames[frame % 2];
            rf.graph->Reset();
            rf.data.Clear();
            SceneFrame* sceneFrame = nullptr;
            CHECK(service->Extract(rf.graph, &rf.data, views.Desc(frame), &sceneFrame) == nvrhi::FS_OK);
            CHECK(sceneFrame && sceneFrame->snapshot);
            const RenderSnapshot& snapshot = *sceneFrame->snapshot;
            CHECK(scene.graph->IsRenderReadDeferring() == pipelined);
            CHECK(snapshot.viewCount <= c_MaxViews);
            for (uint32_t i = 0; i < snapshot.instances.count; ++i)
                CHECK(snapshot.instances.meshInfos[i] != scene.hidden.Get());   // the hidden mesh is named by no view
            FrameHashes hashes;
            hashes.snapshot = HashSnapshot(snapshot);
            for (uint32_t v = 0; v < snapshot.viewCount; ++v)
                for (const SnapshotRange& range : snapshot.views[v].packets)
                    result.packets += range.count;
            // The service's own GpuScene work at extraction (a scene's service does it inside Extract).
            CHECK(gpuScene->Update(scene.graph) == nvrhi::FS_OK);
            CHECK(gpuScene->CreateBuffers() == nvrhi::FS_OK);
            recorder.snapshot = &snapshot;
            recorder.root = scene.graph->GetRootNode();

            if (pipelined)
            {
                CHECK(service->BeginPipelinedRecording() == nvrhi::FS_OK);
                gpuScene->BeginPipelinedRecording();   // a scene's service begins its GpuScene's itself
                recorder.finished = false;
                BusyWorker busy{ &recorder.finished };
                const JobHandle busyJob = stealing ? ScheduleBusyWorker(scene.jobs, busy) : JobHandle::Null;
                thread.Start(&recorder);
                if (frame + 1 < frames)
                    scene.WorldWork(frame + 1, true);   // frame + 1's world, while frame records
                if (stealing)
                    scene.jobs->Wait(busyJob);   // runs the recording's queued tasks meanwhile
                thread.Join();
                gpuScene->EndPipelinedRecording();
                service->EndFrame(true);
                CHECK(!scene.graph->IsInRenderRead() && !SceneGraph::IsAnyRenderReadDeferring());
                // What the overlap dropped went at EndFrame, on this thread: the ephemeral mesh and its material.
                if (scene.droppedMaterial)
                {
                    CHECK(scene.droppedMaterial->IsExpired() != 0);
                    CHECK(g_MeshesDestroyed.load() >= scene.meshesDestroyed + 1);
                    scene.droppedMaterial = nullptr;
                }
                CHECK(!scene.ephemeralMaterial || scene.ephemeralMaterial->IsExpired() == 0);   // the new one lives
                scene.meshesDestroyed = g_MeshesDestroyed.load();
            }
            else
            {
                recorder.Record();
                service->EndFrame(true);
                if (frame + 1 < frames)
                    scene.WorldWork(frame + 1, false);
            }
            if (frame + 1 < frames)
                scene.AfterJoin(frame + 1);

            std::copy(std::begin(recorder.items), std::end(recorder.items), std::begin(hashes.items));
            std::copy(std::begin(recorder.itemCounts), std::end(recorder.itemCounts), std::begin(hashes.itemCounts));
            hashes.upload = recorder.upload;
            CHECK(recorder.snapshotAfter == hashes.snapshot);   // the overlap left the snapshot as extracted
            CHECK(recorder.itemCountMismatches.load() == 0);
            result.frames.push_back(hashes);
        }
        result.tasksOnMainThread = recorder.tasksOnMainThread.load();
        result.walked = recorder.walked;
        service = nullptr;
        scene.ephemeral = nullptr;
        scene.graph->Refresh(frames);
        result.meshesDestroyed = g_MeshesDestroyed.load() - destroyedBefore;
        return result;
    }

    // ------------------------------------------------------------------------------------------------- cases

    // 200 frames pipelined against serial at 4 workers: every frame's snapshot, draw items and upload equal; the
    // walking strategy refused with one warning, no other warning (no owner-check warning from recording work that
    // the main thread ran while it waited in Simulate).
    void test_pipelined_equals_serial()
    {
        const uint32_t warnings = g_Warnings.load();
        const RunResult serial = Run(false, 200, 4);
        CHECK(g_Warnings.load() == warnings);
        const RunResult pipelined = Run(true, 200, 4);
        CHECK(g_Warnings.load() == warnings + 1);   // the walking strategy's refusal, once per object
        g_Warnings = warnings;
        CHECK(pipelined.walked == 0);
        CHECK(serial.frames.size() == pipelined.frames.size());
        uint32_t differing = 0;
        for (size_t f = 0; f < serial.frames.size(); ++f)
            if (!(serial.frames[f] == pipelined.frames[f]))
            {
                if (differing++ < 4)
                    std::fprintf(stderr, "test_render_pipelining: frame %zu differs (snapshot %d, upload %d)\n", f,
                                 int(serial.frames[f].snapshot != pipelined.frames[f].snapshot),
                                 int(serial.frames[f].upload != pipelined.frames[f].upload));
            }
        CHECK(differing == 0);
        CHECK(serial.packets == pipelined.packets && serial.packets > 0);
        CHECK(serial.meshesDestroyed == pipelined.meshesDestroyed && pipelined.meshesDestroyed > 0);
        std::printf("  200 frames: %u packets, %u counted meshes released; %u recording tasks ran on the main "
                    "thread\n", pipelined.packets, pipelined.meshesDestroyed, pipelined.tasksOnMainThread);
    }

    // The same at 1 and 8 workers (60 frames): equal to the serial run at 4.
    void test_pipelined_workers()
    {
        const uint32_t warnings = g_Warnings.load();
        const RunResult serial = Run(false, 60, 4);
        for (const uint32_t workers : { 1u, 8u })
        {
            const RunResult pipelined = Run(true, 60, workers);
            CHECK(pipelined.frames.size() == serial.frames.size());
            for (size_t f = 0; f < serial.frames.size(); ++f)
                CHECK(serial.frames[f] == pipelined.frames[f]);
            std::printf("  %u workers: equal; %u recording tasks on the main thread\n", workers,
                        pipelined.tasksOnMainThread);
        }
        CHECK(g_Warnings.load() == warnings + 2);   // one walking refusal per run
        g_Warnings = warnings;
    }

    // Backlog I6-R1 (fixed in part I7) under the real pattern: on a job system of one worker, kept busy, the main thread
    // waits in the job system during the overlap and runs recording tasks; those are marked, so the owner check stays
    // silent (the walking refusal's one warning only), and the frames still equal the serial run's.
    void test_main_thread_runs_recording()
    {
        const uint32_t warnings = g_Warnings.load();
        const RunResult serial = Run(false, 40, 1);
        const RunResult pipelined = Run(true, 40, 1, true);
        CHECK(g_Warnings.load() == warnings + 1);
        g_Warnings = warnings;
        CHECK(pipelined.frames.size() == serial.frames.size());
        for (size_t f = 0; f < serial.frames.size(); ++f)
            CHECK(serial.frames[f] == pipelined.frames[f]);
        std::printf("  40 frames: %u recording tasks ran on the main thread\n", pipelined.tasksOnMainThread);
        CHECK(pipelined.tasksOnMainThread > 0);
    }

    // Stage T8 (mutants of part I6): what Defer scopes keep goes at the close of the last one, not of an inner Defer
    // scope (two pipelined frames' scopes, or a frame's and the application's); the graph defers until then.
    void test_defer_nested_scopes()
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        nvrhi::AutoPtr<SceneGraphLeaf> light = MAKE_RC_OBJ_PTR(PointLight);
        nvrhi::AutoPtr<SceneGraphNode> node = graph->AttachLeafNode(graph->GetRootNode(), light);
        auto weakNode = WeakOf(node);
        auto weakLight = WeakOf(light);
        light = nullptr;
        graph->Refresh(0);

        graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        (void)graph->Detach(node);
        node = nullptr;
        graph->EndRenderRead();   // the inner Defer scope
        CHECK(graph->IsRenderReadDeferring() && SceneGraph::IsAnyRenderReadDeferring());
        CHECK(weakNode->IsExpired() == 0 && weakLight->IsExpired() == 0);
        graph->EndRenderRead();   // the last one
        CHECK(!graph->IsRenderReadDeferring() && !SceneGraph::IsAnyRenderReadDeferring());
        CHECK(weakNode->IsExpired() != 0 && weakLight->IsExpired() != 0);
    }

    // Stage T8 (part I6): a graph destroyed while a Defer scope is open releases what the scope kept, and no longer
    // counts as deferring for the process (the walking strategies of other graphs walk again).
    void test_defer_scope_at_destruction()
    {
        nvrhi::AutoPtr<nvrhi::IWeakReference> weakNode;
        nvrhi::AutoPtr<nvrhi::IWeakReference> weakLeaf;
        {
            auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
            graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            nvrhi::AutoPtr<SceneGraphLeaf> light = MAKE_RC_OBJ_PTR(SpotLight);
            nvrhi::AutoPtr<SceneGraphNode> node = graph->AttachLeafNode(graph->GetRootNode(), light);
            weakNode = WeakOf(node);
            weakLeaf = WeakOf(light);
            light = nullptr;
            graph->Refresh(0);
            graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
            (void)graph->Detach(node);
            node = nullptr;
            CHECK(weakNode->IsExpired() == 0 && SceneGraph::IsAnyRenderReadDeferring());
        }
        CHECK(weakNode->IsExpired() != 0 && weakLeaf->IsExpired() != 0);
        CHECK(!SceneGraph::IsAnyRenderReadDeferring());
    }

    // Stage T8 (mutant I6-16): a material the tables drop while a Defer scope is open is kept by the retire list itself,
    // not only through the dropped mesh's geometries: here the mesh, still held by the test, lets go of it first.
    void test_defer_retire_material()
    {
        auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        auto material = factory->CreateMaterial();
        material->domain = MaterialDomain::Opaque;
        auto weakMaterial = WeakOf(material);
        auto mesh = factory->CreateMesh();
        auto geometry = factory->CreateMeshGeometry();
        geometry->material = material;
        mesh->geometries.push_back(geometry);
        material = nullptr;
        nvrhi::AutoPtr<SceneGraphNode> node = graph->AttachLeafNode(graph->GetRootNode(),
                                                                    factory->CreateMeshInstance(mesh));
        graph->Refresh(0);

        graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        (void)graph->Detach(node);   // the mesh's last instance: the tables drop the mesh and its material
        node = nullptr;
        mesh->geometries[0]->material = nullptr;
        CHECK(weakMaterial->IsExpired() == 0);   // retired
        graph->EndRenderRead();
        CHECK(weakMaterial->IsExpired() != 0);
    }

    // Stage T8 (mutant I6-29): a walking strategy refused in a pipelined frame gives no item, also after a walk that
    // was not drained (its leftover items are dropped, not handed out).
    void test_walking_refusal_partial()
    {
        TestViews views;
        Scene scene(0x7e8, 2);
        scene.Build(60);
        auto opaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        uint32_t items = 0;
        opaque->PrepareForView(scene.graph->GetRootNode(), *views.perspective);
        while (opaque->GetNextItem())
            ++items;
        CHECK(items >= 2);
        opaque->PrepareForView(scene.graph->GetRootNode(), *views.perspective);
        CHECK(opaque->GetNextItem() != nullptr);   // one taken, the rest left
        const uint32_t warnings = g_Warnings.load();
        scene.graph->BeginRenderRead(SceneGraph::RenderReadMode::Defer);
        opaque->PrepareForView(scene.graph->GetRootNode(), *views.perspective);
        CHECK(opaque->GetNextItem() == nullptr);
        scene.graph->EndRenderRead();
        CHECK(g_Warnings.load() == warnings + 1);
        g_Warnings = warnings;
    }

    // Stage T8 (part I7): BeginPipelinedRecording is refused once the frame has ended (EndFrame) as before the first
    // Extract; IsPipelinedRecording follows the hand-off and the EndFrame; a refused frame keeps no recording.
    void test_service_recording_states()
    {
        TestViews views;
        Scene scene(0x7e7, 2);
        scene.Build(20);
        RenderServiceDesc desc;
        desc.sceneGraph = scene.graph;
        desc.pipelined = true;
        auto service = MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
        RenderFrame frame;
        const uint32_t warnings = g_Warnings.load();
        for (uint32_t f = 0; f < 3; ++f)
        {
            frame.graph->Reset();
            frame.data.Clear();
            SceneFrame* sceneFrame = nullptr;
            CHECK(service->Extract(frame.graph, &frame.data, views.Desc(f), &sceneFrame) == nvrhi::FS_OK);
            CHECK(!service->IsPipelinedRecording());
            CHECK(service->BeginPipelinedRecording() == nvrhi::FS_OK && service->IsPipelinedRecording());
            service->EndFrame(true);
            CHECK(!service->IsPipelinedRecording() && !scene.graph->IsInRenderRead());
            CHECK(service->BeginPipelinedRecording() == nvrhi::FE_INVALID_ARGS);   // the frame ended: one warning
            CHECK(!service->IsPipelinedRecording());
            service->EndFrame(true);   // idempotent
        }
        CHECK(g_Warnings.load() == warnings + 3);
        g_Warnings = warnings;
        service = nullptr;
    }

    struct Case
    {
        const char* name;
        void (*run)();
        const char* backlog;   // an open backlog item this case fails on until it is fixed; run with --backlog
    };
}

int main(int argc, char** argv)
{
#if defined(_MSC_VER)
    _set_error_mode(_OUT_TO_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    const Case cases[] = {
        { "pipelined_equals_serial", test_pipelined_equals_serial, nullptr },
        { "pipelined_workers", test_pipelined_workers, nullptr },
        { "main_thread_runs_recording", test_main_thread_runs_recording, nullptr },
        { "defer_nested_scopes", test_defer_nested_scopes, nullptr },
        { "defer_scope_at_destruction", test_defer_scope_at_destruction, nullptr },
        { "service_recording_states", test_service_recording_states, nullptr },
        { "defer_retire_material", test_defer_retire_material, nullptr },
        { "walking_refusal_partial", test_walking_refusal_partial, nullptr },
    };
    // No argument: every case not marked with a backlog item. --backlog: the marked ones. A name: that case.
    const bool backlog = argc > 1 && std::strcmp(argv[1], "--backlog") == 0;
    const char* only = argc > 1 && !backlog ? argv[1] : nullptr;
    uint32_t failed = 0;
    uint32_t ran = 0;
    for (const Case& c : cases)
    {
        if (only ? std::strcmp(only, c.name) != 0 : backlog != (c.backlog != nullptr))
            continue;
        ++ran;
        g_Warnings = 0;
        std::printf("test_render_pipelining: %s\n", c.name);
        try
        {
            c.run();
            CHECK(g_Warnings.load() == 0);
            std::printf("test_render_pipelining: %s passed%s%s\n", c.name, c.backlog ? " (marked backlog: fixed?) " : "",
                        c.backlog ? c.backlog : "");
        }
        catch (const std::runtime_error& error)
        {
            ++failed;
            std::fprintf(stderr, "%s\n", error.what());
            std::printf("test_render_pipelining: %s FAILED%s%s\n", c.name, c.backlog ? " (backlog) " : "",
                        c.backlog ? c.backlog : "");
        }
    }
    if (only && ran == 0)
    {
        std::fprintf(stderr, "test_render_pipelining: no case %s\n", only);
        return 1;
    }
    return failed ? 1 : 0;
}
