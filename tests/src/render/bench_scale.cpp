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

// bench_scale: design 10.1's rows at the sizes of decision 1, over runtime-only mesh instances (RuntimeMeshInstance,
// SceneComponents.h) made through the public API (aggregate docs/plans/2026-11-26-donut-ecs-e7.md, item E7.3; revision 2,
// stage B). Without a device.
//
//     bench_scale [--smoke] [--rounds N] [--frames N] [--workers N] [--case NAME] [--dynamic N]
//
// Worlds, each a scene graph of its own on the same job system (built when a selected case needs it):
//   dyn      `--dynamic` rows (default 100k) on bench_render_extract's grid (400 a row, 2 m apart), 4 meshes of 1-3
//            geometries, 4 materials (opaque, alpha-tested, blended, double-sided opaque)
//   world    1M static rows (tag Static) on a 1000 x 1000 grid 2 m apart, 3 m below, and the 100k of `dyn` beside them
//   static   the 1M static rows and one dynamic row
//   debris   10k simulation-driven runtime rows and a toy CPU domain (free fall and spin, as bench_simulation's)
// Moves are written by a PreSimulation system (GetComponentForWrite per moved row with the system's version; untimed),
// so that only the chunks (and, with the RowTracked switch, the rows) of moved rows change. Cases, ms per frame:
//   dynamic_100k              Refresh (TransformPropagation, Bounds) after every row of `dyn` moved
//   dynamic_100k_in_world     the same in `world`
//   static_1m                 Refresh of `static` when nothing changed (10.1's literal row: the early-out)
//   static_1m_one_moved       Refresh of `static` after its one dynamic row moved (the row's interpretation)
//   upload_block_10pct, upload_strided_10pct, upload_all   GpuScene scatter Update, CreateBuffers, the records copied
//                             into a staging array and CommitScatter, after 10k consecutive, every 10th, every row of
//                             `dyn` moved (and Refresh, untimed); `_in_world`: the same in `world`
//   extract_view, extract_all RenderService::Extract (no device) over `dyn`, every 10th row moved; the view inside the
//                             grid, and the view above it that sees every row; extract_world_view: the view over `world`
//   move_10k                  AddComponents or RemoveComponents of a tag on every 10th row of `dyn` (10k) through one
//                             command buffer, played back at the sync point after PostSimulation (the PostSimulation
//                             phase with its sync point); add and remove frames alternate, both counted
//   spawn_10k, despawn_10k    CreateEntities and DestroyEntities of 10k runtime rows through a command buffer (slots,
//                             the graph's observer), the phase with its sync point; spawn_100k the same with 100k
//   debris                    SceneGraph::Simulate over `debris` (gather, one fixed step, scatter, interpolation);
//                             debris_refresh: the Refresh after it
//   bytes_per_row             not timed: row bytes (ArchetypeInfo::rowSize) and chunk bytes per row (chunkSize /
//                             rowsPerChunk) of the runtime archetype and of a node-backed mesh instance's
//   bandwidth                 a memcpy of 64 MB: ms and GB/s, the floor each row's bytes per frame are read against
// Breakdown columns: the TransformPropagation and Bounds systems' last run (SystemStats) for Refresh cases; for debris
// the domain's step and scatter and the interpolation. `--smoke` divides the sizes by 100 (ctest bench_scale_smoke).
// Protocol: every round runs every selected case once (the order alternates between rounds); a run keeps the median of
// its frames after two warm-up frames; the table shows the median of the run medians. Release builds only for numbers.

#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>
#include <donut/shaders/gpu_scene_scatter_cb.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;
using namespace donut::render;

namespace bench_scale
{
    struct BenchDynamic { };
    DONUT_ECS_TAG(BenchDynamic, "3b2f7c51-6a0e-4c8e-9b51-2d7f0e6a1c41")
    struct BenchMoved { };
    DONUT_ECS_TAG(BenchMoved, "8c4e2a10-5d3b-4f7a-a6e2-91c0b7d35f62")
}

namespace
{
    using Clock = std::chrono::steady_clock;
    using bench_scale::BenchDynamic;
    using bench_scale::BenchMoved;

    double Milliseconds(Clock::time_point a, Clock::time_point b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    double Median(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    struct Options
    {
        uint32_t dynamic = 100000;
        uint32_t statics = 1000000;
        uint32_t spawn = 10000;
        uint32_t spawnLarge = 100000;
        uint32_t debris = 10000;
        uint32_t rounds = 5;
        uint32_t frames = 20;
        uint32_t workers = 0;   // 0: the shared job system
        std::string only;
    };

    dm::double3 GridPosition(uint32_t i)
    {
        return dm::double3(double(i % 400) * 2.0, 0.0, double(i / 400) * 2.0);
    }

    dm::double3 StaticPosition(uint32_t i)
    {
        return dm::double3(double(i % 1000) * 2.0 + 1.0, -3.0, double(i / 1000) * 2.0 + 1.0);
    }

    enum class MoveMode { None, One, Block, Strided, All };

    // ---------------------------------------------------------------------------------------------- the toy domain

    // Free fall and spin in dense arrays for driven runtime rows (as bench_simulation's FallDomain, without teleports).
    class FallDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(FallDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        struct Chunk { ColumnView bodies, poses, previous; };
        ComponentId pose, previous, body, driven;
        ComponentAccess outputs[3] = {};
        QueryTerm terms[3] = {};
        std::vector<SimPose> poses[2];
        std::vector<dm::double3> velocities;
        uint32_t current = 0;
        std::vector<uint32_t> freeIndices;
        std::vector<Chunk> chunks;
        std::vector<ColumnView> views;

        explicit FallDomain(IWorld* world)
            : pose(FindComponent<SimPose>(world)), previous(FindComponent<SimPosePrevious>(world)),
              body(FindComponent<SimBodyRef>(world)), driven(FindComponent<SimulationDriven>(world))
        {
            outputs[0] = { pose, engine::Access::Write, {} };
            outputs[1] = { previous, engine::Access::Write, {} };
            outputs[2] = { body, engine::Access::Read, {} };
            terms[0] = { driven, TermOp::All, engine::Access::Filter, {} };
            terms[1] = { body, TermOp::All, engine::Access::Write, {} };
            terms[2] = { pose, TermOp::All, engine::Access::Read, {} };
        }
        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override
        {
            retVal = SimulationDomainDesc{};
            retVal.name = "bench_debris";
            retVal.outputs = outputs;
            retVal.outputCount = 3;
            retVal.lifetime.terms = terms;
            retVal.lifetime.termCount = 3;
            retVal.substeps = 1;
            retVal.maxStepsPerFrame = 1;
            retVal.fixedTimeStep = 1.0 / 60.0;
            return retVal;
        }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch& batch, ISimulationContext* context) noexcept override
        {
            SimBodyRef* refs = static_cast<SimBodyRef*>(batch.columns[0]);
            const SimPose* batchPoses = static_cast<const SimPose*>(batch.columns[1]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                uint32_t index = uint32_t(velocities.size());
                if (!freeIndices.empty())
                {
                    index = freeIndices.back();
                    freeIndices.pop_back();
                }
                else
                {
                    poses[0].emplace_back();
                    poses[1].emplace_back();
                    velocities.emplace_back();
                }
                poses[0][index] = poses[1][index] = batchPoses[i];
                velocities[index] = dm::double3(0.125 * double(index % 16), 1.0, 0.0);
                refs[i] = SimBodyRef{ uint32_t(context->GetDomainId()), index };
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch& batch, ISimulationContext*) noexcept override
        {
            SimBodyRef* refs = static_cast<SimBodyRef*>(batch.columns[0]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                freeIndices.push_back(refs[i].index);
                refs[i] = SimBodyRef{};
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT Step(double dt, ISimulationContext* context) noexcept override
        {
            struct Call { const SimPose* from; SimPose* to; dm::double3* velocities; double dt; };
            Call call{ poses[current].data(), poses[current ^ 1].data(), velocities.data(), dt };
            context->GetJobSystem()->ParallelFor(uint32_t(velocities.size()), 1024,
                [](void* data, uint32_t begin, uint32_t end, uint32_t) noexcept
                {
                    const Call& c = *static_cast<const Call*>(data);
                    for (uint32_t i = begin; i < end; ++i)
                    {
                        dm::double3& velocity = c.velocities[i];
                        velocity.y -= 9.0 * c.dt;
                        const dm::double3 translation = c.from[i].translation + velocity * c.dt;
                        if (translation.y < -100.0)
                            velocity.y = std::abs(velocity.y);
                        const dm::dquat& q = c.from[i].rotation;
                        const dm::dquat r(q.w - 0.01 * q.y, q.x + 0.01 * q.z, q.y + 0.01 * q.w, q.z - 0.01 * q.x);
                        c.to[i].translation = translation;
                        c.to[i].rotation = r / std::sqrt(r.w * r.w + r.x * r.x + r.y * r.y + r.z * r.z);
                    }
                }, &call);
            current ^= 1;
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept override
        {
            if (context->GetStepsThisFrame() == 0)
                return nvrhi::FS_OK;
            const uint32_t count = context->GetColumnCount(body);
            if (chunks.size() < count)
            {
                chunks.resize(count);
                views.resize(count);
            }
            const uint32_t n = context->GetColumns(body, 0, c_AllVersions, count, views.data());
            for (uint32_t i = 0; i < n; ++i)
                chunks[i].bodies = views[i];
            context->GetColumns(pose, 0, c_AllVersions, count, views.data());
            for (uint32_t i = 0; i < n; ++i)
                chunks[i].poses = views[i];
            context->GetColumns(previous, 0, c_AllVersions, count, views.data());
            for (uint32_t i = 0; i < n; ++i)
                chunks[i].previous = views[i];
            context->GetJobSystem()->ParallelFor(n, 4, [](void* data, uint32_t begin, uint32_t end, uint32_t) noexcept
            {
                FallDomain& domain = *static_cast<FallDomain*>(data);
                const SimPose* last = domain.poses[domain.current].data();
                const SimPose* before = domain.poses[domain.current ^ 1].data();
                for (uint32_t c = begin; c < end; ++c)
                {
                    const Chunk& chunk = domain.chunks[c];
                    const SimBodyRef* refs = static_cast<const SimBodyRef*>(chunk.bodies.base);
                    SimPose* rowPoses = static_cast<SimPose*>(chunk.poses.base);
                    SimPosePrevious* rowPrevious = static_cast<SimPosePrevious*>(chunk.previous.base);
                    for (uint32_t row = 0; row < chunk.bodies.count; ++row)
                    {
                        const uint32_t index = refs[row].index;
                        rowPoses[row] = last[index];
                        rowPrevious[row] = SimPosePrevious{ before[index].translation, before[index].rotation };
                    }
                }
            }, this);
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT CaptureState(IStateWriter*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
        nvrhi::FRESULT RestoreState(IStateReader*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
    };

    // ---------------------------------------------------------------------------------------------------- worlds

    struct World
    {
        nvrhi::AutoPtr<SceneGraph> graph;
        IWorld* world = nullptr;
        std::vector<nvrhi::AutoPtr<Material>> materials;
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
        MeshId meshIds[4] = {};
        std::vector<ComponentId> plain;
        std::vector<ComponentId> driven;
        ComponentId local = ComponentId::Invalid;
        ComponentId meshRef = ComponentId::Invalid;
        ComponentId pose = ComponentId::Invalid;
        ComponentId previous = ComponentId::Invalid;
        ComponentId scale = ComponentId::Invalid;
        ComponentId dynamicTag = ComponentId::Invalid;
        ComponentId staticTag = ComponentId::Invalid;
        ComponentId movedTag = ComponentId::Invalid;
        std::vector<Entity> dynamicRows;
        std::vector<Entity> staticRows;
        uint32_t frame = 0;
        double time = 0.0;

        // The mover's request for the next PreSimulation run.
        MoveMode mode = MoveMode::None;
        double offset = 0.0;
        // The structural system's request for the next PostSimulation run.
        enum class Structural { None, AddTag, RemoveTag, Spawn, Despawn } structural = Structural::None;
        uint32_t spawnCount = 0;
        std::vector<Entity> spawned;        // resolved after a spawn's playback
        std::vector<Entity> placeholders;
        std::vector<Entity> tagged;         // every 10th dynamic row
        std::vector<LocalTransform> spawnLocals;
        std::vector<MeshRef> spawnRefs;

        // Breakdown sources.
        SystemId propagation = SystemId::Invalid;
        SystemId bounds = SystemId::Invalid;
        SystemId step = SystemId::Invalid;
        SystemId scatter = SystemId::Invalid;
        SystemId interpolation = SystemId::Invalid;

        // Upload and extraction, made when a case needs them.
        nvrhi::AutoPtr<GpuScene> gpu;
        std::vector<uint8_t> staging;
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> renderGraph;
        FrameData frameData;
        nvrhi::AutoPtr<RenderService> service;
        nvrhi::AutoPtr<PlanarView> views[2];
        const IView* viewList[2] = {};
        nvrhi::AutoPtr<FallDomain> domain;
        ComponentAccess moverAccess[1] = {};
    };

    void MakeAssets(World& w)
    {
        const MaterialDomain domains[4] = { MaterialDomain::Opaque, MaterialDomain::AlphaTested,
                                            MaterialDomain::AlphaBlended, MaterialDomain::Opaque };
        for (int i = 0; i < 4; ++i)
        {
            auto material = MAKE_RC_OBJ_PTR(Material);
            material->name = "material" + std::to_string(i);
            material->domain = domains[i];
            material->doubleSided = i == 3;
            w.materials.push_back(material);
        }
        for (int i = 0; i < 4; ++i)
        {
            auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
            mesh->name = "mesh" + std::to_string(i);
            dm::box3 bounds = dm::box3::empty();
            for (int g = 0; g <= i % 3; ++g)
            {
                auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
                geometry->material = w.materials[(i + g) % 4];
                geometry->numIndices = 36;
                geometry->numVertices = 24;
                const float x = -0.75f + 0.5f * float(g);
                geometry->objectSpaceBounds = dm::box3(dm::float3(x, -0.5f, -0.5f), dm::float3(x + 0.5f, 0.5f, 0.5f));
                bounds |= geometry->objectSpaceBounds;
                mesh->geometries.push_back(geometry);
            }
            mesh->objectSpaceBounds = bounds;
            w.meshes.push_back(mesh);
        }
    }

    SystemId FindSystem(IScheduler* scheduler, const char* name)
    {
        for (uint32_t i = 0; i < scheduler->GetSystemCount(); ++i)
        {
            SystemInfo info{};
            if (scheduler->GetSystemInfo(i, info) == nvrhi::FS_OK && info.name && !std::strcmp(info.name, name))
                return info.id;
        }
        return SystemId::Invalid;
    }

    // The system's last run, or 0 when it has not run since `runCount` (updated): a phase that did not run.
    double SystemMs(SceneGraph* graph, SystemId id, uint64_t& runCount)
    {
        SystemStats stats{};
        if (id == SystemId::Invalid || graph->GetScheduler()->GetSystemStats(id, stats) != nvrhi::FS_OK)
            return 0.0;
        const bool ran = stats.runCount != runCount;
        runCount = stats.runCount;
        return ran ? stats.lastMilliseconds : 0.0;
    }

    template <typename F>
    bool AddLambdaSystem(World& w, const char* name, Phase phase, const ComponentAccess* accesses, uint32_t count,
                         F&& update)
    {
        SystemDesc desc{};
        desc.name = name;
        desc.phase = phase;
        desc.flags = SystemFlags::MainThread;
        desc.accesses = accesses;
        desc.accessCount = count;
        nvrhi::AutoPtr<ISystem> system;
        if (NVRHI_FAILED(CreateLambdaSystem(desc, std::forward<F>(update), system.ReleaseAndGetAddressOf())))
            return false;
        SystemId id = SystemId::Invalid;
        return NVRHI_SUCCEEDED(w.graph->GetScheduler()->AddSystem(system, id));
    }

    // Rows with the plain runtime layout plus `extra` tags, at `position(i)`, meshes i % 4.
    bool CreateRows(World& w, uint32_t count, dm::double3 (*position)(uint32_t), const std::vector<ComponentId>& extra,
                    std::vector<Entity>& out)
    {
        std::vector<ComponentId> ids = w.plain;
        ids.insert(ids.end(), extra.begin(), extra.end());
        const uint32_t batch = 65536;
        std::vector<LocalTransform> locals(std::min(count, batch));
        std::vector<MeshRef> refs(locals.size());
        std::vector<const void*> values(ids.size(), nullptr);
        for (size_t c = 0; c < ids.size(); ++c)
        {
            if (ids[c] == w.local)
                values[c] = locals.data();
            else if (ids[c] == w.meshRef)
                values[c] = refs.data();
        }
        const size_t first = out.size();
        out.resize(first + count, Entity::Null);
        for (uint32_t start = 0; start < count; start += batch)
        {
            const uint32_t n = std::min(batch, count - start);
            for (uint32_t i = 0; i < n; ++i)
            {
                locals[i] = LocalTransform{ position(start + i), dm::dquat::identity(), dm::double3(1.0) };
                refs[i] = MeshRef{ w.meshIds[(start + i) % 4] };
            }
            EntityCreateDesc desc{};
            desc.components = ids.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(ids.size());
            desc.entityCount = n;
            if (NVRHI_FAILED(w.world->CreateEntities(desc, out.data() + first + start)))
                return false;
        }
        return true;
    }

    // A world of `dynamicCount` moving rows and `staticCount` static ones on the job system `jobs` (null: shared).
    bool BuildWorld(World& w, uint32_t dynamicCount, uint32_t staticCount, IJobSystem* jobs)
    {
        MakeAssets(w);
        w.graph = MAKE_RC_OBJ_PTR(SceneGraph);
        auto root = MAKE_RC_OBJ_PTR(SceneGraphNode);
        root->SetName("root");
        w.graph->SetRootNode(root);
        if (jobs)
            w.graph->SetJobSystem(jobs);
        w.world = w.graph->GetWorld();
        for (uint32_t m = 0; m < 4; ++m)
        {
            w.meshIds[m] = w.graph->AcquireRuntimeMesh(w.meshes[m]);
            if (w.meshIds[m] == MeshId::Invalid)
                return false;
        }
        w.plain.resize(w.graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
        w.graph->GetRuntimeMeshInstanceComponents(w.plain.data(), uint32_t(w.plain.size()));
        w.driven.resize(w.graph->GetRuntimeMeshInstanceComponents(nullptr, 0, true));
        w.graph->GetRuntimeMeshInstanceComponents(w.driven.data(), uint32_t(w.driven.size()), true);
        w.local = FindComponent<LocalTransform>(w.world);
        w.meshRef = FindComponent<MeshRef>(w.world);
        w.pose = FindComponent<SimPose>(w.world);
        w.previous = FindComponent<SimPosePrevious>(w.world);
        w.scale = FindComponent<SimScale>(w.world);
        w.dynamicTag = RegisterComponent<BenchDynamic>(w.world);
        w.staticTag = RegisterComponent<Static>(w.world);
        w.movedTag = RegisterComponent<BenchMoved>(w.world);
        if (!CreateRows(w, staticCount, &StaticPosition, { w.staticTag }, w.staticRows) ||
            !CreateRows(w, dynamicCount, &GridPosition, { w.dynamicTag }, w.dynamicRows))
            return false;
        for (size_t i = 0; i < w.dynamicRows.size(); i += 10)
            w.tagged.push_back(w.dynamicRows[i]);

        // The mover: GetComponentForWrite per moved row with the system's version.
        w.moverAccess[0] = ComponentAccess{ w.local, engine::Access::Write, {} };
        World* self = &w;
        if (!AddLambdaSystem(w, "bench.mover", Phase::PreSimulation, w.moverAccess, 1, [self](ISystemContext* context)
            {
                World& s = *self;
                const uint32_t n = uint32_t(s.dynamicRows.size());
                if (s.mode == MoveMode::None || !n)
                    return;
                IWorld* world = context->GetWorld();
                const uint64_t version = context->GetVersion();
                auto move = [&](uint32_t i)
                {
                    void* value = world->GetComponentForWrite(s.dynamicRows[i], s.local, version);
                    if (value)
                    {
                        dm::double3 position = GridPosition(i);
                        position.y = s.offset;
                        static_cast<LocalTransform*>(value)->translation = position;
                    }
                };
                switch (s.mode)
                {
                case MoveMode::One: move(n / 2); break;
                case MoveMode::Block: for (uint32_t i = n / 3; i < n / 3 + n / 10; ++i) move(i); break;
                case MoveMode::Strided: for (uint32_t i = 0; i < n; i += 10) move(i); break;
                case MoveMode::All: for (uint32_t i = 0; i < n; ++i) move(i); break;
                default: break;
                }
            }))
            return false;

        // The structural system: one command buffer, played back at the sync point after PostSimulation.
        if (!AddLambdaSystem(w, "bench.structural", Phase::PostSimulation, nullptr, 0, [self](ISystemContext* context)
            {
                World& s = *self;
                ICommandBuffer* commands = context->GetCommandBuffer();
                switch (s.structural)
                {
                case World::Structural::AddTag:
                    commands->AddComponents(s.tagged.data(), uint32_t(s.tagged.size()), s.movedTag, nullptr,
                                            MakeSortKey(0, 0, 0));
                    break;
                case World::Structural::RemoveTag:
                    commands->RemoveComponents(s.tagged.data(), uint32_t(s.tagged.size()), s.movedTag,
                                               MakeSortKey(0, 0, 0));
                    break;
                case World::Structural::Spawn:
                {
                    std::vector<const void*> values(s.plain.size(), nullptr);
                    for (size_t c = 0; c < s.plain.size(); ++c)
                    {
                        if (s.plain[c] == s.local)
                            values[c] = s.spawnLocals.data();
                        else if (s.plain[c] == s.meshRef)
                            values[c] = s.spawnRefs.data();
                    }
                    EntityCreateDesc desc{};
                    desc.components = s.plain.data();
                    desc.values = values.data();
                    desc.componentCount = uint32_t(s.plain.size());
                    desc.entityCount = s.spawnCount;
                    commands->CreateEntities(desc, MakeSortKey(0, 0, 0), s.placeholders.data());
                    break;
                }
                case World::Structural::Despawn:
                    commands->DestroyEntities(s.spawned.data(), uint32_t(s.spawned.size()), MakeSortKey(0, 0, 0));
                    break;
                default:
                    break;
                }
            }))
            return false;

        IScheduler* scheduler = w.graph->GetScheduler();
        w.propagation = FindSystem(scheduler, "SceneGraph.TransformPropagation");
        w.bounds = FindSystem(scheduler, "SceneGraph.Bounds");
        w.interpolation = FindSystem(scheduler, "SceneGraph.SimulationInterpolation");
        w.graph->Refresh(++w.frame);
        w.graph->Refresh(++w.frame);
        return true;
    }

    // Moves of `mode` (untimed: the PreSimulation phase alone).
    void Move(World& w, MoveMode mode)
    {
        w.mode = mode;
        w.offset = (w.frame & 1) ? 0.5 : 0.0;
        if (mode != MoveMode::None)
            w.graph->GetScheduler()->RunPhase(Phase::PreSimulation);
        w.mode = MoveMode::None;
    }

    void EnsureUpload(World& w)
    {
        if (w.gpu)
            return;
        GpuSceneDesc desc;
        desc.geometryAndMaterialBuffers = false;
        desc.rowGranularity = 1024;
        desc.uploadMode = GpuSceneUploadMode::Scatter;
        w.gpu = MAKE_RC_OBJ_PTR(GpuScene, nullptr, desc);
        w.staging.resize(size_t(w.dynamicRows.size() + w.staticRows.size() + 1024) *
                         (sizeof(GpuSceneRowRecord) + sizeof(GpuSceneTransformRecord) + sizeof(uint32_t)));
    }

    // The scatter upload: Update, CreateBuffers, the records into the staging array, CommitScatter. Returns rows.
    uint64_t Upload(World& w)
    {
        w.gpu->Update(w.graph);
        w.gpu->CreateBuffers();
        GpuSceneScatterRecords records;
        w.gpu->GetScatterRecords(records);
        uint8_t* out = w.staging.data();
        auto copy = [&](const void* data, size_t size)
        {
            if (size)
                std::memcpy(out, data, size);
            out += size;
        };
        copy(records.rows, size_t(records.rowCount) * sizeof(GpuSceneRowRecord));
        copy(records.transforms, size_t(records.transformCount) * sizeof(GpuSceneTransformRecord));
        copy(records.settleSlots, size_t(records.settleCount) * sizeof(uint32_t));
        w.gpu->CommitScatter(records.serial);
        return uint64_t(records.rowCount) + records.transformCount + records.settleCount;
    }

    void EnsureExtraction(World& w)
    {
        if (w.service)
            return;
        CreateResourceRegistry(nullptr, &w.registry);
        CreateRenderGraph(w.registry, &w.renderGraph);
        RenderServiceDesc desc;
        desc.sceneGraph = w.graph;
        w.service = MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
        // bench_render_extract's view inside the grid (100k rows deep) and its view above everything
        const uint32_t count = std::max(uint32_t(w.dynamicRows.size()), 1u);
        const float depth = float((count + 399) / 400) * 2.f;
        for (auto& view : w.views)
            view = MAKE_RC_OBJ_PTR(PlanarView);
        w.views[0]->SetViewport(nvrhi::Viewport(1920.f, 1080.f));
        const dm::affine3 eye = dm::translation(dm::float3(400.f, 6.f, depth * 0.45f));
        w.views[0]->SetMatrices(dm::inverse(eye), dm::perspProjD3DStyleReverse(dm::radians(60.f), 16.f / 9.f, 0.1f));
        w.views[0]->UpdateCache();
        w.views[1]->SetViewport(nvrhi::Viewport(1920.f, 1080.f));
        const dm::affine3 above = dm::rotation(dm::float3(1.f, 0.f, 0.f), dm::radians(90.f)) *
                                  dm::translation(dm::float3(400.f, 700.f, depth * 0.5f));
        w.views[1]->SetMatrices(dm::inverse(above), dm::perspProjD3DStyleReverse(dm::radians(90.f), 16.f / 9.f, 0.1f));
        w.views[1]->UpdateCache();
        w.viewList[0] = w.views[0].Get();
        w.viewList[1] = w.views[1].Get();
    }

    // Extract with view `view`; returns the visible instances.
    uint64_t Extract(World& w, uint32_t view)
    {
        w.renderGraph->Reset();
        w.frameData.Clear();
        ExtractDesc desc;
        desc.views = w.viewList + view;
        desc.viewCount = 1;
        desc.frameIndex = w.frame;
        SceneFrame* frame = nullptr;
        w.service->Extract(w.renderGraph, &w.frameData, desc, &frame);
        w.service->EndFrame(true);
        return w.service->GetStats().visibleInstances;
    }

    // The PostSimulation phase with its sync point, for the structural request set before it.
    double RunStructural(World& w)
    {
        const auto t0 = Clock::now();
        w.graph->GetScheduler()->RunPhase(Phase::PostSimulation);
        const auto t1 = Clock::now();
        w.structural = World::Structural::None;
        return Milliseconds(t0, t1);
    }

    void PrepareSpawn(World& w, uint32_t count)
    {
        w.spawnCount = count;
        w.spawnLocals.resize(count);
        w.spawnRefs.resize(count);
        w.placeholders.assign(count, Entity::Null);
        for (uint32_t i = 0; i < count; ++i)
        {
            dm::double3 position = GridPosition(i);
            position.y = 5.0;
            w.spawnLocals[i] = LocalTransform{ position, dm::dquat::identity(), dm::double3(1.0) };
            w.spawnRefs[i] = MeshRef{ w.meshIds[i % 4] };
        }
    }

    // Spawn (timed into `spawnMs`) and despawn (into `despawnMs`) of `count` rows, with the Refresh after each untimed.
    void SpawnPair(World& w, uint32_t count, double& spawnMs, double& despawnMs)
    {
        PrepareSpawn(w, count);
        w.structural = World::Structural::Spawn;
        spawnMs = RunStructural(w);
        w.spawned.resize(count);
        for (uint32_t i = 0; i < count; ++i)
            w.spawned[i] = w.world->ResolvePlaceholder(w.placeholders[i]);
        w.graph->Refresh(++w.frame);
        w.structural = World::Structural::Despawn;
        despawnMs = RunStructural(w);
        w.spawned.clear();
        w.graph->Refresh(++w.frame);
    }

    bool BuildDebris(World& w, uint32_t count, IJobSystem* jobs)
    {
        if (!BuildWorld(w, 0, 0, jobs))
            return false;
        std::vector<ComponentId> ids = w.driven;
        std::vector<LocalTransform> locals(count);
        std::vector<MeshRef> refs(count);
        std::vector<SimPose> poses(count);
        std::vector<SimPosePrevious> previous(count);
        std::vector<SimScale> scales(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            const dm::double3 position = GridPosition(i);
            locals[i] = LocalTransform{ position, dm::dquat::identity(), dm::double3(1.0) };
            refs[i] = MeshRef{ w.meshIds[i % 4] };
            poses[i] = SimPose{ position, dm::dquat::identity() };
            previous[i] = SimPosePrevious{ position, dm::dquat::identity() };
            scales[i] = SimScale{ dm::double3(1.0) };
        }
        std::vector<const void*> values(ids.size(), nullptr);
        for (size_t c = 0; c < ids.size(); ++c)
        {
            if (ids[c] == w.local) values[c] = locals.data();
            else if (ids[c] == w.meshRef) values[c] = refs.data();
            else if (ids[c] == w.pose) values[c] = poses.data();
            else if (ids[c] == w.previous) values[c] = previous.data();
            else if (ids[c] == w.scale) values[c] = scales.data();
        }
        EntityCreateDesc desc{};
        desc.components = ids.data();
        desc.values = values.data();
        desc.componentCount = uint32_t(ids.size());
        desc.entityCount = count;
        std::vector<Entity> rows(count);
        if (NVRHI_FAILED(w.world->CreateEntities(desc, rows.data())))
            return false;
        w.domain = MAKE_RC_OBJ_PTR(FallDomain, w.world);
        SimulationDomainId id = SimulationDomainId::Invalid;
        if (NVRHI_FAILED(w.graph->GetScheduler()->AddSimulationDomain(w.domain, id)))
            return false;
        IScheduler* scheduler = w.graph->GetScheduler();
        w.step = FindSystem(scheduler, "bench_debris.step");
        w.scatter = FindSystem(scheduler, "bench_debris.scatter");
        w.interpolation = FindSystem(scheduler, "SceneGraph.SimulationInterpolation");
        w.graph->Refresh(++w.frame);
        return true;
    }

    // ------------------------------------------------------------------------------------------------- the cases

    struct Sample
    {
        double ms = 0.0;
        double a = 0.0;       // breakdown: propagation, step
        double b = 0.0;       // breakdown: bounds, scatter
        double c = 0.0;       // breakdown: interpolation
        uint64_t count = 0;   // rows uploaded, visible instances, ...
    };

    struct Bench
    {
        Options options;
        nvrhi::AutoPtr<IJobSystem> jobs;
        std::unique_ptr<World> dyn, world, statics, debris;

        World& Dyn()
        {
            if (!dyn)
            {
                dyn = std::make_unique<World>();
                if (!BuildWorld(*dyn, options.dynamic, 0, jobs))
                    std::abort();
            }
            return *dyn;
        }
        World& Full()
        {
            if (!world)
            {
                world = std::make_unique<World>();
                if (!BuildWorld(*world, options.dynamic, options.statics, jobs))
                    std::abort();
            }
            return *world;
        }
        World& Static()
        {
            if (!statics)
            {
                statics = std::make_unique<World>();
                if (!BuildWorld(*statics, 1, options.statics, jobs))
                    std::abort();
            }
            return *statics;
        }
        World& Debris()
        {
            if (!debris)
            {
                debris = std::make_unique<World>();
                if (!BuildDebris(*debris, options.debris, jobs))
                    std::abort();
            }
            return *debris;
        }

        // Frames of a Refresh case: the moves, then the timed Refresh.
        std::vector<Sample> RefreshFrames(World& w, MoveMode mode)
        {
            std::vector<Sample> samples;
            uint64_t runs[2] = { ~0ull, ~0ull };
            SystemMs(w.graph, w.propagation, runs[0]);
            SystemMs(w.graph, w.bounds, runs[1]);
            for (uint32_t f = 0; f < options.frames + 2; ++f)
            {
                Move(w, mode);
                const auto t0 = Clock::now();
                w.graph->Refresh(++w.frame);
                const auto t1 = Clock::now();
                Sample s;
                s.ms = Milliseconds(t0, t1);
                s.a = SystemMs(w.graph, w.propagation, runs[0]);
                s.b = SystemMs(w.graph, w.bounds, runs[1]);
                if (f >= 2)
                    samples.push_back(s);
            }
            return samples;
        }

        std::vector<Sample> UploadFrames(World& w, MoveMode mode)
        {
            EnsureUpload(w);
            // the previous case's moves settle first
            for (int i = 0; i < 3; ++i)
            {
                w.graph->Refresh(++w.frame);
                Upload(w);
            }
            std::vector<Sample> samples;
            for (uint32_t f = 0; f < options.frames + 2; ++f)
            {
                Move(w, mode);
                w.graph->Refresh(++w.frame);
                const auto t0 = Clock::now();
                const uint64_t rows = Upload(w);
                const auto t1 = Clock::now();
                if (f >= 2)
                    samples.push_back(Sample{ Milliseconds(t0, t1), 0.0, 0.0, 0.0, rows });
            }
            return samples;
        }

        std::vector<Sample> ExtractFrames(World& w, uint32_t view)
        {
            EnsureExtraction(w);
            std::vector<Sample> samples;
            for (uint32_t f = 0; f < options.frames + 2; ++f)
            {
                Move(w, MoveMode::Strided);
                w.graph->Refresh(++w.frame);
                const auto t0 = Clock::now();
                const uint64_t visible = Extract(w, view);
                const auto t1 = Clock::now();
                if (f >= 2)
                    samples.push_back(Sample{ Milliseconds(t0, t1), 0.0, 0.0, 0.0, visible });
            }
            return samples;
        }

        std::vector<Sample> MoveFrames(World& w)
        {
            std::vector<Sample> samples;
            const uint32_t pairs = (options.frames + 3) / 2;
            for (uint32_t p = 0; p < pairs; ++p)
            {
                for (int half = 0; half < 2; ++half)
                {
                    w.structural = half ? World::Structural::RemoveTag : World::Structural::AddTag;
                    const double ms = RunStructural(w);
                    w.graph->Refresh(++w.frame);
                    if (p >= 1)
                        samples.push_back(Sample{ ms, 0.0, 0.0, 0.0, w.tagged.size() });
                }
            }
            return samples;
        }

        std::vector<Sample> SpawnFrames(World& w, uint32_t count, bool despawn)
        {
            std::vector<Sample> samples;
            for (uint32_t f = 0; f < options.frames + 2; ++f)
            {
                double spawnMs = 0.0, despawnMs = 0.0;
                SpawnPair(w, count, spawnMs, despawnMs);
                if (f >= 2)
                    samples.push_back(Sample{ despawn ? despawnMs : spawnMs, 0.0, 0.0, 0.0, count });
            }
            return samples;
        }

        std::vector<Sample> DebrisFrames(World& w, bool refresh)
        {
            std::vector<Sample> samples;
            uint64_t runs[3] = { ~0ull, ~0ull, ~0ull };
            for (uint32_t f = 0; f < options.frames + 2; ++f)
            {
                w.time += 1.0 / 60.0;
                const auto t0 = Clock::now();
                w.graph->Simulate(FrameTime{ w.time, 1.0 / 60.0, w.frame });
                const auto t1 = Clock::now();
                w.graph->Refresh(++w.frame);
                const auto t2 = Clock::now();
                Sample s;
                s.ms = refresh ? Milliseconds(t1, t2) : Milliseconds(t0, t1);
                s.a = SystemMs(w.graph, refresh ? w.propagation : w.step, runs[0]);
                s.b = SystemMs(w.graph, refresh ? w.bounds : w.scatter, runs[1]);
                s.c = refresh ? 0.0 : SystemMs(w.graph, w.interpolation, runs[2]);
                if (f >= 2)
                    samples.push_back(s);
            }
            return samples;
        }

        std::vector<Sample> BandwidthFrames()
        {
            const size_t bytes = size_t(64) << 20;
            std::vector<uint8_t> from(bytes, 1), to(bytes, 0);
            std::vector<Sample> samples;
            for (uint32_t f = 0; f < options.frames + 2; ++f)
            {
                from[f % bytes] = uint8_t(f);
                const auto t0 = Clock::now();
                std::memcpy(to.data(), from.data(), bytes);
                const auto t1 = Clock::now();
                const double ms = Milliseconds(t0, t1);
                if (f >= 2)
                    samples.push_back(Sample{ ms, 0.0, 0.0, 0.0, uint64_t(double(bytes) / ms / 1.0e6) });
            }
            return samples;
        }
    };

    struct Case
    {
        const char* name;
        std::function<std::vector<Sample>(Bench&)> run;
        const char* columns;   // what the breakdown columns and the count are
    };

    // bytes_per_row: the runtime archetype of `dyn` and a node-backed mesh instance's archetype.
    void BytesPerRow(Bench& bench)
    {
        auto report = [](const char* what, IWorld* world, ComponentId tag)
        {
            for (uint32_t a = 0; a < world->GetArchetypeCount(); ++a)
            {
                ArchetypeInfo info{};
                if (world->GetArchetypeInfo(a, info) != nvrhi::FS_OK || !info.entityCount)
                    continue;
                if (std::find(info.components, info.components + info.componentCount, tag) ==
                    info.components + info.componentCount)
                    continue;
                std::printf("%-26s %12u %12.1f %12u %12u\n", what, info.rowSize,
                            double(info.chunkSize) / double(info.rowsPerChunk), info.rowsPerChunk, info.entityCount);
                return;
            }
        };
        std::printf("%-26s %12s %12s %12s %12s\n", "bytes_per_row", "row bytes", "chunk bytes", "rows/chunk",
                    "rows");
        World& w = bench.Dyn();
        report("runtime_mesh_instance", w.world, w.dynamicTag);
        // a node-backed population of the same meshes
        World nodes;
        MakeAssets(nodes);
        nodes.graph = MAKE_RC_OBJ_PTR(SceneGraph);
        auto root = MAKE_RC_OBJ_PTR(SceneGraphNode);
        nodes.graph->SetRootNode(root);
        for (uint32_t i = 0; i < 1000; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(GridPosition(i));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, nodes.meshes[i % 4].Get()));
            nodes.graph->Attach(nodes.graph->GetRootNode(), node);
        }
        nodes.graph->Refresh(1);
        report("node_mesh_instance", nodes.graph->GetWorld(), FindComponent<MeshInstanceLeaf>(nodes.graph->GetWorld()));
    }
}

int main(int argc, char** argv)
{
    Options options;
    bool smoke = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--smoke")
            smoke = true;
        else if (arg == "--rounds" && i + 1 < argc)
            options.rounds = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--frames" && i + 1 < argc)
            options.frames = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--workers" && i + 1 < argc)
            options.workers = std::max(1u, uint32_t(std::stoul(argv[++i])));
        else if (arg == "--case" && i + 1 < argc)
            options.only = argv[++i];
        else if (arg == "--dynamic" && i + 1 < argc)
            options.dynamic = uint32_t(std::stoul(argv[++i]));
    }
    if (smoke)
    {
        options.dynamic /= 100;
        options.statics /= 100;
        options.spawn /= 100;
        options.spawnLarge /= 100;
        options.debris /= 100;
        options.rounds = 1;
        options.frames = 3;
    }

    Bench bench;
    bench.options = options;
    if (options.workers)
    {
        JobSystemDesc jobDesc;
        jobDesc.debugName = "bench_scale";
        jobDesc.workerThreadCount = options.workers;
        if (NVRHI_FAILED(CreateJobSystem(jobDesc, &bench.jobs)))
        {
            std::fprintf(stderr, "cannot create a job system with %u workers\n", options.workers);
            return 1;
        }
    }

    const Case cases[] = {
        { "dynamic_100k", [](Bench& b) { return b.RefreshFrames(b.Dyn(), MoveMode::All); }, "prop/bounds" },
        { "dynamic_100k_in_world", [](Bench& b) { return b.RefreshFrames(b.Full(), MoveMode::All); }, "prop/bounds" },
        { "static_1m", [](Bench& b) { return b.RefreshFrames(b.Static(), MoveMode::None); }, "prop/bounds" },
        { "static_1m_one_moved", [](Bench& b) { return b.RefreshFrames(b.Static(), MoveMode::One); }, "prop/bounds" },
        { "upload_block_10pct", [](Bench& b) { return b.UploadFrames(b.Dyn(), MoveMode::Block); }, "rows" },
        { "upload_strided_10pct", [](Bench& b) { return b.UploadFrames(b.Dyn(), MoveMode::Strided); }, "rows" },
        { "upload_all", [](Bench& b) { return b.UploadFrames(b.Dyn(), MoveMode::All); }, "rows" },
        { "upload_block_10pct_in_world", [](Bench& b) { return b.UploadFrames(b.Full(), MoveMode::Block); }, "rows" },
        { "upload_strided_10pct_in_world", [](Bench& b) { return b.UploadFrames(b.Full(), MoveMode::Strided); },
          "rows" },
        { "upload_all_in_world", [](Bench& b) { return b.UploadFrames(b.Full(), MoveMode::All); }, "rows" },
        { "extract_view", [](Bench& b) { return b.ExtractFrames(b.Dyn(), 0); }, "visible" },
        { "extract_all", [](Bench& b) { return b.ExtractFrames(b.Dyn(), 1); }, "visible" },
        { "extract_world_view", [](Bench& b) { return b.ExtractFrames(b.Full(), 0); }, "visible" },
        { "move_10k", [](Bench& b) { return b.MoveFrames(b.Dyn()); }, "rows" },
        { "spawn_10k", [](Bench& b) { return b.SpawnFrames(b.Dyn(), b.options.spawn, false); }, "rows" },
        { "despawn_10k", [](Bench& b) { return b.SpawnFrames(b.Dyn(), b.options.spawn, true); }, "rows" },
        { "spawn_100k", [](Bench& b) { return b.SpawnFrames(b.Dyn(), b.options.spawnLarge, false); }, "rows" },
        { "debris", [](Bench& b) { return b.DebrisFrames(b.Debris(), false); }, "step/scatter/interp" },
        { "debris_refresh", [](Bench& b) { return b.DebrisFrames(b.Debris(), true); }, "prop/bounds" },
        { "bandwidth", [](Bench& b) { return b.BandwidthFrames(); }, "GB/s" },
    };
    const uint32_t caseCount = uint32_t(std::size(cases));

    // Per case: the run medians of each field.
    struct Runs { std::vector<double> ms, a, b, c; uint64_t count = 0; };
    std::vector<Runs> runs(caseCount);
    bool any = false;
    for (uint32_t round = 0; round < options.rounds; ++round)
    {
        for (uint32_t k = 0; k < caseCount; ++k)
        {
            const uint32_t c = (round & 1) ? caseCount - 1 - k : k;
            if (!options.only.empty() && options.only != cases[c].name)
                continue;
            any = true;
            const std::vector<Sample> samples = cases[c].run(bench);
            std::vector<double> ms, a, b, cc;
            for (const Sample& s : samples)
            {
                ms.push_back(s.ms);
                a.push_back(s.a);
                b.push_back(s.b);
                cc.push_back(s.c);
            }
            runs[c].ms.push_back(Median(ms));
            runs[c].a.push_back(Median(a));
            runs[c].b.push_back(Median(b));
            runs[c].c.push_back(Median(cc));
            runs[c].count = samples.empty() ? 0 : samples.back().count;
        }
    }

    const std::string threads = options.workers ? std::to_string(options.workers) + " workers" : "shared job system";
    std::printf("bench_scale: %u dynamic, %u static, spawn %u / %u, debris %u; %u rounds of %u frames (median of run "
                "medians), %s\n", options.dynamic, options.statics, options.spawn, options.spawnLarge, options.debris,
                options.rounds, options.frames, threads.c_str());
    std::printf("%-30s %10s %10s %10s %10s %12s  %s\n", "case", "ms", "a ms", "b ms", "c ms", "count", "a/b/c, count");
    for (uint32_t c = 0; c < caseCount; ++c)
    {
        if (runs[c].ms.empty())
            continue;
        std::printf("%-30s %10.4f %10.4f %10.4f %10.4f %12llu  %s\n", cases[c].name, Median(runs[c].ms),
                    Median(runs[c].a), Median(runs[c].b), Median(runs[c].c), (unsigned long long)runs[c].count,
                    cases[c].columns);
    }
    if (options.only.empty() || options.only == "bytes_per_row")
    {
        BytesPerRow(bench);
        any = true;
    }
    if (!any)
    {
        std::fprintf(stderr, "no case named %s\n", options.only.c_str());
        return 2;
    }
    return 0;
}
