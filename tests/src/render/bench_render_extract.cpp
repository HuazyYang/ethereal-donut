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

// The CPU cost of visibility and draw packets (aggregate docs/plans/2026-11-21-donut-ecs-e5.md, item E5.2; design 10.1
// "visibility and packet generation, 100k instances, one view: < 1 ms on 8 workers"). Without a device.
//
// One source, two builds:
// - default (step E5 and later): RenderService::Extract (its RenderExtract phase: views, lights, cameras, skinned
//   instances, then visibility and packets), and beside it the legacy strategy walk of the same build;
// - BENCH_RENDER_EXTRACT_E0=1, compiled against the E0 tree (donut f49809b, from outside that tree): the legacy
//   strategy walk only.
// The legacy walk is InstancedOpaqueDrawStrategy and TransparentDrawStrategy, each PrepareForView and draining
// GetNextItem, per view: what a geometry pass does today.
//
// Scene: 100k mesh instances under the root on a 400 x 250 grid 2 m apart, 4 meshes of 1-3 geometries (each geometry
// its own box), 4 materials (opaque, alpha-tested, blended, double-sided opaque); every 10th instance moves every frame
// (10 %). Views: a perspective view inside the grid (a fifth of the instances visible), four orthographic views along
// the sun as a cascaded shadow map would set them up, and a perspective view above the grid that sees all of it.
// Cases (a frame: the moves, SceneGraph::Refresh, then the timed work; the Refresh is timed apart, "refresh ms"):
//   view            the view inside the grid
//   cascades        that view and the four cascade views
//   all             the view above the grid: every instance visible
// Release: the median of the run medians over alternating rounds (`--rounds N`, `--frames N`); `--smoke` divides the
// instances by 100; `--workers N`: the graph and the service on a job system of N worker threads (the thread that calls
// Extract works too), default build only; without it, the process's shared job system; `--case NAME`: that case only.
// `--population runtime` (plan step E7, revision 2, stage M; default build only): the instances are runtime-only mesh
// instances (RuntimeMeshInstance, SceneComponents.h) made with one IWorld::CreateEntities instead of node-backed
// MeshInstance leaves (`node`, the default); the moves are written by a PreSimulation system run by hand before the
// frame's Refresh, untimed as the node setters are. The legacy walk sees no runtime row, so only `extract` is reported.

#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/GeometryPasses.h>
#if !BENCH_RENDER_EXTRACT_E0
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneComponents.h>
#include <donut/render/RenderExtract.h>
#include <donut/render/RenderGraph.h>
#include <donut/render/RenderGraphFrameData.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace donut;
using namespace donut::engine;
using namespace donut::render;

namespace
{
    struct Options
    {
        uint32_t instances = 100000;
        uint32_t rounds = 5;
        uint32_t frames = 20;
        uint32_t workers = 0;   // 0: the shared job system
        std::string only;       // one case by name, or every case
        bool runtime = false;   // --population runtime
    };

#if !BENCH_RENDER_EXTRACT_E0
    // --population runtime: the rows, and the moves of a frame, which a PreSimulation system writes.
    struct RuntimeRows
    {
        std::vector<Entity> entities;      // in creation order: row i is instance i
        ComponentId local = ComponentId::Invalid;
        std::vector<uint32_t> moved;       // this frame's rows ...
        std::vector<dm::double3> targets;  // ... and their translations
        ComponentAccess access[1] = {};

        bool Make(SceneGraph* graph, const std::vector<nvrhi::AutoPtr<MeshInfo>>& meshes, uint32_t count,
                  dm::double3 (*position)(uint32_t))
        {
            IWorld* world = graph->GetWorld();
            MeshId ids[4] = {};
            for (uint32_t m = 0; m < 4; ++m)
            {
                ids[m] = graph->AcquireRuntimeMesh(meshes[m]);
                if (ids[m] == MeshId::Invalid)
                    return false;
            }
            std::vector<ComponentId> components(graph->GetRuntimeMeshInstanceComponents(nullptr, 0));
            graph->GetRuntimeMeshInstanceComponents(components.data(), uint32_t(components.size()));
            local = FindComponent<LocalTransform>(world);
            const ComponentId meshRef = FindComponent<MeshRef>(world);
            std::vector<LocalTransform> locals(count);
            std::vector<MeshRef> refs(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                locals[i] = LocalTransform{ position(i), dm::dquat::identity(), dm::double3(1.0) };
                refs[i] = MeshRef{ ids[i % 4] };
            }
            std::vector<const void*> values(components.size(), nullptr);
            for (size_t c = 0; c < components.size(); ++c)
            {
                if (components[c] == local)
                    values[c] = locals.data();
                else if (components[c] == meshRef)
                    values[c] = refs.data();
            }
            EntityCreateDesc desc{};
            desc.components = components.data();
            desc.values = values.data();
            desc.componentCount = uint32_t(components.size());
            desc.entityCount = count;
            entities.assign(count, Entity::Null);
            // the graph's observer holds a mesh reference per row; ours go once the rows exist
            const bool made = NVRHI_SUCCEEDED(world->CreateEntities(desc, entities.data()));
            for (MeshId id : ids)
                graph->ReleaseRuntimeMesh(id);
            moved.reserve(count);
            targets.reserve(count);
            return made;
        }

        // On the graph's current scheduler (after SetJobSystem, which drops the systems of the one before).
        bool AddMover(SceneGraph* graph)
        {
            access[0] = ComponentAccess{ local, engine::Access::Write, {} };
            SystemDesc desc{};
            desc.name = "bench.runtime_moves";
            desc.phase = engine::Phase::PreSimulation;
            desc.flags = SystemFlags::MainThread;
            desc.accesses = access;
            desc.accessCount = 1;
            RuntimeRows* self = this;
            nvrhi::AutoPtr<ISystem> system;
            if (NVRHI_FAILED(CreateLambdaSystem(desc, [self](ISystemContext* context) noexcept
                {
                    IWorld* world = context->GetWorld();
                    for (size_t k = 0; k < self->moved.size(); ++k)
                    {
                        void* value = world->GetComponentForWrite(self->entities[self->moved[k]], self->local,
                                                                  context->GetVersion());
                        if (value)
                            static_cast<LocalTransform*>(value)->translation = self->targets[k];
                    }
                }, system.ReleaseAndGetAddressOf())))
                return false;
            SystemId id = SystemId::Invalid;
            return NVRHI_SUCCEEDED(graph->GetScheduler()->AddSystem(system, id));
        }

        // The queued moves, written by the system; outside the timed work.
        void Apply(SceneGraph* graph)
        {
            if (!moved.empty())
                graph->GetScheduler()->RunPhase(engine::Phase::PreSimulation);
            moved.clear();
            targets.clear();
        }
    };
#endif

    struct BenchScene
    {
        nvrhi::AutoPtr<SceneGraph> graph;
        std::vector<nvrhi::AutoPtr<Material>> materials;
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
        std::vector<SceneGraphNode*> nodes;
#if !BENCH_RENDER_EXTRACT_E0
        RuntimeRows* runtime = nullptr;   // --population runtime: the rows instead of `nodes`
#endif
        uint32_t count = 0;               // instances
        nvrhi::AutoPtr<PlanarView> views[6];
        const IView* list[6] = {};
        uint32_t frame = 0;
    };

    dm::double3 GridPosition(uint32_t i)
    {
        return dm::double3(double(i % 400) * 2.0, 0.0, double(i / 400) * 2.0);
    }

    void Build(BenchScene& s, uint32_t count, bool runtime)
    {
        s.count = count;
        const MaterialDomain domains[4] = { MaterialDomain::Opaque, MaterialDomain::AlphaTested,
                                            MaterialDomain::AlphaBlended, MaterialDomain::Opaque };
        for (int i = 0; i < 4; ++i)
        {
            auto material = MAKE_RC_OBJ_PTR(Material);
            material->name = "material" + std::to_string(i);
            material->domain = domains[i];
            material->doubleSided = i == 3;
            s.materials.push_back(material);
        }
        for (int i = 0; i < 4; ++i)
        {
            auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
            mesh->name = "mesh" + std::to_string(i);
            dm::box3 bounds = dm::box3::empty();
            for (int g = 0; g <= i % 3; ++g)
            {
                auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
                geometry->material = s.materials[(i + g) % 4];
                geometry->numIndices = 36;
                geometry->numVertices = 24;
                const float x = -0.75f + 0.5f * float(g);
                geometry->objectSpaceBounds = dm::box3(dm::float3(x, -0.5f, -0.5f), dm::float3(x + 0.5f, 0.5f, 0.5f));
                bounds |= geometry->objectSpaceBounds;
                mesh->geometries.push_back(geometry);
            }
            mesh->objectSpaceBounds = bounds;
            s.meshes.push_back(mesh);
        }
        s.graph = MAKE_RC_OBJ_PTR(SceneGraph);
        auto root = MAKE_RC_OBJ_PTR(SceneGraphNode);
        root->SetName("root");
        s.graph->SetRootNode(root);
        for (uint32_t i = 0; i < count && !runtime; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(GridPosition(i));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, s.meshes[i % 4].Get()));
            s.graph->Attach(s.graph->GetRootNode(), node);
            s.nodes.push_back(node);
        }

        // the main view: inside the grid, looking along +z; a fifth of the instances in its frustum
        const float depth = float((count + 399) / 400) * 2.f;
        for (auto& view : s.views)
            view = MAKE_RC_OBJ_PTR(PlanarView);
        s.views[0]->SetViewport(nvrhi::Viewport(1920.f, 1080.f));
        const dm::affine3 eye = dm::translation(dm::float3(400.f, 6.f, depth * 0.45f));
        s.views[0]->SetMatrices(dm::inverse(eye), dm::perspProjD3DStyleReverse(dm::radians(60.f), 16.f / 9.f, 0.1f));
        s.views[0]->UpdateCache();
        // four cascades along the sun, around the view's position, of growing extent
        const dm::float3 sun = dm::normalize(dm::float3(0.3f, -1.f, 0.4f));
        const float extents[4] = { 15.f, 40.f, 100.f, 250.f };
        for (uint32_t c = 0; c < 4; ++c)
        {
            PlanarView* view = s.views[1 + c];
            view->SetViewport(nvrhi::Viewport(2048.f, 2048.f));
            const dm::float3 center = dm::float3(400.f, 0.f, depth * 0.45f + extents[c]);
            const dm::affine3 light = dm::lookatZ(sun) * dm::translation(center - sun * 200.f);
            const float e = extents[c];
            view->SetMatrices(dm::inverse(light), dm::orthoProjD3DStyle(-e, e, -e, e, 0.f, 400.f));
            view->UpdateCache();
        }
        // above the grid's centre, looking down: every instance in its frustum
        s.views[5]->SetViewport(nvrhi::Viewport(1920.f, 1080.f));
        const dm::affine3 above = dm::rotation(dm::float3(1.f, 0.f, 0.f), dm::radians(90.f)) *
                                  dm::translation(dm::float3(400.f, 700.f, depth * 0.5f));
        s.views[5]->SetMatrices(dm::inverse(above), dm::perspProjD3DStyleReverse(dm::radians(90.f), 16.f / 9.f, 0.1f));
        s.views[5]->UpdateCache();
        for (uint32_t v = 0; v < 6; ++v)
            s.list[v] = s.views[v].Get();
    }

    // every 10th instance up and down by half a metre
    void Move(BenchScene& s)
    {
        const double offset = double(s.frame & 1) * 0.5;
        for (uint32_t i = 0; i < s.count; i += 10)
        {
            dm::double3 position = GridPosition(i);
            position.y = offset;
#if !BENCH_RENDER_EXTRACT_E0
            if (s.runtime)
            {
                s.runtime->moved.push_back(i);
                s.runtime->targets.push_back(position);
                continue;
            }
#endif
            s.nodes[i]->SetTranslation(position);
        }
#if !BENCH_RENDER_EXTRACT_E0
        if (s.runtime)
            s.runtime->Apply(s.graph);
#endif
    }

    enum class Case { View, Cascades, All, Count };
    const char* const c_CaseNames[] = { "view", "cascades", "all" };
    uint32_t FirstView(Case c) { return c == Case::All ? 5 : 0; }
    uint32_t ViewCount(Case c) { return c == Case::Cascades ? 5 : 1; }

    // The legacy strategies over one view: what InstancedOpaqueDrawStrategy and TransparentDrawStrategy give a
    // geometry pass. Their objects are kept, as a pass keeps its strategy.
    struct LegacyWalk
    {
        nvrhi::AutoPtr<InstancedOpaqueDrawStrategy> opaque = MAKE_RC_OBJ_PTR(InstancedOpaqueDrawStrategy);
        nvrhi::AutoPtr<TransparentDrawStrategy> transparent = MAKE_RC_OBJ_PTR(TransparentDrawStrategy);
        uint64_t items = 0;

        void Run(BenchScene& s, uint32_t firstView, uint32_t viewCount)
        {
            items = 0;
            for (uint32_t v = firstView; v < firstView + viewCount; ++v)
            {
                opaque->PrepareForView(s.graph->GetRootNode(), *s.list[v]);
                while (opaque->GetNextItem())
                    ++items;
                transparent->PrepareForView(s.graph->GetRootNode(), *s.list[v]);
                while (transparent->GetNextItem())
                    ++items;
            }
        }
    };

#if !BENCH_RENDER_EXTRACT_E0
    struct Extraction
    {
        nvrhi::AutoPtr<IResourceRegistry> registry;
        nvrhi::AutoPtr<IRenderGraph> graph;
        FrameData data;
        nvrhi::AutoPtr<RenderService> service;
        uint64_t items = 0;
        uint64_t instances = 0;

        explicit Extraction(SceneGraph* sceneGraph)
        {
            CreateResourceRegistry(nullptr, &registry);
            CreateRenderGraph(registry, &graph);
            RenderServiceDesc desc;
            desc.sceneGraph = sceneGraph;
            service = MAKE_RC_OBJ_PTR(RenderService, nullptr, desc);
        }

        bool Run(BenchScene& s, uint32_t firstView, uint32_t viewCount)
        {
            graph->Reset();
            data.Clear();
            ExtractDesc desc;
            desc.views = s.list + firstView;
            desc.viewCount = viewCount;
            desc.frameIndex = s.frame;
            SceneFrame* frame = nullptr;
            const bool ok = NVRHI_SUCCEEDED(service->Extract(graph, &data, desc, &frame));
            service->EndFrame(true);
            const RenderServiceStats& stats = service->GetStats();
            items = uint64_t(stats.packets[0]) + stats.packets[1];
            instances = stats.visibleInstances;
            return ok;
        }
    };
#endif

    struct Sample
    {
        double ms;
        double refreshMs;   // the frame's SceneGraph::Refresh before the timed work
        uint64_t items;
        uint64_t instances;
    };

    double Median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    // One run of a case: warm-up frames, then `frames` timed ones; the median of the run.
    template <typename Work>
    Sample Run(BenchScene& s, Work& work, Case c, uint32_t frames)
    {
        using clock = std::chrono::steady_clock;
        std::vector<double> times, refreshes;
        Sample sample{};
        for (uint32_t f = 0; f < frames + 2; ++f)
        {
            Move(s);
            const auto r0 = clock::now();
            s.graph->Refresh(++s.frame);
            const auto t0 = clock::now();
            work.Run(s, FirstView(c), ViewCount(c));
            const auto t1 = clock::now();
            if (f < 2)
                continue;   // warm-up
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            refreshes.push_back(std::chrono::duration<double, std::milli>(t0 - r0).count());
            sample.items = work.items;
#if !BENCH_RENDER_EXTRACT_E0
            if constexpr (std::is_same_v<Work, Extraction>)
                sample.instances = work.instances;
#endif
        }
        sample.ms = Median(times);
        sample.refreshMs = Median(refreshes);
        return sample;
    }

    template <typename Work>
    void Report(const char* name, const std::vector<std::vector<Sample>>& samples)
    {
        for (uint32_t c = 0; c < uint32_t(Case::Count); ++c)
        {
            if (samples[c].empty())
                continue;
            std::vector<double> ms, refresh;
            for (const Sample& sample : samples[c])
            {
                ms.push_back(sample.ms);
                refresh.push_back(sample.refreshMs);
            }
            std::printf("%-10s %-10s %10.4f %12.4f %12llu %12llu\n", name, c_CaseNames[c], Median(ms), Median(refresh),
                        (unsigned long long)samples[c].back().items, (unsigned long long)samples[c].back().instances);
        }
    }
}

int main(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--smoke")
        {
            options.instances = 1000;
            options.rounds = 1;
            options.frames = 3;
        }
        else if (arg == "--rounds" && i + 1 < argc)
            options.rounds = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--frames" && i + 1 < argc)
            options.frames = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--instances" && i + 1 < argc)
            options.instances = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--workers" && i + 1 < argc)
            options.workers = std::max(1u, uint32_t(std::stoul(argv[++i])));
        else if (arg == "--case" && i + 1 < argc)
            options.only = argv[++i];
        else if (arg == "--population" && i + 1 < argc)
        {
            const std::string population = argv[++i];
            if (population != "node" && population != "runtime")
            {
                std::fprintf(stderr, "--population takes node or runtime\n");
                return 2;
            }
            options.runtime = population == "runtime";
        }
    }

    BenchScene scene;
#if BENCH_RENDER_EXTRACT_E0
    if (options.workers || options.runtime)
    {
        std::fprintf(stderr, "--workers and --population runtime need the default build\n");
        return 2;
    }
    Build(scene, options.instances, false);
#else
    nvrhi::AutoPtr<IJobSystem> jobs;
    if (options.workers)
    {
        JobSystemDesc jobDesc;
        jobDesc.debugName = "bench_render_extract";
        jobDesc.workerThreadCount = options.workers;
        if (NVRHI_FAILED(CreateJobSystem(jobDesc, &jobs)))
        {
            std::fprintf(stderr, "cannot create a job system with %u workers\n", options.workers);
            return 1;
        }
    }
    Build(scene, options.instances, options.runtime);
    if (jobs)
        scene.graph->SetJobSystem(jobs);
    RuntimeRows runtimeRows;
    if (options.runtime)
    {
        if (!runtimeRows.Make(scene.graph, scene.meshes, options.instances, &GridPosition) ||
            !runtimeRows.AddMover(scene.graph))
        {
            std::fprintf(stderr, "cannot make the runtime population\n");
            return 1;
        }
        scene.runtime = &runtimeRows;
    }
    Extraction extraction(scene.graph);
#endif
    scene.graph->Refresh(++scene.frame);
    LegacyWalk legacy;

    const uint32_t caseCount = uint32_t(Case::Count);
    std::vector<std::vector<Sample>> legacySamples(caseCount);
#if !BENCH_RENDER_EXTRACT_E0
    std::vector<std::vector<Sample>> extractSamples(caseCount);
#endif
    for (uint32_t round = 0; round < options.rounds; ++round)
    {
        // alternate the order of the cases and of the two paths between rounds
        for (uint32_t k = 0; k < caseCount; ++k)
        {
            const uint32_t c = (round & 1) ? caseCount - 1 - k : k;
            if (!options.only.empty() && options.only != c_CaseNames[c])
                continue;
#if !BENCH_RENDER_EXTRACT_E0
            if (options.runtime)
                extractSamples[c].push_back(Run(scene, extraction, Case(c), options.frames));
            else if (round & 1)
            {
                legacySamples[c].push_back(Run(scene, legacy, Case(c), options.frames));
                extractSamples[c].push_back(Run(scene, extraction, Case(c), options.frames));
            }
            else
            {
                extractSamples[c].push_back(Run(scene, extraction, Case(c), options.frames));
                legacySamples[c].push_back(Run(scene, legacy, Case(c), options.frames));
            }
#else
            legacySamples[c].push_back(Run(scene, legacy, Case(c), options.frames));
#endif
        }
    }

    const std::string threads = options.workers ? std::to_string(options.workers) + " workers" : "shared job system";
    std::printf("%s, %u %sinstances, %u rounds of %u frames (median of run medians), %s\n",
#if BENCH_RENDER_EXTRACT_E0
                "E0 legacy strategy walk",
#else
                "RenderService::Extract and the legacy strategy walk",
#endif
                options.instances, options.runtime ? "runtime-only " : "", options.rounds, options.frames,
                threads.c_str());
    std::printf("%-10s %-10s %10s %12s %12s %12s\n", "path", "case", "ms", "refresh ms", "draws", "visible");
#if BENCH_RENDER_EXTRACT_E0
    Report<LegacyWalk>("legacy_e0", legacySamples);
#else
    Report<Extraction>("extract", extractSamples);
    Report<LegacyWalk>("legacy", legacySamples);
#endif
    return 0;
}
