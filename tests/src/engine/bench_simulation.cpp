// bench_simulation: what simulation-driven nodes cost (aggregate docs/plans/2026-11-21-donut-ecs-e6.md, items E6.2a and
// E6.2b and its performance table).
//
//     bench_simulation [--smoke] [--rounds N] [--workers N]
//
// A flat scene of 100k instances under the root, of which 10k (every tenth) or all 100k are driven by a toy engine
// behind a Cpu domain: free fall and spin in dense arrays (the poses before and after a step in two that swap), copied
// through the column views, both loops split on the job system, one step per frame, the interpolation halfway between
// the last two poses. Cases (the sizes are divided
// by 100 with --smoke, which ctest runs), all in ms per frame:
//     <n>_step              the FixedSimulation phase (the toy engine)
//     <n>_scatter           the PostSimulation phase (both poses of every driven node)
//     <n>_interpolation     the SimulationInterpolation phase (their world transforms)
//     <n>_scatter_interp    the two together, per frame
//     <n>_refresh           the Refresh after them (propagation, bounds, subgraphs)
//     <n>_setters           for comparison, without a simulation: SetTransform of the same nodes in the same scene
//     <n>_setters_refresh   ... and the Refresh after it
//     <n>_simulate_refresh  a frame as an application has it: SceneGraph::Simulate with one step, and Refresh
//     teleports_set         the setters of 1k of 100k driven nodes: teleport requests, all parts but one or two
//     teleports_gather      BeginFrame to PreSimulation with them: resolving, and the domain's GatherInputs
//     teleports_gather_none the same without a request
//     teleports_frame       Simulate with one step, and Refresh, with them
//     bind                  SetSimulationDriven(true) of 10k nodes of 100k, with the domain's OnEntitiesAdded
//     unbind                SetSimulationDriven(false) of them, with OnEntitiesRemoved
//     detach                Detach of a subgraph of 10k driven nodes
//     still                 all phases of a frame and the Refresh in which nothing moves, with 100k driven nodes
// Protocol as in bench_scene_graph: every round runs every case once; a run keeps the median of its iterations, the
// table shows the median of the run medians and their range. Meaningful in Release builds only.

#include <donut/engine/EcsQuery.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    using Clock = std::chrono::steady_clock;

    double Milliseconds(Clock::time_point start, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double Median(std::vector<double> samples)
    {
        if (samples.empty())
            return 0.0;
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }

    // The graph's scheduler makes the phase of the interpolation first; named phases get their values from Count up.
    constexpr Phase c_Interpolation = Phase(uint32_t(Phase::Count));

    // The job system of --workers, or null for the shared one.
    nvrhi::AutoPtr<IJobSystem> g_Jobs;

    class FallDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(FallDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        struct ScatterChunk
        {
            ColumnView bodies, poses, previous;
        };

        static constexpr double c_TimeStep = 1.0 / 64.0;
        ComponentId pose, previous, body, driven, scale, teleport;
        std::vector<ComponentAccess> inputs;
        std::vector<ComponentAccess> outputs;
        Query<Read<SimBodyRef>, Write<SimPose>, Write<SimPosePrevious>, Write<SimScale>, Enabled<SimTeleport>> requests;
        uint64_t teleports = 0;                 // taken
        std::vector<QueryTerm> terms;
        std::vector<SimPose> poses[2];          // [current], and the poses before the last step
        std::vector<dm::double3> velocities;
        uint32_t current = 0;
        std::vector<uint32_t> freeIndices;
        std::vector<ScatterChunk> chunks;
        std::vector<ColumnView> views;

        explicit FallDomain(IWorld* world)
            : pose(FindComponent<SimPose>(world)), previous(FindComponent<SimPosePrevious>(world)),
              body(FindComponent<SimBodyRef>(world)), driven(FindComponent<SimulationDriven>(world)),
              scale(FindComponent<SimScale>(world)), teleport(FindComponent<SimTeleport>(world)),
              requests(world, "requests")
        {
            inputs = { { teleport, Access::Write, {} }, { scale, Access::Write, {} }, { pose, Access::Write, {} },
                       { previous, Access::Write, {} }, { body, Access::Read, {} } };
            outputs = { { pose, Access::Write, {} }, { previous, Access::Write, {} }, { body, Access::Read, {} } };
            terms = { { driven, TermOp::All, Access::Filter, {} }, { body, TermOp::All, Access::Write, {} },
                      { pose, TermOp::All, Access::Read, {} } };
        }

        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override
        {
            retVal = SimulationDomainDesc{};
            retVal.name = "fall";
            retVal.inputs = inputs.data();
            retVal.inputCount = uint32_t(inputs.size());
            retVal.outputs = outputs.data();
            retVal.outputCount = uint32_t(outputs.size());
            retVal.lifetime.terms = terms.data();
            retVal.lifetime.termCount = uint32_t(terms.size());
            retVal.substeps = 1;
            retVal.maxStepsPerFrame = 4;
            retVal.fixedTimeStep = c_TimeStep;
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
                refs[i].domain = uint32_t(context->GetDomainId());
                refs[i].index = index;
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
        nvrhi::FRESULT GatherInputs(ISimulationContext* context) noexcept override
        {
            if (context->GetStepsThisFrame() == 0)
                return nvrhi::FS_OK;
            requests.ForEachChunk(context->GetSystemContext(),
                [&](const ChunkView& view, const SimBodyRef* refs, SimPose* rowPoses, SimPosePrevious* rowPrevious,
                    SimScale* rowScales, const SimTeleport* targets)
            {
                for (uint32_t row = 0; row < view.count; ++row)
                {
                    if (!view.IsRowEnabled(4, row))
                        continue;
                    poses[current][refs[row].index] = rowPoses[row] =
                        SimPose{ targets[row].translation, targets[row].rotation };
                    rowPrevious[row] = SimPosePrevious{ targets[row].translation, targets[row].rotation };
                    rowScales[row].scaling = targets[row].scaling;
                    context->GetWorld()->SetEnabled(view.entities[row], teleport, false);
                    ++teleports;
                }
            });
            return nvrhi::FS_OK;
        }
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
            chunks.resize(count);
            views.resize(count);
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
                for (uint32_t c = begin; c < end; ++c)
                {
                    const ScatterChunk& chunk = domain.chunks[c];
                    const SimBodyRef* refs = static_cast<const SimBodyRef*>(chunk.bodies.base);
                    const SimPose* last = domain.poses[domain.current].data();
                    const SimPose* before = domain.poses[domain.current ^ 1].data();
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

    struct Scene
    {
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        nvrhi::AutoPtr<MeshInfo> mesh;
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        nvrhi::AutoPtr<FallDomain> domain;
        std::vector<SceneGraphNode*> nodes;
        uint32_t frame = 0;
        double time = 0.0;

        explicit Scene(size_t count)
        {
            auto geometry = factory->CreateMeshGeometry();
            geometry->material = factory->CreateMaterial();
            geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
            mesh = factory->CreateMesh();
            mesh->geometries.push_back(geometry);
            mesh->objectSpaceBounds = geometry->objectSpaceBounds;
            graph->SetJobSystem(g_Jobs);
            graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
            AddNodes(graph->GetRootNode(), count);
            graph->Refresh(frame++);
            graph->Refresh(frame++);
            domain = MAKE_RC_OBJ_PTR(FallDomain, graph->GetWorld());
            SimulationDomainId id = SimulationDomainId::Invalid;
            if (NVRHI_FAILED(graph->GetScheduler()->AddSimulationDomain(domain, id)))
                std::abort();
        }
        void AddNodes(SceneGraphNode* parent, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                node->SetTranslation(dm::double3(double(i % 317), double(i % 53), double(i / 317)));
                node->SetRotation(dm::dquat(0.8, 0.6, 0.0, 0.0));
                node->SetLeaf(factory->CreateMeshInstance(mesh));
                nodes.push_back(graph->Attach(parent, node));
            }
        }
        std::vector<SceneGraphNode*> Every(size_t stride) const
        {
            std::vector<SceneGraphNode*> result;
            for (size_t i = 0; i < nodes.size(); i += stride)
                result.push_back(nodes[i]);
            return result;
        }
        void SetTime(double delta)
        {
            time += delta;
            graph->GetScheduler()->SetFrameTime(FrameTime{ time, delta, frame });
        }
        // SceneGraph::Simulate and Refresh.
        void Frame(double delta)
        {
            time += delta;
            if (NVRHI_FAILED(graph->Simulate(FrameTime{ time, delta, frame })))
                std::abort();
            graph->Refresh(frame++);
        }
        void Run(Phase first, Phase last)
        {
            if (NVRHI_FAILED(graph->GetScheduler()->RunPhases(first, last)))
                std::abort();
        }
    };

    struct Sizes
    {
        size_t nodes = 100000;
        size_t few = 10000;
        size_t events = 10000;
        size_t teleports = 1000;
        int iterations = 40;
        int eventIterations = 3;
    };

    struct Result
    {
        std::string name;
        size_t nodes = 0;
        int iterations = 0;
        std::vector<double> runMedians;
    };

    class Bench
    {
    public:
        explicit Bench(const Sizes& sizes) : m_Sizes(sizes) { }

        void Round()
        {
            Driven("few", m_Sizes.few);
            Driven("all", m_Sizes.nodes);
            Events();
        }

        void Print(const char* build, int rounds, int workers) const
        {
            char with[32] = "";
            if (workers > 0)
                std::snprintf(with, sizeof(with), ", %d workers", workers);
            std::printf("bench_simulation (%s build, %d rounds%s; %zu nodes, few = %zu driven, all = %zu; median of the "
                        "run medians, ms)\n", build, rounds, with, m_Sizes.nodes, m_Sizes.few, m_Sizes.nodes);
            std::printf("%-26s %9s %6s %12s %12s %12s\n", "case", "nodes", "iters", "median", "min run", "max run");
            for (const Result& r : m_Results)
            {
                const auto [lo, hi] = std::minmax_element(r.runMedians.begin(), r.runMedians.end());
                std::printf("%-26s %9zu %6d %12.6f %12.6f %12.6f\n", r.name.c_str(), r.nodes, r.iterations,
                    Median(r.runMedians), *lo, *hi);
            }
        }

    private:
        Sizes m_Sizes;
        std::vector<Result> m_Results;

        void Record(const std::string& name, size_t nodes, int iterations, double runMedian)
        {
            for (Result& r : m_Results)
            {
                if (r.name == name)
                {
                    r.runMedians.push_back(runMedian);
                    return;
                }
            }
            m_Results.push_back(Result{ name, nodes, iterations, { runMedian } });
        }

        void Driven(const char* prefix, size_t count)
        {
            const std::string name = std::string(prefix) + "_";
            Scene scene(m_Sizes.nodes);
            const auto driven = scene.Every(m_Sizes.nodes / count);

            // the same moves through the setters, before anything is driven
            std::vector<double> set, setRefresh;
            for (int it = 0; it < m_Sizes.iterations; ++it)
            {
                const auto start = Clock::now();
                for (size_t i = 0; i < driven.size(); ++i)
                {
                    const dm::double3 translation(double(it), double(i % 53), double(i / 317));
                    const dm::dquat rotation(0.6, 0.0, 0.8, 0.0);
                    driven[i]->SetTransform(&translation, &rotation, nullptr);
                }
                const auto moved = Clock::now();
                scene.graph->Refresh(scene.frame++);
                const auto end = Clock::now();
                set.push_back(Milliseconds(start, moved));
                setRefresh.push_back(Milliseconds(moved, end));
            }

            for (SceneGraphNode* node : driven)
            {
                if (NVRHI_FAILED(scene.graph->SetSimulationDriven(node, true)))
                    std::abort();
            }
            scene.SetTime(0.5 * FallDomain::c_TimeStep);   // the interpolation stays halfway
            scene.Run(Phase::BeginFrame, c_Interpolation);
            scene.graph->Refresh(scene.frame++);
            std::vector<double> step, scatter, interpolation, both, refresh;
            for (int it = 0; it < m_Sizes.iterations; ++it)
            {
                scene.SetTime(FallDomain::c_TimeStep);
                scene.Run(Phase::BeginFrame, Phase::PreSimulation);
                const auto t0 = Clock::now();
                scene.Run(Phase::FixedSimulation, Phase::FixedSimulation);
                const auto t1 = Clock::now();
                scene.Run(Phase::PostSimulation, Phase::PostSimulation);
                const auto t2 = Clock::now();
                scene.Run(c_Interpolation, c_Interpolation);
                const auto t3 = Clock::now();
                scene.graph->Refresh(scene.frame++);
                const auto t4 = Clock::now();
                step.push_back(Milliseconds(t0, t1));
                scatter.push_back(Milliseconds(t1, t2));
                interpolation.push_back(Milliseconds(t2, t3));
                both.push_back(Milliseconds(t1, t3));
                refresh.push_back(Milliseconds(t3, t4));
            }
            const int n = m_Sizes.iterations;
            Record(name + "step", count, n, Median(step));
            Record(name + "scatter", count, n, Median(scatter));
            Record(name + "interpolation", count, n, Median(interpolation));
            Record(name + "scatter_interp", count, n, Median(both));
            Record(name + "refresh", count, n, Median(refresh));
            Record(name + "setters", count, n, Median(set));
            Record(name + "setters_refresh", count, n, Median(setRefresh));

            // the frame as an application has it: one Simulate with a step, one Refresh
            std::vector<double> whole;
            for (int it = 0; it < m_Sizes.iterations; ++it)
            {
                const auto start = Clock::now();
                scene.Frame(FallDomain::c_TimeStep);
                whole.push_back(Milliseconds(start, Clock::now()));
            }
            Record(name + "simulate_refresh", count, n, Median(whole));

            if (count == m_Sizes.nodes)
            {
                std::vector<double> still;
                for (int it = 0; it < m_Sizes.iterations; ++it)
                {
                    const auto start = Clock::now();
                    scene.SetTime(0.0);
                    scene.Run(Phase::BeginFrame, c_Interpolation);
                    scene.graph->Refresh(scene.frame++);
                    still.push_back(Milliseconds(start, Clock::now()));
                }
                Record("still", count, n, Median(still));
                Teleports(scene);
            }
        }

        // Requests of some of the driven nodes: the setters; what resolving them and the domain's gather cost, and what
        // the two cost without any; the frame of Simulate and Refresh with them.
        void Teleports(Scene& scene)
        {
            const auto chosen = scene.Every(m_Sizes.nodes / m_Sizes.teleports);
            auto request = [&](int it)
            {
                for (size_t i = 0; i < chosen.size(); ++i)
                {
                    const dm::double3 translation(double(it), double(i % 53), double(i / 317));
                    const dm::dquat rotation(0.6, 0.0, 0.8, 0.0);
                    if (i % 2)
                        chosen[i]->SetTransform(&translation, &rotation, nullptr);
                    else
                        chosen[i]->SetTranslation(translation);
                }
            };
            std::vector<double> set, gather, noGather, whole;
            const uint64_t before = scene.domain->teleports;
            for (int it = 0; it < m_Sizes.iterations; ++it)
            {
                for (bool asked : { false, true })
                {
                    const auto t0 = Clock::now();
                    if (asked)
                        request(it);
                    const auto t1 = Clock::now();
                    scene.SetTime(FallDomain::c_TimeStep);
                    scene.Run(Phase::BeginFrame, Phase::PreSimulation);
                    const auto t2 = Clock::now();
                    scene.Run(Phase::FixedSimulation, c_Interpolation);
                    scene.graph->Refresh(scene.frame++);
                    if (asked)
                        set.push_back(Milliseconds(t0, t1));
                    (asked ? gather : noGather).push_back(Milliseconds(t1, t2));
                }
                request(it);
                const auto start = Clock::now();
                scene.Frame(FallDomain::c_TimeStep);
                whole.push_back(Milliseconds(start, Clock::now()));
            }
            if (scene.domain->teleports - before != uint64_t(m_Sizes.iterations) * 2 * chosen.size())
                std::abort();
            const int n = m_Sizes.iterations;
            Record("teleports_set", chosen.size(), n, Median(set));
            Record("teleports_gather", chosen.size(), n, Median(gather));
            Record("teleports_gather_none", 0, n, Median(noGather));
            Record("teleports_frame", chosen.size(), n, Median(whole));
        }

        void Events()
        {
            Scene scene(m_Sizes.nodes);
            const auto chosen = scene.Every(m_Sizes.nodes / m_Sizes.events);
            std::vector<double> bind, unbind, detach;
            for (int it = 0; it < m_Sizes.eventIterations; ++it)
            {
                const auto t0 = Clock::now();
                for (SceneGraphNode* node : chosen)
                    scene.graph->SetSimulationDriven(node, true);
                const auto t1 = Clock::now();
                scene.SetTime(FallDomain::c_TimeStep * 1.5);
                scene.Run(Phase::BeginFrame, c_Interpolation);
                scene.graph->Refresh(scene.frame++);
                const auto t2 = Clock::now();
                for (SceneGraphNode* node : chosen)
                    scene.graph->SetSimulationDriven(node, false);
                const auto t3 = Clock::now();
                scene.graph->Refresh(scene.frame++);
                bind.push_back(Milliseconds(t0, t1));
                unbind.push_back(Milliseconds(t2, t3));

                nvrhi::AutoPtr<SceneGraphNode> group =
                    scene.graph->Attach(scene.graph->GetRootNode(), MAKE_RC_OBJ_PTR(SceneGraphNode));
                const size_t first = scene.nodes.size();
                scene.AddNodes(group, m_Sizes.events);
                scene.graph->Refresh(scene.frame++);
                for (size_t i = first; i < scene.nodes.size(); ++i)
                    scene.graph->SetSimulationDriven(scene.nodes[i], true);
                scene.graph->Refresh(scene.frame++);
                const auto t4 = Clock::now();
                scene.graph->Detach(group);
                detach.push_back(Milliseconds(t4, Clock::now()));
                scene.nodes.resize(first);
                scene.graph->Refresh(scene.frame++);
            }
            Record("bind", m_Sizes.events, m_Sizes.eventIterations, Median(bind));
            Record("unbind", m_Sizes.events, m_Sizes.eventIterations, Median(unbind));
            Record("detach", m_Sizes.events, m_Sizes.eventIterations, Median(detach));
        }
    };
}

int main(int argc, char** argv)
{
    bool smoke = false;
    int rounds = 5;
    int workers = 0;
    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--smoke") == 0)
            smoke = true;
        else if (std::strcmp(argv[i], "--rounds") == 0 && i + 1 < argc)
            rounds = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
            workers = std::max(1, std::atoi(argv[++i]));
        else
        {
            std::fprintf(stderr, "usage: bench_simulation [--smoke] [--rounds N] [--workers N]\n");
            return 2;
        }
    }

    Sizes sizes;
    if (smoke)
    {
        sizes.nodes /= 100;
        sizes.few /= 100;
        sizes.events /= 100;
        sizes.teleports /= 100;
        sizes.iterations = 2;
        sizes.eventIterations = 1;
        rounds = 1;
    }
#ifdef NDEBUG
    const char* build = "Release";
#else
    const char* build = "Debug";
#endif
    if (workers > 0)
    {
        JobSystemDesc desc;
        desc.debugName = "bench_simulation";
        desc.workerThreadCount = uint32_t(workers);
        if (NVRHI_FAILED(CreateJobSystem(desc, &g_Jobs)))
        {
            std::fprintf(stderr, "cannot create a job system with %d workers\n", workers);
            return 1;
        }
    }

    Bench bench(sizes);
    for (int round = 0; round < rounds; round++)
    {
        bench.Round();
        std::fprintf(stderr, "round %d of %d done\n", round + 1, rounds);
    }
    bench.Print(build, rounds, workers);
    return 0;
}
