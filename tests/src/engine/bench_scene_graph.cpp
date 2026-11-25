// bench_scene_graph: baseline timings of today's SceneGraph (aggregate docs/plans/2026-11-21-donut-ecs.md, step E0.4;
// the cases of docs/ecs-design.md 10.1 that exist before the ECS).
//
//     bench_scene_graph [--smoke] [--rounds N] [--workers N] [--set-leaf]
//
// Cases (the sizes are divided by 100 with --smoke, which ctest runs):
//     refresh_flat_moved        Refresh, 100k dynamic instances under the root, 10 % of them moved
//     set_transform_flat_moved  the SetTransform calls that move them (10k per iteration)
//     refresh_static            Refresh, 1M static instances, nothing changed (a sample is the mean of 1000 calls)
//     refresh_static_one_moved  Refresh, the same 1M instances with one of them moved
//     refresh_deep              Refresh, 64 chains of 256 nodes, each with a mesh, the top of every chain moved
//     attach_orphan             Attach of an orphaned subgraph of 100k instances (100 groups of 1000)
//     refresh_after_attach      the first Refresh after that Attach
//     detach_subgraph           Detach of a subgraph of 10k instances from a graph of 100k
//     set_leaf_same_class       only with --set-leaf (E6.3): SetLeaf of a new mesh instance on 2k instances
//     set_leaf_other_class      ... of a point light on those nodes, and of a mesh instance again
// Protocol: every round runs every case once, in this order, so the rounds alternate the cases; a run repeats the
// timed operation and keeps its median; the table shows the median of the run medians and their range.
// The numbers are meaningful in Release builds only; Debug runs only prove that the cases work.
// --workers N (step E3): the graphs refresh on a job system with N worker threads; the thread that calls Refresh also
// takes ranges while it waits. Without it, graphs use the job system that the process shares (hardware threads - 1).

#include <donut/engine/SceneGraph.h>
#include <donut/engine/JobSystem.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

    // xorshift64*, as in the trace test.
    class Rng
    {
    public:
        explicit Rng(uint64_t seed) : m_State(seed ^ 0x9E3779B97F4A7C15ull) { }
        uint64_t Next()
        {
            m_State ^= m_State >> 12;
            m_State ^= m_State << 25;
            m_State ^= m_State >> 27;
            return m_State * 0x2545F4914F6CDD1Dull;
        }
        double Range(double lo, double hi) { return lo + (hi - lo) * (double(Next() >> 11) * (1.0 / 9007199254740992.0)); }
    private:
        uint64_t m_State;
    };

    struct Pose
    {
        dm::double3 translation;
        dm::dquat rotation;
        dm::double3 scaling;
    };

    Pose RandomPose(Rng& rng)
    {
        Pose pose;
        const double tx = rng.Range(-100.0, 100.0);
        const double ty = rng.Range(-100.0, 100.0);
        const double tz = rng.Range(-100.0, 100.0);
        const double qx = rng.Range(-1.0, 1.0);
        const double qy = rng.Range(-1.0, 1.0);
        const double qz = rng.Range(-1.0, 1.0);
        const double qw = rng.Range(0.1, 1.0);
        const double s = rng.Range(0.5, 1.5);
        const double length = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        pose.translation = dm::double3(tx, ty, tz);
        pose.rotation = dm::dquat(qw / length, qx / length, qy / length, qz / length);
        pose.scaling = dm::double3(s);
        return pose;
    }

    void Apply(SceneGraphNode* node, const Pose& pose)
    {
        node->SetTransform(&pose.translation, &pose.rotation, &pose.scaling);
    }

    // Meshes shared by all instances, as in an imported scene: four meshes with one opaque geometry each.
    struct Assets
    {
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;

        Assets()
        {
            auto material = factory->CreateMaterial();
            for (int i = 0; i < 4; i++)
            {
                auto geometry = factory->CreateMeshGeometry();
                geometry->material = material;
                geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f - float(i)), dm::float3(1.f + float(i)));
                auto mesh = factory->CreateMesh();
                mesh->geometries.push_back(geometry);
                mesh->objectSpaceBounds = geometry->objectSpaceBounds;
                meshes.push_back(mesh);
            }
        }

        nvrhi::AutoPtr<SceneGraphNode> NewInstanceNode(Rng& rng, size_t index) const
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            Apply(node, RandomPose(rng));
            node->SetLeaf(factory->CreateMeshInstance(meshes[index % meshes.size()]));
            return node;
        }
    };

    // The job system of --workers, or null for the shared one.
    nvrhi::AutoPtr<IJobSystem> g_Jobs;

    nvrhi::AutoPtr<SceneGraph> NewGraph()
    {
        auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
        graph->SetJobSystem(g_Jobs);
        graph->SetRootNode(MAKE_RC_OBJ_PTR(SceneGraphNode));
        return graph;
    }

    // `count` instances under the root of a graph; returns the nodes.
    std::vector<SceneGraphNode*> AddFlat(SceneGraph* graph, const Assets& assets, Rng& rng, size_t count)
    {
        std::vector<SceneGraphNode*> nodes;
        nodes.reserve(count);
        for (size_t i = 0; i < count; i++)
            nodes.push_back(graph->Attach(graph->GetRootNode(), assets.NewInstanceNode(rng, i)));
        return nodes;
    }

    // An orphaned subgraph: `groups` group nodes with `perGroup` instances each.
    nvrhi::AutoPtr<SceneGraphNode> NewOrphan(SceneGraph* graph, const Assets& assets, Rng& rng, size_t groups, size_t perGroup)
    {
        auto top = MAKE_RC_OBJ_PTR(SceneGraphNode);
        for (size_t g = 0; g < groups; g++)
        {
            auto group = MAKE_RC_OBJ_PTR(SceneGraphNode);
            Apply(group, RandomPose(rng));
            for (size_t i = 0; i < perGroup; i++)
                graph->Attach(group, assets.NewInstanceNode(rng, i));
            graph->Attach(top, group);
        }
        return top;
    }

    // Two refreshes: the first visits the new nodes, the second clears the previous-transform flags.
    void Settle(SceneGraph* graph, uint32_t& frame)
    {
        graph->Refresh(frame++);
        graph->Refresh(frame++);
    }

    struct Sizes
    {
        size_t flat = 100000;
        size_t flatMoved = 10000;
        size_t staticCount = 1000000;
        size_t chains = 64;
        size_t chainDepth = 256;
        size_t orphanGroups = 100;
        size_t orphanPerGroup = 1000;
        size_t detachTotal = 100000;
        size_t detachGroups = 10;
        size_t detachPerGroup = 1000;
        size_t setLeafNodes = 2000;
        int iterations = 15;
        int staticIterations = 50;
        int structuralIterations = 3;
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
            RefreshFlatMoved();
            RefreshStatic();
            RefreshDeep();
            AttachOrphan();
            DetachSubgraph();
            if (setLeaf)
                SetLeaf();
        }

        bool setLeaf = false;   // --set-leaf: without it the cases and their order are those of E0.4

        void Print(const char* build, int rounds, int workers) const
        {
            if (workers > 0)
                std::printf("bench_scene_graph (%s build, %d rounds, %d workers; median of the run medians, ms)\n",
                            build, rounds, workers);
            else
                std::printf("bench_scene_graph (%s build, %d rounds; median of the run medians, ms)\n", build, rounds);
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
        Assets m_Assets;
        std::vector<Result> m_Results;

        // the 1M static graph is built once and kept: nothing in it changes between rounds
        nvrhi::AutoPtr<SceneGraph> m_Static;
        std::vector<SceneGraphNode*> m_StaticNodes;
        uint32_t m_StaticFrame = 0;

        void Record(const char* name, size_t nodes, int iterations, double runMedian)
        {
            for (Result& r : m_Results)
            {
                if (r.name == name)
                {
                    r.runMedians.push_back(runMedian);
                    return;
                }
            }
            Result r;
            r.name = name;
            r.nodes = nodes;
            r.iterations = iterations;
            r.runMedians.push_back(runMedian);
            m_Results.push_back(r);
        }

        void RefreshFlatMoved()
        {
            Rng rng(1);
            auto graph = NewGraph();
            const auto nodes = AddFlat(graph, m_Assets, rng, m_Sizes.flat);
            uint32_t frame = 0;
            Settle(graph, frame);

            const size_t stride = nodes.size() / m_Sizes.flatMoved;
            std::vector<Pose> poses;
            for (size_t i = 0; i < m_Sizes.flatMoved; i++)
                poses.push_back(RandomPose(rng));

            std::vector<double> setSamples, refreshSamples;
            for (int it = 0; it < m_Sizes.iterations; it++)
            {
                const auto start = Clock::now();
                for (size_t i = 0; i < m_Sizes.flatMoved; i++)
                    Apply(nodes[i * stride + size_t(it) % stride], poses[(i + size_t(it)) % poses.size()]);
                const auto moved = Clock::now();
                graph->Refresh(frame++);
                const auto end = Clock::now();
                setSamples.push_back(Milliseconds(start, moved));
                refreshSamples.push_back(Milliseconds(moved, end));
            }
            Record("refresh_flat_moved", m_Sizes.flat + 1, m_Sizes.iterations, Median(refreshSamples));
            Record("set_transform_flat_moved", m_Sizes.flat + 1, m_Sizes.iterations, Median(setSamples));
        }

        void RefreshStatic()
        {
            if (!m_Static)
            {
                Rng rng(2);
                m_Static = NewGraph();
                m_StaticNodes = AddFlat(m_Static, m_Assets, rng, m_Sizes.staticCount);
                Settle(m_Static, m_StaticFrame);
            }

            // a Refresh with nothing changed is shorter than the clock's tick: each sample is the mean of a batch
            constexpr int batch = 1000;
            std::vector<double> samples;
            for (int it = 0; it < m_Sizes.staticIterations; it++)
            {
                const auto start = Clock::now();
                for (int b = 0; b < batch; b++)
                    m_Static->Refresh(m_StaticFrame++);
                samples.push_back(Milliseconds(start, Clock::now()) / batch);
            }
            Record("refresh_static", m_Sizes.staticCount + 1, m_Sizes.staticIterations, Median(samples));

            // one instance moves: today the whole level below the root is visited
            Rng rng(3);
            const Pose pose = RandomPose(rng);
            samples.clear();
            for (int it = 0; it < m_Sizes.iterations; it++)
            {
                Apply(m_StaticNodes[size_t(it) * 7919 % m_StaticNodes.size()], pose);
                const auto start = Clock::now();
                m_Static->Refresh(m_StaticFrame++);
                samples.push_back(Milliseconds(start, Clock::now()));
                m_Static->Refresh(m_StaticFrame++);   // settle the previous-transform flags, untimed
            }
            Record("refresh_static_one_moved", m_Sizes.staticCount + 1, m_Sizes.iterations, Median(samples));
        }

        void RefreshDeep()
        {
            Rng rng(4);
            auto graph = NewGraph();
            std::vector<SceneGraphNode*> tops;
            for (size_t c = 0; c < m_Sizes.chains; c++)
            {
                SceneGraphNode* parent = graph->GetRootNode();
                for (size_t d = 0; d < m_Sizes.chainDepth; d++)
                {
                    auto node = m_Assets.NewInstanceNode(rng, d);
                    node->SetScaling(dm::double3(1.0));   // keep the chain's scale bounded
                    parent = graph->Attach(parent, node);
                    if (d == 0)
                        tops.push_back(parent);
                }
            }
            uint32_t frame = 0;
            Settle(graph, frame);

            std::vector<Pose> poses;
            for (int i = 0; i < 16; i++)
                poses.push_back(RandomPose(rng));

            std::vector<double> samples;
            for (int it = 0; it < m_Sizes.iterations; it++)
            {
                for (size_t c = 0; c < tops.size(); c++)
                    Apply(tops[c], poses[(c + size_t(it)) % poses.size()]);
                const auto start = Clock::now();
                graph->Refresh(frame++);
                samples.push_back(Milliseconds(start, Clock::now()));
            }
            Record("refresh_deep", m_Sizes.chains * m_Sizes.chainDepth + 1, m_Sizes.iterations, Median(samples));
        }

        void AttachOrphan()
        {
            std::vector<double> attachSamples, refreshSamples;
            for (int it = 0; it < m_Sizes.structuralIterations; it++)
            {
                Rng rng(5 + uint64_t(it));
                auto graph = NewGraph();
                uint32_t frame = 0;
                Settle(graph, frame);
                auto orphan = NewOrphan(graph, m_Assets, rng, m_Sizes.orphanGroups, m_Sizes.orphanPerGroup);

                const auto start = Clock::now();
                graph->Attach(graph->GetRootNode(), orphan);
                const auto attached = Clock::now();
                graph->Refresh(frame++);
                const auto end = Clock::now();
                attachSamples.push_back(Milliseconds(start, attached));
                refreshSamples.push_back(Milliseconds(attached, end));
                // the graph is destroyed without Detach, whose leaf unregistration is quadratic
            }
            const size_t nodes = m_Sizes.orphanGroups * (m_Sizes.orphanPerGroup + 1) + 2;
            Record("attach_orphan", nodes, m_Sizes.structuralIterations, Median(attachSamples));
            Record("refresh_after_attach", nodes, m_Sizes.structuralIterations, Median(refreshSamples));
        }

        void DetachSubgraph()
        {
            Rng rng(6);
            auto graph = NewGraph();
            const size_t subgraphInstances = m_Sizes.detachGroups * m_Sizes.detachPerGroup;
            AddFlat(graph, m_Assets, rng, m_Sizes.detachTotal - subgraphInstances);
            // the subgraph is attached last, so its leaves are at the end of the leaf vectors
            auto subgraph = NewOrphan(graph, m_Assets, rng, m_Sizes.detachGroups, m_Sizes.detachPerGroup);
            graph->Attach(graph->GetRootNode(), subgraph);
            uint32_t frame = 0;
            Settle(graph, frame);

            std::vector<double> samples;
            for (int it = 0; it < m_Sizes.structuralIterations; it++)
            {
                const auto start = Clock::now();
                auto detached = graph->Detach(subgraph);
                samples.push_back(Milliseconds(start, Clock::now()));
                graph->Attach(graph->GetRootNode(), detached);   // untimed: back to the same place for the next iteration
                Settle(graph, frame);
            }
            Record("detach_subgraph", m_Sizes.detachTotal + m_Sizes.detachGroups + 2, m_Sizes.structuralIterations, Median(samples));
        }

        // The leaf changes of attached nodes, in a small graph (the leaf vectors' erase is linear); the new leaves are
        // made untimed.
        void SetLeaf()
        {
            Rng rng(7);
            auto graph = NewGraph();
            const auto nodes = AddFlat(graph, m_Assets, rng, m_Sizes.setLeafNodes);
            uint32_t frame = 0;
            Settle(graph, frame);

            auto replace = [&](bool light)
            {
                std::vector<nvrhi::AutoPtr<SceneGraphLeaf>> leaves(nodes.size());
                for (size_t i = 0; i < nodes.size(); i++)
                {
                    if (light)
                        leaves[i] = MAKE_RC_OBJ_PTR(PointLight);
                    else
                        leaves[i] = m_Assets.factory->CreateMeshInstance(m_Assets.meshes[i % 4]);
                }
                const auto start = Clock::now();
                for (size_t i = 0; i < nodes.size(); i++)
                    nodes[i]->SetLeaf(leaves[i]);
                return Milliseconds(start, Clock::now());
            };
            std::vector<double> sameSamples, otherSamples;
            for (int it = 0; it < m_Sizes.iterations; it++)
            {
                sameSamples.push_back(replace(false));
                const double toLight = replace(true);
                otherSamples.push_back(toLight + replace(false));
                Settle(graph, frame);
            }
            Record("set_leaf_same_class", nodes.size() + 1, m_Sizes.iterations, Median(sameSamples));
            Record("set_leaf_other_class", nodes.size() + 1, m_Sizes.iterations, Median(otherSamples));
        }
    };
}

int main(int argc, char** argv)
{
    bool smoke = false;
    int rounds = 5;
    int workers = 0;
    bool setLeaf = false;
    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--smoke") == 0)
            smoke = true;
        else if (std::strcmp(argv[i], "--rounds") == 0 && i + 1 < argc)
            rounds = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
            workers = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--set-leaf") == 0)
            setLeaf = true;
        else
        {
            std::fprintf(stderr, "usage: bench_scene_graph [--smoke] [--rounds N] [--workers N] [--set-leaf]\n");
            return 2;
        }
    }

    Sizes sizes;
    if (smoke)
    {
        sizes.flat /= 100;
        sizes.flatMoved /= 100;
        sizes.staticCount /= 100;
        sizes.chains = 4;
        sizes.chainDepth = 32;
        sizes.orphanGroups /= 10;
        sizes.orphanPerGroup /= 10;
        sizes.detachTotal /= 100;
        sizes.detachGroups = 1;
        sizes.detachPerGroup /= 10;
        sizes.setLeafNodes /= 10;
        sizes.iterations = 2;
        sizes.staticIterations = 2;
        sizes.structuralIterations = 1;
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
        desc.debugName = "bench_scene_graph";
        desc.workerThreadCount = uint32_t(workers);
        if (NVRHI_FAILED(CreateJobSystem(desc, &g_Jobs)))
        {
            std::fprintf(stderr, "cannot create a job system with %d workers\n", workers);
            return 1;
        }
    }

    Bench bench(sizes);
    bench.setLeaf = setLeaf;
    for (int round = 0; round < rounds; round++)
    {
        bench.Round();
        std::fprintf(stderr, "round %d of %d done\n", round + 1, rounds);
    }
    bench.Print(build, rounds, workers);
    return 0;
}
