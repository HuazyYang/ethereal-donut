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

// The API-sequence test of the scene graph over the world (aggregate docs/plans/2026-11-21-donut-ecs.md, step E3.4;
// docs/ecs-design.md 11.1): a recorded sequence of scene-graph calls gives the same world as building the final graph
// directly.
//
// A seeded generator applies a sequence of calls to graph A: attaching new nodes and chains deeper than the last depth
// bucket, transforms through every setter, SetLeaf, Detach and a later Attach of the detached subgraph, deep copies,
// AttachLeafNode, orphaned subgraphs built off the graph, SetRootNode (a new root, or a node promoted to the root) and
// InvalidateContent, with Refresh calls in between. Then:
//   - B is built directly from A's final shape (names, structure, local transforms, leaf kinds and meshes), as one
//     orphaned subgraph attached to a new graph;
//   - C is A's own subgraph, detached from A (into records) and attached to a new graph.
// B and C are refreshed and must have equal world hashes (IWorld::ComputeHash: entities, links, depths, transforms,
// bounds, leaf tags), and A, B and C equal scene dumps (per node in pre-order: depth, name, leaf kind, the world
// transform's bit patterns, the float transform, the global bounding box, the content flags; for B and C also the leaf
// vectors). A's leaf vectors follow its call order, so they are compared only between B and C. The sequence runs at 1
// and 8 workers and twice per seed: A's world hash must not change.

#include <donut/engine/SceneGraph.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/JobSystem.h>
#include <donut/core/query_cast.h>
#include <donut/tests/utils.h>
#include <donut/core/log.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    // xorshift64*, with explicit seeds (no std:: distributions).
    class Rng
    {
    public:
        explicit Rng(uint64_t seed) : m_State(seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull) { }
        uint64_t Next()
        {
            m_State ^= m_State >> 12;
            m_State ^= m_State << 25;
            m_State ^= m_State >> 27;
            return m_State * 0x2545F4914F6CDD1Dull;
        }
        uint32_t Below(uint32_t bound) { return bound ? uint32_t(Next() % bound) : 0; }
        bool Percent(uint32_t p) { return Below(100) < p; }
        double Range(double lo, double hi) { return lo + (hi - lo) * (double(Next() >> 11) * (1.0 / 9007199254740992.0)); }

    private:
        uint64_t m_State;
    };

    enum class LeafKind : int { None, Mesh, PointLight, SpotLight, DirectionalLight, Perspective, Orthographic, Animation, Count };

    struct Assets
    {
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;

        Assets()
        {
            for (int i = 0; i < 4; i++)
            {
                auto material = factory->CreateMaterial();
                material->domain = i == 3 ? MaterialDomain::AlphaTested : MaterialDomain::Opaque;
                auto geometry = factory->CreateMeshGeometry();
                geometry->material = material;
                geometry->objectSpaceBounds = dm::box3(dm::float3(-1.f - float(i)), dm::float3(1.f + 0.5f * float(i)));
                auto mesh = factory->CreateMesh();
                mesh->geometries.push_back(geometry);
                mesh->objectSpaceBounds = geometry->objectSpaceBounds;
                meshes.push_back(mesh);
            }
        }

        nvrhi::AutoPtr<SceneGraphLeaf> MakeLeaf(LeafKind kind, int mesh) const
        {
            switch (kind)
            {
            case LeafKind::Mesh: return factory->CreateMeshInstance(meshes[size_t(mesh)]);
            case LeafKind::PointLight: return MAKE_RC_OBJ_PTR(PointLight);
            case LeafKind::SpotLight: return MAKE_RC_OBJ_PTR(SpotLight);
            case LeafKind::DirectionalLight: return MAKE_RC_OBJ_PTR(DirectionalLight);
            case LeafKind::Perspective: return MAKE_RC_OBJ_PTR(PerspectiveCamera);
            case LeafKind::Orthographic: return MAKE_RC_OBJ_PTR(OrthographicCamera);
            case LeafKind::Animation: return MAKE_RC_OBJ_PTR(SceneGraphAnimation);
            default: return nullptr;
            }
        }

        int MeshIndex(const MeshInfo* mesh) const
        {
            for (size_t i = 0; i < meshes.size(); i++)
                if (meshes[i].Get() == mesh)
                    return int(i);
            return -1;
        }
    };

    LeafKind KindOf(SceneGraphLeaf* leaf)
    {
        if (!leaf) return LeafKind::None;
        if (query_cast<MeshInstance>(leaf)) return LeafKind::Mesh;
        if (query_cast<PointLight>(leaf)) return LeafKind::PointLight;
        if (query_cast<SpotLight>(leaf)) return LeafKind::SpotLight;
        if (query_cast<DirectionalLight>(leaf)) return LeafKind::DirectionalLight;
        if (query_cast<PerspectiveCamera>(leaf)) return LeafKind::Perspective;
        if (query_cast<OrthographicCamera>(leaf)) return LeafKind::Orthographic;
        if (query_cast<SceneGraphAnimation>(leaf)) return LeafKind::Animation;
        return LeafKind::Count;
    }

    std::vector<SceneGraphNode*> Collect(SceneGraph* graph)
    {
        std::vector<SceneGraphNode*> nodes;
        for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
            nodes.push_back(walker.Get());
        return nodes;
    }

    // ------------------------------------------------------------------------------------- the sequence

    class Sequence
    {
    public:
        Sequence(SceneGraph* graph, const Assets& assets, uint64_t seed) : m_Graph(graph), m_Assets(assets), m_Rng(seed) { }

        void Run(int steps)
        {
            m_Graph->SetRootNode(NewNode());
            for (int step = 0; step < steps; step++)
            {
                const uint32_t op = m_Rng.Below(100);
                if (op < 22) AttachNew();
                else if (op < 24) AttachChain();
                else if (op < 44) Move();
                else if (op < 52) SetLeaf();
                else if (op < 58) Detach();
                else if (op < 64) Reattach();
                else if (op < 69) Copy();
                else if (op < 74) LeafNode();
                else if (op < 80) Orphan();
                else if (op < 82) SetRoot();
                else if (op < 85) Invalidate();
                else m_Graph->Refresh(m_Frame++);
            }
            m_Graph->Refresh(m_Frame++);
        }

    private:
        SceneGraph* m_Graph;
        const Assets& m_Assets;
        Rng m_Rng;
        uint32_t m_Frame = 0;
        uint32_t m_NextName = 0;
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> m_Detached;

        nvrhi::AutoPtr<SceneGraphNode> NewNode()
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetName("n" + std::to_string(m_NextName++));
            return node;
        }

        SceneGraphNode* RandomNode()
        {
            const auto nodes = Collect(m_Graph);
            return nodes[m_Rng.Below(uint32_t(nodes.size()))];
        }

        nvrhi::AutoPtr<SceneGraphLeaf> RandomLeaf()
        {
            const LeafKind kind = LeafKind(1 + int(m_Rng.Below(uint32_t(LeafKind::Count) - 1)));
            const int mesh = int(m_Rng.Below(uint32_t(m_Assets.meshes.size())));
            return m_Assets.MakeLeaf(kind, mesh);
        }

        // Every setter form; positions in [-10, 10], unit quaternions from + - * / and sqrt only.
        void RandomMove(SceneGraphNode* node)
        {
            const double tx = m_Rng.Range(-10.0, 10.0);
            const double ty = m_Rng.Range(-10.0, 10.0);
            const double tz = m_Rng.Range(-10.0, 10.0);
            const double qx = m_Rng.Range(-1.0, 1.0);
            const double qy = m_Rng.Range(-1.0, 1.0);
            const double qz = m_Rng.Range(-1.0, 1.0);
            const double qw = m_Rng.Range(0.1, 1.0);
            const double s = m_Rng.Range(0.8, 1.25);
            const double length = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            const dm::double3 translation(tx, ty, tz);
            const dm::dquat rotation(qw / length, qx / length, qy / length, qz / length);
            const dm::double3 scaling(s);
            switch (m_Rng.Below(5))
            {
            case 0: node->SetTransform(&translation, &rotation, &scaling); break;
            case 1: node->SetTranslation(translation); break;
            case 2: node->SetRotation(rotation); break;
            case 3: node->SetScaling(scaling); break;
            default: node->SetTransform(&translation, nullptr, nullptr); break;
            }
        }

        void AttachNew()
        {
            auto node = NewNode();
            if (m_Rng.Percent(60))
                RandomMove(node);
            if (m_Rng.Percent(60))
                node->SetLeaf(RandomLeaf());
            m_Graph->Attach(RandomNode(), node);
        }

        // Deeper than the last depth bucket (c_MaxHierarchyDepth): its serial and parallel paths.
        void AttachChain()
        {
            SceneGraphNode* parent = RandomNode();
            const uint32_t length = 60 + m_Rng.Below(40);
            for (uint32_t i = 0; i < length; i++)
            {
                auto node = NewNode();
                if (m_Rng.Percent(50))
                    RandomMove(node);
                if (m_Rng.Percent(30))
                    node->SetLeaf(RandomLeaf());
                parent = m_Graph->Attach(parent, node);
            }
        }

        void Move()
        {
            const uint32_t count = 1 + m_Rng.Below(8);
            for (uint32_t i = 0; i < count; i++)
                RandomMove(RandomNode());
        }

        void SetLeaf()
        {
            RandomNode()->SetLeaf(RandomLeaf());
        }

        void Detach()
        {
            const auto nodes = Collect(m_Graph);
            if (nodes.size() < 2)
                return;
            m_Detached.push_back(m_Graph->Detach(nodes[1 + m_Rng.Below(uint32_t(nodes.size()) - 1)]));
            // a detached subgraph may be edited while it is in no graph
            if (m_Rng.Percent(50))
                RandomMove(m_Detached.back());
        }

        void Reattach()
        {
            if (m_Detached.empty())
                return;
            const uint32_t which = m_Rng.Below(uint32_t(m_Detached.size()));
            m_Graph->Attach(RandomNode(), m_Detached[which]);
            m_Detached.erase(m_Detached.begin() + ptrdiff_t(which));
        }

        void Copy()
        {
            SceneGraphNode* source = RandomNode();
            size_t size = 0;
            for (SceneGraphWalker walker(source); walker; walker.Next(true))
                ++size;
            if (size > 150)
                return;
            m_Graph->Attach(RandomNode(), source);
        }

        void LeafNode()
        {
            SceneGraphNode* holder = RandomNode();
            if (SceneGraphLeaf* leaf = holder->GetLeaf(); leaf && m_Rng.Percent(50))
                m_Graph->AttachLeafNode(RandomNode(), leaf)->SetName("n" + std::to_string(m_NextName++));
            else
                m_Graph->AttachLeafNode(RandomNode(), RandomLeaf())->SetName("n" + std::to_string(m_NextName++));
        }

        void Orphan()
        {
            auto top = NewNode();
            if (m_Rng.Percent(50))
                RandomMove(top);
            std::vector<SceneGraphNode*> built = { top };
            const uint32_t count = 1 + m_Rng.Below(6);
            for (uint32_t i = 0; i < count; i++)
            {
                auto node = NewNode();
                if (m_Rng.Percent(50))
                    RandomMove(node);
                if (m_Rng.Percent(50))
                    node->SetLeaf(RandomLeaf());
                m_Graph->Attach(built[m_Rng.Below(uint32_t(built.size()))], node);
                built.push_back(node);
            }
            m_Graph->Attach(RandomNode(), top);
        }

        void SetRoot()
        {
            const auto nodes = Collect(m_Graph);
            if (nodes.size() >= 2 && m_Rng.Percent(50))
            {
                SceneGraphNode* node = nodes[1 + m_Rng.Below(uint32_t(nodes.size()) - 1)];
                auto oldRoot = m_Graph->SetRootNode(node);
                m_Graph->Attach(node, oldRoot);
            }
            else
            {
                auto root = NewNode();
                if (m_Rng.Percent(50))
                    RandomMove(root);
                auto oldRoot = m_Graph->SetRootNode(root);
                m_Graph->Attach(root, oldRoot);
            }
        }

        void Invalidate()
        {
            RandomNode()->InvalidateContent();
        }
    };

    // ------------------------------------------------------------------------------- shapes and dumps

    struct ShapeNode
    {
        std::string name;
        uint32_t parent;
        bool hasLocal;
        dm::double3 translation;
        dm::dquat rotation;
        dm::double3 scaling;
        LeafKind kind;
        int mesh;
    };

    std::vector<ShapeNode> CaptureShape(SceneGraph* graph, const Assets& assets)
    {
        IWorld* world = graph->GetWorld();
        const ComponentId local = FindComponent<LocalTransform>(world);
        std::vector<ShapeNode> shape;
        std::unordered_map<SceneGraphNode*, uint32_t> index;
        for (SceneGraphNode* node : Collect(graph))
        {
            ShapeNode s{};
            s.name = node->GetName();
            s.parent = node->GetParent() ? index.at(node->GetParent()) : ~0u;
            s.hasLocal = world->HasComponent(graph->GetEntity(node), local);
            s.translation = node->GetTranslation();
            s.rotation = node->GetRotation();
            s.scaling = node->GetScaling();
            s.kind = KindOf(node->GetLeaf());
            auto mesh = query_cast<MeshInstance>(node->GetLeaf());
            s.mesh = mesh ? assets.MeshIndex(mesh->GetMesh()) : -1;
            index[node] = uint32_t(shape.size());
            shape.push_back(s);
        }
        return shape;
    }

    // The final shape built directly: one orphaned subgraph, linked in pre-order.
    nvrhi::AutoPtr<SceneGraphNode> BuildShape(const std::vector<ShapeNode>& shape, const Assets& assets)
    {
        auto linker = MAKE_RC_OBJ_PTR(SceneGraph);   // links orphans; owns nothing
        std::vector<nvrhi::AutoPtr<SceneGraphNode>> nodes;
        for (const ShapeNode& s : shape)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetName(s.name);
            if (s.hasLocal)
                node->SetTransform(&s.translation, &s.rotation, &s.scaling);
            if (s.kind != LeafKind::None)
                node->SetLeaf(assets.MakeLeaf(s.kind, s.mesh < 0 ? 0 : s.mesh));
            if (s.parent != ~0u)
                linker->Attach(nodes[s.parent], node);
            nodes.push_back(node);
        }
        return nodes.front();
    }

    void AppendBits(std::string& out, const void* data, size_t bytes)
    {
        char text[24];
        for (size_t i = 0; i + 8 <= bytes; i += 8)
        {
            uint64_t word = 0;
            std::memcpy(&word, static_cast<const uint8_t*>(data) + i, 8);
            std::snprintf(text, sizeof(text), " %016llx", (unsigned long long)word);
            out += text;
        }
        if (bytes % 8)
        {
            uint32_t word = 0;
            std::memcpy(&word, static_cast<const uint8_t*>(data) + bytes - 4, 4);
            std::snprintf(text, sizeof(text), " %08x", word);
            out += text;
        }
    }

    std::string Dump(SceneGraph* graph, bool leafVectors)
    {
        std::string out;
        std::unordered_map<const SceneGraphNode*, size_t> index;
        int depth = 0;
        SceneGraphWalker walker(graph->GetRootNode());
        while (walker)
        {
            SceneGraphNode* node = walker.Get();
            index[node] = index.size();
            out += std::to_string(depth) + " " + node->GetName() + " " + std::to_string(int(KindOf(node->GetLeaf())));
            const dm::daffine3 world = node->GetLocalToWorldTransform();
            const dm::affine3 worldFloat = node->GetLocalToWorldTransformFloat();
            const dm::box3 bounds = node->GetGlobalBoundingBox();
            AppendBits(out, &world, sizeof(world));
            AppendBits(out, &worldFloat, sizeof(worldFloat));
            AppendBits(out, &bounds, sizeof(bounds));
            out += " " + std::to_string(uint32_t(node->GetLeafContentFlags())) + " " +
                   std::to_string(uint32_t(node->GetSubgraphContentFlags())) + "\n";
            depth += walker.Next(true);
        }
        if (leafVectors)
        {
            auto vector = [&](const char* name, const auto& leaves)
            {
                out += name;
                for (const auto& leaf : leaves)
                    out += " " + std::to_string(index.at(leaf->GetNode()));
                out += "\n";
            };
            vector("meshes", graph->GetMeshInstances());
            vector("lights", graph->GetLights());
            vector("cameras", graph->GetCameras());
            vector("animations", graph->GetAnimations());
        }
        return out;
    }

    void CheckSameText(const std::string& a, const std::string& b, const char* what)
    {
        if (a == b)
            return;
        size_t line = 0, i = 0;
        while (i < a.size() && i < b.size() && a[i] == b[i])
        {
            if (a[i] == '\n')
                ++line;
            ++i;
        }
        std::fprintf(stderr, "%s differ at line %zu:\n  %s\n  %s\n", what, line,
                     a.substr(a.rfind('\n', i) + 1, 160).c_str(), b.substr(b.rfind('\n', i) + 1, 160).c_str());
        CHECK(a == b);
    }

    struct Outcome
    {
        uint64_t hashA = 0;
        std::string dumpA;
        size_t nodes = 0;
    };

    Outcome RunAndCompare(uint64_t seed, IJobSystem* jobs, const Assets& assets)
    {
        Outcome outcome;
        auto a = MAKE_RC_OBJ_PTR(SceneGraph);
        a->SetJobSystem(jobs);
        Sequence(a, assets, seed).Run(400);
        outcome.hashA = a->GetWorld()->ComputeHash();
        outcome.dumpA = Dump(a, false);
        const auto shape = CaptureShape(a, assets);
        outcome.nodes = shape.size();

        // B: the final shape, built directly
        auto b = MAKE_RC_OBJ_PTR(SceneGraph);
        b->SetJobSystem(jobs);
        b->SetRootNode(BuildShape(shape, assets));
        b->Refresh(0);

        // C: A's subgraph, detached into records and attached to a new graph
        nvrhi::AutoPtr<SceneGraphNode> root = a->GetRootNode();
        a->Detach(root);
        auto c = MAKE_RC_OBJ_PTR(SceneGraph);
        c->SetJobSystem(jobs);
        c->SetRootNode(root);
        c->Refresh(0);

        CHECK(b->GetWorld()->ComputeHash() == c->GetWorld()->ComputeHash());
        const std::string dumpB = Dump(b, true);
        CheckSameText(dumpB, Dump(c, true), "B and C");
        CheckSameText(outcome.dumpA, Dump(b, false), "A and B");

        WorldStats stats;
        b->GetWorld()->GetStats(stats);
        CHECK(stats.entityCount == shape.size());
        c->GetWorld()->GetStats(stats);
        CHECK(stats.entityCount == shape.size());
        // A keeps only the placeholder root, which is in no graph
        a->GetWorld()->GetStats(stats);
        CHECK(stats.entityCount == 0);
        return outcome;
    }
}

void test_sequences_equal_direct_build()
{
    Assets assets;
    nvrhi::AutoPtr<IJobSystem> jobs[2];
    const uint32_t workers[2] = { 1, 8 };
    for (int j = 0; j < 2; j++)
    {
        JobSystemDesc desc;
        desc.debugName = "api_sequence";
        desc.workerThreadCount = workers[j];
        CHECK(NVRHI_SUCCEEDED(CreateJobSystem(desc, &jobs[j])));
    }

    size_t totalNodes = 0;
    for (uint64_t seed = 1; seed <= 6; seed++)
    {
        const Outcome first = RunAndCompare(seed, jobs[0], assets);
        const Outcome again = RunAndCompare(seed, jobs[0], assets);
        const Outcome wide = RunAndCompare(seed, jobs[1], assets);
        // the same calls give the same world, whatever the workers
        CHECK(first.hashA == again.hashA);
        CHECK(first.hashA == wide.hashA);
        CheckSameText(first.dumpA, again.dumpA, "two runs");
        CheckSameText(first.dumpA, wide.dumpA, "1 and 8 workers");
        totalNodes += first.nodes;
        std::printf("seed %llu: %zu nodes, world hash %016llx\n", (unsigned long long)seed, first.nodes,
                    (unsigned long long)first.hashA);
    }
    CHECK(totalNodes > 1000);
}

int main(int, char**)
{
    // Every warning fails the test: the graphs' worlds report validation errors through log::warning in Debug builds.
    static std::atomic<uint32_t> s_Warnings{ 0 };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            s_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    try
    {
        test_sequences_equal_direct_build();
        CHECK(s_Warnings.load() == 0);
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    return 0;
}
