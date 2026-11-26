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

#pragma once

// Two scene graphs of the same mesh instances, for the runtime-only rows of plan step E7, item E7.2 (test_draw_packets,
// test_draw_strategy_adapters). Graph 0 makes every instance a mesh instance on a node under the root; graph 1 makes
// the instances that the runtime share selects runtime-only rows (SceneComponents.h, RuntimeMeshInstance) and the
// others nodes, as graph 0 does. Some node instances have a child node with a mesh of its own in both graphs, so that
// a subgraph box decides their transparent distance. Both graphs take the meshes in one order first
// (SceneGraph::AcquireRuntimeMesh), and the instances are made, moved, removed and made again in one order, so the
// mesh, material and slot tables are alike: a snapshot of graph 1 equals one of graph 0 once the instances are matched
// by their ids (Id) and the slots remapped.

#include <donut/engine/EcsWorld.h>
#include <donut/engine/SceneComponents.h>
#include <donut/engine/SceneGraph.h>
#include <donut/render/RenderExtract.h>
#include <donut/tests/utils.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace donut::tests
{
    class RuntimeTwins
    {
    public:
        static constexpr uint32_t c_NoChild = ~0u;
        static constexpr uint32_t c_NoId = ~0u;

        struct Instance
        {
            uint32_t mesh = 0;                       // index into the meshes
            uint32_t childMesh = c_NoChild;          // a node instance's child, in both graphs
            engine::LocalTransform local{};
            dm::double3 childTranslation = 0.0;
            bool runtime = false;                    // a runtime-only row in graph 1
            engine::SceneGraphNode* nodes[2] = {};   // graph 0's node; graph 1's, for a node instance
            engine::SceneGraphNode* children[2] = {};
            engine::Entity row = engine::Entity::Null;   // graph 1's row, for a runtime instance
        };

        nvrhi::AutoPtr<engine::SceneGraph> graphs[2] = { MAKE_RC_OBJ_PTR(engine::SceneGraph),
                                                         MAKE_RC_OBJ_PTR(engine::SceneGraph) };
        std::vector<engine::MeshInfo*> meshes;
        std::vector<engine::MeshId> meshIds[2];
        std::vector<Instance> instances;
        std::vector<engine::ComponentId> components;   // graph 1's runtime-row components

        explicit RuntimeTwins(uint64_t seed) : m_State(seed ^ 0x2545F4914F6CDD1Dull) { }
        RuntimeTwins(const RuntimeTwins&) = delete;
        RuntimeTwins& operator=(const RuntimeTwins&) = delete;
        ~RuntimeTwins()
        {
            for (uint32_t g = 0; g < 2; ++g)
                for (engine::MeshId id : meshIds[g])
                    graphs[g]->ReleaseRuntimeMesh(id);
        }

        // `count` instances spread across `spread` metres; `runtimeShare` of them runtime rows in graph 1; one in
        // `childEvery` of the node instances with a child.
        void Build(const std::vector<engine::MeshInfo*>& meshList, uint32_t count, double runtimeShare, double spread,
                   uint32_t childEvery)
        {
            meshes = meshList;
            m_Spread = spread;
            for (uint32_t g = 0; g < 2; ++g)
            {
                auto root = MAKE_RC_OBJ_PTR(engine::SceneGraphNode);
                root->SetName("root");
                graphs[g]->SetRootNode(root);
                for (engine::MeshInfo* mesh : meshes)
                {
                    meshIds[g].push_back(graphs[g]->AcquireRuntimeMesh(mesh));
                    CHECK(meshIds[g].back() != engine::MeshId::Invalid);
                }
            }
            CHECK(meshIds[0] == meshIds[1]);
            components.resize(graphs[1]->GetRuntimeMeshInstanceComponents(nullptr, 0));
            graphs[1]->GetRuntimeMeshInstanceComponents(components.data(), uint32_t(components.size()));
            instances.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                Instance& instance = instances[i];
                instance.mesh = Below(uint32_t(meshes.size()));
                instance.local = RandomLocal();
                instance.runtime = Unit() < runtimeShare;
                if (!instance.runtime && childEvery != 0 && Below(childEvery) == 0)
                {
                    instance.childMesh = Below(uint32_t(meshes.size()));
                    instance.childTranslation = dm::double3(Range(-4, 4), Range(-4, 4), Range(-4, 4));
                }
                Add(i);
            }
        }

        // Moves about a tenth of the instances (their translation): nodes through the setter, rows through the world.
        void Move()
        {
            graphs[1]->GetWorld()->AdvanceVersion();   // as a phase would: a new version for the writes
            const uint32_t moves = uint32_t(instances.size()) / 10 + 1;
            for (uint32_t k = 0; k < moves; ++k)
            {
                Instance& instance = instances[Below(uint32_t(instances.size()))];
                instance.local.translation = RandomLocal().translation;
                instance.nodes[0]->SetTranslation(instance.local.translation);
                if (instance.runtime)
                    engine::GetComponentForWrite<engine::LocalTransform>(graphs[1]->GetWorld(), instance.row)
                        ->translation = instance.local.translation;
                else
                    instance.nodes[1]->SetTranslation(instance.local.translation);
            }
        }

        // Removes `count` instances and makes them again in a new place, all removals first (so that slots are freed
        // and reused).
        void Respawn(uint32_t count)
        {
            std::vector<uint32_t> picked;
            for (uint32_t k = 0; k < count; ++k)
            {
                const uint32_t i = Below(uint32_t(instances.size()));
                if (std::find(picked.begin(), picked.end(), i) == picked.end())
                    picked.push_back(i);
            }
            for (uint32_t i : picked)
                Remove(i);
            for (uint32_t i : picked)
            {
                instances[i].local = RandomLocal();
                Add(i);
            }
        }

        void Refresh(uint32_t frame)
        {
            graphs[0]->Refresh(frame);
            graphs[1]->Refresh(frame);
        }

        // The instance's id (2 i for instance i, 2 i + 1 for its child) of a visible instance of a snapshot of graph g.
        uint32_t Id(uint32_t g, const render::VisibleInstances& in, uint32_t i) const
        {
            if (in.leaves[i])
            {
                const auto it = m_LeafIds[g].find(in.leaves[i]);
                return it != m_LeafIds[g].end() ? it->second : c_NoId;
            }
            const auto it = m_RowIds.find(in.entities[i]);
            return g == 1 && it != m_RowIds.end() ? it->second : c_NoId;
        }

        uint32_t RuntimeCount() const
        {
            uint32_t n = 0;
            for (const Instance& instance : instances)
                n += instance.runtime ? 1 : 0;
            return n;
        }

    private:
        uint64_t m_State;
        double m_Spread = 40.0;
        std::map<const engine::MeshInstance*, uint32_t> m_LeafIds[2];
        std::map<engine::Entity, uint32_t> m_RowIds;

        // xorshift64*, as the tests' scene scripts
        uint64_t Next()
        {
            m_State ^= m_State >> 12;
            m_State ^= m_State << 25;
            m_State ^= m_State >> 27;
            return m_State * 0x2545F4914F6CDD1Dull;
        }
        uint32_t Below(uint32_t bound) { return uint32_t(Next() % bound); }
        double Unit() { return double(Next() >> 11) * (1.0 / 9007199254740992.0); }
        double Range(double lo, double hi) { return lo + (hi - lo) * Unit(); }

        engine::LocalTransform RandomLocal()
        {
            engine::LocalTransform local{};
            local.translation = dm::double3(Range(-m_Spread, m_Spread), Range(-m_Spread * 0.25, m_Spread * 0.25),
                                            Range(-m_Spread * 0.5, m_Spread * 1.5));
            local.rotation = dm::dquat::identity();
            local.scaling = dm::double3(1.0);
            if (Below(3) == 0)
            {
                const dm::double3 axis = dm::normalize(dm::double3(Range(-1, 1), Range(0.1, 1), Range(-1, 1)));
                local.rotation = dm::rotationQuat(axis, Range(0.0, 6.28));
            }
            if (Below(5) == 0)
                local.scaling = dm::double3(Range(0.5, 2.0));
            return local;
        }

        engine::SceneGraphNode* AttachNode(uint32_t g, engine::SceneGraphNode* parent, uint32_t mesh,
                                           const engine::LocalTransform& local, uint32_t id)
        {
            auto node = MAKE_RC_OBJ_PTR(engine::SceneGraphNode);
            node->SetTransform(&local.translation, &local.rotation, &local.scaling);
            auto leaf = MAKE_RC_OBJ_PTR(engine::MeshInstance, meshes[mesh]);
            node->SetLeaf(leaf);
            engine::SceneGraphNode* attached = graphs[g]->Attach(parent, node);
            m_LeafIds[g][leaf.Get()] = id;
            return attached;
        }

        void Add(uint32_t i)
        {
            Instance& instance = instances[i];
            for (uint32_t g = 0; g < 2; ++g)
            {
                if (g == 1 && instance.runtime)
                {
                    const engine::MeshRef ref{ meshIds[1][instance.mesh] };
                    std::vector<const void*> values(components.size(), nullptr);
                    for (size_t c = 0; c < components.size(); ++c)
                    {
                        if (components[c] == engine::FindComponent<engine::LocalTransform>(graphs[1]->GetWorld()))
                            values[c] = &instance.local;
                        else if (components[c] == engine::FindComponent<engine::MeshRef>(graphs[1]->GetWorld()))
                            values[c] = &ref;
                    }
                    engine::EntityCreateDesc desc{};
                    desc.components = components.data();
                    desc.values = values.data();
                    desc.componentCount = uint32_t(components.size());
                    desc.entityCount = 1;
                    CHECK(graphs[1]->GetWorld()->CreateEntities(desc, &instance.row) == nvrhi::FS_OK);
                    m_RowIds[instance.row] = 2 * i;
                    continue;
                }
                instance.nodes[g] = AttachNode(g, graphs[g]->GetRootNode(), instance.mesh, instance.local, 2 * i);
                if (instance.childMesh != c_NoChild)
                {
                    engine::LocalTransform childLocal{ instance.childTranslation, dm::dquat::identity(),
                                                       dm::double3(1.0) };
                    instance.children[g] = AttachNode(g, instance.nodes[g], instance.childMesh, childLocal, 2 * i + 1);
                }
            }
        }

        void Remove(uint32_t i)
        {
            Instance& instance = instances[i];
            for (uint32_t g = 0; g < 2; ++g)
            {
                if (g == 1 && instance.runtime)
                {
                    m_RowIds.erase(instance.row);
                    CHECK(graphs[1]->GetWorld()->DestroyEntities(&instance.row, 1) == nvrhi::FS_OK);
                    instance.row = engine::Entity::Null;
                    continue;
                }
                for (engine::SceneGraphNode* node : { instance.children[g], instance.nodes[g] })
                    if (node)
                        m_LeafIds[g].erase(static_cast<const engine::MeshInstance*>(node->GetLeaf()));
                graphs[g]->Detach(instance.nodes[g]);   // the child goes with it
                instance.nodes[g] = nullptr;
                instance.children[g] = nullptr;
            }
        }
    };
}
