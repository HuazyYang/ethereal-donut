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

// Simulation-driven nodes (aggregate docs/plans/2026-11-21-donut-ecs-e6.md, item E6.2a; design 4.7, 8.4): binding and
// unbinding, the driver in propagation (shallow levels and the last bucket), detach, copies, SetRootNode, destruction and
// teardown, the local getters, refusals, the stale flag, joints, the interpolation (its formula bit for bit, which
// chunks it writes, its phase), the scatter contract over a toy engine, a random script against a model that computes
// every world transform, box and subtree value from scratch, equal hashes at 1, 2 and 8 workers, and an allocation
// counter. The phases are driven by hand (SceneGraph::Simulate is item E6.2b). The cases `driver_*` and
// `interpolation_*` each name the behaviour they fix; the private headers give the counters and column versions.

#include "SceneGraphGenerator.h"

#include <donut/core/log.h>
#include <donut/core/query_cast.h>
#include <donut/engine/EcsScheduler.h>
#include <donut/engine/EcsSimulation.h>
#include <donut/engine/EcsWorld.h>
#include <donut/engine/GpuScene.h>
#include <donut/engine/JobSystem.h>
#include <donut/engine/SceneGraph.h>
#include <donut/tests/utils.h>

#include "EcsSchedulerImpl.h"
#include "EcsWorldImpl.h"
#include "SceneGraphStore.h"

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// ------------------------------------------------------------------------------- global allocation counter

// Every operator new of this executable (the test and donut_engine, which links statically). The world's own blocks
// come from nvrhi_core's allocator, which ecs::GetHeapAllocationCount counts.
namespace
{
    std::atomic<uint64_t> g_NewCount{ 0 };

    void* CountedAllocate(size_t size) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
        return std::malloc(size ? size : 1);
    }

    void* CountedAllocateAligned(size_t size, size_t alignment) noexcept
    {
        g_NewCount.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
        return _aligned_malloc(size ? size : 1, alignment);
#else
        return std::aligned_alloc(alignment, ((size ? size : 1) + alignment - 1) & ~(alignment - 1));
#endif
    }

    void CountedFreeAligned(void* memory) noexcept
    {
#if defined(_MSC_VER)
        _aligned_free(memory);
#else
        std::free(memory);
#endif
    }
}

void* operator new(size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size)
{
    if (void* memory = CountedAllocate(size))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new[](size_t size, const std::nothrow_t&) noexcept { return CountedAllocate(size); }
void* operator new(size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t size, std::align_val_t alignment)
{
    if (void* memory = CountedAllocateAligned(size, size_t(alignment)))
        return memory;
    throw std::bad_alloc();
}
void* operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return CountedAllocateAligned(size, size_t(alignment));
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }
void operator delete(void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete[](void* memory, const std::nothrow_t&) noexcept { std::free(memory); }
void operator delete(void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, size_t, std::align_val_t) noexcept { CountedFreeAligned(memory); }
void operator delete(void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }
void operator delete[](void* memory, std::align_val_t, const std::nothrow_t&) noexcept { CountedFreeAligned(memory); }

namespace sim_graph_test
{
    using donut::engine::ComponentFlags;

    // Filters of the two domains' lifetime queries.
    struct HeavyBall { };
    DONUT_ECS_TAG(HeavyBall, "6a1f41e1-0c06-4b0e-8c51-0f5a6d1d8f34");
    struct LightBall { };
    DONUT_ECS_TAG(LightBall, "f2b3c3d9-7e53-45a5-9a3e-5d44a2a7a0c1");

    // A row with it still fits the largest chunk (64 KiB); with the columns of a driven row it does not.
    struct Bulk { uint8_t bytes[64640]; };
    DONUT_ECS_COMPONENT(Bulk, "1c0c3f38-4a5d-4d0f-8a7e-52c3b5b7b1ef", DONUT_ECS_FIELDS());
}

using namespace donut;
using namespace donut::engine;
using namespace sim_graph_test;

// log::warning calls; main fails a case that leaves any (Debug worlds report validation errors through it). A case that
// provokes warnings on purpose checks them and takes them back.
std::atomic<uint32_t> g_Warnings{ 0 };

namespace
{
    // Domain calls are noexcept, so a failed expectation in one is counted, and main checks the count after each case.
    std::atomic<int> g_Errors{ 0 };
    void Expect(bool condition, int line, const char* text) noexcept
    {
        if (condition)
            return;
        g_Errors.fetch_add(1);
        fprintf(stderr, "test_scene_graph_simulation.cpp:%d: expected %s\n", line, text);
    }
#define EXPECT(condition) Expect(bool(condition), __LINE__, #condition)

    uint32_t TakeWarnings() { return g_Warnings.exchange(0); }

    // SceneGraph::m_Store (an explicit instantiation may name a private member).
    template <typename Tag, typename Tag::Type Member>
    struct PrivateMember
    {
        friend typename Tag::Type GetMember(Tag) { return Member; }
    };
    struct StoreTag
    {
        using Type = SceneGraphStore* SceneGraph::*;
        friend Type GetMember(StoreTag);
    };
    template struct PrivateMember<StoreTag, &SceneGraph::m_Store>;
    SceneGraphStore& StoreOf(SceneGraph* graph) { return *(graph->*GetMember(StoreTag())); }

    bool SameBits(const void* a, const void* b, size_t size) { return std::memcmp(a, b, size) == 0; }
    bool SameBits(const dm::daffine3& a, const dm::daffine3& b) { return SameBits(&a, &b, sizeof(a)); }

    // Equal within `tolerance`, relative to the larger of 1 and the values.
    bool Near(double a, double b, double tolerance = 1e-9)
    {
        return std::abs(a - b) <= tolerance * std::max({ 1.0, std::abs(a), std::abs(b) });
    }
    bool Near(const dm::double3& a, const dm::double3& b, double tolerance = 1e-9)
    {
        return Near(a.x, b.x, tolerance) && Near(a.y, b.y, tolerance) && Near(a.z, b.z, tolerance);
    }
    bool Near(const dm::daffine3& a, const dm::daffine3& b, double tolerance = 1e-9)
    {
        return Near(a.m_linear.row0, b.m_linear.row0, tolerance) && Near(a.m_linear.row1, b.m_linear.row1, tolerance) &&
               Near(a.m_linear.row2, b.m_linear.row2, tolerance) && Near(a.m_translation, b.m_translation, tolerance);
    }
    // The same rotation: q and -q are one.
    bool Near(const dm::dquat& a, const dm::dquat& b, double tolerance = 1e-9)
    {
        const double sign = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z < 0.0 ? -1.0 : 1.0;
        return Near(a.w, sign * b.w, tolerance) && Near(a.x, sign * b.x, tolerance) &&
               Near(a.y, sign * b.y, tolerance) && Near(a.z, sign * b.z, tolerance);
    }

    dm::dquat Unit(double w, double x, double y, double z)
    {
        const double length = std::sqrt(w * w + x * x + y * y + z * z);
        return dm::dquat(w / length, x / length, y / length, z / length);
    }

    // What the interpolation must write, spelled out (design 8.4; + - * / sqrt): the previous pose at 0, the last one at
    // 1; between them the translation is linear, and the rotation is linear on the short side and normalised.
    dm::daffine3 DirectWorld(const SimPosePrevious& previous, const SimPose& pose, const SimScale& scale, double alpha)
    {
        double t[3], w, x, y, z;
        if (alpha == 0.0 || alpha == 1.0)
        {
            const dm::double3& translation = alpha == 0.0 ? previous.translation : pose.translation;
            const dm::dquat& rotation = alpha == 0.0 ? previous.rotation : pose.rotation;
            for (int i = 0; i < 3; ++i)
                t[i] = translation[i];
            w = rotation.w, x = rotation.x, y = rotation.y, z = rotation.z;
        }
        else
        {
            const double rest = 1.0 - alpha;
            for (int i = 0; i < 3; ++i)
                t[i] = previous.translation[i] * rest + pose.translation[i] * alpha;
            const dm::dquat& a = previous.rotation;
            const dm::dquat& b = pose.rotation;
            const double dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
            const double side = dot < 0.0 ? -alpha : alpha;
            w = a.w * rest + b.w * side;
            x = a.x * rest + b.x * side;
            y = a.y * rest + b.y * side;
            z = a.z * rest + b.z * side;
            const double length = std::sqrt(w * w + x * x + y * y + z * z);
            if (length > 0.0)
                w = w / length, x = x / length, y = y / length, z = z / length;
        }
        const dm::double3& s = scale.scaling;
        return dm::daffine3(
            (1 - 2 * (y * y + z * z)) * s.x, (2 * (x * y + z * w)) * s.x, (2 * (x * z - y * w)) * s.x,
            (2 * (x * y - z * w)) * s.y, (1 - 2 * (x * x + z * z)) * s.y, (2 * (y * z + x * w)) * s.y,
            (2 * (x * z + y * w)) * s.z, (2 * (y * z - x * w)) * s.z, (1 - 2 * (x * x + y * y)) * s.z,
            t[0], t[1], t[2]);
    }

    dm::daffine3 ComposeTransform(const dm::double3& translation, const dm::dquat& rotation, const dm::double3& scaling)
    {
        dm::daffine3 transform = dm::scaling(scaling);
        transform *= rotation.toAffine();
        transform *= dm::translation(translation);
        return transform;
    }

    nvrhi::AutoPtr<SceneGraphNode> MakeNode(const char* name = "n")
    {
        auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
        node->SetName(name);
        return node;
    }

    nvrhi::AutoPtr<MeshInfo> MakeMesh(SceneTypeFactory* factory, float size, MaterialDomain domain)
    {
        auto material = factory->CreateMaterial();
        material->domain = domain;
        auto geometry = factory->CreateMeshGeometry();
        geometry->material = material;
        geometry->objectSpaceBounds = dm::box3(dm::float3(-size), dm::float3(size, 2.f * size, size));
        auto mesh = factory->CreateMesh();
        mesh->geometries.push_back(geometry);
        mesh->objectSpaceBounds = geometry->objectSpaceBounds;
        return mesh;
    }

    struct Placement
    {
        dm::double3 translation = dm::double3(0.0);
        dm::dquat rotation = dm::dquat::identity();
        dm::double3 scaling = dm::double3(1.0);
    };
    void Place(SceneGraphNode* node, const Placement& p) { node->SetTransform(&p.translation, &p.rotation, &p.scaling); }
    // Uniform scales, so that no product of them shears.
    Placement RandomPlacement(Rng& rng)
    {
        Placement p;
        p.translation = RandomVector(rng, -5.0, 5.0);
        p.rotation = RandomRotation(rng);
        p.scaling = dm::double3(rng.Range(0.8, 1.25));
        return p;
    }

    nvrhi::AutoPtr<IJobSystem> MakeJobs(uint32_t workers)
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        JobSystemDesc desc;
        desc.debugName = "test_scene_graph_simulation";
        desc.workerThreadCount = workers;
        CHECK(NVRHI_SUCCEEDED(CreateJobSystem(desc, &jobs)));
        return jobs;
    }

    // ------------------------------------------------------------------------------------------ a toy engine

    // Balls under gravity that spin, in dense arrays addressed by the index that OnEntitiesAdded stores in SimBodyRef:
    // what an engine behind a domain looks like, with the previous pose kept in engine order (the scatter contract).
    class BallDomain final : public nvrhi::ObjectImpl<ISimulationDomain>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(BallDomain)
        NVRHI_IMPLEMENTS_INTERFACE(ISimulationDomain)
        NVRHI_END_INTERFACE_TABLE()

        struct Ball
        {
            SimPose pose;
            SimPosePrevious previous;
            dm::double3 velocity;
            dm::dquat spin;        // per step
            Entity entity;         // Null: free
        };

        std::string name;
        ComponentId pose, previous, scale, body, driven, filter;
        std::vector<ComponentAccess> outputs;
        std::vector<QueryTerm> terms;
        std::vector<Ball> balls;
        std::vector<uint32_t> freeIndices;
        std::vector<std::pair<Entity, bool>> events;    // in order; true: added
        std::function<void(const EntityBatch&, ISimulationContext*)> onAdded;
        double dt;
        uint32_t maxSteps;
        bool keepEvents = true;
        bool clearsBodyRefs = true;
        bool scattersWithoutSteps = false;     // a domain that breaks the contract
        uint32_t steps = 0;
        uint32_t scatters = 0;                 // that wrote

        BallDomain(const char* domainName, IWorld* world, double fixedTimeStep, uint32_t maxStepsPerFrame,
                   ComponentId filterTag = ComponentId::Invalid)
            : name(domainName), filter(filterTag), dt(fixedTimeStep), maxSteps(maxStepsPerFrame)
        {
            pose = FindComponent<SimPose>(world);
            previous = FindComponent<SimPosePrevious>(world);
            scale = FindComponent<SimScale>(world);
            body = FindComponent<SimBodyRef>(world);
            driven = FindComponent<SimulationDriven>(world);
            outputs = { { pose, Access::Write, {} }, { previous, Access::Write, {} }, { body, Access::Read, {} } };
            terms = { { driven, TermOp::All, Access::Filter, {} }, { body, TermOp::All, Access::Write, {} },
                      { pose, TermOp::All, Access::Read, {} } };
            if (filter != ComponentId::Invalid)
                terms.push_back({ filter, TermOp::All, Access::Filter, {} });
        }

        SimulationDomainDesc& GetDesc(SimulationDomainDesc& retVal) noexcept override
        {
            retVal = SimulationDomainDesc{};
            retVal.name = name.c_str();
            retVal.outputs = outputs.data();
            retVal.outputCount = uint32_t(outputs.size());
            retVal.lifetime.terms = terms.data();
            retVal.lifetime.termCount = uint32_t(terms.size());
            retVal.substeps = 1;
            retVal.maxStepsPerFrame = maxSteps;
            retVal.fixedTimeStep = dt;
            retVal.flags = SimulationDomainFlags::Deterministic;
            return retVal;
        }
        nvrhi::FRESULT OnEntitiesAdded(const EntityBatch& batch, ISimulationContext* context) noexcept override
        {
            SimBodyRef* bodies = static_cast<SimBodyRef*>(batch.columns[0]);
            const SimPose* poses = static_cast<const SimPose*>(batch.columns[1]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                uint32_t index = uint32_t(balls.size());
                if (!freeIndices.empty())
                {
                    index = freeIndices.back();
                    freeIndices.pop_back();
                }
                else
                    balls.emplace_back();
                Ball& ball = balls[index];
                ball.pose = poses[i];
                ball.previous = SimPosePrevious{ poses[i].translation, poses[i].rotation };
                // by the entity, so that every run and every thread count gives the same motion
                const double seed = double(GetEntityIndex(batch.entities[i]) % 17);
                ball.velocity = dm::double3(0.5 * seed - 4.0, 3.0 + 0.25 * seed, 1.0 - 0.125 * seed);
                ball.spin = Unit(8.0, 0.1 * seed - 0.8, 0.5, 0.3);
                ball.entity = batch.entities[i];
                bodies[i].domain = uint32_t(context->GetDomainId());
                bodies[i].index = index;
                if (keepEvents)
                    events.emplace_back(batch.entities[i], true);
            }
            if (onAdded)
                onAdded(batch, context);
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT OnEntitiesRemoved(const EntityBatch& batch, ISimulationContext*) noexcept override
        {
            SimBodyRef* bodies = static_cast<SimBodyRef*>(batch.columns[0]);
            for (uint32_t i = 0; i < batch.count; ++i)
            {
                const uint32_t index = bodies[i].index;
                EXPECT(index < balls.size() && balls[index].entity == batch.entities[i]);
                if (index < balls.size())
                {
                    balls[index] = Ball{};
                    freeIndices.push_back(index);
                }
                if (clearsBodyRefs)
                    bodies[i] = SimBodyRef{};
                if (keepEvents)
                    events.emplace_back(batch.entities[i], false);
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT GatherInputs(ISimulationContext*) noexcept override { return nvrhi::FS_OK; }
        nvrhi::FRESULT Step(double step, ISimulationContext*) noexcept override
        {
            ++steps;
            for (Ball& ball : balls)
            {
                if (ball.entity == Entity::Null)
                    continue;
                ball.previous = SimPosePrevious{ ball.pose.translation, ball.pose.rotation };
                ball.velocity.y -= 9.0 * step;
                ball.pose.translation += ball.velocity * step;
                if (ball.pose.translation.y < -20.0)
                    ball.velocity.y = std::abs(ball.velocity.y);
                const dm::dquat& q = ball.pose.rotation;
                const dm::dquat& r = ball.spin;
                ball.pose.rotation = Unit(q.w * r.w - q.x * r.x - q.y * r.y - q.z * r.z,
                                          q.w * r.x + q.x * r.w + q.y * r.z - q.z * r.y,
                                          q.w * r.y + q.y * r.w + q.z * r.x - q.x * r.z,
                                          q.w * r.z + q.z * r.w + q.x * r.y - q.y * r.x);
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT ScatterOutputs(ISimulationContext* context) noexcept override
        {
            if (context->GetStepsThisFrame() == 0 && !scattersWithoutSteps)
                return nvrhi::FS_OK;
            ++scatters;
            ColumnView bodies[4], poses[4], previousPoses[4];
            const uint32_t count = context->GetColumnCount(body);
            for (uint32_t first = 0; first < count; first += 4)
            {
                const uint32_t n = context->GetColumns(body, first, c_AllVersions, 4, bodies);
                const uint32_t p = context->GetColumns(pose, first, c_AllVersions, 4, poses);
                const uint32_t q = context->GetColumns(previous, first, c_AllVersions, 4, previousPoses);
                EXPECT(n == p && n == q);
                for (uint32_t v = 0; v < std::min({ n, p, q }); ++v)
                {
                    EXPECT(bodies[v].chunkIndex == poses[v].chunkIndex &&
                           bodies[v].chunkIndex == previousPoses[v].chunkIndex);
                    for (uint32_t row = 0; row < bodies[v].count; ++row)
                    {
                        const Ball& ball = balls[static_cast<const SimBodyRef*>(bodies[v].base)[row].index];
                        static_cast<SimPose*>(poses[v].base)[row] = ball.pose;
                        static_cast<SimPosePrevious*>(previousPoses[v].base)[row] = ball.previous;
                    }
                }
            }
            return nvrhi::FS_OK;
        }
        nvrhi::FRESULT CaptureState(IStateWriter*) noexcept override { return nvrhi::FE_UNSUPPORTED; }
        nvrhi::FRESULT RestoreState(IStateReader*) noexcept override { return nvrhi::FE_UNSUPPORTED; }

        const Ball* Find(Entity entity) const
        {
            for (const Ball& ball : balls)
            {
                if (ball.entity == entity && entity != Entity::Null)
                    return &ball;
            }
            return nullptr;
        }
    };

    // ------------------------------------------------------------------------------------------ the rig

    struct Rig
    {
        nvrhi::AutoPtr<IJobSystem> jobs;
        nvrhi::AutoPtr<SceneTypeFactory> factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);
        nvrhi::AutoPtr<SceneGraph> graph = MAKE_RC_OBJ_PTR(SceneGraph);
        nvrhi::AutoPtr<MeshInfo> mesh;
        IWorld* world = nullptr;
        double time = 0.0;
        uint64_t frame = 0;
        uint32_t refreshes = 0;

        explicit Rig(uint32_t workers = 2)
            : jobs(MakeJobs(workers))
        {
            graph->SetJobSystem(jobs);
            graph->SetRootNode(MakeNode("root"));
            mesh = MakeMesh(factory, 1.f, MaterialDomain::Opaque);
            world = graph->GetWorld();
        }

        SceneGraphNode* Root() { return graph->GetRootNode(); }
        SceneGraphStore& Store() { return StoreOf(graph); }
        ecs::WorldImpl& Impl() { return *ecs::GetWorldImpl(world); }
        SceneGraphNode* Add(SceneGraphNode* parent, const Placement* placement = nullptr, bool leaf = false)
        {
            auto node = MakeNode();
            if (placement)
                Place(node, *placement);
            if (leaf)
                node->SetLeaf(factory->CreateMeshInstance(mesh));
            return graph->Attach(parent, node);
        }
        SceneGraphNode* AddAt(SceneGraphNode* parent, double x, double y, double z, bool leaf = false)
        {
            Placement p;
            p.translation = dm::double3(x, y, z);
            return Add(parent, &p, leaf);
        }
        void Refresh() { graph->Refresh(refreshes++); }
        void SetTime(double delta)
        {
            time += delta;
            graph->GetScheduler()->SetFrameTime(FrameTime{ time, delta, frame++ });
        }
        void RunTo(Phase last) { CHECK(graph->GetScheduler()->RunPhases(Phase::BeginFrame, last) == nvrhi::FS_OK); }
        void Interpolate() { CHECK(graph->GetScheduler()->RunPhase(Store().GetInterpolationPhase()) == nvrhi::FS_OK); }
        // One frame of the simulation phases, driven by hand: the named phase runs after PostSimulation.
        void Simulate(double delta)
        {
            SetTime(delta);
            RunTo(Store().GetInterpolationPhase());
        }
        void Frame(double delta)
        {
            Simulate(delta);
            Refresh();
        }
        void Bind(SceneGraphNode* node) { CHECK(graph->SetSimulationDriven(node, true) == nvrhi::FS_OK); }
        void Unbind(SceneGraphNode* node) { CHECK(graph->SetSimulationDriven(node, false) == nvrhi::FS_OK); }
        Entity EntityOf(const SceneGraphNode* node) { return graph->GetEntity(node); }
        bool IsDriven(const SceneGraphNode* node)
        {
            return world->HasComponent(EntityOf(node), FindComponent<SimulationDriven>(world));
        }
        template <typename T> const T* Get(const SceneGraphNode* node) { return GetComponent<T>(world, EntityOf(node)); }
        template <typename T> T* Edit(const SceneGraphNode* node)
        {
            T* value = GetComponentForWrite<T>(world, EntityOf(node));
            CHECK(value);
            return value;
        }
        // A pose written as a domain would (both columns), outside phases: the world's version moves past what the
        // systems saw, as a phase's does.
        void SetPose(const SceneGraphNode* node, const dm::double3& translation, const dm::dquat& rotation)
        {
            world->AdvanceVersion();
            *Edit<SimPose>(node) = SimPose{ translation, rotation };
            *Edit<SimPosePrevious>(node) = SimPosePrevious{ translation, rotation };
        }
        dm::daffine3 Posed(const SceneGraphNode* node)
        {
            return DirectWorld(*Get<SimPosePrevious>(node), *Get<SimPose>(node), *Get<SimScale>(node), 1.0);
        }
        double Alpha(SimulationDomainId id)
        {
            const FixedStepEntry& entry = GetResource<FixedStepState>(world)->domains[uint32_t(id)];
            CHECK(entry.fixedTimeStep > 0.0);
            return entry.accumulator / entry.fixedTimeStep;
        }
        // The version of the chunk column that holds the entity's component.
        uint64_t ColumnVersion(const SceneGraphNode* node, ComponentId component)
        {
            const ecs::EntityRecord* record = Impl().FindAlive(EntityOf(node));
            CHECK(record);
            const ecs::Archetype& archetype = Impl().GetArchetypeStorage(record->archetype);
            const uint32_t column = archetype.FindColumn(component);
            CHECK(column != ecs::c_NoColumn);
            return archetype.Versions(record->chunk)[column];
        }
        uint64_t PropagatedRows() { return Store().GetCounters().propagatedRows; }
        uint64_t InterpolatedChunks() { return Store().GetCounters().interpolatedChunks; }
        SimulationDomainId AddDomain(BallDomain* domain)
        {
            SimulationDomainId id = SimulationDomainId::Invalid;
            CHECK(graph->GetScheduler()->AddSimulationDomain(domain, id) == nvrhi::FS_OK);
            return id;
        }
        nvrhi::AutoPtr<BallDomain> NewDomain(const char* name = "balls", double dt = 0.25, uint32_t maxSteps = 4,
                                             ComponentId filter = ComponentId::Invalid)
        {
            return MAKE_RC_OBJ_PTR(BallDomain, name, world, dt, maxSteps, filter);
        }
    };

    // The local transforms as propagation composes them, from scratch; driven nodes keep what `driven` gives for them.
    dm::daffine3 ExpectedWorld(Rig& rig, SceneGraphNode* node,
                               const std::function<dm::daffine3(SceneGraphNode*)>& driven)
    {
        if (rig.IsDriven(node))
            return driven(node);
        SceneGraphNode* parent = node->GetParent();
        const LocalTransform* local = rig.Get<LocalTransform>(node);
        const dm::daffine3 localMatrix =
            local ? ComposeTransform(local->translation, local->rotation, local->scaling) : dm::daffine3::identity();
        if (!parent)
            return localMatrix;
        const dm::daffine3 parentWorld = ExpectedWorld(rig, parent, driven);
        return local ? localMatrix * parentWorld : parentWorld;
    }

    using ChunkKey = std::pair<uint32_t, uint32_t>;
    ChunkKey ChunkOf(Rig& rig, const SceneGraphNode* node)
    {
        const ecs::EntityRecord* record = rig.Impl().FindAlive(rig.EntityOf(node));
        CHECK(record);
        return { record->archetype, record->chunk };
    }
    // The rows that propagation must compute when the interpolation wrote `written` and nothing else changed: per level
    // those below a written chunk, whose chunks are then written too (versions are per chunk).
    uint64_t RowsBelow(Rig& rig, std::set<ChunkKey> written)
    {
        uint64_t rows = 0;
        std::vector<SceneGraphNode*> level = { rig.Root() };
        while (!level.empty())
        {
            std::vector<SceneGraphNode*> next;
            std::set<ChunkKey> nextWritten;
            for (SceneGraphNode* node : level)
            {
                for (SceneGraphNode* child : node->GetChildren())
                {
                    next.push_back(child);
                    if (written.count(ChunkOf(rig, node)) && !rig.IsDriven(child))
                    {
                        ++rows;
                        nextWritten.insert(ChunkOf(rig, child));
                    }
                }
            }
            written.insert(nextWritten.begin(), nextWritten.end());
            level.swap(next);
        }
        return rows;
    }

    // Every world transform against ExpectedWorld, bit for bit, with driven nodes at their own WorldTransform.
    void CheckChildrenFollow(Rig& rig)
    {
        for (SceneGraphWalker walker(rig.Root()); walker; walker.Next(true))
        {
            const dm::daffine3 expected = ExpectedWorld(rig, walker.Get(),
                [](SceneGraphNode* node) { return node->GetLocalToWorldTransform(); });
            CHECK(SameBits(walker->GetLocalToWorldTransform(), expected));
        }
    }

    const dm::dquat c_Turn = Unit(0.8, 0.2, -0.4, 0.35);
    const dm::dquat c_OtherTurn = Unit(-0.1, 0.7, 0.5, 0.2);
}

// ------------------------------------------------------------------------------------------------- driver

// Binding adds the components of SceneComponents.h in one move, from the node's world transform; the row that the domain
// sees is complete and no request is pending; LocalTransform stays as it is; nothing else changes.
void test_driver_bind()
{
    Rig rig;
    Placement parentPlace, place;
    parentPlace.translation = dm::double3(3.0, -2.0, 7.5);
    parentPlace.rotation = c_Turn;
    parentPlace.scaling = dm::double3(2.0);
    place.translation = dm::double3(0.5, 1.25, -4.0);
    place.rotation = c_OtherTurn;
    place.scaling = dm::double3(1.0, 0.5, 3.0);   // stretched along its own axes: no shear
    SceneGraphNode* parent = rig.Add(rig.Root(), &parentPlace);
    SceneGraphNode* node = rig.Add(parent, &place, true);
    SceneGraphNode* child = rig.AddAt(node, 1.0, 0.0, 0.0);
    rig.Refresh();
    rig.Refresh();
    const dm::daffine3 world = node->GetLocalToWorldTransform();
    const dm::daffine3 childWorld = child->GetLocalToWorldTransform();
    const LocalTransform local = *rig.Get<LocalTransform>(node);
    auto domain = rig.NewDomain();
    int seen = 0;
    domain->onAdded = [&](const EntityBatch& batch, ISimulationContext* context)
    {
        IWorld* w = context->GetWorld();
        for (uint32_t i = 0; i < batch.count; ++i)
        {
            ++seen;
            const Entity e = batch.entities[i];
            EXPECT(w->HasComponent(e, FindComponent<SimTeleport>(w)) && !w->IsEnabled(e, FindComponent<SimTeleport>(w)));
            EXPECT(GetComponent<SimScale>(w, e) && GetComponent<SimPosePrevious>(w, e));
            EXPECT(SameBits(GetComponent<WorldTransform>(w, e), &world, sizeof(world)));
            EXPECT(static_cast<const SimBodyRef*>(batch.columns[0])[i].index == 0);
        }
    };
    const SimulationDomainId id = rig.AddDomain(domain);
    const uint64_t structural = rig.world->GetStructuralVersion();
    CHECK(rig.graph->SetSimulationDriven(node, true) == nvrhi::FS_OK);
    CHECK(seen == 1 && domain->events.size() == 1 && domain->events[0].first == rig.EntityOf(node));
    // one move of the row (its new archetype, a row that comes, a row that goes): not a move per component
    CHECK(rig.world->GetStructuralVersion() == structural + 3);

    CHECK(rig.IsDriven(node) && !rig.IsDriven(child) && !rig.IsDriven(parent));
    const SimPose pose = *rig.Get<SimPose>(node);
    CHECK(SameBits(&pose, rig.Get<SimPosePrevious>(node), sizeof(pose)));
    CHECK(SameBits(&pose.translation, &world.m_translation, sizeof(pose.translation)));
    CHECK(Near(dm::dot(pose.rotation, pose.rotation), 1.0, 1e-15));
    CHECK(Near(rig.Get<SimScale>(node)->scaling, dm::double3(2.0, 1.0, 6.0), 1e-14));
    CHECK(Near(rig.Posed(node), world, 1e-14));
    CHECK(rig.Get<SimBodyRef>(node)->domain == uint32_t(id) && rig.Get<SimBodyRef>(node)->index == 0);
    CHECK(!rig.world->IsEnabled(rig.EntityOf(node), FindComponent<SimTeleport>(rig.world)));
    CHECK(SameBits(rig.Get<LocalTransform>(node), &local, sizeof(local)));
    CHECK(SameBits(node->GetLocalToWorldTransform(), world) && node->GetLeaf() != nullptr);
    CHECK(node->GetParent() == parent && node->GetFirstChild() == child);
    CHECK(!rig.graph->HasStaleWorldTransforms() && !rig.graph->HasPendingTransformChanges() &&
          !rig.graph->HasPendingStructureChanges());

    // a refresh moves nothing, not even the last bit
    rig.Refresh();
    CHECK(SameBits(node->GetLocalToWorldTransform(), world) && SameBits(child->GetLocalToWorldTransform(), childWorld));
    // binding again does nothing, whatever the pose is by now
    rig.SetPose(node, dm::double3(9.0), c_Turn);
    const uint64_t version = rig.world->GetVersion();
    CHECK(rig.graph->SetSimulationDriven(node, true) == nvrhi::FS_OK);
    CHECK(rig.world->GetVersion() == version && rig.Get<SimPose>(node)->translation.x == 9.0 && seen == 1);

    // without a domain: of no domain, and no request; also in rows that a swap-remove moves
    SceneGraphNode* others[3];
    auto none = MAKE_RC_OBJ_PTR(SceneGraph);
    none->SetRootNode(MakeNode());
    for (SceneGraphNode*& other : others)
    {
        other = none->Attach(none->GetRootNode(), MakeNode());
        CHECK(none->SetSimulationDriven(other, true) == nvrhi::FS_OK);
        CHECK(GetComponent<SimBodyRef>(none->GetWorld(), none->GetEntity(other))->domain == c_NoSimulationDomain);
    }
    const ComponentId teleport = FindComponent<SimTeleport>(none->GetWorld());
    CHECK(none->GetWorld()->SetEnabled(none->GetEntity(others[1]), teleport, true) == nvrhi::FS_OK);
    CHECK(none->SetSimulationDriven(others[0], false) == nvrhi::FS_OK);
    CHECK(none->GetWorld()->IsEnabled(none->GetEntity(others[1]), teleport));
    CHECK(!none->GetWorld()->IsEnabled(none->GetEntity(others[2]), teleport));

    domain->onAdded = nullptr;
    {   // the request bit of every row of a chunk, beyond the mask's first word
        Rig many;
        std::vector<SceneGraphNode*> bound;
        for (int i = 0; i < 260; ++i)
        {
            bound.push_back(many.AddAt(many.Root(), double(i), 0.0, 0.0));
            many.Bind(bound.back());
        }
        const ComponentId request = FindComponent<SimTeleport>(many.world);
        uint32_t highest = 0;
        for (SceneGraphNode* at : bound)
        {
            CHECK(!many.world->IsEnabled(many.EntityOf(at), request));
            highest = std::max<uint32_t>(highest, many.Impl().FindAlive(many.EntityOf(at))->row);
        }
        CHECK(highest >= 70);
        // enabled ones stay so when more rows come
        for (size_t i = 0; i < bound.size(); i += 3)
            CHECK(many.world->SetEnabled(many.EntityOf(bound[i]), request, true) == nvrhi::FS_OK);
        for (int i = 0; i < 100; ++i)
        {
            SceneGraphNode* at = many.AddAt(many.Root(), 0.0, double(i), 0.0);
            many.Bind(at);
            CHECK(!many.world->IsEnabled(many.EntityOf(at), request));
        }
        for (size_t i = 0; i < bound.size(); ++i)
            CHECK(many.world->IsEnabled(many.EntityOf(bound[i]), request) == (i % 3 == 0));
    }
    // rotations of every sign pattern, and small ones, come back exactly enough to stand still
    Rng rng(77);
    for (int i = 0; i < 400; ++i)
    {
        Placement p = RandomPlacement(rng);
        if (i % 4 == 1)
            p.rotation = Unit(1.0, 1e-9, -3e-9, 2e-9);
        if (i % 4 == 2)
            p.rotation = Unit(1e-9, i % 3 ? 1.0 : 0.0, i % 3 == 2 ? 1e-3 : 0.0, i % 3 ? 0.0 : 1.0);
        SceneGraphNode* at = rig.Add(rig.Root(), &p);
        rig.Bind(at);
        CHECK(Near(rig.Posed(at), at->GetLocalToWorldTransform(), 3e-15));
        CHECK(Near(rig.Get<SimPose>(at)->rotation, p.rotation, 1e-15));
    }
}

// Binding takes edits that no propagation has processed: of the node, of its ancestors and of new nodes; its world
// transform is then the one that propagation would have given, bit for bit, and its children follow at the next Refresh.
void test_driver_bind_stale()
{
    for (int structure = 0; structure < 3; ++structure)
    {
        Rig rig, twin;
        auto build = [&](Rig& r) -> std::vector<SceneGraphNode*>
        {
            Rng rng(5);
            std::vector<SceneGraphNode*> nodes;
            SceneGraphNode* parent = r.Root();
            for (int i = 0; i < 6; ++i)
            {
                const Placement p = RandomPlacement(rng);
                parent = r.Add(parent, i == 3 ? nullptr : &p);   // one without a transform
                nodes.push_back(parent);
            }
            if (structure != 0)
                r.Refresh();
            if (structure == 1)   // an ancestor moves
            {
                Place(nodes[1], RandomPlacement(rng));
                Place(r.Root(), RandomPlacement(rng));
            }
            if (structure == 2)   // the node itself moves
                Place(nodes[4], RandomPlacement(rng));
            return nodes;
        };
        const auto nodes = build(rig);
        const auto reference = build(twin);
        CHECK(rig.graph->HasStaleWorldTransforms());
        rig.Bind(nodes[4]);
        CHECK(rig.graph->HasStaleWorldTransforms());   // not set or cleared by binding
        twin.Refresh();
        const dm::daffine3 expected = reference[4]->GetLocalToWorldTransform();
        CHECK(SameBits(nodes[4]->GetLocalToWorldTransform(), expected));
        CHECK(Near(rig.Posed(nodes[4]), expected, 1e-13));
        rig.Refresh();
        for (size_t i = 0; i < nodes.size(); ++i)
            CHECK(SameBits(nodes[i]->GetLocalToWorldTransform(), reference[i]->GetLocalToWorldTransform()));
        CheckChildrenFollow(rig);
    }

    // below a driven node: over its simulated pose, not the interpolated one that is shown
    Rig rig;
    auto domain = rig.NewDomain();
    const SimulationDomainId id = rig.AddDomain(domain);
    SceneGraphNode* upper = rig.AddAt(rig.Root(), 0.0, 10.0, 0.0);
    SceneGraphNode* lower = rig.AddAt(upper, 1.0, 2.0, 3.0);
    rig.Refresh();
    rig.Bind(upper);
    rig.Frame(0.25 * 1.5);
    CHECK(rig.Alpha(id) == 0.5);
    lower->SetTranslation(dm::double3(-1.0, 0.5, 0.0));   // stale
    rig.Bind(lower);
    const LocalTransform local = *rig.Get<LocalTransform>(lower);
    const dm::daffine3 matrix = ComposeTransform(local.translation, local.rotation, local.scaling);
    CHECK(SameBits(lower->GetLocalToWorldTransform(), matrix * rig.Posed(upper)));
    CHECK(!Near(matrix * rig.Posed(upper), matrix * upper->GetLocalToWorldTransform(), 1e-3));
    CHECK(SameBits(rig.Get<SimPose>(lower), rig.Get<SimPosePrevious>(lower), sizeof(SimPose)));
}

// Binding does not depend on what the WorldTransform column holds, nor on an unrelated edit: between the interpolation
// and the Refresh, below driven nodes that moved on; and with local transforms written through the world, which no flag
// knows of. Bound and unbound once per frame on a moving parent, a node keeps its place on it.
void test_driver_bind_between()
{
    dm::daffine3 results[2];
    for (int unrelatedEdit = 0; unrelatedEdit < 2; ++unrelatedEdit)
    {
        Rig rig;
        auto domain = rig.NewDomain();
        rig.AddDomain(domain);
        SceneGraphNode* train = rig.AddAt(rig.Root(), 0.0, 10.0, 0.0);
        SceneGraphNode* group = rig.Add(train);
        SceneGraphNode* crate = rig.AddAt(group, 1.0, 0.5, 0.0, true);
        SceneGraphNode* child = rig.AddAt(crate, 0.0, 1.0, 0.0);
        SceneGraphNode* unrelated = rig.AddAt(rig.Root(), 0.0, 0.0, 0.0);
        rig.Bind(train);
        rig.Refresh();
        rig.Frame(0.25 * 1.5);
        rig.Simulate(0.25 * 2.0);   // the train moved on; its children did not yet
        CHECK(!rig.graph->HasStaleWorldTransforms());
        if (unrelatedEdit)
            unrelated->SetTranslation(dm::double3(5.0));
        const dm::daffine3 expected =
            ComposeTransform(dm::double3(1.0, 0.5, 0.0), dm::dquat::identity(), dm::double3(1.0)) * rig.Posed(train);
        CHECK(!Near(crate->GetLocalToWorldTransform(), expected, 1e-3));
        rig.Bind(crate);
        results[unrelatedEdit] = crate->GetLocalToWorldTransform();
        CHECK(SameBits(results[unrelatedEdit], expected) && Near(rig.Posed(crate), expected, 1e-14));
        rig.Refresh();
        CHECK(SameBits(crate->GetLocalToWorldTransform(), expected));
        const dm::box3 box = rig.mesh->objectSpaceBounds * dm::affine3(expected);
        CHECK(all(rig.Get<WorldBounds>(crate)->bounds.m_mins == box.m_mins));
        CheckChildrenFollow(rig);
        rig.Unbind(crate);
        CHECK(Near(crate->GetTranslation(), dm::double3(1.0, 0.5, 0.0), 1e-13));
        CHECK(Near(crate->GetRotation(), dm::dquat::identity(), 1e-13) && child->GetParent() == crate);
    }
    CHECK(SameBits(results[0], results[1]));

    {   // a local transform written through the world: the parent's, then the node's own
        Rig rig;
        SceneGraphNode* parent = rig.AddAt(rig.Root(), 1.0, 0.0, 0.0);
        SceneGraphNode* node = rig.AddAt(parent, 1.0, 0.0, 0.0);
        SceneGraphNode* other = rig.AddAt(parent, 2.0, 0.0, 0.0);
        rig.Refresh();
        rig.world->AdvanceVersion();
        rig.Edit<LocalTransform>(parent)->translation.x = 100.0;
        rig.Edit<LocalTransform>(other)->translation.x = 7.0;
        CHECK(!rig.graph->HasStaleWorldTransforms());
        rig.Bind(node);
        rig.Bind(other);
        CHECK(rig.Get<SimPose>(node)->translation.x == 101.0 && rig.Get<SimPose>(other)->translation.x == 107.0);
        rig.Refresh();
        CHECK(parent->GetLocalToWorldTransform().m_translation.x == 100.0);
        CHECK(node->GetLocalToWorldTransform().m_translation.x == 101.0);
    }

    // bind and unbind in every frame, at every point of it, with an authored node between
    Rig rig;
    const ComponentId heavy = RegisterComponent<HeavyBall>(rig.world);
    auto domain = rig.NewDomain("balls", 0.25, 4, heavy);   // it does not take the crate
    rig.AddDomain(domain);
    SceneGraphNode* train = rig.AddAt(rig.Root(), 0.0, 10.0, 0.0);
    CHECK(rig.world->AddComponent(rig.EntityOf(train), heavy, nullptr) == nvrhi::FS_OK);
    Placement jointPlace, cratePlace;
    jointPlace.rotation = c_Turn;
    jointPlace.scaling = dm::double3(1.5);
    cratePlace.translation = dm::double3(1.0, 2.0, 3.0);
    cratePlace.rotation = c_OtherTurn;
    cratePlace.scaling = dm::double3(0.75);
    SceneGraphNode* joint = rig.Add(train, &jointPlace);
    SceneGraphNode* crate = rig.Add(joint, &cratePlace);
    rig.Bind(train);
    rig.Refresh();
    auto bindUnbind = [&]()
    {
        rig.Bind(crate);
        rig.Unbind(crate);
        CHECK(Near(crate->GetTranslation(), cratePlace.translation, 1e-11) &&
              Near(crate->GetRotation(), cratePlace.rotation, 1e-11) &&
              Near(crate->GetScaling(), cratePlace.scaling, 1e-11));
    };
    for (int frame = 0; frame < 40; ++frame)
    {
        bindUnbind();
        rig.Simulate(0.25 * 1.3);
        bindUnbind();
        rig.Refresh();
        bindUnbind();
    }
    rig.Refresh();
    CHECK(Near(crate->GetLocalToWorldTransform() * inverse(train->GetLocalToWorldTransform()),
               ComposeTransform(cratePlace.translation, cratePlace.rotation, cratePlace.scaling) *
               ComposeTransform(jointPlace.translation, jointPlace.rotation, jointPlace.scaling), 1e-9));
    CheckChildrenFollow(rig);
}

// SimBodyRef's contract: a domain resets it in OnEntitiesRemoved, in the batch's column, also of rows that stay. Such a
// row shows its last pose from then on, whatever domain gets the id later, or still has it.
void test_driver_body_refs()
{
    for (bool leaves : { false, true })
    {
        for (bool clears : { true, false })
        {
            Rig rig;
            const ComponentId heavy = RegisterComponent<HeavyBall>(rig.world);
            const ComponentId light = RegisterComponent<LightBall>(rig.world);
            auto domain = rig.NewDomain("light", 0.25, 4, light);
            domain->clearsBodyRefs = clears;
            const SimulationDomainId id = rig.AddDomain(domain);
            std::vector<SceneGraphNode*> nodes;
            for (int i = 0; i < 5; ++i)
            {
                nodes.push_back(rig.AddAt(rig.Root(), double(i), 10.0, 0.0));
                CHECK(rig.world->AddComponent(rig.EntityOf(nodes[i]), light, nullptr) == nvrhi::FS_OK);
                rig.Bind(nodes[i]);
            }
            rig.Frame(0.25 * 1.5);
            CHECK(rig.Get<SimBodyRef>(nodes[1])->domain == uint32_t(id) && rig.Alpha(id) == 0.5);
            SimulationDomainId ticking = id;
            if (leaves)   // the domain lives on; one row leaves its query, and stays driven
            {
                CHECK(rig.world->RemoveComponent(rig.EntityOf(nodes[1]), light) == nvrhi::FS_OK);
            }
            else          // the domain goes, and another one gets its id; it has no entities
            {
                CHECK(rig.graph->GetScheduler()->RemoveSimulationDomain(id) == nvrhi::FS_OK);
                auto next = rig.NewDomain("heavy", 0.25, 4, heavy);
                ticking = rig.AddDomain(next);
                CHECK(ticking == id && next->balls.empty());
            }
            CHECK(rig.IsDriven(nodes[1]) && !domain->Find(rig.EntityOf(nodes[1])));
            const SimBodyRef body = *rig.Get<SimBodyRef>(nodes[1]);
            CHECK((body.domain == c_NoSimulationDomain && body.index == 0) == clears);
            CHECK((rig.Get<SimBodyRef>(nodes[0])->domain == c_NoSimulationDomain) == (clears && !leaves));
            const SimPose pose = *rig.Get<SimPose>(nodes[1]);
            uint32_t elsewhere = 0;
            for (int frame = 0; frame < 6; ++frame)
            {
                rig.Frame(0.1);
                CHECK(SameBits(rig.Get<SimPose>(nodes[1]), &pose, sizeof(pose)));
                elsewhere += SameBits(nodes[1]->GetLocalToWorldTransform(), rig.Posed(nodes[1])) ? 0 : 1;
                if (leaves)   // the others are still the domain's
                {
                    CHECK(SameBits(nodes[0]->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(nodes[0]),
                          *rig.Get<SimPose>(nodes[0]), *rig.Get<SimScale>(nodes[0]), rig.Alpha(ticking))));
                }
            }
            CHECK((elsewhere == 0) == clears);   // one that does not reset it swings between its two old poses
        }
    }
}

// Propagation leaves driven rows alone, and their children follow them: at several levels, with and without transforms
// of their own, below driven nodes whose own parents move; the rows that are computed are counted.
void test_driver_propagation()
{
    Rig rig;
    Rng rng(9);
    const Placement groupPlace = RandomPlacement(rng);
    SceneGraphNode* group = rig.Add(rig.Root(), &groupPlace);
    std::vector<SceneGraphNode*> driven, children, grandchildren, others;
    for (int i = 0; i < 600; ++i)   // more than one chunk of each kind
    {
        const Placement p = RandomPlacement(rng), q = RandomPlacement(rng), r = RandomPlacement(rng);
        driven.push_back(rig.Add(group, &p, i % 2 == 0));
        children.push_back(rig.Add(driven.back(), i % 3 ? &q : nullptr, i % 2 == 0));
        grandchildren.push_back(rig.Add(children.back(), &r));
        others.push_back(rig.Add(group, &q));
        rig.Add(others.back(), &r)->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));   // in chunks of their own
    }
    rig.Refresh();
    for (SceneGraphNode* node : driven)
        rig.Bind(node);
    rig.Refresh();
    CheckChildrenFollow(rig);
    rig.Refresh();

    rig.Frame(0.0);   // the first interpolation writes every driven row
    for (SceneGraphNode* node : driven)
        CHECK(SameBits(node->GetLocalToWorldTransform(), rig.Posed(node)));
    CheckChildrenFollow(rig);

    // nothing moves: nothing is computed (the phases move the world's version, so the Refresh runs)
    uint64_t rows = rig.PropagatedRows();
    rig.Frame(0.0);
    CHECK(rig.PropagatedRows() == rows);

    // the parent of the driven nodes moves: they stay, and so do their children; the other branch follows
    std::vector<dm::daffine3> before;
    for (size_t i = 0; i < driven.size(); ++i)
    {
        before.push_back(driven[i]->GetLocalToWorldTransform());
        before.push_back(grandchildren[i]->GetLocalToWorldTransform());
    }
    rows = rig.PropagatedRows();
    Place(group, RandomPlacement(rng));
    rig.Refresh();
    for (size_t i = 0; i < driven.size(); ++i)
    {
        CHECK(SameBits(driven[i]->GetLocalToWorldTransform(), before[2 * i]));
        CHECK(SameBits(grandchildren[i]->GetLocalToWorldTransform(), before[2 * i + 1]));
    }
    CHECK(!SameBits(others[0]->GetLocalToWorldTransform(), ExpectedWorld(rig, rig.Root(),
          [](SceneGraphNode*) { return dm::daffine3::identity(); })));
    CheckChildrenFollow(rig);
    CHECK(rig.PropagatedRows() - rows == 1 + 2 * others.size());   // the group, the others and their children

    // one driven node moves: the children of the rows of its chunk are computed, and theirs, and nothing else
    const uint64_t rowsBelow = RowsBelow(rig, { ChunkOf(rig, driven[250]) });
    CHECK(rowsBelow > 2 && rowsBelow < driven.size());
    rows = rig.PropagatedRows();
    rig.SetPose(driven[250], dm::double3(100.0, 200.0, 300.0), c_Turn);
    rig.Simulate(0.0);
    rig.Refresh();
    CHECK(SameBits(driven[250]->GetLocalToWorldTransform(), rig.Posed(driven[250])));
    CHECK(rig.PropagatedRows() - rows == rowsBelow);
    CheckChildrenFollow(rig);

    // all of them move; only the level of the driven rows has none to compute
    rows = rig.PropagatedRows();
    for (SceneGraphNode* node : driven)
        rig.SetPose(node, RandomVector(rng, -30.0, 30.0), RandomRotation(rng));
    rig.Simulate(0.0);
    rig.Refresh();
    for (SceneGraphNode* node : driven)
        CHECK(SameBits(node->GetLocalToWorldTransform(), rig.Posed(node)));
    CHECK(rig.PropagatedRows() - rows == 2 * driven.size());
    CheckChildrenFollow(rig);

    // an authored child moves under a resting driven node
    rows = rig.PropagatedRows();
    children[7]->SetTranslation(dm::double3(0.125, 0.0, 4.0));
    rig.Refresh();
    CHECK(rig.PropagatedRows() > rows);
    CheckChildrenFollow(rig);

    // a driven root: the whole graph follows it
    Rig rooted;
    SceneGraphNode* below = rooted.AddAt(rooted.Root(), 1.0, 2.0, 3.0);
    SceneGraphNode* bare = rooted.Add(below);
    rooted.Refresh();
    rooted.Bind(rooted.Root());
    rooted.SetPose(rooted.Root(), dm::double3(10.0, 0.0, 0.0), c_OtherTurn);
    rooted.Simulate(0.0);
    rooted.Refresh();
    CHECK(Near(below->GetLocalToWorldTransform().m_translation - dm::double3(10.0, 0.0, 0.0),
               dm::double3(1.0, 2.0, 3.0) * c_OtherTurn.toMatrix(), 1e-14));
    CHECK(SameBits(bare->GetLocalToWorldTransform(), below->GetLocalToWorldTransform()));
    CheckChildrenFollow(rooted);
}

namespace
{
    // An entity without a node: a row of the graph's world with the columns propagation reads.
    Entity MakeRuntime(Rig& rig, uint32_t depth, Entity parent, double x, bool driven)
    {
        const HierarchyDepth depthValue{ depth };
        const Parent parentValue{ parent };
        const LocalTransform local{ dm::double3(x, 0.0, 0.0), dm::dquat::identity(), dm::double3(1.0) };
        const WorldTransform world{ dm::daffine3::identity() };
        const SimPose pose{ dm::double3(x, 0.0, 0.0), dm::dquat::identity() };
        const SimPosePrevious previous{ pose.translation, pose.rotation };
        const SimScale scale{ dm::double3(1.0) };
        std::vector<ComponentId> ids = { FindComponent<HierarchyDepth>(rig.world),
                                         FindComponent<WorldTransform>(rig.world) };
        std::vector<const void*> values = { &depthValue, &world };
        auto add = [&](ComponentId id, const void* value)
        {
            ids.push_back(id);
            values.push_back(value);
        };
        if (parent != Entity::Null)
            add(FindComponent<Parent>(rig.world), &parentValue);
        if (driven)
        {
            add(FindComponent<SimulationDriven>(rig.world), nullptr);
            add(FindComponent<SimPose>(rig.world), &pose);
            add(FindComponent<SimPosePrevious>(rig.world), &previous);
            add(FindComponent<SimScale>(rig.world), &scale);
            add(FindComponent<SimBodyRef>(rig.world), nullptr);
        }
        else
            add(FindComponent<LocalTransform>(rig.world), &local);
        EntityCreateDesc desc{};
        desc.components = ids.data();
        desc.values = values.data();
        desc.componentCount = uint32_t(ids.size());
        desc.entityCount = 1;
        Entity entity = Entity::Null;
        CHECK(rig.world->CreateEntities(desc, &entity) == nvrhi::FS_OK);
        return entity;
    }
}

// Entities without nodes: driven ones are interpolated, rows below them follow, also in the last bucket, where nothing
// links the rows (the serial pass), and below a driven row there that has no parent.
void test_driver_runtime_entities()
{
    Rig rig;
    struct Chain { Entity driven, below, last; double offset; };
    std::vector<Chain> chains;
    for (int i = 0; i < 300; ++i)
    {
        const bool deep = i % 2 != 0;
        const uint32_t depth = deep ? c_MaxHierarchyDepth : 0;
        Entity parent = Entity::Null;
        if (i % 4 == 3)   // a deep one below a shallow row: an entry of the bucket
            parent = MakeRuntime(rig, c_MaxHierarchyDepth - 1, Entity::Null, 5.0, false);
        Chain chain;
        chain.driven = MakeRuntime(rig, depth, parent, double(i), true);
        chain.below = MakeRuntime(rig, deep ? depth : 1, chain.driven, 1000.0, false);
        chain.last = MakeRuntime(rig, deep ? depth : 2, chain.below, 0.5, false);
        chains.push_back(chain);
    }
    auto worldOf = [&](Entity e) { return GetComponent<WorldTransform>(rig.world, e)->transform.m_translation.x; };
    auto check = [&]()
    {
        for (const Chain& chain : chains)
        {
            const double x = GetComponent<SimPose>(rig.world, chain.driven)->translation.x;
            CHECK(worldOf(chain.driven) == x && worldOf(chain.below) == 1000.0 + x && worldOf(chain.last) == 1000.5 + x);
        }
    };
    rig.Frame(0.0);
    check();
    rig.Frame(0.0);
    check();
    for (int round = 0; round < 6; ++round)
    {
        rig.world->AdvanceVersion();
        for (size_t i = size_t(round); i < chains.size(); i += 6)
            GetComponentForWrite<SimPose>(rig.world, chains[i].driven)->translation.x += 10000.0;
        rig.Frame(0.0);
        check();
    }
    // one deep driven row without a parent: nothing else in the bucket moved, the level above wrote nothing
    rig.world->AdvanceVersion();
    GetComponentForWrite<SimPose>(rig.world, chains[1].driven)->translation.x = -3.0;
    rig.Frame(0.0);
    check();
}

// The last depth bucket: driven rows just above it, at its entry, inside it and with driven ancestors in it; a moved one
// takes its subtree along, others stay; several chains at once, on several threads.
void test_driver_deep()
{
    for (uint32_t workers : { 1u, 8u })
    {
        Rig rig(workers);
        Rng rng(31);
        constexpr uint32_t c_Chains = 24, c_Depth = 110;
        std::vector<std::vector<SceneGraphNode*>> chains(c_Chains);
        for (auto& chain : chains)
        {
            auto top = MakeNode();
            nvrhi::AutoPtr<SceneGraphNode> at = top;
            for (uint32_t d = 0; d < c_Depth; ++d)
            {
                auto node = d ? MakeNode() : top;
                if (d % 5 != 3)
                {
                    Placement p = RandomPlacement(rng);
                    p.scaling = dm::double3(1.0);
                    Place(node, p);
                }
                if (d)
                    rig.graph->Attach(at, node);
                at = node;
            }
            rig.graph->Attach(rig.Root(), top);
            for (SceneGraphNode* node = top; node; node = node->GetFirstChild())
                chain.push_back(node);
        }
        rig.Refresh();
        CheckChildrenFollow(rig);
        // chain[i] is at depth i + 1: 61..63 are around the bucket's edge (62: the last level, 63: its entries)
        const uint32_t places[] = { 20, 60, 61, 62, 63, 64, 80, 100, 109 };
        std::vector<SceneGraphNode*> driven;
        for (uint32_t c = 0; c < c_Chains; ++c)
        {
            for (size_t k = 0; k < std::size(places); ++k)
            {
                if ((c >> (k % 4)) & 1 || c % 9 == k)
                {
                    driven.push_back(chains[c][places[k]]);
                    rig.Bind(driven.back());
                }
            }
        }
        CHECK(driven.size() > 60);
        rig.Refresh();
        CheckChildrenFollow(rig);
        for (int round = 0; round < 12; ++round)
        {
            std::set<SceneGraphNode*> moved;
            for (int k = 0; k < (round % 4 == 3 ? 40 : 2); ++k)
            {
                SceneGraphNode* node = driven[rng.Below(driven.size())];
                rig.SetPose(node, RandomVector(rng, -30.0, 30.0), RandomRotation(rng));
                moved.insert(node);
            }
            if (round % 3 == 0)   // authored rows move too, in both parts of the chains
            {
                Placement p = RandomPlacement(rng);
                p.scaling = dm::double3(1.0);
                Place(chains[rng.Below(c_Chains)][5], p);
                SceneGraphNode* deep = chains[rng.Below(c_Chains)][85];
                if (!rig.IsDriven(deep))
                    Place(deep, p);
            }
            rig.Simulate(0.0);
            rig.Refresh();
            for (SceneGraphNode* node : moved)
                CHECK(SameBits(node->GetLocalToWorldTransform(), rig.Posed(node)));
            CheckChildrenFollow(rig);
        }
        // only a driven row inside the bucket moved: the bucket runs although the level above wrote nothing
        SceneGraphNode* inside = nullptr;
        for (uint32_t c = 0; c < c_Chains && !inside; ++c)
        {
            if (rig.IsDriven(chains[c][80]))
                inside = chains[c][80];
        }
        CHECK(inside);
        rig.Refresh();
        const uint64_t deepRows = rig.Store().GetCounters().deepRows;
        rig.Simulate(0.0);
        rig.Refresh();
        CHECK(rig.Store().GetCounters().deepRows == deepRows);   // nothing moved: not visited
        // Exactly the rows that follow: those whose nearest driven ancestor is in a chunk that was written (in the bucket
        // an authored row passes it on by its own mark; only driven rows have chunk versions to go by).
        uint64_t follow = 0;
        const ChunkKey written = ChunkOf(rig, inside);
        for (const auto& chain : chains)
        {
            bool moved = false;
            for (SceneGraphNode* node : chain)
            {
                if (rig.IsDriven(node))
                    moved = ChunkOf(rig, node) == written;
                else
                    follow += moved ? 1 : 0;
            }
        }
        const uint64_t rows = rig.PropagatedRows();
        const dm::daffine3 below = inside->GetFirstChild()->GetLocalToWorldTransform();
        rig.SetPose(inside, dm::double3(1000.0, 0.0, 0.0), c_Turn);
        rig.Simulate(0.0);
        rig.Refresh();
        CHECK(!SameBits(inside->GetFirstChild()->GetLocalToWorldTransform(), below));
        CHECK(follow > 0 && rig.PropagatedRows() - rows == follow);
        CHECK(rig.Store().GetCounters().deepRows > deepRows);
        CheckChildrenFollow(rig);
    }
}

// Unbinding makes the simulated pose (not the interpolated one) the local transform, relative to the parent as it is
// now, and removes the rest; the domain gives its object back; a node without a transform gets one.
void test_driver_unbind()
{
    Rig rig;
    auto domain = rig.NewDomain();
    const SimulationDomainId id = rig.AddDomain(domain);
    Placement parentPlace;
    parentPlace.translation = dm::double3(1.0, 2.0, 3.0);
    parentPlace.rotation = c_Turn;
    parentPlace.scaling = dm::double3(2.5);
    SceneGraphNode* parent = rig.Add(rig.Root(), &parentPlace);
    SceneGraphNode* node = rig.AddAt(parent, 4.0, 5.0, 6.0, true);
    SceneGraphNode* bare = rig.Add(parent);   // without a transform
    SceneGraphNode* child = rig.AddAt(node, 0.0, 1.0, 0.0);
    rig.Refresh();
    for (SceneGraphNode* n : { node, bare })
        rig.Bind(n);
    for (int i = 0; i < 5; ++i)
        rig.Frame(0.25 * 1.25);
    CHECK(rig.Alpha(id) == 0.25);
    rig.Refresh();
    CHECK(!rig.graph->HasPendingTransformChanges());
    const dm::daffine3 posed = rig.Posed(node), bareposed = rig.Posed(bare);
    CHECK(!Near(posed, node->GetLocalToWorldTransform(), 1e-3));   // interpolated, behind it
    CHECK(rig.Get<LocalTransform>(node)->translation.x == 4.0 && !rig.Get<LocalTransform>(bare));

    // the parent moves and no propagation ran: the pose is taken relative to where it is now
    Placement moved = parentPlace;
    moved.translation = dm::double3(-7.0, 0.5, 0.0);
    moved.rotation = c_OtherTurn;
    Place(parent, moved);
    domain->events.clear();
    const Entity entity = rig.EntityOf(node);
    for (SceneGraphNode* n : { node, bare })
    {
        rig.Unbind(n);
        CHECK(rig.graph->HasStaleWorldTransforms() && rig.graph->HasPendingTransformChanges());
        CHECK(!rig.IsDriven(n));
        CHECK(!rig.Get<SimPose>(n) && !rig.Get<SimPosePrevious>(n) && !rig.Get<SimScale>(n) && !rig.Get<SimBodyRef>(n));
        CHECK(!rig.world->HasComponent(rig.EntityOf(n), FindComponent<SimTeleport>(rig.world)));
        CHECK(rig.Get<LocalTransform>(n) &&
              rig.world->HasComponent(rig.EntityOf(n), FindComponent<LocalMatrix>(rig.world)));
    }
    CHECK(rig.EntityOf(node) == entity && node->GetLeaf() != nullptr && node->GetFirstChild() == child);
    CHECK(domain->events.size() == 2 && !domain->events[0].second && domain->freeIndices.size() == 2);
    rig.Refresh();
    CHECK(Near(node->GetLocalToWorldTransform(), posed) && Near(bare->GetLocalToWorldTransform(), bareposed));
    CHECK(Near(node->GetTranslation(), rig.Get<LocalTransform>(node)->translation, 0.0));
    CHECK(Near(node->GetScaling(), dm::double3(1.0)));
    CheckChildrenFollow(rig);

    // it is authored again: the setters move it, frames do not
    node->SetTranslation(dm::double3(0.0));
    rig.Frame(0.5);
    CHECK(Near(node->GetLocalToWorldTransform().m_translation, parent->GetLocalToWorldTransform().m_translation));
    // unbinding it again does nothing
    rig.Refresh();
    const uint64_t version = rig.world->GetVersion();
    CHECK(rig.graph->SetSimulationDriven(node, false) == nvrhi::FS_OK);
    CHECK(!rig.graph->HasStaleWorldTransforms() && rig.world->GetVersion() == version);
    // and it can be bound again, from where it is
    rig.Bind(node);
    CHECK(rig.Get<SimBodyRef>(node)->domain == uint32_t(id) && rig.Get<SimBodyRef>(node)->index < 2);
    CHECK(Near(rig.Posed(node), node->GetLocalToWorldTransform(), 1e-13));

    // under a driven parent: both of the same time, so that it keeps its place on the parent while the parent moves on
    Rig train;
    auto engine = train.NewDomain();
    train.AddDomain(engine);
    SceneGraphNode* car = train.AddAt(train.Root(), 0.0, 50.0, 0.0);
    SceneGraphNode* joint = train.AddAt(car, 0.0, 1.0, 0.0);   // authored, between the two
    SceneGraphNode* crate = train.AddAt(joint, 1.0, 1.0, 1.0);
    train.Refresh();
    train.Bind(car);
    train.Bind(crate);
    for (int i = 0; i < 3; ++i)
        train.Frame(0.25 * 1.25);
    CHECK(!Near(train.Posed(car), car->GetLocalToWorldTransform(), 1e-3));
    const dm::daffine3 onCar = train.Posed(crate) * inverse(train.Posed(car));
    train.Unbind(crate);
    const dm::daffine3 local = crate->GetLocalToParentTransform() * joint->GetLocalToParentTransform();
    CHECK(Near(local, onCar));
    train.Frame(0.25);
    CHECK(Near(crate->GetLocalToWorldTransform() * inverse(car->GetLocalToWorldTransform()), onCar));

    // a parent that shears the pose: the nearest local transform, with a warning, and the node is unbound
    Rig sheared;
    Placement stretched;
    stretched.scaling = dm::double3(1.0, 4.0, 1.0);
    SceneGraphNode* stretch = sheared.Add(sheared.Root(), &stretched);
    SceneGraphNode* inner = sheared.AddAt(stretch, 1.0, 0.0, 0.0);
    sheared.Refresh();
    sheared.Bind(inner);
    sheared.SetPose(inner, dm::double3(5.0, 6.0, 7.0), c_Turn);
    CHECK(TakeWarnings() == 0);
    sheared.Unbind(inner);
    CHECK(TakeWarnings() == 1 && !sheared.IsDriven(inner));
    sheared.Refresh();
    CHECK(Near(inner->GetLocalToWorldTransform().m_translation, dm::double3(5.0, 6.0, 7.0)));
    // a parent without an inverse: the pose itself, with a warning; so for Detach and copies
    SceneGraphNode* flat = sheared.Add(sheared.Root());
    SceneGraphNode* onFlat = sheared.AddAt(flat, 0.0, 0.0, 0.0);
    sheared.Bind(onFlat);
    sheared.SetPose(onFlat, dm::double3(5.0, 6.0, 7.0), c_Turn);
    flat->SetScaling(dm::double3(1.0, 0.0, 1.0));
    sheared.Unbind(onFlat);
    CHECK(TakeWarnings() == 1);
    CHECK(Near(onFlat->GetTranslation(), dm::double3(5.0, 6.0, 7.0), 1e-14));
    CHECK(Near(onFlat->GetRotation(), c_Turn, 1e-14));
    SceneGraphNode* second = sheared.AddAt(flat, 0.0, 0.0, 0.0);
    flat->SetScaling(dm::double3(1.0));
    sheared.Bind(second);
    flat->SetScaling(dm::double3(1.0, 0.0, 1.0));
    sheared.graph->Attach(sheared.Root(), second);
    CHECK(TakeWarnings() == 1);
    sheared.graph->Detach(second);
    CHECK(TakeWarnings() == 1);
    // a scale of 0 (a teleport may give one): no rotation
    SceneGraphNode* point = sheared.AddAt(sheared.Root(), 1.0, 2.0, 3.0);
    sheared.Bind(point);
    sheared.world->AdvanceVersion();
    sheared.Edit<SimScale>(point)->scaling = dm::double3(0.0);
    sheared.Unbind(point);
    CHECK(TakeWarnings() == 1 && Near(point->GetTranslation(), dm::double3(1.0, 2.0, 3.0), 0.0));
    CHECK(Near(point->GetScaling(), dm::double3(0.0), 0.0) && Near(point->GetRotation(), dm::dquat::identity(), 0.0));

    // out of memory for the authored row: nothing changed, the flags included
    Rig full;
    SceneGraphNode* mate = full.AddAt(full.Root(), 1.0, 0.0, 0.0);
    SceneGraphNode* victim = full.AddAt(full.Root(), 2.0, 0.0, 0.0);
    full.Bind(mate);
    full.Bind(victim);
    full.Unbind(mate);            // the archetypes and the transition are there
    full.graph->Detach(mate);     // and the authored one has no chunk
    SceneGraphNode* deepest = full.Root();
    for (int i = 0; i < 2000 && full.Impl().GetFreeChunkPageCount() != 0; ++i)
        deepest = i % 60 == 0 ? full.Add(full.Root()) : full.Add(deepest, nullptr, i % 7 == 3);
    CHECK(full.Impl().GetFreeChunkPageCount() == 0);
    full.Refresh();
    full.Refresh();
    CHECK(!full.graph->HasStaleWorldTransforms() && !full.graph->HasPendingTransformChanges());
    ecs::ReleaseSharedSlabs();
    ecs::SetHeapFailureCountdown(0);
    CHECK(full.graph->SetSimulationDriven(victim, false) == nvrhi::FE_OUT_OF_MEMORY);
    ecs::SetHeapFailureCountdown(-1);
    CHECK(TakeWarnings() == 1 && full.IsDriven(victim));
    CHECK(!full.graph->HasStaleWorldTransforms() && !full.graph->HasPendingTransformChanges());
    full.Unbind(victim);
    CHECK(!full.IsDriven(victim) && full.graph->HasStaleWorldTransforms());
    CHECK(Near(victim->GetTranslation(), dm::double3(2.0, 0.0, 0.0), 1e-14));
}

// Detach stores the simulated pose in the record and unbinds, for every driven node of the subgraph; attached again, the
// nodes are authored; the LocalTransform that a row kept while driven comes back nowhere.
void test_driver_detach()
{
    Rig rig;
    auto domain = rig.NewDomain();
    const SimulationDomainId id = rig.AddDomain(domain);
    Placement place;
    place.translation = dm::double3(1.0, 2.0, 3.0);
    place.rotation = c_Turn;
    place.scaling = dm::double3(1.5);
    nvrhi::AutoPtr<SceneGraphNode> parent = rig.Add(rig.Root(), &place);
    nvrhi::AutoPtr<SceneGraphNode> upper = rig.AddAt(parent, 4.0, 0.0, 0.0, true);
    nvrhi::AutoPtr<SceneGraphNode> middle = rig.AddAt(upper, 0.0, 2.0, 0.0);      // authored
    nvrhi::AutoPtr<SceneGraphNode> lower = rig.Add(middle);                        // driven, without a transform
    nvrhi::AutoPtr<SceneGraphNode> sibling = rig.AddAt(middle, 1.0, 1.0, 1.0);     // driven, same parent
    nvrhi::AutoPtr<SceneGraphNode> last = rig.AddAt(lower, 0.5, 0.5, 0.5);         // authored
    rig.Refresh();
    for (SceneGraphNode* n : { upper.Get(), lower.Get(), sibling.Get() })
        rig.Bind(n);
    for (int i = 0; i < 4; ++i)
        rig.Frame(0.25 * 1.3);
    const double alpha = rig.Alpha(id);
    CHECK(alpha > 0.05 && alpha < 0.95);
    const dm::daffine3 worlds[] = { upper->GetLocalToWorldTransform(), lower->GetLocalToWorldTransform() };
    const dm::daffine3 posed[] = { rig.Posed(upper), rig.Posed(lower), rig.Posed(sibling) };
    CHECK(!Near(worlds[0], posed[0], 1e-3));
    const dm::daffine3 parentWorld = parent->GetLocalToWorldTransform();
    domain->events.clear();

    auto detached = rig.graph->Detach(upper);
    CHECK(detached == upper && domain->events.size() == 3 && domain->freeIndices.size() == 3);
    for (const auto& event : domain->events)
        CHECK(!event.second);
    CHECK(upper->GetGraph() == nullptr && rig.EntityOf(upper) == Entity::Null);
    // the records: local transforms, relative to the simulated poses above them; the authored ones as they were
    CHECK(Near(upper->GetLocalToParentTransform() * parentWorld, posed[0]));
    CHECK(Near(lower->GetLocalToParentTransform() * middle->GetLocalToParentTransform() * posed[0], posed[1]));
    CHECK(Near(sibling->GetLocalToParentTransform() * middle->GetLocalToParentTransform() * posed[0], posed[2]));
    CHECK(Near(middle->GetTranslation(), dm::double3(0.0, 2.0, 0.0), 0.0));
    CHECK(Near(last->GetTranslation(), dm::double3(0.5), 0.0));
    CHECK(!Near(upper->GetTranslation(), dm::double3(4.0, 0.0, 0.0), 1e-3));   // what the row kept is not the pose
    // the world transforms kept from the last graph are those of the last frame
    CHECK(SameBits(upper->GetLocalToWorldTransform(), worlds[0]));
    CHECK(SameBits(lower->GetLocalToWorldTransform(), worlds[1]));

    // a node in no graph cannot be driven
    CHECK(rig.graph->SetSimulationDriven(upper, true) == nvrhi::FE_INVALID_ARGS);
    CHECK(rig.graph->SetSimulationDriven(upper, false) == nvrhi::FE_INVALID_ARGS);

    // attached again: authored, at the simulated poses
    CHECK(rig.graph->Attach(parent, upper) == upper);
    CHECK(!rig.IsDriven(upper) && !rig.IsDriven(lower) && !rig.Get<SimPose>(upper) && domain->events.size() == 3);
    rig.Refresh();
    CHECK(Near(upper->GetLocalToWorldTransform(), posed[0]) && Near(lower->GetLocalToWorldTransform(), posed[1]) &&
          Near(sibling->GetLocalToWorldTransform(), posed[2]));
    rig.Frame(0.5);
    CHECK(Near(upper->GetLocalToWorldTransform(), posed[0]));
    CheckChildrenFollow(rig);

    // an ancestor is detached, with edits that were not propagated; the driven node bound again in another graph
    rig.Bind(lower);
    rig.Frame(0.25);
    const dm::daffine3 lowerPosed = rig.Posed(lower);
    Place(parent, Placement{ dm::double3(8.0, 9.0, 10.0), c_OtherTurn, dm::double3(0.5) });
    const dm::daffine3 newParent =
        ComposeTransform(dm::double3(8.0, 9.0, 10.0), c_OtherTurn, dm::double3(0.5));
    auto branch = rig.graph->Detach(parent);
    dm::daffine3 chain = lower->GetLocalToParentTransform();
    for (SceneGraphNode* node = middle; node != parent.Get(); node = node->GetParent())
        chain = chain * node->GetLocalToParentTransform();
    CHECK(Near(chain * newParent, lowerPosed));
    Rig other;
    other.graph->Attach(other.Root(), branch);
    other.Refresh();
    CHECK(Near(lower->GetLocalToWorldTransform(), lowerPosed));
    CHECK(rig.graph->SetSimulationDriven(lower, true) == nvrhi::FE_INVALID_ARGS);   // another graph's node
    other.Bind(lower);
    CHECK(other.IsDriven(lower) && Near(other.Posed(lower), lowerPosed));
}

// A copy of a driven node is authored, at its simulated pose; the original stays driven. In the same graph, in another
// one, and of a subgraph with driven nodes inside.
void test_driver_copy()
{
    Rig rig, other;
    auto domain = rig.NewDomain();
    rig.AddDomain(domain);
    Placement place;
    place.translation = dm::double3(1.0, 2.0, 3.0);
    place.rotation = c_Turn;
    place.scaling = dm::double3(1.5);
    SceneGraphNode* parent = rig.Add(rig.Root(), &place);
    SceneGraphNode* upper = rig.Add(parent, nullptr, true);
    SceneGraphNode* middle = rig.AddAt(upper, 0.0, 2.0, 0.0);
    SceneGraphNode* lower = rig.AddAt(middle, 3.0, 3.0, 3.0);
    rig.Refresh();
    rig.Bind(upper);
    rig.Bind(lower);
    CHECK(rig.world->SetEnabled(rig.EntityOf(upper), FindComponent<SimTeleport>(rig.world), true) == nvrhi::FS_OK);
    for (int i = 0; i < 4; ++i)
        rig.Frame(0.25 * 1.3);
    const dm::daffine3 posedUpper = rig.Posed(upper), posedLower = rig.Posed(lower);
    CHECK(!Near(posedUpper, upper->GetLocalToWorldTransform(), 1e-3));
    domain->events.clear();

    auto check = [&](Rig& r, SceneGraphNode* copy, SceneGraphNode* under, const char* what)
    {
        CHECK(copy && copy != upper && copy->GetParent() == under);
        SceneGraphNode* copyLower = copy->GetFirstChild()->GetFirstChild();
        for (SceneGraphNode* n : { copy, copyLower })
        {
            CHECK(!r.IsDriven(n) && !r.Get<SimPose>(n) && !r.Get<SimBodyRef>(n) && r.Get<LocalTransform>(n));
            CHECK(!r.world->HasComponent(r.EntityOf(n), FindComponent<SimTeleport>(r.world)));
        }
        r.Refresh();
        if (under == parent)
            CHECK(Near(copy->GetLocalToWorldTransform(), posedUpper));
        CHECK(Near(copy->GetLocalToParentTransform() * parent->GetLocalToWorldTransform(), posedUpper));
        CHECK(Near(copyLower->GetLocalToWorldTransform() * inverse(copy->GetLocalToWorldTransform()),
                   posedLower * inverse(posedUpper)));
        CHECK(copy->GetLeaf() && copy->GetLeaf() != upper->GetLeaf());
        (void)what;
    };
    CHECK(rig.graph->HasStaleWorldTransforms() == false);
    nvrhi::AutoPtr<SceneGraphNode> copy = rig.graph->Attach(parent, upper);
    CHECK(rig.graph->HasStaleWorldTransforms() && rig.graph->HasPendingTransformChanges());
    check(rig, copy, parent, "same graph");
    copy = other.graph->Attach(other.Root(), upper);
    check(other, copy, other.Root(), "another graph");
    CHECK(rig.IsDriven(upper) && rig.IsDriven(lower) && domain->events.empty());
    CHECK(Near(rig.Posed(upper), posedUpper, 0.0));

    // of a single driven node, without a transform of its own: the copy has one, which is pending
    SceneGraphNode* bare = other.Add(other.Root());
    other.Bind(bare);
    other.Refresh();
    other.Refresh();
    CHECK(!other.graph->HasPendingTransformChanges());
    copy = other.graph->Attach(other.Root(), bare);
    CHECK(other.graph->HasPendingTransformChanges() && other.Get<LocalTransform>(copy) && !other.IsDriven(copy));

    // a copy of a copy, and a copy of an authored node with driven ones inside, under itself
    copy = rig.graph->Attach(parent, parent);
    CHECK(copy->GetParent() == parent && !rig.IsDriven(copy->GetFirstChild()));
    CHECK(domain->events.empty());
    rig.Refresh();
    CHECK(Near(copy->GetFirstChild()->GetLocalToParentTransform(), upper->GetLocalToParentTransform(), 10.0));
    CHECK(Near(copy->GetFirstChild()->GetLocalToParentTransform() * parent->GetLocalToWorldTransform(), posedUpper));
    CheckChildrenFollow(rig);
}

// SetRootNode: a driven root and a driven node that becomes the root; SetLeaf on a driven node.
void test_driver_set_root_and_leaf()
{
    Rig rig;
    auto domain = rig.NewDomain();
    const SimulationDomainId id = rig.AddDomain(domain);
    SceneGraphNode* inner = rig.AddAt(rig.Root(), 1.0, 2.0, 3.0);
    SceneGraphNode* child = rig.AddAt(inner, 0.0, 0.0, 1.0);
    rig.Refresh();
    rig.Bind(rig.Root());
    rig.Bind(inner);
    for (int i = 0; i < 3; ++i)
        rig.Frame(0.25 * 1.3);
    const dm::daffine3 rootPosed = rig.Posed(rig.Root()), innerPosed = rig.Posed(inner);
    domain->events.clear();
    nvrhi::AutoPtr<SceneGraphNode> keep = inner;
    auto oldRoot = rig.graph->SetRootNode(MakeNode("new root"));
    CHECK(domain->events.size() == 2 && oldRoot->GetGraph() == nullptr && inner->GetGraph() == nullptr);
    CHECK(Near(oldRoot->GetLocalToParentTransform(), rootPosed, 1e-14));
    CHECK(Near(inner->GetLocalToParentTransform() * rootPosed, innerPosed));
    rig.graph->Attach(rig.Root(), oldRoot);
    rig.Refresh();
    CHECK(Near(inner->GetLocalToWorldTransform(), innerPosed) && !rig.IsDriven(inner));

    // a driven node inside the graph becomes the root: it leaves with the old root, and is attached authored
    rig.Bind(inner);
    rig.Frame(0.25);
    const dm::daffine3 posed = rig.Posed(inner);
    const dm::daffine3 above = oldRoot->GetLocalToWorldTransform();
    rig.graph->SetRootNode(inner);
    CHECK(rig.Root() == inner && !rig.IsDriven(inner) && inner->GetFirstChild() == child);
    rig.Refresh();
    CHECK(Near(inner->GetLocalToWorldTransform() * above, posed));
    CheckChildrenFollow(rig);

    // SetLeaf: the row moves with the poses; the domain, which does not look at leaves, hears nothing
    rig.Bind(inner);
    rig.Frame(0.25 * 1.5);
    domain->events.clear();
    const SimPose pose = *rig.Get<SimPose>(inner);
    const SimPosePrevious previous = *rig.Get<SimPosePrevious>(inner);
    const SimBodyRef body = *rig.Get<SimBodyRef>(inner);
    inner->SetLeaf(rig.factory->CreateMeshInstance(rig.mesh));
    CHECK(rig.IsDriven(inner) && domain->events.empty() && rig.graph->GetMeshInstances().size() == 1);
    CHECK(SameBits(&pose, rig.Get<SimPose>(inner), sizeof(pose)));
    CHECK(SameBits(&previous, rig.Get<SimPosePrevious>(inner), sizeof(previous)));
    CHECK(SameBits(&body, rig.Get<SimBodyRef>(inner), sizeof(body)) && body.domain == uint32_t(id));
    CHECK(!rig.world->IsEnabled(rig.EntityOf(inner), FindComponent<SimTeleport>(rig.world)));
    const dm::daffine3 world = inner->GetLocalToWorldTransform();
    rig.Refresh();
    CHECK(SameBits(inner->GetLocalToWorldTransform(), world));
    const dm::box3 expected = rig.mesh->objectSpaceBounds * dm::affine3(world);
    const dm::box3 box = rig.Get<WorldBounds>(inner)->bounds;
    CHECK(all(box.m_mins == expected.m_mins) && all(box.m_maxs == expected.m_maxs));
    rig.Frame(0.25);   // the leaf's bounds follow the simulation
    const dm::box3 moved = rig.mesh->objectSpaceBounds * dm::affine3(inner->GetLocalToWorldTransform());
    CHECK(!all(moved.m_mins == expected.m_mins) && all(rig.Get<WorldBounds>(inner)->bounds.m_mins == moved.m_mins));
    CHECK(all(rig.Root()->GetGlobalBoundingBox().m_mins == moved.m_mins));
    inner->SetLeaf(nullptr);
    CHECK(rig.IsDriven(inner) && !inner->GetLeaf() && domain->events.empty());
    rig.Frame(0.25);
    CHECK(SameBits(inner->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(inner),
          *rig.Get<SimPose>(inner), *rig.Get<SimScale>(inner), rig.Alpha(id))));
}

// Destruction in every form: the domain hears of it and reuses its index; removing the domain leaves the nodes driven,
// at their last poses; teardown keeps referenced nodes at their simulated poses and calls nobody.
void test_driver_destroy()
{
    Rig rig;
    auto domain = rig.NewDomain();
    SimulationDomainId id = rig.AddDomain(domain);
    std::vector<SceneGraphNode*> nodes;
    for (int i = 0; i < 10; ++i)
    {
        nodes.push_back(rig.AddAt(rig.Root(), double(i), 30.0, 0.0, true));
        rig.Bind(nodes.back());
    }
    rig.Refresh();
    rig.Frame(0.25 * 1.5);
    const Entity gone = rig.EntityOf(nodes[3]);
    const uint32_t index = rig.Get<SimBodyRef>(nodes[3])->index;
    domain->events.clear();
    rig.graph->Detach(nodes[3]);   // nobody keeps it
    CHECK(!rig.world->IsAlive(gone) && domain->events.size() == 1 && domain->events[0].first == gone);
    CHECK(domain->freeIndices.size() == 1 && domain->freeIndices[0] == index);
    nodes[3] = rig.AddAt(rig.Root(), 3.0, 30.0, 0.0, true);
    rig.Bind(nodes[3]);
    CHECK(rig.Get<SimBodyRef>(nodes[3])->index == index);
    rig.Frame(0.25 * 1.25);
    for (SceneGraphNode* node : nodes)
    {
        const BallDomain::Ball* ball = domain->Find(rig.EntityOf(node));
        CHECK(ball && SameBits(&ball->pose, rig.Get<SimPose>(node), sizeof(SimPose)));
        CHECK(SameBits(&ball->previous, rig.Get<SimPosePrevious>(node), sizeof(SimPosePrevious)));
        CHECK(SameBits(node->GetLocalToWorldTransform(), DirectWorld(ball->previous, ball->pose,
              *rig.Get<SimScale>(node), rig.Alpha(id))));
    }

    // without its domain a node stays driven and rests at its last pose, whether or not its SimBodyRef still names it
    CHECK(rig.Alpha(id) > 0.0);
    domain->clearsBodyRefs = false;
    CHECK(rig.graph->GetScheduler()->RemoveSimulationDomain(id) == nvrhi::FS_OK);
    CHECK(rig.IsDriven(nodes[0]) && rig.Get<SimBodyRef>(nodes[0])->domain == uint32_t(id));
    rig.Frame(0.25 * 1.5);
    for (SceneGraphNode* node : nodes)
        CHECK(SameBits(node->GetLocalToWorldTransform(), rig.Posed(node)));
    // and another domain takes them over from there
    auto next = rig.NewDomain("next", 0.125);
    id = rig.AddDomain(next);
    CHECK(next->events.size() == nodes.size());
    rig.Frame(0.125 * 1.5);
    CHECK(SameBits(nodes[0]->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(nodes[0]),
          *rig.Get<SimPose>(nodes[0]), *rig.Get<SimScale>(nodes[0]), rig.Alpha(id))));

    // teardown
    nvrhi::AutoPtr<SceneGraphNode> kept = nodes[5];
    const dm::daffine3 posed = rig.Posed(kept), world = kept->GetLocalToWorldTransform();
    next->events.clear();
    rig.graph.Reset();
    CHECK(next->events.empty() && kept->GetGraph() == nullptr);
    CHECK(Near(kept->GetLocalToParentTransform(), posed) && SameBits(kept->GetLocalToWorldTransform(), world));
    CHECK(!Near(posed, world, 1e-3));

    // with the world held: the rows live on, the nodes are in no graph
    Rig held;
    nvrhi::AutoPtr<IWorld> heldWorld = held.world;
    nvrhi::AutoPtr<SceneGraphNode> outlives = held.AddAt(held.Root(), 1.0, 2.0, 3.0);
    held.Bind(outlives);
    held.SetPose(outlives, dm::double3(7.0, 8.0, 9.0), c_Turn);
    const Entity entity = held.EntityOf(outlives);
    held.graph.Reset();
    CHECK(outlives->GetGraph() == nullptr && Near(outlives->GetTranslation(), dm::double3(7.0, 8.0, 9.0), 1e-14));
    CHECK(heldWorld->IsAlive(entity));
}

// The local getters of a driven node: from the world transform and the parent's, on the call; nothing is stored.
void test_driver_getters()
{
    Rig rig;
    Placement place;
    place.translation = dm::double3(1.0, 2.0, 3.0);
    place.rotation = c_Turn;
    place.scaling = dm::double3(2.0);
    SceneGraphNode* parent = rig.Add(rig.Root(), &place);
    SceneGraphNode* node = rig.AddAt(parent, 4.0, 5.0, 6.0);
    SceneGraphNode* bare = rig.Add(node);   // authored, below it, without a transform
    rig.Refresh();
    rig.Bind(node);
    auto check = [&](double tolerance)
    {
        const dm::daffine3 world = node->GetLocalToWorldTransform();
        const dm::daffine3 parentWorld = parent->GetLocalToWorldTransform();
        CHECK(Near(node->GetLocalToParentTransform() * parentWorld, world, tolerance));
        CHECK(Near(ComposeTransform(node->GetTranslation(), node->GetRotation(), node->GetScaling()) * parentWorld, world,
                   tolerance));
    };
    check(1e-14);
    CHECK(Near(node->GetTranslation(), dm::double3(4.0, 5.0, 6.0), 1e-14));
    CHECK(Near(node->GetScaling(), dm::double3(1.0), 1e-14) && Near(node->GetRotation(), dm::dquat::identity(), 1e-14));

    // after a move of the simulation, and of the parent
    rig.SetPose(node, dm::double3(-3.0, 9.0, 2.0), c_OtherTurn);
    rig.Simulate(0.0);
    check(1e-13);
    rig.Refresh();
    check(1e-13);
    CHECK(Near(node->GetScaling(), dm::double3(1.0), 1e-13));
    Place(parent, Placement{ dm::double3(-1.0, 0.0, 4.0), c_OtherTurn, dm::double3(0.25) });
    rig.Refresh();
    check(1e-13);
    CHECK(Near(node->GetScaling(), dm::double3(8.0), 1e-13));
    CHECK(Near(node->GetRotation(), dm::dquat::identity(), 1e-13));
    // small rotations keep their digits
    rig.SetPose(node, dm::double3(0.0), Unit(1.0, 1e-9, -2e-9, 3e-9));
    Place(parent, Placement{});
    rig.Simulate(0.0);
    rig.Refresh();
    CHECK(Near(node->GetRotation().x * 1e9, 1.0, 1e-6) && Near(node->GetRotation().z * 1e9, 3.0, 1e-6));

    // a mirrored parent: a negative scale; a stretched one: its parts cannot shear, the matrix is exact
    Place(parent, Placement{ dm::double3(1.0), c_Turn, dm::double3(-2.0, 2.0, 2.0) });
    rig.SetPose(node, dm::double3(4.0, 5.0, 6.0), c_OtherTurn);
    rig.Simulate(0.0);
    rig.Refresh();
    check(1e-13);
    CHECK(node->GetScaling().x < 0.0);
    Place(parent, Placement{ dm::double3(1.0), c_Turn, dm::double3(1.0, 3.0, 1.0) });
    rig.Refresh();
    CHECK(Near(node->GetLocalToParentTransform() * parent->GetLocalToWorldTransform(), node->GetLocalToWorldTransform(),
               1e-13));
    CHECK(Near(node->GetTranslation(), node->GetLocalToParentTransform().m_translation, 1e-13));
    // a small parent has an inverse, a flat one has none: then the world transform
    Place(parent, Placement{ dm::double3(1.0), c_Turn, dm::double3(1e-8) });
    rig.Refresh();
    CHECK(Near(node->GetScaling() * 1e-8, dm::double3(2.0), 1e-12));
    Place(parent, Placement{ dm::double3(1.0), c_Turn, dm::double3(1.0, 0.0, 1.0) });
    rig.Refresh();
    CHECK(SameBits(node->GetLocalToParentTransform(), node->GetLocalToWorldTransform()));
    CHECK(Near(node->GetTranslation(), dm::double3(4.0, 5.0, 6.0), 0.0));

    // the inverse exists, but the quotient of the translation overflows: the same
    Place(parent, Placement{ dm::double3(0.0), dm::dquat::identity(), dm::double3(1e-5) });
    rig.SetPose(node, dm::double3(1e305, 0.0, 0.0), dm::dquat::identity());
    rig.Simulate(0.0);
    rig.Refresh();
    CHECK(SameBits(node->GetLocalToParentTransform(), node->GetLocalToWorldTransform()));
    CHECK(node->GetTranslation().x == 1e305);
    // or that of the axes
    Place(parent, Placement{ dm::double3(0.0), dm::dquat::identity(), dm::double3(1e-10) });
    rig.SetPose(node, dm::double3(0.0), dm::dquat::identity());
    const SimScale scale = *rig.Get<SimScale>(node);
    *rig.Edit<SimScale>(node) = SimScale{ dm::double3(1e300) };
    rig.Simulate(0.0);
    rig.Refresh();
    CHECK(SameBits(node->GetLocalToParentTransform(), node->GetLocalToWorldTransform()));
    CHECK(TakeWarnings() == 0);   // the getters are silent

    // an authored node without a transform has the identity, also while the driven node above it has moved on
    Place(parent, Placement{ dm::double3(1.0), c_Turn, dm::double3(1.0) });
    rig.SetPose(node, dm::double3(4.0, 5.0, 6.0), c_OtherTurn);
    *rig.Edit<SimScale>(node) = scale;
    rig.Simulate(0.0);
    CHECK(!Near(bare->GetLocalToWorldTransform(), node->GetLocalToWorldTransform(), 1e-3));
    CHECK(Near(bare->GetTranslation(), dm::double3(0.0), 0.0) && Near(bare->GetScaling(), dm::double3(1.0), 0.0));
    CHECK(Near(bare->GetRotation(), dm::dquat::identity(), 0.0));
    CHECK(SameBits(bare->GetLocalToParentTransform(), dm::daffine3::identity()));
    rig.Refresh();

    // nothing is stored: the same rows before and after, the kept LocalTransform included
    const LocalTransform kept = *rig.Get<LocalTransform>(node);
    const uint64_t hash = rig.world->ComputeHash();
    (void)node->GetTranslation(), (void)node->GetRotation(), (void)node->GetScaling();
    (void)node->GetLocalToParentTransform();
    CHECK(rig.world->ComputeHash() == hash);
    CHECK(kept.translation.x == 4.0 && SameBits(&kept, rig.Get<LocalTransform>(node), sizeof(kept)));

    // the authored node below has none: the identity, as ever
    CHECK(Near(bare->GetTranslation(), dm::double3(0.0), 0.0) && Near(bare->GetScaling(), dm::double3(1.0), 0.0));
    CHECK(SameBits(bare->GetLocalToParentTransform(), dm::daffine3::identity()));
    // a driven root
    rig.Bind(rig.Root());
    rig.SetPose(rig.Root(), dm::double3(3.0, 2.0, 1.0), c_Turn);
    rig.Simulate(0.0);
    CHECK(Near(rig.Root()->GetTranslation(), dm::double3(3.0, 2.0, 1.0), 1e-14));
    CHECK(Near(rig.Root()->GetRotation(), c_Turn, 1e-14));
    CHECK(SameBits(rig.Root()->GetLocalToParentTransform(), rig.Root()->GetLocalToWorldTransform()));
}

// What is refused, and that nothing changes then.
void test_driver_refusals()
{
    Rig rig;
    auto domain = rig.NewDomain();
    rig.AddDomain(domain);
    SceneGraphNode* parent = rig.Add(rig.Root());
    SceneGraphNode* node = rig.AddAt(parent, 1.0, 0.0, 0.0);
    rig.Refresh();
    auto refused = [&](const Placement& parentPlace, const Placement& place)
    {
        Place(parent, parentPlace);
        Place(node, place);
        for (bool stale : { true, false })
        {
            if (!stale)
                rig.Refresh();
            const dm::daffine3 world = node->GetLocalToWorldTransform();
            const uint64_t structural = rig.world->GetStructuralVersion();
            CHECK(rig.graph->SetSimulationDriven(node, true) == nvrhi::FE_INVALID_ARGS);
            CHECK(TakeWarnings() == 1);
            CHECK(rig.world->GetStructuralVersion() == structural && !rig.IsDriven(node) && domain->events.empty());
            CHECK(SameBits(node->GetLocalToWorldTransform(), world) && rig.graph->HasStaleWorldTransforms() == stale);
        }
    };
    auto accepted = [&](const Placement& parentPlace, const Placement& place)
    {
        Place(parent, parentPlace);
        Place(node, place);
        rig.Refresh();
        const dm::daffine3 world = node->GetLocalToWorldTransform();
        CHECK(rig.graph->SetSimulationDriven(node, true) == nvrhi::FS_OK && TakeWarnings() == 0);
        CHECK(Near(rig.Posed(node), world, 1e-6));
        rig.Unbind(node);
        domain->events.clear();
    };
    Placement none, turned, stretched, mirrored, flat, broken;
    turned.rotation = c_Turn;
    stretched.scaling = dm::double3(1.0, 2.0, 1.0);
    mirrored.scaling = dm::double3(1.0, -1.0, 1.0);
    flat.scaling = dm::double3(1.0, 1.0, 0.0);
    broken.translation.y = std::numeric_limits<double>::quiet_NaN();
    refused(stretched, turned);                              // shear
    refused(none, mirrored);
    refused(mirrored, turned);
    refused(none, flat);
    refused(flat, none);
    refused(none, broken);
    broken.translation.y = 0.0;
    broken.scaling.x = std::numeric_limits<double>::infinity();
    refused(none, broken);
    broken.scaling.x = std::numeric_limits<double>::quiet_NaN();
    refused(none, broken);
    broken.scaling.x = 1e200;                                // finite, but its square is not
    refused(none, broken);
    // a stretch by 1 + e shears by e * sin(2 angle) / 2 at most; the limit is 1e-5
    Placement slight = none, quarter = none;
    quarter.rotation = dm::rotationQuat(dm::double3(0.0, 0.0, 1.0), 0.25 * 3.14159265358979);
    slight.scaling.x = 1.0 + 1e-4;
    refused(slight, quarter);
    slight.scaling.x = 1.0 + 1e-6;
    accepted(slight, quarter);
    accepted(stretched, none);
    accepted(none, stretched);
    Placement twice = turned;
    twice.scaling = dm::double3(-1.5, -1.5, 1.5);            // mirrored twice: a rotation
    accepted(none, twice);
    accepted(mirrored, mirrored);
    Placement small;
    small.scaling = dm::double3(1e-12);
    accepted(none, small);
    Place(parent, none);
    Place(node, none);
    rig.Refresh();
    rig.Refresh();

    // the wrong node
    CHECK(rig.graph->SetSimulationDriven(nullptr, true) == nvrhi::FE_INVALID_ARGS);
    CHECK(rig.graph->SetSimulationDriven(MakeNode(), true) == nvrhi::FE_INVALID_ARGS);
    Rig other;
    CHECK(rig.graph->SetSimulationDriven(other.Root(), true) == nvrhi::FE_INVALID_ARGS);
    CHECK(!other.IsDriven(other.Root()));

    // the wrong time: inside a phase, and in a callback of the domain
    nvrhi::FRESULT inPhase = nvrhi::FS_OK;
    SystemDesc desc{};
    desc.name = "test.bind";
    desc.phase = Phase::EndFrame;
    desc.flags = SystemFlags::MainThread;
    nvrhi::AutoPtr<ISystem> system;
    CHECK(CreateLambdaSystem(desc, [&](ISystemContext*) noexcept
    {
        inPhase = rig.graph->SetSimulationDriven(node, true);
        return nvrhi::FS_OK;
    }, &system) == nvrhi::FS_OK);
    SystemId systemId = SystemId::Invalid;
    CHECK(rig.graph->GetScheduler()->AddSystem(system, systemId) == nvrhi::FS_OK);
    CHECK(rig.graph->GetScheduler()->RunPhase(Phase::EndFrame) == nvrhi::FS_OK);
    CHECK(inPhase == nvrhi::FE_INVALID_ARGS && TakeWarnings() == 1 && !rig.IsDriven(node));
    CHECK(rig.graph->GetScheduler()->RemoveSystem(systemId) == nvrhi::FS_OK);
    nvrhi::FRESULT inCallback = nvrhi::FS_OK;
    domain->onAdded = [&](const EntityBatch&, ISimulationContext*)
    {
        inCallback = rig.graph->SetSimulationDriven(node, true);
    };
    SceneGraphNode* second = rig.AddAt(rig.Root(), 0.0, 0.0, 1.0);
    rig.Bind(second);
    CHECK(inCallback == nvrhi::FE_INVALID_ARGS && TakeWarnings() == 1 && !rig.IsDriven(node) && rig.IsDriven(second));
    domain->onAdded = nullptr;

    // an observer that fails undoes nothing: a warning, in every build (and validation's, in Debug)
    {
        class Failing final : public nvrhi::ObjectImpl<IObserver>
        {
        public:
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(Failing)
            NVRHI_IMPLEMENTS_INTERFACE(IObserver)
            NVRHI_END_INTERFACE_TABLE()
            nvrhi::FRESULT OnEntities(ObserverEvent, const EntityBatch&, ICommandBuffer*) noexcept override
            {
                return nvrhi::FE_UNSUPPORTED;
            }
        };
        nvrhi::AutoPtr<IObserver> failing = MAKE_RC_OBJ_PTR(Failing);
        ObserverId ids[2];
        ObserverDesc observer{};
        observer.debugName = "failing";
        observer.component = FindComponent<SimulationDriven>(rig.world);
        observer.access = Access::Filter;
        observer.event = ObserverEvent::OnAdd;
        CHECK(rig.world->AddObserver(observer, failing, ids[0]) == nvrhi::FS_OK);
        observer.event = ObserverEvent::OnRemove;
        CHECK(rig.world->AddObserver(observer, failing, ids[1]) == nvrhi::FS_OK);
        SceneGraphNode* third = rig.AddAt(rig.Root(), 0.0, 3.0, 1.0);
#ifdef NDEBUG
        constexpr uint32_t c_Expected = 1;
#else
        constexpr uint32_t c_Expected = 2;
#endif
        for (bool bind : { true, false })
        {
            CHECK(rig.graph->SetSimulationDriven(third, bind) == nvrhi::FS_OK);
            CHECK(TakeWarnings() == c_Expected && rig.IsDriven(third) == bind);
        }
        for (ObserverId observerId : ids)
            CHECK(rig.world->RemoveObserver(observerId) == nvrhi::FS_OK);
    }

    // a row that fits no chunk with the poses: nothing changed, both ways of asking
    SceneGraphNode* bulky = rig.AddAt(rig.Root(), 2.0, 0.0, 0.0);
    rig.Refresh();
    CHECK(rig.world->AddComponent(rig.EntityOf(bulky), RegisterComponent<Bulk>(rig.world), nullptr) == nvrhi::FS_OK);
    const uint64_t structural = rig.world->GetStructuralVersion();
    domain->events.clear();
    CHECK(NVRHI_FAILED(rig.graph->SetSimulationDriven(bulky, true)));
#ifdef NDEBUG
    CHECK(TakeWarnings() == 1);
#else
    CHECK(TakeWarnings() == 2);   // and validation's
#endif
    CHECK(!rig.IsDriven(bulky) && rig.world->GetStructuralVersion() == structural && domain->events.empty());
    CHECK(Near(bulky->GetTranslation(), dm::double3(2.0, 0.0, 0.0), 0.0));
    rig.graph->Detach(bulky);

    // the setters of a driven node change nothing (until E6.2b makes them requests), and neither does an animation
    rig.Refresh();
    rig.Refresh();
    const LocalTransform kept = *rig.Get<LocalTransform>(second);
    const SimPose pose = *rig.Get<SimPose>(second);
    const uint64_t localVersion = rig.ColumnVersion(second, FindComponent<LocalTransform>(rig.world));
    rig.world->AdvanceVersion();
    second->SetTranslation(dm::double3(5.0));
    second->SetRotation(c_Turn);
    second->SetScaling(dm::double3(2.0));
    CHECK(TakeWarnings() == 3);
    CHECK(SameBits(&kept, rig.Get<LocalTransform>(second), sizeof(kept)) && kept.translation.z == 1.0);
    CHECK(SameBits(&pose, rig.Get<SimPose>(second), sizeof(pose)));
    CHECK(rig.ColumnVersion(second, FindComponent<LocalTransform>(rig.world)) == localVersion);
    CHECK(!rig.graph->HasPendingTransformChanges() && !rig.graph->HasStaleWorldTransforms());
    SceneGraphNode* bare = rig.Add(rig.Root());
    rig.Bind(bare);
    bare->SetTranslation(dm::double3(5.0));
    CHECK(TakeWarnings() == 1 && !rig.Get<LocalTransform>(bare) && rig.IsDriven(bare));

    // inside a render-read scope: a warning in Debug builds, and the call applies
    rig.graph->BeginRenderRead();
    rig.Bind(node);
    rig.graph->EndRenderRead();
#ifdef NDEBUG
    CHECK(TakeWarnings() == 0);
#else
    CHECK(TakeWarnings() == 1);
#endif
    CHECK(rig.IsDriven(node));
}

// HasStaleWorldTransforms around every call of this item.
void test_driver_stale_flag()
{
    Rig rig;
    auto domain = rig.NewDomain();
    rig.AddDomain(domain);
    SceneGraphNode* node = rig.AddAt(rig.Root(), 1.0, 2.0, 3.0);
    SceneGraphNode* other = rig.AddAt(rig.Root(), 1.0, 2.0, 3.0);
    rig.AddAt(node, 1.0, 0.0, 0.0);
    CHECK(rig.graph->HasStaleWorldTransforms());
    rig.Bind(other);
    CHECK(rig.graph->HasStaleWorldTransforms());   // as it was
    rig.Refresh();
    rig.Refresh();
    CHECK(!rig.graph->HasStaleWorldTransforms() && !rig.graph->HasPendingTransformChanges());
    rig.Bind(node);
    CHECK(!rig.graph->HasStaleWorldTransforms() && !rig.graph->HasPendingTransformChanges() &&
          !rig.graph->HasPendingStructureChanges());
    rig.Simulate(0.5);
    CHECK(!rig.graph->HasStaleWorldTransforms() && !rig.graph->HasPendingTransformChanges());
    rig.Refresh();
    CHECK(!rig.graph->HasPendingTransformChanges());   // a move of the simulation is no edit
    node->SetTranslation(dm::double3(0.0));
    CHECK(TakeWarnings() == 1 && !rig.graph->HasStaleWorldTransforms());
    CHECK(rig.world->SetEnabled(rig.EntityOf(node), FindComponent<SimTeleport>(rig.world), true) == nvrhi::FS_OK);
    CHECK(!rig.graph->HasStaleWorldTransforms());      // a request is none either
    rig.Unbind(node);
    CHECK(rig.graph->HasStaleWorldTransforms() && rig.graph->HasPendingTransformChanges() &&
          !rig.graph->HasPendingStructureChanges());
    // any run of propagation clears it
    CHECK(rig.graph->GetScheduler()->RunPhase(Phase::TransformPropagation) == nvrhi::FS_OK);
    CHECK(!rig.graph->HasStaleWorldTransforms() && rig.graph->HasPendingTransformChanges());
    rig.Refresh();
    rig.Refresh();
    CHECK(!rig.graph->HasPendingTransformChanges());
    rig.graph->Detach(other);
    CHECK(rig.graph->HasStaleWorldTransforms());
}

// A joint that a simulation drives marks its skinned instance when it moves, as an authored one does; one that rests
// does not.
void test_driver_joints()
{
    Rig rig;
    auto skinned = rig.factory->CreateSkinnedMeshInstance(rig.factory, rig.mesh);
    rig.graph->AttachLeafNode(rig.Root(), skinned);
    SceneGraphNode* joints[2];
    for (int i = 0; i < 2; ++i)
    {
        auto node = MakeNode("joint");
        if (i == 0)
            node->SetTranslation(dm::double3(1.0));   // the other one has no transform
        node->SetLeaf(MAKE_RC_OBJ_PTR(SkinnedMeshReference, skinned.Get()));
        joints[i] = rig.graph->Attach(rig.Root(), node);
    }
    rig.graph->Refresh(1);
    for (SceneGraphNode* joint : joints)
        rig.Bind(joint);
    rig.graph->Refresh(2);
    rig.Simulate(0.0);   // the first interpolation writes every driven row
    rig.graph->Refresh(3);
    uint32_t frame = 10;
    for (SceneGraphNode* joint : joints)
    {
        rig.Simulate(0.0);
        rig.graph->Refresh(frame);
        const uint32_t before = skinned->GetLastUpdateFrameIndex();
        CHECK(before < frame);                        // at rest
        rig.SetPose(joint, dm::double3(double(frame)), c_Turn);
        rig.Simulate(0.0);
        rig.graph->Refresh(frame + 1);
        CHECK(skinned->GetLastUpdateFrameIndex() == frame + 1);
        rig.Simulate(0.0);
        rig.graph->Refresh(frame + 2);
        CHECK(skinned->GetLastUpdateFrameIndex() == frame + 1);
        frame += 10;
    }
    // unbound: authored again, and its setters mark
    rig.Unbind(joints[1]);
    rig.graph->Refresh(frame);
    rig.graph->Refresh(frame + 1);
    joints[1]->SetTranslation(dm::double3(3.0));
    rig.graph->Refresh(frame + 2);
    CHECK(skinned->GetLastUpdateFrameIndex() == frame + 2);
}

// SetJobSystem makes a new scheduler: the phase and the system exist once in each, interpolation goes on.
void test_driver_job_system()
{
    Rig rig;
    auto domain = rig.NewDomain();
    rig.AddDomain(domain);
    std::vector<SceneGraphNode*> nodes;
    for (int i = 0; i < 300; ++i)
    {
        nodes.push_back(rig.AddAt(rig.Root(), double(i), 20.0, 0.0));
        rig.AddAt(nodes.back(), 1.0, 0.0, 0.0);
        rig.Bind(nodes.back());
    }
    rig.Refresh();
    rig.Frame(0.25 * 1.5);
    for (uint32_t workers : { 1u, 4u })
    {
        rig.jobs = MakeJobs(workers);
        nvrhi::AutoPtr<IScheduler> old = rig.graph->GetScheduler();
        rig.graph->SetJobSystem(rig.jobs);
        IScheduler* scheduler = rig.graph->GetScheduler();
        CHECK(scheduler != old.Get());
        const Phase phase = rig.Store().GetInterpolationPhase();
        CHECK(uint32_t(phase) == uint32_t(Phase::Count));   // the scheduler's first named phase
        CHECK(scheduler->Build() == nvrhi::FS_OK && ecs::GetScheduleOrder(scheduler, phase, nullptr, 0) == 1);
        Phase again = Phase::Count;
        CHECK(NVRHI_FAILED(scheduler->AddPhase("SimulationInterpolation", Phase::PostSimulation, again)));
        CHECK(TakeWarnings() == 1);   // its name is taken: the phase is there, once
        // the new scheduler knows no domain: the nodes rest at their last poses, also those it never wrote
        rig.Frame(0.0);
        for (SceneGraphNode* node : nodes)
            CHECK(SameBits(node->GetLocalToWorldTransform(), rig.Posed(node)));
        CheckChildrenFollow(rig);
        domain = rig.NewDomain();
        const SimulationDomainId id = rig.AddDomain(domain);
        CHECK(domain->events.size() == nodes.size() && scheduler->GetAmbiguityCount() == 0);
        for (int i = 0; i < 3; ++i)
            rig.Frame(0.25 * 1.3);
        CHECK(scheduler->GetAmbiguityCount() == 0);
        for (SceneGraphNode* node : nodes)
            CHECK(SameBits(node->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(node),
                  *rig.Get<SimPose>(node), *rig.Get<SimScale>(node), rig.Alpha(id))));
        CheckChildrenFollow(rig);
    }
}

// Propagation in scenes without driven nodes does the work it did before this item: the propagated rows and chunks and
// the visited rows of the last bucket, per frame, over the scenes of the E0 trace. The numbers are those of donut
// 21719ae (E6.1).
void test_driver_propagated_rows()
{
    struct Spec
    {
        const char* name;
        uint64_t seed;
        const Profile* profile;
        void (*build)(Scene&, uint32_t);
        uint32_t size;
        uint64_t rows, chunks, deepRows, hash;
    };
    const Spec specs[] = {
        { "flat_small", 0x1001, &c_SmallProfile, BuildFlat, 12, 1558, 1075, 0, 0x153f57e05ef68cc5ull },
        { "chain_small", 0x1002, &c_SmallProfile, BuildChain, 10, 1565, 1117, 0, 0x6a990f9905aeeb17ull },
        { "tree_small", 0x1003, &c_SmallProfile, BuildTree, 20, 1931, 1367, 0, 0xe7c0bffec79db5ebull },
        { "skinned_small", 0x1004, &c_SmallProfile, BuildCrowd, 1, 1133, 766, 0, 0x650932397f30c017ull },
        { "mixed_small", 0x1005, &c_SmallProfile, BuildMixed, 24, 2061, 1093, 0, 0x8bbafaa2ca5d31e3ull },
        { "flat", 0x2001, &c_LargeProfile, BuildFlat, 4000, 141023, 1641, 0, 0x7b15ce0fb8db0dabull },
        { "chain", 0x2002, &c_LargeProfile, BuildChain, 300, 13166, 3651, 7036, 0xa7b2bf4ae6e33b17ull },
        { "tree", 0x2003, &c_LargeProfile, BuildTree, 3000, 106062, 12002, 0, 0x64dba0fff91fd416ull },
        { "skinned", 0x2004, &c_LargeProfile, BuildCrowd, 40, 12282, 1320, 0, 0x3d6a1227205861e5ull },
        { "mixed", 0x2005, &c_LargeProfile, BuildMixed, 1200, 26845, 6874, 523, 0x7b0b7c29a947dbcaull },
    };
    for (const Spec& spec : specs)
    {
        Scene s(spec.seed, *spec.profile);
        spec.build(s, spec.size);
        s.nodeLimit = CollectNodes(s.graph).size() * 3 / 2 + 64;
        uint64_t hash = 0xcbf29ce484222325ull;
        auto add = [&](uint64_t value)
        {
            for (int i = 0; i < 8; ++i)
            {
                hash ^= (value >> (8 * i)) & 255;
                hash *= 0x100000001b3ull;
            }
        };
        for (uint32_t frame = 0; frame < 40; ++frame)
        {
            if (frame > 0)
                Step(s, frame);
            s.graph->Refresh(frame);
            const SceneGraphStore::Counters& c = StoreOf(s.graph).GetCounters();
            add(c.propagatedRows);
            add(c.propagatedChunks);
            add(c.deepRows);
        }
        const SceneGraphStore::Counters& c = StoreOf(s.graph).GetCounters();
        if (c.propagatedRows != spec.rows || c.propagatedChunks != spec.chunks || c.deepRows != spec.deepRows ||
            hash != spec.hash)
        {
            std::fprintf(stderr, "%s: %llu rows, %llu chunks, %llu deep rows, hash %016llx\n", spec.name,
                         (unsigned long long)c.propagatedRows, (unsigned long long)c.propagatedChunks,
                         (unsigned long long)c.deepRows, (unsigned long long)hash);
            CHECK(false);
        }
        CHECK(c.interpolatedChunks == 0);
    }
}

// ------------------------------------------------------------------------------------------- interpolation

// Poses equal the engine's, and the interpolation equals the formula bit for bit, for frames of no, a part of, one, one
// and more steps and a hitch beyond the limit.
void test_interpolation_formula()
{
    for (uint32_t workers : { 1u, 8u })
    {
        Rig rig(workers);
        auto domain = rig.NewDomain("balls", 1.0 / 64.0, 4);
        const SimulationDomainId id = rig.AddDomain(domain);
        Rng rng(3);
        std::vector<SceneGraphNode*> nodes;
        for (int i = 0; i < 700; ++i)   // several chunks, in two archetypes
        {
            Placement p = RandomPlacement(rng);
            p.scaling = RandomVector(rng, 0.5, 2.0);
            nodes.push_back(rig.Add(rig.Root(), &p, i % 3 == 0));
            rig.Bind(nodes.back());
        }
        rig.Refresh();
        const double dt = domain->dt;
        std::set<double> alphas;
        uint32_t dropped = 0;
        for (double delta : { 0.0, 0.3 * dt, 1.7 * dt, dt, 0.0, 10.0 * dt, 0.1 * dt, 0.9 * dt, 2.0 * dt, 0.5 * dt,
                              3.25 * dt })
        {
            const uint32_t before = domain->steps;
            std::vector<BallDomain::Ball> expected = domain->balls;
            rig.SetTime(delta);
            const uint32_t planned = GetResource<FixedStepState>(rig.world)->domains[uint32_t(id)].stepsPlanned;
            rig.RunTo(rig.Store().GetInterpolationPhase());
            CHECK(domain->steps - before == planned && planned <= 4);
            auto stepped = rig.NewDomain("reference", dt);
            stepped->balls = expected;
            for (uint32_t s = 0; s < planned; ++s)
                CHECK(stepped->Step(dt, nullptr) == nvrhi::FS_OK);
            const double alpha = rig.Alpha(id);
            CHECK(alpha >= 0.0 && alpha < 1.0);
            alphas.insert(alpha);
            dropped += GetResource<FixedStepState>(rig.world)->domains[uint32_t(id)].stepsDropped;
            for (SceneGraphNode* node : nodes)
            {
                const SimBodyRef body = *rig.Get<SimBodyRef>(node);
                CHECK(body.domain == uint32_t(id));
                const BallDomain::Ball& ball = stepped->balls[body.index];
                CHECK(ball.entity == rig.EntityOf(node));
                CHECK(SameBits(rig.Get<SimPose>(node), &ball.pose, sizeof(SimPose)));
                CHECK(SameBits(rig.Get<SimPosePrevious>(node), &ball.previous, sizeof(SimPosePrevious)));
                const dm::daffine3 world = node->GetLocalToWorldTransform();
                CHECK(SameBits(world, DirectWorld(ball.previous, ball.pose, *rig.Get<SimScale>(node), alpha)));
            }
            rig.Refresh();
        }
        CHECK(alphas.size() >= 5 && alphas.count(0.0) && dropped == 6);   // the hitch: 10 steps, of which 4 are taken
    }
}

// The formula on hand-made poses: the ends exactly, also of rotations that are no unit; the short side; the degenerate
// rotations; a scale that mirrors (a teleport may give one).
void test_interpolation_poses()
{
    Rig rig;
    auto domain = rig.NewDomain("rest", 1.0, 1);
    const SimulationDomainId id = rig.AddDomain(domain);
    Rng rng(8);
    std::vector<SceneGraphNode*> nodes;
    struct Pair { SimPosePrevious previous; SimPose pose; SimScale scale; };
    std::vector<Pair> pairs;
    const dm::dquat q = c_Turn;
    const dm::dquat rotations[][2] = {
        { q, q }, { q, -q }, { q, c_OtherTurn }, { -q, c_OtherTurn }, { q, -c_OtherTurn },
        { q * 2.0, c_OtherTurn * 0.5 },                                                    // no units
        { dm::dquat(1.0, 0.0, 0.0, 0.0), dm::dquat(0.0, 1.0, 0.0, 0.0) },                  // half a turn apart: dot is 0
        { dm::dquat(0.0, 0.0, 0.0, 0.0), dm::dquat(0.0, 0.0, 0.0, 0.0) },                  // none at all
        { dm::dquat(0.0, 0.0, 0.0, 0.0), q }, { Unit(1.0, 1e-9, 0.0, 0.0), Unit(1.0, 0.0, 1e-9, 0.0) },
    };
    for (const auto& r : rotations)
    {
        Pair pair{ { RandomVector(rng, -9.0, 9.0), r[0] }, { RandomVector(rng, -9.0, 9.0), r[1] },
                   { RandomVector(rng, -2.0, 2.0) } };
        pairs.push_back(pair);
        nodes.push_back(rig.AddAt(rig.Root(), 0.0, 0.0, 0.0));
        rig.Bind(nodes.back());
    }
    pairs[0].previous.translation = dm::double3(-0.0, 1e300, -1e-300);
    rig.Refresh();
    rig.Simulate(0.0);
    for (double delta : { 0.0, 0.25, 0.125, 0.625 - 1e-16, 0.0 })   // no whole step: the domain does not scatter
    {
        rig.world->AdvanceVersion();
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            *rig.Edit<SimPosePrevious>(nodes[i]) = pairs[i].previous;
            *rig.Edit<SimPose>(nodes[i]) = pairs[i].pose;
            *rig.Edit<SimScale>(nodes[i]) = pairs[i].scale;
        }
        rig.Simulate(delta);
        const double alpha = rig.Alpha(id);
        CHECK(domain->steps == 0 && alpha < 1.0);
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            const dm::daffine3 world = nodes[i]->GetLocalToWorldTransform();
            CHECK(SameBits(world, DirectWorld(pairs[i].previous, pairs[i].pose, pairs[i].scale, alpha)));
            if (alpha == 0.0)
            {
                CHECK(SameBits(&world.m_translation, &pairs[i].previous.translation, sizeof(dm::double3)));
            }
            else if (i < 5)   // a rigid rotation all the way
            {
                const dm::double3& s = pairs[i].scale.scaling;
                CHECK(Near(length(world.m_linear.row0), std::abs(s.x), 1e-13) &&
                      Near(dm::dot(world.m_linear.row0, world.m_linear.row1), 0.0, 1e-12));
            }
        }
        // q and -q are one rotation: it does not move, whatever alpha is
        const dm::daffine3 still = nodes[1]->GetLocalToWorldTransform();
        const dm::daffine3 reference = DirectWorld({ dm::double3(0.0), q }, { dm::double3(0.0), q }, pairs[1].scale, 0.0);
        CHECK(Near(dm::daffine3(still.m_linear, dm::double3(0.0)), reference, 1e-14));
    }
    // the translation is linear, and the short side is the nearer one
    rig.world->AdvanceVersion();
    *rig.Edit<SimPosePrevious>(nodes[0]) = { dm::double3(0.0, 8.0, -4.0), dm::dquat(1.0, 0.0, 0.0, 0.0) };
    const double angle = 0.4;
    *rig.Edit<SimPose>(nodes[0]) = { dm::double3(16.0, 8.0, 4.0),
                                     -dm::dquat(std::cos(0.5 * angle), 0.0, 0.0, std::sin(0.5 * angle)) };
    *rig.Edit<SimScale>(nodes[0]) = { dm::double3(1.0) };
    rig.Simulate(0.0);
    const double alpha = rig.Alpha(id);
    const dm::daffine3 between = nodes[0]->GetLocalToWorldTransform();
    CHECK(Near(between.m_translation, dm::double3(16.0 * alpha, 8.0, -4.0 + 8.0 * alpha), 1e-15));
    const double turned = std::atan2(between.m_linear.row0.y, between.m_linear.row0.x);
    CHECK(turned > 0.0 && turned < angle && Near(turned, alpha * angle, 0.01));
}

// What the interpolation writes: only chunks whose poses, scales or body references changed, or that have rows of a
// domain whose alpha changed; nothing in a still frame. Its chunks' WorldTransform is stamped, the poses are not.
void test_interpolation_changes()
{
    Rig rig;
    const ComponentId heavy = RegisterComponent<HeavyBall>(rig.world);
    const ComponentId light = RegisterComponent<LightBall>(rig.world);
    std::vector<SceneGraphNode*> heavies, lights, free;
    for (int i = 0; i < 400; ++i)
    {
        for (int kind = 0; kind < 3; ++kind)
        {
            SceneGraphNode* node = rig.AddAt(rig.Root(), double(i), double(kind), 0.0);
            if (kind < 2)
            {
                CHECK(rig.world->AddComponent(rig.EntityOf(node), kind ? light : heavy, nullptr) == nvrhi::FS_OK);
            }
            rig.Bind(node);
            (kind == 0 ? heavies : kind == 1 ? lights : free).push_back(node);
        }
    }
    rig.Refresh();
    auto chunksOf = [&](const std::vector<SceneGraphNode*>& nodes)
    {
        std::set<ChunkKey> chunks;
        for (SceneGraphNode* node : nodes)
            chunks.insert(ChunkOf(rig, node));
        return uint64_t(chunks.size());
    };
    const uint64_t heavyChunks = chunksOf(heavies), lightChunks = chunksOf(lights), freeChunks = chunksOf(free);
    CHECK(heavyChunks > 1 && lightChunks > 1 && freeChunks > 1);
    auto written = [&](const std::function<void()>& frame)
    {
        const uint64_t before = rig.InterpolatedChunks();
        frame();
        rig.Refresh();
        return rig.InterpolatedChunks() - before;
    };
    auto heavyDomain = rig.NewDomain("heavy", 0.25, 4, heavy);
    auto lightDomain = rig.NewDomain("light", 0.5, 4, light);
    const SimulationDomainId heavyId = rig.AddDomain(heavyDomain);
    const SimulationDomainId lightId = rig.AddDomain(lightDomain);
    CHECK(heavyDomain->events.size() == heavies.size() && lightDomain->events.size() == lights.size());
    CHECK(written([&]() { rig.Simulate(0.0); }) == heavyChunks + lightChunks + freeChunks);   // the first run
    // both scatters write SimPose, of disjoint rows: the scheduler does not know, and says so (6.4; open point 11)
    CHECK(rig.graph->GetScheduler()->GetAmbiguityCount() == 1);
    (void)TakeWarnings();   // in Debug builds
    CHECK(written([&]() { rig.Simulate(0.0); }) == 0);

    // an alpha that changed: the domain's chunks, not the others'
    const uint64_t freeVersion = rig.ColumnVersion(free[0], FindComponent<WorldTransform>(rig.world));
    const uint64_t poseVersion = rig.ColumnVersion(heavies[0], FindComponent<SimPose>(rig.world));
    const uint64_t worldVersion = rig.ColumnVersion(heavies[0], FindComponent<WorldTransform>(rig.world));
    const uint64_t scatters = heavyDomain->scatters;
    CHECK(written([&]() { rig.Simulate(0.125); }) == heavyChunks + lightChunks);
    CHECK(heavyDomain->scatters == scatters && heavyDomain->steps == 0);   // the scatter contract: no step, no write
    CHECK(rig.ColumnVersion(heavies[0], FindComponent<SimPose>(rig.world)) == poseVersion);
    CHECK(rig.ColumnVersion(heavies[0], FindComponent<WorldTransform>(rig.world)) > worldVersion);
    CHECK(rig.ColumnVersion(free[0], FindComponent<WorldTransform>(rig.world)) == freeVersion);
    // the heavy ones step to alpha 0.5 again (the same alpha, new poses); the light ones to another alpha
    CHECK(rig.Alpha(heavyId) == 0.5);
    CHECK(written([&]() { rig.Simulate(0.25); }) == heavyChunks + lightChunks);
    CHECK(rig.Alpha(heavyId) == 0.5 && heavyDomain->steps == 1 && lightDomain->steps == 0);
    CHECK(SameBits(heavies[9]->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(heavies[9]),
          *rig.Get<SimPose>(heavies[9]), *rig.Get<SimScale>(heavies[9]), 0.5)));
    // both alphas as they were, poses of the light ones only
    rig.Simulate(0.0);
    const double lightAlpha = rig.Alpha(lightId);
    CHECK(written([&]() { rig.Simulate(0.5); }) == heavyChunks + lightChunks && rig.Alpha(lightId) == lightAlpha);

    // one pose, one previous pose, one scale, one body reference: the chunk of that row
    auto edited = [&](auto&& edit)
    {
        return written([&]()
        {
            rig.world->AdvanceVersion();
            edit();
            rig.Simulate(0.0);
        });
    };
    CHECK(edited([&]() { rig.Edit<SimPose>(free[5])->translation.x = 100.0; }) == 1);
    CHECK(SameBits(free[5]->GetLocalToWorldTransform(), rig.Posed(free[5])));
    CHECK(edited([&]() { rig.Edit<SimPosePrevious>(heavies[5])->translation.x = 100.0; }) == 1);
    CHECK(heavies[5]->GetLocalToWorldTransform().m_translation.x != double(5));
    CHECK(edited([&]() { rig.Edit<SimScale>(free[250])->scaling = dm::double3(3.0); }) == 1);
    CHECK(SameBits(free[250]->GetLocalToWorldTransform(), rig.Posed(free[250])));
    *rig.Edit<SimPosePrevious>(free[7]) = SimPosePrevious{ dm::double3(-50.0), c_Turn };
    CHECK(edited([&]() { *rig.Edit<SimBodyRef>(free[7]) = SimBodyRef{ uint32_t(heavyId), 0 }; }) >= 1);
    CHECK(SameBits(free[7]->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(free[7]),
          *rig.Get<SimPose>(free[7]), *rig.Get<SimScale>(free[7]), rig.Alpha(heavyId))));
    // from here on that chunk has a row of the heavy domain: written whole when its alpha changes
    CHECK(written([&]() { rig.Simulate(0.0625); }) == heavyChunks + lightChunks + 1);
    // ids that name no domain: the last pose, at once and for good
    for (uint32_t domain : { c_MaxSimulationDomains, c_MaxSimulationDomains - 1, 31u, 32u, 4242u, c_NoSimulationDomain })
    {
        CHECK(edited([&]() { *rig.Edit<SimBodyRef>(free[7]) = SimBodyRef{ domain, 7 }; }) == 1);
        CHECK(SameBits(free[7]->GetLocalToWorldTransform(), rig.Posed(free[7])));
        CHECK(written([&]() { rig.Simulate(0.0625); }) == heavyChunks + lightChunks);
    }
    // a domain goes: its rows are written once more, at their last poses
    lightDomain->clearsBodyRefs = false;
    CHECK(rig.graph->GetScheduler()->RemoveSimulationDomain(lightId) == nvrhi::FS_OK);
    CHECK(written([&]() { rig.Simulate(0.0); }) == lightChunks);
    CHECK(SameBits(lights[3]->GetLocalToWorldTransform(), rig.Posed(lights[3])));
    CHECK(written([&]() { rig.Simulate(0.0); }) == 0);

    // a domain with an id beyond 8: only its own alpha moves its rows
    {
        Rig many;
        const ComponentId none = RegisterComponent<LightBall>(many.world);
        const ComponentId some = RegisterComponent<HeavyBall>(many.world);
        std::vector<nvrhi::AutoPtr<BallDomain>> domains;
        SimulationDomainId last = SimulationDomainId::Invalid;
        for (int d = 0; d < 11; ++d)
        {
            domains.push_back(many.NewDomain(("d" + std::to_string(d)).c_str(), d == 10 ? 1.0 : 1024.0, 4,
                                             d == 10 ? some : none));
            last = many.AddDomain(domains.back());
        }
        CHECK(uint32_t(last) == 10);
        SceneGraphNode* node = many.AddAt(many.Root(), 1.0, 2.0, 3.0);
        CHECK(many.world->AddComponent(many.EntityOf(node), some, nullptr) == nvrhi::FS_OK);
        many.Bind(node);
        many.Frame(1.0);
        (void)TakeWarnings();   // the domains' scatters are ambiguous, in Debug builds
        CHECK(many.Get<SimBodyRef>(node)->domain == 10);
        for (int frame = 0; frame < 3; ++frame)   // no step: the alphas alone
        {
            const dm::daffine3 world = node->GetLocalToWorldTransform();
            many.Frame(0.125);
            CHECK(SameBits(node->GetLocalToWorldTransform(), DirectWorld(*many.Get<SimPosePrevious>(node),
                  *many.Get<SimPose>(node), *many.Get<SimScale>(node), many.Alpha(last))));
            CHECK(!SameBits(node->GetLocalToWorldTransform(), world));
        }
        CHECK(TakeWarnings() == 0);
    }

    // planned steps that were not taken (the phase alone, after a long delta): the last pose, not beyond it
    rig.SetTime(0.25 * 2.5);
    CHECK(rig.Alpha(heavyId) > 2.0);
    rig.Interpolate();
    CHECK(SameBits(heavies[3]->GetLocalToWorldTransform(), rig.Posed(heavies[3])));
    rig.Simulate(0.0);
    CHECK(SameBits(heavies[3]->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(heavies[3]),
          *rig.Get<SimPose>(heavies[3]), *rig.Get<SimScale>(heavies[3]), rig.Alpha(heavyId))));

    // rows that are driven by their tag alone, or not by it: not the interpolation's
    SceneGraphNode* tagged = rig.AddAt(rig.Root(), 1.0, 2.0, 3.0);
    SceneGraphNode* posedOnly = rig.AddAt(rig.Root(), 1.0, 2.0, 3.0);
    rig.Refresh();
    CHECK(rig.world->AddComponent(rig.EntityOf(tagged), FindComponent<SimulationDriven>(rig.world), nullptr) ==
          nvrhi::FS_OK);
    // made by a world call: an archetype that the store has not seen yet hides the local transform all the same
    tagged->SetTranslation(dm::double3(77.0));
    CHECK(TakeWarnings() == 1 && rig.Get<LocalTransform>(tagged)->translation.x == 1.0);
    *rig.Edit<LocalTransform>(tagged) = LocalTransform{ dm::double3(66.0), c_Turn, dm::double3(2.0) };
    CHECK(Near(tagged->GetTranslation(), dm::double3(1.0, 2.0, 3.0), 0.0) &&
          Near(tagged->GetScaling(), dm::double3(1.0), 0.0));
    for (ComponentId component : { FindComponent<SimPose>(rig.world), FindComponent<SimPosePrevious>(rig.world),
                                   FindComponent<SimScale>(rig.world), FindComponent<SimBodyRef>(rig.world) })
        CHECK(rig.world->AddComponent(rig.EntityOf(posedOnly), component, nullptr) == nvrhi::FS_OK);
    rig.Frame(0.25);
    CHECK(Near(tagged->GetLocalToWorldTransform().m_translation, dm::double3(1.0, 2.0, 3.0), 0.0));
    CHECK(Near(posedOnly->GetLocalToWorldTransform().m_translation, dm::double3(1.0, 2.0, 3.0), 0.0));
    rig.SetPose(posedOnly, dm::double3(-50.0), c_Turn);   // a pose that changes: still nothing of the interpolation's
    rig.Frame(0.25);
    CHECK(Near(posedOnly->GetLocalToWorldTransform().m_translation, dm::double3(1.0, 2.0, 3.0), 0.0));
    Place(rig.Root(), Placement{ dm::double3(10.0, 0.0, 0.0) });
    rig.Refresh();
    CHECK(Near(tagged->GetLocalToWorldTransform().m_translation, dm::double3(1.0, 2.0, 3.0), 0.0));
    CHECK(Near(posedOnly->GetLocalToWorldTransform().m_translation, dm::double3(11.0, 2.0, 3.0), 0.0));
    rig.Unbind(tagged);   // at its world transform, for want of a pose
    rig.Refresh();
    CHECK(Near(tagged->GetLocalToWorldTransform().m_translation, dm::double3(1.0, 2.0, 3.0), 1e-14));
    CHECK(Near(tagged->GetTranslation(), dm::double3(-9.0, 2.0, 3.0), 1e-14));
}

// The phase: right after PostSimulation, whose scatters it sees; no ambiguity with a domain's systems or with
// propagation; Refresh does not run it, and propagation in the same frame sees what it wrote.
void test_interpolation_phase()
{
    Rig rig;
    auto domain = rig.NewDomain();
    const SimulationDomainId id = rig.AddDomain(domain);
    SceneGraphNode* node = rig.AddAt(rig.Root(), 0.0, 30.0, 0.0);
    SceneGraphNode* child = rig.AddAt(node, 1.0, 0.0, 0.0);
    rig.Bind(node);
    rig.Refresh();
    IScheduler* scheduler = rig.graph->GetScheduler();
    const Phase phase = rig.Store().GetInterpolationPhase();

    // order: a system in PostSimulation runs before it, one in TransformPropagation after it
    std::vector<int> order;
    uint64_t lastScatter = 0;
    auto add = [&](const char* name, Phase in, std::vector<ComponentAccess> accesses, int mark)
    {
        SystemDesc desc{};
        desc.name = name;
        desc.phase = in;
        desc.flags = SystemFlags::MainThread;
        desc.accesses = accesses.data();
        desc.accessCount = uint32_t(accesses.size());
        nvrhi::AutoPtr<ISystem> system;
        CHECK(CreateLambdaSystem(desc, [&order, &lastScatter, mark](ISystemContext* context) noexcept
        {
            order.push_back(mark);
            lastScatter = context->GetVersion();
            return nvrhi::FS_OK;
        }, &system) == nvrhi::FS_OK);
        SystemId systemId = SystemId::Invalid;
        CHECK(scheduler->AddSystem(system, systemId) == nvrhi::FS_OK);
    };
    add("test.post", Phase::PostSimulation, {}, 1);
    add("test.interpolation", phase, { { FindComponent<SimPose>(rig.world), Access::Read, {} } }, 2);
    add("test.propagation", Phase::TransformPropagation, {}, 3);
    Phase before = Phase::Count;
    CHECK(scheduler->AddPhase("test.before", Phase::PostSimulation, before) == nvrhi::FS_OK);
    add("test.before.system", before, {}, 4);   // inserted later: it runs before what ran after PostSimulation
    const dm::daffine3 world = node->GetLocalToWorldTransform();
    rig.SetTime(0.25 * 1.5);
    rig.RunTo(Phase::TransformPropagation);
    CHECK(order == std::vector<int>({ 1, 4, 2, 3 }));
    CHECK(scheduler->Build() == nvrhi::FS_OK && scheduler->GetAmbiguityCount() == 0);
    nvrhi::string text;
    scheduler->GetDiagnostics(text);
    CHECK(std::strstr(text.c_str(), "phase 'SimulationInterpolation': 'SceneGraph.SimulationInterpolation'"));
    // the pose of this frame's scatter, and the child follows in the same run
    CHECK(domain->steps == 1 && !SameBits(node->GetLocalToWorldTransform(), world));
    CHECK(SameBits(node->GetLocalToWorldTransform(), DirectWorld(*rig.Get<SimPosePrevious>(node), *rig.Get<SimPose>(node),
          *rig.Get<SimScale>(node), 0.5)));
    CHECK(Near(child->GetLocalToWorldTransform(), ComposeTransform(dm::double3(1.0, 0.0, 0.0), dm::dquat::identity(),
               dm::double3(1.0)) * node->GetLocalToWorldTransform(), 0.0));
    CHECK(rig.Alpha(id) == 0.5);

    // its accesses are declared: another system of the phase that writes what it reads, or touches what it writes, is
    // reported (6.4)
    const std::pair<ComponentId, Access> conflicts[] = {
        { FindComponent<SimPose>(rig.world), Access::Write },
        { FindComponent<SimPosePrevious>(rig.world), Access::Write },
        { FindComponent<SimScale>(rig.world), Access::Write },
        { FindComponent<SimBodyRef>(rig.world), Access::Write },
        { FindComponent<FixedStepState>(rig.world), Access::Write },
        { FindComponent<WorldTransform>(rig.world), Access::Read },
    };
    for (const auto& [component, access] : conflicts)
    {
        SystemDesc desc{};
        const ComponentAccess accesses[] = { { component, access, {} } };
        desc.name = "test.conflict";
        desc.phase = phase;
        desc.accesses = accesses;
        desc.accessCount = 1;
        nvrhi::AutoPtr<ISystem> system;
        CHECK(CreateLambdaSystem(desc, [](ISystemContext*) noexcept { }, &system) == nvrhi::FS_OK);
        SystemId systemId = SystemId::Invalid;
        CHECK(scheduler->AddSystem(system, systemId) == nvrhi::FS_OK && scheduler->Build() == nvrhi::FS_OK);
        // "test.interpolation" reads SimPose too
        const uint32_t expected = component == FindComponent<SimPose>(rig.world) ? 2 : 1;
        CHECK(scheduler->GetAmbiguityCount() == expected);
        CHECK(scheduler->RemoveSystem(systemId) == nvrhi::FS_OK && scheduler->Build() == nvrhi::FS_OK);
        CHECK(scheduler->GetAmbiguityCount() == 0);
        (void)TakeWarnings();   // in Debug builds
    }

    // Refresh runs propagation and bounds only
    rig.SetPose(node, dm::double3(1.0), c_Turn);
    const uint64_t chunks = rig.InterpolatedChunks();
    const dm::daffine3 shown = node->GetLocalToWorldTransform();
    node->GetFirstChild()->SetTranslation(dm::double3(2.0));   // so that it runs
    rig.Refresh();
    CHECK(rig.InterpolatedChunks() == chunks && SameBits(node->GetLocalToWorldTransform(), shown));
    rig.Interpolate();
    CHECK(rig.InterpolatedChunks() == chunks + 1 && !SameBits(node->GetLocalToWorldTransform(), shown));
    // PostSimulation alone does not
    rig.SetPose(node, dm::double3(2.0), c_Turn);
    rig.RunTo(Phase::PostSimulation);
    CHECK(rig.InterpolatedChunks() == chunks + 1);
}

// ------------------------------------------------------------------------------------------ the GPU scene

namespace
{
    // UploadTo's target: the buffers as the device would have them.
    struct Mirror final : IGpuSceneWriter
    {
        std::vector<uint8_t> buffers[3];
        static size_t RowSize(GpuSceneBuffer buffer)
        {
            return buffer == GpuSceneBuffer::Instances ? sizeof(InstanceData)
                 : buffer == GpuSceneBuffer::Geometries ? sizeof(GeometryData) : sizeof(MaterialConstants);
        }
        bool ResizeBuffer(GpuSceneBuffer buffer, uint32_t rows) noexcept override
        {
            buffers[uint32_t(buffer)].assign(size_t(rows) * RowSize(buffer), 0xcd);
            return true;
        }
        void WriteRows(GpuSceneBuffer buffer, uint32_t firstRow, uint32_t rowCount, const void* data,
                       size_t rowSize) noexcept override
        {
            std::memcpy(buffers[uint32_t(buffer)].data() + size_t(firstRow) * rowSize, data, size_t(rowCount) * rowSize);
        }
        void WriteMaterialConstants(Material*, const MaterialConstants&) noexcept override { }
        const InstanceData* Instances() const { return reinterpret_cast<const InstanceData*>(buffers[0].data()); }
        size_t InstanceRows() const { return buffers[0].size() / sizeof(InstanceData); }
    };

    GpuSceneDesc MirrorDesc()
    {
        GpuSceneDesc desc;
        desc.geometryAndMaterialBuffers = true;
        desc.rowGranularity = 64;
        return desc;
    }

    // Follows a graph over frames: every live row equals that of a scene built fresh, but for the previous transform,
    // which is the transform of the frame before (the current one for a new row).
    struct GpuFollower
    {
        GpuScene gpu{ nullptr, MirrorDesc() };
        Mirror mirror;
        std::map<int, std::pair<Entity, std::vector<uint8_t>>> last;
        uint64_t rowsChecked = 0;

        void Check(SceneGraph* graph)
        {
            CHECK(NVRHI_SUCCEEDED(gpu.Update(graph)) && NVRHI_SUCCEEDED(gpu.UploadTo(&mirror)));
            GpuScene fresh(nullptr, MirrorDesc());
            Mirror freshMirror;
            CHECK(NVRHI_SUCCEEDED(fresh.Update(graph)) && NVRHI_SUCCEEDED(fresh.UploadTo(&freshMirror)));
            std::map<int, std::pair<Entity, std::vector<uint8_t>>> next;
            for (const auto& instance : graph->GetMeshInstances())
            {
                const int slot = instance->GetInstanceIndex();
                CHECK(slot >= 0 && size_t(slot) < mirror.InstanceRows() && size_t(slot) < freshMirror.InstanceRows());
                InstanceData row = mirror.Instances()[slot];
                const InstanceData& expected = freshMirror.Instances()[slot];
                float transform[12];
                dm::affineToColumnMajor(instance->GetNode()->GetLocalToWorldTransformFloat(), transform);
                CHECK(SameBits(&row.transform, transform, sizeof(transform)));
                const Entity entity = graph->GetEntity(instance->GetNode());
                auto it = last.find(slot);
                const void* previous = it != last.end() && it->second.first == entity
                    ? static_cast<const void*>(it->second.second.data()) : transform;
                CHECK(SameBits(&row.prevTransform, previous, sizeof(transform)));
                row.prevTransform = expected.prevTransform;
                CHECK(SameBits(&row, &expected, sizeof(row)));
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(transform);
                next[slot] = { entity, std::vector<uint8_t>(bytes, bytes + sizeof(transform)) };
                ++rowsChecked;
            }
            last.swap(next);
        }
    };
}

// The rows of driven instances and of those below them follow frame by frame, through bind and unbind.
void test_driver_gpu_scene()
{
    Rig rig;
    auto domain = rig.NewDomain();
    rig.AddDomain(domain);
    Rng rng(21);
    std::vector<SceneGraphNode*> nodes;
    for (int i = 0; i < 200; ++i)
    {
        const Placement p = RandomPlacement(rng);
        nodes.push_back(rig.Add(i % 4 == 3 ? nodes[rng.Below(nodes.size())] : rig.Root(), &p, true));
    }
    rig.Refresh();
    GpuFollower gpu;
    gpu.Check(rig.graph);
    uint64_t changed = 0;
    for (int frame = 0; frame < 25; ++frame)
    {
        for (int k = 0; k < 6 && frame % 5 != 4; ++k)
        {
            SceneGraphNode* node = nodes[rng.Below(nodes.size())];
            if (frame < 15 || rng.Percent(50))
                rig.Bind(node);
            else
                rig.Unbind(node);
        }
        if (frame % 3)
            rig.Simulate(0.25 * rng.Range(0.0, 1.5));
        rig.Refresh();
        gpu.Check(rig.graph);
        changed += gpu.gpu.GetStats().rowsChanged;
    }
    // the interpolation straight after a GPU update (no phase and no edit between): its stamps are newer than that
    for (int frame = 0; frame < 3; ++frame)
    {
        rig.Frame(0.25 * 1.1);
        gpu.Check(rig.graph);
        rig.SetTime(0.01);
        rig.Interpolate();
        rig.Refresh();
        for (SceneGraphNode* node : nodes)
        {
            const dm::box3 box = rig.mesh->objectSpaceBounds * dm::affine3(node->GetLocalToWorldTransform());
            CHECK(all(rig.Get<WorldBounds>(node)->bounds.m_mins == box.m_mins));
        }
        gpu.Check(rig.graph);
        CHECK(gpu.gpu.GetStats().rowsChanged > 0);
    }
    // at rest: nothing is sent
    for (int frame = 0; frame < 3; ++frame)
    {
        rig.Simulate(0.0);
        rig.Refresh();
        gpu.Check(rig.graph);
    }
    CHECK(gpu.gpu.GetStats().rowsChanged == 0 && changed > 400 && gpu.rowsChecked > 5000);
}

// --------------------------------------------------------------------------------------- the random script

namespace
{
    // What the script says the graph is, kept apart from it: per node its parent, its leaf's box and content and either
    // a local transform or a simulated state. Every value the graph shows is computed from this from scratch.
    struct ModelNode
    {
        ModelNode* parent = nullptr;
        std::vector<ModelNode*> children;
        nvrhi::AutoPtr<SceneGraphNode> node;
        bool hasLocal = false;
        Placement local;
        bool driven = false;
        SimPosePrevious previous{};      // what the interpolation must have seen
        SimPose pose{};
        SimScale scale{};
        bool shown = false;              // an interpolation ran since it was bound
        dm::daffine3 boundWorld;         // its world until then
        uint64_t serial = 0;
        bool hasLeaf = false;
        dm::box3 box = dm::box3::empty();
        SceneContentFlags content = SceneContentFlags::None;
        // the results of the last frame
        dm::daffine3 world;
        dm::box3 subtreeBox;
        SceneContentFlags subtreeContent;
    };

    class Script
    {
    public:
        explicit Script(uint64_t seed, uint32_t workers) : m_Rig(workers), m_Rng(seed)
        {
            m_Meshes.push_back(MakeMesh(m_Rig.factory, 1.f, MaterialDomain::Opaque));
            m_Meshes.push_back(MakeMesh(m_Rig.factory, 0.5f, MaterialDomain::AlphaBlended));
            m_Domain = m_Rig.NewDomain("balls", 1.0 / 32.0, 3);
            m_Domain->keepEvents = false;
            m_Id = m_Rig.AddDomain(m_Domain);
            m_Root = New();
            m_Root->node = m_Rig.Root();
            m_Attached.push_back(m_Root);
            for (int i = 0; i < 60; ++i)
                Grow(m_Attached[m_Rng.Below(m_Attached.size())]);
            // two chains beyond the last depth bucket
            for (int c = 0; c < 2; ++c)
            {
                ModelNode* at = m_Root;
                for (int d = 0; d < 75; ++d)
                    at = Grow(at);
            }
        }

        struct Counts
        {
            uint32_t binds = 0, unbinds = 0, moves = 0, reparents = 0, leaves = 0, detaches = 0, attaches = 0, drops = 0,
                     copies = 0, roots = 0, drivenDetached = 0, deepDriven = 0, simulated = 0, stillFrames = 0,
                     between = 0;
        };
        const Counts& GetCounts() const { return m_Counts; }
        uint64_t GetHash() const { return m_Hash; }

        void Frame(uint32_t frame)
        {
            if (frame % 11 != 10)
            {
                const uint32_t edits = 1 + m_Rng.Below(12);
                for (uint32_t e = 0; e < edits; ++e)
                    EditOnce();
            }
            if (m_Rng.Percent(70))
            {
                const double factors[] = { 0.0, 0.3, 1.0, 1.7, 8.0 };
                m_Rig.Simulate(m_Domain->dt * factors[m_Rng.Below(std::size(factors))]);
                ++m_Counts.simulated;
                AfterSimulation();
                // edits between the interpolation and the Refresh, with driven nodes that moved and children that did not
                for (uint32_t e = m_Rng.Percent(50) ? m_Rng.Below(6) : 0; e > 0; --e)
                {
                    EditOnce();
                    ++m_Counts.between;
                }
            }
            else
                ++m_Counts.stillFrames;
            m_Rig.Refresh();
            Compare(m_Root);
            CheckStorage();
            m_Gpu.Check(m_Rig.graph);
        }

    private:
        Rig m_Rig;
        Rng m_Rng;
        std::vector<nvrhi::AutoPtr<MeshInfo>> m_Meshes;
        nvrhi::AutoPtr<BallDomain> m_Domain;
        SimulationDomainId m_Id = SimulationDomainId::Invalid;
        std::vector<std::unique_ptr<ModelNode>> m_Pool;
        std::vector<ModelNode*> m_Attached;
        std::vector<ModelNode*> m_Detached;   // roots of subgraphs in no graph
        ModelNode* m_Root = nullptr;
        GpuFollower m_Gpu;
        Counts m_Counts;
        uint64_t m_Hash = 0xcbf29ce484222325ull;
        uint64_t m_Serial = 0;
        double m_Alpha = 1.0;

        ModelNode* New()
        {
            m_Pool.push_back(std::make_unique<ModelNode>());
            m_Pool.back()->serial = ++m_Serial;
            return m_Pool.back().get();
        }
        ModelNode* Pick() { return m_Attached[m_Rng.Below(m_Attached.size())]; }
        // Of at most 20 nodes, so that the chains and most of what is driven stay; the root when it finds none.
        ModelNode* PickSmall()
        {
            for (int i = 0; i < 8; ++i)
            {
                ModelNode* node = Pick();
                size_t size = 0;
                Walk(node, [&](ModelNode*) { ++size; });
                if (node != m_Root && size <= 20)
                    return node;
            }
            return m_Root;
        }
        static bool IsUnder(const ModelNode* node, const ModelNode* ancestor)
        {
            for (; node; node = node->parent)
            {
                if (node == ancestor)
                    return true;
            }
            return false;
        }
        void Link(ModelNode* parent, ModelNode* child)
        {
            child->parent = parent;
            parent->children.push_back(child);
        }
        static void Unlink(ModelNode* child)
        {
            auto& siblings = child->parent->children;
            siblings.erase(std::find(siblings.begin(), siblings.end(), child));
            child->parent = nullptr;
        }
        void Walk(ModelNode* node, const std::function<void(ModelNode*)>& visit)
        {
            visit(node);
            for (size_t i = 0; i < node->children.size(); ++i)
                Walk(node->children[i], visit);
        }
        void RebuildAttached()
        {
            m_Attached.clear();
            Walk(m_Root, [&](ModelNode* node) { m_Attached.push_back(node); });
        }
        void GiveLeaf(ModelNode* node)
        {
            const uint32_t kind = m_Rng.Below(4);
            node->hasLeaf = kind != 3;
            node->box = dm::box3::empty();
            node->content = SceneContentFlags::None;
            if (kind < 2)
            {
                node->node->SetLeaf(m_Rig.factory->CreateMeshInstance(m_Meshes[kind]));
                node->box = m_Meshes[kind]->objectSpaceBounds;
                node->content = kind ? SceneContentFlags::BlendedMeshes : SceneContentFlags::OpaqueMeshes;
            }
            else if (kind == 2)
            {
                node->node->SetLeaf(MAKE_RC_OBJ_PTR(PointLight));
                node->content = SceneContentFlags::Lights;
            }
            else
                node->node->SetLeaf(nullptr);
        }
        ModelNode* Grow(ModelNode* parent)
        {
            ModelNode* node = New();
            node->node = MakeNode();
            if (m_Rng.Percent(75))
            {
                node->hasLocal = true;
                node->local = RandomPlacement(m_Rng);
                node->local.scaling = dm::double3(m_Rng.Percent(30) ? m_Rng.Range(0.95, 1.05) : 1.0);
                Place(node->node, node->local);
            }
            if (m_Rng.Percent(60))
                GiveLeaf(node);
            CHECK(m_Rig.graph->Attach(parent->node, node->node) == node->node);
            Link(parent, node);
            m_Attached.push_back(node);
            return node;
        }

        // The world transform that propagation must give `node` now, from scratch. `simulated`: driven ancestors at
        // their simulated poses.
        dm::daffine3 World(const ModelNode* node, bool simulated = false) const
        {
            if (node->driven)
            {
                if (simulated)
                    return DirectWorld(node->previous, node->pose, node->scale, 1.0);
                return node->shown ? DirectWorld(node->previous, node->pose, node->scale, m_Alpha) : node->boundWorld;
            }
            const dm::daffine3 local = node->hasLocal
                ? ComposeTransform(node->local.translation, node->local.rotation, node->local.scaling)
                : dm::daffine3::identity();
            if (!node->parent)
                return local;
            const dm::daffine3 parent = World(node->parent, simulated);
            return node->hasLocal ? local * parent : parent;
        }
        // `node` stops being driven with the local transform that the graph gave it: checked against the model's pose,
        // then taken over, so that everything after it is compared bit for bit.
        void BecomesAuthored(ModelNode* node, const dm::daffine3& simulatedWorld, const dm::daffine3& simulatedParent)
        {
            node->driven = false;
            node->hasLocal = true;
            node->local = { node->node->GetTranslation(), node->node->GetRotation(), node->node->GetScaling() };
            const dm::daffine3 local =
                ComposeTransform(node->local.translation, node->local.rotation, node->local.scaling);
            CHECK(Near(local * simulatedParent, simulatedWorld, 1e-8));
        }
        // Every driven node of the subgraph became authored (Detach).
        void SubgraphBecomesAuthored(ModelNode* top)
        {
            struct Entry { ModelNode* node; dm::daffine3 world, parent; };
            std::vector<Entry> driven;
            Walk(top, [&](ModelNode* node)
            {
                if (node->driven)
                    driven.push_back({ node, World(node, true),
                                       node->parent ? World(node->parent, true) : dm::daffine3::identity() });
            });
            m_Counts.drivenDetached += uint32_t(driven.size());
            m_Pending = [this, driven]()
            {
                for (const Entry& entry : driven)
                    BecomesAuthored(entry.node, entry.world, entry.parent);
            };
        }
        std::function<void()> m_Pending;

        uint32_t Depth(const ModelNode* node) const
        {
            uint32_t depth = 0;
            for (; node->parent; node = node->parent)
                ++depth;
            return depth;
        }

        void EditOnce()
        {
            const uint32_t pick = m_Rng.Below(100);
            if (pick < 22)
            {
                ModelNode* node = Pick();
                for (int i = 0; i < 8 && (node->driven || (m_Rng.Percent(50) && Depth(node) < 60)); ++i)
                    node = Pick();
                if (node->driven)
                    return;
                const dm::daffine3 world = World(node, true);
                m_Rig.Bind(node->node);
                node->driven = true;
                node->shown = false;
                node->boundWorld = world;
                node->pose = *m_Rig.Get<SimPose>(node->node);
                node->previous = SimPosePrevious{ node->pose.translation, node->pose.rotation };
                node->scale = *m_Rig.Get<SimScale>(node->node);
                CHECK(Near(DirectWorld(node->previous, node->pose, node->scale, 1.0), world, 1e-8));
                ++m_Counts.binds;
                m_Counts.deepDriven += Depth(node) >= c_MaxHierarchyDepth ? 1 : 0;
            }
            else if (pick < 36)
            {
                ModelNode* node = Pick();
                for (int i = 0; i < 8 && !node->driven; ++i)
                    node = Pick();
                if (!node->driven)
                    return;
                const dm::daffine3 world = World(node, true);
                const dm::daffine3 parent = node->parent ? World(node->parent, true) : dm::daffine3::identity();
                m_Rig.Unbind(node->node);
                BecomesAuthored(node, world, parent);
                ++m_Counts.unbinds;
            }
            else if (pick < 64)
            {
                ModelNode* node = Pick();
                if (node->driven)
                    return;
                if (!node->hasLocal)
                    node->local = Placement{};
                node->hasLocal = true;
                node->local.translation = RandomVector(m_Rng, -5.0, 5.0);
                node->node->SetTranslation(node->local.translation);
                if (m_Rng.Percent(50))
                {
                    node->local.rotation = RandomRotation(m_Rng);
                    node->node->SetRotation(node->local.rotation);
                }
                ++m_Counts.moves;
            }
            else if (pick < 72)
            {
                GiveLeaf(Pick());
                ++m_Counts.leaves;
            }
            else if (pick < 80)   // reparent: out of one place, into another
            {
                ModelNode* node = PickSmall();
                ModelNode* to = Pick();
                if (node == m_Root || IsUnder(to, node))
                    return;
                SubgraphBecomesAuthored(node);
                Unlink(node);
                CHECK(m_Rig.graph->Detach(node->node) == node->node);
                m_Pending();
                CHECK(m_Rig.graph->Attach(to->node, node->node) == node->node);
                Link(to, node);
                ++m_Counts.reparents;
            }
            else if (pick < 86)
            {
                ModelNode* node = PickSmall();
                if (node == m_Root)
                    return;
                SubgraphBecomesAuthored(node);
                Unlink(node);
                CHECK(m_Rig.graph->Detach(node->node) == node->node);
                m_Pending();
                RebuildAttached();
                if (m_Rng.Percent(40))   // dropped: the objects go
                {
                    Walk(node, [](ModelNode* gone) { gone->node.Reset(); });
                    ++m_Counts.drops;
                }
                else
                {
                    m_Detached.push_back(node);
                    ++m_Counts.detaches;
                }
            }
            else if (pick < 93)
            {
                if (m_Detached.empty())
                {
                    for (int i = 0; i < 4 && m_Attached.size() < 500; ++i)
                        Grow(Pick());
                    return;
                }
                const size_t which = m_Rng.Below(m_Detached.size());
                ModelNode* node = m_Detached[which];
                m_Detached.erase(m_Detached.begin() + ptrdiff_t(which));
                ModelNode* to = Pick();
                CHECK(m_Rig.graph->Attach(to->node, node->node) == node->node);
                Link(to, node);
                RebuildAttached();
                ++m_Counts.attaches;
            }
            else if (pick < 98)
            {
                ModelNode* source = Pick();
                size_t size = 0;
                Walk(source, [&](ModelNode*) { ++size; });
                if (size > 40 || m_Attached.size() > 500)
                    return;
                ModelNode* to = Pick();
                // the values of the copies, before it (the target may be inside the source)
                struct Copied { const ModelNode* source; dm::daffine3 world, parent; };
                std::vector<Copied> order;
                Walk(source, [&](ModelNode* node)
                {
                    const bool hasParent = node->driven && node->parent;
                    order.push_back({ node, node->driven ? World(node, true) : dm::daffine3::identity(),
                                      hasParent ? World(node->parent, true) : dm::daffine3::identity() });
                });
                nvrhi::AutoPtr<SceneGraphNode> top = m_Rig.graph->Attach(to->node, source->node);
                CHECK(top && top != source->node);
                std::map<const ModelNode*, ModelNode*> copies;
                SceneGraphWalker walker(top);
                for (const Copied& c : order)
                {
                    CHECK(walker);
                    ModelNode* copy = New();
                    copy->node = walker.Get();
                    copy->hasLeaf = c.source->hasLeaf;
                    copy->box = c.source->box;
                    copy->content = c.source->content;
                    if (c.source->driven)
                        BecomesAuthored(copy, c.world, c.parent);
                    else
                    {
                        copy->hasLocal = c.source->hasLocal;
                        copy->local = c.source->local;
                    }
                    Link(c.source == source ? to : copies.at(c.source->parent), copy);
                    copies[c.source] = copy;
                    walker.Next(true);
                }
                CHECK(!walker);
                RebuildAttached();
                ++m_Counts.copies;
            }
            else   // a new root above the old one
            {
                if (m_Counts.roots >= 5 || m_Rng.Percent(60))
                    return;
                SubgraphBecomesAuthored(m_Root);
                ModelNode* root = New();
                root->node = MakeNode("root");
                auto old = m_Rig.graph->SetRootNode(root->node);
                CHECK(old == m_Root->node);
                m_Pending();
                CHECK(m_Rig.graph->Attach(root->node, old) == old);
                Link(root, m_Root);
                m_Root = root;
                RebuildAttached();
                ++m_Counts.roots;
            }
        }

        void AfterSimulation()
        {
            m_Alpha = m_Rig.Alpha(m_Id);
            for (ModelNode* node : m_Attached)
            {
                if (!node->driven)
                    continue;
                const BallDomain::Ball* ball = m_Domain->Find(m_Rig.EntityOf(node->node));
                CHECK(ball);
                node->pose = ball->pose;
                node->previous = ball->previous;
                node->shown = true;
            }
        }

        // Everything the graph shows, against the model.
        void Compare(ModelNode* node)
        {
            const dm::daffine3 localMatrix = node->hasLocal
                ? ComposeTransform(node->local.translation, node->local.rotation, node->local.scaling)
                : dm::daffine3::identity();
            node->world = node->driven ? World(node)
                          : !node->parent ? localMatrix
                          : node->hasLocal ? localMatrix * node->parent->world : node->parent->world;
            CHECK(m_Rig.IsDriven(node->node) == node->driven);
            CHECK(SameBits(node->node->GetLocalToWorldTransform(), node->world));
            CHECK(node->node->GetParent() == (node->parent ? node->parent->node.Get() : nullptr));
            Mix(&node->world, sizeof(node->world));
            node->subtreeBox = dm::box3::empty();
            if (node->hasLeaf && !node->box.isempty())
                node->subtreeBox = node->box * dm::affine3(node->world);
            node->subtreeContent = node->content;
            for (ModelNode* child : node->children)
            {
                Compare(child);
                node->subtreeBox |= child->subtreeBox;
                node->subtreeContent |= child->subtreeContent;
            }
            const dm::box3 box = node->node->GetGlobalBoundingBox();
            CHECK(all(box.m_mins == node->subtreeBox.m_mins) && all(box.m_maxs == node->subtreeBox.m_maxs));
            CHECK(node->node->GetSubgraphContentFlags() == node->subtreeContent);
            if (!node->driven)
            {
                CHECK(m_Rig.world->HasComponent(m_Rig.EntityOf(node->node), FindComponent<LocalTransform>(m_Rig.world)) ==
                      node->hasLocal);
                if (node->hasLocal)
                {
                    const dm::double3 translation = node->node->GetTranslation();
                    CHECK(SameBits(&translation, &node->local.translation, sizeof(translation)));
                }
            }
            else
            {
                const dm::daffine3 parent = node->parent ? node->parent->world : dm::daffine3::identity();
                CHECK(Near(node->node->GetLocalToParentTransform() * parent, node->world, 1e-7));
            }
        }

        // The domain's engine holds exactly the driven nodes, and their rows its poses.
        void CheckStorage()
        {
            size_t driven = 0;
            for (ModelNode* node : m_Attached)
            {
                if (!node->driven)
                    continue;
                ++driven;
                const BallDomain::Ball* ball = m_Domain->Find(m_Rig.EntityOf(node->node));
                CHECK(ball && m_Rig.Get<SimBodyRef>(node->node)->domain == uint32_t(m_Id));
                if (node->shown)
                    CHECK(SameBits(m_Rig.Get<SimPose>(node->node), &ball->pose, sizeof(SimPose)));
                Mix(&ball->pose, sizeof(ball->pose));
                const SimBodyRef body = *m_Rig.Get<SimBodyRef>(node->node);
                Mix(&body, sizeof(body));
            }
            size_t alive = 0;
            for (const BallDomain::Ball& ball : m_Domain->balls)
                alive += ball.entity != Entity::Null ? 1 : 0;
            CHECK(alive == driven);
        }

        void Mix(const void* data, size_t size)
        {
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; ++i)
            {
                m_Hash ^= bytes[i];
                m_Hash *= 0x100000001b3ull;
            }
        }
    };
}

// Random binds, unbinds, moves, reparents, SetLeaf, detaches, re-attaches, destructions, copies and new roots, between
// frames with and without simulation: after every frame each world transform (bit for bit), box and subtree content
// equals what the model computes from scratch, the engine holds exactly the driven nodes, and the GPU scene follows.
void test_driver_random()
{
#ifdef NDEBUG
    constexpr uint32_t c_Frames = 220;
#else
    constexpr uint32_t c_Frames = 60;
#endif
    Script::Counts total;
    for (uint64_t seed : { 1ull, 2ull, 3ull, 4ull })
    {
        Script script(seed, 2);
        for (uint32_t frame = 0; frame < c_Frames; ++frame)
            script.Frame(frame);
        const Script::Counts& c = script.GetCounts();
        std::printf("random %llu: %u binds (%u deep), %u unbinds, %u moves, %u reparents, %u SetLeaf, %u detaches, %u "
                    "attaches, %u dropped, %u copies, %u roots, %u driven nodes detached, %u frames simulated, %u not, "
                    "%u "
                    "edits between\n",
                    (unsigned long long)seed, c.binds, c.deepDriven, c.unbinds, c.moves, c.reparents, c.leaves,
                    c.detaches, c.attaches, c.drops, c.copies, c.roots, c.drivenDetached, c.simulated, c.stillFrames,
                    c.between);
        total.binds += c.binds;
        total.deepDriven += c.deepDriven;
        total.unbinds += c.unbinds;
        total.drivenDetached += c.drivenDetached;
        total.copies += c.copies;
        total.roots += c.roots;
        total.drops += c.drops;
    }
    CHECK(total.binds > 200 && total.deepDriven > 20 && total.unbinds > 50 && total.drivenDetached > 50 &&
          total.copies > 10 && total.roots > 4 && total.drops > 10);
}

// The same script at 1, 2 and 8 workers: the same transforms, poses and engine indices in every frame.
void test_driver_workers()
{
    uint64_t first = 0;
    for (uint32_t workers : { 1u, 2u, 8u, 8u })
    {
        Script script(99, workers);
        for (uint32_t frame = 0; frame < 60; ++frame)
            script.Frame(frame);
        if (!first)
            first = script.GetHash();
        CHECK(script.GetHash() == first);
    }
}

// Steady frames (gather, two steps, scatter, interpolation, Refresh) of 1,000 driven nodes with children and leaves
// allocate nothing.
void test_driver_allocations()
{
    for (uint32_t workers : { 1u, 8u })
    {
        Rig rig(workers);
        auto domain = rig.NewDomain("balls", 0.25, 4);
        rig.AddDomain(domain);
        Rng rng(13);
        std::vector<SceneGraphNode*> nodes;
        for (int i = 0; i < 1000; ++i)
        {
            const Placement p = RandomPlacement(rng);
            nodes.push_back(rig.Add(rig.Root(), &p, i % 2 == 0));
            rig.Add(nodes.back(), &p, i % 3 == 0);
            rig.Bind(nodes.back());
        }
        SceneGraphNode* deep = nodes[0];
        for (int i = 0; i < 80; ++i)
            deep = rig.AddAt(deep, 0.0, 0.01, 0.0);
        rig.Bind(deep->GetParent());
        auto frame = [&]()
        {
            rig.Frame(0.25 * 2.125);
            float sum = 0.f;
            for (int i = 0; i < 50; i++)
                sum += float(nodes[rng.Below(nodes.size())]->GetTranslation().x);
            return sum + rig.Root()->GetGlobalBoundingBox().m_mins.x;
        };
        for (int i = 0; i < 8; ++i)
            (void)frame();
        const uint32_t steps = domain->steps;
        const uint64_t newBefore = g_NewCount.load();
        const uint64_t heapBefore = ecs::GetHeapAllocationCount();
        float sum = 0.f;
        for (int i = 0; i < 10; ++i)
            sum += frame();
        const uint64_t newCalls = g_NewCount.load() - newBefore;
        const uint64_t heapBlocks = ecs::GetHeapAllocationCount() - heapBefore;
        std::printf("steady frames, %u workers: %llu operator new, %llu storage blocks, %u steps (checksum %g)\n",
                    workers, (unsigned long long)newCalls, (unsigned long long)heapBlocks, domain->steps - steps,
                    double(sum));
        CHECK(domain->steps - steps >= 20 && newCalls == 0 && heapBlocks == 0);
        std::vector<int> probe(1000);   // the counter sees an allocation
        CHECK(g_NewCount.load() > newBefore && probe.size() == 1000);
    }
}

int main(int argc, char** argv)
{
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        { "driver_bind", test_driver_bind },
        { "driver_bind_stale", test_driver_bind_stale },
        { "driver_bind_between", test_driver_bind_between },
        { "driver_body_refs", test_driver_body_refs },
        { "driver_propagation", test_driver_propagation },
        { "driver_deep", test_driver_deep },
        { "driver_runtime_entities", test_driver_runtime_entities },
        { "driver_unbind", test_driver_unbind },
        { "driver_detach", test_driver_detach },
        { "driver_copy", test_driver_copy },
        { "driver_set_root_and_leaf", test_driver_set_root_and_leaf },
        { "driver_destroy", test_driver_destroy },
        { "driver_getters", test_driver_getters },
        { "driver_refusals", test_driver_refusals },
        { "driver_stale_flag", test_driver_stale_flag },
        { "driver_joints", test_driver_joints },
        { "driver_job_system", test_driver_job_system },
        { "driver_propagated_rows", test_driver_propagated_rows },
        { "driver_gpu_scene", test_driver_gpu_scene },
        { "driver_random", test_driver_random },
        { "driver_workers", test_driver_workers },
        { "driver_allocations", test_driver_allocations },
        { "interpolation_formula", test_interpolation_formula },
        { "interpolation_poses", test_interpolation_poses },
        { "interpolation_changes", test_interpolation_changes },
        { "interpolation_phase", test_interpolation_phase },
    };
    log::SetCallback([](log::Severity severity, const char* message)
    {
        if (severity >= log::Severity::Warning)
        {
            g_Warnings.fetch_add(1);
            std::fprintf(stderr, "%s\n", message);
        }
    });
    try
    {
        for (const Case& c : cases)
        {
            if (argc > 1 && std::strcmp(argv[1], c.name) != 0)
                continue;
            c.run();
            CHECK(g_Warnings.load() == 0);
            CHECK(g_Errors.load() == 0);
            std::printf("%s: passed\n", c.name);
            std::fflush(stdout);
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
