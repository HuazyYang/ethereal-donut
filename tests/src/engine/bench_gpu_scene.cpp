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

// The CPU cost of the instance upload (aggregate docs/plans/2026-11-21-donut-ecs.md, step E4; design 10.1 "upload after
// 10k of 100k instances moved: proportional to the changed chunks"). Without a device: the upload's copy into the
// upload heap is a memcpy into a staging array, which is what writeBuffer does on the CPU.
//
// One source, two builds:
// - default (step E4 and later): GpuScene::Update plus UploadTo with a writer that copies the rows it is given;
// - BENCH_GPU_SCENE_E0=1, compiled against the E0 tree (the scene graph and Scene of donut f49809b, from outside that
//   tree): the instance path of the old Scene::RefreshBuffers, Scene::UpdateInstance for every instance and the whole
//   instance array written whenever a transform or the structure changed.
//
// Cases (a frame: the edits, SceneGraph::Refresh, then the timed upload), 100k instances under the root:
//   idle            nothing moves (after the previous frames' moves settled)
//   one_moved       the same instance moves every frame
//   block_10pct     10k consecutive instances move every frame
//   strided_10pct   every 10th instance moves every frame
//   all_moved       every instance moves every frame
// Release: the median of the run medians over alternating rounds (`--rounds N`); `--smoke` divides the sizes by 100.

#include <donut/engine/SceneGraph.h>
#include <nvrhi/common/misc.h>
#if !BENCH_GPU_SCENE_E0
#include <donut/engine/GpuScene.h>
#endif

using namespace donut::math;
#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace donut;
using namespace donut::engine;

namespace
{
    struct Options
    {
        uint32_t instances = 100000;
        uint32_t rounds = 5;
        uint32_t frames = 20;
    };

    struct Scene
    {
        nvrhi::AutoPtr<SceneGraph> graph;
        std::vector<nvrhi::AutoPtr<Material>> materials;
        std::vector<nvrhi::AutoPtr<MeshInfo>> meshes;
        std::vector<SceneGraphNode*> nodes;   // in attach order: instance i has slot i in both builds
        std::vector<uint8_t> staging;         // the upload heap
        uint32_t frame = 0;
    };

    void Build(Scene& s, uint32_t count)
    {
        for (int i = 0; i < 4; ++i)
        {
            auto material = MAKE_RC_OBJ_PTR(Material);
            material->name = "material" + std::to_string(i);
            s.materials.push_back(material);
        }
        for (int i = 0; i < 4; ++i)
        {
            auto mesh = MAKE_RC_OBJ_PTR(MeshInfo);
            mesh->name = "mesh" + std::to_string(i);
            for (int g = 0; g <= i % 3; ++g)
            {
                auto geometry = MAKE_RC_OBJ_PTR(MeshGeometry);
                geometry->material = s.materials[(i + g) % 4];
                geometry->numIndices = 36;
                geometry->numVertices = 24;
                mesh->geometries.push_back(geometry);
            }
            mesh->objectSpaceBounds = dm::box3(dm::float3(-1.f), dm::float3(1.f));
            s.meshes.push_back(mesh);
        }
        s.graph = MAKE_RC_OBJ_PTR(SceneGraph);
        auto root = MAKE_RC_OBJ_PTR(SceneGraphNode);
        root->SetName("root");
        s.graph->SetRootNode(root);
        for (uint32_t i = 0; i < count; ++i)
        {
            auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
            node->SetTranslation(dm::double3(double(i % 1000), double(i / 1000), 0.0));
            node->SetLeaf(MAKE_RC_OBJ_PTR(MeshInstance, s.meshes[i % 4].Get()));
            s.graph->Attach(s.graph->GetRootNode(), node);
            s.nodes.push_back(node);
        }
        s.staging.resize(size_t(count + 1024) * sizeof(InstanceData));
    }

    enum class Case { Idle, OneMoved, Block10, Strided10, AllMoved, Count };
    const char* const c_CaseNames[] = { "idle", "one_moved", "block_10pct", "strided_10pct", "all_moved" };

    void Move(Scene& s, Case c)
    {
        const uint32_t n = uint32_t(s.nodes.size());
        const double offset = double(s.frame & 1) * 0.5;
        auto move = [&](uint32_t i) { s.nodes[i]->SetTranslation(dm::double3(double(i % 1000), double(i / 1000), offset)); };
        switch (c)
        {
        case Case::OneMoved: move(n / 2); break;
        case Case::Block10: for (uint32_t i = n / 3; i < n / 3 + n / 10; ++i) move(i); break;
        case Case::Strided10: for (uint32_t i = 0; i < n; i += 10) move(i); break;
        case Case::AllMoved: for (uint32_t i = 0; i < n; ++i) move(i); break;
        default: break;
        }
    }

#if BENCH_GPU_SCENE_E0
    // The old Scene's instance path (donut f49809b src/engine/Scene.cpp, RefreshSceneGraph and RefreshBuffers).
    struct Uploader
    {
        std::vector<InstanceData> instanceData;
        bool structureChanged = false;
        bool transformsChanged = false;
        uint64_t bytes = 0;
        uint64_t rows = 0;

        void BeforeRefresh(Scene& s)
        {
            structureChanged = s.graph->HasPendingStructureChanges();
            transformsChanged = s.graph->HasPendingTransformChanges();
        }

        // Scene::UpdateInstance
        void UpdateInstance(const MeshInstance* instance)
        {
            SceneGraphNode* node = instance->GetNode();
            if (!node)
                return;
            InstanceData& idata = instanceData[instance->GetInstanceIndex()];
            affineToColumnMajor(node->GetLocalToWorldTransformFloat(), idata.transform);
            affineToColumnMajor(node->GetPrevLocalToWorldTransformFloat(), idata.prevTransform);
            const auto& mesh = instance->GetMesh();
            idata.firstGeometryInstanceIndex = instance->GetGeometryInstanceIndex();
            idata.numGeometries = uint32_t(mesh->geometries.size());
            idata.firstGeometryIndex = idata.numGeometries > 0 ? mesh->geometries[0]->globalGeometryIndex : -1;
            idata.flags = 0u;
            if (mesh->type == MeshType::CurveDisjointOrthogonalTriangleStrips)
                idata.flags |= InstanceFlags_CurveDisjointOrthogonalTriangleStrips;
            else if (mesh->type == MeshType::CurveLinearSweptSpheres)
                idata.flags |= InstanceFlags_CurveLinearSweptSpheres;
        }

        void Upload(Scene& s)
        {
            bytes = 0;
            rows = 0;
            bool arraysAllocated = false;
            if (s.graph->GetMeshInstances().size() > instanceData.size())
            {
                instanceData.resize(nvrhi::align<size_t>(s.graph->GetMeshInstances().size(), 1024));
                arraysAllocated = true;
            }
            if (structureChanged || transformsChanged || arraysAllocated)
            {
                for (const auto& instance : s.graph->GetMeshInstances())
                    UpdateInstance(instance);
                // WriteInstanceBuffer: the whole array
                std::memcpy(s.staging.data(), instanceData.data(), instanceData.size() * sizeof(InstanceData));
                bytes = instanceData.size() * sizeof(InstanceData);
                rows = s.graph->GetMeshInstances().size();
            }
        }
    };
#else
    struct Uploader final : IGpuSceneWriter
    {
        nvrhi::AutoPtr<GpuScene> gpu;
        Scene* scene = nullptr;
        uint64_t bytes = 0;
        uint64_t rows = 0;

        Uploader()
        {
            GpuSceneDesc desc;
            desc.geometryAndMaterialBuffers = false;   // as the old instance path: instances only
            desc.rowGranularity = 1024;
            gpu = MAKE_RC_OBJ_PTR(GpuScene, nullptr, desc);
        }
        void BeforeRefresh(Scene&) { }
        bool ResizeBuffer(GpuSceneBuffer, uint32_t) noexcept override { return true; }
        void WriteRows(GpuSceneBuffer buffer, uint32_t firstRow, uint32_t rowCount, const void* data,
                       size_t rowSize) noexcept override
        {
            if (buffer != GpuSceneBuffer::Instances)
                return;
            std::memcpy(scene->staging.data() + size_t(firstRow) * rowSize, data, size_t(rowCount) * rowSize);
            bytes += uint64_t(rowCount) * rowSize;
        }
        void WriteMaterialConstants(Material*, const MaterialConstants&) noexcept override { }
        void Upload(Scene& s)
        {
            scene = &s;
            bytes = 0;
            gpu->Update(s.graph);
            gpu->UploadTo(this);
            rows = gpu->GetStats().rowsChanged + gpu->GetStats().rowsSettled;
        }
    };
#endif

    struct Sample
    {
        double uploadMs;
        double refreshMs;
        uint64_t bytes;
        uint64_t rows;
    };

    // One run of a case: warm-up frames, then `frames` timed ones; the medians of the run.
    Sample Run(Scene& s, Uploader& uploader, Case c, uint32_t frames)
    {
        using clock = std::chrono::steady_clock;
        // the previous case's moves settle first
        for (int i = 0; i < 3; ++i)
        {
            uploader.BeforeRefresh(s);
            s.graph->Refresh(++s.frame);
            uploader.Upload(s);
        }
        std::vector<double> upload, refresh;
        uint64_t bytes = 0, rows = 0;
        for (uint32_t f = 0; f < frames + 2; ++f)
        {
            Move(s, c);
            uploader.BeforeRefresh(s);
            const auto t0 = clock::now();
            s.graph->Refresh(++s.frame);
            const auto t1 = clock::now();
            uploader.Upload(s);
            const auto t2 = clock::now();
            if (f < 2)
                continue;   // warm-up: the case's first frames
            refresh.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            upload.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            bytes = uploader.bytes;
            rows = uploader.rows;
        }
        std::sort(upload.begin(), upload.end());
        std::sort(refresh.begin(), refresh.end());
        return Sample{ upload[upload.size() / 2], refresh[refresh.size() / 2], bytes, rows };
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
    }

    Scene scene;
    Build(scene, options.instances);
    Uploader uploader;
    uploader.BeforeRefresh(scene);
    scene.graph->Refresh(++scene.frame);
    uploader.Upload(scene);

    const uint32_t caseCount = uint32_t(Case::Count);
    std::vector<std::vector<Sample>> samples(caseCount);
    for (uint32_t round = 0; round < options.rounds; ++round)
    {
        // alternate the order of the cases between rounds
        for (uint32_t k = 0; k < caseCount; ++k)
        {
            const uint32_t c = (round & 1) ? caseCount - 1 - k : k;
            samples[c].push_back(Run(scene, uploader, Case(c), options.frames));
        }
    }

    std::printf("%s, %u instances, %u rounds of %u frames (median of run medians)\n",
#if BENCH_GPU_SCENE_E0
                "E0 RefreshBuffers instance path",
#else
                "GpuScene Update + upload",
#endif
                options.instances, options.rounds, options.frames);
    std::printf("%-14s %12s %12s %12s %12s\n", "case", "upload ms", "refresh ms", "rows", "bytes");
    for (uint32_t c = 0; c < caseCount; ++c)
    {
        std::vector<double> upload, refresh;
        for (const Sample& sample : samples[c])
        {
            upload.push_back(sample.uploadMs);
            refresh.push_back(sample.refreshMs);
        }
        std::sort(upload.begin(), upload.end());
        std::sort(refresh.begin(), refresh.end());
        std::printf("%-14s %12.4f %12.4f %12llu %12llu\n", c_CaseNames[c], upload[upload.size() / 2],
                    refresh[refresh.size() / 2], (unsigned long long)samples[c].back().rows,
                    (unsigned long long)samples[c].back().bytes);
    }
    return 0;
}
