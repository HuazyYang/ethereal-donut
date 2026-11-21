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

// The instance, geometry and material rows of a scene in a canonical form, for the A/B comparison of plan step E4
// ("buffers equal to the E0 build's after slot remapping", design 11.2). Private and header-only, and written against
// the API that the E0 build has as well (SceneGraphWalker, GetInstanceIndex, GetGeometryInstanceIndex,
// globalGeometryIndex, materialID), so that gpu_scene_dump.cpp, which reads the buffers back from the GPU, builds
// against both trees. Include it after donut/shaders/bindless.h (with donut::math visible).
//
// Remapping: instances are numbered in pre-order of the graph (i<n>); meshes by first use in that order, a skinned
// mesh's prototype right after it (m<n>); geometries as m<n>.<i>; materials by first use over those geometries (mt<n>);
// vertex and index buffer descriptors by first use over the geometry rows (d<n>). Raw slots that do not match their
// object's own index are printed as such ("?" and the raw value), so a broken mapping shows. Floats are bit patterns.

#pragma once

#include <donut/engine/SceneGraph.h>

#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace donut::engine::gpu_scene_dump
{
    inline uint32_t Bits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    inline void PrintFloats(FILE* out, const char* label, const float* values, size_t count)
    {
        std::fprintf(out, " %s=", label);
        for (size_t i = 0; i < count; ++i)
            std::fprintf(out, "%s%08x", i ? "," : "", Bits(values[i]));
    }

    template <typename T>
    uint32_t Ordinal(std::map<T, uint32_t>& ordinals, std::vector<T>& order, T key)
    {
        auto it = ordinals.find(key);
        if (it != ordinals.end())
            return it->second;
        const uint32_t ordinal = uint32_t(order.size());
        ordinals.emplace(key, ordinal);
        order.push_back(key);
        return ordinal;
    }

    inline void Dump(FILE* out, uint32_t frameIndex, SceneGraph* graph,
                     const InstanceData* instances, size_t instanceCount,
                     const GeometryData* geometries, size_t geometryCount,
                     const MaterialConstants* materials, size_t materialCount)
    {
        std::map<const MeshInfo*, uint32_t> meshOrdinals;
        std::vector<const MeshInfo*> meshOrder;
        std::map<const Material*, uint32_t> materialOrdinals;
        std::vector<const Material*> materialOrder;
        std::map<int, uint32_t> descriptorOrdinals;
        std::vector<int> descriptorOrder;

        std::vector<const MeshInstance*> order;
        if (graph && graph->GetRootNode())
        {
            for (SceneGraphWalker walker(graph->GetRootNode()); walker; walker.Next(true))
            {
                if (const MeshInstance* instance = query_cast<MeshInstance>(walker->GetLeaf()))
                    order.push_back(instance);
            }
        }

        std::fprintf(out, "frame %u instances %zu meshes %zu materials %zu\n", frameIndex, order.size(),
                     graph ? graph->GetMeshes().size() : size_t(0), graph ? graph->GetMaterials().size() : size_t(0));

        // Instances; geometry-instance ranges must not overlap.
        std::map<int, std::pair<int, uint32_t>> ranges;   // first geometry-instance -> (end, instance ordinal)
        for (uint32_t i = 0; i < uint32_t(order.size()); ++i)
        {
            const MeshInstance* instance = order[i];
            const MeshInfo* mesh = instance->GetMesh();
            const int slot = instance->GetInstanceIndex();
            std::fprintf(out, "i%u %s", i, instance->GetNode() ? instance->GetNode()->GetPath().generic_string().c_str() : "?");
            if (mesh)
            {
                const uint32_t m = Ordinal(meshOrdinals, meshOrder, mesh);
                if (mesh->skinPrototype)
                    (void)Ordinal(meshOrdinals, meshOrder, static_cast<const MeshInfo*>(mesh->skinPrototype.Get()));
                std::fprintf(out, " mesh=m%u", m);
            }
            if (slot < 0 || size_t(slot) >= instanceCount)
            {
                std::fprintf(out, " slot?%d\n", slot);
                continue;
            }
            const InstanceData& row = instances[slot];
            std::fprintf(out, " flags=%x geometries=%u", row.flags, row.numGeometries);
            if (mesh && !mesh->geometries.empty() && int(row.firstGeometryIndex) == mesh->geometries[0]->globalGeometryIndex)
                std::fprintf(out, " firstGeometry=m%u.0", meshOrdinals[mesh]);
            else if (row.numGeometries != 0)
                std::fprintf(out, " firstGeometry?%u", row.firstGeometryIndex);
            if (int(row.firstGeometryInstanceIndex) == instance->GetGeometryInstanceIndex())
            {
                const int first = int(row.firstGeometryInstanceIndex);
                const int end = first + int(row.numGeometries);
                bool overlap = false;
                if (row.numGeometries != 0)
                {
                    auto next = ranges.lower_bound(first);
                    overlap |= next != ranges.end() && next->first < end;
                    overlap |= next != ranges.begin() && std::prev(next)->second.first > first;
                    ranges.emplace(first, std::make_pair(end, i));
                }
                std::fprintf(out, " geometryInstances=%s", overlap ? "overlap" : "own");
            }
            else
                std::fprintf(out, " geometryInstances?%u", row.firstGeometryInstanceIndex);
            PrintFloats(out, "T", &row.transform[0][0], 12);
            PrintFloats(out, "P", &row.prevTransform[0][0], 12);
            std::fprintf(out, "\n");
        }

        // Meshes and their geometry rows; materials by first use.
        for (uint32_t m = 0; m < uint32_t(meshOrder.size()); ++m)
        {
            const MeshInfo* mesh = meshOrder[m];
            std::fprintf(out, "m%u name=%s geometries=%zu\n", m, mesh->name.c_str(), mesh->geometries.size());
            for (size_t g = 0; g < mesh->geometries.size(); ++g)
            {
                const MeshGeometry* geometry = mesh->geometries[g];
                std::fprintf(out, "m%u.%zu", m, g);
                if (geometry->material)
                    std::fprintf(out, " material=mt%u", Ordinal(materialOrdinals, materialOrder,
                                                               static_cast<const Material*>(geometry->material.Get())));
                const int slot = geometry->globalGeometryIndex;
                if (geometryCount == 0)
                {
                    std::fprintf(out, "\n");
                    continue;
                }
                if (slot < 0 || size_t(slot) >= geometryCount)
                {
                    std::fprintf(out, " slot?%d\n", slot);
                    continue;
                }
                const GeometryData& row = geometries[slot];
                auto descriptor = [&](int index) -> std::string
                {
                    if (index < 0)
                        return "-1";
                    return "d" + std::to_string(Ordinal(descriptorOrdinals, descriptorOrder, index));
                };
                std::fprintf(out, " indices=%u vertices=%u ib=%s ioff=%u vb=%s pos=%x prev=%x uv1=%x uv2=%x n=%x t=%x r=%x",
                             row.numIndices, row.numVertices, descriptor(row.indexBufferIndex).c_str(), row.indexOffset,
                             descriptor(row.vertexBufferIndex).c_str(), row.positionOffset, row.prevPositionOffset,
                             row.texCoord1Offset, row.texCoord2Offset, row.normalOffset, row.tangentOffset,
                             row.curveRadiusOffset);
                if (geometry->material && int(row.materialIndex) == geometry->material->materialID)
                    std::fprintf(out, " materialIndex=mt%u\n", materialOrdinals[geometry->material.Get()]);
                else
                    std::fprintf(out, " materialIndex?%u\n", row.materialIndex);
            }
        }

        for (uint32_t m = 0; m < uint32_t(materialOrder.size()); ++m)
        {
            const Material* material = materialOrder[m];
            std::fprintf(out, "mt%u name=%s", m, material->name.c_str());
            const int slot = material->materialID;
            if (slot < 0 || size_t(slot) >= materialCount)
            {
                std::fprintf(out, " slot?%d\n", slot);
                continue;
            }
            MaterialConstants row = materials[slot];
            std::fprintf(out, " id=%s", row.materialID == slot ? "own" : "?");
            row.materialID = 0;
            uint32_t words[sizeof(MaterialConstants) / 4];
            std::memcpy(words, &row, sizeof(words));
            std::fprintf(out, " words=");
            for (size_t w = 0; w < std::size(words); ++w)
                std::fprintf(out, "%s%08x", w ? "," : "", words[w]);
            std::fprintf(out, "\n");
        }
    }
}
