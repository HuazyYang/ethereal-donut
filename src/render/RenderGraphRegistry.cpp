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

// The registry (design 4): what survives between frames. This step (P1.3) has the compile side: persistent and
// history records, and the bounded plan cache (7.7). Pools, heaps, fences and retirement come with the executor.

#include "RenderGraphInternal.h"

#include <algorithm>

namespace donut::render::rdg
{
    ResourceRegistry::ResourceRegistry(nvrhi::IDevice* device)
        : m_Device(device)
        , m_DeviceReference(device)
    {
    }

    ResourceRegistry::~ResourceRegistry()
    {
        WaitForFrames();
        // Everything that holds device objects goes before the device reference does.
        m_Runtime.lastPlan = nullptr;
        m_Plans.clear();
        m_Persistent.clear();
        m_Runtime = RegistryRuntime();
    }

    void ResourceRegistry::WaitForFrames() noexcept
    {
        if (!m_Device || m_Runtime.framesExecuted == 0)
            return;
        if (m_Runtime.fencesSupported)
        {
            nvrhi::FenceValue values[3];
            uint32_t count = 0;
            for (uint32_t q = 0; q < 3; ++q)
            {
                if (m_Runtime.fences[q] && m_Runtime.fenceValues[q])
                    values[count++] = nvrhi::FenceValue{ m_Runtime.fences[q], m_Runtime.fenceValues[q] };
            }
            if (count)
                m_Device->waitFences(values, count, true);
        }
        else if (m_Runtime.frameQuery && m_Runtime.frameQueryPending)
        {
            m_Device->waitEventQuery(m_Runtime.frameQuery);
            m_Runtime.frameQueryPending = false;
        }
    }

    void ResourceRegistry::ReleaseResources() noexcept
    {
        // After the GPU has finished with them (4). The persistent resources stay, with exact states.
        WaitForFrames();
        FinalizeLastPlan(*this);
        ClearPlanCache();
        m_Runtime.instances.clear();
        m_Runtime.freePool.clear();
        m_Runtime.extractPool.clear();
        // 13.1: the passes drop the framebuffers and binding sets that point at those resources. Cached plans hold no
        // pass, so the passes reached are those of the graph's current frame, each object once.
        if (m_Graph)
        {
            for (uint32_t index = 0; index < m_Graph->PassCount(); ++index)
            {
                IGpuPass* pass = m_Graph->GetPass(index).pass;
                bool seen = false;
                for (uint32_t earlier = 0; earlier < index && !seen; ++earlier)
                    seen = m_Graph->GetPass(earlier).pass == pass;
                if (pass && !seen)
                    pass->OnResourcesReleased();
            }
        }
    }

    nvrhi::FRESULT ResourceRegistry::ReleasePersistent(const char* name) noexcept
    {
        if (!name)
            return nvrhi::FE_INVALID_ARGS;
        const uint64_t hash = HashString(name);
        for (PersistentEntry& entry : m_Persistent)
        {
            if (entry.live && entry.nameHash == hash && entry.name == name)
            {
                // Entries keep their index: plans name them. A released entry is reallocated on its next use; frames
                // in flight keep the old textures alive through their command lists.
                entry.live = false;
                entry.physicals.clear();
                return nvrhi::FS_OK;
            }
        }
        return nvrhi::FE_NOT_FOUND;
    }

    void ResourceRegistry::ClearPlanCache() noexcept
    {
        m_Plans.clear();
    }

    void ResourceRegistry::GetRegistryStatistics(RegistryStatistics& out) noexcept
    {
        out = RegistryStatistics();
        out.cachedPlans = uint32_t(m_Plans.size());
        out.planHits = m_PlanHits;
        out.planMisses = m_PlanMisses;
        for (const PersistentEntry& entry : m_Persistent)
            out.persistentResources += entry.live ? 1 : 0;
        out.pooledResources = uint32_t(m_Runtime.freePool.size());
        out.framesExecuted = m_Runtime.framesExecuted;
        out.objectsCreated = m_Runtime.objectsCreated;
    }

    void ResourceRegistry::GetRunStatistics(RunStatistics& out) noexcept
    {
        out = m_Runtime.run;
    }

    uint32_t ResourceRegistry::GetPassTimingCount() noexcept
    {
        return uint32_t(m_Runtime.timings.size());
    }

    nvrhi::FRESULT ResourceRegistry::GetPassTiming(uint32_t index, PassTiming& out) noexcept
    {
        if (index >= m_Runtime.timings.size())
            return nvrhi::FE_INVALID_ARGS;
        out = m_Runtime.timings[index];
        return nvrhi::FS_OK;
    }

    uint32_t ResourceRegistry::AcquirePersistent(const char* name, uint8_t kind, uint64_t descHash, uint8_t layerCount,
                                                 uint64_t frameIndex, uint32_t& generation, bool& kindMismatch)
    {
        kindMismatch = false;
        const uint64_t hash = HashString(name);
        for (uint32_t index = 0; index < uint32_t(m_Persistent.size()); ++index)
        {
            PersistentEntry& entry = m_Persistent[index];
            if (entry.nameHash != hash || entry.name != name)
                continue;
            if (entry.live && entry.kind != kind)
            {
                kindMismatch = true;
                generation = entry.generation;
                return index;
            }
            // A new description, layer count or a release reallocates (5.2).
            if (!entry.live || entry.kind != kind || entry.descHash != descHash || entry.layerCount != layerCount)
            {
                entry.live = true;
                entry.kind = kind;
                entry.descHash = descHash;
                entry.layerCount = layerCount;
                ++entry.generation;
                entry.allocatedFrame = frameIndex;
            }
            entry.lastFrame = frameIndex;
            generation = entry.generation;
            return index;
        }

        PersistentEntry entry;
        entry.name = name;
        entry.nameHash = hash;
        entry.kind = kind;
        entry.layerCount = layerCount;
        entry.descHash = descHash;
        entry.generation = 1;
        entry.allocatedFrame = frameIndex;
        entry.lastFrame = frameIndex;
        m_Persistent.push_back(std::move(entry));
        generation = 1;
        return uint32_t(m_Persistent.size() - 1);
    }

    CompiledRenderGraph* ResourceRegistry::FindPlan(uint64_t key, uint64_t maskHash, uint32_t passCount,
                                                    uint32_t resourceCount, uint32_t accessCount, uint64_t frameIndex) noexcept
    {
        for (PlanCacheEntry& entry : m_Plans)
        {
            // The counts guard against a hash collision between graphs of different shapes.
            if (entry.key == key && entry.maskHash == maskHash && entry.passCount == passCount &&
                entry.resourceCount == resourceCount && entry.accessCount == accessCount)
            {
                entry.lastUsed = frameIndex;
                ++m_PlanHits;
                return entry.plan.Get();
            }
        }
        return nullptr;
    }

    void ResourceRegistry::InsertPlan(CompiledRenderGraph* plan, uint32_t passCount, uint32_t resourceCount,
                                      uint32_t accessCount, uint32_t maxCachedPlans, uint64_t frameIndex)
    {
        if (maxCachedPlans == 0)
            return;
        while (m_Plans.size() >= maxCachedPlans)
        {
            // Least recently used (7.7).
            auto oldest = std::min_element(m_Plans.begin(), m_Plans.end(),
                [](const PlanCacheEntry& a, const PlanCacheEntry& b) { return a.lastUsed < b.lastUsed; });
            m_Plans.erase(oldest);
        }
        PlanCacheEntry entry;
        entry.key = plan->m_Key;
        entry.maskHash = plan->m_MaskHash;
        entry.passCount = passCount;
        entry.resourceCount = resourceCount;
        entry.accessCount = accessCount;
        entry.lastUsed = frameIndex;
        entry.plan = plan;
        m_Plans.push_back(std::move(entry));
    }

    std::shared_ptr<const StructurePlan> ResourceRegistry::FindStructure(uint64_t key) const noexcept
    {
        // Every mask of one graph shares its structure and memory plan (7.7, 13.5).
        for (const PlanCacheEntry& entry : m_Plans)
        {
            if (entry.key == key && entry.plan->m_Structure)
                return entry.plan->m_Structure;
        }
        return nullptr;
    }
}

namespace donut::render
{
    nvrhi::FRESULT CreateResourceRegistry(nvrhi::IDevice* device, IResourceRegistry** out) noexcept
    {
        if (!out)
            return nvrhi::FE_INVALID_ARGS;
        *out = nullptr;
        try
        {
            *out = MAKE_RC_OBJ(rdg::ResourceRegistry, device);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        return nvrhi::FS_OK;
    }
}
