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

// Graph recording (design 4, 5, 12): resources, passes and the setup context, the frame arena and the structural
// hash. Nothing here allocates from the heap once the arena's blocks have grown to the frame's size.

#include "RenderGraphInternal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace donut::render::rdg
{
    // ------------------------------------------------------------------------------------------------ arena

    FrameArena::~FrameArena()
    {
        for (const Block& block : m_Blocks)
            ::operator delete(block.data);
    }

    void* FrameArena::Allocate(size_t size, size_t alignment) noexcept
    {
        if (alignment == 0)
            alignment = 1;
        if (size == 0)
            size = 1;

        while (m_Current < m_Blocks.size())
        {
            Block& block = m_Blocks[m_Current];
            const uintptr_t base = reinterpret_cast<uintptr_t>(block.data);
            const uintptr_t aligned = (base + m_Offset + alignment - 1) & ~uintptr_t(alignment - 1);
            const size_t end = size_t(aligned - base) + size;
            if (end <= block.size)
            {
                m_Offset = end;
                return reinterpret_cast<void*>(aligned);
            }
            ++m_Current;
            m_Offset = 0;
        }

        // A new block, large enough for this request.
        const size_t blockSize = std::max(c_BlockSize, size + alignment);
        char* data = static_cast<char*>(::operator new(blockSize, std::nothrow));
        if (!data)
            return nullptr;
        try
        {
            m_Blocks.push_back(Block{ data, blockSize });
        }
        catch (...)
        {
            ::operator delete(data);
            return nullptr;
        }
        m_Current = m_Blocks.size() - 1;
        m_Offset = 0;
        return Allocate(size, alignment);
    }

    void FrameArena::Reset() noexcept
    {
        m_Current = 0;
        m_Offset = 0;
    }

    // ------------------------------------------------------------------------------------------------ descriptions

    TextureInfo MakeTextureInfo(const nvrhi::TextureDesc& desc) noexcept
    {
        TextureInfo info{};
        info.width = desc.width;
        info.height = desc.height;
        info.depth = desc.depth;
        info.arraySize = desc.arraySize;
        info.mipLevels = std::max(1u, desc.mipLevels);
        info.sampleCount = desc.sampleCount;
        info.sampleQuality = desc.sampleQuality;
        info.format = desc.format;
        info.dimension = desc.dimension;
        info.isShaderResource = desc.isShaderResource;
        info.isRenderTarget = desc.isRenderTarget;
        info.isUAV = desc.isUAV;
        info.isTypeless = desc.isTypeless;
        info.isShadingRateSurface = desc.isShadingRateSurface;
        info.useClearValue = desc.useClearValue;
        info.keepInitialState = desc.keepInitialState;
        info.clearValue = desc.clearValue;
        info.initialState = desc.initialState;
        info.sharedResourceFlags = desc.sharedResourceFlags;
        return info;
    }

    BufferInfo MakeBufferInfo(const nvrhi::BufferDesc& desc) noexcept
    {
        BufferInfo info{};
        info.byteSize = desc.byteSize;
        info.structStride = desc.structStride;
        info.format = desc.format;
        info.canHaveUAVs = desc.canHaveUAVs;
        info.canHaveTypedViews = desc.canHaveTypedViews;
        info.canHaveRawViews = desc.canHaveRawViews;
        info.isVertexBuffer = desc.isVertexBuffer;
        info.isIndexBuffer = desc.isIndexBuffer;
        info.isConstantBuffer = desc.isConstantBuffer;
        info.isDrawIndirectArgs = desc.isDrawIndirectArgs;
        info.isAccelStructBuildInput = desc.isAccelStructBuildInput;
        info.isAccelStructStorage = desc.isAccelStructStorage;
        info.isShaderBindingTable = desc.isShaderBindingTable;
        info.isVolatile = desc.isVolatile;
        info.keepInitialState = desc.keepInitialState;
        info.initialState = desc.initialState;
        info.cpuAccess = desc.cpuAccess;
        return info;
    }

    // The fields that change the created object (the pool key of 7.5): not the name, the initial state or
    // keepInitialState.
    uint64_t HashTextureInfo(const TextureInfo& info) noexcept
    {
        uint64_t hash = c_HashSeed;
        hash = HashCombine(hash, info.width);
        hash = HashCombine(hash, info.height);
        hash = HashCombine(hash, info.depth);
        hash = HashCombine(hash, info.arraySize);
        hash = HashCombine(hash, info.mipLevels);
        hash = HashCombine(hash, uint64_t(info.sampleCount) | (uint64_t(info.sampleQuality) << 32));
        hash = HashCombine(hash, uint64_t(info.format) | (uint64_t(info.dimension) << 8));
        hash = HashCombine(hash, uint64_t(info.isShaderResource) | (uint64_t(info.isRenderTarget) << 1) |
            (uint64_t(info.isUAV) << 2) | (uint64_t(info.isTypeless) << 3) | (uint64_t(info.isShadingRateSurface) << 4) |
            (uint64_t(info.useClearValue) << 5));
        if (info.useClearValue)
        {
            uint32_t bits[4];
            std::memcpy(bits, &info.clearValue, sizeof(bits));
            hash = HashCombine(hash, uint64_t(bits[0]) | (uint64_t(bits[1]) << 32));
            hash = HashCombine(hash, uint64_t(bits[2]) | (uint64_t(bits[3]) << 32));
        }
        hash = HashCombine(hash, uint64_t(info.sharedResourceFlags));
        return hash;
    }

    uint64_t HashBufferInfo(const BufferInfo& info) noexcept
    {
        uint64_t hash = HashCombine(c_HashSeed, 0xb0ffull);
        hash = HashCombine(hash, info.byteSize);
        hash = HashCombine(hash, uint64_t(info.structStride) | (uint64_t(info.format) << 32));
        hash = HashCombine(hash, uint64_t(info.canHaveUAVs) | (uint64_t(info.canHaveTypedViews) << 1) |
            (uint64_t(info.canHaveRawViews) << 2) | (uint64_t(info.isVertexBuffer) << 3) |
            (uint64_t(info.isIndexBuffer) << 4) | (uint64_t(info.isConstantBuffer) << 5) |
            (uint64_t(info.isDrawIndirectArgs) << 6) | (uint64_t(info.isAccelStructBuildInput) << 7) |
            (uint64_t(info.isAccelStructStorage) << 8) | (uint64_t(info.isShaderBindingTable) << 9) |
            (uint64_t(info.isVolatile) << 10));
        hash = HashCombine(hash, uint64_t(info.cpuAccess));
        return hash;
    }

    static bool IsWholeRegion(const nvrhi::SubresourceRegion& region) noexcept
    {
        return region.baseMip == 0 && region.mipCount == nvrhi::SubresourceRegion::AllMipLevels &&
            region.baseSlice == 0 && region.sliceCount == nvrhi::SubresourceRegion::AllArraySlices &&
            region.planes == nvrhi::PlaneMask::All;
    }

    static constexpr Access c_TextureOnlyAccess = Access::ColorTarget | Access::ColorTargetRead | Access::DepthRead |
        Access::DepthWrite | Access::StencilRead | Access::StencilWrite | Access::ShadingRateRead |
        Access::ResolveSource | Access::ResolveDest | Access::Present;

    // ------------------------------------------------------------------------------------------------ graph

    RenderGraph::RenderGraph(ResourceRegistry* registry)
        : m_Registry(registry)
    {
        m_Registry->m_HasGraph = true;
    }

    RenderGraph::~RenderGraph()
    {
        for (uint32_t index = 0; index < m_Passes.size; ++index)
        {
            if (m_Passes[index].pass)
                m_Passes[index].pass->Release();
        }
        m_Passes.Clear();
        m_Plan = nullptr;
        m_Registry->m_HasGraph = false;
    }

    void RenderGraph::Reset() noexcept
    {
        // The passes go first: lambda passes live in the arena and free nothing, but their destructors must run
        // while the arena still holds them.
        for (uint32_t index = 0; index < m_Passes.size; ++index)
        {
            if (m_Passes[index].pass)
                m_Passes[index].pass->Release();
        }

        m_Resources.Clear();
        m_Passes.Clear();
        m_Accesses.Clear();
        m_Policies.Clear();
        m_Markers.Clear();
        m_Errors.Clear();
        m_Arena.Reset();

        m_StructuralHash = c_HashSeed;
        m_OpenPass = c_None;
        m_InSetup = false;
        m_Frozen = false;
        m_Plan = nullptr;
        ++m_FrameIndex;
    }

    void RenderGraph::RecordErrorAt(RecordError code, uint32_t resource, uint32_t other, uint64_t value) noexcept
    {
        ErrorRecord* error = m_Errors.Push(m_Arena);
        if (!error)
            return;
        error->code = code;
        error->pass = m_InSetup ? m_OpenPass : c_None;
        error->resource = resource;
        error->other = other;
        error->value = value;
    }

    const char* RenderGraph::CopyString(const char* text) noexcept
    {
        if (!text || !*text)
            return nullptr;
        const size_t length = std::strlen(text);
        char* copy = static_cast<char*>(m_Arena.Allocate(length + 1, 1));
        if (!copy)
            return nullptr;
        std::memcpy(copy, text, length + 1);
        return copy;
    }

    bool RenderGraph::CheckNotFrozen() noexcept
    {
        if (m_Frozen)
        {
            RecordErrorAt(RecordError::Frozen);
            return false;
        }
        return true;
    }

    bool RenderGraph::CheckOpen() noexcept
    {
        if (!m_InSetup)
        {
            RecordErrorAt(RecordError::SetupContextClosed);
            return false;
        }
        return CheckNotFrozen();
    }

    ResourceRecord* RenderGraph::NewResource(ResourceKind kind, bool isBuffer) noexcept
    {
        ResourceRecord* record = m_Resources.Push(m_Arena);
        if (!record)
        {
            RecordErrorAt(RecordError::ArenaExhausted);
            return nullptr;
        }
        record->kind = kind;
        record->isBuffer = isBuffer;
        record->currentVersion = m_NextVersion++;
        if (m_NextVersion == ~0u)
            m_NextVersion = 1;
        record->currentWriter = c_None;
        record->ownerPass = c_None;
        record->registryEntry = c_None;
        record->entry = nvrhi::ResourceStates::Unknown;
        record->exit = nvrhi::ResourceStates::Unknown;
        return record;
    }

    ResourceId RenderGraph::AddTextureResource(ResourceKind kind, const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept
    {
        ResourceRecord* record = NewResource(kind, false);
        if (!record)
            return InvalidResourceId;
        const uint32_t index = m_Resources.size - 1;
        record->flags = flags;
        record->texture = MakeTextureInfo(desc);
        record->descHash = HashTextureInfo(record->texture);
        record->name = CopyString(desc.debugName.c_str());
        const nvrhi::FormatInfo& format = nvrhi::getFormatInfo(desc.format);
        record->hasDepth = format.hasDepth;
        record->hasStencil = format.hasStencil;
        Hash(HashCombine(uint64_t(kind) | 0x100, record->descHash));
        Hash(uint64_t(flags));
        if ((kind == ResourceKind::Transient || kind == ResourceKind::PassLocal) && desc.keepInitialState)
            RecordErrorAt(RecordError::KeepInitialStateTransient, index);
        return MakeResourceId(index, record->currentVersion);
    }

    ResourceId RenderGraph::AddBufferResource(ResourceKind kind, const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept
    {
        ResourceRecord* record = NewResource(kind, true);
        if (!record)
            return InvalidResourceId;
        const uint32_t index = m_Resources.size - 1;
        record->flags = flags;
        record->buffer = MakeBufferInfo(desc);
        record->descHash = HashBufferInfo(record->buffer);
        record->name = CopyString(desc.debugName.c_str());
        Hash(HashCombine(uint64_t(kind) | 0x200, record->descHash));
        Hash(uint64_t(flags));
        if ((kind == ResourceKind::Transient || kind == ResourceKind::PassLocal) && desc.keepInitialState)
            RecordErrorAt(RecordError::KeepInitialStateTransient, index);
        return MakeResourceId(index, record->currentVersion);
    }

    ResourceId RenderGraph::CreateTexture(const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        return AddTextureResource(ResourceKind::Transient, desc, flags);
    }

    ResourceId RenderGraph::CreateBuffer(const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        return AddBufferResource(ResourceKind::Transient, desc, flags);
    }

    ResourceId RenderGraph::ImportTexture(nvrhi::ITexture* texture, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        if (!texture)
        {
            RecordErrorAt(RecordError::NullImport);
            return InvalidResourceId;
        }
        const nvrhi::TextureDesc& desc = texture->getDesc();
        const ResourceId id = AddTextureResource(ResourceKind::Imported, desc, ResourceFlags::None);
        if (id == InvalidResourceId)
            return id;
        ResourceRecord& record = m_Resources[GetResourceIndex(id)];
        record.imported = texture;
        record.entry = entry;
        record.exit = exit;
        Hash(HashCombine(uint64_t(entry), uint64_t(exit)));
        if (desc.keepInitialState)
            RecordErrorAt(RecordError::TrackedImportOfHostKept, GetResourceIndex(id));
        return id;
    }

    ResourceId RenderGraph::ImportHostKeptTexture(nvrhi::ITexture* texture) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        if (!texture)
        {
            RecordErrorAt(RecordError::NullImport);
            return InvalidResourceId;
        }
        const nvrhi::TextureDesc& desc = texture->getDesc();
        const ResourceId id = AddTextureResource(ResourceKind::HostKept, desc, ResourceFlags::None);
        if (id == InvalidResourceId)
            return id;
        ResourceRecord& record = m_Resources[GetResourceIndex(id)];
        record.imported = texture;
        record.entry = desc.initialState;
        record.exit = desc.initialState;
        Hash(uint64_t(desc.initialState));
        return id;
    }

    ResourceId RenderGraph::ImportBuffer(nvrhi::IBuffer* buffer, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        if (!buffer)
        {
            RecordErrorAt(RecordError::NullImport);
            return InvalidResourceId;
        }
        const nvrhi::BufferDesc& desc = buffer->getDesc();
        const ResourceId id = AddBufferResource(ResourceKind::Imported, desc, ResourceFlags::None);
        if (id == InvalidResourceId)
            return id;
        ResourceRecord& record = m_Resources[GetResourceIndex(id)];
        record.imported = buffer;
        record.entry = entry;
        record.exit = exit;
        Hash(HashCombine(uint64_t(entry), uint64_t(exit)));
        if (desc.keepInitialState)
            RecordErrorAt(RecordError::TrackedImportOfHostKept, GetResourceIndex(id));
        return id;
    }

    ResourceId RenderGraph::ImportTextureDesc(const nvrhi::TextureDesc& desc, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        // A keepInitialState stand-in is host-kept, like ImportHostKeptTexture.
        const ResourceKind kind = desc.keepInitialState ? ResourceKind::HostKept : ResourceKind::Imported;
        const ResourceId id = AddTextureResource(kind, desc, ResourceFlags::None);
        if (id == InvalidResourceId)
            return id;
        ResourceRecord& record = m_Resources[GetResourceIndex(id)];
        record.standIn = true;
        record.entry = desc.keepInitialState ? desc.initialState : entry;
        record.exit = desc.keepInitialState ? desc.initialState : exit;
        Hash(HashCombine(uint64_t(record.entry), uint64_t(record.exit)));
        return id;
    }

    ResourceId RenderGraph::ImportBufferDesc(const nvrhi::BufferDesc& desc, nvrhi::ResourceStates entry, nvrhi::ResourceStates exit) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        const ResourceKind kind = desc.keepInitialState ? ResourceKind::HostKept : ResourceKind::Imported;
        const ResourceId id = AddBufferResource(kind, desc, ResourceFlags::None);
        if (id == InvalidResourceId)
            return id;
        ResourceRecord& record = m_Resources[GetResourceIndex(id)];
        record.standIn = true;
        record.entry = desc.keepInitialState ? desc.initialState : entry;
        record.exit = desc.keepInitialState ? desc.initialState : exit;
        Hash(HashCombine(uint64_t(record.entry), uint64_t(record.exit)));
        return id;
    }

    ResourceId RenderGraph::GetPersistentTexture(const char* name, const nvrhi::TextureDesc& desc) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        if (!name || !*name)
        {
            RecordErrorAt(RecordError::NullName);
            return InvalidResourceId;
        }
        const ResourceId id = AddTextureResource(ResourceKind::Persistent, desc, ResourceFlags::None);
        if (id == InvalidResourceId)
            return id;
        ResourceRecord& record = m_Resources[GetResourceIndex(id)];
        bool mismatch = false;
        uint32_t generation = 0;
        try
        {
            record.registryEntry = m_Registry->AcquirePersistent(name, 0, record.descHash, 0, m_FrameIndex, generation, mismatch);
        }
        catch (...)
        {
            RecordErrorAt(RecordError::ArenaExhausted);
            return InvalidResourceId;
        }
        record.generation = generation;
        if (!record.name)
            record.name = CopyString(name);
        Hash(HashString(name));
        if (mismatch)
            RecordErrorAt(RecordError::PersistentKindMismatch, GetResourceIndex(id));
        return id;
    }

    ResourceId RenderGraph::GetPersistentBuffer(const char* name, const nvrhi::BufferDesc& desc) noexcept
    {
        if (!CheckNotFrozen())
            return InvalidResourceId;
        if (!name || !*name)
        {
            RecordErrorAt(RecordError::NullName);
            return InvalidResourceId;
        }
        const ResourceId id = AddBufferResource(ResourceKind::Persistent, desc, ResourceFlags::None);
        if (id == InvalidResourceId)
            return id;
        ResourceRecord& record = m_Resources[GetResourceIndex(id)];
        bool mismatch = false;
        uint32_t generation = 0;
        try
        {
            record.registryEntry = m_Registry->AcquirePersistent(name, 1, record.descHash, 0, m_FrameIndex, generation, mismatch);
        }
        catch (...)
        {
            RecordErrorAt(RecordError::ArenaExhausted);
            return InvalidResourceId;
        }
        record.generation = generation;
        if (!record.name)
            record.name = CopyString(name);
        Hash(HashString(name));
        if (mismatch)
            RecordErrorAt(RecordError::PersistentKindMismatch, GetResourceIndex(id));
        return id;
    }

    nvrhi::FRESULT RenderGraph::GetHistory(const char* name, const nvrhi::TextureDesc& desc, uint32_t layerCount,
                                           ResourceId* ids) noexcept
    {
        if (!ids)
            return nvrhi::FE_INVALID_ARGS;
        for (uint32_t layer = 0; layer < std::min(layerCount, c_MaxHistoryLayers); ++layer)
            ids[layer] = InvalidResourceId;
        if (!CheckNotFrozen())
            return nvrhi::FE_INVALID_ARGS;
        if (layerCount < 2 || layerCount > c_MaxHistoryLayers)
        {
            RecordErrorAt(RecordError::BadLayerCount, c_None, c_None, layerCount);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!name || !*name)
        {
            RecordErrorAt(RecordError::NullName);
            return nvrhi::FE_INVALID_ARGS;
        }

        const uint64_t descHash = HashTextureInfo(MakeTextureInfo(desc));
        bool mismatch = false;
        uint32_t generation = 0;
        uint32_t entry = c_None;
        try
        {
            entry = m_Registry->AcquirePersistent(name, 2, descHash, uint8_t(layerCount), m_FrameIndex, generation, mismatch);
        }
        catch (...)
        {
            RecordErrorAt(RecordError::ArenaExhausted);
            return nvrhi::FE_OUT_OF_MEMORY;
        }

        Hash(HashCombine(HashString(name), layerCount));
        for (uint32_t layer = 0; layer < layerCount; ++layer)
        {
            const ResourceId id = AddTextureResource(ResourceKind::History, desc, ResourceFlags::None);
            if (id == InvalidResourceId)
                return nvrhi::FE_OUT_OF_MEMORY;
            ResourceRecord& record = m_Resources[GetResourceIndex(id)];
            record.registryEntry = entry;
            record.generation = generation;
            record.historyLayer = uint8_t(layer);
            record.historyLayerCount = uint8_t(layerCount);
            if (!record.name)
                record.name = CopyString(name);
            Hash(layer);
            ids[layer] = id;
        }
        if (mismatch)
            RecordErrorAt(RecordError::PersistentKindMismatch, GetResourceIndex(ids[0]));
        return nvrhi::FS_OK;
    }

    void RenderGraph::Extract(ResourceId texture, nvrhi::ResourceStates exit, nvrhi::ITexture** outTexture) noexcept
    {
        if (outTexture)
            *outTexture = nullptr;
        if (!CheckNotFrozen())
            return;
        ResourceRecord* record = CheckId(texture, false);
        if (!record)
            return;
        const uint32_t index = GetResourceIndex(texture);
        if (record->kind != ResourceKind::Transient || record->isBuffer)
        {
            RecordErrorAt(RecordError::ExtractNotTransient, index);
            return;
        }
        record->extracted = true;
        record->exit = exit;
        record->extractOut = outTexture;
        Hash(HashCombine(0xe7ull, HashCombine(index, uint64_t(exit))));
    }

    ResourceRecord* RenderGraph::CheckId(ResourceId id, bool forWrite) noexcept
    {
        const uint32_t index = GetResourceIndex(id);
        if (id == InvalidResourceId || index >= m_Resources.size)
        {
            RecordErrorAt(RecordError::InvalidId, index, c_None, uint64_t(id));
            return nullptr;
        }
        ResourceRecord& record = m_Resources[index];
        if (GetResourceVersion(id) != record.currentVersion)
        {
            RecordErrorAt(RecordError::StaleVersion, index, record.currentWriter, GetResourceVersion(id));
            return nullptr;
        }
        if (record.kind == ResourceKind::PassLocal && record.ownerPass != m_OpenPass)
        {
            RecordErrorAt(RecordError::PassLocalForeign, index, record.ownerPass);
            return nullptr;
        }
        if (forWrite && record.kind == ResourceKind::History && record.historyLayer != 0)
        {
            RecordErrorAt(RecordError::HistoryWrite, index, c_None, record.historyLayer);
            return nullptr;
        }
        return &record;
    }

    AccessRecord* RenderGraph::AddAccess(ResourceRecord* record, uint32_t index, Access access, nvrhi::ShaderType stages,
                                         const nvrhi::SubresourceRegion& region, AccessFlags flags, bool isWrite) noexcept
    {
        if (record->isBuffer && ((access & c_TextureOnlyAccess) != Access::None || !IsWholeRegion(region)))
        {
            RecordErrorAt(RecordError::TextureAccessOnBuffer, index);
            return nullptr;
        }
        AccessRecord* entry = m_Accesses.Push(m_Arena);
        if (!entry)
        {
            RecordErrorAt(RecordError::ArenaExhausted);
            return nullptr;
        }
        entry->resource = index;
        entry->version = record->currentVersion;
        entry->prevVersion = record->currentVersion;
        entry->access = access;
        entry->stages = stages;
        entry->flags = flags;
        entry->region = record->isBuffer ? nvrhi::SubresourceRegion() : region;
        entry->attachment = c_NoAttachment;
        entry->isWrite = isWrite;
        return entry;
    }

    // ------------------------------------------------------------------------------------------------ setup context

    ResourceId RenderGraph::CreateTransientTexture(const nvrhi::TextureDesc& desc, ResourceFlags flags) noexcept
    {
        if (!CheckOpen())
            return InvalidResourceId;
        const ResourceId id = AddTextureResource(ResourceKind::PassLocal, desc, flags);
        if (id != InvalidResourceId)
            m_Resources[GetResourceIndex(id)].ownerPass = m_OpenPass;
        return id;
    }

    ResourceId RenderGraph::CreateTransientBuffer(const nvrhi::BufferDesc& desc, ResourceFlags flags) noexcept
    {
        if (!CheckOpen())
            return InvalidResourceId;
        const ResourceId id = AddBufferResource(ResourceKind::PassLocal, desc, flags);
        if (id != InvalidResourceId)
            m_Resources[GetResourceIndex(id)].ownerPass = m_OpenPass;
        return id;
    }

    static uint64_t HashAccess(const AccessRecord& access) noexcept
    {
        uint64_t hash = HashCombine(c_HashSeed, uint64_t(access.resource) | (uint64_t(access.access) << 32));
        hash = HashCombine(hash, uint64_t(access.stages) | (uint64_t(access.flags) << 16) |
            (uint64_t(access.attachment) << 48) | (uint64_t(access.isWrite) << 56) | (uint64_t(access.hasClear) << 57));
        hash = HashCombine(hash, uint64_t(access.region.baseMip) | (uint64_t(access.region.mipCount) << 32));
        hash = HashCombine(hash, uint64_t(access.region.baseSlice) | (uint64_t(access.region.sliceCount) << 32));
        hash = HashCombine(hash, uint64_t(access.region.planes));
        return hash;
    }

    void RenderGraph::Read(ResourceId resource, Access access, nvrhi::ShaderType stages, const nvrhi::SubresourceRegion& region) noexcept
    {
        if (!CheckOpen())
            return;
        ResourceRecord* record = CheckId(resource, false);
        if (!record)
            return;
        const uint32_t index = GetResourceIndex(resource);
        if ((access & Access::AllWrites) != Access::None)
        {
            RecordErrorAt(RecordError::WriteBitsInRead, index);
            return;
        }
        if (AccessRecord* entry = AddAccess(record, index, access, stages, region, AccessFlags::None, false))
            Hash(HashAccess(*entry));
    }

    ResourceId RenderGraph::Write(ResourceId resource, Access access, nvrhi::ShaderType stages,
                                  const nvrhi::SubresourceRegion& region, AccessFlags flags) noexcept
    {
        if (!CheckOpen())
            return InvalidResourceId;
        ResourceRecord* record = CheckId(resource, true);
        if (!record)
            return InvalidResourceId;
        const uint32_t index = GetResourceIndex(resource);
        AccessRecord* entry = AddAccess(record, index, access, stages, region, flags, true);
        if (!entry)
            return InvalidResourceId;
        entry->hasClear = (flags & AccessFlags::Clear) != AccessFlags::None;
        record->currentVersion = m_NextVersion++;
        if (m_NextVersion == ~0u)
            m_NextVersion = 1;
        record->currentWriter = m_OpenPass;
        entry->version = record->currentVersion;
        Hash(HashAccess(*entry));
        return MakeResourceId(index, record->currentVersion);
    }

    ResourceId RenderGraph::SetColorTarget(uint32_t slot, ResourceId texture, const nvrhi::SubresourceRegion& region,
                                           AccessFlags flags, const ClearValue* clear) noexcept
    {
        if (!CheckOpen())
            return InvalidResourceId;
        if (slot >= 8)
        {
            RecordErrorAt(RecordError::BadAttachmentSlot, c_None, c_None, slot);
            return InvalidResourceId;
        }
        if (clear)
            flags |= AccessFlags::Clear;
        ResourceRecord* record = CheckId(texture, true);
        if (!record)
            return InvalidResourceId;
        const uint32_t index = GetResourceIndex(texture);
        AccessRecord* entry = AddAccess(record, index, Access::ColorTarget, nvrhi::ShaderType::None, region, flags, true);
        if (!entry)
            return InvalidResourceId;
        entry->attachment = uint8_t(slot);
        entry->hasClear = (flags & AccessFlags::Clear) != AccessFlags::None;
        if (clear)
            entry->clear = *clear;
        record->currentVersion = m_NextVersion++;
        if (m_NextVersion == ~0u)
            m_NextVersion = 1;
        record->currentWriter = m_OpenPass;
        entry->version = record->currentVersion;
        Hash(HashAccess(*entry));
        return MakeResourceId(index, record->currentVersion);
    }

    ResourceId RenderGraph::SetDepthTarget(ResourceId texture, bool readOnly, AccessFlags flags, const ClearValue* clear) noexcept
    {
        if (!CheckOpen())
            return InvalidResourceId;
        if (clear)
            flags |= AccessFlags::Clear;
        ResourceRecord* record = CheckId(texture, !readOnly);
        if (!record)
            return InvalidResourceId;
        const uint32_t index = GetResourceIndex(texture);
        if (readOnly)
        {
            if ((flags & AccessFlags::Clear) != AccessFlags::None)
            {
                RecordErrorAt(RecordError::ClearOnReadOnlyDepth, index);
                return InvalidResourceId;
            }
            const Access access = record->hasStencil ? Access::DepthRead | Access::StencilRead : Access::DepthRead;
            AccessRecord* entry = AddAccess(record, index, access, nvrhi::ShaderType::None, nvrhi::SubresourceRegion(),
                                            AccessFlags::None, false);
            if (!entry)
                return InvalidResourceId;
            entry->attachment = c_DepthAttachment;
            Hash(HashAccess(*entry));
            return texture;
        }

        const Access access = record->hasStencil ? Access::DepthWrite | Access::StencilWrite : Access::DepthWrite;
        AccessRecord* entry = AddAccess(record, index, access, nvrhi::ShaderType::None, nvrhi::SubresourceRegion(), flags, true);
        if (!entry)
            return InvalidResourceId;
        entry->attachment = c_DepthAttachment;
        entry->hasClear = (flags & AccessFlags::Clear) != AccessFlags::None;
        if (clear)
            entry->clear = *clear;
        record->currentVersion = m_NextVersion++;
        if (m_NextVersion == ~0u)
            m_NextVersion = 1;
        record->currentWriter = m_OpenPass;
        entry->version = record->currentVersion;
        Hash(HashAccess(*entry));
        return MakeResourceId(index, record->currentVersion);
    }

    void RenderGraph::SetDisabledOutput(ResourceId written, DisabledOutput policy, ResourceId substituteOrInput) noexcept
    {
        if (!CheckOpen())
            return;
        const uint32_t index = GetResourceIndex(written);
        const PassRecord& pass = m_Passes[m_OpenPass];
        uint32_t ordinal = c_None;
        for (uint32_t a = pass.firstAccess, count = 0; a < m_Accesses.size; ++a, ++count)
        {
            const AccessRecord& access = m_Accesses[a];
            if (access.isWrite && access.resource == index && access.version == GetResourceVersion(written))
            {
                ordinal = count;
                break;
            }
        }
        if (written == InvalidResourceId || index >= m_Resources.size || ordinal == c_None)
        {
            RecordErrorAt(RecordError::DisabledOutputNotWritten, index < m_Resources.size ? index : c_None);
            return;
        }

        uint32_t substitute = c_None;
        if (policy != DisabledOutput::Cull)
        {
            const uint32_t substituteIndex = GetResourceIndex(substituteOrInput);
            if (substituteOrInput == InvalidResourceId || substituteIndex >= m_Resources.size)
            {
                RecordErrorAt(RecordError::InvalidId, substituteIndex, c_None, uint64_t(substituteOrInput));
                return;
            }
            substitute = substituteIndex;
        }

        PolicyRecord* record = m_Policies.Push(m_Arena);
        if (!record)
        {
            RecordErrorAt(RecordError::ArenaExhausted);
            return;
        }
        record->pass = m_OpenPass;
        record->resource = index;
        record->version = GetResourceVersion(written);
        record->substitute = substitute;
        record->policy = policy;
        Hash(HashCombine(0xd15aull, HashCombine(uint64_t(index) | (uint64_t(ordinal) << 32),
            uint64_t(substitute) | (uint64_t(policy) << 32))));
    }

    // ------------------------------------------------------------------------------------------------ passes

    nvrhi::FRESULT RenderGraph::AddPass(const PassDesc& desc, IGpuPass* pass, void* data, PassId* outPass) noexcept
    {
        if (outPass)
            *outPass = InvalidPassId;
        if (m_InSetup)
        {
            RecordErrorAt(RecordError::NestedAddPass);
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!CheckNotFrozen())
            return nvrhi::FE_INVALID_ARGS;
        if (!pass)
        {
            RecordErrorAt(RecordError::NullPass);
            return nvrhi::FE_INVALID_ARGS;
        }

        PassRecord* record = m_Passes.Push(m_Arena);
        if (!record)
        {
            RecordErrorAt(RecordError::ArenaExhausted);
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        const uint32_t index = m_Passes.size - 1;
        record->name = CopyString(desc.name);
        record->type = desc.type;
        record->flags = desc.flags;
        record->exclusionGroup = desc.exclusionGroup;
        record->nameHash = HashString(desc.name);
        record->pass = pass;
        record->data = data;
        record->firstAccess = m_Accesses.size;
        record->enabled = true;
        pass->AddRef();
        Hash(HashCombine(0x9a55ull, record->nameHash));
        Hash(uint64_t(desc.type) | (uint64_t(desc.flags) << 8) | (uint64_t(desc.exclusionGroup) << 32));

        m_OpenPass = index;
        m_InSetup = true;
        const nvrhi::FRESULT result = pass->Setup(static_cast<ISetupContext*>(this), data);
        m_InSetup = false;
        m_OpenPass = c_None;

        m_Passes[index].accessCount = m_Accesses.size - m_Passes[index].firstAccess;
        if (result != nvrhi::FS_OK)
        {
            ErrorRecord* error = m_Errors.Push(m_Arena);
            if (error)
            {
                error->code = RecordError::SetupFailed;
                error->pass = index;
                error->resource = c_None;
                error->other = c_None;
                error->value = uint64_t(int64_t(result));
            }
            return result;
        }
        if (outPass)
            *outPass = PassId(index);
        return nvrhi::FS_OK;
    }

    void RenderGraph::SetPassEnabled(PassId pass, bool enabled) noexcept
    {
        if (!CheckNotFrozen())
            return;
        const uint32_t index = uint32_t(pass);
        if (index >= m_Passes.size)
            return;
        m_Passes[index].enabled = enabled;
    }

    void RenderGraph::PushMarker(const char* name) noexcept
    {
        if (MarkerRecord* marker = m_Markers.Push(m_Arena))
        {
            marker->beforePass = m_Passes.size;
            marker->name = CopyString(name ? name : "");
            if (!marker->name)
                marker->name = "";
        }
    }

    void RenderGraph::PopMarker() noexcept
    {
        if (MarkerRecord* marker = m_Markers.Push(m_Arena))
        {
            marker->beforePass = m_Passes.size;
            marker->name = nullptr;
        }
    }

    void* RenderGraph::AllocateFrameData(size_t size, size_t alignment) noexcept
    {
        return m_Arena.Allocate(size, alignment);
    }

    uint64_t RenderGraph::GetEnableMaskHash() noexcept
    {
        uint64_t hash = c_HashSeed;
        for (uint32_t index = 0; index < m_Passes.size; ++index)
        {
            if (!m_Passes[index].enabled)
                hash = HashCombine(hash, index);
        }
        return hash;
    }

    // ------------------------------------------------------------------------------------------------ compile

    nvrhi::FRESULT RenderGraph::Compile(const BackendProfile& profile, const CompileOptions& options,
                                        ICompiledRenderGraph** outPlan) noexcept
    {
        if (!outPlan)
            return nvrhi::FE_INVALID_ARGS;
        *outPlan = nullptr;
        if (m_InSetup)
        {
            RecordErrorAt(RecordError::NestedAddPass);
            return nvrhi::FE_INVALID_ARGS;
        }
        m_Frozen = true;

        try
        {
            // Recording errors, including declarations after an earlier Compile of this frame, fail every Compile.
            if (m_Errors.size)
            {
                m_Plan = nullptr;
                m_Diagnostics.clear();
                FormatErrors(m_Diagnostics);
                return nvrhi::FE_INVALID_ARGS;
            }

            const uint64_t profileHash = HashProfile(profile);
            const uint64_t optionsHash = HashOptions(options);
            if (m_Plan && profileHash == m_PlanProfileHash && optionsHash == m_PlanOptionsHash)
            {
                *outPlan = m_Plan.Get();
                m_Plan->AddRef();
                return nvrhi::FS_OK;
            }
            m_Plan = nullptr;
            m_Diagnostics.clear();

            const uint64_t key = HashCombine(HashCombine(m_StructuralHash, profileHash), optionsHash);
            const uint64_t maskHash = GetEnableMaskHash();

            nvrhi::AutoPtr<CompiledRenderGraph> plan;
            if (options.maxCachedPlans > 0)
                plan = m_Registry->FindPlan(key, maskHash, m_Passes.size, m_Resources.size, m_Accesses.size, m_FrameIndex);

            if (plan)
            {
                plan->m_Statistics.cacheHit = true;
            }
            else
            {
                m_Registry->CountMiss();
                std::shared_ptr<const StructurePlan> structure;
                if (options.maxCachedPlans > 0)
                    structure = m_Registry->FindStructure(key);

                const auto start = std::chrono::steady_clock::now();
                CompiledRenderGraph* compiled = nullptr;
                const nvrhi::FRESULT result = CompileGraph(*this, profile, options, key, maskHash, structure, &compiled,
                                                           m_Diagnostics);
                if (result != nvrhi::FS_OK)
                    return result;
                plan = nvrhi::TakeOver(compiled);
                plan->m_Statistics.compileMicroseconds = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count());
                plan->m_Statistics.cacheHit = false;
                if (options.maxCachedPlans > 0)
                    m_Registry->InsertPlan(plan.Get(), m_Passes.size, m_Resources.size, m_Accesses.size,
                                           options.maxCachedPlans, m_FrameIndex);
            }

            if ((options.debugFlags & DebugFlags::StrictValidation) != DebugFlags::None)
            {
                std::string report;
                if (VerifyCompiledPlan(*plan, nullptr, report) != nvrhi::FS_OK)
                {
                    m_Diagnostics += report;
                    return nvrhi::FE_GENERIC_ERROR;
                }
            }

            m_Plan = plan;
            m_PlanProfileHash = profileHash;
            m_PlanOptionsHash = optionsHash;
            *outPlan = plan.Detach();
            return nvrhi::FS_OK;
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
    }

    nvrhi::FRESULT RenderGraph::Execute(ICompiledRenderGraph* plan, const ExecuteDesc& desc) noexcept
    {
        (void)desc;
        if (!plan || plan != static_cast<ICompiledRenderGraph*>(m_Plan.Get()))
            return nvrhi::FE_INVALID_ARGS;
        for (const PlanResource& resource : m_Plan->m_Resources)
        {
            if (resource.standIn)
                return nvrhi::FE_INVALID_ARGS;
        }
        if (!m_Registry->GetDevice())
            return nvrhi::FE_UNSUPPORTED;
        // The executor is step P1.4 of the plan.
        return nvrhi::FE_NOT_IMPLEMENT;
    }

    void RenderGraph::GetDiagnostics(nvrhi::string& out) noexcept
    {
        try
        {
            out = m_Diagnostics.c_str();
        }
        catch (...)
        {
        }
    }

    // ------------------------------------------------------------------------------------------------ diagnostics

    std::string RenderGraph::DescribeResource(uint32_t index) const
    {
        if (index >= m_Resources.size)
            return "resource #" + std::to_string(index);
        const ResourceRecord& record = m_Resources[index];
        std::string text = record.name ? "'" + std::string(record.name) + "'" : std::string("resource");
        text += " (#" + std::to_string(index);
        if (record.kind == ResourceKind::History)
            text += ", history layer " + std::to_string(record.historyLayer);
        return text + ")";
    }

    std::string RenderGraph::DescribePass(uint32_t index) const
    {
        if (index >= m_Passes.size)
            return "outside a pass";
        const PassRecord& pass = m_Passes[index];
        return "pass '" + std::string(pass.name ? pass.name : "") + "' (" + std::to_string(index) + ")";
    }

    void RenderGraph::FormatErrors(std::string& out) const
    {
        for (uint32_t index = 0; index < m_Errors.size; ++index)
        {
            const ErrorRecord& error = m_Errors[index];
            std::string line = "error: " + DescribePass(error.pass) + ": ";
            switch (error.code)
            {
            case RecordError::InvalidId:
                line += "uses an invalid resource id (index " + std::to_string(error.resource) + ")";
                break;
            case RecordError::StaleVersion:
                line += "uses " + DescribeResource(error.resource) + " at version " + std::to_string(error.value) +
                    ", which is not its current version; the current version was written by " + DescribePass(error.other);
                break;
            case RecordError::SetupContextClosed:
                line += "an ISetupContext call outside IGpuPass::Setup (the context is valid only until Setup returns)";
                break;
            case RecordError::NestedAddPass:
                line += "AddPass or Compile called from inside IGpuPass::Setup";
                break;
            case RecordError::Frozen:
                line += "a declaration after Compile; the graph is frozen until Reset";
                break;
            case RecordError::HistoryWrite:
                line += "writes " + DescribeResource(error.resource) + "; only the current layer, ids[0], may be written";
                break;
            case RecordError::PassLocalForeign:
                line += "names " + DescribeResource(error.resource) + ", a pass-local transient of " + DescribePass(error.other);
                break;
            case RecordError::TextureAccessOnBuffer:
                line += "makes a texture-only access, or names a subresource range, on buffer " + DescribeResource(error.resource);
                break;
            case RecordError::SetupFailed:
                line += "IGpuPass::Setup failed (FRESULT " + std::to_string(int64_t(error.value)) + ")";
                break;
            case RecordError::BadLayerCount:
                line += "GetHistory with " + std::to_string(error.value) + " layers; 2 to " +
                    std::to_string(c_MaxHistoryLayers) + " are allowed";
                break;
            case RecordError::ExtractNotTransient:
                line += "extracts " + DescribeResource(error.resource) + ", which is not a texture created on the graph";
                break;
            case RecordError::DisabledOutputNotWritten:
                line += "sets a disabled-output policy on " + DescribeResource(error.resource) + ", which the pass does not write";
                break;
            case RecordError::NullPass:
                line += "AddPass with a null pass";
                break;
            case RecordError::BadAttachmentSlot:
                line += "colour target slot " + std::to_string(error.value) + "; slots 0 to 7 exist";
                break;
            case RecordError::ArenaExhausted:
                line += "the frame arena could not grow";
                break;
            case RecordError::PersistentKindMismatch:
                line += "the name of " + DescribeResource(error.resource) +
                    " is already used for another kind of persistent resource";
                break;
            case RecordError::KeepInitialStateTransient:
                line += "creates " + DescribeResource(error.resource) +
                    " with keepInitialState; transients may be aliased, so the graph must own their states (6.7)";
                break;
            case RecordError::TrackedImportOfHostKept:
                line += "imports " + DescribeResource(error.resource) +
                    " as tracked, but it has keepInitialState; use ImportHostKeptTexture";
                break;
            case RecordError::NullImport:
                line += "imports a null resource";
                break;
            case RecordError::NullName:
                line += "a persistent resource or history without a name";
                break;
            case RecordError::WriteBitsInRead:
                line += "declares write accesses on " + DescribeResource(error.resource) + " through Read";
                break;
            case RecordError::ClearOnReadOnlyDepth:
                line += "clears read-only depth target " + DescribeResource(error.resource);
                break;
            }
            out += line;
            out += '\n';
        }
    }
}

namespace donut::render
{
    nvrhi::FRESULT CreateRenderGraph(IResourceRegistry* registry, IRenderGraph** out) noexcept
    {
        if (!out)
            return nvrhi::FE_INVALID_ARGS;
        *out = nullptr;
        if (!registry)
            return nvrhi::FE_INVALID_ARGS;
        // Only registries made by CreateResourceRegistry are accepted.
        rdg::ResourceRegistry* concrete = static_cast<rdg::ResourceRegistry*>(registry);
        if (concrete->m_HasGraph)
            return nvrhi::FE_INVALID_ARGS;
        try
        {
            *out = MAKE_RC_OBJ(rdg::RenderGraph, concrete);
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
        return nvrhi::FS_OK;
    }
}
