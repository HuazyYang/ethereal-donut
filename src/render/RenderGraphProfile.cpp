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

// Backend profiles (design 9.2) and the per-backend rules the compiler consults: the lowering of accesses to
// nvrhi states (6.1), queue legality (6.8), the read-state lattice (6.3) and clear states (6.9).

#include "RenderGraphInternal.h"

#include <algorithm>

namespace donut::render::rdg
{
    using nvrhi::ResourceStates;

    nvrhi::ShaderType DefaultStages(PassType type) noexcept
    {
        switch (type)
        {
        case PassType::Raster:     return nvrhi::ShaderType::AllGraphics;
        case PassType::Compute:    return nvrhi::ShaderType::Compute;
        case PassType::RayTracing: return nvrhi::ShaderType::AllRayTracing;
        default:                   return nvrhi::ShaderType::None;
        }
    }

    ResourceStates LowerAccess(Access access, nvrhi::ShaderType stages, bool depthStencilTexture,
                               const BackendProfile& profile) noexcept
    {
        ResourceStates state = LowerAccess(access, stages);
        if (depthStencilTexture && profile.barrierModel == BarrierModel::Vulkan &&
            (state & ResourceStates::ShaderResource) != 0)
            state = state | ResourceStates::DepthRead;
        return state;
    }

    ResourceStates LowerAccess(Access access, nvrhi::ShaderType stages) noexcept
    {
        ResourceStates state = ResourceStates::Unknown;
        auto has = [access](Access bits) { return (access & bits) != Access::None; };

        if (has(Access::VertexBuffer))          state = state | ResourceStates::VertexBuffer;
        if (has(Access::IndexBuffer))           state = state | ResourceStates::IndexBuffer;
        if (has(Access::IndirectArgs))          state = state | ResourceStates::IndirectArgument;
        if (has(Access::ConstantBuffer))        state = state | ResourceStates::ConstantBuffer;
        if (has(Access::ShaderRead))
        {
            // nvrhi's visibility-exact shader-resource states (2.1): the pixel stage, the others, or both.
            if (stages == nvrhi::ShaderType::None)
                state = state | ResourceStates::ShaderResource;
            else
            {
                if ((stages & nvrhi::ShaderType::Pixel) != 0)
                    state = state | ResourceStates::PixelShaderResource;
                if ((stages & ~nvrhi::ShaderType::Pixel) != 0)
                    state = state | ResourceStates::NonPixelShaderResource;
            }
        }
        if (has(Access::UAVRead | Access::UAVWrite)) state = state | ResourceStates::UnorderedAccess;
        if (has(Access::ColorTarget | Access::ColorTargetRead)) state = state | ResourceStates::RenderTarget;
        if (has(Access::DepthWrite | Access::StencilWrite))
            state = state | ResourceStates::DepthWrite;
        else if (has(Access::DepthRead | Access::StencilRead))
            state = state | ResourceStates::DepthRead;
        if (has(Access::CopySource))            state = state | ResourceStates::CopySource;
        if (has(Access::CopyDest))              state = state | ResourceStates::CopyDest;
        if (has(Access::ResolveSource))         state = state | ResourceStates::ResolveSource;
        if (has(Access::ResolveDest))           state = state | ResourceStates::ResolveDest;
        if (has(Access::ShadingRateRead))       state = state | ResourceStates::ShadingRateSurface;
        if (has(Access::AccelStructRead))       state = state | ResourceStates::AccelStructRead;
        if (has(Access::AccelStructBuildInput)) state = state | ResourceStates::AccelStructBuildInput;
        if (has(Access::AccelStructWrite))      state = state | ResourceStates::AccelStructWrite;
        if (has(Access::Present))               state = state | ResourceStates::Present;
        // A readback buffer stays where the copy left it: D3D12 readback heaps never leave COPY_DEST.
        if (has(Access::HostRead))              state = state | ResourceStates::CopyDest;
        return state;
    }

    // nvrhi's flag operators are inline, not constexpr.
    static const ResourceStates c_ComputeQueueIllegal = ResourceStates::RenderTarget | ResourceStates::DepthWrite |
        ResourceStates::DepthRead | ResourceStates::PixelShaderResource | ResourceStates::IndexBuffer |
        ResourceStates::VertexBuffer | ResourceStates::StreamOut | ResourceStates::ResolveSource |
        ResourceStates::ResolveDest | ResourceStates::ShadingRateSurface | ResourceStates::Present;

    static const ResourceStates c_CopyQueueLegal = ResourceStates::Common | ResourceStates::CopySource |
        ResourceStates::CopyDest;

    bool IsStateLegalOnQueue(ResourceStates state, nvrhi::CommandQueue queue, const BackendProfile& profile) noexcept
    {
        if (profile.barrierModel == BarrierModel::Implicit)
            return true;
        switch (queue)
        {
        case nvrhi::CommandQueue::Compute:
            // A sampled depth texture on Vulkan (ShaderResource | DepthRead, see LowerAccess): nvrhi keeps its layout
            // and drops the fragment-test stages on a compute queue.
            if (profile.barrierModel == BarrierModel::Vulkan && (state & ResourceStates::ShaderResource) != 0)
                state = state & ~ResourceStates::DepthRead;
            return (state & c_ComputeQueueIllegal) == 0;
        case nvrhi::CommandQueue::Copy:    return (state & ~c_CopyQueueLegal) == 0;
        default:                           return true;
        }
    }

    // Vulkan image layouts, as nvrhi's convertResourceStateForExplicitBarriers chooses them.
    enum class VulkanLayout : uint8_t
    {
        Undefined, General, ShaderReadOnly, DepthReadOnly, DepthAttachment, ColorAttachment, TransferSrc, TransferDst,
        ShadingRate, PresentSrc,
    };

    static VulkanLayout VulkanLayoutOfBit(ResourceStates bit) noexcept
    {
        switch (bit)
        {
        case ResourceStates::PixelShaderResource:
        case ResourceStates::NonPixelShaderResource: return VulkanLayout::ShaderReadOnly;
        case ResourceStates::UnorderedAccess:        return VulkanLayout::General;
        case ResourceStates::RenderTarget:           return VulkanLayout::ColorAttachment;
        case ResourceStates::DepthWrite:             return VulkanLayout::DepthAttachment;
        case ResourceStates::DepthRead:              return VulkanLayout::DepthReadOnly;
        case ResourceStates::CopyDest:
        case ResourceStates::ResolveDest:            return VulkanLayout::TransferDst;
        case ResourceStates::CopySource:
        case ResourceStates::ResolveSource:          return VulkanLayout::TransferSrc;
        case ResourceStates::Present:                return VulkanLayout::PresentSrc;
        case ResourceStates::ShadingRateSurface:     return VulkanLayout::ShadingRate;
        default:                                     return VulkanLayout::Undefined;
        }
    }

    static VulkanLayout VulkanLayoutOf(ResourceStates state) noexcept
    {
        VulkanLayout layout = VulkanLayout::Undefined;
        bool differ = false;
        for (uint32_t bit = 1; bit != 0 && bit <= uint32_t(state); bit <<= 1)
        {
            if ((uint32_t(state) & bit) == 0)
                continue;
            const VulkanLayout bitLayout = VulkanLayoutOfBit(ResourceStates(bit));
            if (bitLayout == VulkanLayout::Undefined)
                continue;
            if (layout == VulkanLayout::Undefined)
                layout = bitLayout;
            else if (layout != bitLayout)
                differ = true;
        }
        if (differ)
        {
            const ResourceStates depthReadable = ResourceStates::DepthRead | ResourceStates::ShaderResource;
            const bool depthAndSampling = (state & ResourceStates::DepthRead) != 0 && (state & ~depthReadable) == 0;
            layout = depthAndSampling ? VulkanLayout::DepthReadOnly : VulkanLayout::General;
        }
        return layout;
    }

    bool ChangesVulkanLayout(ResourceStates before, ResourceStates after) noexcept
    {
        return VulkanLayoutOf(before) != VulkanLayoutOf(after);
    }

    CombineResult CombineReadStates(ResourceStates a, ResourceStates b, bool isTexture, const BackendProfile& profile) noexcept
    {
        if (a == b)
            return { true, a, 0 };
        const ResourceStates combined = a | b;

        switch (profile.barrierModel)
        {
        case BarrierModel::Implicit:
            return { true, combined, 0 };

        case BarrierModel::D3D12Legacy:
        case BarrierModel::D3D12Enhanced:
        {
            // UNORDERED_ACCESS and the acceleration-structure state are not read states and combine with nothing;
            // PRESENT is COMMON. The write states reach here only through HostRead (COPY_DEST).
            const ResourceStates exclusive = ResourceStates::UnorderedAccess | ResourceStates::AccelStructRead |
                ResourceStates::Present | ResourceStates::CopyDest | ResourceStates::RenderTarget |
                ResourceStates::DepthWrite | ResourceStates::Common;
            if ((combined & exclusive) != 0)
                return { false, ResourceStates::Unknown, 0 };
            if (isTexture && profile.barrierModel == BarrierModel::D3D12Enhanced)
            {
                // Corrected in P1.1 (design 6.3): the read sets one layout serves. GENERIC_READ for shader reads
                // and copy sources, DIRECT_QUEUE_GENERIC_READ with depth reads, resolve sources and shading-rate
                // reads; queue legality (6.8) keeps the latter off the compute queue.
                const ResourceStates directQueueRead = ResourceStates::ShaderResource | ResourceStates::CopySource |
                    ResourceStates::DepthRead | ResourceStates::ResolveSource | ResourceStates::ShadingRateSurface;
                if ((combined & ~directQueueRead) != 0)
                    return { false, ResourceStates::Unknown, 0 };
            }
            return { true, combined, 0 };
        }

        case BarrierModel::Vulkan:
        default:
        {
            if (!isTexture)
                return { true, combined, 0 };
            if ((combined & ResourceStates::Present) != 0)
                return { false, ResourceStates::Unknown, 0 };
            const VulkanLayout layout = VulkanLayoutOf(combined);
            // GENERAL instead of a read-only layout costs; when one side already needs GENERAL (a storage image),
            // adding readers costs nothing.
            const bool costs = layout == VulkanLayout::General && VulkanLayoutOf(a) != VulkanLayout::General &&
                VulkanLayoutOf(b) != VulkanLayout::General;
            return { true, combined, costs ? uint32_t(profile.lattice.generalLayout) : 0u };
        }
        }
    }

    ResourceStates ClearState(bool isBuffer, bool hasDepth, bool isRenderTarget, const BackendProfile& profile) noexcept
    {
        switch (profile.barrierModel)
        {
        case BarrierModel::Implicit:
            return ResourceStates::Unknown;
        case BarrierModel::Vulkan:
            // vkCmdClearColorImage, vkCmdClearDepthStencilImage and vkCmdFillBuffer are transfer operations.
            return ResourceStates::CopyDest;
        default:
            // nvrhi's D3D12 clears pick the view from the description: a render target clears through its RTV, any
            // other texture and every buffer through a UAV, depth through its DSV.
            if (isBuffer)
                return ResourceStates::UnorderedAccess;
            if (hasDepth)
                return ResourceStates::DepthWrite;
            return isRenderTarget ? ResourceStates::RenderTarget : ResourceStates::UnorderedAccess;
        }
    }

    bool QueuesShareFamily(nvrhi::CommandQueue a, nvrhi::CommandQueue b, const BackendProfile& profile) noexcept
    {
        return a == b || !profile.queueOwnershipTransfer;
    }

    uint64_t HashProfile(const BackendProfile& profile) noexcept
    {
        uint64_t hash = HashCombine(c_HashSeed, uint64_t(profile.api) | (uint64_t(profile.barrierModel) << 8) |
            (uint64_t(profile.planeModel) << 16) | (uint64_t(profile.maxMemoryTier) << 24) |
            (uint64_t(profile.recordingMode) << 32) | (uint64_t(profile.passTypes) << 40) |
            (uint64_t(profile.resourceHeapTier) << 48));
        hash = HashCombine(hash, uint64_t(profile.explicitBarriers) | (uint64_t(profile.splitBarriers) << 1) |
            (uint64_t(profile.aliasingBarriers) << 2) | (uint64_t(profile.computeQueue) << 3) |
            (uint64_t(profile.copyQueue) << 4) | (uint64_t(profile.queueOwnershipTransfer) << 5) |
            (uint64_t(profile.depthClears) << 6));
        const LatticePolicy& lattice = profile.lattice;
        hash = HashCombine(hash, uint64_t(lattice.transition) | (uint64_t(lattice.generalLayout) << 16) |
            (uint64_t(lattice.queueTransfer) << 32) | (uint64_t(lattice.asyncOverlapGain) << 48));
        hash = HashCombine(hash, uint64_t(lattice.splitMinGap) | (uint64_t(lattice.splitLayoutTransitionsOnly) << 8));
        hash = HashCombine(hash, lattice.splitMinBytes);
        return hash;
    }

    void EstimateResourceSize(const ResourceRecord& record, uint64_t& size, uint64_t& alignment) noexcept
    {
        constexpr uint64_t c_SmallAlignment = 64 * 1024;
        constexpr uint64_t c_MsaaAlignment = 4 * 1024 * 1024;
        if (record.isBuffer)
        {
            alignment = c_SmallAlignment;
            size = (std::max<uint64_t>(record.buffer.byteSize, 1) + alignment - 1) & ~(alignment - 1);
            return;
        }

        const TextureInfo& info = record.texture;
        const nvrhi::FormatInfo& format = nvrhi::getFormatInfo(info.format);
        const uint64_t bytesPerBlock = format.bytesPerBlock ? format.bytesPerBlock : 4;
        const uint32_t blockSize = format.blockSize ? format.blockSize : 1;
        const bool is3D = info.dimension == nvrhi::TextureDimension::Texture3D;
        const uint64_t slices = is3D ? 1 : std::max(1u, info.arraySize);

        uint64_t total = 0;
        for (uint32_t mip = 0; mip < std::max(1u, info.mipLevels); ++mip)
        {
            const uint64_t width = std::max(1u, info.width >> mip);
            const uint64_t height = std::max(1u, info.height >> mip);
            const uint64_t depth = is3D ? std::max(1u, info.depth >> mip) : 1;
            const uint64_t blocksX = (width + blockSize - 1) / blockSize;
            const uint64_t blocksY = (height + blockSize - 1) / blockSize;
            total += blocksX * blocksY * depth * bytesPerBlock;
        }
        total *= slices * std::max(1u, info.sampleCount);
        alignment = info.sampleCount > 1 ? c_MsaaAlignment : c_SmallAlignment;
        size = (std::max<uint64_t>(total, 1) + alignment - 1) & ~(alignment - 1);
    }
}

namespace donut::render
{
    nvrhi::FRESULT GetReferenceProfile(ReferenceProfile kind, BackendProfile* out) noexcept
    {
        if (!out)
            return nvrhi::FE_INVALID_ARGS;
        BackendProfile profile;
        switch (kind)
        {
        case ReferenceProfile::D3D12Legacy:
        case ReferenceProfile::D3D12Enhanced:
            profile.api = nvrhi::GraphicsAPI::D3D12;
            profile.barrierModel = kind == ReferenceProfile::D3D12Legacy ? BarrierModel::D3D12Legacy : BarrierModel::D3D12Enhanced;
            profile.planeModel = PlaneModel::PlaneSlices;
            profile.maxMemoryTier = MemoryTier::Placed;
            profile.recordingMode = RecordingMode::Immediate;
            profile.passTypes = PassTypeMask::All;
            break;

        case ReferenceProfile::Vulkan:
        case ReferenceProfile::VulkanSeparateComputeFamily:
            profile.api = nvrhi::GraphicsAPI::VULKAN;
            profile.barrierModel = BarrierModel::Vulkan;
            profile.planeModel = PlaneModel::AspectMasks;
            // Pooled until nvrhi's createHeap honours memoryTypeBits (7.5).
            profile.maxMemoryTier = MemoryTier::Pooled;
            profile.recordingMode = RecordingMode::Immediate;
            profile.passTypes = PassTypeMask::All;
            profile.queueOwnershipTransfer = kind == ReferenceProfile::VulkanSeparateComputeFamily;
            // Events cost: split only layout transitions of large resources (6.4).
            profile.lattice.splitLayoutTransitionsOnly = true;
            profile.lattice.splitMinBytes = 1024 * 1024;
            break;

        case ReferenceProfile::D3D11:
            profile.api = nvrhi::GraphicsAPI::D3D11;
            profile.barrierModel = BarrierModel::Implicit;
            profile.planeModel = PlaneModel::Ignored;
            profile.maxMemoryTier = MemoryTier::Pooled;
            profile.recordingMode = RecordingMode::ImmediateSerial;
            profile.passTypes = PassTypeMask::Raster | PassTypeMask::Compute | PassTypeMask::Copy;
            profile.splitBarriers = false;
            profile.aliasingBarriers = false;
            profile.computeQueue = false;
            profile.copyQueue = false;
            break;

        case ReferenceProfile::CUDA:
            profile.api = nvrhi::GraphicsAPI::CUDA;
            profile.barrierModel = BarrierModel::Implicit;
            profile.planeModel = PlaneModel::Ignored;
            profile.maxMemoryTier = MemoryTier::Pooled;
            profile.recordingMode = RecordingMode::Deferred;
            profile.passTypes = PassTypeMask::Compute | PassTypeMask::Copy | PassTypeMask::RayTracing;
            profile.splitBarriers = false;
            profile.aliasingBarriers = false;
            profile.depthClears = false;
            break;

        default:
            return nvrhi::FE_INVALID_ARGS;
        }
        *out = profile;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT ComputeBackendProfile(nvrhi::IDevice* device, BackendProfile* out) noexcept
    {
        if (!device || !out)
            return nvrhi::FE_INVALID_ARGS;

        BackendProfile profile;
        nvrhi::FRESULT result = nvrhi::FE_UNSUPPORTED;
        switch (device->getGraphicsAPI())
        {
        case nvrhi::GraphicsAPI::D3D12:
            result = GetReferenceProfile(device->queryFeatureSupport(nvrhi::Feature::EnhancedBarriers)
                ? ReferenceProfile::D3D12Enhanced : ReferenceProfile::D3D12Legacy, &profile);
            // The heap tier is not queried yet: tier 2, every heap class in one heap. The Placed tier's probe is P1.6.
            break;
        case nvrhi::GraphicsAPI::VULKAN:
            result = GetReferenceProfile(ReferenceProfile::Vulkan, &profile);
            profile.queueOwnershipTransfer = device->queryFeatureSupport(nvrhi::Feature::QueueOwnershipTransfer);
            break;
        case nvrhi::GraphicsAPI::D3D11:
            result = GetReferenceProfile(ReferenceProfile::D3D11, &profile);
            break;
        case nvrhi::GraphicsAPI::CUDA:
            result = GetReferenceProfile(ReferenceProfile::CUDA, &profile);
            break;
        }
        if (result != nvrhi::FS_OK)
            return result;

        if (profile.barrierModel != BarrierModel::Implicit)
        {
            profile.splitBarriers = device->queryFeatureSupport(nvrhi::Feature::SplitBarriers);
            profile.aliasingBarriers = device->queryFeatureSupport(nvrhi::Feature::AliasingBarriers);
        }
        if (profile.api != nvrhi::GraphicsAPI::D3D11)
        {
            profile.computeQueue = device->queryFeatureSupport(nvrhi::Feature::ComputeQueue);
            profile.copyQueue = device->queryFeatureSupport(nvrhi::Feature::CopyQueue);
        }
        *out = profile;
        return nvrhi::FS_OK;
    }
}
