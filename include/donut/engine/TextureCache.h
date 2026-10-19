/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <donut/engine/SceneTypes.h>
#include <donut/core/log.h>

#include <nvrhi/nvrhi.h>
#include <atomic>
#include <filesystem>
#include <unordered_map>
#include <shared_mutex>
#include <queue>

namespace donut::vfs
{
    class IFileSystem;
}

namespace donut::engine
{
    class CommonRenderPasses;
    class ThreadPool;

    struct TextureSubresourceData
    {
        size_t rowPitch = 0;
        size_t depthPitch = 0;
        ptrdiff_t dataOffset = 0;
        size_t dataSize = 0;
    };

    struct TextureCacheKey
    {
        std::string path;
        TextureLoadOptions options;

        bool operator==(const TextureCacheKey& o) const { return path == o.path && options == o.options; }
    };

    struct TextureCacheKeyHash
    {
        size_t operator()(const TextureCacheKey& key) const
        {
            size_t hash = std::hash<std::string>()(key.path);
            hash ^= size_t(nvrhi::packComponentMapping(key.options.overrideComponentMapping ? &*key.options.overrideComponentMapping : nullptr)) << 1;
            hash ^= size_t(key.options.sRGBMode) << 17;
            hash ^= size_t(key.options.baseMip) << 20;
            return hash;
        }
    };

    struct TextureData : public LoadedTexture
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
        nvrhi::AutoPtr<nvrhi::IDataBlob> data;

        nvrhi::Format format = nvrhi::Format::UNKNOWN;
        uint32_t width = 1;
        uint32_t height = 1;
        uint32_t depth = 1;
        uint32_t arraySize = 1;
        uint32_t mipLevels = 1;
        nvrhi::TextureDimension dimension = nvrhi::TextureDimension::Unknown;
        bool isRenderTarget = false;

        // What the file itself declares (KTX2 KTXswizzle); containers that cannot
        // carry a mapping leave it unset.
        std::optional<nvrhi::ComponentMapping> fileComponentMapping;

        // ArraySlice -> MipLevel -> TextureSubresourceData
        std::vector<std::vector<TextureSubresourceData>> dataLayout;

        // The caller's override if it set one, else the file's, else identity.
        nvrhi::ComponentMapping ResolveComponentMapping() const
        {
            return loadOptions.overrideComponentMapping.value_or(
                fileComponentMapping.value_or(nvrhi::ComponentMapping{}));
        }
    };

    class TextureCache: public nvrhi::ObjectImpl<nvrhi::IObject>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(TextureCache)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

    protected:
        nvrhi::DeviceHandle m_Device;
        nvrhi::CommandListHandle m_CommandList;
        std::unordered_map<TextureCacheKey, nvrhi::AutoPtr<TextureData>, TextureCacheKeyHash> m_LoadedTextures;
        mutable std::shared_mutex m_LoadedTexturesMutex;

        std::queue<nvrhi::AutoPtr<TextureData>> m_TexturesToFinalize;
        nvrhi::AutoPtr<DescriptorTableManager> m_DescriptorTable;
        std::mutex m_TexturesToFinalizeMutex;

        nvrhi::AutoPtr<vfs::IFileSystem> m_fs;

        uint32_t m_MaxTextureSize = 0;

        bool m_GenerateMipmaps = true;

        std::optional<SRGBMode> m_SRGBOverride;

        // The options a load request actually runs with: the caller's, with the cache-wide sRGB override applied.
        TextureLoadOptions ResolveLoadOptions(const TextureLoadOptions& options) const;

        log::Severity m_InfoLogSeverity = log::Severity::Info;
        log::Severity m_ErrorLogSeverity = log::Severity::Warning;

        std::atomic<uint32_t> m_TexturesRequested = 0;
        std::atomic<uint32_t> m_TexturesLoaded = 0;
        uint32_t m_TexturesFinalized = 0;

        bool FindTextureInCache(const std::filesystem::path& path, const TextureLoadOptions& options, TextureData** texture);
        nvrhi::AutoPtr<nvrhi::IDataBlob> ReadTextureFile(const std::filesystem::path& path) const;

        bool FillTextureData(
            nvrhi::IDataBlob* fileData,
            TextureData *texture,
            const std::string& extension,
            const std::string& mimeType) const;

        void FinalizeTexture(
            TextureData *texture,
            CommonRenderPasses* passes,
            nvrhi::ICommandList* commandList);

        virtual void TextureLoaded(TextureData *texture);
        virtual nvrhi::AutoPtr<TextureData> CreateTextureData();

    public:
        TextureCache(
            nvrhi::IDevice* device,
            vfs::IFileSystem* fs,
            DescriptorTableManager* descriptorTable);
        virtual ~TextureCache();

        // Release all cached textures
        void Reset();

        // Synchronous read and decode, synchronous upload and mip generation on a given command list (must be open).
        // The `passes` argument is optional, and mip generation is disabled if it's NULL.
        virtual nvrhi::AutoPtr<LoadedTexture> LoadTextureFromFile(
            const std::filesystem::path& path,
            const TextureLoadOptions& options,
            CommonRenderPasses* passes,
            nvrhi::ICommandList* commandList);

        // Synchronous read and decode, deferred upload and mip generation (in the ProcessRenderingThreadCommands queue).
        virtual nvrhi::AutoPtr<LoadedTexture> LoadTextureFromFileDeferred(
            const std::filesystem::path& path,
            const TextureLoadOptions& options);

        // Asynchronous read and decode, deferred upload and mip generation (in the ProcessRenderingThreadCommands queue).
        virtual nvrhi::AutoPtr<LoadedTexture> LoadTextureFromFileAsync(
            const std::filesystem::path& path,
            const TextureLoadOptions& options,
            ThreadPool& threadPool);

        // Same as LoadTextureFromFileAsync, but using a memory blob and MIME type instead of file name, and uncached.
        virtual nvrhi::AutoPtr<LoadedTexture> LoadTextureFromMemoryAsync(
            nvrhi::IDataBlob *data,
            const std::string& name,
            const std::string& mimeType,
            const TextureLoadOptions& options,
            ThreadPool& threadPool);

        // Same as LoadTextureFromFile, but using a memory blob and MIME type instead of file name, and uncached.
        virtual nvrhi::AutoPtr<LoadedTexture> LoadTextureFromMemory(
            nvrhi::IDataBlob *data,
            const std::string& name,
            const std::string& mimeType,
            const TextureLoadOptions& options,
            CommonRenderPasses* passes,
            nvrhi::ICommandList* commandList);

        // Same as LoadTextureFromFileDeferred, but using a memory blob and MIME type instead of file name, and uncached.
        virtual nvrhi::AutoPtr<LoadedTexture> LoadTextureFromMemoryDeferred(
            nvrhi::IDataBlob *data,
            const std::string& name,
            const std::string& mimeType,
            const TextureLoadOptions& options);

        virtual nvrhi::AutoPtr<LoadedTexture> LoadTextureFromRawImageMemory(nvrhi::IDataBlob* data, const char* name,
                                                                     uint32_t width, uint32_t height,
                                                                     nvrhi::Format format,
                                                                     const TextureLoadOptions& options,
                                                                     uint32_t imageBitsPerPixel,
                                                                     uint32_t imagePixelStride);

        // Back-compat overloads for callers written against the older `bool sRGB`.
        // They forward through TextureLoadOptions, so an override of the form above
        // still takes effect.
        [[deprecated("Pass a TextureLoadOptions instead")]]
        nvrhi::AutoPtr<LoadedTexture> LoadTextureFromFile(
            const std::filesystem::path& path, bool sRGB,
            CommonRenderPasses* passes, nvrhi::ICommandList* commandList)
        { return LoadTextureFromFile(path, TextureLoadOptions{ SRGBModeFromBool(sRGB) }, passes, commandList); }

        [[deprecated("Pass a TextureLoadOptions instead")]]
        nvrhi::AutoPtr<LoadedTexture> LoadTextureFromFileDeferred(
            const std::filesystem::path& path, bool sRGB)
        { return LoadTextureFromFileDeferred(path, TextureLoadOptions{ SRGBModeFromBool(sRGB) }); }

        [[deprecated("Pass a TextureLoadOptions instead")]]
        nvrhi::AutoPtr<LoadedTexture> LoadTextureFromFileAsync(
            const std::filesystem::path& path, bool sRGB, ThreadPool& threadPool)
        { return LoadTextureFromFileAsync(path, TextureLoadOptions{ SRGBModeFromBool(sRGB) }, threadPool); }

        [[deprecated("Pass a TextureLoadOptions instead")]]
        nvrhi::AutoPtr<LoadedTexture> LoadTextureFromMemoryAsync(
            nvrhi::IDataBlob* data, const std::string& name,
            const std::string& mimeType, bool sRGB, ThreadPool& threadPool)
        { return LoadTextureFromMemoryAsync(data, name, mimeType, TextureLoadOptions{ SRGBModeFromBool(sRGB) }, threadPool); }

        [[deprecated("Pass a TextureLoadOptions instead")]]
        nvrhi::AutoPtr<LoadedTexture> LoadTextureFromMemory(
            nvrhi::IDataBlob* data, const std::string& name,
            const std::string& mimeType, bool sRGB,
            CommonRenderPasses* passes, nvrhi::ICommandList* commandList)
        { return LoadTextureFromMemory(data, name, mimeType, TextureLoadOptions{ SRGBModeFromBool(sRGB) }, passes, commandList); }

        [[deprecated("Pass a TextureLoadOptions instead")]]
        nvrhi::AutoPtr<LoadedTexture> LoadTextureFromMemoryDeferred(
            nvrhi::IDataBlob* data, const std::string& name,
            const std::string& mimeType, bool sRGB)
        { return LoadTextureFromMemoryDeferred(data, name, mimeType, TextureLoadOptions{ SRGBModeFromBool(sRGB) }); }

        // Tells if the texture has been loaded from file successfully and its data is available in the texture object.
        // After the texture is finalized and uploaded to the GPU, the data is no longer available on the CPU,
        // and this function returns false.
        bool IsTextureLoaded(LoadedTexture *texture);

        // Tells if the texture has been uploaded to the GPU
        bool IsTextureFinalized(LoadedTexture *texture);

        // Removes the texture from cache. The texture must *not* be in the deferred finalization queue when it's unloaded.
        // Returns true if the texture has been found and removed from the cache, false otherwise.
        // Note: Any existing handles for the texture remain valid after the texture is unloaded.
        //       Texture lifetimes are tracked by NVRHI and the texture object is only destroyed when no references exist.
        bool UnloadTexture(LoadedTexture *texture);

        // Process a portion of the upload queue, taking up to `timeLimitMilliseconds` CPU time.
        // If `timeLimitMilliseconds` is 0, processes the entire queue.
        // Returns true if any textures have been processed.
        bool ProcessRenderingThreadCommands(CommonRenderPasses& passes, float timeLimitMilliseconds);

        // Destroys the internal command list in order to release the upload buffers used in it.
        void LoadingFinished();

        // Set the maximum texture size allowed after load. Larger textures are resized to fit this constraint.
        // Currently does not affect DDS textures.
        void SetMaxTextureSize(uint32_t size);

        // Enables or disables automatic mip generation for loaded textures.
        void SetGenerateMipmaps(bool generateMipmaps);

        // Replaces the sRGB mode of every subsequent load request, whatever the caller (importer, scene file) asked for.
        // For renderers whose pipeline has no sRGB decode/encode stage, e.g. a UNORM G-buffer written straight to a UNORM
        // swap chain: pass SRGBMode::ForceLinear. Passing std::nullopt restores the per-request behaviour.
        void SetSRGBOverride(std::optional<SRGBMode> mode);

        // Sets the Severity of log messages about textures being loaded.
        void SetInfoLogSeverity(log::Severity value) { m_InfoLogSeverity = value; }

        // Sets the Severity of log messages about textures that couldn't be loaded.
        void SetErrorLogSeverity(log::Severity value) { m_ErrorLogSeverity = value; }

        uint32_t GetNumberOfLoadedTextures() { return m_TexturesLoaded.load(); }
        uint32_t GetNumberOfRequestedTextures() { return m_TexturesRequested.load(); }
        uint32_t GetNumberOfFinalizedTextures() { return m_TexturesFinalized; }

		nvrhi::AutoPtr<TextureData> GetLoadedTexture(std::filesystem::path const& path, const TextureLoadOptions& options = {});

		// Texture cache traversal
		// Note: the iterator locks all cache write-accesses for the duration its lifespan !
		class Iterator
		{
		public:
			typedef std::unordered_map<TextureCacheKey, nvrhi::AutoPtr<TextureData>, TextureCacheKeyHash>::iterator CacheIter;

			Iterator& operator++() { ++m_Iterator; return *this; }
			
			friend bool operator==(Iterator const& a, Iterator const& b) { return a.m_Iterator == b.m_Iterator; }
			friend bool operator!=(Iterator const& a, Iterator const& b) { return !(a == b); }

			CacheIter const& operator->() const { return m_Iterator; }

		private:

			friend class TextureCache;
			
			Iterator(std::shared_mutex& mutex, CacheIter it) : m_Lock(mutex), m_Iterator(it) { }

			std::shared_lock<std::shared_mutex> m_Lock;
			CacheIter m_Iterator;
		};

		Iterator begin() { return Iterator(m_LoadedTexturesMutex, m_LoadedTextures.begin()); }

		Iterator end() { return Iterator(m_LoadedTexturesMutex, m_LoadedTextures.end()); }
    };

    // Saves the contents of texture's slice 0 mip level 0 into an image file.
    // The image format is determined from the file's extension.
    // Supported formats are: BMP, PNG, JPG, TGA.
    // Requires that no immediate command list is open at the time this function is called.
    // Creates and destroys temporary resources internally, so should NOT be called often.
    bool SaveTextureToFile(
        nvrhi::IDevice* device,
        CommonRenderPasses* pPasses,
        nvrhi::ITexture* texture,
        nvrhi::ResourceStates textureState,
        const char* fileName,
        bool saveAlphaChannel = true);
}
