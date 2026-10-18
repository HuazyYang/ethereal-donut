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

#include <donut/engine/ShaderFactory.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/log.h>
#include <donut/core/string_utils.h>
#include <ShaderToolBlob.h>
#include <nvrhi/shader-program.h>
#if DONUT_WITH_AFTERMATH
#include <donut/app/AftermathCrashDump.h>
#endif

using namespace std;
using namespace donut::vfs;
using namespace donut::engine;

#if DONUT_WITH_AFTERMATH
namespace
{
    // Answers the Aftermath crash dump helper's shader binary queries from a ShaderFactory's bytecode cache.
    // Holds a plain back-pointer: the factory unregisters it before it is destroyed.
    class AftermathShaderLookup : public nvrhi::ObjectImpl<nvrhi::IAftermathShaderBinaryLookup>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(AftermathShaderLookup)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IAftermathShaderBinaryLookup)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IObject)
        NVRHI_END_INTERFACE_TABLE()

        explicit AftermathShaderLookup(ShaderFactory* factory) : m_Factory(factory) { }

        bool findShaderBinary(uint64_t shaderHash, nvrhi::PFN_AftermathShaderHashGenerator hashGenerator,
                              const void*& outBinary, size_t& outSize) noexcept override
        {
            try
            {
                auto [binary, size] = m_Factory->FindShaderFromHash(shaderHash, hashGenerator);
                if (size == 0)
                    return false;
                outBinary = binary;
                outSize = size;
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

    private:
        ShaderFactory* m_Factory;
    };
}
#endif

namespace
{
    bool IsShaderArtifact(const void* data, size_t size)
    {
        return data && size >= 4 && memcmp(data, "NSTA", 4) == 0;
    }

    nvrhi::FRESULT FindArtifactShader(nvrhi::IShaderProgram* program, const nvrhi::ShaderDesc& desc,
        nvrhi::IShader** shader) noexcept
    {
        if (!shader) return nvrhi::FE_INVALID_ARGS;
        *shader = nullptr;
        uint32_t index = ~0u;
        const auto result = program->findEntryPoint(desc.entryName.c_str(), desc.shaderType, &index);
        if (NVRHI_FAILED(result)) return result;
        return program->createShaderWithDesc(index, desc, shader);
    }

    class ArtifactShaderLibrary final : public nvrhi::ObjectImpl<nvrhi::IShaderLibrary>
    {
        nvrhi::ShaderProgramHandle m_Program;
        std::vector<unsigned char> m_Bytes;
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(ArtifactShaderLibrary)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IShaderLibrary)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::IRHIObject)
        NVRHI_END_INTERFACE_TABLE()

        ArtifactShaderLibrary(nvrhi::IShaderProgram* program, const void* data, size_t size)
            : m_Program(program), m_Bytes(static_cast<const unsigned char*>(data),
                static_cast<const unsigned char*>(data) + size) { }

        void getBytecode(const void** data, size_t* size) const noexcept override
        {
            if (data) *data = m_Bytes.data();
            if (size) *size = m_Bytes.size();
        }

        nvrhi::FRESULT getShader(const char* name, nvrhi::ShaderType type,
            nvrhi::IShader** shader) noexcept override
        {
            return FindArtifactShader(m_Program, nvrhi::ShaderDesc().setEntryName(name ? name : "").setShaderType(type), shader);
        }
    };
}

// Builds the "permutation not found" message using the two-call sizing convention
// of ShaderToolBlobFormatNotFoundMessage (the required length excludes the NUL,
// the capacity passed with a buffer must include it).
static std::string FormatShaderNotFoundMessage(const void* blob, size_t blobSize,
    const ShaderToolBlobConstant* constants, size_t numConstants)
{
    std::string message = "Couldn't find the required shader permutation in the blob.";

    size_t length = 0;
    if (ShaderToolBlobFormatNotFoundMessage(blob, blobSize, constants, numConstants, nullptr, 0, &length) == 0)
    {
        message.resize(length + 1);
        if (ShaderToolBlobFormatNotFoundMessage(blob, blobSize, constants, numConstants, &message[0], length + 1, nullptr) == 0)
            message.resize(length);
        else
            message = "Couldn't find the required shader permutation in the blob.";
    }

    return message;
}

ShaderFactory::ShaderFactory(nvrhi::IDevice *rendererInterface,
	IFileSystem* fs,
	const std::filesystem::path& basePath)
	: m_Device(rendererInterface)
	, m_fs(fs)
	, m_basePath(basePath)
{
#if DONUT_WITH_AFTERMATH
    if (nvrhi::IAftermathCrashDumpHelper* crashDumpHelper = m_Device->isAftermathEnabled() ? m_Device->getAftermathCrashDumpHelper() : nullptr)
    {
        m_AftermathLookup = MAKE_RC_OBJ_PTR(AftermathShaderLookup, this);
        crashDumpHelper->registerShaderBinaryLookup(m_AftermathLookup);
    }
#endif
}

ShaderFactory::~ShaderFactory()
{
#if DONUT_WITH_AFTERMATH
    if (m_AftermathLookup)
    {
        if (nvrhi::IAftermathCrashDumpHelper* crashDumpHelper = m_Device->getAftermathCrashDumpHelper())
            crashDumpHelper->unregisterShaderBinaryLookup(m_AftermathLookup);
    }
#endif
}

void ShaderFactory::ClearCache()
{
	m_BytecodeCache.clear();
}

nvrhi::AutoPtr<nvrhi::IDataBlob> ShaderFactory::GetBytecode(const char* fileName, const char* entryName)
{
    if (!m_fs)
        return nullptr;
        
    if (!entryName)
        entryName = "main";

    string adjustedName = fileName;
    {
        // Strip only the source suffix; directory names may contain ".hlsl".
        std::filesystem::path sourcePath(adjustedName);
        if (sourcePath.extension() == ".hlsl" || sourcePath.extension() == ".slang")
            adjustedName = sourcePath.replace_extension().generic_string();

        if (entryName && strcmp(entryName, "main"))
            adjustedName += "_" + string(entryName);
    }

    std::filesystem::path shaderFilePath = m_basePath / (adjustedName + ".bin");

    auto& data = m_BytecodeCache[shaderFilePath.generic_string()];

    if (data)
        return data;

    if (NVRHI_FAILED(m_fs->readFile(shaderFilePath, &data)))
    {
        log::error("Couldn't read the binary file for shader %s from %s", fileName, shaderFilePath.generic_string().c_str());
        return nullptr;
    }

    return data;
}

nvrhi::ShaderHandle ShaderFactory::CreateShader(const char* fileName, const char* entryName, const vector<ShaderMacro>* pDefines, const nvrhi::ShaderDesc& desc)
{
    auto byteCode = GetBytecode(fileName, entryName);

    if(!byteCode)
        return nullptr;

    nvrhi::ShaderDesc descCopy = desc;
    descCopy.entryName = entryName;
    if (descCopy.debugName.empty())
        descCopy.debugName = fileName;

    return CreateStaticShader(StaticShader{ byteCode->GetDataPtr(), byteCode->GetSize() }, pDefines, descCopy);
}

nvrhi::ShaderHandle ShaderFactory::CreateShader(const char* fileName, const char* entryName, const vector<ShaderMacro>* pDefines, nvrhi::ShaderType shaderType)
{
    return CreateShader(fileName, entryName, pDefines, nvrhi::ShaderDesc().setShaderType(shaderType));
}

nvrhi::ShaderLibraryHandle ShaderFactory::CreateShaderLibrary(const char* fileName, const std::vector<ShaderMacro>* pDefines)
{
    auto byteCode = GetBytecode(fileName, nullptr);

    if (!byteCode)
        return nullptr;

    return CreateStaticShaderLibrary(StaticShader{ byteCode->GetDataPtr(), byteCode->GetSize() }, pDefines);
}

nvrhi::ShaderHandle ShaderFactory::CreateStaticShader(StaticShader shader, const std::vector<ShaderMacro>* pDefines, const nvrhi::ShaderDesc& desc)
{
    if (!shader.pBytecode || !shader.size)
        return nullptr;

    vector<ShaderToolBlobConstant> constants;
    if (pDefines)
    {
        for (const ShaderMacro& define : *pDefines)
            constants.push_back(ShaderToolBlobConstant{ define.name.c_str(), define.definition.c_str() });
    }

    const void* permutationBytecode = nullptr;
    size_t permutationSize = 0;
    if (ShaderToolBlobFindPermutation(shader.pBytecode, shader.size, constants.data(), constants.size(), &permutationBytecode, &permutationSize) != 0)
    {
        const std::string message = FormatShaderNotFoundMessage(shader.pBytecode, shader.size, constants.data(), constants.size());
        log::error("%s", message.c_str());

        return nullptr;
    }

    nvrhi::ShaderHandle shaderHandle;
    if (IsShaderArtifact(permutationBytecode, permutationSize))
    {
        nvrhi::ShaderProgramHandle program;
        if (NVRHI_FAILED(nvrhi::createShaderProgram(m_Device, permutationBytecode, permutationSize, &program)))
        {
            log::error("Invalid Slang artifact or target does not match the device.");
            return nullptr;
        }
        if (NVRHI_FAILED(FindArtifactShader(program, desc, &shaderHandle)))
            log::error("Slang artifact entry lookup or shader creation failed: %s", desc.entryName.c_str());
        return shaderHandle;
    }
    m_Device->createShader(desc, permutationBytecode, permutationSize, &shaderHandle);
    return shaderHandle;
}

nvrhi::ShaderHandle ShaderFactory::CreateStaticShader(StaticShader shader, const std::vector<ShaderMacro>* pDefines, nvrhi::ShaderType shaderType)
{
    return CreateStaticShader(shader, pDefines, nvrhi::ShaderDesc().setShaderType(shaderType));
}

nvrhi::ShaderHandle ShaderFactory::CreateStaticPlatformShader(StaticShader dxbc, StaticShader dxil, StaticShader spirv, const std::vector<ShaderMacro>* pDefines, const nvrhi::ShaderDesc& desc)
{
    StaticShader shader;
    switch(m_Device->getGraphicsAPI())
    {
        case nvrhi::GraphicsAPI::D3D11:
            shader = dxbc;
            break;
        case nvrhi::GraphicsAPI::D3D12:
            shader = dxil;
            break;
        case nvrhi::GraphicsAPI::VULKAN:
            shader = spirv;
            break;
    }

    return CreateStaticShader(shader, pDefines, desc);
}

nvrhi::ShaderHandle ShaderFactory::CreateStaticPlatformShader(StaticShader dxbc, StaticShader dxil, StaticShader spirv, const std::vector<ShaderMacro>* pDefines, nvrhi::ShaderType shaderType)
{
    return CreateStaticPlatformShader(dxbc, dxil, spirv, pDefines, nvrhi::ShaderDesc().setShaderType(shaderType));
}

nvrhi::ShaderLibraryHandle ShaderFactory::CreateStaticShaderLibrary(StaticShader shader, const std::vector<ShaderMacro>* pDefines)
{
    if (!shader.pBytecode || !shader.size)
        return nullptr;

    vector<ShaderToolBlobConstant> constants;
    if (pDefines)
    {
        for (const ShaderMacro& define : *pDefines)
            constants.push_back(ShaderToolBlobConstant{ define.name.c_str(), define.definition.c_str() });
    }

    const void* permutationBytecode = nullptr;
    size_t permutationSize = 0;
    if (ShaderToolBlobFindPermutation(shader.pBytecode, shader.size, constants.data(), constants.size(), &permutationBytecode, &permutationSize) != 0)
    {
        const std::string message = FormatShaderNotFoundMessage(shader.pBytecode, shader.size, constants.data(), constants.size());
        log::error("%s", message.c_str());

        return nullptr;
    }

    nvrhi::ShaderLibraryHandle shaderLibrary;
    if (IsShaderArtifact(permutationBytecode, permutationSize))
    {
        nvrhi::ShaderProgramHandle program;
        if (NVRHI_FAILED(nvrhi::createShaderProgram(m_Device, permutationBytecode, permutationSize, &program)))
        {
            log::error("Invalid Slang library artifact or target does not match the device.");
            return nullptr;
        }
        return MAKE_RC_OBJ_PTR(ArtifactShaderLibrary, static_cast<nvrhi::IShaderProgram*>(program), permutationBytecode, permutationSize);
    }
    m_Device->createShaderLibrary(permutationBytecode, permutationSize, &shaderLibrary);
    return shaderLibrary;
}

nvrhi::ShaderLibraryHandle ShaderFactory::CreateStaticPlatformShaderLibrary(StaticShader dxil, StaticShader spirv, const std::vector<ShaderMacro>* pDefines)
{
    StaticShader shader;
    switch(m_Device->getGraphicsAPI())
    {
        case nvrhi::GraphicsAPI::D3D12:
            shader = dxil;
            break;
        case nvrhi::GraphicsAPI::VULKAN:
            shader = spirv;
            break;
        default:
            break;
    }

    return CreateStaticShaderLibrary(shader, pDefines);
}

nvrhi::ShaderHandle ShaderFactory::CreateAutoShader(const char* fileName, const char* entryName, StaticShader dxbc, StaticShader dxil, StaticShader spirv, const std::vector<ShaderMacro>* pDefines, const nvrhi::ShaderDesc& desc)
{
    nvrhi::ShaderDesc descCopy = desc;
    descCopy.entryName = entryName;
    if (descCopy.debugName.empty())
        descCopy.debugName = fileName;

    nvrhi::ShaderHandle shader = CreateStaticPlatformShader(dxbc, dxil, spirv, pDefines, descCopy);
    if (shader)
        return shader;
        
    return CreateShader(fileName, entryName, pDefines, desc);
}

nvrhi::ShaderHandle ShaderFactory::CreateAutoShader(const char* fileName, const char* entryName, StaticShader dxbc, StaticShader dxil, StaticShader spirv, const std::vector<ShaderMacro>* pDefines, nvrhi::ShaderType shaderType)
{
    return CreateAutoShader(fileName, entryName, dxbc, dxil, spirv, pDefines, nvrhi::ShaderDesc().setShaderType(shaderType));
}

nvrhi::ShaderLibraryHandle ShaderFactory::CreateAutoShaderLibrary(const char* fileName, StaticShader dxil, StaticShader spirv, const std::vector<ShaderMacro>* pDefines)
{
    nvrhi::ShaderLibraryHandle shader = CreateStaticPlatformShaderLibrary(dxil, spirv, pDefines);
    if (shader)
        return shader;

    return CreateShaderLibrary(fileName, pDefines);
}

std::pair<const void*, size_t> donut::engine::ShaderFactory::FindShaderFromHash(uint64_t hash, nvrhi::PFN_AftermathShaderHashGenerator hashGenerator)
{
    for (auto& entry : m_BytecodeCache)
    {
        const void* shaderBytes = entry.second->GetDataPtr();
        size_t shaderSize = entry.second->GetSize();

        // the bytecode could contain multiple permutations
        std::vector<std::string> permutations;
        auto enumerateCallback = [](void* context, const char* key, size_t keyLength, const void* /*binary*/, size_t /*binarySize*/) -> int
        {
            auto& keys = *static_cast<std::vector<std::string>*>(context);
            if (keyLength == 0)
                keys.push_back("<default>");
            else
                keys.push_back(std::string(key, keyLength));
            return 0;
        };
        ShaderToolBlobEnumeratePermutations(shaderBytes, shaderSize, enumerateCallback, &permutations);

        if (permutations.size() > 1)
        {
            std::vector<ShaderToolBlobConstant> permutationConstants;
            std::unordered_map<std::string, std::string> permutationDefines;
            for (auto permutation : permutations)
            {
                permutationConstants.clear();
                permutationDefines.clear();
                // split the string by spaces to get individual defines
                std::vector<std::string> permutationStrings = donut::string_utils::split(permutation, " ");
                for (auto& s : permutationStrings)
                {
                    std::vector<std::string> keyValue = donut::string_utils::split(s, "=");
                    // Well-formed blob keys are always NAME=VALUE pairs; skip anything
                    // else (e.g. a "<default>" entry from a corrupted blob) instead of
                    // indexing out of bounds.
                    if (keyValue.size() == 2)
                        permutationDefines[keyValue[0]] = keyValue[1];
                }
                // now that we have processed all defines in this permutation, can create the shader constants
                for (const auto& [key, value] : permutationDefines)
                {
                    permutationConstants.push_back(ShaderToolBlobConstant{ key.c_str(), value.c_str() });
                }
                const void* permutationBytecode = nullptr;
                size_t permutationSize = 0;
                if (ShaderToolBlobFindPermutation(shaderBytes, shaderSize, permutationConstants.data(),
                    permutationConstants.size(), &permutationBytecode, &permutationSize) == 0)
                {
                    uint64_t entryHash = hashGenerator(permutationBytecode, permutationSize, m_Device->getGraphicsAPI());
                    if (entryHash == hash)
                    {
                        return std::make_pair(permutationBytecode, permutationSize);
                    }
                }
            }
        }
        else
        {
            uint64_t entryHash = hashGenerator(shaderBytes, shaderSize, m_Device->getGraphicsAPI());
            if (entryHash == hash)
            {
                return std::make_pair(shaderBytes, shaderSize);
            }
        }
    }
    return std::make_pair(nullptr, 0);
}
