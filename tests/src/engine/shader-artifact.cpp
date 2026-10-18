// Artifact loading keeps the existing ShaderFactory options and library behavior.
#include "recording-device.h"
#include <donut/engine/ShaderFactory.h>
#include <donut/core/log.h>
#include <ShaderToolArtifact.h>
#include <cstdio>
#include <stdexcept>
using namespace nvrhi;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("line " + std::to_string(__LINE__) + ": " #x); } while (0)
int main()
{
    donut::log::ConsoleApplicationMode();
    try {
        auto device = MAKE_RC_OBJ_PTR(nvrhi_test::RecordingDevice, GraphicsAPI::VULKAN);
        donut::engine::ShaderFactory factory(device, nullptr, "");
        const char code[] = "SPIR-V fixture";
        ShaderToolArtifactEntry entry{};
        entry.Name = "main"; entry.Symbol = "nativeMain"; entry.Stage = ST_STAGE_COMPUTE;
        entry.ThreadGroupSize[0] = entry.ThreadGroupSize[1] = entry.ThreadGroupSize[2] = 1;
        entry.GlobalParamsAlignment = entry.EntryParamsAlignment = 1;
        entry.Code = code; entry.CodeSize = sizeof(code);
        ShaderToolArtifactDesc artifact{};
        artifact.Target = ST_TARGET_SPIRV; artifact.Entries = &entry; artifact.EntryCount = 1;
        size_t size = 0;
        CHECK(!ShaderToolArtifactWrite(&artifact, nullptr, 0, &size));
        std::vector<unsigned char> bytes(size);
        CHECK(!ShaderToolArtifactWrite(&artifact, bytes.data(), bytes.size(), &size));
        donut::engine::StaticShader staticShader{bytes.data(), bytes.size()};
        ShaderDesc options;
        options.shaderType = ShaderType::Compute; options.debugName = "custom-debug";
        options.hlslExtensionsUAV = 19; options.useSpecificShaderExt = true;
        CustomSemantic semantic{}; uint32_t swizzle[4]{};
        options.numCustomSemantics = 1; options.pCustomSemantics = &semantic;
        options.fastGSFlags = FastGeometryShaderFlags::StrictApiOrder; options.pCoordinateSwizzling = swizzle;
        auto shader = factory.CreateStaticShader(staticShader, nullptr, options);
        CHECK(shader);
        const auto& desc = shader->getDesc();
        CHECK(desc.entryName == "nativeMain" && desc.debugName == options.debugName);
        CHECK(desc.hlslExtensionsUAV == 19 && desc.useSpecificShaderExt && desc.pCustomSemantics == &semantic);
        CHECK(desc.numCustomSemantics == 1 && desc.fastGSFlags == options.fastGSFlags && desc.pCoordinateSwizzling == swizzle);
        auto library = factory.CreateStaticShaderLibrary(staticShader, nullptr);
        CHECK(library);
        std::fill(bytes.begin(), bytes.end(), 0); // Library owns the selected artifact.
        ShaderHandle entryShader;
        CHECK(library->getShader("main", ShaderType::Compute, &entryShader) == FS_OK);
        CHECK(entryShader->getDesc().entryName == "nativeMain");
        CHECK(library->getShader("nativeMain", ShaderType::Compute, &entryShader) == FS_OK);
        IShader* bad = reinterpret_cast<IShader*>(1);
        CHECK(library->getShader("missing", ShaderType::Compute, &bad) == FE_NOT_FOUND && !bad);
        CHECK(library->getShader("main", ShaderType::Pixel, &bad) == FE_NOT_FOUND && !bad);
        const void* owned = nullptr;
        library->getBytecode(&owned, &size);
        CHECK(size == bytes.size() && !memcmp(owned, "NSTA", 4));
        // Legacy raw executable pass-through preserves every option as well.
        shader = factory.CreateStaticShader({code, sizeof(code)}, nullptr, options);
        CHECK(shader && shader->getDesc().entryName == "main" && shader->getDesc().hlslExtensionsUAV == 19);
        std::puts("ShaderFactory artifact contract passed");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
