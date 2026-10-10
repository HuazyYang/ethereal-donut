#include "SceneImporterImpl.h"

#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/TextureCache.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

#if 0

// Assimp
#include <assimp/Importer.hpp>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>
#include <assimp/scene.h>
#include <assimp/GltfMaterial.h>
#include <assimp/postprocess.h>

#define AI_AUX_SUCCEEDED(expr) ((expr) == aiReturn_SUCCESS)
#define AI_AUX_FAILED(expr) ((expr) != aiReturn_SUCCESS)

using namespace donut::math;

namespace donut::engine {

// Presents one donut VFS blob to Assimp as a seekable read-only stream.
class AssimpIOStreamAdapter : public Assimp::IOStream {
 public:
    AssimpIOStreamAdapter(nvrhi::IDataBlob* pBlob) : m_FileBlob(pBlob), m_FileSize(0), m_Offset(0) {
        if (m_FileBlob) {
            m_FileBlob->AddRef();
            m_FileSize = m_FileBlob->GetSize();
        }
    }

    ~AssimpIOStreamAdapter() { nvrhi::SafeRelease(m_FileBlob); }

    size_t Read(void* pvBuffer, size_t pSize, size_t pCount) override {
        if (!m_FileBlob || !pSize || !pCount)
            return 0;

        const size_t remaining = (m_Offset < m_FileSize) ? (m_FileSize - m_Offset) : 0;
        size_t count = remaining / pSize;
        if (count > pCount)
            count = pCount;

        const size_t byteCount = count * pSize;
        if (byteCount) {
            memcpy(pvBuffer, static_cast<const uint8_t*>(m_FileBlob->GetDataPtr()) + m_Offset, byteCount);
            m_Offset += byteCount;
        }
        return count;
    }

    // The VFS hands out read-only blobs; Assimp only writes when exporting.
    size_t Write(const void*, size_t, size_t) override { return 0; }

    aiReturn Seek(size_t pOffset, aiOrigin pOrigin) override {
        size_t offset = 0;

        switch (pOrigin) {
            case aiOrigin_SET: {
                offset = pOffset;
                break;
            }
            case aiOrigin_CUR: {
                offset = m_Offset + pOffset;
                break;
            }
            case aiOrigin_END: {
                if (pOffset > m_FileSize)
                    return aiReturn_FAILURE;
                offset = m_FileSize - pOffset;
                break;
            }
            default:
                return aiReturn_FAILURE;
        }

        if (offset > m_FileSize)
            return aiReturn_FAILURE;

        m_Offset = offset;
        return aiReturn_SUCCESS;
    }

    size_t Tell() const override { return m_Offset; }
    size_t FileSize() const override { return m_FileSize; }
    void Flush() override {}

 private:
    nvrhi::IDataBlob* m_FileBlob;
    size_t m_FileSize;
    size_t m_Offset;
};

// Routes Assimp's file access through donut's VFS, so sidecar files (.mtl and the
// textures it names by relative path) resolve the way the rest of the engine
// resolves them.
class AssimpIOSystemAdapter : public Assimp::IOSystem {
 public:
    AssimpIOSystemAdapter(vfs::IFileSystem* pFS) : m_FS{pFS} {}

    bool Exists(const char* pFile) const override {
        nvrhi::AutoPtr<nvrhi::IDataBlob> pBlob;
        return m_FS && NVRHI_SUCCEEDED(m_FS->readFile(pFile, &pBlob)) && pBlob;
    }

    char getOsSeparator() const override { return '/'; }

    Assimp::IOStream* Open(const char* pFile, const char* pMode = "rb") override {
        // Read-only: refuse write modes rather than return a stream that silently
        // drops everything written to it.
        if (pMode && (strchr(pMode, 'w') || strchr(pMode, 'a')))
            return nullptr;

        if (!m_FS)
            return nullptr;

        nvrhi::AutoPtr<nvrhi::IDataBlob> pBlob;
        if (NVRHI_FAILED(m_FS->readFile(pFile, &pBlob)) || !pBlob)
            return nullptr;

        auto pStream = new AssimpIOStreamAdapter(pBlob);
        return pStream;
    }

    void Close(Assimp::IOStream* pFile) override { delete pFile; }

 private:
    nvrhi::AutoPtr<vfs::IFileSystem> m_FS;
};

AssimpSceneImporter::AssimpSceneImporter(vfs::IFileSystem* fs, SceneTypeFactory* sceneTypeFactory)
    : m_fs(fs), m_SceneTypeFactory(sceneTypeFactory) {}

AssimpSceneImporter::~AssimpSceneImporter() {}

nvrhi::FRESULT AssimpSceneImporter::Load(const std::filesystem::path& fileName, TextureCache& textureCache,
                                  SceneLoadingStats& stats, ThreadPool* threadPool,
                                  SceneImportResult& result) {
    result.rootNode.Reset();

    Assimp::Importer importer;
    importer.SetIOHandler(new AssimpIOSystemAdapter(m_fs));

    const std::string normalizedFileName = fileName.lexically_normal().generic_string();

    // Only triangulation and tangent generation; no handedness or UV conversion,
    // because donut consumes Assimp's native right-handed, Y-up output directly.
    const aiScene* pAiScene = importer.ReadFile(normalizedFileName.c_str(),
                                                aiProcess_Triangulate | aiProcess_CalcTangentSpace);

    if (!pAiScene || !pAiScene->mRootNode) {
        log::error("Couldn't load scene file '%s': %s", normalizedFileName.c_str(), importer.GetErrorString());
        return nvrhi::FE_GENERIC_ERROR;
    }

    const std::filesystem::path fileParentPath = fileName.parent_path();

    std::unordered_map<std::string, nvrhi::AutoPtr<LoadedTexture>> imageCache;

    auto LoadTexture = [fileParentPath, pAiScene, threadPool, &textureCache, &imageCache](
                           const aiString& inlinePath, bool sRGB) -> nvrhi::AutoPtr<LoadedTexture> {
        if (inlinePath.length == 0)
            return nullptr;

        const std::string cacheKey = std::string(inlinePath.C_Str()) + (sRGB ? "#srgb" : "#linear");
        auto it = imageCache.find(cacheKey);
        if (it != imageCache.end())
            return it->second;

        nvrhi::AutoPtr<LoadedTexture> loadedTexture;

        // Textures embedded in the container (GLB, FBX) are named "*<index>".
        const TextureLoadOptions loadOptions{ SRGBModeFromBool(sRGB) };
        const aiTexture* pAiTexture = pAiScene->GetEmbeddedTexture(inlinePath.C_Str());
        if (pAiTexture) {
            // mHeight == 0 means the payload is a compressed file image, not raw texels.
            if (pAiTexture->mHeight == 0) {
                nvrhi::AutoPtr<nvrhi::IDataBlob> textureData;
                if (NVRHI_SUCCEEDED(nvrhi::CreateBlob(pAiTexture->mWidth, &textureData)) && textureData) {
                    memcpy(textureData->GetDataPtr(), pAiTexture->pcData, pAiTexture->mWidth);

                    const std::string name = inlinePath.C_Str();
                    const std::string mimeType = pAiTexture->achFormatHint[0]
                                                     ? std::string("image/") + pAiTexture->achFormatHint
                                                     : std::string();

                    if (threadPool)
                        loadedTexture =
                            textureCache.LoadTextureFromMemoryAsync(textureData, name, mimeType, loadOptions, *threadPool);
                    else
                        loadedTexture = textureCache.LoadTextureFromMemoryDeferred(textureData, name, mimeType, loadOptions);
                }
            } else {
                log::warning("Uncompressed embedded texture '%s' is not supported.", inlinePath.C_Str());
            }
        } else {
            // Assimp returns the path verbatim from the source file, which for .mtl
            // files written on Windows means backslashes.
            std::string relativePath = inlinePath.C_Str();
            std::replace(relativePath.begin(), relativePath.end(), '\\', '/');

            const std::filesystem::path texturePath = (fileParentPath / relativePath).lexically_normal();

            if (threadPool)
                loadedTexture = textureCache.LoadTextureFromFileAsync(texturePath, loadOptions, *threadPool);
            else
                loadedTexture = textureCache.LoadTextureFromFileDeferred(texturePath, loadOptions);
        }

        imageCache[cacheKey] = loadedTexture;
        return loadedTexture;
    };

    // ---------------------------------------------------------------- materials
    std::vector<nvrhi::AutoPtr<Material>> materials((size_t)pAiScene->mNumMaterials);

    for (uint32_t materialIndex = 0; materialIndex < pAiScene->mNumMaterials; ++materialIndex) {
        const aiMaterial* pAiMat = pAiScene->mMaterials[materialIndex];

        auto matInfo = m_SceneTypeFactory->CreateMaterial();
        materials[materialIndex] = matInfo;

        aiString matName;
        if (AI_AUX_SUCCEEDED(pAiMat->Get(AI_MATKEY_NAME, matName)))
            matInfo->name = matName.C_Str();

        matInfo->modelFileName = normalizedFileName;
        matInfo->materialIndexInModel = int(materialIndex);

        // A metallic factor is only published by PBR sources. Classic formats such
        // as OBJ carry Kd/Ks/Ns, which map onto the specular-glossiness model.
        float metallicFactor = 0.f;
        const bool hasMetallicFactor = AI_AUX_SUCCEEDED(pAiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor));

        aiString texturePath;

        if (!hasMetallicFactor) {
            matInfo->useSpecularGlossModel = true;

            // Base colour is loaded as UNORM, not sRGB: donut's own GltfImporter does the
            // same, and this sample's shaders (ported from the GFSDK VXGI sample) run a
            // gamma-space pipeline that expects un-linearised albedo.
            if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath)))
                matInfo->baseOrDiffuseTexture = LoadTexture(texturePath, false);
            if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_SPECULAR, 0, &texturePath)))
                matInfo->metalRoughOrSpecularTexture = LoadTexture(texturePath, true);

            // Prefer the PBR key when the source provides one, else the classic color.
            if (AI_AUX_FAILED(pAiMat->Get(AI_MATKEY_BASE_COLOR, (aiColor3D&)matInfo->baseOrDiffuseColor)))
                pAiMat->Get(AI_MATKEY_COLOR_DIFFUSE, (aiColor3D&)matInfo->baseOrDiffuseColor);
            if (AI_AUX_FAILED(pAiMat->Get(AI_MATKEY_SPECULAR_FACTOR, (aiColor3D&)matInfo->specularColor)))
                pAiMat->Get(AI_MATKEY_COLOR_SPECULAR, (aiColor3D&)matInfo->specularColor);

            // Wavefront shininess is a Blinn-Phong exponent; convert to roughness.
            float shininess = 0.f;
            if (AI_AUX_SUCCEEDED(pAiMat->Get(AI_MATKEY_SHININESS, shininess)) && shininess > 0.f)
                matInfo->roughness = dm::clamp(sqrtf(2.f / (shininess + 2.f)), 0.f, 1.f);
            else
                matInfo->roughness = 1.f;
        } else {
            matInfo->metalness = metallicFactor;

            if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath)))
                matInfo->baseOrDiffuseTexture = LoadTexture(texturePath, false);
            if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
                                                    &texturePath)))
                matInfo->metalRoughOrSpecularTexture = LoadTexture(texturePath, false);

            if (AI_AUX_FAILED(pAiMat->Get(AI_MATKEY_BASE_COLOR, (aiColor3D&)matInfo->baseOrDiffuseColor)))
                pAiMat->Get(AI_MATKEY_COLOR_DIFFUSE, (aiColor3D&)matInfo->baseOrDiffuseColor);

            float roughnessFactor = 1.f;
            if (AI_AUX_SUCCEEDED(pAiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor)))
                matInfo->roughness = roughnessFactor;
        }

        if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_TRANSMISSION, 0, &texturePath))) {
            matInfo->transmissionTexture = LoadTexture(texturePath, false);
            pAiMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, matInfo->transmissionFactor);
        }

        if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_EMISSIVE, 0, &texturePath)))
            matInfo->emissiveTexture = LoadTexture(texturePath, true);
        if (AI_AUX_FAILED(pAiMat->Get(AI_MATKEY_COLOR_EMISSIVE, (aiColor3D&)matInfo->emissiveColor)))
            matInfo->emissiveColor = 0.f;

        matInfo->emissiveIntensity = dm::maxComponent(matInfo->emissiveColor);
        if (matInfo->emissiveIntensity > 0.f)
            matInfo->emissiveColor /= matInfo->emissiveIntensity;
        else
            matInfo->emissiveIntensity = 1.f;

        // OBJ exporters write the normal map under "bump"/"map_Kn" as often as
        // "norm", which Assimp files under HEIGHT rather than NORMALS.
        if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_NORMALS, 0, &texturePath)))
            matInfo->normalTexture = LoadTexture(texturePath, false);
        else if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_HEIGHT, 0, &texturePath)))
            matInfo->normalTexture = LoadTexture(texturePath, false);
        if (AI_AUX_FAILED(pAiMat->Get(AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_NORMALS, 0),
                                      matInfo->normalTextureScale)))
            matInfo->normalTextureScale = 1.f;

        if (AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &texturePath)))
            matInfo->occlusionTexture = LoadTexture(texturePath, false);
        if (AI_AUX_FAILED(pAiMat->Get(AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_AMBIENT_OCCLUSION, 0),
                                      matInfo->occlusionStrength)))
            matInfo->occlusionStrength = 1.f;

        // donut only supports a scale on the normal map transform.
        aiUVTransform uvTransform;
        if (AI_AUX_SUCCEEDED(pAiMat->Get(AI_MATKEY_UVTRANSFORM(aiTextureType_NORMALS, 0), uvTransform)))
            matInfo->normalTextureTransformScale = float2(uvTransform.mScaling.x, uvTransform.mScaling.y);

        // An explicit cutout map (.mtl "map_d") keeps its own slot instead of being
        // folded into the base color alpha.
        const bool hasOpacityTexture = AI_AUX_SUCCEEDED(pAiMat->GetTexture(aiTextureType_OPACITY, 0, &texturePath));
        if (hasOpacityTexture)
            matInfo->opacityTexture = LoadTexture(texturePath, false);

        pAiMat->Get(AI_MATKEY_OPACITY, matInfo->opacity);

        int twoSided = 0;
        if (AI_AUX_SUCCEEDED(pAiMat->Get(AI_MATKEY_TWOSIDED, twoSided)))
            matInfo->doubleSided = twoSided != 0;

        aiString alphaMode;
        if (AI_AUX_SUCCEEDED(pAiMat->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode))) {
            if (strcmp(alphaMode.C_Str(), "MASK") == 0)
                matInfo->domain = MaterialDomain::AlphaTested;
            else if (strcmp(alphaMode.C_Str(), "BLEND") == 0)
                matInfo->domain = MaterialDomain::AlphaBlended;
            else
                matInfo->domain = MaterialDomain::Opaque;

            pAiMat->Get(AI_MATKEY_GLTF_ALPHACUTOFF, matInfo->alphaCutoff);
        } else if (hasOpacityTexture) {
            matInfo->domain = MaterialDomain::AlphaTested;
            // Cutout geometry (foliage, chains) is modelled as single-sided quads.
            matInfo->doubleSided = true;
        } else if (matInfo->opacity < 1.f) {
            matInfo->domain = MaterialDomain::AlphaBlended;
        } else {
            matInfo->domain = MaterialDomain::Opaque;
        }
    }

    // ------------------------------------------------------------------- meshes
    auto IsSupportedMesh = [](const aiMesh* pAiMesh) {
        return (pAiMesh->mPrimitiveTypes & (aiPrimitiveType_LINE | aiPrimitiveType_TRIANGLE)) != 0 &&
               pAiMesh->mNumVertices > 0;
    };

    // Count everything first so the shared buffers are sized exactly once.
    size_t totalIndices = 0;
    size_t totalVertices = 0;

    for (uint32_t meshIndex = 0; meshIndex < pAiScene->mNumMeshes; ++meshIndex) {
        const aiMesh* pAiMesh = pAiScene->mMeshes[meshIndex];
        if (!IsSupportedMesh(pAiMesh))
            continue;

        for (uint32_t faceIndex = 0; faceIndex < pAiMesh->mNumFaces; ++faceIndex)
            totalIndices += pAiMesh->mFaces[faceIndex].mNumIndices;
        totalVertices += pAiMesh->mNumVertices;
    }

    auto buffers = MAKE_RC_OBJ_PTR(BufferGroup);
    buffers->indexData.resize(totalIndices);
    buffers->positionData.resize(totalVertices);
    buffers->normalData.resize(totalVertices);
    buffers->tangentData.resize(totalVertices);
    buffers->texcoord1Data.resize(totalVertices);
    buffers->radiusData.resize(totalVertices);

    totalIndices = 0;
    totalVertices = 0;

    std::unordered_map<aiMesh*, nvrhi::AutoPtr<MeshInfo>> meshes;
    nvrhi::AutoPtr<Material> emptyMaterial;

    for (uint32_t meshIndex = 0; meshIndex < pAiScene->mNumMeshes; ++meshIndex) {
        aiMesh* pAiMesh = pAiScene->mMeshes[meshIndex];
        if (!IsSupportedMesh(pAiMesh))
            continue;

        auto meshInfo = m_SceneTypeFactory->CreateMesh();
        if (pAiMesh->mName.length)
            meshInfo->name = pAiMesh->mName.C_Str();
        meshInfo->buffers = buffers;
        meshInfo->indexOffset = (uint32_t)totalIndices;
        meshInfo->vertexOffset = (uint32_t)totalVertices;

        if (pAiMesh->mPrimitiveTypes == aiPrimitiveType_LINE)
            meshInfo->type = MeshType::CurvePolytubes;

        auto geometry = m_SceneTypeFactory->CreateMeshGeometry();

        // Indices are mesh-relative; the vertex offset is applied by the renderer.
        uint32_t* indexDst = buffers->indexData.data() + totalIndices;
        size_t indexCount = 0;
        for (uint32_t faceIndex = 0; faceIndex < pAiMesh->mNumFaces; ++faceIndex) {
            const aiFace& face = pAiMesh->mFaces[faceIndex];
            for (uint32_t i = 0; i < face.mNumIndices; ++i) {
                *indexDst++ = face.mIndices[i];
                ++indexCount;
            }
        }

        box3 bounds = box3::empty();

        float3* positionDst = buffers->positionData.data() + totalVertices;
        for (uint32_t v = 0; v < pAiMesh->mNumVertices; ++v) {
            const aiVector3D& p = pAiMesh->mVertices[v];
            const float3 position = float3(p.x, p.y, p.z);
            *positionDst++ = position;
            bounds |= position;
        }

        if (pAiMesh->HasNormals()) {
            uint32_t* normalDst = buffers->normalData.data() + totalVertices;
            for (uint32_t v = 0; v < pAiMesh->mNumVertices; ++v) {
                const aiVector3D& n = pAiMesh->mNormals[v];
                *normalDst++ = vectorToSnorm8(float3(n.x, n.y, n.z));
            }
        }

        if (pAiMesh->HasTangentsAndBitangents() && pAiMesh->HasNormals()) {
            uint32_t* tangentDst = buffers->tangentData.data() + totalVertices;
            for (uint32_t v = 0; v < pAiMesh->mNumVertices; ++v) {
                const aiVector3D& t = pAiMesh->mTangents[v];
                const aiVector3D& b = pAiMesh->mBitangents[v];
                const aiVector3D& n = pAiMesh->mNormals[v];

                const float3 tangent = float3(t.x, t.y, t.z);
                const float3 bitangent = float3(b.x, b.y, b.z);
                const float3 normal = float3(n.x, n.y, n.z);

                // donut packs the bitangent handedness into .w. GltfImporter writes
                // `dot(cross(N,T), B) > 0 ? -1 : 1`, but it derives B from dPdt/dTds
                // whereas Assimp hands us mBitangents with the opposite orientation, so
                // the inverted comparison here produces the *same* handedness. Measured:
                // flipping this to match GltfImporter's literal form drives the two
                // importer paths from 3.2% to 24.1% differing pixels.
                const float sign = (dot(cross(normal, tangent), bitangent) < 0.f) ? -1.f : 1.f;
                *tangentDst++ = vectorToSnorm8(float4(tangent, sign));
            }
        }

        if (pAiMesh->HasTextureCoords(0)) {
            float2* texcoordDst = buffers->texcoord1Data.data() + totalVertices;
            for (uint32_t v = 0; v < pAiMesh->mNumVertices; ++v) {
                const aiVector3D& uv = pAiMesh->mTextureCoords[0][v];
                *texcoordDst++ = float2(uv.x, uv.y);
            }
        }

        if (pAiMesh->mMaterialIndex < materials.size()) {
            geometry->material = materials[pAiMesh->mMaterialIndex];
        } else {
            log::warning("Geometry for mesh '%s' doesn't have a material.", meshInfo->name.c_str());
            if (!emptyMaterial) {
                emptyMaterial = MAKE_RC_OBJ_PTR(Material);
                emptyMaterial->name = "(empty)";
            }
            geometry->material = emptyMaterial;
        }

        geometry->indexOffsetInMesh = 0;
        geometry->vertexOffsetInMesh = 0;
        geometry->numIndices = (uint32_t)indexCount;
        geometry->numVertices = pAiMesh->mNumVertices;
        geometry->objectSpaceBounds = bounds;
        geometry->type = (pAiMesh->mPrimitiveTypes == aiPrimitiveType_LINE) ? MeshGeometryPrimitiveType::Lines
                                                                           : MeshGeometryPrimitiveType::Triangles;

        meshInfo->objectSpaceBounds = bounds;
        meshInfo->totalIndices = geometry->numIndices;
        meshInfo->totalVertices = geometry->numVertices;
        meshInfo->geometries.push_back(geometry);

        totalIndices += geometry->numIndices;
        totalVertices += geometry->numVertices;

        meshes[pAiMesh] = meshInfo;
    }

    // ------------------------------------------------------------------ cameras
    std::vector<std::pair<aiString, nvrhi::AutoPtr<SceneCamera>>> cameras((size_t)pAiScene->mNumCameras);

    for (uint32_t cameraIndex = 0; cameraIndex < pAiScene->mNumCameras; ++cameraIndex) {
        const aiCamera* src = pAiScene->mCameras[cameraIndex];

        auto perspectiveCamera = MAKE_RC_OBJ_PTR(PerspectiveCamera);
        perspectiveCamera->zNear = src->mClipPlaneNear;
        perspectiveCamera->zFar = src->mClipPlaneFar;
        // Assimp stores a horizontal half-angle; donut wants a vertical full angle.
        perspectiveCamera->verticalFov = 2.f * atanf(tanf(src->mHorizontalFOV) / std::max(src->mAspect, 1e-4f));
        if (src->mAspect > 0.f)
            perspectiveCamera->aspectRatio = src->mAspect;

        cameras[cameraIndex] = std::make_pair(src->mName, nvrhi::AutoPtr<SceneCamera>(perspectiveCamera));
    }

    // ------------------------------------------------------------------- lights
    std::vector<std::pair<aiString, nvrhi::AutoPtr<Light>>> lights((size_t)pAiScene->mNumLights);

    for (uint32_t lightIndex = 0; lightIndex < pAiScene->mNumLights; ++lightIndex) {
        const aiLight* src = pAiScene->mLights[lightIndex];
        nvrhi::AutoPtr<Light> dst;

        switch (src->mType) {
            case aiLightSource_DIRECTIONAL: {
                nvrhi::AutoPtr<DirectionalLight> directional{
                    query_cast<DirectionalLight>(m_SceneTypeFactory->CreateLeaf("DirectionalLight").Get())};
                directional->color = float3(src->mColorDiffuse.r, src->mColorDiffuse.g, src->mColorDiffuse.b);
                dst = directional;
                break;
            }
            case aiLightSource_POINT: {
                nvrhi::AutoPtr<PointLight> point{
                    query_cast<PointLight>(m_SceneTypeFactory->CreateLeaf("PointLight").Get())};
                point->color = float3(src->mColorDiffuse.r, src->mColorDiffuse.g, src->mColorDiffuse.b);
                dst = point;
                break;
            }
            case aiLightSource_SPOT: {
                nvrhi::AutoPtr<SpotLight> spot{query_cast<SpotLight>(m_SceneTypeFactory->CreateLeaf("SpotLight").Get())};
                spot->color = float3(src->mColorDiffuse.r, src->mColorDiffuse.g, src->mColorDiffuse.b);
                spot->innerAngle = dm::degrees(src->mAngleInnerCone);
                spot->outerAngle = dm::degrees(src->mAngleOuterCone);
                dst = spot;
                break;
            }
            default:
                break;
        }

        if (dst)
            lights[lightIndex] = std::make_pair(src->mName, dst);
    }

    // -------------------------------------------------------------- scene graph
    auto graph = MAKE_RC_OBJ_PTR(SceneGraph);
    auto root = MAKE_RC_OBJ_PTR(SceneGraphNode);
    std::unordered_map<aiNode*, SceneGraphNode*> nodeMap;
    std::vector<std::pair<aiNode*, aiMesh*>> skinnedNodes;

    struct StackItem {
        nvrhi::AutoPtr<SceneGraphNode> dstParent;
        aiNode* const* srcNodes = nullptr;
        size_t srcCount = 0;
    };
    std::vector<StackItem> stack;

    root->SetName(fileName.filename().generic_string());

    // The Assimp root node maps onto the root created above, so descend straight
    // into its children rather than adding a redundant level.
    aiNode* const* rootChildren = pAiScene->mRootNode->mChildren;
    nodeMap[pAiScene->mRootNode] = root;

    StackItem context;
    context.dstParent = root;
    context.srcNodes = rootChildren;
    context.srcCount = pAiScene->mRootNode->mNumChildren;

    for (;;) {
        while (context.srcCount == 0) {
            if (stack.empty()) {
                context.srcNodes = nullptr;
                break;
            }
            context = stack.back();
            stack.pop_back();
        }

        if (!context.srcNodes)
            break;

        aiNode* src = *context.srcNodes;
        ++context.srcNodes;
        --context.srcCount;

        auto dst = MAKE_RC_OBJ_PTR(SceneGraphNode);
        nodeMap[src] = dst;

        if (src->mName.length)
            dst->SetName(src->mName.C_Str());

        // Assimp matrices are row-major; affine3 takes three basis rows plus the
        // translation row.
        const aiMatrix4x4& m = src->mTransformation;
        if (!m.IsIdentity()) {
            affine3 aff = affine3(float3(m.a1, m.b1, m.c1), float3(m.a2, m.b2, m.c2), float3(m.a3, m.b3, m.c3),
                                  float3(m.a4, m.b4, m.c4));

            double3 translation;
            double3 scaling;
            dquat rotation;
            decomposeAffine(daffine3(aff), &translation, &rotation, &scaling);
            dst->SetTransform(&translation, &rotation, &scaling);
        }

        graph->Attach(context.dstParent, dst);

        // One aiMesh becomes one MeshInstance. A node with several meshes gets one
        // child node per mesh, because a SceneGraphNode holds a single leaf.
        std::vector<nvrhi::AutoPtr<MeshInfo>> nodeMeshes;
        for (uint32_t i = 0; i < src->mNumMeshes; ++i) {
            aiMesh* pAiMesh = pAiScene->mMeshes[src->mMeshes[i]];

            if (pAiMesh->HasBones()) {
                // Skinned meshes need the whole graph in place first.
                skinnedNodes.push_back(std::make_pair(src, pAiMesh));
                continue;
            }

            auto found = meshes.find(pAiMesh);
            if (found != meshes.end())
                nodeMeshes.push_back(found->second);
        }

        if (nodeMeshes.size() == 1) {
            dst->SetLeaf(m_SceneTypeFactory->CreateMeshInstance(nodeMeshes[0]));
        } else {
            for (auto& mesh : nodeMeshes) {
                auto meshNode = MAKE_RC_OBJ_PTR(SceneGraphNode);
                meshNode->SetName(mesh->name);
                graph->Attach(dst, meshNode);
                meshNode->SetLeaf(m_SceneTypeFactory->CreateMeshInstance(mesh));
            }
        }

        // Assimp associates cameras and lights with nodes by name.
        for (auto& camera : cameras) {
            if (camera.second && camera.first == src->mName) {
                if (dst->GetLeaf()) {
                    auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                    graph->Attach(dst, node);
                    node->SetLeaf(camera.second);
                } else {
                    dst->SetLeaf(camera.second);
                }
                break;
            }
        }

        for (auto& light : lights) {
            if (light.second && light.first == src->mName) {
                if (dst->GetLeaf()) {
                    auto node = MAKE_RC_OBJ_PTR(SceneGraphNode);
                    graph->Attach(dst, node);
                    node->SetLeaf(light.second);
                } else {
                    dst->SetLeaf(light.second);
                }
                break;
            }
        }

        if (src->mNumChildren > 0) {
            stack.push_back(context);
            context.dstParent = dst;
            context.srcNodes = src->mChildren;
            context.srcCount = src->mNumChildren;
        }
    }

    // Skinned meshes are attached once every joint node exists in the graph.
    for (auto& [srcNode, srcMesh] : skinnedNodes) {
        auto found = meshes.find(srcMesh);
        if (found == meshes.end())
            continue;

        auto dstNode = nodeMap[srcNode];
        if (!dstNode)
            continue;

        auto prototype = found->second;
        prototype->isSkinPrototype = true;

        auto skinnedInstance = m_SceneTypeFactory->CreateSkinnedMeshInstance(m_SceneTypeFactory, prototype);

        for (uint32_t boneIndex = 0; boneIndex < srcMesh->mNumBones; ++boneIndex) {
            const aiBone* bone = srcMesh->mBones[boneIndex];
            aiNode* jointNode = pAiScene->mRootNode->FindNode(bone->mName);
            if (!jointNode)
                continue;

            auto dstJoint = nodeMap[jointNode];
            if (dstJoint)
                dstJoint->SetLeaf(MAKE_RC_OBJ_PTR(SkinnedMeshReference, skinnedInstance));
        }

        dstNode->SetLeaf(skinnedInstance);
    }

    result.rootNode = root;

    return nvrhi::FS_OK;
}

}  // namespace donut::engine

#endif
