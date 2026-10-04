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

#include <donut/core/chunk/chunkFile.h>
#include <donut/core/vfs/VFS.h>

#include <cassert>
#include <cstring>

namespace donut::chunk
{

//
// Header
//

struct ChunkFile::Header
{

    uint8_t signature[8];
    uint32_t version,
             chunkCount,
             chunkTableOffset;

    static char const * validSignature() { return "NVDACHNK"; }

    static uint32_t currentVersion() { return 0x100; }

    bool isValid() const
    {
        return memcmp(signature, validSignature(),
            std::size(signature) * sizeof(uint8_t))==0;
    }
};

//
// Chunks Table
//

struct ChunkFile::ChunkTableEntry
{
    ChunkId  chunkId;
    uint32_t chunkType,
             chunkVersion;
    size_t offset,
           size;
};

//
// Implementation
//

ChunkFile::~ChunkFile() {
    nvrhi::SafeRelease(_data);
    for(auto chunk : _chunks) {
        if (chunk->deleteUserData) delete[] (uint8_t *)chunk->data;
        delete chunk;
    }
}

ChunkId ChunkFile::addChunk(uint32_t type, uint32_t version, void const *data,
                            size_t size) {
    ChunkId chunkId = (uint32_t)_chunks.size() + 1;

    auto chunk = new Chunk({chunkId, type, version, 0u, size, data, true});

    _chunks.push_back(chunk);

    return chunkId;
}

Chunk const * ChunkFile::getChunk(ChunkId const chunkId) const
{
    if (!chunkId.valid()) {
        return nullptr;
    }

    // XXXX manuelk augment this w/ map for faster search
    for (auto const & chunk : _chunks) {
        if (chunk && chunk->chunkId == chunkId) {
            return chunk;
        }
    }
    return nullptr;
}

void ChunkFile::getChunks(uint32_t chunkType, std::vector<Chunk const *> & result) const
{
    result.clear();
    for (auto const & chunk : _chunks)
        if (chunk && chunk->chunkType == chunkType)
            result.push_back(chunk);
}

void ChunkFile::reset()
{
    _filepath.clear();
    _chunks.clear();
    nvrhi::SafeRelease(_data);
    for(auto chunk : _chunks) {
        if (chunk->deleteUserData) delete[] (uint8_t *)chunk->data;
        delete chunk;
    }
    _chunks.clear();
}

nvrhi::FRESULT ChunkFile::deserialize(
    nvrhi::IDataBlob *blobPtr, char const * filepath, ChunkFile **ppChunkFile)
{

    if (auto const blob = blobPtr)
    {
        if (!blob->GetDataPtr() || blob->GetSize() < sizeof(Header))
        {
            log::error("ChunkFile '%s' : invalid header", filepath);
            return nvrhi::FE_GENERIC_ERROR;
        }

        uint8_t const * data = reinterpret_cast<uint8_t const *>(blob->GetDataPtr());

        Header const & header = *(Header const *)(data);

        if (!header.isValid())
        {
            log::error("ChunkFile '%s' : invalid chunkfile signature", filepath);
        }

        uint32_t nchunks = header.chunkCount;
        if (nchunks == 0 || nchunks > 1000000)
        {
            log::error("ChunkFile '%s' : invalid number of chunks in file", filepath);
            return nvrhi::FE_GENERIC_ERROR;
        }

        if (blob->GetSize() < header.chunkTableOffset + nchunks * sizeof(ChunkTableEntry))
        {
            log::error("ChunkFile '%s' : invalid chunks table", filepath);
            return nvrhi::FE_GENERIC_ERROR;
        }

        ChunkTableEntry const * chunktable =
            (ChunkTableEntry const *)(data + header.chunkTableOffset);

        auto result = MAKE_RC_OBJ(ChunkFile);

        result->_chunks.reserve(nchunks);

        for (uint32_t index = 0; index < nchunks; index++)
        {

            ChunkTableEntry const & e = chunktable[index];

            if (blob->GetSize() < e.offset + e.size) {
                log::error("ChunkFile '%s' : chunk %d invalid size/offset", filepath, e.chunkId);
                return nvrhi::FE_GENERIC_ERROR;
            }

            auto chunk = new Chunk
                ({e.chunkId, e.chunkType, e.chunkVersion, e.offset, e.size, data+e.offset, false});

            result->_chunks.push_back(chunk);
        }

        result->_filepath = filepath;
        // The chunks point into the blob, so the file keeps it alive for its own lifetime;
        // ~ChunkFile releases this reference. The caller keeps its own.
        result->_data = blob;
        nvrhi::SafeAddRef(result->_data);

        if(ppChunkFile) {
            *ppChunkFile = result;
            result->AddRef();
        }
        result->Release();

        return nvrhi::FS_OK;
    }
    else
    {
        log::error("ChunkFile '%s' : no data", filepath);
    }
    return nvrhi::FE_GENERIC_ERROR;
}

nvrhi::FRESULT ChunkFile::serialize(nvrhi::IDataBlob **ppBlob) const {

    uint32_t nchunks = (uint32_t)_chunks.size();

    size_t chunkTableSize = nchunks*sizeof(ChunkTableEntry);

    size_t blobSize = 0;
    blobSize += sizeof(Header);
    blobSize += chunkTableSize;
    for (auto const & chunk : _chunks)
        blobSize += chunk->size;

    nvrhi::FRESULT fr;
    nvrhi::IDataBlob *pBlob;

    if (NVRHI_SUCCEEDED(fr = nvrhi::CreateBlob(blobSize, &pBlob)))
    {
        size_t offset = 0;
        auto data = (uint8_t *)pBlob->GetDataPtr();

        // write header
        {
            Header & header = *(Header *)(data+offset);
            header = {{}, Header::currentVersion(), nchunks, sizeof(Header)};
            memcpy(header.signature, Header::validSignature(), 8);

            offset += sizeof(Header);
        }

        // write chunks table

        {
            ChunkTableEntry * chunkTable = (ChunkTableEntry *)(data+offset);

            for (size_t i=0, chunkOffset=offset+chunkTableSize; i<nchunks; ++i)
            {
                Chunk * chunk = const_cast<Chunk *>(_chunks[i]);

                chunkTable[i] = {
                    chunk->chunkId,
                    chunk->chunkType,
                    chunk->chunkVersion,
                    chunk->offset = chunkOffset, // set chunk offset
                    chunk->size
                };
                chunkOffset += chunk->size;
            }

            offset += chunkTableSize;
        }

        // write chunks
        for (auto const & chunk : _chunks) 
        {
//            assert(offset == chunk->offset);
            memcpy(data+chunk->offset, chunk->data, chunk->size);
//            offset += chunk->size;
        }

        if(ppBlob) {
            *ppBlob = pBlob;
            pBlob->AddRef();
        }
        pBlob->Release();
        return nvrhi::FS_OK;
    }
    else
    {
        log::error("Chunkfile '%s' : blob allocation failed", _filepath.c_str());
    }

    return nvrhi::FE_GENERIC_ERROR;
}

};

