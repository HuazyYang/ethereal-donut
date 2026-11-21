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

// World snapshots and the world hash (aggregate docs/ecs-design.md 11.1, 11.4; plan step E1.5): the 64-bit hash both
// use, and the constants of the snapshot format. Module-private, like EcsWorldImpl.h. The format and the hash's
// canonical byte stream are described in EcsSnapshot.cpp.

#pragma once

#include <donut/engine/EcsTypes.h>

#include <cstddef>
#include <cstdint>

namespace donut::engine::ecs
{
    // XXH64 (xxHash, 64-bit variant, version 0.8 output), computed incrementally: Update over any split of a byte stream
    // gives the digest of the whole stream. Implemented here so that hashes do not depend on a library version.
    class Hasher64
    {
    public:
        explicit Hasher64(uint64_t seed = 0) noexcept;
        void Update(const void* data, size_t size) noexcept;
        uint64_t Digest() const noexcept;

        // Little-endian integers.
        void U8(uint8_t value) noexcept { Update(&value, 1); }
        void U32(uint32_t value) noexcept;
        void U64(uint64_t value) noexcept;

    private:
        uint64_t m_Lanes[4];
        uint64_t m_Seed;
        uint64_t m_Total = 0;
        uint8_t m_Buffer[32];
        uint32_t m_Buffered = 0;
    };

    uint64_t Xxh64(const void* data, size_t size, uint64_t seed = 0) noexcept;

    // The domain tag at the start of the hashed stream of IWorld::ComputeHash; it changes when the stream does.
    constexpr char c_WorldHashTag[16] = { 'd', 'o', 'n', 'u', 't', '-', 'e', 'c', 's', '-', 'h', 'a', 's', 'h', '/', '1' };

    // Snapshot: a header, the payload, and the XXH64 of both. All integers little-endian.
    constexpr char c_SnapshotMagic[8] = { 'D', 'O', 'N', 'U', 'T', 'E', 'C', 'S' };
    constexpr uint32_t c_SnapshotVersion = 1;
    // magic, u32 version, u32 flags (0), u64 payload bytes
    constexpr uint32_t c_SnapshotHeaderBytes = 24;
    constexpr uint32_t c_SnapshotChecksumBytes = 8;
    constexpr uint32_t c_SnapshotVersionOffset = 8;
    constexpr uint32_t c_SnapshotPayloadBytesOffset = 16;

    // The state of an entity index, in the snapshot's entity table and in the hashed stream.
    enum class SnapshotEntityState : uint8_t { Free = 0, Reserved = 1, Alive = 2 };

    // Snapshots hold, and hashes cover, the components that are Relocatable and not NoSerialize (EcsWorld.h).
    constexpr bool IsSerialisable(ComponentFlags flags) noexcept
    {
        return !!(flags & ComponentFlags::Relocatable) && !(flags & ComponentFlags::NoSerialize);
    }
}
