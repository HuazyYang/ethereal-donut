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

// The GpuScene scatter (aggregate docs/ecs-design.md 7.2; plan step E5, sub-plan item E5.4). One thread per record:
// settle records, then transform records, then row records, at the offsets and counts of the constants. Rows move as
// uint words only, so no float operation can flush a denormal or canonicalise a NaN. A slot appears in at most one
// record of a dispatch, so threads never touch the same row.

#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/gpu_scene_scatter_cb.h>

ByteAddressBuffer t_Records : register(t0);
RWByteAddressBuffer u_Instances : register(u0);

DECLARE_PUSH_CONSTANTS(GpuSceneScatterConstants, g_Const, 0, 0);

// Copies the 12 transform words at `from` (bytes, in the instance buffer) to `to`.
void CopyTransform(uint from, uint to)
{
    const uint4 a = u_Instances.Load4(from);
    const uint4 b = u_Instances.Load4(from + 16);
    const uint4 c = u_Instances.Load4(from + 32);
    u_Instances.Store4(to, a);
    u_Instances.Store4(to + 16, b);
    u_Instances.Store4(to + 32, c);
}

[numthreads(GPU_SCENE_SCATTER_GROUP_SIZE, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint index = (groupId.y * g_Const.groupsX + groupId.x) * GPU_SCENE_SCATTER_GROUP_SIZE + groupIndex;
    const uint transformOffset = GPU_SCENE_ROW_TRANSFORM_WORD * 4;
    const uint prevOffset = GPU_SCENE_ROW_PREV_TRANSFORM_WORD * 4;

    // prevTransform = transform
    if (index < g_Const.settleRecordCount)
    {
        const uint slot = t_Records.Load(g_Const.settleRecordOffset + index * 4);
        const uint row = slot * g_Const.instanceRowStride;
        CopyTransform(row + transformOffset, row + prevOffset);
        return;
    }
    index -= g_Const.settleRecordCount;

    // prevTransform = transform, then transform = the record's
    if (index < g_Const.transformRecordCount)
    {
        const uint record = g_Const.transformRecordOffset + index * 64;
        const uint slot = t_Records.Load(record);
        const uint row = slot * g_Const.instanceRowStride;
        CopyTransform(row + transformOffset, row + prevOffset);
        u_Instances.Store4(row + transformOffset, t_Records.Load4(record + 16));
        u_Instances.Store4(row + transformOffset + 16, t_Records.Load4(record + 32));
        u_Instances.Store4(row + transformOffset + 32, t_Records.Load4(record + 48));
        return;
    }
    index -= g_Const.transformRecordCount;

    // the whole row; with GPU_SCENE_ROW_FLAG_KEEP_PREVIOUS, prevTransform is the row's transform before this write
    if (index < g_Const.rowRecordCount)
    {
        const uint record = g_Const.rowRecordOffset + index * 128;
        const uint slot = t_Records.Load(record);
        const uint flags = t_Records.Load(record + 4);
        const uint row = slot * g_Const.instanceRowStride;
        if (flags & GPU_SCENE_ROW_FLAG_KEEP_PREVIOUS)
            CopyTransform(row + transformOffset, row + prevOffset);
        else
        {
            u_Instances.Store4(row + prevOffset, t_Records.Load4(record + 16 + prevOffset));
            u_Instances.Store4(row + prevOffset + 16, t_Records.Load4(record + 16 + prevOffset + 16));
            u_Instances.Store4(row + prevOffset + 32, t_Records.Load4(record + 16 + prevOffset + 32));
        }
        // the flags, geometry indices and the transform: words 0..15
        u_Instances.Store4(row, t_Records.Load4(record + 16));
        u_Instances.Store4(row + 16, t_Records.Load4(record + 32));
        u_Instances.Store4(row + 32, t_Records.Load4(record + 48));
        u_Instances.Store4(row + 48, t_Records.Load4(record + 64));
    }
}
