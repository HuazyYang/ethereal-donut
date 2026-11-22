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

// The GpuScene scatter kernel's constants and records (aggregate docs/ecs-design.md 7.2; plan step E5, sub-plan item
// E5.4). The kernel moves uint words only: no float operation touches a row, so no denormal is flushed and no NaN is
// canonicalised. The records sit in one staging buffer, read through a byte-address view, the instance buffer is
// written through a byte-address view (D3D11's instance buffer is raw, not structured). One thread per record.
//
// Within one dispatch a slot appears in at most one record (the settle list leaves out slots that get another
// record this frame), so the threads never race.

#ifndef GPU_SCENE_SCATTER_CB_H
#define GPU_SCENE_SCATTER_CB_H

#define GPU_SCENE_SCATTER_GROUP_SIZE 64

// InstanceData (bindless.h): 112 bytes; transform at word 4, prevTransform at word 16, 12 words each (float3x4).
#define GPU_SCENE_INSTANCE_ROW_WORDS 28
#define GPU_SCENE_ROW_TRANSFORM_WORD 4
#define GPU_SCENE_ROW_PREV_TRANSFORM_WORD 16
#define GPU_SCENE_TRANSFORM_WORDS 12

// GpuSceneRowRecord::flags: the row's prevTransform is the slot's current transform in the buffer (the same entity
// keeps the slot and moved), not the record's own prevTransform words.
#define GPU_SCENE_ROW_FLAG_KEEP_PREVIOUS 0x1

struct GpuSceneScatterConstants
{
    // Byte offsets into the staging buffer. Settle records are uint slots: prevTransform = transform.
    uint settleRecordOffset;
    uint settleRecordCount;
    // GpuSceneTransformRecord: prevTransform = transform, then transform = the record's.
    uint transformRecordOffset;
    uint transformRecordCount;

    // GpuSceneRowRecord: the whole row.
    uint rowRecordOffset;
    uint rowRecordCount;
    uint instanceRowStride;   // bytes, sizeof(InstanceData)
    uint padding;
};

// 64 bytes.
struct GpuSceneTransformRecord
{
    uint slot;
    uint padding0;
    uint padding1;
    uint padding2;
    uint transform[GPU_SCENE_TRANSFORM_WORDS];   // the bits of InstanceData::transform
};

// 128 bytes.
struct GpuSceneRowRecord
{
    uint slot;
    uint flags;
    uint padding0;
    uint padding1;
    uint row[GPU_SCENE_INSTANCE_ROW_WORDS];      // the bits of InstanceData
};

#endif // GPU_SCENE_SCATTER_CB_H
