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

#pragma once

#include <donut/app/ImGuiLayer.h>
#include <donut/app/RenderGraphLayer.h>

namespace donut::app
{
    // The render graph's diagnostics in an ImGui window (design 11.3): the last plan's statistics and the run
    // statistics, per-pass GPU timers, the passes with their barriers and the reasons for them, resource lifetimes
    // with heap placement, the debug switches, and pass toggles that drive SetPassEnabled (13.5).
    class RenderGraphVisualizer : public ImGuiLayer
    {
        NVRHI_INHERIT_INTERFACE_TABLE()
    public:
        RenderGraphVisualizer(DeviceManager* deviceManager, RenderGraphLayer* graphLayer);

        // Also draws the window into another ImGui layer's BuildUI, for apps with one UI layer.
        void DrawWindow();

        bool open = true;

    protected:
        void BuildUI() override;

    private:
        nvrhi::AutoPtr<RenderGraphLayer> m_GraphLayer;
    };
}
