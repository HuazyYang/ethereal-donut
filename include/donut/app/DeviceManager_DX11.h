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

#include <string>

#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>

#include <nvrhi/d3d11.h>

// The D3D11 device manager: nvrhi's D3D11 graphics factory creates the device, its immediate context and the swap
// chain (nvrhi ADR 0014); this class adds the device parameters, the window placement and Streamline.
class DeviceManager_DX11 : public donut::app::DeviceManager
{
    NVRHI_INHERIT_INTERFACE_TABLE()
protected:
    nvrhi::d3d11::GraphicsFactoryHandle m_Factory;
    nvrhi::DeviceHandle m_NvrhiDevice;  // the validation layer around the D3D11 device, if enabled
    uint32_t m_AdapterIndex = 0;
    std::string m_RendererString;

public:
    [[nodiscard]] const char* GetRendererString() const override
    {
        return m_RendererString.c_str();
    }

    [[nodiscard]] nvrhi::IDevice* GetDevice() const override
    {
        return m_NvrhiDevice;
    }

    void ReportLiveObjects() override;

    [[nodiscard]] nvrhi::GraphicsAPI GetGraphicsAPI() const override
    {
        return nvrhi::GraphicsAPI::D3D11;
    }
protected:
    bool CreateInstanceInternal() override;
    bool CreateDevice() override;
    bool CreateSwapChain() override;
    void DestroyDeviceAndSwapChain() override;
    void Shutdown() override;
};
