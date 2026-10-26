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

/*
License for glfw

Copyright (c) 2002-2006 Marcus Geelnard

Copyright (c) 2006-2019 Camilla Lowy

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would
   be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not
   be misrepresented as being the original software.

3. This notice may not be removed or altered from any source
   distribution.
*/


#include <string>
#include <algorithm>

#include <donut/app/DeviceManager.h>
#include <donut/app/DeviceManager_DX12.h>
#include <donut/core/log.h>

#include <dxgidebug.h>

#if DONUT_WITH_STREAMLINE
#include <StreamlineIntegration.h>
#endif

using namespace donut::app;

namespace donut::app
{
    // DeviceManager.cpp
    bool GetDefaultD3D12AgilitySDK(uint32_t& outVersion, std::string& outPath);
}

static bool IsNvDeviceID(UINT id)
{
    return id == 0x10DE;
}

// Adjust window rect so that it is centred on the given adapter.  Clamps to fit if it's too big.
static bool MoveWindowOntoAdapter(IDXGIAdapter* targetAdapter, RECT& rect)
{
    if (!targetAdapter)
        return false;

    HRESULT hres = S_OK;
    unsigned int outputNo = 0;
    while (SUCCEEDED(hres))
    {
        nvrhi::AutoPtr<IDXGIOutput> pOutput;
        hres = targetAdapter->EnumOutputs(outputNo++, &pOutput);

        if (SUCCEEDED(hres) && pOutput)
        {
            DXGI_OUTPUT_DESC OutputDesc;
            pOutput->GetDesc(&OutputDesc);
            const RECT desktop = OutputDesc.DesktopCoordinates;
            const int centreX = (int)desktop.left + (int)(desktop.right - desktop.left) / 2;
            const int centreY = (int)desktop.top + (int)(desktop.bottom - desktop.top) / 2;
            const int winW = rect.right - rect.left;
            const int winH = rect.bottom - rect.top;
            const int left = centreX - winW / 2;
            const int right = left + winW;
            const int top = centreY - winH / 2;
            const int bottom = top + winH;
            rect.left = std::max(left, (int)desktop.left);
            rect.right = std::min(right, (int)desktop.right);
            rect.bottom = std::min(bottom, (int)desktop.bottom);
            rect.top = std::max(top, (int)desktop.top);

            // If there is more than one output, go with the first found.  Multi-monitor support could go here.
            return true;
        }
    }

    return false;
}

void DeviceManager_DX12::ReportLiveObjects()
{
    // dxgi.dll stays loaded once nvrhi's graphics factory loaded it, also after the factory is released.
    typedef HRESULT(WINAPI* PFN_DXGIGetDebugInterface1)(UINT, REFIID, void**);
    const HMODULE dxgi = GetModuleHandleW(L"dxgi.dll");
    const auto getDebugInterface = dxgi ? reinterpret_cast<PFN_DXGIGetDebugInterface1>(
        reinterpret_cast<void*>(GetProcAddress(dxgi, "DXGIGetDebugInterface1"))) : nullptr;
    if (!getDebugInterface)
        return;

    nvrhi::AutoPtr<IDXGIDebug> pDebug;
    getDebugInterface(0, IID_PPV_ARGS(&pDebug));

    if (pDebug)
    {
        DXGI_DEBUG_RLO_FLAGS flags = (DXGI_DEBUG_RLO_FLAGS)(DXGI_DEBUG_RLO_IGNORE_INTERNAL | DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_DETAIL);
        HRESULT hr = pDebug->ReportLiveObjects(DXGI_DEBUG_ALL, flags);
        if (FAILED(hr))
        {
            donut::log::error("ReportLiveObjects failed, HRESULT = 0x%08x", hr);
        }
    }
}

bool DeviceManager_DX12::CreateInstanceInternal()
{
#if DONUT_WITH_STREAMLINE
    StreamlineIntegration::Get().InitializePreDevice(nvrhi::GraphicsAPI::D3D12, m_DeviceParams.streamlineAppId, m_DeviceParams.checkStreamlineSignature, m_DeviceParams.enableStreamlineLog);
#endif

    if (m_Factory)
        return true;

    nvrhi::d3d12::GraphicsFactoryDesc factoryDesc;
    factoryDesc.messageCallback = DefaultMessageCallback::GetInstance();
    factoryDesc.enableDebugRuntime = m_DeviceParams.enableDebugRuntime;
    factoryDesc.enableGPUValidation = m_DeviceParams.enableGPUValidation;
    factoryDesc.enableWarningsAsErrors = m_DeviceParams.enableWarningsAsErrors;

    uint32_t sdkVersion = m_DeviceParams.d3d12AgilitySDKVersion;
    std::string sdkPath = m_DeviceParams.d3d12AgilitySDKPath;
    if (sdkVersion == 0)
        GetDefaultD3D12AgilitySDK(sdkVersion, sdkPath);
    factoryDesc.sdkVersion = sdkVersion;
    factoryDesc.sdkPath = sdkPath.c_str();

#if DONUT_WITH_STREAMLINE
    // Streamline hooks DXGI and D3D12 through its interposer.
    factoryDesc.dxgiLibraryName = "sl.interposer.dll";
    factoryDesc.d3d12LibraryName = "sl.interposer.dll";
#endif

    m_Factory = nvrhi::d3d12::createGraphicsFactory(factoryDesc);
    if (!m_Factory)
    {
        donut::log::error("Cannot create the D3D12 graphics factory.\n"
            "For more info, get log from debug D3D runtime: (1) Install DX SDK, and enable Debug D3D from DX Control Panel Utility. (2) Install and start DbgView. (3) Try running the program again.\n");
        return false;
    }
    m_GraphicsFactory = m_Factory.Get();

    if (sdkVersion != 0)
        donut::log::message(m_DeviceParams.infoLogSeverity, "D3D12 Agility SDK %u from %s", sdkVersion, sdkPath.c_str());

    return true;
}

bool DeviceManager_DX12::CreateDevice()
{
    int adapterIndex = m_DeviceParams.adapterIndex;

#if DONUT_WITH_STREAMLINE
    // Auto select best adapter for streamline features
    if (adapterIndex < 0)
        adapterIndex = StreamlineIntegration::Get().FindBestAdapterDX();
#endif

    if (adapterIndex < 0)
        adapterIndex = 0;

    nvrhi::AdapterInfo adapterInfo;
    if (NVRHI_FAILED(m_Factory->getAdapterInfo(uint32_t(adapterIndex), adapterInfo)))
    {
        if (adapterIndex == 0)
            donut::log::error("Cannot find any DXGI adapters in the system.");
        else
            donut::log::error("The specified DXGI adapter %d does not exist.", adapterIndex);

        return false;
    }

    m_AdapterIndex = uint32_t(adapterIndex);
    m_RendererString = adapterInfo.name.c_str();
    m_IsNvidia = IsNvDeviceID(adapterInfo.vendorID);

    nvrhi::d3d12::DeviceCreationDesc deviceDesc;
    deviceDesc.errorCB = m_DeviceParams.messageCallback ? m_DeviceParams.messageCallback : DefaultMessageCallback::GetInstance();
    deviceDesc.adapterIndex = adapterIndex;
    deviceDesc.featureLevel = m_DeviceParams.featureLevel;
    deviceDesc.enableComputeQueue = m_DeviceParams.enableComputeQueue;
    deviceDesc.enableCopyQueue = m_DeviceParams.enableCopyQueue;
#if DONUT_WITH_AFTERMATH
    deviceDesc.aftermathEnabled = m_DeviceParams.enableAftermath;
#endif
    deviceDesc.logBufferLifetime = m_DeviceParams.logBufferLifetime;
    deviceDesc.enableHeapDirectlyIndexed = m_DeviceParams.enableHeapDirectlyIndexed;
    deviceDesc.enableRayTracingValidation = m_DeviceParams.enableRayTracingValidation;

    nvrhi::d3d12::DeviceHandle d3d12Device;
    m_Factory->createDeviceOnAdapter(deviceDesc, &d3d12Device);
    if (!d3d12Device)
        return false;
    m_NvrhiDevice = d3d12Device.Get();

#if DONUT_WITH_STREAMLINE
    StreamlineIntegration::Get().SetD3DDevice(static_cast<ID3D12Device*>(d3d12Device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device)));
#endif

    if (m_DeviceParams.enableNvrhiValidationLayer)
    {
        nvrhi::DeviceHandle validationLayer;
        m_Factory->createValidationLayer(m_NvrhiDevice, &validationLayer);
        m_NvrhiDevice = validationLayer;
    }

#if DONUT_WITH_STREAMLINE
    StreamlineIntegration::Get().InitializeDeviceDX(m_NvrhiDevice);
#endif

    return true;
}

bool DeviceManager_DX12::CreateSwapChain()
{
    UINT windowStyle = m_DeviceParams.startFullscreen
        ? (WS_POPUP | WS_SYSMENU | WS_VISIBLE)
        : m_DeviceParams.startMaximized
            ? (WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_MAXIMIZE)
            : (WS_OVERLAPPEDWINDOW | WS_VISIBLE);

    RECT rect = { 0, 0, LONG(m_DeviceParams.backBufferWidth), LONG(m_DeviceParams.backBufferHeight) };
    AdjustWindowRect(&rect, windowStyle, FALSE);

    nvrhi::AdapterInfo adapterInfo;
    if (NVRHI_SUCCEEDED(m_Factory->getAdapterInfo(m_AdapterIndex, adapterInfo)) &&
        MoveWindowOntoAdapter(static_cast<IDXGIAdapter*>(adapterInfo.nativeAdapter), rect))
    {
        glfwSetWindowPos(m_Window, rect.left, rect.top);
    }

    nvrhi::d3d12::GraphicsSwapChainHandle swapChain;
    if (NVRHI_FAILED(m_Factory->createSwapChain(GetSwapChainDesc(), m_NvrhiDevice, &swapChain)))
        return false;

    m_SwapChain = swapChain.Get();
    SwapChainCreated();
    return true;
}

void DeviceManager_DX12::DestroyDeviceAndSwapChain()
{
    m_SwapChain = nullptr;
    m_NvrhiDevice = nullptr;
    m_RendererString.clear();
}

void DeviceManager_DX12::Shutdown()
{
    DeviceManager::Shutdown();

    m_GraphicsFactory = nullptr;
    m_Factory = nullptr;

    if (m_DeviceParams.enableDebugRuntime)
    {
        ReportLiveObjects();
    }
}

DeviceManager *DeviceManager::CreateD3D12(void)
{
    return MAKE_RC_OBJ(DeviceManager_DX12);
}
