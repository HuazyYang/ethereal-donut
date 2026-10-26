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
#pragma once

#include <string>
#include <vector>

#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>

#include <nvrhi/vulkan.h>

// The Vulkan device manager: nvrhi's Vulkan graphics factory creates the instance, selects the physical device, and
// creates the device, its queues and the swap chain (nvrhi ADR 0014); this class adds the device parameters and
// Streamline.
class DeviceManager_VK : public donut::app::DeviceManager
{
    NVRHI_INHERIT_INTERFACE_TABLE()
public:
    [[nodiscard]] nvrhi::IDevice* GetDevice() const override
    {
        if (m_ValidationLayer)
            return m_ValidationLayer;

        return m_NvrhiDevice;
    }

    [[nodiscard]] nvrhi::GraphicsAPI GetGraphicsAPI() const override
    {
        return nvrhi::GraphicsAPI::VULKAN;
    }

    const donut::app::DeviceCreationParameters& GetDeviceParams() const { return m_DeviceParams; };

protected:
    bool CreateInstanceInternal() override;
    bool CreateDevice() override;
    bool CreateSwapChain() override;
    void DestroyDeviceAndSwapChain() override;
    void Shutdown() override;

    const char *GetRendererString() const override
    {
        return m_RendererString.c_str();
    }

    bool IsVulkanInstanceExtensionEnabled(const char* extensionName) const override;
    bool IsVulkanDeviceExtensionEnabled(const char* extensionName) const override;
    bool IsVulkanLayerEnabled(const char* layerName) const override;
    void GetEnabledVulkanInstanceExtensions(std::vector<std::string>& extensions) const override;
    void GetEnabledVulkanDeviceExtensions(std::vector<std::string>& extensions) const override;
    void GetEnabledVulkanLayers(std::vector<std::string>& layers) const override;

    nvrhi::vulkan::GraphicsFactoryHandle m_Factory;
    nvrhi::vulkan::DeviceHandle m_NvrhiDevice;
    nvrhi::DeviceHandle m_ValidationLayer;
    std::string m_RendererString;
};
