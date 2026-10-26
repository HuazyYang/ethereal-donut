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
#include <donut/app/DeviceManager_VK.h>

#if DONUT_WITH_STREAMLINE
#include <StreamlineIntegration.h>
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#include <vulkan/vulkan.hpp>
#endif

// The Vulkan-Hpp default dispatcher is NVRHI's (nvrhi ADR 0014): the Vulkan graphics factory initializes it.

using namespace donut;
using namespace donut::app;

namespace
{
    // DeviceCreationParameters::deviceCreateInfoCallback, as the callback interface of nvrhi's device creation.
    class DeviceCreateCallback : public nvrhi::ObjectImpl<nvrhi::vulkan::IDeviceCreateCallback>
    {
    public:
        NVRHI_BEGIN_INTERFACE_TABLE_INLINE(DeviceCreateCallback)
        NVRHI_IMPLEMENTS_INTERFACE(nvrhi::vulkan::IDeviceCreateCallback)
        NVRHI_END_INTERFACE_TABLE()

        explicit DeviceCreateCallback(const std::function<void(VkDeviceCreateInfo&)>& function)
            : m_Function(function)
        { }

        void onCreateDevice(VkDeviceCreateInfo& info) noexcept override
        {
            m_Function(info);
        }

    private:
        std::function<void(VkDeviceCreateInfo&)> m_Function;
    };

    std::vector<std::string> GetNames(uint32_t (*query)(void*, const char**, uint32_t), void* object)
    {
        std::vector<const char*> names(query(object, nullptr, 0));
        query(object, names.data(), uint32_t(names.size()));
        return std::vector<std::string>(names.begin(), names.end());
    }

    uint32_t QueryInstanceExtensions(void* factory, const char** names, uint32_t capacity)
    {
        return static_cast<nvrhi::vulkan::IGraphicsFactory*>(factory)->getEnabledInstanceExtensions(names, capacity);
    }

    uint32_t QueryLayers(void* factory, const char** names, uint32_t capacity)
    {
        return static_cast<nvrhi::vulkan::IGraphicsFactory*>(factory)->getEnabledLayers(names, capacity);
    }

    uint32_t QueryDeviceExtensions(void* device, const char** names, uint32_t capacity)
    {
        return static_cast<nvrhi::vulkan::IDevice*>(device)->getEnabledExtensions(names, capacity);
    }

    void AddStrings(nvrhi::vector<nvrhi::string>& destination, const std::vector<std::string>& source)
    {
        for (const std::string& s : source)
            destination.push_back(nvrhi::string(s.c_str()));
    }
}

bool DeviceManager_VK::CreateInstanceInternal()
{
#if DONUT_WITH_STREAMLINE
    StreamlineIntegration::Get().InitializePreDevice(nvrhi::GraphicsAPI::VULKAN, m_DeviceParams.streamlineAppId, m_DeviceParams.checkStreamlineSignature, m_DeviceParams.enableStreamlineLog);
#endif

    if (m_Factory)
        return true;

#if DONUT_WITH_STREAMLINE
    m_DeviceParams.vulkanLibraryName = "sl.interposer.dll";
#endif

    nvrhi::vulkan::GraphicsFactoryDesc factoryDesc;
    factoryDesc.messageCallback = DefaultMessageCallback::GetInstance();
    factoryDesc.enableDebugRuntime = m_DeviceParams.enableDebugRuntime;
    factoryDesc.enableRayTracingValidation = m_DeviceParams.enableRayTracingValidation;
    factoryDesc.headless = m_DeviceParams.headlessDevice;
    factoryDesc.vulkanLibraryName = m_DeviceParams.vulkanLibraryName.c_str();
    AddStrings(factoryDesc.requiredInstanceExtensions, m_DeviceParams.requiredVulkanInstanceExtensions);
    AddStrings(factoryDesc.optionalInstanceExtensions, m_DeviceParams.optionalVulkanInstanceExtensions);
    AddStrings(factoryDesc.requiredLayers, m_DeviceParams.requiredVulkanLayers);
    AddStrings(factoryDesc.optionalLayers, m_DeviceParams.optionalVulkanLayers);
    for (size_t location : m_DeviceParams.ignoredVulkanValidationMessageLocations)
        factoryDesc.ignoredValidationMessageLocations.push_back(uint64_t(location));

    m_Factory = nvrhi::vulkan::createGraphicsFactory(factoryDesc);
    if (!m_Factory)
    {
        log::error("Cannot create the Vulkan graphics factory.");
        return false;
    }
    m_GraphicsFactory = m_Factory.Get();

    return true;
}

bool DeviceManager_VK::CreateDevice()
{
    nvrhi::vulkan::DeviceCreationDesc deviceDesc;
    deviceDesc.errorCB = DefaultMessageCallback::GetInstance();
    deviceDesc.adapterIndex = m_DeviceParams.adapterIndex;

#if DONUT_WITH_STREAMLINE
    // Auto select best adapter for streamline features, among the discrete GPUs (the selection prefers them).
    if (deviceDesc.adapterIndex < 0)
    {
        std::vector<vk::PhysicalDevice> discreteGPUs;
        std::vector<int32_t> discreteIndices;
        for (uint32_t index = 0; index < m_Factory->getAdapterCount(); ++index)
        {
            nvrhi::AdapterInfo info;
            if (NVRHI_SUCCEEDED(m_Factory->getAdapterInfo(index, info)) && info.isDiscrete)
            {
                discreteGPUs.push_back(vk::PhysicalDevice(static_cast<VkPhysicalDevice>(info.nativeAdapter)));
                discreteIndices.push_back(int32_t(index));
            }
        }
        if (!discreteGPUs.empty())
            deviceDesc.adapterIndex = discreteIndices[StreamlineIntegration::Get().FindBestAdapterVulkan(discreteGPUs)];
    }
#endif

    AddStrings(deviceDesc.requiredExtensions, m_DeviceParams.requiredVulkanDeviceExtensions);
    AddStrings(deviceDesc.optionalExtensions, m_DeviceParams.optionalVulkanDeviceExtensions);
    deviceDesc.enableComputeQueue = m_DeviceParams.enableComputeQueue;
    deviceDesc.enableCopyQueue = m_DeviceParams.enableCopyQueue;
    deviceDesc.enableRayTracingExtensions = m_DeviceParams.enableRayTracingExtensions;
    deviceDesc.enableHeapDirectlyIndexed = m_DeviceParams.enableHeapDirectlyIndexed;
    deviceDesc.enableCbvDescriptorStreaming = m_DeviceParams.enableCbvDescriptorStreaming;
    deviceDesc.enableRayTracingValidation = m_DeviceParams.enableRayTracingValidation;
#if DONUT_WITH_AFTERMATH
    deviceDesc.aftermathEnabled = m_DeviceParams.enableAftermath;
#endif
    deviceDesc.logBufferLifetime = m_DeviceParams.logBufferLifetime;
    deviceDesc.multiview.enabled = m_DeviceParams.multiViewFeature.enabled;
    deviceDesc.multiview.maxMultiviewViewCount = m_DeviceParams.multiViewFeature.maxMultiviewViewCount;
    deviceDesc.multiview.maxMultiviewInstanceIndex = m_DeviceParams.multiViewFeature.maxMultiviewInstanceIndex;
    deviceDesc.physicalDeviceFeatures2Extensions = m_DeviceParams.physicalDeviceFeatures2Extensions;

    nvrhi::AutoPtr<DeviceCreateCallback> createCallback;
    if (m_DeviceParams.deviceCreateInfoCallback)
    {
        createCallback = MAKE_RC_OBJ_PTR(DeviceCreateCallback, m_DeviceParams.deviceCreateInfoCallback);
        deviceDesc.deviceCreateCallback = createCallback;
    }

    if (!m_DeviceParams.headlessDevice)
    {
        // The physical device must be able to present the swap chain to this window.
#ifdef _WIN32
        deviceDesc.surface.window = glfwGetWin32Window(m_Window);
#endif
        deviceDesc.surface.format = m_DeviceParams.swapChainFormat;
        deviceDesc.surface.bufferCount = m_DeviceParams.swapChainBufferCount;
        deviceDesc.surface.width = m_DeviceParams.backBufferWidth;
        deviceDesc.surface.height = m_DeviceParams.backBufferHeight;
    }

    m_NvrhiDevice = nullptr;
    m_Factory->createDeviceOnAdapter(deviceDesc, &m_NvrhiDevice);
    if (!m_NvrhiDevice)
        return false;

    const nvrhi::NativeObject physicalDevice = m_NvrhiDevice->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
    for (uint32_t index = 0; index < m_Factory->getAdapterCount(); ++index)
    {
        nvrhi::AdapterInfo info;
        if (NVRHI_SUCCEEDED(m_Factory->getAdapterInfo(index, info)) && info.nativeAdapter == physicalDevice)
            m_RendererString = info.name.c_str();
    }

    if (m_DeviceParams.enableNvrhiValidationLayer)
    {
        m_ValidationLayer = nullptr;
        m_Factory->createValidationLayer(m_NvrhiDevice, &m_ValidationLayer);
    }

#if DONUT_WITH_STREAMLINE
    // The queue families of the device: the factory takes the first graphics family and the first compute-only one.
    uint32_t graphicsQueueFamily = 0;
    uint32_t computeQueueFamily = 0;
    {
        const vk::PhysicalDevice vkPhysicalDevice(static_cast<VkPhysicalDevice>(physicalDevice));
        const auto families = vkPhysicalDevice.getQueueFamilyProperties();
        bool graphicsFound = false;
        bool computeFound = false;
        for (uint32_t i = 0; i < uint32_t(families.size()); i++)
        {
            if (!graphicsFound && (families[i].queueFlags & vk::QueueFlagBits::eGraphics))
            {
                graphicsQueueFamily = i;
                graphicsFound = true;
            }
            if (!computeFound && (families[i].queueFlags & vk::QueueFlagBits::eCompute) &&
                !(families[i].queueFlags & vk::QueueFlagBits::eGraphics))
            {
                computeQueueFamily = i;
                computeFound = true;
            }
        }
    }

    StreamlineIntegration::VulkanInfo vulkanInfo;
    vulkanInfo.vkDevice = m_NvrhiDevice->getNativeObject(nvrhi::ObjectTypes::VK_Device);
    vulkanInfo.vkInstance = m_NvrhiDevice->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
    vulkanInfo.vkPhysicalDevice = physicalDevice;
    vulkanInfo.computeQueueIndex = 0;
    vulkanInfo.computeQueueFamily = computeQueueFamily;
    vulkanInfo.graphicsQueueIndex = 0;
    vulkanInfo.graphicsQueueFamily = graphicsQueueFamily;

    StreamlineIntegration::Get().InitializeDeviceVK(m_NvrhiDevice, vulkanInfo);
#endif

    return true;
}

bool DeviceManager_VK::CreateSwapChain()
{
    nvrhi::vulkan::GraphicsSwapChainHandle swapChain;
    if (NVRHI_FAILED(m_Factory->createSwapChain(GetSwapChainDesc(), GetDevice(), &swapChain)))
        return false;

    m_SwapChain = swapChain.Get();
    SwapChainCreated();
    return true;
}

void DeviceManager_VK::DestroyDeviceAndSwapChain()
{
    m_SwapChain = nullptr;
    m_ValidationLayer = nullptr;
    m_NvrhiDevice = nullptr;
    m_RendererString.clear();
}

void DeviceManager_VK::Shutdown()
{
    DeviceManager::Shutdown();

    // The VkInstance goes with the factory (the device kept it alive while it existed).
    m_GraphicsFactory = nullptr;
    m_Factory = nullptr;
}

bool DeviceManager_VK::IsVulkanInstanceExtensionEnabled(const char* extensionName) const
{
    std::vector<std::string> names;
    GetEnabledVulkanInstanceExtensions(names);
    return std::find(names.begin(), names.end(), extensionName) != names.end();
}

bool DeviceManager_VK::IsVulkanDeviceExtensionEnabled(const char* extensionName) const
{
    std::vector<std::string> names;
    GetEnabledVulkanDeviceExtensions(names);
    return std::find(names.begin(), names.end(), extensionName) != names.end();
}

bool DeviceManager_VK::IsVulkanLayerEnabled(const char* layerName) const
{
    std::vector<std::string> names;
    GetEnabledVulkanLayers(names);
    return std::find(names.begin(), names.end(), layerName) != names.end();
}

void DeviceManager_VK::GetEnabledVulkanInstanceExtensions(std::vector<std::string>& extensions) const
{
    if (!m_Factory)
        return;
    for (const std::string& name : GetNames(QueryInstanceExtensions, m_Factory.Get()))
        extensions.push_back(name);
}

void DeviceManager_VK::GetEnabledVulkanDeviceExtensions(std::vector<std::string>& extensions) const
{
    if (!m_NvrhiDevice)
        return;
    for (const std::string& name : GetNames(QueryDeviceExtensions, m_NvrhiDevice.Get()))
        extensions.push_back(name);
}

void DeviceManager_VK::GetEnabledVulkanLayers(std::vector<std::string>& layers) const
{
    if (!m_Factory)
        return;
    for (const std::string& name : GetNames(QueryLayers, m_Factory.Get()))
        layers.push_back(name);
}

DeviceManager *DeviceManager::CreateVK()
{
    return MAKE_RC_OBJ(DeviceManager_VK);
}
