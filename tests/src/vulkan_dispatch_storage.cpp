// The Vulkan-Hpp dynamic dispatcher lives in donut_app (DeviceManager_VK.cpp), but a test that never
// touches the device manager does not link that object. Where the linker still pulls an nvrhi_vk member
// (a GNU ld archive member defining a weak std::runtime_error typeinfo is enough), the dispatcher would
// be undefined, so the tests that do not use the device manager define it here.
#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <nvrhi/vulkan.h>
#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
