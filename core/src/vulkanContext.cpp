/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pch.h"
#include "vulkanContext.h"
#include "snn/utils.h"
#include "uvkcUtils.h"
#include "uvkc/benchmark/status_util.h"

namespace snn {

VulkanGpuContext::VulkanGpuContext(VkInstance instance_,
                                   VkPhysicalDevice physicalDevice_,
                                   VkPhysicalDeviceMemoryProperties deviceMemoryProperties_,
                                   VkDevice device_,
                                   VkCommandPool commandPool_,
                                   VkQueue queue_,
                                   uint32_t queueIndex_)
        : GpuContext(GpuBackendType::VULKAN)
        , instance(instance_)
        , physicalDevice(physicalDevice_)
        , deviceMemoryProperties(deviceMemoryProperties_)
        , device(device_)
        , commandPool(commandPool_)
        , queue(queue_)
        , queueIndex(queueIndex_)
{
    SNN_ASSERT(instance);
    SNN_ASSERT(physicalDevice);
    SNN_ASSERT(device);
    SNN_ASSERT(commandPool);
    SNN_ASSERT(queue);

    BM_CHECK_OK_AND_ASSIGN(uvkcContext,
        uvkc::benchmark::CreateVulkanContextExternally(
        instance,
        device,
        physicalDevice_,
        deviceMemoryProperties,
        queue,
        queueIndex,
        commandPool
    ));

    SNN_LOGI("Vulkan context created");

#if defined(__ANDROID__)
    uvkc::AndroidStreamLogger::SetLogger();
#endif
    uvkc::SetExternalLogger(snn::log, snn::isLoggable);
    uvkc::SetExternalTimerFactory(uvkc::timerAdapterFactory);
}

VulkanGpuContext::VulkanGpuContext(std::unique_ptr<uvkc::benchmark::VulkanContext> uvkcContext_)
        : GpuContext(GpuBackendType::VULKAN)
        , instance(VK_NULL_HANDLE)
        , physicalDevice(VK_NULL_HANDLE)
        , deviceMemoryProperties{}
        , device(VK_NULL_HANDLE)
        , commandPool(VK_NULL_HANDLE)
        , queue(VK_NULL_HANDLE)
        , queueIndex(0)
        , uvkcContext(std::move(uvkcContext_))
{
    instance = uvkcContext->driver->getInstance();
    physicalDevice = uvkcContext->devices[0].get()->getDevice();
    deviceMemoryProperties = uvkcContext->devices[0].get()->getPhysicalDeviceMemoryProperties();
    device = uvkcContext->devices[0].get()->getLogicalDevice();
    commandPool = uvkcContext->devices[0].get()->getCommandPool();
    queue = uvkcContext->devices[0].get()->getQueue();
    queueIndex = uvkcContext->devices[0].get()->getQueueFamilyIndex();
    // TODO: Query back the rest of native Vulkan parameters
    SNN_LOGI("Vulkan context created");

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    const VkPhysicalDeviceLimits &dev_limits = physicalDeviceProperties.limits;
    SNN_LOGI("Device maximum image dimensions: 1D = %d, 2D = %d, 3D = %d", dev_limits.maxImageDimension1D, dev_limits.maxImageDimension2D,
        dev_limits.maxImageDimension3D);

#if defined(__ANDROID__)
    uvkc::AndroidStreamLogger::SetLogger();
#endif
    uvkc::SetExternalLogger(snn::log, snn::isLoggable);
    uvkc::SetExternalTimerFactory(uvkc::timerAdapterFactory);
}

GpuContext* createVulkanContext(VkInstance instance_,
                                VkPhysicalDevice physicalDevice_,
                                VkPhysicalDeviceMemoryProperties deviceMemoryProperties_,
                                VkDevice device_,
                                VkCommandPool commandPool_,
                                VkQueue queue_,
                                uint32_t queueIndex_) {
    return new VulkanGpuContext(instance_,
                                physicalDevice_,
                                deviceMemoryProperties_,
                                device_,
                                commandPool_,
                                queue_,
                                queueIndex_);
}

GpuContext* createDefaultVulkanContext() {
    std::unique_ptr<uvkc::benchmark::VulkanContext> uvkcContext;
    BM_CHECK_OK_AND_ASSIGN(uvkcContext, uvkc::benchmark::CreateDefaultVulkanContext("Vulkan Context"));
    return new VulkanGpuContext(std::move(uvkcContext));
}

VulkanGpuContext* VulkanGpuContext::cast(GpuContext* context) {
    if (context->backendType != GpuBackendType::VULKAN) {
        SNN_RIP("Gpu context is not a Vulkan type!");
    }
    return static_cast<VulkanGpuContext*>(context);
}

}
