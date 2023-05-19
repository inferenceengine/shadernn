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
#include "vulkanAppContext.h"
#include "snn/utils.h"

namespace snn {

VulkanAppContext::VulkanAppContext(VkInstance vkInstance,
                                   VkPhysicalDevice physicalDevice,
                                   VkPhysicalDeviceMemoryProperties deviceMemoryProperties,
                                   VkDevice device,
                                   VkCommandPool commandPool,
                                   VkQueue queue,
                                   uint32_t queueIndex,
                                   std::function<void(VkImageView, VkSampler)> outputSamplerUpdater_)
    : AppContext(new VulkanGpuContext(vkInstance, physicalDevice, deviceMemoryProperties, device, commandPool, queue, queueIndex))
    , outputSamplerUpdater(outputSamplerUpdater_)
{}

void VulkanAppContext::createContext(VkInstance vkInstance,
                                     VkPhysicalDevice physicalDevice,
                                     VkPhysicalDeviceMemoryProperties deviceMemoryProperties,
                                     VkDevice device,
                                     VkCommandPool commandPool,
                                     VkQueue queue,
                                     uint32_t queueIndex,
                                     std::function<void(VkImageView, VkSampler)> updateOutputSampler)
{
    if (instance) {
        SNN_RIP("Context is already created !");
    }
    instance = new VulkanAppContext(vkInstance, physicalDevice, deviceMemoryProperties, device, commandPool, queue, queueIndex, updateOutputSampler);
}

}
