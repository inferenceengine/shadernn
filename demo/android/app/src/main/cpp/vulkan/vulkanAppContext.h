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
#pragma once
#include "snn/snn.h"
#include "volk.h"
#include "appContext.h"
#include "vulkanContext.h"
#include <functional>

namespace snn {

class VulkanAppContext : public AppContext {
public:
    static void createContext(VkInstance vkInstance,
                              VkPhysicalDevice physicalDevice,
                              VkPhysicalDeviceMemoryProperties deviceMemoryProperties,
                              VkDevice device,
                              VkCommandPool commandPool,
                              VkQueue queue,
                              uint32_t queueIndex,
                              std::function<void(VkImageView, VkSampler)> updateOutputSampler);

    static void destroyContext() {
        delete instance;
        instance = nullptr;
    }

    static VulkanGpuContext* getVulkanContext() {
        return VulkanGpuContext::cast(AppContext::getContext());
    }

    static void updateOutputSampler(VkImageView view, VkSampler sampler) {
        static_cast<VulkanAppContext*>(instance)->outputSamplerUpdater(view, sampler);
    }

private:
    VulkanAppContext(VkInstance vkInstance,
                     VkPhysicalDevice physicalDevice,
                     VkPhysicalDeviceMemoryProperties deviceMemoryProperties,
                     VkDevice device,
                     VkCommandPool commandPool,
                     VkQueue queue,
                     uint32_t queueIndex,
                     std::function<void(VkImageView, VkSampler)> outputSamplerUpdater_);

    const std::function<void(VkImageView, VkSampler)> outputSamplerUpdater;
};

}
