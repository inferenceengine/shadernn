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
#include "uvkc/benchmark/vulkan_context.h"
#include <memory>

namespace snn {

// This class contains main Vulkan objects
// It also serves as a bridge to uvkc library
class VulkanGpuContext : public GpuContext {
public:
    VulkanGpuContext(VkInstance instance_,
                     VkPhysicalDevice physicalDevice_,
                     VkPhysicalDeviceMemoryProperties deviceMemoryProperties_,
                     VkDevice device_,
                     VkCommandPool commandPool_,
                     VkQueue queue_,
                     uint32_t queueIndex_);

    VulkanGpuContext(std::unique_ptr<uvkc::benchmark::VulkanContext> uvkcContext_);

    static VulkanGpuContext* cast(GpuContext* context);

    uvkc::benchmark::VulkanContext* getUvkcContext() { return uvkcContext.get(); }

    VkInstance getInstance() const {
        return instance;
    }

    VkPhysicalDevice getPhysicalDevice() const {
        return physicalDevice;
    }

    const VkPhysicalDeviceMemoryProperties& getPhysicalDeviceMemoryProperties() const {
        return deviceMemoryProperties;
    }

    VkDevice getDevice() const {
        return device;
    }

    VkCommandPool getCommandPool() const {
        return commandPool;
    }

    VkQueue getQueue() const {
        return queue;
    }

    uint32_t getQueueIndex() const {
        return queueIndex;
    }

private:
    VkInstance instance;

    VkPhysicalDevice physicalDevice;

    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;

    VkDevice device;

    VkCommandPool commandPool;

    VkQueue queue;

    uint32_t queueIndex;

    std::unique_ptr<uvkc::benchmark::VulkanContext> uvkcContext;
};

}
