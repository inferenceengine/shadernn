// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "uvkc/benchmark/vulkan_context.h"

#include "uvkc/base/status.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/base/log.h"

namespace uvkc {
namespace benchmark {

VulkanContext::VulkanContext(
    std::unique_ptr<vulkan::DynamicSymbols> symbols,
    std::unique_ptr<vulkan::Driver> driver,
    std::vector<vulkan::Driver::PhysicalDeviceInfo> physical_devices,
    std::vector<std::unique_ptr<vulkan::Device>> devices)
    : symbols(std::move(symbols)),
      driver(std::move(driver)),
      physical_devices(std::move(physical_devices)),
      devices(std::move(devices)),
      latency_measure({LatencyMeasureMode::kSystemSubmit, 0.}) {}

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateDefaultVulkanContext(
    const char *app_name) {
  UVKC_ASSIGN_OR_RETURN(auto symbols,
                        vulkan::DynamicSymbols::CreateFromSystemLoader());
  UVKC_ASSIGN_OR_RETURN(auto driver,
                        vulkan::Driver::Create(app_name, symbols.get()));
  UVKC_ASSIGN_OR_RETURN(auto physical_devices,
                        driver->EnumeratePhysicalDevices());

  std::vector<std::unique_ptr<vulkan::Device>> devices;
  for (const auto &physical_device : physical_devices) {
    UVKC_ASSIGN_OR_RETURN(
        auto device,
        driver->CreateDevice(physical_device, VK_QUEUE_COMPUTE_BIT));
    devices.push_back(std::move(device));
  }
  return std::make_unique<VulkanContext>(std::move(symbols), std::move(driver),
                                         std::move(physical_devices),
                                         std::move(devices));
}

VulkanContext::~VulkanContext() {
  UVKC_LOGV("destroyed VulkanContext");
}

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContextExternally(
    VkInstance instance,
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceMemoryProperties physical_device_memory_properties,
    VkQueue queue,
    uint32_t queue_family_index,
    VkCommandPool command_pool) {
  UVKC_ASSIGN_OR_RETURN(auto symbols,
                        vulkan::DynamicSymbols::CreateFromSystemLoader());
  UVKC_ASSIGN_OR_RETURN(auto driver,
                        vulkan::Driver::CreateExternally(symbols.get(), instance));
  
  VkPhysicalDeviceSubgroupProperties subgroup_properties = {}; // VkPhysicalDeviceSubgroupProperties never used
  subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  
  VkPhysicalDeviceProperties2 properties2 = {};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &subgroup_properties;
  symbols->vkGetPhysicalDeviceProperties2(physical_device, &properties2);
  std::vector<vulkan::Driver::PhysicalDeviceInfo> physical_devices {vulkan::Driver::PhysicalDeviceInfo{physical_device, properties2.properties, subgroup_properties}};

  uint32_t queue_family_count;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
  if (queue_family_count <= queue_family_index) {
    throw std::runtime_error("Queue family size is too small for queue family index provided.");
  }
  std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_family_properties.data());
  uint32_t timestamp_valid_bits = queue_family_properties[queue_family_index].timestampValidBits;

  UVKC_ASSIGN_OR_RETURN(auto uvkc_device,
      vulkan::Device::CreateExternally(physical_device, physical_device_memory_properties, queue, queue_family_index,
        timestamp_valid_bits, properties2.properties.limits.timestampPeriod,
        device, command_pool, *symbols.get()));
  std::vector<std::unique_ptr<vulkan::Device>> uvkc_devices;
  uvkc_devices.push_back(std::move(uvkc_device));

  return std::make_unique<VulkanContext>(std::move(symbols), std::move(driver),
                                         std::move(physical_devices),
                                         std::move(uvkc_devices));
}

}  // namespace benchmark
}  // namespace uvkc
