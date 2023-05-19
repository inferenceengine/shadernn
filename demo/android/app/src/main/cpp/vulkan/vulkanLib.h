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

#include "volk.h"
#include <utility>

namespace snn {

uint32_t selectMemoryType(const VkPhysicalDeviceMemoryProperties& memory_properties, uint32_t supported_memory_types,
    VkMemoryPropertyFlags desired_memory_properties);

VkShaderModule loadShaderSpirvModule(VkDevice device, const char *spirvPath);

VkDeviceMemory allocateMemory(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory_properties, VkMemoryRequirements memory_requirements,
    VkMemoryPropertyFlags memory_flags);

std::pair<VkImage, VkDeviceMemory> createImage(VkDevice device, const VkPhysicalDeviceMemoryProperties& device_memory_properties,
                                               VkImageUsageFlags usage_flags, VkMemoryPropertyFlags memory_flags, VkImageType image_type,
                                                VkFormat image_format, VkExtent3D dimensions,
                                               VkImageTiling image_tiling, VkImageCreateFlags flags = 0);

VkImageView createImageView(VkDevice device, VkImage image, VkImageViewType view_type, VkFormat image_format);

VkSampler createSampler(VkDevice device);

VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool command_pool);

void transitionImageLayout(VkImage image, VkImageLayout from_layout, VkImageLayout to_layout, VkCommandBuffer cmdBuffer);

void copyBufferToImage(VkBuffer src_buffer, size_t src_offset, VkImage dst_image, VkExtent3D image_dimensions, VkCommandBuffer cmdBuffer);

void copyImageToBuffer(VkImage src_image, VkExtent3D image_dimensions, VkBuffer dst_buffer, size_t dst_offset, VkCommandBuffer cmdBuffer);

void copyImageToImage(VkImage src_image, VkImage dst_image, VkOffset3D src_offset, VkOffset3D dst_offset, VkExtent3D extent, VkCommandBuffer cmdBuffer);

void queueSubmitAndWait(VkDevice device, VkQueue queue, VkCommandBuffer cmdBuffer);

std::pair<VkBuffer, VkDeviceMemory> createBuffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& device_memory_properties,
    VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_flags, VkDeviceSize size_in_bytes);

void copyFromImageViaStagingBuffer(VkDevice device, VkBuffer staging_buffer, VkDeviceMemory staging_memory, VkImage image, VkExtent3D image_dimensions,
                                   VkImageLayout from_layout, void* pixels, size_t size_in_bytes, VkCommandBuffer cmdBuffer, VkQueue queue);
}
