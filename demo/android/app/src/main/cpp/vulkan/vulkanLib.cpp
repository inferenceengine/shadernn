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

#include "vulkanLib.h"
#include "error.h"
#include "snn/utils.h"
#include <string.h>

namespace snn {

/**
 * @brief Helper function to load a shader module.
 * @param device Vulkan logical device
 * @param path The path for the shader (relative to the assets directory).
 * @returns A VkShaderModule handle. Aborts execution if shader creation fails.
 */
VkShaderModule loadShaderSpirvModule(VkDevice device, const char *spirvPath)
{
    std::vector<unsigned char> bytes = snn::loadEmbeddedAsset(spirvPath);
            SNN_ASSERT(bytes.size() > 0);
    const uint32_t *spirvData = reinterpret_cast<const uint32_t*>(bytes.data());
    size_t spirvSize = bytes.size() / sizeof(uint32_t);

    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = spirvSize * sizeof(uint32_t);
    module_info.pCode    = spirvData;

    VkShaderModule shader_module;
    VK_CHECK(vkCreateShaderModule(device, &module_info, nullptr, &shader_module));

    return shader_module;
}

uint32_t selectMemoryType(const VkPhysicalDeviceMemoryProperties& device_memory_properties, uint32_t supported_memory_types,
    VkMemoryPropertyFlags desired_memory_properties)
{
    for (int i = 0; i < device_memory_properties.memoryTypeCount; ++i) {
        if ((supported_memory_types & (1 << i)) &&
            ((device_memory_properties.memoryTypes[i].propertyFlags &
                desired_memory_properties) == desired_memory_properties))
            return i;
    }
    SNN_RIP("Cannot find supported memory type!");
}

VkDeviceMemory allocateMemory(VkDevice device, const VkPhysicalDeviceMemoryProperties& device_memory_properties,
    VkMemoryRequirements memory_requirements, VkMemoryPropertyFlags memory_flags)
{
    VkMemoryAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.pNext = nullptr;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = selectMemoryType(device_memory_properties, memory_requirements.memoryTypeBits, memory_flags);

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(device, &allocate_info, /*pAlloator=*/nullptr, &memory));
    return memory;
}

std::pair<VkImage, VkDeviceMemory>
    createImage(VkDevice device, const VkPhysicalDeviceMemoryProperties& device_memory_properties,
                VkImageUsageFlags usage_flags, VkMemoryPropertyFlags memory_flags, VkImageType image_type, VkFormat image_format, VkExtent3D dimensions,
                VkImageTiling image_tiling, VkImageCreateFlags flags)
{
    VkImageCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    create_info.pNext = nullptr;
    create_info.flags = flags;
    create_info.imageType = image_type;
    create_info.format = image_format;
    create_info.extent = dimensions;
    create_info.mipLevels = 1;
    create_info.arrayLayers = 1;
    create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    create_info.tiling = image_tiling;
    create_info.usage = usage_flags;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
    create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImage(device, &create_info,
            /*allocator=*/nullptr, &image));

    // Get memory requirements for the image
    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device, image, &memory_requirements);

    // Allocate memory for the image
    VkDeviceMemory memory = allocateMemory(device, device_memory_properties, memory_requirements, memory_flags);

    // Bind the memory to the image
    VK_CHECK(vkBindImageMemory(device, image, memory, /*memoryOffset=*/0));

    return std::pair(image, memory);
}

VkImageView createImageView(VkDevice device, VkImage image, VkImageViewType view_type, VkFormat image_format)
{
    // Create image view for the image
    VkImageViewCreateInfo view_create_info = {};
    view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_create_info.pNext = nullptr;
    view_create_info.flags = 0;
    view_create_info.image = image;
    view_create_info.viewType = view_type;
    view_create_info.format = image_format;
    view_create_info.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_create_info.subresourceRange.baseMipLevel = 0;
    view_create_info.subresourceRange.levelCount = 1;
    view_create_info.subresourceRange.baseArrayLayer = 0;
    view_create_info.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device, &view_create_info,
            /*allocator=*/nullptr, &view));

    return view;
}

VkSampler createSampler(VkDevice device)
{
    VkSamplerCreateInfo create_sampler_info = {};
    create_sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    create_sampler_info.pNext = nullptr;
    create_sampler_info.flags = 0;
    create_sampler_info.magFilter = VK_FILTER_LINEAR;
    create_sampler_info.minFilter = VK_FILTER_LINEAR;
    create_sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    create_sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    create_sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    create_sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    create_sampler_info.mipLodBias = 0.0f;
    create_sampler_info.anisotropyEnable = VK_FALSE;
    create_sampler_info.maxAnisotropy = 0.0f;
    create_sampler_info.compareEnable = VK_FALSE;
    create_sampler_info.compareOp = VK_COMPARE_OP_NEVER;
    create_sampler_info.minLod = 0.0f;
    create_sampler_info.maxLod = 0.0f;
    create_sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    create_sampler_info.unnormalizedCoordinates = VK_FALSE;

    VkSampler sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(device, &create_sampler_info, nullptr, &sampler));
    return sampler;
}

std::pair<VkBuffer, VkDeviceMemory> createBuffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& device_memory_properties,
    VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_flags, VkDeviceSize size_in_bytes) {
    VkBufferCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.pNext = nullptr;
    create_info.flags = 0;
    create_info.size = size_in_bytes;
    create_info.usage = usage_flags;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(device, &create_info,
    /*pAllocator=*/nullptr, &buffer));

    // Get memory requirements for the buffer
    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

    // Allocate memory for the buffer
    VkDeviceMemory memory = allocateMemory(device, device_memory_properties, memory_requirements, memory_flags);

    // Bind the memory to the buffer
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, /*memoryOffset=*/0));

    return std::pair<VkBuffer, VkDeviceMemory>(buffer, memory);
}

VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool command_pool)
{
    VkCommandBufferAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.pNext = nullptr;
    allocate_info.commandPool = command_pool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer));
    return command_buffer;
}

void transitionImageLayout(VkImage image, VkImageLayout from_layout, VkImageLayout to_layout, VkCommandBuffer cmdBuffer)
{
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from_layout;
    barrier.newLayout = to_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags src_stage;
    VkPipelineStageFlags dst_stage;
    if (from_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
                to_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (from_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                to_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (from_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
            to_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (from_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
            to_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (from_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
            to_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (from_layout == VK_IMAGE_LAYOUT_GENERAL &&
            to_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        SNN_RIP("Image layout transition not implemented!");
    }
    vkCmdPipelineBarrier(cmdBuffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void copyBufferToImage(VkBuffer src_buffer, size_t src_offset, VkImage dst_image, VkExtent3D image_dimensions, VkCommandBuffer cmdBuffer)
{
    VkBufferImageCopy region = {};
    region.bufferOffset = src_offset;
    // Indicate the buffer is tightly packed
    region.bufferRowLength = region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = image_dimensions;

    vkCmdCopyBufferToImage(cmdBuffer, src_buffer,
                           dst_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            /*regionCount=*/1, &region);
}

void copyImageToBuffer(VkImage src_image, VkExtent3D image_dimensions, VkBuffer dst_buffer, size_t dst_offset, VkCommandBuffer cmdBuffer)
{
    VkBufferImageCopy region = {};
    region.bufferOffset = dst_offset;
    // Indicate the buffer is tightly packed
    region.bufferRowLength = region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = image_dimensions;

    vkCmdCopyImageToBuffer(cmdBuffer, src_image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst_buffer,
                           /*regionCount=*/1, &region);
}

void copyImageToImage(VkImage src_image, VkImage dst_image, VkOffset3D src_offset, VkOffset3D dst_offset, VkExtent3D extent, VkCommandBuffer cmdBuffer)
{
    VkImageSubresourceLayers srcSubresource{};
    srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    srcSubresource.mipLevel = 0;
    srcSubresource.baseArrayLayer = 0;
    srcSubresource.layerCount = 1;

    VkImageSubresourceLayers dstSubresource{};
    dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    dstSubresource.mipLevel = 0;
    dstSubresource.baseArrayLayer = 0;
    dstSubresource.layerCount = 1;

    VkImageCopy regions;
    regions.srcSubresource = srcSubresource;
    regions.srcOffset = src_offset;
    regions.dstSubresource = dstSubresource;
    regions.dstOffset = dst_offset;
    regions.extent = extent;

    vkCmdCopyImage(cmdBuffer, src_image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst_image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &regions);
}

void queueSubmitAndWait(VkDevice device, VkQueue queue, VkCommandBuffer cmdBuffer) {
    VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};

    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, /*pALlocator=*/nullptr, &fence));

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));

    VK_CHECK(vkWaitForFences(device, /*fenceCount=*/1, &fence,
                /*waitAll=*/true,
                /*timeout=*/UINT64_MAX));

    vkDestroyFence(device, fence, /*pAllocator=*/nullptr);
}

void copyFromImageViaStagingBuffer(VkDevice device, VkBuffer staging_buffer, VkDeviceMemory staging_memory, VkImage image, VkExtent3D image_dimensions,
                                   VkImageLayout from_layout, void* pixels, size_t size_in_bytes, VkCommandBuffer cmdBuffer, VkQueue queue)
{
    // Copy data from the device.
    transitionImageLayout(image, from_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cmdBuffer);
    copyImageToBuffer(image, image_dimensions, staging_buffer, 0, cmdBuffer);
    VK_CHECK(vkEndCommandBuffer(cmdBuffer));

    queueSubmitAndWait(device, queue, cmdBuffer);
    vkResetCommandBuffer(cmdBuffer, /*flags=*/0);

    void *dst_staging_ptr = nullptr;
    VK_CHECK(vkMapMemory(device, staging_memory, 0, size_in_bytes, /*flags=*/0, &dst_staging_ptr));
    memcpy(pixels, dst_staging_ptr, size_in_bytes);
    vkUnmapMemory(device, staging_memory);
}

}
