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
#include "volk.h"
#include "vulkanGpuFrameImage.h"
#include "vulkan/vulkanAppContext.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "snn/vulkanImageHandle.h"
#include "snn/utils.h"
#include "colorVulkan.h"
#include "vulkan/vulkanLib.h"
#include "vulkan/error.h"
#include <utility>

namespace snn {

VulkanGpuFrameImage::VulkanGpuFrameImage(const Desc& d)
    : GpuFrameImage(GpuBackendType::VULKAN)
    , context(VulkanAppContext::getVulkanContext())
{
    SNN_ASSERT(context);
    SNN_ASSERT(context->getDevice());
    SNN_ASSERT(context->getCommandPool());

    _desc = d;
    const ColorFormatDesc& cfd = getColorFormatDesc(_desc.format);
    sizeInBytes = _desc.width * _desc.height * _desc.depth * cfd.bytes();

    imageFormat = getNativeColorVulkan(_desc.format);

    std::tie(image, imageMemory) = createImage(context->getDevice(),
        context->getPhysicalDeviceMemoryProperties(),
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_TYPE_3D,
        imageFormat,
        {d.width, d.height, d.depth},
        VK_IMAGE_TILING_OPTIMAL);

    imageView = createImageView(context->getDevice(), image, VK_IMAGE_VIEW_TYPE_3D, imageFormat);

    allocateCommandBuffer();
    initStagingBuffer();

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(context->getDevice(), &fenceCreateInfo, /*pALlocator=*/nullptr, &fence));
}

VulkanGpuFrameImage::~VulkanGpuFrameImage() {
    if (imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->getDevice(), imageView, nullptr);
    }
    if (image != VK_NULL_HANDLE) {
        vkDestroyImage(context->getDevice(), image, nullptr);
    }
    if (imageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(context->getDevice(), imageMemory, nullptr);
    }
    if (commandBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(context->getDevice(), context->getCommandPool(), 1, &commandBuffer);
    }
    if (stagingBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context->getDevice(), stagingBuffer, nullptr);
    }
    if (stagingMemory != VK_NULL_HANDLE) {
        vkFreeMemory(context->getDevice(), stagingMemory, nullptr);
    }
    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(context->getDevice(), fence, nullptr);
    }
}

void VulkanGpuFrameImage::updateTextureContent(ColorFormat format, void* data, uint32_t size) {
    if (size == 0) {
        size = sizeInBytes;
    }
    if (format != _desc.format) {
        SNN_LOGE("incompatible format.");
        return;
    }

    SNN_ASSERT(stagingMemory);
    SNN_ASSERT(stagingBuffer);

    vkWaitForFences(context->getDevice(), 1, &fence, true, UINT64_MAX);
    vkResetFences(context->getDevice(), 1, &fence);
    vkResetCommandBuffer(commandBuffer, /*flags=*/0);

// Uncomment to test. The result must be a green screen.
#if 0
    std::vector<std::array<uint8_t, 4>> greenPixels(_desc.width * _desc.height * _desc.depth, {0, 254, 0, 254});
    data = greenPixels.data();
#endif
    void *srcStagingPtr = nullptr;
    VK_CHECK(vkMapMemory(context->getDevice(), stagingMemory, 0, sizeInBytes, /*flags=*/0, &srcStagingPtr));
    memcpy(srcStagingPtr, data, sizeInBytes);
    vkUnmapMemory(context->getDevice(), stagingMemory);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    // Begin command recording
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, commandBuffer);
    copyBufferToImage(stagingBuffer, 0, image, {_desc.width, _desc.height, _desc.depth}, commandBuffer);
    transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, commandBuffer);

    // Complete the command buffer.
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    // Submit command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VK_CHECK(vkQueueSubmit(context->getQueue(), 1, &submitInfo, fence));

// Uncomment to test the image contents.
#if 0
    std::vector<uint8_t> pixelsRes(sizeInBytes, 0U);
    SNN_ASSERT(context->queue);
    copyFromImageViaStagingBuffer(context->device, context->commandPool, stagingBuffer, stagingMemory, image,
        {_desc.width, _desc.height, _desc.depth},
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, pixelsRes.data(), pixelsRes.size(),
        commandBuffer, context->queue);
    SNN_LOGI("%d %d %d %d", pixelsRes[0], pixelsRes[1], pixelsRes[2], pixelsRes[3]);
#endif
}

void VulkanGpuFrameImage::getGpuImageHandle(GpuImageHandle& handle) const {
    VulkanImageHandle& vkHandle = VulkanImageHandle::cast(handle);
    vkHandle.image = image;
    vkHandle.imageView = imageView;
}


void VulkanGpuFrameImage::allocateCommandBuffer() {
    VkCommandBufferAllocateInfo cmdBufInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdBufInfo.commandPool        = context->getCommandPool();
    cmdBufInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(context->getDevice(), &cmdBufInfo, &commandBuffer));
}


void VulkanGpuFrameImage::initStagingBuffer()
{
    // Create a staging buffer.
    SNN_ASSERT(sizeInBytes > 0);
    std::pair<VkBuffer, VkDeviceMemory> bufAndMem =
        createBuffer(context->getDevice(), context->getPhysicalDeviceMemoryProperties(),
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sizeInBytes);
    stagingBuffer = bufAndMem.first;
    stagingMemory = bufAndMem.second;
}

}
