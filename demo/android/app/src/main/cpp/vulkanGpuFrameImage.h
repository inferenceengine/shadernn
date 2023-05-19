
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
#include "gpuFrameImage.h"
#include "vulkanContext.h"
#include "snn/color.h"
#include "snn/vulkanImageHandle.h"

namespace snn {

class VulkanGpuFrameImage : public GpuFrameImage {
public:
    VulkanGpuFrameImage(const Desc& d);

    ~VulkanGpuFrameImage();

    void updateTextureContent(ColorFormat format, void* data, uint32_t size = 0) override;

    void getGpuImageHandle(GpuImageHandle& handle) const override;

private:
    void allocateCommandBuffer();

    void initStagingBuffer();

    uint32_t sizeInBytes = 0;

    VulkanGpuContext* context = nullptr;

    VkImage image = VK_NULL_HANDLE;

    VkImageView imageView = VK_NULL_HANDLE;

    // Device memory for the image
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;

    VkFormat imageFormat = VK_FORMAT_UNDEFINED;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;

    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    VkFence fence = VK_NULL_HANDLE;
};

}
