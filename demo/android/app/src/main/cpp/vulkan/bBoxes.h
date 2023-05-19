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

#include <vector>
#include <memory>

#define THICK_LINES 1

namespace snn {

class VulkanBBoxes
{
    struct Point2D
    {
        float x = 0.0f;
        float y = 0.0f;
    };

    /**
     * @brief Vulkan objects and global state
     */
    struct Context
    {
        VkDevice device = VK_NULL_HANDLE;

        VkPhysicalDeviceMemoryProperties deviceMemoryProperties;

        VkCommandPool commandPool = VK_NULL_HANDLE;

        VkQueue queue = VK_NULL_HANDLE;
    };

public:
    VulkanBBoxes() = default;

    ~VulkanBBoxes();

    void init(VkDevice device_, const VkPhysicalDeviceMemoryProperties& deviceMemoryProperties_, VkCommandPool commandPool_, VkQueue queue_,
              VkFormat colorFormat_, const VkExtent2D& surfaceSize_);

    void render(VkCommandBuffer cmdBuffer, VkImageView colorAttachment, std::vector<float>& coords);

private:
    std::vector<Point2D> createVertices(std::vector<float>& coords);

    void initVertexBuffer(VkBuffer& vertexBuffer, VkDeviceMemory& vertexBufferMemory, const std::vector<Point2D>& verticesRects);

    std::vector<uint32_t> createIndicies(size_t numBoxes);

    void initIndexBuffer(VkBuffer& indexBuffer, VkDeviceMemory& indexBufferMemory, const std::vector<uint32_t>& indiciesRects);

    void initFramebuffer(VkFramebuffer& frameBuffer, VkImageView colorAttachment);

    void initRenderPass();

    void initPipeline();

    void cleanupTemporaries();

private:
    Context context;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;

    VkImageView colorAttachment;
    VkFormat colorFormat;
    VkExtent2D surfaceSize;

    // Temporary objects
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    VkFramebuffer frameBuffer = VK_NULL_HANDLE;
};

}
