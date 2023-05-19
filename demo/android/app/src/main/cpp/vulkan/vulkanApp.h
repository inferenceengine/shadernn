/* Copyright (c) 2018-2021, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

#include "androidWindow.h"
#include "instance.h"
#include "volk.h"

#include <vector>
#include <memory>
#include <array>
#include <atomic>
#include <mutex>

namespace snn {

class VulkanApp
{
    /**
     * @brief Swapchain state
     */
    struct SwapchainDimensions
    {
        /// Width of the swapchain.
        uint32_t width = 0;

        /// Height of the swapchain.
        uint32_t height = 0;

        /// Pixel format of the swapchain.
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    /**
     * @brief Per-frame data
     */
    struct PerFrame
    {
        VkDevice device = VK_NULL_HANDLE;

        VkFence queue_submit_fence = VK_NULL_HANDLE;

        VkCommandPool primary_command_pool = VK_NULL_HANDLE;

        VkCommandBuffer primary_command_buffer = VK_NULL_HANDLE;

        VkSemaphore swapchain_acquire_semaphore = VK_NULL_HANDLE;

        VkSemaphore swapchain_release_semaphore = VK_NULL_HANDLE;

        int32_t queue_index;
    };

    /**
     * @brief Vulkan objects and global state
     */
    struct Context
    {
        /// The Vulkan instance.
        VkInstance instance = VK_NULL_HANDLE;

        /// The Vulkan physical device.
        VkPhysicalDevice gpu = VK_NULL_HANDLE;

        VkPhysicalDeviceMemoryProperties device_memory_properties;

        /// The Vulkan device.
        VkDevice device = VK_NULL_HANDLE;

        /// The Vulkan device queue.
        VkQueue queue = VK_NULL_HANDLE;

        /// The swapchain.
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;

        /// The swapchain dimensions.
        SwapchainDimensions swapchain_dimensions;

        /// The surface we will render to.
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        // Number of valid timestamp bits
        int32_t timestamp_valid_bits = 0;

        /// The queue family index where graphics work will be submitted.
        int32_t graphics_queue_index = -1;

        /// The image view for each swapchain image.
        std::vector<VkImageView> swapchain_image_views;

        /// The framebuffer for each swapchain image view.
        std::vector<VkFramebuffer> swapchain_framebuffers;

        /// The renderpass description.
        VkRenderPass render_pass = VK_NULL_HANDLE;

        /// The graphics pipeline.
        VkPipeline pipeline = VK_NULL_HANDLE;

        /// The pipeline layout for resources.
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

        /// The debug report callback.
        VkDebugReportCallbackEXT debug_callback = VK_NULL_HANDLE;

        /// A set of semaphores that can be reused.
        std::vector<VkSemaphore> recycled_semaphores;

        /// A set of per-frame data.
        std::vector<PerFrame> per_frame;

        // Command pool for initialization work
        VkCommandPool command_pool;

        // Descriptor set layout
        VkDescriptorSetLayout descriptor_set_layout;

        // Descriptor pool
        VkDescriptorPool descriptor_pool;

        // Descriptor sets
        std::vector<VkDescriptorSet> descriptor_sets;
    };

public:
    VulkanApp() = default;

    ~VulkanApp();

    void prepare(vkb::AndroidWindow& aWindow);

    void update();

    bool resize(const uint32_t width, const uint32_t height);

private:
    bool validate_extensions(const std::vector<const char *> &required,
                             const std::vector<VkExtensionProperties> &available);

    bool validate_layers(const std::vector<const char *> &required,
                         const std::vector<VkLayerProperties> &available);

    void init_instance(const std::vector<const char *> &required_instance_extensions,
                       const std::vector<const char *> &required_validation_layers);

    void init_device(const std::vector<const char *> &required_device_extensions);

    void init_command_pool();

    void init_per_frame(PerFrame &per_frame);

    void teardown_per_frame(PerFrame &per_frame);

    void init_swapchain();

    void init_render_pass();

    void init_descriptor_set_layout();

    void init_pipeline();

    void init_descriptors();

    void update_descriptors(VkImageView texture_image_view, VkSampler texture_sampler);

    void init_gpu_context();

    VkResult acquire_next_image(uint32_t *image);

    void render_surface(uint32_t swapchain_index);

    VkResult present_image(uint32_t index);

    void init_framebuffers();

    void teardown_framebuffers();

    void teardown();

private:
    Context context;
    std::unique_ptr<vkb::Instance> vk_instance;
};

}
