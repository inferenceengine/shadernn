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
#include "vulkanImageResizeOp.h"
#include "snn/utils.h"
#include "uvkc/base/status.h"
#include "uvkc/vulkan/status_util.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include <vector>

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#endif

static constexpr const char* RESIZE_VK_ASSET_NAME = "shaders/shadertemplate_vk_resize.spv";

namespace snn {

void VulkanImageResizeOp::init(uvkc::vulkan::Device *device, bool linearFilter) {
    _linearFilter = linearFilter;

    std::vector<unsigned char> bytes = loadEmbeddedAsset(RESIZE_VK_ASSET_NAME);
    const uint32_t *spirvData = reinterpret_cast<const uint32_t*>(bytes.data());
    size_t spirvSize = bytes.size() / sizeof(uint32_t);

    VulkanImageTransformShaderOp::init(device, spirvData, spirvSize, sizeof(ResizeImageParams));
}

static absl::StatusOr<std::unique_ptr<uvkc::vulkan::Sampler>> createLinearSampler(uvkc::vulkan::Device *device) {
    SNN_ASSERT(device);

    VkSamplerCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    create_info.pNext = nullptr;
    create_info.flags = 0;
    create_info.magFilter = VK_FILTER_LINEAR;
    create_info.minFilter = VK_FILTER_LINEAR;
    create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    create_info.mipLodBias = 0.0f;
    create_info.anisotropyEnable = VK_FALSE;
    create_info.maxAnisotropy = 0.0f;
    create_info.compareEnable = VK_TRUE;
    create_info.compareOp = VK_COMPARE_OP_NEVER;
    create_info.minLod = 0.0f;
    create_info.maxLod = 0.0f;
    create_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    create_info.unnormalizedCoordinates = VK_FALSE;

    VkSampler sampler = VK_NULL_HANDLE;
    UVKC_RETURN_IF_ERROR(uvkc::vulkan::VkResultToStatus(
        device->getSymbols().vkCreateSampler(
        device->getLogicalDevice(), &create_info, nullptr, &sampler)));
    return std::make_unique<uvkc::vulkan::Sampler>(device->getLogicalDevice(), sampler, device->getSymbols());
}

absl::StatusOr<std::unique_ptr<uvkc::vulkan::Sampler>> VulkanImageResizeOp::createSampler() {
    if (!_linearFilter) {
        return VulkanImageTransformShaderOp::createSampler();
    }
    else {
        return createLinearSampler(_device);
    }
}

void VulkanImageResizeOp::updateParams(VkExtent3D dstDimensions, const std::array<float, 4>& means, const std::array<float, 4>& norms) {
    ResizeImageParams resizeParams = {
        {dstDimensions.width, dstDimensions.height, dstDimensions.depth, 0U},
        means,
        norms
    };

    std::array<uint32_t, 3> localSizes = {4U, 8U, 1U};
    std::array<uint32_t, 3> workGroupSizes = {
        UP_DIV(dstDimensions.width, localSizes[0]),
        UP_DIV(dstDimensions.height, localSizes[1]),
        UP_DIV(dstDimensions.depth, localSizes[2])
    };

    VulkanImageTransformShaderOp::updateParams(&resizeParams, workGroupSizes);
}

}
