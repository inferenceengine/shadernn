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

#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/image.h"
#include "uvkc/vulkan/shader_module.h"
#include "uvkc/vulkan/pipeline.h"
#include "uvkc/vulkan/descriptor_pool.h"

#include <memory>
#include <unordered_map>
#include <array>

namespace snn {

// This class provides a general facility to transform input image to output image using a shader
class VulkanImageTransformShaderOp {
public:
    void init(uvkc::vulkan::Device *device, const uint32_t *spirvData, size_t spirvSize, size_t paramsBufferSize);

    void setFinalSrcLayout(VkImageLayout finalSrcLayout);

    void setFinalDstLayout(VkImageLayout finalDstLayout);

    void updateParams(void* params, const std::array<uint32_t, 3>& workGroupSizes);

    void run(uvkc::vulkan::Image *srcImage, uvkc::vulkan::Image *dstImage);

protected:
    virtual absl::StatusOr<std::unique_ptr<uvkc::vulkan::Sampler>> createSampler();

    uvkc::vulkan::Device *_device;
    std::unique_ptr<uvkc::vulkan::ShaderModule> _shaderModule;
    std::vector<uvkc::vulkan::Pipeline::SpecConstant> _specConstants;
    std::unique_ptr<uvkc::vulkan::Pipeline> _pipeline;
    std::unique_ptr<uvkc::vulkan::DescriptorPool> _descriptorPool;
    std::unique_ptr<uvkc::vulkan::Sampler> _sampler;
    std::unique_ptr<uvkc::vulkan::Buffer> _paramsBuffer;
    size_t _paramsBufferSize;
    std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet> _layoutSetMap;
    std::array<uint32_t, 3> _workGroupSizes {1U, 1U, 1U};
    VkImageLayout _finalSrcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout _finalDstLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

}
