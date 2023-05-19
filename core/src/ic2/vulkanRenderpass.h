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
#include "snn/imageTexture.h"
#include "snn/inferencegraph.h"
#include "inferencepassVulkan.h"
#include "renderpass.h"
#include "uvkc/benchmark/vulkan_context.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace snn {

// This classes implements actions, performed during a render pass.
class VulkanRenderPass : public RenderPass {
public:
    virtual ~VulkanRenderPass() = default;

    SNN_NO_COPY(VulkanRenderPass);
    SNN_NO_MOVE(VulkanRenderPass);

    // Creation parameters structure
    struct CreationParameters {
        std::string name;                                   // Name
        InferencePassVulkan pass;                           // Inference pass
        std::vector<uvkc::vulkan::Sampler*> samplers;       // An array of Vulkan sampler objects. Used to sample inputs
        std::vector<uvkc::vulkan::Sampler*> weightSamplers; // An array of Vulkan sampler objects. Used to sample weights
        ImageTextureArrayAccessor texInputs;                // Input images
        ImageTextureArrayAccessor texOutputs;               // Output images
        uvkc::vulkan::Device *device;                       // Pointer to Vulkan device object
        uvkc::vulkan::CommandBuffer *cmdBuffer;             // Pointer to Vulkan command buffer object
    };

    // Constructor
    // params:
    //  context_ - Pointer to Vulkan context object
    //  cp - creation parameters
    VulkanRenderPass(GpuContext* context_, const CreationParameters& cp);

    // Dump layer inputs
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    bool debugPassInputs(const std::string& folderName) override;

    // Dump layer weights
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    bool debugPassWeights(const std::string& foldername, int shaderPass) override;

    // Run render pass
    void run() override;

private:
    GpuContext* context;
    CreationParameters _cp;
    std::unique_ptr<uvkc::vulkan::ShaderModule> _shaderModule;
    std::unique_ptr<uvkc::vulkan::Pipeline> _pipeline;
    std::unique_ptr<uvkc::vulkan::DescriptorPool> _descriptorPool;
    std::vector<::uvkc::vulkan::Device::BoundBuffer> _boundBuffers;
    std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet> _layoutSetMap;
    std::vector<std::unique_ptr<uvkc::vulkan::Buffer>> _uniformBuffers;
    uint32_t _runIdx = 0;
    std::unordered_map<std::string, std::unique_ptr<uvkc::vulkan::Buffer>> _runtimeBuffers;
    std::vector<std::shared_ptr<uvkc::vulkan::Image>> _srcImages, _dstImages;
    std::vector<std::shared_ptr<uvkc::vulkan::Image>> _weightImages;
    std::vector<uvkc::vulkan::Device::BoundImage> _boundImages;
    std::unique_ptr<::uvkc::vulkan::TimestampQueryPool> _tsQueryPool;
};

} // namespace snn
