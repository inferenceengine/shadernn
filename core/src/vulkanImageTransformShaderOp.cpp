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
#include "vulkanImageTransformShaderOp.h"
#include "snn/utils.h"
#include "uvkc/base/status.h"
#include "uvkc/vulkan/status_util.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"

namespace snn {

void VulkanImageTransformShaderOp::init(uvkc::vulkan::Device *device, const uint32_t *spirvData, size_t spirvSize, size_t paramsBufferSize)
{
    SNN_ASSERT(device);

    _device = device;
    _paramsBufferSize = paramsBufferSize;

    // Create shader module
    BM_CHECK_OK_AND_ASSIGN(
        _shaderModule,
        _device->CreateShaderModule(spirvData, spirvSize));

    auto constCount = absl::MakeSpan(_specConstants.data(), _specConstants.size());

    // Create pipeline
    BM_CHECK_OK_AND_ASSIGN(
        _pipeline,
        _device->CreatePipeline(*_shaderModule, "main", constCount));

    // Create descriptor sets
    BM_CHECK_OK_AND_ASSIGN(
        _descriptorPool,
        _device->CreateDescriptorPool(*_shaderModule));

    BM_CHECK_OK_AND_ASSIGN(
        _layoutSetMap,
        _descriptorPool->AllocateDescriptorSets(
        _shaderModule->descriptor_set_layouts()));

    // Create input sampler
    BM_CHECK_OK_AND_ASSIGN(
        _sampler,
        createSampler());

    // Using direct copy to buffer from memory
    size_t paramsBufferSizeAlloc = std::max(16UL, _paramsBufferSize);
    BM_CHECK_OK_AND_ASSIGN(
        _paramsBuffer,
        _device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             paramsBufferSizeAlloc));
}

void VulkanImageTransformShaderOp::setFinalSrcLayout(VkImageLayout finalSrcLayout) {
    _finalSrcLayout = finalSrcLayout;
}

void VulkanImageTransformShaderOp::setFinalDstLayout(VkImageLayout finalDstLayout) {
    _finalDstLayout = finalDstLayout;
}


absl::StatusOr<std::unique_ptr<uvkc::vulkan::Sampler>> VulkanImageTransformShaderOp::createSampler() {
    return _device->CreateSampler();
}

void VulkanImageTransformShaderOp::updateParams(void* params, const std::array<uint32_t, 3>& workGroupSizes) {
    SNN_ASSERT(params);
    // Using direct copy to buffer from memory
    BM_CHECK_OK_AND_ASSIGN(
        void *mappedPtr,
        _paramsBuffer->MapMemory(0, _paramsBufferSize));
    memcpy(mappedPtr, params, _paramsBufferSize);
    _paramsBuffer->UnmapMemory();

    _workGroupSizes = workGroupSizes;
}

void  VulkanImageTransformShaderOp::run(uvkc::vulkan::Image *srcImage, uvkc::vulkan::Image *dstImage) {
    SNN_ASSERT(srcImage);
    SNN_ASSERT(dstImage);

    std::vector<uvkc::vulkan::Device::BoundImage> boundImages = {
        {dstImage, nullptr, /*set=*/0U, /*binding=*/0U},
        {srcImage, _sampler.get(), /*set=*/0U, /*binding=*/1U}
    };
    SNN_LOGD("boundImages (dst): VkImage: %p, VkImageView: %p", dstImage->image(), dstImage->image_view());
    SNN_LOGD("boundImages (src): VkImage: %p, VkImageView: %p", srcImage->image(), srcImage->image_view());

    std::vector<uvkc::vulkan::Device::BoundBuffer> boundBuffers = {
        {_paramsBuffer.get(), /*set=*/0U, /*binding=*/2U},
    };

    std::vector<VkImageLayout> image_layouts;
    BM_CHECK_OK(
        _device->AttachImageToDescriptor(
        *_shaderModule, _layoutSetMap,
        {boundImages.data(), boundImages.size()},
        &image_layouts));
    SNN_ASSERT(boundImages.size() == image_layouts.size());

    // Attach buffers to descriptor
    BM_CHECK_OK(
        _device->AttachBufferToDescriptor(
        *_shaderModule, _layoutSetMap,
        {boundBuffers.data(), boundBuffers.size()}));

    // Query descriptor set layout
    BM_CHECK_EQ(_shaderModule->descriptor_set_layouts().size(), 1)
        << "unexpected number of descriptor sets";
    auto descriptorSetLayout = _shaderModule->descriptor_set_layouts().front();

    std::vector<uvkc::vulkan::CommandBuffer::BoundDescriptorSet> boundDescriptorSets(1);
    boundDescriptorSets[0].index = 0;
    boundDescriptorSets[0].set = _layoutSetMap.at(descriptorSetLayout);

    // Create command buffer
    BM_CHECK_OK_AND_ASSIGN(
        auto cmdBuffer,
        _device->AllocateCommandBuffer());

    // Start recording commands
    BM_CHECK_OK(
        cmdBuffer->Begin());

    for (size_t j = 0; j < boundImages.size(); ++j) {
        BM_CHECK_OK(
            cmdBuffer->TransitionImageLayout(*(boundImages[j].image), image_layouts[j]));
    }

    // Bind pipeline and descriptors to command buffer
    cmdBuffer->BindPipelineAndDescriptorSets(*_pipeline, {boundDescriptorSets.data(), boundDescriptorSets.size()});

    // Dispatch command
    cmdBuffer->Dispatch(_workGroupSizes[0], _workGroupSizes[1], _workGroupSizes[2]);

    if (_finalSrcLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        BM_CHECK_OK(
            cmdBuffer->TransitionImageLayout(*srcImage, _finalSrcLayout));
    }
    if (_finalDstLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        BM_CHECK_OK(
            cmdBuffer->TransitionImageLayout(*dstImage, _finalDstLayout));
    }

    // End recording command, submit command and wait to completion
    BM_CHECK_OK(
        cmdBuffer->End());
    BM_CHECK_OK(
        _device->QueueSubmitAndWait(*cmdBuffer)); //T.B.D.
}

}
