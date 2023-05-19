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
#include "dp.h"
#include "inferencepassVulkan.h"
#include "snn/core.h"
#include "vulkanRenderpass.h"
#include "imageTextureVulkan.h"
#include "colorVulkan.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_image_util.h"
#include <string>
#include <memory>
#include <vector>
#include <variant>

std::unique_ptr<uvkc::vulkan::Buffer> createVkBuffer(uvkc::vulkan::Device* device, void* srcBuffer, uint32_t bufSize) {
    uint32_t allocSize = bufSize;
    if ((bufSize%16) != 0) {
        allocSize = ROUND_UP(bufSize, 16); //Align to vec4 float
    }
    BM_CHECK_OK_AND_ASSIGN(
        auto buffer,
        device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocSize));

    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, buffer.get(), bufSize,
        [&](void *ptr, size_t /*num_bytes*/) {
            uint32_t *dstBuffer = reinterpret_cast<uint32_t *>(ptr);
            memcpy(dstBuffer, srcBuffer, bufSize);
        }));

    return buffer;
}

void updateVkBuffer(uvkc::vulkan::Device* device, uvkc::vulkan::Buffer *buffer, void* srcBuffer, uint32_t bufSize) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, buffer, bufSize,
        [&](void *ptr, size_t /*num_bytes*/) {
            uint32_t *dstBuffer = reinterpret_cast<uint32_t *>(ptr);
            memcpy(dstBuffer, srcBuffer, bufSize);
        }));
}

std::shared_ptr<uvkc::vulkan::Image> createVkImage(uvkc::vulkan::Device* device, snn::ColorFormat format, uint32_t width, uint32_t height,
    uint32_t depth, void* srcBuffer) {
    uint32_t align = 1;
    uint32_t alignedWidth  = width % align ? ROUND_UP(width, align) : width;
    uint32_t alignedHeight = height % align ? ROUND_UP(height, align) : height;
    uint32_t alignedDepth  = depth % align ? ROUND_UP(depth, align) : depth;
    VkExtent3D dimensions = {alignedWidth, alignedHeight, alignedDepth};
    auto vkFormat = snn::getNativeColorVulkan(format);
    BM_CHECK_OK_AND_ASSIGN(
        std::unique_ptr<uvkc::vulkan::Image> vkImageUniq,
        device->CreateImage(
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_3D,
            vkFormat, dimensions, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_VIEW_TYPE_3D));
    std::shared_ptr<uvkc::vulkan::Image> image(vkImageUniq.release());

    size_t outputBytes = alignedWidth * alignedHeight * alignedDepth * snn::getColorFormatDesc(format).bytes();
    size_t elements = alignedWidth * alignedHeight * alignedDepth * snn::getColorFormatDesc(format).ch;

    BM_CHECK_OK(::uvkc::benchmark::SetDeviceImageViaStagingBuffer(
        device, image.get(), dimensions,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, outputBytes,
    [&](void *ptr, size_t numBytes) {
        float *dstBuffer = reinterpret_cast<float *>(ptr);
        if (format == snn::ColorFormat::RGBA16F) {
            for (uint32_t i = 0; i < elements; i++) {
                auto f32 = reinterpret_cast<float *>(srcBuffer) + i;
                *((uint16_t*) ptr + i) = snn::FP32::toHalf(f32[0]);
            }
        } else {
            memcpy(dstBuffer, srcBuffer, outputBytes);
        }
        (void) numBytes;
        (void) outputBytes;
        (void) dstBuffer;
        (void) srcBuffer;
    }));

    return image;
}

// -----------------------------------------------------------------------------
//
snn::VulkanRenderPass::VulkanRenderPass(GpuContext* context_, const snn::VulkanRenderPass::CreationParameters& cp)
    : context(context_)
    , _cp(cp)
{
    SNN_LOGV("this: %p, cp.pass.source: %s", this, _cp.pass.source.c_str());
    for (size_t i = 0; i < _cp.texInputs.size(); ++i) {
        SNN_LOGD("cp.texInput[%lu]: %s", i, _cp.texInputs[i].getTextureInfo2().c_str());
    }
    BM_CHECK_OK_AND_ASSIGN(this->_shaderModule,
                        (_cp.device->CreateShaderModule(_cp.pass.vkCodes.data(), _cp.pass.vkCodes.size())));

    auto constCount = absl::MakeSpan(_cp.pass.specConstants.data(), _cp.pass.specConstants.size());
    BM_CHECK_OK_AND_ASSIGN(this->_pipeline, (_cp.device->CreatePipeline(*_shaderModule, "main", constCount)));

    BM_CHECK_OK_AND_ASSIGN(this->_descriptorPool,
                            (_cp.device->CreateDescriptorPool(*_shaderModule)));
    BM_CHECK_OK_AND_ASSIGN(this->_layoutSetMap,
                            _descriptorPool->AllocateDescriptorSets(
                            _shaderModule->descriptor_set_layouts()));

    BM_CHECK_OK_AND_ASSIGN(this->_tsQueryPool, _cp.device->CreateTimestampQueryPool(2));

    auto uniformBuffers = _cp.pass.uniformBuffers;
    auto objectBuffers = _cp.pass.objectBuffers;
    auto weightBuffers = _cp.pass.weightBuffers;
    auto weightDims = _cp.pass.weightDims;
    this->_uniformBuffers.clear();
    this->_uniformBuffers.resize(uniformBuffers.size() + objectBuffers.size());
    this->_weightImages.clear();
    this->_weightImages.resize(weightBuffers.size());
    this->_boundImages.clear();

    uint32_t idx = 0;

    if (uniformBuffers.size() > 0) {
        for (auto iter = uniformBuffers.begin(); iter != uniformBuffers.end(); ++iter) {
            auto bufferId = std::stoi(iter->first);
            auto uniformBuffer = iter->second;
            _uniformBuffers[idx] = (createVkBuffer(_cp.device, uniformBuffer.data(), uniformBuffer.size()*4));
            _boundBuffers.push_back({_uniformBuffers[idx].get(), 0, static_cast<uint32_t>(bufferId)});
            idx++;
        }
    }
    if (objectBuffers.size() > 0) {
        for (auto iter = objectBuffers.begin(); iter != objectBuffers.end(); ++iter) {
            auto bufferId = std::stoi(iter->first);
            auto uniformBuffer = iter->second;
            _uniformBuffers[idx] = (createVkBuffer(_cp.device, uniformBuffer.data(), uniformBuffer.size()*4));
            _boundBuffers.push_back({_uniformBuffers[idx].get(), 0, static_cast<uint32_t>(bufferId)});
            idx++;
        }
    }

    auto runtimeUniforms = _cp.pass.runtimeUniforms;
    for (auto& [name, value] : cp.pass.runtimeUniforms) {
        auto uniformBuffer = _cp.pass.runtimeData;
        uint32_t offset = value.first;
        uint32_t len = value.second;
        _runtimeBuffers.insert({ name, createVkBuffer(_cp.device, uniformBuffer.data()+offset, len*4) });
    }

    idx = 0;
    if (weightBuffers.size() > 0) {
        auto formats = _cp.pass.weightFormats;
        for (auto iter = weightBuffers.begin(); iter != weightBuffers.end(); ++iter) {
            auto imageId = std::stoi(iter->first);
            auto imageData = iter->second;
            auto dims =  weightDims[iter->first];
            auto format = formats[iter->first];
            _weightImages[idx] = createVkImage(_cp.device, format, dims[0], dims[1], dims[2], imageData.data());
            _boundImages.push_back({_weightImages[idx].get(), _cp.samplers[0], 0, static_cast<uint32_t>(imageId)});
            SNN_LOGD("boundImages: (weights) VkImage: %p, VkImageView: %p", _weightImages[idx].get()->image(), _weightImages[idx].get()->image_view());
            idx++;
        }
    }
}


void snn::VulkanRenderPass::run() {
    // Update the runtime uniforms variables;
    if (_cp.pass.runtimeUniforms.size() > 0) {
        size_t len = _cp.pass.runtimeData.size()/_cp.pass.period;
        for (auto& [name, value] : _cp.pass.runtimeUniforms) {
            auto uniformBuffer = _cp.pass.runtimeData.data() + UP_DIV(_runIdx, _cp.pass.totalPasses) % len * _cp.pass.period;
            uint32_t offset = value.first;
            uint32_t len = value.second;
            updateVkBuffer(_cp.device, _runtimeBuffers[name].get(), uniformBuffer + offset, len*4);
            auto bufferId = std::stoi(name);
            _boundBuffers.push_back({_runtimeBuffers[name].get(), 0, static_cast<uint32_t>(bufferId)});
            SNN_LOGD("Update %d runtime parameter for %s, %d:%d at %d", _runIdx, name.c_str(),
                offset, len, UP_DIV(_runIdx, _cp.pass.totalPasses) % len * _cp.pass.period);
        }
    }
    _runIdx++;

    BM_CHECK_OK(_cp.device->AttachBufferToDescriptor(
        *_shaderModule, _layoutSetMap,
        {_boundBuffers.data(), _boundBuffers.size()}));

    std::vector<std::shared_ptr<uvkc::vulkan::Image>> srcImages;
    snn::ImageTextureVulkanArrayAccessor texInputsVulkan = _cp.texInputs;
    snn::ImageTextureVulkanArrayAccessor texOutputsVulkan = _cp.texOutputs;
    srcImages.resize(_cp.pass.inputs.size());
    for (auto[name, index] : _cp.pass.inputs) {
        SNN_ASSERT(index < texInputsVulkan.size());
        std::shared_ptr<uvkc::vulkan::Image> img = texInputsVulkan[index].vkImage(0);
        srcImages[index] = img;
        SNN_LOGV("input: %s, idx: %d, VkImage: %p, VkImageView: %p", name.c_str(), index, img->image(), img->image_view());
    }

    std::vector<std::shared_ptr<uvkc::vulkan::Image>> dstImages;
    SNN_ASSERT(texOutputsVulkan.size() > 0);
    dstImages.push_back(texOutputsVulkan[0].vkImage(0));

    std::vector<uvkc::vulkan::Device::BoundImage> boundImages(_boundImages);
    boundImages.push_back({dstImages[0].get(), _cp.samplers[0], /*set=*/0, /*binding=*/0});
    SNN_LOGD("boundImages (dest): VkImage: %p, VkImageView: %p", dstImages[0].get()->image(), dstImages[0].get()->image_view());

    for (size_t i = 0; i< srcImages.size(); i++) {
        ::uvkc::vulkan::Device::BoundImage srcImage = {srcImages[i].get(), _cp.samplers[i], /*set=*/0U, /*binding=*/static_cast<uint32_t>(i+1U)};
        boundImages.push_back(srcImage);
        SNN_LOGD("boundImages (src): VkImage: %p, VkImageView: %p", srcImages[i].get()->image(), srcImages[i].get()->image_view());
    }

    std::vector<VkImageLayout> image_layouts;
    BM_CHECK_OK(_cp.device->AttachImageToDescriptor(
        *_shaderModule, _layoutSetMap,
        {boundImages.data(), boundImages.size()},
        &image_layouts));
    SNN_ASSERT(boundImages.size() == image_layouts.size());

    BM_CHECK_EQ(_shaderModule->descriptor_set_layouts().size(), 1)
        << "unexpected number of descriptor sets";
    auto descriptorSetLayout = _shaderModule->descriptor_set_layouts().front();

    std::vector<uvkc::vulkan::CommandBuffer::BoundDescriptorSet> boundDescriptorSets(1);
    boundDescriptorSets[0].index = 0;
    boundDescriptorSets[0].set = _layoutSetMap.at(descriptorSetLayout);

    std::vector<uvkc::vulkan::CommandBuffer::PipelineBarrierInfo> barriers_info;
    for (size_t j = 0; j < boundImages.size(); ++j) {
        BM_CHECK_OK(
        _cp.cmdBuffer->AddTransitionImageLayout(*(boundImages[j].image), image_layouts[j], barriers_info));
    }
    // Minimizing the number of pipeline barriers, by combining image transitions with compatible
    // pipeline stages into oine pipeline barrier
    _cp.cmdBuffer->TransitionImageLayout(barriers_info);

    _cp.cmdBuffer->BindPipelineAndDescriptorSets(
        *_pipeline, {boundDescriptorSets.data(), boundDescriptorSets.size()});

    // Dispatch vulkan pipeline
    const InferencePassVulkan::VkProgram& vk = _cp.pass.program;
    SNN_LOGD("Dispatch vulkan pipeline: %d, %d, %d", vk.dispatchSize[0], vk.dispatchSize[1], vk.dispatchSize[2]);
    _cp.cmdBuffer->Dispatch(vk.dispatchSize[0], vk.dispatchSize[1], vk.dispatchSize[2]);

    _cp.cmdBuffer->DispatchBarrier();
}

bool snn::VulkanRenderPass::debugPassInputs(const std::string& folderName) {
    auto image = _cp.texInputs[0].getRawImage();
    std::string layerName = normalizeName(_cp.name);
    std::string path = formatString("%s/%s", folderName.c_str(), layerName.c_str());
    if (!createDirIfNotExists(path)) {
        return false;
    }

    SNN_LOGD("Saving input dump for layer: %s %zu", layerName.c_str(), image.depth());
    for (std::size_t i = 0; i < image.depth(); i++) {
        auto filename = formatString("%s/%s/%02d_input.png", folderName.c_str(), layerName.c_str(), i);
        image.saveToPNG(filename, i, true);
    }

    auto binFilename = formatString("%s/%s_input.dump", folderName.c_str(), _cp.name.c_str());
    image.saveToBIN(binFilename);
#if DUMP_RESULTS_TXT
    auto dumpTxtFileName = formatString("%s/%s_input.txt", folderName.c_str(), _cp.name.c_str());
    if (FILE* fDumpTxt = createFile(dumpTxtFileName.c_str())) {
        _cp.texInputs[0].prettyPrint(fDumpTxt);
        fclose(fDumpTxt);
    }
#endif
    return true;
}

bool snn::VulkanRenderPass::debugPassWeights(const std::string& folderName, int shaderPass) {
    (void) folderName;
    (void) shaderPass;
#if DUMP_RESULTS_TXT
    std::string path = formatString("%s/%s", folderName.c_str(), _cp.name.c_str());
    if (!createParentDirIfNotExists(path)) {
        return false;
    }
    size_t idx = 0;
    for (auto iter = _cp.pass.weightBuffers.begin(); iter != _cp.pass.weightBuffers.end(); ++iter, ++idx) {
        auto dims =  _cp.pass.weightDims[iter->first];
        auto format = _cp.pass.weightFormats[iter->first];
        ImageTextureVulkan tex(context, {dims[0], dims[1], dims[2]}, format, "weights");
        tex.attach({_weightImages[idx]});
        auto dumpTxtFileName = formatString("%s/%s_weights_%s.txt", folderName.c_str(), _cp.name.c_str(), iter->first.c_str());
        if (FILE* fDumpTxt = createFile(dumpTxtFileName.c_str())) {
            tex.prettyPrint(fDumpTxt);
            fclose(fDumpTxt);
        }
    }
#endif
    return true;
}
