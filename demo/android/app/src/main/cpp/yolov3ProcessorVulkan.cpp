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
#include "yolov3ProcessorVulkan.h"
#include "vulkan/vulkanAppContext.h"
#include "vulkan/vulkanLib.h"
#include "vulkan/error.h"
#include "snn/utils.h"
#include "uvkc/benchmark/vulkan_context.h"
#include <utility>

namespace snn {

Yolov3ProcessorVulkan::Yolov3ProcessorVulkan(ColorFormat format, Precision precision, bool dumpOutputs)
        : Yolov3Processor(format, precision, false, dumpOutputs)
{}

void Yolov3ProcessorVulkan::init(const FrameDims& inputDims_, const FrameDims& outputDims_) {
    Processor::init(inputDims_, outputDims_);

    context = VulkanAppContext::getVulkanContext();

    dp::ShaderGenOptions options = {};
    auto inputTex = InferenceGraph::IODesc {desc().i.format,
                                            modelDims.width, modelDims.height, 1, 4};
    options.desiredInput.push_back(inputTex);

    options.desiredOutputFormat  = desc().o.format;
    options.preferrHalfPrecision = (precision == Precision::FP16);

    options.mrtMode    = snn::MRTMode::SINGLE_PLANE;
    options.weightMode = snn::WeightAccessMethod::CONSTANTS;
    genericModelProcessorVulkan.reset(new GenericModelProcessorVulkan(*this,
        outputDims_,
        options,
        true,   // resizeImage_
        false,   // passThrough_
        0.0f,   // modelInputMean_
        1.0f,   // modelInputNorm_
        0.0f,   // modelOutputMean_
        1.0f));   // modelOutputNorm_

    cmdBuffer = allocateCommandBuffer(context->getDevice(), context->getCommandPool());

    VkFormat imageFormat = getNativeColorVulkan(desc().o.format);  // We will draw on an output image
    bBoxes.init(context->getDevice(), context->getPhysicalDeviceMemoryProperties(), context->getCommandPool(), context->getQueue(),
                imageFormat, {modelDims.width, modelDims.height});

    std::tie(imageOut, imageOutMemory) =
        createImage(context->getDevice(),
                    context->getPhysicalDeviceMemoryProperties(),
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                        | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    VK_IMAGE_TYPE_2D,
                    imageFormat,
                    {modelDims.width, modelDims.height, 1},
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT);

    imageOutView2D = createImageView(context->getDevice(), imageOut, VK_IMAGE_VIEW_TYPE_2D, imageFormat);

    uvkc::benchmark::VulkanContext* uvkcContext = context->getUvkcContext();
    uvkc::vulkan::Device* uvkcDevice = uvkcContext->devices[0].get();
    uvkcImageOut = std::make_shared<uvkc::vulkan::Image>(context->getDevice(), imageOut, imageOutView2D, VK_IMAGE_LAYOUT_UNDEFINED, uvkcDevice->getSymbols());

    GenericModelProcessorVulkan::PostRun postRun = [&](MixedInferenceCore::RunParameters& rp)
    {
        drawBBoxes(rp);
        ImageTextureVulkanArrayAccessor modelOutputs = rp.outputImages;
        modelOutputs[0].attach({uvkcImageOut});
    };
    genericModelProcessorVulkan->setPostRun(postRun);
}

Yolov3ProcessorVulkan::~Yolov3ProcessorVulkan()
{
    if (imageOutView2D != VK_NULL_HANDLE) {
        vkDestroyImageView(context->getDevice(), imageOutView2D, nullptr);
    }
    if (imageOut != VK_NULL_HANDLE) {
        vkDestroyImage(context->getDevice(), imageOut, nullptr);
    }
    if (imageOutMemory != VK_NULL_HANDLE) {
        vkFreeMemory(context->getDevice(), imageOutMemory, nullptr);
    }
}

void Yolov3ProcessorVulkan::submit(Workload& workload) {
    MixedInferenceCore::RunParameters rp = {};
    rp.inputMatrix                       = workload.cpuInputs;
    rp.modelOutput.modelType             = ModelType::DETECTION;

    genericModelProcessorVulkan->submit(workload, rp);
}

void Yolov3ProcessorVulkan::drawBBoxes(MixedInferenceCore::RunParameters& rp)
{
    ImageTextureVulkanArrayAccessor modelInputs = rp.inputImages;
    VkImage srcImage = modelInputs[0].vkImage()->image();

    if (cmdBuffer != VK_NULL_HANDLE) {
        vkResetCommandBuffer(cmdBuffer, /*flags=*/0);
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    copy22DImage(srcImage);

    std::vector<float> coords;
    for (auto &boxDetails : rp.modelOutput.detectionOutput) {
        getBBoxesCoords(boxDetails, coords);
    }
    if (!coords.empty()) {
        bBoxes.render(cmdBuffer, imageOutView2D, coords);
    }

    VK_CHECK(vkEndCommandBuffer(cmdBuffer));
    queueSubmitAndWait(context->getDevice(), context->getQueue(), cmdBuffer);
}

void Yolov3ProcessorVulkan::copy22DImage(VkImage srcImage)
{
    transitionImageLayout(srcImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cmdBuffer);
    transitionImageLayout(imageOut, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, cmdBuffer);

    copyImageToImage(srcImage, imageOut,
                    {0, 0, 0}, // srcOffset
                    {0, 0, 0}, // dstOffset
                    {Yolov3Processor::modelDims.height, Yolov3Processor::modelDims.height, 1}, // extent
                    cmdBuffer);

    transitionImageLayout(srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, cmdBuffer);
    transitionImageLayout(imageOut, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, cmdBuffer);
}

}
