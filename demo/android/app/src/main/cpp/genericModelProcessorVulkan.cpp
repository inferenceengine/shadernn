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
#include "genericModelProcessorVulkan.h"
#include "vulkan/vulkanAppContext.h"
#include "snn/utils.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "snn/vulkanImageHandle.h"
#include "uvkc/vulkan/image.h"
#include <array>
#include <memory>

namespace snn {

GenericModelProcessorVulkan::GenericModelProcessorVulkan(ModelProcessorParams& modelProcessorParams_,
                                                         const Processor::FrameDims& outputDims_,
                                                         dp::ShaderGenOptions options_,
                                                         bool resizeImage_,
                                                         bool inputImagePassThrough_,
                                                         float modelInputMean_,
                                                         float modelInputNorm_,
                                                         float modelOutputMean_,
                                                         float modelOutputNorm_)
    : modelProcessorParams(modelProcessorParams_)
    , outputDims(outputDims_)
    , options(options_)
    , resizeImage(resizeImage_)
    , inputImagePassThrough(inputImagePassThrough_)
    , modelInputMean(modelInputMean_)
    , modelInputNorm(modelInputNorm_)
    , modelOutputMean(modelOutputMean_)
    , modelOutputNorm(modelOutputNorm_)
    , inputUvkcImage()
{
    options.vulkan = true;
    options.compute = false;

    vulkanContext = VulkanAppContext::getVulkanContext();
    uvkcContext = vulkanContext->getUvkcContext();

    modelInput.reset(new ImageTextureVulkan(vulkanContext));
    modelOutput.reset(new ImageTextureVulkan(vulkanContext));

    inputImageResizeOp.reset(new VulkanImageResizeOp());
    outputImageResizeOp.reset(new VulkanImageResizeOp());

    SNN_ASSERT(uvkcContext);
    uvkc::vulkan::Device* device = uvkcContext->devices[0].get();

    modelInput->resetTexture({modelProcessorParams.modelDims.width, modelProcessorParams.modelDims.height, modelProcessorParams.modelDims.depth, 1},
        options.desiredInput[0].format, "Model input");
    if (resizeImage) {
        inputImageResizeOp->init(device, true);
        std::array<float, 4> modelInputMeans = {modelInputMean, modelInputMean, modelInputMean, modelInputMean};
        std::array<float, 4> modelInputNorms = {modelInputNorm, modelInputNorm, modelInputNorm, modelInputNorm};
        inputImageResizeOp->updateParams({modelProcessorParams.modelDims.width, modelProcessorParams.modelDims.height, 1}, modelInputMeans, modelInputNorms);
    }

    if (resizeImage) {
        modelOutput->resetTexture({modelProcessorParams.modelDims.width, modelProcessorParams.modelDims.height, modelProcessorParams.modelDims.depth, 1},
            options.desiredOutputFormat, "Model output");
        outputImageResizeOp->init(device, true);
        std::array<float, 4> modelOutputMeans = {modelOutputMean, modelOutputMean, modelOutputMean, modelOutputMean};
        std::array<float, 4> modelOutputNorms = {modelOutputNorm, modelOutputNorm, modelOutputNorm, modelOutputNorm};
        outputImageResizeOp->updateParams({outputDims.width, outputDims.height, outputDims.depth}, modelOutputMeans, modelOutputNorms);
        outputImageResizeOp->setFinalDstLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    else {
        modelOutput->resetTexture({outputDims.width, outputDims.height, outputDims.depth, 1}, options.desiredOutputFormat, "Model output");
    }
}

void GenericModelProcessorVulkan::submit(Processor::Workload& workload, MixedInferenceCore::RunParameters& rp) {
    VulkanImageHandle inputImageHandle;
    workload.inputs[0]->getGpuImageHandle(inputImageHandle);
    VulkanImageHandle outputImageHandle;
    workload.output->getGpuImageHandle(outputImageHandle);

    uvkc::vulkan::Device* device = uvkcContext->devices[0].get();

    // Create non-owning uvkc::vulkan::Image wrapper around input image
    if (!inputUvkcImage) {
        inputUvkcImage = std::make_shared<uvkc::vulkan::Image>(device->getLogicalDevice(), inputImageHandle.image, inputImageHandle.imageView,
            inputImageHandle.imageLayout, device->getSymbols());
    } else {
        *inputUvkcImage.get() = uvkc::vulkan::Image(device->getLogicalDevice(), inputImageHandle.image, inputImageHandle.imageView,
            inputImageHandle.imageLayout, device->getSymbols());
    }
    // Create non-owning uvkc::vulkan::Image wrapper around output image
    if (!outputUvkcImage) {
        outputUvkcImage = std::make_shared<uvkc::vulkan::Image>(device->getLogicalDevice(), outputImageHandle.image, outputImageHandle.imageView,
            outputImageHandle.imageLayout, device->getSymbols());
    } else {
        *outputUvkcImage.get() = uvkc::vulkan::Image(device->getLogicalDevice(), outputImageHandle.image, outputImageHandle.imageView,
            outputImageHandle.imageLayout, device->getSymbols());
    }

    if (resizeImage) {
        inputImageResizeOp->run(inputUvkcImage.get(), modelInput->vkImage().get());
    } else {
        modelInput->attach({inputUvkcImage});
        modelOutput->attach({outputUvkcImage});
    }
    ImageTextureVulkanArray modelInputs(modelInput, ImageTextureVulkanAllocator(vulkanContext));
    rp.inputImages = modelInputs;

    ImageTextureVulkanArray modelOutputs(modelOutput, ImageTextureVulkanAllocator(vulkanContext));
    if (!inputImagePassThrough) {
        rp.outputImages = modelOutputs;
    }

    if (!modelProcessorParams.ic2) {
        auto dp = snn::dp::loadFromJsonModel(modelProcessorParams.modelFileName, true, options.mrtMode, options.weightMode, options.preferrHalfPrecision);
        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp[0], options);
        cp.dumpOutputs = modelProcessorParams.dumpOutputs;
        modelProcessorParams.ic2 = MixedInferenceCore::create(vulkanContext, cp);
    }

    modelProcessorParams.ic2->run(rp);

    if (postRunSet) {
        postRun(rp);
    }

    if (resizeImage) {
        if (!inputImagePassThrough) {
            outputImageResizeOp->run(modelOutput->vkImage().get(), outputUvkcImage.get());
        }
        else {
            // TODO: This is kinda ugly that we really do dummy resize to get input image as output
            // The same is in OpenGL pass
            // We need to revisit our engine structure to not create an output image frame, but reference the input instead
            outputImageResizeOp->run(inputUvkcImage.get(), outputUvkcImage.get());
        }
    }
}

}
