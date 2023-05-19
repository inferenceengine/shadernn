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
#include "conv2d.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include "uvkc/vulkan/pipeline.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(Conv2D);

using namespace snn;
using namespace snn::dp;

static constexpr const char* CONV2D_VK_ASSET_NAME = "shaders/shadertemplate_vk_conv2d.spv";
static constexpr const char* CONV2D_1x1_VK_ASSET_NAME = "shaders/shadertemplate_vk_conv2d_1x1.spv";
static constexpr const char* CONV2D_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_conv2d_fp16.spv";
static constexpr const char* CONV2D_1x1_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_conv2d_1x1_fp16.spv";

#define VK_WEIGHT_MODE 1
// 0 in SSBO Buffer, 1 in Texture. Changing it also need to change PROFILE_FLAG in Vulkan operators.

InferencePassesSptr Conv2DLayerVulkan::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesSptr ret(new InferencePassesVulkan());

    std::vector<InferencePassVulkan>& passes = InferencePassesVulkan::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassVulkan& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t inputDepth  = inputDims[0].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    getOutputDims(outputWidth, outputHeight, outputDepth);

    uint32_t activation = 0; // No activation
    float leakyValue = _desc.leakyReluAlpha;
    if (!_desc.activation.compare("relu")) {
        activation = 1;
    } else if (!_desc.activation.compare("relu6")) {
        activation = 2;
    } else if (!_desc.activation.compare("tanh")) {
        activation = 3;
    } else if (!_desc.activation.compare("sigmoid")) {
        activation = 4;
    } else if (!_desc.activation.compare("leakyRelu")) {
        activation = 5;
    } else if (!_desc.activation.compare("SiLU")) {
        activation = 6;
    }

    uint32_t paddingMode = 0;
    if (!_desc.paddingMode.compare("constant")) {
        paddingMode = 1;
    } else if (!_desc.paddingMode.compare("replicate")) {
        paddingMode = 2;
    } else if (!_desc.paddingMode.compare("reflect")) {
        paddingMode = 3;
    }

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    uint32_t dilate = 1;

    oihw2hwo4i4(_desc.weightsCvM, pass._vecWeights, _desc.numInputPlanes, _desc.numOutputPlanes, kernel, kernel);

#if VK_WEIGHT_MODE == 0
    std::pair<std::string, std::vector<float>> weightBuffer("2", pass._vecWeights);
    pass.objectBuffers.insert(weightBuffer);
#else
    std::pair<std::string, std::vector<float>> weightBuffer("2", pass._vecWeights);
    pass.weightBuffers.insert(weightBuffer);
    std::pair<std::string, std::array<uint32_t, 3>> weightDim("2", {ic_4*unit, oc_4, (uint32_t)(kernel*kernel)});
    pass.weightDims.insert(weightDim);
    snn::ColorFormat weightFormat;
    if (_desc.preferHp) {
        weightFormat = snn::ColorFormat::RGBA16F;
    } else {
        weightFormat = snn::ColorFormat::RGBA32F;
    }
    SNN_LOGD("vulkan weight format: %s", _desc.preferHp ? "RGBA16F" : "RGBA32F");
    std::pair<std::string, snn::ColorFormat> weightPrecision("2", weightFormat);
    pass.weightFormats.insert(weightPrecision);
#endif

    pass._vecBias.resize(_desc.numOutputPlanes, 0.0f);
    for (size_t i = 0; i < _desc.biases.size(); i++) {
        pass._vecBias[i] = (float) _desc.biases[i];
    }
    uint32_t useBias = 0;
    if (_desc.biases.size() > 0) {
        useBias = 1;
    }

    std::pair<std::string, std::vector<float>> biasBuffer("3", pass._vecBias);
    pass.objectBuffers.insert(biasBuffer);

    uint32_t useBatchNorm = 1;
    if (_desc.useBatchNormalization) {
        std::pair<std::string, std::vector<float>> betaBuffer("4", _desc.batchNormalization.at("beta"));
        pass.objectBuffers.insert(betaBuffer);

        std::pair<std::string, std::vector<float>> gammaBuffer("5", _desc.batchNormalization.at("gamma"));
        pass.objectBuffers.insert(gammaBuffer);

        std::pair<std::string, std::vector<float>> meanBuffer("6", _desc.batchNormalization.at("movingMean"));
        pass.objectBuffers.insert(meanBuffer);

        std::pair<std::string, std::vector<float>> varBuffer("7", _desc.batchNormalization.at("movingVariance"));
        pass.objectBuffers.insert(varBuffer);
    } else {
        useBatchNorm = 0;

        // Insert dummy buffers to make Vulkan validation happy
        std::pair<std::string, std::vector<float>> betaBuffer("4", {0.0f});
        pass.objectBuffers.insert(betaBuffer);

        std::pair<std::string, std::vector<float>> gammaBuffer("5", {0.0f});
        pass.objectBuffers.insert(gammaBuffer);

        std::pair<std::string, std::vector<float>> meanBuffer("6", {0.0f});
        pass.objectBuffers.insert(meanBuffer);

        std::pair<std::string, std::vector<float>> varBuffer("7", {0.0f});
        pass.objectBuffers.insert(varBuffer);
    }

    if (kernel == 1) {
        SNN_LOGD("activation = %d, leakyValue = %f, batchnorm = %d", activation, leakyValue, useBatchNorm);
        std::vector<uvkc::vulkan::Pipeline::SpecConstant> specConstants = {
            {0, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = stride}},
            {1, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = stride}},
            {2, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputWidth}},
            {3, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputHeight}},
            {4, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = oc_4}},
            {5, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputWidth}},
            {6, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputHeight}},
            {7, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = ic_4}},
            {8, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = 4}},
            {9, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = activation}},
            {10, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = paddingMode}},
            {11, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = useBatchNorm}},
            {12, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = useBias}},
            {13, uvkc::vulkan::Pipeline::SpecConstant::Type::f32, { .f32 = leakyValue}},
        };
        pass.specConstants = specConstants;
    } else {
        uint32_t paddingOffsets[4];
        getPaddingOffset(paddingOffsets);
        SNN_LOGD("Padding = %d:%d:%d:%d, kernel = %d, stride = %d, oc_4 = %d, ic_4 = %d, dilate = %d, activation = %d,"
            " leakyValue = %f, useBatchNorm = %d, useBias = %d",
            paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3],
            kernel, stride, oc_4, ic_4, dilate,
            activation, leakyValue, useBatchNorm, useBias);

        std::vector<uvkc::vulkan::Pipeline::SpecConstant> specConstants = {
            {0, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = paddingOffsets[0]}},
            {1, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = paddingOffsets[2]}},
            {2, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = kernel}},
            {3, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = kernel}},
            {4, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = stride}},
            {5, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = stride}},
            {6, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputWidth}},
            {7, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputHeight}},
            {8, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = oc_4}},
            {9, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputWidth}},
            {10, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputHeight}},
            {11, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = ic_4}},
            {12, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = dilate}},
            {13, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = dilate}},
            {14, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = 4}},
            {15, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = activation}},
            {16, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = paddingMode}},
            {17, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = useBatchNorm}},
            {18, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = useBias}},
            {19, uvkc::vulkan::Pipeline::SpecConstant::Type::f32, { .f32 = leakyValue}},
        };
        pass.specConstants = specConstants;
    }

    pass.inputs  = {{"inputImage", 0}};

    std::vector<uchar> bytes;
    if (kernel == 1) {
        if (_desc.preferHp) {
            bytes = snn::loadEmbeddedAsset(CONV2D_1x1_VK_FP16_ASSET_NAME);
            pass.source = CONV2D_1x1_VK_FP16_ASSET_NAME;
        } else {
            bytes = snn::loadEmbeddedAsset(CONV2D_1x1_VK_ASSET_NAME);
            pass.source = CONV2D_1x1_VK_ASSET_NAME;
        }
    } else {
        if (_desc.preferHp) {
            bytes = snn::loadEmbeddedAsset(CONV2D_VK_FP16_ASSET_NAME);
            pass.source = CONV2D_VK_FP16_ASSET_NAME;
        } else {
            bytes = snn::loadEmbeddedAsset(CONV2D_VK_ASSET_NAME);
            pass.source = CONV2D_VK_ASSET_NAME;
        }
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"outputImage",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, unit * mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]),
                                                    UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGD("input = %d:%d:%d, output = %d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
