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
#include "separableconvolution.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include "uvkc/vulkan/pipeline.h"
#include <string>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(SeparableConv2D);

using namespace snn;
using namespace snn::dp;

static constexpr const char* DEPTHWISE_CONV2D_VK_ASSET_NAME = "shaders/shadertemplate_vk_depthwise.spv";
static constexpr const char* DEPTHWISE_CONV2D_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_depthwise_fp16.spv";

InferencePassesSptr SeparableConv2DLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = _desc.numOutputPlanes;

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

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    uint32_t dilate = 1;

    oihw2hwo4i4(_desc.weightsCvM, pass._vecWeights, _desc.numInputPlanes, _desc.numOutputPlanes, kernel, kernel);

    std::pair<std::string, std::vector<float>> weightBuffer("2", pass._vecWeights);
    pass.objectBuffers.insert(weightBuffer);

    pass._vecBias.resize(_desc.numOutputPlanes);
    for (size_t i = 0; i < _desc.biases.size(); i++) {
        pass._vecBias[i] = (float) _desc.biases[i];
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
    }

    uint32_t paddingOffsets[4];
    this->getPaddingOffset(paddingOffsets);
    SNN_LOGD("%s:%d, Padding: %d, %d, %d, %d\n", __FILENAME__, __LINE__, paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3]);

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
        {17, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = useBatchNorm}}
    };

    uvkc::vulkan::Pipeline::SpecConstant tmpConstant;
    tmpConstant.id = 18;
    tmpConstant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::f32;
    tmpConstant.value.f32 = leakyValue;
    specConstants.push_back(tmpConstant);

    pass.specConstants = specConstants;

    pass.inputs  = {{"inputImage", 0}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(DEPTHWISE_CONV2D_VK_FP16_ASSET_NAME);
        pass.source = DEPTHWISE_CONV2D_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(DEPTHWISE_CONV2D_VK_ASSET_NAME);
        pass.source = DEPTHWISE_CONV2D_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"outputImage",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
