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
#include "batchnorm.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

using namespace snn;
using namespace snn::dp;

DECLARE_LAYER_VULKAN_CLASS(BatchNormalization);

static constexpr const char* BATCHNORM_VK_ASSET_NAME = "shaders/shadertemplate_vk_batchnorm.spv";
static constexpr const char* BATCHNORM_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_batchnorm_fp16.spv";

InferencePassesSptr BatchNormalizationLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    std::pair<std::string, std::vector<float>> betaBuffer("3", _desc.batchNormalization.at("beta"));
    pass.objectBuffers.insert(betaBuffer);

    std::pair<std::string, std::vector<float>> gammaBuffer("4", _desc.batchNormalization.at("gamma"));
    pass.objectBuffers.insert(gammaBuffer);

    std::pair<std::string, std::vector<float>> meanBuffer("5", _desc.batchNormalization.at("movingMean"));
    pass.objectBuffers.insert(meanBuffer);

    std::pair<std::string, std::vector<float>> varBuffer("6", _desc.batchNormalization.at("movingVariance"));
    pass.objectBuffers.insert(varBuffer);

    std::vector<uint32_t> uniform(6);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;
    uniform[2] = inputDepth;
    uniform[3] = 1;
    uniform[4] = activation;
    unsigned char* target = (unsigned char*)uniform.data() + 5 * sizeof(uint32_t);
    std::memcpy(target, &leakyValue, sizeof(uint32_t));

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    pass.inputs  = {{"uInput", 0}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(BATCHNORM_VK_FP16_ASSET_NAME);
        pass.source = BATCHNORM_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(BATCHNORM_VK_ASSET_NAME);
        pass.source = BATCHNORM_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                            // div-by-N is determined by work group size defined CS program.
                                            {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGI("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
