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
#include "denselayer.h"
#include "cpulayer.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(Dense);

using namespace snn;
using namespace snn::dp;

static constexpr const char* DENSE_VK_ASSET_NAME = "shaders/shadertemplate_vk_dense.spv";
static constexpr const char* DENSE_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_dense_fp16.spv";

InferencePassesSptr DenseLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    pass._vecWeights.resize(inputWidth * outputWidth);
    float* destWeight = (float*) pass._vecWeights.data();

    unsigned int width = _desc.weights[0].size();
    int kIndex = 0;
    for (size_t i = 0; i < _desc.weights.size(); i++) {
        for (size_t j = 0; j < width; j++) {
            *(destWeight + kIndex) = _desc.weights[i][j];
            kIndex++;
        }
    }

    pass._vecBias.resize(outputWidth);
    float* destBias = (float*) pass._vecBias.data();

    for (size_t i = 0; i < _desc.biases.size(); i++) {
        *(destBias + i) = _desc.biases[i];
    }

    std::pair<std::string, std::vector<float>> weightBuffer("3", pass._vecWeights);
    pass.objectBuffers.insert(weightBuffer);

    std::pair<std::string, std::vector<float>> biasBuffer("4", pass._vecBias);
    pass.objectBuffers.insert(biasBuffer);

    std::vector<uint32_t> uniform(6);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;
    uniform[2] = activation;
    unsigned char* target = (unsigned char*)uniform.data() + 3 * sizeof(uint32_t);
    memcpy(target, &leakyValue, sizeof(uint32_t));

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    pass.inputs  = {{"uInput", 0}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(DENSE_VK_FP16_ASSET_NAME);
        pass.source = DENSE_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(DENSE_VK_ASSET_NAME);
        pass.source = DENSE_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                            {1, outputWidth, 1}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
