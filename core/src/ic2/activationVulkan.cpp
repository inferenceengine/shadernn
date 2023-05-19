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
#include "activation.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include "uvkc/vulkan/pipeline.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(Activation);

using namespace snn;
using namespace snn::dp;
using namespace std;

static constexpr const char* ACTIVATION_VK_ASSET_NAME = "shaders/shadertemplate_vk_activation.spv";
static constexpr const char* ACTIVATION_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_activation_fp16.spv";

InferencePassesSptr ActivationLayerVulkan::createCS(const LayerGenOptions&) const {
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

    std::vector<::uvkc::vulkan::Pipeline::SpecConstant> specConstants;

    uvkc::vulkan::Pipeline::SpecConstant tmpConstant;
    tmpConstant.id = 0;
    tmpConstant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::u32;
    tmpConstant.value.u32 = inputWidth;
    specConstants.push_back(tmpConstant);

    tmpConstant.id = 1;
    tmpConstant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::u32;
    tmpConstant.value.u32 = inputHeight;
    specConstants.push_back(tmpConstant);

    tmpConstant.id = 2;
    tmpConstant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::u32;
    tmpConstant.value.u32 = inputDepth;
    specConstants.push_back(tmpConstant);

    tmpConstant.id = 3;
    tmpConstant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::u32;
    tmpConstant.value.u32 = 0;
    specConstants.push_back(tmpConstant);

    tmpConstant.id = 4;
    tmpConstant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::f32;
    tmpConstant.value.f32 = 1.0f;
    specConstants.push_back(tmpConstant);

    std::pair<std::string, std::vector<uvkc::vulkan::Pipeline::SpecConstant>> reluParam("relu", specConstants);
    pass.specConstants = specConstants;
    pass.pushConstants.insert(reluParam);

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

    std::vector<uint32_t> uniform(8);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;
    uniform[2] = inputDepth;
    uniform[3] = 1;
    uniform[4] = activation;
    unsigned char* target = (unsigned char*)uniform.data() + 5 * sizeof(uint32_t);
    std::memcpy(target, &leakyValue, sizeof(uint32_t));

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    pass.inputs  = {{"uInput", 0}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(ACTIVATION_VK_FP16_ASSET_NAME);
        pass.source = ACTIVATION_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(ACTIVATION_VK_ASSET_NAME);
        pass.source = ACTIVATION_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
