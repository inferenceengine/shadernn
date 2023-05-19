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
#include "flattenlayer.h"
#include "cpulayer.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include <string>
#include <cstring>
#include <vector>

DECLARE_LAYER_VULKAN_CLASS(Flatten);

using namespace snn;
using namespace snn::dp;

static constexpr const char* FLATTEN_VK_ASSET_NAME = "shaders/shadertemplate_vk_flatten.spv";
static constexpr const char* FLATTEN_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_flatten_fp16.spv";

InferencePassesSptr FlattenLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    std::vector<uint32_t> uniform(2);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    pass.inputs  = {{"uInput", 0}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(FLATTEN_VK_FP16_ASSET_NAME);
        pass.source = FLATTEN_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(FLATTEN_VK_ASSET_NAME);
        pass.source = FLATTEN_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                            {1, 1, inputDims[0].depth}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
