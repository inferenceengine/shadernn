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
#include "concatenation.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(Concatenate);

using namespace snn;
using namespace snn::dp;

static constexpr const char* CONCATENATION_VK_ASSET_NAME = "shaders/shadertemplate_vk_concat.spv";
static constexpr const char* CONCATENATION_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_concat_fp16.spv";

InferencePassesSptr ConcatenateLayerVulkan::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesSptr ret(new InferencePassesVulkan());

    std::vector<InferencePassVulkan>& passes = InferencePassesVulkan::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassVulkan& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t input0Depth = inputDims[0].depth;
    uint32_t input1Depth = inputDims[1].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    SNN_LOGD("outputWidth: %d; outputHeight: %d; outputDepth: %d, HalfPrecision:%d", outputWidth, outputHeight, outputDepth, _desc.preferHp);

    outputDepth = _desc.numOutputPlanes;

    uint32_t oc_4 = input0Depth + input1Depth; // UP_DIV(_desc.numOutputPlanes, unit);
    SNN_LOGD("oc_4: %d", oc_4);

    std::vector<uint32_t> uniform(2);
    uniform[0] = input0Depth;
    uniform[1] = input1Depth;

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("3", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    pass.inputs  = {{"uInput0", 0}, {"uInput1", 1}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(CONCATENATION_VK_FP16_ASSET_NAME);
        pass.source = CONCATENATION_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(CONCATENATION_VK_ASSET_NAME);
        pass.source = CONCATENATION_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                            // div-by-N is determined by work group size defined CS program.
                                            {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGD("input = %d:%d:%d+%d, output = %d:%d:%d", inputWidth, inputHeight, input0Depth, input1Depth, outputWidth, outputHeight, outputDepth);

    return ret;
}
