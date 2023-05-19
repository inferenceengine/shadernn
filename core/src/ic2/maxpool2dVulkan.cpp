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
#include "maxpool2d.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(MaxPooling2D);

using namespace snn;
using namespace snn::dp;

static constexpr const char* MAXPOOLING2D_VK_ASSET_NAME = "shaders/shadertemplate_vk_maxpool2d.spv";
static constexpr const char* MAXPOOLING2D_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_maxpool2d_fp16.spv";

InferencePassesSptr MaxPooling2DLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    uint32_t paddingOffsets[4];
    getPaddingOffset(paddingOffsets);
    SNN_LOGV("Padding: %d, %d, %d, %d", paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3]);
    // Hack it. Looks like not padding on top left in NCNN
    paddingOffsets[0] = 0;
    paddingOffsets[2] = 0;

    std::vector<uvkc::vulkan::Pipeline::SpecConstant> specConstants {
        {0, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputHeight}},
        {1, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputWidth}},
        {2, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputDepth}},
        {3, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputHeight}},
        {4, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputWidth}},
        {5, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputDepth}},
        {6, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = _desc.kernelSize}},
        {7, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = _desc.kernelSize}},
        {8, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = _desc.stride}},
        {9, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = _desc.stride}},
        {10, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .s32 = 0}}, // pooling type, 0->maxpool,1->avgpool
    };

    std::pair<std::string, std::vector<uvkc::vulkan::Pipeline::SpecConstant>> reluParam("relu", specConstants);
    pass.specConstants = specConstants;
    pass.pushConstants.insert(reluParam);

    std::vector<uint32_t> uniform(14);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;
    uniform[2] = inputDepth;
    uniform[3] = 1;
    uniform[4] = outputWidth;
    uniform[5] = outputHeight;
    uniform[6] = UP_DIV(outputDepth, 4);
    uniform[7] = 1;
    uniform[8] = paddingOffsets[0];
    uniform[9] = paddingOffsets[2];
    uniform[10] = _desc.kernelSize;
    uniform[11] = _desc.kernelSize;
    uniform[12] = _desc.stride;
    uniform[13] = _desc.stride;

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    pass.inputs  = {{"uInput", 0}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(MAXPOOLING2D_VK_FP16_ASSET_NAME);
    } else {
        bytes = snn::loadEmbeddedAsset(MAXPOOLING2D_VK_ASSET_NAME);
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    SNN_LOGV("vulkan bytes:%zu: %x, %x", bytes.size(), pass.vkCodes[0], pass.vkCodes[pass.vkCodes.size() - 1]);

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                            // div-by-N is determined by work group size defined CS program.
                                            {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
