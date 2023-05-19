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
#include "subpixelmerge.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include "uvkc/vulkan/pipeline.h"
#include <vector>

DECLARE_LAYER_VULKAN_CLASS(Subpixel);

using namespace snn;
using namespace snn::dp;

static constexpr const char* SUBPIXEL_MERGE_VK_ASSET_NAME = "shaders/shadertemplate_vk_subpixel.spv";
static constexpr const char* SUBPIXEL_MERGE_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_subpixel_fp16.spv";

InferencePassesSptr SubpixelLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    uint32_t subPixelFactor = 2;

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = 1;

    std::vector<uvkc::vulkan::Pipeline::SpecConstant> specConstants {
        {0, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = subPixelFactor}},
        {1, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputWidth}},
        {2, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputHeight}},
        {3, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputDepth}},
        {4, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputWidth}},
        {5, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputHeight}},
        {6, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = outputDepth}},
    };

    pass.specConstants = specConstants;

    pass.inputs = {{"uInput", 0}};

    std::vector<uchar> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(SUBPIXEL_MERGE_VK_FP16_ASSET_NAME);
        pass.source = SUBPIXEL_MERGE_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(SUBPIXEL_MERGE_VK_ASSET_NAME);
        pass.source = SUBPIXEL_MERGE_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}

