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
#include "upsampling2d.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include <string>
#include <cstring>
#include <vector>
#include <array>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(UpSampling2D);

using namespace snn;
using namespace snn::dp;

static constexpr const char* UPSAMPLING2D_NEAREST_VK_ASSET_NAME   = "shaders/shadertemplate_vk_upsampling2d_nearest.spv";
static constexpr const char* UPSAMPLING2D_BILINEAR_VK_ASSET_NAME  = "shaders/shadertemplate_vk_upsampling2d_bilinear.spv";
static constexpr const char* UPSAMPLING2D_NEAREST_VK_FP16_ASSET_NAME   = "shaders/shadertemplate_vk_upsampling2d_nearest_fp16.spv";
static constexpr const char* UPSAMPLING2D_BILINEAR_VK_FP16_ASSET_NAME  = "shaders/shadertemplate_vk_upsampling2d_bilinear_fp16.spv";

InferencePassesSptr UpSampling2DLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    std::string shaderName;
    if (_desc.interpolationType.compare("nearest") == 0) {
        if (_desc.preferHp) {
            shaderName = UPSAMPLING2D_NEAREST_VK_FP16_ASSET_NAME;
        } else {
            shaderName = UPSAMPLING2D_NEAREST_VK_ASSET_NAME;
        }
    } else if (_desc.interpolationType.compare("bilinear") == 0) {
        if (_desc.preferHp) {
            shaderName = UPSAMPLING2D_BILINEAR_VK_FP16_ASSET_NAME;
        } else {
            shaderName = UPSAMPLING2D_BILINEAR_VK_ASSET_NAME;
        }
    }

    SNN_LOGD("upsample: %s, %f", shaderName.c_str(), _desc.scale);

    std::vector<uvkc::vulkan::Pipeline::SpecConstant> specConstants {
        {0, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputHeight}},
        {1, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputWidth}},
        {2, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = inputDepth}},
    };

    std::pair<std::string, std::vector<uvkc::vulkan::Pipeline::SpecConstant>> reluParam("relu", specConstants);
    pass.specConstants = specConstants;
    pass.pushConstants.insert(reluParam);

    std::vector<uint32_t> uniform(18);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;
    uniform[2] = inputDepth;
    uniform[3] = 1;
    uniform[4] = outputWidth;
    uniform[5] = outputHeight;
    uniform[6] = outputDepth;
    uniform[7] = 1;

    std::array<float, 4> means{0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char* target = (unsigned char*)uniform.data() + 8 * sizeof(uint32_t);
    std::memcpy(target, means.data(), 4 * sizeof(uint32_t));
    std::array<float, 4> norms{1.0f, 1.0f, 1.0f, 1.0f};
    target = (unsigned char*)uniform.data() + 12 * sizeof(uint32_t);
    std::memcpy(target, norms.data(), 4 * sizeof(uint32_t));

    std::array<float, 2> scales{1.0f / _desc.scale, 1.0f / _desc.scale};
    target = (unsigned char*)uniform.data() + 16 * sizeof(uint32_t);
    std::memcpy(target, scales.data(), 2 * sizeof(uint32_t));

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    pass.inputs  = {{"uInput", 0}};
    auto bytes = snn::loadEmbeddedAsset(shaderName.c_str());
    pass.source = shaderName;
    pass.vkCodes.resize((bytes.size() + 3)/4);
    std::memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGD("input = %d:%d:%d, output = %d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
