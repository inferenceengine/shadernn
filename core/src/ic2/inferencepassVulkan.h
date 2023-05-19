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
#pragma once

#include "inferencepass.h"
#include "snn/color.h"
#include "uvkc/benchmark/vulkan_context.h"
#include <string>
#include <vector>
#include <map>

namespace snn {

// This structure holds information for one Vulkan render pass.
struct InferencePassVulkan : public InferencePass {
    struct VkProgram {
        std::string outputImageUniform;
        uint32_t dispatchSize[3];
    };

    VkProgram program;

    // Vulkan SPIR-V codes
    std::vector<uint32_t> vkCodes;
    std::vector<uvkc::vulkan::Pipeline::SpecConstant> specConstants;
    std::map<std::string, std::vector<uvkc::vulkan::Pipeline::SpecConstant>> pushConstants;
    std::map<std::string, std::vector<uint32_t>> uniformBuffers;
    std::map<std::string, std::vector<float>> objectBuffers;
    std::map<std::string, std::vector<float>> weightBuffers;
    std::map<std::string, snn::ColorFormat> weightFormats;
};

struct InferencePassesVulkan : public InferencePasses {
    InferencePassesVulkan() : InferencePasses(GpuBackendType::VULKAN) {}

    static InferencePassesVulkan* cast(InferencePasses* iPasses) {
        SNN_ASSERT(iPasses->backendType == GpuBackendType::VULKAN);
        return static_cast<InferencePassesVulkan*>(iPasses);
    }

    static const InferencePassesVulkan* cast(const InferencePasses* iPasses) {
        SNN_ASSERT(iPasses->backendType == GpuBackendType::VULKAN);
        return static_cast<const InferencePassesVulkan*>(iPasses);
    }

    std::vector<InferencePassVulkan> passes;
};

}   // namespace snn
