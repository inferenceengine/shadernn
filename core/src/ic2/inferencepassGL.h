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
#include "glUtils.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#define MIN_SSBO_BUFFER_LEN_ARM_MALI 4

namespace snn {

// This structure holds information for one OpenGL render pass.
struct InferencePassGl : public InferencePass {
    struct FsProgram {
        uint32_t outputSliceIndex;
        uint32_t outputSliceCount;
    };
    struct CsProgram {
        // compute shader always bind the texture object as a whole. So there's no need to specify output slice index and count.
        std::string outputImageUniform;
        uint32_t dispatchSize[3];
    };

    // Fragment shader or Compute shader program
    std::variant<FsProgram, CsProgram> program;

    std::vector<cv::Mat> modelWeights;
    std::vector<uint32_t> weightMeta;

    // Other uniforms. Key is shader variable name.
    std::unordered_map<std::string, gl::SimpleUniform::Value> uniforms;

    std::vector<glm::vec4> weightMatrices[4];
};

struct InferencePassesGl : public InferencePasses {
    InferencePassesGl() : InferencePasses(GpuBackendType::GL) {}

    static InferencePassesGl* cast(InferencePasses* iPasses) {
        SNN_ASSERT(iPasses->backendType == GpuBackendType::GL);
        return static_cast<InferencePassesGl*>(iPasses);
    }

    static const InferencePassesGl* cast(const InferencePasses* iPasses) {
        SNN_ASSERT(iPasses->backendType == GpuBackendType::GL);
        return static_cast<const InferencePassesGl*>(iPasses);
    }

    std::vector<InferencePassGl> passes;
};

}   // namespace snn
