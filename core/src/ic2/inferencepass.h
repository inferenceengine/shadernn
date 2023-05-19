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

#include "snn/snn.h"
#include "snn/color.h"
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <optional>
#include <utility>

namespace snn {

// This structure holds information for one render pass.
struct InferencePass {
    // shader source code
    std::string source;

    // key is shader variable name. value is index into the layer's input buffer array.
    std::unordered_map<std::string, uint32_t> inputs; // input buffer uniforms.

    // Other uniforms. Key is shader variable name.
    std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> runtimeUniforms; // name and offset/size of runtime parameter
    std::vector<uint32_t> runtimeData; // run tcv::ime data in a period
    uint32_t period; // loop every period * 4 in runtime data
    uint32_t passId = 0; // nth of the render pass
    uint32_t totalPasses = 1; // total render passes needed for this operator

    uint32_t inputHeight, inputWidth, inputChannels;

    // Weights dimensions
    // TODO: remove map, because it contain only one element
    std::map<std::string, std::array<uint32_t, 3>> weightDims;

    // Weights, biases and other parameters,
    // used by some layers
    std::vector<float> _vecWeights;
    std::vector<float> _vecBias;
    std::vector<float> _vecMean;
    std::vector<float> _vecVariance;
    std::vector<float> _vecBeta;
    std::vector<float> _vecGamma;
};

struct InferencePasses {
protected:
    InferencePasses(GpuBackendType backendType_)
        : backendType(backendType_)
    {}

public:
    const GpuBackendType backendType;
};

typedef std::shared_ptr<InferencePasses> InferencePassesSptr;

}   // namespace snn
