/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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

#include <snn/snn.h>
#include <snn/utils.h>
#include "inferencegraph.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline

// Dummy parent struct to handle both PCU and GPU layer generation options

// TODO: not all options are useful to createFS() and createCS(). Maybe split this into several sub-classes.
struct ShaderGenOptions {
    // Specify desired input buffer dimension and format.
    // format can be set to NONE, if we don't care.
    InferenceGraph::Buffer desiredInput;

    // Define desired output buffer format.
    // output buffer dimension is determined by input dimension and
    // the shader graph properties. So they are not specified here.
    ColorFormat desiredOutputFormat;

    // Set to true to generate compute shaders
    bool compute = false;

    bool preferrHalfPrecision = false; // prefer 16-bit float when set to true. Otherwise, 32-bit float.

    bool ssbo = false; // Set to true to store weights in SSBO.
    snn::MRTMode mrtMode;
    snn::WeightAccessMethod weightMode;
};

struct CpuGenOptions {
    uint32_t inputDims, outputDims;
    bool isFirstLayer      = false;
    bool isLastLayer       = false;
    bool isTransitionLayer = false;
};

}; // namespace dp
} // namespace snn
