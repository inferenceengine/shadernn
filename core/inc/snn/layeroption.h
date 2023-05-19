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

#include <snn/snn.h>
#include <snn/utils.h>
#include "snn/inferencegraph.h"
#include <vector>

namespace snn {
namespace dp { // short for Dynamic Pipeline

// This structure holds layer generation options
// TODO: not all options are useful to createFS() and createCS(). Maybe split this into several sub-classes.
struct ShaderGenOptions {
    // Specifies desired input buffer dimension and format.
    // Format can be set to NONE, if we don't care.
    std::vector<InferenceGraph::IODesc> desiredInput;

    // Defines desired output buffer format.
    // Output buffer dimension is determined by input dimension and
    // the shader graph properties. So they are not specified here.
    ColorFormat desiredOutputFormat;

    // Set to true to generate compute shaders
    bool compute = false;

    // Set to true to generate Vulkan shaders
    bool vulkan = false;

    bool preferrHalfPrecision = false; // prefer 16-bit float when set to true. Otherwise, 32-bit float.

    bool ssbo = false; // Set to true to store weights in SSBO.
    MRTMode mrtMode; // MRT (Multiple Render Target) mode
    WeightAccessMethod weightMode; // Weights access mode
};

}; // namespace dp
} // namespace snn
