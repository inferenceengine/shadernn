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
#include "genericlayer.h"
#include <string>
#include <vector>
#include <memory>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

// Parses the model JSON file and produces the model layers objects
// params:
//  fileName - JSON file path
//  useVulkan - flag to generate graph for Vulkan platform
//  mrtMode - MRT (multi rendering target) mode
//  weightMode - weight access mode
//  preferHp - flag to generate graph for FP16 calculations
// returns:
//  vector of shared pointers to model layers objects
std::vector<std::shared_ptr<GenericModelLayer>> loadFromJsonModel(const std::string& fileName, bool useVulkan, const MRTMode& mrtMode,
                                                                  const WeightAccessMethod& weightMode, bool preferHp = true);

typedef std::vector<std::shared_ptr<GenericModelLayer>> InferenceModel;

// Generate an inference graph with single input
// params:
//  firstLayer - first layer
//  options - shader generating options
// returns:
//  an inference graph
InferenceGraph generateInferenceGraph(const std::shared_ptr<GenericModelLayer> firstLayer, const ShaderGenOptions& options);

// Generate an inference graph with multiple inputs
// params:
//  layers - collection of input layers
//  options - shader generating options
// returns:
//  an inference graph
InferenceGraph generateInferenceGraph(std::vector<std::shared_ptr<GenericModelLayer>> &layers, const ShaderGenOptions& options);

}; // namespace dp
} // namespace snn
