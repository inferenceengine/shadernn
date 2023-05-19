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
#include "denselayer.h"
#include "cpulayer.h"
#include "layerFactory.h"
#include "inferencepass.h"
#include <string>
#include <vector>
#include <utility>

using namespace snn;
using namespace snn::dp;

void DenseLayer::computeImageTexture(snn::ImageTextureArray& inputTex, snn::ImageTextureArray& outputTex) {
    auto inputMat = inputTex[0].getOutputMat();

    auto cpuL = CPUCommonUtil<float> {_desc.activation, _desc.leakyReluAlpha, true};

    auto transformMats = std::pair<std::vector<std::vector<float>>, std::vector<float>>(_desc.weights, _desc.biases);
    if (!cpuL.inputMat.has_value()) {
        cpuL.inputMat.emplace(inputMat);
    }
    cpuL.run(transformMats);
    outputTex[0].setOutputMat(cpuL.getOutputs());
}

InferenceGraph::Transform DenseLayer::getOutputScaleDimAdjustment() const {
    InferenceGraph::Transform ret;
    ret.isFixed     = 1;
    ret.fixedWidth  = (uint32_t) _desc.biases.size();
    ret.fixedHeight = 1;
    ret.fixedDepth  = 1;
    ret.fixedBatch  = 1;
    return ret;
}

void DenseLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    width  = (uint32_t) _desc.biases.size();
    height = 1;
    depth  = 1;
}
