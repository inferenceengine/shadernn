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
#include "flattenlayer.h"
#include "snn/imageTexture.h"
#include "cpulayer.h"
#include "layerFactory.h"
#include "inferencepass.h"
#include <string>
#include <vector>
#include <memory>
#include <utility>

using namespace snn;
using namespace snn::dp;

void snn::dp::FlattenLayer::computeImageTexture(snn::ImageTextureArray& inputTex, snn::ImageTextureArray& outputTex) {
    std::shared_ptr<snn::RawImage> outputTexPtr;

    auto image = inputTex[0].getRawImage();
    SNN_ASSERT(!image.empty());
    outputTexPtr = std::make_shared<snn::RawImage>(inputTex[0].getRawImage());

    std::vector<std::shared_ptr<snn::RawImage>> inputMat{outputTexPtr};

    auto cpuL         = snn::dp::CPUCommonUtil<float> {_desc.activation, _desc.leakyReluAlpha, false};
    auto transformMat = std::pair<std::vector<std::vector<float>>, std::vector<float>>(std::vector<std::vector<float>>(), std::vector<float>());

    if (!cpuL.gpuTexMat.has_value()) {
        cpuL.gpuTexMat.emplace(inputMat);
    }
    cpuL.run(transformMat);
    outputTex[0].setOutputMat(cpuL.getOutputs());
}

InferenceGraph::Transform FlattenLayer::getOutputScaleDimAdjustment() const {
    InferenceGraph::Transform ret;
    ret.isFixed     = 1;
    ret.fixedWidth  = inputDims[0].width * inputDims[0].height * _desc.numInputPlanes;
    ret.fixedHeight = 1;
    ret.fixedDepth  = 1;
    ret.fixedBatch  = 1;
    return ret;
}

void FlattenLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    width  = inputDims[0].width * inputDims[0].height * _desc.numInputPlanes;
    height = 1;
    depth  = 1;
}
