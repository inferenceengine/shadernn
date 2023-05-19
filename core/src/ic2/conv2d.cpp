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
#include "conv2d.h"
#include "layerFactory.h"

using namespace snn;
using namespace snn::dp;

void Conv2DDesc::parse(ModelParser& parser, int layerId) {
    GenericConvDesc::parse(parser, layerId);
    parser.getConvolutionLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, activation, (int&) kernelSize, (int&) stride, biases, weightsCvM,
                               useBatchNormalization, batchNormalization, leakyReluAlpha, paddingT, paddingB, paddingL, paddingR, paddingMode,
                               useMultiInputs);
    SNN_LOGD("useBatchNormalization: %d, useMultiInputs: %d, isRange01: %d, numOutputPlanes: %d, numInputPlanes: %d, activation: %s, leakyReluAlpha: %.2f,"
             " kernelSize: %d, stride: %d, %s,\n\t"
             "useUniformShaders: %d, padding: %s, %s, %s, %s, paddingMode: %s ",
             useBatchNormalization, useMultiInputs, isRange01, numOutputPlanes, numInputPlanes, activation.c_str(), leakyReluAlpha, kernelSize, stride,
             preferHp ? "FP16" : "FP32", useUniformShaders, paddingT.c_str(), paddingB.c_str(), paddingL.c_str(), paddingR.c_str(), paddingMode.c_str());
}

void Conv2DLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    GenericModelLayer::getOutputDims(width, height, depth);
    depth = _desc.numOutputPlanes;
}

void Conv2DLayer::getPaddingOffset(uint32_t (&offsets)[4]) const {
    std::string paddingT = _desc.paddingT;
    std::string paddingB = _desc.paddingB;
    std::string paddingL = _desc.paddingL;
    std::string paddingR = _desc.paddingR;
    bool isdigit         = std::all_of(paddingT.begin(), paddingT.end(), ::isdigit);
    if (isdigit) {
        offsets[0] = std::stoul(paddingT);
        offsets[1] = std::stoul(paddingB);
        offsets[2] = std::stoul(paddingL);
        offsets[3] = std::stoul(paddingR);
    } else {
        if (paddingT == "valid" || paddingT == "none") {
            offsets[0] = 0;
            offsets[1] = 0;
            offsets[2] = 0;
            offsets[3] = 0;
        } else {
            if (_desc.kernelSize > 1) {
                offsets[0] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[1] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[2] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[3] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                if (_desc.kernelSize % 2 == 0) {
                    offsets[0] = offsets[0] - 1;
                    offsets[2] = offsets[2] - 1;
                }
            } else {
                offsets[0] = 0;
                offsets[1] = 0;
                offsets[2] = 0;
                offsets[3] = 0;
            }
        }
    }
}

bool Conv2DLayer::oihw2hwo4i4(const std::vector<cv::Mat>& inputWeights, std::vector<float>& outVec, int inChannels, int outChannels, int fw, int fh, int unit) {
    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh * ROUND_UP(inChannels, unit);

    SNN_LOGD("inChannels = %d, outChannels = %d, fw = %d, fh = %d, all: %d", inChannels, outChannels, fw, fh, alignedWeightSize);

    outVec.clear();
    outVec.resize(alignedWeightSize);
    float* out    = (float*) outVec.data();
    int planeSize = ROUND_UP(outChannels, unit) * ROUND_UP(inChannels, unit);
    memset(out, 0, alignedWeightSize * sizeof(float));
    for (int b = 0; b < outChannels; ++b) {
        int b_4 = b / unit;
        int mx  = b % unit;
        for (int d = 0; d < inChannels; ++d) {
            for (int y = 0; y < fh; ++y) {
                for (int x = 0; x < fw; ++x) {
                    int base                                 = (y * fw + x) * planeSize;
                    int inSize                               = ROUND_UP(inChannels, unit) * unit;
                    out[base + inSize * b_4 + d * unit + mx] = inputWeights[b * inChannels + d].at<float>(y * fw + x);
                }
            }
        }
    }
    return 0;
}

InferenceGraph::Transform Conv2DLayer::getOutputScaleDimAdjustment() const {
    uint32_t offset[4];
    getPaddingOffset(offset);
    float scale       = 1 / static_cast<float>(_desc.stride);
    float translation = 0.0f;
    if (_desc.kernelSize % 2 != 0) {
        translation = 1 + (static_cast<float>(offset[0] + offset[1]) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    } else {
        translation = 1 + (static_cast<float>(offset[0] + offset[1] - 1) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    }
    return {0, {{scale, scale, translation, translation}} };
}


