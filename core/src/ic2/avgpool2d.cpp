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
#include "avgpool2d.h"

using namespace snn;
using namespace snn::dp;

InferenceGraph::Transform AveragePooling2DLayer::getOutputScaleDimAdjustment() const {
    float scale, translation;
    scale = 1.0f / _desc.stride;
    if (_desc.padding == "0" || _desc.padding == "none" || _desc.padding == "valid") {
        translation = 1.0f - ((float)_desc.kernelSize / (float)_desc.stride);
    } else {
        translation = 1.0f - 1.0f / (float)_desc.stride;
    }
    return {0, {{scale, scale, translation, translation}} };
}

void AveragePooling2DLayer::getPaddingOffsetOrig(uint32_t (&offsets)[4], const std::string& paddingT, const std::string& paddingB, const std::string& paddingL,
    const std::string& paddingR, int kernelSize) {
    bool isdigit = std::all_of(paddingT.begin(), paddingT.end(), ::isdigit);
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
            if (kernelSize > 1) {
                offsets[0] = std::max(static_cast<uint32_t>(kernelSize / 2), (uint32_t) 1);
                offsets[1] = std::max(static_cast<uint32_t>(kernelSize / 2), (uint32_t) 1);
                offsets[2] = std::max(static_cast<uint32_t>(kernelSize / 2), (uint32_t) 1);
                offsets[3] = std::max(static_cast<uint32_t>(kernelSize / 2), (uint32_t) 1);
                if (kernelSize % 2 == 0) {
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
