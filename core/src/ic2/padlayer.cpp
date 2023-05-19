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
#include "padlayer.h"
#include "layerFactory.h"
#include "inferencepass.h"
#include <string>
#include <algorithm>
#include <utility>

using namespace snn;
using namespace snn::dp;

void PadLayer::getPaddingOffset(uint32_t (&offsets)[4]) const {
    std::string paddingT = this->_desc.paddingT;
    std::string paddingB = this->_desc.paddingB;
    std::string paddingL = this->_desc.paddingL;
    std::string paddingR = this->_desc.paddingR;
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
            } else {
                offsets[0] = 0;
                offsets[1] = 0;
                offsets[2] = 0;
                offsets[3] = 0;
            }
        }
    }
}

InferenceGraph::Transform PadLayer::getOutputScaleDimAdjustment() const {
    uint32_t offset[4];
    getPaddingOffset(offset);
    float scale        = 1;
    float translation1 = 0.0f, translation2 = 0.0f;
    translation1 = static_cast<float>(offset[2] + offset[3]);
    translation2 = static_cast<float>(offset[0] + offset[1]);
    return {0, {{scale, scale, translation1, translation2}} };
}
