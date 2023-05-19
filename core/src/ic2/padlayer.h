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

#include "genericlayer.h"
#include "snn/snn.h"
#include "modelparser.h"
#include <string>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct PadDesc : GenericConvDesc {
    float constant            = 0.0f;
    std::string paddingT, paddingB, paddingL, paddingR;
    std::string mode = "constant";
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getPaddingLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, (std::string&) paddingT, (std::string&) paddingB,
                               (std::string&) paddingL, (std::string&) paddingR, (std::string&) mode, (float&) constant);
    }
};

// This is a base class to generates a shader for padding
class PadLayer : public GenericConvolutionLayer {
public:
    PadLayer(PadDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    virtual ~PadLayer() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    PadDesc _desc;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;
};

}; // namespace dp
} // namespace snn
