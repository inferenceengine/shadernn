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

struct MaxPooling2DDesc : GenericConvDesc {
    std::string padding, paddingValue, paddingT, paddingB, paddingL, paddingR;
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getMaxPoolLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, (int&) kernelSize, (int&) stride, padding, paddingValue, paddingT,
                               paddingB, paddingL, paddingR);
    }
};

// This is a base class to generates a shader for maxpooling
class MaxPooling2DLayer : public GenericConvolutionLayer {
public:
    MaxPooling2DLayer(MaxPooling2DDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    virtual ~MaxPooling2DLayer() = default;
    InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    MaxPooling2DDesc _desc;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;
};

}; // namespace dp
} // namespace snn
