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
#include <vector>
#include <map>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct InstanceNormDesc : GenericConvDesc {
    bool useInstanceNormalization;
    bool useMultiInputs;
    std::map<std::string, std::vector<float>> instanceNormalization;
    float leakyReluAlpha;
    std::string padding;
    bool useUniformShaders    = true;
    float epsilon;
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getInstanceNormalizationLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, epsilon, instanceNormalization, activation,
                                             leakyReluAlpha);
    }
};

// This is a base class to generates a shader for an instance normalization
class InstanceNormLayer : public GenericConvolutionLayer {
public:
    InstanceNormLayer(InstanceNormDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    virtual ~InstanceNormLayer() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override {
        return {0, {{1.0f, 1.0f, 0.0f, 0.0f}} };
    }

protected:
    InstanceNormDesc _desc;
};

}; // namespace dp
} // namespace snn
