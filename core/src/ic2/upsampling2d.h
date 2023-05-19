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

struct UpSampling2DDesc : CommonLayerDesc {
    float scale;
    std::string interpolationType;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        scale             = parser.getUpSamplingScale(layerId);
        interpolationType = parser.getUpSampling2DInterpolation(layerId);
    }
};

class UpSampling2DLayer : public ShaderLayer {
public:
    UpSampling2DLayer(UpSampling2DDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    virtual ~UpSampling2DLayer() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override {
        return {0, {{ static_cast<float>(_desc.scale), static_cast<float>(_desc.scale), 0.0f, 0.0f}}
    };
}

protected:
    UpSampling2DDesc _desc;
};

} // namespace dp
} // namespace snn
