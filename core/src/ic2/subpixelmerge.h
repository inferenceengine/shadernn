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
#include <vector>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct SubpixelDesc : CommonLayerDesc {
    uint32_t kernelSize;
    std::vector<double> biases; // make it float?
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        kernelSize = 2;
    }
};

// This is a base class to generates a shader for subpixel convolutional layer for ESPCN model
// https://medium.com/@zhuocen93/an-overview-of-espcn-an-efficient-sub-pixel-convolutional-neural-network-b76d0a6c875e
class SubpixelLayer : public ShaderLayer {
public:
    SubpixelLayer(SubpixelDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    virtual ~SubpixelLayer() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override {
        return {0, {{ static_cast<float>(_desc.kernelSize), static_cast<float>(_desc.kernelSize), 0.0f, 0.0f}} };
    }

protected:
    SubpixelDesc _desc;
};

} // namespace dp
} // namespace snn
