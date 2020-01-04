/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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

#include <snn/snn.h>
#include <snn/utils.h>
#include <snn/imageTexture.h>
#include "inferencegraph.h"
#include "modelparser.h"
#include <utility>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>

#include "genericlayer.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct AddDesc : public CommonLayerDesc {
    std::string activation;
    float leakyReluAlpha;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        parser.getAddLayer(layerId, activation, leakyReluAlpha);
    }
};

class AddLayer : public ShaderLayer {
public:
    AddLayer(AddDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    ~AddLayer() {}
    InferenceGraph::Transform getOutputScaleDimAdjustment() const override { return {0, {{1.0f, 1.0f, 0.0f, 0.0f}}}; };

protected:
    GLSLShaders createFS(const LayerGenOptions&) const override;
    GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    AddDesc _desc;
};

}; // namespace dp
} // namespace snn
