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

struct UpSampling2DDesc : CommonLayerDesc {
    float scale;
    string interpolationType;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        scale             = parser.getUpSamplingScale(layerId);
        interpolationType = parser.getUpSampling2DInterpolation(layerId);
    }
};

class UpSampling2DLayer : public ShaderLayer {
public:
    UpSampling2DLayer(UpSampling2DDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    ~UpSampling2DLayer() {}
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    bool generateUpSampling2DGLSamplingCode(int& idxStartPlane, int nOutputChannels, std::string& uniformsDeclaration, std::string& calculation,
                                            const bool& compute) const;
    virtual GLSLShaders createFS(const LayerGenOptions&) const override;
    virtual GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    UpSampling2DDesc _desc;
};

}; // namespace dp
} // namespace snn
