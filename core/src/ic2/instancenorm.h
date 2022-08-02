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

struct InstanceNormDesc : GenericConvDesc {
    bool useInstanceNormalization;
    bool useMultiInputs;
    std::map<std::string, std::vector<float>> instanceNormalization;
    float leakyReluAlpha;
    std::string padding;
    bool useUniformShaders    = true;
    bool preferrHalfPrecision = false;
    float epsilon;
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getInstanceNormalizationLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, epsilon, instanceNormalization, activation,
                                             leakyReluAlpha);
    }
};

class InstanceNormLayer : public GenericConvolutionLayer {
public:
    InstanceNormLayer(InstanceNormDesc&& d);
    ~InstanceNormLayer() {}
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    virtual GLSLShaders createFS(const LayerGenOptions&) const override;
    virtual GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    void getPaddingOffset(uint32_t (&offsets)[4]) const;
    InstanceNormDesc _desc;
    snn::FixedSizeArray<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> weightSSBOBuffers;
    std::vector<double> biases;
};

}; // namespace dp
} // namespace snn
