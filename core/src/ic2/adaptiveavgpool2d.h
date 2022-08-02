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

struct AdaptiveAvgPool2dDesc : GenericConvDesc {
    std::string padding;
    int targetSize = 1;
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getAdaptiveAvgPoolLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, (int&) targetSize);
    }
};

class AdaptiveAvgPool2dLayer : public GenericConvolutionLayer {
public:
    AdaptiveAvgPool2dLayer(AdaptiveAvgPool2dDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    ~AdaptiveAvgPool2dLayer() {}
    InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

    virtual void setStride(uint32_t inputSize) override { this->_desc.stride = inputSize / _desc.targetSize; }

protected:
    GLSLShaders createFS(const LayerGenOptions&) const override;
    GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    AdaptiveAvgPool2dDesc _desc;
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;

    void buildTextureDefLogic(std::ostream& stream, const LayerGenOptions&, uint32_t inputSliceIndex) const;

    void buildCalcDefLogic(std::ostream& stream, const LayerGenOptions& options) const;

    void buildComputePostDefine(std::ostream& stream, uint32_t outputSliceIndex) const;

    void buildFragPostDefine(std::ostream& stream) const;
};

}; // namespace dp
} // namespace snn
