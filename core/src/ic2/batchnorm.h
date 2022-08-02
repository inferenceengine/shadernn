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

struct BatchNormalizationDesc : GenericConvDesc {
    bool preferrHalfPrecision = false;
    float leakyReluAlpha;
    std::map<std::string, std::vector<float>> batchNormalization;
    void parse(ModelParser& parser, int layerId) {
        std::cout << "Before parsing BatchNorm " << std::endl;
        GenericConvDesc::parse(parser, layerId);
        parser.getBatchNormLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, batchNormalization, activation, leakyReluAlpha);
    }
};

class BatchNormalizationLayer : public GenericConvolutionLayer {
public:
    BatchNormalizationLayer(BatchNormalizationDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    ~BatchNormalizationLayer() {}
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    virtual GLSLShaders createFS(const LayerGenOptions&) const override;
    virtual GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    BatchNormalizationDesc _desc;

    void getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass, const int outputChannels) const;

    // Adds predefine to given stringstream.
    // shaderFilePath is the path of the file template this will be appended to
    // and is used to label the preDefine.
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;
};

}; // namespace dp
} // namespace snn
