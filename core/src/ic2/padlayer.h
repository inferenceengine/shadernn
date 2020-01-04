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

struct PadDesc : GenericConvDesc {
    float constant            = 0.0f;
    bool preferrHalfPrecision = false;
    std::string paddingT, paddingB, paddingL, paddingR;
    std::string mode = "constant";
    void parse(ModelParser& parser, int layerId) {
        // std::cout << "Before parsing Pad "<<std::endl;
        GenericConvDesc::parse(parser, layerId);
        parser.getPaddingLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, (std::string&) paddingT, (std::string&) paddingB,
                               (std::string&) paddingL, (std::string&) paddingR, (std::string&) mode, (float&) constant);
    }
};

class PadLayer : public GenericConvolutionLayer {
public:
    PadLayer(PadDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    ~PadLayer() {}
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    virtual GLSLShaders createFS(const LayerGenOptions&) const override;
    virtual GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    PadDesc _desc;

    // Adds predefine to given stringstream.
    // shaderFilePath is the path of the file template this will be appended to
    // and is used to label the preDefine.

    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;
};

}; // namespace dp
} // namespace snn
