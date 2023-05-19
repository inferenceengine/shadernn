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
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct DenseDesc : CommonLayerDesc {
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::string activation;
    int numInputUnits, numOutputUnits;
    float leakyReluAlpha;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        parser.getDenseLayer(layerId, numOutputUnits, numInputUnits, activation, weights, biases, leakyReluAlpha);
    }
};

// This is a base class to generates a shader for a fully connected layer
class DenseLayer : public ShaderLayer {
public:
    DenseLayer(DenseDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    DenseLayer(const DenseLayer& d) = delete;
    DenseLayer& operator=(const DenseLayer& d) = delete;
    virtual ~DenseLayer() = default;

    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;
    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override;

    virtual void computeImageTexture(ImageTextureArray& inputMat, ImageTextureArray& outputMat) override;

    virtual snn::InferenceGraph::LayerExecutionType getLayerExecutionType() const override { return executeBackend; }
    virtual void setLayerExecutionType(InferenceGraph::LayerExecutionType newExecution) override { executeBackend = newExecution; }

protected:
    DenseDesc _desc;

private:
    snn::InferenceGraph::LayerExecutionType executeBackend = InferenceGraph::LayerExecutionType::CPU;
};

}; // namespace dp
} // namespace snn
