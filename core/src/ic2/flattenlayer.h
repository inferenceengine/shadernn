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
#include "denselayer.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct FlattenDesc : CommonLayerDesc {
    std::string activation;
    float leakyReluAlpha;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        parser.getFlattenLayer(layerId, (int&) numInputPlanes, (int&) numOutputPlanes, (std::string&) activation);
    }
};

class FlattenLayer : public ShaderLayer {
public:
    FlattenLayer(FlattenDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    FlattenLayer(const FlattenLayer& d) = delete;
    FlattenLayer& operator=(const FlattenLayer& d) = delete;
    virtual ~FlattenLayer() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;
    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override;
    virtual void computeImageTexture(ImageTextureArray& inputMat, ImageTextureArray& outputMat) override;

    virtual snn::InferenceGraph::LayerExecutionType getLayerExecutionType() const override { return layerExecutionType; }
    virtual void setLayerExecutionType(snn::InferenceGraph::LayerExecutionType layerExecutionType_) override { layerExecutionType = layerExecutionType_; }

    virtual bool isTransition() const override { return true; }

protected:
    FlattenDesc _desc;

private:
    snn::InferenceGraph::LayerExecutionType layerExecutionType = snn::InferenceGraph::LayerExecutionType::CPU;
};
}; // namespace dp
} // namespace snn
