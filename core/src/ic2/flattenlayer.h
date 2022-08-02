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
#include "denselayer.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct FlattenDesc : DenseDesc {
    void parse(ModelParser& parser, int layerId) { parser.getFlattenLayer(layerId, (int&) numInputPlanes, (int&) numOutputPlanes, (std::string&) activation); }
};

class FlattenLayer : public ShaderLayer {
public:
    FlattenLayer(FlattenDesc&& d): ShaderLayer(d), _flattenDesc(std::move(d)) {}
    FlattenLayer(const FlattenLayer& d) = delete;
    FlattenLayer& operator=(const FlattenLayer& d) = delete;
    ~FlattenLayer() {}
    // virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override { return {1.0f, 0.0f}; };
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;
    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override;
    // virtual void computeImage(std::vector<std::shared_ptr<snn::ManagedRawImage>>& inputMat, std::vector<std::vector<float>>& outputMat) override;
    virtual void computeImageTexture(FixedSizeArray<snn::ImageTexture>& inputMat, FixedSizeArray<snn::ImageTexture>& outputMat) override;
    virtual GLSLShaders createGLSLShader(const LayerGenOptions& options) override;
    // virtual GLSLShaders createGLSLShader(const LayerGenOptions & options) const override {
    //     // auto dummyOptions = options;
    //     (void) options;
    //     return GLSLShaders();
    // };
    virtual snn::InferenceGraph::LayerExecution getLayerExecutionLevel() const override { return executeBackend; }
    virtual void setLayerExecutionLevel(snn::InferenceGraph::LayerExecution newExecution) override { executeBackend = newExecution; }

    virtual CPUPasses createCPUPasses(const CpuGenOptions& options) const override {
        // auto dummyOptions = options;
        CPUPasses pass;
        pass.passes.program      = InferenceGraph::Pass::CPUProgram<float> {options.isTransitionLayer, options.isLastLayer};
        pass.passes.transformMat = std::pair<std::vector<std::vector<float>>, std::vector<float>>(std::vector<std::vector<float>>(), std::vector<float>());
        // pass.passes.transformationFunc = [&](
        //     std::shared_ptr<snn::dp::CommonLayerDesc> layerDesc,
        //     std::vector<std::vector<float>> inputMat
        // ) ->std::vector<std::vector<float>> { return this->transformFunc(layerDesc, inputMat); };
        return pass;
    };

    std::vector<std::vector<float>> transformFunc(std::shared_ptr<snn::dp::CommonLayerDesc> layerDesc, std::vector<std::vector<float>> inputMat = {});

    virtual bool isTransition() const override { return true; }

private:
    FlattenDesc _flattenDesc;
    snn::InferenceGraph::LayerExecution executeBackend = snn::InferenceGraph::LayerExecution::CPU;
};
}; // namespace dp
} // namespace snn
