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
struct YOLODesc : CommonLayerDesc {
    void parse(ModelParser& parser, int layerId);
};

class YOLOLayer : public GenericModelLayer {
public:
    YOLOLayer(YOLODesc&& d): GenericModelLayer(d), _yoloDesc(std::move(d)) {}
    YOLOLayer(const YOLOLayer& d) = delete;
    YOLOLayer& operator=(const YOLOLayer& d) = delete;
    ~YOLOLayer() {}

    InferenceGraph::Transform getOutputScaleDimAdjustment() const override { return {0, {{1.0f, 1.0f, 0.0f, 0.0f}}}; };

    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override {
        width  = 100 * 6; // Max 100 bounding box
        height = 1;
        depth  = 1;
    }

    virtual void computeImageTexture(FixedSizeArray<snn::ImageTexture>& inputMat, FixedSizeArray<snn::ImageTexture>& outputMat) override;

    virtual snn::InferenceGraph::LayerExecution getLayerExecutionLevel() const override { return executeBackend; }
    virtual void setLayerExecutionLevel(snn::InferenceGraph::LayerExecution newExecution) override { executeBackend = newExecution; }

    virtual GLSLShaders createGLSLShader(const LayerGenOptions& options) override {
        (void) options;
        return GLSLShaders();
    };

    virtual CPUPasses createCPUPasses(const CpuGenOptions& options) const override {
        // auto dummyOptions = options;
        CPUPasses pass;
        pass.passes.program      = InferenceGraph::Pass::CPUProgram<float> {options.isTransitionLayer, options.isLastLayer};
        pass.passes.transformMat = std::pair<std::vector<std::vector<float>>, std::vector<float>>(std::vector<std::vector<float>>(), std::vector<float>());
        return pass;
    };

    virtual bool isTransition() const override { return true; }

private:
    YOLODesc _yoloDesc;
    snn::InferenceGraph::LayerExecution executeBackend = snn::InferenceGraph::LayerExecution::CPU;
};

}; // namespace dp
} // namespace snn
