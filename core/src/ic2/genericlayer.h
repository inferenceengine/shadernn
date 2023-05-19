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
#include "snn/inferencegraph.h"
#include "snn/layeroption.h"
#include "inferencepass.h"
#include "renderpass.h"
#include "modelparser.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

static std::vector<int> mLocalSize {4, 8, 4}; // For snapdragon 8Gen 2

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct CommonLayerDesc {
    bool isRange01; // determine expecting values in [0,1] range, or [-1,1] range.
    uint32_t numOutputPlanes;
    uint32_t numInputPlanes;
    uint32_t kernelSize = 0; // move here for layer name
    MRTMode mrtMode = MRTMode::NO;
    WeightAccessMethod weightMode = WeightAccessMethod::CONSTANTS;
    // The flag to use FP16 calculations
    bool preferHp = false;
    bool isInputLayer = false;
    // The index of this layer in all layers array
    uint32_t inputIndex = 0;
    void parse(ModelParser& parser, int layerId) {
        isRange01       = parser.isInputRange01();
        numOutputPlanes = (uint32_t) parser.getOutputPlanes(layerId);
        numInputPlanes  = (uint32_t) parser.getInputPlanes(layerId);
        preferHp        = parser.getPrecision();
        mrtMode         = parser.getMRTMode();
        weightMode      = parser.getWeightMode();
    }
};

// This class declares a layer of the inference processing graph at more detailed level.
class GenericModelLayer {
public:
    struct LayerGenOptions : ShaderGenOptions {
        uint32_t desiredOutputWidth, desiredOutputHeight;
        bool isFirstLayer = false;
        bool isLastLayer  = false;
    };

protected:
    std::string name; // for debugging only.
    CommonLayerDesc _desc;
    std::vector<InferenceGraph::IODesc> inputDims;
    // Collection of inference passes
    // It holds data for each render pass
    InferencePassesSptr passes;

private:
    // Collection of render passes
    std::vector<std::shared_ptr<RenderPass>> renderPasses;

public:
    std::vector<std::shared_ptr<GenericModelLayer>> prevLayers;
    std::vector<std::shared_ptr<GenericModelLayer>> nextLayers;

    GenericModelLayer(CommonLayerDesc d): _desc(d) {}

    GenericModelLayer(const GenericModelLayer&) = delete;

    GenericModelLayer& operator=(const GenericModelLayer&) = delete;

    virtual ~GenericModelLayer();

    const CommonLayerDesc& getDesc() const { return _desc; }

    const std::string& getName() { return name; }

    void setName(const std::string& genericName);

    const InferencePasses* getPasses() const {
        return passes.get();
    }

    std::vector<std::shared_ptr<RenderPass>>& getRenderPasses() {
        return renderPasses;
    }

    bool isInputLayer() { return _desc.isInputLayer; }

    uint32_t getInputIndex() { return _desc.inputIndex; }

    void addInputDim(const InferenceGraph::IODesc& dim) {
        inputDims.push_back(dim);
    }

    void setMRTMode(const MRTMode& mode) { _desc.mrtMode = mode; }

    void setWeightAccessMode(const WeightAccessMethod& mode) { _desc.weightMode = mode; }

    virtual void init(DeviceBackend* backend, ImageTextureArray& inputMat, ImageTextureArray& outputMat);

    // Run layer on GPU
    virtual void run(DeviceBackend* backend, bool dumpOutputs);

    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const;

    // Run layer on CPU
    virtual void computeImageTexture(ImageTextureArray& /*inputMat*/, ImageTextureArray& /*outputMat*/) {}

    virtual void createInferencePasses(const LayerGenOptions& options) = 0;

    virtual InferenceGraph::LayerExecutionType getLayerExecutionType() const = 0;

    virtual void setLayerExecutionType(InferenceGraph::LayerExecutionType newExecution) = 0;

    virtual bool isTransition() const { return false; }

private:
    // Defines output shape transformation
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const = 0;
};

struct GenericConvDesc : CommonLayerDesc {
    std::vector<cv::Mat> weightsCvM;
    std::vector<double> biases; // make it float?
    std::string activation;
    uint32_t kernelSize = 0;
    uint32_t stride     = 0;
    void parse(ModelParser& parser, int layerId) { CommonLayerDesc::parse(parser, layerId); }
};

// This class declares a layer that is implemented through GPU shader.
class ShaderLayer : public GenericModelLayer {
public:
    ShaderLayer(CommonLayerDesc d): GenericModelLayer(d) {}

    ShaderLayer(const ShaderLayer&) = delete;

    ShaderLayer& operator=(const ShaderLayer&) = delete;

    virtual ~ShaderLayer() = default;

    static std::string loadShader(const char* path);

    static void findAndReplace(std::string& s, const std::string& from, const std::string& to);

    virtual void createInferencePasses(const LayerGenOptions& options) override;

    virtual InferenceGraph::LayerExecutionType getLayerExecutionType() const override { return executeBackend; }

    virtual void setLayerExecutionType(InferenceGraph::LayerExecutionType newExecution) override { executeBackend = newExecution; }

private:
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override { return {0, {{1.0f, 1.0f, 0.0f, 0.0f}}}; }

    // Creates a collection of inference passes for Fragment shader
    virtual InferencePassesSptr createFS(const LayerGenOptions&) const = 0;

    // Creates a collection of inference passes for Compute shader
    virtual InferencePassesSptr createCS(const LayerGenOptions&) const = 0;

    InferenceGraph::LayerExecutionType executeBackend = InferenceGraph::LayerExecutionType::GPU_FS;
};

// This is a base class for all convolutional layers
class GenericConvolutionLayer : public ShaderLayer {
public:
    GenericConvolutionLayer(GenericConvDesc d): ShaderLayer(d), _desc(std::move(d)) {}
    ~GenericConvolutionLayer() {}

protected:
    struct WeightContants {
        std::string text;
        std::vector<glm::vec4> values;
    };

private:
    GenericConvDesc _desc;
};

} // namespace dp
} // namespace snn
