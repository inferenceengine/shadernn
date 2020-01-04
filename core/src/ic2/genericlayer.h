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
#include "layeroption.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct CommonLayerDesc {
    bool isRange01; // determine expecting values in [0,1] range, or [-1,1] range.
    uint32_t numOutputPlanes;
    uint32_t numInputPlanes;
    uint32_t kernelSize = 0; // move here for layer name
    snn::MRTMode mrtMode;
    snn::WeightAccessMethod weightMode;
    bool preferHp = false;
    void parse(ModelParser& parser, int layerId) {
        isRange01       = parser.isInputRange01();
        numOutputPlanes = (uint32_t) parser.getOutputPlanes(layerId);
        numInputPlanes  = (uint32_t) parser.getInputPlanes(layerId);
        preferHp        = parser.getPrecision();
        mrtMode         = parser.getMRTMode();
        weightMode      = parser.getWeightMode();
    }
};

class GenericModelLayer {
public:
    std::vector<std::shared_ptr<GenericModelLayer>> prevLayers;
    std::vector<std::shared_ptr<GenericModelLayer>> nextLayers;
    std::string name; // optinoal. for debugging only.
    std::vector<InferenceGraph::Buffer> inputDims;

    GenericModelLayer(CommonLayerDesc d): _desc(d) {}
    GenericModelLayer(const GenericModelLayer&) = delete;
    GenericModelLayer& operator=(const GenericModelLayer&) = delete;
    ~GenericModelLayer() {
        prevLayers.clear();
        nextLayers.clear();
    }
    virtual void setStride(uint32_t inputSize) { (void) inputSize; }
    const CommonLayerDesc& getDesc() const { return _desc; }
    int setName(std::string genericName) {
        this->name = genericName;
        return 0;
    }
    std::string getName() { return this->name; }
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const = 0;
    virtual void getInputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
        (void) width, (void) height, (void) depth;
        return;
    }
    virtual InferenceGraph::Buffer getInputDims(const std::size_t idx = 0) const {
        InferenceGraph::Buffer dims;
        try {
            dims = this->inputDims.at(idx);
        } catch (std::exception& e) {
            SNN_LOGE("Cannot find the dims at index %d", idx);
            throw std::out_of_range("IDX too large");
        }
        return dims;
    }

    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
        InferenceGraph::Transform accumulatedTransform = {0, {{0.0f, 0.0f, 0.0f, 0.0f}}};
        auto t                                         = getOutputScaleDimAdjustment();
        for (auto& dim : inputDims) {
            SNN_LOGD("%%%%%%%% %s:%d dim:%d, %d, %d\n", __FUNCTION__, __LINE__, dim.width, dim.height, dim.depth);
            if (!t.isFixed) {
                accumulatedTransform.scaleWidth      = std::max(accumulatedTransform.scaleWidth, t.scaleWidth * dim.width);
                accumulatedTransform.translateWidth  = std::max(accumulatedTransform.translateWidth, t.translateWidth);
                accumulatedTransform.scaleHeight     = std::max(accumulatedTransform.scaleHeight, t.scaleHeight * dim.height);
                accumulatedTransform.translateHeight = std::max(accumulatedTransform.translateHeight, t.translateHeight);
                width                                = accumulatedTransform.scaleWidth + accumulatedTransform.translateWidth;
                height                               = accumulatedTransform.scaleHeight + accumulatedTransform.translateHeight;
                depth                                = std::max(depth, dim.depth);
                SNN_LOGD("%%%%%%%% %s:%d dim:%d, %d, %d\n", __FUNCTION__, __LINE__, width, height, depth);
            } else {
                accumulatedTransform.fixedWidth  = std::max(accumulatedTransform.fixedWidth, t.fixedWidth);
                accumulatedTransform.fixedHeight = std::max(accumulatedTransform.fixedHeight, t.fixedHeight);
                accumulatedTransform.fixedDepth  = std::max(accumulatedTransform.fixedDepth, t.fixedDepth);
                accumulatedTransform.fixedBatch  = std::max(accumulatedTransform.fixedBatch, t.fixedBatch);
                width                            = accumulatedTransform.fixedWidth + accumulatedTransform.translateWidth;
                height                           = accumulatedTransform.fixedHeight + accumulatedTransform.translateHeight;
                depth                            = std::max(depth, dim.depth);
                SNN_LOGD("%%%%%%%% %s:%d dim:%d, %d, %d\n", __FUNCTION__, __LINE__, width, height, depth);
            }
        }
    }

    virtual bool isTransition() const { return false; }

    bool addInputDim(InferenceGraph::Buffer& dim) {
        inputDims.push_back(dim);
        return 0;
    }

    virtual void computeImageTexture(FixedSizeArray<snn::ImageTexture>& inputMat, FixedSizeArray<snn::ImageTexture>& outputMat) {
        (void) inputMat;
        (void) outputMat;
        SNN_LOGI("%%%%%%%% %s:%d :%s\n", __FUNCTION__, __LINE__, name.c_str());
    }

    void setMRTMode(const snn::MRTMode mode) { this->_desc.mrtMode = mode; }

    void setWeightAccessMode(const snn::WeightAccessMethod mode) { this->_desc.weightMode = mode; }

    CommonLayerDesc _desc;

    struct GLSLShaders {
        std::vector<InferenceGraph::Pass> passes;
        snn::FixedSizeArray<gl::TextureObject> weights;
        SNN_DEFAULT_MOVE(GLSLShaders);
        GLSLShaders() = default;
        operator bool() const {
            if (passes.empty()) {
                return false;
            } else {
                return true;
            }
        }
    };

    struct CPUPasses {
        InferenceGraph::Pass passes;
        operator bool() const {
            if (passes.transformMat) {
                return true;
            } else {
                return false;
            }
        }
    };

    struct LayerGenOptions : ShaderGenOptions {
        uint32_t desiredOutputWidth, desiredOutputHeight;
        bool isFirstLayer = false;
        bool isLastLayer  = false;
    };

    virtual GLSLShaders createGLSLShader(const LayerGenOptions& options)                  = 0;
    virtual CPUPasses createCPUPasses(const CpuGenOptions& options) const                 = 0;
    virtual snn::InferenceGraph::LayerExecution getLayerExecutionLevel() const            = 0;
    virtual void setLayerExecutionLevel(snn::InferenceGraph::LayerExecution newExecution) = 0;
};

struct GenericConvDesc : CommonLayerDesc {
    std::vector<cv::Mat> weights;
    std::vector<double> biases; // make it float?
    std::string activation;
    uint32_t kernelSize = 0;
    uint32_t stride     = 0;
    void parse(ModelParser& parser, int layerId) { CommonLayerDesc::parse(parser, layerId); }
};

class ShaderLayer : public GenericModelLayer {
public:
    ShaderLayer(CommonLayerDesc d): GenericModelLayer(d) {}
    ShaderLayer(const ShaderLayer&) = delete;
    ShaderLayer& operator=(const ShaderLayer&) = delete;
    ~ShaderLayer() {}

    // this struct allows us adding more return values easily.

    virtual GLSLShaders createGLSLShader(const LayerGenOptions& options) override {
        GLSLShaders ret;
        // auto executeLevel = getLayerExecutionLevel();
        // Try create compute shader, if the options says so.
        if ((options.compute)) {
            ret = createCS(options);
            setLayerExecutionLevel(snn::InferenceGraph::LayerExecution::GPU_CS);
        }
        // Buf if the shader layer does not support compute shader yet, we'll fallback to fragment shader.
        if (ret.passes.empty()) {
            ret = createFS(options);
            setLayerExecutionLevel(snn::InferenceGraph::LayerExecution::GPU_FS);
        }
        SNN_ASSERT(!ret.passes.empty());
        return ret;
    }

    virtual CPUPasses createCPUPasses(const CpuGenOptions& options) const override {
        // auto dummyOptions = options;
        (void) options;
        return CPUPasses();
    }

    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override { return {0, {{1.0f, 1.0f, 0.0f, 0.0f}}}; }

    static std::string loadShader(const char* path) {
        auto bytes = snn::loadEmbeddedAsset(path);
        return std::string(bytes.begin(), bytes.end());
    }

    static void findAndReplace(std::string& s, const std::string& from, const std::string& to) {
        auto startIndex = s.find(from);
        while (startIndex != std::string::npos) {
            s.replace(startIndex, from.size(), to);
            startIndex = s.find(from, startIndex + to.size());
        }
    }

    virtual snn::InferenceGraph::LayerExecution getLayerExecutionLevel() const override { return executeBackend; }
    virtual void setLayerExecutionLevel(snn::InferenceGraph::LayerExecution newExecution) override { executeBackend = newExecution; }

protected:
    virtual GLSLShaders createFS(const LayerGenOptions&) const { return {}; } // TODO: make this pure virtual
    virtual GLSLShaders createCS(const LayerGenOptions&) const { return {}; } // TODO: make this pure virtual

private:
    snn::InferenceGraph::LayerExecution executeBackend = snn::InferenceGraph::LayerExecution::GPU_FS;
};

class GenericConvolutionLayer : public ShaderLayer {
public:
    GenericConvolutionLayer(GenericConvDesc d): ShaderLayer(d), _desc(std::move(d)) {}
    ~GenericConvolutionLayer() {}

protected:
    struct WeightContants {
        std::string text;
        std::vector<glm::vec4> values;
    };

    virtual snn::InferenceGraph::LayerExecution getLayerExecutionLevel() const override { return snn::InferenceGraph::LayerExecution::GPU_FS; }

private:
    GenericConvDesc _desc;
};

} // namespace dp
} // namespace snn
