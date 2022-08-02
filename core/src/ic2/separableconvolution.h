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

struct SeparableConv2DDesc : GenericConvDesc {
    bool useBatchNormalization;
    std::map<std::string, std::vector<float>> batchNormalization;
    float leakyReluAlpha;
    std::string padding;
    bool useUniformShaders    = true;
    bool preferrHalfPrecision = false;
    std::string paddingT, paddingB, paddingL, paddingR;
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getDepthwiseConvolutionLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, activation, (int&) kernelSize, (int&) stride, biases,
                                            weights, useBatchNormalization, batchNormalization, leakyReluAlpha, paddingT, paddingB, paddingL, paddingR);
    }
};

class SeparableConv2DLayer : public GenericConvolutionLayer {
public:
    SeparableConv2DLayer(SeparableConv2DDesc&& d);
    ~SeparableConv2DLayer() {}
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;
    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override;

protected:
    virtual GLSLShaders createFS(const LayerGenOptions&) const override;
    virtual GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    SeparableConv2DDesc _desc;
    snn::FixedSizeArray<gl::TextureObject> weightTextures;
    snn::FixedSizeArray<gl::BufferObject<GL_UNIFORM_BUFFER>> weightUniformBuffers;
    snn::FixedSizeArray<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> weightSSBOBuffers;
    snn::FixedSizeArray<gl::SamplerObject> weightSamplers;
    std::vector<double> biases;
    gl::TextureObject kernelTexture;

    void setTextureWeights() const;
    void setBufferWeights() const;

    void getWeightConstants(std::ostringstream& weightConstants, const std::vector<float>& vWeightMatrices, int idxOutput4or8Chunk) const;

    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options) const;

    void buildFragmentPostDefine(std::ostream& stream) const;

    void buildFragmentCalc(std::ostringstream& stream) const;

    void buildElementAccess(std::ostream& stream, std::string padding) const;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;

    void getBiasConstants(std::vector<std::ostringstream>& biasConstants, uint32_t numShaderPasses) const;

    void getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass, const int outputChannels) const;
};

}; // namespace dp
} // namespace snn
