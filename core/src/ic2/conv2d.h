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

struct Conv2DDesc : GenericConvDesc {
    bool useBatchNormalization;
    bool useMultiInputs;
    std::map<std::string, std::vector<float>> batchNormalization;
    float leakyReluAlpha;
    std::string padding;
    bool useUniformShaders    = true;
    bool preferrHalfPrecision = false;
    std::string paddingT, paddingB, paddingL, paddingR;
    std::string paddingMode = "constant";
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getConvolutionLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, activation, (int&) kernelSize, (int&) stride, biases, weights,
                                   useBatchNormalization, batchNormalization, leakyReluAlpha, paddingT, paddingB, paddingL, paddingR, paddingMode,
                                   useMultiInputs);
    }
};

class Conv2DLayer : public GenericConvolutionLayer {
public:
    Conv2DLayer(Conv2DDesc&& d);
    ~Conv2DLayer() {}
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    virtual GLSLShaders createFS(const LayerGenOptions&) const override;
    virtual GLSLShaders createCS(const LayerGenOptions&) const override;

private:
    Conv2DDesc _desc;
    snn::FixedSizeArray<gl::TextureObject> weightTextures;
    snn::FixedSizeArray<gl::BufferObject<GL_UNIFORM_BUFFER>> weightUniformBuffers;
    snn::FixedSizeArray<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> weightSSBOBuffers;
    std::vector<double> biases;
    bool isRange01;
    bool isFirstLayer;
    gl::TextureObject kernelTexture;

    // Several non-separable inputs
    void getWeightConstantsMultipleInput(std::vector<std::ostringstream>& weightConstants, const std::vector<std::vector<float>>& vWeightMatrices,
                                         int idxOutput4or8Chunk, int outputChannels) const;
    // Uniform weights
    void setTextureWeights() const;
    void setBufferWeights() const;
    // One input
    void getWeightConstantsSingleInput(std::vector<std::ostringstream>& weightConstants, const std::vector<std::vector<float>>& vWeightMatrices, int shaderPass,
                                       int outputChannels) const;
    // Several separable inputs
    void getWeightConstantsIndividualInputs(std::vector<std::ostringstream>& weightConstants, const std::vector<std::vector<float>>& vWeightMatrices,
                                            int shaderPass, int outputChannels) const;

    //
    void getAllWeightConstants(std::vector<std::ostringstream>& weightConstants, uint32_t numShaderPasses) const;

    void getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass, const int outputChannels) const;

    void getBiasConstants(std::vector<std::ostringstream>& biasConstants, uint32_t numShaderPasses) const;

    void buildDotProductLogic(std::ostringstream& stream) const; // fills the placeholder code in glsl

    // Adds predefine to given stringstream.
    // shaderFilePath is the path of the file template this will be appended to
    // and is used to label the preDefine.
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;

    // Adds last text for fragment shader.
    void buildFragmentPostDefine(std::ostream& stream) const;

    // Adds last text for compute shader.
    void buildComputePostDefine(std::ostream& stream, uint32_t outputSliceIndex) const;

    // Adds element access
    void buildElementAccess(std::ostream& stream, std::string padding, bool isFirstLayer) const;

    // Add conv2d calc logic based on element acces
    void buildFragmentCalc(std::ostringstream& stream) const;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;
};

}; // namespace dp
} // namespace snn
