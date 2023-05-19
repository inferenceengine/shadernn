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

#include "conv2d.h"
#include "snn/utils.h"
#include "glUtils.h"
#include <string>
#include <vector>
#include <sstream>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class Conv2DLayerGl : public Conv2DLayer {
public:
    Conv2DLayerGl(Conv2DDesc&& d);
    virtual ~Conv2DLayerGl() = default;

protected:
    InferencePassesSptr createFS(const LayerGenOptions&) const override;
    InferencePassesSptr createCS(const LayerGenOptions&) const override;

private:
    mutable snn::FixedSizeArray<gl::TextureObject> weightTextures;
    mutable snn::FixedSizeArray<gl::BufferObject<GL_UNIFORM_BUFFER>> weightUniformBuffers;
    mutable snn::FixedSizeArray<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> weightSSBOBuffers;
    std::vector<double> biases;
    bool isRange01;
    bool isFirstLayer;
    mutable gl::TextureObject kernelTexture;

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
    void buildElementAccess(std::ostream& stream, const std::string& padding, bool isFirstLayer) const;

    // Add conv2d calc logic based on element acces
    void buildFragmentCalc(std::ostringstream& stream) const;
};

}; // namespace dp
} // namespace snn
