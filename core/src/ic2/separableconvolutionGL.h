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

#include "separableconvolution.h"
#include "snn/utils.h"
#include "glUtils.h"
#include <string>
#include <vector>
#include <sstream>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class SeparableConv2DLayerGl : public SeparableConv2DLayer {
public:
    SeparableConv2DLayerGl(SeparableConv2DDesc&& d);
    virtual ~SeparableConv2DLayerGl() = default;

protected:
    InferencePassesSptr createFS(const LayerGenOptions&) const override;
    InferencePassesSptr createCS(const LayerGenOptions&) const override;

private:
    mutable snn::FixedSizeArray<gl::TextureObject> weightTextures;
    mutable snn::FixedSizeArray<gl::BufferObject<GL_UNIFORM_BUFFER>> weightUniformBuffers;
    mutable snn::FixedSizeArray<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> weightSSBOBuffers;
    snn::FixedSizeArray<gl::SamplerObject> weightSamplers;
    std::vector<double> biases;
    mutable gl::TextureObject kernelTexture;

    void setTextureWeights() const;
    void setBufferWeights() const;

    void getWeightConstants(std::ostringstream& weightConstants, const std::vector<float>& vWeightMatrices, int idxOutput4or8Chunk) const;

    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options) const;

    void buildFragmentPostDefine(std::ostream& stream) const;

    void buildFragmentCalc(std::ostringstream& stream) const;

    void buildElementAccess(std::ostream& stream, std::string padding) const;

    void getBiasConstants(std::vector<std::ostringstream>& biasConstants, uint32_t numShaderPasses) const;

    void getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass, const int outputChannels) const;
};

} // namespace dp
} // namespace snn
