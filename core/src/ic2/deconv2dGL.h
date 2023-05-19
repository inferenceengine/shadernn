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

#include "deconv2d.h"
#include "snn/snn.h"
#include "snn/utils.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <sstream>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

// This is a class to generates a shader for transposed 2D convolution for OpenGL
class Conv2DTransposeLayerGl : public GenericConvolutionLayer {
public:
    Conv2DTransposeLayerGl(Conv2DTransposeDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    virtual ~Conv2DTransposeLayerGl() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    InferencePassesSptr createFS(const LayerGenOptions&) const override;
    InferencePassesSptr createCS(const LayerGenOptions&) const override;

private:
    Conv2DTransposeDesc _desc;

    void getWeightConstants(std::vector<WeightContants>& weightConstants, const std::vector<std::vector<float>>& vWeightMatrices, int idxOutput4or8Chunk,
                            int outputChannels) const;
    void getAllWeightConstants(std::vector<WeightContants>& weightConstants, uint32_t numShaderPasses) const;

    void getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass, const int outputChannels) const;

    // Adds predefine to given stringstream.
    // shaderFile is the name of the file template this will be appended to
    // and is used to label the preDefine.
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFileName) const;

    // Create the rgba channels.
    void buildRgbaDefine(std::ostringstream& stream, uint32_t outputChannels) const;

    // Adds last text for fragment shader.
    void buildFragmentPostDefine(std::ostream& stream) const;

    void buildFragmentPostDefine(std::ostream& stream, const LayerGenOptions& options) const;

    // Adds last text for compute shader.
    void buildComputePostDefine(std::ostream& stream, const LayerGenOptions& options, uint32_t outputSliceIndex) const;

    // Adds last text for fragment shader.
    void createShaderLayerCode(InferencePassGl& pass, std::string& shaderCode, const std::vector<WeightContants>& weightConstants,
                               std::vector<std::ostringstream>& batchNormalizationConstants, uint32_t shaderPassIndex, uint32_t outputChannels) const;
};

}; // namespace dp
} // namespace snn
