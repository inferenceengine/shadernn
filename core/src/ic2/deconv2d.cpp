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
#include "pch.h"
#include "dp.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <ios>

using namespace snn;
using namespace snn::dp;

constexpr uint32_t NUM_BATCH_NORM_PARAMETERS = 4;

void Conv2DTransposeLayer::getWeightConstants(std::vector<WeightContants>& weightConstants, const std::vector<std::vector<float>>& vWeightMatrices,
                                              int idxOutput4or8Chunk, int outputChannels) const {
    int nWeightStride       = _desc.kernelSize * _desc.kernelSize;
    int numWeightsPerMatrix = nWeightStride * ROUND_UP_DIV_4(_desc.numInputPlanes);

    std::vector<float> shaderOutputWeights(numWeightsPerMatrix * outputChannels, 0.0f);
    for (int idxInput4Chunk = 0; idxInput4Chunk < static_cast<int>(DIV_4_ROUND_UP(_desc.numInputPlanes)); ++idxInput4Chunk) {
        // InputPlane index

        int startOffset    = idxInput4Chunk * 4 * nWeightStride;
        int nInputChannels = std::min(4, static_cast<int>(_desc.numInputPlanes) - idxInput4Chunk * 4);
        // Do a transpose to match the RGBA output layout
        for (int idxSqKernel = 0; idxSqKernel < nWeightStride; ++idxSqKernel) {
            // Serialized element index (kernel x kernel)
            for (int idxOutput = 4 * idxOutput4or8Chunk, idxOutputChannel = 0; idxOutput < (4 * idxOutput4or8Chunk + outputChannels);
                ++idxOutput, ++idxOutputChannel) {
                shaderOutputWeights[(startOffset + idxSqKernel * 4) + (idxOutputChannel * numWeightsPerMatrix)] =
                    vWeightMatrices[idxOutput][startOffset + idxSqKernel];
                if (nInputChannels > 1) {
                    shaderOutputWeights[(startOffset + idxSqKernel * 4 + 1) + (idxOutputChannel * numWeightsPerMatrix)] =
                        vWeightMatrices[idxOutput][startOffset + idxSqKernel + nWeightStride];
                }
                if (nInputChannels > 2) {
                    shaderOutputWeights[(startOffset + idxSqKernel * 4 + 2) + (idxOutputChannel * numWeightsPerMatrix)] =
                        vWeightMatrices[idxOutput][startOffset + idxSqKernel + nWeightStride * 2];
                }
                if (nInputChannels > 3) {
                    shaderOutputWeights[(startOffset + idxSqKernel * 4 + 3) + (idxOutputChannel * numWeightsPerMatrix)] =
                        vWeightMatrices[idxOutput][startOffset + idxSqKernel + nWeightStride * 3];
                }
            }
        }
    }

    for (int idxOutputChannel = 0; idxOutputChannel < outputChannels; ++idxOutputChannel) {
        std::vector<glm::vec4> v;
        std::stringstream ss;
        ss.precision(10);
        ss << std::fixed;
        for (int idxInputAndSqKernel = 0; idxInputAndSqKernel < numWeightsPerMatrix; idxInputAndSqKernel += 4) {
            v.push_back({shaderOutputWeights[(idxOutputChannel * numWeightsPerMatrix) + idxInputAndSqKernel],
                         shaderOutputWeights[(idxOutputChannel * numWeightsPerMatrix) + idxInputAndSqKernel + 1],
                         shaderOutputWeights[(idxOutputChannel * numWeightsPerMatrix) + idxInputAndSqKernel + 2],
                         shaderOutputWeights[(idxOutputChannel * numWeightsPerMatrix) + idxInputAndSqKernel + 3]});
            ss << "vec4(" << v.back().x << ", " << v.back().y << ", " << v.back().z << ", " << v.back().w
                << ((idxInputAndSqKernel + 4 >= numWeightsPerMatrix) ? ")" : "),") << std::endl;
        }
        int idxWeight                     = 4 * idxOutput4or8Chunk + idxOutputChannel;
        weightConstants[idxWeight].text   = ss.str();
        weightConstants[idxWeight].values = std::move(v);
    }
}

void Conv2DTransposeLayer::getAllWeightConstants(std::vector<WeightContants>& weightConstants, uint32_t numShaderPasses) const {
    // TODO: This code is unoptimized; we can do this transposal in place instead
    auto& weightMatrices = _desc.weights;
    auto nWeightStride   = _desc.kernelSize * _desc.kernelSize;

    // Rounding up to the next divide by 4
    std::vector<std::vector<float>> vWeightMatrices(_desc.numOutputPlanes, std::vector<float>(nWeightStride * _desc.numInputPlanes, 0.f));

    for (uint32_t idxInput = 0; idxInput < _desc.numInputPlanes; ++idxInput) {
        uint32_t startOffset = idxInput * nWeightStride;

        for (uint32_t idxOutput = 0; idxOutput < _desc.numOutputPlanes; ++idxOutput) {
            SNN_ASSERT(startOffset < nWeightStride * _desc.numInputPlanes);
            SNN_ASSERT(idxOutput * _desc.numInputPlanes + idxInput < weightMatrices.size());
            memcpy(&vWeightMatrices[idxOutput][startOffset], weightMatrices[idxOutput * _desc.numInputPlanes + idxInput].data,
                   _desc.kernelSize * _desc.kernelSize * sizeof(float));
        }
    }

    for (int i = 0; i < static_cast<int>(numShaderPasses); ++i) {
        int outputChannels = std::min(4, static_cast<int>(_desc.numOutputPlanes) - i * 4);
        getWeightConstants(weightConstants, vWeightMatrices, i, outputChannels);
    }
    // vWeightMatrices.clear();
}

void Conv2DTransposeLayer::getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass,
                                                          const int outputChannels) const {
    auto iter       = _desc.batchNormalization.begin();
    int nStartIndex = 0;
    int index       = shaderPass * 4;
    for (; iter != _desc.batchNormalization.end(); ++iter, ++nStartIndex) {
        std::vector<float> shaderOutputBNConstants(outputChannels);
        batchNormalizationConstants[index + nStartIndex] << "vec4(";
        for (int i = 0; i < outputChannels; ++i) {
            shaderOutputBNConstants[i] = iter->second[index + i];
            batchNormalizationConstants[index + nStartIndex] << std::fixed << shaderOutputBNConstants[i] << ((i != outputChannels - 1) ? ", " : ")");
        }
    }
}

void Conv2DTransposeLayer::buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFileName) const {
    stream << "#version 320 es\n";
    stream << "// " << shaderFileName << std::endl;
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    stream << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << std::endl;
    stream << "#define NUM_KERNEL_SIZE " << _desc.kernelSize << std::endl;
    stream << "#define INPUT_WIDTH " << options.desiredInput.width << std::endl;
    stream << "#define INPUT_HEIGHT " << options.desiredInput.height << std::endl;
    stream << "#define NUM_STRIDE " << _desc.stride << std::endl;
    if (_desc.numInputPlanes <= 4) {
        stream << "#define INPUT_TEXTURE_2D\n";
    }
    if (_desc.useBatchNormalization) {
        stream << "#define USE_BATCH_NORMALIZATION\n";
    }
}

void Conv2DTransposeLayer::buildRgbaDefine(std::ostringstream& stream, uint32_t outputChannels) const {
    switch (outputChannels) {
    case 4:
        stream << "#define USE_COMPONENT_A\n";
        stream << "#define USE_COMPONENT_B\n";
        stream << "#define USE_COMPONENT_G\n";
        break;
    case 3:
        stream << "#define USE_COMPONENT_B\n";
        stream << "#define USE_COMPONENT_G\n";
        break;
    case 2:
        stream << "#define USE_COMPONENT_G\n";
        break;
    case 1:
        break;
    default:
        break;
    }
}

void Conv2DTransposeLayer::buildFragmentPostDefine(std::ostream& stream, const LayerGenOptions& options) const {
    // if (shader.isRGBA) {
    stream << (!_desc.activation.compare("relu")
                ? "s = max(s, vec4(0.0));\n"
                : !_desc.activation.compare("tanh")
                        ? "s = tanh(s);\n"
                        : !_desc.activation.compare("sigmoid")
                            ? "s = vec4(1.0f)/(vec4(1.0f)+ exp(-s));\n"
                            : !_desc.activation.compare("leakyRelu") ? "s = max(s, (s * vec4(" + std::to_string(_desc.leakyReluAlpha) + ")));\n" : "");
    if (options.isLastLayer && !getDesc().isRange01) {
        stream << "o_pixel = 0.5f * (s + vec4(1.0f));\n";
    } else {
        stream << "o_pixel = s;\n";
    }
    stream << "}\n";
}

void Conv2DTransposeLayer::buildComputePostDefine(std::ostream& stream, const LayerGenOptions& options, uint32_t outputSlice) const {
    stream << (!_desc.activation.compare("relu")
                ? "s = max(s, vec4(0.0));\n"
                : !_desc.activation.compare("tanh")
                        ? "s = tanh(s);\n"
                        : !_desc.activation.compare("sigmoid")
                            ? "s = vec4(1.0f)/(vec4(1.0f)+ exp(-s));\n"
                            : !_desc.activation.compare("leakyRelu") ? "s = max(s, (s * vec4(" + std::to_string(_desc.leakyReluAlpha) + ")));\n" : "");
    if (options.isLastLayer && !getDesc().isRange01) {
        stream << "s = 0.5f * (s + vec4(1.0f));\n";
    }
    if (_desc.numOutputPlanes > 4) {
        stream << "imageStore(outTexture,ivec3(gl_GlobalInvocationID.xy, " << outputSlice << "),s);\n";
    } else {
        SNN_ASSERT(0 == outputSlice);
        stream << "imageStore(outTexture,ivec2(gl_GlobalInvocationID.xy),s);\n";
    }
    stream << "}\n";
}

// Adds last text for fragment shader.
void Conv2DTransposeLayer::createShaderLayerCode(InferenceGraph::Pass& pass, std::string& shaderCode, const std::vector<WeightContants>& weightConstants,
                                                 std::vector<std::ostringstream>& batchNormalizationConstants, uint32_t shaderPassIndex,
                                                 uint32_t outputChannels) const {
    uint32_t weightOffset = shaderPassIndex * 4;

    for (uint32_t k = 0; k < outputChannels; ++k) {
        findAndReplace(shaderCode, formatString("_PLACEHOLDER_WEIGHT%d_VEC_CONSTANTS_", k + 1), weightConstants[weightOffset + k].text);
        pass.weightMatrices[k] = weightConstants[weightOffset + k].values;
    }

    if (_desc.useBatchNormalization) {
        getBatchNormalizationConstants(batchNormalizationConstants, (int) shaderPassIndex, (int) outputChannels);

        uint32_t offset = NUM_BATCH_NORM_PARAMETERS * shaderPassIndex;
        findAndReplace(shaderCode, "_PLACEHOLDER_BETA_VEC_CONSTANTS_", batchNormalizationConstants[offset].str());
        findAndReplace(shaderCode, "_PLACEHOLDER_GAMMA_VEC_CONSTANTS_", batchNormalizationConstants[offset + 1].str());
        findAndReplace(shaderCode, "_PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_", batchNormalizationConstants[offset + 2].str());
        findAndReplace(shaderCode, "_PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_", batchNormalizationConstants[offset + 3].str());
    }
}

ShaderLayer::GLSLShaders Conv2DTransposeLayer::createFS(const LayerGenOptions& options) const {
    auto fsFile = std::string("shaders/shadertemplate_fs_") + std::to_string(_desc.kernelSize) + "x_deconv_";
    if (_desc.stride == 2 && _desc.kernelSize == 4 && options.ssbo) {
        fsFile += "2s_RGBA.glsl";
    } else {
        fsFile += "RGBA.glsl";
    }

    std::string shaderTemplateCode = loadShader(fsFile.c_str());

    uint32_t numShaderPasses = DIV_4_ROUND_UP(_desc.numOutputPlanes);

    std::ostringstream preDefineStream;
    buildPreDefine(preDefineStream, options, fsFile);
    std::string preDefine = preDefineStream.str();

    std::ostringstream postDefineStream;
    buildFragmentPostDefine(postDefineStream, options);
    std::string postDefine = postDefineStream.str();

    std::vector<WeightContants> weightConstants(_desc.numOutputPlanes);
    getAllWeightConstants(weightConstants, numShaderPasses);

    std::vector<std::ostringstream> batchNormalizationConstants(NUM_BATCH_NORM_PARAMETERS * _desc.numOutputPlanes);

    GLSLShaders ret;

    // Get the list of passes.
    std::vector<InferenceGraph::Pass>& passes = ret.passes;

    // Create all the passes.
    passes.resize(numShaderPasses);

    // Initialize all the passes.
    for (uint32_t shaderPassIndex = 0; shaderPassIndex < numShaderPasses; ++shaderPassIndex) {
        uint32_t outputChannels = static_cast<uint32_t>(std::min(4, static_cast<int>(_desc.numOutputPlanes) - static_cast<int>(shaderPassIndex) * 4));
        std::ostringstream rgbaDefineStream;
        buildRgbaDefine(rgbaDefineStream, outputChannels);

        // Fetch the pass being modified.
        InferenceGraph::Pass& pass = passes[shaderPassIndex];

        // Create a copy of the template code.
        // After modification, this will contain the shader's true source code.
        std::string shaderCode = shaderTemplateCode;
        this->createShaderLayerCode(pass, shaderCode, weightConstants, batchNormalizationConstants, shaderPassIndex, outputChannels);

        if (_desc.preferHp) {
            findAndReplace(shaderCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(shaderCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        pass.source  = preDefine + rgbaDefineStream.str() + shaderCode + postDefine;
        pass.inputs  = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        pass.program = InferenceGraph::Pass::FsProgram {shaderPassIndex, DIV_4_ROUND_UP(outputChannels)};
    }

    return ret;
}

ShaderLayer::GLSLShaders Conv2DTransposeLayer::createCS(const LayerGenOptions& options) const {
    // Builds the path of the shader template file to load.
    std::ostringstream shaderFilePathStream;

    std::string fileName = std::string("/shadertemplate_cs_");
    fileName += std::to_string(_desc.kernelSize) + "x";
    shaderFilePathStream << "shaders" << fileName << "_deconv_";

    if (_desc.stride == 2 && _desc.kernelSize == 4 && options.compute) {
        shaderFilePathStream << "2s_RGBA.glsl";
    } else {
        shaderFilePathStream << "RGBA.glsl";
    }

    std::string shaderFilePath     = shaderFilePathStream.str();
    std::string shaderTemplateCode = loadShader(shaderFilePath.c_str());

    uint32_t numShaderPasses = DIV_4_ROUND_UP(_desc.numOutputPlanes);

    std::ostringstream preDefineStream;
    buildPreDefine(preDefineStream, options, shaderFilePath);
    std::string preDefine = preDefineStream.str();

    std::vector<WeightContants> weightConstants(_desc.numOutputPlanes);
    getAllWeightConstants(weightConstants, numShaderPasses);

    std::vector<std::ostringstream> batchNormalizationConstants(NUM_BATCH_NORM_PARAMETERS * _desc.numOutputPlanes);

    GLSLShaders ret;

    // Get the list of passes.
    std::vector<InferenceGraph::Pass>& passes = ret.passes;

    // Create all the passes.
    passes.resize(numShaderPasses);

    // Initialize all the passes.
    for (uint32_t shaderPassIndex = 0; shaderPassIndex < numShaderPasses; ++shaderPassIndex) {
        uint32_t outputChannels = static_cast<uint32_t>(std::min(4, static_cast<int>(_desc.numOutputPlanes) - static_cast<int>(shaderPassIndex) * 4));
        std::ostringstream rgbaDefineStream;
        buildRgbaDefine(rgbaDefineStream, outputChannels);

        std::ostringstream postDefineStream;
        buildComputePostDefine(postDefineStream, options, shaderPassIndex);
        std::string postDefine = postDefineStream.str();

        // Fetch the pass being modified.
        InferenceGraph::Pass& pass = passes[shaderPassIndex];

        // Create a copy of the template code.
        // After modification, this will contain the shader's true source code.
        std::string shaderCode = shaderTemplateCode;
        this->createShaderLayerCode(pass, shaderCode, weightConstants, batchNormalizationConstants, shaderPassIndex, outputChannels);

        pass.source  = preDefine + rgbaDefineStream.str() + shaderCode + postDefine;
        pass.inputs  = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        pass.program = InferenceGraph::Pass::CsProgram {
            "outTexture",
            // div-by-N is determined by work group size defined CS program.
            {(options.desiredOutputWidth + 7) / 8, (options.desiredOutputHeight + 7) / 8, 1},
        };
    }

    return ret;
}

InferenceGraph::Transform Conv2DTransposeLayer::getOutputScaleDimAdjustment() const {
    float scale = static_cast<float>(_desc.stride);
    float translation;
    if (_desc.paddingT == "same") {
        translation = 0;
    } else {
        translation = static_cast<float>(_desc.kernelSize - _desc.stride);
    }
    return {0, scale, scale, translation, translation};
}
