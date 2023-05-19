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
#include "pch.h"
#include "separableconvolutionGL.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <utility>

using namespace snn;
using namespace snn::dp;

#define CLAMPED_PADDING 1

static constexpr const char* DEPTHWISE_CONV2D_CS_ASSET_NAME = "shaders/3rdparty/shadertemplate_cs_separableconvolution.glsl";

SeparableConv2DLayerGl::SeparableConv2DLayerGl(SeparableConv2DDesc&& d): SeparableConv2DLayer(std::move(d)) {
    if (_desc.numInputPlanes <= 2048) {
        _desc.useUniformShaders = false;
        _desc.weightMode        = snn::WeightAccessMethod::CONSTANTS;
    }

    if (_desc.weightMode == snn::WeightAccessMethod::CONSTANTS) {
        _desc.useUniformShaders = false;
    }

    this->biases = _desc.biases;

    SNN_LOGD("Size of biases for current layer with input channels %u and output channels %u is %u", _desc.numInputPlanes, _desc.numOutputPlanes,
             this->biases.size());
    if (this->biases.size() % 4 != 0) {
        auto initSize  = this->biases.size();
        auto finalSize = 4 * ((uint32_t)(this->biases.size() / 4) + 1);
        for (std::size_t i = initSize; i < finalSize; i++) {
            this->biases.push_back(0.0);
        }
    }
}

void SeparableConv2DLayerGl::setTextureWeights() const {
    std::vector<float> weightVal(4 * _desc.kernelSize * _desc.kernelSize, 0.0);
    for (std::size_t filter = 0; filter < _desc.numOutputPlanes; filter++) {
        SNN_LOGD("Updating weights for weight texture: %u, %u", this->weightTextures[filter].id(), this->weightTextures[filter].target());
        for (std::size_t i = 0; i < _desc.kernelSize; i++) {
            for (std::size_t j = 0; j < _desc.kernelSize; j++) {
                std::size_t weightValIdx = (4 * _desc.kernelSize * i) + (4 * j) + (filter % 4);
                SNN_LOGD("Filling IDX %u in buffer with size %u", weightValIdx, weightVal.size());
                if (_desc.preferHp) {
                    uint16_t* fp16Addr = (uint16_t*)weightVal.data();
                    *(fp16Addr + weightValIdx) = FP32::toHalf(_desc.weightsCvM[filter].at<float>(i, j));
                } else {
                    weightVal[weightValIdx] = _desc.weightsCvM[filter].at<float>(i, j);
                }
            }
        }
        if ((filter + 1) % 4 == 0) {
            this->weightTextures[filter / 4].bind(0);
            this->weightTextures[filter / 4].setPixels(0, 0, 0, _desc.kernelSize, _desc.kernelSize, 0, weightVal.data());
            glFinish();
            this->weightTextures[filter / 4].unbind();
            weightVal.clear();
            weightVal.resize(_desc.kernelSize * _desc.kernelSize * 4, 0.0);
        }
    }
    if (!weightVal.empty() && _desc.numOutputPlanes % 4 != 0) {
        uint32_t extraChannels = ROUND_UP_DIV_4(_desc.numOutputPlanes) - _desc.numOutputPlanes;
        for (uint32_t i = 0; i < _desc.kernelSize * _desc.kernelSize * extraChannels; i++) {
            weightVal.push_back(0.0f);
        }
        this->weightTextures[DIV_4_ROUND_UP(_desc.numOutputPlanes)].bind(0);
        this->weightTextures[DIV_4_ROUND_UP(_desc.numOutputPlanes)].setPixels(0, 0, 0, _desc.kernelSize, _desc.kernelSize, 0, weightVal.data());
        glFinish();
        this->weightTextures[DIV_4_ROUND_UP(_desc.numOutputPlanes)].unbind();
    }
}

void SeparableConv2DLayerGl::setBufferWeights() const {
    uint8_t byteSize = _desc.preferHp ? 2 : 4;
    std::vector<uint8_t> weightVal(byteSize * 4 * _desc.kernelSize * _desc.kernelSize, 0);
    for (std::size_t filter = 0; filter < _desc.numOutputPlanes; filter++) {
        for (std::size_t i = 0; i < _desc.kernelSize; i++) {
            for (std::size_t j = 0; j < _desc.kernelSize; j++) {
                std::size_t weightValIdx = (byteSize * 4 * _desc.kernelSize * i) + (4 * j) + (filter % 4);
                std::vector<uint8_t> byteRep;
                snn::getByteRepresentation(_desc.weightsCvM[filter].at<float>(j, i), byteRep, _desc.preferHp);
                for (std::size_t byteIdx = 0; byteIdx < byteSize; byteIdx++) {
                    weightVal[weightValIdx + byteIdx] = byteRep.at(byteIdx);
                }
            }
        }
        if ((filter + 1) % 4 == 0) {
            switch (_desc.weightMode) {
            case snn::WeightAccessMethod::UNIFORM_BUFFER:
                this->weightUniformBuffers[filter / 4].update(weightVal.data(), 0, weightVal.size());
                break;

            case snn::WeightAccessMethod::SSBO_BUFFER:
                this->weightSSBOBuffers[filter / 4].update(weightVal.data(), 0, weightVal.size());
                break;

            default:
                break;
            }
            weightVal.clear();
        }
    }
    if (!weightVal.empty() && _desc.numOutputPlanes % 4 != 0) {
        switch (_desc.weightMode) {
        case snn::WeightAccessMethod::UNIFORM_BUFFER:
            this->weightUniformBuffers[DIV_4_ROUND_UP(_desc.numOutputPlanes)].update(weightVal.data(), 0, weightVal.size());
            break;

        case snn::WeightAccessMethod::SSBO_BUFFER:
            this->weightSSBOBuffers[DIV_4_ROUND_UP(_desc.numOutputPlanes)].update(weightVal.data(), 0, weightVal.size());
            break;

        default:
            break;
        }
        weightVal.clear();
    }
}

void SeparableConv2DLayerGl::buildElementAccess(std::ostream& stream, std::string padding) const {
    uint32_t paddingOffsets[4];
    getPaddingOffset(paddingOffsets);

    std::vector<float> coordOffsetsW(_desc.kernelSize), coordOffsetsH(_desc.kernelSize);
    std::iota(coordOffsetsW.begin(), coordOffsetsW.end(), -0.5 - paddingOffsets[0]);
    std::iota(coordOffsetsH.begin(), coordOffsetsH.end(), -0.5 - paddingOffsets[2]);

    std::string paddingStr;
    std::string baseTabLevel = "\t\t";
    auto getTabLevels        = [&](int level) {
        std::string tabLevel = baseTabLevel;
        for (int i = 0; i < level; i++) {
            tabLevel += "\t";
        }
        return tabLevel;
    };
    if (std::isdigit(padding.front()) && padding != "0") {
        paddingStr = "same";
    } else if (padding == "0") {
        paddingStr = "none";
    } else {
        paddingStr = padding;
    }

    int channelsPerPass           = 4;
    SNN_LOGD("%d", channelsPerPass);
    switch (_desc.mrtMode) {
    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;
    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;
    case snn::MRTMode::SINGLE_PLANE:
    default:
        break;
    }
    int planeCount = DIV_4_ROUND_UP(channelsPerPass);

    for (uint32_t i = 0; i < _desc.kernelSize; i++) {
        for (uint32_t j = 0; j < _desc.kernelSize; j++) {
            stream << getTabLevels(0) << "vec2 texCoord_" << _desc.kernelSize * i + j + 1 << " = (vec2(baseCoord) + ";
            stream << "vec2(" << coordOffsetsW.at(j) << ", " << coordOffsetsH.at(i) << ")) / vec2(maxUV);" << std::endl;
        }
    }

    stream << getTabLevels(1) << "int layer = OUTPUTPLANE_INDEX >> 2;" << std::endl;
    for (int planeIdx = 1; planeIdx < planeCount; planeIdx++) {
        stream << "#if PLANE_COUNT > " << planeIdx << "\n";
        stream << getTabLevels(1) << "int layer" << planeIdx << " = layer + " << planeIdx << ";\n";
        stream << "#endif\n";
    }
    for (uint32_t i = 0; i < _desc.kernelSize; i++) {
        for (uint32_t j = 0; j < _desc.kernelSize; j++) {
            int linearDim = _desc.kernelSize * i + j;
#if CLAMPED_PADDING == 0
            if (paddingStr == "same") {
                stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t0_" << linearDim << " = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                stream << "texture(inputTextures, vec3(texCoord_";
                stream << linearDim + 1 << ", layer)) : vec4(PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE);\n";
            } else if (paddingStr == "replicate") {
                stream << getTabLevels(1) << "FLOAT_PRECISION vec2 repCoords" << linearDim << " = replicatePadding(texCoord_" << linearDim + 1 << ");\n";
                stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t0_" << linearDim << " = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                stream << "texture(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer)) : texture(inputTextures, vec3(repCoords" << linearDim + 1
                        << ", layer));\n";
            } else if (paddingStr == "valid" || paddingStr == "none") {
                stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t0_" << linearDim << " = texture(inputTextures, vec3(texCoord_" << linearDim + 1
                        << ", layer));\n";
            }
#else
            stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t" << linearDim << " = texture(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer));\n";
#endif  // CLAMPED_PADDING == 0
            // This line implements clamping of zero valued pixels.
            // This necessary only when the input is between 0 and 1, and the final layer
            // is an image-like output (segmentation or denoised image) with a sigmoid activation.
            // In case of all zeros, this leads to a 0.5 at the output, when we want a 0
            // By adding a small offset or clamp value, we avoid this issue, and the output is zero
            // like we expect
        }
    }
    for (int planeIdx = 1; planeIdx < planeCount; planeIdx++) {
        for (uint32_t i = 0; i < _desc.kernelSize; i++) {
            for (uint32_t j = 0; j < _desc.kernelSize; j++) {
                int linearDim      = _desc.kernelSize * i + j;
                std::string texSub = std::to_string(planeIdx) + "_" + std::to_string(linearDim);
#if CLAMPED_PADDING == 0
                if (paddingStr == "same") {
                    stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t" << texSub << " = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                    stream << "texture(inputTextures, vec3(texCoord_";
                    stream << linearDim + 1 << ", layer" << planeIdx << ")) : vec4(PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE);\n";
                } else if (paddingStr == "replicate") {
                    stream << getTabLevels(1) << "FLOAT_PRECISION vec2 repCoords" << linearDim << " = replicatePadding(texCoord_" << linearDim + 1 << ");\n";
                    stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t" << texSub << " = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                    stream << "texture(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer" << planeIdx << ")) : texture(inputTextures, vec3(repCoords"
                           << linearDim + 1 << ", layer" << planeIdx << "));\n";
                } else if (paddingStr == "valid" || paddingStr == "none") {
                    stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t" << texSub << " = texture(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer"
                           << planeIdx << "));\n";
                }
#else
                stream << getTabLevels(1) << "FLOAT_PRECISION vec4 t" << texSub << " = texture(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer"
                       << planeIdx << "));\n";
#endif  // CLAMPED_PADDING == 0
            }
        }
    }
}

void SeparableConv2DLayerGl::buildFragmentCalc(std::ostringstream& stream) const {
    std::string baseTabLevel = "\t\t";
    uint32_t offsets[4];
    getPaddingOffset(offsets);
    if (offsets[0] == 0) {
        baseTabLevel += "\t";
    }
    int channelsPerPass           = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;

    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;

    default:
        break;
    }
    int planeCount          = DIV_4_ROUND_UP(channelsPerPass);
    const int weightsOffset = _desc.kernelSize * _desc.kernelSize;
    int linearDim           = 0;
    stream << baseTabLevel << "s = (";
    for (std::size_t i = 0; i < _desc.kernelSize; i++) {
        for (std::size_t j = 0; j < _desc.kernelSize; j++) {
            linearDim = i * _desc.kernelSize + j;
            if (linearDim == weightsOffset - 1) {
                stream << baseTabLevel << "t" << linearDim << " * weightMatrix1";
                stream << "[" << linearDim << "]);" << std::endl;
            } else if (linearDim == 0) {
                stream << "t" << linearDim << " * weightMatrix1";
                stream << "[" << linearDim << "] +" << std::endl;
            } else {
                stream << baseTabLevel << "\tt" << linearDim << " * weightMatrix1";
                stream << "[" << linearDim << "] +" << std::endl;
            }
        }
    }
    for (int planeIdx = 1; planeIdx < planeCount; planeIdx++) {
        stream << baseTabLevel << "s" << planeIdx << " = (";
        for (std::size_t i = 0; i < _desc.kernelSize; i++) {
            for (std::size_t j = 0; j < _desc.kernelSize; j++) {
                linearDim = i * _desc.kernelSize + j;
                if (linearDim == weightsOffset - 1) {
                    stream << baseTabLevel << "t" << planeIdx << "_" << linearDim << " * weightMatrix" << planeIdx + 1;
                    stream << "[" << linearDim << "]);" << std::endl;
                } else if (linearDim == 0) {
                    stream << "t" << planeIdx << "_" << linearDim << " * weightMatrix" << planeIdx + 1;
                    stream << "[" << linearDim << "] +" << std::endl;
                } else {
                    stream << baseTabLevel << "\tt" << planeIdx << "_" << linearDim << " * weightMatrix" << planeIdx + 1;
                    stream << "[" << linearDim << "] +" << std::endl;
                }
            }
        }
    }
}

void SeparableConv2DLayerGl::getWeightConstants(std::ostringstream& weightConstants, const std::vector<float>& vWeightMatrices, int idxInput4or8Chunk) const {
    int nWeightStride = _desc.kernelSize * _desc.kernelSize;

    int idxWeightMatrices = nWeightStride * idxInput4or8Chunk * 4;
    int nWeightMatrices   = static_cast<int>(vWeightMatrices.size());
    SNN_ASSERT(idxWeightMatrices < nWeightMatrices);

    weightConstants.precision(10);
    for (int idxWeight = 0; idxWeight < nWeightStride; ++idxWeight, ++idxWeightMatrices) {
        weightConstants << "vec4(";
        weightConstants << std::fixed << ((idxWeightMatrices < nWeightMatrices) ? vWeightMatrices[idxWeightMatrices] : 0);
        weightConstants << ", ";
        weightConstants << std::fixed << ((idxWeightMatrices + nWeightStride * 1 < nWeightMatrices) ? vWeightMatrices[idxWeightMatrices + nWeightStride] : 0);
        weightConstants << ", ";
        weightConstants << std::fixed
                        << ((idxWeightMatrices + nWeightStride * 2 < nWeightMatrices) ? vWeightMatrices[idxWeightMatrices + nWeightStride * 2] : 0);
        weightConstants << ", ";
        weightConstants << std::fixed
                        << ((idxWeightMatrices + nWeightStride * 3 < nWeightMatrices) ? vWeightMatrices[idxWeightMatrices + nWeightStride * 3] : 0);
        weightConstants << ((idxWeight == nWeightStride - 1) ? ")" : "),") << std::endl;
    }
}

void SeparableConv2DLayerGl::getBiasConstants(std::vector<std::ostringstream>& biasConstants, uint32_t numShaderPasses) const {
    uint32_t outputPlanes = _desc.numOutputPlanes;
    int channelsPerPass   = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;
    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;
    case snn::MRTMode::SINGLE_PLANE:
    default:
        break;
    }
    uint32_t currentPlanes = 0;
    for (std::size_t i = 0; i < numShaderPasses; i++) {
        if (outputPlanes > channelsPerPass) {
            currentPlanes = channelsPerPass;
            outputPlanes  = outputPlanes - channelsPerPass;
        } else {
            currentPlanes = outputPlanes;
            outputPlanes  = 0;
        }

        biasConstants.at(i) << "const FLOAT_PRECISION vec4 bias[] = vec4[](";
        for (std::size_t j = 0; j < DIV_4_ROUND_UP(currentPlanes); j++) {
            int remainingPlanes = ((currentPlanes - (j * 4)) > 4) ? 4 : (currentPlanes - (j * 4));
            switch (remainingPlanes) {
            case 4:
                biasConstants.at(i) << "vec4(" << _desc.biases.at(channelsPerPass * i + 4 * j) << ", " << _desc.biases.at(channelsPerPass * i + 4 * j + 1)
                                    << ", " << _desc.biases.at(channelsPerPass * i + 4 * j + 2) << ", " << _desc.biases.at(channelsPerPass * i + 4 * j + 3)
                                    << (((currentPlanes - (j * 4)) > 4) ? "),\n" : "));\n");

                break;

            case 3:
                biasConstants.at(i) << "vec4(" << _desc.biases.at(channelsPerPass * i + 4 * j) << ", " << _desc.biases.at(channelsPerPass * i + 4 * j + 1)
                                    << ", " << _desc.biases.at(channelsPerPass * i + 4 * j + 2) << ", "
                                    << "0.0" << (((currentPlanes - (j * 4)) > 4) ? "),\n" : "));\n");
                break;

            case 2:
                biasConstants.at(i) << "vec4(" << _desc.biases.at(channelsPerPass * i + 4 * j) << ", " << _desc.biases.at(channelsPerPass * i + 4 * j + 1)
                                    << ", "
                                    << "0.0, "
                                    << "0.0" << (((currentPlanes - (j * 4)) > 4) ? "),\n" : "));\n");
                break;

            case 1:
                biasConstants.at(i) << "vec4(" << _desc.biases.at(channelsPerPass * i + 4 * j) << ", "
                                    << "0.0, "
                                    << "0.0, "
                                    << "0.0" << (((currentPlanes - (j * 4)) > 4) ? "),\n" : "));\n");
                break;

            default:
                biasConstants.at(i) << " vec4("
                                    << "0.0, "
                                    << "0.0, "
                                    << "0.0, "
                                    << "0.0" << (((currentPlanes - (j * 4)) > 4) ? "),\n" : "));\n");
                break;
            }
        }
    }
}

void SeparableConv2DLayerGl::getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass,
                                                          const int outputChannels) const {
    auto iter           = _desc.batchNormalization.begin();
    int nStartIndex     = 0;
    int channelsPerPass = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;
    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;
    case snn::MRTMode::SINGLE_PLANE:
    default:
        break;
    }
    int planeCount = DIV_4_ROUND_UP(channelsPerPass);
    int index      = shaderPass * 4;
    for (; iter != _desc.batchNormalization.end(); ++iter, ++nStartIndex) {
        std::vector<float> shaderOutputBNConstants(outputChannels);
        batchNormalizationConstants[index + nStartIndex] << "vec4(";
        for (int i = 0; i < outputChannels; ++i) {
            if (i % 4 == 0 && i > 0) {
                batchNormalizationConstants[index + nStartIndex] << ", \n\tvec4(";
            }
            shaderOutputBNConstants[i] = iter->second[planeCount * index + i];
            batchNormalizationConstants[index + nStartIndex] << std::fixed << shaderOutputBNConstants[i] << (((i + 1) % 4 == 0) ? ") " : ",");
        }
    }
}

void SeparableConv2DLayerGl::buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options) const {
    uint32_t offsets[4];
    getPaddingOffset(offsets);
    stream << "#version 320 es\n";
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    stream << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << std::endl;
    stream << "#define INPUT_WIDTH " << options.desiredInput[0].width << std::endl;
    stream << "#define INPUT_HEIGHT " << options.desiredInput[0].height << std::endl;
    stream << "#define NUM_KERNEL_SIZE " << _desc.kernelSize << std::endl;
    stream << "#define NUM_STRIDE " << _desc.stride << std::endl;
    stream << "#define PAD_VALUE 0.0f" << std::endl;
#if CLAMPED_PADDING == 1
    stream << "#define CLAMPED_PADDING" << std::endl;
#endif  // CLAMPED_PADDING == 1
    switch (_desc.weightMode) {
    case snn::WeightAccessMethod::SSBO_BUFFER:
        stream << "#define USE_WEIGHT_BUFFERS\n";
        stream << "#define STORAGE_FORMAT std430\n";
        stream << "#define VARIABLE_SPECIFIER buffer\n";
        break;

    case snn::WeightAccessMethod::UNIFORM_BUFFER:
        stream << "#define USE_WEIGHT_BUFFERS\n";
        stream << "#define STORAGE_FORMAT std140\n";
        stream << "#define VARIABLE_SPECIFIER uniform\n";
        break;

    case snn::WeightAccessMethod::TEXTURES:
        stream << "#define USE_WEIGHT_TEXTURES\n";
        break;

    case snn::WeightAccessMethod::CONSTANTS:
        stream << "#define USE_WEIGHT_CONSTANTS\n";
        break;

    default:
        SNN_LOGE("No Weight setting found!");
        break;
    }
    if (_desc.numInputPlanes <= 4) {
        stream << "#define INPUT_TEXTURE_2D\n";
    }
    if (_desc.useBatchNormalization) {
        stream << "#define USE_BATCH_NORMALIZATION\n";
    }
    if (_desc.useUniformShaders) {
        stream << "#define USE_UNIFORM_WEIGHTS" << std::endl;
        stream << "#define PADDING_T " << offsets[0] << std::endl;
        stream << "#define PADDING_B " << offsets[1] << std::endl;
        stream << "#define PADDING_L " << offsets[2] << std::endl;
        stream << "#define PADDING_R " << offsets[3] << std::endl;
        if (_desc.paddingT == "same") {
            stream << "#define CONST_PADDING" << std::endl;
        } else if (_desc.paddingT == "replicate") {
            stream << "#define REPLCIATE_PADDING" << std::endl;
        } else {
            stream << "#define CHECKBOARD_PADDING" << std::endl;
        }
    } else {
        stream << "#define PADDING_W " << offsets[0] << std::endl;
        stream << "#define PADDING_H " << offsets[1] << std::endl;
    }
}

// Adds last text for fragment shader.
void SeparableConv2DLayerGl::buildFragmentPostDefine(std::ostream& stream) const {
    const char* identifiers[4] = {"", "1", "2", "3"};
    for (std::size_t i = 0; i < 4; i++) {
        stream << "#if PLANE_COUNT > " << i << "\n";
        stream << (!_desc.activation.compare("relu")
                    ? "\ts" + std::string(identifiers[i]) + " = max(s" + std::string(identifiers[i]) + ", vec4(0.0));\n"
                    : !_desc.activation.compare("relu6")
                            ? "\ts" + std::string(identifiers[i]) + " = min(vec4(6.0),max(s" + std::string(identifiers[i]) + ", vec4(0.0)));\n"
                            : !_desc.activation.compare("tanh")
                                ? "\ts" + std::string(identifiers[i]) + " = tanh(s" + std::string(identifiers[i]) + ");\n"
                                : !_desc.activation.compare("sigmoid")
                                        ? "\ts" + std::string(identifiers[i]) + " = vec4(1.0f)/(vec4(1.0f)+ exp(-s" + std::string(identifiers[i]) + "));\n"
                                        : !_desc.activation.compare("leakyRelu")
                                            ? "\ts" + std::string(identifiers[i]) + " = max(s" + std::string(identifiers[i]) + ", (s" +
                                                    std::string(identifiers[i]) + " * vec4(" + std::to_string(_desc.leakyReluAlpha) + "f)));\n"
                                            : !_desc.activation.compare("leaky_relu")
                                                    ? "\ts" + std::string(identifiers[i]) + " = max(s" + std::string(identifiers[i]) + " , (s" +
                                                        std::string(identifiers[i]) + " * vec4(" + std::to_string(_desc.leakyReluAlpha) + "f)));\n"
                                                    : !_desc.activation.compare("SiLU")
                                                        ? "s" + std::string(identifiers[i]) + " = s" + std::string(identifiers[i]) +
                                                                " * vec4(1.0f)/(vec4(1.0f)+ exp(-s" + std::string(identifiers[i]) + "));\n"
                                                        : "");
        stream << "\to_pixel" << std::string(identifiers[i]) << " = s" << std::string(identifiers[i]) << ";\n";
        stream << "#endif\n";
    }
    stream << "}";
}

InferencePassesSptr SeparableConv2DLayerGl::createFS(const LayerGenOptions& options) const {
    std::ostringstream fsFile;
    std::string fileName = "/shadertemplate_fs_depthwise_RGBA";
    fsFile << "shaders" << fileName << ".glsl";

    int channelsPerPass = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;
    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;
    case snn::MRTMode::SINGLE_PLANE:
    default:
        break;
    }

    auto numShaderPasses = DIV_AND_ROUND_UP(_desc.numInputPlanes, channelsPerPass);
    int planes           = DIV_4_ROUND_UP(channelsPerPass);
    std::vector<std::ostringstream> weightConstants(planes * numShaderPasses);

    // TODO: This code is unoptimized; we can do this transposal in place instead
    auto& weightMatrices = _desc.weightsCvM;
    int nWeightStride    = _desc.kernelSize * _desc.kernelSize;

    std::vector<float> vWeightMatrices(nWeightStride * _desc.numInputPlanes, 0.f);
    std::vector<std::ostringstream> biasStr(numShaderPasses);

    if (!_desc.useUniformShaders) {
        for (int idxInput = 0, startOffset = 0; idxInput < static_cast<int>(_desc.numInputPlanes); ++idxInput, startOffset += nWeightStride) {
            std::memcpy(&vWeightMatrices[startOffset], weightMatrices[idxInput].data, _desc.kernelSize * _desc.kernelSize * sizeof(float));
        }

        for (int i = 0; i < static_cast<int>(planes * numShaderPasses); ++i) {
            getWeightConstants(weightConstants[i], vWeightMatrices, i);
        }
        getBiasConstants(biasStr, numShaderPasses);
        vWeightMatrices.clear();
    }

    InferencePassesSptr ret(new InferencePassesGl());
    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(numShaderPasses);
    for (uint32_t i = 0; i < numShaderPasses; ++i) {
        uint32_t outputChannels =
            static_cast<uint32_t>(std::min(channelsPerPass, static_cast<int>(_desc.numOutputPlanes) - static_cast<int>(i) * channelsPerPass));
        uint32_t planeCount = DIV_4_ROUND_UP(outputChannels);
        std::ostringstream preDefine;
        buildPreDefine(preDefine, options);

        preDefine << "#define PLANE_COUNT " << planeCount << "\n";

        std::ostringstream postDefine;
        buildFragmentPostDefine(postDefine);

        preDefine << "#define OUTPUTPLANE_INDEX " << planeCount * i * 4 << std::endl;

        auto fsCode = loadShader(fsFile.str().c_str());

        const int numBatchNormParameters = 4;
        std::vector<std::ostringstream> batchNormalizationConstants(numBatchNormParameters * _desc.numOutputPlanes);

        if (!_desc.useUniformShaders) {
            for (int j = planeCount * i; j < planeCount * i + planeCount; j++) {
                std::string weightConstantShaderPlaceholder = formatString("_PLACEHOLDER_WEIGHT%d_VEC_CONSTANTS_", j - (planeCount * i) + 1);
                findAndReplace(fsCode, weightConstantShaderPlaceholder, weightConstants[j].str());
            }
            findAndReplace(fsCode, "_PLACEHOLDER_BIAS_CONSTANTS_", biasStr.at(i).str());
        }

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        if (!_desc.useUniformShaders) {
            std::vector<std::ostringstream> batchNormalizationConstants(numBatchNormParameters * _desc.numOutputPlanes);

            std::string weightConstantShaderPlaceholder("_PLACEHOLDER_WEIGHT_VEC_CONSTANTS_");
            findAndReplace(fsCode, weightConstantShaderPlaceholder, weightConstants[i].str());
            findAndReplace(fsCode, "_PLACEHOLDER_BIAS_CONSTANTS_", biasStr.at(i).str());

            std::ostringstream fragCalc, elemAccess;
            buildFragmentCalc(fragCalc);
            buildElementAccess(elemAccess, _desc.paddingT);
            findAndReplace(fsCode, "_PLACEHOLDER_ELEMENT_ACCESS_", elemAccess.str());
            findAndReplace(fsCode, "_PLACEHOLDER_CALC_", fragCalc.str());

            if (_desc.useBatchNormalization) {
                getBatchNormalizationConstants(batchNormalizationConstants, (int) i, (int) outputChannels);
                findAndReplace(fsCode, "_PLACEHOLDER_BETA_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i].str());
                findAndReplace(fsCode, "_PLACEHOLDER_GAMMA_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 1].str());
                findAndReplace(fsCode, "_PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 2].str());
                findAndReplace(fsCode, "_PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 3].str());
            }
        }

        InferencePassGl& pass = passes[i];
        pass.source                = preDefine.str() + fsCode + postDefine.str();
        pass.inputs                = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        if (_desc.useUniformShaders) {
            for (int k = 0, j = i * channelsPerPass; k < DIV_4_ROUND_UP(outputChannels); k++) {
                glm::vec4 dummyBias = {this->biases[j + (4 * k)], this->biases[j + (4 * k) + 1], this->biases[j + (4 * k) + 2], this->biases[j + (4 * k) + 3]};
                if ((DIV_4_ROUND_UP(outputChannels) == 1) || k == 0) {
                    pass.uniforms["bias"] = dummyBias;
                } else {
                    pass.uniforms["bias[" + std::to_string(k) + "]"] = dummyBias;
                }
                if (_desc.useBatchNormalization) {
                    glm::vec4 dummyBeta       = {_desc.batchNormalization.at("beta")[j + (4 * k)], _desc.batchNormalization.at("beta")[j + (4 * k) + 1],
                                           _desc.batchNormalization.at("beta")[j + (4 * k) + 2], _desc.batchNormalization.at("beta")[j + (4 * k) + 3]};
                    glm::vec4 dummyGamma      = {_desc.batchNormalization.at("gamma")[j + (4 * k)], _desc.batchNormalization.at("gamma")[j + (4 * k) + 1],
                                            _desc.batchNormalization.at("gamma")[j + (4 * k) + 2], _desc.batchNormalization.at("gamma")[j + (4 * k) + 3]};
                    glm::vec4 dummyMovingMean = {
                        _desc.batchNormalization.at("movingMean")[j + (4 * k)], _desc.batchNormalization.at("movingMean")[j + (4 * k) + 1],
                        _desc.batchNormalization.at("movingMean")[j + (4 * k) + 2], _desc.batchNormalization.at("movingMean")[j + (4 * k) + 3]};
                    glm::vec4 dummyMovingVariance = {
                        _desc.batchNormalization.at("movingVariance")[j + (4 * k)], _desc.batchNormalization.at("movingVariance")[j + (4 * k) + 1],
                        _desc.batchNormalization.at("movingVariance")[j + (4 * k) + 2], _desc.batchNormalization.at("movingVariance")[j + (4 * k) + 3]};
                    if ((DIV_4_ROUND_UP(outputChannels) == 1) || k == 0) {
                        pass.uniforms["beta"]           = dummyBeta;
                        pass.uniforms["gamma"]          = dummyGamma;
                        pass.uniforms["movingMean"]     = dummyMovingMean;
                        pass.uniforms["movingVariance"] = dummyMovingVariance;
                    } else {
                        pass.uniforms["beta[" + std::to_string(k) + "]"]           = dummyBeta;
                        pass.uniforms["gamma[" + std::to_string(k) + "]"]          = dummyGamma;
                        pass.uniforms["movingMean[" + std::to_string(k) + "]"]     = dummyMovingMean;
                        pass.uniforms["movingVariance[" + std::to_string(k) + "]"] = dummyMovingVariance;
                    }
                }
            }
        }
        passes[i].program = InferencePassGl::FsProgram {planeCount * i, DIV_4_ROUND_UP(outputChannels)};

        if (_desc.useUniformShaders) {
            auto weightMode = _desc.weightMode;
            int sizeUBO;
            glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &sizeUBO);
            if (((_desc.kernelSize * _desc.kernelSize * 16) > static_cast<uint32_t>(sizeUBO)) && _desc.weightMode == snn::WeightAccessMethod::UNIFORM_BUFFER) {
                SNN_LOGW("Kernel too large for uniform buffer. Using SSBO instead");
                weightMode = snn::WeightAccessMethod::SSBO_BUFFER;
            }
            pass.weightMeta.clear();
            pass.weightMeta.push_back((uint32_t) 1); // 0 means Conv2D layout, 1 means DepthWise Conv2D
            pass.weightMeta.push_back((uint32_t) weightMode);
            pass.weightMeta.push_back((uint32_t)_desc.preferHp);
            pass.weightMeta.push_back((uint32_t)_desc.kernelSize);
            pass.weightMeta.push_back((uint32_t)_desc.kernelSize);
            pass.weightMeta.push_back((uint32_t)_desc.numInputPlanes);
            pass.weightMeta.push_back((uint32_t)_desc.numOutputPlanes);
            pass.weightMeta.push_back((uint32_t)channelsPerPass);
            pass.weightMeta.push_back((uint32_t)(channelsPerPass>>2) * i);
            uint32_t start = i * channelsPerPass;
            uint32_t end = start + outputChannels;
            pass.modelWeights = {_desc.weightsCvM.begin() + start, _desc.weightsCvM.begin() + end};
        }
    }

    return ret;
}

static bool oihw2hwo4i4fp16(std::vector<cv::Mat> inputWeights, std::vector<float>& outVec, int inChannels, int outChannels, int fw, int fh, int unit = 4) {
    (void) inChannels;
    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh / 2;

    outVec.clear();
    outVec.resize(alignedWeightSize);
    std::fill(outVec.begin(), outVec.end(), 0);

    uint16_t* out    = (uint16_t*) outVec.data();
    int planeSize = ROUND_UP(outChannels, unit) * fw;

    for (int b = 0; b < outChannels; ++b) {
        int b_4 = b / unit;
        int mx  = b % unit;
        for (int y = 0; y < fh; ++y) {
            for (int x = 0; x < fw; ++x) {
                int base                                 = y * planeSize;
                int inSize                               = ROUND_UP(outChannels, unit); // in the number of floats
                out[base + inSize * x + b_4 * unit + mx] = FP32::toHalf(inputWeights[b].at<float>(y * fw + x));
            }
        }
    }

    return 0;
}

#define TEXTURE_WEIGHTS

InferencePassesSptr SeparableConv2DLayerGl::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesSptr ret(new InferencePassesGl());

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassGl& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = _desc.numOutputPlanes;

    // Conv2D CS
    std::string shaderHeader;
    if (_desc.preferHp) {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION mediump\n"
                       "precision PRECISION float;\n"
                       "layout(std430) buffer;\n"
                       "#define OUTPUT_FORMAT rgba16f\n";
    } else {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION highp\n"
                       "precision PRECISION float;\n"
                       "layout(std430) buffer;\n"
                       "#define OUTPUT_FORMAT rgba32f\n";
    }
    if (_desc.numInputPlanes <= 4) {
        shaderHeader += "#define INPUT_TEXTURE_2D\n";
    }
    if (_desc.numOutputPlanes <= 4) {
        shaderHeader += "#define OUTPUT_TEXTURE_2D\n";
    }

    if (!_desc.activation.compare("relu")) {
        shaderHeader += "#define RELU\n";
    }
    else if (!_desc.activation.compare("relu6")) {
        shaderHeader += "#define RELU6\n";
    }
    else if (!_desc.activation.compare("tanh")) {
        shaderHeader += "#define TANH\n";
    }
    else if (!_desc.activation.compare("sigmoid")) {
        shaderHeader += "#define SIGMOID\n";
    }
    else if (!_desc.activation.compare("leakyRelu")) {
        shaderHeader += ("#define LEAKYRELU_VAL " + std::to_string(_desc.leakyReluAlpha) + "\n");
    }
    else if (!_desc.activation.compare("SiLU")) {
        shaderHeader += "#define SILU\n";
    }

    if (_desc.useBatchNormalization) {
        shaderHeader += "#define USE_BATCH_NORMALIZATION\n";
    }
    std::string debugLayer("[0X] Conv2D");
    std::string shaderUniforms = "#ifdef OUTPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D uOutput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"
                            "#endif\n"
                            "#ifdef INPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput;\n"
                            "#endif\n"
#ifdef TEXTURE_WEIGHTS
                            "layout(binding=2) uniform PRECISION sampler2DArray uKernel;\n";
#else
                            "const PRECISION vec4 weightMatrix1[] = vec4[]( \n"
                            "vec4(-0.0035270236, 0.0014674125, 0.0001864655, -0.0001915364),\n"
                            "vec4(-0.0032459043, -0.0016766586, 0.0002894130, 0.0010224803),\n"
                            "vec4(-0.0042301719, -0.0003793148, 0.0001178368, 0.0010762328),\n"
                            "vec4(-0.0026380727, -0.0019003409, -0.0019275651, 0.0005968372)\n"
                            ");\n";
#endif  // TEXTURE_WEIGHTS

    std::string shaderMain = loadShader(DEPTHWISE_CONV2D_CS_ASSET_NAME);

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    pass.weightMeta.clear();
    pass.weightMeta.push_back((uint32_t) 1); // 0 means Conv2D layout, 1 means DepthWise Conv2D
    pass.weightMeta.push_back((uint32_t)snn::WeightAccessMethod::TEXTURES);
    pass.weightMeta.push_back((uint32_t)_desc.preferHp);
    pass.weightMeta.push_back((uint32_t)_desc.kernelSize);
    pass.weightMeta.push_back((uint32_t)_desc.kernelSize);
    pass.weightMeta.push_back((uint32_t)_desc.numInputPlanes);
    pass.weightMeta.push_back((uint32_t)_desc.numOutputPlanes);
    std::pair<std::string, std::array<uint32_t, 3>> weightDim("2", {oc_4, (uint32_t)kernel, (uint32_t)kernel});
    pass.weightDims.insert(weightDim);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    uint32_t paddingOffsets[4];
    getPaddingOffset(paddingOffsets);
    SNN_LOGD("Padding: %d, %d, %d, %d", paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3]);

    pass.uniforms = {{"uPad", glm::ivec2(paddingOffsets[0], paddingOffsets[2])},
                     {"uKernelSize", glm::ivec2(kernel, kernel)},
                     {"uStride", glm::ivec2(stride, stride)},
                     {"uDilate", glm::ivec2(1, 1)},
                     {"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)},
                     {"uInputSize", glm::ivec3(inputWidth, inputHeight, ic_4)}};

    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferencePassGl::CsProgram {"uOutput",
                                                // div-by-N is determined by work group size defined CS program.
                                                {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    if (_desc.preferHp) {
        oihw2hwo4i4fp16(_desc.weightsCvM, pass._vecWeights, _desc.numInputPlanes, _desc.numOutputPlanes, kernel, kernel);
    } else {
        oihw2hwo4i4(_desc.weightsCvM, pass._vecWeights, _desc.numInputPlanes, _desc.numOutputPlanes, kernel, kernel);
    }

    pass._vecBias.resize(_desc.numOutputPlanes, 0.0f);
    for (size_t i = 0; i < _desc.biases.size(); i++) {
        pass._vecBias[i] = (float) _desc.biases[i];
    }

    if (_desc.useBatchNormalization) {
        std::vector<float> bnDataVector;
        std::string bnString;

        pass._vecBeta.resize(_desc.numOutputPlanes);
        bnString     = "beta";
        bnDataVector = _desc.batchNormalization.at(bnString);
        std::memcpy(pass._vecBeta.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

        pass._vecGamma.resize(_desc.numOutputPlanes);
        bnString     = "gamma";
        bnDataVector = _desc.batchNormalization.at(bnString);
        std::memcpy(pass._vecGamma.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

        pass._vecMean.resize(_desc.numOutputPlanes);
        bnString     = "movingMean";
        bnDataVector = _desc.batchNormalization.at(bnString);
        std::memcpy(pass._vecMean.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

        pass._vecVariance.resize(_desc.numOutputPlanes);
        bnString     = "movingVariance";
        bnDataVector = _desc.batchNormalization.at(bnString);
        std::memcpy(pass._vecVariance.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);
    }

    return ret;
}
