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
#include "dp.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <ios>

using namespace snn;
using namespace snn::dp;

#define CLAMPED_PADDING 1

static constexpr const char* DEPTHWISE_CONV2D_FS_ASSET_NAME = "shaders/shadertemplate_fs_depthwise_RGBA.glsl";
static constexpr const char* DEPTHWISE_CONV2D_CS_ASSET_NAME = "shaders/3rdparty/shadertemplate_cs_separableconvolution.glsl";

SeparableConv2DLayer::SeparableConv2DLayer(SeparableConv2DDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {
    this->weightSamplers.allocate(1);
    snn::ColorFormat weightFormat;
    if (_desc.preferrHalfPrecision) {
        weightFormat = snn::ColorFormat::RGBA16F;
    } else {
        weightFormat = snn::ColorFormat::RGBA32F;
    }

    if (_desc.numInputPlanes <= 2048) {
        _desc.useUniformShaders = false;
        _desc.weightMode        = snn::WeightAccessMethod::CONSTANTS;
    }

    if (_desc.weightMode == snn::WeightAccessMethod::CONSTANTS) {
        _desc.useUniformShaders = false;
    }

    int sizeUBO;
    glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &sizeUBO);
    if (((_desc.kernelSize * _desc.kernelSize * 16) > sizeUBO) && _desc.weightMode == snn::WeightAccessMethod::UNIFORM_BUFFER) {
        SNN_LOGW("Kernel too large for uniform buffer. Using SSBO instead");
        _desc.weightMode = snn::WeightAccessMethod::SSBO_BUFFER;
    }

    switch (_desc.weightMode) {
    case snn::WeightAccessMethod::TEXTURES:
        this->weightTextures.allocate(DIV_4_ROUND_UP(_desc.numOutputPlanes));
        break;

    case snn::WeightAccessMethod::UNIFORM_BUFFER:
        this->weightUniformBuffers.allocate(DIV_4_ROUND_UP(_desc.numOutputPlanes));
        break;

    case snn::WeightAccessMethod::SSBO_BUFFER:
        this->weightSSBOBuffers.allocate(DIV_4_ROUND_UP(_desc.numOutputPlanes));
        break;

    default:
        break;
    }

    for (std::size_t i = 0; i < DIV_4_ROUND_UP(_desc.numOutputPlanes); i++) {
        switch (_desc.weightMode) {
        case snn::WeightAccessMethod::TEXTURES:
            this->weightTextures[i].allocate2D(weightFormat, _desc.kernelSize, _desc.kernelSize, 1);
            break;

        case snn::WeightAccessMethod::UNIFORM_BUFFER: {
            uint32_t count = 4 * _desc.kernelSize * _desc.kernelSize;
            if (_desc.preferrHalfPrecision) {
                std::vector<uint16_t> dummyVal(count);
                this->weightUniformBuffers[i].allocate(count, dummyVal.data());
            } else {
                std::vector<float> dummyVal(count);
                this->weightUniformBuffers[i].allocate(count, dummyVal.data());
            }
            break;
        }

        case snn::WeightAccessMethod::SSBO_BUFFER: {
            uint32_t count = 4 * _desc.kernelSize * _desc.kernelSize;
            if (_desc.preferrHalfPrecision) {
                std::vector<uint16_t> dummyVal(count);
                this->weightSSBOBuffers[i].allocate(count, dummyVal.data());
            } else {
                std::vector<float> dummyVal(count);
                this->weightSSBOBuffers[i].allocate(count, dummyVal.data());
            }
            break;
        }

        default:
            break;
        }
        //  SNN_LOGI("Created weight Texture: %u, %u", this->weights[i].id(), this->weights[i].target());
        // std::cout << this->weights[i].id() << ", " << this->weights[i].target() << std::endl;
    }

    this->kernelTexture.allocate2DArray(weightFormat, DIV_4_ROUND_UP(_desc.numOutputPlanes), _desc.kernelSize, _desc.kernelSize, _desc.kernelSize, 1);

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
    if (_desc.preferrHalfPrecision) {
        snn::convertToMediumPrecision(this->biases);
    }
}

void SeparableConv2DLayer::setTextureWeights() const {
    std::vector<float> weightVal(4 * _desc.kernelSize * _desc.kernelSize, 0.0);
    for (std::size_t filter = 0; filter < _desc.numOutputPlanes; filter++) {
        SNN_LOGD("Updating weights for weight texture: %u, %u", this->weightTextures[filter].id(), this->weightTextures[filter].target());
        for (std::size_t i = 0; i < _desc.kernelSize; i++) {
            for (std::size_t j = 0; j < _desc.kernelSize; j++) {
                std::size_t weightValIdx = (4 * _desc.kernelSize * i) + (4 * j) + (filter % 4);
                SNN_LOGD("Filling IDX %u in buffer with size %u", weightValIdx, weightVal.size());
                if (_desc.preferrHalfPrecision) {
                    weightVal[weightValIdx] = snn::convertToMediumPrecision(_desc.weights[filter].at<float>(i, j));
                } else {
                    weightVal[weightValIdx] = _desc.weights[filter].at<float>(i, j);
                }
                // weightVal[weightValIdx] = _desc.weights[idx].at<float>(i, j);
            }
        }
        if ((filter + 1) % 4 == 0) {
            this->weightTextures[filter / 4].bind(0);
            this->weightTextures[filter / 4].setPixels(0, 0, 0, _desc.kernelSize, _desc.kernelSize, 0, weightVal.data());
            glFinish();
            this->weightTextures[filter / 4].unbind();
            // std::cout << "Weights updated at: " << filter << " ID: " << this->weights[filter].id() << " Target: " << this->weights[filter].target() <<
            // std::endl << std::endl;
            weightVal.clear();
            weightVal.resize(_desc.kernelSize * _desc.kernelSize * 4, 0.0);
        }
    }
    if (!weightVal.empty() && _desc.numOutputPlanes % 4 != 0) {
        // for (auto weightValue : weightVal) {
        //     SNN_LOGI("%f", weightValue);
        // }
        uint32_t extraChannels = ROUND_UP_DIV_4(_desc.numOutputPlanes) - _desc.numOutputPlanes;
        for (uint32_t i = 0; i < _desc.kernelSize * _desc.kernelSize * extraChannels; i++) {
            weightVal.push_back(0.0f);
        }
        this->weightTextures[DIV_4_ROUND_UP(_desc.numOutputPlanes)].bind(0);
        this->weightTextures[DIV_4_ROUND_UP(_desc.numOutputPlanes)].setPixels(0, 0, 0, _desc.kernelSize, _desc.kernelSize, 0, weightVal.data());
        glFinish();
        this->weightTextures[DIV_4_ROUND_UP(_desc.numOutputPlanes)].unbind();
        // std::cout << "Weights updated at: " << filter << " ID: " << this->weights[filter].id() << " Target: " << this->weights[filter].target() << std::endl
        // << std::endl;
    }
}

void SeparableConv2DLayer::setBufferWeights() const {
    uint8_t byteSize = _desc.preferrHalfPrecision ? 2 : 4;
    std::vector<uint8_t> weightVal(byteSize * 4 * _desc.kernelSize * _desc.kernelSize, 0);
    for (std::size_t filter = 0; filter < _desc.numOutputPlanes; filter++) {
        for (std::size_t i = 0; i < _desc.kernelSize; i++) {
            for (std::size_t j = 0; j < _desc.kernelSize; j++) {
                std::size_t weightValIdx = (byteSize * 4 * _desc.kernelSize * i) + (4 * j) + (filter % 4);
                std::vector<uint8_t> byteRep;
                snn::getByteRepresentation(_desc.weights[filter].at<float>(j, i), byteRep, _desc.preferrHalfPrecision);
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

void SeparableConv2DLayer::buildElementAccess(std::ostream& stream, std::string padding) const {
    uint32_t paddingOffsets[4];
    this->getPaddingOffset(paddingOffsets);

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

    const char channelVal[4]      = {'r', 'g', 'b', 'a'};
    const char channelValUpper[4] = {'R', 'G', 'B', 'A'};
    int channelsPerPass           = 4;
    SNN_LOGD("%d", channelsPerPass);
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
    int planeCount = DIV_4_ROUND_UP(channelsPerPass);

    // std::cout << "Padding: " << paddingStr << std::endl;

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
#endif
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
#endif
            }
        }
    }
}

void SeparableConv2DLayer::buildFragmentCalc(std::ostringstream& stream) const {
    std::string baseTabLevel = "\t\t";
    uint32_t offsets[4];
    this->getPaddingOffset(offsets);
    if (offsets[0] == 0) {
        baseTabLevel += "\t";
    }
    const char channelVal[4]      = {'r', 'g', 'b', 'a'};
    const char channelValUpper[4] = {'R', 'G', 'B', 'A'};
    int channelsPerPass           = 4;
    // SNN_LOGD("%d", channelsPerPass);
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
    // if (offsets[0] == 0) {
    //     stream << baseTabLevel << "}" << std::endl;
    // }
}

void SeparableConv2DLayer::getWeightConstants(std::ostringstream& weightConstants, const std::vector<float>& vWeightMatrices, int idxInput4or8Chunk) const {
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

void SeparableConv2DLayer::getBiasConstants(std::vector<std::ostringstream>& biasConstants, uint32_t numShaderPasses) const {
    uint32_t outputPlanes = _desc.numOutputPlanes;
    int channelsPerPass   = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::SINGLE_PLANE:
        channelsPerPass = 4;
        break;

    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;

    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;

    default:
        channelsPerPass = 4;
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

        // Force the Bias to 0. T.B.D.
        // biasConstants.at(i) << "const FLOAT_PRECISION vec4 bias = vec4("
        //             << "0.0, "
        //             << "0.0, "
        //             << "0.0, "
        //             << "0.0);";
        // continue;
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

void SeparableConv2DLayer::getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass,
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

    default:
        channelsPerPass = 4;
        break;
    }
    int planeCount = DIV_4_ROUND_UP(channelsPerPass);
    int index      = shaderPass * 4;
    // SNN_LOGD("Batch Norm Index: %d", index);
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

void SeparableConv2DLayer::getPaddingOffset(uint32_t (&offsets)[4]) const {
    std::string paddingT = this->_desc.paddingT;
    std::string paddingB = this->_desc.paddingB;
    std::string paddingL = this->_desc.paddingL;
    std::string paddingR = this->_desc.paddingR;
    bool isdigit         = std::all_of(paddingT.begin(), paddingT.end(), ::isdigit);
    if (isdigit) {
        offsets[0] = std::stoul(paddingT);
        offsets[1] = std::stoul(paddingB);
        offsets[2] = std::stoul(paddingL);
        offsets[3] = std::stoul(paddingR);
    } else {
        if (paddingT == "valid" || paddingT == "none") {
            offsets[0] = 0;
            offsets[1] = 0;
            offsets[2] = 0;
            offsets[3] = 0;
        } else {
            if (_desc.kernelSize > 1) {
                offsets[0] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[1] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[2] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[3] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                if (_desc.kernelSize % 2 == 0) {
                    offsets[0] = offsets[0] - 1;
                    offsets[2] = offsets[2] - 1;
                }
            } else {
                offsets[0] = 0;
                offsets[1] = 0;
                offsets[2] = 0;
                offsets[3] = 0;
            }
        }
    }
}

void SeparableConv2DLayer::buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options) const {
    uint32_t offsets[4];
    this->getPaddingOffset(offsets);
    stream << "#version 320 es\n";
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    stream << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << std::endl;
    stream << "#define INPUT_WIDTH " << options.desiredInput.width << std::endl;
    stream << "#define INPUT_HEIGHT " << options.desiredInput.height << std::endl;
    stream << "#define NUM_KERNEL_SIZE " << _desc.kernelSize << std::endl;
    stream << "#define NUM_STRIDE " << _desc.stride << std::endl;
    stream << "#define PAD_VALUE 0.0f" << std::endl;
#if CLAMPED_PADDING == 1
    stream << "#define CLAMPED_PADDING" << std::endl;
#endif
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
void SeparableConv2DLayer::buildFragmentPostDefine(std::ostream& stream) const {
    // LOGI("Debug-1:: leakyRelualpha is %f", _leakyReluAlpha);
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

ShaderLayer::GLSLShaders SeparableConv2DLayer::createFS(const LayerGenOptions& options) const {
    std::ostringstream fsFile;
    std::string fileName = "/shadertemplate_fs_depthwise_RGBA";
    fsFile << "shaders" << fileName << ".glsl";

    if (_desc.useUniformShaders) {
        switch (_desc.weightMode) {
        case snn::WeightAccessMethod::CONSTANTS:
            break;
        case snn::WeightAccessMethod::TEXTURES:
            this->setTextureWeights();
            break;
        case snn::WeightAccessMethod::UNIFORM_BUFFER:
            this->setBufferWeights();
            break;
        case snn::WeightAccessMethod::SSBO_BUFFER:
            this->setBufferWeights();
            break;
        default:
            break;
        }
    }

    int channelsPerPass = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::SINGLE_PLANE:
        channelsPerPass = 4;
        break;

    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;

    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;

    default:
        channelsPerPass = 4;
        break;
    }

    auto numShaderPasses = DIV_AND_ROUND_UP(_desc.numInputPlanes, channelsPerPass);
    int planes           = DIV_4_ROUND_UP(channelsPerPass);
    std::vector<std::ostringstream> weightConstants(planes * numShaderPasses);

    // TODO: This code is unoptimized; we can do this transposal in place instead
    auto& weightMatrices = _desc.weights;
    int nWeightStride    = _desc.kernelSize * _desc.kernelSize;

    std::vector<float> vWeightMatrices(nWeightStride * _desc.numInputPlanes, 0.f);
    std::vector<std::ostringstream> biasStr(numShaderPasses);

    if (!_desc.useUniformShaders) {
        for (int idxInput = 0, startOffset = 0; idxInput < static_cast<int>(_desc.numInputPlanes); ++idxInput, startOffset += nWeightStride) {
            memcpy(&vWeightMatrices[startOffset], weightMatrices[idxInput].data, _desc.kernelSize * _desc.kernelSize * sizeof(float));
        }

        for (int i = 0; i < static_cast<int>(planes * numShaderPasses); ++i) {
            getWeightConstants(weightConstants[i], vWeightMatrices, i);
        }
        getBiasConstants(biasStr, numShaderPasses);
        vWeightMatrices.clear();
    }

    GLSLShaders ret;
    ret.passes.resize(numShaderPasses);
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

        int numBatchNormParameters = 4;
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
            this->buildFragmentCalc(fragCalc);
            this->buildElementAccess(elemAccess, _desc.paddingT);
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

        InferenceGraph::Pass& pass = ret.passes[i];
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

            switch (_desc.weightMode) {
            case snn::WeightAccessMethod::TEXTURES:
                pass.weights = std::vector<const gl::TextureObject*>();
                break;

            case snn::WeightAccessMethod::UNIFORM_BUFFER:
                pass.weights = std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>();
                break;

            case snn::WeightAccessMethod::SSBO_BUFFER:
                pass.weights = std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>();
                break;

            default:
                break;
            }
            int jIndex = i * planeCount;
            std::visit(match {[&](std::vector<const gl::TextureObject*>& weightTextures) {
                                for (std::size_t k = 0; k < planeCount; k++) {
                                    // std::cout << "Weights at :" << jIndex + k << " ID: " << this->weightTextures[jIndex+k].id() << " Target: " <<
                                    // this->weightTextures[jIndex+k].target() << std::endl;
                                    weightTextures.push_back(&this->weightTextures[jIndex + k]);
                                }
                            },
                            [&](std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>& weightBuffers) {
                                for (std::size_t k = 0; k < outputChannels; k++) {
                                    weightBuffers.push_back(&this->weightUniformBuffers[jIndex + k]);
                                }
                            },
                            [&](std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>& weightBuffers) {
                                for (std::size_t k = 0; k < outputChannels; k++) {
                                    weightBuffers.push_back(&this->weightSSBOBuffers[jIndex + k]);
                                }
                            }},
                        pass.weights);
        }
        ret.passes[i].program = InferenceGraph::Pass::FsProgram {planeCount * i, DIV_4_ROUND_UP(outputChannels)};
    }

    return ret;
}

InferenceGraph::Transform SeparableConv2DLayer::getOutputScaleDimAdjustment() const {
    uint32_t offset[4];
    this->getPaddingOffset(offset);
    float scale       = 1 / static_cast<float>(_desc.stride);
    float translation = 0.0f;
    if (_desc.kernelSize % 2 != 0) {
        translation = 1 + (static_cast<float>(offset[0] + offset[1]) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    } else {
        translation = 1 + (static_cast<float>(offset[0] + offset[1] - 1) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    }
    return {0, scale, scale, translation, translation};
}

void SeparableConv2DLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    uint32_t paddingOffsets[4];
    this->getPaddingOffset(paddingOffsets);
    // SNN_LOGI("%%%%%%%% %s:%d Padding:%d, %d, %d, %d\n", __FUNCTION__,__LINE__, paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3]);
    for (auto& dim : inputDims) {
        // SNN_LOGI("%%%%%%%% %s:%d dim:%d, %d, %d\n", __FUNCTION__,__LINE__, dim.width, dim.height, dim.depth);
        width  = (dim.width - _desc.kernelSize + paddingOffsets[0] + paddingOffsets[2]) / _desc.stride + 1;
        height = (dim.height - _desc.kernelSize + paddingOffsets[1] + paddingOffsets[3]) / _desc.stride + 1;
        depth  = dim.depth;
        // SNN_LOGI("%%%%%%%% %s:%d dim:%d, %d, %d\n", __FUNCTION__,__LINE__, width, height, depth);
        break;
    }
}

static bool oihw2hwo4i4(std::vector<cv::Mat> inputWeights, std::vector<float>& outVec, int inChannels, int outChannels, int fw, int fh, int unit = 4) {
    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh;

    // SNN_LOGI("Test:%s:%d, %d, %d, %d, %d, all: %d %d\n",__FUNCTION__,__LINE__, inChannels, outChannels, fw, fh, alignedWeightSize, inputWeights.size());

    outVec.clear();
    outVec.resize(alignedWeightSize);
    std::fill(outVec.begin(), outVec.end(), 0);

    float* out    = (float*) outVec.data();
    int planeSize = ROUND_UP(outChannels, unit) * fw;
    memset(out, 0, alignedWeightSize * sizeof(float));
    for (int b = 0; b < outChannels; ++b) {
        int b_4 = b / unit;
        int mx  = b % unit;
        for (int y = 0; y < fh; ++y) {
            for (int x = 0; x < fw; ++x) {
                int base                                 = y * planeSize;
                int inSize                               = ROUND_UP(outChannels, unit); // in the number of floats
                out[base + inSize * x + b_4 * unit + mx] = inputWeights[b].at<float>(y * fw + x);
                // SNN_LOGI("new: %d: %f\n",base + inSize * x + b_4 * unit + mx, inputWeights[b].at<float>(y*fw+x));
            }
        }
    }
    return 0;
}

#define TEXTURE_WEIGHTS

ShaderLayer::GLSLShaders SeparableConv2DLayer::createCS(const LayerGenOptions& options) const {
    (void) options;

    GLSLShaders ret;
    // auto& desc = getDesc();
    // (void)desc;

    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(1);

    InferenceGraph::Pass& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t inputDepth  = inputDims[0].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    snn::dp::GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = _desc.numOutputPlanes;

    // Conv2D CS
    // printf("Test:%s:%d\n",__FUNCTION__,__LINE__);
    string shaderHeader;
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

    if (!_desc.activation.compare("relu6")) {
        shaderHeader += "#define RELU6\n";
    }

    if (!_desc.activation.compare("tanh")) {
        shaderHeader += "#define TANH\n";
    }

    if (!_desc.activation.compare("sigmoid")) {
        shaderHeader += "#define SIGMOID\n";
    }

    if (!_desc.activation.compare("leakyRelu")) {
        shaderHeader += ("#define LEAKYRELU_VAL " + std::to_string(_desc.leakyReluAlpha) + "\n");
    }

    if (!_desc.activation.compare("SiLU")) {
        shaderHeader += "#define SILU\n";
    }

    if (_desc.useBatchNormalization) {
        shaderHeader += "#define USE_BATCH_NORMALIZATION\n";
    }
    string debugLayer("[0X] Conv2D");
    /*
    if (this->name.find(debugLayer) != std::string::npos) {
        shaderHeader += "#define CONV2D_DEBUG\n";
    }
    */
    // SNN_LOGI("Test:%s:%d, %s\n",__FUNCTION__,__LINE__, shaderHeader.c_str());
    string shaderUniforms = "#ifdef OUTPUT_TEXTURE_2D\n"
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
#endif

    string shaderMain;

    if (1) {
        shaderMain = "layout(binding=4) readonly buffer bias{\n"
                    "    vec4 data[];\n"
                    "} uBias;\n"
                    "layout(binding=5) readonly buffer beta{\n"
                    "    vec4 data[];\n"
                    "} uBeta;\n"
                    "layout(binding=6) readonly buffer gamma{\n"
                    "    vec4 data[];\n"
                    "} uGamma;\n"
                    "layout(binding=7) readonly buffer mean{\n"
                    "    vec4 data[];\n"
                    "} uMean;\n"
                    "layout(binding=8) readonly buffer variance{\n"
                    "    vec4 data[];\n"
                    "} uVariance;\n"
                    "layout(location=4) uniform ivec2 uPad;\n"
                    "layout(location=5) uniform ivec2 uKernelSize;\n"
                    "layout(location=6) uniform ivec2 uStride;\n"
                    "layout(location=7) uniform ivec2 uDilate;\n"
                    "// layout(location=8) uniform ivec2 uOffset;\n"
                    "// layout(location=9) uniform float uReluRate;\n"
                    "layout(location=10) uniform ivec3 uOutputSize;\n"
                    "layout(location=11) uniform ivec3 uInputSize;\n"
                    "#define UP_DIV(x, y) (((x)+(y)-1)/(y))\n"
                    "layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;\n"
                    "void main()\n"
                    "{\n"
                    "    ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(1, 1, 1);\n"
                    "    ivec3 outputSize = uOutputSize;\n"
                    "    if (all(lessThan(pos, outputSize)))\n"
                    "    {\n"
                    "        int KSIZE_Y = uKernelSize.y;\n"
                    "        int KSIZE_X = uKernelSize.x;\n"
                    "        ivec3 inputSize = uInputSize;\n"
                    "        ivec2 s0 = pos.xy*uStride-uPad;\n"
                    "        int fx, fy, fz;\n"
                    "        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));\n"
                    "        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));\n"
                    "        vec4 color = uBias.data[pos.z];\n"
                    "        for (fy=sfxy.y; fy<efxy.y; ++fy)\n"
                    "        {\n"
                    "            int sy = fy*uDilate.y + s0.y;\n"
                    "            for (fx=sfxy.x; fx<efxy.x; ++fx)\n"
                    "            {\n"
                    "                int sx1 = fx*uDilate.x + s0.x;\n"
                    "                vec4 k = texelFetch(uKernel, ivec3(pos.z, fx, fy), 0);\n"
                    "                    #ifdef INPUT_TEXTURE_2D\n"
                    "                    color  += k*imageLoad(uInput, ivec2(sx1, sy));\n"
                    "                    #else\n"
                    "                    color  += k*imageLoad(uInput, ivec3(sx1, sy, pos.z));\n"
                    "                    #endif\n"
                    "            }\n"
                    "        }\n"
                    "        #ifdef USE_BATCH_NORMALIZATION \n"
                    "        vec4 movingVariance = uVariance.data[pos.z]; \n"
                    "        vec4 movingMean = uMean.data[pos.z]; \n"
                    "        vec4 gamma = uGamma.data[pos.z]; \n"
                    "        vec4 beta = uBeta.data[pos.z]; \n"
                    // "        vec4 sqrtVar = sqrt(movingVariance + vec4(0.00000001f)); \n"
                    "        vec4 sqrtVar = sqrt(movingVariance + vec4(0.001f)); \n"
                    "        sqrtVar = max(sqrtVar, vec4(0.0001f)); \n"
                    "        color = ((gamma/sqrtVar) * (color - movingMean)) + beta;   \n"
                    "        #endif\n"
                    "        #ifdef RELU\n"
                    "        color = max(color, vec4(0));\n"
                    "        #endif\n"
                    "        #ifdef RELU6\n"
                    "        color = clamp(color, vec4(0), vec4(6));\n"
                    "        #endif\n"
                    "        #ifdef TANH\n"
                    "        color = tanh(color);\n"
                    "        #endif\n"
                    "        #ifdef SIGMOID\n"
                    "        color  = vec4(1.0f)/(vec4(1.0f)+ exp(-color));\n"
                    "        #endif\n"
                    "        #ifdef LEAKYRELU_VAL\n"
                    "        color   = max(color,  (color * vec4(LEAKYRELU_VAL)));\n"
                    "        #endif\n"
                    "        #ifdef SILU\n"
                    "        color    = color  * vec4(1.0f)/(vec4(1.0f)+ exp(-color));\n"
                    "        #endif\n"
                    "        #ifdef OUTPUT_TEXTURE_2D\n"
                    "        imageStore(uOutput, ivec2(pos.x+0, pos.y), color);\n"
                    "        #else\n"
                    "        imageStore(uOutput, ivec3(pos.x+0, pos.y, pos.z), color);\n"
                    "        #endif\n"
                    "    }\n"
                    "}\n";
        shaderMain = loadShader(DEPTHWISE_CONV2D_CS_ASSET_NAME);
    }

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    // std::vector<int> mLocalSize{8, 8, 2};

    SNN_LOGD("Test:%s:%d, %d, %d, %d, %d\n", __FUNCTION__, __LINE__, kernel, stride, ic_4, oc_4);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");
    // SNN_LOGI("Test:%s:%d, %s\n",__FUNCTION__,__LINE__, shaderHeader.c_str());

    if (1) {
        uint32_t paddingOffsets[4];
        this->getPaddingOffset(paddingOffsets);
        SNN_LOGD("%s:%d, Padding: %d, %d, %d, %d\n", __FUNCTION__, __LINE__, paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3]);

        pass.uniforms = {{"uPad", glm::ivec2(paddingOffsets[0], paddingOffsets[2])},
                         {"uKernelSize", glm::ivec2(kernel, kernel)},
                         {"uStride", glm::ivec2(stride, stride)},
                         {"uDilate", glm::ivec2(1, 1)},
                         {"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)},
                         {"uInputSize", glm::ivec3(inputWidth, inputHeight, ic_4)}};
    }
    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferenceGraph::Pass::CsProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    // SNN_LOGI("%s:%d, input:%d:%d:%d, output:%d:%d:%d\n",__FUNCTION__,__LINE__, inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    oihw2hwo4i4(_desc.weights, pass._vecWeights, _desc.numInputPlanes, _desc.numOutputPlanes, kernel, kernel);
    float* kernelBuf = pass._vecWeights.data();

    if (this->name.find(debugLayer) != std::string::npos) {
        for (unsigned int i = 0; i < ROUND_UP(_desc.numOutputPlanes, unit) * kernel * kernel * ROUND_UP(_desc.numInputPlanes, unit); i++) {
            printf("%s:%d: %d, %f\n", __FUNCTION__, __LINE__, i, *(kernelBuf + i));
        }
    }

    this->kernelTexture.bind(0);

    // SNN_LOGI("Updating weights for: %s", this->name.c_str());

    int planeSize = oc_4 * kernel * unit;
    for (int i = 0; i < kernel; i++) {
        // SNN_LOGI("%s:%d: %d, %d\n",__FUNCTION__,__LINE__, i, planeSize*i);
        this->kernelTexture.setPixels(i, 0, 0, 0, oc_4, kernel, 0, kernelBuf + planeSize * i);
    }
    // if (this->name.find(debugLayer) != std::string::npos) {
    // readTexture(kernel*kernel*ic_4, kernelTexture.id(),  ic_4, kernel, kernel, 1);
    // }
    this->kernelTexture.unbind();

    // SNN_LOGI("%s:%d, kernel texture %d:%d\n",__FUNCTION__,__LINE__, kernelTexture.id(), kernelTexture.target());

#ifdef TEXTURE_WEIGHTS
    pass.weightUniformTags.resize(1);
    pass.weightUniformTags[0] = "uKernel";
    pass.weights              = std::vector<const gl::TextureObject*>(1, &kernelTexture);
#endif

    // pass._vecWeights.resize(inputWidth*outputWidth);
    // pass._boWeights.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
    // pass._boWeights->allocate(inputWidth*outputWidth*4, pass._vecWeights.data());
    // float *destWeight = (float *)pass._boWeights->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

    // unsigned int width = _denseDesc.weights[0].size();
    // int k = 0;
    // for (size_t i = 0; i < _denseDesc.weights.size(); i++){
    //     for (size_t j = 0; j < width; j++)  {
    //         //printf("%zu:%zu: %zu, %f\n",i, j, i * inputWidth + j, _denseDesc.weights[i][j]);
    //         *(destWeight + k) =  _denseDesc.weights[i][j];
    //         k++;
    //     }
    // }
    // pass._boWeights->unmap();

    pass._vecBias.resize(_desc.numOutputPlanes);
    pass._boBias.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
    pass._boBias->allocate(_desc.numOutputPlanes, pass._vecBias.data());
    float* destBias = (float*) pass._boBias->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

    // SNN_LOGI("Bias:%s:%d %zu\n",__FUNCTION__,__LINE__, _desc.biases.size());

    for (size_t i = 0; i < _desc.biases.size(); i++) {
        *(destBias + i) = (float) _desc.biases[i];
        // printf("%zu:%f\n",i, _desc.biases[i]);
    }

    pass._boBias->unmap();
    pass.ssboMap = {{4, pass._boBias->getId()}};

    if (_desc.useBatchNormalization) {
        float* bnDataDest;
        std::vector<float> bnDataVector;
        std::string bnString;
        pass._vecBeta.resize(_desc.numOutputPlanes);
        pass._bnBeta.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnBeta->allocate(_desc.numOutputPlanes, pass._vecBeta.data());
        bnDataDest   = (float*) pass._bnBeta->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "beta";
        bnDataVector = _desc.batchNormalization.at(bnString);
        // SNN_LOGI("Beta:%s:%d %zu\n",__FUNCTION__,__LINE__, bnDataVector.size());

        // for (size_t i = 0; i < bnDataVector.size(); i++) {
        //    *(bnDataDest + i) = snn::convertToMediumPrecision(bnDataVector[i]);
        //    //printf("%zu:%f\n",i, bnDataVector[i]);
        // }

        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnBeta->unmap();
        pass.ssboMap[5] = pass._bnBeta->getId();

        pass._vecGamma.resize(_desc.numOutputPlanes);
        pass._bnGamma.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnGamma->allocate(_desc.numOutputPlanes, pass._vecGamma.data());
        bnDataDest   = (float*) pass._bnGamma->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "gamma";
        bnDataVector = _desc.batchNormalization.at(bnString);
        // SNN_LOGI("Gamma:%s:%d %zu\n",__FUNCTION__,__LINE__, bnDataVector.size());

        // for (size_t i = 0; i < bnDataVector.size(); i++) {
        //     *(bnDataDest + i) = snn::convertToMediumPrecision(bnDataVector[i]);
        //     //printf("%zu:%f\n",i, bnDataVector[i]);
        // }

        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnGamma->unmap();
        pass.ssboMap[6] = pass._bnGamma->getId();

        pass._vecMean.resize(_desc.numOutputPlanes);
        pass._bnMean.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnMean->allocate(_desc.numOutputPlanes, pass._vecMean.data());
        bnDataDest   = (float*) pass._bnMean->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "movingMean";
        bnDataVector = _desc.batchNormalization.at(bnString);
        // SNN_LOGI("Mean:%s:%d %zu\n",__FUNCTION__,__LINE__, bnDataVector.size());

        // for (size_t i = 0; i < bnDataVector.size(); i++) {
        //     *(bnDataDest + i) = snn::convertToMediumPrecision(bnDataVector[i]);
        //     //printf("%zu:%f\n",i, bnDataVector[i]);
        // }

        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnMean->unmap();
        pass.ssboMap[7] = pass._bnMean->getId();

        pass._vecVariance.resize(_desc.numOutputPlanes);
        pass._bnVariance.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnVariance->allocate(_desc.numOutputPlanes, pass._vecVariance.data());
        bnDataDest = (float*) pass._bnVariance->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString   = "movingVariance";
        // SNN_LOGI("Variance:%s:%d %zu\n",__FUNCTION__,__LINE__, bnDataVector.size());
        bnDataVector = _desc.batchNormalization.at(bnString);

        // for (size_t i = 0; i < bnDataVector.size(); i++) {
        //     *(bnDataDest + i) = snn::convertToMediumPrecision(bnDataVector[i]);
        //     //printf("%zu:%f\n",i, bnDataVector[i]);
        // }

        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnVariance->unmap();
        pass.ssboMap[8] = pass._bnVariance->getId();
    }

    return ret;
}
