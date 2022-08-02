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
#include <cctype>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <ios>

using namespace snn;
using namespace snn::dp;
using namespace std;

#define MAX_NUM_TEXTURES_PER_LAYER 100;

static constexpr const char* AVGPOOL2D_FS_ASSET_NAME = "shaders/shadertemplate_fs_avgpooling2d.glsl";
static constexpr const char* AVGPOOL2D_CS_ASSET_NAME = "shaders/shadertemplate_cs_avgpooling2d.glsl";

static constexpr const char* INDENT = "    ";

InferenceGraph::Transform AdaptiveAvgPool2dLayer::getOutputScaleDimAdjustment() const {
    float scale, translation;
    scale = 1.0f / _desc.stride;
    if (this->_desc.padding == "0" || this->_desc.padding == "none" || this->_desc.padding == "valid") {
        translation = 1.0f - (static_cast<float>(this->_desc.kernelSize) / static_cast<float>(this->_desc.stride));
    } else {
        translation = 1.0f - 1.0f / static_cast<float>(this->_desc.stride);
    }
    return {0, scale, scale, translation, translation};
}

void AdaptiveAvgPool2dLayer::buildPreDefine(std::ostringstream& stream, const GenericModelLayer::LayerGenOptions& options,
                                            const std::string& shaderFilePath) const {
    stream << "#version 320 es\n";
    stream << "// " << shaderFilePath << "\n";
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << "\n";
    stream << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << "\n";
    stream << "#define INPUT_WIDTH " << options.desiredInput.width << "\n";
    stream << "#define INPUT_HEIGHT " << options.desiredInput.height << "\n";
    stream << "#define NUM_STRIDE " << options.desiredInput.width << "\n";
    stream << "#define N_DIMS " << options.desiredInput.width * options.desiredInput.height << "\n";
    stream << "#define CLAMPED_PADDING\n";

    if (_desc.numInputPlanes <= 4) {
        stream << "#define INPUT_TEXTURE_2D\n";
    }
    stream << std::endl;
}

void AdaptiveAvgPool2dLayer::buildTextureDefLogic(std::ostream& stream, const LayerGenOptions& options, uint32_t inputSliceIndex) const {
    std::vector<float> offsetsW(options.desiredInput.width), offsetsH(options.desiredInput.height);
    std::iota(offsetsW.begin(), offsetsW.end(), -0.5);
    std::iota(offsetsH.begin(), offsetsH.end(), -0.5);

    for (uint32_t i = 0; i < options.desiredInput.width; i++) {
        for (uint32_t j = 0; j < options.desiredInput.height; j++) {
            stream << "\tvec2 texCoord_" << options.desiredInput.width * i + j + 1 << " = (vec2(baseCoord) + ";
            stream << "vec2(" << offsetsH.at(j) << ", " << offsetsW.at(i) << ")) / vec2(maxUV);" << std::endl;
        }
    }

    stream << std::endl;
    switch (_desc.mrtMode) {
    case snn::MRTMode::QUAD_PLANE:
        stream << "#if PLANE_COUNT > 3\n";
        stream << "\tint layer3 = " << inputSliceIndex + 3 << ";" << std::endl;
        stream << "\tint layer2 = " << inputSliceIndex + 2 << ";" << std::endl;
        for (uint32_t i = 0; i < options.desiredInput.width; i++) {
            for (uint32_t j = 0; j < options.desiredInput.height; j++) {
                int linearDim = options.desiredInput.width * i + j;
                stream << "\tFLOAT_PRECISION vec4 t" << linearDim << "_3 = TEXTURE(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer3));\n";
            }
        }
        for (uint32_t i = 0; i < options.desiredInput.width; i++) {
            for (uint32_t j = 0; j < options.desiredInput.height; j++) {
                int linearDim = options.desiredInput.width * i + j;
                stream << "\tFLOAT_PRECISION vec4 t" << linearDim << "_2 = TEXTURE(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer2));\n";
            }
        }
        stream << "#endif\n";
        [[fallthrough]];

    case snn::MRTMode::DOUBLE_PLANE:
        stream << "#if PLANE_COUNT > 1\n";
        stream << "\tint layer1 = " << inputSliceIndex + 1 << ";" << std::endl;
        for (uint32_t i = 0; i < options.desiredInput.width; i++) {
            for (uint32_t j = 0; j < options.desiredInput.height; j++) {
                int linearDim = options.desiredInput.width * i + j;
                stream << "\tFLOAT_PRECISION vec4 t" << linearDim << "_1 = TEXTURE(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer1));\n";
            }
        }
        stream << "#endif\n";
        [[fallthrough]];

    case snn::MRTMode::SINGLE_PLANE:
        stream << "\tint layer = " << inputSliceIndex << ";" << std::endl;
        for (uint32_t i = 0; i < options.desiredInput.width; i++) {
            for (uint32_t j = 0; j < options.desiredInput.height; j++) {
                int linearDim = options.desiredInput.width * i + j;
                stream << "\tFLOAT_PRECISION vec4 t" << linearDim << "_0 = TEXTURE(inputTextures, vec3(texCoord_" << linearDim + 1 << ", layer));\n";
            }
        }
        [[fallthrough]];

    default:
        break;
    }

    stream << std::endl;
}

void AdaptiveAvgPool2dLayer::buildCalcDefLogic(std::ostream& stream, const LayerGenOptions& options) const {
    std::string channels[4]       = {"x", "y", "z", "w"};
    std::string channelsUppers[4] = {"R", "G", "B", "A"};
    std::string planeIds[4]       = {"0", "1", "2", "3"};
    double val                    = 1. / (options.desiredInput.width * options.desiredInput.height);
    stream.precision(4);
    stream << INDENT << "const mediump float val = " << std::fixed << val << ";\n";
    for (auto planeID : planeIds) {
        for (std::size_t idx = 0; idx < 4; idx++) {
            auto currCharUpper = channelsUppers[idx];
            auto curChar       = channels[idx];
            stream << "#ifdef USE_COMPONENT_" << currCharUpper << "_PLANE_" << planeID << "\n";
            if (planeID == "0") {
                for (uint32_t i = 0; i < options.desiredInput.width; i++) {
                    for (uint32_t j = 0; j < options.desiredInput.height; j++) {
                        int linearDim = options.desiredInput.width * i + j;
                        stream << INDENT << "s." << curChar << " += (t" << linearDim << "_" << planeID << "." << curChar << " * val);\n";
                    }
                }
            } else {
                for (uint32_t i = 0; i < options.desiredInput.width; i++) {
                    for (uint32_t j = 0; j < options.desiredInput.height; j++) {
                        int linearDim = options.desiredInput.width * i + j;
                        stream << INDENT << "s" << planeID << "." << curChar << " += (t" << linearDim << "_" << planeID << "." << curChar << " * val);\n";
                    }
                }
            }
            stream << "\n#endif" << std::endl;
        }
    }
}

void AdaptiveAvgPool2dLayer::buildComputePostDefine(std::ostream& stream, uint32_t outputSliceIndex) const {
    if (_desc.numOutputPlanes > 4) {
        stream << INDENT << "imageStore(outTexture,ivec3(gl_GlobalInvocationID.xy, " << outputSliceIndex << "), s);\n";
    } else {
        stream << INDENT << "imageStore(outTexture,ivec2(gl_GlobalInvocationID.xy), s);\n";
    }
    stream << "}\n";
}

void AdaptiveAvgPool2dLayer::buildFragPostDefine(std::ostream& stream) const {
    switch (_desc.mrtMode) {
    case snn::MRTMode::QUAD_PLANE:
        stream << INDENT << "o_pixel3 = s3;\n";
        stream << INDENT << "o_pixel2 = s2;\n";
        [[fallthrough]];

    case snn::MRTMode::DOUBLE_PLANE:
        stream << INDENT << "o_pixel1 = s1;\n";
        [[fallthrough]];

    default:
        stream << INDENT << "o_pixel = s;\n";
        break;
    }
    stream << "}\n";
}

ShaderLayer::GLSLShaders AdaptiveAvgPool2dLayer::createFS(const GenericModelLayer::LayerGenOptions& options) const {
    std::string shaderTemplateFilePath = "shaders/shadertemplate_fs_avgpooling2d.glsl";
    std::string fsTemplateCode         = loadShader(shaderTemplateFilePath.c_str());

    int channelsPerPass = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;

    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;

    default:
        break;
    }

    uint32_t numShaderPasses = DIV_AND_ROUND_UP(_desc.numOutputPlanes, channelsPerPass);
    std::ostringstream preDefineStream;
    this->buildPreDefine(preDefineStream, options, shaderTemplateFilePath);
    auto preDefine = preDefineStream.str();

    std::ostringstream postDefineStream;
    this->buildFragPostDefine(postDefineStream);
    auto postDefine = postDefineStream.str();

    std::ostringstream avgPoolLogicStream;
    this->buildCalcDefLogic(avgPoolLogicStream, options);
    auto avgPoolLogic = avgPoolLogicStream.str();

    GLSLShaders ret;
    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(numShaderPasses);

    for (uint32_t i = 0, j = 0; i < numShaderPasses; i++, j += 4) {
        uint32_t outputChannels =
            static_cast<uint32_t>(std::min(channelsPerPass, static_cast<int>(_desc.numOutputPlanes) - static_cast<int>(i) * channelsPerPass));
        uint32_t planeCount = DIV_4_ROUND_UP(outputChannels);
        std::ostringstream rgbaDefine;
        rgbaDefine << "#define PLANE_COUNT " << planeCount << "\n";
        for (std::size_t planeIdx = 0; planeIdx < planeCount; planeIdx++) {
            std::size_t remainingPlanes = static_cast<uint32_t>(std::min(4, static_cast<int>(outputChannels) - static_cast<int>(planeIdx) * 4));
            switch (remainingPlanes) {
            case 4:
                rgbaDefine << "#define USE_COMPONENT_A_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_B_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_G_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            case 3:
                rgbaDefine << "#define USE_COMPONENT_B_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_G_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            case 2:
                rgbaDefine << "#define USE_COMPONENT_G_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            case 1:
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            default:
                break;
            }
        }

        std::string fsCode = fsTemplateCode;

        std::ostringstream textureDefStream;
        this->buildTextureDefLogic(textureDefStream, options, DIV_4_ROUND_UP(outputChannels) * i);
        auto textureDef = textureDefStream.str();

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        findAndReplace(fsCode, "_PLACEHOLDER_TEXTURE_READ_", textureDef);
        findAndReplace(fsCode, "_PLACEHOLDER_CALCULATION_", avgPoolLogic);
        findAndReplace(fsCode, "_PLACEHOLDER_N_DIMS_", std::to_string(outputChannels));
        findAndReplace(fsCode, "_PLACEHOLDER_DEFINES_", "");
        findAndReplace(fsCode, "_PLACEHOLDER_UNIFORMS_DECLARATION_", "");

        InferenceGraph::Pass& pass = passes[i];
        pass.source                = preDefine + rgbaDefine.str() + fsCode + postDefine;
        pass.inputs                = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        pass.program               = InferenceGraph::Pass::FsProgram {DIV_4_ROUND_UP(outputChannels) * i, DIV_4_ROUND_UP(outputChannels)};
    }

    return ret;
}

ShaderLayer::GLSLShaders AdaptiveAvgPool2dLayer::createCS(const GenericModelLayer::LayerGenOptions& options) const {
    std::string shaderTemplateFilePath = "shaders/shadertemplate_cs_avgpooling2d.glsl";
    std::string csTemplateCode         = loadShader(shaderTemplateFilePath.c_str());

    uint32_t numShaderPasses = DIV_4_ROUND_UP(_desc.numOutputPlanes);
    std::ostringstream preDefineStream;
    this->buildPreDefine(preDefineStream, options, shaderTemplateFilePath);
    auto preDefine = preDefineStream.str();

    std::ostringstream avgPoolLogicStream;
    this->buildCalcDefLogic(avgPoolLogicStream, options);
    auto avgPoolLogic = avgPoolLogicStream.str();

    GLSLShaders ret;
    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(numShaderPasses);

    for (uint32_t i = 0, j = 0; i < numShaderPasses; i++, j += 4) {
        uint32_t outputChannels = static_cast<uint32_t>(std::min(4, static_cast<int>(_desc.numOutputPlanes) - static_cast<int>(i) * 4));
        std::ostringstream rgbaDefine;
        switch (outputChannels) {
        case 4:
            rgbaDefine << "#define USE_COMPONENT_A\n";
            rgbaDefine << "#define USE_COMPONENT_B\n";
            rgbaDefine << "#define USE_COMPONENT_G\n";
            break;
        case 3:
            rgbaDefine << "#define USE_COMPONENT_B\n";
            rgbaDefine << "#define USE_COMPONENT_G\n";
            break;
        case 2:
            rgbaDefine << "#define USE_COMPONENT_G\n";
            break;
        case 1:
            break;
        default:
            break;
        }

        std::string csCode = csTemplateCode;

        std::ostringstream textureDefStream;
        this->buildTextureDefLogic(textureDefStream, options, j);
        auto textureDef = textureDefStream.str();

        findAndReplace(csCode, "_PLACEHOLDER_TEXTURE_READ_", textureDef);
        findAndReplace(csCode, "_PLACEHOLDER_CALCULATION_", avgPoolLogic);

        // std::cout << "===========================================================" << std::endl;
        // std::cout << csCode << std::endl;
        // std::cout << "===========================================================" << std::endl;

        std::ostringstream postDefineStream;
        this->buildComputePostDefine(postDefineStream, i);
        auto postComputeDefine = postDefineStream.str();

        InferenceGraph::Pass& pass = passes[i];
        pass.source                = preDefine + rgbaDefine.str() + csCode + postComputeDefine;
        pass.inputs                = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        pass.program               = InferenceGraph::Pass::CsProgram {
            "outTexture",
            {(options.desiredOutputWidth + 7) / 8, (options.desiredOutputHeight + 7) / 8, 1},
        };
    }

    return ret;
}
