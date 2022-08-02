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

static constexpr const char* ACTIVATION_FS_ASSET_NAME = "shaders/shadertemplate_fs_activation_RGBA.glsl";
static constexpr const char* ACTIVATION_CS_ASSET_NAME = "shaders/shadertemplate_cs_activation.glsl";

ShaderLayer::GLSLShaders ActivationLayer::createFS(const GenericModelLayer::LayerGenOptions& options) const {
    (void) options;
    GLSLShaders ret;

    int channelsPerPass = 4;
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

    uint32_t numShaderPasses = (uint32_t) DIV_AND_ROUND_UP(_desc.numOutputPlanes, channelsPerPass);
    std::unordered_map<std::string, uint32_t> inputs;

    std::string preDefine = "#version 320 es\n";
    if (numShaderPasses == 1) {
        preDefine += "#define INPUT_TEXTURE_2D\n";
    }

    std::string uniformDecl;
    std::string channelCount = std::to_string(numShaderPasses);

    if (numShaderPasses == 1) {
        uniformDecl += "layout (binding = 0) uniform sampler2D inputTextures0;";
    } else {
        uniformDecl += "layout (binding = 0) uniform sampler2DArray inputTextures0;";
    }

    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(numShaderPasses);

    std::string shaderTemplateCode = loadShader(ACTIVATION_FS_ASSET_NAME);

    for (uint32_t i = 0, j = 0; i < numShaderPasses; i++, j += channelsPerPass) {
        int outputChannels = std::min(channelsPerPass, (int) (_desc.numOutputPlanes - j));
        preDefine += "#define PLANE_COUNT " + std::to_string(DIV_4_ROUND_UP(outputChannels)) + "\n";

        std::string fsCode = preDefine + shaderTemplateCode;

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        findAndReplace(fsCode, "_PLACEHOLDER_UNIFORMS_DECLARATION_", uniformDecl);
        findAndReplace(fsCode, "_PLACEHOLDER_CHANNELS_", channelCount);
        findAndReplace(fsCode, "_PLACEHOLDER_LAYER_", std::to_string(DIV_4_ROUND_UP(outputChannels) * i));

        std::string inputTexID = "inputTextures";

        uint32_t textureId = 0;
        for (auto iterCurTexture = prevLayers.begin(); iterCurTexture != prevLayers.end(); ++iterCurTexture) {
            inputs.insert({inputTexID + std::to_string(textureId), textureId});
            textureId++;
        }

        std::ostringstream postDefineStream;

        postDefineStream << (!_desc.activation.compare("relu")
                                ? "\t\ts = max(s, vec4(0.0));\n"
                                : !_desc.activation.compare("relu6")
                                    ? "\t\ts = clamp(s, vec4(0), vec4(6));\n"
                                    : !_desc.activation.compare("tanh")
                                            ? "\t\ts = tanh(s);\n"
                                            : !_desc.activation.compare("sigmoid")
                                                ? "\t\ts = vec4(1.0f)/(vec4(1.0f)+ exp(-s));\n"
                                                : !_desc.activation.compare("leakyRelu")
                                                        ? "\t\ts = max(s, (s * vec4(" + std::to_string(_desc.leakyReluAlpha) + "f)));\n"
                                                        : !_desc.activation.compare("leaky_relu")
                                                            ? "\t\ts = max(s, (s * vec4(" + std::to_string(_desc.leakyReluAlpha) + "f)));\n"
                                                            : !_desc.activation.compare("SiLU") ? "s = s * vec4(1.0f)/(vec4(1.0f)+ exp(-s))" : "");
        for (std::size_t i = 1; i < 4; i++) {
            auto layer2Str = std::to_string(i);
            postDefineStream << "#if PLANE_COUNT > 1\n";
            postDefineStream << (!_desc.activation.compare("relu")
                                    ? "\t\ts" + layer2Str + " = max(s" + layer2Str + ", vec4(0.0));\n"
                                    : !_desc.activation.compare("relu6")
                                        ? "\t\ts = clamp(s, vec4(0), vec4(6));\n"
                                        : !_desc.activation.compare("tanh")
                                                ? "\t\ts" + layer2Str + " = tanh(s" + layer2Str + ");\n"
                                                : !_desc.activation.compare("sigmoid")
                                                    ? "\t\ts" + layer2Str + " = vec4(1.0f)/(vec4(1.0f)+ exp(-s" + layer2Str + "));\n"
                                                    : !_desc.activation.compare("leakyRelu")
                                                            ? "\t\ts" + layer2Str + " = max(s" + layer2Str + ", (s" + layer2Str + " * vec4(" +
                                                                std::to_string(_desc.leakyReluAlpha) + "f)));\n"
                                                            : !_desc.activation.compare("leaky_relu")
                                                                ? "\t\ts" + layer2Str + " = max(s" + layer2Str + ", (s" + layer2Str + " * vec4(" +
                                                                        std::to_string(_desc.leakyReluAlpha) + "f)));\n"
                                                                : !_desc.activation.compare("SiLU")
                                                                        ? "s" + layer2Str + " = s" + layer2Str + " * vec4(1.0f)/(vec4(1.0f)+ exp(-s" +
                                                                            layer2Str + "))"
                                                                        : "");
            postDefineStream << "#endif\n";
        }
        std::string postDefine = postDefineStream.str();
        findAndReplace(fsCode, "_PLACEHOLDER_ACTIVATION_", postDefine);

        InferenceGraph::Pass& pass = passes[i];
        pass.source                = fsCode;
        pass.inputs                = inputs;
        pass.program               = InferenceGraph::Pass::FsProgram {(uint32_t) DIV_4_ROUND_UP(outputChannels) * i, (uint32_t) DIV_4_ROUND_UP(outputChannels)};
    }

    return ret;
}

ShaderLayer::GLSLShaders ActivationLayer::createCS(const LayerGenOptions& options) const {
    (void) options;

    GLSLShaders ret;

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

    // printf("Test:%s:%d\n",__FUNCTION__,__LINE__);
    string shaderHeader;
    if (_desc.preferHp) {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION mediump\n"
                       "precision PRECISION float;\n"
                       "layout(std140) uniform;\n"
                       "#define OUTPUT_FORMAT rgba16f\n";
    } else {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION highp\n"
                       "precision PRECISION float;\n"
                       "layout(std140) uniform;\n"
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
                            "#endif\n";

    string shaderMain = "layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;\n"
                        "void main()\n"
                        "{\n"
                        "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
                        "    #ifdef INPUT_TEXTURE_2D\n"
                        "    vec4 value = imageLoad(uInput, ivec2(pos.x, pos.y));\n"
                        "    #else\n"
                        "    vec4 value = imageLoad(uInput, pos);\n"
                        "    #endif\n"
                        "    #ifdef RELU\n"
                        "    value = max(value, vec4(0));\n"
                        "    #endif\n"
                        "    #ifdef RELU6\n"
                        "    value = clamp(value, vec4(0), vec4(6));\n"
                        "    #endif\n"
                        "    #ifdef TANH\n"
                        "    value = tanh(value);\n"
                        "    #endif\n"
                        "    #ifdef SIGMOID\n"
                        "    value  = vec4(1.0f)/(vec4(1.0f)+ exp(-value));\n"
                        "    #endif\n"
                        "    #ifdef LEAKYRELU_VAL\n"
                        "    value   = max(value,  (value * vec4(LEAKYRELU_VAL)));\n"
                        "    #endif\n"
                        "    #ifdef SILU\n"
                        "    value    = value  * vec4(1.0f)/(vec4(1.0f)+ exp(-value));\n"
                        "    #endif\n"
                        "    #ifdef OUTPUT_TEXTURE_2D\n"
                        "    imageStore(uOutput, ivec2(pos.x, pos.y), value);\n"
                        "    #else\n"
                        "    imageStore(uOutput, pos, value);\n"
                        "    #endif\n"
                        "}\n";
    shaderMain = loadShader(ACTIVATION_CS_ASSET_NAME);

    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");
    SNN_LOGD("Test:%s:%d, %s\n", __FUNCTION__, __LINE__, shaderHeader.c_str());

    pass.uniforms = {
        //                     {"imgSize", glm::ivec4(inputWidth, inputHeight, ic_4, 1)}
    };
    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferenceGraph::Pass::CsProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGI("%s:%d, input:%d:%d:%d, output:%d:%d:%d\n", __FUNCTION__, __LINE__, inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
