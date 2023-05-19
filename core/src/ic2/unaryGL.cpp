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
#include "unary.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

DECLARE_LAYER_GL_CLASS(Unary);

using namespace snn;
using namespace snn::dp;

static constexpr const char* UNARY_FS_ASSET_NAME = "shaders/shadertemplate_fs_unary.glsl";
static constexpr const char* UNARY_CS_ASSET_NAME = "shaders/shadertemplate_cs_unary.glsl";

InferencePassesSptr UnaryLayerGl::createFS(const GenericModelLayer::LayerGenOptions& options) const {
    (void) options;
    InferencePassesSptr ret(new InferencePassesGl());

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

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(numShaderPasses);

    std::string shaderTemplateCode = loadShader(UNARY_FS_ASSET_NAME);

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

        std::string postDefine = postDefineStream.str();
        findAndReplace(fsCode, "_PLACEHOLDER_ACTIVATION_", postDefine);

        InferencePassGl& pass = passes[i];
        pass.source                = fsCode;
        pass.inputs                = inputs;
        pass.program               = InferencePassGl::FsProgram {(uint32_t) DIV_4_ROUND_UP(outputChannels) * i, (uint32_t) DIV_4_ROUND_UP(outputChannels)};
    }

    return ret;
}

InferencePassesSptr UnaryLayerGl::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesSptr ret(new InferencePassesGl());

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassGl& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t inputDepth  = inputDims[0].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = _desc.numOutputPlanes;

    std::string shaderHeader;
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

    std::string shaderUniforms = "#ifdef OUTPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D u_Output;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray u_Output;\n"
                            "#endif\n"
                            "#ifdef INPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D u_Input;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray u_Input;\n"
                            "#endif\n";

    std::string shaderMain = loadShader(UNARY_CS_ASSET_NAME);

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    pass.uniforms = {
        {"uConstantUnaryType", _desc.opType},
        {"uConstantValue", _desc.opValue},
    };

    pass.inputs  = {{"u_Input", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferencePassGl::CsProgram {"u_Output",
                                                // div-by-N is determined by work group size defined CS program.
                                                {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
