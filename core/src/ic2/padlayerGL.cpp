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
#include "padlayerGL.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <sstream>
#include <utility>

using namespace snn;
using namespace snn::dp;

static constexpr const char* PAD_CS_ASSET_NAME = "shaders/shadertemplate_cs_pad.glsl";

void PadLayerGl::buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const {
    uint32_t offsets[4];
    this->getPaddingOffset(offsets);
    stream << "#version 320 es\n";
    stream << "// " << shaderFilePath << std::endl;
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    stream << "#define NUM_OUTPUT_PLANES    " << _desc.numOutputPlanes << std::endl;
    stream << "#define INPUT_WIDTH " << options.desiredInput[0].width << std::endl;
    stream << "#define INPUT_HEIGHT " << options.desiredInput[0].height << std::endl;
    stream << "#define PAD_VALUE 0.0f" << std::endl;
    // Currently we have no way of getting which value
    // is being used to pad. So we default to 0.0
    if (_desc.mode == "constant") {
        stream << "#define CONST_PADDING " << std::endl;
    } else if (_desc.mode == "replicate") {
        stream << "#define REPLICATE_PADDING " << std::endl;
    } else if (_desc.mode == "reflect") {
        stream << "#define REFLECT_PADDING " << std::endl;
    } else if (_desc.mode == "repeat") {
        stream << "#CHECKBOARD_PADDING " << std::endl;
    }

    if (_desc.numInputPlanes <= 4) {
        stream << "#define INPUT_TEXTURE_2D\n";
    }
    stream << "#define PADDING_T " << offsets[0] << std::endl;
    stream << "#define PADDING_B " << offsets[1] << std::endl;
    stream << "#define PADDING_L " << offsets[2] << std::endl;
    stream << "#define PADDING_R " << offsets[3] << std::endl;
}

InferencePassesSptr PadLayerGl::createFS(const LayerGenOptions& options) const {
    std::ostringstream shaderFilePathStream;
    std::string fileName = "/shadertemplate_fs_pad_RGBA.glsl";
    shaderFilePathStream << "shaders" << fileName;

    std::string shaderFilePath = shaderFilePathStream.str();

    std::string fsTemplateCode = loadShader(shaderFilePath.c_str());

    // TODO: need to take MRT into consideration
    uint32_t numShaderPasses = DIV_4_ROUND_UP(_desc.numOutputPlanes);

    // Build beginning shader code.
    std::ostringstream preDefineStream;
    buildPreDefine(preDefineStream, options, shaderFilePath);

    std::string preDefine = preDefineStream.str();

    InferencePassesSptr ret(new InferencePassesGl());
    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(numShaderPasses);
    for (uint32_t i = 0, j = 0; i < numShaderPasses; ++i, j += 4) {
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

        // Create a copy of the template code.
        // After modification, this will contain the shader's true source code.
        std::string fsCode = fsTemplateCode;

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }
        std::ostringstream layerstream;
        layerstream << "int layer = " << i << ";" << std::endl;
        findAndReplace(fsCode, "LAYER_CALCULATION", layerstream.str());
        InferencePassGl& pass = passes[i];
        pass.source                = preDefine + rgbaDefine.str() + fsCode;

        pass.inputs  = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        pass.program = InferencePassGl::FsProgram {i, DIV_4_ROUND_UP(outputChannels)};
    }
    return ret;
}

InferencePassesSptr PadLayerGl::createCS(const LayerGenOptions& options) const {
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

    if (!_desc.mode.compare("constant")) {
        shaderHeader += "#define CONSTANT_PADDING\n";
    } else if (!_desc.mode.compare("replicate")) {
        shaderHeader += "#define REPLICATE_PADDING\n";
    } else if (!_desc.mode.compare("reflect")) {
        shaderHeader += "#define REFLECT_PADDING\n";
    }

    std::string shaderUniforms = "#ifdef OUTPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D uOutput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"
                            "#endif\n"
                            "#ifdef INPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput;\n"
                            "#endif\n";

    std::string shaderMain = loadShader(PAD_CS_ASSET_NAME);

    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    uint32_t paddingOffsets[4];
    this->getPaddingOffset(paddingOffsets);
    SNN_LOGD("Padding, %d, %d, %d", _desc.mode.c_str(), paddingOffsets[0], paddingOffsets[1], paddingOffsets[2],
             paddingOffsets[3]);

    pass.uniforms = {{"uPad", glm::ivec2(paddingOffsets[0], paddingOffsets[2])},
                     {"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)},
                     {"uInputSize", glm::ivec3(inputWidth, inputHeight, ic_4)}};
    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferencePassGl::CsProgram {"uOutput",
                                            // div-by-N is determined by work group size defined CS program.
                                            {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
