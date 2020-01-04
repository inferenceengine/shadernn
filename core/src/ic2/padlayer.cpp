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
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <ios>

using namespace snn;
using namespace snn::dp;

static constexpr const char* PAD_CS_ASSET_NAME = "shaders/shadertemplate_cs_pad.glsl";

void PadLayer::getPaddingOffset(uint32_t (&offsets)[4]) const {
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
                // if (_desc.kernelSize % 2 == 0) {
                //     offsets[0] = offsets[0] - 1;
                //     offsets[2] = offsets[2] - 1;
                // }
            } else {
                offsets[0] = 0;
                offsets[1] = 0;
                offsets[2] = 0;
                offsets[3] = 0;
            }
        }
    }
}

void PadLayer::buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const {
    uint32_t offsets[4];
    this->getPaddingOffset(offsets);
    stream << "#version 320 es\n";
    stream << "// " << shaderFilePath << std::endl;
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    stream << "#define NUM_OUTPUT_PLANES    " << _desc.numOutputPlanes << std::endl;
    stream << "#define INPUT_WIDTH " << options.desiredInput.width << std::endl;
    stream << "#define INPUT_HEIGHT " << options.desiredInput.height << std::endl;
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

ShaderLayer::GLSLShaders PadLayer::createFS(const LayerGenOptions& options) const {
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

    GLSLShaders ret;
    std::vector<InferenceGraph::Pass>& passes = ret.passes;
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
        InferenceGraph::Pass& pass = passes[i];
        pass.source                = preDefine + rgbaDefine.str() + fsCode;

        pass.inputs  = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        pass.program = InferenceGraph::Pass::FsProgram {i, DIV_4_ROUND_UP(outputChannels)};
    }
    return ret;
}

ShaderLayer::GLSLShaders PadLayer::createCS(const LayerGenOptions& options) const {
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

    string shaderMain;

    if (1) {
        shaderMain = "layout(location=4) uniform ivec2 uPad;\n"
                     "layout(location=10) uniform ivec3 uOutputSize;\n"
                     "layout(location=11) uniform ivec3 uInputSize;\n"
                     "layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;\n"
                     "void main()\n"
                     "{\n"
                     "    if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize)))\n"
                     "    {\n"
                     "        ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
                     "        ivec2 s0 = pos.xy-uPad;\n"
                     "        int sy = s0.y;\n"
                     "        #ifdef CONSTANT_PADDING \n"
                     "        sy = ((sy >= 0) && (sy < uInputSize.y)) ? sy: uInputSize.y;\n"
                     "        #endif\n"
                     "        #ifdef REPLICATE_PADDING \n"
                     "        sy =  min(max(sy, 0), uInputSize.y-1);\n"
                     "        #endif\n"
                     "        #ifdef REFLECT_PADDING \n"
                     "        sy = (sy < 0) ? -sy:sy;\n"
                     "        sy = (sy >= uInputSize.y) ? 2*uInputSize.y-2-sy:sy;\n"
                     "        #endif\n"
                     "        int sx = s0.x;\n"
                     "        #ifdef CONSTANT_PADDING \n"
                     "        sx = ((sx >= 0) && (sx < uInputSize.x)) ? sx: uInputSize.x;\n"
                     "        #endif\n"
                     "        #ifdef REPLICATE_PADDING \n"
                     "        sx = min(max(sx, 0), uInputSize.x-1);\n"
                     "        #endif\n"
                     "        #ifdef REFLECT_PADDING \n"
                     "        sx = (sx < 0) ? -sx:sx;\n"
                     "        sx = (sx >= uInputSize.x) ? 2*uInputSize.x-2-sx:sx;\n"
                     "        #endif\n"
                     "        #ifdef INPUT_TEXTURE_2D\n"
                     "        vec4 sum = imageLoad(uInput, ivec2(sx, sy)); \n"
                     "        #else\n"
                     "        vec4 sum = imageLoad(uInput, ivec3(sx, sy, pos.z)); \n"
                     "        #endif\n"
                     "        #ifdef OUTPUT_TEXTURE_2D\n"
                     "        imageStore(uOutput, ivec2(pos.x, pos.y), sum);\n"
                     "        #else\n"
                     "        imageStore(uOutput, pos, sum);\n"
                     "        #endif\n"
                     "    }\n"
                     "}\n";
        shaderMain = loadShader(PAD_CS_ASSET_NAME);
    }

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    SNN_LOGD("Test:%s:%d, %d, %d, %d, %d\n", __FUNCTION__, __LINE__, kernel, stride, ic_4, oc_4);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");
    SNN_LOGD("Test:%s:%d, %s\n", __FUNCTION__, __LINE__, shaderHeader.c_str());

    if (1) {
        uint32_t paddingOffsets[4];
        this->getPaddingOffset(paddingOffsets);
        SNN_LOGD("%s:%d, Padding: %s: %d, %d, %d, %d\n", __FUNCTION__, __LINE__, _desc.mode.c_str(), paddingOffsets[0], paddingOffsets[1], paddingOffsets[2],
                 paddingOffsets[3]);

        pass.uniforms = {{"uPad", glm::ivec2(paddingOffsets[0], paddingOffsets[2])},
                         {"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)},
                         {"uInputSize", glm::ivec3(inputWidth, inputHeight, ic_4)}};
    }
    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferenceGraph::Pass::CsProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGD("%s:%d, input:%d:%d:%d, output:%d:%d:%d\n", __FUNCTION__, __LINE__, inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}

InferenceGraph::Transform PadLayer::getOutputScaleDimAdjustment() const {
    uint32_t offset[4];
    this->getPaddingOffset(offset);
    float scale        = 1;
    float translation1 = 0.0f, translation2 = 0.0f;
    translation1 = static_cast<float>(offset[2] + offset[3]);
    translation2 = static_cast<float>(offset[0] + offset[1]);
    return {0, scale, scale, translation1, translation2};
}
