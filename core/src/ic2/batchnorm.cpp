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

static constexpr const char* BATCHNORM_FS_ASSET_NAME = "shaders/shadertemplate_fs_batchnorm_RGBA.glsl";
static constexpr const char* BATCHNORM_CS_ASSET_NAME = "shaders/shadertemplate_cs_batchnorm.glsl";

void BatchNormalizationLayer::getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass,
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

void BatchNormalizationLayer::buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const {
    stream << "#version 320 es\n";
    stream << "// " << shaderFilePath << std::endl;
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    stream << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << std::endl;
    stream << "#define INPUT_WIDTH " << options.desiredInput.width << std::endl;
    stream << "#define INPUT_HEIGHT " << options.desiredInput.height << std::endl;
    // Currently we have no way of getting which value
    // is being used to pad. So we default to 0.0

    if (options.isFirstLayer) {
        if (getDesc().isRange01) {
            stream << "#define REMOVE_ZERO 1" << std::endl;
        } else {
            stream << "#define SCALE_INPUT 1" << std::endl;
        }
    }
    if (_desc.numInputPlanes <= 4) {
        stream << "#define INPUT_TEXTURE_2D\n";
    }
}

ShaderLayer::GLSLShaders BatchNormalizationLayer::createFS(const LayerGenOptions& options) const {
    std::string shaderFilePath = BATCHNORM_FS_ASSET_NAME;
    std::string fsTemplateCode = loadShader(shaderFilePath.c_str());

    // TODO: need to take MRT into consideration
    uint32_t numShaderPasses = DIV_4_ROUND_UP(_desc.numOutputPlanes);

    // Build beginning shader code.
    std::ostringstream preDefineStream;
    buildPreDefine(preDefineStream, options, shaderFilePath);

    std::string preDefine = preDefineStream.str();

    int numBatchNormParameters = 4;
    std::vector<std::ostringstream> batchNormalizationConstants(numBatchNormParameters * _desc.numOutputPlanes);

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

        getBatchNormalizationConstants(batchNormalizationConstants, (int) i, (int) outputChannels);
        findAndReplace(fsCode, "_PLACEHOLDER_BETA_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i].str());
        findAndReplace(fsCode, "_PLACEHOLDER_GAMMA_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 1].str());
        findAndReplace(fsCode, "_PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 2].str());
        findAndReplace(fsCode, "_PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 3].str());

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

ShaderLayer::GLSLShaders BatchNormalizationLayer::createCS(const LayerGenOptions& options) const {
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

    string shaderUniforms;
    shaderUniforms = "#ifdef OUTPUT_TEXTURE_2D\n"
                     "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D uOutput;\n"
                     "#else\n"
                     "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"
                     "#endif\n"
                     "#ifdef INPUT_TEXTURE_2D\n"
                     "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput;\n"
                     "#else\n"
                     "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput;\n"
                     "#endif\n";

    string shaderMain = "layout(binding=5) readonly buffer beta{\n"
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
                        "layout(location=9) uniform ivec3 uOutputSize;\n"
                        "layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;\n"
                        "void main()\n"
                        "{\n"
                        "    ivec3 outputSize = uOutputSize;\n"
                        "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
                        "    if (all(lessThan(pos, outputSize)))\n"
                        "    {\n"
                        "        #ifdef INPUT_TEXTURE_2D\n"
                        "        vec4 color = imageLoad(uInput, ivec2(pos.x, pos.y));\n"
                        "        #else\n"
                        "        vec4 color = imageLoad(uInput, pos);\n"
                        "        #endif\n"
                        "        vec4 movingVariance = uVariance.data[pos.z]; \n"
                        "        vec4 movingMean = uMean.data[pos.z]; \n"
                        "        vec4 gamma = uGamma.data[pos.z]; \n"
                        "        vec4 beta = uBeta.data[pos.z]; \n"
                        "        vec4 sqrtVar = sqrt(movingVariance + vec4(0.001f)); \n"
                        "        sqrtVar = max(sqrtVar, vec4(0.0001f)); \n"
                        "        color = ((gamma/sqrtVar) * (color - movingMean)) + beta;   \n"
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

    shaderMain = loadShader(BATCHNORM_CS_ASSET_NAME);

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    pass.uniforms = {{"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)}};

    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferenceGraph::Pass::CsProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGD("%s:%d, input:%d:%d:%d, output:%d:%d:%d\n", __FUNCTION__, __LINE__, inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    if (1) {
        float* bnDataDest;
        std::vector<float> bnDataVector;
        std::string bnString;
        pass._vecBeta.resize(_desc.numOutputPlanes);
        pass._bnBeta.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnBeta->allocate(_desc.numOutputPlanes, pass._vecBeta.data());
        bnDataDest   = (float*) pass._bnBeta->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "beta";
        bnDataVector = _desc.batchNormalization.at(bnString);
        SNN_LOGD("Beta:%s:%d \n", __FUNCTION__, __LINE__);
        /*
        for (size_t i = 0; i < bnDataVector.size(); i++) {
           *(bnDataDest + i) = bnDataVector[i];
           // printf("%zu:%f\n",i, bnDataVector[i]);
        }
        */
        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnBeta->unmap();
        pass.ssboMap[5] = pass._bnBeta->getId();

        pass._vecGamma.resize(_desc.numOutputPlanes);
        pass._bnGamma.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnGamma->allocate(_desc.numOutputPlanes, pass._vecGamma.data());
        bnDataDest   = (float*) pass._bnGamma->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "gamma";
        bnDataVector = _desc.batchNormalization.at(bnString);
        SNN_LOGD("Gamma:%s:%d \n", __FUNCTION__, __LINE__);
        /*
        for (size_t i = 0; i < bnDataVector.size(); i++) {
            *(bnDataDest + i) = bnDataVector[i];
            // printf("%zu:%f\n",i, bnDataVector[i]);
        }
        */
        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnGamma->unmap();
        pass.ssboMap[6] = pass._bnGamma->getId();

        pass._vecMean.resize(_desc.numOutputPlanes);
        pass._bnMean.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnMean->allocate(_desc.numOutputPlanes, pass._vecMean.data());
        bnDataDest   = (float*) pass._bnMean->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "movingMean";
        bnDataVector = _desc.batchNormalization.at(bnString);
        SNN_LOGD("Mean:%s:%d \n", __FUNCTION__, __LINE__);
        /*
        for (size_t i = 0; i < bnDataVector.size(); i++) {
            *(bnDataDest + i) = bnDataVector[i];
            // printf("%zu:%f\n",i, bnDataVector[i]);
        }
        */
        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnMean->unmap();
        pass.ssboMap[7] = pass._bnMean->getId();

        pass._vecVariance.resize(_desc.numOutputPlanes);
        pass._bnVariance.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnVariance->allocate(_desc.numOutputPlanes, pass._vecVariance.data());
        bnDataDest = (float*) pass._bnVariance->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString   = "movingVariance";
        SNN_LOGD("Variance:%s:%d \n", __FUNCTION__, __LINE__);
        bnDataVector = _desc.batchNormalization.at(bnString);
        /*
        for (size_t i = 0; i < bnDataVector.size(); i++) {
            *(bnDataDest + i) = bnDataVector[i];
            // printf("%zu:%f\n",i, bnDataVector[i]);
        }
        */
        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnVariance->unmap();
        pass.ssboMap[8] = pass._bnVariance->getId();
    }

    return ret;
}

InferenceGraph::Transform BatchNormalizationLayer::getOutputScaleDimAdjustment() const {
    float scale       = 1;
    float translation = 0.0f;
    return {0, scale, scale, translation, translation};
}
