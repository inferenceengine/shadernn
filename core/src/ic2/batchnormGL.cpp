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
#include "batchnormGL.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <sstream>
#include <utility>

using namespace snn;
using namespace snn::dp;

static constexpr const char* BATCHNORM_FS_ASSET_NAME = "shaders/shadertemplate_fs_batchnorm_RGBA.glsl";
static constexpr const char* BATCHNORM_CS_ASSET_NAME = "shaders/shadertemplate_cs_batchnorm.glsl";

void BatchNormalizationLayerGl::getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass,
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

void BatchNormalizationLayerGl::buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const {
    stream << "#version 320 es\n";
    stream << "// " << shaderFilePath << std::endl;
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    stream << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << std::endl;
    stream << "#define INPUT_WIDTH " << options.desiredInput[0].width << std::endl;
    stream << "#define INPUT_HEIGHT " << options.desiredInput[0].height << std::endl;
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

InferencePassesSptr BatchNormalizationLayerGl::createFS(const LayerGenOptions& options) const {
    std::string shaderFilePath = BATCHNORM_FS_ASSET_NAME;
    std::string fsTemplateCode = loadShader(shaderFilePath.c_str());

    // TODO: need to take MRT into consideration
    uint32_t numShaderPasses = DIV_4_ROUND_UP(_desc.numOutputPlanes);

    // Build beginning shader code.
    std::ostringstream preDefineStream;
    buildPreDefine(preDefineStream, options, shaderFilePath);

    std::string preDefine = preDefineStream.str();

    const int numBatchNormParameters = 4;
    std::vector<std::ostringstream> batchNormalizationConstants(numBatchNormParameters * _desc.numOutputPlanes);

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

        getBatchNormalizationConstants(batchNormalizationConstants, (int) i, (int) outputChannels);
        findAndReplace(fsCode, "_PLACEHOLDER_BETA_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i].str());
        findAndReplace(fsCode, "_PLACEHOLDER_GAMMA_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 1].str());
        findAndReplace(fsCode, "_PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 2].str());
        findAndReplace(fsCode, "_PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_", batchNormalizationConstants[numBatchNormParameters * i + 3].str());

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

InferencePassesSptr BatchNormalizationLayerGl::createCS(const LayerGenOptions& options) const {
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

    std::string shaderUniforms;
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

    std::string shaderMain = loadShader(BATCHNORM_CS_ASSET_NAME);

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    pass.uniforms = {{"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)}};

    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferencePassGl::CsProgram {"uOutput",
                                            // div-by-N is determined by work group size defined CS program.
                                            {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    std::vector<float> bnDataVector;
    std::string bnString;

    pass._vecBeta.resize(_desc.numOutputPlanes);
    bnString     = "beta";
    bnDataVector = _desc.batchNormalization.at(bnString);
    memcpy(pass._vecBeta.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

    pass._vecGamma.resize(_desc.numOutputPlanes);
    bnString     = "gamma";
    bnDataVector = _desc.batchNormalization.at(bnString);
    memcpy(pass._vecGamma.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

    pass._vecMean.resize(_desc.numOutputPlanes);
    bnString     = "movingMean";
    bnDataVector = _desc.batchNormalization.at(bnString);
    memcpy(pass._vecMean.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

    pass._vecVariance.resize(_desc.numOutputPlanes);
    bnString     = "movingVariance";
    bnDataVector = _desc.batchNormalization.at(bnString);
    memcpy(pass._vecVariance.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

    pass.weightMeta.clear();
    pass.weightMeta.push_back((uint32_t) 0); // 0 means Conv2D layout, 1 means DepthWise Conv2D
    pass.weightMeta.push_back((uint32_t)snn::WeightAccessMethod::SSBO_BUFFER);

    return ret;
}
