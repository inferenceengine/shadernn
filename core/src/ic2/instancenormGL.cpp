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
#include "instancenorm.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

DECLARE_LAYER_GL_CLASS(InstanceNorm);

using namespace snn;
using namespace snn::dp;

static constexpr const char* INSTANCENORM_CS_ASSET_NAME = "shaders/shadertemplate_cs_instancenorm.glsl";

InferencePassesSptr InstanceNormLayerGl::createFS(const LayerGenOptions& options) const {
    (void) options;
    InferencePassesSptr ret(new InferencePassesGl());
    SNN_LOGW("No FS implementation for InstanceNorm layer !");
    return ret;
}

InferencePassesSptr InstanceNormLayerGl::createCS(const LayerGenOptions& options) const {
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
                        "#define PRECISION highp\n" // Mean and Variance need highp?
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

    if (!_desc.activation.compare("relu") || !_desc.activation.compare("Relu")) {
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

    if (_desc.useInstanceNormalization) {
        shaderHeader += "#define USE_BATCH_NORMALIZATION\n";
    }
    std::string debugLayer("[0X] Conv2D");
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

    std::string shaderMain = loadShader(INSTANCENORM_CS_ASSET_NAME);

    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    uint32_t maxThreads   = 256;
    uint32_t threadWidth  = std::min(maxThreads, outputWidth);
    uint32_t threadHeight = std::min(maxThreads / threadWidth, outputHeight);
    std::vector<uint32_t> localSize {threadWidth, threadHeight, 1};

    shaderHeader += ("#define WORK_X " + std::to_string(localSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(localSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(localSize[2]) + "\n");

    pass.uniforms = {{"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)}, {"uInputSize", glm::ivec3(inputWidth, inputHeight, ic_4)}};

    pass.inputs = {{"uInput", 0}};

    pass.program = InferencePassGl::CsProgram {"uOutput",
                                            // div-by-N is determined by work group size defined CS program.
                                            {1, 1, UP_DIV(oc_4, localSize[2])}};
    std::string tmpStr = shaderHeader + shaderUniforms;
    pass.source   = (tmpStr + shaderMain);

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    std::vector<float> bnDataVector;
    std::string bnString;

    pass._vecBeta.resize(_desc.numOutputPlanes);
    bnString     = "beta";
    bnDataVector = _desc.instanceNormalization.at(bnString);
    std::memcpy(pass._vecBeta.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

    pass._vecGamma.resize(_desc.numOutputPlanes);
    bnString     = "gamma";
    bnDataVector = _desc.instanceNormalization.at(bnString);
    std::memcpy(pass._vecGamma.data(), bnDataVector.data(), _desc.numOutputPlanes * 4);

    pass.weightMeta.clear();
    pass.weightMeta.push_back((uint32_t) 0); // 0 means Conv2D layout, 1 means DepthWise Conv2D
    pass.weightMeta.push_back((uint32_t)snn::WeightAccessMethod::SSBO_BUFFER);

    return ret;
}
