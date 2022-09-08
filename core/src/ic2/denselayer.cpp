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
#include <memory>

using namespace std;
using namespace snn;
using namespace snn::dp;

static constexpr const char* DENSE_CS_ASSET_NAME = "shaders/shadertemplate_cs_dense.glsl";

snn::dp::GenericModelLayer::GLSLShaders snn::dp::DenseLayer::createGLSLShader(const LayerGenOptions& options) {
    // auto dummyOptions = options;
    (void) options;

    GLSLShaders ret;

    int inputWidth       = (int) _desc.weights.size();
    uint32_t outputWidth = (uint32_t) _desc.biases.size();

    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(1);

    InferenceGraph::Pass& pass = passes[0];

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
    // SNN_LOGI("%s:%d\nShader Header: %s\n", __FUNCTION__, __LINE__, shaderHeader.c_str());
    std::string sourceCode = "layout(OUTPUT_FORMAT, binding=0) writeonly uniform PRECISION image2DArray uOutImage;\n"
                            "layout(OUTPUT_FORMAT, binding=1) readonly uniform PRECISION image2DArray uInImage;\n"
                            "layout(binding=2) readonly buffer weightBuffer{\n"
                            "    float data[];\n"
                            "} uWightBuffer;\n"
                            "layout(binding=3) readonly buffer biasBuffer{\n"
                            "    float data[];\n"
                            "} uBiasBuffer;\n"
                            "layout(location = 4) uniform int uWidth;\n"
                            "layout(location = 5) uniform int uHeight;\n"
                            "layout(location = 6) uniform int activation;\n"
                            // "layout(binding=7) writeonly buffer destBuffer{\n"
                            // "    float data[];\n"
                            // "} uOutBuffer;\n"
                            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
                            "float relu(float i);\n"
                            "float sigmoid(float i);\n"
                            "float activeValue(int type, float v);\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
                            "    float res = 0.0f;\n"
                            "    for (int w = 0; w < uWidth; w+=4) \n"
                            "    {\n"
                            "        vec4 color;\n"
                            "        vec4 weight;\n"
                            "        int z = pos.z*4;\n"
                            "        vec4 color0 = imageLoad(uInImage, ivec3(w, 0, 0));\n"
                            "        vec4 color1 = imageLoad(uInImage, ivec3(w+1, 0, 0));\n"
                            "        vec4 color2 = imageLoad(uInImage, ivec3(w+2, 0, 0));\n"
                            "        vec4 color3 = imageLoad(uInImage, ivec3(w+3, 0, 0));\n"
                            "        weight.r = uWightBuffer.data[pos.y*uWidth+w];\n"
                            "        weight.g = uWightBuffer.data[pos.y*uWidth+w+1];\n"
                            "        weight.b = uWightBuffer.data[pos.y*uWidth+w+2];\n"
                            "        weight.a = uWightBuffer.data[pos.y*uWidth+w+3];\n"
                            // "        float test = float(w);\n"
                            // "        uOutBuffer.data[pos.y] = color0.r;\n"
                            "        \n"
                            "        res += dot(vec4(color0.r, color1.r, color2.r, color3.r), weight);\n"
                            // "        res += dot(vec4(0.1f, 0.2f, 0.3f, 0.4f), weight);\n"
                            "    }\n"
                            "    res += uBiasBuffer.data[pos.y];\n"
                            "    res = activeValue(activation, res);\n"
                            "    float test = float(uWidth);\n"
                            "    imageStore(uOutImage,ivec3(pos.y, 0, 0),vec4(res,0,0,0));\n"
                            // "    vec4 color = imageLoad(uInImage, ivec3(0, 0, 0));\n"
                            // "    float test = float(uWidth);\n"
                            // "    uOutBuffer.data[pos.y] = res;\n"
                            "}\n"
                            "\n"
                            "float relu(float i){\n"
                            "   if (i > 0.0){\n"
                            "       return i;\n"
                            "   } else {\n"
                            "       return 0.0;\n"
                            "   }\n"
                            "}\n"
                            "\n"
                            "float sigmoid(float i){\n"
                            "    return 1.0 / (1.0 + exp(-i));\n"
                            "}\n"
                            "\n"
                            "float activeValue(int type, float v){\n"
                            "    if (type == 0) {\n"
                            "        return (v);\n"
                            "    } else if (type == 1) {\n"
                            "        return relu(v);\n"
                            "    } else if (type == 2) {\n"
                            "        return sigmoid(v);\n"
                            "    } else if (type == 3){\n"
                            "        return tanh(v);\n"
                            "    } else {\n"
                            "        return v;\n"
                            "    }\n"
                            "}\n";

    sourceCode = loadShader(DENSE_CS_ASSET_NAME);

    pass.uniforms = {{"uWidth", inputWidth}, // w*h*channels
                                            //{"uHeight", inputHeight}, // 1
                    {"activation", 0}};
    pass.inputs   = {{"uInImage", 0}};
    pass.source   = shaderHeader + sourceCode;
    pass.program  = InferenceGraph::Pass::CsProgram {
        "uOutImage",
        // div-by-N is determined by work group size defined CS program.
        {1, outputWidth, 1},
    };

    // pass.transformMat.emplace(std::pair<std::vector<std::vector<float>>, std::vector<float>>(
    //     _desc.weights,
    //     _desc.biases
    // ));

#if USE_BUFFER_OBJECT
    pass._vecWeights.resize(inputWidth * outputWidth);
    pass._boWeights.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
    pass._boWeights->allocate(inputWidth * outputWidth * 4, pass._vecWeights.data());
    float* destWeight = (float*) pass._boWeights->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
#else
    SNN_LOGD("Test:%s:%d input: %d-%zu, output: %d-%zu\n", __FUNCTION__, __LINE__, inputWidth, _desc.weights[0].size(), outputWidth,
             _desc.weights.size());
    pass._ssboWeights.reset(new gl::GLSSBOBuffer(inputWidth * outputWidth * 4));
    //_ssboWeights = new GLSSBOBuffer(options.desiredInput.width * options.desiredOutputHeight * 4);
    float* destWeight = (float*) pass._ssboWeights->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
#endif

    unsigned int width = _desc.weights[0].size();
    // for (size_t j = 0; j < width; j++) {
    //     for (size_t i = 0; i < _desc.weights.size(); i++) {
    //         *(destWeight + j * inputWidth + i) = _desc.weights[i][j]; //_desc.weights is 512 x 10, we need 10 x 512 here.
    //         SNN_LOGD("%zu:%zu: %zu, %f\n",i, j, j * inputWidth + i, _desc.weights[i][j]);
    //     }
    // }
    int kIndex = 0;
    for (size_t i = 0; i < _desc.weights.size(); i++) {
        for (size_t j = 0; j < width; j++) {
            // SNN_LOGD("%zu:%zu: %zu, %f\n",i, j, i * inputWidth + j, _desc.weights[i][j]);
            *(destWeight + kIndex) = _desc.weights[i][j];
            kIndex++;
        }
    }

#if USE_BUFFER_OBJECT
    pass._vecBias.resize(outputWidth);
    pass._boBias.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER, MIN_SSBO_BUFFER_LEN_ARM_MALI>());
    pass._boBias->allocate(outputWidth * 4, pass._vecBias.data());
    float* destBias = (float*) pass._boBias->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
#else
    pass._ssboBias.reset(new gl::GLSSBOBuffer(outputWidth * 4, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW));
    SNN_LOGD("Test:%s:%d bias:%d - %zu\n", __FUNCTION__, __LINE__, outputWidth, _desc.biases.size());

    //_ssboBias = new GLSSBOBuffer(options.desiredOutputHeight * 4);
    float* destBias = (float*) pass._ssboBias->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
#endif

    for (size_t i = 0; i < _desc.biases.size(); i++) {
        *(destBias + i) = _desc.biases[i];
        // SNN_LOGD("%zu:%f\n",i, _desc.biases[i]);
    }

#if USE_BUFFER_OBJECT
    pass._boWeights->unmap();
    pass._boBias->unmap();
    pass.ssboMap = {{2, pass._boWeights->getId()}, {3, pass._boBias->getId()}};
#else
    pass._ssboWeights->unmap();
    pass._ssboBias->unmap();
    pass.ssboMap = {{2, pass._ssboWeights->getId()}, {3, pass._ssboBias->getId()}};
#endif

    setLayerExecutionLevel(snn::InferenceGraph::LayerExecution::GPU_CS);
    return ret;
}

void snn::dp::DenseLayer::computeImageTexture(FixedSizeArray<snn::ImageTexture>& inputTex, FixedSizeArray<snn::ImageTexture>& outputTex) {
    auto inputMat = inputTex[0].outputMat;

    auto cpuL = snn::dp::CPUCommonUtil<float> {_desc.activation, _desc.leakyReluAlpha, true};

    auto transformMats = std::pair<std::vector<std::vector<float>>, std::vector<float>>(_desc.weights, _desc.biases);
    if (!cpuL.inputMat.has_value()) {
        cpuL.inputMat.emplace(inputMat);
    }
    cpuL.run(transformMats);
    cpuL.getOutputs(outputTex[0].outputMat);
    SNN_LOGD("%%%%%%%% %s:%d :%s\n", __FUNCTION__, __LINE__, name.c_str());
}

InferenceGraph::Transform DenseLayer::getOutputScaleDimAdjustment() const {
    InferenceGraph::Transform ret;
    ret.isFixed     = 1;
    ret.fixedWidth  = (uint32_t) _desc.biases.size();
    ret.fixedHeight = 1;
    ret.fixedDepth  = 1;
    ret.fixedBatch  = 1;
    return ret;
}

void snn::dp::DenseLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    width  = (uint32_t) _desc.biases.size();
    height = 1;
    depth  = 1;
}
