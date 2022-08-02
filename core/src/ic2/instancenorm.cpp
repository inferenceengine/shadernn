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

static constexpr const char* INSTANCENORM_CS_ASSET_NAME = "shaders/shadertemplate_cs_instancenorm.glsl";

InstanceNormLayer::InstanceNormLayer(InstanceNormDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {
    snn::ColorFormat weightFormat;
    if (_desc.preferrHalfPrecision) {
        weightFormat = snn::ColorFormat::RGBA16F;
    } else {
        weightFormat = snn::ColorFormat::RGBA32F;
    }

    for (std::size_t i = 0; i < DIV_4_ROUND_UP(_desc.numOutputPlanes); i++) {
        switch (_desc.weightMode) {
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

    this->biases = _desc.biases;

    // SNN_LOGI("Size of biases for current layer with input channels %u and output channels %u is %u", _desc.numInputPlanes, _desc.numOutputPlanes,
    // this->biases.size());
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

ShaderLayer::GLSLShaders InstanceNormLayer::createFS(const LayerGenOptions& options) const {
    (void) options;
    GLSLShaders ret;

    return ret;
}

InferenceGraph::Transform InstanceNormLayer::getOutputScaleDimAdjustment() const {
    float scale       = 1;
    float translation = 0.0f;
    return {0, scale, scale, translation, translation};
}

ShaderLayer::GLSLShaders InstanceNormLayer::createCS(const LayerGenOptions& options) const {
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

    if (_desc.useInstanceNormalization) {
        shaderHeader += "#define USE_BATCH_NORMALIZATION\n";
    }
    string debugLayer("[0X] Conv2D");
    /*
    if (this->name.find(debugLayer) != std::string::npos) {
        shaderHeader += "#define CONV2D_DEBUG\n";
    }
    */
    // SNN_LOGI("Test:%s:%d, %s\n",__FUNCTION__,__LINE__, shaderHeader.c_str());
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
                        "layout(location=7) uniform ivec3 uOutputSize;\n"
                        "layout(location=8) uniform ivec3 uInputSize;\n"
                        "void retirePhase() { memoryBarrierShared(); barrier(); }\n"
                        "layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;\n"
                        "shared vec4 shared_mem[256]; \n"
                        "void main()\n"
                        "{\n"
                        "    ivec3 gid = ivec3(gl_GlobalInvocationID); \n"
                        "    ivec3 outputSize = uOutputSize;\n"
                        "    int tid = int(gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x); \n"
                        "    if (all(lessThan(gid, outputSize)))\n"
                        "    {\n"
                        "        int width  = uInputSize.x; \n"
                        "        int height = uInputSize.y; \n"
                        "        int thread_count = WORK_X * WORK_Y; \n"
                        "        ivec2 tg_size = ivec2(WORK_X, WORK_Y); \n"
                        "        vec4 sum = vec4(0.0f); \n"
                        "        for(int xIndex = gid.x; xIndex < width; xIndex += tg_size.x) { \n"
                        "            for(int yIndex = gid.y; yIndex < height; yIndex += tg_size.y) { \n"
                        "                #ifdef INPUT_TEXTURE_2D\n"
                        "                vec4 val = imageLoad(uInput, ivec2(xIndex, yIndex)); \n"
                        "                #else\n"
                        "                vec4 val = imageLoad(uInput, ivec3(xIndex, yIndex, gid.z)); \n"
                        "                #endif\n"
                        "                sum += val; \n"
                        "            } \n"
                        "        } \n"
                        "        shared_mem[tid] = sum; \n"
                        "        retirePhase(); \n"
                        "        sum = vec4(0.0f);; \n"
                        "        if (tid < 32) { \n"
                        "            for (int i = tid + 32; i < thread_count; i += 32) { \n"
                        "                sum += shared_mem[i]; \n"
                        "            } \n"
                        "        } \n"
                        "        shared_mem[tid] += sum; \n"
                        "        retirePhase(); \n"
                        "        // Calculate mean \n"
                        "        sum = vec4(0.0f);; \n"
                        "        if (tid == 0) { \n"
                        "            int top = min(int(32), thread_count); \n"
                        "            for (int i = 0; i < top; i += 1) { \n"
                        "                sum += shared_mem[i]; \n"
                        "            } \n"
                        "            shared_mem[0] = sum / float(width * height); \n"
                        "        } \n"
                        "        retirePhase(); \n"
                        "        vec4 mean = shared_mem[0]; \n"
                        "        retirePhase(); \n"
                        "        // Variance     \n"
                        "        sum = vec4(0.0f); \n"
                        "        for(int xIndex = gid.x; xIndex < width; xIndex += tg_size.x) { \n"
                        "            for(int yIndex = gid.y; yIndex < height; yIndex += tg_size.y) { \n"
                        "                #ifdef INPUT_TEXTURE_2D\n"
                        "                vec4 val = imageLoad(uInput, ivec2(xIndex, yIndex)); \n"
                        "                #else\n"
                        "                vec4 val = imageLoad(uInput, ivec3(xIndex, yIndex, gid.z)); \n"
                        "                #endif\n"
                        "                sum += (val-mean) * (val-mean); \n"
                        "            } \n"
                        "        } \n"
                        "        shared_mem[tid] = sum; \n"
                        "        retirePhase(); \n"
                        "        // Reduce to 32 values  \n"
                        "        sum = vec4(0.0f); \n"
                        "        if (tid < 32) { \n"
                        "            for (int i = tid + 32; i < thread_count; i += 32) { \n"
                        "                sum += shared_mem[i]; \n"
                        "            } \n"
                        "        } \n"
                        "        shared_mem[tid] += sum; \n"
                        "        retirePhase(); \n"
                        "        // Calculate variance   \n"
                        "        sum = vec4(0.0f); \n"
                        "        if (tid == 0) { \n"
                        "            int top = min(int(32), thread_count); \n"
                        "            for (int i = 0; i < top; i += 1) { \n"
                        "                sum += shared_mem[i]; \n"
                        "            } \n"
                        "            shared_mem[0] = sum / float(width * height); \n"
                        "        } \n"
                        "        retirePhase(); \n"
                        "        vec4 sigma = sqrt(shared_mem[0] + vec4(0.00001f)); \n"
                        "        vec4 multiplier = uGamma.data[gid.z] / sigma; \n"
                        "        for(int xIndex = gid.x; xIndex < width; xIndex += tg_size.x) { \n"
                        "            for(int yIndex = gid.y; yIndex < height; yIndex += tg_size.y) { \n"
                        // "                   vec4 val = vec4(0.0f); \n"
                        "                #ifdef INPUT_TEXTURE_2D\n"
                        "                vec4 val = imageLoad(uInput, ivec2(xIndex, yIndex)); \n"
                        "                #else\n"
                        "                vec4 val = imageLoad(uInput, ivec3(xIndex, yIndex, gid.z)); \n"
                        "                #endif\n"
                        // "                vec4 color = clamp((val - mean) * multiplier + uBeta.data[gid.z], -10.0f, 10.0f); \n"
                        "                vec4 color = (val - mean) * multiplier + uBeta.data[gid.z]; \n"
                        "                #ifdef RELU\n"
                        "                color = max(color, vec4(0));\n"
                        "                #endif\n"
                        "                #ifdef RELU6\n"
                        "                color = clamp(color, vec4(0), vec4(6));\n"
                        "                #endif\n"
                        "                #ifdef TANH\n"
                        "                color = tanh(color);\n"
                        "                #endif\n"
                        "                #ifdef SIGMOID\n"
                        "                color  = vec4(1.0f)/(vec4(1.0f)+ exp(-color));\n"
                        "                #endif\n"
                        "                #ifdef LEAKYRELU_VAL\n"
                        "                color   = max(color,  (color * vec4(LEAKYRELU_VAL)));\n"
                        "                #endif\n"
                        "                #ifdef SILU\n"
                        "                color    = color  * vec4(1.0f)/(vec4(1.0f)+ exp(-color));\n"
                        "                #endif\n"
                        "                #ifdef OUTPUT_TEXTURE_2D\n"
                        "                imageStore(uOutput, ivec2(xIndex, yIndex), color);\n"
                        "                #else\n"
                        "                imageStore(uOutput, ivec3(xIndex, yIndex, gid.z), color);\n"
                        "                #endif\n"
                        "            } \n"
                        "        }\n"
                        "    }\n"
                        "}\n";
    shaderMain = loadShader(INSTANCENORM_CS_ASSET_NAME);

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    uint32_t maxThreads   = 256;
    uint32_t threadWidth  = std::min(maxThreads, outputWidth);
    uint32_t threadHeight = std::min(maxThreads / threadWidth, outputHeight);
    std::vector<uint32_t> localSize {threadWidth, threadHeight, 1};

    SNN_LOGD("Test:%s:%d, %d, %d, %d, %d\n", __FUNCTION__, __LINE__, kernel, stride, ic_4, oc_4);

    shaderHeader += ("#define WORK_X " + std::to_string(localSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(localSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(localSize[2]) + "\n");
    // SNN_LOGI("Test:%s:%d, %s\n",__FUNCTION__,__LINE__, shaderHeader.c_str());

    pass.uniforms = {{"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)}, {"uInputSize", glm::ivec3(inputWidth, inputHeight, ic_4)}};

    pass.inputs = {{"uInput", 0}};

    pass.program = InferenceGraph::Pass::CsProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {1, 1, UP_DIV(oc_4, localSize[2])}};
    // SNN_LOGI("Test:%s:%d, %s\n",__FUNCTION__,__LINE__, shaderUniforms.c_str());
    string tmpStr = shaderHeader + shaderUniforms;
    pass.source   = (tmpStr + shaderMain);
    // SNN_LOGI("Test:%s:%d, %s, %d\n",__FUNCTION__,__LINE__, shaderMain.c_str(), pass.source.length());

    SNN_LOGD("%s:%d, input:%d:%d:%d, output:%d:%d:%d\n", __FUNCTION__, __LINE__, inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    // SNN_LOGI("Test:%s:%d, %s\n",__FUNCTION__,__LINE__, pass.source.c_str());

    if (1) {
        float* bnDataDest;
        std::vector<float> bnDataVector;
        std::string bnString;
        pass._vecBeta.resize(_desc.numOutputPlanes);
        pass._bnBeta.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnBeta->allocate(_desc.numOutputPlanes, pass._vecBeta.data());
        bnDataDest   = (float*) pass._bnBeta->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "beta";
        bnDataVector = _desc.instanceNormalization.at(bnString);
        SNN_LOGD("Beta:%s:%d \n", __FUNCTION__, __LINE__);

        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnBeta->unmap();
        pass.ssboMap[5] = pass._bnBeta->getId();

        pass._vecGamma.resize(_desc.numOutputPlanes);
        pass._bnGamma.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        pass._bnGamma->allocate(_desc.numOutputPlanes, pass._vecGamma.data());
        bnDataDest   = (float*) pass._bnGamma->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        bnString     = "gamma";
        bnDataVector = _desc.instanceNormalization.at(bnString);
        SNN_LOGD("Gamma:%s:%d \n", __FUNCTION__, __LINE__);

        memcpy(bnDataDest, bnDataVector.data(), _desc.numOutputPlanes * 4);
        pass._bnGamma->unmap();
        pass.ssboMap[6] = pass._bnGamma->getId();
    }

    return ret;
}
