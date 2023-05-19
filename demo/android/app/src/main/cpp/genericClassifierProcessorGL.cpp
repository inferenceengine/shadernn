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
#include "genericClassifierProcessorGL.h"
#include "glAppContext.h"
#include "processor.h"
#include "glUtils.h"
#include "ic2/dp.h"
#include "snn/utils.h"
#include "imageTextureGL.h"
#include <glad/glad.h>
#include "snn/glImageHandle.h"

using namespace snn;

static gl::TextureObject resizedInputTex;
static bool texNotAllocated = true;

static void preProcessTexture(GLuint& inId, GLuint& outId, int scaleX, int scaleY, int inWidth, int inHeight, int outWidth, int outHeight,
    float modelInputMean, float modelInputNorm) {
    std::string sourceCode = std::string(
                            "#version 320 es \n"
                            "#define PRECISION mediump\n"
                            "precision PRECISION float;\n"
                            "layout(rgba8, binding=0) readonly uniform PRECISION image2D uInput;\n"
                            "layout(rgba32f, binding=1) writeonly uniform PRECISION image2D uOutput;\n"
                            "layout(location=2) uniform ivec4 inImgSize;\n"
                            "layout(location=3) uniform ivec4 outImgSize;\n"
                            "layout(location=4) uniform vec2 scale;\n"
                            "layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
                            "void main()\n"
                            "{\n"
                            "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
                            "    ivec3 inputImgSize = inImgSize.xyz;\n"
                            "    ivec3 outputImgSize = outImgSize.xyz;\n"
                            "    \n"
                            "    if(pos.x < outputImgSize.x && pos.y < outputImgSize.y)\n"
                            "    {\n"
                            "        float srcX = float(pos.x) * scale.x;\n"
                            "        int x1 = int(floor(srcX));\n"
                            "        int x11 = clamp(x1, 0, inputImgSize.x - 1);\n"
                            "        \n"
                            "        float srcY = float(pos.y) * scale.y;\n"
                            "        int y1 = int(floor(srcY));\n"
                            "        int y11 = clamp(y1, 0, inputImgSize.y - 1);\n"
                            "        \n"
                            "        vec4 outValue = imageLoad(uInput, ivec2(x11, y11));\n"
                            "        \n"
                            ) + ((modelInputMean != 0.0f && modelInputNorm != 1.0f) ?
                            "        outValue = (outValue - vec4(" + std::to_string(modelInputMean) + ")) * vec4(" + std::to_string(modelInputNorm) + ");\n"
                            :
                            "")
                            +
                            "        imageStore(uOutput, ivec2(pos.x, pos.y), outValue);\n"
                            "    }\n"
                            "    \n"
                            "}\n";

    gl::SimpleGlslProgram csProgram;
    csProgram.loadCs(sourceCode.c_str());
    csProgram.use();

    glBindImageTexture(0, inId, 0, true, 0, GL_READ_ONLY, GL_RGBA8);
    CHECK_GL_ERROR("glBindImageTexture");
    glBindImageTexture(1, outId, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
    CHECK_GL_ERROR("glBindImageTexture");

    glUniform4i(2, inWidth, inHeight, 4, 1);
    glUniform4i(3, outWidth, outHeight, 4, 1);
    glUniform2f(4, scaleX, scaleY);

    glFinish();
    glDispatchCompute(1920 / 8, 1080 / 8, 1);
}

static void postProcessTexture(GLuint& inId, GLuint& outId, int scaleX, int scaleY, int inWidth, int inHeight, int outWidth, int outHeight) {
    std::string sourceCode = "#version 320 es \n"
                             "#define PRECISION mediump\n"
                             "precision PRECISION float;\n"
                             "layout(rgba8, binding=0) readonly uniform PRECISION image2D uInput;\n"
                             "layout(rgba8, binding=1) writeonly uniform PRECISION image2D uOutput;\n"
                             "layout(location=2) uniform ivec4 inImgSize;\n"
                             "layout(location=3) uniform ivec4 outImgSize;\n"
                             "layout(location=4) uniform vec2 scale;\n"
                             "layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
                             "void main()\n"
                             "{\n"
                             "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
                             "    ivec3 inputImgSize = inImgSize.xyz;\n"
                             "    ivec3 outputImgSize = outImgSize.xyz;\n"
                             "    \n"
                             "    if(pos.x < outputImgSize.x && pos.y < outputImgSize.y)\n"
                             "    {\n"
                             "        float srcX = float(pos.x) * scale.x;\n"
                             "        int x1 = int(floor(srcX));\n"
                             "        int x11 = clamp(x1, 0, inputImgSize.x - 1);\n"
                             "        \n"
                             "        float srcY = float(pos.y) * scale.y;\n"
                             "        int y1 = int(floor(srcY));\n"
                             "        int y11 = clamp(y1, 0, inputImgSize.y - 1);\n"
                             "        \n"
                             "        vec4 outValue = imageLoad(uInput, ivec2(x11, y11));\n"
                             "        \n"
                             "        outValue = outValue;"
                             "        imageStore(uOutput, ivec2(pos.x, pos.y), outValue);\n"
                             "    }\n"
                             "    \n"
                             "}\n";

    gl::SimpleGlslProgram csProgram;
    csProgram.loadCs(sourceCode.c_str());
    csProgram.use();

    glBindImageTexture(0, inId, 0, true, 0, GL_READ_ONLY, GL_RGBA8);
    CHECK_GL_ERROR("glBindImageTexture");
    glBindImageTexture(1, outId, 0, true, 0, GL_WRITE_ONLY, GL_RGBA8);
    CHECK_GL_ERROR("glBindImageTexture");

    glUniform4i(2, inWidth, inHeight, 4, 1);
    glUniform4i(3, outWidth, outHeight, 4, 1);
    glUniform2f(4, scaleX, scaleY);

    glFinish();
    glDispatchCompute(1920 / 8, 1080 / 8, 1);
}

void GenericClassifierProcessorGL::submit(Processor::Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    auto& inputDesc = workload.inputs[0]->desc();

    GlImageHandle inputTexture;
    workload.inputs[0]->getGpuImageHandle(inputTexture);
    GlImageHandle outputTexture;
    workload.output->getGpuImageHandle(outputTexture);

    if (texNotAllocated) {
        resizedInputTex.allocate2D(snn::ColorFormat::RGBA32F, modelProcessorParams.modelDims.width, modelProcessorParams.modelDims.height);
        texNotAllocated = false;
        SNN_LOGD("reset: id = %d, target = %d, format = %d", resizedInputTex.id(), resizedInputTex.target(), (int)resizedInputTex.getDesc().format);
    }

    GLuint resizedTexId = resizedInputTex.id();
    preProcessTexture(inputTexture.textureId, resizedTexId, inputDesc.width / (float)modelProcessorParams.modelDims.width,
        inputDesc.height / (float)modelProcessorParams.modelDims.height, inputDesc.width, inputDesc.height,
        modelProcessorParams.modelDims.width, modelProcessorParams.modelDims.height, modelInputMean, modelInputNorm);

    const auto& outputDesc = workload.output->desc();

    if (!modelProcessorParams.ic2) {
        dp::ShaderGenOptions options = {};
        auto inputTex = InferenceGraph::IODesc {inputDesc.format, modelProcessorParams.modelDims.width,
                                                    modelProcessorParams.modelDims.height, 1, 4};
        options.desiredInput.push_back(inputTex);

        options.compute              = modelProcessorParams.compute;
        options.desiredOutputFormat  = inputDesc.format;
        options.preferrHalfPrecision = modelProcessorParams.precision == Precision::FP16;

        options.mrtMode    = snn::MRTMode::SINGLE_PLANE;
        options.weightMode = weightMode;

        auto dp = snn::dp::loadFromJsonModel(modelProcessorParams.modelFileName, false, options.mrtMode, options.weightMode, options.preferrHalfPrecision);

        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);

        cp.dumpOutputs = modelProcessorParams.dumpOutputs;
        modelProcessorParams.ic2 = MixedInferenceCore::create(GlAppContext::getGlContext(), cp);
    }

    SNN_ASSERT(inputDesc.device == Device::GPU);

    MixedInferenceCore::RunParameters rp = {};
    rp.inputMatrix                       = workload.cpuInputs;
    rp.modelOutput.modelType             = ModelType::CLASSIFICATION;

    snn::ImageTextureGLArray inputTexs;
    inputTexs.allocate(1);
    inputTexs[0].texture(0)->attach(resizedInputTex);
    rp.inputImages = inputTexs;

    modelProcessorParams.ic2->run(rp);

    postProcessTexture(inputTexture.textureId, outputTexture.textureId, inputDesc.width/(float)outputDesc.width, inputDesc.height/(float)outputDesc.height,
        inputDesc.width, inputDesc.height, outputDesc.width, outputDesc.height);

    workload.modelOutput = rp.modelOutput;
}
