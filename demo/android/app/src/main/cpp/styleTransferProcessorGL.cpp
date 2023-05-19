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
#include "styleTransferProcessorGL.h"
#include "glUtils.h"
#include "imageTextureGL.h"
#include <glad/glad.h>
#include "snn/glImageHandle.h"
#include "glAppContext.h"
#include "ic2/dp.h"

using namespace std;
using namespace snn;

static gl::TextureObject resizedInputTex;
static gl::TextureObject modelOutputTex;
static bool texNotAllocated = true;

static void preprocessTexture(GLuint& inId, GLuint& outId, int scaleX, int scaleY, int inWidth, int inHeight, int outWidth, int outHeight) {
    std::string sourceCode = "#version 320 es \n"
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
                             "        vec4 outValue = imageLoad(uInput, ivec2(x11, y11)) * vec4(255.f);\n"
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
    glBindImageTexture(1, outId, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
    CHECK_GL_ERROR("glBindImageTexture");

    glUniform4i(2, inWidth, inHeight, 4, 1);
    glUniform4i(3, outWidth, outHeight, 4, 1);
    glUniform2f(4, scaleX, scaleY);

    glFinish();
    glDispatchCompute(1920 / 8, 1080 / 8, 1);
}

static void postProcessTexture(GLuint& inId, GLuint& outId, int inWidth, int inHeight, int outWidth, int outHeight, float scaleX, float scaleY) {
    std::string sourceCode = "#version 320 es \n"
                             "#define PRECISION mediump\n"
                             "precision PRECISION float;\n"
                             "layout(rgba32f, binding=0) readonly uniform PRECISION image2D uInput;\n"
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
                             "        vec4 outValue = imageLoad(uInput, ivec2(x11, y11)) / vec4(255.f);"
                             "        \n"
                             "        imageStore(uOutput, ivec2(pos.x, pos.y), outValue);\n"
                             "    }\n"
                             "    \n"
                             "}\n";

    gl::SimpleGlslProgram csProgram;
    csProgram.loadCs(sourceCode.c_str());
    csProgram.use();

    GLCHK(glBindImageTexture(0, inId, 0, true, 0, GL_READ_ONLY, GL_RGBA32F));
    GLCHK(glBindImageTexture(1, outId, 0, true, 0, GL_WRITE_ONLY, GL_RGBA8));

    GLCHK(glUniform4i(2, inWidth, inHeight, 4, 1));
    GLCHK(glUniform4i(3, outWidth, outHeight, 4, 1));
    GLCHK(glUniform2f(4, scaleX, scaleY));

    glFinish();
    GLCHK(glDispatchCompute(1920 / 8, 1080 / 8, 1));
}

void StyleTransferProcessorGL::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    const auto& inputDesc = workload.inputs[0]->desc();
    SNN_ASSERT(inputDesc.device == Device::GPU);

    // we have to delay creating ic2 because we need to know the frame size.
    if (!ic2) {
        dp::ShaderGenOptions options = {};
        options.preferrHalfPrecision = precision == Precision::FP16;
        auto inputTex = InferenceGraph::IODesc {options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F,
                                                modelDims.width, modelDims.height, 1, 4};
        options.desiredInput.push_back(inputTex);
        options.desiredOutputFormat  = options.preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;
        options.compute              = compute;
        options.mrtMode              = snn::MRTMode::SINGLE_PLANE;
        options.weightMode           = snn::WeightAccessMethod::CONSTANTS;
        auto dp                      = snn::dp::loadFromJsonModel(modelFileName, false, options.mrtMode, options.weightMode, options.preferrHalfPrecision);
        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp[0], options);
        cp.dumpOutputs = false;
        ic2                   = MixedInferenceCore::create(GlAppContext::getGlContext(), cp);
    }

    GlImageHandle inputTexture;
    workload.inputs[0]->getGpuImageHandle(inputTexture);
    GlImageHandle outputTexture;
    workload.output->getGpuImageHandle(outputTexture);

    if (texNotAllocated) {
        resizedInputTex.allocate2D(snn::ColorFormat::RGBA32F, modelDims.width, modelDims.height);
        modelOutputTex.allocate2D(snn::ColorFormat::RGBA32F, modelDims.width, modelDims.height);
        texNotAllocated = false;
    }

    GLuint resizedTexId = resizedInputTex.id();
    preprocessTexture(inputTexture.textureId, resizedTexId, inputDesc.width / (float)modelDims.width, inputDesc.height / (float)modelDims.height,
        inputDesc.width, inputDesc.height, modelDims.width, modelDims.height);

    MixedInferenceCore::RunParameters rp{};

    ImageTextureGLArray inputImageTexs;
    inputImageTexs.allocate(1);
    inputImageTexs[0].texture(0)->attach(resizedInputTex);
    rp.inputImages = inputImageTexs;

    ImageTextureGLArray outputImageTexs;
    outputImageTexs.allocate(1);
    GLuint modelOutId = modelOutputTex.id();
    outputImageTexs[0].texture(0)->attach(modelOutputTex);
    rp.outputImages = outputImageTexs;

    ic2->run(rp);

    float scaleX = modelDims.width / (float)inputDesc.width;
    float scaleY = modelDims.height / (float)inputDesc.height;
    SNN_LOGD("Output texture id = %d", outputTexture.textureId);
    postProcessTexture(modelOutId, outputTexture.textureId, modelDims.width, modelDims.height, inputDesc.width, inputDesc.height, scaleX, scaleY);
}
