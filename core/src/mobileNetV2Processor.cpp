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
#include "mobileNetV2Processor.h"
#include "ic2/dp.h"
#include "snn/utils.h"

using namespace std;
using namespace snn;

static gl::TextureObject resizedInputTex;
static bool texNotAllocated = true;

static void preProcessTexture(GLuint& inId, GLuint& outId, int scaleX, int scaleY, int inWidth, int inHeight, int outWidth, int outHeight) {
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
                             "        vec4 outValue = imageLoad(uInput, ivec2(x11, y11));\n"
                             "        \n"
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

void MobileNetV2Processor::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    auto& inputDesc = workload.inputs[0]->desc();
    auto inputGpuData   = workload.inputs[0]->getGpuData();

    auto inputTexture  = ((GpuFrameImage*) workload.inputs[0])->getGpuData();
    auto outputTexture = ((GpuFrameImage*) workload.output)->getGpuData();

    gl::TextureObject inputTex;
    inputTex.attach(inputTexture.target, inputTexture.texture);

    if (texNotAllocated) {
        resizedInputTex.allocate2D(snn::ColorFormat::RGBA32F, expectedWidth, expectedHeight);
        texNotAllocated = false;
    }

    GLuint inputTexId   = inputTex.id();
    GLuint resizedTexId = resizedInputTex.id();

    preProcessTexture(inputTexId, resizedTexId, inputDesc.width / (float)expectedWidth, inputDesc.height / (float)expectedHeight, inputDesc.width, inputDesc.height, expectedWidth, expectedHeight);

    const auto& outputDesc = workload.output->desc();

    if (!ic2_) {
        dp::ShaderGenOptions options = {};
        options.desiredInput.width   = expectedWidth;
        options.desiredInput.height  = expectedHeight;
        options.desiredInput.depth   = 1;
        options.desiredInput.format  = inputDesc.format;
        options.compute              = true;
        options.desiredOutputFormat  = inputDesc.format;
        options.preferrHalfPrecision = true;

        options.mrtMode    = snn::MRTMode::SINGLE_PLANE;
        options.weightMode = snn::WeightAccessMethod::TEXTURES;

        auto dp = snn::dp::loadFromJsonModel(modelFileName_, options.mrtMode, options.weightMode, options.preferrHalfPrecision);

        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);

        cp.dumpOutputs         = this->dumpOutputs;
        ic2_                   = MixedInferenceCore::create(cp);
    }

    SNN_ASSERT(inputDesc.device == Device::GPU);

    MixedInferenceCore::RunParameters rp = {};
    auto inputTextures                   = getFrameTexture(resizedTexId);
    rp.inputTextures                     = &inputTextures;
    rp.inputCount                        = 1;
    rp.inputMatrix                       = workload.cpuInputs;
    rp.output                            = std::vector<std::vector<std::vector<float>>>();
    rp.modelOutput.modelType             = InferenceEngine::ModelType::CLASSIFICATION;
    ic2_->run(rp);

    postProcessTexture(inputTexId, outputTexture.texture, inputDesc.width/(float)outputDesc.width, inputDesc.height/(float)outputDesc.height, inputDesc.width, inputDesc.height, outputDesc.width, outputDesc.height);

    workload.modelOutput = rp.modelOutput;
}
