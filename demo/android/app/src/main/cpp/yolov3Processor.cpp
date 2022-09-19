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
#include "yolov3Processor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include "ic2/dp.h"
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include "inferenceProcessor.h"
#include "BoundingBoxUtil.h"

using namespace std;
using namespace snn;
extern AAssetManager* g_assetManager;

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinytest(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    // load input image
    auto ip = new snn::InferenceProcessor();
    //ip->registerLayer("YOLO", YOLOCreator);

    auto modelFileName = "yolov3-tiny.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {416, 416, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::TEXTURES, false, false, true});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.deallocate();
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        auto input = snn::ManagedRawImage::loadFromAsset("images/arduino.png");
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        auto input32f = snn::toRgba32f(input, 0.0, 1.0);
        gl::TextureObject scaleTex;
        scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 1, 4);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        scaleTex.setPixels(0, 0, 0, 416, 416, 0, input32f.data());
        scaleTex.detach();
        printf("%s:%d tex:%d, %d\n", __FUNCTION__, __LINE__, scaleTex.target(), scaleTex.id());
        inputTexs.allocate(1);
        inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
#ifdef __ANDROID__
        outputTexs[0].texture(0)->allocate2DArray(snn::ColorFormat::RGBA32F, 1, 1, 256, 1024, 1);
#endif
        ip->process(outputTexs);
        scaleTex.cleanup();
    }

    ip->finalize();

    // test done successfully.
    return 0;
}

static gl::TextureObject resizedInputTex;
static bool texNotAllocated = true;

static void processTextureResize(GLuint& inId, bool inFloat, GLuint& outId, bool outFloat, float scaleX, float scaleY, int inWidth, int inHeight, int outWidth, int outHeight) {
    std::string sourceCode = "#version 320 es \n"
                             "#define PRECISION mediump\n"
                             "precision PRECISION float;\n";
    if (inFloat) {
        sourceCode += "layout(rgba32f, binding=0) readonly uniform PRECISION image2D uInput;\n";
    } else {
        sourceCode += "layout(rgba8, binding=0) readonly uniform PRECISION image2D uInput;\n";
    }

    if (outFloat) {
        sourceCode += "layout(rgba32f, binding=1) writeonly uniform PRECISION image2D uOutput;\n";
    } else {
        sourceCode += "layout(rgba8, binding=1) writeonly uniform PRECISION image2D uOutput;\n";
    }

    sourceCode += "layout(location=2) uniform ivec4 inImgSize;\n"
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

    if (inFloat) {
        glBindImageTexture(0, inId, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
    } else {
        glBindImageTexture(0, inId, 0, true, 0, GL_READ_ONLY, GL_RGBA8);
    }
    CHECK_GL_ERROR("glBindImageTexture");

    if (outFloat) {
        glBindImageTexture(1, outId, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
    } else {
        glBindImageTexture(1, outId, 0, true, 0, GL_WRITE_ONLY, GL_RGBA8);
    }
    CHECK_GL_ERROR("glBindImageTexture");

    glUniform4i(2, inWidth, inHeight, 4, 1);
    glUniform4i(3, outWidth, outHeight, 4, 1);
    glUniform2f(4, scaleX, scaleY);

    glFinish();
    glDispatchCompute(1920 / 8, 1080 / 8, 1);
}

void Yolov3Processor::submit(Workload& workload) {
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

    processTextureResize(inputTexId, false,resizedTexId, true, inputDesc.width / (float)expectedWidth, inputDesc.height / (float)expectedHeight, inputDesc.width, inputDesc.height, expectedWidth, expectedHeight);

    const auto& outputDesc = workload.output->desc();

    if (!ic2_) {
        dp::ShaderGenOptions options = {};
        options.desiredInput.width   = expectedWidth;
        options.desiredInput.height  = expectedHeight;
        options.desiredInput.depth   = 1;
        options.desiredInput.format  = inputDesc.format;
        options.compute              = true;
        options.desiredOutputFormat  = inputDesc.format;
        options.preferrHalfPrecision = false;

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
    rp.modelOutput.modelType             = InferenceEngine::ModelType::DETECTION;
    ic2_->run(rp);

    BoundingBoxUtil bbUtil;

    for(auto &boxDetails : rp.modelOutput.detectionOutput) {
        float confidence = boxDetails.at(1);

        if (confidence >= 0.4) {
            float x = boxDetails.at(2);
            float y = boxDetails.at(3);
            float w = boxDetails.at(4);
            float h = boxDetails.at(5);

            float TL_x = (x - w / 2.f);
            float TL_y = (y + h / 2.f);
            float BR_x = (x + w / 2.f);
            float BR_y = (y - h / 2.f);

            float vertices[] = {TL_x, TL_y, BR_x, BR_y};
            if (w > 0.01 && h > 0.01) {
                bbUtil.drawBoundingBox(resizedInputTex.target(), resizedInputTex.id(), vertices);
            }
        }
    }

    processTextureResize(resizedTexId, true, outputTexture.texture, false, expectedWidth / (float) inputDesc.width, expectedHeight / (float) inputDesc.height, expectedWidth, expectedHeight, inputDesc.width, inputDesc.height);
}
