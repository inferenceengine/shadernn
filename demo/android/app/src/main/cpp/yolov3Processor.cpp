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

using namespace std;
using namespace snn;
extern AAssetManager* g_assetManager; 

DECLARE_LAYER(YOLO);

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinytest(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    // load input image
    auto ip = new snn::InferenceProcessor();
    ip->registerLayer("YOLO", YOLOCreator);

    auto modelFileName = "yolov3-tiny.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {416, 416, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::TEXTURES, false, false, false});

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

void Yolov3Processor::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    const auto& inputDesc  = workload.inputs[0]->desc();
    const auto& outputDesc = workload.output->desc();

    // std::cout << "Output Width: " << outputDesc.width << std::endl;
    // std::cout << "Output Height: " << outputDesc.height << std::endl;

    if (!ic2_) {
        // for (std::size_t layerIdx = 0; layerIdx < dp.size(); layerIdx++) {
        //     std::cout << dp.at(layerIdx)->getName() << std::endl;
        // }

        dp::ShaderGenOptions options = {};
        options.desiredInput.width   = inputDesc.width;
        options.desiredInput.height  = inputDesc.height;
        options.desiredInput.depth   = 1;
        options.desiredInput.format  = outputDesc.format;
        options.compute              = this->compute_;

        options.desiredOutputFormat  = inputDesc.format;
        options.preferrHalfPrecision = false;

        options.mrtMode    = snn::MRTMode::DOUBLE_PLANE;
        options.weightMode = snn::WeightAccessMethod::TEXTURES;

        auto dp = snn::dp::loadFromJsonModel(modelFileName_, options.mrtMode, options.weightMode, options.preferrHalfPrecision);
        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);
        cp.dumpOutputs         = this->dumpOutputs;
        ic2_                   = MixedInferenceCore::create(cp);
    }

    SNN_ASSERT(inputDesc.device == Device::GPU);
    SNN_ASSERT(outputDesc.device == Device::GPU);
    auto inputTexture  = ((GpuFrameImage*) workload.inputs[0])->getGpuData();
    auto outputTexture = ((GpuFrameImage*) workload.output)->getGpuData();

    auto& currentFrameTexture = frameTextures_[inputTexture.texture];
    if (!currentFrameTexture) {
        currentFrameTexture = Texture::createAttached(inputTexture.target, inputTexture.texture); // Create a thin shell around textureId
    }

    std::vector<std::vector<std::vector<float>>> inVec, outVec = std::vector<std::vector<std::vector<float>>>();

    MixedInferenceCore::RunParameters rp = {};
    auto inputTextures                   = getFrameTexture(inputTexture.texture);
    rp.inputTextures                     = &inputTextures;
    rp.inputCount                        = 1;
    rp.inputMatrix                       = inVec;
    rp.output                            = outVec;
    // rp.transitionOutput = getFrameTexture(transitionOutputTexture.texture);
    rp.textureOut = getFrameTexture(outputTexture.texture);
    ic2_->run(rp);
    // ic2_->dumpStageOutputs("/data/data/com.innopeaktech.seattle.snndemo/files/yolov3_1");
}
