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
#include "../../../../../common/demoutils.h"
#include "nativeFrameProvider.h"
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <string>
#include <thread>
#include <chrono>
#include <queue>
#include <sys/system_properties.h>
#include <opencv2/opencv.hpp>
#include "src/inferenceProcessor.h"

#define APP_DIR "/sdcard/Android/data/com.innopeaktech.seattle.snndemo"

static const char* const OFF_STR = "0";
using namespace snn;

static std::unique_ptr<MainProcessingLoop> g_mainLoop;
static std::unique_ptr<NativeFrameProvider> g_frameProvider;

extern AAssetManager* g_assetManager; // defined in utils.cpp

// -----------------------------------------------------------------------------
// A utility class to retrieve native pointer from jbyteArray
struct AutoByteArray {
    JNIEnv* env;
    jbyteArray arr;
    uint8_t* ptr;

    AutoByteArray(JNIEnv* e, jbyteArray a): env(e), arr(a), ptr((uint8_t*) e->GetByteArrayElements(a, nullptr)) {}

    ~AutoByteArray() { env->ReleaseByteArrayElements(arr, (jbyte*) ptr, JNI_ABORT); }
};

auto toNativeAlgorithmConfig(JNIEnv* env, jobject jalgo) {
    jclass algoCfgClass = env->FindClass("com/oppo/seattle/snndemo/AlgorithmConfig");
    bool dumpOutputs    = false;
    InferenceEngine::AlgorithmConfig algo;
    InferenceEngine::ModelType modelType = InferenceEngine::ModelType::OTHER;
    { // denoise
        jmethodID isDenoiseNONE            = env->GetMethodID(algoCfgClass, "isDenoiseNONE", "()Z");
        jmethodID isDenoiseSPATIALDENOISER = env->GetMethodID(algoCfgClass, "isDenoiseSPATIALDENOISER", "()Z");

        if (env->CallBooleanMethod(jalgo, isDenoiseNONE)) {
        } else if (env->CallBooleanMethod(jalgo, isDenoiseSPATIALDENOISER)) {
            algo.denoisers = InferenceEngine::AlgorithmConfig::Denoisers {InferenceEngine::AlgorithmConfig::Denoisers::Denoiser::FRAGMENTSHADER,
                                                                          InferenceEngine::AlgorithmConfig::Denoisers::DenoiserAlgorithm ::SPATIALDENOIRSER};
        } else {
            throw std::runtime_error("no such denoiser");
        }
    }

    {
        // classifiers
        jmethodID isClassifierNONE           = env->GetMethodID(algoCfgClass, "isClassifierNONE", "()Z");
        jmethodID isClassifierResnet18       = env->GetMethodID(algoCfgClass, "isClassifierResnet18", "()Z");
        jmethodID isClassifierMobilenetv2    = env->GetMethodID(algoCfgClass, "isClassifierMobilenetv2", "()Z");

        if (env->CallBooleanMethod(jalgo, isClassifierResnet18) || env->CallBooleanMethod(jalgo, isClassifierMobilenetv2)) {
            modelType = InferenceEngine::ModelType::CLASSIFICATION;
        }

        if (env->CallBooleanMethod(jalgo, isClassifierNONE)) {
        } else if (env->CallBooleanMethod(jalgo, isClassifierResnet18)) {
            // Connect this part with snn.h's algo struct
            SNN_LOGI("resnet18 compute shader");
            algo.classifiers =
                    InferenceEngine::AlgorithmConfig::Classifiers {dumpOutputs, InferenceEngine::AlgorithmConfig::Classifiers::Classifier::COMPUTESHADER,
                                                                   InferenceEngine::AlgorithmConfig::Classifiers::ClassifierAlgorithm::RESNET18};
        }  else if (env->CallBooleanMethod(jalgo, isClassifierMobilenetv2)) {
            // Connect this part with snn.h's algo struct
            SNN_LOGI("Mobilenetv2 compute shader");
            algo.classifiers =
                    InferenceEngine::AlgorithmConfig::Classifiers {dumpOutputs, InferenceEngine::AlgorithmConfig::Classifiers::Classifier::COMPUTESHADER,
                                                                   InferenceEngine::AlgorithmConfig::Classifiers::ClassifierAlgorithm::MOBILENETV2};
        }
    }
    {
        // detections
        jmethodID isDetectionNONE           = env->GetMethodID(algoCfgClass, "isDetectionNONE", "()Z");
        jmethodID isDetectionYolov3         = env->GetMethodID(algoCfgClass, "isDetectionYolov3", "()Z");

        if (env->CallBooleanMethod(jalgo, isDetectionYolov3)) {
            modelType = InferenceEngine::ModelType::DETECTION;
        }

        if (env->CallBooleanMethod(jalgo, isDetectionNONE)) {
        } else if (env->CallBooleanMethod(jalgo, isDetectionYolov3)) {
            // Connect this part with snn.h's algo struct
            SNN_LOGI("yolov3 compute shader");
            algo.detections =
                    InferenceEngine::AlgorithmConfig::Detections {InferenceEngine::AlgorithmConfig::Detections::Detection::COMPUTESHADER,
                                                                  InferenceEngine::AlgorithmConfig::Detections::DetectionAlgorithm::YOLOV3, dumpOutputs};
        }
    }

    {
        jmethodID isStyleTransferNONE         = env->GetMethodID(algoCfgClass, "isStyleTransferNONE", "()Z");
        jmethodID isStyleTransferCANDY        = env->GetMethodID(algoCfgClass, "isStyleTransferCANDY", "()Z");
        jmethodID isStyleTransferMOSAIC       = env->GetMethodID(algoCfgClass, "isStyleTransferMOSAIC", "()Z");
        jmethodID isStyleTransferPOINTILISM   = env->GetMethodID(algoCfgClass, "isStyleTransferPOINTILISM", "()Z");
        jmethodID isStyleTransferRAINPRINCESS = env->GetMethodID(algoCfgClass, "isStyleTransferRAINPRINCESS", "()Z");
        jmethodID isStyleTransferUDNIE        = env->GetMethodID(algoCfgClass, "isStyleTransferUDNIE", "()Z");

        if (env->CallBooleanMethod(jalgo, isStyleTransferNONE)) {
        } else if (env->CallBooleanMethod(jalgo, isStyleTransferCANDY)) {
            algo.styleTransferModels =
                InferenceEngine::AlgorithmConfig::StyleTransfer {InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::CANDY};
        } else if (env->CallBooleanMethod(jalgo, isStyleTransferMOSAIC)) {
            algo.styleTransferModels =
                InferenceEngine::AlgorithmConfig::StyleTransfer {InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::MOSAIC};
        } else if (env->CallBooleanMethod(jalgo, isStyleTransferPOINTILISM)) {
            algo.styleTransferModels =
                InferenceEngine::AlgorithmConfig::StyleTransfer {InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::POINTILISM};
        } else if (env->CallBooleanMethod(jalgo, isStyleTransferRAINPRINCESS)) {
            algo.styleTransferModels =
                InferenceEngine::AlgorithmConfig::StyleTransfer {InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::RAIN_PRINCESS};
        } else if (env->CallBooleanMethod(jalgo, isStyleTransferUDNIE)) {
            algo.styleTransferModels =
                InferenceEngine::AlgorithmConfig::StyleTransfer {InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::UDNIE};
        } else {
            throw std::runtime_error("no such style transfer model");
        }
    }

    std::pair<InferenceEngine::AlgorithmConfig, InferenceEngine::ModelType> retVal = {algo, modelType};

    return retVal;
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_init(JNIEnv* env, jobject, jobject java_am, jstring internalStorageDir,
                                                                                   jstring externalStorageDir) {
    (void) internalStorageDir;
    (void) externalStorageDir;

    // remember the pointer to the asset manager
    SNN_CHK(env);
    SNN_CHK(java_am);
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    g_frameProvider.reset(new NativeFrameProvider(4));
    // delay main loop creation until we know the size of the frame.
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_initDSP(JNIEnv* env, jobject, jstring jSkelLocation) {
    const char* skelLocation = env->GetStringUTFChars(jSkelLocation, 0);
    setenv("ADSP_LIBRARY_PATH", skelLocation, 1);
    // SNN_LOGD("ADSP_LIBRARY_PATH=%s", skelLocation);
    env->ReleaseStringUTFChars(jSkelLocation, skelLocation);
}

// -----------------------------------------------------------------------------
//
static int vizW = 0, vizH = 0;
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_resize(JNIEnv*, jobject, int w, int h) {
    if (g_mainLoop) {
        g_mainLoop->resize(w, h);
    } else {
        vizW = w;
        vizH = h;
    }
}

// -----------------------------------------------------------------------------
//
uint32_t frameW = 0, frameH = 0;
extern "C" JNIEXPORT void JNICALL

Java_com_oppo_seattle_snndemo_NativeLibrary_draw(JNIEnv* env, jclass, jobject jAlgo) {
    // jstring jstr = (*env)->NewStringUTF(env, "This comes from jni.");

    if (0 == frameW || 0 == frameH) {
        return;
    }
    if (!g_mainLoop) {
        g_mainLoop.reset(new MainProcessingLoop({frameW, frameH, g_frameProvider.get()}));
        if (vizW > 0) {
            g_mainLoop->resize(vizW, vizH);
        }
    }

    auto algoConfig = toNativeAlgorithmConfig(env, jAlgo);

    static const char* currentEvSetting = nullptr;
    const char* evSetting               = OFF_STR;

    if (evSetting != currentEvSetting) {
        __system_property_set("vendor.oppo.stream.PVEVenable", evSetting);
        currentEvSetting = evSetting;
    }

    MainProcessingLoop::RenderParameters rp;
    rp.playing   = true;
    rp.algorithm = algoConfig.first;
    rp.modelType = algoConfig.second;

    g_mainLoop->render(rp);

    if (rp.modelType == InferenceEngine::ModelType::CLASSIFICATION) {
        jclass cls = env->GetObjectClass(jAlgo);
        env->SetIntField(jAlgo, env->GetFieldID(cls, "classifierIndex", "I"), rp.modelOutput.classifierOutput);
    }
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_destroy(JNIEnv*, jobject) {
    g_mainLoop.reset();
    g_frameProvider.reset();
}

// -----------------------------------------------------------------------------
//

extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_queueFrame(JNIEnv* env, jclass, jint w, jint h, jint rotation_degrees,
                                                                                         jlong timestamp, jobject y_plane, jobject u_plane, jobject v_plane) {
    (void) rotation_degrees;
    (void) timestamp;

    if (0 == frameW) {
        frameW = (uint32_t) w;
        frameH = (uint32_t) h;
    }

    const uint8_t* yPlaneData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(y_plane));
    uint32_t width            = static_cast<uint32_t>(w);
    uint32_t height           = static_cast<uint32_t>(h);
    uint32_t fullPlaneSize    = width * height;
    uint32_t uvDataSize       = fullPlaneSize / 2;
    uint32_t frameDataSize    = fullPlaneSize + uvDataSize;

    // Validate plane sizes
    {
        jlong yPlaneSize = env->GetDirectBufferCapacity(y_plane);
        if (yPlaneSize < fullPlaneSize) {
            SNN_LOGE("Y-channel buffer is too small: w=%d h=%d yPlaneSize=%d", width, height, yPlaneSize);
        }
        jlong uPlaneSize = env->GetDirectBufferCapacity(u_plane);
        if (uPlaneSize < uvDataSize - 1) {
            SNN_LOGE("U-channel buffer is too small: w=%d h=%d uPlaneSize=%d", w, h, uPlaneSize);
        }
        jlong vPlaneSize = env->GetDirectBufferCapacity(v_plane);
        if (vPlaneSize < uvDataSize - 1) {
            SNN_LOGE("V-channel buffer is too small: w=%d h=%d vPlaneSize=%d", w, h, vPlaneSize);
        }
    }

    // TODO: call NativeFrameProvider2::queueYUVFrame() to save one frame copy

    // Copy y plane data
    std::vector<uint8_t> frameData;
    frameData.reserve(frameDataSize);
    auto yDataEnd = std::copy(yPlaneData, yPlaneData + fullPlaneSize, frameData.begin());

    // Copy uv plane data
    ColorFormat format        = ColorFormat::NV12;
    const uint8_t* uPlaneData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(u_plane));
    const uint8_t* vPlaneData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(v_plane));
    const uint8_t* uvData     = uPlaneData;
    if (uPlaneData == vPlaneData + 1) {
        format = ColorFormat::NV21;
        uvData = vPlaneData;
    } else if (vPlaneData != uPlaneData + 1) {
        SNN_LOGE("Invalid data layout for NV12 format y=0x%x y_size=%d u=0x%x v=0x%x", yPlaneData, uPlaneData, vPlaneData);
    }
    std::copy(uvData, uvData + uvDataSize, yDataEnd);

    // Queue frame
    g_frameProvider->queueFrame(format, size_t(w), size_t(h), frameData.data(), frameDataSize, timestamp);
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_queueMetaData(JNIEnv*, jclass, jlong timestamp, jboolean low_exposure) {
    // SNN_LOGI("queueMetaData(%llu,%d)", timestamp, low_exposure);
    NativeFrameProvider* frameProvider = dynamic_cast<NativeFrameProvider*>(g_frameProvider.get());
    frameProvider->queueMetadata(timestamp, low_exposure);
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_startRecording(JNIEnv* jniEnv, jclass, jobject surface) {
    ANativeWindow* window = ANativeWindow_fromSurface(jniEnv, surface);
    g_mainLoop->startRecording((intptr_t) window);
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_stopRecording(JNIEnv*, jclass) { g_mainLoop->stopRecording(); }

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_updateSavePath(JNIEnv*, jclass, jstring savePath) {
    (void) savePath;
    // TODO: implement updateSavePath()
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_renameRecording(JNIEnv*, jclass, jstring newName) {
    (void) newName;
    // TODO: implement renameRecording()
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_compareFrameExposure(JNIEnv* env, jclass, jint w, jint h, jobject image0,
                                                                                                  jobject image1) {
    // Retrieve the array pointers of the byte buffers.
    jbyte* image0Address = (jbyte*) env->GetDirectBufferAddress(image0);
    jbyte* image1Address = (jbyte*) env->GetDirectBufferAddress(image1);

    // Wrap matrices around the byte buffers.
    cv::Mat m0(h, w, CV_8UC1, image0Address);
    cv::Mat m1(h, w, CV_8UC1, image1Address);

    auto s0 = cv::sum(m0)[0];
    auto s1 = cv::sum(m1)[0];

    return s0 < s1 ? -1 : s0 > s1 ? 1 : 0;
}

// -----------------------------------------------------------------------------
//

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18test(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    // load input image
    auto ip = new snn::InferenceProcessor();

    auto modelFileName = "resnet18_cifar10_0223.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {32, 32, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::TEXTURES, true, false, false});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;
        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();
        SNN_LOGI("%s:%d\n", __FUNCTION__, __LINE__);
        inputTexs[0].loadFromFile(formatString("%s/images/cifar_test.png", APP_DIR).c_str());
        std::vector<float> means {0, 0, 0, 0};
        std::vector<float> norms {1, 1, 1, 1};
        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
        inputTexs[0].convertFormat(snn::ColorFormat::RGBA16F, means, norms);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].upload();

        std::vector<float> resizeMeans {127.5, 127.5, 127.5, 127.5};
        std::vector<float> resizeNorms {1 / 127.5, 1 / 127.5, 1 / 127.5, 1 / 127.5};
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        inputTexs[0].resize(1, 1, resizeMeans, resizeNorms);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].readTexture(0);

        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__, __LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
               (int) inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
#ifdef __ANDROID__
        outputTexs[0].texture(0)->allocate2DArray(snn::ColorFormat::RGBA16F, 1, 1, 256, 1024, 1);
#endif
        ip->process(outputTexs);
        rc.swapBuffers();
    }

    ip->finalize();

    // test done successfully.
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spacialdenoisetest(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    auto modelFilename = "spatialDenoiser_0416.json";

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {1920, 1080, 1, 1}});

    auto ip = new snn::InferenceProcessor();

    // create input texture
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();
    ip->initialize({modelFilename, inputList, &rc, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, false});

    int loopcount = 1;
    for (int i = 0; i < loopcount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;
        auto input = snn::ManagedRawImage::loadFromAsset("images/bright_night_view_street_1080x1920.jpg");
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        auto input32f = snn::toR32f(input, -1.0, 1.0);
        gl::TextureObject scaleTex;
        scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 4, 1);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        scaleTex.setPixels(0, 0, 0, 1080, 1920, 0, input32f.data());
        scaleTex.detach();
        printf("%s:%d tex:%d, %d\n", __FUNCTION__, __LINE__, scaleTex.target(), scaleTex.id());
        inputTexs.allocate(1);
        inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

        ip->preProcess(inputTexs);
        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
#ifdef __ANDROID__
        outputTexs[0].texture(0)->allocate2D(snn::ColorFormat::RGBA32F, 1920, 1080);
#endif
        ip->process(outputTexs);
        // rc.swapBuffers();
    }

    ip->finalize();

    glFinish();
    SNN_LOGI("%s:%d\n", __FUNCTION__, __LINE__);
    // test done successfully.
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xtest(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    auto modelFilename = "ESPCN_2X.json";

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {1080, 1920, 1, 1}});

    auto ip = new snn::InferenceProcessor();

    // create input texture
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();
    ip->initialize({modelFilename, inputList, &rc, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, true});

    int loopcount = 1;
    for (int i = 0; i < loopcount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;
        auto input = snn::ManagedRawImage::loadFromAsset("images/bright_night_view_street_1080x1920.jpg");
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        auto input32f = snn::toRgba32f(input, -1.0, 1.0);
        SNN_LOGI("Size of the input image is : %d", input32f.size());
        gl::TextureObject scaleTex;
        scaleTex.allocate2D(snn::ColorFormat::RGBA32F, input32f.width(), input32f.height(), 4, 1);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        scaleTex.setPixels(0, 0, 0, 1080, 1920, 0, input32f.data());
        scaleTex.detach();
        printf("%s:%d tex:%d, %d\n", __FUNCTION__, __LINE__, scaleTex.target(), scaleTex.id());
        inputTexs.allocate(1);
        inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

        ip->preProcess(inputTexs);
        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
#ifdef __ANDROID__
        outputTexs[0].texture(0)->allocate2D(snn::ColorFormat::RGBA32F, 1920, 1080, 4, 1);
#endif
        ip->process(outputTexs);
        // rc.swapBuffers();
    }

    ip->finalize();

    glFinish();
    SNN_LOGI("%s:%d\n", __FUNCTION__, __LINE__);
    // test done successfully.
    return 0;
}

// extern "C"
// JNIEXPORT int JNICALL
// Java_com_oppo_seattle_snndemo_NativeTests_espcn2xtestmi(JNIEnv * env, jclass, jobject java_am) {
//    // remember the pointer to the asset manager
//    g_assetManager = AAssetManager_fromJava(env, java_am);
//    SNN_CHK(g_assetManager);
//
//    auto modelFilename = "ESPCN_2X.json";
//
//    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
//    rc.makeCurrent();
//
//    std::shared_ptr<MixedInferenceCore> ic2_;
//
//    if(!ic2_) {
//        auto dp = snn::dp::loadFromJsonModel(modelFilename);
//        dp::ShaderGenOptions options = {};
//        options.desiredInput.width = 1920;
//        options.desiredInput.height = 1080;
//        options.desiredInput.depth = 1;
//        options.desiredInput.format = ColorFormat::R32F;
//        options.compute = false;
//        options.desiredOutputFormat = ColorFormat::RGBA32F;
//        options.preferrHalfPrecision = true;
//
//        MixedInferenceCore::CreationParameters cp;
//        (InferenceGraph&&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);
//        cp.dumpOutputs = false;
//        ic2_ = MixedInferenceCore::create(cp);
//    }
//
//    auto input = snn::ManagedRawImage::loadFromAsset("images/bright_night_view_street_1080x1920.jpg");
//
//    auto input32f = snn::toR32f(input, -1.0, 1.0);
//    gl::TextureObject scaleTex;
//    scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 1, 1);
//
//    scaleTex.setPixels(
//            0,
//            0, 0,
//            1920, 1080,
//            0,
//            input32f.data()
//    );
//    //scaleTex.detach();
//
//    const gl::TextureObject* inputs[] = { &scaleTex };
//    auto outVec = std::vector<std::vector<std::vector<float>>>();
//    auto inVec = std::vector<std::vector<std::vector<float>>>();
//    gl::TextureObject outputTexture;
//    outputTexture.allocate2D(ColorFormat::RGBA32F, 1920, 1080, 1, 1);
//
//    int loopcount = 70;
//    for (int i = 0; i < loopcount; i++) {
//        ic2_->run({inputs, &outputTexture, 1, inVec, outVec});
//        rc.swapBuffers();
//    }
//    // test done successfully.
//    glFinish();
//    SNN_LOGI("%s:%d\n", __FUNCTION__,__LINE__);
//
//    return 0;
//}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unettest(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    // load input image
    auto ip = new snn::InferenceProcessor();

    auto modelFileName = "unet.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {256, 256, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::TEXTURES, true, false, false});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;
        inputTexs.allocate(1);

        // inputTexs[0].loadFromFile("images/test_image_unet_gray.png");
        // inputTexs[0].convertFormat(snn::ColorFormat::R32F);
        // printf("%s:%d\n", __FUNCTION__,__LINE__);
        // // inputTexs[0].printOutWH();
        // // inputTexs[0].upload();

        auto input = snn::ManagedRawImage::loadFromAsset("images/test_image_unet_gray.png");
        // for (std::size_t i = 0; i < input.size(); i++) {
        //     SNN_LOGI("%d", *(input.data() + i));
        // }
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        auto input32f = snn::toR32f(input, -1.0, 1.0);
        // for (std::size_t i = 0; i < input32f.size(); i+=4) {
        //     float f;
        //     uint8_t buffer[4] = {*(input32f.data() + i), *(input32f.data() + i+1), *(input32f.data() + i+2), *(input32f.data() + i+3)};
        //     std::memcpy(&f, buffer, sizeof(f));
        //     SNN_LOGI("%f", f);
        // }
        gl::TextureObject scaleTex;
        scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 1, 1);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        scaleTex.setPixels(0, 0, 0, 256, 256, 0, input32f.data());
        scaleTex.detach();
        printf("%s:%d tex:%d, %d\n", __FUNCTION__, __LINE__, scaleTex.target(), scaleTex.id());
        inputTexs.allocate(1);
        inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__, __LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
               (int) inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);

#ifdef __ANDROID__
        outputTexs[0].texture(0)->allocate2DArray(snn::ColorFormat::RGBA32F, 1, 1, 256, 1024, 1);
#endif
        ip->process(outputTexs);
    }

    ip->finalize();

    // test done successfully.
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2test(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    // load input image
    auto ip = new snn::InferenceProcessor();

    auto modelFileName = "mobilenetv2_keras.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {224, 224, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::TEXTURES, true, false, false});

    int loopCount = 2;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        auto input = snn::ManagedRawImage::loadFromAsset("images/arduino.png");
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        auto input32f = snn::toR32f(input, 0.0, 1.0);
        gl::TextureObject scaleTex;
        scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 1, 1);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        scaleTex.setPixels(0, 0, 0, 224, 224, 0, input32f.data());
        scaleTex.detach();
        printf("%s:%d tex:%d, %d\n", __FUNCTION__, __LINE__, scaleTex.target(), scaleTex.id());
        inputTexs.allocate(1);
        inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

        //        inputTexs[0].loadFromFile(formatString("%s/images/ant.png",APP_DIR).c_str());
        //        std::vector<float> means{0.0, 0.0, 0.0, 0.0};
        //        std::vector<float> norms{1.0, 1.0, 1.0, 1.0};
        //        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        //        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
        //        inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);
        //        printf("%s:%d\n", __FUNCTION__,__LINE__);
        //        // inputTexs[0].printOutWH();
        //        // inputTexs[0].upload();
        //
        //        std::vector<float> resizeMeans{0.0, 0.0, 0.0, 0.0};
        //        std::vector<float> resizeNorms{1.0/255.0, 1.0/255.0, 1.0/255.0, 1.0/255.0};
        //        printf("%s:%d\n", __FUNCTION__,__LINE__);
        //        inputTexs[0].resize(1, 1, resizeMeans, resizeNorms);
        //        printf("%s:%d\n", __FUNCTION__,__LINE__);
        //        // inputTexs[0].printOutWH();
        //        // inputTexs[0].readTexture(0);
        //
        //        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__,__LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
        //               (int)inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
#ifdef __ANDROID__
        outputTexs[0].texture(0)->allocate2DArray(snn::ColorFormat::RGBA32F, 1, 1, 256, 1024, 1);
#endif
        ip->process(outputTexs);
    }

    ip->finalize();

    // test done successfully.
    return 0;
}

void my_null_deleter(snn::dp::GenericModelLayer* layer) {
    (void) layer;
    return;
}

ManagedRawImage cvMat2ManagedRawImage(cv::Mat& in, char* imageMemory) {
    ManagedRawImage result(ImageDesc(ColorFormat::RGBA32F, in.size[1], in.size[0], in.size[2]), imageMemory);

    for (int i = 0; i < in.size[0]; i++) {
        for (int j = 0; j < in.size[1]; j++) {
            for (int k = 0; k < in.size[2]; k++) {
                // std::cout << "M(" << i << ", " << j << ", " << k << "): " << in.at<float>(i,j,k) << ",";
                // std::cout  << std::setw(7) << in.at<float>(i,j,k) <<  ",";
                //*((float *)result.at(0, i, j, k)) = in.at<float>(i,j,k);
            }
            Rgba32f* dst = (Rgba32f*) result.at(0, j, i, 0);
            (*dst).red     = in.at<float>(i, j, 0);
            (*dst).green     = in.at<float>(i, j, 1);
            (*dst).blue     = in.at<float>(i, j, 2);
            (*dst).alpha     = in.at<float>(i, j, 3);

            // std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
        }
        // std::cout  << std::endl;
        // std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }

    return result;
}

//#include "testutil.h"
//#include "matutil.h"

// extern "C"
// JNIEXPORT int JNICALL
// Java_com_oppo_seattle_snndemo_NativeTests_conv2dtest(JNIEnv * env, jclass, jobject java_am) {
//    // remember the pointer to the asset manager
//    g_assetManager = AAssetManager_fromJava(env, java_am);
//    SNN_CHK(g_assetManager);
//
//    SRAND(7767517);
//
//    int width = 1920;
//    int height = 1080;
//    int inChannels = 64;
//    int outChannels = 32;
//    int kernel = 3;
//
//
//    int dilation = 1;
//    int stride = 1;
//    int pad = 0;
//    int bias = 1;
//    bool useOldShader = false;
//
//    ncnn::Mat padA = RandomMat(width, height, inChannels);
//    //SetValue(padA, 1.0f);
//
//    cv::Mat inputMat = NCNNMat2CVMat(padA);
//    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
//
//    weights[0] = RandomMat(outChannels * inChannels * kernel * kernel);
////    SetValue(weights[0], 0.2f);
//
////    for (size_t i = 0; i < weights[0].total(); i++)
////    {
////        weights[0][i] = 0.0003f * i;
////    }
//    if (bias) {
//        weights[1] = RandomMat(outChannels);
////        SetValue(weights[1], 1.0f);
//
////        for (size_t i = 0; i < weights[1].total(); i++)
////        {
////            weights[1][i] = 0.0003f * i;
////        }
//    }
//
//    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChannels * outChannels);
//    std::vector<float> inputBias = std::vector<float>(outChannels, 0.0f);
//
//    for (size_t p=0; p<inputWeights.size(); p++)
//    {
//        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
//        memcpy((uchar*)inputWeights[p].data, (uchar*)weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
//        //std::cout << "M = " << std::endl << " "  << inputWeights[p] << std::endl << std::endl;
//    }
//
//    if (bias) {
//        const float* ptr = weights[1].channel(0);
//        for (size_t p=0; p<inputBias.size(); p++)
//        {
//            inputBias[p] = ptr[p];
//        }
//    }
//
//    int ret = 0;
//
//    vector<double> doubleBias(inputBias.size(), 0);
//    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double)x;});
//
//    printf("%%%%%%%% %s:%d: width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useOldShader:%d \n",
//           __FUNCTION__,__LINE__, width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useOldShader);
//
//    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
//    rc.makeCurrent();
//
//    int paddingSize = (int) std::floor(kernel / 2);
//    int outWidth = (width - kernel + 2 * paddingSize) / stride + 1;
//    int outHeight = (height - kernel + 2 * paddingSize) / stride + 1;
//    char *imageMemory = (char *)malloc(inChannels * outWidth * outHeight * 4 * 4);
//
//    // Create single layer from Layer class
//    snn::dp::InputLayerDesc inputDesc;
//    inputDesc.inputHeight = width;
//    inputDesc.inputWidth = height;
//    inputDesc.inputChannels = inChannels;
//    auto inputLayer = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
//    inputLayer->prevLayers.clear();
//
//    snn::dp::Conv2DDesc desc;
//    desc.isRange01 = 0;
//    desc.numOutputPlanes = outChannels;
//    desc.numInputPlanes = inChannels;
//    desc.weights = inputWeights;
//    desc.biases = doubleBias;
//    desc.activation = "relu";
//    desc.kernelSize = kernel;
//    desc.stride = stride;
//    desc.useBatchNormalization = false;
//    desc.useMultiInputs = false;
//    desc.padding = "same";
//    desc.paddingT = "same";
//    desc.paddingB = "same";
//    desc.paddingL = "same";
//    desc.paddingR = "same";
//
//    auto layer = std::shared_ptr<snn::dp::Conv2DLayer>(new snn::dp::Conv2DLayer(std::move(desc)), &my_null_deleter);
//
//    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
//    layer->prevLayers.push_back(inputLayer);
//    layer->nextLayers.clear();
//    layer->name = "resnet18_cifar10_0223.json layer [01] Conv2D";
//
//    inputLayer->nextLayers.push_back(layer);
//    inputLayer->prevLayers.clear();
//    layers.emplace_back(inputLayer);
//    layers.emplace_back(layer);
//
//    gl::TextureObject inputTexture;
//    inputTexture.allocate2DArray(ColorFormat::RGBA32F, width, height, (inChannels+3)/4);
//
//    SNN_LOGI("INPUT TEXTURE: %u, %u", inputTexture.id(), inputTexture.target());
//    {
//        SNN_LOGI("######## %s:%d, %d, %d, %d\n", __FUNCTION__,__LINE__, inputMat.size[0], inputMat.size[1], inputMat.size[2]);
//        auto input32f = cvMat2ManagedRawImage(inputMat, imageMemory);
//
//        inputTexture.setPixels(0, 0, 0, width, height, 0, input32f.data());
//    }
//
//    auto input = snn::ManagedRawImage::loadFromAsset("images/bright_night_view_street_1080x1920.jpg");
//
//    auto input32f = snn::toR32f(input, -1.0, 1.0);
//    gl::TextureObject scaleTex;
//    scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 1, 1);
//
//    scaleTex.setPixels(
//            0,
//            0, 0,
//            1920, 1080,
//            0,
//            input32f.data()
//    );
//    //scaleTex.detach();
//
//    SNN_LOGI("Model read from file successfully");
//
//    // setup options
//    snn::dp::ShaderGenOptions sgo = {};
//    sgo.desiredInput.width = width;
//    sgo.desiredInput.height = height;
//    sgo.desiredInput.depth = (inChannels+3)/4;
//    sgo.desiredInput.format = ColorFormat::RGBA32F; //input.format();
//    sgo.desiredOutputFormat = ColorFormat::RGBA32F;
//    sgo.preferrHalfPrecision = true;
//
//    // generate graph
//    snn::MixedInferenceCore::CreationParameters graph;
//    graph.dumpOutputs = false;
//
//    // dp.erase(dp.begin()+2, dp.end());
//    // dp[1]->nextLayers.clear();
//    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());
//
//    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
//    (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
//    SNN_LOGI("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, graph.layers.size());
//
//    if (graph.layers.empty()) return ret;
//
//    // create ic2
//    auto ic2 = MixedInferenceCore::create(graph);
//
//    // generate output texture
//    auto outdesc = graph.layers.back()->output;
//    (void) outdesc;
//    // SNN_CHK(1 == outdesc.depth);
//    gl::TextureObject outTexture;
//    outTexture.allocate2D(ColorFormat::RGBA32F, outWidth, outHeight);
//    const gl::TextureObject* inputs[] = { &inputTexture };
//    //const gl::TextureObject* inputs[] = { &scaleTex };
//
//    auto outVec = std::vector<std::vector<std::vector<float>>>();
//    auto inVec = std::vector<std::vector<std::vector<float>>>();
//
//    int loopCount = 70;
//#define PROFILING
//#ifdef PROFILING
//    unordered_map<string, vector<double>> timeMap;
//    SNN_LOGI("before loops");
//#endif
//    for (int i = 0; i < loopCount; i++) {
//        ic2->run({inputs, &outTexture, 1, inVec, outVec});
//#ifdef PROFILING
//        ic2->writeTimeStat(timeMap);
//#endif
//        rc.swapBuffers();
//    }
//#ifdef PROFILING
//    int skip_loop = 5;
//    for(auto arr : timeMap) {
//        std::stringstream ss;
//        ss << "\n=============================  Final Time Stats  ==============================\n";
//        ss << arr.first.c_str() << std::endl;
//
//        auto v = arr.second;
//        v.erase(std::remove(begin(v), end(v), 0), end(v));
//        v.erase(v.begin(), v.begin() + skip_loop);
//        auto mean = accumulate(v.begin(), v.end(), 0.0) / v.size();
//        double accum = 0;
//        for (auto n : v) {
//            accum += (n - mean) * (n - mean);
//        }
//        double stdev = sqrt(accum / (v.size()-1));
//        ss << "mean: " << mean << std::endl;
//        ss << "stdev: " << stdev << std::endl;
//        ss << "=================================================================================\n";
//        SNN_LOGI(ss.str().c_str());
//    }
//#endif
//
//    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
//
//    glFinish();
//    SNN_LOGI("%s:%d\n", __FUNCTION__,__LINE__);
//
//    free(imageMemory);
//    return ret;
//
//}
