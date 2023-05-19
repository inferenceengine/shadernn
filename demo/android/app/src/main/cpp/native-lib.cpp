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
#include "snn/contextFactory.h"
#include "inferenceengine.h"
#include "demoutils.h"
#include "nativeFrameProvider.h"
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <sys/system_properties.h>
#include <opencv2/opencv.hpp>
#ifdef SUPPORT_GL
    #include "glAppContext.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "vulkan/vulkanApp.h"
    #include "vulkan/androidWindow.h"
#endif
#include <atomic>
#include <condition_variable>
#include <mutex>

static const char* const OFF_STR = "0";
using namespace snn;

static std::unique_ptr<MainProcessingLoop> g_mainLoop;
static std::unique_ptr<NativeFrameProvider> g_frameProvider;
#ifdef SUPPORT_VULKAN
    static std::unique_ptr<VulkanApp> g_VulkanApp;
#endif

static std::atomic<bool> g_cameraReady;
static std::condition_variable g_cv;
static std::mutex g_mtx;

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

std::pair<InferenceEngine::AlgorithmConfig, ModelType> toNativeAlgorithmConfig(JNIEnv* env, jobject jalgo) {
    jclass algoCfgClass = env->FindClass("com/oppo/seattle/snndemo/AlgorithmConfig");
    bool dumpOutputs    = false;
    InferenceEngine::AlgorithmConfig algo;
    ModelType modelType = ModelType::OTHER;
    { // denoise
        jmethodID isDenoiseSPATIALDENOISER = env->GetMethodID(algoCfgClass, "isDenoiseSPATIALDENOISER", "()Z");
        jmethodID isDenoiseComputeShader   = env->GetMethodID(algoCfgClass, "isDenoiseComputeShader", "()Z");

        InferenceEngine::AlgorithmConfig::Denoisers::Denoiser denoiserShader = InferenceEngine::AlgorithmConfig::Denoisers::Denoiser::FRAGMENTSHADER;
        if (env->CallBooleanMethod(jalgo, isDenoiseComputeShader)) {
            denoiserShader = InferenceEngine::AlgorithmConfig::Denoisers::Denoiser::COMPUTESHADER;
        }

        if (env->CallBooleanMethod(jalgo, isDenoiseSPATIALDENOISER)) {
            algo.denoisers = InferenceEngine::AlgorithmConfig::Denoisers {denoiserShader,
                                                                          InferenceEngine::AlgorithmConfig::Denoisers::DenoiserAlgorithm ::SPATIALDENOIRSER};
        }
    }

    {
        // classifiers
        jmethodID isClassifierResnet18       = env->GetMethodID(algoCfgClass, "isClassifierResnet18", "()Z");
        jmethodID isClassifierMobilenetv2    = env->GetMethodID(algoCfgClass, "isClassifierMobilenetv2", "()Z");
        jmethodID isClassifierComputeShader  = env->GetMethodID(algoCfgClass, "isClassifierComputeShader", "()Z");

        if (env->CallBooleanMethod(jalgo, isClassifierResnet18) || env->CallBooleanMethod(jalgo, isClassifierMobilenetv2)) {
            modelType = ModelType::CLASSIFICATION;
        }

        InferenceEngine::AlgorithmConfig::Classifiers::Classifier classifierShader = InferenceEngine::AlgorithmConfig::Classifiers::Classifier::FRAGMENTSHADER;
        if (env->CallBooleanMethod(jalgo, isClassifierComputeShader)) {
            classifierShader = InferenceEngine::AlgorithmConfig::Classifiers::Classifier::COMPUTESHADER;
        }

        if (env->CallBooleanMethod(jalgo, isClassifierResnet18)) {
            // Connect this part with snn.h's algo struct
            SNN_LOGV("resnet18 compute shader");
            algo.classifiers =
                    InferenceEngine::AlgorithmConfig::Classifiers {dumpOutputs, classifierShader,
                                                                   InferenceEngine::AlgorithmConfig::Classifiers::ClassifierAlgorithm::RESNET18};
        }  else if (env->CallBooleanMethod(jalgo, isClassifierMobilenetv2)) {
            // Connect this part with snn.h's algo struct
            SNN_LOGD("Mobilenetv2 compute shader");
            algo.classifiers =
                    InferenceEngine::AlgorithmConfig::Classifiers {dumpOutputs, classifierShader,
                                                                   InferenceEngine::AlgorithmConfig::Classifiers::ClassifierAlgorithm::MOBILENETV2};
        }
    }
    {
        // detections
        jmethodID isDetectionYolov3         = env->GetMethodID(algoCfgClass, "isDetectionYolov3", "()Z");
        jmethodID isDetectionComputeShader  = env->GetMethodID(algoCfgClass, "isDetectionComputeShader", "()Z");

        if (env->CallBooleanMethod(jalgo, isDetectionYolov3)) {
            modelType = ModelType::DETECTION;
        }

        InferenceEngine::AlgorithmConfig::Detections::Detection detectionShader = InferenceEngine::AlgorithmConfig::Detections::Detection::FRAGMENTSHADER;
        if (env->CallBooleanMethod(jalgo, isDetectionComputeShader)) {
            detectionShader = InferenceEngine::AlgorithmConfig::Detections::Detection::COMPUTESHADER;
        }

        if (env->CallBooleanMethod(jalgo, isDetectionYolov3)) {
            // Connect this part with snn.h's algo struct
            SNN_LOGV("yolov3 compute shader");
            algo.detections =
                    InferenceEngine::AlgorithmConfig::Detections {detectionShader,
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

    jmethodID isFP16 = env->GetMethodID(algoCfgClass, "isFP16", "()Z");
    if (env->CallBooleanMethod(jalgo, isFP16)) {
        algo.precision = Precision::FP16;
    } else {
        algo.precision = Precision::FP32;
    }

    std::pair<InferenceEngine::AlgorithmConfig, ModelType> retVal = {algo, modelType};

    return retVal;
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_init(JNIEnv* env, jobject, jobject java_am, jstring /*internalStorageDir*/,
                                                                                   jstring /*externalStorageDir*/) {
    // remember the pointer to the asset manager
    SNN_CHK(env);
    SNN_CHK(java_am);
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    static const char* currentEvSetting = nullptr;
    const char* evSetting               = OFF_STR;
    if (evSetting != currentEvSetting) {
        __system_property_set("vendor.oppo.stream.PVEVenable", evSetting);
        currentEvSetting = evSetting;
    }

    g_frameProvider.reset(new NativeFrameProvider(4));
    // delay main loop creation until we know the size of the frame.
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

static uint32_t frameW = 0, frameH = 0;

namespace snn
{

extern void renderOutput(MainProcessingLoop::RenderParameters& rp) {
    SNN_ASSERT(0 != frameW && 0 != frameH);
    if (!g_mainLoop) {
        g_mainLoop.reset(new MainProcessingLoop({frameW, frameH, g_frameProvider.get()}));
        if (vizW > 0) {
            g_mainLoop->resize(vizW, vizH);
        }
    }

    g_mainLoop->render(rp);
}

}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL
Java_com_oppo_seattle_snndemo_NativeLibrary_drawGL(JNIEnv* env, jclass, jobject jAlgo) {
    (void) env;
    (void) jAlgo;
#ifdef SUPPORT_GL
    if (!g_cameraReady) {
        std::unique_lock lk(g_mtx);
        g_cv.wait(lk, []{return (bool)g_cameraReady;});
    }
    SNN_ASSERT(0 != frameW && 0 != frameH);

    auto algoConfig = toNativeAlgorithmConfig(env, jAlgo);
    MainProcessingLoop::RenderParameters rp;
    rp.playing   = true;
    rp.algorithm = algoConfig.first;
    rp.modelType = algoConfig.second;

    snn::renderOutput(rp);

    if (rp.modelType == ModelType::CLASSIFICATION && jAlgo != nullptr) {
        jclass cls = env->GetObjectClass(jAlgo);
        env->SetIntField(jAlgo, env->GetFieldID(cls, "classifierIndex", "I"), rp.modelOutput.classifierOutput);
    }
#endif
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL
Java_com_oppo_seattle_snndemo_NativeLibrary_drawVulkan(JNIEnv* env, jclass, jobject jAlgo) {
    (void) env;
    (void) jAlgo;
#ifdef SUPPORT_VULKAN
    if (!g_cameraReady) {
        std::unique_lock lk(g_mtx);
        g_cv.wait(lk, []{return (bool)g_cameraReady;});
    }

    SNN_ASSERT(0 != frameW && 0 != frameH);

    auto algoConfig = toNativeAlgorithmConfig(env, jAlgo);
    MainProcessingLoop::RenderParameters rp;
    rp.playing   = true;
    rp.algorithm = algoConfig.first;
    rp.modelType = algoConfig.second;

    snn::renderOutput(rp);

    g_VulkanApp->update();

    if (rp.modelType == ModelType::CLASSIFICATION && jAlgo != nullptr) {
        jclass cls = env->GetObjectClass(jAlgo);
        SNN_LOGD("Classifier output = %d", rp.modelOutput.classifierOutput);
        env->SetIntField(jAlgo, env->GetFieldID(cls, "classifierIndex", "I"), rp.modelOutput.classifierOutput);
    }
#endif
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_destroy(JNIEnv*, jobject) {
    g_mainLoop.reset();
    g_frameProvider.reset();
#ifdef SUPPORT_VULKAN
    g_VulkanApp.reset();
#endif
}

// -----------------------------------------------------------------------------
//

extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_queueFrame(JNIEnv* env, jclass, jint w, jint h, jint /*rotation_degrees*/,
                                                                                         jlong timestamp, jobject y_plane, jobject u_plane, jobject v_plane) {
    if (!g_cameraReady) {
        frameW = (uint32_t) w;
        frameH = (uint32_t) h;
        g_cameraReady = true;
        g_cv.notify_all();
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
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_initGL(JNIEnv*, jclass) {
#ifdef SUPPORT_GL
    GlAppContext::createContext();
#endif
}

// -----------------------------------------------------------------------------
//
extern "C" JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeLibrary_initVulkan(JNIEnv* jniEnv, jclass, jobject surface) {
    (void) jniEnv;
    (void) surface;
#ifdef SUPPORT_VULKAN
    ANativeWindow *nativeWindow = ANativeWindow_fromSurface(jniEnv, surface);
    vkb::AndroidWindow window(nativeWindow);

    g_VulkanApp.reset(new VulkanApp);
    g_VulkanApp->prepare(window);
#endif
}
