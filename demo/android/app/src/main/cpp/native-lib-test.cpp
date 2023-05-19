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
#include "../../../../../common/demoutils.h"
#include "nativeFrameProvider.h"
#include <jni.h>
#include "inferenceProcessor.h"
#include "snn/snn.h"
#include "snn/colorUtils.h"
#ifdef SUPPORT_GL
    #include "imageTextureGL.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "volk.h"
    #include "vulkanContext.h"
    #include "imageTextureVulkan.h"
    #include "vulkanImageResizeOp.h"
#endif
#include "cpu.h"
#include "net.h"
#include "layer/selu.h"
#include "layer/relu.h"
#include "layer/tanh.h"
#include "layer/sigmoid.h"
#include "layer/swish.h"
#include "layer/clip.h"
#include "modelInference.h"
#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

#include <string>
#include <cstring>
#include <vector>
#include <array>
#include <map>
#include <memory>
#include <unistd.h>
#include <pthread.h>

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <sys/system_properties.h>
#include <opencv2/opencv.hpp>

using namespace snn;

extern AAssetManager* g_assetManager; // defined in utils.cpp

#include <android/log.h>
static int pfd[2];
static pthread_t thr;
static const char *tag = "myapp";
static void *thread_func(void*)
{
    ssize_t rdsz;
    char buf[128];
    while ((rdsz = read(pfd[0], buf, sizeof buf - 1)) > 0) {
        if (buf[rdsz - 1] == '\n') {
            --rdsz;
        }
        buf[rdsz] = 0;  /* add null-terminator */
        __android_log_write(ANDROID_LOG_DEBUG, tag, buf);
    }
    return 0;
}
int start_logger(const char *app_name)
{
    tag = app_name;

    /* make stdout line-buffered and stderr unbuffered */
    setvbuf(stdout, 0, _IOLBF, 0);
    setvbuf(stderr, 0, _IONBF, 0);

    /* create the pipe and redirect stdout and stderr */
    pipe(pfd);
    dup2(pfd[1], 1);
    dup2(pfd[1], 2);

    /* spawn the logging thread */
    if (pthread_create(&thr, 0, thread_func, 0) == -1) {
        return -1;
    }
    pthread_detach(thr);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_imagetexturetest(JNIEnv* env, jclass, jobject java_am) {
    (void) env;
    (void) java_am;
#ifdef SUPPORT_GL
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);

    start_logger("com.innopeaktech.seattle.snndemo");

    snn::createGlContext();

    int width = 8;
    int height = 8;
    int inChannels = 6;

    int size[3] = {height, width, inChannels};
    cv::Mat inputMat(3, size, CV_32FC1, cv::Scalar(1.0f));

    std::array<uint32_t, 4> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    std::shared_ptr<snn::ImageTexture> img;
    img.reset(new snn::ImageTextureGL(dims, ColorFormat::RGBA32F, dest));
    img->upload();
    img->prettyPrint();
#endif
    return 0;
}

#ifdef SUPPORT_VULKAN
static const std::array<float, 4> DUMMY_MEANS = {0.0f, 0.0f, 0.0f, 0.0f};
static const std::array<float, 4> DUMMY_NORMS = {1.0f, 1.0f, 1.0f, 1.0f};
#endif

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_imageTextureVulkanResizeTest(JNIEnv*, jclass) {
#ifdef SUPPORT_VULKAN
    const uint32_t w = 2;
    const uint32_t h = 2;
    const uint32_t w1 = 4;
    const uint32_t h1 = 4;
    const uint32_t c = 4;
    const float scale = 0.5f;
    const snn::ColorFormat cf = snn::ColorFormat::RGBA8;
    std::vector<float> values {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> valuesAllCh(w * h * c, 0.0f); //
    for (size_t i = 0, j = 0; i < values.size() && i < w * h; ++i, j += c) {
        for (size_t q = 0; q < c; ++q) {
            valuesAllCh[j + q] = values[i];
        }
    }
    std::vector<uint8_t> buf = snn::convertColorBuffer(cf, valuesAllCh.data(), valuesAllCh.size());
    std::vector<uint8_t> emptyBuf(buf.size(), 0U);

    snn::GpuContext* context = snn::createDefaultVulkanContext();
    uvkc::benchmark::VulkanContext* uvkcContext = VulkanGpuContext::cast(context)->getUvkcContext();

    snn::ImageTextureVulkan textureIn(context, std::array<uint32_t, 4>{w, h, 1, 1}, cf, buf.data());
    snn::ImageTextureVulkan textureOut(context, std::array<uint32_t, 4>{w, h, 1, 1}, cf, emptyBuf.data());
    uvkc::vulkan::Device* device = uvkcContext->devices[0].get();

    std::unique_ptr<VulkanImageResizeOp> vulkanImageResizeOp = std::make_unique<VulkanImageResizeOp>();
    vulkanImageResizeOp->init(device, true);
    vulkanImageResizeOp->updateParams({w, h, 1}, DUMMY_MEANS, DUMMY_NORMS);

    textureIn.upload();
    textureOut.upload();

    textureIn.resize(scale, scale, DUMMY_MEANS, DUMMY_NORMS, true, snn::ColorFormat::RGBA32F);

    textureIn.download();
    {
        SNN_ASSERT(textureIn.getDims()[0] == w1);
        SNN_ASSERT(textureIn.getDims()[1] == h1);
        snn::TypedImage<snn::Rgba32f> img = textureIn.getRawImage();
        SNN_ASSERT(img.width() == w1);
        SNN_ASSERT(img.height() == h1);
        SNN_ASSERT(img.depth() == 1);
        SNN_ASSERT(img.planes() == 1);
        SNN_ASSERT(img.channels() == 4);

        float row[w1];
        SNN_LOGI("----w: %d, h: %d, c: %d----", w1, h1, c);
        for (uint32_t q = 0; q < c; q++) {
            for (uint32_t y = 0; y < h1; y++) {
                for (uint32_t x = 0; x < w1; x++) {
                    snn::Rgba32f v = img.at(0, x, y, 0);
                    row[x] = v.f32[q] * 255.0f;
                }
                SNN_LOGI("%f %f %f %f", row[0], row[1], row[2], row[3]);
            }
            SNN_LOGI("----------%d--------------", q);
        }
    }

    // Resize back
    vulkanImageResizeOp->run(textureIn.vkImage().get(), textureOut.vkImage().get());

    textureOut.download();
    {
        SNN_ASSERT(textureOut.getDims()[0] == w);
        SNN_ASSERT(textureOut.getDims()[1] == h);
        snn::TypedImage<snn::Rgba8> img = textureOut.getRawImage();
        SNN_ASSERT(img.width() == w);
        SNN_ASSERT(img.height() == h);
        SNN_ASSERT(img.depth() == 1);
        SNN_ASSERT(img.planes() == 1);
        SNN_ASSERT(img.channels() == 4);

        uint8_t row[w];
        SNN_LOGI("----w: %d, h: %d, c: %d----", w, h, c);
        for (uint32_t q = 0; q < c; q++) {
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    snn::Rgba8 v = img.at(0, x, y, 0);
                    row[x] = v.u8[q];
                }
                SNN_LOGI("%d %d", row[0], row[1]);
            }
            SNN_LOGI("----------%d--------------", q);
        }
    }
#endif
    return 0;
}

#ifdef PROFILING
    static const bool useFineTuned = false;
#else
    static const bool useFineTuned = true;
#endif

static uint32_t innerLoops = 1;
static bool dumpResults = true;

// NCNN-based tests
extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18FS32Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18FS32Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18FS16Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18FS16Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18CS32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18CS16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18VK32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_resnet18VK16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runResnet18(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyFS32Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyFS32Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyFS16Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyFS16Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyCS32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyCS16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyVK32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_yolov3tinyVK16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runYolov3Tiny(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseFS32Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseFS32Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseFS16Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseFS16Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseCS32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseCS16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseVK32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_spatialdenoiseVK16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runSpatialDenoiser(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xFS32Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::TEXTURES, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xFS32Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::TEXTURES, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xFS16Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::TEXTURES, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xFS16Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::TEXTURES, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xCS32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::TEXTURES, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xCS16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::TEXTURES, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xVK32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::TEXTURES, false, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_espcn2xVK16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runESPCN(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::TEXTURES, true, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetFS32Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetFS32Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetFS16Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetFS16Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetCS32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetCS16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetVK32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_unetVK16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runUNet(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2FS32Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2FS32Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2FS16Single(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, false, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2FS16Double(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2CS32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2CS16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2VK32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_mobilenetv2VK16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runMobilenetV2(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_styletransferCS32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runStyleTransfer(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_styletransferCS16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runStyleTransfer(dumpResults, true, snn::MRTMode::SINGLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, false, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_styletransferVK32(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runStyleTransfer(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, false, true, useFineTuned, innerLoops);
    return 0;
}

extern "C" JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_styletransferVK16(JNIEnv* env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    runStyleTransfer(dumpResults, false, snn::MRTMode::DOUBLE_PLANE, snn::WeightAccessMethod::CONSTANTS, true, true, useFineTuned, innerLoops);
    return 0;
}

extern "C"
JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_conv2dCS32(JNIEnv * env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    int w = 1080;
    int h = 1920;
    int inChs = 3;
    int outChs = 16;
    int kernel = 5;
    std::string activation = "relu";
    bool bias = true;
    int dilation = 1;
    int stride = 1;
    int pad = 0;
    bool useBN = false;

    ncnn::Mat matA = RandomMat(w, h, inChs);
    ShaderUnitTest test(snn::GpuBackendType::GL);
    cv::Mat inputMat = NCNNMat2CVMat(matA);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outChs * inChs * kernel * kernel);
    if (bias) {
        weights[1] = RandomMat(outChs);
    }

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    std::vector<float> inputBias      = std::vector<float>(outChs, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar*) inputWeights[p].data, (uchar*) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }

    if (bias) {
        const float* ptr = weights[1].channel(0);
        for (size_t p = 0; p < inputBias.size(); p++) {
            inputBias[p] = ptr[p];
        }
    }

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    int channels = outChs;
    std::vector<ncnn::Mat> bnweights(4);
    bnweights[0] = RandomMat(channels);
    bnweights[1] = RandomMat(channels);
    bnweights[2] = RandomMat(channels);

    bnweights[3] = RandomMat(channels);

    ncnnToVec(bnweights[0], bnGamma);
    ncnnToVec(bnweights[1], bnMean);
    ncnnToVec(bnweights[2], bnVar);
    ncnnToVec(bnweights[3], bnBeta);

    std::map<std::string, std::vector<float>> batchNormalization;
    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, w, h, inChs, outChs, kernel, dilation, stride, pad, false,
        snn::MRTMode::SINGLE_PLANE, useBN, batchNormalization, false, false);
    (void) outFile;
    return 0;
}

extern "C"
JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_conv2dCS16(JNIEnv * env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    int w = 1080;
    int h = 1920;
    int inChs = 3;
    int outChs = 16;
    int kernel = 5;
    std::string activation = "relu";
    bool bias = true;
    int dilation = 1;
    int stride = 1;
    int pad = 0;
    bool useBN = false;

    ncnn::Mat matA = RandomMat(w, h, inChs);
    ShaderUnitTest test(snn::GpuBackendType::GL);
    cv::Mat inputMat = NCNNMat2CVMat(matA);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outChs * inChs * kernel * kernel);
    if (bias) {
        weights[1] = RandomMat(outChs);
    }

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    std::vector<float> inputBias      = std::vector<float>(outChs, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        std::memcpy((uchar*) inputWeights[p].data, (uchar*) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }

    if (bias) {
        const float* ptr = weights[1].channel(0);
        for (size_t p = 0; p < inputBias.size(); p++) {
            inputBias[p] = ptr[p];
        }
    }

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    int channels = outChs;
    std::vector<ncnn::Mat> bnweights(4);
    bnweights[0] = RandomMat(channels);
    bnweights[1] = RandomMat(channels);
    bnweights[2] = RandomMat(channels);
    bnweights[3] = RandomMat(channels);

    ncnnToVec(bnweights[0], bnGamma);
    ncnnToVec(bnweights[1], bnMean);
    ncnnToVec(bnweights[2], bnVar);
    ncnnToVec(bnweights[3], bnBeta);

    std::map<std::string, std::vector<float>> batchNormalization;
    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, w, h, inChs, outChs, kernel, dilation, stride, pad, false,
        snn::MRTMode::SINGLE_PLANE, useBN, batchNormalization, false, true);
    (void) outFile;
    return 0;
}

extern "C"
JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_conv2dVK32(JNIEnv * env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    int w = 1080;
    int h = 1920;
    int inChs = 3;
    int outChs = 16;
    int kernel = 5;
    std::string activation = "relu";
    bool bias = true;
    int dilation = 1;
    int stride = 1;
    int pad = 0;
    bool useBN = false;

    ncnn::Mat matA = RandomMat(w, h, inChs);
    ShaderUnitTest test(snn::GpuBackendType::VULKAN);
    cv::Mat inputMat = NCNNMat2CVMat(matA);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outChs * inChs * kernel * kernel);
    if (bias) {
        weights[1] = RandomMat(outChs);
    }

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    std::vector<float> inputBias      = std::vector<float>(outChs, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        std::memcpy((uchar*) inputWeights[p].data, (uchar*) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }

    if (bias) {
        const float* ptr = weights[1].channel(0);
        for (size_t p = 0; p < inputBias.size(); p++) {
            inputBias[p] = ptr[p];
        }
    }

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    int channels = outChs;
    std::vector<ncnn::Mat> bnweights(4);
    bnweights[0] = RandomMat(channels);
    bnweights[1] = RandomMat(channels);
    bnweights[2] = RandomMat(channels);
    bnweights[3] = RandomMat(channels);

    ncnnToVec(bnweights[0], bnGamma);
    ncnnToVec(bnweights[1], bnMean);
    ncnnToVec(bnweights[2], bnVar);
    ncnnToVec(bnweights[3], bnBeta);

    std::map<std::string, std::vector<float>> batchNormalization;
    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, w, h, inChs, outChs, kernel, dilation, stride, pad, false,
        snn::MRTMode::SINGLE_PLANE, useBN, batchNormalization, false, false);
    return 0;
}

extern "C"
JNIEXPORT int JNICALL Java_com_oppo_seattle_snndemo_NativeTests_conv2dVK16(JNIEnv * env, jclass, jobject java_am) {
    // remember the pointer to the asset manager
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    int w = 1080;
    int h = 1920;
    int inChs = 3;
    int outChs = 16;
    int kernel = 5;
    std::string activation = "relu";
    bool bias = true;
    int dilation = 1;
    int stride = 1;
    int pad = 0;
    bool useBN = false;

    ncnn::Mat matA = RandomMat(w, h, inChs);
    ShaderUnitTest test(snn::GpuBackendType::VULKAN);
    cv::Mat inputMat = NCNNMat2CVMat(matA);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outChs * inChs * kernel * kernel);
    if (bias) {
        weights[1] = RandomMat(outChs);
    }

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    std::vector<float> inputBias      = std::vector<float>(outChs, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        std::memcpy((uchar*) inputWeights[p].data, (uchar*) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }

    if (bias) {
        const float* ptr = weights[1].channel(0);
        for (size_t p = 0; p < inputBias.size(); p++) {
            inputBias[p] = ptr[p];
        }
    }

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    int channels = outChs;
    std::vector<ncnn::Mat> bnweights(4);
    bnweights[0] = RandomMat(channels);
    bnweights[1] = RandomMat(channels);
    bnweights[2] = RandomMat(channels);
    bnweights[3] = RandomMat(channels);

    ncnnToVec(bnweights[0], bnGamma);
    ncnnToVec(bnweights[1], bnMean);
    ncnnToVec(bnweights[2], bnVar);
    ncnnToVec(bnweights[3], bnBeta);

    std::map<std::string, std::vector<float>> batchNormalization;
    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, w, h, inChs, outChs, kernel, dilation, stride, pad, false,
        snn::MRTMode::SINGLE_PLANE, useBN, batchNormalization, false, true);
    (void) outFile;
    return 0;
}

extern "C"
JNIEXPORT void JNICALL Java_com_oppo_seattle_snndemo_NativeTests_setNumLoops(JNIEnv *, jclass, int numLoops) {
    innerLoops = numLoops;
    if (innerLoops > 1) {
        dumpResults = false;
    }
}
