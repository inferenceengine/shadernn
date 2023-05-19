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
#include "modelInference.h"
#include "snn/imageTexture.h"
#include "inferenceProcessor.h"
#include "testutil.h"
#include <string>
#include <vector>

// Helper function to load and preprocess the image
// params:
//  tex - input image
//  imageFileName - image file name
//  useHalfFP - flag to use FP16 calculations
//  mean - mean value. Used to perform normalizations
//  norm - normalization value (multiplier). Used to perform normalizations
static void loadAndPreprocessImage(snn::ImageTexture& tex, const std::string& imageFileName, bool useHalfFP, float mean, float norm) {
    tex.loadFromFile(imageFileName.c_str());
    std::vector<float> means(4, mean);
    std::vector<float> norms(4, norm);
    tex.convertToRGBA32FAndNormalize(means, norms);
    if (useHalfFP) {
        tex.convertFormat(snn::ColorFormat::RGBA16F);
    }
    tex.upload();
    SNN_LOGD("texture: %s", tex.getTextureInfo2().c_str());
}

// Helper function to do common work for each model
// params:
//  context - pointer to GPU context
//  imageFileName - image file name
//  useHalfFP - flag to use FP16 calculations
//  mean - mean value. Used to perform normalizations
//  norm - normalization value (multiplier). Used to perform normalizations
//  shared pointer to InferenceProcessor object
static void processModel(snn::GpuContext* context, const std::string& imageFileName, bool useHalfFP, float mean, float norm,
    std::shared_ptr<snn::InferenceProcessor> ip) {
    snn::ImageTextureArray inputTexs{snn::ImageTextureAllocator(context)};
    inputTexs.allocate(1);
    loadAndPreprocessImage(inputTexs[0], imageFileName, useHalfFP, mean, norm);

    ip->preProcess(inputTexs);

    snn::ImageTextureArray dummyOutputTexs{snn::ImageTextureAllocator(context)};
    ip->process(dummyOutputTexs);
}

int runSpatialDenoiser(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool /*useFinetuned*/, uint32_t innerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(useVulkan);

    std::string modelFileName = "SpatialDenoise/spatialDenoise.json";

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {1080, 1920, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::OTHER, innerLoops});

    processModel(ip->getContext(), snn::formatString("%sassets/images/bright_night_view_street_1080x1920.jpg", ASSETS_DIR), useHalfFP, 0.0f, 1.0 / 255.0f, ip);
    return 0;
}

int runAIDenoiser(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool /*useFinetuned*/, uint32_t innerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(false);

    std::string modelFileName = "AIDenoise/eff_predenoise_20200330.json";

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {1080, 1920, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::OTHER, innerLoops});

    snn::ImageTextureArray inputTexs{snn::ImageTextureAllocator(ip->getContext())};
    inputTexs.allocate(1);

    inputTexs[0].loadFromFile(snn::formatString("%sassets/images/empty_test_image.png", ASSETS_DIR).c_str());
    inputTexs[0].convertToRGBA32FAndNormalize();

    std::array<float, 4> resizeMeans {0.0, 0.0, 0.0, 0.0};
    std::array<float, 4> resizeNorms {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
    inputTexs[0].resize(1, 1, resizeMeans, resizeNorms);

    ip->preProcess(inputTexs);

    snn::ImageTextureArray dummyOutputTexs{snn::ImageTextureAllocator(ip->getContext())};
    ip->process(dummyOutputTexs);
    return 0;
}

int runResnet18(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned, uint32_t innerLoops, uint32_t outerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(useVulkan);

    std::string modelFileName;
    if (useFinetuned) {
        modelFileName = "Resnet18/resnet18_cifar10.json";
    } else {
        modelFileName = "Resnet18/resnet18_cifar10_0223_layers.json";
    }

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {32, 32, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::CLASSIFICATION, innerLoops});

    for (int i = 0; i < outerLoops; i++) {
        snn::ImageTextureArray inputTexs{snn::ImageTextureAllocator(ip->getContext())};
        inputTexs.allocate(1);

        std::string imageName;
        if (i + 1 < outerLoops) {
            imageName = snn::formatString("%sassets/images/cifar_test_dog.png", ASSETS_DIR);
        } else {
            // Testing that the next imput image overrides previous input image
            imageName = snn::formatString("%sassets/images/cifar_test.png", ASSETS_DIR);
        }

        loadAndPreprocessImage(inputTexs[0], imageName, useHalfFP, 127.5f, 1 / 127.5f);

        ip->preProcess(inputTexs);

        snn::ImageTextureArray dummyOutputTexs{snn::ImageTextureAllocator(ip->getContext())};
        ip->process(dummyOutputTexs);
    }
    return 0;
}

int runMobilenetV2(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned, uint32_t innerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(useVulkan);

    std::string modelFileName;
    std::string imageName;
    if (useFinetuned) {
        modelFileName = "MobileNetV2/mobilenetv2_pretrained_imagenet.json";
        imageName = "imagenet1.png";
    } else {
        modelFileName = "MobileNetV2/mobilenetv2_keras_layers.json";
        imageName = "ant.png";
    }

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {224, 224, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::CLASSIFICATION, innerLoops});

    processModel(ip->getContext(), snn::formatString("%sassets/images/%s", ASSETS_DIR, imageName.c_str()), useHalfFP, 0.0f, 1.0 / 255.0f, ip);
    return 0;
}

int runYolov3Tiny(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned, uint32_t innerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(useVulkan);

    std::string modelFileName;
    if (useFinetuned) {
        modelFileName = "Yolov3-tiny/yolov3-tiny_finetuned.json";
    } else {
        // This model misses the final layer and does not output any detections !!!
        modelFileName = "Yolov3-tiny/yolov3_tiny_bb_layers.json";
    }

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {416, 416, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::DETECTION, innerLoops});

    processModel(ip->getContext(), snn::formatString("%sassets/images/coco1_416.png", ASSETS_DIR), useHalfFP, 127.5f, 1 / 127.5f, ip);
    return 0;
}

int runUNet(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned, uint32_t innerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(useVulkan);

    std::string modelFileName;
    if (useFinetuned) {
        modelFileName = "U-Net/unet.json";
    } else {
        modelFileName = "U-Net/unet_layers.json";
    }

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {256, 256, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::OTHER, innerLoops});

    processModel(ip->getContext(), snn::formatString("%sassets/images/test_image_unet_gray.png", ASSETS_DIR), useHalfFP, 127.5f, 1 / 127.5f, ip);
    return 0;
}

int runStyleTransfer(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool /*useFinetuned*/, uint32_t innerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(useVulkan);

    std::string modelFileName = "StyleTransfer/candy-9_simplified.json";

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {224, 224, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::OTHER, innerLoops});

    processModel(ip->getContext(), snn::formatString("%sassets/images/ant.png", ASSETS_DIR), useHalfFP, 0.0f, 1.0f, ip);
    return 0;
}

int runESPCN(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool /*useFinetuned*/, uint32_t innerLoops) {
    CHECK_PLATFORM_SUPPORT(useVulkan)
    auto ip = snn::InferenceProcessor::create(useVulkan);

    std::string modelFileName = "ESPCN/ESPCN_2X_16_16_4.json";

    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {1080, 1920, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, useVulkan, snn::ModelType::OTHER, innerLoops});

    processModel(ip->getContext(), snn::formatString("%sassets/images/bright_night_view_street_1080x1920.jpg", ASSETS_DIR), useHalfFP, 127.5, 1 / 127.5, ip);
    return 0;
}
