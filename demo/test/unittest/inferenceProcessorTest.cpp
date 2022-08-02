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
#include "inferenceProcessor.h"

#include "snn/glUtils.h"
#include "snn/utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>

#include "matutil.h"

int runSpatialDenoiser(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP) {
    auto ip = new snn::InferenceProcessor();

    auto modelFileName = "SpatialDenoise/spatialDenoise.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {1080, 1920, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        inputTexs[0].loadFromFile(formatString("%sassets/images/bright_night_view_street_1080x1920.jpg", ASSETS_DIR).c_str());
        std::vector<float> means {0, 0, 0, 0};
        std::vector<float> norms {1, 1, 1, 1};
        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
        if (dumpOutputs) {
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);
        } else {
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA16F, means, norms);
        }
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].upload();

        std::vector<float> resizeMeans {0.0, 0.0, 0.0, 0.0};
        std::vector<float> resizeNorms {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
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
        ip->process(outputTexs);
    }

    ip->finalize();

    return 0;
}

int runAIDenoiser(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP) {
    auto ip = new snn::InferenceProcessor();

    auto modelFileName = "eff_predenoise_20200330-210658_e635_mixloss1.h5.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {1080, 1920, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        inputTexs[0].loadFromFile(formatString("%sassets/images/empty_test_image.png", ASSETS_DIR).c_str());
        std::vector<float> means {0, 0, 0, 0};
        std::vector<float> norms {1, 1, 1, 1};
        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
        if (dumpOutputs) {
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);
        } else {
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA16F, means, norms);
        }
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].upload();

        std::vector<float> resizeMeans {0.0, 0.0, 0.0, 0.0};
        std::vector<float> resizeNorms {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
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
        ip->process(outputTexs);
    }

    ip->finalize();

    return 0;
}

int runResnet18(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP) {
    // load input image
    auto ip = new snn::InferenceProcessor();

    // auto modelFileName = "Resnet18/resnet18_cifar10_0223.json";
    auto modelFileName = "Resnet18/resnet18_cifar10.json";
    // auto modelFileName = "resnet18_cifar10_0223_layers.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {32, 32, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        inputTexs[0].loadFromFile(formatString("%sassets/images/cifar_test.png", ASSETS_DIR).c_str());
        std::vector<float> means {0, 0, 0, 0};
        std::vector<float> norms {1, 1, 1, 1};
        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
        inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);
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

        if (useHalfFP) {
            inputTexs[0].download();
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA16F, means, norms);
            inputTexs[0].upload();
            // inputTexs[0].printOutWH();
        }

        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__, __LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
               (int) inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
        ip->process(outputTexs);
    }

    ip->finalize();

    return 0;
}

int runMobilenetV2(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP) {
    // load input image
    auto ip = new snn::InferenceProcessor();

    // auto modelFileName = "MobileNetV2/mobilenetv2_keras_dummy.json";
    // auto modelFileName = "MobileNetV2/mobilenetV2.json";
    auto modelFileName = "MobileNetV2/mobilenetv2_pretrained_imagenet.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {224, 224, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute});

    int loopCount = 2;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        // inputTexs[0].loadFromFile(formatString("%sassets/images/ant.png", ASSETS_DIR).c_str());
        inputTexs[0].loadFromFile(formatString("%sassets/images/imagenet1.png", ASSETS_DIR).c_str());
        std::vector<float> means {0.0, 0.0, 0.0, 0.0};
        std::vector<float> norms {1.0, 1.0, 1.0, 1.0};
        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
        inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].upload();

        std::vector<float> resizeMeans {0.0, 0.0, 0.0, 0.0};
        std::vector<float> resizeNorms {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        inputTexs[0].resize(1, 1, resizeMeans, resizeNorms);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].readTexture(0);

        if (useHalfFP) {
            inputTexs[0].download();
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA16F, means, norms);
            inputTexs[0].upload();
            // inputTexs[0].printOutWH();
        }

        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__, __LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
               (int) inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
        ip->process(outputTexs);
    }

    ip->finalize();

    return 0;
}

DECLARE_LAYER(YOLO);

int runYolov3Tiny(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP) {
    // load input image
    auto ip = new snn::InferenceProcessor();
    ip->registerLayer("YOLO", YOLOCreator);

    // auto modelFileName = "Yolov3-tiny/yolov3-tiny_dummy.json";
    auto modelFileName = "Yolov3-tiny/yolov3-tiny_finetuned.json";
    // auto modelFileName = "yolov3_tiny_bb_layers.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {416, 416, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        // inputTexs[0].loadFromFile(formatString("%sassets/images/arduino.png", ASSETS_DIR).c_str());
        inputTexs[0].loadFromFile(formatString("%sassets/images/coco1_416.png", ASSETS_DIR).c_str());
        std::vector<float> means {0, 0, 0, 0};
        std::vector<float> norms {1, 1, 1, 1};
        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
        inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);
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

        if (useHalfFP) {
            inputTexs[0].download();
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA16F, means, norms);
            inputTexs[0].upload();
            // inputTexs[0].printOutWH();
        }

        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__, __LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
               (int) inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
        ip->process(outputTexs);
    }

    ip->finalize();

    return 0;
}

int runUNet(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP) {
    // load input image
    auto ip = new snn::InferenceProcessor();

    // auto modelFileName = "U-Net/unet_dummy.json";
    auto modelFileName = "U-Net/unet.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {256, 256, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;
        inputTexs.allocate(1);

        // inputTexs[0].loadFromFile("images/test_image_unet_gray.png");
        // inputTexs[0].convertFormat(snn::ColorFormat::R32F);
        // printf("%s:%d\n", __FUNCTION__,__LINE__);
        // // inputTexs[0].printOutWH();
        // // inputTexs[0].upload();

        auto input = snn::ManagedRawImage::loadFromFile(formatString("%sassets/images/test_image_unet_gray.png", ASSETS_DIR).c_str());
        // for (std::size_t i = 0; i < input.size(); i++) {
        //     SNN_LOGI("%d", *(input.data() + i));
        // }
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        auto input32f = snn::toRgba32f(input, -1.0, 1.0);
        // for (std::size_t i = 0; i < input32f.size(); i+=4) {
        //     float f;
        //     uint8_t buffer[4] = {*(input32f.data() + i), *(input32f.data() + i+1), *(input32f.data() + i+2), *(input32f.data() + i+3)};
        //     std::memcpy(&f, buffer, sizeof(f));
        //     SNN_LOGI("%f", f);
        // }

        inputTexs.allocate(1);
        gl::TextureObject scaleTex;

        if (useHalfFP) {
            auto input16f = snn::toRgba16f(input32f);
            scaleTex.allocate2D(input16f.format(), input16f.width(), input16f.height(), 1, 1);
            printf("%s:%d\n", __FUNCTION__, __LINE__);
            scaleTex.setPixels(0, 0, 0, 256, 256, 0, input16f.data());
            scaleTex.detach();
        } else {
            scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 1, 1);
            printf("%s:%d\n", __FUNCTION__, __LINE__);
            scaleTex.setPixels(0, 0, 0, 256, 256, 0, input32f.data());
            scaleTex.detach();
        }
        inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

        // printf("%s:%d tex:%d, %d\n", __FUNCTION__,__LINE__, scaleTex.target(), scaleTex.id());
        // inputTexs.allocate(1);
        // inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__, __LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
               (int) inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
        ip->process(outputTexs);
    }

    ip->finalize();

    return 0;
}

int runStyleTransfer(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP) {
    auto ip = new snn::InferenceProcessor();

    auto modelFileName = "StyleTransfer/candy-9_simplified.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t> {224, 224, 1, 1}});
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

    ip->initialize({modelFileName, inputList, &rc, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute});

    int loopCount = 1;
    for (int i = 0; i < loopCount; i++) {
        snn::FixedSizeArray<snn::ImageTexture> inputTexs;

        inputTexs.allocate(1);
        // inputTexs[0].loadFromFile("images/cifar_test.png");
        // inputTexs[0].convertFormat(ColorFormat::RGBA32F);
        // inputTexs[0].upload();
        // inputTexs[0].printOutWH();

        inputTexs[0].loadFromFile(formatString("%sassets/images/ant.png", ASSETS_DIR).c_str());
        std::vector<float> means {0, 0, 0, 0};
        std::vector<float> norms {1, 1, 1, 1};
        // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
        // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};

        inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);

        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].upload();

        std::vector<float> resizeMeans {0.0, 0.0, 0.0, 0.0};
        // std::vector<float> resizeNorms{1.0/255.0, 1.0/255.0, 1.0/255.0, 1.0/255.0};
        std::vector<float> resizeNorms {1.0, 1.0, 1.0, 1.0};
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        inputTexs[0].resize(1, 1, resizeMeans, resizeNorms);
        printf("%s:%d\n", __FUNCTION__, __LINE__);
        // inputTexs[0].printOutWH();
        // inputTexs[0].readTexture(0);

        if (useHalfFP) {
            inputTexs[0].download();
            inputTexs[0].convertFormat(snn::ColorFormat::RGBA16F, means, norms);
            inputTexs[0].upload();
            // inputTexs[0].printOutWH();
        }

        printf("%s:%d, texture: %d, %d format: %d\n", __FUNCTION__, __LINE__, inputTexs[0].texture(0)->id(), inputTexs[0].texture(0)->target(),
               (int) inputTexs[0].texture(0)->getDesc().format);
        ip->preProcess(inputTexs);

        snn::FixedSizeArray<snn::ImageTexture> outputTexs;
        outputTexs.allocate(1);
        ip->process(outputTexs);
    }

    ip->finalize();

    return 0;
}

std::string printMRTMode(snn::MRTMode mrtMode) {
    switch (mrtMode) {
    case snn::MRTMode::SINGLE_PLANE:
        return std::string("SINGLE_PLANE MRT");

    case snn::MRTMode::DOUBLE_PLANE:
        return std::string("DOUBLE_PLANE MRT");

    case snn::MRTMode::QUAD_PLANE:
        return std::string("QUAD_PLANE MRT");

    default:
        return std::string("NO MRT");
    }
}

int main(int argc, char** argv) {
    SRAND(7767517);
    std::vector<int> modelIdx          = {};
    bool dumpOutputs                   = false;
    bool useCompute                    = false;
    bool useHalfFP                     = false;
    snn::MRTMode mrtMode               = snn::MRTMode::DOUBLE_PLANE;
    snn::WeightAccessMethod weightMode = snn::WeightAccessMethod::TEXTURES;
    if (argc <= 2) {
        printf("You can specify the model with argument\n");
        printf("The format expected is --use_1ch_mrt --use_constants --use_compute <model idx>, [<model idx> .. ] <dumpOutputs>\n");
    } else {
        int idx = 1;
        while (idx != argc - 1) {
            // printf("%d: %d, %s\n", idx, argc-1, argv[idx]);
            try {
                modelIdx.push_back(std::stoi(argv[idx]));
            } catch (std::exception& e) {
                printf("%s:%d: %d %s\n", __FUNCTION__, __LINE__, idx, argv[idx]);
                // printf("%d", strcmp(argv[idx], "--use_1ch_mrt"));
                if (strcmp(argv[idx], "--use_1ch_mrt") == 0) {
                    mrtMode = snn::MRTMode::SINGLE_PLANE;
                }

                if (strcmp(argv[idx], "--use_constants") == 0) {
                    weightMode = snn::WeightAccessMethod::CONSTANTS;
                }

                if (strcmp(argv[idx], "--use_compute") == 0) {
                    useCompute = true;
                }

                if (strcmp(argv[idx], "--use_half") == 0) {
                    useHalfFP = true;
                }
            }
            idx++;
        }
        if (modelIdx.empty()) {
            modelIdx.push_back(0);
        }
        dumpOutputs = std::stoi(argv[argc - 1]);
    }
    for (auto model : modelIdx) {
        switch (model) {
        case 0: {
            printf("To test model Resnet18\n%s\n", printMRTMode(mrtMode).c_str());
            runResnet18(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP);
            break;
        }
        case 1: {
            printf("To test model YOLO v3 Tiny\n%s\n", printMRTMode(mrtMode).c_str());
            runYolov3Tiny(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP);
            break;
        }
        case 2: {
            printf("To test model UNet\n%s\n", printMRTMode(mrtMode).c_str());
            runUNet(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP);
            break;
        }
        case 3: {
            printf("To test model Mobilenet V2\n%s\n", printMRTMode(mrtMode).c_str());
            runMobilenetV2(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP);
            break;
        }
        case 4: {
            printf("To test model spatial denoiser\n%s\n", printMRTMode(mrtMode).c_str());
            runSpatialDenoiser(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP);
            break;
        }
        case 5: {
            printf("To test model AI denoiser\n%s\n", printMRTMode(mrtMode).c_str());
            runAIDenoiser(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP);
            break;
        }
        case 6: {
            printf("To test model Style Transfer\n%s\n", printMRTMode(mrtMode).c_str());
            runStyleTransfer(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP);
            break;
        }
        default: {
            auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);

            snn::FixedSizeArray<snn::ImageTexture> inputTexs;
            inputTexs.allocate(1);

            inputTexs[0].loadFromFile("../data/assets/images/cifar_test.png");
            std::vector<float> means {0, 0, 0, 0};
            std::vector<float> norms {1, 1, 1, 1};
            // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
            // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
            // inputTexs[0].convertFormat(snn::ColorFormat::RGBA32F, means, norms);
            printf("%s:%d\n", __FUNCTION__, __LINE__);
            inputTexs[0].printOutWH();
            inputTexs[0].upload();

            // std::vector<float> resizeMeans{127.5, 127.5, 127.5, 127.5};
            // std::vector<float> resizeNorms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
            std::vector<float> resizeMeans {0, 0, 0, 0};
            std::vector<float> resizeNorms {1, 1, 1, 1};
            printf("%s:%d\n", __FUNCTION__, __LINE__);
            // inputTexs[0].resize(4, 4, resizeMeans, resizeNorms);
            // printf("%s:%d\n", __FUNCTION__,__LINE__);
            // inputTexs[0].printOutWH();

            gl::TextureObject _resizeTexture;
            if (inputTexs[0].depth(0) > 1) {
                printf("%s:%d\n", __FUNCTION__, __LINE__);
                _resizeTexture.allocate2DArray(snn::ColorFormat::RGBA32F, 8, 8, inputTexs[0].depth(0));
            } else {
                printf("%s:%d\n", __FUNCTION__, __LINE__);
                _resizeTexture.allocate2D(snn::ColorFormat::RGBA32F, 8, 8);
            }
            inputTexs[0].resizeTexture(*(inputTexs[0].texture(0)), _resizeTexture, 4, 4, resizeMeans, resizeNorms);
        }
        }
    }

    return 0;
}
