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
#include "shaderUnitTest.h"
#include "snn/snn.h"
#include "snn/contextFactory.h"
#include "snn/imageTextureFactory.h"
#include "snn/core.h"
#include "net.h"
#include "layer.h"
#include "ic2/dp.h"
#include "ic2/layerFactory.h"
#include "ic2/activation.h"
#include "ic2/addlayer.h"
#include "ic2/avgpool2d.h"
#include "ic2/batchnorm.h"
#include "ic2/concatenation.h"
#include "ic2/conv2d.h"
#include "ic2/cpulayer.h"
#include "ic2/denselayer.h"
#include "ic2/flattenlayer.h"
#include "ic2/inputlayer.h"
#include "ic2/instancenorm.h"
#include "ic2/maxpool2d.h"
#include "ic2/padlayer.h"
#include "ic2/separableconvolution.h"
#include "ic2/subpixelmerge.h"
#include "ic2/upsampling2d.h"
#include "ic2/yololayer.h"
#include "ic2/unary.h"

#define NEW_LAYER(layer, desc) \
    std::shared_ptr<snn::dp::GenericModelLayer>(snn::dp::layer##Creator1(std::move(desc), useVulkan()));

void logTimingStatistics(const std::map<std::string, std::vector<double>>& timeMap);

#ifndef LOG_TAG
    #define LOG_TAG "ShaderUnitTest"
#endif

static const uint32_t ALIGNED_CH = 4U; // For RGBA texture or VkImage

ShaderUnitTest::ShaderUnitTest(snn::GpuBackendType backend)
{
    context = snn::createDefaultContext(backend == snn::GpuBackendType::VULKAN);
}

void ShaderUnitTest::_tic(timespec& ticker) {
    clock_gettime(CLOCK_MONOTONIC, &ticker);
}

double ShaderUnitTest::_toc(timespec& ticker) {
    struct timespec after;
    clock_gettime(CLOCK_MONOTONIC, &after);
    double us = (after.tv_sec - ticker.tv_sec) * 1000000.0f;

    return us + (after.tv_nsec - ticker.tv_nsec) / 1000.0f;
}

snn::ManagedRawImage cvMat2ManagedRawImage(cv::Mat& in, char* imageMemory) {
    snn::ManagedRawImage result(snn::ImageDesc(snn::ColorFormat::RGBA32F, in.size[1], in.size[0], in.size[2]), imageMemory);

    for (int i = 0; i < in.size[0]; i++) {
        for (int j = 0; j < in.size[1]; j++) {
            snn::Rgba32f* dst = (snn::Rgba32f*) result.at(0, j, i, 0);
            (*dst).red     = in.at<float>(i, j, 0);
            (*dst).green   = in.at<float>(i, j, 1);
            (*dst).blue    = in.at<float>(i, j, 2);
            (*dst).alpha   = in.at<float>(i, j, 3);
        }
    }

    return result;
}

void hwcToC4(float *buffer, int input_h, int input_w, int input_c, float* dst, bool preferrHalfPrecision = false) {
    if (preferrHalfPrecision) {
        // Change OpenCV HWC to C/4HW4
        uint16_t* inputData = (uint16_t*) dst;
        for (int p = 0; p < ROUND_UP(input_c, ALIGNED_CH); p += ALIGNED_CH) {
            for (int i = 0; i < input_h * input_w; i++) {
                if (p + 0 < input_c) {
                    *(inputData + i * 4 + 0) = snn::FP32::toHalf(*((float*) buffer + i * input_c + p + 0));
                }
                if (p + 1 < input_c) {
                    *(inputData + i * 4 + 1) = snn::FP32::toHalf(*((float*) buffer + i * input_c + p + 1));
                }
                if (p + 2 < input_c) {
                    *(inputData + i * 4 + 2) = snn::FP32::toHalf(*((float*) buffer + i * input_c + p + 2));
                }
                if (p + 3 < input_c) {
                    *(inputData + i * 4 + 3) = snn::FP32::toHalf(*((float*) buffer + i * input_c + p + 3));
                }
            }
            inputData += input_h * input_w * 4;
        }
    } else {
        // Change OpenCV HWC to C/4HW4
        float* inputData = (float*) dst;
        for (int p = 0; p < ROUND_UP(input_c, ALIGNED_CH); p += ALIGNED_CH) {
            for (int i = 0; i < input_h * input_w; i++) {
                if (p + 0 < input_c) {
                    *(inputData + i * 4 + 0) = *((float*) buffer + i * input_c + p + 0);
                }
                if (p + 1 < input_c) {
                    *(inputData + i * 4 + 1) = *((float*) buffer + i * input_c + p + 1);
                }
                if (p + 2 < input_c) {
                    *(inputData + i * 4 + 2) = *((float*) buffer + i * input_c + p + 2);
                }
                if (p + 3 < input_c) {
                    *(inputData + i * 4 + 3) = *((float*) buffer + i * input_c + p + 3);
                }
            }
            inputData += input_h * input_w * 4;
        }
    }
}

static int getPassIndex(int idex, snn::MRTMode mrt) {
    int depth = 0;
    switch (mrt) {
    case snn::MRTMode::SINGLE_PLANE:
        return idex;

    case snn::MRTMode::DOUBLE_PLANE:
        depth = (idex + 1) * 4;
        return DIV_AND_ROUND_UP(depth, 8) - 1;

    case snn::MRTMode::QUAD_PLANE:
        depth = (idex + 1) * 4;
        return DIV_AND_ROUND_UP(depth, 16) - 1;

    default:
        return 0;
    }
}

std::shared_ptr<snn::ImageTexture> ShaderUnitTest::createInputImgTxt(const std::array<uint32_t, 4>& dims, snn::ColorFormat colorFormat, const float* pixels) {
    std::shared_ptr<snn::ImageTexture> imgTxt = snn::ImageTextureFactory::createImageTexture(context, dims, colorFormat, pixels);
    imgTxt->upload();
    return imgTxt;
}

snn::ImageTextureArray ShaderUnitTest::createInputImgTxt(float* dest, int width, int height, int inChannels, bool fp16) {
    snn::ColorFormat colorFormat = fp16 ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;
    const uint32_t inDepth = UP_DIV(inChannels, ALIGNED_CH);
    std::array<uint32_t, 4> dims {(uint32_t) width, (uint32_t) height, inDepth, 1U};
    std::shared_ptr<snn::ImageTexture> imgTxt = createInputImgTxt(dims, colorFormat, dest);
    snn::ImageTextureArray inputTexs{imgTxt, snn::ImageTextureAllocator(context)};
    return inputTexs;
}

snn::ImageTextureArray ShaderUnitTest::createOutputImgTxt(int outWidth, int outHeight, int outChannels, bool fp16) {
    snn::ColorFormat colorFormat = fp16 ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;
    const uint32_t outDepth = UP_DIV(outChannels, ALIGNED_CH);
    std::array<uint32_t, 4> dims {(uint32_t)outWidth, (uint32_t)outHeight, outDepth, 1U};
    std::shared_ptr<snn::ImageTexture> imgTxt = snn::ImageTextureFactory::createImageTexture(context, dims, colorFormat);
    snn::ImageTextureArray outputTexs(imgTxt, snn::ImageTextureAllocator(context));
    return outputTexs;
}

std::string ShaderUnitTest::snnConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width, int height,
    int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, bool useCompute, snn::MRTMode mrtMode,
    bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput, bool fp16) {
    std::string ret;
    std::vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    SNN_LOGD("width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, useCompute:%d",
             width, height, inChannels, outChannels, kernel, dilation, stride, pad, useCompute);

    bool preferrHalfPrecision = fp16;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::Conv2DDesc desc;
    desc.mrtMode               = mrtMode;
    desc.isRange01             = 0;
    desc.numOutputPlanes       = outChannels;
    desc.numInputPlanes        = inChannels;
    desc.weightsCvM            = inputWeights;
    desc.biases                = doubleBias;
    desc.activation            = "";
    desc.kernelSize            = kernel;
    desc.stride                = stride;
    desc.useBatchNormalization = useBatchNorm;
    desc.batchNormalization    = batchNormalization;
    desc.useMultiInputs        = false;
    desc.padding               = "same";
    desc.paddingT              = std::to_string(kernel / 2);
    desc.paddingB              = std::to_string(kernel / 2);
    desc.paddingL              = std::to_string(kernel / 2);
    desc.paddingR              = std::to_string(kernel / 2);
    if (pad == 0) {
        desc.paddingMode = "constant";
    } else if (pad == 1) {
        desc.paddingMode = "replicate";
    } else if (pad == 2) {
        desc.paddingMode = "reflect";
    }
    desc.preferHp = preferrHalfPrecision;
    /* if input planes is greater than 64, set the weight mode TEXTURES */
    desc.weightMode = snn::WeightAccessMethod::TEXTURES;

    auto layer = NEW_LAYER(Conv2D, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] Conv2D");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest, preferrHalfPrecision);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = useCompute;
    sgo.vulkan                    = useVulkan();

    sgo.mrtMode = mrtMode;
    sgo.preferrHalfPrecision = preferrHalfPrecision;

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels, fp16);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels, fp16);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    if (!sgo.compute && !sgo.vulkan) {
        ret = layer->getName() + " pass[" + std::to_string(getPassIndex((outChannels+3)/4-1, sgo.mrtMode)) + "].dump";
    }
    return ret;
}

std::string ShaderUnitTest::snnDenseTestWithLayer(cv::Mat& inputMat, std::vector<std::vector<float>>& inputWeights,
    std::vector<float>& inputBias, int width, int height, int inChannels, int outChannels, bool dumpOutput) {
    std::string ret;

    SNN_LOGV("width:%d, height:%d, inChannels:%d, outChannels:%d",
             width, height, inChannels, outChannels);

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int outWidth    = inputBias.size();
    int outHeight   = 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::DenseDesc desc;
    desc.isRange01       = 0;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.weights         = inputWeights;
    desc.biases          = inputBias;
    desc.activation            = "";
    desc.preferHp = preferrHalfPrecision;
    // if input planes is greater than 64, set the weight mode TEXTURES
    desc.weightMode = snn::WeightAccessMethod::TEXTURES;

    auto layer = NEW_LAYER(Dense, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] Dense");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    sgo.mrtMode = snn::MRTMode::SINGLE_PLANE;

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest, preferrHalfPrecision);

    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}


std::string ShaderUnitTest::snnPoolingTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel,
    int stride, int poolingType, int padMode, bool dumpOutput) {
    std::string ret;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    std::shared_ptr<snn::dp::GenericModelLayer> layer;
    if (poolingType == 0) {
        snn::dp::MaxPooling2DDesc desc;
        desc.numOutputPlanes = outChannels;
        desc.numInputPlanes  = inChannels;
        desc.activation      = "";
        desc.kernelSize      = kernel;
        desc.stride          = stride;
        if (padMode == 2) {
            desc.padding = "same_upper";
        } else if (padMode == 3) {
            desc.padding = "same_lower";
        }
        desc.paddingT = "same";
        desc.paddingB = "same";
        desc.paddingL = "same";
        desc.paddingR = "same";
        desc.preferHp = preferrHalfPrecision;
        desc.mrtMode  = snn::MRTMode::SINGLE_PLANE;
        layer = NEW_LAYER(MaxPooling2D, desc);
    } else if (poolingType == 1) {
        snn::dp::AveragePooling2DDesc desc;
        desc.numOutputPlanes = outChannels;
        desc.numInputPlanes  = inChannels;
        desc.activation      = "";
        desc.kernelSize      = kernel;
        desc.stride          = stride;
        if (padMode == 2) {
            desc.padding = "same_upper";
        } else if (padMode == 3) {
            desc.padding = "same_lower";
        }
        desc.paddingT = "same";
        desc.paddingB = "same";
        desc.paddingL = "same";
        desc.paddingR = "same";
        desc.preferHp = preferrHalfPrecision;
        desc.mrtMode  = snn::MRTMode::SINGLE_PLANE;
        layer = NEW_LAYER(AveragePooling2D, desc);
    }

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] Pooling");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    sgo.mrtMode = snn::MRTMode::SINGLE_PLANE;

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

std::string ShaderUnitTest::snnAddTestWithLayer2(cv::Mat& inputMat, int width, int height, int inChannels, int kernel,
    int stride, const std::string& activation, bool dumpOutput) {
    std::string ret;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::MaxPooling2DDesc desc;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.activation      = activation;
    desc.kernelSize      = kernel;
    desc.stride          = stride;
    desc.paddingT        = "valid";
    desc.paddingB        = "valid";
    desc.paddingL        = "valid";
    desc.paddingR        = "valid";
    desc.preferHp        = preferrHalfPrecision;
    desc.mrtMode         = snn::MRTMode::SINGLE_PLANE;

    auto layer = NEW_LAYER(MaxPooling2D, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] MaxPooling");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    snn::dp::AddDesc addDesc;
    addDesc.numOutputPlanes = outChannels;
    addDesc.numInputPlanes  = inChannels;
    addDesc.activation      = activation;
    addDesc.preferHp = preferrHalfPrecision;
    addDesc.mrtMode  = snn::MRTMode::SINGLE_PLANE;

    auto addLayer = NEW_LAYER(Add, addDesc);

    addLayer->prevLayers.push_back(inputLayer);
    addLayer->prevLayers.push_back(layer);
    addLayer->nextLayers.clear();
    addLayer->setName("resnet18_cifar10_0223.json layer [02] Add");

    // Do we need these two?
    layer->nextLayers.push_back(addLayer);
    inputLayer->nextLayers.push_back(addLayer);

    layers.emplace_back(addLayer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest, preferrHalfPrecision);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    sgo.mrtMode = snn::MRTMode::SINGLE_PLANE;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = addLayer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

std::string ShaderUnitTest::snnMultiInputsTestWithLayer(cv::Mat& inputMat1, cv::Mat& inputMat2, int width, int height, int inChannels, bool dumpOutput) {
    std::string ret;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int outWidth    = width;
    int outHeight   = height;
    int outChannels = inChannels;

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc input0Desc;
    input0Desc.inputHeight   = width;
    input0Desc.inputWidth    = height;
    input0Desc.inputChannels = inChannels;
    input0Desc.numInputPlanes  = inChannels;
    input0Desc.numOutputPlanes = inChannels;
    input0Desc.isInputLayer = true;
    input0Desc.inputIndex = 0;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer0(new snn::dp::InputLayerLayer(std::move(input0Desc)));
    inputLayer0->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer0->prevLayers.clear();

    snn::dp::InputLayerDesc input1Desc;
    input1Desc.inputHeight   = width;
    input1Desc.inputWidth    = height;
    input1Desc.inputChannels = inChannels;
    input1Desc.numInputPlanes  = inChannels;
    input1Desc.numOutputPlanes = inChannels;
    input1Desc.isInputLayer = true;
    input1Desc.inputIndex = 1;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer1(new snn::dp::InputLayerLayer(std::move(input1Desc)));

    inputLayer1->nextLayers.clear();
    inputLayer1->setName("resnet18_cifar10_0223.json layer [01] InputLayerLayer");

    layers.emplace_back(inputLayer0);
    layers.emplace_back(inputLayer1);

    snn::dp::AddDesc addDesc;
    addDesc.numOutputPlanes = outChannels;
    addDesc.numInputPlanes  = inChannels;
    addDesc.activation      = "linear";
    addDesc.preferHp = preferrHalfPrecision;
    addDesc.mrtMode  = snn::MRTMode::SINGLE_PLANE;

    auto addLayer = NEW_LAYER(Add, addDesc);

    addLayer->prevLayers.push_back(inputLayer0);
    addLayer->prevLayers.push_back(inputLayer1);
    addLayer->nextLayers.clear();
    addLayer->setName("resnet18_cifar10_0223.json layer [02] Add");

    // Do we need these two?
    inputLayer0->nextLayers.push_back(addLayer);
    inputLayer1->nextLayers.push_back(addLayer);

    layers.emplace_back(addLayer);

    std::array<uint32_t, 4> dims {(uint32_t) width, (uint32_t) height, UP_DIV(inChannels, ALIGNED_CH), 1};

    std::vector<float> dest_vec1(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest1 = dest_vec1.data();
    hwcToC4((float*)inputMat1.data, inputMat1.size[0], inputMat1.size[1], inputMat1.size[2], dest1, preferrHalfPrecision);

    std::vector<float> dest_vec2(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest2 = dest_vec2.data();
    hwcToC4((float*)inputMat2.data, inputMat2.size[0], inputMat2.size[1], inputMat2.size[2], dest2, preferrHalfPrecision);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    sgo.mrtMode = snn::MRTMode::SINGLE_PLANE;

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    std::shared_ptr<snn::ImageTexture> img1 = createInputImgTxt(dims, snn::ColorFormat::RGBA32F, dest1);
    std::shared_ptr<snn::ImageTexture> img2 = createInputImgTxt(dims, snn::ColorFormat::RGBA32F, dest2);
    snn::ImageTextureArray imgs{snn::ImageTextureAllocator(context)};
    imgs.allocate(2);
    imgs.data()[0] = img1;
    imgs.data()[1] = img2;

    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = addLayer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

static void print_3d_cvmat(cv::Mat outputMat) {
    printf("---------------output of opencv 3d mat w first----------- \n");
    for (int k = 0; k < outputMat.size[2]; k++) {
        for (int i = 0; i < outputMat.size[0]; i++) {
            for (int j = 0; j < outputMat.size[1]; j++) {
                std::cout << std::setw(7) << outputMat.at<float>(i, j, k) << ",";
            }
            std::cout << std::endl;
        }
        std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }
    std::cout << std::endl;
}

std::string ShaderUnitTest::snnUpsampleTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int scale, int type,
    bool useCompute, bool dumpOutput) {
    std::string ret;
    (void) useCompute;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int outWidth    = (width) *scale;
    int outHeight   = (height) *scale;
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight     = width;
    inputDesc.inputWidth      = height;
    inputDesc.inputChannels   = inChannels;
    inputDesc.numOutputPlanes = outChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::UpSampling2DDesc desc;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.scale           = scale;
    if (type == 1) {
        desc.interpolationType = "nearest";
    }
    if (type == 2) {
        desc.interpolationType = "bilinear";
    }
    desc.preferHp = preferrHalfPrecision;

    std::shared_ptr<snn::dp::GenericModelLayer> layer;
    layer = NEW_LAYER(UpSampling2D, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] Upsample");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

std::string ShaderUnitTest::snnConcateTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride,
    const std::string& activation, bool dumpOutput) {
    std::string ret;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int outWidth    = width;
    int outHeight   = height;
    int outChannels = 2 * inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::MaxPooling2DDesc maxPoolDesc;
    maxPoolDesc.numOutputPlanes = inChannels;
    maxPoolDesc.numInputPlanes  = inChannels;
    maxPoolDesc.activation      = activation;
    maxPoolDesc.kernelSize      = kernel;
    maxPoolDesc.stride          = stride;
    maxPoolDesc.paddingT        = "valid";
    maxPoolDesc.paddingB        = "valid";
    maxPoolDesc.paddingL        = "valid";
    maxPoolDesc.paddingR        = "valid";
    maxPoolDesc.preferHp        = preferrHalfPrecision;
    maxPoolDesc.mrtMode         = snn::MRTMode::SINGLE_PLANE;

    auto maxPoolLayer = NEW_LAYER(MaxPooling2D, maxPoolDesc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    maxPoolLayer->prevLayers.push_back(inputLayer);
    maxPoolLayer->nextLayers.clear();
    maxPoolLayer->setName("resnet18_cifar10_0223.json actLayer [01] MaxPooling");

    inputLayer->nextLayers.push_back(maxPoolLayer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(maxPoolLayer);

    snn::dp::ConcatenateDesc concatDesc0;
    concatDesc0.numOutputPlanes = 2 * inChannels;
    concatDesc0.numInputPlanes  = inChannels;
    concatDesc0.preferHp        = preferrHalfPrecision;
    concatDesc0.mrtMode         = snn::MRTMode::SINGLE_PLANE;

    SNN_LOGV("numInputPlanes: %d; numOutputPlanes: %d", concatDesc0.numInputPlanes, concatDesc0.numOutputPlanes);

    auto concatLayer0 = NEW_LAYER(Concatenate, concatDesc0);

    concatLayer0->prevLayers.push_back(inputLayer);
    concatLayer0->prevLayers.push_back(maxPoolLayer);
    concatLayer0->nextLayers.clear();
    concatLayer0->setName("resnet18_cifar10_0223.json actLayer [02] Concate0");

    // Do we need these two?
    maxPoolLayer->nextLayers.push_back(concatLayer0);
    inputLayer->nextLayers.push_back(concatLayer0);

    layers.emplace_back(concatLayer0);

    snn::dp::ConcatenateDesc concatDesc1;
    concatDesc1.numOutputPlanes = (2 + 1) * inChannels;
    concatDesc1.preferHp = preferrHalfPrecision;
    concatDesc1.mrtMode  = snn::MRTMode::SINGLE_PLANE;

    auto concatLayer1 = NEW_LAYER(Concatenate, concatDesc1);

    concatLayer1->prevLayers.push_back(inputLayer);
    concatLayer1->prevLayers.push_back(concatLayer0);
    concatLayer1->nextLayers.clear();
    concatLayer1->setName("resnet18_cifar10_0223.json actLayer [02] Concate1");

    // Do we need these two?
    inputLayer->nextLayers.push_back(concatLayer1);
    concatLayer0->nextLayers.push_back(concatLayer1);

    layers.emplace_back(concatLayer1);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = concatLayer1->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}


std::string ShaderUnitTest::snnDepthConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width,
    int height, int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, int bias,
    bool useCompute, bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput) {
    std::string ret;
    std::vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    SNN_LOGV("width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useCompute:%d",
             width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useCompute);

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::SeparableConv2DDesc desc;
    desc.isRange01       = 0;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.weightsCvM      = inputWeights;
    desc.biases          = doubleBias;
    desc.activation            = "";
    desc.kernelSize            = kernel;
    desc.stride                = stride;
    desc.useBatchNormalization = useBatchNorm;
    desc.batchNormalization    = batchNormalization;
    desc.padding               = "valid";
    desc.paddingT              = "valid";
    desc.paddingB              = "valid";
    desc.paddingL              = "valid";
    desc.paddingR              = "valid";
    desc.preferHp              = preferrHalfPrecision;
    desc.mrtMode               = snn::MRTMode::DOUBLE_PLANE;
    desc.weightMode            = snn::WeightAccessMethod::TEXTURES;

    auto layer = NEW_LAYER(SeparableConv2D, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] DepthConv2D");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = useCompute;
    sgo.vulkan                    = useVulkan();
    if (!sgo.compute && !sgo.vulkan) {
        sgo.mrtMode                   = snn::MRTMode::DOUBLE_PLANE;
        sgo.weightMode                = snn::WeightAccessMethod::TEXTURES;
    }
    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    if (!sgo.compute && !sgo.vulkan) {
        ret = layer->getName() + " pass[" + std::to_string(getPassIndex((outChannels+3)/4-1, sgo.mrtMode)) + "].dump";
    }
    return ret;
}

std::string ShaderUnitTest::snnInstanceNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width,
    int height, int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, int bias,
    bool useCompute, bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput) {
    std::string ret;
    std::vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    SNN_LOGV("width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useCompute:%d",
             width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useCompute);

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::InstanceNormDesc desc;
    desc.isRange01       = 0;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.weightsCvM      = inputWeights;
    desc.biases          = doubleBias;
    desc.activation               = "";
    desc.kernelSize               = kernel;
    desc.stride                   = stride;
    desc.useInstanceNormalization = useBatchNorm;
    desc.instanceNormalization    = batchNormalization;
    desc.padding                  = "same";
    desc.preferHp = preferrHalfPrecision;

    auto layer = NEW_LAYER(InstanceNorm, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] InstanceNorm");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

std::string ShaderUnitTest::snnPadTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel,
    int stride, int type, float value, bool dumpOutput) {
    std::string ret;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width + 2 * paddingSize);
    int outHeight   = (height + 2 * paddingSize);
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    std::shared_ptr<snn::dp::GenericModelLayer> layer;

    snn::dp::PadDesc desc;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.activation      = "";
    desc.kernelSize      = kernel;
    desc.stride          = stride;
    desc.paddingT        = "same";
    desc.paddingB        = "same";
    desc.paddingL        = "same";
    desc.paddingR        = "same";
    desc.preferHp        = preferrHalfPrecision;
    desc.mrtMode         = snn::MRTMode::SINGLE_PLANE;
    if (type == 0) {
        desc.mode = "constant";
    } else if (type == 1) {
        desc.mode = "replicate";
    } else if (type == 2) {
        desc.mode = "reflect";
    }
    layer = NEW_LAYER(Pad, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] Padding");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

std::string ShaderUnitTest::snnBatchNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width,
    int height, int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, int bias,
    bool useCompute, bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput) {
    std::string ret;
    std::vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    SNN_LOGD("width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useCompute:%d",
             width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useCompute);

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::BatchNormalizationDesc desc;
    desc.isRange01          = 0;
    desc.numOutputPlanes    = outChannels;
    desc.numInputPlanes     = inChannels;
    desc.weightsCvM         = inputWeights;
    desc.biases             = doubleBias;
    desc.batchNormalization = batchNormalization;
    desc.activation         = "";
    desc.kernelSize         = kernel;
    desc.stride             = stride;
    desc.preferHp           = preferrHalfPrecision;

    auto layer = NEW_LAYER(BatchNormalization, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] BatchNorm");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

std::string ShaderUnitTest::snnActivationTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel,
    int stride, const std::string& activation, float leaky_val, bool dumpOutput) {
    std::string ret;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;
    int outChannels = inChannels;

    // Create single actLayer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    snn::dp::ActivationDesc desc;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.activation      = activation;
    desc.kernelSize      = kernel;
    desc.preferHp        = preferrHalfPrecision;
    desc.mrtMode         = snn::MRTMode::SINGLE_PLANE;
    desc.leakyReluAlpha  = leaky_val;

    auto actLayer = NEW_LAYER(Activation, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    actLayer->prevLayers.push_back(inputLayer);
    actLayer->nextLayers.clear();
    actLayer->setName("resnet18_cifar10_0223.json actLayer [01] Activation");

    inputLayer->nextLayers.push_back(actLayer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(actLayer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.mrtMode                   = snn::MRTMode::SINGLE_PLANE;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = actLayer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}

/**/
static float* oihw2hwo4i4(std::vector<cv::Mat>& inputWeights, int inChannels, int outChannels, int fw, int fh, int unit = 4) {
    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh * ROUND_UP(inChannels, unit);
    float* out            = (float*) malloc(alignedWeightSize * sizeof(float));
    int planeSize         = ROUND_UP(outChannels, unit) * ROUND_UP(inChannels, unit);
    memset(out, 0, alignedWeightSize * sizeof(float));
    for (int b = 0; b < outChannels; ++b) {
        int b_4 = b / unit;
        int mx  = b % unit;
        for (int d = 0; d < inChannels; ++d) {
            for (int y = 0; y < fh; ++y) {
                for (int x = 0; x < fw; ++x) {
                    int base                                 = (y * fw + x) * planeSize;
                    int inSize                               = ROUND_UP(inChannels, unit) * unit;
                    out[base + inSize * b_4 + d * unit + mx] = inputWeights[b * inChannels + d].at<float>(y * fw + x);
                }
            }
        }
    }
    return out;
}

static cv::Mat texture2NCNNMat(float* buf, int width, int height, int actualChanels) {
    cv::Mat outputMat;
    {
        // int size[3] = {width, height, (actualChanels+3)/4*4};
        int size[3] = {width, height, actualChanels};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        printf("Dim: %d, %d, %d, actual channels: %d\n", width, height, ROUND_UP(actualChanels, ALIGNED_CH), actualChanels);
        int idx = 0;
        for (int p4 = 0; p4 < UP_DIV(size[2], ALIGNED_CH); p4++) {
            float fr      = 0;
            int actualLen = 4;
            for (int i = 0; i < size[0] * size[1]; i++) {
                float* ind = (float*) outputMat.data;
                for (int j = 0; j < actualLen; j++) {
                    fr = *(buf + (idx++));
                    if (p4 * 4 + j < actualChanels) {
                        *(ind + i * size[2] + p4 * 4 + j) = fr;
                    }
                }
            }
        }
    }

    return outputMat;
}

using namespace snn;

void printManagedRawImage(ManagedRawImage& img) {
    printf("--------output of ManagedRawImage width first-- %d:%d:%d:%d-- \n", img.planes(), img.width(), img.height(), img.depth());
    for (uint32_t p = 0; p < img.planes(); p++) {
        for (uint32_t k = 0; k < img.depth(); k++) {
            for (uint32_t j = 0; j < img.height(); j++) {
                for (uint32_t i = 0; i < img.width(); i++) {
                    // std::cout << "M(" << p << ", " << k << ", " << j <<  ", " << i << "): ";
                    if (img.format() == snn::ColorFormat::RGBA8) {
                        auto v = (int) *(img.at(p, i, j, k));
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(img.at(p, i, j, k) + 1);
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(img.at(p, i, j, k) + 2);
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(img.at(p, i, j, k) + 3);
                        std::cout << std::setw(4) << v << ",";
                    } else if (img.format() == snn::ColorFormat::RGBA32F) {
                        auto v = *((float*) img.at(p, i, j, k));
                        std::cout << std::setw(7) << v << ",";
                        v = *((float*) (img.at(p, i, j, k) + 4));
                        std::cout << std::setw(7) << v << ",";
                        v = *((float*) (img.at(p, i, j, k) + 8));
                        std::cout << std::setw(7) << v << ",";
                        v = *((float*) (img.at(p, i, j, k) + 12));
                        std::cout << std::setw(7) << v << ",";
                    } else {
                        auto v = (int) *(img.at(p, i, j, k));
                        std::cout << std::setw(4) << v << ",";
                    }
                }
                std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
                std::cout << std::endl;
            }
            std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
        }
        std::cout << "**************" + std::to_string(p) + "***********" << std::endl;
    }
}

void memoryTest() {
    for (int i = 0; i < 10; i++) {
        char* mem = (char*) malloc(2048);
        memset(mem, 0, 2048);
        free(mem);
    }

    for (int i = 0; i < 10; i++) {
        char* mem = (char*) malloc(4096);
        memset(mem, 0, 4096);
        free(mem);
    }
}

void my_print_3d_cvmat(cv::Mat outputMat) {
    printf("---------------output of opencv 3d mat w first----------- \n");
    for (int k = 0; k < outputMat.size[2]; k++) {
        for (int i = 0; i < outputMat.size[0]; i++) {
            for (int j = 0; j < outputMat.size[1]; j++) {
                std::cout << std::setw(7) << outputMat.at<float>(i, j, k) << ",";
            }
            std::cout << std::endl;
        }
        std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }
    std::cout << std::endl;
}

void ShaderUnitTest::testImageTexture(cv::Mat& inputMat, int width, int height, int inChannels) {
    std::array<uint32_t, 4> dims {(uint32_t) width, (uint32_t) height, UP_DIV(inChannels, ALIGNED_CH), 1};

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    std::shared_ptr<snn::ImageTexture> img = createInputImgTxt(dims, snn::ColorFormat::RGBA32F, dest);
    img->prettyPrint();
}

void ShaderUnitTest::testImageTexture() {
    std::string imgName = snn::formatString("%sassets/images/cifar_test.jpg", ASSETS_DIR).c_str();
    auto input     = ManagedRawImage::loadFromFile(imgName);

    printf("format: %d\n", (int) input.format());
    printf("width: %d\n", input.width());
    printf("height: %d\n", input.height());
    printf("depth: %d\n", input.depth());
    printf("step: %d\n", input.step());
    printf("pitch: %d\n", input.pitch());
    printf("sliceSize: %d\n", input.sliceSize());
    printf("size: %d\n", input.size());
    printf("planeNum: %d\n", input.planes());

    printManagedRawImage(input);

    auto pixels = input.data();
    auto w      = input.width();
    auto h      = input.height();

    std::vector<snn::ImagePlaneDesc> planes;
    for (int i = 0; i < 4; i++) {
        snn::ImagePlaneDesc pd = {};
        pd.format              = snn::ColorFormat::R8;
        pd.width               = (uint32_t) w;
        pd.height              = (uint32_t) h;
        pd.depth               = (uint32_t) 1;
        pd.step                = (uint32_t) 0;
        pd.pitch               = (uint32_t) 0;
        pd.slice               = (uint32_t) 0;
        planes.push_back(pd);
    }

    snn::ManagedRawImage multiImages(ImageDesc(std::vector<snn::ImagePlaneDesc> {planes[0], planes[1], planes[2], planes[3]}), pixels);

    printf("format: %d\n", (int) multiImages.format());
    printf("width: %d\n", multiImages.width());
    printf("height: %d\n", multiImages.height());
    printf("depth: %d\n", multiImages.depth());
    printf("step: %d\n", multiImages.step());
    printf("pitch: %d\n", multiImages.pitch());
    printf("sliceSize: %d\n", multiImages.sliceSize());
    printf("size: %d\n", multiImages.size());
    printf("planeNum: %d\n", multiImages.planes());
    printManagedRawImage(multiImages);

    std::shared_ptr<snn::ImageTexture> imgTxt = snn::ImageTextureFactory::createImageTexture(context, imgName);
    snn::ImageTexture& img = *imgTxt;

    printf("format: %d\n", (int) img.format());
    printf("width: %d\n", img.width());
    printf("height: %d\n", img.height());
    printf("depth: %d\n", img.depth());
    printf("step: %d\n", img.step());
    printf("pitch: %d\n", img.pitch());
    printf("sliceSize: %d\n", img.sliceSize());
    printf("size: %d\n", img.size());
    printf("planeNum: %d\n", img.planes());
    printf("name: %s\n", img.getName().c_str());
    auto dims = img.getDims();
    printf("dims: %d, %d, %d, %d\n", dims[0], dims[1], dims[2], dims[3]);
    img.prettyPrint();

    memoryTest();

    img.upload();

    memoryTest();

    img.download();
    img.prettyPrint();

    memoryTest();

    img.convertFormat(snn::ColorFormat::RGBA32F);
    img.prettyPrint();

    memoryTest();

    img.convertFormat(snn::ColorFormat::RGBA8);
    img.prettyPrint();

    memoryTest();

    img.convertFormat(snn::ColorFormat::RGB8);
    img.prettyPrint();

    uint32_t chs  = getColorFormatDesc(img.format()).ch;
    int numPixels = img.height() * img.width() * img.depth() * chs;
    uint8_t* ret  = (uint8_t*) malloc(numPixels * sizeof(float));
    img.getCVMatData(ret);
    int size[3] = {(int) img.height(), (int) img.width(), (int) (img.depth() * chs)};
    cv::Mat inputMat(3, size, CV_32FC1, (void*) ret);
    my_print_3d_cvmat(inputMat);
    free(ret);

    img.convertFormat(snn::ColorFormat::RGBA32F);
    img.prettyPrint();

    imgTxt = snn::ImageTextureFactory::createImageTexture(context, imgName);
    snn::ImageTexture& imageRGBA8 = *imgTxt;
    imageRGBA8.convertToRGBA32FAndNormalize();
    imageRGBA8.prettyPrint();

    std::array<float, 4> resizeMeans {127.5, 127.5, 127.5, 127.5};
    std::array<float, 4> resizeNorms {1 / 127.5, 1 / 127.5, 1 / 127.5, 1 / 127.5};
    imageRGBA8.resize(2, 2, resizeMeans, resizeNorms);

    imageRGBA8.prettyPrint();
}

void writeDataToFile(std::string filePath, cv::Mat input) {
    // write Mat to file
    std::ostringstream filename2;
    filename2 << filePath + "_dump.txt";
    std::ofstream fout2;
    fout2.open(filename2.str());
    if (fout2) {
        SNN_LOGD("writeDataToFile : create a file");
    }

    for (int i = 0; i < input.size().height; i++) {
        for (int j = 0; j < input.size().width; j++) {
            fout2 << input.at<float>(i, j) << ' ';
        }
        fout2 << std::endl;
    }

    filename2.clear();
    fout2.close();
}


std::string ShaderUnitTest::snnFlattenTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, bool dumpOutput) {
    std::string ret;

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    int outWidth    = width * height * inChannels;
    int outHeight   = 1;
    int outChannels = 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    inputDesc.numInputPlanes  = inChannels;
    inputDesc.numOutputPlanes = inChannels;
    inputDesc.isInputLayer = true;
    std::shared_ptr<snn::dp::InputLayerLayer> inputLayer(new snn::dp::InputLayerLayer(std::move(inputDesc)));
    inputLayer->setName("resnet18_cifar10_0223.json layer [00] InputLayerLayer");
    inputLayer->prevLayers.clear();

    std::shared_ptr<snn::dp::GenericModelLayer> layer;

    snn::dp::FlattenDesc desc;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.preferHp        = preferrHalfPrecision;

    layer = NEW_LAYER(Flatten, desc);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->setName("resnet18_cifar10_0223.json layer [01] Flatten");

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<float> dest_vec(width * height * ROUND_UP(inChannels, ALIGNED_CH), 0.0f);
    float* dest = dest_vec.data();
    hwcToC4((float*)inputMat.data, inputMat.size[0], inputMat.size[1], inputMat.size[2], dest);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    auto inputTex = snn::InferenceGraph::IODesc {colorFormat,
                                                 (uint32_t)width, (uint32_t)height, UP_DIV(inChannels, ALIGNED_CH), 4U};
    sgo.desiredInput.push_back(inputTex);

    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = !useVulkan();
    sgo.vulkan                    = useVulkan();

    snn::MixedInferenceCore::CreationParameters graph;
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers, sgo);
    graph.dumpOutputs = dumpOutput;
    if (graph.layers.empty()) {
        return ret;
    }
    snn::ImageTextureArray imgs = createInputImgTxt(dest, width, height, inChannels);
    snn::ImageTextureArray outputTexs = createOutputImgTxt(outWidth, outHeight, outChannels);

    auto ic2 = snn::MixedInferenceCore::create(context, graph);
    snn::MixedInferenceCore::RunParameters rp = {imgs, outputTexs, {}, {}, {}};
    ic2->run(rp);

    ret = layer->getName() + " pass[" + std::to_string(0) + "].dump";
    return ret;
}
