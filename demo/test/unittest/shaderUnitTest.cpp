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
#include "net.h"
#include "layer.h"

#include "ic2/dp.h"
#include "inferenceProcessor.h"
#include "shaderUnitTest.h"

#ifndef LOG_TAG
    #define LOG_TAG "ShaderUnitTest"
#endif

using namespace std;
using namespace cv;
using namespace snn;
using namespace gl;

void my_null_deleter(snn::dp::GenericModelLayer* layer) {
    (void) layer;
    return;
}

bool getOutputMat(cv::Mat& outputMat, int _kernelSize, int outputHeight, int outputWidth, int _numOutputPlanes, gl::TextureObject& _outputTexture) {
    int sf      = 1;
    int size[3] = {outputHeight * sf, outputWidth * sf, _numOutputPlanes};

    outputMat = cv::Mat(3, size, CV_32FC1);

    printf("%s:%d: %d\n", __FUNCTION__, __LINE__, sf);

    for (int opIndex = 0, numPass = 0; opIndex < _numOutputPlanes; numPass++) {
        int elementSize      = sizeof(float);
        int step             = 4;
        int stride           = step * sf * sf;
        unsigned int dataLen = outputWidth * outputHeight * stride * elementSize;

        // GL_WRAPPER(glFramebufferTextureLayer, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        //            _outputTextureContent.textureId, 0, shader.isRGBA ? opIndex / 4 : opIndex);

        _outputTexture.isArray() ? GL_WRAPPER(glFramebufferTextureLayer, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _outputTexture.id(), 0, opIndex / 4)
                                : GL_WRAPPER(glFramebufferTexture2D, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _outputTexture.target(), _outputTexture.id(), 0);

        // printf("%s:%d: %d, %d, %d, %d, idx:%d, RGBA:%d\n",__FUNCTION__, __LINE__, _numOutputPlanes, stride, elementSize, dataLen,
        // maxShaderIndex,shader.isRGBA);

        unsigned char* data = new unsigned char[dataLen];
        GL_WRAPPER(glReadPixels, 0, 0, outputWidth * sf, outputHeight * sf, GL_RGBA, GL_FLOAT, data);

        // memcpy(outputMat.data + dataLen*numPass, data, dataLen);
        int actualLen = (opIndex + step > _numOutputPlanes) ? (_numOutputPlanes % step) : step;
        actualLen     = actualLen * sf * sf;

        // printf("%s:%d: %d, %d, %d, %d, %d\n",__FUNCTION__, __LINE__, step, stride, opIndex, numPass, actualLen);

        for (int i = 0; i < size[0] * size[1]; i++) {
            float* ind = (float*) outputMat.data;
            for (int j = 0; j < actualLen; j++) {
                // printf("%s:%d: %d, %d, %d, %d, %f \n",__FUNCTION__, __LINE__, i, j, i * size[2] + opIndex + j, i * stride + j, *((float *)data + i * stride +
                // j));
                *(ind + i * size[2] + opIndex + j) = *((float*) data + i * stride + j);
            }
        }

        opIndex += 4;

        delete[] data;
    }

    // printf("---------------output of opencv mat ch first----------- \n");
    // for (int i = 0; i < outputMat.size[0]; i++) {
    //     for (int j = 0; j < outputMat.size[1]; j++) {
    //         for (int k = 0; k < outputMat.size[2]; k++) {
    //             //std::cout << "M(" << i << ", " << j << ", " << k << "): " << outputMat.at<float>(i,j,k) << ",";
    //             std::cout  << std::setw(7) << outputMat.at<float>(i,j,k) <<  ",";
    //         }
    //         //std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
    //     }
    //     std::cout  << std::endl;
    //     //std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    // }
    // std::cout << std::endl;

    printf("Test:%s:%d, %d, %d, %d\n", __FUNCTION__, __LINE__, outputMat.size[0], outputMat.size[1], outputMat.size[2]);

    return true;
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
            (*dst).green   = in.at<float>(i, j, 1);
            (*dst).blue    = in.at<float>(i, j, 2);
            (*dst).alpha   = in.at<float>(i, j, 3);

            // std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
        }
        // std::cout  << std::endl;
        // std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }

    return result;
}

std::string ShaderUnitTest::snnConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width, int height,
                                                 int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, int bias, bool useOldShader,
                                                 bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization) {
    std::string ret = "";
    vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    printf("%%%%%%%% %s:%d: width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useOldShader:%d \n",
           __FUNCTION__, __LINE__, width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useOldShader);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
    inputLayer->prevLayers.clear();

    snn::dp::Conv2DDesc desc;
    desc.isRange01       = 0;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.weights         = inputWeights;
    desc.biases          = doubleBias;
    // desc.activation = "leakyRelu";
    // desc.leakyReluAlpha = 0.1;
    // desc.activation = "relu";
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

    auto layer = std::shared_ptr<snn::dp::Conv2DLayer>(new snn::dp::Conv2DLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] Conv2D";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    if (preferrHalfPrecision) {
        uint16_t* inputData = (uint16_t*) inputMat.data;
        // Change OpenCV HWC to C/4HW4
        inputData = (uint16_t*) dest;
        for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
            for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
                if (p + 0 < inputMat.size[2]) {
                    *(inputData + i * 4 + 0) = FP32::toHalf(*((float*) inputMat.data + i * inputMat.size[2] + p + 0));
                }
                if (p + 1 < inputMat.size[2]) {
                    *(inputData + i * 4 + 1) = FP32::toHalf(*((float*) inputMat.data + i * inputMat.size[2] + p + 1));
                }
                if (p + 2 < inputMat.size[2]) {
                    *(inputData + i * 4 + 2) = FP32::toHalf(*((float*) inputMat.data + i * inputMat.size[2] + p + 2));
                }
                if (p + 3 < inputMat.size[2]) {
                    *(inputData + i * 4 + 3) = FP32::toHalf(*((float*) inputMat.data + i * inputMat.size[2] + p + 3));
                }
            }
            inputData += inputMat.size[0] * inputMat.size[1] * 2;
            // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
        }
    } else {
        float* inputData = (float*) inputMat.data;
        // Change OpenCV HWC to C/4HW4
        inputData = dest;
        for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
            for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
                if (p + 0 < inputMat.size[2]) {
                    *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
                }
                if (p + 1 < inputMat.size[2]) {
                    *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
                }
                if (p + 2 < inputMat.size[2]) {
                    *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
                }
                if (p + 3 < inputMat.size[2]) {
                    *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
                }
            }
            inputData += inputMat.size[0] * inputMat.size[1] * 4;
            // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
        }
    }
    snn::ImageTexture img(dims, colorFormat, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.mrtMode                   = snn::MRTMode::SINGLE_PLANE;
    sgo.compute                   = false;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    ret = layer->name + " pass[" + to_string((outChannels + 3) / 4 - 1) + "].dump";
    //    ret = layer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnPoolingTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, int poolingType,
                                                    int padMode) {
    std::string ret = "";

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
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
        desc.paddingB = "same";
        desc.paddingL = "same";
        desc.paddingR = "same";
        desc.preferHp = preferrHalfPrecision;
        desc.mrtMode  = snn::MRTMode::SINGLE_PLANE;
        layer         = std::shared_ptr<snn::dp::MaxPooling2DLayer>(new snn::dp::MaxPooling2DLayer(std::move(desc)), &my_null_deleter);
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
        desc.preferHp = preferrHalfPrecision;
        desc.mrtMode  = snn::MRTMode::SINGLE_PLANE;
        layer         = std::shared_ptr<snn::dp::AveragePooling2DLayer>(new snn::dp::AveragePooling2DLayer(std::move(desc)), &my_null_deleter);
    }

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] Pooling";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = false;

    sgo.mrtMode = snn::MRTMode::SINGLE_PLANE;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = layer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = layer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnAddTestWithLayer2(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, string activation) {
    std::string ret = "";

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
    inputLayer->prevLayers.clear();

    snn::dp::MaxPooling2DDesc desc;
    // desc.inputHeight = width;
    // desc.inputWidth = height;
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

    auto layer = std::shared_ptr<snn::dp::MaxPooling2DLayer>(new snn::dp::MaxPooling2DLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] MaxPooling";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    snn::dp::AddDesc addDesc;
    addDesc.numOutputPlanes = outChannels;
    addDesc.numInputPlanes  = inChannels;
    addDesc.activation      = activation;
    // addDesc.inputHeight = width;
    // addDesc.inputWidth = height;
    addDesc.preferHp = preferrHalfPrecision;
    addDesc.mrtMode  = snn::MRTMode::SINGLE_PLANE;

    auto addLayer = std::shared_ptr<snn::dp::AddLayer>(new snn::dp::AddLayer(std::move(addDesc)), &my_null_deleter);

    addLayer->prevLayers.push_back(inputLayer);
    addLayer->prevLayers.push_back(layer);
    addLayer->nextLayers.clear();
    addLayer->name = "resnet18_cifar10_0223.json layer [01] Add";

    // Do we need these two?
    layer->nextLayers.push_back(addLayer);
    inputLayer->nextLayers.push_back(addLayer);

    layers.emplace_back(addLayer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.mrtMode                   = snn::MRTMode::SINGLE_PLANE;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = layer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = addLayer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

static void print_3d_cvmat(cv::Mat outputMat) {
    printf("---------------output of opencv 3d mat w first----------- \n");
    for (int k = 0; k < outputMat.size[2]; k++) {
        for (int i = 0; i < outputMat.size[0]; i++) {
            for (int j = 0; j < outputMat.size[1]; j++) {
                // std::cout << "M(" << i << ", " << j << ", " << k << "): " << outputMat.at<float>(i,j,k) << ",";
                std::cout << std::setw(7) << outputMat.at<float>(i, j, k) << ",";
            }
            // std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
            std::cout << std::endl;
        }
        std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }
    std::cout << std::endl;
}

std::string ShaderUnitTest::snnUpsampleTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int scale, int type) {
    std::string ret = "";

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int outWidth    = (width) *scale;
    int outHeight   = (height) *scale;
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight     = width;
    inputDesc.inputWidth      = height;
    inputDesc.inputChannels   = inChannels;
    inputDesc.numOutputPlanes = outChannels;
    auto inputLayer           = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
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

    layer = std::shared_ptr<snn::dp::UpSampling2DLayer>(new snn::dp::UpSampling2DLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] Upsample";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = true;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = layer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = layer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnConcateTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, string activation) {
    std::string ret = "";

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int outWidth    = width;
    int outHeight   = height;
    int outChannels = 2 * inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
    inputLayer->prevLayers.clear();

    snn::dp::MaxPooling2DDesc maxPoolDesc;
    // maxPoolDesc.inputHeight = width;
    // maxPoolDesc.inputWidth = height;
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

    auto maxPoolLayer = std::shared_ptr<snn::dp::MaxPooling2DLayer>(new snn::dp::MaxPooling2DLayer(std::move(maxPoolDesc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    maxPoolLayer->prevLayers.push_back(inputLayer);
    maxPoolLayer->nextLayers.clear();
    maxPoolLayer->name = "resnet18_cifar10_0223.json actLayer [01] MaxPooling";

    inputLayer->nextLayers.push_back(maxPoolLayer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(maxPoolLayer);

    snn::dp::ConcatenateDesc concatDesc0;
    concatDesc0.numOutputPlanes = 2 * inChannels;
    concatDesc0.numInputPlanes  = inChannels;
    concatDesc0.preferHp        = preferrHalfPrecision;
    concatDesc0.mrtMode         = snn::MRTMode::SINGLE_PLANE;

    printf("%s:%d; numInputPlanes: %d; numOutputPlanes: %d;\n", __FUNCTION__, __LINE__, concatDesc0.numInputPlanes, concatDesc0.numOutputPlanes);

    auto concatLayer0 = std::shared_ptr<snn::dp::ConcatenateLayer>(new snn::dp::ConcatenateLayer(std::move(concatDesc0)), &my_null_deleter);

    concatLayer0->prevLayers.push_back(inputLayer);
    concatLayer0->prevLayers.push_back(maxPoolLayer);
    concatLayer0->nextLayers.clear();
    concatLayer0->name = "resnet18_cifar10_0223.json actLayer [02] Concate0";

    // Do we need these two?
    maxPoolLayer->nextLayers.push_back(concatLayer0);
    inputLayer->nextLayers.push_back(concatLayer0);

    layers.emplace_back(concatLayer0);

    snn::dp::ConcatenateDesc concatDesc1;
    concatDesc1.numOutputPlanes = (2 + 1) * inChannels;
    //    concatDesc1.numInputPlanes = 2 * inChannels;
    concatDesc1.preferHp = preferrHalfPrecision;
    concatDesc1.mrtMode  = snn::MRTMode::SINGLE_PLANE;

    auto concatLayer1 = std::shared_ptr<snn::dp::ConcatenateLayer>(new snn::dp::ConcatenateLayer(std::move(concatDesc1)), &my_null_deleter);

    concatLayer1->prevLayers.push_back(inputLayer);
    concatLayer1->prevLayers.push_back(concatLayer0);
    concatLayer1->nextLayers.clear();
    concatLayer1->name = "resnet18_cifar10_0223.json actLayer [02] Concate1";

    // Do we need these two?
    inputLayer->nextLayers.push_back(concatLayer1);
    concatLayer0->nextLayers.push_back(concatLayer1);

    layers.emplace_back(concatLayer1);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.mrtMode                   = snn::MRTMode::SINGLE_PLANE;
    sgo.compute                   = true;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = maxPoolLayer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = concatLayer1->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnDepthConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width,
                                                      int height, int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, int bias,
                                                      bool useOldShader, bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization) {
    std::string ret = "";
    vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    printf("%%%%%%%% %s:%d: width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useOldShader:%d \n",
           __FUNCTION__, __LINE__, width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useOldShader);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
    inputLayer->prevLayers.clear();

    snn::dp::SeparableConv2DDesc desc;
    desc.isRange01       = 0;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.weights         = inputWeights;
    desc.biases          = doubleBias;
    // desc.activation = "leakyRelu";
    // desc.leakyReluAlpha = 0.1;
    // desc.activation = "relu";
    desc.activation            = "";
    desc.kernelSize            = kernel;
    desc.stride                = stride;
    desc.useBatchNormalization = useBatchNorm;
    desc.batchNormalization    = batchNormalization;
    desc.padding               = "same";
    desc.paddingT              = "same";
    desc.paddingB              = "same";
    desc.paddingL              = "same";
    desc.paddingR              = "same";
    desc.preferHp              = preferrHalfPrecision;

    auto layer = std::shared_ptr<snn::dp::SeparableConv2DLayer>(new snn::dp::SeparableConv2DLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] DepthConv2D";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = true;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = layer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = layer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnInstanceNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width,
                                                         int height, int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, int bias,
                                                         bool useOldShader, bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization) {
    std::string ret = "";
    vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    printf("%%%%%%%% %s:%d: width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useOldShader:%d \n",
           __FUNCTION__, __LINE__, width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useOldShader);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
    inputLayer->prevLayers.clear();

    snn::dp::InstanceNormDesc desc;
    desc.isRange01       = 0;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.weights         = inputWeights;
    desc.biases          = doubleBias;
    // desc.activation = "leakyRelu";
    // desc.leakyReluAlpha = 0.1;
    // desc.activation = "relu";
    desc.activation               = "";
    desc.kernelSize               = kernel;
    desc.stride                   = stride;
    desc.useInstanceNormalization = useBatchNorm;
    desc.instanceNormalization    = batchNormalization;
    desc.padding                  = "same";
    // desc.paddingT = "same";
    // desc.paddingB = "same";
    // desc.paddingL = "same";
    // desc.paddingR = "same";
    desc.preferHp = preferrHalfPrecision;

    auto layer = std::shared_ptr<snn::dp::InstanceNormLayer>(new snn::dp::InstanceNormLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] InstanceNorm";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = true;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = layer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = layer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnPadTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, int type, float value) {
    std::string ret = "";

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;
    int outChannels = inChannels;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
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
    layer = std::shared_ptr<snn::dp::PadLayer>(new snn::dp::PadLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] Padding";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;

    sgo.mrtMode = snn::MRTMode::SINGLE_PLANE;
    sgo.compute = true;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = layer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = layer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnBatchNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int width,
                                                      int height, int inChannels, int outChannels, int kernel, int dilation, int stride, int pad, int bias,
                                                      bool useOldShader, bool useBatchNorm, std::map<std::string, std::vector<float>>& batchNormalization) {
    std::string ret = "";
    vector<double> doubleBias(inputBias.size(), 0);
    std::transform(inputBias.begin(), inputBias.end(), doubleBias.begin(), [](float x) { return (double) x; });

    printf("%%%%%%%% %s:%d: width:%d, height:%d, inChannels:%d, outChannels:%d, kernel:%d, dilation:%d, stride:%d, pad:%d, bias:%d, useOldShader:%d \n",
           __FUNCTION__, __LINE__, width, height, inChannels, outChannels, kernel, dilation, stride, pad, bias, useOldShader);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;

    // Create single layer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
    inputLayer->prevLayers.clear();

    snn::dp::BatchNormalizationDesc desc;
    desc.isRange01          = 0;
    desc.numOutputPlanes    = outChannels;
    desc.numInputPlanes     = inChannels;
    desc.weights            = inputWeights;
    desc.biases             = doubleBias;
    desc.batchNormalization = batchNormalization;
    desc.activation         = "";
    desc.kernelSize         = kernel;
    desc.stride             = stride;
    desc.preferHp           = preferrHalfPrecision;

    auto layer = std::shared_ptr<snn::dp::BatchNormalizationLayer>(new snn::dp::BatchNormalizationLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    layer->prevLayers.push_back(inputLayer);
    layer->nextLayers.clear();
    layer->name = "resnet18_cifar10_0223.json layer [01] BatchNorm";

    inputLayer->nextLayers.push_back(layer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(layer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.compute                   = true;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = layer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = layer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
    return ret;
}

std::string ShaderUnitTest::snnActivationTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, string activation,
                                                       float leaky_val) {
    std::string ret = "";

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    bool preferrHalfPrecision = false;
    auto colorFormat          = preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F;

    int paddingSize = (int) std::floor(kernel / 2);
    int outWidth    = (width - kernel + 2 * paddingSize) / stride + 1;
    int outHeight   = (height - kernel + 2 * paddingSize) / stride + 1;
    int outChannels = inChannels;

    // Create single actLayer from Layer class
    snn::dp::InputLayerDesc inputDesc;
    inputDesc.inputHeight   = width;
    inputDesc.inputWidth    = height;
    inputDesc.inputChannels = inChannels;
    auto inputLayer         = std::shared_ptr<snn::dp::InputLayerLayer>(new snn::dp::InputLayerLayer(std::move(inputDesc)), &my_null_deleter);
    inputLayer->prevLayers.clear();

    snn::dp::ActivationDesc desc;
    // desc.inputHeight = width;
    // desc.inputWidth = height;
    desc.numOutputPlanes = outChannels;
    desc.numInputPlanes  = inChannels;
    desc.activation      = activation;
    desc.kernelSize      = kernel;
    desc.preferHp        = preferrHalfPrecision;
    desc.mrtMode         = snn::MRTMode::SINGLE_PLANE;
    desc.leakyReluAlpha  = leaky_val;

    auto actLayer = std::shared_ptr<snn::dp::ActivationLayer>(new snn::dp::ActivationLayer(std::move(desc)), &my_null_deleter);

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    actLayer->prevLayers.push_back(inputLayer);
    actLayer->nextLayers.clear();
    actLayer->name = "resnet18_cifar10_0223.json actLayer [01] Activation";

    inputLayer->nextLayers.push_back(actLayer);
    inputLayer->prevLayers.clear();
    layers.emplace_back(inputLayer);
    layers.emplace_back(actLayer);

    std::vector<uint32_t> dims {(uint32_t) width, (uint32_t) height, (uint32_t)(inChannels + 3) / 4, 1};
    float* dest = (float*) malloc(width * height * (inChannels + 3) / 4 * 4 * 4);
    memset(dest, 0, width * height * (inChannels + 3) / 4 * 4 * 4);

    float* inputData = (float*) inputMat.data;
    // Change OpenCV HWC to C/4HW4
    inputData = dest;
    for (int p = 0; p < (inChannels + 3) / 4 * 4; p += 4) {
        for (int i = 0; i < inputMat.size[0] * inputMat.size[1]; i++) {
            if (p + 0 < inputMat.size[2]) {
                *(inputData + i * 4 + 0) = *((float*) inputMat.data + i * inputMat.size[2] + p + 0);
            }
            if (p + 1 < inputMat.size[2]) {
                *(inputData + i * 4 + 1) = *((float*) inputMat.data + i * inputMat.size[2] + p + 1);
            }
            if (p + 2 < inputMat.size[2]) {
                *(inputData + i * 4 + 2) = *((float*) inputMat.data + i * inputMat.size[2] + p + 2);
            }
            if (p + 3 < inputMat.size[2]) {
                *(inputData + i * 4 + 3) = *((float*) inputMat.data + i * inputMat.size[2] + p + 3);
            }
        }
        inputData += inputMat.size[0] * inputMat.size[1] * 4;
        // printf("Test:%s:%d, %d <-> %d\n",__FUNCTION__,__LINE__, p, inputMat.size[2]);
    }

    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    // img.loadCVMatData((uint8_t*)dest);

    img.upload();
    // readTexture(width*height*UP_DIV(inChannels, 4), img.texture(0)->id(), width, height, UP_DIV(inChannels,4), 1);

    // setup options
    snn::dp::ShaderGenOptions sgo = {};
    sgo.desiredInput.width        = width;
    sgo.desiredInput.height       = height;
    sgo.desiredInput.depth        = (inChannels + 3) / 4;
    sgo.desiredInput.format       = colorFormat;
    sgo.desiredOutputFormat       = colorFormat;
    sgo.preferrHalfPrecision      = preferrHalfPrecision;
    sgo.mrtMode                   = snn::MRTMode::SINGLE_PLANE;
    sgo.compute                   = true;

    // generate graph
    snn::MixedInferenceCore::CreationParameters graph;
    graph.dumpOutputs = true;

    // dp.erase(dp.begin()+2, dp.end());
    // dp[1]->nextLayers.clear();
    // printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__,__LINE__, dp.size());

    // (snn::InferenceGraph&&)graph = snn::dp::generateInferenceGraph(dp.at(0), sgo);
    (snn::InferenceGraph &&) graph = snn::dp::generateInferenceGraph(layers.at(0), sgo);
    printf("%%%%%%%% %s:%d :%zu\n", __FUNCTION__, __LINE__, graph.layers.size());

    if (graph.layers.empty()) {
        return ret;
    }
    // create ic2
    auto ic2 = MixedInferenceCore::create(graph);

    // generate output texture
    auto outdesc = graph.layers.back()->output;
    (void) outdesc;
    // SNN_CHK(1 == outdesc.depth);
    gl::TextureObject outTexture;
    if (outChannels <= 4) {
        outTexture.allocate2D(colorFormat, outWidth, outHeight);
    } else {
        outTexture.allocate2DArray(colorFormat, outWidth, outHeight, (outChannels + 3) / 4);
    }

    // const gl::TextureObject* inputs[] = { &inputTexture };
    const gl::TextureObject* inputs[] = {img.texture(0)};
    auto outVec                       = std::vector<std::vector<std::vector<float>>>();
    auto inVec                        = std::vector<std::vector<std::vector<float>>>();
    InferenceEngine::SNNModelOutput modelOutput;
    MixedInferenceCore::RunParameters rp = {inputs, &outTexture, 1, inVec, outVec, modelOutput};

    ic2->run(rp);

    rc.swapBuffers();

    // getOutputMat(ret, kernel, outHeight, outWidth, outChannels, outTexture);
    // ret = actLayer->name + " pass[" + to_string((outChannels+3)/4-1) + "].dump";
    ret = actLayer->name + " pass[" + to_string(0) + "].dump";

    glFinish();

    free(dest);
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
            // int my       = d % unit;
            // int d_4      = d / unit;
            for (int y = 0; y < fh; ++y) {
                for (int x = 0; x < fw; ++x) {
                    int base                                 = (y * fw + x) * planeSize;
                    int inSize                               = ROUND_UP(inChannels, unit) * unit;
                    out[base + inSize * b_4 + d * unit + mx] = inputWeights[b * inChannels + d].at<float>(y * fw + x);
                    // printf("new: %d: %f\n",base + inSize * b_4 + d * unit + mx, inputWeights[b*inChannels+d].at<float>(y*fw+x));
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

        printf("%s:%d, Dim: %d, %d, %d, actual channels: %d\n", __FUNCTION__, __LINE__, width, height, (actualChanels + 3) / 4 * 4, actualChanels);
        int idx = 0;
        for (int p4 = 0; p4 < (size[2] + 3) / 4; p4++) {
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
                    if (img.format() == ColorFormat::RGBA8) {
                        auto v = (int) *(img.at(p, i, j, k));
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(img.at(p, i, j, k) + 1);
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(img.at(p, i, j, k) + 2);
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(img.at(p, i, j, k) + 3);
                        std::cout << std::setw(4) << v << ",";
                    } else if (img.format() == ColorFormat::RGBA32F) {
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
                // std::cout << "M(" << i << ", " << j << ", " << k << "): " << outputMat.at<float>(i,j,k) << ",";
                std::cout << std::setw(7) << outputMat.at<float>(i, j, k) << ",";
            }
            // std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
            std::cout << std::endl;
        }
        std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }
    std::cout << std::endl;
}

void ShaderUnitTest::testImageTexture() {
    // load input image
    // auto input = ManagedRawImage::loadFromAsset("images/cifar_test.png");
    string imgName = "../../../../core/data/assets/images/cifar_test.png";
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
    printf("%s:%d\n", __FUNCTION__, __LINE__);

    printManagedRawImage(input);

    auto pixels = input.data();
    auto w      = input.width();
    auto h      = input.height();

    std::vector<snn::ImagePlaneDesc> planes;
    for (int i = 0; i < 4; i++) {
        snn::ImagePlaneDesc pd = {};
        pd.format              = ColorFormat::R8;
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
    printf("%s:%d\n", __FUNCTION__, __LINE__);
    printManagedRawImage(multiImages);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();

    snn::ImageTexture img(imgName);

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
    printf("%s:%d\n", __FUNCTION__, __LINE__);
    img.printOut();

    memoryTest();

    img.upload();
    printf("%s:%d\n", __FUNCTION__, __LINE__);

    memoryTest();

    img.download();
    printf("%s:%d\n", __FUNCTION__, __LINE__);
    img.printOut();

    memoryTest();

    img.convertFormat(ColorFormat::RGBA32F, 0.0f, 1.0f);
    printf("%s:%d\n", __FUNCTION__, __LINE__);
    img.printOut();

    memoryTest();

    img.convertFormat(ColorFormat::RGBA8);
    printf("%s:%d\n", __FUNCTION__, __LINE__);
    img.printOut();

    memoryTest();

    img.convertFormat(ColorFormat::RGB8);
    printf("%s:%d\n", __FUNCTION__, __LINE__);
    img.printOut();

    printf("%s:%d\n", __FUNCTION__, __LINE__);
    uint32_t chs  = getColorFormatDesc(img.format()).ch;
    int numPixels = img.height() * img.width() * img.depth() * chs;
    uint8_t* ret  = (uint8_t*) malloc(numPixels * sizeof(float));
    img.getCVMatData(ret);
    int size[3] = {(int) img.height(), (int) img.width(), (int) (img.depth() * chs)};
    cv::Mat inputMat(3, size, CV_32FC1, (void*) ret);
    my_print_3d_cvmat(inputMat);
    free(ret);

    printf("%s:%d\n", __FUNCTION__, __LINE__);
    img.convertFormat(ColorFormat::RGBA32F);
    img.printOut();
    // {
    //     printf("%s:%d\n", __FUNCTION__,__LINE__);
    //     uint32_t chs = getColorFormatDesc(img.format()).ch;
    //     int numPixels = img.height() * img.width() * img.depth() * chs;
    //     uint8_t* ret = (uint8_t *) malloc(numPixels * sizeof(float));
    //     img.getCVMatData(ret);
    //     int size[3] = { (int)img.height(), (int)img.width(), (int)(img.depth() * chs) };
    //     cv::Mat inputMat(3, size, CV_32FC1, (void*)ret);
    //     my_print_3d_cvmat(inputMat);

    //     printf("%s:%d\n", __FUNCTION__,__LINE__);
    //     img.loadCVMatData(ret);
    //     img.printOut();
    //     free(ret);
    // }

    // gl::TextureObject gpuTex;
    // auto imgInput = ManagedRawImage::loadFromAsset("images/cifar_test.png");
    // auto input32f = snn::toRgba32f(imgInput);
    // gpuTex.allocate2DArray(input32f.format(), input32f.width(), input32f.height(), input32f.depth());
    // gpuTex.setPixels(
    //         0, 0,
    //         0, 0,
    //         input32f.width(), input32f.height(),
    //         0,
    //         input32f.data()
    //     );
    // auto texOut = gpuTex.getBaseLevelPixels();
    // printManagedRawImage(texOut);

    // snn::ImageTexture tex(gpuTex);
    // tex.printOut();

    snn::ImageTexture imageRGBA8(imgName);
    // std::vector<float> means{127.5, 127.5, 127.5, 127.5};
    // std::vector<float> norms{1/127.5, 1/127.5, 1/127.5, 1/127.5};
    std::vector<float> means {0, 0, 0, 0};
    std::vector<float> norms {1, 1, 1, 1};
    imageRGBA8.convertFormat(ColorFormat::RGBA32F, means, norms);
    printf("%s:%d\n", __FUNCTION__, __LINE__);
    imageRGBA8.printOutWH();

    // imageRGBA8.upload();
    std::vector<float> resizeMeans {127.5, 127.5, 127.5, 127.5};
    std::vector<float> resizeNorms {1 / 127.5, 1 / 127.5, 1 / 127.5, 1 / 127.5};
    imageRGBA8.resize(2, 2, resizeMeans, resizeNorms);

    imageRGBA8.printOutWH();

    glFinish();
}

void writeDataToFile(std::string filePath, cv::Mat input) {
    // write Mat to file
    std::ostringstream filename2;
    filename2 << filePath + "_dump.txt";
    ofstream fout2;
    fout2.open(filename2.str());
    if (fout2) {
        SNN_LOGD("writeDataToFile : create a file");
    }

    for (int i = 0; i < input.size().height; i++) {
        for (int j = 0; j < input.size().width; j++) {
            fout2 << input.at<float>(i, j) << ' ';
        }
        fout2 << endl;
    }

    filename2.clear();
    fout2.close();
}
