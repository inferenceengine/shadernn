/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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
// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "layer/convolutiondepthwise.h"
#include "layer/padding.h"
#include "layer/batchnorm.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

static int test_depth_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, float inputValue = 1.0f,
                                  float padValue = 0.0f, bool useOldShader = false) {
    ncnn::ParamDict padPD;
    if (pad == 2) {
        padPD.set(0, 0);
        padPD.set(1, 0);
        padPD.set(2, 0);
        padPD.set(3, 0);
    } else {
        padPD.set(0, kernel / 2);
        padPD.set(1, kernel / 2);
        padPD.set(2, kernel / 2);
        padPD.set(3, kernel / 2);
    }
    padPD.set(4, pad);
    padPD.set(5, padValue);

    // ncnn::Mat padA = SetValueMat(w, h, c, 0.1f);
    // float* ptr  = padA.channel(0);
    // *(ptr + 10 * padA.w + 10) = 1.0f;
    ncnn::Mat padA = RandomMat(w, h, c);

    // pretty_print_ncnn(padA);
    std::vector<ncnn::Mat> padW(0);
    ncnn::Mat padB;
    test_layer_naive<ncnn::Padding>(ncnn::layer_to_index("Padding"), padPD, padW, padA, padB, (void (*)(ncnn::Padding*)) 0, 0);
    // pretty_print_ncnn(padB);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, 0);        // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch * kernel * kernel);
    pd.set(7, outch);
    // pd.set(18, 1.0f);

    int activation_type = 0; // RAND() % 7; // 0 1 2 3 4 5 6 // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.1f;              // RandomFloat(0, 1); //(activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1); // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);

    weights[0] = RandomMat(outch * kernel * kernel);
    // weights[0] = SetValueMat(outch * kernel * kernel, 1.0f);
    if (bias) {
        weights[1] = RandomMat(outch);
        // weights[1] = SetValueMat(outch, 0.0f);
    }

    ncnn::Mat ncnnOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("ConvolutionDepthWise"), pd, weights, padB, ncnnOutput, (void (*)(ncnn::ConvolutionDepthWise*)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
            fprintf(stderr, "test_depth_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d\n", w, h, c, outch, kernel,
                    dilation, stride, pad, bias);
            return -1;
        }
    }
    // pretty_print_ncnn(ncnnOutput);

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    ncnn::Mat bnOutput;
    {
        int channels = outch;
        pd.set(0, channels); // channels
        pd.set(1, 0.f);      // eps

        std::vector<ncnn::Mat> bnweights(4);
        bnweights[0] = RandomMat(channels);
        // bnweights[0] = SetValueMat(channels, 1.0f); // gamma
        bnweights[1] = RandomMat(channels);
        // bnweights[1] = SetValueMat(channels, 1.0f); // mean
        bnweights[2] = RandomMat(channels);
        bnweights[3] = RandomMat(channels);
        // bnweights[3] = SetValueMat(channels, 0.0f); // Beta

        // var must be positive
        Randomize(bnweights[2], 0.001f, 2.f);

        // bnweights[0] = SetValueMat(channels, 0.00000000001f);
        // bnweights[1] = SetValueMat(channels, 0.00000000002f);
        // bnweights[2] = SetValueMat(channels, 0.00000000003f);
        // bnweights[3] = SetValueMat(channels, 0.0000000000004f);

        ncnnToVec(bnweights[0], bnGamma);
        ncnnToVec(bnweights[1], bnMean);
        ncnnToVec(bnweights[2], bnVar);
        ncnnToVec(bnweights[3], bnBeta);

        test_layer<ncnn::BatchNorm>("BatchNorm", pd, bnweights, ncnnOutput);
        int ret = test_layer_naive(ncnn::layer_to_index("BatchNorm"), pd, bnweights, ncnnOutput, bnOutput, (void (*)(ncnn::BatchNorm*)) 0, 0);
        printf("BN Test res: %d\n", ret);
    }
    // pretty_print_ncnn(bnOutput);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    ShaderUnitTest test;

    cv::Mat inputMat = NCNNMat2CVMat(padA);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(c * outch);
    std::vector<float> inputBias      = std::vector<float>(outch, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar*) inputWeights[p].data, (uchar*) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
        // std::cout << "M = " << std::endl << " "  << inputWeights[p] << std::endl << std::endl;
    }

    if (bias) {
        const float* ptr = weights[1].channel(0);
        for (size_t p = 0; p < inputBias.size(); p++) {
            inputBias[p] = ptr[p];
        }
    }

    // std::vector<int> sz = { size[0],size[1], size[2] };
    // std::cout<< sliceMat(inputMat, 1, sz) << std::endl;
    printf("Test:%s:%d\n", __FUNCTION__, __LINE__);
    // print_3d_cvmat(inputMat);

    int ret = 0;

    bool useBN = true;
    std::map<std::string, std::vector<float>> batchNormalization;

    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnDepthConvTestWithLayer(inputMat, inputWeights, inputBias, w, h, c, outch, kernel, dilation, stride, pad, bias, useOldShader, useBN,
                                                  batchNormalization);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);
    // pretty_print_ncnn(snnOutput);

    ret = CompareMat(bnOutput, snnOutput, 0.1);
    pretty_print_ncnn(bnOutput);
    pretty_print_ncnn(snnOutput);

    printf("test_convolution test res: %d for w=%d, h=%d, c=%d, outch=%d, kernel=%d, dialation=%d, stride=%d, pad=%d, bias=%d\n", ret, w, h, c, outch, kernel,
           dilation, stride, pad, bias);

    return ret;
}

static void hwc_to_chw(const float* input, size_t fh, size_t fw, int channels, float* output) {
    /*hwc to chw*/
    uint32_t stride = fw * fh;
    for (size_t i = 0; i < stride; ++i) {
        for (size_t c = 0; c < channels; ++c) {
            output[c * stride + i] = input[i * channels + c];
        }
    }
}

int compareDepthwiseConvolution(string modelName, int layerId, string inputDump, string outputDump, int inChs, int outChs, int dim, int kernel, int padding,
                                int stride, bool force3Channels = false) {
    auto weights = getDepthwiseWeigitBiasFromNCNN(modelName, layerId);
    pretty_print_ncnn(weights[0]);
    pretty_print_ncnn(weights[1]);

    std::vector<float> chw(outChs * kernel * kernel);
    hwc_to_chw((float*) weights[0].data, kernel, kernel, outChs, chw.data());
    for (size_t p = 0; p < chw.size(); p++) {
        if (p % (kernel * kernel) == 0) {
            printf("\n");
        }
        printf("%f ", chw[p]);
    }
    printf("\n");

    auto inputNCNN = getSNNLayer(inputDump, false, inChs);
    // pretty_print_ncnn(inputNCNN);

    // inChs = 3; outChs = 1; kernel = 3;
    // dim = 224;
    // padding = 0; stride = 1;
    // inputNCNN = RandomMat(dim, dim, inChs);
    // weights[0] = RandomMat(outChs * inChs * kernel * kernel);
    // weights[0] = SetValueMat(outChs * inChs * kernel * kernel, 1.0f);

    ncnn::ParamDict padPD;

    padPD.set(0, kernel / 2);
    padPD.set(1, kernel / 2);
    padPD.set(2, kernel / 2);
    padPD.set(3, kernel / 2);

    padPD.set(4, padding);
    padPD.set(5, 0);

    // ncnn::Mat padA = SetValueMat(dim, dim, inChs, 1.0f);

    // pretty_print_ncnn(padA);
    std::vector<ncnn::Mat> padW(0);
    ncnn::Mat padB;
    test_layer_naive<ncnn::Padding>(ncnn::layer_to_index("Padding"), padPD, padW, inputNCNN, padB, (void (*)(ncnn::Padding*)) 0, 0);
    // pretty_print_ncnn(padB);

    ncnn::ParamDict pd;
    pd.set(0, outChs); // num_output
    pd.set(1, kernel); // kernel_w
    pd.set(2, 1);      // dilation_w
    pd.set(3, stride); // stride_w
    pd.set(4, 0);      // pad_w
    pd.set(5, 0);      // bias_term
    pd.set(6, outChs * kernel * kernel);
    pd.set(7, outChs);

    int activation_type = 0; // RAND() % 7; // 0 1 2 3 4 5 6 // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.1f;              // RandomFloat(0, 1); //(activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1); // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    // Hack the value of Bias
    // float* biasPtr = weights[1].channel(0);
    // for (size_t p=0; p < outChs; p++)
    // {
    //     *(biasPtr + p) = 0;
    // }
    //*(biasPtr + 40) = -1e-5;
    // printf("%s:%d: %f\n", __FUNCTION__, __LINE__, *(biasPtr + 40));
    // //*(biasPtr + 40) = (float)(*(biasPtr + 40));
    // for (size_t p=0; p < outChs; p++)
    // {
    //     if (abs(*(biasPtr + p)) < 0.0001f) {
    //         printf("%s:%d: %zu,%f\n", __FUNCTION__, __LINE__, p, *(biasPtr + p));
    //         *(biasPtr + p) = 0.0001f;
    //         printf("%s:%d: %zu,%f\n", __FUNCTION__, __LINE__, p, *(biasPtr + p));
    //     }
    // }
    // printf("%s:%d: %f\n", __FUNCTION__, __LINE__, *(biasPtr + 40));

    auto bnWeights = getBatchNormFromNCNN(modelName, layerId + 1);

    // bnWeights[0] = SetValueMat(outChs, 1.0f);
    // bnWeights[1] = SetValueMat(outChs, 0.0f);
    // bnWeights[2] = SetValueMat(outChs, 0.0001f);
    // bnWeights[3] = SetValueMat(outChs, 0.0f);

    pretty_print_ncnn(bnWeights[0]);
    pretty_print_ncnn(bnWeights[1]);
    pretty_print_ncnn(bnWeights[2]);
    pretty_print_ncnn(bnWeights[3]);

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    ncnn::ParamDict bnPD;

    bnPD.set(0, outChs); // channels
    bnPD.set(1, 0.001f); // eps, this one looks not valid

    ncnnToVec(bnWeights[0], bnGamma);
    ncnnToVec(bnWeights[1], bnMean);
    ncnnToVec(bnWeights[2], bnVar);
    ncnnToVec(bnWeights[3], bnBeta);

    ncnn::Mat ncnnOutput;
    {
        ncnn::Mat convOutput;
        int ret = test_layer_naive(ncnn::layer_to_index("ConvolutionDepthWise"), pd, weights, padB, convOutput, (void (*)(ncnn::ConvolutionDepthWise*)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
        }

        test_layer_naive(ncnn::layer_to_index("BatchNorm"), pd, bnWeights, convOutput, ncnnOutput, (void (*)(ncnn::BatchNorm*)) 0, 0);
    }
    // pretty_print_ncnn(ncnnOutput);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    ShaderUnitTest test;

    cv::Mat inputMat = NCNNMat2CVMat(inputNCNN);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(outChs);
    std::vector<float> inputBias      = std::vector<float>(outChs, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar*) inputWeights[p].data, (uchar*) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
        std::cout << "M = " << std::endl << " " << inputWeights[p] << std::endl << std::endl;
    }

    if (0) {
        const float* ptr = weights[1].channel(0);
        for (size_t p = 0; p < inputBias.size(); p++) {
            inputBias[p] = ptr[p];
        }
    }

    // print_3d_cvmat(inputMat);
    bool useBN = true;
    std::map<std::string, std::vector<float>> batchNormalization;
    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile =
        test.snnDepthConvTestWithLayer(inputMat, inputWeights, inputBias, dim, dim, inChs, outChs, kernel, 1, stride, 0, 0, false, useBN, batchNormalization);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outChs);
    // pretty_print_ncnn(snnOutput);

    auto snnDump = getSNNLayer(outputDump, false, outChs);

    int ret = CompareMat(ncnnOutput, snnOutput, 0.1);

    if (1) {
        pretty_print_ncnn(ncnnOutput);
        pretty_print_ncnn(snnOutput);
    }
    printf("-----------------------------Compare ncnn output with snn output res: %d\n", ret);

    ret = CompareMat(ncnnOutput, snnDump, 0.01);
    if (ret) {
        // pretty_print_ncnn(ncnnOutput);
        // pretty_print_ncnn(snnDump);
    }
    printf("-----------------------------Compare ncnn output with snn dump res: %d\n", ret);

    return ret;
}

int main() {
    SRAND(7767517);

    test_depth_convolution(29, 29, 1, 1, 3, 1, 2, 0, 1, 1.0, 0.0, false);

    // compareDepthwiseConvolution(formatString("%s/../../modelzoo/MobileNetV2/mobilenetv2_pretrained_imagenet", ASSETS_DIR).c_str(), 5,
    //     formatString("%s/MobileNetV2/mobilenetv2_pretrained_imagenet.json layer [02] DepthwiseConv2D pass[0]_input.dump", DUMP_DIR).c_str(),
    //     formatString("%s/MobileNetV2/mobilenetv2_pretrained_imagenet.json layer [02] DepthwiseConv2D pass[0].dump", DUMP_DIR).c_str(),
    //     32, 32, 112, 3, 0, 1, false);

    return 0;
}
