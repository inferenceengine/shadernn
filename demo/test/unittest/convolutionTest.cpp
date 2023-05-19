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

#include "layer/convolution.h"
#include "layer/padding.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
    #undef Success
#endif
#include "CLI/CLI.hpp"

static int test_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, float padValue, bool useCompute,
                            snn::MRTMode mrtMode, snn::GpuBackendType backend, bool fp16, bool printMismatch) {
    ncnn::ParamDict padPD;

    padPD.set(0, kernel / 2);
    padPD.set(1, kernel / 2);
    padPD.set(2, kernel / 2);
    padPD.set(3, kernel / 2);

    padPD.set(4, pad);
    padPD.set(5, padValue);

    ncnn::Mat padA = RandomMat(w, h, c);
    padA           = SetValueMat(w, h, c, 1.0f);

    std::vector<ncnn::Mat> padW(0);
    ncnn::Mat              padB;
    test_layer_naive<ncnn::Padding>(ncnn::layer_to_index("Padding"), padPD, padW, padA, padB, (void (*)(ncnn::Padding *)) 0, 0);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, 0);        // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch * c * kernel * kernel);

    int       activation_type = 0; // RAND() % 7; // 0 1 2 3 4 5 6 // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.1f;              // RandomFloat(0, 1); //(activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1); // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);

    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias) {
        weights[1] = RandomMat(outch);
        weights[1] = SetValueMat(outch, 0.0f);
    }

    ncnn::Mat ncnnOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("Convolution"), pd, weights, padB, ncnnOutput, (void (*)(ncnn::Convolution *)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d\n", w, h, c, outch, kernel,
                    dilation, stride, pad, bias);
            return -1;
        }
    }

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
        bnweights[0] = SetValueMat(channels, 1.0f); // gamma
        bnweights[1] = RandomMat(channels);
        bnweights[1] = SetValueMat(channels, 0.0f); // mean
        bnweights[2] = RandomMat(channels);
        bnweights[2] = SetValueMat(channels, 1.0f); // variance
        bnweights[3] = RandomMat(channels);
        bnweights[3] = SetValueMat(channels, 0.0f); // Beta

        ncnnToVec(bnweights[0], bnGamma);
        ncnnToVec(bnweights[1], bnMean);
        ncnnToVec(bnweights[2], bnVar);
        ncnnToVec(bnweights[3], bnBeta);

        test_layer<ncnn::BatchNorm>("BatchNorm", pd, bnweights, ncnnOutput);
        int ret = test_layer_naive(ncnn::layer_to_index("BatchNorm"), pd, bnweights, ncnnOutput, bnOutput, (void (*)(ncnn::BatchNorm *)) 0, 0);
        printf("BN Test res: %d\n", ret);
    }

    ShaderUnitTest test(backend);
    cv::Mat        inputMat = NCNNMat2CVMat(padA);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(c * outch);
    std::vector<float>   inputBias    = std::vector<float>(outch, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar *) inputWeights[p].data, (uchar *) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }

    if (bias) {
        const float * ptr = weights[1].channel(0);
        for (size_t p = 0; p < inputBias.size(); p++) { inputBias[p] = ptr[p]; }
    }

    int ret = 0;

    bool                                      useBN = true;
    std::map<std::string, std::vector<float>> batchNormalization;

    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, w, h, c, outch, kernel, dilation, stride, pad, useCompute, mrtMode, useBN,
                                             batchNormalization, true, fp16);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);

    ret = CompareMat(bnOutput, snnOutput, 0.01);
    printf("convolution test res: %d for w=%d, h=%d, c=%d, outch=%d, kernel=%d, dialation=%d, stride=%d, pad=%d, bias=%d\n", ret, w, h, c, outch, kernel,
           dilation, stride, pad, bias);
    if (ret && printMismatch) {
        pretty_print_ncnn(bnOutput);
        pretty_print_ncnn(snnOutput, "SNN");
    }

    return ret;
}

#define SNN_MODEL_NAME "resnet18_cifar10_0223.json"
void debug_conv2d_layer4(snn::GpuBackendType backend) {
    auto convWeights = getWeigitBiasFromNCNN(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(), 7);
    pretty_print_ncnn(convWeights[1]);

    auto bnWeights = getBatchNormFromNCNN(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(), 8);
    pretty_print_ncnn(bnWeights[0]);
    pretty_print_ncnn(bnWeights[1]);
    pretty_print_ncnn(bnWeights[2]);
    pretty_print_ncnn(bnWeights[3]);

#if FRAGMENT_SHADER
    auto snnMat = getSNNLayer(formatString("%s/%s layer [04] Conv2D pass[15]_input.dump", DUMP_DIR, SNN_MODEL_NAME).c_str());
#else
    auto snnMat = getSNNLayer(formatString("%s/%s layer [04] Conv2D pass[0]_input.dump", DUMP_DIR, SNN_MODEL_NAME).c_str());
#endif

    int inChs    = 64;
    int outChs   = 64;
    int kernel   = 3;
    int width    = 8;
    int height   = 8;
    int dilation = 1, stride = 1, pad = kernel / 2, bias = true, useCompute = false;

    ncnn::ParamDict pd;
    pd.set(0, outChs);     // num_output
    pd.set(1, kernel);     // kernel_w
    pd.set(2, 1);          // dilation_w
    pd.set(3, 1);          // stride_w
    pd.set(4, kernel / 2); // pad_w
    pd.set(5, 1);          // bias_term
    pd.set(6, outChs * inChs * kernel * kernel);

    ncnn::Mat ncnnOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("Convolution"), pd, convWeights, snnMat, ncnnOutput, (void (*)(ncnn::Convolution *)) 0, 0);
        if (ret != 0) { fprintf(stderr, "test_layer_naive failed\n"); }
    }
    auto ncnnMat = getNCNNLayer(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(),
                                formatString("%s/assets/images/cifar_test.png", ASSETS_DIR).c_str(), "conv2d_2_blob", 32);

    int ret = CompareMat(ncnnOutput, ncnnMat, 0.1);
    printf("test_convolution test res: %d \n", ret);

    ShaderUnitTest test(backend);

    cv::Mat inputMat = NCNNMat2CVMat(snnMat);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar *) inputWeights[p].data, (uchar *) convWeights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }
    std::vector<float> inputBias;
    ncnnToVec(convWeights[1], inputBias);

    bool                                      useBN = true;
    std::map<std::string, std::vector<float>> batchNormalization;

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    ncnnToVec(bnWeights[0], bnGamma);
    ncnnToVec(bnWeights[1], bnMean);
    ncnnToVec(bnWeights[2], bnVar);
    ncnnToVec(bnWeights[3], bnBeta);

    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile   = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, width, height, inChs, outChs, kernel, dilation, stride, pad, useCompute,
                                             snn::MRTMode::SINGLE_PLANE, useBN, batchNormalization);
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outChs);

    ncnnMat = getNCNNLayer(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(),
                           formatString("%s/assets/images/cifar_test.png", ASSETS_DIR).c_str(), "batch_normalization_1_blob", 32);
    ret     = CompareMat(ncnnMat, snnOutput, 0.1);
    printf("test_convolution test res: %d for w=%d, h=%d, c=%d, outch=%d, kernel=%d, dialation=%d, stride=%d, pad=%d, bias=%d\n", ret, width, height, inChs,
           outChs, kernel, dilation, stride, pad, bias);
}

void debug_conv2d_layer3(snn::GpuBackendType backend) {
    auto convWeights = getWeigitBiasFromNCNN(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(), 6);
    pretty_print_ncnn(convWeights[1]);

    auto snnMat = getSNNLayer(formatString("%s/%s layer [03] Conv2D pass[0]_input.dump", DUMP_DIR, SNN_MODEL_NAME).c_str());

    int inChs    = 64;
    int outChs   = 64;
    int kernel   = 3;
    int width    = 8;
    int height   = 8;
    int dilation = 1, stride = 1, pad = kernel / 2, bias = true, useCompute = false;

    ncnn::ParamDict pd;
    pd.set(0, outChs);     // num_output
    pd.set(1, kernel);     // kernel_w
    pd.set(2, 1);          // dilation_w
    pd.set(3, 1);          // stride_w
    pd.set(4, kernel / 2); // pad_w
    pd.set(5, 1);          // bias_term
    pd.set(6, outChs * inChs * kernel * kernel);

    ncnn::Mat ncnnOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("Convolution"), pd, convWeights, snnMat, ncnnOutput, (void (*)(ncnn::Convolution *)) 0, 0);
        if (ret != 0) { fprintf(stderr, "test_layer_naive failed\n"); }
    }
    auto ncnnMat = getNCNNLayer(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(),
                                formatString("%s/assets/images/cifar_test.png", ASSETS_DIR).c_str(), "max_pooling2d_blob", 32);

    int ret = CompareMat(ncnnOutput, ncnnMat, 0.1);
    printf("test_convolution test res: %d \n", ret);

    ShaderUnitTest test(backend);

    cv::Mat inputMat = NCNNMat2CVMat(snnMat);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar *) inputWeights[p].data, (uchar *) convWeights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }
    std::vector<float> inputBias;
    ncnnToVec(convWeights[1], inputBias);

    bool                                      useBN = false;
    std::map<std::string, std::vector<float>> batchNormalization;

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    auto outFile   = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, width, height, inChs, outChs, kernel, dilation, stride, pad, useCompute,
                                             snn::MRTMode::SINGLE_PLANE, useBN, batchNormalization);
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outChs);
    ncnnMat        = getNCNNLayer(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(),
                           formatString("%s/assets/images/cifar_test.png", ASSETS_DIR).c_str(), "conv2d_1_blob", 32);
    ret            = CompareMat(ncnnMat, snnOutput, 0.1);
    printf("test_convolution test res: %d for w=%d, h=%d, c=%d, outch=%d, kernel=%d, dialation=%d, stride=%d, pad=%d, bias=%d\n", ret, width, height, inChs,
           outChs, kernel, dilation, stride, pad, bias);
}

int compareConvolution(const std::string & modelName, int layerId, const std::string & inputDump, const std::string & outputDump, int inChs, int outChs,
                       int dim, int kernel, int padding, int stride, bool force3Channels, snn::GpuBackendType backend) {
    auto weights   = getWeigitBiasFromNCNN(modelName, layerId);
    auto inputNCNN = getSNNLayer(inputDump, false, inChs);

    ncnn::ParamDict padPD;

    padPD.set(0, kernel / 2);
    padPD.set(1, kernel / 2);
    padPD.set(2, kernel / 2);
    padPD.set(3, kernel / 2);

    padPD.set(4, padding);
    padPD.set(5, 0);

    std::vector<ncnn::Mat> padW(0);
    ncnn::Mat              padB;
    test_layer_naive<ncnn::Padding>(ncnn::layer_to_index("Padding"), padPD, padW, inputNCNN, padB, (void (*)(ncnn::Padding *)) 0, 0);

    ncnn::ParamDict pd;
    pd.set(0, outChs); // num_output
    pd.set(1, kernel); // kernel_w
    pd.set(2, 1);      // dilation_w
    pd.set(3, stride); // stride_w
    pd.set(4, 0);      // pad_w
    pd.set(5, 0);      // bias_term
    pd.set(6, outChs * inChs * kernel * kernel);

    int       activation_type = 0; // RAND() % 7; // 0 1 2 3 4 5 6 // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.1f;              // RandomFloat(0, 1); //(activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1); // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    auto bnWeights = getBatchNormFromNCNN(modelName, layerId + 1);

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
        int       ret = test_layer_naive(ncnn::layer_to_index("Convolution"), pd, weights, padB, convOutput, (void (*)(ncnn::Convolution *)) 0, 0);
        if (ret != 0) { fprintf(stderr, "test_layer_naive failed\n"); }

        test_layer_naive(ncnn::layer_to_index("BatchNorm"), pd, bnWeights, convOutput, ncnnOutput, (void (*)(ncnn::BatchNorm *)) 0, 0);
    }
    ShaderUnitTest test(backend);

    cv::Mat inputMat = NCNNMat2CVMat(inputNCNN);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    std::vector<float>   inputBias    = std::vector<float>(outChs, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar *) inputWeights[p].data, (uchar *) weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
    }

    bool                                      useBN = true;
    std::map<std::string, std::vector<float>> batchNormalization;
    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile   = test.snnConvTestWithLayer(inputMat, inputWeights, inputBias, dim, dim, inChs, outChs, kernel, 1, stride, 0, false,
                                             snn::MRTMode::SINGLE_PLANE, useBN, batchNormalization);
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outChs);

    auto snnDump = getSNNLayer(outputDump, false, outChs);

    int ret = CompareMat(ncnnOutput, snnOutput, 0.1);
    printf("-----------------------------Compare ncnn output with snn output res: %d\n", ret);

    ret = CompareMat(ncnnOutput, snnDump, 0.01);
    printf("-----------------------------Compare ncnn output with snn dump res: %d\n", ret);

    return ret;
}

int main(int argc, char ** argv) {
    SRAND(7767517);

    int          width         = 8;
    int          height        = 8;
    int          channel       = 128;
    int          outch         = 1;
    int          kernel        = 1;
    int          stride        = 1;
    bool         useCompute    = false;
    snn::MRTMode mrtMode       = snn::MRTMode::SINGLE_PLANE;
    bool         use2chMrt     = false;
    bool         useVulkan     = false;
    bool         useHalf       = false;
    bool         printMismatch = false;

    CLI::App app;
    app.add_option("-W", width, "width");
    app.add_option("-H", height, "height");
    app.add_option("-K", channel, "input channels");
    app.add_option("-C", outch, "output channels");
    app.add_option("-R", kernel, "kernel size");
    app.add_option("-S", stride, "stride");
    app.add_flag("--use_compute", useCompute, "Use compute shader");
    app.add_flag("--use_2ch_mrt", use2chMrt, "Use double plane MRT (OpenGL only)");
    app.add_flag("--use_vulkan", useVulkan, "Use Vulkan");
    app.add_flag("--use_half", useHalf, "Use half-precision floating point values (fp16)");
    app.add_flag("--print_mismatch", printMismatch, "Print results mismatch");
    CLI11_PARSE(app, argc, argv);
    CHECK_PLATFORM_SUPPORT(useVulkan)

    snn::GpuBackendType backend = useVulkan ? snn::GpuBackendType::VULKAN : snn::GpuBackendType::GL;

    if (use2chMrt) { mrtMode = snn::MRTMode::DOUBLE_PLANE; }

    test_convolution(width, height, channel, outch, kernel, 1, stride, 0 /*padding*/, 1, 0.0, useCompute, mrtMode, backend, useHalf, printMismatch);
}
