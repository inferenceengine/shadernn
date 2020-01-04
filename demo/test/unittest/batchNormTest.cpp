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

#include "layer/batchnorm.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

static int test_batchnorm(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, float inputValue = 1.0f,
                          float padValue = 0.0f, bool useOldShader = false) {
    ncnn::Mat padA = RandomMat(w, h, c);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * kernel * kernel);
    // weights[0] = SetValueMat(outch * kernel * kernel, 1.0f);
    if (bias) {
        weights[1] = RandomMat(outch);
        // weights[1] = SetValueMat(outch, 0.0f);
    }

    std::vector<float> bnGamma;
    std::vector<float> bnMean;
    std::vector<float> bnVar;
    std::vector<float> bnBeta;

    ncnn::ParamDict pd;
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

        int ret = test_layer_naive(ncnn::layer_to_index("BatchNorm"), pd, bnweights, padA, bnOutput, (void (*)(ncnn::BatchNorm*)) 0, 0);
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

    int ret = 0;

    bool useBN = true;
    std::map<std::string, std::vector<float>> batchNormalization;

    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnBatchNormTestWithLayer(inputMat, inputWeights, inputBias, w, h, c, outch, kernel, dilation, stride, pad, bias, useOldShader, useBN,
                                                  batchNormalization);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);
    // pretty_print_ncnn(snnOutput);

    ret = CompareMat(bnOutput, snnOutput, 0.001);
    pretty_print_ncnn(bnOutput);
    pretty_print_ncnn(snnOutput);

    printf("test_batchnorm test res: %d for w=%d, h=%d, c=%d, outch=%d, kernel=%d, dialation=%d, stride=%d, pad=%d, bias=%d\n", ret, w, h, c, outch, kernel,
           dilation, stride, pad, bias);

    return ret;
}

int main() {
    SRAND(7767517);

    test_batchnorm(8, 8, 1, 1, 1, 1, 2, 0, 1, 1.0, 0.0, false);

    return 0;
}
