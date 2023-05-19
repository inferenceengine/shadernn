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

#include "layer/batchnorm.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

static int test_batchnorm(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, float inputValue,
                          float padValue, bool useCompute, snn::GpuBackendType backend, bool printMismatch) {
    ncnn::Mat padA = RandomMat(w, h, c);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * kernel * kernel);
    if (bias) {
        weights[1] = RandomMat(outch);
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
        bnweights[1] = RandomMat(channels);
        bnweights[2] = RandomMat(channels);
        bnweights[3] = RandomMat(channels);

        // var must be positive
        Randomize(bnweights[2], 0.001f, 2.f);

        ncnnToVec(bnweights[0], bnGamma);
        ncnnToVec(bnweights[1], bnMean);
        ncnnToVec(bnweights[2], bnVar);
        ncnnToVec(bnweights[3], bnBeta);

        int ret = test_layer_naive(ncnn::layer_to_index("BatchNorm"), pd, bnweights, padA, bnOutput, (void (*)(ncnn::BatchNorm*)) 0, 0);
        printf("BN Test res: %d\n", ret);
    }
    ShaderUnitTest test(backend);

    cv::Mat inputMat = NCNNMat2CVMat(padA);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(c * outch);
    std::vector<float> inputBias      = std::vector<float>(outch, 0.0f);

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

    int ret = 0;

    bool useBN = true;
    std::map<std::string, std::vector<float>> batchNormalization;

    batchNormalization["gamma"]          = bnGamma;
    batchNormalization["movingMean"]     = bnMean;
    batchNormalization["movingVariance"] = bnVar;
    batchNormalization["beta"]           = bnBeta;

    auto outFile = test.snnBatchNormTestWithLayer(inputMat, inputWeights, inputBias, w, h, c, outch, kernel, dilation, stride, pad, bias, useCompute, useBN,
                                                  batchNormalization);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);

    ret = CompareMat(bnOutput, snnOutput, 0.001);
    printf("batchnorm test res: %d for w=%d, h=%d, c=%d, outch=%d, kernel=%d, dialation=%d, stride=%d, pad=%d, bias=%d\n", ret, w, h, c, outch, kernel,
           dilation, stride, pad, bias);
    if (ret && printMismatch) {
        pretty_print_ncnn(bnOutput);
        pretty_print_ncnn(snnOutput, "SNN");
    }

    return ret;
}

int main(int argc, char **argv) {
    SRAND(7767517);

    bool useVulkan = false;
    bool printMismatch = false;

    CLI::App app;
    app.add_flag("--use_vulkan", useVulkan, "Use Vulkan");
    app.add_flag("--print_mismatch", printMismatch, "Print results mismatch");
    CLI11_PARSE(app, argc, argv);
    CHECK_PLATFORM_SUPPORT(useVulkan)

    snn::GpuBackendType backend = useVulkan ? snn::GpuBackendType::VULKAN : snn::GpuBackendType::GL;

    test_batchnorm(8, 8, 1, 1, 1, 1, 1, 0, 1, 1.0, 0.0, false, backend, printMismatch);

    return 0;
}
