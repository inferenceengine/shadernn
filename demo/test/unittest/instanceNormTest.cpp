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

#include "layer/instancenorm.h"
#include "layer/padding.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

static int test_instancenorm(int w, int h, int c, float eps /*= 0.00001f*/, int affine /*= 1*/, snn::GpuBackendType backend, bool printMismatch) {
    int outch  = c;
    int kernel = 1, dilation = 1, stride = 1, pad = 0, bias = 0;
    bool useCompute = true;
    ncnn::Mat padA    = RandomMat(w, h, c);

    int channels = padA.c;

    ncnn::ParamDict pd;
    pd.set(0, affine ? w * h : 0);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);

    ncnn::Mat ncnnOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("InstanceNorm"), pd, weights, padA, ncnnOutput, (void (*)(ncnn::InstanceNorm*)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
            fprintf(stderr, "test_depth_convolution failed w=%d h=%d c=%d eps= %f, affine= %d\n", w, h, c, eps, affine);
            return -1;
        }
    }

    std::vector<float> bnGamma;
    std::vector<float> bnBeta;
    ncnnToVec(weights[0], bnGamma);
    ncnnToVec(weights[1], bnBeta);

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

    batchNormalization["gamma"] = bnGamma;
    batchNormalization["beta"]  = bnBeta;

    auto outFile = test.snnInstanceNormTestWithLayer(inputMat, inputWeights, inputBias, w, h, c, outch, kernel, dilation, stride, pad, bias, useCompute,
                                                     useBN, batchNormalization);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);

    // TODO: Investigate, why such a big error
    ret = CompareMat(ncnnOutput, snnOutput, 0.05);

    printf("instancenorm test res: %d for w=%d, h=%d, c=%d\n", ret, w, h, c);
    if (ret && printMismatch) {
        pretty_print_ncnn(ncnnOutput);
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

    test_instancenorm(224, 224, 32, 0.00001f, 1, backend, printMismatch);

    return 0;
}
