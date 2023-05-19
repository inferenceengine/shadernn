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

#include "layer/innerproduct.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

static int test_dense(int w, int h, int c, int outw, int outch,  bool bias, snn::GpuBackendType backend, bool printMismatch) {
    ncnn::Mat padB = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * w * h * c); // weight size

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);

    weights[0] = RandomMat(w * outw);
    if (bias) {
        weights[1] = SetValueMat(outw, 0.0f);
    }

    ncnn::Mat ncnnOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("InnerProduct"), pd, weights, padB, ncnnOutput, (void (*)(ncnn::InnerProduct*)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
            fprintf(stderr, "test_dense failed w=%d h=%d c=%d outw=%d, outch=%d \n", w, h, c, outw, outch);
            return -1;
        }
    }
    pretty_print_ncnn(ncnnOutput);

    ShaderUnitTest test(backend);

    cv::Mat inputMat = NCNNMat2CVMat(padB);

    std::vector<std::vector<float>> inputWeights;
    std::vector<float> inputBias;

    weights[0] = weights[0].reshape(w, outw);
    ncnnToMat(weights[0], inputWeights);

    if (bias) {
        ncnnToVec(weights[1], inputBias);
    }

    int ret = 0;

    auto outFile = test.snnDenseTestWithLayer(inputMat, inputWeights, inputBias, w, h, c, outch);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);

    ret = CompareMat(ncnnOutput, snnOutput, 0.01);
    printf("dense test res: %d for w=%d, h=%d, c=%d, outw:%d outch=%d, bias=%d\n", ret, w, h, c, outw, outch, bias);
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

    test_dense(11, 1, 1, 5, 1, 1, backend, printMismatch);

    return 0;
}
