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

#include "layer/selu.h"
#include "layer/relu.h"
#include "layer/tanh.h"
#include "layer/sigmoid.h"
#include "layer/swish.h"
#include "layer/clip.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

static int test_activation(int w, int h, int c, const std::string& activation, float leaky_value, snn::GpuBackendType backend, bool printMismatch) {
    ncnn::Mat matA = RandomMat(w, h, c);

    ncnn::Mat resNCNN;
    ncnn::ParamDict pd;
    std::vector<ncnn::Mat> weights(0);

    int ret = 1;
    if (!activation.compare("relu")) {
        ret = test_layer_naive<ncnn::ReLU>(ncnn::layer_to_index("ReLU"), pd, weights, matA, resNCNN, (void (*)(ncnn::ReLU*)) 0, 0);
    }
    if (!activation.compare("relu6")) {
        pd.set(0, 0.f);
        pd.set(1, 6.f); // relu6
        ret = test_layer_naive<ncnn::Clip>(ncnn::layer_to_index("Clip"), pd, weights, matA, resNCNN, (void (*)(ncnn::Clip*)) 0, 0);
    }
    if (!activation.compare("tanh")) {
        ret = test_layer_naive<ncnn::TanH>(ncnn::layer_to_index("TanH"), pd, weights, matA, resNCNN, (void (*)(ncnn::TanH*)) 0, 0);
    }
    if (!activation.compare("sigmoid")) {
        ret = test_layer_naive<ncnn::Sigmoid>(ncnn::layer_to_index("Sigmoid"), pd, weights, matA, resNCNN, (void (*)(ncnn::Sigmoid*)) 0, 0);
    }
    if (!activation.compare("leakyRelu")) {
        if (leaky_value > 1) {
            fprintf(stderr, "leaky_value = %f out of range (-inf, 1)\n", leaky_value);
            return EXIT_FAILURE;
        }
        pd.set(0, leaky_value);
        ret = test_layer_naive<ncnn::ReLU>(ncnn::layer_to_index("ReLU"), pd, weights, matA, resNCNN, (void (*)(ncnn::ReLU*)) 0, 0);
    }
    if (!activation.compare("SiLU")) {
        ret = test_layer_naive<ncnn::Swish>(ncnn::layer_to_index("Swish"), pd, weights, matA, resNCNN, (void (*)(ncnn::Swish*)) 0, 0);
    }
    if (ret != 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        fprintf(stderr, "test_concate failed a.dims=%d a=(%d %d %d)\n", matA.dims, matA.w, matA.h, matA.c);
    }

    ShaderUnitTest test(backend);

    cv::Mat inputMat = NCNNMat2CVMat(matA);
    auto outFile     = test.snnActivationTestWithLayer(inputMat, w, h, c, 1, 1, activation, leaky_value);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());

    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, c);

    ret = CompareMat(resNCNN, snnOutput, 0.001);
    printf("\nactivation test, res: %d\n\n", ret);
    if (ret && printMismatch) {
        pretty_print_ncnn(resNCNN);
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

    test_activation(16, 16, 4, "SiLU", -5, backend, printMismatch);

    return 0;
}
