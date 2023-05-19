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

#include "layer/concat.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

static int test_concate(int w, int h, int c, const std::string& activation, snn::GpuBackendType backend, bool printMismatch) {
    ncnn::Mat matA = RandomMat(w, h, c);
    ncnn::Mat matB = matA;

    std::vector<ncnn::Mat> resNCNN;

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 0);   // with_scalar
    pd.set(2, 0.f); // b

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = matA;
    ab[1] = matB;

    int ret = test_layer_naive<ncnn::Concat>(ncnn::layer_to_index("Concat"), pd, weights, ab, 2, resNCNN, (void (*)(ncnn::Concat*)) 0, 0);
    if (ret != 0 || resNCNN.size() == 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        fprintf(stderr, "test_concate failed a.dims=%d a=(%d %d %d) b.dims=%d b=(%d %d %d)\n", matA.dims, matA.w, matA.h, matA.c, matB.dims, matB.w, matB.h,
                matB.c);
    }

    ab[0] = matA;
    ab[1] = resNCNN[0];
    std::vector<ncnn::Mat> resNCNN1;
    int ret1 = test_layer_naive<ncnn::Concat>(ncnn::layer_to_index("Concat"), pd, weights, ab, 2, resNCNN1, (void (*)(ncnn::Concat*)) 0, 0);
    if (ret1 != 0 || resNCNN1.size() == 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        fprintf(stderr, "test_concate failed a.dims=%d a=(%d %d %d) b.dims=%d b=(%d %d %d)\n", matA.dims, matA.w, matA.h, matA.c, ab[1].dims, ab[1].w, ab[1].h,
                ab[1].c);
    }

    ShaderUnitTest test(backend);

    cv::Mat inputMat1 = NCNNMat2CVMat(matA);
    cv::Mat inputMat2 = NCNNMat2CVMat(matB);

    auto outFile = test.snnConcateTestWithLayer(inputMat1, w, h, c, 1, 1, activation);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, c + 2 * c);

    ret = CompareMat(resNCNN1[0], snnOutput, 0.1);
    printf("concat test res: %d\n", ret);
    if (ret && printMismatch) {
        pretty_print_ncnn(resNCNN1[0]);
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

    test_concate(5, 7, 4, "relu", backend, printMismatch);

    return 0;
}
