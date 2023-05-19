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
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/pooling.h"
#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

static int test_pooling2(int w, int h, int c, int pooling_type, int kernel, int stride, int pad, int global_pooling, int pad_mode,
                         int avgpool_count_include_pad, int adaptive_pooling, int out_w, snn::GpuBackendType backend, bool printMismatch) {
    ncnn::Mat padA  = SetValueMat(w, h, c, 1.0f);
    padA[0]         = 100;
    padA[w * h - 1] = 200;

    ncnn::ParamDict pd;
    pd.set(0, pooling_type);              // pooling_type
    pd.set(1, kernel);                    // kernel_w
    pd.set(2, stride);                    // stride_w
    pd.set(3, pad);                       // pad_w
    pd.set(4, global_pooling);            // global_pooling
    pd.set(5, pad_mode);                  // pad_mode
    pd.set(6, avgpool_count_include_pad); // avgpool_count_include_pad
    pd.set(7, adaptive_pooling);          // adaptive_pooling
    pd.set(8, out_w);                     // out_w

    printf("%%%%%%%% %s:%d: kernel %d, stride: %d\n", __FUNCTION__, __LINE__, kernel, stride);
    std::vector<ncnn::Mat> weights(0);
    pretty_print_ncnn(padA);

    ncnn::Mat b;
    int ret = test_layer_naive<ncnn::Pooling>(ncnn::layer_to_index("Pooling"), pd, weights, padA, b, (void (*)(ncnn::Pooling*)) 0, 0);
    if (ret != 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        fprintf(
            stderr,
            "test_pooling failed w=%d h=%d c=%d pooling_type=%d kernel=%d stride=%d pad=%d "
            "global_pooling=%d pad_mode=%d avgpool_count_include_pad=%d adaptive_pooling=%d out_w=%d\n",
            w, h, c, pooling_type, kernel, stride, pad, global_pooling, pad_mode, avgpool_count_include_pad, adaptive_pooling, out_w);
    }

    ShaderUnitTest test(backend);

    cv::Mat inputMat = NCNNMat2CVMat(padA);

    auto outFile = test.snnPoolingTestWithLayer(inputMat, w, h, c, kernel, stride, pooling_type, pad_mode);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, c);

    ret = CompareMat(b, snnOutput);
    printf("pooling test res: %d\n", ret);
    if (ret && printMismatch) {
        pretty_print_ncnn(b);
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

    test_pooling2(9, 9, 4, 1, 2, 3, 0, 0, 3 /*padding*/, 1, 0, 13, backend, printMismatch);
    return 0;
}
