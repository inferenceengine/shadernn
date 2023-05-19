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

#include "layer/padding.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

static int test_pad(int w, int h, int c, int outch, int type, int kernel, float value, snn::GpuBackendType backend, bool printMismatch) {
    ShaderUnitTest test(backend);

    ncnn::Mat padA = RandomMat(w, h, c);
    ncnn::Mat ncnnMat;

    ncnn::ParamDict pd;
    pd.set(0, kernel / 2);
    pd.set(1, kernel / 2);
    pd.set(2, kernel / 2);
    pd.set(3, kernel / 2);
    pd.set(4, type); // 0=CONSTANT 1=REPLICATE 2=REFLECT
    pd.set(5, value);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_naive<ncnn::Padding>(ncnn::layer_to_index("Padding"), pd, weights, padA, ncnnMat, (void (*)(ncnn::Padding*)) 0, 0);
    if (ret != 0) {
        fprintf(stderr, "test_layer_naive failed\n");
    }

    auto inputMat = NCNNMat2CVMat(padA);

    auto outFile = test.snnPadTestWithLayer(inputMat, w, h, c, kernel, 1, type, value);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);

    ret = CompareMat(ncnnMat, snnOutput, 0.1);
    printf("pad test test res: %d for w=%d, h=%d, c=%d, outch=%d, type=%d, value=%f\n", ret, w, h, c, outch, type, value);
    if (ret && printMismatch) {
        pretty_print_ncnn(ncnnMat);
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

    test_pad(8, 8, 4, 4, 2/*padding type*/, 4 /*kernel*/, 0.0f, backend, printMismatch);

    return 0;
}
