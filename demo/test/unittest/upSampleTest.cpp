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

#include "layer/interp.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
    #undef Success
#endif
#include "CLI/CLI.hpp"

static int test_resize(int w, int h, int type, bool useCompute, int scale, const std::vector<double> & values, snn::GpuBackendType backend,
                       bool printMismatch) {
    ShaderUnitTest test(backend);

    const int ch    = 1;
    const int outch = 1;

    ncnn::Mat ncnnMatOrig = SetValueMat(w, h, ch, 0.0f);
    for (size_t i = 0, k = 0; i < h; ++i) {
        for (size_t j = 0; j < w && k < values.size(); ++j, ++k) { ncnnMatOrig[k] = values[k]; }
    }

    ncnn::ParamDict pd;
    pd.set(0, type);
    pd.set(1, scale);
    pd.set(2, scale);
    pd.set(3, h * scale);
    pd.set(4, w * scale);

    ncnn::Mat ncnnMatResized;
    int       ret = test_layer_naive<ncnn::Interp>(ncnn::layer_to_index("Interp"), pd, std::vector<ncnn::Mat>(), ncnnMatOrig, ncnnMatResized,
                                             (void (*)(ncnn::Interp *)) 0, 0);
    if (ret != 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        return ret;
    }

    auto inputMat  = NCNNMat2CVMat(ncnnMatOrig);
    auto outFile   = test.snnUpsampleTestWithLayer(inputMat, w, h, ch, scale, type, useCompute);
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);

    ret = CompareMat(ncnnMatResized, snnOutput, 0.1);
    printf("\nupsample test res: %s for w=%d, h=%d, c=%d, outch=%d, type=%d, scale=%d\n", ret ? "FAILED" : "succeeded", w, h, ch, outch, type, scale);
    if (ret && printMismatch) {
        pretty_print_ncnn(ncnnMatResized);
        pretty_print_ncnn(snnOutput, "SNN");
    }

    return ret;
}

int main(int argc, char ** argv) {
    SRAND(7767517);

    int                 width         = 2;
    int                 height        = 2;
    int                 scale         = 2;
    int                 type          = 1;
    bool                useCompute    = false;
    bool                useVulkan     = false;
    bool                printMismatch = false;
    std::vector<double> values {1.0, 2.0, 3.0, 4.0};

    CLI::App app;
    app.add_option("-W", width, "width");
    app.add_option("-H", height, "height");
    app.add_option("-S", scale, "scale");
    app.add_flag("--use_compute", useCompute, "Use compute shader");
    app.add_set("--type", type, {1, 2}, "Interpolation type. 1 = Nearest, 2 = Bilinear");
    app.add_flag("--use_vulkan", useVulkan, "Use Vulkan");
    app.add_flag("--print_mismatch", printMismatch, "Print results mismatch");
    app.add_option("--values", values, "Values");
    CLI11_PARSE(app, argc, argv);
    CHECK_PLATFORM_SUPPORT(useVulkan)

    snn::GpuBackendType backend = useVulkan ? snn::GpuBackendType::VULKAN : snn::GpuBackendType::GL;

    printf("Using %s type\n", type == 1 ? "NEAREST" : "BILINEAR");
    printf("Using %s shader\n", useCompute ? "COMPUTE" : "FRAGMENT");
    printf("Using %s backend\n", useVulkan ? "Vulkan" : "OpnGL");

    test_resize(width, height, type, useCompute, scale, values, backend, printMismatch);

    return 0;
}
