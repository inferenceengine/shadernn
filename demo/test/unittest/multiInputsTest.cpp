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

#include "layer/binaryop.h"
#include "layer/pooling.h"
#include "layer/interp.h"
#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

#define OP_TYPE_MAX 9

static int op_type = 0;

static int test_multiInputs(int w, int h, int c, int pooling_type, int kernel, int stride, snn::GpuBackendType backend, bool printMismatch) {
    ncnn::Mat matA = RandomMat(w, h, c);
    ncnn::Mat matB = RandomMat(w/kernel, h/kernel, c);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type); // pooling_type
    pd.set(1, kernel);       // kernel_w
    pd.set(2, stride);       // stride_w
    pd.set(3, 0);            // pad_w
    pd.set(4, 0);            // global_pooling

    SNN_LOGD("kernel %d, stride: %d", kernel, stride);
    std::vector<ncnn::Mat> weights(0);

    ncnn::Mat poolA;
    int ret = test_layer_naive<ncnn::Pooling>(ncnn::layer_to_index("Pooling"), pd, weights, matA, poolA, (void (*)(ncnn::Pooling*)) 0, 0);

    std::vector<ncnn::Mat> resNCNN;

    ncnn::ParamDict addPd;
    addPd.set(0, 0);
    addPd.set(1, 0);   // with_scalar
    addPd.set(2, 0.f); // b

    std::vector<ncnn::Mat> addWeights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = poolA;
    ab[1] = matB;

    int addRet = test_layer_naive<ncnn::BinaryOp>(ncnn::layer_to_index("BinaryOp"), addPd, addWeights, ab, 2, resNCNN, (void (*)(ncnn::BinaryOp*)) 0, 0);
    if (addRet != 0 || resNCNN.size() == 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d) b.dims=%d b=(%d %d %d) op_type=%d\n", matA.dims, matA.w, matA.h, matA.c, matB.dims, matB.w,
                matB.h, matB.c, op_type);
    }

    ShaderUnitTest test(backend);

    auto inputMat1 = NCNNMat2CVMat(poolA);
    auto inputMat2 = NCNNMat2CVMat(matB);

    auto outFile = test.snnMultiInputsTestWithLayer(inputMat1, inputMat2, w/kernel, h/kernel, c);
    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, c);

    ret = CompareMat(resNCNN[0], snnOutput, 0.1);
    printf("multiinput test res: %d\n", ret);
    if (ret && printMismatch) {
        pretty_print_ncnn(resNCNN[0]);
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

    test_multiInputs(8, 8, 2, 0, 2, 2, backend, printMismatch);

    return 0;
}
