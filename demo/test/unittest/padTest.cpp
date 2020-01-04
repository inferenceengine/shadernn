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

#include "layer/padding.h"

#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

static int test_pad(int w, int h, int c, int outch, int type, int kernel, float value) {
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();
    ShaderUnitTest test;

    ncnn::Mat padA = RandomMat(w, h, c);
    // ncnn::Mat padA = SetValueMat(w, h, c, 1.0f);
    pretty_print_ncnn(padA);

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
    // pretty_print_ncnn(snnOutput);

    ret = CompareMat(ncnnMat, snnOutput, 0.1);
    pretty_print_ncnn(ncnnMat);
    pretty_print_ncnn(snnOutput);

    printf("test_upsample test res: %d for w=%d, h=%d, c=%d, outch=%d, type=%d, value=%f\n", ret, w, h, c, outch, type, value);

    return 0;
}

int main() {
    SRAND(7767517);

    // test_pad(8, 8, 1, 1, 0/*padding type*/, 4/*kernel*/, 0.0f);
    // test_pad(8, 8, 1, 1, 1/*padding type*/, 4/*kernel*/, 0.0f);
    test_pad(8, 8, 1, 1, 2 /*padding type*/, 4 /*kernel*/, 0.0f);

    return 0;
}
