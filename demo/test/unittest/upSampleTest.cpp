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

static int test_resize(int w, int h, int c, int outch, int type, int scale) {
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();
    ShaderUnitTest test;

    //    ncnn::Mat padA = RandomMat(w, h, c);
    ncnn::Mat padA = SetValueMat(w, h, c, 0.0f);
    padA[12]       = 100;
    //     padA[4] = 0.44;
    //    ncnn::Mat padA = {1,2,3};
    //    for (size_t i = 0; i < padA.total(); i++)
    //    {
    //        padA[i] = i;
    //    }

    ncnn::Mat ncnnMat;

    ncnn::ParamDict pd;
    pd.set(0, type);
    pd.set(1, scale);
    pd.set(2, scale);
    pd.set(3, h * scale);
    pd.set(4, w * scale);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_naive<ncnn::Interp>(ncnn::layer_to_index("Interp"), pd, weights, padA, ncnnMat, (void (*)(ncnn::Interp*)) 0, 0);
    if (ret != 0) {
        fprintf(stderr, "test_layer_naive failed\n");
    }

    auto inputMat = NCNNMat2CVMat(padA);
    auto outFile  = test.snnUpsampleTestWithLayer(inputMat, w, h, c, scale, type);

    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, outch);
    // pretty_print_ncnn(snnOutput);

    pretty_print_ncnn(padA);
    ret = CompareMat(ncnnMat, snnOutput, 0.1);
    pretty_print_ncnn(ncnnMat);
    pretty_print_ncnn(snnOutput);

    printf("test_upsample test res: %d for w=%d, h=%d, c=%d, outch=%d, type=%d, scale=%d\n", ret, w, h, c, outch, type, scale);

    return 0;
}

int main() {
    SRAND(7767517);

    test_resize(4, 4, 1, 1, 1, 2);

    return 0;
}
