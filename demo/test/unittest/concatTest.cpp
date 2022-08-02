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

static int test_concate(int w, int h, int c, std::string activation) {
    ncnn::Mat matA = RandomMat(w, h, c);
    // ncnn::Mat matB = RandomMat(w, h, c);
    ncnn::Mat matB = matA;

    pretty_print_ncnn(matA);

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
    if (resNCNN.size() > 0) {
        SNN_LOGI("OUTPUT FROM NCNN CONCAT0\n");
        pretty_print_ncnn(resNCNN.at(0));
    }

    ab[0] = matA;
    ab[1] = resNCNN.at(0);
    std::vector<ncnn::Mat> resNCNN1;
    int ret1 = test_layer_naive<ncnn::Concat>(ncnn::layer_to_index("Concat"), pd, weights, ab, 2, resNCNN1, (void (*)(ncnn::Concat*)) 0, 0);
    if (ret1 != 0 || resNCNN1.size() == 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        fprintf(stderr, "test_concate failed a.dims=%d a=(%d %d %d) b.dims=%d b=(%d %d %d)\n", matA.dims, matA.w, matA.h, matA.c, ab[1].dims, ab[1].w, ab[1].h,
                ab[1].c);
    }
    if (resNCNN1.size() > 0) {
        SNN_LOGI("OUTPUT FROM NCNN CONCAT1\n");
        pretty_print_ncnn(resNCNN1.at(0));
    }

    //    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    ShaderUnitTest test;

    cv::Mat inputMat1 = NCNNMat2CVMat(matA);
    cv::Mat inputMat2 = NCNNMat2CVMat(matB);

    printf("%%%%%%%% %s:%d\n", __FUNCTION__, __LINE__);

    auto outFile = test.snnConcateTestWithLayer(inputMat1, w, h, c, 1, 1, activation);

    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, c + 2 * c);

    printf("pretty_print_ncnn: resNCNN.at(0)\n");
    pretty_print_ncnn(resNCNN.at(0));
    printf("\n\npretty_print_ncnn: snnOutput\n");
    pretty_print_ncnn(snnOutput);

    ret = CompareMat(resNCNN1.at(0), snnOutput, 0.1);

    printf("test_concat test res: %d\n", ret);

    return ret;
}

int main() {
    SRAND(7767517);

    test_concate(5, 7, 4, "relu");

    return 0;
}
