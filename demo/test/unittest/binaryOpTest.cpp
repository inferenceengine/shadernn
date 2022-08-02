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
#include "testutil.h"
#include "matutil.h"
#include "shaderUnitTest.h"

#define OP_TYPE_MAX 9

static int op_type = 0;

static int test_addop2(int w, int h, int c, std::string activation) {
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

    int ret = test_layer_naive<ncnn::BinaryOp>(ncnn::layer_to_index("BinaryOp"), pd, weights, ab, 2, resNCNN, (void (*)(ncnn::BinaryOp*)) 0, 0);
    if (ret != 0 || resNCNN.size() == 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d) b.dims=%d b=(%d %d %d) op_type=%d\n", matA.dims, matA.w, matA.h, matA.c, matB.dims, matB.w,
                matB.h, matB.c, op_type);
    }
    if (resNCNN.size() > 0) {
        SNN_LOGI("OUTPUT FROM NCNN\n");
        pretty_print_ncnn(resNCNN.at(0));
    }

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    ShaderUnitTest test;

    int size[3] = {w, h, c};
    // cv::Mat inputMat(3, size, CV_32FC1, padA.data);
    cv::Mat inputMat1(3, size, CV_32FC1);
    cv::Mat inputMat2(3, size, CV_32FC1);
    printf("%%%%%%%% %s:%d\n", __FUNCTION__, __LINE__);

    for (int q = 0; q < matA.c; q++) {
        const float* ptr  = matA.channel(q);
        const float* ptr2 = matB.channel(q);

        for (int y = 0; y < matA.h; y++) {
            for (int x = 0; x < matA.w; x++) {
                inputMat1.at<float>(y, x, q) = ptr[x];
                inputMat2.at<float>(y, x, q) = ptr2[x];
            }
            ptr += matA.w;
            ptr2 += matB.w;
        }
    }
    printf("%%%%%%%% %s:%d\n", __FUNCTION__, __LINE__);

    auto outFile = test.snnAddTestWithLayer2(inputMat1, w, h, c, 1, 1, activation);

    printf("Output file:%s\n", formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str());
    auto snnOutput = getSNNLayer(formatString("%s/%s", DUMP_DIR, outFile.c_str()).c_str(), false, c);

    pretty_print_ncnn(resNCNN.at(0));
    pretty_print_ncnn(snnOutput);

    ret = CompareMat(resNCNN.at(0), snnOutput, 0.1);

    printf("test_add test res: %d\n", ret);

    return ret;
}

int main() {
    SRAND(7767517);

    test_addop2(4, 4, 1, "linear");

    return 0;
}
