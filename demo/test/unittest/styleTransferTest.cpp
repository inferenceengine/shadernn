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

#include "layer/convolution.h"
#include "layer/padding.h"
#include "layer/pooling.h"
#include "layer/interp.h"

#include "cpu.h"
#include "net.h"

#include "matutil.h"

#define NCNN_MODEL_NAME "candy-9_simplified-opt"
#define SNN_MODEL_NAME  "candy-9_simplified.json"
#define TEST_IMAGE      "ant.png"

#define COMPARE_THRESHOLD 0.01

int main(int argc, char** argv) {
    SRAND(7767517);

    snn::MRTMode mrtMode = snn::MRTMode::SINGLE_PLANE;

    if (argc > 1) {
        if (strcmp("--use_2ch_mrt", argv[1]) == 0) {
            mrtMode = snn::MRTMode::DOUBLE_PLANE;
            printf("MRT_MODE SET TO: DOUBLE_PLANE\nSHADER MODE: FRAGMENT\n");
        }

        if (strcmp("--use_4ch_mrt", argv[1]) == 0) {
            mrtMode = snn::MRTMode::QUAD_PLANE;
            printf("MRT_MODE SET TO: QUAD_PLANE\nSHADER MODE: FRAGMENT\n");
        }

        if (strcmp("--use_compute", argv[1]) == 0) {
            mrtMode = (snn::MRTMode) 0;
            printf("MRT_MODE SET TO: NULL\nSHADER MODE: COMPUTE\n");
        }
    } else {
        printf("DEFAULT MRT MODE SET TO: snn::MRTMode::SINGLE_PLANE\n");
        printf("DEFAULT SHADER MODE: FRAGMENT SHADER\n");
        printf("To change this mode, use the following options. Note that they are mutually exclusive:\n");
        printf("\t--use_2ch_mrt : Use 2 render targets (DOUBLE_PLANE MRT)\n");
        printf("\t--use_4ch_mrt : Use 4 render targets (QUAD_PLANE MRT)\n");
        printf("\t--use_compute : Use compute shader\n");
    }

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "input1", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         true);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "64", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "66", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [02] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "68", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Conv_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "70", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [04] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------InstanceNorm_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "72", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Conv_layer_3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "74", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [06] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------InstanceNorm_3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "76", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block1 Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "78", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [08] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block1 InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "80", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [09] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block1 Conv_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "81", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [10] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block1 InstanceNorm_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "82", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [11] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block1 Add output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "84", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [12] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block2 Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "86", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [13] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block2 InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "88", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [14] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block2 Conv_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "89", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [15] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block2 InstanceNorm_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "90", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [16] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block2 Add output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "92", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [17] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block3 Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "94", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [18] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block3 InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "96", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [19] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block3 Conv_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "97", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [20] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block3 InstanceNorm_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "98", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [21] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block3 Add output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "100", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [22] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block4 Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "102", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [23] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block4 InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "104", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [24] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block4 Conv_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "105", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [25] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block4 InstanceNorm_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "106", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [26] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block4 Add output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "108", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [27] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block5 Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "110", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [28] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block5 InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "112", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [29] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block5 Conv_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "113", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [30] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block5 InstanceNorm_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "114", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [31] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------Block5 Add output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "139", 224, false);
    snnMat = getSNNLayer(formatString("%s/StyleTransfer/%s layer [32] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(),
                         false);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------1st UpSample output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "141", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [33] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("----------------------------- 1st UpSample Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "143", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [34] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------1st UpSample InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "168", 224, false);
    snnMat = getSNNLayer(formatString("%s/StyleTransfer/%s layer [35] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(),
                         false);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------2nd UpSample output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "170", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [36] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("----------------------------- 2nd UpSample Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "172", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [37] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    printf("-----------------------------2nd UpSample InstanceNorm_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/StyleTransfer/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "output1", 224, false);
    snnMat = getSNNLayer(formatString("%s/StyleTransfer/%s layer [38] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(0, mrtMode)).c_str(), true);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // return 0;
    }
    // pretty_print_ncnn(ncnnMat);
    // pretty_print_ncnn(snnMat);
    printf("----------------------------- 2nd UpSample Conv_layer_2 output res: %d\n", ret);

    return 0;
}
