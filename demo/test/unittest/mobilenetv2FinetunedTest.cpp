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

#include "layer/convolution.h"
#include "layer/padding.h"
#include "layer/pooling.h"
#include "layer/interp.h"

#include "cpu.h"
#include "net.h"

#include "matutil.h"

#define NCNN_MODEL_NAME   "mobilenetv2_pretrained_imagenet"
#define SNN_MODEL_NAME    "mobilenetv2_pretrained_imagenet.json"
#define TEST_IMAGE        "imagenet1.png"
#define COMPARE_THRESHOLD 0.01

int main(int argc, char** argv) {
    SRAND(7767517);

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret              = 0;
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

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "input_1_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(formatString("%s/MobileNetV2/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         true, 32);
    ret     = CompareMat(ncnnMat, snnMat);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);
    ////pretty_print_ncnn(snnMat);

    // compareNCNNLayerSNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), 1,
    //     formatString("%s/MobileNetV2/%s layer [01] Conv2D_7x7 pass[15]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str(),
    //     3, 64, 32, 7, 0, 2, true);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "Conv1_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret = CompareMat(ncnnMat, snnMat, 0.1);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);

        // SNN_LOGI("Testing if padding is the problem\n");
        // ncnn::Mat centerPixelsNcnn, centerPixelsSnn;
        // ncnn::copy_cut_border(ncnnMat, centerPixelsNcnn, 1, 1, 1, 1);
        // ncnn::copy_cut_border(snnMat, centerPixelsSnn, 1, 1, 1, 1);

        // ret = CompareMat(centerPixelsNcnn, centerPixelsSnn, COMPARE_THRESHOLD);
        // SNN_LOGI("------------------------------- Conv_Layer_1 center output res: %d\n", ret);
    }
    printf("-----------------------------1st Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "expanded_conv_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [02] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         false, 32);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st DepthWise Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "expanded_conv_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str(), false, 16);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------2nd Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_1_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------3rd Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_1_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [05] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 96);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [05] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 23).c_str(), false, 96);
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st ZeroPadding2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_1_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [06] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------DepthwiseConv2D after 1st Zero Padding  output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_1_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(5, mrtMode)).c_str(), false, 24);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv2D after 1st Padding res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_2_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [09] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_2_add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [11] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(5, mrtMode)).c_str(), false, 24);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_3_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [12] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------After 1st Add Block Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_3_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [13] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 144);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [13] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 35).c_str(), false, 144);
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------2nd Zero Padding res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_3_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [14] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------After 2nd Zero Padding DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_3_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [15] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------After 2nd Zero Padding Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_4_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [16] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------2nd Block Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_4_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [17] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------2nd Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_4_add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [19] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------2nd Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_5_add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [23] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------3rd Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_6_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [24] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------The Conv2d before 3rd padding res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_6_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 192);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 47).c_str(), false, 192);
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------3rd Zero Padding res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_6_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [26] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------The DepthwiseConv2D after 3rd Padding output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_6_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [27] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------The Conv2d after 3rd padding res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_7_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [28] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------4th Block Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_7_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [29] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------4th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_7_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [31] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------4th Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_8_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [33] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------5th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_8_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [35] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------5th Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_9_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [37] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------6th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_9_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [39] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------6th Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_10_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [41] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------After 6th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_10_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [42] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------After 6th Block Conv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_11_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [44] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(143, mrtMode)).c_str(), false, 576);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------7th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_11_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [46] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------7th Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_12_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [48] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(143, mrtMode)).c_str(), false, 576);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------8th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_12_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [50] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------8th Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_13_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [52] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 576);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [52] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 143).c_str(), false, 576);
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------4th Zero Padding res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_13_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [53] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(143, mrtMode)).c_str(), false, 576);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Block DepthwiseConv2D after 4th Padding output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_13_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [54] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(39, mrtMode)).c_str(), false, 160);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------The Conv2d after 4th padding res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_14_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [56] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(239, mrtMode)).c_str(), false, 960);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------9th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_14_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [58] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(39, mrtMode)).c_str(), false, 160);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------9th Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_15_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [60] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(239, mrtMode)).c_str(), false, 960);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------10th Block DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_15_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [62] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(39, mrtMode)).c_str(), false, 160);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------10th Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "block_16_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [64] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(239, mrtMode)).c_str(), false, 960);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Last DepthwiseConv2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "out_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [66] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(319, mrtMode)).c_str(), false,
                         1280);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Last Conv2D output res: %d\n", ret);

    return 0;
}
