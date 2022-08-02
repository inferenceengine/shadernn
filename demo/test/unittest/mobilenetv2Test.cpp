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

#define NCNN_MODEL_NAME   "mobilenetV2"
#define SNN_MODEL_NAME    "mobilenetV2.json"
#define TEST_IMAGE        "ant.png"
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

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "input_1_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(formatString("%s/MobileNetV2/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         true, 32);
    ret     = CompareMat(ncnnMat, snnMat);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);
    ////pretty_print_ncnn(snnMat);

    // compareNCNNLayerSNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), 1,
    //     formatString("%s/MobileNetV2/%s layer [01] Conv2D_7x7 pass[15]_input.dump", DUMP_DIR).c_str(),
    //     3, 64, 32, 7, 0, 2, true);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);

        SNN_LOGI("Testing if padding is the problem\n");
        ncnn::Mat centerPixelsNcnn, centerPixelsSnn;
        ncnn::copy_cut_border(ncnnMat, centerPixelsNcnn, 1, 1, 1, 1);
        ncnn::copy_cut_border(snnMat, centerPixelsSnn, 1, 1, 1, 1);

        ret = CompareMat(centerPixelsNcnn, centerPixelsSnn, COMPARE_THRESHOLD);
        SNN_LOGI("------------------------------- Conv_Layer_1 center output res: %d\n", ret);
    }
    printf("-----------------------------Conv_layer_1 output res: %d\n", ret);

    // auto weightBias = getWeigitBiasFromNCNN(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), 1);

    // auto ncnnWeightMat = weightBias[0].reshape(3, 32, 3, 3);
    // pretty_print_ncnn(weightBias[1]);
    // // pretty_print_ncnn(ncnnWeightMat);

    // auto snnWeightMat = getSNNLayer(formatString("%s/weights/%s layer [01] Conv2D pass[0]_0.dump", DUMP_DIR).c_str());
    // // pretty_print_ncnn(snnWeightMat);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_1_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [02] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret = CompareMat(ncnnMat, snnMat, 0.5);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_2_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [03] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         false, 32);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Depthwise_conv_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_3_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str(), false, 16);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_layer_3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_3_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_4 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_9_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [06]
    // MaxPooling2D_3x3 pass[15].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------1st Conv2D_layer_3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [11] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(5, mrtMode)).c_str(), false, 24);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_8_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [13] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);

    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------DepthWise Conv 4 res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "add_1_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [18] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------2nd Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "add_2_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [22] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------3rd Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_14_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [24] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------DepthWise Conv 6 res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_21_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv Layer 12 res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_16_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [27] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------DepthWise Conv 7 res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_24_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [28] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv Layer 15 res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_21_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv Layer 15 res: %d\n", ret);

    // SNN_LOGD("Input to add layer\n----------------------------------------------------");
    // pretty_print_ncnn(snnMat);
    // SNN_LOGD("----------------------------------------------------");
    // pretty_print_ncnn(ncnnMat);
    // SNN_LOGD("----------------------------------------------------");

    // SNN_LOGD("Residual Connection to add layer\n----------------------------------------------------");
    // auto resSnnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] Conv2D_1x1 pass[15].dump", DUMP_DIR).c_str());
    // auto resNcnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_21_blob", 224, true, 0.0f, 1.0f);

    // pretty_print_ncnn(resSnnMat);
    // SNN_LOGD("----------------------------------------------------");
    // pretty_print_ncnn(resNcnnMat);
    // SNN_LOGD("----------------------------------------------------");

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "add_3_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [29] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------4th Block Add_layer_2 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_10_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [08]
    // MaxPooling2D_3x3 pass[31].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------1st Conv2D_layer_4 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "leaky_re_lu_16_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [09]
    // Conv2D_3x3 pass[63].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------1st Conv2D_layer_5 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_11_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [10]
    // MaxPooling2D_3x3 pass[63].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------1st Conv2D_layer_6 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "leaky_re_lu_17_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [11]
    // Conv2D_3x3 pass[127].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------1st Block Add_layer_2 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_12_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [12]
    // MaxPooling2D_3x3 pass[127].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------activation 12 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "leaky_re_lu_18_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [13]
    // Conv2D_3x3 pass[255].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------activation 13 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "leaky_re_lu_19_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [14]
    // Conv2D_1x1 pass[63].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------activation 14 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "leaky_re_lu_21_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [15]
    // Conv2D_1x1 pass[31].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------activation 15 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "up_sampling2d_2_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [16]
    // UpSampling2D_1x1 pass[31].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Final Average_Layer_1 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "concatenate_2_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [17]
    // Concatenate_1x1 pass[95].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Final Average_Layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_51_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [62] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(79, mrtMode)).c_str(), false, 320);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("----------------------------- Conv Layer 34 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_35_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [63] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(319, mrtMode)).c_str(), false,
                         1280);
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("----------------------------- Conv Layer 35 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "flatten_blob", 224, true, 0.0f, 1.0f); snnMat = getSNNLayerText(formatString("%s/MobileNetV2/%s layer [31] Flatten_3x3
    // cpu layer.txt", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Flatten_Layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "global_average_pooling2d_blob", 224, true, 0.0f, 1.0f);
    ncnnMat = ncnnMat.reshape(1, 1, 1280);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [64] AveragePooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(319, mrtMode)).c_str(), false, 1280);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("----------------------------- Final Average Pooling 2D output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/MobileNetV2/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_36_blob", 224, true, 0.0f, 1.0f);
    ncnnMat = ncnnMat.reshape(1, 1, 2);
    snnMat  = getSNNLayer(formatString("%s/MobileNetV2/%s layer [65] Conv2D pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 2);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (1) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Output Blob output res: %d\n", ret);

    return 0;
}
