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

// #define NCNN_MODEL_NAME "resnet18_cifar10_0223"
// #define SNN_MODEL_NAME "resnet18_cifar10_0223.json"
#define NCNN_MODEL_NAME "resnet18_cifar10"
#define SNN_MODEL_NAME  "resnet18_cifar10.json"
#define TEST_IMAGE      "cifar_test.png"

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

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "input_1_blob", 32);
    snnMat =
        getSNNLayer(formatString("%s/Resnet18/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), true);
    ret = CompareMat(ncnnMat, snnMat, 0.1);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);
    // pretty_print_ncnn(snnMat);

    // compareNCNNLayerSNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), 1,
    //     formatString("%s/Resnet18/%s layer [01] Conv2D_7x7 pass[15]_input.dump", DUMP_DIR,SNN_MODEL_NAME).c_str(),
    //     3, 64, 32, 7, 0, 2, true);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [02] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Pooling_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_1_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Conv_Layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_1_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block batch_norm_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_1_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [05] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_2_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [06] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_3_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_3_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [08] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_4_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [09] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_4 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "batch_normalization_5_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [10] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_5 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_7_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [11] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_6 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_5_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [12] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_12_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [23] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 12 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_13_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [26] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 13 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_14_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [27] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 14 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "activation_15_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [29] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 15 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "average_pooling2d_blob", 32);
    snnMat =
        getSNNLayer(formatString("%s/Resnet18/%s layer [30] AveragePooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Final Average_Layer_1 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/images/%s", ASSETS_DIR,
    // TEST_IMAGE).c_str(), "flatten_blob", 32); snnMat = getSNNLayerText(formatString("%s/Resnet18/%s layer [31] Flatten cpu layer.txt",
    // DUMP_DIR,SNN_MODEL_NAME).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     pretty_print_ncnn(ncnnMat);
    //     pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Flatten_Layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "dense_Softmax_blob", 32);
    snnMat  = getSNNLayerText(formatString("%s/Resnet18/%s layer [32] Dense cpu layer.txt", DUMP_DIR, SNN_MODEL_NAME).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (1) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Output Blob output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Resnet18/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s",
    // ASSETS_DIR, TEST_IMAGE).c_str(), "dense_blob", 32); auto snnMat4 = getSNNLayer(formatString("%s/Resnet18/%s layer [32] Dense pass[0].dump", DUMP_DIR,
    // SNN_MODEL_NAME).c_str()); auto snnMat1 = snnMat4.channel_range(0,1); snnMat = snnMat1.reshape(snnMat1.w * snnMat1.h * snnMat1.c); ret =
    // CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (1) {
    //     pretty_print_ncnn(ncnnMat);
    //     pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Output Blob output res: %d\n", ret);

    return 0;
}
