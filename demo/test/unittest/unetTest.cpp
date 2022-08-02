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

#define NCNN_MODEL_NAME   "unet_dummy"
#define SNN_MODEL_NAME    "unet_dummy.json"
#define TEST_IMAGE        "test_image_unet_gray.png"
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

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "input_4_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false,
                         true);
    ret     = CompareMat(ncnnMat, snnMat);
    if (ret) {
        // pretty_print_ncnn(ncnnMat.channel(0));
        // pretty_print_ncnn(snnMat.channel(0));
    }
    printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);
    ////pretty_print_ncnn(snnMat);

    // compareNCNNLayerSNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), 1,
    //     formatString("%s/U-Net/%s layer [01] Conv2D pass[15]_input.dump", DUMP_DIR,SNN_MODEL_NAME).c_str(),
    //     3, 64, 32, 7, 0, 2, true);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_73_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_74_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [02] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_13_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [03] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_75_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 4 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_76_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 5 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_14_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [06] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 6 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_77_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 7 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_78_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [08] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 8 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_15_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [09] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 9 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_79_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [10] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 10 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_80_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [11] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 11 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "max_pooling2d_16_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [12] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 12 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_81_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [13] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(255, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 13 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_82_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [14] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(255, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 14 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "up_sampling2d_13_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [15] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [15] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 255).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 15 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_83_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [16] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat.channel(0));
        // pretty_print_ncnn(snnMat.channel(0));
    }
    printf("-----------------------------Layer 16 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString("%s/assets/images/%s", ASSETS_DIR,
    // TEST_IMAGE).c_str(), "flatten_blob", 256, true, -1.0f, 1.0f, false); snnMat = getSNNLayerText(formatString("%s/U-Net/%s layer [31] Flatten cpu
    // layer.txt", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Flatten_Layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "concatenate_13_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 255).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 17 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_84_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [18] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 18 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_85_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [19] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 19 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "up_sampling2d_14_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [20] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [20] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 127).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 20 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_86_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------:layer 21 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "concatenate_14_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [22] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [22] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 127).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 22 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_87_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [23] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 23 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_88_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [24] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 24 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "up_sampling2d_15_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [25] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [25] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 63).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 25 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_89_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [26] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 26 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "concatenate_15_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [27] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [27] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 63).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 27 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_90_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [28] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 28 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_91_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [29] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 29 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "up_sampling2d_16_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [30] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [30] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 31).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------layer 30 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_92_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [31] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 31 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "concatenate_16_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [32] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [32] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 31).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 32 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_93_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [33] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 33 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_94_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [34] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (1) {
        // pretty_print_ncnn(ncnnMat, 32);
        // pretty_print_ncnn(snnMat, 32);
    }
    printf("-----------------------------Layer 34 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_95_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [35] Conv2D pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 2);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (1) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 35 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/U-Net/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                           formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE).c_str(), "conv2d_96_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [36] Conv2D pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 1);
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (1) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Layer 36 output res: %d\n", ret);

    return 0;
}
