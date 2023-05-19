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
#ifndef __ANDROID__
#include <experimental/filesystem>
#endif

#include "matutil.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

#define NCNN_MODEL_NAME   "unet"
#define SNN_MODEL_NAME    "unet.json"
#define TEST_IMAGE        "test_image_unet_gray.png"

int main(int argc, char** argv) {
    SRAND(7767517);

    snn::MRTMode mrtMode = snn::MRTMode::SINGLE_PLANE;

#ifdef __ANDROID__
    std::string path = MODEL_DIR;
#else
    std::string path = std::experimental::filesystem::current_path();
    path += ("/" + std::string(MODEL_DIR));
#endif
    std::string ncnnModelName = path + formatString("/U-Net/%s", NCNN_MODEL_NAME);
    std::string ncnnImageName = formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE);;

    bool use_1ch_mrt = false;
    bool use_2ch_mrt = false;
    bool use_4ch_mrt = false;
    bool useHalfFP = false;
    bool stopOnMismatch = false;
    bool printMismatch = false;
    bool useVulkan = false;

    CLI::App app;
    app.add_flag("--use_1ch_mrt", use_1ch_mrt, "Use 1 render target (SINGLE_PLANE MRT)");
    app.add_flag("--use_2ch_mrt", use_2ch_mrt, "Use 2 render targets (DOUBLE_PLANE MRT)");
    app.add_flag("--use_4ch_mrt", use_4ch_mrt, "Use 4 render targets (QUAD_PLANE MRT)");
    app.add_flag("--use_half", useHalfFP, "Use half-precision floating point values (fp16)");
    app.add_flag("--stop_on_mismatch", stopOnMismatch, "Stop on results mismatch");
    app.add_flag("--print_mismatch", printMismatch, "Print results mismatch");
    app.add_flag("--use_vulkan", useVulkan, "Use Vulkan");
    CLI11_PARSE(app, argc, argv);
    CHECK_PLATFORM_SUPPORT(useVulkan)

    if (use_1ch_mrt) {
        mrtMode = snn::MRTMode::SINGLE_PLANE;
    } else if (use_2ch_mrt) {
        mrtMode = snn::MRTMode::DOUBLE_PLANE;
    } else if (use_4ch_mrt) {
        mrtMode = snn::MRTMode::QUAD_PLANE;
    } else {
        mrtMode = (snn::MRTMode) 0;
    }
    const double COMPARE_THRESHOLD = useHalfFP ? COMPARE_THRESHOLD_FP16 : COMPARE_THRESHOLD_FP32;

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "input_2_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false,
                         true);
    COMPARE_MAT1(ncnnMat, snnMat, "----Conv_layer_1 layer input res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_24_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_25_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [02] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "max_pooling2d_4_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [03] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 3 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_26_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 4 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_27_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 5 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "max_pooling2d_5_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [06] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "layer 6 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_28_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 7 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_29_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [08] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 8 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "max_pooling2d_6_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [09] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 9 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_30_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [10] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 10 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_31_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [11] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 11 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "max_pooling2d_7_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [12] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "layer 12 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_32_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [13] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(255, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 13 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_33_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [14] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(255, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 14 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "up_sampling2d_4_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [15] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [15] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 255).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "layer 15 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_34_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [16] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 16 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "concatenate_4_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 255).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 17 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_35_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [18] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 18 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_36_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [19] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 19 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "up_sampling2d_5_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [20] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [20] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 127).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 20 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_37_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 21 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "concatenate_5_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [22] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [22] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 127).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 22 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_38_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [23] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "layer 23 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_39_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [24] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "layer 24 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "up_sampling2d_6_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [25] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [25] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 63).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 25 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_40_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [26] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 26 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "concatenate_6_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [27] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [27] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 63).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "layer 27 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_41_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [28] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "layer 28 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_42_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [29] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 29 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "up_sampling2d_7_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [30] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [30] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 31).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "layer 30 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_43_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [31] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 31 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "concatenate_7_blob", 256, true, -1.0f, 1.0f, false);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [32] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/U-Net/%s layer [32] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 31).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 32 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_44_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [33] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 33 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_45_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [34] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 34 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_46_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [35] Conv2D pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 2);
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 35 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_47_blob", 256, true, -1.0f, 1.0f, false);
    snnMat  = getSNNLayer(formatString("%s/U-Net/%s layer [36] Conv2D pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 1);
    COMPARE_MAT1(ncnnMat, snnMat, "Layer 36 output res:");

    return 0;
}
