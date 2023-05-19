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

#define NCNN_MODEL_NAME   "mobilenetv2_pretrained_imagenet"
#define SNN_MODEL_NAME    "mobilenetv2_pretrained_imagenet.json"
#define TEST_IMAGE        "imagenet1.png"

int main(int argc, char** argv) {
    SRAND(7767517);

    snn::MRTMode mrtMode = snn::MRTMode::SINGLE_PLANE;

#ifdef __ANDROID__
    std::string path = MODEL_DIR;
#else
    std::string path = std::experimental::filesystem::current_path();
    path += ("/" + std::string(MODEL_DIR));
#endif
    std::string ncnnModelName = path + formatString("/MobileNetV2/%s", NCNN_MODEL_NAME);
    std::string ncnnImageName = formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE);;

    bool use_1ch_mrt = false;
    bool use_2ch_mrt = false;
    bool use_4ch_mrt = false;
    bool useHalfFP = false;
    bool stopOnMismatch = false;
    bool printMismatch = false;
    bool printLastLayer = false;
    bool useVulkan = false;

    CLI::App app;
    app.add_flag("--use_1ch_mrt", use_1ch_mrt, "Use 1 render target (SINGLE_PLANE MRT)");
    app.add_flag("--use_2ch_mrt", use_2ch_mrt, "Use 2 render targets (DOUBLE_PLANE MRT)");
    app.add_flag("--use_4ch_mrt", use_4ch_mrt, "Use 4 render targets (QUAD_PLANE MRT)");
    app.add_flag("--use_half", useHalfFP, "Use half-precision floating point values (fp16)");
    app.add_flag("--stop_on_mismatch", stopOnMismatch, "Stop on results mismatch");
    app.add_flag("--print_mismatch", printMismatch, "Print results mismatch");
    app.add_flag("--print_last_layer", printLastLayer, "Print last layer");
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
    // TODO: investigate why so big error
    const double COMPARE_THRESHOLD = useHalfFP ? 0.7 : 0.2;

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "input_1_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(formatString("%s/MobileNetV2/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         true, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "----Conv_layer_1 layer input res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "Conv1_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        if (printMismatch) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat, "SNN");
        }

        printf("Testing if padding is the problem\n");
        ncnn::Mat centerPixelsNcnn, centerPixelsSnn;
        ncnn::copy_cut_border(ncnnMat, centerPixelsNcnn, 1, 1, 1, 1);
        ncnn::copy_cut_border(snnMat, centerPixelsSnn, 1, 1, 1, 1);

        COMPARE_MAT1(centerPixelsNcnn, centerPixelsSnn, "Conv_Layer_1 center output res:");
    }
    printf("-----------------------------1st Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "expanded_conv_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [02] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "1st DepthWise Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "expanded_conv_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str(), false, 16);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_1_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    COMPARE_MAT1(ncnnMat, snnMat, "3rd Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_1_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [05] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 96);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [05] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 23).c_str(), false, 96);
    }
    COMPARE_MAT1(ncnnMat, snnMat, "1st ZeroPadding2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_1_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [06] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    COMPARE_MAT1(ncnnMat, snnMat, "DepthwiseConv2D after 1st Zero Padding  output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_1_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(5, mrtMode)).c_str(), false, 24);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv2D after 1st Padding res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_2_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [09] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_2_add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [11] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(5, mrtMode)).c_str(), false, 24);
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_3_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [12] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);
    COMPARE_MAT1(ncnnMat, snnMat, "After 1st Add Block Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_3_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [13] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 144);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [13] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 35).c_str(), false, 144);
    }
    COMPARE_MAT1(ncnnMat, snnMat, "2nd Zero Padding res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_3_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [14] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);
    COMPARE_MAT1(ncnnMat, snnMat, "After 2nd Zero Padding DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_3_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [15] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "After 2nd Zero Padding Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_4_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [16] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd Block Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_4_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [17] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_4_add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [19] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_5_add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [23] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "3rd Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_6_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [24] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    COMPARE_MAT1(ncnnMat, snnMat, "The Conv2d before 3rd padding res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_6_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 192);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 47).c_str(), false, 192);
    }
    COMPARE_MAT1(ncnnMat, snnMat, "3rd Zero Padding res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_6_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [26] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    COMPARE_MAT1(ncnnMat, snnMat, "The DepthwiseConv2D after 3rd Padding output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_6_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [27] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "The Conv2d after 3rd padding res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_7_expand_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [28] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    COMPARE_MAT1(ncnnMat, snnMat, "4th Block Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_7_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [29] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    COMPARE_MAT1(ncnnMat, snnMat, "4th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_7_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [31] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "4th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_8_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [33] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    COMPARE_MAT1(ncnnMat, snnMat, "5th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_8_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [35] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "5th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_9_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [37] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    COMPARE_MAT1(ncnnMat, snnMat, "6th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_9_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [39] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "6th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_10_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [41] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    COMPARE_MAT1(ncnnMat, snnMat, "After 6th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_10_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [42] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    COMPARE_MAT1(ncnnMat, snnMat, "After 6th Block Conv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_11_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [44] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(143, mrtMode)).c_str(), false, 576);
    COMPARE_MAT1(ncnnMat, snnMat, "7th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_11_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [46] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    COMPARE_MAT1(ncnnMat, snnMat, "7th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_12_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [48] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(143, mrtMode)).c_str(), false, 576);
    COMPARE_MAT1(ncnnMat, snnMat, "8th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_12_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [50] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    COMPARE_MAT1(ncnnMat, snnMat, "8th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_13_pad_blob", 224, true, 0.0f, 1.0f);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [52] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str(), false, 576);
    } else {
        snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [52] ZeroPadding2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 143).c_str(), false, 576);
    }
    COMPARE_MAT1(ncnnMat, snnMat, "4th Zero Padding res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_13_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [53] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(143, mrtMode)).c_str(), false, 576);
    COMPARE_MAT1(ncnnMat, snnMat, "Block DepthwiseConv2D after 4th Padding output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_13_project_BN_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [54] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(39, mrtMode)).c_str(), false, 160);
    COMPARE_MAT1(ncnnMat, snnMat, "The Conv2d after 4th padding res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_14_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [56] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(239, mrtMode)).c_str(), false, 960);
    COMPARE_MAT1(ncnnMat, snnMat, "9th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_14_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [58] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(39, mrtMode)).c_str(), false, 160);
    COMPARE_MAT1(ncnnMat, snnMat, "9th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_15_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [60] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(239, mrtMode)).c_str(), false, 960);
    COMPARE_MAT1(ncnnMat, snnMat, "10th Block DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_15_add_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [62] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(39, mrtMode)).c_str(), false, 160);
    COMPARE_MAT1(ncnnMat, snnMat, "10th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "block_16_depthwise_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [64] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(239, mrtMode)).c_str(), false, 960);
    COMPARE_MAT1(ncnnMat, snnMat, "Last DepthwiseConv2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "out_relu_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [66] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(319, mrtMode)).c_str(), false,
                         1280);
    COMPARE_MAT1(ncnnMat, snnMat, "Last Conv2D output res:");

    return 0;
}
