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

#define NCNN_MODEL_NAME   "mobilenetv2_keras_dummy"
#define SNN_MODEL_NAME    "mobilenetv2_keras_layers.json"
#define TEST_IMAGE        "ant.png"

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
    const double COMPARE_THRESHOLD = useHalfFP ? COMPARE_THRESHOLD_FP16 : COMPARE_THRESHOLD_FP32;

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "input_1_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(formatString("%s/MobileNetV2/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         true, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "----Conv_layer_1 layer input res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_blob", 224, true, 0.0f, 1.0f);
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
    printf("-----------------------------Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_1_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [02] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_2_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [03] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "Depthwise_conv_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_3_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str(), false, 16);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_3 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_3_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(23, mrtMode)).c_str(), false, 96);
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_4 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "add_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [11] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(5, mrtMode)).c_str(), false, 24);
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_8_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [13] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(35, mrtMode)).c_str(), false, 144);

    COMPARE_MAT1(ncnnMat, snnMat, "DepthWise Conv 4 res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "add_1_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [18] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "add_2_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [22] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false, 32);
    COMPARE_MAT1(ncnnMat, snnMat, "3rd Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_14_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [24] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(47, mrtMode)).c_str(), false, 192);
    COMPARE_MAT1(ncnnMat, snnMat, "DepthWise Conv 6 res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_21_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv Layer 12 res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_16_blob", 224, true, 0.0f, 1.0f);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [27] DepthwiseConv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(95, mrtMode)).c_str(), false, 384);
    COMPARE_MAT1(ncnnMat, snnMat, "DepthWise Conv 7 res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_24_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [28] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv Layer 15 res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_21_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [25] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv Layer 15 res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "add_3_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [29] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false, 64);
    COMPARE_MAT1(ncnnMat, snnMat, "4th Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_51_blob", 224, true, 0.0f, 1.0f);
    snnMat =
        getSNNLayer(formatString("%s/MobileNetV2/%s layer [62] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(79, mrtMode)).c_str(), false, 320);
    COMPARE_MAT1(ncnnMat, snnMat, " Conv Layer 34 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_35_blob", 224, true, 0.0f, 1.0f);
    snnMat = getSNNLayer(formatString("%s/MobileNetV2/%s layer [63] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(319, mrtMode)).c_str(), false,
                         1280);
    COMPARE_MAT1(ncnnMat, snnMat, " Conv Layer 35 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "global_average_pooling2d_blob", 224, true, 0.0f, 1.0f);
    ncnnMat = ncnnMat.reshape(1, 1, 1280);
    snnMat  = getSNNLayer(
        formatString("%s/MobileNetV2/%s layer [64] AveragePooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(319, mrtMode)).c_str(), false, 1280);
    COMPARE_MAT1(ncnnMat, snnMat, " Final Average Pooling 2D output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_36_blob", 224, true, 0.0f, 1.0f);
    ncnnMat = ncnnMat.reshape(1, 1, 2);
    snnMat  = getSNNLayer(formatString("%s/MobileNetV2/%s layer [65] Conv2D pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 2);
    COMPARE_MAT1(ncnnMat, snnMat, "Output Blob output res:");

    return 0;
}
