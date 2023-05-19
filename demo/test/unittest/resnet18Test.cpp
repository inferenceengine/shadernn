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

#ifndef NCNN_MODEL_NAME
    #define NCNN_MODEL_NAME "resnet18_cifar10_0223"
#endif
#ifndef SNN_MODEL_NAME
    #define SNN_MODEL_NAME "resnet18_cifar10_0223_layers.json"
#endif
#define TEST_IMAGE      "cifar_test.png"

int main(int argc, char** argv) {
    SRAND(7767517);

    snn::MRTMode mrtMode = snn::MRTMode::SINGLE_PLANE;

#ifdef __ANDROID__
    std::string path = MODEL_DIR;
#else
    std::string path = std::experimental::filesystem::current_path();
    path += ("/" + std::string(MODEL_DIR));
#endif
    std::string ncnnModelName = path + formatString("/Resnet18/%s", NCNN_MODEL_NAME);
    std::string ncnnImageName = formatString("%s/assets/images/%s", ASSETS_DIR, TEST_IMAGE);

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
                           ncnnImageName.c_str(), "input_1_blob", 32);
    snnMat =
        getSNNLayer(formatString("%s/Resnet18/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), true);
    COMPARE_MAT1(ncnnMat, snnMat, "----Conv_layer_1 layer input res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "max_pooling2d_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [02] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Pooling_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_1_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Conv_Layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_1_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [04] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block batch_norm_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_1_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [05] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_2_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [06] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_3_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_3 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_3_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [08] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_4_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [09] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_4 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "batch_normalization_5_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [10] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_5 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "conv2d_7_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [11] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_6 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_5_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [12] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_12_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [23] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 12 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_13_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [26] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 13 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_14_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [27] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 14 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "activation_15_blob", 32);
    snnMat  = getSNNLayer(formatString("%s/Resnet18/%s layer [29] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 15 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "average_pooling2d_blob", 32);
    snnMat =
        getSNNLayer(formatString("%s/Resnet18/%s layer [30] AveragePooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Final Average_Layer_1 output res:");
#if 0
    // This causes weird error in ncnn library:
    // FATAL ERROR! pool allocator destroyed too early
    // and later call into ncnn causes a crash.
    // Probably bug in ncnn
    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), formatString("%s/assets/images/%s",
         ASSETS_DIR, TEST_IMAGE).c_str(), "flatten_blob", 32);
    snnMat = getSNNLayer(formatString("%s/Resnet18/%s layer [31] Flatten pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 0, true);
    COMPARE_MAT1(ncnnMat, snnMat, "Flatten_Layer_1 output res:");
#endif

    // Rebuild core with -DDUMP_RESULTS_TXT
    // and uncomment this to test output in text format
#if 0
    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                        ncnnImageName.c_str(), "dense_Softmax_blob", 32);
    snnMat  = getSNNLayerText(formatString("%s/Resnet18/%s layer [32] Dense cpu layer.txt", DUMP_DIR, SNN_MODEL_NAME).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Output Blob output res:");
#endif

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), formatString("%s/assets/images/%s",
        ASSETS_DIR, TEST_IMAGE).c_str(), "dense_Softmax_blob", 32);
    snnMat = getSNNLayer(formatString("%s/Resnet18/%s layer [32] Dense pass[0].dump", DUMP_DIR, SNN_MODEL_NAME).c_str(), false, 0, true);
    COMPARE_MAT1(ncnnMat, snnMat, "Output Blob output res:");
    if (printLastLayer) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat, "SNN");
    }

    return 0;
}
