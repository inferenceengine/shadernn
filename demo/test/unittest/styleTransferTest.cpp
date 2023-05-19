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

#define NCNN_MODEL_NAME "candy-9_simplified-opt"
#define SNN_MODEL_NAME  "candy-9_simplified.json"
#define TEST_IMAGE      "ant.png"

int main(int argc, char** argv) {
    SRAND(7767517);

    snn::MRTMode mrtMode = snn::MRTMode::SINGLE_PLANE;

#ifdef __ANDROID__
    std::string path = MODEL_DIR;
#else
    std::string path = std::experimental::filesystem::current_path();
    path += ("/" + std::string(MODEL_DIR));
#endif
    std::string ncnnModelName = path + formatString("/StyleTransfer/%s", NCNN_MODEL_NAME);
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
    const double COMPARE_THRESHOLD = useHalfFP ? 2.0 : COMPARE_THRESHOLD_FP32;

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "input1", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [01] Conv2D pass[%d]_input.dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(),
                         true);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_1 layer input res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "64", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "66", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [02] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "68", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "70", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [04] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "InstanceNorm_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "72", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_3 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "74", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [06] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "InstanceNorm_3 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "76", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block1 Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "78", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [08] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block1 InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "80", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [09] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block1 Conv_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "81", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [10] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block1 InstanceNorm_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "82", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [11] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block1 Add output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "84", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [12] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block2 Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "86", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [13] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block2 InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "88", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [14] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block2 Conv_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "89", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [15] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block2 InstanceNorm_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "90", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [16] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block2 Add output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "92", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [17] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block3 Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "94", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [18] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block3 InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "96", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [19] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block3 Conv_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "97", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [20] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block3 InstanceNorm_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "98", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [21] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block3 Add output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "100", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [22] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block4 Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "102", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [23] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block4 InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "104", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [24] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block4 Conv_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "105", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [25] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block4 InstanceNorm_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "106", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [26] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block4 Add output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "108", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [27] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block5 Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "110", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [28] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block5 InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "112", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [29] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block5 Conv_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "113", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [30] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block5 InstanceNorm_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "114", 224, false);
    snnMat  = getSNNLayer(formatString("%s/StyleTransfer/%s layer [31] Add pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "Block5 Add output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "139", 224, false);
    snnMat = getSNNLayer(formatString("%s/StyleTransfer/%s layer [32] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str(),
                         false);
    COMPARE_MAT1(ncnnMat, snnMat, "1st UpSample output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "141", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [33] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "1st UpSample Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "143", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [34] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "1st UpSample InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "168", 224, false);
    snnMat = getSNNLayer(formatString("%s/StyleTransfer/%s layer [35] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str(),
                         false);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd UpSample output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "170", 224, false);
    snnMat =
        getSNNLayer(formatString("%s/StyleTransfer/%s layer [36] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd UpSample Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "172", 224, false);
    snnMat  = getSNNLayer(
        formatString("%s/StyleTransfer/%s layer [37] InstanceNormalization pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str(), false);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd UpSample InstanceNorm_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(),
                           ncnnImageName.c_str(), "output1", 224, false);
    snnMat = getSNNLayer(formatString("%s/StyleTransfer/%s layer [38] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(0, mrtMode)).c_str(), true);
    COMPARE_MAT1(ncnnMat, snnMat, "2nd UpSample Conv_layer_2 output res:");

    return 0;
}
