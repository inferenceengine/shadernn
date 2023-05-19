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

#include "layer/yolov3detectionoutput.h"
#include "matutil.h"
#include "snn/utils.h"

// Global namespace is polluted somewhere
#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

#define NCNN_MODEL_NAME "yolov3-tiny_dummy"
#define SNN_MODEL_NAME "yolov3_tiny_bb_layers.json"
#define TEST_IMAGE      "coco1_416.png"

#include "yolov3TinyTestCommon.cpp"

int main(int argc, char** argv) {
    SRAND(7767517);

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    snn::MRTMode mrtMode = snn::MRTMode::SINGLE_PLANE;

#ifdef __ANDROID__
    std::string path = MODEL_DIR;
#else
    std::string path = std::experimental::filesystem::current_path();
    path += ("/" + std::string(MODEL_DIR));
#endif
    std::string ncnnModelName = path + formatString("/Yolov3-tiny/%s", NCNN_MODEL_NAME);
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

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_12_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Conv_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "max_pooling2d_7_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [02] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Pooling_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_13_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block batch_norm_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "max_pooling2d_8_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [04] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_1 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_14_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "max_pooling2d_9_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [06] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_3 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_15_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "max_pooling2d_10_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [08] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_4 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_16_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [09] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_5 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "max_pooling2d_11_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [10] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Conv2D_layer_6 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_17_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [11] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "1st Block Add_layer_2 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "max_pooling2d_12_blob", 416);
    snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [12] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 12 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_18_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [13] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(255, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 13 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_19_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [14] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 14 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_21_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [15] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "activation 15 output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "up_sampling2d_2_blob", 416);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [16] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [16] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 31).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "Final Upsample layer output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "concatenate_2_blob", 416);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 95).c_str());
    }
    COMPARE_MAT1(ncnnMat, snnMat, "Final concate layer output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_20_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [18] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Final 18th conv2d output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "leaky_re_lu_22_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [19] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    COMPARE_MAT1(ncnnMat, snnMat, "Final 19th conv2d output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "conv2d_23_blob", 416);
    snnMat =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [20] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    COMPARE_MAT1(ncnnMat, snnMat, "Output 20th conv2d output res:");

    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "conv2d_26_blob", 416);
    snnMat =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    COMPARE_MAT1(ncnnMat, snnMat, "Output 21st conv2d output res:");

    auto ncnnMat13 = getNCNNLayer(ncnnModelName.c_str(),
                                  ncnnImageName.c_str(), "conv2d_23_blob", 416);
    auto ncnnMat26 = getNCNNLayer(ncnnModelName.c_str(),
                                  ncnnImageName.c_str(), "conv2d_26_blob", 416);
    auto snnMat13 =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [20] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    auto snnMat26 =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    ncnn::Mat matOut;
    ret = test_snn_input_v3tiny(ncnnMat13, ncnnMat26, matOut);
    if (ret && stopOnMismatch) {
        return ret;
    }

#if 0
    ncnnMat = getNCNNLayer(ncnnModelName.c_str(), ncnnImageName.c_str(),
                           "detection_out", 416);
    pretty_print_ncnn(ncnnMat);
#endif

    if (printLastLayer) {
        std::vector<cv::Mat> cvMats;
        auto cvMat13 = getCVMatFromDump(
            formatString("%s/Yolov3-tiny/%s layer [20] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
        auto cvMat26 = getCVMatFromDump(
            formatString("%s/Yolov3-tiny/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);

        cvMats.push_back(cvMat13);
        cvMats.push_back(cvMat26);
        getBoundBoxFromYOLO(cvMats, 416, 416, 416, 416, 0.35f, 0.45f);
    }

    return 0;
}
