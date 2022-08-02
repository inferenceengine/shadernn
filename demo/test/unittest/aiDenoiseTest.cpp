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

int main() {
    SRAND(7767517);

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(),
                           formatString("%s/assets/images/empty_test_image.png", ASSETS_DIR).c_str(), "input_1_blob", 0, false);
    snnMat  = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [01] Conv2D pass[1]_input.dump", DUMP_DIR).c_str(), true);
    ret     = CompareMat(ncnnMat, snnMat, 0.1);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);
    // pretty_print_ncnn(snnMat);

    // compareNCNNLayerSNNLayer(formatString("%s/jsonModel/resnet18_cifar10_0223", ASSETS_DIR).c_str(), 1,
    //     formatString("%s/resnet18_cifar10_0223.json layer [01] Conv2D_7x7 pass[15]_input.dump", DUMP_DIR).c_str(),
    //     3, 64, 32, 7, 0, 2, true);

    ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(),
                           formatString("%s/assets/images/empty_test_image.png", ASSETS_DIR).c_str(), "conv2d_TanH_blob_idx_0", 0, false);
    snnMat  = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [01] Conv2D pass[1].dump", DUMP_DIR).c_str());
    ret     = CompareMat(ncnnMat, snnMat, 0.1);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(),
                           formatString("%s/assets/images/empty_test_image.png", ASSETS_DIR).c_str(), "conv2d_1_TanH_blob_idx_0", 0, false);
    snnMat  = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [02] Conv2D pass[3].dump", DUMP_DIR).c_str());
    ret     = CompareMat(ncnnMat, snnMat, 0.1);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_Layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(),
                           formatString("%s/assets/images/empty_test_image.png", ASSETS_DIR).c_str(), "conv2d_2_TanH_blob_idx_0", 0, false);
    snnMat  = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [03] Conv2D pass[7].dump", DUMP_DIR).c_str());
    ret     = CompareMat(ncnnMat, snnMat, 0.1);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_Layer_3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(),
                           formatString("%s/assets/images/empty_test_image.png", ASSETS_DIR).c_str(), "conv2d_3_TanH_blob", 0, false);
    snnMat  = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [04] Conv2D pass[7].dump", DUMP_DIR).c_str());
    ret     = CompareMat(ncnnMat, snnMat, 0.1);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_Layer_4 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(), formatString("%s/assets/images/empty_test_image.png",
    // ASSETS_DIR).c_str(), "conv2d_transpose_TanH_blob", 0, false); snnMat = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json
    // layer [05] Conv2DTranspose pass[3].dump", DUMP_DIR).c_str()); ncnnMat.create_like(snnMat); ncnnMat.fill(0.0);
    SNN_LOGD("Size of ncnnMat: %d, %d, %d. Total size: %d", ncnnMat.w, ncnnMat.h, ncnnMat.c, ncnnMat.total());
    // pretty_print_ncnn(snnMat);
    // pretty_print_ncnn(ncnnMat);
    // ret = CompareMat(ncnnMat, snnMat, 0.1);
    // if (ret) {
    //     pretty_print_ncnn(ncnnMat);
    //     pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Deconv_Layer_1 output res: %d\n", ret);

    // // ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(), formatString("%s/assets/images/empty_test_image.png",
    // ASSETS_DIR).c_str(), "concatenate_blob", 0, false);
    // // snnMat = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [06] Concatenate pass[11].dump", DUMP_DIR).c_str());
    // // ret = CompareMat(ncnnMat, snnMat, 0.1);
    // // if (ret) {
    // //     pretty_print_ncnn(ncnnMat);
    // //     pretty_print_ncnn(snnMat);
    // // }
    // // printf("-----------------------------Concat Layer 1 output res: %d\n", ret);

    // // ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(), formatString("%s/assets/images/empty_test_image.png",
    // ASSETS_DIR).c_str(), "conv2d_transpose_1_TanH_blob", 0, false); snnMat =
    // getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [07] Conv2DTranspose pass[3].dump", DUMP_DIR).c_str());
    // ncnnMat.create_like(snnMat);
    // ncnnMat.fill(0.0);
    SNN_LOGD("Size of ncnnMat: %d, %d, %d. Total size: %d", ncnnMat.w, ncnnMat.h, ncnnMat.c, ncnnMat.total());
    // ret = CompareMat(ncnnMat, snnMat, 0.1);
    // if (ret) {
    //     pretty_print_ncnn(ncnnMat);
    //     pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Deconv_Layer_2 output res: %d\n", ret);

    // // ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(), formatString("%s/assets/images/empty_test_image.png",
    // ASSETS_DIR).c_str(), "concatenate_1_blob", 0, false);
    // // snnMat = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [08] Concatenate pass[7].dump", DUMP_DIR).c_str());
    // // ret = CompareMat(ncnnMat, snnMat, 0.1);
    // // if (ret) {
    // //     pretty_print_ncnn(ncnnMat);
    // //     pretty_print_ncnn(snnMat);
    // // }
    // // printf("-----------------------------Concat Layer 2 output res: %d\n", ret);

    // // ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(), formatString("%s/assets/images/empty_test_image.png",
    // ASSETS_DIR).c_str(), "conv2d_transpose_2_TanH_blob", 0, false); snnMat =
    // getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [09] Conv2DTranspose pass[1].dump", DUMP_DIR).c_str());
    // ncnnMat.create_like(snnMat);
    // ncnnMat.fill(0.0);
    // ret = CompareMat(ncnnMat, snnMat, 0.1);
    // if (ret) {
    //     pretty_print_ncnn(ncnnMat);
    //     pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Deconv_Layer_3 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(), formatString("%s/assets/images/empty_test_image.png",
    // ASSETS_DIR).c_str(), "concatenate_2_blob", 0, false); snnMat = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer
    // [10] Concatenate pass[3].dump", DUMP_DIR).c_str()); ret = CompareMat(ncnnMat, snnMat, 0.1); if (ret) {
    //     pretty_print_ncnn(ncnnMat);
    //     pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Concat Layer 3 output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/jsonModel/aidenoise", ASSETS_DIR).c_str(), formatString("%s/assets/images/empty_test_image.png",
    // ASSETS_DIR).c_str(), "conv2d_transpose_3_blob", 0, false);
    snnMat = getSNNLayer(formatString("%s/eff_predenoise_20200330-210658_e635_mixloss1.h5.json layer [11] Conv2DTranspose pass[0].dump", DUMP_DIR).c_str());
    pretty_print_ncnn(snnMat);
    // ncnnMat.create_like(snnMat);
    // ncnnMat.fill(0.0);
    // ret = CompareMat(ncnnMat, snnMat, 0.1);
    // if (ret) {
    //     pretty_print_ncnn(ncnnMat);
    //     pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Deconv_Layer_4 output res: %d\n", ret);
    return 0;
}
