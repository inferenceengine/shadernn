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

#include "snn/contextFactory.h"
#include "snn/imageTextureFactory.h"
#include "testutil.h"
// This header also brings JPEG library to link
#include "matutil.h"
#include "snn/utils.h"
#include "snn/colorUtils.h"
#include <vector>
#include <array>

// Global namespace is polluted somewhere
#ifdef Success
    #undef Success
#endif
#include "CLI/CLI.hpp"

static int testResize(snn::GpuContext * context, uint32_t w, uint32_t h, const std::string & format, int method, float scale, const std::vector<float> & values,
                      float mean, float norm, int printMismatch) {
    SNN_CHK(w > 0 && h > 0);

    snn::ColorFormat cf = snn::fromName(format.c_str());
    SNN_CHK(cf != snn::ColorFormat::NONE);
    snn::ColorFormatDesc cfd = snn::getColorFormatDesc(cf);
    uint32_t             c   = cfd.ch;

    std::vector<float> valuesAllCh(w * h * c, 0.0f);
    for (size_t i = 0, j = 0; i < values.size() && i < w * h; ++i, j += c) {
        for (size_t q = 0; q < c; ++q) { valuesAllCh[j + q] = values[i]; }
    }
    std::vector<uint8_t> buf = snn::convertColorBuffer(cf, valuesAllCh.data(), valuesAllCh.size());

    ncnn::ParamDict pd;
    float           invScale = 1.0 / scale;
    pd.set(0, method);
    pd.set(1, invScale);
    pd.set(2, invScale);
    pd.set(3, static_cast<int>(ceil(h * invScale)));
    pd.set(4, static_cast<int>(ceil(w * invScale)));

    std::shared_ptr<snn::ImageTexture> upImageTexture =
        snn::ImageTextureFactory::createImageTexture(context, std::array<uint32_t, 4> {w, h, 1U, 1U}, cf, buf.data());
    snn::ImageTexture & texture = *upImageTexture;

    ncnn::Mat ncnnMatOrig = hwc2NCNNMat(texture.at(0, 0), h, w, cf);
    ncnn::Mat ncnnMatResized;
    int       ret = test_layer_naive<ncnn::Interp>(ncnn::layer_to_index("Interp"), pd, std::vector<ncnn::Mat>(), ncnnMatOrig, ncnnMatResized, nullptr, 0);
    if (ret != 0) {
        std::cerr << "test_layer_naive failed" << std::endl;
        return ret;
    }
    std::vector<float> means_vec(4, mean);
    std::vector<float> norms_vec(4, norm);
    ncnnMatResized.substract_mean_normalize(means_vec.data(), norms_vec.data());

    texture.upload();
    texture.download();

    std::array<float, 4> means {mean, mean, mean, mean};
    std::array<float, 4> norms {norm, norm, norm, norm};
    // Dummy resize, just to test that resize initialization works correctly
    texture.resize(1.0f, 1.0f, means, norms);
    // Now real resize
    texture.resize(scale, scale, means, norms);

    int w1 = texture.getDims()[0];
    int h1 = texture.getDims()[1];

    ncnn::Mat snnMatResized = hwc2NCNNMat(texture.at(0, 0), h1, w1, cf);
    ret                     = CompareMat(ncnnMatResized, snnMatResized, 0.1);

    printf("\nimageTextureResize test res: %s for w=%d, h=%d, c=%d, type=%d, scale=%f\n", ret ? "FAILED" : "succeeded", w, h, c, method, scale);
    if (ret && printMismatch) {
        prettyPrintHWCBuf(texture.at(0, 0, 0, 0), h1, w1, c, cf);
        pretty_print_ncnn(ncnnMatResized, "SNN");
    }

    return ret;
}

int main(int argc, char ** argv) {
    uint32_t           w          = 2;
    uint32_t           h          = 2;
    std::string        format     = "RGBA32F";
    float              scale      = 0.5f;
    int                method     = 2;
    bool               useCompute = false;
    bool               useVulkan  = false;
    std::vector<float> values {1.0f, 2.0f, 3.0f, 4.0f};
    float              mean          = 0.0f;
    float              norm          = 1.0f;
    bool               printMismatch = false;

    CLI::App app;
    app.add_option("-W", w, "width");
    app.add_option("-H", h, "height");
    app.add_set("--format", format, snn::getAllColorNames(), "color format");
    app.add_option("-S", scale, "scale");
    app.add_flag("--use_compute", useCompute, "Use compute shader");
    app.add_set("--type", method, {1, 2}, "Interpolation method. 1 = Nearest, 2 = Bilinear");
    app.add_flag("--use_vulkan", useVulkan, "Use Vulkan");
    app.add_option("--values", values, "Values");
    app.add_option("--mean", mean, "Mean to subtract");
    app.add_option("--var", norm, "Variance to multiply");
    app.add_flag("--print_mismatch", printMismatch, "Print results mismatch");
    CLI11_PARSE(app, argc, argv);
    CHECK_PLATFORM_SUPPORT(useVulkan)

    printf("Using %s type\n", method == 1 ? "NEAREST" : "BILINEAR");
    printf("Using %s shader\n", useCompute ? "COMPUTE" : "FRAGMENT");

    snn::GpuContext * context = snn::createDefaultContext(useVulkan);
    testResize(context, w, h, format, method, scale, values, mean, norm, printMismatch);

    return 0;
}
