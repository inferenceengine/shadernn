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
#pragma once
#include "snn/snn.h"
#include "snn/imageTexture.h"
#include "snn/utils.h"
#include <map>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <opencv2/core.hpp>

class ShaderUnitTest {
private:
    snn::GpuContext* context;

    bool useVulkan() const {
        return context->backendType == snn::GpuBackendType::VULKAN;
    }

    static void _tic(timespec& ticker);

    static double _toc(timespec& ticker);

    int calculateConvDim(int dim, int kernelSize, int stride, int padding) { return (int) ((dim - kernelSize + 2 * padding) / stride + 1); }

    int calculateDeconvDim(int dim, int kernelSize, int stride, int padding) { return (int) ((dim - 1) * stride - 2 * padding + kernelSize); }

    std::shared_ptr<snn::ImageTexture> createInputImgTxt(const std::array<uint32_t, 4>& dims, snn::ColorFormat colorFormat, const float* pixels);

    snn::ImageTextureArray createInputImgTxt(float* dest, int width, int height, int inChannels, bool fp16 = false);

    snn::ImageTextureArray createOutputImgTxt(int outWidth, int outHeight, int outChannels, bool fp16 = false);

public:
    ShaderUnitTest(snn::GpuBackendType backend);

    ~ShaderUnitTest() = default;

    void testImageTexture();
    void testImageTexture(cv::Mat& inputMat, int width, int height, int inChannels);

    std::string snnConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c, int outch,
                                     int kernel, int dilation, int stride, int pad, bool useCompute, snn::MRTMode mrtMode, bool useBatchNorm,
                                     std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput = true, bool fp16 = false);

    std::string snnDenseTestWithLayer(cv::Mat& inputMat, std::vector<std::vector<float>>& inputWeights, std::vector<float>& inputBias, int w, int h, int c,
        int outch, bool dumpOutput = true);

    std::string snnPoolingTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, int poolingType, int padMode,
        bool dumpOutput = true);

    std::string snnAddTestWithLayer2(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, const std::string& activation,
        bool dumpOutput = true);

    std::string snnMultiInputsTestWithLayer(cv::Mat& inputMat1, cv::Mat& inputMat2, int width, int height, int inChannels, bool dumpOutput = true);

    std::string snnUpsampleTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int scale, int type, bool useCompute = true,
        bool dumpOutput = true);

    std::string snnConcateTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, const std::string& activation,
        bool dumpOutput = true);

    std::string snnDepthConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c, int outch,
                                          int kernel, int dilation, int stride, int pad, int bias, bool useCompute, bool useBatchNorm,
                                          std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput = true);

    std::string snnInstanceNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c,
                                             int outch, int kernel, int dilation, int stride, int pad, int bias, bool useCompute, bool useBatchNorm,
                                             std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput = true);

    std::string snnPadTestWithLayer(cv::Mat& inputMat, int w, int h, int channels, int kernel, int stride, int type, float value, bool dumpOutput = true);

    std::string snnBatchNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c, int outch,
                                          int kernel, int dilation, int stride, int pad, int bias, bool useCompute, bool useBatchNorm,
                                          std::map<std::string, std::vector<float>>& batchNormalization, bool dumpOutput = true);

    std::string snnActivationTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, const std::string& activation,
        float leaky_val, bool dumpOutput = true);

    std::string snnFlattenTestWithLayer(cv::Mat& inputMat, int w, int h, int channels, bool dumpOutput = true);
};
