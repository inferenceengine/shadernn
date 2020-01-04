/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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
#ifndef SHADER_UNITTEST_H
#define SHADER_UNITTEST_H

#include "snn/glUtils.h"
#include "snn/utils.h"
#include "snn/snn.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <EGL/egl.h>

class ShaderUnitTest {
private:
    void _tic(timespec& ticker) { clock_gettime(CLOCK_MONOTONIC, &ticker); }

    double _toc(timespec& ticker) {
        struct timespec after;
        clock_gettime(CLOCK_MONOTONIC, &after);
        double us = (after.tv_sec - ticker.tv_sec) * 1000000.0f;

        return us + (after.tv_nsec - ticker.tv_nsec) / 1000.0f;
    }

    int calculateConvDim(int dim, int kernelSize, int stride, int padding) { return (int) ((dim - kernelSize + 2 * padding) / stride + 1); }

    int calculateDeconvDim(int dim, int kernelSize, int stride, int padding) { return (int) ((dim - 1) * stride - 2 * padding + kernelSize); }

public:
    void testImageTexture();

    std::string snnConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c, int outch,
                                     int kernel, int dilation, int stride, int pad, int bias, bool useOldShader, bool useBatchNorm,
                                     std::map<std::string, std::vector<float>>& batchNormalization);

    std::string snnPoolingTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, int poolingType, int padMode);

    std::string snnAddTestWithLayer(cv::Mat& inputMat, cv::Mat& inputMat2, int width, int height, int inChannels, string activation);

    std::string snnAddTestWithLayer2(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, string activation);

    std::string snnUpsampleTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int scale, int type);

    std::string snnConcateTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, string activation);

    std::string snnDepthConvTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c, int outch,
                                          int kernel, int dilation, int stride, int pad, int bias, bool useOldShader, bool useBatchNorm,
                                          std::map<std::string, std::vector<float>>& batchNormalization);

    std::string snnInstanceNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c,
                                             int outch, int kernel, int dilation, int stride, int pad, int bias, bool useOldShader, bool useBatchNorm,
                                             std::map<std::string, std::vector<float>>& batchNormalization);

    std::string snnPadTestWithLayer(cv::Mat& inputMat, int w, int h, int channels, int kernel, int stride, int type, float value);

    std::string snnBatchNormTestWithLayer(cv::Mat& inputMat, std::vector<cv::Mat>& inputWeights, std::vector<float>& inputBias, int w, int h, int c, int outch,
                                          int kernel, int dilation, int stride, int pad, int bias, bool useOldShader, bool useBatchNorm,
                                          std::map<std::string, std::vector<float>>& batchNormalization);

    std::string snnActivationTestWithLayer(cv::Mat& inputMat, int width, int height, int inChannels, int kernel, int stride, string activation,
                                           float leaky_val);
};

#endif // DEMOAPP_SHADERUNITTEST_H
