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
// Help function for unit test with MAT
#ifndef __MATUTIL_H__
#define __MATUTIL_H__

#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/padding.h"
#include "layer/pooling.h"
#include "layer/interp.h"
#include "layer/batchnorm.h"

#include "testutil.h"
#include "snn/color.h"
#include "snn/snn.h"
#include "snn/utils.h"
#include "mat.h"
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdarg.h>

using snn::formatString;

int getPassIndex(int idex, snn::MRTMode mrt);

void printC4Buffer(float *buffer, int input_h, int input_w, int input_c, std::ostream& stream = std::cout);

void printHWC(float *buffer, int input_h, int input_w, int input_c, std::ostream& stream = std::cout);

void hwcToC4(float *buffer, int input_h, int input_w, int input_c, float* dst);

void c4ToHWC(float *buffer, int input_h, int input_w, int input_c4, float* dst);

ncnn::Mat hwc2NCNNMat(float *buffer, int input_h, int input_w, int input_c);

ncnn::Mat hwc2NCNNMat(const uint8_t *raw_buf, int h, int w, snn::ColorFormat colorFormat);

void ncnn2HWC(ncnn::Mat padA, float* dest);

template<typename Functor>
void initNCNNMat(ncnn::Mat padA, Functor functor)
{
    for (int q = 0; q < padA.c; q++) {
        float* ptr = padA.channel(q);
        for (int y = 0; y < padA.h; y++) {
            for (int x = 0; x < padA.w; x++) {
                ptr[x] = functor(y, x, q);
            }
            ptr += padA.w;
        }
    }
}

void pretty_print_ncnn(const ncnn::Mat& m, const char* header = "NCNN");

void pretty_print_cvmat(const std::vector<cv::Mat> m);

cv::Mat sliceMat(cv::Mat L, int dim, std::vector<int> _sz);

void print_3d_cvmat(cv::Mat outputMat);

void print_3d_cvmat_byte(cv::Mat outputMat);

ncnn::Mat CVMat2NCNNMat(cv::Mat output);

cv::Mat NCNNMat2CVMat(ncnn::Mat padA);

void ncnnToVec(const ncnn::Mat& a, std::vector<float>& b);

void ncnnToMat(const ncnn::Mat& a, std::vector<std::vector<float>>& b);

void pretty_print_vec(const std::vector<float>& vec);

void pretty_print_mat(const std::vector<std::vector<float>>& mat);

ncnn::Mat getNCNNLayer(const std::string& modelName, const std::string& inputImage, const std::string& outputName, int target_size = 0,
    bool scale = true, float min = -1.0f, float max = 1.0f, bool color = true);

ncnn::Mat getSNNLayer(const std::string& inputName, bool force3Channels = false, int actualChanels = 0, bool flat = false, int forceLen = -1);

cv::Mat getCVMatFromDump(const std::string& inputName, bool force3Channels = false, int actualChanels = 0);

ncnn::Mat getSNNLayerText(const std::string& inputName);

std::vector<ncnn::Mat> getWeigitBiasFromNCNN(const std::string& modelName, int layerId);

std::vector<ncnn::Mat> getDepthwiseWeigitBiasFromNCNN(const std::string& modelName, int layerId);

std::vector<ncnn::Mat> getBatchNormFromNCNN(const std::string& modelName, int layerId);

ncnn::Mat customizeNCNNLayer(const std::string& modelName, const std::string& inputImage, const std::string& layerType,
    const std::string& layerInputFile, int target_size, int inputChannels, int outputChannels, int kernelSize, int padding, int stride, int layerId);
#endif  //__MATUTIL_H__
