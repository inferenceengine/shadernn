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

#include "cpu.h"
#include "net.h"

#include "testutil.h"
#include "snn/utils.h"
#include "snn/snn.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdarg.h>
#include <iomanip>

using namespace std;
using namespace cv;

int getPassIndex(int idex, snn::MRTMode mrt) {
    int depth = 0;
    switch (mrt) {
    case snn::MRTMode::SINGLE_PLANE:
        return idex;

    case snn::MRTMode::DOUBLE_PLANE:
        depth = (idex + 1) * 4;
        return DIV_AND_ROUND_UP(depth, 8) - 1;

    case snn::MRTMode::QUAD_PLANE:
        depth = (idex + 1) * 4;
        return DIV_AND_ROUND_UP(depth, 16) - 1;

    default:
        return 0;
    }
}

void pretty_print_ncnn(const ncnn::Mat& m) {
    printf("------NCNN Mat--%d--%d--%d-------\n", m.w, m.h, m.c);
    for (int q = 0; q < m.c; q++) {
        const float* ptr = m.channel(q);
        for (int y = 0; y < m.h; y++) {
            for (int x = 0; x < m.w; x++) {
                printf("%f, ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("\n----------%d--------------\n", q);
    }
}

void pretty_print_cvmat(const std::vector<cv::Mat> m) {
    printf("-------CV Mat-%d--%d--%zu-------\n", m[0].size[0], m[0].size[1], m.size());
    for (size_t q = 0; q < m.size(); q++) {
        std::cout << "M = " << std::endl << " " << m[q] << std::endl << std::endl;
    }
}

cv::Mat sliceMat(cv::Mat L, int dim, std::vector<int> _sz) {
    (void) dim;
    cv::Mat M(L.dims - 1, std::vector<int>(_sz.begin() + 1, _sz.end()).data(), CV_32FC1, L.data + L.step[0] * 0);
    return M;
}

std::string formatString(const char* format, ...) {
    va_list args;
    va_start(args, format);
    thread_local static std::vector<char> buf1(4096);
    while (std::vsnprintf(buf1.data(), buf1.size(), format, args) > (int) buf1.size()) {
        buf1.resize(buf1.size() * 2);
    }
    va_end(args);
    return buf1.data();
}

void print_3d_cvmat(cv::Mat outputMat) {
    printf("---------------output of opencv 3d mat w first----------- \n");
    for (int k = 0; k < outputMat.size[2]; k++) {
        for (int i = 0; i < outputMat.size[0]; i++) {
            for (int j = 0; j < outputMat.size[1]; j++) {
                // std::cout << "M(" << i << ", " << j << ", " << k << "): " << outputMat.at<float>(i,j,k) << ",";
                std::cout << std::setw(7) << outputMat.at<float>(i, j, k) << ",";
            }
            // std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
            std::cout << std::endl;
        }
        std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }
    std::cout << std::endl;
}

void print_3d_cvmat_byte(cv::Mat outputMat) {
    printf("---------------output of opencv 3d mat w first----%d-%d-%d----- \n", outputMat.size[0], outputMat.size[1], outputMat.size[2]);
    for (int k = 0; k < outputMat.size[2]; k++) {
        for (int i = 0; i < outputMat.size[0]; i++) {
            for (int j = 0; j < outputMat.size[1]; j++) {
                std::cout << "M(" << i << ", " << j << ", " << k << "): " << std::to_string(outputMat.at<unsigned char>(i, j, k)) << ",";
                // std::cout << outputMat.at<unsigned char>(i,j,k) <<  ",";
            }
            // std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
            std::cout << std::endl;
        }
        std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }
    std::cout << std::endl;
}

ncnn::Mat CVMat2NCNNMat(cv::Mat output) {
    printf("%s:%d-------CV Mat-%d--%d--%d-------\n", __FUNCTION__, __LINE__, output.size[0], output.size[1], output.size[2]);
    ncnn::Mat snnOutput(output.size[1], output.size[0], output.size[2]);
    // for (int p=0; p<snnOutput.c; p++)
    // {
    //     memcpy(snnOutput.channel(p), (const unsigned char*)output.data + p * snnOutput.w * snnOutput.h * sizeof(float), snnOutput.w * snnOutput.h *
    //     sizeof(float));
    // }

    for (int q = 0; q < snnOutput.c; q++) {
        float* ptr = snnOutput.channel(q);
        for (int y = 0; y < snnOutput.h; y++) {
            for (int x = 0; x < snnOutput.w; x++) {
                ptr[x] = output.at<float>(y, x, q);
            }
            ptr += snnOutput.w;
        }
    }
    return snnOutput;
}

cv::Mat NCNNMat2CVMat(ncnn::Mat padA) {
    int iw       = padA.w;
    int ih       = padA.h;
    int ic       = padA.c;
    int size[3] = {ih, iw, ic};
    // cv::Mat inputMat(3, size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat inputMat(3, size, CV_32FC1);
    // memcpy((uchar*)inputMat.data, padA.data, padA.w * padA.h * padA.c * sizeof(float));
    for (int q = 0; q < padA.c; q++) {
        const float* ptr = padA.channel(q);
        for (int y = 0; y < padA.h; y++) {
            for (int x = 0; x < padA.w; x++) {
                inputMat.at<float>(y, x, q) = ptr[x];
            }
            ptr += padA.w;
        }
    }
    return inputMat;
}

void ncnnToVec(const ncnn::Mat& a, std::vector<float>& b) {
    for (int i = 0; i < a.c; i++) {
        const float* ptr = a.channel(i);
        for (int j = 0; j < a.h; j++) {
            for (int k = 0; k < a.w; k++) {
                b.push_back(ptr[k]);
            }
            ptr += a.w;
        }
    }
}

void ncnnToMat(const ncnn::Mat& a, std::vector<std::vector<float>>& b) {
    for (int i = 0; i < a.c; i++) {
        const float* ptr = a.channel(i);
        for (int j = 0; j < a.h; j++) {
            std::vector<float> c;
            for (int k = 0; k < a.w; k++) {
                c.push_back(ptr[k]);
            }
            ptr += a.w;
            b.push_back(c);
        }
    }
}

void pretty_print_vec(const std::vector<float>& vec) {
    printf("\n---------- Float Vector -- %ld ----------\n", vec.size());
    for (auto element : vec) {
        printf("%f\t", element);
    }
    printf("\n----------------------------------------\n");
}

void pretty_print_mat(const std::vector<std::vector<float>>& mat) {
    printf("\n---------- Float Matrix -- %ld -- %ld ----------\n", mat.size(), mat.at(0).size());
    for (auto row : mat) {
        for (auto element : row) {
            printf("%f\t", element);
        }
        printf("\n");
    }
    printf("\n--------------------------------------------\n");
}

ncnn::Mat getNCNNLayer(string modelName, string inputImage, string outputName, int target_size = 0, bool scale = true, float min = -1.0f, float max = 1.0f,
                       bool color = true) {
    // int g_loop_count = 4;
    // bool g_enable_cooling_down = true;

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    int num_threads = ncnn::get_cpu_count();
    int powersave   = 0;
    // int gpu_device = -1;

    // default option
    ncnn::Option opt;
    opt.lightmode   = true;
    opt.num_threads = 1;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    // fprintf(stderr, "loop_count = %d\n", g_loop_count);
    // fprintf(stderr, "num_threads = %d\n", num_threads);
    // fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    // fprintf(stderr, "gpu_device = %d\n", gpu_device);
    // fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    // ncnn::Mat in = ncnn::Mat(112, 112, 3);
    // in.fill(0.01f);
    cv::Mat dst;
    ncnn::Mat in;

    if (color) {
        cv::Mat bgr = cv::imread(inputImage.c_str(), 1);
        cv::cvtColor(bgr, dst, cv::COLOR_BGR2RGB);
        if (target_size != 0) {
            in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_RGB, dst.cols, dst.rows, target_size, target_size);
        } else {
            in = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_RGB, dst.rows, dst.cols);
        }
        // printf("Image file name: %s, %d, %d, %d\n",inputImage.c_str(), bgr.channels(), dst.rows, dst.cols);
    } else {
        cv::Mat bgr = cv::imread(inputImage.c_str(), 0);
        dst         = Mat(bgr);
        if (target_size != 0) {
            in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_GRAY, dst.cols, dst.rows, target_size, target_size);
        } else {
            in = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_GRAY, dst.rows, dst.cols);
        }
        // printf("Image file name: %s, %d, %d, %d\n",inputImage.c_str(), bgr.channels(), dst.rows, dst.cols);
    }
    // print_3d_cvmat_byte(dst);
    // PrintMatrix(std::string("original image").c_str(), dst);

    int img_w = dst.cols;
    int img_h = dst.rows;
    (void) img_w;
    (void) img_h;

    if (scale) {
        const float scale = (max - min) / 255.0f;
        const float mean  = (255.0f - 255.0f / (max - min));
        if (color) {
            const float mean_vals[3] = {mean, mean, mean};
            const float norm_vals[3] = {scale, scale, scale};
            in.substract_mean_normalize(mean_vals, norm_vals);
        } else {
            const float mean_vals[1] = {mean};
            const float norm_vals[1] = {scale};
            in.substract_mean_normalize(mean_vals, norm_vals);
        }
    }

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    printf("%s:%d, to get: %s\n", __FUNCTION__, __LINE__, outputName.c_str());

    ncnn::Net net;
    net.opt = opt;

    // printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_param(string(modelName + ".param").c_str());

    // printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_model(string(modelName + ".bin").c_str());

    // printf("%s:%d\n", __FUNCTION__, __LINE__);

    const std::vector<const char*>& input_names = net.input_names();
    // const std::vector<const char*>& output_names = net.output_names();

    // printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Mat out;

    ncnn::Extractor ex = net.create_extractor();

    // printf("%s:%d\n", __FUNCTION__, __LINE__);

    ex.input(input_names[0], in);

    // printf("%s:%d\n", __FUNCTION__, __LINE__);

    // for (unsigned int i = 0; i < output_names.size(); i++){
    //     printf("%s:%d: %s\n", __FUNCTION__, __LINE__, output_names[i]);
    // }

    ex.extract(outputName.c_str(), out);

    // pretty_print_ncnn(out);

    return out;
}

ncnn::Mat getSNNLayer(string inputName, bool force3Channels = false, int actualChanels = 0) {
    // string inputName = "./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[7]_input.dump";
    std::ifstream file(inputName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        SNN_RIP("File %s not found!", inputName.c_str());
    }
    file.seekg(0, std::ios::beg);

    char* buf = new char[32];
    if (file.read(buf, 32)) {
        for (int i = 0; i < 32; i++) {
            // printf("%s:%d, %d\n", __FUNCTION__, __LINE__, buf[i]);
        }
    }
    file.close();

    // printf("%s:%d\n", __FUNCTION__, __LINE__);

    string space_delimiter = " ";
    vector<int> words {};
    string text = string(buf); // + " ";
    size_t pos  = 0;
    while ((pos = text.find(space_delimiter)) != string::npos) {
        // printf("%s:%d: %s\n", __FUNCTION__, __LINE__, text.substr(0, pos).c_str());
        words.push_back(std::stoi(text.substr(0, pos)));
        text.erase(0, pos + space_delimiter.length());
        // printf("%d\n", words[words.size()-1]);
    }

    std::ifstream fin(inputName, std::ios::binary);
    fin.read(buf, 32);

    // SNN_LOGI("Size of words array: %d", words.size());

    // ncnn::Mat snnOutput(words[0], words[1], words[2]*4);

    if (actualChanels == 0) {
        actualChanels = words[2] * 4;
    }
    cv::Mat outputMat;
    if (force3Channels) {
        int size[3] = {words[1], words[0], 3};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        printf("%s:%d, Dim: %d, %d, %d\n", __FUNCTION__, __LINE__, words[0], words[1], 3);

        for (int p4 = 0; p4 < (size[2] + 3) / 4; p4++) {
            float fr       = 0;
            int actualLen = 4;
            for (int i = 0; i < size[0] * size[1]; i++) {
                float* ind = (float*) outputMat.data;
                for (int j = 0; j < actualLen; j++) {
                    fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
                    if (j < 3) {
                        *(ind + i * size[2] + p4 * 3 + j) = fr;
                    }
                }
            }
        }

    } else {
        int size[3] = {words[1], words[0], actualChanels};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        printf("%s:%d, Dim: %d, %d, %d, actual channels: %d\n", __FUNCTION__, __LINE__, words[0], words[1], words[2] * 4, actualChanels);

        for (int p4 = 0; p4 < (size[2] + 3) / 4; p4++) {
            float fr      = 0;
            int actualLen = 4;
            for (int i = 0; i < size[0] * size[1]; i++) {
                float* ind = (float*) outputMat.data;
                for (int j = 0; j < actualLen; j++) {
                    fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
                    if (p4 * 4 + j < actualChanels) {
                        *(ind + i * size[2] + p4 * 4 + j) = fr;
                    }
                }
            }
        }
    }
    // printf("---------------output of opencv mat ch first----------- \n");
    // for (int k = 0; k < outputMat.size[2]; k++) {
    //     std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    //     for (int j = 0; j < outputMat.size[1]; j++) {
    //         for (int i = 0; i < outputMat.size[0]; i++) {
    //             printf("%1.7f,",outputMat.at<float>(i,j,k));
    //         }
    //         //std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
    //         std::cout  << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    delete[] buf;

    fin.close();

    // ncnn::Mat snnOutput(size[0], size[1], size[2]);

    // for (int q=0; q<snnOutput.c; q++)
    // {
    //     float* ptr = snnOutput.channel(q);
    //     for (int y=0; y<snnOutput.h; y++)
    //     {
    //         for (int x=0; x<snnOutput.w; x++)
    //         {
    //             ptr[x] = outputMat.at<float>(y, x, q);
    //         }
    //         ptr += snnOutput.w;
    //     }
    // }

    ncnn::Mat snnOutput = CVMat2NCNNMat(outputMat);

    // pretty_print_ncnn(snnOutput);

    return snnOutput;
}

cv::Mat getCVMatFromDump(string inputName, bool force3Channels = false, int actualChanels = 0) {
    // string inputName = "./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[7]_input.dump";

    std::ifstream file(inputName, std::ios::binary | std::ios::ate);
    file.seekg(0, std::ios::beg);

    char* buf = new char[32];
    if (file.read(buf, 32)) {
        // for (int i=0; i<32; i++)
        // {
        //     printf("%s:%d, %d\n", __FUNCTION__, __LINE__, buf[i]);
        // }
    }
    file.close();

    printf("%s:%d from: %s\n", __FUNCTION__, __LINE__, inputName.c_str());

    string space_delimiter = " ";
    vector<int> words {};
    string text = string(buf); // + " ";
    size_t pos  = 0;
    while ((pos = text.find(space_delimiter)) != string::npos) {
        // printf("%s:%d: %s\n", __FUNCTION__, __LINE__, text.substr(0, pos).c_str());
        words.push_back(std::stoi(text.substr(0, pos)));
        text.erase(0, pos + space_delimiter.length());
        // printf("%d\n", words[words.size()-1]);
    }

    std::ifstream fin(inputName, std::ios::binary);
    fin.read(buf, 32);

    // ncnn::Mat snnOutput(words[0], words[1], words[2]*4);

    if (actualChanels == 0) {
        actualChanels = words[2] * 4;
    }
    cv::Mat outputMat;
    if (force3Channels) {
        int size[3] = {words[0], words[1], 3};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        printf("%s:%d, Dim: %d, %d, %d\n", __FUNCTION__, __LINE__, words[0], words[1], 3);

        for (int p4 = 0; p4 < (size[2] + 3) / 4; p4++) {
            float fr      = 0;
            int actualLen = 4;
            for (int i = 0; i < size[0] * size[1]; i++) {
                float* ind = (float*) outputMat.data;
                for (int j = 0; j < actualLen; j++) {
                    fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
                    if (j < 3) {
                        *(ind + i * size[2] + p4 * 3 + j) = fr;
                    }
                }
            }
        }

    } else {
        int size[3] = {words[0], words[1], actualChanels};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        printf("%s:%d, Dim: %d, %d, %d, actual channels: %d\n", __FUNCTION__, __LINE__, words[0], words[1], words[2] * 4, actualChanels);

        for (int p4 = 0; p4 < (size[2] + 3) / 4; p4++) {
            float fr      = 0;
            int actualLen = 4;
            for (int i = 0; i < size[0] * size[1]; i++) {
                float* ind = (float*) outputMat.data;
                for (int j = 0; j < actualLen; j++) {
                    fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
                    if (p4 * 4 + j < actualChanels) {
                        *(ind + i * size[2] + p4 * 4 + j) = fr;
                    }
                }
            }
        }
    }

    delete[] buf;

    fin.close();

    return outputMat;
}

ncnn::Mat getSNNLayerText(string inputName) {
    ifstream file(inputName);
    string data = "";
    std::vector<float> values;
    while (getline(file, data, ',')) {
        // cout << data << endl;
        try {
            values.push_back(std::stof(data));
        } catch (std::invalid_argument const& ex) {
            std::cout << "End of file " << ex.what() << '\n';
            break;
        } catch (std::out_of_range const& e) { values.push_back(0.0); }
    }
    file.close();

    ncnn::Mat snnOutput(values.size());

    for (int q = 0; q < snnOutput.c; q++) {
        float* ptr = snnOutput.channel(q);
        for (int y = 0; y < snnOutput.h; y++) {
            for (int x = 0; x < snnOutput.w; x++) {
                int idx = q * snnOutput.h * snnOutput.h + y * snnOutput.w + x;
                // printf("Index:%d, %f\n", idx, values[idx]);
                ptr[x] = values[idx];
            }
            ptr += snnOutput.w;
        }
    }

    // pretty_print_ncnn(snnOutput);

    return snnOutput;
}

std::vector<ncnn::Mat> getWeigitBiasFromNCNN(string modelName, int layerId) {
    std::vector<ncnn::Mat> res;

    int g_loop_count           = 4;
    bool g_enable_cooling_down = true;

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    int num_threads = ncnn::get_cpu_count();
    int powersave   = 0;
    int gpu_device  = -1;

    // default option
    ncnn::Option opt;
    opt.lightmode   = true;
    opt.num_threads = 1;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int) g_enable_cooling_down);

    // pretty_print_ncnn(in);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Net net;
    net.opt = opt;

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_param(string(modelName + ".param").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_model(string(modelName + ".bin").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    const std::vector<const char*>& output_names = net.output_names();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Extractor ex = net.create_extractor();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    for (unsigned int i = 0; i < output_names.size(); i++) {
        printf("%s:%d: %s\n", __FUNCTION__, __LINE__, output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        printf("%s:%d: %zu, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }
    // pretty_print_ncnn(outputMat);

    auto conv   = (ncnn::Convolution*) layers[layerId];
    auto weight = conv->weight_data;
    auto bias   = conv->bias_data;

    res.push_back(weight);
    res.push_back(bias);
    return res;
}

std::vector<ncnn::Mat> getDepthwiseWeigitBiasFromNCNN(string modelName, int layerId) {
    std::vector<ncnn::Mat> res;

    int g_loop_count           = 4;
    bool g_enable_cooling_down = true;

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    int num_threads = ncnn::get_cpu_count();
    int powersave   = 0;
    int gpu_device  = -1;

    // default option
    ncnn::Option opt;
    opt.lightmode   = true;
    opt.num_threads = 1;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int) g_enable_cooling_down);

    // pretty_print_ncnn(in);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Net net;
    net.opt = opt;

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_param(string(modelName + ".param").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_model(string(modelName + ".bin").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    const std::vector<const char*>& output_names = net.output_names();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Extractor ex = net.create_extractor();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    for (unsigned int i = 0; i < output_names.size(); i++) {
        printf("%s:%d: %s\n", __FUNCTION__, __LINE__, output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        printf("%s:%d: %zu, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }
    // pretty_print_ncnn(outputMat);

    auto conv   = (ncnn::ConvolutionDepthWise*) layers[layerId];
    auto weight = conv->weight_data;
    auto bias   = conv->bias_data;

    res.push_back(weight);
    res.push_back(bias);
    return res;
}

std::vector<ncnn::Mat> getBatchNormFromNCNN(string modelName, int layerId) {
    std::vector<ncnn::Mat> res;

    int g_loop_count           = 4;
    bool g_enable_cooling_down = true;

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    int num_threads = ncnn::get_cpu_count();
    int powersave   = 0;
    int gpu_device  = -1;

    // default option
    ncnn::Option opt;
    opt.lightmode   = true;
    opt.num_threads = 1;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int) g_enable_cooling_down);

    // pretty_print_ncnn(in);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Net net;
    net.opt = opt;

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_param(string(modelName + ".param").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_model(string(modelName + ".bin").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    const std::vector<const char*>& output_names = net.output_names();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Extractor ex = net.create_extractor();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    for (unsigned int i = 0; i < output_names.size(); i++) {
        printf("%s:%d: %s\n", __FUNCTION__, __LINE__, output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        printf("%s:%d: %zu, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }
    // pretty_print_ncnn(outputMat);

    auto bn = (ncnn::BatchNorm*) layers[layerId];

    auto slope_data = bn->slope_data;
    auto mean_data  = bn->mean_data;
    auto var_data   = bn->var_data;
    auto bias_data  = bn->bias_data;

    res.push_back(slope_data);
    res.push_back(mean_data);
    res.push_back(var_data);
    res.push_back(bias_data);

    return res;
}

#if 0
int compareNCNNLayerSNNLayer(string modelName, int layerId, string inputDump, int inChs, int outChs, int dim, int kernel,
    int padding, int stride, bool force3Channels = false)
{
    auto weights = getWeigitBiasFromNCNN(modelName, layerId);
    //pretty_print_ncnn(weights[0]);
    //pretty_print_ncnn(weights[1]);

    auto inputNCNN = getSNNLayer(inputDump, force3Channels);

    ncnn::ParamDict pd;
    pd.set(0, outChs);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, 1); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, -233);      // pad_w
    pd.set(5, 1);     // bias_term
    pd.set(6, outChs * inChs * kernel * kernel);

    //Hack the value of Bias
    // float* biasPtr = weights[1].channel(0);
    // for (size_t p=0; p < outChs; p++)
    // {
    //     *(biasPtr + p) = 0;
    // }
    //*(biasPtr + 40) = -1e-5;
    // printf("%s:%d: %f\n", __FUNCTION__, __LINE__, *(biasPtr + 40));
    // //*(biasPtr + 40) = (float)(*(biasPtr + 40));
    // for (size_t p=0; p < outChs; p++)
    // {
    //     if (abs(*(biasPtr + p)) < 0.0001f) {
    //         printf("%s:%d: %zu,%f\n", __FUNCTION__, __LINE__, p, *(biasPtr + p));
    //         *(biasPtr + p) = 0.0001f;
    //         printf("%s:%d: %zu,%f\n", __FUNCTION__, __LINE__, p, *(biasPtr + p));
    //     }
    // }
    // printf("%s:%d: %f\n", __FUNCTION__, __LINE__, *(biasPtr + 40));

    ncnn::Mat ncnnOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("Convolution"), pd, weights, inputNCNN, ncnnOutput, (void (*)(ncnn::Convolution*))0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
        }
    }
    //pretty_print_ncnn(ncnnOutput);

    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    ShaderUnitTest test;

    cv::Mat inputMat = NCNNMat2CVMat(inputNCNN);

    std::vector<cv::Mat> inputWeights = std::vector<cv::Mat>(inChs * outChs);
    std::vector<float> inputBias = std::vector<float>(outChs, 0.0f);

    for (size_t p = 0; p < inputWeights.size(); p++) {
        inputWeights[p] = cv::Mat(kernel, kernel, CV_32FC1);
        memcpy((uchar*)inputWeights[p].data, (uchar*)weights[0].data + kernel * kernel * sizeof(float) * p, kernel * kernel * sizeof(float));
        //std::cout << "M = " << std::endl << " "  << inputWeights[p] << std::endl << std::endl;
    }

    if (1) {
        const float* ptr = weights[1].channel(0);
        for (size_t p=0; p < inputBias.size(); p++) {
            inputBias[p] = ptr[p];
        }
    }

    //print_3d_cvmat(inputMat);

    auto output = test.snnConvTest(inputMat, inputWeights, inputBias, dim, dim, inChs, outChs, kernel, 1, stride, 0, 1, false);

    ncnn::Mat snnOutput = CVMat2NCNNMat(output);

    //pretty_print_ncnn(snnOutput);

    int ret = CompareMat(ncnnOutput, snnOutput, 0.01);

    if (ret) {
        pretty_print_ncnn(ncnnOutput);
        pretty_print_ncnn(snnOutput);
    }
    printf("-----------------------------Compare ncnn output with snn res: %d\n", ret);

    return ret;
}
#endif

ncnn::Mat customizeNCNNLayer(string modelName, string inputImage, string layerType, string layerInputFile, int target_size, int inputChannels,
                             int outputChannels, int kernelSize, int padding, int stride, int layerId) {
    int g_loop_count           = 4;
    bool g_enable_cooling_down = true;

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    int num_threads = ncnn::get_cpu_count();
    int powersave   = 0;
    int gpu_device  = -1;

    // default option
    ncnn::Option opt;
    opt.lightmode   = true;
    opt.num_threads = 1;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int) g_enable_cooling_down);

    cv::Mat bgr = cv::imread(inputImage.c_str(), 1);

    cv::Mat dst;
    cv::cvtColor(bgr, dst, cv::COLOR_BGR2RGB);

    printf("Image file name: %s, %d, %d, %d\n", inputImage.c_str(), bgr.channels(), dst.rows, dst.cols);
    // print_3d_cvmat_byte(dst);
    //_PrintMatrix(std::string("original image").c_str(), dst);

    int img_w = dst.cols;
    int img_h = dst.rows;
    (void) img_w;
    (void) img_h;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_RGB, dst.cols, dst.rows, target_size, target_size);

    // pretty_print_ncnn(in);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Net net;
    net.opt = opt;

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_param(string(modelName + ".param").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    net.load_model(string(modelName + ".bin").c_str());

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    const std::vector<const char*>& input_names  = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ncnn::Extractor ex = net.create_extractor();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ex.input(input_names[0], in);

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    for (unsigned int i = 0; i < output_names.size(); i++) {
        printf("%s:%d: %s\n", __FUNCTION__, __LINE__, output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        printf("%s:%d: %zu, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }

    ncnn::Mat inputMat;

    ncnn::Mat layerOutput;
    {
        int ret = 0;
        if (layerType == "Convolution") {
            auto conv   = (ncnn::Convolution*) layers[layerId];
            auto weight = conv->weight_data;
            auto bias   = conv->bias_data;
            // pretty_print_ncnn(weights);
            // pretty_print_ncnn(bias);

            ncnn::ParamDict pd;
            pd.set(0, outputChannels); // num_output
            pd.set(1, kernelSize);     // kernel_w
            pd.set(2, 1);              // dilation_w
            pd.set(3, stride);         // stride_w
            pd.set(4, padding);        // pad_w
            pd.set(5, 1);              // bias_term
            pd.set(6, inputChannels * outputChannels * kernelSize * kernelSize);
            pd.set(18, 0.0f);
            pd.set(9, 1); // activation = relu

            // // Set input to fixed values
            // for (int q=0; q<inputMat.c; q++)
            // {
            //     float* ptr = inputMat.channel(q);
            //     for (int y=0; y<inputMat.h; y++)
            //     {
            //         for (int x=0; x<inputMat.w; x++)
            //         {
            //             ptr[x] = 1.0f;
            //         }
            //         ptr += inputMat.w;
            //     }
            // }
            inputMat = getSNNLayer(layerInputFile);

            std::vector<ncnn::Mat> weights(2);
            // Set Weight to fixed values
            // for (int q=0; q<weight.c; q++)
            // {
            //     float* ptr = weight.channel(q);
            //     for (int y=0; y<weight.h; y++)
            //     {
            //         for (int x=0; x<weight.w; x++)
            //         {
            //             ptr[x] = 1.0f;
            //         }
            //         ptr += weight.w;
            //     }
            // }

            // Set bias to fixed values
            // for (int q=0; q<bias.c; q++)
            // {
            //     float* ptr = bias.channel(q);
            //     for (int y=0; y<bias.h; y++)
            //     {
            //         for (int x=0; x<bias.w; x++)
            //         {
            //             ptr[x] = 0.0f;
            //         }
            //         ptr += bias.w;
            //     }
            // }

            weights[0] = weight;
            weights[1] = bias;

            // weights[0] = SetValueMat(32 * 3 * 3 * 3, 1.0f);
            // weights[1] = SetValueMat(32, 1.0f);

            // pretty_print_ncnn(weights[0]);
            // pretty_print_ncnn(weights[1]);

            ret = test_layer_naive(ncnn::layer_to_index(layerType.c_str()), pd, weights, inputMat, layerOutput, (void (*)(ncnn::Convolution*)) 0, 0);
        } else if (layerType == "Pooling") {
            ncnn::ParamDict pd;
            pd.set(0, 0);          // pooling type, max = 0, avg = 1
            pd.set(1, kernelSize); // kernel_w
            pd.set(2, stride);     // stride_w
            pd.set(3, 0);          // pad_left
            inputMat = getSNNLayer(layerInputFile);
            std::vector<ncnn::Mat> weights(0);
            ret = test_layer_naive(ncnn::layer_to_index(layerType.c_str()), pd, weights, inputMat, layerOutput, (void (*)(ncnn::Pooling*)) 0, 0);
        } else if (layerType == "Interp") {
            // ret = test_layer_naive(ncnn::layer_to_index(layerType.c_str()), pd, weights, inputMat, layerOutput, (void (*)(ncnn::Interp*))0, 0);
        }

        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
        }
    }
    // pretty_print_ncnn(layerOutput);

    return layerOutput;
}

#endif  //__MATUTIL_H__
