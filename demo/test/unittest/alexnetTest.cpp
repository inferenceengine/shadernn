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
// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "layer/convolution.h"
#include "layer/padding.h"
#include "layer/pooling.h"
#include "layer/interp.h"
#include "testutil.h"

#include "cpu.h"
#include "net.h"

#include "shaderUnitTest.h"

#define INPUT_CHS   32
#define OUTPUT_CHS  64
#define MODEL_INPUT 112

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
        printf("\n----------%d------------\n", q);
    }
}

void pretty_print_cvmat(const std::vector<cv::Mat> m) {
    printf("-------CV Mat-%d--%d--%zu-------\n", m[0].size[0], m[0].size[1], m.size());
    for (size_t q = 0; q < m.size(); q++) {
        std::cout << "M = " << std::endl << " " << m[q] << std::endl << std::endl;
    }
}

cv::Mat sliceMat(cv::Mat L, int dim, std::vector<int> _sz) {
    cv::Mat M(L.dims - 1, std::vector<int>(_sz.begin() + 1, _sz.end()).data(), CV_32FC1, L.data + L.step[0] * 0);
    return M;
}

void _PrintMatrix(const char* pMessage, cv::Mat& mat) {
    printf("\n%s\n", pMessage);

    for (int r = 0; r < mat.rows; r++) {
        for (int c = 0; c < mat.cols; c++) {
            switch (mat.depth()) {
            case CV_8U: {
                printf("%*u ", 3, mat.at<uchar>(r, c));
                break;
            }
            case CV_8S: {
                printf("%*hhd ", 4, mat.at<schar>(r, c));
                break;
            }
            case CV_16U: {
                printf("%*hu ", 5, mat.at<ushort>(r, c));
                break;
            }
            case CV_16S: {
                printf("%*hd ", 6, mat.at<short>(r, c));
                break;
            }
            case CV_32S: {
                printf("%*d ", 6, mat.at<int>(r, c));
                break;
            }
            case CV_32F: {
                printf("%*.4f ", 10, mat.at<float>(r, c));
                break;
            }
            case CV_64F: {
                printf("%*.4f ", 10, mat.at<double>(r, c));
                break;
            }
            }
        }
        printf("\n");
    }
    printf("\n");
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

ncnn::Mat runAlexNet(string modelName, string inputImage, string outputName) {
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
    // opt.blob_allocator = &g_blob_pool_allocator;
    // opt.workspace_allocator = &g_workspace_pool_allocator;
    // opt.use_winograd_convolution = true;
    // opt.use_sgemm_convolution = true;
    // opt.use_int8_inference = false;
    // opt.use_vulkan_compute = false;
    // opt.use_fp16_packed = true;
    // opt.use_fp16_storage = true;
    // opt.use_fp16_arithmetic = true;
    // opt.use_int8_storage = true;
    // opt.use_int8_arithmetic = true;
    // opt.use_packing_layout = true;
    // opt.use_shader_pack8 = false;
    // opt.use_image_storage = false;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int) g_enable_cooling_down);

    // ncnn::Mat in = ncnn::Mat(112, 112, 3);
    // in.fill(0.01f);

    cv::Mat bgr = cv::imread(inputImage.c_str(), 1);

    cv::Mat dst;
    cv::cvtColor(bgr, dst, cv::COLOR_BGR2RGB);

    printf("Image file name: %s, %d, %d, %d\n", inputImage.c_str(), bgr.channels(), dst.rows, dst.cols);
    // print_3d_cvmat_byte(dst);
    //_PrintMatrix(std::string("original image").c_str(), dst);

    const int target_size = MODEL_INPUT;
    int img_w             = dst.cols;
    int img_h             = dst.rows;
    (void) img_w;
    (void) img_h;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_RGB, dst.cols, dst.rows, target_size, target_size);

    // pretty_print_ncnn(in);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    printf("%s:%d, to get: %s\n", __FUNCTION__, __LINE__, outputName.c_str());

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

    ncnn::Mat out;

    ncnn::Extractor ex = net.create_extractor();

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    ex.input(input_names[0], in);

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    for (unsigned int i = 0; i < output_names.size(); i++) {
        printf("%s:%d: %s\n", __FUNCTION__, __LINE__, output_names[i]);
    }

    ex.extract(outputName.c_str(), out);

    // pretty_print_ncnn(out);

    return out;
}

ncnn::Mat getSNNLayer(string inputName) {
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

    printf("%s:%d\n", __FUNCTION__, __LINE__);

    string space_delimiter = " ";
    vector<int> words {};
    string text = string(buf);
    size_t pos  = 0;
    while ((pos = text.find(space_delimiter)) != string::npos) {
        words.push_back(std::stoi(text.substr(0, pos)));
        text.erase(0, pos + space_delimiter.length());
        printf("%d\n", words[words.size() - 1]);
    }

    std::ifstream fin(inputName, std::ios::binary);
    fin.read(buf, 32);

    // ncnn::Mat snnOutput(words[0], words[1], words[2]*4);

    int size[3]    = {words[0], words[1], words[2] * 4};
    auto outputMat = cv::Mat(3, size, CV_32FC1);

    printf("%s:%d, Dim: %d, %d, %d\n", __FUNCTION__, __LINE__, words[0], words[1], words[2] * 4);

    for (int p4 = 0; p4 < (size[2] + 3) / 4; p4++) {
        float fr      = 0;
        int actualLen = 4;
        for (int i = 0; i < size[0] * size[1]; i++) {
            float* ind = (float*) outputMat.data;
            for (int j = 0; j < actualLen; j++) {
                fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
                *(ind + i * size[2] + p4 * 4 + j) = fr;
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

    ncnn::Mat snnOutput(size[0], size[1], size[2]);

    for (int q = 0; q < snnOutput.c; q++) {
        float* ptr = snnOutput.channel(q);
        for (int y = 0; y < snnOutput.h; y++) {
            for (int x = 0; x < snnOutput.w; x++) {
                ptr[x] = outputMat.at<float>(y, x, q);
            }
            ptr += snnOutput.w;
        }
    }

    // pretty_print_ncnn(snnOutput);

    return snnOutput;
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
        }
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

ncnn::Mat CVMat2NCNNMat(cv::Mat output) {
    ncnn::Mat snnOutput(output.size[0], output.size[1], output.size[2]);
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
    int size[3] = {iw, ih, ic};
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
    for (int i = 0; i < layers.size(); i++) {
        printf("%s:%d: %d, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }
    // pretty_print_ncnn(outputMat);

    auto conv   = (ncnn::Convolution*) layers[layerId];
    auto weight = conv->weight_data;
    auto bias   = conv->bias_data;

    res.push_back(weight);
    res.push_back(bias);
    return res;
}

ncnn::Mat verifyConv2DLayer(string modelName, string inputImage, string inputName, string outputName, int layerId) {
    static int g_loop_count           = 4;
    static bool g_enable_cooling_down = true;

    static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    static ncnn::PoolAllocator g_workspace_pool_allocator;

    int num_threads = ncnn::get_cpu_count();
    int powersave   = 0;
    int gpu_device  = -1;

    // default option
    ncnn::Option opt;
    opt.lightmode   = true;
    opt.num_threads = 1;
    // opt.blob_allocator = &g_blob_pool_allocator;
    // opt.workspace_allocator = &g_workspace_pool_allocator;
    // opt.use_winograd_convolution = true;
    // opt.use_sgemm_convolution = true;
    // opt.use_int8_inference = false;
    // opt.use_vulkan_compute = false;
    // opt.use_fp16_packed = true;
    // opt.use_fp16_storage = true;
    // opt.use_fp16_arithmetic = true;
    // opt.use_int8_storage = true;
    // opt.use_int8_arithmetic = true;
    // opt.use_packing_layout = true;
    // opt.use_shader_pack8 = false;
    // opt.use_image_storage = false;

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

    const int target_size = MODEL_INPUT;
    int img_w             = dst.cols;
    int img_h             = dst.rows;
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
    for (int i = 0; i < layers.size(); i++) {
        printf("%s:%d: %d, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
        if (layers[i]->name == outputName) {
            printf("Found layer index = %d\n", i);
        }
    }

    ncnn::Mat inputMat;
    ncnn::Mat outputMat;

    ex.extract(inputName.c_str(), inputMat);
    ex.extract(outputName.c_str(), outputMat);

    // pretty_print_ncnn(outputMat);

    auto conv   = (ncnn::Convolution*) layers[layerId];
    auto weight = conv->weight_data;
    auto bias   = conv->bias_data;
    // pretty_print_ncnn(weights);
    // pretty_print_ncnn(bias);

    ncnn::ParamDict pd;
    pd.set(0, OUTPUT_CHS); // num_output
    pd.set(1, 3);          // kernel_w
    pd.set(2, 1);          // dilation_w
    pd.set(3, 1);          // stride_w
    pd.set(4, 1);          // pad_w
    pd.set(5, 1);          // bias_term
    pd.set(6, INPUT_CHS * OUTPUT_CHS * 3 * 3);
    pd.set(18, 0.0f);
    pd.set(9, 1); // activation = relu

    std::vector<ncnn::Mat> weights(2);

    weights[0] = weight;
    weights[1] = bias;

    ncnn::Mat layerOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("Convolution"), pd, weights, inputMat, layerOutput, (void (*)(ncnn::Convolution*)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
        }
    }
    // pretty_print_ncnn(layerOutput);

    int ret = CompareMat(outputMat, layerOutput);

    printf("NCNN test_convolution test res: %d\n", ret);

    return layerOutput;
}

ncnn::Mat customizeConv2DLayer(string modelName, string inputImage, string inputName, string snnInput, int layerId) {
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
    // opt.blob_allocator = &g_blob_pool_allocator;
    // opt.workspace_allocator = &g_workspace_pool_allocator;
    // opt.use_winograd_convolution = true;
    // opt.use_sgemm_convolution = true;
    // opt.use_int8_inference = false;
    // opt.use_vulkan_compute = false;
    // opt.use_fp16_packed = true;
    // opt.use_fp16_storage = true;
    // opt.use_fp16_arithmetic = true;
    // opt.use_int8_storage = true;
    // opt.use_int8_arithmetic = true;
    // opt.use_packing_layout = true;
    // opt.use_shader_pack8 = false;
    // opt.use_image_storage = false;

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

    const int target_size = MODEL_INPUT;
    int img_w             = dst.cols;
    int img_h             = dst.rows;
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
    for (int i = 0; i < layers.size(); i++) {
        printf("%s:%d: %d, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }

    ncnn::Mat inputMat;
    ex.extract(inputName.c_str(), inputMat);

    auto conv   = (ncnn::Convolution*) layers[layerId];
    auto weight = conv->weight_data;
    auto bias   = conv->bias_data;
    // pretty_print_ncnn(weights);
    // pretty_print_ncnn(bias);

    ncnn::ParamDict pd;
    pd.set(0, OUTPUT_CHS); // num_output
    pd.set(1, 3);          // kernel_w
    pd.set(2, 1);          // dilation_w
    pd.set(3, 1);          // stride_w
    pd.set(4, 1);          // pad_w
    pd.set(5, 1);          // bias_term
    pd.set(6, INPUT_CHS * OUTPUT_CHS * 3 * 3);
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
    inputMat = getSNNLayer(snnInput);

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

    ncnn::Mat layerOutput;
    {
        int ret = test_layer_naive(ncnn::layer_to_index("Convolution"), pd, weights, inputMat, layerOutput, (void (*)(ncnn::Convolution*)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
        }
    }
    // pretty_print_ncnn(layerOutput);

    return layerOutput;
}

ncnn::Mat customizeNCNNLayer(string modelName, string inputImage, string layerType, string layerInputFile, int inputChannels, int outputChannels,
                             int kernelSize, int layerId) {
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
    // opt.blob_allocator = &g_blob_pool_allocator;
    // opt.workspace_allocator = &g_workspace_pool_allocator;
    // opt.use_winograd_convolution = true;
    // opt.use_sgemm_convolution = true;
    // opt.use_int8_inference = false;
    // opt.use_vulkan_compute = false;
    // opt.use_fp16_packed = true;
    // opt.use_fp16_storage = true;
    // opt.use_fp16_arithmetic = true;
    // opt.use_int8_storage = true;
    // opt.use_int8_arithmetic = true;
    // opt.use_packing_layout = true;
    // opt.use_shader_pack8 = false;
    // opt.use_image_storage = false;

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

    const int target_size = MODEL_INPUT;
    int img_w             = dst.cols;
    int img_h             = dst.rows;
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
    for (int i = 0; i < layers.size(); i++) {
        printf("%s:%d: %d, %s, %s\n", __FUNCTION__, __LINE__, i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }

    ncnn::Mat inputMat;

    ncnn::Mat layerOutput;
    {
        int ret;
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
            pd.set(3, 1);              // stride_w
            pd.set(4, 1);              // pad_w
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
            pd.set(2, 2);          // stride_w
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

int main() {
    SRAND(7767517);
    // For padding option: 0=CONSTANT 1=REPLICATE 2=VALID
    // test_convolution(10, 10, 7, 6, 3, 1, 1, 1, 0, 1.0, 0.0, true);

#if NCNN_INT8
    // return 0
    //        || test_convolution_0()
    //        || test_convolution_1()
    //        || test_convolution_2();
#else
    // return 0
    //        || test_convolution_0()
    //        || test_convolution_2();
#endif

    const string ALEXNET_MODEL = "112x32";

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    if (ALEXNET_MODEL == "112x4") {
        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_1_input_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0]_input.dump"); // 7 for large model
        ret     = CompareMat(ncnnMat, snnMat);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);
        // pretty_print_ncnn(snnMat);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_1_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0].dump"); // 7 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.01);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_layer_1 output res: %d\n", ret);

        // ncnnMat = verifyConv2DLayer("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_1_input_blob", "Conv_Layer_1_blob",
        // 1);

        // ncnnMat = customizeConv2DLayer("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_1_input_blob",
        //                                 "./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0]_input.dump", 1);
        // snnMat = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0].dump");
        // ret = CompareMat(ncnnMat, snnMat);
        // if (ret) {
        //     pretty_print_ncnn(ncnnMat);
        //     pretty_print_ncnn(snnMat);
        // }
        //  printf("---------------------------------Compare 1st layer output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_1_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [02] MaxPooling2D_3x3 pass[0].dump"); // 7 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_1 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_2_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [03] Conv2D_3x3 pass[0].dump"); // 15 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_Layer_2_ output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_2_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [04] MaxPooling2D_3x3 pass[0].dump"); // 15 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_2 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_3_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [05] Conv2D_3x3 pass[0].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_Layer_3_ output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_3_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [06] MaxPooling2D_3x3 pass[0].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_3 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_4_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [07] Conv2D_3x3 pass[0].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_Layer_4_ output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_4_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [08] MaxPooling2D_3x3 pass[0].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_4 output res: %d\n", ret);

        // Input of Flattern layer
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "flatten_4_blob"); //flatten_blob for large model
        // snnMat =  getSNNLayerText("./inferenceCoreDump/basic_cnn_model.json layer [09] Flatten_3x3 cpu layer.txt");
        // ret = CompareMat(ncnnMat, snnMat, 0.1);
        // if (ret) {
        //     pretty_print_ncnn(ncnnMat);
        //     pretty_print_ncnn(snnMat);
        // }
        // printf("-----------------------------flattern_4 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Dense_Layer_1_blob");
        snnMat  = getSNNLayerText("./inferenceCoreDump/basic_cnn_model.json layer [10] Dense_3x3 cpu layer.txt");
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Dense_Layer_1 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Output_Layer_Softmax_blob");
        snnMat  = getSNNLayerText("./inferenceCoreDump/basic_cnn_model.json layer [11] Dense_3x3 cpu layer.txt");
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Output_Layer output res: %d\n", ret);

    } else {
        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_1_input_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[7]_input.dump"); // 7 for large model
        ret     = CompareMat(ncnnMat, snnMat);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);
        // pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0]_0.dump");
        pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0]_1.dump");
        pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0]_2.dump");
        pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[0]_3.dump");
        pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[1]_4.dump");
        pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[1]_5.dump");
        pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[1]_6.dump");
        pretty_print_ncnn(snnMat);

        snnMat = getSNNLayer("./inferenceCoreDump/weights/basic_cnn_model.json layer [01] Conv2D_3x3 pass[1]_7.dump");
        pretty_print_ncnn(snnMat);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_1_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [01] Conv2D_3x3 pass[7].dump"); // 7 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_layer_1 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_1_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [02] MaxPooling2D_3x3 pass[7].dump"); // 7 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_1 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_2_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [03] Conv2D_3x3 pass[15].dump"); // 15 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_Layer_2_ output res: %d\n", ret);

        // ncnnMat = verifyConv2DLayer("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_1_blob", "Conv_Layer_2_blob",
        // 3);
        // ncnnMat = customizeConv2DLayer("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_1_blob",
        //                                  "./inferenceCoreDump/basic_cnn_model.json layer [03] Conv2D_3x3 pass[15]_input.dump", 3);
        ncnnMat  = customizeNCNNLayer("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Convolution",
                                     "./inferenceCoreDump/basic_cnn_model.json layer [03] Conv2D_3x3 pass[15]_input.dump", 32, 64, 3, 3);
        ncnnMat2 = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_2_blob");
        ret      = CompareMat(ncnnMat, ncnnMat2, 0.1);
        if (ret) {
            pretty_print_ncnn(ncnnMat);
            pretty_print_ncnn(ncnnMat2);
        }
        printf("---------------------------------Compare ncnn model with ncnn single layer res: %d\n", ret);

        snnMat = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [03] Conv2D_3x3 pass[15].dump");
        ret    = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("---------------------------------Conv_Layer_2_blob output res: %d\n", ret);

        // ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_2_blob");
        ncnnMat = customizeNCNNLayer("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Pooling",
                                     "./inferenceCoreDump/basic_cnn_model.json layer [04] MaxPooling2D_3x3 pass[15]_input.dump", 63, 64, 2, 4);
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [04] MaxPooling2D_3x3 pass[15].dump"); // 15 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_2 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_3_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [05] Conv2D_3x3 pass[31].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_Layer_3_ output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_3_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [06] MaxPooling2D_3x3 pass[31].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_3 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Conv_Layer_4_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [07] Conv2D_3x3 pass[31].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Conv_Layer_4_ output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "MaxPool_Layer_4_blob");
        snnMat  = getSNNLayer("./inferenceCoreDump/basic_cnn_model.json layer [08] MaxPooling2D_3x3 pass[31].dump"); // 31 for large model
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------MaxPool_Layer_4 output res: %d\n", ret);

        // Input of Flattern layer
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
        // ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "flatten_blob"); //flatten_blob for large model
        // snnMat =  getSNNLayerText("./inferenceCoreDump/basic_cnn_model.json layer [09] Flatten_3x3 cpu layer.txt");
        // ret = CompareMat(ncnnMat, snnMat, 0.1);
        // if (ret) {
        //     pretty_print_ncnn(ncnnMat);
        //     pretty_print_ncnn(snnMat);
        // }
        // printf("-----------------------------flattern_4 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Dense_Layer_1_blob");
        snnMat  = getSNNLayerText("./inferenceCoreDump/basic_cnn_model.json layer [10] Dense_3x3 cpu layer.txt");
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Dense_Layer_1 output res: %d\n", ret);

        ncnnMat = runAlexNet("../data/assets/jsonModel/basic_cnn", "../data/assets/images/rose_1.png", "Output_Layer_Softmax_blob");
        snnMat  = getSNNLayerText("./inferenceCoreDump/basic_cnn_model.json layer [11] Dense_3x3 cpu layer.txt");
        ret     = CompareMat(ncnnMat, snnMat, 0.1);
        if (ret) {
            // pretty_print_ncnn(ncnnMat);
            // pretty_print_ncnn(snnMat);
        }
        printf("-----------------------------Output_Layer output res: %d\n", ret);
    }
    // const int sz[] = {4, 3, 6};
    // cv::Mat bigm(3, sz, CV_8UC1);
    // cout << bigm.dims << '\t';
    // for (int i=0; i<bigm.dims; ++i)
    //     cout << bigm.size[i] << ',';
    // cout << endl;

    // unsigned char *pointer = (unsigned char *)bigm.data;
    // for (int m = 0; m< bigm.size[0] * bigm.size[1] * bigm.size[2]; m++)
    // {
    //     *(pointer + m) = (unsigned char)m;
    // }
    // print_3d_cvmat_byte(bigm);

    // printf("Test %d, %d, %d", bigm.channels(), bigm.rows, bigm.cols);
}
