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

#include "matutil.h"
#include "net.h"
#include "snn/utils.h"
#include "snn/colorUtils.h"
#include "snn/image.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <fstream>
#include <iomanip>

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y) - 1)/(y))
#endif

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

void printC4Buffer(float *buffer, int input_h, int input_w, int input_c, std::ostream& stream) {
    printf("--------------------C4 Buffer with width:%d, height:%d, depth:%d -------------------- \n", input_w, input_h, input_c);
    for (uint32_t k = 0; k < input_c; k++) {
        for (uint32_t j = 0; j < input_h; j++) {
            for (uint32_t i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 0);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------R----" + std::to_string(j) + "------------" << std::endl;
        }
        for (uint32_t j = 0; j < input_h; j++) {
            for (uint32_t i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 1);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------G----" + std::to_string(j) + "------------" << std::endl;
        }
        for (uint32_t j = 0; j < input_h; j++) {
            for (uint32_t i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 2);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------B----" + std::to_string(j) + "------------" << std::endl;
        }
        for (uint32_t j = 0; j < input_h; j++) {
            for (uint32_t i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 3);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------A----" + std::to_string(j) + "------------" << std::endl;
        }
        stream << "*************C4 channel: " + std::to_string(k) + "***********" << std::endl;
    }
}

void printHWC(float *buffer, int input_h, int input_w, int input_c, std::ostream& stream) {
    printf("--------------------HWC buffer with width:%d, height:%d, depth:%d -------------------- \n", input_w, input_h, input_c);
    for (uint32_t k = 0; k < input_c; k++) {
        for (uint32_t j = 0; j < input_h; j++) {
            for (uint32_t i = 0; i < input_w; i++) {
                auto v = *(buffer + j * input_w * input_c + i * input_c + k);
                stream << std::setw(7) << v << ",";
            }
            stream << std::endl;
        }
        stream << "*************HWC channel: " + std::to_string(k) + "***********" << std::endl;
        stream << std::endl;
    }
}

void hwcToC4(float *buffer, int input_h, int input_w, int input_c, float* dst) {
    uint32_t input_c4 = UP_DIV(input_c, 4);
    for (uint32_t k = 0; k < input_c4; k++) {
        for (uint32_t j = 0; j < input_h; j++) {
            for (uint32_t i = 0; i < input_w; i++) {
                auto dest = dst + (k *input_h * input_w + j * input_w  + i) * 4;
                auto src = buffer + j*input_w*input_c + i * input_c + k*4;
                uint32_t count = (k+1)*4 > input_c? input_c-4*k:4;
                memcpy(dest, src, count*sizeof(float));
            }
        }
    }
}

void c4ToHWC(float *buffer, int input_h, int input_w, int input_c4, float* dst) {
    uint32_t input_c = input_c4*4;
    for (uint32_t k = 0; k < input_c4; k++) {
        for (uint32_t j = 0; j < input_h; j++) {
            for (uint32_t i = 0; i < input_w; i++) {
                auto src = buffer + (k *input_h * input_w + j * input_w  + i) * 4;
                auto dest = dst + j*input_w*input_c + i * input_c + k*4;
                memcpy(dest, src, 4*sizeof(float));
            }
        }
    }
}

ncnn::Mat hwc2NCNNMat(float *buffer, int input_h, int input_w, int input_c) {
    printf("%s:%d-------HWC Buffer-%d--%d--%d-------\n", __FUNCTION__, __LINE__, input_h, input_w, input_c);
    ncnn::Mat snnOutput(input_w, input_h, input_c);
    for (int q = 0; q < snnOutput.c; q++) {
        float* ptr = snnOutput.channel(q);
        for (int y = 0; y < snnOutput.h; y++) {
            for (int x = 0; x < snnOutput.w; x++) {
                ptr[x] = buffer[y * input_w * input_c + x * input_c + q];
            }
            ptr += snnOutput.w;
        }
    }
    return snnOutput;
}

ncnn::Mat hwc2NCNNMat(const uint8_t *raw_buf, int h, int w, snn::ColorFormat colorFormat) {
    snn::ColorFormatDesc cfd = snn::getColorFormatDesc(colorFormat);
    int c = cfd.ch;
    snn::ColorFormatType cft = snn::getColorFormatType(colorFormat);
    size_t bytes = cfd.bytes();
    ncnn::Mat snnOutput(w, h, c);
    for (uint32_t q = 0, k = 0; q < c; ++q) {
        for (uint32_t i = 0; i < h; ++i) {
            for (uint32_t j = 0; j < w; ++j, ++k) {
                const uint8_t *raw_ptr = raw_buf + ((i * w + j) * bytes + q * bytes / c);
                switch (cft) {
                    case snn::ColorFormatType::UINT8:
                        {
                            const uint8_t* ptr = raw_ptr;
                            snnOutput[k] = static_cast<float>(*ptr);
                        }
                        break;
                    case snn::ColorFormatType::UINT16:
                        {
                            const uint16_t* ptr = reinterpret_cast<const uint16_t*>(raw_ptr);
                            snnOutput[k] = static_cast<float>(*ptr);
                        }
                        break;
                    case snn::ColorFormatType::FLOAT16:
                        {
                            const snn::FP16* ptr = reinterpret_cast<const snn::FP16*>(raw_ptr);
                            snnOutput[k] = snn::FP16::toFloat(ptr->u);
                        }
                        break;
                    case snn::ColorFormatType::FLOAT32:
                        {
                            const float* ptr = reinterpret_cast<const float*>(raw_ptr);
                            snnOutput[k] = *ptr;
                        }
                        break;
                    default:
                        SNN_CHK(false);
                }
            }
        }
    }
    return snnOutput;
}

void ncnn2HWC(ncnn::Mat padA, float* dest) {
    printf("%s:%d-------NCNN mat-%d--%d--%d-------\n", __FUNCTION__, __LINE__, padA.h, padA.w, padA.c);
    int input_w = padA.w, input_c = padA.c;
    for (int q = 0; q < padA.c; q++) {
        float* ptr = padA.channel(q);
        for (int y = 0; y < padA.h; y++) {
            for (int x = 0; x < padA.w; x++) {
                dest[y * input_w * input_c + x * input_c + q] = ptr[x];
            }
            ptr += padA.w;
        }
    }
}

void pretty_print_ncnn(const ncnn::Mat& m, const char* header) {
    printf("------%s Mat--%d--%d--%d-------%zu\n", header, m.w, m.h, m.c, m.elemsize);
    for (int q = 0; q < m.c; q++) {
        const char* ptr = m.channel(q);
        for (int y = 0; y < m.h; y++) {
            for (int x = 0; x < m.w; x++) {
                printf("%f, ", *((float*)((char*)ptr+m.elemsize*x)));
            }
            ptr += (m.w*m.elemsize);
            printf("\n");
        }
        printf("----------%d--------------\n", q);
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

void print_3d_cvmat(cv::Mat outputMat) {
    printf("---------------output of opencv 3d mat w first----------- \n");
    for (int k = 0; k < outputMat.size[2]; k++) {
        for (int i = 0; i < outputMat.size[0]; i++) {
            for (int j = 0; j < outputMat.size[1]; j++) {
                std::cout << std::setw(7) << outputMat.at<float>(i, j, k) << ",";
            }
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
            }
            std::cout << std::endl;
        }
        std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
    }
    std::cout << std::endl;
}

ncnn::Mat CVMat2NCNNMat(cv::Mat output) {
    ncnn::Mat snnOutput(output.size[1], output.size[0], output.size[2]);

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
    cv::Mat inputMat(3, size, CV_32FC1);
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

ncnn::Mat getNCNNLayer(const std::string& modelName, const std::string& inputImage, const std::string& outputName, int target_size,
    bool scale, float min, float max, bool color) {
    SNN_LOGD("input image = %s", inputImage.c_str());

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    int num_threads = ncnn::get_cpu_count();
    int powersave   = 0;

    // default option
    ncnn::Option opt;
    opt.lightmode   = true;
    opt.num_threads = 1;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

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
    } else {
        cv::Mat bgr = cv::imread(inputImage.c_str(), 0);
        dst         = cv::Mat(bgr);
        if (target_size != 0) {
            in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_GRAY, dst.cols, dst.rows, target_size, target_size);
        } else {
            in = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_GRAY, dst.rows, dst.cols);
        }
    }
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

    ncnn::Net net;
    net.opt = opt;

    net.load_param(std::string(modelName + ".param").c_str());
    net.load_model(std::string(modelName + ".bin").c_str());

    const std::vector<const char*>& input_names = net.input_names();
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_names[0], in);
    ex.extract(outputName.c_str(), out);

    return out;
}

ncnn::Mat getSNNLayer(const std::string& inputName, bool force3Channels, int actualChanels, bool flat, int forceLen) {
    SNN_LOGD("input image = %s", inputName.c_str());

    std::ifstream file(inputName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        SNN_RIP("File %s not found!", inputName.c_str());
    }
    file.seekg(0, std::ios::beg);

    char* buf = new char[32];
    file.read(buf, 32);
    SNN_LOGV("%s header: %s", inputName.c_str(), buf);
    file.close();

    std::string space_delimiter = " ";
    std::vector<int> words;
    std::string text = std::string(buf) + space_delimiter;
    size_t pos  = 0;
    while ((pos = text.find(space_delimiter)) != std::string::npos) {
        words.push_back(std::stoi(text.substr(0, pos)));
        text.erase(0, pos + space_delimiter.length());
    }

    std::ifstream fin(inputName, std::ios::binary);
    fin.read(buf, 32);

    SNN_ASSERT(words.size() == 4);

    int width = words[0];
    int height = words[1];
    int depth = words[2];
    int channels = words[3];

    ncnn::Mat snnOutput;

    if (actualChanels == 0) {
        actualChanels = depth * 4;
    }
    cv::Mat outputMat;
    if (force3Channels) {
        int size[3] = {height, width, 3};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        SNN_LOGV("Dim: %d, %d, %d", words[0], words[1], 3);

        float fr       = 0;
        const int actualLen = 4;
        float* ind = (float*) outputMat.data;
        for (int i = 0; i < height * width; i++) {
            for (int j = 0; j < actualLen; j++) {
                fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
                if (j < 3) {
                    *(ind + i * 3 + j) = fr;
                }
            }
        }
    } else if (flat) {
        int len;
        if (forceLen > 0) {
            len = forceLen;
        } else {
            len = width * height * depth * channels;
        }
        snnOutput.create(len);
        float fr;
        float* ind = snnOutput.channel(0);
        for (int i = 0; i < len; ++i, ++ind) {
            fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
            *ind = fr;
        }
    } else {
        int size[3] = {height, width, actualChanels};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        SNN_LOGD("Dim: %d, %d, %d, actual channels: %d", width, height, depth * 4, actualChanels);

        float fr;
        float* ind = (float*) outputMat.data;
        for (int p4 = 0; p4 < (actualChanels + 3) / 4; p4++) {
            for (int i = 0; i < width * height; i++) {
                for (int j = 0; j < 4; j++) {
                    fin.read(reinterpret_cast<char*>(&fr), sizeof(float));
                    if (p4 * 4 + j < actualChanels) {
                        *(ind + i * actualChanels + p4 * 4 + j) = fr;
                    }
                }
            }
        }
    }

    delete[] buf;

    fin.close();

    if (!flat) {
        snnOutput = CVMat2NCNNMat(outputMat);
    }

    return snnOutput;
}

cv::Mat getCVMatFromDump(const std::string& inputName, bool force3Channels, int actualChanels) {
    SNN_LOGD("input image = %s", inputName.c_str());

    std::ifstream file(inputName, std::ios::binary | std::ios::ate);
    file.seekg(0, std::ios::beg);

    char* buf = new char[32];
    file.read(buf, 32);
    file.close();

    std::string space_delimiter = " ";
    std::vector<int> words {};
    std::string text = std::string(buf);
    size_t pos  = 0;
    while ((pos = text.find(space_delimiter)) != std::string::npos) {
        words.push_back(std::stoi(text.substr(0, pos)));
        text.erase(0, pos + space_delimiter.length());
    }

    std::ifstream fin(inputName, std::ios::binary);
    fin.read(buf, 32);

    if (actualChanels == 0) {
        actualChanels = words[2] * 4;
    }
    cv::Mat outputMat;
    if (force3Channels) {
        int size[3] = {words[0], words[1], 3};
        outputMat   = cv::Mat(3, size, CV_32FC1);

        SNN_LOGV("Dim: %d, %d, %d", words[0], words[1], 3);

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

        SNN_LOGD("Dim: %d, %d, %d, actual channels: %d", words[0], words[1], words[2] * 4, actualChanels);

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

ncnn::Mat getSNNLayerText(const std::string& inputName) {
    std::ifstream file(inputName);
    std::string data = "";
    std::vector<float> values;
    while (getline(file, data, ',')) {
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
                ptr[x] = values[idx];
            }
            ptr += snnOutput.w;
        }
    }

    return snnOutput;
}

std::vector<ncnn::Mat> getWeigitBiasFromNCNN(const std::string& modelName, int layerId) {
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

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    ncnn::Net net;
    net.opt = opt;

    net.load_param(std::string(modelName + ".param").c_str());

    net.load_model(std::string(modelName + ".bin").c_str());

    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Extractor ex = net.create_extractor();

    for (unsigned int i = 0; i < output_names.size(); i++) {
        SNN_LOGV("%s", output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        SNN_LOGV("%zu, %s, %s\n", i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }

    auto conv   = (ncnn::Convolution*) layers[layerId];
    auto weight = conv->weight_data;
    auto bias   = conv->bias_data;

    res.push_back(weight);
    res.push_back(bias);
    return res;
}

std::vector<ncnn::Mat> getDepthwiseWeigitBiasFromNCNN(const std::string& modelName, int layerId) {
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

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    ncnn::Net net;
    net.opt = opt;

    net.load_param(std::string(modelName + ".param").c_str());

    net.load_model(std::string(modelName + ".bin").c_str());

    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Extractor ex = net.create_extractor();

    for (unsigned int i = 0; i < output_names.size(); i++) {
        SNN_LOGV("%s", output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        SNN_LOGV("%zu, %s, %s", i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }

    auto conv   = (ncnn::ConvolutionDepthWise*) layers[layerId];
    auto weight = conv->weight_data;
    auto bias   = conv->bias_data;

    res.push_back(weight);
    res.push_back(bias);
    return res;
}

std::vector<ncnn::Mat> getBatchNormFromNCNN(const std::string& modelName, int layerId) {
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

    ncnn::Net net;
    net.opt = opt;

    net.load_param(std::string(modelName + ".param").c_str());

    net.load_model(std::string(modelName + ".bin").c_str());

    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Extractor ex = net.create_extractor();

    for (unsigned int i = 0; i < output_names.size(); i++) {
        SNN_LOGV("%s", output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        SNN_LOGV("%zu, %s, %s", i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }

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

ncnn::Mat customizeNCNNLayer(const std::string& modelName, const std::string& inputImage, const std::string& layerType,
const std::string& layerInputFile, int target_size, int inputChannels, int outputChannels, int kernelSize, int padding, int stride, int layerId) {
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

    SNN_LOGD("Image file name: %s, %d, %d, %d", inputImage.c_str(), bgr.channels(), dst.rows, dst.cols);

    int img_w = dst.cols;
    int img_h = dst.rows;
    (void) img_w;
    (void) img_h;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_RGB, dst.cols, dst.rows, target_size, target_size);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    ncnn::Net net;
    net.opt = opt;

    net.load_param(std::string(modelName + ".param").c_str());

    net.load_model(std::string(modelName + ".bin").c_str());

    const std::vector<const char*>& input_names  = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Extractor ex = net.create_extractor();

    ex.input(input_names[0], in);

    for (unsigned int i = 0; i < output_names.size(); i++) {
        SNN_LOGV("%s", output_names[i]);
    }

    auto layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++) {
        SNN_LOGV("%zu, %s, %s", i, layers[i]->name.c_str(), layers[i]->type.c_str());
    }

    ncnn::Mat inputMat;

    ncnn::Mat layerOutput;
    {
        int ret = 0;
        if (layerType == "Convolution") {
            auto conv   = (ncnn::Convolution*) layers[layerId];
            auto weight = conv->weight_data;
            auto bias   = conv->bias_data;

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

            inputMat = getSNNLayer(layerInputFile);

            std::vector<ncnn::Mat> weights(2);
            weights[0] = weight;
            weights[1] = bias;

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
        }

        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
        }
    }

    return layerOutput;
}
