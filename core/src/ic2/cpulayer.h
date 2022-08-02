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

#include <snn/snn.h>
#include <snn/utils.h>
#include <snn/imageTexture.h>
#include "inferencegraph.h"
#include "modelparser.h"
#include <utility>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>

#include <Eigen/Dense>

namespace snn {
namespace dp { // short for Dynamic Pipeline

template<typename T>
struct CPUCommonUtil {
    typedef enum class ActivationFunction { RELU, LEAKY_RELU, SIGMOID, SOFTMAX, TANH, SILU, IDENTITY } ActivationFunction;

    std::unordered_map<std::string, ActivationFunction> activationFuncMap = {
        {"relu", ActivationFunction::RELU},         {"leakyRelu", ActivationFunction::LEAKY_RELU},
        {"sigmoid", ActivationFunction::SIGMOID},   {"softmax", ActivationFunction::SOFTMAX},
        {"tanh", ActivationFunction::TANH},         {"SiLU", ActivationFunction::SILU},
        {"identity", ActivationFunction::IDENTITY}, {"", ActivationFunction::IDENTITY}};

    // CPUProgram pass only stores input and output in vec<vec<float>>
    std::optional<std::vector<std::vector<T>>> inputMat;
    std::optional<std::vector<std::shared_ptr<snn::ManagedRawImage>>> gpuTexMat;
    std::vector<std::vector<T>> outputMat;
    std::string activationClass;
    std::optional<float> alpha;
    bool isLastLayer;

    CPUCommonUtil(std::string activationClass, float alpha, bool lastLayer) {
        this->activationClass = activationClass;
        this->alpha           = alpha;
        this->isLastLayer     = lastLayer;
    }

    void operator=(const CPUCommonUtil& that) {
        this->activationClass = that.activationClass;
        this->alpha           = that.alpha;
        this->inputMat        = that.inputMat;
        this->gpuTexMat       = that.gpuTexMat;
        this->outputMat       = that.outputMat;
    }

    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FloatMat;
    typedef Eigen::Vector<T, Eigen::Dynamic> FloatVec;

    void flatten2d(std::vector<std::vector<T>>& inputMat, std::vector<T>& outputMat) {
        for (auto row : inputMat) {
            outputMat.insert(outputMat.end(), row.begin(), row.end());
        }
    }

    void flatten2d(std::vector<std::shared_ptr<snn::ManagedRawImage>>& inputMat, std::vector<T>& outputMat) {
        for (auto image : inputMat) {
            uint32_t size       = image->size();
            auto rawDataPointer = image->data();
            std::vector<uint8_t> dataArray(rawDataPointer, rawDataPointer + size);
            uint32_t width    = image->width();
            uint32_t height   = image->height();
            uint32_t depth    = image->depth();
            uint32_t channels = image->channels();
            auto format       = image->format();
            auto formatDesc   = snn::getColorFormatDesc(format);
            if (channels == 0) {
                channels = depth * 4;
            }
            uint32_t channelPerPlane = 4;
            assert(width * height * depth * (formatDesc.bits / 8) == size);
            // for (std::size_t row = 0; row < width; row++) {
            //     for (std::size_t column = 0; column < height; column++) {
            //         for (std::size_t channel = 0; channel < depth; channel++) {
            //             uint32_t offset = channel + depth * row + depth * width * column;
            //             std::vector<uint8_t> floatData;
            //             T data = 0.0;
            //             std::size_t byteSize = sizeof(data);
            //             for (std::size_t j = 0; j < byteSize; j++) {
            //                 floatData.push_back(dataArray.at(offset + j));
            //             }
            //             std::memcpy(&data, floatData.data(), sizeof(data));
            //             outputMat.push_back(data);
            //             // if (offset % 50 == 0) {
            //             //     SNN_LOGI("Value at %lu, %lu, %lu is %f", row, column, channel, data);
            //             // }
            //             floatData.clear();
            //         }
            //     }
            // }
            std::vector<std::size_t> indices(size);
            std::vector<std::size_t> reorderedIndices;
            std::iota(indices.begin(), indices.end(), 0);
            std::size_t byteSize = formatDesc.bits / (8 * formatDesc.ch);
            for (std::size_t row = 0; row < height; row++) {
                for (std::size_t column = 0; column < width; column++) {
                    for (std::size_t plane = 0; plane < depth; plane++) {
                        for (std::size_t channel = 0; channel < channelPerPlane; channel++) {
                            std::size_t index;
                            if (plane == depth - 1) {
                                uint32_t nChannels = channels - plane * channelPerPlane;
                                if (channel < nChannels) {
                                    index = channel + nChannels * column + nChannels * width * row + nChannels * height * width * plane;
                                    reorderedIndices.push_back(indices.at(byteSize * index));
                                }
                            } else {
                                std::size_t index =
                                    channel + channelPerPlane * column + channelPerPlane * width * row + channelPerPlane * height * width * plane;
                                reorderedIndices.push_back(indices.at(byteSize * index));
                            }
                        }
                    }
                }
            }
            // std::vector<std::size_t> indices(size);
            // std::vector<std::size_t> reorderedIndices;
            // std::iota(indices.begin(), indices.end(), 0);
            // std::size_t byteSize = sizeof(T);
            // for (std::size_t row = 0; row < height; row++) {
            //     for (std::size_t column = 0; column < width; column++) {
            //         for (std::size_t plane = 0; plane < depth; plane++) {
            //             for (std::size_t channel = 0; channel < channelPerPlane; channel++) {
            //                 std::size_t index = channel + channelPerPlane * row + channelPerPlane * height * column + channelPerPlane * height * width *
            //                 plane; reorderedIndices.push_back(indices.at(byteSize * index));
            //             }
            //         }
            //     }
            // }

            // std::size_t byteSize = sizeof(T);
            for (auto i : reorderedIndices) {
                // for ( size_t i = 0; i < size; i += byteSize ) {
                std::vector<uint8_t> floatData;
                T data = 0;
                for (std::size_t j = 0; j < byteSize; j++) {
                    floatData.push_back(dataArray.at(i + j));
                }
                if (byteSize == 2) {
                    uint16_t tempf16Val;
                    std::memcpy(&tempf16Val, floatData.data(), byteSize);
                    data = snn::convertToHighPrecision(tempf16Val);
                } else if (byteSize == 1) {
                    data = (float) floatData.at(0);
                } else {
                    std::memcpy(&data, floatData.data(), byteSize);
                }
                outputMat.push_back(data);
            }
        }
    }

    void transform(std::pair<std::vector<std::vector<T>>, std::vector<T>>& transformMats) {
        std::vector<T> flattenInput, flattenWeight;

        FloatMat weights, inputs, outputs;
        FloatVec biases;

        if (this->gpuTexMat) {
            this->flatten2d(this->gpuTexMat.value(), flattenInput);
        } else if (this->inputMat) {
            this->flatten2d(this->inputMat.value(), flattenInput);
        } else {
            SNN_RIP("Can't recognize the input type.");
            return;
        }

        if (transformMats.first.empty() || !this->inputMat.has_value()) {
            weights = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(flattenInput.size(), flattenInput.size());
            biases  = Eigen::Vector<T, Eigen::Dynamic>::Zero(flattenInput.size());
            inputs  = Eigen::Map<FloatVec>(flattenInput.data(), flattenInput.size());
        } else {
            this->flatten2d(transformMats.first, flattenWeight);

            T* weightsPointer = &flattenWeight.at(0);
            T* inputsPointer  = &flattenInput.at(0);
            T* biasPointer    = &transformMats.second.at(0);

            weights = Eigen::Map<FloatMat>(weightsPointer, transformMats.first.at(0).size(), transformMats.first.size());
            inputs  = Eigen::Map<FloatMat>(inputsPointer, this->inputMat.value().at(0).size(), this->inputMat.value().size());
            biases  = Eigen::Map<FloatVec>(biasPointer, transformMats.first.at(0).size());
        }

        outputs = weights * inputs + biases;
        for (auto row : outputs.colwise()) {
            this->outputMat.push_back(std::vector<T>(row.begin(), row.end()));
        }

        // this->outputMat(&outputs[0], outputs.data() + outputs.rows()*outputs.cols());
    }

    void leakyRelu(T& inputVal, float alpha) { inputVal = inputVal > 0 ? inputVal : alpha * inputVal; }

    void softmax(std::vector<T>& inputVal) {
        float max = -FLT_MAX;

        for (std::size_t i = 0; i < inputVal.size(); i++) {
            max = std::max(max, inputVal.at(i));
        }

        auto negExp = [max](T x) { return exp(x - max); };

        std::transform(inputVal.begin(), inputVal.end(), inputVal.begin(), negExp);

        T div = std::accumulate(inputVal.begin(), inputVal.end(), (T) 0.0);

        for (std::size_t i = 0; i < inputVal.size(); i++) {
            inputVal.at(i) = inputVal.at(i) / div;
        }
    }

    void sigmoid(T& inputVal) { inputVal = 1.0f / (1.0f + exp(-inputVal)); }

    void tanh(T& inputVal) { inputVal = (exp(2 * inputVal) - 1) / (exp(2 * inputVal) + 1); }

    void SiLU(T& inputVal) { inputVal = inputVal * 1.0f / (1.0f + exp(-inputVal)); }

    void activation() {
        ActivationFunction activationFunc = this->activationFuncMap[this->activationClass];
        switch (activationFunc) {
        case ActivationFunction::RELU: {
            for (auto& row : this->outputMat) {
                for (auto& val : row) {
                    this->leakyRelu(val, 0.0);
                }
            }
            break;
        };

        case ActivationFunction::LEAKY_RELU: {
            for (auto& row : this->outputMat) {
                for (auto& val : row) {
                    this->leakyRelu(val, this->alpha.value());
                }
            }
            break;
        };

        case ActivationFunction::SIGMOID: {
            for (auto& row : this->outputMat) {
                for (auto& val : row) {
                    this->sigmoid(val);
                }
            }
            break;
        };

        case ActivationFunction::SOFTMAX: {
            for (auto& row : this->outputMat) {
                this->softmax(row);
            }
            break;
        };

        case ActivationFunction::TANH: {
            for (auto& row : this->outputMat) {
                for (auto& val : row) {
                    this->tanh(val);
                }
            }
            break;
        };

        case ActivationFunction::SILU: {
            for (auto row : this->outputMat) {
                for (auto val : row) {
                    this->SiLU(val);
                }
            }
            break;
        };

        case ActivationFunction::IDENTITY: {
            break;
        }

        default:
            break;
        }
    }

    void run(std::pair<std::vector<std::vector<T>>, std::vector<T>>& transformMats) {
        this->transform(transformMats);
        // std::cout << "----------- [DEBUG ACTIVATION IN] -----------" << std::endl;
        // for (auto val : this->outputMat.at(0)) {
        //     std::cout << val << std::endl;
        // }
        // std::cout << "---------------------------------------------" << std::endl;
        this->activation();
    }

    void getOutputs(std::vector<std::vector<T>>& outputs) {
        outputs = outputMat;
        // Sanitize the outputs

        outputMat = std::vector<std::vector<T>>();
    }
};

}; // namespace dp
} // namespace snn
