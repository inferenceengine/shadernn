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
#include "pch.h"
#include <snn/utils.h>
#include "modelparser.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <exception>
#include <utility>
#ifndef __ANDROID__
#include <experimental/filesystem>
#endif

using namespace snn::dp;

bool ModelParser::isInputRange01() {
    auto range = _modelOb.get("inputRange");
    auto s     = range.is<std::string>() ? range.get<std::string>() : "";
    return s == "[0,1]";
}

bool ModelParser::getPrecision() { return this->preferHp; }

int ModelParser::getLayerCount() {
    SNN_LOGV("ModelParser:: Get number of _shaderLayers in the model");
    picojson::object& numNode = _modelOb.get("numLayers").get<picojson::object>();
    int32_t layerCount        = static_cast<int32_t>(numNode["count"].get<double_t>());
    return layerCount;
}

std::string ModelParser::getActivation(int layerId) {
    SNN_LOGV("ModelParser:: Get activation of the layer");
    if (getLayerName(layerId).compare("Convolution") == 0) {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        std::string activation          = layerObj["activation"].get<std::string>();
        return activation;
    } else if (getLayerName(layerId).compare("SeparableConv2D") == 0) {
        std::string activation = "";
        return activation;
    } else {
        SNN_LOGW("ModelParser:: accessing activation in a non convolution layer");
        return "";
    }
}

int ModelParser::getInputPlanes(int layerId) {
    if (getNumInbound(layerId) != 0) {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        int inputPlanes            = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        return inputPlanes;
    } else {
        return 0;
    }
}

int ModelParser::getOutputPlanes(int layerId) {
    SNN_LOGV("ModelParser:: Get number of output planes of the layer");
    picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
    int inputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
    return inputPlanes;
}

std::string ModelParser::getLayerName(int layerId) {
    SNN_LOGV("ModelParser:: Get layer name");
    picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
    std::string class_name          = layerObj["type"].get<std::string>();
    if (class_name.compare("Lambda") == 0) {
        class_name = layerObj["name"].get<std::string>();
    }
    return class_name;
}

picojson::array ModelParser::getWeights(int layerId) {
    SNN_LOGV("ModelParser:: Get weight picojson array");
    if (getLayerName(layerId).compare("Convolution") == 0) {
        picojson::object& layerObj  = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        picojson::object& weightObj = layerObj["weights"].get<picojson::object>();
        picojson::array weightArray = weightObj["kernel"].get<picojson::array>();
        return weightArray;
    } else {
        SNN_LOGW("ModelParser:: accessing weights in a non convolution layer");
        return {};
    }
}

picojson::array ModelParser::getDepthWiseWeights(int layerId) {
    SNN_LOGV("ModelParser:: Get weights for depthwise layer");
    if (getLayerName(layerId).compare("SeparableConv2D") == 0) {
        picojson::object& layerObj  = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        picojson::array weightArray = layerObj["depthwise_weights"].get<picojson::array>();
        return weightArray;
    } else {
        SNN_LOGW("ModelParser:: accessing weights in a non convolution lsyer");
        return {};
    }
}

picojson::array ModelParser::getBias(int layerId) {
    SNN_LOGV("ModelParser:: Get bias picojson array");
    if (getLayerName(layerId).compare("Convolution") == 0 || getLayerName(layerId).compare("SeparableConv2D")) {
        picojson::object& layerObj  = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        picojson::object& weightObj = layerObj["weights"].get<picojson::object>();
        picojson::array biasArray   = weightObj["bias"].get<picojson::array>();
        return biasArray;
    } else {
        SNN_LOGW("ModelParser:: accessing bias in a non convolution layer");
        return {};
    }
}

int ModelParser::getNumInbound(int layerId) {
    SNN_LOGV("ModelParser:: Get number of inbounds");
    picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
    int numIn = static_cast<int>(layerObj["numInputs"].get<double>());
    return numIn;
}

std::vector<int> ModelParser::getInboundLayerId(int layerId) {
    int numIn                  = getNumInbound(layerId);
    picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
    std::vector<int> inboundLayers;
    picojson::array inputNodes = layerObj["inputId"].get<picojson::array>();
    for (int i = 0; i < numIn; i++) {
        int id = static_cast<int>(inputNodes[i].get<double>());
        inboundLayers.emplace_back(id);
    }
    return inboundLayers;
}

int ModelParser::getInlayerId(int layerId, int inboundNum) {
    SNN_LOGV("ModelParser:: Get inbound layer id");
    picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
    picojson::array inputNodes = layerObj["inputId"].get<picojson::array>();
    int layer                  = static_cast<int>(inputNodes[inboundNum].get<double>());
    return layer;
}

int ModelParser::getKernelSize(int layerId) {
    SNN_LOGV("ModelParser:: getKernelSize");
    if (getLayerName(layerId).compare("Convolution") == 0) {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        int kernelSize             = static_cast<int>(layerObj["kernel_size"].get<double_t>());
        return kernelSize;
    } else {
        SNN_LOGW("ModelParser:: accessing kernel in a non convolution layer");
        return 0;
    }
}

int ModelParser::getDepthwiseKernelSize(int layerId) {
    SNN_LOGV("ModelParser:: getKernelSize");
    if (getLayerName(layerId).compare("SeparableConv2D") == 0) {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        int kernelSize             = static_cast<int>(layerObj["Depthwise_Kernel"].get<double_t>());
        return kernelSize;
    } else {
        SNN_LOGW("ModelParser:: accessing kernel in a non convolution layer");
        return 0;
    }
}

int ModelParser::getDepthwiseMultiplier(int layerId) {
    SNN_LOGV("ModelParser:: getDepthwiseMultiplier");
    if (getLayerName(layerId).compare("SeparableConv2D") == 0) {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        int depthwiseMultiplier    = static_cast<int>(layerObj["depth_multiplier"].get<double_t>());
        return depthwiseMultiplier;
    } else {
        SNN_LOGW("ModelParser:: accessing depth multiplier in a non depthwise layer");
        return 0;
    }
}

std::string ModelParser::getUseBias(int layerId) {
    SNN_LOGV("ModelParser:: getUseBias");
    if (getLayerName(layerId).compare("Convolution") == 0 || getLayerName(layerId).compare("SeparableConv2D") == 0) {
        return "True";
    } else {
        SNN_LOGW("ModelParser:: accessing bias in a non convolution layer");
        return "";
    }
}

std::string ModelParser::getPadding(int layerId) {
    SNN_LOGV("ModelParser:: getPaddingInfo");
    if (getLayerName(layerId).compare("Convolution") == 0 || getLayerName(layerId).compare("SeparableConv2D") == 0) {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        return layerObj["padding"].get<std::string>();
    } else {
        SNN_LOGW("ModelParser:: accessing padding in a non convolution layer");
        return "";
    }
}

ModelParser::ModelParser(const CreationParameters cp) {
    const std::string& name = cp.filename;
    this->preferHp   = cp.preferHp;
    this->mrtMode    = cp.mrtMode;
    this->weightMode = cp.weightMode;

#ifdef __ANDROID__
    auto jsonBytes = snn::loadJsonFromStorage(name.c_str());
#else
    auto jsonBytes = snn::loadEmbeddedAsset(name.c_str());
    if (jsonBytes.empty()) {
        jsonBytes = snn::loadJsonFromStorage((name).c_str());
    }
#endif
    if (jsonBytes.empty()) {
        SNN_RIP("ModelParser:: Could not load JSON file %s", name.c_str());
    }
    SNN_LOGV("start parse model");
    std::string err = picojson::parse(_modelOb, std::string(jsonBytes.begin(), jsonBytes.end()));
    if (!err.empty()) {
        SNN_RIP("ModelParser:: Could not parse JSON file %s", name.c_str());
    }
    SNN_LOGV("end parse model");

    std::string fileName;
    picojson::object& numNode = _modelOb.get("numLayers").get<picojson::object>();
    if (numNode.count("bin_file_name") > 0) {
        fileName    = numNode["bin_file_name"].get<std::string>();
        isBinWeight = true;

#ifdef __ANDROID__
        size_t pos = name.find("/");
        std::string subPath = name.substr(0, pos);
        std::string path = std::string(MODEL_DIR) + "/" + subPath + "/";
#else
        std::string path = std::experimental::filesystem::current_path();
        size_t pos = name.find("/");
        std::string subPath = name.substr(0, pos);
        path += ("/" + std::string(MODEL_DIR) + "/" + subPath + "/");
#endif
        SNN_LOGD("bin file %s", (path + fileName).c_str());
        binFile.exceptions(std::ofstream::failbit); // may throw
        try {
            binFile.open((path + fileName).c_str(), std::ios::binary);
        } catch (const std::ios_base::failure& fail) {
            SNN_RIP("open %s: %s", (path + fileName).c_str(), fail.what());
        }
    }
}

int ModelParser::getInputHeight() {
    picojson::object& layerObj = _modelOb.get("block_0").get<picojson::object>();
    int inputHeight            = static_cast<int>(layerObj["Input Height"].get<double_t>());
    return inputHeight;
}

int ModelParser::getInputWidth() {
    picojson::object& layerObj = _modelOb.get("block_0").get<picojson::object>();
    int inputWidth             = static_cast<int>(layerObj["Input Width"].get<double_t>());
    return inputWidth;
}

int ModelParser::getUpscale() {
    picojson::object& numNode = _modelOb.get("node").get<picojson::object>();
    int32_t upscale           = static_cast<int32_t>(numNode["upscale"].get<double_t>());
    return upscale;
}

int ModelParser::getInputlayerChannels() {
    picojson::object& numNode = _modelOb.get("node").get<picojson::object>();
    int32_t channels          = static_cast<int32_t>(numNode["inputChannels"].get<double_t>());
    return channels;
}

bool ModelParser::useSubPixel() {
    picojson::object& numNode = _modelOb.get("node").get<picojson::object>();
    bool useSubPixel          = numNode["useSubpixel"].get<bool>();
    return useSubPixel;
}

bool ModelParser::normalize() {
    bool normalize = false;
    return normalize;
}

bool ModelParser::mergeY2GB() {
    bool mergeY2GB = false;
    return mergeY2GB;
}

snn::MRTMode ModelParser::getMRTMode() { return this->mrtMode; }

snn::WeightAccessMethod ModelParser::getWeightMode() { return this->weightMode; }

int ModelParser::getMaxPoolLayer(int& layerID, int& numOutputPlanes, int& numInputPlanes, int& poolSize, int& stride, std::string& paddingMode,
                                 std::string& paddingValue, std::string& paddingT, std::string& paddingB, std::string& paddingL, std::string& paddingR) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerID)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        picojson::array poolArray  = layerObj["pool"].get<picojson::array>();
        poolSize                   = static_cast<int>(poolArray[0].get<double_t>());
        try {
            if (layerObj.count("stride")) {
                stride = static_cast<int>(layerObj["stride"].get<double_t>());
            } else if (layerObj.count("strides")) {
                stride = static_cast<int>(layerObj["strides"].get<double_t>());
            } else {
                stride = poolSize;
            }
        } catch (std::exception& e) {
            try {
                if (layerObj.count("stride")) {
                    auto strideObj = layerObj["stride"].get<picojson::array>();
                    stride         = static_cast<int>(strideObj[0].get<double_t>());
                } else if (layerObj.count("strides")) {
                    auto strideObj = layerObj["strides"].get<picojson::array>();
                    stride         = static_cast<int>(strideObj[0].get<double_t>());
                } else {
                    stride = poolSize;
                }
            } catch (std::exception& e) { stride = poolSize; }
        }
        SNN_LOGD("%s:%d stride: %d\n", __FILENAME__, __LINE__, stride);
        try {
            auto paddingObj = layerObj["padding"].get<picojson::array>();
            try {
                auto upPadding   = paddingObj[0].get<picojson::array>();
                auto sidePadding = paddingObj[1].get<picojson::array>();
                paddingT         = std::to_string(static_cast<uint32_t>(upPadding[0].get<double>()));
                paddingB         = std::to_string(static_cast<uint32_t>(upPadding[1].get<double>()));
                paddingL         = std::to_string(static_cast<uint32_t>(sidePadding[0].get<double>()));
                paddingR         = std::to_string(static_cast<uint32_t>(sidePadding[1].get<double>()));
            } catch (std::exception& e) {
                paddingT = std::to_string(static_cast<uint32_t>(paddingObj[0].get<double>()));
                paddingL = std::to_string(static_cast<uint32_t>(paddingObj[1].get<double>()));
                paddingB = paddingT;
                paddingR = paddingL;
            }
        } catch (std::exception& e) {
            try {
                paddingT    = std::to_string(static_cast<uint32_t>(layerObj["padding"].get<double>()));
                paddingMode = std::to_string(static_cast<uint32_t>(layerObj["padding"].get<double>()));
            } catch (std::exception& e) {
                paddingT    = layerObj["padding"].get<std::string>();
                paddingMode = layerObj["padding"].get<std::string>();
            }
            paddingB = paddingT;
            paddingL = paddingT;
            paddingR = paddingT;
        }
        try {
            paddingValue = layerObj["padding_value"].get<std::string>();
        } catch (std::exception& e) {
            paddingValue = "constant"; // replicate OR reflection
        }
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getMaxPoolLayer : Issues parsing layer %d, %s", layerID, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getAvgPoolLayer(int& layerID, int& numOutputPlanes, int& numInputPlanes, int& poolSize, int& stride, std::string& padding) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerID)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        picojson::array poolArray;
        try {
            poolArray = layerObj["pool"].get<picojson::array>();
        } catch (std::exception& e) { poolArray = layerObj["pool_size"].get<picojson::array>(); }
        poolSize = static_cast<int>(poolArray[0].get<double_t>());
        try {
            stride = static_cast<int>(layerObj["stride"].get<double_t>());
        } catch (std::exception& e) {
            try {
                auto strideObj = layerObj["stride"].get<picojson::array>();
                stride         = static_cast<int>(strideObj[0].get<double_t>());
            } catch (std::exception& e) { stride = poolSize; }
        }
        padding = layerObj["padding"].get<std::string>();
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getAvgPoolLayer : Issues parsing layer %d, %s", layerID, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getAddLayer(int& layerID, std::string& activation, float& leakyReluAlpha) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerID)).get<picojson::object>();
        try {
            activation = layerObj["activation"].get<std::string>();
        } catch (std::exception& e) { activation = "linear"; }
        if (activation.compare("leaky_relu") == 0) {
            try {
                if (layerObj.count("leakyReluAlpha")) {
                    leakyReluAlpha = static_cast<float>(layerObj["leakyReluAlpha"].get<double>());
                } else {
                    leakyReluAlpha = static_cast<float>(layerObj["alpha"].get<double>());
                }
            } catch (std::exception& e) { leakyReluAlpha = 0.3; }
        }
        return 0;
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getAddLayer : Issues parsing layer %d, %s", layerID, e.what());
        return -1;
    }
}

int ModelParser::getActivationLayer(int& layerID, std::string& activation, float& leakyReluAlpha) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerID)).get<picojson::object>();
        try {
            activation = layerObj["activation"].get<std::string>();
        } catch (std::exception& e) { activation = "linear"; }
        if (activation.compare("leaky_relu") == 0) {
            try {
                leakyReluAlpha = static_cast<float>(layerObj["leakyReluAlpha"].get<double>());
            } catch (std::exception& e) { leakyReluAlpha = 0.3; }
        }
        return 0;
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getAddLayer : Issues parsing layer %d, %s", layerID, e.what());
        return -1;
    }
}

int ModelParser::getAdaptiveAvgPoolLayer(int& layerID, int& numOutputPlanes, int& numInputPlanes, int& poolSize) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerID)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        picojson::array poolArray  = layerObj["pool"].get<picojson::array>();
        poolSize                   = static_cast<int>(poolArray[0].get<double_t>());
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getAdaptiveAvgPoolLayer : Issues parsing layer %d, %s", layerID, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getFlattenLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& activation) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        try {
            activation = layerObj["activation"].get<std::string>();
        } catch (std::exception& e) { activation = "linear"; }
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getFlattenLayer : Issues parsing layer %d, %s", layerId, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getYOLOLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getYOLOLayer : Issues parsing layer %d, %s", layerId, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getInputLayer(int& layerId, uint32_t& inputWidth, uint32_t& inputHeight, uint32_t& inputChannels, uint32_t& inputIndex) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        inputWidth                 = static_cast<uint32_t>(layerObj["Input Width"].get<double_t>());
        inputHeight                = static_cast<uint32_t>(layerObj["Input Height"].get<double_t>());
        inputChannels              = static_cast<uint32_t>(layerObj["outputPlanes"].get<double_t>());
        if (layerObj.count("inputIndex")) {
            inputIndex = static_cast<uint32_t>(layerObj["inputIndex"].get<double_t>());
        }
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getInputLayer : Issues parsing layer %d, %s", layerId, e.what());
        inputWidth    = 0;
        inputHeight   = 0;
        inputChannels = 0;
        inputIndex = 0;
    }
    return 0;
}

int ModelParser::getDenseLayer(int& layerID, int& numOutputUnits, int& numInputUnits, std::string& activation, std::vector<std::vector<float>>& weights,
                               std::vector<float>& biases, float& leakyReluAlpha) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerID)).get<picojson::object>();
        int numOutputPlanes        = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        int numInputPlanes         = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        try {
            numOutputUnits = static_cast<int>(layerObj["units"].get<double_t>());
        } catch (std::exception& e) { numOutputUnits = numOutputPlanes; }
        picojson::object& weightObj = layerObj["weights"].get<picojson::object>();
        std::vector<std::vector<float>> weightMat;
        int64_t elementIndex = 0;
        float value;
        if (isBinWeight) {
            for (int i = 0; i < numInputPlanes; i++) {
                std::vector<float> weightRow;
                for (int j = 0; j < numOutputUnits; j++) {
                    binFile.read((char*) &value, sizeof(float));
                    weightRow.push_back(value);
                }
                weightMat.push_back(weightRow);
            }
            weights = std::move(weightMat);
            // SNN_LOGD("Conv2D loaded kernel last element %f", value);
        } else {
            picojson::array weightArray = weightObj["kernel"].get<picojson::array>();
            std::size_t weightSize      = weightArray.size();
            numInputUnits               = (int) weightSize / numOutputUnits;
            for (int i = 0; i < numInputUnits; i++) {
                std::vector<float> weightRow;
                for (int j = 0; j < numOutputUnits; j++) {
                    float data = static_cast<float>(weightArray[elementIndex].get<double_t>());
                    weightRow.push_back(data);
                    elementIndex++;
                }
                weightMat.push_back(weightRow);
            }
            weights = std::move(weightMat);
        }

        if (layerObj["useBias"].get<std::string>().compare("True") == 0) {
            if (isBinWeight) {
                for (int i = 0; i < numOutputUnits; i++) {
                    binFile.read((char*) &value, sizeof(float));
                    biases.push_back(value);
                }
            } else {
                picojson::array biasArray = weightObj["bias"].get<picojson::array>();
                for (int i = 0; i < numOutputUnits; i++) {
                    biases.push_back((float) biasArray[i].get<double_t>());
                }
            }
        } else {
            for (int i = 0; i < numOutputPlanes; i++) {
                biases.push_back(0);
            }
        }

        activation = layerObj["activation"].get<std::string>();
        if (activation.compare("leaky_relu") == 0) {
            try {
                if (layerObj.count("leakyReluAlpha")) {
                    leakyReluAlpha = static_cast<float>(layerObj["leakyReluAlpha"].get<double>());
                } else {
                    leakyReluAlpha = static_cast<float>(layerObj["alpha"].get<double>());
                }
            } catch (std::exception& e) { leakyReluAlpha = 0.3; }
        }
        return 0;
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getDenseLayer : Issues parsing layer %d, %s", layerID, e.what());
        return -1;
    }
}

int ModelParser::getConvolutionLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& activation, int& kernelSize, int& stride,
                                     std::vector<double>& biases, std::vector<cv::Mat>& weights, bool& useBatchNormalization,
                                     std::map<std::string, std::vector<float>>& batchNormalization, float& leakyReluAlpha, std::string& paddingT,
                                     std::string& paddingB, std::string& paddingL, std::string& paddingR, std::string& paddingMode, bool& useMultiInputs) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        activation                 = layerObj["activation"].get<std::string>();
        try {
            auto paddingObj = layerObj["padding"].get<picojson::array>();
            try {
                auto upPadding   = paddingObj[0].get<picojson::array>();
                auto sidePadding = paddingObj[1].get<picojson::array>();
                paddingT         = std::to_string(static_cast<uint32_t>(upPadding[0].get<double>()));
                paddingB         = std::to_string(static_cast<uint32_t>(upPadding[1].get<double>()));
                paddingL         = std::to_string(static_cast<uint32_t>(sidePadding[0].get<double>()));
                paddingR         = std::to_string(static_cast<uint32_t>(sidePadding[1].get<double>()));
                paddingMode      = layerObj["mode"].get<std::string>();
            } catch (std::exception& e) {
                paddingT = std::to_string(static_cast<uint32_t>(paddingObj[0].get<double>()));
                paddingL = std::to_string(static_cast<uint32_t>(paddingObj[1].get<double>()));
                paddingB = paddingT;
                paddingR = paddingL;
            }
        } catch (std::exception& e) {
            try {
                paddingT = std::to_string(static_cast<uint32_t>(layerObj["padding"].get<double>()));
            } catch (std::exception& e) { paddingT = layerObj["padding"].get<std::string>(); }
            paddingB = paddingT;
            paddingL = paddingT;
            paddingR = paddingT;
        }

        // TO DO: Add support for fixed padding, rather than calculate from filter size
        kernelSize = static_cast<int>(layerObj["kernel_size"].get<double_t>());
        stride     = static_cast<int>(layerObj["strides"].get<double_t>());
        if (layerObj.count("use_multi_inputs")) {
            useMultiInputs = layerObj["use_multi_inputs"].get<std::string>().compare("True") == 0;
        } else {
            useMultiInputs = false;
        }

        picojson::object& weightObj = layerObj["weights"].get<picojson::object>();
        weights                     = std::vector<cv::Mat>(numInputPlanes * numOutputPlanes, cv::Mat(kernelSize, kernelSize, CV_32FC1));
        int matProgress             = 0;
        float value;
        if (isBinWeight) {
            for (int i = 0; i < numOutputPlanes; i++) {
                for (int j = 0; j < numInputPlanes; j++) {
                    cv::Mat writeMatrix = cv::Mat::zeros(kernelSize, kernelSize, CV_32FC1);
                    for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
                        for (int writingCol = 0; writingCol < kernelSize; writingCol++) {
                            binFile.read((char*) &value, sizeof(float));
                            if (this->preferHp) {
                                value = snn::convertToMediumPrecision(value);
                            }
                            writeMatrix.at<float>(writingRow, writingCol) = value;
                        }
                    }
                    weights.at(matProgress) = std::move(writeMatrix);
                    matProgress++;
                }
            }
        } else {
            picojson::array weightArray = weightObj["kernel"].get<picojson::array>();
            int element_number          = 0;
            for (int i = 0; i < numOutputPlanes; i++) {
                for (int j = 0; j < numInputPlanes; j++) {
                    cv::Mat writeMatrix = cv::Mat::zeros(kernelSize, kernelSize, CV_32FC1);
                    for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
                        for (int writingCol = 0; writingCol < kernelSize; writingCol++) {
                            float data = static_cast<float>(weightArray[element_number].get<double_t>());
                            if (this->preferHp) {
                                data = snn::convertToMediumPrecision(data);
                            }
                            element_number++;
                            writeMatrix.at<float>(writingRow, writingCol) = data;
                        }
                    }
                    weights.at(matProgress) = std::move(writeMatrix);
                    matProgress++;
                }
            }
        }

        biases.resize(numOutputPlanes);
        if (layerObj["useBias"].get<std::string>().compare("True") == 0) {
            if (isBinWeight) {
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*) &value, sizeof(float));
                    biases[i] = value;
                    if (this->preferHp) {
                        biases[i] = snn::convertToMediumPrecision(biases[i]);
                    }
                }
            } else {
                picojson::array biasArray = weightObj["bias"].get<picojson::array>();
                for (int i = 0; i < numOutputPlanes; i++) {
                    biases[i] = biasArray[i].get<double_t>();
                    if (this->preferHp) {
                        biases[i] = snn::convertToMediumPrecision(biases[i]);
                    }
                }
            }
        } else {
            for (int i = 0; i < numOutputPlanes; i++) {
                biases[i] = 0.0f;
            }
        }

        useBatchNormalization = (layerObj["useBatchNormalization"].get<std::string>().compare("True") == 0) ? true : false;

        if (useBatchNormalization) {
            std::vector<float> betaBN;
            std::vector<float> gammaBN;
            std::vector<float> meanBN;
            std::vector<float> varianceBN;

            picojson::object& batchNormObj = layerObj["batchNormalization"].get<picojson::object>();
            if (isBinWeight) {
                gammaBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*) &value, sizeof(float));
                    gammaBN[i] = value;
                    if (this->preferHp) {
                        gammaBN[i] = snn::convertToMediumPrecision(value);
                    }
                }
                betaBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*) &value, sizeof(float));
                    betaBN[i] = value;
                    if (this->preferHp) {
                        betaBN[i] = snn::convertToMediumPrecision(value);
                    }
                }
                meanBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*) &value, sizeof(float));
                    meanBN[i] = value;
                    if (this->preferHp) {
                        meanBN[i] = snn::convertToMediumPrecision(value);
                    }
                }
                varianceBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*) &value, sizeof(float));
                    varianceBN[i] = value;
                    if (this->preferHp) {
                        varianceBN[i] = snn::convertToMediumPrecision(value);
                    }
                }
            } else {
                picojson::array betaArray  = batchNormObj["beta"].get<picojson::array>();
                picojson::array gammaArray = batchNormObj["gamma"].get<picojson::array>();
                picojson::array movingMean;
                if (batchNormObj.count("moving_mean")) {
                    movingMean = batchNormObj["moving_mean"].get<picojson::array>();
                } else {
                    movingMean = batchNormObj["movingMean"].get<picojson::array>();
                }

                picojson::array movingVariance;
                if (batchNormObj.count("moving_variance")) {
                    movingVariance = batchNormObj["moving_variance"].get<picojson::array>();
                } else {
                    movingVariance = batchNormObj["movingVariance"].get<picojson::array>();
                }

                for (int i = 0; i < numOutputPlanes; i++) {
                    if (this->preferHp) {
                        betaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(betaArray[i].get<double>())));
                        gammaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(gammaArray[i].get<double>())));
                        meanBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(movingMean[i].get<double>())));
                        varianceBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(movingVariance[i].get<double>())));
                    } else {
                        betaBN.emplace_back(static_cast<float>(betaArray[i].get<double>()));
                        gammaBN.emplace_back(static_cast<float>(gammaArray[i].get<double>()));
                        meanBN.emplace_back(static_cast<float>(movingMean[i].get<double>()));
                        varianceBN.emplace_back(static_cast<float>(movingVariance[i].get<double>()));
                    }
                }
            }
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("beta", betaBN));
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("gamma", gammaBN));
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("movingMean", meanBN));
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("movingVariance", varianceBN));
        }

        if (activation.compare("leakyRelu") == 0) {
            if (layerObj.count("leakyReluAlpha")) {
                leakyReluAlpha = static_cast<float>(layerObj["leakyReluAlpha"].get<double>());
            } else {
                leakyReluAlpha = static_cast<float>(layerObj["alpha"].get<double>());
            }
            if (this->preferHp) {
                leakyReluAlpha = snn::convertToMediumPrecision(leakyReluAlpha);
            }
        }
    }

    catch (std::exception& e) {
        SNN_LOGE("ModelParser::getConvolutionLayer : Issues parsing layer %d, %s", layerId, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getDepthwiseConvolutionLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& activation, int& kernelSize, int& stride,
                                              std::vector<double>& biases, std::vector<cv::Mat>& weights, bool& useBatchNormalization,
                                              std::map<std::string, std::vector<float>>& batchNormalization, float& leakyReluAlpha, std::string& paddingT,
                                              std::string& paddingB, std::string& paddingL, std::string& paddingR) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        numOutputPlanes            = (int) layerObj["outputPlanes"].get<double_t>();
        numInputPlanes             = (int) layerObj["inputPlanes"].get<double_t>();
        SNN_LOGD("Depthwise conv number of output Planes: %d", numOutputPlanes);
        activation = layerObj["activation"].get<std::string>();
        try {
            auto paddingObj = layerObj["padding"].get<picojson::array>();
            try {
                auto upPadding   = paddingObj[0].get<picojson::array>();
                auto sidePadding = paddingObj[1].get<picojson::array>();
                paddingT         = std::to_string(static_cast<uint32_t>(upPadding[0].get<double>()));
                paddingB         = std::to_string(static_cast<uint32_t>(upPadding[1].get<double>()));
                paddingL         = std::to_string(static_cast<uint32_t>(sidePadding[0].get<double>()));
                paddingR         = std::to_string(static_cast<uint32_t>(sidePadding[1].get<double>()));
            } catch (std::exception& e) {
                paddingT = std::to_string(static_cast<uint32_t>(paddingObj[0].get<double>()));
                paddingL = std::to_string(static_cast<uint32_t>(paddingObj[1].get<double>()));
                paddingB = paddingT;
                paddingR = paddingL;
            }
        } catch (std::exception& e) {
            try {
                paddingT = std::to_string(static_cast<uint32_t>(layerObj["padding"].get<double>()));
            } catch (std::exception& e) { paddingT = layerObj["padding"].get<std::string>(); }
            paddingB = paddingT;
            paddingL = paddingT;
            paddingR = paddingT;
        }

        kernelSize = (int) layerObj["kernel_size"].get<double_t>();
        stride     = (int) layerObj["strides"].get<double_t>();

        picojson::object& weightObj = layerObj["weights"].get<picojson::object>();
        weights = std::vector<cv::Mat>(numInputPlanes,
                                       cv::Mat(kernelSize, kernelSize, CV_32FC1));

        int matProgress = 0;
        float value;
        if (isBinWeight) {
            for (int j = 0; j < numInputPlanes; j++) {
                cv::Mat writeMatrix = cv::Mat::zeros(kernelSize, kernelSize, CV_32FC1);
                for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
                    for (int writingCol = 0; writingCol < kernelSize; writingCol++) {
                        binFile.read((char*)&value, sizeof(float));
                        if (this->preferHp) {
                            value = snn::convertToMediumPrecision(value);
                        }
                        writeMatrix.at<float>(writingRow, writingCol) = value;
                    }
                }
                weights.at(matProgress) = std::move(writeMatrix);
                matProgress++;
            }
        } else {
            picojson::array weightArray = weightObj["kernel"].get<picojson::array>();
            std::vector<float> chw(numInputPlanes * kernelSize * kernelSize, 0.0f);
            uint32_t planeSize = kernelSize * kernelSize;

            for (size_t i = 0; i < planeSize; ++i) {
                for (int c = 0; c < numInputPlanes; ++c) {
                    chw[c * planeSize + i] = (float) weightArray[i * numInputPlanes + c].get<double_t>();
                }
            }
            int element_number = 0;
            for (int j = 0; j < numInputPlanes; j++) {
                cv::Mat writeMatrix = cv::Mat::zeros(kernelSize, kernelSize, CV_32FC1);
                for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
                    for (int writingCol = 0; writingCol < kernelSize; writingCol++) {
                        float data = chw[element_number];
                        if (this->preferHp) {
                            data = snn::convertToMediumPrecision(data);
                        }
                        element_number++;
                        writeMatrix.at<float>(writingRow, writingCol) = data;
                    }
                }
                weights.at(matProgress) = std::move(writeMatrix);
                matProgress++;
            }
        }

        biases.resize(numOutputPlanes);
        if (layerObj["useBias"].get<std::string>().compare("True") == 0) {
            if (isBinWeight) {
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*)&value, sizeof(float));
                    biases[i] = value;
                    if (this->preferHp) {
                        biases[i] = snn::convertToMediumPrecision(biases[i]);
                    }
                }
            } else {
                picojson::array biasArray = weightObj["bias"].get<picojson::array>();
                for (int i = 0; i < numOutputPlanes; i++) {
                    biases[i] = biasArray[i].get<double_t>();
                    if (this->preferHp) {
                        biases[i] = snn::convertToMediumPrecision(biases[i]);
                    }
                }
            }
        }
        else {
            for (int i = 0; i < numOutputPlanes; i++) {
                biases[i] = 0;
            }
        }

        useBatchNormalization = (layerObj["useBatchNormalization"].get<std::string>().compare("True") == 0) ? true : false;

        if (useBatchNormalization) {
            std::vector<float> betaBN;
            std::vector<float> gammaBN;
            std::vector<float> meanBN;
            std::vector<float> varianceBN;
            picojson::object& batchNormObj = layerObj["batchNormalization"].get<picojson::object>();
            if (isBinWeight) {
                gammaBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*)&value, sizeof(float));
                    gammaBN[i] = value;
                }
                SNN_LOGV("Conv2D loaded gammaBN last element %f", value);
                betaBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*)&value, sizeof(float));
                    betaBN[i] = value;
                }
                SNN_LOGV("Conv2D loaded betaBN last element %f", value);
                meanBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*)&value, sizeof(float));
                    meanBN[i] = value;
                }
                SNN_LOGV("Conv2D loaded meanBN last element %f", value);
                varianceBN = std::vector<float>(numOutputPlanes);
                for (int i = 0; i < numOutputPlanes; i++) {
                    binFile.read((char*)&value, sizeof(float));
                    varianceBN[i] = value;
                }
                SNN_LOGV("Conv2D loaded varianceBN last element %f", value);
            } else {
                picojson::array betaArray = batchNormObj["beta"].get<picojson::array>();
                picojson::array gammaArray = batchNormObj["gamma"].get<picojson::array>();
                picojson::array movingMean;
                picojson::array movingVariance;
                if (batchNormObj.count("moving_mean")) {
                    movingMean = batchNormObj["moving_mean"].get<picojson::array>();
                } else {
                    movingMean = batchNormObj["movingMean"].get<picojson::array>();
                }
                if (batchNormObj.count("moving_variance")) {
                    movingVariance = batchNormObj["moving_variance"].get<picojson::array>();
                } else {
                    movingVariance = batchNormObj["movingVariance"].get<picojson::array>();
                }

                for (int i = 0; i < numOutputPlanes; i++) {
                    if (this->preferHp) {
                        betaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(betaArray[i].get<double>())));
                        gammaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(gammaArray[i].get<double>())));
                        meanBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(movingMean[i].get<double>())));
                        varianceBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(movingVariance[i].get<double>())));
                    } else {
                        betaBN.emplace_back(static_cast<float>(betaArray[i].get<double>()));
                        gammaBN.emplace_back(static_cast<float>(gammaArray[i].get<double>()));
                        meanBN.emplace_back(static_cast<float>(movingMean[i].get<double>()));
                        varianceBN.emplace_back(static_cast<float>(movingVariance[i].get<double>()));
                    }
                }
            }
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("beta", betaBN));
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("gamma", gammaBN));
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("movingMean", meanBN));
            batchNormalization.insert(std::pair<std::string, std::vector<float>>("movingVariance", varianceBN));
        }

        if (activation.compare("leakyRelu") == 0) {
            if (this->preferHp) {
                if (layerObj.count("leakyReluAlpha")) {
                    leakyReluAlpha = snn::convertToMediumPrecision(static_cast<float>(layerObj["leakyReluAlpha"].get<double>()));
                } else {
                    leakyReluAlpha = snn::convertToMediumPrecision(static_cast<float>(layerObj["alpha"].get<double>()));
                }
            } else {
                if (layerObj.count("leakyReluAlpha")) {
                    leakyReluAlpha = static_cast<float>(layerObj["leakyReluAlpha"].get<double>());
                } else {
                    leakyReluAlpha = static_cast<float>(layerObj["alpha"].get<double>());
                }
            }
        }
        SNN_LOGD("Successfully loaded Depthwise conv layer");
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getConvolutionLayer : Issues parsing layer %d, %s", layerId, e.what());
        return -1;
    }
    return 0;
}

float ModelParser::getUpSamplingScale(int layerId) {
    SNN_LOGD("ModelParser:: getUpSamplingScale");
    if (getLayerName(layerId).compare("UpSampling2D") == 0) {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        float scale                = static_cast<float>(layerObj["scaleFactor"].get<double_t>());
        return scale;
    } else {
        SNN_LOGW("ModelParser:: accessing scale in a non upsampling2D layer");
        return 0;
    }
}

std::string ModelParser::getUpSampling2DInterpolation(int layerId) {
    SNN_LOGD("ModelParser:: getUpSamplingScale");
    if (getLayerName(layerId).compare("UpSampling2D") == 0) {
        picojson::object& layerObj    = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        std::string interpolationType = layerObj["interpolation"].get<std::string>();
        return interpolationType;
    } else {
        SNN_LOGW("ModelParser:: accessing interpolation in a non upsampling2D layer");
        return 0;
    }
}

int ModelParser::getBatchNormLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::map<std::string, std::vector<float>>& batchNormalization,
                                   std::string& activation, float& leakyReluAlpha) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());

        std::vector<float> betaBN;
        std::vector<float> gammaBN;
        std::vector<float> meanBN;
        std::vector<float> varianceBN;
        picojson::object& batchNormObj = layerObj["batchNormalization"].get<picojson::object>();
        picojson::array betaArray, gammaArray;

        if (batchNormObj.count("beta")) {
            betaArray = batchNormObj["beta"].get<picojson::array>();
        }

        if (batchNormObj.count("gamma")) {
            gammaArray = batchNormObj["gamma"].get<picojson::array>();
        }

        picojson::array movingMean;
        if (batchNormObj.count("moving_mean")) {
            movingMean = batchNormObj["moving_mean"].get<picojson::array>();
        } else {
            movingMean = batchNormObj["movingMean"].get<picojson::array>();
        }

        picojson::array movingVariance;
        if (batchNormObj.count("moving_variance")) {
            movingVariance = batchNormObj["moving_variance"].get<picojson::array>();
        } else {
            movingVariance = batchNormObj["movingVariance"].get<picojson::array>();
        }

        for (int i = 0; i < numOutputPlanes; i++) {
            if (this->preferHp) {
                if (batchNormObj.count("beta")) {
                    betaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(betaArray[i].get<double>())));
                } else {
                    betaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(0.0f)));
                }

                if (batchNormObj.count("gamma")) {
                    gammaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(gammaArray[i].get<double>())));
                } else {
                    gammaBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(1.0f)));
                }

                meanBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(movingMean[i].get<double>())));
                varianceBN.emplace_back(snn::convertToMediumPrecision(static_cast<float>(movingVariance[i].get<double>())));
            } else {
                if (batchNormObj.count("beta")) {
                    betaBN.emplace_back(static_cast<float>(betaArray[i].get<double>()));
                } else {
                    betaBN.emplace_back(static_cast<float>(0.0f));
                }

                if (batchNormObj.count("gamma")) {
                    gammaBN.emplace_back(static_cast<float>(gammaArray[i].get<double>()));
                } else {
                    gammaBN.emplace_back(static_cast<float>(1.0f));
                }

                meanBN.emplace_back(static_cast<float>(movingMean[i].get<double>()));
                varianceBN.emplace_back(static_cast<float>(movingVariance[i].get<double>()));
            }
        }
        batchNormalization.insert(std::pair<std::string, std::vector<float>>("beta", betaBN));
        batchNormalization.insert(std::pair<std::string, std::vector<float>>("gamma", gammaBN));
        batchNormalization.insert(std::pair<std::string, std::vector<float>>("movingMean", meanBN));
        batchNormalization.insert(std::pair<std::string, std::vector<float>>("movingVariance", varianceBN));

        if (layerObj.count("activation")) {
            activation = layerObj["activation"].get<std::string>();
        }
        if (activation.compare("leakyRelu") == 0) {
            if (this->preferHp) {
                if (layerObj.count("leakyReluAlpha")) {
                    leakyReluAlpha = snn::convertToMediumPrecision(static_cast<float>(layerObj["leakyReluAlpha"].get<double>()));
                } else {
                    leakyReluAlpha = snn::convertToMediumPrecision(static_cast<float>(layerObj["alpha"].get<double>()));
                }
            } else {
                if (layerObj.count("leakyReluAlpha")) {
                    leakyReluAlpha = static_cast<float>(layerObj["leakyReluAlpha"].get<double>());
                } else {
                    leakyReluAlpha = static_cast<float>(layerObj["alpha"].get<double>());
                }
            }
        }
    }

    catch (std::exception& e) {
        SNN_LOGE("ModelParser::BatchNormLayer : Issues parsing layer %d, %s", layerId, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getPaddingLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& paddingT, std::string& paddingB, std::string& paddingL,
                                 std::string& paddingR, std::string&, float& /*constant*/) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
        try {
            auto paddingObj = layerObj["pads"].get<picojson::array>();
            paddingT        = std::to_string(static_cast<uint32_t>(paddingObj[2].get<double>()));
            paddingB        = std::to_string(static_cast<uint32_t>(paddingObj[6].get<double>()));
            paddingL        = std::to_string(static_cast<uint32_t>(paddingObj[3].get<double>()));
            paddingR        = std::to_string(static_cast<uint32_t>(paddingObj[7].get<double>()));
        } catch (std::exception& e) {
            auto paddingObj = layerObj["padding"].get<picojson::array>();
            try {
                auto upPadding   = paddingObj[0].get<picojson::array>();
                auto sidePadding = paddingObj[1].get<picojson::array>();
                paddingT         = std::to_string(static_cast<uint32_t>(upPadding[0].get<double>()));
                paddingB         = std::to_string(static_cast<uint32_t>(upPadding[1].get<double>()));
                paddingL         = std::to_string(static_cast<uint32_t>(sidePadding[0].get<double>()));
                paddingR         = std::to_string(static_cast<uint32_t>(sidePadding[1].get<double>()));
            } catch (std::exception& e) {
                paddingT = std::to_string(static_cast<uint32_t>(paddingObj[0].get<double>()));
                paddingL = std::to_string(static_cast<uint32_t>(paddingObj[1].get<double>()));
                paddingB = paddingT;
                paddingR = paddingL;
            }
        }
    }

    catch (std::exception& e) {
        SNN_LOGE("ModelParser::getPaddingLayer : Issues parsing layer %d, %s", layerId, e.what());
        return -1;
    }
    return 0;
}

int ModelParser::getInstanceNormalizationLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, float& epsilon,
                                               std::map<std::string, std::vector<float>>& batchNormalization, std::string& activation, float& leakyReluAlpha) {
    try {
        picojson::object& layerObj = _modelOb.get("Layer_" + std::to_string(layerId)).get<picojson::object>();
        numOutputPlanes            = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes             = static_cast<int>(layerObj["inputPlanes"].get<double_t>());

        if (layerObj.count("activation")) {
            activation = layerObj["activation"].get<std::string>();
        }

        epsilon                     = static_cast<float>(layerObj["epsilon"].get<double_t>());
        picojson::object& weightObj = layerObj["weights"].get<picojson::object>();

        std::vector<float> bias;
        std::vector<float> scale;

        bias.resize(numOutputPlanes);
        picojson::array biasArray = weightObj["bias"].get<picojson::array>();
        for (int i = 0; i < numOutputPlanes; i++) {
            bias.at(i) = biasArray[i].get<double_t>();
            if (this->preferHp) {
                bias.at(i) = snn::convertToMediumPrecision(bias.at(i));
            }
        }

        scale.resize(numOutputPlanes);
        picojson::array scaleArray = weightObj["scale"].get<picojson::array>();
        for (int i = 0; i < numOutputPlanes; i++) {
            scale.at(i) = scaleArray[i].get<double_t>();
            if (this->preferHp) {
                scale.at(i) = snn::convertToMediumPrecision(scale.at(i));
            }
        }

        batchNormalization.insert(std::pair<std::string, std::vector<float>>("beta", bias));
        batchNormalization.insert(std::pair<std::string, std::vector<float>>("gamma", scale));

        if (activation.compare("leakyRelu") == 0) {
            if (this->preferHp) {
                leakyReluAlpha = snn::convertToMediumPrecision(static_cast<float>(layerObj["leakyReluAlpha"].get<double>()));
            } else {
                leakyReluAlpha = static_cast<float>(layerObj["leakyReluAlpha"].get<double>());
            }
        }
    }

    catch (std::exception& e) {
        SNN_LOGE("ModelParser::getInstanceNormLayer : Issues parsing layer %d, %s", layerId, e.what());
        return -1;
    }
    return 0;
}
