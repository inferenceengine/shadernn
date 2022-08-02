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
#include "mobileNetV2Processor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include "ic2/dp.h"

using namespace std;
using namespace snn;

void MobileNetV2Processor::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    auto& initInputDesc = workload.inputs[0]->desc();
    auto inputGpuData   = workload.inputs[0]->getGpuData();

    if (initInputDesc.height != this->expectedHeight || initInputDesc.width != this->expectedWidth) {
        gl::TextureObject inputTex;
        inputTex.attach(inputGpuData.target, inputGpuData.texture);

        auto inputImage = inputTex.getBaseLevelPixels();

#ifdef __ANDROID__
        mkdir("/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump", 0700);
        inputImage.saveToPNG("/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump/input_image.png");

#endif
        auto formatDesc = getColorFormatDesc(inputImage.format());
        cv::Mat inputMat, resizeMat, convertMat;
        convertMat = cv::Mat(cv::Size(this->expectedWidth, this->expectedHeight), CV_32FC(inputImage.depth() * 4));
        if (formatDesc.glType == GL_FLOAT) {
            inputMat = cv::Mat(cv::Size(inputImage.height(), inputImage.width()), CV_32FC(inputImage.depth() * 4), inputImage.data());
        } else {
            inputMat = cv::Mat(cv::Size(inputImage.height(), inputImage.width()), CV_8UC(inputImage.depth() * 4), inputImage.data());
        }
        // cv::imwrite("/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump/input_image.png", inputMat);
        cv::resize(inputMat, resizeMat, cv::Size(this->expectedHeight, this->expectedWidth), cv::INTER_AREA);
        convertMat = cv::Mat(resizeMat.size(), CV_32FC(resizeMat.channels()));

        // cv::normalize(convertMat, convertMat, 1, -1, cv::NORM_MINMAX, CV_32F);
        resizeMat = (resizeMat - 127.5) / 127.5;

        if (inputMat.empty() || resizeMat.empty()) {
            SNN_LOGE("Input data is empty!");
            return;
        }

        if (formatDesc.glType != GL_FLOAT) {
            resizeMat.convertTo(convertMat, CV_32F);
        } else {
            convertMat = resizeMat;
        }

        gl::TextureObject resizedInputTex;
        if (inputImage.depth() > 4) {
            resizedInputTex.allocate2DArray(snn::ColorFormat::RGBA32F, expectedWidth, expectedHeight, inputImage.depth());

            std::size_t layerCount = 0;
            std::size_t offset     = expectedWidth * expectedHeight * 4;
            for (std::size_t i = 0; i < inputImage.depth(); i += 4) {
                layerCount = (std::size_t) i / 4;
                if (formatDesc.glType == GL_FLOAT) {
                    std::vector<float> tempData((float*) (convertMat.data + 4 * layerCount * offset),
                                                (float*) (convertMat.data + 4 * (layerCount + 1) * offset));
                    resizedInputTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                } else {
                    std::vector<uint8_t> tempData((convertMat.data + 4 * layerCount * offset), (convertMat.data + 4 * (layerCount + 1) * offset));
                    resizedInputTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                }
            }
            if (4 * layerCount < inputImage.depth()) {
                if (formatDesc.glType == GL_FLOAT) {
                    std::vector<float> tempData((float*) (convertMat.data + 4 * layerCount * offset),
                                                (float*) (convertMat.data + convertMat.total() * convertMat.elemSize()));
                    resizedInputTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                } else {
                    std::vector<uint8_t> tempData((convertMat.data + 4 * layerCount * offset), (convertMat.data + convertMat.total() * convertMat.elemSize()));
                    resizedInputTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                }
            }
        } else {
            resizedInputTex.allocate2D(snn::ColorFormat::RGBA32F, expectedWidth, expectedHeight);
            std::vector<float> tempData((float*) (convertMat.data), (float*) (convertMat.data + convertMat.total() * convertMat.elemSize()));
            resizedInputTex.setPixels(0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
        }
        resizedInputTex.detach();
        ((GpuFrameImage*) workload.inputs[0])->attach(resizedInputTex.target(), resizedInputTex.id());
    } else if (initInputDesc.format != snn::ColorFormat::RGBA32F) {
        gl::TextureObject scaleTex, inputTex;
        inputTex.attach(inputGpuData.target, inputGpuData.texture);
        auto inputImage = inputTex.getBaseLevelPixels();

        cv::Mat inputMat = cv::Mat(cv::Size(inputImage.height(), inputImage.width()), CV_8UC(inputImage.depth() * 4), inputImage.data());
        cv::Mat resizeMat;
        inputMat.convertTo(resizeMat, CV_32F);

        resizeMat       = resizeMat / 255.0;
        auto formatDesc = getColorFormatDesc(inputImage.format());

        if (inputImage.depth() > 4) {
            scaleTex.allocate2DArray(snn::ColorFormat::RGBA32F, expectedWidth, expectedHeight, inputImage.depth());

            std::size_t layerCount = 0;
            std::size_t offset     = expectedWidth * expectedHeight * 4;
            for (std::size_t i = 0; i < inputImage.depth(); i += 4) {
                layerCount = (std::size_t) i / 4;
                if (formatDesc.glType == GL_FLOAT) {
                    std::vector<float> tempData((float*) (resizeMat.data + 4 * layerCount * offset), (float*) (resizeMat.data + 4 * (layerCount + 1) * offset));
                    scaleTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                } else {
                    std::vector<uint8_t> tempData((resizeMat.data + 4 * layerCount * offset), (resizeMat.data + 4 * (layerCount + 1) * offset));
                    scaleTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                }
            }
            if (4 * layerCount < inputImage.depth()) {
                if (formatDesc.glType == GL_FLOAT) {
                    std::vector<float> tempData((float*) (resizeMat.data + 4 * layerCount * offset),
                                                (float*) (resizeMat.data + resizeMat.total() * resizeMat.elemSize()));
                    scaleTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                } else {
                    std::vector<uint8_t> tempData((resizeMat.data + 4 * layerCount * offset), (resizeMat.data + resizeMat.total() * resizeMat.elemSize()));
                    scaleTex.setPixels(layerCount, 0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
                }
            }
        } else {
            std::cout << "else Part in line 129 " << std::endl;
            scaleTex.allocate2D(snn::ColorFormat::RGBA32F, expectedWidth, expectedHeight);
            std::vector<float> tempData((float*) (resizeMat.data), (float*) (resizeMat.data + resizeMat.total() * resizeMat.elemSize()));
            scaleTex.setPixels(0, 0, 0, expectedWidth, expectedHeight, 0, tempData.data());
        }

        scaleTex.detach();
        ((GpuFrameImage*) workload.inputs[0])->attach(scaleTex.target(), scaleTex.id());
    }

    const auto& inputDesc  = workload.inputs[0]->desc();
    const auto& outputDesc = workload.output->desc();

    std::cout << "Output Width: " << outputDesc.width << std::endl;
    std::cout << "Output Height: " << outputDesc.height << std::endl;

    if (!ic2_) {
        dp::ShaderGenOptions options = {};
        options.mrtMode              = snn::MRTMode::DOUBLE_PLANE;
        options.weightMode           = snn::WeightAccessMethod::TEXTURES;
        options.preferrHalfPrecision = false;
        auto dp                      = snn::dp::loadFromJsonModel(modelFileName_, options.mrtMode, options.weightMode, false);
        options.desiredInput.width   = inputDesc.width;
        options.desiredInput.height  = inputDesc.height;
        options.desiredInput.depth   = 1;
        options.desiredInput.format  = inputDesc.format;
        options.compute              = this->compute_;

        options.desiredOutputFormat = inputDesc.format;

        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);
        cp.dumpOutputs         = this->dumpOutputs;
        ic2_                   = MixedInferenceCore::create(cp);
    }

    SNN_ASSERT(inputDesc.device == Device::GPU);
    SNN_ASSERT(outputDesc.device == Device::CPU);
    auto inputTexture  = ((GpuFrameImage*) workload.inputs[0])->getGpuData();
    auto outputTexture = ((GpuFrameImage*) workload.output)->getGpuData();

    auto& currentFrameTexture = frameTextures_[inputTexture.texture];
    if (!currentFrameTexture) {
        currentFrameTexture = Texture::createAttached(inputTexture.target, inputTexture.texture); // Create a thin shell around textureId
    }

    MixedInferenceCore::RunParameters rp = {};
    auto inputTextures                   = getFrameTexture(inputTexture.texture);
    rp.inputTextures                     = &inputTextures;
    rp.inputCount                        = 1;
    // rp.textureOut - getFrameTexture(outputTexture.texture);
    (void) outputTexture;
    // rp.transitionOutput = getFrameTexture(transitionOutputTexture.texture);
    rp.inputMatrix           = workload.cpuInputs;
    rp.output                = std::vector<std::vector<std::vector<float>>>();
    rp.modelOutput.modelType = InferenceEngine::ModelType::CLASSIFICATION;
    ic2_->run(rp);
    workload.modelOutput = rp.modelOutput;
}
