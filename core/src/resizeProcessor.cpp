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
#include "processor.h"
#include <snn/texture.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "ic2/core.h"

void snn::ResizeProcessor::submit(snn::Processor::Workload& w) {
    for (size_t i = 0; i < w.inputCount; ++i) {
        SNN_ASSERT(Device::CPU == w.inputs[i]->desc().device || Device::GPU_CPU == w.inputs[i]->desc().device);
    }
    SNN_ASSERT(Device::CPU == w.output->desc().device || Device::GPU_CPU == w.output->desc().device);

    // allocate output buffer
    const auto& outDesc = w.output->desc();
    // const auto& fmtDesc = getColorFormatDesc(outDesc.format);

    // const auto& inFmtDesc = getColorFormatDesc(w.inputs[0]->desc().format);

    // std::cout << "Input Color Format: " << inFmtDesc.name << std::endl;
    // std::cout << "Output Color Format: " << fmtDesc.name << std::endl;

    for (std::size_t i = 0; i < w.inputCount; i++) {
        auto imageData = w.inputs[i]->getCpuData();
        // std::cout << "--------------------------------" << std::endl;
        // std::cout << "Size of input buffer: " << sizeof(imageData.data()[0]) << std::endl;
        // for (std::size_t i = 0; i < 10; i++) {
        //     std::cout << "Input Buffer values: " << (int) imageData.data()[i] << std::endl;
        // }
        // std::cout << "--------------------------------" << std::endl;
        cv::Mat dummyImage(cv::Size((int) w.inputs[i]->desc().width, (int) w.inputs[i]->desc().height), CV_8UC4, imageData.data());
        cv::resize(dummyImage, dummyImage, cv::Size(112, 112));

        if (!this->scale) {
            cv::Mat floatImage;
            dummyImage.convertTo(floatImage, CV_32F);
            auto& outputData = w.output->getCpuData();
            // outputData.vertFlipInpace();
            std::memcpy(outputData.data(), floatImage.data, outDesc.width * outDesc.height * dummyImage.channels() * sizeof(float));
        } else {
            auto& outputData = w.output->getCpuData();
            // outputData.vertFlipInpace();
            std::memcpy(outputData.data(), dummyImage.data, outDesc.width * outDesc.height * dummyImage.channels());
        }

        // cv::imwrite("/home/us000145/bitbucket/debug_out/basic_cnn/input.png", dummyImage);

        // std::cout << "Resized Image out: " << dummyImage.rows  << ", " << dummyImage.cols << std::endl;

        // w.output->getCpuData().saveToPNG("/home/us000145/bitbucket/debug_out/basic_cnn/input_custom_image_format.png");
    }
}

void snn::ResizeProcessor::setScaling(bool scale) { this->scale = scale; }
