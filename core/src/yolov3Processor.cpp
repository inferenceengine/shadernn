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
#include "pch.h"
#include "yolov3Processor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include "ic2/dp.h"

using namespace std;
using namespace snn;

DECLARE_LAYER(YOLO);

void Yolov3Processor::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    const auto& inputDesc  = workload.inputs[0]->desc();
    const auto& outputDesc = workload.output->desc();

    // std::cout << "Output Width: " << outputDesc.width << std::endl;
    // std::cout << "Output Height: " << outputDesc.height << std::endl;

    if (!ic2_) {
        // for (std::size_t layerIdx = 0; layerIdx < dp.size(); layerIdx++) {
        //     std::cout << dp.at(layerIdx)->getName() << std::endl;
        // }

        dp::ShaderGenOptions options = {};
        options.desiredInput.width   = inputDesc.width;
        options.desiredInput.height  = inputDesc.height;
        options.desiredInput.depth   = 1;
        options.desiredInput.format  = outputDesc.format;
        options.compute              = this->compute_;

        options.desiredOutputFormat  = inputDesc.format;
        options.preferrHalfPrecision = false;

        options.mrtMode    = snn::MRTMode::DOUBLE_PLANE;
        options.weightMode = snn::WeightAccessMethod::TEXTURES;

        auto dp = snn::dp::loadFromJsonModel(modelFileName_, options.mrtMode, options.weightMode, options.preferrHalfPrecision);
        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);
        cp.dumpOutputs         = this->dumpOutputs;
        ic2_                   = MixedInferenceCore::create(cp);
    }

    SNN_ASSERT(inputDesc.device == Device::GPU);
    SNN_ASSERT(outputDesc.device == Device::GPU);
    auto inputTexture  = ((GpuFrameImage*) workload.inputs[0])->getGpuData();
    auto outputTexture = ((GpuFrameImage*) workload.output)->getGpuData();

    auto& currentFrameTexture = frameTextures_[inputTexture.texture];
    if (!currentFrameTexture) {
        currentFrameTexture = Texture::createAttached(inputTexture.target, inputTexture.texture); // Create a thin shell around textureId
    }

    std::vector<std::vector<std::vector<float>>> inVec, outVec = std::vector<std::vector<std::vector<float>>>();

    MixedInferenceCore::RunParameters rp = {};
    auto inputTextures                   = getFrameTexture(inputTexture.texture);
    rp.inputTextures                     = &inputTextures;
    rp.inputCount                        = 1;
    rp.inputMatrix                       = inVec;
    rp.output                            = outVec;
    // rp.transitionOutput = getFrameTexture(transitionOutputTexture.texture);
    rp.textureOut = getFrameTexture(outputTexture.texture);
    ic2_->run(rp);
    // ic2_->dumpStageOutputs("/data/data/com.innopeaktech.seattle.snndemo/files/yolov3_1");
}
