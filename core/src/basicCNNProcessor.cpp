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
#include "basicCNNProcessor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include "ic2/dp.h"

using namespace std;
using namespace snn;

void BasicCNNProcessor::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    const auto& inputDesc  = workload.inputs[0]->desc();
    const auto& outputDesc = workload.output->desc();
    (void) outputDesc;

    // auto image = castTo<GpuFrameImage>(workload.inputs[0])->dumpTexture().data();

    // std::cout << "------------- [Debug image values] -------------" << std::endl;
    // for (std::size_t i = 0; i < 11; i++) {
    //     std::cout << "At " << i << ": " << (int) image[i] << std::endl;
    // }
    // std::cout << "------------------------------------------------" << std::endl;

    if (!ic2) {
        // for (std::size_t layerIdx = 0; layerIdx < dp.size(); layerIdx++) {
        //     std::cout << dp.at(layerIdx)->getName() << std::endl;
        // }

        dp::ShaderGenOptions options = {};
        options.desiredInput.width   = inputDesc.width;
        options.desiredInput.height  = inputDesc.height;
        options.desiredInput.depth   = 1;
        options.desiredInput.format  = inputDesc.format;
        options.compute              = this->compute;
        options.mrtMode              = snn::MRTMode::DOUBLE_PLANE;
        options.weightMode           = snn::WeightAccessMethod::TEXTURES;

        options.desiredOutputFormat  = inputDesc.format;
        options.preferrHalfPrecision = false;

        auto dp = snn::dp::loadFromJsonModel(modelName, options.mrtMode, options.weightMode);

        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);
        cp.dumpOutputs         = this->dumpOutputs;
        ic2                    = MixedInferenceCore::create(cp);
    }

    SNN_ASSERT(inputDesc.device == Device::GPU);
    SNN_ASSERT(outputDesc.device == Device::GPU);
    // const auto& fmtDesc = getColorFormatDesc(outputDesc.format);

    // const auto& inFmtDesc = getColorFormatDesc(inputDesc.format);

    MixedInferenceCore::RunParameters rp = {};

#ifdef LOCAL_DEBUG
    auto input = ManagedRawImage::loadFromAsset("images/cat_224x224.jpg");
    gl::TextureObject inputTexture;
    inputTexture.allocate2D(input.format(), input.width(), input.height());
    inputTexture.setPixels(0, 0, 0, input.width(), input.height(), input.pitch(), input.data());
    const gl::TextureObject* inputs[] = {&inputTexture};
    rp.inputTextures                  = inputs;
#else
    auto inputTexture = ((GpuFrameImage*) workload.inputs[0])->getGpuData();

    // auto outputTexture = ((GpuFrameImage*)workload.output)->getGpuData();
    auto& currentFrameTexture = frameTextures[inputTexture.texture];
    // auto & transititonFrameTexture = frameTextures[ic2->transitionLayerIndex];
    if (!currentFrameTexture) {
        currentFrameTexture = Texture::createAttached(inputTexture.target, inputTexture.texture); // Create a thin shell around textureId
    }
    auto inputs      = getFrameTexture(inputTexture.texture);
    rp.inputTextures = &inputs;
#endif

    // if (!transititonFrameTexture) {
    //     transititonFrameTexture = Texture::createAttached(outputTexture.target, outputTexture.texture);
    // }
    // auto inputTextures = getFrameTexture(inputTexture.texture);
    rp.inputCount = 1;

    // rp.textureOut = getFrameTexture(outputTexture.texture);
    // rp.transitionOutput = getFrameTexture(transitionOutputTexture.texture);
    rp.inputMatrix = workload.cpuInputs;
    rp.output      = std::vector<std::vector<std::vector<float>>>();
    ic2->run(rp);
}
