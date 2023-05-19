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
#include "spatialDenoiserGL.h"
#include "glAppContext.h"
#include "imageTextureGL.h"
#include <glad/glad.h>
#include "snn/glImageHandle.h"
#include "ic2/dp.h"

using namespace std;
using namespace snn;

// -----------------------------------------------------------------------------
// the following is spatialDenoiser
// -----------------------------------------------------------------------------

void SpatialDenoiserGL::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    const auto& inputDesc = workload.inputs[0]->desc();
    const auto& outDesc   = workload.output->desc();
    SNN_ASSERT(inputDesc.device == Device::GPU);
    SNN_ASSERT(outDesc.device == Device::GPU);

    // we have to delay creating ic2 because we need to know the frame size.
    if (!ic2) {
        dp::ShaderGenOptions options = {};
        auto inputTex = InferenceGraph::IODesc {inputDesc.format,
                                                inputDesc.width, inputDesc.height, 1, 4};
        options.desiredInput.push_back(inputTex);

        options.desiredOutputFormat  = outDesc.format;
        options.preferrHalfPrecision = precision == Precision::FP16;
        options.compute              = compute;
        options.mrtMode              = snn::MRTMode::SINGLE_PLANE;
        options.weightMode           = snn::WeightAccessMethod::CONSTANTS;
        auto dp                      = snn::dp::loadFromJsonModel(modelFileName, false, options.mrtMode, options.weightMode, options.preferrHalfPrecision);
        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp[0], options);
        cp.dumpOutputs = dumpOutputs;
        ic2                   = MixedInferenceCore::create(GlAppContext::getGlContext(), cp);
    }

    GlImageHandle inputTexture;
    workload.inputs[0]->getGpuImageHandle(inputTexture);
    GlImageHandle outputTexture;
    workload.output->getGpuImageHandle(outputTexture);

    MixedInferenceCore::RunParameters rp = {};
    ImageTextureGLArray inputImageTexs;
    inputImageTexs.allocate(1);
    inputImageTexs[0].texture(0)->attach(inputTexture.target, inputTexture.textureId);
    rp.inputImages = inputImageTexs;

    ImageTextureGLArray outputImageTexs;
    outputImageTexs.allocate(1);
    outputImageTexs[0].texture(0)->attach(outputTexture.target, outputTexture.textureId);
    rp.outputImages = outputImageTexs;

    ic2->run(rp);
}
