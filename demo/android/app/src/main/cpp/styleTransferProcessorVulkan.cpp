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
#include "styleTransferProcessorVulkan.h"
#include "snn/utils.h"
#include "ic2/dp.h"

namespace snn {

StyleTransferProcessorVulkan::StyleTransferProcessorVulkan(ColorFormat format, const std::string& modelFileName, Precision precision, bool dumpOutputs)
    : StyleTransferProcessor(format, modelFileName, precision, false, dumpOutputs)
{}

void StyleTransferProcessorVulkan::init(const FrameDims& inputDims_, const FrameDims& outputDims_) {
    Processor::init(inputDims_, outputDims_);

    dp::ShaderGenOptions options = {};
    options.preferrHalfPrecision = (precision == Precision::FP16);
    auto inputTex = InferenceGraph::IODesc {options.preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F,
                                            modelDims.width, modelDims.height, 1, 4};
    options.desiredInput.push_back(inputTex);

    options.desiredOutputFormat  = options.preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;

    genericModelProcessorVulkan.reset(new GenericModelProcessorVulkan(*this,
        outputDims_,
        options,
        true,   // resizeImage_
        false,   // passThrough_
        0.0f,   // modelInputMean_
        255.0f, // modelInputNorm_
        0.0f,   // modelOutputMean_
        1.0f / 255.0f));  // modelOutputNorm_
}

void StyleTransferProcessorVulkan::submit(Workload& workload) {
    MixedInferenceCore::RunParameters rp = {};
    genericModelProcessorVulkan->submit(workload, rp);
}

}
