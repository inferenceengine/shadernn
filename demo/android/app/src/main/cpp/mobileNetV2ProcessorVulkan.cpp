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
#include "mobileNetV2ProcessorVulkan.h"
#include "snn/utils.h"
#include "ic2/dp.h"

namespace snn {

MobileNetV2ProcessorVulkan::MobileNetV2ProcessorVulkan(ColorFormat format, Precision precision, bool dumpOutputs)
        : MobileNetV2Processor(format, precision, false, dumpOutputs)
{}

void MobileNetV2ProcessorVulkan::init(const FrameDims& inputDims_, const FrameDims& outputDims_) {
    Processor::init(inputDims_, outputDims_);

    dp::ShaderGenOptions options = {};
    options.preferrHalfPrecision = (precision == Precision::FP16);

    auto inputTex = InferenceGraph::IODesc {options.preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F,
                                            modelDims.width, modelDims.height, 1, 4};
    options.desiredInput.push_back(inputTex);

    options.desiredOutputFormat  = options.preferrHalfPrecision ? snn::ColorFormat::RGBA16F : snn::ColorFormat::RGBA32F;
    options.mrtMode              = snn::MRTMode::SINGLE_PLANE;
    options.weightMode           = snn::WeightAccessMethod::TEXTURES;

    genericModelProcessorVulkan.reset(new GenericModelProcessorVulkan(*this,
        outputDims_,
        options,
        true, // resizeImage_
        true, // inputImagePassThrough
        0.0f, // modelInputMean_
        1.0f, // modelInputNorm_
        0.0f, // modelOutputMean_
        1.0f));  // modelOutputNorm_
}

void MobileNetV2ProcessorVulkan::submit(Workload& workload) {
    MixedInferenceCore::RunParameters rp = {};
    rp.inputMatrix                       = workload.cpuInputs;
    rp.modelOutput.modelType             = ModelType::CLASSIFICATION;

    genericModelProcessorVulkan->submit(workload, rp);

    workload.modelOutput = rp.modelOutput;
}

}
