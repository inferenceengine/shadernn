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
#include "spatialDenoiserVulkan.h"
#include "snn/utils.h"
#include "ic2/dp.h"

namespace snn {

SpatialDenoiserVulkan::SpatialDenoiserVulkan(ColorFormat format, Precision precision, bool dumpOutputs)
    : SpatialDenoiser(format, precision, false, dumpOutputs)
{}

void SpatialDenoiserVulkan::init(const FrameDims& inputDims_, const FrameDims& outputDims_) {
    Processor::init(inputDims_, outputDims_);

    modelDims.width = inputDims_.width;
    modelDims.height = inputDims_.height;

    dp::ShaderGenOptions options = {};
    auto inputTex = InferenceGraph::IODesc {desc().i.format,
                                            inputDims_.width, inputDims_.height, 1, 4};
    options.desiredInput.push_back(inputTex);

    options.desiredOutputFormat  = desc().o.format;
    options.preferrHalfPrecision = (precision == Precision::FP16);
    options.mrtMode              = snn::MRTMode::SINGLE_PLANE;
    options.weightMode           = snn::WeightAccessMethod::CONSTANTS;

    genericModelProcessorVulkan.reset(new GenericModelProcessorVulkan(*this,
        outputDims_,
        options,
        false));   // resizeImage_
}

void SpatialDenoiserVulkan::submit(Workload& workload) {
    MixedInferenceCore::RunParameters rp = {};
    genericModelProcessorVulkan->submit(workload, rp);
}

}
