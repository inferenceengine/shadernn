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

#include "processor.h"
#include "modelProcessorParams.h"

#include <string>
#include <memory>

static constexpr const char* SPATIAL_DENOISE_MODEL_NAME = "SpatialDenoise/spatialDenoise.json";

namespace snn {

class SpatialDenoiser : public Processor, protected ModelProcessorParams {
public:
    virtual ~SpatialDenoiser() = default;
    std::string getModelName() override { return "Spatial Denoiser"; }

    static std::unique_ptr<SpatialDenoiser> createPreDenoiser(ColorFormat format, Precision precision, bool compute = false, bool dumpOutputs = false);

protected:
    SpatialDenoiser(ColorFormat format, Precision precision, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}})
        , ModelProcessorParams(SPATIAL_DENOISE_MODEL_NAME, {0, 0, 1, 4}, precision, compute, dumpOutputs)  // Dimensions will be updated in init()
    {}
};

} // namespace snn
