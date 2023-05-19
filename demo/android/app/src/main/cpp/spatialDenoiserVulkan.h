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

#include "volk.h"
#include "spatialDenoiser.h"
#include "snn/color.h"
#include "genericModelProcessorVulkan.h"
#include <string>
#include <memory>

namespace snn {

class SpatialDenoiserVulkan : public SpatialDenoiser {
public:
    SpatialDenoiserVulkan(ColorFormat format, Precision precision, bool dumpOutputs);
    virtual ~SpatialDenoiserVulkan() = default;
    void init(const FrameDims& inputDims_, const FrameDims& outputDims_) override;
    void submit(Workload&) override;

private:
    std::unique_ptr<GenericModelProcessorVulkan> genericModelProcessorVulkan;
};

}
