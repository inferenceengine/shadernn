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

static constexpr const char* MOBILENETV2_MODEL_NAME = "MobileNetV2/mobilenetV2.json";

namespace snn {

class MobileNetV2Processor : public Processor, protected ModelProcessorParams {
public:
    ~MobileNetV2Processor() = default;

    static std::unique_ptr<MobileNetV2Processor> createMobileNetV2Processor(ColorFormat format, Precision precision,
        bool compute = false, bool dumpOutputs = false);

    std::string getModelName() override { return "MobileNetV2 Model"; }

    static const std::size_t MODEL_IMAGE_WIDTH = 224;
    static const std::size_t MODEL_IMAGE_HEIGHT  = 224;

protected:
    MobileNetV2Processor(ColorFormat format, Precision precision, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}})
        , ModelProcessorParams(MOBILENETV2_MODEL_NAME, {MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, 1, 4}, precision, compute, dumpOutputs)
    {}
};

} // namespace snn
