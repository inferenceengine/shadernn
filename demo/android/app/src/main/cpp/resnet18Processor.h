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

static constexpr const char* RESNET18_MODEL_NAME = "Resnet18/resnet18_cifar10.json";

namespace snn {

class ResNet18Processor : public Processor, protected ModelProcessorParams {
public:
    ~ResNet18Processor() = default;

    static std::unique_ptr<ResNet18Processor> createResNet18Processor(ColorFormat format, Precision precision, bool compute = false, bool dumpOutputs = false);

    std::string getModelName() override { return "ResNet18 Model"; }

    static const uint32_t MODEL_IMAGE_WIDTH = 32;
    static const uint32_t MODEL_IMAGE_HEIGHT = 32;

protected:
    ResNet18Processor(ColorFormat format, Precision precision, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}})
        , ModelProcessorParams(RESNET18_MODEL_NAME, {MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, 1, 4}, precision, compute, dumpOutputs)
    {}
};

} // namespace snn
