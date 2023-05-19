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

#include <memory>
#include <string>
#include <vector>

static constexpr const char* YOLOV3_MODEL_NAME = "Yolov3-tiny/yolov3-tiny_finetuned.json";

namespace snn {

class Yolov3Processor : public Processor, protected ModelProcessorParams {
public:
    virtual ~Yolov3Processor() = default;
    static std::unique_ptr<Yolov3Processor> createYolov3Processor(ColorFormat format, Precision precision, bool compute = false, bool dumpOutputs = false);

    std::string getModelName() override { return "Yolov3 Model"; }

    static const uint32_t MODEL_IMAGE_WIDTH = 416;
    static const uint32_t MODEL_IMAGE_HEIGHT = 416;

    static void getBBoxesCoords(const std::vector<float>& boxDetails, std::vector<float>& coords);

protected:
    Yolov3Processor(ColorFormat format, Precision precision, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}})
        , ModelProcessorParams(YOLOV3_MODEL_NAME, {MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, 1, 4}, precision, compute, dumpOutputs)
    {}
};

} // namespace snn

