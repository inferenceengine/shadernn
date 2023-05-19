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

#include "mobileNetV2Processor.h"
#include "genericClassifierProcessorGL.h"

#include <memory>

namespace snn {

class MobileNetV2ProcessorGL : public MobileNetV2Processor {
public:
    MobileNetV2ProcessorGL(ColorFormat format, Precision precision, bool compute, bool dumpOutputs)
        : MobileNetV2Processor(format, precision, compute, dumpOutputs)
    {
        genericClassifierProcessorGL.reset(
            new GenericClassifierProcessorGL(*this, 0.0f /*means*/, 1.0f /*norms*/, WeightAccessMethod::TEXTURES)
        );
    }

    virtual ~MobileNetV2ProcessorGL() = default;
    void submit(Workload& w) override
    {
        genericClassifierProcessorGL->submit(w);
    }

private:
    std::unique_ptr<GenericClassifierProcessorGL> genericClassifierProcessorGL;
};

} // namespace snn
