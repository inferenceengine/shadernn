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

#include "styleTransferProcessor.h"
#include <memory>
#include <unordered_map>

namespace snn {

class StyleTransferProcessorGL : public StyleTransferProcessor {
public:
    StyleTransferProcessorGL(ColorFormat format, const std::string& modelFileName, Precision precision, bool compute, bool dumpOutputs)
        : StyleTransferProcessor(format, modelFileName, precision, compute, dumpOutputs)
    {}
    virtual ~StyleTransferProcessorGL() = default;
    void submit(Workload&) override;
};

}
