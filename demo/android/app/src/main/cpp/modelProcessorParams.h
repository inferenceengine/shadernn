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

#include <string>
#include "processor.h"
#include "snn/core.h"

namespace snn {

struct ModelProcessorParams {
    std::shared_ptr<MixedInferenceCore> ic2;
    std::string modelFileName;
    Processor::FrameDims modelDims;
    Precision precision;
    bool compute;
    bool dumpOutputs;

    ModelProcessorParams(const std::string& modelFilename_, Processor::FrameDims modelDims_, Precision precision_, bool compute_, bool dumpOutputs_)
        : modelFileName(modelFilename_)
        , modelDims(modelDims_)
        , precision(precision_)
        , compute(compute_)
        , dumpOutputs(dumpOutputs_)
    {}
};

} // namespace snn
