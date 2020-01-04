/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <string>

#include "snn/color.h"
#include "ic2/core.h"
#include "ic2/dp.h"

namespace snn {
class InferenceProcessor {
public:
    struct InitializationParameters {
        std::string modelName;
        std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
        gl::RenderContext* rc;
        snn::MRTMode mrtMode;
        snn::WeightAccessMethod weightMode;
        bool halfPrecision    = false;
        bool dumpOutputs      = false;
        bool useComputeShader = false;
    };

    InferenceProcessor()  = default;
    ~InferenceProcessor() = default;

    int32_t initialize(const InitializationParameters cp);
    int32_t finalize(void);
    int32_t preProcess(FixedSizeArray<snn::ImageTexture>& inputTex);
    int32_t process(FixedSizeArray<snn::ImageTexture>& outputTex);
    bool registerLayer(std::string layerName, snn::dp::LayerCreator creator);

    std::string getModelName() { return _modelFileName; }

    std::vector<uint32_t> getInputDims(uint32_t idx) { return _inputList[idx].second; }

private:
    std::shared_ptr<MixedInferenceCore> _ic2;
    std::string _modelFileName;
    std::vector<std::pair<std::string, std::vector<uint32_t>>> _inputList;
    FixedSizeArray<snn::ImageTexture> _inputTexs;
    FixedSizeArray<snn::ImageTexture> _outputTexs;
    gl::RenderContext* _rc;
    bool dumpOutputs, halfPrecision;
};
} // namespace snn
