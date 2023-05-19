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

#include "snn/snn.h"
#include "snn/imageTexture.h"
#include "snn/core.h"
#include "ic2/layerFactory.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>

const float MAX_TEST_TIME_SECONDS = 20.0f;

namespace snn {

// This class is a front-end to MixedInferenceCore
// It is used for benchmarking/testing
class InferenceProcessor {
public:
    struct InitializationParameters {
        std::string modelName;
        std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
        snn::MRTMode mrtMode;
        snn::WeightAccessMethod weightMode;
        bool halfPrecision    = false;
        bool dumpOutputs      = false;
        bool useComputeShader = false;
        bool useVulkanShader  = false;
        ModelType modelType = ModelType::OTHER;
        uint32_t maxLoops = 1;
    };

    // Creates an object of InferenceProcessor class
    // params:
    //  useVulkan - flag indicating whether to use Vulkan platform
    // return:
    //  Shared pointer to a new object of InferenceProcessor class
    static std::shared_ptr<InferenceProcessor> create(bool useVullkan);

    // Initializes an object of InferenceProcessor class
    // params:
    //  cp - initialization parameters
    void initialize(const InitializationParameters& cp);

    // Performs actions before each inference
    // params:
    //  inputTex - input images
    int32_t preProcess(ImageTextureArrayAccessor inputTex);

    // Runs an inference
    // params:
    //  outputTex - output images
    int32_t process(ImageTextureArrayAccessor outputTex);

    // Returns GPU context
    // return:
    //  Pointer to GPU context class object
    GpuContext* getContext() const { return _context; }

private:
    // Constructor
    // params:
    //  context - pointer to GPU context
    InferenceProcessor(GpuContext* context);

    GpuContext* _context;
    std::shared_ptr<MixedInferenceCore> _ic2;
    std::string _modelFileName;
    std::vector<std::pair<std::string, std::vector<uint32_t>>> _inputList;
    ImageTextureArray _inputTexs;
    ImageTextureArray _outputTexs;
    bool dumpOutputs, halfPrecision;
    ModelType modelType = ModelType::OTHER;
    uint32_t maxLoops = 1;
    std::chrono::time_point<std::chrono::high_resolution_clock> startRun;
};

} // namespace snn
