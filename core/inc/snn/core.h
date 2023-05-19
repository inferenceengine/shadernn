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

#include "snn/defines.h"
#include "snn/snn.h"
#include "snn/layeroption.h"
#include "snn/utils.h"
#include "snn/deviceTimer.h"
#include "snn/inferencegraph.h"
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace snn {

namespace dp {
class DeviceBackend;
}

typedef enum class Transition { Backend_CPU_GPU, Backend_GPU_CPU, NOT_DEFINED = 200 } Transition;

// This structure holds a state of a single render stage
struct RenderStage {
    RenderStage(GpuContext* context)
        : stageInputs(ImageTextureAllocator(context))
        , stageOutputs(ImageTextureAllocator(context))
    {}

    std::shared_ptr<InferenceGraph::Layer> layer;
    bool flattenLayer = false;  // True if fully-connected layer.
    std::shared_ptr<DeviceTimer> timer;
    Backend backend       = Backend::Backend_GPU;
    Transition transition = Transition::NOT_DEFINED;
    // Input images
    ImageTextureArray stageInputs;
    // Output images
    ImageTextureArray stageOutputs;
    // Indicies of the output images from previous layer(s) (or model input images array),
    // correspondent to stageInputs
    std::vector<int> inputIds;
    // This mask indicates that the input to the stage is model input image
    // 0: input to output binding happens at initialization time. Input is a previous hidden laer output.
    // 1: input to output binding happens at runtime time (delayed). Input is a model input image.
    std::vector<int> delayBindMask;
};

typedef ArrayParamAllocator<RenderStage, GpuContext*> RenderStagesArrayAllocator;

typedef FixedSizeArray<RenderStage, RenderStagesArrayAllocator> RenderStagesArray;

// Main class for SNN inference execution
class MixedInferenceCore {
public:
    typedef enum MixedLayerInputType { GL_TEXTURE_OBJECT = 1, FLOAT_VEC = 2 } MixedLayerInputType;

    virtual ~MixedInferenceCore();

    SNN_NO_MOVE(MixedInferenceCore);
    SNN_NO_COPY(MixedInferenceCore);

    // This structure holds runtime inference parameters
    struct RunParameters {
        // Input images
        ImageTextureArrayAccessor inputImages;
        // Output images
        ImageTextureArrayAccessor outputImages;
        // Input if located on CPU
        std::vector<std::vector<std::vector<float>>> inputMatrix;
        // Output if located on CPU
        std::vector<std::vector<std::vector<float>>> output;
        // Output from special model types
        SNNModelOutput modelOutput;
    };

    // Runs one inference of a model
    // params:
    //  rp - parameters, known at runtime
    void run(RunParameters& rp);

    struct CreationParameters : InferenceGraph {
        uint32_t outputWidth, outputHeight, outputDepth;
        bool dumpOutputs;
    };

    // Creates an instance of MixedInferenceCore given creation parameters
    // params:
    //  context - GPU context
    //  cp - creation parameters
    // returns:
    //  unique pointer to MixedInferenceCore 
    static std::unique_ptr<MixedInferenceCore> create(GpuContext* context, const CreationParameters& cp);

    // Creates an instance of MixedInferenceCore given model file name
    // params:
    //  context - GPU context
    //  modelFileName - model file in JSON format
    //  options - shader generation options
    //  dumpOutputs - flag to dump layer outputs when running an inference
    // returns:
    //  unique pointer to MixedInferenceCore 
    static std::unique_ptr<MixedInferenceCore> create(GpuContext* context, const std::string& modelFileName,
        const dp::ShaderGenOptions& options, bool dumpOutputs = false);

    // Writes timing statistics.
    // Used for profiling and benchmarking.
    // params:
    //  timeArray - a map with keys of layer names and values of timing of successive runs
    void writeTimeStat(std::map<std::string, std::vector<double>>& timeArray);

private:
    GpuContext* context;

    // Flag, indicating that we need to bind model output
    // to the model output image(s)
    // Currently we do this for Android only
    bool bindOutput = true;

    CreationParameters cp;
    RenderStagesArray stages;

    dp::DeviceBackend* backend = NULL;
    DeviceTimer* gpuRunTime = NULL;

    Timer cpuRunTime = Timer("IC2 Total CPU Runtime");
    // Model output if located on CPU
    std::vector<std::vector<float>> output;

    std::string printTimingStats() const;
    MixedInferenceCore(GpuContext* context_);

    bool init(const CreationParameters& cp);
};

} // namespace snn
