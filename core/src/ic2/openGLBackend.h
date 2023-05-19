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

#include "backend.h"
#include "snn/snn.h"
#include "snn/utils.h"
#include "snn/imageTexture.h"
#include "glUtils.h"
#include "snn/core.h"
#include <string>
#include <vector>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class GenericModelLayer;

// This class implements OPenGL backend
class OpenGLBackend : public DeviceBackend {
public:
    struct CreationParameters {
        MRTMode mrtMode               = MRTMode::DOUBLE_PLANE;        // set during generateInferenceGraph. Defaults to SINGLE_PLANE
        WeightAccessMethod weightMode = WeightAccessMethod::TEXTURES; // set during generateInferenceGraph. Defaults to TEXTURE
    };

    // Constructor
    // params:
    //  cp - creation parameters
    OpenGLBackend(const CreationParameters& cp);

    virtual ~OpenGLBackend() = default;

    SNN_NO_COPY(OpenGLBackend);
    SNN_NO_MOVE(OpenGLBackend);

    // Initializes render passes for a model layer
    // params:
    //  modelLayer - model layer
    //  texInputs - input images
    //  texOutputs - output images
    void initRenderPasses(dp::GenericModelLayer* modelLayer, ImageTextureArrayAccessor texInputs, ImageTextureArrayAccessor texOutputs) override;

    // Actions, performed before inference run
    // params:
    //  rp - run parameters
    //  stages - array of render stages
    //  bindOutput - flag, indicating whether we need to bind output image to external image
    //  bindIndex - the index of the layer to bind
    void prepareRun(MixedInferenceCore::RunParameters& rp,
            RenderStagesArray &stages, bool bindOutput, uint32_t bindIndex) override;

    // Actions, performed before each layer run
    // params:
    //  rp - run parameters
    //  stage - render stage
    void prepareStage(MixedInferenceCore::RunParameters& rp, RenderStage& stage) override;

    // Wait on GPU after the inference
    bool sync() override;

    // Actions, performed after all other actions of the inference run
    void cleanupRun() override;

    // Checks if profiling is enabled
    // params:
    //  queryPerLayerTime - check per layer or not
    // return:
    //  true if profiling is enabled; false if not
    bool isProfilingEnabled(bool queryPerLayerTime = false) override;

    // Creates a device timer
    // params:
    //  name - timer's name
    // returns:
    //  a pointer to a new device timer object
    DeviceTimer* createDeviceTimer(const std::string& name) override;

private:
    CreationParameters _cp;
    gl::DebugSSBO debugger;
    std::vector<gl::SamplerObject> weightSamplers;
    gl::SamplerObject sampler, sampler2;
    gl::GpuTimestamps timestamps;

    std::vector<GLuint> samplers;
    std::vector<GLuint> weightSamplersUint;
    size_t runCounter = 0;
};

}; // namespace dp
} // namespace snn
