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
#include "snn/utils.h"
#include "snn/imageTexture.h"
#include "snn/deviceTimer.h"
#include "snn/core.h"
#include <string>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class GenericModelLayer;

// This is a base class for platform-specific "backends"
// Backend implements common functionality, independent of
// specific model layers
class DeviceBackend {
public:
    DeviceBackend() = default;

    virtual ~DeviceBackend() = default;

    SNN_NO_COPY(DeviceBackend);
    SNN_NO_MOVE(DeviceBackend);

    // Initializes render passes for a model layer
    // params:
    //  modelLayer - model layer
    //  texInputs - input images
    //  texOutputs - output images
    virtual void initRenderPasses(GenericModelLayer* modelLayer, ImageTextureArrayAccessor texInputs, ImageTextureArrayAccessor texOutputs) {
        (void) modelLayer;
        (void) texInputs;
        (void) texOutputs;
    }

    // Actions, performed before inference run
    virtual void prepareRun(MixedInferenceCore::RunParameters& rp,
            RenderStagesArray &stages, bool bindOutput, uint32_t bindIndex) {
        (void) rp;
        (void) stages;
        (void) bindOutput;
        (void) bindIndex;
    }

    // Actions, performed before each layer run
    virtual void prepareStage(MixedInferenceCore::RunParameters& rp, RenderStage& stage) {
        (void) rp;
        (void) stage;
    }

    // Actions, performed after the inference run
    virtual void postRun(RenderStagesArray &stages, bool dumpOutput, const std::string &folder) {
        (void) stages;
        (void) dumpOutput;
        (void) folder;
    }

    // Abstract actions to wait on GPU after the inference
    virtual bool sync() {
        return 0;
    }

    // Actions, performed after all other actions of the inference run
    virtual void cleanupRun() {}

    virtual DeviceTimer* createDeviceTimer(const std::string& name) {
        (void) name;
        return NULL;
    }

    virtual bool isProfilingEnabled(bool queryPerLayerTime = false) {
        (void) queryPerLayerTime;
        return true;
    }
};

}; // namespace dp
} // namespace snn
