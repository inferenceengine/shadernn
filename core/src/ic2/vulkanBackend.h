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
#include "snn/core.h"
#include "snn/snn.h"
#include "snn/imageTexture.h"
#include "snn/deviceTimer.h"
#include "uvkc/benchmark/vulkan_context.h"
#include <string>
#include <memory>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class GenericModelLayer;

// This class implements OPenGL backend
class VulkanBackend : public DeviceBackend {
public:
    // Constructor
    // params:
    //  cp - creation parameters
    VulkanBackend(GpuContext* context_);

    virtual ~VulkanBackend() = default;

    SNN_NO_COPY(VulkanBackend);
    SNN_NO_MOVE(VulkanBackend);

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

    // Actions, performed after the inference run
    // params:
    //  stages - array of render stages
    //  dumpOutput - flag indicating whether to dump layers outputs
    //  folder - directory name where to dump outputs
    void postRun(RenderStagesArray &stages, bool dumpOutput, const std::string &folder) override;

    // Wait on GPU after the inference
    bool sync() override;

    // Actions, performed after all other actions of the inference run
    void cleanupRun() override;

    // Creates a device timer
    // params:
    //  name - timer's name
    // returns:
    //  a pointer to a new device timer object
    DeviceTimer* createDeviceTimer(const std::string& name) override;

private:
    GpuContext* context;
    std::unique_ptr<uvkc::vulkan::Sampler> _sampler0, _sampler1, _sampler2;
    std::unique_ptr<uvkc::vulkan::Sampler> _weightSampler0;
    std::unique_ptr<uvkc::vulkan::CommandBuffer> _cmdBuffer;
    uvkc::vulkan::Device* _device;
    bool _isSynced = false;
};

} // namespace dp
} // namespace snn
