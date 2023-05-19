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

#include "volk.h"
#include "processor.h"
#include "modelProcessorParams.h"
#include "imageTextureVulkan.h"
#include "snn/color.h"
#include "snn/core.h"
#include "snn/layeroption.h"
#include "vulkanContext.h"
#include "ic2/dp.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "vulkanImageResizeOp.h"
#include <string>
#include <memory>
#include <functional>

namespace snn {

class GenericModelProcessorVulkan {
public:
    GenericModelProcessorVulkan(ModelProcessorParams& modelProcessorParams_,
        const Processor::FrameDims& outputDims_,
        dp::ShaderGenOptions options_,
        bool resizeImage_,
        bool inputImagePassThrough_ = false,
        float modelInputMean_ = 0.0f,
        float modelInputNorm_ = 1.0f,
        float modelOutputMean_ = 0.0f,
        float modelOutputNorm_ = 1.0f
    );

    ~GenericModelProcessorVulkan() = default;

    void submit(Processor::Workload& workload, MixedInferenceCore::RunParameters& rp);

    typedef std::function<void(MixedInferenceCore::RunParameters& rp)> PostRun;

    void setPostRun(PostRun postRun_) {
        postRun = postRun_;
        postRunSet = true;
    }

private:
    ModelProcessorParams& modelProcessorParams;
    Processor::FrameDims outputDims;
    dp::ShaderGenOptions options;
    bool resizeImage;
    bool inputImagePassThrough;
    float modelInputMean;
    float modelInputNorm;
    float modelOutputMean;
    float modelOutputNorm;
    std::shared_ptr<uvkc::vulkan::Image> inputUvkcImage;
    std::shared_ptr<uvkc::vulkan::Image> outputUvkcImage;
    PostRun postRun;
    bool postRunSet = false;

    std::shared_ptr<snn::ImageTextureVulkan> modelInput;
    std::shared_ptr<snn::ImageTextureVulkan> modelOutput;
    std::unique_ptr<VulkanImageResizeOp> inputImageResizeOp;
    std::unique_ptr<VulkanImageResizeOp> outputImageResizeOp;
    snn::VulkanGpuContext* vulkanContext = nullptr;
    uvkc::benchmark::VulkanContext* uvkcContext = nullptr;
};

}
