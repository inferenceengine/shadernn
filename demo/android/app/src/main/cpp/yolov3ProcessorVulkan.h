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
#include "yolov3Processor.h"
#include "vulkanContext.h"
#include "snn/color.h"
#include "colorVulkan.h"
#include "ic2/dp.h"
#include "uvkc/vulkan/image.h"
#include "genericModelProcessorVulkan.h"
#include "vulkan/bBoxes.h"
#include <memory>

namespace snn {

class Yolov3ProcessorVulkan : public Yolov3Processor {
public:
    Yolov3ProcessorVulkan(ColorFormat format, Precision precision, bool dumpOutputs);
    virtual ~Yolov3ProcessorVulkan();
    void init(const FrameDims& inputDims_, const FrameDims& outputDims_) override;
    void submit(Workload&) override;

private:
    void drawBBoxes(MixedInferenceCore::RunParameters& rp);
    void copy22DImage(VkImage srcImage);

    VulkanGpuContext* context = nullptr;
    std::unique_ptr<GenericModelProcessorVulkan> genericModelProcessorVulkan;
    VulkanBBoxes bBoxes;
    VkImage imageOut = VK_NULL_HANDLE;
    VkDeviceMemory imageOutMemory = VK_NULL_HANDLE;
    VkImageView imageOutView2D = VK_NULL_HANDLE;
    std::shared_ptr<uvkc::vulkan::Image> uvkcImageOut;
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
};

}
