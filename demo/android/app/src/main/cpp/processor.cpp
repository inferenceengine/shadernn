
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

#include "pch.h"
#include "processor.h"
#include "demoutils.h"
#include "snn/snn.h"
#include "appContext.h"
#ifdef SUPPORT_GL
    #include "glGpuFrameImage.h"
    #include "glCpuFrameImage.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "volk.h"
    #include "vulkanGpuFrameImage.h"
    #include "vulkanCpuFrameImage.h"
#endif

namespace snn {

std::unique_ptr<FrameImage2> createFrameImage(const FrameImage2::Desc& d) {
    switch (d.device) {
    case Device::GPU:
        switch (AppContext::getContext()->backendType) {
#ifdef SUPPORT_GL
            case GpuBackendType::GL:
                return std::make_unique<GlGpuFrameImage>(d);
#endif
#ifdef SUPPORT_VULKAN
            case GpuBackendType::VULKAN:
                return std::make_unique<VulkanGpuFrameImage>(d);
#endif
            default:
                SNN_RIP("Unexpected GpuBackendType");
        }
        break;
    case Device::CPU:
        switch (AppContext::getContext()->backendType) {
#ifdef SUPPORT_GL
            case GpuBackendType::GL:
                return std::make_unique<GlCpuFrameImage>(d);
#endif
#ifdef SUPPORT_VULKAN
            case GpuBackendType::VULKAN:
                return std::make_unique<VulkanCpuFrameImage>(d);
#endif
            default:
                SNN_RIP("Unexpected GpuBackendType");
        }
    }
    return nullptr;
}

std::ostream& operator<<(std::ostream& os, const Processor::FrameVectorDesc& desc) {
    os << "Size: " << desc.size << std::endl;
    switch (desc.device) {
    case Device::GPU:
        os << "Device: GPU" << std::endl;
        break;
    case Device::CPU:
        os << "Device: CPU" << std::endl;
        break;
    default:
        break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Processor::Desc& desc) {
    os << "Input: " << std::endl << desc.i;
    os << std::endl << "Output: " << desc.o << std::endl;
    return os;
}

}
