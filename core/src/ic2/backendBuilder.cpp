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

#include "backendBuilder.h"
#include "snn/utils.h"
#ifdef SUPPORT_GL
    #include "openGLBackend.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "vulkanBackend.h"
#endif
namespace snn {
namespace dp { // short for Dynamic Pipeline

DeviceBackend* BackendBuilder::build(GpuContext* context, const InferenceGraph& ig) {
    (void) ig;
    SNN_ASSERT(context);
    DeviceBackend* backendPtr = nullptr;
    switch (context->backendType) {
#ifdef SUPPORT_GL
    case GpuBackendType::GL:
        {
            OpenGLBackend::CreationParameters glCP {ig.mrtMode, ig.weightMode};
            backendPtr = new OpenGLBackend(glCP);
        }
        break;
#endif
#ifdef SUPPORT_VULKAN
    case GpuBackendType::VULKAN:
        backendPtr = new VulkanBackend(context);
        break;
#endif
    default:
        SNN_CHK(false);
    }
    return backendPtr;
}

} // namespace dp
} // namespace snn
