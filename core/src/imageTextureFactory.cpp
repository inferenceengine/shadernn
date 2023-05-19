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
#include "snn/imageTextureFactory.h"
#include "snn/utils.h"
#ifdef SUPPORT_GL
    #include "imageTextureGL.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "imageTextureVulkan.h"
#endif

namespace snn {

std::shared_ptr<ImageTexture> ImageTextureFactory::createImageTexture(GpuContext* context) {
    std::shared_ptr<ImageTexture> tptr;
    switch (context->backendType) {
#ifdef SUPPORT_GL
    case GpuBackendType::GL:
        tptr.reset(new ImageTextureGL());
        break;
#endif
#ifdef SUPPORT_VULKAN
    case GpuBackendType::VULKAN:
        tptr.reset(new ImageTextureVulkan(context));
        break;
#endif
    default:
        SNN_CHK(false);
    }
    return tptr;
}

std::shared_ptr<ImageTexture> ImageTextureFactory::createImageTexture(GpuContext* context, const std::array<uint32_t, 4>& dims, ColorFormat format,
    const void* buffer, const std::string& name) {
    std::shared_ptr<ImageTexture> tptr;
    switch (context->backendType) {
#ifdef SUPPORT_GL
    case GpuBackendType::GL:
        tptr.reset(new ImageTextureGL(dims, format, buffer, name));
        break;
#endif
#ifdef SUPPORT_VULKAN
    case GpuBackendType::VULKAN:
        tptr.reset(new ImageTextureVulkan(context, dims, format, buffer, name));
        break;
#endif
    default:
        SNN_CHK(false);
    }
    return tptr;
}

std::shared_ptr<ImageTexture> ImageTextureFactory::createImageTexture(GpuContext* context, const std::string& fileName) {
    std::shared_ptr<ImageTexture> tptr;
    switch (context->backendType) {
#ifdef SUPPORT_GL
    case GpuBackendType::GL:
        tptr.reset(new ImageTextureGL(fileName));
        break;
#endif
#ifdef SUPPORT_VULKAN
    case GpuBackendType::VULKAN:
        tptr.reset(new ImageTextureVulkan(context, fileName));
        break;
#endif
    default:
        SNN_CHK(false);
    }
    return tptr;
}

} // namespace snn
