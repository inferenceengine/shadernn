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

#include "../pch.h"
#include "snn/snn.h"
#include "appContext.h"
#include "demoutils.h"
#ifdef SUPPORT_GL
    #include "glFrameviz.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "vulkanFrameViz.h"
#endif

namespace snn {

NightVisionFrameViz::NightVisionFrameViz() {
    switch (AppContext::getContext()->backendType) {
#ifdef SUPPORT_GL
        case GpuBackendType::GL:
            _impl = new GlNightVisionFrameViz();
            break;
#endif
#ifdef SUPPORT_VULKAN
        case GpuBackendType::VULKAN:
            _impl = new VulkanNightVisionFrameViz();
            break;
#endif
        default:
            SNN_RIP("Unexpected GpuBackendType");
    }
}

NightVisionFrameViz::NightVisionFrameViz(NightVisionFrameVizImpl *impl) : _impl(impl) {}

NightVisionFrameViz::~NightVisionFrameViz() { delete _impl; }

void NightVisionFrameViz::resize(uint32_t w, uint32_t h)
{
    _impl->resize(w, h);
}

void NightVisionFrameViz::render(const RenderParameters & p)
{
    return _impl->render(p);
}

void NightVisionFrameViz::reset() {
    return _impl->reset();
}

void NightVisionFrameRec::render(const NightVisionFrameViz::RenderParameters & rp) {
    return _impl->render(rp);
}

NightVisionFrameRec::NightVisionFrameRec()
    : NightVisionFrameViz(nullptr)
{
    switch (AppContext::getContext()->backendType) {
#ifdef SUPPORT_GL
        case GpuBackendType::GL:
            _impl = new GlNightVisionFrameRec();
            break;
#endif
#ifdef SUPPORT_VULKAN
        case GpuBackendType::VULKAN:
            _impl = new VulkanNightVisionFrameRec();
            break;
#endif
        default:
            SNN_RIP("Unexpected GpuBackendType");
    }
}

void NightVisionFrameRec::setWindow(intptr_t window) {
    _impl->setWindow(window);
}

}
