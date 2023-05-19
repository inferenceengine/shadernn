
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
#include "glContext.h"
#include "glUtils.h"
#include "snn/utils.h"

namespace snn {

GlGpuContext::GlGpuContext(bool onlyInitGlExtensions)
    : GpuContext(GpuBackendType::GL)
{
    if (onlyInitGlExtensions) {
        gl::initGLExtensions();
    } else {
        rc = new gl::RenderContext(gl::RenderContext::STANDALONE);
    }
    SNN_LOGI("OpenGL context created");
}

GlGpuContext::~GlGpuContext() {
    delete rc;
}

GlGpuContext* GlGpuContext::cast(GpuContext* context) {
    if (context->backendType != GpuBackendType::GL) {
        SNN_RIP("Gpu context is not an OpenGL type!");
    }
    return static_cast<GlGpuContext*>(context);
}

GlGpuContext* createGlContext(bool onlyInitGlExtensions) {
    return new GlGpuContext(onlyInitGlExtensions);
}

}
