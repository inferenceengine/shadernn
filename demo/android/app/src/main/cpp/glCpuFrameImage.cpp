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
#include "glCpuFrameImage.h"
#include "demoutils.h"

namespace snn {

PackPbo::PackPbo(const FrameImage2::Desc& d): desc(d) {
    auto bufferSize = getColorFormatDesc(d.format).calcImageSizeInBytes(d.width, d.height);
    b.allocate<uint8_t>(bufferSize, nullptr, GL_STREAM_READ);
}

void PackPbo::readPixels() {
    b.bind();
    PROFILE_TIME(PPBO, "PPBO - read pixels")
    const auto& fd = getNativeColorGL(desc.format);
    GLCHK(glReadPixels(0, 0, desc.width, desc.height, fd.glFormat, fd.glType, (void*) 0));
    b.unbind();
    hasData = true;
}

void PackPbo::download(RawImage& dst) {
    if (!hasData) {
        return;
    }
    PROFILE_TIME(PPBO, "PPBO - map buffer")
    hasData         = false;
    const void* src = b.map();
    memcpy(dst.data(), src, std::min<size_t>(dst.size(), b.length));
    b.unmap();
}

}
