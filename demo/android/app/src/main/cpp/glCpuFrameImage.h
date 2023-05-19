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

#include "cpuFrameImage.h"
#include "demoutils.h"
#include "glUtils.h"
#include "processor.h"

namespace snn {

struct PackPbo {
    FrameImage2::Desc desc;
    gl::BufferObject<GL_PIXEL_PACK_BUFFER> b;
    bool hasData = 0;

    PackPbo(const FrameImage2::Desc& d);

    void readPixels();

    void download(RawImage& dst);
};

class GlCpuFrameImage : public CpuFrameImage {
public:
    mutable PackPbo ppbo;

    GlCpuFrameImage(const Desc& d)
        : CpuFrameImage(d)
        , ppbo(d)
    {}

    virtual void getCpuImage(RawImage& image) const override {
        ppbo.download(_image);
        CpuFrameImage::getCpuImage(image);
    }

    static GlCpuFrameImage* cast(FrameImage2* frame) {
        return (frame->desc().device == Device::CPU) ? static_cast<GlCpuFrameImage*>(frame) : nullptr;
    }
};

}
