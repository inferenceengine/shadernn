
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

#include "gpuFrameImage.h"
#include "glUtils.h"

namespace snn {

class GlGpuFrameImage : public GpuFrameImage {
public:
    GlGpuFrameImage(const Desc& d);

    void updateTextureContent(ColorFormat format, void* data, uint32_t size = 0) override;

    void getGpuImageHandle(GpuImageHandle& handle) const override;

private:
    gl::TextureObject _texture;
};

}