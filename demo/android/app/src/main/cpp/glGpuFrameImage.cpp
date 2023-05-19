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
#include "glGpuFrameImage.h"
#include <glad/glad.h>
#include "snn/glImageHandle.h"

namespace snn {

GlGpuFrameImage::GlGpuFrameImage(const Desc& d)
    : GpuFrameImage(GpuBackendType::GL)
{
    _desc = d;
    if (d.depth == 1) {
        _texture.allocate2D(d.format, d.width, d.height, _desc.channels, 1);
    } else {
        _texture.allocate2DArray(_desc.format, _desc.width, _desc.height, _desc.depth, _desc.channels, 1);
    }
}

void GlGpuFrameImage::updateTextureContent(ColorFormat format, void* data, uint32_t size) {
    if (size == 0) {
        size = _desc.width * _desc.height * _desc.channels;
    }
    if (format != _desc.format) {
        SNN_LOGE("incompatible format.");
        return;
    }
    if (_desc.depth > 4) {
        auto formatDesc = getNativeColorGL(_desc.format);
        std::size_t layerCount = 0;
        std::size_t offset     = _desc.width * _desc.height * 4;
        for (std::size_t i = 0; i < _desc.depth; i += 4) {
            layerCount = (std::size_t) i / 4;
            if (formatDesc.glType == GL_FLOAT) {
                std::vector<float> tempData((float*) ((float*) data + 4 * layerCount * offset),
                                            (float*) ((float*) data + 4 * (layerCount + 1) * offset));
                _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
            } else {
                std::vector<uint8_t> tempData(((uint8_t*) data + 4 * layerCount * offset), ((uint8_t*) data + 4 * (layerCount + 1) * offset));
                _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
            }
        }
        if (4 * layerCount < _desc.depth) {
            if (formatDesc.glType == GL_FLOAT) {
                std::vector<float> tempData((float*) ((float*) data + 4 * layerCount * offset), (float*) ((float*) data + size));
                _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
            } else {
                std::vector<uint8_t> tempData(((uint8_t*) data + 4 * layerCount * offset), ((uint8_t*) data + size));
                _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
            }
        }
    } else {
        _texture.setPixels(0, 0, 0, _desc.width, _desc.height, 0, data);
    }
    SNN_LOGD("%s:%d, updated texture id = %d", __FILENAME__, __LINE__, _texture.id());
}

// This method is not in use
#if 0
void GlGpuFrameImage::attach(GLenum target, GLuint texture) {
    _texture.attach(target, texture);
    auto& texDesc  = _texture.getDesc();
    _desc.channels = texDesc.channels;
    _desc.depth    = texDesc.depth;
    _desc.width    = texDesc.width;
    _desc.height   = texDesc.height;
    _desc.format   = texDesc.format;
}
#endif

void GlGpuFrameImage::getGpuImageHandle(GpuImageHandle& handle) const {
    GlImageHandle& glHandle = GlImageHandle::cast(handle);
    glHandle.target = _texture.target();
    glHandle.textureId = _texture.id();
}

}
