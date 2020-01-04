/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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

#include <snn/glUtils.h>
#include "image.h"

#include <vector>
#include <memory>

enum class Filter { Nearest, Linear };

class Texture : public gl::TextureObject {
public:
    static std::shared_ptr<Texture> createAttached(GLenum target, GLuint textureId) {
        auto t = std::make_shared<Texture>();
        t->attach(target, textureId);
        return t;
    }

    static std::shared_ptr<Texture> create2D(snn::ColorFormat f, uint32_t w, uint32_t h, uint32_t channels = 0) {
        auto t = std::make_shared<Texture>();
        if (channels == 0) {
            channels = 4;
        }
        t->allocate2D(f, w, h, channels);
        return t;
    }

    static std::shared_ptr<Texture> createArray(snn::ColorFormat f, uint32_t w, uint32_t h, uint32_t l, uint32_t channels = 0) {
        auto t = std::make_shared<Texture>();
        if (channels == 0) {
            channels = 4 * l;
        }
        t->allocate2DArray(f, w, h, l, channels);
        return t;
    }

    GLuint GetId() const { return getDesc().id; }

    void SetFilter(Filter filter) {
        bind(0);
        if (filter == Filter::Nearest) {
            glTexParameteri(getDesc().target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(getDesc().target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        } else {
            glTexParameteri(getDesc().target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(getDesc().target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
    }

    void Bind(size_t stage) { return bind(stage); }

    void Unbind() { glBindTexture(getDesc().target, 0); }

    uint32_t GetWidth() const { return getDesc().width; }

    uint32_t GetHeight() const { return getDesc().height; }

    void ApplyParameters() const { /*do nothing*/
    }
};
