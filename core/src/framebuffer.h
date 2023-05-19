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

#include "snn/snn.h"
#include "snn/utils.h"
#include "glUtils.h"

namespace snn {


// This is a front-end class for glUtils
class FrameBuffer2 {
public:
    static bool isComplete() { return GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER); }

    static void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

    SNN_NO_COPY(FrameBuffer2);
    SNN_NO_MOVE(FrameBuffer2);

    FrameBuffer2() { glGenFramebuffers(1, &_id); }

    virtual ~FrameBuffer2() { glDeleteFramebuffers(1, &_id); }

    GLuint id() const { return _id; }

    // attach to the new texture. assumes this framebuffer is bound.
    void attachTexture(const gl::TextureObject& texture, size_t firstLayer = 0, size_t layerCount = 1);

    // detach from any texture. assumes this framebuffer is bound.
    void detachTexture() {
        SNN_ASSERT(_current == this);
        GLCHKDBG(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
        GLenum c0 = GL_COLOR_ATTACHMENT0;
        GLCHKDBG(glDrawBuffers(1, &c0));
    }

    void bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, _id);
        _current = this;
    }

private:
    static const FrameBuffer2* _current; // pointing to current bound frame buffer object.
    GLuint _id = 0;
};

} // namespace snn
