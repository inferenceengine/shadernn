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
#include "framebuffer.h"

const snn::FrameBuffer2* snn::FrameBuffer2::_current = nullptr;

void snn::FrameBuffer2::attachTexture(const gl::TextureObject& texture, size_t firstLayer, size_t layerCount) {
    SNN_ASSERT(_current == this);

    // attach to the new texture
    switch (texture.getDesc().target) {
    case GL_TEXTURE_1D:
    case GL_TEXTURE_2D:
        SNN_ASSERT(0 == firstLayer && 1 == layerCount);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        break;

    case GL_TEXTURE_2D_ARRAY:
        for (size_t i = 0; i < layerCount; ++i) {
            SNN_LOGD("Attaching layer : %d", firstLayer + i);
            glFramebufferTextureLayer(GL_FRAMEBUFFER, (GLenum)(GL_COLOR_ATTACHMENT0 + i), texture, 0, (GLint)(firstLayer + i));
        }
        break;

    default:
        // 3D or cube texture
        SNN_LOGE("not implemented.");
        unbind();
        return;
    }

    // setup draw buffers
    static constexpr const GLenum c_DrawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
                                                     GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
    glDrawBuffers((GLsizei) layerCount, c_DrawBuffers);

    SNN_ASSERT(isComplete());
}
