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
#include <glad/glad.h>
#include "snn/glImageAccessor.h"
#include "imageTextureGL.h"
#include "colorGL.h"

namespace snn {

bool getGlImage(ImageTexture& texture, GlImageHandle& imageHandle) {
    if (texture.getType() != GpuBackendType::GL) {
        SNN_RIP("Image texture does not have OpenGL backend!");
    }
    ImageTextureGL& textureGL = static_cast<ImageTextureGL&>(texture);
    if (!textureGL.isValid()) {
        return false;
    }
    gl::TextureObject* glTex = textureGL.texture();

    imageHandle.textureId = glTex->id();
    imageHandle.target = glTex->target();
    return true;
}

void setGlImage(ImageTexture& texture, const GlImageHandle& imageHandle) {
    if (texture.getType() != GpuBackendType::GL) {
        SNN_RIP("Image texture does not have OpenGL backend!");
    }
    ImageTextureGL& textureGL = static_cast<ImageTextureGL&>(texture);
    textureGL.attach({&imageHandle});
}

}
