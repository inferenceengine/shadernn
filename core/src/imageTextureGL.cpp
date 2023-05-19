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
#include "imageTextureGL.h"
#include <glad/glad.h>
#include "snn/glImageHandle.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <iostream>
#include <inttypes.h>

namespace snn {

ImageTextureGL::ImageTextureGL()
    : ImageTexture(GpuBackendType::GL)
{
    _textures.allocate(1);
}

ImageTextureGL::ImageTextureGL(const std::array<uint32_t, 4>& dims, ColorFormat format, const void* buffer, const std::string& name)
    : ImageTexture(GpuBackendType::GL, dims, format, buffer, name)
{}

ImageTextureGL::ImageTextureGL(const std::string& fileName)
    : ImageTexture(GpuBackendType::GL, fileName)
{}

gl::TextureObject* ImageTextureGL::texture(size_t index) {
    _backend = Backend::Backend_GPU;
    return &_textures[index];
}

std::string ImageTextureGL::getTextureInfo() const {
    if (_textures.empty()) {
        return "<empty>";
    }
    return std::to_string(_textures[0].id()) + ", " + std::to_string(_textures[0].target());
}

std::string ImageTextureGL::getTextureInfo2() const {
    if (_textures.empty()) {
        return "<empty>";
    }
    char buf[256];
    snprintf(buf, sizeof(buf), "id: %d, target: %d, w: %d, h: %d, d: %d, c: %d, format: %s",
        _textures[0].id(), _textures[0].target(),
        _textures[0].getDesc().width, _textures[0].getDesc().height, _textures[0].getDesc().depth, _textures[0].getDesc().channels,
        getColorFormatDesc(_textures[0].getDesc().format).name
    );
    return std::string(buf);
}

void ImageTextureGL::attach(ImageTexture* src) {
    _backend = Backend::Backend_GPU;
    SNN_ASSERT(src->getType() == GpuBackendType::GL);
    ImageTextureGL* srcGL = static_cast<ImageTextureGL*>(src);
    SNN_ASSERT(_textures.size() > 0);
    SNN_ASSERT(_textures.size() <= srcGL->_textures.size());
    for (size_t i = 0; i < _textures.size(); i++) {
        gl::TextureObject* tex = srcGL->texture(i);
        SNN_LOGD("input tex:%d, %d, %d", _textures[i].id(), _textures[i].target(), (int)_textures[i].getDesc().format);
        if (tex->target() != 0 && tex->id() != 0) {
            texture(i)->attach(*tex);
        }
        SNN_LOGD("input tex:%d, %d, %d", texture(i)->id(), texture(i)->target(), (int)texture(i)->getDesc().format);
    }
}

void ImageTextureGL::attach(const std::vector<const GpuImageHandle*>& images) {
    _backend = Backend::Backend_GPU;
    SNN_ASSERT(images.size() == _dims[3]);

    _textures.allocate(images.size());
    for (unsigned int i = 0; i < images.size(); i++) {
        SNN_ASSERT(images[i]);
        const GlImageHandle& glImageHandle = GlImageHandle::cast(*images[i]);
        _textures[i].attach(glImageHandle.target, glImageHandle.textureId);

        SNN_ASSERT(width(i) == _textures[i].getDesc().width);
        SNN_ASSERT(height(i) == _textures[i].getDesc().height);
        SNN_ASSERT(format(i) == _textures[i].getDesc().format);
        SNN_ASSERT(depth(i) == _textures[i].getDesc().depth);
    }
}

void ImageTextureGL::resetTexture() {
    _backend = Backend::Backend_GPU;

    uint32_t width  = _dims[0];
    uint32_t height = _dims[1];
    uint32_t depth  = _dims[2];
    uint32_t planes = _dims[3];
    _textures.allocate(planes);

    for (unsigned int i = 0; i < planes; i++) {
        if (depth > 1) {
            _textures[i].allocate2DArray(_format, width, height, depth);
        } else {
            _textures[i].allocate2D(_format, width, height);
        }
        _textures[i].bind(0);
        GLfloat black[4] = {};
        glTexParameterfv(_textures[i].getDesc().target, GL_TEXTURE_BORDER_COLOR, black);
        SNN_LOGD("id = %d, target = %d, format = %d", _textures[i].id(), _textures[i].target(), (int)_format);
    }
}

void ImageTextureGL::resetTexture(const std::array<uint32_t, 4>& dims, ColorFormat format, const std::string& name) {
    _name = name;
    _dims   = dims;
    _format = format;

    resetTexture();
}

bool ImageTextureGL::resizeTexture(gl::TextureObject& inputTex, gl::TextureObject& outputTex, float xScale, float yScale, const std::array<float, 4>& means,
    const std::array<float, 4>& norms, bool linearFilter) {

    std::string shaderHeader;
    shaderHeader = "#version 320 es \n"
                   "#define PRECISION highp\n"
                   "precision PRECISION float;\n"
                   "layout(std140) uniform;\n"
                   "#define OUTPUT_FORMAT rgba32f\n";

    if (inputTex.getDesc().depth <= 1) {
        shaderHeader += "#define INPUT_TEXTURE_2D\n";
        shaderHeader += "#define OUTPUT_TEXTURE_2D\n";
    }
    shaderHeader += ("#define WORK_X 8 \n");
    shaderHeader += ("#define WORK_Y 8 \n");
    shaderHeader += ("#define WORK_Z 1 \n");

    std::string shaderUniforms = "#ifdef INPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput;\n"
                            "#endif\n"
                            "#ifdef OUTPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D uOutput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"
                            "#endif\n";

    std::string sourceCode, shaderMain;
    // 2. Compile the Shader
    if (!linearFilter) {
        shaderMain = loadShader(RESIZE_NEAREST_CS_ASSET_NAME);
    } else {
        shaderMain = loadShader(RESIZE_BILINEAR_CS_ASSET_NAME);
    }
    sourceCode = shaderHeader + shaderUniforms + shaderMain;
    gl::SimpleGlslProgram csProgram;
    csProgram.loadCs(sourceCode.c_str());

    // 3. Bind input/output texture
    csProgram.use();

    glBindImageTexture(3, outputTex.getDesc().id, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
    CHECK_GL_ERROR("glBindImageTexture");
    glBindImageTexture(0, inputTex.getDesc().id, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
    CHECK_GL_ERROR("glBindImageTexture");

    SNN_LOGD("Test:%s:%d, %d, %d, %d\n", __FILENAME__, __LINE__, inputTex.getDesc().width, inputTex.getDesc().height, inputTex.getDesc().depth);
    glUniform4i(2, inputTex.getDesc().width, inputTex.getDesc().height, inputTex.getDesc().depth, 1);
    glUniform4i(3, outputTex.getDesc().width, outputTex.getDesc().height, outputTex.getDesc().depth, 1);
    glUniform2f(4, xScale, yScale);
    glUniform4f(5, means[0], means[1], means[2], means[3]);
    glUniform4f(6, norms[0], norms[1], norms[2], norms[3]);

    // 4. Run the shader
    glDispatchCompute(std::max(1U, outputTex.getDesc().width / 8), std::max(1U, outputTex.getDesc().height / 8), inputTex.getDesc().depth);
    glFinish();
    return 0;
}

bool ImageTextureGL::resize(float xScale, float yScale, const std::array<float, 4>& means, const std::array<float, 4>& norms,
    bool linearFilter, ColorFormat /*cf*/) {
    if (_backend != Backend::Backend_GPU) {
        upload();
    }

    // Changing format is not implemented in ImageTextureGL

    auto outputWidth  = (size_t)(_dims[0] / xScale);
    auto outputHeight = (size_t)(_dims[1] / yScale);

    for (uint32_t i = 0; i < planes(); i++) {
        SNN_LOGD("%s:%d: index:%d format: %d w: %d h: %d depth: %d\n", __FILENAME__, __LINE__, i, (int) format(i), width(i), height(i), depth(i));
        gl::TextureObject resizeTex;
        if (depth(i) > 1) {
            resizeTex.allocate2DArray(ColorFormat::RGBA32F, outputWidth, outputHeight, depth(i), depth(i) * 4, 1);
        } else {
            resizeTex.allocate2D(ColorFormat::RGBA32F, outputWidth, outputHeight, 4, 1);
        }

        resizeTexture(_textures[i], resizeTex, xScale, yScale, means, norms, linearFilter);
        _textures[i] = std::move(resizeTex);
    }

    _format  = ColorFormat::RGBA32F;
    _dims[0] = outputWidth;
    _dims[1] = outputHeight;
    _backend = Backend::Backend_GPU;

    return 0;
}

// From device to host
void ImageTextureGL::download() {
    _backend = Backend::Backend_CPU;

    if (_dims[0] == 0 || _dims[1] == 0 || _dims[2] == 0 || _dims[3] == 0) {
        if (_textures.size() > 0) {
            _dims[0] = (GLuint) _textures[0].getDesc().width;
            _dims[1] = (GLuint) _textures[0].getDesc().height;
            _dims[2] = (GLuint) _textures[0].getDesc().depth;
            _dims[3] = (GLuint) _textures.size();
            _format  = _textures[0].getDesc().format;
            SNN_LOGD("ImageTexture from Texture directly %d, %d, %d, %d, format: %d\n", _dims[0], _dims[1], _dims[2], _dims[3], (int) _format);
        }
    }

    SNN_ASSERT(_textures.size() > 0);
    resetImages();
    for (uint32_t i = 0; i < _textures.size(); i++) {
        auto oneImage = _textures[i].getBaseLevelPixels();
        SNN_ASSERT(_images.size() == oneImage.size());
        _images = std::move(oneImage);
    }
    SNN_LOGD("%d:%d:%d:%d", _dims[0], _dims[1], _dims[2], _dims[3]);
}

// From host to device
void ImageTextureGL::upload() {
    _backend = Backend::Backend_GPU;
    _textures.allocate(planes());
    for (uint32_t i = 0; i < planes(); i++) {
        SNN_LOGD("index:%d format: %d w: %d h: %d depth: %d", i, (int) format(i), width(i), height(i), depth(i));
        if (!_textures[i].empty()) {
            SNN_LOGD("index:%d cleanup", i);
            _textures[i].cleanup();
        }
        if (depth(i) > 1) {
            _textures[i].allocate2DArray(format(i), width(i), height(i), depth(i));
            for (uint32_t j = 0; j < depth(i); j++) {
                _textures[i].setPixels(j, 0, 0, 0, width(i), height(i), 0, _images.at(i, 0, 0, j));
            }
        } else {
            _textures[i].allocate2D(format(i), width(i), height(i));
            _textures[i].setPixels(0, 0, 0, width(i), height(i), 0, _images.at(i, 0, 0, 0));
        }
        SNN_LOGD("idx:%d  tex: %d, %d, %d", i, _textures[i].id(), _textures[i].target(), (int)_textures[i].getDesc().format);
    }
}

}
