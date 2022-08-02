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
#include "utils.h"
#include "image.h"
#include <memory>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <variant>
#include <algorithm>
#include <sstream>
#include <KHR/khrplatform.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>

#include "color.h"
#include <iostream>
using namespace std;
#include <inttypes.h>

#include "glUtils.h"

static constexpr const char* RESIZE_NEAREST_CS_ASSET_NAME  = "shaders/3rdparty/shadertemplate_cs_upsampling2d_nearest.glsl";
static constexpr const char* RESIZE_BILINEAR_CS_ASSET_NAME = "shaders/3rdparty/shadertemplate_cs_upsampling2d_bilinear.glsl";

namespace snn {
typedef enum class Backend { Backend_CPU, Backend_GPU, Backend_DSP, NOT_DEFINED = 200 } Backend;

typedef enum class Transition { Backend_CPU_GPU, Backend_GPU_CPU, NOT_DEFINED = 200 } Transition;

class ImageTexture {
public:
    ImageTexture() {
        _backend = Backend::Backend_CPU;
        _textures.allocate(1);
        _name = "";
        _dims.clear();
        _dims.insert(_dims.end(), {0, 0, 0, 0});
        _format   = ColorFormat::NONE;
        outputMat = std::vector<std::vector<float>>();
    }
    ImageTexture(std::vector<uint32_t>& dims, ColorFormat format, void* buffer = NULL, std::string name = "") {
        (void) dims;
        (void) format;
        (void) buffer;
        _dims = dims;
        // Create empty ImageTexture
        _backend = Backend::Backend_CPU;
        _name    = name;
        std::vector<snn::ImagePlaneDesc> planes(dims[3]);
        for (auto& p : planes) {
            p.format = format;
            p.width  = dims[0];
            p.height = dims[1];
            p.depth  = dims[2];
            p.step   = 0;
            p.pitch  = 0;
            p.slice  = 0;
            p.offset = 0;
        }
        switch (format) {
        case ColorFormat::RGBA8:
            _images = ManagedImage<Rgba8>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGB8:
            _images = ManagedImage<Rgb8>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGBA32F:
            _images = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::R32F:
            _images = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGBA16F:
            _images = ManagedImage<Rgba16f>(ImageDesc(std::move(planes)));
            break;
        default:
            break;
        }
        if (buffer) {
            memcpy(_images.data(), buffer, _images.size());
        }
        _format   = format;
        outputMat = std::vector<std::vector<float>>();
    }

    ImageTexture(std::string& fileName) {
        (void) fileName;
        _name = fileName;
        loadFromFile(fileName);
        outputMat = std::vector<std::vector<float>>();
    }

    ImageTexture(gl::TextureObject& tex, std::string name = "") {
        (void) tex;
        _backend = Backend::Backend_GPU;
        // _textures.clear();
        // _textures.push_back(std::move(tex));
        _textures.allocate(1);
        _textures[0] = std::move(tex);

        _name     = name;
        auto desc = tex.getDesc();

        _dims.clear();
        _dims.insert(_dims.end(), {desc.width, desc.height, desc.depth, 1});
        _format   = desc.format;
        outputMat = std::vector<std::vector<float>>();
    }

    ImageTexture(std::vector<gl::TextureObject>& textures, std::string name = "") {
        (void) textures;
        _backend = Backend::Backend_GPU;
        SNN_ASSERT(textures.size() > 0);
        // _textures.clear();
        auto desc = textures[0].getDesc();
        _textures.allocate(textures.size());
        for (unsigned int i = 0; i < textures.size(); i++) {
            // this->_textures.push_back(std::move(textures[i]));
            this->_textures[i] = std::move(textures[i]);
        }

        _name = name;

        _dims.clear();
        _dims.insert(_dims.end(), {desc.width, desc.height, desc.depth, (uint32_t) _textures.size()});
        _format   = desc.format;
        outputMat = std::vector<std::vector<float>>();
    }

    ~ImageTexture() {
        // To free memory here
    }

    void resetImage(std::vector<uint32_t>& dims, ColorFormat format, void* buffer = NULL, std::string name = "") {
        (void) dims;
        (void) format;
        (void) buffer;
        _dims = dims;
        // Create empty ImageTexture
        _backend = Backend::Backend_CPU;
        _name    = name;
        std::vector<snn::ImagePlaneDesc> planes(dims[2]);
        for (auto& p : planes) {
            p.format   = format;
            p.width    = dims[0];
            p.height   = dims[1];
            p.depth    = dims[2];
            p.channels = dims[3];
            p.step     = 0;
            p.pitch    = 0;
            p.slice    = 0;
            p.offset   = 0;
        }
        switch (format) {
        case ColorFormat::RGBA8:
            _images = ManagedImage<Rgba8>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGB8:
            _images = ManagedImage<Rgb8>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGBA32F:
            _images = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGBA16F:
            _images = ManagedImage<Rgba16f>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::R32F:
            _images = ManagedImage<R32f>(ImageDesc(std::move(planes)));
            break;
        default:
            break;
        }
        if (buffer) {
            memcpy(_images.data(), buffer, _images.size());
        }
        _format = format;
    }

    bool resetTexture(std::vector<uint32_t>& dims, ColorFormat format, std::string name = "") {
        _backend = Backend::Backend_GPU;
        // _textures.clear();

        uint32_t width  = dims[0];
        uint32_t height = dims[1];
        uint32_t depth  = dims[2];
        this->_textures.allocate(dims[3]);

        for (unsigned int i = 0; i < dims[3]; i++) {
            /// Allocate output texture for the stage
            if (depth > 1) {
                this->_textures[i].allocate2DArray(format, width, height, depth);
                // const auto& fmtDesc = getColorFormatDesc(format);
                // std::cout << "Allocated output Textures for Layer " << i << " with Depth " << depth << ": " << name << std::endl;
                // std::cout << "Output Texture which was allocated: " << this->_textures[i].id() << ", " << this->_textures[i].target() << std::endl;
            } else {
                this->_textures[i].allocate2D(format, width, height);
            }
            // setup output texture parameter
            this->_textures[i].bind(0);
            GLfloat black[4] = {};
            glTexParameterfv(this->_textures[i].getDesc().target, GL_TEXTURE_BORDER_COLOR, black);
        }

        _name = name;

        _dims   = dims;
        _format = format;
        return 0;
    }

    bool resetTexture(gl::TextureObject& tex, std::string name = "") {
        (void) tex;
        SNN_LOGD("%s:%d, reset: %d, %d\n", __FUNCTION__, __LINE__, tex.id(), tex.target());
        _backend = Backend::Backend_GPU;
        // _textures.clear();
        // _textures.push_back(std::move(tex));
        _textures.allocate(1);
        _textures[0] = std::move(tex);

        _name     = name;
        auto desc = tex.getDesc();

        _dims.clear();
        _dims.insert(_dims.end(), {desc.width, desc.height, desc.depth, 1});
        _format = desc.format;
        return 0;
    }

    void loadFromFile(std::string fileName) {
        _images  = ManagedRawImage::loadFromFile(fileName);
        _backend = Backend::Backend_CPU;
        _name    = fileName;
        _dims.clear();
        _dims.insert(_dims.end(), {_images.width(), _images.height(), _images.depth(), _images.planes()});
        _format = _images.format();
    }

    void convertFormat(snn::ColorFormat format, float minVal = -1.0f, float maxVal = 1.0f) {
        (void) format;
        if (format == this->format()) {
            return;
        }
        switch (format) {
        case ColorFormat::RGBA8: {
            // auto tmp = snn::toRgba8(_images);
            _images = snn::toRgba8(_images);
            break;
        }
        case ColorFormat::RGB8: {
            // auto tmp = snn::toRgb8(_images);
            _images = snn::toRgb8(_images);
            break;
        }
        case ColorFormat::RGBA32F: {
            // auto tmp = snn::toRgba32f(_images);
            _images = snn::toRgba32f(_images, minVal, maxVal);
            // printOut();
            break;
        }
        case ColorFormat::RGBA16F: {
            // auto tmp = snn::toRgba32f(_images);
            _images = snn::toRgba16f(_images, minVal, maxVal);
            // printOut();
            break;
        }
        case ColorFormat::R32F: {
            _images = snn::toR32f(_images, minVal, maxVal);
            break;
        }
        default:
            break;
        }
        // printOut();
        _format = format;
    }

    void convertFormat(snn::ColorFormat format, std::vector<float>& means, std::vector<float>& norms) {
        (void) format;
        if (format == this->format()) {
            return;
        }
        switch (format) {
        case ColorFormat::RGBA8: {
            // auto tmp = snn::toRgba8(_images);
            _images = snn::toRgba8(_images);
            break;
        }
        case ColorFormat::RGB8: {
            // auto tmp = snn::toRgb8(_images);
            _images = snn::toRgb8(_images);
            break;
        }
        case ColorFormat::RGBA32F: {
            // auto tmp = snn::toRgba32f(_images);
            _images = snn::normalize(_images, means, norms);
            // printOut();
            break;
        }
        case ColorFormat::RGBA16F: {
            // auto tmp = snn::toRgba32f(_images);
            _images = snn::toRgba16f(_images);
            // printOut();
            break;
        }
        default:
            break;
        }
        // printOut();
        _format = format;
    }

    void readTexture(int buf_size, GLuint textureId, uint32_t w, uint32_t h, uint32_t d, uint32_t p) {
        char* outBuffer = (char*) malloc(buf_size * 16);
        glActiveTexture(GL_TEXTURE0);
        CHECK_GL_ERROR("glActiveTexture");
        if (d > 1) {
            glBindTexture(GL_TEXTURE_2D_ARRAY, textureId);
            CHECK_GL_ERROR("glBindTexture");
            glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_FLOAT, outBuffer);
            CHECK_GL_ERROR("glGetTexImage");
        } else {
            glBindTexture(GL_TEXTURE_2D, textureId);
            CHECK_GL_ERROR("glBindTexture");
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, outBuffer);
            CHECK_GL_ERROR("glGetTexImage");
        }

        float* dest = (float*) outBuffer;
        printf(">>>>>>>>>>readTexture>>>>>>>>>>>>>\n");
        for (int i = 0; i < buf_size * 4; i += 4) {
            printf("%f\n", *(dest + i));
        }
        printf("<<<<<<<<<<<<<<<<<<<<<<<\n");

        std::vector<uint32_t> dims {w, h, d, p};
        snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
        img.printOutWH();

        free(outBuffer);
    }

    void readTexture(int idx) {
        int buf_size     = _textures[idx].getDesc().width * _textures[idx].getDesc().height * _textures[idx].getDesc().depth;
        GLuint textureId = _textures[idx].getDesc().id;
        uint32_t w       = _textures[idx].getDesc().width;
        uint32_t h       = _textures[idx].getDesc().height;
        uint32_t d       = _textures[idx].getDesc().depth;
        uint32_t p       = 1;
        readTexture(buf_size, textureId, w, h, d, p);
    }

    std::string loadShader(const char* path) {
        auto bytes = snn::loadEmbeddedAsset(path);
        return std::string(bytes.begin(), bytes.end());
    }

    bool resizeTexture(gl::TextureObject& inputTex, gl::TextureObject& outputTex, float xScale, float yScale, std::vector<float>& means,
                       std::vector<float>& norms, int resizeType = 1) {
        string shaderHeader;
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

        // SNN_LOGI("Test:%s:%d, %s\n",__FUNCTION__,__LINE__, shaderHeader.c_str());
        string shaderUniforms = "#ifdef INPUT_TEXTURE_2D\n"
                                "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput;\n"
                                "#else\n"
                                "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput;\n"
                                "#endif\n"
                                "#ifdef OUTPUT_TEXTURE_2D\n"
                                "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D uOutput;\n"
                                "#else\n"
                                "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"
                                "#endif\n";

        string sourceCode, shaderMain;
        // 2. Compile the Shader
        // glsl_resizeNearest_glsl
        if (resizeType == 0) {
            shaderMain = loadShader(RESIZE_NEAREST_CS_ASSET_NAME);
        } else if (resizeType == 1) {
            // glsl_resizeBilinear_glsl
            shaderMain = loadShader(RESIZE_BILINEAR_CS_ASSET_NAME);
        }
        sourceCode = shaderHeader + shaderUniforms + shaderMain;
        // utils.createCompileCSStrShader(sourceCode, w, h, channels, scale);
        gl::SimpleGlslProgram csProgram;
        csProgram.loadCs(sourceCode.c_str());

        // 3. Bind input/output texture
        csProgram.use();

        glBindImageTexture(3, outputTex.getDesc().id, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindImageTexture(0, inputTex.getDesc().id, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        CHECK_GL_ERROR("glBindImageTexture");

        SNN_LOGD("Test:%s:%d, %d, %d, %d\n", __FUNCTION__, __LINE__, inputTex.getDesc().width, inputTex.getDesc().height, inputTex.getDesc().depth);
        glUniform4i(2, inputTex.getDesc().width, inputTex.getDesc().height, inputTex.getDesc().depth, 1);
        glUniform4i(3, outputTex.getDesc().width, outputTex.getDesc().height, outputTex.getDesc().depth, 1);
        glUniform2f(4, xScale, yScale);
        glUniform4f(5, means[0], means[1], means[2], means[3]);
        glUniform4f(6, norms[0], norms[1], norms[2], norms[3]);

        // SNN_LOGD("Test:%s:%d\n",__FUNCTION__,__LINE__);
        // 4. Run the shader
        glDispatchCompute(outputTex.getDesc().width / 8, outputTex.getDesc().height / 8, inputTex.getDesc().depth);
        glFinish();
        // readTexture(outputTex.getDesc().width*outputTex.getDesc().height*outputTex.getDesc().depth,
        //    outputTex.getDesc().id,
        //    outputTex.getDesc().width, outputTex.getDesc().height, outputTex.getDesc().depth, 1);

        // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
        return 0;
    }

    bool resize(float xScale, float yScale, std::vector<float>& means, std::vector<float>& norms) {
        // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
        if (_backend != Backend::Backend_GPU) {
            upload();
        }

        auto outputWidth  = (size_t)(_dims[0] / xScale);
        auto outputHeight = (size_t)(_dims[1] / yScale);

        for (uint32_t i = 0; i < this->planes(); i++) {
            SNN_LOGD("%s:%d: index:%d format: %d w: %d h: %d depth: %d\n", __FUNCTION__, __LINE__, i, (int) format(i), width(i), height(i), depth(i));
            if (depth(i) > 1) {
                // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
                _resizeTexture.allocate2DArray(ColorFormat::RGBA32F, outputWidth, outputHeight, depth(i), 1, depth(i) * 4);
            } else {
                // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
                _resizeTexture.allocate2D(ColorFormat::RGBA32F, outputWidth, outputHeight, 1, 4);
            }

            resizeTexture(_textures[i], _resizeTexture, xScale, yScale, means, norms);
            // SNN_LOGD("%s:%d, %d %d, %d\n", __FUNCTION__,__LINE__,i, _textures[i].id(), _textures[i].target());
            // _textures[i].cleanup();
            _textures[i].attach(_resizeTexture);
            // SNN_LOGD("%s:%d, %d %d, %d\n", __FUNCTION__,__LINE__,i, _textures[i].id(), _textures[i].target());
        }

        _format  = ColorFormat::RGBA32F;
        _dims[0] = outputWidth;
        _dims[1] = outputHeight;
        _backend = Backend::Backend_GPU;

        // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
        // readTexture(0);

        return 0;
    }

    ColorFormat format(size_t index = 0) const { return _images.format(index); }
    uint32_t width(size_t index = 0) const { return _images.width(index); }
    uint32_t height(size_t index = 0) const { return _images.height(index); }
    uint32_t depth(size_t index = 0) const { return _images.depth(index); }
    uint32_t step(size_t index = 0) const { return _images.step(index); }
    uint32_t pitch(size_t index = 0) const { return _images.pitch(index); }
    uint32_t sliceSize(size_t index = 0) const { return _images.sliceSize(index); }
    uint32_t size() { return (uint32_t) _images.size(); }
    uint32_t planes() { return (uint32_t) _images.planes(); }

    gl::TextureObject* texture(size_t index = 0) { return &_textures[index]; }

    ImageDesc getDesc() { return _images.getDesc(); }

    uint8_t* plane(size_t p) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.plane(p);
    }

    uint8_t* slice(size_t p, size_t z) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.slice(p, z);
    }

    uint8_t* row(size_t p, size_t y, size_t z = 0) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.row(p, y, z);
    }

    uint8_t* at(size_t p, size_t x, size_t y = 0, size_t z = 0) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.at(p, x, y, z);
    }

    std::string getName() { return _name; }

    // cv::Mat getOpenCVMat() {
    //     cv::Mat ret;
    //     return ret;
    // }
    bool loadCVMatData(uint8_t* cvdata) {
        (void) cvdata;
        SNN_LOGD("----format:%d planes:%d, width:%d, height:%d, depth:%d, textures: %zu----- \n", (int) _images.format(), _images.planes(), _images.width(),
                 _images.height(), _images.depth(), _textures.size());
        uint32_t chs         = getColorFormatDesc(_images.format()).ch;
        uint32_t planeSize   = (_images.width() * _images.height() * _images.depth()) * chs;
        uint32_t lineSize    = _images.width() * _images.depth() * chs;
        uint32_t channelSize = _images.depth() * chs;
        SNN_LOGD("----allSize: %d, planSize:%d, lineSize:%d, channelSize:%d, chs:%d----- \n", this->_images.size(), planeSize, lineSize, channelSize, chs);

        for (uint32_t p = 0; p < _images.planes(); p++) {
            for (uint32_t j = 0; j < _images.height(); j++) {
                for (uint32_t i = 0; i < _images.width(); i++) {
                    for (uint32_t k = 0; k < _images.depth(); k++) {
                        switch (_images.format()) {
                        case ColorFormat::RGBA8: {
                            uint8_t* tmpAddr = (uint8_t*) cvdata;
                            auto dest        = (uint8_t*) (_images.at(p, i, j, k));
                            uint8_t* src     = tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs;
                            memcpy(dest, src, 4);
                            auto v = *src;
                            std::cout << std::setw(4) << v << ",";
                            v = *(src + 1);
                            std::cout << std::setw(4) << v << ",";
                            v = *(src + 2);
                            std::cout << std::setw(4) << v << ",";
                            v = *(src + 3);
                            std::cout << std::setw(4) << v << ",";
                            break;
                        }
                        case ColorFormat::RGB8: {
                            uint8_t* tmpAddr = (uint8_t*) cvdata;
                            auto dest        = (uint8_t*) (_images.at(p, i, j, k));
                            uint8_t* src     = tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs;
                            memcpy(dest, src, 3);
                            auto v = *src;
                            std::cout << std::setw(4) << v << ",";
                            v = *(src + 1);
                            std::cout << std::setw(4) << v << ",";
                            v = *(src + 2);
                            std::cout << std::setw(4) << v << ",";
                            break;
                        }
                        case ColorFormat::RGBA32F: {
                            float* tmpAddr = (float*) cvdata;
                            auto dest      = (float*) (_images.at(p, i, j, k));
                            float* src     = tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs;
                            memcpy(dest, src, 4 * 4);
                            auto v = *src;
                            std::cout << std::setw(7) << v << ",";
                            v = *(src + 1);
                            std::cout << std::setw(7) << v << ",";
                            v = *(src + 2);
                            std::cout << std::setw(7) << v << ",";
                            v = *(src + 3);
                            std::cout << std::setw(7) << v << ",";
                            break;
                        }
                        case ColorFormat::R32F: {
                            float* tmpAddr = (float*) cvdata;
                            auto dest      = (float*) (_images.at(p, i, j, k));
                            float* src     = tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs;
                            *dest          = *src;
                            auto v         = *src;
                            std::cout << std::setw(7) << v << ",";
                            break;
                        }
                        default:
                            break;
                        }
                    }
                }
                std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
            }
        }
        _backend = Backend::Backend_CPU;
        return 0;
    }

    bool getCVMatData(uint8_t* ret) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        // Convert to NHWC format with float type
        // uint8_t* ret = (uint8_t *) malloc(this->_images.size() * sizeof(float));
        float* tmpAddr = (float*) ret;
        SNN_LOGD("----format:%d planes:%d, width:%d, height:%d, depth:%d, textures: %zu----- \n", (int) _images.format(), _images.planes(), _images.width(),
                 _images.height(), _images.depth(), _textures.size());
        uint32_t chs         = getColorFormatDesc(_images.format()).ch;
        uint32_t planeSize   = (_images.width() * _images.height() * _images.depth()) * chs;
        uint32_t lineSize    = _images.width() * _images.depth() * chs;
        uint32_t channelSize = _images.depth() * chs;
        SNN_LOGD("----allSize: %d, planSize:%d, lineSize:%d, channelSize:%d, chs:%d----- \n", this->_images.size(), planeSize, lineSize, channelSize, chs);

        for (uint32_t p = 0; p < _images.planes(); p++) {
            for (uint32_t j = 0; j < _images.height(); j++) {
                for (uint32_t i = 0; i < _images.width(); i++) {
                    for (uint32_t k = 0; k < _images.depth(); k++) {
                        switch (_images.format()) {
                        case ColorFormat::RGBA8: {
                            auto v                                                                    = (float) *(_images.at(p, i, j, k));
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 0) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = (float) *(_images.at(p, i, j, k) + 1);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 1) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = (float) *(_images.at(p, i, j, k) + 2);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 2) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = (float) *(_images.at(p, i, j, k) + 3);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 3) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            break;
                        }
                        case ColorFormat::RGB8: {
                            auto v                                                                    = (float) *(_images.at(p, i, j, k));
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 0) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = (float) *(_images.at(p, i, j, k) + 1);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 1) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = (float) *(_images.at(p, i, j, k) + 2);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 2) = v;
                            // std::cout << " " << (p * planeSize + j * lineSize + i * channelSize + k * chs + 2) <<  ",";
                            // std::cout  << std::setw(7) << v <<  ",";
                            break;
                        }
                        case ColorFormat::RGBA32F: {
                            auto v                                                                    = *((float*) _images.at(p, i, j, k));
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 0) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = *((float*) _images.at(p, i, j, k) + 1);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 1) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = *((float*) _images.at(p, i, j, k) + 2);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 2) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            v                                                                         = *((float*) _images.at(p, i, j, k) + 3);
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 3) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            break;
                        }
                        case ColorFormat::R32F: {
                            auto v                                                                    = *((float*) _images.at(p, i, j, k));
                            *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + 0) = v;
                            // std::cout  << std::setw(7) << v <<  ",";
                            break;
                        }
                        default:
                            break;
                        }
                    }
                }
                // std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
            }
        }
        return 0;
    }

    // From device to host
    void download() {
        SNN_LOGD("%s:%d: %d:%d:%d:%d\n", __FUNCTION__, __LINE__, _dims[0], _dims[1], _dims[2], _dims[3]);
        _backend = Backend::Backend_CPU;

        if (_dims[0] == 0 || _dims[1] == 0 || _dims[2] == 0 || _dims[3] == 0) {
            if (this->_textures.size() > 0) {
                _dims[0] = (GLuint) this->_textures[0].getDesc().width;
                _dims[1] = (GLuint) this->_textures[0].getDesc().height;
                _dims[2] = (GLuint) this->_textures[0].getDesc().depth;
                _dims[3] = (GLuint) this->_textures.size();
                _format  = this->_textures[0].getDesc().format;
                SNN_LOGD("ImageTexture from Texture directly %d, %d, %d, %d, format: %d\n", _dims[0], _dims[1], _dims[2], _dims[3], (int) _format);
            }
        }

        SNN_ASSERT(this->_textures.size() > 0);
        // SNN_LOGD("%s:%d, _images size: %d\n", __FUNCTION__,__LINE__, _images.size());
        std::vector<snn::ImagePlaneDesc> planes(_dims[3]);
        for (auto& p : planes) {
            p.format = _format;
            p.width  = _dims[0];
            p.height = _dims[1];
            p.depth  = _dims[2];
            p.step   = 0;
            p.pitch  = 0;
            p.slice  = 0;
            p.offset = 0;
        }
        switch (planes[0].format) {
        case ColorFormat::RGBA8:
            _images = ManagedImage<Rgba8>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGB8:
            _images = ManagedImage<Rgb8>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGBA32F:
            _images = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::R32F:
            _images = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
            break;
        case ColorFormat::RGBA16F:
            _images = ManagedImage<Rgba16f>(ImageDesc(std::move(planes)));
        default:
            break;
        }
        uint32_t offset = 0;
        for (uint32_t i = 0; i < this->_textures.size(); i++) {
            auto oneImage = this->_textures[i].getBaseLevelPixels();
            memcpy(_images.at(i, 0, 0, 0), oneImage.data(), oneImage.size());
            offset += oneImage.size();
        }
        SNN_LOGD("%s:%d: %d:%d:%d:%d\n", __FUNCTION__, __LINE__, _dims[0], _dims[1], _dims[2], _dims[3]);
    }

    // From host to device
    void upload() {
        SNN_LOGD("%s:%d\n", __FUNCTION__, __LINE__);
        _backend = Backend::Backend_GPU;
        // this->_textures.clear();
        if (this->_textures.size() <= 0) {
            this->_textures.allocate(planes());
        }
        for (uint32_t i = 0; i < this->planes(); i++) {
            SNN_LOGD("%s:%d: index:%d format: %d w: %d h: %d depth: %d\n", __FUNCTION__, __LINE__, i, (int) format(i), width(i), height(i), depth(i));
            if (!this->_textures[i].empty()) {
                SNN_LOGD("%s:%d: index:%d cleanup\n", __FUNCTION__, __LINE__, i);
                this->_textures[i].cleanup();
            }
            if (depth(i) > 1) {
                this->_textures[i].allocate2DArray(format(i), width(i), height(i), depth(i));
                for (uint32_t j = 0; j < depth(i); j++) {
                    this->_textures[i].setPixels(j, 0, 0, 0, width(i), height(i), 0, _images.at(i, 0, 0, j));
                }
            } else {
                this->_textures[i].allocate2D(format(i), width(i), height(i));
                this->_textures[i].setPixels(0, 0, 0, width(i), height(i), 0, _images.at(i, 0, 0, 0));
            }
        }
    }

    std::vector<uint32_t> getDims() { return _dims; }

    void saveToBIN(const std::string& filename) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        _images.saveToBIN(filename);
    }

    void printOut(std::ostream& stream = std::cout) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        SNN_LOGD("----format:%d planes:%d, width:%d, height:%d, depth:%d, textures: %zu----- \n", (int) _images.format(), _images.planes(), _images.width(),
                 _images.height(), _images.depth(), _textures.size());
        for (uint32_t p = 0; p < _images.planes(); p++) {
            for (uint32_t k = 0; k < _images.depth(); k++) {
                for (uint32_t j = 0; j < _images.height(); j++) {
                    for (uint32_t i = 0; i < _images.width(); i++) {
                        // stream << "M(" << p << ", " << k << ", " << j <<  ", " << i << "): ";
                        if (_images.format() == ColorFormat::RGBA8) {
                            auto v = (int) *(_images.at(p, i, j, k));
                            stream << std::setw(4) << v << ",";
                            v = (int) *(_images.at(p, i, j, k) + 1);
                            stream << std::setw(4) << v << ",";
                            v = (int) *(_images.at(p, i, j, k) + 2);
                            stream << std::setw(4) << v << ",";
                            v = (int) *(_images.at(p, i, j, k) + 3);
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA32F) {
                            auto v = *((float*) _images.at(p, i, j, k));
                            stream << std::setw(7) << v << ",";
                            v = *((float*) (_images.at(p, i, j, k) + 4));
                            stream << std::setw(7) << v << ",";
                            v = *((float*) (_images.at(p, i, j, k) + 8));
                            stream << std::setw(7) << v << ",";
                            v = *((float*) (_images.at(p, i, j, k) + 12));
                            stream << std::setw(7) << v << ",";
                        } else if (_images.format() == ColorFormat::RGB8) {
                            auto v = (int) *(_images.at(p, i, j, k));
                            stream << std::setw(4) << v << ",";
                            v = (int) *(_images.at(p, i, j, k) + 1);
                            stream << std::setw(4) << v << ",";
                            v = (int) *(_images.at(p, i, j, k) + 2);
                            stream << std::setw(4) << v << ",";
                        } else {
                            auto v = (int) *(_images.at(p, i, j, k));
                            stream << std::setw(4) << v << ",";
                        }
                    }
                    stream << "\n-----------" + std::to_string(j) + "------------" << std::endl;
                    stream << std::endl;
                }
                stream << "**************" + std::to_string(k) + "***********" << std::endl;
            }
            stream << "**************" + std::to_string(p) + "***********" << std::endl;
        }
    }

    void printOutWH(std::ostream& stream = std::cout) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        SNN_LOGD("----format:%d planes:%d, width:%d, height:%d, depth:%d, textures: %zu----- \n", (int) _images.format(), _images.planes(), _images.width(),
                 _images.height(), _images.depth(), _textures.size());
        for (uint32_t p = 0; p < _images.planes(); p++) {
            for (uint32_t k = 0; k < _images.depth(); k++) {
                for (uint32_t j = 0; j < _images.height(); j++) {
                    for (uint32_t i = 0; i < _images.width(); i++) {
                        // stream << "M(" << p << ", " << k << ", " << j <<  ", " << i << "): ";
                        if (_images.format() == ColorFormat::RGBA8) {
                            auto v = (int) *(_images.at(p, i, j, k));
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA32F) {
                            auto v = *((float*) _images.at(p, i, j, k));
                            stream << std::setw(7) << v << ",";
                        } else if (_images.format() == ColorFormat::RGB8) {
                            auto v = (int) *(_images.at(p, i, j, k));
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA16F) {
                            auto v = FP16::toFloat(*((uint16_t*) _images.at(p, i, j, k)));
                            stream << std::setw(7) << v << ",";
                        } else {
                            auto v = (int) *(_images.at(p, i, j, k));
                            stream << std::setw(4) << v << ",";
                        }
                    }
                    stream << "\n-------R----" + std::to_string(j) + "------------" << std::endl;
                    stream << std::endl;
                }
                for (uint32_t j = 0; j < _images.height(); j++) {
                    for (uint32_t i = 0; i < _images.width(); i++) {
                        // stream << "M(" << p << ", " << k << ", " << j <<  ", " << i << "): ";
                        if (_images.format() == ColorFormat::RGBA8) {
                            auto v = (int) *(_images.at(p, i, j, k) + 1);
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA32F) {
                            auto v = *((float*) (_images.at(p, i, j, k) + 4));
                            stream << std::setw(7) << v << ",";
                        } else if (_images.format() == ColorFormat::RGB8) {
                            auto v = (int) *(_images.at(p, i, j, k) + 1);
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA16F) {
                            auto v = FP16::toFloat(*((uint16_t*) _images.at(p, i, j, k) + 1));
                            stream << std::setw(7) << v << ",";
                        } else {
                        }
                    }
                    stream << "\n-------G----" + std::to_string(j) + "------------" << std::endl;
                    stream << std::endl;
                }
                for (uint32_t j = 0; j < _images.height(); j++) {
                    for (uint32_t i = 0; i < _images.width(); i++) {
                        // stream << "M(" << p << ", " << k << ", " << j <<  ", " << i << "): ";
                        if (_images.format() == ColorFormat::RGBA8) {
                            auto v = (int) *(_images.at(p, i, j, k) + 2);
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA32F) {
                            auto v = *((float*) (_images.at(p, i, j, k) + 8));
                            stream << std::setw(7) << v << ",";
                        } else if (_images.format() == ColorFormat::RGB8) {
                            auto v = (int) *(_images.at(p, i, j, k) + 2);
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA16F) {
                            auto v = FP16::toFloat(*((uint16_t*) _images.at(p, i, j, k) + 2));
                            stream << std::setw(7) << v << ",";
                        } else {
                        }
                    }
                    stream << "\n-------B----" + std::to_string(j) + "------------" << std::endl;
                    stream << std::endl;
                }
                for (uint32_t j = 0; j < _images.height(); j++) {
                    for (uint32_t i = 0; i < _images.width(); i++) {
                        // stream << "M(" << p << ", " << k << ", " << j <<  ", " << i << "): ";
                        if (_images.format() == ColorFormat::RGBA8) {
                            auto v = (int) *(_images.at(p, i, j, k) + 3);
                            stream << std::setw(4) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA32F) {
                            auto v = *((float*) (_images.at(p, i, j, k) + 12));
                            stream << std::setw(7) << v << ",";
                        } else if (_images.format() == ColorFormat::RGBA16F) {
                            auto v = FP16::toFloat(*((uint16_t*) _images.at(p, i, j, k) + 3));
                            stream << std::setw(7) << v << ",";
                        } else if (_images.format() == ColorFormat::RGB8) {
                        } else {
                        }
                    }
                    stream << "\n-----A------" + std::to_string(j) + "------------" << std::endl;
                    stream << std::endl;
                }
                stream << "**************" + std::to_string(k) + "***********" << std::endl;
            }
            stream << "**************" + std::to_string(p) + "***********" << std::endl;
        }
    }

    // To work with current CPU Flatten/Dense Layer. To be changed.
    std::vector<std::vector<float>> outputMat;

private:
    std::string _name;
    ColorFormat _format = ColorFormat::NONE;
    std::vector<uint32_t> _dims;
    snn::Backend _backend = Backend::Backend_CPU;
    snn::RawImage _images;
    FixedSizeArray<gl::TextureObject> _textures;
    gl::TextureObject _resizeTexture;
};

void readTexture(int buf_size, GLuint textureId);
vector<float> readTexture(int buf_size, GLuint textureId, uint32_t w, uint32_t h, uint32_t d, uint32_t p);

} // namespace snn
