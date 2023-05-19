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
#include "snn/imageTexture.h"
#include "snn/utils.h"
#include "snn/imageTextureFactory.h"

namespace snn {

ImageTexture::ImageTexture(GpuBackendType type, const std::array<uint32_t, 4>& dims, ColorFormat format, const void* buffer /*= NULL*/,
    const std::string& name /*= ""*/)
    : _type(type)
    , _name(name)
    , _format(format)
    , _dims(dims)
    , _backend(Backend::Backend_CPU)
{
    resetImages();
    if (buffer) {
        memcpy(_images.data(), buffer, _images.size());
    }
}

ImageTexture::ImageTexture(GpuBackendType type, const std::string& fileName)
    : _type(type)
    , _name(fileName)
{
    loadFromFile(fileName);
}

void ImageTexture::loadFromFile(const std::string& fileName, bool fromBin) {
    _images  = ManagedRawImage::loadFromFile(fileName, fromBin);
    _backend = Backend::Backend_CPU;
    _name    = fileName;
    _dims = {_images.width(), _images.height(), _images.depth(), _images.planes()};
    _format = _images.format();
}

void ImageTexture::convertFormat(snn::ColorFormat format) {
    if (format == this->format()) {
        return;
    }
    if (_backend != Backend::Backend_CPU) {
        download();
    }
    switch (format) {
    case ColorFormat::R8: {
        _images = snn::toR8(_images);
        break;
    }
    case ColorFormat::RGB8: {
        _images = snn::toRgb8(_images);
        break;
    }
    case ColorFormat::RGBA8: {
        _images = snn::toRgba8(_images);
        break;
    }
    case ColorFormat::R32F: {
        _images = snn::toR32f(_images);
        break;
    }
    case ColorFormat::RGBA32F: {
        _images = snn::toRgba32f(_images);
        break;
    }
    case ColorFormat::RGBA16F: {
        _images = snn::toRgba16f(_images);
        break;
    }
    default:
        SNN_RIP("Conversion not implemented !");
        break;
    }
    _format = format;
}

void ImageTexture::convertToRGBA32FAndNormalize(const std::vector<float>& means, const std::vector<float>& norms) {
    _images = snn::normalize(_images, means, norms);
    _format = ColorFormat::RGBA32F;
}


void ImageTexture::reset(const std::array<uint32_t, 4>& dims, ColorFormat format, void* buffer /*= NULL*/, const std::string& name /*= ""*/) {
    _backend = Backend::Backend_CPU;
    _name    = name;
    _dims = dims;
    _format = format;
    const ColorFormatDesc& fd = getColorFormatDesc(_format);
    std::vector<snn::ImagePlaneDesc> planes(dims[3]);
    for (auto& p : planes) {
        p.format   = format;
        p.width    = dims[0];
        p.height   = dims[1];
        p.depth    = dims[2];
        p.channels = fd.ch * p.depth;
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
}

void ImageTexture::resetImages() {
    std::vector<snn::ImagePlaneDesc> planes(_dims[3]);
    for (auto& p : planes) {
        p.format = _format;
        p.width  = _dims[0];
        p.height = _dims[1];
        p.depth  = _dims[2];
        p.channels = getColorFormatDesc(_format).ch * p.depth;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    switch (_format) {
    case ColorFormat::RGBA8:
        _images = ManagedImage<Rgba8>(ImageDesc(std::move(planes)));
        break;
    case ColorFormat::SRGB8_A8:
        _images = ManagedImage<SRgba8>(ImageDesc(std::move(planes)));
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
        SNN_RIP("Color format not implemented !");
        break;
    }
}

bool ImageTexture::getCVMatData(uint8_t* ret) {
    if (_backend != Backend::Backend_CPU) {
        download();
    }
    // Convert to NHWC format with float type
    // uint8_t* ret = (uint8_t *) malloc(this->_images.size() * sizeof(float));
    float* tmpAddr = (float*) ret;
    SNN_LOGD("----format:%d planes:%d, width:%d, height:%d, depth:%d, textures: %zu----- \n", (int) _images.format(), _images.planes(), _images.width(),
             _images.height(), _images.depth(), getNumTextures());
    uint32_t chs         = getColorFormatDesc(_images.format()).ch;
    uint32_t planeSize   = (_images.width() * _images.height() * _images.depth()) * chs;
    uint32_t lineSize    = _images.width() * _images.depth() * chs;
    uint32_t channelSize = _images.depth() * chs;
    SNN_LOGD("----allSize: %d, planSize:%d, lineSize:%d, channelSize:%d, chs:%d----- \n", this->_images.size(), planeSize, lineSize, channelSize, chs);

    for (uint32_t p = 0; p < _images.planes(); p++) {
        for (uint32_t j = 0; j < _images.height(); j++) {
            for (uint32_t i = 0; i < _images.width(); i++) {
                for (uint32_t k = 0; k < _images.depth(); k++) {
                    Rgba32f dst{};
                    if (!toRgba32f(dst, _images.at(p, i, j, k), _images.format(p))) {
                        return false;
                    }
                    for (uint32_t ch = 0; ch < chs; ++ch) {
                        *(tmpAddr + p * planeSize + j * lineSize + i * channelSize + k * chs + ch) = dst.f32[ch];
                    }
                }
            }
        }
    }
    return 0;
}

void ImageTexture::prettyPrint(FILE* fp) {
    SNN_LOGD("----format:%d planes:%d, width:%d, height:%d, depth:%d, textures: %zu----- \n", (int) _images.format(), _images.planes(), _images.width(),
             _images.height(), _images.depth(), getNumTextures());
    prettyPrintHWCBuf(at(0, 0, 0, 0), _dims[1], _dims[0], getColorFormatDesc(_format).ch * _dims[2], _format, fp);
}

std::shared_ptr<ImageTexture>* ImageTextureAllocator::allocate(size_t n) {
    std::shared_ptr<ImageTexture>* ptr = new std::shared_ptr<ImageTexture>[n];
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = ImageTextureFactory::createImageTexture(context);
    }
    return ptr;
}

void ImageTextureAllocator::deallocate(std::shared_ptr<ImageTexture>* ptr, size_t) {
    delete[] ptr;
}

}
