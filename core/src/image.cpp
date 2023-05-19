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
#include "snn/image.h"
#include "snn/colorUtils.h"
#include <libyuv.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace snn;

static std::vector<ImagePlaneDesc> convertPlanes(const RawImage& src, ColorFormat format) {
    std::vector<ImagePlaneDesc> planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = format;
        p.channels = p.depth * getColorFormatDesc(format).ch;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    return planes;
}

void ImageDesc::reset(std::vector<ImagePlaneDesc>&& planes_, uint32_t alignment_ /*= 0*/) {
    planes    = std::move(planes_);
    size      = 0;
    alignment = (0 == alignment_) ? 4 : alignment_;
    for (auto& p : this->planes) {
        SNN_ASSERT(p.width > 0);
        SNN_ASSERT(p.height > 0);
        SNN_ASSERT(p.depth > 0);
        const ColorFormatDesc& fd = getColorFormatDesc(p.format);
        SNN_ASSERT(p.channels == fd.ch * p.depth);
        p.step   = std::max(p.step, (uint32_t) fd.bits);
        p.pitch  = std::max(p.width * p.step / 8u, p.pitch);
        p.pitch  = (p.pitch + alignment - 1) / alignment * alignment; // make sure pitch meets alignment requirement.
        p.slice  = std::max(p.pitch * p.height, p.slice);

        // calculate plane offset
        if (0 == p.offset) {
            p.offset = this->size;
        }

        // update image size
        this->size = std::max(p.offset + p.size(), this->size);

        // double check image plane's alignment.
        SNN_ASSERT(0 == (p.pitch % alignment));
        SNN_ASSERT(0 == (p.slice % alignment));
        SNN_ASSERT(0 == (p.offset % alignment));
        SNN_ASSERT(0 == (p.size() % alignment));
        SNN_ASSERT(0 == (this->size % alignment));
    }
}

void ImageDesc::reset(uint32_t alignment_ /*= 0*/) {
    alignment = (0 == alignment_) ? 4 : alignment_;
    for (auto& p : this->planes) {
        SNN_ASSERT(p.width > 0);
        SNN_ASSERT(p.height > 0);
        SNN_ASSERT(p.depth > 0);
        const ColorFormatDesc& fd = getColorFormatDesc(p.format);
        SNN_ASSERT(p.channels == fd.ch * p.depth);
        p.step   = std::max(p.step, (uint32_t) fd.bits);
        p.pitch  = std::max(p.width * p.step / 8u, p.pitch);
        p.pitch  = (p.pitch + alignment - 1) / alignment * alignment; // make sure pitch meets alignment requirement.
        p.slice  = std::max(p.pitch * p.height, p.slice);

        // calculate plane offset
        if (0 == p.offset) {
            p.offset = this->size;
        }

        // update image size
        this->size = std::max(p.offset + p.size(), this->size);

        // double check image plane's alignment.
        SNN_ASSERT(0 == (p.pitch % alignment));
        SNN_ASSERT(0 == (p.slice % alignment));
        SNN_ASSERT(0 == (p.offset % alignment));
        SNN_ASSERT(0 == (p.size() % alignment));
        SNN_ASSERT(0 == (this->size % alignment));
    }
}

void RawImage::reset(ImageDesc&& desc, void* pixels) {
    _desc = std::move(desc);
    if (_desc.empty()) {
        SNN_LOGI("Seems like description is empty");
    }
#ifdef _DEBUG
    for (auto& p : desc.planes) {
        SNN_ASSERT(p.width > 0);
        SNN_ASSERT(p.height > 0);
        SNN_ASSERT(p.depth > 0);
        SNN_ASSERT(p.channels == getColorFormatDesc(p.format).ch * p.depth);
    }
#endif
    _pixels = _desc.empty() ? nullptr : (uint8_t*) pixels;
    if (_pixels && 0 != ((intptr_t) _pixels % _desc.alignment)) {
        SNN_RIP("the pixel buffer pointer does not meet alignment requirement.");
    }
}

void RawImage::vertFlipInpace() {
    std::vector<uint8_t> v;
    for (size_t p = 0; p < _desc.planes.size(); ++p) {
        auto w = width(p);
        auto h = height(p);
        auto d = depth(p);
        if (0 == w || 0 == h || 0 == d) {
            continue;
        }

        v.resize(pitch(p));
        for (size_t z = 0; z < d; ++z) {
            size_t y1 = 0, y2 = h - 1;
            while (y1 < y2) {
                auto r1 = row(p, y1, z);
                auto r2 = row(p, y2, z);
                memcpy(v.data(), r1, v.size());
                memcpy(r1, r2, v.size());
                memcpy(r2, v.data(), v.size());
                ++y1;
                --y2;
            }
        }
    }
}

void RawImage::saveToPNG(const std::string& filepath, size_t sliceIndex, bool makeOpaque, bool clamp) const {
    if (empty()) {
        SNN_LOGW("Trying to save an empty image !");
        return;
    }
    if (sliceIndex >= depth()) {
        SNN_RIP("invalid slice index.");
    }
    const auto& colorDesc = getColorFormatDesc(format());
    if (step() != getColorFormatDesc(format()).bits) {
        SNN_RIP("does not support interleaved layout.");
    }
    int channels = colorDesc.ch;
    switch (format()) {
    case ColorFormat::R8:
    case ColorFormat::RGB8:
        break;
    case ColorFormat::RGBA8:
        if (makeOpaque) {
            snn::ManagedRawImage image = toRgba8(*this, true);
            image.saveToPNG(filepath, sliceIndex);
            return;
        }
        break;
    case ColorFormat::R32F:
    case ColorFormat::R16F:
        {
            snn::ManagedRawImage image;
            if (clamp) {
                image = toR8(::clamp(*this));
            } else {
                image = toR8(*this);
            }
            image.saveToPNG(filepath, sliceIndex);
            return;
        }
    case ColorFormat::RGB32F:
    case ColorFormat::RGB16F:
        {
            snn::ManagedRawImage image;
            if (clamp) {
                image = toRgb8(::clamp(*this));
            } else {
                image = toRgb8(*this);
            }
            image.saveToPNG(filepath, sliceIndex);
            return;
        }
    case ColorFormat::RGBA32F:
    case ColorFormat::RGBA16F:
        {
            snn::ManagedRawImage image;
            if (clamp) {
                image = toRgba8(::clamp(*this), makeOpaque);
            } else {
                image = toRgba8(*this, makeOpaque);
            }
            image.saveToPNG(filepath, sliceIndex);
            return;
        }
        break;
    default:
        SNN_RIP("unsupported color format: %s", colorDesc.name);
    }
    stbi_write_png(filepath.c_str(), (int) width(), (int) height(), channels, slice(0, sliceIndex), pitch(0));
}

void RawImage::saveToBIN(const std::string& filepath, bool convertToFP32) const {
    ManagedRawImage rgba128f;
    const uint8_t* pixels;
    size_t len;
    if (convertToFP32 && format() != snn::ColorFormat::RGBA32F) {
        if (format() == snn::ColorFormat::RGBA8 || format() == snn::ColorFormat::RGBA16F) {
            rgba128f = toRgba32f(*this);
        } else {
            rgba128f = toR32f(*this);
        }
        pixels = rgba128f.data();
        len = rgba128f.size();
    } else {
        pixels = data();
        len = size();
    }

    std::ofstream fp;
    fp.exceptions(std::ofstream::failbit); // may throw
    try {
        fp.open(filepath, std::ios::binary);
    } catch (const std::ios_base::failure& fail) {
        SNN_LOGE("open %s: %s", filepath.c_str(), fail.what());
        return;
    }
    char header[32] = {};
    std::snprintf(header, 32, "%d %d %d %d", width(), height(), depth(), channels());
    fp.write(header, 32);
    fp.write((const char*) pixels, len);
}

ColorFormat DesiredFormat(int nComponents) {
    SNN_ASSERT(nComponents >= 1 && nComponents <= 4);
    return nComponents == 1 ? ColorFormat::R8 : nComponents == 2 ? ColorFormat::RG8 : nComponents == 3 ? ColorFormat::RGB8 : ColorFormat::RGBA8;
}

void ManagedRawImage::store(const void* buffer, size_t length) {
    size_t imageSize = size();
    _pixels = AlignedAllocator::allocate(imageSize, alignment());
    if (!_pixels) {
        return;
    }
    if (buffer) {
        if (0 == length) {
            length = imageSize;
        } else if (length != imageSize) {
            SNN_LOGW("incoming pixel buffer size does not equal to calculated image size.");
        }
        memcpy(_pixels, buffer, std::min(imageSize, length));
    }
}

ManagedRawImage ManagedRawImage::loadFromFile(const std::string& filename, bool fromBin) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.good()) {
        SNN_RIP("Failed to open image file %s", filename.c_str());
    }
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    SNN_ASSERT(0 == file.tellg());
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    if (!file.good()) {
        SNN_RIP("Failed to read image file %s", filename.c_str());
    }
    int w, h, c, n_comps;
    void* pixels;
    SNN_LOGI("Opening file: %s", filename.c_str());
    if (!fromBin) {
        stbi_info(filename.c_str(), &w, &h, &n_comps);
        stbi_set_flip_vertically_on_load(false);
        pixels = stbi_load_from_memory((const stbi_uc*) buf.data(), (int) buf.size(), &w, &h, &c, n_comps);
        if (!pixels) {
            SNN_RIP(stbi_failure_reason());
        }
    } else {
        if (buf.size() < 32) {
            SNN_RIP("File %s has incorrect bin format !", filename.c_str());
        }
        int d = w = h = c = -1;
        std::stringstream header(std::string(buf.data(), 32));
        header >> w >> h >> d >> c;
        if (w < 0 || h < 0 || d < 0 || c < 0 || c % d != 0) {
            SNN_RIP("File %s has incorrect bin header format !", filename.c_str());
        }
        n_comps = c / d;
        pixels = &buf[32];
    }
    ColorFormat cf = DesiredFormat(n_comps);
    auto& fd = getColorFormatDesc(cf);
    SNN_ASSERT(c % fd.ch == 0);
    ManagedRawImage result(ImageDesc(cf, (uint32_t) w, (uint32_t) h, (uint32_t)(c / fd.ch), (uint32_t) c), pixels);
    if (!fromBin) {
        stbi_image_free(pixels);
    }
    return result;
}

ManagedRawImage ManagedRawImage::loadFromAsset(const std::string& assetName, bool fromBin) {
    std::vector<uint8_t> file = snn::loadEmbeddedAsset(assetName.c_str());
    if (file.empty()) {
        return {};
    }
    int width, height, channels;
    void* pixels;
    if (!fromBin) {
        stbi_set_flip_vertically_on_load(false);
        pixels = stbi_load_from_memory((const stbi_uc*) file.data(), (int) file.size(), &width, &height, &channels, STBI_rgb_alpha);
        if (!pixels) {
            SNN_RIP(stbi_failure_reason());
        }
    } else {
        if (file.size() < 32) {
            SNN_RIP("Asset %s has incorrect bin format !", assetName.c_str());
        }
        int depth = width = height = channels = -1;
        char* buf = reinterpret_cast<char*>(file.data());
        std::stringstream header(std::string(buf, 32));
        header >> width >> height >> depth >> channels;
        if (width < 0 || height < 0 || channels < 0) {
            SNN_RIP("Asset %s has incorrect bin header format !", assetName.c_str());
        }
        pixels = &buf[32];
    }
    ManagedRawImage result(ImageDesc(ColorFormat::RGBA8, (uint32_t) width, (uint32_t) height, (uint32_t)(((channels + 3) / 4)), (uint32_t) channels), pixels);
    if (!fromBin) {
        stbi_image_free(pixels);
    }

    return result;
}

float FP16::toFloat() const {
    // https://gist.github.com/rygorous/2156668
    static constexpr FP32 magic = {126 << 23};
    FP32 o;
    if (exponent == 0) {
        o.u = magic.u + mantissa;
        o.flt -= magic.flt;
    } else {
        o.mantissa = mantissa << 13;
        if (exponent == 0x1f) {
            o.exponent = 255;
        }
        else {
            o.exponent = 127 - 15 + exponent;
        }
    }
    o.sign = sign;
    return o.flt;
}

bool snn::toRgba32f(Rgba32f& dst, const uint8_t* src, ColorFormat srcFormat) {
    switch (srcFormat) {
    case ColorFormat::RGBA32F:
        dst = *(Rgba32f*) src;
        break;
    case ColorFormat::RGB32F:
        dst.red = ((const float*) src)[0];
        dst.green = ((const float*) src)[1];
        dst.blue = ((const float*) src)[2];
        dst.alpha = 1.0f;
        break;
    case ColorFormat::RGBA16F: {
        auto f16 = (const uint16_t*) src;
        dst.red    = FP16::toFloat(f16[0]);
        dst.green    = FP16::toFloat(f16[1]);
        dst.blue    = FP16::toFloat(f16[2]);
        dst.alpha    = FP16::toFloat(f16[3]);
        break;
    }
    case ColorFormat::RGBA8:
        dst.red = (float) src[0] / 255.0f;
        dst.green = (float) src[1] / 255.0f;
        dst.blue = (float) src[2] / 255.0f;
        dst.alpha = (float) src[3] / 255.0f;
        // SNN_LOGD("%s:%d: %d:%d:%d:%d -- %f:%f:%f:%f\n", __FILENAME__,__LINE__,src[0], src[1],src[2],src[3], dst.red,dst.green, dst.blue, dst.alpha);
        break;
    case ColorFormat::RGB8:
        dst.red = (float) src[0] / 255.0f;
        dst.green = (float) src[1] / 255.0f;
        dst.blue = (float) src[2] / 255.0f;
        dst.alpha = 1.0f;
        break;
    case ColorFormat::R8:
        dst.red = (float) src[0] / 255.0f;
        dst.green = 0.0f;
        dst.blue = 0.0f;
        dst.alpha = 0.0f;
        break;
    case ColorFormat::RGBA16U:
    case ColorFormat::RGB16F:
    case ColorFormat::R32F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::R16F:
    case ColorFormat::RG8:
    default:
        SNN_RIP("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
    }

    return true;
}

bool snn::toRgba16f(Rgba16f& dst, const uint8_t* src, ColorFormat srcFormat) {
    switch (srcFormat) {
    case ColorFormat::RGBA16F:
        dst = *(Rgba16f*) src;
        break;
    case ColorFormat::RGB16F:
        dst.r = ((const uint16_t*) src)[0];
        dst.g = ((const uint16_t*) src)[1];
        dst.b = ((const uint16_t*) src)[2];
        dst.a = FP32::toHalf(1.0f);
        break;
    case ColorFormat::RGBA32F: {
        auto f32 = (const float*) src;
        dst.r    = FP32::toHalf(f32[0]);
        dst.g    = FP32::toHalf(f32[1]);
        dst.b    = FP32::toHalf(f32[2]);
        dst.a    = FP32::toHalf(f32[3]);
        // printf("Convert:%f->%f\n", f32[0], FP16::toFloat(dst.r));
        break;
    }
    case ColorFormat::RGB32F: {
        auto f32 = (const float*) src;
        dst.r    = FP32::toHalf(f32[0]);
        dst.g    = FP32::toHalf(f32[1]);
        dst.b    = FP32::toHalf(f32[2]);
        dst.a    = FP32::toHalf(1.0f);
        break;
    }
    case ColorFormat::RGBA8:
        dst.r = FP32::toHalf((float) src[0] / 255.0f);
        dst.g = FP32::toHalf((float) src[1] / 255.0f);
        dst.b = FP32::toHalf((float) src[2] / 255.0f);
        dst.a = FP32::toHalf((float) src[3] / 255.0f);
        // printf("%s:%d: %d:%d:%d:%d -- %f:%f:%f:%f\n", __FILENAME__,__LINE__,src[0], src[1],src[2],src[3], dst.r,dst.g, dst.b, dst.a);
        break;
    case ColorFormat::RGB8:
        dst.r = FP32::toHalf((float) src[0] / 255.0f);
        dst.g = FP32::toHalf((float) src[1] / 255.0f);
        dst.b = FP32::toHalf((float) src[2] / 255.0f);
        dst.a = FP32::toHalf(1.0f);
        break;
    case ColorFormat::RGBA16U:
    case ColorFormat::R32F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::R16F:
    case ColorFormat::RG8:
    case ColorFormat::R8:
    default:
        SNN_RIP("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
    }

    return true;
}

bool snn::toR32f(R32f& dst, const uint8_t* src, ColorFormat srcFormat) {
    // std::cout << "The format inside image.cpp toR32f() function is "<<snn::getColorFormatDesc(srcFormat).name<<std::endl;
    switch (srcFormat) {
    case ColorFormat::R32F: {
        dst = *(R32f*) src;
        break;
    }
    case ColorFormat::R16F: {
        auto f16 = (const uint16_t*) src;
        dst.red    = FP16::toFloat(f16[0]);
        break;
    }
    case ColorFormat::R8: {
        dst.red = (float) src[0] / 255.0f;
        break;
    }
    case ColorFormat::RGBA8: {
        dst.red = (float) src[0] / 255.0f;
        break;
    }
    case ColorFormat::RGBA16U:
    case ColorFormat::RGB16F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::RG8:
    default:
        SNN_RIP("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
    }
    return true;
}

bool snn::toRgba8(Rgba8& dst, const uint8_t* src, ColorFormat srcFormat, bool normalize) {
    auto f32ToU8N = normalize
        // The reason for this double cast is to ensure that on all compilers
        // when casting negative floats, we want the result roll over, not saturate
        ? [](float f) { return (uint8_t)(int8_t)(f * 255.0f); }
        : [](float f) { return (uint8_t)(int8_t)f; };
    switch (srcFormat) {
    case ColorFormat::RGBA32F: {
        auto f32 = (const float*) src;
        dst.r    = f32ToU8N(f32[0]);
        dst.g    = f32ToU8N(f32[1]);
        dst.b    = f32ToU8N(f32[2]);
        dst.a    = f32ToU8N(f32[3]);
        break;
    }
    case ColorFormat::RGB32F: {
        auto f32 = (const float*) src;
        dst.r    = f32ToU8N(f32[0]);
        dst.g    = f32ToU8N(f32[1]);
        dst.b    = f32ToU8N(f32[2]);
        dst.a    = 255;
        break;
    }
    case ColorFormat::RGBA16F: {
        auto f16 = (const uint16_t*) src;
        dst.r    = f32ToU8N(FP16::toFloat(f16[0]));
        dst.g    = f32ToU8N(FP16::toFloat(f16[1]));
        dst.b    = f32ToU8N(FP16::toFloat(f16[2]));
        dst.a    = f32ToU8N(FP16::toFloat(f16[3]));
        break;
    }
    case ColorFormat::RGB16F: {
        auto f16 = (const uint16_t*) src;
        dst.r    = f32ToU8N(FP16::toFloat(f16[0]));
        dst.g    = f32ToU8N(FP16::toFloat(f16[1]));
        dst.b    = f32ToU8N(FP16::toFloat(f16[2]));
        dst.a    = 255;
        break;
    }
    case ColorFormat::RGBA8:
        dst.r = src[0];
        dst.g = src[1];
        dst.b = src[2];
        dst.a = src[3];
        break;
    case ColorFormat::RGB8:
        dst.r = src[0];
        dst.g = src[1];
        dst.b = src[2];
        dst.a = 255;
        break;
    case ColorFormat::RGBA16U:
    case ColorFormat::R32F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::R16F:
    case ColorFormat::RG8:
    case ColorFormat::R8:
    default:
        SNN_RIP("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
    }

    return true;
}

bool snn::toR8(R8& dst, const uint8_t* src, ColorFormat srcFormat, bool normalize) {
    auto f32ToU8N = normalize
        // The reason for this double cast is to ensure that on all compilers
        // when casting negative floats, we want the resunt roll over, not saturate
        ? [](float f) { return (uint8_t)(int8_t)(f * 255.0f); }
        : [](float f) { return (uint8_t)(int8_t)f; };
    switch (srcFormat) {
    case ColorFormat::R8: {
        dst.r = src[0];
        break;
    }
    case ColorFormat::R32F: {
        auto f32 = (const float*) src;
        dst.r    = f32ToU8N(f32[0]);
        break;
    }
    case ColorFormat::R16F: {
        auto f16 = (const uint16_t*) src;
        dst.r    = f32ToU8N(FP16::toFloat(f16[0]));
        break;
    }
    case ColorFormat::RGBA16U:
    case ColorFormat::RGB16F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::RG8:
    default:
        SNN_RIP("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
    }
    return true;
}

bool snn::toRgb8(Rgb8& dst, const uint8_t* src, ColorFormat srcFormat, bool normalize) {
    Rgba8 t;
    if (!toRgba8(t, src, srcFormat, normalize)) {
        return false;
    }
    dst.r = t.r;
    dst.g = t.g;
    dst.b = t.b;
    return true;
}

bool snn::toRgba32f(const RawImage& src, TypedImage<Rgba32f>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toRgba16f(const RawImage& src, TypedImage<Rgba16f>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba16f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toR32f(const RawImage& src, TypedImage<R32f>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toR32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toRgba32f(const RawImage& src, TypedImage<Rgba32f>& dst, float min, float max) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return false;
                    }
                    dst.at(p, x, y, z) = (dst.at(p, x, y, z) * (max - min)) + min;
                }
            }
        }
    }
    return true;
}

static bool norm2rgba32f(Rgba32f& dst, const uint8_t* src, ColorFormat srcFormat, const std::vector<float>& means, const std::vector<float>& norms) {
    switch (srcFormat) {
    case ColorFormat::RGBA32F:
        dst.red = (((const float*) src)[0] - means[0]) * norms[0];
        dst.green = (((const float*) src)[1] - means[1]) * norms[1];
        dst.blue = (((const float*) src)[2] - means[2]) * norms[2];
        dst.alpha = (((const float*) src)[3] - means[3]) * norms[3];
        break;
    case ColorFormat::RGB32F:
        dst.red = (((const float*) src)[0] - means[0]) * norms[0];
        dst.green = (((const float*) src)[1] - means[1]) * norms[1];
        dst.blue = (((const float*) src)[2] - means[2]) * norms[2];
        dst.alpha = 1.0f;
        break;
    case ColorFormat::RGBA16F: {
        auto f16 = (const uint16_t*) src;
        dst.red    = FP16::toFloat((f16[0] - means[0]) * norms[0]);
        dst.green    = FP16::toFloat((f16[1] - means[1]) * norms[1]);
        dst.blue    = FP16::toFloat((f16[2] - means[2]) * norms[2]);
        dst.alpha    = FP16::toFloat((f16[3] - means[3]) * norms[3]);
        break;
    }
    case ColorFormat::RGBA8:
        dst.red = (float) ((src[0] - means[0]) * norms[0]);
        dst.green = (float) ((src[1] - means[1]) * norms[1]);
        dst.blue = (float) ((src[2] - means[2]) * norms[2]);
        dst.alpha = (float) ((src[3] - means[3]) * norms[3]);
        break;
    case ColorFormat::RGB8:
        dst.red = (float) ((src[0] - means[0]) * norms[0]);
        dst.green = (float) ((src[1] - means[1]) * norms[1]);
        dst.blue = (float) ((src[2] - means[2]) * norms[2]);
        dst.alpha = 1.0f;
        break;
    case ColorFormat::R8:
        dst.red = (float)   ((src[0] - means[0]) * norms[0]);
        dst.green = (float) ((0.0f - means[0]) * norms[0]);
        dst.blue =  (float) ((0.0f - means[0]) * norms[0]);
        dst.alpha = (float) ((0.0f - means[0]) * norms[0]);
        break;
    case ColorFormat::RGBA16U:
    case ColorFormat::RGB16F:
    case ColorFormat::R32F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::R16F:
    case ColorFormat::RG8:
    default:
        SNN_RIP("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
    }

    return true;
}

bool snn::normalize(const RawImage& src, TypedImage<Rgba32f>& dst, const std::vector<float>& means, const std::vector<float>& norms) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }

    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::norm2rgba32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p), means, norms)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

ManagedImage<Rgba32f> snn::normalize(const RawImage& src, const std::vector<float>& means, const std::vector<float>& norms) {
    auto planes = convertPlanes(src, ColorFormat::RGBA32F);
    auto dst = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
    normalize(src, dst, means, norms);
    return dst;
}

void snn::clamp(const RawImage& src, RawImage& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }

    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        auto format  = src.format(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth || dst.format(p) != format) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        const auto& cfd = getColorFormatDesc(format);
        auto ch = cfd.ch;
        auto cd = cfd.colorDepth();
        auto cft = getColorFormatType(format);
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    uint8_t* dstPtr = dst.at(p, x, y, z);
                    const uint8_t* srcPtr = src.at(p, x, y, z);
                    for (size_t c = 0; c < ch; ++c, srcPtr += cd, dstPtr += cd) {
                        switch (cft) {
                        case snn::ColorFormatType::UINT8:
                            {
                                *dstPtr = *srcPtr;
                            }
                            break;
                        case snn::ColorFormatType::FLOAT16:
                            {
                                const FP16* ptrFP16 = reinterpret_cast<const snn::FP16*>(srcPtr);
                                float val = FP16::toFloat(ptrFP16->u);
                                val = std::min(1.0f, std::max(0.0f, val));
                                *dstPtr = FP32::toHalf(val);
                            }
                            break;
                        case snn::ColorFormatType::FLOAT32:
                            {
                                float val = *reinterpret_cast<const float*>(srcPtr);
                                float* dstFloat = reinterpret_cast<float*>(dstPtr);
                                *dstFloat = std::min(1.0f, std::max(0.0f, val));
                            }
                            break;
                        default:
                            SNN_RIP("unsupported color format: %s", cfd.name);
                        }
                    }
                }
            }
        }
    }
}

ManagedRawImage snn::clamp(const RawImage& src) {
    auto dst = ManagedRawImage(src.getDesc());
    clamp(src, dst);
    return dst;
}

void clamp(RawImage& srcDst) {
    clamp(srcDst, srcDst);
}

ManagedImage<Rgba32f> snn::toRgba32f(const RawImage& src) {
    auto planes = convertPlanes(src, ColorFormat::RGBA32F);
    auto dst = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
    toRgba32f(src, dst);
    return dst;
}

ManagedImage<Rgba16f> snn::toRgba16f(const RawImage& src) {
    auto planes = convertPlanes(src, ColorFormat::RGBA16F);
    auto dst = ManagedImage<Rgba16f>(ImageDesc(std::move(planes)));
    toRgba16f(src, dst);
    return dst;
}

ManagedImage<R32f> snn::toR32f(const RawImage& src) {
    auto planes = convertPlanes(src, ColorFormat::R32F);
    auto dst = ManagedImage<R32f>(ImageDesc(std::move(planes)));
    toR32f(src, dst);
    return dst;
}

ManagedImage<Rgba32f> snn::toRgba32f(const RawImage& src, float min, float max) {
    auto planes = convertPlanes(src, ColorFormat::RGBA32F);
    auto dst = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
    toRgba32f(src, dst, min, max);
    return dst;
}

ManagedImage<R32f> snn::toR32f(const RawImage& src, float min, float max) {
    auto planes = convertPlanes(src, ColorFormat::R32F);
    auto dst = ManagedImage<R32f>(ImageDesc(std::move(planes)));
    toR32f(src, dst, min, max);
    return dst;
}

ManagedImage<Rgba8> snn::toRgba8(const RawImage& src, bool makeOpaque) {
    auto planes = convertPlanes(src, ColorFormat::RGBA8);
    auto dst = ManagedImage<Rgba8>(ImageDesc(std::move(planes)));
    if (!makeOpaque) {
        toRgba8(src, dst);
    } else {
        toRgba8(src, dst, ColorFormat::RGB8);
    }
    return dst;
}

ManagedImage<Rgb8> snn::toRgb8(const RawImage& src) {
    auto planes = convertPlanes(src, ColorFormat::RGB8);
    auto dst = ManagedImage<Rgb8>(ImageDesc(std::move(planes)));
    toRgb8(src, dst);
    return dst;
}

ManagedImage<R8> snn::toR8(const RawImage& src) {
    auto planes = convertPlanes(src, ColorFormat::R8);
    auto dst = ManagedImage<R8>(ImageDesc(std::move(planes)));
    toR8(src, dst);
    return dst;
}

bool snn::toR32f(const RawImage& src, TypedImage<R32f>& dst, float min, float max) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toR32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return false;
                    }
                    dst.at(p, x, y, z) = (dst.at(p, x, y, z) * (max - min)) + min;
                }
            }
        }
    }
    return true;
}

bool snn::toRgba8(const RawImage& src, TypedImage<Rgba8>& dst, ColorFormat format) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba8(dst.at(p, x, y, z), src.at(p, x, y, z), format, true)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toRgba8(const RawImage& src, TypedImage<Rgba8>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba8(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p), true)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toR8(const RawImage& src, TypedImage<R8>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toR8(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toRgb8(const RawImage& src, TypedImage<Rgb8>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_RIP("mismatched src/dst image dimension.");
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_RIP("mismatched src/dst image dimension.");
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgb8(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

static bool isNv12(const snn::RawImage& i) { return 2 == i.desc().planes.size() && ColorFormat::R8 == i.format(0) && ColorFormat::RG8 == i.format(1); }

static bool isI420(const snn::RawImage& i) {
    return 3 == i.desc().planes.size() && ColorFormat::R8 == i.format(0) && ColorFormat::R8 == i.format(1) && ColorFormat::R8 == i.format(2);
}

static bool isRgba8(const snn::RawImage& i) { return 1 == i.desc().planes.size() && ColorFormat::RGBA8 == i.format(); }

bool snn::rgba8ToI420(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(src) || !isI420(dst)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::ABGRToI420(src.plane(0), int(src.pitch(0)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)), dst.plane(2), int(dst.pitch(2)),
                       int(width), int(height));
    return true;
}

bool snn::rgba8ToNv12(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(src) || !isNv12(dst)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::ABGRToNV12(src.plane(0), int(src.pitch(0)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)), int(width), int(height));
    return true;
}

bool snn::nv12ToRgba8(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(dst) || !isNv12(src)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::NV12ToABGR(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), dst.plane(0), int(dst.pitch(0)), int(width), int(height));
    return true;
}

bool snn::rgba8ToNv21(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(src) || !isNv12(dst)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::ABGRToNV21(src.plane(0), int(src.pitch(0)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)), int(width), int(height));
    return true;
}

bool snn::nv21ToRgba8(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(dst) || !isNv12(src)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::NV21ToABGR(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), dst.plane(0), int(dst.pitch(0)), int(width), int(height));
    return true;
}

bool snn::nv12ToI420(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isNv12(src) || !isI420(dst)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = src.width(), height = src.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::NV12ToI420(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)),
                       dst.plane(2), int(dst.pitch(2)), int(width), int(height));
    return true;
}

bool snn::i420ToRgba8(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isI420(src) || !isRgba8(dst)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = src.width(), height = src.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::I420ToABGR(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), src.plane(2), int(src.pitch(2)), dst.plane(0), int(dst.pitch(0)),
                       int(width), int(height));
    return true;
}

bool snn::i420ToNv12(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isI420(src) || !isNv12(dst)) {
        SNN_RIP("invalid src/dst image format");
    }
    auto width = src.width(), height = src.height();
    if (src.width() != width || src.height() != height) {
        SNN_RIP("mismatched src/dst image size");
    }
    libyuv::I420ToNV12(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), src.plane(2), int(src.pitch(2)), dst.plane(0), int(dst.pitch(0)),
                       dst.plane(1), int(dst.pitch(1)), int(width), int(height));
    return true;
}


void snn::printC4Buffer(float *buffer, int input_h, int input_w, int input_c, std::ostream& stream) {
    printf("--------------------C4 Buffer with width:%d, height:%d, depth:%d -------------------- \n", input_w, input_h, input_c);
    for (int k = 0; k < input_c; k++) {
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 0);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------R----" + std::to_string(j) + "------------" << std::endl;
        }
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 1);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------G----" + std::to_string(j) + "------------" << std::endl;
        }
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 2);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------B----" + std::to_string(j) + "------------" << std::endl;
        }
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = *(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 3);
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------A----" + std::to_string(j) + "------------" << std::endl;
        }
        stream << "*************C4 channel: " + std::to_string(k) + "***********" << std::endl;
    }
}

void snn::printC4BufferFP16(uint16_t *buffer, int input_h, int input_w, int input_c, std::ostream& stream) {
    printf("--------------------C4 Buffer FP16 with width:%d, height:%d, depth:%d -------------------- \n", input_w, input_h, input_c);
    for (int k = 0; k < input_c; k++) {
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = FP16::toFloat(*(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 0));
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------R----" + std::to_string(j) + "------------" << std::endl;
        }
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = FP16::toFloat(*(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 1));
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------G----" + std::to_string(j) + "------------" << std::endl;
        }
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = FP16::toFloat(*(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 2));
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------B----" + std::to_string(j) + "------------" << std::endl;
        }
        for (int j = 0; j < input_h; j++) {
            for (int i = 0; i < input_w; i++) {
                auto v = FP16::toFloat(*(buffer + (k *input_h * input_w + j * input_w  + i) * 4 + 3));
                stream << std::setw(7) << v << ",";
            }
            stream << "\n-------A----" + std::to_string(j) + "------------" << std::endl;
        }
        stream << "*************C4 channel: " + std::to_string(k) + "***********" << std::endl;
    }
}
