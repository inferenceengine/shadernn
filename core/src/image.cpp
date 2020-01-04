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
#include "pch.h"
#include "snn/image.h"
#include <iostream>
#include <libyuv.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <fstream>

using namespace snn;

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

void RawImage::saveToPNG(const std::string& filepath, size_t sliceIndex) const {
    if (empty()) {
        return;
    }

    if (sliceIndex >= depth()) {
        SNN_LOGE("invalid slice index.");
        return;
    }
    if (step() != getColorFormatDesc(format()).bits) {
        SNN_LOGE("does not support interleaved layout.");
        return;
    }
    int channels;
    switch (format()) {
    case ColorFormat::RGB8:
        channels = 3;
        break;
    case ColorFormat::RGBA8:
        channels = 4;
        break;
    case ColorFormat::RGBA32F:
        channels = 4;
        break;
    case ColorFormat::RGBA16F:
        channels = 4;
        break;
    case ColorFormat::R32F:
        channels = 1;
        break;
    case ColorFormat::R8:
        channels = 1;
        break;
    case ColorFormat::R16F:
        channels = 1;
        break;
    default:
        // SNN_LOGE("unsupported image format.");
        return;
    }
    if (format() == ColorFormat::RGBA32F || format() == ColorFormat::RGBA16F || format() == ColorFormat::R32F) {
        std::vector<uint8_t> dataBuffer;
        auto pixelBuffer = this->data();
        auto& colorDesc  = snn::getColorFormatDesc(format());
        int byteSize     = colorDesc.bits / (8 * colorDesc.ch);
        for (std::size_t i = 0; i < this->size(); i += byteSize) {
            float val;
            if (format() == ColorFormat::RGBA32F) {
                unsigned char buffer[] = {pixelBuffer[i], pixelBuffer[i + 1], pixelBuffer[i + 2], pixelBuffer[i + 3]};
                memcpy(&val, &buffer, sizeof(float));
            } else {
                unsigned char buffer[] = {pixelBuffer[i], pixelBuffer[i + 1]};
                uint16_t uintVal;
                memcpy(&uintVal, buffer, sizeof(uint16_t));
                val = snn::convertToHighPrecision(uintVal);
            }
            if (val <= 1.0f) {
                val = val * 255.0f;
            } else if (val <= 0.0f && val >= -1.0f) {
                val = val * 127.5f + 127.5f;
            }
            dataBuffer.push_back((uint8_t) val);
        }
        stbi_write_png(filepath.c_str(), (int) width(), (int) height(), channels, dataBuffer.data(), pitch(0));
    } else {
        stbi_write_png(filepath.c_str(), (int) width(), (int) height(), channels, slice(0, sliceIndex), pitch(0));
    }
}

void RawImage::saveToBIN(const std::string& filepath) const {
    RawImage rgba128f;
    if (this->format() == snn::ColorFormat::RGBA8 || this->format() == snn::ColorFormat::RGBA32F || this->format() == snn::ColorFormat::RGBA16F) {
        rgba128f = toRgba32f(*this);
    } else {
        rgba128f = toR32f(*this);
    }

    auto fp = std::ofstream(filepath, std::ios::binary);
    if (!fp.good()) {
        return;
    }
    char header[32] = {};
    std::snprintf(header, 32, "%d %d %d %d", width(), height(), depth(), channels());
    fp.write(header, 32);
    fp.write((const char*) rgba128f.data(), rgba128f.size());
}

ColorFormat DesiredFormat(int nComponents) {
    return nComponents == 1 ? ColorFormat::R8 : nComponents == 2 ? ColorFormat::RG8 : nComponents == 3 ? ColorFormat::RGB8 : ColorFormat::RGBA8;
}

ManagedRawImage ManagedRawImage::loadFromFile(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.good()) {
        SNN_LOGE("Failed to open image file %s", filename.c_str());
        return {};
    }
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    SNN_ASSERT(0 == file.tellg());
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    if (!file.good()) {
        SNN_LOGE("Failed to read image file %s", filename.c_str());
        return {};
    }
    int w, h, c, n_comps;
    SNN_LOGI("Opening file: %s", filename.c_str());
    stbi_info(filename.c_str(), &w, &h, &n_comps);
    stbi_set_flip_vertically_on_load(false);
    auto pixels = stbi_load_from_memory((const stbi_uc*) buf.data(), (int) buf.size(), &w, &h, &c, n_comps);
    if (!pixels) {
        SNN_LOGE(stbi_failure_reason());
        return {};
    }
    ManagedRawImage result(ImageDesc(DesiredFormat(n_comps), (uint32_t) w, (uint32_t) h, (uint32_t)(c / 4), (uint32_t) c), pixels);
    stbi_image_free(pixels);
    return result;
}

ManagedRawImage ManagedRawImage::loadFromAsset(std::string assetName) {
    auto file = snn::loadEmbeddedAsset(assetName.c_str());
    if (file.empty()) {
        return {};
    }
    int width, height, channels;
    stbi_set_flip_vertically_on_load(false);
    auto pixels = stbi_load_from_memory((const stbi_uc*) file.data(), (int) file.size(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        SNN_LOGE(stbi_failure_reason());
        return {};
    }
    ManagedRawImage result(ImageDesc(ColorFormat::RGBA8, (uint32_t) width, (uint32_t) height, (uint32_t)(((channels + 3) / 4)), (uint32_t) channels), pixels);
    stbi_image_free(pixels);
    return result;
}

inline float FP16::toFloat() const {
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

static bool toRgba32f(Rgba32f& dst, const uint8_t* src, ColorFormat srcFormat) {
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
        // SNN_LOGD("%s:%d: %d:%d:%d:%d -- %f:%f:%f:%f\n", __FUNCTION__,__LINE__,src[0], src[1],src[2],src[3], dst.red,dst.green, dst.blue, dst.alpha);
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
        SNN_LOGE("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
        return false;
    }

    return true;
}

static bool toRgba16f(Rgba16f& dst, const uint8_t* src, ColorFormat srcFormat) {
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
        // printf("%s:%d: %d:%d:%d:%d -- %f:%f:%f:%f\n", __FUNCTION__,__LINE__,src[0], src[1],src[2],src[3], dst.r,dst.g, dst.b, dst.a);
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
        SNN_LOGE("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
        return false;
    }

    return true;
}

static bool toR32f(R32f& dst, const uint8_t* src, ColorFormat srcFormat) {
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
        SNN_LOGE("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
        return false;
    }
    return true;
}

static bool toRgba8(Rgba8& dst, const uint8_t* src, ColorFormat srcFormat) {
    auto f32ToU8N = [](float f) {
        if (f < 1.0001f) {
            auto u = (uint32_t)(f * 255.0f);
            auto r = (u < 255) ? u : 255u;
            return (uint8_t) r;
        } else {
            auto u = (uint32_t)(f);
            auto r = (u < 255) ? u : 255u;
            return (uint8_t) r;
        }
    };

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
    case ColorFormat::RGB16F:
    case ColorFormat::R32F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::R16F:
    case ColorFormat::RG8:
    case ColorFormat::R8:
    default:
        SNN_LOGE("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
        return false;
    }

    return true;
}

static bool toR8(R8& dst, const uint8_t* src, ColorFormat srcFormat) {
    // std::cout << "Entered 3rd part "<< std::endl;
    auto f32ToU8N = [](float f) {
        if (f < 1.0f) {
            auto u = (uint32_t)(f * 255.0f);
            auto r = (u < 255) ? u : 255u;
            return (uint8_t) r;
        } else {
            auto u = (uint32_t)(f);
            auto r = (u < 255) ? u : 255u;
            return (uint8_t) r;
        }
    };
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
        SNN_LOGE("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
        return false;
    }
    return true;
}

static bool toRgb8(Rgb8& dst, const uint8_t* src, ColorFormat srcFormat) {
    Rgba8 t;
    if (!toRgba8(t, src, srcFormat)) {
        return false;
    }
    dst.r = t.r;
    dst.g = t.g;
    dst.b = t.b;
    return true;
}

bool snn::toRgba32f(const RawImage& src, TypedImage<Rgba32f>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    // SNN_LOGD("%s:%d: %zu:%d:%d:%d\n", __FUNCTION__,__LINE__,src.desc().planes.size(), src.width(0), src.height(0), src.depth(0));
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toRgba16f(const RawImage& src, TypedImage<Rgba16f>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    // printf("%s:%d: %zu:%d:%d:%d\n", __FUNCTION__,__LINE__,src.desc().planes.size(), src.width(0), src.height(0), src.depth(0));
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba16f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toR32f(const RawImage& src, TypedImage<R32f>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toR32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toRgba32f(const RawImage& src, TypedImage<Rgba32f>& dst, float min, float max) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                    // SNN_LOGD("%f %f %f %f", dst.at(p, x, y, z).r, dst.at(p, x, y, z).g, dst.at(p, x, y, z).b, dst.at(p, x, y, z).a);
                    if (dst.at(p, x, y, z) < 1.0f) {
                        dst.at(p, x, y, z) = (dst.at(p, x, y, z) * (max - min)) + min;
                    } else {
                        dst.at(p, x, y, z) = ((dst.at(p, x, y, z) / 255.0) * (max - min)) + min;
                    }
                    // std::cout << dst.at(p, x, y, z).r << ", " << dst.at(p, x, y, z).g << ", " << dst.at(p, x, y, z).b << ", " << dst.at(p, x, y, z).a <<
                    // std::endl;
                }
            }
        }
    }
    return true;
}

bool snn::toRgba16f(const RawImage& src, TypedImage<Rgba16f>& dst, float min, float max) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba16f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

static bool norm2rgba32f(Rgba32f& dst, const uint8_t* src, ColorFormat srcFormat, std::vector<float>& means, std::vector<float>& norms) {
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
        // SNN_LOGD("%s:%d: %d:%d:%d:%d -- %f:%f:%f:%f\n", __FUNCTION__,__LINE__,src[0], src[1],src[2],src[3], dst.r,dst.g, dst.b, dst.a);
        break;
    case ColorFormat::RGB8:
        dst.red = (float) ((src[0] - means[0]) * norms[0]);
        dst.green = (float) ((src[1] - means[1]) * norms[1]);
        dst.blue = (float) ((src[2] - means[2]) * norms[2]);
        dst.alpha = 1.0f;
        break;
    case ColorFormat::RGBA16U:
    case ColorFormat::RGB16F:
    case ColorFormat::R32F:
    case ColorFormat::SRGB8_A8:
    case ColorFormat::SRGB8:
    case ColorFormat::R16F:
    case ColorFormat::RG8:
    case ColorFormat::R8:
    default:
        SNN_LOGE("unsupported color format: %s", getColorFormatDesc(srcFormat).name);
        return false;
    }

    return true;
}

bool snn::normalize(const RawImage& src, TypedImage<Rgba32f>& dst, std::vector<float>& means, std::vector<float>& norms) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }

    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::norm2rgba32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p), means, norms)) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

ManagedImage<Rgba32f> snn::normalize(const RawImage& src, std::vector<float>& means, std::vector<float>& norms) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::RGBA32F;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
    normalize(src, dst, means, norms);
    return dst;
}

void printOut(RawImage _images) {
    SNN_LOGD("----format:%d planes:%d, width:%d, height:%d, depth:%d----- \n", (int) _images.format(), _images.planes(), _images.width(), _images.height(),
             _images.depth());
    for (uint32_t p = 0; p < _images.planes(); p++) {
        for (uint32_t k = 0; k < _images.depth(); k++) {
            for (uint32_t j = 0; j < _images.height(); j++) {
                for (uint32_t i = 0; i < _images.width(); i++) {
                    std::cout << "M(" << p << ", " << k << ", " << j << ", " << i << "): ";
                    if (_images.format() == ColorFormat::RGBA8) {
                        auto v = (int) *(_images.at(p, i, j, k));
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(_images.at(p, i, j, k) + 1);
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(_images.at(p, i, j, k) + 2);
                        std::cout << std::setw(4) << v << ",";
                        v = (int) *(_images.at(p, i, j, k) + 3);
                        std::cout << std::setw(4) << v << ",";
                    } else if (_images.format() == ColorFormat::RGBA32F) {
                        auto v = *((float*) _images.at(p, i, j, k));
                        std::cout << std::setw(7) << v << ",";
                        v = *((float*) (_images.at(p, i, j, k) + 4));
                        std::cout << std::setw(7) << v << ",";
                        v = *((float*) (_images.at(p, i, j, k) + 8));
                        std::cout << std::setw(7) << v << ",";
                        v = *((float*) (_images.at(p, i, j, k) + 12));
                        std::cout << std::setw(7) << v << ",";
                    } else {
                        auto v = (int) *(_images.at(p, i, j, k));
                        std::cout << std::setw(4) << v << ",";
                    }
                }
                std::cout << "\n-----------" + std::to_string(j) + "------------" << std::endl;
                std::cout << std::endl;
            }
            std::cout << "**************" + std::to_string(k) + "***********" << std::endl;
        }
        std::cout << "**************" + std::to_string(p) + "***********" << std::endl;
    }
}

bool snn::toR32f(const RawImage& src, TypedImage<R32f>& dst, float min, float max) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toR32f(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                    // // SNN_LOGD("%f %f %f %f", dst.at(p, x, y, z).r, dst.at(p, x, y, z).g, dst.at(p, x, y, z).b, dst.at(p, x, y, z).a);
                    if (dst.at(p, x, y, z) < 1.0f) {
                        dst.at(p, x, y, z) = (dst.at(p, x, y, z) * (max - min)) + min;
                    } else {
                        dst.at(p, x, y, z) = ((dst.at(p, x, y, z) / 255.0) * (max - min)) + min;
                    }
                    // std::cout << dst.at(p, x, y, z).r << ", " << dst.at(p, x, y, z).g << ", " << dst.at(p, x, y, z).b << ", " << dst.at(p, x, y, z).a <<
                    // std::endl;
                }
            }
        }
    }
    return true;
}

ManagedImage<Rgba32f> snn::toRgba32f(const RawImage& src) {
    // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::RGBA32F;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
    toRgba32f(src, dst);
    // printOut(dst);
    return dst;
}

ManagedImage<Rgba16f> snn::toRgba16f(const RawImage& src) {
    // printf("%s:%d\n", __FUNCTION__,__LINE__);
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::RGBA16F;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<Rgba16f>(ImageDesc(std::move(planes)));
    toRgba16f(src, dst);
    // printOut(dst);
    return dst;
}

ManagedImage<R32f> snn::toR32f(const RawImage& src) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::R32F;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<R32f>(ImageDesc(std::move(planes)));
    toR32f(src, dst);
    return dst;
}

ManagedImage<Rgba32f> snn::toRgba32f(const RawImage& src, float min, float max) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::RGBA32F;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<Rgba32f>(ImageDesc(std::move(planes)));
    toRgba32f(src, dst, min, max);
    return dst;
}

ManagedImage<Rgba16f> snn::toRgba16f(const RawImage& src, float min, float max) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::RGBA16F;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<Rgba16f>(ImageDesc(std::move(planes)));
    toRgba16f(src, dst, min, max);
    return dst;
}

ManagedImage<R32f> snn::toR32f(const RawImage& src, float min, float max) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::R32F;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<R32f>(ImageDesc(std::move(planes)));
    toR32f(src, dst, min, max);
    return dst;
}

bool snn::toRgba8(const RawImage& src, TypedImage<Rgba8>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
        }
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!::toRgba8(dst.at(p, x, y, z), src.at(p, x, y, z), src.format(p))) {
                        return {};
                    }
                }
            }
        }
    }
    return true;
}

bool snn::toR8(const RawImage& src, TypedImage<R8>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
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

ManagedImage<Rgba8> snn::toRgba8(const RawImage& src) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::RGBA8;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<Rgba8>(ImageDesc(std::move(planes)));
    toRgba8(src, dst);
    return dst;
}

ManagedImage<R8> snn::toR8(const RawImage& src) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::R8;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<R8>(ImageDesc(std::move(planes)));
    toR8(src, dst);
    return dst;
}

bool snn::toRgb8(const RawImage& src, TypedImage<Rgb8>& dst) {
    if (src.desc().planes.size() != dst.desc().planes.size()) {
        SNN_LOGE("mismatched src/dst image dimension.");
        return false;
    }
    for (size_t p = 0; p < src.desc().planes.size(); ++p) {
        auto width  = src.width(p);
        auto height = src.height(p);
        auto depth  = src.depth(p);
        if (dst.width(p) != width || dst.height(p) != height || dst.depth(p) != depth) {
            SNN_LOGE("mismatched src/dst image dimension.");
            return false;
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

ManagedImage<Rgb8> snn::toRgb8(const RawImage& src) {
    auto planes = src.desc().planes;
    for (auto& p : planes) {
        p.format = ColorFormat::RGB8;
        p.step   = 0;
        p.pitch  = 0;
        p.slice  = 0;
        p.offset = 0;
    }
    auto dst = ManagedImage<Rgb8>(ImageDesc(std::move(planes)));
    toRgb8(src, dst);
    return dst;
}

static bool isNv12(const snn::RawImage& i) { return 2 == i.desc().planes.size() && ColorFormat::R8 == i.format(0) && ColorFormat::RG8 == i.format(1); }

static bool isI420(const snn::RawImage& i) {
    return 3 == i.desc().planes.size() && ColorFormat::R8 == i.format(0) && ColorFormat::R8 == i.format(1) && ColorFormat::R8 == i.format(2);
}

static bool isRgba8(const snn::RawImage& i) { return 1 == i.desc().planes.size() && ColorFormat::RGBA8 == i.format(); }

bool snn::rgba8ToI420(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(src) || !isI420(dst)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::ABGRToI420(src.plane(0), int(src.pitch(0)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)), dst.plane(2), int(dst.pitch(2)),
                       int(width), int(height));
    return true;
}

bool snn::rgba8ToNv12(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(src) || !isNv12(dst)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::ABGRToNV12(src.plane(0), int(src.pitch(0)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)), int(width), int(height));
    return true;
}

bool snn::nv12ToRgba8(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(dst) || !isNv12(src)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::NV12ToABGR(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), dst.plane(0), int(dst.pitch(0)), int(width), int(height));
    return true;
}

bool snn::rgba8ToNv21(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(src) || !isNv12(dst)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::ABGRToNV21(src.plane(0), int(src.pitch(0)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)), int(width), int(height));
    return true;
}

bool snn::nv21ToRgba8(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isRgba8(dst) || !isNv12(src)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = dst.width(), height = dst.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::NV21ToABGR(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), dst.plane(0), int(dst.pitch(0)), int(width), int(height));
    return true;
}

bool snn::nv12ToI420(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isNv12(src) || !isI420(dst)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = src.width(), height = src.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::NV12ToI420(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), dst.plane(0), int(dst.pitch(0)), dst.plane(1), int(dst.pitch(1)),
                       dst.plane(2), int(dst.pitch(2)), int(width), int(height));
    return true;
}

bool snn::i420ToRgba8(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isI420(src) || !isRgba8(dst)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = src.width(), height = src.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::I420ToABGR(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), src.plane(2), int(src.pitch(2)), dst.plane(0), int(dst.pitch(0)),
                       int(width), int(height));
    return true;
}

bool snn::i420ToNv12(const snn::RawImage& src, snn::RawImage& dst) {
    if (!isI420(src) || !isNv12(dst)) {
        SNN_LOGE("invalid src/dst image format");
        return false;
    }
    auto width = src.width(), height = src.height();
    if (src.width() != width || src.height() != height) {
        SNN_LOGE("mismatched src/dst image size");
        return false;
    }
    libyuv::I420ToNV12(src.plane(0), int(src.pitch(0)), src.plane(1), int(src.pitch(1)), src.plane(2), int(src.pitch(2)), dst.plane(0), int(dst.pitch(0)),
                       dst.plane(1), int(dst.pitch(1)), int(width), int(height));
    return true;
}
