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
#include <stdint.h>
#include <stddef.h>
#include <KHR/khrplatform.h>
#include <glad/glad.h>
#include <iterator>

namespace snn {

enum class ColorFormat : uint32_t {
    NONE,
    RGBA32F,
    RGB32F,
    RGBA16F,
    RGBA16U,
    RGB16F,
    R32F,
    RGBA8,
    SRGB8_A8,
    RGB8,
    SRGB8,
    R16F,
    RG8,
    R8,
    NV12, // It is a dual-plane image format with a full-res Y plane and half-res interleaved U/V plane.
    NV21, // It is a dual-plane image format with a full-res Y plane and half-res interleaved V/U plane.
    NUM_COLOR_FORMATS,
};

struct ColorFormatDesc {
    const char* name;
    uint32_t glInternalFormat; // internal format to create GL texture.
    uint32_t glFormat;         // data format used to create GL texture.
    uint32_t glType;           // data type used to create GL texture.
    size_t bits;               // bits per pixel
    size_t ch;                 // number of channels

    size_t calcImageSizeInBytes(size_t w, size_t h) const { return w * h * bits / 8; }
};

constexpr ColorFormatDesc COLOR_FORMAT_DESC_TABLE[] = {
    {
        "NONE",
        GL_NONE,
        GL_NONE,
        GL_NONE,
        0 * 8,
        0,
    }, // NONE
    {
        "RGBA32F",
        GL_RGBA32F,
        GL_RGBA,
        GL_FLOAT,
        16 * 8,
        4,
    }, // RGBA32F
    {
        "RGB32F",
        GL_RGB32F,
        GL_RGB,
        GL_FLOAT,
        16 * 8,
        3,
    }, // RGBA32F
    {
        "RGBA16F",
        GL_RGBA16F,
        GL_RGBA,
        GL_HALF_FLOAT,
        16 * 4,
        4,
    }, // RGBA16F
    {
        "RGBA16U",
        GL_RGBA16UI,
        GL_RGBA,
        GL_UNSIGNED_SHORT,
        8 * 8,
        4,
    }, // RGBA16UI
    {
        "RGB16F",
        GL_RGB16F,
        GL_RGB,
        GL_FLOAT,
        6 * 8,
        3,
    }, // RGB16F
    {
        "R32F",
        GL_R32F,
        GL_RED,
        GL_FLOAT,
        4 * 8,
        1,
    }, // R32F
    {
        "RGBA8",
        GL_RGBA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        4 * 8,
        4,
    }, // RGBA8
    {
        "SRGB8_A8",
        GL_SRGB8_ALPHA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        4 * 8,
        4,
    }, // SRGB8_A8
    {
        "RGB8",
        GL_RGB8,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        3 * 8,
        3,
    }, // RGB8
    {
        "SRGB8",
        GL_SRGB8,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        3 * 8,
        3,
    }, // SRGB8
    {
        "R16F",
        GL_R16F,
        GL_RED,
        GL_FLOAT,
        2 * 8,
        1,
    }, // R16F
    {
        "RG8",
        GL_RG8,
        GL_RED,
        GL_UNSIGNED_BYTE,
        2 * 8,
        2,
    }, // RG8
    {
        "R8",
        GL_R8,
        GL_RED,
        GL_UNSIGNED_BYTE,
        1 * 8,
        1,
    }, // R8
    {
        "NV12",
        GL_NONE,
        GL_NONE,
        GL_NONE,
        12,
        3,
    }, // NV12 (dual plan image)
    {
        "NV21",
        GL_NONE,
        GL_NONE,
        GL_NONE,
        12,
        3,
    }, // NV21 (dual plan image)
};

inline constexpr ColorFormatDesc const& getColorFormatDesc(snn::ColorFormat cf) {
    static_assert(std::size(COLOR_FORMAT_DESC_TABLE) == (size_t) snn::ColorFormat::NUM_COLOR_FORMATS);
    return (0 <= (int) cf && cf < ColorFormat::NUM_COLOR_FORMATS) ? COLOR_FORMAT_DESC_TABLE[(int) cf] : COLOR_FORMAT_DESC_TABLE[0];
}

inline constexpr ColorFormat fromGLInternalFormat(uint32_t glInternalFormat) {
    for (size_t i = 1; i < std::size(COLOR_FORMAT_DESC_TABLE); ++i) {
        if (COLOR_FORMAT_DESC_TABLE[i].glInternalFormat == glInternalFormat) {
            return (ColorFormat) i;
        }
    }
    return ColorFormat::NONE;
}
} // namespace snn
