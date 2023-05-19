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

#include "snn/color.h"

#include <stdint.h>
#include <stddef.h>
#include <glad/glad.h>

// This file contains structures and functions for color description in OpenGL format
namespace snn {

struct NativeColorGL {
    uint32_t glInternalFormat; // internal format to create GL texture.
    uint32_t glFormat;         // data format used to create GL texture.
    uint32_t glType;           // data type used to create GL texture.
};

struct ColorFormatDescGL : public ColorFormatDesc, public NativeColorGL {
    ColorFormatDescGL(const ColorFormatDesc& cf, const NativeColorGL& nc)
        : ColorFormatDesc(cf)
        , NativeColorGL(nc)
    {}
};

constexpr NativeColorGL NATIVE_COLOR_GL_TABLE[size_t(ColorFormat::NUM_COLOR_FORMATS)] = {
    {
        GL_NONE,
        GL_NONE,
        GL_NONE,
    }, // NONE
    {
        GL_RGBA32F,
        GL_RGBA,
        GL_FLOAT,
    }, // RGBA32F
    {
        GL_RGB32F,
        GL_RGB,
        GL_FLOAT,
    }, // RGB32F
    {
        GL_RGBA16F,
        GL_RGBA,
        GL_HALF_FLOAT,
    }, // RGBA16F
    {
        GL_RGBA16UI,
        GL_RGBA,
        GL_UNSIGNED_SHORT,
    }, // RGBA16UI
    {
        GL_RGB16F,
        GL_RGB,
        GL_FLOAT,
    }, // RGB16F
    {
        GL_R32F,
        GL_RED,
        GL_FLOAT,
    }, // R32F
    {
        GL_RGBA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
    }, // RGBA8
    {
        GL_SRGB8_ALPHA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
    }, // SRGB8_A8
    {
        GL_RGB8,
        GL_RGB,
        GL_UNSIGNED_BYTE,
    }, // RGB8
    {
        GL_SRGB8,
        GL_RGB,
        GL_UNSIGNED_BYTE,
    }, // SRGB8
    {
        GL_R16F,
        GL_RED,
        GL_FLOAT,
    }, // R16F
    {
        GL_RG8,
        GL_RED,
        GL_UNSIGNED_BYTE,
    }, // RG8
    {
        GL_R8,
        GL_RED,
        GL_UNSIGNED_BYTE,
    }, // R8
    {
        GL_NONE,
        GL_NONE,
        GL_NONE,
    }, // NV12 (dual plan image)
    {
        GL_NONE,
        GL_NONE,
        GL_NONE,
    }, // NV21 (dual plan image)
};

inline ColorFormatDescGL getColorFormatDescGL(ColorFormat cf) {
    return ColorFormatDescGL(COLOR_FORMAT_DESC_TABLE[(size_t) cf], NATIVE_COLOR_GL_TABLE[(size_t) cf]);
}

inline constexpr NativeColorGL const& getNativeColorGL(ColorFormat cf) {
    return NATIVE_COLOR_GL_TABLE[(size_t) cf];
}

inline constexpr ColorFormat fromGLInternalFormat(uint32_t glInternalFormat) {
    for (size_t i = 1; i < std::size(NATIVE_COLOR_GL_TABLE); ++i) {
        if (NATIVE_COLOR_GL_TABLE[i].glInternalFormat == glInternalFormat) {
            return (ColorFormat) i;
        }
    }
    return ColorFormat::NONE;
}

}
