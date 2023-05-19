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
#include <stdint.h>
#include <stddef.h>
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
    size_t bits;               // bits per pixel
    size_t ch;                 // number of channels

    size_t bytes() const { return bits / 8U; }

    size_t colorDepth() const { return bits / 8U / ch; }

    size_t calcImageSizeInBytes(size_t w, size_t h) const { return w * h * bytes(); }
};

constexpr ColorFormatDesc COLOR_FORMAT_DESC_TABLE[size_t(ColorFormat::NUM_COLOR_FORMATS)] = {
    {
        "NONE",
        0 * 8,
        0,
    }, // NONE
    {
        "RGBA32F",
        16 * 8,
        4,
    }, // RGBA32F
    {
        "RGB32F",
        16 * 8,
        3,
    }, // RGBA32F
    {
        "RGBA16F",
        8 * 8,
        4,
    }, // RGBA16F
    {
        "RGBA16U",
        8 * 8,
        4,
    }, // RGBA16UI
    {
        "RGB16F",
        6 * 8,
        3,
    }, // RGB16F
    {
        "R32F",
        4 * 8,
        1,
    }, // R32F
    {
        "RGBA8",
        4 * 8,
        4,
    }, // RGBA8
    {
        "SRGB8_A8",
        4 * 8,
        4,
    }, // SRGB8_A8
    {
        "RGB8",
        3 * 8,
        3,
    }, // RGB8
    {
        "SRGB8",
        3 * 8,
        3,
    }, // SRGB8
    {
        "R16F",
        2 * 8,
        1,
    }, // R16F
    {
        "RG8",
        2 * 8,
        2,
    }, // RG8
    {
        "R8",
        1 * 8,
        1,
    }, // R8
    {
        "NV12",
        12,
        3,
    }, // NV12 (dual plan image)
    {
        "NV21",
        12,
        3,
    }, // NV21 (dual plan image)
};

// Gets color format description from color format
inline constexpr ColorFormatDesc const& getColorFormatDesc(ColorFormat cf) {
    static_assert(std::size(COLOR_FORMAT_DESC_TABLE) == (size_t) ColorFormat::NUM_COLOR_FORMATS);
    return (0 <= (int) cf && cf < ColorFormat::NUM_COLOR_FORMATS) ? COLOR_FORMAT_DESC_TABLE[(int) cf] : COLOR_FORMAT_DESC_TABLE[0];
}

} // namespace snn
