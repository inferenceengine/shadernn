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
#include <string>
#include <set>
#include <map>

namespace snn {

// Helper function
// Gets color format from color format name
ColorFormat fromName(const char* name);

// Helper function
// Retrieves all color format names
std::set<std::string> getAllColorNames();

enum class ColorFormatType {
    NONE,
    UINT8,
    UINT16,
    FLOAT16,
    FLOAT32,
};

static const std::map<ColorFormat, ColorFormatType> ColorFormatTypesTable = {
    {ColorFormat::NONE, ColorFormatType::NONE},
    {ColorFormat::RGBA32F, ColorFormatType::FLOAT32},
    {ColorFormat::RGB32F, ColorFormatType::FLOAT32},
    {ColorFormat::RGBA16F, ColorFormatType::FLOAT16},
    {ColorFormat::RGBA16U, ColorFormatType::UINT16},
    {ColorFormat::RGB16F, ColorFormatType::FLOAT32},
    {ColorFormat::R32F, ColorFormatType::FLOAT32},
    {ColorFormat::RGBA8, ColorFormatType::UINT8},
    {ColorFormat::SRGB8_A8, ColorFormatType::UINT8},
    {ColorFormat::RGB8, ColorFormatType::UINT8},
    {ColorFormat::SRGB8, ColorFormatType::UINT8},
    {ColorFormat::R16F, ColorFormatType::FLOAT32},
    {ColorFormat::RG8, ColorFormatType::UINT8},
    {ColorFormat::R8, ColorFormatType::UINT8},
    {ColorFormat::NV12, ColorFormatType::NONE},
    {ColorFormat::NV21, ColorFormatType::NONE},
    {ColorFormat::NUM_COLOR_FORMATS, ColorFormatType::NONE},
};

ColorFormatType getColorFormatType(ColorFormat cf);

// Converts image buffer from float format to SNN internal format-dependent binary format
std::vector<uint8_t> convertColorBuffer(ColorFormat cf, const float* buf, size_t len);

}
