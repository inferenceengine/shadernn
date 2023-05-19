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
#include <vulkan/vulkan.h>

// This file contains structures and functions for color description in Vulkan format
namespace snn {

struct ColorFormatDescVulkan : public ColorFormatDesc {
    ColorFormatDescVulkan(const ColorFormatDesc& cf, VkFormat vf_)
        : ColorFormatDesc(cf)
        , vf(vf_)
    {}

    VkFormat vf;
};

constexpr VkFormat NATIVE_COLOR_VULKAN_TABLE[size_t(ColorFormat::NUM_COLOR_FORMATS)] = {
    VK_FORMAT_UNDEFINED,            // NONE
    VK_FORMAT_R32G32B32A32_SFLOAT,  // RGBA32F
    VK_FORMAT_R32G32B32_SFLOAT,     // RGB32F
    VK_FORMAT_R16G16B16A16_SFLOAT,  // RGBA16F
    VK_FORMAT_R16G16B16A16_UNORM,   // RGBA16UI
    VK_FORMAT_R16G16B16A16_SFLOAT,  // RGB16F
    VK_FORMAT_R32_SFLOAT,           // R32F
    VK_FORMAT_R8G8B8A8_UNORM,       // RGBA8
    VK_FORMAT_R8G8B8A8_SRGB,        // SRGB8_A8
    VK_FORMAT_R8G8B8_UNORM,         // RGB8
    VK_FORMAT_R8G8B8_SRGB,          // SRGB8
    VK_FORMAT_R16_SFLOAT,           // R16F
    VK_FORMAT_R8G8_UNORM,           // RG8
    VK_FORMAT_R8_UNORM,             // R8
    VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,    // NV12 (dual plan image)
    VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,    // NV21 (dual plan image)
};

inline ColorFormatDescVulkan getColorFormatDescVulkan(ColorFormat cf) {
    return ColorFormatDescVulkan(COLOR_FORMAT_DESC_TABLE[(size_t) cf], NATIVE_COLOR_VULKAN_TABLE[(size_t) cf]);
}

inline constexpr VkFormat const& getNativeColorVulkan(ColorFormat cf) {
    return NATIVE_COLOR_VULKAN_TABLE[(size_t) cf];
}

inline constexpr ColorFormat fromVulkanInternalFormat(VkFormat vf) {
    for (size_t i = 1; i < std::size(NATIVE_COLOR_VULKAN_TABLE); ++i) {
        if (NATIVE_COLOR_VULKAN_TABLE[i] == vf) {
            return (ColorFormat) i;
        }
    }
    return ColorFormat::NONE;
}

}
