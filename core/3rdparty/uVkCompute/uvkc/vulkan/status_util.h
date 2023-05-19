// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UVKC_VULKAN_STATUS_UTIL_H_
#define UVKC_VULKAN_STATUS_UTIL_H_

#include <vulkan/vulkan.h>

#include "uvkc/base/status.h"

namespace uvkc {
namespace vulkan {

// Converts a VkResult to an absl::Status.
absl::Status VkResultToStatus(VkResult result);

// Executes an expression `rexpr` that returns a `VkResult`. On error, returns
// from the current function.
#define VK_RETURN_IF_ERROR(rexpr) UVKC_RETURN_IF_ERROR(VkResultToStatus(rexpr))

static constexpr const char* VkImageLayoutNames[] = {
    "VK_IMAGE_LAYOUT_UNDEFINED",
    "VK_IMAGE_LAYOUT_GENERAL",
    "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL",
    "VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL",
    "VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL",
    "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL",
    "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL",
    "VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL"
};

inline const char* GetVkImageLayoutName(VkImageLayout layout) {
    return (size_t)layout < sizeof(VkImageLayoutNames) ? VkImageLayoutNames[(size_t)layout] : "other";
}

static constexpr const char* VkDescriptorTypeNames[] = {
    "VK_DESCRIPTOR_TYPE_SAMPLER",
    "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE",
    "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    "VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER",
    "VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER",
    "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
    "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC",
    "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC",
    "VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT"
};

inline const char* GetVkDescriptorTypeName(VkDescriptorType type) {
    return (size_t)type < sizeof(VkDescriptorTypeNames) ? VkDescriptorTypeNames[(size_t)type] : "other";
}

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_STATUS_UTIL_H_
