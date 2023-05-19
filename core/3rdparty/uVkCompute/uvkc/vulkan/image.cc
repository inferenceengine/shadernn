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

#include "uvkc/vulkan/image.h"

#include "absl/status/status.h"
#include "uvkc/vulkan/status_util.h"
#include "uvkc/base/log.h"

namespace uvkc {
namespace vulkan {

Image::Image(VkDevice device, VkDeviceMemory memory, VkImage image,
             VkImageView image_view, const DynamicSymbols &symbols)
    : image_(image)
    , image_view_(image_view)
    , image_layout_(VK_IMAGE_LAYOUT_UNDEFINED)
    , device_(device)
    , memory_(memory)
    , symbols_(symbols) {
}

Image::Image(VkDevice device, VkImage image,
             VkImageView image_view, VkImageLayout image_layout, const DynamicSymbols &symbols)
    : image_(image)
    , image_view_(image_view)
    , image_layout_(image_layout)
    , device_(device)
    , memory_(VK_NULL_HANDLE)
    , symbols_(symbols)
    , externallyOwned_(true) {
}

Image::Image(Image&& other)
  : image_view_(other.image_view_)
  , image_(other.image_)
  , image_layout_(other.image_layout_)
  , memory_(other.memory_)
  , symbols_(other.symbols_)
  , externallyOwned_(other.externallyOwned_)
{
  other.image_view_ = VK_NULL_HANDLE;
  other.image_ = VK_NULL_HANDLE;
  other.image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  other.memory_ = VK_NULL_HANDLE;
  other.externallyOwned_ = false;
}

Image& Image::operator = (Image&& other) {
  if (this != &other) {
    if (!externallyOwned_) {
      if (image_view_) {
        symbols_.vkDestroyImageView(device_, image_view_, /*pAllocator=*/nullptr);
      }
      if (image_) {
        symbols_.vkDestroyImage(device_, image_, /*pAllocator=*/nullptr);
      }
      if (memory_) {
        symbols_.vkFreeMemory(device_, memory_, /*pAllocator=*/nullptr);
      }
    }

    image_view_ = other.image_view_;
    image_ = other.image_;
    memory_ = other.memory_;
    externallyOwned_ = other.externallyOwned_;

    other.image_view_ = VK_NULL_HANDLE;
    other.image_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.externallyOwned_ = false;
  }

  return *this;
}

Image::~Image() {
  if (!externallyOwned_) {
    if (image_view_) {
      symbols_.vkDestroyImageView(device_, image_view_, /*pAllocator=*/nullptr);
    }
    if (image_) {
      symbols_.vkDestroyImage(device_, image_, /*pAllocator=*/nullptr);
    }
    if (memory_) {
      symbols_.vkFreeMemory(device_, memory_, /*pAllocator=*/nullptr);
    }
    UVKC_LOGV("destroyed VkImage: %p, VkImageView: %p", image_, image_view_);
  }
}

VkImage Image::image() const { return image_; }

VkImageView Image::image_view() const { return image_view_; }

VkImageLayout Image::image_layout() const { return image_layout_; }

void Image::SetImageLayout(VkImageLayout image_layout) {
  image_layout_ = image_layout;
}

Sampler::Sampler(VkDevice device, VkSampler sampler,
                 const DynamicSymbols &symbols)
    : sampler_(sampler), device_(device), symbols_(symbols) {}

Sampler::~Sampler() {
  symbols_.vkDestroySampler(device_, sampler_, /*pAllocator=*/nullptr);
}

VkSampler Sampler::sampler() const { return sampler_; }

}  // namespace vulkan
}  // namespace uvkc
