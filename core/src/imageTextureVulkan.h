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
#include "snn/imageTexture.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "snn/vulkanImageHandle.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/benchmark/status_util.h"
#include "vulkanImageResizeOp.h"

#include <memory>

namespace snn {

// This class objects holds Vulkan images and provides access to them either from GPU or from CPU
class ImageTextureVulkan : public ImageTexture {
public:
    // Constructor
    // params:
    //  context - GPU context
    ImageTextureVulkan(GpuContext* context);

    // Constructor from a pixel buffer
    // params:
    //  context - GPU context
    //  dims - dimensions
    //  format - color format
    //  buffer - image pixel buffer
    //  name - optional image name
    ImageTextureVulkan(GpuContext* context, const std::array<uint32_t, 4>& dims, ColorFormat format, const void* buffer = NULL, const std::string& name = "");

    // Constructor from an image file
    // params:
    //  context - GPU context
    //  fileName - image file name
    ImageTextureVulkan(GpuContext* context, const std::string& fileName);

    // Destructor
    virtual ~ImageTextureVulkan();

    // Casts an ImageTexture reference  to ImageTextureVulkan reference
    // params:
    //  src - source reference to ImageTexture
    // returns:
    //  target reference to ImageTextureVulkan
    static ImageTextureVulkan& cast(ImageTexture& src) {
        SNN_ASSERT(src.getType() == GpuBackendType::VULKAN);
        return static_cast<ImageTextureVulkan&>(src);
    }

    // Gets underlying object of uvkc::vulkan::Image class
    // params:
    //  index - index of an image to return
    // return:
    //  Sharep pointer to an object of uvkc::vulkan::Image class
    std::shared_ptr<uvkc::vulkan::Image> vkImage(size_t index = 0);

    // Attaches other Vulkan images
    // params:
    //  vkImages - Vulkan images to attach
    void attach(const std::vector<std::shared_ptr<uvkc::vulkan::Image>>& vkImages);

    // Attaches a content of another ImageTexture object
    // params:
    //  src - source object
    virtual void attach(ImageTexture *src) override;

    // Attaches another Vulkan images
    // params:
    //  images - Vulkan images to attach
    virtual void attach(const std::vector<const GpuImageHandle*>& images) override;

    virtual bool isValid() const override {
        return _vkImages.size() > 0;
    }

    // Allocates empty Vulkan image on GPU with the same dimensions and color format
    virtual void resetTexture() override;

    // Resizes the texture on GPU and optionally normalizes it.
    // params:
    //  xScale - horizontal scale factor
    //  yScale - vertical scale factor
    //  means - mean values for 4 channels. Used for normalization.
    //  norms - normalization values for 4 channels (multipliers). Used for normalization.
    //  linearFilter - true if using linear filter. false if using nearest-neighbor filter.
    //  cf - optional color format to convert resized image
    // return:
    // true if resizing was successful; false if not.
    virtual bool resize(float xScale, float yScale, const std::array<float, 4>& means, const std::array<float, 4>& norms, bool linearFilter = true,
        ColorFormat cf = ColorFormat::NONE) override;

    // Downloads textures from device to host
    virtual void download() override;

    // Uploads textures from host to device
    virtual void upload() override;

    // Returns detailed texture information in human-readable format. Used for debugging.
    virtual std::string getTextureInfo2() const override;

    // Allocates empty GPU texture
    // params:
    //  dims - dimensions
    //  format - color format
    //  name - optional image name
    virtual void resetTexture(const std::array<uint32_t, 4>& dims, ColorFormat format, const std::string& name = "") override;

private:
    // Sets the Vulkan Device object
    // params:
    //  context - pointer to GPU context
    void setDevice(GpuContext* context);

    // Vulkan Device object
    uvkc::vulkan::Device* _device = nullptr;

    // Array of Vulkan objects
    std::vector<std::shared_ptr<uvkc::vulkan::Image>> _vkImages;

    // Vulkan resize op.
    std::unique_ptr<VulkanImageResizeOp> _vulkanImageResizeOp;
};

// This structure holds GPU context
// and it used to allocate an array of ImageTextureVulkan shared pointers
struct ImageTextureVulkanAllocator {
    // Constructor
    // params:
    //  context_ - pointer to GPU context
    ImageTextureVulkanAllocator(GpuContext* context_)
        : context(context_)
    {
        SNN_ASSERT(context_);
    }

    // Allocate an an array of ImageTextureVulkan shared pointers
    std::shared_ptr<ImageTexture>* allocate(size_t n);

    // Deallocate an an array of ImageTextureVulkan shared pointers
    void deallocate(std::shared_ptr<ImageTexture>* ptr, size_t);

    // pointer to GPU context
    GpuContext* context;
};

typedef ImageTextureTypeCheck<GpuBackendType::VULKAN> ImageTextureVulkanTypeCheck;

typedef PolyArrayAccessor<ImageTextureVulkan, ImageTexture, ImageTextureVulkanTypeCheck> ImageTextureVulkanArrayAccessor;

typedef PolyArray<ImageTextureVulkan, ImageTexture, ImageTextureVulkanTypeCheck, ImageTextureVulkanAllocator> ImageTextureVulkanArray;

} // namespace snn
