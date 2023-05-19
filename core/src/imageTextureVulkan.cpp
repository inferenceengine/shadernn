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
#include "imageTextureVulkan.h"
#include "colorVulkan.h"
#include "vulkanContext.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/benchmark/vulkan_image_util.h"

namespace snn {

ImageTextureVulkan::ImageTextureVulkan(GpuContext* context)
    : ImageTexture(GpuBackendType::VULKAN)
{
    setDevice(context);
}

ImageTextureVulkan::ImageTextureVulkan(GpuContext* context, const std::array<uint32_t, 4>& dims, ColorFormat format,
    const void* buffer, const std::string& name)
    : ImageTexture(GpuBackendType::VULKAN, dims, format, buffer, name)
{
    setDevice(context);
}

ImageTextureVulkan::ImageTextureVulkan(GpuContext* context, const std::string& fileName)
    : ImageTexture(GpuBackendType::VULKAN, fileName)
{
    setDevice(context);
}

ImageTextureVulkan::~ImageTextureVulkan() {
    _vkImages.clear();
    SNN_LOGV("ImageTextureVulkan destroyed");
}

void ImageTextureVulkan::setDevice(GpuContext* context) {
    SNN_ASSERT(context);
    uvkc::benchmark::VulkanContext* ukvcContext = VulkanGpuContext::cast(context)->getUvkcContext();
    _device = (ukvcContext->devices[0].get());
}

void ImageTextureVulkan::attach(const std::vector<std::shared_ptr<uvkc::vulkan::Image>>& vkImages) {
    _backend = Backend::Backend_GPU;
    SNN_ASSERT(vkImages.size() == _dims[3]);

    _vkImages.resize(vkImages.size());
    for (unsigned int i = 0; i < vkImages.size(); i++) {
        _vkImages[i] = vkImages[i];
    }
}

void ImageTextureVulkan::attach(ImageTexture *src) {
    SNN_ASSERT(src->getType() == GpuBackendType::VULKAN);
    ImageTextureVulkan* srcVulkan = static_cast<ImageTextureVulkan*>(src);
    if (srcVulkan->getDims()[3] > 0) {
        SNN_LOGD("dim:%d,%d,%d, %d, vkImage:%p", srcVulkan->getDims()[0], srcVulkan->getDims()[1], srcVulkan->getDims()[2], srcVulkan->getDims()[3],
            srcVulkan->vkImage(0).get());
        _device = srcVulkan->_device;
        _vkImages = srcVulkan->_vkImages;
        _dims = srcVulkan->_dims;
        _format = srcVulkan->_format;
        _backend = Backend::Backend_GPU;
        if (srcVulkan->_vkImages.size() == 0) {
            SNN_LOGW("attaching texture with no images !");
        }
    } else {
        SNN_LOGW("trying to attach empty texture !");
    }
}

void ImageTextureVulkan::attach(const std::vector<const GpuImageHandle*>& images) {
    _backend = Backend::Backend_GPU;
    SNN_ASSERT(images.size() == _dims[3]);

    _vkImages.resize(images.size());
    for (unsigned int i = 0; i < images.size(); i++) {
        SNN_ASSERT(images[i]);
        const VulkanImageHandle& vkImageHandle = VulkanImageHandle::cast(*images[i]);
        _vkImages[i] = std::make_shared<uvkc::vulkan::Image>(_device->getLogicalDevice(), vkImageHandle.image, vkImageHandle.imageView,
            vkImageHandle.imageLayout, _device->getSymbols());
    }
}

std::shared_ptr<uvkc::vulkan::Image> ImageTextureVulkan::vkImage(size_t index) {
    SNN_ASSERT(_vkImages.size() > index);
    _backend = Backend::Backend_GPU;
    return _vkImages[index];
}

void ImageTextureVulkan::resetTexture() {
    _backend = Backend::Backend_GPU;

    uint32_t width  = _dims[0];
    uint32_t height = _dims[1];
    uint32_t depth  = _dims[2];
    uint32_t planes = _dims[3];
    VkExtent3D dimensions = {width, height, depth};

    _vkImages.clear();
    auto vkFormat = getNativeColorVulkan(_format);
    for (unsigned int i = 0; i < planes; i++) {
        SNN_LOGD("idx:%d, dims:%d, %d, %d format:%d", i, width, height, depth, (int)vkFormat);
        BM_CHECK_OK_AND_ASSIGN(
            std::unique_ptr<uvkc::vulkan::Image> vkImageUniq,
            _device->CreateImage(
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_3D,
                vkFormat, dimensions, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_VIEW_TYPE_3D));
            SNN_LOGD("created vkImage: %p", vkImageUniq.get());
        _vkImages.emplace_back(vkImageUniq.release());
    }
}

void ImageTextureVulkan::resetTexture(const std::array<uint32_t, 4>& dims, ColorFormat format, const std::string& name) {
    SNN_LOGD("dims: %d:%d:%d:%d format: %s", dims[0], dims[1], dims[2], dims[3], getColorFormatDesc(format).name);

    _name = name;
    _dims = dims;
    _format = format;

    resetTexture();
}

bool ImageTextureVulkan::resize(float xScale, float yScale, const std::array<float, 4>& means, const std::array<float, 4>& norms,
    bool linearFilter, ColorFormat cf) {
    if (!_device) {
        SNN_RIP("Vulkan device was not assigned to ImageTexture");
    }
    if (_backend != Backend::Backend_GPU) {
        upload();
    }

    if (cf != ColorFormat::NONE) {
        _format = cf;
    }

    uint32_t outputWidth  = static_cast<uint32_t>(round(_dims[0] / xScale));
    uint32_t outputHeight = static_cast<uint32_t>(round(_dims[1] / yScale));

    VkExtent3D destDimensions = {outputWidth, outputHeight, _dims[2]};
    auto vkFormat = getNativeColorVulkan(_format);

    if (!_vulkanImageResizeOp) {
        _vulkanImageResizeOp = std::make_unique<VulkanImageResizeOp>();
        _vulkanImageResizeOp->init(_device, linearFilter);
    }
    _vulkanImageResizeOp->updateParams(destDimensions, means, norms);

    uint32_t planes = _dims[3];
    SNN_ASSERT(_vkImages.size() == planes);
    for (uint32_t i = 0; i < planes; i++) {
        SNN_LOGD("index:%d format: %s w: %d h: %d depth: %d", i, getColorFormatDesc(format(i)).name, width(i), height(i), depth(i));

        // Create destination image
        BM_CHECK_OK_AND_ASSIGN(
            std::unique_ptr<uvkc::vulkan::Image> dstImage,
            _device->CreateImage(
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_3D,
                vkFormat, destDimensions, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_VIEW_TYPE_3D));
        _vulkanImageResizeOp->run(_vkImages[i].get(), dstImage.get());
        // Replace source image by destination
        _vkImages[i].reset(dstImage.release());
    }

    _dims[0] = outputWidth;
    _dims[1] = outputHeight;

    return 0;
}

// From device to host
void ImageTextureVulkan::download() {
    SNN_LOGD("%d:%d:%d:%d", _dims[0], _dims[1], _dims[2], _dims[3]);
    if (!_device) {
        SNN_RIP("Vulkan device was not assigned to ImageTexture");
    }
    _backend = Backend::Backend_CPU;

    uint32_t planes = _dims[3];
    SNN_ASSERT(_vkImages.size() == planes);
    resetImages();
    SNN_ASSERT(_images.planes() == planes);

    VkExtent3D outDimensions = {_dims[0], _dims[1], _dims[2]};
    size_t outputBytes = _dims[0] * _dims[1] * _dims[2] * snn::getColorFormatDesc(_format).bytes();
    for (uint32_t i = 0; i < planes; i++) {
        uvkc::vulkan::Image* vkImage = _vkImages[i].get();
        SNN_ASSERT(vkImage);
        BM_CHECK_OK(uvkc::benchmark::GetDeviceImageViaStagingBuffer(
            _device, vkImage, outDimensions, outputBytes,
            [&](void *ptr, size_t numBytes) {
                memcpy(_images.at(i, 0, 0, 0), ptr, numBytes);
            }));
    }
    SNN_LOGD("%d:%d:%d:%d", _dims[0], _dims[1], _dims[2], _dims[3]);
}

// From host to device
void ImageTextureVulkan::upload() {
    if (!_device) {
        SNN_RIP("Vulkan device was not assigned to ImageTexture");
    }
    _backend = Backend::Backend_GPU;
    auto vkFormat = getNativeColorVulkan(_format);
    _vkImages.clear();
    for (uint32_t i = 0; i < planes(); i++) {
        SNN_LOGD("index:%d format: %s w: %d h: %d depth: %d", i, getColorFormatDesc(format(i)).name, width(i), height(i), depth(i));
        VkExtent3D dimensions = {width(i), height(i), depth(i)};
        BM_CHECK_OK_AND_ASSIGN(
            std::unique_ptr<uvkc::vulkan::Image> vkImageUniq,
            _device->CreateImage(
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_3D,
                vkFormat, dimensions, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_VIEW_TYPE_3D));
        SNN_LOGD("created vkImage: %p", vkImageUniq.get());

        size_t outputBytes = width(i) * height(i) * depth(i) * snn::getColorFormatDesc(_format).bytes();
        SNN_ASSERT(_images.sliceSize(i) * _images.depth(i) == outputBytes);
        BM_CHECK_OK(uvkc::benchmark::SetDeviceImageViaStagingBuffer(
            _device, vkImageUniq.get(), dimensions,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, outputBytes,
            [&](void *ptr, size_t numBytes) {
                memcpy(ptr, _images.at(i, 0, 0, 0), numBytes);
            }));
        _vkImages.emplace_back(vkImageUniq.release());
    }
}

static thread_local char buf[512];

std::string ImageTextureVulkan::getTextureInfo2() const {
    if (_vkImages.empty()) {
        snprintf(buf, sizeof(buf), "tex addr: %p", this);
    }
    else {
        auto& cf = getColorFormatDesc(_format);
        char* printBuf = buf;
        size_t bufSize = sizeof(buf);
        for (size_t i = 0; i < _vkImages.size();) {
            char eol = i + 1 < _vkImages.size() ? '\n' : ' ';
            int printLen = snprintf(printBuf, bufSize, "tex: %p _vkImages[%lu]: %p, VkImage: %p, VkImageView: %p, w: %d, h: %d, d: %d, c: %d, format: %s %c",
                this, i, _vkImages[i].get(), _vkImages[i]->image(), _vkImages[i]->image_view(),
                _dims[0], _dims[1], _dims[2], static_cast<int>(cf.ch),
                cf.name, eol
            );
            if (++i < _vkImages.size() && printLen >= 0 && printLen < bufSize) {
                printBuf += printLen;
                bufSize -= printLen;
            }
            else {
                break;
            }
        }
    }
    return std::string(buf);
}

std::shared_ptr<ImageTexture>* ImageTextureVulkanAllocator::allocate(size_t n) {
    std::shared_ptr<ImageTexture>* ptr = new std::shared_ptr<ImageTexture>[n];
    for (size_t i = 0; i < n; ++i) {
        ImageTextureVulkan* tptr = new ImageTextureVulkan(context);
        ptr[i].reset(tptr);
    }
    return ptr;
}

void ImageTextureVulkanAllocator::deallocate(std::shared_ptr<ImageTexture>* ptr, size_t) {
    delete[] ptr;
}

}
