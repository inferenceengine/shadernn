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
#include "uvkc/vulkan/dynamic_symbols.h"
#include "snn/vulkanImageAccessor.h"
#include "imageTextureVulkan.h"
#include "colorVulkan.h"

namespace snn {

bool getVkImage(ImageTexture& texture, VulkanImageHandle& imageHandle) {
    if (texture.getType() != GpuBackendType::VULKAN) {
        SNN_RIP("Image texture does not have Vulkan backend!");
    }
    ImageTextureVulkan& textureVulkan = static_cast<ImageTextureVulkan&>(texture);
    if (!textureVulkan.isValid()) {
        return false;
    }
    imageHandle.image = textureVulkan.vkImage(0)->image();
    imageHandle.imageView = textureVulkan.vkImage(0)->image_view();
    imageHandle.imageLayout = textureVulkan.vkImage(0)->image_layout();

    return true;
}

void setVkImage(ImageTexture& texture, const VulkanImageHandle& imageHandle) {
    if (texture.getType() != GpuBackendType::VULKAN) {
        SNN_RIP("Image texture does not have Vulkan backend!");
    }
    ImageTextureVulkan& textureVulkan = static_cast<ImageTextureVulkan&>(texture);
    textureVulkan.attach({&imageHandle});
}

}
