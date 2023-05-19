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

#include "volk.h"
#include "vulkanFrameViz.h"
#include "vulkan/vulkanAppContext.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "snn/vulkanImageHandle.h"
#include "vulkan/vulkanLib.h"

namespace snn {

VulkanNightVisionFrameViz::VulkanNightVisionFrameViz()
    : context(VulkanAppContext::getVulkanContext())
{
    SNN_ASSERT(context->getDevice());

    textureSampler = createSampler(context->getDevice());
}

VulkanNightVisionFrameViz::~VulkanNightVisionFrameViz() {
    if (textureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->getDevice(), textureSampler, nullptr);
        textureSampler = VK_NULL_HANDLE;
    }
}

void VulkanNightVisionFrameViz::render(const NightVisionFrameViz::RenderParameters & rp) {
    VulkanImageHandle vulkanImageHandle;
    rp.frame->getGpuImageHandle(vulkanImageHandle);
    if (vulkanImageHandle.empty()) {
        return;
    }

    SNN_ASSERT(textureSampler);
    SNN_ASSERT(vulkanImageHandle.imageView);
    if (!samplerUpdated) {
        // Update sampler only at the 1st rendering
        // NightVisionFrameViz::render() called at the very end of the pipeline
        // and the incoming input image is the one that will be sampled on a screen surface.
        // So, here we need to update app's descriptots with it, but only once per concrete model (or empty model) run.
        VulkanAppContext::updateOutputSampler(vulkanImageHandle.imageView, textureSampler);
        SNN_LOGV("VkImage: %p, VkImageView: %p", vulkanImageHandle.image, vulkanImageHandle.imageView);
        samplerUpdated = true;
    }
}

}
