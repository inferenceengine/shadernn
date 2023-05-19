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
#include "bBoxes.h"
#include "vulkan/vulkanLib.h"
#include "vulkan/error.h"
#include "vulkan/helpers.h"
#include "snn/utils.h"
#include <cstring>
#include <tuple>
#include <array>

namespace snn {

void VulkanBBoxes::init(VkDevice device_, const VkPhysicalDeviceMemoryProperties& deviceMemoryProperties_, VkCommandPool commandPool_, VkQueue queue_,
                        VkFormat colorFormat_, const VkExtent2D& surfaceSize_)
{
    SNN_ASSERT(device_);
    SNN_ASSERT(commandPool_);
    SNN_ASSERT(queue_);

    context.device = device_;
    context.deviceMemoryProperties = deviceMemoryProperties_;
    context.commandPool = commandPool_;
    context.queue = queue_;

    colorFormat = colorFormat_;
    surfaceSize = surfaceSize_;

    initRenderPass();
    initPipeline();
}

VulkanBBoxes::~VulkanBBoxes()
{
    cleanupTemporaries();

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    }
    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(context.device, renderPass, nullptr);
    }
}

void VulkanBBoxes::cleanupTemporaries()
{
    if (frameBuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(context.device, frameBuffer, nullptr);
        frameBuffer = VK_NULL_HANDLE;
    }
    if (vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context.device, vertexBuffer, nullptr);
        vertexBuffer = VK_NULL_HANDLE;
    }
    if (vertexBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(context.device, vertexBufferMemory, nullptr);
        vertexBufferMemory = VK_NULL_HANDLE;
    }
    if (vertexBufferMemory != VK_NULL_HANDLE) {
        vkDestroyBuffer(context.device, indexBuffer, nullptr);
        indexBuffer = VK_NULL_HANDLE;
    }
    if (indexBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(context.device, indexBufferMemory, nullptr);
        indexBufferMemory = VK_NULL_HANDLE;
    }
}

std::vector<VulkanBBoxes::Point2D> VulkanBBoxes::createVertices(std::vector<float>& coords) {
    SNN_ASSERT(coords.size() % 4 == 0);
    size_t numBoxes = coords.size() / 4;
    std::vector<Point2D> verticesRects;
    for (size_t i = 0, j = 0; i < numBoxes; ++i, j += 4) {
        verticesRects.push_back({coords[j + 0], coords[j + 1]}); //TopLeft
        verticesRects.push_back({coords[j + 2], coords[j + 1]}); //TopRight
        verticesRects.push_back({coords[j + 2], coords[j + 3]}); //BottomRight
        verticesRects.push_back({coords[j + 0], coords[j + 3]}); //BottomLeft
    }
    return verticesRects;
}

void VulkanBBoxes::initVertexBuffer(VkBuffer& vertexBuffer, VkDeviceMemory& vertexBufferMemory, const std::vector<Point2D>& verticesRects)
{
    SNN_ASSERT(!verticesRects.empty());
    VkDeviceSize sizeInBytes = sizeof(verticesRects[0]) * verticesRects.size();

    std::tie(vertexBuffer, vertexBufferMemory) = createBuffer(context.device, context.deviceMemoryProperties,
                                                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                              sizeInBytes);

    void* data;
    vkMapMemory(context.device, vertexBufferMemory, 0, sizeInBytes, 0, &data);
    std::memcpy(data, verticesRects.data(), (size_t)sizeInBytes);
    vkUnmapMemory(context.device, vertexBufferMemory);
}

std::vector<uint32_t> VulkanBBoxes::createIndicies(size_t numBoxes) {
    std::vector<uint32_t> indiciesRects;
    for (size_t i = 0, j = 0; i < numBoxes; ++i, j += 4) {
        // Closed polygon around 4 rectangles
        indiciesRects.push_back(j);
        indiciesRects.push_back(j + 1);
        indiciesRects.push_back(j + 2);
        indiciesRects.push_back(j + 3);
        indiciesRects.push_back(j);
        if (i + 1 < numBoxes) {
            // Start new polygon
            indiciesRects.push_back(0xFFFFFFFF);
        }
    }
    return indiciesRects;
}

void VulkanBBoxes::initIndexBuffer(VkBuffer& indexBuffer, VkDeviceMemory& indexBufferMemory, const std::vector<uint32_t>& indiciesRects) {
    SNN_ASSERT(!indiciesRects.empty());
    VkDeviceSize sizeInBytes = sizeof(indiciesRects[0]) * indiciesRects.size();

    std::tie(indexBuffer, indexBufferMemory) = createBuffer(context.device, context.deviceMemoryProperties,
                                                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                            sizeInBytes);

    void* data;
    vkMapMemory(context.device, indexBufferMemory, 0, sizeInBytes, 0, &data);
    std::memcpy(data, indiciesRects.data(), (size_t)sizeInBytes);
    vkUnmapMemory(context.device, indexBufferMemory);
}

void VulkanBBoxes::initFramebuffer(VkFramebuffer& frameBuffer, VkImageView colorAttachment)
{
    VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.renderPass      = renderPass;
    fbInfo.attachmentCount = 1;
    fbInfo.pAttachments    = &colorAttachment;
    fbInfo.width           = surfaceSize.width;
    fbInfo.height          = surfaceSize.height;
    fbInfo.layers          = 1;

    VK_CHECK(vkCreateFramebuffer(context.device, &fbInfo, nullptr, &frameBuffer));
}

void VulkanBBoxes::initRenderPass()
{
    VkAttachmentDescription attachment = {0};
    // Backbuffer format.
    attachment.format = colorFormat;
    // Not multisampled.
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;

    // When starting the frame, we want all content to be preserved
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

    // When ending the frame, we want tiles to be written out.
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // Don't care about stencil since we're not using it.
    attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    attachment.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {0};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &color_ref;

    // Use subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Finally, create the renderpass.
    VkRenderPassCreateInfo rp_info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp_info.attachmentCount        = 1;
    rp_info.pAttachments           = &attachment;
    rp_info.subpassCount           = 1;
    rp_info.pSubpasses             = &subpass;
    rp_info.dependencyCount        = static_cast<uint32_t>(dependencies.size());
    rp_info.pDependencies          = dependencies.data();

    VK_CHECK(vkCreateRenderPass(context.device, &rp_info, nullptr, &renderPass));
}

void VulkanBBoxes::initPipeline()
{
    // Create a blank pipeline layout.
    // We are not binding any resources to the pipeline
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VK_CHECK(vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipelineLayout));

    VkVertexInputBindingDescription vertexRectInputBindings[] = {
            {0 /*binding*/, sizeof(Point2D) /*stride*/, VK_VERTEX_INPUT_RATE_VERTEX }
    };

    VkVertexInputAttributeDescription vertexRectAttributes[] = {
            {0 /*location*/, 0 /*binding*/, VK_FORMAT_R32G32_SFLOAT /*2D coortinates*/, 0 /*offset*/}
    };

    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = vertexRectInputBindings;
    vertex_input.vertexAttributeDescriptionCount = 1;
    vertex_input.pVertexAttributeDescriptions = vertexRectAttributes;

    // Specify we will use triangle lists to draw geometry.
    VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    // Specify we will use line lists to draw geometry.
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    /// Will be changed when drawing multiple rectangles
    input_assembly.primitiveRestartEnable = VK_TRUE;

    // Specify rasterization state.
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.cullMode  = VK_CULL_MODE_NONE;
    raster.polygonMode = VK_POLYGON_MODE_LINE;
#if THICK_LINES
    raster.lineWidth = 3.0f;
#else
    raster.lineWidth = 1.0f;
#endif

    // Our attachment will write to all color channels, but no blending is enabled.
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_attachment;

    // We will have one viewport and scissor box.
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount  = 1;

    // Disable all depth testing.
    VkPipelineDepthStencilStateCreateInfo depth_stencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    // No multisampling.
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Specify that these states will be dynamic, i.e. not part of pipeline state object.
    std::vector<VkDynamicState> dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
#if THICK_LINES
    // Thick lines work fine, despite one validation error
    // Enabling VK_DYNAMIC_STATE_LINE_WIDTH leads to more validation errors
#if 0
    dynamics.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);
#endif
#endif

    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.pDynamicStates    = dynamics.data();
    dynamic.dynamicStateCount = vkb::to_u32(dynamics.size());

    // Load our SPIR-V shaders.
    std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{};

    // Vertex stage of the pipeline
    shader_stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = loadShaderSpirvModule(context.device, "shaders/point2D.vert.spv");
    shader_stages[0].pName  = "main";

    // Fragment stage of the pipeline
    shader_stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = loadShaderSpirvModule(context.device, "shaders/point2D.frag.spv");
    shader_stages[1].pName  = "main";

    VkGraphicsPipelineCreateInfo pipe{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipe.stageCount          = vkb::to_u32(shader_stages.size());
    pipe.pStages             = shader_stages.data();
    pipe.pVertexInputState   = &vertex_input;
    pipe.pInputAssemblyState = &input_assembly;
    pipe.pRasterizationState = &raster;
    pipe.pColorBlendState    = &blend;
    pipe.pMultisampleState   = &multisample;
    pipe.pViewportState      = &viewport;
    pipe.pDepthStencilState  = &depth_stencil;
    pipe.pDynamicState       = &dynamic;

    // We need to specify the pipeline layout and the render pass description up front as well.
    pipe.renderPass = renderPass;
    pipe.layout     = pipelineLayout;

    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipe, nullptr, &pipeline));

    // Pipeline is baked, we can delete the shader modules now.
    vkDestroyShaderModule(context.device, shader_stages[0].module, nullptr);
    vkDestroyShaderModule(context.device, shader_stages[1].module, nullptr);
}

void VulkanBBoxes::render(VkCommandBuffer cmdBuffer, VkImageView colorAttachment, std::vector<float>& coords)
{
    std::vector<Point2D> verticesRects = createVertices(coords);
    std::vector<uint32_t> indiciesRects = createIndicies(coords.size() / 4);

    cleanupTemporaries();

    initVertexBuffer(vertexBuffer, vertexBufferMemory, verticesRects);
    initIndexBuffer(indexBuffer, indexBufferMemory, indiciesRects);
    initFramebuffer(frameBuffer, colorAttachment);

    // Begin the render pass.
    VkRenderPassBeginInfo rpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBegin.renderPass               = renderPass;
    rpBegin.framebuffer              = frameBuffer;
    rpBegin.renderArea.extent.width  = surfaceSize.width;
    rpBegin.renderArea.extent.height = surfaceSize.height;
    vkCmdBeginRenderPass(cmdBuffer, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkViewport vp{};
    vp.width    = static_cast<float>(surfaceSize.width);
    vp.height   = static_cast<float>(surfaceSize.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    // Set viewport dynamically
    vkCmdSetViewport(cmdBuffer, 0, 1, &vp);

    VkRect2D scissor{};
    scissor.extent.width  = surfaceSize.width;
    scissor.extent.height = surfaceSize.height;
    // Set scissor dynamically
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0 /*1st binding*/, 1 /*bindings count*/, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmdBuffer, indexBuffer, 0 /*offset*/, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmdBuffer, static_cast<uint32_t>(indiciesRects.size()), 1 /*instance count*/, 0 /*first index*/,
        0 /*vertex offset*/, 0 /*first instance*/);

    vkCmdEndRenderPass(cmdBuffer);
}

}
