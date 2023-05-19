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
#include "vulkanContext.h"
#include "vulkanBackend.h"
#include "vulkanRenderpass.h"
#include "dp.h"
#include "inferencepassVulkan.h"
#include "vkUtils.h"
#include <vector>

using namespace snn;
using namespace snn::dp;

VulkanBackend::VulkanBackend(GpuContext* context_)
    : context(context_)
{
    uvkc::benchmark::VulkanContext* ukvcContext = VulkanGpuContext::cast(context)->getUvkcContext();
    _device = (ukvcContext->devices[0].get());

    BM_CHECK_OK_AND_ASSIGN(_sampler0, (_device->CreateSampler()));
    BM_CHECK_OK_AND_ASSIGN(_sampler1, (_device->CreateSampler()));
    BM_CHECK_OK_AND_ASSIGN(_sampler2, (_device->CreateSampler()));

    // TODO: set atrributes like padding of sampler here

    BM_CHECK_OK_AND_ASSIGN(_weightSampler0,  (_device->CreateSampler()));
    BM_CHECK_OK_AND_ASSIGN(_cmdBuffer, (_device->AllocateCommandBuffer()));
}

void VulkanBackend::initRenderPasses(snn::dp::GenericModelLayer* modelLayer, snn::ImageTextureArrayAccessor texInputs,
    snn::ImageTextureArrayAccessor texOutputs) {
    modelLayer->getRenderPasses().clear();

    if (modelLayer->isInputLayer()) {
        return;
    }

    std::vector<uvkc::vulkan::Sampler*> samplers;
    samplers.push_back(_sampler0.get());
    samplers.push_back(_sampler1.get());
    samplers.push_back(_sampler2.get());

    std::vector<uvkc::vulkan::Sampler*> weightSamplers;
    weightSamplers.push_back(_weightSampler0.get());

    const InferencePassesVulkan* passesVulkan = InferencePassesVulkan::cast(modelLayer->getPasses());
    for (size_t i = 0; i < passesVulkan->passes.size(); i++) {
        auto& pass = passesVulkan->passes[i];

        VulkanRenderPass::CreationParameters rpcp = {
            formatString("%s pass[%d]", modelLayer->getName().c_str(), i),
            pass,
            samplers,
            weightSamplers,
            texInputs,
            texOutputs,
            _device,
            _cmdBuffer.get(),
        };

        auto renderPass = std::make_shared<snn::VulkanRenderPass>(context, rpcp);

        modelLayer->getRenderPasses().push_back(renderPass);
    }
}

void VulkanBackend::prepareRun(snn::MixedInferenceCore::RunParameters& rp,
        RenderStagesArray &stages, bool bindOutput, uint32_t bindIndex) {
    (void) rp;
    (void) stages;
    (void) bindOutput;
    (void) bindIndex;
#ifdef __ANDROID__
    if (rp.outputImages()) {
        if (bindOutput) {
            stages[bindIndex].stageOutputs[0].attach(&rp.outputImages[0]);
        }
    }
#endif
    BM_CHECK_OK(_cmdBuffer->Begin());
    _isSynced = false;
}

bool VulkanBackend::sync() {
    if (_isSynced) {
        SNN_LOGD("already synced");
        return false;
    }
    BM_CHECK_OK(_cmdBuffer->End());
    BM_CHECK_OK(_device->QueueSubmitAndWait(*_cmdBuffer));
    _isSynced = true;
    return true;
}

void VulkanBackend::postRun(RenderStagesArray &stages, bool dumpOutput, const std::string &folder) {
    if (!dumpOutput) {
        return;
    }
    for (size_t j = 0; j < stages.size(); j++) {
        auto& s = stages[j];
        if (s.layer->isInputLayer) {
            continue;
        }
        if (s.backend == Backend::Backend_CPU) {
            continue;
        }
        std::string path = formatString("%s/%s", folder.c_str(), s.layer->name.c_str());
        if (!createDirIfNotExists(path)) {
            return;
        }

        auto image = s.stageOutputs[0].getRawImage();
        SNN_LOGD("Saving outputs for layer: %s %zu", s.layer->name.c_str(), image.depth());
        for (std::size_t i = 0; i < image.depth(); i++) {
            auto filename = formatString("%s/%s/%02d.png", folder.c_str(), s.layer->name.c_str(), i);
            image.saveToPNG(filename, i, true);
        }

        auto dumpFileName = formatString("%s/%s pass[0].dump", folder.c_str(), s.layer->name.c_str());
        SNN_LOGD("Saving dump to %s", dumpFileName.c_str());
        s.stageOutputs[0].saveToBIN(dumpFileName);
#if DUMP_RESULTS_TXT
        auto dumpTxtFileName = formatString("%s/%s.txt", folder.c_str(), s.layer->name.c_str());
        if (FILE* fDumpTxt = createFile(dumpTxtFileName.c_str())) {
            s.stageOutputs[0].prettyPrint(fDumpTxt);
            fclose(fDumpTxt);
        }
#endif
    }
}

void VulkanBackend::cleanupRun() {
    return;
}

DeviceTimer* VulkanBackend::createDeviceTimer(const std::string& name) {
    return new vk::GpuTimeElapsedQuery(name, _cmdBuffer.get(), _device);
}
