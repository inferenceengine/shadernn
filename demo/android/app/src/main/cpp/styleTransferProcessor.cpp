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
#include "inferenceengine.h"
#include "appContext.h"
#include "styleTransferProcessor.h"
#ifdef SUPPORT_GL
    #include "styleTransferProcessorGL.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "volk.h"
    #include "vulkanContext.h"
    #include "styleTransferProcessorVulkan.h"
#endif

using namespace snn;

std::unique_ptr<StyleTransferProcessor>
StyleTransferProcessor::createStyleTransfer(ColorFormat format, const snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm styleModel,
    Precision precision, bool compute, bool dumpOutputs) {
    (void) compute;
    const std::string& modelFileName = modelMap[styleModel];
    switch (AppContext::getContext()->backendType) {
#ifdef SUPPORT_GL
        case GpuBackendType::GL:
            return std::unique_ptr<StyleTransferProcessor>(new StyleTransferProcessorGL(format, modelFileName.c_str(), precision, compute, dumpOutputs));
#endif
#ifdef SUPPORT_VULKAN
        case GpuBackendType::VULKAN:
            return std::unique_ptr<StyleTransferProcessor>(new StyleTransferProcessorVulkan(format, modelFileName.c_str(), precision, dumpOutputs));
#endif
        default:
            SNN_RIP("Unexpected GpuBackendType");
    }
}
