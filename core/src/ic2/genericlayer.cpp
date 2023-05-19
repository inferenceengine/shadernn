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
#include "genericlayer.h"
#include "backend.h"
#include <string>
#include <utility>

namespace snn {
namespace dp {

GenericModelLayer::~GenericModelLayer() {
    prevLayers.clear();
    nextLayers.clear();
}

void GenericModelLayer::init(dp::DeviceBackend *backend, ImageTextureArray& inputMat, ImageTextureArray& outputMat) {
    SNN_LOGD("Layer initialized: %s", name.c_str());
    backend->initRenderPasses(this, inputMat, outputMat);
}

void GenericModelLayer::setName(const std::string& genericName) {
    name = genericName;
    SNN_LOGD("Layer name: %s", name.c_str());
}

void GenericModelLayer::run(snn::dp::DeviceBackend *, bool dumpOutputs) {
    SNN_LOGD("Layer run: %s, %lu passes", name.c_str(), renderPasses.size());

    for (std::size_t passCount = 0; passCount < renderPasses.size(); ++passCount) {
        auto& renderPass = renderPasses[passCount];
        bool lastPass = passCount == renderPasses.size() - 1;
        if (dumpOutputs && lastPass) {
            if (!renderPass->debugPassInputs(OUTPUT_DIR)) {
                SNN_LOGE("Error dumping inputs for layer %s", name.c_str());
            }
            if (!renderPass->debugPassWeights(OUTPUT_DIR, passCount)) {
                SNN_LOGE("Error dumping weights for layer %s", name.c_str());
            }
        }

        renderPass->run();

        if (dumpOutputs && lastPass) {
            if (!renderPass->debugPassOutput(OUTPUT_DIR)) {
                SNN_LOGE("Error dumping outputs for layer %s", name.c_str());
            }
        }
    }
}

void GenericModelLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    width = height = depth = 0U;
    InferenceGraph::Transform accumulatedTransform = {0, {{0.0f, 0.0f, 0.0f, 0.0f}}};
    auto t                                         = getOutputScaleDimAdjustment();
    for (auto& dim : inputDims) {
        SNN_LOGV("dim:%d, %d, %d", dim.width, dim.height, dim.depth);
        if (!t.isFixed) {
            accumulatedTransform.scaleWidth      = std::max(accumulatedTransform.scaleWidth, t.scaleWidth * dim.width);
            accumulatedTransform.translateWidth  = std::max(accumulatedTransform.translateWidth, t.translateWidth);
            accumulatedTransform.scaleHeight     = std::max(accumulatedTransform.scaleHeight, t.scaleHeight * dim.height);
            accumulatedTransform.translateHeight = std::max(accumulatedTransform.translateHeight, t.translateHeight);
            width                                = accumulatedTransform.scaleWidth + accumulatedTransform.translateWidth;
            height                               = accumulatedTransform.scaleHeight + accumulatedTransform.translateHeight;
            depth                                = std::max(depth, dim.channels);
            SNN_LOGV("dim:%d, %d, %d", width, height, depth);
        } else {
            accumulatedTransform.fixedWidth  = std::max(accumulatedTransform.fixedWidth, t.fixedWidth);
            accumulatedTransform.fixedHeight = std::max(accumulatedTransform.fixedHeight, t.fixedHeight);
            accumulatedTransform.fixedDepth  = std::max(accumulatedTransform.fixedDepth, t.fixedDepth);
            accumulatedTransform.fixedBatch  = std::max(accumulatedTransform.fixedBatch, t.fixedBatch);
            width                            = accumulatedTransform.fixedWidth + accumulatedTransform.translateWidth;
            height                           = accumulatedTransform.fixedHeight + accumulatedTransform.translateHeight;
            depth                            = std::max(depth, dim.channels);
            SNN_LOGV("dim:%d, %d, %d", width, height, depth);
        }
    }
}

void ShaderLayer::createInferencePasses(const LayerGenOptions& options) {
    InferencePassesSptr ret;
    // Try create Vulkan shader, if the options says so.
    if (options.vulkan) {
        ret = createCS(options);
        setLayerExecutionType(InferenceGraph::LayerExecutionType::GPU_VK);
    } else {
        // auto executeLevel = getLayerExecutionType();
        // Try create compute shader, if the options says so.
        if ((options.compute)) {
            ret = createCS(options);
            setLayerExecutionType(InferenceGraph::LayerExecutionType::GPU_CS);
        }
        // Buf if the shader layer does not support compute shader yet, we'll fallback to fragment shader.
        if (!ret) {
            ret = createFS(options);
            setLayerExecutionType(InferenceGraph::LayerExecutionType::GPU_FS);
        }
    }
    SNN_ASSERT(ret);

    passes = ret;
}

std::string ShaderLayer::loadShader(const char* path) {
    auto bytes = loadEmbeddedAsset(path);
    return std::string(bytes.begin(), bytes.end());
}

void ShaderLayer::findAndReplace(std::string& s, const std::string& from, const std::string& to) {
    auto startIndex = s.find(from);
    while (startIndex != std::string::npos) {
        s.replace(startIndex, from.size(), to);
        startIndex = s.find(from, startIndex + to.size());
    }
}

}   // namespace dp
}   // namespace snn
