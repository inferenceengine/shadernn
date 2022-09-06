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

#include "processor.h"
#include <snn/texture.h>
#include <opencv2/core/mat.hpp>
#include <string>
#include "ic2/core.h"

static std::map<snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm, std::string> modelMap = {
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::CANDY, "candy-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::RAIN_PRINCESS, "rain-princess-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::MOSAIC, "mosaic-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::POINTILISM, "pointilism-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::UDNIE, "udnie-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::NONE, ""}};

namespace snn {
class styleTransferProcessor : public Processor {
public:
    ~styleTransferProcessor() override = default;
    void submit(Workload&) override;

    static std::unique_ptr<styleTransferProcessor>
    createStyleTransfer(ColorFormat format, const snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm styleModel, bool compute = false,
                        bool dumpOutputs = false) {
        std::string modelFileName = modelMap[styleModel];
        return std::unique_ptr<styleTransferProcessor>(new styleTransferProcessor(format, modelFileName.c_str(), compute, dumpOutputs));
    }

    std::string getModelName() override { return ""; }

private:
    snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm _currentAlgorithm;
    std::unique_ptr<MixedInferenceCore> _ic2;
    std::unordered_map<GLuint, std::shared_ptr<Texture>> _frameTextures;
    std::string _modelFileName;
    bool _compute, _dumpOutputs;

    styleTransferProcessor(ColorFormat format, const std::string& modelFileName, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}}), _modelFileName(modelFileName), _compute(compute), _dumpOutputs(dumpOutputs) {}

    gl::TextureObject* getFrameTexture(GLuint id) {
        auto& t = _frameTextures[id];
        // Create a thin shell around textureId
        if (!t) {
            t = Texture::createAttached(GL_TEXTURE_2D, id);
        }
        return t.get();
    }
};
} // namespace snn
