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
#include "inferenceengine.h"
#include "modelProcessorParams.h"

#include <string>
#include <memory>

static std::map<snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm, std::string> modelMap = {
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::CANDY, "StyleTransfer/candy-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::RAIN_PRINCESS, "StyleTransfer/rain-princess-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::MOSAIC, "StyleTransfer/mosaic-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::POINTILISM, "StyleTransfer/pointilism-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::UDNIE, "StyleTransfer/udnie-9_simplified.json"},
    {snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm::NONE, ""}};

namespace snn {

class StyleTransferProcessor : public Processor, protected ModelProcessorParams {
public:
    virtual ~StyleTransferProcessor() = default;
    // TODO: Why returning empty string ?
    std::string getModelName() override { return ""; }

    static const uint32_t MODEL_IMAGE_WIDTH = 224;
    static const uint32_t MODEL_IMAGE_HEIGHT = 224;

    static std::unique_ptr<StyleTransferProcessor>
    createStyleTransfer(ColorFormat format, const snn::InferenceEngine::AlgorithmConfig::StyleTransfer::StyleTransferAlgorithm styleModel, Precision precision,
                        bool compute = true, bool dumpOutputs = false);

protected:
    StyleTransferProcessor(ColorFormat format, const std::string& modelFileName, Precision precision, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}})
        , ModelProcessorParams(modelFileName, {MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, 1, 4}, precision, compute, dumpOutputs)
    {}
};

} // namespace snn
