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

#include "genericlayer.h"
#include "adaptiveavgpool2d.h"
#include "snn/snn.h"
#include "snn/utils.h"
#include "snn/inferencegraph.h"
#include <string>
#include <utility>
#include <sstream>

namespace snn {
namespace dp { // short for Dynamic Pipeline

// This is a class to generates a shader for adaptive pooling for OpenGL
// https://arxiv.org/pdf/1803.01534v4.pdf
class AdaptiveAvgPool2dLayerGl : public GenericConvolutionLayer {
public:
    AdaptiveAvgPool2dLayerGl(AdaptiveAvgPool2dDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {}
    ~AdaptiveAvgPool2dLayerGl() = default;
    InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

protected:
    InferencePassesSptr createFS(const LayerGenOptions&) const override;
    InferencePassesSptr createCS(const LayerGenOptions&) const override;

private:
    AdaptiveAvgPool2dDesc _desc;
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;

    void buildTextureDefLogic(std::ostream& stream, const LayerGenOptions&, uint32_t inputSliceIndex) const;

    void buildCalcDefLogic(std::ostream& stream, const LayerGenOptions& options) const;

    void buildComputePostDefine(std::ostream& stream, uint32_t outputSliceIndex) const;

    void buildFragPostDefine(std::ostream& stream) const;
};

} // namespace dp
} // namespace snn
