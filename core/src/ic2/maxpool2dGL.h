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

#include "maxpool2d.h"
#include "snn/utils.h"
#include <string>
#include <sstream>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class MaxPooling2DLayerGl : public MaxPooling2DLayer {
public:
    MaxPooling2DLayerGl(MaxPooling2DDesc&& d): MaxPooling2DLayer(std::move(d)) {}
    virtual ~MaxPooling2DLayerGl() = default;

protected:
    InferencePassesSptr createFS(const LayerGenOptions&) const override;
    InferencePassesSptr createCS(const LayerGenOptions&) const override;

private:
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;

    void buildTextureDefLogic(std::ostream& stream, uint32_t inputSliceIndex) const;

    void buildCalcDefLogic(std::ostream& stream) const;

    void buildComputePostDefine(std::ostream& stream, uint32_t outputSliceIndex) const;

    void buildFragPostDefine(std::ostream& stream) const;
};

}; // namespace dp
} // namespace snn
