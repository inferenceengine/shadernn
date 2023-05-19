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

#include "upsampling2d.h"
#include "snn/utils.h"
#include <string>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class UpSampling2DLayerGl : public UpSampling2DLayer {
public:
    UpSampling2DLayerGl(UpSampling2DDesc&& d): UpSampling2DLayer(std::move(d)) {}
    virtual ~UpSampling2DLayerGl() = default;

private:
    bool generateUpSampling2DGLSamplingCode(int& idxStartPlane, int nOutputChannels, std::string& uniformsDeclaration, std::string& calculation,
                                            const bool& compute) const;
protected:
    InferencePassesSptr createFS(const LayerGenOptions&) const override;
    InferencePassesSptr createCS(const LayerGenOptions&) const override;
};

} // namespace dp
} // namespace snn
