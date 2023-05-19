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
#include "snn/snn.h"
#include "modelparser.h"
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct ConcatenateDesc : CommonLayerDesc {};

class ConcatenateLayer : public ShaderLayer {
public:
    ConcatenateLayer(ConcatenateDesc&& d): ShaderLayer(d), _desc(std::move(d)) {}
    virtual ~ConcatenateLayer() = default;

protected:
    void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depthOut) const override {
        width    = inputDims[0].width;
        height   = inputDims[0].height;
        depthOut = inputDims[0].depth + inputDims[1].depth;
    }

    ConcatenateDesc _desc;
};

}; // namespace dp
} // namespace snn
