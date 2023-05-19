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

struct UnaryDesc : CommonLayerDesc {
    int32_t opType = 0;
    float opValue = 1.0f;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
    }
};

// This is a base class to generates a shader for a family of elementwise unary functions
class UnaryLayer : public ShaderLayer {
public:
    UnaryLayer(UnaryDesc&& d): ShaderLayer(std::move(d)), _desc(std::move(d)) {}
    virtual ~UnaryLayer() = default;

protected:
    UnaryDesc _desc;
};


}; // namespace dp
} // namespace snn
