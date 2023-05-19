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
#include "snn/utils.h"
#include "modelparser.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct InputLayerDesc : CommonLayerDesc {
    uint32_t inputHeight, inputWidth, inputChannels;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        parser.getInputLayer(layerId, inputWidth, inputHeight, inputChannels, inputIndex);
        isInputLayer = true;
    }
};

// This is a class that represents input layer. It does not generate any shader.
class InputLayerLayer : public ShaderLayer {
public:
    InputLayerLayer(InputLayerDesc desc): ShaderLayer(desc) {}
    virtual ~InputLayerLayer() = default;

protected:
    virtual InferencePassesSptr createFS(const LayerGenOptions&) const override { SNN_RIP("Not implemented !"); }
    virtual InferencePassesSptr createCS(const LayerGenOptions&) const override { SNN_RIP("Not implemented !"); }

private:
    InputLayerDesc desc;
};

}; // namespace dp
} // namespace snn
