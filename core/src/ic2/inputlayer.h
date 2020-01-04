/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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

#include <snn/snn.h>
#include <snn/utils.h>
#include <snn/imageTexture.h>
#include "inferencegraph.h"
#include "modelparser.h"
#include <utility>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>

#include "genericlayer.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct InputLayerDesc : CommonLayerDesc {
    uint32_t inputHeight, inputWidth, inputChannels;
    void parse(ModelParser& parser, int layerId) {
        CommonLayerDesc::parse(parser, layerId);
        parser.getInputLayer(layerId, inputWidth, inputHeight, inputChannels);
    }
};

class InputLayerLayer : public ShaderLayer {
public:
    InputLayerLayer(InputLayerDesc desc): ShaderLayer(desc) {}
    ~InputLayerLayer() {}
    virtual void getInputDims(uint32_t& height, uint32_t& width, uint32_t& depth) const override {
        height = this->desc.inputHeight;
        width  = this->desc.inputWidth;
        depth  = this->desc.inputChannels;
    }

private:
    InputLayerDesc desc;
};

}; // namespace dp
} // namespace snn
