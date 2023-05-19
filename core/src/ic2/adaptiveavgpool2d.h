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
#include <string>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct AdaptiveAvgPool2dDesc : GenericConvDesc {
    std::string padding;
    int targetSize = 1;
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getAdaptiveAvgPoolLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, (int&) targetSize);
    }
};

} // namespace dp
} // namespace snn
