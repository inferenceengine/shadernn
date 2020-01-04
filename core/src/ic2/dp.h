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

#include "layeroption.h"
#include "genericlayer.h"
#include "activation.h"
#include "adaptiveavgpool2d.h"
#include "addlayer.h"
#include "avgpool2d.h"
#include "batchnorm.h"
#include "calculation.h"
#include "concatenation.h"
#include "conv2d.h"
#include "cpulayer.h"
#include "deconv2d.h"
#include "denselayer.h"
#include "flattenlayer.h"
#include "inputlayer.h"
#include "instancenorm.h"
#include "maxpool2d.h"
#include "padlayer.h"
#include "separableconvolution.h"
#include "subpixelmerge.h"
#include "upsampling2d.h"
#include "yololayer.h"

// TODO: replace them with inline functions
#define DIV_4_ROUND_UP(i)      (((i) + 3) / 4)
#define DIV_AND_ROUND_UP(x, y) ((x + (y - 1)) / y)
#define ROUND_UP_DIV_4(i)      (((i) + 3) / 4 * 4)
#define ROUND_UP_AND_DIV(x, y) ((x + (y - 1)) / y * y)
#define UP_DIV(x, y)           (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y)         (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x)           ROUND_UP((x), 4)

#define USE_CPU_LAYER     1
#define USE_BUFFER_OBJECT 1

#define DECLARE_LAYER(layer)                                                                                                                                   \
    snn::dp::GenericModelLayer* layer##Creator(snn::dp::ModelParser& parser, int i) {                                                                          \
        snn::dp::layer##Desc desc;                                                                                                                             \
        desc.parse(parser, i);                                                                                                                                 \
        return new snn::dp::layer##Layer(std::move(desc));                                                                                                     \
    }

#define REGISTER_LAYER(layer) registerLayer(std::string(#layer), layer##Creator)

// static std::vector<int> mLocalSize{8, 8, 2};  //For snapdragon 888
static std::vector<int> mLocalSize {4, 8, 4}; // For snapdragon 8Gen 1

namespace snn {
namespace dp { // short for Dynamic Pipeline

std::vector<std::shared_ptr<GenericModelLayer>> loadFromJsonModel(const std::string& fileName, const snn::MRTMode& mrtMode,
                                                                  const snn::WeightAccessMethod& weightMode, bool preferHp = true);

typedef std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> InferenceModel;

InferenceGraph generateInferenceGraph(const std::shared_ptr<snn::dp::GenericModelLayer> firstLayer, const ShaderGenOptions& options);

typedef GenericModelLayer* (*LayerCreator)(ModelParser& parser, int i);
struct ShaderLayerEntry {
    std::string layerNme;
    LayerCreator creator;
};

bool initLayerRegisty();

bool registerLayer(std::string layerName, LayerCreator creator);

}; // namespace dp
} // namespace snn
