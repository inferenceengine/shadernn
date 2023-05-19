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
#include "pch.h"
#include "denselayer.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>

DECLARE_LAYER_GL_CLASS(Dense);

using namespace snn;
using namespace snn::dp;

static constexpr const char* DENSE_CS_ASSET_NAME = "shaders/shadertemplate_cs_dense.glsl";

InferencePassesSptr DenseLayerGl::createFS(const LayerGenOptions&) const {
    InferencePassesSptr ret(new InferencePassesGl());
    SNN_LOGW("%%%%%%%% %s:%d No FS implementation for Dense layer !");
    return ret;
}

InferencePassesSptr DenseLayerGl::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesSptr ret(new InferencePassesGl());

    int inputWidth       = (int) _desc.weights[0].size();
    uint32_t outputWidth = (uint32_t) _desc.biases.size();

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassGl& pass = passes[0];

    std::string shaderHeader;
    if (_desc.preferHp) {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION mediump\n"
                       "precision PRECISION float;\n"
                       "layout(std430) buffer;\n"
                       "#define OUTPUT_FORMAT rgba16f\n";
    } else {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION highp\n"
                       "precision PRECISION float;\n"
                       "layout(std430) buffer;\n"
                       "#define OUTPUT_FORMAT rgba32f\n";
    }

    std::string sourceCode = loadShader(DENSE_CS_ASSET_NAME);

    pass.uniforms = {{"uWidth", inputWidth},
                    {"activation", 0}};
    pass.inputs   = {{"uInImage", 0}};
    pass.source   = shaderHeader + sourceCode;
    pass.program  = InferencePassGl::CsProgram {
        "uOutImage",
        // div-by-N is determined by work group size defined CS program.
        {1, outputWidth, 1},
    };

    pass._vecWeights.resize(inputWidth * outputWidth);
    float* destWeight = pass._vecWeights.data();
    unsigned int width = _desc.weights[0].size();
    int kIndex = 0;
    for (size_t i = 0; i < _desc.weights.size(); i++) {
        for (size_t j = 0; j < width; j++) {
            *(destWeight + kIndex) = _desc.weights[i][j];
            kIndex++;
        }
    }

    pass._vecBias.resize(_desc.numOutputPlanes, 0.0f);
    for (size_t i = 0; i < _desc.biases.size(); i++) {
        pass._vecBias[i] = (float) _desc.biases[i];
    }
    return ret;
}
