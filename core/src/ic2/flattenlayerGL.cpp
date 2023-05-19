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
#include "flattenlayer.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>

DECLARE_LAYER_GL_CLASS(Flatten);

using namespace snn;
using namespace snn::dp;

static constexpr const char* FLATTEN_CS_ASSET_NAME = "shaders/shadertemplate_cs_flattenlayer.glsl";

InferencePassesSptr snn::dp::FlattenLayerGl::createFS(const LayerGenOptions&) const {
    InferencePassesSptr ret(new InferencePassesGl());
    SNN_LOGW("No FS implementation for Flatten layer !");
    return ret;
}

InferencePassesSptr snn::dp::FlattenLayerGl::createCS(const LayerGenOptions& options) const {
    (void) options;
    InferencePassesSptr ret(new InferencePassesGl());

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;
    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

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
    std::string sourceCode = loadShader(FLATTEN_CS_ASSET_NAME);

    pass.uniforms = {{"uWidth", (int)inputDims[0].width}, {"uHeight", (int)inputDims[0].height}};
    pass.inputs   = {{"uInImage", 0}};
    pass.source   = shaderHeader + sourceCode;
    pass.program  = InferencePassGl::CsProgram {
        "uOutImage",
        {1, 1, inputDims[0].depth},
    };

    return ret;
}
