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
#include "subpixelmerge.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

DECLARE_LAYER_GL_CLASS(Subpixel);

using namespace snn;
using namespace snn::dp;

static constexpr const char* SUBPIXEL_MERGE_FS_ASSET_NAME = "shaders/shadertemplate_fs_subpixel.glsl";

InferencePassesSptr SubpixelLayerGl::createFS(const LayerGenOptions&) const {
    if (_desc.numInputPlanes <= 1) {
        SNN_LOGD("Input number is less and equal to 1: layer = %s", name.c_str());
    }

    InferencePassesSptr ret(new InferencePassesGl());
    std::ostringstream preDefine;
    std::ostringstream postDefine;

    preDefine << "#version 320 es\n";
    preDefine << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << std::endl;
    preDefine << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << std::endl;
    preDefine << "#define NUM_KERNEL_SIZE " << _desc.kernelSize << std::endl;
    if (_desc.numInputPlanes <= 4) {
        preDefine << "#define INPUT_TEXTURE_2D\n";
    }
    if (_desc.kernelSize > 2) {
        preDefine << "#define KERNEL_LARGER_THAN_2\n";
    }

    postDefine << ("s = tanh(s);\n");
    postDefine << "o_pixel = s;\n";
    postDefine << "}\n";

    auto fsCode = loadShader(SUBPIXEL_MERGE_FS_ASSET_NAME);

    if (_desc.preferHp) {
        findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
    } else {
        findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
    }

    std::unordered_map<std::string, gl::SimpleUniform::Value> uniforms;

    // Get the list of passes.
    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassGl& pass  = passes[0];
    pass.source                 = preDefine.str() + fsCode + postDefine.str();
    pass.inputs                 = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
    pass.program                = InferencePassGl::FsProgram {(uint32_t) 0, DIV_4_ROUND_UP(_desc.numOutputPlanes)};
    pass.uniforms["kernelSize"] = _desc.kernelSize;

    return ret;
}

InferencePassesSptr SubpixelLayerGl::createCS(const LayerGenOptions&) const {
    SNN_LOGW("Compute shader not implemented! Falling back to fragment shader.");
    return {};
}
