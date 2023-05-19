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
#include "calculationGL.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <utility>

using namespace snn;
using namespace snn::dp;

static constexpr const char* CALCULATION_FS_ASSET_NAME = "shaders/shadertemplate_fs_calculation.glsl";

InferencePassesSptr CalculateLayerGl::createFS(const LayerGenOptions&) const {
    auto& desc           = getDesc();
    auto numShaderPasses = static_cast<int>(DIV_4_ROUND_UP(desc.numOutputPlanes));

    auto preDefine          = "#version 320 es\n";
    auto shaderTemplateCode = loadShader(CALCULATION_FS_ASSET_NAME);

    InferencePassesSptr ret(new InferencePassesGl());
    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(numShaderPasses);
    for (int i = 0, j = 0; i < numShaderPasses; i++, j += 4) {
        auto outputChannels = std::min(4, static_cast<int>(desc.numOutputPlanes) - j);

        // Create a copy of the template code.
        // After modification, this will contain the shader's true source code.
        auto fsCode = shaderTemplateCode;

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        InferencePassGl& pass = passes[i];
        pass.source                = preDefine + fsCode;
        pass.inputs                = {{"inputTextures", 0}};
        pass.program               = InferencePassGl::FsProgram {static_cast<uint32_t>(i), static_cast<uint32_t>(DIV_4_ROUND_UP(outputChannels))};
    }
    return ret;
}
