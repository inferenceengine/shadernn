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
#include "pch.h"
#include "aiDenoiseProcessor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include "ic2/dp.h"

using namespace std;
using namespace snn;

const char* fragmentShaderSourceArray = R"glsl(#version 320 es
precision mediump float;
precision mediump sampler2DArray;

uniform sampler2DArray inputTexture;

out vec4 color;

void main()
{
    ivec2 texCoord = ivec2(gl_FragCoord.xy);
    color = texelFetch(inputTexture, ivec3(texCoord, 0), 0);
}
)glsl";

const char* adjustTextureFragmentShaderSource = R"glsl(#version 320 es
precision highp float;

uniform sampler2D inputTexture;
out vec4 color;

void main()
{
    ivec2 texCoord = ivec2(gl_FragCoord.xy);
    vec3 rgb = texelFetch(inputTexture, texCoord, 0).rgb;
    color = vec4(rgb, 0.0);
}
)glsl";

// -----------------------------------------------------------------------------
// the following is AIDenoiseProcessor2
// -----------------------------------------------------------------------------

void AIDenoiseProcessor2::submit(Workload& workload) {
    if (workload.inputCount == 0) {
        return;
    }

    const auto& inputDesc = workload.inputs[0]->desc();
    const auto& outDesc   = workload.output->desc();

    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> dp;

    // we have to delay creating ic2 because we need to know the frame size.
    if (!_ic2) {
        dp::ShaderGenOptions options = {};
        options.mrtMode              = snn::MRTMode::SINGLE_PLANE;
        options.weightMode           = snn::WeightAccessMethod::CONSTANTS;
        options.desiredInput.width   = inputDesc.width;
        options.desiredInput.height  = inputDesc.height;
        options.desiredInput.depth   = 1;
        options.desiredInput.format  = inputDesc.format;
        options.desiredOutputFormat  = outDesc.format;
        options.preferrHalfPrecision = true;
        options.compute              = _compute;

        dp = snn::dp::loadFromJsonModel(_modelFileName, options.mrtMode, options.weightMode);

        MixedInferenceCore::CreationParameters cp;
        (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp.at(0), options);
        cp.dumpOutputs         = this->_dumpOutputs;
        _ic2                   = MixedInferenceCore::create(cp);
    }

    SNN_ASSERT(inputDesc.device == Device::GPU);
    SNN_ASSERT(outDesc.device == Device::GPU);

    // const auto& fmtDesc = getColorFormatDesc(outDesc.format);

    // const auto& inFmtDesc = getColorFormatDesc(workload.inputs[0]->desc().format);

    // std::cout << "AiDenoise Input Color Format: " << inFmtDesc.name << std::endl;
    // std::cout << "AiDenoise Output Color Format: " << fmtDesc.name << std::endl;

    auto inputTexture = ((GpuFrameImage*) workload.inputs[0])->getGpuData();

    auto outputTexture = ((GpuFrameImage*) workload.output)->getGpuData();

    auto& currentFrameTexture = _frameTextures[inputTexture.texture];
    if (!currentFrameTexture) {
        currentFrameTexture = Texture::createAttached(inputTexture.target, inputTexture.texture); // Create a thin shell around textureId
    }

    auto outVec = std::vector<std::vector<std::vector<float>>>();
    auto inVec  = std::vector<std::vector<std::vector<float>>>();

    MixedInferenceCore::RunParameters rp = {};
    auto inputTextures                   = getFrameTexture(inputTexture.texture, inputTexture.target);
    rp.inputTextures                     = &inputTextures;
    rp.inputCount                        = 1;
    rp.textureOut                        = getFrameTexture(outputTexture.texture, outputTexture.target);
    rp.inputMatrix                       = inVec;
    rp.output                            = outVec;

    _ic2->run(rp);
}
