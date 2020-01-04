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
#include "dp.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <ios>

using namespace std;
using namespace snn;
using namespace snn::dp;

static constexpr const char* FLATTEN_CS_ASSET_NAME = "shaders/shadertemplate_cs_activation.glsl";

snn::dp::GenericModelLayer::GLSLShaders snn::dp::FlattenLayer::createGLSLShader(const LayerGenOptions& options) {
    (void) options;
    GLSLShaders ret;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;
    snn::dp::GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(1);

    InferenceGraph::Pass& pass = passes[0];

    std::string sourceCode = "#version 320 es \n"
                             "#define PRECISION mediump\n"
                             "precision PRECISION float;\n"
                             "layout(rgba32f, binding=0) writeonly uniform PRECISION image2DArray uOutImage;\n"
                             "layout(rgba32f, binding=1) readonly uniform PRECISION image2DArray uInImage;\n"
                             "layout(location = 2) uniform int uWidth;\n"
                             "layout(location = 3) uniform int uHeight;\n"
                             "layout(binding=5) writeonly buffer destBuffer{\n"
                             "    float data[];\n"
                             "} uOutBuffer;\n"
                             "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
                             "\n"
                             "void main()\n"
                             "{\n"
                             "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
                             "    int z = pos.z/(uWidth*uHeight*4);\n"
                             "    int offset = z*uWidth*uHeight*4;\n"
                             "    int wh = uWidth*uHeight;"
                             "    for (int w = 0; w < uWidth; w+=1) \n"
                             "    {\n"
                             "for (int h = 0; h < uHeight; h+=1) \n"
                             "{\n"
                             "           vec4 color0 = imageLoad(uInImage, ivec3(w, h, z));\n"
                             // "        float test = float(w);\n"
                             // "           uOutBuffer.data[offset+wh*0+h*uWidth+w] = color0.r;\n"
                             // "           uOutBuffer.data[offset+wh*1+h*uWidth+w] = color0.g;\n"
                             // "           uOutBuffer.data[offset+wh*2+h*uWidth+w] = color0.b;\n"
                             // "           uOutBuffer.data[offset+wh*3+h*uWidth+w] = color0.a;\n"
                             "           imageStore(uOutImage,ivec3(offset+wh*0+h*uWidth+w, 0, 0),vec4(color0.r,0,0,0));\n"
                             "           imageStore(uOutImage,ivec3(offset+wh*1+h*uWidth+w, 0, 0),vec4(color0.g,0,0,0));\n"
                             "           imageStore(uOutImage,ivec3(offset+wh*2+h*uWidth+w, 0, 0),vec4(color0.b,0,0,0));\n"
                             "           imageStore(uOutImage,ivec3(offset+wh*3+h*uWidth+w, 0, 0),vec4(color0.a,0,0,0));\n"
                             "       }\n"
                             "    }\n"
                             "}\n";
    sourceCode = loadShader(FLATTEN_CS_ASSET_NAME);

    pass.uniforms = {{"uWidth", 1}, {"uHeight", 1}};
    pass.inputs   = {{"uInImage", 0}};
    pass.source   = sourceCode;
    pass.program  = InferenceGraph::Pass::CsProgram {
        "uOutImage",
        // div-by-N is determined by work group size defined CS program.
        //{ 1, 1, (inputChannels+3)/4},
        {1, 1, outputWidth},
    };

    setLayerExecutionLevel(snn::InferenceGraph::LayerExecution::GPU_CS);

    return ret;
}

void snn::dp::FlattenLayer::computeImageTexture(FixedSizeArray<snn::ImageTexture>& inputTex, FixedSizeArray<snn::ImageTexture>& outputTex) {
    std::shared_ptr<snn::ManagedRawImage> outputTexPtr = std::make_shared<snn::ManagedRawImage>(inputTex[0].texture(0)->getBaseLevelPixels());
    std::vector<std::shared_ptr<snn::ManagedRawImage>> inputMat {outputTexPtr};

    auto cpuL         = snn::dp::CPUCommonUtil<float> {_flattenDesc.activation, _flattenDesc.leakyReluAlpha, false};
    auto transformMat = std::pair<std::vector<std::vector<float>>, std::vector<float>>(std::vector<std::vector<float>>(), std::vector<float>());

    if (!cpuL.gpuTexMat.has_value()) {
        cpuL.gpuTexMat.emplace(inputMat);
    }
    cpuL.run(transformMat);
    cpuL.getOutputs(outputTex[0].outputMat);

    SNN_LOGD("%%%%%%%% %s:%d :%s\n", __FUNCTION__, __LINE__, name.c_str());
}

InferenceGraph::Transform FlattenLayer::getOutputScaleDimAdjustment() const {
    InferenceGraph::Transform ret;
    ret.isFixed     = 1;
    ret.fixedWidth  = 512;
    ret.fixedHeight = 1;
    ret.fixedDepth  = 1;
    ret.fixedBatch  = 1;
    return ret;
}

void FlattenLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    width  = inputDims[0].width * inputDims[0].height * inputDims[0].depth * 4;
    height = 1;
    depth  = 1;
}
