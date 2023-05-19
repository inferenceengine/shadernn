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
#include "upsampling2dGL.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <utility>

using namespace snn;
using namespace snn::dp;

static constexpr const char* UPSAMPLING2D_FS_ASSET_NAME          = "shaders/shadertemplate_fs_upsampling2d.glsl";
static constexpr const char* UPSAMPLING2D_NEAREST_CS_ASSET_NAME  = "shaders/3rdparty/shadertemplate_cs_upsampling2d_nearest.glsl";
static constexpr const char* UPSAMPLING2D_BILINEAR_CS_ASSET_NAME = "shaders/3rdparty/shadertemplate_cs_upsampling2d_bilinear.glsl";

InferencePassesSptr UpSampling2DLayerGl::createFS(const LayerGenOptions&) const {
    InferencePassesSptr ret(new InferencePassesGl());

    auto& desc = getDesc();

    int numShaderPasses = (int) DIV_4_ROUND_UP(desc.numOutputPlanes);

    std::string preDefine          = "#version 320 es\n#define SCALE_FACTOR " + std::to_string(_desc.scale) + "\n";
    std::string shaderTemplateCode = loadShader(UPSAMPLING2D_FS_ASSET_NAME);

    int idxStartPlane = 0;

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(numShaderPasses);
    for (int i = 0, j = 0; i < numShaderPasses; i++, j += 4) {
        int outputChannels = std::min(4, (int) desc.numOutputPlanes - j);

        // Create a copy of the template code.
        // After modification, this will contain the shader's true source code.
        std::string fsCode = shaderTemplateCode;
        std::string uniformsDeclaration;
        std::string calculation;
        if (!generateUpSampling2DGLSamplingCode(idxStartPlane, outputChannels, uniformsDeclaration, calculation, false)) {
            return {};
        }

        std::string declarationShaderPlaceholder("_PLACEHOLDER_UNIFORMS_DECLARATION_");
        findAndReplace(fsCode, declarationShaderPlaceholder, uniformsDeclaration);

        std::string calCulationShaderPlaceholder("_PLACEHOLDER_CALCULATION_");
        findAndReplace(fsCode, calCulationShaderPlaceholder, calculation);

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        InferencePassGl& pass = passes[i];
        pass.inputs                = {{"inputTextures", 0}};
        pass.source                = preDefine + fsCode;
        pass.program               = InferencePassGl::FsProgram {(uint32_t) i, (uint32_t) DIV_4_ROUND_UP(outputChannels)};
    }
    return ret;
}

InferencePassesSptr UpSampling2DLayerGl::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesSptr ret(new InferencePassesGl());

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassGl& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t inputDepth  = inputDims[0].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = _desc.numOutputPlanes;

    std::string shaderHeader;
    if (_desc.preferHp) {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION mediump\n"
                       "precision PRECISION float;\n"
                       "layout(std140) uniform;\n"
                       "#define OUTPUT_FORMAT rgba16f\n";
    } else {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION highp\n"
                       "precision PRECISION float;\n"
                       "layout(std140) uniform;\n"
                       "#define OUTPUT_FORMAT rgba32f\n";
    }
    if (_desc.numInputPlanes <= 4) {
        shaderHeader += "#define INPUT_TEXTURE_2D\n";
    }
    if (_desc.numOutputPlanes <= 4) {
        shaderHeader += "#define OUTPUT_TEXTURE_2D\n";
    }
    std::string shaderUniforms = "#ifdef INPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput;\n"
                            "#endif\n"
                            "#ifdef OUTPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D uOutput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"
                            "#endif\n";

    std::string shaderMain;

    int resizeType = 0;
    if (_desc.interpolationType.compare("nearest") == 0) {
        resizeType = 0; // 0 nearest neighbor ; 1 bilinear
    } else if (_desc.interpolationType.compare("bilinear") == 0) {
        resizeType = 1;
    }

    if (resizeType == 0) {
        shaderMain = loadShader(UPSAMPLING2D_NEAREST_CS_ASSET_NAME);
    } else if (resizeType == 1) {
        shaderMain = loadShader(UPSAMPLING2D_BILINEAR_CS_ASSET_NAME);
    }

    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    uint32_t paddingOffsets[4] = {0U, 0U, 0U, 0U};
    SNN_LOGD("%s:%d, Padding: %d, %d, %d, %d\n", __FILENAME__, __LINE__, paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3]);
    // Hack it. Looks like not padding on top left in NCNN
    paddingOffsets[0] = 0;
    paddingOffsets[2] = 0;

    pass.uniforms = {
        {"inImgSize", glm::ivec4(inputWidth, inputHeight, ic_4, 1)},
        {"outImgSize", glm::ivec4(outputWidth, outputHeight, oc_4, 1)},
        {"scale", glm::vec2(1 / _desc.scale, 1 / _desc.scale)},
        {"means", glm::vec4(0.0f, 0.0f, 0.0f, 0.0f)},
        {"norms", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)},
    };
    pass.inputs  = {{"uInput", 0}};
    pass.source  = (shaderHeader + shaderUniforms + shaderMain);
    pass.program = InferencePassGl::CsProgram {"uOutput",
                                                // div-by-N is determined by work group size defined CS program.
                                                {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}

bool UpSampling2DLayerGl::generateUpSampling2DGLSamplingCode(int& idxStartPlane, int nOutputChannels, std::string& uniformsDeclaration,
    std::string& calculation, const bool& computeShader) const {
    uniformsDeclaration = "";
    calculation         = "";

    auto curTexture = prevLayers.begin();

    std::string varsAssignment, resultAssigment;
    if (1 == (*curTexture)->getDesc().numOutputPlanes) {
        varsAssignment =
            "float value = texelFetch(inputTextures, ivec2(int(float(uv.x)/SCALE_FACTOR), int(float(uv.y)/SCALE_FACTOR))," + std::to_string(0) + ").r;\n";
        resultAssigment     = "    s = vec4(value, 0.0f, 0.0f, 0.0f); ";
        uniformsDeclaration = "uniform sampler2D inputTextures;";
    } else {
        int layer       = idxStartPlane >> 2;
        std::string str = std::to_string(layer);
        if (computeShader) {
            varsAssignment = "FLOAT_PRECISION vec4 value = texelFetch(inputTextures, ivec3(ivec2(int((uv.x)/SCALE_FACTOR), int((uv.y)/SCALE_FACTOR))," + str +
                            "), 0).rgba;\n";
        } else {
            varsAssignment =
                "FLOAT_PRECISION vec4 value = texelFetch(inputTextures, ivec3(ivec2(int(float(uv.x)/SCALE_FACTOR), int(float(uv.y)/SCALE_FACTOR))," + str +
                "), 0).rgba;\n";
        }

        resultAssigment     = "    s = vec4(value); ";
        uniformsDeclaration = "uniform sampler2DArray inputTextures;";
    }

    calculation = varsAssignment + resultAssigment;

    idxStartPlane += nOutputChannels;

    return true;
}
