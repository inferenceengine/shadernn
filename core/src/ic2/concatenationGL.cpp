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
#include "concatenationGL.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <set>
#include <utility>

using namespace snn;
using namespace snn::dp;
using namespace std::literals;

static constexpr const char* CONCATENATION_FS_ASSET_NAME = "shaders/shadertemplate_fs_concat.glsl";
static constexpr const char* CONCATENATION_CS_ASSET_NAME = "shaders/shadertemplate_cs_concat.glsl";

InferencePassesSptr ConcatenateLayerGl::createFS(const LayerGenOptions&) const {
    InferencePassesSptr ret(new InferencePassesGl());

    auto& desc = getDesc();

    int numShaderPasses = (int) DIV_4_ROUND_UP(desc.numOutputPlanes);

    std::string preDefine          = "#version 320 es\n"s;
    std::string shaderTemplateCode = loadShader(CONCATENATION_FS_ASSET_NAME);

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
        std::set<int> inputTextures;
        if (!generateConcatGLSamplingCode(idxStartPlane, outputChannels, uniformsDeclaration, inputTextures, calculation)) {
            return {};
        }

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        std::string declarationShaderPlaceholder("_PLACEHOLDER_UNIFORMS_DECLARATION_");
        findAndReplace(fsCode, declarationShaderPlaceholder, uniformsDeclaration);

        std::string calCulationShaderPlaceholder("_PLACEHOLDER_CALCULATION_");
        findAndReplace(fsCode, calCulationShaderPlaceholder, calculation);

        // setup input uniforms.
        for (auto& t : inputTextures) {
            auto s = "inputTextures"s;
            if (t > 0) {
                s += std::to_string(t);
            }
            passes[i].inputs[s] = (uint32_t) t;
        }

        InferencePassGl& pass = passes[i];
        pass.source                = preDefine + fsCode;
        pass.program               = InferencePassGl::FsProgram {(uint32_t) i, (uint32_t) DIV_4_ROUND_UP(outputChannels)};
    }
    return ret;
}

InferencePassesSptr ConcatenateLayerGl::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesSptr ret(new InferencePassesGl());

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassGl& pass = passes[0];

    uint32_t input0Depth = inputDims[0].depth;
    uint32_t input1Depth = inputDims[1].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0; // FIXME: outputDepth cannot get right value

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    SNN_LOGD("outputWidth: %d; outputHeight: %d; outputDepth: %d, HalfPrecision:%d", outputWidth, outputHeight, outputDepth, _desc.preferHp);

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
    if (input0Depth <= 1) {
        shaderHeader += "#define INPUT0_TEXTURE_2D\n";
    }
    if (input1Depth <= 1) {
        shaderHeader += "#define INPUT1_TEXTURE_2D\n";
    }

    SNN_LOGD("numInputPlanes: %d; numOutputPlanes: %d", _desc.numInputPlanes, _desc.numOutputPlanes);

    std::string shaderUniforms = "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"

                            "#ifdef INPUT0_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput0;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput0;\n"
                            "#endif\n"

                            "#ifdef INPUT1_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=1) readonly uniform PRECISION image2D uInput1;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=1) readonly uniform PRECISION image2DArray uInput1;\n"
                            "#endif\n";

    std::string shaderMain = loadShader(CONCATENATION_CS_ASSET_NAME);

    uint32_t oc_4 = input0Depth + input1Depth; // UP_DIV(_desc.numOutputPlanes, unit);
    SNN_LOGD("oc_4: %d", oc_4);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    pass.uniforms = {{"inImgDepths", glm::ivec2(input0Depth, input1Depth)}};
    pass.inputs   = {{"uInput0", 0}, {"uInput1", 1}};
    pass.source   = (shaderHeader + shaderUniforms + shaderMain);
    pass.program  = InferencePassGl::CsProgram {"uOutput",
                                                // div-by-N is determined by work group size defined CS program.
                                                {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};
    return ret;
}

// FIXME: This methods is twisted and convoluted! need some serious refactoring!
bool ConcatenateLayerGl::generateConcatGLSamplingCode(int& idxStartPlane, int nOutputChannels, std::string& uniformsDeclaration, std::set<int>& inputTextures,
                                                    std::string& calculation) const {
    uniformsDeclaration = "";
    calculation         = "";

    int idxTextureStartPlane = -1;
    int nTexturePlanes       = 0;

    auto iterCurTexture = prevLayers.begin(); // iterator to prev layers array
    int idxCurTexture   = 0;                  // index into prevLayers array

    int idxCurPlane = 0;
    for (; iterCurTexture != prevLayers.end(); ++iterCurTexture, ++idxCurTexture) {
        if (1 == (*iterCurTexture)->getDesc().numOutputPlanes) {
            nTexturePlanes       = 1;
            idxTextureStartPlane = 0;
        } else {
            nTexturePlanes = (*iterCurTexture)->getDesc().numOutputPlanes;
        }
        if (idxCurPlane + nTexturePlanes > idxStartPlane) {
            idxTextureStartPlane = idxStartPlane - idxCurPlane;
            break;
        } else {
            idxCurPlane += nTexturePlanes;
        }
    }
    SNN_ASSERT(idxTextureStartPlane >= 0);

    std::string uniformDeclaration;
    std::string varsList;
    std::string varsAssignment;
    std::string resultAssignment;
    const std::string RGBA = "rgba";

    int idxVar = 0;
    std::string varName;
    int idxPrevTexture = -1;
    for (int nSampledChannels = 0;;) {
        std::string textureUniformName; //(buf);
        if (idxCurTexture == 0) {
            textureUniformName = "inputTextures";
        } else {
            textureUniformName = formatString("inputTextures%d", idxCurTexture);
        }

        int nCurSampledChannels = 1;
        if (1 == (*iterCurTexture)->getDesc().numOutputPlanes) {
            varName = formatString("in1_%d", idxVar);

            if (idxCurTexture != idxPrevTexture) {
                uniformDeclaration = "uniform sampler2D " + textureUniformName + ";";
            }

            std::string varAssignment = "float " + varName + " = texelFetch(" + textureUniformName + ", uv, 0).r;\n";

            varsAssignment += varAssignment + "\n";

            ++nSampledChannels;
        } else {
            int idxTextureStartT     = idxTextureStartPlane / 4;
            int nTextureStartChannel = idxTextureStartPlane % 4;
            int nTextureChannelsLeft = 4 - nTextureStartChannel;
            nCurSampledChannels      = std::min(nOutputChannels - nSampledChannels, nTextureChannelsLeft);
            int nVarSize             = nCurSampledChannels;

            varName = formatString("in%d_%d", nVarSize, idxVar);
            std::string varType;
            if (nVarSize > 1) {
                varType = formatString("FLOAT_PRECISION vec%d", nVarSize);
            } else {
                varType = "float";
            }
            std::string samplingSubset = RGBA.substr(nTextureStartChannel, nCurSampledChannels);

            if (idxCurTexture != idxPrevTexture) {
                if ((*iterCurTexture)->getDesc().numOutputPlanes > 4) {
                    uniformDeclaration = "uniform sampler2DArray " + textureUniformName + ";";
                } else {
                    uniformDeclaration = "uniform sampler2D " + textureUniformName + ";";
                }
            }

            std::string varAssignment;
            if ((*iterCurTexture)->getDesc().numOutputPlanes > 4) {
                varAssignment = varType + " " + varName + " = texelFetch(" + textureUniformName + ", ivec3(uv," + std::to_string(idxTextureStartT) + "), 0)." +
                                samplingSubset + ";";
            } else {
                SNN_ASSERT(0 == idxTextureStartT);
                varAssignment = varType + " " + varName + " = texelFetch(" + textureUniformName + ", ivec2(uv), 0)." + samplingSubset + ";";
            }

            varsAssignment += varAssignment + "\n";

            nSampledChannels += nCurSampledChannels;
        }

        inputTextures.insert(idxCurTexture);

        if (idxCurTexture != idxPrevTexture) {
            uniformsDeclaration += uniformDeclaration + "\n";
        }

        varsList += (varsList.empty() ? "" : ",") + varName;

        SNN_ASSERT(nSampledChannels <= nOutputChannels);
        if (nSampledChannels == nOutputChannels) {
            break;
        }

        ++idxVar;
        idxPrevTexture = idxCurTexture;
        if (idxTextureStartPlane + nCurSampledChannels < nTexturePlanes) {
            idxTextureStartPlane += nCurSampledChannels;
        } else {
            SNN_ASSERT(idxTextureStartPlane + nCurSampledChannels == nTexturePlanes);
            ++iterCurTexture;
            ++idxCurTexture;
            nTexturePlanes       = (*iterCurTexture)->getDesc().numOutputPlanes;
            idxTextureStartPlane = 0;
        }
    }

    idxStartPlane += nOutputChannels;

    for (int i = nOutputChannels; i < 4; ++i) {
        varsList = varsList + ",0.0f";
    }

    resultAssignment = "s = vec4(" + varsList + ");\n";

    calculation = varsAssignment + resultAssignment;

    return true;
}
