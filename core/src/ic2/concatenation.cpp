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
#include "dp.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <ios>

using namespace snn;
using namespace snn::dp;
using namespace std;

static constexpr const char* CONCATENATION_FS_ASSET_NAME = "shaders/shadertemplate_fs_concat.glsl";
static constexpr const char* CONCATENATION_CS_ASSET_NAME = "shaders/shadertemplate_cs_concat.glsl";

ShaderLayer::GLSLShaders ConcatenateLayer::createFS(const LayerGenOptions&) const {
    GLSLShaders ret;

    auto& desc = getDesc();

    int numShaderPasses = (int) DIV_4_ROUND_UP(desc.numOutputPlanes);

    std::string preDefine          = "#version 320 es\n";
    std::string shaderTemplateCode = loadShader(CONCATENATION_FS_ASSET_NAME);

    int idxStartPlane = 0;

    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(numShaderPasses);
    for (int i = 0, j = 0; i < numShaderPasses; i++, j += 4) {
        int outputChannels = min(4, (int) desc.numOutputPlanes - j);

        // Create a copy of the template code.
        // After modification, this will contain the shader's true source code.
        std::string fsCode = shaderTemplateCode;
        string uniformsDeclaration;
        string calculation;
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
            ret.passes[i].inputs[s] = (uint32_t) t;
        }

        InferenceGraph::Pass& pass = passes[i];
        pass.source                = preDefine + fsCode;
        pass.program               = InferenceGraph::Pass::FsProgram {(uint32_t) i, (uint32_t) DIV_4_ROUND_UP(outputChannels)};
    }
    return ret;
}

ShaderLayer::GLSLShaders ConcatenateLayer::createCS(const LayerGenOptions& options) const {
    (void) options;

    GLSLShaders ret;

    std::vector<InferenceGraph::Pass>& passes = ret.passes;
    passes.resize(1);

    InferenceGraph::Pass& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t input0Depth = inputDims[0].depth;
    uint32_t input1Depth = inputDims[1].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0; // FIXME: outputDepth cannot get right value

    snn::dp::GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    SNN_LOGD("outputWidth: %d; outputHeight: %d; outputDepth: %d, HalfPrecision:%d", outputWidth, outputHeight, outputDepth, _desc.preferHp);

    string shaderHeader;
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

    SNN_LOGD("%s:%d; numInputPlanes: %d; numOutputPlanes: %d", __FUNCTION__, __LINE__, _desc.numInputPlanes, _desc.numOutputPlanes);

    //    if (!desc.activation.compare("relu")) { shaderHeader += "#define RELU\n";  SNN_LOGI("%s:%d; activation func: relu", __FUNCTION__,__LINE__);}
    //    if (!desc.activation.compare("relu6"))  shaderHeader += "#define RELU6\n";
    //    if (!desc.activation.compare("tanh"))  shaderHeader += "#define TANH\n";
    //    if (!desc.activation.compare("sigmoid"))  shaderHeader += "#define SIGMOID\n";
    //    if (!desc.activation.compare("leakyRelu"))  shaderHeader += ("#define LEAKYRELU_VAL " + std::to_string(desc.leakyReluAlpha) + "\n");
    //    if (!desc.activation.compare("SiLU"))  shaderHeader += "#define SILU\n";

    string shaderUniforms = "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"

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

    string shaderMain =

        "layout(location=1) uniform ivec2 inImgDepths;\n"

        // work group size
        "layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;\n"
        "void main()\n"
        "{\n"
        "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
        "    \n"
        "    vec4 outValue;\n"
        "    vec4 outValue1;\n"
        "    if(pos.z < inImgDepths.x)\n"
        "    {\n"
        "        #ifdef INPUT0_TEXTURE_2D\n"
        "        outValue = imageLoad(uInput0, ivec2(pos.x, pos.y));\n"
        "        #else\n"
        "        outValue = imageLoad(uInput0, ivec3(pos.x, pos.y, pos.z));\n"
        "        #endif\n"
        "    }\n"
        "    else\n"
        "    {\n"
        "        #ifdef INPUT1_TEXTURE_2D\n"
        "        outValue = imageLoad(uInput1, ivec2(pos.x, pos.y));\n"
        "        #else\n"
        "        outValue = imageLoad(uInput1, ivec3(pos.x, pos.y, pos.z - inImgDepths.x));\n"
        "        #endif\n"
        "    }\n"
        "    \n"
        //            "    #ifdef RELU\n"
        //            "    outValue = max(outValue, vec4(0));\n"
        //            "    #endif\n"
        //            "    #ifdef RELU6\n"
        //            "    outValue = clamp(outValue, vec4(0), vec4(6));\n"
        //            "    #endif\n"
        //            "    #ifdef TANH\n"
        //            "    outValue = tanh(outValue);\n"
        //            "    #endif\n"
        //            "    #ifdef SIGMOID\n"
        //            "    outValue  = vec4(1.0f)/(vec4(1.0f)+ exp(-outValue));\n"
        //            "    #endif\n"
        //            "    #ifdef LEAKYRELU_VAL\n"
        //            "    outValue   = max(outValue,  (outValue * vec4(LEAKYRELU_VAL)));\n"
        //            "    #endif\n"
        //            "    #ifdef SILU\n"
        //            "    outValue    = outValue  * vec4(1.0f)/(vec4(1.0f)+ exp(-outValue));\n"
        //            "    #endif\n"
        "    imageStore(uOutput, pos, outValue);\n"
        "}\n";
    shaderMain = loadShader(CONCATENATION_CS_ASSET_NAME);

    uint32_t oc_4 = input0Depth + input1Depth; // UP_DIV(_desc.numOutputPlanes, unit);
    SNN_LOGD("oc_4: %d", oc_4);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    pass.uniforms = {{"inImgDepths", glm::ivec2(input0Depth, input1Depth)}};
    pass.inputs   = {{"uInput0", 0}, {"uInput1", 1}};
    pass.source   = (shaderHeader + shaderUniforms + shaderMain);
    pass.program  = InferenceGraph::Pass::CsProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};
    return ret;
}

// FIXME: This methods is twisted and convoluted! need some serious refactoring!
bool ConcatenateLayer::generateConcatGLSamplingCode(int& idxStartPlane, int nOutputChannels, string& uniformsDeclaration, std::set<int>& inputTextures,
                                                    string& calculation) const {
    uniformsDeclaration = "";
    calculation         = "";

    int idxTextureStartPlane = -1;
    int nTexturePlanes       = 0;

    // auto iterCurTexture = _inputTextureContentsMap.begin()
    auto iterCurTexture = prevLayers.begin(); // iterator to prev layers array
    int idxCurTexture   = 0;                  // index into prevLayers array

    int idxCurPlane = 0;
    // for (; iterCurTexture != _inputTextureContentsMap.end(); ++ iterCurTexture, ++ idxCurTexture) {
    for (; iterCurTexture != prevLayers.end(); ++iterCurTexture, ++idxCurTexture) {
        // if (iterCurTexture->second->is2D()) {
        if (1 == (*iterCurTexture)->getDesc().numOutputPlanes) {
            // single plane output
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

    string uniformDeclaration;
    string varsList;
    string varsAssignment;
    string resultAssignment;
    // char buf[128] = {0};
    const string RGBA = "rgba";

    int idxVar = 0;
    string varName;
    int idxPrevTexture = -1;
    for (int nSampledChannels = 0;;) {
        string textureUniformName; //(buf);
        if (idxCurTexture == 0) {
            // snprintf(buf, sizeof(buf) - 1, "inputTextures");
            textureUniformName = "inputTextures";
        } else {
            // snprintf(buf, sizeof(buf) - 1, "inputTextures%d", idxCurTexture);
            textureUniformName = formatString("inputTextures%d", idxCurTexture);
        }

        int nCurSampledChannels = 1;
        // const shared_ptr<Texture>& curTexInfo = iterCurTexture->second;
        // if (curTexInfo->is2D()) {
        if (1 == (*iterCurTexture)->getDesc().numOutputPlanes) {
            // Currently we support only 1 channel 2D textures
            varName = formatString("in1_%d", idxVar);

            if (idxCurTexture != idxPrevTexture) {
                uniformDeclaration = "uniform sampler2D " + textureUniformName + ";";
                // LOGI("%s", uniformDeclaration.c_str());
            }

            string varAssignment = "float " + varName + " = texelFetch(" + textureUniformName + ", uv, 0).r;\n";
            // LOGI("%s", varAssignment.c_str());

            varsAssignment += varAssignment + "\n";

            ++nSampledChannels;
        } else {
            int idxTextureStartT     = idxTextureStartPlane / 4;
            int nTextureStartChannel = idxTextureStartPlane % 4;
            int nTextureChannelsLeft = 4 - nTextureStartChannel;
            nCurSampledChannels      = min(nOutputChannels - nSampledChannels, nTextureChannelsLeft);
            int nVarSize             = nCurSampledChannels;

            // snprintf(buf, sizeof(buf) - 1, "in%d_%d", nVarSize, idxVar);
            varName = formatString("in%d_%d", nVarSize, idxVar);
            string varType;
            if (nVarSize > 1) {
                // snprintf(buf, sizeof(buf) - 1, "FLOAT_PRECISION vec%d", nVarSize);
                varType = formatString("FLOAT_PRECISION vec%d", nVarSize);
            } else {
                varType = "float";
            }
            // snprintf(buf, sizeof(buf) - 1, "%d", idxTextureStartT);
            // string t(buf);
            string samplingSubset = RGBA.substr(nTextureStartChannel, nCurSampledChannels);

            if (idxCurTexture != idxPrevTexture) {
                if ((*iterCurTexture)->getDesc().numOutputPlanes > 4) {
                    uniformDeclaration = "uniform sampler2DArray " + textureUniformName + ";";
                } else {
                    uniformDeclaration = "uniform sampler2D " + textureUniformName + ";";
                }
                // LOGI("%s", uniformDeclaration.c_str());
            }

            string varAssignment;
            if ((*iterCurTexture)->getDesc().numOutputPlanes > 4) {
                varAssignment = varType + " " + varName + " = texelFetch(" + textureUniformName + ", ivec3(uv," + std::to_string(idxTextureStartT) + "), 0)." +
                                samplingSubset + ";";
            } else {
                SNN_ASSERT(0 == idxTextureStartT);
                varAssignment = varType + " " + varName + " = texelFetch(" + textureUniformName + ", ivec2(uv), 0)." + samplingSubset + ";";
            }
            // LOGI("%s", varAssignment.c_str());

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
            // if (iterCurTexture->second->is2D()) {
            //     nTexturePlanes = 1;
            // }
            // else {
            //     int idxConnection = iterCurTexture->first / MAX_NUM_TEXTURES_PER_LAYER - 1;
            //     SNN_ASSERT(idxConnection > 0);
            //     SNN_ASSERT(idxConnection < (int) desc._input.size());
            //     nTexturePlanes = desc._input[idxConnection]->getNumOutputPlanes();
            // }
            nTexturePlanes       = (*iterCurTexture)->getDesc().numOutputPlanes;
            idxTextureStartPlane = 0;
        }
    }

    idxStartPlane += nOutputChannels;

    for (int i = nOutputChannels; i < 4; ++i) {
        varsList = varsList + string(",0.0f");
    }

    resultAssignment = "s = vec4(" + varsList + ");\n";
    // LOGI("%s", resultAssignment.c_str());

    calculation = varsAssignment + resultAssignment;

    return true;
}

void ConcatenateLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depthOut) const {
    width    = inputDims[0].width;
    height   = inputDims[0].height;
    depthOut = inputDims[0].depth + inputDims[1].depth;
    //    SNN_LOGI("output width: %d; height: %d; depth0: %d; depth1: %d; depthOut: %d", (int)width, (int)height, (int)inputDims[0].depth, inputDims[1].depth,
    //    depthOut);
}
