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
#include "openGLRenderpass.h"
#include "dp.h"
#include "inferencepassGL.h"
#include "snn/core.h"
#include "imageTextureGL.h"
#include <string>
#include <vector>
#include <variant>
#include <utility>

// -----------------------------------------------------------------------------
//
snn::OpenGLRenderPass::OpenGLRenderPass(const snn::OpenGLRenderPass::CreationParameters& cp)
    : _cp(cp)
{
    SNN_LOGD("Render pass created: %s", snn::ImageTextureGLArrayAccessor(_cp.texOutputs)[0].getTextureInfo2().c_str());
    _quad.allocate();

    // create program
    _program.name = cp.name;
    if (isCompute()) {
        if (!_program.loadCs(cp.pass.source.c_str())) {
            return;
        }
    } else {
        const char* vscode = R"glsl(#version 320 es
            out vec2 v_uv;
            void main()
            {
                const vec4 v[] = vec4[](
                    vec4(-1., -1.,  1.,  1.),
                    vec4( 3., -1.,  1., -1.),
                    vec4(-1.,  3., -1.,  1.));
                gl_Position = vec4(v[gl_VertexID].xy, 0., 1.);
                v_uv = v[gl_VertexID].zw;
            }
        )glsl";
        if (!_program.loadVsPs(vscode, cp.pass.source.c_str())) {
            return;
        }
    }

    // query all uniform locations.
    for (auto& [name, value] : cp.pass.uniforms) {
        gl::SimpleUniform un(name, value);
        if (un.init(_program)) {
            _uniforms.push_back(un);
        } else {
            SNN_LOGE("Uniform %s in %s not found. %s", name.c_str(), cp.name.c_str(), cp.pass.source.c_str());
        }
    }

    // query all run time uniforms.
    for (auto& [name, value] : cp.pass.runtimeUniforms) {
        (void) value;
        gl::SimpleUniform un(name, 0);
        if (un.init(_program)) {
            _runtimeUniforms.push_back(un);
        } else {
            SNN_LOGE("Uniform %s in %s not found. %s", name.c_str(), cp.name.c_str(), cp.pass.source.c_str());
        }
    }

    int numUniforms;
    glGetProgramiv(_program, GL_ACTIVE_UNIFORMS, &numUniforms);
    SNN_LOGD("Number of active uniforms in the program: %d", numUniforms);
    for (int i = 0; i < numUniforms; i++) {
        GLenum type  = GL_ZERO;
        GLint length = 0, size = 0;
        char name[128];
        glGetActiveUniform(_program, (GLuint) i, 128, &length, &size, &type, name);
        SNN_LOGD("%d. %s (%d) (%d)", i + 1, name, type, size);
    }

    if (_cp.pass.weightMeta.size() > 0) {
        uint32_t layout = _cp.pass.weightMeta[0];
        if (isCompute()) {
            uint32_t weightMethod = _cp.pass.weightMeta[1];
            uint32_t fp16 = _cp.pass.weightMeta[2];
            uint32_t kernelW = _cp.pass.weightMeta[3];
            uint32_t kernelH = _cp.pass.weightMeta[4];
            uint32_t numInputPlanes = _cp.pass.weightMeta[5];
            uint32_t numOutputPlanes = _cp.pass.weightMeta[6];
            initGLCSData(weightMethod, fp16, kernelW, kernelH, numInputPlanes, numOutputPlanes);
        } else {
            uint32_t weightMethod = _cp.pass.weightMeta[1];
            uint32_t fp16 = _cp.pass.weightMeta[2];
            uint32_t kernelW = _cp.pass.weightMeta[3];
            uint32_t kernelH = _cp.pass.weightMeta[4];
            uint32_t numInputPlanes = _cp.pass.weightMeta[5];
            uint32_t numOutputPlanes = _cp.pass.weightMeta[6];
            uint32_t channelsPerPass = _cp.pass.weightMeta[7];
            uint32_t fsPlaneIndex = _cp.pass.weightMeta[8];
            if (layout == 0) {  //Conv2D
                initGLFSData(weightMethod, fp16, kernelW, kernelH, numInputPlanes, numOutputPlanes, channelsPerPass, fsPlaneIndex);
                snn::WeightAccessMethod weightMode = (snn::WeightAccessMethod) weightMethod;
                if (snn::WeightAccessMethod::TEXTURES == weightMode) {
                    setTextureWeights(weightMethod, fp16, kernelW, kernelH, numInputPlanes, numOutputPlanes, channelsPerPass, fsPlaneIndex);
                } else {
                    setBufferWeights(weightMethod, fp16, kernelW, kernelH, numInputPlanes, numOutputPlanes, channelsPerPass, fsPlaneIndex);
                }
            } else if (layout == 1) { //Conv2D DepthWise
                initGLFSDataDW(weightMethod, fp16, kernelW, kernelH, numInputPlanes, numOutputPlanes, channelsPerPass, fsPlaneIndex);
                snn::WeightAccessMethod weightMode = (snn::WeightAccessMethod) weightMethod;
                if (snn::WeightAccessMethod::TEXTURES == weightMode) {
                    setTextureWeightsDW(weightMethod, fp16, kernelW, kernelH, numInputPlanes, numOutputPlanes, channelsPerPass, fsPlaneIndex);
                } else {
                    setBufferWeightsDW(weightMethod, fp16, kernelW, kernelH, numInputPlanes, numOutputPlanes, channelsPerPass, fsPlaneIndex);
                }
            }
        }
    }
}

void snn::OpenGLRenderPass::initGLFSData(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
    uint32_t numInputPlanes, uint32_t numOutputPlanes, uint32_t channelsPerPass, uint32_t fsPlaneIndex) {

    uint32_t kernelSize = (uint32_t) kernelW;
    (void) kernelH;
    snn::WeightAccessMethod weightMode = (snn::WeightAccessMethod) weightMethod;
    bool preferHp = (bool) fp16;

    uint32_t outputPlanes = DIV_4_ROUND_UP(channelsPerPass);
    uint32_t passIndex = fsPlaneIndex / outputPlanes;
    uint32_t outputChannels = std::min(channelsPerPass, numOutputPlanes -  passIndex * channelsPerPass);

    switch (weightMode) {
    case snn::WeightAccessMethod::TEXTURES:
        _weightTextures.allocate(outputChannels);
        break;

    case snn::WeightAccessMethod::UNIFORM_BUFFER:
        _weightUniformBuffers.allocate(outputChannels);
        break;

    case snn::WeightAccessMethod::SSBO_BUFFER:
        _weightSSBOBuffers.allocate(outputChannels);
        break;

    default:
        break;
    }

    snn::ColorFormat weightFormat;
    if (preferHp) {
        weightFormat = snn::ColorFormat::RGBA16F;
    } else {
        weightFormat = snn::ColorFormat::RGBA32F;
    }

    for (std::size_t i = 0; i < outputChannels; i++) {
        std::size_t depth = std::max(DIV_4_ROUND_UP(numInputPlanes), (uint32_t) 1);

        switch (weightMode) {
        case snn::WeightAccessMethod::TEXTURES: {
            if (depth == 1) {
                _weightTextures[i].allocate2D(weightFormat, kernelSize, kernelSize);
            } else {
                _weightTextures[i].allocate2DArray(weightFormat, kernelSize, kernelSize, depth);
            }
            break;
        }

        case snn::WeightAccessMethod::UNIFORM_BUFFER: {
            uint32_t count = 4 * kernelSize * kernelSize * numInputPlanes;
            if (preferHp) {
                std::vector<uint16_t> dummyVal(count, 0);
                _weightUniformBuffers[i].allocate(count, dummyVal.data());
            } else {
                std::vector<float> dummyVal(count, 0.0f);
                _weightUniformBuffers[i].allocate(count, dummyVal.data());
            }
            break;
        }

        case snn::WeightAccessMethod::SSBO_BUFFER: {
            uint32_t count = 4 * kernelSize * kernelSize * numInputPlanes;
            if (preferHp) {
                std::vector<uint16_t> dummyVal(count, 0);
                _weightSSBOBuffers[i].allocate(count, dummyVal.data());
            } else {
                std::vector<float> dummyVal(count, 0.0f);
                _weightSSBOBuffers[i].allocate(count, dummyVal.data());
            }
            break;
        }

        default:
            break;
        }
    }

    switch (weightMode) {
    case snn::WeightAccessMethod::TEXTURES:
        _weights = std::vector<const gl::TextureObject*>();
        break;

    case snn::WeightAccessMethod::UNIFORM_BUFFER:
        _weights = std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>();
        break;

    case snn::WeightAccessMethod::SSBO_BUFFER:
        _weights = std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>();
        break;

    default:
        break;
    }
    std::visit(match {[&](std::vector<const gl::TextureObject*>& weightTextures) {
                        for (std::size_t k = 0; k < outputChannels; k++) {
                            weightTextures.push_back(&_weightTextures[k]);
                        }
                    },
                    [&](std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>& weightBuffers) {
                        for (std::size_t k = 0; k < outputChannels; k++) {
                            weightBuffers.push_back(&_weightUniformBuffers[k]);
                        }
                    },
                    [&](std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>& weightBuffers) {
                        for (std::size_t k = 0; k < outputChannels; k++) {
                            weightBuffers.push_back(&_weightSSBOBuffers[k]);
                        }
                    }},
                _weights);
}

void snn::OpenGLRenderPass::setTextureWeights(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
    uint32_t numInputPlanes, uint32_t numOutputPlanes, uint32_t channelsPerPass, uint32_t fsPlaneIndex) const {

    uint32_t kernelSize = (uint32_t) kernelW;
    (void) kernelH;
    (void) weightMethod;
    bool preferHp = (bool) fp16;

    uint32_t outputPlanes = DIV_4_ROUND_UP(channelsPerPass);
    uint32_t passIndex = fsPlaneIndex/outputPlanes;
    uint32_t outputChannels = std::min(channelsPerPass, numOutputPlanes -  passIndex * channelsPerPass);

    for (std::size_t filter = 0; filter < outputChannels; filter++) {
        std::vector<float> weightVal(4 * kernelSize * kernelSize, 0.0);
        for (std::size_t filterPlane = 0; filterPlane < numInputPlanes; filterPlane++) {
            uint32_t outputChannel = filter;
            std::size_t idx = outputChannel * numInputPlanes + filterPlane;
            for (std::size_t i = 0; i < kernelSize; i++) {
                for (std::size_t j = 0; j < kernelSize; j++) {
                    std::size_t weightValIdx = (4 * kernelSize * i) + (4 * j) + (filterPlane % 4);
                    if (preferHp) {
                        uint16_t* fp16Addr = (uint16_t*)weightVal.data();
                        *(fp16Addr + weightValIdx) = FP32::toHalf(_cp.pass.modelWeights[idx].at<float>(i, j));
                    } else {
                        weightVal[weightValIdx] = _cp.pass.modelWeights[idx].at<float>(i, j);
                    }
                }
            }
            if ((filterPlane + 1) % 4 == 0) {
                if (numInputPlanes > 4) {
                    _weightTextures[filter].bind(0);
                    _weightTextures[filter].setPixels((int) (filterPlane / 4), 0, 0, 0, kernelSize, kernelSize, 0, weightVal.data());
                    glFinish();
                    _weightTextures[filter].unbind();
                } else {
                    _weightTextures[filter].bind(0);
                    _weightTextures[filter].setPixels(0, 0, 0, kernelSize, kernelSize, 0, weightVal.data());
                    glFinish();
                    _weightTextures[filter].unbind();
                }
            }
        }
        if (!weightVal.empty() && numInputPlanes % 4 != 0) {
            if (numInputPlanes > 4) {
                _weightTextures[filter].bind(0);
                _weightTextures[filter].setPixels((int) (numInputPlanes / 4), 0, 0, 0, kernelSize, kernelSize, 0, weightVal.data());
                glFinish();
                _weightTextures[filter].unbind();
            } else {
                _weightTextures[filter].bind(0);
                _weightTextures[filter].setPixels(0, 0, 0, kernelSize, kernelSize, 0, weightVal.data());
                glFinish();
                _weightTextures[filter].unbind();
            }
        }
    }
}

void snn::OpenGLRenderPass::setBufferWeights(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
    uint32_t numInputPlanes, uint32_t numOutputPlanes, uint32_t channelsPerPass, uint32_t fsPlaneIndex) const {

    uint32_t kernelSize = (uint32_t) kernelW;
    (void) kernelH;
    snn::WeightAccessMethod weightMode = (snn::WeightAccessMethod) weightMethod;
    bool preferHp = (bool) fp16;

    uint32_t outputPlanes = DIV_4_ROUND_UP(channelsPerPass);
    uint32_t passIndex = fsPlaneIndex/outputPlanes;
    uint32_t outputChannels = std::min(channelsPerPass, numOutputPlanes -  passIndex * channelsPerPass);

    uint8_t byteSize = preferHp ? 2 : 4;
    for (std::size_t filter = 0; filter < outputChannels; filter++) {
        std::vector<uint8_t> weightVal(byteSize * 4 * numInputPlanes * kernelSize * kernelSize, 0);
        for (std::size_t filterPlane = 0; filterPlane < numInputPlanes; filterPlane++) {
            uint32_t outputChannel = passIndex * channelsPerPass + filter;
            std::size_t idx = outputChannel * numInputPlanes + filterPlane;

            for (std::size_t i = 0; i < kernelSize; i++) {
                for (std::size_t j = 0; j < kernelSize; j++) {
                    std::size_t weightValIdx = (byteSize * 4 * kernelSize * i) + (4 * j) + (filterPlane % 4);
                    std::vector<uint8_t> byteRep;
                    snn::getByteRepresentation(_cp.pass.modelWeights[idx].at<float>(i, j), byteRep, preferHp);
                    for (std::size_t byteIdx = 0; byteIdx < byteSize; byteIdx++) {
                        weightVal[weightValIdx + byteIdx] = byteRep.at(byteIdx);
                    }
                }
            }
        }
        switch (weightMode) {
        case snn::WeightAccessMethod::UNIFORM_BUFFER:
            _weightUniformBuffers[filter].update(weightVal.data(), 0, weightVal.size());
            break;

        case snn::WeightAccessMethod::SSBO_BUFFER:
            _weightSSBOBuffers[filter].update(weightVal.data(), 0, weightVal.size());
            break;

        default:
            break;
        }
    }
}

void snn::OpenGLRenderPass::initGLFSDataDW(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
    uint32_t numInputPlanes, uint32_t numOutputPlanes, uint32_t channelsPerPass, uint32_t fsPlaneIndex) {

    uint32_t kernelSize = (uint32_t) kernelW;
    (void) kernelH;
    snn::WeightAccessMethod weightMode = (snn::WeightAccessMethod) weightMethod;
    bool preferHp = (bool) fp16;

    uint32_t outputPlanes = DIV_4_ROUND_UP(channelsPerPass);
    uint32_t passIndex = fsPlaneIndex/outputPlanes;
    uint32_t outputChannels = std::min(channelsPerPass, numOutputPlanes -  passIndex * channelsPerPass);
    (void) numInputPlanes;
    (void) outputChannels;

    snn::ColorFormat weightFormat;
    if (preferHp) {
        weightFormat = snn::ColorFormat::RGBA16F;
    } else {
        weightFormat = snn::ColorFormat::RGBA32F;
    }

    switch (weightMode) {
    case snn::WeightAccessMethod::TEXTURES:
        _weightTextures.allocate(DIV_4_ROUND_UP(outputChannels));
        break;

    case snn::WeightAccessMethod::UNIFORM_BUFFER:
        _weightUniformBuffers.allocate(DIV_4_ROUND_UP(outputChannels));
        break;

    case snn::WeightAccessMethod::SSBO_BUFFER:
        _weightSSBOBuffers.allocate(DIV_4_ROUND_UP(outputChannels));
        break;

    default:
        break;
    }

    for (std::size_t i = 0; i < DIV_4_ROUND_UP(outputChannels); i++) {
        switch (weightMode) {
        case snn::WeightAccessMethod::TEXTURES:
            _weightTextures[i].allocate2D(weightFormat, kernelSize, kernelSize, 1);
            break;

        case snn::WeightAccessMethod::UNIFORM_BUFFER: {
            uint32_t count = 4 * kernelSize * kernelSize;
            if (preferHp) {
                std::vector<uint16_t> dummyVal(count);
                _weightUniformBuffers[i].allocate(count, dummyVal.data());
            } else {
                std::vector<float> dummyVal(count);
                _weightUniformBuffers[i].allocate(count, dummyVal.data());
            }
            break;
        }

        case snn::WeightAccessMethod::SSBO_BUFFER: {
            uint32_t count = 4 * kernelSize * kernelSize;
            if (preferHp) {
                std::vector<uint16_t> dummyVal(count);
                _weightSSBOBuffers[i].allocate(count, dummyVal.data());
            } else {
                std::vector<float> dummyVal(count);
                _weightSSBOBuffers[i].allocate(count, dummyVal.data());
            }
            break;
        }

        default:
            break;
        }
    }
    switch (weightMode) {
    case snn::WeightAccessMethod::TEXTURES:
        _weights = std::vector<const gl::TextureObject*>();
        break;

    case snn::WeightAccessMethod::UNIFORM_BUFFER:
        _weights = std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>();
        break;

    case snn::WeightAccessMethod::SSBO_BUFFER:
        _weights = std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>();
        break;

    default:
        break;
    }
    std::visit(match {[&](std::vector<const gl::TextureObject*>& weightTextures) {
                        for (std::size_t k = 0; k < DIV_4_ROUND_UP(outputChannels); k++) {
                            weightTextures.push_back(&_weightTextures[k]);
                        }
                    },
                    [&](std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>& weightBuffers) {
                        for (std::size_t k = 0; k < DIV_4_ROUND_UP(outputChannels); k++) {
                            weightBuffers.push_back(&_weightUniformBuffers[k]);
                        }
                    },
                    [&](std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>& weightBuffers) {
                        for (std::size_t k = 0; k < DIV_4_ROUND_UP(outputChannels); k++) {
                            weightBuffers.push_back(&_weightSSBOBuffers[k]);
                        }
                    }},
                _weights);
}

void snn::OpenGLRenderPass::setTextureWeightsDW(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
    uint32_t numInputPlanes, uint32_t numOutputPlanes, uint32_t channelsPerPass, uint32_t fsPlaneIndex) const {

    uint32_t kernelSize = (uint32_t) kernelW;
    (void) kernelH;
    (void) weightMethod;
    bool preferHp = (bool) fp16;

    uint32_t outputPlanes = DIV_4_ROUND_UP(channelsPerPass);
    uint32_t passIndex = fsPlaneIndex/outputPlanes;
    uint32_t outputChannels = std::min(channelsPerPass, numOutputPlanes -  passIndex * channelsPerPass);
    (void) numInputPlanes;

    std::vector<float> weightVal(4 * kernelSize * kernelSize, 0.0);
    for (std::size_t filter = 0; filter < outputChannels; filter++) {
        for (std::size_t i = 0; i < kernelSize; i++) {
            for (std::size_t j = 0; j < kernelSize; j++) {
                std::size_t weightValIdx = (4 * kernelSize * i) + (4 * j) + (filter % 4);
                if (preferHp) {
                    uint16_t* fp16Addr = (uint16_t*)weightVal.data();
                    *(fp16Addr + weightValIdx) = FP32::toHalf(_cp.pass.modelWeights[filter].at<float>(i, j));
                } else {
                    weightVal[weightValIdx] = _cp.pass.modelWeights[filter].at<float>(i, j);
                }
            }
        }
        if ((filter + 1) % 4 == 0) {
            _weightTextures[filter / 4].bind(0);
            _weightTextures[filter / 4].setPixels(0, 0, 0, kernelSize, kernelSize, 0, weightVal.data());
            glFinish();
            _weightTextures[filter / 4].unbind();
            weightVal.clear();
            weightVal.resize(kernelSize * kernelSize * 4, 0.0);
        }
    }
    if (!weightVal.empty() && outputChannels % 4 != 0) {
        _weightTextures[DIV_4_ROUND_UP(outputChannels)].bind(0);
        _weightTextures[DIV_4_ROUND_UP(outputChannels)].setPixels(0, 0, 0, kernelSize, kernelSize, 0, weightVal.data());
        glFinish();
        _weightTextures[DIV_4_ROUND_UP(outputChannels)].unbind();
    }
}

void snn::OpenGLRenderPass::setBufferWeightsDW(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
    uint32_t numInputPlanes, uint32_t numOutputPlanes, uint32_t channelsPerPass, uint32_t fsPlaneIndex) const {

    uint32_t kernelSize = (uint32_t) kernelW;
    (void) kernelH;
    snn::WeightAccessMethod weightMode = (snn::WeightAccessMethod) weightMethod;
    bool preferHp = (bool) fp16;

    uint32_t outputPlanes = DIV_4_ROUND_UP(channelsPerPass);
    uint32_t passIndex = fsPlaneIndex/outputPlanes;
    uint32_t outputChannels = std::min(channelsPerPass, numOutputPlanes -  passIndex * channelsPerPass);
    (void) numInputPlanes;
    (void) outputChannels;

    uint8_t byteSize = preferHp ? 2 : 4;
    std::vector<uint8_t> weightVal(byteSize * 4 * kernelSize * kernelSize, 0);
    for (std::size_t filter = 0; filter < outputChannels; filter++) {
        for (std::size_t i = 0; i < kernelSize; i++) {
            for (std::size_t j = 0; j < kernelSize; j++) {
                std::size_t weightValIdx = (byteSize * 4 * kernelSize * i) + (4 * j) + (filter % 4);
                std::vector<uint8_t> byteRep;
                snn::getByteRepresentation(_cp.pass.modelWeights[filter].at<float>(j, i), byteRep, preferHp);
                for (std::size_t byteIdx = 0; byteIdx < byteSize; byteIdx++) {
                    weightVal[weightValIdx + byteIdx] = byteRep.at(byteIdx);
                }
            }
        }
        if ((filter + 1) % 4 == 0) {
            switch (weightMode) {
            case snn::WeightAccessMethod::UNIFORM_BUFFER:
                _weightUniformBuffers[filter / 4].update(weightVal.data(), 0, weightVal.size());
                break;

            case snn::WeightAccessMethod::SSBO_BUFFER:
                _weightSSBOBuffers[filter / 4].update(weightVal.data(), 0, weightVal.size());
                break;

            default:
                break;
            }
            weightVal.clear();
        }
    }
    if (!weightVal.empty() && outputChannels % 4 != 0) {
        switch (weightMode) {
        case snn::WeightAccessMethod::UNIFORM_BUFFER:
            _weightUniformBuffers[DIV_4_ROUND_UP(outputChannels)].update(weightVal.data(), 0, weightVal.size());
            break;

        case snn::WeightAccessMethod::SSBO_BUFFER:
            _weightSSBOBuffers[DIV_4_ROUND_UP(outputChannels)].update(weightVal.data(), 0, weightVal.size());
            break;

        default:
            break;
        }
        weightVal.clear();
    }
}

void snn::OpenGLRenderPass::initGLCSData(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
    uint32_t numInputPlanes, uint32_t numOutputPlanes) {
    (void) kernelW;
    (void) kernelH;
    snn::WeightAccessMethod weightMode = (snn::WeightAccessMethod) weightMethod;
    bool preferHp = (bool) fp16;
    (void) numInputPlanes;
    (void) numOutputPlanes;
    if (weightMode == snn::WeightAccessMethod::TEXTURES) {
        snn::ColorFormat weightFormat;
        if (preferHp) {
            weightFormat = snn::ColorFormat::RGBA16F;
        } else {
            weightFormat = snn::ColorFormat::RGBA32F;
        }

        auto dims = _cp.pass.weightDims["2"];
        kernelTexture.allocate2DArray(weightFormat, dims[0], dims[1], dims[2], dims[2] * 4, 1);

        if (preferHp) {
            uint16_t* kernelBuf = (uint16_t*)_cp.pass._vecWeights.data();
            kernelTexture.bind(0);

            uint32_t planeSize = dims[0] * dims[1] * 4;
            for (uint32_t i = 0; i < dims[2]; i++) {
                kernelTexture.setPixels(i, 0, 0, 0, dims[0], dims[1], 0, kernelBuf + planeSize  * i);
            }
            kernelTexture.unbind();
        } else {
            float* kernelBuf = _cp.pass._vecWeights.data();

            kernelTexture.bind(0);

            uint32_t planeSize = dims[0] * dims[1] * 4;
            for (uint32_t i = 0; i < dims[2]; i++) {
                kernelTexture.setPixels(i, 0, 0, 0, dims[0], dims[1], 0, kernelBuf + planeSize * i);
            }
            kernelTexture.unbind();
        }

        _weightUniformTags.resize(1);
        _weightUniformTags[0] = "uKernel";
        _weights             = std::vector<const gl::TextureObject*>(1, &kernelTexture);
    } else {
        if (_cp.pass._vecWeights.size() > 0) {
            _boWeights.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
            _boWeights->allocate(_cp.pass._vecWeights.size(), _cp.pass._vecWeights.data());
            _ssboMap[3] = _boWeights->getId();
        }
    }

    if (_cp.pass._vecBias.size() > 0) {
        _boBias.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER, MIN_SSBO_BUFFER_LEN_ARM_MALI>());
        _boBias->allocate(_cp.pass._vecBias.size(), _cp.pass._vecBias.data());
        _ssboMap[4] = _boBias->getId();
    }

    if (_cp.pass._vecBeta.size() > 0) {
        _bnBeta.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        _bnBeta->allocate(_cp.pass._vecBeta.size(), _cp.pass._vecBeta.data());
        _ssboMap[5] = _bnBeta->getId();
    }

    if (_cp.pass._vecGamma.size() > 0) {
        _bnGamma.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        _bnGamma->allocate(_cp.pass._vecGamma.size(), _cp.pass._vecGamma.data());
        _ssboMap[6] = _bnGamma->getId();
    }

    if (_cp.pass._vecMean.size() > 0) {
        _bnMean.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        _bnMean->allocate(_cp.pass._vecMean.size(), _cp.pass._vecMean.data());
        _ssboMap[7] = _bnMean->getId();
    }

    if (_cp.pass._vecVariance.size() > 0) {
        _bnVariance.reset(new gl::BufferObject<GL_SHADER_STORAGE_BUFFER>());
        _bnVariance->allocate(_cp.pass._vecVariance.size(), _cp.pass._vecVariance.data());
        _ssboMap[8] = _bnVariance->getId();
    }
}

void snn::OpenGLRenderPass::updateParameters() {
    if (_cp.pass.runtimeData.empty()) {
        return;
    }
    uint32_t* jitterOffset = _cp.pass.runtimeData.data();
    size_t len = _cp.pass.runtimeData.size()/_cp.pass.period;
    for (size_t i = 0; i < _runtimeUniforms.size(); i++) {
        uint32_t paramIdx = _cp.pass.runtimeUniforms[_runtimeUniforms[i].getName()].first;
        uint32_t value = jitterOffset[UP_DIV(_runIdx, _cp.pass.totalPasses) % len * _cp.pass.period + paramIdx];
        _runtimeUniforms[i].update((int)value);
        _runtimeUniforms[i].apply();
        SNN_LOGD("Update runtime parameter for %s, %d-%zu, %d at %d",
             _runtimeUniforms[i].getName().c_str(), _runIdx, i, value,
             UP_DIV(_runIdx, _cp.pass.totalPasses) % len * _cp.pass.period + paramIdx);
    }
}

// -----------------------------------------------------------------------------
//
void snn::OpenGLRenderPass::run() {
    _program.use();
    bindProgramInputs();
    snn::ImageTextureGLArrayAccessor texOutputsGL = _cp.texOutputs;
    auto ssboMap = _ssboMap;

    std::visit(match {
                    [&](const InferencePassGl::FsProgram& fs) {
                        // bind output texture to frame buffer
                        // note: dont' apply viewport here. viewport is already applied by call already.
                        _fb.bind();
                        _fb.attachTexture(*(texOutputsGL[0].texture(0)), fs.outputSliceIndex, fs.outputSliceCount);
                        gl::clearScreen(GL_COLOR_BUFFER_BIT);
                        _quad.draw();
                        _fb.detachTexture();
                        FrameBuffer2::unbind();
                    },
                    [&](const InferencePassGl::CsProgram& cs) {
                        for (std::pair<uint32_t, GLuint> element : ssboMap) {
                            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, element.first, element.second);
                        }
                        auto outputBinding = _program.getUniformBinding(cs.outputImageUniform.c_str());

                        const gl::TextureObject::TextureDesc& texOutputDesc = texOutputsGL[0].texture(0)->getDesc();
                        auto internalFormat = getNativeColorGL(texOutputDesc.format).glInternalFormat;
                        GLCHKDBG(glBindImageTexture(outputBinding, *(texOutputsGL[0].texture(0)), 0, true, 0, GL_WRITE_ONLY, internalFormat));
                        SNN_LOGD("Bind output: %s", texOutputsGL[0].getTextureInfo2().c_str());

                        GLCHKDBG(glDispatchCompute(cs.dispatchSize[0], cs.dispatchSize[1], cs.dispatchSize[2]));
                        SNN_LOGD("dispatch sizes: %d:%d:%d", cs.dispatchSize[0], cs.dispatchSize[1], cs.dispatchSize[2]);
                    },
               },
               _cp.pass.program);
}

// -----------------------------------------------------------------------------
//
void snn::OpenGLRenderPass::bindProgramInputs() {
    // bind input textures
    snn::ImageTextureGLArrayAccessor texInputsGL = _cp.texInputs;
    for (auto[name, index] : _cp.pass.inputs) {
        auto tex = texInputsGL[index].texture(0);

        auto binding = _program.getUniformBinding(name.c_str());
        if (binding >= 0) {
            if (isCompute()) {
                auto internalFormat = getNativeColorGL(tex->getDesc().format).glInternalFormat;
                glBindImageTexture(binding, tex->id(), 0, true, 0, GL_READ_ONLY, internalFormat);
                CHECK_GL_ERROR("glBindImageTexture");
            } else {
                SNN_LOGV("Bind input: %s: %d, with %d %d", name.c_str(), index, tex->id(), binding);
                tex->bind(binding);
                glBindSampler(binding, _cp.sampler.at(index));
            }
        } else {
            SNN_LOGE("Binding input not found: %s: index: %d, texture id: %d", name.c_str(), index, tex->id());
        }
    }

    auto passWeights = _weights;
    auto weightUniformTags = _weightUniformTags;

    std::visit(match {[&](const std::vector<const gl::TextureObject*>& weightTextures) {
                        for (std::size_t index = 0; index < weightTextures.size(); index++) {
                            auto tex     = weightTextures[index];
                            auto binding = _program.getUniformBinding(weightUniformTags[index].c_str());
                            if (binding >= 0) {
                                tex->bind(binding);
                                glBindSampler(binding, _cp.weightSamplers[index]);
                            } else {
                                SNN_LOGE("Binding input not found: %s: index: %d, texture id: %d", weightUniformTags[index].c_str(), index, tex->id());
                            }
                        }},
                        [&](const std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>& weightBuffers) {
                            for (std::size_t index = 0; index < weightBuffers.size(); index++) {
                                auto buf        = weightBuffers[index];
                                auto blockIndex = glGetUniformBlockIndex(_program, weightUniformTags[index].c_str());
                                auto binding    = _program.getUniformBinding(weightUniformTags[index].c_str());
                                glUniformBlockBinding(_program, blockIndex, binding);
                                buf->bindBase(binding);
                            }
                        },
                        [&](const std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>& weightBuffers) {
                            for (std::size_t index = 0; index < weightBuffers.size(); index++) {
                                auto buf        = weightBuffers[index];
                                auto blockIndex = glGetProgramResourceIndex(_program, GL_SHADER_STORAGE_BUFFER, weightUniformTags[index].c_str());
                                auto binding    = index + 2;
                                glShaderStorageBlockBinding(_program, blockIndex, binding);
                                buf->bindBase(binding);
                            }
                        }
            }, passWeights);

    // bind uniforms
    for (auto& u : _uniforms) {
        u.apply();
    }

    updateParameters();
    // bind runtime uniforms
    for (auto& u : _runtimeUniforms) {
        u.apply();
    }
    _runIdx++;
}

// ----------------------------------------------------------------------------------
// Dump debug outputs to the folder

bool snn::OpenGLRenderPass::debugPassOutput(const std::string& folderName) {
    auto image = _cp.texOutputs[0].getRawImage();
    SNN_LOGD("output: %s", _cp.texOutputs[0].getTextureInfo().c_str());

    if (glGetError() != 0) {
        SNN_LOGE("Error dumping output of pass %s", _cp.name.c_str());
        return false;
    }

    std::string layerName = normalizeName(_cp.name);

    std::string path = formatString("%s/%s", folderName.c_str(), layerName.c_str());
    if (!createDirIfNotExists(path)) {
        return false;
    }

    SNN_LOGD("Saving outputs for: %s", _cp.name.c_str());
    for (std::size_t i = 0; i < image.depth(); i++) {
        auto filename = formatString("%s/%s/%02d.png", folderName.c_str(), layerName.c_str(), i);
        image.saveToPNG(filename, i, true);
    }
    auto dumpFileName = formatString("%s/%s.dump", folderName.c_str(), _cp.name.c_str());
    SNN_LOGD("Saving dump to %s", dumpFileName.c_str());
    image.saveToBIN(dumpFileName);
#if DUMP_RESULTS_TXT
    auto dumpTxtFileName = formatString("%s/%s.txt", folderName.c_str(), _cp.name.c_str());
    if (FILE* fDumpTxt = createFile(dumpTxtFileName.c_str())) {
        _cp.texOutputs[0].prettyPrint(fDumpTxt);
        fclose(fDumpTxt);
    }
#endif
    return true;
}

bool snn::OpenGLRenderPass::debugPassWeights(const std::string& folderName, int shaderPass) {
    std::string layerName = normalizeName(_cp.name);
    std::string path = formatString("%s/%s", folderName.c_str(), layerName.c_str());
    if (!createDirIfNotExists(path)) {
        return false;
    }

    auto filepath = formatString("%s/%s/%02d.glsl", folderName.c_str(), layerName.c_str(), shaderPass);
    std::ofstream shaderSource;
    shaderSource.exceptions(std::ofstream::failbit); // may throw
    try {
        shaderSource.open(filepath);
    } catch (const std::ios_base::failure& fail) {
        SNN_LOGE("open %s: %s", filepath.c_str(), fail.what());
        return false;
    }
    shaderSource << _cp.pass.source << std::endl;
    shaderSource.close();

    auto passWeights = _weights;

    std::visit(
        match {[&](std::vector<const gl::TextureObject*>& weightTextures) {
                    for (std::size_t i = 0; i < weightTextures.size(); i++) {
                        auto image = weightTextures[i]->getBaseLevelPixels(true);
                        auto error = glGetError();
                        if (error != GL_NO_ERROR) {
                            SNN_LOGE("Error: GLError (%d) dumping weight of pass %s", error, _cp.name.c_str());
                            return;
                        }
                        auto binFilename = formatString("%s/%s_weights_%u.dump", folderName.c_str(), _cp.name.c_str(), weightTextures.size() * shaderPass + i);
                        image.saveToBIN(binFilename);
                    }
                },
                [&](std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>& weightBuffers) {
                    for (std::size_t i = 0; i < weightBuffers.size(); i++) {
                        std::vector<char> dump(weightBuffers[i]->length);
                        weightBuffers[i]->getData(dump.data(), 0, dump.size());
                        auto error = glGetError();
                        if (error != GL_NO_ERROR) {
                            SNN_LOGE("Error: GLError (%d) dumping weight of pass %s", error, _cp.name.c_str());
                            return;
                        }
                        auto binFilename = formatString("%s/%s_weights_%u.dump", folderName.c_str(), _cp.name.c_str(), weightBuffers.size() * shaderPass + i);
                        std::fstream dumpFile(binFilename.c_str(), std::ios::out | std::ios::binary);
                        if (!dumpFile.is_open()) {
                            SNN_LOGE("Error: (File did not open) dumping weight of pass %s", _cp.name.c_str());
                            return;
                        }
                        dumpFile.write(dump.data(), dump.size());
                        dumpFile.close();
                    }
                },
                [&](std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>& weightBuffers) {
                    for (std::size_t i = 0; i < weightBuffers.size(); i++) {
                        std::vector<char> dump(weightBuffers[i]->length);
                        weightBuffers[i]->getData(dump.data(), 0, dump.size());
                        auto error = glGetError();
                        if (error != GL_NO_ERROR) {
                            SNN_LOGE("Error: GLError (%d) dumping weight of pass %s", error, _cp.name.c_str());
                            return;
                        }
                        auto binFilename = formatString("%s/%s_weights_%u.dump", folderName.c_str(), _cp.name.c_str(), weightBuffers.size() * shaderPass + i);
                        std::fstream dumpFile(binFilename.c_str(), std::ios::out | std::ios::binary);
                        if (!dumpFile.is_open()) {
                            SNN_LOGE("Error: (File did not open) dumping weight of pass %s", _cp.name.c_str());
                            return;
                        }
                        dumpFile.write(dump.data(), dump.size());
                        dumpFile.close();
                    }
                }
        },
        passWeights);
    return true;
}

bool snn::OpenGLRenderPass::debugPassInputs(const std::string& folderName) {
    snn::ImageTextureGLArrayAccessor texInputsGL = _cp.texInputs;
    SNN_LOGD("input texture: %s", texInputsGL[0].getTextureInfo().c_str());
    auto image  = texInputsGL[0].texture(0)->getBaseLevelPixels();
    if (glGetError() != 0) {
        SNN_LOGE("Error dumping output of pass %s", _cp.name.c_str());
        return false;
    }

    std::string layerName = normalizeName(_cp.name);
    std::string path = formatString("%s/%s", folderName.c_str(), layerName.c_str());
    if (!createDirIfNotExists(path)) {
        return false;
    }

    SNN_LOGD("Saving input dump for layer: %s %zu", layerName.c_str(), image.depth());
    for (std::size_t i = 0; i < image.depth(); i++) {
        auto filename = formatString("%s/%s/%02d_input.png", folderName.c_str(), layerName.c_str(), i);
        image.saveToPNG(filename, i, true);
    }

    auto binFilename = formatString("%s/%s_input.dump", folderName.c_str(), _cp.name.c_str());
    image.saveToBIN(binFilename);
    return true;
}
