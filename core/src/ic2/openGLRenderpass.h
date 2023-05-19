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
#pragma once

#include "renderpass.h"
#include "snn/snn.h"
#include "snn/utils.h"
#include <snn/imageTexture.h>
#include "inferencepassGL.h"
#include "framebuffer.h"
#include "glUtils.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>

namespace snn {

// This classes implements actions, performed during a render pass.
class OpenGLRenderPass : public RenderPass {
public:
    OpenGLRenderPass() = default;

    virtual ~OpenGLRenderPass() = default;

    SNN_NO_COPY(OpenGLRenderPass);
    SNN_NO_MOVE(OpenGLRenderPass);

    // Creation parameters structure
    struct CreationParameters {
        std::string name;                       // Name
        InferencePassGl pass;                   // Inference pass
        std::vector<GLuint> sampler;            // An array of OpenGL samplers. Used to sample inputs
        std::vector<GLuint> weightSamplers;     // An array of OpenGL samplers. Used to sample weights
        ImageTextureArrayAccessor texInputs;    // Input images
        ImageTextureArrayAccessor texOutputs;   // Output images
    };

    // Constructor
    // params:
    //  cp - creation parameters
    OpenGLRenderPass(const CreationParameters& cp);

    // Dump layer outputs
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    bool debugPassOutput(const std::string& folderName) override;

    // Dump layer inputs
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    bool debugPassInputs(const std::string& folderName) override;

    // Dump layer weights
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    bool debugPassWeights(const std::string& foldername, int shaderPass) override;

    // Run render pass
    void run() override;

private:
    CreationParameters _cp;
    gl::FullScreenQuad _quad;
    FrameBuffer2 _fb; // do we need per-pass FBO? maybe an global one per inference core is enough.
    gl::SimpleGlslProgram _program;
    std::vector<gl::SimpleUniform> _uniforms;
    uint32_t _runIdx = 0;
    std::vector<gl::SimpleUniform> _runtimeUniforms;
    void updateParameters();

    //For Fragment Shader
    snn::FixedSizeArray<gl::TextureObject> _weightTextures;
    snn::FixedSizeArray<gl::BufferObject<GL_UNIFORM_BUFFER>> _weightUniformBuffers;
    snn::FixedSizeArray<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _weightSSBOBuffers;
    std::variant<std::vector<const gl::TextureObject*>, std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>,
                std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>> _weights;

    // Initializes internal arrays for fragment shader
    // params:
    //  weightMethod - weight access method
    //  fp16 - flag indicating whether FP16 computation is used
    //  kernelW - kernel width
    //  kernelH - kernel height
    //  numInputPlanes - number of input planes
    //  numOutputPlanes - number of output planes
    //  channelsPerPass - number of channels per pass
    //  fsPlaneIndex - index of fragment shader plane
    void initGLFSData(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
        uint32_t numInputPlanes, uint32_t numOutputPlanes,
        uint32_t channelsPerPass, uint32_t fsPlaneIndex);

    // Copy weights stored as textures to GPU
    //  weightMethod - weight access method
    //  fp16 - flag indicating whether FP16 computation is used
    //  kernelW - kernel width
    //  kernelH - kernel height
    //  numInputPlanes - number of input planes
    //  numOutputPlanes - number of output planes
    //  channelsPerPass - number of channels per pass
    //  fsPlaneIndex - index of fragment shader plane
    void setTextureWeights(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
        uint32_t numInputPlanes, uint32_t numOutputPlanes,
        uint32_t channelsPerPass, uint32_t fsPlaneIndex) const;

    // Copy weights stored as uniform buffers to GPU
    //  weightMethod - weight access method
    //  fp16 - flag indicating whether FP16 computation is used
    //  kernelW - kernel width
    //  kernelH - kernel height
    //  numInputPlanes - number of input planes
    //  numOutputPlanes - number of output planes
    //  channelsPerPass - number of channels per pass
    //  fsPlaneIndex - index of fragment shader plane
    void setBufferWeights(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
        uint32_t numInputPlanes, uint32_t numOutputPlanes,
        uint32_t channelsPerPass, uint32_t fsPlaneIndex) const;

    // Initializes internal arrays for fragment shader for depthwise convolution
    //  weightMethod - weight access method
    //  fp16 - flag indicating whether FP16 computation is used
    //  kernelW - kernel width
    //  kernelH - kernel height
    //  numInputPlanes - number of input planes
    //  numOutputPlanes - number of output planes
    //  channelsPerPass - number of channels per pass
    //  fsPlaneIndex - index of fragment shader plane
    void initGLFSDataDW(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
        uint32_t numInputPlanes, uint32_t numOutputPlanes,
        uint32_t channelsPerPass, uint32_t fsPlaneIndex);

    // Copy weights stored as textures to GPU for depthwise convolution
    //  weightMethod - weight access method
    //  fp16 - flag indicating whether FP16 computation is used
    //  kernelW - kernel width
    //  kernelH - kernel height
    //  numInputPlanes - number of input planes
    //  numOutputPlanes - number of output planes
    //  channelsPerPass - number of channels per pass
    //  fsPlaneIndex - index of fragment shader plane
    void setTextureWeightsDW(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
        uint32_t numInputPlanes, uint32_t numOutputPlanes,
        uint32_t channelsPerPass, uint32_t fsPlaneIndex) const;

    // Copy weights stored as uniform buffers to GPU for depthwise convolution
    //  weightMethod - weight access method
    //  fp16 - flag indicating whether FP16 computation is used
    //  kernelW - kernel width
    //  kernelH - kernel height
    //  numInputPlanes - number of input planes
    //  numOutputPlanes - number of output planes
    //  channelsPerPass - number of channels per pass
    //  fsPlaneIndex - index of fragment shader plane
    void setBufferWeightsDW(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
        uint32_t numInputPlanes, uint32_t numOutputPlanes,
        uint32_t channelsPerPass, uint32_t fsPlaneIndex) const;

    // For Compute Shader
    gl::TextureObject kernelTexture;
    std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _boWeights;
    std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER, MIN_SSBO_BUFFER_LEN_ARM_MALI>> _boBias;
    std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnMean;
    std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnVariance;
    std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnBeta;
    std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnGamma;

    // Other uniforms. Key is shader variable name.
    std::vector<std::string> _weightUniformTags = std::vector<std::string>(
        {"weightMatrix1", "weightMatrix2", "weightMatrix3", "weightMatrix4", "weightMatrix5", "weightMatrix6", "weightMatrix7", "weightMatrix8",
         "weightMatrix9", "weightMatrix10", "weightMatrix11", "weightMatrix12", "weightMatrix13", "weightMatrix14", "weightMatrix15", "weightMatrix16"});
    std::unordered_map<uint32_t, GLuint> _ssboMap;

    // Initializes internal arrays for compute shader
    //  weightMethod - weight access method
    //  fp16 - flag indicating whether FP16 computation is used
    //  kernelW - kernel width
    //  kernelH - kernel height
    //  numInputPlanes - number of input planes
    //  numOutputPlanes - number of output planes
    //  channelsPerPass - number of channels per pass
    void initGLCSData(uint32_t weightMethod, uint32_t fp16, uint32_t kernelW, uint32_t kernelH,
        uint32_t numInputPlanes, uint32_t numOutputPlanes);

    // Checks if render pass uses a compute shader
    // returns:
    //  true if the pass uses compute shader, false if not
    bool isCompute() const { return std::holds_alternative<InferencePassGl::CsProgram>(_cp.pass.program); }

    // Binds program inputs
    void bindProgramInputs();
};

} // namespace snn
