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
#pragma once

#include "snn/defines.h"
#include "snn/snn.h"
#include "inferencegraph.h"

#include <memory>
#include <iostream>
#include <map>
#ifndef __has_include
static_assert(false, "__has_include not supported");
#else
    #if __cplusplus >= 201703L && __has_include(<filesystem>)
        #include <filesystem>
namespace fs = std::filesystem;
    #elif __has_include(<experimental/filesystem>)
        #include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
    #elif __has_include(<boost/filesystem.hpp>)
        #include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
    #endif
#endif

#define USE_NEW_IMAGE_TEXTURE 1

namespace snn {

class FrameBuffer2 {
public:
    static bool isComplete() { return GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER); }

    static void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

    SNN_NO_COPY(FrameBuffer2);
    SNN_NO_MOVE(FrameBuffer2);

    FrameBuffer2() { glGenFramebuffers(1, &_id); }

    virtual ~FrameBuffer2() { glDeleteFramebuffers(1, &_id); }

    GLuint id() const { return _id; }

    // attach to the new texture. assumes this framebuffer is bound.
    void attachTexture(const gl::TextureObject& texture, size_t firstLayer = 0, size_t layerCount = 1);

    // detach from any texture. assumes this framebuffer is bound.
    void detachTexture() {
        SNN_ASSERT(_current == this);
        GLCHKDBG(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
        GLenum c0 = GL_COLOR_ATTACHMENT0;
        GLCHKDBG(glDrawBuffers(1, &c0));
    }

    void bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, _id);
        _current = this;
    }

private:
    static const FrameBuffer2* _current; // pointing to current bound frame buffer object.
    GLuint _id = 0;
};

class RenderPass {
public:
    RenderPass() = default;

    virtual ~RenderPass() = default;

    SNN_NO_COPY(RenderPass);
    SNN_NO_MOVE(RenderPass);

    struct CreationParameters {
        std::string name;
        InferenceGraph::Pass pass;
        std::vector<GLuint> sampler;
        std::vector<GLuint> weightSamplers;
        snn::ImageTexture* texInputs  = NULL;
        snn::ImageTexture* texOutputs = NULL;
    };

    bool init(const CreationParameters&);

    bool debugPassOutput(std::string folderName);
    bool debugPassInputs(std::string folderName);
    bool debugPassWeights(std::string foldername, int shaderPass);

    void run();
    snn::ManagedRawImage getTextureDataOut();
    void run(std::vector<std::shared_ptr<snn::ManagedRawImage>>& input);
    void run(std::vector<std::vector<float>>& input);
    bool dumpOutputs(std::string& filename);
    bool dumpInputs(std::string& filename);
    std::vector<std::vector<float>> getOutput();

private:
    CreationParameters _cp;
    gl::FullScreenQuad _quad;
    FrameBuffer2 _fb; // do we need per-pass FBO? maybe an global one per inference core is enough.
    gl::SimpleGlslProgram _program;
    std::vector<gl::TextureObject*> _inputs;
    std::vector<gl::SimpleUniform> _uniforms;
    gl::TypedBufferObject<glm::vec4, GL_SHADER_STORAGE_BUFFER> _weights[4];

    bool isCompute() const { return std::holds_alternative<InferenceGraph::Pass::CsProgram>(_cp.pass.program); }

    void bindProgramInputs();
};

class MixedInferenceCore {
public:
    typedef enum MixedLayerInputType { GL_TEXTURE_OBJECT = 1, FLOAT_VEC = 2 } MixedLayerInputType;

    size_t transitionLayerIndex;

    virtual ~MixedInferenceCore() = default;

    SNN_NO_MOVE(MixedInferenceCore);
    SNN_NO_COPY(MixedInferenceCore);

    struct RunParameters {
        const gl::TextureObject* const* inputTextures;
        gl::TextureObject* textureOut;
        std::size_t inputCount;
        std::vector<std::vector<std::vector<float>>> inputMatrix, output;
        InferenceEngine::SNNModelOutput modelOutput;
    };

    void run(RunParameters& rp);

    void getOutput(std::vector<std::vector<float>>& output) { output = this->output; }

    // void getTransitionLayerInput();

    struct CreationParameters : InferenceGraph {
        uint32_t outputWidth, outputHeight, outputDepth;
        bool dumpOutputs;
    };

    static std::unique_ptr<MixedInferenceCore> create(const CreationParameters& cp) {
        SNN_LOGI("%s:%d\n", __FUNCTION__, __LINE__);
        std::unique_ptr<MixedInferenceCore> p(new MixedInferenceCore());
        if (!p->init(cp)) {
            return nullptr;
        }
        return p;
    }

    void getInputDims(uint32_t& width, uint32_t& height, uint32_t& depth) {
        width  = this->inputWidth;
        height = this->inputHeight;
        depth  = this->inputDepth;
    }

    void writeTimeStat(std::map<string, vector<double>>& timeArray);

private:
    struct RenderStage {
        std::shared_ptr<InferenceGraph::Layer> layer;
        FixedSizeArray<RenderPass> renderPasses;
        bool flattenLayer = false;
        gl::GpuTimeElapsedQuery timer;
        Backend backend       = Backend::Backend_GPU;
        Transition transition = Transition::NOT_DEFINED;
        FixedSizeArray<snn::ImageTexture> stageInputs;
        FixedSizeArray<snn::ImageTexture> stageOutputs;
        std::vector<size_t> inputIds;
    };

    uint32_t inputWidth, inputHeight, inputDepth;

    CreationParameters cp;
    FixedSizeArray<RenderStage> stages;
    gl::DebugSSBO debugger;
    std::vector<gl::SamplerObject> weightSamplers;
    gl::SamplerObject sampler, sampler2;
    gl::GpuTimestamps timestamps;
    gl::GpuTimeElapsedQuery gpuRunTime = gl::GpuTimeElapsedQuery("IC2 Total GPU runtime");
    Timer cpuRunTime                   = Timer("IC2 Total CPU Runtime");
    std::vector<std::vector<float>> output;

    FixedSizeArray<snn::ImageTexture> modelInputs;
    FixedSizeArray<snn::ImageTexture> modelOutputs;

    bool queryPerLayerTime = false;

    std::string printTimingStats() const;
    MixedInferenceCore() = default;

    bool init(const CreationParameters& cp);

    void printId(size_t i);
};

} // namespace snn
