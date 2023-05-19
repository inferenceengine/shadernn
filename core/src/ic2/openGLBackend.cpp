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
#include "openGLBackend.h"
#include "dp.h"
#include "inferencepassGL.h"
#include "snn/core.h"
#include "openGLRenderpass.h"
#include "imageTextureGL.h"
#include <string>
#include <memory>
#include <variant>
#include <utility>

using namespace snn;
using namespace snn::dp;

OpenGLBackend::OpenGLBackend(const snn::dp::OpenGLBackend::CreationParameters& cp) {
    _cp = cp;

    samplers.resize(2);

    int channelsPerPass = static_cast<int>(this->_cp.mrtMode);
    SNN_LOGD("Using channels per pass: %d", channelsPerPass);

    sampler.allocate();
    sampler2.allocate();

    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (this->_cp.weightMode == snn::WeightAccessMethod::TEXTURES) {
        weightSamplers.resize(channelsPerPass);
        for (std::size_t i = 0; i < channelsPerPass; i++) {
            auto& wtSampler = weightSamplers[i];
            wtSampler.allocate();

            glSamplerParameteri(wtSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glSamplerParameteri(wtSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glSamplerParameteri(wtSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glSamplerParameteri(wtSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glSamplerParameteri(wtSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }

        for (std::size_t i = 0; i < channelsPerPass; i++) {
            weightSamplersUint.push_back(GLuint(weightSamplers[i]));
        }
    } else {
        weightSamplers.resize(1);
        auto& wtSampler0 = weightSamplers[0];
        wtSampler0.allocate();

        glSamplerParameteri(wtSampler0, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(wtSampler0, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(wtSampler0, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(wtSampler0, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(wtSampler0, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        weightSamplersUint.push_back(GLuint(weightSamplers[0]));
    }

    glSamplerParameteri(sampler2, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler2, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler2, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler2, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(sampler2, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    samplers.at(0) = std::move(sampler);
    samplers.at(1) = std::move(sampler2);

    // allocate debug buffer
    debugger.allocate(16 * 1024);
}

void OpenGLBackend::initRenderPasses(snn::dp::GenericModelLayer* modelLayer, snn::ImageTextureArrayAccessor texInputs,
    snn::ImageTextureArrayAccessor texOutputs) {
    modelLayer->getRenderPasses().clear();

    if (modelLayer->isInputLayer()) {
        return;
    }

    const InferencePassesGl* passesGl = InferencePassesGl::cast(modelLayer->getPasses());
    for (size_t i = 0; i < passesGl->passes.size(); i++) {
        auto& pass = passesGl->passes[i];

        std::visit(match {[&](const InferencePassGl::FsProgram&) {
                            sampler.bind(0);
                            sampler2.bind(1);
                            samplers.at(0) = std::move(sampler);
                            samplers.at(1) = std::move(sampler2);
                            weightSamplersUint.clear();
                            int index = 2;
                            for (auto& sampler : weightSamplers) {
                                sampler.bind(index);
                                weightSamplersUint.push_back((GLuint) sampler);
                                index++;
                            }
                        },
                        [&](const InferencePassGl::CsProgram&) {
                            sampler.bind(0);
                            sampler2.bind(1);
                            samplers.at(0)      = std::move(sampler);
                            samplers.at(1)      = std::move(sampler2);
                            auto& weightSampler = weightSamplers.at(0);
                            weightSampler.bind(2);
                            weightSamplersUint.clear();
                            weightSamplersUint.resize(1);
                            weightSamplersUint.at(0) = (GLuint) weightSampler;
                        }},
                    pass.program);

        OpenGLRenderPass::CreationParameters rpcp = {
            formatString("%s pass[%d]", modelLayer->getName().c_str(), i),
            pass,
            samplers,
            weightSamplersUint,
            texInputs,
            texOutputs,
        };

        auto renderPass = std::make_shared<snn::OpenGLRenderPass>(rpcp);

        modelLayer->getRenderPasses().push_back(renderPass);
    }
}

void OpenGLBackend::prepareRun(snn::MixedInferenceCore::RunParameters& rp,
        RenderStagesArray &stages, bool bindOutput, uint32_t bindIndex) {
    (void) rp;
    (void) stages;
    (void) bindOutput;
    (void) bindIndex;
#ifdef __ANDROID__
    if (rp.outputImages()) {
        if (bindOutput) {
#ifdef _DEBUG
            ImageTextureGL& outputImageGL = ImageTextureGL::cast(rp.outputImages[0]);
            const InferenceGraph::IODesc& outputDesc = stages[bindIndex].layer->outputDesc;
            for (size_t i = 0; i < outputImageGL.getNumTextures(); i++) {
                gl::TextureObject* tex = outputImageGL.texture(i);
#if 0
                // Texture format in fact might not match the description.
                // For example we describe output format in a shader as RGBAF16, but write into RGBAF32.
                // TODO: figure out why this works just fine.
                SNN_ASSERT(tex->getDesc().format == outputDesc.format);
#endif
                SNN_LOGI("%d %d", tex->getDesc().width, outputDesc.width);
                SNN_ASSERT(tex->getDesc().width == outputDesc.width);
                SNN_ASSERT(tex->getDesc().height == outputDesc.height);
                SNN_ASSERT(tex->getDesc().depth == outputDesc.depth);
                // We do not compare channels as they can be rounded up to 4. Comparing format is enough
            }
#endif
            stages[bindIndex].stageOutputs[0].attach(&rp.outputImages[0]);
        }
    }
#endif

    // setup common GL states
    glDisable(GL_DEPTH_TEST);
    // bind debug buffer
    debugger.clearCounter();
    debugger.bind();
    return;
}

void OpenGLBackend::prepareStage(snn::MixedInferenceCore::RunParameters& rp, snn::RenderStage& stage) {
    (void) rp;
    snn::ImageTextureGLArrayAccessor stageOutputsGL = stage.stageOutputs;
    glViewport(0, 0, (GLsizei) stageOutputsGL[0].texture(0)->getDesc().width, (GLsizei) stageOutputsGL[0].texture(0)->getDesc().height);
}

bool OpenGLBackend::sync() {
    glFinish();
    return true;
}

void OpenGLBackend::cleanupRun() {
    debugger.pullDataFromGPU();
    debugger.printLastResult();

    ++runCounter;

    return;
}

bool OpenGLBackend::isProfilingEnabled(bool queryPerLayerTime) {
    // Query overall runtime and each layers' time alternatively. This is to workaround
    // OpenGL's limitation that time query can't be overlapped.
    if (queryPerLayerTime) {
        return (runCounter % 2 == 0);
    } else {
        return (runCounter % 2 == 1);
    }
}

DeviceTimer* OpenGLBackend::createDeviceTimer(const std::string& name) {
    return new gl::GpuTimeElapsedQuery(name);
}
