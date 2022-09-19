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
#include "core.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <chrono>

#include "snn/image.h"

using namespace snn;

const snn::FrameBuffer2* snn::FrameBuffer2::_current = nullptr;

void snn::FrameBuffer2::attachTexture(const gl::TextureObject& texture, size_t firstLayer, size_t layerCount) {
    SNN_ASSERT(_current == this);

    // attach to the new texture
    switch (texture.getDesc().target) {
    case GL_TEXTURE_1D:
    case GL_TEXTURE_2D:
        SNN_ASSERT(0 == firstLayer && 1 == layerCount);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        break;

    case GL_TEXTURE_2D_ARRAY:
        for (size_t i = 0; i < layerCount; ++i) {
            SNN_LOGD("Attaching layer : %d", firstLayer + i);
            // SNN_LOGD("First Layer is: %d", firstLayer);
            glFramebufferTextureLayer(GL_FRAMEBUFFER, (GLenum)(GL_COLOR_ATTACHMENT0 + i), texture, 0, (GLint)(firstLayer + i));
        }
        break;

    default:
        // 3D or cube texture
        SNN_LOGE("not implemented.");
        unbind();
        return;
    }

    // setup draw buffers
    static constexpr const GLenum c_DrawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
                                                     GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
    glDrawBuffers((GLsizei) layerCount, c_DrawBuffers);

    SNN_ASSERT(isComplete());
}

// -----------------------------------------------------------------------------
//
bool snn::RenderPass::init(const snn::RenderPass::CreationParameters& cp) {
    _cp = cp;

    _quad.allocate();

    // create program
    _program.name = cp.name;
    if (isCompute()) {
        if (!_program.loadCs(cp.pass.source.c_str())) {
            return false;
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
            return false;
        }
    }

    // query all uniform locations.
    for (auto& [name, value] : cp.pass.uniforms) {
        _uniforms.emplace_back(name);
        if (!_uniforms.back().init(_program)) {
            SNN_LOGE("Uniform %s in %s not found.", name.c_str(), cp.name.c_str());
            SNN_LOGI("%s", cp.pass.source.c_str());
            return false;
        }
        _uniforms.back().value = value;
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

    // create weight buffer
    for (int i = 0; i < 4; ++i) {
        if (cp.pass.weightMatrices[i].empty()) {
            continue;
        }
        _weights[i].c = cp.pass.weightMatrices[i];
        _weights[i].allocateGpuBuffer();
    }

    // done
    return true;
}

// -----------------------------------------------------------------------------
//
void snn::RenderPass::run() {
    _program.use();
    // auto start = std::chrono::high_resolution_clock::now();
    bindProgramInputs();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
    // SNN_LOGI("Pass input binding took %f usecs", duration);
    std::visit(match {
                    [&](const InferenceGraph::Pass::FsProgram& fs) {
                        // bind output texture to frame buffer
                        // note: dont' apply viewport here. viewport is already applied by call already.
                        _fb.bind();
                        _fb.attachTexture(*(_cp.texOutputs[0].texture(0)), fs.outputSliceIndex, fs.outputSliceCount);
                        gl::clearScreen(GL_COLOR_BUFFER_BIT);
                        // auto beginDraw = std::chrono::high_resolution_clock::now();
                        _quad.draw();
                        // auto endDraw = std::chrono::high_resolution_clock::now();
                        // auto drawDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endDraw - beginDraw).count() / 1000.0;
                        _fb.detachTexture();
                        FrameBuffer2::unbind();
                    },
                    [&](const InferenceGraph::Pass::CsProgram& cs) {
                        // glFinish();
                        // SNN_LOGD("Test:%s:%d\n",__FUNCTION__,__LINE__);
                        for (std::pair<uint32_t, GLuint> element : _cp.pass.ssboMap) {
                            // SNN_LOGI("Test:%s:%d bind ssbo: %d: %d\n",__FUNCTION__,__LINE__, element.first, element.second);
                            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, element.first, element.second);
                        }
                        // SNN_LOGD("Test:%s:%d\n",__FUNCTION__,__LINE__);
                        auto outputBinding = _program.getUniformBinding(cs.outputImageUniform.c_str());

                        auto internalFormat = getColorFormatDesc(_cp.texOutputs[0].texture(0)->getDesc().format).glInternalFormat;
                        // SNN_LOGI("Test:%s:%d, output bind:%s %d\n",__FUNCTION__,__LINE__, cs.outputImageUniform.c_str(), outputBinding);
                        GLCHKDBG(glBindImageTexture(outputBinding, *(_cp.texOutputs[0].texture(0)), 0, true, 0, GL_WRITE_ONLY, internalFormat));
                        // SNN_LOGI("%s:%d Bind output: %d:%d\n",__FUNCTION__,__LINE__, _cp.texOutputs[0].texture(0)->id(),
                        // _cp.texOutputs[0].texture(0)->target());

                        GLCHKDBG(glDispatchCompute(cs.dispatchSize[0], cs.dispatchSize[1], cs.dispatchSize[2]));
                        //GLCHK(;);
                        // SNN_LOGI("Test:%s:%d dispatch: %d:%d:%d\n",__FUNCTION__,__LINE__, cs.dispatchSize[0], cs.dispatchSize[1], cs.dispatchSize[2]);
                    },
                    [&](const InferenceGraph::Pass::CPUProgram<float>& cpuL) {
                        (void) cpuL;
                        return;
                    },
               },
               _cp.pass.program);
}

// -----------------------------------------------------------------------------
//
void snn::RenderPass::bindProgramInputs() {
    // bind input textures
    for (auto[name, index] : _cp.pass.inputs) {
        auto tex = _cp.texInputs[index].texture(0);

        auto binding = _program.getUniformBinding(name.c_str());
        // SNN_LOGD("Test:%s:%d: %d\n",__FUNCTION__,__LINE__,binding);
        if (isCompute()) {
            // SNN_LOGI("Bind input:%s:%d, %s: %p, %d, %d on %d\n",__FUNCTION__,__LINE__, name.c_str(), tex, tex->id(), tex->target(), binding);
            auto internalFormat = getColorFormatDesc(tex->getDesc().format).glInternalFormat;
            glBindImageTexture(binding, tex->id(), 0, true, 0, GL_READ_ONLY, internalFormat);
            // glBindTexture(binding, tex->id());
            // int texId = 0;
            // glActiveTexture(GL_TEXTURE0 + texId);
            // glUniform1i(1, texId);
            // glBindTexture(GL_TEXTURE_2D_ARRAY, tex->id());
            // CHECK_GL_ERROR("glBindImageTexture");
        } else {
            // SNN_LOGD("Test:%s:%d\n",__FUNCTION__,__LINE__);
            SNN_LOGD("%s:%d, Bind input: %s: %d, with %d %d\n", __FUNCTION__, __LINE__, name.c_str(), index, tex->id(), binding);
            tex->bind(binding);
            glBindSampler(binding, _cp.sampler.at(index));
        }
        // SNN_LOGD("Test:%s:%d\n",__FUNCTION__,__LINE__);
    }

    std::visit(match {[&](const std::vector<const gl::TextureObject*>& weightTextures) {
                        for (std::size_t index = 0; index < weightTextures.size(); index++) {
                            auto tex     = weightTextures[index];
                            auto binding = _program.getUniformBinding(_cp.pass.weightUniformTags[index].c_str());
                            tex->bind(binding);
                            // std::cout << "Texture Weight at index " << index << ": " << tex->id() << ", " << tex->target() << " Bound at: " << binding <<
                            // std::endl;
                            glBindSampler(binding, _cp.weightSamplers[index]);
                        }},
                        [&](const std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>& weightBuffers) {
                            for (std::size_t index = 0; index < weightBuffers.size(); index++) {
                                auto buf        = weightBuffers[index];
                                auto blockIndex = glGetUniformBlockIndex(_program, _cp.pass.weightUniformTags[index].c_str());
                                auto binding    = _program.getUniformBinding(_cp.pass.weightUniformTags[index].c_str());
                                glUniformBlockBinding(_program, blockIndex, binding);
                                buf->bindBase(binding);
                                // std::cout << "Texture Weight at index " << index << ": " << tex->id() << ", " << tex->target() << " Bound at: " << binding <<
                                // std::endl;
                            }
                        },
                        [&](const std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>& weightBuffers) {
                            for (std::size_t index = 0; index < weightBuffers.size(); index++) {
                                auto buf        = weightBuffers[index];
                                auto blockIndex = glGetProgramResourceIndex(_program, GL_SHADER_STORAGE_BUFFER, _cp.pass.weightUniformTags[index].c_str());
                                auto binding    = index + 2;
                                glShaderStorageBlockBinding(_program, blockIndex, binding);
                                buf->bindBase(binding);
                                // std::cout << "Texture Weight at index " << index << ": " << tex->id() << ", " << tex->target() << " Bound at: " << binding <<
                                // std::endl;
                            }
                        }
            }, _cp.pass.weights);

    // bind uniforms
    for (auto& u : _uniforms) {
        u.apply();
    }
    /*
    // bind weights buffer (currently hard coded to binding index 0)
    for(GLuint i = 0; i < 4; ++i) {
        if (!_weights[i].g.empty()) _weights[i].g.bindBase(i);
    }
    */
}

// ----------------------------------------------------------------------------------
// Dump debug outputs to the folder

bool snn::RenderPass::debugPassOutput(std::string folderName) {
    // auto desc = _cp.texOutputs[0].texture(0)->getDesc();
    // readTexture(desc.width * desc.height * desc.depth, _cp.texOutputs[0].texture(0)->id(), desc.width, desc.height, desc.depth, 1);
    auto image = _cp.texOutputs[0].texture(0)->getBaseLevelPixels();
    SNN_LOGD("%s:%d output: %d:%d\n", __FUNCTION__, __LINE__, _cp.texOutputs[0].texture(0)->id(), _cp.texOutputs[0].texture(0)->target());

    if (glGetError() != 0) {
        SNN_LOGE("Error dumping output of pass %s", _cp.name.c_str());
        return false;
    }
    auto imageRGB8 = toRgba8(image);
    std::stringstream splitFilename(_cp.name);
    std::string word;
    std::vector<std::string> tokens;
    while (splitFilename >> word) {
        tokens.push_back(word);
    }

    tokens.pop_back();

    std::string layerName = "";
    for (auto word : tokens) {
        layerName += word;
        layerName += " ";
    }

    layerName.pop_back();
#ifdef __ANDROID__
    mkdir(formatString("%s/%s", folderName.c_str(), layerName.c_str()).c_str(), 0700);
#else
    fs::create_directories(formatString("%s/%s", folderName.c_str(), layerName.c_str()).c_str());
#endif
    SNN_LOGD("Saving dump for layer: %s", layerName.c_str());
    for (std::size_t i = 0; i < image.depth(); i++) {
        auto filename = formatString("%s/%s/%02d.png", folderName.c_str(), layerName.c_str(), i);
        imageRGB8.saveToPNG(filename, i);
    }
    auto binFilename = formatString("%s/%s.dump", folderName.c_str(), _cp.name.c_str());
    image.saveToBIN(binFilename);
    return true;
}

bool snn::RenderPass::debugPassWeights(std::string folderName, int shaderPass) {
    std::stringstream splitFilename(_cp.name);
    std::string word;
    std::vector<std::string> tokens;
    while (splitFilename >> word) {
        tokens.push_back(word);
    }

    tokens.pop_back();

    std::string layerName = "";
    for (auto word : tokens) {
        layerName += word;
        layerName += " ";
    }

    layerName.pop_back();
#ifdef __ANDROID__
    mkdir(formatString("%s/weights/", folderName.c_str(), layerName.c_str()).c_str(), 0700);
#else
    fs::create_directories(formatString("%s/weights/", folderName.c_str(), layerName.c_str()).c_str());
#endif
    std::visit(
        match {[&](std::vector<const gl::TextureObject*>& weightTextures) {
                    for (std::size_t i = 0; i < weightTextures.size(); i++) {
                        auto image = weightTextures[i]->getBaseLevelPixels();
                        auto error = glGetError();
                        if (error != GL_NO_ERROR) {
                            SNN_LOGE("Error: GLError (%d) dumping weight of pass %s", error, _cp.name.c_str());
                            return false;
                        }
                        auto binFilename = formatString("%s/weights/%s_%u.dump", folderName.c_str(), _cp.name.c_str(), weightTextures.size() * shaderPass + i);
                        image.saveToBIN(binFilename);
                    }
                },
                [&](std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>& weightBuffers) {
                    // SNN_LOGD("Dumping weight for pass: %s", _cp.name.c_str());
                    // SNN_LOGD("Size of weight buffers: %d", weightBuffers.size());
                    for (std::size_t i = 0; i < weightBuffers.size(); i++) {
                        std::vector<char> dump(weightBuffers[i]->length);
                        weightBuffers[i]->getData(dump.data(), 0, dump.size());
                        auto error = glGetError();
                        if (error != GL_NO_ERROR) {
                            SNN_LOGE("Error: GLError (%d) dumping weight of pass %s", error, _cp.name.c_str());
                            return false;
                        }
                        auto binFilename = formatString("%s/weights/%s_%u.dump", folderName.c_str(), _cp.name.c_str(), weightBuffers.size() * shaderPass + i);
                        std::fstream dumpFile(binFilename.c_str(), std::ios::out | std::ios::binary);
                        if (!dumpFile.is_open()) {
                            SNN_LOGE("Error: (File did not open) dumping weight of pass %s", _cp.name.c_str());
                            return false;
                        }
                        dumpFile.write(dump.data(), dump.size());
                        dumpFile.close();
                    }
                },
                [&](std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>& weightBuffers) {
                    // SNN_LOGD("Dumping weight for pass: %s", _cp.name.c_str());
                    // SNN_LOGD("Size of weight buffers: %d", weightBuffers.size());
                    for (std::size_t i = 0; i < weightBuffers.size(); i++) {
                        std::vector<char> dump(weightBuffers[i]->length);
                        weightBuffers[i]->getData(dump.data(), 0, dump.size());
                        auto error = glGetError();
                        if (error != GL_NO_ERROR) {
                            SNN_LOGE("Error: GLError (%d) dumping weight of pass %s", error, _cp.name.c_str());
                            return false;
                        }
                        auto binFilename = formatString("%s/weights/%s_%u.dump", folderName.c_str(), _cp.name.c_str(), weightBuffers.size() * shaderPass + i);
                        std::fstream dumpFile(binFilename.c_str(), std::ios::out | std::ios::binary);
                        if (!dumpFile.is_open()) {
                            SNN_LOGE("Error: (File did not open) dumping weight of pass %s", _cp.name.c_str());
                            return false;
                        }
                        dumpFile.write(dump.data(), dump.size());
                        dumpFile.close();
                    }
                }
        },
        _cp.pass.weights);
    return true;
}

bool snn::RenderPass::debugPassInputs(std::string folderName) {
    auto image  = _cp.texInputs[0].texture(0)->getBaseLevelPixels();
    auto format = _cp.texInputs[0].texture(0)->getDesc().format;
    // SNN_LOGD("%s:%d, input texture: %d, %d\n", __FUNCTION__,__LINE__, _cp.texInputs[0].texture(0)->id(), _cp.texInputs[0].texture(0)->target());

    if (glGetError() != 0) {
        SNN_LOGE("Error dumping output of pass %s", _cp.name.c_str());
        return false;
    }
    snn::RawImage imageRGB8;
    if (format == snn::ColorFormat::RGBA32F || format == snn::ColorFormat::RGBA8 || format == snn::ColorFormat::RGBA16F) {
        imageRGB8 = toRgba8(image);
    } else {
        imageRGB8 = toR8(image);
    }
    std::stringstream splitFilename(_cp.name);
    std::string word;
    std::vector<std::string> tokens;
    while (splitFilename >> word) {
        tokens.push_back(word);
    }

    tokens.pop_back();

    std::string layerName = "";
    for (auto word : tokens) {
        layerName += word;
        layerName += " ";
    }

    layerName.pop_back();
#ifdef __ANDROID__
    mkdir(formatString("%s/%s", folderName.c_str(), layerName.c_str()).c_str(), 0700);
#else
    fs::create_directories(formatString("%s/%s", folderName.c_str(), layerName.c_str()).c_str());
#endif

    SNN_LOGD("Saving input dump for layer: %s %zu", layerName.c_str(), image.depth());
    for (std::size_t i = 0; i < image.depth(); i++) {
        auto filename = formatString("%s/%s/%02d_input.png", folderName.c_str(), layerName.c_str(), i);
        imageRGB8.saveToPNG(filename, i);
    }
    auto binFilename = formatString("%s/%s_input.dump", folderName.c_str(), _cp.name.c_str());
    image.saveToBIN(binFilename);
    return true;
}

bool dumpTextOutputs(std::string& dirname, std::string& filename, std::vector<std::vector<float>>& outputMat) {
    std::ostringstream dumpFilename;
    dumpFilename << dirname << "/" << filename << ".txt";
    std::ofstream dumpFile;
    dumpFile.open(dumpFilename.str());
    auto& dumpOut = outputMat;
    // dumpFile << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < dumpOut.size(); i++) {
        for (auto val : dumpOut.at(i)) {
            dumpFile << val << ", ";
        }
        dumpFile << "\n";
    }
    dumpFile.close();
    return true;
}

// -----------------------------------------------------------------------------
//
void snn::MixedInferenceCore::run(snn::MixedInferenceCore::RunParameters& rp) {
#ifdef __ANDROID__
    mkdir(OUTPUT_DIR, 0700);
#endif
    SNN_LOGD("######## %s:%d %zu\n", __FUNCTION__, __LINE__, cp.inputs.size());
    if (rp.inputCount != cp.inputs.size()) {
        SNN_LOGE("Wrong input texture count %d <-> %d", rp.inputCount, cp.inputs.size());
        return;
    }

    for (size_t i = 0; i < rp.inputCount; ++i) {
        modelInputs[i].texture(0)->attach(*rp.inputTextures[i]);
        auto desc = rp.inputTextures[i]->getDesc();
        std::vector<uint32_t> dims {desc.width, desc.height, desc.depth, desc.channels};
        // (void) desc;
        // (void) dims;
        // modelInputs[i].resetImage(dims, desc.format);
        SNN_LOGD("%s:%d, %zu: texture: %d, %d\n", __FUNCTION__, __LINE__, i, rp.inputTextures[i]->id(), rp.inputTextures[i]->target());
        SNN_LOGD("%s:%d, %zu: texture: %d, %d\n", __FUNCTION__, __LINE__, i, modelInputs[i].texture(0)->id(), modelInputs[i].texture(0)->target());
        // Hack it.
        // stages[0].stageInputs[i].texture(0)->attach(*rp.inputTextures[i]);
        // stages[0].stageInputs[i].resetImage(dims, desc.format);
        SNN_LOGD("%s:%d, %zu: texture: %d, %d\n", __FUNCTION__, __LINE__, i, stages[0].stageInputs[i].texture(0)->id(),
                 stages[0].stageInputs[i].texture(0)->target());
    }

#ifdef __ANDROID__
    if (rp.textureOut) {
        if (this->transitionLayerIndex == stages.size()) {
            stages[this->transitionLayerIndex - 1].stageOutputs[0].texture(0)->attach(*(rp.textureOut));
        }
    }
#endif

    // setup common GL states
    glDisable(GL_DEPTH_TEST);

    // bind debug buffer
    debugger.clearCounter();
    debugger.bind();

    {
        ScopedTimer st1(cpuRunTime);
        std::vector<std::vector<float>> inputs;
#ifdef PROFILING
        if (!queryPerLayerTime) {
            gpuRunTime.start();
        }
#endif
        for (std::size_t i = 0; i < stages.size(); i++) {
            auto& s = stages[i];

            if (s.backend == Backend::Backend_GPU) {
                if (s.transition == Transition::Backend_CPU_GPU) {
                    // T.B.D. Copy CPU memory to Texture
                }
#ifdef PROFILING
                if (queryPerLayerTime) {
                    s.timer.start();
                }
#endif

                glViewport(0, 0, (GLsizei) s.stageOutputs[0].texture(0)->getDesc().width, (GLsizei) s.stageOutputs[0].texture(0)->getDesc().height);
                // Hack the input
                for (size_t n = 0; n < s.stageInputs.size(); n++) {
                    auto inputTex = s.stageInputs[n].texture(0);
                    if (inputTex->target() == 0 && inputTex->id() == 0) {
                        SNN_LOGD("######## %s:%d: Stage: %d, Render pass old %d: tex: %d, %d\n", __FUNCTION__, __LINE__, i, n, inputTex->id(),
                                 inputTex->target());
                        inputTex->attach(*rp.inputTextures[0]);
                        SNN_LOGD("######## %s:%d: Stage: %d, Render pass new %d: tex: %d, %d\n", __FUNCTION__, __LINE__, i, n, inputTex->id(),
                                 inputTex->target());
                    }
                }

                std::size_t passCount = 0;
                for (auto& p : s.renderPasses) {
                    // SNN_LOGI("%s:%d\n", __FUNCTION__,__LINE__);
                    if (this->cp.dumpOutputs) {
                        if (passCount == (s.renderPasses.size() - 1)) {
                            bool debugIn = p.debugPassInputs(OUTPUT_DIR);
                            if (!debugIn) {
                                SNN_LOGE("Error dumping inputs for layer %u", i);
                            }
                        }
#ifndef __ANDROID__
                        bool debugOut = p.debugPassWeights(OUTPUT_DIR, passCount);
                        if (!debugOut) {
                            SNN_LOGE("Error dumping weights for layer %u", i);
                        }
#endif
                    }
                    // SNN_LOGI("%s:%d\n", __FUNCTION__,__LINE__);
                    p.run();
                    // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
                    if (this->cp.dumpOutputs) {
                        if (passCount == (s.renderPasses.size() - 1)) {
                            bool debugOut = p.debugPassOutput(OUTPUT_DIR);
                            if (!debugOut) {
                                SNN_LOGE("Error dumping outputs for layer %u", i);
                            }
                        }
                    }
                    SNN_LOGD("%s:%d\n", __FUNCTION__, __LINE__);

                    passCount++;
                }
#ifdef PROFILING
                if (queryPerLayerTime) {
                    // glFinish();
                    s.timer.stop();
                    s.timer.getTime();
                }
#endif
            } else if (s.backend == Backend::Backend_CPU) {
                std::string baseFilename(OUTPUT_DIR);

                // Copy CPU output from input to current layer
                for (size_t j = 0; j < s.inputIds.size(); j++) {
                    SNN_LOGD("######## stage %zd with input: %zu\n", i, s.inputIds[j]);
                    s.stageInputs[j].outputMat = stages[s.inputIds[j]].stageOutputs[0].outputMat;
                }
                s.layer->imageTextureFunPtr(s.stageInputs, s.stageOutputs);
                if (this->cp.dumpOutputs) {
                    auto fileName = formatString("%s cpu layer", s.layer->name.c_str());
                    dumpTextOutputs(baseFilename, fileName, s.stageOutputs[0].outputMat);
                }
                SNN_LOGD("######## %s:%d, layer:%zu\n", __FUNCTION__, __LINE__, i);
            }
            //_timestamps.mark(s.layer->name);
            SNN_LOGD("######## %s:%d, layer:%zu\n", __FUNCTION__, __LINE__, i);
        }
// glFinish();
#ifdef PROFILING
        if (!queryPerLayerTime) {
            gpuRunTime.stop();
            gpuRunTime.getTime();
        }
#endif

        SNN_LOGD("%s:%d\n", __FUNCTION__, __LINE__);
        this->output = std::move(inputs);
        inputs.clear();

        // Query overall runtime and each layers' time alternatively. This is to workaround
        // OpenGL's limitation that time query can't be overlapped.
        queryPerLayerTime = !queryPerLayerTime;
    }

#ifdef PROFILING
    SNN_LOGD(printTimingStats().c_str());
#endif

    if (!this->output.empty()) {
        glFinish();
        std::stringstream alexNetOut(std::ios_base::out);
        alexNetOut << "----------------------- [Model Output] -----------------------" << std::endl;

        SNN_LOGI("######## %s:%d output size %zu\n", __FUNCTION__, __LINE__, this->output.size());

        if (this->output.size() > 0) {
            for (auto outputVal : this->output.at(0)) {
                alexNetOut << "[SNNLOG] [INFO] " << outputVal << std::endl;
            }
        }
        alexNetOut << "[SNNLOG] [INFO] "
                   << "-------------------------------------------------------------" << std::endl;
        SNN_LOGI("%s", alexNetOut.str().c_str());
    }

    if (rp.modelOutput.modelType == InferenceEngine::ModelType::CLASSIFICATION && stages[stages.size() - 1].backend == Backend::Backend_CPU) {
        // 0 = None; Add 1 to start index in classifier
        rp.modelOutput.classifierOutput = std::distance(stages[stages.size() - 1].stageOutputs[0].outputMat.at(0).begin(),
                                                        std::max_element(stages[stages.size() - 1].stageOutputs[0].outputMat.at(0).begin(),
                                                                         stages[stages.size() - 1].stageOutputs[0].outputMat.at(0).end())) +
                                        1;
    }

    if (rp.modelOutput.modelType == InferenceEngine::ModelType::DETECTION && stages[stages.size() - 1].backend == Backend::Backend_CPU) {
        rp.modelOutput.detectionOutput = stages[stages.size() - 1].stageOutputs[0].outputMat;
    }

    debugger.pullDataFromGPU();
    // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);
    debugger.printLastResult();
    // SNN_LOGD("%s:%d\n", __FUNCTION__,__LINE__);

    // Print out time stats every 5 seconds
    // SNN_LOG_EVERY_N_SEC(1, INFO, printTimingStats().c_str());
}

std::pair<snn::Backend, snn::Transition> mapDeviceBackend(snn::InferenceGraph::LayerExecution prevLayer, snn::InferenceGraph::LayerExecution currLayer) {
    snn::Backend retBackend  = snn::Backend::NOT_DEFINED;
    snn::Transition retTrans = snn::Transition::NOT_DEFINED;
    if ((prevLayer == currLayer) || (prevLayer == snn::InferenceGraph::LayerExecution::NOT_DEFINED) ||
        (prevLayer == snn::InferenceGraph::LayerExecution::GPU_FS && currLayer == snn::InferenceGraph::LayerExecution::GPU_CS) ||
        (prevLayer == snn::InferenceGraph::LayerExecution::GPU_CS && currLayer == snn::InferenceGraph::LayerExecution::GPU_FS)) {
        switch (currLayer) {
        case snn::InferenceGraph::LayerExecution::CPU:
            retBackend = snn::Backend::Backend_CPU;
            break;
        case snn::InferenceGraph::LayerExecution::GPU_FS:
            retBackend = snn::Backend::Backend_GPU;
            break;
        case snn::InferenceGraph::LayerExecution::GPU_CS:
            retBackend = snn::Backend::Backend_GPU;
            break;
        default:
            break;
        }
    } else {
        if (prevLayer == snn::InferenceGraph::LayerExecution::CPU &&
            (currLayer == snn::InferenceGraph::LayerExecution::GPU_FS || currLayer == snn::InferenceGraph::LayerExecution::GPU_CS)) {
            retBackend = snn::Backend::Backend_GPU;
            retTrans   = snn::Transition::Backend_CPU_GPU;
        } else if ((prevLayer == snn::InferenceGraph::LayerExecution::GPU_FS || prevLayer == snn::InferenceGraph::LayerExecution::GPU_CS) &&
                currLayer == snn::InferenceGraph::LayerExecution::CPU) {
            retBackend = snn::Backend::Backend_CPU;
            retTrans   = snn::Transition::Backend_GPU_CPU;
        }
    }
    return std::pair<snn::Backend, snn::Transition>(retBackend, retTrans);
}

void snn::MixedInferenceCore::printId(size_t end) {
    if (end <= 0) {
        return;
    }
    for (size_t i = 0; i <= end; ++i) {
        auto& stage = stages[i];
        for (size_t j = 0; j < stage.stageInputs.size(); j++) {
            // SNN_LOGD("%s:%d, %zu: %zu input: %d, %d \n", __FUNCTION__,__LINE__, i, j,
            //     stages[i].stageInputs[j].texture(0)->id(), stages[i].stageInputs[j].texture(0)->target());
        }
        SNN_LOGD("%zu: output: %d, %d \n", i, stages[i].stageOutputs[0].texture(0)->id(), stages[i].stageOutputs[0].texture(0)->target());
    }
}
// -----------------------------------------------------------------------------
//

bool snn::MixedInferenceCore::init(const snn::MixedInferenceCore::CreationParameters& cp) {
#ifdef __ANDROID__
    // mkdir(OUTPUT_DIR, 0700);
#endif
    auto initTimeStart = std::chrono::high_resolution_clock::now();

    SNN_LOGD("%s:%d\n", __FUNCTION__, __LINE__);
    this->cp            = cp;
    int channelsPerPass = static_cast<int>(this->cp.mrtMode);
    SNN_LOGI("Using channels per pass: %d", channelsPerPass);

    sampler.allocate();
    sampler2.allocate();

    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    std::vector<GLuint> weightSamplersUint;

    if (this->cp.weightMode == snn::WeightAccessMethod::TEXTURES) {
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

    SNN_LOGD("%s:%d\n", __FUNCTION__, __LINE__);

    glSamplerParameteri(sampler2, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler2, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler2, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(sampler2, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(sampler2, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    std::vector<GLuint> samplers(2);
    samplers.at(0) = std::move(sampler);
    samplers.at(1) = std::move(sampler2);

    // allocate input texture array
    modelInputs.allocate(cp.inputs.size());

    std::string rootFilename(OUTPUT_DIR);

    SNN_LOGD("%s:%d\n", __FUNCTION__, __LINE__);
    this->transitionLayerIndex = cp.layers.size();

    // process stages/layers one by one
    stages.allocate(cp.layers.size());

    snn::InferenceGraph::LayerExecution preDev = snn::InferenceGraph::LayerExecution::NOT_DEFINED;
    for (std::size_t i = 0; i < cp.layers.size(); i++) {
        if (cp.layers[i]->flattenLayer) {
            this->transitionLayerIndex = i;
        }
        // Should remove this hard coded transitionLayerIndex binded to Layer.
        auto backTrans       = mapDeviceBackend(preDev, cp.layers[i]->layerLoc);
        stages[i].backend    = backTrans.first;
        stages[i].transition = backTrans.second;
        SNN_LOGD("%s:%d %d, %d, %d - %d\n", __FUNCTION__, __LINE__, (int) preDev, (int) cp.layers[i]->layerLoc, (int) stages[i].backend,
                 (int) stages[i].transition);
        preDev = cp.layers[i]->layerLoc;
    }

    for (size_t i = 0; i < stages.size(); ++i) { // TODO: use zip function to simpliy loop syntax
        auto& layer = *cp.layers[i];
        auto& stage = stages[i];

        stage.layer = std::make_shared<InferenceGraph::Layer>(layer);

        // initialize stage's input array
        if (stage.backend == Backend::Backend_GPU) {
            if (stage.transition == Transition::Backend_CPU_GPU) {
                // T.B.D. Copy CPU memory to Texture
            }

            stage.stageInputs.allocate(layer.inputs.size());
            stage.stageOutputs.allocate(1);
            // stage.gpuInputs.allocate(layer.inputs.size());

            for (size_t j = 0; j < layer.inputs.size(); ++j) {
                auto& bufref = layer.inputs[j];

                if (bufref.isStageOutput) {
                    if (bufref.index >= i) {
                        SNN_LOGE("%s: can't reference buffer from descedent layer.", layer.name.c_str());
                        return false;
                    }
                    // *texImgPtr = *(stages[bufref.index].stageOutputs[0].texture(0));
                    // stage.stageInputs[j].resetTexture(*stages[bufref.index].stageOutputs[0].texture(0));
                    stage.stageInputs[j].texture(0)->attach(*stages[bufref.index].stageOutputs[0].texture(0));
                } else {
                    if (bufref.index >= modelInputs.size()) {
                        SNN_LOGE("%s: buffer reference index is out of range.", layer.name.c_str());
                        return false;
                    }
                    // *texImgPtr = *(modelInputs[bufref.index]._textures[0]);
                    // stage.stageInputs[j].resetTexture(*modelInputs[bufref.index].texture(0));
                    auto tex = modelInputs[bufref.index].texture(0);
                    SNN_LOGD("%s:%d, input tex:%d, %d \n", __FUNCTION__, __LINE__, tex->target(), tex->id());
                    if (tex->target() != 0 && tex->id() != 0) {
                        stage.stageInputs[j].texture(0)->attach(*tex);
                    }
                    SNN_LOGD("%s:%d \n", __FUNCTION__, __LINE__);
                }
                SNN_LOGD("%s:%d, %zu:%zu:%zu texture: %d, %d\n", __FUNCTION__, __LINE__, i, j, bufref.index, stage.stageInputs[j].texture(0)->id(),
                         stage.stageInputs[j].texture(0)->target());
                stage.inputIds.push_back(bufref.index);
            }
            std::vector<uint32_t> dims {layer.output.width, layer.output.height, layer.output.depth, 1};
            stage.stageOutputs[0].resetTexture(dims, layer.output.format);
            SNN_LOGD("%s:%d, %zu: texture: %d, %d, format: %d dims: %d: %d: %d\n", __FUNCTION__, __LINE__, i, stage.stageOutputs[0].texture(0)->id(),
                     stage.stageOutputs[0].texture(0)->target(), layer.output.format, layer.output.width, layer.output.height, layer.output.depth);

            // initialize stage's pass array
            stage.renderPasses.allocate(layer.passes.size());
            for (size_t j = 0; j < layer.passes.size(); ++j) {
                if (this->cp.dumpOutputs) {
#ifdef __ANDROID__
                    mkdir(formatString("%s/%s", rootFilename.c_str(), layer.name.c_str()).c_str(), 0700);
#else
                    fs::create_directories(formatString("%s/%s", rootFilename.c_str(), layer.name.c_str()).c_str());
#endif
                    auto filename = formatString("%s/%s/%02d.glsl", rootFilename.c_str(), layer.name.c_str(), j);
                    std::ofstream shaderSource(filename);
                    shaderSource << layer.passes[j].source << std::endl;
                    shaderSource.close();
                }

                std::visit(match {[&](const InferenceGraph::Pass::FsProgram& fs) {
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
                                [&](const InferenceGraph::Pass::CsProgram& cs) {
                                    sampler.bind(0);
                                    sampler2.bind(1);
                                    samplers.at(0)      = std::move(sampler);
                                    samplers.at(1)      = std::move(sampler2);
                                    auto& weightSampler = weightSamplers.at(0);
                                    weightSampler.bind(2);
                                    weightSamplersUint.clear();
                                    weightSamplersUint.resize(1);
                                    weightSamplersUint.at(0) = (GLuint) weightSampler;
                                },
                                [&](const InferenceGraph::Pass::CPUProgram<float>& cpuP) { (void) cpuP; }},
                           layer.passes[j].program);

                RenderPass::CreationParameters rpcp = {
                    formatString("%s pass[%d]", layer.name.c_str(), j),
                    layer.passes[j],
                    // stage.gpuInputs.data(),
                    // &stage.gpuOutput,
                    samplers,
                    weightSamplersUint,
                    stage.stageInputs.data(),
                    stage.stageOutputs.data(),
                };
                if (!stage.renderPasses[j].init(rpcp)) {
                    return false;
                }
            }

            SNN_LOGD("INIT Layer %d : %s\n", i, layer.name.c_str());
        } else if (stage.backend == Backend::Backend_CPU) {
            SNN_LOGD("%%%%%%%% %s:%d dim:%zu, %zu, %zu\n", __FUNCTION__, __LINE__, layer.output.width, layer.output.height, layer.output.depth);

            stage.stageInputs.allocate(layer.inputs.size());
            stage.stageOutputs.allocate(1);
            std::vector<uint32_t> dims {layer.output.width, layer.output.height, layer.output.depth, 1};
            stage.stageOutputs[0].resetTexture(dims, layer.output.format);
            // SNN_LOGD("%s:%d, %zu: texture: %d, %d\n", __FUNCTION__,__LINE__, i, stage.stageOutputs[0].texture(0)->id(),
            // stage.stageOutputs[0].texture(0)->target());
            for (size_t j = 0; j < layer.inputs.size(); ++j) {
                auto& bufref = layer.inputs[j];
                stage.inputIds.push_back(bufref.index);
                stage.stageInputs[j].texture(0)->attach(*stages[bufref.index].stageOutputs[0].texture(0));
            }
            SNN_LOGD("INIT CPU Layer %d : %s\n", i, layer.name.c_str());
        }
        // std::cout << "Layer Name: " << layer.name << std::endl;
        // initialize timer
        // Get Input/Output Dims
        auto inputBuf  = stages[layer.inputs[0].index].layer->output;
        auto outputBuf = layer.output;
        SNN_LOGD("%%%%%%%% %s:%d, input: %d, %d, %d, %d output: %d, %d, %d, %d\n", __FUNCTION__, __LINE__, inputBuf.width, inputBuf.height, inputBuf.depth,
                 inputBuf.channels, outputBuf.width, outputBuf.height, outputBuf.depth, outputBuf.channels);
        std::string dimStr =
            formatString("%dx%dx%d_%dx%dx%d", inputBuf.width, inputBuf.height, inputBuf.channels, outputBuf.width, outputBuf.height, outputBuf.channels);

        stage.timer.name = layer.name + "_" + dimStr;
    }

    // allocate debug buffer
    debugger.allocate(16 * 1024);

    // done
    //GLCHK(;); // make sure we are error free.
    auto initEndTime = std::chrono::high_resolution_clock::now();
    auto duration    = std::chrono::duration_cast<std::chrono::microseconds>(initEndTime - initTimeStart);
    SNN_LOGI("Time spent in initialization for MixedInferenceCore: %f secs", duration.count() / 1000000.0f);
    return true;
}

std::string snn::MixedInferenceCore::printTimingStats() const {
    size_t maxlen = gpuRunTime.name.size();
    for (auto& s : stages) {
        maxlen = std::max(s.timer.name.size(), maxlen);
    }

    std::stringstream ss;
    ss << "\n";
    ss << "=========================  GPU Inference Core Time Stats =========================\n";
    ss << "Time returned via GPU elapsed time query:\n";
    ss << "    " << std::setw(maxlen) << std::left << gpuRunTime.name.c_str() << std::setw(0) << " : " << gpuRunTime.duration() / 1000000.0 << " ms"
       << std::endl;
    for (auto& s : stages) {
        ss << "    " << std::setw(maxlen) << std::left << s.timer.name.c_str() << std::setw(0) << " : " << s.timer.duration() / 1000000.0 << " ms" << std::endl;
    }
    // ss << "----------------------------------------------------------------------------------\n";
    // ss << "Time returned via GPU timestamp query:\n";
    // ss << _timestamps.print("    ");
    ss << "----------------------------------------------------------------------------------\n";
    ss << cpuRunTime.print(0, nullptr) << std::endl;
    ss << "==================================================================================\n";
    return ss.str();
}

void snn::MixedInferenceCore::writeTimeStat(map<string, vector<double>>& timeArray) {
    timeArray[gpuRunTime.name].push_back(gpuRunTime.duration() / 1000000.0);
    for (auto& s : stages) {
        timeArray[s.timer.name].push_back(s.timer.duration() / 1000000.0);
    }
    // timeArray[cpuRunTime.name].push_back(d2s(cpuRunTime.ave.average));
}
