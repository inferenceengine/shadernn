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
#include "snn/core.h"
#include "snn/image.h"
#include "backend.h"
#include "backendBuilder.h"
#include "dp.h"
#include <string>
#include <vector>
#include <array>
#include <map>
#include <memory>
#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <utility>

using namespace snn;

bool dumpTextOutputs(const std::string& dirname, const std::string& filename, const std::vector<std::vector<float>>& outputMat) {
    std::ostringstream dumpFilename;
    dumpFilename << dirname << "/" << filename << ".txt";
    std::ofstream dumpFile;
    dumpFile.open(dumpFilename.str());
    for (size_t i = 0; i < outputMat.size(); i++) {
        for (size_t j = 0; j < outputMat[i].size(); ++j) {
            auto val = outputMat[i][j];
            if (j > 0) {
                dumpFile << ", ";
            }
            dumpFile << val;
        }
        dumpFile << "\n";
    }
    dumpFile.close();
    return true;
}

void dumpBinOutputs(const std::string& binFilename, const std::vector<std::vector<float>>& outputMat) {
    if (outputMat.size() == 0 || outputMat[0].size() == 0) {
        return;
    }

    size_t matW = outputMat[0].size();
    size_t matH = outputMat.size();
    size_t matSize = matW * matH;

    std::vector<float> pixels(matSize);
    for (size_t i = 0, k = 0; i < matH; ++i, k += matW) {
        SNN_ASSERT(outputMat[i].size() == matW);
        memcpy(&pixels[k], outputMat[i].data(), matW * sizeof(float));
    }
    ImageDesc imageDesc(ColorFormat::R32F, matW, matH);
    RawImage image(std::move(imageDesc), pixels.data());
    image.saveToBIN(binFilename, false);
}

// -----------------------------------------------------------------------------
//
MixedInferenceCore::MixedInferenceCore(GpuContext* context_)
    : context(context_)
    , stages(RenderStagesArrayAllocator(context_))
{}

std::unique_ptr<MixedInferenceCore> snn::MixedInferenceCore::create(GpuContext* context, const CreationParameters& cp) {
    std::unique_ptr<MixedInferenceCore> p(new MixedInferenceCore(context));
    p->init(cp);
    return p;
}

std::unique_ptr<MixedInferenceCore> snn::MixedInferenceCore::create(GpuContext* context, const std::string& modelFileName,
    const dp::ShaderGenOptions& options, bool dumpOutputs) {
    bool useVulan = context->backendType == GpuBackendType::VULKAN;
    auto dp = snn::dp::loadFromJsonModel(modelFileName, useVulan, options.mrtMode, options.weightMode, options.preferrHalfPrecision);
    MixedInferenceCore::CreationParameters cp;
    (InferenceGraph &&) cp = snn::dp::generateInferenceGraph(dp[0], options);

    cp.dumpOutputs = dumpOutputs;
    return MixedInferenceCore::create(context, cp);
}

void snn::MixedInferenceCore::run(MixedInferenceCore::RunParameters& rp) {
    SNN_LOGV("");
    SNN_ASSERT(rp.inputImages.size() > 0);
    if (rp.inputImages.size() != cp.inputsDesc.size()) {
        SNN_LOGE("Wrong input texture count %d <-> %d", rp.inputImages.size(), cp.inputsDesc.size());
        return;
    }

    backend->prepareRun(rp, stages, bindOutput, stages.size() - 1);
    {
        ScopedTimer st1(cpuRunTime);
        std::vector<std::vector<float>> inputs;
#ifdef PROFILING
        if (backend->isProfilingEnabled()) {
            gpuRunTime->start();
        }
#endif
        for (std::size_t i = 0; i < stages.size(); i++) {
            auto& s = stages[i];

            // Nothing to run for the input layer.
            if (s.layer->isInputLayer) {
#ifdef PROFILING
                if (backend->isProfilingEnabled(true)) {
                    s.timer->start();
                    s.timer->stop();
                }
#endif
                continue;
            }
            for (size_t n = 0; n < s.delayBindMask.size(); ++n) {
                if (s.delayBindMask[n] > 0) {
                    auto inputIdx = s.inputIds[n];
                    s.stageInputs[n].attach(&rp.inputImages[inputIdx]);
                    SNN_LOGD("Delay binding input # %zu : %d  to input texture: %s", n, inputIdx, s.stageInputs[n].getTextureInfo2().c_str());
                }
            }

            SNN_LOGD("%zu / %zu, %s, backend:%d", i, stages.size(), s.layer->name.c_str(), s.backend);
            if (s.backend == Backend::Backend_GPU) {
                if (s.transition == Transition::Backend_CPU_GPU) {
                    // T.B.D. Copy CPU memory to Texture
                }
#ifdef PROFILING
                if (backend->isProfilingEnabled(true)) {
                    s.timer->start();
                }
#endif
                backend->prepareStage(rp, stages[i]);
                auto backendPtr = backend;
                s.layer->runFunPtr(backendPtr, this->cp.dumpOutputs);

#ifdef PROFILING
                if (backend->isProfilingEnabled(true)) {
                    s.timer->stop();
                }
#endif
            } else if (s.backend == Backend::Backend_CPU) {
                if (s.transition == Transition::Backend_GPU_CPU) {
#ifdef PROFILING
                    if (backend->isProfilingEnabled()) {
                        // TODO: Do we need this stop here?
                        // For CPU layer, GPU timing will not be affected anyway
                        // and the stop will be called later.
                        // What about a scenario if we have another GPU layer after CPU layer?
                        gpuRunTime->stop();
                    }
#endif
                    backend->sync();
                }
                if (s.transition == Transition::Backend_GPU_CPU) {
                    PROFILE_TIME(download, "download to CPU") // We exclude sync() time from CPU timing statistics
                    for (size_t j = 0; j < s.stageInputs.size(); j++) {
                        s.stageInputs[j].download();
                    }
                }

                // Copy CPU output from input to current layer
                for (size_t j = 0; j < s.inputIds.size(); j++) {
                    SNN_LOGD("######## stage %zd with input: %zu", i, s.inputIds[j]);
                    SNN_ASSERT(s.inputIds[j] >= 0);
                    s.stageInputs[j].setOutputMat(stages[s.inputIds[j]].stageOutputs[0].getOutputMat());
                }
                PROFILE_TIME(Backend_CPU, "Backend CPU") // We exclude sync() time from CPU timing statistics
                s.layer->imageTextureFunPtr(s.stageInputs, s.stageOutputs);
                if (this->cp.dumpOutputs) {
#if DUMP_RESULTS_TXT
                    auto fileName = formatString("%s cpu layer", s.layer->name.c_str());
                    dumpTextOutputs(std::string(OUTPUT_DIR), fileName, s.stageOutputs[0].getOutputMat());
#endif
                    std::string dumpFileName = formatString("%s/%s pass[0].dump", OUTPUT_DIR, s.layer->name.c_str());
                    SNN_LOGD("Saving dump to %s", dumpFileName.c_str());
                    dumpBinOutputs(dumpFileName, s.stageOutputs[0].getOutputMat());
                }
            }
            SNN_LOGD("########, layer:%zu", i);
        }

#ifdef PROFILING
        if (backend->isProfilingEnabled()) {
            gpuRunTime->stop();
        }
#endif

        this->output = std::move(inputs);
        inputs.clear();
        backend->sync();
        backend->postRun(stages, this->cp.dumpOutputs, OUTPUT_DIR);

#ifdef PROFILING
        if (backend->isProfilingEnabled(true)) {
            for (size_t i = 0; i < stages.size(); i++) {
                auto& s = stages[i];
                if (s.backend == Backend::Backend_GPU) {
                    s.timer->getTime();
                }
            }
        }
#endif

#ifdef PROFILING
        if (backend->isProfilingEnabled()) {
            gpuRunTime->getTime();
        }
#endif
    }

#ifdef PROFILING
    SNN_LOGD(printTimingStats().c_str());
#endif

    if (rp.modelOutput.modelType == ModelType::CLASSIFICATION && stages[stages.size() - 1].backend == Backend::Backend_CPU) {
        // 0 = None; Add 1 to start index in classifier
        rp.modelOutput.classifierOutput = std::distance(stages[stages.size() - 1].stageOutputs[0].getOutputMat().at(0).begin(),
                                                        std::max_element(stages[stages.size() - 1].stageOutputs[0].getOutputMat().at(0).begin(),
                                                                         stages[stages.size() - 1].stageOutputs[0].getOutputMat().at(0).end())) + 1;
        SNN_LOGD("Classifier output: %d", rp.modelOutput.classifierOutput);
    }
    else if (rp.modelOutput.modelType == ModelType::DETECTION && stages[stages.size() - 1].backend == Backend::Backend_CPU) {
        rp.modelOutput.detectionOutput = stages[stages.size() - 1].stageOutputs[0].getOutputMat();
    }

    backend->cleanupRun();

    // Print out time stats every 5 seconds
#ifdef PROFILING
    SNN_LOG_EVERY_N_SEC(5, INFO, printTimingStats().c_str());
#endif
}

std::pair<Backend, Transition> mapDeviceBackend(InferenceGraph::LayerExecutionType prevLayer, InferenceGraph::LayerExecutionType currLayer) {
    Backend retBackend  = Backend::NOT_DEFINED;
    Transition retTrans = Transition::NOT_DEFINED;
    if ((prevLayer == currLayer) || (prevLayer == InferenceGraph::LayerExecutionType::NOT_DEFINED) ||
        (prevLayer == InferenceGraph::LayerExecutionType::GPU_FS && currLayer == InferenceGraph::LayerExecutionType::GPU_CS) ||
        (prevLayer == InferenceGraph::LayerExecutionType::GPU_CS && currLayer == InferenceGraph::LayerExecutionType::GPU_FS)) {
        switch (currLayer) {
        case InferenceGraph::LayerExecutionType::CPU:
            retBackend = Backend::Backend_CPU;
            break;
        case InferenceGraph::LayerExecutionType::GPU_FS:
            retBackend = Backend::Backend_GPU;
            break;
        case InferenceGraph::LayerExecutionType::GPU_CS:
            retBackend = Backend::Backend_GPU;
            break;
        case InferenceGraph::LayerExecutionType::GPU_VK:
            retBackend = Backend::Backend_GPU;
            break;
        default:
            break;
        }
    } else {
        if (prevLayer == InferenceGraph::LayerExecutionType::CPU &&
            (currLayer == InferenceGraph::LayerExecutionType::GPU_FS || currLayer == InferenceGraph::LayerExecutionType::GPU_CS)) {
            retBackend = Backend::Backend_GPU;
            retTrans   = Transition::Backend_CPU_GPU;
        } else if ((prevLayer == InferenceGraph::LayerExecutionType::GPU_FS || prevLayer == InferenceGraph::LayerExecutionType::GPU_CS) &&
                currLayer == InferenceGraph::LayerExecutionType::CPU) {
            retBackend = Backend::Backend_CPU;
            retTrans   = Transition::Backend_GPU_CPU;
        } else if ((prevLayer == InferenceGraph::LayerExecutionType::CPU) &&
                currLayer == InferenceGraph::LayerExecutionType::GPU_VK) {
            retBackend = Backend::Backend_GPU;
            retTrans   = Transition::Backend_CPU_GPU;
        } else if ((prevLayer == InferenceGraph::LayerExecutionType::GPU_VK) &&
                currLayer == InferenceGraph::LayerExecutionType::CPU) {
            retBackend = Backend::Backend_CPU;
            retTrans   = Transition::Backend_GPU_CPU;
        }
    }
    return std::pair<Backend, Transition>(retBackend, retTrans);
}

// -----------------------------------------------------------------------------
//

bool snn::MixedInferenceCore::init(const MixedInferenceCore::CreationParameters& cp) {
    if (cp.dumpOutputs) {
        createDirIfNotExists(OUTPUT_DIR);
    }
    auto initTimeStart = std::chrono::high_resolution_clock::now();

    SNN_ASSERT(cp.inputsDesc.size() > 0);
    this->cp = cp;
    int channelsPerPass = static_cast<int>(this->cp.mrtMode);
    SNN_LOGD("Using channels per pass: %d", channelsPerPass);

    backend = dp::BackendBuilder::build(context, cp);
#ifdef PROFILING
    gpuRunTime = (backend->createDeviceTimer("IC2 Total GPU runtime"));
#endif
    std::string rootFilename(OUTPUT_DIR);

    // process stages/layers one by one
    stages.allocate(cp.layers.size());

    InferenceGraph::LayerExecutionType preDev = InferenceGraph::LayerExecutionType::NOT_DEFINED;
    for (std::size_t i = 0; i < cp.layers.size(); i++) {
        if (cp.layers[i]->flattenLayer) {
            bindOutput = false; // If model contains a flatten layer, we do not bind last layer to output image(s)
        }
        auto backTrans       = mapDeviceBackend(preDev, cp.layers[i]->layerLoc);
        stages[i].backend    = backTrans.first;
        stages[i].transition = backTrans.second;
        stages[i].delayBindMask.resize(cp.layers[i]->inputRefs.size(), 0);
        SNN_LOGD("%d, %d, %d - %d layer:%s", (int) preDev, (int) cp.layers[i]->layerLoc, (int) stages[i].backend,
                 (int) stages[i].transition, cp.layers[i]->name.c_str());
        preDev = cp.layers[i]->layerLoc;
    }

    for (size_t i = 0; i < stages.size(); ++i) { // TODO: use zip function to simpliy loop syntax
        InferenceGraph::Layer& layer = *cp.layers[i];
        RenderStage& stage = stages[i];

        stage.layer = std::make_shared<InferenceGraph::Layer>(layer);
        SNN_LOGD("stage: %zu, backend:%d, isInput:%d", i, (int)stage.backend, stage.layer->isInputLayer);
        // initialize stage's input array
        if (stage.backend == Backend::Backend_GPU) {
            if (stage.transition == Transition::Backend_CPU_GPU) {
                // TODO: Copy CPU memory to Texture
            }

            stage.stageInputs.allocate(layer.inputRefs.size());
            stage.stageOutputs.allocate(1);

            // Skip create new texture or binding for input layer
            if (stage.layer->isInputLayer) {
#ifdef PROFILING
                auto inputIdx = stage.layer->inputIndex;
                InferenceGraph::IODesc inputDesc  = cp.inputsDesc[inputIdx];
                std::string dimStr =
                    formatString("%dx%dx%d_%dx%dx%d", inputDesc.width, inputDesc.height, inputDesc.channels, inputDesc.width,
                        inputDesc.height, inputDesc.channels);
                stage.timer.reset(backend->createDeviceTimer(layer.name + "_" + dimStr));
#endif
                continue;
            }

            for (size_t j = 0; j < layer.inputRefs.size(); ++j) {
                const auto& inputRef = layer.inputRefs[j];
                if (inputRef.isStageOutput) {
                    SNN_LOGD("Backend_GPU: Stage: %zu, input: %zu, inputRef:%d", i, j, inputRef.index);
                    SNN_ASSERT(inputRef.index < i); // can't reference buffer from descendant layer
                    stage.stageInputs[j].attach(&stages[inputRef.index].stageOutputs[0]);
                    stage.inputIds.push_back(inputRef.index);
                    SNN_LOGD("Backend_GPU: Stage: %zu, input: %zu, inputRef:%d %s", i, j, inputRef.index, stage.stageInputs[j].getTextureInfo2().c_str());
                } else {  // Delay binding for input layer
                    auto inputIdx = stages[inputRef.index].layer->inputIndex;
                    stage.delayBindMask[j] = 1;
                    stage.inputIds.push_back(inputIdx);
                    SNN_LOGD("Backend_GPU: Stage: %zu, input: %zu, inputRef:%d delay binding: %d", i, j, inputRef.index, inputIdx);
                }
            }
            std::array<uint32_t, 4> dims {layer.outputDesc.width, layer.outputDesc.height, layer.outputDesc.depth, 1};
            stage.stageOutputs[0].resetTexture(dims, layer.outputDesc.format, "");
            SNN_LOGD("Layer %zu: texture: %s", i, stage.stageOutputs[0].getTextureInfo2().c_str());
            layer.initFunPtr(backend, stage.stageInputs, stage.stageOutputs);
        } else if (stage.backend == Backend::Backend_CPU) {
            SNN_LOGD("%%%%%%%% dim:%d, %d, %d", layer.outputDesc.width, layer.outputDesc.height, layer.outputDesc.depth);
            stage.stageInputs.allocate(layer.inputRefs.size());
            stage.stageOutputs.allocate(1);
            std::array<uint32_t, 4> dims {layer.outputDesc.width, layer.outputDesc.height, layer.outputDesc.depth, 1};
            stage.stageOutputs[0].resetTexture(dims, layer.outputDesc.format, "");

            for (size_t j = 0; j < layer.inputRefs.size(); ++j) {
                const auto& inputRef = layer.inputRefs[j];
                if (inputRef.index < 0) {
                    SNN_RIP("CPU layer currently cannot accept inputs from the input images !");
                }
                stage.inputIds.push_back(inputRef.index);
                stage.stageInputs[j].attach(&stages[inputRef.index].stageOutputs[0]);
                SNN_LOGD("Backend_CPU: Stage: %zu, input: %zu, inputRef:%d %s", i, j, inputRef.index, stage.stageInputs[j].getTextureInfo2().c_str());
            }
        }
#ifdef PROFILING
        // initialize timer
        // Get Input/Output Dims
        int prevLayerIdx = layer.inputRefs[0].index;
        InferenceGraph::IODesc inputDesc  = prevLayerIdx >= 0 ? stages[prevLayerIdx].layer->outputDesc : cp.inputsDesc[0];
        const InferenceGraph::IODesc& outputDesc = layer.outputDesc;
        SNN_LOGD("%%%%%%%%, input: %d, %d, %d, %d output: %d, %d, %d, %d", inputDesc.width, inputDesc.height, inputDesc.depth,
                 inputDesc.channels, outputDesc.width, outputDesc.height, outputDesc.depth, outputDesc.channels);
        std::string dimStr =
            formatString("%dx%dx%d_%dx%dx%d", inputDesc.width, inputDesc.height, inputDesc.channels, outputDesc.width, outputDesc.height, outputDesc.channels);

        stage.timer.reset(backend->createDeviceTimer(layer.name + "_" + dimStr));
#endif
    }
    auto initEndTime = std::chrono::high_resolution_clock::now();
    auto duration    = std::chrono::duration_cast<std::chrono::microseconds>(initEndTime - initTimeStart);
    SNN_LOGD("Time spent in initialization for MixedInferenceCore: %f secs", duration.count() / 1000000.0f);
    return true;
}

std::string snn::MixedInferenceCore::printTimingStats() const {
    size_t maxlen = gpuRunTime->getName().size();
    for (auto& s : stages) {
        maxlen = std::max(s.timer->getName().size(), maxlen);
    }
    std::stringstream ss;
    ss << "\n";
    ss << "=========================  GPU Inference Core Time Stats =========================\n";
    ss << "Time returned via GPU elapsed time query:\n";
    ss << "    " << std::setw(maxlen) << std::left << gpuRunTime->getName().c_str() << std::setw(0) << " : " << gpuRunTime->duration() / 1000000.0 << " ms"
       << std::endl;
    uint64_t total = 0;
    for (auto& s : stages) {
        ss << "    " << std::setw(maxlen) << std::left << s.timer->getName().c_str() << std::setw(0) << " : " << s.timer->duration() / 1000000.0
            << " ms" << std::endl;
        total += s.timer->duration();
    }
    ss << "    " << std::setw(maxlen) << std::left << "Total: " << std::setw(0) << " : " << total / 1000000.0 << " ms"
       << std::endl;
    ss << "----------------------------------------------------------------------------------\n";
    ss << cpuRunTime.print(0) << std::endl;
    ss << "==================================================================================\n";
    return ss.str();
}

void snn::MixedInferenceCore::writeTimeStat(std::map<std::string, std::vector<double>>& timeArray) {
    timeArray[gpuRunTime->getName()].push_back(gpuRunTime->duration() / 1000000.0);
    for (auto& s : stages) {
        timeArray[s.timer->getName()].push_back(s.timer->duration() / 1000000.0);
    }
}


snn::MixedInferenceCore::~MixedInferenceCore() {
    if (backend) {
        delete backend;
    }
    if (gpuRunTime) {
        delete gpuRunTime;
    }
    SNN_LOGV("MixedInferenceCore destroyed");
}
