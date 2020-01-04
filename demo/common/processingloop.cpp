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

#include "../pch.h"
#include "demoutils.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <stb_image.h>

using namespace snn;

class MainProcessingLoop::Impl
{
    std::unique_ptr<NightVisionFrameViz> _viz;
    std::shared_ptr<NightVisionFrameRec> _rec;
    std::atomic<bool> _isRecording = false;
    InferenceEngine::AlgorithmConfig _algorithm;
    std::unique_ptr<InferenceEngine> _inferenceEngine;
    CreationParamaters _cp;
    Timer _frameTime  = Timer("Total frame time.");
    Timer _renderTime = Timer("Main rendering loop");
    Timer _fetchDataTime = Timer("Main rendering loop - fetch frame data");
    Timer _processTime = Timer("Main rendering loop - engine process");
    Timer _autoFrameSetInitTime = Timer("Main rendering loop - retrieve frames");

    void switchEngineTo(InferenceEngine::AlgorithmConfig a) {
        if (_inferenceEngine && _algorithm == a) {
            return; // check for redundant switch
        }
        _algorithm = a;
        _inferenceEngine.reset(InferenceEngine::createInstance({a, _cp.windowWidth, _cp.windowHeight, _cp.serialized, _cp.compute }));

        //if(_algorithm.denoisers->denoiser == InferenceEngine::AlgorithmConfig::Denoisers::Denoiser::COMPUTESHADER){
        //    _inferenceEngine.reset(InferenceEngine::createInstance({a, _cp.windowWidth, _cp.windowHeight, _cp.serialized, true }));
        //}
        //else{
        //    _inferenceEngine.reset(InferenceEngine::createInstance({a, _cp.windowWidth, _cp.windowHeight, _cp.serialized, false }));
        //}
    }

public:
    Impl(const CreationParamaters & cp)
    {
        _cp = cp;
        switchEngineTo(InferenceEngine::AlgorithmConfig{});
        _viz.reset(new NightVisionFrameViz());
        _viz->resize(cp.windowWidth, cp.windowHeight);
        _rec = std::make_shared<NightVisionFrameRec>();
        _rec->resize(_cp.windowWidth, _cp.windowHeight);
    }

    ~Impl()
    {
    }

    void startRecording(intptr_t window) {
        _rec->setWindow(window);
        _isRecording = true;
    }

    void stopRecording() {
        _isRecording = false;
    }

    void render(RenderParameters & rp) {
        // this timer will count the whole frame time, including time that is not part of this render funcion.
        _frameTime.stop();
        _frameTime.start();

        {
        ScopedTimer strt(_renderTime);

        switchEngineTo(rp.algorithm);

        // InferenceEngine* engine = _inferenceEngine.get();

        // issue new work to engine
        _autoFrameSetInitTime.start();
        while (!_inferenceEngine) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(10)
            );
        }
        InferenceEngine::Item inputs = _inferenceEngine->beginEnqueue();
        _autoFrameSetInitTime.stop();
        if (!inputs.frames.empty()) {
            bool frameAvailable;
            {
                ScopedTimer st(_fetchDataTime);
                frameAvailable = _cp.frameProvider->fetchData(inputs.frames);
            }
            {
                ScopedTimer st(_processTime);
                if (frameAvailable) {
                    _inferenceEngine->endEnqueue();
                } else {
                    _inferenceEngine->abortEnqueue();
                }
            }
        }

        // check engine output
        auto output = _inferenceEngine->beginDequeue();
        if (!output.frames.empty()) {
            _viz->render({output.frames[0]});
            if (_isRecording) {
                _rec->render({output.frames[0]});
            }
            _inferenceEngine->endDequeue();
        }

        if (!output.tensors.empty()) {
            for (auto Blob : output.tensors) {
                for (auto batch : Blob) {
                    SNN_LOGI("[");
                    for (std::size_t outputIdx = 0; outputIdx < batch.size(); outputIdx++) {
                        SNN_LOGI("\t%f", batch.at(outputIdx));
                    }
                    SNN_LOGI("]");
                }
            }
        }

        if (rp.modelType == InferenceEngine::ModelType::CLASSIFICATION) {
            rp.modelOutput.classifierOutput = output.snnModelOutput.classifierOutput;
        } else if (rp.modelType == InferenceEngine::ModelType::DETECTION) {
        } else {
        }
        }

        SNN_LOG_EVERY_N_SEC(5, INFO, printKeyRenderLoopTimings().c_str());
    }


    void resize(uint32_t w, uint32_t h)
    {
        _viz->resize(w, h);
        _rec->resize(w, h);
    };

    std::string printKeyRenderLoopTimings() const
    {
        size_t namelen = _frameTime.name.size();
        for (auto & c : _frameTime.children) {
            namelen = std::max(namelen, c->name.size() + 2);
        }

        std::vector<Timer*> sorted(_frameTime.children.begin(), _frameTime.children.end());
        std::sort(sorted.begin(), sorted.end(), [](Timer* a, Timer* b){
            return a->begin < b->begin;
        });

        std::stringstream ss;
        ss << "\n";
        ss << "===========================  Key Render Loop Time Stats ==========================\n";
        ss << _frameTime.print(namelen + 2) << std::endl;
        for (auto & c : sorted) {
            ss << "  " << c->print(namelen, &_frameTime) << std::endl;
        }
        ss << "==================================================================================\n";
        return ss.str();
    }
};

snn::MainProcessingLoop::MainProcessingLoop(const CreationParamaters & cp) : _impl(new Impl(cp))
{
}

snn::MainProcessingLoop::~MainProcessingLoop()
{
    delete _impl;
}

void snn::MainProcessingLoop::render(RenderParameters & rp)
{
    _impl->render(rp);
}

void snn::MainProcessingLoop::resize(uint32_t w, uint32_t h)
{
    _impl->resize(w, h);
}

void snn::MainProcessingLoop::startRecording(intptr_t window) {
    _impl->startRecording(window);
}

void snn::MainProcessingLoop::stopRecording() {
    _impl->stopRecording();
}
