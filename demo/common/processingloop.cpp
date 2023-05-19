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

#include "../pch.h"
#include "inferenceengine.h"
#include "demoutils.h"
#include <iostream>
#include <strstream>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstddef>

using namespace snn;

// General rendering pipeline implementation
class MainProcessingLoop::Impl
{
    std::unique_ptr<NightVisionFrameViz> _viz;
    std::shared_ptr<NightVisionFrameRec> _rec;
    std::atomic<bool> _isRecording = false;
    InferenceEngine::AlgorithmConfig _algorithm;
    std::unique_ptr<InferenceEngine> _inferenceEngine;
    CreationParamaters _cp;
    Timer _frameTime  = Timer("Total frame time.");

    bool switchEngineTo(InferenceEngine::AlgorithmConfig a) {
        if (_inferenceEngine && _algorithm == a) {
            return false; // check for redundant switch
        }
        _algorithm = a;
        _inferenceEngine.reset(InferenceEngine::createInstance({a, _cp.windowWidth, _cp.windowHeight, _cp.serialized, _cp.compute }));

        return true;
    }

public:
    // Constructor
    // params:
    //  cp - creation parameters
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

    // This method is not currently used
    void startRecording(intptr_t window) {
        _rec->setWindow(window);
        _isRecording = true;
    }

    // This method is not currently used
    void stopRecording() {
        _isRecording = false;
    }

    // Main rendering loop implementation
    // params:
    //  rp - render parameters
    void render(RenderParameters & rp) {
        // this timer will count the whole frame time, including time that is not part of this render function.
        static bool firstStart = true;
        if (firstStart) {
            _frameTime.start();
            firstStart = false;
        }

        {
        PROFILE_TIME(Render, "Main rendering loop")

        if (switchEngineTo(rp.algorithm)) {
            _viz->reset();
        }

        // issue new work to engine
        InferenceEngine::Item inputs;
        {
            PROFILE_TIME(AutoFrameSetInit, "Main rendering loop - retrieve frames")
            while (!_inferenceEngine) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(10)
                );
            }
            inputs = _inferenceEngine->beginEnqueue();
        }
        if (!inputs.frames.empty()) {
            bool frameAvailable;
            {
                PROFILE_TIME(FetchData, "Main rendering loop - fetch frame data")
                frameAvailable = _cp.frameProvider->fetchData(inputs.frames);
            }
            {
                PROFILE_TIME(Process, "Main rendering loop - engine process")
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
                    SNN_LOGD("[");
                    for (std::size_t outputIdx = 0; outputIdx < batch.size(); outputIdx++) {
                        SNN_LOGD("\t%f", batch.at(outputIdx));
                    }
                    SNN_LOGD("]");
                }
            }
        }

        if (rp.modelType == ModelType::CLASSIFICATION) {
            rp.modelOutput.classifierOutput = output.snnModelOutput.classifierOutput;
        } else if (rp.modelType == ModelType::DETECTION) {
        } else {
        }
        }

        _frameTime.stop();
        SNN_LOG_EVERY_N_SEC(5, INFO, printKeyRenderLoopTimings().c_str());
        _frameTime.start();
    }

    // Resizes the output surface
    // params:
    //  w - resized width
    //  h - resized height
    void resize(uint32_t w, uint32_t h)
    {
        _viz->resize(w, h);
        _rec->resize(w, h);
    };

    // Prints main rendering timing statistics
    // returns:
    //  Timing statistics in human-readable format
    std::string printKeyRenderLoopTimings()
    {
        std::stringstream ss;
        ss << std::endl;
        ss << "===========================  Key Render Loop Time Stats ==========================\n";
        ss << Timer::print(10, true);
        ss << "==================================================================================\n";
        // We collect timing statistics between calls, not over total app life
        Timer::reset();
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

// Main rendering loop implementation
// params:
//  rp - render parameters
void snn::MainProcessingLoop::render(RenderParameters & rp)
{
    _impl->render(rp);
}

// Resizes the output surface
// params:
//  w - resized width
//  h - resized height
void snn::MainProcessingLoop::resize(uint32_t w, uint32_t h)
{
    _impl->resize(w, h);
}

// This method is not currently used
void snn::MainProcessingLoop::startRecording(intptr_t window) {
    _impl->startRecording(window);
}

// This method is not currently used
void snn::MainProcessingLoop::stopRecording() {
    _impl->stopRecording();
}
