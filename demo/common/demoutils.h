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
#include "snn/snn.h"
#include "snn/glUtils.h"

namespace snn
{
/// frame visualizer
class NightVisionFrameViz
{
protected:
    class Impl;

    Impl * _impl;
    NightVisionFrameViz(Impl*);

public:
    NightVisionFrameViz();

    virtual ~NightVisionFrameViz();

    void resize(uint32_t w, uint32_t h);

    struct RenderParameters
    {
        FrameImage2 * frame;
    };

    // returns a fence that marks the end of the rendering process.
    virtual void render(const RenderParameters &);
};

class NightVisionFrameRec : public NightVisionFrameViz
{
    class Impl;

public:
    NightVisionFrameRec();
    void setWindow(intptr_t);

    void render(const RenderParameters &) override;
};

class MainProcessingLoop
{
    class Impl;
    Impl * _impl;
public:
    struct CreationParamaters
    {
        // note: this is how big the preview image is on screen. it could be different than the actual video frame resolution.
        uint32_t windowWidth = 1080, windowHeight = 1920;
        FrameProvider2 * frameProvider;
        bool serialized = false;
        bool compute = false; // perfer compute shader over fragment shader.
    };
    MainProcessingLoop(const CreationParamaters &);
    ~MainProcessingLoop();

    struct RenderParameters
    {
        bool playing;
        InferenceEngine::AlgorithmConfig algorithm;
        snn::InferenceEngine::ModelType modelType;
        snn::InferenceEngine::SNNModelOutput modelOutput;
    };
    void render(RenderParameters &);

    // change render window size
    void resize(uint32_t, uint32_t);

    void startRecording(intptr_t);
    void stopRecording();
};
}
