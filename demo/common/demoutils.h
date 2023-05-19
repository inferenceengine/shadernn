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
#include "snn/color.h"
#include "snn/image.h"
#include "inferenceengine.h"
#include <iostream>

namespace snn
{

// This is the new frame image class that is used in the engine.
class SNN_API FrameImage2 {
public:
    struct Desc {
        Device device;
        ColorFormat format;    // pixel format of the image.
        uint32_t width;        // width of the image in pixels
        uint32_t height;       // height of the image in pixels.
        uint32_t depth    = 1; // channels / 4 of the image
        uint32_t channels = 1;

        friend std::ostream& operator<<(std::ostream& os, const Desc& desc) {
            os << "Device: " << desc.device << std::endl;
            os << "Width: " << desc.width << std::endl;
            os << "Height: " << desc.height << std::endl;
            return os;
        }
    };

    virtual ~FrameImage2() = default;

    SNN_NO_COPY(FrameImage2);
    SNN_NO_MOVE(FrameImage2);

    const Desc& desc() const {
        return _desc;
    }

    virtual void getGpuImageHandle(GpuImageHandle& /*handle*/) const {}

    virtual void getCpuImage(RawImage& /*image*/) const {}

protected:
    FrameImage2() = default;

    Desc _desc;
};

struct FrameProvider2 {
    virtual ~FrameProvider2() = default;

    // returns false, if the data is not ready yet.
    virtual bool fetchData(const InferenceEngine::FrameVec&) = 0;

protected:
    FrameProvider2() = default;

private:
    SNN_NO_COPY(FrameProvider2);
    SNN_NO_MOVE(FrameProvider2);
};

/// frame visualizer
class NightVisionFrameVizImpl;

class NightVisionFrameViz
{
protected:
    NightVisionFrameVizImpl * _impl;
    NightVisionFrameViz(NightVisionFrameVizImpl*);

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

    virtual void reset();
};

class NightVisionFrameRec : public NightVisionFrameViz
{
public:
    NightVisionFrameRec();
    void setWindow(intptr_t);

    void render(const RenderParameters &) override;
};

// This class implements general rendering pipeline
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
        snn::ModelType modelType;
        snn::SNNModelOutput modelOutput;
    };
    void render(RenderParameters &);

    // change render window size
    void resize(uint32_t, uint32_t);

    void startRecording(intptr_t);
    void stopRecording();
};
}
