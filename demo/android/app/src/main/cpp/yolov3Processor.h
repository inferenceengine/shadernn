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

#include "processor.h"
#include <snn/texture.h>
#include <opencv2/core/mat.hpp>
#include <string>
#include "ic2/core.h"

static constexpr const char* YOLOV3_MODEL_NAME = "yolov3-tiny_finetuned.json";

namespace snn {
class Yolov3Processor : public Processor {
public:
    ~Yolov3Processor() override = default;
    static constexpr const char* defaultYolov3Model() { return YOLOV3_MODEL_NAME; }
    void submit(Workload& w) override;
    static std::unique_ptr<Yolov3Processor> createYolov3Processor(ColorFormat format, bool compute = false, bool dumpOutputs = false) {
        return std::unique_ptr<Yolov3Processor>(new Yolov3Processor(format, defaultYolov3Model(), compute, dumpOutputs));
    }

    std::string getModelName() override { return "Yolov3 Model"; }

private:
    std::unique_ptr<MixedInferenceCore> ic2_;
    std::unordered_map<GLuint, std::shared_ptr<Texture>> frameTextures_;
    std::string modelFileName_;
    bool compute_;
    bool dumpOutputs;

    const std::size_t expectedHeight = 416;
    const std::size_t expectedWidth  = 416;

    Yolov3Processor(ColorFormat format, const std::string& modelFilename, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}}), modelFileName_(modelFilename), compute_(compute), dumpOutputs(dumpOutputs) {}

    gl::TextureObject* getFrameTexture(GLuint id) {
        auto& t = frameTextures_[id];
        // Create a thin shell around textureId
        if (!t) {
            t = Texture::createAttached(GL_TEXTURE_2D, id);
        }
        return t.get();
    }
};
} // namespace snn
