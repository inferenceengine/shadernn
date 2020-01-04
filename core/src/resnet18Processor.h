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

#include "processor.h"
#include <snn/texture.h>
#include <opencv2/core/mat.hpp>
#include <string>
#include "ic2/core.h"

static constexpr const char* RESNET18_MODEL_NAME = "resnet18_cifar10.json";

namespace snn {
class ResNet18Processor : public Processor {
public:
    ~ResNet18Processor() override = default;
    void submit(Workload& w) override;

    static constexpr const char* defaultResNet18Model() { return RESNET18_MODEL_NAME; }

    static std::unique_ptr<ResNet18Processor> createResNet18Processor(ColorFormat format, bool compute = false, bool dumpOutputs = false) {
        return std::unique_ptr<ResNet18Processor>(new ResNet18Processor(format, defaultResNet18Model(), compute, dumpOutputs));
    }

    std::string getModelName() override { return "ResNet18 Model"; }

private:
    std::shared_ptr<MixedInferenceCore> ic2_;
    std::unordered_map<GLuint, std::shared_ptr<Texture>> frameTextures_;
    std::string modelFileName_;
    bool compute_;
    bool dumpOutputs;

    const std::size_t expectedHeight = 32;
    const std::size_t expectedWidth  = 32;

    ResNet18Processor(ColorFormat format, const std::string& modelFilename, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::CPU, format, 1}}), modelFileName_(modelFilename), compute_(compute), dumpOutputs(dumpOutputs) {}

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
