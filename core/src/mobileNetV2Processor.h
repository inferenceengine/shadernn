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

static constexpr const char* MOBILENETV2_MODEL_NAME = "mobilenetV2.json";

namespace snn {
class MobileNetV2Processor : public Processor {
public:
    ~MobileNetV2Processor() override = default;
    void submit(Workload& w) override;

    static constexpr const char* defaultMobileNetV2Model() { return MOBILENETV2_MODEL_NAME; }

    static std::unique_ptr<MobileNetV2Processor> createMobileNetV2Processor(ColorFormat format, bool compute = false, bool dumpOutputs = false) {
        return std::unique_ptr<MobileNetV2Processor>(new MobileNetV2Processor(format, defaultMobileNetV2Model(), compute, dumpOutputs));
    }

    std::string getModelName() override { return "MobileNetV2 Model"; }

private:
    std::shared_ptr<MixedInferenceCore> ic2_;
    std::unordered_map<GLuint, std::shared_ptr<Texture>> frameTextures_;
    std::string modelFileName_;
    bool compute_;
    bool dumpOutputs;

    const std::size_t expectedHeight = 224;
    const std::size_t expectedWidth  = 224;

    MobileNetV2Processor(ColorFormat format, const std::string& modelFilename, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}}), modelFileName_(modelFilename), compute_(compute),
          dumpOutputs(dumpOutputs) {}

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
