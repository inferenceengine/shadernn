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
#include "ic2/dp.h"

static constexpr const char* BASICCNN_MODEL_NAME = "basic_cnn_model.json";

namespace snn {
class BasicCNNProcessor : public Processor {
public:
    ~BasicCNNProcessor() override = default;
    void submit(Workload& w) override;

    static constexpr const char* defaultBasicCNNModel() { return BASICCNN_MODEL_NAME; }

    static std::unique_ptr<BasicCNNProcessor> createBasicCNNProcessor(ColorFormat format, bool compute = false, bool dumpOutputs = false) {
        return std::unique_ptr<BasicCNNProcessor>(new BasicCNNProcessor(format, defaultBasicCNNModel(), compute, dumpOutputs));
    }

    std::string getModelName() override { return "Basic CNN Model"; }

private:
    std::shared_ptr<MixedInferenceCore> ic2;
    std::unordered_map<GLuint, std::shared_ptr<Texture>> frameTextures;
    std::string modelName;

    bool compute;
    bool dumpOutputs;

    gl::TextureObject* getFrameTexture(GLuint id) {
        auto& t = frameTextures[id];
        if (!t) {
            t = Texture::createAttached(GL_TEXTURE_2D, id); // Create a thin shell around textureId
        }
        return t.get();
    }

    snn::dp::InferenceModel dp;

    BasicCNNProcessor(ColorFormat format, const std::string& modelFilename, bool compute, bool dumpOutputs)
        : Processor({{Device::GPU, format, 1}, {Device::GPU, format, 1}}), modelName(modelFilename), compute(compute), dumpOutputs(dumpOutputs) {
        this->setInputDims({112, 112, 1, 3});
        this->setOutputDims({7, 7, 32, 128});
        /*

            We need a way to get the output dims. Currently, we hard code them
            There are two ways to do this:
            1. Set up a getter to get the description of each layer. Use the Conv2D
            output dims formula to get the output of each layer, till we reach the final
            Conv (or MaxPool/AvgPool) layer.
            2. Get the input and output dims to each layer from the keras modelConfig in the
            convertTool. Use modelParser to parse these values and set the appropriate variables
            for each layer.

        */
    }
};
} // namespace snn
