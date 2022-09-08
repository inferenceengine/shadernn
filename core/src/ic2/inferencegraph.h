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

#include <snn/color.h>
#include <snn/glUtils.h>
#include <snn/imageTexture.h>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <numeric>
#include <unordered_map>
#include <variant>
#include <optional>
#include <functional>
#include <memory>
#include <stack>
#include <fstream>

#include <Eigen/Dense>
// ARM MALI DITING phone needs min buffer len to be 4.
#define MIN_SSBO_BUFFER_LEN_ARM_MALI 4

namespace snn {

// This class is where the inference shaders and processing graph are defined.
// It is the interface between shader authoring module and GPU inferencing module.
struct InferenceGraph {
    typedef enum class LayerExecution { CPU = 0, GPU_FS, GPU_CS, NOT_DEFINED = 200 } LayerExecution;

    struct Buffer {
        ColorFormat format;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        uint32_t channels;
        // std::vector<std::vector<float>> data; // Optional field for CPU based data
    };

    struct BufferRef {
        bool isStageOutput;
        size_t index;
    };

    struct Pass {
        struct FsProgram {
            uint32_t outputSliceIndex;
            uint32_t outputSliceCount;
        };
        struct CsProgram {
            // compute shader always bind the texture object as a whole. So there's no need to specify output slice index and count.
            std::string outputImageUniform;
            uint32_t dispatchSize[3];
        };

        template<typename T>
        struct CPUProgram {
            bool isTransitionLayer;
            bool isLastLayer;

            CPUProgram(bool transitionLayer, bool lastLayer) {
                this->isTransitionLayer = transitionLayer;
                this->isLastLayer       = lastLayer;
            }
        };

        std::variant<FsProgram, CsProgram, CPUProgram<float>> program;

        // shader source code
        std::string source;

        // key is shader variable name. value is index into the layer's input buffer array.
        std::unordered_map<std::string, uint32_t> inputs; // input buffer uniforms.
        std::optional<std::pair<std::vector<std::vector<float>>, std::vector<float>>> transformMat;
        std::unordered_map<uint32_t, GLuint> ssboMap;

        // Other uniforms. Key is shader variable name.
        std::unordered_map<std::string, gl::SimpleUniform::Value> uniforms;
        std::vector<std::string> weightUniformTags = std::vector<std::string>(
            {"weightMatrix1", "weightMatrix2", "weightMatrix3", "weightMatrix4", "weightMatrix5", "weightMatrix6", "weightMatrix7", "weightMatrix8",
             "weightMatrix9", "weightMatrix10", "weightMatrix11", "weightMatrix12", "weightMatrix13", "weightMatrix14", "weightMatrix15", "weightMatrix16"});

        std::variant<std::vector<const gl::TextureObject*>, std::vector<const gl::BufferObject<GL_UNIFORM_BUFFER>*>,
                     std::vector<const gl::BufferObject<GL_SHADER_STORAGE_BUFFER>*>>
            weights;

        uint32_t inputHeight, inputWidth, inputChannels;

        std::vector<glm::vec4> weightMatrices[4];

        std::shared_ptr<gl::GLSSBOBuffer> _ssboWeights;
        std::shared_ptr<gl::GLSSBOBuffer> _ssboBias;

        std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _boWeights;
        std::vector<float> _vecWeights;
        std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER, MIN_SSBO_BUFFER_LEN_ARM_MALI>> _boBias;
        std::vector<float> _vecBias;
        std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnMean;
        std::vector<float> _vecMean;
        std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnVariance;
        std::vector<float> _vecVariance;
        std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnBeta;
        std::vector<float> _vecBeta;
        std::shared_ptr<gl::BufferObject<GL_SHADER_STORAGE_BUFFER>> _bnGamma;
        std::vector<float> _vecGamma;
    };

    struct Transform {
        bool isFixed = 0;
        union {
            struct {
                float scaleWidth;
                float scaleHeight;
                float translateWidth;
                float translateHeight;
            };
            struct {
                uint32_t fixedWidth;
                uint32_t fixedHeight;
                uint32_t fixedDepth;
                uint32_t fixedBatch;
            };
        };

        static constexpr Transform identity() { return {0, {{1.0f, 1.0f, 0.0f, 0.0f}}}; }
    };

    struct Layer {
        LayerExecution layerLoc; // Location of layer (CPU or GPU)
        std::string name;        // layer name (optional, for debugging and logging only)
        std::string activationClass;
        std::vector<BufferRef> inputs; // input buffers of the layer.
        Buffer output;                 // output buffer of the layer.
        Transform accumulatedTransform;
        std::vector<Pass> passes; // render passes of the layer.
        bool flattenLayer;        // True if flatten (transition) layer.
        using TImageTextureFunc = std::function<void(FixedSizeArray<snn::ImageTexture>& inputMat, FixedSizeArray<snn::ImageTexture>& outputMat)>;
        TImageTextureFunc imageTextureFunPtr;
    };

    // data mebers that defines the graph
    std::vector<Buffer> inputs;
    std::vector<std::shared_ptr<Layer>> layers; // TODO: use value type instead of pointer.

    uint32_t inputWidth = 0, inputHeight = 0, inputChannels = 0;
    snn::MRTMode mrtMode               = snn::MRTMode::DOUBLE_PLANE;        // set during generateInferenceGraph. Defaults to SINGLE_PLANE
    snn::WeightAccessMethod weightMode = snn::WeightAccessMethod::TEXTURES; // set during generateInferenceGraph. Defaults to TEXTURE

    bool getInputDims(uint32_t& width, uint32_t& height, uint32_t& depth) {
        if (inputWidth == 0 && inputHeight == 0 && inputChannels == 0) {
            return false;
        } else {
            width  = inputWidth;
            height = inputHeight;
            depth  = inputChannels;
            return true;
        }
    }

    void dumpAllShaders(const std::string& folder) const {
        for (auto& l : layers) {
            for (size_t i = 0; i < l->passes.size(); ++i) {
                auto& c       = l->passes[i];
                auto filename = formatString("%s/%s pass[%02d].glsl", folder.c_str(), l->name.c_str(), i);
                SNN_LOGI("save inference shader: %s", filename.c_str());
                std::ofstream fs(filename);
                if (fs.good()) {
                    fs << c.source;
                }
            }
        }
    }
};
} // namespace snn
