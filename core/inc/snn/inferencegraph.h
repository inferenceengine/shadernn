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
#include <snn/imageTexture.h>
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace snn {

namespace dp {
class DeviceBackend;
}

// This class declares inference processing graph at high level.
struct InferenceGraph {
    enum class LayerExecutionType { CPU = 0, GPU_FS, GPU_CS, GPU_VK, NOT_DEFINED = 200 };

    // This structure describes the shape and color format of input/output images
    struct IODesc {
        ColorFormat format;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        uint32_t channels;
    };

    // This structure describes a reference to another layer in all layers array
    struct LayerRef {
        bool isStageOutput = false;
        int index = -1;
    };

    // This structure describes image shape transformation
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

    // This structure declares one layer of the inference processing graph at high level.
    struct Layer {
        LayerExecutionType layerLoc; // Location of layer (CPU or GPU)
        std::string name; // layer name (optional, for debugging and logging only)
        std::vector<LayerRef> inputRefs; // inputs of the layer.
        IODesc outputDesc; // output description of the layer.
        bool flattenLayer; // True if layer is fully-connected.
        bool isInputLayer = false;  // True if layer is input layer.
        uint32_t inputIndex = 0;    // The index of this layer in all layers array

        // Pointer to run on CPU function 
        using TImageTextureFunc = std::function<void(ImageTextureArray& inputMat, ImageTextureArray& outputMat)>;
        TImageTextureFunc imageTextureFunPtr;

        // Pointer to init function
        using TInitFunc = std::function<void(snn::dp::DeviceBackend *backend, ImageTextureArray& inputMat, ImageTextureArray& outputMat)>;
        TInitFunc initFunPtr;

        // Pointer to run function
        using TRunFunc = std::function<void(snn::dp::DeviceBackend *backend, bool dumpOutputs)>;
        TRunFunc runFunPtr;
    };

    // Input shapes description. Used for debugging only.
    std::vector<IODesc> inputsDesc;
    // All layers
    std::vector<std::shared_ptr<Layer>> layers;

    snn::MRTMode mrtMode               = snn::MRTMode::DOUBLE_PLANE;        // set up during generateInferenceGraph. Defaults to SINGLE_PLANE
    snn::WeightAccessMethod weightMode = snn::WeightAccessMethod::TEXTURES; // set up during generateInferenceGraph. Defaults to TEXTURE
};

} // namespace snn
