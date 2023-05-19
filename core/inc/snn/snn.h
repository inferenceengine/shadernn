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
// Main header of SNN
#pragma once
#include "defines.h"
#include <vector>

#ifdef _MSC_VER
    #pragma warning(disable : 4201) // nameless struct/union
#endif

#ifdef __ANDROID__
    //#define ASSETS_DIR "/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/"
#if 0
    // For some reason I cannot grant permissions to create directories in /data/local/tmp/files/
    #define OUTPUT_DIR "/data/local/tmp/files/inferenceCoreDump"
#else
    // Create /sdcard/Android/data/com.innopeaktech.seattle.snndemo in adb shell and grant all permissions to it
    #define OUTPUT_DIR "/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump/"
#endif
    //For Non root phones
    #define MODEL_DIR  "/data/local/tmp/jsonModel/"
    #define ASSETS_DIR "/data/local/tmp/files/"
#else
    #define MODEL_DIR  "../../../../modelzoo/"
    #define ASSETS_DIR "../../../../core/data/"
    #define OUTPUT_DIR "../../../../core/inferenceCoreDump"
#endif

#define DUMP_DIR OUTPUT_DIR

// TODO: replace them with inline functions
#define DIV_4_ROUND_UP(i)      (((i) + 3) / 4)
#define DIV_AND_ROUND_UP(x, y) ((x + (y - 1)) / y)
#define ROUND_UP_DIV_4(i)      (((i) + 3) / 4 * 4)
#define UP_DIV(x, y)           (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y)         (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x)           ROUND_UP((x), 4)

namespace snn {

enum Device {
    GPU,
    CPU,
};

enum class GpuBackendType {
    GL,
    VULKAN,
};

enum class Precision {
    FP32,
    FP16,
};

// Weights storage method:
//  CONSTANTS: Constants in a shader code
//  TEXTURES: textures
//  UNIFORM_BUFFER: uniform buffers
//  SSBO_BUFFER: shader storage buffer object
enum class WeightAccessMethod { CONSTANTS, TEXTURES, UNIFORM_BUFFER, SSBO_BUFFER };

enum class MRTMode { NO = 0, SINGLE_PLANE = 4, DOUBLE_PLANE = 8, QUAD_PLANE = 16 };

// Base class to hold GPU context structures
class GpuContext {
public:
    const GpuBackendType backendType;

    SNN_NO_COPY(GpuContext);
    SNN_NO_MOVE(GpuContext);

    virtual ~GpuContext() = default;

protected:
    GpuContext(GpuBackendType backendType_)
        : backendType(backendType_)
    {}
};

// Base class to hold GPU image
class GpuImageHandle {
protected:
    GpuImageHandle(GpuBackendType backendType_)
        : backendType(backendType_)
    {}

    virtual ~GpuImageHandle() = default;

public:
    const GpuBackendType backendType;
};

// Model type
enum ModelType { CLASSIFICATION, DETECTION, SEGMENTATION, OTHER };

// This structure holds output from special model types
struct SNNModelOutput {
    ModelType modelType = ModelType::OTHER;
    int classifierOutput;
    std::vector<std::vector<float>> detectionOutput;
};

} // namespace snn
