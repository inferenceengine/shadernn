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
#include "image.h"
#include "defines.h"
#include <stdint.h>
#include <memory>
#include <variant>
#include <optional>
#include <vector>
#include <optional>

#ifdef _MSC_VER
    #pragma warning(disable : 4201) // nameless struct/union
#endif

#ifndef __ANDROID__
    #define OUTPUT_DIR "../../../../core/inferenceCoreDump"
#else
    #define OUTPUT_DIR "/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump"
#endif

#define DIV_AND_ROUND_UP(x, y) ((x + (y - 1)) / y)
#ifdef __ANDROID__
    #define DUMP_DIR   "/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump"
    #define ASSETS_DIR "/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files"
#else
    #define DUMP_DIR   "../../../../core/inferenceCoreDump/"
    #define ASSETS_DIR "../../../../core/data/"
#endif

namespace snn {
enum Device {
    GPU,
    CPU,
    CPU_GPU,
    GPU_CPU,
};

enum class WeightAccessMethod { CONSTANTS, TEXTURES, UNIFORM_BUFFER, SSBO_BUFFER };

enum class MRTMode { SINGLE_PLANE = 4, DOUBLE_PLANE = 8, QUAD_PLANE = 16 };

// Fence class that is used to mark the end of an async operation. So that Fence(0) is reserved and never pending.
union Fence {
    uint64_t u64;
    struct {
        uint64_t value : 62;
        Device type : 4;
    };

    Fence(uint64_t u64 = 0): u64(u64) {}

    operator uint64_t() const { return u64; }
};

// singleton fence manager
class SNN_API FenceManager {
public:
    static std::shared_ptr<FenceManager> getInstance();

    ~FenceManager();

    Fence insertGpuFence();

    Fence createCpuFence();
    void signalCpuFence(Fence);

    bool isFencePending(Fence);
    void waitForFence(Fence);

private:
    class Impl;

    Impl* _impl;

    FenceManager();

    SNN_NO_COPY(FenceManager);
    SNN_NO_MOVE(FenceManager);
};

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

    struct OnGPU {
        GLenum target;
        GLuint texture;
        bool empty() const { return 0 == target || 0 == texture; }
    };
    // Returns' empty structure with target and texture set to 0, if the frame image does not support this operation.
    virtual OnGPU getGpuData() const { return {}; }

    // Return's empty image, if the frame image does not support this operation.
    virtual RawImage& getCpuData() const {
        static RawImage emptyImage;
        return emptyImage;
    }

protected:
    FrameImage2() = default;

    Desc _desc;
};

struct InferenceEngine {
    struct AlgorithmConfig {
        struct Denoisers {
            enum class Denoiser { NONE, COMPUTESHADER, FRAGMENTSHADER };

            enum class DenoiserAlgorithm { NONE, AIDENOISER, SPATIALDENOIRSER };
            Denoiser denoiser                   = Denoiser ::FRAGMENTSHADER;
            DenoiserAlgorithm denoiserAlgorithm = DenoiserAlgorithm ::AIDENOISER;
            bool dumpOutputs                    = false;

            bool scale = false;
            float min  = 0.0f;
            float max  = 255.0f;

            auto operator==(const Denoisers& that) const -> bool { return that.denoiser == denoiser && that.denoiserAlgorithm == denoiserAlgorithm; }

            auto operator!=(const Denoisers& that) const -> bool { return !operator==(that); }
        };
        struct Clearance {
            enum class Version { SPRINT3, SPRINT4, SPRINT5 };

            Version version;
            Device device;
            bool temporalFiltering;
            bool dumpOutputs;

            bool scale = false;
            float min  = 0.0f;
            float max  = 255.0f;

            Clearance(): version(Version::SPRINT5), device(Device::CPU), temporalFiltering(false) {}

            Clearance(Version v, Device d, bool tmpF): version(v), device(d), temporalFiltering(tmpF) {}

            auto operator==(const Clearance& that) const -> bool {
                return that.version == version && that.device == device && that.temporalFiltering == temporalFiltering;
            }
            auto operator!=(const Clearance& that) const -> bool { return !operator==(that); }
        };

        struct BasicCNN {
            Device device;
            bool scale       = false;
            bool dumpOutputs = false;
            float min        = 0.0f;
            float max        = 255.0f;

            auto operator==(const BasicCNN& that) const -> bool {
                (void) that;
                return true;
            }
            auto operator!=(const BasicCNN& that) const -> bool { return !operator==(that); }
        };

        struct Classifiers {
            enum class Classifier { FRAGMENTSHADER, COMPUTESHADER };

            enum class ClassifierAlgorithm { NONE, RESNET18, MOBILENETV2 };

            bool dumpOutputs;
            Classifier classifier                   = Classifier::FRAGMENTSHADER;
            ClassifierAlgorithm classifierAlgorithm = ClassifierAlgorithm::NONE;

            bool scale = true;
            float min  = -1.0f;
            float max  = 1.0f;

            auto operator==(const Classifiers& that) const -> bool { return that.classifier == classifier && that.classifierAlgorithm == classifierAlgorithm; }

            auto operator!=(const Classifiers& that) const -> bool { return !operator==(that); }
        };

        struct Detections {
            enum class Detection { FRAGMENTSHADER, COMPUTESHADER };

            enum class DetectionAlgorithm { NONE, YOLOV3 };

            Detection detection                   = Detection::FRAGMENTSHADER;
            DetectionAlgorithm detectionAlgorithm = DetectionAlgorithm::NONE;
            bool dumpOutputs;

            bool scale = false;
            float min  = 0.0f;
            float max  = 255.0f;

            auto operator==(const Detections& that) const -> bool { return that.detection == detection && that.detectionAlgorithm == detectionAlgorithm; }

            auto operator!=(const Detections& that) const -> bool { return !operator==(that); }
        };

        struct StyleTransfer {
            enum class StyleTransferAlgorithm { NONE, CANDY, MOSAIC, POINTILISM, RAIN_PRINCESS, UDNIE };

            StyleTransferAlgorithm styleTransferAlgorithm = StyleTransferAlgorithm::NONE;
            bool dumpOutputs                              = false;

            bool scale = false;
            float min  = 0.0f;
            float max  = 255.0f;

            auto operator==(const StyleTransfer& that) const -> bool { return that.styleTransferAlgorithm == styleTransferAlgorithm; }

            auto operator!=(const StyleTransfer& that) const -> bool { return !operator==(that); }
        };

        std::optional<Denoisers> denoisers;
        std::optional<BasicCNN> basicCNN;
        std::optional<Classifiers> classifiers;
        std::optional<Detections> detections;
        std::optional<StyleTransfer> styleTransferModels;

        operator bool() const { return denoisers || basicCNN || classifiers || detections || styleTransferModels; }

        auto operator==(const AlgorithmConfig& that) const -> bool {
            return that.denoisers == denoisers && that.basicCNN == basicCNN && that.classifiers == classifiers && that.detections == detections &&
                    that.styleTransferModels == styleTransferModels;
        }
        auto operator!=(const AlgorithmConfig& that) const -> bool { return !operator==(that); }
    };

    virtual ~InferenceEngine() = default;

    enum ModelType { CLASSIFICATION, DETECTION, SEGMENTATION, OTHER };

    struct SNNModelOutput {
        ModelType modelType = ModelType::OTHER;
        int classifierOutput;
    };

    typedef struct Item {
        std::vector<FrameImage2*> frames;
        std::vector<std::vector<std::vector<float>>> tensors;
        SNNModelOutput snnModelOutput;
    } Item;

    typedef std::vector<FrameImage2*> FrameVec;

    virtual auto beginEnqueue() -> Item = 0;
    virtual void endEnqueue()           = 0;
    virtual void abortEnqueue()         = 0;
    virtual auto beginDequeue() -> Item = 0;
    virtual void endDequeue()           = 0;

    struct CreationParameters {
        AlgorithmConfig algorithm;

        uint32_t width, height;

        // run the inference engine processing pipeline in fully serialized way. this hurts performance a lot,
        // but also makes debugging the pipeline a lot easier.
        bool serialized = false;

        bool compute = false; // set to true to use compute shader when available.
    };

    SNN_API static InferenceEngine* createInstance(const CreationParameters&); // create a real instance.

protected:
    InferenceEngine() = default;

private:
    SNN_NO_COPY(InferenceEngine);
    SNN_NO_MOVE(InferenceEngine);
};

struct FrameProvider2 {
    virtual ~FrameProvider2() = default;

    // returns false, if the data is not ready yet.
    virtual bool fetchData(const InferenceEngine::FrameVec&) = 0;

    // System only: return frame to original buffer (NV12)
#ifdef SYSTEM_LIB
    virtual void getOutput(const InferenceEngine::FrameVec& frames, void* output, size_t width, size_t height) = 0;
#endif
protected:
    FrameProvider2() = default;

private:
    SNN_NO_COPY(FrameProvider2);
    SNN_NO_MOVE(FrameProvider2);
};
} // namespace snn
