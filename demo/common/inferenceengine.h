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
#include <optional>
#include <vector>

namespace snn {

class FrameImage2;

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

        Precision precision = Precision::FP32;

        operator bool() const { return denoisers || basicCNN || classifiers || detections || styleTransferModels; }

        auto operator==(const AlgorithmConfig& that) const -> bool {
            return that.denoisers == denoisers && that.basicCNN == basicCNN && that.classifiers == classifiers && that.detections == detections &&
                    that.styleTransferModels == styleTransferModels;
        }
        auto operator!=(const AlgorithmConfig& that) const -> bool { return !operator==(that); }
    };

    virtual ~InferenceEngine() = default;

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

}
