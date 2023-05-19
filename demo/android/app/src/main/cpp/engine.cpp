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
#include "pch.h"
#include "inferenceengine.h"
#include "demoutils.h"
#include "upload2GpuProcessor.h"
#include "resnet18Processor.h"
#include "mobileNetV2Processor.h"
#include "spatialDenoiser.h"
#include "yolov3Processor.h"
#include "styleTransferProcessor.h"

using namespace snn;

typedef std::vector<std::vector<float>> CPUBlob;
typedef std::vector<CPUBlob> BlobVec;

class Engine : public snn::InferenceEngine {
    struct FrameQueue {
        struct Item {
            FrameVec frames;
            BlobVec tensors;
            bool busy = false;
            SNNModelOutput modelOutput;
        };

        enum class Status {
            PRODUCING,
            CONSUMING,
            IDLE,
        };

        Status status = Status::IDLE;
        std::queue<std::shared_ptr<Item>> items;

        FrameQueue(const FrameImage2::Desc& cp, size_t vecSize, size_t queSize) {
            for (size_t i = 0; i < queSize; ++i) {
                items.emplace(std::make_shared<Item>());
                items.back()->frames.resize(vecSize);
                items.back()->tensors.resize(vecSize, CPUBlob());
                for (auto& f : items.back()->frames) {
                    f = createFrameImage(cp).release();
                }
                for (auto& t : items.back()->tensors) {
                    t = CPUBlob();
                }
            }

            SNN_LOGD("FrameVec initialized with sizes %u and %u", items.front()->frames.size(), items.front()->tensors.size());
            SNN_LOGD("FrameVec size is: %u", items.size());
        }

        ~FrameQueue() {}

        Item& beginProduce() {
            SNN_ASSERT(Status::IDLE == status);
            status    = Status::PRODUCING;
            auto item = items.front().get();
            SNN_ASSERT(!item->busy);
            return *item;
        }

        void endProduce() {
            SNN_ASSERT(Status::PRODUCING == status);
            status    = Status::IDLE;
            auto item = items.front();
            SNN_ASSERT(!item->busy);
            item->busy = true;
            // moves item to the end/back of the queue.
            items.push(item);
            items.pop();
        }

        void abortProduce() {
            SNN_ASSERT(Status::PRODUCING == status);
            status = Status::IDLE;
        }

        Item* beginConsume() {
            SNN_ASSERT(Status::IDLE == status);
            auto item = items.front().get();
            if (item->busy) {
                status = Status::CONSUMING;
                return item;
            } else {
                return nullptr;
            }
        }

        void endConsume() {
            SNN_ASSERT(Status::CONSUMING == status);
            status     = Status::IDLE;
            auto item  = items.front().get();
            item->busy = false;
        }
    };

    // Processing node. Each represents 1 step/layer in the processing pipeline.
    struct Node {
        std::unique_ptr<Processor> proc;
        FrameQueue* input;
        std::unique_ptr<FrameQueue> output;
        Timer timer;

        SNN_NO_COPY(Node);
        SNN_NO_MOVE(Node);

        Node(std::string name, const FrameImage2::Desc& cp, size_t outputSize, size_t queueSize): timer("node " + name) {
            // std::cout << cp << std::endl;
            output.reset(new FrameQueue(cp, outputSize, queueSize));
        }

        void run() {
            ScopedTimer st(timer);

            SNN_ASSERT(input != output.get());

            // get input
            auto i = input->beginConsume();
            if (!i) {
                return;
            }

            // get output
            auto& o = output->beginProduce();

            // run the processor
            SNNModelOutput modelOutput;
            Processor::Workload w = {i->frames.data(), i->tensors, i->frames.size(), o.frames[0], o.tensors, modelOutput};
            proc->submit(w);
            o.modelOutput = w.modelOutput;

            // done
            input->endConsume();
            output->endProduce();
        }
    };

    CreationParameters _cp;
    std::vector<std::unique_ptr<Node>> _nodes; // list of process nodes that is already sorted.
    std::unique_ptr<FrameQueue> _input;

public:
    // -------------------------------------------------------------------------
    //
    Engine(const CreationParameters& cp): _cp(cp) { initialize(); }

    // -------------------------------------------------------------------------
    //
    ~Engine() override {}

    Item beginEnqueue() override {
        auto& item      = _input->beginProduce();
        Item engineItem = {item.frames, item.tensors, {}};
        return engineItem;
    }

    void endEnqueue() override {
        _input->endProduce();
        for (auto& node : _nodes) {
            node->run();
        }
    }

    void abortEnqueue() override {
        _input->abortProduce();
        for (auto& node : _nodes) {
            node->run();
        }
    }

    Item beginDequeue() override {
        auto item = _nodes.back()->output->beginConsume();
        if (!item) {
            return {};
        }

        Item engineItem = {
            item->frames,
            item->tensors,
            item->modelOutput,
        };
        return engineItem;
    }

    void endDequeue() override { _nodes.back()->output->endConsume(); }

private:
    // -------------------------------------------------------------------------
    //
    void initialize() {
        // Initialize main proessing node(s) based on the algorithm, assuming input frames are on CPU.
        ColorFormat inputFormat = ColorFormat::RGBA8; // input format is currently hard coded to RGBA8
        uint32_t nFrames        = 1;
        if (_cp.algorithm) {
            // Remove unused variable first to make it run
            auto currentDevice = Device::CPU;
            (void) currentDevice;

            if (_cp.algorithm.styleTransferModels) {
                nFrames       = 1;
                currentDevice = Device::GPU;
                bool dumpOutputs = false;
                bool scale = false;
                float min = 0.0f;
                float max = 0.0f;
                if (_cp.algorithm.styleTransferModels.has_value()) {
                    dumpOutputs = _cp.algorithm.styleTransferModels.value().dumpOutputs;
                    scale       = _cp.algorithm.styleTransferModels.value().scale;
                    min         = _cp.algorithm.styleTransferModels.value().min;
                    max         = _cp.algorithm.styleTransferModels.value().max;
                }
                Processor::FrameVectorDesc d = {Device::CPU, inputFormat, nFrames, scale, min, max};
                _cp.compute = true;
                addNewNode("cpu->gpu", std::make_unique<Upload2GpuProcessor>(d));
                addNewNode("styleTransfer",
                           StyleTransferProcessor::createStyleTransfer(
                            _nodes.back()->proc->desc().o.format,
                            _cp.algorithm.styleTransferModels->styleTransferAlgorithm, _cp.algorithm.precision, _cp.compute, dumpOutputs));
            }

            if (_cp.algorithm.denoisers) {
                nFrames       = 1;
                currentDevice = Device::GPU;
                bool dumpOutputs = false;
                bool scale = false;
                float min = 0.0f;
                float max = 0.0f;
                if (_cp.algorithm.denoisers.has_value()) {
                    dumpOutputs = _cp.algorithm.denoisers.value().dumpOutputs;
                    scale       = _cp.algorithm.denoisers.value().scale;
                    min         = _cp.algorithm.denoisers.value().min;
                    max         = _cp.algorithm.denoisers.value().max;
                }
                Processor::FrameVectorDesc d = {Device::CPU, inputFormat, nFrames, scale, min, max};
                switch (_cp.algorithm.denoisers->denoiserAlgorithm) {
                case AlgorithmConfig::Denoisers::DenoiserAlgorithm ::SPATIALDENOIRSER:
                    addNewNode("cpu->gpu", std::make_unique<Upload2GpuProcessor>(d));
                    switch (_cp.algorithm.denoisers->denoiser) {
                    case AlgorithmConfig::Denoisers::Denoiser ::FRAGMENTSHADER:
                        _cp.compute = false;
                        break;
                    case AlgorithmConfig::Denoisers::Denoiser ::COMPUTESHADER:
                    default:
                        _cp.compute = true;
                        break;
                    }
                    addNewNode("denoiser",
                        SpatialDenoiser::createPreDenoiser(_nodes.back()->proc->desc().o.format, _cp.algorithm.precision, _cp.compute, dumpOutputs));
                    break;
                default:
                    break;
                }
            }

            if (_cp.algorithm.classifiers) {
                bool dumpOutputs = false;
                bool scale = false;
                float min = 0.0f;
                float max = 0.0f;
                if (_cp.algorithm.classifiers.has_value()) {
                    dumpOutputs = _cp.algorithm.classifiers.value().dumpOutputs;
                    scale       = _cp.algorithm.classifiers.value().scale;
                    min         = _cp.algorithm.classifiers.value().min;
                    max         = _cp.algorithm.classifiers.value().max;
                }
                Processor::FrameVectorDesc d = {Device::CPU, snn::ColorFormat::RGBA8, nFrames, scale, min, max};
                switch (_cp.algorithm.classifiers->classifier) {
                case AlgorithmConfig::Classifiers::Classifier::FRAGMENTSHADER:
                    _cp.compute = false;
                    break;
                case AlgorithmConfig::Classifiers::Classifier::COMPUTESHADER:
                default:
                    _cp.compute = true;
                    break;
                }
                switch (_cp.algorithm.classifiers->classifierAlgorithm) {
                case AlgorithmConfig::Classifiers::ClassifierAlgorithm::RESNET18:
                    SNN_LOGD("Current size of nodes is: %d", _nodes.size());
                    if (_nodes.size() == 0) {
                        addNewNode("cpu->gpu", std::make_unique<Upload2GpuProcessor>(d), 1);
                    }
                    addNewNode("resnet18",
                        ResNet18Processor::createResNet18Processor(_nodes.back()->proc->desc().o.format, _cp.algorithm.precision, _cp.compute, dumpOutputs),
                        1);

                    break;
                case AlgorithmConfig::Classifiers::ClassifierAlgorithm::MOBILENETV2:
                    SNN_LOGD("Current size of nodes is: %d", _nodes.size());
                    if (_nodes.size() == 0) {
                        addNewNode("cpu->gpu", std::make_unique<Upload2GpuProcessor>(d), 1);
                    }
                    addNewNode("mobilenetv2",
                        MobileNetV2Processor::createMobileNetV2Processor(_nodes.back()->proc->desc().o.format, _cp.algorithm.precision,
                            _cp.compute, dumpOutputs),
                        1);

                    break;
                case AlgorithmConfig::Classifiers::ClassifierAlgorithm::NONE:
                    break;
                }
            }

            if (_cp.algorithm.detections) {
                // TODO: Add Detection Logic
                SNN_LOGD("Start Detection Processing");
                bool dumpOutputs = false;
                bool scale = false;
                float min = 0.0f;
                float max = 0.0f;
                if (_cp.algorithm.detections.has_value()) {
                    dumpOutputs = _cp.algorithm.detections.value().dumpOutputs;
                    scale       = _cp.algorithm.detections.value().scale;
                    min         = _cp.algorithm.detections.value().min;
                    max         = _cp.algorithm.detections.value().max;
                }
                Processor::FrameVectorDesc d = {Device::CPU, inputFormat, nFrames, scale, min, max};
                switch (_cp.algorithm.detections->detection) {
                case AlgorithmConfig::Detections::Detection::FRAGMENTSHADER:
                    _cp.compute = false;
                    break;
                case AlgorithmConfig::Detections::Detection::COMPUTESHADER:
                default:
                    _cp.compute = true;
                    break;
                }
                switch (_cp.algorithm.detections->detectionAlgorithm) {
                case AlgorithmConfig::Detections::DetectionAlgorithm::YOLOV3:
                    addNewNode("cpu->gpu", std::make_unique<Upload2GpuProcessor>(d), 1);
                    addNewNode("yolov3",
                        Yolov3Processor::createYolov3Processor(_nodes.back()->proc->desc().o.format, _cp.algorithm.precision, _cp.compute, dumpOutputs),
                        1);

                    break;
                case AlgorithmConfig::Detections::DetectionAlgorithm::NONE:

                    break;
                }
            }
        } else {
            nFrames                      = 1;
            Processor::FrameVectorDesc d = {Device::CPU, inputFormat, nFrames};
            addNewNode("passthrough", std::make_unique<Upload2GpuProcessor>(d));
        }

        // create input frame images
        SNN_ASSERT(!_nodes.empty());
        auto& firstProcessorDesc = _nodes[0]->proc->desc();
        SNN_LOGI("Nodes in engine: %u", _nodes.size());
        for (std::size_t nodeIdx = 0; nodeIdx < _nodes.size(); nodeIdx++) {
            SNN_LOGI("%s", _nodes[nodeIdx]->proc->getModelName().c_str());
        }

        _input.reset(
            new FrameQueue(FrameImage2::Desc {firstProcessorDesc.i.device, firstProcessorDesc.i.format, _cp.width, _cp.height}, firstProcessorDesc.i.size, 1));
        _nodes[0]->input = _input.get();

        SNN_LOGI("Inference Engine initialized");
    }

    void addNewNode(std::string name, std::unique_ptr<Processor>&& p, uint32_t queueSize = 1) {
        auto& pd = p->desc();
        SNN_LOGD("Adding Node: %s", p->getModelName().c_str());
        SNN_LOGD("Output Size: %u", pd.o.size);
        FrameImage2::Desc ficp;
        ficp.device = pd.o.device;
        ficp.format = pd.o.format;
        if (p->hasDims()) {
            auto outputDims = p->getOutputDims();
            ficp.width      = outputDims.width;
            ficp.height     = outputDims.height;
            ficp.depth      = outputDims.depth;
            ficp.channels   = outputDims.channels;
        } else {
            if (!_nodes.empty()) {
                auto outputDims = _nodes[_nodes.size() - 1]->proc->getOutputDims();
                p->init(outputDims, outputDims);
                ficp.width  = outputDims.width;
                ficp.height = outputDims.height;
                ficp.depth  = outputDims.depth;
                ficp.channels = outputDims.channels;
            } else {
                // TODO: Fix this hardcoded logic
                p->init({_cp.width, _cp.height, 1, 4}, {_cp.width, _cp.height, 1, 4});
                ficp.width  = _cp.width;
                ficp.height = _cp.height;
                ficp.depth  = 1;
                ficp.channels = 4;
            }
        }
        _nodes.emplace_back(std::make_unique<Node>(formatString("[%d] - %s", _nodes.size(), name.c_str()), ficp, pd.o.size, queueSize));
        if (_nodes.size() > 1) {
            auto& prev     = _nodes[_nodes.size() - 2];
            auto& prevDesc = prev->proc->desc();

            SNN_CHK(prevDesc.o == pd.i); // make sure input of the new node is compatible with output of the last one.
            _nodes.back()->input = prev->output.get();
        }
        _nodes.back()->proc = std::move(p);
    }

    class FrameImageKernel {
    public:
        void operator()(FrameImage2* const* inputs, size_t count, const Range<uint8_t*, uint8_t*>& output) const { process(inputs, count, output); }

        template<class Proc, class... Args>
        static std::unique_ptr<Upload2GpuProcessor>
        createProc(const Args&... args) { // FIXME what does this mean? createProc necessarily uploads to GPU? this needs to be refactored
            auto proc = Proc {args...};
            return std::make_unique<Upload2GpuProcessor>(proc._inputDesc, proc);
        }

    protected:
        Processor::FrameVectorDesc _inputDesc;

        FrameImageKernel(Device device, uint64_t inVectorSize) {
            _inputDesc.device = device;
            _inputDesc.format = ColorFormat::RGBA8; // currently the pipeline is hardcoded to process RGBA8 data.
            _inputDesc.size   = static_cast<uint32_t>(inVectorSize);
        }

        virtual void process(FrameImage2* const* inputs, size_t count, const Range<uint8_t*, uint8_t*>& output) const = 0;
    };
};

// -----------------------------------------------------------------------------
//
InferenceEngine* InferenceEngine::createInstance(const CreationParameters& cp) { return new Engine(cp); }
