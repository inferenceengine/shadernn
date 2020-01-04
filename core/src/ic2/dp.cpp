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
#include "pch.h"
#include "dp.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <deque>

using namespace snn;
using namespace snn::dp;

DECLARE_LAYER(InputLayer);
DECLARE_LAYER(Conv2D);
DECLARE_LAYER(Conv2DTranspose);
DECLARE_LAYER(Subpixel);
DECLARE_LAYER(Concatenate);
DECLARE_LAYER(Calculate);
DECLARE_LAYER(UpSampling2D);
DECLARE_LAYER(Add);
DECLARE_LAYER(SeparableConv2D);
DECLARE_LAYER(Dense);
DECLARE_LAYER(MaxPooling2D);
DECLARE_LAYER(AveragePooling2D);
DECLARE_LAYER(AdaptiveAvgPool2d);
DECLARE_LAYER(Flatten);
DECLARE_LAYER(Pad);
DECLARE_LAYER(BatchNormalization);
DECLARE_LAYER(InstanceNorm);

std::unordered_map<std::string, snn::dp::LayerCreator> LayerRegistryDict;

// a small BFS graph traverse utility function
template<typename T, typename GetChildren, typename F>
void BFSTraverse(T& head, GetChildren getChildren, F f) {
    if (!head) {
        return;
    }
    std::queue<T> q;
    std::set<T> visited;
    q.push(head);
    visited.insert(head);
    while (!q.empty()) {
        auto current = q.front();
        q.pop();
        for (auto& c : getChildren(current)) {
            if (visited.end() == visited.find(c)) {
                q.push(c);
                visited.insert(c);
            }
        }
        f(current);
    }
}

std::vector<std::shared_ptr<GenericModelLayer>> topologicalSort(std::shared_ptr<GenericModelLayer> head) {
    std::vector<std::shared_ptr<GenericModelLayer>> sortedNodes;

    // For each visited and not processed node we store the current number of inputs and other information
    std::unordered_set<std::shared_ptr<GenericModelLayer>> visitedNodes;
    std::unordered_map<std::shared_ptr<GenericModelLayer>, size_t> processingNodes;

    // This nodes are ready to go
    std::deque<std::shared_ptr<GenericModelLayer>> pendingNodes;

    BFSTraverse(
        head, [](std::shared_ptr<GenericModelLayer> s) { return s->nextLayers; },
        [&](std::shared_ptr<GenericModelLayer> current) {
            if (current->prevLayers.empty()) {
                pendingNodes.push_front(current);
                processingNodes[current] = 0;
            }
        });

    // if (pendingNodes.empty()) {
    //     return "All incoming nodes are internal";
    // }

    while (!pendingNodes.empty()) {
        std::shared_ptr<GenericModelLayer> node;
        node.reset(pendingNodes[0].get());
        // Either do a real computation or mock test or build a piece of execution plan
        sortedNodes.push_back(node);
        // Done with this node. It cannot be visited again in acyclic praph
        pendingNodes.erase(pendingNodes.begin());
        processingNodes.erase(node);
        if (!visitedNodes.insert(node).second) {
            SNN_RIP("Not a DAG - cycle in graph or incorrect number of input nodes");
        }
        // Iterate through next nodes
        for (auto nextNode : node->nextLayers) {
            // Increment the number of ready inputs for every next node
            if (++processingNodes[nextNode] == nextNode->prevLayers.size()) {
                // This node is ready to go. Insert it in the beginning
                pendingNodes.push_front(nextNode);
            }
            // If not ready, it must be visited again
        }
    }

    // All nodes should have been processed";
    SNN_ASSERT(processingNodes.empty());

    // done
    return sortedNodes;
}

void null_deleter(snn::dp::GenericModelLayer* layer) {
    (void) layer;
    return;
}

bool snn::dp::registerLayer(std::string layerName, LayerCreator creator) {
    // struct ShaderLayerEntry entry = {layerName, creator};
    // LayerRegistry.push_back(entry);
    SNN_LOGD("%s:%d register layer: %s\n", __FUNCTION__, __LINE__, layerName.c_str());
    LayerRegistryDict.emplace(layerName, creator);
    return 0;
}

bool snn::dp::initLayerRegisty() {
    REGISTER_LAYER(InputLayer);
    REGISTER_LAYER(Conv2D);
    REGISTER_LAYER(Conv2DTranspose);
    REGISTER_LAYER(Subpixel);
    REGISTER_LAYER(Concatenate);
    REGISTER_LAYER(Calculate);
    REGISTER_LAYER(UpSampling2D);
    REGISTER_LAYER(Add);
    REGISTER_LAYER(SeparableConv2D);
    REGISTER_LAYER(Dense);
    REGISTER_LAYER(MaxPooling2D);
    REGISTER_LAYER(AveragePooling2D);
    REGISTER_LAYER(AdaptiveAvgPool2d);
    REGISTER_LAYER(Flatten);
    REGISTER_LAYER(Pad);
    REGISTER_LAYER(BatchNormalization);
    REGISTER_LAYER(InstanceNorm);
    return 0;
}

snn::dp::GenericModelLayer* createLayerInstance(std::string layerName, ModelParser& parser, int i) {
    snn::dp::GenericModelLayer* ret = NULL;
    if (layerName == "DepthwiseConv2D" || layerName == "Depthwise") {
        layerName = "SeparableConv2D";
    }
    if (layerName == "InstanceNormalization") {
        layerName = "InstanceNorm";
    }
    if (layerName == "ZeroPadding2D") {
        layerName = "Pad";
    }

    if (LayerRegistryDict.find(layerName) != LayerRegistryDict.end()) {
        auto func = LayerRegistryDict[layerName];
        SNN_LOGD("%s:%d found layer: %s\n", __FUNCTION__, __LINE__, layerName.c_str());
        ret = func(parser, i);
    } else {
        SNN_LOGD("%s:%d Not found layer: %s\n", __FUNCTION__, __LINE__, layerName.c_str());
    }
    return ret;
}

std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> snn::dp::loadFromJsonModel(const std::string& fileName, const snn::MRTMode& mrtMode,
                                                                                    const snn::WeightAccessMethod& weightMode, bool preferHp) {
    std::vector<std::shared_ptr<snn::dp::GenericModelLayer>> layers;
    ModelParser parser({fileName, preferHp, mrtMode, weightMode});
    int32_t layerCount = parser.getLayerCount();
    int headNodeIndex  = -1;
    std::string kernel; // kernel name (for creating layer name)

    initLayerRegisty();

    for (int i = 0; i < layerCount; i++) {
        int numInbound = parser.getNumInbound(i);
        SNN_ASSERT(numInbound == (int) parser.getInboundLayerId(i).size());
        const auto& layerName = parser.getLayerName(i);

        auto newLayer = createLayerInstance(layerName, parser, i);
        layers.emplace_back(std::shared_ptr<GenericModelLayer>(newLayer, &null_deleter));
        headNodeIndex = 0;
        (void) numInbound;

        layers.back()->setName(snn::formatString("%s layer [%02d] %s", fileName.c_str(), i, layerName.c_str()));
    }

    // make sure there's an input layer defined.
    if (headNodeIndex < 0) {
        SNN_LOGE("head layer not found.");
        return {};
    }

    // build layer connections
    for (int i = 0; i < layerCount; i++) {
        std::vector<int> inLayerIndices = parser.getInboundLayerId(i);
        int numInbound                  = static_cast<int>(inLayerIndices.size());
        for (auto ii : inLayerIndices) {
            layers[i]->prevLayers.push_back(layers[ii]);
        }
        SNN_ASSERT(numInbound == parser.getNumInbound(i));
        for (int j = 0; j < numInbound; j++) {
            layers[inLayerIndices[j]]->nextLayers.push_back(layers[i]);
        }
    }

    // move head node to slot 0 of the array.
    if (headNodeIndex != 0) {
        std::swap(layers[0], layers[headNodeIndex]);
    }

    // TODO: calculate layer scale and translation factor

    SNN_LOGD("Model loaded from JSON");

    return layers;
}

snn::InferenceGraph snn::dp::generateInferenceGraph(std::shared_ptr<GenericModelLayer> head, const ShaderGenOptions& options) {
    // generate an topological sorted shader list
    auto shaders = topologicalSort(head);

    InferenceGraph graph;
    graph.mrtMode    = options.mrtMode;
    graph.weightMode = options.weightMode;
    std::map<std::shared_ptr<GenericModelLayer>, InferenceGraph::Layer*> s2l;
    std::map<InferenceGraph::Layer*, std::shared_ptr<GenericModelLayer>> l2s;

    // create a collection of graph layers
    for (auto s : shaders) {
        // Currently fixed. We can set this per layer based on layer params later.
        graph.layers.emplace_back(new InferenceGraph::Layer);
        auto l = graph.layers.back().get();

        l->imageTextureFunPtr = [s](FixedSizeArray<snn::ImageTexture>& inputMat, FixedSizeArray<snn::ImageTexture>& outputMat) {
            s->computeImageTexture(inputMat, outputMat);
        };

        s2l[s] = l;
        l2s[l] = s;

        l->layerLoc = s->getLayerExecutionLevel();
    }

    // drop the dummy input layer
    auto inputShader = *shaders.begin();
    inputShader->getInputDims(graph.inputWidth, graph.inputHeight, graph.inputChannels);
    graph.layers.erase(graph.layers.begin());

    auto prevOutputWidth = options.desiredInput.width;
    auto prevOutputHight = options.desiredInput.height;

    int numInputUnits  = options.desiredInput.width * options.desiredInput.height;
    int numOutputUnits = numInputUnits;

    auto prevLayerLoc = graph.layers.at(0)->layerLoc;
    std::ostringstream modelFormat;
    modelFormat << "================================================================\n";
    modelFormat << "|  Layer ID  |              Name                 | Output Dims |\n";
    modelFormat << "================================================================\n";
    // loop through all layers
    std::map<InferenceGraph::Layer*, size_t> l2i;
    for (size_t i = 0; i < graph.layers.size(); ++i) {
        auto layer  = graph.layers[i].get();
        auto shader = l2s[layer];
        shader->setMRTMode(options.mrtMode);
        shader->setWeightAccessMode(options.weightMode);

        // build a layer to array index map
        l2i[layer] = i;

        uint32_t inputWidth = 0, inputHeight = 0, inputDepth = 0, width = 0, height = 0, depth = 0;
        // build input array for each layer
        for (auto& prev : shader->prevLayers) {
            InferenceGraph::BufferRef ref;
            if (prev == head) {
                ref.isStageOutput = false;
                ref.index         = 0; // TODO: is this always zero?
                auto imageInput   = InferenceGraph::Buffer {options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F,
                                                          options.desiredInput.width, options.desiredInput.height, options.desiredInput.depth, 4};
                shader->addInputDim(imageInput);
                inputWidth       = std::max(inputWidth, imageInput.width);
                inputHeight      = std::max(inputHeight, imageInput.height);
                inputDepth       = std::max(inputDepth, imageInput.depth);
                auto shaderInput = shader->getInputDims(0);
            } else {
                ref.isStageOutput = true;
                ref.index         = l2i[s2l[prev]];
                auto imageInput   = graph.layers[ref.index].get()->output;
                shader->addInputDim(imageInput);
                inputWidth  = std::max(inputWidth, imageInput.width);
                inputHeight = std::max(inputHeight, imageInput.height);
                inputDepth  = std::max(inputDepth, imageInput.depth);
            }
            layer->inputs.push_back(ref);
        }
        shader->getOutputDims(width, height, depth);
        auto name      = shader->getName();
        auto layerName = name.substr(name.find("["), std::string::npos);
        // Format the model structure for output
        int idxBuffer  = i > 9 ? 10 : 11;
        idxBuffer      = (i > 99) ? 9 : idxBuffer;
        int nameBuffer = 35 - layerName.length();
        auto dimsStr   = std::to_string(width) + " x " + std::to_string(height) + " x " + std::to_string(depth);
        int dimsBuffer = 12 - dimsStr.length();

        if (nameBuffer - 1 < 0) {
            layerName = layerName.substr(0, 31) + "...";
        }

        modelFormat << "|" << std::string(idxBuffer / 2, ' ') << i << std::string((idxBuffer % 2 == 0) ? (idxBuffer / 2) : (idxBuffer / 2 + 1), ' ') << "| ";
        modelFormat << layerName << ((nameBuffer - 1) > 0 ? std::string(nameBuffer - 1, ' ') : "") << "| ";
        modelFormat << dimsStr << (dimsBuffer > 0 ? std::string(dimsBuffer, ' ') : "") << "|\n";
        SNN_LOGD("%%%%%%%% layer: %zu, name : %s, output dim: %d %d %d\n", i, shader->getName().c_str(), width, height, depth);

        if (layer->layerLoc == snn::InferenceGraph::LayerExecution::GPU_FS || layer->layerLoc == snn::InferenceGraph::LayerExecution::GPU_CS) {
            ShaderLayer::LayerGenOptions opt;
            (ShaderGenOptions&) opt = options;
            opt.desiredInput.width  = inputWidth; // FIXME: what if the input layer is not right infront of current layer?
            opt.desiredInput.height = inputHeight;
            opt.desiredOutputWidth  = width;
            opt.desiredOutputHeight = height;
            opt.isFirstLayer        = (i == 0);
            opt.isLastLayer         = (i == (graph.layers.size() - 1));
            // opt.compute = (layer->layerLoc == snn::InferenceGraph::LayerExecution::GPU_CS);
            opt.compute      = options.compute;
            opt.mrtMode      = options.mrtMode;
            opt.weightMode   = options.weightMode;
            auto glslShaders = shader->createGLSLShader(opt);
            layer->layerLoc  = shader->getLayerExecutionLevel();
            layer->passes    = std::move(glslShaders.passes);

            layer->output = {
                options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F, width, height, DIV_4_ROUND_UP(shader->getDesc().numOutputPlanes),
                shader->getDesc().numOutputPlanes
                // std::vector<std::vector<float>>(),
            };

            prevOutputWidth = width;
            prevOutputHight = height;

            SNN_ASSERT(layer->output.width > 0);
            SNN_ASSERT(layer->output.height > 0);
            SNN_ASSERT(layer->output.depth > 0);
            prevLayerLoc        = layer->layerLoc;
            layer->flattenLayer = false;

            SNN_LOGD("Test:%s:%d: Layer: %zu %s, layer output: %d:%d:%d, input: width:%d, height:%d, depth:%d, output: width:%d, height:%d, depth:%d\n",
                     __FUNCTION__, __LINE__, i, shader->getName().c_str(), layer->output.width, layer->output.height, layer->output.depth,
                     opt.desiredInput.width, opt.desiredInput.height, opt.desiredInput.depth, opt.desiredOutputWidth, opt.desiredOutputHeight,
                     DIV_4_ROUND_UP(shader->getDesc().numOutputPlanes));

        } else {
            if (prevLayerLoc == snn::InferenceGraph::LayerExecution::GPU_FS || prevLayerLoc == snn::InferenceGraph::LayerExecution::GPU_CS) {
                layer->flattenLayer = true;
                numInputUnits       = prevOutputWidth * prevOutputHight;
                numOutputUnits      = (int) (width);
            } else {
                numInputUnits       = numOutputUnits;
                numOutputUnits      = (int) (numOutputUnits * numInputUnits);
                layer->flattenLayer = false;
            }
            snn::dp::CpuGenOptions opt;
            opt.isFirstLayer = (i == 0);
            opt.isLastLayer  = (i == (graph.layers.size() - 1));
            opt.inputDims    = numInputUnits;
            opt.outputDims   = numOutputUnits;
            auto cpuPasses   = shader->createCPUPasses(opt);
            layer->passes.push_back(std::move(cpuPasses.passes));

            layer->output = {
                options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F, width, height, depth, shader->getDesc().numOutputPlanes
                // std::vector<std::vector<float>>()
            };

            prevLayerLoc = layer->layerLoc;
        }

        layer->name = shader->name;
        // std::cout << "Layer " << shader->getName() << " Loaded" << std::endl;
        modelFormat << "----------------------------------------------------------------\n";
    }

    head = shaders.at(1);

    // std::cout << head->getName() << std::endl;

    // setup input array of the whole graph
    auto& inputDesc = head->getDesc();

    // std::cout << inputDesc.numInputPlanes << std::endl;

    // graph.inputs.resize(DIV_4_ROUND_UP(inputDesc.numInputPlanes));
    graph.inputs.resize(1); // Hacked for testing
    (void) inputDesc;
    for (auto& i : graph.inputs) {
        i = options.desiredInput;
    }

    modelFormat << "================================================================\n";
    SNN_LOGI("\n%s", modelFormat.str().c_str());

    // std::cout << "Inference graph generated" << std::endl;
    SNN_LOGD("%s:%d\n", __FUNCTION__, __LINE__);

    return graph;
}
