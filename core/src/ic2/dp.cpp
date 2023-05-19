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
#include "dp.h"
#include "layerFactory.h"
#include <string>
#include <algorithm>
#include <sstream>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <deque>

using namespace snn;
using namespace snn::dp;

// a small BFS graph traverse utility function
// params:
//  head - head of the graph
//  getChildren - child nodes iterable
//  f - function to perform on each node
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

// Topological sort of the graph with GenericModelLayer objects
// params:
//  head - the head of the graph
// returns:
//  sorted list of GenericModelLayer objects
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

    return sortedNodes;
}

void null_deleter(snn::dp::GenericModelLayer* layer) {
    (void) layer;
    return;
}

std::vector<std::shared_ptr<GenericModelLayer>> snn::dp::loadFromJsonModel(const std::string& fileName, bool useVulkan, const MRTMode& mrtMode,
                                                                           const WeightAccessMethod& weightMode, bool preferHp) {
    std::vector<std::shared_ptr<GenericModelLayer>> layers;
    ModelParser parser({fileName, preferHp, mrtMode, weightMode});
    int32_t layerCount = parser.getLayerCount();
    int headNodeIndex  = -1;
    std::string kernel; // kernel name (for creating layer name)

    initLayerRegisty();

    for (int i = 0; i < layerCount; i++) {
        int numInbound = parser.getNumInbound(i);
        SNN_ASSERT(numInbound == (int) parser.getInboundLayerId(i).size());
        const auto& layerName = parser.getLayerName(i);

        auto newLayer = createLayerInstance(layerName, parser, i, useVulkan);
        layers.emplace_back(std::shared_ptr<GenericModelLayer>(newLayer, &null_deleter));
        headNodeIndex = 0;
        (void) numInbound;

        layers.back()->setName(formatString("%s layer [%02d] %s", fileName.c_str(), i, layerName.c_str()));
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

InferenceGraph snn::dp::generateInferenceGraph(std::shared_ptr<GenericModelLayer> head, const ShaderGenOptions& options) {
    // generate an topological sorted shader list
    auto modelLayers = topologicalSort(head);
    InferenceGraph graph;
    graph.mrtMode    = options.mrtMode;
    graph.weightMode = options.weightMode;
    std::map<std::shared_ptr<GenericModelLayer>, InferenceGraph::Layer*> s2l;
    std::map<InferenceGraph::Layer*, std::shared_ptr<GenericModelLayer>> l2s;

    uint32_t inputLayers = 0;

    // create a collection of graph layers
    for (auto modelLayer : modelLayers) {
        // Currently fixed. We can set this per layer based on layer params later.
        graph.layers.emplace_back(new InferenceGraph::Layer);
        auto igLayer = graph.layers.back().get();

        igLayer->imageTextureFunPtr = [modelLayer](ImageTextureArray& inputMat, ImageTextureArray& outputMat) {
            modelLayer->computeImageTexture(inputMat, outputMat);
        };

        igLayer->initFunPtr = [modelLayer](DeviceBackend *backend, ImageTextureArray& inputMat, ImageTextureArray& outputMat) {
            modelLayer->init(backend, inputMat, outputMat);
        };

        igLayer->runFunPtr = [modelLayer](DeviceBackend *backend, bool dumpOutputs) {
            modelLayer->run(backend, dumpOutputs);
        };

        s2l[modelLayer] = igLayer;
        l2s[igLayer] = modelLayer;

        igLayer->layerLoc = modelLayer->getLayerExecutionType();
        igLayer->name = modelLayer->getName();
        igLayer->isInputLayer = modelLayer->isInputLayer();
        if (modelLayer->isInputLayer()) {
            inputLayers++;
            igLayer->inputIndex = modelLayer->getInputIndex();
        }
        SNN_LOGD("layer name: %s, loc: %d, input: %d, # inputs: %d", igLayer->name.c_str(),
            (int)igLayer->layerLoc, igLayer->isInputLayer, inputLayers);
    }

    auto prevOutputWidth = options.desiredInput[0].width;
    auto prevOutputHight = options.desiredInput[0].height;

    int numInputUnits  = options.desiredInput[0].width * options.desiredInput[0].height;
    int numOutputUnits = numInputUnits;

    InferenceGraph::LayerExecutionType prevLayerLoc = InferenceGraph::LayerExecutionType::NOT_DEFINED;
    std::ostringstream modelFormat;
    modelFormat << "================================================================\n";
    modelFormat << "|  Layer ID  |              Name                 | Output Dims |\n";
    modelFormat << "================================================================\n";
    // loop through all layers
    std::map<InferenceGraph::Layer*, size_t> l2i;
    for (size_t i = 0; i < graph.layers.size(); ++i) {
        auto igLayer  = graph.layers[i].get();
        auto modelLayer = l2s[igLayer];
        modelLayer->setMRTMode(options.mrtMode);
        modelLayer->setWeightAccessMode(options.weightMode);

        // build a layer to array index map
        l2i[igLayer] = i;

        uint32_t inputWidth = 0, inputHeight = 0, width = 0, height = 0, depth = 0;
        if (modelLayer->prevLayers.size() > 0) {
            // build input array for each layer
            for (auto& prevModelLayer : modelLayer->prevLayers) {
                InferenceGraph::LayerRef ref;
                if (prevModelLayer->isInputLayer()) {
                    ref.isStageOutput = false;
                    ref.index         = l2i[s2l[prevModelLayer]];
                    auto inputIdx = prevModelLayer->getInputIndex();
                    auto imageInput   = InferenceGraph::IODesc {options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F,
                                                            options.desiredInput[inputIdx].width, options.desiredInput[inputIdx].height,
                                                            options.desiredInput[inputIdx].depth, 4 * options.desiredInput[inputIdx].depth};
                    modelLayer->addInputDim(imageInput);
                    inputWidth       = std::max(inputWidth, imageInput.width);
                    inputHeight      = std::max(inputHeight, imageInput.height);
                } else {
                    ref.isStageOutput = true;
                    ref.index         = l2i[s2l[prevModelLayer]];
                    const InferenceGraph::IODesc& imageInput   = graph.layers[ref.index].get()->outputDesc;
                    modelLayer->addInputDim(imageInput);
                    inputWidth  = std::max(inputWidth, imageInput.width);
                    inputHeight = std::max(inputHeight, imageInput.height);
                }
                igLayer->inputRefs.push_back(ref);
            }
            modelLayer->getOutputDims(width, height, depth);
            SNN_LOGD("%%%%%%%% layer: %zu, name : %s, prev:%zu", i, modelLayer->getName().c_str(), igLayer->inputRefs.size());
        } else {  // For all of the Input layers
            InferenceGraph::LayerRef ref;
            ref.isStageOutput = false;
            ref.index         = -1;
            auto inputIdx = modelLayer->getInputIndex();
            auto imageInput   = InferenceGraph::IODesc {options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F,
                                                        options.desiredInput[inputIdx].width, options.desiredInput[inputIdx].height,
                                                        options.desiredInput[inputIdx].depth, 4 * options.desiredInput[inputIdx].depth};
            modelLayer->addInputDim(imageInput);
            inputWidth =  imageInput.width;
            inputHeight = imageInput.height;
            modelLayer->getOutputDims(width, height, depth);
            igLayer->inputRefs.push_back(ref);
            SNN_LOGD("%%%%%%%% layer: %zu, name : %s, prev:%zu", i, modelLayer->getName().c_str(), igLayer->inputRefs.size());
        }

        auto name      = modelLayer->getName();
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
        SNN_LOGD("%%%%%%%% layer = %zu, name  = %s, output dim = %d %d %d loc = %d", i, modelLayer->getName().c_str(), width, height, depth,
            (int)igLayer->layerLoc);

        if (igLayer->layerLoc == InferenceGraph::LayerExecutionType::GPU_FS || igLayer->layerLoc == InferenceGraph::LayerExecutionType::GPU_CS
            || igLayer->layerLoc == InferenceGraph::LayerExecutionType::GPU_VK) {
            ShaderLayer::LayerGenOptions opt;
            (ShaderGenOptions&) opt = options;
            opt.desiredInput[0].width  = inputWidth; // FIXME: what if the input layer is not right in front of current layer?
            opt.desiredInput[0].height = inputHeight;

            opt.desiredOutputWidth  = width;
            opt.desiredOutputHeight = height;
            opt.isFirstLayer        = (i == inputLayers);
            opt.isLastLayer         = (i == (graph.layers.size() - 1));
            opt.compute      = options.compute;
            opt.mrtMode      = options.mrtMode;
            opt.weightMode   = options.weightMode;
            opt.vulkan       = options.vulkan;
            if (modelLayer->isInputLayer()) {
                if ((options.vulkan)) {
                    modelLayer->setLayerExecutionType(snn::InferenceGraph::LayerExecutionType::GPU_VK);
                } else if ((options.compute)) {
                    modelLayer->setLayerExecutionType(snn::InferenceGraph::LayerExecutionType::GPU_CS);
                } else {
                    modelLayer->setLayerExecutionType(snn::InferenceGraph::LayerExecutionType::GPU_FS);
                }
                SNN_LOGD("%%%%%%%% layer: %zu, name : %s, output dim: %d %d %d loc: %d", i, modelLayer->getName().c_str(), width, height, depth,
                    (int)igLayer->layerLoc);
            } else {
                modelLayer->createInferencePasses(opt);
                SNN_LOGD("%%%%%%%% layer: %zu, name : %s, output dim: %d %d %d loc: %d", i, modelLayer->getName().c_str(), width, height, depth,
                    (int)igLayer->layerLoc);
            }
            igLayer->layerLoc  = modelLayer->getLayerExecutionType();

            igLayer->outputDesc = {
                options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F, width, height,
                    DIV_4_ROUND_UP(modelLayer->getDesc().numOutputPlanes),
                modelLayer->getDesc().numOutputPlanes
            };

            prevOutputWidth = width;
            prevOutputHight = height;

            SNN_ASSERT(igLayer->outputDesc.width > 0);
            SNN_ASSERT(igLayer->outputDesc.height > 0);
            SNN_ASSERT(igLayer->outputDesc.depth > 0);
            prevLayerLoc        = igLayer->layerLoc;
            igLayer->flattenLayer = false;

            SNN_LOGD("Layer: %zu %s, layer output = %d:%d:%d, input: width = %d, height = %d, depth = %d, output: width = %d"
                     ", height = %d, depth = %d, loc = %d",
                     i, modelLayer->getName().c_str(), igLayer->outputDesc.width, igLayer->outputDesc.height, igLayer->outputDesc.depth,
                     opt.desiredInput[0].width, opt.desiredInput[0].height, opt.desiredInput[0].depth, opt.desiredOutputWidth, opt.desiredOutputHeight,
                     DIV_4_ROUND_UP(modelLayer->getDesc().numOutputPlanes), (int)igLayer->layerLoc);

        } else {
            if (i == 0) {
                SNN_RIP("CPU layer currently cannot cannot be the 1-st layer in the graph !");
            }
            if (prevLayerLoc == InferenceGraph::LayerExecutionType::GPU_FS || prevLayerLoc == InferenceGraph::LayerExecutionType::GPU_CS
                || prevLayerLoc == InferenceGraph::LayerExecutionType::GPU_VK) {
                igLayer->flattenLayer = true;
                numInputUnits       = prevOutputWidth * prevOutputHight;
                numOutputUnits      = (int) (width);
            } else {
                numInputUnits       = numOutputUnits;
                numOutputUnits      = (int) (numOutputUnits * numInputUnits);
                igLayer->flattenLayer = false;
            }

            igLayer->outputDesc = {
                options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F, width, height, depth, modelLayer->getDesc().numOutputPlanes
            };

            prevLayerLoc = igLayer->layerLoc;
        }

        igLayer->name = modelLayer->getName();
        modelFormat << "----------------------------------------------------------------\n";
    }

    head = modelLayers.at(1);

    graph.inputsDesc = options.desiredInput;

    modelFormat << "================================================================\n";
    SNN_LOGI("\n%s", modelFormat.str().c_str());
    return graph;
}

// This new topological sort supports multiple input layers in the model.
// params:
//  layers - multiple heads of the graph
// returns:
//  sorted list of GenericModelLayer objects
std::vector<std::shared_ptr<GenericModelLayer>> topologicalSort2(const std::vector<std::shared_ptr<GenericModelLayer>> &layers) {
    size_t nums = layers.size();
    std::vector<std::shared_ptr<GenericModelLayer>> sortedNodes;

    std::unordered_map<std::shared_ptr<GenericModelLayer>, size_t> inDegree;
    // calculate in degree for each node.
    for (int idx = 0; idx < nums; idx++) {
        std::shared_ptr<GenericModelLayer> node = layers[idx];
        for (auto nextNode : node->nextLayers) {
            inDegree[nextNode]++;
        }
    }
    // the nodes with in degree are ready for pop up.
    std::queue<std::shared_ptr<GenericModelLayer>> processing;
    for (auto node : layers) {
        if (inDegree[node] == 0) {
            processing.push(node);
        }
    }
    // pickup node with in degree==0, and descrease in degree of its next layers.
    int cnt = 0;
    while (!processing.empty()) {
        auto toPop = processing.front();
        processing.pop();
        sortedNodes.push_back(toPop);

        for (auto nextNode : toPop->nextLayers) {
            inDegree[nextNode]--;
            if (inDegree[nextNode] == 0) {
                processing.push(nextNode);
            }
        }
        cnt++;
    }

    if (cnt != nums) {
        SNN_LOGW("There exists a cycle in the graph !");
    }

    return sortedNodes;
}

// This new generateInferenceGraph support multiple inputs with new topological sort algorithm.
InferenceGraph snn::dp::generateInferenceGraph(std::vector<std::shared_ptr<GenericModelLayer>> &layers, const ShaderGenOptions& options) {
    // generate an topological sorted shader list
    auto modelLayers = topologicalSort2(layers);
    InferenceGraph graph;
    graph.mrtMode    = options.mrtMode;
    graph.weightMode = options.weightMode;
    std::map<std::shared_ptr<GenericModelLayer>, InferenceGraph::Layer*> s2l;
    std::map<InferenceGraph::Layer*, std::shared_ptr<GenericModelLayer>> l2s;

    uint32_t inputLayers = 0;

    // create a collection of graph layers
    for (auto modelLayer : modelLayers) {
        // Currently fixed. We can set this per layer based on layer params later.
        graph.layers.emplace_back(new InferenceGraph::Layer);
        auto igLayer = graph.layers.back().get();

        igLayer->imageTextureFunPtr = [modelLayer](ImageTextureArray& inputMat, ImageTextureArray& outputMat) {
            modelLayer->computeImageTexture(inputMat, outputMat);
        };

        igLayer->initFunPtr = [modelLayer](DeviceBackend *backend, ImageTextureArray& inputMat, ImageTextureArray& outputMat) {
            modelLayer->init(backend, inputMat, outputMat);
        };

        igLayer->runFunPtr = [modelLayer](DeviceBackend *backend, bool dumpOutputs) {
            modelLayer->run(backend, dumpOutputs);
        };

        s2l[modelLayer] = igLayer;
        l2s[igLayer] = modelLayer;

        igLayer->layerLoc = modelLayer->getLayerExecutionType();
        igLayer->name = modelLayer->getName();
        igLayer->isInputLayer = modelLayer->isInputLayer();
        if (modelLayer->isInputLayer()) {
            inputLayers++;
            igLayer->inputIndex = modelLayer->getInputIndex();
        }
        SNN_LOGD("layer name: %s, loc: %d, input: %d, # inputs: %d", igLayer->name.c_str(),
            (int)igLayer->layerLoc, igLayer->isInputLayer, inputLayers);
    }

    auto prevOutputWidth = options.desiredInput[0].width;
    auto prevOutputHight = options.desiredInput[0].height;

    int numInputUnits  = options.desiredInput[0].width * options.desiredInput[0].height;
    int numOutputUnits = numInputUnits;

    InferenceGraph::LayerExecutionType prevLayerLoc = InferenceGraph::LayerExecutionType::NOT_DEFINED;
    std::ostringstream modelFormat;
    modelFormat << "================================================================\n";
    modelFormat << "|  Layer ID  |              Name                 | Output Dims |\n";
    modelFormat << "================================================================\n";
    // loop through all layers
    std::map<InferenceGraph::Layer*, size_t> l2i;
    for (size_t i = 0; i < graph.layers.size(); ++i) {
        auto igLayer  = graph.layers[i].get();
        auto modelLayer = l2s[igLayer];
        modelLayer->setMRTMode(options.mrtMode);
        modelLayer->setWeightAccessMode(options.weightMode);

        // build a layer to array index map
        l2i[igLayer] = i;
        uint32_t inputWidth = 0, inputHeight = 0, width = 0, height = 0, depth = 0;
        if (modelLayer->prevLayers.size() > 0) {
            // build input array for each layer
            for (auto& prevModelLayer : modelLayer->prevLayers) {
                InferenceGraph::LayerRef ref;
                if (prevModelLayer->isInputLayer()) {
                    ref.isStageOutput = false;
                    ref.index         = l2i[s2l[prevModelLayer]];
                    auto inputIdx = prevModelLayer->getInputIndex();
                    auto imageInput   = InferenceGraph::IODesc {options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F,
                                                            options.desiredInput[inputIdx].width, options.desiredInput[inputIdx].height,
                                                            options.desiredInput[inputIdx].depth, 4 * options.desiredInput[inputIdx].depth};
                    SNN_LOGV("%d,%d,%d, %d", imageInput.width, imageInput.height, imageInput.depth, imageInput.channels);
                    modelLayer->addInputDim(imageInput);
                    inputWidth       = std::max(inputWidth, imageInput.width);
                    inputHeight      = std::max(inputHeight, imageInput.height);
                } else {
                    ref.isStageOutput = true;
                    ref.index         = l2i[s2l[prevModelLayer]];
                    const InferenceGraph::IODesc& imageInput   = graph.layers[ref.index].get()->outputDesc;
                    modelLayer->addInputDim(imageInput);
                    inputWidth  = std::max(inputWidth, imageInput.width);
                    inputHeight = std::max(inputHeight, imageInput.height);
                }
                igLayer->inputRefs.push_back(ref);
            }
            modelLayer->getOutputDims(width, height, depth);
            SNN_LOGD("%%%%%%%% layer: %zu, name : %s, # inputs: %zu", i, modelLayer->getName().c_str(), igLayer->inputRefs.size());
        } else {  // For all of the Input layers
            InferenceGraph::LayerRef ref;
            ref.isStageOutput = false;
            ref.index         = -1;
            auto inputIdx = modelLayer->getInputIndex();
            auto imageInput   = InferenceGraph::IODesc {options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F,
                                                        options.desiredInput[inputIdx].width, options.desiredInput[inputIdx].height,
                                                        options.desiredInput[inputIdx].depth, 4 * options.desiredInput[inputIdx].depth};
            modelLayer->addInputDim(imageInput);
            inputWidth =  imageInput.width;
            inputHeight = imageInput.height;
            modelLayer->getOutputDims(width, height, depth);
            igLayer->inputRefs.push_back(ref);
            SNN_LOGD("%%%%%%%% layer: %zu, name : %s, # inputs: %zu", i, modelLayer->getName().c_str(), igLayer->inputRefs.size());
        }

        auto name      = modelLayer->getName();
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
        SNN_LOGD("%%%%%%%% layer: %zu, name : %s, output dim: %d %d %d loc: %d", i, modelLayer->getName().c_str(), width, height, depth,
            (int)igLayer->layerLoc);

        if (igLayer->layerLoc == InferenceGraph::LayerExecutionType::GPU_FS || igLayer->layerLoc == InferenceGraph::LayerExecutionType::GPU_CS
            || igLayer->layerLoc == InferenceGraph::LayerExecutionType::GPU_VK) {
            ShaderLayer::LayerGenOptions opt;
            (ShaderGenOptions&) opt = options;
            SNN_LOGV("%%%%%%%% layer: %zu, name : %s, dims desired %d, %d, dims real %d, %d", i, modelLayer->getName().c_str(),
                opt.desiredInput[0].width, opt.desiredInput[0].height, inputWidth, inputHeight);
            opt.desiredInput[0].width  = inputWidth; // FIXME: what if the input layer is not right in front of current layer?
            opt.desiredInput[0].height = inputHeight;

            opt.desiredOutputWidth  = width;
            opt.desiredOutputHeight = height;
            opt.isFirstLayer        = (i == inputLayers);
            opt.isLastLayer         = (i == (graph.layers.size() - 1));
            opt.compute      = options.compute;
            opt.mrtMode      = options.mrtMode;
            opt.weightMode   = options.weightMode;
            opt.vulkan       = options.vulkan;
            if (modelLayer->isInputLayer()) {
                if ((options.vulkan)) {
                    modelLayer->setLayerExecutionType(snn::InferenceGraph::LayerExecutionType::GPU_VK);
                } else if ((options.compute)) {
                    modelLayer->setLayerExecutionType(snn::InferenceGraph::LayerExecutionType::GPU_CS);
                } else {
                    modelLayer->setLayerExecutionType(snn::InferenceGraph::LayerExecutionType::GPU_FS);
                }
            } else {
                modelLayer->createInferencePasses(opt);
            }
            igLayer->layerLoc  = modelLayer->getLayerExecutionType();

            igLayer->outputDesc = {
                options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F, width, height,
                    DIV_4_ROUND_UP(modelLayer->getDesc().numOutputPlanes),
                modelLayer->getDesc().numOutputPlanes
            };

            prevOutputWidth = width;
            prevOutputHight = height;

            SNN_ASSERT(igLayer->outputDesc.width > 0);
            SNN_ASSERT(igLayer->outputDesc.height > 0);
            SNN_ASSERT(igLayer->outputDesc.depth > 0);
            prevLayerLoc        = igLayer->layerLoc;
            igLayer->flattenLayer = false;

            SNN_LOGV("Layer: %zu %s, layer output: %d:%d:%d, input: width:%d, height:%d, depth:%d, output: width:%d, height:%d, depth:%d, loc: %d",
                     i, modelLayer->getName().c_str(), igLayer->outputDesc.width, igLayer->outputDesc.height, igLayer->outputDesc.depth,
                     opt.desiredInput[0].width, opt.desiredInput[0].height, opt.desiredInput[0].depth, opt.desiredOutputWidth, opt.desiredOutputHeight,
                     DIV_4_ROUND_UP(modelLayer->getDesc().numOutputPlanes), (int)igLayer->layerLoc);

        } else {
            if (i == 0) {
                SNN_RIP("CPU layer currently cannot cannot be the 1-st layer in the graph !");
            }
            if (prevLayerLoc == InferenceGraph::LayerExecutionType::GPU_FS || prevLayerLoc == InferenceGraph::LayerExecutionType::GPU_CS
                || prevLayerLoc == InferenceGraph::LayerExecutionType::GPU_VK) {
                igLayer->flattenLayer = true;
                numInputUnits       = prevOutputWidth * prevOutputHight;
                numOutputUnits      = (int) (width);
            } else {
                numInputUnits       = numOutputUnits;
                numOutputUnits      = (int) (numOutputUnits * numInputUnits);
                igLayer->flattenLayer = false;
            }

            igLayer->outputDesc = {
                options.preferrHalfPrecision ? ColorFormat::RGBA16F : ColorFormat::RGBA32F, width, height, depth, modelLayer->getDesc().numOutputPlanes
            };

            prevLayerLoc = igLayer->layerLoc;
        }

        igLayer->name = modelLayer->getName();
        modelFormat << "----------------------------------------------------------------\n";
    }

    graph.inputsDesc = options.desiredInput;

    modelFormat << "================================================================\n";
    SNN_LOGI("\n%s", modelFormat.str().c_str());
    return graph;
}
