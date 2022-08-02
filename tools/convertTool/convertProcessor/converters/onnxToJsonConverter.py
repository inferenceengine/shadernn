# Copyright (C) 2020 - 2022 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import base64
import json

import onnx
from convertProcessor.converters import converter
from convertProcessor.processorConfig.processConfig import getModelFile, isDecoupleWeights
from google.protobuf.json_format import MessageToJson
from layers.supportedLayers import layerFactory, layerHelper
from onnx import numpy_helper, shape_inference
from utils.toolUtils import getDecoupledWeightsFileName, getFileNameWithoutExtension, getFileName, \
    getCompressedWeightBin


class ONNXToJsonConverter(converter.Converter):
    layerNameToInfoMap = {}  # Truth

    def __init__(self):
        print("Initializing ONNXToJsonConverter")

    def convert(self, **kwargs):
        print("Converting using ONNXToJsonConverter")
        model = self.getModelFromFile()
        shapes = self.prepareShapes(model)
        weights = self.prepareWeights(model)
        self.preProcessLayers(model, weights, shapes)
        layernameToObjMap = dict()
        # self.printlayers()

        for layername in list(self.layerNameToInfoMap.keys()):
            layerinfo = self.layerNameToInfoMap[layername]
            # print("\n")
            # print(layerinfo)
            if 'opType' in layerinfo:
                layerType = layerinfo['opType']
            else:
                layerType = 'InputLayer'
            layer = layerFactory.getLayer(layerType)
            layernameToObjMap[layername] = layer
            # print(layerType)
            layer.handleLayer(layername=layername, layerinfo=self.layerNameToInfoMap[layername],
                              layerNameToInfoMap=self.layerNameToInfoMap, weights=weights, shapes=shapes)

        layernameToIndexMap = dict()
        index = 0
        for layername in list(self.layerNameToInfoMap.keys()):
            # print(layername)
            layernameToObjMap[layername].getConvertedJSONForLayer()
            layernameToIndexMap[layername] = index
            # print(layername + " "+ str(index))
            index = index + 1

        self.fillInputIds(layernameToIndexMap)
        self.fillInputOutputPlanes()
        # print(layerHelper.layersToJSON)
        if isDecoupleWeights():
            self.seperateWeightsFromLayerJSON()
            self.dump2json(getFileNameWithoutExtension(getFileName(getModelFile())) + "_layers" + ".json")
        else:
            self.dump2json(getFileNameWithoutExtension(getFileName(getModelFile())) + ".json")
        print("Conversion complete from onnx to JSON.")

    def printlayers(self):
        for layername in list(self.layerNameToInfoMap.keys()):
            layerinfo = self.layerNameToInfoMap[layername]
            print("\n" + layername)
            print(layerinfo)

    def seperateWeightsFromLayerJSON(self):
        weights_name = getDecoupledWeightsFileName()
        weight_bin_file = open(weights_name, "wb")
        # order needs to be maintained
        for layername in list(self.layerNameToInfoMap.keys()):
            layerJsonObj = layerHelper.layersToJSON[layername]
            if 'weights' in layerJsonObj:
                if 'kernel' in layerJsonObj['weights']:
                    weight_bin_file.write(getCompressedWeightBin(layerJsonObj['weights']['kernel']))
                if 'bias' in layerJsonObj['weights']:
                    weight_bin_file.write(getCompressedWeightBin(layerJsonObj['weights']['bias']))
                del layerHelper.layersToJSON[layername]['weights']
            if 'batchNormalization' in layerJsonObj:
                if 'gamma' in layerJsonObj['batchNormalization']:
                    weight_bin_file.write(getCompressedWeightBin(layerJsonObj['batchNormalization']['gamma']))
                if 'beta' in layerJsonObj['batchNormalization']:
                    weight_bin_file.write(getCompressedWeightBin(layerJsonObj['batchNormalization']['beta']))
                if 'moving_mean' in layerJsonObj['batchNormalization']:
                    weight_bin_file.write(getCompressedWeightBin(layerJsonObj['batchNormalization']['moving_mean']))
                if 'moving_variance' in layerJsonObj['batchNormalization']:
                    weight_bin_file.write(getCompressedWeightBin(layerJsonObj['batchNormalization']['moving_variance']))
                del layerHelper.layersToJSON[layername]['batchNormalization']
        weight_bin_file.close()

    def dump2json(self, path):
        jsonListObj = dict()
        jsonListObj['numLayers'] = dict()
        jsonListObj['numLayers']['count'] = len(self.layerNameToInfoMap.keys())
        if isDecoupleWeights():
            jsonListObj['numLayers']['bin_file_name'] = getDecoupledWeightsFileName()
        index = 0
        for layername in list(self.layerNameToInfoMap.keys()):
            jsonListObj["Layer_" + str(index)] = layerHelper.layersToJSON[layername]
            index = index + 1
        with open(path, 'w+') as file:
            json.dump(jsonListObj, file, indent=4)
            file.close()

    def getModelFromFile(self):
        modelfile = getModelFile()
        model = onnx.load(modelfile)
        return model

    def prepareShapes(self, model):
        shapes = dict()
        inferred_model = shape_inference.infer_shapes(model)
        for info in inferred_model.graph.value_info:
            shapeInfo = json.loads(MessageToJson(info))
            # print(shapeInfo['name'])
            shapes[info.name] = shapeInfo
        return shapes

    def preProcessLayers(self, model, weights, shapes):
        self.prepareInput(model)
        layers = model.graph.node
        self.mergeLayers(layers, 'Sigmoid', 'Mul', 'Swish')

        for layer in layers:
            self.layerNameToInfoMap[layer.name] = json.loads(MessageToJson(layer))

        self.checkAndUpdateLayersOpType(model, weights)
        self.prepareInputOutputsShapes(shapes)

    def checkAndUpdateLayersOpType(self, model, weights):
        self.checkForConv()
        self.checkForResize(weights)
        self.checkForGlobalAvgPool(weights)

    def checkForConv(self):
        for layername in self.layerNameToInfoMap:
            if 'opType' in self.layerNameToInfoMap[layername]:
                if self.layerNameToInfoMap[layername]['opType'] == 'Conv':
                    if 'attribute' in self.layerNameToInfoMap[layername]:
                        attributList = self.layerNameToInfoMap[layername]['attribute']
                        # print("\nBefore" + str(attributList))
                        for attribute in attributList:
                            if attribute['name'] == 'group':
                                if int(attribute['i']) > 1:
                                    self.layerNameToInfoMap[layername]['opType'] = 'DepthwiseConv2D'
                                    self.layerNameToInfoMap[layername]['inputPlanes'] = int(attribute['i'])
                                    attribute['i'] = '1'
                        # print("\nAfter" + str(attributList))

    def checkForResize(self, weights):
        for layername in self.layerNameToInfoMap:
            if 'opType' in self.layerNameToInfoMap[layername]:
                if self.layerNameToInfoMap[layername]['opType'] == 'Resize':
                    resizename = self.layerNameToInfoMap[layername]['input'][2]
                    resizeValList = weights[resizename][1]
                    maxVal = 1.0
                    needToUpSample = True
                    for resizeval in resizeValList:
                        if (resizeval < 1.0) or (maxVal != 1.0 and resizeval != maxVal):
                            needToUpSample = False
                            break
                        if resizeval > 1.0 and maxVal == 1.0:
                            maxVal = resizeval
                    if needToUpSample:
                        self.layerNameToInfoMap[layername]['opType'] = 'Upsample'
                        self.layerNameToInfoMap[layername]['scaleFactor'] = maxVal
                        for attribute in self.layerNameToInfoMap[layername]['attribute']:
                            if attribute['name'] == 'mode':
                                self.layerNameToInfoMap[layername]['interpolation'] = base64.b64decode(
                                    str(attribute['s'])).decode('utf-8')

    def checkForGlobalAvgPool(self, weights):
        for layername in self.layerNameToInfoMap:
            if 'opType' in self.layerNameToInfoMap[layername]:
                if self.layerNameToInfoMap[layername]['opType'] == 'GlobalAveragePool':
                    # print(weights[layername])
                    self.layerNameToInfoMap[layername]['opType'] = 'AveragePool'

    def prepareInputOutputsShapes(self, shapes):
        outputToLayerNameMap = dict()
        for layername in self.layerNameToInfoMap.keys():
            if 'output' in self.layerNameToInfoMap[layername]:
                outputList = self.layerNameToInfoMap[layername]['output']
                outputToLayerNameMap[outputList[0]] = layername

        #print(outputToLayerNameMap)

        for layername in self.layerNameToInfoMap.keys():
            if 'input' in self.layerNameToInfoMap[layername]:
                inputList = self.layerNameToInfoMap[layername]['input']
                idx = 0
                for input in inputList:
                    if input in outputToLayerNameMap.keys():
                        inputList[idx] = outputToLayerNameMap[input]
                    idx = idx + 1

        for shapeName in list(shapes.keys()):
            shapeInfo = shapes[shapeName]
            if shapeName in outputToLayerNameMap.keys():
                shapes[outputToLayerNameMap[shapeName]] = shapeInfo
            del shapes[shapeName]

        # print(shapes.keys())

    def mergeLayers(self, layers, mergeLayerOpType, mergeWithLayerOpType, newLayerOpType):
        idx = 0
        while idx < len(layers):
            layer = layers[idx]
            # Merge Sigmoid & Mul to Swish
            if layer.op_type == mergeLayerOpType:
                if (idx + 1) < len(layers) and layers[idx + 1].op_type == mergeWithLayerOpType:
                    layer.op_type = newLayerOpType
                    # Need to update output after merged
                    layers[idx].output[:] = layers[idx + 1].output[:]
                    del layers[idx + 1]
            idx = idx + 1

    def prepareInput(self, model):
        graphInput = model.graph.input
        inputLayer = graphInput[0]
        self.layerNameToInfoMap[inputLayer.name] = json.loads(MessageToJson(inputLayer))

    def prepareWeights(self, model):
        weights = model.graph.initializer
        weightDict = dict()
        for weight in weights:
            '''weight_json = {'name': weight.name,
                           'data_type': weight.data_type,
                           'dims': weight.dims[0:],
                           'raw_data': str(weight.raw_data)}'''
            weightDict[weight.name] = [weight.dims,
                                       (numpy_helper.to_array(weight)).flatten().tolist()]
            # model_json.append(weight_json)
        return weightDict

    def fillInputIds(self, layernameToIndexMap):
        for layername in list(self.layerNameToInfoMap.keys()):
            layerJsonObj = layerHelper.layersToJSON[layername]
            inputIds = []
            layerInbounds = layerJsonObj['inbounds']
            for layerInbound in layerInbounds:
                inputIds.append(layernameToIndexMap[layerInbound])
            layerHelper.layersToJSON[layername]['numInputs'] = len(layerInbounds)
            layerHelper.layersToJSON[layername]['inputId'] = inputIds

    def fillInputOutputPlanes(self):
        for layername in list(self.layerNameToInfoMap.keys()):
            layerJsonObj = layerHelper.layersToJSON[layername]
            layerInbounds = layerJsonObj['inbounds']
            # print(layerInbounds)
            inputPlanes = 0
            if len(layerInbounds) != 0:
                # print(layerInbounds)
                if layerJsonObj['type'] == "Add" or layerJsonObj['type'] == "Mul":
                    inputPlanes = inputPlanes + layerHelper.layersToJSON[layerInbounds[0]]['outputPlanes']
                else:
                    for layerInbound in layerInbounds:
                        inputPlanes = inputPlanes + layerHelper.layersToJSON[layerInbound]['outputPlanes']
            layerHelper.layersToJSON[layername]['inputPlanes'] = inputPlanes
            if 'outputPlanes' not in layerJsonObj:
                layerHelper.layersToJSON[layername]['outputPlanes'] = inputPlanes
            if layerJsonObj['type'] == "Calculate":
                layerHelper.layersToJSON['outputPlanes'] = 3
