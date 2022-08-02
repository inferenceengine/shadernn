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
import json

import numpy as np
import tensorflow as tf
from convertProcessor.converters import converter
from convertProcessor.processorConfig.processConfig import getModelFile, getCustomLayers, isDecoupleWeights
from layers.supportedLayers import layerFactory, layerHelper
from utils.toolUtils import getFileName, getFileNameWithoutExtension, getDecoupledWeightsFileName, \
    getCompressedWeightBin


class H5ToJsonConverter(converter.Converter):
    CLASSNAME_TAG = 'class_name'
    LAYERS_TAG = 'layers'
    CONFIG_TAG = 'config'
    NAME_TAG = 'name'
    layerNameToInfoMap = {}  # Truth

    def __init__(self):
        print("Initializing H5ToJsonConverter")

    def convert(self, **kwargs):
        print("Converting using H5ToJsonConverter")
        model = self.getModelFromFile()
        # print(model.summary())
        modelConfiguration = model.get_config()
        # print(str(modelConfiguration))
        self.preprocessLayers(modelConfiguration)
        weights = self.prepareWeights(model)

        self.checkAndFillInbounds()
        # print("\nBefore:" + str(self.layerNameToInfoMap))
        layernameToObjMap = dict()
        for layername in list(self.layerNameToInfoMap.keys()):
            layerType = self.layerNameToInfoMap[layername]['class_name']
            layer = layerFactory.getLayer(layerType)
            layernameToObjMap[layername] = layer
            layer.handleLayer(layername=layername, layerinfo=self.layerNameToInfoMap[layername],
                              layerNameToInfoMap=self.layerNameToInfoMap, weights=weights)

        layernameToIndexMap = dict()
        index = 0
        for layername in list(self.layerNameToInfoMap.keys()):
            # print(layername)
            layernameToObjMap[layername].getConvertedJSONForLayer()
            layernameToIndexMap[layername] = index
            index = index + 1

        self.fillInputIds(layernameToIndexMap)
        self.fillInputOutputPlanes()

        if isDecoupleWeights():
            self.seperateWeightsFromLayerJSON()
            self.dump2json(getFileNameWithoutExtension(getFileName(getModelFile())) + "_layers" + ".json")
        else:
            self.dump2json(getFileNameWithoutExtension(getFileName(getModelFile())) + ".json")
        print("Conversion complete from h5 to JSON.")
        pass

    def orderWeightsinBinFormat(self, weights, isDeconv=False, isDense=False):
        if not isDense:
            weights = np.swapaxes(weights, 0, 2)
            weights = np.swapaxes(weights, 1, 3)
            weights = np.swapaxes(weights, 0, 1)
            if isDeconv:
                weights = np.swapaxes(weights, 0, 1)
            weights = np.resize(weights,
                                (weights.shape[0] * weights.shape[1] * weights.shape[2] * weights.shape[3])).tolist()
        else:
            if len(weights.shape) == 1:
                weights = np.expand_dims(weights, 1)
            weights = np.swapaxes(weights, 0, 1)
            weights = np.resize(weights, (weights.shape[0] * weights.shape[1])).tolist()
            # print(len(weights))
        return weights

    def getLayerType(self, layerName):
        # print(self.layerNameToInfoMap[layerName]['class_name'])
        return self.layerNameToInfoMap[layerName]['class_name']

    def getModelFromFile(self):
        modelfile = getModelFile()
        custom_objects = getCustomLayers()
        model = tf.keras.models.load_model(modelfile, custom_objects=custom_objects)
        return model

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

    def prepareWeights(self, model):
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weightList = dict()
        for name, weight in zip(names, model.get_weights()):
            # print(name)
            result = name.split(':')
            res = result[0].split('/')
            layername = res[0]
            variable = res[1]
            if layername not in weightList.keys():
                weightList[layername] = dict()
            if variable == 'kernel':
                weightList[layername][variable] = self.orderWeightsinBinFormat(weight,
                                                                               self.layerNameToInfoMap[layername][
                                                                                   'class_name'] == 'Conv2DTranspose',
                                                                               self.layerNameToInfoMap[layername][
                                                                                   'class_name'] == 'Dense')
            else:
                weightList[layername][variable] = weight.flatten().tolist()
            # print(layername+ " " + str(weightList[layername].keys()))
        return weightList

    def fillInputIds(self, layernameToIndexMap):
        for layername in list(self.layerNameToInfoMap.keys()):
            layerJsonObj = layerHelper.layersToJSON[layername]
            inputIds = []
            # print(layername)
            layerInbounds = layerJsonObj['inbounds']
            # print(layerInbounds)
            for layerInbound in layerInbounds:
                inputIds.append(layernameToIndexMap[layerInbound])
            layerHelper.layersToJSON[layername]['numInputs'] = len(layerInbounds)
            layerHelper.layersToJSON[layername]['inputId'] = inputIds

    def fillInputOutputPlanes(self):
        for layername in list(self.layerNameToInfoMap.keys()):
            layerJsonObj = layerHelper.layersToJSON[layername]
            layerInbounds = layerJsonObj['inbounds']
            inputPlanes = 0
            if len(layerInbounds) != 0:
                # print(layerInbounds)
                if layerJsonObj['type'] == "Add" or layerJsonObj['type'] == "Multiply":
                    inputPlanes = inputPlanes + layerHelper.layersToJSON[layerInbounds[0]]['outputPlanes']
                else:
                    for layerInbound in layerInbounds:
                        inputPlanes = inputPlanes + layerHelper.layersToJSON[layerInbound]['outputPlanes']
            layerHelper.layersToJSON[layername]['inputPlanes'] = inputPlanes
            if 'outputPlanes' not in layerJsonObj:
                layerHelper.layersToJSON[layername]['outputPlanes'] = inputPlanes
            if layerJsonObj['type'] == "Calculate":
                layerHelper.layersToJSON['outputPlanes'] = 3

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

    def preprocessLayers(self, modelConfig):
        layers = modelConfig['layers']
        updatedLayers = []
        updatePendingInboundMap = dict()
        for layer in layers:
            self.processLayer(layer, layer['class_name'], updatedLayers, updatePendingInboundMap)

        self.checkAndUpdatePendingInbounds(updatePendingInboundMap, updatedLayers)
        for layer in updatedLayers:
            layerName = layer['config']['name']
            self.layerNameToInfoMap[layerName] = layer

        self.checkAndUpdateLayerTypes()

    def checkAndUpdateLayerTypes(self):
        self.checkForGlobalAveragePool()

    def checkForGlobalAveragePool(self):
        for layername in self.layerNameToInfoMap:
            if 'class_name' in self.layerNameToInfoMap[layername]:
                layerType = self.layerNameToInfoMap[layername]['class_name']
                if layerType == 'GlobalAveragePooling2D':
                    self.layerNameToInfoMap[layername]['class_name'] = 'AveragePooling2D'


    def processLayer(self, layer, layerType, updatedLayers, updatePendingInboundMap):
        if(layerType == 'Sequential'):
            innerLayers = layer['config']['layers']
            lastLayer = None
            for innerLayer in innerLayers:
                lastLayer = self.processLayer(innerLayer, innerLayer['class_name'], updatedLayers, updatePendingInboundMap)
            # print(lastLayer['config']['name'])
            updatePendingInboundMap[layer['config']['name']] = lastLayer['config']['name']
            return innerLayers[-1]
        else:
            updatedLayers.append(layer)
            return layer
        pass

    def checkAndUpdatePendingInbounds(self, updatePendingInboundMap, updatedLayers):
        if len(updatePendingInboundMap) == 0:
            return
        for layer in updatedLayers:
            if 'inbound_nodes' in layer:
                # print(layer['inbound_nodes'])
                for inboundNodeList in layer['inbound_nodes']:
                    for inbound in inboundNodeList:
                        # print(inbound[0])
                        if inbound[0] in updatePendingInboundMap.keys():
                            oldLayerName = inbound[0]
                            newLayerName = updatePendingInboundMap[oldLayerName]
                            inbound[0] = newLayerName

    def checkAndFillInbounds(self):
        # fill previous layer as inbound if inbound_nodes not found
        layernames = list(self.layerNameToInfoMap.keys())
        for layername in layernames:
            idx = layernames.index(layername)
            layerType = self.layerNameToInfoMap[layername]['class_name']
            if 'inbound_nodes' not in self.layerNameToInfoMap[layername]:
                if layernames.index(layername) != 0:
                    previousLayerName = layernames[idx - 1]
                    self.layerNameToInfoMap[layername]['inbound_nodes'] = [[[previousLayerName, 0, 0, {}]]]
                else:
                    self.layerNameToInfoMap[layername]['inbound_nodes'] = []
