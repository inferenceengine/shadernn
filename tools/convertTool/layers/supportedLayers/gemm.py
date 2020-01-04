# Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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
from config.formatConfig.supportedFormats import SUPPORTED_FORMATS
from config.frameworkConfig.supportedFrameworks import SUPPORTED_COMPATIBLE_FRAMEWORKS
from convertProcessor.processorConfig import processConfig
from convertProcessor.processorConfig.processConfig import COMPATIBLE_FRAMEWORK
from layers.supportedLayers import layerHelper
from layers.supportedLayers.layer import Layer


class Gemm(Layer):
    layerInfo = dict()

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        # print(layerinfo)
        self.layerInfo = layerinfo
        weights = kwargs['weights']
        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
            self.fillWeightsForONNX(layername, weights)
            self.fillInputOutputPlanesForONNX(layername, weights)

    def fillWeightsForONNX(self, layername, weights):
        self.layerInfo['weights'] = dict()
        if 'input' in self.layerInfo:
            inputList = self.layerInfo['input']
            kernelName = inputList[1]
            self.layerInfo['weights']['kernel'] = weights[kernelName][1]
            if len(inputList) > 2:
                biasName = inputList[2]
                self.layerInfo['weights']['bias'] = weights[biasName][1]
                self.layerInfo['useBias'] = "True"
            else:
                self.layerInfo['useBias'] = "False"

    def fillInputOutputPlanesForONNX(self, layername, weights):
        if 'input' in self.layerInfo:
            inputList = self.layerInfo['input']
            kernelName = inputList[1]
            self.layerInfo['inputPlanes'] = weights[kernelName][0][1]
            self.layerInfo['outputPlanes'] = weights[kernelName][0][0]

    def getConvertedJSONForLayer(self):
        if COMPATIBLE_FRAMEWORK == SUPPORTED_COMPATIBLE_FRAMEWORKS.SNN:
            if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
                self.getSNNCompatibleJSONFromONNX()

    def getSNNCompatibleJSONFromONNX(self):
        layerJSON = dict()
        layername = self.layerInfo['name']
        layerJSON['name'] = layername
        layerJSON['type'] = self.layerInfo['opType']
        layerJSON['weights'] = self.layerInfo['weights']
        layerJSON['useBias'] = self.layerInfo['useBias']
        if 'inputPlanes' in self.layerInfo:
            layerJSON['inputPlanes'] = self.layerInfo['inputPlanes']
        if 'outputPlanes' in self.layerInfo:
            layerJSON['outputPlanes'] = self.layerInfo['outputPlanes']
        for attribute in self.layerInfo['attribute']:
            # print(attribute)
            if attribute['name'] == 'alpha':
                layerJSON['alpha'] = float(attribute['f'])
            if attribute['name'] == 'beta':
                layerJSON['beta'] = float(attribute['f'])
            if attribute['name'] == 'transB':
                layerJSON['transB'] = int(attribute['i'])
        self.addInbounds(layerJSON)
        layerHelper.layersToJSON[layername] = layerJSON

    def addInbounds(self, layerJSON):
        layerJSON['inbounds'] = []
        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
            currentInbounds = self.layerInfo['inbound_nodes']
            for inboundList in currentInbounds:
                for inbound in inboundList:
                    inboundLayername = inbound[0]
                    layerJSON['inbounds'].append(inboundLayername)

            # print(layerJSON['inbounds'])
        elif processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
            if 'input' in self.layerInfo:
                inputList = self.layerInfo['input']
                # for gemm only first element
                layerJSON['inbounds'].append(inputList[0])

