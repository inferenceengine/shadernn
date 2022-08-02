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
from config.formatConfig.supportedFormats import SUPPORTED_FORMATS
from config.frameworkConfig.supportedFrameworks import SUPPORTED_COMPATIBLE_FRAMEWORKS
from convertProcessor.processorConfig import processConfig
from convertProcessor.processorConfig.processConfig import COMPATIBLE_FRAMEWORK
from layers.supportedLayers import layerHelper
from layers.supportedLayers.layer import Layer


class Conv2D(Layer):
    layerInfo = dict()

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        self.layerInfo = layerinfo
        layerNameToInfoMap = kwargs['layerNameToInfoMap']
        weights = kwargs['weights']
        # print(layerinfo)
        layerinfo['useBatchNormalization'] = "False"
        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
            self.fillWeightsForH5(layername, weights)
        elif processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
            self.fillWeightsForONNX(layername, weights)
            self.fillInputOutputPlanesForONNX(layername, weights)

    def fillWeightsForH5(self, layername, weights):
        self.layerInfo['weights'] = dict()
        self.layerInfo['weights']['kernel'] = weights[layername]['kernel']
        if 'bias' in weights[layername]:
            self.layerInfo['weights']['bias'] = weights[layername]['bias']


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

    def fillInputOutputPlanesForONNX(self,layername, weights):
        if 'input' in self.layerInfo:
            inputList = self.layerInfo['input']
            kernelName = inputList[1]
            self.layerInfo['inputPlanes'] = weights[kernelName][0][1]
            self.layerInfo['outputPlanes'] = weights[kernelName][0][0]

    def getConvertedJSONForLayer(self):
        if COMPATIBLE_FRAMEWORK == SUPPORTED_COMPATIBLE_FRAMEWORKS.SNN:
            if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
                self.getSNNCompatibleJSONFromH5()
            elif processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
                self.getSNNCompatibleJSONFromONNX()

    def getSNNCompatibleJSONFromH5(self):
        layerJSON = dict()
        layername = self.layerInfo['config']['name']
        layerJSON['name'] = layername
        layerJSON['type'] = self.layerInfo['class_name']
        layerJSON['kernel_size'] = self.layerInfo['config']['kernel_size'][0]
        layerJSON['padding'] = self.layerInfo['config']['padding']
        layerJSON['strides'] = self.layerInfo['config']['strides'][0]
        layerJSON['outputPlanes'] = self.layerInfo['config']['filters']
        layerJSON['useBias'] = str(self.layerInfo['config']['use_bias'])
        layerJSON['weights'] = self.layerInfo['weights']
        layerJSON['useBatchNormalization'] = str(self.layerInfo['useBatchNormalization'])
        if 'batchNormalization' in self.layerInfo:
            layerJSON['batchNormalization'] = self.layerInfo['batchNormalization']
        if 'bn_name' in self.layerInfo:
            layerJSON['bn_name'] = self.layerInfo['bn_name']
        if 'act_name' in self.layerInfo:
            layerJSON['act_name'] = self.layerInfo['act_name']
        if 'activation' in self.layerInfo['config']:
            layerJSON['activation'] = self.layerInfo['config']['activation']
        if 'alpha' in self.layerInfo['config']:
            layerJSON['alpha'] = self.layerInfo['config']['alpha']

        self.addInbounds(layerJSON)
        layerHelper.layersToJSON[layername] = layerJSON
        # print(layerJSON)

    def getSNNCompatibleJSONFromONNX(self):
        layerJSON = dict()
        layername = self.layerInfo['name']
        layerJSON['name'] = layername
        layerJSON['type'] = 'Conv2D'
        for attribute in self.layerInfo['attribute']:
            # print(attribute)
            if attribute['name'] == 'kernel_shape':
                layerJSON['kernel_size'] = int(attribute['ints'][0])
            if attribute['name'] == 'pads':
                layerJSON['padding'] = attribute['ints'][0]
            if attribute['name'] == 'strides':
                layerJSON['strides'] = int(attribute['ints'][0])
            if attribute['name'] == 'group':
                layerJSON['group'] = int(attribute['i'])
        layerJSON['weights'] = self.layerInfo['weights']
        layerJSON['useBias'] = str(self.layerInfo['useBias'])
        layerJSON['useBatchNormalization'] = str(self.layerInfo['useBatchNormalization'])
        if 'batchNormalization' in self.layerInfo:
            layerJSON['batchNormalization'] = self.layerInfo['batchNormalization']
        if 'bn_name' in self.layerInfo:
            layerJSON['bn_name'] = self.layerInfo['bn_name']
        if 'act_name' in self.layerInfo:
            layerJSON['act_name'] = self.layerInfo['act_name']
        if 'activation' in self.layerInfo:
            layerJSON['activation'] = self.layerInfo['activation']
        if 'alpha' in self.layerInfo:
            layerJSON['alpha'] = self.layerInfo['alpha']
        if 'outputPlanes' in self.layerInfo:
            layerJSON['outputPlanes'] = self.layerInfo['outputPlanes']

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
                # for conv only first element
                layerJSON['inbounds'].append(inputList[0])
