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


class Dense(Layer):
    layerInfo = dict()

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        self.layerInfo = layerinfo
        layerNameToInfoMap = kwargs['layerNameToInfoMap']
        weights = kwargs['weights']

        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
            self.fillWeightsForH5(layername, weights)


    def fillWeightsForH5(self, layername, weights):
        self.layerInfo['weights'] = dict()
        self.layerInfo['weights']['kernel'] = weights[layername]['kernel']
        if 'bias' in weights[layername]:
            self.layerInfo['weights']['bias'] = weights[layername]['bias']

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
        layerJSON['weights'] = self.layerInfo['weights']
        layerJSON['activation'] = self.layerInfo['config']['activation']
        layerJSON['useBias'] = str(self.layerInfo['config']['use_bias'])
        layerJSON['units'] = self.layerInfo['config']['units']
        self.addInbounds(layerJSON)
        layerHelper.layersToJSON[layername] = layerJSON
        # print(layerJSON)

    def addInbounds(self,layerJSON):
        layerJSON['inbounds'] = []
        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
            currentInbounds = self.layerInfo['inbound_nodes']
            for inboundList in currentInbounds:
                for inbound in inboundList:
                    inboundLayername = inbound[0]
                    layerJSON['inbounds'].append(inboundLayername)

            # print(layerJSON['inbounds'])
