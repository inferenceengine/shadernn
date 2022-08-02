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

from config.formatConfig.supportedFormats import SUPPORTED_FORMATS
from config.frameworkConfig.supportedFrameworks import SUPPORTED_COMPATIBLE_FRAMEWORKS
from convertProcessor.processorConfig import processConfig
from convertProcessor.processorConfig.processConfig import COMPATIBLE_FRAMEWORK
from layers.supportedLayers import layerHelper
from layers.supportedLayers.layer import Layer


class Pad(Layer):
    layerInfo = dict()

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        weights = kwargs['weights']
        # print(layerinfo)
        self.layerInfo = layerinfo

        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
            self.checkAndFillPads(weights)

    def checkAndFillPads(self, weights):
        if 'input' in self.layerInfo:
            inputList = self.layerInfo['input']
            if len(inputList) > 1:
                padsName = inputList[1]
                self.layerInfo['pads'] = weights[padsName][1]
            if len(inputList) > 2:
                constantsname = inputList[2]
                self.layerInfo['constant_value'] = weights[constantsname][1]
            else:
                self.layerInfo['constant_value'] = 0  # default is 0

    def getConvertedJSONForLayer(self):
        if COMPATIBLE_FRAMEWORK == SUPPORTED_COMPATIBLE_FRAMEWORKS.SNN:
            if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
                self.getSNNCompatibleJSONFromONNX()

    def getSNNCompatibleJSONFromONNX(self):
        layerJSON = dict()
        layername = self.layerInfo['name']
        layerJSON['name'] = layername
        layerJSON['type'] = self.layerInfo['opType']

        for attribute in self.layerInfo['attribute']:
            # print(attribute)
            if attribute['name'] == 'mode':
                # This will have base64 encoded string, convert to string before using in modelparser
                layerJSON['mode'] = base64.b64decode(str(attribute['s'])).decode('utf-8')
            if attribute['name'] == 'pads':
                if 'pads' not in self.layerInfo:
                    layerJSON['pads'] = attribute['ints']
                else:
                    layerJSON['pads'] = self.layerInfo['pads']

        self.addInbounds(layerJSON)
        layerHelper.layersToJSON[layername] = layerJSON

    def addInbounds(self, layerJSON):
        layerJSON['inbounds'] = []
        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
            if 'input' in self.layerInfo:
                inputList = self.layerInfo['input']
                # for pad there may be other weight elements in inputlist
                layerJSON['inbounds'].append(inputList[0])

        # print(layerJSON['inbounds'])
