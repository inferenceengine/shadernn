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


class AveragePooling2D(Layer):
    layerInfo = dict()

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        layerNameToInfoMap = kwargs['layerNameToInfoMap']
        weights = kwargs['weights']

        self.layerInfo = layerinfo

        if 'shapes' in kwargs:
            shapes = kwargs['shapes']
            self.fillPoolSize(shapes)

        # print(layerinfo)

    def fillPoolSize(self, shapes):
        inputListForLayer = self.layerInfo['input']
        currentInputLayer = inputListForLayer[0]
        shapeInfoOfInput = shapes[currentInputLayer]

        if 'type' in shapeInfoOfInput:
            if 'tensorType' in shapeInfoOfInput['type']:
                tensorType = shapeInfoOfInput['type']['tensorType']
                if 'shape' in tensorType:
                    shape = tensorType['shape']
                    dims = shape['dim']
                    if len(dims) > 3:
                        poolsize = int(dims[2]['dimValue'])
                        self.layerInfo['pool_size'] = [poolsize, poolsize]
                        self.layerInfo['strides'] = [1, 1]
                        self.layerInfo['padding'] = "valid"

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
        if 'pool_size' not in self.layerInfo:
            if 'pool_size' in self.layerInfo['config']:
                layerJSON['pool_size'] = self.layerInfo['config']['pool_size']
            if 'padding' in self.layerInfo['config']:
                layerJSON['padding'] = self.layerInfo['config']['padding']
            if 'strides' in self.layerInfo['config']:
                layerJSON['strides'] = self.layerInfo['config']['strides']
            if 'data_format' in self.layerInfo['config']:
                layerJSON['data_format'] = self.layerInfo['config']['data_format']
        else:
            layerJSON['pool_size'] = self.layerInfo['pool_size']
            layerJSON['strides'] = self.layerInfo['strides']
            layerJSON['padding'] = self.layerInfo['padding']
        self.addInbounds(layerJSON)
        layerHelper.layersToJSON[layername] = layerJSON
        # print(layerJSON)

    def getSNNCompatibleJSONFromONNX(self):
        layerJSON = dict()
        layername = self.layerInfo['name']
        layerJSON['name'] = layername
        layerJSON['type'] = 'AveragePooling2D'
        if 'pool_size' not in self.layerInfo:
            if 'attribute' in self.layerInfo:
                for attribute in self.layerInfo['attribute']:
                    # print(attribute)
                    if attribute['name'] == 'kernel_shape':
                        layerJSON['pool_size'] = int(attribute['ints'][0])
                    if attribute['name'] == 'strides':
                        layerJSON['strides'] = int(attribute['ints'][0])
        else:
            layerJSON['pool_size'] = self.layerInfo['pool_size']
            layerJSON['strides'] = self.layerInfo['strides']
            layerJSON['padding'] = self.layerInfo['padding']

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
                layerJSON['inbounds'] = inputList
