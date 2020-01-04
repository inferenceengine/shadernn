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
from convertProcessor.processorConfig import processConfig
from layers.supportedLayers import layerHelper
from layers.supportedLayers.layer import Layer


class Activation(Layer):

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        layerNameToInfoMap = kwargs['layerNameToInfoMap']
        weights = kwargs['weights']

        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
            layerType = layerNameToInfoMap[layername]['class_name']
            layerHelper.layersToJSON[layername] = layerinfo
            layerHelper.fuseLayersForH5(self, layername, layerType, layerNameToInfoMap, weights)
        elif processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
            layerType = layerNameToInfoMap[layername]['opType']
            layerHelper.layersToJSON[layername] = layerinfo
            layerHelper.fuseLayersForONNX(self, layername, layerType, layerNameToInfoMap, weights)

    def mergeLayersDataForH5(self, layername, inboundLayername, layerNameToInfoMap, weights):
        inboundLayerInfo = layerNameToInfoMap[inboundLayername]

        if layerNameToInfoMap[layername]['class_name'] == 'LeakyReLU':
            inboundLayerInfo['config']['activation'] = 'leakyRelu'
            if 'alpha' in layerNameToInfoMap[layername]['config']:
                inboundLayerInfo['config']['alpha'] = layerNameToInfoMap[layername]['config']['alpha']
            else :
                inboundLayerInfo['config']['alpha'] = 0.3
        elif layerNameToInfoMap[layername]['class_name'] == 'ReLU':
            if 'max_value' in layerNameToInfoMap[layername]['config'] \
                    and layerNameToInfoMap[layername]['config']['max_value'] == 6.0:
                inboundLayerInfo['config']['activation'] = 'relu6'
            else:
                inboundLayerInfo['config']['activation'] = 'relu'
        elif 'activation' in layerNameToInfoMap[layername]['config']:
            inboundLayerInfo['config']['activation'] = layerNameToInfoMap[layername]['config']['activation']

        inboundLayerInfo['act_name'] = layername

    def mergeLayersDataForONNX(self, layername, inputLayername, layerNameToInfoMap, weights):
        inputLayerInfo = layerNameToInfoMap[inputLayername]

        if layerNameToInfoMap[layername]['opType'] == 'Clip':
            inputLayerInfo['activation'] = 'relu6'
        elif layerNameToInfoMap[layername]['opType'] == 'LeakyRelu':
            inputLayerInfo['activation'] = 'leakyRelu'
            if 'alpha' in layerNameToInfoMap[layername]:
                inputLayerInfo['alpha'] = layerNameToInfoMap[layername]['alpha']
            else:
                inputLayerInfo['alpha'] = 0.01
        else:
            inputLayerInfo['activation'] = layerNameToInfoMap[layername]['opType']

        inputLayerInfo['act_name'] = layername


    def getConvertedJSONForLayer(self):
        pass