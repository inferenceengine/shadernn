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


class BatchNormalization(Layer):

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        layerNameToInfoMap = kwargs['layerNameToInfoMap']
        weights = kwargs['weights']

        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
            layerType = layerNameToInfoMap[layername]['class_name']
            layerHelper.layersToJSON[layername] = layerinfo
            layerHelper.fuseLayersForH5(self, layername, layerType, layerNameToInfoMap, weights)


    def mergeLayersDataForH5(self, layername, inboundLayername, layerNameToInfoMap, weights):
        inboundLayerInfo = layerNameToInfoMap[inboundLayername]
        inboundLayerType = inboundLayerInfo['class_name']
        # Merge batchNorm data with inbound layer's data
        inboundLayerInfo['useBatchNormalization'] = True
        inboundLayerInfo['batchNormalization'] = weights[layername]
        inboundLayerInfo['bn_name'] = layername
        pass

    def getConvertedJSONForLayer(self):
        pass