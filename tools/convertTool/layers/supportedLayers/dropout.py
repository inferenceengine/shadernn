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
from convertProcessor.processorConfig import processConfig
from layers.supportedLayers.layer import Layer


class Dropout(Layer):

    def handleLayer(self, **kwargs):
        layername = kwargs['layername']
        layerinfo = kwargs['layerinfo']
        layerNameToInfoMap = kwargs['layerNameToInfoMap']

        if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
            self.handleDropoutForH5(layername, layerinfo, layerNameToInfoMap)

        # As it is dropout layer , No need to add in layersToJSON map
        # layerHelper.layersToJSON[layername] = layerinfo

    def handleDropoutForH5(self, layername, layerinfo, layerNameToInfoMap):
        dropoutInbounds = layerinfo['inbound_nodes']

        # print("\n" + layername + " Inbounds" + str(dropoutInbounds))
        for layer in list(layerNameToInfoMap.keys()):
            layerInbounds = layerNameToInfoMap[layer]['inbound_nodes']
            for layerInboundList in layerInbounds:
                for layerInbound in layerInboundList:
                    if layerInbound[0] == layername:
                        self.updateInbound(layer, layerInbound, layerInboundList, dropoutInbounds)
        del layerNameToInfoMap[layername]

    def updateInbound(self, layer, layerInbound, layerInboundList, dropoutInbounds):
        # print("Removing: " + str(layerInbound) + " for " + layer)
        idx = layerInboundList.index(layerInbound)
        layerInboundList.remove(layerInbound)
        insertAtIdx = idx
        for dropoutInboundList in dropoutInbounds:
            for dropoutInbound in dropoutInboundList:
                # print("Inserting: " + str(dropoutInbound))
                layerInboundList.insert(insertAtIdx, dropoutInbound)
                insertAtIdx = insertAtIdx + 1
                # print(layerInboundList)

    def getConvertedJSONForLayer(self):
        pass
