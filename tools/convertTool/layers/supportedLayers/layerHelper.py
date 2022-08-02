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
from utils.toolUtils import isFusibleWith

layersToJSON = {}


def fuseLayersForH5(layerClassObj, layername, layerType, layerNameToInfoMap, weights):
    currentInbounds = layerNameToInfoMap[layername]['inbound_nodes']
    for inboundList in currentInbounds:
        for inbound in inboundList:
            inboundLayername = inbound[0]
            inboundLayerType = layerNameToInfoMap[inboundLayername]['class_name']
            # print(inboundLayername + " " + inboundLayerType)
            if isFusibleWith(layerType, inboundLayerType):
                layerClassObj.mergeLayersDataForH5(layername, inboundLayername, layerNameToInfoMap, weights)
                updateInboundsForH5(layername, inboundLayername, currentInbounds, layerNameToInfoMap)
                del layerNameToInfoMap[layername]
                del layersToJSON[layername]


def updateInboundsForH5(layername, inboundLayername, currentInbounds, layerNameToInfoMap):
    # print(layername + " " + inboundLayername)
    for layer in list(layerNameToInfoMap.keys()):
        layerInbounds = layerNameToInfoMap[layer]['inbound_nodes']
        for layerInboundList in layerInbounds:
            for layerInbound in layerInboundList:
                if layerInbound[0] == layername:
                    addInboundsForH5(layer, layerInbound, layerInboundList, currentInbounds,
                                     inboundLayername)


def addInboundsForH5(layer, layerInbound, layerInboundList, currentInbounds, fusingWith):
    # print("Removing: " + str(layerInbound) + " for " + layer)
    idx = layerInboundList.index(layerInbound)
    layerInboundList.remove(layerInbound)
    insertAtIdx = idx
    for currentInboundList in currentInbounds:
        for currentInbound in currentInboundList:
            if currentInbound[0] == fusingWith:
                # print("Appending: " + str(currentInbound) + " for " + layer)
                layerInboundList.insert(insertAtIdx, currentInbound)
                insertAtIdx = insertAtIdx + 1


def fuseLayersForONNX(layerClassObj, layername, layerType, layerNameToInfoMap, weights):
    if 'input' not in layerNameToInfoMap[layername]:
        return

    currentInputs = layerNameToInfoMap[layername]['input']
    for inputLayername in currentInputs:
        if inputLayername in list(layerNameToInfoMap.keys()):
            inputLayerType = layerNameToInfoMap[inputLayername]['opType']
            # print(inputLayername + " " + inputLayerType)
            if isFusibleWith(layerType, inputLayerType):
                layerClassObj.mergeLayersDataForONNX(layername, inputLayername, layerNameToInfoMap, weights)
                updateInputsForONNX(layername, inputLayername, currentInputs, layerNameToInfoMap)
                del layerNameToInfoMap[layername]
                del layersToJSON[layername]

def updateInputsForONNX(layername, inputLayername, currentInputs, layerNameToInfoMap):
    # print(layername + " " + inputLayername)
    for layer in list(layerNameToInfoMap.keys()):
        if 'input' not in layerNameToInfoMap[layer]:
            continue
        layerInputList = layerNameToInfoMap[layer]['input']
        for layerInput in layerInputList:
                if layerInput == layername:
                    addInputsForONNX(layer, layerInput, layerInputList, currentInputs,
                                     inputLayername)

def addInputsForONNX(layer, layerInput, layerInputList, currentInputs, fusingWith):
    # print("Removing: " + str(layerInput) + " for " + layer)
    idx = layerInputList.index(layerInput)
    layerInputList.remove(layerInput)
    insertAtIdx = idx
    for currentInput in currentInputs:
            if currentInput == fusingWith:
                # print("Appending: " + str(currentInput) + " for " + layer)
                layerInputList.insert(insertAtIdx, currentInput)
                insertAtIdx = insertAtIdx + 1