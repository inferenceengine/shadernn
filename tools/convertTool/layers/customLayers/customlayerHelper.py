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
import importlib
import json
from inspect import getmembers, isfunction
from utils import toolUtils
from layers.customLayers import supportedCustomLayers
from convertProcessor.processorConfig import processConfig

def getKerasDefFilePaths(jsonfilepath):
    jsonFp = open(jsonfilepath)
    customlayers = json.load(jsonFp)
    filepaths = []
    for customlayer in customlayers:
        # print("Printed: " + customlayer['kerasDefFilePath'])
        filepaths.append(customlayer['kerasDefFilePath'])
    jsonFp.close()
    return filepaths

def handleCustomLayersFiles(filepaths):
    if filepaths is None:
        return
    customObjects = processConfig.getCustomLayers()

    toolSupportedCustomLayers = getmembers(supportedCustomLayers, isfunction)
    for toolSupportedCustomLayer in toolSupportedCustomLayers:
        customObjects[toolSupportedCustomLayer[0]] = toolSupportedCustomLayer[1]

    for filepath in filepaths:
        if filepath is not None:
            if filepath == '':
                continue
            customlayerfileSpec = importlib.util.spec_from_file_location(
                toolUtils.getFileNameWithoutExtension(filepath),
                filepath)
            customLayerModule = importlib.util.module_from_spec(customlayerfileSpec)
            customlayerfileSpec.loader.exec_module(customLayerModule)

            # For custom definitions
            userProvidedCustomLayers = getmembers(customLayerModule, isfunction)

            for userProvidedCustomLayer in userProvidedCustomLayers:
                customObjects[userProvidedCustomLayer[0]] = userProvidedCustomLayer[1]

            # For custom variables
            for member in getmembers(customLayerModule):
                if member[0] == 'dictionary_of_custom_variables':
                    for key in member[1]:
                        customObjects[key] = member[1][key]

    processConfig.CUSTOM_LAYERS = customObjects
    # print(processConfig.getCustomLayers())
