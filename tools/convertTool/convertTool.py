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
from argParser import argHelper
from convertProcessor.converters.converterImpl import *
from convertProcessor.processorConfig.processConfig import *
from convertProcessor.processors.convertProcessorFactory import getConvertProcessor
from layers.customLayers.customlayerHelper import handleCustomLayersFiles, getKerasDefFilePaths
from utils.toolUtils import convertCSVToJSON, getSupportedLayersConfigFromJSON

if __name__ == '__main__':
    argHelper.preapreArgumentParser()
    argHelper.parseArguments()

    if COMPATIBLE_FRAMEWORK == SUPPORTED_COMPATIBLE_FRAMEWORKS.SNN:
        convertCSVToJSON(SNN_SUPPORTEDKERASLAYERS_CSV, SNN_SUPPORTEDKERASLAYERS_JSON)
        getSupportedLayersConfigFromJSON(SNN_SUPPORTEDKERASLAYERS_JSON, iskeras=True)
        convertCSVToJSON(SNN_SUPPORTEDONNXLAYERS_CSV, SNN_SUPPORTEDONNXLAYERS_JSON)
        getSupportedLayersConfigFromJSON(SNN_SUPPORTEDONNXLAYERS_JSON, iskeras=False)
        convertCSVToJSON(SNN_CUSTOMLAYERS_CSV, SNN_CUSTOMLAYERS_JSON)
        customLayerKerasDefFilePaths = getKerasDefFilePaths(SNN_CUSTOMLAYERS_JSON)
        handleCustomLayersFiles(customLayerKerasDefFilePaths)

    # print(convertProcessor.processorConfig.processConfig.getCustomLayers())
    # print(getCurrentFormat())
    convertprocessor = getConvertProcessor(getCurrentFormat())
    converter = convertprocessor.buildConverter(convertImpl=FormatToImpl[getCurrentFormat()])
    converter.convert()
