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
# toolUtils will act as current state manager for convert tool for the session
# It can be used to save/get global variables across the tool
# It can have common utility functions
import csv
import json
import ntpath
import os

import numpy as np
from config.formatConfig.supportedFormats import *
from config.frameworkConfig.supportedFrameworks import *
from convertProcessor.processorConfig import processConfig
from convertProcessor.processorConfig.processConfig import SNN_SUPPORTEDKERASLAYERS_CONFIGS, \
    getModelFile, SNN_SUPPORTEDONNXLAYERS_CONFIGS
from errorProcessor.errorConfig.errorCodes import Errors
from errorProcessor.errorHandler import *

CUSTOM_LAYER_FILE_FORMAT = {".py"}


def checkFileExists(filepath):
    if not os.path.isfile(filepath):
        handleError(Errors.FILE_NOT_FOUND, filepath=filepath)


def getFileName(filepath):
    return ntpath.basename(filepath)


def getFileNameWithoutExtension(filepath):
    # print(os.path.splitext(filepath)[0])
    return os.path.splitext(filepath)[0]


def checkModelFile(filepath):
    checkFileExists(filepath)
    filename = getFileName(filepath)
    validFileFormat = False
    for modelFormat in modelFormatMap:
        if filename.lower().endswith(modelFormat):
            processConfig.CURRENT_FORMAT = modelFormatMap[modelFormat]
            processConfig.MODEL_FILE = filepath
            validFileFormat = True
            break

    if not validFileFormat:
        handleError(Errors.INVALID_MODELFILE_FORMAT, filepath=filepath)

    checkFileSize(filepath)


def checkFileSize(filepath):
    if os.path.getsize(filepath) > 0:
        return
    handleError(Errors.EMPTY_FILE, filepath=filepath)


def checkCompatibleFramework(input_compatible_framework):
    validCompatibleFramework = False
    for compatible_framework in compatible_frameworks:
        if input_compatible_framework == compatible_framework:
            processConfig.COMPATIBLE_FRAMEWORK = compatible_frameworks[compatible_framework]
            validCompatibleFramework = True
            break

    if not validCompatibleFramework:
        handleError(Errors.INVALID_COMPATIBLE_FRAMEWORK, compatible_framework=input_compatible_framework)


def checkCustomLayerFile(filepath):
    checkFileExists(filepath)
    filename = getFileName(filepath)
    validFileFormat = False
    for fileformat in CUSTOM_LAYER_FILE_FORMAT:
        if filename.lower().endswith(fileformat):
            validFileFormat = True
            break

    if not validFileFormat:
        handleError(Errors.INVALID_CUSTOM_LAYER_FILE_FORMAT, filepath=filepath)

    checkFileSize(filepath)


def convertCSVToJSON(inputCSVFile, outputJSONFile):

    with open(inputCSVFile, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        data = []
        for rows in csvReader:
            data.append(rows)

    with open(outputJSONFile, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))


def getSupportedLayersConfigFromJSON(jsonFile, iskeras):
    jsonFp = open(jsonFile)
    jsondata = json.load(jsonFp)
    for supportedlayerConfig in jsondata:
        if supportedlayerConfig['fusibleWith'] is not None:
            res = supportedlayerConfig['fusibleWith'].split(';')
            supportedlayerConfig['fusibleWith'] = res
        if iskeras:
            SNN_SUPPORTEDKERASLAYERS_CONFIGS[supportedlayerConfig['layerType']] = supportedlayerConfig
        else:
            SNN_SUPPORTEDONNXLAYERS_CONFIGS[supportedlayerConfig['layerType']] = supportedlayerConfig
    # print(getSNNSupporedLayersConfigs())
    jsonFp.close()


def isFusible(layerType):
    if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
        return SNN_SUPPORTEDKERASLAYERS_CONFIGS[layerType]['isFusible']
    elif processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
        return SNN_SUPPORTEDONNXLAYERS_CONFIGS[layerType]['isFusible']
    else:
        return False


def isFusibleWith(layerType, withLayerType):
    fusibleWith = None
    if processConfig.getCurrentFormat() == SUPPORTED_FORMATS.H5ToJson:
        fusibleWith = SNN_SUPPORTEDKERASLAYERS_CONFIGS[layerType]['fusibleWith']
    elif processConfig.getCurrentFormat() == SUPPORTED_FORMATS.ONNXToJson:
        fusibleWith = SNN_SUPPORTEDONNXLAYERS_CONFIGS[layerType]['fusibleWith']
    # print(fusibleWith)
    if fusibleWith is not None:
        if withLayerType in fusibleWith:
            return True
    return False

def setDecoupleWeights(isDecoupleWeights):
    processConfig.DECOUPLE_WEIGHTS = isDecoupleWeights

def getDecoupledWeightsFileName():
    return (getFileNameWithoutExtension(getFileName(getModelFile()))) + "_weights.bin"

def getCompressedWeightBin(weights):
    weights = np.asarray(weights, dtype=np.float32)
    return weights.tobytes()