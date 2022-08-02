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
from config.frameworkConfig.supportedFrameworks import SUPPORTED_COMPATIBLE_FRAMEWORKS

COMPATIBLE_FRAMEWORK = SUPPORTED_COMPATIBLE_FRAMEWORKS.SNN  # default can be set as SNN
CURRENT_FORMAT = None  # It will be set using file format
MODEL_FILE = None   # It will be set using file path
CUSTOM_LAYERS = {}
SNN_SUPPORTEDKERASLAYERS_CSV = 'layers/supportedLayers/supportedKerasLayerSNNConfig.csv'
SNN_SUPPORTEDKERASLAYERS_JSON = 'layers/supportedLayers/supportedKerasLayerSNNConfig.json'
SNN_SUPPORTEDONNXLAYERS_CSV = 'layers/supportedLayers/supportedONNXLayerSNNConfig.csv'
SNN_SUPPORTEDONNXLAYERS_JSON = 'layers/supportedLayers/supportedONNXLayerSNNConfig.json'
SNN_CUSTOMLAYERS_CSV = 'layers/customLayers/customLayerSNNConfig.csv'
SNN_CUSTOMLAYERS_JSON = 'layers/customLayers/customLayerSNNConfig.json'
SNN_SUPPORTEDKERASLAYERS_CONFIGS = {}
SNN_SUPPORTEDONNXLAYERS_CONFIGS = {}
DECOUPLE_WEIGHTS = False  # It will be set using decoupledweights flag

def getCurrentFormat():
    return CURRENT_FORMAT

def getCustomLayers():
    return CUSTOM_LAYERS

def getModelFile():
    return MODEL_FILE

def getSNNSupportedKerasLayersConfigs():
    return SNN_SUPPORTEDKERASLAYERS_CONFIGS

def getSNNSupportedONNXLayersConfigs():
    return SNN_SUPPORTEDONNXLAYERS_CONFIGS

def isDecoupleWeights():
    return DECOUPLE_WEIGHTS