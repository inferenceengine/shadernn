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
import argparse

# Create the parser
from argParser.argConfig.supportedArgs import SUPPORTED_ARGS
from utils.toolUtils import *
from layers.customLayers import customlayerHelper

argument_parser = argparse.ArgumentParser(description='Convert H5/ONNX to JSON !')


def preapreArgumentParser():
    # Add the arguments
    for supportedArg in SUPPORTED_ARGS:
        arg = supportedArg
        kwargs = {}

        for argMetadata in SUPPORTED_ARGS[supportedArg]:
            kwargs[argMetadata] = SUPPORTED_ARGS[supportedArg][argMetadata]

        argument_parser.add_argument(arg, **kwargs)


def parseArguments():
    # Execute the parse_args() method
    args = argument_parser.parse_args()

    input_filepath = args.filepath
    compatible_framework = args.supportedFramework
    custom_layerfilepath = args.customLayersFile
    decoupleWeights = args.decoupleWeights

    checkModelFile(input_filepath)
    checkCompatibleFramework(compatible_framework)
    checkCustomLayerFile(custom_layerfilepath)
    setDecoupleWeights(decoupleWeights)

    custom_layerfilepaths = [custom_layerfilepath]
    customlayerHelper.handleCustomLayersFiles(custom_layerfilepaths)