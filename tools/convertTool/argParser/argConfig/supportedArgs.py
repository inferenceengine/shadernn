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
# Supported Arguments for argparser

SUPPORTED_ARGS = {
    # Structure -> argumentName : {arugments metadata dictionary with key,value as argparser supported key,value}
    '-filepath': {'action': 'store',
                  'type': str,
                  'required': True,
                  'help': 'Provide Filepath to H5/Pth file!'
                  },
    '-customLayersFile': {'action': 'store',
                          'type': str,
                          'required': False,
                          'help': 'Provide Custom Layer Definition File Path!',
                          'default': 'layers/customLayers/supportedCustomLayers.py'
                          },
    '-supportedFramework': {'action': 'store',
                            'type': str,
                            'required': False,
                            'help': 'Provide supported framework',
                            'default': 'SNN'
                            },
    '-layer': {
        'action': 'store',
        'type': str,
        'required': False,
        'help': 'Provide custom layer for SNN'
    },

    '-gpu': {
        'action': 'store',
        'type': str,
        'required': False,
        'help': 'Provide shader option for GPU. Kindly check supportedSNNOptions.py for options'
    },

    '-cpufile': {
        'action': 'store',
        'type': str,
        'required': False,
        'help': 'Provide program filepath for CPU.'
    },

    '-shaderfile': {
        'action': 'store',
        'type': str,
        'required': False,
        'help': 'Provide shader filepath.'
    },

    '-programfile': {
        'action': 'store',
        'type': str,
        'required': False,
        'help': 'Provide program filepath fro shader compatibility.'
    },

    '-decoupleWeights': {
        'action': 'store_true',
        'required': False,
        'help': 'Add -d flag to get decoupled weights in binary format.'
    }
}
