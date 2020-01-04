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
from convertProcessor.processors.h5ToJsonProcessor import *
from convertProcessor.processors.onnxToJsonProcessor import *
from config.formatConfig.supportedFormats import *


def getConvertProcessor(format):
    # print(format)
    convertProcessor = None
    if format == SUPPORTED_FORMATS.H5ToJson:
        convertProcessor = H5ToJsonProcessor()
    if format == SUPPORTED_FORMATS.ONNXToJson:
        convertProcessor = ONNXToJsonProcessor()
    return convertProcessor
