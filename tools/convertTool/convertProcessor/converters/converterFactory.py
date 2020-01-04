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
from convertProcessor.converters.converterImpl import *
from convertProcessor.converters.customConverter import CustomConverter
from convertProcessor.converters.h5ToJsonConverter import H5ToJsonConverter
from convertProcessor.converters.onnxToJsonConverter import ONNXToJsonConverter


def getConverter(convertImpl):
    # print(convertImpl)
    if convertImpl == ConverterImpl.H5ToJsonConverter:
        return H5ToJsonConverter()
    if convertImpl == ConverterImpl.ONNXToJsonConverter:
        return ONNXToJsonConverter()
    if convertImpl == ConverterImpl.CustomConverter:
        return CustomConverter()