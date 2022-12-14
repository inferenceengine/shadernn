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
import enum

from config.formatConfig.supportedFormats import SUPPORTED_FORMATS


class ConverterImpl(enum.Enum):
    H5ToJsonConverter = 1,
    ONNXToJsonConverter = 2,
    CustomConverter = 3


FormatToImpl = {
    SUPPORTED_FORMATS.H5ToJson: ConverterImpl.H5ToJsonConverter,
    SUPPORTED_FORMATS.ONNXToJson: ConverterImpl.ONNXToJsonConverter
}
