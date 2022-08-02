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


class Errors(enum.Enum):
    # Environment related errors

    INVALID_MODELFILE_FORMAT = 101
    EMPTY_FILE = 102
    FILE_NOT_FOUND = 103
    INVALID_COMPATIBLE_FRAMEWORK = 104
    INVALID_CUSTOM_LAYER_FILE_FORMAT = 105

    # Layer related errors
    UNSUPPORTED_LAYER = 201