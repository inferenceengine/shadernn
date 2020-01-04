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
from errorProcessor.errorConfig.errorCodes import Errors
from errorProcessor.errors.EmptyFileError import EmptyFileError
from errorProcessor.errors.FileNotPresentError import FileNotPresentError
from errorProcessor.errors.InvalidCompatibleFrameworkError import InvalidCompatibleFrameworkError
from errorProcessor.errors.InvalidModelFileFormatError import InvalidModelFileFormatError
from errorProcessor.errors.UnsupportedLayerError import UnsupportedLayerError
from errorProcessor.errors.InvalidCustomLayerFileFormatError import InvalidCustomLayerFileFormatError
from errorProcessor.errors.GenericError import GenricError


def getError(errorCode):
    if errorCode == Errors.FILE_NOT_FOUND:
        error = FileNotPresentError()
    elif errorCode == Errors.INVALID_MODELFILE_FORMAT:
        error = InvalidModelFileFormatError()
    elif errorCode == Errors.EMPTY_FILE:
        error = EmptyFileError()
    elif errorCode == Errors.INVALID_COMPATIBLE_FRAMEWORK:
        error = InvalidCompatibleFrameworkError()
    elif errorCode == Errors.UNSUPPORTED_LAYER:
        error = UnsupportedLayerError()
    elif errorCode == Errors.INVALID_CUSTOM_LAYER_FILE_FORMAT:
        error = InvalidCustomLayerFileFormatError()
    else:
        error = GenricError()
    return error