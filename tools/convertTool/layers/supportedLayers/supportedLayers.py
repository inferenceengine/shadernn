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
import enum


class SUPPORTED_LAYERS(enum.Enum):
    ADD = 1,
    AVERAGEPOOLING2D = 2,
    CALCULATE = 3,
    CONCATENATE = 4,
    CONV2D = 5,
    CONV2DTRANSPOSE = 6,
    DENSE = 7,
    DEPTHWISE = 8,
    DEPTHWISECONV2D = 9,
    FLATTEN = 10,
    GLOBALAVERAGEPOOLING2D = 11,
    INPUTLAYER = 12,
    LAMBDA = 13,
    MAXPOOLING2D = 14,
    MULTIPLY = 15,
    RESHAPE = 16,
    SEQUENTIAL = 17,
    UPSAMPLING2D = 18,
    ZEROPADDING2D = 19


LayersToSNN = {
    SUPPORTED_LAYERS.ADD: "Add",
    SUPPORTED_LAYERS.AVERAGEPOOLING2D: "AveragePooling2D",
    SUPPORTED_LAYERS.CALCULATE: "Calculate",
    SUPPORTED_LAYERS.CONCATENATE: "Concatenate",
    SUPPORTED_LAYERS.CONV2D: "Conv2D",
    SUPPORTED_LAYERS.CONV2DTRANSPOSE: "Conv2DTranspose",
    SUPPORTED_LAYERS.DENSE: "Dense",
    SUPPORTED_LAYERS.DEPTHWISE: "Depthwise",
    SUPPORTED_LAYERS.DEPTHWISECONV2D: "DepthwiseConv2D",
    SUPPORTED_LAYERS.FLATTEN: "Flatten",
    SUPPORTED_LAYERS.GLOBALAVERAGEPOOLING2D: "GlobalAveragePooling2D",
    SUPPORTED_LAYERS.INPUTLAYER: "InputLayer",
    SUPPORTED_LAYERS.LAMBDA: "Lambda",
    SUPPORTED_LAYERS.MAXPOOLING2D: "MaxPooling2D",
    SUPPORTED_LAYERS.MULTIPLY: "Multiply",
    SUPPORTED_LAYERS.RESHAPE: "Reshape",
    SUPPORTED_LAYERS.SEQUENTIAL: "Sequential",
    SUPPORTED_LAYERS.UPSAMPLING2D: "UpSampling2D",
    SUPPORTED_LAYERS.ZEROPADDING2D: "ZeroPadding2D",
}

SNNtoLayers = {
    "Add": SUPPORTED_LAYERS.ADD,
    "AveragePooling2D": SUPPORTED_LAYERS.AVERAGEPOOLING2D,
    "Calculate": SUPPORTED_LAYERS.CALCULATE,
    "Concatenate": SUPPORTED_LAYERS.CONCATENATE,
    "Conv2D": SUPPORTED_LAYERS.CONV2D,
    "Conv2DTranspose": SUPPORTED_LAYERS.CONV2DTRANSPOSE,
    "Dense": SUPPORTED_LAYERS.DENSE,
    "Depthwise": SUPPORTED_LAYERS.DEPTHWISE,
    "DepthwiseConv2D": SUPPORTED_LAYERS.DEPTHWISECONV2D,
    "Flatten": SUPPORTED_LAYERS.FLATTEN,
    "GlobalAveragePooling2D": SUPPORTED_LAYERS.GLOBALAVERAGEPOOLING2D,
    "InputLayer": SUPPORTED_LAYERS.INPUTLAYER,
    "Lambda": SUPPORTED_LAYERS.LAMBDA,
    "MaxPooling2D": SUPPORTED_LAYERS.MAXPOOLING2D,
    "Multiply": SUPPORTED_LAYERS.MULTIPLY,
    "Reshape": SUPPORTED_LAYERS.RESHAPE,
    "Sequential": SUPPORTED_LAYERS.SEQUENTIAL,
    "UpSampling2D": SUPPORTED_LAYERS.UPSAMPLING2D,
    "ZeroPadding2D": SUPPORTED_LAYERS.ZEROPADDING2D
}
