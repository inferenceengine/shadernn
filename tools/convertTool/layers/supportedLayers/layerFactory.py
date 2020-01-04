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
import errorProcessor.errorHandler
from errorProcessor.errorConfig.errorCodes import Errors
from layers.supportedLayers.Lambda import Lambda
from layers.supportedLayers.activation import Activation
from layers.supportedLayers.add import Add
from layers.supportedLayers.averagepooling2d import AveragePooling2D
from layers.supportedLayers.batchNormalization import BatchNormalization
from layers.supportedLayers.calculate import Calculate
from layers.supportedLayers.concat import Concat
from layers.supportedLayers.conv2d import Conv2D
from layers.supportedLayers.conv2dtranspose import Conv2DTranspose
from layers.supportedLayers.dense import Dense
from layers.supportedLayers.depthwise import Depthwise
from layers.supportedLayers.dropout import Dropout
from layers.supportedLayers.flatten import Flatten
from layers.supportedLayers.gemm import Gemm
from layers.supportedLayers.globalaveragepooling2d import GlobalAveragePooling2D
from layers.supportedLayers.input import Input
from layers.supportedLayers.maxpooling2d import MaxPooling2D
from layers.supportedLayers.multiply import Multiply
from layers.supportedLayers.pad import Pad
from layers.supportedLayers.reshape import Reshape
from layers.supportedLayers.sequential import Sequential
from layers.supportedLayers.upsampling2d import UpSampling2D
from layers.supportedLayers.zeropadding2d import ZeroPadding2D


def getLayer(type):
    layer = None
    if type == 'Add':
        layer = Add()
    elif type == 'Activation' or type == 'ReLU' or type == 'LeakyReLU' or type == 'Relu' \
            or type == 'Swish' or type == 'Clip' or type == 'Tanh':
        layer = Activation()
    elif type == 'AveragePooling2D' or type == 'AveragePool':
        layer = AveragePooling2D()
    elif type == 'BatchNormalization':
        layer = BatchNormalization()
    elif type == 'Calculate':
        layer = Calculate()
    elif type == 'Concatenate' or type == 'Concat':
        layer = Concat()
    elif type == 'Conv2D' or type == 'Conv':
        layer = Conv2D()
    elif type == 'Conv2DTranspose':
        layer = Conv2DTranspose()
    elif type == 'Dense':
        layer = Dense()
    elif type == 'Depthwise' or type == 'DepthwiseConv2D':
        layer = Depthwise()
    elif type == 'Dropout':
        layer = Dropout()
    elif type == 'Flatten':
        layer = Flatten()
    elif type == 'GlobalAveragePooling2D' or type == 'GlobalAveragePool':
        layer = GlobalAveragePooling2D()
    elif type == 'InputLayer':
        layer = Input()
    elif type == 'Lambda':
        layer = Lambda()
    elif type == 'MaxPooling2D' or type == 'MaxPool':
        layer = MaxPooling2D()
    elif type == 'Multiply':
        layer = Multiply()
    elif type == 'Reshape':
        layer = Reshape()
    elif type == 'Sequential':
        layer = Sequential()
    elif type == 'UpSampling2D' or type == 'Upsample':
        layer = UpSampling2D()
    elif type == 'ZeroPadding2D':
        layer = ZeroPadding2D()
    elif type == 'Gemm':
        layer = Gemm()
    elif type == 'Pad':
        layer = Pad()
    else:
        # print("LayerClass not found for: "+layerType)
        errorProcessor.errorHandler.handleError(Errors.UNSUPPORTED_LAYER, layername=type)
    return layer

