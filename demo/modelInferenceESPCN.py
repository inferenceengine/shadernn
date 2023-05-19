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
import tensorflow as tf
import numpy as np
import argparse, os, cv2

from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Input, Conv2D, Lambda
from keras.layers import Activation

scale_x = 1

def SubpixelConv2D(name="subpixel"):
    def subpixel_shape(input_shape):
        global scale_x
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale_x,
                None if input_shape[2] is None else input_shape[2] * scale_x,
                1]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        global scale_x
        return tf.compat.v1.depth_to_space(x, scale_x)

    return Lambda(subpixel, output_shape= subpixel_shape, name=name)

class ESPCN:

    def __init__(self, scale):
        global scale_x
        scale_x = scale
        self.model = self.buildModel()

    def buildModel(self):
        global scale_x
        inputs = Input(shape=(None, None, 1), name= 'input')

        x = Conv2D(filters = 16, kernel_size = (5,5), strides=1,
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None),bias_initializer='zeros', 
                    padding = "same",activation='relu',name='conv_1')(inputs)

        x = Conv2D(filters = 16, kernel_size = (3,3), strides=1,
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None),bias_initializer='zeros', 
                padding = "same",activation='relu',name='conv_2')(x)

        x = Conv2D(filters = scale_x**2*1, kernel_size = (3,3), strides=1,
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None),bias_initializer='zeros', 
            padding = "same",name='conv_3')(x)           

        x = SubpixelConv2D(name='subpixel')(x)

        x = Activation('tanh')(x)
        
        model = Model(inputs=inputs, outputs=x)

        return model

def preprocessImage(path):
    I = cv2.imread(path)
    ## In case of using ant-y-channel image, use following method and skip conversion: 
    # Y = I[:, :, 2] 
    YCbCr = cv2.cvtColor(I, cv2.COLOR_BGR2YCrCb) 
    Y = YCbCr[:,:,0]
    Y = np.reshape(Y, (1, Y.shape[0], Y.shape[1], 1))
    Y = Y/255
    return Y

def inferModel(image, _model, savePath):
    print("\nTesting Image")
    Y_input = preprocessImage(image)
    Y_output = _model.predict(Y_input)
    print("\nModel Inference complete")
    
    Y_output = Y_output * 255
    subpixelFactor = 2
    Y_output = Y_output.reshape(Y_input.shape[1] * subpixelFactor, Y_input.shape[2] * subpixelFactor)
    
    print("\nWriting Result")
    cv2.imwrite(os.path.join(savePath, "espcnResult.png"), Y_output)
    return

def loadModel(modelPath):
    print("\nLoading Model")
    _model = ESPCN(scale=2).model
    _model.load_weights(modelPath)
    print("\nModel Weights loaded")
    return _model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model path", required=True)
    parser.add_argument("--image", help="Path to image", required=True)
    parser.add_argument("--outdir", help="Directory for saving result", required=True)
    args = parser.parse_args()
    
    testImage  = str(args.image)
    modelPath  = str(args.model)
    resultPath = str(args.outdir)

    _model = loadModel(modelPath)
    inferModel(testImage, _model, resultPath)
    