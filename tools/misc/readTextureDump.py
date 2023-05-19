#!/usr/bin/python3
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
import os
import argparse
import struct
from struct import *
import cv2
import numpy as np

def pattern():
    print("\n> > > > > > > > > \
        \n> > > > > > > > > ")

def openReadFile(texturePath):
    try:
        with open(texturePath, mode="rb") as infile:
            content = infile.read()
            return content
    except:
        pattern()
        print("\nError: Cannot open file; Probably invalid path")
        print("\nPath specified is "+str(texturePath))
        return -1
    
def readTexture(texturePath, width, height, channels, normalizationType, imageSave):
    content = openReadFile(texturePath) 
    path, filename = os.path.split(texturePath) 
    if content == -1:
        return 0
    unableToRead = 0
    header = struct.unpack("32s", content[0: 32])[0]
    header = header.decode("utf-8")
    pattern()
    print("\nHeader read complete")
    print("\nwidth is "+str(width)+" height is "+str(height)+" channels is "+str(channels))
    pixelValues = []
    index = 32

    while index < len(content):
        try:
            if imageSave == "save":
                multiplyBy = 1
                addBy = 0
                if normalizationType == "0-1":
                    multiplyBy = 255
                    addBy = 0
                if normalizationType == "-1-1":
                    multiplyBy = 127.5
                    addBy = 127.5
                pixelValues.append(min(struct.unpack("f", content[index: index+4])[0] * multiplyBy + addBy, 255))
            else:
                pixelValues.append(struct.unpack("f", content[index: index+4])[0])
            index += 4
        except:
            index += 4
            unableToRead += 1
            pass
    pattern()

    print("\nNumber of values skipped while reading = " + str(unableToRead))
    
    pixelArray = np.array(pixelValues, dtype = np.float32)
    resultImage = os.path.join(path, filename[:-5])
    
    try:
        np.savetxt(resultImage + "_entireArray.txt", pixelArray, fmt = '%.6f')
        I = np.reshape(pixelArray, (height, width, channels)) 
    except:
        pattern()
        print(pixelArray.shape)
        print("\nError: dimension issues; Cannot reshape as image")
        return 0
    
    for i in range(0, channels):
        I_channel = I[:,:,i] 
        np.savetxt(resultImage + "_" + str(i) + ".txt", I_channel, fmt = '%.6f')
        if imageSave == "save":
            cv2.imwrite(resultImage + "_" + str(i) + ".png", I_channel)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path of input texture", required=True)
    parser.add_argument("--width", help="Width of texture", required=True)
    parser.add_argument("--height", help="Height of texture", required=True)
    parser.add_argument("--channels", help="Channels in dumpFile", required=True)
    parser.add_argument("--normalization", help="Option for normalization", choices=["NA", "0-1", "-1-1"], required=False, default="NA")
    parser.add_argument("--image", help="Option to save image", choices=["save", "don't_save"], required=False, default= "don't_save")
    args        = parser.parse_args()
    texturePath = args.path
    texWidth    = int(args.width)
    texHeight   = int(args.height)
    texChannels = int(args.channels)
    normalize   = str(args.normalization)
    imageSave   = str(args.image)
    readTexture(texturePath, texWidth, texHeight, texChannels, normalize, imageSave)
    pattern()
    print("\nExit code")
    pattern()
