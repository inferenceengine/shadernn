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

def openReadFile(path):
    try:
        with open(path, mode="rb") as infile:
            content = infile.read()
            return content
    except:
        pattern()
        print("\nError: Cannot open file; Probably invalid path")
        print("\nPath specified is "+str(path))
        return -1
    

def readWeight(path):
    content = openReadFile(path) 
    path, filename = os.path.split(path) 
    if content == -1:
        return 0
    unableToRead = 0
    header = struct.unpack("32s", content[0: 32])[0]
    pixelValues = []

    for index in range(0, len(content), 4):
        try:
            pixelValues.append(struct.unpack("f", content[index: index+4])[0])
        except:
            pass
   
    pixelArray = np.array(pixelValues, dtype = np.float32)
    resultImage = os.path.join(path, filename[:-5])
    
    np.savetxt(resultImage + "_entireArray.txt", pixelArray, fmt = '%.6f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path of input texture", required=True)
    args        = parser.parse_args()
    path = args.path
    readWeight(path)
