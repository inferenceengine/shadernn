# Copyright (C) 2020 - 2022 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
ROOT=`dirname "$(realpath $0)"`

NCNN_RELEASE_ANDROID="ncnn-20211208-android"
NCNN_RELEASE_ANDROID_URL="https://github.com/Tencent/ncnn/releases/download/20211208/ncnn-20211208-android.zip"
NCNN_RELEASE_UBUNTU="ncnn-20211208-ubuntu-1804"
NCNN_RELEASE_UBUNTU_URL="https://github.com/Tencent/ncnn/releases/download/20211208/ncnn-20211208-ubuntu-1804.zip"
NCNN_SRC="ncnn-20211208"
NCNN_SRC_URL="https://github.com/Tencent/ncnn/archive/refs/tags/20211208.zip"
OPENCV_RELEASE_ANDROID="OpenCV-android-sdk"
OPENCV_RELEASE_ANDROID_URL="https://github.com/opencv/opencv/releases/download/4.5.4/opencv-4.5.4-android-sdk.zip"

make_dir()
{
    if [ ! -d $1 ]
    then
        mkdir $1
    fi
}

clean()
{
    rm -rf "${ROOT}/test/${NCNN_RELEASE_ANDROID}"
    rm -rf "${ROOT}/test/${NCNN_RELEASE_UBUNTU}"    
    rm -rf "${ROOT}/test/${NCNN_SRC}"    
    rm -rf "${ROOT}/test/${OPENCV_RELEASE_ANDROID}"        
}

print_usage()
{
    echo
    echo "SNN Install Dependences Script..."
    echo
    echo "Usage: Linux Dependences: `basename $0` linux"
    echo
    echo "Usage: Android Dependences: `basename $0` android"
    echo
    echo "Usage: Clean: `basename $0` clean"
    echo
}

build_android()
{
    if [ ! -d "${ROOT}/test/${NCNN_SRC}" ]; then
        echo "NCNN Source Codes Not Found"
        wget "${NCNN_SRC_URL}" -O "${ROOT}/test/ncnn_src.zip"
        unzip "${ROOT}/test/ncnn_src.zip" -d "${ROOT}/test"
        rm "${ROOT}/test/ncnn_src.zip"
    fi        
    if [ ! -d "${ROOT}/test/${NCNN_RELEASE_ANDROID}" ]; then
        echo "NCNN Android Release Not Found"
        wget "${NCNN_RELEASE_ANDROID_URL}" -O "${ROOT}/test/ncnn_android_release.zip"
        unzip "${ROOT}/test/ncnn_android_release.zip" -d "${ROOT}/test"
        rm "${ROOT}/test/ncnn_android_release.zip"
        sed -i 's/INTERFACE_COMPILE_OPTIONS "-fno-rtti;-fno-exceptions"/#INTERFACE_COMPILE_OPTIONS "-fno-rtti;-fno-exceptions"/' "${ROOT}/test/${NCNN_RELEASE_ANDROID}/arm64-v8a/lib/cmake/ncnn/ncnn.cmake"
        sed -i 's/INTERFACE_COMPILE_OPTIONS "-fno-rtti;-fno-exceptions"/#INTERFACE_COMPILE_OPTIONS "-fno-rtti;-fno-exceptions"/' "${ROOT}/test/${NCNN_RELEASE_ANDROID}/armeabi-v7a/lib/cmake/ncnn/ncnn.cmake"    
    fi
    if [ ! -d "${ROOT}/test/${OPENCV_RELEASE_ANDROID}" ]; then
        echo "OpenCV Android Release Not Found"
        wget "${OPENCV_RELEASE_ANDROID_URL}" -O "${ROOT}/test/opencv_android_release.zip"
        unzip "${ROOT}/test/opencv_android_release.zip" -d "${ROOT}/test"
        rm "${ROOT}/test/opencv_android_release.zip"        
    fi
}

build_linux()
{
    if [ ! -d "${ROOT}/test/${NCNN_SRC}" ]; then
        echo "NCNN Source Codes Not Found"
        wget "${NCNN_SRC_URL}" -O "${ROOT}/test/ncnn_src.zip"
        unzip "${ROOT}/test/ncnn_src.zip" -d "${ROOT}/test"
        rm "${ROOT}/test/ncnn_src.zip"
    fi   
    if [ ! -d "${ROOT}/test/${NCNN_RELEASE_UBUNTU}" ]; then
        echo "NCNN Ubuntu Release Not Found"
        wget "${NCNN_RELEASE_UBUNTU_URL}" -O "${ROOT}/test/ncnn_ubuntu_release.zip"
        unzip "${ROOT}/test/ncnn_ubuntu_release.zip" -d "${ROOT}/test"
        rm "${ROOT}/test/ncnn_ubuntu_release.zip"
    fi 
}


if [ "$1" == "clean" ]; then
    clean
    exit 0
elif [ "$1" == "android" ]; then
    build_android
    exit 0
elif [ "$1" == "linux" ]; then
    build_linux
    exit 0
else 
    print_usage
    exit 1
fi
