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

cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 17)
set(root-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)

set(core-dir ${root-dir}/core)
set(demo-dir ${root-dir}/demo)
set(snn-install-dir ${root-dir}/snn-core-install)

set(CMAKE_CXX_FLAGS "-D__ANDROID__ -Wno-self-assign -Wnon-pod-varargs")
set(OpenCV_DIR ${demo-dir}/test/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_library(native-lib SHARED
        native-lib.cpp
        ${demo-dir}/common/frameviz.cpp
        ${demo-dir}/common/processingloop.cpp
        nativeFrameProvider.cpp
        )

set(headers-3rdparty
    ${core-dir}/3rdparty/glad/include/
    ${core-dir}/3rdparty/glm/include/
    ${core-dir}/3rdparty/libyuv/include/
    ${core-dir}/3rdparty/eigen-3.4.0/
    ${core-dir}/3rdparty/picojson/include/
    ${core-dir}/3rdparty/stb_image/include/
    ${core-dir}/3rdparty/opencv/android/include/
    ${core-dir}/3rdparty/readerwriterqueue/
)

target_include_directories(native-lib PRIVATE ${snn-install-dir}/includes/inc ${snn-install-dir}/includes/ ${headers-3rdparty})

target_compile_options(native-lib PRIVATE -Werror -Wall -Wextra -Wno-unused-function)

set(opencv-lib-dir ${core-dir}/3rdparty/opencv/android/lib/${ANDROID_ABI})
target_link_libraries(native-lib
        android
        log
        ${snn-install-dir}/lib/arm64-v8a/libsnn_core.so
        ${OpenCV_LIBS}
        )
