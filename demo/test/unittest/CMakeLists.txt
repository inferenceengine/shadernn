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

cmake_minimum_required(VERSION 3.9)

project(unit-tests)
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn-20211208-ubuntu-1804/lib/cmake/ncnn)
    find_package(OpenCV 4 REQUIRED)
elseif (DEFINED ANDROID_ABI)
    set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn-20211208-android/${ANDROID_ABI}/lib/cmake/ncnn)
    set(CMAKE_CXX_FLAGS "-D__ANDROID__ -Wno-self-assign -Wnon-pod-varargs")
    set(opencv-lib-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/3rdparty/opencv/android/lib/${ANDROID_ABI})
    set(opencv-inc-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/3rdparty/opencv/android/include/)
    find_library( # Sets the name of the path variable.
            log-lib
            # Specifies the name of the NDK library that
            # you want CMake to locate.
            log)
    set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../OpenCV-android-sdk/sdk/native/jni)
    find_package(OpenCV REQUIRED)
endif()

find_package(ncnn REQUIRED)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(OpenGL REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -ftree-vectorize -Wno-unused-parameter -Wno-missing-braces")

set(snn-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../snn-core-install)
set(3rdparty-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/3rdparty)

file(GLOB_RECURSE snn_headers 
    ${snn-dr}/includes/inc/snn/*.h 
    ${snn-dir}/includes/inc/*.h
    ${snn-dir}/includes/src/*.h
)

macro(snn_add_test name suffix)
    add_executable(${name}${suffix} ${CMAKE_CURRENT_SOURCE_DIR}/${name}${suffix}.cpp shaderUnitTest.cpp )

    if (DEFINED ANDROID_ABI)
        set(CMAKE_CXX_FLAGS " -Wno-self-assign -Wnon-pod-varargs")
    endif()

    target_include_directories(${name}${suffix} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn-20211208/src)
    target_include_directories(${name}${suffix} PRIVATE ${3rdparty-dir}/glad/include)
    target_include_directories(${name}${suffix} PRIVATE ${3rdparty-dir}/glm/include)
    target_include_directories(${name}${suffix} PRIVATE ${snn-dir}/includes/inc)
    target_include_directories(${name}${suffix} PRIVATE ${snn-dir}/includes/src)  
    target_include_directories(${name}${suffix} PRIVATE ${3rdparty-dir}/eigen-3.4.0)
    target_include_directories(${name}${suffix} PRIVATE ${3rdparty-dir}/misc/include)
    target_include_directories(${name}${suffix} PRIVATE ${3rdparty-dir}/readerwriterqueue)
    target_include_directories(${name}${suffix} PRIVATE ${3rdparty-dir}/cli11/include)
    target_include_directories(${name}${suffix} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options (${name}${suffix} PRIVATE -fexceptions -frtti -Wno-unused-function -Wno-sign-compare -Wno-format-extra-args)

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_include_directories(${name}${suffix} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${name}${suffix} PRIVATE ${snn-dir}/lib/linux_x86_64/libsnn_core.so ncnn ${OpenCV_LIBS} OpenGL::EGL OpenGL::OpenGL stdc++fs glfw)
    elseif (DEFINED ANDROID_ABI)
        find_library(ANDROID_LOG_LIB log) 
        target_include_directories(${name}${suffix} PRIVATE ${opencv-inc-dir})
        target_link_libraries(${name}${suffix} PRIVATE ncnn android EGL GLESv2 GLESv3 
                                                   ${opencv-lib-dir}/libopencv_core.so 
                                                   ${opencv-lib-dir}/libopencv_imgproc.so 
                                                   ${opencv-lib-dir}/libopencv_video.so 
                                                   ${OpenCV_LIBS}
                                                   ${ANDROID_LOG_LIB}
                                                   ${snn-dir}/lib/${ANDROID_ABI}/libsnn_core.so 
                                                   )
    endif()
endmacro()

#Unit test for Operator
snn_add_test(binaryOp Test)
snn_add_test(convolution Test)
snn_add_test(pooling Test)
snn_add_test(imageTexture Test)
snn_add_test(activation Test)
snn_add_test(batchNorm Test)
snn_add_test(concat Test)
snn_add_test(depthwiseConv2D Test)
snn_add_test(instanceNorm Test)
snn_add_test(pad Test)
snn_add_test(upSample Test)

#Model test
snn_add_test(alexnet Test)
snn_add_test(resnet18 Test)
snn_add_test(inferenceProcessor Test)
snn_add_test(yolov3Tiny Test)
snn_add_test(yolov3TinyFinetuned Test)
snn_add_test(unet Test)
snn_add_test(unetFinetuned Test)
snn_add_test(mobilenetv2 Test)
snn_add_test(mobilenetv2Finetuned Test)
snn_add_test(aiDenoise Test)
snn_add_test(styleTransfer Test)


