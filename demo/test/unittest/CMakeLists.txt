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
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(snn-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../snn-core-install)

set(config_file "${snn-dir}/config.txt")
if(NOT EXISTS ${config_file})
    message(FATAL_ERROR "${config_file} file not found. Build the core first")
endif()
file(STRINGS ${config_file} PLATFORMS)
message(STATUS "Build for platforms: ${PLATFORMS}")
string(FIND ${PLATFORMS} "GL" POS_GL)
string(FIND ${PLATFORMS} "VULKAN" POS_VULKAN)
if (${POS_GL} GREATER_EQUAL 0)
    message(STATUS "Building with OpenGL support")
    set(SUPPORT_GL 1)
    add_definitions( -DSUPPORT_GL )
endif()
if (${POS_VULKAN} GREATER_EQUAL 0)
    message(STATUS "Building with Vulkan support")
    set(SUPPORT_VULKAN 1)
    add_definitions( -DSUPPORT_VULKAN )
endif()
if (NOT DEFINED SUPPORT_GL AND NOT DEFINED SUPPORT_VULKAN)
    message(FATAL_ERROR "Neither OpenGL nor Vulkan is supported! Build the core first")
endif()

if (DEFINED ENV{SNN_PROFILING})
    add_definitions( -DPROFILING )
endif()

set (NCNN_RELEASE_UBUNTU "ncnn-20211208-ubuntu-1804")
if (DEFINED ENV{NCNN_RELEASE_UBUNTU})
    set (NCNN_RELEASE_UBUNTU $ENV{NCNN_RELEASE_UBUNTU})
endif()
message("NCNN_RELEASE_UBUNTU=${NCNN_RELEASE_UBUNTU}")

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set (NCNN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../${NCNN_RELEASE_UBUNTU})
    set(ncnn_DIR ${NCNN_ROOT_DIR}/lib/cmake/ncnn)
    find_package(OpenCV 4 REQUIRED)
elseif (DEFINED ANDROID_ABI)
    set (NCNN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ncnn-20211208-android/${ANDROID_ABI})
    set(ncnn_DIR ${NCNN_ROOT_DIR}/lib/cmake/ncnn)
    set(CMAKE_CXX_FLAGS "-D__ANDROID__ -Wno-self-assign -Wnon-pod-varargs")
    set(opencv-lib-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/3rdparty/opencv/android/lib/${ANDROID_ABI})
    set(opencv-inc-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/3rdparty/opencv/android/include/)
    find_library( # Sets the name of the path variable.
            log-lib
            # Specifies the name of the NDK library that
            # you want CMake to locate.
            log)
    set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../OpenCV-android-sdk/sdk/native/jni)
    find_package(OpenCV REQUIRED)
endif()

find_package(ncnn REQUIRED)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND DEFINED SUPPORT_GL)
    find_package(OpenGL REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -ftree-vectorize -Wno-unused-parameter -Wno-missing-braces")

if (DEFINED ANDROID_ABI)
    set(CMAKE_CXX_FLAGS " -Wno-self-assign -Wnon-pod-varargs")
endif()

set(3rdparty-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/3rdparty)

set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
set(SRC_COMMON
    ${COMMON_DIR}/matutil.cpp
    ${COMMON_DIR}/testutil.cpp
    ${COMMON_DIR}/shaderUnitTest.cpp
    ${COMMON_DIR}/modelInference.cpp
    ${COMMON_DIR}/inferenceProcessor.cpp
)

set(snn_3rdparty_headers
    ${3rdparty-dir}/eigen-3.4.0/
    ${3rdparty-dir}/picojson/include/
    ${3rdparty-dir}/cli11/include/
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${snn-dir}/lib/linux_x86_64/lib ${ncnn_DIR})
endif()

add_library(snn_common STATIC ${SRC_COMMON})
target_include_directories(snn_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ncnn-20211208/src)
target_include_directories(snn_common PRIVATE ${NCNN_ROOT_DIR}/include/ncnn)
target_include_directories(snn_common PRIVATE ${snn_3rdparty_headers})
target_include_directories(snn_common PRIVATE ${snn-dir}/includes/inc)
target_include_directories(snn_common PRIVATE ${snn-dir}/includes/src)
target_include_directories(snn_common PRIVATE ${COMMON_DIR})

target_compile_options (snn_common PRIVATE -fexceptions -frtti -Wno-unused-function -Wno-sign-compare -Wno-format-extra-args -Wno-maybe-uninitialized)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(snn_common PRIVATE -D_DEBUG -g)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_include_directories(snn_common PRIVATE ${OpenCV_INCLUDE_DIRS})
elseif (DEFINED ANDROID_ABI)
    target_include_directories(snn_common PRIVATE ${opencv-inc-dir})
endif()

macro(snn_add_test name suffix)
    add_executable(${name}${suffix} ${CMAKE_CURRENT_SOURCE_DIR}/${name}${suffix}.cpp)

    target_include_directories(${name}${suffix} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ncnn-20211208/src)
    target_include_directories(${name}${suffix} PRIVATE ${snn_3rdparty_headers})
    target_include_directories(${name}${suffix} PRIVATE ${snn-dir}/includes/inc)
    target_include_directories(${name}${suffix} PRIVATE ${snn-dir}/includes/src)
    target_include_directories(${name}${suffix} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${name}${suffix} PRIVATE ${COMMON_DIR})

    target_compile_options (${name}${suffix} PRIVATE -fexceptions -frtti -Wno-unused-function -Wno-sign-compare -Wno-format-extra-args -Wno-maybe-uninitialized)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${name}${suffix} PRIVATE -D_DEBUG -g)
    endif()

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_include_directories(${name}${suffix} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${name}${suffix} PRIVATE -Wl,--start-group dl ${snn-dir}/lib/linux_x86_64/libsnn_core.so snn_common ncnn ${OpenCV_LIBS} stdc++fs dl)
        if (DEFINED SUPPORT_GL)
            target_link_libraries(${name}${suffix} PRIVATE OpenGL::EGL OpenGL::OpenGL glfw)
        endif()
    elseif (DEFINED ANDROID_ABI)
        find_library(ANDROID_LOG_LIB log)
        target_include_directories(${name}${suffix} PRIVATE ${opencv-inc-dir})
        target_link_libraries(${name}${suffix} PRIVATE snn_common ncnn android EGL GLESv2 GLESv3 
                                                   ${opencv-lib-dir}/libopencv_core.so 
                                                   ${opencv-lib-dir}/libopencv_imgproc.so 
                                                   ${opencv-lib-dir}/libopencv_video.so 
                                                   ${OpenCV_LIBS}
                                                   ${ANDROID_LOG_LIB}
                                                   ${snn-dir}/lib/${ANDROID_ABI}/libsnn_core.so 
                                                   )
        if (DEFINED SUPPORT_GL)
            target_link_libraries(${name}${suffix} PRIVATE vulkan)
        endif()
    endif()
endmacro()

# Unit tests for operators
snn_add_test(binaryOp Test)
snn_add_test(convolution Test)
snn_add_test(pooling Test)
snn_add_test(imageTexture Test)
snn_add_test(imageTextureResize Test)
snn_add_test(activation Test)
snn_add_test(batchNorm Test)
snn_add_test(concat Test)
snn_add_test(depthwiseConv2D Test)
snn_add_test(instanceNorm Test)
snn_add_test(pad Test)
snn_add_test(upSample Test)
snn_add_test(flatten Test)
snn_add_test(dense Test)
snn_add_test(multiInputs Test)
# Unit tests for models
snn_add_test(resnet18 Test)
snn_add_test(resnet18Finetuned Test)
snn_add_test(inferenceProcessor Test)
snn_add_test(yolov3Tiny Test)
snn_add_test(yolov3TinyFinetuned Test)
snn_add_test(unet Test)
snn_add_test(unetFinetuned Test)
snn_add_test(mobilenetv2 Test)
snn_add_test(mobilenetv2Finetuned Test)
snn_add_test(styleTransfer Test)
