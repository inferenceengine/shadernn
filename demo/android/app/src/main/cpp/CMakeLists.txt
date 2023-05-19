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

set(config_file "${snn-install-dir}/config.txt")
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

set(3rdparty-dir ${core-dir}/3rdparty/)
if (DEFINED SUPPORT_VULKAN)
    set(UVKC_SOURCE_ROOT ${3rdparty-dir}/uVkCompute/)
    set(VOLK_SOURCE_ROOT ${3rdparty-dir}/volk/)
endif()

set(OpenCV_DIR ${demo-dir}/OpenCV-android-sdk/sdk/native/jni)
set(ncnn_DIR ${demo-dir}/ncnn-20211208-android/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

set(sources
    native-lib.cpp
    native-lib-test.cpp
    appContext.cpp
    frameviz.cpp
    processor.cpp
    upload2GpuProcessor.cpp
    nativeFrameProvider.cpp
    mobileNetV2Processor.cpp
    resnet18Processor.cpp
    spatialDenoiser.cpp
    styleTransferProcessor.cpp
    yolov3Processor.cpp
    engine.cpp
    ${demo-dir}/common/matutil.cpp
    ${demo-dir}/common/testutil.cpp
    ${demo-dir}/common/processingloop.cpp
    ${demo-dir}/common/modelInference.cpp
    ${demo-dir}/common/inferenceProcessor.cpp
    ${demo-dir}/common/shaderUnitTest.cpp
)
if (DEFINED SUPPORT_GL)
    set(sources
        ${sources}
        glFrameviz.cpp
        glGpuFrameImage.cpp
        glCpuFrameImage.cpp
        genericClassifierProcessorGL.cpp
        spatialDenoiserGL.cpp
        styleTransferProcessorGL.cpp
        yolov3ProcessorGL.cpp
    )
endif()
if (DEFINED SUPPORT_VULKAN)
    set(sources
        ${sources}
        vulkanFrameViz.cpp
        vulkanGpuFrameImage.cpp
        genericModelProcessorVulkan.cpp
        mobileNetV2ProcessorVulkan.cpp
        resnet18ProcessorVulkan.cpp
        spatialDenoiserVulkan.cpp
        styleTransferProcessorVulkan.cpp
        yolov3ProcessorVulkan.cpp
        vulkan/error.cpp
        vulkan/strings.cpp
        vulkan/androidWindow.cpp
        vulkan/physicalDevice.cpp
        vulkan/instance.cpp
        vulkan/vulkanApp.cpp
        vulkan/vulkanAppContext.cpp
        vulkan/vulkanLib.cpp
        vulkan/bBoxes.cpp
    )
endif()

add_library(native-lib SHARED
    ${sources}
)

set(headers-3rdparty
    ${core-dir}/3rdparty/glm/include/
    ${core-dir}/3rdparty/libyuv/include/
    ${core-dir}/3rdparty/eigen-3.4.0/
    ${core-dir}/3rdparty/picojson/include/
    ${core-dir}/3rdparty/stb_image/include/
    ${core-dir}/3rdparty/opencv/android/include/
    ${core-dir}/3rdparty/readerwriterqueue/
)
if (DEFINED SUPPORT_GL)
    set(headers-3rdparty
        ${headers-3rdparty}
        ${core-dir}/3rdparty/glad/include/
    )
endif()

target_include_directories(native-lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(native-lib PRIVATE ${demo-dir}/common)
target_include_directories(native-lib PRIVATE ${snn-install-dir}/includes/inc ${snn-install-dir}/includes/src ${snn-install-dir}/includes/ ${headers-3rdparty})
target_include_directories(native-lib PRIVATE ${demo-dir}/common)
target_include_directories(native-lib PRIVATE ${demo-dir}/ncnn-20211208/src)
if (DEFINED SUPPORT_VULKAN)
    target_include_directories(native-lib PRIVATE ${UVKC_SOURCE_ROOT}/third_party/abseil-cpp)
    target_include_directories(native-lib PRIVATE ${UVKC_SOURCE_ROOT}/third_party/benchmark/include)
    target_include_directories(native-lib PRIVATE ${UVKC_SOURCE_ROOT})
    target_include_directories(native-lib PRIVATE ${VOLK_SOURCE_ROOT})
endif()

target_compile_options(native-lib PRIVATE -D__ANDROID__ -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-private-field -Wno-sign-compare -Wno-missing-field-initializers)
set_source_files_properties(${demo-dir}/common/shaderUnitTest.cpp PROPERTIES COMPILE_FLAGS "-Wno-unused-parameter -Wno-sometimes-uninitialized")
if (DEFINED SUPPORT_VULKAN)
    target_compile_options(native-lib PRIVATE -DVK_USE_PLATFORM_ANDROID_KHR)
endif()

if (DEFINED ENV{SNN_PROFILING})
    target_compile_options(native-lib PRIVATE -DPROFILING)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(native-lib PRIVATE -DPROFILING)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(native-lib PRIVATE -D_DEBUG -g)
    if (DEFINED SUPPORT_VULKAN)
        target_compile_options(native-lib PRIVATE -DVKB_DEBUG)
        target_compile_options(native-lib PRIVATE -DVKB_VALIDATION_LAYERS)
    endif()        
endif()

# Export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

target_link_libraries(native-lib
    android
    log
    ${snn-install-dir}/lib/arm64-v8a/libsnn_core.so
    ${OpenCV_LIBS}
    ncnn
)

if (DEFINED SUPPORT_VULKAN)
    execute_process (
        COMMAND bash -c "glslc -o ${demo-dir}/android/app/src/main/appAssets/shaders/quad.vert.spv ${demo-dir}/android/app/src/main/shaders/quad.vert > glslc.log 2>&1"
        COMMAND bash -c "glslc -o ${demo-dir}/android/app/src/main/appAssets/shaders/quad.frag.spv ${demo-dir}/android/app/src/main/shaders/quad.frag >> glslc.log 2>&1"
        OUTPUT_VARIABLE glslc_result
    )
    message(STATUS, ${glslc_result})
endif()