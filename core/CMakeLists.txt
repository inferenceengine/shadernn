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

project(snn_core)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(config_file "${CMAKE_CURRENT_SOURCE_DIR}/config.txt")
if(NOT EXISTS ${config_file})
    message(FATAL_ERROR "${config_file} file not found. Run config.sh script")
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
    message(FATAL_ERROR "Neither OpenGL nor Vulkan is supported! Run config.sh script")
endif()

if (DEFINED ENV{SNN_PROFILING})
    add_definitions( -DPROFILING )
endif()
if (DEFINED ENV{SNN_DUMP_RESULTS_TXT})
    add_definitions( -DDUMP_RESULTS_TXT )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(root-dir ${CMAKE_CURRENT_SOURCE_DIR}/../)

# Headers
file(GLOB_RECURSE public_headers 
    inc/snn/*.h 
    inc/*.h
)

set(sources
    src/pch.cpp
    src/utils.cpp
    src/colorUtils.cpp
    src/image.cpp
    src/imageTexture.cpp
    src/contextFactory.cpp
    src/imageTextureFactory.cpp
    src/ic2/conv2d.cpp
    src/ic2/core.cpp
    src/ic2/layerFactory.cpp
    src/ic2/backendBuilder.cpp
    src/ic2/dp.cpp
    src/ic2/genericlayer.cpp
    src/ic2/separableconvolution.cpp
    src/ic2/modelparser.cpp
    src/ic2/maxpool2d.cpp
    src/ic2/avgpool2d.cpp
    src/ic2/denselayer.cpp
    src/ic2/flattenlayer.cpp
    src/ic2/yololayer.cpp
    src/ic2/padlayer.cpp
)
if (DEFINED SUPPORT_GL)
    set(sources_gl
        src/framebuffer.cpp
        src/glUtils.cpp
        src/imageTextureGL.cpp
        src/glContext.cpp
        src/glImageAccessor.cpp
        src/ic2/addlayerGL.cpp
        src/ic2/deconv2dGL.cpp
        src/ic2/conv2dGL.cpp
        src/ic2/subpixelmergeGL.cpp
        src/ic2/separableconvolutionGL.cpp
        src/ic2/concatenationGL.cpp
        src/ic2/calculationGL.cpp
        src/ic2/upsampling2dGL.cpp
        src/ic2/maxpool2dGL.cpp
        src/ic2/avgpool2dGL.cpp
        src/ic2/adaptiveavgpool2dGL.cpp
        src/ic2/denselayerGL.cpp
        src/ic2/flattenlayerGL.cpp
        src/ic2/instancenormGL.cpp
        src/ic2/batchnormGL.cpp
        src/ic2/padlayerGL.cpp
        src/ic2/activationGL.cpp
        src/ic2/openGLBackend.cpp
        src/ic2/openGLRenderpass.cpp
        src/ic2/unaryGL.cpp
    )
    set(sources
        ${sources}
        ${sources_gl}
    )
endif()
if (DEFINED SUPPORT_VULKAN)
    set(sources_vulkan
        src/imageTextureVulkan.cpp
        src/uvkcUtils.cpp
        src/vkUtils.cpp   
        src/vulkanContext.cpp
        src/vulkanImageAccessor.cpp
        src/vulkanImageTransformShaderOp.cpp
        src/vulkanImageResizeOp.cpp
        src/ic2/addlayerVulkan.cpp
        src/ic2/conv2dVulkan.cpp
        src/ic2/subpixelmergeVulkan.cpp
        src/ic2/separableconvolutionVulkan.cpp
        src/ic2/concatenationVulkan.cpp
        src/ic2/upsampling2dVulkan.cpp
        src/ic2/maxpool2dVulkan.cpp
        src/ic2/avgpool2dVulkan.cpp
        src/ic2/denselayerVulkan.cpp
        src/ic2/flattenlayerVulkan.cpp
        src/ic2/instancenormVulkan.cpp
        src/ic2/batchnormVulkan.cpp
        src/ic2/padlayerVulkan.cpp
        src/ic2/activationVulkan.cpp
        src/ic2/vulkanBackend.cpp
        src/ic2/vulkanRenderpass.cpp
        src/ic2/unaryVulkan.cpp
    )
    set(sources
        ${sources}
        ${sources_vulkan}
    )
endif()

set(snn_3rdparty_sources
    3rdparty/stb_image/src/stb_image.cpp
    3rdparty/stb_image/src/stb_image_write.cpp
)
if (DEFINED SUPPORT_GL)
    set(snn_3rdparty_sources
        ${snn_3rdparty_sources}
        3rdparty/glad/src/glad.c
    )
endif()

set(snn_3rdparty_headers
    3rdparty/glm/include/
    3rdparty/libyuv/include/
    3rdparty/eigen-3.4.0/
    3rdparty/picojson/include/
    3rdparty/stb_image/include/
)
if (DEFINED SUPPORT_GL)
    set(snn_3rdparty_headers
        ${snn_3rdparty_headers}
        3rdparty/glad/include/
    )
endif()

if (DEFINED SUPPORT_VULKAN)
    set(UVKC_SOURCE_ROOT ${root-dir}/core/3rdparty/uVkCompute/)
    message(STATUS "cmake UVKC_SOURCE_ROOT = ${UVKC_SOURCE_ROOT}")

    set(VOLK_SOURCE_ROOT ${root-dir}/core/3rdparty/volk/)

    set(WORK_GROUP_X 4)
    set(WORK_GROUP_Y 8)
    set(WORK_GROUP_Z 4)

    set(VK_COMPILE_OPTIONS "-DWG_X=${WORK_GROUP_X} -DWG_Y=${WORK_GROUP_Y} -DWG_Z=${WORK_GROUP_Z}")
    set(VK_COMPILE_OPTIONS_FP16 "-DWG_X=${WORK_GROUP_X} -DWG_Y=${WORK_GROUP_Y} -DWG_Z=${WORK_GROUP_Z} -DFP16_PRECISION=1")

    execute_process (
        COMMAND bash -c "rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/*.spv"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_activation.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_activation.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_add.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_add.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_concat.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_concat.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_maxpool2d.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_maxpool2d.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_avgpool2d.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_avgpool2d.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_conv2d.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_conv2d.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_conv2d_1x1.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_conv2d_1x1.comp" 
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_batchnorm.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_batchnorm.comp"        
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_pad.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_pad.comp"        
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_flatten.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_flatten.comp" 
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_dense.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_dense.comp"            
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_instancenorm.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_instancenorm.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_depthwise.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_depthwise.comp"                
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_resize.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_resize.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_upsampling2d_bilinear.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_upsampling2d_bilinear.comp"  
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_upsampling2d_nearest.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_upsampling2d_nearest.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_subpixel.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_subpixel.comp"   
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_unary.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_unary.comp"   
        # Compile for FP16
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_activation_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_activation.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_add_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_add.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_concat_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_concat.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_maxpool2d_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_maxpool2d.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_avgpool2d_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_avgpool2d.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_conv2d_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_conv2d.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_conv2d_1x1_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_conv2d_1x1.comp" 
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_batchnorm_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_batchnorm.comp"        
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_pad_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_pad.comp"        
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_flatten_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_flatten.comp" 
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_dense_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_dense.comp"            
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_instancenorm_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_instancenorm.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_depthwise_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_depthwise.comp"                
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_resize_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_resize.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_upsampling2d_bilinear_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_upsampling2d_bilinear.comp"  
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_upsampling2d_nearest_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_upsampling2d_nearest.comp"
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS_FP16} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_subpixel_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_subpixel.comp"       
        COMMAND bash -c "glslc ${VK_COMPILE_OPTIONS} -o ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/shaders/shadertemplate_vk_unary_fp16.spv ${root-dir}/core/data/assets/shaders/shadertemplate_vk_unary.comp"   
        OUTPUT_VARIABLE outVar
    )
    message(STATUS, ${outVar})
endif() # SUPPORT_VULKAN

add_library(snn_core SHARED ${sources} ${public_headers} ${snn_3rdparty_sources})

target_compile_options(snn_core PUBLIC  -fPIC -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-private-field -Wno-sign-compare)
# Eliminating picojson compile error
set_source_files_properties(${root-dir}/core/src/ic2/modelparser.cpp PROPERTIES COMPILE_FLAGS "-Wno-uninitialized")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(snn_core PRIVATE -D_DEBUG -g)
    message(STATUS "cmake Build Type: Debug")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(snn_core PUBLIC inc/ src/)
target_include_directories(snn_core PUBLIC ${snn_3rdparty_headers})
if (DEFINED SUPPORT_VULKAN)
    target_include_directories(snn_core PRIVATE ${UVKC_SOURCE_ROOT}/third_party/abseil-cpp)
    target_include_directories(snn_core PRIVATE ${UVKC_SOURCE_ROOT}/third_party/benchmark/include)
    target_include_directories(snn_core PRIVATE ${UVKC_SOURCE_ROOT})
    target_include_directories(snn_core PRIVATE ${VOLK_SOURCE_ROOT})
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libyuv)
if (DEFINED SUPPORT_VULKAN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/uVkCompute)
endif()

if (DEFINED ANDROID_ABI)
    set(opencv-lib-dir ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/android/lib/${ANDROID_ABI})
    target_include_directories(snn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/android/include/)
    target_link_libraries(snn_core PUBLIC android log libyuv-static
                                    ${opencv-lib-dir}/libopencv_core.so 
                                    ${opencv-lib-dir}/libopencv_imgproc.so
                                    ${opencv-lib-dir}/libopencv_video.so)
    if (DEFINED SUPPORT_GL)
        target_link_libraries(snn_core PUBLIC EGL GLESv2 GLESv3)
    endif()
else()
    find_package(OpenCV 4 REQUIRED)
    
    # Assets
    include("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cmrc/CMakeRC.cmake")
    set(shader-dir data/assets/shaders)
    set(image-dir data/assets/images)
    set(assets
        ${image-dir}/rose_1.png
        ${image-dir}/cifar_test.png
        ${image-dir}/arduino.png
        ${image-dir}/test_image_unet_gray.png
        ${image-dir}/ant.png
        ${image-dir}/imagenet1.png
        ${image-dir}/coco1_416.png
    )
    if (DEFINED SUPPORT_GL)
        set(assets
            ${assets}
            ${shader-dir}/shadertemplate_cs_3x_deconv_RGBA.glsl
            ${shader-dir}/shadertemplate_cs_4x_deconv_2s_RGBA.glsl
            ${shader-dir}/shadertemplate_cs_4x_deconv_RGBA.glsl
            ${shader-dir}/shadertemplate_cs_activation.glsl
            ${shader-dir}/shadertemplate_cs_batchnorm.glsl
            ${shader-dir}/shadertemplate_cs_concat.glsl
            ${shader-dir}/shadertemplate_cs_dense.glsl
            ${shader-dir}/shadertemplate_cs_flattenlayer.glsl
            ${shader-dir}/shadertemplate_cs_instancenorm.glsl
            ${shader-dir}/shadertemplate_cs_pad.glsl
            ${shader-dir}/shadertemplate_fs_3x_deconv_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_4x_deconv_2s_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_4x_deconv_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_5x_deconv_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_activation_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_add_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_avgpooling2d.glsl
            ${shader-dir}/shadertemplate_fs_batchnorm_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_calculation.glsl
            ${shader-dir}/shadertemplate_fs_concat.glsl
            ${shader-dir}/shadertemplate_fs_conv2d_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_depthwise_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_maxpooling2d.glsl
            ${shader-dir}/shadertemplate_fs_pad_RGBA.glsl
            ${shader-dir}/shadertemplate_fs_subpixel.glsl
            ${shader-dir}/shadertemplate_fs_upsampling2d.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_add.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_avgpool2d.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_conv2d.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_conv2d_1x1.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_maxpool2d.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_separableconvolution.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_upsampling2d_bilinear.glsl
            ${shader-dir}/3rdparty/shadertemplate_cs_upsampling2d_nearest.glsl
            ${shader-dir}/shadertemplate_cs_unary.glsl   
        )
    endif()
    if (DEFINED SUPPORT_VULKAN)
        set(assets
            ${assets}
            ${shader-dir}/shadertemplate_vk_activation.spv
            ${shader-dir}/shadertemplate_vk_add.spv
            ${shader-dir}/shadertemplate_vk_concat.spv
            ${shader-dir}/shadertemplate_vk_maxpool2d.spv
            ${shader-dir}/shadertemplate_vk_avgpool2d.spv
            ${shader-dir}/shadertemplate_vk_conv2d.spv
            ${shader-dir}/shadertemplate_vk_conv2d_1x1.spv
            ${shader-dir}/shadertemplate_vk_batchnorm.spv
            ${shader-dir}/shadertemplate_vk_pad.spv
            ${shader-dir}/shadertemplate_vk_flatten.spv
            ${shader-dir}/shadertemplate_vk_dense.spv
            ${shader-dir}/shadertemplate_vk_instancenorm.spv
            ${shader-dir}/shadertemplate_vk_resize.spv
            ${shader-dir}/shadertemplate_vk_upsampling2d_bilinear.spv
            ${shader-dir}/shadertemplate_vk_upsampling2d_nearest.spv
            ${shader-dir}/shadertemplate_vk_depthwise.spv
            ${shader-dir}/shadertemplate_vk_subpixel.spv
            ${shader-dir}/shadertemplate_vk_unary.spv
            ${shader-dir}/shadertemplate_vk_activation_fp16.spv
            ${shader-dir}/shadertemplate_vk_add_fp16.spv
            ${shader-dir}/shadertemplate_vk_concat_fp16.spv
            ${shader-dir}/shadertemplate_vk_maxpool2d_fp16.spv
            ${shader-dir}/shadertemplate_vk_avgpool2d_fp16.spv
            ${shader-dir}/shadertemplate_vk_conv2d_fp16.spv
            ${shader-dir}/shadertemplate_vk_conv2d_1x1_fp16.spv
            ${shader-dir}/shadertemplate_vk_batchnorm_fp16.spv
            ${shader-dir}/shadertemplate_vk_pad_fp16.spv
            ${shader-dir}/shadertemplate_vk_flatten_fp16.spv
            ${shader-dir}/shadertemplate_vk_dense_fp16.spv
            ${shader-dir}/shadertemplate_vk_instancenorm_fp16.spv
            ${shader-dir}/shadertemplate_vk_resize_fp16.spv
            ${shader-dir}/shadertemplate_vk_upsampling2d_bilinear_fp16.spv
            ${shader-dir}/shadertemplate_vk_upsampling2d_nearest_fp16.spv
            ${shader-dir}/shadertemplate_vk_depthwise_fp16.spv
            ${shader-dir}/shadertemplate_vk_subpixel_fp16.spv
            ${shader-dir}/shadertemplate_vk_unary_fp16.spv
        )
    endif()
    cmrc_add_resource_library(snn-assets ALIAS snn::rc NAMESPACE snn ${assets})
    target_compile_options(snn-assets PRIVATE -fPIC)
    
    target_include_directories(snn_core PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(snn_core PUBLIC snn-assets stdc++fs yuv jpeg)
endif()

if (DEFINED SUPPORT_VULKAN)
    target_link_libraries(snn_core PUBLIC 
        absl_strings 
        absl_base 
        absl_flags 
        absl_flags_parse 
        absl_strings_internal 
        absl_str_format_internal 
        absl_status 
        absl_statusor 
        absl_strings 
        absl_cord_internal 
        absl_cordz_info 
        absl_base 
        absl_cord 
        absl_cordz_functions 
        absl_cordz_handle 
        absl_malloc_internal 
        absl_time 
        absl_time_zone 
        absl_civil_time
        absl_log_severity 
        absl_raw_logging_internal 
        absl_throw_delegate 
        absl_spinlock_wait 
        absl_stacktrace 
        absl_debugging_internal 
        absl_synchronization 
        absl_graphcycles_internal 
        absl_base 
        absl_symbolize 
        absl_debugging_internal 
        absl_demangle_internal 
        absl_exponential_biased 
        absl_int128 
        uvkc_base_log 
        uvkc_base_file 
        uvkc_base_dynamic_library 
        uvkc_vulkan_image 
        uvkc_vulkan_buffer 
        uvkc_vulkan_driver 
        uvkc_vulkan_timestamp_query_pool
        uvkc_vulkan_device 
        uvkc_vulkan_command_buffer 
        uvkc_vulkan_descriptor_pool 
        uvkc_vulkan_shader_module 
        uvkc_vulkan_status_util 
        uvkc_vulkan_dynamic_symbols 
        uvkc_vulkan_pipeline 
        uvkc_vulkan_pipeline_util 
        benchmark 
        uvkc_benchmark_status_util 
        uvkc_benchmark_fp16_util 
        uvkc_benchmark_main 
        uvkc_benchmark_vulkan_buffer_util 
        uvkc_benchmark_vulkan_context 
        uvkc_benchmark_vulkan_image_util
    )
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc DESTINATION ${root-dir}/snn-core-install/includes/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src DESTINATION ${root-dir}/snn-core-install/includes/ FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src DESTINATION ${root-dir}/snn-core-install/includes/ FILES_MATCHING PATTERN "*.hpp")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.txt DESTINATION ${root-dir}/snn-core-install)

if (DEFINED ANDROID_ABI)
    install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsnn_core.so DESTINATION ${root-dir}/snn-core-install/lib/${ANDROID_ABI})
else()
    install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsnn_core.so DESTINATION ${root-dir}/snn-core-install/lib/linux_x86_64/)
endif()
