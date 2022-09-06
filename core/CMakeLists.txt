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

add_definitions( -DPROFILING )

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
    src/image.cpp
    src/glUtils.cpp
    src/fence.cpp
    src/imageTexture.cpp
    src/inferenceProcessor.cpp
    src/ic2/addlayer.cpp
    src/ic2/deconv2d.cpp
    src/ic2/conv2d.cpp
    src/ic2/core.h
    src/ic2/core.cpp
    src/ic2/dp.h
    src/ic2/dp.cpp
    src/ic2/inferencegraph.h
	src/ic2/activation.h
	src/ic2/adaptiveavgpool2d.h
	src/ic2/addlayer.h
	src/ic2/avgpool2d.h
	src/ic2/batchnorm.h
	src/ic2/calculation.h
	src/ic2/concatenation.h
	src/ic2/conv2d.h
	src/ic2/core.h
	src/ic2/cpulayer.h
	src/ic2/deconv2d.h
	src/ic2/denselayer.h
	src/ic2/dp.h
	src/ic2/flattenlayer.h
	src/ic2/genericlayer.h
	src/ic2/inputlayer.h
	src/ic2/instancenorm.h
	src/ic2/layeroption.h
	src/ic2/maxpool2d.h
	src/ic2/padlayer.h
	src/ic2/separableconvolution.h
	src/ic2/subpixelmerge.h
	src/ic2/upsampling2d.h
	src/ic2/yololayer.h
    src/ic2/subpixelmerge.cpp
    src/ic2/separableconvolution.cpp
    src/ic2/concatenation.cpp
    src/ic2/calculation.cpp
    src/ic2/modelparser.cpp
    src/ic2/upsampling2d.cpp
    src/ic2/maxpool2d.cpp
    src/ic2/avgpool2d.cpp
    src/ic2/adaptiveavgpool2d.cpp
    src/ic2/denselayer.cpp
    src/ic2/flattenlayer.cpp
    src/ic2/yololayer.cpp
    src/ic2/instancenorm.cpp
    src/ic2/batchnorm.cpp
    src/ic2/padlayer.cpp
    src/ic2/activation.cpp    
)

set(snn_3rdparty_sources
    3rdparty/glad/src/glad.c
    3rdparty/stb_image/src/stb_image.cpp
    3rdparty/stb_image/src/stb_image_write.cpp
)

set(snn_3rdparty_headers
    3rdparty/glad/include/
    3rdparty/glm/include/
    3rdparty/libyuv/include/
    3rdparty/eigen-3.4.0/
    3rdparty/picojson/include/
    3rdparty/stb_image/include/
)

add_library(snn_core SHARED ${sources} ${public_headers} ${snn_3rdparty_sources})

target_compile_options(snn_core PUBLIC -w -Wall -Wextra)

target_include_directories(snn_core PUBLIC inc/)
target_include_directories(snn_core PUBLIC ${snn_3rdparty_headers})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libyuv)

if (DEFINED ANDROID_ABI)
    set(opencv-lib-dir ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/android/lib/${ANDROID_ABI})
    target_include_directories(snn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/android/include/)
    target_link_libraries(snn_core PUBLIC android log EGL GLESv2 GLESv3 libyuv-static
                                    ${opencv-lib-dir}/libopencv_core.so 
                                    ${opencv-lib-dir}/libopencv_imgproc.so
                                    ${opencv-lib-dir}/libopencv_video.so)
else()
    find_package(OpenGL REQUIRED)
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
    )
    cmrc_add_resource_library(snn-assets ALIAS snn::rc NAMESPACE snn ${assets})
    target_compile_options(snn-assets PRIVATE -fPIC)
    
    target_include_directories(snn_core PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(snn_core PUBLIC OpenGL::EGL OpenGL::OpenGL ${OpenCV_LIBS} snn-assets ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libyuv.so stdc++fs)
endif()


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc DESTINATION ${root-dir}/snn-core-install/includes/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src DESTINATION ${root-dir}/snn-core-install/includes/ FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src DESTINATION ${root-dir}/snn-core-install/includes/ FILES_MATCHING PATTERN "*.hpp")

if (DEFINED ANDROID_ABI)
    install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsnn_core.so DESTINATION ${root-dir}/snn-core-install/lib/${ANDROID_ABI})
else()
    install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsnn_core.so DESTINATION ${root-dir}/snn-core-install/lib/linux_x86_64/)
endif()
