/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pch.h"
#include "snn/utils.h"
#include "layerFactory.h"
#include "activation.h"
#include "adaptiveavgpool2d.h"
#include "addlayer.h"
#include "avgpool2d.h"
#include "batchnorm.h"
#include "calculation.h"
#include "concatenation.h"
#include "conv2d.h"
#include "cpulayer.h"
#include "denselayer.h"
#include "deconv2d.h"
#include "flattenlayer.h"
#include "inputlayer.h"
#include "instancenorm.h"
#include "maxpool2d.h"
#include "padlayer.h"
#include "separableconvolution.h"
#include "subpixelmerge.h"
#include "upsampling2d.h"
#include "yololayer.h"
#include "unary.h"
#include <string>
#include <unordered_map>

#ifdef SUPPORT_GL
    DECLARE_LAYER_GL_CLASS(Add);
    DECLARE_LAYER_GL_CLASS(Activation);
    #include "adaptiveavgpool2dGL.h"
    #include "avgpool2dGL.h"
    #include "batchnormGL.h"
    #include "calculationGL.h"
    #include "concatenationGL.h"
    #include "conv2dGL.h"
    #include "deconv2dGL.h"
    DECLARE_LAYER_GL_CLASS(Dense);
    DECLARE_LAYER_GL_CLASS(Flatten);
    DECLARE_LAYER_GL_CLASS(InstanceNorm);
    #include "maxpool2dGL.h"
    #include "padlayerGL.h"
    #include "separableconvolutionGL.h"
    DECLARE_LAYER_GL_CLASS(Subpixel);
    DECLARE_LAYER_GL_CLASS(Unary);
    #include "upsampling2dGL.h"
#endif

#ifdef SUPPORT_VULKAN
    DECLARE_LAYER_VULKAN_CLASS(Add);
    DECLARE_LAYER_VULKAN_CLASS(Activation);
    DECLARE_LAYER_VULKAN_CLASS(AveragePooling2D);
    DECLARE_LAYER_VULKAN_CLASS_NOT_IMPL(AdaptiveAvgPool2d);
    DECLARE_LAYER_VULKAN_CLASS(BatchNormalization);
    DECLARE_LAYER_VULKAN_CLASS(Concatenate);
    DECLARE_LAYER_VULKAN_CLASS_NOT_IMPL(Calculate);
    DECLARE_LAYER_VULKAN_CLASS(Conv2D);
    DECLARE_LAYER_VULKAN_CLASS_NOT_IMPL(Conv2DTranspose);
    DECLARE_LAYER_VULKAN_CLASS(Dense);
    DECLARE_LAYER_VULKAN_CLASS(Flatten);
    DECLARE_LAYER_VULKAN_CLASS(InstanceNorm);
    DECLARE_LAYER_VULKAN_CLASS(MaxPooling2D);
    DECLARE_LAYER_VULKAN_CLASS(Pad);
    DECLARE_LAYER_VULKAN_CLASS(SeparableConv2D);
    DECLARE_LAYER_VULKAN_CLASS(Subpixel);
    DECLARE_LAYER_VULKAN_CLASS(Unary);
    DECLARE_LAYER_VULKAN_CLASS(UpSampling2D);
#endif

DECLARE_LAYER(InputLayer);
DECLARE_SHADER_LAYER(Conv2D);
DECLARE_SHADER_LAYER(Conv2DTranspose);
DECLARE_SHADER_LAYER(Subpixel);
DECLARE_SHADER_LAYER(Concatenate);
DECLARE_SHADER_LAYER(Calculate);
DECLARE_SHADER_LAYER(UpSampling2D);
DECLARE_SHADER_LAYER(Add);
DECLARE_SHADER_LAYER(SeparableConv2D);
DECLARE_SHADER_LAYER(Dense);
DECLARE_SHADER_LAYER(MaxPooling2D);
DECLARE_SHADER_LAYER(AveragePooling2D);
DECLARE_SHADER_LAYER(AdaptiveAvgPool2d);
DECLARE_SHADER_LAYER(Flatten);
DECLARE_SHADER_LAYER(Pad);
DECLARE_SHADER_LAYER(BatchNormalization);
DECLARE_SHADER_LAYER(InstanceNorm);
DECLARE_LAYER(YOLO);
DECLARE_SHADER_LAYER(Unary);

static std::unordered_map<std::string, snn::dp::LayerCreator> LayerRegistryDict;

namespace snn {
namespace dp {

void initLayerRegisty() {
    REGISTER_LAYER(InputLayer);
    REGISTER_LAYER(Conv2D);
    REGISTER_LAYER(Conv2DTranspose);
    REGISTER_LAYER(Subpixel);
    REGISTER_LAYER(Concatenate);
    REGISTER_LAYER(Calculate);
    REGISTER_LAYER(UpSampling2D);
    REGISTER_LAYER(Add);
    REGISTER_LAYER(SeparableConv2D);
    REGISTER_LAYER(Dense);
    REGISTER_LAYER(MaxPooling2D);
    REGISTER_LAYER(AveragePooling2D);
    REGISTER_LAYER(AdaptiveAvgPool2d);
    REGISTER_LAYER(Flatten);
    REGISTER_LAYER(Pad);
    REGISTER_LAYER(BatchNormalization);
    REGISTER_LAYER(InstanceNorm);
    REGISTER_LAYER(YOLO);
    REGISTER_LAYER(Unary);
}

void registerLayer(const std::string& layerName, LayerCreator creator) {
    SNN_LOGD("register layer: %s", layerName.c_str());
    LayerRegistryDict.emplace(layerName, creator);
}

snn::dp::GenericModelLayer* createLayerInstance(std::string layerName, ModelParser& parser, int i, bool useVulkan) {
    GenericModelLayer* ret = NULL;
    if (layerName == "DepthwiseConv2D" || layerName == "Depthwise") {
        layerName = "SeparableConv2D";
    }
    if (layerName == "InstanceNormalization") {
        layerName = "InstanceNorm";
    }
    if (layerName == "ZeroPadding2D") {
        layerName = "Pad";
    }
    if (layerName == "subpixel" || layerName == "depth_to_space") {
        layerName = "Subpixel";
    }

    if (LayerRegistryDict.find(layerName) != LayerRegistryDict.end()) {
        auto func = LayerRegistryDict[layerName];
        SNN_LOGD("found layer: %s", layerName.c_str());
        ret = func(parser, i, useVulkan);
    } else {
        SNN_RIP("Not found layer: %s", layerName.c_str());
    }
    return ret;
}

// Used to create shader layer classes, passing specific desc parameter
#ifdef SUPPORT_GL
#ifdef SUPPORT_VULKAN

// Gl and Vulkan
#define DEFINE_SHADER_LAYER1(layer) \
    snn::dp::GenericModelLayer* layer##Creator1(snn::dp::layer##Desc && desc, bool useVulkan) { \
        if (useVulkan) { \
            return new snn::dp::layer##LayerVulkan(std::move(desc)); \
        } else { \
            return new snn::dp::layer##LayerGl(std::move(desc)); \
        } \
    }

#else

// Gl only
#define DEFINE_SHADER_LAYER1(layer) \
    snn::dp::GenericModelLayer* layer##Creator1(snn::dp::layer##Desc && desc, bool) { \
        return new snn::dp::layer##LayerGl(std::move(desc)); \
    }

#endif // SUPPORT_VULKAN
#else
#ifdef SUPPORT_VULKAN

// Vulkan only
#define DEFINE_SHADER_LAYER1(layer) \
    snn::dp::GenericModelLayer* layer##Creator1(snn::dp::layer##Desc && desc, bool) { \
        return new snn::dp::layer##LayerVulkan(std::move(desc)); \
    }

#endif // SUPPORT_VULKAN
#endif // SUPPORT_GL

DEFINE_SHADER_LAYER1(Conv2D);
DEFINE_SHADER_LAYER1(Conv2DTranspose);
DEFINE_SHADER_LAYER1(Subpixel);
DEFINE_SHADER_LAYER1(Concatenate);
DEFINE_SHADER_LAYER1(Calculate);
DEFINE_SHADER_LAYER1(UpSampling2D);
DEFINE_SHADER_LAYER1(Add);
DEFINE_SHADER_LAYER1(Activation);
DEFINE_SHADER_LAYER1(SeparableConv2D);
DEFINE_SHADER_LAYER1(Dense);
DEFINE_SHADER_LAYER1(MaxPooling2D);
DEFINE_SHADER_LAYER1(AveragePooling2D);
DEFINE_SHADER_LAYER1(AdaptiveAvgPool2d);
DEFINE_SHADER_LAYER1(Flatten);
DEFINE_SHADER_LAYER1(Pad);
DEFINE_SHADER_LAYER1(BatchNormalization);
DEFINE_SHADER_LAYER1(InstanceNorm);
DEFINE_SHADER_LAYER1(Unary);

} // namespace dp
} // namespace snn
