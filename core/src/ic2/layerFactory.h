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
#pragma once
#include "modelparser.h"
#include "genericlayer.h"
#include <string>

// This file contains set of classes amd macro to create layer classes

#define DECLARE_LAYER(layer) \
    snn::dp::GenericModelLayer* layer##Creator(snn::dp::ModelParser& parser, int i, bool) { \
        snn::dp::layer##Desc desc; \
        desc.parse(parser, i); \
        return new snn::dp::layer##Layer(std::move(desc)); \
    }

#define DECLARE_LAYER_GL_CLASS(layer) \
    namespace snn { \
    namespace dp { \
    class layer##LayerGl : public layer##Layer { \
    public: \
        layer##LayerGl(layer##Desc&& d): layer##Layer(std::move(d)) {} \
        virtual ~layer##LayerGl() = default; \
    protected: \
        InferencePassesSptr createFS(const LayerGenOptions&) const override; \
        InferencePassesSptr createCS(const LayerGenOptions&) const override; \
    }; \
    } \
    }

#define DECLARE_LAYER_VULKAN_CLASS(layer) \
    namespace snn { \
    namespace dp { \
    class layer##LayerVulkan : public layer##Layer { \
    public: \
        layer##LayerVulkan(layer##Desc&& d): layer##Layer(std::move(d)) {} \
        virtual ~layer##LayerVulkan() = default; \
    protected: \
        InferencePassesSptr createFS(const LayerGenOptions&) const override { SNN_RIP("Not implemented !"); } \
        InferencePassesSptr createCS(const LayerGenOptions&) const override; \
    }; \
    } \
    }

#define DECLARE_LAYER_VULKAN_CLASS_NOT_IMPL(layer) \
    namespace snn { \
    namespace dp { \
    class layer##LayerVulkan : public ShaderLayer { \
    public: \
        layer##LayerVulkan(layer##Desc&& d): ShaderLayer(std::move(d)) { SNN_RIP("Not implemented !"); } \
        virtual ~layer##LayerVulkan() = default; \
    protected: \
        InferencePassesSptr createFS(const LayerGenOptions&) const override { SNN_RIP("Not implemented !"); } \
        InferencePassesSptr createCS(const LayerGenOptions&) const override { SNN_RIP("Not implemented !"); } \
    }; \
    } \
    }

#ifdef SUPPORT_GL
#ifdef SUPPORT_VULKAN

// Gl and Vulkan
#define DECLARE_SHADER_LAYER(layer) \
    snn::dp::GenericModelLayer* layer##Creator(snn::dp::ModelParser& parser, int i, bool useVulkan) { \
        snn::dp::layer##Desc desc; \
        desc.parse(parser, i); \
        if (useVulkan) { \
            return new snn::dp::layer##LayerVulkan(std::move(desc)); \
        } else { \
            return new snn::dp::layer##LayerGl(std::move(desc)); \
        } \
    }

#else

// Gl only
#define DECLARE_SHADER_LAYER(layer) \
    snn::dp::GenericModelLayer* layer##Creator(snn::dp::ModelParser& parser, int i, bool) { \
        snn::dp::layer##Desc desc; \
        desc.parse(parser, i); \
        return new snn::dp::layer##LayerGl(std::move(desc)); \
    }

#endif // SUPPORT_VULKAN
#else
#ifdef SUPPORT_VULKAN

// Vulkan only
#define DECLARE_SHADER_LAYER(layer) \
    snn::dp::GenericModelLayer* layer##Creator(snn::dp::ModelParser& parser, int i, bool) { \
        snn::dp::layer##Desc desc; \
        desc.parse(parser, i); \
        return new snn::dp::layer##LayerVulkan(std::move(desc)); \
    }

#endif // SUPPORT_VULKAN
#endif // SUPPORT_GL

#define REGISTER_LAYER(layer) registerLayer(std::string(#layer), layer##Creator)

namespace snn {
namespace dp { // short for Dynamic Pipeline

typedef GenericModelLayer* (*LayerCreator)(ModelParser& parser, int i, bool useVulkan);

void initLayerRegisty();

void registerLayer(const std::string& layerName, LayerCreator creator);

GenericModelLayer* createLayerInstance(std::string layerName, ModelParser& parser, int i, bool useVulkan);

// Used to create shader layer classes, passing specific desc parameter
#define DECLARE_SHADER_LAYER1(layer) \
    struct layer##Desc; \
    GenericModelLayer* layer##Creator1(layer##Desc && desc, bool useVulkan);

DECLARE_SHADER_LAYER1(InputLayer);
DECLARE_SHADER_LAYER1(Conv2D);
DECLARE_SHADER_LAYER1(Conv2DTranspose);
DECLARE_SHADER_LAYER1(Subpixel);
DECLARE_SHADER_LAYER1(Concatenate);
DECLARE_SHADER_LAYER1(Calculate);
DECLARE_SHADER_LAYER1(UpSampling2D);
DECLARE_SHADER_LAYER1(Add);
DECLARE_SHADER_LAYER1(Activation);
DECLARE_SHADER_LAYER1(SeparableConv2D);
DECLARE_SHADER_LAYER1(Dense);
DECLARE_SHADER_LAYER1(MaxPooling2D);
DECLARE_SHADER_LAYER1(AveragePooling2D);
DECLARE_SHADER_LAYER1(AdaptiveAvgPool2d);
DECLARE_SHADER_LAYER1(Flatten);
DECLARE_SHADER_LAYER1(Pad);
DECLARE_SHADER_LAYER1(BatchNormalization);
DECLARE_SHADER_LAYER1(InstanceNorm);
DECLARE_SHADER_LAYER1(YOLO);
DECLARE_SHADER_LAYER1(Unary);

} // namespace dp
} // namespace snn
