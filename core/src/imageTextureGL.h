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
#include "snn/imageTexture.h"
#include "glUtils.h"

static constexpr const char* RESIZE_NEAREST_CS_ASSET_NAME  = "shaders/3rdparty/shadertemplate_cs_upsampling2d_nearest.glsl";
static constexpr const char* RESIZE_BILINEAR_CS_ASSET_NAME = "shaders/3rdparty/shadertemplate_cs_upsampling2d_bilinear.glsl";

namespace snn {

// This class object holds OpenGL textures and provides access to them either from GPU or from CPU
class ImageTextureGL : public ImageTexture {
public:
    // Default constructor
    ImageTextureGL();

    // Constructor from a pixel buffer
    // params:
    //  dims - dimensions
    //  format - color format
    //  buffer - image pixel buffer
    //  name - optional image name
    ImageTextureGL(const std::array<uint32_t, 4>& dims, ColorFormat format, const void* buffer = NULL, const std::string& name = "");

    // Constructor from an image file
    // params:
    //  fileName - image file name
    ImageTextureGL(const std::string& fileName);

    virtual ~ImageTextureGL() = default;

    // Casts an ImageTexture reference  to ImageTextureGL reference
    // params:
    //  src - source reference to ImageTexture
    // returns:
    //  target reference to ImageTextureGL
    static ImageTextureGL& cast(ImageTexture& src) {
        SNN_ASSERT(src.getType() == GpuBackendType::GL);
        return static_cast<ImageTextureGL&>(src);
    }

    // Attaches a content of another ImageTexture object
    // params:
    //  src - source object
    virtual void attach(ImageTexture *src) override;

    // Attaches another texture
    // params:
    //  images - source texture
    virtual void attach(const std::vector<const GpuImageHandle*>& images) override;

    // Checks if current objects holds a texture on GPU
    // return:
    //  true if yes; false if no
    virtual bool isValid() const override {
        SNN_LOGV("OpenGL Texture:%zu, id: %d",
            _textures.size(), _textures.size() > 0 ? _textures[0].id() : 0);
        return _textures.size() > 0 && _textures[0].id() > 0;
    }

    // Allocates empty GPU texture with the same dimensions and color format
    virtual void resetTexture() override;

    // Allocates empty GPU texture
    // params:
    //  dims - dimensions
    //  format - color format
    //  name - optional image name
    virtual void resetTexture(const std::array<uint32_t, 4>& dims, ColorFormat format, const std::string& name = "") override;

    // Resizes the texture on GPU and optionally normalizes it.
    // params:
    //  xScale - horizontal scale factor
    //  yScale - vertical scale factor
    //  means - mean values for 4 channels. Used for normalization.
    //  norms - normalization values for 4 channels (multipliers). Used for normalization.
    //  linearFilter - true if using linear filter. false if using nearest-neighbor filter.
    //  cf - not used
    // return:
    // true if resizing was successful; false if not.
    virtual bool resize(float xScale, float yScale, const std::array<float, 4>& means, const std::array<float, 4>& norms, bool linearFilter = true,
        ColorFormat cf = ColorFormat::NONE) override;

    // Downloads textures from device to host
    virtual void download() override;

    // Uploads textures from host to device
    virtual void upload() override;

    // Gets a number of held textures
    virtual size_t getNumTextures() const override {
        return _textures.size();
    }

    // Returns texture information in human-readable format. Used for debugging.
    virtual std::string getTextureInfo() const override;

    // Returns detailed texture information in human-readable format. Used for debugging.
    virtual std::string getTextureInfo2() const override;

    // Returns an underlying object of gl::TextureObject class
    // params:
    //  index of texture to return
    // returns:
    //  Pointer to an object of gl::TextureObject class
    gl::TextureObject* texture(size_t index = 0);

private:
    // Resizes a texture on GPU and optionally normalizes it.
    // params:
    //  inputTex - input texture object
    //  outputTex - output texture object
    //  xScale - horizontal scale factor
    //  yScale - vertical scale factor
    //  means - mean values for 4 channels. Used for normalization.
    //  norms - normalization values for 4 channels (multipliers). Used for normalization.
    //  linearFilter - true if using linear filter. false if using nearest-neighbor filter.
    // return:
    // true if resizing was successful; false if not.
    bool resizeTexture(gl::TextureObject& inputTex, gl::TextureObject& outputTex, float xScale, float yScale, const std::array<float, 4>& means,
                       const std::array<float, 4>& norms, bool linearFilter = true);

    // Array of texture objects
    FixedSizeArray<gl::TextureObject> _textures;
};

typedef ImageTextureTypeCheck<GpuBackendType::GL> ImageTextureGLTypeCheck;

typedef PolyArrayAccessor<ImageTextureGL, ImageTexture, ImageTextureGLTypeCheck> ImageTextureGLArrayAccessor;

typedef PolyArray<ImageTextureGL, ImageTexture, ImageTextureGLTypeCheck> ImageTextureGLArray;

} // namespace snn
