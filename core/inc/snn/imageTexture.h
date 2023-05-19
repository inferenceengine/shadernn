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
#include "snn/snn.h"
#include "snn/utils.h"
#include "snn/image.h"
#include "snn/color.h"
#include <string>
#include <memory>
#include <vector>
#include <array>

namespace snn {

typedef enum class Backend { Backend_CPU, Backend_GPU, NOT_DEFINED = 200 } Backend;

// This class describes an image, located on GPU or CPU
class ImageTexture {
protected:
    // Constructor
    // params:
    //  type - GPU backend type
    ImageTexture(GpuBackendType type)
        : _type(type)
        , _format(ColorFormat::NONE)
        , _dims{}
        , _backend(Backend::Backend_CPU)
    {}

    // Constructor from a pixel buffer
    // params:
    //  type - GPU backend type
    //  dims - dimensions
    //  format - color format
    //  buffer - image pixel buffer
    //  name - optional image name
    ImageTexture(GpuBackendType type, const std::array<uint32_t, 4>& dims, ColorFormat format, const void* buffer = NULL, const std::string& name = "");

    // Constructor from an image file
    // params:
    //  type - GPU backend type
    //  fileName - image file name
    ImageTexture(GpuBackendType type, const std::string& fileName);

    // Clears an image on CPU
    void resetImages();

public:
    virtual ~ImageTexture() = default;

    // Attaches a content of another ImageTexture object
    // params:
    //  src - source object
    // TODO: This API is unnecessary
    // We can just change a pointer in places where we use it
    virtual void attach(ImageTexture* /*src*/) {
    }

    // Attaches another GPU images
    // params:
    //  images - GPU images to attach
    virtual void attach(const std::vector<const GpuImageHandle*>& /*images*/) {
    }

    // Checks if an object of the class holds valid GPU image
    virtual bool isValid() const {
        return true;
    }

    // Resets CPU images
    // params:
    //  dims - dimensions
    //  format - color format
    //  buffer - image pixel buffer
    //  name - optional image name
    void reset(const std::array<uint32_t, 4>& dims, ColorFormat format, void* buffer = NULL, const std::string& name = "");

    // Allocates empty GPU images with the same dimensions and color format
    virtual void resetTexture()
    {}

    // Allocates empty GPU images
    virtual void resetTexture(const std::array<uint32_t, 4>& /*dims*/, ColorFormat /*format*/, const std::string& /*name*/ = "")
    {}

    // Loads CPU images from file
    // params:
    //  fileName - image file name
    //  fromBin - flag indicating that the image file is in raw dump format
    void loadFromFile(const std::string& fileName, bool fromBin = false);

    // Converts contained images to another format
    // params:
    //  format - new color format
    void convertFormat(snn::ColorFormat format);

    // Converts contained images to RGBA32F format and optionally normalizes
    // params:
    //  means - mean values for 4 channels. Used for normalization.
    // norms - norm values for 4 channels (multipliers). Used for normalization.
    void convertToRGBA32FAndNormalize(const std::vector<float>& means = {0.0f, 0.0f, 0.0f, 0.0f}, const std::vector<float>& norms = {1.0f, 1.0f, 1.0f, 1.0f});

    // Helper function
    // Loads GPU shader from file
    // params:
    //  path - path to the shader file
    // return:
    //  shader code as a string
    std::string loadShader(const char* path) {
        auto bytes = snn::loadEmbeddedAsset(path);
        return std::string(bytes.begin(), bytes.end());
    }

    // Resizes image on GPU and optionally normalizes it.
    // params:
    //  xScale - horizontal scale factor
    //  yScale - vertical scale factor
    //  means - mean values for 4 channels. Used for normalization.
    //  norms - normalization values for 4 channels (multipliers). Used for normalization.
    //  linearFilter - true if using linear filter. false if using nearest-neighbor filter.
    //  cf - optional color format to convert resized image
    // return:
    // true if resizing was successful; false if not.
    virtual bool resize(float xScale, float yScale, const std::array<float, 4>& means, const std::array<float, 4>& norms, bool linearFilter = true,
        ColorFormat cf = ColorFormat::NONE) {
        (void) xScale;
        (void) yScale;
        (void) means;
        (void) norms;
        (void) linearFilter;
        (void) cf;

        return 0;
    }

    // Gets image color format
    // params:
    //  index - index of an image plane
    // returns:
    //  color format
    ColorFormat format(size_t index = 0) const { (void)index; return _format; }

    // Gets image width
    // params:
    //  index - index of an image plane
    // returns:
    //  image width
    uint32_t width(size_t index = 0) const { return _images.width(index); }

    // Gets image height
    // params:
    //  index - index of an image plane
    // returns:
    //  image height
    uint32_t height(size_t index = 0
    ) const { return _images.height(index); }

    // Gets image depth
    // params:
    //  index - index of an image plane
    // returns:
    //  image depth
    uint32_t depth(size_t index = 0) const { return _images.depth(index); }

    // Gets image step
    // This is bits (not BYTES) from one pixel to next. Minimal valid value is pixel size.
    // params:
    //  index - index of an image plane
    // returns:
    //  image step
    uint32_t step(size_t index = 0) const { return _images.step(index); }

    // Gets image pitch
    // This is bytes from one row to next. Minimal valid value is (width * step) and aligned to alignment.
    // params:
    //  index - index of an image plane
    // returns:
    //  image slice size
    uint32_t pitch(size_t index = 0) const { return _images.pitch(index); }

    // Gets image slice size
    // This is bytes from one 2D slice to next. Minimal valid value is (pitch * height)
    // params:
    //  index - index of an image plane
    // returns:
    //  image depth
    uint32_t sliceSize(size_t index = 0) const { return _images.sliceSize(index); }

    // Gets image plane size in bytes
    // returns:
    //  image slice size
    uint32_t size() { return (uint32_t) _images.size(); }
    
    // Gets number of planes in the image
    // return:
    //  number of planes in the image
    uint32_t planes() { return (uint32_t) _images.planes(); }

    ImageDesc getDesc() const { return _images.getDesc(); }

    // Gets address of 3D image plane buffer
    // params:
    //  p - index of an image plane
    // return:
    //  address of a plane buffer
    uint8_t* plane(size_t p) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.plane(p);
    }

    // Gets address of 2D image slice buffer
    // params:
    //  p - index of an image plane
    //  z - z coordinate of the plane
    // return:
    //  address of a slice buffer
    uint8_t* slice(size_t p, size_t z) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.slice(p, z);
    }

    // Gets address of an image row buffer
    // params:
    //  p - index of an image plane
    //  y - y coordinate of the plane
    //  z - z coordinate of the plane
    // return:
    //  address of a row buffer
    uint8_t* row(size_t p, size_t y, size_t z = 0) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.row(p, y, z);
    }

    // Gets address of an image single pixel
    // params:
    //  p - index of an image plane
    //  x - x coordinate of the plane
    //  y - y coordinate of the plane
    //  z - z coordinate of the plane
    // return:
    //  address of a pixel
    uint8_t* at(size_t p, size_t x, size_t y = 0, size_t z = 0) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images.at(p, x, y, z);
    }

    // Gets contained raw image
    // return:
    //  const reference to an image object
    const snn::RawImage& getRawImage() {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        return _images;
    }

    // Gets supplemental float buffer
    // returns:
    //  const reference to a float buffer
    const std::vector<std::vector<float>>& getOutputMat() const {
        return outputMat;
    }

    // Sets supplemental float buffer
    // params:
    //  float buffer
    void setOutputMat(const std::vector<std::vector<float>>& outputMat_) {
        outputMat = outputMat_;
    }

    // Gets GPU backend type
    GpuBackendType getType() const { return _type; }

    // Gets the name
    const std::string& getName() const { return _name; }

    // Gets a number of GPU images
    // return
    //  Number of GPU images
    virtual size_t getNumTextures() const { return 0; }

    // Returns texture information in human-readable format. Used for debugging.
    virtual std::string getTextureInfo() const { return ""; }

    // Returns detailed texture information in human-readable format. Used for debugging.
    virtual std::string getTextureInfo2() const { return ""; }

    // Downloads image from device to host
    virtual void download() {}

    // Uploads image from host to device
    virtual void upload() {}

    // Gets images dimensions
    const std::array<uint32_t, 4>& getDims() const { return _dims; }

    // Gets color format
    ColorFormat getFormat() const { return _format; }

    // Save the image to bin format
    // params:
    //  filename - file name to save
    void saveToBIN(const std::string& filename) {
        if (_backend != Backend::Backend_CPU) {
            download();
        }
        _images.saveToBIN(filename);
    }

    // Helper method. Converts image to OpenCV format
    // params:
    //  ret - buffer tpo write to
    bool getCVMatData(uint8_t* ret);

    // Helper method. Prints the image content in human readable form.
    // params:
    //  fp - file object
    void prettyPrint(FILE* fp = stdout);

protected:
    GpuBackendType _type;
    std::string _name;
    ColorFormat _format = ColorFormat::NONE;
    std::array<uint32_t, 4> _dims; // Width, Height, Depth, Planes
    Backend _backend = Backend::Backend_CPU;
    ManagedRawImage _images;
    // To work with current CPU Flatten/Dense Layer. To be changed.
    std::vector<std::vector<float>> outputMat;
};

template<GpuBackendType TYPE>
struct ImageTextureTypeCheck {
    static bool checkType(const ImageTexture* tex) {
        return tex->getType() == TYPE;
    }
};

// Structures to support polymorphic array of ImageTexture objects
struct ImageTextureAllocator {
    ImageTextureAllocator(GpuContext* context_)
        : context(context_)
    {
        SNN_ASSERT(context_);
    }

    std::shared_ptr<ImageTexture>* allocate(size_t n);

    void deallocate(std::shared_ptr<ImageTexture>* ptr, size_t);

    GpuContext* context;
};

typedef PolyArrayAccessor<ImageTexture> ImageTextureArrayAccessor;

typedef PolyArray<ImageTexture, ImageTexture, NoCheck, ImageTextureAllocator> ImageTextureArray;

} // namespace snn
