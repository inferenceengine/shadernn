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

// This file contains data structures
// to hold CPU image

#pragma once

#include "snn/utils.h"
#include "color.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <type_traits>
#include <utility>
#include <stdlib.h>
#include <string.h>
#include <iostream>

namespace snn {
union FP16 {
    uint16_t u;
    struct {
        uint16_t mantissa : 10;
        uint16_t exponent : 5;
        uint16_t sign : 1;
    };
    float toFloat() const;
    static float toFloat(uint16_t u) { return ((FP16*) &u)->toFloat(); }
};

union FP32 {
    uint32_t u;
    float flt;
    struct {
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    };

    uint16_t toHalf() const {
        // https://gist.github.com/rygorous/2156668
        FP32 infty = {31 << 23};
        FP32 magic = {15 << 23};
        FP32 f32;
        f32.flt  = flt;
        FP16 o = {0};

        auto sign = f32.sign;
        f32.sign  = 0;

        // Based on ISPC reference code (with minor modifications)
        if (f32.exponent == 255) { // Inf or NaN (all exponent bits set)
            o.exponent = 31;
            o.mantissa = f32.mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
        } else                                     // (De)normalized number or zero
        {
            f32.u &= ~0xfff; // Make sure we don't get sticky bits
            f32.flt *= magic.flt;

            f32.u += 0x1000; // Rounding bias
            if (f32.u > infty.u) {
                f32.u = infty.u; // Clamp to signed infinity if overflowed
            }

            o.u = (uint16_t)(f32.u >> 13); // Take the bits!
        }

        o.sign = sign;
        return o.u;
    }

    static uint16_t toHalf(float f32) { return ((FP32*) &f32)->toHalf(); }
};

union Rgba8 {
    static const ColorFormat FORMAT = ColorFormat::RGBA8;

    struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };
    uint8_t u8[4];
    uint32_t u32;
};

union SRgba8 {
    static const ColorFormat FORMAT = ColorFormat::SRGB8_A8;

    struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };
    uint8_t u8[4];
    uint32_t u32;
};

union R8 {
    static const ColorFormat FORMAT = ColorFormat::R8;

    struct {
        uint8_t r;
    };
    uint8_t u8;
};

union Rgba32f {
    static const ColorFormat FORMAT = ColorFormat::RGBA32F;

    struct {
        float red;
        float green;
        float blue;
        float alpha;
    };
    float f32[4];

    Rgba32f operator+(float val) {
        Rgba32f retVal;
        retVal.red = this->red + val;
        retVal.green = this->green + val;
        retVal.blue = this->blue + val;
        retVal.alpha = this->alpha + val;
        return retVal;
    }

    Rgba32f operator-(float val) {
        Rgba32f retVal;
        retVal.red = this->red - val;
        retVal.green = this->green - val;
        retVal.blue = this->blue - val;
        retVal.alpha = this->alpha - val;
        return retVal;
    }

    Rgba32f operator*(float val) {
        Rgba32f retVal;
        retVal.red = this->red * val;
        retVal.green = this->green * val;
        retVal.blue = this->blue * val;
        retVal.alpha = this->alpha * val;
        return retVal;
    }

    Rgba32f operator/(float val) {
        Rgba32f retVal;
        retVal.red = this->red / val;
        retVal.green = this->green / val;
        retVal.blue = this->blue / val;
        retVal.alpha = this->alpha / val;
        return retVal;
    }

    bool operator<(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red < val);
        retVal      = retVal && (this->green < val);
        retVal      = retVal && (this->blue < val);
        return retVal;
    }

    bool operator>(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red > val);
        retVal      = retVal && (this->green > val);
        retVal      = retVal && (this->blue > val);
        return retVal;
    }

    bool operator>=(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red >= val);
        retVal      = retVal && (this->green >= val);
        retVal      = retVal && (this->blue >= val);
        return retVal;
    }

    bool operator<=(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red <= val);
        retVal      = retVal && (this->green <= val);
        retVal      = retVal && (this->blue <= val);
        return retVal;
    }
};

union Rgba16f {
    static const ColorFormat FORMAT = ColorFormat::RGBA16F;

    struct {
        uint16_t r;
        uint16_t g;
        uint16_t b;
        uint16_t a;
    };
    uint16_t f16[4];
};

union R32f {
    static const ColorFormat FORMAT = ColorFormat::R32F;

    struct {
        float red;
    };
    float f32;

    R32f operator+(float val) {
        R32f retVal;
        retVal.red = this->red + val;
        return retVal;
    }

    R32f operator-(float val) {
        R32f retVal;
        retVal.red = this->red - val;
        return retVal;
    }

    R32f operator*(float val) {
        R32f retVal;
        retVal.red = this->red * val;
        return retVal;
    }

    R32f operator/(float val) {
        R32f retVal;
        retVal.red = this->red / val;
        return retVal;
    }

    bool operator<(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red < val);
        return retVal;
    }

    bool operator>(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red > val);
        return retVal;
    }

    bool operator>=(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red >= val);
        return retVal;
    }

    bool operator<=(float val) {
        bool retVal = true;
        retVal      = retVal && (this->red <= val);
        return retVal;
    }
};

union Rgb8 {
    static const ColorFormat FORMAT = ColorFormat::RGB8;

    struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
    };
    uint8_t u8[3];
};

// Note: avoid using size_t in this structure. So the size of the structure will never change, regardless of compile platform.
struct ImagePlaneDesc {
    // pixel format
    ColorFormat format;

    // Plane width in pixels
    uint32_t width;

    // Plane height in pixels
    uint32_t height;

    // Plane depth in pixels
    uint32_t depth;

    uint32_t channels;

    /// bits (not BYTES) from one pixel to next. Minimal valid value is pixel size.
    /// When constructing the image, can be set to zero to use value calculated based on pixel format.
    uint32_t step;

    /// Bytes from one row to next. Minimal valid value is (width * step) and aligned to alignment.
    /// When constructing the image, can be set to zero to use value calculated based on width and step size.
    uint32_t pitch;

    // Bytes from one slice to next. Minimal valid value is (pitch * height)
    // When constructing the image, can be set to zero to use value calucalted based on picth and height.
    uint32_t slice;

    // Bytes between first pixel of the plane to the first pixel of the whole image.
    // When constructing the image, can be set to zero, means current plane is immediatly after the previous plane
    // with no extra space between.
    // of the current plane and the and of the previous plane.
    // Be aware that planse can interleave with each other. But there should be one and only one plane with zero offset.
    uint32_t offset;

    // return size of the plane in bytes
    uint32_t size() const { return slice * depth; }
};

struct ImageDesc {
    std::vector<ImagePlaneDesc> planes;
    uint32_t size      = 0; // total size in bytes;
    uint32_t alignment = 0;

    ImageDesc() {}

    ImageDesc(std::vector<ImagePlaneDesc>&& planes, uint32_t alignment = 0) { reset(std::move(planes), alignment); }

    ImageDesc(const std::vector<ImagePlaneDesc>& planes, uint32_t alignment = 0) {
        this->planes = planes;
        reset(alignment);
    }

    ImageDesc(ColorFormat f, size_t w, size_t h, size_t d, size_t channels, size_t step = 0, size_t pitch = 0, size_t slice = 0,
              uint32_t alignment = 0) {
        ImagePlaneDesc pd = {};
        pd.format         = f;
        pd.width          = (uint32_t) w;
        pd.height         = (uint32_t) h;
        pd.depth          = (uint32_t) d;
        pd.channels       = channels;
        pd.step           = (uint32_t) step;
        pd.pitch          = (uint32_t) pitch;
        pd.slice          = (uint32_t) slice;
        reset(&pd, 1, alignment);
    }

    ImageDesc(ColorFormat f, size_t w, size_t h = 1, size_t d = 1) {
        ImagePlaneDesc pd = {};
        pd.format         = f;
        pd.width          = (uint32_t) w;
        pd.height         = (uint32_t) h;
        pd.depth          = (uint32_t) d;
        pd.channels       = getColorFormatDesc(f).ch * d;
        reset(&pd, 1, alignment);
    }

    // can copy
    ImageDesc(const ImageDesc&) = default;
    ImageDesc& operator=(const ImageDesc&) = default;

    // can move
    ImageDesc(ImageDesc&& rhs) {
        planes        = std::move(rhs.planes);
        size          = rhs.size;
        rhs.size      = 0;
        alignment     = rhs.alignment;
        rhs.alignment = 0;
    }
    ImageDesc& operator=(ImageDesc&& rhs) {
        if (&rhs != this) {
            planes        = std::move(rhs.planes);
            size          = rhs.size;
            rhs.size      = 0;
            alignment     = rhs.alignment;
            rhs.alignment = 0;
        }
        return *this;
    }

    void reset(std::vector<ImagePlaneDesc>&& planes_, uint32_t alignment_ = 0);

    void reset(uint32_t alignment_ = 0);

    void reset(const ImagePlaneDesc* planes_, size_t count, uint32_t alignment_ = 0) { reset({planes_, planes_ + count}, alignment_); }

    bool empty() const { return planes.empty(); }

    ImagePlaneDesc& operator[](size_t index) { return planes[index]; }
    const ImagePlaneDesc& operator[](size_t index) const { return planes[index]; }

    // methods to calculate pixel offset
    size_t plane(size_t p) const { return validate(planes[p].offset); }
    size_t slice(size_t p, size_t z) const {
        const auto& d = planes[p];
        SNN_ASSERT(z <= d.depth);
        return validate(plane(p) + z * d.slice);
    }
    size_t row(size_t p, size_t y, size_t z = 0) const {
        const auto& d = planes[p];
        SNN_ASSERT(y < d.height);
        return validate(slice(p, z) + y * d.pitch);
    }
    size_t pixel(size_t p, size_t x, size_t y = 0, size_t z = 0) const {
        const auto& d = planes[p];
        SNN_ASSERT(x < d.width);
        return validate(row(p, y, z) + x * d.step / 8);
    }

    // constructs NV12 dual plane image descriptor
    static ImageDesc nv12(uint32_t w, uint32_t h) {
        ImageDesc desc;
        uint32_t y_slice  = w * h;
        uint32_t uv_w     = (w + 1) / 2; // half resolution (round up)
        uint32_t uv_h     = (h + 1) / 2;
        uint32_t uv_slice = uv_w * uv_h * 2;
        desc.planes       = {
            // Y plan with full resolution
            {
                ColorFormat::R8,
                w,
                h,
                1,
                3,
                8,
                w,
                y_slice,
                0,
            },

            // UV plan with half resolution
            {ColorFormat::RG8, uv_w, uv_h, 1, 3, 1, uv_w * 2, uv_slice, y_slice},
        };
        desc.size      = y_slice + uv_slice;
        desc.alignment = 1;
        return desc;
    }

    // TODO :- : This is similar to nv12, but since nv21 is unimplemented, we need something in its place!
    static ImageDesc nv21(uint32_t w, uint32_t h) { return nv12(w, h); }

    // construct I420 tri-plane image descriptor
    static ImageDesc i420(uint32_t w, uint32_t h) {
        ImageDesc desc;
        uint32_t y_slice  = w * h;
        uint32_t uv_w     = (w + 1) / 2; // half resolution (round up)
        uint32_t uv_h     = (h + 1) / 2;
        uint32_t uv_slice = uv_w * uv_h;
        desc.planes       = {
            // Y plan with full resolution
            {
                ColorFormat::R8,
                w,
                h,
                1,
                3,
                8,
                w,
                y_slice,
                0,
            },

            // U plan with half resolution
            {ColorFormat::R8, uv_w, uv_h, 1, 3, 1, uv_w, uv_slice, y_slice},

            // V plan with half resolution
            {ColorFormat::R8, uv_w, uv_h, 1, 3, 1, uv_w, uv_slice, y_slice + uv_slice},
        };
        desc.size      = y_slice + uv_slice * 2;
        desc.alignment = 1;
        return desc;
    }

private:
    // in debug build, make sure the pointer is in valid range.
    size_t validate(size_t offset) const {
        SNN_ASSERT(offset < size);
        return offset;
    }
};

class ManagedRawImage;

template<typename PIXEL>
class ManagedImage;

// This class describes an image with a known at runtime format.
// The class objects do not own image pixel buffer.
class RawImage {
public:
    RawImage() {}

    explicit RawImage(ImageDesc&& desc, void* pixels) { reset(std::move(desc), pixels); }

    // can copy
    RawImage(const RawImage&) = default;
    RawImage& operator=(const RawImage&) = default;

    // cannot copy from managed images !!! //
    RawImage(const ManagedRawImage&) = delete;
    RawImage& operator=(const ManagedRawImage&) = delete;

    template<typename PIXEL>
    RawImage(const ManagedImage<PIXEL>&) = delete;
    template<typename PIXEL>
    RawImage& operator=(const ManagedImage<PIXEL>&) = delete;
    /////////////////////////////////////////////

    // cannot move from managed images !!! //
    RawImage(ManagedRawImage&& rhs) = delete;
    RawImage& operator=(ManagedRawImage&& rhs) = delete;

    template<typename PIXEL>
    RawImage(ManagedImage<PIXEL>&& rhs) = delete;

    template<typename PIXEL>
    RawImage& operator=(ManagedImage<PIXEL>&& rhs) = delete;
    /////////////////////////////////////////////

    // return descriptor of the whole image
    const ImageDesc& desc() const { return _desc; }
    const ImagePlaneDesc& desc(size_t index) const { return _desc[index]; }

    // return pointer to pixel buffer.
    const uint8_t* data() const { return _pixels; }
    uint8_t* data() { return _pixels; }

    bool empty() const { return _desc.empty(); }

    // return size of the whole image in bytes.
    uint32_t size() const { return _desc.size; }

    // alignment in bytes
    uint32_t alignment() const { return _desc.alignment; }

    // methods to return properties of the specific plane.
    ColorFormat format(size_t index = 0) const { return _desc[index].format; }
    uint32_t width(size_t index = 0) const { return _desc[index].width; }
    uint32_t height(size_t index = 0) const { return _desc[index].height; }
    uint32_t depth(size_t index = 0) const { return _desc[index].depth; }
    uint32_t step(size_t index = 0) const { return _desc[index].step; }
    uint32_t channels(size_t index = 0) const { return _desc[index].channels; }
    uint32_t pitch(size_t index = 0) const { return _desc[index].pitch; }
    uint32_t sliceSize(size_t index = 0) const { return _desc[index].slice; }

    // Gets 3D plane
    const uint8_t* plane(size_t p) const { return _pixels + _desc.plane(p); }
    uint8_t* plane(size_t p) { return _pixels + _desc.plane(p); }

    // Gets 2D slice
    const uint8_t* slice(size_t p, size_t z) const { return _pixels + _desc.slice(p, z); }
    uint8_t* slice(size_t p, size_t z) { return _pixels + _desc.slice(p, z); }

    // Gets 1D row
    const uint8_t* row(size_t p, size_t y, size_t z = 0) const { return _pixels + _desc.row(p, y, z); }
    uint8_t* row(size_t p, size_t y, size_t z = 0) { return _pixels + _desc.row(p, y, z); }

    // Gets 1 pixel
    const uint8_t* at(size_t p, size_t x, size_t y = 0, size_t z = 0) const { return _pixels + _desc.pixel(p, x, y, z); }
    uint8_t* at(size_t p, size_t x, size_t y = 0, size_t z = 0) { return _pixels + _desc.pixel(p, x, y, z); }

    void vertFlipInpace();

    void saveToPNG(const std::string& filename, size_t sliceIndex = 0, bool makeOpaque = false, bool clamp = false) const;

    void saveToBIN(const std::string& filename, bool convertToFP32 = true) const;

    // Gets number of planes
    uint32_t planes() { return (uint32_t) this->_desc.planes.size(); }
    ImageDesc getDesc() const { return _desc; }

protected:
    uint8_t* _pixels = nullptr;

    void reset(ImageDesc&& desc, void* pixels);

    void moveFrom(RawImage& rhs) {
        if (this == &rhs) {
            return;
        }
        _desc       = std::move(rhs._desc);
        _pixels     = rhs._pixels;
        rhs._pixels = 0;
    }

private:
    ImageDesc _desc;
};

struct AlignedAllocator {
    static inline uint8_t* allocate(size_t size, size_t alignment) {
        SNN_ASSERT(alignment > 0);
        SNN_ASSERT(0 == (size % alignment));
#ifdef _MSC_VER
        auto p = _aligned_malloc(size, alignment);
#else
        auto p = aligned_alloc(alignment, size);
#endif
        if (!p) {
            SNN_LOGE("out of memory.");
        }
        return (uint8_t*) p;
    }
    static inline void deallocate(uint8_t*& p) {
        if (!p) {
            return;
        }
#ifdef _MSC_VER
        _aligned_free(p);
#else
        free(p);
#endif
        p = nullptr;
    }
};

template<typename PIXEL>
class ManagedImage;

// The objects of this class own pixel buffer.
class ManagedRawImage : public RawImage {
public:
    ManagedRawImage() {}

    ManagedRawImage(ImageDesc&& desc, const void* pixels = nullptr, size_t pixelBufferSizeInBytes = 0) {
        reset(std::move(desc), nullptr);
        store(pixels, pixelBufferSizeInBytes);
    }

    ~ManagedRawImage() { AlignedAllocator::deallocate(_pixels); }

    SNN_NO_COPY(ManagedRawImage);

    // can move
    ManagedRawImage(ManagedRawImage&& that) { moveFrom(that); }

    ManagedRawImage& operator=(ManagedRawImage&& that) {
        AlignedAllocator::deallocate(_pixels);
        moveFrom(that);
        return *this;
    }

    template<typename PIXEL>
    ManagedRawImage(ManagedImage<PIXEL>&& that) { moveFrom(that); }

    template<typename PIXEL>
    ManagedRawImage& operator=(ManagedImage<PIXEL>&& that) {
        AlignedAllocator::deallocate(_pixels);
        moveFrom(that);
        return *this;
    }

    static ManagedRawImage loadFromFile(const std::string& filename, bool fromBin = false);
    static ManagedRawImage loadFromAsset(const std::string& assetName, bool fromBin = false);

private:
    void store(const void* buffer, size_t length);
};

// This class describes an image with a known at compile time format.
// Format of each planes must be identical.
template<typename PIXEL>
class TypedImage : public RawImage {
    static_assert(getColorFormatDesc(PIXEL::FORMAT).bits == sizeof(PIXEL) * 8);

public:
    typedef PIXEL Pixel;

    static constexpr ColorFormat format() { return PIXEL::FORMAT; }

    TypedImage() {}

    TypedImage(ImageDesc&& desc, void* pixels = nullptr) {
        reset(std::move(desc), pixels);
        verifyPlaneFormat();
    }

    /// construct a simple 2D image.
    TypedImage(size_t width, size_t height, void* pixels = nullptr) {
        reset(ImageDesc(PIXEL::FORMAT, width, height), pixels);
        verifyPlaneFormat();
    }

    // copy from raw image
    TypedImage(const RawImage& image): RawImage(image) { verifyPlaneFormat(); }
    TypedImage& operator=(const RawImage& rhs) {
        RawImage::operator=(rhs);
        verifyPlaneFormat();
        return *this;
    }

    // can move
    TypedImage(TypedImage&& rhs) { moveFrom(rhs); }
    TypedImage(RawImage&& rhs) {
        moveFrom(rhs);
        verifyPlaneFormat();
    }
    TypedImage& operator=(TypedImage&& rhs) {
        moveFrom(rhs);
        return *this;
    }
    TypedImage& operator=(RawImage&& rhs) {
        moveFrom(rhs);
        verifyPlaneFormat();
        return *this;
    }

    const PIXEL* plane(size_t p) const { return (const PIXEL*) RawImage::plane(p); }
    PIXEL* plane(size_t p) { return (PIXEL*) RawImage::plane(p); }

    const PIXEL* slice(size_t p, size_t z) const { return (const PIXEL*) RawImage::slice(p, z); }
    PIXEL* slice(size_t p, size_t z) { return (PIXEL*) RawImage::slice(p, z); }

    const PIXEL* row(size_t p, size_t y, size_t z = 0) const { return (const PIXEL*) RawImage::row(p, y, z); }
    PIXEL* row(size_t p, size_t y, size_t z = 0) { return (PIXEL*) RawImage::row(p, y, z); }

    const PIXEL& at(size_t p, size_t x, size_t y, size_t z = 0) const { return *(const PIXEL*) RawImage::at(p, x, y, z); }
    PIXEL& at(size_t p, size_t x, size_t y, size_t z = 0) { return *(PIXEL*) RawImage::at(p, x, y, z); }

    const PIXEL& operator()(size_t p, size_t x, size_t y, size_t z = 0) const { return at(p, x, y, z); }
    PIXEL& operator()(size_t p, size_t x, size_t y, size_t z = 0) { return at(p, x, y, z); }

private:
    void verifyPlaneFormat() const {
#ifdef _DEBUG
        for (auto& pd : desc().planes) {
            SNN_ASSERT(pd.format == PIXEL::FORMAT);
        }
#endif
    }
};

// This class describes an image with a known at compile time format.
// The objects of this class own pixel buffer.
// Format of each planes must be identical.
template<typename PIXEL>
class ManagedImage : public TypedImage<PIXEL> {
public:
    ManagedImage() {}

    ManagedImage(ImageDesc&& d) {
        SNN_ASSERT(d.planes[0].format == PIXEL::FORMAT);
        RawImage::reset(std::move(d), nullptr);
        allocate();
    }

    // construct simple 2D image
    ManagedImage(size_t w, size_t h) {
        RawImage::reset(ImageDesc(PIXEL::FORMAT, w, h), nullptr);
        allocate();
    }

    ~ManagedImage() { deallocate(); }

    SNN_NO_COPY(ManagedImage);

    // can move
    ManagedImage(ManagedImage&& rhs) { RawImage::moveFrom(rhs); }
    ManagedImage(ManagedRawImage&& rhs) { RawImage::moveFrom(rhs); }
    ManagedImage& operator=(ManagedImage&& rhs) {
        deallocate();
        moveFrom(rhs);
        return *this;
    }
    ManagedImage& operator=(ManagedRawImage&& rhs) {
        deallocate();
        RawImage::moveFrom(rhs);
        return *this;
    }

private:
    void allocate() {
        // TODO: call constructor for non-aggregate types.
        RawImage::_pixels = AlignedAllocator::allocate(RawImage::size(), RawImage::alignment());
    }

    void deallocate() {
        if (RawImage::_pixels) {
            // TODO: call destructors for non-aggregate pixel type
            AlignedAllocator::deallocate(RawImage::_pixels);
            RawImage::_pixels = nullptr;
        }
    }
};

// Conversion functions
bool toRgba32f(Rgba32f& dst, const uint8_t* src, ColorFormat srcFormat);
bool toRgba32f(const RawImage& src, TypedImage<Rgba32f>& dst);
bool toRgba32f(const RawImage& src, TypedImage<Rgba32f>& dst, float min, float max);
ManagedImage<Rgba32f> toRgba32f(const RawImage& src);
ManagedImage<Rgba32f> toRgba32f(const RawImage& src, float min, float max);

bool toRgba16f(Rgba16f& dst, const uint8_t* src, ColorFormat srcFormat);
bool toRgba16f(const RawImage& src, TypedImage<Rgba16f>& dst);
ManagedImage<Rgba16f> toRgba16f(const RawImage& src);

bool toR32f(R32f& dst, const uint8_t* src, ColorFormat srcFormat);
bool toR32f(const RawImage& src, TypedImage<R32f>& dst);
bool toR32f(const RawImage& src, TypedImage<R32f>& dst, float min, float max);
ManagedImage<R32f> toR32f(const RawImage& src);
ManagedImage<R32f> toR32f(const RawImage& src, float min, float max);

bool toR8(R8& dst, const uint8_t* src, ColorFormat srcFormat, bool normalize = true);
bool toR8(const RawImage& src, TypedImage<R8>& dst);
ManagedImage<R8> toR8(const RawImage& src);

bool toRgb8(Rgb8& dst, const uint8_t* src, ColorFormat srcFormat, bool normalize = true);
bool toRgb8(const RawImage& src, TypedImage<Rgb8>& dst);
ManagedImage<Rgb8> toRgb8(const RawImage& src);

bool toRgba8(Rgba8& dst, const uint8_t* src, ColorFormat srcFormat, bool normalize = true);
bool toRgba8(const RawImage& src, TypedImage<Rgba8>& dst);
bool toRgba8(const RawImage& src, TypedImage<Rgba8>& dst, ColorFormat format);
ManagedImage<Rgba8> toRgba8(const RawImage& src, bool makeOpaque = false);

bool normalize(const RawImage& src, TypedImage<Rgba32f>& dst, const std::vector<float>& means, const std::vector<float>& norms);
ManagedImage<Rgba32f> normalize(const RawImage& src, const std::vector<float>& means, const std::vector<float>& norms);

void clamp(const RawImage& src, RawImage& dst);
ManagedRawImage clamp(const RawImage& src);
void clamp(RawImage& srcDst);

bool rgba8ToI420(const RawImage& rgba, RawImage& i420);
bool rgba8ToNv12(const RawImage& rgba, RawImage& nv12);
bool nv12ToRgba8(const RawImage& nv12, RawImage& rgba);
bool rgba8ToNv21(const RawImage& rgba, RawImage& nv21);
bool nv21ToRgba8(const RawImage& nv21, RawImage& rgba);
bool nv12ToI420(const RawImage& nv12, RawImage& i420);
bool i420ToRgba8(const RawImage& i420, RawImage& rgba);
bool i420ToNv12(const RawImage& i420, RawImage& nv12);

void printC4Buffer(float *buffer, int input_h, int input_w, int input_c, std::ostream& stream = std::cout);
void printC4BufferFP16(uint16_t *buffer, int input_h, int input_w, int input_c, std::ostream& stream = std::cout);

} // namespace snn
