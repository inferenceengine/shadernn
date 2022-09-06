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

#include <iostream>
#include "snn/snn.h"
#include "snn/utils.h"
#include "snn/glUtils.h"
#include <functional>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include "pch.h"
#include <sys/stat.h>
#include <sys/types.h>
#include "ic2/dp.h"

namespace snn {
struct PackPbo {
    FrameImage2::Desc desc;
    gl::BufferObject<GL_PIXEL_PACK_BUFFER> b;
    bool hasData = 0;

    PackPbo(const FrameImage2::Desc& d): desc(d) {
        auto bufferSize = getColorFormatDesc(d.format).calcImageSizeInBytes(d.width, d.height);
        b.allocate<uint8_t>(bufferSize, nullptr, GL_STREAM_READ);
    }

    void readPixels() {
        b.bind();
        static Timer t("PPBO - read pixels");
        ScopedTimer st(t);
        auto fd = getColorFormatDesc(desc.format);
        GLCHK(glReadPixels(0, 0, desc.width, desc.height, fd.glFormat, fd.glType, (void*) 0));
        b.unbind();
        hasData = true;
    }

    void download(RawImage& dst) {
        if (!hasData) {
            return;
        }
        static Timer t("PPBO - map buffer");
        ScopedTimer st(t);
        hasData         = false;
        const void* src = b.map();
        memcpy(dst.data(), src, std::min<size_t>(dst.size(), b.length));
        b.unmap();
    }
};

class CpuFrameImageBase : public FrameImage2 {
protected:
    mutable RawImage _image;

public:
    mutable PackPbo ppbo;

    CpuFrameImageBase(const Desc& d): ppbo(d) { _desc = d; }

    RawImage& getCpuData() const override {
        ppbo.download(_image);
        return _image;
    }
};

class CpuFrameImage : public CpuFrameImageBase {
    std::vector<uint8_t> _buffer;

public:
    CpuFrameImage(const Desc& d): CpuFrameImageBase(d) {
        ImageDesc desc(d.format, d.width, d.height);
        _buffer.resize(desc.size);
        _image = RawImage(std::move(desc), _buffer.data());
    }
};

class GpuFrameImage : public FrameImage2 {
public:
    GpuFrameImage(const Desc& d) {
        _desc = d;
        if (d.depth == 1) {
            _texture.allocate2D(d.format, d.width, d.height, _desc.channels, 1);
        } else {
            _texture.allocate2DArray(_desc.format, _desc.width, _desc.height, _desc.depth, _desc.channels, 1);
        }
    }

    void updateTextureContent(ColorFormat format, void* data, uint32_t size = 0) {
        if (size == 0) {
            size = _desc.width * _desc.height * _desc.channels;
        }
        auto formatDesc = getColorFormatDesc(_desc.format);
        if (format == _desc.format) {
            if (_desc.depth > 4) {
                std::size_t layerCount = 0;
                std::size_t offset     = _desc.width * _desc.height * 4;
                for (std::size_t i = 0; i < _desc.depth; i += 4) {
                    layerCount = (std::size_t) i / 4;
                    if (formatDesc.glType == GL_FLOAT) {
                        std::vector<float> tempData((float*) ((float*) data + 4 * layerCount * offset),
                                                    (float*) ((float*) data + 4 * (layerCount + 1) * offset));
                        _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
                    } else {
                        std::vector<uint8_t> tempData(((uint8_t*) data + 4 * layerCount * offset), ((uint8_t*) data + 4 * (layerCount + 1) * offset));
                        _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
                    }
                }
                if (4 * layerCount < _desc.depth) {
                    if (formatDesc.glType == GL_FLOAT) {
                        std::vector<float> tempData((float*) ((float*) data + 4 * layerCount * offset), (float*) ((float*) data + size));
                        _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
                    } else {
                        std::vector<uint8_t> tempData(((uint8_t*) data + 4 * layerCount * offset), ((uint8_t*) data + size));
                        _texture.setPixels(layerCount, 0, 0, 0, _desc.width, _desc.height, 0, tempData.data());
                    }
                }
            } else {
                _texture.setPixels(0, 0, 0, _desc.width, _desc.height, 0, data);
            }
            std::cout << "Inside TextureUpdateContent final" << std::endl;
        } else {
            SNN_LOGE("incompatible format.");
        }
    }

    void attach(GLenum target, GLuint texture) {
        _texture.attach(target, texture);
        auto& texDesc  = _texture.getDesc();
        _desc.channels = texDesc.channels;
        _desc.depth    = texDesc.depth;
        _desc.width    = texDesc.width;
        _desc.height   = texDesc.height;
        _desc.format   = texDesc.format;
    }

    OnGPU getGpuData() const override { return {_texture.target(), _texture.id()}; }

    void dumpTexture(const std::string& filename, size_t level = 0) {
        // std::cout << "------------------ Dumping Textures ---------------------" << std::endl;
        // std::cout << _texture << std::endl;
        // std::cout << _desc << std::endl;
        // _texture.bind();
        auto texImage = _texture.getBaseLevelPixels();
        // _texture.unbind();
        std::string filenameStr(filename);
        filenameStr = filenameStr + "_" + std::to_string(_texture.target()) + "_" + std::to_string(_texture.id());
        filenameStr = filenameStr + "_" + std::to_string(_desc.width) + "_" + std::to_string(_desc.height) + ".png";
        texImage.saveToPNG(filenameStr.c_str(), level);
        // std::cout << "---------------------------------------------------------" << std::endl;
    }

    snn::ManagedRawImage dumpTexture() { return _texture.getBaseLevelPixels(); }

private:
    gl::TextureObject _texture;
};

inline std::unique_ptr<FrameImage2> createFrameImage(const FrameImage2::Desc& d) {
    switch (d.device) {
    case Device::GPU:
        return std::make_unique<GpuFrameImage>(d);
    case Device::CPU:
        return std::make_unique<CpuFrameImage>(d);
    case Device::CPU_GPU:
        return std::make_unique<CpuFrameImage>(d);
    case Device::GPU_CPU:
        return std::make_unique<GpuFrameImage>(d);
    }
    return nullptr;
}

template<typename T>
inline T* castTo(const FrameImage2* frame) {
    if constexpr (std::is_same_v<T, CpuFrameImage>) {
        return (frame->desc().device == Device::CPU) ? (T*) frame : nullptr;
    } else if constexpr (std::is_same_v<T, GpuFrameImage>) {
        return (frame->desc().device == Device::GPU) ? (T*) frame : nullptr;
    } else if constexpr (std::is_same_v<T, CpuFrameImageBase>) {
        return (frame->desc().device == Device::CPU) ? (T*) frame : nullptr;
    } else {
        // TODO: static_assert(false);std::vector<uint8_t> _outbuf;
        SNN_LOGE("invalid type.");
        return nullptr;
    }
}

// -------------------------------------------------------------------------
// The generic processor class that represents one step in the processing graph
class Processor {
public:
    virtual ~Processor() = default;

    typedef struct FrameVectorDesc {
        Device device;
        ColorFormat format;
        uint32_t size; // number of frame images in the vector

        bool scale = false;
        float min  = 0.0f;
        float max  = 255.0f;

        bool operator==(const FrameVectorDesc& rhs) const { return device == rhs.device && format == rhs.format && size == rhs.size; }

        friend std::ostream& operator<<(std::ostream& os, const FrameVectorDesc& desc) {
            os << "Size: " << desc.size << std::endl;
            switch (desc.device) {
            case Device::GPU:
                os << "Device: GPU" << std::endl;
                break;
            case Device::CPU:
                os << "Device: CPU" << std::endl;
                break;
            case Device::GPU_CPU:
                os << "Device: GPU -> CPU" << std::endl;
                break;
            case Device::CPU_GPU:
                os << "Device: GPU -> CPU" << std::endl;
                break;
            default:
                break;
            }
            return os;
        }
    } FrameVectorDesc;

    struct FrameDims {
        uint32_t width, height, depth, channels;
    };

    struct FrameDims inputDims, outputDims;
    bool hasInputDims = false, hasOutputDims = false;

    void setInputDims(const FrameDims& dims) {
        inputDims    = dims;
        hasInputDims = true;
    }

    void setOutputDims(const FrameDims& dims) {
        outputDims    = dims;
        hasOutputDims = true;
    }

    void getOutputDims(FrameDims& dims) { dims = outputDims; }

    void getInputDims(FrameDims& dims) { dims = inputDims; }

    bool hasDims() { return (hasInputDims && hasOutputDims); }

    struct Desc {
        FrameVectorDesc i; // input frame vector descriptor
        FrameVectorDesc o; // output frame vector descriptor

        friend std::ostream& operator<<(std::ostream& os, const Desc& desc) {
            os << "Input: " << std::endl << desc.i;
            os << std::endl << "Output: " << desc.o << std::endl;
            return os;
        }
    };

    const Desc& desc() const { return _desc; }

    // submit the processing work. returns a fence that marks the end of the processing; or 0, if the
    // proccess is done completely synchronously.
    struct Workload {
        FrameImage2** inputs;
        std::vector<std::vector<std::vector<float>>> cpuInputs;
        size_t inputCount;
        FrameImage2* output;
        std::vector<std::vector<std::vector<float>>> cpuOutput;
        InferenceEngine::SNNModelOutput modelOutput;
    };

    virtual void submit(Workload&)     = 0;
    virtual std::string getModelName() = 0;

protected:
    Processor(const Desc& d): _desc(d) {}

private:
    Desc _desc;
};

// -------------------------------------------------------------------------
// Processor that transfer frame image GPU to CPU
class DownloadFromGpuProcessor : public Processor {
public:
    DownloadFromGpuProcessor(const FrameVectorDesc& i, Device outdev): Processor({i, {outdev, i.format, i.size}}), _fm(snn::FenceManager::getInstance()) {
        SNN_ASSERT(i.device == Device::GPU);
        SNN_ASSERT(outdev == Device::CPU);
        GLCHK(glGenFramebuffers(1, &_fbo));
    }

    ~DownloadFromGpuProcessor() {
        if (_fbo) {
            glDeleteFramebuffers(1, &_fbo), _fbo = 0;
        }
    }

    void submit(Workload& w) override {
        SNN_ASSERT(1 == w.inputCount);
        auto src = w.inputs[0]->getGpuData();
        auto& sd = w.inputs[0]->desc();
        auto& dd = w.output->desc();

        // const auto& outDesc = w.output->desc();
        // const auto& fmtDesc = getColorFormatDesc(outDesc.format);

        // const auto& inFmtDesc = getColorFormatDesc(w.inputs[0]->desc().format);

        // std::cout << "DownloadFromGPU Input Color Format: " << inFmtDesc.name << std::endl;
        // std::cout << "DownloadFromGPU Output Color Format: " << fmtDesc.name << std::endl;

        if (sd.width == dd.width && sd.height == dd.height && sd.format == dd.format) {
            auto& fd = getColorFormatDesc(sd.format);
            GLCHK(glBindFramebuffer(GL_FRAMEBUFFER, _fbo));
            GLCHK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, src.target, src.texture, 0));
            auto cpu = castTo<CpuFrameImageBase>(w.output);
            if (cpu) {
                cpu->ppbo.readPixels();
            } else {
                auto dst = w.output->getCpuData();
                if (!dst.empty()) {
                    static Timer t("glReadPixels()");
                    ScopedTimer st(t);
                    GLCHK(glReadPixels(0, 0, sd.width, sd.height, fd.glFormat, fd.glType, (void*) dst.data()));
                } else {
                    SNN_LOGE("unsupported output frame type");
                }
            }
            GLCHK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        } else {
            SNN_LOGE("incomaptible input and output frame.");
        }
    }

    std::string getModelName() override { return "Download from GPU Processor"; }

private:
    std::shared_ptr<FenceManager> _fm;
    GLuint _fbo = 0;
};

// -------------------------------------------------------------------
// Processor that resizes image on CPU to match input dims of network
// Fixed for current Basic CNN
class ResizeProcessor : public Processor {
public:
    ResizeProcessor(const FrameVectorDesc& i): Processor({i, i}) {
        this->setInputDims({1080, 1920, 1, 3});
        this->setOutputDims({112, 112, 1, 3});
    }

    ResizeProcessor(const FrameVectorDesc& i, const FrameVectorDesc& o): Processor({i, o}) {
        this->setInputDims({1080, 1920, 1, 3});
        this->setOutputDims({112, 112, 1, 3});
        this->setScaling(false);
    }

    ~ResizeProcessor() override {}

    void submit(Workload& w) override;

    std::string getModelName() override { return "Resize Processor"; }
    void setScaling(bool scale);

private:
    bool scale;
};

// -------------------------------------------------------------------
// Processor that process and transfer frame image from CPU to GPU
class Upload2GpuProcessor : public Processor {
public:
    typedef std::function<void(FrameImage2* const*, size_t, const Range<uint8_t*, uint8_t*>&)> KernelFunc;

    Upload2GpuProcessor(const FrameVectorDesc& i, KernelFunc f = {}): Processor({i, {Device::GPU, i.format, 1}}), _func(f) {
        this->scale = i.scale;
        this->min   = i.min;
        this->max   = i.max;
    }

    ~Upload2GpuProcessor() override {}

    void submit(Workload& w) override {
        // Make sure input and output frames are on correct device.
        // If these assert are triggered, it is most likely due to
        // incorrect pipeline setup is incorrect in Engine::initialize()
        for (size_t i = 0; i < w.inputCount; ++i) {
            SNN_ASSERT(Device::CPU == w.inputs[i]->desc().device || Device::GPU_CPU == w.inputs[i]->desc().device);

            if (w.inputs[i]->desc().format == snn::ColorFormat::RGBA32F) {
                auto& image = w.inputs[i]->getCpuData();

                if (this->scale) {
                    image = snn::toRgba32f(image, this->min, this->max);
                } else {
                    image = snn::toRgba32f(image);
                }
            }
        }
        SNN_ASSERT(Device::GPU == w.output->desc().device || Device::GPU_CPU == w.output->desc().device);

        // allocate output buffer
        const auto& outDesc = w.output->desc();
        // std::cout << outDesc << std::endl;
        const auto& fmtDesc = getColorFormatDesc(outDesc.format);

        // const auto& inFmtDesc = getColorFormatDesc(w.inputs[0]->desc().format);

        _outbuf.resize(fmtDesc.calcImageSizeInBytes(outDesc.width, outDesc.height));

        // std::cout << "Upload2GPU Input Color Format: " << inFmtDesc.name << std::endl;
        // std::cout << "Upload2GPU Output Color Format: " << fmtDesc.name << std::endl;
        // std::cout << "Output Buffer size: " << _outbuf.size() << std::endl;

        // call the kernel function
        if (_func) {
            _func(w.inputs, w.inputCount, {_outbuf.data(), _outbuf.data() + _outbuf.size()});
        }

        // update output frame image
        if (_func) {
            if (Device::GPU == w.output->desc().device) {
                castTo<GpuFrameImage>(w.output)->updateTextureContent(outDesc.format, _outbuf.data());
                // castTo<GpuFrameImage>(w.output)->dumpTexture("/home/us000145/bitbucket/debug_out/basic_cnn/image_uploaded_to_gpu");
            } else {
                auto dst = w.output->getCpuData();
                memcpy(dst.data(), _outbuf.data(), std::min<size_t>(dst.size(), _outbuf.size()));
            }
        } else {
            // passthrough copy
            copy(*w.inputs[0], *w.output);
            // castTo<GpuFrameImage>(w.output)->dumpTexture("/home/us000145/bitbucket/debug_out/basic_cnn/image_uploaded_to_gpu_no_func");
            // auto outputGPUData = w.output->getGpuData();
            // w.inputs[0]->getCpuData().saveToPNG("/home/us000145/bitbucket/debug_out/basic_cnn/image_from_resize_proc.png");
            // std::cout << "------------------------------------------------------" << std::endl;
            // std::cout << "Output Texture: " << outputGPUData.texture << std::endl;
            // std::cout << "Output Target: " << outputGPUData.target << std::endl;
            // std::cout << "------------------------------------------------------" << std::endl;
        }
    }

    std::string getModelName() override { return "Upload to GPU Processor"; }

private:
    KernelFunc _func;
    std::vector<uint8_t> _outbuf;

    bool scale = false;
    float min  = 0.0;
    float max  = 255.0;

    static void copy(FrameImage2& src, FrameImage2& dst) {
        auto srcimg = src.getCpuData();
        if (srcimg.empty()) {
            SNN_LOGW("unsupported source frame device.");
            return;
        }
        if (isCompatible(src, dst)) {
            if (Device::GPU == dst.desc().device) {
                castTo<GpuFrameImage>(&dst)->updateTextureContent(dst.desc().format, srcimg.data());

            } else {
                auto dstimg = dst.getCpuData();
                memcpy(dstimg.data(), srcimg.data(), std::min<size_t>(dstimg.size(), srcimg.size()));
            }
        } else {
            // TODO: do format conversion here.
            SNN_LOGW("incompatible frame.");
        }
    }

    static bool isCompatible(const FrameImage2& src, const FrameImage2& dst) {
        auto& s = src.desc();
        auto& d = dst.desc();
        return s.width == d.width && s.height == d.height && s.format == d.format;
    }
};
} // namespace snn
