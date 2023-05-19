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
#include "upload2GpuProcessor.h"
#include "demoutils.h"
#include "gpuFrameImage.h"

namespace snn {

void Upload2GpuProcessor::submit(Workload& w) {
    // Make sure input and output frames are on correct device.
    // If these assert are triggered, it is most likely due to
    // incorrect pipeline setup is incorrect in Engine::initialize()
    for (size_t i = 0; i < w.inputCount; ++i) {
        SNN_ASSERT(Device::CPU == w.inputs[i]->desc().device);
    }
    SNN_ASSERT(Device::GPU == w.output->desc().device);

    // allocate output buffer
    const auto& outDesc = w.output->desc();
    const auto& fmtDesc = getColorFormatDesc(outDesc.format);

    _outbuf.resize(fmtDesc.calcImageSizeInBytes(outDesc.width, outDesc.height));

    // call the kernel function
    if (_func) {
        _func(w.inputs, w.inputCount, {_outbuf.data(), _outbuf.data() + _outbuf.size()});
    }

    // update output frame image
    if (_func) {
        if (Device::GPU == w.output->desc().device) {
            GpuFrameImage::cast(w.output)->updateTextureContent(outDesc.format, _outbuf.data());
        } else {
            RawImage dst;
            w.output->getCpuImage(dst);
            memcpy(dst.data(), _outbuf.data(), std::min<size_t>(dst.size(), _outbuf.size()));
        }
    } else {
        // passthrough copy
        copy(*w.inputs[0], *w.output);
    }
}

void Upload2GpuProcessor::copy(FrameImage2& src, FrameImage2& dst) {
    RawImage srcimg;
    src.getCpuImage(srcimg);
    if (srcimg.empty()) {
        SNN_LOGW("unsupported source frame device.");
        return;
    }
    if (isCompatible(src, dst)) {
        if (Device::GPU == dst.desc().device) {
            GpuFrameImage::cast(&dst)->updateTextureContent(dst.desc().format, srcimg.data());
        } else {
            RawImage dstimg;
            dst.getCpuImage(dstimg);
            memcpy(dstimg.data(), srcimg.data(), std::min<size_t>(dstimg.size(), srcimg.size()));
        }
    } else {
        // TODO: do format conversion here.
        SNN_LOGW("incompatible frame.");
    }
}

}
