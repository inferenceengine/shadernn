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
#include "processor.h"
#include "demoutils.h"
#include <functional>
#include <string>
#include <vector>

namespace snn {

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

    virtual ~Upload2GpuProcessor() override {}

    void submit(Workload& w) override;
    std::string getModelName() override { return "Upload to GPU Processor"; }

private:
    KernelFunc _func;
    std::vector<uint8_t> _outbuf;

    bool scale = false;
    float min  = 0.0;
    float max  = 255.0;

    static void copy(FrameImage2& src, FrameImage2& dst);

    static bool isCompatible(const FrameImage2& src, const FrameImage2& dst) {
        auto& s = src.desc();
        auto& d = dst.desc();
        return s.width == d.width && s.height == d.height && s.format == d.format;
    }
};

}
