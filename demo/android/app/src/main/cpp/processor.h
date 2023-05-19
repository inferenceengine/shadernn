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
#include "demoutils.h"
#include <iostream>

namespace snn {

std::unique_ptr<FrameImage2> createFrameImage(const FrameImage2::Desc& d);

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

        friend std::ostream& operator<<(std::ostream& os, const FrameVectorDesc& desc);
    } FrameVectorDesc;

    struct FrameDims {
        uint32_t width, height, depth, channels;
    };

    virtual void init(const FrameDims& inputDims_, const FrameDims& outputDims_) {
        inputDims    = inputDims_;
        hasInputDims = true;
        outputDims    = outputDims_;
        hasOutputDims = true;
    }

    const FrameDims& getOutputDims() const { return outputDims; }

    const FrameDims& getInputDims() const { return inputDims; }

    bool hasDims() { return (hasInputDims && hasOutputDims); }

    struct Desc {
        FrameVectorDesc i; // input frame vector descriptor
        FrameVectorDesc o; // output frame vector descriptor

        friend std::ostream& operator<<(std::ostream& os, const Desc& desc);
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
        SNNModelOutput modelOutput;
    };

    virtual void submit(Workload&)     = 0;
    virtual std::string getModelName() = 0;

protected:
    Processor(const Desc& d): _desc(d) {}

private:
    Desc _desc;
    FrameDims inputDims;
    FrameDims outputDims;
    bool hasInputDims = false;
    bool hasOutputDims = false;
};

} // namespace snn
