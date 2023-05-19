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
#include "snn/image.h"
#include "inferenceengine.h"
#include "demoutils.h"
#include "queues.h"
#include <vector>
#include <utility>
#include <type_traits>
#include <functional>
#include <variant>

namespace snn {
class NativeFrameProvider : public FrameProvider2 {
public:
    using Timestamp = long;
    using LowFlag   = bool;

    bool fetchData(const InferenceEngine::FrameVec&) override;

    NativeFrameProvider(size_t frameBufferCapacity);

    void queueFrame(ColorFormat f, size_t w, size_t h, const uint8_t* img, size_t length, Timestamp timestamp);
    void queueMetadata(Timestamp timestamp, LowFlag isLowFrame);

private:
    void setFrameSetSize(size_t);

    using DefaultQueue = queue::BlockingReaderWriter<snn::ManagedRawImage>;
    struct HiLoQueue {
        struct ValueType {
            snn::ManagedRawImage* low;
            snn::ManagedRawImage* high;
        };

        template<class T>
        using TimedQueue = queue::BlockingReaderWriter<std::pair<Timestamp, T>>;

        TimedQueue<snn::ManagedRawImage> framesIn;
        TimedQueue<LowFlag> metadataIn;
        queue::SlidingWindow<2, queue::JoinedMonotonic<TimedQueue<snn::ManagedRawImage>&, TimedQueue<LowFlag>&>> framesOut;

        HiLoQueue(size_t capacity)
            : framesIn(capacity), metadataIn(capacity), framesOut(queue::slidingWindow<2>(queue::joinedMonotonic(framesIn, metadataIn))) {}

        template<class Read>
        auto operator()(const Read& read, ValueType& v) -> std::enable_if_t<std::is_base_of_v<queue::ReadOp, Read>, bool> {
            std::vector<std::reference_wrapper<std::tuple<std::pair<Timestamp, snn::ManagedRawImage>, std::pair<Timestamp, LowFlag>>>> captures;

            if (framesOut(read, captures)) {
                auto& frame0 = std::get<snn::ManagedRawImage>(std::get<0>(captures[0].get()));
                auto& frame1 = std::get<snn::ManagedRawImage>(std::get<0>(captures[1].get()));

                auto& frame0IsLow = std::get<LowFlag>(std::get<1>(captures[0].get()));
                auto& frame1IsLow = std::get<LowFlag>(std::get<1>(captures[1].get()));

                if (frame0IsLow ^ frame1IsLow) {
                    if (frame0IsLow) {
                        v.low  = &frame0;
                        v.high = &frame1;
                    } else if (frame1IsLow) {
                        v.low  = &frame1;
                        v.high = &frame0;
                    }
                    return true;
                } else {
                    SNN_LOG_FIRST_N_TIMES(10, ERR,
                                          "Exposure compensation alternation not detected! Either Hi/Lo frames is not enabled or frames are being dropped.");
                    v.low  = &frame0;
                    v.high = &frame1;
                    return true; // TODO if a frame is dropped, this will put a high/high or low/low frame pair into the pipeline
                }
            } else
                return false;
        }

        HiLoQueue(const HiLoQueue&) = delete;
        HiLoQueue(HiLoQueue&&)      = delete;
    };

    DefaultQueue defaultQueue;
    HiLoQueue hiLoQueue;
    std::variant<std::reference_wrapper<DefaultQueue>, std::reference_wrapper<HiLoQueue>> frameQueue;
    size_t frameSetSize;

    static void copyImageData(const snn::ManagedRawImage& src, snn::RawImage& dst);
};

} // namespace snn
