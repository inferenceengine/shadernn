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
#include "nativeFrameProvider.h"
#include "demoutils.h"
#include "snn/utils.h"
#include <utility>
#include <functional>
#include <variant>

using namespace snn;

NativeFrameProvider::NativeFrameProvider(size_t frameBufferCapacity)
    : defaultQueue(frameBufferCapacity), hiLoQueue(frameBufferCapacity), frameQueue(std::in_place_type<std::reference_wrapper<DefaultQueue>>, defaultQueue) {
    setFrameSetSize(1);
}

bool NativeFrameProvider::fetchData(const InferenceEngine::FrameVec& frameSet) {
    setFrameSetSize(frameSet.size());
    auto copyData = [](snn::ManagedRawImage& src, const FrameImage2& dst) {
        RawImage dstimg;
        dst.getCpuImage(dstimg);
        if (dstimg.empty()) {
            SNN_LOGE("the incoming frame is not CPU accessible.");
            return false;
        } else {
            copyImageData(src, dstimg);
            return true;
        }
    };
    return std::visit(match {[&](DefaultQueue& q) { // TODO if queue empty, prefer to re-check if the queue type changes before attempting to dequeue again,
                                                    // else a change of queue will result in some arbitrary wait
                                snn::ManagedRawImage frame;
                                if (q(queue::waitDequeue, frame)) {
                                    return copyData(frame, *frameSet.at(0));
                                } else
                                    return false;
                                },
                             [&](HiLoQueue& q) {
                                HiLoQueue::ValueType capture;
                                if (q(queue::waitDequeue, capture)) {
                                    return copyData(*capture.low, *frameSet.at(0)) && copyData(*capture.high, *frameSet.at(1));
                                } else
                                    return false;
                                }
                            },
                      frameQueue);
}

void NativeFrameProvider::setFrameSetSize(size_t n) {
    if (n == frameSetSize) {
        return;
    }

    frameSetSize = n;

    switch (n) {
    case 1:
        frameQueue.emplace<std::reference_wrapper<DefaultQueue>>(defaultQueue);
        return;
    case 2:
        frameQueue.emplace<std::reference_wrapper<HiLoQueue>>(hiLoQueue);
        return;
    default:
        SNN_RIP("invalid frame set size (we currently support 1 and 2)");
    }
}

void NativeFrameProvider::queueFrame(ColorFormat f, size_t w, size_t h, const uint8_t* ptr, size_t length, Timestamp timestamp) {
    auto frame = snn::ManagedRawImage({f, w, h}, ptr, length);
    if (!std::visit(match {
                            [&](DefaultQueue& q) {
                                return q(queue::tryEmplace, std::move(frame));
                            },
                            [&](HiLoQueue& q) {
                                return q.framesIn(queue::tryEmplace, timestamp, std::move(frame)); // FIXME rotation messes up the image
                            }
                        },
                    frameQueue)) {
        SNN_LOGV("frame queue is full! dropping a frame");
    }
}

void NativeFrameProvider::queueMetadata(Timestamp timestamp, LowFlag isLowFrame) {
    std::visit(match {
                [](DefaultQueue&) {},
                [&](HiLoQueue& q) {
                    q.metadataIn(queue::tryEmplace, timestamp, isLowFrame);
                }
            }, frameQueue);
}

void NativeFrameProvider::copyImageData(const snn::ManagedRawImage& src, snn::RawImage& dst) {
    if (src.width() != dst.width() || src.height() != dst.height()) {
        SNN_LOGE("incompatible image dimensions.");
    } else if ((ColorFormat::NV12 == src.format() || ColorFormat::NV21 == src.format()) && ColorFormat::RGBA8 == dst.format()) {
        PROFILE_TIME(CopyCpuCpu, "NativeFrameProvider - copy frame content CPU -> CPU")
        auto yuv = snn::RawImage(ImageDesc::nv12(src.width(), src.height()), (void*) src.data());
        SNN_ASSERT(yuv.size() == src.size());
        if (ColorFormat::NV12 == src.format()) {
            nv12ToRgba8(yuv, dst);
        } else {
            nv21ToRgba8(yuv, dst);
        }
    } else {
        SNN_LOGE("unsupported incoming image format.");
    }
}
