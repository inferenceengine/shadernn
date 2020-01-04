/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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
#include <chrono>
#include <snn/utils.h>
#include <opencv2/core.hpp>
#include <glad/glad.h>
#include "nativeFrameProvider.h"

using namespace snn;

NativeFrameProvider::NativeFrameProvider(size_t frameBufferCapacity)
    : defaultQueue(frameBufferCapacity), hiLoQueue(frameBufferCapacity), frameQueue(std::in_place_type<std::reference_wrapper<DefaultQueue>>, defaultQueue) {
    setFrameSetSize(1);
}

bool NativeFrameProvider::fetchData(const InferenceEngine::FrameVec& frameSet) {
    setFrameSetSize(frameSet.size());
    //    SNN_LOGI("The current queue type being used is: %s",
    //             std::visit(match{
    //                 [&](DefaultQueue& q) {
    //                     (void) q;
    //                     return "DefaultQueue";
    //                 },
    //                 [&](HiLoQueue& q) {
    //                     (void) q;
    //                     return "HiLoQueue";
    //                 }
    //             }, frameQueue));
    auto copyData = [](snn::ManagedRawImage& src, const FrameImage2& dst) {
        auto dstimg = dst.getCpuData();
        if (dstimg.empty()) {
            SNN_LOGE("the incoming frame is not CPU accessible.");
            return false;
        } else {
            copyImageData(src, dstimg);
            //            SNN_LOGI("Size of output image is: %u", dstimg.size());
            //            SNN_LOGI("Dims of output image are: %u %u",  dstimg.height(), dstimg.width());
            //            if (dstimg.format() == snn::ColorFormat::RGBA32F) {
            //                for (std::size_t i = 0; i < dstimg.size(); i+=4) {
            //                    float dest;
            //                    unsigned char buffer[] = {dstimg.data()[i], dstimg.data()[i+1], dstimg.data()[i+2], dstimg.data()[i+3]};
            //                    memcpy(&dest, &buffer, sizeof(float));
            //                    if (-0.0f <= dest && dest <= 0.0f) {
            //                        SNN_LOGI("Zeros start at %u", i);
            //                        SNN_LOGI("Max Size is: %u", dstimg.size());
            //                        break;
            //                    }
            //                }
            //                SNN_LOGI("Loaded Image");
            //            }
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
    src.saveToPNG("/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump/input_cam_image.png");
    if (src.width() != dst.width() || src.height() != dst.height()) {
        SNN_LOGE("incompatible image dimensions.");
    } else if ((ColorFormat::NV12 == src.format() || ColorFormat::NV21 == src.format()) && ColorFormat::RGBA8 == dst.format()) {
        static Timer timer("NativeFrameProvider - copy frame content CPU -> CPU");
        ScopedTimer st(timer);
        auto yuv = snn::RawImage(ImageDesc::nv12(src.width(), src.height()), (void*) src.data());
        SNN_ASSERT(yuv.size() == src.size());
        if (ColorFormat::NV12 == src.format()) {
            nv12ToRgba8(yuv, dst);
        } else {
            nv21ToRgba8(yuv, dst);
        }
        //    } else if ((ColorFormat::NV12 == src.format() || ColorFormat::NV21 == src.format()) && ColorFormat::RGBA32F == dst.format()) {
        //        static Timer timer("NativeFrameProvider - copy frame content CPU -> CPU");
        //        ScopedTimer st(timer);
        //        auto yuv = snn::RawImage(ImageDesc::nv12(src.width(), src.height()), (void*)src.data());
        //        std::vector<uint8_t> intermediateBuffer(src.size(), 0);
        //        snn::RawImage intermediateRGB8(ImageDesc(snn::ColorFormat::RGBA8, src.width(), src.height()), intermediateBuffer.data());
        //        // std::vector<uint8_t> floatBuffer;
        //        SNN_ASSERT(yuv.size() == src.size());
        //        auto &destCf = snn::getColorFormatDesc(dst.format());
        //        SNN_LOGI("Destination is of the type: %s", destCf.name);
        //        if (ColorFormat::NV12 == src.format()) {
        //            nv12ToRgba8(yuv, intermediateRGB8);
        //            intermediateRGB8.saveToPNG("/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump/input_intermediate_image.png");
        //        } else {
        //            nv21ToRgba8(yuv, intermediateRGB8);
        //            intermediateRGB8.saveToPNG("/sdcard/Android/data/com.innopeaktech.seattle.snndemo/files/inferenceCoreDump/input_intermediate_image.png");
        //        }
        //        std::size_t offset = 0;
        //        for (std::size_t i = 0; i < intermediateRGB8.size(); i++) {
        //            typedef union {
        //                unsigned char buffer[4];
        //                float f;
        //            } temp;
        //            temp tempUnion;
        //            tempUnion.f = (float) intermediateRGB8.data()[i];
        //            if (i == intermediateRGB8.size() / 2) {
        //                SNN_LOGI("Float is: %f", tempUnion.f);
        //                SNN_LOGI("Buffer contents are: %d, %d, %d, %d", (int) tempUnion.buffer[0], (int) tempUnion.buffer[1], (int) tempUnion.buffer[2], (int)
        //                tempUnion.buffer[3]); SNN_LOGI("Index is: %u", i);
        //            }
        //            memcpy(dst.data() + offset, &tempUnion.buffer, sizeof(float));
        //            offset += 4;
        //        }
        //        // SNN_LOGD("float data buffer size: %u", floatBuffer.size());
        ////        for (std::size_t i = 0; i < floatBuffer.size(); i+=4) {
        ////            SNN_LOGI("Input Val: %d, %d, %d, %d", (int) floatBuffer.at(i), (int) floatBuffer.at(i + 1), (int) floatBuffer.at(i + 2), (int)
        ///floatBuffer.at(i + 3)); /        }
        //        SNN_LOGI("Size of destination data buffer is: %u", offset);
        //        SNN_LOGI("Successfully converted to RGBA32F");
        //    } else {
    } else {
        SNN_LOGE("unsupported incoming image format.");
    }
}
