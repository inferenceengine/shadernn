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
#include "yolov3Processor.h"
#include "appContext.h"
#ifdef SUPPORT_GL
    #include "yolov3ProcessorGL.h"
#endif
#ifdef SUPPORT_VULKAN
    #include "volk.h"
    #include "yolov3ProcessorVulkan.h"
#endif

using namespace snn;

std::unique_ptr<Yolov3Processor>
Yolov3Processor::createYolov3Processor(ColorFormat format, Precision precision, bool compute, bool dumpOutputs) {
    (void) compute;
    switch (AppContext::getContext()->backendType) {
#ifdef SUPPORT_GL
        case GpuBackendType::GL:
            return std::unique_ptr<Yolov3Processor>(new Yolov3ProcessorGL(format, precision, compute, dumpOutputs));
#endif
#ifdef SUPPORT_VULKAN
        case GpuBackendType::VULKAN:
            return std::unique_ptr<Yolov3Processor>(new Yolov3ProcessorVulkan(format, precision, dumpOutputs));
#endif
        default:
            SNN_RIP("Unexpected GpuBackendType");
    }
}

void Yolov3Processor::getBBoxesCoords(const std::vector<float>& boxDetails, std::vector<float>& coords) {
    SNN_ASSERT(boxDetails.size() >= 6);
    float confidence = boxDetails[1];

    if (confidence >= 0.4) {
        float x = boxDetails[2];
        float y = boxDetails[3];
        float w = boxDetails[4];
        float h = boxDetails[5];

        if (w > 0.01 && h > 0.01) {
            float TL_x = (x - w / 2.f);
            float TL_y = (y + h / 2.f);
            float BR_x = (x + w / 2.f);
            float BR_y = (y - h / 2.f);

            coords.push_back(TL_x);
            coords.push_back(TL_y);
            coords.push_back(BR_x);
            coords.push_back(BR_y);
        }
    }
}
