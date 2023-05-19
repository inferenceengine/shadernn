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
#include "snn/colorUtils.h"
#include "snn/utils.h"
#include "snn/image.h"
#include <string.h>

snn::ColorFormat snn::fromName(const char* name) {
    for (size_t i = 1; i < std::size(snn::COLOR_FORMAT_DESC_TABLE); ++i) {
        if (strcmp(snn::COLOR_FORMAT_DESC_TABLE[i].name, name) == 0) {
            return (snn::ColorFormat) i;
        }
    }
    return snn::ColorFormat::NONE;
}

std::set<std::string> snn::getAllColorNames() {
    std::set<std::string> names;
    for (size_t i = 1; i < std::size(snn::COLOR_FORMAT_DESC_TABLE); ++i) {
        names.insert(snn::COLOR_FORMAT_DESC_TABLE[i].name);
    }
    return names;
}

snn::ColorFormatType snn::getColorFormatType(ColorFormat cf) {
    auto iter = ColorFormatTypesTable.find(cf);
    SNN_ASSERT(iter != ColorFormatTypesTable.end());
    return iter->second;
}

std::vector<uint8_t> snn::convertColorBuffer(snn::ColorFormat cf, const float* buf, size_t len) {
    snn::ColorFormatDesc cfd = snn::getColorFormatDesc(cf);
    snn::ColorFormatType cft = snn::getColorFormatType(cf);
    size_t bytes = cfd.bits / cfd.ch / 8;
    std::vector<uint8_t> raw_buf(len * bytes);
    uint8_t* raw_ptr = raw_buf.data();
    for (size_t i = 0; i < len; ++i) {
        float val = buf[i];
        switch (cft) {
        case snn::ColorFormatType::UINT8:
            {
                uint8_t* ptr = raw_ptr;
                *ptr = static_cast<uint8_t>(val);
            }
            break;
        case snn::ColorFormatType::UINT16:
            {
                uint16_t* ptr = reinterpret_cast<uint16_t*>(raw_ptr);
                *ptr = static_cast<uint16_t>(val);
            }
            break;
        case snn::ColorFormatType::FLOAT16:
            {
                snn::FP16* ptr = reinterpret_cast<snn::FP16*>(raw_ptr);
                ptr->u = snn::FP32::toHalf(val);
            }
            break;
        case snn::ColorFormatType::FLOAT32:
            {
                float* ptr = reinterpret_cast<float*>(raw_ptr);
                *ptr = val;
            }
            break;
        default:
            SNN_CHK(false);
        }
        raw_ptr += bytes;
    }
    return raw_buf;
}
