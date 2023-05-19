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
#include "vkUtils.h"
#include "uvkc/benchmark/status_util.h"

void vk::GpuTimeElapsedQuery::start() {
    if (started) {
        SNN_LOGD("gpu time already started");
        return;
    }
    _cmdBuf->ResetQueryPool(*_tsQueryPool);
    _cmdBuf->WriteTimestamp(*_tsQueryPool, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0);
    started = true;
}


void vk::GpuTimeElapsedQuery::stop() {
    if (!started) {
        SNN_LOGD("gpu time not started yet");
        return;
    }
    _cmdBuf->WriteTimestamp(*_tsQueryPool, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 1);
    started = false;
}

void vk::GpuTimeElapsedQuery::getTime() {
    BM_CHECK_OK_AND_ASSIGN(
        double timestampSeconds,
        _tsQueryPool->CalculateElapsedSecondsBetween(0, 1));
    _result = (uint64_t)(timestampSeconds * (1e9));
}

// -----------------------------------------------------------------------------
//
std::string vk::GpuTimeElapsedQuery::print() const {
    return snn::formatString("%s : %s"), getName().c_str(), snn::ns2s(duration()).c_str();
}
