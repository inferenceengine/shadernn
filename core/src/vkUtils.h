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
#include "snn/deviceTimer.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include <string>
#include <iostream>
#include <memory>

namespace vk {
// -----------------------------------------------------------------------------
// For asynchronous timer (not time stamp) queries
class GpuTimeElapsedQuery : public DeviceTimer {
public:
    GpuTimeElapsedQuery(const std::string& n, uvkc::vulkan::CommandBuffer* cmdBuf, uvkc::vulkan::Device* device)
        : DeviceTimer(n)
        , _cmdBuf(cmdBuf)
        , _device(device)
    {
        BM_CHECK_OK_AND_ASSIGN(_tsQueryPool, device->CreateTimestampQueryPool(2));
    }

    ~GpuTimeElapsedQuery() {}

    // returns duration in nanoseconds
    uint64_t duration() const  override { return _result; }

    void start() override;

    void stop() override;

    void getTime() override;

    // Print stats to string
    std::string print() const override;

    friend inline std::ostream& operator<<(std::ostream& s, const GpuTimeElapsedQuery& t) {
        s << t.print();
        return s;
    }

private:
    uint64_t _result = 0;
    uvkc::vulkan::CommandBuffer* _cmdBuf;
    uvkc::vulkan::Device* _device;
    std::unique_ptr<::uvkc::vulkan::TimestampQueryPool> _tsQueryPool;
    bool started = false;
};

} // namespace vk
