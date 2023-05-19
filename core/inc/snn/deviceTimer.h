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
#include "snn/utils.h"

// -----------------------------------------------------------------------------
// For asynchronous timer (not time stamp) queries
class DeviceTimer {
public:
    DeviceTimer(const std::string n = "")
        : name(n)
    {}

    virtual ~DeviceTimer() {}

    // returns duration in nanoseconds
    virtual uint64_t duration() const { return 0; }

    virtual void start() {}

    virtual void stop() {}

    virtual void getTime() {}

    // Print stats to string
    virtual std::string print() const { return snn::formatString("%s : %s"), name.c_str(), snn::ns2s(duration()).c_str(); }

    friend inline std::ostream& operator<<(std::ostream& s, const DeviceTimer& t) {
        s << t.print();
        return s;
    }

    const std::string& getName() const {
        return name;
    }

private:
    const std::string name;
};
