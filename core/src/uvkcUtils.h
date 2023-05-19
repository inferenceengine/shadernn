
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

#include "uvkc/base/log.h"
#include "uvkc/base/timer.h"
#include "snn/utils.h"
#include <sstream>

// This file contains extensions for uvkc library
// The design goal is to avoid bringing uvkc dependency on ShaderNN
// and keep it as a "3rdParty"

namespace uvkc {

#if defined(__ANDROID__)

// This class adds "normal" Android logger into uvkc
class AndroidStreamLogger : public Logger {
public:
static void SetLogger();

private:
    explicit AndroidStreamLogger(std::stringstream* tmpStream)
        : Logger(tmpStream)
        , tmpStream_(tmpStream)
    {
        SNN_ASSERT(tmpStream);
    }

    std::stringstream* tmpStream_;
    std::stringstream bufStream_;
    virtual void logInternal() override;
};

#endif

// This structure adds profiling timer to uvkc
struct TimerAdapter : public TimerExternal {
    TimerAdapter(const std::string& n)
        : timer(n)
    {}

    virtual ~TimerAdapter() = default;

    void start() override {
        timer.start();
    }

    void stop() override {
        timer.stop();
    }

    snn::Timer timer;
};

TimerExternal* timerAdapterFactory(const std::string& name);

}  // namespace uvkc
