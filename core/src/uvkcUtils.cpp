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

#include "uvkcUtils.h"

namespace uvkc {

#if defined(__ANDROID__)

#include <android/log.h>

static const size_t FORMAT_BUFFER_SIZE = 16 * 1024;
thread_local static char buf2[FORMAT_BUFFER_SIZE];

void AndroidStreamLogger::logInternal() {
    std::string inputStr = tmpStream_->str();
    uint32_t numEols = 0;
    for (size_t pos = 0;; ++numEols) {
        pos = inputStr.find('\n', pos);
        if (pos == std::string::npos) {
            break;
        }
        pos = pos + 1;
    }
    bufStream_ << inputStr;
    while (numEols--) {
        bufStream_.getline(buf2, FORMAT_BUFFER_SIZE);
        __android_log_print(ANDROID_LOG_ERROR, "SNN", "%s", buf2);
    }
    // Clear temporary buffer
    tmpStream_->str("");
}

void AndroidStreamLogger::SetLogger() {
    thread_local static std::stringstream tmpStream;
    static AndroidStreamLogger logger(&tmpStream);
    SetExternalStreamLogger(&logger);
}

#endif

TimerExternal* timerAdapterFactory(const std::string& name) {
    return new TimerAdapter(name);
}

}
