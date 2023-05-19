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

#include <string>

namespace uvkc {

struct TimerExternal {
    virtual ~TimerExternal() = default;

    virtual void start() {}

    virtual void stop() {}
};

typedef TimerExternal* (*PFUN_TIMER_FACTORY)(const std::string& name);
extern PFUN_TIMER_FACTORY gExternalTimerFactory;

class Timer {
public:
    Timer(const std::string& name)
        : _name(name)
    {}

    ~Timer() {
        delete _timer_external;
    }

    Timer(const Timer&) = delete;
    Timer& operator = (const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator = (Timer&&) = delete;

    void start() {
        checkTimer();
        if (_timer_external) {
            _timer_external->start();
        }
    }

    void stop() {
        checkTimer();
        if (_timer_external) {
            _timer_external->stop();
        }
    }

private:
    void checkTimer() {
        if (!_timer_external && gExternalTimerFactory) {
            _timer_external = (*gExternalTimerFactory)(_name);
        }
    }

    TimerExternal* _timer_external = nullptr;
    std::string _name;
};

class ScopedTimer {
    Timer& _timer;

public:
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator = (const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator = (ScopedTimer&&) = delete;

    explicit ScopedTimer(Timer& t): _timer(t) { _timer.start(); }

    ~ScopedTimer() { _timer.stop(); }
};

void SetExternalTimerFactory(PFUN_TIMER_FACTORY factory);

#define UVKC_PROFILE_TIME(name, desc) \
    static Timer timer##name (desc); \
    ScopedTimer scopedTimer##name (timer##name);

} // uvkc