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
#pragma once
#include "defines.h"
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <memory>
#include <set>
#include <variant>
#include <tuple>

// Disable some known "harmless" warnings. So we can use /W4 throughout our code base.
#ifdef _MSC_VER
    #pragma warning(disable : 4201) // nameless struct/union
#endif

// Log macros
#define SNN_LOG(severity, ...)                                                                                                                                 \
    do {                                                                                                                                                       \
        if ((int) (snn::LogSeverity::severity) <= snn::g_enabledLogSeverity) {                                                                                 \
            snn::log(__FILE__, __LINE__, __FUNCTION__, (int) snn::LogSeverity::severity, __VA_ARGS__);                                                         \
        }                                                                                                                                                      \
    } while (0)
#define SNN_LOGE(...) SNN_LOG(ERR, __VA_ARGS__)
#define SNN_LOGW(...) SNN_LOG(WARN, __VA_ARGS__)
#define SNN_LOGI(...) SNN_LOG(INFO, __VA_ARGS__)
#define SNN_LOGD(...) SNN_LOG(DEBUG, __VA_ARGS__)
#define SNN_LOGV(...) SNN_LOG(VERBOSE, __VA_ARGS__)
#define SNN_RIP(...)                                                                                                                                           \
    do {                                                                                                                                                       \
        SNN_LOG(FATAL, __VA_ARGS__);                                                                                                                           \
        snn::rip();                                                                                                                                            \
    } while (0)

#define SNN_LOG_EVERY_N_SEC(interval, severity, ...)                                                                                                           \
    do {                                                                                                                                                       \
        if ((int) (snn::LogSeverity::severity) <= snn::g_enabledLogSeverity) {                                                                                 \
            static snn::details::LogTimer timer____;                                                                                                           \
            if (timer____.isTimeToLog(interval)) {                                                                                                             \
                snn::log(__FILE__, __LINE__, __FUNCTION__, (int) snn::LogSeverity::severity, __VA_ARGS__);                                                     \
            }                                                                                                                                                  \
        }                                                                                                                                                      \
    } while (0)
#define SNN_LOG_FIRST_N_TIMES(n, severity, ...)                                                                                                                \
    do {                                                                                                                                                       \
        if ((int) (snn::LogSeverity::severity) <= snn::g_enabledLogSeverity) {                                                                                 \
            static int counter__ = 0;                                                                                                                          \
            if (counter__++ < (n)) {                                                                                                                           \
                snn::log(__FILE__, __LINE__, __FUNCTION__, (int) snn::LogSeverity::severity, __VA_ARGS__);                                                     \
            }                                                                                                                                                  \
        }                                                                                                                                                      \
    } while (0)

// Runtime check macros
#define SNN_CHK(x)                                                                                                                                             \
    if (!(x)) {                                                                                                                                                \
        SNN_RIP(#x);                                                                                                                                           \
    } else                                                                                                                                                     \
        void(0)
#ifdef _DEBUG
    #define SNN_ASSERT SNN_CHK
#else
    #define SNN_ASSERT(...) (void) 0
#endif

namespace snn {
namespace details {

class LogTimer {
    std::chrono::high_resolution_clock::time_point _lastLogTime;

public:
    LogTimer() {}
    bool isTimeToLog(int interval) {
        using namespace std::chrono;
        if (interval <= 0) {
            return true;
        }
        auto now = high_resolution_clock::now();
        auto dur = duration_cast<seconds>(now - _lastLogTime).count();
        if (dur < interval) {
            return false;
        }
        _lastLogTime = now;
        return true;
    }
};
} // namespace details

enum class LogSeverity {
    FATAL = 0,
    ERR,
    WARN,
    INFO,
    DEBUG,
    VERBOSE,
};

// Only logs with severity value equal to or lower than this is enabled. The default value is INFO.
extern SNN_API int g_enabledLogSeverity;

float convertToMediumPrecision(float in);
void convertToMediumPrecision(std::vector<float>& in);
void convertToMediumPrecision(std::vector<double>& in);
float convertToHighPrecision(uint16_t in);

void getByteRepresentation(float in, std::vector<unsigned char>& byteRep, bool fp16 = false);

// Log function
void log(const char* file, int line, const char* function, int severity, const char* format, ...);

[[noreturn]] void rip();

// utility functions
std::string formatString(const char* format, ...);
void convertEndianness(uint8_t* buffer, uint32_t size);
void convertEndianness(float* buffer, uint32_t size);

// Utility function to dump current call stack into string
std::string dumpCallStack(int indent = 8);

// Load asset/resources embeded into the executable or package.
std::vector<uint8_t> loadEmbeddedAsset(const char* path);
std::vector<uint8_t> loadJsonFromStorage(const char* path);

// std::vector<std::string> grepEmbeddedAssetFolder(const char * basedir, const char * pattern, bool recursive);

// convert duration in nanoseconds to string
std::string ns2s(uint64_t ns);

template<class... Fs>
struct match : Fs... {
    using Fs::operator()...;
};
template<class... Fs>
match(Fs...) -> match<Fs...>;

// This class does not move or copy elements at all. So the class T can have both move and copy operators deleted.
template<typename T>
class FixedSizeArray {
    T* _ptr      = nullptr;
    size_t _size = 0;

public:
    FixedSizeArray() = default;

    FixedSizeArray(size_t n) { allocate(n); }

    SNN_NO_COPY(FixedSizeArray);

    // can move
    FixedSizeArray(FixedSizeArray&& that) {
        _ptr       = that._ptr;
        that._ptr  = nullptr;
        _size      = that._size;
        that._size = 0;
    }
    FixedSizeArray& operator=(FixedSizeArray&& that) {
        if (this != &that) {
            deallocate();
            _ptr       = that._ptr;
            that._ptr  = nullptr;
            _size      = that._size;
            that._size = 0;
        }
        return *this;
    }

    ~FixedSizeArray() { deallocate(); }

    void allocate(size_t n) {
        if (n == _size) {
            return;
        }
        deallocate();
        _ptr  = new T[n];
        _size = n;
    }

    void deallocate() {
        if (_ptr) {
            delete[] _ptr;
            _ptr = nullptr;
        }
        _size = 0;
    }

    bool empty() const { return !_ptr; }

    auto size() const -> size_t { return _size; }

    auto data() -> T* { return _ptr; }
    auto data() const -> const T* { return _ptr; }

    auto begin() -> T* { return _ptr; }
    auto begin() const -> const T* { return _ptr; }

    auto end() -> T* { return _ptr + _size; }
    auto end() const -> const T* { return _ptr + _size; }

    auto back() -> T& { return _ptr[_size - 1]; }
    auto back() const -> const T& { return _ptr[_size - 1]; }

    auto operator[](size_t index) const -> const T& {
        SNN_ASSERT(index < _size);
        return _ptr[index];
    }
    auto operator[](size_t index) -> T& {
        SNN_ASSERT(index < _size);
        return _ptr[index];
    }
};

struct Timer {
    template<typename T, size_t N = 10>
    struct Averager {
        T buffer[N];
        size_t cursor = 0;
        bool notFull  = true;
        T low, high, average;

        Averager() { reset(); }

        void reset() {
            for (size_t i = 0; i < N; ++i) {
                buffer[i] = T {0};
            }
            cursor  = 0;
            notFull = true;
            low     = T {0};
            high    = T {0};
            average = T {0};
        }

        void update(T value) {
            buffer[cursor] = value;
            ++cursor;
            if (cursor >= N) {
                notFull = false;
            }
            cursor %= N;
        }

        void refreshAverage() {
            size_t count_ = notFull ? cursor : N;

            if (0 == count_) {
                low     = T {0};
                high    = T {0};
                average = T {0};
                return;
            }

            low    = buffer[0];
            high   = buffer[0];
            T sum_ = buffer[0];
            for (size_t i = 1; i < count_; ++i) {
                auto value = buffer[i];
                sum_ += value;
                if (value < low) {
                    low = value;
                }
                if (value > high) {
                    high = value;
                }
            }
            average = sum_ / count_;
        }
    };

    std::string name;
    size_t count                                                       = 0;
    bool running                                                       = false;
    std::chrono::high_resolution_clock::duration sum                   = {};
    std::chrono::high_resolution_clock::time_point begin               = {};
    mutable Averager<std::chrono::high_resolution_clock::duration> ave = {};

    thread_local static Timer* root;
    Timer* possibleParent = nullptr;
    std::set<Timer*> children;
    Timer(std::string n): name(n) {}
    ~Timer();

    SNN_NO_COPY(Timer);
    SNN_NO_MOVE(Timer);

    void start() {
        if (!running) {
            running = true;
            begin   = std::chrono::high_resolution_clock::now();

            if (!root) {
                root = this;
                children.clear();
            } else {
                SNN_ASSERT(root != this);
                possibleParent = root;
            }
        }
    }

    void stop() {
        if (running) {
            if (root == this) {
                root = nullptr;
            } else if (possibleParent == root) {
                root->children.insert(root->children.end(), this);
            }
            running = false;
            count++;
            auto duration = std::chrono::high_resolution_clock::now() - begin;
            ave.update(duration);
            sum += duration;
        }
    }

    void reset() {
        sum     = {};
        count   = 0;
        running = false;
        ave.reset();
    }

    std::chrono::high_resolution_clock::duration getAverageDuration() const {
        ave.refreshAverage();
        return ave.average;
    }

    uint64_t getAverageDurationMs() const {
        using namespace std::chrono;
        return duration_cast<milliseconds>(getAverageDuration()).count();
    }

    uint64_t getAverageDurationUs() const {
        using namespace std::chrono;
        return duration_cast<microseconds>(getAverageDuration()).count();
    }

    uint64_t getAverageDurationNs() const {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(getAverageDuration()).count();
    }

    std::string print(size_t nameColumnWidth, const Timer* parent = nullptr) const;
};

template<typename TIMER>
class ScopedTimer {
    TIMER& _timer;

public:
    SNN_NO_COPY(ScopedTimer);

    explicit ScopedTimer(TIMER& t): _timer(t) { _timer.start(); }

    ~ScopedTimer() { _timer.stop(); }
};

template<class Iterator, class Sentinel>
struct Range : std::tuple<Iterator, Sentinel> {
    using std::tuple<Iterator, Sentinel>::tuple;

    auto begin() const -> Iterator { return std::get<0>(*this); }
    auto end() const -> Iterator { return std::get<1>(*this); }

    template<class = std::enable_if_t<std::is_convertible_v<decltype(std::declval<Sentinel>() - std::declval<Iterator>()), size_t>>>
    auto size() const -> size_t {
        return end() - begin();
    }
};
template<class Iterator, class Sentinel>
Range(Iterator, Sentinel) -> Range<Iterator, Sentinel>;

namespace range {
template<class X>
auto fromPtr(X* ptr, size_t n) -> Range<X*, X*> {
    return {ptr, ptr + n};
}
} // namespace range

template<size_t i>
struct Get {
    template<class T>
    constexpr auto operator()(const T& x) const -> const auto& {
        return std::get<i>(x);
    }
    template<class T>
    constexpr auto operator()(T& x) const -> auto& {
        return std::get<i>(x);
    }
};

} // namespace snn
