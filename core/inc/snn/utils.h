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
#include "defines.h"
#include "snn/color.h"
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <tuple>
#include <math.h>

// Disable some known "harmless" warnings. So we can use /W4 throughout our code base.
#ifdef _MSC_VER
    #pragma warning(disable : 4201) // nameless struct/union
#endif

#ifndef __FILENAME__
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

// Log macros
#define SNN_LOG(sev, ...) \
    do { \
        int severity = (int)snn::LogSeverity::sev; \
        static int androidSeverity = 0; \
        static bool loggable = snn::isLoggable(severity, androidSeverity, __FILE__); \
        if (loggable) { \
            snn::log(__FILE__, __LINE__, __FUNCTION__, severity, androidSeverity, __VA_ARGS__); \
        } \
    } while (0)

#define SNN_LOGE(...) SNN_LOG(ERR, __VA_ARGS__)
#define SNN_LOGW(...) SNN_LOG(WARN, __VA_ARGS__)
#define SNN_LOGI(...) SNN_LOG(INFO, __VA_ARGS__)
#define SNN_LOGD(...) SNN_LOG(DEBUG, __VA_ARGS__)
#define SNN_LOGV(...) SNN_LOG(VERBOSE, __VA_ARGS__)
#define SNN_RIP(...) \
    do { \
        SNN_LOG(FATAL, __VA_ARGS__); \
        snn::rip(); \
    } while (0)

#define SNN_LOG_EVERY_N_SEC(interval, sev, ...) \
    do { \
        int severity = (int)snn::LogSeverity::sev; \
        static int androidSeverity = 0; \
        static bool loggable = snn::isLoggable(severity, androidSeverity, __FILE__); \
        if (loggable) { \
            static snn::details::LogTimer timer____; \
            if (timer____.isTimeToLog(interval)) { \
                snn::log(__FILE__, __LINE__, __FUNCTION__, severity, androidSeverity, __VA_ARGS__); \
            } \
        } \
    } while (0)
#define SNN_LOG_FIRST_N_TIMES(n, sev, ...) \
    do { \
        int severity = (int)snn::LogSeverity::sev; \
        static int androidSeverity = 0; \
        static bool loggable = snn::isLoggable(severity, androidSeverity, __FILE__); \
        if (loggable) { \
            static int counter__ = 0; \
            if (counter__++ < (n)) { \
                snn::log(__FILE__, __LINE__, __FUNCTION__, severity, androidSeverity, __VA_ARGS__); \
            } \
        } \
    } while (0)

// Runtime check macros
#define SNN_CHK(x) \
    if (!(x)) { \
        SNN_RIP(#x); \
    } else { \
        void(0); \
    }
#ifdef _DEBUG
    #define SNN_ASSERT SNN_CHK
#else
    #define SNN_ASSERT(...) (void) 0
#endif

namespace snn {
namespace details {

// Utility class for periodic logging
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

// Utility functions for half precision floating point
float convertToMediumPrecision(float in);
void convertToMediumPrecision(std::vector<float>& in);
void convertToMediumPrecision(std::vector<double>& in);
float convertToHighPrecision(uint16_t in);

void getByteRepresentation(float in, std::vector<unsigned char>& byteRep, bool fp16 = false);

// Log functions
bool isLoggable(int severity, int& androidSeverity, const char* file);
void log(const char* file, int line, const char* function, int severity, int androidSeverity, const char* format, ...);

[[noreturn]] void rip();

// utility functions
std::string formatString(const char* format, ...);
void convertEndianness(uint8_t* buffer, uint32_t size);
void convertEndianness(float* buffer, uint32_t size);

// Utility function to dump current call stack into string
std::string dumpCallStack(int indent = 8);

// Load asset/resources embedded into the executable or package.
std::vector<uint8_t> loadEmbeddedAsset(const char* path);
std::vector<uint8_t> loadJsonFromStorage(const char* path);

// convert duration in nanoseconds to string
std::string ns2s(uint64_t ns);

template<class... Fs>
struct match : Fs... {
    using Fs::operator()...;
};
template<class... Fs>
match(Fs...) -> match<Fs...>;

// Template classes for fixed-sized array
template<typename T>
struct ArrayAllocator {
    T* allocate(size_t n) {
        return new T[n];
    }

    void deallocate(T* ptr, size_t) {
        delete[] ptr;
    }
};

template<typename T, typename... Args>
struct ArrayParamAllocator {
    ArrayParamAllocator(Args... args)
        : params(args...)
    {}

    T* allocate(size_t n) {
        void* ptr = malloc(sizeof(T) * n);
        T* t0 = (T*)ptr;
        T* t = t0;
        for (size_t i = 0; i < n; ++i, ++t) {
            std::apply([&t](Args... args) {
                new(t) T(args...);
            }, params);
        }
        return t0;
    }

    void deallocate(T* t0, size_t n) {
        T* t = t0;
        for (size_t i = 0; i < n; ++i, ++t) {
            t->~T();
        }
        free(t0);
    }

    std::tuple<Args...> params;
};

// This class does not move or copy elements at all. So the class T can have both move and copy operators deleted.
template<typename T, typename A = ArrayAllocator<T>>
class FixedSizeArray {
    T* _ptr      = nullptr;
    size_t _size = 0;
    A _allocator;

public:
    FixedSizeArray(A a = A())
        : _allocator(a)
    {}

    FixedSizeArray(size_t n, A a = A())
        : _allocator(a)
    {
        allocate(n);
    }

    SNN_NO_COPY(FixedSizeArray);

    // can move
    FixedSizeArray(FixedSizeArray&& that) {
        _ptr       = that._ptr;
        that._ptr  = nullptr;
        _size      = that._size;
        that._size = 0;
        _allocator = that._allocator;
    }

    FixedSizeArray& operator=(FixedSizeArray&& that) {
        if (this != &that) {
            deallocate();
            _ptr       = that._ptr;
            that._ptr  = nullptr;
            _size      = that._size;
            that._size = 0;
            _allocator = that._allocator;
        }
        return *this;
    }

    ~FixedSizeArray() { deallocate(); }

    void allocate(size_t n) {
        if (n == _size) {
            return;
        }
        deallocate();
        _ptr  = _allocator.allocate(n);
        _size = n;
    }

    void allocate(size_t n, A a) {
        _allocator = a;
        allocate(n);
    }

    void deallocate() {
        if (_ptr) {
            _allocator.deallocate(_ptr, _size);
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

struct NoCheck {
    static bool checkType(const void*) {
        return true;
    }
};

// Static size array with polymorphic access to the elements
// Does not own the underlying array
template<typename T, typename B = T, typename C = NoCheck>
class PolyArrayAccessor {
public:
    PolyArrayAccessor()
        : _ptr(nullptr)
        , _size(0U)
    {
        static_assert(std::is_base_of<B, T>::value);
    }

    PolyArrayAccessor(std::shared_ptr<B>* ptr, size_t size)
        : _ptr(ptr)
        , _size(size)
    {
        SNN_ASSERT(ptr && size > 0);
        static_assert(std::is_base_of<B, T>::value);
    }

    template<typename U, typename V>
    PolyArrayAccessor(PolyArrayAccessor<U, B, V>& that)
        : _ptr(that.data())
        , _size(that.size())
    {
        static_assert(std::is_base_of<B, T>::value);
    }

    template<typename U, typename V>
    PolyArrayAccessor(PolyArrayAccessor<U, B, V>&& that)
        : _ptr(that.data())
        , _size(that.size())
    {
        static_assert(std::is_base_of<B, T>::value);
    }

    template<typename U, typename V>
    PolyArrayAccessor& operator = (PolyArrayAccessor<U, B, V>& that)
    {
        _ptr = that.data();
        _size = that.size();
        return *this;
    }

    template<typename U, typename V>
    PolyArrayAccessor& operator = (PolyArrayAccessor<U, B, V>&& that)
    {
        _ptr = that.data();
        _size = that.size();
        return *this;
    }

    std::shared_ptr<B>* data() { return _ptr; }

    bool empty() const { return (_size == 0); }

    bool operator()() const { return _ptr; }

    auto size() const -> size_t { return _size; }

    auto operator[](size_t index) const -> const T& {
        SNN_ASSERT(index < _size);
        SNN_ASSERT(C::checkType(_ptr[index].get()));
        return *(static_cast<T*>(_ptr[index].get()));
    }

    auto operator[](size_t index) -> T& {
        SNN_ASSERT(index < _size);
        SNN_ASSERT(C::checkType(_ptr[index].get()));
        return *(static_cast<T*>(_ptr[index].get()));
    }

protected:
    std::shared_ptr<B>* _ptr;
    size_t _size = 0;
};

template<typename T, typename B>
struct PolyArrayAllocator {
    std::shared_ptr<B>* allocate(size_t n) {
        auto ptr = new std::shared_ptr<B>[n];
        for (size_t i = 0; i < n; ++i) {
            ptr[i].reset(new T());
        }
        return ptr;
    }

    void deallocate(std::shared_ptr<B>* ptr, size_t) {
        delete[] ptr;
    }
};

// Array with polymorphic access to the elements
// Does not own the underlying array
template<typename T, typename B = T, typename C = NoCheck, typename A = PolyArrayAllocator<T, B>>
class PolyArray : public PolyArrayAccessor<T, B, C> {
public:
    PolyArray(A a = A())
        : _allocator(a)
    {}

    PolyArray(size_t n, A a = A())
        : _allocator(a)
    {
        allocate(n);
    }

    // Wrap existing shared pointer into array of 1 element
    PolyArray(std::shared_ptr<B> bptr, A a = A())
        : _allocator(a)
    {
        this->_ptr = new std::shared_ptr<B>[1];
        this->_ptr[0] = bptr;
        this->_size = 1U;
    }

    SNN_NO_COPY(PolyArray);

    // can move
    PolyArray(PolyArray&& that)
        : PolyArrayAccessor<T, B, C>(that._ptr, that._size)
        , _allocator(that._allocator)
    {
        that._ptr  = nullptr;
        that._size = 0;
    }

    PolyArray& operator=(PolyArray&& that) {
        if (this != &that) {
            deallocate();
            this->_ptr       = that._ptr;
            that._ptr  = nullptr;
            this->_size      = that._size;
            that._size = 0;
            this->_allocator = that._allocator;
        }
        return *this;
    }

    ~PolyArray() { deallocate(); }

    void allocate(size_t n) {
        if (n == this->_size) {
            return;
        }
        deallocate();
        this->_ptr = _allocator.allocate(n);
        this->_size = n;
    }

    void allocate(size_t n, A a) {
        _allocator = a;
        allocate(n);
    }

    // Wrap existing shared pointer into array of 1 element
    void allocate(std::shared_ptr<B> bptr) {
        deallocate();
        this->_ptr = new std::shared_ptr<B>[1];
        this->_ptr[0] = bptr;
        this->_size = 1U;
    }

    void deallocate() {
        if (this->_ptr) {
            _allocator.deallocate(this->_ptr, this->_size);
            this->_ptr = nullptr;
        }
        this->_size = 0;
    }

private:
    A _allocator;
};

// Utility template class to compute simple statistics
template<typename T>
struct Averager {
    T low, high, sum, average;
    size_t count;
    Averager() { reset(); }

    void reset() {
        low     = T {0};
        high    = T {0};
        sum     = T {0};
        average = T {0};
        count   = 0;
    }

    void update(T value) {
        if (count++ == 0) {
            low = high = sum = average = value;
        } else {
            low = std::min(low, value);
            high = std::max(high, value);
            sum += value;

            average = sum / count;
        }
    }
};

// This class computes timing for hierarchical calls
class Timer {
public:
    const uint64_t id;

private:
    typedef Averager<std::chrono::high_resolution_clock::duration> AveragerHR;
    struct AveragerMs: public AveragerHR {
        double m2 = 0;  // sum of squares of differences from current mean in milliseconds

        void reset();

        void update(std::chrono::high_resolution_clock::duration value);

        // See Welford's online algorithm
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        double getStdDev() const {
            return count == 0 ? 0.0 : sqrt(m2 / count);
        }
    };

    std::string name;
    int nestedLevel = 0;
    std::chrono::high_resolution_clock::time_point begin = {};

    thread_local static uint64_t counter;

    AveragerMs ave = {};

    struct CallNode {
        CallNode(uint64_t id_, std::string n)
            : id(id_)
            , name(n)
        {}

        const uint64_t id;
        std::string name;
        std::chrono::high_resolution_clock::time_point begin = {};
        AveragerMs ave = {};
        std::unordered_map<uint64_t, std::shared_ptr<CallNode>> children;

        void reset();

        void print(std::stringstream& s, const size_t numLevels) const;

        void print(std::stringstream& s, size_t nameColumnWidth, size_t level, const size_t numLevels,
            std::chrono::high_resolution_clock::duration parentSum) const;
    };

    std::shared_ptr<CallNode> parentCall;

    thread_local static std::shared_ptr<CallNode> rootCall;
    thread_local static std::shared_ptr<CallNode> currentCall;
    thread_local static std::unordered_map<uint64_t, Timer*>* allTimers;

public:
    Timer(const std::string& n);

    ~Timer();

    SNN_NO_COPY(Timer);
    SNN_NO_MOVE(Timer);

    const std::string& getName() const {
        return name;
    }

    void start();

    void stop();

    static void reset();

    static std::string print(size_t numLevels = 0, bool drillDownByRank = false);

#ifdef PROFILING
    static double getStdDev(const AveragerHR& a);
#endif
};

// Front-end for the Timer class.
// Calls start() at the scope entrance
// and calls end() at the scope exit
template<typename TIMER>
class ScopedTimer {
    TIMER& _timer;

public:
    SNN_NO_COPY(ScopedTimer);

    explicit ScopedTimer(TIMER& t): _timer(t) { _timer.start(); }

    ~ScopedTimer() { _timer.stop(); }
};

// Helper macro to profile calls
// in the current scope
#define PROFILE_TIME(name, desc) \
    static Timer timer##name(desc); \
    ScopedTimer scopedTimer##name(timer##name);


// Helper classs for ranges
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

// Removes last token from the string
// and separates all tokens by single space
std::string normalizeName(const std::string& name);

FILE* createFile(const char* path);

bool createDirIfNotExists(const std::string& path);

bool createParentDirIfNotExists(std::string path);

// Prints image pixel buffer in human readable format
void prettyPrintHWCBuf(const uint8_t* buffer, int h, int w, int c, ColorFormat cf, FILE* fp = stdout);

} // namespace snn
