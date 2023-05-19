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
#include "snn/utils.h"
#include "snn/snn.h"
#include "snn/colorUtils.h"
#include "snn/image.h"
#include <stdarg.h>
#include <algorithm>
#include <functional>
#include <string>
#include <cstring>
#include <sstream>
#include <filesystem>
#include <stdlib.h>
#include <unordered_set>
#include <iostream>
#ifdef __ANDROID__
    #include <android/log.h>
    #include <unwind.h>
    #include <dlfcn.h>
    #include <cxxabi.h>
    #include <android/asset_manager.h>
    #include <set>
    #include <sys/stat.h>
    #include <libgen.h>
#ifndef SYSTEM_LIB
    /* removing android dependencies */
    AAssetManager* g_assetManager = nullptr;
#endif  // SYSTEM_LIB
#else
    #include <chrono>
    #include <cstdio>
    #include <cmrc/cmrc.hpp>
    #include <experimental/filesystem>
    CMRC_DECLARE(snn);
#endif

#ifdef _WIN32
    #include <windows.h>
#endif

#ifndef __has_include
static_assert(false, "__has_include not supported");
#else
    #if __cplusplus >= 201703L && __has_include(<filesystem>)
        #include <filesystem>
namespace fs = std::filesystem;
    #elif __has_include(<experimental/filesystem>)
        #include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
    #elif __has_include(<boost/filesystem.hpp>)
        #include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
    #endif
#endif

using namespace snn;

// -----------------------------------------------------------------------------
// setup and restore console color
class ConsoleColor {
#ifdef _WIN32
    HANDLE _console;
    WORD _attrib;

public:
    FILE* _file;

public:
    ConsoleColor(LogSeverity level) {
        // store console attributes
        _console = GetStdHandle((level >= LogSeverity::INFO) ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
        _file    = (level >= LogSeverity::INFO) ? stdout : stderr;
        CONSOLE_SCREEN_BUFFER_INFO csbf;
        GetConsoleScreenBufferInfo(_console, &csbf);
        _attrib = csbf.wAttributes;

        // change console color
        WORD attrib;
        switch (level) {
        case LogSeverity::FATAL:
        case LogSeverity::ERR:
            attrib = FOREGROUND_RED;
            break;

        case LogSeverity::WARN:
            attrib = FOREGROUND_RED | FOREGROUND_GREEN;
            break;

        case LogSeverity::INFO:
            attrib = FOREGROUND_GREEN;
            break;

        default:
            attrib = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
            break;
        }
        SetConsoleTextAttribute(_console, attrib);
    }

    ~ConsoleColor() {
        // restore console attributes
        SetConsoleTextAttribute(_console, _attrib);
    }

#else

public:
    ConsoleColor(LogSeverity) {}
    ~ConsoleColor() {}
#endif
};

float snn::convertToMediumPrecision(float in) {
    union tmp {
        unsigned int unsint;
        float flt;
    };

    tmp _16to32, _32to16;

    _32to16.flt = in;

    unsigned short sign     = (_32to16.unsint & 0x80000000) >> 31;
    unsigned short exponent = (_32to16.unsint & 0x7F800000) >> 23;
    unsigned int mantissa   = _32to16.unsint & 0x7FFFFF;

    short newexp = exponent + (-127 + 15);

    unsigned int newMantissa;
    if (newexp >= 31) {
        newexp      = 31;
        newMantissa = 0x00;
    } else if (newexp <= 0) {
        newexp      = 0;
        newMantissa = 0;
    } else {
        newMantissa = mantissa >> 13;
    }

    if (newexp == 0) {
        if (newMantissa == 0) {
            _16to32.unsint = sign << 31;
        } else {
            newexp = 0;
            while ((newMantissa & 0x200) == 0) {
                newMantissa <<= 1;
                newexp++;
            }
            newMantissa <<= 1;
            newMantissa &= 0x3FF;
            _16to32.unsint = (sign << 31) | ((newexp + (-15 + 127)) << 23) | (newMantissa << 13);
        }
    } else if (newexp == 31) {
        _16to32.unsint = (sign << 31) | (0xFF << 23) | (newMantissa << 13);
    } else {
        _16to32.unsint = (sign << 31) | ((newexp + (-15 + 127)) << 23) | (newMantissa << 13);
    }

    return _16to32.flt;
}

float snn::convertToHighPrecision(uint16_t in) {
    union tmp {
        unsigned int unsint;
        float flt;
    };

    unsigned short sign     = (in & 0x8000) >> 15;
    unsigned short exponent = (in & 0x7C00) >> 10;
    unsigned int mantissa   = (in & 0x3FF);

    tmp _16to32;

    short newexp             = exponent + 127 - 15;
    unsigned int newMantissa = mantissa;
    if (newexp <= 0) {
        newexp      = 0;
        newMantissa = 0;
    }

    _16to32.unsint = (sign << 31) | (newexp << 23) | (newMantissa << 13);
    return _16to32.flt;
}

void snn::convertToMediumPrecision(std::vector<float>& in) {
    for (auto& val : in) {
        val = convertToMediumPrecision(val);
    }
}

void snn::convertToMediumPrecision(std::vector<double>& in) {
    for (auto& val : in) {
        val = convertToMediumPrecision(val);
    }
}

void snn::getByteRepresentation(float in, std::vector<unsigned char>& byteRep, bool fp16) {
    if (fp16) {
        union tmp {
            unsigned int unsint;
            float flt;
        };
        tmp _32to16;
        _32to16.flt = in;
        unsigned short fp16Val;

        unsigned short sign     = (_32to16.unsint & 0x80000000) >> 31;
        unsigned short exponent = (_32to16.unsint & 0x7F800000) >> 23;
        unsigned int mantissa   = (_32to16.unsint & 0x7FFFFF);

        short newexp = exponent + (-127 + 15);

        unsigned int newMantissa;
        if (newexp >= 31) {
            newexp      = 31;
            newMantissa = 0x00;
        } else if (newexp <= 0) {
            newexp      = 0;
            newMantissa = 0;
        } else {
            newMantissa = mantissa >> 13;
        }

        fp16Val = sign << 15;
        fp16Val = fp16Val | (newexp << 10);
        fp16Val = fp16Val | (newMantissa);

        byteRep.push_back(fp16Val & 0xFF);
        byteRep.push_back(fp16Val >> 8 & 0xFF);
    } else {
        uint32_t fp32Val;
        std::memcpy(&fp32Val, &in, 4);
        for (int i = 0; i < 4; i++) {
            byteRep.push_back((fp32Val >> 8 * i) & 0xFF);
        }
    }
}

static constexpr const char* SNN_LOG_LEVEL = "SNN_LOG_LEVEL";
static constexpr const char* SNN_LOG_PER_TAG_LEVEL = "SNN_LOG_PER_TAG_LEVEL";
static constexpr const char* SNN_LOG_TAGS = "SNN_LOG_TAGS";

static int getEnvLogSeverity(bool forTags) {
    char* envLogSeverity = getenv(forTags ? SNN_LOG_PER_TAG_LEVEL : SNN_LOG_LEVEL);
    if (!envLogSeverity) {
        return (int)LogSeverity::INFO;
    }
    if (strcmp(envLogSeverity, "FATAL") == 0) {
        return (int)LogSeverity::FATAL;
    }
    if (strcmp(envLogSeverity, "ERR") == 0) {
        return (int)LogSeverity::ERR;
    }
    if (strcmp(envLogSeverity, "WARN") == 0) {
        return (int)LogSeverity::WARN;
    }
    if (strcmp(envLogSeverity, "WARN") == 0) {
        return (int)LogSeverity::WARN;
    }
    if (strcmp(envLogSeverity, "INFO") == 0) {
        return (int)LogSeverity::INFO;
    }
    if (strcmp(envLogSeverity, "DEBUG") == 0) {
        return (int)LogSeverity::DEBUG;
    }
    if (strcmp(envLogSeverity, "VERBOSE") == 0) {
        return (int)LogSeverity::VERBOSE;
    }
    std::cerr << "Unrecognized SNN_LOG_LEVEL environment variable value: " << envLogSeverity << " !";
    return (int)LogSeverity::INFO;
}

std::unordered_set<std::string> getEnvLogTags() {
    std::unordered_set<std::string> tagFilter;
    char* envLogTags = getenv(SNN_LOG_TAGS);
    if (!envLogTags) {
        return tagFilter;
    }
    while (*envLogTags) {
        // Filter leading spaces if any
        while (*envLogTags == ' ') {
            ++envLogTags;
        }
        if (!*envLogTags) {
            break;
        }

        char* posSpace = strchr(envLogTags, ' ');
        if (posSpace) {
            std::string tag;
            tag.append(envLogTags, posSpace);
            tagFilter.insert(tag);
            envLogTags = posSpace + 1;
        } else {
            // Last token
            tagFilter.insert(envLogTags);
            break;
        }
    }
    return tagFilter;
}

static int getLogSeverity(bool forTags) {
#if defined(__ANDROID__)
    return (int) LogSeverity::VERBOSE;
#endif
    if (!forTags) {
        static int logSeverity = getEnvLogSeverity(false);
        return logSeverity;
    }  else {
        static int logSeverityForTags = getEnvLogSeverity(true);
        return logSeverityForTags;
    }
}

#if defined(__ANDROID__)
static constexpr const int ANDROID_LOG_SEVERITY[] = {
    ANDROID_LOG_FATAL,    // FATAL
    ANDROID_LOG_ERROR,    // ERR
    ANDROID_LOG_WARN,     // WARN
    ANDROID_LOG_INFO,     // INFO
    ANDROID_LOG_DEBUG,    // DEBUG
    ANDROID_LOG_VERBOSE   // VERBOSE
};
#endif

static constexpr const char* SNN = "SNN";

// If severity <= SNN_LOG_LEVEL, then the line will be logged,
// otherwise: if file name is in the SNN_LOG_TAGS list AND severity <= SNN_LOG_PER_TAG_LEVEL, then the line will be logged,
// otherwise: it will be not.
// Examples:
// Linux:
// export SNN_LOG_LEVEL=INFO
// export SNN_LOG_PER_TAG_LEVEL=DEBUG
// export SNN_LOG_TAGS="core vulkanRenderpass"
// Android:
// adb shell setprop log.tag.SNN INFO
// adb shell setprop log.tag.core DEBUG
// adb shell setprop log.tag.vulkanRenderpass DEBUG
bool snn::isLoggable(int severity, int& androidSeverity, const char* file) {
#if !defined(__ANDROID__)
    (void) androidSeverity;
    // Application-wide logging
    if (severity <= getLogSeverity(false)) {
        return true;
    }
    // Logging for tags
    if (severity > getLogSeverity(true)) {
        return false;
    }
#else
    androidSeverity = ANDROID_LOG_SEVERITY[(int)severity];
    // Default application-wide logging is INFO
    int loggable = __android_log_is_loggable(androidSeverity, SNN, ANDROID_LOG_INFO);
    if (loggable) {
        return true;
    }
#endif
    const char* fileBase = file;
    if (const char* pos = strrchr(fileBase, '/')) {
        fileBase = pos + 1;
    }
    const char* fileBaseNameOnly = fileBase;
    std::string tag;
    if (const char* pos = strchr(fileBaseNameOnly, '.')) {
        tag.append(fileBaseNameOnly, pos);
    } else {
        tag = fileBaseNameOnly;
    }
#if !defined(__ANDROID__)
    static const std::unordered_set<std::string> tagFilter = getEnvLogTags();
    if (tagFilter.find(tag) != tagFilter.end()) {
        return true;
    }
#else
    // Default per-file logging is FATAL
    loggable = __android_log_is_loggable(androidSeverity, tag.c_str(), ANDROID_LOG_FATAL);
    if (loggable) {
        // This is to allow logging in __android_log_print()
        androidSeverity = (int)ANDROID_LOG_INFO;
        return true;
    }
#endif
    return false;
}

static const size_t FORMAT_BUFFER_SIZE = 16 * 1024;

// -----------------------------------------------------------------------------
//
void snn::log(const char* file, int line, const char* function, int severity, int androidSeverity, const char* format, ...) {
#if !defined(__ANDROID__)
    (void) androidSeverity;
#endif
    if (0 == format || 0 == *format) {
        return;
    }
    if (severity >= (int) LogSeverity::INFO) {
        // Logging only base file name
        if (const char* pos = strrchr(file, '/')) {
            file = pos + 1;
        }
    }

    va_list args;
    va_start(args, format);
    thread_local static std::vector<char> buf1(FORMAT_BUFFER_SIZE);
    while (std::vsnprintf(buf1.data(), buf1.size(), format, args) >= (int) buf1.size()) {
        buf1.resize(buf1.size() * 2);
    }
    va_end(args);

    thread_local static std::vector<char> buf2(FORMAT_BUFFER_SIZE);

#if defined(__ANDROID__)
    #define FILE_LINE_FORMAT "[%s:%d] "
#else
    #define FILE_LINE_FORMAT "%s(%d): "
#endif

#if !defined(__ANDROID__)
    // Logging tag explicitly
    const char* tag;
    switch (severity) {
    case (int) LogSeverity::FATAL:
        tag = "F/SNN";
        break;
    case (int) LogSeverity::ERR:
        tag = "E/SNN";
        break;
    case (int) LogSeverity::WARN:
        tag = "W/SNN";
        break;
    case (int) LogSeverity::INFO:
        tag = "I/SNN";
        break;
    case (int) LogSeverity::DEBUG:
        tag = "D/SNN";
        break;
    case (int) LogSeverity::VERBOSE:
        tag = "V/SNN";
        break;
    default:
        tag = "";
        break;
    }
    if (severity < (int) LogSeverity::INFO) {
        // Formatting error messages to jump to the code line by click
        while (std::snprintf(buf2.data(), buf2.size(), FILE_LINE_FORMAT "(%s)\n\t%s %s\n", file, line, function, tag, buf1.data()) >= (int) buf2.size()) {
            buf2.resize(buf2.size() * 2);
        }
    } else {
        // More compact logging message
        while (std::snprintf(buf2.data(), buf2.size(), "%s %s(%d): (%s) %s\n", tag, file, line, function, buf1.data()) >= (int) buf2.size()) {
            buf2.resize(buf2.size() * 2);
        }
    }
#else
    // Android will add tag in the log message explicitly
    if (severity < (int) LogSeverity::INFO) {
        // Formatting error messages to jump to the code line by click
        while (std::snprintf(buf2.data(), buf2.size(), FILE_LINE_FORMAT "(%s)\n\t%s\n", file, line, function, buf1.data()) >= (int) buf2.size()) {
            buf2.resize(buf2.size() * 2);
        }
    } else {
        // More compact logging message
        while (std::snprintf(buf2.data(), buf2.size(), "%s(%d): (%s) %s\n", file, line, function, buf1.data()) >= (int) buf2.size()) {
            buf2.resize(buf2.size() * 2);
        }
    }
#endif

#ifdef _WIN32
    ConsoleColor cc((LogSeverity) severity);
    OutputDebugStringA(buf2.data());
    fprintf(cc._file, "%s", buf2.data());
#elif defined(__ANDROID__)
    size_t n         = buf2.size();
    if (n > 1024) {
        // break message into lines
        std::istringstream text(buf2.data());
        for (std::string line; std::getline(text, line);) {
            __android_log_print(androidSeverity, SNN, "%s", line.c_str());
        }
    } else
        __android_log_print(androidSeverity, SNN, "%s", buf2.data());
#elif __linux__
    auto output = severity > (int) LogSeverity::INFO ? stderr : stdout;
    fprintf(output, "%s", buf2.data());
#else
    #error "Unsupported OS."
#endif
}

// -----------------------------------------------------------------------------
//
[[noreturn]] void snn::rip() {
#ifdef _WIN32
    if (::IsDebuggerPresent()) {
        ::DebugBreak();
    } else {
        throw std::runtime_error("Fatal error. RIP...");
    }
#else
    throw std::runtime_error("Fatal error. RIP...");
#endif
}

// -----------------------------------------------------------------------------
//
std::string snn::formatString(const char* format, ...) {
    va_list args;
    va_start(args, format);
    thread_local static std::vector<char> buf1(FORMAT_BUFFER_SIZE);
    while (std::vsnprintf(buf1.data(), buf1.size(), format, args) > (int) buf1.size()) {
        buf1.resize(buf1.size() * 2);
    }
    va_end(args);
    return buf1.data();
}

// -----------------------------------------------------------------------------
//
std::string snn::dumpCallStack(int indent) {
#ifdef __ANDROID__
    struct android_backtrace_state {
        void** current;
        void** end;

        static _Unwind_Reason_Code android_unwind_callback(struct _Unwind_Context* context, void* arg) {
            android_backtrace_state* state = (android_backtrace_state*) arg;
            uintptr_t pc                   = _Unwind_GetIP(context);
            if (pc) {
                if (state->current == state->end) {
                    return _URC_END_OF_STACK;
                } else {
                    *state->current++ = reinterpret_cast<void*>(pc);
                }
            }
            return _URC_NO_REASON;
        }
    };

    std::string prefix;
    for (int i = 0; i < indent; ++i) {
        prefix += ' ';
    }

    std::stringstream ss;
    ss << prefix << "android stack dump\n";

    const int max = 100;
    void* buffer[max];

    android_backtrace_state state;
    state.current = buffer;
    state.end     = buffer + max;

    _Unwind_Backtrace(android_backtrace_state::android_unwind_callback, &state);

    int count = (int) (state.current - buffer);

    for (int idx = 0; idx < count; idx++) {
        const void* addr   = buffer[idx];
        const char* symbol = "<no symbol>";

        Dl_info info;
        if (dladdr(addr, &info) && info.dli_sname) {
            symbol = info.dli_sname;
        }
        int status      = 0;
        char* demangled = __cxxabiv1::__cxa_demangle(symbol, 0, 0, &status);
        ss << prefix << formatString("%03d: 0x%p %s\n", idx, addr, (NULL != demangled && 0 == status) ? demangled : symbol);
        if (NULL != demangled) {
            free(demangled);
        }
    }

    ss << prefix << "android stack dump done\n";

    return ss.str();
#else
    (void) indent;
    return {}; // not implemented on WIN32 yet.
#endif
}

// -----------------------------------------------------------------------------
// Load asset/resources embeded into the executable or package.
std::vector<uint8_t> snn::loadEmbeddedAsset(const char* assetName) {
    SNN_LOGD("Loading asset %s ...", assetName);
#ifdef __ANDROID__
#ifdef SYSTEM_LIB
    // On android load asset using asset manager.

    std::string path("/vendor/oppo/assets/");
    path += assetName;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    return buffer;
#else
        // On android load asset using asset manager.
        #define CR(x, message)                                                                                                                                 \
            if (!(x)) {                                                                                                                                        \
                SNN_LOGE(message);                                                                                                                             \
                return {};                                                                                                                                     \
            } else                                                                                                                                             \
                void(0)

    if (!g_assetManager) {
        std::string path = std::string(ASSETS_DIR) + "/assets/";
        path += assetName;

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size);

        return buffer;
    }

    CR(g_assetManager, "Android asset manager pointer is not set yet.");
    auto asset = AAssetManager_open(g_assetManager, assetName, AASSET_MODE_STREAMING);
    CR(asset, formatString("Asset %s not found.", assetName).c_str());
    size_t size = AAsset_getLength(asset);
    CR(size > 0, formatString("Failed to load asset %s: (asset size = 0).", assetName).c_str());
    std::vector<uint8_t> content(size);
    AAsset_read(asset, content.data(), size);
    AAsset_close(asset);
    return content;
#endif  // SYSTEM_LIB
#else
    // On desktop system, load resouce embeded in the executable.
    auto fs   = cmrc::snn::get_filesystem();
    auto path = std::string("data/assets/") + assetName;
    // SNN_LOGD("%s", path.c_str());
    if (!fs.is_file(path)) {
        SNN_LOGW("image file %s not found in Asset list.", assetName);
        return {};
    }
    auto fd = fs.open(path);
    return {fd.begin(), fd.end()};
#endif  // __ANDROID__
}

std::vector<uint8_t> snn::loadJsonFromStorage(const char* assetName) {
    SNN_LOGI("Loading asset %s ...", assetName);
    // On android load asset using asset manager.

    try {
#ifdef __ANDROID__
        std::string path = std::string(MODEL_DIR);
#else
        std::string path = std::experimental::filesystem::current_path();
        path += ("/" + std::string(MODEL_DIR));
#endif
        path += assetName;
        SNN_LOGD("Loading asset path %s ...", path.c_str());
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        SNN_LOGD("Loading asset size %d ...", size);
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size);

        return buffer;
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::loadEmbeddedAsset : %s", e.what());
        return std::vector<uint8_t>();
    }
}

void snn::convertEndianness(uint8_t* buffer, uint32_t size) {
    for (std::size_t i = 0; i < size; i += 4) {
        std::swap(*(buffer + i), *(buffer + i + 3));
        std::swap(*(buffer + i + 1), *(buffer + i + 2));
    }
}

void snn::convertEndianness(float* buffer, uint32_t size) {
    typedef union {
        uint8_t buffer[4];
        float flt;
    } tempUnion;
    for (std::size_t i = 0; i < size; i++) {
        tempUnion temp;
        temp.flt = *(buffer + i);
        std::swap(temp.buffer[0], temp.buffer[3]);
        std::swap(temp.buffer[1], temp.buffer[2]);
        *(buffer + i) = temp.flt;
    }
}

// -----------------------------------------------------------------------------
// convert nanoseconds (10^-9) to string
std::string snn::ns2s(uint64_t ns) {
    auto us  = ns / 1000ul;
    auto ms  = us / 1000ul;
    auto sec = ms / 1000ul;
    std::stringstream s;
    s << std::fixed << std::setw(5) << std::setprecision(1);
    if (sec > 0) {
        s << (float) (ms) / 1000.0f << " s ";
    } else if (ms > 0) {
        s << (float) (us) / 1000.0f << " ms";
    } else if (us > 0) {
        s << (float) (ns) / 1000.0f << " us";
    } else {
        s << ns << "ns";
    }
    return s.str();
}

// -----------------------------------------------------------------------------
//
static std::string d2s(std::chrono::high_resolution_clock::duration d) {
    using namespace std::chrono;
    return snn::ns2s(duration_cast<nanoseconds>(d).count());
}

// -----------------------------------------------------------------------------
//
thread_local uint64_t snn::Timer::counter = 0;
thread_local std::shared_ptr<snn::Timer::CallNode> snn::Timer::rootCall;
thread_local std::shared_ptr<snn::Timer::CallNode> snn::Timer::currentCall;
thread_local std::unordered_map<uint64_t, Timer*>* snn::Timer::allTimers = new std::unordered_map<uint64_t, Timer*>();

snn::Timer::Timer(const std::string& n)
    : id(counter++)
    , name(n)
{
    (*allTimers)[id] = this;
}

snn::Timer::~Timer() {
    allTimers->erase(id);
}

void snn::Timer::start() {
    ++nestedLevel;

    auto now = std::chrono::high_resolution_clock::now();
    parentCall = currentCall;
    if (parentCall) {
        auto iter = parentCall->children.find(id);
        if (iter != parentCall->children.end()) {
            currentCall = iter->second;
        } else {
            currentCall = std::make_shared<CallNode>(id, name);
            parentCall->children[id] = currentCall;
        }
    } else {
        if (!rootCall || rootCall->id != id) {
            currentCall = std::make_shared<CallNode>(id, name);
            rootCall = currentCall;
        } else {
            currentCall = rootCall;
        }
    }
    currentCall->begin = now;
    if (nestedLevel == 1) {
        begin = now;
    }
}

void snn::Timer::stop() {
    SNN_ASSERT(nestedLevel >= 0);
    SNN_ASSERT(currentCall);

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now - currentCall->begin;
    // Updating statistics for current parent
    currentCall->ave.update(duration);
    currentCall = parentCall;
    parentCall.reset();

    if (nestedLevel == 1) {
        // Updating global statistics
        auto duration = now - begin;
        ave.update(duration);
    }

    --nestedLevel;
}

void snn::Timer::reset() {
    for (auto iter = allTimers->begin(); iter != allTimers->end(); ++iter) {
        iter->second->ave.reset();
    }
    if (rootCall) {
        rootCall->reset();
    }
}

void snn::Timer::CallNode::reset() {
    for (auto iter = children.begin(); iter != children.end(); ++iter) {
        iter->second->reset();
    }
    children.clear();
    ave.reset();
}

void Timer::AveragerMs::reset() {
    AveragerHR::reset();
    m2 = 0.0;
}

// See Welford's online algorithm
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
void Timer::AveragerMs::update(std::chrono::high_resolution_clock::duration value) {
    std::chrono::high_resolution_clock::duration oldAve = average;
    AveragerHR::update(value);
    std::chrono::high_resolution_clock::duration delta = value - oldAve;
    std::chrono::high_resolution_clock::duration delta2 = value - average;
    double deltaMs = std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count() / 1000000.0;
    double delta2Ms = std::chrono::duration_cast<std::chrono::nanoseconds>(delta2).count() / 1000000.0;
    m2 += deltaMs * delta2Ms;
}

std::string snn::Timer::print(size_t numLevels, bool drillDownByRank) {
    std::stringstream s;
    s << std::fixed;
    if (rootCall) {
        rootCall->print(s, numLevels);
    }
    if (drillDownByRank) {
        std::vector<Timer*> sorted;
        for (auto iter = allTimers->begin(); iter != allTimers->end(); ++iter) {
            sorted.push_back(iter->second);
        }
        std::sort(sorted.begin(), sorted.end(), [](Timer* a, Timer* b){
            return a->ave.sum > b->ave.sum;
        });
        size_t nameColumnWidth = 0;
        for (auto & c : sorted) {
            nameColumnWidth = std::max(nameColumnWidth, c->name.size() + 2);
        }
        s << "Downstream calls by rank:" << std::endl;
        for (auto& c : sorted) {
            s << '\t' << std::setw(nameColumnWidth) << std::left << c->name << std::setw(0) << " : "
                << "sum = " << d2s(c->ave.sum)
                << ", calls = " << c->ave.count
                << ", ave = " << d2s(c->ave.average)
                << ", min = " << d2s(c->ave.low)
                << ", max = " << d2s(c->ave.high);
                s << ", stddev = "  << c->ave.getStdDev() << " ms";
            s << std::endl;
        }
    }
    return s.str();
}

void snn::Timer::CallNode::print(std::stringstream& s, const size_t numLevels) const {
    size_t nameColumnWidth = name.size();
    print(s, nameColumnWidth, 0, numLevels, {});
}

void snn::Timer::CallNode::print(std::stringstream& s, size_t nameColumnWidth, size_t level, const size_t numLevels,
    std::chrono::high_resolution_clock::duration parentSum) const {
    if (level >= numLevels) {
        return;
    }
    for (size_t l = 0; l < level; ++l) {
        s << '\t';
    }
    ++level;
    s << std::setw(nameColumnWidth) << std::left << name << std::setw(0) << " : "
        << "sum = " << d2s(ave.sum)
        << ", calls = " << ave.count
        << ", ave = " << d2s(ave.average)
        << ", min = " << d2s(ave.low)
        << ", max = " << d2s(ave.high);
    if (parentSum.count() > 0) {
        s << ", (" << std::setprecision(2) << std::right << ave.sum * 100.0f / parentSum << "%%)";
    }
    s << ", stddev = "  << ave.getStdDev() << " ms";
    s << std::endl;

    std::vector<std::shared_ptr<CallNode>> sorted;
    for (auto & c : children) {
        nameColumnWidth = std::max(nameColumnWidth, c.second->name.size() + 2);
        sorted.push_back(c.second);
    }
    std::sort(sorted.begin(), sorted.end(), [](const std::shared_ptr<CallNode>& a, const std::shared_ptr<CallNode>& b){
        return a->begin < b->begin;
    });
    for (auto& c : sorted) {
        c->print(s, nameColumnWidth, level, numLevels, ave.sum);
    }
}

std::string snn::normalizeName(const std::string& name) {
    std::stringstream splitName(name);
    std::string word;

    std::vector<std::string> tokens;
    while (splitName >> word) {
        tokens.push_back(word);
    }
    tokens.pop_back();

    std::string normName = "";
    for (auto word : tokens) {
        normName += word;
        normName += " ";
    }
    normName.pop_back();
    return normName;
}

bool snn::createDirIfNotExists(const std::string& path) {
#ifdef __ANDROID__
    std::string currPath;
    size_t pos = 0;
    do {
        pos = path.find('/', pos);
        if (pos != std::string::npos) {
            currPath = path.substr(0, pos);
            pos = pos + 1;
            if (currPath.empty()) {
                continue;
            }
        } else {
            currPath = path;
        }
        if (0 != mkdir(currPath.c_str(), 0700)) {
            int errnum = errno;
            if (errnum != EEXIST) {
                const char* errMsg = strerror(errnum);
                SNN_LOGE("mkdir(%s): %s", currPath.c_str(), errMsg);
                return false;
            }
        }
    } while (pos != std::string::npos);
    return true;
#else
    bool res = false;
    try {
        fs::create_directories(path.c_str());
        res = true;
    }
    catch(fs::filesystem_error& err) {
        SNN_LOGE("fs::create_directories(%s): %s", path.c_str(), err.what());
    }
    return res;
#endif
}

FILE* snn::createFile(const char* path) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        int errnum = errno;
        const char* errMsg = strerror(errnum);
        SNN_LOGE("fopen(%s): %s", path, errMsg);
    }
    return fp;
}

bool snn::createParentDirIfNotExists(std::string path) {
    size_t pos = path.rfind("/");
    if (pos != std::string::npos) {
        path = path.substr(0, pos);
    }
    return createDirIfNotExists(path);
}

void snn::prettyPrintHWCBuf(const uint8_t* buffer, int h, int w, int c, snn::ColorFormat cf, FILE* fp) {
    fprintf(fp, "----w: %d, h: %d, c: %d----\n", w, h, c);
    snn::ColorFormatDesc cfd = snn::getColorFormatDesc(cf);
    snn::ColorFormatType cft = snn::getColorFormatType(cf);
    SNN_ASSERT(c % cfd.ch == 0);
    int d = c / cfd.ch;
    size_t bytes = cfd.bytes() * d;
    for (uint32_t q = 0; q < c; q++) {
        const uint8_t* raw_ptr = buffer;
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                switch (cft) {
                    case snn::ColorFormatType::UINT8:
                        {
                            const uint8_t* ptr = raw_ptr;
                            fprintf(fp, "%d, ", ptr[q]);
                        }
                        break;
                    case snn::ColorFormatType::UINT16:
                        {
                            const uint16_t* ptr = reinterpret_cast<const uint16_t*>(raw_ptr);
                            fprintf(fp, "%d, ", ptr[q]);
                        }
                        break;
                    case snn::ColorFormatType::FLOAT16:
                        {
                            const snn::FP16* ptr = reinterpret_cast<const snn::FP16*>(raw_ptr);
                            fprintf(fp, "%f, ", snn::FP16::toFloat(ptr[q].u));
                        }
                        break;
                    case snn::ColorFormatType::FLOAT32:
                        {
                            const float* ptr = reinterpret_cast<const float*>(raw_ptr);
                            fprintf(fp, "%f, ", ptr[q]);
                        }
                        break;
                    default:
                        SNN_CHK(false);
                }
                raw_ptr += bytes;
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "----------%d--------------\n", q);
    }
}
