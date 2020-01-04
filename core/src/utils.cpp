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
#include "pch.h"
#include "snn/utils.h"
#include <stdarg.h>
#include <functional>
#include <cstring>
#include <sstream>
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
    #endif
#else
    #include <chrono>
    #include <cstdio>
    #include <cmrc/cmrc.hpp>
    #include <string>
    #include <experimental/filesystem>
CMRC_DECLARE(snn);
#endif

#ifdef _WIN32
    #include <windows.h>
#endif

using namespace std;
using namespace snn;

SNN_API int snn::g_enabledLogSeverity = (int) snn::LogSeverity::INFO;

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
        val = snn::convertToMediumPrecision(val);
    }
}

void snn::convertToMediumPrecision(std::vector<double>& in) {
    for (auto& val : in) {
        val = snn::convertToMediumPrecision(val);
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

static const size_t FORMAT_BUFFER_SIZE = 16 * 1024;

// -----------------------------------------------------------------------------
//
void snn::log(const char* file, int line, const char* function, int severity, const char* format, ...) {
    if (0 == format || 0 == *format) {
        return;
    }

    va_list args;
    va_start(args, format);
    thread_local static std::vector<char> buf1(FORMAT_BUFFER_SIZE);
    while (std::vsnprintf(buf1.data(), buf1.size(), format, args) >= (int) buf1.size()) {
        buf1.resize(buf1.size() * 2);
    }
    va_end(args);

    thread_local static std::vector<char> buf2(FORMAT_BUFFER_SIZE);

#ifdef __ANDROID__
    #define FILE_LINE_FORMAT "[%s:%d] "
#else
    #define FILE_LINE_FORMAT "%s(%d): "
#endif

    const char* tag;
    switch (severity) {
    case (int) LogSeverity::FATAL:
        tag = "[SNNLOG] [FATAL]";
        break;
    case (int) LogSeverity::ERR:
        tag = "[SNNLOG] [ERROR]";
        break;
    case (int) LogSeverity::WARN:
        tag = "[SNNLOG] [WARNING]";
        break;
    case (int) LogSeverity::INFO:
        tag = "[SNNLOG] [INFO]";
        break;
    case (int) LogSeverity::DEBUG:
        tag = "[SNNLOG] [DEBUG]";
        break;
    case (int) LogSeverity::VERBOSE:
        tag = "[SNNLOG] [VERBOSE]";
        break;
    default:
        tag = "";
        break;
    }

    if ((int) LogSeverity::INFO == severity) {
        while (std::snprintf(buf2.data(), buf2.size(), "%s %s\n", tag, buf1.data()) >= (int) buf2.size()) {
            buf2.resize(buf2.size() * 2);
        }
    } else {
        while (std::snprintf(buf2.data(), buf2.size(), FILE_LINE_FORMAT "(%s)\n\t%s %s\n", file, line, function, tag, buf1.data()) >= (int) buf2.size()) {
            buf2.resize(buf2.size() * 2);
        }
    }

#ifdef _WIN32
    ConsoleColor cc((LogSeverity) severity);
    OutputDebugStringA(buf2.data());
    fprintf(cc._file, "%s", buf2.data());
#elif defined(__ANDROID__)
    static int pri[] = {ANDROID_LOG_FATAL, ANDROID_LOG_ERROR, ANDROID_LOG_WARN, ANDROID_LOG_INFO, ANDROID_LOG_DEBUG, ANDROID_LOG_VERBOSE};
    int priority     = pri[severity];
    size_t n         = buf2.size();
    if (n > 1024) {
        // break message into lines
        std::istringstream text(buf2.data());
        for (std::string line; std::getline(text, line);) {
            __android_log_print(priority, "SNN", "%s", line.c_str());
        }
    } else
        __android_log_print(priority, "SNN", "%s", buf2.data());
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
    throw runtime_error("Fatal error. RIP...");
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

    ifstream file(path, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    vector<uint8_t> buffer(size);
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
    CR(g_assetManager, "Android asset manager pointer is not set yet.");
    auto asset = AAssetManager_open(g_assetManager, assetName, AASSET_MODE_STREAMING);
    CR(asset, formatString("Asset %s not found.", assetName).c_str());
    size_t size = AAsset_getLength(asset);
    CR(size > 0, formatString("Failed to load asset %s: (asset size = 0).", assetName).c_str());
    std::vector<uint8_t> content(size);
    AAsset_read(asset, content.data(), size);
    AAsset_close(asset);
    return content;
    #endif
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
#endif
}

std::vector<uint8_t> snn::loadJsonFromStorage(const char* assetName) {
    SNN_LOGD("Loading asset %s ...", assetName);
    // On android load asset using asset manager.

    try {
#ifdef __ANDROID__
        std::string path("/data/local/tmp/");
#else
        std::string path = std::experimental::filesystem::current_path();
        path += std::string("/../../../../modelzoo/");
#endif
        path += assetName;
        // SNN_LOGD("Loading asset path %s ...", path.c_str());
        ifstream file(path, ios::binary | ios::ate);
        streamsize size = file.tellg();
        SNN_LOGD("Loading asset size %d ...", size);
        file.seekg(0, ios::beg);

        vector<uint8_t> buffer(size);
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

thread_local Timer* snn::Timer::root = nullptr;

// -----------------------------------------------------------------------------
//
//snn::Timer::Timer(std::string n): name(n) {}

// -----------------------------------------------------------------------------
//
snn::Timer::~Timer() {}

// -----------------------------------------------------------------------------
// convert nanoseconds (10^-9) to string
std::string snn::ns2s(uint64_t ns) {
    auto us  = ns / 1000ul;
    auto ms  = us / 1000ul;
    auto sec = ms / 1000ul;
    std::stringstream s;
    s << std::fixed << std::setw(5) << std::setprecision(1);
    if (sec > 0) {
        s << (float) (ms) / 1000.0f << "s ";
    } else if (ms > 0) {
        s << (float) (us) / 1000.0f << "ms";
    } else if (us > 0) {
        s << (float) (ns) / 1000.0f << "us";
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
std::string snn::Timer::print(size_t nameColumnWidth, const Timer* parent) const {
    std::stringstream s;
    ave.refreshAverage();
    s << std::setw(nameColumnWidth) << std::left << name << std::setw(0) << " : " << d2s(ave.average) << ", min = " << d2s(ave.low)
        << ", max = " << d2s(ave.high);
    if (parent) {
        if (0 != parent->ave.average.count()) {
            s << ", (" << std::setw(2) << std::right << ave.average * 100 / parent->ave.average << "%%)";
        }
    }
    return s.str();
}
