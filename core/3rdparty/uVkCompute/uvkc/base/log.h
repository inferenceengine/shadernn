// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UVKC_LOG_H_
#define UVKC_LOG_H_

#include <ostream>
#include <sstream>

namespace uvkc {

class Logger;

// Returns the logger that discards all messages.
Logger &GetNullLogger();

// Returns the logger that writes messages to std::clog.
Logger &GetErrorLogger();

// A simple logger that writes messages to an output stream if not null.
//
// This logger uses standard I/O so it should only be used in binaries.

class Logger {
 public:
  friend Logger &GetNullLogger();
  friend Logger &GetErrorLogger();

  template <class T>
  Logger &operator<<(const T &content);

  bool log(const char* file, int line, const char* function, int severity, const char* format, ...);

 protected:
  explicit Logger(std::ostream *stream) : stream_(stream) {}

  // Disable copy construction and assignment
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  virtual void logInternal() {};

  std::ostream *stream_;
};

template <class T>
Logger &Logger::operator<<(const T &content) {
  if (stream_) {
    *stream_ << content;
    logInternal();
  }

  return *this;
}

// Original stream-based uvkc logging implementation does not support logging for Android and works for errors logging only.
// This is to enable stream-based uvkc logging on Android
void SetExternalStreamLogger(Logger* logger);

typedef void (*PFUN_LOG)(const char* file, int line, const char* function, int severity, int androidSeverity, const char* format, ...);
extern PFUN_LOG gExternalLogger;
typedef bool (*PFUN_IS_LOGGABLE)(int severity, int& androidSeverity, const char* file);
extern PFUN_IS_LOGGABLE gExternalIsLoggable;

// This is to enable normal SNN logging
void SetExternalLogger(PFUN_LOG logger, PFUN_IS_LOGGABLE loggable);

enum class LogSeverity {
    FATAL = 0,
    ERR,
    WARN,
    INFO,
    DEBUG,
    VERBOSE,
};

} // uvkc

#define UVKC_LOG(sev, ...) \
        do { \
          int severity = (int)uvkc::LogSeverity::sev; \
          static int androidSeverity = 0; \
          static bool loggable = (gExternalIsLoggable && gExternalLogger) ? (*gExternalIsLoggable)(severity, androidSeverity, __FILE__) : false; \
          if (loggable) { \
            (*gExternalLogger)(__FILE__, __LINE__, __FUNCTION__, severity, androidSeverity, __VA_ARGS__); \
          } \
        } while (0)

#define UVKC_LOGE(...) UVKC_LOG(ERR, __VA_ARGS__)
#define UVKC_LOGW(...) UVKC_LOG(WARN, __VA_ARGS__)
#define UVKC_LOGI(...) UVKC_LOG(INFO, __VA_ARGS__)
#define UVKC_LOGD(...) UVKC_LOG(DEBUG, __VA_ARGS__)
#define UVKC_LOGV(...) UVKC_LOG(VERBOSE, __VA_ARGS__)

#endif  // UVKC_LOG_H_
