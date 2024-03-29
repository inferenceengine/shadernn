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

#include "uvkc/base/log.h"

#include <iostream>

namespace uvkc {

Logger &GetNullLogger() {
  static Logger logger(nullptr);
  return logger;
}

static Logger* gExternalStreamLogger = nullptr;

void SetExternalStreamLogger(Logger* logger) {
  gExternalStreamLogger = logger;
}

Logger &GetErrorLogger() {
  if (gExternalStreamLogger) {
    return *gExternalStreamLogger;
  }
  static Logger logger(&std::clog);
  return logger;
}

PFUN_LOG gExternalLogger = nullptr;
PFUN_IS_LOGGABLE gExternalIsLoggable = nullptr;

void SetExternalLogger(PFUN_LOG logger, PFUN_IS_LOGGABLE loggable) {
  gExternalLogger = logger;
  gExternalIsLoggable = loggable;
}

}  // namespace uvkc
