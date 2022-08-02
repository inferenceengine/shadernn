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
// Header for commonly used project wide macros.
#pragma once

#ifdef SNN_STATIC
    #define SNN_API
#else
    #ifdef _MSC_VER
        #ifdef SNN_CORE_IMPL
            #define SNN_API __declspec(dllexport)
        #else
            #define SNN_API __declspec(dllimport)
        #endif
        #pragma warning(disable : 4201) // nameless struct/union
    #elif defined(__GNUC__) && defined(SNN_CORE_IMPL)
        #define SNN_API __attribute__((visibility("default")))
    #else
        #define SNN_API
    #endif
#endif

// disable copy semantic of a class.
#define SNN_NO_COPY(X)                                                                                                                                         \
    X(const X&) = delete;                                                                                                                                      \
    X& operator=(const X&) = delete;

// disable move semantic of a class.
#define SNN_NO_MOVE(X)                                                                                                                                         \
    X(X&&)     = delete;                                                                                                                                       \
    X& operator=(X&&) = delete;

// enable default move semantics
#define SNN_DEFAULT_MOVE(X)                                                                                                                                    \
    X(X&&)     = default;                                                                                                                                      \
    X& operator=(X&&) = default;

#define SNN_NO_COPY_CAN_MOVE(X) SNN_NO_COPY(X) SNN_DEFAULT_MOVE(X)

// disable known 'harmless' warnings.
#ifdef _MSC_VER
    #pragma warning(disable : 4458) // declaration hides class member
#endif
