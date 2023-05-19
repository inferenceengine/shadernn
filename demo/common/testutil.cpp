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
// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "testutil.h"
#include "snn/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if NCNN_VULKAN
    #include "command.h"
    #include "gpu.h"
#endif // NCNN_VULKAN

struct prng_rand_t g_prng_rand_state;

float RandomFloat(float a /*= -1.2f*/, float b /*= 1.2f*/) {
    float random = ((float) RAND()) / (float) uint64_t(-1); // RAND_MAX;
    float diff   = b - a;
    float rd      = random * diff;
    return a + rd;
}

int RandomInt(int a /*= -10000*/, int b /*= 10000*/) {
    float random = ((float) RAND()) / (float) uint64_t(-1); // RAND_MAX;
    int diff     = b - a;
    float rd      = random * diff;
    return a + (int) rd;
}

signed char RandomS8() { return (signed char) RandomInt(-127, 127); }

void Randomize(ncnn::Mat& m, float a /*= -1.2f*/, float b /*= 1.2f*/) {
    for (size_t i = 0; i < m.total(); i++) {
        m[i] = RandomFloat(a, b);
    }
}

void RandomizeInt(ncnn::Mat& m, int a /*= -10000*/, int b /*= 10000*/) {
    for (size_t i = 0; i < m.total(); i++) {
        ((int*) m)[i] = RandomInt(a, b);
    }
}

void RandomizeS8(ncnn::Mat& m) {
    for (size_t i = 0; i < m.total(); i++) {
        ((signed char*) m)[i] = RandomS8();
    }
}

ncnn::Mat RandomMat(int w) {
    ncnn::Mat m(w);
    Randomize(m);
    return m;
}

ncnn::Mat RandomMat(int w, int h) {
    ncnn::Mat m(w, h);
    Randomize(m);
    return m;
}

ncnn::Mat RandomMat(int w, int h, int c) {
    ncnn::Mat m(w, h, c);
    Randomize(m);
    return m;
}

ncnn::Mat RandomMat(int w, int h, int d, int c) {
    ncnn::Mat m(w, h, d, c);
    Randomize(m);
    return m;
}

ncnn::Mat RandomIntMat(int w) {
    ncnn::Mat m(w);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomIntMat(int w, int h) {
    ncnn::Mat m(w, h);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomIntMat(int w, int h, int c) {
    ncnn::Mat m(w, h, c);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomIntMat(int w, int h, int d, int c) {
    ncnn::Mat m(w, h, d, c);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w) {
    ncnn::Mat m(w, (size_t) 1u);
    RandomizeS8(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w, int h) {
    ncnn::Mat m(w, h, (size_t) 1u);
    RandomizeS8(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w, int h, int c) {
    ncnn::Mat m(w, h, c, (size_t) 1u);
    RandomizeS8(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w, int h, int d, int c) {
    ncnn::Mat m(w, h, d, c, (size_t) 1u);
    RandomizeS8(m);
    return m;
}

void SetOne(ncnn::Mat& m, float a /*= -1.2f*/, float b /*= 1.2f*/) {
    (void) a;
    (void) b;
    for (size_t i = 0; i < m.total(); i++) {
        m[i] = 1.0f;
    }
}

void SetOneInt(ncnn::Mat& m, int a /*= -10000*/, int b /*= 10000*/) {
    (void) a;
    (void) b;
    for (size_t i = 0; i < m.total(); i++) {
        ((int*) m)[i] = 1;
    }
}

void SetOneS8(ncnn::Mat& m) {
    for (size_t i = 0; i < m.total(); i++) {
        ((signed char*) m)[i] = 1;
    }
}

ncnn::Mat SetOneMat(int w) {
    ncnn::Mat m(w);
    SetOne(m);
    return m;
}

ncnn::Mat SetOneMat(int w, int h) {
    ncnn::Mat m(w, h);
    SetOne(m);
    return m;
}

ncnn::Mat SetOneMat(int w, int h, int c) {
    ncnn::Mat m(w, h, c);
    SetOne(m);
    return m;
}

ncnn::Mat SetOneMat(int w, int h, int d, int c) {
    ncnn::Mat m(w, h, d, c);
    SetOne(m);
    return m;
}

ncnn::Mat SetOneIntMat(int w) {
    ncnn::Mat m(w);
    SetOneInt(m);
    return m;
}

ncnn::Mat SetOneIntMat(int w, int h) {
    ncnn::Mat m(w, h);
    SetOneInt(m);
    return m;
}

ncnn::Mat SetOneIntMat(int w, int h, int c) {
    ncnn::Mat m(w, h, c);
    SetOneInt(m);
    return m;
}

ncnn::Mat SetOneIntMat(int w, int h, int d, int c) {
    ncnn::Mat m(w, h, d, c);
    SetOneInt(m);
    return m;
}

ncnn::Mat SetOneS8Mat(int w) {
    ncnn::Mat m(w, (size_t) 1u);
    SetOneS8(m);
    return m;
}

ncnn::Mat SetOneS8Mat(int w, int h) {
    ncnn::Mat m(w, h, (size_t) 1u);
    SetOneS8(m);
    return m;
}

ncnn::Mat SetOneS8Mat(int w, int h, int c) {
    ncnn::Mat m(w, h, c, (size_t) 1u);
    SetOneS8(m);
    return m;
}

ncnn::Mat SetOneS8Mat(int w, int h, int d, int c) {
    ncnn::Mat m(w, h, d, c, (size_t) 1u);
    SetOneS8(m);
    return m;
}

void SetValue(ncnn::Mat& m, float v /*= 0.0f*/) {
    for (size_t i = 0; i < m.total(); i++) {
        m[i] = v;
    }
}

void SetValueInt(ncnn::Mat& m, int v /*= 0*/) {
    for (size_t i = 0; i < m.total(); i++) {
        ((int*) m)[i] = v;
    }
}

void SetValueS8(ncnn::Mat& m, signed char v /*= 0*/) {
    for (size_t i = 0; i < m.total(); i++) {
        ((signed char*) m)[i] = v;
    }
}

ncnn::Mat SetValueMat(int w, float v /*= 0.0f*/) {
    ncnn::Mat m(w);
    SetValue(m, v);
    return m;
}

ncnn::Mat SetValueMat(int w, int h, float v /*= 0.0f*/) {
    ncnn::Mat m(w, h);
    SetValue(m, v);
    return m;
}

ncnn::Mat SetValueMat(int w, int h, int c, float v /*= 0.0f*/) {
    ncnn::Mat m(w, h, c);
    SetValue(m, v);
    return m;
}

ncnn::Mat SetValueMat(int w, int h, int d, int c, float v /*= 0.0f*/) {
    ncnn::Mat m(w, h, d, c);
    SetValue(m, v);
    return m;
}

ncnn::Mat SetValueIntMat(int w, int v /*= 0*/) {
    ncnn::Mat m(w);
    SetValueInt(m, v);
    return m;
}

ncnn::Mat SetValueIntMat(int w, int h, int v /*= 0*/) {
    ncnn::Mat m(w, h);
    SetValueInt(m, v);
    return m;
}

ncnn::Mat SetValueIntMat(int w, int h, int c, int v /*= 0*/) {
    ncnn::Mat m(w, h, c);
    SetValueInt(m, v);
    return m;
}

ncnn::Mat SetValueIntMat(int w, int h, int d, int c, int v /*= 0*/) {
    ncnn::Mat m(w, h, d, c);
    SetValueInt(m, v);
    return m;
}

ncnn::Mat SetValueS8Mat(int w, signed char v /*= 0*/) {
    ncnn::Mat m(w, (size_t) 1u);
    SetValueS8(m, v);
    return m;
}

ncnn::Mat SetValueS8Mat(int w, int h, signed char v /*= 0*/) {
    ncnn::Mat m(w, h, (size_t) 1u);
    SetValueS8(m, v);
    return m;
}

ncnn::Mat SetValueS8Mat(int w, int h, int c, signed char v /*= 0*/) {
    ncnn::Mat m(w, h, c, (size_t) 1u);
    SetValueS8(m, v);
    return m;
}

ncnn::Mat SetValueS8Mat(int w, int h, int d, int c, signed char v /*= 0*/) {
    ncnn::Mat m(w, h, d, c, (size_t) 1u);
    SetValueS8(m, v);
    return m;
}

ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx) {
    ncnn::Mat weight_scales(m);
    for (int i = 0; i < m; ++i) {
        float min = mat[0], _max = mat[0];
        const float* ptr = (const float*) (mat.data) + i * ldx;
        for (int j = 0; j < k; ++j) {
            if (min > ptr[j]) {
                min = ptr[j];
            }
            if (_max < ptr[j]) {
                _max = ptr[j];
            }
        }
        const float abs_min = abs(min), abs_max = abs(_max);
        weight_scales[i] = 127.f / (abs_min > abs_max ? abs_min : abs_max);
    }
    return weight_scales;
}

bool NearlyEqual(float a, float b, float epsilon) {
    if (a == b) {
        return true;
    }
    float diff = (float) fabs(a - b);
    if (diff <= epsilon) {
        return true;
    }
    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon /*= 0.001*/) {
#define CHECK_MEMBER(m)                                                                                                                                        \
    if (a.m != b.m) {                                                                                                                                          \
        fprintf(stderr, #m " not match    expect %d but got %d\n", (int) a.m, (int) b.m);                                                                      \
        return -1;                                                                                                                                             \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(w)
    CHECK_MEMBER(h)
    CHECK_MEMBER(d)
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q = 0; q < a.c; q++) {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int z = 0; z < a.d; z++) {
            const ncnn::Mat da = ma.depth(z);
            const ncnn::Mat db = mb.depth(z);
            for (int i = 0; i < a.h; i++) {
                const float* pa = da.row(i);
                const float* pb = db.row(i);
                for (int j = 0; j < a.w; j++) {
                    // printf("value compare:%d,  at c:%d d:%d h:%d w:%d    expect %f but got %f\n", NearlyEqual(pa[j], pb[j], epsilon), q, z, i, j, pa[j],
                    // pb[j]);
                    if (!NearlyEqual(pa[j], pb[j], epsilon)) {
                        fprintf(stderr, "value not match  at c:%d d:%d h:%d w:%d    expect %f but got %f\n", q, z, i, j, pa[j], pb[j]);
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}

int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon /*= 0.001*/) {
    ncnn::Option opt;
    opt.num_threads = 1;

    if (a.elempack != 1) {
        ncnn::Mat a1;
        ncnn::convert_packing(a, a1, 1, opt);
        return CompareMat(a1, b, epsilon);
    }

    if (b.elempack != 1) {
        ncnn::Mat b1;
        ncnn::convert_packing(b, b1, 1, opt);
        return CompareMat(a, b1, epsilon);
    }

    if (a.elemsize == 2u) {
        ncnn::Mat a32;
        cast_float16_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }
    if (a.elemsize == 1u) {
        ncnn::Mat a32;
        cast_int8_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }

    if (b.elemsize == 2u) {
        ncnn::Mat b32;
        cast_float16_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }
    if (b.elemsize == 1u) {
        ncnn::Mat b32;
        cast_int8_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }

    return Compare(a, b, epsilon);
}

int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon /*= 0.001*/) {
    if (a.size() != b.size()) {
        fprintf(stderr, "output blob count not match %zu %zu\n", a.size(), b.size());
        return -1;
    }

    for (size_t i = 0; i < a.size(); i++) {
        if (CompareMat(a[i], b[i], epsilon)) {
            fprintf(stderr, "output blob %zu not match\n", i);
            return -1;
        }
    }

    return 0;
}

bool checkPlatFormSupport(bool useVulkan) {
    (void) useVulkan;
#ifndef SUPPORT_VULKAN
    if (useVulkan) {
        SNN_LOGD("Vulkan is not supported. Test ignored.");
        return false;
    }
#endif
#ifndef SUPPORT_GL
    if (!useVulkan) {
        SNN_LOGD("OpenGL is not supported. Test ignored.");
        return false;
    }
#endif
    return true;
}
