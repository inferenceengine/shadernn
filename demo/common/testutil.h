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

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "prng.h"

#if NCNN_VULKAN
    #include "command.h"
    #include "gpu.h"
#endif // NCNN_VULKAN

extern struct prng_rand_t g_prng_rand_state;

static void dummy_using_of_g_prng_rand_state() {
    (void) g_prng_rand_state;
}

#if NCNN_VULKAN
class GlobalGpuInstance {
public:
    GlobalGpuInstance() { ncnn::create_gpu_instance(); }
    ~GlobalGpuInstance() { ncnn::destroy_gpu_instance(); }
};
    // HACK workaround nvidia driver crash on exit
    #define SRAND(seed)                                                                                                                                        \
        GlobalGpuInstance __ncnn_gpu_instance_guard;                                                                                                           \
        prng_srand(seed, &g_prng_rand_state)
    #define RAND() prng_rand(&g_prng_rand_state)
#else // NCNN_VULKAN
    #define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
    #define RAND()      prng_rand(&g_prng_rand_state)
#endif // NCNN_VULKAN

#define TEST_LAYER_DISABLE_AUTO_INPUT_PACKING (1 << 0)
#define TEST_LAYER_DISABLE_AUTO_INPUT_CASTING (1 << 1)
#define TEST_LAYER_DISABLE_GPU_TESTING        (1 << 2)
#define TEST_LAYER_ENABLE_FORCE_INPUT_PACK8   (1 << 3)

float RandomFloat(float a = -1.2f, float b = 1.2f);

int RandomInt(int a = -10000, int b = 10000);

signed char RandomS8();

void Randomize(ncnn::Mat& m, float a = -1.2f, float b = 1.2f);

void RandomizeInt(ncnn::Mat& m, int a = -10000, int b = 10000);

void RandomizeS8(ncnn::Mat& m);

ncnn::Mat RandomMat(int w);

ncnn::Mat RandomMat(int w, int h);

ncnn::Mat RandomMat(int w, int h, int c);

ncnn::Mat RandomMat(int w, int h, int d, int c);

ncnn::Mat RandomIntMat(int w);

ncnn::Mat RandomIntMat(int w, int h);

ncnn::Mat RandomIntMat(int w, int h, int c);

ncnn::Mat RandomIntMat(int w, int h, int d, int c);

ncnn::Mat RandomS8Mat(int w);

ncnn::Mat RandomS8Mat(int w, int h);

ncnn::Mat RandomS8Mat(int w, int h, int c);

ncnn::Mat RandomS8Mat(int w, int h, int d, int c);

void SetOne(ncnn::Mat& m, float a = -1.2f, float b = 1.2f);

void SetOneInt(ncnn::Mat& m, int a = -10000, int b = 10000);

void SetOneS8(ncnn::Mat& m);

ncnn::Mat SetOneMat(int w);

ncnn::Mat SetOneMat(int w, int h);

ncnn::Mat SetOneMat(int w, int h, int c);

ncnn::Mat SetOneMat(int w, int h, int d, int c);

ncnn::Mat SetOneIntMat(int w);

ncnn::Mat SetOneIntMat(int w, int h);

ncnn::Mat SetOneIntMat(int w, int h, int c);

ncnn::Mat SetOneIntMat(int w, int h, int d, int c);

ncnn::Mat SetOneS8Mat(int w);

ncnn::Mat SetOneS8Mat(int w, int h);

ncnn::Mat SetOneS8Mat(int w, int h, int c);

ncnn::Mat SetOneS8Mat(int w, int h, int d, int c);

void SetValue(ncnn::Mat& m, float v = 0.0f);

void SetValueInt(ncnn::Mat& m, int v = 0);

void SetValueS8(ncnn::Mat& m, signed char v = 0);

ncnn::Mat SetValueMat(int w, float v = 0.0f);

ncnn::Mat SetValueMat(int w, int h, float v = 0.0f);

ncnn::Mat SetValueMat(int w, int h, int c, float v = 0.0f);

ncnn::Mat SetValueMat(int w, int h, int d, int c, float v = 0.0f);

ncnn::Mat SetValueIntMat(int w, int v = 0);

ncnn::Mat SetValueIntMat(int w, int h, int v = 0);

ncnn::Mat SetValueIntMat(int w, int h, int c, int v = 0);

ncnn::Mat SetValueIntMat(int w, int h, int d, int c, int v = 0);

ncnn::Mat SetValueS8Mat(int w, signed char v = 0);

ncnn::Mat SetValueS8Mat(int w, int h, signed char v = 0);

ncnn::Mat SetValueS8Mat(int w, int h, int c, signed char v = 0);

ncnn::Mat SetValueS8Mat(int w, int h, int d, int c, signed char v = 0);

ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx);

bool NearlyEqual(float a, float b, float epsilon);

int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001);

int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001);

int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon = 0.001);

template<typename T>
int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count,
                     std::vector<ncnn::Mat>& b, void (*func)(T*), int flag) {
    (void) flag;
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func) {
        (*func)((T*) op);
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1) {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads             = 1;
    opt.use_packing_layout      = false;
    opt.use_fp16_packed         = false;
    opt.use_fp16_storage        = false;
    opt.use_fp16_arithmetic     = false;
    opt.use_shader_pack8        = false;
    opt.use_image_storage       = false;
    opt.use_bf16_storage        = false;
    opt.use_vulkan_compute      = false;
    opt.use_weight_fp16_storage = false;

    op->create_pipeline(opt);

    b.resize(top_blob_count);

    if (op->support_inplace) {
        for (size_t i = 0; i < a.size(); i++) {
            b[i] = a[i].clone();
        }

        ((T*) op)->T::forward_inplace(b, opt);
    } else {
        ((T*) op)->T::forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a,
                   int top_blob_count, std::vector<ncnn::Mat>& c, const std::vector<ncnn::Mat>& top_shapes, void (*func)(T*), int flag) {
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func) {
        (*func)((T*) op);
    }

    if (!top_shapes.empty()) {
        op->bottom_shapes = a;
        op->top_shapes    = top_shapes;
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1) {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt       = _opt;
    opt.num_threads        = 1;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    std::vector<ncnn::Mat> a4(a.size());

    for (size_t i = 0; i < a4.size(); i++) {
        if (opt.use_fp16_storage && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
            ncnn::cast_float32_to_float16(a[i], a4[i], opt);
        } else if (opt.use_bf16_storage && op->support_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
            ncnn::cast_float32_to_bfloat16(a[i], a4[i], opt);
        } else {
            a4[i] = a[i];
        }

        if (opt.use_packing_layout && op->support_packing && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_PACKING)) {
            // resolve dst_elempack
            int dims      = a4[i].dims;
            int elemcount = 0;
            if (dims == 1) {
                elemcount = a4[i].elempack * a4[i].w;
            }
            if (dims == 2) {
                elemcount = a4[i].elempack * a4[i].h;
            }
            if (dims == 3) {
                elemcount = a4[i].elempack * a4[i].c;
            }
            int elembits = a4[i].elembits();

            int dst_elempack = 1;

            if (elembits == 32) {
#if (NCNN_AVX2 || NCNN_AVX)
                if (elemcount % 8 == 0 && (ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx())) {
                    dst_elempack = 8;
                } else if (elemcount % 4 == 0) {
                    dst_elempack = 4;
                }
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / (elembits / 8);
                if (elemcount % packn == 0) {
                    dst_elempack = packn;
                }
#else
                if (elemcount % 4 == 0) {
                    dst_elempack = 4;
                }
#endif
            }
            if (elembits == 16) {
#if NCNN_ARM82
                if (elemcount % 8 == 0 && opt.use_fp16_storage && opt.use_fp16_arithmetic && op->support_fp16_storage) {
                    dst_elempack = 8;
                } else if (elemcount % 4 == 0) {
                    dst_elempack = 4;
                }
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 2;
                if (elemcount % packn == 0) {
                    dst_elempack = packn;
                }
#else
                if (elemcount % 4 == 0) {
                    dst_elempack = 4;
                }
#endif
            }
            if (elembits == 8) {
#if NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 1;
                if (elemcount % packn == 0) {
                    dst_elempack = packn;
                }
#else
                if (elemcount % 8 == 0) {
                    dst_elempack = 8;
                }
#endif
            }

            if (flag & TEST_LAYER_ENABLE_FORCE_INPUT_PACK8) {
                dst_elempack = 8;
            }

            ncnn::Mat a4_packed;
            ncnn::convert_packing(a4[i], a4_packed, dst_elempack, opt);
            a4[i] = a4_packed;
        }
    }

    c.resize(top_blob_count);

    if (op->support_inplace) {
        for (size_t i = 0; i < a4.size(); i++) {
            c[i] = a4[i].clone();
        }

        op->forward_inplace(c, opt);
    } else {
        op->forward(a4, c, opt);
    }

    for (size_t i = 0; i < c.size(); i++) {
        if (opt.use_fp16_storage && op->support_fp16_storage && c[i].elembits() == 16) {
            ncnn::Mat c_fp32;
            ncnn::cast_float16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        } else if (opt.use_bf16_storage && op->support_bf16_storage && c[i].elembits() == 16) {
            ncnn::Mat c_fp32;
            ncnn::cast_bfloat16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_VULKAN
template<typename T>
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a,
                   int top_blob_count, std::vector<ncnn::Mat>& d, const std::vector<ncnn::Mat>& top_shapes, void (*func)(T*), int flag) {
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (!op->support_vulkan) {
        delete op;
        return 233;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    op->vkdev = vkdev;

    if (func) {
        (*func)((T*) op);
    }

    if (!top_shapes.empty()) {
        op->bottom_shapes = a;
        op->top_shapes    = top_shapes;
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1) {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator    = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt       = _opt;
    opt.num_threads        = 1;
    opt.use_vulkan_compute = true;

    #if __APPLE__
    opt.use_image_storage = false;
    #endif

    opt.blob_vkallocator      = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator   = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) {
        opt.use_fp16_packed = false;
    }
    if (!vkdev->info.support_fp16_storage()) {
        opt.use_fp16_storage = false;
    }
    if (!vkdev->info.support_fp16_arithmetic()) {
        opt.use_fp16_arithmetic = false;
    }
    // FIXME fp16a may produce large error
    opt.use_fp16_arithmetic = false;

    op->create_pipeline(opt);

    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload          = opt;
        opt_upload.blob_vkallocator      = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator   = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }

    d.resize(top_blob_count);

    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (op->support_image_storage && opt.use_image_storage) {
            // upload
            std::vector<ncnn::VkImageMat> a_gpu(a.size());
            for (size_t i = 0; i < a_gpu.size(); i++) {
                cmd.record_upload(a[i], a_gpu[i], opt);
            }

            std::vector<ncnn::VkImageMat> d_gpu(top_blob_count);
            if (op->support_inplace) {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            } else {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            for (size_t i = 0; i < d_gpu.size(); i++) {
                cmd.record_download(d_gpu[i], d[i], opt);
            }
        } else {
            // upload
            std::vector<ncnn::VkMat> a_gpu(a.size());
            for (size_t i = 0; i < a_gpu.size(); i++) {
                cmd.record_upload(a[i], a_gpu[i], opt);
            }

            std::vector<ncnn::VkMat> d_gpu(top_blob_count);
            if (op->support_inplace) {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            } else {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            for (size_t i = 0; i < d_gpu.size(); i++) {
                cmd.record_download(d_gpu[i], d[i], opt);
            }
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    return 0;
}
#endif // NCNN_VULKAN

template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a,
               int top_blob_count, const std::vector<ncnn::Mat>& top_shapes = std::vector<ncnn::Mat>(), float epsilon = 0.001, void (*func)(T*) = 0,
               int flag = 0) {
    (void) top_shapes;
    // naive
    std::vector<ncnn::Mat> b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, top_blob_count, b, func, flag);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, std::vector<ncnn::Mat>(), func, flag);
        if (ret != 0 || CompareMat(b, c, epsilon) != 0) {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, b, func, flag);
        if (ret != 0 || CompareMat(b, c, epsilon) != 0) {
            fprintf(stderr, "test_layer_cpu failed with shape hint\n");
            return -1;
        }
    }

#if NCNN_VULKAN
    // gpu
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING)) {
        std::vector<ncnn::Mat> d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, std::vector<ncnn::Mat>(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0)) {
            fprintf(stderr, "test_layer_gpu failed\n");
            return -1;
        }
    }

    // gpu shape hint
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING)) {
        std::vector<ncnn::Mat> d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0)) {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

template<typename T>
int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, void (*func)(T*),
                     int flag) {
    (void) flag;
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func) {
        (*func)((T*) op);
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads             = 1;
    opt.use_packing_layout      = false;
    opt.use_fp16_packed         = false;
    opt.use_fp16_storage        = false;
    opt.use_fp16_arithmetic     = false;
    opt.use_shader_pack8        = false;
    opt.use_image_storage       = false;
    opt.use_bf16_storage        = false;
    opt.use_vulkan_compute      = false;
    opt.use_weight_fp16_storage = false;

    op->create_pipeline(opt);

    if (op->support_inplace) {
        b = a.clone();
        ((T*) op)->T::forward_inplace(b, opt);
    } else {
        ((T*) op)->T::forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& c,
                   const ncnn::Mat& top_shape, void (*func)(T*), int flag) {
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func) {
        (*func)((T*) op);
    }

    if (top_shape.dims) {
        op->bottom_shapes.resize(1);
        op->top_shapes.resize(1);
        op->bottom_shapes[0] = a;
        op->top_shapes[0]    = top_shape;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt       = _opt;
    opt.num_threads        = 1;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    ncnn::Mat a4;

    if (opt.use_fp16_storage && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
        ncnn::cast_float32_to_float16(a, a4, opt);
    } else if (opt.use_bf16_storage && op->support_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
        ncnn::cast_float32_to_bfloat16(a, a4, opt);
    } else {
        a4 = a;
    }

    if (opt.use_packing_layout && op->support_packing && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_PACKING)) {
        // resolve dst_elempack
        int dims      = a4.dims;
        int elemcount = 0;
        if (dims == 1) {
            elemcount = a4.elempack * a4.w;
        }
        if (dims == 2) {
            elemcount = a4.elempack * a4.h;
        }
        if (dims == 3) {
            elemcount = a4.elempack * a4.c;
        }
        int elembits = a4.elembits();

        int dst_elempack = 1;

        if (elembits == 32) {
#if (NCNN_AVX2 || NCNN_AVX)
            if (elemcount % 8 == 0 && (ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx())) {
                dst_elempack = 8;
            } else if (elemcount % 4 == 0) {
                dst_elempack = 4;
            }
#elif NCNN_RVV
            const int packn = ncnn::cpu_riscv_vlenb() / (elembits / 8);
            if (elemcount % packn == 0) {
                dst_elempack = packn;
            }
#else
            if (elemcount % 4 == 0) {
                dst_elempack = 4;
            }
#endif
        }
        if (elembits == 16) {
#if NCNN_ARM82
            if (elemcount % 8 == 0 && opt.use_fp16_storage && opt.use_fp16_arithmetic && op->support_fp16_storage) {
                dst_elempack = 8;
            } else if (elemcount % 4 == 0) {
                dst_elempack = 4;
            }
#elif NCNN_RVV
            const int packn = ncnn::cpu_riscv_vlenb() / 2;
            if (elemcount % packn == 0) {
                dst_elempack = packn;
            }
#else
            if (elemcount % 4 == 0) {
                dst_elempack = 4;
            }
#endif
        }
        if (elembits == 8) {
#if NCNN_RVV
            const int packn = ncnn::cpu_riscv_vlenb() / 1;
            if (elemcount % packn == 0) {
                dst_elempack = packn;
            }
#else
            if (elemcount % 8 == 0) {
                dst_elempack = 8;
            }
#endif
        }

        if (flag & TEST_LAYER_ENABLE_FORCE_INPUT_PACK8) {
            dst_elempack = 8;
        }
        ncnn::Mat a4_packed;
        ncnn::convert_packing(a4, a4_packed, dst_elempack, opt);
        a4 = a4_packed;
    }

    if (op->support_inplace) {
        c = a4.clone();
        op->forward_inplace(c, opt);
    } else {
        op->forward(a4, c, opt);
    }

    if (opt.use_fp16_storage && op->support_fp16_storage && c.elembits() == 16) {
        ncnn::Mat c_fp32;
        ncnn::cast_float16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    } else if (opt.use_bf16_storage && op->support_bf16_storage && c.elembits() == 16) {
        ncnn::Mat c_fp32;
        ncnn::cast_bfloat16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_VULKAN
template<typename T>
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& d,
                   const ncnn::Mat& top_shape, void (*func)(T*), int flag) {
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (!op->support_vulkan) {
        delete op;
        return 233;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    op->vkdev = vkdev;

    if (func) {
        (*func)((T*) op);
    }

    if (top_shape.dims) {
        op->bottom_shapes.resize(1);
        op->top_shapes.resize(1);
        op->bottom_shapes[0] = a;
        op->top_shapes[0]    = top_shape;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator    = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt       = _opt;
    opt.num_threads        = 1;
    opt.use_vulkan_compute = true;

    #if __APPLE__
    opt.use_image_storage = false;
    #endif

    opt.blob_vkallocator      = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator   = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) {
        opt.use_fp16_packed = false;
    }
    if (!vkdev->info.support_fp16_storage()) {
        opt.use_fp16_storage = false;
    }
    if (!vkdev->info.support_fp16_arithmetic()) {
        opt.use_fp16_arithmetic = false;
    }
    // FIXME fp16a may produce large error
    opt.use_fp16_arithmetic = false;

    op->create_pipeline(opt);

    if (!op->support_vulkan) {
        delete op;
        return 233;
    }

    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload          = opt;
        opt_upload.blob_vkallocator      = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator   = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }

    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (op->support_image_storage && opt.use_image_storage) {
            // upload
            ncnn::VkImageMat a_gpu;
            cmd.record_upload(a, a_gpu, opt);

            ncnn::VkImageMat d_gpu;
            if (op->support_inplace) {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            } else {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            cmd.record_download(d_gpu, d, opt);
        } else {
            // upload
            ncnn::VkMat a_gpu;
            cmd.record_upload(a, a_gpu, opt);

            ncnn::VkMat d_gpu;
            if (op->support_inplace) {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            } else {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            cmd.record_download(d_gpu, d, opt);
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    return 0;
}
#endif // NCNN_VULKAN

template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a,
               const ncnn::Mat& top_shape = ncnn::Mat(), float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0) {
    (void) top_shape;
    // naive
    ncnn::Mat b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, b, func, flag);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, ncnn::Mat(), func, flag);
        if (ret != 0 || CompareMat(b, c, epsilon) != 0) {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, b, func, flag);
        if (ret != 0 || CompareMat(b, c, epsilon) != 0) {
            fprintf(stderr, "test_layer_cpu failed with shape hint\n");
            return -1;
        }
    }

#if NCNN_VULKAN
    // gpu
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING)) {
        ncnn::Mat d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, ncnn::Mat(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0)) {
            fprintf(stderr, "test_layer_gpu failed\n");
            return -1;
        }
    }

    // gpu shape hint
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING)) {
        ncnn::Mat d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0)) {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a,
               int top_blob_count = 1, float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0) {
    ncnn::Option opts[5];

    opts[0].use_packing_layout      = false;
    opts[0].use_fp16_packed         = false;
    opts[0].use_fp16_storage        = false;
    opts[0].use_fp16_arithmetic     = false;
    opts[0].use_bf16_storage        = false;
    opts[0].use_shader_pack8        = false;
    opts[0].use_image_storage       = false;
    opts[0].use_weight_fp16_storage = false;

    opts[1].use_packing_layout      = false;
    opts[1].use_fp16_packed         = true;
    opts[1].use_fp16_storage        = true;
    opts[1].use_fp16_arithmetic     = true;
    opts[1].use_bf16_storage        = true;
    opts[1].use_shader_pack8        = false;
    opts[1].use_image_storage       = false;
    opts[1].use_weight_fp16_storage = false;

    opts[2].use_packing_layout      = true;
    opts[2].use_fp16_packed         = true;
    opts[2].use_fp16_storage        = false;
    opts[2].use_fp16_arithmetic     = false;
    opts[2].use_bf16_storage        = false;
    opts[2].use_shader_pack8        = true;
    opts[2].use_image_storage       = false;
    opts[2].use_weight_fp16_storage = false;

    opts[3].use_packing_layout      = true;
    opts[3].use_fp16_packed         = true;
    opts[3].use_fp16_storage        = true;
    opts[3].use_fp16_arithmetic     = false;
    opts[3].use_bf16_storage        = true;
    opts[3].use_shader_pack8        = true;
    opts[3].use_image_storage       = true;
    opts[3].use_weight_fp16_storage = true;

    opts[4].use_packing_layout      = true;
    opts[4].use_fp16_packed         = true;
    opts[4].use_fp16_storage        = true;
    opts[4].use_fp16_arithmetic     = true;
    opts[4].use_bf16_storage        = false;
    opts[4].use_shader_pack8        = true;
    opts[4].use_image_storage       = true;
    opts[4].use_weight_fp16_storage = true;

    for (int i = 0; i < 5; i++) {
        opts[i].num_threads = 1;
    }

    for (int i = 0; i < 5; i++) {
        const ncnn::Option& opt = opts[i];

        // fp16 representation
        std::vector<ncnn::Mat> a_fp16;
        if (opt.use_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
            a_fp16.resize(a.size());
            for (size_t j = 0; j < a.size(); j++) {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_bfloat16(a[j], tmp, opt);
                ncnn::cast_bfloat16_to_float32(tmp, a_fp16[j], opt);
            }
        } else if ((opt.use_fp16_packed || opt.use_fp16_storage) && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
            a_fp16.resize(a.size());
            for (size_t j = 0; j < a.size(); j++) {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_float16(a[j], tmp, opt);
                ncnn::cast_float16_to_float32(tmp, a_fp16[j], opt);
            }
        } else {
            a_fp16 = a;
        }

        std::vector<ncnn::Mat> weights_fp16;
        float epsilon_fp16;
        if (opt.use_bf16_storage) {
            weights_fp16.resize(weights.size());
            for (size_t j = 0; j < weights.size(); j++) {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_bfloat16(weights[j], tmp, opt);
                ncnn::cast_bfloat16_to_float32(tmp, weights_fp16[j], opt);
            }
            epsilon_fp16 = epsilon * 100; // 0.1
        } else if (opt.use_fp16_packed || opt.use_fp16_storage) {
            weights_fp16.resize(weights.size());
            for (size_t j = 0; j < weights.size(); j++) {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_float16(weights[j], tmp, opt);
                ncnn::cast_float16_to_float32(tmp, weights_fp16[j], opt);
            }
            epsilon_fp16 = epsilon * 100; // 0.1
        } else {
            weights_fp16 = weights;
            epsilon_fp16 = epsilon;
        }

        if (opt.use_fp16_arithmetic) {
            epsilon_fp16 = epsilon * 1000; // 1.0
        }

        std::vector<ncnn::Mat> top_shapes;
        int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_blob_count, top_shapes, epsilon_fp16, func, flag);
        if (ret != 0) {
            fprintf(
                stderr,
                "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d "
                "use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d\n",
                layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8,
                opt.use_bf16_storage, opt.use_image_storage);
            return ret;
        }
    }

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, float epsilon = 0.001,
               void (*func)(T*) = 0, int flag = 0) {
    ncnn::Option opts[5];

    opts[0].use_packing_layout      = false;
    opts[0].use_fp16_packed         = false;
    opts[0].use_fp16_storage        = false;
    opts[0].use_fp16_arithmetic     = false;
    opts[0].use_bf16_storage        = false;
    opts[0].use_shader_pack8        = false;
    opts[0].use_image_storage       = false;
    opts[0].use_weight_fp16_storage = false;

    opts[1].use_packing_layout      = false;
    opts[1].use_fp16_packed         = true;
    opts[1].use_fp16_storage        = true;
    opts[1].use_fp16_arithmetic     = true;
    opts[1].use_bf16_storage        = true;
    opts[1].use_shader_pack8        = false;
    opts[1].use_image_storage       = false;
    opts[1].use_weight_fp16_storage = false;

    opts[2].use_packing_layout      = true;
    opts[2].use_fp16_packed         = true;
    opts[2].use_fp16_storage        = false;
    opts[2].use_fp16_arithmetic     = false;
    opts[2].use_bf16_storage        = false;
    opts[2].use_shader_pack8        = true;
    opts[2].use_image_storage       = false;
    opts[2].use_weight_fp16_storage = false;

    opts[3].use_packing_layout      = true;
    opts[3].use_fp16_packed         = true;
    opts[3].use_fp16_storage        = true;
    opts[3].use_fp16_arithmetic     = false;
    opts[3].use_bf16_storage        = true;
    opts[3].use_shader_pack8        = true;
    opts[3].use_image_storage       = true;
    opts[3].use_weight_fp16_storage = true;

    opts[4].use_packing_layout      = true;
    opts[4].use_fp16_packed         = true;
    opts[4].use_fp16_storage        = true;
    opts[4].use_fp16_arithmetic     = true;
    opts[4].use_bf16_storage        = false;
    opts[4].use_shader_pack8        = true;
    opts[4].use_image_storage       = true;
    opts[4].use_weight_fp16_storage = true;

    for (int i = 0; i < 5; i++) {
        opts[i].num_threads = 1;
    }

    for (int i = 0; i < 5; i++) {
        const ncnn::Option& opt = opts[i];

        // fp16 representation
        ncnn::Mat a_fp16;
        if (opt.use_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_bfloat16(a, tmp, opt);
            ncnn::cast_bfloat16_to_float32(tmp, a_fp16, opt);
        } else if ((opt.use_fp16_packed || opt.use_fp16_storage) && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)) {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_float16(a, tmp, opt);
            ncnn::cast_float16_to_float32(tmp, a_fp16, opt);
        } else {
            a_fp16 = a;
        }

        std::vector<ncnn::Mat> weights_fp16;
        float epsilon_fp16;
        if (opt.use_bf16_storage) {
            weights_fp16.resize(weights.size());
            for (size_t j = 0; j < weights.size(); j++) {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_bfloat16(weights[j], tmp, opt);
                ncnn::cast_bfloat16_to_float32(tmp, weights_fp16[j], opt);
            }
            epsilon_fp16 = epsilon * 100; // 0.1
        } else if (opt.use_fp16_packed || opt.use_fp16_storage) {
            weights_fp16.resize(weights.size());
            for (size_t j = 0; j < weights.size(); j++) {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_float16(weights[j], tmp, opt);
                ncnn::cast_float16_to_float32(tmp, weights_fp16[j], opt);
            }
            epsilon_fp16 = epsilon * 100; // 0.1
        } else {
            weights_fp16 = weights;
            epsilon_fp16 = epsilon;
        }

        if (opt.use_fp16_arithmetic) {
            epsilon_fp16 = epsilon * 1000; // 1.0
        }

        ncnn::Mat top_shape;
        int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_shape, epsilon_fp16, func, flag);
        if (ret != 0) {
            fprintf(
                stderr,
                "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d "
                "use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d\n",
                layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8,
                opt.use_bf16_storage, opt.use_image_storage);
            return ret;
        }
    }

    return 0;
}

#define COMPARE_THRESHOLD_FP32 0.01
#define COMPARE_THRESHOLD_FP16 0.1

#define COMPARE_MAT(ncnnMat, snnMat) { \
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); \
    if (ret) { \
        if (printMismatch) { \
            pretty_print_ncnn(ncnnMat); \
            pretty_print_ncnn(snnMat, "SNN"); \
        } \
        if (stopOnMismatch) { \
            return ret; \
        } \
    } \
}

#define COMPARE_MAT1(ncnnMat, snnMat, layer) { \
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); \
    printf("-----------------------------%s %d\n", layer, ret); \
    if (ret) { \
        if (printMismatch) { \
            pretty_print_ncnn(ncnnMat); \
            pretty_print_ncnn(snnMat, "SNN"); \
        } \
        if (stopOnMismatch) { \
            return ret; \
        } \
    } \
}


bool checkPlatFormSupport(bool useVulkan);

#define CHECK_PLATFORM_SUPPORT(useVulkan) \
    if (!checkPlatFormSupport(useVulkan)) { \
        return 0; \
    }

#endif // TESTUTIL_H
