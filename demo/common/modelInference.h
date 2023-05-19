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
#include "snn/snn.h"
#include <stdint.h>

// This header contains API to run specific models.
// Used fopr benchmarking/testing

// Run Spatial denoiser
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runSpatialDenoiser(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1);

// Run AI denoiser
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runAIDenoiser(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1);

// Run Resnet18
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runResnet18(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1, uint32_t outerLoops = 1);

// Run Mobilenet v2
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runMobilenetV2(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1);

// Run YOLO v3 (tiny)
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runYolov3Tiny(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1);

// Run Unet
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runUNet(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1);

// Run Style transfer
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runStyleTransfer(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1);

// Run ESPCN
// params:
//      dumpOutputs - flag to to dump outputs
//      useCompute - flag to use compute shader
//      mrtMode - MRT (Multi Render Target) mode to use
//      weightMode - Weights access mode to use
//      useHalfFP - flag to use FP16 calculations
//      useVulkan - flag to use Vulkan
//      useFinetuned - flag to use fine-tuned model
//      innerLoops - number of inner loops (successive inference runs on the same input image). Used for benchmarking.
int runESPCN(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode,
        bool useHalfFP, bool useVulkan, bool useFinetuned = false, uint32_t innerLoops = 1);

