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
#include "snn/snn.h"
#include "modelInference.h"
#include "inferenceProcessor.h"
#include "testutil.h"
#include <string>
#include <vector>
#include <array>

// Global namespace is polluted somewhere
#ifdef Success
    #undef Success
#endif
#include "CLI/CLI.hpp"

void runESPCN(bool dumpOutputs, bool useCompute, snn::MRTMode mrtMode, snn::WeightAccessMethod weightMode, bool useHalfFP, bool useVulkan) {
    if (!useVulkan) {
        SNN_LOGI("Error: OpenGL test not yet supported");
        return;
    }

    auto ip = snn::InferenceProcessor::create(true);

    auto                                                       modelFileName = "ESPCN/ESPCN_2X_16_16_4.json";
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", std::vector<uint32_t> {224, 224, 1, 1}});

    ip->initialize({modelFileName, inputList, mrtMode, weightMode, useHalfFP, dumpOutputs, useCompute, true});

    snn::ImageTextureArray inputTexs {snn::ImageTextureAllocator(ip->getContext())};

    inputTexs.allocate(1);

    inputTexs[0].loadFromFile(snn::formatString("%sassets/images/ant_y_channel.png", ASSETS_DIR).c_str());
    inputTexs[0].convertToRGBA32FAndNormalize();

    std::array<float, 4> resizeMeans {0.0, 0.0, 0.0, 0.0};
    std::array<float, 4> resizeNorms {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};

    inputTexs[0].upload();
    inputTexs[0].resize(1, 1, resizeMeans, resizeNorms);

    ip->preProcess(inputTexs);

    snn::ImageTextureArray outputTexs {snn::ImageTextureAllocator(ip->getContext())};
    outputTexs.allocate(1);
    ip->process(outputTexs);
}

std::string printMRTMode(snn::MRTMode mrtMode) {
    switch (mrtMode) {
    case snn::MRTMode::SINGLE_PLANE:
        return std::string("SINGLE_PLANE MRT");
    case snn::MRTMode::DOUBLE_PLANE:
        return std::string("DOUBLE_PLANE MRT");
    case snn::MRTMode::QUAD_PLANE:
        return std::string("QUAD_PLANE MRT");
    default:
        return std::string("NO MRT");
    }
}

int main(int argc, char ** argv) {
    bool                    dumpOutputs  = false;
    bool                    useCompute   = false;
    bool                    use1chMrt    = false;
    bool                    use2chMrt    = false;
    bool                    useConstants = false;
    bool                    useHalfFP    = false;
    bool                    useVulkan    = false;
    bool                    useFinetuned = false;
    snn::MRTMode            mrtMode      = snn::MRTMode::SINGLE_PLANE;
    snn::WeightAccessMethod weightMode   = snn::WeightAccessMethod::TEXTURES;
    uint32_t                innerLoops   = 1; // For testing performance
    uint32_t                outerLoops   = 1; // For testing functionalitu on different input images

    const std::vector<std::string> MODELS = {"resnet18", "yolov3tiny", "unet", "mobilenetv2", "spatialdenoise", "aidenoise", "styletransfer", "espcn2x"};
    std::vector<std::string>       modelsToRun;
    std::vector<int>               modelIdx;

    CLI::App app;
    app.add_flag("--use_vulkan", useVulkan, "Use Vulkan");
    app.add_flag("--use_compute", useCompute, "Use compute shader (OpenGL only)");
    app.add_flag("--use_1ch_mrt", use1chMrt, "Use single plane MRT (OpenGL only)");
    app.add_flag("--use_2ch_mrt", use2chMrt, "Use double plane MRT (OpenGL only)");
    app.add_flag("--use_constants", useConstants, "Store weight as constants (OpenGL only)");
    app.add_flag("--use_half", useHalfFP, "Use half-precision floating point values (fp16)");
    app.add_flag("--use_finetuned", useFinetuned, "Use fine-tuned models");
    app.add_flag("--dump_outputs", dumpOutputs, "Dump outputs");
    app.add_option("--inner_loops", innerLoops, "Number of inner loops (after model loading)");
    app.add_option("--outer_loops", outerLoops, "Number of outer loops (before model loading)");
    app.add_option("model", modelsToRun,
                   "Model(s) to run: {resnet18 | yolov3tiny | unet | mobilenetv2 | spatialdenoise | aidenoise | styletransfer | espcn2x}");
    CLI11_PARSE(app, argc, argv);
    CHECK_PLATFORM_SUPPORT(useVulkan)

    for (size_t i = 0; i < modelsToRun.size(); ++i) {
        auto iter = std::find(MODELS.begin(), MODELS.end(), modelsToRun[i]);
        if (iter == modelsToRun.end()) { SNN_RIP("Incorrect model name: %s", modelsToRun[i].c_str()); }
        int idx = iter - MODELS.begin();
        modelIdx.push_back(idx);
    }
    if (use1chMrt) {
        mrtMode = snn::MRTMode::SINGLE_PLANE;
    } else if (use2chMrt) {
        mrtMode = snn::MRTMode::DOUBLE_PLANE;
    }
    if (useConstants) { weightMode = snn::WeightAccessMethod::CONSTANTS; }
    innerLoops = std::max(1U, innerLoops);
    outerLoops = std::max(1U, outerLoops);
    for (int model : modelIdx) {
        SNN_LOGI("\n\n**************************************************\n Model: %s %s %s %s %s %s %s \n**************************************************",
                 MODELS[model].c_str(), (useFinetuned ? "finetuned" : "non-finetuned"), (useHalfFP ? ", fp16" : ", fp32"),
                 (useVulkan ? ", Vulkan" : ", OpenGL"), (useVulkan ? "" : (useCompute ? ", Compute shader" : ", Fragment shader")),
                 (useVulkan ? "" : (use2chMrt ? ", Double plane MRT" : ", Single plane MRT")),
                 (useVulkan ? "" : (useConstants ? ", Weights in constants" : "")));
        switch (model) {
        case 0: {
            runResnet18(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops, outerLoops);
            break;
        }
        case 1: {
            runYolov3Tiny(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops);
            break;
        }
        case 2: {
            runUNet(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops);
            break;
        }
        case 3: {
            runMobilenetV2(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops);
            break;
        }
        case 4: {
            runSpatialDenoiser(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops);
            break;
        }
        case 5: {
            runAIDenoiser(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops);
            break;
        }
        case 6: {
            runStyleTransfer(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops);
            break;
        }
        case 7: {
            runESPCN(dumpOutputs, useCompute, mrtMode, weightMode, useHalfFP, useVulkan, useFinetuned, innerLoops);
            break;
        }
        default:
            SNN_ASSERT(false);
            break;
        }
    }

    return 0;
}
