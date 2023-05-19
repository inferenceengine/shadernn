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
#include "inferenceProcessor.h"
#include "ic2/dp.h"
#include "snn/contextFactory.h"
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <chrono>

#ifdef PROFILING
// Logs timing statistics
// params:
//  timeArray - a map with keys of layer names and values of timing of successive runs
void logTimingStatistics(const std::map<std::string, std::vector<double>>& timeMap);
#endif

snn::InferenceProcessor::InferenceProcessor(GpuContext* context)
    : _context(context)
    , _inputTexs(ImageTextureAllocator(context))
    , _outputTexs(ImageTextureAllocator(context))
{}

std::shared_ptr<snn::InferenceProcessor> snn::InferenceProcessor::create(bool useVullkan) {
    snn::GpuContext* context = snn::createDefaultContext(useVullkan);
    return std::shared_ptr<snn::InferenceProcessor>(new snn::InferenceProcessor(context));
}

void snn::InferenceProcessor::initialize(const InitializationParameters& cp) {
    _modelFileName      = cp.modelName;
    _inputList          = cp.inputList;
    this->dumpOutputs   = cp.dumpOutputs;
    this->halfPrecision = cp.halfPrecision;
    startRun = std::chrono::high_resolution_clock::now();
    if (!_ic2) {
        dp::ShaderGenOptions options = {};
        options.preferrHalfPrecision = cp.halfPrecision;
        SNN_LOGI("Loading asset : %s ..., fp16: %d", _modelFileName.c_str(), cp.halfPrecision);
        auto dp = snn::dp::loadFromJsonModel(_modelFileName, cp.useVulkanShader, cp.mrtMode, cp.weightMode, options.preferrHalfPrecision);

        auto inputTex = InferenceGraph::IODesc {ColorFormat::RGBA8, cp.inputList[0].second[0],
                                                    cp.inputList[0].second[1], cp.inputList[0].second[2], 4};
        options.desiredInput.push_back(inputTex);

        options.compute             = cp.useComputeShader;
        options.desiredOutputFormat = ColorFormat::RGBA8;
        options.mrtMode             = cp.mrtMode;
        options.weightMode          = cp.weightMode;
        options.vulkan              = cp.useVulkanShader;

        MixedInferenceCore::CreationParameters inferenceCP;
        (InferenceGraph &&) inferenceCP = snn::dp::generateInferenceGraph(dp, options);
        inferenceCP.dumpOutputs         = cp.dumpOutputs;

        _ic2  = MixedInferenceCore::create(_context, inferenceCP);
    }
    modelType = cp.modelType;
    maxLoops = cp.maxLoops;
}

int32_t snn::InferenceProcessor::preProcess(snn::ImageTextureArrayAccessor inputTexs) {
    _inputTexs.deallocate();
    _inputTexs.allocate(_inputList.size());

    // Copy the input textures
    for (size_t i = 0; i < _inputList.size(); ++i) {
        _inputTexs[i].attach(&inputTexs[i]);
    }
    SNN_LOGD("texture: %s", _inputTexs[0].getTextureInfo2().c_str());
    return 0;
}

#ifdef PROFILING
static constexpr size_t NUM_EXCLUDE_FIRST_LOOPS = 5;
#endif

int32_t snn::InferenceProcessor::process(snn::ImageTextureArrayAccessor outputTexs) {
    SNN_LOGD("texture: %s", _inputTexs[0].getTextureInfo2().c_str());

    auto outVec    = std::vector<std::vector<std::vector<float>>>();
    auto inVec     = std::vector<std::vector<std::vector<float>>>();

    snn::SNNModelOutput modelOutput;
    modelOutput.modelType = modelType;
    snn::MixedInferenceCore::RunParameters rp = {_inputTexs, outputTexs, inVec, outVec, modelOutput};
#ifdef PROFILING
    std::map<std::string, std::vector<double>> timeMap;
    snn::Timer cpuRunTime("IC2 CPU time all iteraiions");
#endif
    auto startRun = std::chrono::high_resolution_clock::now();
    uint32_t l = 0;
    for (; l < maxLoops;) {
        SNN_LOGD("Run # %d", l);
#ifdef PROFILING
        if (l == NUM_EXCLUDE_FIRST_LOOPS) {
            SNN_LOGI("Benchmark begin (%d : %d)", l, maxLoops);
        }
        if (l >= NUM_EXCLUDE_FIRST_LOOPS) {
            cpuRunTime.start();
        }
#endif
        _ic2->run(rp);
        auto end = std::chrono::high_resolution_clock::now();
#ifdef PROFILING
        if (l >= NUM_EXCLUDE_FIRST_LOOPS) {
            cpuRunTime.stop();
        }
        _ic2->writeTimeStat(timeMap);
#endif
        if (modelType == snn::ModelType::CLASSIFICATION) {
            SNN_LOGD("Classification output: %d", rp.modelOutput.classifierOutput);
        }
        else if (modelType == snn::ModelType::DETECTION) {
            SNN_LOGD("Detection output:");
            for (auto &boxDetails : rp.modelOutput.detectionOutput) {
                SNN_ASSERT(boxDetails.size() >= 6);
                SNN_LOGD("score: %f, coord: %f, %f, %f, %f", boxDetails[1], boxDetails[2], boxDetails[3], boxDetails[4], boxDetails[5]);
            }
        }
        auto infTimeTotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end - startRun);
        ++l;
        if (infTimeTotal.count() / 1000000000.0 > MAX_TEST_TIME_SECONDS) {
            break;
        }
    }
#ifdef PROFILING
    if (l > NUM_EXCLUDE_FIRST_LOOPS) {
        SNN_LOGI("Benchmark end (%d)", l);
        std::string cpuRunTimeStats = snn::Timer::print(1, false);
        SNN_LOGI("%s", cpuRunTimeStats.c_str());
    }
    logTimingStatistics(timeMap);
#endif
    return 0;
}

#ifdef PROFILING
void logTimingStatistics(const std::map<std::string, std::vector<double>>& timeMap) {
    std::stringstream ss;
    int index = -1;
    ss << "\n=============================  Final Time Stats  ==============================\n";
    ss << "\n|  Layer ID  |                Name                  |    Mean   |   Std Dev   |\n";
    ss << "\n===============================================================================\n";
    for (auto arr : timeMap) {
        auto name      = arr.first;
        auto layerName = (name.find("[") != std::string::npos) ? name.substr(name.find("["), name.length() - 1) : name;
        auto idxBuffer = index > 9 ? 10 : 11;
        idxBuffer      = index < 0 ? 10 : idxBuffer;
        idxBuffer      = (index > 99) ? 9 : idxBuffer;
        int nameBuffer = 38 - layerName.length();

        if (nameBuffer - 1 < 0) {
            layerName = layerName.substr(0, 34) + "...";
        }

        ss << "|" << std::string(idxBuffer / 2, ' ') << index << std::string((idxBuffer % 2 == 0) ? (idxBuffer / 2) : (idxBuffer / 2 + 1), ' ') << "| ";
        ss << layerName;
        ss << ((nameBuffer - 1) > 0 ? std::string(nameBuffer - 1, ' ') : "") << "| ";

        auto v = arr.second;
        v.erase(std::remove(begin(v), std::end(v), 0), std::end(v));
        if (v.size() > NUM_EXCLUDE_FIRST_LOOPS) {
            v.erase(v.begin(), v.begin() + NUM_EXCLUDE_FIRST_LOOPS);
        } else {
            v.clear();
        }
        auto mean = v.size() == 0 ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double accum = 0;
        for (auto n : v) {
            accum += (n - mean) * (n - mean);
        }
        // We divide by N, not (N - 1), because we have a true mean here
        // http://johnthemathguy.blogspot.com/2014/07/standard-deviation-why-n-and-n-1.html
        double stdev = v.size() == 0 ? 0.0 : sqrt(accum / v.size());
        std::string meanStr = std::to_string(mean);
        std::string stdDevStr = std::to_string(stdev);
        ss.precision(6);
        ss << meanStr << ((10 - (int) meanStr.length() > 0) ? std::string(10 - (int) meanStr.length(), ' ') : "") << "| ";
        ss << stdDevStr << ((12 - (int) stdDevStr.length() > 0) ? std::string(12 - (int) stdDevStr.length(), ' ') : "") << "|\n";
        ss << "-------------------------------------------------------------------------------\n";
        index++;
    }
    ss << "==================================================================================\n";
    SNN_LOGI(ss.str().c_str());
}
#endif // PROFILING
