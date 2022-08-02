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

int32_t snn::InferenceProcessor::initialize(const InitializationParameters cp) {
    _modelFileName      = cp.modelName;
    _inputList          = cp.inputList;
    this->dumpOutputs   = cp.dumpOutputs;
    this->halfPrecision = cp.halfPrecision;
    if (!_ic2) {
        dp::ShaderGenOptions options = {};
        options.preferrHalfPrecision = cp.halfPrecision;
        SNN_LOGI("Loading asset : %s ...", _modelFileName.c_str());
        auto dp = snn::dp::loadFromJsonModel(_modelFileName, cp.mrtMode, cp.weightMode, options.preferrHalfPrecision);

        options.desiredInput.width  = cp.inputList[0].second[0];
        options.desiredInput.height = cp.inputList[0].second[1];
        options.desiredInput.depth  = cp.inputList[0].second[2];
        options.desiredInput.format = ColorFormat::RGBA8;
        options.compute             = cp.useComputeShader;
        options.desiredOutputFormat = ColorFormat::RGBA8;
        options.mrtMode             = cp.mrtMode;
        options.weightMode          = cp.weightMode;

        MixedInferenceCore::CreationParameters inferenceCP;
        (InferenceGraph &&) inferenceCP = snn::dp::generateInferenceGraph(dp.at(0), options);
        inferenceCP.dumpOutputs         = cp.dumpOutputs;
        _ic2                            = MixedInferenceCore::create(inferenceCP);
    }
    _rc = cp.rc;
    _rc->makeCurrent();
    return 0;
}

int32_t snn::InferenceProcessor::finalize(void) {
    // clean resource here
    glFinish();
    return 0;
}

int32_t snn::InferenceProcessor::preProcess(FixedSizeArray<snn::ImageTexture>& inputTexs) {
    _inputTexs.deallocate();
    _inputTexs.allocate(_inputList.size());

    // Copy the input textures
    for (size_t i = 0; i < _inputList.size(); ++i) {
        //_inputTexs[i].setName(_inputList[i].first);
        _inputTexs[i].texture(0)->attach(*inputTexs[i].texture(0));
    }
    SNN_LOGD("%s:%d, texture: %d, %d\n", __FUNCTION__, __LINE__, _inputTexs[0].texture(0)->id(), _inputTexs[0].texture(0)->target());
    return 0;
}

bool snn::InferenceProcessor::registerLayer(std::string layerName, snn::dp::LayerCreator creator) {
    SNN_LOGD("%s:%d register layer: %s\n", __FUNCTION__, __LINE__, layerName.c_str());
    return snn::dp::registerLayer(layerName, creator);
}

int32_t snn::InferenceProcessor::process(FixedSizeArray<snn::ImageTexture>& outputTexs) {
    auto outTexture = outputTexs[0].texture(0);

    std::vector<gl::TextureObject*> inputs;
    for (size_t i = 0; i < _inputTexs.size(); i++) {
        inputs.push_back(_inputTexs[i].texture(0));
    }
    SNN_LOGD("%s:%d, texture: %d, %d\n", __FUNCTION__, __LINE__, inputs[0]->id(), inputs[0]->target());

    auto outVec    = std::vector<std::vector<std::vector<float>>>();
    auto inVec     = std::vector<std::vector<std::vector<float>>>();
    double avgTime = 0.0;
    int loopcount;
    if (this->dumpOutputs) {
        loopcount = 1;
    } else {
        loopcount = 70;
    }

// //Hack the code
// auto input = ManagedRawImage::loadFromAsset("images/cifar_test.png");
// gl::TextureObject inputTexture;
// inputTexture.allocate2D(input.format(), input.width(), input.height());
// inputTexture.setPixels(0, 0, 0, input.width(), input.height(), input.pitch(), input.data());
// // SNN_LOGD("INPUT TEXTURE: %u, %u", inputTexture.id(), inputTexture.target());
// {
//     auto input32f = snn::toRgba32f(input, -1.0, 1.0);
//     gl::TextureObject scaleTex;
//     scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height());
//     scaleTex.setPixels(
//         0,
//         0, 0,
//         32, 32,
//         0,
//         input32f.data()
//     );
//     scaleTex.detach();
//     inputTexture.attach(scaleTex.target(), scaleTex.id());

// }
// const gl::TextureObject* inputs[] = { &inputTexture };
#ifdef PROFILING
    auto start = std::chrono::high_resolution_clock::now();
    auto end   = std::chrono::high_resolution_clock::now();
#endif
    InferenceEngine::SNNModelOutput modelOutput;
    snn::MixedInferenceCore::RunParameters rp = {inputs.data(), outTexture, inputs.size(), inVec, outVec, modelOutput};
#ifdef PROFILING
    map<string, vector<double>> timeMap;
    int skip_loop = 5;
#endif
    for (int l = 0; l < loopcount; ++l) {
#ifdef PROFILING
        start = std::chrono::high_resolution_clock::now();
#endif
        _ic2->run(rp);
#ifdef PROFILING
        _ic2->writeTimeStat(timeMap);
        end          = std::chrono::high_resolution_clock::now();
        auto infTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        SNN_LOGI("Inference took %f milliseconds", infTime.count() / 1000000.0);
        if ((l > 5) && (l < loopcount - 5)) {
            avgTime += infTime.count() / (1000000.0 * (loopcount - 10.0));
        }
#endif
        // _ic2->run({ inputs, outTexture, 1, inVec, outVec});
        _rc->swapBuffers();
    }
#ifdef PROFILING
    SNN_LOGI("Average Time over %d loops: %f", loopcount - 10, avgTime);
#endif

#ifdef PROFILING
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
        if (v.size() > skip_loop) {
            v.erase(v.begin(), v.begin() + skip_loop);
        }
        auto mean    = accumulate(v.begin(), v.end(), 0.0) / v.size();
        double accum = 0;
        for (auto n : v) {
            accum += (n - mean) * (n - mean);
        }
        double stdev = sqrt(accum / (v.size() - 1));
        std::string meanStr, stdDevStr;
        try {
            meanStr = isnan(mean) ? "nan   " : std::to_string(mean);
        } catch (std::exception& e) { meanStr = "nan   "; }
        try {
            stdDevStr = isnan(stdev) ? "nan   " : std::to_string(stdev);
        } catch (std::exception& e) { stdDevStr = "nan   "; }
        ss.precision(6);
        ss << meanStr << ((10 - (int) meanStr.length() > 0) ? std::string(10 - (int) meanStr.length(), ' ') : "") << "| ";
        ss << stdDevStr << ((12 - (int) stdDevStr.length() > 0) ? std::string(12 - (int) stdDevStr.length(), ' ') : "") << "|\n";
        ss << "-------------------------------------------------------------------------------\n";
        index++;
    }
    ss << "==================================================================================\n";
    SNN_LOGI(ss.str().c_str());
#endif
    return 0;
}
