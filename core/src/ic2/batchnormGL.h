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

#include "batchnorm.h"
#include "snn/utils.h"
#include <string>
#include <sstream>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class BatchNormalizationLayerGl : public BatchNormalizationLayer {
public:
    BatchNormalizationLayerGl(BatchNormalizationDesc&& d): BatchNormalizationLayer(std::move(d)) {}
    virtual ~BatchNormalizationLayerGl() = default;

protected:
    InferencePassesSptr createFS(const LayerGenOptions&) const override;
    InferencePassesSptr createCS(const LayerGenOptions&) const override;

private:
    void getBatchNormalizationConstants(std::vector<std::ostringstream>& batchNormalizationConstants, const int shaderPass, const int outputChannels) const;

    // Adds predefine to given stringstream.
    // shaderFilePath is the path of the file template this will be appended to
    // and is used to label the preDefine.
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;
};

}; // namespace dp
} // namespace snn
