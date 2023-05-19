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
#include "vulkanImageTransformShaderOp.h"
#include <array>

namespace snn {

// This class provides a facility to resize an image using a shader
class VulkanImageResizeOp : public VulkanImageTransformShaderOp {
public:
    virtual ~VulkanImageResizeOp() = default;

    // These parameters are passed to resize shader uniform buffer
    struct ResizeImageParams {
        std::array<uint32_t, 4> outputSize;
        std::array<float, 4> means;
        std::array<float, 4> norms;
    };

    void init(uvkc::vulkan::Device *device, bool linearFilter = true);

    void updateParams(VkExtent3D dstDimensions, const std::array<float, 4>& means, const std::array<float, 4>& norms);

protected:
    virtual absl::StatusOr<std::unique_ptr<uvkc::vulkan::Sampler>> createSampler() override;

private:
    bool _linearFilter = true;
};

}
