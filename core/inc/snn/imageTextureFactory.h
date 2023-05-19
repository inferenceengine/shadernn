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
#include "imageTexture.h"
#include "snn/snn.h"
#include "snn/color.h"
#include <string>
#include <array>
#include <memory>

namespace snn {

// This class methods creates an object of of ImageTexture - derived class (ImageTextureGL or ImageTextureVulkan)
struct ImageTextureFactory {
    static std::shared_ptr<ImageTexture> createImageTexture(GpuContext* context);

    static std::shared_ptr<ImageTexture> createImageTexture(GpuContext* context, const std::array<uint32_t, 4>& dims, ColorFormat format,
        const void* buffer = NULL, const std::string& name = "");

    static std::shared_ptr<ImageTexture> createImageTexture(GpuContext* context, const std::string& fileName);
};

} // namespace snn
