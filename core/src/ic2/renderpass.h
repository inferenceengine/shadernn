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

#include "snn/utils.h"
#include <string>

namespace snn {

// This is a base class of one render pass.
// Derived classes implement actions, performed during a render pass.
class RenderPass {
public:
    RenderPass() = default;

    virtual ~RenderPass() = default;

    SNN_NO_COPY(RenderPass);
    SNN_NO_MOVE(RenderPass);

    // Dump layer outputs
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    virtual bool debugPassOutput(const std::string& folderName) {
        (void) folderName;
        return true;
    }

    // Dump layer inputs
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    virtual bool debugPassInputs(const std::string& folderName) {
        (void) folderName;
        return true;
    }

    // Dump layer weights
    // params:
    //  folderName - directory where to dump
    // return:
    //  true if success, false if not
    virtual bool debugPassWeights(const std::string& foldername, int shaderPass) {
        (void) foldername;
        (void) shaderPass;
        return true;
    }

    virtual void run(){
        return;
    }
};

} // namespace snn
