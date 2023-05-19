/* Copyright (c) 2018-2022, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

#include <android/native_window_jni.h>
#include "instance.h"
#include "volk.h"

namespace vkb
{

class AndroidWindow {
public:
    /**
     * @brief Constructor
     * @param window A reference to the location of the Android native window
     */
    AndroidWindow(ANativeWindow *&window);

    ~AndroidWindow() = default;

    /**
     * @brief Creates a Vulkan surface to the native window
     *        If headless, this will return VK_NULL_HANDLE
     */
    VkSurfaceKHR createSurface(Instance &instance);

    /**
     * @brief Creates a Vulkan surface to the native window
     */
    VkSurfaceKHR createSurface(VkInstance instance, VkPhysicalDevice physical_device);

    int32_t getWidth() const { return w; }
    int32_t getHeighth() const { return h; }

private:
    ANativeWindow *&handle;
    ANativeWindow *window;
    int32_t h;
    int32_t w;
};

}
