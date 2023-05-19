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
#include "pch.h"

#include "androidWindow.h"
#include "error.h"
#include "snn/utils.h"

namespace vkb {

AndroidWindow::AndroidWindow(ANativeWindow *&window)
    : handle(window)
{
    SNN_ASSERT(window);
    h = ANativeWindow_getHeight(window);
    w = ANativeWindow_getWidth(window);
    SNN_LOGI("Android window was created. handle = %p, w = %d, h = %d", handle, w, h);
}

VkSurfaceKHR AndroidWindow::createSurface(Instance &instance)
{
    return createSurface(instance.get_handle(), VK_NULL_HANDLE);
}

VkSurfaceKHR AndroidWindow::createSurface(VkInstance instance, VkPhysicalDevice)
{
    VkSurfaceKHR surface{};

    VkAndroidSurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};

    info.window = handle;

    VK_CHECK(vkCreateAndroidSurfaceKHR(instance, &info, nullptr, &surface));

    SNN_LOGI("Vulkan Android surface KHR is created from native window");

    return surface;
}

}
