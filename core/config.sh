# Copyright (C) 2020 - 2022 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
ROOT=`dirname "$(realpath $0)"`
CONGIG_FILE=${ROOT}/config.txt

print_usage()
{
    echo
    echo "SNN configure script"
    echo
    echo "Usage: `basename $0` [gl] [vulkan]"
    echo
    echo "gl: Build with OpenGL support"
    echo "vulkan: Build with Vulkan support"
    echo
}

echo "" > config.txt

if [ "$1" == "gl" ]; then
    GL_TOKEN="GL"
elif [ "$1" == "vulkan" ]; then
    VULKAN_TOKEN="VULKAN"
else 
    print_usage
    exit 1
fi

if [ "$2" == "gl" ]; then
    GL_TOKEN="GL"
elif [ "$2" == "vulkan" ]; then
    VULKAN_TOKEN="VULKAN"
elif [ "$2" != "" ]; then
    print_usage
fi

if [ "$GL_TOKEN" == "" ] && [ "$VULKAN_TOKEN" == "" ]; then
    echo "The build should support either OpenGL or Vulkan"
    exit 1
fi

echo "$GL_TOKEN $VULKAN_TOKEN" > config.txt
