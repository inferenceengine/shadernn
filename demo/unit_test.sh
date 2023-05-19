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

if [ -z  "$NO_REBUILD_IN_TESTS" ]; then
    ./build-tests.sh clean
    ./build-tests.sh linux
fi

cd build-test/test/unittest/

rm -rf ../../../../core/inferenceCoreDump/*;

./activationTest
./activationTest --use_vulkan

./poolingTest
./poolingTest --use_vulkan

./upSampleTest -W 8 -H 2 --type 2 --use_compute
./upSampleTest -W 8 -H 2 --type 2 --use_vulkan

./binaryOpTest
./binaryOpTest --use_vulkan

./convolutionTest
./convolutionTest --use_vulkan

./concatTest
./concatTest --use_vulkan

./batchNormTest
./batchNormTest --use_vulkan

./padTest
./padTest --use_vulkan

#./flattenTest
#./flattenTest --use_vulkan

#./denseTest
#./denseTest --use_vulkan

./instanceNormTest
./instanceNormTest --use_vulkan

./depthwiseConv2DTest
./imageTextureTest
./imageTextureResizeTest

cd ../../../
