# Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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

./build-tests.sh clean
./build-tests.sh linux

cd build-test/test/unittest/

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_compute 1 1
./yolov3TinyTest --use_compute

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_compute --use_half 1 1
./yolov3TinyTest --use_compute

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_1ch_mrt 1 1
./yolov3TinyTest --use_1ch_mrt

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_1ch_mrt --use_half 1 1
./yolov3TinyTest --use_1ch_mrt

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest 1 1
./yolov3TinyTest --use_2ch_mrt

cd ../../../
