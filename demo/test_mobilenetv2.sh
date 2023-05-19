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

rm -rf ../../../../core/inferenceCoreDump/*
./inferenceProcessorTest --use_compute --dump_outputs mobilenetv2
./mobilenetv2Test  --stop_on_mismatch

rm -rf ../../../../core/inferenceCoreDump/*
./inferenceProcessorTest --use_vulkan --dump_outputs mobilenetv2
./mobilenetv2Test --stop_on_mismatch --use_vulkan

rm -rf ../../../../core/inferenceCoreDump/*
./inferenceProcessorTest --use_compute --use_half --dump_outputs mobilenetv2
./mobilenetv2Test  --use_half --stop_on_mismatch

rm -rf ../../../../core/inferenceCoreDump/*
./inferenceProcessorTest --use_1ch_mrt --dump_outputs mobilenetv2
./mobilenetv2Test --use_1ch_mrt --stop_on_mismatch

rm -rf ../../../../core/inferenceCoreDump/*
./inferenceProcessorTest --use_1ch_mrt --use_half --dump_outputs mobilenetv2
./mobilenetv2Test --use_half --use_1ch_mrt --stop_on_mismatch

rm -rf ../../../../core/inferenceCoreDump/*
./inferenceProcessorTest --use_2ch_mrt --dump_outputs mobilenetv2
./mobilenetv2Test --use_2ch_mrt --stop_on_mismatch

echo "done"

cd ../../../
