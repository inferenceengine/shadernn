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

./build-tests.sh clean
./build-tests.sh linux

cd build-test/test/unittest/

if [ 1 -eq 1 ]; then

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_compute 2 1
./unetTest --use_compute

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_compute --use_half 2 1
./unetTest --use_compute

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_1ch_mrt 2 1
./unetTest --use_1ch_mrt

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest --use_1ch_mrt --use_half 2 1
./unetTest --use_1ch_mrt

rm -rf ../../../../core/inferenceCoreDump/*;
./inferenceProcessorTest 2 1
./unetTest --use_2ch_mrt

fi

cd ../../../
