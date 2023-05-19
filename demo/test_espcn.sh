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
FILE_PATH=`dirname "$(realpath $0)"`
CORE_DUMP="${FILE_PATH}/../core/inferenceCoreDump"
MODEL_ZOO="${FILE_PATH}/../modelzoo"
IMAGE_DIR="${FILE_PATH}/../core/data/assets/images"
PY_TOOLS="${FILE_PATH}/../tools/misc/"
ESPCN_DIR="ESPCN_2X_16_16_4.json layer [04] subpixel"
ESPCN_OUT="ESPCN_2X_16_16_4.json layer [04] subpixel pass[0]"

print_usage()
{
    echo
    echo "Usage: `basename $0` [installDeps /*install python dependencies*/]"
    echo "  installDeps - install python dependencies"
    echo
}

if [ "$1" == "--help" ]; then
    print_usage
    exit 0
fi

if [ "$1" == "installDeps" ]; then
    pip install --upgrade pip
    pip install opencv-python
    pip install tensorflow
fi

if [ -z  "$NO_REBUILD_IN_TESTS" ]; then
    ${FILE_PATH}/build-tests.sh clean
    ${FILE_PATH}/build-tests.sh linux
fi

rm -rf ${CORE_DUMP}/*;
cd ${FILE_PATH}/build-test/test/unittest/ && ./inferenceProcessorTest --use_vulkan --dump_outputs espcn2x

python3 ${PY_TOOLS}/readTextureDump.py --path "${CORE_DUMP}/ESPCN/${ESPCN_OUT}.dump" --width 448 --height 448 --channels 4 --image save --normalization 0-1

python3 ${FILE_PATH}/modelInferenceESPCN.py --model ${MODEL_ZOO}/ESPCN/ESPCN_2X_16_16_4.h5 --image ${IMAGE_DIR}/bright_night_view_street_1080x1920.jpg --outdir ${CORE_DUMP}/ESPCN/

python3 ${PY_TOOLS}/imageComparison.py --image1 ${CORE_DUMP}/ESPCN/espcnResult.png --image2 "${CORE_DUMP}/ESPCN/${ESPCN_DIR}/00.png" --outdir ${CORE_DUMP}/ESPCN/
