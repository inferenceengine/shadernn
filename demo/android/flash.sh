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

#adb root
#adb remount
#adb shell setenforce 0

clean_phone_json_models()
{
    adb shell rm -rf /data/local/tmp/jsonModel
    adb shell mkdir /data/local/tmp/jsonModel
    adb shell rm -rf /data/local/tmp/files
    adb shell mkdir -p /data/local/tmp/files
    adb shell rm -rf /data/local/tmp/inferenceCoreDump
    adb shell mkdir -p /data/local/tmp/inferenceCoreDump
}

push_phone_json_models()
{   
    clean_phone_json_models
    adb push ${ROOT}/../../modelzoo/ESPCN /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/MobileNetV2 /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/Resnet18 /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/SpatialDenoise /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/U-Net /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/Yolov3-tiny /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/StyleTransfer /data/local/tmp/jsonModel
    adb push ${ROOT}/../../core/data/assets /data/local/tmp/files
    adb shell chmod -R 777 /data/local/tmp/files
    adb shell chmod -R 777 /data/local/tmp/jsonModel
}

if [ "$1" == "models" ]; then
    push_phone_json_models
else 
    ${ROOT}/gradlew installDebug
fi

