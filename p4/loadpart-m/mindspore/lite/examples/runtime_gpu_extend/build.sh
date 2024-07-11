#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)
get_version() {
    VERSION_MAJOR=$(grep "const int ms_version_major =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
    VERSION_MINOR=$(grep "const int ms_version_minor =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
    VERSION_REVISION=$(grep "const int ms_version_revision =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}
get_version
MODEL_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/quick_start/add_extend.ms"
MODEL_DOWNLOAD_URL2="https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.ms"
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/linux/x86_64/${MINDSPORE_FILE}"

mkdir -p build
mkdir -p model
if [ ! -e ${BASEPATH}/model/add_extend.ms ]; then
    wget -c -O ${BASEPATH}/model/add_extend.ms --no-check-certificate ${MODEL_DOWNLOAD_URL}
fi
if [ ! -e ${BASEPATH}/model/add.ms ]; then
    wget -c -O ${BASEPATH}/model/add.ms --no-check-certificate ${MODEL_DOWNLOAD_URL2}
fi
if [ ! -e ${BASEPATH}/build/${MINDSPORE_FILE} ]; then
  wget -c -O ${BASEPATH}/build/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
fi
tar -xzf ${BASEPATH}/build/${MINDSPORE_FILE}
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/runtime ${BASEPATH}/
cd ${BASEPATH}/build || exit
cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19" \
   -DANDROID_ABI="arm64-v8a"  -DCMAKE_BUILD_TYPE="Release" ${BASEPATH}
make