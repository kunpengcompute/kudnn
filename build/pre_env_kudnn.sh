#!/bin/bash
#------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# KuDNN is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#        http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#-----------------------------------------------------------------------------------

set -Eeuo pipefail

# project root path
BUILD_SCRIPT_DIR=$(readlink -e "$(dirname "${0}")")
PROJECT_DIR=${BUILD_SCRIPT_DIR%/*}

if [ "${PROJECT_DIR}" = "" ]; then
    echo "PROJECT_DIR is error."
    exit 1
fi

OPENSOURCE_PATH=${PROJECT_DIR}/opensource
OPENSOURCE_INSTALL_PATH=${PROJECT_DIR}/opensource/install

GTEST_SRC_PATH=${PROJECT_DIR}/opensource/googletest
GTEST_INSTALL_PATH=${PROJECT_DIR}/opensource/install/googletest
GTEST_INSTALL_INCLUDES=${PROJECT_DIR}/opensource/install/googletest/include
GTEST_INSTALL_LIBS=${PROJECT_DIR}/opensource/install/googletest/lib64

KUDNN_GTEST_INCLUDES=${PROJECT_DIR}/test/dnn/llt/gtest/include
# Bisheng
KUDNN_GTEST_LIBS=${PROJECT_DIR}/test/dnn/llt/gtest/gtest_lib/bisheng

install_googletest() {
  [ -n "${GTEST_INSTALL_PATH}" ] && rm -rf "${GTEST_INSTALL_PATH}"
  echo "${GTEST_INSTALL_PATH}"
  if [[ ! -d "${GTEST_INSTALL_PATH}" ]]; then
    mkdir -p "${GTEST_INSTALL_PATH}"
  fi

  cd "${GTEST_SRC_PATH}"
  echo "${GTEST_SRC_PATH}"
  export CC=clang
  export CXX=clang++
  cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_INSTALL_PREFIX="${GTEST_INSTALL_PATH}" . && make -j$(nproc --all)&& make install
}
if [ -d "${OPENSOURCE_PATH}" ]; then
  rm -fr ${OPENSOURCE_PATH}
fi
mkdir -p ${OPENSOURCE_PATH}
cd ${OPENSOURCE_PATH}
git clone https://github.com/google/googletest.git -b v1.15.2 --depth 1
install_googletest

if [ -d "${KUDNN_GTEST_INCLUDES}/" ]; then
  rm -fr ${KUDNN_GTEST_INCLUDES}/*
fi

if [ -d "${KUDNN_GTEST_LIBS}/" ]; then
  rm -fr ${KUDNN_GTEST_LIBS}/*
fi

mkdir -p ${KUDNN_GTEST_INCLUDES}
mkdir -p ${KUDNN_GTEST_LIBS}
cp -af ${GTEST_INSTALL_INCLUDES}/*   ${KUDNN_GTEST_INCLUDES}
cp -af ${GTEST_INSTALL_LIBS}/*.a   ${KUDNN_GTEST_LIBS}

set +Eeuo pipefail
