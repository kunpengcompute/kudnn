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

# manual for the build script
function print_help(){
    echo "[Usage] sh build.sh --xxx=xxx"
    echo "Parameters:"
    echo "--install_path=absolute_path_to_output_lib_directory  "
    echo "--build_type=[Release(default)|Debug]                |  build Release or Debug version lib"
    echo "--enable_gcov=[off(default)|on]                      |  build gcov version lib"
    echo "--enable_asan=[off(default)|on]                      |  build asan version lib"
    echo "--from_scratch=[off|on(default)]                     |  removes previous build files and builds from scratch"
    echo "--only_kudnn=[off(default)|on]                       |  build only KuDNN without tests"
    echo "--only_tests=[off(default)|on]                       |  build only test without KuDNN"
    echo "--compiler=[gcc(default)|clang]                      |  compiler type, clang is recommended when MatrixComputation enabled"
    return
}

# project root path
BUILD_SCRIPT_DIR=$(readlink -e "$(dirname "${0}")")
PROJECT_DIR=${BUILD_SCRIPT_DIR%/*}
KUDNN_SRC_DIR=${PROJECT_DIR}/src/dnn
TEST_DIR=${PROJECT_DIR}/test/dnn
LLT_SRC_DIR=${PROJECT_DIR}/test/dnn/llt
FUZZ_SRC_DIR=${PROJECT_DIR}/test/dnn/llt/fuzz
GTEST_SRC_DIR=${PROJECT_DIR}/test/dnn/llt/gtest

# default parameter
OUTPUT_DIR=${PROJECT_DIR}/out
BUILD_TYPE=Release
TARGET_PLATFORM=KP920Pro
ENABLE_ASAN=off
ENABLE_GCOV=off
FROM_SCRATCH=on
ENABLE_KUDNN=on
ONLY_KUDNN=off
ONLY_TESTS=off
DNNL_CODE_COVERAGE=OFF
COMP_TOOLCHAIN=gcc
C_COMP_TO_USE=gcc
CXX_COMP_TO_USE=g++

# read input parameter
for i in "$@"; do
  case "$i" in
    --install_path=*)
      OUTPUT_DIR="${i#*=}"
    ;;
    --build_type=*)
      BUILD_TYPE="${i#*=}"
    ;;
    --enable_gcov=*)
      ENABLE_GCOV="${i#*=}"
    ;;
    --enable_asan=*)
      ENABLE_ASAN="${i#*=}"
    ;;
    --from_scratch=*)
      FROM_SCRATCH="${i#*=}"
    ;;
    --only_kudnn=*)
      ONLY_KUDNN="${i#*=}"
    ;;
    --only_tests=*)
      ONLY_TESTS="${i#*=}"
    ;;
    --compiler=*)
      COMP_TOOLCHAIN="${i#*=}"
    ;;
    --help|-h)
      print_help
      exit
    ;;
  esac
done

# output test directory
KUDNN_BUILD_DIR=${PROJECT_DIR}/out/build
OUTPUT_TEST_DIR=${PROJECT_DIR}/out/test/dnn
LLT_OUT_DIR=${PROJECT_DIR}/out/test/dnn/llt
FUZZ_BUILD_DIR=${PROJECT_DIR}/out/test/dnn/llt/fuzz
GTEST_BUILD_DIR=${PROJECT_DIR}/out/test/dnn/llt/gtest

if [ "${ONLY_TESTS}" = "on" ]  && [ "${ONLY_KUDNN}" = "on" ]; then
    echo -e "these two parameters can NOT be set on at the same time\n"
    exit 1
fi
 
#if the ONLY_TEST is on, it is not neccesary for us to build from the very beginning
if [ "${ONLY_TESTS}" = "on" ]; then
    FROM_SCRATCH=off 
fi

# compiler tool chain
if [ "${COMP_TOOLCHAIN}" = "clang" ]; then
    C_COMP_TO_USE=clang
    CXX_COMP_TO_USE=clang++
fi

# build Kunpeng Lib
function build_lib()
{
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${KUDNN_BUILD_DIR} && cd ${KUDNN_BUILD_DIR}
    echo -e "\033[32m=========Start To Build Kunpeng DNN Operators==============\033[0m"
    CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake ${PROJECT_DIR}/src/dnn/ -DTARGET_PLATFORM=${TARGET_PLATFORM} -DENABLE_GCOV=${ENABLE_GCOV} -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR} 2>&1 | tee ${KUDNN_BUILD_DIR}/build.log
    CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} make VERBOSE=1 install -j $(nproc --all) 2>&1 | tee -a ${KUDNN_BUILD_DIR}/build.log
    echo -e "\033[32m=========Build Kunpeng DNN Operators Successfully==============\033[0m"
}

# build gtest
function build_gtest()
{
  mkdir -p ${GTEST_BUILD_DIR} && cd ${GTEST_BUILD_DIR}
  echo -e "\033[32m=========Start To Build GTest==============\033[0m"
  KUDNN_ROOT_DIR=${OUTPUT_DIR} CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} KUDNN_SRC_DIRS=${KUDNN_SRC_DIR} cmake ${GTEST_SRC_DIR} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_ASAN=${ENABLE_ASAN} -DENABLE_GCOV=${ENABLE_GCOV} -DCOMP_TOOLCHAIN=${COMP_TOOLCHAIN} 2>&1 | tee -a ${KUDNN_BUILD_DIR}/build.log
  echo -e "\033[32m=========Build GTest Successfully==============\033[0m"

  CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake --build . --parallel $(nproc --all)  2>&1 | tee -a ${GTEST_BUILD_DIR}/build.log
  echo -e "\033[32m=========Build GTest Successfully==============\033[0m"
}

# build ut
function build_test()
{
    build_gtest
}



if [ "${FROM_SCRATCH}" = "on" ]; then
    # remove existed old build and output directory
    if [ -d "${OUTPUT_DIR}" ]; then
        rm -r ${OUTPUT_DIR}
    fi
fi

if [ "${ONLY_TESTS}" = "off" ]; then
    build_lib
fi

if [ "${ONLY_KUDNN}" = "off" ]; then
    build_test
fi

set +Eeuo pipefail