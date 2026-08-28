#!/bin/bash

set -Eeuo pipefail

# manual for the build script
function print_help(){
    echo "[Usage] sh build.sh --xxx=yyyy"
    echo "Parameters:"
    echo "--output_dir=absolute_path_to_output_lib_directory  "
    echo "--target_platform=[KP920B(default)|KP920F|KP920G]    |  hardware platform"
    echo "--build_type=[Release(default)|Debug]                |  build Release or Debug version lib"
    echo "--enable_gcov=[off(default)|on]                      |  build gcov version lib"
    echo "--enable_asan=[off(default)|on]                      |  build asan version lib"
    echo "--from_scratch=[off|on(default)]                     |  removes previous build files and builds from scratch"
    echo "--enable_kudnn=[on(default)|off]                      |  enable kudnn routine for oneDNN"
    echo "--only_kudnn=[off(default)|on]                        |  build only KuDNN without tests"
    echo "--only_tests=[off(default)|on]                       |  build only test without KuDNN"
    echo "--compiler=[gnu(default)|clang]                      |  compiler type，clang is recommended when MatrixCompute enabled"
    echo "--oneDNN_ver=[3.4(default)|3.7]                      |  set the specific oneDNN version"
    echo "--thread_impl=[omp(default)|threadpool]              |  threading implementation."
    return
}

# project root path
BUILD_SCRIPT_DIR=$(readlink -e "$(dirname "${0}")")
PROJECT_DIR=${BUILD_SCRIPT_DIR%/*}
KUDNN_SRC_DIR=${PROJECT_DIR}/src/dnn
DEPENDENCIES_DIR=${PROJECT_DIR}/dependencies
TEST_DIR=${PROJECT_DIR}/test/dnn
LLT_SRC_DIR=${TEST_DIR}/llt
FUZZ_SRC_DIR=${LLT_SRC_DIR}/fuzz
GTEST_SRC_DIR=${LLT_SRC_DIR}/gtest
ACL_SRC_DIR=${TEST_DIR}/acl
ONEDNN_VER=oneDNN-3.4
ACL_VER=23.11

# default parameter
OUTPUT_DIR=${PROJECT_DIR}/out
BUILD_TYPE=Release
TARGET_PLATFORM=KP920B

ENABLE_ASAN=off
ENABLE_GCOV=off
FROM_SCRATCH=on
ENABLE_KUDNN=on
ONLY_KUDNN=off
ONLY_TESTS=off
COMP_TOOLCHAIN=gnu
C_COMP_TO_USE=gcc
CXX_COMP_TO_USE=g++
PARALLEL_FACTOR=$(nproc --all)
THREAD_IMPL=OMP

# read input parameter
for i in "$@"; do
  case $i in
    --output_dir=*)
      OUTPUT_DIR="${i#*=}"
    ;;
    --target_platform=*)
      TARGET_PLATFORM="${i#*=}"
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
    --enable_kudnn=*)
      ENABLE_KUDNN="${i#*=}"
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
    --oneDNN_ver=*)
      ONEDNN_VER=oneDNN-"${i#*=}"
    ;;
    --thread_impl=*)
      THREAD_IMPL="${i#*=}"
    ;;
    --help|-h)
      print_help
      exit
    ;;
  esac
done

if [ "${ONLY_TESTS}" = "on" ]  && [ "${ONLY_KUDNN}" = "on" ]; then
    echo -e "these two parameters can NOT be set on at the same time\n"
    exit 1
fi

#if the ONLY_TEST is on, it is not neccesary for us to build from the very beginning
if [ "${ONLY_TESTS}" = "on" ]; then
    FROM_SCRATCH=off
fi


ONEDNN_SRC_DIR=${TEST_DIR}/${ONEDNN_VER}
if [[ ${ONEDNN_VER} == "oneDNN-3.4" ]]; then
    ACL_VER=23.11
    ACL_SRC_DIR=${TEST_DIR}/acl/23.11
else
    ACL_VER=25.02
    ACL_SRC_DIR=${TEST_DIR}/acl/25.02
fi

# output test directory
KUDNN_BUILD_DIR=${OUTPUT_DIR}/build
OUTPUT_TEST_DIR=${OUTPUT_DIR}/test/dnn
LLT_OUT_DIR=${OUTPUT_TEST_DIR}/llt
FUZZ_BUILD_DIR=${LLT_OUT_DIR}/fuzz
GTEST_BUILD_DIR=${LLT_OUT_DIR}/gtest
ACL_BUILD_DIR=${OUTPUT_TEST_DIR}/acl/${ACL_VER}/build
ONEDNN_BUILD_DIR=${OUTPUT_TEST_DIR}/${ONEDNN_VER}/build
ONEDNN_INSTALL_DIR=${OUTPUT_TEST_DIR}/${ONEDNN_VER}/install

# compiler tool chain
if [ ${COMP_TOOLCHAIN} = "clang" ]; then
    C_COMP_TO_USE=clang
    CXX_COMP_TO_USE=clang++
fi

# threadpool option
# DNNL_THREAD_CMAKE_OPTION=""
# if [ ${THREAD_IMPL^^} = "THREADPOOL" ]; then
#   DNNL_THREAD_CMAKE_OPTION="-DDNNL_CPU_RUNTIME=THREADPOOL -D_DNNL_TEST_THREADPOOL_IMPL=EIGEN"
# fi


# build Kunpeng Lib
function build_kdnn()
{
    ENABLE_HBM=off
    if [ ${TARGET_PLATFORM} = "KP920F" ]; then
         ENABLE_HBM=on
    fi
    echo -e "\033[32m=========Start To Build Kunpeng DNN Operators==============\033[0m"
    CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake ${PROJECT_DIR}/src/dnn/ -DTARGET_PLATFORM=${TARGET_PLATFORM} -DENABLE_HBM=${ENABLE_HBM} -DENABLE_GCOV=${ENABLE_GCOV} -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR} -DKUDNN_CPU_RUNTIME=${THREAD_IMPL^^} 2>&1 | tee ${KUDNN_BUILD_DIR}/build.log
    MAKEFLAGS="-j${PARALLEL_FACTOR}" CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} make VERBOSE=1 install -j $(nproc --all) 2>&1 | tee -a ${KUDNN_BUILD_DIR}/build.log

    echo -e "\033[32m=========Build Kunpeng DNN Operators Successfully==============\033[0m"
}

# copy llt scripts and inputs
function copy_llt_files()
{
  mkdir -p ${LLT_OUT_DIR}
  cp -r ${LLT_SRC_DIR}/inputs ${LLT_OUT_DIR}
  cp -r ${LLT_SRC_DIR}/scripts ${LLT_OUT_DIR}
}

# build gtest
function build_gtest()
{
  mkdir -p ${GTEST_BUILD_DIR} && cd ${GTEST_BUILD_DIR}
  echo -e "\033[32m=========Start To Build GTest==============\033[0m"
  KUDNN_ROOT_DIR=${OUTPUT_DIR} CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} KUDNN_SRC_DIRS=${KUDNN_SRC_DIR} cmake ${GTEST_SRC_DIR} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DTARGET_PLATFORM=${TARGET_PLATFORM} -DENABLE_ASAN=${ENABLE_ASAN} -DCOMP_TOOLCHAIN=${COMP_TOOLCHAIN} -DKUDNN_CPU_RUNTIME=OMP 2>&1 | tee -a ${KUDNN_BUILD_DIR}/build.log
  MAKEFLAGS="-j${PARALLEL_FACTOR}" CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} KUDNN_SRC_DIRS=${KUDNN_SRC_DIR} cmake --build . --parallel $(nproc --all) 2>&1 | tee -a ${GTEST_BUILD_DIR}/build.log
  echo -e "\033[32m=========Build GTest Successfully==============\033[0m"
}

# build fuzz test cases
function build_fuzz_test()
{
  mkdir -p ${FUZZ_BUILD_DIR} && cd ${FUZZ_BUILD_DIR}
  echo -e "\033[32m=========Start To Build Fuzz Test==============\033[0m"
  KUDNN_ROOT_DIR=${OUTPUT_DIR} CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} KUDNN_SRC_DIRS=${KUDNN_SRC_DIR} cmake ${FUZZ_SRC_DIR} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_ASAN=${ENABLE_ASAN} -DCOMP_TOOLCHAIN=${COMP_TOOLCHAIN} -DTARGET_PLATFORM=${TARGET_PLATFORM} 2>&1 | tee -a ${KUDNN_BUILD_DIR}/build.log
  MAKEFLAGS="-j${PARALLEL_FACTOR}" CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} KUDNN_SRC_DIRS=${KUDNN_SRC_DIR} cmake --build . --parallel $(nproc --all) 2>&1 | tee -a ${FUZZ_BUILD_DIR}/build.log
  echo -e "\033[32m=========Build Fuzz Tests Successfully==============\033[0m"
}

# build ACL
function build_acl()
{
    mkdir -p ${ACL_BUILD_DIR} &&  cd ${ACL_BUILD_DIR}
    # need to copy ACL sources because oneDNN uses internal ACL header files
    cp -r ${ACL_SRC_DIR}/* ${ACL_BUILD_DIR}/..
    echo -e "\033[32m=========Start To Build ARM Compute Library==============\033[0m"
    CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake ${ACL_SRC_DIR} -DCMAKE_BUILD_TYPE=Release -DARM_COMPUTE_OPENMP=1 -DARM_COMPUTE_WERROR=0 -DARM_COMPUTE_BUILD_EXAMPLES=0 -DARM_COMPUTE_BUILD_TESTING=0 -DCMAKE_INSTALL_LIBDIR=${ACL_BUILD_DIR} 2>&1 | tee -a ${ACL_BUILD_DIR}/build.log
    MAKEFLAGS="-j${PARALLEL_FACTOR}" CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake --build . --parallel $(nproc --all) 2>&1 | tee -a ${ACL_BUILD_DIR}/build.log
    echo -e "\033[32m=========Build ARM Compute Library Successfully==============\033[0m"
}

# build oneDNN
function build_onednn()
{
    mkdir -p ${ONEDNN_BUILD_DIR} &&  cd ${ONEDNN_BUILD_DIR}
    echo -e "\033[32m=========Start To Build oneDNN==============\033[0m"
    KUDNN_ROOT_DIR=${OUTPUT_DIR} ACL_ROOT_DIR=${ACL_BUILD_DIR}/.. CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake ${ONEDNN_SRC_DIR} -DDNNL_AARCH64_USE_ACL=ON -DENABLE_ASAN=${ENABLE_ASAN^^} -DDNNL_AARCH64_USE_KUDNN=${ENABLE_KUDNN^^} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} 2>&1 | tee -a ${ONEDNN_BUILD_DIR}/build.log
    MAKEFLAGS="-j${PARALLEL_FACTOR}" KUDNN_ROOT_DIR=${OUTPUT_DIR} ACL_ROOT_DIR=${ACL_BUILD_DIR}/.. CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake --build . --parallel $(nproc --all) 2>&1 | tee -a ${ONEDNN_BUILD_DIR}/build.log
    cmake --install ${ONEDNN_BUILD_DIR} --prefix ${ONEDNN_INSTALL_DIR} 2>&1 | tee -a ${ONEDNN_BUILD_DIR}/build.log
    echo -e "\033[32m=========Build oneDNN Successfully==============\033[0m"
}

# build ut
function build_test()
{
  if [ "${ENABLE_KUDNN}" = "on" ]; then
    build_gtest
    build_fuzz_test
  fi
    build_acl
    build_onednn
}

if [ "${FROM_SCRATCH}" = "on" ]; then
    # remove existed old build and output directory
    if [ -d ${OUTPUT_DIR} ]; then
        rm -r ${OUTPUT_DIR}
    fi
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${KUDNN_BUILD_DIR} && cd ${KUDNN_BUILD_DIR}

if [ "${ONLY_TESTS}" = "off" ] && [ "${ENABLE_KUDNN}" = "on" ]; then
    build_kdnn
fi

if [ "${ONLY_KUDNN}" = "off" ]; then
    build_test
    copy_llt_files
fi

set +Eeuo pipefail
