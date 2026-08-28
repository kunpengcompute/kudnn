#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR=$(readlink -e "$(dirname "${0}")")

# default parameter
BUILD_TYPE=release
TARGET_PLATFORM=KP920F
ENABLE_GCOV=off
ENABLE_ASAN=off
OUTPUT_DIR=${SCRIPT_DIR}/out
DEPENDENCIES_DIR=${SCRIPT_DIR}/lib
BUILD_BLAS=on

# read input parameter
for i in "$@"; do
  case $i in
    --target_platform=*)
      TARGET_PLATFORM="${i#*=}"
    ;;
    --build_type=*)
      BUILD_TYPE="${i#*=}"
    ;;
    --output_dir=*)
      OUTPUT_DIR="${i#*=}"
    ;;
    --enable_gcov=*)
      ENABLE_GCOV="${i#*=}"
    ;;
    --enable_asan=*)
      ENABLE_ASAN="${i#*=}"
    ;;
    --dependencies_dir=*)
      DEPENDENCIES_DIR="${i#*=}"
    ;;
    --build_blas=*)
      BUILD_BLAS="${i#*=}"
    ;;
  esac
done

BLAS_SOURCE_DIR=${SCRIPT_DIR}/KPL_BLAS
EXETERN_LIB_DIR=${OUTPUT_DIR}/lib
LLT_OUT_DIR=${OUTPUT_DIR}/llt

function copy_llt_files()
{
    DIR_LEVEL=$(echo ${SCRIPT_DIR} | awk -F'/' '{print NF-1}')
    mkdir ${LLT_OUT_DIR}
    cp -r ${SCRIPT_DIR}/llt/scripts ${LLT_OUT_DIR}
    cp -r ${SCRIPT_DIR}/llt/hikpex_test ${LLT_OUT_DIR}
    sed -i "s/export GCOV_PREFIX_STRIP=999/export GCOV_PREFIX_STRIP=${DIR_LEVEL}/g" ${LLT_OUT_DIR}/scripts/test_hikpex.sh
    if [ ${ENABLE_ASAN} == "on" ]; then
        mkdir -p ${LLT_OUT_DIR}/fuzz
        cp ${SCRIPT_DIR}/llt/fuzz/fuzz_test.py ${LLT_OUT_DIR}/fuzz
    else
        rm ${LLT_OUT_DIR}/scripts/test_fuzz.sh
    fi
}

function build_blas()
{
    sed -i "s/ENABLE_OPENMP=0/ENABLE_OPENMP=1/g" ${BLAS_SOURCE_DIR}/blas/build/build.sh
    if [ ${TARGET_PLATFORM} = "KP920F" ]; then
        IS_BISHENG=enable sh ${BLAS_SOURCE_DIR}/blas/build/build.sh blas multi release arm64 KP1636SVE
    else
        IS_BISHENG=enable sh ${BLAS_SOURCE_DIR}/blas/build/build.sh blas multi release arm64 KP1630
    fi
    cp ${BLAS_SOURCE_DIR}/blas/out/release/kplblas/lib/*.so ${EXETERN_LIB_DIR}
}

function post_build_ops()
{
    mv ${SCRIPT_DIR}/dist ${OUTPUT_DIR}/
    mv ${SCRIPT_DIR}/build ${OUTPUT_DIR}/
    rm -rf ${SCRIPT_DIR}/src/*.egg-info
}

function install_hikpex()
{
    HIKPEX_WHL_NAME=$(ls ${OUTPUT_DIR}/dist | grep hikp* | head -n 1)
    pip install ${OUTPUT_DIR}/dist/${HIKPEX_WHL_NAME} --force-reinstall
}

export CC=clang
export CXX=clang++
export HIKPEX_BUILD_TYPE=${BUILD_TYPE}
export HIKPEX_ENABLE_GCOV=${ENABLE_GCOV}
export HIKPEX_ENABLE_ASAN=${ENABLE_ASAN}
export HIKPEX_TARGET_PLATFORM=${TARGET_PLATFORM}
export HIKPEX_OUTPUT_DIR=${OUTPUT_DIR}
export HIKPEX_DEPENDENCIES_DIR=${DEPENDENCIES_DIR}

if [ -d ${OUTPUT_DIR} ]; then
    rm -rf ${OUTPUT_DIR}
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${EXETERN_LIB_DIR}

cp ${DEPENDENCIES_DIR}/*.so ${EXETERN_LIB_DIR}/

if [ ${BUILD_BLAS} == "on" ]; then
  build_blas
fi

python ${SCRIPT_DIR}/setup.py clean --all
python ${SCRIPT_DIR}/setup.py bdist_wheel

post_build_ops

install_hikpex

copy_llt_files

set +Eeuo pipefail