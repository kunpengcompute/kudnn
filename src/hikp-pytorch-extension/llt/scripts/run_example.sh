#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
LLT_DIR=$(cd ${SCRIPT_DIR}/.. && pwd -P)
OUTPUT_DIR=$(cd ${SCRIPT_DIR}/../.. && pwd -P)
TEST_DIR=${LLT_DIR}/hikpex_test

THREADS=64
ITEM=function
ITERATION=10

for i in "$@"; do
  case $i in
    --module=*)
      MODULE="${i#*=}"
    ;;
    --operator=*)
      OPERATOR="${i#*=}"
    ;;
    --para=*)
      PARA="${i#*=}"
    ;;
    --item=*)
      ITEM="${i#*=}"
    ;;
    --iteration=*)
      ITERATION="${i#*=}"
    ;;
    --thread=*)
      THREADS="${i#*=}"
    ;;
  esac
done

unset LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH}:${TEST_DIR}"
export BLAS_NUM_THREADS=1
export OMP_NUM_THREADS=${THREADS}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=${OUTPUT_DIR}/lib/libkplblas.so:${LD_PRELOAD}
export LD_PRELOAD=${OUTPUT_DIR}/lib/libomp.so:${LD_PRELOAD}

python ${TEST_DIR}/test.py \
    --para=${PARA} \
    --item=${ITEM} \
    --iteration=${ITERATION} \
    --module=${MODULE} \
    --operator=${OPERATOR}