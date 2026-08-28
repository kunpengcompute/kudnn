#!/bin/bash

# default output path
CUR_PATH=$(cd `dirname $0` ; pwd)
ROOT_DIR=${CUR_PATH%test*}
BUILD_SCRIPT_DIR="${ROOT_DIR}/build"
TEST_SCRIPT_DIR=${CUR_PATH}
OUTPUT_PATH="${ROOT_DIR}/out"
TEST_TYPE=ut

# parse input parameters
for i in "$@"; do
    case $i in
        --output_dir=*)
        OUTPUT_PATH="${i#*=}"
        ;;
        --test_type=*)
        TEST_TYPE="${i#*=}"
        ;;
    esac
done

# build lib
cd ${ROOT_DIR}/test/dnn/llt/hdt
hdt build

# run test
hdt run --task ${TEST_TYPE}


