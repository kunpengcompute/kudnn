#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
LLT_DIR=$(cd ${SCRIPT_DIR}/.. && pwd -P)
OUTPUT_DIR=$(cd ${SCRIPT_DIR}/../.. && pwd -P)
TEST_DIR=${LLT_DIR}/fuzz
TEST_RESULT_DIR=${LLT_DIR}/fuzz_out
TEST_LOG_DIR=${TEST_RESULT_DIR}/log
FUZZ_SUMMARY_LOG="${TEST_LOG_DIR}/fuzz_log_hikpex.log"

unset LD_LIBRARY_PATH
export BLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=${OUTPUT_DIR}/lib/libkplblas.so:${LD_PRELOAD}
export LD_PRELOAD=${OUTPUT_DIR}/lib/libomp.so:${LD_PRELOAD}

FUZZ_DETECT_KEYWORDS="failed\|Aborted\|fault\|dumped\|FAILED\|Exception\|AddressSanitizer\|crash\|Crash\|No[[:space:]]such[[:space:]]file"
fuzz_success_flag=1
function check_fuzz_result()
{
    operator=$1
    log_file="${TEST_LOG_DIR}/${operator}_fuzz.log"
    if [[ -z "`grep ${FUZZ_DETECT_KEYWORDS} ${log_file} -a`" ]];then
        echo "${operator} ... pass" >> ${FUZZ_SUMMARY_LOG}
    else
        echo "${operator} ... failed" >> ${FUZZ_SUMMARY_LOG}
        cat "${TEST_LOG_DIR}/${operator}_fuzz.log"
        fuzz_success_flag=0
    fi
}

function fuzz_test_operator()
{
    operator=$1
    echo "run fuzz test, operator: ${operator}"
    python ${TEST_DIR}/fuzz_test.py --operator=${operator} | tee ${TEST_LOG_DIR}/${operator}_fuzz.log
    check_fuzz_result ${operator}
}

function run_fuzz_test()
{
    fuzz_test_operator gating_attention
    fuzz_test_operator triangle_multiplication
    fuzz_test_operator outer_product_mean
    fuzz_test_operator multi_head_attention
    fuzz_test_operator linear
    fuzz_test_operator sigmoid
    fuzz_test_operator flash_attention
    fuzz_test_operator paged_attention
    fuzz_test_operator attention
    if [ ${fuzz_success_flag} -ne 0 ]; then
        echo "DT-Fuzz Test Success"
    else
        echo "DT-Fuzz Test Fail"
    fi
}

mkdir -p ${TEST_RESULT_DIR}
mkdir -p ${TEST_LOG_DIR}

run_fuzz_test