#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

TEST_TYPE=ut
GEN_COVERAGE_REPORT=off
THREAD_NUM=64
TARGET_PLATFORM=KP920F

for i in "$@"; do
  case $i in
    --type=*)
      TEST_TYPE="${i#*=}"
    ;;
    --gen_report=*)
      GEN_COVERAGE_REPORT="${i#*=}"
    ;;
    --thread_num=*)
      THREAD_NUM="${i#*=}"
    ;;
    --target_platform=*)
      TARGET_PLATFORM="${i#*=}"
    ;;
  esac
done

LLT_DIR=$(cd ${SCRIPT_DIR}/.. && pwd -P)
OUTPUT_DIR=$(cd ${SCRIPT_DIR}/../.. && pwd -P)
TEST_RESULT_DIR=${LLT_DIR}/${TEST_TYPE}_out
TEST_LOG_DIR=${TEST_RESULT_DIR}/log
TEST_REPORT_DIR=${TEST_RESULT_DIR}/report
TEST_COVERAGE_INFO_DIR=${TEST_REPORT_DIR}/coverage_info
TEST_COVERAGE_REPORT_DIR=${TEST_REPORT_DIR}/coverage_report

tst_count=0
tst_success_count=0
tst_failure_count=0
ut_success_flag=0

function prepare_test()
{
    [[ -d "${TEST_RESULT_DIR}" ]] && rm -rf ${TEST_RESULT_DIR}
    mkdir -p ${TEST_RESULT_DIR}
    [[ -d "${TEST_LOG_DIR}" ]] && rm -rf ${TEST_LOG_DIR}
    mkdir -p ${TEST_LOG_DIR}

    [[ -d "${TEST_REPORT_DIR}" ]] && rm -rf ${TEST_REPORT_DIR}
    mkdir -p ${TEST_REPORT_DIR}
    if [[ ${GEN_COVERAGE_REPORT} == "on" ]] ;then
        [[ -d "${TEST_COVERAGE_REPORT_DIR}" ]] && rm -rf ${TEST_COVERAGE_REPORT_DIR}
        mkdir -p ${TEST_COVERAGE_REPORT_DIR}
        [[ -d "${TEST_COVERAGE_INFO_DIR}" ]] && rm -rf ${TEST_COVERAGE_INFO_DIR}
        mkdir -p ${TEST_COVERAGE_INFO_DIR}
    fi
}

function profiling_log()
{
    local msg_level format_time
    msg_level=$(echo "${1}" | tr '[:lower:]' '[:upper:]')
    local log_msg="${2}"
    format_time="$(date "+%a %d %b %Y %T")"
    echo "${format_time} [${msg_level}] [HIKPEX] ${log_msg}"
}

# generate coverage report
function gen_coverage_report()
{
    search_lcov=`which lcov | grep "no lcov"`
    search_genhtml=`which lcov | grep "no genhtml"`
    if [[ ${search_lcov} != "" ]] || [[ ${search_genhtml} != "" ]]; then
        echo "lcov or genhtml tool is not found !!!"
        exit 1
    fi

    search_llvm_cov=`which llvm-cov | grep "no llvm-cov"`
    if [[ ${search_llvm_cov} != "" ]]; then
        echo "llvm-cov tool is not found !!!"
        exit 1
    fi

    echo "[LLT]lcov starttime: `date +%F' '%H:%M:%S`"
    cd ${OUTPUT_DIR}

    chmod 777 ${SCRIPT_DIR}/llvm_gcov.sh
    echo "${SCRIPT_DIR}/llvm_gcov.sh"
    echo "${TEST_COVERAGE_INFO_DIR}"
    lcov -c -d ${OUTPUT_DIR}/build -o ${TEST_COVERAGE_INFO_DIR}/coverage.info --rc lcov_branch_coverage=1 --gcov-tool ${SCRIPT_DIR}/llvm_gcov.sh


    lcov --remove ${TEST_COVERAGE_INFO_DIR}/coverage.info '*usr*' '*ATen*'  '*c10*' '*pybind*' '*anaconda*' 'csrc/hikp.cpp' -o ${TEST_COVERAGE_INFO_DIR}/coverage.info --rc lcov_branch_coverage=1
    genhtml --branch-coverage -o ${TEST_COVERAGE_REPORT_DIR} ${TEST_COVERAGE_INFO_DIR}/*.info
    RET=$?
    echo "[LLT]lcov stoptime: `date +%F' '%H:%M:%S`"
    if [[ "${RET}" -ne "${SUCCESS}" ]]; then
        profiling_log 'error' 'lcov failed'
        exit 1
    fi
}

function run_test()
{
    module=$1
    operator=$2
    para=$3
    tst_count=`expr ${tst_count} + 1`
    echo "Starting Test Operator : ${operator}, para=${para}"
    sh ${SCRIPT_DIR}/run_example.sh --module=${module} --operator=${operator} --para=${para} --thread=${THREAD_NUM} >> ${TEST_LOG_DIR}/test_log.log
}

function check_test_result()
{
    file_path=${TEST_LOG_DIR}/test_log.log

    tst_success_count=`grep -Fx "Forward pass check:  True" ${file_path} | wc -l`

    if [ -z ${tst_success_count} ]; then
        passed_tst_cnt=0
    fi

    tst_failure_count=`expr ${tst_count} - ${tst_success_count}`

    if [ ${tst_success_count} = ${tst_count} ]; then
        ut_success_flag=1
    fi

    if [ ${ut_success_flag} -ne 0 ]; then
        echo "[HIKPEX] ${TEST_TYPE} test success"
    else
        echo "[HIKPEX] ${TEST_TYPE} test fail"
    fi

}

function gen_test_summary()
{
    echo '<?xml version="1.0" encoding="UTF-8"?>' > ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
    echo "<testsuites tests=\"${tst_count}\" pass=\"${tst_success_count}\" failure=\"${tst_failure_count}\">" >> ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
    echo '</testsuites>' >> ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
}

function test_flash_attention()
{
    input_type=("fp32" "fp16")
    mask_type=("fp32" "fp16")

    for type1 in "${input_type[@]}"; do
        for type2 in "${mask_type[@]}"; do
            run_test common flash_attention 1,1,3840,3840,72,1,1,${type1},${type2}
            run_test common flash_attention 1,1,3840,3840,72,1,0,${type1},${type2}
            run_test common flash_attention 1,1,128,128,72,1,0,${type1},${type2}
            run_test common flash_attention 1,1,256,256,72,1,0,${type1},${type2}
        done
    done
}

function test_common_attention()
{
    input_type=("fp32" "fp16" "bf16")
    mask_type=("fp32" "fp16" "bf16")

    for type1 in "${input_type[@]}"; do
        for type2 in "${mask_type[@]}"; do
            run_test common attention 1,1,64,64,128,1,1,${type1},${type2}
            run_test common attention 1,1,512,512,128,1,0,${type1},${type2}
            run_test common attention 1,1,768,768,128,1,0,${type1},${type2}
            run_test common attention 1,1,1536,1536,128,1,0,${type1},${type2}
            run_test common attention 1,1,2048,2048,128,1,0,${type1},${type2}
        done
    done
}

function run_ut_test()
{
    run_test common multi_head_attention 1,256,32,8,1
    run_test common multi_head_attention 1,255,32,8,0
    run_test common linear 128,128,1
    run_test common linear 128,200,0
    run_test common sigmoid 128
    run_test alphafold gating_attention 1,256,32,8,1
    run_test alphafold outer_product_mean 1,256,32,32,32
    run_test alphafold triangle_multiplication 255,32,32,0
    run_test alphafold triangle_multiplication 255,32,32,1

    run_test common paged_attention 4,40,8,512,128,16,fp16

    # flash attention
    test_flash_attention

    # common attention
    test_common_attention

    check_test_result

    gen_test_summary
}

function run_fuzz_test()
{
    sh ${SCRIPT_DIR}/test_fuzz.sh
}

export GCOV_PREFIX=${OUTPUT_DIR}
export GCOV_PREFIX_STRIP=999
prepare_test

if [ ${TEST_TYPE} = "fuzz" ]; then
    run_fuzz_test
else
    run_ut_test
fi

if [ ${GEN_COVERAGE_REPORT} == "on" ]; then
    gen_coverage_report
fi