#!/bin/bash
# help function

function print_help()
{
    echo "test_dnn.sh"
    echo "Parameters:"
    echo "--output_dir=path_to_output_dir            | output directory after build the library"
    echo "--test_type=[ut(default)|st|fuzz]          | unit test, system test, fuzz test"
    echo "--gen_report=[off(default)|on]             | create coverage rate report "
    echo "--only_gen_report=[off(default)|on]        | only create coverage rate report while do not run tests "
    echo "--compiler=[gnu(default)|clang]            | Compiler used to build package "
    echo "--use_fastcov=[on(default)|off]            | use fastcov to speedup lcov process "
    echo "--thread_type=[single(default)|multi]      | single: run single thread test; multi: run single and multi-thread test"
    echo "--help                                     | print user guide"
    return
}

# default output path
CUR_PATH=$(cd `dirname $0` ; pwd)
ROOT_DIR=${CUR_PATH%test*}
OUTPUT_PATH=${ROOT_DIR}/out
TEST_TYPE=ut
THREAD_TYPE=single
GEN_REPORT=off
FAST_COV=on
# default report generation option
ONLY_GEN_REPORT=off
COMPILER=gnu

# parse input parameters
for i in "$@"; do
    case $i in
        --output_dir=*)
        OUTPUT_PATH="${i#*=}"
        ;;
        --test_type=*)
        TEST_TYPE="${i#*=}"
        ;;
        --gen_report=*)
        GEN_REPORT="${i#*=}"
        ;;
        --only_gen_report=*)
        ONLY_GEN_REPORT="${i#*=}"
        ;;
        --compiler=*)
        COMPILER="${i#*=}"
        ;;
        --thread_type=*)
        THREAD_TYPE="${i#*=}"
        ;;
        --use_fastcov=*)
        FAST_COV="${i#*=}"
        ;;
        --help|-h)
        print_help
        exit
        ;;
    esac
done

# test output directory
OUTPUT_TEST_DIR="${OUTPUT_PATH}/test/dnn"
BENCHDNN_DIR="${OUTPUT_TEST_DIR}/oneDNN-3.4/build/tests/benchdnn"
BENCHDNN_BIN="${BENCHDNN_DIR}/benchdnn"
BENCHDNN_INPUTS_DIR="${BENCHDNN_DIR}/inputs"
LLT_OUTPUT_DIR="${OUTPUT_TEST_DIR}/llt"
LLT_INPUTS_DIR="${LLT_OUTPUT_DIR}/inputs"
TEST_OUT_DIR="${OUTPUT_PATH}/${TEST_TYPE}_out"
TEST_LOG_DIR="${TEST_OUT_DIR}/log"
TEST_REPORT_DIR="${TEST_OUT_DIR}/report"
TEST_COVERAGE_RESULT_DIR="${TEST_REPORT_DIR}/coverage_result"
TEST_COVERAGE_INFO_DIR="${TEST_REPORT_DIR}/coverage_info"

# gtest binaries
GTEST_OUTPUT_DIR="${LLT_OUTPUT_DIR}/gtest"
GTEST_BIN="${GTEST_OUTPUT_DIR}/kudnn_gtest"


# fuzz test binaries
FUZZ_OUTPUT_DIR="${LLT_OUTPUT_DIR}/fuzz"
CODEGEN_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/codegen_fuzz
ELTWISE_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/eltwise_fuzz
SOFTMAX_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/softmax_fuzz
BINARY_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/binary_fuzz
PRELU_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/prelu_fuzz
MATMUL_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/matmul_fuzz
REDUCTION_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/reduction_fuzz
INNER_PRODUCT_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/inner_product_fuzz
LNORM_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/lnorm_fuzz
CONVOLUTION_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/convolution_fuzz
DECONV_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/deconvolution_fuzz
SUM_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/sum_fuzz
TENSOR_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/tensor_fuzz
REORDER_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/reorder_fuzz
SHUFFLE_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/shuffle_fuzz
CONCAT_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/concat_fuzz
RESAMPLING_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/resampling_fuzz
POOL_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/pool_fuzz
BNORM_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/bnorm_fuzz
LRN_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/lrn_fuzz
GNORM_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/gnorm_fuzz
RNN_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/rnn_fuzz
POST_OPS_FUZZ_TEST=${FUZZ_OUTPUT_DIR}/post_ops_fuzz
FUZZ_SUMMARY_LOG="${TEST_LOG_DIR}/Fuzz_log_dnn.log"

# test counts
all_tsts_amount=0
passed_tsts_amount=0
skip_tsts_amount=0
mistrust_tsts_amount=0
unimpl_tsts_amount=0
invalid_tsts_amount=0
failed_tsts_amount=0
listed_tsts_amount=0
kdnn_unsupported_amount=0

ut_success_flag=1
fuzz_success_flag=1
cmd_pids=()
cmd_list=()

run_command() {
    echo "Starting command : $1"
    eval $1 &
    cmd_pids+=($!)
    cmd_list+=("$1")
}

check_return_values() {
    for (( i=0; i<${#cmd_pids[@]}; i++ )); do
        wait ${cmd_pids[i]}
        return_value=$?
        if [ $return_value -ne 0 ]; then
            failed_cmd=${cmd_list[i]}
            file_name=${failed_cmd#*>> }
            echo $file_name
            echo "--------ERROR--------"
            echo "FAIL COMMAND: $failed_cmd failed with status $return_value."
            echo "FAIL COMMAND: $failed_cmd failed with status $return_value." >> $file_name
            ut_success_flag=0
            exit 1
        fi
    done
    cmd_pids=()
    cmd_list=()
}

function profiling_log()
{
    # Install script format logger to stdout.
    local msg_level format_time
    msg_level=$(echo "${1}" | tr '[:lower:]' '[:upper:]')
    local log_msg="${2}"
    format_time="$(date "+%a %d %b %Y %T")"
    echo "${format_time} [${msg_level}] [DNN] ${log_msg}"
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

    if [[ ${COMPILER} == "clang" ]]; then
        search_llvm_cov=`which llvm-cov | grep "no llvm-cov"`
        if [[ ${search_llvm_cov} != "" ]]; then
            echo "llvm-cov tool is not found !!!"
            exit 1
        fi
    fi

    echo "[LLT]lcov starttime: `date +%F' '%H:%M:%S`"
    cd ${OUTPUT_PATH}

    if [[ ${COMPILER} == "clang" ]]; then
        chmod 777 ${CUR_PATH}/llvm_gcov.sh
        echo "${CUR_PATH}/llvm_gcov.sh"
        echo "${TEST_COVERAGE_INFO_DIR}"
        lcov -c -d ${OUTPUT_PATH}/build -d ${GTEST_OUTPUT_DIR} -o ${TEST_COVERAGE_INFO_DIR}/coverage.info --rc lcov_branch_coverage=1 --gcov-tool ${CUR_PATH}/llvm_gcov.sh
    else
        if [[ ${FAST_COV} == "on" ]]; then
            fastcov -d ${OUTPUT_PATH} --exceptional-branch-coverage --lcov -o ${TEST_COVERAGE_INFO_DIR}/coverage.info -j40
        else
            lcov -c -d ${OUTPUT_PATH}/build -d ${GTEST_OUTPUT_DIR} -o ${TEST_COVERAGE_INFO_DIR}/coverage.info --rc lcov_branch_coverage=1
        fi
    fi
    lcov --remove ${TEST_COVERAGE_INFO_DIR}/coverage.info '*/include/*' '${OUTPUT_TEST_DIR}' -o ${TEST_COVERAGE_INFO_DIR}/coverage.info --rc lcov_branch_coverage=1
    genhtml --branch-coverage -o ${TEST_COVERAGE_RESULT_DIR} ${TEST_COVERAGE_INFO_DIR}/*.info
    RET=$?
    echo "[LLT]lcov stoptime: `date +%F' '%H:%M:%S`"
    if [[ "${RET}" -ne "${SUCCESS}" ]]; then
        profiling_log 'error' 'lcov failed'
        exit 1
    fi
}

function gen_test_summary()
{
    echo '<?xml version="1.0" encoding="UTF-8"?>' > ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
    echo "<testsuites tests=\"${all_tsts_amount}\" pass=\"${passed_tsts_amount}\" failures=\"${failed_tsts_amount}\"
        skips=\"${skip_tsts_amount}\" mistrust=\"${mistrust_tsts_amount}\" unimplemented=\"${unimpl_tsts_amount}\"
        invalid=\"${invalid_tsts_amount}\" listed=\"${listed_tsts_amount}\" kdnn_unsupported=\"${kdnn_unsupported_amount}\"
        name=\"${TEST_TYPE}_test\">" >> ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
    echo '</testsuites>' >> ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
}

function prepare_test()
{
    [[ -d "${TEST_OUT_DIR}" ]] && rm -rf ${TEST_OUT_DIR}
    mkdir -p ${TEST_OUT_DIR}
    [[ -d "${TEST_LOG_DIR}" ]] && rm -rf ${TEST_LOG_DIR}
    mkdir -p ${TEST_LOG_DIR}

    [[ -d "${TEST_REPORT_DIR}" ]] && rm -rf ${TEST_REPORT_DIR}
    mkdir -p ${TEST_REPORT_DIR}
    if [[ ${GEN_REPORT} == "on" || ${ONLY_GEN_REPORT} == "on" ]] ;then
        [[ -d "${TEST_COVERAGE_RESULT_DIR}" ]] && rm -rf ${TEST_COVERAGE_RESULT_DIR}
        mkdir -p ${TEST_COVERAGE_RESULT_DIR}
        [[ -d "${TEST_COVERAGE_INFO_DIR}" ]] && rm -rf ${TEST_COVERAGE_INFO_DIR}
        mkdir -p ${TEST_COVERAGE_INFO_DIR}
    fi
}

function count_gtest_status()
{
    local file_name=$1

    local passed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "^\[  PASSED  \] [0-9]+" | grep -Eo "[0-9]+"`
    local failed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "^\[  FAILED  \] [0-9]+" | grep -Eo "[0-9]+"`

    if [ -z ${passed_tst_cnt} ];then
        passed_tst_cnt=0
    fi

    if [ -z ${failed_tst_cnt} ];then
        failed_tst_cnt=0
    fi

    local tst_cnt=`expr ${passed_tst_cnt} + ${failed_tst_cnt}`
    all_tsts_amount=`expr ${all_tsts_amount} + ${tst_cnt}`
    passed_tsts_amount=`expr ${passed_tsts_amount} + ${passed_tst_cnt}`
    failed_tsts_amount=`expr ${failed_tsts_amount} + ${failed_tst_cnt}`
}

function count_tsts_status()
{
    local file_name=$1

    local tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "tests:[0-9]*" | grep -Eo "[0-9]*"`
    local passed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "passed:[0-9]*" | grep -Eo "[0-9]*"`
    local skipped_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "skipped:[0-9]*" | grep -Eo "[0-9]*"`
    local mistrust_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "mistrusted:[0-9]*" | grep -Eo "[0-9]*"`
    local unimpl_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "unimplemented:[0-9]*" | grep -Eo "[0-9]*"`
    local invalid_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "invalid_arguments:[0-9]*" | grep -Eo "[0-9]*"`
    local failed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "failed:[0-9]*" | grep -Eo "[0-9]*"`
    local listed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "listed:[0-9]*" | grep -Eo "[0-9]*"`
    local kdnn_unsupported_cnt=$(grep "implementation:" ${TEST_LOG_DIR}/${file_name} | grep -v "kdnn" | grep -v "CPU" | wc -l)
    echo "kdnn unsupported: $kdnn_unsupported_cnt" >> ${TEST_LOG_DIR}/${file_name}

    if [ -z "$tst_cnt" ]; then
        echo "Program terminated abnormally" >> ${TEST_LOG_DIR}/${file_name}
        echo "Test fail" >> ${TEST_LOG_DIR}/${file_name}
        tst_cnt=0
        passed_tst_cnt=0
        skipped_tst_cnt=0
        mistrust_tst_cnt=0
        unimpl_tst_cnt=0
        unimpl_tst_cnt=0
        invalid_tst_cnt=0
        listed_tst_cnt=0
        failed_tst_cnt=1
        kdnn_unsupported_cnt=0
    fi

    if [ $failed_tst_cnt -ne 0 ]; then
        ut_success_flag=0
    fi

    all_tsts_amount=`expr ${all_tsts_amount} + ${tst_cnt}`
    passed_tsts_amount=`expr ${passed_tsts_amount} + ${passed_tst_cnt}`
    skip_tsts_amount=`expr ${skip_tsts_amount} + ${skipped_tst_cnt}`
    mistrust_tsts_amount=`expr ${mistrust_tsts_amount} + ${mistrust_tst_cnt}`
    unimpl_tsts_amount=`expr ${unimpl_tsts_amount} + ${unimpl_tst_cnt}`
    invalid_tsts_amount=`expr ${invalid_tsts_amount} + ${invalid_tst_cnt}`
    failed_tsts_amount=`expr ${failed_tsts_amount} + ${failed_tst_cnt}`
    listed_tsts_amount=`expr ${listed_tsts_amount} + ${listed_tst_cnt}`
    kdnn_unsupported_amount=`expr ${kdnn_unsupported_amount} + ${kdnn_unsupported_cnt}`

    if [ ${failed_tst_cnt} -ne 0 ]; then
        ut_success_flag=0
        echo "--------ERROR--------"
        echo "Detect ${failed_tst_cnt} test cases fail in ${file_name}"
        exit 1
    fi

    if [ ${kdnn_unsupported_cnt} -ne 0 ]; then
        ut_success_flag=0
        echo "--------ERROR--------"
        echo "Detect ${kdnn_unsupported_cnt} test cases not supported by kdnn in ${file_name}"
        exit 1
    fi
}

function run_ut_test()
{
    echo "${TEST_TYPE} Test Begin"
    local num_threads=$1

    # run gtest
    run_command "OMP_NUM_THREADS=${num_threads} ${GTEST_BIN} >> ${TEST_LOG_DIR}/gtest.log"
    count_gtest_status gtest.log

    # elementwise
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --eltwise --mode=C --batch=${BENCHDNN_INPUTS_DIR}/eltwise/test_eltwise_kdnn_ci >> ${TEST_LOG_DIR}/eltwise_log1.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --eltwise --mode=C --batch=${LLT_INPUTS_DIR}/eltwise/eltwise_func >> ${TEST_LOG_DIR}/eltwise_log2.log"

    # softmax
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --softmax --mode=C --batch=${LLT_INPUTS_DIR}/softmax/softmax_func >> ${TEST_LOG_DIR}/sofmax_log.log"

    # prelu
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --prelu --mode=C --batch=${LLT_INPUTS_DIR}/prelu/prelu_func >> ${TEST_LOG_DIR}/prelu_log.log"

    # matmul
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --matmul --mode=C --batch=${LLT_INPUTS_DIR}/matmul/matmul_func > ${TEST_LOG_DIR}/matmul_log.log"
    export KUDNN_FORCE_NEON=1
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --matmul --mode=C --batch=${LLT_INPUTS_DIR}/matmul/matmul_neon_f32 >> ${TEST_LOG_DIR}/matmul_log.log"
    unset KUDNN_FORCE_NEON
    #inner product
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --ip --mode=C --batch=${LLT_INPUTS_DIR}/ip/ip_func > ${TEST_LOG_DIR}/ip_log.log"

    # reduction
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --reduction --skip-impl=jit,acl,ref --p=2 --eps=0.5 --mode=C --alg=min,max,mean,sum,mul,norm_lp_max,norm_lp_sum,norm_lp_power_p_max,norm_lp_power_p_sum --batch=${BENCHDNN_INPUTS_DIR}/reduction/shapes_ci >> ${TEST_LOG_DIR}/reduction_log_f32.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --reduction --skip-impl=jit,acl,ref --p=2 --eps=0.5 --sdt=f16 --ddt=f16 --mode=C --alg=min,max,mean,sum,mul,norm_lp_max,norm_lp_sum,norm_lp_power_p_max,norm_lp_power_p_sum --batch=${BENCHDNN_INPUTS_DIR}/reduction/shapes_ci >> ${TEST_LOG_DIR}/reduction_log_f16.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --reduction --skip-impl=jit,acl,ref --p=2 --eps=0.5 --sdt=bf16 --ddt=bf16 --mode=C --alg=min,max,mean,sum,mul,norm_lp_max,norm_lp_sum,norm_lp_power_p_max,norm_lp_power_p_sum --batch=${BENCHDNN_INPUTS_DIR}/reduction/shapes_ci >> ${TEST_LOG_DIR}/reduction_log_bf16.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --reduction --skip-impl=jit,acl,ref --sdt=s8 --ddt=s8 --mode=C --alg=min,max,mean,sum,mul --batch=${BENCHDNN_INPUTS_DIR}/reduction/shapes_ci >> ${TEST_LOG_DIR}/reduction_log_s8.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --reduction  --mode=C --batch=${LLT_INPUTS_DIR}/reduction/reduction_func >> ${TEST_LOG_DIR}/reduction_log.log"

    # layer normalization
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --lnorm  --mode=C --batch=${LLT_INPUTS_DIR}/lnorm/lnorm_func >> ${TEST_LOG_DIR}/lnorm_log.log"

    # group normalization
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --gnorm  --mode=C --batch=${LLT_INPUTS_DIR}/gnorm/gnorm_func >> ${TEST_LOG_DIR}/gnorm_log.log"

    # binary operations
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --binary --mode=C --batch=${LLT_INPUTS_DIR}/binary/binary_func >> ${TEST_LOG_DIR}/binary_log.log"

    # sum operations
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --sum --skip-impl=jit,acl,ref,simple --mode=C --batch=${LLT_INPUTS_DIR}/sum/sum_func >> ${TEST_LOG_DIR}/sum_log.log"

    # convolution operations
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_fwd_flt_cases >> ${TEST_LOG_DIR}/conv_fwd_flt_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_fwd_int_cases >> ${TEST_LOG_DIR}/conv_fwd_int_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_bwd_flt_cases >> ${TEST_LOG_DIR}/conv_bwd_flt_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_bwd_mixed_cases >> ${TEST_LOG_DIR}/conv_bwd_mixed_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_post_op_cases >> ${TEST_LOG_DIR}/conv_post_op_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_dts_cases >> ${TEST_LOG_DIR}/conv_dts_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_jit_f32_cases >> ${TEST_LOG_DIR}/conv_jit_f32_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_jit_f16_cases >> ${TEST_LOG_DIR}/conv_jit_f16_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_jit_bf16_cases >> ${TEST_LOG_DIR}/conv_jit_bf16_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_sme_cases >> ${TEST_LOG_DIR}/conv_sme_cases_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/conv_dw_cases >> ${TEST_LOG_DIR}/conv_dw_cases_log.log"

    # deconvolution operations
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --deconv --mode=C --batch=${LLT_INPUTS_DIR}/deconvolution/deconvolution_func_fwd_i >> ${TEST_LOG_DIR}/deconv_fwd_i_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --deconv --mode=C --batch=${LLT_INPUTS_DIR}/deconvolution/deconvolution_func_fwd_d >> ${TEST_LOG_DIR}/deconv_fwd_d_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --deconv --mode=C --batch=${LLT_INPUTS_DIR}/deconvolution/deconvolution_func_fwd_b >> ${TEST_LOG_DIR}/deconv_fwd_b_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --deconv --mode=C --batch=${LLT_INPUTS_DIR}/deconvolution/deconvolution_func_bwd_d >> ${TEST_LOG_DIR}/deconv_bwd_d_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --deconv --mode=C --batch=${LLT_INPUTS_DIR}/deconvolution/deconvolution_func_bwd_w >> ${TEST_LOG_DIR}/deconv_bwd_w_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --deconv --mode=C --batch=${LLT_INPUTS_DIR}/deconvolution/deconvolution_func_bwd_wb >> ${TEST_LOG_DIR}/deconv_bwd_wb_log.log"

    # reorder
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --reorder --mode=C --batch=${LLT_INPUTS_DIR}/reorder/reorder_func >> ${TEST_LOG_DIR}/reorder_log.log"

    # shuffle
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --shuffle --mode=C --batch=${LLT_INPUTS_DIR}/shuffle/shuffle_func >> ${TEST_LOG_DIR}/shuffle_log.log"

    # pool
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --pool --mode=C --batch=${LLT_INPUTS_DIR}/pool/pool_func >> ${TEST_LOG_DIR}/pool_log.log"
    run_command "OMP_NUM_THREADS=1 ${BENCHDNN_BIN} -v5 --pool --mode=C --batch=${LLT_INPUTS_DIR}/pool/pool_func >> ${TEST_LOG_DIR}/pool_single_log.log"

    # resampling
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --resampling --mode=C --batch=${LLT_INPUTS_DIR}/resampling/resampling_fwd_func >> ${TEST_LOG_DIR}/resampling_fwd_log.log"
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --resampling --mode=C --batch=${LLT_INPUTS_DIR}/resampling/resampling_bwd_func >> ${TEST_LOG_DIR}/resampling_bwd_log.log"

    # concat
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --concat --mode=C --batch=${LLT_INPUTS_DIR}/concat/concat_func >> ${TEST_LOG_DIR}/concat_log.log"

    #lrn
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --lrn --mode=C --batch=${LLT_INPUTS_DIR}/lrn/lrn_func >> ${TEST_LOG_DIR}/lrn_log.log"

    #bnorm
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --bnorm --mode=C --batch=${LLT_INPUTS_DIR}/bnorm/bnorm_func >> ${TEST_LOG_DIR}/bnorm_log.log"

    # rnn
    run_command "OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} -v5 --rnn --mode=C --batch=${LLT_INPUTS_DIR}/rnn/rnn_func >> ${TEST_LOG_DIR}/rnn_log.log"

    check_return_values

    run_command "OMP_NUM_THREADS=40 ${BENCHDNN_BIN} -v5 --resampling --mode=C --batch=${LLT_INPUTS_DIR}/resampling/reasampling_dts_cases >> ${TEST_LOG_DIR}/reasampling_dts_cases_log.log"
    check_return_values

    count_tsts_status matmul_log.log
    count_tsts_status ip_log.log
    count_tsts_status eltwise_log1.log
    count_tsts_status eltwise_log2.log
    count_tsts_status sofmax_log.log
    count_tsts_status prelu_log.log
    count_tsts_status reduction_log_f32.log
    count_tsts_status reduction_log_f16.log
    count_tsts_status reduction_log_bf16.log
    count_tsts_status reduction_log_s8.log
    count_tsts_status reduction_log.log
    count_tsts_status lnorm_log.log
    count_tsts_status gnorm_log.log
    count_tsts_status binary_log.log
    count_tsts_status sum_log.log
    count_tsts_status conv_fwd_flt_cases_log.log
    count_tsts_status conv_fwd_int_cases_log.log
    count_tsts_status conv_bwd_flt_cases_log.log
    count_tsts_status conv_bwd_mixed_cases_log.log
    count_tsts_status conv_post_op_cases_log.log
    count_tsts_status conv_dts_cases_log.log
    count_tsts_status conv_jit_f32_cases_log.log
    count_tsts_status conv_jit_f16_cases_log.log
    count_tsts_status conv_jit_bf16_cases_log.log
    count_tsts_status conv_sme_cases_log.log
    count_tsts_status conv_dw_cases_log.log
    count_tsts_status deconv_fwd_i_log.log
    count_tsts_status deconv_fwd_d_log.log
    count_tsts_status deconv_fwd_b_log.log
    count_tsts_status deconv_bwd_d_log.log
    count_tsts_status deconv_bwd_w_log.log
    count_tsts_status deconv_bwd_wb_log.log
    count_tsts_status reorder_log.log
    count_tsts_status shuffle_log.log
    count_tsts_status pool_log.log
    count_tsts_status pool_single_log.log
    count_tsts_status reasampling_dts_cases_log.log
    count_tsts_status resampling_fwd_log.log
    count_tsts_status resampling_bwd_log.log
    count_tsts_status concat_log.log
    count_tsts_status lrn_log.log
    count_tsts_status bnorm_log.log
    count_tsts_status rnn_log.log

    if [ $ut_success_flag -ne 0 ]; then
        echo "${TEST_TYPE} Test Sucess"
    else
        echo "${TEST_TYPE} Test Fail"
    fi

    gen_test_summary
}

FUZZ_DETECT_KEYWORDS="failed\|Aborted\|fault\|dumped\|FAILED\|Exception\|AddressSanitizer\|crash\|Crash\|No[[:space:]]such[[:space:]]file"

function get_fuzz_summary()
{
    test_func=$1
    log_file="${TEST_LOG_DIR}/${test_func}_fuzz.log"
    if [[ -z "`grep ${FUZZ_DETECT_KEYWORDS} ${log_file} -a`" ]];then
        echo "${test_func} ... pass" >> ${FUZZ_SUMMARY_LOG}
    else
        echo "${test_func} ... failed" >> ${FUZZ_SUMMARY_LOG}
        cat "${TEST_LOG_DIR}/${test_func}_fuzz.log"
        fuzz_success_flag=0
    fi
}


function run_fuzz_test()
{
    echo "DT-Fuzz Test Begin"
    local num_threads=$1

    OMP_FLAGS="OMP_NUM_THREADS=${num_threads}"

    # parallelize
    run_command "${OMP_FLAGS} ${CODEGEN_FUZZ_TEST} >> ${TEST_LOG_DIR}/codegen_fuzz.log"
    run_command "${OMP_FLAGS} ${TENSOR_FUZZ_TEST} >> ${TEST_LOG_DIR}/tensor_fuzz.log"
    run_command "${OMP_FLAGS} ${ELTWISE_FUZZ_TEST} >> ${TEST_LOG_DIR}/eltwise_fuzz.log"

    run_command "${OMP_FLAGS} ${SOFTMAX_FUZZ_TEST} 10000 ForwardSoftmax >> ${TEST_LOG_DIR}/softmax_fwd_fuzz.log"
    run_command "${OMP_FLAGS} ${SOFTMAX_FUZZ_TEST} 10000 BackwardSoftmax >> ${TEST_LOG_DIR}/softmax_bwd_fuzz.log"

    run_command "OMP_NUM_THREADS=$((2 * num_threads)) ${BINARY_FUZZ_TEST} 10000 >> ${TEST_LOG_DIR}/binary_fuzz.log"
    run_command "OMP_NUM_THREADS=$((2 * num_threads)) ${BENCHDNN_BIN} -v0 --binary --mode=C --batch=${LLT_INPUTS_DIR}/binary/binary_func >> ${TEST_LOG_DIR}/binary_ut_fuzz.log"

    run_command "${OMP_FLAGS} ${PRELU_FUZZ_TEST} 10000 ForwardPreLU >> ${TEST_LOG_DIR}/prelu_fwd_fuzz.log"
    run_command "${OMP_FLAGS} ${PRELU_FUZZ_TEST} 10000 BackwardPreLU >> ${TEST_LOG_DIR}/prelu_bwd_fuzz.log"
    run_command "${OMP_FLAGS} ${MATMUL_FUZZ_TEST} 15000 MatmulFltCase >> ${TEST_LOG_DIR}/matmul_flt_fuzz.log"
    run_command "${OMP_FLAGS} ${MATMUL_FUZZ_TEST} 10000 MatmulIntCase >> ${TEST_LOG_DIR}/matmul_int_fuzz.log"
    export KUDNN_FORCE_NEON=1
    run_command "${OMP_FLAGS} ${MATMUL_FUZZ_TEST} 1500 MatmulNEONF32Case >> ${TEST_LOG_DIR}/matmul_neon_f32_fuzz.log"
    unset KUDNN_FORCE_NEON
    run_command "${OMP_FLAGS} ${REDUCTION_FUZZ_TEST} >> ${TEST_LOG_DIR}/reduction_fuzz.log"

    run_command "${OMP_FLAGS} ${INNER_PRODUCT_FUZZ_TEST} 10000 Inner_Product_Fwd >> ${TEST_LOG_DIR}/inner_product_fwd_fuzz.log"
    run_command "${OMP_FLAGS} ${INNER_PRODUCT_FUZZ_TEST} 10000 Inner_Product_Bwd >> ${TEST_LOG_DIR}/inner_product_bwd_data_fuzz.log"
    run_command "${OMP_FLAGS} ${INNER_PRODUCT_FUZZ_TEST} 10000 Inner_Product_Bwd_Weights >> ${TEST_LOG_DIR}/inner_product_bwd_weights_fuzz.log"

    run_command "${OMP_FLAGS} ${LNORM_FUZZ_TEST} >> ${TEST_LOG_DIR}/lnorm_fuzz.log"
    run_command "${OMP_FLAGS} ${CONVOLUTION_FUZZ_TEST} >> ${TEST_LOG_DIR}/convolution_fuzz.log"

    run_command "${OMP_FLAGS} ${DECONV_FUZZ_TEST} 10000 DeconvolutionFwd >> ${TEST_LOG_DIR}/deconvolution_fwd_fuzz.log"
    run_command "${OMP_FLAGS} ${DECONV_FUZZ_TEST} 10000 DeconvolutionBwdData >> ${TEST_LOG_DIR}/deconvolution_bwd_data_fuzz.log"
    run_command "${OMP_FLAGS} ${DECONV_FUZZ_TEST} 10000 DeconvolutionBwdWeights >> ${TEST_LOG_DIR}/deconvolution_bwd_weights_fuzz.log"

    run_command "${OMP_FLAGS} ${SUM_FUZZ_TEST} >> ${TEST_LOG_DIR}/sum_fuzz.log"
    run_command "${OMP_FLAGS} ${REORDER_FUZZ_TEST} >> ${TEST_LOG_DIR}/reorder_fuzz.log"
    run_command "${OMP_FLAGS} ${SHUFFLE_FUZZ_TEST} >> ${TEST_LOG_DIR}/shuffle_fuzz.log"
    run_command "${OMP_FLAGS} ${CONCAT_FUZZ_TEST} >> ${TEST_LOG_DIR}/concat_fuzz.log"
    run_command "${OMP_FLAGS} ${RESAMPLING_FUZZ_TEST} >> ${TEST_LOG_DIR}/resampling_fuzz.log"
    run_command "${OMP_FLAGS} ${POOL_FUZZ_TEST} >> ${TEST_LOG_DIR}/pool_fuzz.log"

    run_command "${OMP_FLAGS} ${BNORM_FUZZ_TEST} 4000 Nomalization_Layer_Fwd  >> ${TEST_LOG_DIR}/bnorm_fwd_fuzz.log"
    run_command "${OMP_FLAGS} ${BNORM_FUZZ_TEST} 4000 Nomalization_Layer_Fwd_Cross_Layout >> ${TEST_LOG_DIR}/bnorm_fwd_cross_fuzz.log"
    run_command "${OMP_FLAGS} ${BNORM_FUZZ_TEST} 2000 Nomalization_Layer_Bwd_3D  >> ${TEST_LOG_DIR}/bnorm_bwd_3d_fuzz.log"
    run_command "${OMP_FLAGS} ${BNORM_FUZZ_TEST} 2000 Nomalization_Layer_Bwd_4D  >> ${TEST_LOG_DIR}/bnorm_bwd_4d_fuzz.log"
    run_command "${OMP_FLAGS} ${BNORM_FUZZ_TEST} 2000 Nomalization_Layer_Bwd_5D  >> ${TEST_LOG_DIR}/bnorm_bwd_5d_fuzz.log"

    run_command "${OMP_FLAGS} ${LRN_FUZZ_TEST} >> ${TEST_LOG_DIR}/lrn_fuzz.log"
    run_command "${OMP_FLAGS} ${GNORM_FUZZ_TEST} >> ${TEST_LOG_DIR}/gnorm_fuzz.log"
    run_command "${OMP_FLAGS} ${RNN_FUZZ_TEST} >> ${TEST_LOG_DIR}/rnn_fuzz.log"
    run_command "${OMP_FLAGS} ${POST_OPS_FUZZ_TEST} >> ${TEST_LOG_DIR}/post_ops_fuzz.log"

    check_return_values

    run_command "sed -i 's/invalid_arguments:0 failed:0 listed:0/invalid_arguments:0 listed:0/g' ${TEST_LOG_DIR}/binary_ut_fuzz.log"

    check_return_values

    get_fuzz_summary codegen
    get_fuzz_summary tensor
    get_fuzz_summary eltwise

    get_fuzz_summary softmax_fwd
    get_fuzz_summary softmax_bwd
    get_fuzz_summary binary
    get_fuzz_summary binary_ut
    get_fuzz_summary prelu_fwd
    get_fuzz_summary prelu_bwd
    get_fuzz_summary matmul_flt
    get_fuzz_summary matmul_int
    get_fuzz_summary matmul_neon_f32
    get_fuzz_summary reduction
    get_fuzz_summary inner_product_fwd
    get_fuzz_summary inner_product_bwd_data
    get_fuzz_summary inner_product_bwd_weights

    get_fuzz_summary lnorm
    get_fuzz_summary convolution
    get_fuzz_summary deconvolution_fwd
    get_fuzz_summary deconvolution_bwd_data
    get_fuzz_summary deconvolution_bwd_weights
    get_fuzz_summary sum
    get_fuzz_summary reorder
    get_fuzz_summary shuffle
    get_fuzz_summary concat
    get_fuzz_summary resampling
    get_fuzz_summary bnorm_fwd
    get_fuzz_summary bnorm_fwd_cross
    get_fuzz_summary bnorm_bwd_3d
    get_fuzz_summary bnorm_bwd_4d
    get_fuzz_summary bnorm_bwd_5d
    get_fuzz_summary pool
    get_fuzz_summary lrn
    get_fuzz_summary gnorm
    get_fuzz_summary rnn
    get_fuzz_summary post_ops

    if [ $fuzz_success_flag -ne 0 ]; then
        echo "DT-Fuzz Test Sucess"
    else
        echo "DT-Fuzz Test Fail"
    fi
}

prepare_test

if [[ ${ONLY_GEN_REPORT} != "on" ]]; then
    if [[ "${TEST_TYPE}" == "fuzz" ]];then
        run_fuzz_test 4
    else
        run_ut_test 4
    fi
fi

if [[ ${GEN_REPORT} == "on" || ${ONLY_GEN_REPORT} == "on" ]]; then
    gen_coverage_report
fi