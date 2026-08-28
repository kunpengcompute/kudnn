#!/bin/bash

# print help info
function print_help_info()
{
    echo "+-------------+-----------------------------------------+-------------------------------------------------------------+"
    echo "| options     | value                                   | description                                                 |"
    echo "+-------------+-----------------------------------------+-------------------------------------------------------------+"
    echo "| --server=   | 920B(default)|920F|920F_cache|920F_flat | server type                                                 |"
    echo "| --numa=     | 0(default)|...                          | numa id, must be a natural number                           |"
    echo "| --threads=  | all(default)|single|multi               | run with single thread or multi threads on one numa or both |"
    echo "| --mode=     | all(default)|P|C                        | run '--mode=P' or '--mode=C' or both in benchdnn            |"
    echo "| --operator= | all(default)|...                        | operator for testing                                        |"
    echo "| --help(-h)  |                                         | print user guide                                            |"
    echo "+-------------+-----------------------------------------+-------------------------------------------------------------+"
    return
}

function print_hint_and_exit()
{
    echo "You can use '--help' or '-h' to get manuals."
    print_dividing_line
    exit
}

function print_dividing_line()
{
    echo "-----------------------------------------------------------------------------------------------------------------------"
}

function make_dir()
{
    new_path=$1
    [[ -d "${new_path}" ]] && rm -rf ${new_path}
    mkdir -p ${new_path}
}

# check input parameters
function check_input()
{
    print_dividing_line
    # check if '--server' is valid
    if [[ $server != "920B" && $server != "920F" &&  $server != "920F_flat" && $server != "920F_cache" ]]
    then
        echo "Error parameter: '--server=$server' is invaild."
        print_hint_and_exit
    fi

    # check if '--numa' is a natural number
    case "$numa" in
        ''|*[!0-9]*)
            echo "Error parameter: '--numa=$numa' is not a natural number."
            print_hint_and_exit
            ;;
        0*[0-9]*)
            echo "Error parameter: '--numa=$numa' is not a natural number."
            print_hint_and_exit
            ;;
        *)
            ;;
    esac
    if [[ $numa -lt 0 ]]
    then
        echo "Error parameter: '--numa=$numa' is not a natural number."
        print_hint_and_exit
    fi

    # check if '--numa' is valid
    if [[ $server == "920B" || $server == "920F" ]] && [[ $numa -gt 3 ]]
    then
        # 920B and 920F have 4 numa.
        echo "Error parameter: '--numa=$numa' is too big for '--server=$server'."
        print_hint_and_exit
    elif [[ $server == "920F_flat" || $server == "920F_cache" ]] && [[ $numa -gt 15 ]]
    then
        # 920B and 920F have 16 numa.
        echo "Error parameter: '--numa=$numa' is too big."
        print_hint_and_exit
    fi

    # check if '--threads' is valid
    if [[ $threads != "all" && $threads != "single" && $threads != "multi" ]]
    then
        echo "Error parameter: '--threads=$threads' is invalid."
        print_hint_and_exit
    fi

    # check if '--mode' is valid
    if [[ $mode != "all" && $mode != "P" && $mode != "C" ]]
    then
        echo "Error parameter: '--mode=$mode' is invalid."
        print_hint_and_exit
    fi

    # check if '--operator' is valid.
    if [[ $operator != "all" ]]
    then
        for op in ${SUPPORTED_OPERATORS[*]}
        do
            if [[ $operator == $op ]]
            then
                return
            fi
        done
    else
        return
    fi

    echo "Error parameter: '--operator=$operator' is invalid."
    print_hint_and_exit

    return 
}

# set and print server infos
function set_and_print_server_infos()
{
    if [[ $server == "920B" ]]
    then
        cpu_per_numa=40
    elif [[ $server == "920F" ]]
    then
        cpu_per_numa=144
    elif [[ $server == "920F_flat" || $server == "920F_cache" ]]
    then
        cpu_per_numa=38
    fi

    cpu_s=$[${numa}*${cpu_per_numa}]
    cpu_e=$[${cpu_s}+${cpu_per_numa}-1]

    mem_id_gaps=16
    ddr_id=$numa
    hbm_id=$[${ddr_id}+${mem_id_gaps}]

    echo "       server : ${server}"
    echo "      numa_id : ${numa}"
    echo " cpu_per_numa : ${cpu_per_numa}"
    if [[ $threads == "all" || $threads == "single" ]]
    then
        echo "single thread : cpu ${cpu_s}"
    fi
    if [[ $threads == "all" || $threads == "multi" ]]
    then
        echo " multi thread : cpu ${cpu_s}-${cpu_e}"
    fi
    echo "       ddr_id : ${ddr_id}"
    if [[ $server == "920F_flat" ]]
    then
        echo "       hbm_id : ${hbm_id}"
    fi
    if [[ $mode == "all" ]]
    then
        echo "         mode : mode=P and mode=C"
    elif [[ $mode == "P" ]]
    then
        echo "         mode : mode=P"
    elif [[ $mode == "C" ]]
    then
        echo "         mode : mode=C"
    fi
    print_dividing_line
}

# set and print paths
function set_and_print_paths()
{
    SCRIPTS_PATH=$(cd `dirname $0`; pwd)
    ROOT_PATH=${SCRIPTS_PATH%test*}
    OUT_PATH=${ROOT_PATH}out
    BENCHDNN_PATH=${OUT_PATH}/test/dnn/oneDNN-3.4/build/tests/benchdnn
    TEST_LOG_PATH=${OUT_PATH}/benchdnn_out

    make_dir ${TEST_LOG_PATH}

    echo "    root path : $ROOT_PATH"
    echo "     out path : $OUT_PATH"
    echo " scripts path : $SCRIPTS_PATH"
    echo "benchdnn path : $BENCHDNN_PATH"
    echo "test log path : $TEST_LOG_PATH"
    print_dividing_line
}

function set_and_print_environment_infos()
{
    # LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${OUT_PATH}/dependencies/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${OUT_PATH}/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${OUT_PATH}/test/dnn/oneDNN-3.4/build/src/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${OUT_PATH}/test/dnn/acl/build/:$LD_LIBRARY_PATH
    local ld_library_path_array=(${LD_LIBRARY_PATH//:/ })
    # MEM and ulimit
    if [[ $server == "920F_flat" ]]
    then
        export MEM_TYPE="HBM"
        export MEMKIND_HBW_NODES="16-31"
    elif [[ $server == "920F_cache" ]]
    then
        export MEMKIND_HBW_NODES="0-15"
        ulimit -c 0
    fi

    echo "  LD_LIBRARY_PATH = ${ld_library_path_array[0]}"
    echo "                    ${ld_library_path_array[1]}"
    echo "                    ${ld_library_path_array[2]}"
    echo "                    ${ld_library_path_array[3]}"
    if [[ $server == "920F_flat" ]]
    then
        echo "           MEM_TYPE = $MEM_TYPE"
        echo "  MEMKIND_HBW_NODES = $MEMKIND_HBW_NODES"
    elif [[ $server == "920F_cache" ]]
    then
        echo "MEMKIND_HBW_NODES = $MEMKIND_HBW_NODES"
        print_dividing_line
        echo "You are using a 920F_cache server, 'ulimit -c 0' is set."
    fi
    print_dividing_line
}

function set_and_print_tested_operator()
{
    if [[ $operator == "all" ]]
    then
        operators_list=${SUPPORTED_OPERATORS}
    else
        operators_list=("$operator")
    fi
    echo "Testing operator : $operators_list"
    print_dividing_line
}

function prepare_to_test()
{
    local thread_mode=$1
    # create log files
    thread_mode_path=${tested_op_path}/${thread_mode}
    mode_p_path=${thread_mode_path}/mode_p.log
    mode_c_path=${thread_mode_path}/mode_c.log
    make_dir $thread_mode_path
    touch $mode_p_path
    touch $mode_c_path
    # combine cmd prefix
    command_prefix="taskset -c ${cpu_s}"
    if [[ $thread_mode == "multi" ]]
    then
        command_prefix="${command_prefix}-${cpu_e}"
    fi
    command_prefix="${command_prefix} numactl -m ${ddr_id}"
    if [[ $server == "920F_flat" ]]
    then
        command_prefix="${command_prefix},${hbm_id}"
    fi
}

# operator 'sum' testing function
function run_sum_test()
{
    local thread_mode=$1
    if [[ $thread_mode != "single" && $thread_mode != "multi" ]]
    then
        return
    fi
    prepare_to_test $thread_mode

    # set cmd options
    local tag="abx"
    local type_list=("f32 f16 bf16 s32 s8 u8")
    local f32_scales_list=("1.0 3.8 -2.6")
    local f16_scales_list=("1.0 3.8 -2.6")
    local bf16_scales_list=("1.0 3.8 -2.6")
    local s32_scales_list=("1 3 -2")
    local s8_scales_list=("1 3 -2")
    local u8_scales_list=("1 3")
    local prob_list=("17x3206 512x768 8x256x256 6x2x128x1024 10x100x100x100 9x13x175x211x8")
    if [[ $thread_mode == "multi" ]]
    then
        prob_list=("25x25 300x30 5x5x15 4x8x16x32 3x1x7x386 3x2x128x13x4")
    fi

    for ddt in ${type_list[*]}
    do
    local sdt_list=("${ddt}:${ddt} ${ddt}:${ddt}:${ddt} ${ddt}:${ddt}:${ddt}:${ddt}")
    for sdt in ${sdt_list[*]}
    do
    local scales_list=`eval echo '$'{${ddt}_scales_list}`
    for scales in ${scales_list[*]}
    do
    for prob_desc in ${prob_list[*]}
    do

    # --mode=P test
    if [[ $mode == "all" || $mode == "P" ]]
    then
        cmd="$command_prefix ${BENCHDNN_PATH}/benchdnn --${tested_op} --mode=P --sdt=$sdt --ddt=$ddt --stag=$tag --dtag=$tag --scales=$scales $prob_desc" 
        echo $cmd >> $mode_p_path
        $cmd >> $mode_p_path
    fi
    # --mode=C test
    if [[ $mode == "all" || $mode == "C" ]]
    then
        cmd="$command_prefix ${BENCHDNN_PATH}/benchdnn --${tested_op} --mode=C --sdt=$sdt --ddt=$ddt --stag=$tag --dtag=$tag --scales=$scales $prob_desc"
        echo $cmd >> $mode_c_path
        $cmd >> $mode_c_path
    fi

    done
    done
    done
    done

    python3 $SCRIPTS_PATH/benchdnn_test_data_collector.py $thread_mode_path
}

# default input parameters
server="920B"
numa=0
threads="all"
mode="all"
operator="all"

# parse input parameters
for i in "$@"; do
    case $i in
        --server=*)
        server="${i#*=}"
        ;;
        --numa=*)
        numa="${i#*=}"
        ;;
        --threads=*)
        threads="${i#*=}"
        ;;
        --mode=*)
        mode="${i#*=}"
        ;;
        --operator=*)
        operator="${i#*=}"
        ;;
        --help|-h)
        print_help_info
        exit
        ;;
        *)
        echo "Error parameter: Some parameters may not supported."
        print_hint_and_exit
        ;;
    esac
done

SUPPORTED_OPERATORS=("sum")

check_input

set_and_print_server_infos
set_and_print_paths
set_and_print_environment_infos
set_and_print_tested_operator

# if you do not want the scripts pause here, comment out this line.
read -p "Please check if the above information is correct. Pressing 'enter' to continue testing..."
print_dividing_line

for tested_op in ${operators_list[*]}
do
    tested_op_path=${TEST_LOG_PATH}/${tested_op}
    make_dir $tested_op_path
    # single thread test
    if [[ $threads == "all" || $threads == "single" ]]
    then
        export OMP_NUM_THREADS=1
        export OMP_PROC_BIND="close"
        run_${tested_op}_test "single"
        echo "Testing of operator '$tested_op' on single thread finished."
    fi
    if [[ $threads == "all" || $threads == "multi" ]]
    then
        # multi threads test
        export OMP_NUM_THREADS=${cpu_per_numa}
        export OMP_PROC_BIND="close"
        run_${tested_op}_test "multi"
        echo "Testing of operator '$tested_op' on multi thread finished."
    fi
done

echo "All tests finished, the test data has been stored in the directory '$TEST_LOG_PATH'."
print_dividing_line
