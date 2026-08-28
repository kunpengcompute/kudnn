add_subdirectory(matmul_jit)
add_subdirectory(reorder)

set(ALL_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16" "int8_t" "uint8_t" "int32_t")
set(FLOAT_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16")
set(COMPUTE_PRECISIONS "float")
set(SHORT_NAME_float f32)
set(SHORT_NAME___fp16 f16)
set(SHORT_NAME___bf16 bf16)
set(SHORT_NAME_int8_t s8)
set(SHORT_NAME_uint8_t u8)
set(SHORT_NAME_int32_t s32)

# Binary
set(BINARY_FUNC_TYPES "Add" "Div" "Eq" "Ge" "Gt" "Le" "Lt" "Max" "Min" "Mul" "Ne" "Sub")
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/binary BINARY_INSTANCES)
set(BINARY_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float" OR IN_OUT_PREC STREQUAL "__fp16"))
            set(USE_NEON 1)
        endif()
        foreach(BINARY_FUNC ${BINARY_FUNC_TYPES})
            if(IN_OUT_PREC STREQUAL "__fp16")
                add_library(
                    binary_instances_${BINARY_FUNC}_f16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                    OBJECT ${BINARY_INSTANCES})
                target_compile_options(
                    binary_instances_${BINARY_FUNC}_f16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                    PUBLIC -DFUNC_KIND=${BINARY_FUNC} -DCOMP_TYPE=__fp16 -DSRC0_TYPE=${IN_OUT_PREC}
                    -DSRC1_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
                list(APPEND BINARY_KERNELS_LIST
                    $<TARGET_OBJECTS:binary_instances_${BINARY_FUNC}_f16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
            else()
                add_library(
                    binary_instances_${BINARY_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                    OBJECT ${BINARY_INSTANCES})
                target_compile_options(
                    binary_instances_${BINARY_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                    PUBLIC -DFUNC_KIND=${BINARY_FUNC} -DCOMP_TYPE=${COMP_PREC} -DSRC0_TYPE=${IN_OUT_PREC}
                    -DSRC1_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
                list(APPEND BINARY_KERNELS_LIST
                    $<TARGET_OBJECTS:binary_instances_${BINARY_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
            endif()
        endforeach()
    endforeach()
endforeach()
set(BINARY_KERNELS_LIST "${BINARY_KERNELS_LIST}" CACHE INTERNAL "")

#BNorm
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/bnorm BNORM_INSTANCES)
set(BNORM_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float"))
            set(USE_NEON 1)
        endif()
        add_library(
            bnorm_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${BNORM_INSTANCES})
        target_compile_options(
            bnorm_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
        list(APPEND BNORM_KERNELS_LIST
            $<TARGET_OBJECTS:bnorm_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(BNORM_KERNELS_LIST "${BNORM_KERNELS_LIST}" CACHE INTERNAL "")

# Convolution
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/convolution/ CONVOLUTION_PIPELINE)

add_library(convolution_kernels_pipeline OBJECT ${CONVOLUTION_PIPELINE})

list(APPEND CONV_KERNELS_LIST
    $<TARGET_OBJECTS:convolution_kernels_pipeline>)

set(CONV_KERNELS_LIST "${CONV_KERNELS_LIST}" CACHE INTERNAL "")

# GNorm
set(GNORM_FORWARD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/gnorm/kudnn_gnorm_instances_fwd.cpp")
set(GNORM_BACKWARD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/gnorm/kudnn_gnorm_instances_bwd.cpp")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float"))
            set(USE_NEON 1)
        endif()
        add_library(
            gnorm_forward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${GNORM_FORWARD_INSTANCES})
        target_compile_options(
            gnorm_forward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DSCALE_SHIFT_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
        list(APPEND GNORM_KERNELS_LIST
            $<TARGET_OBJECTS:gnorm_forward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        add_library(
            gnorm_backward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${GNORM_BACKWARD_INSTANCES})
        target_compile_options(
            gnorm_backward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DSCALE_SHIFT_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND GNORM_KERNELS_LIST
            $<TARGET_OBJECTS:gnorm_backward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(GNORM_KERNELS_LIST "${GNORM_KERNELS_LIST}" CACHE INTERNAL "")

# LNorm
set(LNORM_FWD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_instances_fwd_cross.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_instances_fwd_last.cpp")

set(LNORM_BWD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_instances_bwd_cross.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_instances_bwd_last.cpp")

foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float"))
            set(USE_NEON 1)
        endif()
        add_library(
            lnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${LNORM_FWD_INSTANCES})
        target_compile_options(
            lnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DSCALE_SHIFT_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
        list(APPEND LNORM_KERNELS_LIST
            $<TARGET_OBJECTS:lnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()

#set(LNORM_KERNELS_LIST "")
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        add_library(
            lnorm_bwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${LNORM_BWD_INSTANCES})
        target_compile_options(
            lnorm_bwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DSCALE_SHIFT_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND LNORM_KERNELS_LIST
            $<TARGET_OBJECTS:lnorm_bwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(LNORM_KERNELS_LIST "${LNORM_KERNELS_LIST}" CACHE INTERNAL "")

# LRNorm
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/lrn LRN_INSTANCES)
set(LRN_KERNELS_LIST "")
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        add_library(
            lrn_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${LRN_INSTANCES})
        target_compile_options(
            lrn_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND LRN_KERNELS_LIST
            $<TARGET_OBJECTS:lrn_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(LRN_KERNELS_LIST "${LRN_KERNELS_LIST}" CACHE INTERNAL "")

# Pooling fwd
set(POOLING_INSTANCES1 "${CMAKE_CURRENT_LIST_DIR}/pooling/kudnn_pooling_inst_abx_fwd.cpp"
                        "${CMAKE_CURRENT_LIST_DIR}/pooling/kudnn_pooling_inst_axb_fwd.cpp")
set(SHORT_NAME_max PoolingFunction::MAX)
set(SHORT_NAME_avg_np PoolingFunction::AVG_EXCLUDE_PADDING)
set(SHORT_NAME_avg_p PoolingFunction::AVG_INCLUDE_PADDING)
set(SHORT_NAME_float f32)
set(WS_PRECISIONS "uint8_t" "int32_t")
set(WS_PRECISIONS1 "uint8_t")
set(POOLING_FUNC_TYPES1 "max")
set(POOLING_FUNC_TYPES2 "max" "avg_np" "avg_p")
set(USING_WS "true")
set(NOT_USING_WS  "false")
set(POOLING_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float" OR IN_OUT_PREC STREQUAL "__fp16"))
            set(USE_NEON 1)
        endif()
        foreach(WS_PREC ${WS_PRECISIONS})
            foreach(USE_WS ${USING_WS})
                foreach(POOLING_FUNC ${POOLING_FUNC_TYPES1})
                    add_library(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}
                        OBJECT ${POOLING_INSTANCES1})
                    target_compile_options(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}
                        PUBLIC -DPOOL_TYPE=${SHORT_NAME_${POOLING_FUNC}} -DCOMP_TYPE=${COMP_PREC} -DSRC_TYPE=${IN_OUT_PREC}
                        -DDST_TYPE=${IN_OUT_PREC} -DWS_TYPE=${WS_PREC} -DUSE_WS=${USE_WS} -DUSE_NEON=${USE_NEON})
                    list(APPEND POOLING_KERNELS_LIST
                        $<TARGET_OBJECTS:pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}>)
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float" OR IN_OUT_PREC STREQUAL "__fp16"))
            set(USE_NEON 1)
        endif()
        foreach(WS_PREC ${WS_PRECISIONS1})
            foreach(USE_WS ${NOT_USING_WS})
                foreach(POOLING_FUNC ${POOLING_FUNC_TYPES2})
                    add_library(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}
                        OBJECT ${POOLING_INSTANCES1})
                    target_compile_options(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}
                        PUBLIC -DPOOL_TYPE=${SHORT_NAME_${POOLING_FUNC}} -DCOMP_TYPE=${COMP_PREC} -DSRC_TYPE=${IN_OUT_PREC}
                        -DDST_TYPE=${IN_OUT_PREC} -DWS_TYPE=${WS_PREC} -DUSE_WS=${USE_WS} -DUSE_NEON=${USE_NEON})
                    list(APPEND POOLING_KERNELS_LIST
                        $<TARGET_OBJECTS:pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}>)
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endforeach()
set(POOLING_KERNELS_LIST "${POOLING_KERNELS_LIST}" CACHE INTERNAL "")

# Pooling bwd
set(POOLING_INSTANCES2 "${CMAKE_CURRENT_LIST_DIR}/pooling/kudnn_pooling_inst_abx_bwd.cpp"
                        "${CMAKE_CURRENT_LIST_DIR}/pooling/kudnn_pooling_inst_axb_bwd.cpp")
set(SHORT_NAME_max_bwd PoolingFunction::MAX)
set(SHORT_NAME_avg_np_bwd PoolingFunction::AVG_EXCLUDE_PADDING)
set(SHORT_NAME_avg_p_bwd PoolingFunction::AVG_INCLUDE_PADDING)
set(SHORT_NAME_float f32)
set(WS_PRECISIONS "uint8_t" "int32_t")
set(WS_PRECISIONS1 "int32_t")
set(POOLING_FUNC_TYPES1 "max_bwd")
set(POOLING_FUNC_TYPES2 "avg_np_bwd" "avg_p_bwd")
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        foreach(WS_PREC ${WS_PRECISIONS})
                foreach(POOLING_FUNC ${POOLING_FUNC_TYPES1})
                    add_library(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}
                        OBJECT ${POOLING_INSTANCES2})
                    target_compile_options(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}
                        PUBLIC -DPOOL_TYPE=${SHORT_NAME_${POOLING_FUNC}} -DCOMP_TYPE=${COMP_PREC} -DSRC_TYPE=${IN_OUT_PREC}
                        -DDST_TYPE=${IN_OUT_PREC} -DWS_TYPE=${WS_PREC})
                    list(APPEND POOLING_KERNELS_LIST
                        $<TARGET_OBJECTS:pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}>)
                endforeach()
            endforeach()
    endforeach()
endforeach()

foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        foreach(WS_PREC ${WS_PRECISIONS1})
                foreach(POOLING_FUNC ${POOLING_FUNC_TYPES2})
                    add_library(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}
                        OBJECT ${POOLING_INSTANCES2})
                    target_compile_options(
                        pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}
                        PUBLIC -DPOOL_TYPE=${SHORT_NAME_${POOLING_FUNC}} -DCOMP_TYPE=${COMP_PREC} -DSRC_TYPE=${IN_OUT_PREC}
                        -DDST_TYPE=${IN_OUT_PREC} -DWS_TYPE=${WS_PREC})
                    list(APPEND POOLING_KERNELS_LIST
                        $<TARGET_OBJECTS:pooling_instances_${POOLING_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${WS_PREC}}_${USE_WS}>)
                endforeach()
            endforeach()
    endforeach()
endforeach()
set(POOLING_KERNELS_LIST "${POOLING_KERNELS_LIST}" CACHE INTERNAL "")

# PReLU
set(PRELU_INSTANCES1 "${CMAKE_CURRENT_LIST_DIR}/prelu/kudnn_prelu_instances_bwd_whole_tensor.cpp"
                     "${CMAKE_CURRENT_LIST_DIR}/prelu/kudnn_prelu_instances_bwd_channel_shared.cpp")
set(PRELU_INSTANCES2 "${CMAKE_CURRENT_LIST_DIR}/prelu/kudnn_prelu_instances_fwd_non_shared_axes.cpp"
                     "${CMAKE_CURRENT_LIST_DIR}/prelu/kudnn_prelu_instances_fwd_shared_axes.cpp"
                     "${CMAKE_CURRENT_LIST_DIR}/prelu/kudnn_prelu_instances_fwd_generic.cpp")
set(PRELU_INSTANCES3 "${CMAKE_CURRENT_LIST_DIR}/prelu/kudnn_prelu_instances_bwd_shared_axes.cpp")
set(PRELU_LD_KINDS Load Dup)
set(PRELU_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float"))
            set(USE_NEON 1)
        endif()
        add_library(
            prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${PRELU_INSTANCES1})
        target_compile_options(
            prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DSRC_TYPE=${IN_OUT_PREC} -DWEIGHTS_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
        list(APPEND PRELU_KERNELS_LIST
            $<TARGET_OBJECTS:prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        foreach(PRELU_LD_KIND ${PRELU_LD_KINDS})
            if(IN_OUT_PREC STREQUAL "__fp16")
                add_library(
                    prelu_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}
                    OBJECT ${PRELU_INSTANCES2})
                target_compile_options(
                    prelu_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}
                    PUBLIC -DCOMP_TYPE=__fp16 -DSRC_TYPE=${IN_OUT_PREC} -DWEIGHTS_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DLD_KIND=${PRELU_LD_KIND} -DUSE_NEON=${USE_NEON})
                list(APPEND PRELU_KERNELS_LIST
                    $<TARGET_OBJECTS:prelu_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}>)
            else()
                add_library(
                    prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}
                    OBJECT ${PRELU_INSTANCES2})
                target_compile_options(
                    prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}
                    PUBLIC -DCOMP_TYPE=${COMP_PREC} -DSRC_TYPE=${IN_OUT_PREC} -DWEIGHTS_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DLD_KIND=${PRELU_LD_KIND} -DUSE_NEON=${USE_NEON})
                list(APPEND PRELU_KERNELS_LIST
                    $<TARGET_OBJECTS:prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}>)
            endif()
            if(PRELU_LD_KIND STREQUAL "Load")
                set(PRELU_ST_KIND "Store")
            else()
                set(PRELU_ST_KIND "Update")
            endif()
            add_library(
                prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}_${PRELU_ST_KIND}
                OBJECT ${PRELU_INSTANCES3})
            target_compile_options(
                prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}_${PRELU_ST_KIND}
                PUBLIC -DCOMP_TYPE=${COMP_PREC} -DSRC_TYPE=${IN_OUT_PREC} -DWEIGHTS_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DLD_KIND=${PRELU_LD_KIND} -DST_KIND=${PRELU_ST_KIND} -DUSE_NEON=${USE_NEON})
            list(APPEND PRELU_KERNELS_LIST
                $<TARGET_OBJECTS:prelu_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${PRELU_LD_KIND}_${PRELU_ST_KIND}>)
        endforeach()
    endforeach()
endforeach()
set(PRELU_KERNELS_LIST "${PRELU_KERNELS_LIST}" CACHE INTERNAL "")

# Reduction
set(REDUCTION_FUNC_TYPES "Min" "Max" "Mean" "Sum" "Mul")
set(REDUCTION_FUNC_NORMLP_TYPES "NormLp")
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/reduction REDUCTION_INSTANCES)
set(REDUCTION_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        foreach(REDUCTION_FUNC ${REDUCTION_FUNC_TYPES})
            set(USE_NEON 0)
            if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float"))
                set(USE_NEON 1)
            endif()
            add_library(
                reduction_instances_${REDUCTION_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                OBJECT ${REDUCTION_INSTANCES})
            target_compile_options(
                reduction_instances_${REDUCTION_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                PUBLIC -DFUNC_KIND=${REDUCTION_FUNC} -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
            list(APPEND REDUCTION_KERNELS_LIST
                $<TARGET_OBJECTS:reduction_instances_${REDUCTION_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        endforeach()
    endforeach()
endforeach()
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        foreach(REDUCTION_FUNC ${REDUCTION_FUNC_NORMLP_TYPES})
            set(USE_NEON 0)
            if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float"))
                set(USE_NEON 1)
            endif()
            add_library(
                reduction_instances_${REDUCTION_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                OBJECT ${REDUCTION_INSTANCES})
            target_compile_options(
                reduction_instances_${REDUCTION_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                PUBLIC -DFUNC_KIND=${REDUCTION_FUNC} -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
            list(APPEND REDUCTION_KERNELS_LIST
                $<TARGET_OBJECTS:reduction_instances_${REDUCTION_FUNC}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        endforeach()
    endforeach()
endforeach()
set(REDUCTION_KERNELS_LIST "${REDUCTION_KERNELS_LIST}" CACHE INTERNAL "")

# Shuffle
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/shuffle SHUFFLE_INSTANCES)
set(SHUFFLE_KERNELS_LIST "")
foreach(IN_PREC ${ALL_SUPPORTED_PRECISIONS})
    add_library(
        shuffle_instances_${SHORT_NAME_${IN_PREC}}
        OBJECT ${SHUFFLE_INSTANCES})
    target_compile_options(
        shuffle_instances_${SHORT_NAME_${IN_PREC}}
        PUBLIC -DIN_TYPE=${IN_PREC})
    list(APPEND SHUFFLE_KERNELS_LIST
        $<TARGET_OBJECTS:shuffle_instances_${SHORT_NAME_${IN_PREC}}>)
endforeach()
set(SHUFFLE_KERNELS_LIST "${SHUFFLE_KERNELS_LIST}" CACHE INTERNAL "")

# Softmax
set(SOFTMAX_FORWARD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_instances_fwd_cross.cpp"
                            "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_instances_fwd_last.cpp")
set(SOFTMAX_BACKWARD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_instances_bwd_cross.cpp"
                             "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_instances_bwd_last.cpp")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        if(IN_OUT_PREC STREQUAL "__fp16")
            add_library(
                softmax_fwd_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                OBJECT ${SOFTMAX_FORWARD_INSTANCES})
            target_compile_options(
                softmax_fwd_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                PUBLIC -DCOMP_TYPE=__fp16 -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
            list(APPEND SOFTMAX_KERNELS_LIST
                $<TARGET_OBJECTS:softmax_fwd_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        else()
            add_library(
                softmax_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                OBJECT ${SOFTMAX_FORWARD_INSTANCES})
            target_compile_options(
                softmax_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
            list(APPEND SOFTMAX_KERNELS_LIST
                $<TARGET_OBJECTS:softmax_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        endif()
    endforeach()
endforeach()
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        add_library(
            softmax_bwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${SOFTMAX_BACKWARD_INSTANCES})
        target_compile_options(
            softmax_bwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND SOFTMAX_KERNELS_LIST
            $<TARGET_OBJECTS:softmax_bwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(SOFTMAX_KERNELS_LIST "${SOFTMAX_KERNELS_LIST}" CACHE INTERNAL "")

# Sum
set(SUM_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/sum/kudnn_sum_instances.cpp")
set(SUM_INSTANCES_N "${CMAKE_CURRENT_LIST_DIR}/sum/kudnn_sum_instances_n.cpp")
set(SUM_N_INPUTS 1 2 3 4 5 6 7 8)
set(SUM_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float" OR IN_OUT_PREC STREQUAL "__fp16" OR IN_OUT_PREC STREQUAL "__bf16"))
            set(USE_NEON 1)
        endif()
        add_library(
            sum_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${SUM_INSTANCES})
        target_compile_options(
            sum_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
        list(APPEND SUM_KERNELS_LIST
            $<TARGET_OBJECTS:sum_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        foreach(SUM_N_INPUT ${SUM_N_INPUTS})
            add_library(
                sum_instances_${SUM_N_INPUT}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                OBJECT ${SUM_INSTANCES_N})
            target_compile_options(
                sum_instances_${SUM_N_INPUT}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC}
                -DN_INPUTS=${SUM_N_INPUT} -DUSE_NEON=${USE_NEON})
            list(APPEND SUM_KERNELS_LIST
                $<TARGET_OBJECTS:sum_instances_${SUM_N_INPUT}_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        endforeach()
    endforeach()
endforeach()
set(SUM_KERNELS_LIST "${SUM_KERNELS_LIST}" CACHE INTERNAL "")

# Concat
set(CONCAT_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/concat/kudnn_concat_inst.cpp")
set(CONCAT_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        set(USE_NEON 0)
        if ((${TARGET_PLATFORM} STREQUAL "KP920G") AND (IN_OUT_PREC STREQUAL "float"))
            set(USE_NEON 1)
        endif()
        add_library(
            concat_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${CONCAT_INSTANCES})
        target_compile_options(
            concat_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC} -DUSE_NEON=${USE_NEON})
        list(APPEND CONCAT_KERNELS_LIST
            $<TARGET_OBJECTS:concat_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(CONCAT_KERNELS_LIST "${CONCAT_KERNELS_LIST}" CACHE INTERNAL "")

# Vml
add_subdirectory(vml)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR} KERNELS_LIST)
add_library(kernels OBJECT ${KERNELS_LIST})

# Rnn
file(GLOB_RECURSE RNN_INSTANCES src "${CMAKE_CURRENT_SOURCE_DIR}/rnn/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/rnn/*.cpp")
add_library(rnn_instances OBJECT ${RNN_INSTANCES})
set(RNN_KERNELS_LIST $<TARGET_OBJECTS:rnn_instances>)
set(RNN_KERNELS_LIST "${RNN_KERNELS_LIST}" CACHE INTERNAL "")

# RMSNorm
set(RMSNORM_FWD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/rmsnorm/kudnn_rmsnorm_instances_fwd_cross.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/rmsnorm/kudnn_rmsnorm_instances_fwd_last.cpp")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        add_library(
            rmsnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${RMSNORM_FWD_INSTANCES})
        target_compile_options(
            rmsnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DSCALE_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND RMSNORM_KERNELS_LIST
            $<TARGET_OBJECTS:rmsnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(RMSNORM_KERNELS_LIST "${RMSNORM_KERNELS_LIST}" CACHE INTERNAL "")