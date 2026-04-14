set(ALL_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16" "int8_t" "uint8_t" "int32_t")
set(FLOAT_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16")
set(COMPUTE_PRECISIONS "float")
set(SHORT_NAME_float f32)
set(SHORT_NAME___fp16 f16)
set(SHORT_NAME___bf16 bf16)
set(SHORT_NAME_int8_t s8)
set(SHORT_NAME_uint8_t u8)
set(SHORT_NAME_int32_t s32)

# Reorder
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/reorder REORDER_INSTANCES)
set(REORDER_KERNELS_LIST "")
foreach(IN_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
        add_library(
            reorder_instances_${SHORT_NAME_${IN_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${REORDER_INSTANCES})
        target_compile_options(
            reorder_instances_${SHORT_NAME_${IN_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DIN_TYPE=${IN_PREC} -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND REORDER_KERNELS_LIST
            $<TARGET_OBJECTS:reorder_instances_${SHORT_NAME_${IN_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(REORDER_KERNELS_LIST "${REORDER_KERNELS_LIST}" CACHE INTERNAL "")

# Convolution
include(convolution/conv_gen.cmake)

add_custom_command(
    OUTPUT ${CONVOLUTION_KERNELS_F32}
    COMMAND python3 ${CMAKE_SOURCE_DIR}/src/kernels/convolution/convcodegen.py f32
)

add_custom_command(
    OUTPUT ${CONVOLUTION_KERNELS_F16}
    COMMAND python3 ${CMAKE_SOURCE_DIR}/src/kernels/convolution/convcodegen.py f16
)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/convolution/ CONVOLUTION_PIPELINE)

add_library(convolution_kernels_pipeline OBJECT ${CONVOLUTION_PIPELINE})
add_library(convolution_kernels_f32 OBJECT ${CONVOLUTION_KERNELS_F32})
add_library(convolution_kernels_f16 OBJECT ${CONVOLUTION_KERNELS_F16})

list(APPEND CONV_KERNELS_LIST
    $<TARGET_OBJECTS:convolution_kernels_pipeline>
    $<TARGET_OBJECTS:convolution_kernels_f16>
    $<TARGET_OBJECTS:convolution_kernels_f32>)

set(CONV_KERNELS_LIST "${CONV_KERNELS_LIST}" CACHE INTERNAL "")

# GNorm
set(GNORM_FORWARD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/gnorm/kudnn_gnorm_instances_fwd.cpp")
set(GNORM_BACKWARD_INSTANCES "${CMAKE_CURRENT_LIST_DIR}/gnorm/kudnn_gnorm_instances_bwd.cpp")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        add_library(
            gnorm_forward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${GNORM_FORWARD_INSTANCES})
        target_compile_options(
            gnorm_forward_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DSCALE_SHIFT_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
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
        add_library(
            lnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${LNORM_FWD_INSTANCES})
        target_compile_options(
            lnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC}
            -DSCALE_SHIFT_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND LNORM_KERNELS_LIST
            $<TARGET_OBJECTS:lnorm_fwd_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()

# set(LNORM_KERNELS_LIST "")
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

# Softmax
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/softmax SOFTMAX_INSTANCES)
set(SOFTMAX_KERNELS_LIST "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    foreach(COMP_PREC ${COMPUTE_PRECISIONS})
        if(IN_OUT_PREC STREQUAL "__fp16")
            add_library(
                softmax_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                OBJECT ${SOFTMAX_INSTANCES})
            target_compile_options(
                softmax_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
                PUBLIC -DCOMP_TYPE=__fp16 -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
            list(APPEND SOFTMAX_KERNELS_LIST
                $<TARGET_OBJECTS:softmax_instances_fp16_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
        endif()
        add_library(
            softmax_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            OBJECT ${SOFTMAX_INSTANCES})
        target_compile_options(
            softmax_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}
            PUBLIC -DCOMP_TYPE=${COMP_PREC} -DIN_TYPE=${IN_OUT_PREC} -DOUT_TYPE=${IN_OUT_PREC})
        list(APPEND SOFTMAX_KERNELS_LIST
            $<TARGET_OBJECTS:softmax_instances_${SHORT_NAME_${COMP_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}_${SHORT_NAME_${IN_OUT_PREC}}>)
    endforeach()
endforeach()
set(SOFTMAX_KERNELS_LIST "${SOFTMAX_KERNELS_LIST}" CACHE INTERNAL "")

add_subdirectory(vml)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR} KERNELS_LIST)
add_library(kernels OBJECT ${KERNELS_LIST})
if(ENABLE_GCOV STREQUAL "on" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(kernels PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()