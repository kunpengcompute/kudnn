set(ALL_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16" "int8_t" "uint8_t" "int32_t")
set(FLOAT_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16")
set(COMPUTE_PRECISIONS "float")
set(SHORT_NAME_float f32)
set(SHORT_NAME___fp16 f16)
set(SHORT_NAME___bf16 bf16)
set(SHORT_NAME_int8_t s8)
set(SHORT_NAME_uint8_t u8)
set(SHORT_NAME_int32_t s32)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR} REORDER_INSTANCES)
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
