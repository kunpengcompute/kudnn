cmake_minimum_required(VERSION 3.14.1)

set(ULP_NAME ulp)

include_directories(${MATHLIB_INCLUDE_PATH})

file(GLOB common_source_list
    src
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.c"
    )

file(GLOB ulp_source_list
    src
    "${CMAKE_CURRENT_SOURCE_DIR}/ulp/*.c"
    )

file(GLOB vml_source_list
    "${MATH_VML_PATH}/kernels/vml/*.c"
    "${MATH_VML_PATH}/kernels/vml/*/*.c"
    "${MATH_VML_PATH}/kernels/*.c"
    )

add_executable(${ULP_NAME}
    ${ulp_source_list}
    ${common_source_list}
    ${CMAKE_CURRENT_SOURCE_DIR}/ulp.c
    ${vml_source_list}
    )

target_include_directories(${ULP_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/ulp
    ${MATH_VML_PATH}
    ${MATH_VML_PATH}/kernels
    ${MATH_VML_PATH}/kernels/vml
    )

target_compile_options(${ULP_NAME} PRIVATE ${COMPILE_OPTIONS} -march=armv8.5-a+fp+simd+bf16+sve)
set_target_properties(${ULP_NAME} PROPERTIES LINK_FLAGS ${LINK_CFLAGS})
target_link_libraries(${ULP_NAME} -lm -lmpfr -lgmp)

