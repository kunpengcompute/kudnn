aux_source_directory(${CMAKE_CURRENT_LIST_DIR} MATMUL_LIST)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/VectorComputation MATMUL_IMPLS)

if (${TARGET_PLATFORM} STREQUAL "KP920Pro")
    aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/MatrixComputation MATMUL_IMPLS_MatrixComputation)
    list(APPEND MATMUL_IMPLS ${MATMUL_IMPLS_MatrixComputation})
endif()

add_library(matmul OBJECT ${MATMUL_LIST} ${MATMUL_IMPLS})
# 在 src/CMakeLists.txt 中
if(ENABLE_GCOV STREQUAL "on" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(matmul PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()
