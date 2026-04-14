aux_source_directory(${CMAKE_CURRENT_LIST_DIR} SRC_LIST)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(kdnn_gemm.cpp PROPERTIES COMPILE_FLAGS "-O2")
endif()

add_library(src OBJECT ${SRC_LIST})
if(ENABLE_GCOV STREQUAL "on" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(src PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()

add_subdirectory(kernels)
add_subdirectory(aarch64_codegen)
add_subdirectory(matmul_jit)
