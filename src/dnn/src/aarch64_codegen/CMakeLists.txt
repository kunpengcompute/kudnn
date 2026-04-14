cmake_minimum_required(VERSION 3.14)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/tools TOOLS_FILES)
include_directories(${CMAKE_CURRENT_LIST_DIR}/)

if (${TARGET_PLATFORM} STREQUAL "KP920Pro")
    add_compile_options(-DKUDNNL_920Pro)
endif()

message("Start building aarch64 codegen")

add_library(aarch64_codegen OBJECT ${SRC_FILES} ${TOOLS_FILES})
if(ENABLE_GCOV STREQUAL "on" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(aarch64_codegen PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()