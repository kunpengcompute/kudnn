cmake_minimum_required(VERSION 3.14)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/tools TOOLS_FILES)
include_directories(${CMAKE_CURRENT_LIST_DIR}/)

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    add_compile_options(-DKUDNNL_920F)
endif()

message("Start building aarch64 codegen")

add_library(aarch64_codegen OBJECT ${SRC_FILES} ${TOOLS_FILES})