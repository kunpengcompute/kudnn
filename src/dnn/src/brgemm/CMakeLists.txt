cmake_minimum_required(VERSION 3.14)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/ SRC_FILES)
include_directories(${CMAKE_CURRENT_LIST_DIR}/)

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    add_compile_options(-DKUDNNL_920F)
endif()

message("Start building kdnn brgemm")
add_library(brgemm OBJECT ${SRC_FILES})
