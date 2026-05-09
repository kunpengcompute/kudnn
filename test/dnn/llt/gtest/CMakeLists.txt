cmake_minimum_required(VERSION 3.14)

project(KUDNN_GTEST LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(-fPIC -Wall -Wpedantic -fopenmp -Wextra -Wnarrowing -Wpointer-arith -march=armv8.5-a+fp+simd+bf16+sve)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fstack-protector-strong -O3)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
else ()
    add_compile_options(-O2 -fveclib=MATHLIB)
endif ()

if (TESTS_TYPE STREQUAL "MatrixComputation")
    add_definitions(-DMatrixComputation920Pro)
else ()
    add_definitions(-DVectorComputation920Pro)
endif ()

include("${PROJECT_SOURCE_DIR}/cmake/FindKuDNN.cmake")
if (KUDNN_FOUND)
    include_directories(${KUDNN_INCLUDE_DIRS})
    message(STATUS "GTest depend Library: ${KUDNN_LIBRARIES}")
    message(STATUS "GTest depend Library headers: ${KUDNN_INCLUDE_DIRS}")
endif ()

# header directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${AARCH64_CODEGEN_PATH})

# lib directories
if (${COMP_TOOLCHAIN} STREQUAL "clang")
link_directories(${PROJECT_SOURCE_DIR}/gtest_lib/bisheng)
else ()
link_directories(${PROJECT_SOURCE_DIR}/gtest_lib/gcc)
endif ()

# add gtest source files
aux_source_directory(${PROJECT_SOURCE_DIR}/src GTEST_SRC_FILES)

# add sanitizer link options
if("${ENABLE_ASAN}" STREQUAL "on")
    add_link_options(-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-omit-frame-pointer)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_link_options(-lasan)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_link_options(-shared-libsan)
    endif ()
endif ()

# add gcov options
if(${ENABLE_GCOV} STREQUAL "on")
    message(STATUS "Build GCOV version Library")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fprofile-arcs -ftest-coverage)
        add_link_options(-fprofile-arcs -ftest-coverage)
        list(APPEND LINKED_LIBS gcov)
    else ()
        add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
        add_link_options(-fprofile-instr-generate)
    endif ()
endif()

add_link_options(-fPIC)
link_libraries(gtest_main gtest ${KUDNN_LIBRARIES} pthread)

# link omp when using clang while link gomp while using gcc
if (${COMP_TOOLCHAIN} STREQUAL "clang")
    link_libraries(omp)
else ()
    link_libraries(gomp)
endif ()

add_executable(kudnn_gtest ${GTEST_SRC_FILES})
