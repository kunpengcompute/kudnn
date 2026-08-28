cmake_minimum_required(VERSION 3.14)

project(KUDNN_GTEST LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC -Wall -Werror -Wpedantic -fopenmp -Wextra -Werror -Wnarrowing -Wpointer-arith -march=armv8.5-a+fp+simd+bf16+sve)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fstack-protector-strong -O3)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
else ()
    add_compile_options(-O1)
endif ()

include("${PROJECT_SOURCE_DIR}/cmake/FindKuDNN.cmake")
if (KUDNN_FOUND)
    include_directories(${KUDNN_INCLUDE_DIRS})
    message(STATUS "GTest Library: ${KUDNN_LIBRARIES}")
    message(STATUS "GTest Library headers: ${KUDNN_INCLUDE_DIRS}")
endif ()

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    add_compile_options(-DKUDNNL_920F)
endif()

# header directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${AARCH64_CODEGEN_PATH})
include_directories(${BRGEMM_PATH})
include_directories(${SRC_PATH})

# lib directories
if (${COMP_TOOLCHAIN} STREQUAL "clang")
link_directories(${PROJECT_SOURCE_DIR}/gtest_lib/bisheng)
else ()
link_directories(${PROJECT_SOURCE_DIR}/gtest_lib/gcc)
endif ()

# add gtest source files
aux_source_directory(${PROJECT_SOURCE_DIR}/src GTEST_SRC_FILES)

if (${KUDNN_CPU_RUNTIME} STREQUAL "THREADPOOL")
    set(KUDNN_CPU_THREADING_RUNTIME "THREADPOOL")
    find_package(Eigen3 REQUIRED 3.3 NO_MODULE)
    if(Eigen3_FOUND)
        include_directories(${EIGEN3_INCLUDE_DIRS})
        message(STATUS "Threadpool Config: Eigen (${EIGEN3_INCLUDE_DIRS})")
    endif()
else ()
    list(REMOVE_ITEM GTEST_SRC_FILES "${PROJECT_SOURCE_DIR}/src/matmul_threadpool_gtest.cpp")
endif ()

# add sanitizer link options
if("${ENABLE_ASAN}" STREQUAL "on")
    add_link_options(-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-omit-frame-pointer)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_link_options(-lasan)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_link_options(-shared-libsan)
    endif ()
endif ()
link_libraries(gtest_main gtest gcov ${KUDNN_LIBRARIES} pthread)

add_compile_options(-fprofile-arcs -ftest-coverage)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND LINKED_LIBS gcov) # gcc的话链接gcov
else ()
    add_link_options(--coverage) # clang的话用这个
endif ()

# link omp when using clang while link gomp while using gcc
if (${COMP_TOOLCHAIN} STREQUAL "clang")
    link_libraries(omp)
else ()
    link_libraries(gomp)
endif ()

add_executable(kudnn_gtest ${GTEST_SRC_FILES})
