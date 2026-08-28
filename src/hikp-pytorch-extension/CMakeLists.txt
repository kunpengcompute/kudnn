cmake_minimum_required(VERSION 3.14)

project(HIKPEX LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_C_FLAGS "")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SKIP_RPATH TRUE)

set(LINKED_LIBS "")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

list(APPEND LINKED_LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

include_directories(${PROJECT_SOURCE_DIR}/src/csrc)
include_directories(${PROJECT_SOURCE_DIR}/src/csrc/alphafold)
include_directories(${PROJECT_SOURCE_DIR}/src/csrc/common)
include_directories(${PROJECT_SOURCE_DIR}/src/csrc/utils)
include_directories(${KPL_BLAS_DIR}/blas/out/release/kplblas/generated)
include_directories(${TORCH_INCLUDE_DIRS})

add_compile_options(-fPIC -Wall -fopenmp)
add_link_options(-L${PROJECT_OUTPUT_DIR}/lib)

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    add_compile_options(-march=armv8.2-a+sve2+bf16)
    add_compile_options(-msve-vector-bits=512)
    add_compile_options(-DBGEMM -DKPEX_920F)
    list(APPEND LINKED_LIBS ${DEPENDENCIES_DIR}/libunigemm.a numa)
else ()
    add_compile_options(-march=armv8.2-a+sve+fp16+bf16)
    add_compile_options(-msve-vector-bits=256)
    add_compile_options(-DBGEMM)
endif ()

add_link_options(-Wl,-z,relro,-z,now)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# release version lib compile options
if (NOT CMAKE_BUILD_TYPE STREQUAL "debug")
    message(STATUS "Build Release Type Library")
    add_compile_options(-fstack-protector-strong -O3)
    add_link_options(-s)
else ()
    add_compile_options(-O2)
endif ()

if(${ENABLE_GCOV} STREQUAL "on")
    message(STATUS "Build GCOV version Library")
    add_compile_options(-fprofile-arcs -ftest-coverage -fno-exceptions)
    add_link_options(--coverage)
endif()

if (${ENABLE_ASAN} STREQUAL "on")
    message(STATUS "Build ASAN version Library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIKPEX_FUZZ_ENABLE")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHIKPEX_FUZZ_ENABLE")
    add_compile_options(-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-omit-frame-pointer)
    include_directories(${PROJECT_SOURCE_DIR}/llt/fuzz/include)
    include_directories(${PROJECT_SOURCE_DIR}/../../test/dnn/llt/fuzz/include)
    list(APPEND LINKED_LIBS ${DEPENDENCIES_DIR}/libSecodefuzz.a ${DEPENDENCIES_DIR}/libclang_rt.asan.a)
endif ()

list(APPEND LINKED_LIBS omp kplblas)
add_subdirectory(src/csrc)
target_link_libraries(_C PRIVATE ${LINKED_LIBS})

