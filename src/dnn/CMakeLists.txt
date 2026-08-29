cmake_minimum_required(VERSION 3.14)

project(KuDNN LANGUAGES C CXX)

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

include_directories(${PROJECT_SOURCE_DIR}/src/aarch64_codegen)
include_directories(${PROJECT_SOURCE_DIR}/src/brgemm)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
add_compile_options(-fPIC -Wall -Werror -Wnarrowing -Wpointer-arith -Wextra -fopenmp)
if (${TARGET_PLATFORM} STREQUAL "KP920F")
    add_compile_options(-march=armv9.2-a+fp+simd+bf16+sve+sve2+sme)
elseif (${TARGET_PLATFORM} STREQUAL "KP920G")
    message(STATUS "KuDNN is built with 920G features!!!")
    add_compile_options(-march=armv9.2-a+fp+simd+i8mm+sve2)
else ()
    add_compile_options(-march=armv8.5-a+fp+simd+bf16+sve)
endif ()

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,relro,-z,now")

# release version lib compile options
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build Release Type Library")
    add_compile_options(-fstack-protector-strong -O3)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
else ()
    add_compile_options(-O2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKUDNN_DEBUG")
    message(STATUS "Build Debug Type Library")
endif ()

set(LINKED_LIBS "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DUSE_OPENMP -Wall -Wuninitialized -Wundef -Wunused -Wdate-time -Wvla -Wdisabled-optimization -Wempty-body -Wignored-qualifiers -Wtype-limits -Wshift-negative-value -Wswitch-default -Wframe-larger-than=65536 -Wwrite-strings -Wmissing-format-attribute -Wformat-nonliteral -Wmissing-include-dirs -Wmissing-declarations -Wreturn-local-addr -Wredundant-decls -Woverloaded-virtual -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -Wno-conversion-null")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wuninitialized -Wundef -Wunused -Wdate-time -Wvla -Wdisabled-optimization -Wempty-body -Wignored-qualifiers -Wtype-limits -Wshift-negative-value -Wswitch-default -Wframe-larger-than=65536 -Wwrite-strings -Wmissing-format-attribute -Wformat-nonliteral -Wmissing-include-dirs -Wmissing-declarations -Wreturn-local-addr -Wredundant-decls -Wmissing-prototypes -Wstrict-prototypes")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND LINKED_LIBS gomp)
    set(EXTEND_CMAKE_CXX_FLAGS "-Wimplicit-fallthrough=3 -Wshift-overflow=2 -Wduplicated-cond -Wtrampolines -Wlogical-op -Wreturn-local-addr -Wsized-deallocation")
    set(EXTEND_CMAKE_C_FLAGS "-Wimplicit-fallthrough=3 -Wshift-overflow=2 -Wformat-nonliteral -Wduplicated-cond -Wtrampolines -Wlogical-op -Wsuggest-attribute=format -Wduplicated-branches")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND LINKED_LIBS omp)
    set(CMAKE_CXX_FLAGS "-Wno-error=alias-template-in-declaration-name")
endif ()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTEND_CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTEND_CMAKE_CXX_FLAGS}")

if(${ENABLE_ASAN} STREQUAL "on")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(--param max-gcse-memory=167772160)
    endif ()
endif()

if(${ENABLE_GCOV} STREQUAL "on")
    message(STATUS "Build GCOV version Library")
    add_compile_options(-fprofile-arcs -ftest-coverage)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.3.1")
            message(WARNING "GCC version ${CMAKE_CXX_COMPILER_VERSION} is too old. the -ftree-vrp has to turn off")
	        add_compile_options(-fno-tree-vrp)
        endif() 
        list(APPEND LINKED_LIBS gcov)
    else ()
        add_link_options(--coverage)
    endif ()
endif()

if (${ENABLE_ASAN} STREQUAL "on")
    message(STATUS "Build ASAN version Library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKUDNN_ASAN")
    add_compile_options(-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-omit-frame-pointer)
    if (${TARGET_PLATFORM} STREQUAL "KP920F")
        add_compile_options(-msve-vector-bits=512)
    else ()
        add_compile_options(-msve-vector-bits=256)
    endif ()
endif ()

if (${TARGET_PLATFORM} STREQUAL "KP920B")
    add_compile_options(-DKUDNNL_920B)
endif()

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    add_compile_options(-DKUDNNL_920F)
endif()

if (${TARGET_PLATFORM} STREQUAL "KP920G")
    add_compile_options(-DKUDNNL_920G)
endif()

# multi-thread implementation
set(KUDNN_CPU_THREADING_RUNTIME "OMP")
if (${KUDNN_CPU_RUNTIME} STREQUAL "THREADPOOL")
    set(KUDNN_CPU_THREADING_RUNTIME "THREADPOOL")
endif ()

configure_file(
    "${PROJECT_SOURCE_DIR}/include/kudnn_config.h.in"
    "${PROJECT_SOURCE_DIR}/include/kudnn_config.h"
)

if (${TARGET_PLATFORM} STREQUAL "KP920F" AND ${ENABLE_HBM} STREQUAL "on")
    add_compile_definitions("USE_HBM")
    list(APPEND LINKED_LIBS memkind)
endif ()

add_subdirectory(src)

set(FAST_REORDER_MC_KERNELS_LIST "")

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    set(FAST_REORDER_MC_KERNELS_LIST $<TARGET_OBJECTS:reorder_sme_bfp16>)
endif()

add_library(kudnn SHARED $<TARGET_OBJECTS:src>
                        $<TARGET_OBJECTS:matmul>
                        $<TARGET_OBJECTS:reorder>
                        $<TARGET_OBJECTS:reorder_fast>
                        $<TARGET_OBJECTS:reorder_neon_su8_diff_lt>
                        $<TARGET_OBJECTS:kernels>
                        $<TARGET_OBJECTS:aarch64_codegen>
                        $<TARGET_OBJECTS:brgemm>
                        ${BINARY_KERNELS_LIST}
                        ${BNORM_KERNELS_LIST}
                        ${CONV_KERNELS_LIST}
                        ${GNORM_KERNELS_LIST}
                        ${LNORM_KERNELS_LIST}
                        ${LRN_KERNELS_LIST}
                        ${POOLING_KERNELS_LIST}
                        ${PRELU_KERNELS_LIST}
                        ${SHUFFLE_KERNELS_LIST}
                        ${REDUCTION_KERNELS_LIST}
                        ${REORDER_KERNELS_LIST}
                        ${FAST_REORDER_KERNELS_LIST}
                        ${FAST_REORDER_MC_KERNELS_LIST}
                        ${SOFTMAX_KERNELS_LIST}
                        ${SUM_KERNELS_LIST}
                        ${CONCAT_KERNELS_LIST}
                        ${RMSNORM_KERNELS_LIST}
                        ${RNN_KERNELS_LIST})

set(PACKAGE_NUM_VERSION $ENV{package_version})
if ("$ENV{package_version}" STREQUAL "")
    set(PACKAGE_NUM_VERSION "26.2.0")
endif()
set_target_properties(kudnn PROPERTIES VERSION "1.${PACKAGE_NUM_VERSION}" SOVERSION 1)

target_link_libraries(kudnn ${LINKED_LIBS})

install(TARGETS kudnn LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    PATTERN "kudnn_config.h.in" EXCLUDE)
