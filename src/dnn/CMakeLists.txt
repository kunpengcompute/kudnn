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
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

add_compile_options(-fPIC -Wall -Werror -Wnarrowing -Wpointer-arith -Wextra -fopenmp -D_FORTIFY_SOURCE=2 -O2)
if (${TARGET_PLATFORM} STREQUAL "KP920Pro")
    add_compile_options(-march=armv9.2-a+fp+simd+bf16+sve+sve2+sme)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND LINKED_LIBS gomp)
    set(EXTEND_CMAKE_CXX_FLAGS "-Wimplicit-fallthrough=3 -Wshift-overflow=2 -Wduplicated-cond -Wtrampolines -Wlogical-op -Wreturn-local-addr -Wsized-deallocation")
    set(EXTEND_CMAKE_C_FLAGS "-Wimplicit-fallthrough=3 -Wshift-overflow=2 -Wformat-nonliteral -Wduplicated-cond -Wtrampolines -Wlogical-op -Wsuggest-attribute=format -Wduplicated-branches")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND LINKED_LIBS omp)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mllvm -treat-scalable-fixed-error-as-warning=true")
endif ()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTEND_CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTEND_CMAKE_CXX_FLAGS}")

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

if (${ENABLE_ASAN} STREQUAL "on")
    message(STATUS "Build ASAN version Library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKUDNN_ASAN")
    add_compile_options(-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-omit-frame-pointer)
    if (${TARGET_PLATFORM} STREQUAL "KP920Pro")
        add_compile_options(-msve-vector-bits=512)
    else ()
        add_compile_options(-msve-vector-bits=256)
    endif ()
endif ()

if (${TARGET_PLATFORM} STREQUAL "KP920B")
    add_compile_options(-DKUDNNL_920B)
endif()

if (${TARGET_PLATFORM} STREQUAL "KP920Pro")
    add_compile_options(-DKUDNNL_920Pro)
endif()

list(APPEND LINKED_LIBS kupl kblas)



add_subdirectory(src)

add_library(kudnn SHARED $<TARGET_OBJECTS:src>
                        $<TARGET_OBJECTS:matmul>
                        $<TARGET_OBJECTS:kernels>
                        $<TARGET_OBJECTS:aarch64_codegen>
                        ${CONV_KERNELS_LIST}
                        ${GNORM_KERNELS_LIST}
                        ${LNORM_KERNELS_LIST}
                        ${RMSNORM_KERNELS_LIST}
                        ${REORDER_KERNELS_LIST}
                        ${SOFTMAX_KERNELS_LIST})                  

set(PACKAGE_NUM_VERSION $ENV{PACKAGE_NUM_VERSION})
if ("$ENV{PACKAGE_NUM_VERSION}" STREQUAL "")
    set(PACKAGE_NUM_VERSION "1.26.0.RC1")
endif()
set_target_properties(kudnn PROPERTIES VERSION ${PACKAGE_NUM_VERSION} SOVERSION 1)

target_link_libraries(kudnn ${LINKED_LIBS})

install(TARGETS kudnn LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
