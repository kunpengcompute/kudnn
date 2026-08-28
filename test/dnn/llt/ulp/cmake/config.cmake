# --------------------------------------------------------------------
# cmake options
# --------------------------------------------------------------------
option(USE_CLANG "use clang compiler"                               OFF)

if (${USE_CLANG})
    execute_process(COMMAND which clang OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE CMAKE_C_COMPILER)
    execute_process(COMMAND which clang++ OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE CMAKE_CXX_COMPILER)
    set(CMAKE_C_COMPILER_ID "Clang")
else ()
    execute_process(COMMAND which gcc OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE CMAKE_C_COMPILER)
    execute_process(COMMAND which g++ OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE CMAKE_CXX_COMPILER)
    set(CMAKE_C_COMPILER_ID "GNU")
endif()
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion  OUTPUT_VARIABLE C_VERSION)
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion  OUTPUT_VARIABLE CXX_VERSION)
if (NOT ${C_VERSION} STREQUAL ${CXX_VERSION})
    message(FATAL_ERROR "the C and C++ compiler version is not match, terminate the CMAKE")
endif()

set(MATH_VML_PATH           "${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/dnn/src")
set(MATHLIB_PATH            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../dependencies/lib")
set(MATHLIB_INCLUDE_PATH    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../dependencies/include")

set(COMPILE_OPTIONS  -std=c99 -pipe -O3 -Wall -Wno-missing-braces -Werror -DNDEBUG -DUSE_MPFR)

set(LINK_CFLAGS "-Wl,-z,relro -Wl,-z,noexecstack")

link_directories(${MATHLIB_PATH})


