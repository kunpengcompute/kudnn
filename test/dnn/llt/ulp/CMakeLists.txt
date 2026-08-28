cmake_minimum_required(VERSION 3.14.1)
project(KPL_VML_TEST LANGUAGES C CXX)
# --------------------------------------------------------------------
# config the version of the shared library:
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# config build option
# --------------------------------------------------------------------
# Store in CPU architecture var the current build architecture
execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)

# --------------------------------------------------------------------
# cmake options
# --------------------------------------------------------------------
message(STATUS "CPU Architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

include("${PROJECT_SOURCE_DIR}/cmake/config.cmake")

add_subdirectory(src)
