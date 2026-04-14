# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# ----------
# FindKPL
# ----------
#
# This module defines the following variables:
#
#   KUDNN_FOUND          - True if KPL was found
#   KPL_INCLUDE_DIRS   - include directories for KPL
#   KPL_LIBRARIES      - link against this library to use KPL
#

# Use KUDNN_ROOT_DIR environment variable to find the library and headers
find_path(KUDNN_INCLUDE_DIR
  NAMES kudnn.hpp
  PATHS ENV KUDNN_ROOT_DIR
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)
set(AARCH64_CODEGEN_PATH "$ENV{KUDNN_SRC_DIRS}/src/aarch64_codegen/")

find_library(KUDNN_LIBRARY
  NAMES kudnn
  PATHS ENV KUDNN_ROOT_DIR
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KuDNN DEFAULT_MSG
  KUDNN_INCLUDE_DIR
  KUDNN_LIBRARY
)

mark_as_advanced(
  KUDNN_LIBRARY
  KUDNN_INCLUDE_DIR
)

# Find the extra libraries and include dirs
if(KUDNN_FOUND)
  list(APPEND KUDNN_INCLUDE_DIRS ${KUDNN_INCLUDE_DIR})
  list(APPEND KUDNN_LIBRARIES
    ${KUDNN_LIBRARY}
    kblas)
endif()