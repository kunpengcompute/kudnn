file(GLOB_RECURSE ALPHAFOLD_SOURCES "alphafold/*")
file(GLOB_RECURSE COMMON_SOURCES "common/*")
file(GLOB_RECURSE UTILS_SOURCES "utils/*")
file(GLOB_RECURSE FUZZ_SOURCES "${PROJECT_SOURCE_DIR}/llt/fuzz/src/*")

if (${ENABLE_ASAN} STREQUAL "off")
    pybind11_add_module(_C
        hikp.cpp
        ${ALPHAFOLD_SOURCES}
        ${COMMON_SOURCES}
        ${UTILS_SOURCES})
else()
    pybind11_add_module(_C
        hikp.cpp
        ${ALPHAFOLD_SOURCES}
        ${COMMON_SOURCES}
        ${UTILS_SOURCES}
        ${FUZZ_SOURCES})
endif ()