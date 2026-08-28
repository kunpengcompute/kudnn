aux_source_directory(${CMAKE_CURRENT_LIST_DIR} REORDER_LIST)

add_library(reorder OBJECT ${REORDER_LIST})

add_subdirectory(generic/default)
add_subdirectory(sve)
add_subdirectory(neon)

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    add_subdirectory(sme)
endif()