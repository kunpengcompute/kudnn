aux_source_directory(${CMAKE_CURRENT_LIST_DIR} REORDER_FAST_IMPLS)

add_library(reorder_fast OBJECT ${REORDER_FAST_IMPLS})
