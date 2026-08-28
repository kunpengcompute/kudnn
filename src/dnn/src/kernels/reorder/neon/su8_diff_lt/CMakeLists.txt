aux_source_directory(${CMAKE_CURRENT_LIST_DIR} REORDER_NEON_SU8_DIFF_LT_IMPL)

add_library(reorder_neon_su8_diff_lt OBJECT ${REORDER_NEON_SU8_DIFF_LT_IMPL})
