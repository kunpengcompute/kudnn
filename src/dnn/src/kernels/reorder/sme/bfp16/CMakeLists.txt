aux_source_directory(${CMAKE_CURRENT_LIST_DIR} REORDER_MC_BFP16_IMPLS)

add_library(reorder_sme_bfp16 OBJECT ${REORDER_MC_BFP16_IMPLS})
