aux_source_directory(${CMAKE_CURRENT_LIST_DIR} MATMUL_LIST)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/sve MATMUL_IMPLS)

list(REMOVE_ITEM MATMUL_IMPLS ${CMAKE_CURRENT_LIST_DIR}/sve/kudnn_default_impl_su8su8s32s32_mmla.cpp)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/neon MATMUL_IMPLS_NEON)

if (${TARGET_PLATFORM} STREQUAL "KP920G")
    list(REMOVE_ITEM MATMUL_IMPLS ${CMAKE_CURRENT_LIST_DIR}/sve/kudnn_default_impl_su8su8s32s32_dot.cpp)
else()
    list(REMOVE_ITEM MATMUL_IMPLS_NEON ${CMAKE_CURRENT_LIST_DIR}/neon/kudnn_default_impl_su8su8s32s32_mmla.cpp)
endif()

list(APPEND MATMUL_IMPLS ${MATMUL_IMPLS_NEON})

if (${TARGET_PLATFORM} STREQUAL "KP920F")
    aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/sme MATMUL_IMPLS_MC)
    list(APPEND MATMUL_IMPLS ${MATMUL_IMPLS_MC})
endif()

add_library(matmul OBJECT ${MATMUL_LIST} ${MATMUL_IMPLS})
