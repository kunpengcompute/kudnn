aux_source_directory(${CMAKE_CURRENT_LIST_DIR} SRC_LIST)



if (NOT (${KUDNN_CPU_RUNTIME} STREQUAL "THREADPOOL"))
    list(REMOVE_ITEM SRC_LIST "${CMAKE_CURRENT_LIST_DIR}/threadpool_impl.cpp")
endif ()

add_library(src OBJECT ${SRC_LIST})

add_subdirectory(kernels)
add_subdirectory(aarch64_codegen)
add_subdirectory(brgemm)
