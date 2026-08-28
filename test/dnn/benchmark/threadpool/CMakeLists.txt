cmake_minimum_required(VERSION 3.14)

project(MATMUL_BENCHMARK LANGUAGES C CXX)

# set header and dependencies
set(ONEDNN_HEADER_DIR your_path_to/KPL_DNN/out/test/dnn/oneDNN-3.4/install/include)
set(ONEDNN_LIB_DIR your_path_to/KPL_DNN/out/test/dnn/oneDNN-3.4/install/lib64)
set(ACL_LIB_DIR your_path_to/KPL_DNN/out/test/dnn/acl/23.11/build)
set(KUDNN_HEADER_DIR your_path_to/KPL_DNN/out/include)
set(KUDNN_LIB_DIR your_path_to/KPL_DNN/out/lib)

include_directories(${ONEDNN_HEADER_DIR})
include_directories(${KUDNN_HEADER_DIR})
link_directories(${KUDNN_LIB_DIR})
link_directories(${ONEDNN_LIB_DIR})
link_directories(${ACL_LIB_DIR})

# eigen dependencies
find_package(Eigen3 REQUIRED 3.3 NO_MODULE)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Eigen: (${EIGEN3_INCLUDE_DIRS})")
endif()

# eigen bench
add_compile_options(-O3)
add_executable(bench_mamtul_eigen main.cpp)
target_compile_definitions(bench_mamtul_eigen PUBLIC TEST_EIGEN)

# kdnn matmul bench
add_executable(bench_mamtul_kdnn main.cpp)
target_link_libraries(bench_mamtul_kdnn PRIVATE kdnn)
target_compile_definitions(bench_mamtul_kdnn PUBLIC TEST_KUDNN_MATMUL)

# kdnn ip bench
add_executable(bench_ip_kdnn main.cpp)
target_link_libraries(bench_ip_kdnn PRIVATE kdnn)
target_compile_definitions(bench_ip_kdnn PUBLIC TEST_KUDNN_IP)

# onednn ip bench
add_executable(bench_mamtul_onednn main.cpp)
target_link_libraries(bench_mamtul_onednn PRIVATE dnnl kdnn arm_compute)
target_compile_definitions(bench_mamtul_onednn PUBLIC TEST_KUDNN_IP)