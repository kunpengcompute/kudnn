/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuDNN is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <vector>
#include <random>
#include <cstdlib>
#include <getopt.h>

// eigen headers
#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/ThreadPool"
// detailed impl of abstract threadpool

// operator implementations
#include "blocking_counter.hpp"

#if defined(TEST_EIGEN)
#include "eigen_contraction.hpp"
#elif defined(TEST_KUDNN_MATMUL)
#include "kudnn_matmul_op.hpp"
#elif defined(TEST_KUDNN_IP)
#include "kudnn_ip_op.hpp"
#endif

void PrintUsage()
{
    std::cout << "" << std::endl;
    std::cout << "  Usage: ./bench_matmul --m <M> --n <N> --k <K>  [options]" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "    --inter=<COUNT>    Number of inter op threads" << std::endl;
    std::cout << "    --intra=<COUNT>    Number of intra op threads" << std::endl;
    std::cout << "    --loop=<COUNT>     Number of test loops" << std::endl;
    std::cout << "    --concurrency=<COUNT>  Number of concurrent runs in one loop" << std::endl;
    std::cout << "    --help           Print usage" << std::endl;
    std::cout << "" << std::endl;
}

template <typename T, size_t alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };

    T *allocate(size_t n) const noexcept(false)
    {
        return static_cast<T*>(aligned_alloc(alignment, n * sizeof(T)));
    }

    void deallocate(T *p, size_t n = 0) const noexcept
    {
        free(p);
    }
    virtual ~AlignedAllocator() = default;
};

int main(int argc, char *argv[])
{
    int num_inter_op_threads = 1;  // threading information
    int num_intra_op_threads = 1;

    // default value set to BOSS example
    int M = 512;
    int N = 128;
    int K = 1324;
    int nloop = 100;
    int nconcurrent = 1;

    // default cache size 8 MB
    int cachesize = 8 * 1024 * 1024;

    // timers
    std::chrono::time_point<std::chrono::high_resolution_clock> tstart;
    double tduration;

    // get inputs from command line
    static const char *opt_string = "m:n:k:x:y:l:c:h";
    static struct option long_options[] = {{"m", required_argument, NULL, 'm'},
        {"n", required_argument, NULL, 'n'},
        {"k", required_argument, NULL, 'k'},
        {"inter", required_argument, NULL, 'x'},
        {"intra", required_argument, NULL, 'y'},
        {"loop", required_argument, NULL, 'l'},
        {"concurrency", required_argument, NULL, 'c'},
        {"help", no_argument, NULL, 'h'},
        {NULL, no_argument, NULL, 0}};

    int long_index = 0;
    int opt = getopt_long(argc, argv, opt_string, long_options, &long_index);
    size_t val;
    while (opt != -1) {
        switch (opt) {
            case 'm':
                M = atoi(optarg);
                break;
            case 'n':
                N = atoi(optarg);
                break;
            case 'k':
                K = atoi(optarg);
                break;
            case 'x':
                num_inter_op_threads = atoi(optarg);
                break;
            case 'y':
                num_intra_op_threads = atoi(optarg);
                break;
            case 'l':
                nloop = atoi(optarg);
                break;
            case 'c':
                nconcurrent = atoi(optarg);
                break;
            case 'h':
                PrintUsage();
                return 1;
            default:
                break;
        }
        opt = getopt_long(argc, argv, opt_string, long_options, &long_index);
    }

    std::cout << "Matrix shape (M, N, K):         " << M << ", " << N << ", " << K << std::endl;
    std::cout << "Number of inter-op threads:     " << num_inter_op_threads << std::endl;
    std::cout << "Number of intra-op threads:     " << num_intra_op_threads << std::endl;
    std::cout << "Test for number of loops:       " << nloop << std::endl;
    std::cout << "Concurrent requests count:      " << nconcurrent << std::endl;
    std::cout << std::endl;

    // create 2 threadpools: one for interop scheduling, the other one for operation executions
    EigenThreadPool *threadpool_interop = new EigenThreadPool(num_inter_op_threads);
    EigenThreadPool *threadpool_intraop = new EigenThreadPool(num_intra_op_threads);

    std::cout << "---- Start generating random matrices ----" << std::endl;
    tstart = std::chrono::high_resolution_clock::now();

    // create 1 set of random matrices with C-matrix using at most "cachesize" for nloop
    int num_matrix = 1 + std::min(nloop, cachesize / M / N / (int)sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> mat_A_set(M * K * num_matrix, 0.0);
    std::vector<float, AlignedAllocator<float, 64>> mat_B_set(N * K * num_matrix, 0.0);
    std::vector<float, AlignedAllocator<float, 64>> mat_C_set(M * N * num_matrix, 0.0);
    std::vector<float, AlignedAllocator<float, 64>> mat_bias_set(M * N * num_matrix, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // initialize matrices
    std::generate(mat_A_set.begin(), mat_A_set.end(), [&]() { return dis(gen); });
    std::generate(mat_B_set.begin(), mat_B_set.end(), [&]() { return dis(gen); });
    std::generate(mat_C_set.begin(), mat_C_set.end(), [&]() { return dis(gen); });
    std::generate(mat_bias_set.begin(), mat_bias_set.end(), [&]() { return dis(gen); });

    // duplicate matrices for different concurrent requests
    std::vector<std::vector<float, AlignedAllocator<float, 64>>> mat_A_group(nconcurrent), mat_B_group(nconcurrent),
        mat_C_group(nconcurrent), mat_bias_group(nconcurrent);;
    for (int i = 0; i < nconcurrent; ++i) {
        mat_A_group[i].assign(mat_A_set.begin(), mat_A_set.end());
        mat_B_group[i].assign(mat_B_set.begin(), mat_B_set.end());
        mat_C_group[i].assign(mat_C_set.begin(), mat_C_set.end());
        mat_bias_group[i].assign(mat_bias_set.begin(), mat_bias_set.end());
    }
    std::cout << "Random inputs generated in "
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart).count()
              << " seconds. " << std::endl
              << std::endl;

    // =================================================================================================================
    //  Main part of matmul performance benchmark
    // =================================================================================================================
    std::cout << "---- Benchmarking MATMUL implementation ----" << std::endl;

    // initilize matmul op
#if defined(TEST_EIGEN)
    MatMulOp *matmul_fn = new EigenTensorContractionOp(M, N, K, threadpool_intraop);
#elif defined(TEST_KUDNN_MATMUL)
    MatMulOp *matmul_fn = new KdnnMatmulOp(M, N, K, threadpool_intraop);
#elif defined(TEST_KUDNN_IP)
    MatMulOp *matmul_fn = new KdnnIpOp(M, N, K, threadpool_intraop);
#elif defined(TEST_ONEDNN_MATMUL)
    MatMulOp *matmul_fn = new OnednnMatmulOp(M, N, K, threadpool_intraop);
#endif

    // reset timers
    tduration = 0.0;
    tstart = std::chrono::high_resolution_clock::now();

    // loops
    for (int i = 0; i < nloop; ++i) {
        int offset_A = (i % num_matrix) * M * K;
        int offset_B = (i % num_matrix) * N * K;
        int offset_C = (i % num_matrix) * M * N;
        int offset_Bias = (i % num_matrix) * N;
        CountingBarrier barrier;
        barrier.Init(nconcurrent);
        for (int j = 0; j < nconcurrent; ++j) {
            threadpool_interop->Schedule([&, j]() {
                matmul_fn->Compute(mat_A_group[j].data() + offset_A,
                    mat_B_group[j].data() + offset_B,
                    mat_C_group[j].data() + offset_C,
                    mat_bias_group[j].data() + offset_Bias);
                barrier.Notify();
            });
        }
        barrier.Wait();
    }
    tduration += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart).count();

    std::cout << "Concurrent matmul finished in : " << std::setw(16) << tduration << " seconds" << std::endl;
    std::cout << "Concurrent matmul finished in : " << std::setw(16) << tduration * 1000 / nloop << " ms / loop" << std::endl;
    // define M as the batchsize
    std::cout << "Average throughput: " << std::setw(16) << (double)M * nloop * nconcurrent / tduration << " QPS"
              << std::endl;
    std::cout << std::endl;

    delete matmul_fn;
    delete threadpool_interop;
    delete threadpool_intraop;

    return 0;
}