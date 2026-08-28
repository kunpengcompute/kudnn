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

#ifndef KUDNN_MATMUL_BENCH_HPP
#define KUDNN_MATMUL_BENCH_HPP

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"
using EigenThreadPool = Eigen::ThreadPool;
#include "matmul_op.hpp"
#include "kudnn.hpp"

template <typename T, typename U>
inline void Balance211(T n, U team, U tid, T& nStart, T& nEnd)
{
    if (team <= 1 || n == 0) {
        nStart = 0;
        nEnd = n;
        return;
    }
    T minPerTeam = n / team;
    T remainder = n - minPerTeam * team;  // i.e., n % teams.
    nStart = tid * minPerTeam + std::min(tid, remainder);
    nEnd = nStart + minPerTeam + (tid < remainder);
}

// detailed impl of abstract threadpool

class TestThreadPool : public KuDNN::Threading::ThreadpoolIface {
public:
    TestThreadPool(EigenThreadPool *eigenInterface)
    {
        tp.reset(eigenInterface);
    }

    int GetNumThreads() const override {
        return tp->NumThreads();
    }

    uint64_t GetFlags() const override {
        return KUDNN_ASYNCHRONOUS;
    }

    void ParallelFor(int n, const std::function<void(int, int)> &fn) override {
        int nthr = GetNumThreads();
        int njobs = std::min(n, nthr);

        for (int i = 0; i < njobs; i++) {
            tp->Schedule([i, n, njobs, fn]() {
                int start, end;
                Balance211(n, njobs, i, start, end);
                for (int j = start; j < end; j++) {
                    fn(j, n);
                }
            });
        }
    };

    bool IsInParallel() const override {
        return tp->CurrentThreadId() != -1;
    }


private:
    std::unique_ptr<EigenThreadPool> tp;
};

// kdnn matmul op impl
class KdnnMatmulOp : public MatMulOp {
public:
    KdnnMatmulOp(int m, int n, int k, EigenThreadPool *tp)
    {
        threadpool = new TestThreadPool(tp);
        matATensor = {{m, k}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
        matBTensor = {{k, n}, KuDNN::Element::TypeT::F32, KuDNN::Layout::BA};
        matCTensor = {{m, n}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    }

    ~KdnnMatmulOp()
    {

    }

    // implementation of matmul using eigen
    void Compute(float *aPtr, float *bPtr, float *cPtr, float *biasPtr)
    {
        KuDNN::Threading::ActivateThreadpool(threadpool);
        KuDNN::Gemm gemmLayer(matATensor, matBTensor, matCTensor);
        gemmLayer.Run(aPtr, bPtr, cPtr);
    }

private:
    KuDNN::TensorInfo matATensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    KuDNN::TensorInfo matBTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    KuDNN::TensorInfo matCTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    TestThreadPool *threadpool = nullptr;
};

#endif