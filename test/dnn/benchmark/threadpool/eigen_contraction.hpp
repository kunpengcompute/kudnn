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

#ifndef EIGEN_CONTRACTION_HPP
#define EIGEN_CONTRACTION_HPP
#include "matmul_op.hpp"
#include "unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;
using Eigen::TensorMap;
using EigenThreadPool = Eigen::ThreadPool;

class EigenTensorContractionOp : public MatMulOp {
public:
    EigenTensorContractionOp(int M, int N, int K, EigenThreadPool *tp)
    {
        // C = A * B^T
        device = new Eigen::ThreadPoolDevice(tp, tp->NumThreads());
        sizeA[0] = M;
        sizeA[1] = K;
        sizeB[0] = N;
        sizeB[1] = K;
        sizeC[0] = M;
        sizeC[1] = N;
    }
    ~EigenTensorContractionOp()
    {
        delete device;
    }

    // implementation of matmul using eigen
    void Compute(float *aPtr, float *bPtr, float *cPtr, float *biasPtr)
    {
        const Eigen::TensorMap<Tensor<float, 2, Eigen::ColMajor>, Eigen::Aligned> A(aPtr, sizeA[0], sizeA[1]);
        const Eigen::TensorMap<Tensor<float, 2, Eigen::ColMajor>, Eigen::Aligned> B(bPtr, sizeB[0], sizeB[1]);
        Eigen::TensorMap<Tensor<float, 2, Eigen::ColMajor>, Eigen::Aligned> C(cPtr, sizeC[0], sizeC[1]);

        // contraction of A and B in K-dimention
        Eigen::array<Tensor<float, 2, Eigen::ColMajor>::DimensionPair, 1> dims;
        dims[0] = Tensor<float, 2, Eigen::ColMajor>::DimensionPair(1, 1);

        // do matmul
        C.device(*device) = A.contract(B, dims);
    }

private:
    Eigen::ThreadPoolDevice *device;
    Eigen::array<int, 2> sizeA, sizeB, sizeC;
};

#endif