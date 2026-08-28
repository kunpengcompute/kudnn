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

#ifndef ONEDNN_MATMUL_OP_HPP
#define ONEDNN_MATMUL_OP_HPP

#include "matmul_op.hpp"
#include "oneapi/dnnl/dnnl.hpp"

class OnednnMatmulOp : public MatMulOp {
public:
    OnednnMatmulOp(int M, int N, int K, Eigen::ThreadPool *eigenTp)
    {
        // initialize engine and stream for onednn
        threadpool.reset(new OneDnnThreadPool(eigenTp));
        std::cout << "threapool has #" << threadpool->get_num_threads() << std::endl;

        // initialize matmul primitive parameters
        A_dims = {M, K};
        B_dims = {K, N};
        C_dims = {M, N};
    }
    ~OnednnMatmulOp()
    {}

    // implementation of matmul using eigen
    void Compute(float *aPtr, float *bPtr, float *cPtr, float *biasPtr)
    {
        // make stream
        auto cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
        auto cpu_stream = dnnl::stream(dnnl::threadpool_interop::make_stream(cpu_engine, threadpool.get()));

        // set data handle to input matrices
        std::shared_ptr<dnnl::matmul::primitive_desc> fwd_pd;
        std::shared_ptr<dnnl::primitive> matmul_fwd;

        std::shared_ptr<dnnl::memory::desc> src_md;
        std::shared_ptr<dnnl::memory::desc> weight_md;
        std::shared_ptr<dnnl::memory::desc> dst_md;
        std::shared_ptr<dnnl::memory> src_mem;
        std::shared_ptr<dnnl::memory> weight_mem;
        std::shared_ptr<dnnl::memory> dst_mem;

        // memory description
        src_md.reset(new dnnl::memory::desc(A_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab));
        weight_md.reset(new dnnl::memory::desc(B_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab));
        dst_md.reset(new dnnl::memory::desc(C_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab));

        // inner-product primitive generation (w/o bias and post-ops)
        fwd_pd.reset(new dnnl::matmul::primitive_desc(cpu_engine, *src_md, *weight_md, *dst_md));
        matmul_fwd.reset(new dnnl::matmul(*fwd_pd));

        // assign input data to onednn memory buffers
        src_mem.reset(new dnnl::memory(*src_md, cpu_engine, aPtr));
        weight_mem.reset(new dnnl::memory(*weight_md, cpu_engine, bPtr));
        dst_mem.reset(new dnnl::memory(*dst_md, cpu_engine, cPtr));

        // matmul args
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, *src_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, *weight_mem});
        matmul_args.insert({DNNL_ARG_DST, *dst_mem});

        // compute matmul
        matmul_fwd->execute(cpu_stream, matmul_args);
    }

private:
    std::shared_ptr<OneDnnThreadPool> threadpool = nullptr;

    // Memory descriptors.
    dnnl::memory::dims A_dims;
    dnnl::memory::dims B_dims;
    dnnl::memory::dims C_dims;
};