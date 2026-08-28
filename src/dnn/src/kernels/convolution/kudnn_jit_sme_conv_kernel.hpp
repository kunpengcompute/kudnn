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
#ifndef KUDNN_JIT_MC_CONV_KERNEL_HPP
#define KUDNN_JIT_MC_CONV_KERNEL_HPP

#include "cg_jit_generator.hpp"
#include "kudnn_conv_const.hpp"
#include "kudnn_primitive_conf.hpp"

#ifdef KUDNNL_920F

namespace KuDNN {

struct JitConvMCKernel : public ARMCG::JitGenerator {
    explicit JitConvMCKernel(const KdnnConvJitConfT &ajcp) : jcp(ajcp) {}
    KdnnConvJitConfT jcp;
public:
    const char *Name() const override
    {
        return "KdnnJitConvMCKernel";
    }
    const char *SourceFile() const override
    {
        return __FILE__;
    }
private:
    using Reg64T = const ARMCG::XReg;

    Reg64T param = ARMCG::ABI_PARAM1;
    Reg64T regInp = x2;
    Reg64T regWei = x3;
    Reg64T regOut = x4;
    Reg64T regBia = x5;
    Reg64T regM = x6;
    Reg64T regN = x7;
    Reg64T regK = x8;

    Reg64T regCurM = x9;
    Reg64T regCurN = x10;
    Reg64T regCurK = x23;
    Reg64T regCurkTail = x22;
    Reg64T regCurBia = x13;
    Reg64T regTmp = x14;
    Reg64T regCurOut = x25;
    Reg64T regCurA = x26;

    void Generate() override;

    void GemmKernelGenerate();
    void DirectKernelGenerate();
    void ComputeLoop(const IntType& urW);
    void ComputeLoopWithStride(const IntType& urW);
    void StoreOutput(Reg64T &cStride);

    void BiasLoad();
    void SetMatrixPred(Reg64T &mReg, Reg64T &nReg, Reg64T &mCurReg, Reg64T &nCurReg);
    void DirectKernelPrepare(const IntType& urW);
    void KfirstLoad(Reg64T &aStride);
    void KloopFmopaKernel(Reg64T &aStride);
    void KloopFmopaLast();
    void KloopFmopaTail(Reg64T &aStride);
};
} // namespace KuDNN

#endif // KUDNNL_920F

#endif
