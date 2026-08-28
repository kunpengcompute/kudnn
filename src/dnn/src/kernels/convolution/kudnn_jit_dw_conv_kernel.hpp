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

#ifndef KUDNN_JIT_DW_CONV_KERNEL_HPP
#define KUDNN_JIT_DW_CONV_KERNEL_HPP

#include "cg_jit_generator.hpp"
#include "kudnn_brgemm_conv_utils.hpp"
#include "kudnn_primitive_conf.hpp"

namespace KuDNN {
using namespace ARMCG;
template<CPUISAT isa>
struct KdnnJitDwConvFwdKernel : public JitGenerator {
public:
    explicit KdnnJitDwConvFwdKernel(const KdnnConvJitConfT &jcc) : jcp(jcc) {}

    const char *Name() const override
    {
        return "KdnnJitDwConvFwdKernel";
    }
    const char *SourceFile() const override
    {
        return __FILE__;
    }

    KdnnConvJitConfT jcp;

private:
    inline ZRegS GetSrcRegS(IntType idx)
    {
        return ZRegS(idx + RINDEX1);
    }
    inline ZRegS GetKerRegS(IntType idx)
    {
        return ZRegS(idx);
    }
    inline ZRegS GetAccRegS(IntType idx)
    {
        return ZRegS(idx + RINDEX4);
    }
    IntType GetOWStart(IntType ki, IntType padL)
    {
        IntType midKval = Math::CeilDiv(padL - ki * (jcp.dilateW + 1), jcp.strideW);
        return (midKval > 0 ? midKval : 0);
    }
    IntType GetOWEnd(IntType urW, IntType ki, IntType padR)
    {
        IntType resOwEnd = Math::CeilDiv(padR - (jcp.kw - 1 - ki) * (jcp.dilateW + 1), jcp.strideW);
        return urW - (resOwEnd > 0 ? resOwEnd : 0);
    }
    inline bool IsLayoutNXC(Layout tag)
    {
        return tag == Layout::NDHWC || tag == Layout::NHWC || tag == Layout::ACB;
    }
    inline bool IsSrcLayoutNXC()
    {
        return IsLayoutNXC(jcp.srcTag);
    }
    inline bool IsDstLayoutNXC()
    {
        return IsLayoutNXC(jcp.dstTag);
    }

    inline void LoadSrc(IntType urChBlock, IntType urW);
    inline void ApplyFilterUnrolled(IntType urChBlock, IntType urW, IntType padL, IntType padR);
    inline void StoreDst(IntType urChBlock, IntType urW);
    inline void ComputeLoop(IntType urW, IntType urChBlock, IntType padL, IntType padR);
    inline void OWLoop(IntType urChBlock);
    void Generate() override;

    const IntType vlen = CpuIsaTraits<isa>::VLEN;
    const XReg regInp = x1;
    const XReg regInpAux = x2;
    const XReg regKer = x3;
    const XReg regKerAux = x5;
    const XReg regChBlock = x6;
    const XReg regChBlockAux = x12;
    const XReg regOut = x7;
    const XReg regBia = x8;
    const XReg regKH = x9;
    const XReg regIterKH = x10;
    const XReg regOI = x11;
    const XReg regInpBufPtr = x13;
    const XReg regInpBufPtrAux = x14;
    const XReg regIWOffset = regInp;
    const XReg regTmpImm = x15;
    const XReg regInpStack = x17;
    const XReg regKerStack = x16;
    const XReg regOutStack = x27;
    const XReg regBiaStack = x19;
    const XReg regTmpAdr = x20;
};

} // namespace KuDNN

#endif // KUDNN_JIT_DW_CONV_KERNEL_HPP
