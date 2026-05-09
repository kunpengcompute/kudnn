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
#include "kudnn_primitive_conf.hpp"

namespace KuDNN {
using namespace ARMCG;

template <CPUISAT isa>
struct KudnnJitDwConvFwdKernel : public JitGenerator {
public:
    explicit KudnnJitDwConvFwdKernel(const KudnnConvJitConfT &jcc) : jcp(jcc) {}

    const char *Name() const override
    {
        return "KudnnJitDwConvFwdKernel";
    }
    const char *SourceFile() const override
    {
        return __FILE__;
    }

    KudnnConvJitConfT jcp;

private:
    inline ZRegS GetSrcRegS(int idx)
    {
        assert(idx + RINDEX1 <= RINDEX31);
        return ZRegS(idx + RINDEX1);
    }
    inline ZRegS GetKerRegS(int idx)
    {
        assert(idx <= RINDEX31);
        return ZRegS(idx);
    }
    inline ZRegS GetAccRegS(int idx)
    {
        assert(idx + RINDEX4 <= RINDEX31);
        return ZRegS(idx + RINDEX4);
    }

    inline int DivUp(int a, int b)
    {
        return (a + b - 1) / b;
    }
    int GetOWStart(int ki, int padL)
    {
        return std::max(0, DivUp(padL - ki * (jcp.dilateW + 1), jcp.strideW));
    }
    int GetOWEnd(int urW, int ki, int padR)
    {
        return urW - std::max(0, DivUp(padR - (jcp.kw - 1 - ki) * (jcp.dilateW + 1), jcp.strideW));
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

    inline void LoadSrc(int urChBlock, int urW);
    inline void ApplyFilterUnrolled(int urChBlock, int urW, int padL, int padR);
    inline void StoreDst(int urChBlock, int urW);
    inline void ComputeLoop(int urW, int urChBlock, int padL, int padR);
    inline void OWLoop(int urChBlock);
    void Generate() override;

    const int vlen = CpuIsaTraits<isa>::VLEN;
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
