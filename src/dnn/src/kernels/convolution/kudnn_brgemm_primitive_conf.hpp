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

#ifndef KUDNN_BRGEMM_PRIMITIVE_CONF_HPP
#define KUDNN_BRGEMM_PRIMITIVE_CONF_HPP

#include <memory>
#include <cstddef>

#include "kudnn_brgemm_containers.hpp"
#include "kudnn_brgemm_types.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_layout.hpp"
#include "types/kudnn_propagation_type.hpp"

#include "kudnn_primitive_conf.hpp"

namespace KuDNN {

constexpr IntType BRGEMM_PRIM_CONF_NUM_16 = 16;

enum class KdnnBrgemmConvExecTypeT {
    UNDEFINED = 0,
    BASE,
    TRANS,
    VPAD
};

enum class KdnnBrgemmConvLoopOrderT {
    NDHWGC = 0,
    NGCDHW
};

enum class KdnnBrgconvImplT {
    UNDEFINED = 0,
    BRGCONV,
    BRGCONV_1X1,
    BRDGMM_DW
};

struct KdnnBrgemmConvConfT {
    ARMCG::CPUISAT isa;
    Propagation propKind;
    KdnnConvVersionT ver;
    KdnnBrgemmConvLoopOrderT loopOrder;
    KdnnConvHarnessT harness;
    IntType simdW;
    IntType accSimdW;
    IntType ndims;
    IntType mb;
    IntType ngroups;
    IntType ic;
    IntType oc;
    IntType ocWithoutPadding;
    IntType icWithoutPadding;
    IntType odBlock;
    IntType ohBlock;
    IntType nbOd;
    IntType nbOh;
    IntType idBlock;
    IntType ihBlock;
    IntType nbId;
    IntType nbIh;
    IntType inpBufferSize;
    IntType inpBufferMaskSize;
    KdnnBrgemmConvExecTypeT execType;
    IntType id;
    IntType ih;
    IntType iw;
    IntType od;
    IntType oh;
    IntType ow;
    IntType os;
    IntType is;
    IntType idp;
    IntType ihp;
    IntType iwp;
    IntType icp;
    IntType odp;
    IntType ohp;
    IntType owp;
    IntType ocp;
    IntType fPad;
    IntType lPad;
    IntType tPad;
    IntType backPad;
    IntType rPad;
    IntType bPad;
    IntType lOvf;
    IntType rovf;
    IntType tOvf;
    IntType bOvf;
    IntType fOvf;
    IntType backOvf;
    IntType kd;
    IntType kh;
    IntType kw;
    IntType extKd;
    IntType extKh;
    IntType extKw;
    IntType kdBlock;
    IntType khBlock;
    IntType kwBlock;
    IntType kdBlockPad;
    IntType khBlockPad;
    IntType kwBlockPad;
    IntType strideD;
    IntType strideH;
    IntType strideW;
    IntType dilateD;
    IntType dilateH;
    IntType dilateW;
    Layout srcTag;
    Layout weiTag;
    Layout dstTag;
    bool withBias;
    bool withSum;
    bool withEltwise;
    bool withBinary;
    bool isFusedConv;
    bool isIsBlocking;
    bool isOsBlocking;
    bool isRtus;
    bool ununrollBdLoop {false};
    IntType nbIc;
    IntType icBlock;
    IntType nbOc;
    IntType ocBlock;
    IntType nbIw;
    IntType iwBlock;
    IntType iwTail;
    IntType nbOw;
    IntType owBlock;
    IntType owTail;
    IntType nbIs;
    IntType isBlock;
    IntType nbOs;
    IntType osBlock;
    IntType nbOcBlocking;
    IntType nbIcBlocking;
    IntType nbIsBlocking;
    IntType nbOsBlocking;
    Element::TypeT srcDt;
    Element::TypeT dstDt;
    Element::TypeT weiDt;
    Element::TypeT accDt;
    Element::TypeT biaDt;
    SizeType srcDsz;
    SizeType weiDsz;
    SizeType dstDsz;
    SizeType accDsz;
    SizeType biaDsz;
    bool useBuffer;
    IntType bufferSize;
    IntType kerRangesSize;
    IntType compABufferSize;
    IntType s8s8CompBufferSize;
    bool withScales;
    IntType isIcScale;
    IntType isOcScale;
    IntType ldA;
    IntType ldB;
    IntType ldC;
    IntType ldD;
    IntType m;
    IntType n;
    IntType k;
    IntType mTail;
    IntType nTail;
    IntType kTail;
    // m for brgemm kernel, for useStoreMask
    // it is usually greater than m (mTail), otherwise it is equal to m (mTail)
    IntType brgM;
    IntType brgMTail;
    IntType gemmBatchSize;
    IntType adjustedBatchSize;
    BrgemmBatchKindT brgType;
    // strides for brgType is STRIDE
    IntType brgStrideA;
    IntType brgStrideB;
    IntType nthr;
    IntType maxBatch;
    IntType maxVpad;
    bool weiPlain;
    bool isIcPadded;
    bool isOcPadded;
    IntType kwSets;
    IntType khSets;
    bool copyBlockOnly;
    IntType useMMask;
    IntType oskip;
    IntType iskip;
    bool brgemmBdLoopInnermost;
    bool useUker;
    bool varBs {false};
    bool useInterleaveStores;
    BrgemmKernelPrefetchingT hintPrefetching;
    bool is1x1;
    bool s8s8CompensationRequired;
    bool srcZeroPoint;
    bool dstZeroPoint;
    bool reqBrgCompPad;
    bool reqCalCompPad;
    bool isBF32;
    bool compWithVpads;
    IntType nthrMb;
    IntType nthrG;
    IntType nthrOcB;
    IntType nthrIcB;
    IntType nthrOh;
    bool transformToVnni;
    bool hasInt8Vnni;
    float scaleAdjustFactor;
    IntType icTail;
    IntType ocTail;
    SizeType trSrcBufSize;
    SizeType trSrcBufCount;
    SizeType trDiffDstBufSize;
    SizeType trDiffDstBufCount;
    IntType trSrcNumGuardElems;
    bool globalTranspose;
    IntType nthrMbWork;
    IntType trIw, trOw;
    IntType spatialBlkSize;
    IntType typeSizeIn;
    IntType typeSizeOut;
    bool trOcbChunk = false;
    bool trIcbChunk = false;
};

struct KdnnBrgconvDataT {
    bool useBrgconv = false;
    KdnnBrgconvImplT brgconvImpl = KdnnBrgconvImplT::UNDEFINED;
    KdnnBrgemmConvConfT bcp = KdnnBrgemmConvConfT();
    std::shared_ptr<BrgemmDescContainerT> brgs = nullptr;
    BrgemmKernelContainerT brgKernels {BRGEMM_PRIM_CONF_NUM_16};
};

} // namespace KuDNN

#endif // KUDNN_BRGEMM_PRIMITIVE_CONF_HPP
