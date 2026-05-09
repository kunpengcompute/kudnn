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

#ifndef KUDNN_PRIMITIVE_CONF_HPP
#define KUDNN_PRIMITIVE_CONF_HPP

#include <cstdint>
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_layout.hpp"
#include "types/kudnn_propagation_type.hpp"

namespace KuDNN {

enum class KudnnConvVersionT {
    KUDNN_VER_UNUSED,
    KUDNN_VER_FMA,
    KUDNN_VER_VectorComputation_512,
    KUDNN_VER_VectorComputation_256,
    KUDNN_VER_MatrixComputation_DIRECT,
    KUDNN_VER_MatrixComputation_GEMM
};

enum class KudnnConvLoopOrderT {
    KUDNN_LOOP_CGN,
    KUDNN_LOOP_GNC,
    KUDNN_LOOP_NGC,
    KUDNN_LOOP_GNCW,
    KUDNN_LOOP_CWGN,
    KUDNN_LOOP_NGCW,
    KUDNN_LOOP_NHWCG,
    KUDNN_LOOP_NWCG
};

enum class KudnnConvKernelKindT {
    KUDNN_EMBD_BCAST,
    KUDNN_EXPL_BCAST,
    KUDNN_CONT_BCAST,
};

enum class KudnnConvHarnessT {
    KUDNN_HARNESS_2D_REDUCTION,
    KUDNN_HARNESS_3D_REDUCTION,
    KUDNN_HARNESS_MB_REDUCTION,
    KUDNN_HARNESS_COMPUTE_FULL_SPATIAL,
    KUDNN_HARNESS_NXC
};

enum class KudnnConvFlag : unsigned {
    KUDNN_FLAG_MB_FIRST = 1 << 0,
    KUDNN_FLAG_MB_LAST = 1 << 1,
    KUDNN_FLAG_OC_FIRST = 1 << 2,
    KUDNN_FLAG_OC_LAST = 1 << 3,
    KUDNN_FLAG_IC_FIRST = 1 << 4,
    KUDNN_FLAG_IC_LAST = 1 << 5,
    KUDNN_FLAG_SP_FIRST = 1 << 6,
    KUDNN_FLAG_SP_LAST = 1 << 7,
    KUDNN_FLAG_REDUCE_FIRST = 1 << 8,
    KUDNN_FLAG_REDUCE_LAST = 1 << 9,
    KUDNN_FLAG_ZERO_FILTER = 1 << 0,  /* Controls whether the inner kernel skips
                                   loading weights-data from memory; this
                                   needs to happen on the first Group/16
                                   iteration. */
    KUDNN_FLAG_ZERO_BIAS = 1 << 1,    /* Controls whether the inner kernel skip
                               loading bias data from memory */
    KUDNN_FLAG_COMPUTE_BIAS = 1 << 2, /* Controls bias computation during execution
                                    pass */
};

struct KudnnConvJitConfT {
    Propagation propKind;
    KudnnConvVersionT ver;
    KudnnConvLoopOrderT loopOrder;
    KudnnConvHarnessT harness;

    IntType simdW;
    IntType ndims;
    IntType mb;
    IntType ngroups, ic, oc, ocWithoutPadding, icWithoutPadding;
    IntType id, ih, iw, od, oh, ow;
    IntType fPad, lPad, tPad;
    IntType backPad, rPad, bPad;
    IntType kd, kh, kw;
    IntType strideD, strideH, strideW;
    IntType dilateD, dilateH, dilateW;
    Layout srcTag, weiTag, dstTag; // temporary workaround
    bool withBias;
    bool withSum;
    bool withEltwise;
    bool withBinary;

    bool isFusedConv;
    IntType dwConvBufferOc;

    IntType nThr, nThrMb, nThrg, nThrOcb, nThrIcb;

    IntType idp, ihp, iwp, ohp, owp;
    IntType nbIc, icBlock;
    IntType nbOc, ocBlock;
    IntType nbIw, iwBlock;
    IntType nbOw, owBlock;
    IntType nbOcBlocking;                  /* used in jit kernels for nbOc work blocking taking
                           into account vector registers distribution */
    IntType nbOcBlockingThrChunk;          /* used for distribution of nbOc work
                                      within threads */
    IntType nbIcBlocking, nbIcBlockingMax; // blocking of nbIc work
    IntType nbIcL2;
    IntType hBlocking;
    IntType nbOcL2;
    IntType icTail, ocTail;
    IntType urH, urW;
    IntType urWTail;
    IntType urIc, urKw;
    bool is1stConv;
    IntType nonblkGroupOff;
    /* fma sve core */
    KudnnConvKernelKindT kernelKind;

    IntType trIw, trIh;
    IntType trKw, trKh;
    IntType trSrcNumGuardElems;

    // Transpose buffer management
    SizeType trSrcBufSize, trSrcBufCount;
    SizeType trDiffDstBufSize, trDiffDstBufCount;
    IntType nThrMbWork;

    /* 1st conv */
    IntType trLd;
    IntType khStep;
    /* sve_512_u8s8u8 */
    IntType typeSizeIn;
    IntType typeSizeOut;
    IntType typeSizeBia;
    IntType typeSizeAcc;
    IntType icNb1, icNb2;
    IntType ocNb1;
    IntType urOwMax, urOw, urOwTail;
    IntType urOwNsteps;
    Element::TypeT biaDt;
    /* bf16 data-type for output */
    Element::TypeT dstDt;
    Element::TypeT srcDt;
    /* bf16 weights update */
    Element::TypeT weiDt;
    Element::TypeT dSrcDt;
    Element::TypeT dWeiDt;
    bool explBcast;
    bool largeSpatial, largeWFilter;
    IntType isOcScale;
    IntType maxRegsUr; // maximum accumulation registers
    // dw conv
    IntType nbCh, chBlock, nbChBlocking;
    bool isDepthwise, isFastDepthwise, isResrcDepthwise;
    IntType alignedThreads;
    // large spatial
    IntType hBlkSize, ohBlkSize;
    // s8s8 convolution
    bool signedInput;
    bool needSaturation;
    float weiAdjScale;
    // zero-point compensation
    bool srcZeroPoint;
    bool dstZeroPoint;
    bool zpSrcIsCommon; // common, otherwise per-channel

    bool usesPermwTransposition;
    bool transposeSrc;
    bool transposeDst;
    IntType icBlockStep;

    bool isHwTransp; // spatial dim height-width transposed
    bool isWithGroups;
};

struct KudnnConvJitCallS {
    const void *src;  /* hack, non-const for backward_data */
    const void *dst;  /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *srcPrf;
    const void *dstPrf;
    const void *filtPrf;
    const void *biasPrf;
    const void *scales;
    const void *accS32;
    const void *compensation;
    const int32_t *zpCompensation;
    const int32_t *srcZeroPoint;
    const int32_t *dstZeroPoint;
    const void *tileCfg;
    const void *tileCfgTail;

    // ptr to table of void * elements that are pointers to
    // post_op binary src1 tensors
    const void *postOpsBinaryRhsArgVec;
    // logical (# of elems) offset to the processed output channel
    // (for broadcasting [1,OC,1,1])
    SizeType ocLOff;
    const void *dstOrig; // pointer to dst memory (no offset)

    SizeType ocLOffPrf;
    const void *dstOrigPrf;

    SizeType kdOffset;
    SizeType kdOffsetPrf;
    SizeType khOffset;
    SizeType khOffsetPrf;
    SizeType osIndexBegin;
    SizeType osIndexBeginPrf;
    SizeType osIndexEnd;
    SizeType osIndexEndPrf;
    SizeType kdPadding;
    SizeType kdPaddingPrf;
    SizeType khPadding;
    SizeType khPaddingPrf;
    SizeType iwb;
    SizeType iwbPrf;
    SizeType owb;
    SizeType owbPrf;
    SizeType kwPadding;
    SizeType channel;
    SizeType channelPrf;
    SizeType ocBlocks;
    SizeType urW;
    SizeType urStrW;
    SizeType chBlocks;
    SizeType chBlocksPrf;
    SizeType reduceWork;
    SizeType reduceWorkPrf;
    SizeType loadWork;
    SizeType loadWorkPrf;
    SizeType tOverflow;
    SizeType bOverflow;
    SizeType fOverflow;
    SizeType backOverflow;
    SizeType lastH;
    SizeType tail;
    SizeType currentIw;
    SizeType isOsb;
    IntType flags;
    IntType flagsPrf;
    IntType ocFlag;
};

struct JitGemmCallS {
    const void *matrixA;
    const void *matrixB;
    const void *matrixC;
    const void *bias;

    size_t matrixM;
    size_t matrixN;
    size_t matrixK;
};

} // namespace KuDNN

#endif // KUDNN_PRIMITIVE_CONF_HPP
