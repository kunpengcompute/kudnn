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

#ifndef RNN_KERNEL_OPERATORS_HPP
#define RNN_KERNEL_OPERATORS_HPP

#include "internal_funcs.hpp"
#include "kernels/vml/vml_sve_base.h"

namespace KuDNN {
namespace Detail {
namespace Rnn {

svfloat32_t RnnTanhFWDF32X(svfloat32_t x, svfloat32_t va, svbool_t pg);
svfloat32_t RnnLogisticFWDF32X(svfloat32_t x, svfloat32_t va, svbool_t pg);
svfloat32_t RnnReLUFWDF32X(svfloat32_t x, svfloat32_t va, svbool_t pg);
svfloat32_t RnnLinearF32X(svfloat32_t x, svfloat32_t va, svbool_t pg);
void VanillaRnnVCFWDF32(const int64_t len, const float *scratch, const float *bias, float *cellDstLay,
    float* cellDstIter, float *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha);
void VanillaRnnVCFWDBF16(const int64_t len, const float *scratch, const __bf16 *bias, __bf16 *cellDstLay,
    __bf16* cellDstIter, __bf16 *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha);
void VanillaRnnVCFWDBF16F32(const int64_t len, const float *scratch, const float *bias, __bf16 *cellDstLay,
    __bf16* cellDstIter, __bf16 *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha);
void RnnAuGruVCBWDF32Part1(const int64_t len, float *diffAugruAttenOff, float augruAttenOff,
    const float *cellWsGates, const float *cellSrcIter, const float *cellDiffDstLay,
    const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch);
void RnnAuGruVCBWDBF16Part1(const int64_t len, float *diffAugruAttenOff,
    __bf16 augruAttenOff, const __bf16 *cellWsGates, const __bf16 *cellSrcIter, const float *cellDiffDstLay,
    const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch);

// bwd
svfloat32_t RnnDTanh(svfloat32_t x, svfloat32_t va, svbool_t pg);
svfloat32_t RnnDLogistic(svfloat32_t x, svfloat32_t va, svbool_t pg);
svfloat32_t RnnDAlphaReLU(svfloat32_t x, svfloat32_t va, svbool_t pg);
void RnnVCBWDF32(const int64_t len, const float *cellDiffDstLay, const float *cellDiffDstIter,
    float *scratch, const float *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha);
void RnnVCBWDBF16(const int64_t len, const float *cellDiffDstLay, const float *cellDiffDstIter,
    __bf16 *scratch, const __bf16 *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha);

void RnnLbrgruVCBWDF32(const int64_t len, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch,
    float *wsWhb, float *scratchGatesR);
void RnnLbrgruVCBWDBF16(const int64_t len, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch,
    __bf16 *wsWhb, __bf16 *scratchGatesR);
void RnnLbrAugruVCBWDF32(const int64_t len, int64_t mb, float *diffAugruAttenOff,
    float augruAttenOff, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch,
    float *wsWhb, float *scratchGatesR);
void RnnLstmVCBWDF32(const int64_t len, const float *cellWsGates, const float *cellSrcIterC,
    const float *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, float *scratch, float *cellWeightPeephole, float *cellWeightProjection,
    const float *cellDiffDstIter);
void RnnLstmVCBWDBF16(const int64_t len, const __bf16 *cellWsGates, const  __bf16 *cellSrcIterC,
    const __bf16 *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, __bf16 *scratch, float *cellWeightPeephole, const float *cellDiffDstIter);
void RnnLstmVCBWDBF16F32(const int64_t len, const __bf16 *cellWsGates,
    const float *cellSrcIterC, const float *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, __bf16 *scratch, float *cellWeightPeephole, const float *cellDiffDstIter);
void RnnLbrAugruVCBWDBF16(const int64_t len, int64_t mb, float *diffAugruAttenOff,
    __bf16 augruAttenOff, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch,
    __bf16 *wsWhb, __bf16 *scratchGatesR);

// FWD
template <typename SrcType, typename BiasType>
void RnnLstmVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, const float *scratch,
    const BiasType *cellBias, const BiasType *cellSrcIterC, SrcType *cellDstLay, SizeType cellDstLayLd,
    SrcType *cellDstIter, BiasType *cellDstIterC, SrcType *cellWsGates);

template <typename SrcType, typename BiasType>
void RnnLstmPeepholeVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, const float *scratch,
    const BiasType *cellBias, const BiasType *cellSrcIterC, SrcType *cellDstLay, SizeType cellDstLayLd,
    SrcType *cellDstIter, BiasType *cellDstIterC, float *cellWeightPeephole, SrcType *cellWsGates);

template <typename SrcType, typename BiasType>
inline void VanillaRnnVCFWDWrapper(const int64_t len, const float *scratch, const BiasType *bias,
    SrcType *cellDstLay, SrcType *cellDstIter, SrcType *cellWsGates,
    svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t), const float alpha);

template <>
inline void VanillaRnnVCFWDWrapper<float, float>(const int64_t len, const float *scratch, const float *bias,
    float *cellDstLay, float *cellDstIter, float *cellWsGates,
    svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t), const float alpha)
{
    VanillaRnnVCFWDF32(len, scratch, bias, cellDstLay, cellDstIter, cellWsGates, actFunc, alpha);
}

template <>
inline void VanillaRnnVCFWDWrapper<__bf16, __bf16>(const int64_t len, const float *scratch, const __bf16 *bias,
    __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates,
    svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t), const float alpha)
{
    VanillaRnnVCFWDBF16(len, scratch, bias, cellDstLay, cellDstIter, cellWsGates, actFunc, alpha);
}

template <>
inline void VanillaRnnVCFWDWrapper<__bf16, float>(const int64_t len, const float *scratch, const float *bias,
    __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates,
    svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t), const float alpha)
{
    VanillaRnnVCFWDBF16F32(len, scratch, bias, cellDstLay, cellDstIter, cellWsGates, actFunc, alpha);
}

template <typename SrcType, typename BiasType>
void RnnGruVCFWDPart1Wrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, const BiasType *cellBias,
    const SrcType *cellSrcIter, SrcType *cellDstLay, SrcType *cellDstIter, SrcType *cellWsGates);

template <typename SrcType, typename BiasType>
void RnnGruVCFWDPart2Wrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, const BiasType *cellBias,
    const SrcType *cellSrcIter, SrcType *cellDstLay, SrcType *cellDstIter, SrcType *cellWsGates);

template <typename SrcType, typename BiasType>
void RnnAugruVCFWDPart2Wrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    const BiasType *cellBias, SrcType *augruAttenOff, const SrcType *cellSrcIter, SrcType *cellDstLay,
    SrcType *cellDstIter, SrcType *cellWsGates);

template <typename SrcType, typename BiasType>
void RnnLbrGruVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, float *scratchCell,
    const BiasType *cellBias, const SrcType *cellSrcIter, SrcType *cellDstLay, SrcType *cellDstIter,
    SrcType *cellWsGate, SrcType *cellWsGrid);

template <typename SrcType, typename BiasType>
void RnnLbrAugruVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, float *scratchCell,
    const BiasType *cellBias, SrcType *augruAttenOff, const SrcType *cellSrcIter, SrcType *cellDstLay,
    SrcType *cellDstIter, SrcType *cellWsGate, SrcType *cellWsGrid);

// BWD
template <typename SrcType>
inline void RnnVCBWDWrapper(const int64_t len, const float *cellDiffDstLay,
    const float *cellDiffDstIter, SrcType *scratch, const SrcType *cellWsGates,
    svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t), const float alpha);

template <>
inline void RnnVCBWDWrapper<float>(const int64_t len, const float *cellDiffDstLay, const float *cellDiffDstIter,
    float *scratch, const float *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha)
{
    RnnVCBWDF32(len, cellDiffDstLay, cellDiffDstIter, scratch, cellWsGates, actFunc, alpha);
}

template <>
inline void RnnVCBWDWrapper<__bf16>(const int64_t len, const float *cellDiffDstLay, const float *cellDiffDstIter,
    __bf16 *scratch, const __bf16 *cellWsGates,
    svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t), const float alpha)
{
    RnnVCBWDBF16(len, cellDiffDstLay, cellDiffDstIter, scratch, cellWsGates, actFunc, alpha);
}

void RnnGruVCBWDWrapperPart1(const int64_t len, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch);

void RnnGruVCBWDWrapperPart1(const int64_t len, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch);

void RnnGruVCBWDWrapperPart2(const int64_t len, float *dhG1, float *hG1, const float *cellWsGates,
    const float *cellSrcIter, float *cellDiffSrcIter, float *scratch);

void RnnGruVCBWDWrapperPart2(const int64_t len, float *dhG1, __bf16 *hG1, const __bf16 *cellWsGates,
    const __bf16 *cellSrcIter, float *cellDiffSrcIter, __bf16 *scratch);

template <typename SrcType>
inline void RnnAuGruVCBWDWrapperPart1(const int64_t len, float *diffAugruAttenOff,
    SrcType augruAttenOff, const SrcType *cellWsGates, const SrcType *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, SrcType *scratch);

template <>
inline void RnnAuGruVCBWDWrapperPart1<float>(const int64_t len, float *diffAugruAttenOff,
    float augruAttenOff, const float *cellWsGates, const float *cellSrcIter, const float *cellDiffDstLay,
    const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch)
{
    RnnAuGruVCBWDF32Part1(len, diffAugruAttenOff, augruAttenOff,
        cellWsGates, cellSrcIter, cellDiffDstLay, cellDiffDstIter, cellDiffSrcIter, scratch);
}

template <>
inline void RnnAuGruVCBWDWrapperPart1<__bf16>(const int64_t len, float *diffAugruAttenOff,
    __bf16 augruAttenOff, const __bf16 *cellWsGates, const __bf16 *cellSrcIter, const float *cellDiffDstLay,
    const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch)
{
    RnnAuGruVCBWDBF16Part1(len, diffAugruAttenOff, augruAttenOff,
        cellWsGates, cellSrcIter, cellDiffDstLay, cellDiffDstIter, cellDiffSrcIter, scratch);
}

template <typename SrcType>
inline void RnnLbrgruVCBWDWrapper(const int64_t len, const SrcType *cellWsGates, const SrcType *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, SrcType *scratch,
    SrcType *wsWhb, SrcType *scratchGatesR);

template <>
inline void RnnLbrgruVCBWDWrapper<float>(const int64_t len, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch,
    float *wsWhb, float *scratchGatesR)
{
    RnnLbrgruVCBWDF32(len, cellWsGates, cellSrcIter, cellDiffDstLay, cellDiffDstIter, cellDiffSrcIter, scratch,
        wsWhb, scratchGatesR);
}

template <>
inline void RnnLbrgruVCBWDWrapper<__bf16>(const int64_t len, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch,
    __bf16 *wsWhb, __bf16 *scratchGatesR)
{
    RnnLbrgruVCBWDBF16(len, cellWsGates, cellSrcIter, cellDiffDstLay, cellDiffDstIter, cellDiffSrcIter, scratch,
        wsWhb, scratchGatesR);
}

template <typename SrcType>
inline void RnnLbrAugruVCBWDWrapper(const int64_t len, int64_t mb, float *diffAugruAttenOff,
    SrcType augruAttenOff, const SrcType *cellWsGates, const SrcType *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, SrcType *scratch,
    SrcType *wsWhb, SrcType *scratchGatesR);

template <>
inline void RnnLbrAugruVCBWDWrapper<float>(const int64_t len, int64_t mb, float *diffAugruAttenOff,
    float augruAttenOff, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch,
    float *wsWhb, float *scratchGatesR)
{
    RnnLbrAugruVCBWDF32(len, mb, diffAugruAttenOff, augruAttenOff, cellWsGates, cellSrcIter, cellDiffDstLay,
        cellDiffDstIter, cellDiffSrcIter, scratch, wsWhb, scratchGatesR);
}

template <>
inline void RnnLbrAugruVCBWDWrapper<__bf16>(const int64_t len, int64_t mb, float *diffAugruAttenOff,
    __bf16 augruAttenOff, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch,
    __bf16 *wsWhb, __bf16 *scratchGatesR)
{
    RnnLbrAugruVCBWDBF16(len, mb, diffAugruAttenOff, augruAttenOff, cellWsGates, cellSrcIter, cellDiffDstLay,
        cellDiffDstIter, cellDiffSrcIter, scratch, wsWhb, scratchGatesR);
}

template <typename SrcType, typename BiasType>
inline void RnnLstmVCBWDWrapper(const int64_t len, const SrcType *cellWsGates, const BiasType *cellSrcIterC,
    const BiasType *cellDstIterC, const float *cellDiffDstLay,
    const float *cellDiffDstIterC, float *cellDiffSrcIterC, SrcType *scratch, float *cellWeightPeephole,
    float *cellWeightProjection, const float *cellDiffDstIter);

template <>
inline void RnnLstmVCBWDWrapper<float>(const int64_t len, const float *cellWsGates, const float *cellSrcIterC,
    const float *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, float *scratch, float *cellWeightPeephole,  float *cellWeightProjection,
    const float *cellDiffDstIter)
{
    RnnLstmVCBWDF32(len, cellWsGates, cellSrcIterC, cellDstIterC, cellDiffDstLay, cellDiffDstIterC,
        cellDiffSrcIterC, scratch, cellWeightPeephole, cellWeightProjection, cellDiffDstIter);
}

template <>
inline void RnnLstmVCBWDWrapper<__bf16>(const int64_t len, const __bf16 *cellWsGates, const __bf16 *cellSrcIterC,
    const __bf16 *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, __bf16 *scratch, float *cellWeightPeephole, float *cellWeightProjection,
    const float *cellDiffDstIter)
{
    Service::Unused(cellWeightProjection);
    RnnLstmVCBWDBF16(len, cellWsGates, cellSrcIterC, cellDstIterC, cellDiffDstLay, cellDiffDstIterC,
        cellDiffSrcIterC, scratch, cellWeightPeephole, cellDiffDstIter);
}

template <>
inline void RnnLstmVCBWDWrapper<__bf16, float>(const int64_t len, const __bf16 *cellWsGates,
    const float *cellSrcIterC, const float *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, __bf16 *scratch, float *cellWeightPeephole, float *cellWeightProjection,
    const float *cellDiffDstIter)
{
    Service::Unused(cellWeightProjection);
    RnnLstmVCBWDBF16F32(len, cellWsGates, cellSrcIterC, cellDstIterC, cellDiffDstLay, cellDiffDstIterC,
        cellDiffSrcIterC, scratch, cellWeightPeephole, cellDiffDstIter);
}

} // Rnn
} // Detail
} // KuDNN

#endif // RNN_KERNEL_OPERATORS_HPP
